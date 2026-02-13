//! Python `monolith.native_training.prefetch_queue` parity (Rust-native).
//!
//! The upstream Python module builds TF queue resources to prefetch nested
//! tensors and to support async function execution via hooks. In Rust we
//! implement the observable behavior in a backend-agnostic way:
//! - FIFO queue with a fixed capacity.
//! - Optional CPU/GPU split via a lightweight `DevicePlacement` trait.
//! - Flattening and rebuilding nested structures while preserving non-tensor leaves.
//! - Async function manager that mirrors the "first enqueue primes the queue"
//!   behavior from the Python `AsyncPushHook` tests.
//!
//! This module intentionally does *not* depend on TensorFlow.

use crate::hooks::{Hook, HookResult};
use crate::metrics::Metrics;
use std::collections::{BTreeMap, VecDeque};
use std::fmt;
use std::sync::{Arc, Condvar, Mutex};

/// Device placement marker (CPU/GPU) used by `_MultiFIFOQueue` parity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Gpu,
}

/// Trait to provide device placement for queue components.
///
/// For parity with Python's `"GPU" in tensor.device` check, this is kept very small.
pub trait DevicePlacement {
    fn device(&self) -> Device;
}

/// A helper wrapper to explicitly tag a value with a device.
#[derive(Clone, Debug, PartialEq)]
pub struct Placed<T> {
    pub device: Device,
    pub value: T,
}

impl<T> Placed<T> {
    pub fn cpu(value: T) -> Self {
        Self {
            device: Device::Cpu,
            value,
        }
    }

    pub fn gpu(value: T) -> Self {
        Self {
            device: Device::Gpu,
            value,
        }
    }
}

impl<T> DevicePlacement for Placed<T> {
    fn device(&self) -> Device {
        self.device
    }
}

// =============================================================================
// Nested structure helpers (Rust equivalent of `nested_tensors.NestedTensors`)
// =============================================================================

/// A nested structure used by the queue helpers.
#[derive(Clone, PartialEq)]
pub enum Nested<T> {
    /// Tensor-like leaf.
    Tensor(T),
    /// Placeholder token (used by enqueue/dequeue plumbing).
    Token(usize),
    /// String leaf.
    Str(String),
    /// Null/None leaf.
    Null,
    /// Integer leaf.
    I64(i64),
    /// Float leaf.
    F64(f64),
    /// Boolean leaf.
    Bool(bool),
    /// Ordered collection.
    Seq(Vec<Nested<T>>),
    /// String-keyed map.
    Map(BTreeMap<String, Nested<T>>),
}

impl<T: fmt::Debug> fmt::Debug for Nested<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Nested::Tensor(t) => f.debug_tuple("Tensor").field(t).finish(),
            Nested::Token(i) => f.debug_tuple("Token").field(i).finish(),
            Nested::Str(s) => f.debug_tuple("Str").field(s).finish(),
            Nested::Null => write!(f, "Null"),
            Nested::I64(v) => f.debug_tuple("I64").field(v).finish(),
            Nested::F64(v) => f.debug_tuple("F64").field(v).finish(),
            Nested::Bool(v) => f.debug_tuple("Bool").field(v).finish(),
            Nested::Seq(v) => f.debug_tuple("Seq").field(v).finish(),
            Nested::Map(m) => f.debug_tuple("Map").field(m).finish(),
        }
    }
}

impl<T> Nested<T> {
    /// Collects all tensor leaves in traversal order.
    pub fn flatten_tensors(&self, out: &mut Vec<T>)
    where
        T: Clone,
    {
        match self {
            Nested::Tensor(t) => out.push(t.clone()),
            Nested::Seq(v) => v.iter().for_each(|x| x.flatten_tensors(out)),
            Nested::Map(m) => m.values().for_each(|x| x.flatten_tensors(out)),
            _ => {}
        }
    }

    /// Returns a copy where tensor leaves are replaced with `Token(i)` in traversal order,
    /// while non-tensor leaves are preserved.
    pub fn to_token_template(&self, next: &mut usize) -> Nested<T> {
        match self {
            Nested::Tensor(_) => {
                let idx = *next;
                *next += 1;
                Nested::Token(idx)
            }
            Nested::Seq(v) => Nested::Seq(v.iter().map(|x| x.to_token_template(next)).collect()),
            Nested::Map(m) => Nested::Map(
                m.iter()
                    .map(|(k, v)| (k.clone(), v.to_token_template(next)))
                    .collect(),
            ),
            Nested::Token(i) => Nested::Token(*i),
            Nested::Str(s) => Nested::Str(s.clone()),
            Nested::Null => Nested::Null,
            Nested::I64(v) => Nested::I64(*v),
            Nested::F64(v) => Nested::F64(*v),
            Nested::Bool(v) => Nested::Bool(*v),
        }
    }

    /// Replaces `Token(i)` leaves using `tensors[i]`.
    pub fn fill_from_tokens(&self, tensors: &[T]) -> Nested<T>
    where
        T: Clone,
    {
        match self {
            Nested::Token(i) => Nested::Tensor(tensors[*i].clone()),
            Nested::Seq(v) => Nested::Seq(v.iter().map(|x| x.fill_from_tokens(tensors)).collect()),
            Nested::Map(m) => Nested::Map(
                m.iter()
                    .map(|(k, v)| (k.clone(), v.fill_from_tokens(tensors)))
                    .collect(),
            ),
            Nested::Tensor(t) => Nested::Tensor(t.clone()),
            Nested::Str(s) => Nested::Str(s.clone()),
            Nested::Null => Nested::Null,
            Nested::I64(v) => Nested::I64(*v),
            Nested::F64(v) => Nested::F64(*v),
            Nested::Bool(v) => Nested::Bool(*v),
        }
    }
}

// =============================================================================
// FIFO queues (Rust-native replacement for TF queues)
// =============================================================================

#[derive(Debug)]
struct FifoInner<T> {
    buf: VecDeque<T>,
    capacity: usize,
    closed: bool,
}

/// A simple bounded FIFO queue.
#[derive(Debug)]
pub struct FifoQueue<T> {
    inner: Arc<(Mutex<FifoInner<T>>, Condvar, Condvar)>,
}

impl<T> Clone for FifoQueue<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<T> FifoQueue<T> {
    pub fn new(capacity: usize) -> Self {
        let inner = FifoInner {
            buf: VecDeque::new(),
            capacity: capacity.max(1),
            closed: false,
        };
        Self {
            inner: Arc::new((Mutex::new(inner), Condvar::new(), Condvar::new())),
        }
    }

    pub fn enqueue(&self, item: T) -> Result<(), String> {
        let (lock, not_empty, not_full) = &*self.inner;
        let mut g = lock.lock().map_err(|_| "mutex poisoned".to_string())?;
        while !g.closed && g.buf.len() >= g.capacity {
            g = not_full.wait(g).map_err(|_| "mutex poisoned".to_string())?;
        }
        if g.closed {
            return Err("queue closed".to_string());
        }
        g.buf.push_back(item);
        not_empty.notify_one();
        Ok(())
    }

    pub fn dequeue(&self) -> Result<T, String> {
        let (lock, not_empty, not_full) = &*self.inner;
        let mut g = lock.lock().map_err(|_| "mutex poisoned".to_string())?;
        while !g.closed && g.buf.is_empty() {
            g = not_empty
                .wait(g)
                .map_err(|_| "mutex poisoned".to_string())?;
        }
        if let Some(item) = g.buf.pop_front() {
            not_full.notify_one();
            Ok(item)
        } else {
            Err("queue closed".to_string())
        }
    }

    pub fn try_dequeue(&self) -> Option<T> {
        let (lock, _, not_full) = &*self.inner;
        let mut g = lock.lock().ok()?;
        let item = g.buf.pop_front();
        if item.is_some() {
            not_full.notify_one();
        }
        item
    }

    pub fn size(&self) -> usize {
        let (lock, _, _) = &*self.inner;
        let g = lock.lock().unwrap_or_else(|e| e.into_inner());
        g.buf.len()
    }

    pub fn close(&self) {
        let (lock, not_empty, not_full) = &*self.inner;
        let mut g = lock.lock().unwrap_or_else(|e| e.into_inner());
        g.closed = true;
        not_empty.notify_all();
        not_full.notify_all();
    }
}

/// A multi-device FIFO queue mirroring Python `_MultiFIFOQueue`.
///
/// This queue stores one element as a vector of components. Components can be
/// split by device and stored in separate queues, then merged back on dequeue.
#[derive(Debug, Clone)]
pub struct MultiFifoQueue<T> {
    cpu: FifoQueue<Vec<T>>,
    gpu: Option<FifoQueue<Vec<T>>>,
    split_indices: Vec<(Device, usize)>,
}

impl<T: DevicePlacement + Clone> MultiFifoQueue<T> {
    pub fn new(components: &[T], capacity: usize) -> Self {
        let mut split_indices = Vec::with_capacity(components.len());
        let mut cpu_count = 0usize;
        let mut gpu_count = 0usize;
        for c in components {
            match c.device() {
                Device::Cpu => {
                    split_indices.push((Device::Cpu, cpu_count));
                    cpu_count += 1;
                }
                Device::Gpu => {
                    split_indices.push((Device::Gpu, gpu_count));
                    gpu_count += 1;
                }
            }
        }

        Self {
            cpu: FifoQueue::new(capacity),
            gpu: if gpu_count > 0 {
                Some(FifoQueue::new(capacity))
            } else {
                None
            },
            split_indices,
        }
    }

    pub fn enqueue(&self, components: &[T]) -> Result<(), String> {
        let mut cpu = Vec::new();
        let mut gpu = Vec::new();
        for c in components {
            match c.device() {
                Device::Cpu => cpu.push(c.clone()),
                Device::Gpu => gpu.push(c.clone()),
            }
        }
        self.cpu.enqueue(cpu)?;
        if let Some(q) = &self.gpu {
            q.enqueue(gpu)?;
        }
        Ok(())
    }

    pub fn dequeue(&self) -> Result<Vec<T>, String> {
        let cpu_parts = self.cpu.dequeue()?;
        let gpu_parts = if let Some(q) = &self.gpu {
            Some(q.dequeue()?)
        } else {
            None
        };

        let mut out = Vec::with_capacity(self.split_indices.len());
        for (dev, idx) in &self.split_indices {
            match dev {
                Device::Cpu => {
                    let item = cpu_parts.get(*idx).ok_or_else(|| {
                        format!(
                            "cpu split index {idx} out of bounds for dequeued cpu parts (len={})",
                            cpu_parts.len()
                        )
                    })?;
                    out.push(item.clone());
                }
                Device::Gpu => {
                    let gpu = gpu_parts.as_ref().ok_or_else(|| {
                        "gpu split index requested but gpu queue parts are unavailable".to_string()
                    })?;
                    let item = gpu.get(*idx).ok_or_else(|| {
                        format!(
                            "gpu split index {idx} out of bounds for dequeued gpu parts (len={})",
                            gpu.len()
                        )
                    })?;
                    out.push(item.clone());
                }
            }
        }
        Ok(out)
    }

    pub fn size(&self) -> usize {
        // Like Python, use CPU queue size for multi-queue.
        self.cpu.size()
    }

    pub fn close(&self) {
        self.cpu.close();
        if let Some(q) = &self.gpu {
            q.close();
        }
    }
}

// =============================================================================
// Public API: enqueue_dicts_with_queue_return + async function manager
// =============================================================================

/// The return of `enqueue_dicts_with_queue_return` for non-zero capacity.
#[derive(Debug, Clone)]
pub struct EnqueueResult<T> {
    /// The nested structure with tensor leaves replaced by `Token(i)` placeholders.
    pub token_template: Nested<T>,
    /// The queue storing the flattened tensors as one element per enqueue.
    pub queue: MultiFifoQueue<T>,
}

/// Equivalent to Python `enqueue_dicts_with_queue_return`.
///
/// - `capacity == 0`: returns the original nested structure and no queue.
/// - else: returns a token-template and a queue. Call `queue.enqueue(flattened)` to push,
///   then `queue.dequeue()` and `token_template.fill_from_tokens(...)` to rebuild.
pub fn enqueue_dicts_with_queue_return<T: DevicePlacement + Clone>(
    tensors: Nested<T>,
    capacity: usize,
) -> (Nested<T>, Option<EnqueueResult<T>>) {
    if capacity == 0 {
        return (tensors, None);
    }

    let mut flat = Vec::new();
    tensors.flatten_tensors(&mut flat);

    let mut next = 0usize;
    let token_template = tensors.to_token_template(&mut next);

    let queue = MultiFifoQueue::new(&flat, capacity);
    (
        token_template.clone(),
        Some(EnqueueResult {
            token_template,
            queue,
        }),
    )
}

/// Hook that drains a queue by executing `run_one` until it is empty.
#[derive(Clone)]
pub struct AsyncPushHook {
    queue: FifoQueue<Box<dyn FnOnce() + Send>>,
    // Mirrors Python's `_queue_init` behavior: the first enqueue primes the hook.
    queue_init: Arc<Mutex<bool>>,
}

impl fmt::Debug for AsyncPushHook {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AsyncPushHook")
            .field("queue_size", &self.queue.size())
            .finish()
    }
}

impl AsyncPushHook {
    fn new(queue: FifoQueue<Box<dyn FnOnce() + Send>>, queue_init: Arc<Mutex<bool>>) -> Self {
        Self { queue, queue_init }
    }

    fn maybe_run_one(&self) {
        if let Some(f) = self.queue.try_dequeue() {
            f();
        }
    }
}

impl Hook for AsyncPushHook {
    fn name(&self) -> &str {
        "async_push_hook"
    }

    fn before_step(&mut self, _step: u64) -> HookResult<()> {
        // In TF, this happens as part of `before_run` once queue is initialized.
        let init = *self.queue_init.lock().unwrap_or_else(|e| e.into_inner());
        if init {
            self.maybe_run_one();
        }
        Ok(())
    }

    fn end(&mut self, _step: u64, _metrics: Option<&Metrics>) -> HookResult<()> {
        // Drain remaining queued work.
        while let Some(f) = self.queue.try_dequeue() {
            f();
        }
        Ok(())
    }
}

/// Async function manager mirroring Python `AsyncFunctionMgr`.
#[derive(Debug, Clone)]
pub struct AsyncFunctionMgr {
    is_async: bool,
    hooks: Vec<AsyncPushHook>,
}

impl AsyncFunctionMgr {
    pub fn new(is_async: bool) -> Self {
        Self {
            is_async,
            hooks: Vec::new(),
        }
    }

    pub fn default() -> Self {
        Self::new(true)
    }

    /// Add a function to be executed asynchronously.
    ///
    /// Returns an "enqueue op" closure. Calling it enqueues work. The first call only
    /// primes the queue (mirrors Python's `AsyncPushHook` init); subsequent calls
    /// enqueue and also execute one item (in the hook).
    pub fn add_async_function<F>(
        &mut self,
        target: F,
        is_async: Option<bool>,
    ) -> Box<dyn Fn() + Send>
    where
        F: Fn() + Send + Sync + 'static,
    {
        let is_async = is_async.unwrap_or(self.is_async);
        if !is_async {
            return Box::new(move || target());
        }

        let work_q: FifoQueue<Box<dyn FnOnce() + Send>> = FifoQueue::new(1024);
        let queue_init = Arc::new(Mutex::new(false));
        self.hooks
            .push(AsyncPushHook::new(work_q.clone(), Arc::clone(&queue_init)));

        let target = Arc::new(target);
        let enqueue = move || {
            // Enqueue work.
            let t = Arc::clone(&target);
            let _ = work_q.enqueue(Box::new(move || (t)()));
            // First enqueue primes the queue (no execution yet), like TF hook's `_queue_init`.
            let mut init = queue_init.lock().unwrap_or_else(|e| e.into_inner());
            if !*init {
                *init = true;
            }
        };
        Box::new(enqueue)
    }

    pub fn hooks(&self) -> Vec<Box<dyn Hook>> {
        self.hooks
            .clone()
            .into_iter()
            .map(|h| Box::new(h) as Box<dyn Hook>)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicI64, Ordering};

    #[test]
    fn fifo_queue_capacity_roundtrip() {
        let q = FifoQueue::new(4);
        for _ in 0..4 {
            q.enqueue(vec![Placed::cpu(2i64)])
                .expect("enqueue should succeed within queue capacity");
        }
        for _ in 0..4 {
            let v = q
                .dequeue()
                .expect("dequeue should succeed for previously enqueued items");
            assert_eq!(v[0].value, 2);
        }
    }

    #[test]
    fn enqueue_dicts_with_queue_return_zero_capacity_passthrough() {
        let mut m = BTreeMap::new();
        m.insert("dense".to_string(), Nested::Tensor(Placed::cpu(0i64)));
        let input = Nested::Map(m);

        let (out, q) = enqueue_dicts_with_queue_return(input.clone(), 0);
        assert!(q.is_none());
        assert_eq!(out, input);
    }

    #[test]
    fn enqueue_dicts_with_queue_return_preserves_non_tensor_leaves() {
        // Mirrors the effective Python test `PrefetchTest.test_enqueue_dicts_with_queue_return`.
        let mut list0 = BTreeMap::new();
        list0.insert("a".to_string(), Nested::Tensor(Placed::cpu(1i64)));
        list0.insert("b".to_string(), Nested::Str("abc".to_string()));
        list0.insert("c".to_string(), Nested::Null);
        list0.insert("d".to_string(), Nested::Null);

        let mut top = BTreeMap::new();
        top.insert("list".to_string(), Nested::Seq(vec![Nested::Map(list0)]));
        top.insert("v".to_string(), Nested::Tensor(Placed::cpu(5i64)));

        let (token_template, res) = enqueue_dicts_with_queue_return(Nested::Map(top.clone()), 2);
        let res = res.expect("queue exists");

        // Non-tensor values should be present in the token template immediately.
        let root = if let Nested::Map(root) = &token_template {
            root
        } else {
            assert!(
                false,
                "expected map token template root, got {token_template:?}"
            );
            return;
        };
        let list_node = root.get("list").expect("list key should exist");
        let items = if let Nested::Seq(items) = list_node {
            items
        } else {
            assert!(false, "expected list at key 'list', got {list_node:?}");
            return;
        };
        let first = &items[0];
        let item0 = if let Nested::Map(item0) = first {
            item0
        } else {
            assert!(false, "expected map element inside list, got {first:?}");
            return;
        };
        assert_eq!(
            item0.get("b").expect("key 'b' should exist in template map"),
            &Nested::<Placed<i64>>::Str("abc".to_string())
        );
        assert_eq!(
            item0.get("c").expect("key 'c' should exist in template map"),
            &Nested::<Placed<i64>>::Null
        );

        // Enqueue one element.
        let mut flat = Vec::new();
        Nested::Map(top).flatten_tensors(&mut flat);
        res.queue
            .enqueue(&flat)
            .expect("queue enqueue should succeed for flattened payload");

        // Dequeue and rebuild.
        let flat_out = res
            .queue
            .dequeue()
            .expect("queue dequeue should succeed for previously enqueued payload");
        let rebuilt = res.token_template.fill_from_tokens(&flat_out);

        let root = if let Nested::Map(root) = rebuilt {
            root
        } else {
            assert!(false, "expected rebuilt map root, got {rebuilt:?}");
            return;
        };
        let v_node = root.get("v").expect("v key should exist");
        let v = if let Nested::Tensor(v) = v_node {
            v
        } else {
            assert!(false, "expected tensor at key 'v', got {v_node:?}");
            return;
        };
        assert_eq!(v.value, 5);
    }

    #[test]
    fn enqueue_dicts_with_control_flow_side_effects_on_enqueue() {
        // Rust equivalent of Python control_dependencies test: side effects happen when enqueue runs.
        let v = Arc::new(AtomicI64::new(0));
        let q = FifoQueue::new(8);

        let v2 = Arc::clone(&v);
        let q2 = q.clone();
        let enqueue = move || {
            v2.fetch_add(1, Ordering::SeqCst);
            q2.enqueue(Placed::cpu(0i64))
                .expect("enqueue from side-effect closure should succeed");
        };

        enqueue();
        assert_eq!(v.load(Ordering::SeqCst), 1);
        let _ = q
            .dequeue()
            .expect("dequeue should succeed after side-effect enqueue");
    }

    #[test]
    fn multi_fifo_queue_dequeue_missing_gpu_parts_returns_error() {
        let q = MultiFifoQueue {
            cpu: FifoQueue::new(1),
            gpu: None,
            split_indices: vec![(Device::Gpu, 0)],
        };
        q.cpu
            .enqueue(Vec::<Placed<i64>>::new())
            .expect("cpu queue enqueue should succeed for malformed queue regression setup");

        let err = q
            .dequeue()
            .expect_err("dequeue should return an explicit error when gpu parts are unavailable");
        assert!(
            err.contains("gpu split index requested but gpu queue parts are unavailable"),
            "expected missing-gpu-parts diagnostics, got {err:?}"
        );
    }

    #[test]
    fn multi_fifo_queue_dequeue_out_of_bounds_cpu_split_index_returns_error() {
        let mut q = MultiFifoQueue::new(&[Placed::cpu(1i64)], 1);
        q.split_indices = vec![(Device::Cpu, 1)];
        q.enqueue(&[Placed::cpu(3i64)])
            .expect("enqueue should succeed for cpu split-index bounds regression setup");

        let err = q.dequeue().expect_err(
            "dequeue should return an explicit error when cpu split index exceeds dequeued parts",
        );
        assert!(
            err.contains("cpu split index 1 out of bounds"),
            "expected cpu split-index bounds diagnostics, got {err:?}"
        );
    }

    #[test]
    fn async_function_mgr_basic_primes_then_runs() {
        let x = Arc::new(AtomicI64::new(0));
        let x2 = Arc::clone(&x);

        let mut mgr = AsyncFunctionMgr::default();
        let enqueue = mgr.add_async_function(
            move || {
                x2.fetch_add(1, Ordering::SeqCst);
            },
            None,
        );

        // First enqueue primes the hook; no run yet.
        enqueue();
        assert_eq!(x.load(Ordering::SeqCst), 0);

        // Simulate a training step calling hooks; second enqueue then one hook run.
        enqueue();
        let mut hooks = mgr.hooks();
        for h in hooks.iter_mut() {
            h.before_step(0)
                .expect("hook before_step should succeed for async manager");
        }
        assert_eq!(x.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn async_function_mgr_sync_executes_immediately() {
        let x = Arc::new(AtomicI64::new(0));
        let x2 = Arc::clone(&x);

        let mut mgr = AsyncFunctionMgr::new(false);
        let op = mgr.add_async_function(
            move || {
                x2.fetch_add(1, Ordering::SeqCst);
            },
            None,
        );
        op();
        assert_eq!(x.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn async_function_mgr_empty_input_ok() {
        let x = Arc::new(AtomicI64::new(0));
        let x2 = Arc::clone(&x);

        let mut mgr = AsyncFunctionMgr::default();
        let enqueue = mgr.add_async_function(
            move || {
                x2.fetch_add(1, Ordering::SeqCst);
            },
            None,
        );
        enqueue();
        enqueue();
        let mut hooks = mgr.hooks();
        for h in hooks.iter_mut() {
            h.before_step(0)
                .expect("hook before_step should succeed for async manager");
        }
        assert_eq!(x.load(Ordering::SeqCst), 1);
    }
}
