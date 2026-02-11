//! Python `monolith.native_training.native_task_context` parity.
//!
//! Python stores a process-global "current" context used by many helper
//! functions. For Rust we use a thread-local context with an RAII guard.

use std::cell::RefCell;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NativeTaskContext {
    pub num_ps: i32,
    pub ps_index: i32,
    pub num_workers: i32,
    pub worker_index: i32,
    pub model_name: String,
    pub server_type: String,
}

thread_local! {
    static CTX: RefCell<Option<NativeTaskContext>> = const { RefCell::new(None) };
}

pub struct NativeTaskContextGuard {
    prev: Option<NativeTaskContext>,
}

impl Drop for NativeTaskContextGuard {
    fn drop(&mut self) {
        let prev = self.prev.take();
        CTX.with(|c| {
            *c.borrow_mut() = prev;
        });
    }
}

/// Set the current context for the duration of the returned guard.
pub fn with_ctx(ctx: NativeTaskContext) -> NativeTaskContextGuard {
    let prev = CTX.with(|c| c.borrow().clone());
    CTX.with(|c| {
        *c.borrow_mut() = Some(ctx);
    });
    NativeTaskContextGuard { prev }
}

/// Get the current context, or a default empty context if unset.
pub fn get() -> NativeTaskContext {
    CTX.with(|c| c.borrow().clone()).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_with_ctx_roundtrip() {
        assert_eq!(get(), NativeTaskContext::default());
        let ctx = NativeTaskContext {
            num_ps: 2,
            ps_index: 1,
            num_workers: 3,
            worker_index: 0,
            model_name: "m".to_string(),
            server_type: "worker".to_string(),
        };
        {
            let _g = with_ctx(ctx.clone());
            assert_eq!(get(), ctx);
        }
        assert_eq!(get(), NativeTaskContext::default());
    }
}
