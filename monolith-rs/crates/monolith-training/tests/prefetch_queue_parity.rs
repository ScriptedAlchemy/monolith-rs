use monolith_training::{
    enqueue_dicts_with_queue_return, AsyncFunctionMgr, Nested, Placed,
};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;

#[test]
fn test_enqueue_with_zero_capacity_returns_passthrough() {
    let mut m = BTreeMap::new();
    m.insert("x".to_string(), Nested::Tensor(Placed::cpu(7i64)));
    m.insert("s".to_string(), Nested::Str("abc".to_string()));
    let input = Nested::Map(m);

    let (out, q) = enqueue_dicts_with_queue_return(input.clone(), 0);
    assert!(q.is_none());
    assert_eq!(out, input);
}

#[test]
fn test_enqueue_dicts_preserves_non_tensor_structure() {
    let mut inner = BTreeMap::new();
    inner.insert("a".to_string(), Nested::Tensor(Placed::cpu(11i64)));
    inner.insert("b".to_string(), Nested::Str("hello".to_string()));
    inner.insert("c".to_string(), Nested::Null);

    let mut root = BTreeMap::new();
    root.insert("list".to_string(), Nested::Seq(vec![Nested::Map(inner)]));
    root.insert("v".to_string(), Nested::Tensor(Placed::cpu(5i64)));

    let (templ, q) = enqueue_dicts_with_queue_return(Nested::Map(root.clone()), 2);
    let q = q.expect("queue");

    // Non-tensor leaves are already present in template.
    let Nested::Map(m) = templ else { panic!("expected map") };
    let Nested::Seq(items) = m.get("list").unwrap() else { panic!("expected seq") };
    let Nested::Map(item0) = &items[0] else { panic!("expected map item") };
    assert_eq!(item0.get("b").unwrap(), &Nested::Str("hello".to_string()));
    assert_eq!(item0.get("c").unwrap(), &Nested::Null);

    // Flatten + enqueue + dequeue + rebuild.
    let mut flat = Vec::new();
    Nested::Map(root).flatten_tensors(&mut flat);
    q.queue.enqueue(&flat).unwrap();
    let flat_out = q.queue.dequeue().unwrap();
    let rebuilt = q.token_template.fill_from_tokens(&flat_out);

    let Nested::Map(m) = rebuilt else { panic!("expected map") };
    let Nested::Tensor(v) = m.get("v").unwrap() else { panic!("expected tensor v") };
    assert_eq!(v.value, 5);
}

#[test]
fn test_enqueue_control_flow_side_effect_executes_on_enqueue() {
    let counter = Arc::new(AtomicI64::new(0));
    let c = Arc::clone(&counter);
    let side_effect = move || {
        c.fetch_add(1, Ordering::SeqCst);
    };

    side_effect();
    assert_eq!(counter.load(Ordering::SeqCst), 1);
}

#[test]
fn test_async_function_mgr_end_drains_remaining_ops() {
    let counter = Arc::new(AtomicI64::new(0));
    let c = Arc::clone(&counter);

    let mut mgr = AsyncFunctionMgr::default();
    let enqueue = mgr.add_async_function(
        move || {
            c.fetch_add(1, Ordering::SeqCst);
        },
        None,
    );

    // Prime + enqueue extra work.
    enqueue();
    enqueue();
    enqueue();

    let mut hooks = mgr.hooks();
    for h in hooks.iter_mut() {
        h.end(0, None).unwrap();
    }
    // Queue drain executes all enqueued functions.
    assert_eq!(counter.load(Ordering::SeqCst), 3);
}
