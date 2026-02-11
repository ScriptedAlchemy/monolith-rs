use monolith_serving::resource_utils::{
    cal_available_memory_v2, cal_cpu_usage_v2, total_memory_v2,
};

// Mirrors `monolith/agent_service/resource_utils_test.py` (v2 functions only).
#[test]
fn test_cal_available_memory_v2() {
    let total = total_memory_v2();
    let available = cal_available_memory_v2();
    assert!(available > 0);
    assert!(available < total, "available={available} total={total}");
}

#[test]
fn test_cal_cpu_usage_v2() {
    let usage = cal_cpu_usage_v2();
    assert!((0.0..=100.0).contains(&usage), "usage={usage}");
}
