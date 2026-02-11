//! Resource utilities (Python parity for `monolith/agent_service/resource_utils.py`).
//!
//! The Python implementation provides a mix of cgroup-based and `psutil`-based helpers. The
//! parity tests only cover the v2 versions (psutil-backed). For Rust, we implement:
//! - `total_memory_v2`: total system memory
//! - `cal_available_memory_v2`: available system memory
//! - `cal_cpu_usage_v2`: current CPU usage percent
//!
//! The other helpers are not required by the referenced Python tests.

use std::time::Duration;

/// Total system memory in bytes.
pub fn total_memory_v2() -> u64 {
    // `sysinfo` returns KiB.
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    sys.total_memory().saturating_mul(1024)
}

/// Available system memory in bytes.
pub fn cal_available_memory_v2() -> u64 {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    // Some platforms report 0 for "available". Fall back to free in that case.
    let avail = sys.available_memory();
    let kib = if avail > 0 { avail } else { sys.free_memory() };
    kib.saturating_mul(1024)
}

/// CPU usage percent in the range [0, 100].
pub fn cal_cpu_usage_v2() -> f32 {
    // sysinfo requires two refresh cycles to compute CPU usage.
    let mut sys = sysinfo::System::new();
    sys.refresh_cpu();
    std::thread::sleep(Duration::from_millis(100));
    sys.refresh_cpu();
    let cpus = sys.cpus();
    if cpus.is_empty() {
        return 0.0;
    }
    let sum: f32 = cpus.iter().map(|c| c.cpu_usage()).sum();
    let avg = sum / (cpus.len() as f32);
    // Clamp to match Python test expectations.
    avg.clamp(0.0, 100.0)
}
