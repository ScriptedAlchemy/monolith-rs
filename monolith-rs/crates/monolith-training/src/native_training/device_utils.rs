//! Tiny parity subset for Python `monolith.native_training.device_utils`.
//!
//! The full Python module is TensorFlow-specific. For the Rust port we only
//! expose the pieces that can be meaningfully tested/used without TF:
//! - global allow/deny toggle for "gpu placement"
//! - `get_visible_gpus(local_rank, processes_per_gpu)`

use std::sync::atomic::{AtomicBool, Ordering};

static GPU_PLACEMENT_ALLOWED: AtomicBool = AtomicBool::new(false);

pub fn enable_gpu_training() {
    GPU_PLACEMENT_ALLOWED.store(true, Ordering::SeqCst);
}

pub fn disable_gpu_training() {
    GPU_PLACEMENT_ALLOWED.store(false, Ordering::SeqCst);
}

pub fn is_gpu_training() -> bool {
    GPU_PLACEMENT_ALLOWED.load(Ordering::SeqCst)
}

/// Visible GPU devices string for a process.
///
/// Mirrors Python:
/// `str(int(local_rank / processes_per_gpu))`
pub fn get_visible_gpus(local_rank: i32, processes_per_gpu: i32) -> String {
    assert!(processes_per_gpu >= 1, "processes_per_gpu must be >= 1");
    (local_rank / processes_per_gpu).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_toggle() {
        disable_gpu_training();
        assert!(!is_gpu_training());
        enable_gpu_training();
        assert!(is_gpu_training());
    }

    #[test]
    fn test_get_visible_gpus_matches_python() {
        assert_eq!(get_visible_gpus(2, 1), "2");
        assert_eq!(get_visible_gpus(1, 2), "0");
        assert_eq!(get_visible_gpus(2, 2), "1");
        assert_eq!(get_visible_gpus(3, 2), "1");
    }
}
