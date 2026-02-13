//! Tiny parity subset for Python `monolith.native_training.device_utils`.
//!
//! The full Python module is TensorFlow-specific. For the Rust port we only
//! expose the pieces that can be meaningfully tested/used without TF:
//! - global allow/deny toggle for "gpu placement"
//! - `get_visible_gpus(local_rank, processes_per_gpu)`

use std::sync::atomic::{AtomicBool, Ordering};

static GPU_PLACEMENT_ALLOWED: AtomicBool = AtomicBool::new(false);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceUtilsError {
    InvalidProcessesPerGpu { processes_per_gpu: i32 },
}

impl std::fmt::Display for DeviceUtilsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidProcessesPerGpu { processes_per_gpu } => write!(
                f,
                "processes_per_gpu must be >= 1, got {processes_per_gpu}"
            ),
        }
    }
}

impl std::error::Error for DeviceUtilsError {}

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
///
/// # Examples
///
/// ```
/// use monolith_training::native_training::device_utils::{
///     get_visible_gpus, DeviceUtilsError,
/// };
///
/// assert_eq!(get_visible_gpus(3, 2), Ok("1".to_string()));
/// assert!(matches!(
///     get_visible_gpus(3, 0),
///     Err(DeviceUtilsError::InvalidProcessesPerGpu {
///         processes_per_gpu: 0
///     })
/// ));
/// ```
pub fn get_visible_gpus(local_rank: i32, processes_per_gpu: i32) -> Result<String, DeviceUtilsError> {
    if processes_per_gpu < 1 {
        return Err(DeviceUtilsError::InvalidProcessesPerGpu { processes_per_gpu });
    }
    Ok((local_rank / processes_per_gpu).to_string())
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
        assert_eq!(
            get_visible_gpus(2, 1).expect("processes_per_gpu=1 should succeed"),
            "2"
        );
        assert_eq!(
            get_visible_gpus(1, 2).expect("processes_per_gpu=2 should succeed"),
            "0"
        );
        assert_eq!(
            get_visible_gpus(2, 2).expect("processes_per_gpu=2 should succeed"),
            "1"
        );
        assert_eq!(
            get_visible_gpus(3, 2).expect("processes_per_gpu=2 should succeed"),
            "1"
        );
    }

    #[test]
    fn test_get_visible_gpus_rejects_non_positive_processes_per_gpu() {
        let err = get_visible_gpus(3, 0)
            .expect_err("non-positive processes_per_gpu should return explicit error");
        assert!(
            matches!(
                err,
                DeviceUtilsError::InvalidProcessesPerGpu {
                    processes_per_gpu: 0
                }
            ),
            "expected InvalidProcessesPerGpu for zero, got {err:?}"
        );
    }
}
