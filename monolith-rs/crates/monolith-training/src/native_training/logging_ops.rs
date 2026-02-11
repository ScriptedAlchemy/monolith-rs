//! TF-free parity for `monolith/native_training/logging_ops.py`.
//!
//! The Python version is a thin wrapper around custom TF ops. In Rust we keep
//! the same high-level behavior with TF runtime optional by:
//! - Providing a monotonic timestamp helper (`tensors_timestamp`)
//! - Emitting timers as a no-op (hook points can integrate later)
//! - Implementing `machine_info` + `check_machine_health` using the protobuf
//!   type and the "empty bytes means OK" convention.

use monolith_proto::tensorflow::monolith_tf::machine_health_result::MachineHealthStatus;
use monolith_proto::tensorflow::monolith_tf::MachineHealthResult;
use prost::Message;
use std::sync::OnceLock;
use std::time::Instant;

/// Default value for `mem_limit` in machine info (bytes).
pub const DEFAULT_MACHINE_INFO_MEM_LIMIT: i64 = 1i64 << 62;

/// Placeholder struct representing the TF resource `MachineInfo`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MachineInfo {
    pub mem_limit: i64,
    pub shared_name: Option<String>,
}

static START: OnceLock<Instant> = OnceLock::new();

fn monotonic_ns() -> u64 {
    let start = START.get_or_init(Instant::now);
    start.elapsed().as_nanos() as u64
}

/// Gets the timestamp (monotonic nanoseconds) when the tensors are ready.
///
/// Rust doesn't have a TF runtime here, so `tensors` are returned unchanged.
pub fn tensors_timestamp<T>(tensors: Vec<T>) -> (Vec<T>, u64) {
    (tensors, monotonic_ns())
}

/// Emits a timer metric.
///
/// In Python this returns a TF op. In Rust it's currently a no-op hook point.
pub fn emit_timer(_key: &str, _value: f64, _tags: Option<&[(&str, &str)]>) {}

/// Returns a `MachineInfo` instance (TF resource in Python).
pub fn machine_info(mem_limit: Option<i64>, shared_name: Option<&str>) -> MachineInfo {
    MachineInfo {
        mem_limit: mem_limit.unwrap_or(DEFAULT_MACHINE_INFO_MEM_LIMIT),
        shared_name: shared_name.map(|s| s.to_string()),
    }
}

/// Returns serialized `MachineHealthResult`.
///
/// Parity detail: Python guarantees this is an empty string (`b""`) when OK.
pub fn check_machine_health(machine_info: &MachineInfo) -> Vec<u8> {
    if machine_info.mem_limit <= 0 {
        let msg = MachineHealthResult {
            status: Some(MachineHealthStatus::OutOfMemory as i32),
            message: Some("out of memory".to_string()),
        };
        msg.encode_to_vec()
    } else {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensors_timestamp_is_monotonic() {
        let (t, ts) = tensors_timestamp(vec![0u8]);
        let (_t2, ts2) = tensors_timestamp(t);
        assert!(ts2 >= ts);
    }

    #[test]
    fn machine_health_ok_is_empty_bytes() {
        let info = machine_info(Some(DEFAULT_MACHINE_INFO_MEM_LIMIT), None);
        let bytes = check_machine_health(&info);
        assert!(bytes.is_empty());
    }

    #[test]
    fn machine_health_oom_sets_status() {
        let info = machine_info(Some(0), None);
        let bytes = check_machine_health(&info);
        let decoded = MachineHealthResult::decode(bytes.as_slice()).unwrap();
        assert_eq!(
            decoded.status,
            Some(MachineHealthStatus::OutOfMemory as i32)
        );
    }
}
