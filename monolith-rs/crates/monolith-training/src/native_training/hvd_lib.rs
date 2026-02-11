//! Python `monolith.native_training.hvd_lib` parity.
//!
//! Python delays importing either `byteps.tensorflow` or `horovod.tensorflow`
//! until first use, based on `MONOLITH_WITH_BYTEPS`.
//!
//! In Rust we don't load Python libs; instead we provide the same "choice"
//! signal and small stubs to make call sites testable.

/// Whether BytePS is enabled (`MONOLITH_WITH_BYTEPS` != 0).
pub fn enable_bps() -> bool {
    std::env::var("MONOLITH_WITH_BYTEPS")
        .ok()
        .and_then(|s| s.trim().parse::<i32>().ok())
        .unwrap_or(0)
        != 0
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Byteps,
    Horovod,
}

pub fn backend() -> Backend {
    if enable_bps() {
        Backend::Byteps
    } else {
        Backend::Horovod
    }
}

/// Stub: `init()` in Python delegates to the selected library.
pub fn init() {}

/// Stub: always returns 0 for single-process Rust tests.
pub fn rank() -> i32 {
    0
}

/// Stub: always returns 1 for single-process Rust tests.
pub fn size() -> i32 {
    1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_default_horovod() {
        std::env::remove_var("MONOLITH_WITH_BYTEPS");
        assert_eq!(backend(), Backend::Horovod);
    }

    #[test]
    fn test_backend_byteps() {
        std::env::set_var("MONOLITH_WITH_BYTEPS", "1");
        assert_eq!(backend(), Backend::Byteps);
    }
}
