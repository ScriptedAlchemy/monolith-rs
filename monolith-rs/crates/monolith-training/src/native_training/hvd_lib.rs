//! Python `monolith.native_training.hvd_lib` parity.
//!
//! Python delays importing either `byteps.tensorflow` or `horovod.tensorflow`
//! until first use, based on `MONOLITH_WITH_BYTEPS`.
//!
//! In Rust we don't load Python libs; instead we provide the same "choice"
//! signal and small stubs to make call sites testable.

fn parse_bool_env(var: &str) -> Option<bool> {
    let raw = std::env::var(var).ok()?;
    let s = raw.trim().to_ascii_lowercase();
    match s.as_str() {
        "1" | "true" | "yes" | "y" | "on" => Some(true),
        "0" | "false" | "no" | "n" | "off" => Some(false),
        _ => None,
    }
}

fn parse_i32_env(vars: &[&str]) -> Option<i32> {
    vars.iter()
        .find_map(|name| std::env::var(name).ok())
        .and_then(|v| v.trim().parse::<i32>().ok())
}

/// Whether BytePS is enabled (`MONOLITH_WITH_BYTEPS` != 0).
pub fn enable_bps() -> bool {
    parse_bool_env("MONOLITH_WITH_BYTEPS")
        .or_else(|| parse_i32_env(&["MONOLITH_WITH_BYTEPS"]).map(|v| v != 0))
        .unwrap_or(false)
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

/// Returns global rank from BytePS/Horovod-style environment variables.
pub fn rank() -> i32 {
    let vars = match backend() {
        Backend::Byteps => &["BYTEPS_RANK", "RANK"][..],
        Backend::Horovod => &["HOROVOD_RANK", "OMPI_COMM_WORLD_RANK", "RANK"][..],
    };
    parse_i32_env(vars).unwrap_or(0).max(0)
}

/// Returns global world size from BytePS/Horovod-style environment variables.
pub fn size() -> i32 {
    let vars = match backend() {
        Backend::Byteps => &["BYTEPS_SIZE", "WORLD_SIZE"][..],
        Backend::Horovod => &["HOROVOD_SIZE", "OMPI_COMM_WORLD_SIZE", "WORLD_SIZE"][..],
    };
    parse_i32_env(vars).unwrap_or(1).max(1)
}

/// Returns local rank from environment variables.
pub fn local_rank() -> i32 {
    let vars = match backend() {
        Backend::Byteps => &["BYTEPS_LOCAL_RANK", "LOCAL_RANK"][..],
        Backend::Horovod => &["HOROVOD_LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK", "LOCAL_RANK"][..],
    };
    parse_i32_env(vars).unwrap_or(0).max(0)
}

/// Returns local world size from environment variables.
pub fn local_size() -> i32 {
    let vars = match backend() {
        Backend::Byteps => &["BYTEPS_LOCAL_SIZE", "LOCAL_WORLD_SIZE"][..],
        Backend::Horovod => {
            &["HOROVOD_LOCAL_SIZE", "OMPI_COMM_WORLD_LOCAL_SIZE", "LOCAL_WORLD_SIZE"][..]
        }
    };
    parse_i32_env(vars).unwrap_or(1).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_MUTEX: Mutex<()> = Mutex::new(());

    const TEST_ENV_KEYS: &[&str] = &[
        "MONOLITH_WITH_BYTEPS",
        "BYTEPS_RANK",
        "BYTEPS_SIZE",
        "BYTEPS_LOCAL_RANK",
        "BYTEPS_LOCAL_SIZE",
        "HOROVOD_RANK",
        "HOROVOD_SIZE",
        "HOROVOD_LOCAL_RANK",
        "HOROVOD_LOCAL_SIZE",
        "OMPI_COMM_WORLD_RANK",
        "OMPI_COMM_WORLD_SIZE",
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "OMPI_COMM_WORLD_LOCAL_SIZE",
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
    ];

    struct EnvGuard(Vec<(&'static str, Option<String>)>);

    impl EnvGuard {
        fn capture() -> Self {
            let saved = TEST_ENV_KEYS
                .iter()
                .map(|&k| (k, std::env::var(k).ok()))
                .collect::<Vec<_>>();
            Self(saved)
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (key, value) in &self.0 {
                match value {
                    Some(v) => std::env::set_var(key, v),
                    None => std::env::remove_var(key),
                }
            }
        }
    }

    fn clear_test_env() {
        for key in TEST_ENV_KEYS {
            std::env::remove_var(key);
        }
    }

    #[test]
    fn test_backend_default_horovod() {
        let _lock = ENV_MUTEX
            .lock()
            .expect("hvd env mutex should not be poisoned");
        let _guard = EnvGuard::capture();
        clear_test_env();
        assert_eq!(backend(), Backend::Horovod);
    }

    #[test]
    fn test_backend_byteps() {
        let _lock = ENV_MUTEX
            .lock()
            .expect("hvd env mutex should not be poisoned");
        let _guard = EnvGuard::capture();
        clear_test_env();
        std::env::set_var("MONOLITH_WITH_BYTEPS", "1");
        assert_eq!(backend(), Backend::Byteps);
    }

    #[test]
    fn test_rank_size_horovod_env() {
        let _lock = ENV_MUTEX
            .lock()
            .expect("hvd env mutex should not be poisoned");
        let _guard = EnvGuard::capture();
        clear_test_env();
        std::env::set_var("HOROVOD_RANK", "2");
        std::env::set_var("HOROVOD_SIZE", "8");
        std::env::set_var("HOROVOD_LOCAL_RANK", "1");
        std::env::set_var("HOROVOD_LOCAL_SIZE", "4");

        assert_eq!(backend(), Backend::Horovod);
        assert_eq!(rank(), 2);
        assert_eq!(size(), 8);
        assert_eq!(local_rank(), 1);
        assert_eq!(local_size(), 4);
    }

    #[test]
    fn test_rank_size_byteps_env() {
        let _lock = ENV_MUTEX
            .lock()
            .expect("hvd env mutex should not be poisoned");
        let _guard = EnvGuard::capture();
        clear_test_env();
        std::env::set_var("MONOLITH_WITH_BYTEPS", "true");
        std::env::set_var("BYTEPS_RANK", "3");
        std::env::set_var("BYTEPS_SIZE", "16");
        std::env::set_var("BYTEPS_LOCAL_RANK", "2");
        std::env::set_var("BYTEPS_LOCAL_SIZE", "8");

        assert_eq!(backend(), Backend::Byteps);
        assert_eq!(rank(), 3);
        assert_eq!(size(), 16);
        assert_eq!(local_rank(), 2);
        assert_eq!(local_size(), 8);
    }

    #[test]
    fn test_rank_size_defaults_on_invalid_values() {
        let _lock = ENV_MUTEX
            .lock()
            .expect("hvd env mutex should not be poisoned");
        let _guard = EnvGuard::capture();
        clear_test_env();
        std::env::set_var("HOROVOD_RANK", "invalid");
        std::env::set_var("HOROVOD_SIZE", "-2");
        std::env::set_var("LOCAL_WORLD_SIZE", "0");

        assert_eq!(rank(), 0);
        assert_eq!(size(), 1);
        assert_eq!(local_size(), 1);
    }
}
