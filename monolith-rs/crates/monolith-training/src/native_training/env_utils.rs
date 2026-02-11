//! Python `monolith.native_training.env_utils` parity.

/// Placeholder for Python's `setup_hdfs_env()` (no-op in this Rust port).
pub fn setup_hdfs_env() {}

/// Python's `generate_psm_from_uuid()` is a passthrough in this repository snapshot.
pub fn generate_psm_from_uuid(s: &str) -> String {
    s.to_string()
}

/// Returns ZooKeeper auth data from `ZK_AUTH` env, matching Python behavior.
///
/// Python returns `[("digest", ZK_AUTH)]` if `ZK_AUTH` is set (and prints it),
/// otherwise `None`.
pub fn get_zk_auth_data() -> Option<Vec<(String, String)>> {
    let zk_auth = std::env::var("ZK_AUTH").ok().and_then(|s| {
        let t = s.trim().to_string();
        if t.is_empty() {
            None
        } else {
            Some(t)
        }
    })?;

    // Keep the side-effect (print) for parity. This is intentionally stdout.
    println!("ZK_AUTH {zk_auth}");
    Some(vec![("digest".to_string(), zk_auth)])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    fn zk_auth_test_mutex() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    struct EnvSnapshot {
        key: &'static str,
        value: Option<String>,
    }

    impl EnvSnapshot {
        fn capture(key: &'static str) -> Self {
            Self {
                key,
                value: std::env::var(key).ok(),
            }
        }
    }

    impl Drop for EnvSnapshot {
        fn drop(&mut self) {
            if let Some(value) = &self.value {
                std::env::set_var(self.key, value);
            } else {
                std::env::remove_var(self.key);
            }
        }
    }

    #[test]
    fn test_get_zk_auth_data_none() {
        let _guard = zk_auth_test_mutex().lock().unwrap();
        let _snapshot = EnvSnapshot::capture("ZK_AUTH");
        std::env::remove_var("ZK_AUTH");
        assert!(get_zk_auth_data().is_none());
    }

    #[test]
    fn test_get_zk_auth_data_some() {
        let _guard = zk_auth_test_mutex().lock().unwrap();
        let _snapshot = EnvSnapshot::capture("ZK_AUTH");
        std::env::set_var("ZK_AUTH", "user:pass");
        let v = get_zk_auth_data().unwrap();
        assert_eq!(v, vec![("digest".to_string(), "user:pass".to_string())]);
    }

    #[test]
    fn test_get_zk_auth_data_empty_after_trim_is_none() {
        let _guard = zk_auth_test_mutex().lock().unwrap();
        let _snapshot = EnvSnapshot::capture("ZK_AUTH");
        std::env::set_var("ZK_AUTH", "   ");
        assert!(get_zk_auth_data().is_none());
    }
}
