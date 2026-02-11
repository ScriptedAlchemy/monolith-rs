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

    #[test]
    fn test_get_zk_auth_data_none() {
        std::env::remove_var("ZK_AUTH");
        assert!(get_zk_auth_data().is_none());
    }

    #[test]
    fn test_get_zk_auth_data_some() {
        std::env::set_var("ZK_AUTH", "user:pass");
        let v = get_zk_auth_data().unwrap();
        assert_eq!(v, vec![("digest".to_string(), "user:pass".to_string())]);
    }
}
