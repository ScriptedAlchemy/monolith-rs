//! Memory profiling env setup mirroring Python `monolith/common/python/mem_profiling.py`.
//!
//! The Python version:
//! - appends `libtcmalloc.so` to `LD_PRELOAD`
//! - sets HEAPPROFILE-related env vars using gperftools conventions
//! - uses `MLPEnv().index` for per-worker file naming
//!
//! Rust parity: provide the same env variable side effects, but keep it optional and
//! non-fatal if the shared library does not exist (since we don't vendor TF/ops).

use std::path::{Path, PathBuf};

use crate::error::{MonolithError, Result};
use crate::path_utils::get_libops_path;

fn mlp_index_from_env() -> i32 {
    // Match Python's common case: MLP_ROLE_INDEX or 0.
    // (Python has MPI logic, but for heap profile file naming we only need a
    // stable integer; respecting explicit role index is sufficient.)
    std::env::var("MLP_ROLE_INDEX")
        .ok()
        .and_then(|s| s.trim().parse::<i32>().ok())
        .unwrap_or(0)
}

/// Enable tcmalloc by appending its path to `LD_PRELOAD`.
pub fn enable_tcmalloc() -> Result<()> {
    let mut libs: Vec<String> = std::env::var("LD_PRELOAD")
        .unwrap_or_default()
        .split(':')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.to_string())
        .collect();

    // Python:
    // utils.get_libops_path("../gperftools/libtcmalloc/lib/libtcmalloc.so")
    let tcmalloc = get_libops_path("../gperftools/libtcmalloc/lib/libtcmalloc.so")?;
    libs.push(tcmalloc.to_string_lossy().to_string());
    std::env::set_var("LD_PRELOAD", libs.join(":"));
    Ok(())
}

/// Setup heap profile env vars (gperftools).
pub fn setup_heap_profile(
    heap_profile_inuse_interval: i64,
    heap_profile_allocation_interval: i64,
    heap_profile_time_interval: i64,
    sample_ratio: f64,
    heap_profile_mmap: bool,
    heap_pro_file: Option<&Path>,
) -> Result<()> {
    if sample_ratio <= 0.0 {
        return Err(MonolithError::PyValueError {
            message: format!("sample_ratio must be > 0, got {}", sample_ratio),
        });
    }

    enable_tcmalloc()?;

    let base: PathBuf = match heap_pro_file {
        Some(p) => p.to_path_buf(),
        None => crate::path_utils::find_main()?,
    };

    let idx = mlp_index_from_env();
    let out = base.join(format!("hprof_{}", idx));

    std::env::set_var("HEAPPROFILE", out.to_string_lossy().to_string());
    std::env::set_var(
        "HEAP_PROFILE_INUSE_INTERVAL",
        (((heap_profile_inuse_interval as f64) / sample_ratio)
            .floor()
            .max(0.0) as i64)
            .to_string(),
    );
    std::env::set_var(
        "HEAP_PROFILE_ALLOCATION_INTERVAL",
        (((heap_profile_allocation_interval as f64) / sample_ratio)
            .floor()
            .max(0.0) as i64)
            .to_string(),
    );
    std::env::set_var("HEAP_PROFILE_SAMPLE_RATIO", sample_ratio.to_string());
    std::env::set_var(
        "HEAP_PROFILE_TIME_INTERVAL",
        heap_profile_time_interval.to_string(),
    );
    std::env::set_var(
        "HEAP_PROFILE_MMAP",
        heap_profile_mmap.to_string().to_lowercase(),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_setup_heap_profile_sets_env() {
        // Use override so find_main doesn't depend on Bazel layout.
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();
        std::env::set_var("MONOLITH_MAIN_DIR", root.to_string_lossy().to_string());
        std::env::set_var("MLP_ROLE_INDEX", "7");

        setup_heap_profile(100, 1000, 0, 1.0, false, None).unwrap();
        let hp = std::env::var("HEAPPROFILE").unwrap();
        assert!(hp.ends_with("hprof_7"));
        assert_eq!(std::env::var("HEAP_PROFILE_INUSE_INTERVAL").unwrap(), "100");
        assert_eq!(
            std::env::var("HEAP_PROFILE_ALLOCATION_INTERVAL").unwrap(),
            "1000"
        );
        assert_eq!(std::env::var("HEAP_PROFILE_MMAP").unwrap(), "false");

        std::env::remove_var("MONOLITH_MAIN_DIR");
        std::env::remove_var("MLP_ROLE_INDEX");
    }
}
