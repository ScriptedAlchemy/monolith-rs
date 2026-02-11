//! Python `monolith.native_training.runner_utils` parity helpers.
//!
//! The Python module mostly deals with:
//! - choosing discovery backend (TF_CONFIG/MLP/Consul/ZK)
//! - copying a `checkpoint` file from `restore_dir` into `model_dir` (chief-only)
//!
//! In Rust we implement the behavior used by tests:
//! - `get_discovery_*` builders (already exist in `py_discovery`)
//! - `copy_checkpoint_from_restore_dir` which mirrors the file behavior.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::native_training::service_discovery::{
    ConsulServiceDiscovery, ServiceDiscovery as NativeServiceDiscovery, ServiceDiscoveryType,
};
use crate::native_training::env_utils::generate_psm_from_uuid;
use crate::py_discovery::{MlpServiceDiscovery, PyServiceDiscovery, TfConfigServiceDiscovery};
use crate::run_config::RunnerConfig;
use monolith_proto::monolith::native_training::monolith_checkpoint_state::HashTableType;
use monolith_proto::monolith::native_training::MonolithCheckpointState;
use monolith_proto::text_format;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RunnerUtilsError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("restore_dir does not contain checkpoint file: {0}")]
    MissingCheckpoint(PathBuf),
    #[error("restore_ckpt {restore_ckpt} is not in checkpoint.all_model_checkpoint_paths")]
    RestoreCkptNotFound { restore_ckpt: String },
    #[error("read ckpt error!")]
    ReadCheckpointFailed,
    #[error("TF_CONFIG is required for Primus discovery")]
    MissingTfConfig,
    #[error("psm is required for Consul discovery")]
    MissingPsm,
    #[error("Discovery error: {0}")]
    Discovery(String),
    #[error("Timed out waiting for restore synchronization file: {path}")]
    RestoreSyncTimeout { path: PathBuf },
}

/// Minimal subset of TensorFlow's `CheckpointState` used by the Python tests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointState {
    pub model_checkpoint_path: String,
    pub all_model_checkpoint_paths: Vec<String>,
}

/// Mode marker used by checkpoint override logic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunnerMode {
    Train,
    Eval,
    Predict,
}

impl RunnerMode {
    fn is_train(self) -> bool {
        matches!(self, RunnerMode::Train)
    }
}

/// Python-parity absolute path helper.
///
/// In upstream Python code `os.path.isabs` is monkey-patched so that `hdfs:/...`
/// is treated as absolute in addition to local filesystem absolute paths.
pub fn isabs(path: impl AsRef<str>) -> bool {
    let p = path.as_ref();
    p.starts_with("hdfs:/") || Path::new(p).is_absolute()
}

fn parse_checkpoint_pbtxt(input: &str) -> Option<CheckpointState> {
    // Extremely small parser matching the pbtxt written by Python tests:
    // model_checkpoint_path: 'model.ckpt-61'
    // all_model_checkpoint_paths: 'model.ckpt-61'
    // ...
    let mut model_checkpoint_path: Option<String> = None;
    let mut all: Vec<String> = Vec::new();

    for line in input.lines() {
        let line = line.trim();
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let (k, v) = match line.split_once(':') {
            Some(kv) => kv,
            None => continue,
        };
        let k = k.trim();
        let mut v = v.trim().to_string();
        // Drop surrounding quotes.
        if (v.starts_with('\'') && v.ends_with('\'')) || (v.starts_with('"') && v.ends_with('"')) {
            v = v[1..v.len() - 1].to_string();
        }
        match k {
            "model_checkpoint_path" => model_checkpoint_path = Some(v),
            "all_model_checkpoint_paths" => all.push(v),
            _ => {}
        }
    }

    model_checkpoint_path.map(|m| CheckpointState {
        model_checkpoint_path: m,
        all_model_checkpoint_paths: all,
    })
}

fn write_checkpoint_pbtxt(state: &CheckpointState) -> String {
    // Keep format close to TF's text_format output.
    let mut out = String::new();
    out.push_str(&format!(
        "model_checkpoint_path: \"{}\"\n",
        state.model_checkpoint_path
    ));
    for p in &state.all_model_checkpoint_paths {
        out.push_str(&format!("all_model_checkpoint_paths: \"{}\"\n", p));
    }
    out
}

/// Python-parity checkpoint query + restore override helper.
///
/// This mirrors the core behavior of Python's `gen_get_checkpoint_state()`:
/// - retries reads when checkpoint file exists but parse returns `None`,
/// - optionally overrides with `restore_ckpt` when allowed by mode/marker.
pub fn get_checkpoint_state_with_restore_override(
    model_dir: &Path,
    latest_filename: &str,
    restore_ckpt: Option<&str>,
    mode: RunnerMode,
    max_retries: usize,
    retry_interval: Duration,
) -> Result<Option<CheckpointState>, RunnerUtilsError> {
    let latest_file = model_dir.join(latest_filename);
    let mut state = None;
    for _ in 0..=max_retries {
        if latest_file.exists() {
            let txt = fs::read_to_string(&latest_file)?;
            state = parse_checkpoint_pbtxt(&txt);
            if state.is_some() {
                break;
            }
            std::thread::sleep(retry_interval);
        } else {
            return Ok(None);
        }
    }

    let mut state = state.ok_or(RunnerUtilsError::ReadCheckpointFailed)?;

    if latest_filename == "checkpoint" {
        if let Some(restore_ckpt) = restore_ckpt {
            let restore_base = Path::new(restore_ckpt)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or(restore_ckpt);
            let restore_path = Path::new(&state.model_checkpoint_path)
                .parent()
                .map(|p| p.join(restore_base))
                .unwrap_or_else(|| PathBuf::from(restore_base));
            let restore_path = restore_path.to_string_lossy().to_string();
            let has_restore = state
                .all_model_checkpoint_paths
                .iter()
                .any(|p| p == &restore_path);

            if has_restore {
                let restore_marker = model_dir.join("restore_ckpt");
                let should_apply = if mode.is_train() {
                    !restore_marker.exists()
                } else {
                    true
                };
                if should_apply && state.model_checkpoint_path != restore_path {
                    state.model_checkpoint_path = restore_path.clone();
                    state.all_model_checkpoint_paths = vec![restore_path.clone()];
                    fs::write(&latest_file, write_checkpoint_pbtxt(&state))?;
                    fs::write(restore_marker, restore_path)?;
                }
            }
        }
    }

    Ok(Some(state))
}

/// Discovery backends produced by [`get_discovery`].
#[derive(Debug)]
pub enum RunnerDiscovery {
    TfConfig(TfConfigServiceDiscovery),
    Mlp(MlpServiceDiscovery),
    Consul(ConsulServiceDiscovery),
    Zk {
        deep_insight_name: String,
        zk_server: String,
    },
}

impl RunnerDiscovery {
    pub fn kind(&self) -> &'static str {
        match self {
            RunnerDiscovery::TfConfig(_) => "tf_config",
            RunnerDiscovery::Mlp(_) => "mlp",
            RunnerDiscovery::Consul(_) => "consul",
            RunnerDiscovery::Zk { .. } => "zk",
        }
    }

    pub fn zk_config(&self) -> Option<(&str, &str)> {
        match self {
            RunnerDiscovery::Zk {
                deep_insight_name,
                zk_server,
            } => Some((deep_insight_name.as_str(), zk_server.as_str())),
            _ => None,
        }
    }

    pub fn close(&self) -> Result<(), RunnerUtilsError> {
        match self {
            RunnerDiscovery::TfConfig(d) => d
                .close()
                .map_err(|e| RunnerUtilsError::Discovery(e.to_string())),
            RunnerDiscovery::Mlp(d) => d
                .close()
                .map_err(|e| RunnerUtilsError::Discovery(e.to_string())),
            RunnerDiscovery::Consul(d) => d
                .close()
                .map_err(|e| RunnerUtilsError::Discovery(e.to_string())),
            RunnerDiscovery::Zk { .. } => Ok(()),
        }
    }
}

/// Context-style guard for discovery lifecycle, mirroring Python `monolith_discovery`.
pub struct MonolithDiscoveryGuard {
    discovery: Option<RunnerDiscovery>,
}

impl MonolithDiscoveryGuard {
    pub fn discovery(&self) -> Option<&RunnerDiscovery> {
        self.discovery.as_ref()
    }
}

impl Drop for MonolithDiscoveryGuard {
    fn drop(&mut self) {
        if let Some(d) = &self.discovery {
            let _ = d.close();
        }
    }
}

/// Select discovery backend from runner config.
pub fn get_discovery(
    runner_conf: &RunnerConfig,
    psm: Option<&str>,
) -> Result<Option<RunnerDiscovery>, RunnerUtilsError> {
    if runner_conf.is_local {
        return Ok(None);
    }

    let discovery = match runner_conf.discovery_type {
        ServiceDiscoveryType::Primus => {
            let tf_config = runner_conf
                .tf_config
                .clone()
                .or_else(|| std::env::var("TF_CONFIG").ok())
                .ok_or(RunnerUtilsError::MissingTfConfig)?;
            let d = TfConfigServiceDiscovery::new(&tf_config)
                .map_err(|e| RunnerUtilsError::Discovery(e.to_string()))?;
            RunnerDiscovery::TfConfig(d)
        }
        ServiceDiscoveryType::Consul => {
            let psm = psm.ok_or(RunnerUtilsError::MissingPsm)?;
            RunnerDiscovery::Consul(ConsulServiceDiscovery::new(psm.to_string()))
        }
        ServiceDiscoveryType::Mlp => RunnerDiscovery::Mlp(MlpServiceDiscovery::new()),
        ServiceDiscoveryType::Zk => RunnerDiscovery::Zk {
            deep_insight_name: runner_conf.deep_insight_name.clone(),
            zk_server: runner_conf.zk_server.clone(),
        },
    };
    Ok(Some(discovery))
}

/// Builds a discovery guard; discovery will be closed automatically on drop.
pub fn monolith_discovery(
    runner_conf: &RunnerConfig,
    psm: Option<&str>,
) -> Result<MonolithDiscoveryGuard, RunnerUtilsError> {
    let resolved_psm = if matches!(runner_conf.discovery_type, ServiceDiscoveryType::Consul)
        && psm.is_none()
    {
        Some(generate_psm_from_uuid(&runner_conf.deep_insight_name))
    } else {
        psm.map(|s| s.to_string())
    };

    Ok(MonolithDiscoveryGuard {
        discovery: get_discovery(runner_conf, resolved_psm.as_deref())?,
    })
}

/// Copy a `checkpoint` file from `restore_dir` into `model_dir` and write a `restore_ckpt` marker.
///
/// Mirrors the behavior in Python's `RunnerConfig._copy_ckpt_file()` as used by
/// `monolith/native_training/runner_utils_test.py`.
pub fn copy_checkpoint_from_restore_dir(
    restore_dir: &Path,
    model_dir: &Path,
    restore_ckpt: Option<&str>,
) -> Result<CheckpointState, RunnerUtilsError> {
    let src = restore_dir.join("checkpoint");
    if !src.exists() {
        return Err(RunnerUtilsError::MissingCheckpoint(src));
    }

    fs::create_dir_all(model_dir)?;

    let src_txt = fs::read_to_string(&src)?;
    let restore_state = parse_checkpoint_pbtxt(&src_txt).unwrap_or(CheckpointState {
        model_checkpoint_path: "model.ckpt-0".to_string(),
        all_model_checkpoint_paths: Vec::new(),
    });

    let picked = if let Some(ckpt) = restore_ckpt {
        if restore_state
            .all_model_checkpoint_paths
            .iter()
            .any(|p| p == ckpt)
        {
            ckpt.to_string()
        } else if restore_state
            .all_model_checkpoint_paths
            .iter()
            .any(|p| Path::new(p).file_name().and_then(|s| s.to_str()) == Some(ckpt))
        {
            // Accept basename-only input.
            let base = ckpt;
            let found = restore_state
                .all_model_checkpoint_paths
                .iter()
                .find(|p| Path::new(p).file_name().and_then(|s| s.to_str()) == Some(base))
                .cloned()
                .unwrap();
            found
        } else {
            return Err(RunnerUtilsError::RestoreCkptNotFound {
                restore_ckpt: ckpt.to_string(),
            });
        }
    } else {
        restore_state.model_checkpoint_path.clone()
    };

    let dst_state = CheckpointState {
        model_checkpoint_path: picked.clone(),
        all_model_checkpoint_paths: vec![picked.clone()],
    };

    let dst_checkpoint_file = model_dir.join("checkpoint");
    if !dst_checkpoint_file.exists() {
        fs::write(&dst_checkpoint_file, write_checkpoint_pbtxt(&dst_state))?;
    }

    // Write restore_ckpt marker (Python uses it as a flag that restore_dir ckpt already applied).
    let restore_marker = model_dir.join("restore_ckpt");
    if !restore_marker.exists() {
        fs::write(&restore_marker, picked)?;
    }

    // Create/update monolith_checkpoint for parity.
    //
    // Python writes MonolithCheckpointState pbtxt and carries forward any existing state
    // already in model_dir, then appends the chosen restore checkpoint path into
    // `exempt_model_checkpoint_paths`.
    let monolith_checkpoint = model_dir.join("monolith_checkpoint");
    if !monolith_checkpoint.exists() {
        // Use defaults consistent with Python's SaverHook path: type = CUCKOO_HASH_MAP.
        let mut st = MonolithCheckpointState::default();
        st.builtin_hash_table_type = Some(HashTableType::CuckooHashMap as i32);
        st.exempt_model_checkpoint_paths = dst_state.all_model_checkpoint_paths.clone();
        let pbtxt = text_format::to_pbtxt("monolith.native_training.MonolithCheckpointState", &st)
            .map_err(|e| RunnerUtilsError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;
        fs::write(&monolith_checkpoint, pbtxt)?;
    } else {
        // Best-effort merge: parse, update exempt list, then overwrite.
        let existing = fs::read_to_string(&monolith_checkpoint)?;
        let mut st: MonolithCheckpointState = text_format::parse_pbtxt(
            "monolith.native_training.MonolithCheckpointState",
            &existing,
        )
        .unwrap_or_default();
        for p in &dst_state.all_model_checkpoint_paths {
            if !st.exempt_model_checkpoint_paths.iter().any(|x| x == p) {
                st.exempt_model_checkpoint_paths.push(p.clone());
            }
        }
        if st.builtin_hash_table_type.is_none() {
            st.builtin_hash_table_type = Some(HashTableType::CuckooHashMap as i32);
        }
        let pbtxt = text_format::to_pbtxt("monolith.native_training.MonolithCheckpointState", &st)
            .map_err(|e| RunnerUtilsError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;
        fs::write(&monolith_checkpoint, pbtxt)?;
    }

    Ok(dst_state)
}

/// Chief/non-chief restore synchronization behavior.
///
/// - chief: performs restore checkpoint copy immediately.
/// - non-chief: waits until chief has written synchronization artifacts.
pub fn prepare_restore_checkpoint(
    restore_dir: &Path,
    model_dir: &Path,
    restore_ckpt: Option<&str>,
    is_chief: bool,
    timeout: Duration,
    poll_interval: Duration,
) -> Result<CheckpointState, RunnerUtilsError> {
    if is_chief {
        return copy_checkpoint_from_restore_dir(restore_dir, model_dir, restore_ckpt);
    }

    let checkpoint_file = model_dir.join("checkpoint");
    let monolith_checkpoint_file = model_dir.join("monolith_checkpoint");
    let deadline = Instant::now() + timeout;
    loop {
        if checkpoint_file.exists() && monolith_checkpoint_file.exists() {
            let txt = fs::read_to_string(&checkpoint_file)?;
            if let Some(st) = parse_checkpoint_pbtxt(&txt) {
                return Ok(st);
            }
        }
        if Instant::now() >= deadline {
            return Err(RunnerUtilsError::RestoreSyncTimeout {
                path: monolith_checkpoint_file,
            });
        }
        std::thread::sleep(poll_interval);
    }
}

/// RunnerConfig-driven restore initialization helper.
///
/// Mirrors Python `RunnerConfig.__post_init__` restore behavior:
/// - if `restore_dir` is not set, do nothing,
/// - chief role performs copy immediately,
/// - non-chief waits for synchronized checkpoint artifacts.
pub fn initialize_restore_checkpoint_from_runner(
    runner_conf: &RunnerConfig,
    timeout: Duration,
    poll_interval: Duration,
) -> Result<Option<CheckpointState>, RunnerUtilsError> {
    crate::run_config::RunConfig::apply_runtime_env_exports(runner_conf);

    let Some(restore_dir) = runner_conf.restore_dir.as_ref() else {
        return Ok(None);
    };
    let is_chief = runner_conf.is_local || runner_conf.index == 0;
    let st = prepare_restore_checkpoint(
        restore_dir,
        &runner_conf.model_dir,
        runner_conf.restore_ckpt.as_deref(),
        is_chief,
        timeout,
        poll_interval,
    )?;
    Ok(Some(st))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_training::service_discovery::ServiceDiscoveryType;
    use crate::run_config::RunnerConfig;
    use tempfile::tempdir;

    #[test]
    fn test_copy_checkpoint_from_restore_dir() {
        let tmp = tempdir().unwrap();
        let restore_dir = tmp.path().join("restore_dir");
        let model_dir = tmp.path().join("model_dir");
        fs::create_dir_all(&restore_dir).unwrap();

        let pbtxt = r#"
model_checkpoint_path: "model.ckpt-61"
all_model_checkpoint_paths: "model.ckpt-61"
all_model_checkpoint_paths: "model.ckpt-30"
all_model_checkpoint_paths: "model.ckpt-0"
"#;
        fs::write(restore_dir.join("checkpoint"), pbtxt).unwrap();

        let state =
            copy_checkpoint_from_restore_dir(&restore_dir, &model_dir, Some("model.ckpt-30"))
                .unwrap();
        assert_eq!(
            Path::new(&state.model_checkpoint_path).file_name().unwrap(),
            "model.ckpt-30"
        );
        assert!(model_dir.join("monolith_checkpoint").exists());
        assert!(model_dir.join("restore_ckpt").exists());
        assert!(model_dir.join("checkpoint").exists());
    }

    #[test]
    fn test_isabs_supports_hdfs_scheme() {
        assert!(isabs("hdfs:/tmp/model"));
        assert!(isabs("/tmp/model"));
        assert!(!isabs("relative/path"));
    }

    #[test]
    fn test_get_discovery_local() {
        let rc = RunnerConfig {
            is_local: true,
            ..RunnerConfig::default()
        };
        let discovery = get_discovery(&rc, None).unwrap();
        assert!(discovery.is_none());
    }

    #[test]
    fn test_get_discovery_primus() {
        let tf_config = serde_json::json!({
          "cluster": {
            "chief": ["host0:2222"],
            "ps": ["host1:2222"],
            "worker": ["host2:2222"]
          },
          "task": {"type": "worker", "index": 0}
        })
        .to_string();
        let rc = RunnerConfig {
            is_local: false,
            discovery_type: ServiceDiscoveryType::Primus,
            tf_config: Some(tf_config),
            ..RunnerConfig::default()
        };
        let discovery = get_discovery(&rc, None).unwrap();
        let d = discovery.expect("expected discovery");
        assert_eq!(d.kind(), "tf_config");
    }

    #[test]
    fn test_get_discovery_consul_requires_psm() {
        let rc = RunnerConfig {
            is_local: false,
            discovery_type: ServiceDiscoveryType::Consul,
            ..RunnerConfig::default()
        };
        let err = get_discovery(&rc, None).unwrap_err();
        assert!(matches!(err, RunnerUtilsError::MissingPsm));
    }

    #[test]
    fn test_get_discovery_consul_with_psm() {
        let rc = RunnerConfig {
            is_local: false,
            discovery_type: ServiceDiscoveryType::Consul,
            ..RunnerConfig::default()
        };
        let discovery = get_discovery(&rc, Some("test_psm")).unwrap();
        assert_eq!(discovery.unwrap().kind(), "consul");
    }

    #[test]
    fn test_get_discovery_zk() {
        let rc = RunnerConfig {
            is_local: false,
            discovery_type: ServiceDiscoveryType::Zk,
            deep_insight_name: "test_job".to_string(),
            zk_server: "127.0.0.1:2181".to_string(),
            ..RunnerConfig::default()
        };
        let discovery = get_discovery(&rc, None).unwrap().expect("discovery");
        assert_eq!(discovery.kind(), "zk");
        let (job, zk) = discovery.zk_config().expect("zk config");
        assert_eq!(job, "test_job");
        assert_eq!(zk, "127.0.0.1:2181");
    }

    #[test]
    fn test_prepare_restore_checkpoint_non_chief_waits_for_chief_sync() {
        let tmp = tempdir().unwrap();
        let restore_dir = tmp.path().join("restore");
        let model_dir = tmp.path().join("model");
        fs::create_dir_all(&restore_dir).unwrap();

        let pbtxt = r#"
model_checkpoint_path: "model.ckpt-61"
all_model_checkpoint_paths: "model.ckpt-61"
all_model_checkpoint_paths: "model.ckpt-30"
"#;
        fs::write(restore_dir.join("checkpoint"), pbtxt).unwrap();
        fs::create_dir_all(&model_dir).unwrap();

        let restore_dir_bg = restore_dir.clone();
        let model_dir_bg = model_dir.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            let _ = copy_checkpoint_from_restore_dir(&restore_dir_bg, &model_dir_bg, None);
        });

        let st = prepare_restore_checkpoint(
            &restore_dir,
            &model_dir,
            None,
            false,
            Duration::from_secs(2),
            Duration::from_millis(20),
        )
        .unwrap();
        assert_eq!(
            Path::new(&st.model_checkpoint_path).file_name().unwrap(),
            "model.ckpt-61"
        );
    }

    #[test]
    fn test_prepare_restore_checkpoint_non_chief_timeout() {
        let tmp = tempdir().unwrap();
        let restore_dir = tmp.path().join("restore");
        let model_dir = tmp.path().join("model");
        fs::create_dir_all(&restore_dir).unwrap();
        fs::create_dir_all(&model_dir).unwrap();

        let err = prepare_restore_checkpoint(
            &restore_dir,
            &model_dir,
            None,
            false,
            Duration::from_millis(150),
            Duration::from_millis(20),
        )
        .unwrap_err();
        assert!(matches!(err, RunnerUtilsError::RestoreSyncTimeout { .. }));
    }

    #[test]
    fn test_monolith_discovery_consul_auto_psm() {
        let rc = RunnerConfig {
            is_local: false,
            discovery_type: ServiceDiscoveryType::Consul,
            deep_insight_name: "uuid-like-job".to_string(),
            ..RunnerConfig::default()
        };
        let guard = monolith_discovery(&rc, None).unwrap();
        let d = guard.discovery().expect("consul discovery");
        assert_eq!(d.kind(), "consul");
    }

    #[test]
    fn test_get_checkpoint_state_with_restore_override_train() {
        let tmp = tempdir().unwrap();
        let model_dir = tmp.path().join("model");
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(
            model_dir.join("checkpoint"),
            r#"
model_checkpoint_path: "model.ckpt-61"
all_model_checkpoint_paths: "model.ckpt-61"
all_model_checkpoint_paths: "model.ckpt-30"
"#,
        )
        .unwrap();

        let st = get_checkpoint_state_with_restore_override(
            &model_dir,
            "checkpoint",
            Some("model.ckpt-30"),
            RunnerMode::Train,
            1,
            Duration::from_millis(1),
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            Path::new(&st.model_checkpoint_path).file_name().unwrap(),
            "model.ckpt-30"
        );
        assert!(model_dir.join("restore_ckpt").exists());
    }

    #[test]
    fn test_get_checkpoint_state_with_restore_override_train_marker_blocks() {
        let tmp = tempdir().unwrap();
        let model_dir = tmp.path().join("model");
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(
            model_dir.join("checkpoint"),
            r#"
model_checkpoint_path: "model.ckpt-61"
all_model_checkpoint_paths: "model.ckpt-61"
all_model_checkpoint_paths: "model.ckpt-30"
"#,
        )
        .unwrap();
        fs::write(model_dir.join("restore_ckpt"), "model.ckpt-30").unwrap();

        let st = get_checkpoint_state_with_restore_override(
            &model_dir,
            "checkpoint",
            Some("model.ckpt-30"),
            RunnerMode::Train,
            1,
            Duration::from_millis(1),
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            Path::new(&st.model_checkpoint_path).file_name().unwrap(),
            "model.ckpt-61"
        );
    }

    #[test]
    fn test_get_checkpoint_state_with_restore_override_eval_always_applies() {
        let tmp = tempdir().unwrap();
        let model_dir = tmp.path().join("model");
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(
            model_dir.join("checkpoint"),
            r#"
model_checkpoint_path: "model.ckpt-61"
all_model_checkpoint_paths: "model.ckpt-61"
all_model_checkpoint_paths: "model.ckpt-30"
"#,
        )
        .unwrap();
        fs::write(model_dir.join("restore_ckpt"), "model.ckpt-30").unwrap();

        let st = get_checkpoint_state_with_restore_override(
            &model_dir,
            "checkpoint",
            Some("model.ckpt-30"),
            RunnerMode::Eval,
            1,
            Duration::from_millis(1),
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            Path::new(&st.model_checkpoint_path).file_name().unwrap(),
            "model.ckpt-30"
        );
    }

    #[test]
    fn test_get_checkpoint_state_with_restore_override_read_error() {
        let tmp = tempdir().unwrap();
        let model_dir = tmp.path().join("model");
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("checkpoint"), "not a checkpoint state").unwrap();

        let err = get_checkpoint_state_with_restore_override(
            &model_dir,
            "checkpoint",
            Some("model.ckpt-30"),
            RunnerMode::Train,
            1,
            Duration::from_millis(1),
        )
        .unwrap_err();
        assert!(matches!(err, RunnerUtilsError::ReadCheckpointFailed));
    }

    #[test]
    fn test_initialize_restore_checkpoint_from_runner_none_when_restore_missing() {
        static ENV_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _env_guard = ENV_MUTEX.lock().unwrap();
        std::env::remove_var("TF_GRPC_WORKER_CACHE_THREADS");
        std::env::remove_var("MONOLITH_GRPC_WORKER_SERVICE_HANDLER_MULTIPLIER");

        let rc = RunnerConfig {
            restore_dir: None,
            tf_grpc_worker_cache_threads: Some(6),
            monolith_grpc_worker_service_handler_multiplier: Some(4),
            ..RunnerConfig::default()
        };
        let st =
            initialize_restore_checkpoint_from_runner(&rc, Duration::from_secs(1), Duration::from_millis(1))
                .unwrap();
        assert!(st.is_none());
        assert_eq!(
            std::env::var("TF_GRPC_WORKER_CACHE_THREADS").unwrap(),
            "6"
        );
        assert_eq!(
            std::env::var("MONOLITH_GRPC_WORKER_SERVICE_HANDLER_MULTIPLIER").unwrap(),
            "4"
        );
    }

    #[test]
    fn test_initialize_restore_checkpoint_from_runner_worker_waits_for_chief() {
        let tmp = tempdir().unwrap();
        let restore_dir = tmp.path().join("restore");
        let model_dir = tmp.path().join("model");
        fs::create_dir_all(&restore_dir).unwrap();
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(
            restore_dir.join("checkpoint"),
            r#"
model_checkpoint_path: "model.ckpt-61"
all_model_checkpoint_paths: "model.ckpt-61"
all_model_checkpoint_paths: "model.ckpt-30"
"#,
        )
        .unwrap();

        let chief = RunnerConfig {
            is_local: false,
            index: 0,
            model_dir: model_dir.clone(),
            restore_dir: Some(restore_dir.clone()),
            restore_ckpt: Some("model.ckpt-30".to_string()),
            ..RunnerConfig::default()
        };
        let worker = RunnerConfig {
            is_local: false,
            index: 1,
            model_dir: model_dir.clone(),
            restore_dir: Some(restore_dir.clone()),
            restore_ckpt: Some("model.ckpt-30".to_string()),
            ..RunnerConfig::default()
        };

        let chief_bg = chief.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            let _ = initialize_restore_checkpoint_from_runner(
                &chief_bg,
                Duration::from_secs(1),
                Duration::from_millis(10),
            );
        });

        let st = initialize_restore_checkpoint_from_runner(
            &worker,
            Duration::from_secs(2),
            Duration::from_millis(10),
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            Path::new(&st.model_checkpoint_path).file_name().unwrap(),
            "model.ckpt-30"
        );
    }
}
