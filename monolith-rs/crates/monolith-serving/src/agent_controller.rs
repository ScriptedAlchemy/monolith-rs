//! Agent controller helpers (Python parity for `monolith/agent_service/agent_controller.py`).
//!
//! The upstream Python controller is a small CLI-oriented module that:
//! - declares saved model deploy configs under `/{bzid}/saved_models/{model}/{sub_graph}`
//! - publishes/unpublishes saved models to a layout path (ZK subtree)
//! - prints `bzid_info` for debugging
//!
//! For Rust parity we implement the same core helpers on top of `ZkBackend`.

use crate::backends::{SavedModel, SavedModelDeployConfig, ZkBackend, ZkError};
use std::path::{Path, PathBuf};

/// Best-effort model name discovery.
///
/// Python parses `saved_model.pb` to find the remote predict model name. The repository
/// testdata doesn't include `saved_model.pb`, so Rust parity falls back to `None`.
pub fn find_model_name(_exported_models_path: &Path) -> Option<String> {
    None
}

/// Declare all sub-graphs under `export_base` as saved models for `model_name`.
///
/// Python parity notes:
/// - Only `entry_ps` arch is supported (entry + ps_* sub-graphs).
/// - `entry` uses `latest` version policy; other sub-graphs use `latest_once`.
pub fn declare_saved_model(
    bd: &ZkBackend,
    export_base: &Path,
    model_name: Option<&str>,
    overwrite: bool,
    arch: &str,
) -> Result<String, ZkError> {
    if arch != "entry_ps" {
        return Err(ZkError::Other(
            "only entry + ps architecture supported".to_string(),
        ));
    }

    let model_name_from_export = find_model_name(export_base);
    let model_name = model_name
        .map(|s| s.to_string())
        .or(model_name_from_export)
        .ok_or_else(|| ZkError::Other("Model name is None".to_string()))?;

    let existing = bd.list_saved_models(&model_name)?;
    if !existing.is_empty() && !overwrite {
        return Err(ZkError::Other(format!(
            "{model_name} exists and not in overwrite mode"
        )));
    }

    // Python uses `tf.io.gfile.listdir()`. To keep deterministic behavior in tests, sort.
    let mut sub_graphs: Vec<String> = std::fs::read_dir(export_base)
        .map_err(|e| ZkError::Other(e.to_string()))?
        .filter_map(|e| e.ok())
        .filter_map(|e| e.file_name().to_str().map(|s| s.to_string()))
        .collect();
    sub_graphs.sort();

    for sub_graph in sub_graphs {
        let version_policy = if sub_graph == "entry" {
            "latest"
        } else {
            "latest_once"
        };
        let deploy_config = SavedModelDeployConfig {
            model_base_path: Some(
                PathBuf::from(export_base)
                    .join(&sub_graph)
                    .to_string_lossy()
                    .to_string(),
            ),
            version_policy: Some(version_policy.to_string()),
        };
        let saved_model = SavedModel::new(&model_name, sub_graph);
        bd.decl_saved_model(&saved_model, &deploy_config)?;
    }

    Ok(model_name)
}

/// Publish or unpublish saved models that match `model_pattern` to/from a layout path.
///
/// `model_pattern` should be of form `{model_name}:{sub_graph_glob}` e.g. `m:ps_*`.
pub fn map_model_to_layout(
    bd: &ZkBackend,
    model_pattern: &str,
    layout_path: &str,
    action: &str,
) -> Result<(), ZkError> {
    let (model_name, sub_graph_pattern) = model_pattern
        .split_once(':')
        .ok_or_else(|| ZkError::Other("model_pattern must contain ':'".to_string()))?;

    let saved = bd.list_saved_models(model_name)?;
    let sub_graphs = saved.into_iter().map(|s| s.sub_graph).collect::<Vec<_>>();
    let pat = glob::Pattern::new(sub_graph_pattern)
        .map_err(|e| ZkError::Other(format!("invalid sub_graph pattern: {e}")))?;

    for sub_graph in sub_graphs.into_iter().filter(|sg| pat.matches(sg)) {
        let saved_model = SavedModel::new(model_name, sub_graph);
        match action {
            "pub" => bd.add_to_layout(layout_path, &saved_model)?,
            "unpub" => bd.remove_from_layout(layout_path, &saved_model)?,
            other => {
                return Err(ZkError::Other(format!(
                    "unsupported action {other} (expected pub/unpub)"
                )))
            }
        }
    }
    Ok(())
}

/// Return a debug JSON structure for the backend state (Python parity for `bzid_info()`).
pub fn bzid_info(bd: &ZkBackend) -> Result<serde_json::Value, ZkError> {
    bd.bzid_info()
}
