//! Model export functionality for serving.
//!
//! This module provides `ModelExporter` for exporting trained models
//! to a format suitable for serving inference requests.

use crate::state::{HashTableState, ModelState};
use crate::CheckpointError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Model spec file name used by the Candle-backed serving stack.
pub const MODEL_SPEC_FILENAME: &str = "model_spec.json";

/// Minimal model specification for serving.
///
/// This is intentionally decoupled from Python/TensorFlow graph export: the Rust
/// serving path uses Candle and needs a small JSON schema to reconstruct the
/// inference graph.
///
/// The naming convention for dense params must match monolith-serving's Candle
/// loader (`monolith-serving/src/inference.rs`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ModelSpec {
    /// Simple feed-forward network.
    Mlp {
        input_dim: usize,
        hidden_dims: Vec<usize>,
        output_dim: usize,
        #[serde(default)]
        activation: String,
    },
    /// Deep & Cross Network.
    Dcn {
        input_dim: usize,
        cross_layers: usize,
        #[serde(default)]
        deep_hidden_dims: Vec<usize>,
        output_dim: usize,
        #[serde(default)]
        activation: String,
    },
    /// Multi-gate Mixture-of-Experts.
    Mmoe {
        input_dim: usize,
        num_experts: usize,
        expert_hidden_dims: Vec<usize>,
        num_tasks: usize,
        gate_hidden_dims: Vec<usize>,
        task_output_dim: usize,
        #[serde(default)]
        activation: String,
    },
}

fn normalize_activation(act: &str) -> &'static str {
    // Keep in sync with monolith-serving Activation serde rename_all = snake_case.
    match act.trim().to_lowercase().as_str() {
        "relu" => "relu",
        "tanh" => "tanh",
        "sigmoid" => "sigmoid",
        "none" | "linear" => "none",
        _ => "relu",
    }
}

fn guess_model_spec_from_state(state: &ModelState) -> Option<ModelSpec> {
    // Heuristic-based: if user provided metadata, prefer it; otherwise try common param keys.
    let ty = state
        .metadata
        .get("model_spec_type")
        .map(|s| s.trim().to_lowercase());

    let model_type = ty.as_deref().or_else(|| {
        if state.dense_params.keys().any(|k| k.starts_with("mmoe.")) {
            Some("mmoe")
        } else if state.dense_params.keys().any(|k| k.starts_with("dcn.")) {
            Some("dcn")
        } else if state.dense_params.keys().any(|k| k.starts_with("mlp.")) {
            Some("mlp")
        } else {
            None
        }
    })?;

    match model_type {
        "mlp" => {
            let input_dim = state
                .metadata
                .get("model_input_dim")
                .and_then(|v| v.parse::<usize>().ok())?;
            let output_dim = state
                .metadata
                .get("model_output_dim")
                .and_then(|v| v.parse::<usize>().ok())?;
            let hidden_dims: Vec<usize> = state
                .metadata
                .get("model_hidden_dims")
                .map(|s| {
                    s.split(',')
                        .filter_map(|x| x.trim().parse::<usize>().ok())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            let activation = normalize_activation(
                state
                    .metadata
                    .get("model_activation")
                    .map(|s| s.as_str())
                    .unwrap_or("relu"),
            )
            .to_string();
            Some(ModelSpec::Mlp {
                input_dim,
                hidden_dims,
                output_dim,
                activation,
            })
        }
        "dcn" => {
            let input_dim = state
                .metadata
                .get("model_input_dim")
                .and_then(|v| v.parse::<usize>().ok())?;
            let output_dim = state
                .metadata
                .get("model_output_dim")
                .and_then(|v| v.parse::<usize>().ok())?;
            let cross_layers = state
                .metadata
                .get("model_cross_layers")
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(2);
            let deep_hidden_dims: Vec<usize> = state
                .metadata
                .get("model_deep_hidden_dims")
                .map(|s| {
                    s.split(',')
                        .filter_map(|x| x.trim().parse::<usize>().ok())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            let activation = normalize_activation(
                state
                    .metadata
                    .get("model_activation")
                    .map(|s| s.as_str())
                    .unwrap_or("relu"),
            )
            .to_string();
            Some(ModelSpec::Dcn {
                input_dim,
                cross_layers,
                deep_hidden_dims,
                output_dim,
                activation,
            })
        }
        "mmoe" => {
            let input_dim = state
                .metadata
                .get("model_input_dim")
                .and_then(|v| v.parse::<usize>().ok())?;
            let num_experts = state
                .metadata
                .get("model_num_experts")
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(4);
            let num_tasks = state
                .metadata
                .get("model_num_tasks")
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(1);
            let task_output_dim = state
                .metadata
                .get("model_task_output_dim")
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(1);
            let expert_hidden_dims: Vec<usize> = state
                .metadata
                .get("model_expert_hidden_dims")
                .map(|s| {
                    s.split(',')
                        .filter_map(|x| x.trim().parse::<usize>().ok())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            let gate_hidden_dims: Vec<usize> = state
                .metadata
                .get("model_gate_hidden_dims")
                .map(|s| {
                    s.split(',')
                        .filter_map(|x| x.trim().parse::<usize>().ok())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            let activation = normalize_activation(
                state
                    .metadata
                    .get("model_activation")
                    .map(|s| s.as_str())
                    .unwrap_or("relu"),
            )
            .to_string();
            Some(ModelSpec::Mmoe {
                input_dim,
                num_experts,
                expert_hidden_dims,
                num_tasks,
                gate_hidden_dims,
                task_output_dim,
                activation,
            })
        }
        _ => None,
    }
}

fn write_model_spec_if_present(output_dir: &Path, state: &ModelState) -> Result<()> {
    let Some(spec) = guess_model_spec_from_state(state) else {
        // If we can't infer a spec, skip writing it. Serving will fall back to baseline inference.
        return Ok(());
    };
    let path = output_dir.join(MODEL_SPEC_FILENAME);
    let json = serde_json::to_string_pretty(&spec).map_err(CheckpointError::Serialization)?;
    std::fs::write(&path, json).map_err(|e| CheckpointError::Io { path, source: e })?;
    Ok(())
}

/// Result type for export operations.
pub type Result<T> = std::result::Result<T, CheckpointError>;

/// Export format for the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ExportFormat {
    /// JSON-based export format (human-readable).
    Json,

    /// Binary format (more compact).
    Binary,

    /// SavedModel-like directory structure.
    #[default]
    SavedModel,
}

/// Configuration for model export.
#[derive(Debug, Clone)]
pub struct ExportConfig {
    /// Output directory for the exported model.
    pub output_dir: PathBuf,

    /// Export format to use.
    pub format: ExportFormat,

    /// Whether to include optimizer state in export.
    /// Usually false for inference-only exports.
    pub include_optimizer: bool,

    /// Whether to freeze embedding tables (convert to dense).
    pub freeze_embeddings: bool,

    /// Optional model version/tag.
    pub version: Option<String>,

    /// Additional metadata to include in export.
    pub metadata: HashMap<String, String>,
}

impl ExportConfig {
    /// Create a new export configuration.
    ///
    /// # Arguments
    ///
    /// * `output_dir` - Directory where the model will be exported
    pub fn new(output_dir: impl Into<PathBuf>) -> Self {
        Self {
            output_dir: output_dir.into(),
            format: ExportFormat::default(),
            include_optimizer: false,
            freeze_embeddings: false,
            version: None,
            metadata: HashMap::new(),
        }
    }

    /// Set the export format.
    pub fn with_format(mut self, format: ExportFormat) -> Self {
        self.format = format;
        self
    }

    /// Include optimizer state in export.
    pub fn with_optimizer(mut self, include: bool) -> Self {
        self.include_optimizer = include;
        self
    }

    /// Freeze embedding tables.
    pub fn with_freeze_embeddings(mut self, freeze: bool) -> Self {
        self.freeze_embeddings = freeze;
        self
    }

    /// Set model version.
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = Some(version.into());
        self
    }

    /// Add metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Exported model manifest.
///
/// Contains metadata about an exported model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportManifest {
    /// Model version.
    pub version: String,

    /// Export format used.
    pub format: String,

    /// Global step at export time.
    pub global_step: u64,

    /// Timestamp of export (Unix epoch seconds).
    pub timestamp: u64,

    /// List of embedding table names.
    pub embedding_tables: Vec<String>,

    /// List of dense parameter names.
    pub dense_params: Vec<String>,

    /// Whether optimizer state is included.
    pub includes_optimizer: bool,

    /// Additional metadata.
    pub metadata: HashMap<String, String>,
}

/// Exports trained models for serving.
///
/// `ModelExporter` converts model state to a format optimized for
/// inference serving, optionally stripping training-specific state.
///
/// # Examples
///
/// ```no_run
/// use monolith_checkpoint::{ModelExporter, ExportConfig, ExportFormat, ModelState};
///
/// fn main() -> monolith_checkpoint::Result<()> {
///     let config = ExportConfig::new("/tmp/exported_model")
///         .with_format(ExportFormat::SavedModel)
///         .with_version("1.0.0");
///
///     let exporter = ModelExporter::new(config);
///     let model_state = ModelState::new(1000);
///     exporter.export(&model_state)?;
///     Ok(())
/// }
/// ```
#[derive(Debug)]
pub struct ModelExporter {
    /// Export configuration.
    config: ExportConfig,
}

impl ModelExporter {
    /// Create a new model exporter.
    ///
    /// # Arguments
    ///
    /// * `config` - Export configuration
    pub fn new(config: ExportConfig) -> Self {
        Self { config }
    }

    /// Export model state.
    ///
    /// # Arguments
    ///
    /// * `state` - Model state to export
    ///
    /// # Returns
    ///
    /// Returns the path to the exported model.
    pub fn export(&self, state: &ModelState) -> Result<PathBuf> {
        tracing::info!(
            output = %self.config.output_dir.display(),
            format = ?self.config.format,
            step = state.global_step,
            "Exporting model"
        );

        // Create output directory
        std::fs::create_dir_all(&self.config.output_dir).map_err(|e| CheckpointError::Io {
            path: self.config.output_dir.clone(),
            source: e,
        })?;

        match self.config.format {
            ExportFormat::Json => self.export_json(state),
            ExportFormat::Binary => self.export_binary(state),
            ExportFormat::SavedModel => self.export_saved_model(state),
        }
    }

    /// Export to JSON format.
    fn export_json(&self, state: &ModelState) -> Result<PathBuf> {
        let export_state = self.prepare_export_state(state);
        let output_path = self.config.output_dir.join("model.json");

        let json =
            serde_json::to_string_pretty(&export_state).map_err(CheckpointError::Serialization)?;

        std::fs::write(&output_path, json).map_err(|e| CheckpointError::Io {
            path: output_path.clone(),
            source: e,
        })?;

        // Write manifest
        self.write_manifest(state)?;

        Ok(self.config.output_dir.clone())
    }

    /// Export to binary format.
    fn export_binary(&self, state: &ModelState) -> Result<PathBuf> {
        let export_state = self.prepare_export_state(state);
        let output_path = self.config.output_dir.join("model.bin");

        // Use JSON as placeholder for binary format
        let data = serde_json::to_vec(&export_state).map_err(CheckpointError::Serialization)?;

        std::fs::write(&output_path, data).map_err(|e| CheckpointError::Io {
            path: output_path.clone(),
            source: e,
        })?;

        // Write manifest
        self.write_manifest(state)?;

        Ok(self.config.output_dir.clone())
    }

    /// Export to SavedModel-like directory structure.
    ///
    /// Structure:
    /// ```text
    /// output_dir/
    ///   manifest.json
    ///   embeddings/
    ///     table_name.json
    ///   dense/
    ///     params.json
    ///   optimizer/  (if included)
    ///     state.json
    /// ```
    fn export_saved_model(&self, state: &ModelState) -> Result<PathBuf> {
        // Create subdirectories
        let embeddings_dir = self.config.output_dir.join("embeddings");
        let dense_dir = self.config.output_dir.join("dense");

        std::fs::create_dir_all(&embeddings_dir).map_err(|e| CheckpointError::Io {
            path: embeddings_dir.clone(),
            source: e,
        })?;
        std::fs::create_dir_all(&dense_dir).map_err(|e| CheckpointError::Io {
            path: dense_dir.clone(),
            source: e,
        })?;

        // Export embedding tables
        for table in &state.hash_tables {
            let table_path = embeddings_dir.join(format!("{}.json", table.name));
            let json =
                serde_json::to_string_pretty(table).map_err(CheckpointError::Serialization)?;
            std::fs::write(&table_path, json).map_err(|e| CheckpointError::Io {
                path: table_path,
                source: e,
            })?;
        }

        // Export dense parameters
        let dense_path = dense_dir.join("params.json");
        let json = serde_json::to_string_pretty(&state.dense_params)
            .map_err(CheckpointError::Serialization)?;
        std::fs::write(&dense_path, json).map_err(|e| CheckpointError::Io {
            path: dense_path,
            source: e,
        })?;

        // Export optimizer state if requested
        if self.config.include_optimizer && !state.optimizers.is_empty() {
            let optimizer_dir = self.config.output_dir.join("optimizer");
            std::fs::create_dir_all(&optimizer_dir).map_err(|e| CheckpointError::Io {
                path: optimizer_dir.clone(),
                source: e,
            })?;

            let optimizer_path = optimizer_dir.join("state.json");
            let json = serde_json::to_string_pretty(&state.optimizers)
                .map_err(CheckpointError::Serialization)?;
            std::fs::write(&optimizer_path, json).map_err(|e| CheckpointError::Io {
                path: optimizer_path,
                source: e,
            })?;
        }

        // Write model spec for Candle-backed serving if available.
        write_model_spec_if_present(&self.config.output_dir, state)?;

        // Write manifest
        self.write_manifest(state)?;

        tracing::info!(
            output = %self.config.output_dir.display(),
            tables = state.hash_tables.len(),
            params = state.dense_params.len(),
            "SavedModel export complete"
        );

        Ok(self.config.output_dir.clone())
    }

    /// Prepare model state for export (strip unnecessary data).
    fn prepare_export_state(&self, state: &ModelState) -> ModelState {
        let mut export_state = state.clone();

        // Remove optimizer state unless explicitly included
        if !self.config.include_optimizer {
            export_state.optimizers.clear();
        }

        // Add export metadata
        export_state.set_metadata("export_format", format!("{:?}", self.config.format));
        if let Some(ref version) = self.config.version {
            export_state.set_metadata("export_version", version);
        }

        // Merge config metadata
        for (key, value) in &self.config.metadata {
            export_state.set_metadata(key, value);
        }

        export_state
    }

    /// Write the export manifest.
    fn write_manifest(&self, state: &ModelState) -> Result<()> {
        let manifest = ExportManifest {
            version: self
                .config
                .version
                .clone()
                .unwrap_or_else(|| "1.0.0".to_string()),
            format: format!("{:?}", self.config.format),
            global_step: state.global_step,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            embedding_tables: state.hash_tables.iter().map(|t| t.name.clone()).collect(),
            dense_params: state.dense_params.keys().cloned().collect(),
            includes_optimizer: self.config.include_optimizer,
            metadata: self.config.metadata.clone(),
        };

        let manifest_path = self.config.output_dir.join("manifest.json");
        let json =
            serde_json::to_string_pretty(&manifest).map_err(CheckpointError::Serialization)?;

        std::fs::write(&manifest_path, json).map_err(|e| CheckpointError::Io {
            path: manifest_path,
            source: e,
        })?;

        Ok(())
    }

    /// Load an exported model manifest.
    ///
    /// # Arguments
    ///
    /// * `export_dir` - Directory containing the exported model
    pub fn load_manifest(export_dir: &Path) -> Result<ExportManifest> {
        let manifest_path = export_dir.join("manifest.json");

        if !manifest_path.exists() {
            return Err(CheckpointError::NotFound(manifest_path));
        }

        let json = std::fs::read_to_string(&manifest_path).map_err(|e| CheckpointError::Io {
            path: manifest_path,
            source: e,
        })?;

        serde_json::from_str(&json).map_err(CheckpointError::Deserialization)
    }

    /// Load an exported model's embedding table.
    ///
    /// # Arguments
    ///
    /// * `export_dir` - Directory containing the exported model
    /// * `table_name` - Name of the embedding table to load
    pub fn load_embedding_table(export_dir: &Path, table_name: &str) -> Result<HashTableState> {
        let table_path = export_dir
            .join("embeddings")
            .join(format!("{}.json", table_name));

        if !table_path.exists() {
            return Err(CheckpointError::NotFound(table_path));
        }

        let json = std::fs::read_to_string(&table_path).map_err(|e| CheckpointError::Io {
            path: table_path,
            source: e,
        })?;

        serde_json::from_str(&json).map_err(CheckpointError::Deserialization)
    }

    /// Load exported dense parameters.
    ///
    /// # Arguments
    ///
    /// * `export_dir` - Directory containing the exported model
    pub fn load_dense_params(export_dir: &Path) -> Result<HashMap<String, Vec<f32>>> {
        let params_path = export_dir.join("dense").join("params.json");

        if !params_path.exists() {
            return Err(CheckpointError::NotFound(params_path));
        }

        let json = std::fs::read_to_string(&params_path).map_err(|e| CheckpointError::Io {
            path: params_path,
            source: e,
        })?;

        serde_json::from_str(&json).map_err(CheckpointError::Deserialization)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::OptimizerState;
    use tempfile::tempdir;

    fn create_test_state() -> ModelState {
        let mut state = ModelState::new(5000);

        // Add embedding table
        let mut table = HashTableState::new("user_embeddings", 32);
        table.insert(1, vec![0.1; 32]);
        table.insert(2, vec![0.2; 32]);
        state.add_hash_table(table);

        // Add dense params
        state.add_dense_param("fc1.weight", vec![0.5; 100]);
        state.add_dense_param("fc1.bias", vec![0.0; 10]);

        // Add optimizer
        let opt = OptimizerState::new("adam", "all", 0.001, 5000);
        state.add_optimizer(opt);

        state.set_metadata("model_name", "test_model");

        state
    }

    #[test]
    fn test_export_config_builder() {
        let config = ExportConfig::new("/tmp/export")
            .with_format(ExportFormat::Binary)
            .with_optimizer(true)
            .with_version("2.0.0")
            .with_metadata("author", "test");

        assert_eq!(config.output_dir, PathBuf::from("/tmp/export"));
        assert_eq!(config.format, ExportFormat::Binary);
        assert!(config.include_optimizer);
        assert_eq!(config.version, Some("2.0.0".to_string()));
        assert_eq!(config.metadata.get("author"), Some(&"test".to_string()));
    }

    #[test]
    fn test_export_json() {
        let dir = tempdir().unwrap();
        let config = ExportConfig::new(dir.path()).with_format(ExportFormat::Json);
        let exporter = ModelExporter::new(config);

        let state = create_test_state();
        exporter
            .export(&state)
            .expect("json export should succeed for valid test state");

        // Check files exist
        assert!(dir.path().join("model.json").exists());
        assert!(dir.path().join("manifest.json").exists());
    }

    #[test]
    fn test_export_binary() {
        let dir = tempdir().unwrap();
        let config = ExportConfig::new(dir.path()).with_format(ExportFormat::Binary);
        let exporter = ModelExporter::new(config);

        let state = create_test_state();
        exporter
            .export(&state)
            .expect("binary export should succeed for valid test state");

        assert!(dir.path().join("model.bin").exists());
        assert!(dir.path().join("manifest.json").exists());
    }

    #[test]
    fn test_export_saved_model() {
        let dir = tempdir().unwrap();
        let config = ExportConfig::new(dir.path())
            .with_format(ExportFormat::SavedModel)
            .with_version("1.0.0");
        let exporter = ModelExporter::new(config);

        let state = create_test_state();
        exporter
            .export(&state)
            .expect("saved-model export should succeed for valid test state");

        // Check directory structure
        assert!(dir.path().join("manifest.json").exists());
        assert!(dir.path().join("embeddings").is_dir());
        assert!(dir.path().join("embeddings/user_embeddings.json").exists());
        assert!(dir.path().join("dense").is_dir());
        assert!(dir.path().join("dense/params.json").exists());

        // Optimizer should not be exported by default
        assert!(!dir.path().join("optimizer").exists());

        // Model spec is optional; default tests do not include the required metadata.
        assert!(!dir.path().join(MODEL_SPEC_FILENAME).exists());
    }

    #[test]
    fn test_export_saved_model_writes_model_spec_when_metadata_present() {
        let dir = tempdir().unwrap();
        let config = ExportConfig::new(dir.path()).with_format(ExportFormat::SavedModel);
        let exporter = ModelExporter::new(config);

        let mut state = create_test_state();
        state.set_metadata("model_spec_type", "mlp");
        state.set_metadata("model_input_dim", "4");
        state.set_metadata("model_hidden_dims", "3");
        state.set_metadata("model_output_dim", "1");
        state.set_metadata("model_activation", "relu");

        exporter.export(&state).unwrap();

        let spec_path = dir.path().join(MODEL_SPEC_FILENAME);
        assert!(spec_path.exists());

        let json = std::fs::read_to_string(spec_path).unwrap();
        let spec: ModelSpec = serde_json::from_str(&json).unwrap();
        match spec {
            ModelSpec::Mlp {
                input_dim,
                hidden_dims,
                output_dim,
                activation,
            } => {
                assert_eq!(input_dim, 4);
                assert_eq!(hidden_dims, vec![3]);
                assert_eq!(output_dim, 1);
                assert_eq!(activation, "relu");
            }
            _ => panic!("expected mlp spec"),
        }
    }

    #[test]
    fn test_export_saved_model_with_optimizer() {
        let dir = tempdir().unwrap();
        let config = ExportConfig::new(dir.path())
            .with_format(ExportFormat::SavedModel)
            .with_optimizer(true);
        let exporter = ModelExporter::new(config);

        let state = create_test_state();
        exporter.export(&state).unwrap();

        // Optimizer should be exported
        assert!(dir.path().join("optimizer").is_dir());
        assert!(dir.path().join("optimizer/state.json").exists());
    }

    #[test]
    fn test_load_manifest() {
        let dir = tempdir().unwrap();
        let config = ExportConfig::new(dir.path())
            .with_format(ExportFormat::SavedModel)
            .with_version("2.0.0");
        let exporter = ModelExporter::new(config);

        let state = create_test_state();
        exporter.export(&state).unwrap();

        // Load manifest
        let manifest = ModelExporter::load_manifest(dir.path()).unwrap();
        assert_eq!(manifest.version, "2.0.0");
        assert_eq!(manifest.global_step, 5000);
        assert!(manifest
            .embedding_tables
            .contains(&"user_embeddings".to_string()));
    }

    #[test]
    fn test_load_embedding_table() {
        let dir = tempdir().unwrap();
        let config = ExportConfig::new(dir.path()).with_format(ExportFormat::SavedModel);
        let exporter = ModelExporter::new(config);

        let state = create_test_state();
        exporter.export(&state).unwrap();

        // Load embedding table
        let table = ModelExporter::load_embedding_table(dir.path(), "user_embeddings").unwrap();
        assert_eq!(table.name, "user_embeddings");
        assert_eq!(table.dim, 32);
        assert_eq!(table.len(), 2);
    }

    #[test]
    fn test_load_dense_params() {
        let dir = tempdir().unwrap();
        let config = ExportConfig::new(dir.path()).with_format(ExportFormat::SavedModel);
        let exporter = ModelExporter::new(config);

        let state = create_test_state();
        exporter.export(&state).unwrap();

        // Load dense params
        let params = ModelExporter::load_dense_params(dir.path()).unwrap();
        assert!(params.contains_key("fc1.weight"));
        assert!(params.contains_key("fc1.bias"));
        assert_eq!(params["fc1.weight"].len(), 100);
    }

    #[test]
    fn test_load_manifest_not_found() {
        let dir = tempdir().unwrap();
        let result = ModelExporter::load_manifest(dir.path());
        assert!(matches!(result, Err(CheckpointError::NotFound(_))));
    }

    #[test]
    fn test_export_format_default() {
        assert_eq!(ExportFormat::default(), ExportFormat::SavedModel);
    }
}
