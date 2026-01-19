//! Model loading functionality for serving.
//!
//! This module provides the [`ModelLoader`] struct for loading and managing
//! exported Monolith models for inference serving.

use crate::config::ModelLoaderConfig;
use crate::error::{ServingError, ServingResult};
use crate::inference::{build_model, InferenceModel, ModelSpec};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{debug, info, warn};

/// A loaded model ready for serving.
///
/// This struct holds the model configuration, embeddings, and any other
/// resources needed for inference.
pub struct LoadedModel {
    /// Path where the model was loaded from
    pub path: PathBuf,

    /// Model version identifier
    pub version: String,

    /// Timestamp when the model was loaded
    pub loaded_at: std::time::Instant,

    /// Model metadata
    pub metadata: ModelMetadata,

    /// Optional Candle model spec used by Rust-native serving.
    pub model_spec: Option<ModelSpec>,

    /// Optional Candle inference model.
    pub inference_model: Option<Arc<dyn InferenceModel>>,

    /// Slot configurations for embeddings
    slot_configs: HashMap<i32, SlotConfig>,

    /// Whether the model is ready for serving
    ready: bool,
}

impl std::fmt::Debug for LoadedModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadedModel")
            .field("path", &self.path)
            .field("version", &self.version)
            .field("loaded_at", &self.loaded_at)
            .field("metadata", &self.metadata)
            .field("slot_configs_len", &self.slot_configs.len())
            .field("ready", &self.ready)
            .field("model_spec", &self.model_spec)
            .field("has_inference_model", &self.inference_model.is_some())
            .finish()
    }
}

impl LoadedModel {
    /// Check if the model is ready for serving.
    pub fn is_ready(&self) -> bool {
        self.ready
    }

    /// Get the slot configuration for a given slot ID.
    pub fn get_slot_config(&self, slot_id: i32) -> Option<&SlotConfig> {
        self.slot_configs.get(&slot_id)
    }

    /// Get all slot IDs configured in this model.
    pub fn slot_ids(&self) -> impl Iterator<Item = i32> + '_ {
        self.slot_configs.keys().copied()
    }
}

/// Model metadata loaded from the export.
#[derive(Debug, Clone, Default)]
pub struct ModelMetadata {
    /// Model name
    pub name: String,

    /// Model description
    pub description: String,

    /// Export timestamp
    pub export_timestamp: u64,

    /// Training step at export
    pub global_step: i64,

    /// Custom metadata key-value pairs
    pub custom: HashMap<String, String>,
}

/// Configuration for an embedding slot.
#[derive(Debug, Clone)]
pub struct SlotConfig {
    /// Slot ID
    pub slot_id: i32,

    /// Embedding dimension
    pub dim: usize,

    /// Feature name associated with this slot
    pub feature_name: String,

    /// Whether this slot uses pooling
    pub pooled: bool,
}

/// Model loader for serving infrastructure.
///
/// The `ModelLoader` is responsible for loading exported models from disk,
/// managing model versions, and providing access to loaded models for inference.
///
/// # Example
///
/// ```no_run
/// use monolith_serving::model_loader::ModelLoader;
/// use monolith_serving::config::ModelLoaderConfig;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let config = ModelLoaderConfig::default();
/// let loader = ModelLoader::new(config);
///
/// loader.load("/path/to/model").await?;
///
/// if let Some(model) = loader.current_model() {
///     println!("Loaded model version: {}", model.version);
/// }
/// # Ok(())
/// # }
/// ```
pub struct ModelLoader {
    /// Configuration for model loading
    config: ModelLoaderConfig,

    /// Currently loaded model
    current_model: Arc<RwLock<Option<Arc<LoadedModel>>>>,

    /// Model version history
    version_history: Arc<RwLock<Vec<String>>>,
}

impl ModelLoader {
    /// Create a new model loader with the given configuration.
    pub fn new(config: ModelLoaderConfig) -> Self {
        Self {
            config,
            current_model: Arc::new(RwLock::new(None)),
            version_history: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Load a model from the specified path.
    ///
    /// This method loads model configuration, embeddings, and metadata from
    /// the export directory. If a model is already loaded, it will be replaced.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the exported model directory
    ///
    /// # Errors
    ///
    /// Returns an error if the model cannot be loaded due to missing files,
    /// invalid format, or I/O errors.
    pub async fn load(&self, path: impl AsRef<Path>) -> ServingResult<()> {
        let path = path.as_ref().to_path_buf();
        info!("Loading model from: {:?}", path);

        // Validate path exists
        if !path.exists() {
            return Err(ServingError::ModelLoadError(format!(
                "Model path does not exist: {:?}",
                path
            )));
        }

        // Load model metadata
        let metadata = self.load_metadata(&path).await?;
        debug!("Loaded metadata: {:?}", metadata);

        // Load slot configurations
        let slot_configs = self.load_slot_configs(&path).await?;
        debug!("Loaded {} slot configurations", slot_configs.len());

        // Optional: load Candle model spec + weights for real inference.
        let (model_spec, inference_model) = self
            .try_load_candle_model(&path)
            .await
            .unwrap_or((None, None));

        // Determine version
        let version = self.determine_version(&path, &metadata)?;
        info!("Model version: {}", version);

        // Create loaded model
        let model = LoadedModel {
            path: path.clone(),
            version: version.clone(),
            loaded_at: std::time::Instant::now(),
            metadata,
            model_spec,
            inference_model,
            slot_configs,
            ready: true,
        };

        // Update current model
        {
            let mut current = self.current_model.write();
            *current = Some(Arc::new(model));
        }

        // Update version history
        {
            let mut history = self.version_history.write();
            history.push(version);
        }

        info!("Model loaded successfully from: {:?}", path);
        Ok(())
    }

    /// Get the currently loaded model.
    ///
    /// Returns `None` if no model has been loaded yet.
    pub fn current_model(&self) -> Option<Arc<LoadedModel>> {
        self.current_model.read().clone()
    }

    /// Check if a model is currently loaded and ready.
    pub fn is_ready(&self) -> bool {
        self.current_model
            .read()
            .as_ref()
            .map(|m| m.is_ready())
            .unwrap_or(false)
    }

    /// Reload the current model from disk.
    ///
    /// This is useful when the model files have been updated.
    pub async fn reload(&self) -> ServingResult<()> {
        let path = {
            let current = self.current_model.read();
            current
                .as_ref()
                .map(|m| m.path.clone())
                .ok_or(ServingError::ModelNotLoaded)?
        };

        info!("Reloading model from: {:?}", path);
        self.load(&path).await
    }

    /// Unload the current model, freeing resources.
    pub fn unload(&self) {
        let mut current = self.current_model.write();
        if current.is_some() {
            info!("Unloading current model");
            *current = None;
        }
    }

    /// Get the version history of loaded models.
    pub fn version_history(&self) -> Vec<String> {
        self.version_history.read().clone()
    }

    /// Get the loader configuration.
    pub fn config(&self) -> &ModelLoaderConfig {
        &self.config
    }

    // Private helper methods

    async fn load_metadata(&self, path: &Path) -> ServingResult<ModelMetadata> {
        let metadata_path = path.join("metadata.json");

        if metadata_path.exists() {
            // In a real implementation, we would read and parse the JSON file
            // For now, return default metadata
            debug!("Loading metadata from: {:?}", metadata_path);
            Ok(ModelMetadata {
                name: path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
                    .to_string(),
                ..Default::default()
            })
        } else {
            warn!("No metadata.json found, using defaults");
            Ok(ModelMetadata::default())
        }
    }

    async fn load_slot_configs(&self, path: &Path) -> ServingResult<HashMap<i32, SlotConfig>> {
        let config_path = path.join("slot_config.json");
        let mut configs = HashMap::new();

        if config_path.exists() {
            // In a real implementation, we would read and parse the config file
            debug!("Loading slot configs from: {:?}", config_path);
        } else {
            debug!("No slot_config.json found, using empty configuration");
        }

        // Default configuration for testing
        if configs.is_empty() && self.config.preload_embeddings {
            // Add a default slot for testing purposes
            configs.insert(
                0,
                SlotConfig {
                    slot_id: 0,
                    dim: 64,
                    feature_name: "default".to_string(),
                    pooled: false,
                },
            );
        }

        Ok(configs)
    }

    fn determine_version(&self, path: &Path, metadata: &ModelMetadata) -> ServingResult<String> {
        // Try to get version from metadata first
        if !metadata.name.is_empty() && metadata.global_step > 0 {
            return Ok(format!("{}-step{}", metadata.name, metadata.global_step));
        }

        // Fall back to directory name
        let version = path
            .file_name()
            .and_then(|n| n.to_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| "unknown".to_string());

        Ok(version)
    }

    async fn try_load_candle_model(
        &self,
        path: &Path,
    ) -> ServingResult<(Option<ModelSpec>, Option<Arc<dyn InferenceModel>>)> {
        // We treat model_spec.json as an opt-in for Rust-native serving.
        let spec_path = path.join("model_spec.json");
        let dense_path = path.join("dense").join("params.json");

        if !spec_path.exists() || !dense_path.exists() {
            return Ok((None, None));
        }

        let spec_json = std::fs::read_to_string(&spec_path).map_err(|e| {
            ServingError::ModelLoadError(format!(
                "Failed to read model_spec.json at {:?}: {}",
                spec_path, e
            ))
        })?;
        let spec: ModelSpec = serde_json::from_str(&spec_json).map_err(|e| {
            ServingError::ModelLoadError(format!(
                "Failed to parse model_spec.json at {:?}: {}",
                spec_path, e
            ))
        })?;

        let dense_json = std::fs::read_to_string(&dense_path).map_err(|e| {
            ServingError::ModelLoadError(format!(
                "Failed to read dense params at {:?}: {}",
                dense_path, e
            ))
        })?;
        let dense_params: HashMap<String, Vec<f32>> =
            serde_json::from_str(&dense_json).map_err(|e| {
                ServingError::ModelLoadError(format!(
                    "Failed to parse dense params at {:?}: {}",
                    dense_path, e
                ))
            })?;

        let device = monolith_tensor::CandleTensor::best_device();
        let model = build_model(&spec, &dense_params, &device)?;
        Ok((Some(spec), Some(Arc::from(model))))
    }
}

impl std::fmt::Debug for ModelLoader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelLoader")
            .field("config", &self.config)
            .field("has_model", &self.current_model.read().is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_model_loader_creation() {
        let config = ModelLoaderConfig::default();
        let loader = ModelLoader::new(config);

        assert!(!loader.is_ready());
        assert!(loader.current_model().is_none());
        assert!(loader.version_history().is_empty());
    }

    #[tokio::test]
    async fn test_load_model_from_temp_dir() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("test_model");
        std::fs::create_dir_all(&model_path).unwrap();

        let config = ModelLoaderConfig {
            preload_embeddings: true,
            ..Default::default()
        };
        let loader = ModelLoader::new(config);

        let result = loader.load(&model_path).await;
        assert!(result.is_ok());
        assert!(loader.is_ready());

        let model = loader.current_model().unwrap();
        assert_eq!(model.version, "test_model");
        assert!(model.is_ready());
    }

    #[tokio::test]
    async fn test_load_nonexistent_path() {
        let config = ModelLoaderConfig::default();
        let loader = ModelLoader::new(config);

        let result = loader.load("/nonexistent/path/to/model").await;
        assert!(result.is_err());
        assert!(matches!(result, Err(ServingError::ModelLoadError(_))));
    }

    #[tokio::test]
    async fn test_unload_model() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("test_model");
        std::fs::create_dir_all(&model_path).unwrap();

        let config = ModelLoaderConfig::default();
        let loader = ModelLoader::new(config);

        loader.load(&model_path).await.unwrap();
        assert!(loader.is_ready());

        loader.unload();
        assert!(!loader.is_ready());
        assert!(loader.current_model().is_none());
    }

    #[tokio::test]
    async fn test_version_history() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("model_v1");
        std::fs::create_dir_all(&model_path).unwrap();

        let config = ModelLoaderConfig::default();
        let loader = ModelLoader::new(config);

        loader.load(&model_path).await.unwrap();

        let history = loader.version_history();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0], "model_v1");
    }

    #[test]
    fn test_loaded_model_slot_access() {
        let mut slot_configs = HashMap::new();
        slot_configs.insert(
            1,
            SlotConfig {
                slot_id: 1,
                dim: 32,
                feature_name: "user_id".to_string(),
                pooled: false,
            },
        );
        slot_configs.insert(
            2,
            SlotConfig {
                slot_id: 2,
                dim: 64,
                feature_name: "item_id".to_string(),
                pooled: true,
            },
        );

        let model = LoadedModel {
            path: PathBuf::from("/test"),
            version: "v1".to_string(),
            loaded_at: std::time::Instant::now(),
            metadata: ModelMetadata::default(),
            model_spec: None,
            inference_model: None,
            slot_configs,
            ready: true,
        };

        assert!(model.is_ready());
        assert!(model.get_slot_config(1).is_some());
        assert!(model.get_slot_config(2).is_some());
        assert!(model.get_slot_config(3).is_none());

        let slot_ids: Vec<_> = model.slot_ids().collect();
        assert_eq!(slot_ids.len(), 2);
    }
}
