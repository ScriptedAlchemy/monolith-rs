//! Agent service implementation for handling prediction requests.
//!
//! This module provides the gRPC agent service implementation that handles
//! prediction requests, embedding lookups, and model inference.

use crate::error::{ServingError, ServingResult};
use crate::model_loader::{LoadedModel, ModelLoader};
use crate::parameter_sync::ParameterSyncClient;
use candle_core::Tensor as CandleTensor;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info, warn};

/// Request for a prediction.
#[derive(Debug, Clone)]
pub struct PredictRequest {
    /// Request ID for tracing
    pub request_id: String,

    /// Feature inputs for prediction
    pub features: Vec<FeatureInput>,

    /// Whether to return embeddings in the response
    pub return_embeddings: bool,

    /// Optional context for the request
    pub context: Option<RequestContext>,
}

/// Feature input for prediction.
#[derive(Debug, Clone)]
pub struct FeatureInput {
    /// Feature name
    pub name: String,

    /// Slot ID for this feature
    pub slot_id: i32,

    /// Feature IDs (fids)
    pub fids: Vec<i64>,

    /// Optional feature values (weights)
    pub values: Option<Vec<f32>>,
}

/// Optional context for a prediction request.
#[derive(Debug, Clone, Default)]
pub struct RequestContext {
    /// User ID for personalization
    pub user_id: Option<i64>,

    /// Session ID for session-based features
    pub session_id: Option<String>,

    /// Timestamp of the request
    pub timestamp: Option<u64>,

    /// Custom context key-value pairs
    pub custom: HashMap<String, String>,
}

/// Response from a prediction request.
#[derive(Debug, Clone)]
pub struct PredictResponse {
    /// Request ID echoed back
    pub request_id: String,

    /// Prediction scores
    pub scores: Vec<f32>,

    /// Optional embeddings if requested
    pub embeddings: Option<Vec<EmbeddingOutput>>,

    /// Latency information
    pub latency_ms: f64,

    /// Whether the prediction was successful
    pub success: bool,

    /// Error message if prediction failed
    pub error_message: Option<String>,
}

/// Embedding output in response.
#[derive(Debug, Clone)]
pub struct EmbeddingOutput {
    /// Slot ID
    pub slot_id: i32,

    /// Feature name
    pub name: String,

    /// Pooled embedding vector
    pub embedding: Vec<f32>,
}

/// Statistics for the agent service.
#[derive(Debug, Clone, Default)]
pub struct ServiceStats {
    /// Total number of requests received
    pub total_requests: u64,

    /// Successful requests
    pub successful_requests: u64,

    /// Failed requests
    pub failed_requests: u64,

    /// Total embedding lookups
    pub embedding_lookups: u64,

    /// Cache hits for embeddings
    pub cache_hits: u64,

    /// Average latency in milliseconds
    pub avg_latency_ms: f64,

    /// P99 latency in milliseconds
    pub p99_latency_ms: f64,
}

/// Agent service implementation for handling predictions.
///
/// The `AgentServiceImpl` provides the core prediction functionality,
/// including embedding lookups, feature processing, and model inference.
///
/// # Example
///
/// ```no_run
/// use monolith_serving::agent_service::{AgentServiceImpl, PredictRequest, FeatureInput};
/// use monolith_serving::model_loader::ModelLoader;
/// use monolith_serving::config::ModelLoaderConfig;
/// use std::sync::Arc;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let model_loader = Arc::new(ModelLoader::new(ModelLoaderConfig::default()));
/// let service = AgentServiceImpl::new(model_loader, None);
///
/// let request = PredictRequest {
///     request_id: "req-001".to_string(),
///     features: vec![
///         FeatureInput {
///             name: "user_id".to_string(),
///             slot_id: 0,
///             fids: vec![12345],
///             values: None,
///         },
///     ],
///     return_embeddings: false,
///     context: None,
/// };
///
/// let response = service.predict(request).await?;
/// println!("Prediction score: {:?}", response.scores);
/// # Ok(())
/// # }
/// ```
type EmbeddingCache = HashMap<i32, HashMap<i64, Vec<f32>>>;

/// Main agent-service implementation for prediction requests.
pub struct AgentServiceImpl {
    /// Model loader for accessing loaded models
    model_loader: Arc<ModelLoader>,

    /// Optional parameter sync client for embedding updates
    param_sync: Option<Arc<ParameterSyncClient>>,

    /// Embedding cache (slot_id -> fid -> embedding)
    embedding_cache: Arc<RwLock<EmbeddingCache>>,

    /// Service statistics
    stats: Arc<RwLock<ServiceStats>>,

    /// Maximum batch size for embedding lookups
    max_batch_size: usize,
}

impl AgentServiceImpl {
    /// Create a new agent service instance.
    ///
    /// # Arguments
    ///
    /// * `model_loader` - The model loader for accessing loaded models
    /// * `param_sync` - Optional parameter sync client for embedding updates
    pub fn new(
        model_loader: Arc<ModelLoader>,
        param_sync: Option<Arc<ParameterSyncClient>>,
    ) -> Self {
        Self {
            model_loader,
            param_sync,
            embedding_cache: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(ServiceStats::default())),
            max_batch_size: 1024,
        }
    }

    /// Set the maximum batch size for embedding lookups.
    pub fn with_max_batch_size(mut self, size: usize) -> Self {
        self.max_batch_size = size;
        self
    }

    /// Handle a prediction request.
    ///
    /// This method processes the input features, looks up embeddings,
    /// runs model inference, and returns prediction scores.
    ///
    /// # Arguments
    ///
    /// * `request` - The prediction request
    ///
    /// # Returns
    ///
    /// A prediction response containing scores and optionally embeddings.
    pub async fn predict(&self, request: PredictRequest) -> ServingResult<PredictResponse> {
        let start = Instant::now();
        debug!("Processing prediction request: {}", request.request_id);

        // Update request count
        {
            self.stats.write().total_requests += 1;
        }

        // Check if model is loaded
        let model = self.model_loader.current_model().ok_or_else(|| {
            self.stats.write().failed_requests += 1;
            ServingError::ModelNotLoaded
        })?;

        // Look up embeddings for all features
        let embeddings = self.lookup_embeddings(&request.features, &model).await?;

        // Run inference (simplified - in reality this would run the model)
        let scores = self.run_inference(&model, &embeddings)?;

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        // Prepare embedding output if requested
        let embedding_output = if request.return_embeddings {
            Some(
                embeddings
                    .iter()
                    .map(|(slot_id, name, emb)| EmbeddingOutput {
                        slot_id: *slot_id,
                        name: name.clone(),
                        embedding: emb.clone(),
                    })
                    .collect(),
            )
        } else {
            None
        };

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.successful_requests += 1;

            // Update average latency
            let total = stats.successful_requests as f64;
            stats.avg_latency_ms = (stats.avg_latency_ms * (total - 1.0) + latency_ms) / total;
        }

        debug!(
            "Prediction completed for {} in {:.2}ms",
            request.request_id, latency_ms
        );

        Ok(PredictResponse {
            request_id: request.request_id,
            scores,
            embeddings: embedding_output,
            latency_ms,
            success: true,
            error_message: None,
        })
    }

    /// Look up embeddings for the given features.
    async fn lookup_embeddings(
        &self,
        features: &[FeatureInput],
        model: &LoadedModel,
    ) -> ServingResult<Vec<(i32, String, Vec<f32>)>> {
        let mut results = Vec::with_capacity(features.len());

        for feature in features {
            let slot_config = model.get_slot_config(feature.slot_id);
            let dim = slot_config.map(|c| c.dim).unwrap_or(64);

            // Try cache first
            let cached = self.get_cached_embeddings(feature.slot_id, &feature.fids);

            let embeddings = if cached.len() == feature.fids.len() {
                // All embeddings found in cache
                self.stats.write().cache_hits += 1;
                cached
            } else {
                // Need to fetch from param sync or use defaults
                self.stats.write().embedding_lookups += 1;
                self.fetch_embeddings(feature.slot_id, &feature.fids, dim)
                    .await?
            };

            // Pool embeddings (mean pooling)
            let pooled = self.pool_embeddings(&embeddings, &feature.values);

            results.push((feature.slot_id, feature.name.clone(), pooled));
        }

        Ok(results)
    }

    /// Get embeddings from the cache.
    fn get_cached_embeddings(&self, slot_id: i32, fids: &[i64]) -> Vec<Vec<f32>> {
        let cache = self.embedding_cache.read();

        if let Some(slot_cache) = cache.get(&slot_id) {
            fids.iter()
                .filter_map(|fid| slot_cache.get(fid).cloned())
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Fetch embeddings from parameter sync or generate defaults.
    async fn fetch_embeddings(
        &self,
        slot_id: i32,
        fids: &[i64],
        dim: usize,
    ) -> ServingResult<Vec<Vec<f32>>> {
        // Try to fetch from parameter sync if available
        if let Some(ref param_sync) = self.param_sync {
            if param_sync.is_connected() {
                let data = param_sync.pull(slot_id, fids).await?;

                // Cache the fetched embeddings
                let mut cache = self.embedding_cache.write();
                let slot_cache = cache.entry(slot_id).or_default();

                for emb_data in &data {
                    slot_cache.insert(emb_data.fid, emb_data.embedding.clone());
                }

                return Ok(data.into_iter().map(|d| d.embedding).collect());
            }
        }

        // Fall back to default embeddings (zeros)
        warn!(
            "Using default embeddings for slot {} ({} fids)",
            slot_id,
            fids.len()
        );
        Ok(fids.iter().map(|_| vec![0.0; dim]).collect())
    }

    /// Pool multiple embeddings into a single vector.
    fn pool_embeddings(&self, embeddings: &[Vec<f32>], weights: &Option<Vec<f32>>) -> Vec<f32> {
        if embeddings.is_empty() {
            return Vec::new();
        }

        let dim = embeddings[0].len();
        let mut pooled = vec![0.0; dim];

        match weights {
            Some(w) if w.len() == embeddings.len() => {
                // Weighted mean pooling
                let weight_sum: f32 = w.iter().sum();
                for (emb, &weight) in embeddings.iter().zip(w.iter()) {
                    for (i, &val) in emb.iter().enumerate() {
                        pooled[i] += val * weight / weight_sum;
                    }
                }
            }
            _ => {
                // Simple mean pooling
                let n = embeddings.len() as f32;
                for emb in embeddings {
                    for (i, &val) in emb.iter().enumerate() {
                        pooled[i] += val / n;
                    }
                }
            }
        }

        pooled
    }

    /// Run model inference on the embeddings.
    fn run_inference(
        &self,
        model: &LoadedModel,
        embeddings: &[(i32, String, Vec<f32>)],
    ) -> ServingResult<Vec<f32>> {
        // If the loaded model contains a Candle inference graph (via model_spec.json),
        // use it. Otherwise, fall back to a deterministic baseline score.
        let Some(infer) = model.inference_model.as_ref() else {
            // Baseline: same as previous behavior (for compatibility).
            if embeddings.is_empty() {
                return Ok(vec![0.5]);
            }
            let concat: Vec<f32> = embeddings
                .iter()
                .flat_map(|(_, _, emb)| emb.iter().cloned())
                .collect();
            let score = if concat.is_empty() {
                0.5
            } else {
                let mean: f32 = concat.iter().sum::<f32>() / concat.len() as f32;
                1.0 / (1.0 + (-mean).exp())
            };
            return Ok(vec![score]);
        };

        // Build input vector from pooled embeddings: concat slot embeddings in request order.
        let mut x: Vec<f32> = Vec::new();
        for (_, _, emb) in embeddings {
            x.extend_from_slice(emb);
        }

        if x.len() != infer.input_dim() {
            return Err(ServingError::PredictionError(format!(
                "Inference input dim mismatch: got {}, expected {}",
                x.len(),
                infer.input_dim()
            )));
        }

        let device = monolith_tensor::CandleTensor::best_device();
        let input = CandleTensor::from_slice(&x, (1, x.len()), &device).map_err(|e| {
            ServingError::PredictionError(format!("Candle input tensor failed: {e}"))
        })?;

        let out = infer.predict(&input)?;
        let vec = out.to_vec2::<f32>().map_err(|e| {
            ServingError::PredictionError(format!("Candle output decode failed: {e}"))
        })?;

        // Primary head: first row. If multi-dim, return all.
        Ok(vec.into_iter().next().unwrap_or_default())
    }

    /// Get service statistics.
    pub fn stats(&self) -> ServiceStats {
        self.stats.read().clone()
    }

    /// Clear the embedding cache.
    pub fn clear_cache(&self) {
        self.embedding_cache.write().clear();
        info!("Embedding cache cleared");
    }

    /// Get cache statistics.
    pub fn cache_size(&self) -> usize {
        self.embedding_cache
            .read()
            .values()
            .map(|slot| slot.len())
            .sum()
    }

    /// Warm up the cache by preloading embeddings.
    pub async fn warmup(&self, slot_ids: &[i32]) -> ServingResult<()> {
        info!("Warming up cache for {} slots", slot_ids.len());

        for &slot_id in slot_ids {
            // In a real implementation, we would load common embeddings
            debug!("Warming up slot {}", slot_id);
        }

        info!("Cache warmup completed");
        Ok(())
    }

    /// Check if the service is ready to handle requests.
    pub fn is_ready(&self) -> bool {
        self.model_loader.is_ready()
    }
}

impl std::fmt::Debug for AgentServiceImpl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentServiceImpl")
            .field("model_ready", &self.model_loader.is_ready())
            .field("cache_size", &self.cache_size())
            .field("has_param_sync", &self.param_sync.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelLoaderConfig;
    use tempfile::tempdir;

    async fn create_test_service() -> AgentServiceImpl {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("test_model");
        std::fs::create_dir_all(&model_path).unwrap();

        let loader = Arc::new(ModelLoader::new(ModelLoaderConfig::default()));
        loader.load(&model_path).await.unwrap();

        // Keep the tempdir alive by leaking it (for tests only)
        std::mem::forget(dir);

        AgentServiceImpl::new(loader, None)
    }

    #[tokio::test]
    async fn test_service_creation() {
        let loader = Arc::new(ModelLoader::new(ModelLoaderConfig::default()));
        let service = AgentServiceImpl::new(loader, None);

        assert!(!service.is_ready());
        assert_eq!(service.cache_size(), 0);
    }

    #[tokio::test]
    async fn test_predict_no_model() {
        let loader = Arc::new(ModelLoader::new(ModelLoaderConfig::default()));
        let service = AgentServiceImpl::new(loader, None);

        let request = PredictRequest {
            request_id: "test-001".to_string(),
            features: vec![],
            return_embeddings: false,
            context: None,
        };

        let result = service.predict(request).await;
        assert!(result.is_err());
        assert!(matches!(result, Err(ServingError::ModelNotLoaded)));
    }

    #[tokio::test]
    async fn test_predict_with_model() {
        let service = create_test_service().await;

        let request = PredictRequest {
            request_id: "test-001".to_string(),
            features: vec![FeatureInput {
                name: "user_id".to_string(),
                slot_id: 0,
                fids: vec![1, 2, 3],
                values: None,
            }],
            return_embeddings: false,
            context: None,
        };

        let response = service
            .predict(request)
            .await
            .expect("predict should succeed with loaded test model");
        assert_eq!(response.request_id, "test-001");
        assert!(response.success);
        assert!(!response.scores.is_empty());
        assert!(response.embeddings.is_none());
    }

    #[tokio::test]
    async fn test_predict_with_embeddings() {
        let service = create_test_service().await;

        let request = PredictRequest {
            request_id: "test-002".to_string(),
            features: vec![FeatureInput {
                name: "item_id".to_string(),
                slot_id: 1,
                fids: vec![100, 200],
                values: Some(vec![1.0, 2.0]),
            }],
            return_embeddings: true,
            context: None,
        };

        let response = service
            .predict(request)
            .await
            .expect("predict with embeddings should succeed with loaded test model");
        assert!(response.embeddings.is_some());

        let embeddings = response.embeddings.unwrap();
        assert_eq!(embeddings.len(), 1);
        assert_eq!(embeddings[0].slot_id, 1);
        assert_eq!(embeddings[0].name, "item_id");
    }

    #[tokio::test]
    async fn test_stats_tracking() {
        let service = create_test_service().await;

        // Make several requests
        for i in 0..5 {
            let request = PredictRequest {
                request_id: format!("test-{}", i),
                features: vec![],
                return_embeddings: false,
                context: None,
            };

            let _ = service.predict(request).await;
        }

        let stats = service.stats();
        assert_eq!(stats.total_requests, 5);
        assert_eq!(stats.successful_requests, 5);
        assert_eq!(stats.failed_requests, 0);
    }

    #[tokio::test]
    async fn test_cache_operations() {
        let service = create_test_service().await;

        // Initial cache should be empty
        assert_eq!(service.cache_size(), 0);

        // Make a request that populates cache
        let request = PredictRequest {
            request_id: "test-cache".to_string(),
            features: vec![FeatureInput {
                name: "feature".to_string(),
                slot_id: 0,
                fids: vec![1, 2, 3],
                values: None,
            }],
            return_embeddings: false,
            context: None,
        };

        service.predict(request).await.unwrap();

        // Clear cache
        service.clear_cache();
        assert_eq!(service.cache_size(), 0);
    }

    #[test]
    fn test_pool_embeddings_mean() {
        let loader = Arc::new(ModelLoader::new(ModelLoaderConfig::default()));
        let service = AgentServiceImpl::new(loader, None);

        let embeddings = vec![vec![1.0, 2.0, 3.0], vec![2.0, 4.0, 6.0]];

        let pooled = service.pool_embeddings(&embeddings, &None);

        assert_eq!(pooled.len(), 3);
        assert!((pooled[0] - 1.5).abs() < 1e-6);
        assert!((pooled[1] - 3.0).abs() < 1e-6);
        assert!((pooled[2] - 4.5).abs() < 1e-6);
    }

    #[test]
    fn test_pool_embeddings_weighted() {
        let loader = Arc::new(ModelLoader::new(ModelLoaderConfig::default()));
        let service = AgentServiceImpl::new(loader, None);

        let embeddings = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let weights = Some(vec![1.0, 3.0]); // Second embedding has 3x weight

        let pooled = service.pool_embeddings(&embeddings, &weights);

        assert_eq!(pooled.len(), 2);
        // Expected: (1.0 * 1 + 3.0 * 3) / 4 = 10 / 4 = 2.5
        assert!((pooled[0] - 2.5).abs() < 1e-6);
        // Expected: (2.0 * 1 + 4.0 * 3) / 4 = 14 / 4 = 3.5
        assert!((pooled[1] - 3.5).abs() < 1e-6);
    }

    #[test]
    fn test_feature_input() {
        let input = FeatureInput {
            name: "test_feature".to_string(),
            slot_id: 5,
            fids: vec![1, 2, 3],
            values: Some(vec![0.5, 0.3, 0.2]),
        };

        assert_eq!(input.name, "test_feature");
        assert_eq!(input.slot_id, 5);
        assert_eq!(input.fids.len(), 3);
        assert_eq!(input.values.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_request_context() {
        let mut context = RequestContext::default();
        context.user_id = Some(12345);
        context.session_id = Some("session-abc".to_string());
        context
            .custom
            .insert("key".to_string(), "value".to_string());

        assert_eq!(context.user_id, Some(12345));
        assert_eq!(context.session_id, Some("session-abc".to_string()));
        assert_eq!(context.custom.get("key"), Some(&"value".to_string()));
    }
}
