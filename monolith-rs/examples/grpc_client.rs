//! gRPC Serving Client Example for Monolith-rs
//!
//! This example demonstrates how to create a gRPC client for the Monolith serving
//! infrastructure, including:
//! - Connecting to a gRPC server
//! - Getting model metadata
//! - Running batch predictions
//! - Health checks
//! - Retry logic and error handling
//!
//! # Usage
//!
//! ```bash
//! cargo run --example grpc_client -- --server localhost:50051 --batch-size 10 --num-requests 5
//! ```
//!
//! # Mock Mode
//!
//! If no server is available, use `--mock` flag to simulate responses:
//!
//! ```bash
//! cargo run --example grpc_client -- --mock --batch-size 10 --num-requests 5
//! ```

use clap::Parser;
use monolith_serving::{
    AgentServiceGrpcImpl, FeatureInput, GrpcServerConfig, PredictRequest, PredictResponse,
    ServingError, ServingResult,
};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::sleep;
use tracing::{debug, error, info, warn};

// ============================================================================
// Command Line Arguments
// ============================================================================

/// gRPC client for Monolith serving infrastructure
#[derive(Parser, Debug)]
#[command(name = "grpc_client")]
#[command(about = "Monolith gRPC serving client example", long_about = None)]
struct Args {
    /// Server address in format host:port
    #[arg(short, long, default_value = "localhost:50051")]
    server: String,

    /// Model name to query
    #[arg(short, long, default_value = "default")]
    model_name: String,

    /// Batch size for predictions
    #[arg(short, long, default_value_t = 10)]
    batch_size: usize,

    /// Number of prediction requests to send
    #[arg(short, long, default_value_t = 5)]
    num_requests: usize,

    /// Maximum number of retries for failed requests
    #[arg(long, default_value_t = 3)]
    max_retries: u32,

    /// Timeout for requests in seconds
    #[arg(short, long, default_value_t = 30)]
    timeout: u64,

    /// Run in mock mode (simulate server responses)
    #[arg(long)]
    mock: bool,

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,
}

// ============================================================================
// Model Metadata
// ============================================================================

/// Metadata about a loaded model
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    /// Model name
    pub name: String,
    /// Model version
    pub version: String,
    /// Number of slots configured
    pub num_slots: usize,
    /// Whether the model is ready
    pub ready: bool,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl Default for ModelMetadata {
    fn default() -> Self {
        Self {
            name: "unknown".to_string(),
            version: "1.0".to_string(),
            num_slots: 0,
            ready: false,
            metadata: HashMap::new(),
        }
    }
}

// ============================================================================
// Health Status
// ============================================================================

/// Health status of the serving server
#[derive(Debug, Clone)]
pub struct HealthCheckResult {
    /// Whether the server is healthy
    pub healthy: bool,
    /// Server uptime in seconds
    pub uptime_secs: u64,
    /// Number of active connections
    pub active_connections: u64,
    /// Model is loaded
    pub model_loaded: bool,
    /// Latency of health check in milliseconds
    pub latency_ms: f64,
}

// ============================================================================
// Prediction Statistics
// ============================================================================

/// Statistics from prediction requests
#[derive(Debug, Clone, Default)]
pub struct PredictionStats {
    /// Total requests sent
    pub total_requests: u64,
    /// Successful requests
    pub successful_requests: u64,
    /// Failed requests
    pub failed_requests: u64,
    /// Total latency in milliseconds
    pub total_latency_ms: f64,
    /// Minimum latency
    pub min_latency_ms: f64,
    /// Maximum latency
    pub max_latency_ms: f64,
    /// Total instances processed
    pub total_instances: u64,
}

impl PredictionStats {
    pub fn new() -> Self {
        Self {
            min_latency_ms: f64::MAX,
            max_latency_ms: 0.0,
            ..Default::default()
        }
    }

    pub fn record_success(&mut self, latency_ms: f64, batch_size: usize) {
        self.total_requests += 1;
        self.successful_requests += 1;
        self.total_latency_ms += latency_ms;
        self.total_instances += batch_size as u64;
        self.min_latency_ms = self.min_latency_ms.min(latency_ms);
        self.max_latency_ms = self.max_latency_ms.max(latency_ms);
    }

    pub fn record_failure(&mut self) {
        self.total_requests += 1;
        self.failed_requests += 1;
    }

    pub fn avg_latency_ms(&self) -> f64 {
        if self.successful_requests > 0 {
            self.total_latency_ms / self.successful_requests as f64
        } else {
            0.0
        }
    }

    pub fn qps(&self) -> f64 {
        if self.total_latency_ms > 0.0 {
            self.successful_requests as f64 / (self.total_latency_ms / 1000.0)
        } else {
            0.0
        }
    }
}

// ============================================================================
// Prediction Client
// ============================================================================

/// Client for making predictions against a Monolith serving server.
///
/// Supports both real gRPC connections and mock mode for testing.
pub struct PredictionClient {
    /// Server address
    server_address: String,
    /// Model name to use
    model_name: String,
    /// Request timeout (used in real gRPC mode)
    #[allow(dead_code)]
    timeout: Duration,
    /// Maximum retries
    max_retries: u32,
    /// Whether to use mock mode
    mock_mode: bool,
    /// Mock service (used when mock_mode is true)
    mock_service: Option<Arc<AgentServiceGrpcImpl>>,
    /// Connection state
    connected: bool,
}

impl PredictionClient {
    /// Create a new prediction client.
    ///
    /// # Arguments
    ///
    /// * `server_address` - The server address in format host:port
    /// * `model_name` - The model name to query
    /// * `timeout` - Request timeout duration
    /// * `max_retries` - Maximum number of retries for failed requests
    pub fn new(
        server_address: impl Into<String>,
        model_name: impl Into<String>,
        timeout: Duration,
        max_retries: u32,
    ) -> Self {
        Self {
            server_address: server_address.into(),
            model_name: model_name.into(),
            timeout,
            max_retries,
            mock_mode: false,
            mock_service: None,
            connected: false,
        }
    }

    /// Enable mock mode for testing without a real server.
    pub fn with_mock_mode(mut self) -> Self {
        self.mock_mode = true;
        self
    }

    /// Connect to the gRPC server.
    ///
    /// In mock mode, this creates a mock service instead of connecting.
    pub async fn connect(&mut self) -> ServingResult<()> {
        if self.mock_mode {
            info!("Creating mock service (mock mode enabled)");
            let config = GrpcServerConfig::builder()
                .bind_address(&self.server_address)
                .build();
            self.mock_service = Some(Arc::new(AgentServiceGrpcImpl::new(config)));
            self.connected = true;
            info!("Mock service created successfully");
            return Ok(());
        }

        info!("Connecting to server at {}", self.server_address);

        // In a real implementation, we would use tonic to connect:
        //
        // ```rust
        // let endpoint = tonic::transport::Endpoint::from_shared(
        //     format!("http://{}", self.server_address)
        // )?
        // .timeout(self.timeout)
        // .connect_timeout(Duration::from_secs(10));
        //
        // let channel = endpoint.connect().await?;
        // self.client = Some(AgentServiceClient::new(channel));
        // ```

        // For now, simulate connection with retry logic
        let mut last_error = None;
        for attempt in 1..=self.max_retries {
            debug!("Connection attempt {} of {}", attempt, self.max_retries);

            // Simulate connection attempt
            sleep(Duration::from_millis(100)).await;

            // In mock mode or for demo, always succeed
            // In real mode without actual server, this would fail
            if self.mock_mode {
                self.connected = true;
                info!("Connected to server at {}", self.server_address);
                return Ok(());
            }

            // Simulate connection failure without real server
            last_error = Some(ServingError::server(format!(
                "Cannot connect to {} (no real gRPC server available, use --mock flag)",
                self.server_address
            )));

            if attempt < self.max_retries {
                let backoff = Duration::from_millis(100 * 2u64.pow(attempt - 1));
                warn!(
                    "Connection attempt {} failed, retrying in {:?}",
                    attempt, backoff
                );
                sleep(backoff).await;
            }
        }

        Err(last_error.unwrap_or_else(|| {
            ServingError::server("Connection failed after all retries")
        }))
    }

    /// Check if the client is connected.
    pub fn is_connected(&self) -> bool {
        self.connected
    }

    /// Get model metadata from the server.
    pub async fn get_model_metadata(&self) -> ServingResult<ModelMetadata> {
        self.ensure_connected()?;

        info!("Fetching model metadata for '{}'", self.model_name);

        if let Some(ref service) = self.mock_service {
            // Use mock service to get modules
            let response = service
                .get_modules(monolith_serving::grpc::GetModulesRequest { name_prefix: None })
                .await?;

            let mut metadata = ModelMetadata {
                name: self.model_name.clone(),
                version: "1.0.0".to_string(),
                num_slots: response.modules.len(),
                ready: true,
                metadata: HashMap::new(),
            };

            for module in &response.modules {
                metadata
                    .metadata
                    .insert(module.name.clone(), module.version.clone());
            }

            info!("Retrieved metadata: {:?}", metadata);
            return Ok(metadata);
        }

        // In real implementation, query the server
        // For demo, return mock metadata
        Ok(ModelMetadata {
            name: self.model_name.clone(),
            version: "1.0.0-mock".to_string(),
            num_slots: 10,
            ready: true,
            metadata: HashMap::new(),
        })
    }

    /// Perform a health check on the server.
    pub async fn health_check(&self) -> ServingResult<HealthCheckResult> {
        self.ensure_connected()?;

        let start = Instant::now();
        info!("Performing health check on {}", self.server_address);

        if let Some(ref service) = self.mock_service {
            // Use mock service to get resource info
            let _response = service
                .get_resource(monolith_serving::grpc::GetResourceRequest {})
                .await?;

            let latency = start.elapsed();

            let result = HealthCheckResult {
                healthy: true,
                uptime_secs: service.uptime_secs(),
                active_connections: service.active_connections(),
                model_loaded: true,
                latency_ms: latency.as_secs_f64() * 1000.0,
            };

            info!(
                "Health check passed: latency={:.2}ms, uptime={}s",
                result.latency_ms, result.uptime_secs
            );
            return Ok(result);
        }

        // Mock health check for demo
        let latency = start.elapsed();
        Ok(HealthCheckResult {
            healthy: true,
            uptime_secs: 0,
            active_connections: 0,
            model_loaded: true,
            latency_ms: latency.as_secs_f64() * 1000.0,
        })
    }

    /// Run a batch prediction.
    ///
    /// # Arguments
    ///
    /// * `instances` - Vector of feature inputs for prediction
    ///
    /// # Returns
    ///
    /// Prediction response containing scores and optional embeddings.
    pub async fn predict(&self, instances: Vec<FeatureInput>) -> ServingResult<PredictResponse> {
        self.ensure_connected()?;

        let request_id = format!("req-{}", uuid_v4());
        let num_instances = instances.len();

        debug!(
            "Sending prediction request {} with {} instances",
            request_id, num_instances
        );

        let request = PredictRequest {
            request_id: request_id.clone(),
            features: instances,
            return_embeddings: true,
            context: None,
        };

        // Execute with retry logic
        let mut last_error = None;
        for attempt in 1..=self.max_retries {
            match self.execute_predict(&request).await {
                Ok(response) => {
                    debug!(
                        "Prediction {} completed in {:.2}ms",
                        request_id, response.latency_ms
                    );
                    return Ok(response);
                }
                Err(e) => {
                    warn!(
                        "Prediction attempt {} failed: {}",
                        attempt,
                        e
                    );
                    last_error = Some(e);

                    if attempt < self.max_retries {
                        let backoff = Duration::from_millis(50 * 2u64.pow(attempt - 1));
                        sleep(backoff).await;
                    }
                }
            }
        }

        error!("Prediction {} failed after {} retries", request_id, self.max_retries);
        Err(last_error.unwrap_or_else(|| ServingError::prediction("Unknown error")))
    }

    /// Execute a single prediction request (no retry).
    async fn execute_predict(&self, request: &PredictRequest) -> ServingResult<PredictResponse> {
        if let Some(ref _service) = self.mock_service {
            // Generate mock response
            let start = Instant::now();

            // Simulate processing time proportional to batch size
            let processing_time = Duration::from_micros(100 * request.features.len() as u64);
            sleep(processing_time).await;

            let latency = start.elapsed();

            // Generate mock scores (one per feature batch = one final score)
            let scores: Vec<f32> = (0..1)
                .map(|_| 0.3 + rand_float() * 0.4) // Random scores between 0.3 and 0.7
                .collect();

            return Ok(PredictResponse {
                request_id: request.request_id.clone(),
                scores,
                embeddings: None,
                latency_ms: latency.as_secs_f64() * 1000.0,
                success: true,
                error_message: None,
            });
        }

        // Real gRPC call would go here
        Err(ServingError::server(
            "Real gRPC not implemented in this example",
        ))
    }

    /// Ensure the client is connected.
    fn ensure_connected(&self) -> ServingResult<()> {
        if !self.connected {
            return Err(ServingError::server("Not connected to server"));
        }
        Ok(())
    }
}

// ============================================================================
// Test Instance Generation
// ============================================================================

/// Generate random sparse feature instances for testing.
///
/// # Arguments
///
/// * `batch_size` - Number of instances to generate
/// * `num_features` - Number of features per instance
/// * `max_fids_per_feature` - Maximum feature IDs per feature
pub fn generate_test_instances(
    batch_size: usize,
    num_features: usize,
    max_fids_per_feature: usize,
) -> Vec<FeatureInput> {
    let mut instances = Vec::with_capacity(batch_size * num_features);

    let feature_names = vec![
        "user_id",
        "item_id",
        "category",
        "brand",
        "price_bucket",
        "user_history",
        "context",
        "device_type",
    ];

    for batch_idx in 0..batch_size {
        for (slot_id, feature_name) in feature_names.iter().enumerate().take(num_features) {
            // Generate random FIDs
            let num_fids = 1 + (rand_u64() % max_fids_per_feature as u64) as usize;
            let fids: Vec<i64> = (0..num_fids)
                .map(|_| (rand_u64() % 1_000_000) as i64)
                .collect();

            // Generate optional values (weights)
            let values = if slot_id % 2 == 0 {
                // Some features have weights
                Some(fids.iter().map(|_| rand_float()).collect())
            } else {
                None
            };

            instances.push(FeatureInput {
                name: format!("{}_{}", feature_name, batch_idx),
                slot_id: slot_id as i32,
                fids,
                values,
            });
        }
    }

    instances
}

/// Serialize instances to TensorFlow Example format (simplified).
///
/// In a real implementation, this would use protobuf to create tf.Example.
pub fn serialize_to_example(instances: &[FeatureInput]) -> Vec<u8> {
    // Simplified serialization for demo
    // In production, use prost to serialize tf.Example protos
    let mut buffer = Vec::new();

    for instance in instances {
        // Write slot_id (4 bytes)
        buffer.extend_from_slice(&instance.slot_id.to_le_bytes());

        // Write number of FIDs (4 bytes)
        buffer.extend_from_slice(&(instance.fids.len() as u32).to_le_bytes());

        // Write FIDs (8 bytes each)
        for fid in &instance.fids {
            buffer.extend_from_slice(&fid.to_le_bytes());
        }

        // Write values if present
        if let Some(ref values) = instance.values {
            buffer.push(1); // Has values flag
            for value in values {
                buffer.extend_from_slice(&value.to_le_bytes());
            }
        } else {
            buffer.push(0); // No values flag
        }
    }

    buffer
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Simple UUID v4 generator (for request IDs).
fn uuid_v4() -> String {
    let random_bytes: [u8; 16] = [
        (rand_u64() & 0xFF) as u8,
        ((rand_u64() >> 8) & 0xFF) as u8,
        ((rand_u64() >> 16) & 0xFF) as u8,
        ((rand_u64() >> 24) & 0xFF) as u8,
        ((rand_u64() >> 32) & 0xFF) as u8,
        ((rand_u64() >> 40) & 0xFF) as u8,
        ((rand_u64() >> 48) & 0xFF) as u8,
        ((rand_u64() >> 56) & 0xFF) as u8,
        (rand_u64() & 0xFF) as u8,
        ((rand_u64() >> 8) & 0xFF) as u8,
        ((rand_u64() >> 16) & 0xFF) as u8,
        ((rand_u64() >> 24) & 0xFF) as u8,
        ((rand_u64() >> 32) & 0xFF) as u8,
        ((rand_u64() >> 40) & 0xFF) as u8,
        ((rand_u64() >> 48) & 0xFF) as u8,
        ((rand_u64() >> 56) & 0xFF) as u8,
    ];

    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        random_bytes[0], random_bytes[1], random_bytes[2], random_bytes[3],
        random_bytes[4], random_bytes[5],
        random_bytes[6], random_bytes[7],
        random_bytes[8], random_bytes[9],
        random_bytes[10], random_bytes[11], random_bytes[12], random_bytes[13], random_bytes[14], random_bytes[15]
    )
}

/// Simple random number generator using system time.
fn rand_u64() -> u64 {
    use std::time::SystemTime;
    static mut SEED: u64 = 0;

    unsafe {
        if SEED == 0 {
            SEED = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64;
        }
        // LCG parameters
        SEED = SEED.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        SEED
    }
}

/// Generate random float between 0.0 and 1.0.
fn rand_float() -> f32 {
    (rand_u64() % 1_000_000) as f32 / 1_000_000.0
}

// ============================================================================
// Main Demo Workflow
// ============================================================================

async fn run_demo(args: Args) -> ServingResult<()> {
    // Initialize logging
    if args.verbose {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .init();
    } else {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .init();
    }

    info!("=== Monolith gRPC Client Demo ===");
    info!("Server: {}", args.server);
    info!("Model: {}", args.model_name);
    info!("Batch size: {}", args.batch_size);
    info!("Num requests: {}", args.num_requests);
    info!("Mock mode: {}", args.mock);

    // Step 1: Create client
    let mut client = PredictionClient::new(
        &args.server,
        &args.model_name,
        Duration::from_secs(args.timeout),
        args.max_retries,
    );

    if args.mock {
        client = client.with_mock_mode();
    }

    // Step 2: Connect to server
    info!("\n--- Connecting to Server ---");
    client.connect().await?;
    info!("Connection established successfully");

    // Step 3: Health check
    info!("\n--- Health Check ---");
    let health = client.health_check().await?;
    info!("Server Status:");
    info!("  Healthy: {}", health.healthy);
    info!("  Uptime: {}s", health.uptime_secs);
    info!("  Active Connections: {}", health.active_connections);
    info!("  Model Loaded: {}", health.model_loaded);
    info!("  Health Check Latency: {:.2}ms", health.latency_ms);

    // Step 4: Get model metadata
    info!("\n--- Model Metadata ---");
    let metadata = client.get_model_metadata().await?;
    info!("Model: {}", metadata.name);
    info!("Version: {}", metadata.version);
    info!("Num Slots: {}", metadata.num_slots);
    info!("Ready: {}", metadata.ready);
    if !metadata.metadata.is_empty() {
        info!("Additional Metadata:");
        for (key, value) in &metadata.metadata {
            info!("  {}: {}", key, value);
        }
    }

    // Step 5: Run predictions
    info!("\n--- Running Predictions ---");
    let mut stats = PredictionStats::new();
    let overall_start = Instant::now();

    for i in 0..args.num_requests {
        info!("\nRequest {}/{}:", i + 1, args.num_requests);

        // Generate test instances
        let instances = generate_test_instances(args.batch_size, 4, 5);
        info!("  Generated {} feature instances", instances.len());

        // Serialize for demo
        let serialized = serialize_to_example(&instances);
        info!("  Serialized size: {} bytes", serialized.len());

        // Run prediction
        let start = Instant::now();
        match client.predict(instances).await {
            Ok(response) => {
                let latency = start.elapsed().as_secs_f64() * 1000.0;
                stats.record_success(latency, args.batch_size);

                info!("  Prediction succeeded:");
                info!("    Request ID: {}", response.request_id);
                info!("    Scores: {:?}", response.scores);
                info!("    Latency: {:.2}ms", response.latency_ms);
                info!("    Success: {}", response.success);

                if let Some(ref embeddings) = response.embeddings {
                    info!("    Embeddings returned: {}", embeddings.len());
                }
            }
            Err(e) => {
                stats.record_failure();
                error!("  Prediction failed: {}", e);

                // Check if retriable
                if e.is_retriable() {
                    warn!("  Error is retriable");
                }
            }
        }
    }

    // Step 6: Print statistics
    let total_time = overall_start.elapsed();
    info!("\n--- Final Statistics ---");
    info!("Total Requests: {}", stats.total_requests);
    info!("Successful: {}", stats.successful_requests);
    info!("Failed: {}", stats.failed_requests);
    info!(
        "Success Rate: {:.1}%",
        (stats.successful_requests as f64 / stats.total_requests as f64) * 100.0
    );
    info!("Total Instances: {}", stats.total_instances);
    info!("Total Time: {:.2}s", total_time.as_secs_f64());

    if stats.successful_requests > 0 {
        info!("Latency (avg): {:.2}ms", stats.avg_latency_ms());
        info!("Latency (min): {:.2}ms", stats.min_latency_ms);
        info!("Latency (max): {:.2}ms", stats.max_latency_ms);
        info!("Throughput: {:.1} QPS", stats.qps());
    }

    info!("\n=== Demo Complete ===");
    Ok(())
}

// ============================================================================
// Entry Point
// ============================================================================

#[tokio::main]
async fn main() {
    let args = Args::parse();

    match run_demo(args).await {
        Ok(()) => {
            std::process::exit(0);
        }
        Err(e) => {
            eprintln!("Error: {}", e);

            if e.is_retriable() {
                eprintln!("This error may be transient. Try again or use --mock flag.");
            }

            std::process::exit(1);
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_instances() {
        let instances = generate_test_instances(2, 4, 3);
        assert_eq!(instances.len(), 8); // 2 batches * 4 features

        for instance in &instances {
            assert!(!instance.fids.is_empty());
            assert!(instance.fids.len() <= 3);
        }
    }

    #[test]
    fn test_serialize_to_example() {
        let instances = vec![FeatureInput {
            name: "test".to_string(),
            slot_id: 0,
            fids: vec![1, 2, 3],
            values: Some(vec![1.0, 2.0, 3.0]),
        }];

        let serialized = serialize_to_example(&instances);
        assert!(!serialized.is_empty());
    }

    #[test]
    fn test_prediction_stats() {
        let mut stats = PredictionStats::new();

        stats.record_success(10.0, 5);
        stats.record_success(20.0, 5);
        stats.record_failure();

        assert_eq!(stats.total_requests, 3);
        assert_eq!(stats.successful_requests, 2);
        assert_eq!(stats.failed_requests, 1);
        assert_eq!(stats.total_instances, 10);
        assert!((stats.avg_latency_ms() - 15.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_mock_client() {
        let mut client = PredictionClient::new(
            "localhost:50051",
            "test_model",
            Duration::from_secs(10),
            3,
        )
        .with_mock_mode();

        // Connect
        assert!(client.connect().await.is_ok());
        assert!(client.is_connected());

        // Health check
        let health = client.health_check().await.unwrap();
        assert!(health.healthy);

        // Metadata
        let metadata = client.get_model_metadata().await.unwrap();
        assert_eq!(metadata.name, "test_model");

        // Prediction
        let instances = generate_test_instances(2, 2, 3);
        let response = client.predict(instances).await.unwrap();
        assert!(response.success);
        assert!(!response.scores.is_empty());
    }

    #[test]
    fn test_uuid_generation() {
        let uuid1 = uuid_v4();
        let uuid2 = uuid_v4();

        // UUIDs should be different
        assert_ne!(uuid1, uuid2);

        // Should have correct format (36 chars with hyphens)
        assert_eq!(uuid1.len(), 36);
        assert_eq!(uuid2.len(), 36);
    }

    #[test]
    fn test_model_metadata_default() {
        let metadata = ModelMetadata::default();
        assert_eq!(metadata.name, "unknown");
        assert!(!metadata.ready);
    }
}
