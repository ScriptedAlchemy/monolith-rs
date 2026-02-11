//! Python `monolith.native_training.entry` parity (config/proto builders).
//!
//! The Python side mostly builds protobuf configs for embedding hash tables and provides a few
//! small helper wrappers. In Rust we keep the same surface area where it matters for tests and
//! interoperability, without depending on TensorFlow.

use base64::Engine;
use monolith_proto::monolith::hash_table as pb;
use prost::Message;
use std::fmt;

#[derive(Debug, thiserror::Error)]
pub enum EntryError {
    #[error("init_step_interval should be >= 1, while got {0}")]
    InvalidInitStepInterval(f32),
    #[error("Learning_rate_fns must be not empty.")]
    EmptyLearningRateFns,
}

// -----------------------------------------------------------------------------
// Optimizers
// -----------------------------------------------------------------------------

pub trait Optimizer: Send + Sync {
    fn as_proto(&self) -> pb::OptimizerConfig;
}

pub struct StochasticRoundingFloat16OptimizerWrapper<O> {
    inner: O,
}

impl<O> StochasticRoundingFloat16OptimizerWrapper<O> {
    pub fn new(inner: O) -> Self {
        Self { inner }
    }
}

impl<O: Optimizer> Optimizer for StochasticRoundingFloat16OptimizerWrapper<O> {
    fn as_proto(&self) -> pb::OptimizerConfig {
        let mut p = self.inner.as_proto();
        p.stochastic_rounding_float16 = Some(true);
        p
    }
}

#[derive(Debug, Clone, Default)]
pub struct SgdOptimizer {
    pub learning_rate: Option<f32>,
    pub warmup_steps: Option<i64>,
}

impl SgdOptimizer {
    pub fn new(learning_rate: Option<f32>) -> Self {
        Self {
            learning_rate,
            warmup_steps: None,
        }
    }
}

impl Optimizer for SgdOptimizer {
    fn as_proto(&self) -> pb::OptimizerConfig {
        let mut cfg = pb::SgdOptimizerConfig::default();
        if let Some(v) = self.learning_rate {
            cfg.learning_rate = Some(v);
        }
        if let Some(v) = self.warmup_steps {
            cfg.warmup_steps = Some(v);
        }
        pb::OptimizerConfig {
            r#type: Some(pb::optimizer_config::Type::Sgd(cfg)),
            stochastic_rounding_float16: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AdagradOptimizer {
    pub learning_rate: Option<f32>,
    pub initial_accumulator_value: Option<f32>,
    pub hessian_compression_times: i32,
    pub warmup_steps: i64,
    pub weight_decay_factor: f32,
}

impl Default for AdagradOptimizer {
    fn default() -> Self {
        Self {
            learning_rate: None,
            initial_accumulator_value: None,
            hessian_compression_times: 1,
            warmup_steps: 0,
            weight_decay_factor: 0.0,
        }
    }
}

impl AdagradOptimizer {
    pub fn new(learning_rate: Option<f32>, initial_accumulator_value: Option<f32>) -> Self {
        Self {
            learning_rate,
            initial_accumulator_value,
            ..Default::default()
        }
    }
}

impl Optimizer for AdagradOptimizer {
    fn as_proto(&self) -> pb::OptimizerConfig {
        let mut cfg = pb::AdagradOptimizerConfig::default();
        if let Some(v) = self.learning_rate {
            cfg.learning_rate = Some(v);
        }
        if let Some(v) = self.initial_accumulator_value {
            cfg.initial_accumulator_value = Some(v);
        }
        cfg.hessian_compression_times = Some(self.hessian_compression_times);
        cfg.warmup_steps = Some(self.warmup_steps);
        cfg.weight_decay_factor = Some(self.weight_decay_factor);
        pb::OptimizerConfig {
            r#type: Some(pb::optimizer_config::Type::Adagrad(cfg)),
            stochastic_rounding_float16: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AdadeltaOptimizer {
    pub learning_rate: Option<f32>,
    pub weight_decay_factor: f32,
    pub averaging_ratio: f32,
    pub epsilon: f32,
    pub warmup_steps: i64,
}

impl Default for AdadeltaOptimizer {
    fn default() -> Self {
        Self {
            learning_rate: None,
            weight_decay_factor: 0.0,
            averaging_ratio: 0.9,
            epsilon: 0.01,
            warmup_steps: 0,
        }
    }
}

impl Optimizer for AdadeltaOptimizer {
    fn as_proto(&self) -> pb::OptimizerConfig {
        let mut cfg = pb::AdadeltaOptimizerConfig::default();
        if let Some(v) = self.learning_rate {
            cfg.learning_rate = Some(v);
        }
        cfg.weight_decay_factor = Some(self.weight_decay_factor);
        cfg.averaging_ratio = Some(self.averaging_ratio);
        cfg.epsilon = Some(self.epsilon);
        cfg.warmup_steps = Some(self.warmup_steps);
        pb::OptimizerConfig {
            r#type: Some(pb::optimizer_config::Type::Adadelta(cfg)),
            stochastic_rounding_float16: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AdamOptimizer {
    pub learning_rate: Option<f32>,
    pub beta1: f32,
    pub beta2: f32,
    pub use_beta1_warmup: bool,
    pub weight_decay_factor: f32,
    pub use_nesterov: bool,
    pub epsilon: f32,
    pub warmup_steps: i64,
}

impl Default for AdamOptimizer {
    fn default() -> Self {
        Self {
            learning_rate: None,
            beta1: 0.9,
            beta2: 0.99,
            use_beta1_warmup: false,
            weight_decay_factor: 0.0,
            use_nesterov: false,
            epsilon: 0.01,
            warmup_steps: 0,
        }
    }
}

impl Optimizer for AdamOptimizer {
    fn as_proto(&self) -> pb::OptimizerConfig {
        let mut cfg = pb::AdamOptimizerConfig::default();
        if let Some(v) = self.learning_rate {
            cfg.learning_rate = Some(v);
        }
        cfg.beta1 = Some(self.beta1);
        cfg.beta2 = Some(self.beta2);
        cfg.use_beta1_warmup = Some(self.use_beta1_warmup);
        cfg.weight_decay_factor = Some(self.weight_decay_factor);
        cfg.use_nesterov = Some(self.use_nesterov);
        cfg.epsilon = Some(self.epsilon);
        cfg.warmup_steps = Some(self.warmup_steps);
        pb::OptimizerConfig {
            r#type: Some(pb::optimizer_config::Type::Adam(cfg)),
            stochastic_rounding_float16: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AmsgradOptimizer {
    pub learning_rate: Option<f32>,
    pub beta1: f32,
    pub beta2: f32,
    pub weight_decay_factor: f32,
    pub use_nesterov: bool,
    pub epsilon: f32,
    pub warmup_steps: i64,
}

impl Default for AmsgradOptimizer {
    fn default() -> Self {
        Self {
            learning_rate: None,
            beta1: 0.9,
            beta2: 0.99,
            weight_decay_factor: 0.0,
            use_nesterov: false,
            epsilon: 0.01,
            warmup_steps: 0,
        }
    }
}

impl Optimizer for AmsgradOptimizer {
    fn as_proto(&self) -> pb::OptimizerConfig {
        let mut cfg = pb::AmsgradOptimizerConfig::default();
        if let Some(v) = self.learning_rate {
            cfg.learning_rate = Some(v);
        }
        cfg.beta1 = Some(self.beta1);
        cfg.beta2 = Some(self.beta2);
        cfg.weight_decay_factor = Some(self.weight_decay_factor);
        cfg.use_nesterov = Some(self.use_nesterov);
        cfg.epsilon = Some(self.epsilon);
        cfg.warmup_steps = Some(self.warmup_steps);
        pb::OptimizerConfig {
            r#type: Some(pb::optimizer_config::Type::Amsgrad(cfg)),
            stochastic_rounding_float16: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MomentumOptimizer {
    pub learning_rate: Option<f32>,
    pub weight_decay_factor: f32,
    pub use_nesterov: bool,
    pub momentum: f32,
    pub warmup_steps: i64,
}

impl Default for MomentumOptimizer {
    fn default() -> Self {
        Self {
            learning_rate: None,
            weight_decay_factor: 0.0,
            use_nesterov: false,
            momentum: 0.9,
            warmup_steps: 0,
        }
    }
}

impl Optimizer for MomentumOptimizer {
    fn as_proto(&self) -> pb::OptimizerConfig {
        let mut cfg = pb::MomentumOptimizerConfig::default();
        if let Some(v) = self.learning_rate {
            cfg.learning_rate = Some(v);
        }
        cfg.weight_decay_factor = Some(self.weight_decay_factor);
        cfg.use_nesterov = Some(self.use_nesterov);
        cfg.momentum = Some(self.momentum);
        cfg.warmup_steps = Some(self.warmup_steps);
        pb::OptimizerConfig {
            r#type: Some(pb::optimizer_config::Type::Momentum(cfg)),
            stochastic_rounding_float16: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MovingAverageOptimizer {
    pub momentum: f32,
}

impl Default for MovingAverageOptimizer {
    fn default() -> Self {
        Self { momentum: 0.9 }
    }
}

impl Optimizer for MovingAverageOptimizer {
    fn as_proto(&self) -> pb::OptimizerConfig {
        let mut cfg = pb::MovingAverageOptimizerConfig::default();
        cfg.momentum = Some(self.momentum);
        pb::OptimizerConfig {
            r#type: Some(pb::optimizer_config::Type::MovingAverage(cfg)),
            stochastic_rounding_float16: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RmspropOptimizer {
    pub learning_rate: Option<f32>,
    pub weight_decay_factor: f32,
    pub momentum: f32,
}

impl Default for RmspropOptimizer {
    fn default() -> Self {
        Self {
            learning_rate: None,
            weight_decay_factor: 0.0,
            momentum: 0.9,
        }
    }
}

impl Optimizer for RmspropOptimizer {
    fn as_proto(&self) -> pb::OptimizerConfig {
        let mut cfg = pb::RmspropOptimizerConfig::default();
        if let Some(v) = self.learning_rate {
            cfg.learning_rate = Some(v);
        }
        cfg.weight_decay_factor = Some(self.weight_decay_factor);
        cfg.momentum = Some(self.momentum);
        pb::OptimizerConfig {
            r#type: Some(pb::optimizer_config::Type::Rmsprop(cfg)),
            stochastic_rounding_float16: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RmspropV2Optimizer {
    pub learning_rate: Option<f32>,
    pub weight_decay_factor: f32,
    pub momentum: f32,
}

impl Default for RmspropV2Optimizer {
    fn default() -> Self {
        Self {
            learning_rate: None,
            weight_decay_factor: 0.0,
            momentum: 0.9,
        }
    }
}

impl Optimizer for RmspropV2Optimizer {
    fn as_proto(&self) -> pb::OptimizerConfig {
        let mut cfg = pb::RmspropV2OptimizerConfig::default();
        if let Some(v) = self.learning_rate {
            cfg.learning_rate = Some(v);
        }
        cfg.weight_decay_factor = Some(self.weight_decay_factor);
        cfg.momentum = Some(self.momentum);
        pb::OptimizerConfig {
            r#type: Some(pb::optimizer_config::Type::Rmspropv2(cfg)),
            stochastic_rounding_float16: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BatchSoftmaxOptimizer {
    pub learning_rate: Option<f32>,
}

impl Default for BatchSoftmaxOptimizer {
    fn default() -> Self {
        Self {
            learning_rate: None,
        }
    }
}

impl Optimizer for BatchSoftmaxOptimizer {
    fn as_proto(&self) -> pb::OptimizerConfig {
        let mut cfg = pb::BatchSoftmaxOptimizerConfig::default();
        if let Some(v) = self.learning_rate {
            cfg.learning_rate = Some(v);
        }
        pb::OptimizerConfig {
            r#type: Some(pb::optimizer_config::Type::BatchSoftmax(cfg)),
            stochastic_rounding_float16: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FtrlOptimizer {
    pub learning_rate: Option<f32>,
    pub initial_accumulator_value: Option<f32>,
    pub beta: Option<f32>,
    pub warmup_steps: i64,
    pub l1_regularization_strength: Option<f32>,
    pub l2_regularization_strength: Option<f32>,
}

impl Default for FtrlOptimizer {
    fn default() -> Self {
        Self {
            learning_rate: None,
            initial_accumulator_value: None,
            beta: None,
            warmup_steps: 0,
            l1_regularization_strength: None,
            l2_regularization_strength: None,
        }
    }
}

impl Optimizer for FtrlOptimizer {
    fn as_proto(&self) -> pb::OptimizerConfig {
        let mut cfg = pb::FtrlOptimizerConfig::default();
        if let Some(v) = self.learning_rate {
            cfg.learning_rate = Some(v);
        }
        if let Some(v) = self.beta {
            cfg.beta = Some(v);
        }
        if let Some(v) = self.initial_accumulator_value {
            cfg.initial_accumulator_value = Some(v);
        }
        if let Some(v) = self.l1_regularization_strength {
            cfg.l1_regularization_strength = Some(v);
        }
        if let Some(v) = self.l2_regularization_strength {
            cfg.l2_regularization_strength = Some(v);
        }
        cfg.warmup_steps = Some(self.warmup_steps);
        pb::OptimizerConfig {
            r#type: Some(pb::optimizer_config::Type::Ftrl(cfg)),
            stochastic_rounding_float16: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DynamicWdAdagradOptimizer {
    pub learning_rate: Option<f32>,
    pub initial_accumulator_value: Option<f32>,
    pub hessian_compression_times: i32,
    pub warmup_steps: i64,
    pub weight_decay_factor: f32,
    pub decouple_weight_decay: bool,
    pub enable_dynamic_wd: bool,
    pub flip_direction: bool,
    pub dynamic_wd_temperature: f32,
}

impl Default for DynamicWdAdagradOptimizer {
    fn default() -> Self {
        Self {
            learning_rate: None,
            initial_accumulator_value: None,
            hessian_compression_times: 1,
            warmup_steps: 0,
            weight_decay_factor: 0.0,
            decouple_weight_decay: true,
            enable_dynamic_wd: true,
            flip_direction: true,
            dynamic_wd_temperature: 1.0,
        }
    }
}

impl Optimizer for DynamicWdAdagradOptimizer {
    fn as_proto(&self) -> pb::OptimizerConfig {
        let mut cfg = pb::DynamicWdAdagradOptimizerConfig::default();
        if let Some(v) = self.learning_rate {
            cfg.learning_rate = Some(v);
        }
        if let Some(v) = self.initial_accumulator_value {
            cfg.initial_accumulator_value = Some(v);
        }
        cfg.hessian_compression_times = Some(self.hessian_compression_times);
        cfg.warmup_steps = Some(self.warmup_steps);
        cfg.weight_decay_factor = Some(self.weight_decay_factor);
        cfg.decouple_weight_decay = Some(self.decouple_weight_decay);
        cfg.enable_dynamic_wd = Some(self.enable_dynamic_wd);
        cfg.flip_direction = Some(self.flip_direction);
        cfg.dynamic_wd_temperature = Some(self.dynamic_wd_temperature);
        pb::OptimizerConfig {
            r#type: Some(pb::optimizer_config::Type::DynamicWdAdagrad(cfg)),
            stochastic_rounding_float16: None,
        }
    }
}

// -----------------------------------------------------------------------------
// Initializers
// -----------------------------------------------------------------------------

pub trait Initializer: Send + Sync {
    fn as_proto(&self) -> pb::InitializerConfig;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ZerosInitializer;

impl Initializer for ZerosInitializer {
    fn as_proto(&self) -> pb::InitializerConfig {
        pb::InitializerConfig {
            r#type: Some(pb::initializer_config::Type::Zeros(
                pb::ZerosInitializerConfig::default(),
            )),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ConstantsInitializer {
    pub constant: f32,
}

impl ConstantsInitializer {
    pub fn new(constant: f32) -> Self {
        Self { constant }
    }
}

impl Initializer for ConstantsInitializer {
    fn as_proto(&self) -> pb::InitializerConfig {
        let mut cfg = pb::ConstantsInitializerConfig::default();
        cfg.constant = Some(self.constant);
        pb::InitializerConfig {
            r#type: Some(pb::initializer_config::Type::Constants(cfg)),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RandomUniformInitializer {
    pub minval: Option<f32>,
    pub maxval: Option<f32>,
}

impl RandomUniformInitializer {
    pub fn new(minval: Option<f32>, maxval: Option<f32>) -> Self {
        Self { minval, maxval }
    }
}

impl Initializer for RandomUniformInitializer {
    fn as_proto(&self) -> pb::InitializerConfig {
        let mut cfg = pb::RandomUniformInitializerConfig::default();
        if let Some(v) = self.minval {
            cfg.minval = Some(v);
        }
        if let Some(v) = self.maxval {
            cfg.maxval = Some(v);
        }
        pb::InitializerConfig {
            r#type: Some(pb::initializer_config::Type::RandomUniform(cfg)),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BatchSoftmaxInitializer {
    constant: f32,
}

impl BatchSoftmaxInitializer {
    pub fn new(init_step_interval: f32) -> Result<Self, EntryError> {
        if init_step_interval < 1.0 {
            return Err(EntryError::InvalidInitStepInterval(init_step_interval));
        }
        Ok(Self {
            constant: init_step_interval,
        })
    }
}

impl Initializer for BatchSoftmaxInitializer {
    fn as_proto(&self) -> pb::InitializerConfig {
        let mut cfg = pb::ConstantsInitializerConfig::default();
        cfg.constant = Some(self.constant);
        pb::InitializerConfig {
            r#type: Some(pb::initializer_config::Type::Constants(cfg)),
        }
    }
}

// -----------------------------------------------------------------------------
// Compressors
// -----------------------------------------------------------------------------

pub trait Compressor: Send + Sync {
    fn as_proto(&self) -> pb::FloatCompressorConfig;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Fp16Compressor;

impl Compressor for Fp16Compressor {
    fn as_proto(&self) -> pb::FloatCompressorConfig {
        pb::FloatCompressorConfig {
            r#type: Some(pb::float_compressor_config::Type::Fp16(
                pb::float_compressor_config::Fp16::default(),
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Fp32Compressor;

impl Compressor for Fp32Compressor {
    fn as_proto(&self) -> pb::FloatCompressorConfig {
        pb::FloatCompressorConfig {
            r#type: Some(pb::float_compressor_config::Type::Fp32(
                pb::float_compressor_config::Fp32::default(),
            )),
        }
    }
}

#[derive(Debug, Clone)]
pub struct FixedR8Compressor {
    pub r: f32,
}

impl Default for FixedR8Compressor {
    fn default() -> Self {
        Self { r: 1.0 }
    }
}

impl Compressor for FixedR8Compressor {
    fn as_proto(&self) -> pb::FloatCompressorConfig {
        let mut cfg = pb::float_compressor_config::FixedR8::default();
        cfg.r = Some(self.r);
        pb::FloatCompressorConfig {
            r#type: Some(pb::float_compressor_config::Type::FixedR8(cfg)),
        }
    }
}

#[derive(Debug, Clone)]
pub struct OneBitCompressor {
    pub step_size: i64,
    pub amplitude: f32,
}

impl Default for OneBitCompressor {
    fn default() -> Self {
        Self {
            step_size: 200,
            amplitude: 0.05,
        }
    }
}

impl Compressor for OneBitCompressor {
    fn as_proto(&self) -> pb::FloatCompressorConfig {
        let mut cfg = pb::float_compressor_config::OneBit::default();
        cfg.step_size = Some(self.step_size);
        cfg.amplitude = Some(self.amplitude);
        pb::FloatCompressorConfig {
            r#type: Some(pb::float_compressor_config::Type::OneBit(cfg)),
        }
    }
}

// -----------------------------------------------------------------------------
// Helper traits for CombineAsSegment
// -----------------------------------------------------------------------------

pub trait ToInitializerProto {
    fn to_initializer_proto(&self) -> pb::InitializerConfig;
}
impl<T: Initializer> ToInitializerProto for T {
    fn to_initializer_proto(&self) -> pb::InitializerConfig {
        self.as_proto()
    }
}
impl ToInitializerProto for pb::InitializerConfig {
    fn to_initializer_proto(&self) -> pb::InitializerConfig {
        self.clone()
    }
}

pub trait ToOptimizerProto {
    fn to_optimizer_proto(&self) -> pb::OptimizerConfig;
}
impl<T: Optimizer> ToOptimizerProto for T {
    fn to_optimizer_proto(&self) -> pb::OptimizerConfig {
        self.as_proto()
    }
}
impl ToOptimizerProto for pb::OptimizerConfig {
    fn to_optimizer_proto(&self) -> pb::OptimizerConfig {
        self.clone()
    }
}

pub trait ToCompressorProto {
    fn to_compressor_proto(&self) -> pb::FloatCompressorConfig;
}
impl<T: Compressor> ToCompressorProto for T {
    fn to_compressor_proto(&self) -> pb::FloatCompressorConfig {
        self.as_proto()
    }
}
impl ToCompressorProto for pb::FloatCompressorConfig {
    fn to_compressor_proto(&self) -> pb::FloatCompressorConfig {
        self.clone()
    }
}

pub fn combine_as_segment<I: ToInitializerProto, O: ToOptimizerProto, C: ToCompressorProto>(
    dim_size: i32,
    initializer: I,
    optimizer: O,
    compressor: C,
) -> pb::entry_config::Segment {
    pb::entry_config::Segment {
        init_config: Some(initializer.to_initializer_proto()),
        opt_config: Some(optimizer.to_optimizer_proto()),
        comp_config: Some(compressor.to_compressor_proto()),
        dim_size: Some(dim_size),
    }
}

// -----------------------------------------------------------------------------
// Hash table config + learning rate functions (TF-free)
// -----------------------------------------------------------------------------

pub trait HashTableConfig: Send + Sync {
    fn mutate_table(&self, table_config: &mut pb::EmbeddingHashTableConfig);
}

#[derive(Debug, Clone)]
pub struct CuckooHashTableConfig {
    pub initial_capacity: u64,
    pub feature_evict_every_n_hours: i32,
}

impl Default for CuckooHashTableConfig {
    fn default() -> Self {
        Self {
            initial_capacity: 1,
            feature_evict_every_n_hours: 0,
        }
    }
}

impl HashTableConfig for CuckooHashTableConfig {
    fn mutate_table(&self, table_config: &mut pb::EmbeddingHashTableConfig) {
        table_config.initial_capacity = Some(self.initial_capacity);
        table_config.r#type = Some(pb::embedding_hash_table_config::Type::Cuckoo(
            pb::CuckooEmbeddingHashTableConfig::default(),
        ));
        if self.feature_evict_every_n_hours > 0 {
            table_config.enable_feature_eviction = Some(true);
            table_config.feature_evict_every_n_hours = Some(self.feature_evict_every_n_hours);
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum LearningRateFn {
    Constant(f32),
    PolynomialDecay {
        initial_learning_rate: f32,
        decay_steps: u64,
        end_learning_rate: f32,
    },
}

impl fmt::Display for LearningRateFn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LearningRateFn::Constant(v) => write!(f, "{v}"),
            LearningRateFn::PolynomialDecay {
                initial_learning_rate,
                decay_steps,
                end_learning_rate,
            } => write!(
                f,
                "PolynomialDecay(initial_learning_rate={initial_learning_rate}, decay_steps={decay_steps}, end_learning_rate={end_learning_rate})"
            ),
        }
    }
}

#[derive(Debug, Clone)]
pub struct HashTableConfigInstance {
    table_config: pb::EmbeddingHashTableConfig,
    pub extra_restore_names: Vec<String>,
    learning_rate_fns: Vec<LearningRateFn>,
    learning_rate_tensor: Option<Vec<f32>>,
}

impl HashTableConfigInstance {
    pub fn new(
        table_config: pb::EmbeddingHashTableConfig,
        learning_rate_fns: Vec<LearningRateFn>,
        extra_restore_names: Option<Vec<String>>,
    ) -> Self {
        Self {
            table_config,
            extra_restore_names: extra_restore_names.unwrap_or_default(),
            learning_rate_fns,
            learning_rate_tensor: None,
        }
    }

    pub fn table_config(&self) -> &pb::EmbeddingHashTableConfig {
        &self.table_config
    }

    pub fn learning_rate_fns(&self) -> &[LearningRateFn] {
        &self.learning_rate_fns
    }

    pub fn learning_rate_tensor(&self) -> Option<&[f32]> {
        self.learning_rate_tensor.as_deref()
    }

    pub fn set_learning_rate_tensor(&mut self, lr: Vec<f32>) {
        self.learning_rate_tensor = Some(lr);
    }

    pub fn call_learning_rate_fns(&self) -> Result<Vec<f32>, EntryError> {
        if self.learning_rate_fns.is_empty() {
            return Err(EntryError::EmptyLearningRateFns);
        }
        Ok(self
            .learning_rate_fns
            .iter()
            .map(|lr| match lr {
                LearningRateFn::Constant(v) => *v,
                LearningRateFn::PolynomialDecay {
                    initial_learning_rate,
                    ..
                } => *initial_learning_rate,
            })
            .collect())
    }

    pub fn call_learning_rate_fns_fewer_ops(&self) -> Result<Vec<f32>, EntryError> {
        self.call_learning_rate_fns()
    }
}

impl fmt::Display for HashTableConfigInstance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let bytes = self.table_config.encode_to_vec();
        let pb_b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
        let fns = self
            .learning_rate_fns
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        write!(f, "TableConfigPB: {pb_b64}, LearningRateFns: [{fns}]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimizers_as_proto() {
        SgdOptimizer::new(Some(0.01)).as_proto();
        AdagradOptimizer::new(Some(0.01), Some(0.1)).as_proto();
        let mut ada = AdagradOptimizer::new(Some(0.01), Some(0.1));
        ada.hessian_compression_times = 10;
        ada.as_proto();
        let mut ftrl = FtrlOptimizer::default();
        ftrl.learning_rate = Some(0.01);
        ftrl.initial_accumulator_value = Some(0.1);
        ftrl.beta = Some(1.0);
        ftrl.as_proto();
        let mut dyn_wd = DynamicWdAdagradOptimizer::default();
        dyn_wd.learning_rate = Some(0.01);
        dyn_wd.initial_accumulator_value = Some(0.1);
        dyn_wd.hessian_compression_times = 1;
        dyn_wd.as_proto();
        let mut adadelta = AdadeltaOptimizer::default();
        adadelta.learning_rate = Some(0.01);
        adadelta.as_proto();
        let mut adam = AdamOptimizer::default();
        adam.learning_rate = Some(0.01);
        adam.as_proto();
        AmsgradOptimizer::default().as_proto();
        MomentumOptimizer::default().as_proto();
        MovingAverageOptimizer::default().as_proto();
        RmspropOptimizer::default().as_proto();
        RmspropV2Optimizer::default().as_proto();
        BatchSoftmaxOptimizer::default().as_proto();

        // Wrapper flips the flag.
        let wrapped = StochasticRoundingFloat16OptimizerWrapper::new(SgdOptimizer::new(Some(0.1)));
        let p = wrapped.as_proto();
        assert_eq!(p.stochastic_rounding_float16, Some(true));
    }

    #[test]
    fn test_initializers_as_proto() {
        ZerosInitializer.as_proto();
        RandomUniformInitializer::new(Some(-0.5), Some(0.5)).as_proto();
        BatchSoftmaxInitializer::new(1.0).unwrap().as_proto();
    }

    #[test]
    fn test_compressors_as_proto() {
        Fp16Compressor.as_proto();
        Fp32Compressor.as_proto();
        FixedR8Compressor::default().as_proto();
        OneBitCompressor::default().as_proto();
    }

    #[test]
    fn test_combine_as_segment() {
        let seg = combine_as_segment(5, ZerosInitializer, SgdOptimizer::new(None), Fp16Compressor);
        assert_eq!(seg.dim_size, Some(5));
        assert!(seg.init_config.is_some());
        assert!(seg.opt_config.is_some());
        assert!(seg.comp_config.is_some());
    }

    #[test]
    fn test_hashtable_config() {
        let cfg = CuckooHashTableConfig::default();
        let mut table = pb::EmbeddingHashTableConfig::default();
        cfg.mutate_table(&mut table);
        assert_eq!(table.initial_capacity, Some(1));
        assert!(matches!(
            table.r#type,
            Some(pb::embedding_hash_table_config::Type::Cuckoo(_))
        ));
    }

    #[test]
    fn test_hashtable_config_instance_str() {
        let table_config1 = pb::EmbeddingHashTableConfig::default();
        let config1 =
            HashTableConfigInstance::new(table_config1, vec![LearningRateFn::Constant(0.1)], None);

        let table_config2 = pb::EmbeddingHashTableConfig::default();
        let config2 =
            HashTableConfigInstance::new(table_config2, vec![LearningRateFn::Constant(0.1)], None);
        assert_eq!(config1.to_string(), config2.to_string());

        let table_config3 = pb::EmbeddingHashTableConfig::default();
        let config3 = HashTableConfigInstance::new(
            table_config3,
            vec![LearningRateFn::PolynomialDecay {
                initial_learning_rate: 0.01,
                decay_steps: 20,
                end_learning_rate: 0.05,
            }],
            None,
        );
        let table_config4 = pb::EmbeddingHashTableConfig::default();
        let config4 = HashTableConfigInstance::new(
            table_config4,
            vec![LearningRateFn::PolynomialDecay {
                initial_learning_rate: 0.01,
                decay_steps: 20,
                end_learning_rate: 0.05,
            }],
            None,
        );
        assert_eq!(config3.to_string(), config4.to_string());
        assert_ne!(config1.to_string(), config3.to_string());
    }
}
