//! Neural network layers for Monolith.
//!
//! This crate provides a collection of neural network layers commonly used in
//! recommendation systems and deep learning applications. It includes:
//!
//! - **Dense layers**: Fully connected linear transformations
//! - **MLP**: Multi-layer perceptrons with configurable architectures
//! - **MMoE**: Multi-gate Mixture of Experts for multi-task learning
//! - **DCN**: Deep & Cross Network layers for feature interactions
//! - **DIN**: Deep Interest Network attention for user behavior sequences
//! - **AGRU**: Attention GRU for sequential user behavior modeling
//! - **DIEN**: Deep Interest Evolution Network for interest evolution modeling
//! - **SENet**: Squeeze-and-Excitation Network for feature importance
//! - **FFM**: Field-aware Factorization Machines for field-aware interactions
//! - **Group Interaction**: Grouped feature interactions (intra/inter-group)
//! - **Normalization**: Layer normalization and batch normalization
//! - **Embeddings**: Hash-based embedding lookup for sparse features
//! - **Activations**: Common activation functions (ReLU, Sigmoid, Tanh, GELU)
//!
//! # Quick Start
//!
//! ```
//! use monolith_layers::prelude::*;
//!
//! // Create a simple MLP
//! let mlp = MLPConfig::new(128)
//!     .add_layer(64, ActivationType::relu())
//!     .add_layer(32, ActivationType::relu())
//!     .add_layer(1, ActivationType::None)
//!     .build()
//!     .unwrap();
//!
//! // Forward pass
//! let input = Tensor::rand(&[32, 128]);  // batch of 32
//! let output = mlp.forward(&input).unwrap();
//! ```
//!
//! # Layer Trait
//!
//! All layers implement the [`Layer`] trait, which provides a unified interface
//! for forward and backward passes:
//!
//! ```
//! use monolith_layers::prelude::*;
//!
//! fn process_layer<L: Layer>(layer: &L, input: &Tensor) -> Tensor {
//!     layer.forward(input).unwrap()
//! }
//! ```
//!
//! # Embedding Lookup
//!
//! For sparse categorical features, use the embedding lookup layer:
//!
//! ```
//! use monolith_layers::embedding::{EmbeddingLookup, EmbeddingHashTable};
//!
//! let mut table = EmbeddingHashTable::new(64);  // 64-dim embeddings
//! table.insert(1, vec![0.1; 64]);
//! table.insert(2, vec![0.2; 64]);
//!
//! let lookup = EmbeddingLookup::new(table);
//! let embeddings = lookup.lookup(&[1, 2, 1]);  // Look up by feature IDs
//! ```
//!
//! # Sequential Models with AGRU
//!
//! For modeling user behavior sequences:
//!
//! ```
//! use monolith_layers::agru::AGRU;
//! use monolith_layers::tensor::Tensor;
//!
//! let agru = AGRU::new(32, 64);  // input_dim=32, hidden_dim=64
//! let sequence = Tensor::rand(&[8, 10, 32]);  // batch=8, seq_len=10
//! let attention = Tensor::ones(&[8, 10]);     // attention weights
//! let hidden = agru.forward_with_attention(&sequence, &attention).unwrap();
//! ```

#![warn(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]

pub mod activation;
pub mod activation_layer;
pub mod add_bias;
pub mod agru;
pub mod constraint;
pub mod dcn;
pub mod dense;
pub mod dien;
pub mod din;
pub mod dmr;
pub mod embedding;
pub mod error;
pub mod feature_cross;
pub mod feature_trans;
pub mod ffm;
pub mod group_interaction;
pub mod initializer;
pub mod layer;
pub mod lhuc;
pub mod logit_correction;
pub mod merge;
pub mod mixed_emb_op_comb_nws;
pub mod mlp;
pub mod mmoe;
pub mod normalization;
pub mod pooling;
pub mod regularizer;
pub mod senet;
pub mod snr;
pub mod tensor;

// Re-export main types at crate level
pub use activation::{
    Exponential, HardSigmoid, LeakyReLU, Linear, Mish, PReLU, ReLU, Sigmoid, Sigmoid2, Softmax,
    Softplus, Softsign, Swish, Tanh, ThresholdedReLU, ELU, GELU, SELU,
};
pub use activation_layer::ActivationLayer;
pub use add_bias::{AddBias, DataFormat};
pub use agru::{AGRUConfig, AGRU};
pub use constraint::Constraint;
pub use dcn::{CrossLayer, CrossNetwork, DCNConfig, DCNMode};
pub use dense::Dense;
pub use dien::{AUGRUCell, DIENConfig, DIENLayer, GRUCell, GRUType};
pub use din::{DINAttention, DINConfig, DINOutputMode};
pub use dmr::{DMRU2IConfig, DMRU2I};
pub use embedding::{
    EmbeddingHashTable, EmbeddingLookup, PooledEmbeddingLookup, PoolingMode,
    SequenceEmbeddingLookup,
};
pub use error::{LayerError, LayerResult};
pub use feature_cross::{AllInt, CDot, GroupInt, GroupIntType, CAN, CIN};
pub use feature_trans::{AutoInt, AutoIntConfig, IRazor, IRazorConfig};
pub use ffm::{FFMConfig, FFMLayer};
pub use group_interaction::{
    GroupInteractionConfig, GroupInteractionLayer, GroupInteractionWithProjection, InteractionType,
};
pub use initializer::Initializer;
pub use layer::Layer;
pub use lhuc::{LHUCConfig, LHUCOutputDims, LHUCOverrides, LHUCTower};
pub use logit_correction::LogitCorrection;
pub use merge::{merge_tensor_list, merge_tensor_list_tensor, MergeOutput, MergeType};
pub use mixed_emb_op_comb_nws::MixedEmbedOpCombNws;
pub use mlp::{ActivationType, MLPConfig, MLP};
pub use mmoe::{Expert, Gate, GateType, MMoE, MMoEConfig};
pub use normalization::{BatchNorm, GradNorm, LayerNorm};
pub use pooling::{AvgPooling, MaxPooling, Pooling, SumPooling};
pub use regularizer::Regularizer;
pub use senet::{SENetConfig, SENetLayer};
pub use snr::{SNRConfig, SNRType, SNR};
pub use tensor::Tensor;

/// Prelude module for convenient imports.
///
/// Import everything commonly needed with:
/// ```
/// use monolith_layers::prelude::*;
/// ```
pub mod prelude {
    pub use crate::activation::{
        Exponential, HardSigmoid, LeakyReLU, Linear, Mish, PReLU, ReLU, Sigmoid, Sigmoid2, Softmax,
        Softplus, Softsign, Swish, Tanh, ThresholdedReLU, ELU, GELU, SELU,
    };
    pub use crate::activation_layer::ActivationLayer;
    pub use crate::add_bias::{AddBias, DataFormat};
    pub use crate::agru::{AGRUConfig, AGRU};
    pub use crate::constraint::Constraint;
    pub use crate::dcn::{CrossLayer, CrossNetwork, DCNConfig, DCNMode};
    pub use crate::dense::Dense;
    pub use crate::dien::{AUGRUCell, DIENConfig, DIENLayer, GRUCell, GRUType};
    pub use crate::din::{DINAttention, DINConfig, DINOutputMode};
    pub use crate::dmr::{DMRU2IConfig, DMRU2I};
    pub use crate::embedding::{
        EmbeddingHashTable, EmbeddingLookup, PooledEmbeddingLookup, PoolingMode,
    };
    pub use crate::error::{LayerError, LayerResult};
    pub use crate::feature_cross::{AllInt, CDot, GroupInt, GroupIntType, CAN, CIN};
    pub use crate::feature_trans::{AutoInt, AutoIntConfig, IRazor, IRazorConfig};
    pub use crate::ffm::{FFMConfig, FFMLayer};
    pub use crate::group_interaction::{
        GroupInteractionConfig, GroupInteractionLayer, GroupInteractionWithProjection,
        InteractionType,
    };
    pub use crate::initializer::Initializer;
    pub use crate::layer::Layer;
    pub use crate::lhuc::{LHUCConfig, LHUCOutputDims, LHUCOverrides, LHUCTower};
    pub use crate::logit_correction::LogitCorrection;
    pub use crate::merge::{merge_tensor_list, merge_tensor_list_tensor, MergeOutput, MergeType};
    pub use crate::mixed_emb_op_comb_nws::MixedEmbedOpCombNws;
    pub use crate::mlp::{ActivationType, MLPConfig, MLP};
    pub use crate::mmoe::{Expert, Gate, GateType, MMoE, MMoEConfig};
    pub use crate::normalization::{BatchNorm, GradNorm, LayerNorm};
    pub use crate::pooling::{AvgPooling, MaxPooling, Pooling, SumPooling};
    pub use crate::regularizer::Regularizer;
    pub use crate::senet::{SENetConfig, SENetLayer};
    pub use crate::snr::{SNRConfig, SNRType, SNR};
    pub use crate::tensor::Tensor;
}

#[cfg(test)]
mod tests {
    use super::prelude::*;

    #[test]
    fn test_prelude_imports() {
        // Test that all types are accessible through prelude
        let _tensor = Tensor::zeros(&[2, 2]);
        let _dense = Dense::new(10, 5);
        let _relu = ReLU::new();
        let _ln = LayerNorm::new(10);
    }

    #[test]
    fn test_layer_composition() {
        // Test composing multiple layers
        let dense = Dense::new(10, 5);
        let relu = ReLU::new();

        let input = Tensor::rand(&[3, 10]);
        let h = dense.forward(&input).unwrap();
        let output = relu.forward(&h).unwrap();

        assert_eq!(output.shape(), &[3, 5]);
    }

    #[test]
    fn test_mlp_end_to_end() {
        let mlp = MLPConfig::new(10)
            .add_layer(8, ActivationType::relu())
            .add_layer(4, ActivationType::relu())
            .add_layer(2, ActivationType::None)
            .build()
            .unwrap();

        let input = Tensor::rand(&[5, 10]);
        let output = mlp.forward(&input).unwrap();

        assert_eq!(output.shape(), &[5, 2]);
    }

    #[test]
    fn test_embedding_end_to_end() {
        let mut table = EmbeddingHashTable::new(8);
        table.insert(100, vec![1.0; 8]);
        table.insert(200, vec![2.0; 8]);

        let lookup = EmbeddingLookup::new(table);
        let ids = Tensor::from_data(&[3], vec![100.0, 200.0, 100.0]);
        let embeddings = lookup.forward(&ids).unwrap();

        assert_eq!(embeddings.shape(), &[3, 8]);
    }

    #[test]
    fn test_normalization() {
        let ln = LayerNorm::new(8);
        let input = Tensor::rand(&[4, 8]);
        let output = ln.forward(&input).unwrap();

        assert_eq!(output.shape(), &[4, 8]);
    }

    #[test]
    fn test_agru_integration() {
        let agru = AGRU::new(8, 16);
        let input = Tensor::rand(&[2, 5, 8]);
        let attention = Tensor::ones(&[2, 5]);

        let output = agru.forward_with_attention(&input, &attention).unwrap();
        assert_eq!(output.shape(), &[2, 16]);
    }

    #[test]
    fn test_senet_integration() {
        let senet = SENetConfig::new(32)
            .with_reduction_ratio(4)
            .with_bias(true)
            .build()
            .unwrap();

        let input = Tensor::rand(&[8, 32]);
        let output = senet.forward(&input).unwrap();

        // SENet preserves input shape
        assert_eq!(output.shape(), &[8, 32]);
    }

    #[test]
    fn test_dien_integration() {
        let dien = DIENConfig::new(8, 8)
            .with_gru_type(GRUType::AUGRU)
            .with_use_auxiliary_loss(true)
            .build()
            .unwrap();

        let behavior_seq = Tensor::rand(&[2, 5, 8]);
        let target_item = Tensor::rand(&[2, 8]);

        let output = dien
            .forward_dien(&behavior_seq, &target_item, None)
            .unwrap();
        assert_eq!(output.shape(), &[2, 8]);
    }

    #[test]
    fn test_ffm_integration() {
        // Create FFM layer with 3 fields and 4-dimensional embeddings
        let ffm = FFMLayer::new(3, 4);

        // Batch of 2 samples, 3 features each
        let field_indices = Tensor::from_data(&[2, 3], vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0]);
        let field_values = Tensor::ones(&[2, 3]);

        let output = ffm
            .forward_with_fields(&field_indices, &field_values)
            .unwrap();
        assert_eq!(output.shape(), &[2, 1]);
    }

    #[test]
    fn test_group_interaction_integration() {
        // Create group interaction layer: 2 groups, 3 features each, embedding dim 4
        let config = GroupInteractionConfig::new(2, 3, 4)
            .with_inter_group(true)
            .with_intra_group(false);

        let layer = GroupInteractionLayer::from_config(&config);

        // Input: [batch_size, num_groups * features_per_group * embedding_dim]
        let input = Tensor::rand(&[2, 24]); // 2 * 3 * 4 = 24
        let output = layer.forward(&input).unwrap();

        let expected_dim = config.output_dim();
        assert_eq!(output.shape(), &[2, expected_dim]);
    }

    #[test]
    fn test_group_interaction_with_projection_integration() {
        let config = GroupInteractionConfig::new(2, 2, 4)
            .with_interaction_type(InteractionType::InnerProduct)
            .with_inter_group(true);

        let layer = GroupInteractionWithProjection::new(&config, 16);

        let input = Tensor::rand(&[2, 16]); // 2 * 2 * 4 = 16
        let output = layer.forward(&input).unwrap();

        assert_eq!(output.shape(), &[2, 16]);
    }
}
