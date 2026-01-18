//! Neural Network Layers Demo for Monolith-RS
//!
//! This example demonstrates all layer types available in the monolith-layers crate:
//!
//! 1. **Basic Layers**: Dense, MLP, LayerNorm, BatchNorm
//! 2. **Recommendation Layers**: DIN, DCN, MMoE, FFM, SENet, GroupInteraction
//! 3. **Sequence Layers**: AGRU, DIEN
//! 4. **Complete Model**: A recommendation model combining multiple layer types
//!
//! Run with: cargo run --example layers_demo

use monolith_layers::{
    // Basic layers
    dense::Dense,
    mlp::{MLP, MLPConfig, ActivationType},
    normalization::{LayerNorm, BatchNorm},

    // Recommendation layers
    din::{DINAttention, DINConfig},
    dcn::{CrossNetwork, DCNConfig, DCNMode},
    mmoe::{MMoE, MMoEConfig},
    ffm::{FFMLayer, FFMConfig},
    senet::{SENetLayer, SENetConfig},
    group_interaction::{GroupInteractionLayer, GroupInteractionConfig, InteractionType},

    // Sequence layers
    agru::AGRU,
    dien::{DIENLayer, DIENConfig, GRUType},

    // Embedding
    embedding::{EmbeddingHashTable, EmbeddingLookup, PooledEmbeddingLookup, PoolingMode},

    // Core traits and types
    layer::Layer,
    tensor::Tensor,
};

fn main() {
    println!("=============================================================");
    println!("         Monolith-RS Neural Network Layers Demo");
    println!("=============================================================\n");

    // Run all demos
    basic_layers_demo();
    recommendation_layers_demo();
    sequence_layers_demo();
    complete_model_demo();
}

// ============================================================================
// PART 1: BASIC LAYERS DEMO
// ============================================================================

fn basic_layers_demo() {
    println!("-------------------------------------------------------------");
    println!("  PART 1: Basic Layers Demo");
    println!("-------------------------------------------------------------\n");

    dense_layer_demo();
    mlp_demo();
    layer_norm_demo();
    batch_norm_demo();
}

/// Demonstrates the Dense (fully connected) layer.
///
/// Dense layer performs: y = xW + b
/// - x: input [batch_size, in_features]
/// - W: weights [in_features, out_features]
/// - b: bias [out_features]
/// - y: output [batch_size, out_features]
fn dense_layer_demo() {
    println!("--- Dense Layer ---\n");

    // Configuration: 64 input features -> 32 output features
    let mut dense = Dense::new(64, 32);
    println!("Created Dense layer: {} -> {} features", dense.in_features(), dense.out_features());
    println!("  Weights shape: {:?}", dense.weights().shape());
    println!("  Bias shape: {:?}", dense.bias().shape());

    // Forward pass
    // Input shape: [batch_size=4, in_features=64]
    let input = Tensor::rand(&[4, 64]);
    println!("\nInput shape: {:?}", input.shape());

    let output = dense.forward(&input).expect("Forward pass failed");
    println!("Output shape: {:?}  (batch=4, out_features=32)", output.shape());

    // Training forward (caches input for backward)
    let output_train = dense.forward_train(&input).expect("Forward train failed");
    println!("Training output shape: {:?}", output_train.shape());

    // Backward pass
    // Gradient has same shape as output
    let grad = Tensor::ones(&[4, 32]);
    let input_grad = dense.backward(&grad).expect("Backward pass failed");
    println!("Input gradient shape: {:?}", input_grad.shape());
    println!("Weight gradient available: {}", dense.weights_grad().is_some());
    println!("Bias gradient available: {}", dense.bias_grad().is_some());

    // Parameter access
    let params = dense.parameters();
    println!("\nTotal parameters: {} tensors", params.len());
    let total_params: usize = params.iter().map(|t| t.numel()).sum();
    println!("Total parameter count: {} (64*32 weights + 32 bias = {})",
             total_params, 64*32 + 32);

    // Dense without bias
    let dense_no_bias = Dense::new_no_bias(64, 32);
    println!("\nDense without bias: {} parameters",
             dense_no_bias.parameters().len());

    println!();
}

/// Demonstrates the MLP (Multi-Layer Perceptron).
///
/// MLP stacks multiple Dense layers with activation functions.
/// Architecture: input -> [Dense -> Activation] x N -> output
fn mlp_demo() {
    println!("--- Multi-Layer Perceptron (MLP) ---\n");

    // Method 1: Using MLPConfig builder pattern
    let config = MLPConfig::new(128)
        .add_layer(64, ActivationType::ReLU)   // 128 -> 64 with ReLU
        .add_layer(32, ActivationType::ReLU)   // 64 -> 32 with ReLU
        .add_layer(10, ActivationType::None);  // 32 -> 10 (no activation for output)

    let mlp1 = MLP::from_config(config).expect("MLP creation failed");
    println!("Created MLP via config:");
    println!("  Input dim: {}", mlp1.input_dim());
    println!("  Output dim: {}", mlp1.output_dim());
    println!("  Number of layers: {}", mlp1.num_layers());

    // Method 2: Using MLP::new with uniform hidden layers
    // Creates: input -> hidden1 -> hidden2 -> output
    let mlp2 = MLP::new(128, &[64, 32], 10, ActivationType::ReLU)
        .expect("MLP creation failed");
    println!("\nCreated MLP via new():");
    println!("  Number of layers: {}", mlp2.num_layers());

    // Forward pass
    // Input: [batch=8, features=128]
    let input = Tensor::rand(&[8, 128]);
    println!("\nInput shape: {:?}", input.shape());

    let output = mlp1.forward(&input).expect("Forward failed");
    println!("Output shape: {:?}  (128 -> 64 -> 32 -> 10)", output.shape());

    // Training with backward pass
    let mut mlp_train = MLP::new(128, &[64], 10, ActivationType::ReLU).unwrap();
    let _ = mlp_train.forward_train(&input).expect("Forward train failed");
    let grad = Tensor::ones(&[8, 10]);
    let input_grad = mlp_train.backward(&grad).expect("Backward failed");
    println!("Input gradient shape: {:?}", input_grad.shape());

    // Parameter count
    let params = mlp1.parameters();
    println!("\nMLP parameters: {} tensors", params.len());
    let total: usize = params.iter().map(|t| t.numel()).sum();
    println!("Total parameters: {}", total);

    // Different activation types
    println!("\nSupported activations:");
    for activation in [
        ActivationType::ReLU,
        ActivationType::Sigmoid,
        ActivationType::Tanh,
        ActivationType::GELU,
        ActivationType::None,
    ] {
        println!("  - {:?}", activation);
    }

    println!();
}

/// Demonstrates Layer Normalization.
///
/// LayerNorm normalizes across the feature dimension:
/// y = (x - mean) / sqrt(var + eps) * gamma + beta
///
/// Unlike BatchNorm, it normalizes each sample independently.
fn layer_norm_demo() {
    println!("--- Layer Normalization ---\n");

    // Create LayerNorm for 64-dimensional features
    let mut layer_norm = LayerNorm::new(64);
    println!("Created LayerNorm:");
    println!("  Normalized shape: {}", layer_norm.normalized_shape());
    println!("  Gamma (scale) shape: {:?}", layer_norm.gamma().shape());
    println!("  Beta (shift) shape: {:?}", layer_norm.beta().shape());

    // Custom epsilon for numerical stability
    let layer_norm_eps = LayerNorm::with_eps(64, 1e-6);
    println!("  Custom eps LayerNorm created");
    let _ = layer_norm_eps; // suppress warning

    // Forward pass
    // Input: [batch=4, features=64]
    let input = Tensor::rand(&[4, 64]);
    println!("\nInput shape: {:?}", input.shape());
    println!("Input sample mean (before norm): {:.4}",
             input.data()[0..64].iter().sum::<f32>() / 64.0);

    let output = layer_norm.forward(&input).expect("Forward failed");
    println!("Output shape: {:?}", output.shape());

    // Verify normalization - each row should have ~0 mean
    let row_mean: f32 = output.data()[0..64].iter().sum::<f32>() / 64.0;
    println!("Output sample mean (after norm): {:.4} (should be ~0)", row_mean);

    // Training mode forward + backward
    let output_train = layer_norm.forward_train(&input).expect("Forward train failed");
    let grad = Tensor::ones(&[4, 64]);
    let input_grad = layer_norm.backward(&grad).expect("Backward failed");
    println!("Input gradient shape: {:?}", input_grad.shape());

    // Parameters
    println!("\nLayerNorm has {} learnable parameters (gamma, beta)",
             layer_norm.parameters().len());

    println!();
}

/// Demonstrates Batch Normalization.
///
/// BatchNorm normalizes across the batch dimension:
/// - Training: uses batch statistics
/// - Inference: uses running statistics
fn batch_norm_demo() {
    println!("--- Batch Normalization ---\n");

    // Create BatchNorm for 64 features
    let mut batch_norm = BatchNorm::new(64);
    println!("Created BatchNorm:");
    println!("  Num features: {}", batch_norm.num_features());
    println!("  Training mode: {}", batch_norm.is_training());

    // Custom momentum and epsilon
    let batch_norm_custom = BatchNorm::with_params(64, 0.1, 1e-5);
    let _ = batch_norm_custom; // suppress warning

    // Forward pass
    // Input: [batch=8, features=64] - needs larger batch for meaningful normalization
    let input = Tensor::rand(&[8, 64]);
    println!("\nInput shape: {:?}", input.shape());

    let output = batch_norm.forward(&input).expect("Forward failed");
    println!("Output shape: {:?}", output.shape());

    // Switch to eval mode (uses running statistics)
    batch_norm.set_training(false);
    println!("\nSwitched to eval mode: training={}", batch_norm.is_training());

    let eval_output = batch_norm.forward(&input).expect("Eval forward failed");
    println!("Eval output shape: {:?}", eval_output.shape());

    // Back to training mode
    batch_norm.set_training(true);

    // Parameters (gamma, beta)
    println!("\nBatchNorm has {} learnable parameters",
             batch_norm.parameters().len());

    println!();
}

// ============================================================================
// PART 2: RECOMMENDATION LAYERS DEMO
// ============================================================================

fn recommendation_layers_demo() {
    println!("-------------------------------------------------------------");
    println!("  PART 2: Recommendation Layers Demo");
    println!("-------------------------------------------------------------\n");

    din_attention_demo();
    dcn_demo();
    mmoe_demo();
    ffm_demo();
    senet_demo();
    group_interaction_demo();
}

/// Demonstrates DIN (Deep Interest Network) Attention.
///
/// DIN Attention adaptively weights user behavior items based on
/// their relevance to the target item.
///
/// For each key k_i:
/// 1. Compute features: [query, key, query-key, query*key]
/// 2. Pass through attention MLP
/// 3. Apply optional softmax
/// 4. Compute weighted sum of values
fn din_attention_demo() {
    println!("--- DIN (Deep Interest Network) Attention ---\n");

    // Create DIN attention with config
    let config = DINConfig::new(32)  // 32-dim embeddings
        .with_attention_hidden_units(vec![64, 32])
        .with_activation(ActivationType::Sigmoid)
        .with_use_softmax(false);  // Raw attention scores

    let mut din = DINAttention::from_config(config).expect("DIN creation failed");
    println!("Created DINAttention:");
    println!("  Embedding dim: {}", din.embedding_dim());
    println!("  Hidden units: {:?}", din.hidden_units());

    // Input tensors:
    // - Query: target item embedding [batch=2, embedding_dim=32]
    // - Keys: user behavior sequence [batch=2, seq_len=10, embedding_dim=32]
    // - Values: same as keys (or separate value embeddings)
    let query = Tensor::rand(&[2, 32]);
    let keys = Tensor::rand(&[2, 10, 32]);
    let values = keys.clone();

    println!("\nInput shapes:");
    println!("  Query (target item): {:?}", query.shape());
    println!("  Keys (behavior seq): {:?}", keys.shape());
    println!("  Values: {:?}", values.shape());

    // Forward pass
    let output = din.forward_attention(&query, &keys, &values, None)
        .expect("Forward failed");
    println!("\nOutput shape: {:?}  (weighted sum of values)", output.shape());

    // Get attention weights for visualization
    let attention_weights = din.get_attention_weights(&query, &keys, None)
        .expect("Get attention weights failed");
    println!("Attention weights shape: {:?}", attention_weights.shape());
    println!("Sample attention weights: {:?}",
             &attention_weights.data()[0..5.min(attention_weights.numel())]);

    // With mask (handle padding in sequences)
    let mask = Tensor::from_data(&[2, 10], vec![
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0,  // First sample: 7 valid items
        1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,  // Second sample: 5 valid items
    ]);
    let masked_output = din.forward_attention(&query, &keys, &values, Some(&mask))
        .expect("Masked forward failed");
    println!("Masked output shape: {:?}", masked_output.shape());

    // Training mode with backward
    let _ = din.forward_attention_train(&query, &keys, &values, None)
        .expect("Forward train failed");
    let grad = Tensor::ones(&[2, 32]);
    let input_grad = din.backward(&grad).expect("Backward failed");
    println!("\nBackward - gradient shape: {:?}", input_grad.shape());

    // Parameters from attention MLP
    println!("DIN has {} parameter tensors", din.parameters().len());

    println!();
}

/// Demonstrates DCN (Deep & Cross Network).
///
/// DCN efficiently captures feature interactions of bounded degrees.
/// Cross layer: x_{l+1} = x_0 * (w^T * x_l + b) + x_l
///
/// Supports two modes:
/// - Vector mode (original DCN): weight is [d, 1]
/// - Matrix mode (DCN-V2): weight is [d, d] for richer interactions
fn dcn_demo() {
    println!("--- DCN (Deep & Cross Network) ---\n");

    // Create DCN with vector mode (original)
    let config_v1 = DCNConfig::new(64, 3);  // 64 features, 3 cross layers
    let cross_net_v1 = CrossNetwork::from_config(&config_v1);

    println!("Created CrossNetwork (Vector mode):");
    println!("  Input dim: {}", cross_net_v1.input_dim());
    println!("  Num layers: {}", cross_net_v1.num_layers());
    println!("  Mode: {:?}", cross_net_v1.mode());

    // DCN-V2 with matrix mode
    let config_v2 = DCNConfig::new(64, 3).with_mode(DCNMode::Matrix);
    let cross_net_v2 = CrossNetwork::from_config(&config_v2);
    println!("\nCreated CrossNetwork (Matrix mode/DCN-V2):");
    println!("  Mode: {:?}", cross_net_v2.mode());

    // Forward pass
    // Input: [batch=8, features=64]
    let input = Tensor::rand(&[8, 64]);
    println!("\nInput shape: {:?}", input.shape());

    let output_v1 = cross_net_v1.forward(&input).expect("V1 forward failed");
    println!("V1 output shape: {:?}", output_v1.shape());

    let output_v2 = cross_net_v2.forward(&input).expect("V2 forward failed");
    println!("V2 output shape: {:?}", output_v2.shape());

    // Training with backward
    let mut cross_train = CrossNetwork::new(64, 2, DCNMode::Vector);
    let _ = cross_train.forward_train(&input).expect("Forward train failed");
    let grad = Tensor::ones(&[8, 64]);
    let input_grad = cross_train.backward(&grad).expect("Backward failed");
    println!("\nBackward - input gradient shape: {:?}", input_grad.shape());

    // Parameters: each layer has weight + bias
    println!("\nV1 parameters: {} tensors (3 layers * 2)",
             cross_net_v1.parameters().len());

    // Access individual layers
    println!("Cross layer weights shapes:");
    for (i, layer) in cross_net_v1.layers().iter().enumerate() {
        println!("  Layer {}: weight {:?}, bias {:?}",
                 i, layer.weight().shape(), layer.bias().shape());
    }

    println!();
}

/// Demonstrates MMoE (Multi-gate Mixture of Experts).
///
/// MMoE is for multi-task learning with:
/// - Shared expert networks
/// - Task-specific gating networks
/// - Per-task weighted combination of expert outputs
fn mmoe_demo() {
    println!("--- MMoE (Multi-gate Mixture of Experts) ---\n");

    // Create MMoE: 4 experts, 2 tasks
    let config = MMoEConfig::new(64, 4, 2)  // 64 input, 4 experts, 2 tasks
        .with_expert_hidden_units(vec![32, 32])
        .with_expert_activation(ActivationType::ReLU);

    let mmoe = MMoE::from_config(config).expect("MMoE creation failed");
    println!("Created MMoE:");
    println!("  Input dim: {}", mmoe.input_dim());
    println!("  Num experts: {}", mmoe.num_experts());
    println!("  Num tasks: {}", mmoe.num_tasks());
    println!("  Expert output dim: {}", mmoe.expert_output_dim());

    // Forward pass - returns output for each task
    // Input: [batch=8, features=64]
    let input = Tensor::rand(&[8, 64]);
    println!("\nInput shape: {:?}", input.shape());

    let task_outputs = mmoe.forward_multi(&input).expect("Forward failed");
    println!("Number of task outputs: {}", task_outputs.len());
    for (i, output) in task_outputs.iter().enumerate() {
        println!("  Task {} output shape: {:?}", i, output.shape());
    }

    // Training with multi-task backward
    let mut mmoe_train = MMoE::new(64, 3, 2, &[32], ActivationType::ReLU).unwrap();
    let _ = mmoe_train.forward_multi_train(&input).expect("Forward train failed");

    // Provide gradient for each task
    let grads = vec![
        Tensor::ones(&[8, 32]),  // Task 0 gradient
        Tensor::ones(&[8, 32]),  // Task 1 gradient
    ];
    let input_grad = mmoe_train.backward_multi(&grads).expect("Backward failed");
    println!("\nBackward - input gradient shape: {:?}", input_grad.shape());

    // Access experts and gates
    println!("\nExperts: {} networks", mmoe.experts().len());
    println!("Gates: {} networks (one per task)", mmoe.gates().len());

    // Parameters
    let params = mmoe.parameters();
    println!("Total parameters: {} tensors", params.len());

    println!();
}

/// Demonstrates FFM (Field-aware Factorization Machine).
///
/// FFM learns field-specific embeddings for feature interactions.
/// Each feature has multiple embeddings - one for each field it may interact with.
///
/// Output = sum over field pairs (i,j): <v_{i,fj}, v_{j,fi}>
fn ffm_demo() {
    println!("--- FFM (Field-aware Factorization Machine) ---\n");

    // Create FFM: 5 fields, 8-dim embeddings
    let config = FFMConfig::new(5, 8).with_bias(true);
    let mut ffm = FFMLayer::from_config(&config);

    println!("Created FFMLayer:");
    println!("  Num fields: {}", ffm.num_fields());
    println!("  Embedding dim: {}", ffm.embedding_dim());
    println!("  Use bias: {}", ffm.use_bias());
    println!("  Embeddings shape: {:?}", ffm.embeddings().shape());

    // Forward pass with explicit field indices and values
    // Field indices: which field each feature belongs to
    // Field values: feature values (e.g., 1.0 for one-hot, actual value for numerical)
    let field_indices = Tensor::from_data(&[2, 5], vec![
        0.0, 1.0, 2.0, 3.0, 4.0,  // Sample 1: features from fields 0,1,2,3,4
        0.0, 1.0, 2.0, 3.0, 4.0,  // Sample 2: same field structure
    ]);
    let field_values = Tensor::ones(&[2, 5]);  // All features active with value 1.0

    println!("\nField indices shape: {:?}", field_indices.shape());
    println!("Field values shape: {:?}", field_values.shape());

    let output = ffm.forward_with_fields(&field_indices, &field_values)
        .expect("Forward failed");
    println!("Output shape: {:?}  (one scalar per sample)", output.shape());

    // Training with backward
    let _ = ffm.forward_train_with_fields(&field_indices, &field_values)
        .expect("Forward train failed");
    let grad = Tensor::ones(&[2, 1]);
    ffm.backward_ffm(&grad).expect("Backward failed");
    println!("\nGradients computed: embeddings_grad={}",
             ffm.embeddings_grad().is_some());

    // Parameters
    println!("FFM has {} parameter tensors", ffm.parameters().len());

    println!();
}

/// Demonstrates SENet (Squeeze-and-Excitation Network) for feature importance.
///
/// SENet architecture:
/// 1. Squeeze: global average pooling (for 2D features, we use identity)
/// 2. Excitation: FC1 -> ReLU -> FC2 -> Sigmoid (learns feature importance)
/// 3. Scale: input * attention_weights (reweight features)
fn senet_demo() {
    println!("--- SENet (Squeeze-and-Excitation Network) ---\n");

    // Create SENet with reduction ratio 4
    // Bottleneck dim = 64 / 4 = 16
    let config = SENetConfig::new(64)
        .with_reduction_ratio(4)
        .with_bias(true);

    let mut senet = SENetLayer::from_config(config).expect("SENet creation failed");
    println!("Created SENetLayer:");
    println!("  Input dim: {}", senet.input_dim());
    println!("  Bottleneck dim: {} (64 / 4)", senet.bottleneck_dim());
    println!("  Reduction ratio: {}", senet.reduction_ratio());

    // Forward pass
    // Input: [batch=8, features=64]
    let input = Tensor::rand(&[8, 64]);
    println!("\nInput shape: {:?}", input.shape());

    let output = senet.forward(&input).expect("Forward failed");
    println!("Output shape: {:?}  (same as input)", output.shape());

    // Training forward to get attention weights
    let _ = senet.forward_train(&input).expect("Forward train failed");
    if let Some(attention) = senet.last_attention_weights() {
        println!("\nAttention weights shape: {:?}", attention.shape());
        println!("Attention sample (first 5): {:?}",
                 &attention.data()[0..5.min(attention.numel())]);
        // Verify sigmoid output range [0, 1]
        let in_range = attention.data().iter().all(|&v| v >= 0.0 && v <= 1.0);
        println!("All attention weights in [0,1]: {}", in_range);
    }

    // Backward pass
    let grad = Tensor::ones(&[8, 64]);
    let input_grad = senet.backward(&grad).expect("Backward failed");
    println!("\nBackward - input gradient shape: {:?}", input_grad.shape());

    // Parameters: fc1 (64->16) + fc2 (16->64), each with weight+bias
    println!("SENet has {} parameter tensors", senet.parameters().len());

    println!();
}

/// Demonstrates Group Interaction Layer.
///
/// Computes feature interactions within and across feature groups.
/// Useful when features have semantic groupings (user, item, context).
///
/// Supports:
/// - Intra-group interactions (within same group)
/// - Inter-group interactions (across different groups)
/// - Different interaction types (inner product, Hadamard, concat)
fn group_interaction_demo() {
    println!("--- Group Interaction Layer ---\n");

    // Create GroupInteraction: 3 groups, 4 features per group, 8-dim embeddings
    let config = GroupInteractionConfig::new(3, 4, 8)
        .with_interaction_type(InteractionType::InnerProduct)
        .with_inter_group(true)
        .with_intra_group(false)
        .with_include_original(true);

    let mut layer = GroupInteractionLayer::from_config(&config);
    println!("Created GroupInteractionLayer:");
    println!("  Num groups: {}", config.num_groups);
    println!("  Features per group: {}", config.features_per_group);
    println!("  Embedding dim: {}", config.embedding_dim);
    println!("  Interaction type: {:?}", config.interaction_type);
    println!("  Inter-group: {}", config.inter_group);
    println!("  Intra-group: {}", config.intra_group);
    println!("  Input dim: {}", layer.input_dim());
    println!("  Output dim: {}", layer.output_dim());

    // Forward pass
    // Input: [batch=2, total_features * embedding_dim]
    // Total features = 3 groups * 4 features = 12
    // Input dim = 12 * 8 = 96
    let input = Tensor::rand(&[2, 96]);
    println!("\nInput shape: {:?}", input.shape());

    let output = layer.forward(&input).expect("Forward failed");
    println!("Output shape: {:?}", output.shape());

    // Different interaction types
    println!("\nInteraction types comparison:");
    for interaction_type in [
        InteractionType::InnerProduct,
        InteractionType::Hadamard,
        InteractionType::Concat,
    ] {
        let test_config = GroupInteractionConfig::new(2, 3, 4)
            .with_interaction_type(interaction_type)
            .with_inter_group(true)
            .with_include_original(false);
        println!("  {:?}: output_dim={}", interaction_type, test_config.output_dim());
    }

    // Backward pass
    let _ = layer.forward_train(&input).expect("Forward train failed");
    let grad = Tensor::ones(&[2, layer.output_dim()]);
    let input_grad = layer.backward(&grad).expect("Backward failed");
    println!("\nBackward - input gradient shape: {:?}", input_grad.shape());

    // Note: GroupInteractionLayer has no learnable parameters
    println!("Parameters: {} (no learnable params)", layer.parameters().len());

    println!();
}

// ============================================================================
// PART 3: SEQUENCE LAYERS DEMO
// ============================================================================

fn sequence_layers_demo() {
    println!("-------------------------------------------------------------");
    println!("  PART 3: Sequence Layers Demo");
    println!("-------------------------------------------------------------\n");

    agru_demo();
    dien_demo();
}

/// Demonstrates AGRU (Attention GRU).
///
/// AGRU modifies the GRU update gate with attention scores:
/// - Standard GRU: h_t = (1-z) * h_{t-1} + z * h_tilde
/// - AGRU: uses attention to scale the update gate
///
/// This allows the model to focus on relevant parts of the sequence.
fn agru_demo() {
    println!("--- AGRU (Attention GRU) ---\n");

    // Create AGRU: 32 input dim, 64 hidden dim
    let mut agru = AGRU::new(32, 64);
    println!("Created AGRU:");
    println!("  Input dim: {}", agru.input_dim());
    println!("  Hidden dim: {}", agru.hidden_dim());

    // Forward pass with attention
    // Input: [batch=2, seq_len=10, input_dim=32]
    // Attention: [batch=2, seq_len=10] - importance scores for each timestep
    let input = Tensor::rand(&[2, 10, 32]);
    let attention = Tensor::rand(&[2, 10]);  // Attention scores (e.g., from DIN)

    println!("\nInput shape: {:?}", input.shape());
    println!("Attention shape: {:?}", attention.shape());

    // Forward with attention modulation
    let output = agru.forward_with_attention(&input, &attention)
        .expect("Forward with attention failed");
    println!("Output shape: {:?}  (final hidden state)", output.shape());

    // Standard GRU forward (without attention modulation)
    let standard_output = agru.forward_standard(&input)
        .expect("Standard forward failed");
    println!("Standard GRU output shape: {:?}", standard_output.shape());

    // Get all hidden states (for sequence-to-sequence tasks)
    let all_states = agru.forward_all_states(&input, &attention)
        .expect("Forward all states failed");
    println!("All hidden states shape: {:?}  (batch, seq_len, hidden)", all_states.shape());

    // Parameters: 3 gates (reset, update, candidate) each with input and hidden weights
    let params = agru.parameters();
    println!("\nAGRU has {} parameter tensors", params.len());
    println!("  (3 gates * 2 weight matrices + 3 biases)");

    // Show parameter shapes
    println!("\nParameter shapes:");
    for (i, param) in params.iter().enumerate() {
        println!("  Param {}: {:?}", i, param.shape());
    }

    println!();
}

/// Demonstrates DIEN (Deep Interest Evolution Network).
///
/// DIEN has a two-layer GRU structure:
/// 1. Interest Extraction Layer: standard GRU to capture temporal patterns
/// 2. Interest Evolution Layer: AUGRU to evolve interests based on target item
///
/// Also supports auxiliary loss for interest supervision.
fn dien_demo() {
    println!("--- DIEN (Deep Interest Evolution Network) ---\n");

    // Create DIEN with config
    let config = DIENConfig::new(32, 64)  // 32-dim embeddings, 64-dim hidden
        .with_gru_type(GRUType::AUGRU)    // Use AUGRU for evolution layer
        .with_attention_hidden_units(vec![64, 32])
        .with_use_auxiliary_loss(true)
        .with_use_softmax(false);

    let mut dien = DIENLayer::from_config(config).expect("DIEN creation failed");
    println!("Created DIENLayer:");
    println!("  Embedding dim: {}", dien.embedding_dim());
    println!("  Hidden size: {}", dien.hidden_size());

    // Input tensors:
    // - Behavior sequence: [batch=2, seq_len=10, embedding_dim=32]
    // - Target item: [batch=2, embedding_dim=32]
    // - Mask: [batch=2, seq_len=10] for valid positions
    let behavior_seq = Tensor::rand(&[2, 10, 32]);
    let target_item = Tensor::rand(&[2, 32]);
    let mask = Tensor::from_data(&[2, 10], vec![
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0,  // 7 valid items
        1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,  // 5 valid items
    ]);

    println!("\nInput shapes:");
    println!("  Behavior sequence: {:?}", behavior_seq.shape());
    println!("  Target item: {:?}", target_item.shape());
    println!("  Mask: {:?}", mask.shape());

    // Forward pass
    let output = dien.forward_dien(&behavior_seq, &target_item, Some(&mask))
        .expect("Forward failed");
    println!("\nOutput shape: {:?}  (evolved interest representation)", output.shape());

    // Get attention scores for visualization
    let attention_scores = dien.get_attention_scores(&behavior_seq, &target_item, Some(&mask))
        .expect("Get attention scores failed");
    println!("Attention scores shape: {:?}", attention_scores.shape());
    println!("Sample attention: {:?}", &attention_scores.data()[0..5.min(attention_scores.numel())]);

    // Auxiliary loss for interest extraction supervision
    // Negative samples: [batch, seq_len, embedding_dim]
    let negative_samples = Tensor::rand(&[2, 10, 32]);
    let aux_loss = dien.auxiliary_loss(&behavior_seq, &negative_samples)
        .expect("Auxiliary loss failed");
    println!("\nAuxiliary loss: {:.6}", aux_loss);

    // Parameters
    let params = dien.parameters();
    println!("\nDIEN has {} parameter tensors", params.len());

    println!();
}

// ============================================================================
// PART 4: COMPLETE MODEL DEMO
// ============================================================================

/// Demonstrates a complete recommendation model combining multiple layer types.
///
/// Architecture:
/// 1. Embedding Lookup for categorical features
/// 2. DIN Attention for user behavior sequence
/// 3. DCN Cross Network for feature interactions
/// 4. MMoE for multi-task learning (CTR + CVR)
/// 5. Final dense layers for predictions
fn complete_model_demo() {
    println!("-------------------------------------------------------------");
    println!("  PART 4: Complete Recommendation Model");
    println!("-------------------------------------------------------------\n");

    println!("Building a recommendation model with:");
    println!("  1. Embedding Lookup");
    println!("  2. DIN Attention for user history");
    println!("  3. DCN Cross Network");
    println!("  4. MMoE for multi-task (CTR + CVR)");
    println!("  5. Dense output layers\n");

    // Hyperparameters
    let embedding_dim = 16;
    let hidden_dim = 64;
    let num_behavior_items = 10;
    let batch_size = 4;

    // ========================================
    // 1. Embedding Lookup (simulated)
    // ========================================
    println!("Step 1: Embedding Lookup");

    // Create embedding hash table
    let mut hash_table = EmbeddingHashTable::new(embedding_dim);

    // Insert some item embeddings
    for i in 0..100 {
        let embedding: Vec<f32> = (0..embedding_dim).map(|j| (i * j) as f32 * 0.01).collect();
        hash_table.insert(i as u64, embedding);
    }
    println!("  Created hash table with 100 item embeddings");

    // Lookup embeddings for user behavior sequence
    let user_item_ids: Vec<u64> = (0..batch_size * num_behavior_items)
        .map(|i| (i % 100) as u64)
        .collect();

    let lookup = EmbeddingLookup::new(hash_table.clone());
    let behavior_embeddings = lookup.lookup(&user_item_ids);
    // Reshape to [batch, seq_len, embedding_dim]
    let behavior_seq = behavior_embeddings.reshape(&[batch_size, num_behavior_items, embedding_dim]);
    println!("  Behavior sequence shape: {:?}", behavior_seq.shape());

    // Lookup target item embeddings
    let target_ids: Vec<u64> = (0..batch_size).map(|i| (i * 10) as u64).collect();
    let target_embeddings = lookup.lookup(&target_ids);
    // Shape: [batch, embedding_dim]
    println!("  Target embeddings shape: {:?}", target_embeddings.shape());

    // ========================================
    // 2. DIN Attention on user behavior
    // ========================================
    println!("\nStep 2: DIN Attention");

    let din_config = DINConfig::new(embedding_dim)
        .with_attention_hidden_units(vec![32, 16]);
    let din = DINAttention::from_config(din_config).expect("DIN creation failed");

    // Query: target item, Keys/Values: user behavior
    let user_interest = din.forward_attention(
        &target_embeddings,
        &behavior_seq,
        &behavior_seq,
        None
    ).expect("DIN forward failed");
    println!("  User interest shape: {:?}  (attention-weighted behavior)", user_interest.shape());

    // ========================================
    // 3. Concatenate features for cross network
    // ========================================
    println!("\nStep 3: Feature Concatenation");

    // Concatenate: target embedding + user interest + simulated user features
    let user_features = Tensor::rand(&[batch_size, 16]);  // Simulated user features

    // Manual concatenation: [batch, embedding_dim + embedding_dim + 16]
    let concat_dim = embedding_dim + embedding_dim + 16;  // 16 + 16 + 16 = 48
    let mut concat_data = vec![0.0f32; batch_size * concat_dim];

    for b in 0..batch_size {
        let offset = b * concat_dim;
        // Copy target embedding
        for i in 0..embedding_dim {
            concat_data[offset + i] = target_embeddings.data()[b * embedding_dim + i];
        }
        // Copy user interest
        for i in 0..embedding_dim {
            concat_data[offset + embedding_dim + i] = user_interest.data()[b * embedding_dim + i];
        }
        // Copy user features
        for i in 0..16 {
            concat_data[offset + 2 * embedding_dim + i] = user_features.data()[b * 16 + i];
        }
    }
    let combined_features = Tensor::from_data(&[batch_size, concat_dim], concat_data);
    println!("  Combined features shape: {:?}", combined_features.shape());

    // ========================================
    // 4. DCN Cross Network
    // ========================================
    println!("\nStep 4: DCN Cross Network");

    let dcn_config = DCNConfig::new(concat_dim, 2);  // 2 cross layers
    let cross_network = CrossNetwork::from_config(&dcn_config);

    let cross_output = cross_network.forward(&combined_features)
        .expect("DCN forward failed");
    println!("  Cross network output shape: {:?}", cross_output.shape());

    // ========================================
    // 5. MMoE for multi-task learning
    // ========================================
    println!("\nStep 5: MMoE Multi-Task");

    let mmoe_config = MMoEConfig::new(concat_dim, 3, 2)  // 3 experts, 2 tasks
        .with_expert_hidden_units(vec![32])
        .with_expert_activation(ActivationType::ReLU);
    let mmoe = MMoE::from_config(mmoe_config).expect("MMoE creation failed");

    let task_outputs = mmoe.forward_multi(&cross_output).expect("MMoE forward failed");
    println!("  Task 0 (CTR) output shape: {:?}", task_outputs[0].shape());
    println!("  Task 1 (CVR) output shape: {:?}", task_outputs[1].shape());

    // ========================================
    // 6. Final dense layers for predictions
    // ========================================
    println!("\nStep 6: Output Layers");

    // CTR tower
    let ctr_hidden = Dense::new(32, 16);
    let ctr_output = Dense::new(16, 1);

    let ctr_h = ctr_hidden.forward(&task_outputs[0]).expect("CTR hidden failed");
    let ctr_h = ctr_h.map(|x| x.max(0.0));  // ReLU
    let ctr_logits = ctr_output.forward(&ctr_h).expect("CTR output failed");
    let ctr_probs = ctr_logits.map(|x| 1.0 / (1.0 + (-x).exp()));  // Sigmoid
    println!("  CTR predictions shape: {:?}", ctr_probs.shape());

    // CVR tower
    let cvr_hidden = Dense::new(32, 16);
    let cvr_output = Dense::new(16, 1);

    let cvr_h = cvr_hidden.forward(&task_outputs[1]).expect("CVR hidden failed");
    let cvr_h = cvr_h.map(|x| x.max(0.0));  // ReLU
    let cvr_logits = cvr_output.forward(&cvr_h).expect("CVR output failed");
    let cvr_probs = cvr_logits.map(|x| 1.0 / (1.0 + (-x).exp()));  // Sigmoid
    println!("  CVR predictions shape: {:?}", cvr_probs.shape());

    // ========================================
    // Summary
    // ========================================
    println!("\n--- Model Summary ---");
    println!("Data flow:");
    println!("  Input: {} users, {} behavior items, {} embedding dim",
             batch_size, num_behavior_items, embedding_dim);
    println!("  Behavior embeddings: {:?}", behavior_seq.shape());
    println!("  DIN attention output: {:?}", user_interest.shape());
    println!("  Combined features: {:?}", combined_features.shape());
    println!("  DCN output: {:?}", cross_output.shape());
    println!("  MMoE task outputs: {:?}, {:?}", task_outputs[0].shape(), task_outputs[1].shape());
    println!("  Final CTR prediction: {:?}", ctr_probs.shape());
    println!("  Final CVR prediction: {:?}", cvr_probs.shape());

    // Print sample predictions
    println!("\nSample predictions:");
    for b in 0..batch_size {
        println!("  Sample {}: CTR={:.4}, CVR={:.4}",
                 b, ctr_probs.data()[b], cvr_probs.data()[b]);
    }

    // Count total parameters
    let total_params: usize =
        din.parameters().iter().map(|t| t.numel()).sum::<usize>() +
        cross_network.parameters().iter().map(|t| t.numel()).sum::<usize>() +
        mmoe.parameters().iter().map(|t| t.numel()).sum::<usize>() +
        ctr_hidden.parameters().iter().map(|t| t.numel()).sum::<usize>() +
        ctr_output.parameters().iter().map(|t| t.numel()).sum::<usize>() +
        cvr_hidden.parameters().iter().map(|t| t.numel()).sum::<usize>() +
        cvr_output.parameters().iter().map(|t| t.numel()).sum::<usize>();

    println!("\nTotal trainable parameters: {}", total_params);

    println!("\n=============================================================");
    println!("                    Demo Complete!");
    println!("=============================================================");
}
