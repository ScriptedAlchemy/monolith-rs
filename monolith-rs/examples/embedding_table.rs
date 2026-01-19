//! Comprehensive example demonstrating embedding hash table features in Monolith.
//!
//! This example covers:
//! 1. Basic hash table operations (assign, lookup, apply_gradients)
//! 2. Different initializers (Zeros, RandomNormal, RandomUniform, TruncatedNormal, Xavier)
//! 3. Different optimizers (SGD, Adam, Adagrad, FTRL, Momentum)
//! 4. MultiHashTable for sharded/parallel access
//! 5. Eviction policies (TimeBasedEviction, LRUEviction)
//! 6. Compression (Fp16Compressor, FixedR8Compressor)
//! 7. Performance benchmarking
//!
//! Run with: cargo run --example embedding_table

use std::sync::Arc;
use std::time::Instant;

use monolith_hash_table::{
    // Compressors
    compressor::{Compressor, FixedR8Compressor, Fp16Compressor, NoCompression, OneBitCompressor},
    // Eviction policies
    eviction::{LRUEviction, TimeBasedEviction},

    // Initializers
    ConstantInitializer,
    // Core types
    CuckooEmbeddingHashTable,
    EmbeddingEntry,
    EmbeddingHashTable,
    Initializer,
    MultiHashTable,
    OnesInitializer,
    OptimizerState,

    RandomNormalInitializer,
    RandomUniformInitializer,
    TruncatedNormalInitializer,
    XavierNormalInitializer,
    XavierUniformInitializer,
    ZerosInitializer,
};

// ============================================================================
// Section 1: Basic Hash Table Operations
// ============================================================================

/// Demonstrates basic hash table operations: assign, lookup, and apply_gradients.
fn demonstrate_basic_operations() {
    println!("\n=== Section 1: Basic Hash Table Operations ===\n");

    // Create a cuckoo hash table with:
    // - capacity: 1024 entries
    // - dimension: 8 (each embedding has 8 float values)
    let mut table = CuckooEmbeddingHashTable::new(1024, 8);

    println!(
        "Created CuckooEmbeddingHashTable with capacity={}, dim={}",
        table.capacity(),
        table.dim()
    );

    // Assign embeddings to feature IDs
    // Each ID gets a vector of `dim` floats
    let ids = vec![100, 200, 300];
    let embeddings = vec![
        // Embedding for ID 100
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, // Embedding for ID 200
        1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, // Embedding for ID 300
        2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8,
    ];

    table
        .assign(&ids, &embeddings)
        .expect("Failed to assign embeddings");
    println!(
        "Assigned {} embeddings, table size: {}",
        ids.len(),
        table.size()
    );

    // Look up embeddings by ID
    let lookup_ids = vec![100, 300]; // Look up 2 of the 3 IDs
    let mut output = vec![0.0; lookup_ids.len() * table.dim()];
    table
        .lookup(&lookup_ids, &mut output)
        .expect("Failed to lookup");

    println!("Looked up IDs {:?}:", lookup_ids);
    println!("  ID 100 embedding: {:?}", &output[0..8]);
    println!("  ID 300 embedding: {:?}", &output[8..16]);

    // Apply gradients to update embeddings (training step)
    // The table uses SGD by default: embedding = embedding - lr * gradient
    let gradients = vec![
        0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, // gradients for ID 100
        0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, // gradients for ID 300
    ];

    table
        .apply_gradients(&lookup_ids, &gradients)
        .expect("Failed to apply gradients");
    println!("Applied gradients to IDs {:?}", lookup_ids);

    // Look up again to see the updated values
    table
        .lookup(&lookup_ids, &mut output)
        .expect("Failed to lookup");
    println!("After gradient update:");
    println!("  ID 100 embedding: {:?}", &output[0..8]);

    // Check if an ID exists
    println!("Contains ID 100: {}", table.contains(100));
    println!("Contains ID 999: {}", table.contains(999));

    // Get table statistics
    println!("Load factor: {:.4}", table.load_factor());
    println!("Memory usage: {} bytes", table.memory_usage());
}

// ============================================================================
// Section 2: Different Initializers
// ============================================================================

/// Demonstrates all available embedding initializers.
fn demonstrate_initializers() {
    println!("\n=== Section 2: Embedding Initializers ===\n");

    let dim = 8;

    // 1. ZerosInitializer - All values set to 0
    // Use case: When you want embeddings to start from zero (not recommended for most cases
    // due to symmetry issues during training)
    let zeros = ZerosInitializer;
    let zeros_embedding = zeros.initialize(dim);
    println!("ZerosInitializer: {:?}", zeros_embedding);
    println!("  Name: {}", zeros.name());

    // 2. OnesInitializer - All values set to 1
    let ones = OnesInitializer;
    let ones_embedding = ones.initialize(dim);
    println!("OnesInitializer: {:?}", ones_embedding);

    // 3. ConstantInitializer - All values set to a constant
    let constant = ConstantInitializer::new(0.5);
    let constant_embedding = constant.initialize(dim);
    println!("ConstantInitializer(0.5): {:?}", constant_embedding);

    // 4. RandomUniformInitializer - Uniform distribution in [min, max)
    // Use case: Good default choice for many models
    let uniform = RandomUniformInitializer::new(-0.1, 0.1);
    let uniform_embedding = uniform.initialize(dim);
    println!(
        "RandomUniformInitializer(-0.1, 0.1): {:?}",
        uniform_embedding
    );
    println!(
        "  Values in range [-0.1, 0.1): {}",
        uniform_embedding.iter().all(|&v| (-0.1..0.1).contains(&v))
    );

    // 5. RandomNormalInitializer - Normal/Gaussian distribution
    // Use case: When you want values centered around a mean with standard deviation
    let normal = RandomNormalInitializer::new(0.0, 0.05);
    let normal_embedding = normal.initialize(dim);
    println!(
        "RandomNormalInitializer(mean=0, stddev=0.05): {:?}",
        normal_embedding
    );

    // 6. TruncatedNormalInitializer - Truncated normal distribution
    // Values outside 2 standard deviations are resampled
    // Use case: Prevents extreme outliers while maintaining normal distribution
    let truncated = TruncatedNormalInitializer::new(0.0, 0.1);
    let truncated_embedding = truncated.initialize(dim);
    println!(
        "TruncatedNormalInitializer(mean=0, stddev=0.1): {:?}",
        truncated_embedding
    );
    let within_bounds = truncated_embedding
        .iter()
        .all(|&v| (-0.2..=0.2).contains(&v));
    println!("  All values within 2 stddev: {}", within_bounds);

    // 7. XavierUniformInitializer - Xavier/Glorot uniform initialization
    // Range: [-sqrt(6 / (fan_in + fan_out)), sqrt(6 / (fan_in + fan_out))]
    // Use case: Deep networks, helps maintain gradient flow
    let xavier_uniform = XavierUniformInitializer::new(1.0);
    let xavier_uniform_embedding = xavier_uniform.initialize(dim);
    println!(
        "XavierUniformInitializer(gain=1.0): {:?}",
        xavier_uniform_embedding
    );

    // 8. XavierNormalInitializer - Xavier/Glorot normal initialization
    // Use case: Similar to XavierUniform but with normal distribution
    let xavier_normal = XavierNormalInitializer::new(1.0);
    let xavier_normal_embedding = xavier_normal.initialize(dim);
    println!(
        "XavierNormalInitializer(gain=1.0): {:?}",
        xavier_normal_embedding
    );

    // Using initializers with hash tables
    println!("\n--- Using Initializers with Hash Tables ---");

    let mut table = CuckooEmbeddingHashTable::with_initializer(
        1024,
        dim,
        Arc::new(TruncatedNormalInitializer::new(0.0, 0.01)),
    );
    println!(
        "Created table with initializer: {}",
        table.initializer_name()
    );

    // get_or_initialize creates new entries using the configured initializer
    table.get_or_initialize(42).expect("Failed to initialize");
    let mut output = vec![0.0; dim];
    table.lookup(&[42], &mut output).expect("Failed to lookup");
    println!("Auto-initialized embedding for ID 42: {:?}", output);

    // lookup_or_initialize combines lookup with automatic initialization
    let ids = vec![100, 101, 102];
    let mut outputs = vec![0.0; ids.len() * dim];
    table
        .lookup_or_initialize(&ids, &mut outputs)
        .expect("Failed to lookup_or_initialize");
    println!(
        "lookup_or_initialize created {} new entries, table size: {}",
        ids.len(),
        table.size()
    );
}

// ============================================================================
// Section 3: Different Optimizers
// ============================================================================

/// Demonstrates different optimizer states for embedding updates.
fn demonstrate_optimizers() {
    println!("\n=== Section 3: Optimizer States ===\n");

    let dim = 4;
    let learning_rate = 0.1;
    let initial_embedding = vec![1.0, 2.0, 3.0, 4.0];
    let gradients = vec![0.1, 0.2, 0.3, 0.4];

    // 1. SGD (OptimizerState::None)
    // Update rule: embedding = embedding - lr * gradient
    println!("--- SGD (OptimizerState::None) ---");
    let mut entry_sgd = EmbeddingEntry::new(1, initial_embedding.clone());
    println!("Initial embedding: {:?}", entry_sgd.embedding());
    entry_sgd.apply_gradient_update(&gradients, learning_rate);
    println!("After SGD update:  {:?}", entry_sgd.embedding());
    println!("Expected: [0.99, 1.98, 2.97, 3.96] (embedding - 0.1 * gradient)");

    // 2. Adam Optimizer
    // Adaptive learning rate using first and second moment estimates
    // Good for: Sparse gradients, noisy data, large models
    println!("\n--- Adam Optimizer ---");
    let mut entry_adam = EmbeddingEntry::with_optimizer_state(
        2,
        initial_embedding.clone(),
        OptimizerState::new_adam(dim),
    );
    println!("Initial embedding: {:?}", entry_adam.embedding());

    // Apply multiple gradient updates to see Adam's adaptive behavior
    for i in 0..3 {
        entry_adam.apply_gradient_update(&gradients, learning_rate);
        println!("After update {}: {:?}", i + 1, entry_adam.embedding());
    }

    if let OptimizerState::Adam { m, v: _, t } = entry_adam.optimizer_state() {
        println!("Adam state: m={:?}, t={}", m, t);
    }

    // 3. Adagrad Optimizer
    // Adapts learning rate per-parameter based on historical gradients
    // Good for: Sparse features (like in recommendation systems)
    println!("\n--- Adagrad Optimizer ---");
    let mut entry_adagrad = EmbeddingEntry::with_optimizer_state(
        3,
        initial_embedding.clone(),
        OptimizerState::new_adagrad(dim),
    );
    println!("Initial embedding: {:?}", entry_adagrad.embedding());

    for i in 0..3 {
        entry_adagrad.apply_gradient_update(&gradients, learning_rate);
        println!("After update {}: {:?}", i + 1, entry_adagrad.embedding());
    }

    if let OptimizerState::Adagrad { accumulator } = entry_adagrad.optimizer_state() {
        println!("Adagrad accumulator: {:?}", accumulator);
    }

    // 4. FTRL (Follow The Regularized Leader) Optimizer
    // Designed for sparse linear models with L1/L2 regularization
    // Good for: CTR prediction, ads ranking
    println!("\n--- FTRL Optimizer ---");
    let mut entry_ftrl = EmbeddingEntry::with_optimizer_state(
        4,
        initial_embedding.clone(),
        OptimizerState::new_ftrl(dim),
    );
    println!("Initial embedding: {:?}", entry_ftrl.embedding());

    for i in 0..3 {
        entry_ftrl.apply_gradient_update(&gradients, learning_rate);
        println!("After update {}: {:?}", i + 1, entry_ftrl.embedding());
    }

    // 5. Momentum Optimizer
    // Accumulates velocity to accelerate convergence
    // Good for: Navigating ravines and saddle points
    println!("\n--- Momentum Optimizer ---");
    let mut entry_momentum = EmbeddingEntry::with_optimizer_state(
        5,
        initial_embedding.clone(),
        OptimizerState::new_momentum(dim),
    );
    println!("Initial embedding: {:?}", entry_momentum.embedding());

    for i in 0..3 {
        entry_momentum.apply_gradient_update(&gradients, learning_rate);
        println!("After update {}: {:?}", i + 1, entry_momentum.embedding());
    }

    if let OptimizerState::Momentum { velocity } = entry_momentum.optimizer_state() {
        println!("Momentum velocity: {:?}", velocity);
    }

    // Using optimizer states with hash tables
    println!("\n--- Using Optimizers with Hash Tables ---");
    let mut table = CuckooEmbeddingHashTable::with_learning_rate(1024, dim, 0.01);

    // Manually insert entries with different optimizer states
    table.assign(&[10], &initial_embedding).unwrap();

    // Get the entry and set its optimizer state
    if let Some(entry) = table.get_mut(10) {
        entry.set_optimizer_state(OptimizerState::new_adam(dim));
    }

    // Apply gradients - the entry will use Adam
    table.apply_gradients(&[10], &gradients).unwrap();

    let mut output = vec![0.0; dim];
    table.lookup(&[10], &mut output).unwrap();
    println!("After applying gradients with Adam: {:?}", output);
}

// ============================================================================
// Section 4: MultiHashTable (Sharded Table)
// ============================================================================

/// Demonstrates MultiHashTable for sharded/parallel access patterns.
fn demonstrate_multi_hash_table() {
    println!("\n=== Section 4: MultiHashTable (Sharded Table) ===\n");

    // Create a sharded hash table with:
    // - 4 shards (for parallel access)
    // - 256 capacity per shard (1024 total)
    // - 8 dimensional embeddings
    let mut multi_table = MultiHashTable::new(4, 256, 8);

    println!("Created MultiHashTable:");
    println!("  Num shards: {}", multi_table.num_shards());
    println!("  Total capacity: {}", multi_table.total_capacity());
    println!("  Dimension: {}", multi_table.dim());

    // Insert embeddings - IDs are automatically routed to shards via (id % num_shards)
    let ids: Vec<i64> = (0..100).collect();
    let embeddings: Vec<f32> = (0..800).map(|i| i as f32 * 0.01).collect();

    multi_table
        .assign(&ids, &embeddings)
        .expect("Failed to assign");
    println!("\nAssigned {} embeddings", ids.len());

    // Show shard distribution
    let shard_sizes = multi_table.shard_sizes();
    let shard_loads = multi_table.shard_load_factors();
    println!("\nShard distribution:");
    for (i, (size, load)) in shard_sizes.iter().zip(shard_loads.iter()).enumerate() {
        println!("  Shard {}: {} entries, {:.2}% load", i, size, load * 100.0);
    }

    // Lookup works the same way
    let lookup_ids = vec![0, 25, 50, 75, 99];
    let mut output = vec![0.0; lookup_ids.len() * 8];
    multi_table
        .lookup(&lookup_ids, &mut output)
        .expect("Failed to lookup");
    println!("\nLooked up IDs {:?}", lookup_ids);

    // Apply gradients across shards
    let gradients = vec![0.001; lookup_ids.len() * 8];
    multi_table
        .apply_gradients(&lookup_ids, &gradients)
        .expect("Failed to apply gradients");
    println!("Applied gradients across shards");

    // Access individual shards
    println!("\nAccessing individual shard:");
    if let Ok(shard) = multi_table.get_shard(0) {
        println!("  Shard 0 size: {}", shard.size());
        println!("  Shard 0 load factor: {:.4}", shard.load_factor());
    }

    // Memory usage
    println!("\nTotal memory usage: {} bytes", multi_table.memory_usage());

    // Performance comparison: Single table vs Multi table
    println!("\n--- Performance Comparison ---");

    let num_entries = 10000;
    let dim = 64;
    let test_ids: Vec<i64> = (0..num_entries as i64).collect();
    let test_embeddings: Vec<f32> = (0..(num_entries * dim))
        .map(|i| i as f32 * 0.0001)
        .collect();

    // Single table
    let mut single = CuckooEmbeddingHashTable::new(num_entries * 2, dim);
    let start = Instant::now();
    single.assign(&test_ids, &test_embeddings).unwrap();
    let single_insert_time = start.elapsed();

    // Multi table (4 shards)
    let mut multi = MultiHashTable::new(4, num_entries / 2, dim);
    let start = Instant::now();
    multi.assign(&test_ids, &test_embeddings).unwrap();
    let multi_insert_time = start.elapsed();

    println!("Insert {} entries (dim={}):", num_entries, dim);
    println!("  Single table: {:?}", single_insert_time);
    println!("  Multi table (4 shards): {:?}", multi_insert_time);

    // Lookup benchmark
    let mut output_single = vec![0.0; num_entries * dim];
    let mut output_multi = vec![0.0; num_entries * dim];

    let start = Instant::now();
    single.lookup(&test_ids, &mut output_single).unwrap();
    let single_lookup_time = start.elapsed();

    let start = Instant::now();
    multi.lookup(&test_ids, &mut output_multi).unwrap();
    let multi_lookup_time = start.elapsed();

    println!("Lookup {} entries:", num_entries);
    println!("  Single table: {:?}", single_lookup_time);
    println!("  Multi table (4 shards): {:?}", multi_lookup_time);
}

// ============================================================================
// Section 5: Eviction Policies
// ============================================================================

/// Demonstrates eviction policies for automatic entry removal.
fn demonstrate_eviction() {
    println!("\n=== Section 5: Eviction Policies ===\n");

    let dim = 4;

    // 1. TimeBasedEviction - Evict entries older than expire_time seconds
    println!("--- TimeBasedEviction ---");

    let mut table = CuckooEmbeddingHashTable::with_eviction_policy(
        1024,
        dim,
        Box::new(TimeBasedEviction::new(100)), // 100 seconds expire time
    );

    // Insert some entries with different timestamps
    let ids = vec![1, 2, 3];
    let embeddings = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2];
    table.assign(&ids, &embeddings).unwrap();

    // Manually set timestamps to simulate aging
    if let Some(entry) = table.get_mut(1) {
        entry.set_timestamp(0); // Old entry (timestamp = 0)
    }
    if let Some(entry) = table.get_mut(2) {
        entry.set_timestamp(50); // Medium age entry
    }
    if let Some(entry) = table.get_mut(3) {
        entry.set_timestamp(150); // Recent entry
    }

    println!("Inserted 3 entries with timestamps: 0, 50, 150");
    println!("Table size before eviction: {}", table.size());

    // Evict at time 160: entries with age > 100 will be evicted
    // - Entry 1: age = 160 - 0 = 160 > 100 -> EVICT
    // - Entry 2: age = 160 - 50 = 110 > 100 -> EVICT
    // - Entry 3: age = 160 - 150 = 10 < 100 -> KEEP
    let evicted = table.evict(160);
    println!("Evicted {} entries at time 160 (expire_time=100)", evicted);
    println!("Table size after eviction: {}", table.size());
    println!(
        "Remaining entries: 1={}, 2={}, 3={}",
        table.contains(1),
        table.contains(2),
        table.contains(3)
    );

    // 2. LRUEviction - Evict least recently used entries
    println!("\n--- LRUEviction ---");

    let mut lru_table = CuckooEmbeddingHashTable::with_eviction_policy(
        1024,
        dim,
        Box::new(LRUEviction::new(50)), // Max age of 50 seconds
    );

    lru_table.assign(&[10, 20, 30], &embeddings).unwrap();

    // Simulate usage patterns - update timestamps
    if let Some(entry) = lru_table.get_mut(10) {
        entry.set_timestamp(0); // Not accessed recently
    }
    if let Some(entry) = lru_table.get_mut(20) {
        entry.set_timestamp(90); // Recently accessed
    }
    if let Some(entry) = lru_table.get_mut(30) {
        entry.set_timestamp(95); // Most recently accessed
    }

    println!("Entry access times: 10->0, 20->90, 30->95");
    println!("Current time: 100, max_age: 50");

    let evicted = lru_table.evict(100);
    println!("Evicted {} LRU entries", evicted);
    println!(
        "Entry 10 (age=100): {}",
        if lru_table.contains(10) {
            "kept"
        } else {
            "evicted"
        }
    );
    println!(
        "Entry 20 (age=10): {}",
        if lru_table.contains(20) {
            "kept"
        } else {
            "evicted"
        }
    );
    println!(
        "Entry 30 (age=5): {}",
        if lru_table.contains(30) {
            "kept"
        } else {
            "evicted"
        }
    );

    // 3. Updating timestamps during gradient application
    println!("\n--- Timestamp Updates During Training ---");

    let mut training_table = CuckooEmbeddingHashTable::with_eviction_policy(
        1024,
        dim,
        Box::new(TimeBasedEviction::new(3600)), // 1 hour expire
    );

    training_table.assign(&[1], &[0.1, 0.2, 0.3, 0.4]).unwrap();
    println!(
        "Initial timestamp: {}",
        training_table.get(1).unwrap().get_timestamp()
    );

    // apply_gradients_with_timestamp updates the entry's timestamp
    let gradients = vec![0.01, 0.01, 0.01, 0.01];
    training_table
        .apply_gradients_with_timestamp(&[1], &gradients, Some(1000))
        .unwrap();
    println!(
        "After training at time 1000: {}",
        training_table.get(1).unwrap().get_timestamp()
    );

    // This keeps frequently-trained entries fresh and prevents their eviction

    // 4. NoEviction - Default policy, never evicts
    println!("\n--- NoEviction (Default) ---");
    let mut no_evict_table = CuckooEmbeddingHashTable::new(1024, dim);
    no_evict_table.assign(&[1, 2, 3], &embeddings).unwrap();

    // Even at time MAX, nothing is evicted
    let evicted = no_evict_table.evict(u64::MAX);
    println!("Evicted with NoEviction policy: {} (always 0)", evicted);

    // 5. Changing eviction policy dynamically
    println!("\n--- Dynamic Policy Change ---");
    let mut dynamic_table = CuckooEmbeddingHashTable::new(1024, dim);
    dynamic_table.assign(&[1], &[0.1, 0.2, 0.3, 0.4]).unwrap();

    // Nothing evicted with NoEviction
    let evicted = dynamic_table.evict(1000);
    println!("Before policy change, evicted: {}", evicted);

    // Switch to time-based eviction
    dynamic_table.set_eviction_policy(Box::new(TimeBasedEviction::new(100)));
    let evicted = dynamic_table.evict(1000);
    println!(
        "After policy change to TimeBasedEviction, evicted: {}",
        evicted
    );
}

// ============================================================================
// Section 6: Compression
// ============================================================================

/// Demonstrates embedding compression for memory efficiency.
fn demonstrate_compression() {
    println!("\n=== Section 6: Embedding Compression ===\n");

    let embedding = vec![
        0.123456, -0.789012, 0.345678, -0.901234, 0.567890, -0.123456, 0.789012, -0.345678, 0.0,
        1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.125,
    ];
    let dim = embedding.len();

    println!("Original embedding ({} floats, {} bytes):", dim, dim * 4);
    println!("  {:?}", &embedding[0..8]);
    println!("  {:?}", &embedding[8..16]);

    // 1. NoCompression - Baseline (no compression)
    println!("\n--- NoCompression (baseline) ---");
    let no_comp = NoCompression;
    let no_comp_bytes = no_comp.compress(&embedding);
    let no_comp_restored = no_comp.decompress(&no_comp_bytes, dim);

    println!(
        "  Compressed size: {} bytes ({}x)",
        no_comp_bytes.len(),
        no_comp_bytes.len() as f32 / (dim * 4) as f32
    );
    println!("  Lossless: {}", embedding == no_comp_restored);

    // 2. Fp16Compressor - Half-precision floating point
    // Reduces size by 50% with minimal precision loss
    println!("\n--- Fp16Compressor (half-precision) ---");
    let fp16 = Fp16Compressor;
    let fp16_bytes = fp16.compress(&embedding);
    let fp16_restored = fp16.decompress(&fp16_bytes, dim);

    println!(
        "  Compressed size: {} bytes ({:.2}x)",
        fp16_bytes.len(),
        fp16_bytes.len() as f32 / (dim * 4) as f32
    );

    let fp16_max_error: f32 = embedding
        .iter()
        .zip(fp16_restored.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    println!("  Max error: {:.6}", fp16_max_error);
    println!("  Restored: {:?}", &fp16_restored[0..8]);

    // 3. FixedR8Compressor - 8-bit quantization
    // Achieves ~4x compression with some precision loss
    println!("\n--- FixedR8Compressor (8-bit quantization) ---");
    let r8 = FixedR8Compressor;
    let r8_bytes = r8.compress(&embedding);
    let r8_restored = r8.decompress(&r8_bytes, dim);

    println!(
        "  Compressed size: {} bytes ({:.2}x)",
        r8_bytes.len(),
        r8_bytes.len() as f32 / (dim * 4) as f32
    );
    println!(
        "  Structure: 4 bytes scale + {} bytes quantized values",
        dim
    );

    let r8_max_error: f32 = embedding
        .iter()
        .zip(r8_restored.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    println!("  Max error: {:.6}", r8_max_error);
    println!("  Restored: {:?}", &r8_restored[0..8]);

    // 4. OneBitCompressor - Binary quantization
    // Maximum compression (32x) but only preserves sign
    println!("\n--- OneBitCompressor (binary quantization) ---");
    let one_bit = OneBitCompressor;
    let one_bit_bytes = one_bit.compress(&embedding);
    let one_bit_restored = one_bit.decompress(&one_bit_bytes, dim);

    println!(
        "  Compressed size: {} bytes ({:.2}x)",
        one_bit_bytes.len(),
        one_bit_bytes.len() as f32 / (dim * 4) as f32
    );
    println!("  Only preserves sign: positive -> +1.0, negative -> -1.0");
    println!("  Restored: {:?}", &one_bit_restored[0..8]);

    // Show sign preservation
    let signs_match = embedding
        .iter()
        .zip(one_bit_restored.iter())
        .all(|(orig, restored)| (*orig >= 0.0) == (*restored >= 0.0));
    println!("  All signs preserved: {}", signs_match);

    // Compression comparison summary
    println!("\n--- Compression Summary ---");
    println!("Original:    {} bytes (100%)", dim * 4);
    println!(
        "Fp16:        {} bytes ({:.1}%)",
        fp16_bytes.len(),
        100.0 * fp16_bytes.len() as f32 / (dim * 4) as f32
    );
    println!(
        "FixedR8:     {} bytes ({:.1}%)",
        r8_bytes.len(),
        100.0 * r8_bytes.len() as f32 / (dim * 4) as f32
    );
    println!(
        "OneBit:      {} bytes ({:.1}%)",
        one_bit_bytes.len(),
        100.0 * one_bit_bytes.len() as f32 / (dim * 4) as f32
    );

    // Use case example: storing compressed embeddings
    println!("\n--- Use Case: Compressed Storage ---");
    let large_dim = 256;
    let _large_embedding: Vec<f32> = (0..large_dim).map(|i| (i as f32 * 0.01) - 1.28).collect();

    let original_size = large_dim * 4;
    let fp16_size = fp16.compressed_size(large_dim);
    let r8_size = r8.compressed_size(large_dim);
    let one_bit_size = one_bit.compressed_size(large_dim);

    println!("For 1 million embeddings of dimension {}:", large_dim);
    println!(
        "  Original:  {:.2} GB",
        1_000_000.0 * original_size as f64 / 1e9
    );
    println!(
        "  Fp16:      {:.2} GB ({:.1}x compression)",
        1_000_000.0 * fp16_size as f64 / 1e9,
        original_size as f32 / fp16_size as f32
    );
    println!(
        "  FixedR8:   {:.2} GB ({:.1}x compression)",
        1_000_000.0 * r8_size as f64 / 1e9,
        original_size as f32 / r8_size as f32
    );
    println!(
        "  OneBit:    {:.2} GB ({:.1}x compression)",
        1_000_000.0 * one_bit_size as f64 / 1e9,
        original_size as f32 / one_bit_size as f32
    );
}

// ============================================================================
// Section 7: Benchmarking
// ============================================================================

/// Benchmarks various hash table operations.
fn benchmark_operations() {
    println!("\n=== Section 7: Performance Benchmarks ===\n");

    let iterations = 1000;
    let batch_size = 100;
    let dim = 64;

    // Prepare test data
    let ids: Vec<i64> = (0..batch_size as i64).collect();
    let embeddings: Vec<f32> = (0..batch_size * dim).map(|i| i as f32 * 0.001).collect();
    let gradients: Vec<f32> = vec![0.001; batch_size * dim];

    // 1. Benchmark lookup latency
    println!("--- Lookup Latency ---");
    let mut table = CuckooEmbeddingHashTable::new(batch_size * 10, dim);
    table.assign(&ids, &embeddings).unwrap();

    let mut output = vec![0.0; batch_size * dim];
    let start = Instant::now();
    for _ in 0..iterations {
        table.lookup(&ids, &mut output).unwrap();
    }
    let lookup_total = start.elapsed();
    let lookup_per_op = lookup_total / iterations as u32;
    let lookup_per_id = lookup_total / (iterations * batch_size) as u32;

    println!("  {} lookups of {} IDs each", iterations, batch_size);
    println!("  Total time: {:?}", lookup_total);
    println!("  Per batch: {:?}", lookup_per_op);
    println!("  Per ID: {:?}", lookup_per_id);

    // 2. Benchmark insert throughput
    println!("\n--- Insert Throughput ---");
    let start = Instant::now();
    for i in 0..iterations {
        let mut fresh_table = CuckooEmbeddingHashTable::new(batch_size * 2, dim);
        let offset = (i * batch_size) as i64;
        let batch_ids: Vec<i64> = (offset..offset + batch_size as i64).collect();
        fresh_table.assign(&batch_ids, &embeddings).unwrap();
    }
    let insert_total = start.elapsed();
    let inserts_per_sec = (iterations * batch_size) as f64 / insert_total.as_secs_f64();

    println!("  {} batches of {} inserts each", iterations, batch_size);
    println!("  Total time: {:?}", insert_total);
    println!("  Throughput: {:.0} inserts/sec", inserts_per_sec);

    // 3. Benchmark gradient application
    println!("\n--- Gradient Application ---");
    let mut grad_table = CuckooEmbeddingHashTable::with_learning_rate(batch_size * 10, dim, 0.01);
    grad_table.assign(&ids, &embeddings).unwrap();

    let start = Instant::now();
    for _ in 0..iterations {
        grad_table.apply_gradients(&ids, &gradients).unwrap();
    }
    let grad_total = start.elapsed();
    let grad_per_op = grad_total / iterations as u32;

    println!(
        "  {} gradient updates of {} IDs each",
        iterations, batch_size
    );
    println!("  Total time: {:?}", grad_total);
    println!("  Per batch: {:?}", grad_per_op);

    // 4. Compare initializer overhead
    println!("\n--- Initializer Overhead ---");
    let init_iterations = 10000;

    let zeros = ZerosInitializer;
    let start = Instant::now();
    for _ in 0..init_iterations {
        let _ = zeros.initialize(dim);
    }
    let zeros_time = start.elapsed();

    let uniform = RandomUniformInitializer::new(-0.1, 0.1);
    let start = Instant::now();
    for _ in 0..init_iterations {
        let _ = uniform.initialize(dim);
    }
    let uniform_time = start.elapsed();

    let normal = RandomNormalInitializer::new(0.0, 0.1);
    let start = Instant::now();
    for _ in 0..init_iterations {
        let _ = normal.initialize(dim);
    }
    let normal_time = start.elapsed();

    let xavier = XavierUniformInitializer::new(1.0);
    let start = Instant::now();
    for _ in 0..init_iterations {
        let _ = xavier.initialize(dim);
    }
    let xavier_time = start.elapsed();

    println!("  {} initializations of dimension {}", init_iterations, dim);
    println!("  ZerosInitializer:        {:?}", zeros_time);
    println!("  RandomUniformInitializer: {:?}", uniform_time);
    println!("  RandomNormalInitializer:  {:?}", normal_time);
    println!("  XavierUniformInitializer: {:?}", xavier_time);

    // 5. Compression benchmark
    println!("\n--- Compression Benchmark ---");
    let large_dim = 256;
    let large_embedding: Vec<f32> = (0..large_dim).map(|i| i as f32 * 0.01).collect();
    let compress_iterations = 10000;

    let fp16 = Fp16Compressor;
    let start = Instant::now();
    for _ in 0..compress_iterations {
        let compressed = fp16.compress(&large_embedding);
        let _ = fp16.decompress(&compressed, large_dim);
    }
    let fp16_time = start.elapsed();

    let r8 = FixedR8Compressor;
    let start = Instant::now();
    for _ in 0..compress_iterations {
        let compressed = r8.compress(&large_embedding);
        let _ = r8.decompress(&compressed, large_dim);
    }
    let r8_time = start.elapsed();

    println!(
        "  {} compress+decompress cycles, dimension {}",
        compress_iterations, large_dim
    );
    println!("  Fp16Compressor:    {:?}", fp16_time);
    println!("  FixedR8Compressor: {:?}", r8_time);

    // 6. Memory efficiency comparison
    println!("\n--- Memory Efficiency ---");
    let num_entries = 10000;

    let mut table_no_opt = CuckooEmbeddingHashTable::new(num_entries, dim);
    let all_ids: Vec<i64> = (0..num_entries as i64).collect();
    let all_embeddings: Vec<f32> = vec![0.1; num_entries * dim];
    table_no_opt.assign(&all_ids, &all_embeddings).unwrap();

    println!("  {} entries, dimension {}", num_entries, dim);
    println!(
        "  Memory usage: {} bytes ({:.2} KB per entry)",
        table_no_opt.memory_usage(),
        table_no_opt.memory_usage() as f64 / num_entries as f64 / 1024.0
    );
    println!(
        "  Theoretical minimum: {} KB",
        (num_entries * dim * 4) / 1024
    );
}

// ============================================================================
// Main Entry Point
// ============================================================================

fn main() {
    println!("========================================");
    println!("Monolith Embedding Hash Table Examples");
    println!("========================================");

    // Run all demonstrations
    demonstrate_basic_operations();
    demonstrate_initializers();
    demonstrate_optimizers();
    demonstrate_multi_hash_table();
    demonstrate_eviction();
    demonstrate_compression();
    benchmark_operations();

    println!("\n========================================");
    println!("All examples completed successfully!");
    println!("========================================");
}
