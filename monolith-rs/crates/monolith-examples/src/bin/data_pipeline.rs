// Copyright 2022 ByteDance and/or its affiliates.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Comprehensive Data Pipeline Example for Monolith-RS
//!
//! This example demonstrates the full data pipeline capabilities of monolith-rs,
//! including TFRecord reading/writing, transforms, Instance parsing, negative
//! sampling, compression, and FID utilities.
//!
//! Run with:
//! ```bash
//! cargo run --example data_pipeline --features compression
//! ```
//!
//! Or with custom arguments:
//! ```bash
//! cargo run --example data_pipeline --features compression -- \
//!     --input-file data/train.tfrecord \
//!     --batch-size 32 \
//!     --num-batches 10
//! ```

use std::collections::HashMap;
use std::io::Cursor;
use std::path::PathBuf;

use monolith_data::{
    // Example utilities
    add_feature,
    compress,
    create_example,
    decompress,
    extract_feature,
    get_feature_data,
    extract_slot,
    feature_names,
    get_feature,
    has_feature,
    make_fid,
    total_fid_count,
    // Batch utilities
    Batch,
    // Compression
    CompressionType,
    // Core dataset types
    Dataset,
    FilterTransform,
    FrequencyNegativeSampler,
    // Instance format
    Instance,
    InstanceBatch,
    InstanceParser,
    MapTransform,
    // Negative sampling
    NegativeSampler,
    NegativeSamplingConfig,
    SamplingStrategy,
    // TFRecord support
    TFRecordReader,
    TFRecordWriter,
    // Transforms
    TransformChain,
    UniformNegativeSampler,
    VecDataset,
};

// =============================================================================
// Command-line Arguments
// =============================================================================

/// Command-line configuration for the data pipeline example.
struct Config {
    /// Path to input TFRecord file. If None, synthetic data will be generated.
    input_file: Option<PathBuf>,
    /// Batch size for processing.
    batch_size: usize,
    /// Number of batches to process (0 = all).
    num_batches: usize,
    /// Whether to show verbose output.
    verbose: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            input_file: None,
            batch_size: 16,
            num_batches: 5,
            verbose: true,
        }
    }
}

fn parse_args() -> Config {
    let args: Vec<String> = std::env::args().collect();
    let mut config = Config::default();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--input-file" | "-i" => {
                if i + 1 < args.len() {
                    config.input_file = Some(PathBuf::from(&args[i + 1]));
                    i += 1;
                }
            }
            "--batch-size" | "-b" => {
                if i + 1 < args.len() {
                    config.batch_size = args[i + 1].parse().unwrap_or(16);
                    i += 1;
                }
            }
            "--num-batches" | "-n" => {
                if i + 1 < args.len() {
                    config.num_batches = args[i + 1].parse().unwrap_or(5);
                    i += 1;
                }
            }
            "--quiet" | "-q" => {
                config.verbose = false;
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }

    config
}

fn print_usage() {
    println!("Monolith-RS Data Pipeline Example");
    println!();
    println!("Usage: data_pipeline [OPTIONS]");
    println!();
    println!("Options:");
    println!(
        "  -i, --input-file <FILE>   Input TFRecord file (generates synthetic if not provided)"
    );
    println!("  -b, --batch-size <SIZE>   Batch size for processing (default: 16)");
    println!("  -n, --num-batches <NUM>   Number of batches to process, 0 for all (default: 5)");
    println!("  -q, --quiet               Suppress verbose output");
    println!("  -h, --help                Show this help message");
}

// =============================================================================
// Main Entry Point
// =============================================================================

fn main() {
    let config = parse_args();

    println!("{}", "=".repeat(80));
    println!("Monolith-RS Data Pipeline Tutorial");
    println!("{}", "=".repeat(80));
    println!();

    // Section 1: TFRecord Reading and Writing
    demo_tfrecord_io(&config);

    // Section 2: Dataset Transforms
    demo_transforms(&config);

    // Section 3: Instance Format
    demo_instance_format(&config);

    // Section 4: Negative Sampling
    demo_negative_sampling(&config);

    // Section 5: Compression
    demo_compression();

    // Section 6: FID Utilities
    demo_fid_utilities();

    // Section 7: Pipeline Composition
    demo_pipeline_composition(&config);

    // Section 8: Full Pipeline with Batching
    demo_full_pipeline(&config);

    println!();
    println!("{}", "=".repeat(80));
    println!("Tutorial Complete!");
    println!("{}", "=".repeat(80));
}

// =============================================================================
// Section 1: TFRecord Reading and Writing
// =============================================================================

fn demo_tfrecord_io(config: &Config) {
    section_header("1. TFRecord Reading and Writing");

    // Create synthetic TFRecord data
    println!("Creating synthetic TFRecord data in memory...");

    let mut buffer = Vec::new();
    let num_examples = 100;

    // Write synthetic examples to buffer
    {
        let mut writer = TFRecordWriter::new(&mut buffer);

        for i in 0..num_examples {
            let mut example = create_example();

            // User features (slot 1)
            let user_id = make_fid(1, 1000 + (i % 50) as i64);
            add_feature(&mut example, "user_id", vec![user_id], vec![1.0]);

            // Item features (slot 2)
            let item_id = make_fid(2, 5000 + (i % 100) as i64);
            add_feature(&mut example, "item_id", vec![item_id], vec![1.0]);

            // Context features (slot 3)
            let hour = make_fid(3, (i % 24) as i64);
            let day = make_fid(3, 100 + (i % 7) as i64);
            add_feature(&mut example, "context", vec![hour, day], vec![1.0, 1.0]);

            // Label (click/no-click)
            let label = if i % 3 == 0 { 1.0 } else { 0.0 };
            add_feature(&mut example, "label", vec![(label as i64)], vec![label]);

            // Dense features (embeddings)
            let embedding: Vec<f32> = (0..4).map(|j| (i + j) as f32 * 0.1).collect();
            add_feature(&mut example, "embedding", vec![], embedding);

            writer
                .write_example(&example)
                .expect("Failed to write example");
        }
        writer.flush().expect("Failed to flush writer");
    }

    println!(
        "  Written {} examples ({} bytes)",
        num_examples,
        buffer.len()
    );

    // Read examples back
    println!("\nReading TFRecord data from buffer...");
    let mut reader = TFRecordReader::new(Cursor::new(&buffer), true);

    let mut read_count = 0;
    let mut sample_example = None;

    while let Ok(Some(example)) = reader.read_example() {
        if read_count == 0 {
            sample_example = Some(example.clone());
        }
        read_count += 1;
    }

    println!("  Read {} examples", read_count);

    // Show sample example structure
    if let Some(example) = sample_example {
        if config.verbose {
            println!("\nSample example structure:");
            println!("  Features: {:?}", feature_names(&example));
            println!("  Total FIDs: {}", total_fid_count(&example));

            if let Some(d) = get_feature_data(&example, "user_id") {
                println!("  user_id FID: {:?}", d.fid);
            }
        }
    }

    // Demonstrate iteration and batching
    println!(
        "\nIterating with batching (batch_size={}):",
        config.batch_size
    );
    let examples: Vec<_> = {
        let mut reader = TFRecordReader::new(Cursor::new(&buffer), true);
        let mut examples = Vec::new();
        while let Ok(Some(example)) = reader.read_example() {
            examples.push(example);
        }
        examples
    };

    let dataset = VecDataset::new(examples);
    let mut batch_count = 0;

    for batch in dataset.batch(config.batch_size).iter() {
        batch_count += 1;
        if config.verbose && batch_count <= 3 {
            println!("  Batch {}: {} examples", batch_count, batch.len());
        }
    }
    println!("  Total batches: {}", batch_count);
}

// =============================================================================
// Section 2: Dataset Transforms
// =============================================================================

fn demo_transforms(config: &Config) {
    section_header("2. Dataset Transforms");

    // Create test data
    let examples = create_synthetic_examples(50);
    println!("Created {} synthetic examples", examples.len());

    // 2.1: Map Transform
    println!("\n2.1 Map Transform (add timestamp feature):");
    let dataset = VecDataset::new(examples.clone());
    let mapped: Vec<_> = dataset
        .map(|mut ex| {
            // Add a timestamp feature
            let timestamp = 1700000000i64;
            add_feature(&mut ex, "timestamp", vec![timestamp], vec![1.0]);
            ex
        })
        .iter()
        .collect();

    let has_timestamp = mapped.iter().all(|ex| has_feature(ex, "timestamp"));
    println!("  All examples have timestamp: {}", has_timestamp);

    // 2.2: Filter Transform
    println!("\n2.2 Filter Transform (keep positive labels only):");
    let dataset = VecDataset::new(examples.clone());
    let positive_only: Vec<_> = dataset
        .filter(|ex| {
            get_feature(ex, "label")
                .and_then(|_| get_feature_data(ex, "label"))
                .map(|d| d.value.first().copied().unwrap_or(0.0) > 0.5)
                .unwrap_or(false)
        })
        .iter()
        .collect();

    println!("  Original: {} examples", examples.len());
    println!("  After filter: {} examples", positive_only.len());

    // 2.3: Shuffle Transform
    println!("\n2.3 Shuffle Transform (buffer_size=20):");
    let dataset = VecDataset::new(examples.clone());
    let shuffled: Vec<_> = dataset.shuffle(20).iter().take(10).collect();

    println!("  First 10 shuffled examples collected");
    if config.verbose {
        let first_ids: Vec<_> = shuffled
            .iter()
            .filter_map(|ex| get_feature_data(ex, "user_id"))
            .filter_map(|d| d.fid.first().copied())
            .take(5)
            .collect();
        println!("  First 5 user_ids: {:?}", first_ids);
    }

    // 2.4: Batch Transform
    println!("\n2.4 Batch Transform (batch_size={}):", config.batch_size);
    let dataset = VecDataset::new(examples.clone());
    let batches: Vec<Batch> = dataset.batch(config.batch_size).iter().collect();

    println!("  Number of batches: {}", batches.len());
    if !batches.is_empty() {
        println!("  First batch size: {}", batches[0].len());
        println!("  Last batch size: {}", batches.last().unwrap().len());
    }

    // 2.5: TransformChain
    println!("\n2.5 Transform Chain (filter + map):");

    let chain = TransformChain::new()
        .add(
            FilterTransform::new(|ex| {
                // Keep examples with even user IDs
                get_feature_data(ex, "user_id")
                    .and_then(|d| d.fid.first().copied())
                    .map(|fid| extract_feature(fid) % 2 == 0)
                    .unwrap_or(false)
            })
            .with_name("EvenUserFilter"),
        )
        .add(
            MapTransform::new(|mut ex| {
                add_feature(&mut ex, "processed", vec![1], vec![1.0]);
                ex
            })
            .with_name("AddProcessedFlag"),
        );

    println!("  Chain length: {} transforms", chain.len());

    // Apply chain to dataset
    let dataset = VecDataset::new(examples.clone());
    let processed: Vec<_> = dataset.transform(chain).iter().collect();

    println!("  Original: {} examples", examples.len());
    println!("  After chain: {} examples", processed.len());

    // Verify all processed examples have the flag
    let all_processed = processed.iter().all(|ex| has_feature(ex, "processed"));
    println!("  All have 'processed' flag: {}", all_processed);
}

// =============================================================================
// Section 3: Instance Format
// =============================================================================

fn demo_instance_format(_config: &Config) {
    section_header("3. Instance Format");

    // 3.1: Create Instance manually
    println!("3.1 Create Instance manually:");

    let mut instance = Instance::new();

    // Add sparse features
    instance.add_sparse_feature("user_id", vec![make_fid(1, 12345)], vec![1.0]);
    instance.add_sparse_feature("item_id", vec![make_fid(2, 67890)], vec![1.0]);
    instance.add_sparse_feature(
        "tags",
        vec![make_fid(3, 100), make_fid(3, 101), make_fid(3, 102)],
        vec![0.8, 0.5, 0.3],
    );

    // Add dense features
    instance.add_dense_feature("user_embedding", vec![0.1, 0.2, 0.3, 0.4]);
    instance.add_dense_feature("item_embedding", vec![0.5, 0.6, 0.7, 0.8]);

    // Set label and weight
    instance.set_label(vec![1.0]);
    instance.set_instance_weight(1.5);

    println!("  Sparse features: {:?}", instance.sparse_feature_names());
    println!("  Dense features: {:?}", instance.dense_feature_names());
    println!("  Total FIDs: {}", instance.total_fid_count());
    println!("  All slots: {:?}", instance.all_slots());
    println!("  Label: {:?}", instance.label());
    println!("  Instance weight: {}", instance.instance_weight());

    // 3.2: Parse from JSON
    println!("\n3.2 Parse Instance from JSON:");

    let json = r#"{
        "sparse_features": {
            "user_id": {"fids": [100], "values": [1.0]},
            "item_id": {"fids": [200], "values": [1.0]},
            "categories": {"fids": [300, 301, 302], "values": [0.9, 0.7, 0.5]}
        },
        "dense_features": {
            "position_embedding": [0.1, 0.2, 0.3, 0.4, 0.5]
        },
        "label": [1.0, 0.0],
        "instance_weight": 2.0
    }"#;

    let parser = InstanceParser::new();
    match parser.parse_from_json(json) {
        Ok(parsed_instance) => {
            println!("  Parsed successfully!");
            println!(
                "  Sparse features: {:?}",
                parsed_instance.sparse_feature_names()
            );
            println!(
                "  Dense features: {:?}",
                parsed_instance.dense_feature_names()
            );
            println!("  Label: {:?}", parsed_instance.label());
        }
        Err(e) => {
            println!("  Parse error: {}", e);
        }
    }

    // 3.3: Parse from CSV
    println!("\n3.3 Parse Instance from CSV:");

    let csv_line = "1.0,100:0.5,200:0.8,300:0.3";
    match parser.parse_from_csv(csv_line, ',') {
        Ok(csv_instance) => {
            println!("  CSV: {}", csv_line);
            println!("  Label: {:?}", csv_instance.label());
            if let Some(features) = csv_instance.get_sparse_feature("features") {
                println!("  FIDs: {:?}", features.fids);
                println!("  Values: {:?}", features.values);
            }
        }
        Err(e) => {
            println!("  Parse error: {}", e);
        }
    }

    // 3.4: Convert to tensor dictionary
    println!("\n3.4 Convert Instance to Tensor Dictionary:");

    let tensor_dict = instance.to_tensor_dict();
    println!("  Keys: {:?}", tensor_dict.keys().collect::<Vec<_>>());

    if let Some(user_fids) = tensor_dict.get("user_id_fids") {
        if let Some(fids) = user_fids.as_int64() {
            println!("  user_id_fids: {:?}", fids);
        }
    }

    if let Some(embedding) = tensor_dict.get("user_embedding") {
        if let Some(values) = embedding.as_float() {
            println!("  user_embedding: {:?}", values);
        }
    }

    // 3.5: InstanceBatch
    println!("\n3.5 InstanceBatch for batch processing:");

    let mut batch = InstanceBatch::new();
    for i in 0..3 {
        let mut inst = Instance::new();
        inst.add_sparse_feature("user_id", vec![make_fid(1, i as i64)], vec![1.0]);
        inst.add_sparse_feature("item_id", vec![make_fid(2, i as i64 * 10)], vec![1.0]);
        inst.set_label(vec![if i % 2 == 0 { 1.0 } else { 0.0 }]);
        batch.push(inst);
    }

    println!("  Batch size: {}", batch.len());

    let batch_tensors = batch.to_tensor_dict();
    println!(
        "  Batch tensor keys: {:?}",
        batch_tensors.keys().collect::<Vec<_>>()
    );

    if let Some(batch_labels) = batch_tensors.get("label") {
        if let Some(labels) = batch_labels.as_float() {
            println!("  Batch labels: {:?}", labels);
        }
    }

    // 3.6: Convert Instance to Example proto
    println!("\n3.6 Convert Instance to Example proto:");

    let example = instance.to_example();
    println!(
        "  Named features in Example: {}",
        example.named_feature.len()
    );
    println!("  Feature names: {:?}", feature_names(&example));
}

// =============================================================================
// Section 4: Negative Sampling
// =============================================================================

fn demo_negative_sampling(_config: &Config) {
    section_header("4. Negative Sampling");

    // 4.1: Uniform Negative Sampler
    println!("4.1 Uniform Negative Sampler:");

    // Create vocabulary of item IDs
    let item_pool: Vec<i64> = (10000..10100).collect();
    println!("  Item pool size: {}", item_pool.len());

    let uniform_sampler = UniformNegativeSampler::new(item_pool.clone(), true)
        .with_item_feature_name("item_id")
        .with_seed(42);

    // Create a positive example
    let mut positive = create_example();
    add_feature(&mut positive, "user_id", vec![1001], vec![1.0]);
    add_feature(&mut positive, "item_id", vec![10050], vec![1.0]);
    add_feature(&mut positive, "label", vec![1], vec![1.0]);

    // Generate negative samples
    let negatives = uniform_sampler.sample(&positive, 5);

    println!("  Generated {} negative samples", negatives.len());

    for (i, neg) in negatives.iter().enumerate() {
        if let (Some(item), Some(label)) =
            (get_feature_data(neg, "item_id"), get_feature_data(neg, "label"))
        {
            println!(
                "    Negative {}: item_id={:?}, label={:?}",
                i,
                item.fid.first().copied(),
                label.value.first().copied()
            );
        }
    }

    // 4.2: Frequency-based Negative Sampler
    println!("\n4.2 Frequency-based Negative Sampler:");

    // Create item frequencies (power-law distribution)
    let mut item_frequencies: HashMap<i64, f32> = HashMap::new();
    for (i, &item_id) in item_pool.iter().enumerate() {
        // Higher frequency for lower indices (head items)
        let freq = 1000.0 / (i + 1) as f32;
        item_frequencies.insert(item_id, freq);
    }

    let freq_sampler = FrequencyNegativeSampler::new(item_frequencies, 0.75)
        .with_item_feature_name("item_id")
        .with_seed(123);

    let freq_negatives = freq_sampler.sample(&positive, 5);
    println!(
        "  Generated {} frequency-based negatives",
        freq_negatives.len()
    );

    // Count samples from head vs tail items
    let mut head_count = 0;
    let mut tail_count = 0;
    for neg in &freq_negatives {
        if let Some(item_id) = get_feature_data(neg, "item_id").and_then(|d| d.fid.first().copied())
        {
            if item_id < 10020 {
                head_count += 1;
            } else {
                tail_count += 1;
            }
        }
    }
    println!(
        "  Head items (high freq): {}, Tail items: {}",
        head_count, tail_count
    );

    // 4.3: Negative Sampling Configuration
    println!("\n4.3 Negative Sampling Configuration:");

    let config = NegativeSamplingConfig::new(10)
        .with_strategy(SamplingStrategy::Frequency)
        .with_temperature(0.5)
        .with_replacement(false)
        .with_item_feature_name("product_id");

    println!("  Num negatives: {}", config.num_negatives);
    println!("  Strategy: {:?}", config.sampling_strategy);
    println!("  Temperature: {}", config.temperature);
    println!("  With replacement: {}", config.sample_with_replacement);
    println!("  Item feature: {}", config.item_feature_name);

    // 4.4: Apply to dataset
    println!("\n4.4 Apply negative sampling to dataset:");

    let examples = create_synthetic_examples(10);
    let dataset = VecDataset::new(examples);

    let sampler = Box::new(UniformNegativeSampler::new(item_pool, true).with_seed(42));

    let neg_dataset = monolith_data::NegativeSamplingDataset::new(dataset, 3, sampler);
    let all_examples: Vec<_> = neg_dataset.iter().collect();

    // Count positives and negatives
    let positive_count = all_examples
        .iter()
        .filter(|ex| {
            get_feature_data(ex, "label")
                .map(|d| d.value.first().copied().unwrap_or(0.0) == 1.0)
                .unwrap_or(false)
        })
        .count();

    let negative_count = all_examples.len() - positive_count;

    println!("  Total examples: {}", all_examples.len());
    println!("  Positives: {}", positive_count);
    println!("  Negatives: {}", negative_count);
}

// =============================================================================
// Section 5: Compression
// =============================================================================

fn demo_compression() {
    section_header("5. Compression");

    // Create sample data
    let original_data = b"This is sample data for compression testing. ".repeat(100);
    println!("Original data size: {} bytes", original_data.len());

    // 5.1: Compression type detection
    println!("\n5.1 Compression type detection from file extension:");

    let test_paths = [
        "data/train.tfrecord",
        "data/train.tfrecord.gz",
        "data/train.tfrecord.snappy",
        "data/train.tfrecord.zlib",
    ];

    for path in test_paths {
        let compression = CompressionType::from_extension(path);
        println!("  {} -> {:?}", path, compression);
    }

    // 5.2: Gzip compression (if feature enabled)
    println!("\n5.2 Compression options:");

    // Try each compression type
    for compression_type in [
        CompressionType::None,
        CompressionType::Gzip,
        CompressionType::Zlib,
        CompressionType::Snappy,
    ] {
        match compress(&original_data, compression_type) {
            Ok(compressed) => {
                let ratio = compressed.len() as f64 / original_data.len() as f64;
                print!(
                    "  {:?}: {} bytes (ratio: {:.2})",
                    compression_type,
                    compressed.len(),
                    ratio
                );

                // Verify round-trip
                match decompress(&compressed, compression_type) {
                    Ok(decompressed) => {
                        if decompressed == original_data {
                            println!(" [verified]");
                        } else {
                            println!(" [MISMATCH]");
                        }
                    }
                    Err(e) => println!(" [decompress error: {}]", e),
                }
            }
            Err(e) => println!("  {:?}: not available ({})", compression_type, e),
        }
    }

    // 5.3: TFRecord with compression
    println!("\n5.3 TFRecord with compression:");

    let examples = create_synthetic_examples(50);

    // Write uncompressed
    let mut uncompressed_buffer = Vec::new();
    {
        let mut writer = TFRecordWriter::new(&mut uncompressed_buffer);
        for example in &examples {
            writer.write_example(example).expect("Failed to write");
        }
    }
    println!(
        "  Uncompressed TFRecord: {} bytes",
        uncompressed_buffer.len()
    );

    // Write with gzip compression (if available)
    let mut compressed_buffer = Vec::new();
    {
        let mut writer =
            TFRecordWriter::new(&mut compressed_buffer).with_compression(CompressionType::Gzip);

        for example in &examples {
            match writer.write_example(example) {
                Ok(_) => {}
                Err(e) => {
                    println!("  Gzip compression not available: {}", e);
                    return;
                }
            }
        }
    }

    if !compressed_buffer.is_empty() {
        let ratio = compressed_buffer.len() as f64 / uncompressed_buffer.len() as f64;
        println!(
            "  Gzip compressed TFRecord: {} bytes (ratio: {:.2})",
            compressed_buffer.len(),
            ratio
        );

        // Verify we can read it back
        let mut reader = TFRecordReader::new(Cursor::new(&compressed_buffer), true)
            .with_compression(CompressionType::Gzip);

        let mut read_count = 0;
        while let Ok(Some(_)) = reader.read_example() {
            read_count += 1;
        }
        println!("  Read back {} examples", read_count);
    }
}

// =============================================================================
// Section 6: FID Utilities
// =============================================================================

fn demo_fid_utilities() {
    section_header("6. FID Utilities");

    // 6.1: FID structure
    println!("6.1 FID structure (slot + feature):");
    println!("  FID = (slot << 54) | (feature & FEATURE_MASK)");
    println!("  Upper 10 bits: slot ID (0-1023)");
    println!("  Lower 54 bits: feature hash");

    // 6.2: Create FIDs with make_fid
    println!("\n6.2 Create FIDs with make_fid:");

    let test_cases = [
        (1, 12345i64, "user_id"),
        (2, 67890, "item_id"),
        (3, 100, "hour_of_day"),
        (10, 999_999_999, "large_feature"),
        (100, 0, "zero_feature"),
    ];

    for (slot, feature, description) in test_cases {
        let fid = make_fid(slot, feature);
        println!(
            "  {}: make_fid({}, {}) = {}",
            description, slot, feature, fid
        );
    }

    // 6.3: Extract slot and feature
    println!("\n6.3 Extract slot and feature from FID:");

    for (slot, feature, _description) in test_cases {
        let fid = make_fid(slot, feature);
        let extracted_slot = extract_slot(fid);
        let extracted_feature = extract_feature(fid);

        let slot_match = extracted_slot == slot;
        let feature_match = extracted_feature == feature;

        println!(
            "  FID {}: slot={} ({}), feature={} ({})",
            fid,
            extracted_slot,
            if slot_match { "OK" } else { "MISMATCH" },
            extracted_feature,
            if feature_match { "OK" } else { "MISMATCH" }
        );
    }

    // 6.4: Slot-based feature grouping
    println!("\n6.4 Slot-based feature grouping:");

    // Create an example with features from multiple slots
    let mut example = create_example();

    // Slot 1: User features
    add_feature(
        &mut example,
        "user_id",
        vec![make_fid(1, 100), make_fid(1, 101)],
        vec![1.0, 1.0],
    );

    // Slot 2: Item features
    add_feature(&mut example, "item_id", vec![make_fid(2, 200)], vec![1.0]);

    // Slot 3: Context features
    add_feature(
        &mut example,
        "context",
        vec![make_fid(3, 300), make_fid(3, 301), make_fid(3, 302)],
        vec![0.8, 0.5, 0.3],
    );

    // Slot 4: Category features
    add_feature(
        &mut example,
        "categories",
        vec![make_fid(4, 400), make_fid(4, 401)],
        vec![1.0, 1.0],
    );

    // Group FIDs by slot
    let mut slot_groups: HashMap<i32, Vec<i64>> = HashMap::new();

    for nf in &example.named_feature {
        if let Some(feature) = &nf.feature {
            let d = monolith_data::example::extract_feature_data(feature);
            for fid in d.fid {
                let slot = extract_slot(fid);
                slot_groups.entry(slot).or_default().push(fid);
            }
        }
    }

    println!("  FIDs grouped by slot:");
    let mut slots: Vec<_> = slot_groups.keys().copied().collect();
    slots.sort();
    for slot in slots {
        let fids = &slot_groups[&slot];
        let features: Vec<_> = fids.iter().map(|&f| extract_feature(f)).collect();
        println!(
            "    Slot {}: {} FIDs, features: {:?}",
            slot,
            fids.len(),
            features
        );
    }

    // 6.5: FID statistics
    println!("\n6.5 FID statistics:");

    let all_fids: Vec<i64> = example
        .named_feature
        .iter()
        .filter_map(|nf| nf.feature.as_ref())
        .flat_map(|f| monolith_data::example::extract_feature_data(f).fid.into_iter())
        .collect();

    println!("  Total FIDs in example: {}", all_fids.len());
    println!(
        "  Unique slots: {:?}",
        slot_groups.keys().collect::<Vec<_>>()
    );

    // Min/max feature hash per slot
    for (slot, fids) in &slot_groups {
        let features: Vec<_> = fids.iter().map(|&f| extract_feature(f)).collect();
        let min_feature = features.iter().min().unwrap();
        let max_feature = features.iter().max().unwrap();
        println!(
            "    Slot {}: feature range [{}, {}]",
            slot, min_feature, max_feature
        );
    }
}

// =============================================================================
// Section 7: Pipeline Composition
// =============================================================================

fn demo_pipeline_composition(config: &Config) {
    section_header("7. Pipeline Composition");

    // 7.1: Lazy evaluation pattern
    println!("7.1 Lazy evaluation pattern:");

    let examples = create_synthetic_examples(100);
    println!("  Created {} source examples", examples.len());

    // Build a complex pipeline (nothing executes yet)
    let pipeline = VecDataset::new(examples.clone())
        // Filter: keep positive examples
        .filter(|ex| {
            get_feature(ex, "label")
                .map(|f| f.value.first().copied().unwrap_or(0.0) > 0.5)
                .unwrap_or(false)
        })
        // Map: add processing timestamp
        .map(|mut ex| {
            add_feature(&mut ex, "processed_at", vec![1700000000], vec![1.0]);
            ex
        })
        // Shuffle
        .shuffle(20)
        // Take first N
        .take(30);

    println!("  Pipeline built (lazy - not executed yet)");

    // Execute by consuming the iterator
    let results: Vec<_> = pipeline.iter().collect();
    println!("  Pipeline executed: {} examples produced", results.len());

    // Verify all results have the processed_at feature
    let all_processed = results.iter().all(|ex| has_feature(ex, "processed_at"));
    println!("  All have 'processed_at': {}", all_processed);

    // 7.2: Chaining transforms
    println!("\n7.2 Chaining transforms with TransformChain:");

    // Create a reusable transform chain
    let preprocessing_chain = TransformChain::new()
        .add(
            FilterTransform::new(|ex| {
                // Remove examples with no user_id
                has_feature(ex, "user_id")
            })
            .with_name("RequireUserId"),
        )
        .add(
            FilterTransform::new(|ex| {
                // Remove examples with no item_id
                has_feature(ex, "item_id")
            })
            .with_name("RequireItemId"),
        )
        .add(
            MapTransform::new(|mut ex| {
                // Normalize: ensure label exists
                if !has_feature(&ex, "label") {
                    add_feature(&mut ex, "label", vec![0], vec![0.0]);
                }
                ex
            })
            .with_name("EnsureLabel"),
        );

    println!("  Transform chain has {} steps", preprocessing_chain.len());

    // Apply chain to dataset
    let dataset = VecDataset::new(examples.clone());
    let preprocessed: Vec<_> = dataset.transform(preprocessing_chain).iter().collect();
    println!("  After preprocessing: {} examples", preprocessed.len());

    // 7.3: Combine with batching
    println!("\n7.3 Combine transforms with batching:");

    let dataset = VecDataset::new(examples.clone())
        .filter(|ex| {
            get_feature(ex, "label")
                .map(|f| f.value.first().copied().unwrap_or(0.0) > 0.5)
                .unwrap_or(false)
        })
        .map(|mut ex| {
            add_feature(&mut ex, "is_positive", vec![1], vec![1.0]);
            ex
        })
        .batch(config.batch_size);

    let batches: Vec<Batch> = dataset.iter().collect();
    println!(
        "  Produced {} batches of size {}",
        batches.len(),
        config.batch_size
    );

    // 7.4: Skip and Take patterns
    println!("\n7.4 Skip and Take patterns:");

    // Pagination pattern: skip first N, take next M
    let page_size = 10;
    let page_number = 2;
    let skip_count = page_number * page_size;

    let page = VecDataset::new(examples.clone())
        .skip(skip_count)
        .take(page_size)
        .iter()
        .collect::<Vec<_>>();

    println!(
        "  Page {} (skip {}, take {}): {} examples",
        page_number,
        skip_count,
        page_size,
        page.len()
    );

    // 7.5: Collect statistics during iteration
    println!("\n7.5 Collect statistics during iteration:");

    let mut positive_count = 0;
    let mut negative_count = 0;
    let mut total_fids = 0;

    for example in VecDataset::new(examples.clone()).iter() {
        let label = get_feature(&example, "label")
            .and_then(|f| f.value.first().copied())
            .unwrap_or(0.0);

        if label > 0.5 {
            positive_count += 1;
        } else {
            negative_count += 1;
        }

        total_fids += total_fid_count(&example);
    }

    println!("  Statistics:");
    println!("    Positive examples: {}", positive_count);
    println!("    Negative examples: {}", negative_count);
    println!(
        "    Positive ratio: {:.2}%",
        100.0 * positive_count as f64 / (positive_count + negative_count) as f64
    );
    println!("    Total FIDs: {}", total_fids);
    println!(
        "    Avg FIDs per example: {:.1}",
        total_fids as f64 / examples.len() as f64
    );
}

// =============================================================================
// Section 8: Full Pipeline Demo
// =============================================================================

fn demo_full_pipeline(config: &Config) {
    section_header("8. Full Pipeline Demo");

    println!("Building end-to-end training data pipeline...");
    println!("  Batch size: {}", config.batch_size);
    println!(
        "  Num batches: {}",
        if config.num_batches == 0 {
            "all".to_string()
        } else {
            config.num_batches.to_string()
        }
    );

    // Step 1: Create or load data
    println!("\nStep 1: Create synthetic training data");
    let raw_examples = create_synthetic_examples(200);
    println!("  Generated {} raw examples", raw_examples.len());

    // Step 2: Define preprocessing transforms
    println!("\nStep 2: Define preprocessing transforms");
    let preprocessing = TransformChain::new()
        // Validate required features
        .add(
            FilterTransform::new(|ex| has_feature(ex, "user_id") && has_feature(ex, "item_id"))
                .with_name("ValidateFeatures"),
        )
        // Add derived features
        .add(
            MapTransform::new(|mut ex| {
                // Add cross feature (user_id x item_id)
                let user_fid = get_feature(&ex, "user_id")
                    .and_then(|f| f.fid.first().copied())
                    .unwrap_or(0);
                let item_fid = get_feature(&ex, "item_id")
                    .and_then(|f| f.fid.first().copied())
                    .unwrap_or(0);

                // Simple cross: combine slot 1 and slot 2 features into slot 5
                let cross_fid = make_fid(5, (user_fid ^ item_fid) & 0xFFFFFF);
                add_feature(&mut ex, "user_item_cross", vec![cross_fid], vec![1.0]);
                ex
            })
            .with_name("AddCrossFeature"),
        )
        // Add timestamp
        .add(
            MapTransform::new(|mut ex| {
                add_feature(&mut ex, "process_time", vec![1700000000], vec![1.0]);
                ex
            })
            .with_name("AddTimestamp"),
        );

    println!("  Preprocessing chain: {} transforms", preprocessing.len());

    // Step 3: Create negative sampling configuration
    println!("\nStep 3: Configure negative sampling");
    let item_pool: Vec<i64> = (10000..10500).collect();
    let neg_sampler = Box::new(UniformNegativeSampler::new(item_pool, true).with_seed(42));
    println!("  Item pool size: 500");
    println!("  Negatives per positive: 3");

    // Step 4: Build complete pipeline
    println!("\nStep 4: Build and execute pipeline");

    // Create preprocessing dataset
    // First apply the transform chain, then collect and shuffle
    let transformed: Vec<_> = VecDataset::new(raw_examples.clone())
        .transform(preprocessing)
        .iter()
        .collect();

    // Then shuffle the transformed data
    let preprocessed: Vec<_> = VecDataset::new(transformed).shuffle(50).iter().collect();

    println!("  After preprocessing: {} examples", preprocessed.len());

    // Apply negative sampling
    let neg_dataset = monolith_data::NegativeSamplingDataset::new(
        VecDataset::new(preprocessed),
        3, // 3 negatives per positive
        neg_sampler,
    );

    let with_negatives: Vec<_> = neg_dataset.iter().collect();
    println!(
        "  After negative sampling: {} examples",
        with_negatives.len()
    );

    // Step 5: Batch and process
    println!("\nStep 5: Batch and process");

    let batched = VecDataset::new(with_negatives).batch(config.batch_size);

    let mut batch_num = 0;
    let mut total_examples = 0;
    let mut total_positives = 0;
    let mut total_negatives = 0;

    for batch in batched.iter() {
        batch_num += 1;
        total_examples += batch.len();

        for example in batch.iter() {
            let label = get_feature(example, "label")
                .and_then(|f| f.value.first().copied())
                .unwrap_or(0.0);

            if label > 0.5 {
                total_positives += 1;
            } else {
                total_negatives += 1;
            }
        }

        if config.verbose && batch_num <= 3 {
            println!("  Batch {}: {} examples", batch_num, batch.len());
        }

        // Limit batches if configured
        if config.num_batches > 0 && batch_num >= config.num_batches {
            break;
        }
    }

    println!("\nPipeline Summary:");
    println!("  Total batches processed: {}", batch_num);
    println!("  Total examples: {}", total_examples);
    println!("  Positives: {}", total_positives);
    println!("  Negatives: {}", total_negatives);
    println!(
        "  Positive ratio: {:.2}%",
        100.0 * total_positives as f64 / total_examples.max(1) as f64
    );

    // Step 6: Convert final batch to Instance format
    println!("\nStep 6: Convert batch to Instance format");

    // Take a sample batch
    let sample_batch: Vec<_> = VecDataset::new(create_synthetic_examples(config.batch_size))
        .iter()
        .collect();

    let parser = InstanceParser::new();
    let mut instance_batch = InstanceBatch::new();

    for example in &sample_batch {
        let instance = parser.from_example(example);
        instance_batch.push(instance);
    }

    println!(
        "  Converted {} examples to InstanceBatch",
        instance_batch.len()
    );

    let batch_tensors = instance_batch.to_tensor_dict();
    println!(
        "  Tensor dict keys: {:?}",
        batch_tensors.keys().collect::<Vec<_>>()
    );
}

// =============================================================================
// Helper Functions
// =============================================================================

fn section_header(title: &str) {
    println!();
    println!("{}", "-".repeat(80));
    println!("{}", title);
    println!("{}", "-".repeat(80));
    println!();
}

fn create_synthetic_examples(count: usize) -> Vec<monolith_proto::Example> {
    (0..count)
        .map(|i| {
            let mut example = create_example();

            // User features (slot 1)
            let user_id = make_fid(1, 1000 + (i % 50) as i64);
            add_feature(&mut example, "user_id", vec![user_id], vec![1.0]);

            // Item features (slot 2)
            let item_id = make_fid(2, 5000 + (i % 100) as i64);
            add_feature(&mut example, "item_id", vec![item_id], vec![1.0]);

            // Context features (slot 3)
            let hour = make_fid(3, (i % 24) as i64);
            add_feature(&mut example, "hour", vec![hour], vec![1.0]);

            // Label
            let label = if i % 3 == 0 { 1.0 } else { 0.0 };
            add_feature(&mut example, "label", vec![(label as i64)], vec![label]);

            example
        })
        .collect()
}
