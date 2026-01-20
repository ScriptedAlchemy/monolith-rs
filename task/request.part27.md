<!--
Source: task/request.md
Lines: 6685-6900 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/data/kafka_dataset_test.py`
<a id="monolith-native-training-data-kafka-dataset-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 239
- Purpose/role: Integration test for KafkaDataset ingestion and label parsing.
- Key symbols/classes/functions: `start_producer`, `KafkaDatasetTest.test_kafka_dataset`.
- External dependencies: `kafka.KafkaProducer`, `tensorflow`, `KafkaDataset`, `parse_instances/parse_examples`, `add_label`.
- Side effects: produces Kafka messages to a real cluster; sleeps and joins producer thread.

**Required Behavior (Detailed)**
- Flags control Kafka connection, topic, and data generation.
- `start_producer(input_type)`:
  - Generates Example/Instance/ExampleBatch protos and writes to Kafka with length-prefixed encoding.
  - Uses hard-coded SASL credentials and sleeps 10s before production.
- `test_kafka_dataset(input_type, output_type)`:
  - Starts producer thread.
  - Creates `KafkaDataset` with given variant/output types.
  - Applies `add_label` with config string (click head optional).
  - Batches, parses into features, splits label vector into task labels.
  - Iterates for `num_batch` and prints results.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/tests`.
- Rust public API surface: Kafka dataset ingestion tests.
- Data model mapping: Kafka stream → dataset parser.
- Feature gating: Kafka support.
- Integration points: data pipeline.

**Implementation Steps (Detailed)**
1. Provide integration tests only in environments with Kafka.
2. Mock Kafka for unit tests to avoid hard-coded credentials.

**Tests (Detailed)**
- Python tests: this file (integration).
- Rust tests: TODO (manual)
- Cross-language parity test: TODO (manual)

**Gaps / Notes**
- TODO (manual)

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/data/multi_flow_test.py`
<a id="monolith-native-training-data-multi-flow-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 125
- Purpose/role: Tests split/merge flow on instance dataset using lagrangex headers.
- Key symbols/classes/functions: `MultiFlowTest.test_data_flow`.
- External dependencies: `tensorflow`, `PBDataset`, `parse_instances`, `Instance` proto.
- Side effects: writes/reads `data.pb` under `TEST_TMPDIR`.

**Required Behavior (Detailed)**
- `setUpClass`:
  - Generates `NUM_INSTANCE` Instance protos with random fids, line_id fields.
  - Writes lagrangex header and length-prefixed data to `data.pb`.
- `mk_kgx_header(dataflow)`:
  - Computes Java hash code for `dataflow`, writes 4-byte header.
- `test_data_flow`:
  - Reads dataset with `lagrangex_header=True`.
  - Splits into flows by device_types and merges back.
  - Parses instances and batches; expects 8 batches of size 512.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/tests`.
- Rust public API surface: split_flow / merge_flow dataset operations.
- Data model mapping: lagrangex header parsing.
- Feature gating: none.
- Integration points: data pipeline.

**Implementation Steps (Detailed)**
1. Add lagrangex header parsing and flow split/merge in Rust datasets.
2. Add test for split/merge on synthetic instance data.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: TODO (manual)
- Cross-language parity test: TODO (manual)

**Gaps / Notes**
- TODO (manual)

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/data/negative_gen_test.py`
<a id="monolith-native-training-data-negative-gen-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 253
- Purpose/role: Tests negative sampling generation for Instance/Example datasets.
- Key symbols/classes/functions: `NegativeGenTest.test_dataset_target`.
- External dependencies: `tensorflow`, `PBDataset`, `negative_gen`, `parse_instances/parse_examples`.
- Side effects: writes a temporary `{variant_type}.pb` file.

**Required Behavior (Detailed)**
- `setUpClass`:
  - Generates sample data with random FIDs and labels; writes length-prefixed protos.
  - Tracks per-channel pos/neg counts and per-gid counts.
- `test_dataset_target`:
  - Reads PBDataset and applies `negative_gen` with configured params:
    - `neg_num`, `start_num`, `max_item_num`, `cache_only_pos`, `per_channel`, `throw_origin`, `throw_origin_neg`.
  - Parses dataset and counts pos/neg labels; verifies counts and expected ranges.
  - Ensures total count equals pos+neg.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/tests`.
- Rust public API surface: negative sampling dataset transform.
- Data model mapping: `negative_gen` functionality in Rust.
- Feature gating: none.
- Integration points: dataset pipeline.

**Implementation Steps (Detailed)**
1. Implement negative sampling logic in Rust datasets.
2. Add tests for per-channel and non-channel sampling boundaries.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: TODO (manual)
- Cross-language parity test: TODO (manual)

**Gaps / Notes**
- TODO (manual)

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/data/parse_sparse_feature_test.py`
<a id="monolith-native-training-data-parse-sparse-feature-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 1833
- Purpose/role: Validates sparse feature sharding logic and fused layout parsing across ExampleBatch/Example/Instance.
- Key symbols/classes/functions: `DataOpsV2Test`, `DataOpsV2TestFitPreV2`, `DataOpsV2Testv4`, `DataOpsV2TestFitPre`.
- External dependencies: `tensorflow`, `parse_instances/parse_examples/parse_example_batch`, `sharding_sparse_fids`, proto `FeatureConfigs`.
- Side effects: reads training_instance fixtures; prints debug output.

**Required Behavior (Detailed)**
- Implements reference sharding calculations for multiple versions (v2/v3/v4).
- Validates that `sharding_sparse_fids` outputs (`fid_map`, offsets, row splits) match manually computed results.
- Tests for:
  - ExampleBatch sharding with shared features.
  - Example sharding with generated v2 features.
  - Instance sharding with v1+v2 features.
  - Pre-v2 compatibility path (`DataOpsV2TestFitPre`).
- Uses `ParserCtx.enable_fused_layout` toggle to compare base vs fused outputs.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/tests`.
- Rust public API surface: sparse sharding utilities and fused layout parser.
- Data model mapping: feature configs → shard maps and offsets.
- Feature gating: none.
- Integration points: parsing pipeline for distributed embedding.

**Implementation Steps (Detailed)**
1. Implement sharding_sparse_fids equivalent in Rust.
2. Port the reference sharding calculations to Rust tests.
3. Compare fused vs non-fused parsing outputs.

**Tests (Detailed)**
- Python tests: this file (extensive).
- Rust tests: TODO (manual)
- Cross-language parity test: TODO (manual)

**Gaps / Notes**
- TODO (manual)

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed
