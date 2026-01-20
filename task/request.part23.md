<!--
Source: task/request.md
Lines: 5957-6181 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/data/data_service_test.py`
<a id="monolith-native-training-data-data-service-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 98
- Purpose/role: Tests data service split provider with `DynamicMatchingFilesDataset`.
- Key symbols/classes/functions: `DataServiceTest.testSplitProvider`.
- External dependencies: `tensorflow.data.experimental.service`, `DynamicMatchingFilesDataset`.
- Side effects: starts dispatcher/workers on port 7080.

**Required Behavior (Detailed)**
- Registers `DynamicMatchingFilesDataset` and consumes it via two distributed_epoch consumers.
- Alternates pulling items until both consumers are exhausted.
- Asserts total count equals 19.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/tests`.
- Rust public API surface: data service dataset tests.
- Data model mapping: dynamic file matching dataset.
- Feature gating: TF data service equivalent required.
- Integration points: dataset registry + consumer.

**Implementation Steps (Detailed)**
1. Only port if Rust data service is implemented.

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

### `monolith/native_training/data/datasets.py`
<a id="monolith-native-training-data-datasets-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 1642
- Purpose/role: Dataset factory and dataset classes for PB files, Kafka streams, Parquet, TFRecord, transforms, and data service.
- Key symbols/classes/functions: `PBDataset`, `FilePBDataset`, `DistributedFilePBDataset`, `KafkaDataset`, `ParquetDataset`, `InstanceReweightDataset`, `NegativeGenDataset`, `SplitFlowDataset`, `MergeFlowDataset`, `TransformDataset`, `DatasetMetaclass`, `distribute`, `merged_window`.
- External dependencies: TensorFlow internals, custom ops `gen_monolith_ops`, Kafka consumer, data service APIs, feature utils.
- Side effects: defines many flags; registers graph collections; disables iterator save/restore; spawns Kafka polling thread.

**Required Behavior (Detailed)**
- Flags defined: `data_service_dispatcher`, `dataset_use_dataservice`, `dataset_input_patterns`, `dataset_input_use_snappy`, `dataset_input_compression_type`, `dataset_input_use_parquet`, `dataset_input_use_tfrecord`, `dataset_worker_idx`, `dataset_num_workers`, `kafka_other_metadata`.
- `DatasetMetaclass.__call__`:
  - Normalizes shorthand kwargs (`topics_or_files`, `buffer_size_or_group_id`, `input_pb_type_or_servers`).
  - Expands `dataset_input_patterns` into file patterns (DATE/INT range syntax); overwrites `patterns`; removes `file_name`.
  - Forces parquet/tfrecord flags from global FLAGS; prevents both true.
  - If Kafka args present, returns `KafkaDataset`.
  - Otherwise returns `DistributedFilePBDataset`, `ParquetDataset`, `TFRecordDatasetWrapper`, or `FilePBDataset` based on args.
- `PBDataset`: empty init (factory via metaclass).
- `PBDataset.gen_patterns(...)`:
  - Expands date/hour ranges into path patterns.
- `DynamicMatchingFilesDataset`: uses custom op `dynamic_matching_files_dataset`.
- `ParquetDataset`: validates columns/types, sets `OUTPUT_PB_TYPE_GRAPH_KEY`, uses custom parquet op.
- `FilePBDataset`:
  - Determines input/output pb types, uses FeatureList to prune if possible.
  - Configures flags: has_sort_id, kafka_dump, lagrangex_header, compression, snappy.
  - Adds output pb type to collection; calls `pb_dataset` op.
- `DistributedFilePBDataset`:
  - Creates file list, optional dynamic sharding, data service or matching_files.
  - Supports parquet/tfrecord mapping; handles sharding by worker or explicit shard_num.
- `InstanceReweightDataset`: wraps custom op `instance_reweight_dataset` based on action priorities.
- `NegativeGenDataset`: wraps custom op `instance_negative_gen_dataset`, creates item pool.
- `SplitFlowDataset` / `MergeFlowDataset`: custom ops for flow splitting/merging.
- `KafkaDataset`:
  - Initializes Kafka resource via custom ops; uses `kafka_read_next_v2`, unbatch.
  - Sets output pb type collection and flags (sort_id, dump, lagrangex_header).
- `PyKafkaDataset`:
  - Python KafkaConsumer with background polling thread; converts strings to variant via `string_to_variant`.
- `register_dataset` / `from_dataset_id`:
  - Data service register/uncompress; external_state_policy preserved.
- `merged_window`: window and reshape dataset elements.
- `distribute`: data service integration with sync training / ps-worker queue; handles Horovod/BytePS.
- `TransformDataset`: wraps `transform_dataset` op with serialized Transform config.
- Monkey patches: `Dataset.instance_reweight`, `.negative_gen`, `.split_flow`, `.merge_flow`, `.distribute`, `.merged_window`, `.transform`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/src`.
- Rust public API surface: dataset factory + dataset operators.
- Data model mapping: TF Dataset/Variant → Rust streaming dataset abstractions.
- Feature gating: Kafka, Parquet, TFRecord, DataService, custom ops.
- Integration points: data ingestion and training input pipeline.

**Implementation Steps (Detailed)**
1. Decide which dataset sources are in-scope for Rust (file/kafka/parquet).
2. Implement dataset factory/dispatch logic and pattern expansion.
3. Provide custom ops or pure-Rust replacements for instance reweight, negative gen, transform.
4. Add data service integration or document unsupported.

**Tests (Detailed)**
- Python tests: `data_ops_test.py`, `data_service_test.py`, `negative_gen_test.py`, etc.
- Rust tests: dataset factory, pattern expansion, ops parity tests.
- Cross-language parity test: compare dataset counts and parsed outputs for fixtures.

**Gaps / Notes**
- Heavy reliance on custom TF ops; Rust needs replacements or TF runtime backend.

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

### `monolith/native_training/data/eager_mode_test.py`
<a id="monolith-native-training-data-eager-mode-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 186
- Purpose/role: Eager-mode dataset parsing tests for Instance/Example/ExampleBatch.
- Key symbols/classes/functions: `DataOpsTest.target`, test methods.
- External dependencies: `tensorflow`, `PBDataset`, `parse_instances/parse_examples/parse_example_batch`, `switch_slot`, `feature_combine`.
- Side effects: reads training_instance fixtures.

**Required Behavior (Detailed)**
- `target(input_pb_type, output_pb_type)`:
  - Builds `PBDataset` for Instance/Example/ExampleBatch and applies instance reweight for ExampleBatch.
  - Parses via `parse_inst_exam` or `parse_eb` depending on pb types.
  - Batches then iterates `dataset.take(5)` and asserts feature dict length ∈ {26,27}.
- `testExampleBatch2Instance`, `testExample2Instance`, `testInstance2Instance` call `target`.
- `testExampleBatch`:
  - Parses ExampleBatch to ExampleBatch via `parse_example_batch`, asserts len ∈ {26,27}.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/tests`.
- Rust public API surface: dataset parsing tests (eager mode equivalent).
- Data model mapping: dataset pipelines in Rust.
- Feature gating: none.
- Integration points: `monolith-data` parsing pipeline.

**Implementation Steps (Detailed)**
1. Add parsing tests for all pb types with fixed fixtures.
2. Ensure eager-mode behavior maps to Rust pipeline semantics.

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

### `monolith/native_training/data/extract_fid_test.py`
<a id="monolith-native-training-data-extract-fid-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 30
- Purpose/role: Tests custom op `extract_fid`.
- Key symbols/classes/functions: `ExtraFidTest.test_parse_search`.
- External dependencies: `tensorflow`, `gen_monolith_ops.extract_fid`.
- Side effects: none.

**Required Behavior (Detailed)**
- `extract_fid(185, 4)` must return `1153447759131936`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/tests` (or runtime ops crate).
- Rust public API surface: `extract_fid` equivalent.
- Data model mapping: custom op to Rust function.
- Feature gating: requires runtime ops.
- Integration points: feature parsing pipeline.

**Implementation Steps (Detailed)**
1. Implement `extract_fid` in Rust runtime ops.
2. Add a unit test for the exact constant output.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: add exact-match test.
- Cross-language parity test: compare output for fixed inputs.

**Gaps / Notes**
- Relies on custom TF op; no Rust implementation yet.

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
