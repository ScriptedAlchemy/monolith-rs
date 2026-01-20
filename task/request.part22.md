<!--
Source: task/request.md
Lines: 5709-5956 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/cpu_training_test.py`
<a id="monolith-native-training-cpu-training-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 597
- Purpose/role: Comprehensive CPU training tests covering feature slots, export modes, occurrence/expire configs, distributed training, debugging server, and local train.
- Key symbols/classes/functions: `FeatureTask`, `FloatFeatureTask`, `SequenceFeatureTask`, `FeatureWithSlotOccurrenceThresholdTask`, `FeatureWithExpireTimeTask`, `NonFeatureTask`, `CpuTrainTest`, `DistributedTrainTest`, `LocalTrainTest`.
- External dependencies: `tensorflow`, `cpu_training`, `entry`, `feature`, `utils`, `debugging_server`, `saved_model_exporters`, `ExportMode`, `debugging_info_pb2`, `embedding_hash_table_pb2`, `ServiceDiscovery`.
- Side effects: spawns subprocesses for distributed tests; writes checkpoints and export dirs; reads debugging info files.

**Required Behavior (Detailed)**
- Shared helpers:
  - `inc_global_step_op()` increments global step and returns grouped op.
  - `FLAGS.use_native_multi_hash_table` controls hash table implementation.
- `FeatureTask`:
  - Input: ragged feature tensor.
  - Model: create slot/slice, embedding lookup; grads applied via feature factory; predict returns sum.
  - Serving input receiver uses ragged constant + placeholder for serialized input.
- `FloatFeatureTask`:
  - Uses ragged embedding + float feature; predictions from float feature; training uses embedding grads.
- `SequenceFeatureTask`:
  - Uses combiner `FeatureColumnV1.first_n(2)`; predictions from sequence feature sum.
- `FeatureWithSlotOccurrenceThresholdTask`:
  - Creates slot with `slot_id=2021`, `occurrence_threshold=3`; asserts training captures threshold map.
- `FeatureWithExpireTimeTask`:
  - Two slots with `expire_time=0` and `1`; uses `ZerosInitializer`.
  - After training, checks `_slot_to_expire_time` map and prediction values.
- `NonFeatureTask`:
  - Input dataset yields scalar; train op uses input with global step increment.
- `CpuTrainTest`:
  - `test_cpu_training_feature` basic training.
  - `test_with_misc_features`: `feature_eviction_on_save=True`.
  - `test_with_export_when_saving`: `serving.export_when_saving=True`.
  - `test_dense_only_export`: export mode `DISTRIBUTED` + `dense_only_save_checkpoints_steps=10`.
  - `test_with_prefetch_postpush`: enables variable prefetch/postpush and embedding postpush.
  - `test_cpu_training_float_feature` and `test_cpu_training_sequence_feature` run those tasks.
  - `test_cpu_training_with_slot_occurrence_threshold` checks internal threshold map.
  - `test_cpu_training_with_expire_time` checks expire time map and prediction values.
  - `test_cpu_training_non_feature` runs non-feature task.
  - `test_gpu_export`: exports saved model with remote GPU.
- `DistributedTrainTest`:
  - Spawns `cpu_training_distributed_test_binary` processes for PS/worker.
  - Tests: basic, extra_ps, redundant_ps, debugging server (case=1), temporary error (case=2).
  - `test1_with_debugging_server` waits for checkpoints then reads debugging info proto; checks variable/feature fetch via debugging server.
- `LocalTrainTest`:
  - `local_train` with and without PS.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests`.
- Rust public API surface: training loop, export pipeline, debug info tooling.
- Data model mapping: TF Estimator/FeatureFactory → Rust training/feature abstractions.
- Feature gating: depends on TF runtime and distributed runner.
- Integration points: `TrainingConfig`, export utilities, discovery and debugging.

**Implementation Steps (Detailed)**
1. Build Rust integration tests that cover: basic training, feature thresholds, expire time, and export flow.
2. Add distributed runner test harness or mark as Python-only.
3. Provide debug info export and retrieval parity tests.

**Tests (Detailed)**
- Python tests: this file plus `cpu_training_distributed_test_binary`.
- Rust tests: integration tests for training/export/debug info.
- Cross-language parity test: compare embedding predictions and debug info outputs.

**Gaps / Notes**
- Uses private internals (`training._slot_to_occurrence_threshold`, `_slot_to_expire_time`).
- Debugging server depends on proto + embedding hash table dumps not present in Rust.

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

### `monolith/native_training/data/__init__.py`
<a id="monolith-native-training-data-init-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 20
- Purpose/role: Package exports for dataset builders and feature utilities.
- Key symbols/classes/functions: re-exports `PBDataset`, `InstanceReweightDataset`, `NegativeGenDataset`, `PbType`, `parse_examples`, `parse_instances`, `parse_example_batch`, `filter_by_*`, `feature_combine`, `negative_sample`, `switch_slot`, `special_strategy`.
- External dependencies: internal modules `datasets`, `parsers`, `feature_utils`.
- Side effects: imports modules at package import time.

**Required Behavior (Detailed)**
- Importing package exposes symbols listed above at `monolith.native_training.data.*`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/src`.
- Rust public API surface: module re-exports for dataset and feature utils.
- Data model mapping: match Python export surface.
- Feature gating: none.
- Integration points: data pipeline and training input.

**Implementation Steps (Detailed)**
1. Re-export dataset and parser APIs in Rust modules.

**Tests (Detailed)**
- Python tests: none specific.
- Rust tests: module visibility tests.
- Cross-language parity test: not applicable.

**Gaps / Notes**
- None.

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

### `monolith/native_training/data/data_ops_test.py`
<a id="monolith-native-training-data-data-ops-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 502
- Purpose/role: End-to-end tests for PB datasets, parsing, filtering, compression, and dataset variants.
- Key symbols/classes/functions: `DataOpsTest`, `CahceOneDatasetTest`, `DecompressTest`, helper parsers.
- External dependencies: `tensorflow`, `PBDataset`, `parse_examples/instances/example_batch`, `feature_utils` filters, `ExampleBatch`, `gen_random_data_file`, `session_hooks`.
- Side effects: creates temp files under `tmp_data`; invokes external `zstd` binary; writes compressed files.

**Required Behavior (Detailed)**
- `DataOpsTest.setUpClass`:
  - Generates random instance data files (3 parts) using `gen_random_data_file`.
- `pb_dataset_target(input_pb_type, output_pb_type, filter_fn=None)`:
  - Builds `PBDataset` for Instance/Example/ExampleBatch with appropriate flags.
  - For ExampleBatch, applies `instance_reweight`.
  - Applies optional `filter_fn`.
  - Batches and maps to parser (`parse_inst_exam`/`parse_eb`).
  - Iterates through dataset and counts elements (logs count).
- Tests validate permutations:
  - Instance→Instance/Example, Example→Example/Instance, ExampleBatch→Example/Instance.
  - PLAINTEXT output for Instance/Example.
  - `filter_by_fids`, `filter_by_value` (ge/in/eq/between/str/any/all/diff), `special_strategy`.
  - `parse_example_batch` for scalar and batch inputs.
  - `PBDataset` resolves to `FilePBDataset` or `KafkaDataset` based on flags/args.
  - `testCreateInstanceDatasetHdfs` reads generated files via `PBDataset`.
  - `PBDataset.gen_patterns` returns correct date range size.
- `CahceOneDatasetTest`:
  - `CacheOneDataset` wraps dataset; second element flagged `True`.
- `DecompressTest`:
  - Creates copies of examplebatch data and tests `CompressType.ZSTD/ZLIB/GZIP`.
  - Uses `parse_example_batch` to validate decompression.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/tests`.
- Rust public API surface: dataset parsing, compression, filter utils.
- Data model mapping: TF datasets → Rust data pipeline equivalents.
- Feature gating: compression codecs and Kafka support.
- Integration points: `monolith-data` crate and training input.

**Implementation Steps (Detailed)**
1. Add Rust tests for parsing Instance/Example/ExampleBatch formats.
2. Add filter/feature utility tests mirroring Python cases.
3. Add compression decode tests (zstd/zlib/gzip).

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: dataset parsing + compression + filter tests.
- Cross-language parity test: compare parsed feature dict shapes and counts.

**Gaps / Notes**
- Requires external `zstd` binary for decompression test.
- Relies on hard-coded proto files under `monolith/native_training/data/training_instance`.

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

### `monolith/native_training/data/data_service_parquet_test.py`
<a id="monolith-native-training-data-data-service-parquet-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 145
- Purpose/role: Integration test for TF data service reading Parquet via `PBDataset`.
- Key symbols/classes/functions: `DataServiceTest2.testDataServiceWithParquetDataset`.
- External dependencies: `tensorflow.data.experimental.service`, `PBDataset`, `PbType`, `ExampleBatch`, `json`.
- Side effects: starts dispatcher/workers on port 7080, reads local files.

**Required Behavior (Detailed)**
- `setUpClass`/`tearDownClass`: create and destroy dispatcher + workers.
- `testDataServiceWithParquetDataset`:
  - Reads `META_JSON_PATH` and `PARQUET_DIR` env vars (defaults under `$HOME/temp`).
  - Loads meta JSON, builds column names/types mapping.
  - Creates `PBDataset` with `use_data_service=True`, `use_parquet=True`, `output_pb_type=PLAINTEXT`.
  - Registers dataset and reads from two consumers (distributed_epoch).
  - Parses `ExampleBatch` from bytes, accumulates `batch_size`.
  - Prints row count (assertion against parquet row count is commented out).

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/tests`.
- Rust public API surface: data service + parquet ingestion tests (if supported).
- Data model mapping: `PBDataset` + Parquet pipeline.
- Feature gating: Parquet and data service support.
- Integration points: dataset registration/consumers.

**Implementation Steps (Detailed)**
1. Add Rust integration test only if parquet + data service are implemented.
2. Otherwise, document as Python-only system test.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: optional integration tests.
- Cross-language parity test: compare total row counts.

**Gaps / Notes**
- Test depends on local parquet files and meta JSON; not hermetic.

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
