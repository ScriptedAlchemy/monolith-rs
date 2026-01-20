<!--
Source: task/request.md
Lines: 17480-17771 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/model_export/warmup_data_gen.py`
<a id="monolith-native-training-model-export-warmup-data-gen-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 253
- Purpose/role: CLI tool to generate TF Serving warmup PredictionLog data from existing pb data or random generation.
- Key symbols/classes/functions: `PBReader`, `gen_prediction_log_from_file`, `tf_dtype`, `main`.
- External dependencies: TensorFlow, TF Serving protos, `data_gen_utils.gen_prediction_log`, `env_utils`.
- Side effects: Reads input files, writes TFRecord warmup data to `output_path`.

**Required Behavior (Detailed)**
- CLI flags cover file input, batch sizes, feature lists/types, and generation mode (`gen_type` = file/random).
- `PBReader`:
  - Iterates over binary input stream (stdin or file), reading size-prefixed records.
  - Supports lagrangex header, sort_id, kafka_dump_prefix/kafka_dump.
  - For `example_batch`, reads one record per batch; otherwise reads `batch_size` records.
  - `set_max_iter(max_records)` sets max iterations based on variant type.
- `gen_prediction_log_from_file(...)`:
  - Chooses input name based on variant_type (`instances`, `examples`, `example_batch`).
  - Ensures `serving_default` signature included.
  - Yields `PredictionLog` entries with `PredictRequest` containing the batch tensor.
- `tf_dtype(dtype: str)`:
  - Maps string/int aliases to TF dtypes; **bug**: returns `tf.int46` for int64 cases (invalid dtype).
- `main`:
  - Calls `env_utils.setup_hdfs_env()`.
  - Writes PredictionLog records to `FLAGS.output_path` using TFRecordWriter.
  - If `gen_type == 'file'`, uses `gen_prediction_log_from_file`.
  - Else constructs feature specs from CLI flags and calls `data_gen_utils.gen_prediction_log(...)`.

**Rust Mapping (Detailed)**
- Target crate/module: N/A.
- Rust public API surface: none.
- Data model mapping: TF Serving PredictionLog proto and tensor encoding.
- Feature gating: TF Serving protos required.
- Integration points: warmup data generation tooling.

**Implementation Steps (Detailed)**
1. Fix `tf_dtype` mapping if porting (use `tf.int64` for int64/long).
2. Clarify `gen_prediction_log` API usage; current call signature appears outdated.
3. If porting to Rust, implement size-prefixed reader and PredictionLog writer.

**Tests (Detailed)**
- Python tests: none.
- Rust tests: add unit tests for PBReader and dtype mapping.
- Cross-language parity test: compare generated TFRecord entries for fixed inputs.

**Gaps / Notes**
- `tf_dtype` uses `tf.int46` (typo).
- `gen_prediction_log` call signature likely mismatched with current `data_gen_utils` (uses ParserArgs now).

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

### `monolith/native_training/model_export/warmup_example_batch.py`
<a id="monolith-native-training-model-export-warmup-example-batch-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 57
- Purpose/role: Converts saved example-batch files into TF Serving PredictionLog warmup records.
- Key symbols/classes/functions: `gen_prediction_log`, `main`.
- External dependencies: TensorFlow, TF Serving protos, `env_utils`.
- Side effects: Reads input folder, writes TFRecord output.

**Required Behavior (Detailed)**
- Flags: `input_folder`, `output_path` (both required for use).
- `gen_prediction_log(input_folder)`:
  - Iterates files in input folder.
  - Reads file bytes and parses into `PredictRequest`.
  - Sets `model_spec.name="default"` and `signature_name="serving_default"`.
  - Wraps in `PredictionLog` and yields.
  - Prints parse result (debug).
- `main`:
  - Writes logs to TFRecord at `output_path`.
- `__main__` calls `env_utils.setup_hdfs_env()` then runs app.

**Rust Mapping (Detailed)**
- Target crate/module: N/A.
- Rust public API surface: none.
- Data model mapping: TF Serving PredictionLog.
- Feature gating: TF Serving protos.
- Integration points: warmup data generation.

**Implementation Steps (Detailed)**
1. If porting, read binary example-batch files and wrap into PredictionLog.
2. Preserve default model/signature names.

**Tests (Detailed)**
- Python tests: none.
- Rust tests: none.
- Cross-language parity test: compare serialized output logs.

**Gaps / Notes**
- No validation of input file format; assumes each file is a serialized PredictRequest.

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

### `monolith/native_training/monolith_export.py`
<a id="monolith-native-training-monolith-export-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 18
- Purpose/role: No-op decorator used to mark classes/functions for export.
- Key symbols/classes/functions: `monolith_export`.
- External dependencies: none.
- Side effects: Adds `__monolith_doc` attribute set to `None` on the object.

**Required Behavior (Detailed)**
- `monolith_export(obj)`:
  - Sets `obj.__monolith_doc = None`.
  - Returns the original object.
  - Used as decorator on classes/functions.

**Rust Mapping (Detailed)**
- Target crate/module: N/A.
- Rust public API surface: none.
- Data model mapping: none.
- Feature gating: none.
- Integration points: documentation/export tooling only.

**Implementation Steps (Detailed)**
1. If Rust needs similar tagging, add a marker trait or attribute macro (optional).

**Tests (Detailed)**
- Python tests: none.
- Rust tests: none.
- Cross-language parity test: not applicable.

**Gaps / Notes**
- Pure annotation; no runtime behavior beyond attribute set.

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

### `monolith/native_training/multi_hash_table_ops.py`
<a id="monolith-native-training-multi-hash-table-ops-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 695
- Purpose/role: Implements multi-hash-table ops wrapper around custom TF ops, including lookup/assign/optimize and checkpoint save/restore hooks.
- Key symbols/classes/functions: `CachedConfig`, `MultiHashTable`, `MultiHashTableCheckpointSaverListener`, `MultiHashTableCheckpointRestorerListener`, `MultiHashTableRestorerSaverListener`.
- External dependencies: TensorFlow custom ops (`gen_monolith_ops`), hash table protobufs, save_utils, distributed_serving_ops.
- Side effects: Registers proto functions, adds tables to TF collections, writes ckpt info files.

**Required Behavior (Detailed)**
- Constants: `_TIMEOUT_IN_MS` (1 hour), `_MULTI_HASH_TABLE_GRAPH_KEY`.
- `CachedConfig`:
  - Stores configs, table_names, serialized mconfig, tensor, dims, slot_expire_time_config.
- `infer_dims`/`convert_to_cached_config`:
  - Builds `MultiEmbeddingHashTableConfig`, sets entry_type=SERVING when exporting.
  - Serializes config and returns `CachedConfig`.
- `MultiHashTable`:
  - Creates/reads multi hash table handle via custom ops, registers resource, adds to collection.
  - `from_cached_config` sets device based on table type (gpucuco -> GPU).
  - Lookup/assign/add/optimize operations delegate to custom ops.
  - `raw_lookup`, `raw_assign`, `raw_apply_gradients` use ragged ids and flat values.
  - Provides fused lookup/optimize for sync training.
  - `save`/`restore` use custom ops with basename.
  - `to_proto`/`from_proto` allow graph serialization.
- Helpers: ragged concatenation and flattening utilities for input/outputs.
- Checkpoint listeners:
  - `MultiHashTableCheckpointSaverListener` saves tables before saver, optionally writes `ckpt.info-<step>` with feature counts.
  - `MultiHashTableCheckpointRestorerListener` restores tables before restore, with optional PS monitor skip.
  - `MultiHashTableRestorerSaverListener` triggers restore after save.
- Registers proto functions on `_MULTI_HASH_TABLE_GRAPH_KEY` and marks `IsHashTableInitialized` as not differentiable.

**Rust Mapping (Detailed)**
- Target crate/module: N/A (TF custom ops).
- Rust public API surface: would require binding the custom multi-hash-table ops.
- Data model mapping: protobuf configs, table handles, ragged IDs.
- Feature gating: TF runtime + custom ops.
- Integration points: embedding tables, checkpointing, distributed serving.

**Implementation Steps (Detailed)**
1. Bind custom ops for multi-hash-table if TF runtime backend is enabled.
2. Mirror ragged-id flattening and value slicing for embeddings.
3. Implement checkpoint listeners or hook equivalents for save/restore.
4. Match proto serialization for export/import.

**Tests (Detailed)**
- Python tests: `multi_hash_table_ops_test.py`.
- Rust tests: none.
- Cross-language parity test: compare lookup/assign outputs and checkpoint restores.

**Gaps / Notes**
- Requires custom ops and protobuf definitions; no Rust equivalent today.

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

### `monolith/native_training/multi_hash_table_ops_test.py`
<a id="monolith-native-training-multi-hash-table-ops-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 249
- Purpose/role: Tests for MultiHashTable operations (lookup, assign_add, reinitialize, apply_gradients, save/restore, hooks).
- Key symbols/classes/functions: `MultiTypeHashTableTest.*`.
- External dependencies: TensorFlow, multi_hash_table_ops custom ops, save_utils.
- Side effects: Writes checkpoint and asset files under `TEST_TMPDIR`.

**Required Behavior (Detailed)**
- `test_lookup_assign_add_reinitialize`:
  - Builds table with slots; assign_add values; lookup matches expected.
  - Reinitializes slot2 and slot3; slot3 returns status -1.
- `test_apply_gradients`:
  - Applies gradients and checks embeddings updated (negative values).
- `test_save_restore`:
  - Saves table to basename, restores into new graph with different slots.
  - Restored values for overlapping slots match expected.
- `test_save_restore_hook`:
  - Uses saver and restorer hooks; ensures restore overwrites sub_op updates.
  - Verifies values match initial add_op.
- `test_meta_graph_export`:
  - Ensures multi hash table collection appears in exported meta_graph.

**Rust Mapping (Detailed)**
- Target crate/module: N/A.
- Rust public API surface: none.
- Data model mapping: custom ops and checkpoint assets.
- Feature gating: TF runtime + custom ops.
- Integration points: embedding table subsystem.

**Implementation Steps (Detailed)**
1. If Rust binds multi hash table ops, add tests for assign/lookup/save/restore.
2. Mirror hook behavior in Rust checkpointing.

**Tests (Detailed)**
- Python tests: `multi_hash_table_ops_test.py`.
- Rust tests: none.
- Cross-language parity test: compare lookup outputs and restore behavior.

**Gaps / Notes**
- Heavily depends on custom ops; cannot run without TF runtime.

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
