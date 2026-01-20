<!--
Source: task/request.md
Lines: 10935-11193 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/hash_filter_ops_test.py`
<a id="monolith-native-training-hash-filter-ops-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 217
- Purpose/role: Tests hash filter gradient interception and save/restore behavior.
- Key symbols/classes/functions: `HashFilterOpsTest` cases.
- External dependencies: TensorFlow, `hash_filter_ops`, `embedding_hash_table_pb2`, TFRecord reader.
- Side effects: Writes TFRecord checkpoint shards under `TEST_TMPDIR`.

**Required Behavior (Detailed)**
- `test_hash_filter_basic`:
  - Creates hash filter with occurrence threshold 3; intercepts gradient.
  - Gradients are filtered progressively: first two runs zero out more ids, later runs allow gradients.
- `test_hash_filter_save_restore`:
  - Saves filter to basename; verifies 7 split files created.
  - Restores and verifies gradient filtering state persists across saves.
- `test_hash_filter_save_restore_across_multiple_filters`:
  - Creates filter with split_num=100; verifies each shard contains expected `HashFilterSplitMetaDump` fields.
  - After second save, first 4 shards contain 2 elements; remaining shards contain 0.
- `test_dummy_hash_filter_basic`:
  - Dummy filter should not filter gradients (all ones).
- `test_dummy_hash_filter_save_restore`:
  - Save/restore with dummy filter produces no files and no effect on gradients.
- `test_restore_not_found`:
  - Restoring from non-existent path raises exception.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-tf/tests/hash_filter_ops_test.rs` (new).
- Rust public API surface: hash filter creation, save/restore, intercept gradient ops.
- Feature gating: `tf-runtime` + custom ops; skip otherwise.
- Integration points: TFRecord parsing of `HashFilterSplitMetaDump`.

**Implementation Steps (Detailed)**
1. Port test cases to Rust TF runtime harness.
2. Implement TFRecord reader for `HashFilterSplitMetaDump` proto in Rust.
3. Validate gradient filtering sequence and save/restore shard counts.

**Tests (Detailed)**
- Python tests: `monolith/native_training/hash_filter_ops_test.py`.
- Rust tests: `monolith-rs/crates/monolith-tf/tests/hash_filter_ops_test.rs`.
- Cross-language parity test: compare shard metadata and gradient outputs for same ids.

**Gaps / Notes**
- Tests depend on custom ops and TFRecord writer/reader compatibility.

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

### `monolith/native_training/hash_table_ops.py`
<a id="monolith-native-training-hash-table-ops-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 1208
- Purpose/role: Core TensorFlow hash table wrapper with custom ops for lookup, update, save/restore, and fused operations.
- Key symbols/classes/functions: `BaseHashTable`, `HashTable`, `hash_table_from_config`, `test_hash_table`, `fused_lookup`, `fused_apply_gradient`, checkpoint saver/restorer listeners.
- External dependencies: TensorFlow, custom ops `gen_monolith_ops`, `embedding_hash_table_pb2`, `hash_filter_ops`, `distributed_serving_ops`, `save_utils`, `graph_meta`.
- Side effects: Registers hash tables in graph collection, writes/reads checkpoint assets, registers proto serialization hooks.

**Required Behavior (Detailed)**
- `BaseHashTable` abstract API: `assign`, `assign_add`, `lookup`, `apply_gradients`, `as_op`, `dim_size`.
- `_HASH_TABLE_GRAPH_KEY` collection stores `HashTable` instances for save/restore.
- `HashTable`:
  - Ensures unique `shared_name` via metadata; raises if duplicate.
  - Wraps `monolith_hash_table_*` custom ops for assign/assign_add/lookup/lookup_entry/optimize/save/restore/size.
  - `apply_gradients` uses `monolith_hash_table_optimize` and returns a new table with control dependency.
  - `save_as_tensor` dumps entries as serialized `EntryDump` strings with sharding and offsets.
  - `to_proto`/`from_proto` serialize state via `hash_table_ops_pb2.HashTableProto` and `_BOOL_MAP`.
- `fused_lookup`:
  - Calls `monolith_hash_table_fused_lookup` and returns embeddings, recv_splits, id_offsets, emb_offsets.
- `fused_apply_gradient`:
  - Calls `monolith_hash_table_fused_optimize` with ids, indices, fused_slot_size, grads, offsets, learning rates, req_time, global_step.
- `hash_table_from_config`:
  - Builds table op using `EmbeddingHashTableConfig` and `HashTableConfigInstance`.
  - Chooses GPU vs CPU based on table type; forces SERVING entry type when exporting.
  - Creates hash filter and sync client if not provided.
- `test_hash_table` and `vocab_hash_table` helpers create simple tables for tests.
- `HashTableCheckpointSaverListener`:
  - Builds save ops using placeholders; writes asset files with randomized sleep to reduce metadata pressure.
- `HashTableCheckpointRestorerListener`:
  - Restores from latest checkpoint assets; supports sparse-only assets; uses thread pool to resolve prefixes.
- `HashTableRestorerSaverListener`:
  - Triggers restore after save (used for evicting stale entries).
- Registers proto serialization with `ops.register_proto_function`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-hash-table` + TF runtime wrappers in `monolith-rs/crates/monolith-tf`.
- Rust public API surface: hash table struct with assign/lookup/optimize, fused lookup/optimize wrappers, save/restore helpers.
- Data model mapping: use `monolith::hash_table` protos for config and serialization.
- Feature gating: TF custom ops required; Candle backend may implement a native hash table for local use.
- Integration points: feature factory, embedding gradients, distributed serving.

**Implementation Steps (Detailed)**
1. Implement a Rust hash table wrapper that mirrors assign/assign_add/lookup semantics and returns updated handles.
2. Add proto serialization helpers and maintain a registry for save/restore discovery.
3. Wrap fused lookup/optimize in TF runtime; add no-op stubs otherwise.
4. Implement checkpoint saver/restorer listeners and asset dir layout matching Python.
5. Add thread-pool based restore prefix matching for extra restore names.

**Tests (Detailed)**
- Python tests: `hash_table_ops_test.py`.
- Rust tests: `monolith-rs/crates/monolith-tf/tests/hash_table_ops_test.rs` (TF runtime) and native hash table unit tests.
- Cross-language parity test: compare lookup/update outputs and serialized dumps.

**Gaps / Notes**
- Heavy reliance on custom TF ops; full parity requires TF runtime + compiled ops.
- Save/restore path handling must match asset dir naming to avoid silent failures.

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

### `monolith/native_training/hash_table_ops_benchmark.py`
<a id="monolith-native-training-hash-table-ops-benchmark-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 148
- Purpose/role: Benchmarks hash table lookup and optimize paths (single-thread vs multi-thread).
- Key symbols/classes/functions: `HashTableOpsBenchmark` tests.
- External dependencies: TensorFlow, `hash_table_ops`.
- Side effects: Prints timing to stdout.

**Required Behavior (Detailed)**
- `test_lookup` / `test_lookup_multi_thread`:
  - Build table with len=10000, dim=32; assign ones for all but last 5 IDs.
  - Run lookup in a loop and validate embeddings (ones vs zeros).
- `test_basic_optimize` / `test_multi_threads_optimize` / `test_multi_threads_optimize_with_dedup`:
  - Build table with len=1,000,000; lookup, compute grads, apply gradients (optionally MT/dedup), print timing.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-examples/src/bin/hash_table_ops_benchmark.rs` (new).
- Rust public API surface: hash table benchmark harness.
- Feature gating: TF runtime + custom ops for parity.

**Implementation Steps (Detailed)**
1. Port benchmark loops to Rust, mirroring data sizes and operations.
2. Add CLI flags for MT/dedup variants.
3. Validate outputs in debug mode.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: none; benchmark only.
- Cross-language parity test: compare output correctness and rough timing.

**Gaps / Notes**
- Uses `tf.test.TestCase` for benchmarking; Rust should use criterion or custom timers.

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

### `monolith/native_training/hash_table_ops_test.py`
<a id="monolith-native-training-hash-table-ops-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 1206
- Purpose/role: Comprehensive tests for hash table lookup, update, save/restore, eviction, fused ops, and hooks.
- Key symbols/classes/functions: `HashTableOpsTest` methods, helper `test_hash_table_with_hash_filters`.
- External dependencies: TensorFlow, `hash_table_ops`, `hash_filter_ops`, `embedding_hash_table_pb2`, `learning_rate_functions`.
- Side effects: Writes checkpoint assets under `TEST_TMPDIR`.

**Required Behavior (Detailed)**
- Basic ops:
  - `test_basic` assigns values and verifies lookup and size; table name auto-unique.
  - `test_assign` verifies assign overwrites values under control dependencies.
  - `test_lookup_entry` parses `EntryDump` strings; missing ids return empty bytes.
  - `test_save_as_tensor` ensures serialized entries can be parsed.
  - `testNameConflict` duplicate shared_name raises `ValueError`.
- Gradient updates:
  - `test_gradients` updates embeddings with SGD (learning_rate=0.1) -> `[[0.2],[0.1]]` for ids [0,1].
  - `test_gradients_with_learning_rate_fn` accepts callable LR.
  - `test_gradients_with_learning_rate_decay` uses PolynomialDecay; expected outputs `[[0.04],[0.02]]`.
  - `test_gradients_with_dedup` enables dedup; expected outputs `[[0.3...],[0.2...]]` for vec_dim=10.
  - `test_gradients_with_different_ids` applies grads with mismatched ids -> `[[0.1],[0.2]]`.
  - `test_gradients_with_hash_filter` verifies occurrence threshold gating across repeated updates.
- Save/restore:
  - `test_save_restore` round-trips assign_add values.
  - `test_restore_from_another_table` uses extra_restore_names to restore.
- Feature eviction / TTL:
  - `test_save_restore_with_feature_eviction_assign_add` and `...apply_gradients` evict entries older than expire_time.
  - `test_entry_ttl_zero` evicts all entries on restore.
  - `test_entry_ttl_not_zero` preserves entries when TTL positive.
  - `test_entry_ttl_by_slots` uses slot_expire_time_config for per-slot TTL.
- Hooks and restore flows:
  - `test_restore_not_found` raises on missing checkpoint.
  - `test_save_restore_hook` saver + restorer hook restores after sub_op.
  - Additional tests validate restore-after-save, feature eviction with hooks, and cleanup of save paths.
- Advanced/fused ops:
  - `test_fused_lookup` and `test_fused_optimize` cover fused operations.
  - `test_batch_softmax_optimizer` verifies BatchSoftmax behavior.
  - `test_extract_fid` checks `extract_slot_from_entry`.
  - `test_meta_graph_export` ensures proto export/import works.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-tf/tests/hash_table_ops_test.rs` (new).
- Rust public API surface: hash table operations, fused lookup/optimize, save/restore, eviction logic.
- Feature gating: TF runtime + custom ops required; skip otherwise.
- Integration points: hash filter ops, learning rate functions, save_utils equivalents.

**Implementation Steps (Detailed)**
1. Port basic lookup/assign/size tests to Rust TF runtime.
2. Implement gradient update tests for SGD and learning rate schedules.
3. Add save/restore tests with asset files; verify eviction by TTL and per-slot settings.
4. Add hook-based save/restore ordering tests.
5. Add fused op tests and meta-graph export/import tests.

**Tests (Detailed)**
- Python tests: `monolith/native_training/hash_table_ops_test.py`.
- Rust tests: `monolith-rs/crates/monolith-tf/tests/hash_table_ops_test.rs`.
- Cross-language parity test: compare serialized `EntryDump` bytes and lookup results.

**Gaps / Notes**
- Many tests depend on TF custom ops and checkpoint assets; may be too heavy for Rust CI.

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
