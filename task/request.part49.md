<!--
Source: task/request.md
Lines: 11194-11464 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/hash_table_utils.py`
<a id="monolith-native-training-hash-table-utils-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 46
- Purpose/role: Utility helpers to iterate hash tables and infer embedding dim sizes from config.
- Key symbols/classes/functions: `iterate_table_and_apply`, `infer_dim_size`.
- External dependencies: TensorFlow, `embedding_hash_table_pb2`.
- Side effects: Iterates table via `save_as_tensor` and calls `apply_fn`.

**Required Behavior (Detailed)**
- `iterate_table_and_apply(table, apply_fn, limit=1000, nshards=4, name="IterateTable")`:
  - Runs in `tf.function`.
  - Iterates `nshards` shards; for each shard, repeatedly calls `table.save_as_tensor(i, nshards, limit, offset)` until dump size < limit and offset != 0.
  - Uses `tf.autograph.experimental.set_loop_options` with shape invariants to allow dynamic `dump` size.
  - Calls `apply_fn(dump)` for each dump batch (serialized EntryDump strings).
- `infer_dim_size(config)`:
  - Sums `segment.dim_size` across `config.entry_config.segments`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-hash-table/src/utils.rs` (new) or `monolith-rs/crates/monolith-hash-table/src/lib.rs`.
- Rust public API surface: iterator over table dumps and dim-size inference from proto config.
- Data model mapping: `EmbeddingHashTableConfig` from `monolith-proto`.
- Feature gating: table iteration requires TF runtime or native hash table implementation.
- Integration points: model dump utilities and table export.

**Implementation Steps (Detailed)**
1. Implement a Rust iterator that pages through table dumps with `limit` and `offset` semantics.
2. Provide a safe callback API for applying functions to each batch.
3. Implement `infer_dim_size` by summing segment dims from proto.

**Tests (Detailed)**
- Python tests: none.
- Rust tests: add unit tests for `infer_dim_size` and mock iteration semantics.
- Cross-language parity test: compare dim_size for a known config.

**Gaps / Notes**
- `iterate_table_and_apply` depends on `save_as_tensor` behavior; Rust must match sharding and offset semantics.

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

### `monolith/native_training/hash_table_utils_test.py`
<a id="monolith-native-training-hash-table-utils-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 45
- Purpose/role: Tests `iterate_table_and_apply` paging across shards.
- Key symbols/classes/functions: `HashTableUtilsTest.test_iterate_table_and_apply`.
- External dependencies: TensorFlow, `hash_table_utils`, `hash_table_ops`.
- Side effects: Creates a test hash table and updates a counter variable.

**Required Behavior (Detailed)**
- Creates a test hash table with 100 ids.
- Uses `iterate_table_and_apply` with `limit=2` and `nshards=10` to iterate.
- `count_fn` increments a counter by `tf.size(dump)` for each batch.
- Final count must equal 100.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-hash-table/tests/hash_table_utils_test.rs` (new).
- Rust public API surface: table iteration helper and callback support.
- Feature gating: TF runtime or native hash table required.
- Integration points: hash table test helper equivalent.

**Implementation Steps (Detailed)**
1. Implement a Rust test that fills a table with 100 entries.
2. Iterate with small limit and shard count; accumulate total dumped entries.
3. Assert total count equals 100.

**Tests (Detailed)**
- Python tests: `monolith/native_training/hash_table_utils_test.py`.
- Rust tests: `monolith-rs/crates/monolith-hash-table/tests/hash_table_utils_test.rs`.
- Cross-language parity test: compare counts for same config.

**Gaps / Notes**
- Depends on `save_as_tensor` semantics; ensure Rust matches offset/limit behavior.

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

### `monolith/native_training/hooks/ckpt_hooks.py`
<a id="monolith-native-training-hooks-ckpt-hooks-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 193
- Purpose/role: Checkpoint hooks for worker iterator state and barrier coordination during saves.
- Key symbols/classes/functions: `BarrierSaverListener`, `WorkerCkptHelper`, `assign_ckpt_info`, `get_ckpt_info`, `disable_iterator_save_restore`.
- External dependencies: TensorFlow, `barrier_ops`, `basic_restore_hook`, `graph_meta`, `ckpt_hooks_pb2`.
- Side effects: Creates local variables and placeholders, manipulates iterator saveables, blocks workers with barriers.

**Required Behavior (Detailed)**
- `_get_meta()`:
  - Lazily creates `WorkerCkptMetaInfo` local variable + placeholder + assign op.
- `assign_ckpt_info(session, info)`:
  - Assigns serialized `WorkerCkptInfo` to info_var.
- `get_ckpt_info(session)`:
  - Reads info_var and parses to `WorkerCkptInfo`.
- `BarrierSaverListener`:
  - On `before_save`, places a barrier and waits for workers to block (up to max_pending_seconds).
  - On `after_save`, removes barrier if it was placed by this listener.
- `WorkerCkptHelper`:
  - Creates iterator saveables (if enabled) and a Saver to save per-worker iterator state.
  - `create_save_iterator_callback` saves iterator checkpoints using current global_step from `WorkerCkptInfo`.
  - `create_restorer_hook` restores iterator state on session creation.
- `disable_iterator_save_restore()`:
  - Disables iterator save/restore globally (must be called before helper creation).

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/hooks/ckpt_hooks.rs` (new).
- Rust public API surface: iterator checkpoint helper and barrier-based saver listener.
- Feature gating: TF runtime required for iterator saveables and session hooks.
- Integration points: training loops and distributed barrier coordination.

**Implementation Steps (Detailed)**
1. Implement worker checkpoint metadata storage and serialization.
2. Implement barrier saver listener with wait/remove semantics.
3. Implement iterator save/restore helper for dataset iterators.
4. Provide a global toggle to disable iterator save/restore.

**Tests (Detailed)**
- Python tests: `ckpt_hooks_test.py`.
- Rust tests: `monolith-rs/crates/monolith-training/tests/ckpt_hooks_test.rs` (new).
- Cross-language parity test: verify iterator restore resumes from same element.

**Gaps / Notes**
- Uses TF iterator saveables and should be skipped if dataset iterators are not used in Rust.

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

### `monolith/native_training/hooks/ckpt_hooks_test.py`
<a id="monolith-native-training-hooks-ckpt-hooks-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 181
- Purpose/role: Tests worker iterator save/restore and barrier-based saver listener behavior.
- Key symbols/classes/functions: `WorkerCkptHooksTest`, `CountCheckpointSaverListener`.
- External dependencies: TensorFlow, `ckpt_hooks`, `barrier_ops`, `save_utils`.
- Side effects: Writes checkpoints under `TEST_TMPDIR`, spawns a thread to run a monitored session.

**Required Behavior (Detailed)**
- `testIteratorSaveRestore`:
  - Saves iterator state at global_step=10 and restores; next element after restore matches expected value.
- `testNoCkpt`:
  - Restorer hook is a no-op when no checkpoint exists.
- `testNoSaveables`:
  - If no saveables, saving iterator state is skipped without error.
- `testCkptDisabled`:
  - `disable_iterator_save_restore()` prevents restore; iterator restarts from beginning.
- `test_saver_with_barrier`:
  - Verifies barrier placement during save and that saver listener callbacks are invoked in order.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests/ckpt_hooks_test.rs` (new).
- Rust public API surface: barrier saver listener and worker iterator checkpoint helper.
- Feature gating: TF runtime required.
- Integration points: save_utils and barrier operations.

**Implementation Steps (Detailed)**
1. Add a Rust test dataset iterator and verify save/restore steps.
2. Add a test for disabled iterator restore.
3. Add a barrier coordination test ensuring save waits for workers.

**Tests (Detailed)**
- Python tests: `monolith/native_training/hooks/ckpt_hooks_test.py`.
- Rust tests: `monolith-rs/crates/monolith-training/tests/ckpt_hooks_test.rs`.
- Cross-language parity test: compare iterator position after restore.

**Gaps / Notes**
- Threaded monitored session in test may be hard to replicate in Rust; can approximate with explicit calls.

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

### `monolith/native_training/hooks/ckpt_info.py`
<a id="monolith-native-training-hooks-ckpt-info-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 98
- Purpose/role: Saves per-slot feature-id counts from hash tables into a `ckpt.info-*` file at checkpoint time.
- Key symbols/classes/functions: `FidSlotCountSaverListener`.
- External dependencies: TensorFlow, `hash_table_ops`, `hash_table_utils`, `ckpt_info_pb2`.
- Side effects: Writes `ckpt.info-<global_step>` to model_dir.

**Required Behavior (Detailed)**
- `FidSlotCountSaverListener.__init__(model_dir)`:
  - Collects hash tables from graph collections; errors if none exist.
  - Groups tables by device and allocates per-device count variables of size `_MAX_SLOT`.
  - Builds `iterate_table_and_apply` ops to accumulate slot counts.
- `before_save`:
  - Skips if multi-hash tables exist.
  - Initializes count vars, runs count op, sums counts across devices.
  - Writes `ckpt_info_pb2.CkptInfo` text to `ckpt.info-<step>`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/hooks/ckpt_info.rs` (new).
- Rust public API surface: saver listener that writes slot counts.
- Feature gating: TF runtime required for table iteration.
- Integration points: hash table iteration helper and checkpoint hooks.

**Implementation Steps (Detailed)**
1. Implement slot-count accumulation using table dump iteration.
2. Serialize `CkptInfo` via protobuf text format.
3. Write `ckpt.info-<step>` file during save.

**Tests (Detailed)**
- Python tests: `ckpt_info_test.py`.
- Rust tests: `monolith-rs/crates/monolith-training/tests/ckpt_info_test.rs` (new).
- Cross-language parity test: compare output file contents for the same table.

**Gaps / Notes**
- `_MAX_SLOT` constant must match; slot extraction uses `extract_slot_from_entry` custom op.

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
