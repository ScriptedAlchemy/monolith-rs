<!--
Source: task/request.md
Lines: 8389-8613 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/distribute/str_queue.py`
<a id="monolith-native-training-distribute-str-queue-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 114
- Purpose/role: A TF-based string queue with save/restore support, critical section synchronization, and optional auto-enqueue when empty.
- Key symbols/classes/functions: `StrQueue`, `enqueue_many`, `dequeue`, `_raw_enqueue_many`, `_raw_dequeue`.
- External dependencies: TensorFlow `CriticalSection`, variables, tf.function.
- Side effects: maintains internal TF variables (`_arr`, `_offset`, `_arr_size`), uses critical section for synchronization.

**Required Behavior (Detailed)**
- `StrQueue.__init__(initial_elements, critical_section, auto_enqueue_fn, capacity, name)`:
  - Creates a shared `CriticalSection` (or reuses provided).
  - Initializes `_arr` (string array of size `capacity`), `_offset`, `_arr_size` as variables.
  - Enqueues `initial_elements` during initialization via control deps.
  - Uses `_var_for_init` dummy variable to ensure initial enqueue runs.
- `enqueue_many(elements)`:
  - Converts to string tensor and calls `_raw_enqueue_many` inside critical section.
- `dequeue()`:
  - Executes `_raw_dequeue` inside critical section; returns `(element, out_of_range)`.
- `_raw_enqueue_many(elements)` (tf.function):
  - Computes `old_arr_size = _arr_size - _offset`, `new_arr_size = old_arr_size + size(elements)`.
  - Asserts `new_arr_size <= capacity`.
  - Compacts array by shifting remaining elements to front, appends new elements, resets `_offset` to 0, updates `_arr_size`.
- `_raw_dequeue()` (tf.function):
  - Asserts `_offset <= _arr_size`.
  - If `auto_enqueue_fn` provided, loops while empty: calls auto fn to get `(elements, out_of_range)`; enqueues elements unless out_of_range.
  - If still empty, returns `("", True)`.
  - Else returns element at `_offset` and increments `_offset`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/src` (distributed queues).
- Rust public API surface: string queue with enqueue/dequeue and optional auto-fill.
- Data model mapping: TF variables → Rust in-memory or shared queue state.
- Feature gating: requires distributed synchronization if used across workers.
- Integration points: `distributed_dataset.create_dynamic_sharding_dataset`.

**Implementation Steps (Detailed)**
1. Implement a thread-safe queue with capacity and offset/size semantics.
2. Provide auto-enqueue hook that is called when empty.
3. Match out_of_range behavior and empty return value (`""`).
4. If using TF runtime, preserve CriticalSection semantics for shared state.

**Tests (Detailed)**
- Python tests: `str_queue_test.py`.
- Rust tests: queue enqueue/dequeue, auto-enqueue loop, capacity assert.
- Cross-language parity test: compare sequence of dequeued elements for the same auto-enqueue function.

**Gaps / Notes**
- TF CriticalSection semantics may need a custom mutex + barrier if implemented in Rust.

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

### `monolith/native_training/distribute/str_queue_test.py`
<a id="monolith-native-training-distribute-str-queue-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 67
- Purpose/role: Tests basic enqueue/dequeue behavior, initialization, out-of-range handling, and auto-enqueue logic for `StrQueue`.
- Key symbols/classes/functions: `QueueTest` with `testBasic`, `testInit`, `testOutOfRange`, `testAutoEnqueue`.
- External dependencies: TensorFlow, `str_queue.StrQueue`.
- Side effects: none.

**Required Behavior (Detailed)**
- `testBasic`:
  - Enqueues `test1`, `test2` and dequeues in order.
- `testInit`:
  - Initializes queue with `initial_elements=['test1']` and dequeues `test1`.
- `testOutOfRange`:
  - Dequeue from empty queue returns `out_of_range=True`.
- `testAutoEnqueue`:
  - `auto_enqueue` increments variable `v` and enqueues stringified values until `v > 2`, then returns out_of_range.
  - Dequeues yield `"1"`, `"2"`, then `out_of_range=True` for subsequent dequeues.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/tests`.
- Rust public API surface: `StrQueue` equivalent.
- Data model mapping: string queue semantics.
- Feature gating: none.
- Integration points: `distributed_dataset` uses StrQueue.

**Implementation Steps (Detailed)**
1. Add Rust tests that validate enqueue/dequeue ordering and init behavior.
2. Implement auto-enqueue hook test with controlled counter.
3. Verify out_of_range behavior persists after exhaustion.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: `str_queue_test.rs`.
- Cross-language parity test: compare dequeued sequences for fixed auto-enqueue behavior.

**Gaps / Notes**
- TensorFlow session semantics are not required; Rust can implement a pure queue test.

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

### `monolith/native_training/distributed_ps.py`
<a id="monolith-native-training-distributed-ps-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 2108
- Purpose/role: Distributed parameter server embedding hash table implementation (single-type and multi-type), sharded lookup/apply gradients, fused layout embedding pipelines, and GPU/Horovod/BytePS all-to-all paths.
- Key symbols/classes/functions: `ps_device`, `DistributedHashTable`, `DistributedMultiTypeHashTable`, `PartitionedHashTable`, `get_sub_table_name`, `PartitionedHashTable.gen_feature_configs`, `merge_feature_config`, `lookup`, `apply_gradients`, `_lookup_gpu`, `_apply_gradients_gpu`.
- External dependencies: TensorFlow, custom ops (`distribution_ops`, `multi_hash_table_ops`), `export_context`, `prefetch_queue`, `hvd`/`bps` (if enabled), FeatureConfigs protos.
- Side effects: creates PS-side graphs/signatures during export; emits lookup timers; enqueues prefetch queues; uses global parser context for sharding configs.

**Required Behavior (Detailed)**
- `ps_device(i)`:
  - Context manager that clears device stack (`colocate_with(None, True)`) and sets device to `utils.ps_device(i)` for PS ops.
- `DistributedHashTable` (single-type table):
  - Constructor builds per-PS tables; sends learning-rate tensors to each PS (unless exporting standalone).
  - `lookup(ids)`:
    - `tf.unique` ids, shard by `id % ps_num`, lookup on each PS, and `map_id_to_embedding` back to original order.
    - Tracks input/output tensors for backprop.
  - `assign/assign_add`: split ids/values by PS and call underlying table method.
  - `apply_gradients`: unique ids, split gradients with `map_id_to_embedding_gradient_back_prop`, apply on each PS (dedup disabled).
  - `as_op`: aggregates PS table ops.
- `DistributedMultiTypeHashTable` (multi-slot table):
  - Builds per-PS multi-type tables; supports raw API if tables are `RawMultiTypeHashTable`.
  - Export mode builds PS subgraphs with `lookup` and optional `raw_lookup` signatures.
  - `lookup(slot_to_id)`:
    - Raw API path: uses ragged IDs and `unique_key_with_value_and_offset` to reduce duplicate lookups, splits by PS, uses raw lookup and `fill_with_offset_map`, then reconstructs embeddings; returns only requested slots.
    - Non-raw path: per-slot unique/split, PS lookup (remote_predict if exporting distributed); maps back by `map_id_to_embedding`.
  - `assign/assign_add`: per-slot split by PS and call underlying table methods.
  - `reinitialize(slot, ids)`: raw-only; splits ids and concatenates status.
  - `apply_gradients`: raw path uses `raw_apply_gradients` with fused flat grads; non-raw path packs keyed tensors, optional float16 transfer.
  - `as_op` combines PS tables; `get_table_dim_sizes` delegates to cc dims.
- `get_sub_table_name(strs)`:
  - Returns `(concat, md5(concat))` for merged table naming.
- `PartitionedHashTable`:
  - `gen_feature_configs`: builds `FeatureConfigs` and `ShardingSparseFidsOpParams` based on feature configs and combiners; supports native multi-hash-table and GPU embedding modes.
  - `merge_feature_config` / `no_merge_feature_config`: compute merged sub-table names (with md5) or keep per-feature tables; handles `fc_slot_` → `slot_` extra restore names.
  - Constructor:
    - Reads `parser_ctx.sharding_sparse_fids_op_params` for PS count, native multi-table mode, feature configs, and GPU options.
    - Creates per-PS tables or GPU table; builds export signatures for lookup/raw_lookup when exporting.
    - Sets up learning-rate tensors for each sub-table.
  - `lookup(features, auxiliary_bundle, ...)`:
    - If GPU embedding enabled, delegates to `_lookup_gpu` and optionally returns callable.
    - Otherwise obtains sharded fids via `sharding_sparse_fids` or `ParserCtx`-encoded features, stores offsets and sizes in `auxiliary_bundle`.
    - Optionally returns `lookup_callable_fn` or `fused_layout_callable_fn` for two-phase lookup.
    - `call_lookup`:
      - Uses raw/native lookup or packed lookup; remote_predict in export mode.
      - Stores per-PS embeddings and optional fids/row_splits in `auxiliary_bundle`.
      - Optionally moves auxiliary tensors to GPU and enqueues prefetch queues.
    - `fused_layout_callable_fn`:
      - Calls `distribution_ops.fused_embedding_to_layout` to reconstruct layout embeddings (CPU/GPU depending on export or `_use_gpu`).
      - Uses `nest_layout` to produce output dict.
  - `apply_gradients(layout_grads_and_vars, global_step, req_time, auxiliary_bundle, async_function_mgr, async_push, grad_scale)`:
    - For non-GPU path: uses `fused_embedding_to_layout_grad` to compute per-PS grads, then applies via raw or packed update; supports async push queues.
    - Includes tensor move CPU helper for GPU-derived tensors.
    - For GPU path, delegates to `_apply_gradients_gpu`.
  - `_lookup_gpu`:
    - Uses all-to-all (HVD/BPS/custom) to exchange ids and embeddings; calls `fused_lookup` on GPU table; then all-to-all embeddings back; finally `fused_embedding_to_layout` (version 4) and `nest_layout`.
    - Populates `auxiliary_bundle` with many intermediate tensors (id_flat_t, splits, offsets, recv embeddings, etc.) and optional pipeline queues.
  - `_apply_gradients_gpu`:
    - Computes `fused_embedding_to_layout_grad` on GPU, performs all-to-all backprop (HVD/BPS/custom), and calls `fused_apply_gradient` on GPU table; supports async optimize queues.
  - `assign/assign_add`:
    - Non-GPU only; routes to `_update` or `_native_hash_table_update` depending on native multi-table mode.
  - `flatten_layout` / `nest_layout`:
    - Deterministic ordering by `feature_configs.out_configs` (sorted names); `OutType.NONE` yields list per slices.
  - Queue hooks:
    - `add_queue_hook` stores local hooks; `get_queue_hooks` collects hooks from tables.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/ps` + `monolith-hash-table` + `monolith-data`.
- Rust public API surface: distributed hash table abstractions, partitioned multi-type tables, lookup/apply_gradients APIs.
- Data model mapping: FeatureConfigs and sharded fids, embedding layouts, and fused layout conversions.
- Feature gating: export mode, raw API support, GPU embedding, Horovod/BytePS all-to-all.
- Integration points: `parsers.sharding_sparse_fids`, `embedding_combiners`, `prefetch_queue`, export signatures.

**Implementation Steps (Detailed)**
1. Implement Rust equivalents for `DistributedHashTable` and `DistributedMultiTypeHashTable` with sharding by `id % ps_num`.
2. Recreate packed tensor transfer and optional float16 transport.
3. Implement `PartitionedHashTable` with sharding feature configs and `fused_embedding_to_layout`/`_grad` equivalents.
4. Add GPU embedding path + all-to-all (if supported); otherwise gate behind features.
5. Mirror export signatures for PS-side lookups.
6. Port queue hook logic for pipelined execution.

**Tests (Detailed)**
- Python tests: `distributed_ps_test.py`, `distributed_ps_sync_test.py`, `distribution_ops_test.py`.
- Rust tests: integration tests for lookup/apply_gradients with small sharded tables and layout configs.
- Cross-language parity test: compare embedding outputs for fixed ids across PS shards.

**Gaps / Notes**
- This module is large and deeply tied to TF custom ops; full parity likely requires a TF backend or substantial Rust kernel work.
- GPU/Horovod/BytePS paths are specialized; may need staged parity plan.

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
