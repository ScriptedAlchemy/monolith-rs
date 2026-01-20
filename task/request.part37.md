<!--
Source: task/request.md
Lines: 8798-8996 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/distributed_ps_sync.py`
<a id="monolith-native-training-distributed-ps-sync-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 531
- Purpose/role: Horovod/BytePS synchronous all-to-all embedding lookup and update for distributed multi-type hash tables.
- Key symbols/classes/functions: `DistributedMultiTypeHashTableMpi.lookup`, `.apply_gradients`, `.as_op`.
- External dependencies: Horovod/BytePS env flags, `distribution_ops`, `feature_utils` (control/dense_opt ops), `prefetch_queue`.
- Side effects: uses enqueue queues for pipelined execution; emits alltoall metrics summaries when enabled.

**Required Behavior (Detailed)**
- Environment flags:
  - `MONOLITH_WITH_HOROVOD`, `MONOLITH_WITH_OPTIMIZED_HOROVOD`, `MONOLITH_WITH_BYTEPS` and related G2G/GDR flags determine alltoall backend and GPU paths.
  - `FLAGS.enable_alltoall_metrics` + `enable_alltoall_metrics_for_slot` control summary emission.
- `DistributedMultiTypeHashTableMpi.__init__(shard_num, table_factory, queue_configs)`:
  - Determines rank from BytePS or Horovod; builds local shard table via `table_factory`.
  - Stores output dims, queue configs, and dependency ops.
- `lookup(slot_to_id, auxiliary_bundle, early_reorder_indicies_res_pack)`:
  - Requires `early_reorder_indicies_res_pack` (support for `reorder_fids_in_data_pipeline=False` dropped).
  - Unpacks `(all_fids, shard_sizes, sharded_slot_sizes, emb_offset_sz, fused_embedding_offsets, req_time)`.
  - Performs alltoall on fids and per-slot sizes via BPS/HVD/custom optimized HVD.
  - Stores key tensors in `auxiliary_bundle` (id_flat_t, id_size_flat_t, emb offsets, recv splits, etc.).
  - Calls `self._table.fused_lookup(...)` on GPU, yielding `fused_embeddings`, splits, offsets, indices.
  - Performs embedding alltoall (fwd) and queues prefetch if configured.
  - Uses `distribution_ops.fused_gather_embeddings_by_input` to assemble per-slot embeddings on GPU.
  - Returns `(slot_to_embedding, auxiliary_bundle)`.
- `apply_gradients(slot_to_grad, auxiliary_bundle, global_step, req_time, scale)`:
  - Uses `feature_utils.control_ops` dependency.
  - Computes `grad_flat` via `fused_gather_embeddings_by_input_gradient`.
  - Optionally casts for BPS bwd.
  - Enqueues async optimize queue if configured.
  - Performs backward alltoall using BPS/HVD/custom optimized HVD.
  - Emits alltoall metrics summaries when enabled.
  - Calls `self._table.fused_apply_gradient` with id/grad buffers and offsets.
  - Supports async optimize queue via `AsyncPushHook`.
- `assign/assign_add/reinitialize`:
  - Not implemented (raises `NotImplementedError`).
- `as_op`:
  - Returns `self._table.as_op` with dependency ops.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/ps`.
- Rust public API surface: synchronous alltoall embedding lookup/update for multi-type tables.
- Data model mapping: packed fid buffers and fused embedding offsets.
- Feature gating: Horovod/BytePS support; GPU alltoall paths.
- Integration points: `distributed_ps_factory.create_in_worker_multi_type_hash_table`.

**Implementation Steps (Detailed)**
1. Implement Rust backend selection for alltoall (HVD/BPS equivalents) or gate feature.
2. Port `fused_lookup` + `fused_gather_embeddings_by_input` and gradient counterparts.
3. Preserve auxiliary_bundle keys and queue-based pipelining.
4. Mirror alltoall metric summaries (if logging/metrics available in Rust).

**Tests (Detailed)**
- Python tests: `distributed_ps_sync_test.py`.
- Rust tests: integration tests with small shard_num and deterministic ids.
- Cross-language parity test: compare embeddings and gradients for small fixtures.

**Gaps / Notes**
- Requires GPU kernels and alltoall comms; may need staged parity.

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

### `monolith/native_training/distributed_ps_sync_test.py`
<a id="monolith-native-training-distributed-ps-sync-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 109
- Purpose/role: Validates synchronous alltoall distributed multi-type hash table lookup and gradient updates under Horovod.
- Key symbols/classes/functions: `DistributedMultiTypeHashTableMpiTest.testBasic`, `gen_test_configs`.
- External dependencies: Horovod, `distribution_ops.fused_reorder_by_indices`, `distributed_ps_sync.DistributedMultiTypeHashTableMpi`.
- Side effects: sets `MONOLITH_WITH_HOROVOD=True` and initializes Horovod.

**Required Behavior (Detailed)**
- `gen_test_configs()`:
  - Builds two test hash table configs: slot "1" dim=1 lr=1.0; slot "2" dim=2 with PolynomialDecay LR.
- `testBasic(use_native_multi_hash_table=False)`:
  - Initializes Horovod, global_step=0.
  - Creates table with `DistributedMultiTypeHashTableMpi(hvd.size(), table_factory)`.
  - `slot_to_ids = {"1": [1,1], "2": [2]}`.
  - Uses `distribution_ops.fused_reorder_by_indices` to produce `reordred` pack (plus None timestamp).
  - First lookup returns zeros.
  - Applies gradients `{1: [[0.5],[0.5]], 2: [[0.5,1.0]]}` with `global_step=0`.
  - Second lookup returns negative values scaled by `hvd.size()`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests`.
- Rust public API surface: `DistributedMultiTypeHashTableMpi` and reorder helpers.
- Data model mapping: slot configs, id arrays, gradient arrays.
- Feature gating: Horovod alltoall support.
- Integration points: `distributed_ps_sync` implementation.

**Implementation Steps (Detailed)**
1. Add Rust test that initializes the sync table and performs lookup + apply_gradients.
2. Implement fused reorder (or provide equivalent packed inputs).
3. Validate outputs match expected scaled negatives.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: `distributed_ps_sync_test.rs` with small fixed ids.
- Cross-language parity test: compare outputs for same ids and gradients.

**Gaps / Notes**
- Requires Horovod or equivalent alltoall backend; gate test if unavailable.

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

### `monolith/native_training/distributed_ps_test.py`
<a id="monolith-native-training-distributed-ps-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 979
- Purpose/role: Comprehensive tests for distributed hash tables, multi-type hash tables, export behavior, and partitioned hash table lookup/apply gradients (CPU/GPU).
- Key symbols/classes/functions: `DistributedHashTableTest`, `DistributedMultiTypeHashTableTest`, `DistributedMultiTypeHashTableServingTest`, `PartitionedHashTableTest`.
- External dependencies: TF PS clusters, Horovod env, `distribution_ops`, `export_context`, `sharding_sparse_fids_with_context`.
- Side effects: sets `MONOLITH_WITH_HOROVOD=1` and uses test clusters.

**Required Behavior (Detailed)**
- `DistributedHashTableTest`:
  - `test_basic`: assign_add then lookup equals assigned values.
  - `test_assign`: second assign overwrites ids; lookup after control dependency yields updated values.
  - `test_lookup_dedup`: duplicate ids return repeated embeddings.
  - `test_apply_gradients`: gradients with loss `2*values` updates to `-2` for dim=1.
  - `test_apply_gradients_with_learning_rate_function`: polynomial decay learning rate affects updates; after global_step increment, values change to `-4.2`.
  - `test_apply_gradients_with_duplicates`: duplicate ids produce accumulated gradient; expected `-4` for duplicate id.
  - `test_apply_gradients_with_different_ids`: bp_ids differ from ids; updates only bp ids.
- `DistributedMultiTypeHashTableTest` (param native vs non-native):
  - `testBasic`: assign_add per slot, lookup values, apply_gradients halves values.
  - `test_assign_and_reinitialize`: assign then assign with half values; native mode tests `reinitialize` status and zeros for slot.
  - `test_apply_gradients_with_learning_rate_function`: similar to single-table with polynomial decay; values update with global_step.
  - `test_apply_gradients_float16`: transfer_float16 path; verifies lookup output after apply gradients.
- `DistributedMultiTypeHashTableServingTest`:
  - `test_export_model`: export distributed/standalone/normal training and ensure lookup shapes; verifies `export_ctx.sub_graph_num`.
- `PartitionedHashTableTest`:
  - Helpers: `gen_table_config`, `gen_out_config`, `get_parser_ctx`, `gen_data`, `gen_variant_tensor`.
  - `_test_basic`: assign + assign_add and `_lookup_raw` should return sum of embeddings; runs CPU and GPU variants.
  - `_test_lookup`: assigns const embeddings, sharding sparse fids + lookup yields expected layout tensors (`bias`, `vec`, `deep`).
  - `_test_apply_gradients`: assigns const values, lookup+apply_gradients; verifies updated embeddings against expected FTRL/AdaGrad formulas.
  - `test_apply_gradients_for_gpu_emb`: compares GPU embedding path with CPU path using same gradients; outputs must match.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests`.
- Rust public API surface: distributed hash tables, multi-type tables, partitioned hash table API.
- Data model mapping: fids/embeddings, layout configs, and gradient updates.
- Feature gating: PS clusters, Horovod, GPU embedding path.
- Integration points: `distributed_ps`, `distributed_ps_sync`, `distribution_ops`.

**Implementation Steps (Detailed)**
1. Port test utilities to build PS clusters and configs in Rust (or provide Python-driven fixtures).
2. Recreate expected numeric outputs for assign/lookup/apply_gradients.
3. Implement layout config generation and sharding for partitioned hash table tests.
4. Add GPU embedding parity tests comparing CPU and GPU paths.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: `distributed_ps_test.rs` covering key cases above.
- Cross-language parity test: compare lookup/apply_gradients outputs for small fixed inputs.

**Gaps / Notes**
- File is extensive; ensure Rust tests focus on correctness for representative cases if full coverage is too costly.

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
