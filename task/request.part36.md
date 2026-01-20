<!--
Source: task/request.md
Lines: 8614-8797 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/distributed_ps_benchmark.py`
<a id="monolith-native-training-distributed-ps-benchmark-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 168
- Purpose/role: Benchmark tests for distributed hash table lookup and apply_gradients performance, optionally with profiling.
- Key symbols/classes/functions: `_generate_config`, `_get_vocab_hash_table_factory`, `DistributedHashTableTest.lookup`, `DistributedHashTableTest.apply_gradients`.
- External dependencies: TensorFlow local servers, `distributed_ps.DistributedHashTable`, `hash_filter_ops`, `hash_table_ops`, `embedding_hash_table_pb2`.
- Side effects: creates local PS servers, may write profiler logs under `/tmp/distributed_ps_benchmark`.

**Required Behavior (Detailed)**
- `_generate_config(servers, job_name=utils.PS_JOB_NAME)`:
  - Builds `ClusterDef` with job tasks derived from server targets; returns `ConfigProto`.
- `_get_vocab_hash_table_factory(dim)`:
  - Returns factory that builds a hash table with `EmbeddingHashTableConfig` using cuckoo + SGD(1.0) + zeros init and segment dim `dim`.
- `DistributedHashTableTest.lookup(enable_dedup, real_run=True)`:
  - Creates `ps_num=10` local servers; uses server0 with cluster config.
  - Builds hash filters and a `DistributedHashTable`, assigns add for ids 0..num_elements-1.
  - If `real_run`: lookup ids `x//2`, check values equal `x//2` repeated per dim; prints wall time; optional profiler.
  - If `real_run=False`: just runs `hash_table.as_op()` to measure overhead.
- `apply_gradients(real_run=True)`:
  - Similar setup; assigns ones to embeddings, looks up, computes `loss=0.3*embeddings`, grads; applies gradients.
  - If `real_run`: after apply, looks up and expects values `0.4` (1.0 + 0.3*?); prints timing.
  - If `real_run=False`: checks grads equal `0.3` if not profiling.
- Tests invoke both real and overhead modes.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/benches`.
- Rust public API surface: benchmark harness for distributed PS table lookup/apply gradients.
- Data model mapping: hash table config, embedding values.
- Feature gating: requires distributed PS runtime and hash table ops.
- Integration points: `distributed_ps` implementation.

**Implementation Steps (Detailed)**
1. Implement a Rust benchmark that spins up local PS servers (or mock) and measures lookup/apply_gradients.
2. Mirror data sizes (1e6 ids, dim=16) and expected outputs.
3. Optionally add profiling hooks matching Python behavior.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: bench-only; optional correctness assertions for small sizes.
- Cross-language parity test: compare outputs for small benchmark sizes.

**Gaps / Notes**
- Uses TF local servers and profiling; Rust may need a simplified harness.

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

### `monolith/native_training/distributed_ps_factory.py`
<a id="monolith-native-training-distributed-ps-factory-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 262
- Purpose/role: Factory helpers to build distributed or local multi-type hash tables and partitioned hash tables with different network/packet-reduction strategies.
- Key symbols/classes/functions: `MultiHashTableFactory`, `create_in_worker_multi_type_hash_table`, `create_multi_type_hash_table`, `create_native_multi_hash_table`, `create_in_worker_native_multi_hash_table`, `create_partitioned_hash_table`.
- External dependencies: `distributed_ps`, `distributed_ps_sync`, `hash_table_ops`, `hash_filter_ops`, `multi_type_hash_table`, `multi_hash_table_ops`, `entry.HashTableConfigInstance`.
- Side effects: none beyond table creation.

**Required Behavior (Detailed)**
- `MultiHashTableFactory`:
  - Caches converted configs via `multi_hash_table_ops.convert_to_cached_config` keyed by `id(slot_to_config)`.
  - `__call__(idx, slot_to_config)` returns `MultiHashTable.from_cached_config` using hash_filter and sync_client for shard `idx`.
- `create_in_worker_multi_type_hash_table(shard_num, slot_to_config, hash_filter, sync_client, queue_configs)`:
  - Builds a `MergedMultiTypeHashTable` whose underlying factory is `DistributedMultiTypeHashTableMpi` (alltoall) created from a per-worker `MultiTypeHashTable` factory.
- `create_multi_type_hash_table(num_ps, slot_to_config, hash_filters, sync_clients, reduce_network_packets, max_rpc_deadline_millis)`:
  - Validates sync_clients length; fills with None if missing.
  - `num_ps==0`: returns local `MergedMultiTypeHashTable` backed by `MultiTypeHashTable` and local hash tables.
  - `reduce_network_packets=False`: uses `DistributedHashTable` per slot within `MultiTypeHashTable` (dedup on worker, distribute to PS).
  - `reduce_network_packets=True`: uses `DistributedMultiTypeHashTable` (multi-type on PS) to reduce RPC count.
- `create_native_multi_hash_table(num_ps, slot_to_config, hash_filters, sync_clients, max_rpc_deadline_millis)`:
  - `num_ps==0`: returns local `MultiHashTable.from_configs`.
  - Else returns `DistributedMultiTypeHashTable` with `MultiHashTableFactory`.
- `create_in_worker_native_multi_hash_table(shard_num, slot_to_config, hash_filter, sync_client, queue_configs)`:
  - Returns `DistributedMultiTypeHashTableMpi` with a local native `MultiHashTable` per shard.
- `create_partitioned_hash_table(num_ps, use_native_multi_hash_table, max_rpc_deadline_millis, hash_filters, sync_clients, enable_gpu_emb, queue_configs)`:
  - Normalizes hash_filters/sync_clients lists.
  - Chooses `multi_type_factory` based on native vs non-native multi-hash table:
    - Native: `MultiHashTableFactory`.
    - Non-native: `MultiTypeHashTable` with hash tables created per PS.
  - Returns `distributed_ps.PartitionedHashTable` with queue configs.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/ps`.
- Rust public API surface: factory functions for hash table backends (local vs distributed).
- Data model mapping: slot configs → hash tables; shard selection logic.
- Feature gating: `reduce_network_packets`, native multi-hash table, GPU embedding.
- Integration points: used by training setup to instantiate embedding tables.

**Implementation Steps (Detailed)**
1. Implement Rust factories mirroring the three strategies (local, distributed per slot, distributed multi-type).
2. Preserve caching for expensive config conversion (if needed).
3. Ensure `num_ps==0` uses local tables and no RPC.
4. Add `PartitionedHashTable` factory with queue config propagation.

**Tests (Detailed)**
- Python tests: `distributed_ps_factory_test.py`.
- Rust tests: unit tests for factory selection logic and returned table types.
- Cross-language parity test: compare selected strategy for given flags.

**Gaps / Notes**
- Depends on TF custom ops for hash tables; Rust must provide equivalent backends.

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

### `monolith/native_training/distributed_ps_factory_test.py`
<a id="monolith-native-training-distributed-ps-factory-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 87
- Purpose/role: Smoke tests for distributed hash table factory functions; primarily checks that constructors run without errors.
- Key symbols/classes/functions: `_get_test_slot_to_config`, `_get_test_hash_filters`, `FactoryTest`.
- External dependencies: Horovod (enabled via env), TensorFlow PS cluster utilities, `test_utils.generate_test_hash_table_config`.
- Side effects: sets `MONOLITH_WITH_HOROVOD=True`; may initialize Horovod.

**Required Behavior (Detailed)**
- `_get_test_slot_to_config()`:
  - Uses `test_utils.generate_test_hash_table_config(4, learning_rate=0.1)`; returns slot map with keys `"1"`, `"2"`.
- `_get_test_hash_filters(num)`:
  - Returns `hash_filter_ops.create_hash_filters(num, False)`.
- Tests:
  - `test_create_in_worker_multi_type_hash_table*`: calls `create_in_worker_multi_type_hash_table` with hvd initialized.
  - `test_create_multi_type_hash_table_0_ps`: local (no PS) creation.
  - `test_create_multi_type_hash_table_2_ps`: creates PS cluster and calls factory under a session.
  - `test_create_multi_type_hash_table_2_ps_with_reduced_packets`: same with `reduce_network_packets=True`.
  - `test_create_native_multi_hash_table_0_ps` and `_2_ps`: native multi-hash table creation.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests`.
- Rust public API surface: factory functions in `distributed_ps_factory`.
- Data model mapping: slot configs to table instances.
- Feature gating: Horovod/PS cluster support.
- Integration points: distributed PS creation.

**Implementation Steps (Detailed)**
1. Add Rust smoke tests to ensure factory functions are callable in local/PS modes.
2. If Horovod not supported, gate tests accordingly.
3. Verify hash filter creation and config plumbing.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: `distributed_ps_factory_test.rs` (smoke only).
- Cross-language parity test: not required beyond constructor success.

**Gaps / Notes**
- These are grammar/smoke tests, not functional correctness tests.

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
