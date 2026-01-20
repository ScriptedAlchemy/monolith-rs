<!--
Source: task/request.md
Lines: 8997-9242 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/distributed_serving_ops.py`
<a id="monolith-native-training-distributed-serving-ops-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 160
- Purpose/role: Remote predict and parameter sync client/server utilities for distributed serving; wraps custom ops for TF Serving RPC and sync.
- Key symbols/classes/functions: `remote_predict`, `create_parameter_sync_clients`, `parameter_sync_client_from_config`, `refresh_sync_config`, `ParameterSyncClient`, `DummySyncServer`.
- External dependencies: `gen_monolith_ops`, `parameter_sync_pb2`, `SyncBackend`, `ServerType`.
- Side effects: creates sync clients/servers on PS devices; uses RPC to remote predict.

**Required Behavior (Detailed)**
- `remote_predict(...)`:
  - Validates `model_name` non-null.
  - Calls `tf_serving_remote_predict` custom op with input/output aliases, model name, task, version, deadline, signature; returns output tensors (index 2 of op result).
- `create_parameter_sync_clients(ps_num)`:
  - For `ps_num==0`, returns single client.
  - Else creates one client per PS on PS device (unless exporting standalone).
- `parameter_sync_client_from_config(config, name_suffix)`:
  - Creates `MonolithParameterSyncClient` op with serialized config and shared_name.
- `refresh_sync_config(sync_backend, ps_index)`:
  - Fetches sync targets; populates `ClientConfig` with targets and extra info; sets model name, signature `hashtable_assign`, timeout 3000ms; returns serialized bytes.
- `create_dummy_sync_client` / `create_dummy_sync_server`:
  - Wrap dummy sync ops.
- `ParameterSyncClient`:
  - `create_sync_op` calls `monolith_parameter_sync` op with client handle and config string.
  - `as_op` wraps client handle with `tf.group`.
- `DummySyncServer`:
  - `shutdown` and `get_port` wrap dummy server ops; `as_op` groups server handle.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/serving` or `monolith-serving`.
- Rust public API surface: remote predict wrapper + parameter sync client/server wrappers.
- Data model mapping: `ClientConfig` protobuf and sync target lists.
- Feature gating: TF runtime + custom ops for remote predict and sync.
- Integration points: distributed PS and export paths.

**Implementation Steps (Detailed)**
1. Implement RPC wrapper for remote predict (TF Serving or custom stub).
2. Implement parameter sync client creation and config refresh logic.
3. Provide dummy client/server for tests.

**Tests (Detailed)**
- Python tests: `distributed_serving_ops_test.py`.
- Rust tests: integration tests for config building and dummy sync ops.
- Cross-language parity test: compare serialized config bytes.

**Gaps / Notes**
- `remote_predict` relies on custom op; Rust likely needs TF C API bindings.

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

### `monolith/native_training/distributed_serving_ops_test.py`
<a id="monolith-native-training-distributed-serving-ops-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 142
- Purpose/role: Tests parameter sync client/server ops and sync config generation for ZK backend and replica watcher.
- Key symbols/classes/functions: `ParameterSyncOpsTest`, `test_parameter_sync_client`, `test_refresh_sync_config_1`, `test_refresh_sync_config_2`.
- External dependencies: `DummySyncServer`, `ParameterSyncClient`, agent_service backend mocks, FakeKazooClient, `parameter_sync_pb2`.
- Side effects: creates dummy sync servers and ZK backend state; uses fake clients.

**Required Behavior (Detailed)**
- `test_parameter_sync_client`:
  - Creates two `DummySyncServer`s, gets ports.
  - Builds `ParameterSyncClient` with targets; creates hash table with `sync_client`.
  - Runs lookup + apply_gradients; expects embeddings `[[0.2,0.2,0.2],[0.1,0.1,0.1]]`.
  - Calls `client.create_sync_op` with config; prints JSON; shuts down servers.
- `test_refresh_sync_config_1`:
  - Mocks `ReplicaWatcher` with FakeKazooClient; sets replica meta with address `localhost:8500`.
  - `refresh_sync_config` should set `model_name='ps_1'` and targets `['localhost:8500']`.
- `test_refresh_sync_config_2`:
  - Sets up ZK backend with container services; syncs available saved models.
  - `refresh_sync_config` with ps_index 1 yields `model_name='test_ffm_model:ps_1'` and targets `['localhost:8888']`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests`.
- Rust public API surface: parameter sync client/server wrappers + config refresh.
- Data model mapping: ZK backend or equivalent; ClientConfig proto.
- Feature gating: requires sync backend mocks or fixtures.
- Integration points: distributed_serving_ops + agent_service backends.

**Implementation Steps (Detailed)**
1. Implement dummy sync server/client wrappers in Rust for test harness.
2. Add tests that apply gradients and verify embedding updates.
3. Add mock backend tests for `refresh_sync_config` with ZK-style data.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: `distributed_serving_ops_test.rs` with mocks.
- Cross-language parity test: compare ClientConfig bytes and target lists.

**Gaps / Notes**
- Depends on agent_service backends; Rust will need lightweight mocks.

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

### `monolith/native_training/distribution_ops.py`
<a id="monolith-native-training-distribution-ops-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 889
- Purpose/role: Wrapper utilities around custom distribution ops for sharding, embedding layout transforms, and gradient backprop helpers.
- Key symbols/classes/functions: `split_by_indices`, `ragged_split_by_indices`, `unique_key_with_value_and_offset`, `fill_with_offset_map`, `finalize_shared_tensor`, `reorder_by_indices`, `fused_reorder_by_indices`, `map_id_to_embedding`, `fused_embedding_to_layout` (+ grads), `map_id_to_embedding_gradient_back_prop`, `fused_gather_embeddings_by_input`, `fused_gather_embeddings_by_input_gradient`, reduce/sorted-segment ops.
- External dependencies: `gen_monolith_ops` custom kernels, `FeatureConfigs` proto.
- Side effects: registers custom gradients for several ops.

**Required Behavior (Detailed)**
- `split_by_indices(indices, tensor, num_splits)`:
  - Calls `monolith_split_by_indices` custom op; gradient registered via `monolith_split_by_indices_gradient`.
- `ragged_split_by_indices(indices, num, num_splits)`:
  - Splits ragged tensor by indices; returns list of ragged tensors + list of corresponding positions.
- `unique_key_with_value_and_offset(key, dims, generate_buffer=True)`:
  - Deduplicates ragged keys and returns `unique_key`, `value_offset` (ragged) and `value_buffer` sized by dims.
- `fill_with_offset_map(pos, value, value_offset_map, value_buffer, dims)`:
  - Fills `value_buffer` positions from offsets; gradient registered via `fill_with_offset_map_gradient`.
- `finalize_shared_tensor(shared_tensor_handles, dtype, shape)`:
  - Finalizes shared tensor handles; gradient returns upstream grad (identity).
- `reorder_by_indices` / `fused_reorder_by_indices`:
  - Reorders input ids by shard indices; `fused_reorder_by_indices` returns packed tensors and offsets for fused pipelines.
- `map_id_to_embedding(ids, embeddings, input)`:
  - Maps sharded embeddings back to original id order; gradient hook registered.
- `fused_embedding_to_layout(embeddings_list, fid_offset, feature_offset, nfl_offset, batch_size, ...)`:
  - Converts flattened embeddings into layout tensors using `FeatureConfigs` and offsets; supports multiple versions and GPU paths.
  - Gradient functions `_fused_embedding_to_layout_grad_v{1..5}` and `fused_embedding_to_layout_grad` wrap custom ops.
- `map_id_to_embedding_gradient_back_prop(ids, input, grads)`:
  - Builds gradients back to sharded embeddings.
- `gather_embeddings_by_input` / `fused_gather_embeddings_by_input`:
  - Gathers embedding vectors using ids and offsets; fused variants use offsets and sizes.
- Reduce ops:
  - `reduce_mean`, `reduce_sum`, `reduce_sqrtn` with custom gradients.
  - `fused_sorted_segment_sum`, `fused_reduce_sum_and_split`, `fused_reduce_and_split_gpu` for GPU fused reductions with gradients.
- `normalize_merged_split(row_split, size)`:
  - Normalizes row split sizes for merged ragged splits.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/ops` (or `monolith-tensor` for ragged).
- Rust public API surface: distribution ops wrappers, gradient-friendly functions if TF backend.
- Data model mapping: ragged tensors, embedding offsets, FeatureConfigs.
- Feature gating: requires custom kernel bindings or reimplementation.
- Integration points: distributed_ps, partitioned hash table, sharding pipelines.

**Implementation Steps (Detailed)**
1. Bind or reimplement each custom op with identical signatures and gradient behavior.
2. Implement ragged split/unique/offset map logic in Rust if not using TF.
3. Support fused embedding to layout and gradient versions used by PartitionedHashTable.
4. Add tests for each op with small deterministic tensors.

**Tests (Detailed)**
- Python tests: `distribution_ops_test.py`, `distribution_ops_fused_test.py`.
- Rust tests: unit tests for each op wrapper; integration tests with PartitionedHashTable.
- Cross-language parity test: compare outputs for fixed inputs and gradient checks.

**Gaps / Notes**
- Many ops are custom kernels; full parity requires substantial backend work.

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

### `monolith/native_training/distribution_ops_benchmark.py`
<a id="monolith-native-training-distribution-ops-benchmark-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 118
- Purpose/role: Benchmarks `map_id_to_embedding` and `gather_embeddings_by_input` with/without multi-threading; optional profiler output.
- Key symbols/classes/functions: `DistributionOpsBenchmarkTest.map_id_to_embedding`, `test_gather_embeddings_by_ids_basic`, `test_gather_embeddings_by_ids_multi_threads`.
- External dependencies: TensorFlow profiler, `distribution_ops`.
- Side effects: writes profiler logs under `/tmp/distribution_ops_benchmark/*`.

**Required Behavior (Detailed)**
- `map_id_to_embedding(use_multi_threads)`:
  - Creates 1e6 ids, dim=16, ps_num=10; splits ids/embeddings and maps back.
  - Asserts mapped embeddings equal original; starts/stops TF profiler in log dir.
- `test_gather_embeddings_by_ids_basic`:
  - Benchmarks gather with 100k features; runs for dim=32 and dim=256, different input lengths.
- `test_gather_embeddings_by_ids_multi_threads`:
  - Same as above but `use_multi_threads=True`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/benches`.
- Rust public API surface: bench harness for distribution ops.
- Data model mapping: ids/embeddings tensors.
- Feature gating: multi-threaded execution support.
- Integration points: `distribution_ops` implementation.

**Implementation Steps (Detailed)**
1. Implement Rust benchmarks for `map_id_to_embedding` and `gather_embeddings_by_input`.
2. Mirror tensor sizes and check correctness for small sizes; run benchmarks for large sizes.
3. Add optional profiling hooks.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: benches or microbench harness.
- Cross-language parity test: not required beyond correctness checks.

**Gaps / Notes**
- Pure benchmark; not a correctness test.

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
