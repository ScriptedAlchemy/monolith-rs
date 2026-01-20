<!--
Source: task/request.md
Lines: 684-4603 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
## Per-File Parity Checklists (All Python Files)

Every file listed below must be fully mapped to Rust with parity behavior verified.

### `monolith/__init__.py`
<a id="monolith-init-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 55
- Purpose/role: Package bootstrap that re-exports key submodules and enables TensorFlow monkey patching.
- Key symbols/classes/functions: `add_module`, side-effect imports of `data`, `layers`, `model_export`, `entry`, `native_model` (as `base_model`), `estimator`.
- External dependencies: `tensorflow.python.tools.module_util` (imported), `absl.logging`, `importlib`, `monolith.utils.enable_monkey_patch`.
- Side effects: imports training modules on import; injects modules into `sys.modules`; may modify TensorFlow monitored_session behavior.

**Required Behavior (Detailed)**
- `add_module(module)`:
  - Accepts a module object or a module string.
  - If a string, imports it; derives `name` from the last path component.
  - If `name == 'native_model'`, rename to `'base_model'`.
  - Registers module in `sys.modules` under `monolith.<name>`.
- On import:
  - Calls `add_module` for `data`, `layers`, `model_export`, `entry`, `native_model` (as `base_model`), `estimator`.
  - Calls `enable_monkey_patch()`; on exception, logs `enable_monkey_patch failed`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/src/lib.rs` (crate-level re-exports), plus `monolith-rs/crates/monolith-training` for training APIs.
- Rust public API surface: `pub use` re-exports of subcrates/modules to mimic `monolith.*` namespace.
- Data model mapping: N/A (module wiring only).

**Implementation Steps (Detailed)**
1. Define a top-level Rust crate that re-exports equivalent submodules (data/layers/training/export/estimator).
2. Ensure any TF monkey patch logic is represented (or explicitly documented as unsupported) in Rust.
3. Avoid heavy side effects on import; if unavoidable, document and feature-gate.

**Tests (Detailed)**
- Python tests: covered indirectly by `monolith/utils_test.py` monkey patch test.
- Rust tests: add a smoke test that `monolith` re-exports are accessible.

**Gaps / Notes**
- Python import side effects are heavy; Rust should provide explicit init if needed.

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

### `monolith/native_training/file_ops.py`
<a id="monolith-native-training-file-ops-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 51
- Purpose/role: TensorFlow custom-op wrappers for writing to files inside graph execution; provides a close hook.
- Key symbols/classes/functions: `WritableFile`, `FileCloseHook`.
- External dependencies: TensorFlow, `gen_monolith_ops` (custom ops).
- Side effects: Creates/updates files on disk via custom ops.

**Required Behavior (Detailed)**
- `WritableFile.__init__(filename)`:
  - Calls `monolith_writable_file(filename)` and stores a resource handle.
- `WritableFile.append(content)`:
  - Appends a 0-D string tensor to the file (via `monolith_writable_file_append`).
- `WritableFile.append_entry_dump(item_id, bias, embedding)`:
  - Calls `monolith_entry_dump_file_append(handle, item_id, bias, embedding)`.
  - Used for embedding entry dump format (custom op handles encoding).
- `WritableFile.close()`:
  - Calls `monolith_writable_file_close(handle)`.
- `FileCloseHook(files)`:
  - `files` must be a `list` of `WritableFile`.
  - Builds `_close_ops = [f.close() for f in files]`.
  - `end(session)` runs all close ops.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/file_ops.rs` (new).
- Rust public API surface:
  - `struct WritableFile { handle: ... }`
  - `impl WritableFile { fn new(path: &str) -> Self; fn append(&self, s: &str) -> Op; fn append_entry_dump(...); fn close(&self) -> Op }`
  - `struct FileCloseHook` implementing a hook/callback trait.
- Data model mapping:
  - TF custom ops ↔ Rust TF runtime wrapper (if using TF backend).
  - Native backend may use direct file I/O instead of graph ops.
- Feature gating:
  - TF runtime required for graph-op parity; native mode may use std::fs writes.
- Integration points:
  - Used by summary/export logic that writes per-worker outputs.

**Implementation Steps (Detailed)**
1. Add Rust wrappers for the custom file ops (or a native file writer in non-TF mode).
2. Ensure `append` supports scalar string tensors and preserves order.
3. Implement `append_entry_dump` with the same encoding as the TF op.
4. Provide a hook that closes files at session end.

**Tests (Detailed)**
- Python tests: `monolith/native_training/file_ops_test.py`.
- Rust tests:
  - `writable_file_basic` (append 1000 times and verify content).
  - `file_close_hook_runs` (MonitoredSession equivalent if TF backend).
- Cross-language parity test:
  - Compare output file contents after a fixed sequence of appends.

**Gaps / Notes**
- `WritableFile.append` assumes a scalar string tensor; no Python-side validation.

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

### `monolith/native_training/file_ops_test.py`
<a id="monolith-native-training-file-ops-test-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 56
- Purpose/role: Tests `WritableFile` append behavior and `FileCloseHook` closure semantics.
- Key symbols/classes/functions: `WritableFileTest.test_basic`, `WritableFileTest.test_hook`.
- External dependencies: TensorFlow, `file_ops`, `tf.io.gfile`.
- Side effects: Writes files under `TEST_TMPDIR`.

**Required Behavior (Detailed)**
- `test_basic`:
  - Create `WritableFile`, append `"1234"` in a `tf.function` loop `times=1000`, close.
  - Verify file content equals `"1234" * times`.
- `test_hook`:
  - Create `WritableFile`, run append in a `MonitoredSession` with `FileCloseHook`.
  - Verify file content equals `"1234"`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests/file_ops.rs` (new).
- Rust public API surface: `WritableFile`, `FileCloseHook`.
- Data model mapping: TF runtime session + ops (TF backend).
- Feature gating: tests require TF backend custom ops.
- Integration points: file ops wrapper + hook system.

**Implementation Steps (Detailed)**
1. Port `test_basic` using Rust TF runtime or native file writer.
2. Port `test_hook` with a session hook that triggers close at end.
3. Ensure file output matches exact concatenation ordering.

**Tests (Detailed)**
- Python tests: `WritableFileTest.test_basic`, `.test_hook`.
- Rust tests:
  - `writable_file_basic`
  - `writable_file_close_hook`
- Cross-language parity test:
  - Compare file bytes for identical append sequences.

**Gaps / Notes**
- Tests rely on `TEST_TMPDIR` env var; Rust tests should mirror temp dir handling.

### `monolith/native_training/fused_embedding_to_layout_test.py`
<a id="monolith-native-training-fused-embedding-to-layout-test-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 1333
- Purpose/role: Validates `distribution_ops.fused_embedding_to_layout` output and gradients across multiple op versions, sharding modes, and example/example_batch variants.
- Key symbols/classes/functions: `infer_shape`, `pooling`, `FusedEmbeddingToLayoutTest`, `FusedEmbeddingToLayoutFitPreTest`.
- External dependencies: TensorFlow, `distribution_ops`, `parse_examples`, `sharding_sparse_fids`, Example/ExampleBatch protos.
- Side effects: Disables TF v2 behavior, sets RNG seed, logs debug output.

**Required Behavior (Detailed)**
- Helpers:
  - `infer_shape(out_conf, out_type, max_sequence_length=0)`:
    - Populates `out_conf.shape` for `OutType.NONE/CONCAT/STACK/ADDN`.
    - Uses `[-1, ...]` batch dimension, optionally `[max_sequence_length]`.
  - `get_key(ln, sc)` returns `"{layout}_{feature_name}_{start}_{end}"`.
  - `pooling(pooling_type, in_data, max_length)`:
    - `SUM`: elementwise sum.
    - `MEAN`: elementwise mean.
    - Else: returns padded/truncated sequence matrix of shape `(max_length, dim)`.
- `FusedEmbeddingToLayoutTest`:
  - `get_feature_cfg` builds per-feature/table metadata and indices (sorted by table/feature names).
  - `test_fused_embedding_to_layout(...)`:
    - Builds `FeatureConfigs` with multiple tables, slice configs, pooling types, and output configs (`bias`, `vec`, `ffm1`, `ffm2`, `firstN`).
    - Generates random fids per slot and expected pooled outputs.
    - Builds offsets (`fid_offset_list`, `feature_offset_list`, `nfl_offset_list`) and embedding lists per PS/table.
    - If `shard_op_version` provided, uses `parse_examples` + `sharding_sparse_fids` to generate offsets.
    - Calls `distribution_ops.fused_embedding_to_layout` with `version` and `parallel_flag`, optionally GPU.
    - Asserts output tensors match numpy-truth via `np.allclose`.
  - Variants:
    - `test_fused_embedding_to_layout_use_shard_op` (version 2).
    - `test_fused_embedding_to_layout_use_shard_op3(_gpu)` (version 3, GPU optional).
    - `test_fused_embedding_to_layout_use_shard_op4(_gpu)` (version 4, GPU optional).
    - `test_fused_embedding_to_layout_parallel` (parallel_flag=0).
  - `test_fused_embedding_to_layout_grad(...)`:
    - Constructs inputs similarly, computes `tf.gradients` of layouts w.r.t embeddings.
    - Builds “truth” counts for each fid (SUM/MEAN/FIRSTN handling).
    - For version 4, splits gradients to per-table/per-ps ordering.
    - Asserts grads are uniform per fid and equal to expected count.
    - Wrapper tests mirror the output tests for shard op versions and GPU.
- `FusedEmbeddingToLayoutFitPreTest`:
  - `test_fused_embedding_to_layout`:
    - Uses `ExampleBatch` with shared/individual features; sets `SHARD_BIT` for shared.
    - Builds offsets, embeddings, and expected outputs for `bias/vec/ffm1/ffm2`.
    - Calls `fused_embedding_to_layout` version 1 (variant_type `example_batch`) and compares outputs.
  - `test_fused_embedding_to_layout_grad`:
    - Smaller gradient test (version 1) on `Example` input; verifies gradient counts.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests/fused_embedding_to_layout.rs` (new).
- Rust public API surface: `distribution_ops::fused_embedding_to_layout` equivalent + gradient support.
- Data model mapping:
  - Example/ExampleBatch proto parsing + fid offsets.
  - Output layout configs (`FeatureConfigs`, `OutConfig`, `SliceConfig`).
- Feature gating:
  - GPU tests under a `cuda`/GPU feature.
  - Sharding op versions (2/3/4) require corresponding Rust implementations.
- Integration points:
  - `distribution_ops`, `parser_utils`, and fused layout feature pipeline.

**Implementation Steps (Detailed)**
1. Implement fused layout op in Rust with version semantics (1–4) and `parallel_flag`.
2. Port sharding_sparse_fids logic for versions 2/3/4 so test inputs can be generated.
3. Implement pooling semantics and expected-shape inference for test verification.
4. Add gradient checks (per-fid uniform gradient, equals usage counts).
5. Provide GPU test path for versions >=3.

**Tests (Detailed)**
- Python tests: this file (multiple output + gradient tests).
- Rust tests:
  - `fused_embedding_to_layout_v1_example_batch`
  - `fused_embedding_to_layout_v2_v3_v4` (CPU/GPU)
  - `fused_embedding_to_layout_grad` variants
- Cross-language parity test:
  - Generate identical inputs in Python and Rust and compare layout outputs + grads.

**Gaps / Notes**
- For `op_version >= 3` without `shard_op_version`, the test raises `TypeError('Not imple')`.
- GPU tests require `test_util.use_gpu()` and only run for versions 3/4.
- Uses global RNG seeds (`np.random.seed(2)` and `random.randint`), so determinism depends on Python RNG behavior.

### `monolith/agent_service/__init__.py`
<a id="monolith-agent-service-init-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 0
- Purpose/role: Empty package initializer for `monolith.agent_service`.
- Key symbols/classes/functions: none
- External dependencies: none
- Side effects: none

**Required Behavior (Detailed)**
- Module is intentionally empty; importing it must have no side effects.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/src/lib.rs` (module boundary only).
- Rust public API surface: none required.
- Data model mapping: none.

**Implementation Steps (Detailed)**
1. Ensure the Rust crate module layout mirrors the Python package (no runtime behavior needed).

**Tests (Detailed)**
- Python tests: none
- Rust tests: none (module boundary only)

**Gaps / Notes**
- None; this file is a no-op.

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

### `monolith/agent_service/agent.py`
<a id="monolith-agent-service-agent-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 96
- Purpose/role: Main entrypoint for running agent processes; selects AgentV1/AgentV3 and manages dense multi-process mode.
- Key symbols/classes/functions: `run_agent`, `main`
- External dependencies: `absl`, `multiprocessing`, `subprocess`, `signal`, `ModelManager`, `AgentV1`, `AgentV3`
- Side effects: env var mutations, process spawning, logging, OS signals.

**Required Behavior (Detailed)**
- `run_agent(conf_path, tfs_log, use_mps, replica_id, dense_service_index)`:
  - Mutates `REPLICA_ID` and `DENSE_SERVICE_IDX` when `use_mps` is true.
  - If `use_mps`: sets env vars and suffixes `tfs_log` with `.mps{dense_service_index}`.
  - Loads `AgentConfig` from file; uses `conf_path = dirname(agent_config_path)`.
  - `agent_version` dispatch:
    - `1` -> `AgentV1(config, conf_path, tfs_log)`.
    - `2` -> raises `Exception('agent_version v2 is not support')`.
    - `3` -> `AgentV3(config, conf_path, tfs_log)`.
    - else -> raises `Exception("agent_version error ...")`.
  - Starts `ModelManager(rough_sort_model_name, rough_sort_model_p2p_path, rough_sort_model_local_path, True)`.
  - If `model_manager.start()` returns False: log error, `os.kill(os.getpid(), SIGKILL)`.
  - Calls `agent.start()` then `agent.wait_for_termination()`.
- `main()`:
  - Calls `env_utils.setup_hdfs_env()` in try/except; logs failure.
  - Logs full env via `logging.info(f'environ is : {os.environ!r}')`.
  - If `FLAGS.conf` missing: prints `FLAGS.get_help()` and returns.
  - Loads `AgentConfig.from_file(FLAGS.conf)`.
  - If `deploy_type == DENSE` and `dense_service_num > 1`:
    - Spawns `dense_service_num` processes running `run_agent`.
    - `cur_rid = config.replica_id * dense_service_num + i`.
    - Joins all processes.
  - Else: calls `run_agent(FLAGS.conf, FLAGS.tfs_log, False, config.replica_id, 0)`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-cli` or `monolith-rs/crates/monolith-serving/src/bin/*` (new binary).
- Rust public API surface: CLI command mirroring flags `--conf` and `--tfs_log`.
- Integration points: use Rust `AgentConfig` loader, start AgentV1/V3 ported classes.

**Implementation Steps (Detailed)**
1. Add Rust CLI command equivalent to Python `agent.py` entrypoint.
2. Implement env var behavior for `REPLICA_ID` and `DENSE_SERVICE_IDX`.
3. Add multiprocessing behavior for dense service count.
4. Port `ModelManager` startup and failure semantics.
5. Wire into AgentV1/AgentV3 implementations.

**Tests (Detailed)**
- Python tests: none specific.
- Rust tests: add integration test to verify process launch plan and env var behavior.

**Gaps / Notes**
- Rust currently lacks equivalent entrypoint for agent daemonization.

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

### `monolith/agent_service/agent_base.py`
<a id="monolith-agent-service-agent-base-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 86
- Purpose/role: Agent base class + helpers for launching TFS/proxy binaries and log redirection.
- Key symbols/classes/functions: `get_cmd_path`, `get_cmd_and_port`, `ServingLog`, `AgentBase`
- External dependencies: `os`, `abc`, `AgentConfig`, `TFS_HOME`, `TFSServerType`
- Side effects: resolves binary paths, builds shell commands, changes CWD in context manager.

**Required Behavior (Detailed)**
- Constants:
  - `TFS_BINARY = {TFS_HOME}/bin/tensorflow_model_server`.
  - `PROXY_BINARY = {TFS_HOME}/bin/server`.
- `get_cmd_path()` returns `os.path.abspath(__file__)`.
- `get_cmd_and_port(config, conf_path=None, server_type=None, config_file=None, tfs_binary=TFS_BINARY, proxy_binary=PROXY_BINARY)`:
  - For `server_type` in `{PS, ENTRY, DENSE}`: delegates to `config.get_cmd_and_port(tfs_binary, server_type=..., config_file=...)`.
  - Else (proxy): uses `{conf_path}/proxy.conf` if present.
    - Command string: `{proxy_binary} --port={config.proxy_port} --grpc_target=localhost:{config.tfs_entry_port} [--conf_file=proxy.conf] &`.
    - Returns `(cmd, config.proxy_port)`.
- `ServingLog` context manager:
  - On enter: builds `log_filename = dirname(tfs_log)/{log_prefix}_{basename(tfs_log)}`.
  - Saves current cwd, `chdir` to `{TFS_HOME}/bin`, returns `open(log_filename, 'a')`.
  - On exit: `chdir` back to previous cwd (does not close file handle itself).
- `AgentBase`:
  - Stores `config` and defines abstract `start()` and `wait_for_termination()`.

**Rust Mapping (Detailed)**
- Target crate/module: new module in `monolith-rs/crates/monolith-serving/src/agent_base.rs` (or similar).
- Rust public API surface: `AgentBase` trait, `get_cmd_and_port`, `ServingLog` equivalent.
- Data model mapping: `AgentConfig` in Rust config module.

**Implementation Steps (Detailed)**
1. Port command construction rules exactly (including proxy.conf behavior).
2. Add Rust CWD + log file handling in a scoped guard.
3. Define a trait for Agent lifecycle parity (`start`, `wait_for_termination`).
4. Ensure paths match Python defaults (`TFS_HOME`, binary names).

**Tests (Detailed)**
- Python tests: none specific.
- Rust tests: add unit tests for command generation + log filename behavior.

**Gaps / Notes**
- Rust currently lacks explicit AgentBase and ServingLog helpers.

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

### `monolith/agent_service/agent_client.py`
<a id="monolith-agent-service-agent-client-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 216
- Purpose/role: CLI for AgentService + ZooKeeper operations (heartbeat, replica lookup, publish/load/unload, resource and portal inspection).
- Key symbols/classes/functions: `main`
- External dependencies: `grpc`, `kazoo`, `monolith.agent_service.*`, `monolith.native_training.env_utils`
- Side effects: gRPC calls, ZooKeeper reads/writes/deletes, prints to stdout, reads env vars.

**Required Behavior (Detailed)**
- Startup:
  - `env_utils.setup_hdfs_env()` is called.
  - `AgentConfig.from_file(FLAGS.conf)` is loaded; if `FLAGS.port != 0` overrides `agent_port`.
  - Host is `MY_HOST_IP` env var or `socket.gethostbyname(gethostname())`.
  - gRPC channel to `{host}:{agent_port}`; `AgentServiceStub` created.
  - `model_name` resolved as `agent_conf.base_name` or `FLAGS.model_name`.
- `FLAGS.server_type` mapping: `ps` -> `ServerType.PS`, `dense` -> `ServerType.DENSE`, else `ServerType.ENTRY`.
- `FLAGS.cmd_type` dispatch:
  - `hb`: send `HeartBeatRequest(server_type=...)`; print each key with number of addrs and list.
  - `gr`: assert `model_name`; send `GetReplicasRequest(server_type, task, model_name)`; print reply address list.
  - `addr` or (`get` + `args=addr`):
    - Connect `MonolithKazooClient` with `agent_conf.zk_servers`.
    - Traverse `/{bzid}/service/{model_name}`; support dc-aware layout (`idc:cluster/server_type:task`) and non-dc (`server_type:task`).
    - Uses regex `TASK = r'^(\\w+):(\\d+)$'` to detect `server_type:task` nodes; otherwise treats as `idc:cluster` prefix.
    - For each replica node, read `ReplicaMeta`, print path + `archon_address`, `address`, and `ModelState.Name`.
    - Sort output; handle `NoNodeError` by printing "{model_name} has not load !" and returning.
  - `get` + `args=info`: print `cal_model_info_v2(model_dir, ckpt)`.
  - `get` + `args in {res,pub,portal,lock,elect}`:
    - Select ZK path prefix: `/bzid/resource|publish|portal|lock|election`.
    - If path missing, print `no {args} found !`.
    - Otherwise list children, read data for each, print sorted keys with values.
  - `load`: assert `model_name`; create `ModelMeta(model_name, model_dir, ckpt, num_shard)`; write to `/bzid/portal/{model_name}` (create or set).
  - `unload`: delete `/bzid/portal/{model_name}`; ignore errors.
  - `clean`: delete all nodes under portal/publish/service/resource based on `FLAGS.args`.
- All ZK clients are started and stopped per operation.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-cli/src/bin/agent_client.rs` (or CLI subcommand).
- Rust public API surface: CLI entrypoint + small helper functions for each `cmd_type`.
- Data model mapping: use Rust protobuf types for `AgentService` and `ReplicaMeta`/`ModelMeta` equivalents.
- Feature gating: `tf-runtime` or `zk` features for ZK access; gRPC feature for AgentService.
- Integration points: reuse Rust ZK client + gRPC client modules.

**Implementation Steps (Detailed)**
1. Recreate CLI flags: `port`, `args`, `server_type`, `task`, `model_dir`, `ckpt`, `num_shard`, plus shared `FLAGS` from `client.py`.
2. Implement gRPC client calls (`HeartBeat`, `GetReplicas`) with identical formatting to stdout.
3. Implement ZK traversal for `addr` respecting dc-aware and non-dc-aware layouts.
4. Implement portal/publish/resource/lock/election reads with exact error messages.
5. Implement `load/unload/clean` ZK mutations with same node paths.
6. Mirror env var usage for host selection and config overrides.

**Tests (Detailed)**
- Python tests: none specific
- Rust tests: add CLI integration tests with fake ZK + fake gRPC stub
- Cross-language parity test: run Python CLI and Rust CLI against same fake ZK/gRPC environment and compare output.

**Gaps / Notes**
- This CLI hard-depends on ZK and gRPC; Rust implementation needs compatible test doubles.

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

### `monolith/agent_service/agent_controller.py`
<a id="monolith-agent-service-agent-controller-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 145
- Purpose/role: CLI to declare SavedModel configs in ZK, and publish/unpublish layouts.
- Key symbols/classes/functions: `find_model_name`, `declare_saved_model`, `map_model_to_layout`, `bzid_info`, `main`
- External dependencies: `tensorflow`, `saved_model_pb2`, `compat`, `monolith.agent_service.backends.*`, ZK backend.
- Side effects: reads `saved_model.pb`, writes/updates ZK nodes, prints JSON.

**Required Behavior (Detailed)**
- `find_model_name(exported_models_path)`:
  - Uses `entry/` subdirectory and picks `sorted(tf.io.gfile.listdir(entry_path))[0]` as timestamp.
  - Reads `saved_model.pb`, parses `SavedModel`, scans graph nodes with `op == 'TfServingRemotePredict'`.
  - Returns first `model_name` attribute (decoded, before `:`), or `None` if not present.
- `declare_saved_model(bd, export_base, model_name=None, overwrite=False, arch='entry_ps')`:
  - Asserts `arch == 'entry_ps'`.
  - Determines `model_name` via `find_model_name` if not supplied.
  - Logs mismatch if supplied name differs from export name; asserts non-None.
  - Asserts no existing saved_models unless `overwrite`.
  - For each subgraph in `export_base`:
    - Build `SavedModelDeployConfig(model_base_path=..., version_policy='latest' for entry, else 'latest_once')`.
    - Call `bd.decl_saved_model(SavedModel(model_name, sub_graph), deploy_config)`.
  - Logs success; returns `model_name`.
- `map_model_to_layout(bd, model_pattern, layout_path, action)`:
  - Parses `model_pattern` as `model_name:sub_graph_pattern`.
  - `fnmatch.filter` on available subgraphs.
  - `pub` => `bd.add_to_layout`, `unpub` => `bd.remove_from_layout`.
- `bzid_info(bd)`: `print(json.dumps(bd.bzid_info(), indent=2))`.
- `main`:
  - Validates `FLAGS.cmd` in `decl|pub|unpub|bzid_info`.
  - Creates `ZKBackend`, `start()` then executes command; `stop()` in `finally`.
  - For `pub/unpub`, layout path is `/{bzid}/layouts/{layout}`.
  - Uses `env_utils.setup_hdfs_env()` in `__main__` guard; sets logging.
  - Flags: `zk_servers`, `bzid`, `export_base`, `overwrite`, `model_name`, `layout`, `arch`, `cmd`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-cli/src/bin/agent_controller.rs`.
- Rust public API surface: CLI entrypoint + helpers for `declare_saved_model` and layout mapping.
- Data model mapping: `SavedModel`, `SavedModelDeployConfig`, `ZKBackend` analogs in Rust.
- Feature gating: `tf-runtime`/`saved-model` parsing feature for `find_model_name` (optional if using tf runtime).
- Integration points: ZK backend + layout manager.

**Implementation Steps (Detailed)**
1. Port CLI flags (`cmd`, `bzid`, `zk_servers`, `export_base`, `model_name`, `layout`, `arch`, `overwrite`).
2. Implement saved_model.pb parsing in Rust (protobuf decode + graph scan for `TfServingRemotePredict`).
3. Recreate `declare_saved_model` and `map_model_to_layout` logic exactly.
4. Ensure `latest` vs `latest_once` policy mapping is preserved.
5. Port `bzid_info` output formatting to JSON.

**Tests (Detailed)**
- Python tests: `monolith/agent_service/agent_controller_test.py`
- Rust tests: add equivalent tests using fake ZK and test saved_model fixture.
- Cross-language parity test: compare `bzid_info`/layout changes after publish/unpublish.

**Gaps / Notes**
- Rust needs SavedModel graph parsing for remote predict op discovery.

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

### `monolith/agent_service/agent_controller_test.py`
<a id="monolith-agent-service-agent-controller-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 95
- Purpose/role: Tests ZKBackend + agent_controller declare/publish/unpublish flows using FakeKazooClient.
- Key symbols/classes/functions: `AgentControllerTest.test_decl_saved_models`, `.test_pub`
- External dependencies: `FakeKazooClient`, `ZKBackend`, test saved_model fixtures.
- Side effects: creates ZK nodes in fake client.

**Required Behavior (Detailed)**
- `setUpClass`:
  - `bzid='gip'`.
  - `bd = ZKBackend(bzid, zk_servers='127.0.0.1:9999')`.
  - Replace `bd._zk` with `FakeKazooClient()`.
  - Call `bd.start()`.
- `test_decl_saved_models`:
  - Uses exported saved_model dir under `TEST_SRCDIR/TEST_WORKSPACE/monolith/native_training/model_export/testdata/saved_model`.
  - Calls `declare_saved_model(..., model_name='test_ffm_model', overwrite=True)`.
  - Verify `bd.list_saved_models('test_ffm_model')` matches `{ps_0..ps_4, entry}`.
- `test_pub`:
  - Declare saved model again.
  - `map_model_to_layout(..., "test_ffm_model:entry", "/gip/layouts/test_layout1", "pub")` -> `layout_info['test_layout1'] == ['test_ffm_model:entry']`.
  - `map_model_to_layout(..., "test_ffm_model:ps_*", "pub")` -> adds `ps_0..ps_4` (ordered list).
  - `map_model_to_layout(..., "test_ffm_model:ps_*", "unpub")` -> back to entry only.
  - `map_model_to_layout(..., "test_ffm_model:entry", "unpub")` -> empty list.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-cli/tests/agent_controller.rs` (or serving tests if backend lives there).
- Rust public API surface: ZK backend + controller helpers.
- Data model mapping: `SavedModel` and `SavedModelDeployConfig` equality comparisons.

**Implementation Steps (Detailed)**
1. Port fake ZK backend for tests.
2. Port `declare_saved_model` and `map_model_to_layout` in Rust.
3. Add fixtures for saved_model testdata or stub saved_model parsing.

**Tests (Detailed)**
- Python tests: this file
- Rust tests: exact parity assertions for saved_models and layout contents.

**Gaps / Notes**
- Requires test saved_model fixture to exist and be accessible in Rust tests.

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

### `monolith/agent_service/agent_service.py`
<a id="monolith-agent-service-agent-service-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 155
- Purpose/role: gRPC AgentService implementation + server wrapper for replica discovery and heartbeat.
- Key symbols/classes/functions: `AgentDataProvider`, `AgentServiceImpl`, `AgentService`
- External dependencies: `grpc`, `concurrent.futures`, `functools.singledispatchmethod`, `monolith.agent_service.*`
- Side effects: gRPC server binds to ports; reads env/config; logs heartbeat.

**Required Behavior (Detailed)**
- `AgentServiceImpl.__init__` is **overloaded** via `@singledispatchmethod`:
  - `ReplicaWatcher` + optional `AgentConfig`
  - `ZKMirror` + `AgentConfig`
  - `AgentDataProvider` + `AgentConfig`
- `GetReplicas` behavior:
  - If `conf is None` or `agent_version == 1`, fetch replicas via `ReplicaWatcher` with `idc/cluster`.
  - If `agent_version == 2`, fetch via ZKMirror (`get_task_replicas`).
  - Else: `NotImplementedError` for v3.
- `HeartBeat` behavior:
  - `agent_version == 1`: call `ReplicaWatcher.get_all_replicas`, optionally strip DC prefix when `dc_aware`.
  - `agent_version == 2`: call `ZKMirror.get_all_replicas`.
  - `agent_version == 3`: use `AgentDataProvider` callback map to fill addresses.
- `GetResource` behavior:
  - `agent_version == 1`: return empty `GetResourceResponse`.
  - Else: fill address with `get_local_ip()` + `agent_port`, plus memory via `cal_available_memory_v2()`.
- `AgentService` wrapper:
  - Constructs grpc server with `ThreadPoolExecutor`, registers `AgentServiceImpl`, binds to port.
  - `AgentDataProvider` wraps `addrs_fn` callback returning `{saved_model_name: [addr...]}`.
  - `GetReplicas` v1 uses `_watcher._conf.idc/cluster` and `get_replicas(server_type, task, idc, cluster)`; v2 maps `ReplicaMeta.address`.
  - `HeartBeat` v1 uses `dc_aware` to strip path prefix (`key.split('/')[-1]`) and populates `AddressList`.
  - `HeartBeat` v3 uses `_data_provider._addrs_fn()` if non-empty.
  - `GetResource` v2+ fills `shard_id`, `replica_id`, `memory`, `cpu/network/work_load=-1.0`.
  - `AgentService` binds `[::]:{port or 0}` for watcher path; uses `conf.agent_port` for zk/data_provider.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/src/grpc_agent.rs` (gRPC), `monolith-rs/crates/monolith-serving/src/server.rs` (server lifecycle), plus new module for AgentConfig integration.
- Rust public API surface: `AgentServiceRealImpl` + `AgentGrpcServer` must be extended to mirror multi-backend behaviors and `AgentServiceImpl` logic.
- Data model mapping: use `monolith_proto::monolith::serving::agent_service::*` for request/response.
- Feature gating: always available with `grpc` feature.
- Integration points: `monolith-serving::server::Server` should optionally host AgentService parity endpoints.

**Implementation Steps (Detailed)**
1. Add AgentConfig wiring to Rust AgentService (support v1/v2/v3 equivalent selection).
2. Port `ReplicaWatcher` + `ZKMirror` behaviors or stub with explicit guards and tracking notes.
3. Implement `dc_aware` key rewriting in HeartBeat response.
4. Implement `GetResource` response with local IP and memory (port parity).
5. Add AgentDataProvider path for v3-style maps.
6. Mirror Python error semantics (NotImplemented) when agent_version == 3 in GetReplicas.
7. Add tests mirroring `agent_service_test.py`.

**Tests (Detailed)**
- Python tests: `monolith/agent_service/agent_service_test.py`
- Rust tests: add to `monolith-rs/crates/monolith-serving/tests/*` covering GetReplicas/HeartBeat/GetResource.
- Cross-language parity test: run Python client against Rust server for v1/v2 paths.

**Gaps / Notes**
- Rust currently provides minimal AgentService; needs full v1/v2/v3 parity.

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

### `monolith/agent_service/agent_service_test.py`
<a id="monolith-agent-service-agent-service-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 107
- Purpose/role: Tests gRPC AgentService heartbeats and replica lookup using FakeKazooClient + ReplicaWatcher.
- Key symbols/classes/functions: `AgentServiceTest.setUpClass`, `test_heart_beat`, `test_get_replicas`
- External dependencies: Fake ZK, `ReplicaWatcher`, `AgentService`, gRPC client stubs.
- Side effects: starts gRPC server, creates ephemeral ZK nodes.

**Required Behavior (Detailed)**
- `setUpClass`:
  - Set `TCE_INTERNAL_IDC='lf'`, `TCE_LOGICAL_CLUSTER='default'`.
  - Start `FakeKazooClient`.
  - Build `AgentConfig(bzid='test_model', base_name=MODEL_NAME, deploy_type='ps', base_path=BASE_PATH, num_ps=20, dc_aware=True)`.
  - Create `ReplicaWatcher(zk, agent_conf)`.
  - Call `register(zk)` to seed replica nodes; then `watcher.watch_data()`.
  - Start `AgentService(watcher, port=agent_conf.agent_port)`.
  - Create `SvrClient(agent_conf)`.
- `register(zk)`:
  - `path_prefix = agent_conf.path_prefix`.
  - For each PS task `0..num_ps-1` and replica `0..1`:
    - `ReplicaMeta(address='192.168.1.{idx}:{find_free_port()}', stat=ModelState.AVAILABLE)`.
    - Create ephemeral node at `{path_prefix}/ps:{task_id}/{replica_id}` with `makepath=True`.
  - For entry replicas `0..1`: create `{path_prefix}/entry:0/{replica_id}` similarly.
  - Uses `zk.retry(zk.create, ...)` and falls back to `zk.set` on `NodeExistsError`.
- `test_heart_beat`:
  - `client.heart_beat(server_type=ServerType.PS)`; asserts `len(resp.addresses) == 20`.
- `test_get_replicas`:
  - `client.get_replicas(server_type=ServerType.PS, task=NUM_PS_REPLICAS - 1)` (task=1).
  - Asserts `len(resp.address_list.address) == NUM_PS_REPLICAS` (2).

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/tests/agent_service.rs`.
- Rust public API surface: gRPC AgentService + ReplicaWatcher equivalent.
- Data model mapping: `ReplicaMeta`, `ModelState`, and gRPC proto types.

**Implementation Steps (Detailed)**
1. Implement fake ZK and replica registration helpers in Rust tests.
2. Start Rust AgentService server on a free port.
3. Call HeartBeat and GetReplicas via gRPC client; assert counts.

**Tests (Detailed)**
- Python tests: this file
- Rust tests: parity test with fake ZK + gRPC client.

**Gaps / Notes**
- Rust needs fake ZK client and ReplicaWatcher equivalent to reproduce dc-aware paths.

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

### `monolith/agent_service/agent_v1.py`
<a id="monolith-agent-service-agent-v1-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 390
- Purpose/role: v1 agent orchestrator that launches TensorFlow Serving processes (ps/entry/dense) and hosts AgentService.
- Key symbols/classes/functions: `ProcessType`, `ProcessNode`, `ProcessMgr`, `AgentV1`
- External dependencies: `subprocess`, `signal`, `threading`, `ReplicaManager`, `AgentService`, `AgentBase`
- Side effects: starts and kills subprocesses; registers signal handlers; logs to files.

**Required Behavior (Detailed)**
- `ProcessType` enum: `PS=1`, `ENTRY=2`, `PROXY=3`, `UNKONWN=4`, `DENSE=5`.
- `ProcessNode.__init__`:
  - Stores config, replica_mgr, proc_type, tfs_log, env, etc; `_shell=False`, `_stderr=STDOUT`.
  - For `PS/ENTRY/DENSE`: uses `get_cmd_and_port` with server_type and `tfs_binary`.
  - For ENTRY: `_env = os.environ.copy()` and sets `PORT2` to `agent_port`.
  - For other types: `get_cmd_and_port` without server_type (proxy command).
  - Tracks `_sub_procs` map and `_popen` handle.
- `ProcessNode.run()`:
  - If ENTRY: waits for PS replicas to start:
    - Sleeps `update_model_status_interval * 2` and loops while `!replica_mgr.is_ps_set_started()` up to `max_waiting_sec=3600`.
    - If `dense_alone`: waits for `is_dense_set_started()` similarly.
    - On timeout logs error and returns False.
  - Launches subprocess via `ServingLog(proc_type.name.lower(), tfs_log)`:
    - `Popen(self._cmd.split(), shell=False, stderr=STDOUT, stdout=log)` unless `MLP_POD_NAME` env is set (then stdout=None).
  - Calls `wait_for_started()`; on failure logs and returns False.
  - Starts each sub-proc via `proc.run()`, aborting on failure.
- `ProcessNode.wait_for_started()`:
  - If `_port==0` returns True.
  - Polls `check_port_open(_port)` every 10s up to 3600s.
- `ProcessNode.kill()`:
  - Kills sub-procs first; for each, retries up to 3 times with `kill()` and 1s sleep.
  - Kills self `_popen` with same retry loop.
- `ProcessNode.poll/returncode`:
  - If `_is_failover` True: return None; else delegates to `_popen.poll()/returncode`.
- `ProcessNode.failover()`:
  - Sets `_is_failover`; if not `is_tce_main` and proc_type in {PS,DENSE} -> `run()`; else `kill()`; resets flag.
- `ProcessMgr`:
  - Class vars `_is_killed=False`, `_lock=RLock`.
  - Registers SIGTERM/SIGINT handler that kills all and SIGKILLs self.
  - `_poll` thread flattens process tree and checks `returncode`; if any exited, kills all (no failover).
  - `start()` runs each subproc; starts poll thread; on failure kills all.
  - `kill_all(include_self=True)` kills all subprocs and optionally `os.kill(self, SIGKILL)`.
- `AgentV1.__init__`:
  - Creates `MonolithKazooClient`, `ReplicaManager`, and `AgentService(replica_mgr.watcher, port=agent_port)`.
  - Builds `ProcessMgr` and adds `ProcessNode`s based on `deploy_type`:
    - MIXED: ps + (dense if `dense_alone`) + entry (entry marked `is_tce_main=True`).
    - ENTRY: single ENTRY node (named `proxy_proc` in code).
    - PS: single PS node (`is_tce_main=True`).
    - Else: single DENSE node.
- `AgentV1.start()`:
  - Starts ZK, ReplicaManager, AgentService, then ProcessMgr (with logs for each).
- `AgentV1.wait_for_termination()` delegates to AgentService.
- `AgentV1.stop()`:
  - `process_mgr.kill_all(include_self=False)`, stop AgentService, ReplicaManager, and ZK.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/src/agent_v1.rs` + process supervisor module.
- Rust public API surface: `AgentV1` struct implementing `AgentBase` trait.
- Data model mapping: `AgentConfig`, `DeployType`, `TFSServerType` in Rust config module.
- Integration points: gRPC AgentService + ReplicaManager + process management.

**Implementation Steps (Detailed)**
1. Implement process supervisor with child processes and log redirection.
2. Port ENTRY wait-for-PS/dense logic with same timeouts and intervals.
3. Port signal handling to trigger shutdown of all children.
4. Port startup order and error handling semantics.
5. Ensure `PORT2` env var is injected for ENTRY.

**Tests (Detailed)**
- Python tests: none directly
- Rust tests: integration test that spawns dummy processes and validates kill-on-exit behavior.
- Cross-language parity test: simulate ReplicaManager readiness and verify ENTRY wait logic.

**Gaps / Notes**
- Requires replacement for `check_port_open` and process supervision in Rust.

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

### `monolith/agent_service/agent_v3.py`
<a id="monolith-agent-service-agent-v3-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 210
- Purpose/role: v3 unified agent that manages TFS model config via layout updates, registers service info, and serves address maps via AgentService.
- Key symbols/classes/functions: `gen_empty_model_config_file`, `AgentV3`.
- External dependencies: `TFSWrapper`, `ZKBackend`, `Container`, `ContainerServiceInfo`, protobuf text/json, `model_server_config_pb2`.
- Side effects: writes model_config file, starts/stops TFS process, registers ZK service info, background threads, SIGKILL on exit.

**Required Behavior (Detailed)**
- `gen_empty_model_config_file()`:
  - Uses `tempfile.mktemp()` and writes `model_config_list {}`.
- `AgentV3.__init__`:
  - Asserts `deploy_type == UNIFIED` and `agent_version == 3`.
  - Installs SIGTERM/SIGINT handlers to set `_exit_event`.
  - Creates `_model_config_path` and initializes `TFSWrapper` with ports + config path.
  - Builds `_layout_filters` from `config.layout_filters`:
    - Replaces `${shard_id}` and `${shard_num}`.
    - Splits `match;cond`, normalizes regex via `normalize_regex`.
  - Builds `ContainerServiceInfo` with local IP and ports; `debug_info` JSON includes layout path + filters.
  - Initializes `ZKBackend` and `AgentService(AgentDataProvider(_gen_addrs_map))`.
- `_gen_addrs_map()`:
  - Reads `backend.get_service_map()` and returns `{model:sub_graph: [addr...]}`.
  - Uses `grpc` addr if `tfs_wrapper.is_grpc_remote_op`, else `archon` addr.
- `sync_available_saved_models()`:
  - Calls `tfs_wrapper.list_saved_models_status()`.
  - For `State.AVAILABLE`, converts `model_name:sub_graph` into `SavedModel` and syncs via backend.
- `layout_update_callback(saved_models)`:
  - Builds `ModelServerConfig` with `model_config_list.SetInParent()`.
  - Applies layout filters: `re.match(match, sub_graph)` + `eval(cond, None, {groupdict as int})`.
  - Adds `ModelConfig` generated by `gen_model_config(name, base_path, version_policy)`.
  - Writes protobuf text to `_model_config_path`.
- `start_bg_thread(fn, interval)`:
  - Runs `fn()` in loop until `_exit_event` set; logs exceptions; sleeps `interval`.
- `start()`:
  - Starts TFSWrapper, ZKBackend, AgentService.
  - Starts background threads:
    - `backend.report_service_info(container, service_info)` every 60s.
    - `sync_available_saved_models` every 30s.
  - Registers layout callback on `config.layout_path`.
- `stop()`:
  - Sets exit event, joins threads, stops AgentService, backend, and TFSWrapper (logs warnings on error).
- `wait_for_termination()`:
  - Polls `tfs_wrapper.poll()`; if exit, sets `_exit_event`, stops, sleeps 1s, and `SIGKILL`s self.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/src/agent_v3.rs`.
- Rust public API surface: `AgentV3` implementing `AgentBase` with background tasks.
- Data model mapping: `Container`, `SavedModel`, `SavedModelDeployConfig`, `ContainerServiceInfo`, layout filters, service map.
- Feature gating: `tf-runtime` (TFSWrapper), `zk` (ZKBackend), gRPC for AgentService.

**Implementation Steps (Detailed)**
1. Port layout filter parsing and evaluation (regex + safe expression engine).
2. Implement model_config protobuf text writer (matching TF Serving config).
3. Port service info reporting + available model syncing to ZK.
4. Implement address map provider for AgentService v3.
5. Preserve shutdown semantics (exit event, thread joins, SIGKILL).

**Tests (Detailed)**
- Python tests: `monolith/agent_service/agent_v3_test.py`
- Rust tests: FakeTFSWrapper + FakeZK to verify layout updates + service map.
- Cross-language parity test: generate identical config updates and compare model_config text.

**Gaps / Notes**
- Python uses `tempfile.mktemp()` (unsafe); Rust should use secure temp files but preserve behavior.
- Layout filter `cond` uses `eval` on regex groupdict (needs a safe Rust equivalent).

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

### `monolith/agent_service/agent_v3_test.py`
<a id="monolith-agent-service-agent-v3-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 114
- Purpose/role: Tests AgentV3 layout/publish flow with FakeTFSWrapper and FakeKazooClient.
- Key symbols/classes/functions: `AgentV3Test.test_service_info`, `.test_publish_models`
- External dependencies: Fake TFS wrapper, fake ZK backend.
- Side effects: starts AgentV3 (with fake components), writes model_config file.

**Required Behavior (Detailed)**
- Class setup:
  - `bzid='gip'`, set `MY_HOST_IP=127.0.0.1`.
  - Build `AgentConfig(bzid='gip', deploy_type='unified', agent_version=3, layout_pattern='/gip/layout', zk_servers='127.0.0.1:8888')`.
  - `base_path = os.environ['TEST_TMPDIR']`.
  - Construct `AgentV3` with:
    - `conf_path = os.path.join(base_path, '/monolith_serving/conf')` (note: absolute suffix).
    - `tfs_log = os.path.join('monolith_serving/logs/log.log')` (relative path).
  - Replace `_tfs_wrapper` with `FakeTFSWrapper(agent._model_config_path)`.
  - Replace `_backend._zk` with `FakeKazooClient`.
  - Call `agent.start()`.
  - For sub_graph in `['entry','ps_0','ps_1','ps_2']`:
    - `config={'model_base_path': TEST_TMPDIR/test_ffm_model/exported_models/{sub_graph}, 'version_policy': 'latest'}`.
    - Write JSON bytes to ZK path `/gip/saved_models/test_ffm_model/{sub_graph}` with `makepath=True`.
- `test_service_info`: backend `get_service_info(container)` equals agent's `_service_info`.
- `test_publish_models`:
  - Assert `tfs_wrapper.list_saved_models()` initially empty.
  - `zk.ensure_path('/gip/layout/test_ffm_model:entry')` and `...:ps_0`.
  - Expect `list_saved_models()` == `['test_ffm_model:entry','test_ffm_model:ps_0']` (order matters).
  - Call `agent.sync_available_saved_models()`.
  - Expect `backend.get_service_map()` equals:
    - `{'test_ffm_model': {'entry': [agent._service_info], 'ps_0': [agent._service_info]}}`.
  - Delete `/gip/layout/test_ffm_model:ps_0`.
  - Expect `list_saved_models()` == `['test_ffm_model:entry']`.
  - Call `sync_available_saved_models()` and expect service_map only has `entry`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/tests/agent_v3.rs`.
- Rust public API surface: AgentV3, FakeTFSWrapper, Fake ZK backend.

**Implementation Steps (Detailed)**
1. Implement FakeTFSWrapper in Rust with model_config file parsing.
2. Implement Fake ZK backend with nodes under saved_models/layouts.
3. Port tests for service_info equality and publish/unpublish behavior.

**Tests (Detailed)**
- Python tests: this file
- Rust tests: direct parity test using fake components.

**Gaps / Notes**
- Requires deterministic model_config parsing in Rust to match FakeTFSWrapper behavior.

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

### `monolith/agent_service/backends.py`
<a id="monolith-agent-service-backends-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 402
- Purpose/role: Backend abstractions + ZK implementation for layouts, saved models, service info, and sync targets.
- Key symbols/classes/functions: `SavedModel`, `SavedModelDeployConfig`, `Container`, `ContainerServiceInfo`, `AgentBackend`, `CtrlBackend`, `SyncBackend`, `ZKBackend`.
- External dependencies: `kazoo` (ChildrenWatch, state, errors), `dataclasses_json`, `MonolithKazooClient`.
- Side effects: ZK reads/writes, ephemeral nodes, watchers, retry/set on conflicts.

**Required Behavior (Detailed)**
- Data classes:
  - `SavedModel(model_name, sub_graph)` is frozen; `__str__` → `"model:sub_graph"`.
  - `SavedModelDeployConfig` + `ContainerServiceInfo` are `dataclass_json` with `serialize()` → UTF-8 JSON bytes; `deserialize()` uses `from_json`.
  - `Container(ctx_cluster, ctx_id)` string format `"cluster:id"`.
- Abstract backends:
  - `AgentBackend`: layout callbacks, sync_available_saved_models, report/get service info, get_service_map, start/stop.
  - `CtrlBackend`: list/declare saved models, add/remove layout, bzid_info, start/stop.
  - `SyncBackend`: subscribe_model, get_sync_targets, start/stop.
- `ZKBackend.__init__(bzid, zk_servers)`:
  - Creates `MonolithKazooClient` and registers a listener:
    - `KazooState.LOST` → sets `_is_lost` event.
    - other states → clears `_is_lost`.
  - Tracks `_available_saved_model` (set), `_service_info_map` (dict), `_children_watcher_map` (path→ChildrenWatch), `_sync_model_name`.
- `sync_available_saved_models(container, saved_models)`:
  - If `_is_lost` set: clears available set, calls `_zk.restart()`, returns.
  - Computes add/remove sets; for each add: create ephemeral binding node at
    `/{bzid}/binding/{model}/{sub_graph}:{container}` (makepath=True).
  - For each remove: delete the binding node.
  - Updates `_available_saved_model` to the new set.
- `register_layout_callback(layout_path, callback)`:
  - Ensures layout path exists and sets a ChildrenWatch via `_children_watch`.
  - `callback_wrap(children)`:
    - Parses `model_name:sub_graph` from each child.
    - Reads deploy config from `/saved_models/{model}/{sub_graph}`; if missing logs error.
    - Builds list of `(SavedModel, SavedModelDeployConfig)` and `model_names` set.
    - Resets `_service_info_map` to only those model_names (preserving existing sub-maps).
    - For each model_name, registers binding watch on `/{bzid}/binding/{model_name}` with `_bind_callback`.
    - Calls `callback(saved_models)` and returns its result.
- `_bind_callback(model_name, children)`:
  - If model_name not tracked, returns False.
  - For each child `sub_graph:ctx_cluster:ctx_id:...`, fetches `ContainerServiceInfo`.
  - Populates `_service_info_map[model_name][sub_graph]` with service infos.
- `report_service_info(container, service_info)`:
  - Creates ephemeral node at `/{bzid}/container_service/{container}` with serialized JSON bytes.
- `get_service_info(container)`:
  - Reads node and deserializes `ContainerServiceInfo` (returns None on NoNode).
- `_children_watch(path, callback)`:
  - If watcher exists and not stopped, log and skip.
  - Ensures path exists before creating `ChildrenWatch`.
- `list_saved_models(model_name)`:
  - Reads children under `/{bzid}/saved_models/{model_name}`, returns `SavedModel` list.
  - Returns empty list on `NoNodeError`.
- `decl_saved_model(saved_model, deploy_config)`:
  - Creates znode at `/{bzid}/saved_models/{model}/{sub_graph}` with JSON bytes (makepath=True).
- `add_to_layout(layout, saved_model)`:
  - Ensures path `"{layout}/{model:sub_graph}"` exists.
- `remove_from_layout(layout, saved_model)`:
  - Deletes path; ignores NoNodeError.
- `bzid_info()`:
  - Collects `model_info` (deploy configs + bindings), `container_info` (service info + saved_models), `layout_info` (sorted saved_model list).
  - Increments `sub_graphs_available` when bindings exist.
  - Returns sorted dicts for stable output.
- Sync backend:
  - `subscribe_model(model_name)`: sets `_sync_model_name` (asserts only once) and registers binding watch.
  - `get_sync_targets(sub_graph)`:
    - If `_is_lost` set, clears available and restarts ZK.
    - Returns `(f"{model}:{sub_graph}", [service_info.grpc...])`.
- ZK helpers:
  - `create_znode`: on `NodeExistsError` uses `_zk.retry(_zk.set, path=..., value=...)`; logs other exceptions.
  - `delete_znode`: logs errors on exception.
  - `get_znode`: returns bytes or None on `NoNodeError`.
- `start()`/`stop()` delegate to `_zk.start()` / `_zk.stop()`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/src/agent_backend.rs` + ZK integration (feature gated).
- Rust public API surface:
  - Structs: `SavedModel`, `SavedModelDeployConfig`, `Container`, `ContainerServiceInfo` (serde JSON).
  - Traits: `AgentBackend`, `CtrlBackend`, `SyncBackend`.
  - `ZkBackend` implementing all three.
- Data model mapping:
  - ZK nodes: `/saved_models`, `/layouts`, `/binding`, `/container_service` with JSON payloads.
  - Service map: `HashMap<Model, HashMap<SubGraph, Vec<ContainerServiceInfo>>>`.
- Integration points: `AgentV3`, `agent_controller`, `tfs_monitor`, `replica_manager`.

**Implementation Steps (Detailed)**
1. Port data classes with `serde_json` encode/decode to bytes.
2. Implement backend traits and ZK client wrapper with watch support.
3. Match ChildrenWatch semantics (invoke callback on changes).
4. Preserve ZK LOST behavior (restart + clear available set).
5. Mirror `bzid_info` output structure and sorting.

**Tests (Detailed)**
- Python tests: `monolith/agent_service/backends_test.py`
- Rust tests: mock ZK for layout watches, binding updates, and bzid_info structure.
- Cross-language parity test: compare serialized JSON payloads and service_map updates.

**Gaps / Notes**
- Requires Rust ZK client with watcher support; ensure watcher stop semantics match `ChildrenWatch`.

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

### `monolith/agent_service/backends_test.py`
<a id="monolith-agent-service-backends-test-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 134
- Purpose/role: Tests ZKBackend layout callbacks, service info, binding updates, and sync targets using a fake ZK client.
- Key symbols/classes/functions: `ZKBackendTest` methods (`test_register_service`, `test_layout_callback`, `test_sync_available_models`, `test_service_map`, `test_sync_backend`).
- External dependencies: `FakeKazooClient`, `ZKBackend`, `SavedModel`, `SavedModelDeployConfig`.
- Side effects: Creates/deletes ZK nodes in fake client.

**Required Behavior (Detailed)**
- `setUpClass`:
  - `bzid='gip'`, `container=Container("default","asdf")`.
  - `service_info` with grpc/http/archon/agent/idc.
  - Instantiate `ZKBackend`, replace `_zk` with `FakeKazooClient`.
  - Register layout callback on `"/gip/layouts/test_layout/mixed"`.
  - Call `report_service_info(container, service_info)`.
- `test_register_service`:
  - `get_service_info(container)` returns the same `service_info`.
- `test_layout_callback`:
  - Declares saved models `entry, ps_0, ps_1, ps_2` and adds to layout.
  - Expects `layout_record` to be list of `(SavedModel, SavedModelDeployConfig)` for each subgraph.
  - After removing `entry`, expects callback list with only ps_* entries.
- `test_sync_available_models`:
  - Syncs available models `entry, ps_0, ps_1`.
  - Asserts binding znodes exist at `/gip/binding/test_ffm_model/<sub_graph>:<container>`.
- `test_service_map`:
  - Syncs available models `entry, ps_0`.
  - Expects service map `{'test_ffm_model': {'ps_0': [service_info], 'entry': [service_info]}}`.
- `test_sync_backend`:
  - `subscribe_model("test_ffm_model")`, then sync available models `ps_0, ps_1, ps_2`.
  - `get_sync_targets("ps_1")` returns `("test_ffm_model:ps_1", [service_info.grpc])`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/tests/backends.rs` (new).
- Rust public API surface: `ZkBackend` + fake ZK client.
- Data model mapping: JSON serialization for deploy config + service info.
- Feature gating: tests require fake ZK or in-memory backend.

**Implementation Steps (Detailed)**
1. Implement FakeZK client in Rust with create/get/delete and ChildrenWatch semantics.
2. Port layout callback test to verify `SavedModelDeployConfig` list ordering.
3. Port binding and service_map assertions.
4. Port sync backend target selection test.

**Tests (Detailed)**
- Python tests: `ZKBackendTest` in this file.
- Rust tests:
  - `zk_backend_register_service`
  - `zk_backend_layout_callback`
  - `zk_backend_sync_available_models`
  - `zk_backend_service_map`
  - `zk_backend_sync_backend`
- Cross-language parity test:
  - Compare service_map and binding paths generated by Python vs Rust for identical inputs.

**Gaps / Notes**
- Tests rely on `TEST_TMPDIR` for `SavedModelDeployConfig` base paths.
- `test_register_service`: `get_service_info(container)` equals originally reported service info.
- `test_layout_callback`:
  - Declare saved_models and add to layout; callback receives list of `(SavedModel, DeployConfig)` in order.
  - Removing entry updates callback list.
- `test_sync_available_models`: `sync_available_saved_models` creates binding nodes under `/bzid/binding/...`.
- `test_service_map`: `get_service_map` returns service info list for entry/ps_0.
- `test_sync_backend`: `subscribe_model` + `sync_available_saved_models` produces `get_sync_targets` for a subgraph.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/tests/backends.rs`.
- Rust public API surface: ZKBackend + SavedModel + DeployConfig.

**Implementation Steps (Detailed)**
1. Port Fake ZK backend + ZKBackend logic.
2. Port tests for layout callback and binding map parity.
3. Ensure ordering of saved_models matches Python list order.

**Tests (Detailed)**
- Python tests: this file
- Rust tests: parity tests mirroring the same assertions.

**Gaps / Notes**
- Requires deterministic layout callback ordering in Rust.

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

### `monolith/agent_service/client.py`
<a id="monolith-agent-service-client-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 126
- Purpose/role: Lightweight CLI client for load/unload/status via ZKMirror (portal/publish/service inspection).
- Key symbols/classes/functions: `ServingClient`, `LoadSate`, `main`
- External dependencies: `MonolithKazooClient`, `ZKMirror`, `ModelMeta`, `ReplicaMeta`, TF Serving `ModelVersionStatus`.
- Side effects: ZK reads/writes, prints status, env setup.

**Required Behavior (Detailed)**
- Flags: `cmd_type` enum (`hb`, `gr`, `addr`, `get`, `clean`, `load`, `unload`, `meta`, `status`, `profile`), `zk_servers`, `bzid`, `model_name`, `target`, `input_type`, `input_file`.
- `LoadSate` dataclass: `portal: bool`, `publish: bool`, `service: dict` (default empty).
- `ServingClient.__init__`:
  - Creates `MonolithKazooClient` and `ZKMirror(zk, bzid)`; starts mirror with `is_client=True`.
- `load(model_name, model_dir, ckpt=None, num_shard=-1)`:
  - Creates `ModelMeta`, computes path under `portal_base_path`.
  - If path exists, raise `RuntimeError('{model_name} has exists')`.
  - Otherwise create node with serialized meta.
- `unload(model_name)`:
  - Delete portal node if exists; else log warning.
- `get_status(model_name)`:
  - `portal` True if `/bzid/portal/{model_name}` exists (via `kazoo.exists`).
  - `publish` True if any `/bzid/publish/{shard}:{replica}:{model_name}` exists.
  - `service` map `{node}:{replica} -> ReplicaMeta.stat` for all replicas under `/bzid/service/{model_name}`.
- `main`:
  - Calls `env_utils.setup_host_ip()`.
  - Requires `zk_servers` and `bzid` flags; asserts `model_name` provided.
  - `cmd_type == load`: requires `model_dir`; calls `client.load(model_name, model_dir, ckpt, num_shard)`.
  - `cmd_type == unload`: calls `client.unload(model_name)`.
  - Else: prints `client.get_status(model_name)`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-cli/src/bin/serving_client.rs`.
- Rust public API surface: CLI entrypoint + `ServingClient` struct.
- Data model mapping: `ModelMeta`, `ReplicaMeta`, `ModelState` enums.

**Implementation Steps (Detailed)**
1. Port `ServingClient` with ZKMirror client mode.
2. Implement portal node create/delete semantics and error message parity.
3. Implement status inspection across portal/publish/service paths.
4. Port CLI flags and default cmd_type handling.

**Tests (Detailed)**
- Python tests: none specific
- Rust tests: add fake ZK tests for `load/unload/get_status`.

**Gaps / Notes**
- Rust needs ZKMirror equivalent and a way to read/write serialized ModelMeta.

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

### `monolith/agent_service/constants.py`
<a id="monolith-agent-service-constants-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 3
- Purpose/role: constant definitions for agent service.
- Key symbols/classes/functions: `HOST_SHARD_ENV`
- External dependencies: none
- Side effects: none

**Required Behavior (Detailed)**
- Export `HOST_SHARD_ENV = "MONOLITH_HOST_SHARD_N"`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/src/constants.rs` (new) or existing config module.

**Implementation Steps (Detailed)**
1. Add Rust constant with identical name/value.
2. Ensure all config code references same constant.

**Tests (Detailed)**
- Python tests: none.
- Rust tests: optional constant presence test.

**Gaps / Notes**
- Trivial but must be mirrored.

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

### `monolith/agent_service/data_def.py`
<a id="monolith-agent-service-data-def-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 173
- Purpose/role: Data model definitions for agent service (ModelMeta, ResourceSpec, PublishMeta, ReplicaMeta, Event, enums).
- Key symbols/classes/functions: `ModelMeta`, `ResourceSpec`, `PublishMeta`, `ReplicaMeta`, `Event`, `PublishType`, `EventType`.
- External dependencies: `dataclasses_json`, `tensorflow_serving` protos, `AddressFamily`.
- Side effects: none; pure data serialization + address selection logic.

**Required Behavior (Detailed)**
- Type aliases:
  - `ModelState = ModelVersionStatus.State`.
  - `ModelName`, `SubModelName`, `SubModelSize`, `TFSModelName`, `VersionPath` are `NewType` wrappers.
  - `EmptyStatus = StatusProto()` (defined but unused).
- `ModelMeta`:
  - Fields: `model_name`, `model_dir`, `ckpt`, `num_shard=-1`, `action='NONE'`, `spec_replicas=[]`.
  - `get_path(base_path)` -> `os.path.join(base_path, model_name)`.
  - `serialize()` -> UTF-8 JSON bytes via `dataclasses_json`.
  - `deserialize(bytes)` -> `from_json` on UTF-8 string.
- `ResourceSpec`:
  - Fields: `address`, `shard_id`, `replica_id`, `memory`, `cpu=-1.0`, `network=-1.0`, `work_load=-1.0`.
  - `get_path(base_path)` -> `base_path/{shard_id}:{replica_id}`.
  - `serialize`/`deserialize` mirror `ModelMeta`.
- `PublishType` enum: `LOAD=1`, `UNLOAD=2`.
- `PublishMeta`:
  - Fields: `shard_id`, `replica_id=-1`, `model_name`, `num_ps`, `total_publish_num=1`, `sub_models`, `ptype=PublishType.LOAD`, `is_spec=False`.
  - `get_path(base_path)` -> `base_path/{shard_id}:{replica_id}:{model_name}`.
  - `serialize`/`deserialize` mirror `ModelMeta`.
- `ReplicaMeta`:
  - Fields: `address`, `address_ipv6`, `stat=ModelState.UNKNOWN`, `model_name`, `server_type`, `task=-1`, `replica=-1`, `archon_address`, `archon_address_ipv6`.
  - `get_path(bzid, sep='/')` -> `['', bzid, 'service', model_name, f'{server_type}:{task}', str(replica)]` joined by `sep`.
  - `get_address(use_archon=False, address_family=AddressFamily.IPV4)`:
    - Chooses `archon_address`/`archon_address_ipv6` when `use_archon`.
    - Treats `0.0.0.0*` and `[::]*` as invalid (set to None).
    - If `address_family == IPV4`: prefer ipv4, fall back to ipv6; else prefer ipv6 then ipv4.
- `EventType` enum: `PORTAL=1`, `SERVICE=2`, `PUBLISH=3`, `RESOURCE=4`, `UNKNOWN=1` (alias of PORTAL).
- `Event`:
  - Fields: `path=None`, `data=b''`, `etype=EventType.UNKNOWN`.
  - `serialize`/`deserialize` via `dataclasses_json` UTF-8 JSON bytes.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/src/data_def.rs` (new) or `monolith-rs/crates/monolith-core/src`.
- Rust public API surface: structs with serde JSON; enums for state.
- Data model mapping: use `monolith_proto` for ModelState where appropriate.

**Implementation Steps (Detailed)**
1. Port all dataclasses to Rust structs with serde JSON.
2. Implement `serialize`/`deserialize` helpers to match Python bytes encoding.
3. Port `ReplicaMeta.get_address` and path helpers exactly.

**Tests (Detailed)**
- Python tests: `monolith/agent_service/data_def_test.py`
- Rust tests: add roundtrip JSON + address selection tests.

**Gaps / Notes**
- No Rust equivalents yet.

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

### `monolith/agent_service/data_def_test.py`
<a id="monolith-agent-service-data-def-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 52
- Purpose/role: Tests serialization/deserialization roundtrip for `ModelMeta`, `ResourceSpec`, `ReplicaMeta`.
- Key symbols/classes/functions: `DataDefTest.serde` and tests.
- External dependencies: `monolith.agent_service.data_def`.
- Side effects: none.

**Required Behavior (Detailed)**
- `serde(item)`:
  - `cls = item.__class__`.
  - `serialized = item.serialize()`.
  - `recom = cls.deserialize(serialized)`.
  - Asserts `item == recom`.
- `test_model_info`:
  - `ModelMeta(model_name='monolith', num_shard=3, model_dir='/tmp/opt', ckpt='model.ckpt-1234')`.
  - Roundtrip equality via `serde`.
- `test_resource`:
  - `ResourceSpec(address='localhost:123', shard_id=10, replica_id=2, memory=12345, cpu=3.5)` roundtrip.
- `test_replica_meta`:
  - `ReplicaMeta(address='localhost:123', model_name='monolith', server_type='ps', task=0, replica=0)` roundtrip.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/tests/data_def.rs`.
- Rust public API surface: serialization/deserialization methods for the same structs.

**Implementation Steps (Detailed)**
1. Port `serialize`/`deserialize` format (JSON or bytes) to Rust.
2. Ensure equality comparisons are field-wise identical.
3. Add tests for roundtrip parity.

**Tests (Detailed)**
- Python tests: this file
- Rust tests: same data roundtrip assertions.

**Gaps / Notes**
- Serialization format must match Python exactly (field names, defaults).

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

### `monolith/agent_service/mocked_tfserving.py`
<a id="monolith-agent-service-mocked-tfserving-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 399
- Purpose/role: In-process fake TensorFlow Serving gRPC server for tests.
- Key symbols/classes/functions: `ModelConf`, `ModelVersion`, `ModelMeta`, `Event`, `ModelMgr`, `ModelServiceImpl`, `PredictionServiceImpl`, `FakeTFServing`.
- External dependencies: `grpc`, `tensorflow_serving.apis.*`, `model_server_config_pb2`, utils for status/model_config.
- Side effects: starts gRPC server; spawns background thread to update model states.

**Required Behavior (Detailed)**
- Data classes:
  - `ModelConf`: `model_name`, `base_path`, `version_policy='latest'`, `version_data=None`, `model_platform='tensorflow'`, `signature_name=('update','predict')`.
  - `ModelVersion`: `version=0`, `version_label=None`, `state=ModelState.UNKNOWN`.
  - `ModelMeta`: holds `conf`, `versions` (defaults to `[ModelVersion()]`), `_unloading` flag; `is_unloading()`/`set_unloading()`.
  - `Event`: `model_name`, `version`, `state`.
- `ModelMgr`:
  - Holds `_models` dict, `_lock`, `_queue`, `_thread`, `_has_stopped`.
  - `__init__(model_config_list=None)` calls `load` if provided.
  - `load(model_config_list)`:
    - If `latest`: `version_policy='latest'`, `version_data=num_versions`, versions `1..num_versions`.
    - If `all`: `version_policy='latest'`, `version_data=None`, versions `[1]`.
    - Else: `version_policy='specific'`, `version_data=sorted(versions)`, versions per list.
    - Adds `ModelMeta` to `_models` and enqueues `Event(START)` per version.
  - `remove(model_name_list)`:
    - Marks model unloading and enqueues `UNLOADING` for each version.
  - `get_status(model_spec)`:
    - If model exists and no version_choice: returns status for all versions.
    - If `version` set: returns matching version.
    - If `version_label` set: returns matching label.
    - If none found: returns single status with `version=-1`, `NOT_FOUND`, message `{name} is not found`.
  - `get_metadata(model_spec, metadata_field)`:
    - For each requested field, pulls from `ModelConf` if present.
    - If `model_spec.version` set, overlays fields from matching `ModelVersion`.
  - `get_alive_model_names()` returns models not unloading.
  - `start()` spawns `_poll` thread; `stop()` sets `_has_stopped` and joins.
  - `_poll()`:
    - Processes queued events via `_event_handler`.
    - Every 30s: pick random model; if policy != `specific`, append new version (last+1) and enqueue `START`.
  - `_event_handler(event)` transitions:
    - `UNKNOWN -> START` (enqueue LOADING) -> `LOADING` (enqueue AVAILABLE) -> `AVAILABLE`.
    - On AVAILABLE: if policy `latest` and `len(versions) > version_data`, enqueue `UNLOADING` for oldest.
    - `UNLOADING` enqueues `END` (unless already UNLOADING/END).
    - `END` removes version; if none left remove model.
    - Logs transitions; logs error on missing model or unknown event.
- `ModelServiceImpl`:
  - `GetModelStatus` builds response with `model_mgr.get_status`.
  - `HandleReloadConfigRequest`:
    - `old_names = alive`, `new_names` from request configs.
    - Remove `old_names - new_names`, load configs for `new_names - old_names`.
    - Returns `ReloadConfigResponse` with `OK` status.
- `PredictionServiceImpl`:
  - `Predict` is no-op.
  - `GetModelMetadata`:
    - Uses `metadata_field = set(request.metadata_field)`.
    - For each metadata item: `Any(value=bytes(repr(v), 'utf-8'))` assigned to response.
- `FakeTFServing`:
  - `model_config_file` handling:
    - None: create `ModelMgr` from single `gen_model_config(model_name, base_path, version_data=num_versions)`.
    - str: parse pbtxt to `ModelServerConfig`, use config list.
    - else: expect `ModelServerConfig` instance.
  - Starts gRPC server with ModelService + PredictionService, binds `[::]:{port}`.
  - `start()` starts model_mgr and waits for termination; `stop()` stops server and model_mgr.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/tests/support/fake_tfserving.rs`.
- Rust public API surface: Fake gRPC server that emulates ModelService and PredictionService.
- Data model mapping: ModelVersionStatus, ModelSpec, ReloadConfigRequest, MetadataResponse.

**Implementation Steps (Detailed)**
1. Implement in-memory model manager with same state transitions and policies.
2. Implement gRPC services for ModelService and PredictionService.
3. Parse ModelServerConfig text/proto to initialize models.
4. Provide `start()`/`stop()` APIs for tests.

**Tests (Detailed)**
- Python tests: `monolith/agent_service/mocked_tfserving_test.py`
- Rust tests: ensure fake server supports metadata/status/reload as in Python.

**Gaps / Notes**
- Must preserve timing semantics for state transitions used by other tests.

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

### `monolith/agent_service/mocked_tfserving_test.py`
<a id="monolith-agent-service-mocked-tfserving-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 92
- Purpose/role: Tests FakeTFServing gRPC endpoints for metadata, status, and reload config.
- Key symbols/classes/functions: `MockedTFSTest` methods.
- External dependencies: gRPC stubs for ModelService and PredictionService.
- Side effects: starts FakeTFServing in background thread.

**Required Behavior (Detailed)**
- Class setup:
  - `MODEL_NAME='test_model_test'`, `BASE_PATH='/tmp/test_model/monolith'`.
  - `PORT = utils.find_free_port()`.
  - `Address = f'{socket.gethostbyname(socket.gethostname())}:{PORT}'`.
  - Start `FakeTFServing(MODEL_NAME, BASE_PATH, num_versions=2, port=PORT)` in a thread; `sleep(5)`.
- `test_get_model_metadata`:
  - Build `GetModelMetadataRequest` with `gen_model_spec(MODEL_NAME, 2, signature_name='predict')`.
  - `metadata_field` includes `base_path`, `num_versions`, `signature_name`.
  - Call `PredictionServiceStub.GetModelMetadata` and assert `GetModelMetadataResponse` type.
- `test_get_model_status`:
  - Build `GetModelStatusRequest` with `gen_model_spec(MODEL_NAME, 1, signature_name='predict')`.
  - Call `ModelServiceStub.GetModelStatus` and assert `GetModelStatusResponse` type.
- `test_handle_reload_config_request`:
  - Build `ReloadConfigRequest`, extend config list with two `gen_model_config` entries (same name `test_model`, different base paths/versions).
  - Call `HandleReloadConfigRequest` and assert `ReloadConfigResponse` type.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/tests/mocked_tfserving.rs`.
- Rust public API surface: FakeTFServing server for test; gRPC client stubs.

**Implementation Steps (Detailed)**
1. Recreate FakeTFServing server in Rust tests.
2. Add client requests for metadata/status/reload.
3. Assert response types and basic fields.

**Tests (Detailed)**
- Python tests: this file
- Rust tests: parity tests for gRPC responses.

**Gaps / Notes**
- Ensure server thread lifecycle (start/stop) matches Python behavior.

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

### `monolith/agent_service/mocked_zkclient.py`
<a id="monolith-agent-service-mocked-zkclient-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 377
- Purpose/role: In-memory fake Kazoo/ZooKeeper client with watches, nodes, and basic CRUD.
- Key symbols/classes/functions: `ChildrenWatch`, `DataWatch`, `Election`, `Node`, `Catalog`, `FakeKazooClient`.
- External dependencies: `kazoo.protocol.states`, `kazoo.exceptions`.
- Side effects: in-memory state updates and watch callbacks.

**Required Behavior (Detailed)**
- Watch helpers:
  - `ChildrenWatch`: registers with catalog; `__call__` passes `(children,event)` if `send_event`, else `(children)`.
  - `DataWatch`: registers with catalog; `__call__` first tries `(data,state,event)` then falls back to `(data,state)` on `TypeError`.
  - `Election.run` executes callback under a lock; `cancel()` calls `self.lock.cancel()` (not supported by threading.Lock).
- `Node`:
  - Fields: `path`, `value`, `ephemeral`, `children`, `_ctime/_mtime` (unix seconds), `_version`.
  - On init: fires CREATED event to data/children watches if present.
  - `state` returns `ZnodeStat` with `dataLength=len(value)` and `numChildren=len(children)`.
  - `set(value)` updates mtime/version, triggers CHANGED event.
  - `set_data_watch`/`set_children_watch` immediately invoke watchers with current state and `event=None`.
  - `create_child(path, ...)`:
    - Computes child path from basename (root special-cased).
    - Creates `Node`, adds to children.
    - Triggers CHILD event on parent children watch.
  - `get_or_create_child`, `get_child`, `has_child` helpers.
  - `remove_child(path, recursive=False)`:
    - Raises `NotEmptyError` if child has children and `recursive` False.
    - Deletes child and triggers CHILD event; raises `NoNodeError` if missing.
  - `__del__`: fires DELETED event to data/children watches, deletes children, clears fields.
- `Catalog`:
  - Holds root `Node('/')`, watch registries, and `_sequence_paths`.
  - `add_data_watch`/`add_children_watch` attach to existing node if found.
  - `ensure_path(path)` creates nodes along path and attaches registered watches.
  - `create(path, value=b'', ephemeral=False, makepath=False, sequence=False)`:
    - If `sequence`: appends zero-padded 10-digit counter (starts at 0000000000).
    - If `makepath`: ensures parent path; else requires parent to exist.
    - Raises `NodeExistsError` if child exists.
  - `delete`, `set`, `get` delegate to nodes (raises `NoNodeError` if missing).
- `FakeKazooClient`:
  - `start()` initializes `Catalog`; `stop()` clears it.
  - CRUD: `create`, `delete(recursive=True)`, `set`, `get` (returns `(value,state)`), `exists`, `get_children`.
  - `ensure_path` delegates to catalog.
  - `retry` calls function directly.
  - `DataWatch`, `ChildrenWatch`, `Election` exposed via `partial`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/tests/support/fake_zk.rs`.
- Rust public API surface: fake ZK client with watches and CRUD for tests.

**Implementation Steps (Detailed)**
1. Implement Node tree with watch callbacks and version tracking.
2. Implement create/delete/set/get semantics and exceptions.
3. Support `sequence` numbering for `create`.
4. Provide compatible DataWatch/ChildrenWatch interfaces for tests.

**Tests (Detailed)**
- Python tests: `monolith/agent_service/mocked_zkclient_test.py`
- Rust tests: add unit tests for CRUD and watches.

**Gaps / Notes**
- Must mimic Kazoo event types and ordering as closely as possible.

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

### `monolith/agent_service/mocked_zkclient_test.py`
<a id="monolith-agent-service-mocked-zkclient-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 130
- Purpose/role: Tests FakeKazooClient CRUD and watch behaviors.
- Key symbols/classes/functions: `MockedZKClientTest` methods.
- External dependencies: `FakeKazooClient`, kazoo exceptions.
- Side effects: creates and deletes in-memory nodes; prints watch outputs.

**Required Behavior (Detailed)**
- Class setup/teardown:
  - `setUpClass`: `client = FakeKazooClient(); client.start()`.
  - `tearDownClass`: `client.stop()`.
- `test_create`:
  - `client.create('/monolith/zk/data', makepath=True)` returns same path.
  - Catches/ logs `NoNodeError` or `NodeExistsError`.
- `test_set_get`:
  - Create with `include_data=True`; expect `(path, state)`.
  - `set` on non-existent child raises `NoNodeError` (logged).
  - `get` on non-existent child raises `NoNodeError` (logged).
  - `get` on existing path returns original bytes.
- `test_delete`:
  - Create then `client.delete(path)` and `client.delete('/monolith')`.
- `test_data_watch`:
  - Create path, register `DataWatch(path, func)`; callback prints args.
- `test_children_watch`:
  - Register `ChildrenWatch('/monolith/zk', send_event=True)`; create child path.
  - Register `DataWatch` on `/monolith/zk/data`.
  - Create `/monolith/zk/test` to trigger watch callbacks.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/tests/mocked_zkclient.rs`.
- Rust public API surface: Fake ZK client and watch callbacks.

**Implementation Steps (Detailed)**
1. Port FakeKazooClient tests to Rust.
2. Assert correct errors for missing nodes.
3. Ensure watch callbacks are invoked with expected events.

**Tests (Detailed)**
- Python tests: this file
- Rust tests: direct port with similar assertions.

**Gaps / Notes**
- Rust watch callback API may need adaptation; keep behavior parity.

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

### `monolith/agent_service/model_manager.py`
<a id="monolith-agent-service-model-manager-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 371
- Purpose/role: Copies latest model versions from a source (p2p) path to a receive path with lock/marker semantics and periodic refresh.
- Key symbols/classes/functions: `ModelManager` and its helpers (`start`, `loop_once`, `copy_model`, `get_source_data`, `remove_read_lock`).
- External dependencies: `os`, `shutil`, `threading`, `monolith.native_training.metric.cli`.
- Side effects: filesystem reads/writes, directory copies, lock files, optional metrics emission.

**Required Behavior (Detailed)**
- Constants: `WRITE_DONE=".write.done"`, `READ_LOCK=".read.lock"`.
- Defaults:
  - `_wait_timeout=1200s`, `_loop_interval=30s`, `_remain_version_num=5`.
- `start()` / `_start()`:
  - If `model_name` is None: log and return True.
  - Deletes receive path (`delete(receive_path)`).
  - `wait_for_download()` waits for `source_path` existence and a matching `model_name*.write.done` marker.
  - Runs `loop_once()` until a model copy succeeds; then `remove_read_lock()` and starts background thread.
- `run()`:
  - While not `_exist`:
    - `loop_once()` then `remove_read_lock()`.
    - Optionally `check_model_update_time()`; sleeps `loop_interval`.
    - `remove_old_file()` to trim versions.
- Lock files:
  - `create_read_lock(path)` creates `<path>.read.lock` by touching; adds to `_lock_files`.
  - `remove_read_lock()` deletes all `_lock_files` and any `*.read.lock` in source root.
- `wait_for_download()`:
  - Polls every 10s up to `wait_timeout`.
  - Requires a `.write.done` file with prefix `model_name`.
- `get_source_data()`:
  - Walks `source_path` root; collects `.write.done` files and model dirs `model@version`.
  - For each model dir: creates read lock, checks done file exists, parses `model@version`.
  - `real_path = root/model@version/model_name`.
  - `get_version_data(real_path, version)` returns list of `(sub_model/version, full_path)` for each subgraph.
  - Selects latest version by **string comparison** (`old_data[0] < version`).
  - Returns `{model_name: (version, version_data, real_path)}`.
- `copy_model(model_name, version, model_data)`:
  - For each `(sub_model/version, src_path)`:
    - `dst_file = receive_path/model_name/sub_model/version`.
    - Copy to `dst_file-temp` via `shutil.copytree`, then rename to `dst_file`.
    - If `dst_file` exists, counts as ready; if `dst_file-temp` exists, rename later.
  - If any copy fails or `ready_num != sub_model_num`, cleans temp dirs and returns False.
  - Returns `(True, [dst_file...])` on success.
- `loop_once()`:
  - If new version > old, calls `copy_model`; on success updates `_models` and `_latest_models[model]= (version, update_time)`.
- Metrics (`use_metrics=True`):
  - Prefix `data.monolith_serving.online`.
  - `check_model_update_time()` emits:
    - `version.delay = now - int(version)`.
    - `update.delay = now - update_time`.
  - If model missing in `_latest_models`, emits counter `loop_once_failed`.
- `remove_old_file()`:
  - Keeps newest `_remain_version_num` per model; deletes older file paths via `delete()`.
- `delete(path)`:
  - Removes file or directory; logs error on failure.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/src/model_manager.rs`.
- Rust public API surface: `ModelManager` with `start/stop` and loop helpers.
- Data model mapping: filesystem-only (no TF dependencies).

**Implementation Steps (Detailed)**
1. Port lock/marker semantics and path layout.
2. Implement `copy_model` with temp dirs + atomic rename.
3. Preserve polling intervals/timeouts and string-based version comparison.
4. Port metrics emission (feature-gated) with matching names.

**Tests (Detailed)**
- Python tests: `monolith/agent_service/model_manager_test.py`
- Rust tests: build temp tree; verify latest version selection and ignore-old behavior.

**Gaps / Notes**
- `remove_read_lock` uses `os.join` (likely typo) when deleting stray locks.
- Version comparison is string-based, not numeric.

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

### `monolith/agent_service/model_manager_test.py`
<a id="monolith-agent-service-model-manager-test-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 113
- Purpose/role: Tests ModelManager copy behavior and ignores older versions.
- Key symbols/classes/functions: `ModelManagerTest.create_file`, `test_start`, `test_ignore_old`.
- External dependencies: filesystem, `ModelManager`.
- Side effects: creates/deletes temp directories under `TEST_TMPDIR`.

**Required Behavior (Detailed)**
- `create_file(model_name, timestamp, p2p_data_path)`:
  - Creates directories:
    - `p2p/<model>@<ts>/<model>/ps_item_embedding_0/<ts>`
    - `p2p/<model>@<ts>/<model>/ps_item_embedding_1/<ts>`
  - Writes marker `<model>@<ts>.write.done` in `p2p`.
- `test_start`:
  - Creates one version (`1234567`), starts ModelManager.
  - Sets `_wait_timeout=5`, `_loop_interval=5`.
  - Asserts copied directories exist under `model_data/<model>/ps_item_embedding_*`.
  - Stops manager and removes paths.
- `test_ignore_old`:
  - Creates newer version (`1234567`) and starts manager.
  - Adds an older version (`1234566`) afterward; sleeps 11s.
  - Asserts older version dirs do **not** exist under receive path.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/tests/model_manager.rs`.
- Rust public API surface: `ModelManager` with configurable wait/intervals.

**Implementation Steps (Detailed)**
1. Port `create_file` helper to build p2p layout + .write.done.
2. Validate successful copy of newest version.
3. Ensure older version is ignored after manager is running.

**Tests (Detailed)**
- Python tests: `ModelManagerTest.test_start`, `.test_ignore_old`.
- Rust tests: parity tests with temp dirs and shortened intervals.

**Gaps / Notes**
- Ensure Rust tests use temp directories and cleanup to avoid flakiness.

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

### `monolith/agent_service/replica_manager.py`
<a id="monolith-agent-service-replica-manager-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 835
- Purpose/role: Maintains replica registration and status updates in ZK, watches for replica changes, and exposes lookup APIs for serving + parameter sync.
- Key symbols/classes/functions: `ReplicaWatcher`, `ReplicaUpdater`, `ZKListener`, `ReplicaManager`, `SyncBackendWrapper`.
- External dependencies: `kazoo`, `TFSMonitor`, `AgentConfig`, `ReplicaMeta`, `ModelState`, `ClientConfig.TargetExtraInfo`, metrics CLI.
- Side effects: ZK watches + ephemeral nodes, periodic polling, metrics emission.

**Required Behavior (Detailed)**
- `ReplicaWatcher`:
  - Chooses `_zk_watch_address_family`:
    - If requested IPv4 but `is_ipv6_only()` returns True, switches to IPv6.
  - `path_prefix = /{bzid}/service/{base_name}`.
  - `watch_data()`:
    - DC-aware: watch `path_prefix` for `idc:cluster`, then tasks, then replicas.
    - Non-DC: watch tasks directly.
    - Starts `_poll` thread (daemon).
  - `DataWatch` handler:
    - If data empty: mark stat UNKNOWN if existing.
    - Event `CREATED`/None: add/update.
    - `CHANGED`: update.
    - `DELETED`: remove replica; delete task path if empty.
    - `NONE`: set UNKNOWN.
  - `_poll()` every 60s:
    - Rebuilds `replicas_tmp` by scanning ZK.
    - Computes removed replicas; for PS/DENSE deploys, re-registers missing local replicas with correct grpc/archon ports.
    - Updates `self.replicas = replicas_tmp`.
  - Lookup helpers:
    - `get_all_replicas`: returns `{task -> [addr...]}` for AVAILABLE replicas; dc-aware key includes `idc/cluster`.
    - `get_replicas`: returns list for specific task.
    - `get_replica`: returns single addr or list or None.
    - `get_replicas_with_extra_info`: returns `{addr -> TargetExtraInfo(idc, cluster, replica_id)}`.
  - `to_sync_wrapper()`: returns `SyncBackendWrapper`.
- `ReplicaUpdater`:
  - Tracks `meta` map of `replica_path -> ReplicaMeta`.
  - `model_names` includes:
    - `ps_{task_id}` for PS shards owned by this shard id.
    - `entry` if deploy_type MIXED or ENTRY.
    - `dense_0` if dense_alone and deploy_type MIXED or DENSE.
  - `_do_register(replica_path, grpc_port, archon_port)`:
    - Builds host from `MY_HOST_IP` or hostname; IPv6 from `MY_HOST_IPV6` or getaddrinfo.
    - Creates ReplicaMeta with UNKNOWN state and addresses (ipv4/ipv6 + archon).
    - Creates ephemeral znode; for ENTRY with replica_id == -1 uses `sequence=True` and updates config.replica_id.
    - If node exists, updates value if different.
  - `register()`:
    - Registers entry, ps shards, and dense based on deploy_type and shard assignment.
  - `_do_update(name)`:
    - Calls `TFSMonitor.get_model_status(name)`.
    - If error: set stat UNKNOWN and update/create znode.
    - Else: select latest AVAILABLE version (or latest version); if error_code != OK, raise.
    - If state changed, update znode.
  - `_updater()`:
    - Loops every 1s; skips if `_should_update` is False.
    - Updates all `model_names`; logs exceptions with traceback.
  - `_check_version()`:
    - Emits metrics:
      - `serving_model.latest_version`
      - `serving_model.since_last_update` (global metric)
      - `serving_model.update_ts`
    - Tags include model_name, idc/cluster, replica_id, shard_id, base_name.
  - `_watch_update()` runs `_check_version()` every 60s.
  - `_reregister()` every 10s: if `_should_reregister` True, calls `register()` and sets `_should_update=True`.
  - `start()` starts TFSMonitor and threads; `stop()` joins threads and clears meta.
- `ZKListener`:
  - On `LOST`: disables watcher polling + updater updates, sets `_has_lost`.
  - On reconnect after LOST: sets `_should_reregister=True`, sleeps 5s, re-enables polling.
- `ReplicaManager`:
  - Wires watcher + updater and registers ZKListener.
  - `start()`: `updater.register()`, `watcher.watch_data()`, `updater.start()`.
  - `stop()`: stops updater then watcher.
  - `is_ps_set_started()` ensures each PS task has at least one AVAILABLE replica; `is_dense_set_started()` checks dense replicas when enabled.
- `SyncBackendWrapper`:
  - `subscribe_model` stores model name.
  - `get_sync_targets("ps_i")` returns `(sub_graph, watcher.get_replicas_with_extra_info(...))`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/src/replica_manager.rs`.
- Rust public API surface: `ReplicaWatcher`, `ReplicaUpdater`, `ReplicaManager`, `SyncBackendWrapper`.
- Data model mapping: `ReplicaMeta`, `ModelState`, `TFSServerType`, `ServerType`, `ClientConfig.TargetExtraInfo`.
- Integration points: ZK client, TFSMonitor, metrics client.

**Implementation Steps (Detailed)**
1. Implement ZK watchers and periodic polling reconciliation.
2. Port registration/update logic with identical address selection and replica_id sequencing.
3. Port metrics emission with same prefixes/tags and cadence.
4. Implement ZK LOST handling and re-registration semantics.
5. Provide sync backend wrapper returning extra-info targets.

**Tests (Detailed)**
- Python tests: `monolith/agent_service/replica_manager_test.py`
- Rust tests: fake ZK + fake TFS server to validate registration, updates, and watcher maps.

**Gaps / Notes**
- `get_replicas_with_extra_info` returns a dict despite type hint List[str].
- Uses environment variables (`MY_HOST_IP`, `MY_HOST_IPV6`, `MONOLITH_METRIC_PREFIX`, `TCE_PSM`) for address/metrics.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/src/replica_manager.rs`.
- Rust public API surface: `ReplicaManager`, `ReplicaWatcher`, `ReplicaUpdater`, `SyncBackendWrapper`.
- Data model mapping: `ReplicaMeta`, `ModelState`, `TFSServerType`, `ServerType`.
- Integration points: ZK client + TFSMonitor + metrics client.

**Implementation Steps (Detailed)**
1. Implement watcher with ZK watches and periodic polling for reconciliation.
2. Implement updater with registration, status updates, and metrics.
3. Port dc-aware path parsing (`ZKPath`) and address family handling.
4. Port connection loss handling and re-registration semantics.
5. Provide SyncBackend wrapper for parameter sync features.

**Tests (Detailed)**
- Python tests: `monolith/agent_service/replica_manager_test.py`
- Rust tests: fake ZK + fake TFS server to validate registration and updates.

**Gaps / Notes**
- This module depends on TFSMonitor (gRPC) and ZK; ensure fakes are available for tests.

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

### `monolith/agent_service/replica_manager_test.py`
<a id="monolith-agent-service-replica-manager-test-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 126
- Purpose/role: Sets up FakeTFServing and helper registration for ReplicaManager tests; no actual test cases.
- Key symbols/classes/functions: `ReplicaMgrTest.setUpClass`, `register`.
- External dependencies: `FakeTFServing`, `FakeKazooClient`, `ReplicaMeta`, `AgentConfig`.
- Side effects: starts FakeTFServing servers for entry/ps in background threads.

**Required Behavior (Detailed)**
- `setUpClass`:
  - Sets env vars:
    - `MONOLITH_HOST_SHARD_N=5`, `SHARD_ID=1`, `REPLICA_ID=2`,
      `TCE_INTERNAL_IDC=lf`, `TCE_LOGICAL_CLUSTER=default`.
  - Builds `AgentConfig` with:
    - `bzid='bzid'`, `base_name=MODEL_NAME`, `deploy_type='mixed'`,
      `base_path=BASE_PATH`, `num_ps=20`, `num_shard=5`, `dc_aware=True`.
  - Extracts `model_config_file` path from `agent_conf.get_cmd(...)` for ENTRY and PS.
  - Starts two `FakeTFServing` instances (entry + ps) in background threads.
- `tearDownClass`:
  - Stops FakeTFServing servers and joins threads.
- `register(zk)` helper:
  - Creates ReplicaMeta entries for all shard/replica combos except current shard/replica.
  - Adds ps task replicas (`ps:{task_id}/{replica_id}`) and entry replicas (`entry:0/{replica_id}`).
  - Uses `find_free_port()` for each address and creates ephemeral nodes in ZK.
- Note: No actual `test_*` methods are implemented beyond helper setup.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/tests/replica_manager.rs`.
- Rust public API surface: FakeTFServing + ReplicaManager registration utilities.
- Data model mapping: same path formatting and ReplicaMeta JSON encoding.

**Implementation Steps (Detailed)**
1. Port setup/teardown as a test fixture in Rust.
2. Implement `register` helper to populate fake ZK.
3. Add real assertions in Rust tests (missing in Python).

**Tests (Detailed)**
- Python tests: none (setup only).
- Rust tests: verify watcher/updater interactions with fake TFS and ZK nodes.

**Gaps / Notes**
- Python test file lacks assertions; Rust should add coverage for lookups and updates.

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

### `monolith/agent_service/resource_utils.py`
<a id="monolith-agent-service-resource-utils-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 269
- Purpose/role: Resource and model-size utilities (memory, CPU, model size computation, HDFS helpers).
- Key symbols/classes/functions: `cal_model_info_v2`, `total_memory`, `cal_available_memory`, `CPU`, `num_cpu`, `cal_cpu_usage`.
- External dependencies: `tensorflow.io.gfile`, `psutil`, `subprocess`, `export_state_utils`.
- Side effects: reads cgroup files, runs shell commands, reads filesystem/HDFS.

**Required Behavior (Detailed)**
- `_get_pod_cgroup_path()`:
  - Runs `cat /proc/1/cgroup`, finds line containing `:memory:`, returns trailing path without slashes.
  - On error returns `None`.
  - `_POD_CGROUP_PATH` computed at import time.
- `exists(dirname)` -> `tf.io.gfile.isdir(dirname) or tf.io.gfile.exists(dirname)`.
- `open_hdfs(fname)`:
  - Builds cmd `[_HADOOP_BIN, 'fs', '-text', ...]` (global `_HADOOP_BIN` expected).
  - Accepts string or list/tuple of paths.
  - Retries up to 3 times; logs exceptions; asserts output list is not None.
  - Yields non-empty stripped lines.
- `cal_model_info_v2(exported_models_path, ckpt=None, version=None)`:
  - Normalizes path to absolute (rstrip `/`), requires `tf.io.gfile.exists`.
  - Initializes `model_info` with each sub_model under export dir (excluding dotfiles) size 0.
  - If `ckpt` None: uses `tf.train.get_checkpoint_state(ckpt_base_path)` and basename of `model_checkpoint_path`.
  - `global_step = -1` if ckpt None else `int(ckpt.split('-')[-1])`.
  - If `version` None:
    - For each sub_model, attempt `export_state_utils.get_export_saver_listener_state(tfs_base_path)`; if state and `global_step>=0`, collect versions from entries and pick matching `global_step`.
    - Else list numeric subdirs under sub_model.
    - Intersect versions across sub_models; pick max if `version` still None.
  - Else cast `version` to int.
  - For each sub_model: sum all file sizes under `exported_models_path/sub_model/version` via `tf.io.gfile.walk`.
  - If assets dir `ckpt_base_path/{ckpt}.assets` exists:
    - For files matching `ROW` regex, add size to `model_info['ps_{index}']`.
  - Returns `{sub_model_name: (size, version_path)}`.
- Memory helpers:
  - `total_memory` reads `/sys/fs/cgroup/memory/{_POD_CGROUP_PATH}/memory.limit_in_bytes`; if 0 uses `MY_MEM_LIMIT`.
  - `total_memory_v2` uses `psutil.virtual_memory().total`.
  - `cal_available_memory` reads cgroup `memory.usage_in_bytes` and `memory.limit_in_bytes`, returns limit - usage.
  - `cal_available_memory_v2` uses `psutil.virtual_memory().available`.
- CPU helpers:
  - `CPU.wall_clock()` uses `time.time_ns()` (fallback to `date +%s%N` subprocess).
  - `CPU.cpu_clock()` reads cpuacct usage from file.
  - `CPU.cpu_usage()` returns delta_cpu/delta_wall and updates stored clocks.
  - `num_cpu()` reads `cpu.cfs_quota_us` and `cpu.cfs_period_us`, fallback to `MY_CPU_LIMIT` if period 0.
  - `cal_cpu_usage()` samples 5 times, 1s sleep, averages `round(cpu_usage*100,2)`.
  - `cal_cpu_usage_v2()` uses `psutil.cpu_percent()`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/src/resource_utils.rs`.
- Rust public API surface: functions for model size + resource metrics.
- Data model mapping: `SubModelName`, `SubModelSize`, `VersionPath` equivalents.

**Implementation Steps (Detailed)**
1. Implement cgroup parsing for memory/cpu or provide platform-specific fallbacks.
2. Port `cal_model_info_v2` using filesystem traversal and checkpoint state parsing.
3. Replace `tf.io.gfile` with Rust FS/HDFS abstraction as needed.
4. Provide `psutil`-equivalent data via `sysinfo` or `/proc` parsing.

**Tests (Detailed)**
- Python tests: `monolith/agent_service/resource_utils_test.py`
- Rust tests: validate memory/cpu functions and model info on fixture dirs.

**Gaps / Notes**
- Requires HDFS support or a stub for `open_hdfs`.

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

### `monolith/agent_service/resource_utils_test.py`
<a id="monolith-agent-service-resource-utils-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 36
- Purpose/role: Tests resource utility functions (memory/cpu).
- Key symbols/classes/functions: `UtilTest.test_cal_avaiable_memory_v2`, `.test_cal_cpu_usage_v2`
- External dependencies: `resource_utils`.
- Side effects: reads system stats.

**Required Behavior (Detailed)**
- `test_cal_avaiable_memory_v2`:
  - Calls `total_memory_v2()` and `cal_available_memory_v2()`; asserts `0 < available < total`.
- `test_cal_cpu_usage_v2`:
  - Calls `cal_cpu_usage_v2()`; asserts `0 <= usage <= 100`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/tests/resource_utils.rs`.
- Rust public API surface: resource utility functions.

**Implementation Steps (Detailed)**
1. Port tests with `sysinfo` or `/proc` sources.
2. Ensure numeric bounds match Python behavior.

**Tests (Detailed)**
- Python tests: this file
- Rust tests: same assertions.

**Gaps / Notes**
- Tests may be flaky on constrained CI; consider tolerances.

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

### `monolith/agent_service/run.py`
<a id="monolith-agent-service-run-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 39
- Purpose/role: Entry-point multiplexer for `agent`, `agent_client`, and `tfs_client` binaries.
- Key symbols/classes/functions: `main`
- External dependencies: `absl.app`, `flags`, agent and client entrypoints.
- Side effects: dispatches into selected CLI.

**Required Behavior (Detailed)**
- Flag `bin_name` selects:
  - `agent` -> `monolith.agent_service.agent.main`
  - `agent_client` -> `monolith.agent_service.agent_client.main`
  - `tfs_client` -> `monolith.agent_service.tfs_client.main`
- Unknown value raises `ValueError`.
- Default `bin_name` is `"agent"`; `app.run(main)` invoked in `__main__`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-cli/src/bin/monolith.rs` (dispatcher) or separate binaries.
- Rust public API surface: CLI dispatch or multi-bin setup.

**Implementation Steps (Detailed)**
1. Decide whether to provide a dispatcher binary or separate `cargo` bins.
2. If dispatcher, implement `bin_name` flag with same choices.
3. Forward args to target subcommand.

**Tests (Detailed)**
- Python tests: none
- Rust tests: optional CLI smoke test.

**Gaps / Notes**
- Rust may prefer multiple bins; document any deviations.

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

### `monolith/agent_service/svr_client.py`
<a id="monolith-agent-service-svr-client-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 70
- Purpose/role: Thin gRPC client wrapper for AgentService (heart_beat and get_replicas).
- Key symbols/classes/functions: `SvrClient`, `heart_beat`, `get_replicas`
- External dependencies: `grpc`, `AgentServiceStub`, `AgentConfig`.
- Side effects: gRPC calls, prints responses.

**Required Behavior (Detailed)**
- `__init__`: accepts config path or `AgentConfig` object; defers stub creation.
- `stub` property:
  - Uses `MY_HOST_IP` env or local hostname; connects to `{host}:{agent_port}`.
- `get_server_type`:
  - If input is string, ignores the value and maps **`FLAGS.server_type`** to enum (ps/entry/dense); else returns `st`.
- `heart_beat`: sends `HeartBeatRequest` and prints `resp.addresses` (flush=True).
- `get_replicas`: sends `GetReplicasRequest` and prints `resp.address_list.address` (flush=True).

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-cli/src/agent_svr_client.rs` or library module.
- Rust public API surface: `SvrClient` with `heart_beat` and `get_replicas`.

**Implementation Steps (Detailed)**
1. Implement gRPC stub creation with env host selection.
2. Port enum mapping for server types.
3. Preserve stdout printing behavior (for CLI usage).

**Tests (Detailed)**
- Python tests: none
- Rust tests: add gRPC stub tests using a fake AgentService.

**Gaps / Notes**
- The string mapping uses global FLAGS; ensure Rust CLI has equivalent.

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

### `monolith/agent_service/tfs_client.py`
<a id="monolith-agent-service-tfs-client-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 503
- Purpose/role: CLI for TensorFlow Serving status/metadata/load/predict/profile; includes data generation and format conversion.
- Key symbols/classes/functions: `read_header`, `read_data`, `generate_random_instance`, `generate_random_example_batch`, `get_instance_proto`, `get_example_batch_proto`, `gen_random_file`, `get_example_batch_proto_v2`, `get_example_batch_to_instance`, `ProfileThread`, `main`.
- External dependencies: TF Serving protos, matrix `ExampleBatch`/`Instance`, `FeatureList`, `data_gen_utils`.
- Side effects: reads/writes files, makes gRPC requests, spawns threads, prints output.

**Required Behavior (Detailed)**
- Flags/global constants:
  - Flags: `signature_name="serving_default"`, `feature_list=None`, `file_type="pb"`, `batch_size=8`, `lagrangex_header=False`, `has_sort_id=False`, `kafka_dump=False`, `kafka_dump_prefix=False`, `parallel_num=1`, `profile_duration=600`, `profile_data_dir=None`.
  - Globals: `VALID_SLOTS=[]`, `_NUM_SLOTS=6`, `_VOCAB_SIZES=[5,5,5,5,5,5]`, `SKIP_LIST` set.
- Input parsing helpers:
  - `read_header(stream)`:
    - `int_size=8`.
    - If `FLAGS.lagrangex_header`, read 8 bytes and return.
    - Else:
      - `aggregate_page_sortid_size=0`.
      - If `FLAGS.kafka_dump_prefix`:
        - Read `<Q` size; if size==0 read another `<Q`; else set `aggregate_page_sortid_size=size`.
      - If `FLAGS.has_sort_id`:
        - If `aggregate_page_sortid_size==0`, read `<Q` size; else use `aggregate_page_sortid_size`.
        - Read `size` bytes (sort_id payload).
      - If `FLAGS.kafka_dump`, read 8 bytes.
  - `read_data(stream)`: calls `read_header`, reads `<Q` size (8 bytes) then reads and returns that many bytes.
- Data generation:
  - `generate_random_instance(slots=None, vocab_sizes=_VOCAB_SIZES)`:
    - If slots None: `slots = [1..len(_VOCAB_SIZES)]`.
    - `max_vocab = max(vocab_sizes)`.
    - Build `fids = (slot << 54) | (i * max_vocab + rand(1, vocab_sizes[i]))` for each `i, slot` and repeat `vocab_sizes[i]` times.
    - Return serialized `Instance` with `fid` populated.
  - `generate_random_example_batch(feature_list, batch_size=256)`:
    - Creates `ExampleBatch` with `batch_size`.
    - Skips features whose name contains any token in `SKIP_LIST`.
    - Only processes features with `"_id"` or `"_name"` in name.
    - For each kept feature:
      - Adds `named_feature_list` with `name=feature.name`.
      - For each example in batch:
        - If `feature.method` startswith `vectortop` and `feature.args[0]` is numeric:
          - `num=int(args[0])`; if `num>0`, replace with `randint(1, num)`.
          - Create `num` fids `(feature.slot << 48) | rand(1, sys.maxsize-1)` into `fid_v2_list.value`.
        - Else: single fid `(feature.slot << 48) | rand(1, (1<<48)-1)` into `fid_v2_list.value`.
    - Adds `__LINE_ID__` feature list:
      - For each example: `LineId(sample_rate=0.001, req_time=now_ts - rand(1,1000), actions=[rand(1,3), rand(3,5)])` serialized into `bytes_list.value`.
    - Returns serialized `ExampleBatch`.
  - `gen_random_file(input_file, variant_type="example_batch")`:
    - Asserts `input_file` not None and `VALID_SLOTS` non-empty.
    - Builds `ParserArgs` with sparse_features from `VALID_SLOTS`, extra_features list, shapes, `batch_size=FLAGS.batch_size`, `variant_type`.
    - Uses `data_gen_utils.gen_random_data_file(..., sort_id=FLAGS.has_sort_id, kafka_dump=FLAGS.kafka_dump, num_batch=1, actions=[1..12])`.
- Tensor proto conversion:
  - `get_instance_proto(input_file=None, batch_size=256)`:
    - If `input_file` None: generate `batch_size` random instances.
    - Else: read `batch_size` instances via `read_data` and parse `Instance` per record.
    - Return `utils.make_tensor_proto(instances)` (list of serialized bytes).
  - `get_example_batch_proto(input_file=None, feature_list=None, batch_size=256, file_type='pb')`:
    - If no file: generate random ExampleBatch.
    - Else parse ExampleBatch from pb (`read_data`) or pbtxt (`text_format.Parse`).
    - Return `utils.make_tensor_proto([example_batch])`.
  - `get_example_batch_proto_v2(input_file)`:
    - If file missing, call `gen_random_file`.
    - Parse ExampleBatch from `read_data`.
    - `user_fname_set = {get_feature_name_and_slot(slot)[0] for slot in user_features}`.
    - For each `named_feature_list` with name in set:
      - Set `type = FeatureListType.SHARED`.
      - Replace feature list with `batch_size` copies of the first feature.
    - Return `utils.make_tensor_proto([serialized ExampleBatch])`.
  - `get_example_batch_to_instance(input_file, file_type)`:
    - Parse ExampleBatch from pb or pbtxt.
    - For each example index, create `Instance`:
      - Use shared feature if `named_feature_list.type == SHARED`.
      - Map `__LABEL__` to `inst.label` (float_list).
      - Map `__LINE_ID__` bytes to `inst.line_id.ParseFromString`.
      - For other features, select first non-empty value list in order:
        - `fid_v1_list`: compute `slot_id = fid >> 54`; convert each fid to v2 with `(slot_id << 48) | (mask & v)` where `mask=(1<<48)-1`.
        - `fid_v2_list`: copy.
        - `float_list` or `double_list`: copy into `float_value`.
        - `int64_list`: copy into `int64_value`.
        - `bytes_list`: copy into `bytes_value`.
    - Return `utils.make_tensor_proto(inst_list)` (serialized instances).
- Profiling thread:
  - `ProfileThread.run()` loops until `int(time.time()) - run_st < repeat_time`:
    - Job 0 logs progress every 60 seconds (`show_count` gate).
    - Builds PredictRequest with signature name flag, picks random cached example_batch and random stub.
    - Predict timeout = 30s.
    - Records per-request latency (ms) into list, increments count.
    - On exception: logs warning and continues.
  - `get_result()` joins thread and returns latency list.
- CLI (`main`):
  - Calls `enable_tob_env()` and `env_utils.setup_host_ip()`.
  - Loads `agent_conf = AgentConfig.from_file(FLAGS.conf)`.
  - `host = env["MY_HOST_IP"]` or `socket.gethostbyname(socket.gethostname())`.
  - Default `model_name` if flag unset:
    - PS: `ps_{shard_id}`; DENSE: `TFSServerType.DENSE`; else ENTRY.
  - `target` by deploy_type: entry/ps/dense port; allows override `FLAGS.target` and comma-separated targets.
  - Creates `channel_list` from targets, `channel` for first target.
  - `cmd_type` behaviors:
    - `status`: `ModelServiceStub.GetModelStatus` for model name/signature; print response.
    - `meta`: `PredictionServiceStub.GetModelMetadata`, metadata fields `base_path`, `num_versions`, `signature_name`; print response.
    - `load`: parse `ModelServerConfig` from pbtxt `FLAGS.input_file`, `HandleReloadConfigRequest`, log "load done", return status.
    - `profile`:
      - Assert `VALID_SLOTS` non-empty; ensure `profile_data_dir` exists.
      - Build list of up to 500 data files; generate missing via `gen_random_file` (uuid filenames).
      - Load each file with `get_example_batch_proto_v2`.
      - Spawn `parallel_num` `ProfileThread`s; gather latency list.
      - Compute avg latency, p99 (index `round((n-1)*0.99)`), QPS; log summary.
    - Default (`get`): build PredictRequest; choose inputs:
      - `input_type == "instance"`: if file missing, `gen_random_file(..., "instance")`; `get_instance_proto`.
      - `input_type == "example_batch"`: use provided file or new uuid, `get_example_batch_proto_v2`.
      - Else: `get_example_batch_to_instance`.
      - Call `Predict(timeout=30)` and print response.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-cli/src/bin/tfs_client.rs`.
- Rust public API surface: CLI entrypoint + helper functions for data parsing and proto conversion.
- Data model mapping: matrix ExampleBatch/Instance protos + TF Serving PredictRequest.
- Feature gating: `tf-runtime` and `matrix-protos` features.

**Implementation Steps (Detailed)**
1. Port file header parsing and size-prefixed record reading.
2. Implement ExampleBatch/Instance generation and conversion logic.
3. Implement gRPC clients for ModelService and PredictionService.
4. Port profiling logic with multi-threaded request loop and latency stats.
5. Mirror CLI flags and defaults exactly.

**Tests (Detailed)**
- Python tests: `monolith/agent_service/tfs_client_test.py`
- Rust tests: port get_instance_proto and get_example_batch_to_instance tests.

**Gaps / Notes**
- `user_features` referenced in `get_example_batch_proto_v2` is not defined in this module; Rust port must identify the intended source or expose configuration.
- `VALID_SLOTS` defaults empty; profile path asserts it is non-empty (likely set via external config/feature list).
- `_NUM_SLOTS` is unused in this module; `_VOCAB_SIZES` governs random instance generation.

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

### `monolith/agent_service/tfs_client_test.py`
<a id="monolith-agent-service-tfs-client-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 50
- Purpose/role: Tests tensor proto generation and ExampleBatch-to-Instance conversion.
- Key symbols/classes/functions: `TFSClientTest.test_get_instance_proto`, `.test_get_example_batch_to_instance_*`
- External dependencies: `tfs_client` helpers.
- Side effects: reads test data files.

**Required Behavior (Detailed)**
- `test_get_instance_proto`:
  - Calls `get_instance_proto()` with default `batch_size=256`.
  - Asserts `tensor_proto.dtype == 7` and `tensor_shape.dim[0].size == 256`.
- `test_get_example_batch_to_instance_from_pb`:
  - Uses file `monolith/native_training/data/training_instance/examplebatch.data`.
  - Sets `FLAGS.lagrangex_header = True`.
  - Calls `get_example_batch_to_instance(file, 'pb')`.
- `test_get_example_batch_to_instance_from_pbtxt`:
  - Uses file `monolith/agent_service/example_batch.pbtxt`.
  - Sets `FLAGS.lagrangex_header = True`.
  - Calls `get_example_batch_to_instance(file, 'pbtxt')`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-cli/tests/tfs_client.rs`.
- Rust public API surface: helper functions for instance/example batch parsing.

**Implementation Steps (Detailed)**
1. Port tests using fixture files (`examplebatch.data`, `example_batch.pbtxt`).
2. Assert dtype and shapes match Python behavior.

**Tests (Detailed)**
- Python tests: this file
- Rust tests: parity tests for parsing and tensor construction.

**Gaps / Notes**
- Ensure Rust uses the same byte-order and header handling as Python.

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

### `monolith/agent_service/tfs_monitor.py`
<a id="monolith-agent-service-tfs-monitor-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 303
- Purpose/role: gRPC monitor for TensorFlow Serving model status and config reload.
- Key symbols/classes/functions: `TFSMonitor`, `get_model_status` (singledispatch), `gen_model_config`, `handle_reload_config_request`.
- External dependencies: TF Serving gRPC stubs, `ModelServerConfig`, `PublishMeta`, `StatusProto`.
- Side effects: gRPC calls to TFS servers; opens/closes channels.

**Required Behavior (Detailed)**
- Host selection:
  - `host` property: if unset or localhost, resolves to `get_local_ip()`.
- Stubs:
  - `connect()` creates `grpc.insecure_channel` + `ModelServiceStub` and `PredictionServiceStub` per deploy_type:
    - ENTRY: `tfs_entry_port`
    - PS: `tfs_ps_port`
    - DENSE: `tfs_dense_port` (only if dense_alone).
  - `start()` resets stubs and calls `connect()`.
  - `stop()` closes channels and clears stubs.
- Address selection:
  - `get_addr(sub_model_name)` chooses port based on deploy_type and sub_model.
  - `get_service_type(sub_model_name)` maps to `TFSServerType` or None.
- `get_model_status(PublishMeta, fix_dense_version=False)`:
  - For each sub_model in `pm.sub_models`:
    - Builds `tfs_model_name = f"{model}:{sub_model}"`.
    - Dense node detection:
      - If not dense_alone and entry → dense node.
      - If dense_alone and dense → dense node.
    - If dense node and `fix_dense_version=False`, request **no version** (latest).
    - Else request specific version = basename of version path.
  - If no statuses returned: create `ModelVersionStatus` with `state=UNKNOWN` and `StatusProto(error_code=NOT_FOUND)`.
  - If multiple statuses: sort by version and select latest (or AVAILABLE if present).
  - On `_InactiveRpcError`: returns UNKNOWN with StatusProto error_code from `e.code().value[0]`.
  - Returns `{tfs_model_name: (version_path, ModelVersionStatus)}`.
- `get_model_status(name, version=None, signature_name=None)` overload:
  - Determines service_type via `get_service_type`.
  - If None, returns empty list.
  - Else sends `GetModelStatusRequest` and returns list of ModelVersionStatus.
- `gen_model_config(pms, fix_dense_version=False)`:
  - Builds `ModelServerConfig` per service type.
  - Skips `PublishMeta` where `ptype == UNLOAD`.
  - Dense nodes:
    - version_policy = `latest` if not fix_dense_version; else `specific`.
    - version_data = `1` if latest, else version basename.
  - Non-dense nodes: `specific` version policy with basename.
  - Appends configs into appropriate service type list.
- `handle_reload_config_request(service_type, model_configs)`:
  - Ensures `DEFAULT_MODEL_CONFIG` is present if missing.
  - Sends `HandleReloadConfigRequest` to appropriate stub; returns `StatusProto`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/src/tfs_monitor.rs`.
- Rust public API surface: `TfsMonitor` with status query + reload APIs.
- Data model mapping: TF Serving protos, `PublishMeta`, `ModelServerConfig`.

**Implementation Steps (Detailed)**
1. Port gRPC client setup for entry/ps/dense.
2. Implement overload semantics for `get_model_status`.
3. Port dense-node version policy logic (latest vs specific).
4. Add default model config injection on reload.

**Tests (Detailed)**
- Python tests: `monolith/agent_service/tfs_monitor_test.py`
- Rust tests: fake TFS servers; verify reload responses + status selection.

**Gaps / Notes**
- Error code mapping uses `e.code().value[0]` (non-obvious; must preserve).

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

### `monolith/agent_service/tfs_monitor_test.py`
<a id="monolith-agent-service-tfs-monitor-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 182
- Purpose/role: Tests TFSMonitor reload and remove config with FakeTFServing.
- Key symbols/classes/functions: `TFSMonitorTest.test_reload_config`, `.test_remove_config`
- External dependencies: FakeTFServing, ModelServerConfig, PublishMeta.
- Side effects: starts fake TF serving servers.

**Required Behavior (Detailed)**
- Class setup:
  - Set env vars: `HOST_SHARD_ENV=10`, `SHARD_ID=1`, `REPLICA_ID=2`.
  - Build `AgentConfig(bzid='bzid', deploy_type='mixed')`.
  - Start `FakeTFServing` for entry and ps with `num_versions=2`, ports from config, `ModelServerConfig()`.
  - Start servers in threads, `sleep(2)` for readiness.
  - Create `TFSMonitor`, call `connect()`, init `data = {}`.
- Class teardown: `monitor.stop()`, stop both fake servers.
- `setUp`:
  - Build `sub_models` dict with `entry`, `ps_0`, `ps_3`, `ps_5` pointing at `path.format(sub_model, version)`.
  - `PublishMeta(model_name='test_1', num_ps=5, shard_id/replica_id from config)`.
  - Save `data['setUp'] = monitor.get_model_status(pm)`.
- `tearDown`:
  - Rebuild same `PublishMeta`, `sleep(1)`.
  - Compare `before_status` vs `after_status = monitor.get_model_status(pm)`; assert same length.
  - If `data['execute'] == 'reload_config'`:
    - For each model, ensure version path unchanged.
    - `before` status must be `version == -1` and `error_code == 5` (NOT_FOUND).
    - `after` status cases:
      - `version == -1`: allowed (no extra check).
      - `version == 1`: only for `entry` model names.
      - else `version == int(os.path.basename(version_path))`.
  - Else (remove_config):
    - For each model, `after.version == -1`, version path unchanged.
    - If `before.version == -1`, `before.status.error_code == 5` (NOT_FOUND).
    - Else `before.version > 0`.
- `test_reload_config`:
  - For `i in range(10)`:
    - `num_ps = random.randint(5, 20)`.
    - `sub_models` includes `ps_{j}` for `j in range(num_ps)` where `j % 3 == 0`.
    - Always include `entry` submodel.
    - Build `PublishMeta(model_name=f'test_{i}', num_ps=num_ps)` and append.
  - `model_configs = monitor.gen_model_config(pms)`.
  - For each service_type, if config list non-empty: `handle_reload_config_request`.
  - Set `data['execute'] = 'reload_config'`.
- `test_remove_config`:
  - Same as reload but `i in range(5, 10)` (different model names), set `data['execute'] = 'remove_config'`.
- Randomness: no seed set; tests rely on structural invariants rather than specific PS counts.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/tests/tfs_monitor.rs`.
- Rust public API surface: TFSMonitor + FakeTFServing.

**Implementation Steps (Detailed)**
1. Port fake TFS server setup.
2. Port PublishMeta-based config generation and reload requests.
3. Assert status responses match Python expectations (NOT_FOUND or version numbers).

**Tests (Detailed)**
- Python tests: this file
- Rust tests: parity test for reload/remove behavior.

**Gaps / Notes**
- Requires deterministic fake TFS server behavior for version states.

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

### `monolith/agent_service/tfs_wrapper.py`
<a id="monolith-agent-service-tfs-wrapper-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 202
- Purpose/role: Wraps TensorFlow Serving process launch, config file handling, and model status queries; provides a fake wrapper for tests.
- Key symbols/classes/functions: `TFSWrapper`, `FakeTFSWrapper`.
- External dependencies: `subprocess`, `grpc`, TF Serving protos, `monolith.utils.find_main`.
- Side effects: launches external process, writes logs, opens gRPC channel, kills process.

**Required Behavior (Detailed)**
- Globals:
  - `TFS_BINARY = os.environ.get("MONOLITH_TFS_BINARY", None)`.
  - `State = ModelVersionStatus.State`.
- `TFSWrapper.__init__`:
  - Stores ports, model config path, binary config, log file.
  - Runs `strings $TFS_BINARY | grep PredictionServiceGrpc` (shell=True) to detect gRPC remote op support.
  - Sets `_is_grpc_remote_op` based on `returncode == 0`.
- `_prepare_cmd()`:
  - Base flags:
    - `--model_config_file=...`
    - `--port=<grpc_port>`
    - `--rest_api_port=<http_port>`
    - `--model_config_file_poll_wait_seconds=60`
    - Archon flags: `--archon_port`, `--archon_rpc_psm`, `--archon_rpc_cluster`
    - `--metrics_namespace_prefix=<psm>`
  - If not gRPC remote op: add `--archon_entry_to_ps_rpc_timeout=<fetch_ps_timeout_ms>`.
  - Always adds `--conf_file=conf/service.conf` and `--log_conf=conf/log4j.properties`.
  - For each field in `TfServingConfig` type hints:
    - Uses default vs current; skips if equal.
    - For `platform_config_file`: if None, set `--platform_config_file=conf/platform_config_file.cfg`.
    - For bool: lower-case value.
- `start()`:
  - `os.chdir(find_main())`.
  - Launches process via `subprocess.Popen(tfs_cmd.split(), shell=False, stderr=STDOUT, stdout=log_file, env=os.environ)`.
  - Opens gRPC channel to `localhost:<grpc_port>` and `ModelServiceStub`.
- `stop()`:
  - Closes channel; closes stdout if exists; kills process.
- `poll()`:
  - Calls `proc.poll()` and returns `returncode`.
- `model_config_text()`:
  - Reads config file content.
- `list_saved_models()`:
  - Parses `ModelServerConfig` from text and returns `config.name` list.
- `list_saved_models_status()`:
  - For each saved model:
    - Calls `GetModelStatus`.
    - If multiple versions: picks latest AVAILABLE else latest by version.
    - On RPC error: returns `ModelVersionStatus(state=UNKNOWN, status=StatusProto(error_code=e.code().value[0], error_message=e.details()))`.
  - Returns `{model_name: ModelVersionStatus}`.
- `FakeTFSWrapper`:
  - No process; reads config file and returns AVAILABLE for all models.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/src/tfs_wrapper.rs`.
- Rust public API surface: `TfsWrapper` + `FakeTfsWrapper`.
- Feature gating: `grpc` for ModelService client; optional process spawning in non-test builds.

**Implementation Steps (Detailed)**
1. Port command builder with exact flags and `TfServingConfig` mapping.
2. Implement process spawn with log redirection and working dir `find_main()`.
3. Implement gRPC GetModelStatus handling and version selection.
4. Implement FakeTFSWrapper for tests.

**Tests (Detailed)**
- Python tests: used indirectly by `agent_v3_test`.
- Rust tests: `FakeTfsWrapper.list_saved_models` + `list_saved_models_status` correctness.

**Gaps / Notes**
- `TFS_BINARY` must be set; Python will crash if missing.
- Error_code mapping uses `e.code().value[0]` (non-obvious; must mirror).

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

### `monolith/agent_service/utils.py`
<a id="monolith-agent-service-utils-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: ~1000+
- Purpose/role: Core config + helper utilities for agent service, TF Serving configs, TensorProto creation, network utilities, and config file generation.
- Key symbols/classes/functions: `TfServingConfig`, `AgentConfig`, `DeployType`, `TFSServerType`, `RoughSortModel*`, `conf_parser`, `find_free_port`, `gen_model_spec`, `gen_model_config`, `make_tensor_proto`, `InstanceFormater`, `ZKPath`, `get_local_ip`, many helpers.
- External dependencies: `tensorflow`, `tensorflow_serving` protos, `protobuf.text_format`, `json`, `socket`, `os`.
- Side effects: overrides `os.path.isabs`; writes platform config files; reads/writes files; inspects env; opens sockets.

**Required Behavior (Detailed)**
- Module globals:
  - Defines `flags.DEFINE_string("conf", "", "agent conf file")`.
  - `TFS_HOME="/opt/tiger/monolith_serving"`.
  - `DEFAULT_PLATFORM_CONFIG_FILE = "{TFS_HOME}/conf/platform_config_file.cfg"`.
  - Overrides `os.path.isabs` to treat paths starting with `hdfs:/` as absolute.
  - `DEFAULT_MODEL_CONFIG` computed via `gen_model_config` with name `default` and base_path `${TFS_HOME}/dat/saved_models/entry`.
  - `FeatureKeys` set used by `InstanceFormater.from_dump`.
- Small helpers:
  - `conf_parser(file_name, args)`:
    - Ignores missing file.
    - Strips comments/blank lines.
    - Supports `include <path>` directive (recursive parse).
    - Parses `key=value` or `key value` via regex split `SEQ`.
    - If key repeats: first value becomes list; subsequent appended.
  - `find_free_port()` binds `('localhost', 0)` and returns ephemeral port.
  - `check_port_open(port)` connects to `127.0.0.1:port`; logs and returns bool.
  - `write_to_tmp_file(content)` writes `str(content)` to tempfile and returns path.
  - `replica_id_from_pod_name()`:
    - If `MY_POD_NAME` set: MD5 hash, take hex `[10:20]` slice, parse base16.
    - Else returns `-1`; any exception returns `-1`.
- Type helpers and enums:
  - `TFSServerType` constants: `ps`, `entry`, `dense`, `unified`.
  - `DeployType` wraps server type; validates type; equality works vs string or DeployType.
  - `DeployType.compat_server_type(server_type)`:
    - If `server_type` None or `mixed`:
      - If self is `mixed`, raises RuntimeError.
      - Else returns self type.
    - If self is `mixed`, returns server_type.
    - Else asserts equality and returns server_type.
  - Rough sort enums: `RoughSortModelLoadedServer` and `RoughSortModelPrefix`.
- `TfServingConfig` dataclass defaults:
  - `enable_batching=False`, `allow_version_labels_for_unavailable_models=False`, `batching_parameters_file=None`.
  - `num_load_threads=0`, `num_unload_threads=0`, `max_num_load_retries=5`, `load_retry_interval_micros=60*1000*1000`.
  - `file_system_poll_wait_seconds=1`, `flush_filesystem_caches=True`.
  - `tensorflow_session_parallelism=0`, `tensorflow_intra_op_parallelism=0`, `tensorflow_inter_op_parallelism=0`.
  - `ssl_config_file=None`, `platform_config_file=None`.
  - `per_process_gpu_memory_fraction=0`, `allow_growth=True`, `saved_model_tags=None`.
  - `grpc_channel_arguments=None`, `grpc_max_threads=0`, `enable_model_warmup=True`, `version=None`.
  - `remove_unused_fields_from_bundle_metagraph=True`, `enable_signature_method_name_check=False`.
  - `xla_cpu_compilation_enabled=False`, `enable_profiler=True`.
- `AgentConfig` dataclass defaults (in addition to `TfServingConfig`):
  - `bzid=None`, `base_name=None`, `base_path=None`, `num_ps=1`, `num_shard=None`, `deploy_type=None`.
  - `replica_id=None`, `stand_alone_serving=False`, `zk_servers=None`.
  - `proxy_port=None`, `tfs_entry_port=None`, `tfs_entry_http_port=None`, `tfs_entry_archon_port=None`.
  - `tfs_ps_port=None`, `tfs_ps_http_port=None`, `tfs_ps_archon_port=None`.
  - `dense_alone=False`, `dense_service_num=3`, `tfs_dense_port=None`, `tfs_dense_http_port=None`, `tfs_dense_archon_port=None`.
  - `agent_port=None`, `update_model_status_interval=1`, `model_config_file=None`, `agent_version=1`.
  - `max_waiting_sec=1200`, `preload_jemalloc=True`.
  - `version_policy='latest'`, `version_data=1`.
  - `fetch_ps_timeout_ms=200`, `fetch_ps_long_conn_num=100`, `fetch_ps_long_conn_enable=True`, `fetch_ps_retry=2`, `aio_thread_num=30`.
  - `file_system_poll_wait_seconds_ps=0`.
  - Rough sort: `rough_sort_model_name=None`, `rough_sort_model_local_path=None`, `rough_sort_model_loaded_server=entry`, `rough_sort_model_p2p_path=None`, `rough_sort_resource_constrained=False`.
  - `dc_aware=False`.
  - Unified: `layout_pattern=None`, `layout_filters=None`, `tfs_port_archon=None`, `tfs_port_grpc=None`, `tfs_port_http=None`, `use_metrics=True`.
- `AgentConfig.__post_init__`:
  - Updates `zk_servers` via `_update_zk_servers(self.zk_servers, is_ipv6_only())`.
  - If `stand_alone_serving`: `deploy_type = DeployType(mixed)`; else requires `deploy_type` and wraps in `DeployType`.
  - `num_shard` defaults to `num_tce_shard`; otherwise asserts equal to `num_tce_shard`.
  - Port allocation (uses `find_free_port` and env overrides):
    - Mixed: proxy + entry/ps ports from `PORT`, `PORT3..PORT7`; dense ports from `PORT8..PORT10` if `dense_alone` and `DENSE_SERVICE_IDX==0`, else free ports.
    - Entry: proxy + ps ports free; entry ports from `PORT`, `PORT3`, `PORT4`; optional dense free ports.
    - PS: proxy + entry ports free; ps ports from `PORT`, `PORT3`, `PORT4`; optional dense free ports.
    - Dense: requires `dense_alone=True`; entry/ps ports free; dense ports from `PORT`, `PORT3`, `PORT4` if `DENSE_SERVICE_IDX==0`, else free.
    - Unified: `tfs_port_archon/ grpc / http` from `PORT`, `PORT3`, `PORT4`.
  - `agent_port` from `PORT2` or free.
  - `replica_id`: if `agent_version==1`, use `replica_id_from_pod_name`; else env `REPLICA_ID` or fallback to pod hash.
  - If `platform_config_file` unset, use `DEFAULT_PLATFORM_CONFIG_FILE`.
  - Calls `generate_platform_config_file()`.
- `generate_platform_config_file()`:
  - Builds `ConfigProto` with `intra_op_parallelism_threads` and `inter_op_parallelism_threads`:
    - Use configured values or `MY_CPU_LIMIT` or default `16`.
  - `allow_soft_placement=True`, `gpu_options.allow_growth=allow_growth`.
  - If `dense_alone` and `enable_batching`:
    - Build `BatchingParameters` with `max_batch_size=1024`, `batch_timeout_micros=800`, `max_enqueued_batches=100000`, `num_batch_threads=8`, `support_diff_dim_size_inputs=True`.
    - Build `SessionBundleConfig` with session+batching.
  - Else `SessionBundleConfig` with session only.
  - Set `enable_model_warmup`.
  - Wrap in `SavedModelBundleSourceAdapterConfig`, pack into `PlatformConfigMap` under key `tensorflow`.
  - Serialize to text and write to `platform_config_file`.
  - On exception: logs and attempts to remove file if it exists.
- AgentConfig properties:
  - `num_tce_shard`: `HOST_SHARD_ENV` or `1`.
  - `shard_id`: env `SHARD_ID` or `-1`.
  - `idc`: env `TCE_INTERNAL_IDC` lowercased.
  - `cluster`: env `TCE_LOGICAL_CLUSTER` or `TCE_CLUSTER` or `TCE_PHYSICAL_CLUSTER` lowercased.
  - `location`: `"{idc}:{cluster}"` if both; else None.
  - `path_prefix`: `/bzid/service/base_name[/idc:cluster]` if `dc_aware` else without location.
  - `layout_path`: if `layout_pattern` absolute, use it; else `/{bzid}/layouts/{layout_pattern}`.
  - `container_cluster`: `"{TCE_PSM};{idc};{cluster}"`.
  - `container_id`: env `MY_POD_NAME` or `get_local_ip()`.
- Command helpers:
  - `get_cmd_and_port(binary, server_type=None, config_file=None)`:
    - Normalizes `server_type` via `compat_server_type`.
    - If `config_file` None, generates model server config and writes to temp file.
    - Adds flags: `--model_config_file`, archon psm/cluster, metrics prefix, log_conf.
    - For mixed and non-entry: suffix psm with `_ps`/`_dense`, change log conf.
    - Adds port/rest/archon flags per server type.
    - ENTRY/DENSE add `archon_entry_to_ps_*` flags and `archon_async_dispatcher_threads`.
    - DENSE adds `--enable_batching=true` if enabled.
    - `agent_version != 1` adds `--model_config_file_poll_wait_seconds=0`.
    - Iterates `TfServingConfig` fields: if value differs from default, emit flag; special case `file_system_poll_wait_seconds`:
      - If `agent_version==1` and server_type==PS, use `file_system_poll_wait_seconds_ps`.
      - Else use configured value if not default.
    - Returns command string and chosen gRPC port.
  - `get_cmd` returns command only.
  - `get_server_schedule_iter(server_type)`:
    - Mixed/PS: for PS yields indices where `i % num_shard == shard_id`; else yields None.
    - Dense: if server_type dense, yields `replica_id` (commented NOTE about bug).
    - Else yields None.
- Model config generation:
  - `_gen_model_server_config(server_type)`:
    - Uses `version_policy` and `version_data`, `compat_server_type`.
    - PS: create config per schedule index `ps_i`, base_path = `base_path/ps_i`.
      - If `rough_sort_model_name` and loaded_server=PS: add `ps_item_embedding_i` with base_path `${rough_sort_model_local_path}/${rough_sort_model_name}/${name}`.
    - Entry/Dense: add `entry` or `dense_0` model with base_path `${base_path}/{name}`.
      - If `rough_sort_resource_constrained` and loaded_server=ENTRY: add `entry_item_embedding_0` with base_path `${base_path}/{name}`.
      - Else if `rough_sort_model_name` and loaded_server in ENTRY/DENSE: add `{entry|dense}_item_embedding_0` with base_path `${rough_sort_model_local_path}/${rough_sort_model_name}/${name}`.
- Config parsing:
  - `AgentConfig.from_file(fname)`:
    - Calls `conf_parser` to build `kwarg`.
    - For each annotated field: convert strings to bool/int/float/str/List; `str == "none"` -> `None`; `int/float` uses `eval`.
    - If missing `deploy_type`, uses legacy `server_type`.
    - Returns `AgentConfig(**args)` (triggers `__post_init__`).
  - `_update_zk_servers(zk_servers, use_ipv6)`:
    - If `use_ipv6` and `zk_servers` contains IPv4 hosts:
      - If equals `default_zk_servers_ipv4` (constructed from `_HOSTS` + `_PORT`), return `default_zk_servers(True)` with warning.
      - Else raise exception.
    - Else returns original `zk_servers`.
- ZK path parser:
  - `ZKPath.PAT` regex supports `/bzid/service/base[/idc:cluster]/server_type:index[/replica_id]`.
  - `__init__`: if `path is None or len(path) != 0` then attempts match; sets `_group_dict` or logs "path not matched".
  - `__getattr__` returns captured group or None.
  - `task`: `"server_type:index"` if both present.
  - `location`: `"idc:cluster"` if both present.
  - `ship_in(idc, cluster)`: if either arg None, returns True; else compares to parsed idc/cluster.
- Pattern helpers:
  - `parse_pattern` splits string by `{}` (or `[]` in `expand_pattern`) and combines via `comb_fn`.
  - `normalize_regex` replaces `{name}` with named capture group `(?P<name>\\d+)`.
  - `expand_pattern` expands bracket ranges like `[1-3]` or `[1,2]` into list; uses `range(int(s), int(e))` for dash.
- Model spec/config helpers:
  - `gen_model_spec(name, version=None, signature_name=None)`:
    - Sets `version.value` if int; else `version_label` if str.
    - Adds `signature_name` if provided.
  - `gen_model_config(name, base_path, version_policy='latest', version_data=1, model_platform='tensorflow', version_labels=None)`:
    - `latest`, `latest_once` set `num_versions`.
    - `all` sets `ServableVersionPolicy.All()`.
    - `specific` accepts int or list and extends versions.
    - Else raises ValueError.
    - `version_labels` applied if provided.
  - `gen_status_proto` and `gen_model_version_status` wrap TF Serving status proto fields.
  - `make_tensor_proto(instances)` builds DT_STRING TensorProto with `dim.size=len(instances)` and `string_val` set.
- InstanceFormater:
  - `to_tensor_proto(batch_size)` repeats serialized instance `batch_size` times.
  - `to_pb(fname=None)` writes serialized bytes to temp file or given path.
  - `to_json(fname=None)` uses `MessageToJson` and `write_to_tmp_file` or writes to `fname`.
  - `to_pb_text(fname=None)` writes text format via `write_to_tmp_file` or to `fname`.
  - `from_json(fname)` reads JSON and `ParseDict` to Instance.
  - `from_pb_text(fname)` reads file, strips lines, `text_format.Parse`.
  - `from_dump(fname)`:
    - Parses key/value lines into nested dict/list structure (`kwargs`) with `stack` control.
    - Interprets numeric keys as list indices.
    - Handles feature-level merging using `FeatureKeys` and stack lookback.
    - Converts numeric values to int.
    - Parses resulting dict into Instance via `ParseDict`.
- Misc:
  - `pasre_sub_model_name(sub_model_name)` (typo in name) splits by `_`, returns `(lower, index)` with default `0`.
  - `get_local_ip()`:
    - First tries `MY_HOST_IP` or `socket.gethostbyname(hostname)`.
    - Falls back to UDP socket connect to `8.8.8.8:80` to infer local IP.
    - Returns `'localhost'` if no usable IP found.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/src/config.rs` + new `utils.rs`.
- Rust public API surface: `AgentConfig` struct + helpers for model spec and TensorProto assembly.
- Data model mapping: use `monolith_proto::tensorflow_serving::apis` for ModelSpec and `tensorflow_core::TensorProto`.

**Implementation Steps (Detailed)**
1. Port `AgentConfig` with all fields + defaults + env overrides.
2. Recreate port allocation logic, deploy type handling, and platform config file generation.
3. Port `gen_model_spec` and `gen_model_config` helpers with identical proto fields.
4. Implement `make_tensor_proto` for DT_STRING using TF Serving proto types.
5. Port network/IP helper methods (`get_local_ip`, `find_free_port`, etc.).
6. Mirror all file I/O (platform config) and text_format behavior.

**Tests (Detailed)**
- Python tests: `monolith/agent_service/utils_test.py`
- Rust tests: add unit tests for config defaults, model spec generation, and TensorProto creation.

**Gaps / Notes**
- This file is high-risk; many behaviors are implicit and must be traced manually.

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

### `monolith/agent_service/utils_test.py`
<a id="monolith-agent-service-utils-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 170
- Purpose/role: Tests utility helpers (model spec/config, status proto, AgentConfig parsing, instance parsing, ZKPath parsing).
- Key symbols/classes/functions: `ServingUtilsTest` methods.
- External dependencies: `monolith.agent_service.utils`.
- Side effects: reads config and test data files.

**Required Behavior (Detailed)**
- `setUpClass`: sets `MY_HOST_IP=127.0.0.1`.
- `test_gen_model_spec`:
  - `gen_model_spec('model', 1, 'predict')` -> name, version.value, signature_name set.
- `test_gen_model_config`:
  - `gen_model_config` with `version_data=2` and version_labels.
  - Asserts `name`, `base_path`, and `latest.num_versions == 2`.
- `test_gen_status_proto`:
  - `gen_status_proto(ErrorCode.CANCELLED, 'CANCELLED')` sets fields.
- `test_gen_model_version_status`:
  - `gen_model_version_status(version=1, state=START, error_code=NOT_FOUND, error_message="NOT_FOUND")`.
  - Asserts version and state match.
- `test_gen_from_file`:
  - `AgentConfig.from_file('monolith/agent_service/agent.conf')` sets `stand_alone_serving=True`.
- `test_list_field`:
  - Same config file; asserts `layout_filters == ['ps_0', 'ps_1']`.
- `test_instance_wrapper_from_json`:
  - `InstanceFormater.from_json('monolith/agent_service/test_data/inst.json')`.
  - `to_tensor_proto(5)` yields dtype 7, dim[0].size 5.
- `test_instance_wrapper_from_pbtext`:
  - `from_pb_text('monolith/agent_service/test_data/inst.pbtext')`; same tensor checks.
- `test_instance_wrapper_from_dump`:
  - `from_dump('monolith/agent_service/test_data/inst.dump')`; same tensor checks.
- `test_get_cmd_and_port`:
  - `AgentConfig.from_file(...); conf.agent_version=2; conf.get_cmd_and_port(binary='tensorflow_model_server', server_type='ps')`.
  - Asserts `'model_config_file_poll_wait_seconds'` is in cmd string.
- `ZKPath` tests:
  - Full dc-aware: `/bzid/service/base_name/idc:cluster/server_type:0/1` -> bzid/base_name/idc/cluster/server_type/index/replica_id; `location='idc:cluster'`, `task='server_type:0'`, `ship_in(None,None)` True.
  - Partial dc-aware: `/bzid/service/base_name/idc:cluster/server_type:0` -> replica_id None, `ship_in('idc','cluster')` True.
  - Old full: `/bzid/service/base_name/server_type:0/1` -> idc/cluster None, replica_id `1`, `ship_in(None,None)` True.
  - Old partial: `/bzid/service/base_name/server_type:0` -> replica_id None.
  - Old partial 2: `/1_20001223_.../service/20001223_.../ps:1` -> bzid/base_name parsed, server_type `ps`, index `1`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/tests/utils.rs`.
- Rust public API surface: utils module equivalents in Rust.

**Implementation Steps (Detailed)**
1. Port utilities and add tests for every helper above.
2. Ensure AgentConfig parsing matches Python (including list parsing for layout filters).
3. Port InstanceFormater-like parsing for JSON/pbtext/dump inputs.
4. Implement ZKPath parser with dc-aware and legacy formats.

**Tests (Detailed)**
- Python tests: this file
- Rust tests: parity tests with same fixtures.

**Gaps / Notes**
- InstanceFormater depends on test_data fixtures; ensure Rust can read them.

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

### `monolith/agent_service/zk_mirror.py`
<a id="monolith-agent-service-zk-mirror-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 672
- Purpose/role: ZK mirror/cache with watches for portal/publish/resource/service paths; coordinates scheduling and replica updates.
- Key symbols/classes/functions: `ZKMirror` and methods (watch_portal, watch_publish, expected_loading, update_service, election).
- External dependencies: `MonolithKazooClient`, `kazoo` watchers/election, `PublishMeta`, `ReplicaMeta`, `ResourceSpec`.
- Side effects: ZK watch registration, ZK CRUD, background threads, queue events.

**Required Behavior (Detailed)**
- Core state:
  - `_data` in-memory cache of `path -> bytes`.
  - Base paths: `/bzid/resource`, `/bzid/portal`, `/bzid/publish`, `/bzid/service`, `/bzid/locks`, `/bzid/election`.
  - `_local_host = get_local_ip()`, `_deploy_type` set from ctor.
- CRUD:
  - `create`: tries ZK create; on `NodeExistsError` sets value; rethrows other exceptions.
  - `set`: `zk.set` with retry; on `NoNodeError` creates with `makepath=True`.
  - `exists`: uses `zk.exists` (bool or stat); on `ZookeeperError` falls back to `_data`.
  - `delete`: retries delete; on `NotEmptyError` deletes recursively; on `NoNodeError` logs.
  - `get`: returns cached bytes; `get_children` returns child names by prefix (no dedupe).
- Resource/reporting:
  - `report_resource` writes `ResourceSpec` (ephemeral) under `/resource/{shard}:{replica}`.
  - `resources` property deserializes all cached resource nodes.
  - `tce_replica_id` from env `REPLICA_ID` or `replica_id_from_pod_name()`.
  - `num_tce_replica` intends to wait for all shards to report; implementation currently buggy (inner helper does not return).
- Publish/loading:
  - `publish_loadding` (typo): creates publish nodes if cache differs; supports list or single.
  - `expected_loading`:
    - Reads publish nodes; counts per model; keeps PublishMeta with **fewest** sub_models.
    - Selects model when count == `total_publish_num`.
    - If local shard+replica match: use pm as-is.
    - Else if same shard and `not is_spec`: override `replica_id` to local.
    - Else: override shard_id/replica_id to local and **filter sub_models to entry only**.
    - Skips non-LOAD ptypes.
  - `get_published_path` returns publish paths ending with model_name.
- Service updates/query:
  - `update_service(replicas)`:
    - Computes desired local paths via `ReplicaMeta.get_path`.
    - Deletes local replicas not in desired set.
    - Creates/updates remaining (ephemeral) when cache is missing or different.
  - `local_replica_paths`: cached service nodes whose host matches `_local_host` and `replica == tce_replica_id`.
  - `get_all_replicas(server_type)`: returns `{model:server_type:task -> [ReplicaMeta]}` for AVAILABLE only.
  - `get_model_replicas(model_name, server_type)`: returns `{model:task -> [ReplicaMeta]}` for AVAILABLE only.
  - `get_task_replicas(model_name, server_type, task)` returns list of AVAILABLE replicas.
  - `get_replica` returns AVAILABLE replica or None.
- Watches:
  - `watch_portal`:
    - Ensures portal/publish consistency; deletes publish entries not in portal.
    - Installs DataWatch per portal node; on CREATED/DELETED emits `Event(PORTAL)` with ModelMeta/action.
  - `watch_publish`:
    - DataWatch per publish node; updates cache on CREATED/DELETED.
    - When count is 0 or == total_publish_num, emits `Event(PUBLISH)`.
  - `watch_resource`:
    - Watches resource nodes; updates cache on CREATED/DELETED/CHANGED.
  - `watch_service`:
    - Watches model -> task -> replica hierarchy; caches data for each replica node.
- Election:
  - `election(leader, sched, identifier)` no-ops for ENTRY deploy.
  - Uses Kazoo election; on `ConnectionClosedError` with disconnected state: clears cache/queue and restarts.
- `start(is_client=False)`:
  - Starts ZK, watches service, and if not client also watches publish.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/src/zk_mirror.rs`.
- Rust public API surface: `ZKMirror` with same watch APIs and cache.
- Data model mapping: `PublishMeta`, `ResourceSpec`, `ReplicaMeta`, `Event`.

**Implementation Steps (Detailed)**
1. Implement ZK cache and watch registration with thread-safe locks.
2. Port expected_loading logic (publish count and shard/replica overrides).
3. Port update_service and replica query helpers.
4. Implement leader election and reconnect behavior.
5. Provide Queue/Event mechanism for scheduler integration.

**Tests (Detailed)**
- Python tests: `monolith/agent_service/zk_mirror_test.py`
- Rust tests: use Fake ZK to validate watches, publish flow, and replica queries.

**Gaps / Notes**
- Requires a robust fake ZK in Rust to test watch behavior.

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

### `monolith/agent_service/zk_mirror_test.py`
<a id="monolith-agent-service-zk-mirror-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 229
- Purpose/role: End-to-end test of ZKMirror portal/publish/service/resource flows using FakeKazooClient.
- Key symbols/classes/functions: `ZKMirrorTest.test_crud`, `.test_zk_mirror`
- External dependencies: FakeZK, FakeTFServing, data_def types.
- Side effects: ZK node creation, event queue handling.

**Required Behavior (Detailed)**
- `setUpClass`:
  - Set `HOST_SHARD_ENV=10`, `SHARD_ID=2`, `REPLICA_ID=2`.
  - Create `ZKMirror(zk=FakeKazooClient(), bzid='bzid', queue=Queue(), tce_shard_id=2, num_tce_shard=10)` and start.
  - Build `ResourceSpec(address='{local_ip}:1234', shard_id=2, replica_id=2, memory=12345, cpu=5.6, network=3.2, work_load=0.7)`.
- `test_crud`:
  - `ensure_path('/model/crud')`, `exists` True.
  - `create('/model/crud/data', b'test', makepath=True)`.
  - `get/set` via underlying `_zk` to verify value change.
  - `delete('/model/crud', recursive=False)` then `exists` False.
  - Assert `num_tce_shard==10`, `tce_replica_id==2`, `tce_shard_id==2`.
- `test_zk_mirror`:
  - Step0: `watch_portal()` and `watch_resource()`, then create portal node for `ModelMeta(model_name, model_dir, num_shard=5)`.
  - Step1: dequeue `Event` from `queue`, assert `etype==PORTAL`, path matches, deserialize `ModelMeta`.
  - Build scheduler PublishMeta list:
    - `version=123456`, `num_ps=10`, `NUM_REPLICAS=3`.
    - For each `i in range(mm.num_shard)`:
      - `sub_models`: `ps_k` for `k % mm.num_shard == i`, plus `entry`.
      - Choose `shard_id`: `self.shard_id` for `i==0`, else pop from shuffled shard list (avoiding current shard).
      - For each replica_id 0..2, create `PublishMeta(shard_id, replica_id, model_name, num_ps, sub_models)`.
    - Set `pm.total_publish_num = len(pms)` for all; call `publish_loadding(pms)`.
  - Step2: `expected_loading()` returns map where:
    - model_name == `MODEL_NAME`, `pm.shard_id == self.shard_id`, and `'entry'` in `pm.sub_models`.
  - Step3: `update_service`:
    - For each sub_model in expected `pm`, build `ReplicaMeta(address='{local_ip}:8080', model_name, server_type, task, replica=2, stat=AVAILABLE)`.
    - Call `update_service(replicas)`.
  - Step4: replica query checks:
    - Build `entry_replica`, `ps0_replica`, `ps5_replica` with `stat=30`.
    - Validate `get_all_replicas('ps')`, `get_model_replicas('entry')`, `get_task_replicas('ps',0)`, `get_replica('ps',5,2)`.
    - `local_replica_paths` equals set of three expected paths.
  - Step5/6: `report_resource(resource)` then `resources[0] == resource`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/tests/zk_mirror.rs`.
- Rust public API surface: ZKMirror with fake ZK.

**Implementation Steps (Detailed)**
1. Port fake ZK + ZKMirror to Rust.
2. Implement scheduler simulation and verify queue events.
3. Validate replica query helpers and resource roundtrip.

**Tests (Detailed)**
- Python tests: this file
- Rust tests: direct parity tests.

**Gaps / Notes**
- Requires deterministic ordering of publish nodes and queue events.

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

### `monolith/base_runner.py`
<a id="monolith-base-runner-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 46
- Purpose/role: Base runner class with TensorFlow summary writing helper.
- Key symbols/classes/functions: `BaseRunner`, `write_summary`.
- External dependencies: `tensorflow`.
- Side effects: writes TensorFlow summary events to a writer.

**Required Behavior (Detailed)**
- `BaseRunner.__init__` accepts `*args, **kwargs` (no stored fields in base).
- `run()` is abstract and raises `NotImplementedError`.
- `write_summary(logs, summary_writer, current_step)`:
  - Creates a new TF v1 Graph context.
  - Builds `tf.compat.v1.Summary.Value` entries from `logs` dict.
  - Writes to `summary_writer.add_summary(tf_summary, current_step)`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/base_runner.rs`.
- Rust public API surface: `BaseRunner` trait/struct with `run` and `write_summary` helper.
- Data model mapping: summary writer abstraction (likely TB event writer).

**Implementation Steps (Detailed)**
1. Implement a minimal runner trait with `run()`.
2. Provide a summary writer abstraction compatible with TF event files (or document absence).
3. Mirror log-to-summary conversion semantics.

**Tests (Detailed)**
- Python tests: none specific.
- Rust tests: unit test for summary serialization if supported.

**Gaps / Notes**
- Rust likely needs a TensorBoard writer crate to match TF summary outputs.

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

### `monolith/common/python/mem_profiling.py`
<a id="monolith-common-python-mem-profiling-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 51
- Purpose/role: Configure tcmalloc heap profiling via environment variables.
- Key symbols/classes/functions: `enable_tcmalloc`, `setup_heap_profile`.
- External dependencies: `MLPEnv`, `monolith.utils`.
- Side effects: modifies `LD_PRELOAD` and `HEAP_PROFILE_*` env vars.

**Required Behavior (Detailed)**
- `enable_tcmalloc()`:
  - Appends `../gperftools/libtcmalloc/lib/libtcmalloc.so` (resolved via `utils.get_libops_path`) to `LD_PRELOAD`.
- `setup_heap_profile(...)`:
  - Calls `enable_tcmalloc()`.
  - Uses `MLPEnv().index` to name heap profile file `hprof_<index>` in `heap_pro_file` or `utils.find_main()`.
  - Sets env vars:
    - `HEAPPROFILE` path
    - `HEAP_PROFILE_INUSE_INTERVAL` and `HEAP_PROFILE_ALLOCATION_INTERVAL` scaled by `1/sample_ratio`
    - `HEAP_PROFILE_SAMPLE_RATIO`, `HEAP_PROFILE_TIME_INTERVAL`, `HEAP_PROFILE_MMAP` (lowercased bool).
  - Defaults: `heap_profile_inuse_interval=104857600`, `heap_profile_allocation_interval=1073741824`, `heap_profile_time_interval=0`, `sample_ratio=1.0`, `heap_profile_mmap=False`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/mem_profiling.rs`.
- Rust public API surface: functions to set environment variables and preload tcmalloc.

**Implementation Steps (Detailed)**
1. Implement environment variable setup with the same names and scaling.
2. Provide an MLPEnv equivalent or configuration injection for index.
3. Ensure LD_PRELOAD modification is appended (not overwritten).

**Tests (Detailed)**
- Python tests: none
- Rust tests: unit tests for env var values given sample_ratio.

**Gaps / Notes**
- Rust cannot preload shared libs at runtime on all platforms; document OS-specific behavior.

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

### `monolith/core/__init__.py`
<a id="monolith-core-init-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 0
- Purpose/role: Empty initializer for `monolith.core`.
- Key symbols/classes/functions: none
- External dependencies: none
- Side effects: none

**Required Behavior (Detailed)**
- Importing this module must not execute any side effects.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/src/lib.rs`.
- Rust public API surface: module boundary only.

**Implementation Steps (Detailed)**
1. Ensure the Rust crate exposes `monolith-core` without implicit side effects.

**Tests (Detailed)**
- Python tests: none
- Rust tests: none

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

### `monolith/core/auto_checkpoint_feed_hook.py`
<a id="monolith-core-auto-checkpoint-feed-hook-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 376
- Purpose/role: TPU infeed/outfeed SessionRunHook with thread-managed queues, TPU init/shutdown, and end-of-stream stopping signals.
- Key symbols/classes/functions: `PeriodicLogger`, `_SIGNAL`, `_OpQueueContext`, `_OpSignalOnceQueueContext`, `TPUInfeedOutfeedSessionWithEndOfStreamHandlingHook`.
- External dependencies: `threading`, `time`, `os`, `six.moves.queue/xrange`, `tensorflow.compat.v1`, `config_pb2`, `tpu_compilation_result`, `summary_ops_v2` (plus implicit `ops`, `tpu_config`, `session_support`).
- Side effects: spawns threads, initializes/shuts down TPU, runs TF sessions, reads env vars, logs via TF logging.

**Required Behavior (Detailed)**
- `PeriodicLogger(seconds)`:
  - `log()` emits TF log only if elapsed time exceeds `seconds`.
- `_SIGNAL`:
  - `NEXT_BATCH = -1`, `STOP = -2` (negative values reserved for control).
- `_OpQueueContext`:
  - Starts a daemon thread running `target(self, *args)`.
  - `send_next_batch_signal(iterations)` enqueues integer iterations.
  - `read_iteration_counts()` yields iterations until `_SIGNAL.STOP` then returns.
  - `join()` logs, sends STOP, joins thread.
- `_OpSignalOnceQueueContext`:
  - Only allows the first `send_next_batch_signal`; subsequent calls ignored.
- `TPUInfeedOutfeedSessionWithEndOfStreamHandlingHook`:
  - `__init__`:
    - Stores enqueue/dequeue ops, rendezvous, master, session config, init ops, outfeed cadence.
    - Reads embedding config from `ctx` if present.
    - Sets `_should_initialize_tpu=False` when model-parallel + per-host input broadcast; else true.
    - Sets `stopping_signal=False`.
  - `_create_or_get_iterations_per_loop()`:
    - Uses graph collection `tpu_estimator_iterations_per_loop`.
    - If >1 existing var → `RuntimeError("Multiple iterations_per_loop_var in collection.")`.
    - Else creates resource variable in scope `tpu_estimator`, int32 scalar, non-trainable, in LOCAL_VARIABLES, colocated with global step.
  - `begin()`:
    - Records `_iterations_per_loop_var`.
    - Adds TPU shutdown op to `_finalize_ops` if `_should_initialize_tpu`.
    - Adds summary writer init ops to `_init_ops` and flush ops to `_finalize_ops`.
  - `_run_infeed(queue_ctx, session)`:
    - Optional sleep (`initial_infeed_sleep_secs`).
    - If `run_infeed_loop_on_coordinator`: for each iteration signal, runs `enqueue_ops` `steps` times.
    - Else runs `enqueue_ops` once per signal.
  - `_run_outfeed(queue_ctx, session)`:
    - Runs `dequeue_ops` every `outfeed_every_n_steps`.
    - If output includes `_USER_PROVIDED_SIGNAL_NAME`, expects a dict containing `stopping`.
    - When first stopping signal seen, sets `stopping_signals=True` and later flips `self.stopping_signal=True`.
  - `_assertCompilationSucceeded(result, coord)`:
    - Parses TPU compilation proto; if `status_error_message` set → log error + `coord.request_stop()`.
  - `after_create_session()`:
    - If `_should_initialize_tpu`, runs `tf.tpu.initialize_system` in a fresh graph/session.
    - Runs `_init_ops` with 30 minute timeout.
    - If `TPU_SPLIT_COMPILE_AND_EXECUTE=1`, runs `tpu_compile_op` and asserts compilation success.
    - Starts infeed/outfeed controller threads.
    - If `TF_TPU_WATCHDOG_TIMEOUT>0`, starts worker watchdog.
  - `before_run()`:
    - If `stopping_signal` is set, raises `tf.errors.OutOfRangeError`.
    - Reads `iterations_per_loop`, sends signals to infeed/outfeed controllers.
  - `end()`:
    - Joins infeed/outfeed threads; calls rendezvous `record_done`; runs finalize ops (flush + shutdown).
  - `get_stopping_signals_and_name(features)`:
    - If `_USER_PROVIDED_SIGNAL_NAME` in `features`, uses `tf.tpu.cross_replica_sum` to compute `stopping` boolean.
    - Returns `(stopping_signals, _USER_PROVIDED_SIGNAL_NAME)`.
- Env vars:
  - `TPU_SPLIT_COMPILE_AND_EXECUTE=1` triggers separate compile step.
  - `TF_TPU_WATCHDOG_TIMEOUT` enables worker watchdog.
- Threading: two daemon threads (infeed/outfeed) controlled via queues.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-tf/src` (TPU runtime integration).
- Rust public API surface: no equivalent; would need a TPU session hook abstraction.
- Data model mapping: TF session/run hooks have no direct Rust equivalent.
- Feature gating: only relevant if TF-runtime backend is enabled.
- Integration points: training loop / input pipeline in `monolith-training`.

**Implementation Steps (Detailed)**
1. Decide if TPU SessionRunHook is in-scope for Rust TF backend.
2. If in-scope, implement infeed/outfeed controller threads and queue signaling.
3. Expose env-var toggles for compile/execute split and watchdog timeout.
4. Provide cross-replica stopping signal handling for outfeed.

**Tests (Detailed)**
- Python tests: none dedicated (covered indirectly by TPU training flows).
- Rust tests: none yet; would need integration tests with TF runtime.
- Cross-language parity test: not applicable unless TF runtime implemented in Rust.

**Gaps / Notes**
- Missing imports in Python (`ops`, `tpu_config`, `session_support`) are referenced but not imported.
- Rust has no TPU SessionRunHook analog; full parity requires TF runtime + hook infrastructure.

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

### `monolith/core/base_embedding_host_call.py`
<a id="monolith-core-base-embedding-host-call-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 643
- Purpose/role: Host call logic for embedding tasks, including TPU variable caching and DeepInsight metrics.
- Key symbols/classes/functions: `BaseEmbeddingHostCall`, `update_tpu_variables_ops`, `generate_host_call_hook`.
- External dependencies: `tensorflow.compat.v1`, `tensorflow`, `BaseHostCall`, `ReplicatedVariable`.
- Side effects: creates TPU variables on all replicas; writes TF summaries; computes AUC.

**Required Behavior (Detailed)**
- Constants define metric names and TPU variable names.
- `TPUVariableRestoreHook` runs assign op after session creation.
- `BaseEmbeddingHostCall.__init__`:
  - Stores flags for host call, deepinsight, scalar metrics, caching mode.
  - Extracts `context` unless `cpu_test`.
  - Creates TPU variables if `enable_host_call` and caching enabled.
- `_create_tpu_var`:
  - Creates per-replica TPU variables across hosts with zeros, adds to `TPU_VAR` collection.
  - Wraps in `ReplicatedVariable`; registers restore hooks.
- `_compute_new_value(base, delta, update_offset)`:
  - Pads `delta` to base length, then `tf.roll` by offset, then adds to base.
- `update_tpu_variables_ops(...)`:
  - Clears TPU vars when `global_step % host_call_every_n_steps == 1`.
  - Writes labels/preds/uid_buckets and req_times/sample_rates into TPU vars at computed offsets.
  - Updates accumulated counter only after tensor updates.
- `record_summary_tpu_variables()` collects TPU vars into host call tensors.
- `record_summary_tensor` filters based on `enable_host_call_scalar_metrics` and deprecated metric names.
- `_write_summary_ops`:
  - Filters by uid_bucket and stopping_signals; computes AUC; writes scalar summaries and serialized tensors.
- `generate_host_call_hook()`:
  - Calls `compress_tensors()` and returns either `_host_call` or `_host_call_with_tpu` plus tensor list; or `None` if disabled.
  - Host call writes summaries under `output_dir/host_call` with `tf2.summary`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/src/base_embedding_host_call.rs`.
- Rust public API surface: host call builder with optional TPU caching mode.
- Data model mapping: summary writer and AUC metric implementation.

**Implementation Steps (Detailed)**
1. Recreate TPU variable caching logic (or provide equivalent buffer) with host-call step windowing.
2. Implement summary writing and AUC computation matching TF semantics.
3. Mirror filtering by UID buckets and stopping signals.
4. Provide compatibility with TPU/CPU test modes.

**Tests (Detailed)**
- Python tests: `monolith/core/base_embedding_host_call_test.py`
- Rust tests: unit tests for `_compute_new_value` and host call output shapes.

**Gaps / Notes**
- Full parity requires TF TPU runtime; Rust may need to stub or bridge this functionality.

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

### `monolith/core/base_embedding_host_call_test.py`
<a id="monolith-core-base-embedding-host-call-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 77
- Purpose/role: Tests `_compute_new_value` in BaseEmbeddingHostCall.
- Key symbols/classes/functions: `BaseEmbeddingHostCallTest.test_compute_new_value`.
- External dependencies: `tensorflow.compat.v1`.
- Side effects: none.

**Required Behavior (Detailed)**
- Disables eager execution (`tf.disable_eager_execution()`).
- `test_compute_new_value`:
  - Creates `global_step = tf.train.get_or_create_global_step()` (unused).
  - Builds `params = {enable_host_call=False, context=None, cpu_test=False, host_call_every_n_steps=100}`.
  - Instantiates `BaseEmbeddingHostCall("", False, False, False, False, 10, params)`.
  - `base_value = zeros([10], int32)`, `delta_value = ones([2], int32)`.
  - Offset=1: `_compute_new_value(base, delta, 1)` -> `[0,1,1,0,0,0,0,0,0,0]`.
  - Offset=5: `_compute_new_value(prev, delta, 5)` -> `[0,1,1,0,0,1,1,0,0,0]`.
  - Offset=6: `_compute_new_value(prev, delta, 6)` -> `[0,1,1,0,0,1,2,1,0,0]`.
  - Each expected tensor verified via `tf.reduce_all(tf.math.equal(...))` inside new `tf.Session()`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/tests/base_embedding_host_call.rs`.
- Rust public API surface: `_compute_new_value` equivalent.

**Implementation Steps (Detailed)**
1. Port `_compute_new_value` logic in Rust and add unit tests for offsets.

**Tests (Detailed)**
- Python tests: this file
- Rust tests: same assertions.

**Gaps / Notes**
- Needs tensor ops; may require a small tensor wrapper or TF binding in Rust tests.

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

### `monolith/core/base_embedding_task.py`
<a id="monolith-core-base-embedding-task-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 611
- Purpose/role: Base class for TPU embedding tasks; builds input pipeline, vocab sizing, and TPU embedding configs.
- Key symbols/classes/functions: `BaseEmbeddingTask.params`, `create_input_fn`, `_post_process_example`, `create_feature_and_table_config_dict`, `process_features_for_cpu_test`.
- External dependencies: `tensorflow.compat.v1`, `tpu_embedding`, `FeatureSlot/FeatureColumn`, `auto_checkpoint_feed_hook`, `util`.
- Side effects: reads vocab file (possibly from HDFS), constructs TF datasets and embedding configs.

**Required Behavior (Detailed)**
- `params()`:
  - Defines embedding params:
    - `vocab_size_per_slot=None`, `custom_vocab_size_mapping=None`, `vocab_size_offset=None`.
    - `qr_multi_hashing=False`, `qr_hashing_threshold=100000000`, `qr_collision_rate=4`.
    - `vocab_file_path=None`, `enable_deepinsight=False`.
    - `enable_host_call_scalar_metrics=False`, `enable_host_call_norm_metrics=False`.
    - `files_interleave_cycle_length=4`, `deterministic=False`.
    - `gradient_multiplier=1.0`, `enable_caching_with_tpu_var_mode=False`.
    - `top_k_sampling_num_per_core=6`, `use_random_init_embedding_for_oov=False`.
    - `merge_vector=False`.
  - Defines training params:
    - `train.file_folder=None`, `train.date_and_file_name_format="*/*/part*"`.
    - `train.start_date=None`, `train.end_date=None`.
    - `train.vocab_file_folder_prefix=None`.
- `__init__(params)`:
  - Calls `super().__init__`, stores `self.p`.
  - Sets `_enable_deepinsight`, `_enable_host_call_scalar_metrics`, `_enable_caching_with_tpu_var_mode`, `_top_k_sampling_num_per_core`.
  - Logs fixed vocab settings when `vocab_size_per_slot` or `custom_vocab_size_mapping` provided.
  - Builds `vocab_size_dict = _create_vocab_dict()` and `Env(vocab_size_dict, params)`.
  - Initializes `_feature_to_config_dict` and `_table_to_config_dict` as empty dicts.
- `download_vocab_size_file_from_hdfs()`:
  - Deletes and recreates local `temp/` folder.
  - Builds HDFS path: `"{vocab_file_folder_prefix}{end_date}/part*.csv"`.
  - Runs `hadoop fs -copyToLocal` to temp folder.
  - If returncode==0 and exactly one file downloaded:
    - Sets `p.vocab_file_path` to that file path.
  - Else logs downloaded files and keeps existing `p.vocab_file_path`.
- `_create_vocab_dict()`:
  - If `train.end_date` and `train.vocab_file_folder_prefix` set, calls `download_vocab_size_file_from_hdfs()`.
  - Asserts `p.vocab_file_path` exists.
  - Reads TSV with 2 fields per line; ignores non-numeric slot IDs.
  - For each slot:
    - `distinct_count = vocab_size_per_slot` if set, else parsed count, overridden by `custom_vocab_size_mapping` when present.
    - Applies `vocab_size_offset` if set.
    - Stores `vocab_size_dict[slot_id] = distinct_count`.
  - Logs dict; returns it.
- `_parse_inputs(return_values)`:
  - If tuple: returns `(features, labels)`; else `(return_values, None)`.
- `create_input_fn(mode=TRAIN)`:
  - Asserts TRAIN mode only.
  - `file_pattern = p.train.file_pattern`.
  - `tf_example_parser` builds feature_map via `_get_feature_map`, parses batch with `tf.io.parse_example`, then `_post_process_example`.
  - `insert_stopping_signal(stop, batch_size, name)`:
    - Adds `name` bool tensor: ones if `stop`, zeros otherwise.
    - For sparse tensors, replaces with empty SparseTensor when `stop=True`.
  - `input_fn(params)`:
    - If `params["cpu_test"]` True:
      - `TFRecordDataset(file_pattern)` -> batch(drop_remainder) -> map(tf_example_parser) -> repeat.
    - Else:
      - If `file_pattern` provided: `Dataset.list_files(file_pattern, shuffle=False)`.
      - Else:
        - Require `train.file_folder` + `train.date_and_file_name_format`.
        - Build `file_pattern_` and list_files (shuffle=False).
        - Require `train.end_date` and `params["enable_stopping_signals"]` not None.
        - Apply `util.range_dateset(..., start_date, end_date)`.
      - Shard files: `_, call_index, num_calls, _ = params["context"].current_input_fn_deployment()` then `files.shard(num_calls, call_index)`.
      - `files.interleave(TFRecordDataset, cycle_length=files_interleave_cycle_length, num_parallel_calls=AUTOTUNE, deterministic=p.deterministic)`.
      - Batch + map tf_example_parser (AUTOTUNE, deterministic=p.deterministic).
      - If `p.train.repeat`: assert `enable_stopping_signals` is False, then `dataset.repeat()`.
      - If `enable_stopping_signals`:
        - `user_provided_dataset = dataset.map(insert_stopping_signal(stop=False), deterministic=False)`.
        - `final_batch_dataset = dataset.repeat().map(insert_stopping_signal(stop=True), deterministic=False)`.
        - `dataset = user_provided_dataset.concatenate(final_batch_dataset)`.
      - `dataset.prefetch(AUTOTUNE)`.
    - Returns dataset.
- `logits_fn()`: abstract, raises `NotImplementedError`.
- `init_slot_to_env()`:
  - Logs, calls `self.logits_fn()` (to register slots), then `self._env.finalize()`.
- `create_model_fn()`: abstract, raises `NotImplementedError('Abstract method.')`.
- `_get_feature_map()`: abstract, raises `NotImplementedError`.
- `_post_process_example(example)`:
  - For each `(slot_id, feature_slot)` in `env.slot_id_to_feature_slot`:
    - Skip if `feature_columns` empty.
    - For each `feature_column`:
      - `embedding_tensor = example[f"{fc_name}_0"]`.
      - If `FeatureColumn3D`: `embedding_tensor.to_sparse()`, then clamp values >=0; else clamp values on SparseTensor.
      - If `vocab_size_per_slot`: mod values; else set `vocab_size = env._vocab_size_dict.get(slot_id,10)` and apply `custom_vocab_size_mapping` if present.
      - If `qr_multi_hashing` and `vocab_size > qr_hashing_threshold`:
        - `R_vocab_size = vocab_size // qr_collision_rate + 1`, `Q_vocab_size = qr_collision_rate + 1`.
        - Deletes original `example[f"{fc_name}_0"]`.
        - For each feature_slice: add `fc_name_{slice}_0` (floormod by R) and `fc_name_{slice}_1` (floordiv by R).
      - Else:
        - If `FeatureColumn3D`: compute `row_lengths`, store `"{fc_name}_0_row_lengths"`, and slice sparse tensor to `max_seq_length`.
        - Set `example[f"{fc_name}_0"] = tf.sparse.reorder(new_sparse)`.
        - For each `feature_slice` where `slice_index != 0`, alias to the `_0` sparse tensor.
  - If `_UID` in example: compute `uid_bucket = uid % _RATIO_N` and store as int32 under `_UID_BUCKET`.
  - Returns example.
- `create_feature_and_table_config_dict()`:
  - Asserts env finalized.
  - For each slot/feature_column/feature_slice:
    - `vocab_size = env.vocab_size_dict.get(slot_id,1)`.
    - If `qr_multi_hashing` and vocab_size > threshold:
      - Create remainder and quotient `TableConfig`s and `FeatureConfig`s (`*_0`, `*_1`) if not already present.
    - Always create base `table_{slot}_{slice}` if missing.
    - Create `FeatureConfig` for `fc_name_{slice}`; for `FeatureColumn3D`, include `max_sequence_length`.
  - Returns `(feature_to_config_dict, table_to_config_dict)`.
- `cross_shard_optimizer(optimizer, params)`:
  - If `params["cpu_test"]`: return optimizer; else wrap with `tf.tpu.CrossShardOptimizer`.
- `process_features_for_cpu_test(features)`:
  - For each SparseTensor feature:
    - Look up `FeatureConfig`/`TableConfig` to get `dim`, `max_sequence_length`, `vocab_size`.
    - Random init array shaped `[vocab_size, dim]` or `[vocab_size, max_seq_len*dim]`, cast to float32.
    - Create `tf.get_variable(name=feature_name, initializer=initvalue)`.
    - Mod feature ids by vocab_size; `safe_embedding_lookup_sparse(..., combiner="sum")`.
    - If max_sequence_length != 0: reshape to `[-1, max_seq_len, dim]`.
    - Store in `processed_features`.
  - Non-sparse features passed through.
  - Clears `_feature_to_config_dict` and `_table_to_config_dict` before return.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/src/base_embedding_task.rs`.
- Rust public API surface: base embedding task trait/struct with dataset and embedding config helpers.
- Data model mapping: TF dataset pipeline and TPU embedding configs (likely via TF runtime bridge).

**Implementation Steps (Detailed)**
1. Port parameter schema and vocab file parsing.
2. Implement dataset pipeline generation or equivalent data loader.
3. Recreate embedding config generation and QR hashing logic.
4. Implement CPU test path for embeddings.

**Tests (Detailed)**
- Python tests: none direct; used by TPU runner tests.
- Rust tests: unit tests for vocab dict creation and QR hashing config creation.

**Gaps / Notes**
- Full parity depends on TF TPU embedding APIs; Rust likely needs a bridge or compatibility layer.

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

### `monolith/core/base_host_call.py`
<a id="monolith-core-base-host-call-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 145
- Purpose/role: Collects tensors for TPU host calls, with compression/decompression by dtype.
- Key symbols/classes/functions: `BaseHostCall.record_summary_tensor`, `compress_tensors`, `decompress_tensors`.
- External dependencies: `tensorflow`, `absl.logging`.
- Side effects: builds internal tensor lists; uses global_step tensor.

**Required Behavior (Detailed)**
- `__init__(output_dir, enable_host_call)`:
  - Initializes `_tensor_names` with `"global_step"` and `_tensors` with reshaped global step.
  - `_lists_tensor_sizes` tracks original sizes for decompression.
- `record_summary_tensor(name, tensor)`:
  - No-op if host call disabled.
  - Asserts unique name, asserts tensor rank <= 1, reshapes to `[-1]`, appends.
- `compress_tensors()`:
  - Groups tensors by dtype; concatenates each dtype list along axis 0, then `expand_dims` to add batch dimension.
  - Stores per-group tensor sizes in `_lists_tensor_sizes`.
  - Replaces `_tensor_names` and `_tensors` with compressed versions.
- `decompress_tensors(tensors)`:
  - Splits each compressed tensor by recorded sizes; squeezes to 1D.
  - Asserts first tensor name is `global_step` (error message uses first character only).
  - Returns `(global_step_scalar, decompressed_tensor_list)` where global step is `decompressed_tensor_list[0][0]`.
- `generate_host_call_hook()` default returns `None` (to be overridden).

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/src/base_host_call.rs`.
- Rust public API surface: `BaseHostCall` struct with tensor collection and (de)compression.

**Implementation Steps (Detailed)**
1. Implement tensor collection and dtype grouping logic.
2. Preserve compression/decompression shapes and per-dtype grouping semantics.
3. Mirror global_step positioning and return shape.

**Tests (Detailed)**
- Python tests: none specific (used indirectly by embedding host call tests).
- Rust tests: unit tests for compress/decompress roundtrip.

**Gaps / Notes**
- Error message in assert uses `self._tensor_names[0][0]` (first char); keep for parity.

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

### `monolith/core/base_layer.py`
<a id="monolith-core-base-layer-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 161
- Purpose/role: Base class for layers with child management, name assignment, and per-graph layer loss tracking.
- Key symbols/classes/functions: `BaseLayer`, `get_uname`, `add_layer_loss`, `get_layer_loss`.
- External dependencies: `tensorflow`, `InstantiableParams`, `NestedMap`.
- Side effects: global registries `_layer_loss` and `_name_inuse` are mutated.

**Required Behavior (Detailed)**
- `params()`:
  - Returns `InstantiableParams(cls)` and defines `name` using `get_uname(cls.__name__)`.
- `__init__(params)`:
  - Asserts `params.name` is set.
  - Initializes `_private_children` as `NestedMap`.
- `children` property returns `_private_children`.
- `__getattr__`:
  - Raises AttributeError if `_private_children` not created.
  - If `name` in children, returns it.
  - If class has a property with that name, calls the property's getter to surface the same AttributeError.
  - Else raises `"<name> is not a sub-layer of <self>."`.
- `__call__` forwards to `fprop()`.
- `fprop()` is abstract: raises `NotImplementedError('Abstract method of %s' % self)`.
- `create_child(name, params)`:
  - If `params.name` empty, assigns `self.p.name` (assumes BaseLayer has `p` attribute set by InstantiableParams).
  - Instantiates child via `params.instantiate()` and stores under `_private_children[name]`.
- `create_children(name, params_list)`:
  - Creates list; for each param, sets `param.name = f"{name}_{index}"` if missing; instantiates and appends.
- `get_uname(name)`:
  - Uses `_name_inuse` defaultdict; **note**: code checks membership but never inserts, so it currently always returns `name` (no suffix) unless keys are inserted elsewhere.
- `add_layer_loss(name, loss)`:
  - Adds loss into `_layer_loss` keyed by default graph and layer name.
- `get_layer_loss()`:
  - Returns the dict for the current default graph.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/src/base_layer.rs`.
- Rust public API surface: `BaseLayer` trait/struct, child map, `get_uname`, layer loss registry.
- Data model mapping: `NestedMap` equivalent and parameter instantiation mechanism.

**Implementation Steps (Detailed)**
1. Implement a base layer trait with child management and `fprop` entrypoint.
2. Provide a `NestedMap` equivalent with attribute-like access.
3. Mirror `get_uname` behavior (including current no-op uniqueness unless explicitly fixed).
4. Implement per-graph loss registry or document deviation if Rust lacks TF graphs.

**Tests (Detailed)**
- Python tests: `monolith/core/base_layer_test.py`
- Rust tests: verify create_child/create_children and `__getattr__`-like behavior.

**Gaps / Notes**
- BaseLayer relies on `self.p` being set by hyperparams instantiation; mirror this in Rust or document.

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

### `monolith/core/base_layer_test.py`
<a id="monolith-core-base-layer-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 41
- Purpose/role: Tests child creation APIs in `BaseLayer`.
- Key symbols/classes/functions: `BaseLayerTest.test_create_child`, `.test_create_children`.
- External dependencies: `base_layer`.
- Side effects: none.

**Required Behavior (Detailed)**
- `test_create_child`:
  - Instantiates BaseLayer params, sets name, creates layer, calls `create_child`, and asserts child exists.
- `test_create_children`:
  - Creates two child layers and asserts list length is 2.
- Both tests set `layer._disable_create_child = False` (attribute not defined on BaseLayer but set anyway).

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/tests/base_layer.rs`.
- Rust public API surface: BaseLayer child creation.

**Implementation Steps (Detailed)**
1. Port tests using Rust BaseLayer + params instantiation.
2. Assert child map/list presence and lengths.

**Tests (Detailed)**
- Python tests: this file
- Rust tests: same assertions.

**Gaps / Notes**
- Python tests access `_disable_create_child` which is not defined in BaseLayer; confirm if Rust needs similar flag (likely no-op).

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

### `monolith/core/base_model_params.py`
<a id="monolith-core-base-model-params-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 25
- Purpose/role: Defines abstract params holder for single-task models.
- Key symbols/classes/functions: `SingleTaskModelParams.task`.
- External dependencies: none.
- Side effects: none.

**Required Behavior (Detailed)**
- `SingleTaskModelParams.task()` is abstract and must be overridden to return task params; raises `NotImplementedError('Abstract method')` by default.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/src/base_model_params.rs`.
- Rust public API surface: trait or struct with `task()`/`task_params()` abstract method.

**Implementation Steps (Detailed)**
1. Create a Rust trait for model params with `task()` returning task config.
2. Ensure errors are explicit when called on base type.

**Tests (Detailed)**
- Python tests: none specific.
- Rust tests: optional compile-time trait enforcement or runtime panic test.

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
