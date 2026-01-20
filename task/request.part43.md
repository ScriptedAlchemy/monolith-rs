<!--
Source: task/request.md
Lines: 9951-10117 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/estimator.py`
<a id="monolith-native-training-estimator-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 667
- Purpose/role: High-level Estimator API for local/distributed training, evaluation, prediction, and saved_model export/import.
- Key symbols/classes/functions: `EstimatorSpec`, `RunConfig`, `Estimator`, `import_saved_model`.
- External dependencies: TensorFlow Estimator, Kazoo/ZK, AgentService, CpuTraining, RunnerConfig, service discovery, DumpUtils, device_utils, distribution_utils.
- Side effects: mutates env vars, initializes ZK clients/backends, logs metrics, writes model dumps, may start/stop distributed services.

**Required Behavior (Detailed)**
- `EstimatorSpec` (namedtuple):
  - Fields: `label`, `pred`, `head_name`, `loss`, `optimizer`, `classification`.
  - `__new__` sets defaults `head_name=None`, `loss=None`, `optimizer=None`, `classification=True`.
  - `_replace` forbids changing `mode` if present in kwargs and different from existing (defensive check).
- `RunConfig` (dataclass_json):
  - Fields include: `is_local`, `num_ps`, `num_workers`, timeout settings, layout flags, retry settings, parameter-sync fields, checkpoint/export settings, profiling flags, alias map settings, kafka settings, metrics flags, summary/log cadence.
  - `to_runner_config()`:
    - Builds `RunnerConfig` with key fields.
    - For each RunConfig field, if value differs from RunConfig default and differs from current conf value, updates conf (preserves CLI overrides).
    - Converts `ServiceDiscoveryType.CONSUL` to `ServiceDiscoveryType.ZK`.
    - If `enable_gpu_training` is False, ensures `embedding_prefetch_capacity >= 1` and `enable_embedding_postpush=True`.
    - If `enable_parameter_sync` is True, sets `enable_realtime_training` or `enable_parameter_sync` on `RunnerConfig` (raises if neither exists).
  - `__post_init__`:
    - Serializes to JSON and records config in `DumpUtils`.
    - Records user params that differ from defaults to `DumpUtils`.
- `Estimator.__init__(model, conf, warm_start_from=None)`:
  - Converts `RunConfig` to `RunnerConfig` if needed.
  - Sets deep-insight metrics on the model based on local vs distributed and runner_conf values.
  - If realtime training on PS, initializes sync backend via ZK (either `ZKBackend` or `ReplicaWatcher` with `MonolithKazooClient`).
  - Applies `params_override` JSON to model `.p` or `.params` when present.
  - Attempts `env_utils.setup_hdfs_env()` if `HADOOP_HDFS_HOME` missing (logs errors).
  - Exports env vars `TF_GRPC_WORKER_CACHE_THREADS` and `MONOLITH_GRPC_WORKER_SERVICE_HANDLER_MULTIPLIER` from runner_conf.
- `Estimator._est` (lazy property):
  - Deep-copies model; sets mode `PREDICT` and instantiates `CpuTraining` task.
  - Deletes `TF_CONF` env var if present.
  - Constructs `tf.estimator.Estimator` with `model_fn`, `model_dir`, and `RunConfig(log_step_count_steps=...)`, with `warm_start_from` if provided.
- `Estimator.train(steps=None, max_steps=None, hooks=None)`:
  - Validates hooks are `tf.estimator.SessionRunHook`.
  - Sets metric prefix `monolith.training.<deep_insight_name>`.
  - Deep-copies model, sets mode `TRAIN`, overrides steps/max_steps if provided.
  - If local: choose model_dir (default `/tmp/<user>/<model>`), call `local_train_internal`, and write `DumpUtils` to `model_dump`.
  - If distributed: disable DumpUtils; start sync backend and subscribe model; log env + flags + params.
    - If `enable_full_sync_training`: call `init_sync_train_and_update_conf` then `distributed_sync_train`.
    - Else: use `monolith_discovery` context; if `enable_gpu_training` -> `device_utils.enable_gpu_training()` and disable `use_gpu_emb_table`; if partial sync and worker -> `try_init_cuda()` and set `device_fn`.
    - Call `distributed_train`.
  - Calls `close()` at end.
- `Estimator.evaluate(steps=None, hooks=None)`:
  - Mirrors `train()` but uses mode `EVAL` and `distributed_train` (no user hooks in distributed eval except full sync).
- `Estimator.predict(...)`:
  - Creates estimator via `_est`, builds input_fn, calls `est.predict(...)`, then `close()`.
- `Estimator.export_saved_model(batch_size=64, name=None, dense_only=False, enable_fused_layout=False)`:
  - Copies model/conf; sets `enable_fused_layout`, model name, batch size, mode `PREDICT`.
  - Creates `CpuTraining` task and exporter; uses `ParserCtx(enable_fused_layout=...)`.
  - Calls `exporter.export_saved_model` with `serving_input_receiver_fn`.
- `import_saved_model(saved_model_path, input_name='instances', output_name='output', signature=None)`:
  - Context manager that resolves latest numeric version directory if `saved_model_path` not numeric.
  - Loads SavedModel with `tf.saved_model.load`, chooses signature (default serving).
  - Builds placeholders dict from requested inputs.
  - Builds output dict from requested outputs or all outputs if none provided.
  - Returns `infer(features)` callable that runs session and maps output tensor names to output names.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/estimator.rs` plus new `run_config.rs`/`runner_config.rs` and export/import helpers.
- Rust public API surface:
  - `EstimatorSpec` struct analog for model outputs.
  - `RunConfig` struct with JSON serialization + `to_runner_config` merge semantics.
  - `Estimator` wrapper that orchestrates training/eval/predict, plus saved-model import/export if TF runtime is enabled.
- Data model mapping: use `monolith-training` model traits (`ModelFn`) for Candle backend; optional TF runtime path for SavedModel import/export.
- Feature gating: `tf-runtime` feature for SavedModel and TF Estimator parity; default Candle backend implements local training only.
- Integration points: `cpu_training`/`distributed_train` equivalents, service discovery, device utils, dump utilities, parameter sync backend.

**Implementation Steps (Detailed)**
1. Implement `RunConfig` in Rust with all fields and defaults; add JSON serialization to match Python `dataclass_json`.
2. Implement `to_runner_config` merge logic (only override when RunConfig value differs from default and from current runner config).
3. Implement `Estimator` struct with local training/eval/predict flows mirroring Python.
4. Add optional ZK/AgentService integration or stub with clear errors when unavailable.
5. Implement env var exports (`TF_GRPC_WORKER_CACHE_THREADS`, `MONOLITH_GRPC_WORKER_SERVICE_HANDLER_MULTIPLIER`).
6. Implement SavedModel export/import only when TF runtime is enabled; otherwise document as unsupported.
7. Add parity tests for RunConfig merging and local train/eval/predict call flow.

**Tests (Detailed)**
- Python tests: `estimator_test.py`, `estimator_dist_test.py`, `estimator_mode_test.py`.
- Rust tests: add `estimator_test.rs` for local flow and config merging; integration tests for distributed modes as available.
- Cross-language parity test: compare config merge outputs, model_dir resolution, and SavedModel import/export behavior.

**Gaps / Notes**
- Python relies on TF Estimator and distributed training stack; Rust currently has only stubs for distributed execution.
- SavedModel import/export likely requires TF runtime and custom ops.

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

### `monolith/native_training/estimator_dist_test.py`
<a id="monolith-native-training-estimator-dist-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 166
- Purpose/role: Integration test for distributed training/eval using TF_CONFIG-style discovery and multi-process PS/worker setup.
- Key symbols/classes/functions: `EstimatorTrainTest`, `get_cluster`, `get_free_port`.
- External dependencies: `tensorflow`, `RunnerConfig`, `TestFFMModel`, `TfConfigServiceDiscovery`, `Estimator`.
- Side effects: Spawns multiple processes, binds local ports, writes checkpoints under tmp.

**Required Behavior (Detailed)**
- `get_free_port()`:
  - Binds a local socket on port 0 to find an available port; closes socket and returns port.
- `get_cluster(ps_num, worker_num)`:
  - Returns dict with `ps`, `worker`, and `chief` addresses on free ports (workers exclude chief).
- `EstimatorTrainTest.setUpClass`:
  - Removes existing `model_dir` if present.
  - Creates `TestFFMModel` params with deep insight disabled and batch size 64.
- `EstimatorTrainTest.train()`:
  - Spawns `ps_num` PS processes and `worker_num` worker/chief processes.
  - Each process builds `TF_CONFIG`-like dict, uses `TfConfigServiceDiscovery`, constructs `RunnerConfig`, and calls `Estimator.train(steps=10)`.
  - Waits for all processes; asserts exitcode 0 for each.
- `EstimatorTrainTest.eval()`:
  - Same as train but calls `Estimator.evaluate(steps=10)`.
- `test_dist`:
  - Runs `train()` then `eval()`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests/estimator_dist_test.rs` (new).
- Rust public API surface: distributed training harness and config discovery equivalent.
- Data model mapping: distributed cluster config (ps/worker/chief), runner config, model params.
- Feature gating: likely `tf-runtime` or `distributed` feature; skip if distributed stack not available.
- Integration points: service discovery and process orchestration.

**Implementation Steps (Detailed)**
1. Implement a Rust integration test that spawns multiple processes (or threads) for PS/worker roles.
2. Provide a discovery config equivalent to `TfConfigServiceDiscovery` and ensure `Estimator` can use it.
3. Run short train/eval steps and assert clean exit.
4. Add timeouts and cleanup for spawned processes and temp directories.

**Tests (Detailed)**
- Python tests: `monolith/native_training/estimator_dist_test.py`.
- Rust tests: `monolith-rs/crates/monolith-training/tests/estimator_dist_test.rs` (integration).
- Cross-language parity test: verify training/eval complete under equivalent cluster topology.

**Gaps / Notes**
- Uses real multi-process TF; Rust currently lacks distributed PS/worker runtime.
- Port binding is fragile; consider deterministic port assignment for CI.

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
