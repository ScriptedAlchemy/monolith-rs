<!--
Source: task/request.md
Lines: 4655-4913 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/core/util_test.py`
<a id="monolith-core-util-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 149
- Purpose/role: Tests `range_dateset` date filtering behavior.
- Key symbols/classes/functions: `UtilTest.test_range_dataset_*`.
- External dependencies: `tensorflow.compat.v1`.
- Side effects: none.

**Required Behavior (Detailed)**
- Tests that `range_dateset` filters dataset elements based on date substring in path.
- Covers single date, multiple dates, out-of-bound ranges, missing start or end date.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/tests/util.rs`.
- Rust public API surface: `range_dateset` equivalent.

**Implementation Steps (Detailed)**
1. Port tests using a mock dataset list and filter function.
2. Ensure output order and content match Python.

**Tests (Detailed)**
- Python tests: this file
- Rust tests: same scenarios.

**Gaps / Notes**
- Requires a dataset abstraction; may be tested with plain lists in Rust.

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

### `monolith/core/variance_scaling.py`
<a id="monolith-core-variance-scaling-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 188
- Purpose/role: Numpy-based variance scaling initializer (truncated/untruncated normal or uniform).
- Key symbols/classes/functions: `_compute_fans`, `VarianceScaling.__call__`, `get_config`.
- External dependencies: `numpy`, `scipy.stats.truncnorm`.
- Side effects: uses NumPy RNG (seeded per call).

**Required Behavior (Detailed)**
- `_compute_fans(shape, data_format)` computes fan_in/fan_out for dense or conv shapes; supports `channels_first/last`.
- `VarianceScaling.__init__` validates `scale > 0`, `mode` in `fan_in/fan_out/fan_avg`, distribution in `truncated_normal/untruncated_normal/uniform`.
- `__call__(shape, dtype)`:
  - Computes scaled variance based on mode.
  - Seeds NumPy RNG with `self.seed` each call.
  - For `truncated_normal`: uses scipy truncnorm with cutoff ±2 stddev, stddev adjusted by constant 0.87962566103423978.
  - For `untruncated_normal`: uses `np.random.normal`.
  - For `uniform`: uses `np.random.uniform` with limit `sqrt(3*scale)`.
- `get_config()` returns dict with scale/mode/distribution/seed.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/src/variance_scaling.rs`.
- Rust public API surface: `VarianceScaling` struct implementing initializer to produce arrays.

**Implementation Steps (Detailed)**
1. Port `_compute_fans` logic and mode handling.
2. Implement truncated normal sampling (use rand_distr or custom truncation) matching SciPy behavior.
3. Ensure seeding behavior matches (seed per call).

**Tests (Detailed)**
- Python tests: none.
- Rust tests: sample shape and distribution sanity checks; seed determinism.

**Gaps / Notes**
- Matching SciPy truncnorm exactly may require careful implementation.

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

### `monolith/gpu_runner.py`
<a id="monolith-gpu-runner-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 226
- Purpose/role: GPU training/eval runner for TF Estimator, with optional Horovod multi-GPU support.
- Key symbols/classes/functions: `GPURunner`, `create_estimator`, `run`, CLI `main`.
- External dependencies: `tensorflow`, `horovod`, `mpi4py`, `absl.flags`, `model_registry`.
- Side effects: initializes Horovod, configures GPU visibility, trains/evaluates model, writes summaries.

**Required Behavior (Detailed)**
- CLI flags: `task`, `model_dir`, `save_checkpoints_steps`, `mode` (`train|eval|train_and_eval`).
- `GPURunner.__init__`:
  - Reads flags and task_param; sets `_mode`.
- `create_estimator(model_fn)`:
  - If `task_param.accelerator == 'horovod'`:
    - `hvd.rank()` controls checkpoint saving (only rank 0).
    - Configures `tf.compat.v1.ConfigProto` with XLA JIT ON and GPU memory growth.
    - Sets `visible_device_list` to local rank.
    - `num_gpus = hvd.size()`.
  - Else: `num_gpus = 1` and uses `tf.compat.v1.estimator.RunConfig`.
  - Returns `tf.compat.v1.estimator.Estimator` with params: train/eval batch sizes, accelerator, num_replicas, hvd_rank.
- `run()`:
  - Loads global step (or 0).
  - Instantiates task; builds input_fn_train/eval and model_fn.
  - If horovod: `hvd.init()`, sets visible GPU.
  - For `train`:
    - Horovod: uses `BroadcastGlobalVariablesHook(0)`.
    - Non-horovod: `est.train`.
  - For `eval`:
    - Computes `num_steps` from `eval_examples` and batch size.
    - Runs `est.evaluate` and writes summary under `model_dir/eval`.
  - For `train_and_eval`:
    - Loop train for `steps_per_eval` up to max_steps, evaluate and write summary.
    - Horovod uses MPI barrier after each eval cycle.
- `main`: fetches task params from registry and runs `GPURunner`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/gpu_runner.rs` (or CLI bin).
- Rust public API surface: runner struct + CLI entrypoint.
- Data model mapping: task params, estimator equivalent (likely still Python-backed or TensorFlow runtime binding).
- Feature gating: `tf-runtime` and `horovod` (if supported) features.

**Implementation Steps (Detailed)**
1. Recreate CLI flags and task registry lookup in Rust.
2. Decide how to execute training: native Rust pipeline or Python/TF bridge.
3. If bridging TF, maintain Horovod initialization and GPU pinning logic.
4. Mirror evaluation loop, summary writing, and MPI barrier behavior.

**Tests (Detailed)**
- Python tests: none specific.
- Rust tests: integration tests for CLI argument parsing and control flow (mocked task).

**Gaps / Notes**
- Full parity likely requires TF Estimator execution; consider keeping Python runner or embedding TF runtime.

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

### `monolith/native_training/alert/alert_manager.py`
<a id="monolith-native-training-alert-alert-manager-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 31
- Purpose/role: Placeholder alert manager module defining a flag and stub accessor.
- Key symbols/classes/functions: `get_default_alert_manager()`.
- External dependencies: `absl.flags`, `absl.logging`, `google.protobuf.text_format`, `threading`, `time`, `traceback`.
- Side effects: defines `--monolith_alert_proto` flag at import time.

**Required Behavior (Detailed)**
- Flag definition:
  - `monolith_alert_proto` (string), default `""`, description: "The text format of alert proto."
- `get_default_alert_manager()`:
  - Returns `None`.
- No other behavior implemented.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src` (if alerts are ported).
- Rust public API surface: optional alert manager accessor + config flag.
- Data model mapping: map CLI flag to config struct; no runtime behavior.
- Feature gating: none.
- Integration points: training runner or alert subsystem (not present).

**Implementation Steps (Detailed)**
1. Add a config flag or CLI option mirroring `monolith_alert_proto`.
2. Provide a stub `get_default_alert_manager` returning `None`/`Option::None`.

**Tests (Detailed)**
- Python tests: none.
- Rust tests: none required (flag parsing only).
- Cross-language parity test: not applicable.

**Gaps / Notes**
- Module is largely empty; functionality likely lives elsewhere (or removed).

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

### `monolith/native_training/alert/alert_manager_test.py`
<a id="monolith-native-training-alert-alert-manager-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 32
- Purpose/role: Placeholder test module; no actual tests defined.
- Key symbols/classes/functions: none (only `__main__` entrypoint).
- External dependencies: `absl.testing`, `absl.flags/app`, `google.protobuf.text_format`.
- Side effects: running as main executes `absltest.main()`.

**Required Behavior (Detailed)**
- No tests; `absltest.main()` called if executed.

**Rust Mapping (Detailed)**
- Target crate/module: none.
- Rust public API surface: none.
- Data model mapping: none.
- Feature gating: none.
- Integration points: none.

**Implementation Steps (Detailed)**
1. No Rust tests required unless alert manager gains functionality.

**Tests (Detailed)**
- Python tests: none.
- Rust tests: none.
- Cross-language parity test: not applicable.

**Gaps / Notes**
- No tests defined; consider removing or adding real coverage if alerting is implemented.

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
