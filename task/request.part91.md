<!--
Source: task/request.md
Lines: 21625-21836 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/path_utils.py`
<a id="monolith-path-utils-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 47
- Purpose/role: Locate monolith base directory and resolve paths to bundled libraries.
- Key symbols/classes/functions: `find_main`, `get_libops_path`.
- External dependencies: `os` only (explicitly avoids third-party imports).
- Side effects: raises ValueError when base directory cannot be found.

**Required Behavior (Detailed)**
- `find_main()`:
  - Uses `__file__` path; searches for split markers in order: `/__main__/`, `/site-packages/`, `/monolith/`.
  - If marker found:
    - For `/monolith/`: `main_dir` is path prefix before marker.
    - Else: `main_dir` is prefix joined with the marker path component.
  - Returns `main_dir` only if `main_dir/monolith` exists; else raises ValueError with full path in message.
- `get_libops_path(lib_name)`:
  - Returns `os.path.join(find_main(), lib_name)`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/src/path_utils.rs`.
- Rust public API surface: `find_main()` and `get_libops_path()`.

**Implementation Steps (Detailed)**
1. Implement path resolution with the same marker order and behavior.
2. Preserve error message details for diagnostics.

**Tests (Detailed)**
- Python tests: `monolith/utils_test.py` (find_main / get_libops_path).
- Rust tests: replicate `find_main` behavior under test harness.

**Gaps / Notes**
- Behavior depends on Bazel layout (`__main__`); Rust tests should simulate or adjust.

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

### `monolith/tpu_runner.py`
<a id="monolith-tpu-runner-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 429
- Purpose/role: TPU training/eval runner using `tf.estimator.tpu.TPUEstimator` with embedding support and optional CPU test mode.
- Key symbols/classes/functions: `TPURunner`, `create_tpu_estimator`, `create_tpu_estimator_on_cpu`, `run`, CLI `main`.
- External dependencies: `tensorflow.compat.v1`, `cloud_tpu_client`, `BaseEmbeddingTask`, `model_registry`.
- Side effects: configures TPU versions, launches training/eval, writes summaries.

**Required Behavior (Detailed)**
- CLI flags include TPU location (`tpu`, `gcp_project`, `tpu_zone`), mode, model_dir, checkpoint interval, iteration counts, embedding options, CPU test, partition strategy, overwrite_end_date.
- `TPURunner.__init__`:
  - Reads flags and task_param; sets accelerator to `tpu`.
  - Allows task_param to override save_checkpoints_steps.
  - Optionally overwrites `train.end_date` when flag provided.
- `create_tpu_estimator`:
  - Optionally configures TPU version and waits for healthy.
  - Builds TPU cluster resolver, computes total replicas and global batch size.
  - Sets TPUConfig with iterations_per_loop and host_call settings.
  - Uses `TPUInfeedOutfeedSessionWithEndOfStreamHandlingHook` when stopping signals enabled.
  - Builds embedding_config_spec if feature/table configs exist.
  - Returns TPUEstimator and total_replicas.
- `create_tpu_estimator_on_cpu`:
  - Creates TPUEstimator with `use_tpu=False` and small batch size; used for CPU test.
- `run()`:
  - Loads global step (or 0).
  - Instantiates task; if BaseEmbeddingTask, prepares feature/table configs.
  - Uses CPU test wrapper if enabled.
  - `train`: `est.train`.
  - `eval`: iterates checkpoints and evaluates; writes summaries and stops at max_steps.
  - `train_and_eval`: not supported (raises TypeError).
- `main`: logs FLAGS and task_param, runs runner.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/tpu_runner.rs`.
- Rust public API surface: runner struct + CLI entrypoint.
- Data model mapping: TPU/embedding configs likely require TF runtime bindings.
- Feature gating: `tf-runtime`, `tpu` features.

**Implementation Steps (Detailed)**
1. Port CLI flag set and task registry lookup.
2. Decide runtime strategy (native Rust vs TF bridge).
3. Preserve TPU version config and host_call/embedding config semantics if using TF.
4. Mirror evaluation-by-checkpoint loop and summary writing.

**Tests (Detailed)**
- Python tests: none specific.
- Rust tests: CLI parsing + control flow tests; TPU behavior likely gated/skipped in CI.

**Gaps / Notes**
- Full parity depends on TensorFlow TPU Estimator which is not available natively in Rust.

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

### `monolith/utils.py`
<a id="monolith-utils-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 81
- Purpose/role: Small utility helpers for TF monkey patching and recursive file copy with tf.io.gfile.
- Key symbols/classes/functions: `enable_monkey_patch`, `CopyFile`, `CopyRecursively`.
- External dependencies: `tensorflow`, `ThreadPoolExecutor`.
- Side effects: modifies `tensorflow.python.training.monitored_session` module attribute; copies files (possibly remote).

**Required Behavior (Detailed)**
- `enable_monkey_patch()`:
  - Imports `tensorflow.python.training.monitored_session` and sets `_PREEMPTION_ERRORS` to `(tf.errors.AbortedError,)`.
- `CopyFile(src, dst, overwrite=True, skip_nonexist=True, max_retries=5)`:
  - Uses `tf.io.gfile.copy` and retries on NotFoundError (skip if `skip_nonexist`).
- `CopyRecursively(src, dst, max_workers=1, skip_nonexist=True, max_retries=5)`:
  - Recursively copies a directory tree via `tf.io.gfile`.
  - If `src` missing and `skip_nonexist`, returns; else raises ValueError.
  - If `dst` exists, removes it then recreates.
  - If `max_workers > 1`, uses ThreadPoolExecutor to copy files in parallel.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-tf/src/utils.rs` (TF-dependent utilities) plus `monolith-core/src/path_utils.rs` for re-exports.
- Rust public API surface: equivalents for monkey patch (if bridging TF) and recursive copy.

**Implementation Steps (Detailed)**
1. Provide TF monkey patch equivalent when using Python/TF runtime; otherwise document unsupported.
2. Implement recursive copy with retries and optional parallelism.
3. Ensure semantics match tf.io.gfile for remote paths (HDFS/GCS) where needed.

**Tests (Detailed)**
- Python tests: `monolith/utils_test.py`
- Rust tests: port tests for `find_main`, `get_libops_path`, and CopyRecursively.

**Gaps / Notes**
- Full parity requires tf.io.gfile support; Rust may need a virtual filesystem abstraction.

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

### `monolith/utils_test.py`
<a id="monolith-utils-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 65
- Purpose/role: Tests path utilities, monkey patch, and recursive copy.
- Key symbols/classes/functions: `UtilsTest` test cases.
- External dependencies: `tensorflow`, `monitored_session`.
- Side effects: writes temp files in `/tmp`.

**Required Behavior (Detailed)**
- `testFindMain`: `utils.find_main()` base dir last path component is `__main__` (Bazel layout assumption).
- `testGetLibopsPath`: `utils.get_libops_path("monolith/utils_test.py")` exists.
- `testLoadMonitoredSession`: `_PREEMPTION_ERRORS` equals `(errors.AbortedError,)` after monkey patch.
- `testMultiThreadedCopy`: creates nested dirs/files and verifies copied content with `CopyRecursively(max_workers=2)`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-tf/tests/utils.rs`.
- Rust public API surface: path utils + recursive copy.

**Implementation Steps (Detailed)**
1. Port tests with temp dirs and file content checks.
2. If TF monkey patch is unsupported, document and adjust tests accordingly.

**Tests (Detailed)**
- Python tests: this file
- Rust tests: parity tests for path and copy semantics.

**Gaps / Notes**
- `find_main` behavior is tied to Bazel execution layout; Rust tests may need harness adjustments.

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
