<!--
Source: task/request.md
Lines: 15145-15383 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/metric/cli.py`
<a id="monolith-native-training-metric-cli-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 28
- Purpose/role: Stub/no-op CLI client placeholder; provides a `Client` with no-op methods to satisfy callers.
- Key symbols/classes/functions: `Client`, `get_cli`.
- External dependencies: `absl.logging`, `threading` (imported but unused).
- Side effects: None.

**Required Behavior (Detailed)**
- `Client.__init__(*args, **kwargs)`:
  - No-op constructor; ignores all args/kwargs.
- `Client.__getattr__(name)`:
  - Returns a function `method(*args, **kwargs)` that does nothing and returns `None`.
  - Allows arbitrary attribute access without raising `AttributeError`.
- `get_cli(*args, **kwargs)`:
  - Returns a new `Client()`; ignores args/kwargs.
- No logging, no threads, no I/O.

**Rust Mapping (Detailed)**
- Target crate/module: N/A (stub).
- Rust public API surface: optional `NoopClient` with methods that accept any inputs and do nothing.
- Data model mapping: none.
- Feature gating: none.
- Integration points: callers expecting a CLI client can receive a stub.

**Implementation Steps (Detailed)**
1. If Rust needs a CLI client, add a no-op struct with methods used by callers.
2. Ensure missing method calls do not panic (mirror Python `__getattr__` permissiveness).

**Tests (Detailed)**
- Python tests: none.
- Rust tests: optional smoke test that unknown method calls are no-ops if implemented.
- Cross-language parity test: not needed (stub behavior only).

**Gaps / Notes**
- This is a pure stub; threading/logging imports are unused.

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

### `monolith/native_training/metric/deep_insight_ops.py`
<a id="monolith-native-training-metric-deep-insight-ops-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 134
- Purpose/role: Thin wrappers around custom Monolith Deep Insight TF ops (create client, write metrics).
- Key symbols/classes/functions: `deep_insight_client`, `write_deep_insight`, `write_deep_insight_v2`, `deep_insight_ops`.
- External dependencies: TensorFlow, `gen_monolith_ops` (custom ops), `socket.gethostname()`.
- Side effects: Custom ops create a Deep Insight client resource and emit metrics to a databus; may dump to file if `dump_filename` is provided by the op.

**Required Behavior (Detailed)**
- Constants: `_FEATURE_REQ_TIME`, `_SAMPLE_RATE`, `_UID` are defined but unused in this file.
- `deep_insight_client(enable_metrics_counter=False, is_fake=False, dump_filename=None, container=socket.gethostname())`:
  - Default `container` is evaluated at import time (module load), not at call time.
  - Calls `deep_insight_ops.monolith_create_deep_insight_client(enable_metrics_counter, is_fake, dump_filename, container)`.
  - Returns a `tf.Tensor` handle to the client resource.
- `write_deep_insight(...)`:
  - Args:
    - `deep_insight_client_tensor`: handle from `deep_insight_client`.
    - `uids`: 1-D int64 tensor.
    - `req_times`: 1-D int64 tensor.
    - `labels`, `preds`, `sample_rates`: 1-D float tensors.
    - `model_name`, `target`: strings.
    - `sample_ratio` float (default 0.01).
    - `return_msgs` bool: whether op returns serialized messages.
    - `use_zero_train_time` bool: if True uses 0 as train time (tests).
  - Calls `monolith_write_deep_insight` op with named args and returns 1-D string tensor.
- `write_deep_insight_v2(...)`:
  - Args:
    - `req_times`: 1-D int64 tensor (batch_size).
    - `labels`, `preds`, `sample_rates`: 2-D float tensors of shape `(num_targets, batch_size)`.
    - `extra_fields_values`: list of 1-D tensors (each batch_size).
    - `extra_fields_keys`: list of strings, same length as `extra_fields_values`.
    - `targets`: list of strings (num_targets).
  - Calls `monolith_write_deep_insight_v2` op with named args and returns 1-D string tensor.
- No Python-side validation of shapes/dtypes; relies on op validation.
- Threading/concurrency: op-level; Python wrapper is pure.
- Determinism: depends on external Deep Insight system; no RNG here.
- Logging/metrics: metrics emission happens inside the custom op.

**Rust Mapping (Detailed)**
- Target crate/module: N/A (custom TF ops not bound in Rust).
- Rust public API surface: optional wrappers when TF-runtime backend is present.
- Data model mapping: Tensor handles and string vectors must map to TF runtime types.
- Feature gating: TF-runtime + custom ops only.
- Integration points: training metrics pipeline, databus output.

**Implementation Steps (Detailed)**
1. Expose `monolith_create_deep_insight_client` and write ops in Rust TF-runtime backend (FFI).
2. Mirror function signatures and defaults (especially container default semantics).
3. Add validation if desired, but preserve op behavior for parity.
4. Add tests using fake client mode (`is_fake=True`) to avoid external dependencies.

**Tests (Detailed)**
- Python tests: `monolith/native_training/metric/deep_insight_ops_test.py`.
- Rust tests: N/A until TF custom ops are bound.
- Cross-language parity test: compare returned messages (if `return_msgs=True`) and shape/dtype.

**Gaps / Notes**
- Requires custom TF ops; no Rust bindings exist today.

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

### `monolith/native_training/metric/deep_insight_ops_test.py`
<a id="monolith-native-training-metric-deep-insight-ops-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 33
- Purpose/role: Placeholder test module for Deep Insight ops; currently no assertions.
- Key symbols/classes/functions: `DeepInsightOpsTest.dummy_test`.
- External dependencies: TensorFlow test harness, `absl.logging`, `json`, `time` (unused).
- Side effects: None.

**Required Behavior (Detailed)**
- `DeepInsightOpsTest.dummy_test`: no-op; test does nothing.
- `__main__` block disables eager execution and runs `tf.test.main()`.
- No validation of deep insight ops; effectively a stub.

**Rust Mapping (Detailed)**
- Target crate/module: N/A (empty test).
- Rust public API surface: none.
- Data model mapping: none.
- Feature gating: none.
- Integration points: none.

**Implementation Steps (Detailed)**
1. If Deep Insight ops are implemented in Rust, add real tests; otherwise keep as stub-equivalent.

**Tests (Detailed)**
- Python tests: `monolith/native_training/metric/deep_insight_ops_test.py` (no assertions).
- Rust tests: none.
- Cross-language parity test: not applicable until ops exist.

**Gaps / Notes**
- Tests are effectively empty; add real assertions when ops are implemented.

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

### `monolith/native_training/metric/exit_hook.py`
<a id="monolith-native-training-metric-exit-hook-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 48
- Purpose/role: Installs signal handlers and an `atexit` hook that emits an "exit_hook" counter when the process exits due to a signal.
- Key symbols/classes/functions: `sig_no`, `sig_handler`, `exit_hook`.
- External dependencies: `atexit`, `signal`, `sys`, `monolith.native_training.utils`, `native_task_context`, `metric.cli`.
- Side effects: Registers signal handlers on import; registers an `atexit` handler; may call `sys.exit` on signal.

**Required Behavior (Detailed)**
- Module global `sig_no` initialized to `None`.
- `sig_handler(signo, frame)`:
  - Sets global `sig_no = signo`.
  - Calls `sys.exit(signo)` to terminate process.
- On import, installs handlers:
  - `signal.signal(signal.SIGHUP, sig_handler)`
  - `signal.signal(signal.SIGINT, sig_handler)`
  - `signal.signal(signal.SIGTERM, sig_handler)`
- `exit_hook()` (decorated with `@atexit.register`):
  - Fetches context via `native_task_context.get()`.
  - Builds metric client: `cli.get_cli(utils.get_metric_prefix())`.
  - `index = ctx.worker_index` if `ctx.server_type == 'worker'`, else `ctx.ps_index`.
  - Builds tags: `server_type`, `index` (string), `sig` (stringified `sig_no`).
  - Only emits counter if `sig_no is not None`: `mcli.emit_counter("exit_hook", 1, tags)`.
- No explicit error handling; depends on `native_task_context` and `cli` behavior.
- Determinism: signal arrival timing; otherwise deterministic.
- Logging/metrics: emits a counter metric when terminating due to signal.

**Rust Mapping (Detailed)**
- Target crate/module: N/A (no Rust signal/exit hook yet).
- Rust public API surface: optional `exit_hook` module that registers signal handlers + exit hook.
- Data model mapping: tag map `server_type/index/sig` -> metrics client.
- Feature gating: none; only needed if Rust training/runtime needs parity.
- Integration points: metrics client (Rust equivalent of `cli.get_cli`), task context.

**Implementation Steps (Detailed)**
1. Implement signal handling in Rust (e.g., `signal-hook`) for HUP/INT/TERM.
2. Record the signal number in a global and trigger process exit.
3. Register an exit hook that emits `exit_hook` counter with identical tags.
4. Mirror `server_type`/index selection from `native_task_context`.

**Tests (Detailed)**
- Python tests: none.
- Rust tests: optional integration test using signal simulation.
- Cross-language parity test: validate emitted tags and counter name.

**Gaps / Notes**
- Python `cli.get_cli` is a stub; metric emission may be a no-op unless replaced.

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
