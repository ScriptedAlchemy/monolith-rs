<!--
Source: task/request.md
Lines: 11735-12027 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/hooks/hook_utils_test.py`
<a id="monolith-native-training-hooks-hook-utils-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 35
- Purpose/role: Smoke test for BeforeSaveListener and AfterSaveListener wrappers.
- Key symbols/classes/functions: `HookUtilsTest.testBeforeAfterSaverListener`.
- External dependencies: TensorFlow, `hook_utils`.
- Side effects: None.

**Required Behavior (Detailed)**
- Wraps a base `CheckpointSaverListener` and calls before/after save methods to ensure no errors.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests/hook_utils_test.rs` (new).
- Rust public API surface: hook wrapper types.
- Feature gating: TF runtime.

**Implementation Steps (Detailed)**
1. Instantiate wrapper types around a dummy listener.
2. Invoke before/after save methods and ensure no panic.

**Tests (Detailed)**
- Python tests: `hook_utils_test.py`.
- Rust tests: `monolith-rs/crates/monolith-training/tests/hook_utils_test.rs`.
- Cross-language parity test: not required beyond smoke test.

**Gaps / Notes**
- This is a compile/smoke test only.

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

### `monolith/native_training/hooks/ps_check_hooks.py`
<a id="monolith-native-training-hooks-ps-check-hooks-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 97
- Purpose/role: Health-check hooks for PS machines, reporting failures and placing barriers.
- Key symbols/classes/functions: `PsHealthCheckerHook`, `Config`, `get_ps_machine_info_shared_name`.
- External dependencies: TensorFlow, `logging_ops`, `barrier_ops`, `logging_ops_pb2`.
- Side effects: Spawns background thread, places barrier on failure, logs error details.

**Required Behavior (Detailed)**
- `get_ps_machine_info_shared_name(index)` returns `"ps_machine_info_<index>"`.
- `_default_report(results)`:
  - Logs per-PS MachineHealthResult using text_format one-line strings.
- `Config`:
  - Contains `barrier_op`, `num_ps`, `ps_device_fn` (default `utils.ps_device`), `report_fn` (default `_default_report`).
- `_PsHealthChecker`:
  - Builds `machine_info` and `check_machine_health` ops per PS device.
  - Runs in a daemon thread registered with coordinator.
  - If any status is non-empty, parses `MachineHealthResult`, calls report_fn, places barrier, and waits for stop.
  - Sleeps/waits via `coord.wait_for_stop(timeout=30)` in loop.
- `PsHealthCheckerHook`:
  - Creates checker on `begin()` and starts thread in `after_create_session`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/hooks/ps_check_hooks.rs` (new).
- Rust public API surface: PS health checker hook with background polling.
- Feature gating: TF runtime/custom ops required for machine health ops.
- Integration points: barrier ops and logging/alerting system.

**Implementation Steps (Detailed)**
1. Wrap logging_ops machine_info and health check ops in Rust TF runtime.
2. Spawn a background thread to poll health status.
3. On failure, call report_fn and place barrier, then request stop.

**Tests (Detailed)**
- Python tests: `ps_check_hooks_test.py`.
- Rust tests: `monolith-rs/crates/monolith-training/tests/ps_check_hooks_test.rs` (new).
- Cross-language parity test: simulate healthy vs OOM conditions and verify report hook invocation.

**Gaps / Notes**
- Health status is encoded as serialized proto bytes; Rust must parse with `logging_ops_pb2` equivalent.

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

### `monolith/native_training/hooks/ps_check_hooks_test.py`
<a id="monolith-native-training-hooks-ps-check-hooks-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 112
- Purpose/role: Tests PS health checker hook and error handling.
- Key symbols/classes/functions: `PsCheckHooksTest` cases.
- External dependencies: TensorFlow, `ps_check_hooks`, `logging_ops`.
- Side effects: Uses monitored sessions and sleeps briefly.

**Required Behavior (Detailed)**
- `test_basic`:
  - Healthy machine info should not trigger report.
- `test_oom`:
  - mem_limit=0 triggers report_fn once.
- `test_raise_in_after_create_session` / `test_raise_in_before_run`:
  - Raising in hooks should propagate DeadlineExceededError.
- `test_default_report`:
  - Calls `_default_report` with a MachineHealthResult for smoke coverage.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests/ps_check_hooks_test.rs` (new).
- Rust public API surface: PS health checker hook.
- Feature gating: TF runtime/custom ops required.

**Implementation Steps (Detailed)**
1. Add a test harness that simulates healthy and unhealthy machine_info results.
2. Assert report function called under unhealthy case.
3. Verify exceptions propagate from hook callbacks.

**Tests (Detailed)**
- Python tests: `ps_check_hooks_test.py`.
- Rust tests: `monolith-rs/crates/monolith-training/tests/ps_check_hooks_test.rs`.
- Cross-language parity test: compare report invocation counts.

**Gaps / Notes**
- Tests rely on custom logging_ops; may need to skip if ops unavailable.

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

### `monolith/native_training/hooks/server/client_lib.py`
<a id="monolith-native-training-hooks-server-client-lib-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 30
- Purpose/role: gRPC client helper to connect to controller server from model_dir.
- Key symbols/classes/functions: `get_stub_from_model_dir`.
- External dependencies: `grpc`, TensorFlow gfile, generated `service_pb2_grpc`.
- Side effects: Reads controller server address file.

**Required Behavior (Detailed)**
- Reads `<model_dir>/controller_server_addr.txt`.
- Creates `grpc.insecure_channel(addr)` and returns `ControllerStub`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/hooks/server/client_lib.rs` (new).
- Rust public API surface: helper to read addr file and create gRPC client.
- Feature gating: gRPC required.

**Implementation Steps (Detailed)**
1. Read server addr file from model_dir.
2. Create gRPC channel and Controller client stub.

**Tests (Detailed)**
- Python tests: `server_lib_test.py`.
- Rust tests: `monolith-rs/crates/monolith-training/tests/server_lib_test.rs` (integration).
- Cross-language parity test: ensure client can connect to server hook.

**Gaps / Notes**
- Assumes address file is present and readable.

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

### `monolith/native_training/hooks/server/constants.py`
<a id="monolith-native-training-hooks-server-constants-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 15
- Purpose/role: Defines filename for controller server address.
- Key symbols/classes/functions: `SERVER_ADDR_FILENAME`.
- External dependencies: None.
- Side effects: None.

**Required Behavior (Detailed)**
- `SERVER_ADDR_FILENAME = "controller_server_addr.txt"`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/hooks/server/constants.rs` (new).
- Rust public API surface: constant for addr file name.

**Implementation Steps (Detailed)**
1. Define `SERVER_ADDR_FILENAME` constant.

**Tests (Detailed)**
- Python tests: covered indirectly in `server_lib_test.py`.
- Rust tests: none required beyond integration tests.

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

### `monolith/native_training/hooks/server/server_lib.py`
<a id="monolith-native-training-hooks-server-server-lib-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 95
- Purpose/role: gRPC controller server for training control (stop/resume/save/status).
- Key symbols/classes/functions: `ControllerServicer`, `ServerHook`.
- External dependencies: gRPC, TensorFlow, `barrier_ops`, `save_utils`, `net_utils`.
- Side effects: Starts a gRPC server, writes address file, triggers barrier ops and checkpoints.

**Required Behavior (Detailed)**
- `ControllerServicer`:
  - `StopTraining`: places barrier; if already placed, aborts with ALREADY_EXISTS.
  - `ResumeTraining`: removes barrier.
  - `GetBlockStatus`: returns blocked and unblocked indices for barrier.
  - `SaveCheckpoint`: calls `saver_hook.trigger_save`.
  - `GetTrainingStatus`: returns global_step from session.
- `ServerHook`:
  - On `after_create_session`, starts gRPC server on ephemeral port, writes addr to `<model_dir>/controller_server_addr.txt`.
  - On `end`, stops server (grace 20s).

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/hooks/server/server_lib.rs` (new).
- Rust public API surface: Controller gRPC service + ServerHook.
- Feature gating: gRPC required; TF runtime for SessionRunHook lifecycle.
- Integration points: barrier ops, checkpoint saver hooks, training global_step.

**Implementation Steps (Detailed)**
1. Define gRPC service with Stop/Resume/Status/Save endpoints.
2. Start server in hook after session creation; write addr file.
3. Implement barrier operations and save trigger forwarding.

**Tests (Detailed)**
- Python tests: `server_lib_test.py`.
- Rust tests: `monolith-rs/crates/monolith-training/tests/server_lib_test.rs`.
- Cross-language parity test: issue gRPC commands and verify barrier state changes.

**Gaps / Notes**
- Uses `net_utils.get_local_server_addr` for address formatting; Rust should mirror semantics.

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
