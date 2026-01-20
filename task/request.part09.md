<!--
Source: task/request.md
Lines: 23562-24033 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/yarn_runtime.py`
<a id="monolith-native-training-yarn-runtime-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 127
- Purpose/role: Yarn/Primus runtime helpers for local host resolution and AppMaster gRPC control (kill, finish, savepoint).
- Key symbols/classes/functions: `get_local_host`, `_get_primus_am_host`, `_get_channel`, `maybe_kill_application`, `maybe_finish_application`, `create_primus_save_point`.
- External dependencies: `grpc`, `primus_am_service_pb2(_grpc)`, `net_utils.get_local_ip`, env vars.
- Side effects:
  - Creates/caches gRPC channels.
  - Sends kill/succeed/savepoint requests to AppMaster.
  - Sleeps while waiting for savepoint completion.
  - Logs info/error messages.

**Required Behavior (Detailed)**
- `get_local_host()`:
  - If `CLOUDNATIVE_INET_ADDR` env var exists: use first entry before comma.
  - Else if `YARN_INET_ADDR` exists: use that value.
  - Else: `net_utils.get_local_ip()`.
  - Asserts non-empty result and returns it.
- `_get_primus_am_host()`:
  - If both `PRIMUS_AM_RPC_HOST` and `PRIMUS_AM_RPC_PORT` are set, returns `"host:port"`.
  - Else returns empty string.
- `_get_channel(addr)`:
  - Caches `grpc.insecure_channel(addr)` in `_CHANNEL_MAP`.
  - Returns cached channel for the same addr.
- `maybe_kill_application(reason) -> bool`:
  - If Primus AM host is available:
    - Builds `KillRequest` with `exit_code=1`, `diagnose=reason`, `graceful_shutdown_timeout_ms=20000`.
    - Calls `stub.kill(req, timeout=10)`.
    - Returns `True` on success; `False` on `grpc.RpcError`.
  - If no host: logs “Current framework doesn't support kill. Ignore killing...” and returns `False`.
- `maybe_finish_application()`:
  - If Primus AM host is available:
    - Builds `SucceedRequest` with `graceful_shutdown_timeout_ms=20000`.
    - Calls `stub.succeed(req, timeout=10)`.
    - Returns `True` on success; logs on `grpc.RpcError` (returns `None`).
  - If no host: returns `None` (no log).
- `create_primus_save_point(dst) -> bool`:
  - If Primus AM host is available:
    - Calls `createSavepoint` with `savepoint_dir=dst`.
    - If response `code != 0`, logs error and returns `False`.
    - Else polls `createSavepointStatus` every 5s:
      - `PENDING`/`RUNNING`: sleep and continue.
      - `SUCCEEDED`: log success and return `True`.
      - Any other state: log error and return `False`.
    - On `grpc.RpcError`: logs and returns `False`.
  - If no host: returns `None`.
- Threading/concurrency:
  - `_CHANNEL_MAP` is a global dict without a lock; concurrent callers can race.
- Performance:
  - Savepoint polling is blocking with 5s sleep intervals.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/yarn_runtime.rs` (new).
- Rust public API surface:
  - `get_local_host() -> String`
  - `maybe_kill_application(reason: &str) -> bool`
  - `maybe_finish_application() -> Option<bool>`
  - `create_primus_save_point(dst: &str) -> Option<bool>`
- Data model mapping:
  - gRPC service `primus_am_service.proto` already in `monolith-rs/proto`.
  - Use tonic/grpc-generated client stubs for `AppMasterService`.
- Feature gating:
  - Requires gRPC + proto build; optional if runtime doesn't use Primus.
- Integration points:
  - Distributed training orchestration that needs to terminate or checkpoint jobs.

**Implementation Steps (Detailed)**
1. Generate Rust gRPC client from `primus_am_service.proto`.
2. Implement env-var host resolution (`CLOUDNATIVE_INET_ADDR` > `YARN_INET_ADDR` > `net_utils` equivalent).
3. Add a channel cache (e.g., `DashMap` or `Mutex<HashMap<...>>`) to mirror `_CHANNEL_MAP`.
4. Implement `maybe_kill_application` and `maybe_finish_application` with the same timeout and fields.
5. Implement `create_primus_save_point` with polling loop + 5s sleep.
6. Mirror return semantics (False vs None) or document a deliberate normalization.

**Tests (Detailed)**
- Python tests: `monolith/native_training/yarn_runtime_test.py`.
- Rust tests:
  - `get_local_host_env_override` (CLOUDNATIVE_INET_ADDR + YARN_INET_ADDR cases).
  - gRPC kill/finish/savepoint flow using an in-process gRPC server.
- Cross-language parity test:
  - Simulate the same gRPC server and verify request fields + timeouts.

**Gaps / Notes**
- `maybe_finish_application` does not return `False` on errors; it logs and returns `None`.
- `create_primus_save_point` and `maybe_finish_application` return `None` when no Primus host is configured.

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

### `monolith/native_training/yarn_runtime_test.py`
<a id="monolith-native-training-yarn-runtime-test-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 133
- Purpose/role: Exercises env-based host resolution and gRPC AppMaster controls (kill/finish/savepoint).
- Key symbols/classes/functions: `YarnRuntimeTest`, `test_get_local_host_*`, `test_kill`, `test_finish`, `test_save_primus`.
- External dependencies: gRPC, Primus AM proto stubs, `unittest.mock` for env vars.
- Side effects: Starts in-process gRPC servers (unix: addresses).

**Required Behavior (Detailed)**
- `test_get_local_host_overwrite`:
  - With `YARN_INET_ADDR=1.2.3.4`, `get_local_host()` returns `1.2.3.4`.
- `test_get_local_host_overwrite_by_cloudnative`:
  - With `CLOUDNATIVE_INET_ADDR=1.2.3.4,5.6.7.8`, returns `1.2.3.4`.
- `test_get_local_host_basic`:
  - Calls `get_local_host()` without explicit env overrides (no assertion).
- `test_kill`:
  - Sets `PRIMUS_AM_RPC_HOST=unix`, `PRIMUS_AM_RPC_PORT=test_kill`.
  - Starts gRPC server with `kill` handler; validates `request.diagnose` equals reason.
  - Calls `maybe_kill_application(reason)` and asserts handler invoked.
- `test_finish`:
  - Similar setup; installs `succeed` handler; asserts invoked after `maybe_finish_application()`.
- `test_save_primus`:
  - Sets Primus host/port; installs `createSavepoint` and `createSavepointStatus` handlers.
  - Status handler returns `SUCCEEDED`; `create_primus_save_point(dst)` returns `True`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests/yarn_runtime.rs` (new).
- Rust public API surface: `get_local_host`, `maybe_kill_application`, `maybe_finish_application`, `create_primus_save_point`.
- Data model mapping: use Rust gRPC server/client for `AppMasterService`.
- Feature gating: tests require gRPC + proto build; optional for non-Primus builds.
- Integration points: gRPC server harness for tests (likely tokio + tonic).

**Implementation Steps (Detailed)**
1. Port env override tests for `get_local_host`.
2. Implement a local gRPC server with unary handlers for `kill`, `succeed`, `createSavepoint`, `createSavepointStatus`.
3. Validate request fields (e.g., `diagnose` for kill) and that calls occur.
4. Ensure unix-domain or loopback TCP addressing works in Rust test harness.

**Tests (Detailed)**
- Python tests: `YarnRuntimeTest.test_get_local_host_*`, `.test_kill`, `.test_finish`, `.test_save_primus`.
- Rust tests:
  - `get_local_host_env_overrides`
  - `maybe_kill_application_calls_stub`
  - `maybe_finish_application_calls_stub`
  - `create_primus_save_point_succeeds`
- Cross-language parity test:
  - Compare request fields and call ordering with a mock gRPC server.

**Gaps / Notes**
- Python tests use gRPC addresses like `unix:test_kill`; Rust must support unix-domain channels or adapt tests to TCP.

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

### `monolith/native_training/zk_utils.py`
<a id="monolith-native-training-zk-utils-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 96
- Purpose/role: Zookeeper utilities for host selection (IPv4 vs IPv6), default server list, authenticated Kazoo client, and cleanup of stale ZK paths.
- Key symbols/classes/functions: `_PORT`, `_HOSTS`, `_HOSTS_IPV6`, `is_ipv6_only`, `default_zk_servers`, `MonolithKazooClient`, `clear_zk_path`.
- External dependencies: `kazoo.client.KazooClient`, `env_utils.get_zk_auth_data`, `socket`, `datetime`.
- Side effects:
  - Connects to ZK, mutates nodes (delete paths).
  - Logs informational messages about IP resolution.

**Required Behavior (Detailed)**
- Globals:
  - `_PORT = 2181`.
  - `_HOSTS` and `_HOSTS_IPV6` are defined with IPs, then overwritten to empty lists later (current effective values are empty).
- `is_ipv6_only()`:
  - If any of `MY_HOST_IP`, `MY_POD_IP`, `MY_HOST_IPV6` env vars are set:
    - Treat as “tce/byterec env”.
    - `ipv4_addr` from `MY_HOST_IP` or `MY_POD_IP` (may be `None`).
  - Else:
    - `ipv4_addr = socket.gethostbyname(socket.gethostname())` (except → `None`).
  - Logs `ipv4_addr`, then sets `ipv6_only = not ipv4_addr` and logs the result.
  - Returns `ipv6_only`.
- `default_zk_servers(use_ipv6: bool = False)`:
  - If `use_ipv6` or `is_ipv6_only()`:
    - Returns comma-joined `"[ip]:port"` entries from `_HOSTS_IPV6`.
  - Else:
    - Returns comma-joined `"ip:port"` entries from `_HOSTS`.
  - With current `_HOSTS`/`_HOSTS_IPV6` emptied, returns empty string.
- `MonolithKazooClient`:
  - Subclasses `KazooClient`.
  - If `auth_data` not provided, injects `get_zk_auth_data()` into kwargs.
- `clear_zk_path(zk_server, job_name, force_clear_zk_path)`:
  - Connects with `MonolithKazooClient(zk_server or default_zk_servers())`.
  - `base_path = "/monolith"`, `delta = timedelta(weeks=9)`.
  - `ensure_path(base_path)`.
  - For each child under `/monolith`:
    - `get_children(path, include_data=True)` → stat; if `stat.mtime // 1000 + delta < now`, delete recursively (ignore errors).
  - For current `job_name`:
    - If exists and `force_clear_zk_path`: delete recursively.
    - Else: raise `ValueError("there are [<children>] in monolith zk path")` using current children list.
  - Always stops client in `finally`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/zk_utils.rs` (new).
- Rust public API surface:
  - `is_ipv6_only() -> bool`
  - `default_zk_servers(use_ipv6: bool) -> String`
  - `MonolithZkClient` wrapper (auth inject)
  - `clear_zk_path(zk_server: Option<&str>, job_name: &str, force: bool) -> Result<(), Error>`
- Data model mapping:
  - Kazoo client ↔ Rust ZK client (e.g., `zookeeper` crate).
  - `stat.mtime` (ms) ↔ Rust stat `mtime` (ms).
- Feature gating:
  - ZK client optional for environments without ZK.
- Integration points:
  - Any training components relying on ZK-based coordination or cleanup.

**Implementation Steps (Detailed)**
1. Decide Rust ZK client crate and auth mechanism matching `get_zk_auth_data()`.
2. Implement `is_ipv6_only` using env vars and hostname resolution.
3. Port `default_zk_servers` formatting (IPv6 `[ip]:port`).
4. Implement `clear_zk_path` with the same TTL logic (9 weeks) and error message.
5. Preserve the “ignore delete errors” behavior on stale node cleanup.

**Tests (Detailed)**
- Python tests: none in repo.
- Rust tests:
  - Unit tests for `default_zk_servers` formatting and env-driven `is_ipv6_only`.
  - Integration tests for `clear_zk_path` if ZK test container is available.
- Cross-language parity test:
  - Compare TTL deletion behavior with a mocked ZK server.

**Gaps / Notes**
- `_HOSTS` and `_HOSTS_IPV6` are overwritten to empty lists, so `default_zk_servers` returns empty string by default (likely a sanitization artifact).

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
