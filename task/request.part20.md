<!--
Source: task/request.md
Lines: 5210-5510 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/clip_ops_test.py`
<a id="monolith-native-training-clip-ops-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 92
- Purpose/role: Validates custom clip-by-global-norm against TF behavior and input immutability.
- Key symbols/classes/functions: `ClipOpsTest`, `NormOpsTest`.
- External dependencies: `tensorflow`, `numpy`, `test_util`.
- Side effects: runs GPU-only tests if available.

**Required Behavior (Detailed)**
- `ClipOpsTest._test_clip_by_global_norm`:
  - Runs op on GPU; compares output to TF `clip_by_global_norm` (unless expected provided).
  - Asserts input tensors are not modified in-place.
- `test_clip_by_global_norm`:
  - Covers simple, uneven shapes, no clipping, zero norm, inf -> NaN, and large random grads.
- `NormOpsTest`:
  - `test_it` (GPU only) checks `_global_norm` for inf and known values.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-optimizer/tests`.
- Rust public API surface: clip-by-global-norm tests.
- Data model mapping: use CPU tensors unless GPU backend exists.
- Feature gating: GPU-only tests optional.
- Integration points: optimizer utilities.

**Implementation Steps (Detailed)**
1. Add CPU tests matching the Python expected outputs.
2. Add NaN/inf propagation test.
3. Add randomized large-shape test if memory allows.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: add temp-dir PS file round-trip test.
- Cross-language parity test: not applicable.

**Gaps / Notes**
- Rust uses different discovery stack; PS file persistence may not be needed.

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

### `monolith/native_training/cluster_manager.py`
<a id="monolith-native-training-cluster-manager-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 184
- Purpose/role: Build TF cluster specs for distributed training and manage PS discovery persistence.
- Key symbols/classes/functions: `generate_session_config`, `get_training_cluster`, `_query_ps_cluster`, `_query_chief_addr`, `_save_ps_cluster_to_file`, `_fetch_ps_cluster_from_file`.
- External dependencies: `tensorflow`, `absl.logging`, `ServiceDiscovery`, `metric.cli`.
- Side effects: reads/writes PS cluster file via `tf.io.gfile`, sleeps on retries, emits metrics.

**Required Behavior (Detailed)**
- `emit_store(name, value, tagkv=None)`:
  - Delegates to `_MCLI.emit_store` (metrics).
- `generate_session_config(cluster_and_task=None)`:
  - If `None`, returns `ConfigProto(allow_soft_placement=True)`.
  - Else builds `ClusterSpec` and `ConfigProto(cluster_def=...)` with:
    - `device_filters` including `/job:ps` and `/job:chief`; non-chief adds its own job/task filter.
  - Sets `share_cluster_devices_in_session=True`.
  - Sets `experimental.share_session_state_in_clusterspec_propagation=True`.
  - Disables Grappler meta optimizer.
- `get_training_cluster(...)`:
  - If `index == 0` (chief):
    - If `num_redundant_ps`: try reading PS addrs from file (timeout=0); if insufficient, query discovery then save to file.
    - Else query discovery for PS addrs.
    - Builds cluster with `chief=[worker_addr]`, `worker=fake_worker_list`, `ps=ps_addrs`.
    - `task={"type":"chief","index":0}`.
  - Else (worker):
    - Gets `chief_addr` via `_query_chief_addr`.
    - Builds `fake_worker_list` of size `num_workers-1` and assigns `worker_addr` at `index-1`.
    - PS addrs from file (if redundant) or discovery.
    - `task={"type":"worker","index":index-1}`.
  - Asserts PS count equals `num_required_ps`.
- `_query_chief_addr(discovery)`:
  - Polls `discovery.query("worker")` until index 0 present; sleeps 5s between retries.
- `_query_ps_cluster(discovery, num_required_ps, model_name=None, cluster_type="stable")`:
  - Polls `discovery.query("ps")` until enough PS; logs count and emits metrics if model_name provided.
  - Returns sorted PS addresses by index, truncated to `num_required_ps`.
- `_save_ps_cluster_to_file(file_name, ps_addrs)`:
  - Writes comma-separated list to temp file and atomically renames.
- `_fetch_ps_cluster_from_file(file_name, timeout=1800)`:
  - Repeatedly attempts read; returns list if found, else empty after timeout.
- `_get_ps_cluster_file_name(model_dir, uuid)`:
  - Path: `<model_dir>/ps_cluster_dir/<uuid or "ps_info">`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/distributed.rs`, `py_discovery.rs`.
- Rust public API surface: `ClusterConfig`, `ServiceDiscovery` implementations.
- Data model mapping: TF `ClusterSpec` → Rust `ClusterConfig`; discovery polling → async discovery APIs.
- Feature gating: none.
- Integration points: distributed runner and service discovery.

**Implementation Steps (Detailed)**
1. Map discovery polling to Rust async discovery with retry/backoff.
2. Add PS cluster file persistence helpers (optional).
3. Provide cluster config builder mirroring fake worker list behavior if needed for TF_CONFIG parity.

**Tests (Detailed)**
- Python tests: `monolith/native_training/cluster_manager_test.py`.
- Rust tests: add unit test for PS file round-trip and cluster config validation.
- Cross-language parity test: compare generated cluster dictionaries for sample inputs.

**Gaps / Notes**
- Python uses fake worker list due to TF_CONFIG limitation; Rust cluster config may not need this.

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

### `monolith/native_training/cluster_manager_test.py`
<a id="monolith-native-training-cluster-manager-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 35
- Purpose/role: Tests PS cluster file persistence helpers.
- Key symbols/classes/functions: `ClusterManagerTest.testBasic`.
- External dependencies: `os`, `cluster_manager`.
- Side effects: writes temp files under `TEST_TMPDIR`.

**Required Behavior (Detailed)**
- `testBasic`:
  - Writes PS addrs to file via `_save_ps_cluster_to_file`.
  - Reads via `_fetch_ps_cluster_from_file` and asserts equality.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests`.
- Rust public API surface: PS cluster file helper tests.
- Data model mapping: same string list round-trip.
- Feature gating: none.
- Integration points: optional in discovery integration.

**Implementation Steps (Detailed)**
1. Add Rust test that writes/reads PS cluster file in a temp dir.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: TODO (manual)
- Cross-language parity test: TODO (manual)

**Gaps / Notes**
- TODO (manual)

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

### `monolith/native_training/consul.py`
<a id="monolith-native-training-consul-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 149
- Purpose/role: Minimal Consul client for service lookup/register/deregister using HTTP or Unix socket.
- Key symbols/classes/functions: `Client`, `UnixHTTPConnection`, `ConsulException`.
- External dependencies: `six.moves.http_client.HTTPConnection`, `socket`, `threading`, `json`, `os`.
- Side effects: spawns a health-check thread on register; caches lookup results.

**Required Behavior (Detailed)**
- `UnixHTTPConnection(path)`:
  - Connects via UNIX domain socket to Consul.
- `Client.__init__()`:
  - Determines consul host:
    - `CONSUL_HTTP_HOST` or `TCE_HOST_IP` env vars.
    - Else uses `/opt/tmp/sock/consul.sock` if file exists.
    - Else defaults to `"127.0.0.1"`.
  - Port from `CONSUL_HTTP_PORT` or `2280`.
  - Initializes `_cache` and `_lock`.
- `lookup(name, timeout=3, cachetime=0)`:
  - If `cachetime>0` and cached entry is fresh, returns it.
  - Else uses `_lookup`, with longer timeout (30s) on cache miss.
  - Caches result with timestamp.
- `_lookup(name, timeout)`:
  - Uses Unix socket if host starts with `/`, else TCP.
  - GET `/v1/lookup/name?name=<name>&addr-family=dual-stack`.
  - If status != 200: logs error and returns `[]`.
  - Otherwise returns JSON-decoded list.
- `register(name, port, tags=None, check_script=None, host=None)`:
  - Builds payload with id `<name>-<port>`, TTL check (60s).
  - Adds tags as `["k:v"]`.
  - If `check_script` provided: replaces check with `interval=30s, script=...`.
  - Registers via PUT `/v1/agent/service/register`.
  - On non-200 → raises `ConsulException`.
  - Spawns daemon thread that periodically `GET /v1/agent/check/pass/service:<name>-<port>`; on socket error sleeps 2s and retries.
- `deregister(name, port, host=None)`:
  - PUT `/v1/agent/service/deregister/<name>-<port>`.
  - On non-200 → raises `ConsulException`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/discovery.rs`.
- Rust public API surface: `ConsulDiscovery` (stub), `ServiceDiscovery` traits.
- Data model mapping: Python client → Rust Consul discovery abstraction.
- Feature gating: `consul` feature.
- Integration points: distributed runner discovery.

**Implementation Steps (Detailed)**
1. Implement Consul HTTP client or keep stub and document missing features.
2. Add cache semantics and Unix socket support if parity required.
3. Add optional background health-check ticker.

**Tests (Detailed)**
- Python tests: `monolith/native_training/consul_test.py`.
- Rust tests: add mock HTTP tests for lookup/register/deregister (feature-gated).
- Cross-language parity test: compare HTTP request paths and payloads.

**Gaps / Notes**
- Python uses a ByteDance-specific `/v1/lookup/name` API, not stock Consul.
- Rust `ConsulDiscovery` targets standard Consul catalog APIs; endpoint mismatch.

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

### `monolith/native_training/consul_test.py`
<a id="monolith-native-training-consul-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 59
- Purpose/role: Unit tests for Consul client lookup/register/deregister using mocked HTTPConnection.
- Key symbols/classes/functions: `ConsulTest`.
- External dependencies: `unittest.mock`, `six.moves.http_client.OK`.
- Side effects: none (network mocked).

**Required Behavior (Detailed)**
- `test_lookup`:
  - Mock HTTP 200 with JSON payload; `Client.lookup` returns decoded list.
- `test_register` / `test_deregister`:
  - Mock HTTP 200; call methods without error.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests`.
- Rust public API surface: Consul discovery tests.
- Data model mapping: mock HTTP client behavior.
- Feature gating: `consul`.
- Integration points: discovery subsystem.

**Implementation Steps (Detailed)**
1. Add mocked HTTP tests for lookup/register/deregister under `consul` feature.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: TODO (manual)
- Cross-language parity test: TODO (manual)

**Gaps / Notes**
- TODO (manual)

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
