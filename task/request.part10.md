<!--
Source: task/request.md
Lines: 2644-2953 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
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
- `gen_model_spec` and `gen_model_config` must set fields correctly (name, version, signature).
- `gen_status_proto` should preserve error_code and message.
- `gen_model_version_status` should match version and state.
- `AgentConfig.from_file` should load `agent.conf` and expose values (e.g., `stand_alone_serving`, `layout_filters`).
- `InstanceFormater` should parse json/pbtext/dump and produce TensorProto of correct dtype and shape.
- `get_cmd_and_port` should include `model_config_file_poll_wait_seconds` for agent_version 2.
- `ZKPath` parsing:
  - Full dc-aware path, partial path, and old (non-dc) paths must parse bzid/base_name/idc/cluster/server_type/index/replica_id correctly.

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
- Maintains in-memory `_data` cache of ZK paths to bytes.
- Defines base paths: `resource`, `portal`, `publish`, `service`, `locks`, `election`.
- CRUD helpers (`create`, `set`, `delete`, `exists`, `ensure_path`) wrap ZK and fall back to cache on errors.
- `report_resource`: writes ResourceSpec as ephemeral node.
- `resources` property: returns list of ResourceSpec from cached paths.
- `num_tce_replica`: waits until every replica id appears for all shards; returns count.
- `tce_replica_id`: uses env `REPLICA_ID` or derives from pod name.
- `publish_loadding`: writes PublishMeta entries to publish path; updates cache.
- `expected_loading`:
  - Groups PublishMeta by model_name; selects when all publish nodes have arrived.
  - Adjusts shard_id/replica_id for autoscaler and entry cases; filters sub_models to entry when needed.
- `update_service(replicas)`:
  - Computes paths for local replicas; removes outdated nodes; creates/updates current replicas.
- Replica queries: `get_all_replicas`, `get_model_replicas`, `get_task_replicas`, `get_replica` (AVAILABLE only).
- Watchers:
  - `watch_portal`: ensures portal/publish consistency; installs DataWatch for model meta; emits Event(PORTAL).
  - `watch_publish`: installs watches on publish nodes; emits Event(PUBLISH) when all publish nodes arrive.
  - `watch_resource`: watches resource nodes into cache.
  - `watch_service`: watches service hierarchy into cache.
- `election`: uses Kazoo Election to run leader function; handles reconnects.
- `start(is_client=False)`: starts ZK, watches service, and optionally publish.

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
- `test_crud`:
  - `ensure_path`, `exists`, `create`, `get/set`, `delete` operations.
  - Checks derived properties: `num_tce_shard`, `tce_replica_id`, `tce_shard_id`.
- `test_zk_mirror`:
  - `watch_portal` + `watch_resource`.
  - Portal event should be emitted for new ModelMeta.
  - Scheduler simulation publishes PublishMeta to all shards/replicas.
  - `expected_loading` selects correct PublishMeta for current shard.
  - `update_service` writes ReplicaMeta nodes; verify replica query APIs.
  - `report_resource` and `resources` should roundtrip ResourceSpec.

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
    - `HEAP_PROFILE_SAMPLE_RATIO`, `HEAP_PROFILE_TIME_INTERVAL`, `HEAP_PROFILE_MMAP`.

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
