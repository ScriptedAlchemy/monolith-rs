<!--
Source: task/request.md
Lines: 2375-2643 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
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
- `test_get_instance_proto`: asserts dtype and tensor shape for random instance batch.
- `test_get_example_batch_to_instance_from_pb`: reads binary examplebatch file with header.
- `test_get_example_batch_to_instance_from_pbtxt`: reads pbtxt example batch file.

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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 303
- Purpose/role: gRPC monitor for TensorFlow Serving model status and config reload.
- Key symbols/classes/functions: `TFSMonitor`, `get_model_status` (singledispatch), `gen_model_config`, `handle_reload_config_request`.
- External dependencies: TF Serving gRPC stubs, `ModelServerConfig`, `PublishMeta`.
- Side effects: gRPC calls to TFS servers.

**Required Behavior (Detailed)**
- Maintains gRPC stubs for ENTRY/PS/DENSE based on deploy_type and ports.
- `get_addr(sub_model_name)` chooses port based on deploy type and sub_model type.
- `get_service_type(sub_model_name)` returns TFSServerType or None.
- `get_model_status(PublishMeta)`:
  - For each sub_model, builds `GetModelStatusRequest`.
  - For dense nodes (entry when dense-along-entry), may omit version unless `fix_dense_version`.
  - On RPC errors, returns UNKNOWN with StatusProto error code/details.
  - Returns map `{tfs_model_name: (version_path, ModelVersionStatus)}`.
- `get_model_status(name, version=None, signature_name=None)`:
  - Returns list of ModelVersionStatus for a model via `GetModelStatus`.
- `gen_model_config(pms)`:
  - Builds ModelServerConfig per service type from PublishMeta list.
  - For dense nodes: use `latest` policy unless `fix_dense_version`.
  - For ps/entry: `specific` policy with version number.
- `handle_reload_config_request(service_type, model_configs)`:
  - Ensures default model config is present.
  - Sends ReloadConfigRequest to appropriate service.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/src/tfs_monitor.rs`.
- Rust public API surface: `TFSMonitor` struct with status and reload APIs.
- Data model mapping: TF Serving protos and `PublishMeta` equivalents.

**Implementation Steps (Detailed)**
1. Port gRPC client setup for entry/ps/dense.
2. Implement singledispatch-like overloads (Rust traits/enum arguments).
3. Port model config generation logic and dense version policy.
4. Add default model config injection.

**Tests (Detailed)**
- Python tests: `monolith/agent_service/tfs_monitor_test.py`
- Rust tests: start fake TFS servers and verify reload/status.

**Gaps / Notes**
- Requires TF Serving protos + stubs in Rust.

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
- Setup:
  - Start FakeTFServing for entry and ps ports; wait for readiness.
  - Create `TFSMonitor` and connect.
- `test_reload_config`:
  - Generate PublishMeta list with random ps counts and entry submodel.
  - Call `gen_model_config` then `handle_reload_config_request` per service type.
- `test_remove_config`:
  - Similar to reload config but with different models; ensures reload path can remove models.
- `tearDown`: compares before/after status; ensures NOT_FOUND responses for unloaded models.

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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 202
- Purpose/role: Wraps TensorFlow Serving process launch, config file handling, and model status queries.
- Key symbols/classes/functions: `TFSWrapper`, `FakeTFSWrapper`
- External dependencies: `subprocess`, `grpc`, TF Serving protos.
- Side effects: launches external process, writes logs, opens gRPC channel.

**Required Behavior (Detailed)**
- `TFSWrapper.__init__`:
  - Saves ports, config file, binary config, log path.
  - Uses `strings $TFS_BINARY | grep PredictionServiceGrpc` to detect grpc remote op support.
- `_prepare_cmd()`:
  - Builds CLI flags: model_config_file, ports, poll interval, archon settings, metrics prefix.
  - If grpc remote op absent, adds `archon_entry_to_ps_rpc_timeout`.
  - Fills in defaults from `TfServingConfig` (incl. platform_config_file).
- `start()`:
  - `os.chdir(find_main())` and `subprocess.Popen` with stdout to log file.
  - Creates gRPC channel to `localhost:grpc_port` and ModelServiceStub.
- `stop()`:
  - Closes channel, closes log, kills process.
- `list_saved_models()`:
  - Parses model config file text into `ModelServerConfig` and returns model names.
- `list_saved_models_status()`:
  - For each saved model, calls `GetModelStatus`, selects available version or last, handles RPC errors.
- `FakeTFSWrapper`:
  - No process; reads model_config file and returns AVAILABLE for all models.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/src/tfs_wrapper.rs`.
- Rust public API surface: `TFSWrapper` + `FakeTFSWrapper` for tests.
- Feature gating: `tf-runtime` and `grpc` features.

**Implementation Steps (Detailed)**
1. Port command building logic and TfServingConfig mapping.
2. Implement process spawn + logging.
3. Implement gRPC status queries and model_config parsing.
4. Implement FakeTFSWrapper for tests.

**Tests (Detailed)**
- Python tests: used indirectly by `agent_v3_test`.
- Rust tests: use FakeTFSWrapper to validate list_saved_models/STATUS.

**Gaps / Notes**
- `TFS_BINARY` path and `find_main()` must map correctly in Rust.

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
- Key symbols/classes/functions: `AgentConfig`, `DeployType`, `TFSServerType`, `gen_model_spec`, `gen_model_config`, `make_tensor_proto`, `get_local_ip`, many helpers.
- External dependencies: `tensorflow`, `tensorflow_serving` protos, `protobuf.text_format`, `json`, `socket`, `os`.
- Side effects: overrides `os.path.isabs`; writes platform config files; reads/writes files; inspects env; opens sockets.

**Required Behavior (Detailed)**
- Must preserve ALL defaults in `AgentConfig` and flag parsing (`flags.DEFINE_string('conf', ...)`).
- `AgentConfig.__post_init__` logic is critical for port allocation and layout config.
- `gen_model_spec` and `gen_model_config` must match TF Serving proto semantics.
- `make_tensor_proto` must mirror TF string tensor encoding for PredictRequest inputs.
- `get_local_ip` and port helpers must match network selection logic (IPv4/IPv6).

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
