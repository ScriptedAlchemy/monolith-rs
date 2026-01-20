<!--
Source: task/request.md
Lines: 7675-7924 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/data/transform/transforms_test.py`
<a id="monolith-native-training-data-transform-transforms-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 70
- Purpose/role: Smoke tests that build transform configs and log the resulting protobufs; no assertions beyond successful construction.
- Key symbols/classes/functions: `TransformsTest`, test methods for each transform type.
- External dependencies: `transforms` module, `absl.logging`, `unittest`.
- Side effects: logs serialized proto configs.

**Required Behavior (Detailed)**
- `test_filter_by_fid`: builds `FilterByFid(has_fids=[1], filter_fids=[2,3], select_fids=None)` and logs proto.
- `test_filter_by_action`: builds `FilterByAction(has_actions=[4])` and logs proto.
- `test_filter_by_label`: builds `FilterByLabel(thresholds=[-100, -100])` and logs proto.
- `test_add_label`: builds `AddLabel(config='1,2:3:1.0;4::0.5', negative_value=0.0, new_sample_rate=0.3)` and logs proto.
- `test_logical_or`: builds `LogicalOr(FilterByAction([1,2]), FilterByFid([10000000]))` and logs proto.
- `test_compose`: builds `Compose([...])` with multiple transforms including `LogicalOr`, logs proto.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/tests`.
- Rust public API surface: transform builders and `as_proto` serialization.
- Data model mapping: Rust transforms → TransformConfig protobufs.
- Feature gating: none.
- Integration points: verifies transform config serialization used by `TransformDataset`.

**Implementation Steps (Detailed)**
1. Add Rust tests that construct equivalent transforms and ensure `as_proto` succeeds.
2. Optionally compare serialized proto bytes to Python output for deterministic configs.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: `transforms_test.rs` with equivalent constructions.
- Cross-language parity test: serialize each config in both languages and compare bytes.

**Gaps / Notes**
- Tests are smoke-only; Rust should at least mirror construction and serialization.

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

### `monolith/native_training/data/utils.py`
<a id="monolith-native-training-data-utils-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 55
- Purpose/role: Simple slot/feature-name helpers for training_instance parsing; global mapping of feature names to slots with TOB env toggle.
- Key symbols/classes/functions: `enable_tob_env`, `get_slot_feature_name`, `get_slot_from_feature_name`, `register_slots`, globals `TOBENV`, `USED_FREATUE_NAMES`, `NAME_TO_SLOT`.
- External dependencies: none.
- Side effects: mutates global dictionaries for name/slot mapping.

**Required Behavior (Detailed)**
- Globals:
  - `TOBENV` default `False` toggles slot name prefix (`slot_` vs `fc_slot_`).
  - `USED_FREATUE_NAMES` maps arbitrary feature names to assigned slot ids (incrementing).
  - `NAME_TO_SLOT` maps feature name → slot id (explicit).
- `enable_tob_env()`:
  - Sets `TOBENV = True` globally.
- `get_slot_feature_name(slot)`:
  - Returns `"fc_slot_{slot}"` if `TOBENV` else `"slot_{slot}"`.
- `get_slot_from_feature_name(feature_name)`:
  - If in `NAME_TO_SLOT`, return mapped slot.
  - Else if name starts with `slot_` or `fc_slot_`, parse suffix int; return int or `None` if non-numeric.
  - Else use `USED_FREATUE_NAMES`: assign a new slot id (`len+1`) if missing and return it.
- `register_slots(sparse_features)`:
  - Accepts list/tuple of ints or dict name→slot.
  - For list: asserts ints and converts to dict via `get_slot_feature_name`.
  - Updates `NAME_TO_SLOT` with provided mapping.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/src` (feature utils).
- Rust public API surface: slot/feature-name mapping utilities with global registry or context-bound mapping.
- Data model mapping: feature names → slots used by parsing/feature extraction.
- Feature gating: TOB env toggle.
- Integration points: `parse_instance_ops` (fidv1 slot naming), feature list parsing.

**Implementation Steps (Detailed)**
1. Implement a global or context-local registry for `NAME_TO_SLOT` and `USED_FEATURE_NAMES` with deterministic assignment.
2. Provide `enable_tob_env` toggle and `get_slot_feature_name` logic.
3. Mirror `get_slot_from_feature_name` fallback behavior for unknown names.
4. Implement `register_slots` with list/dict handling and type checks.

**Tests (Detailed)**
- Python tests: none explicit.
- Rust tests: add unit tests for TOB/non-TOB naming and deterministic slot assignment.
- Cross-language parity test: compare mapping outputs for a fixed sequence of names.

**Gaps / Notes**
- Uses global mutable state; Rust must be careful about concurrency or test isolation.

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

### `monolith/native_training/debugging/debugging_client.py`
<a id="monolith-native-training-debugging-debugging-client-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 98
- Purpose/role: CLI client to query debugging server endpoints for variable values or feature embeddings.
- Key symbols/classes/functions: `main`, CLI flags `type`, `variable_names`, `feature_ids`, `feature_name`, `feature_names`.
- External dependencies: `requests`, `json`, protobuf `text_format`, `embedding_hash_table_pb2.EntryDump`.
- Side effects: HTTP POSTs to local debugging server; logs results; raises exceptions on invalid flag combos.

**Required Behavior (Detailed)**
- Flags:
  - `--type` must be `debugging_variables` or `debugging_features`.
  - `--variable_names` list for variable lookup.
  - `--feature_ids` list for feature lookup.
  - `--feature_name` single name to pair with all ids.
  - `--feature_names` list of names; must be same length as `feature_ids` if provided.
- `debugging_variables` flow:
  - If `variable_names` empty → log and return.
  - POST JSON `{"variable_names": [...]}` to `http://127.0.0.1:<port>/debugging/variables`.
  - Response JSON contains `STATUS`, `SUCCESS/FAIL`, `MSG` keys; on FAIL log reason and return.
  - `MSG` is JSON-encoded dict name→value; logs each variable value or "Not exist".
- `debugging_features` flow:
  - Disallow providing both `feature_name` and `feature_names`.
  - If `feature_ids` empty → log and return.
  - If `feature_name` set, expand to list same length as ids.
  - Validate `len(feature_names) == len(feature_ids)` else raise.
  - POST JSON `{"feature_names": [...], "feature_ids": [...]}` to `/debugging/features`.
  - On FAIL log reason and return.
  - `MSG` is JSON-encoded dict name→id→textproto of `EntryDump`.
  - If present, parse textproto into `EntryDump` and log; else log "Not exist".
- Script mode: sets logging verbosity INFO, disables eager, and runs app.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/debugging` (or CLI crate).
- Rust public API surface: CLI command for debugging server queries.
- Data model mapping: JSON request/response; `EntryDump` textproto parsing.
- Feature gating: requires debugging server running locally.
- Integration points: `debugging_server.py` endpoints.

**Implementation Steps (Detailed)**
1. Implement a Rust CLI that mirrors flags and validation.
2. POST to `/debugging/variables` and `/debugging/features` with identical JSON payloads.
3. Parse response JSON; for features, parse textproto into `EntryDump` (protobuf text format parser).
4. Match logging output patterns and error handling (exceptions for invalid flags).

**Tests (Detailed)**
- Python tests: none.
- Rust tests: integration tests with a mocked debugging server (or golden responses).
- Cross-language parity test: compare outputs against Python client for same server responses.

**Gaps / Notes**
- Depends on `requests` and protobuf text parsing; Rust needs equivalent libraries.

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

### `monolith/native_training/debugging/debugging_server.py`
<a id="monolith-native-training-debugging-debugging-server-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 217
- Purpose/role: Flask server that exposes debugging endpoints to fetch variable values and feature embeddings from a running training cluster using saved graph metadata.
- Key symbols/classes/functions: `DebuggingWorker`, `create_app`, `/debugging/variables`, `/debugging/features`, `main`.
- External dependencies: Flask, TensorFlow server/meta_graph import, `debugging_info_pb2`, `embedding_hash_table_pb2`, custom ops `monolith_hash_table_lookup_entry`.
- Side effects: spins up TF server, reads model_dir debugging info, imports meta graph, starts Flask server.

**Required Behavior (Detailed)**
- Flags:
  - `--host`, `--port`, `--model_dir` required to bind server and load debugging info.
- `DebuggingWorker(model_dir)`:
  - Reads `DebuggingInfo` proto from `utils.get_debugging_info_file_name(model_dir)`.
  - Starts a local TF server and builds a fake worker cluster where the last worker is the local server; chief/ps addresses come from debugging info.
  - Builds `feature_name_config_map` from debugging info and initializes a `MergedMultiTypeHashTable` with a dummy factory.
  - Creates session config via `cluster_manager.generate_session_config` and imports the meta graph from `utils.get_meta_graph_file_name(model_dir)`.
- `fetch_variables(variable_names)`:
  - Filters requested variables to those present in imported graph; returns dict name → stringified value.
- `fetch_features(feature_names, feature_ids)`:
  - Maps feature names to merged table names via `slot_mapping` in merged table.
  - Buckets by PS index using `fid % num_ps`.
  - For each table/ps index, fetches EntryDump via `monolith_hash_table_lookup_entry`.
  - Parses EntryDump bytes to text proto and returns dict `feature_name -> {fid: textproto}`.
- `create_app()`:
  - Constructs Flask app and a `DebuggingWorker`.
  - `/debugging/variables`: expects JSON `variable_names`; returns `{status, msg}` with msg JSON string.
  - `/debugging/features`: expects JSON `feature_names` + `feature_ids` (same length); returns `{status, msg}`.
  - On exceptions, returns `status=fail` with traceback in msg.
- `main`:
  - Calls `env_utils.setup_hdfs_env()` and runs Flask app.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/debugging` (server) + TF backend bindings for graph/variables.
- Rust public API surface: debugging server binary exposing `/debugging/variables` and `/debugging/features`.
- Data model mapping: DebuggingInfo proto, EntryDump parsing, table lookup by PS index.
- Feature gating: requires TF runtime + custom ops for hash table lookup.
- Integration points: debugging_client CLI, model_dir metadata generation.

**Implementation Steps (Detailed)**
1. Implement a Rust server (e.g., axum/warp) with matching endpoints and JSON payloads.
2. Load DebuggingInfo and meta graph; create TF session with cluster config matching Python.
3. Implement variable lookup and table lookup logic with PS sharding by `fid % num_ps`.
4. Return responses with identical JSON structure and error handling.

**Tests (Detailed)**
- Python tests: none.
- Rust tests: integration tests with mocked debugging info + TF session (if feasible).
- Cross-language parity test: compare responses for same model_dir and queries.

**Gaps / Notes**
- Requires TF runtime and custom ops; Rust implementation may need to shell out or depend on TF C API.

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
