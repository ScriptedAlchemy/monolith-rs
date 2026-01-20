<!--
Source: task/request.md
Lines: 10669-10934 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/gflags_utils.py`
<a id="monolith-native-training-gflags-utils-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 282
- Purpose/role: Utilities to extract flags from dataclass docstrings and link flags to dataclass defaults.
- Key symbols/classes/functions: `extract_help_info`, `extract_flags`, `extract_flags_decorator`, `update`, `LinkDataclassToFlags`, `update_by_flags`.
- External dependencies: `absl.flags`, `dataclasses`, `Enum`, `inspect`, `re`.
- Side effects: Defines gflags and mutates dataclass instances based on flags.

**Required Behavior (Detailed)**
- `extract_help_info` parses `:param` lines and returns normalized help strings.
- `extract_flags` defines flags for type-hinted fields (int/bool/str/float/enum) with defaults; skips missing help or skip list.
- `extract_flags_decorator` returns decorator that calls `extract_flags`.
- `get_flags_parser` returns parser that logs errors and exits on invalid flags.
- `update` applies flags to config when config field is default and flag is non-default.
- `LinkDataclassToFlags` validates fields/flags and records mappings.
- `update_by_flags` patches `__init__` to apply linked flags when field is default.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-cli/src/gflags_utils.rs` (new) or `monolith-core` config utilities.
- Rust public API surface: config flag extraction and update helpers.
- Feature gating: CLI only.

**Implementation Steps (Detailed)**
1. Implement help metadata parsing (or explicit metadata in Rust).
2. Provide flag registration for primitive and enum types.
3. Implement update logic that respects defaults vs overrides.
4. Provide helper to link flags to dataclass fields.

**Tests (Detailed)**
- Python tests: `gflags_utils_test.py`.
- Rust tests: unit tests for help parsing and update/flag linking behavior.
- Cross-language parity test: compare config updates for equivalent flags.

**Gaps / Notes**
- Python relies on docstring parsing; Rust likely needs explicit metadata.

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

### `monolith/native_training/gflags_utils_test.py`
<a id="monolith-native-training-gflags-utils-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 217
- Purpose/role: Tests help parsing, flag extraction, config update, and flag linking.
- Key symbols/classes/functions: `GflagUtilsTest`.
- External dependencies: `absl.flags`, `absltest`, `gflags_utils`.
- Side effects: Defines flags in test scope.

**Required Behavior (Detailed)**
- `test_extract_help_info`: parses `:param` lines and joins multi-line help.
- `test_update`: updates config fields only when default and flag is non-default.
- `test_extract_gflags_decorator`: ensures flags are defined for decorated dataclasses and skipped for removed/base fields.
- `test_link_flag` / `test_link_flag_inheritance`: validates linked flags override defaults and inheritance behavior.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-cli/tests/gflags_utils_test.rs` (new).
- Rust public API surface: flag extraction and linking utilities.
- Feature gating: CLI only.

**Implementation Steps (Detailed)**
1. Add tests for help parsing and update logic.
2. Add tests for linked flags and inheritance behavior.

**Tests (Detailed)**
- Python tests: `gflags_utils_test.py`.
- Rust tests: `monolith-rs/crates/monolith-cli/tests/gflags_utils_test.rs`.
- Cross-language parity test: compare flag update outcomes.

**Gaps / Notes**
- Flag registration order in Rust may differ; ensure deterministic behavior.

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

### `monolith/native_training/graph_meta.py`
<a id="monolith-native-training-graph-meta-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 30
- Purpose/role: Stores per-graph metadata in a TF collection.
- Key symbols/classes/functions: `get_meta`.
- External dependencies: TensorFlow.
- Side effects: Mutates graph collections.

**Required Behavior (Detailed)**
- Uses graph collection `monolith_graph_meta` to store a single dict.
- If key missing, calls `MetaFactory()` and stores result.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-tf/src/graph_meta.rs` (new).
- Rust public API surface: graph metadata helper.
- Feature gating: TF runtime only.

**Implementation Steps (Detailed)**
1. Implement get-or-create metadata storage keyed in graph collection.
2. Mirror single-dict behavior.

**Tests (Detailed)**
- Python tests: none.
- Rust tests: add unit test for get_meta caching.

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

### `monolith/native_training/graph_utils.py`
<a id="monolith-native-training-graph-utils-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 26
- Purpose/role: Adds batch-norm moving average assign ops into TF UPDATE_OPS collection.
- Key symbols/classes/functions: `add_batch_norm_into_update_ops`.
- External dependencies: TensorFlow.
- Side effects: Mutates default graph collections.

**Required Behavior (Detailed)**
- `add_batch_norm_into_update_ops()`:
  - Scans default graph operations.
  - Selects ops where `"AssignMovingAvg"` is in the op name and `op.type == "AssignSubVariableOp"`.
  - Adds each to `tf.GraphKeys.UPDATE_OPS` collection.

**Rust Mapping (Detailed)**
- Target crate/module: TF runtime utility module (e.g., `monolith-rs/crates/monolith-tf/src/graph_utils.rs`).
- Rust public API surface: `add_batch_norm_into_update_ops` equivalent for TF graphs.
- Feature gating: `tf-runtime` only.
- Integration points: training graph construction when using batch norm layers.

**Implementation Steps (Detailed)**
1. Implement graph op scan in TF runtime bindings.
2. Filter ops by name substring and op type.
3. Add ops to UPDATE_OPS collection.

**Tests (Detailed)**
- Python tests: none.
- Rust tests: add a small TF graph with batch norm and verify UPDATE_OPS contains moving average assigns.
- Cross-language parity test: compare the count of added ops for a known graph.

**Gaps / Notes**
- No-op for Candle backend; document as TF-only behavior.

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

### `monolith/native_training/hash_filter_ops.py`
<a id="monolith-native-training-hash-filter-ops-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 217
- Purpose/role: Builds hash filter resources, intercepts gradients, and handles save/restore for hash filters.
- Key symbols/classes/functions: `FilterType`, `create_hash_filters`, `save_hash_filter`, `restore_hash_filter`, `intercept_gradient`, `HashFilterCheckpointSaverListener`, `HashFilterCheckpointRestorerListener`.
- External dependencies: TensorFlow custom ops `gen_monolith_ops`, `save_utils.SaveHelper`, `basic_restore_hook`, `utils.ps_device`.
- Side effects: Creates TF resources, writes checkpoint assets, registers gradient for custom op.

**Required Behavior (Detailed)**
- Constants:
  - `HASH_FILTER_CAPACITY=300000000`, `HASH_FILTER_SPLIT_NUM=7`, `_TIMEOUT_IN_MS=1800000`.
- `FilterType`: string constants `SLIDING_HASH_FILTER`, `PROBABILISTIC_FILTER`, `NO_FILTER`.
- `create_hash_filter(capacity, split_num, config, name_suffix)`:
  - Calls `MonolithHashFilter` custom op with shared_name `MonolithHashFilter<suffix>`.
- `create_probabilistic_filter(equal_probability, config, name_suffix)`:
  - Calls `MonolithProbabilisticFilter` op with shared_name `MonolithProbabilisticFilter<suffix>`.
- `create_dummy_hash_filter(name_suffix)`:
  - Calls `MonolithDummyHashFilter` op with shared_name `DummyHashFilter<suffix>`.
- `_create_hash_filter(...)`:
  - Selects real or dummy filter based on `enable_hash_filter` and `filter_type`; invalid type raises `ValueError`.
- `create_hash_filters(ps_num, enable_hash_filter, ...)`:
  - If `ps_num==0`, returns a single filter.
  - Else, for each PS index, creates filter on `utils.ps_device(i)` unless exporting standalone.
- `save_hash_filter(hash_filter, basename, enable_hash_filter)`:
  - If enabled, uses `monolith_hash_filter_save` custom op; else returns `tf.no_op()`.
- `restore_hash_filter(hash_filter, basename, enable_hash_filter)`:
  - If enabled, uses `monolith_hash_filter_restore` custom op; else returns `tf.no_op()`.
- `intercept_gradient(filter_tensor, ids, embeddings)`:
  - Calls `MonolithHashFilterInterceptGradient`; filters gradients based on ids.
- `HashFilterCheckpointSaverListener`:
  - Builds save graph with placeholders per hash filter; writes to asset dir using SaveHelper.
  - `before_save` writes to `hash_filter_<ps_idx>` files under asset dir and runs save op with timeout.
- `HashFilterCheckpointRestorerListener`:
  - Looks up latest checkpoint; restores from asset dir or legacy prefix.
  - Uses placeholders for hash_filter basenames and runs restore op with timeout.
- Registered gradient `MonolithHashFilterInterceptGradient`:
  - Uses `MonolithHashFilterInterceptGradientGradient` custom op to produce filtered gradients; returns `(None, None, filtered_grad)`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-tf/src/hash_filter_ops.rs` (new) or TF runtime adapter.
- Rust public API surface: hash filter creation, save/restore, and gradient intercept wrappers.
- Feature gating: `tf-runtime` + custom ops required; no-op in Candle backend.
- Integration points: hash filter use in embedding tables and training loops.

**Implementation Steps (Detailed)**
1. Wrap custom ops for hash filter creation and gradient intercept.
2. Implement save/restore helpers using TF session run with timeouts.
3. Add Rust equivalents of saver/restorer listeners if using TF Estimator.
4. Ensure asset dir layout matches Python (`hash_filter_<ps_idx>` files).

**Tests (Detailed)**
- Python tests: `monolith/native_training/hash_filter_ops_test.py`.
- Rust tests: integration tests under `monolith-rs/crates/monolith-tf/tests/hash_filter_ops_test.rs` (new).
- Cross-language parity test: compare gradient filtering behavior and save/restore contents.

**Gaps / Notes**
- Requires custom ops from `libmonolith_ops` and TF runtime; not supported on pure Candle backend.

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
