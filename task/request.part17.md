<!--
Source: task/request.md
Lines: 4352-4654 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/core/model_registry.py`
<a id="monolith-core-model-registry-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 174
- Purpose/role: Global registry for `SingleTaskModelParams` classes with dynamic import helpers.
- Key symbols/classes/functions: `_ModelRegistryHelper`, `RegisterSingleTaskModel`, `GetAllRegisteredClasses`, `GetClass`, `GetParams`.
- External dependencies: `absl.logging`, `model_imports`, `SingleTaskModelParams`.
- Side effects: stores classes in global dict; logs registration; raises on duplicates.

**Required Behavior (Detailed)**
- Registry maps keys to param classes. Key format: `<module>.<ClassName>`; also a shortcut key with `monolith.tasks.` prefix stripped and `params.` removed.
- `_RegisterModel(src_cls)`:
  - Registers both full and shortcut keys.
  - Raises ValueError on duplicate key.
  - Logs module registration once per module.
- `RegisterSingleTaskModel` decorator:
  - Validates subclass of `SingleTaskModelParams` else TypeError.
  - Registers class and returns it.
- `GetAllRegisteredClasses()`:
  - Logs warning if none registered.
- `GetClass(class_key)`:
  - Raises LookupError if missing; logs known models.
- `GetParams(class_key)`:
  - Instantiates model params class, calls `.task()` to get task config.
- Module-level helpers call `model_imports.ImportAllParams` or `ImportParams` before returning registry results.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/src/model_registry.rs`.
- Rust public API surface: registry with registration macro/trait and lookup by string key.

**Implementation Steps (Detailed)**
1. Implement a global registry (static map) for model param factories.
2. Provide a registration macro that enforces trait bounds.
3. Implement lookup with detailed error listing known keys.
4. Replace dynamic import with explicit registration in Rust.

**Tests (Detailed)**
- Python tests: none specific.
- Rust tests: register dummy models, verify duplicate detection and lookup errors.

**Gaps / Notes**
- Shortcut key behavior must be preserved for compatibility with existing names.

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

### `monolith/core/optimizers.py`
<a id="monolith-core-optimizers-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 25
- Purpose/role: Maps string names to TensorFlow v1 optimizer classes.
- Key symbols/classes/functions: `optimizers` dict.
- External dependencies: `tf.compat.v1.train` optimizers.
- Side effects: none.

**Required Behavior (Detailed)**
- `optimizers` contains:
  - `'adagrad'`: `AdagradOptimizer`
  - `'momentum'`: `MomentumOptimizer`
  - `'rmsprop'`: `RMSPropOptimizer`
  - `'adam'`: `AdamOptimizer`

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-optimizer/src/lib.rs`.
- Rust public API surface: map from string to optimizer factory or enum.

**Implementation Steps (Detailed)**
1. Define optimizer registry in Rust with equivalent names.
2. Map to native implementations or TF bindings if used.

**Tests (Detailed)**
- Python tests: none.
- Rust tests: verify lookup table matches expected keys.

**Gaps / Notes**
- Rust may not support all TF optimizers natively; document fallbacks.

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

### `monolith/core/py_utils.py`
<a id="monolith-core-py-utils-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 313
- Purpose/role: NestedMap container with attribute access, flatten/pack utilities, and validation.
- Key symbols/classes/functions: `NestedMap` and its methods.
- External dependencies: `six`, `re`, `numpy` (imported but unused here).
- Side effects: enforces key validation on set.

**Required Behavior (Detailed)**
- Keys must be valid identifiers (regex `[A-Za-z_][A-Za-z0-9_]*`) and not reserved dict attributes.
- Attribute access maps to items; missing keys raise AttributeError listing available attributes.
- `GetItem` and `Get` support dotted key paths; no list indexing.
- `Set` creates nested maps along dotted path; raises if intermediate is not map/dict.
- `Flatten`, `FlattenItems`, `Pack`, `Transform`, `IsCompatible`, `Filter`, `FilterKeyVal`, `DebugString`, `VLog` mirror Lingvo-style NestedMap.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/src/py_utils.rs`.
- Rust public API surface: `NestedMap` with typed key validation and traversal helpers.

**Implementation Steps (Detailed)**
1. Implement NestedMap with key validation and attribute-like access (maybe via methods).
2. Provide flatten/pack utilities preserving order (sorted keys).
3. Implement filter logic with delete sentinel behavior.

**Tests (Detailed)**
- Python tests: indirect via core layers.
- Rust tests: unit tests for key validation, Get/Set, Flatten/Pack compatibility.

**Gaps / Notes**
- Python uses dynamic attributes; Rust should expose explicit methods.

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

### `monolith/core/testing_utils.py`
<a id="monolith-core-testing-utils-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 203
- Purpose/role: Test helper for Keras/BaseLayer compatibility; adapted from TF Keras testing_utils.
- Key symbols/classes/functions: `layer_test`.
- External dependencies: TF internal Keras/test utilities.
- Side effects: builds models, runs predict, validates shapes/dtypes.

**Required Behavior (Detailed)**
- `layer_test`:
  - Generates `input_data` if not provided; uses `input_shape` and `input_dtype` defaults.
  - Instantiates layer with `kwargs`; optionally calls `adapt`.
  - Tests `get_weights`/`set_weights` and re-instantiation with `weights` kwarg.
  - Builds functional model `y = layer(x)` and checks output dtype.
  - Checks expected output shape and computed output signature.
  - Runs `model.predict` and compares output to expected if provided.
  - Chooses assertion method based on dtype (string vs numeric).

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/tests/testing_utils.rs`.
- Rust public API surface: test helper for layer validation (if Rust layers exist).

**Implementation Steps (Detailed)**
1. Implement a minimal test harness for Rust layers with shape/dtype checks.
2. If Rust uses different layer API, map semantics accordingly.

**Tests (Detailed)**
- Python tests: used by layer tests in core.
- Rust tests: provide helper functions for layer unit tests.

**Gaps / Notes**
- TF internal APIs used here are not available in Rust; might require simplified harness.

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

### `monolith/core/tpu_variable.py`
<a id="monolith-core-tpu-variable-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 214
- Purpose/role: ReplicatedVariable wrapper for TPU context handling and resource variable ops.
- Key symbols/classes/functions: `ReplicatedVariable`, `_enclosing_tpu_context`, `_tensor_conversion`.
- External dependencies: TF internal ops.
- Side effects: registers tensor conversion function globally.

**Required Behavior (Detailed)**
- `_enclosing_tpu_context()` walks control flow contexts to find XLA TPU context.
- `ReplicatedVariable`:
  - Wraps list of per-replica variables; exposes `handle` that is replicated in TPU context.
  - Implements `assign`, `assign_add`, `assign_sub`, `read_value` using resource ops.
  - Provides `initializer`, `dtype`, `shape`, `get_shape`, `to_proto` by delegating to primary var.
  - `_should_act_as_resource_variable` is a no-op placeholder.
- Registers tensor conversion function and dense tensor like type (pre-TF2.3).

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-tf/src/tpu_variable.rs`.
- Rust public API surface: replicated variable wrapper if TF TPU is supported.

**Implementation Steps (Detailed)**
1. Provide wrapper around per-replica variables with TPU context awareness.
2. Implement assign/read operations using TF runtime or stub.
3. Ensure tensor conversion registration or equivalent behavior in Rust.

**Tests (Detailed)**
- Python tests: none.
- Rust tests: optional integration tests if TPU runtime is available.

**Gaps / Notes**
- TF TPU context is highly specific; Rust likely needs a bridge or stubs.

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

### `monolith/core/util.py`
<a id="monolith-core-util-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 269
- Purpose/role: GCS utilities, checkpoint reload helpers, and dataset date filtering.
- Key symbols/classes/functions: `download_gcs_file`, `update_params`, `range_dateset`.
- External dependencies: `google.cloud.storage`, `tensorflow.compat.v1`, `gsutil` via subprocess.
- Side effects: network calls to GCS, subprocess execution, logging.

**Required Behavior (Detailed)**
- `get_bucket_name_and_relavite_path(gs_file_path)` parses `gs://bucket/path` into bucket + relative path.
- `download_gcs_file` and `download_gcs_file_with_relative_path` fetch blobs to local filename.
- `list_gcs_files_with_prefix` returns bucket and blob iterator for prefix.
- `parse_example_number_meta_file` reads `file,count` lines (comma separated) and ensures file names are ascending.
- `calculate_shard_skip_file_number` computes per-shard skip counts based on completed steps and batch sizes.
- `get_checkpoint_completed_step_number` scans GCS checkpoint `.meta` files to find max step.
- `update_params`:
  - Computes batch sizes based on shard count (TPU workers) and validates consistency.
  - Uses checkpoint step to compute `shard_skip_file_number` based on meta file.
- `get_per_file_example_numbers_for_checkpoint_reload`:
  - Uses `gsutil ls` on dataset path, checks ordering, and matches against meta list.
- `range_dateset` filters dataset elements by date substring between start/end dates.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/src/util.rs`.
- Rust public API surface: GCS helpers and dataset date filtering.

**Implementation Steps (Detailed)**
1. Implement GCS helpers (use Google Cloud Storage SDK or `gsutil` fallback).
2. Port checkpoint reload logic and shard skip calculations.
3. Implement dataset date range filter (likely for TF dataset bridge).

**Tests (Detailed)**
- Python tests: `monolith/core/util_test.py`
- Rust tests: replicate range_dateset filtering behavior.

**Gaps / Notes**
- `gsutil` subprocess usage may need replacement or explicit dependency in Rust.

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
