<!--
Source: task/request.md
Lines: 14189-14519 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/layers/norms_test.py`
<a id="monolith-native-training-layers-norms-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 84
- Purpose/role: Smoke tests for LayerNorm and GradNorm instantiation and serialization; simple forward for LayerNorm.
- Key symbols/classes/functions: `NormTest` methods `test_ln_instantiate`, `test_ln_serde`, `test_ln_call`, `test_gn_instantiate`, `test_gn_serde`.
- External dependencies: TensorFlow v1 session mode, NumPy.
- Side effects: Runs session for LayerNorm forward.

**Required Behavior (Detailed)**
- LayerNorm:
  - Instantiate via params with `initializer=GlorotNormal`.
  - Direct constructor with `initializer=HeUniform`.
  - `test_ln_call`: input `(100,100)`, sum outputs, run session.
- GradNorm:
  - Instantiate via params with `loss_names=["abc","defg"]`.
  - Direct constructor with `relative_diff=True`.
  - `get_config`/`from_config` round-trip.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-layers/tests/norms_test.rs`.
- Rust public API surface: `LayerNorm`, `GradNorm`.
- Data model mapping:
  - Params-based instantiation ↔ Rust config/builder.
  - `get_config`/`from_config` ↔ serde round-trip.
- Feature gating: None.
- Integration points: `monolith_layers::normalization`.

**Implementation Steps (Detailed)**
1. Add Rust tests for LayerNorm config round-trip and forward output shape.
2. Add Rust tests for GradNorm config round-trip.
3. Add deterministic assertions where possible.

**Tests (Detailed)**
- Python tests: `monolith/native_training/layers/norms_test.py`.
- Rust tests: `monolith-rs/crates/monolith-layers/tests/norms_test.rs` (new).
- Cross-language parity test:
  - Fix inputs and compare LayerNorm output sums.

**Gaps / Notes**
- Python tests do not cover BatchNorm; ensure BatchNorm tests are added elsewhere in Rust.

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

### `monolith/native_training/layers/pooling.py`
<a id="monolith-native-training-layers-pooling-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 101
- Purpose/role: Defines list-based pooling layers: base `Pooling`, `SumPooling`, `AvgPooling`, `MaxPooling`.
- Key symbols/classes/functions: `Pooling.call`, `SumPooling.pool`, `AvgPooling.pool`, `MaxPooling.pool`.
- External dependencies: TensorFlow ops (`math_ops`, `array_ops`), Monolith `check_list`.
- Side effects: None.

**Required Behavior (Detailed)**
- `Pooling.call(vec_list)`:
  - Validates list with `check_list(vec_list, lambda x: x > 0)` (ensures non-empty).
  - If list length is 1, returns first tensor.
  - Otherwise calls `self.pool`.
- `SumPooling.pool`: `math_ops.add_n(vec_list)`.
- `AvgPooling.pool`: `math_ops.add_n(vec_list) / len(vec_list)`.
- `MaxPooling.pool`: `reduce_max(stack(vec_list), axis=0)`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-layers/src/pooling.rs`.
- Rust public API surface: `SumPooling`, `AvgPooling`, `MaxPooling` implementing `Pooling` trait.
- Data model mapping: list of tensors → slice `&[Tensor]`.
- Feature gating: None.
- Integration points: `monolith_layers::pooling`.

**Implementation Steps (Detailed)**
1. Ensure Rust pooling checks non-empty list and returns first tensor when length is 1 (Python behavior).
2. Confirm max pooling uses elementwise max (not stacking + reduce if already implemented).
3. Add config/params metadata if needed for with_params parity.

**Tests (Detailed)**
- Python tests: none specific.
- Rust tests: `monolith-rs/crates/monolith-layers/tests/pooling_test.rs` (new) or unit tests in module.
- Cross-language parity test:
  - Compare sums/means/maxes on fixed tensors with list length 1 and >1.

**Gaps / Notes**
- Python `check_list` enforces non-empty; Rust returns error on empty list, but must also return input directly when len=1.

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

### `monolith/native_training/layers/pooling_test.py`
<a id="monolith-native-training-layers-pooling-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 141
- Purpose/role: Smoke tests for Sum/Max/Avg pooling instantiation, serialization, and forward call.
- Key symbols/classes/functions: `PoolingTest` methods `test_sp_*`, `test_mp_*`, `test_ap_*`.
- External dependencies: TensorFlow v1 session mode, NumPy.
- Side effects: Runs session after variable init.

**Required Behavior (Detailed)**
- SumPooling:
  - Params-based instantiate and direct constructor.
  - `test_sp_call`: list of 5 tensors `(100,10)`, sum output.
- MaxPooling:
  - Params-based instantiate and direct constructor.
  - `test_mp_call`: list of 5 tensors `(100,10)`, sum output.
- AvgPooling:
  - Params-based instantiate and direct constructor.
  - `test_ap_call`: list of 5 tensors `(100,10)`, sum output.
- Serialization:
  - `get_config` and `from_config` for each pooling type.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-layers/tests/pooling_test.rs`.
- Rust public API surface: `SumPooling`, `MaxPooling`, `AvgPooling`.
- Data model mapping:
  - Params-based instantiation ↔ Rust config/builder (if needed).
  - No parameters beyond defaults.
- Feature gating: None.
- Integration points: `monolith_layers::pooling`.

**Implementation Steps (Detailed)**
1. Add Rust tests for pooling ops on list of tensors (len=5).
2. Add tests for len=1 list to match Python `Pooling.call`.
3. Add serialization tests if config metadata is implemented.

**Tests (Detailed)**
- Python tests: `monolith/native_training/layers/pooling_test.py`.
- Rust tests: `monolith-rs/crates/monolith-layers/tests/pooling_test.rs` (new).
- Cross-language parity test:
  - Fix input tensors; compare sums for each pooling type.

**Gaps / Notes**
- Python pooling layers rely on Keras `Layer` base; Rust pooling is a trait. Map serialization accordingly.

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

### `monolith/native_training/layers/sparse_nas.py`
<a id="monolith-native-training-layers-sparse-nas-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 31
- Purpose/role: Placeholder module containing only imports; no classes or functions defined.
- Key symbols/classes/functions: None.
- External dependencies: TF/Keras, FeatureList, SummaryType, logging/flags; all imported but unused.
- Side effects: None.

**Required Behavior (Detailed)**
- No runtime behavior; module only defines imports.
- If later extended, define behavior for sparse NAS utilities.

**Rust Mapping (Detailed)**
- Target crate/module: N/A (no behavior to port).
- Rust public API surface: None.
- Data model mapping: None.
- Feature gating: None.
- Integration points: None.

**Implementation Steps (Detailed)**
1. Confirm this file is a stub; if not, locate missing code or history.
2. If future Python changes add functionality, update parity mapping accordingly.

**Tests (Detailed)**
- Python tests: `monolith/native_training/layers/sparse_nas_test.py`.
- Rust tests: N/A unless functionality is added.
- Cross-language parity test: N/A until implemented.

**Gaps / Notes**
- This file appears to be an empty scaffold; confirm if code was removed or moved.

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

### `monolith/native_training/layers/sparse_nas_test.py`
<a id="monolith-native-training-layers-sparse-nas-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 23
- Purpose/role: Empty test scaffold; no test cases defined.
- Key symbols/classes/functions: None.
- External dependencies: TensorFlow (imported), NumPy (imported), unused.
- Side effects: Runs `tf.test.main()` when executed directly.

**Required Behavior (Detailed)**
- No assertions or tests; file does not exercise any functionality.

**Rust Mapping (Detailed)**
- Target crate/module: N/A.
- Rust public API surface: None.
- Data model mapping: None.
- Feature gating: None.
- Integration points: None.

**Implementation Steps (Detailed)**
1. Confirm no test coverage required unless sparse_nas gains functionality.

**Tests (Detailed)**
- Python tests: `monolith/native_training/layers/sparse_nas_test.py` (empty).
- Rust tests: N/A.
- Cross-language parity test: N/A.

**Gaps / Notes**
- If sparse_nas gains code, add corresponding tests and parity section updates.

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

### `monolith/native_training/layers/utils.py`
<a id="monolith-native-training-layers-utils-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 159
- Purpose/role: Shared utilities for layer code: merge semantics, shape helpers, and Gumbel-based subset sampling.
- Key symbols/classes/functions: `MergeType`, `DCNType`, `check_dim`, `dim_size`, `merge_tensor_list`, `gumbel_keys`, `continuous_topk`, `sample_subset`.
- External dependencies: TensorFlow, NumPy.
- Side effects: None.

**Required Behavior (Detailed)**
- `MergeType`: string constants `concat`, `stack`, `None`.
- `DCNType`: string constants `vector`, `matrix`, `mixed`.
- `check_dim(dim)`:
  - `None` → `-1`, `int` → itself, `tf.compat.v1.Dimension` → `.value`, else raise.
- `dim_size(inputs, axis)`:
  - Uses static shape; if unknown (`-1`), returns dynamic `array_ops.shape(inputs)[axis]`.
- `merge_tensor_list(tensor_list, merge_type='concat', num_feature=None, axis=1, keep_list=False)`:
  - Accepts tensor or list; if single tensor, uses shape to decide:
    - 3D: `stack` returns `[tensor]` or tensor; `concat` reshapes to `[B, num_feat*emb]`; `None` unstack on axis.
    - 2D with `num_feature>1`: `stack` reshapes to `[B, num_feature, emb]`; `concat` returns as-is; `None` unstack.
    - 2D without `num_feature`: returns as-is.
    - Else: raise shape error.
  - For list length >1: `stack`, `concat`, or return list.
- `gumbel_keys(w)`: samples Gumbel noise and adds to `w`.
- `continuous_topk(w, k, t, separate=False)`:
  - Iteratively computes soft top-k masks; returns sum or list.
- `sample_subset(w, k, t=0.1)`:
  - `w = gumbel_keys(w)` then `continuous_topk`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-layers/src/merge.rs` for merge utilities; DCNType maps to `monolith-layers/src/dcn.rs` (`DCNMode`).
- Rust public API surface: `MergeType`, `merge_tensor_list`, `merge_tensor_list_tensor`.
- Data model mapping:
  - `MergeType::None` corresponds to `MergeOutput::List`.
  - `check_dim`/`dim_size` are implicit in Rust shape handling; consider helper utilities.
  - Gumbel subset sampling functions not currently present in Rust.
- Feature gating: None.
- Integration points: feature_cross, feature_trans, senet, etc.

**Implementation Steps (Detailed)**
1. Verify `merge_tensor_list` semantics in Rust match Python (including single-tensor reshape/unstack cases).
2. Add Rust equivalents for `check_dim`/`dim_size` if needed for dynamic shapes.
3. Implement Gumbel subset sampling helpers if required by future layers.

**Tests (Detailed)**
- Python tests: none specific.
- Rust tests: add unit tests in `monolith-rs/crates/monolith-layers/tests/merge_test.rs` if not present.
- Cross-language parity test:
  - Compare merge outputs for 2D/3D inputs with `num_feature` and `keep_list` settings.

**Gaps / Notes**
- Gumbel subset sampling utilities are missing in Rust; add if used elsewhere.

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
