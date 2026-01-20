<!--
Source: task/request.md
Lines: 13436-13664 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/layers/layer_ops.py`
<a id="monolith-native-training-layers-layer-ops-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 131
- Purpose/role: Python wrappers around custom TensorFlow ops (FFM, FeatureInsight, MonolithFidCounter) and their gradients.
- Key symbols/classes/functions: `ffm`, `feature_insight`, `fid_counter` and registered gradients `_ffm_grad`, `_feature_insight`, `_fid_counter_grad`.
- External dependencies: `gen_monolith_ops` custom op library, TensorFlow gradient registry.
- Side effects: Registers gradients for custom ops; uses TF summary in downstream layers.

**Required Behavior (Detailed)**
- `ffm(left, right, dim_size, int_type='multiply')`:
  - Calls `layer_ops_lib.FFM` with given attrs and returns the op output.
  - `int_type` determines multiply vs dot behavior.
- `_ffm_grad` (gradient for `FFM`):
  - Calls `layer_ops_lib.FFMGrad` with `grad`, `left`, `right`, `dim_size`, `int_type`.
  - Returns `(left_grad, right_grad)`.
- `feature_insight(input_embedding, weight, segment_sizes, aggregate=False)`:
  - Asserts `segment_sizes` provided and `input_embedding.shape[-1] == weight.shape[0]`.
  - Calls `FeatureInsight` custom op.
  - If `aggregate=True`:
    - Builds `segment_ids` of length `k * num_feature` where `k = weight.shape[-1]`.
    - Returns `transpose(segment_sum(transpose(out * out), segment_ids))`.
  - Else returns `out` directly.
- `_feature_insight` (gradient for `FeatureInsight`):
  - Calls `FeatureInsightGrad` with `grad`, `input`, `weight`, `segment_sizes`, `K`.
  - Returns gradients for input_embedding and weight.
- `fid_counter(counter, counter_threshold, step=1.0)`:
  - Calls `MonolithFidCounter` op with `counter`, `step`, `counter_threshold`.
  - Adds `step` to counter, then clamps at threshold.
  - Docstring notes counter slice should use `SgdOptimizer(1.0)` and suggests `Fp32Compressor`.
- `_fid_counter_grad`:
  - Gradient is `-step` (as a constant) until `counter >= counter_threshold`, then zero.

**Rust Mapping (Detailed)**
- Target crate/module: custom ops would live in `monolith-rs/crates/monolith-tf` (TF runtime) or in native Rust layers (for pure Rust path).
- Rust public API surface:
  - Provide equivalents for `ffm`, `feature_insight`, `fid_counter` if needed by higher-level layers.
- Data model mapping:
  - `int_type` → Rust enum for multiply/dot.
  - `segment_sizes` and `aggregate` → explicit API arguments.
- Feature gating: TF runtime only for custom ops unless reimplemented in Rust.
- Integration points: `feature_cross.GroupInt` uses `ffm`; other code may rely on `feature_insight` or `fid_counter`.

**Implementation Steps (Detailed)**
1. Decide whether to reimplement `FFM`, `FeatureInsight`, and `MonolithFidCounter` in Rust or provide TF-runtime wrappers.
2. If TF runtime is used, expose safe Rust bindings and gradient equivalents.
3. For pure Rust path, implement `ffm` and its backward, plus feature_insight/ fid_counter logic.
4. Add tests for forward and backward (if training supported).

**Tests (Detailed)**
- Python tests: None dedicated in this file (covered by layer tests).
- Rust tests: `monolith-rs/crates/monolith-layers/tests/layer_ops_test.rs` (new, if implemented).
- Cross-language parity test:
  - Compare FFM outputs/gradients and fid_counter behavior between Python and Rust.

**Gaps / Notes**
- `feature_insight` and `fid_counter` depend on custom TF ops; Rust currently lacks equivalents.

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

### `monolith/native_training/layers/layer_ops_test.py`
<a id="monolith-native-training-layers-layer-ops-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 232
- Purpose/role: Validates custom ops (FFM, FeatureInsight, MonolithFidCounter) across CPU/GPU and checks gradients.
- Key symbols/classes/functions: `LayerOpsTest` methods `test_ffm_mul`, `test_ffm_mul_grad`, `test_ffm_dot`, `test_ffm_dot_grad`, `test_feature_insight`, `test_feature_insight_grad`, `test_fid_counter_grad`.
- External dependencies: TensorFlow GPU test utilities, custom ops via `layer_ops`.
- Side effects: Forces GPU contexts when available; uses global `tf.random.set_seed(0)`.

**Required Behavior (Detailed)**
- `test_ffm_mul`:
  - Uses `ffm(left, right, dim_size=4)` with `left` shape `(8,40)` and `right` `(8,48)` (10*4 and 12*4).
  - Checks GPU device placement if GPU available; compares CPU and GPU outputs.
  - Expects output shape `(8, 480)` for multiply mode.
- `test_ffm_mul_grad`:
  - Computes gradients of sum of FFM output wrt left/right.
  - Expects left_grad shape `(8,40)` and right_grad `(8,48)`; CPU and GPU grads equal.
- `test_ffm_dot`:
  - Uses `int_type='dot'`.
  - Expects output shape `(8,120)`; CPU and GPU outputs equal.
- `test_ffm_dot_grad`:
  - Same gradient checks as multiply, output dims unchanged.
- `test_feature_insight`:
  - Builds expected result by splitting input/weights per `segment_sizes=[3,2,4]`, matmul per segment, then optional aggregate using segment_sum of squared outputs.
  - Calls `layer_ops.feature_insight(..., aggregate=True)` and asserts close to expected.
- `test_feature_insight_grad`:
  - Compares gradients of `feature_insight` output vs explicit matmul concatenation.
  - Asserts outputs and gradients match.
- `test_fid_counter_grad`:
  - Verifies fid_counter increments and gradient = `-step` until threshold, then 0 at threshold.
  - Checks counter values for step=1, step=0.01, and threshold case at 1000.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-layers/tests/layer_ops_test.rs` (if ops reimplemented) or `monolith-rs/crates/monolith-tf/tests` for TF-runtime bindings.
- Rust public API surface: FFM op (multiply/dot), FeatureInsight, fid_counter equivalents.
- Data model mapping:
  - Output shapes match Python expectations (multiply: `B * (L*R*D)` flattened, dot: `B * (L*R)`).
  - Gradients should match analytic gradients of FFM/FeatureInsight.
- Feature gating: GPU tests optional; must be skipped if GPU backend unavailable.
- Integration points: `feature_cross` uses FFM.

**Implementation Steps (Detailed)**
1. Implement Rust tests mirroring CPU/GPU parity (skip GPU when not supported).
2. Add gradient checks for FFM and FeatureInsight if backward is implemented.
3. Add fid_counter unit test that verifies saturation and gradient behavior.
4. Ensure deterministic seeding for random tensors.

**Tests (Detailed)**
- Python tests: `monolith/native_training/layers/layer_ops_test.py`.
- Rust tests: `monolith-rs/crates/monolith-layers/tests/layer_ops_test.rs` (new).
- Cross-language parity test:
  - Compare outputs and gradients for FFM multiply/dot and FeatureInsight aggregate mode.

**Gaps / Notes**
- Python tests rely on custom TF ops and GPU placement; Rust needs an equivalent implementation or explicit skip/feature gate.

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

### `monolith/native_training/layers/lhuc.py`
<a id="monolith-native-training-layers-lhuc-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 296
- Purpose/role: LHUCTower: augments a dense tower with LHUC gating MLPs per layer; supports shared or per-layer LHUC configs, optional batch normalization, and weight norm.
- Key symbols/classes/functions: `LHUCTower`, `lhuc_params`, `build`, `call`, `get_config`, `from_config`.
- External dependencies: TensorFlow/Keras (`Layer`, `BatchNormalization`, `Sequential`, regularizers), Monolith layers (`MLP`, `Dense`), `extend_as_list`, `advanced_activations`.
- Side effects: Creates nested Dense/MLP layers and BatchNorm; collects trainable/non-trainable weights; supports LHUC-specific overrides via `lhuc_*` kwargs.

**Required Behavior (Detailed)**
- Initialization:
  - Splits kwargs into `_lhuc_kwargs` with `lhuc_` prefix; remaining kwargs passed to base Layer.
  - `output_dims` defines dense tower layers; `n_layers = len(output_dims)`.
  - `activations`:
    - None → `[relu]*(n_layers-1) + [None]`.
    - List/tuple length must match `n_layers`; maps via `ad_acts.get`.
    - Single activation string/function → same for all but last layer (None).
  - `initializers` expanded to list length `n_layers` via `extend_as_list`.
  - LHUC output dims:
    - If `lhuc_output_dims` is list of lists: each last dim must equal corresponding `output_dims[i]`.
    - If list of ints: applied to every layer and auto-append `[dim]`.
    - Else default: `[[dim] for dim in output_dims]`.
  - `lhuc_activations`: for each LHUC MLP, uses `relu` for all but last, last is `sigmoid2`.
- `build`:
  - Optional input BatchNorm if `enable_batch_normalization`.
  - For each layer:
    - Create `Sequential` block with Dense (custom monolith Dense) + optional BatchNorm + activation.
    - Dense uses weight norm options and regularizers.
    - Build LHUC MLP (`MLP`) with per-layer `lhuc_output_dims` and overrides via `lhuc_params`.
  - Extends trainable/non-trainable weights from sublayers.
- `call`:
  - If inputs is tuple/list: `(dense_input, lhuc_input)` else both are inputs.
  - Apply `extra_layers` (input BatchNorm) to dense_input.
  - For each layer and corresponding lhuc MLP:
    - `output_t = layer(dense_input) * lhuc_layer(lhuc_input)`.
    - Feed output to next layer.
- `get_config`:
  - Serializes activations via `ad_acts.serialize`, initializers via `tf.initializers.serialize`.
  - Includes batch norm settings and regularizers.
  - Adds `_lhuc_kwargs` into config.
- `from_config`:
  - Creates params via `params().copy()`, fills known keys, deserializes initializers/activations and regularizers.
  - Pops used keys from config and returns `p.instantiate()`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-layers/src/lhuc.rs`.
- Rust public API surface: `LHUCTower`, `LHUCConfig`, `LHUCOverrides`, `LHUCOutputDims`.
- Data model mapping:
  - Python activations list → Rust `ActivationType` list; last layer forced to `None`.
  - `sigmoid2` in LHUC MLP last layer → Rust `ActivationType::Sigmoid2`.
  - `lhuc_*` kwargs → `LHUCOverrides`.
- Feature gating: None.
- Integration points: `Dense`, `MLP`, `BatchNorm`.

**Implementation Steps (Detailed)**
1. Ensure Rust LHUCTower uses same activation and initializer expansion rules.
2. Mirror LHUC output dims logic (shared list vs per-layer list).
3. Add input BatchNorm and per-layer BatchNorm gating with same defaults.
4. Implement LHUCOverrides mapping to override LHUC MLP settings.
5. Ensure config serialization/deserialization matches Python (including `lhuc_*` fields).

**Tests (Detailed)**
- Python tests: `monolith/native_training/layers/lhuc_test.py`.
- Rust tests: `monolith-rs/crates/monolith-layers/tests/lhuc_test.rs` (new).
- Cross-language parity test:
  - Fix weights and inputs; compare outputs for single-input and `(dense_input, lhuc_input)` modes.

**Gaps / Notes**
- Python `from_config` ignores `kernel_regularizer` and `bias_regularizer` assignments (calls deserialize but does not set); decide whether to mirror or fix in Rust.

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
