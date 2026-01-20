<!--
Source: task/request.md
Lines: 13665-13918 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/layers/lhuc_test.py`
<a id="monolith-native-training-layers-lhuc-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 73
- Purpose/role: Smoke tests for LHUCTower instantiation, config serialization, and forward call with separate dense/LHUC inputs.
- Key symbols/classes/functions: `LHUCTowerTest` methods `test_lhuc_instantiate`, `test_lhuc_serde`, `test_lhuc_call`.
- External dependencies: TensorFlow v1 session mode, NumPy.
- Side effects: Runs session after variable initialization.

**Required Behavior (Detailed)**
- `test_lhuc_instantiate`:
  - Params-based instantiate with `output_dims=[1,3,4,5]`, `activations=None`, `initializers=GlorotNormal`.
  - Direct constructor with same output dims and `initializers=HeUniform`.
- `test_lhuc_serde`:
  - Instantiates via params, `cfg = get_config()`, `LHUCTower.from_config(cfg)` should succeed.
- `test_lhuc_call`:
  - Builds LHUCTower with:
    - `output_dims=[50,20,1]`,
    - `lhuc_output_dims=[[50,50],[50,50,20],[100,1]]`,
    - `lhuc_use_bias=False`, `use_bias=True`, `activations=None`.
  - Inputs: `dense_data` `(100,100)` and `lhuc_data` `(100,50)`.
  - Calls layer with `[dense_data, lhuc_data]`, sums output, runs session.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-layers/tests/lhuc_test.rs`.
- Rust public API surface: `LHUCTower`, `LHUCConfig`, `LHUCOverrides`.
- Data model mapping:
  - `lhuc_*` kwargs ↔ `LHUCOverrides`.
  - Params-based instantiation ↔ Rust config builder.
- Feature gating: None.
- Integration points: `monolith_layers::lhuc`.

**Implementation Steps (Detailed)**
1. Add Rust tests for constructor/config round-trip.
2. Add forward test with separate dense/LHUC inputs and per-layer LHUC output dims.
3. Add assertions on output shape and deterministic sum.

**Tests (Detailed)**
- Python tests: `monolith/native_training/layers/lhuc_test.py`.
- Rust tests: `monolith-rs/crates/monolith-layers/tests/lhuc_test.rs` (new).
- Cross-language parity test:
  - Fix weights and inputs; compare output sums.

**Gaps / Notes**
- Python uses `lhuc_use_bias` override (via `lhuc_*` kwargs); ensure Rust override wiring matches.

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

### `monolith/native_training/layers/logit_correction.py`
<a id="monolith-native-training-layers-logit-correction-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 88
- Purpose/role: Logit correction layer to compensate for sampling bias during training or inference.
- Key symbols/classes/functions: `LogitCorrection`, `safe_log_sigmoid`, `get_sample_logits`.
- External dependencies: TensorFlow/Keras (`Layer`, `InputSpec`, activations), `with_params`.
- Side effects: None beyond computation.

**Required Behavior (Detailed)**
- Inputs: `(logits, sample_rate)` where both are max 2D tensors.
- `call`:
  - `corrected = get_sample_logits(logits, sample_rate, sample_bias)`.
  - If `activation` is set, apply it.
- `safe_log_sigmoid(logits)`:
  - Stable computation of `log(sigmoid(logits))` using `log1p(exp(neg_abs))` trick.
- `get_sample_logits`:
  - `sample_rate is None` and `sample_bias=True`: return `safe_log_sigmoid(logits)`.
  - `sample_rate not None` and `sample_bias=False`: return `logits - log(sample_rate)`.
  - `sample_rate not None` and `sample_bias=True`: return `safe_log_sigmoid(logits) - log(sample_rate)`.
  - Else: return `logits`.
- `compute_output_shape` returns a 1D shape `([None])` (questionable but part of API).
- `get_config` serializes `activation` and `sample_bias`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-layers/src/logit_correction.rs`.
- Rust public API surface: `LogitCorrection` with `forward_with_sample_rate`.
- Data model mapping:
  - Python activation → Rust `ActivationType`/`ActivationLayer`.
  - `sample_rate` optional tensor → `Option<&Tensor>`.
- Feature gating: None.
- Integration points: Used in training heads that correct logits for sampling.

**Implementation Steps (Detailed)**
1. Ensure `safe_log_sigmoid` matches TF numeric behavior.
2. Confirm `get_sample_logits` branch logic matches Python.
3. Add optional activation layer application.
4. Add config serialization to match `activation` and `sample_bias`.

**Tests (Detailed)**
- Python tests: `monolith/native_training/layers/logit_correction_test.py`.
- Rust tests: `monolith-rs/crates/monolith-layers/tests/logit_correction_test.rs` (new).
- Cross-language parity test:
  - Compare corrected logits for combinations of sample_rate present/absent and sample_bias true/false.

**Gaps / Notes**
- Python `compute_output_shape` always returns `[None]` regardless of input; Rust may not expose shape inference.

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

### `monolith/native_training/layers/logit_correction_test.py`
<a id="monolith-native-training-layers-logit-correction-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 65
- Purpose/role: Smoke tests for LogitCorrection instantiation, serialization, and forward call.
- Key symbols/classes/functions: `SailSpecialTest` methods `test_sr_instantiate`, `test_sr_serde`, `test_sr_call`.
- External dependencies: TensorFlow v1 session mode, NumPy.
- Side effects: Runs session after variable init.

**Required Behavior (Detailed)**
- `test_sr_instantiate`:
  - Params-based instantiation with `activation=relu`.
  - Direct constructor with `activation=relu`.
- `test_sr_serde`:
  - Instantiate with `activation=sigmoid`, then `get_config` and `from_config`.
- `test_sr_call`:
  - Instantiate via params with `activation=tanh`.
  - Inputs: logits `x` shape `(100,10)` and sample_rate `sr` shape `(100,1)` sampled in `(1e-10, 1)`.
  - Runs `layer((x, sr))` and sums outputs in session.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-layers/tests/logit_correction_test.rs`.
- Rust public API surface: `LogitCorrection` with optional activation.
- Data model mapping:
  - Params-based instantiation ↔ Rust config/builder.
  - `get_config`/`from_config` ↔ serde round-trip.
- Feature gating: None.
- Integration points: `monolith_layers::logit_correction`.

**Implementation Steps (Detailed)**
1. Add Rust tests for constructor/config round-trip.
2. Add forward test with logits + sample_rate; assert output shape and deterministic sum.

**Tests (Detailed)**
- Python tests: `monolith/native_training/layers/logit_correction_test.py`.
- Rust tests: `monolith-rs/crates/monolith-layers/tests/logit_correction_test.rs` (new).
- Cross-language parity test:
  - Fix logits/sample_rate and compare outputs for activation on/off.

**Gaps / Notes**
- Python test uses `activation=tanh`; ensure Rust supports this activation.

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

### `monolith/native_training/layers/mlp.py`
<a id="monolith-native-training-layers-mlp-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 211
- Purpose/role: Core MLP layer built from custom Dense layers with optional batch normalization, weight norm, and feature insight logging.
- Key symbols/classes/functions: `MLP`, `build`, `call`, `get_config`, `from_config`, `get_layer`.
- External dependencies: TensorFlow/Keras (`Layer`, `BatchNormalization`, regularizers), Monolith `Dense`, `extend_as_list`, `advanced_activations`, `feature_insight_data`.
- Side effects: Adds BN/Dense losses, uses `feature_insight_data` hook on first Dense when segment info provided.

**Required Behavior (Detailed)**
- Initialization:
  - `output_dims` defines layers; `use_weight_norm`, `use_learnable_weight_norm`, `use_bias`, regularizers and BN params stored.
  - `initializers` expanded to list of length `_n_layers` via `extend_as_list` and `tf.initializers.get`.
  - `activations`:
    - None → `[relu]*(n_layers-1) + [None]`.
    - List/tuple length must match `n_layers`; each mapped via `ad_acts.get`.
    - Single activation → applied to all but last (None).
- `build`:
  - If BN enabled, prepend input BN layer.
  - For each layer:
    - Create custom `Dense` with activation=None, bias/init/regularizer and kernel norm settings.
    - Optional BatchNorm between layers (not on final layer).
    - Append activation layer if not None.
  - Tracks trainable/non-trainable weights and adds sub-layer losses.
- `call`:
  - Sequentially applies `_stacked_layers`.
  - When a layer name ends with `dense_0` and kwargs provided, calls `feature_insight_data` with segment metadata and layer kernel.
- `get_config` serializes activations and initializers, regularizers, BN settings, weight norm flags.
- `from_config`:
  - Deserializes `initializers` and `activations`; others assigned directly.
  - Ignores leftover keys, then `instantiate()`.
- `get_layer(index)` returns `_stacked_layers[index]`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-layers/src/mlp.rs`.
- Rust public API surface: `MLP`, `MLPConfig`, `ActivationType`, `ActivationLayer`.
- Data model mapping:
  - Python activations list → Rust `ActivationType` list; last layer activation None.
  - `feature_insight_data` hook → optional instrumentation in Rust.
- Feature gating: None.
- Integration points: `Dense`, `BatchNorm`, activation registry.

**Implementation Steps (Detailed)**
1. Ensure MLPConfig defaults align with Python (weight norm on, BN off).
2. Add support for per-layer initializers and activations list expansion (1 or N).
3. Add optional input BatchNorm and per-layer BN (skip last).
4. Add optional feature insight hook (or document omission).
5. Add config serialization compatible with Python `get_config`.

**Tests (Detailed)**
- Python tests: `monolith/native_training/layers/mlp_test.py`.
- Rust tests: `monolith-rs/crates/monolith-layers/tests/mlp_test.rs` (new).
- Cross-language parity test:
  - Fix weights and inputs; compare output sums with/without BN and weight norm.

**Gaps / Notes**
- Python `feature_insight_data` side effect not yet mirrored in Rust.

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
