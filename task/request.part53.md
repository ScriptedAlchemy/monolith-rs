<!--
Source: task/request.md
Lines: 12265-12499 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/layers/__init__.py`
<a id="monolith-native-training-layers-init-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 46
- Purpose/role: Aggregates and re-exports Keras layers plus Monolith custom layers, then patches Keras layer classes with a `params()` helper for InstantiableParams construction.
- Key symbols/classes/functions:
  - Re-exported custom layers: `MLP`, `Dense` (custom override), `AddBias`, `LHUCTower`, `LogitCorrection`, `LayerNorm`, `GradNorm`, `SumPooling`, `AvgPooling`, `MaxPooling`, `MergeType`, `DCNType`, `MMoE`, `SNR`, plus everything from `feature_cross`, `feature_trans`, `feature_seq`, `advanced_activations`.
  - `keras_layers` dict: name → Keras layer class with `params` monkey-patched.
- External dependencies: `tensorflow` (`tf.keras.layers`, `Layer`), `types.MethodType`, `monolith.native_training.utils.params` (`params` helper).
- Side effects: Module import-time monkey-patching of Keras layer classes to inject `params()`; removal of wildcard-imported `Dense` symbol to replace with custom Dense.

**Required Behavior (Detailed)**
- Import order and namespace behavior:
  - `from tensorflow.keras.layers import *` makes all Keras layer classes available in module namespace.
  - `del globals()['Dense']` removes the Keras `Dense` symbol created by the wildcard import.
  - `from monolith.native_training.layers.dense import Dense` inserts the custom Dense in its place.
- Custom layer re-exports:
  - Re-exports layer modules so downstream Python code can do `from monolith.native_training.layers import X` for both Keras layers and custom layers.
- Keras layer patching:
  - Creates `keras_layers = {}`.
  - Iterates `dir(tf.keras.layers)`, skipping names that start with `_` or are exactly `"Layer"`.
  - For each candidate:
    - Retrieves `cls = getattr(tf.keras.layers, name)`.
    - If `issubclass(cls, Layer)` and `cls` does **not** already have `params`, then attaches `cls.params = MethodType(_params, cls)` and inserts `keras_layers[name] = cls`.
    - All errors in this process are swallowed (`except: pass`) to avoid import-time failures for non-class attributes or invalid `issubclass` checks.
  - Result: `keras_layers` includes only Keras layer classes that were patched.
- Error handling and determinism:
  - No explicit errors thrown; all reflection errors are suppressed.
  - Determinism is not relevant; import-time logic is pure reflection/patching.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-layers/src/lib.rs` + `monolith-rs/crates/monolith-layers/src/prelude` (existing `prelude` module).
- Rust public API surface:
  - Provide a `prelude` module that re-exports Monolith layers analogous to Python's `__init__` aggregator.
  - Ensure `Dense` refers to the Monolith implementation (`monolith_layers::Dense`).
  - Expose `MergeType` and `DCNType` equivalents (`MergeType` already exists; map Python `DCNType` to Rust `DCNMode` or add an alias if necessary).
- Data model mapping:
  - Python's `params()` returns `InstantiableParams`; Rust should expose a `LayerParams`/`BuildConfig` trait that returns serializable config metadata per layer, or central registry entries.
- Feature gating:
  - If TF runtime backend is enabled (`cfg(feature = "tf-runtime")`), consider an optional registry for TensorFlow-native layers.
  - Otherwise, provide Monolith-only registry/prelude without TF/Keras dependencies.
- Integration points:
  - Downstream uses `monolith.native_training.layers` as a convenience import; in Rust this maps to `monolith_layers::prelude::*` or a `layers` module in the top-level crate.

**Implementation Steps (Detailed)**
1. Confirm `monolith_layers::prelude` exports all custom layers used in Python (`AddBias`, `MLP`, `LHUCTower`, `LogitCorrection`, `LayerNorm`, `GradNorm`, pooling layers, `MMoE`, `SNR`, feature cross/trans/seq layers).
2. Add missing exports or alias types in `monolith-rs/crates/monolith-layers/src/lib.rs` to mirror Python names (`DCNType` alias if needed).
3. Create an optional `LayerRegistry` (e.g., `HashMap<&'static str, LayerFactory>`) to mimic `keras_layers` if dynamic layer lookup is required; document that Python only registers patched Keras layers.
4. Implement a `LayerParams` trait or per-layer config builder to mirror Python `params()` (if used by config/codegen tooling).
5. Add a top-level `monolith-rs/crates/monolith/src/layers.rs` (or re-export from `monolith_layers`) to provide a single import path similar to Python.
6. Document that there is no exact Keras wildcard import equivalent; in Rust, this is replaced by explicit prelude exports and optional registry for dynamic construction.

**Tests (Detailed)**
- Python tests: None directly for `layers/__init__.py`.
- Rust tests:
  - Add a compile-time test that `monolith_layers::prelude::*` includes expected names (e.g., `Dense`, `MLP`, `AddBias`, pooling layers).
  - If a registry is added, include a test that registry contains expected layer names and no duplicates.
- Cross-language parity test:
  - (Optional) capture list of exported layer names in Python and compare to Rust prelude/registry list for overlap.

**Gaps / Notes**
- `keras_layers` is only populated for Keras classes missing `params`; if Rust introduces a registry, decide whether it should include only "patched" entries or the full set of Rust layers.
- Python explicitly suppresses all errors during reflection; Rust should avoid panicking on optional registry population.

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

### `monolith/native_training/layers/add_bias.py`
<a id="monolith-native-training-layers-add-bias-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 110
- Purpose/role: Keras layer that adds a learnable bias to inputs while handling `channels_first`/`channels_last` formats for 3D/4D/5D tensors.
- Key symbols/classes/functions: `AddBias(Layer)` with `build`, `call`, `get_config`.
- External dependencies: TensorFlow (`Layer`, `InputSpec`, `initializers`, `regularizers`, `tf.nn.bias_add`), Monolith utils (`get_ndim`, `int_shape`, `with_params`), layer utils (`check_dim`, `dim_size`), `monolith_export`.
- Side effects: Adds a trainable weight named `"bias"` during `build`; decorated with `@with_params` and `@monolith_export`.

**Required Behavior (Detailed)**
- Initialization:
  - `initializer = initializers.get(initializer) or tf.initializers.Zeros()`.
  - `regularizer = regularizers.get(regularizer)`.
  - `input_spec = InputSpec(min_ndim=2)`; `bias=None`.
- Build:
  - `shape = list(map(check_dim, input_shape[1:]))` (batch dim removed).
  - `check_dim(None) -> -1`, so unknown dims propagate to bias shape.
  - `self.add_weight(name='bias', shape=shape, dtype=tf.float32, initializer=initializer, regularizer=regularizer)`.
- Call:
  - `data_format = kwargs.get('data_format', 'channels_last')`.
  - Validate `data_format` is `"channels_first"` or `"channels_last"`; otherwise raise `ValueError('Unknown data_format: ' + str(data_format))`.
  - `bias_shape = int_shape(self.bias)` (tuple, `-1` for unknown dims).
  - If `len(bias_shape)` is not `1` and not `get_ndim(inputs) - 1`, raise:
    - `ValueError('Unexpected bias dimensions %d, expect to be 1 or %d dimensions' % (len(bias_shape), get_ndim(inputs)))`.
  - For `get_ndim(inputs) == 5`:
    - `channels_first`:
      - `len(bias_shape)==1`: reshape to `(1, C, 1, 1, 1)`.
      - `len(bias_shape)>1`: reshape to `(1, bias_shape[3]) + bias_shape[:3]`.
    - `channels_last`:
      - `len(bias_shape)==1`: reshape to `(1, 1, 1, C)`.
      - `len(bias_shape)>1`: reshape to `(1,) + bias_shape`.
  - For `get_ndim(inputs) == 4`:
    - `channels_first`:
      - `len(bias_shape)==1`: reshape to `(1, C, 1, 1)`.
      - `len(bias_shape)>1`: reshape to `(1, bias_shape[2]) + bias_shape[:2]`.
    - `channels_last`:
      - `len(bias_shape)==1`: `tf.nn.bias_add(inputs, bias, data_format='NHWC')`.
      - `len(bias_shape)>1`: reshape to `(1,) + bias_shape`.
  - For `get_ndim(inputs) == 3`:
    - `channels_first`:
      - `len(bias_shape)==1`: reshape to `(1, C, 1)`.
      - `len(bias_shape)>1`: reshape to `(1, bias_shape[1], bias_shape[0])`.
    - `channels_last`:
      - `len(bias_shape)==1`: reshape to `(1, 1, C)`.
      - `len(bias_shape)>1`: reshape to `(1,) + bias_shape`.
  - Else (2D or other): `tf.nn.bias_add(inputs, bias)`.
- Serialization:
  - `get_config()` serializes initializer/regularizer and merges with base config.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-layers/src/add_bias.rs`.
- Rust public API surface: `AddBias`, `DataFormat`, `forward_with_format`, builder-style setters.
- Data model mapping:
  - Python `initializer` → Rust `Initializer`.
  - Python `regularizer` → Rust `Regularizer`.
  - Python `data_format` string → Rust `DataFormat`.
- Feature gating: None; pure Rust.
- Integration points: Re-export `AddBias` from `monolith_layers::prelude` for parity with Python `layers` aggregator.

**Implementation Steps (Detailed)**
1. Enforce Python error cases in Rust:
   - Reject bias shapes not equal to `1` or `ndim-1` with the same message text.
   - Reject invalid `data_format` string on parse.
2. Verify reshape permutations match Python for 3D/4D/5D (channels-first permutations).
3. Match `tf.nn.bias_add` semantics for 4D channels-last and 2D inputs (broadcast + dtype behavior).
4. Decide how to handle unknown dimensions (`-1` in Python) in Rust, document and test.
5. Add `LayerParams` metadata or config serialization to mirror `with_params` and `get_config`.

**Tests (Detailed)**
- Python tests: `monolith/native_training/layers/add_bias_test.py`.
- Rust tests: `monolith-rs/crates/monolith-layers/tests/add_bias_test.rs` (new).
- Cross-language parity test:
  - Generate fixed bias + random tensors (3D/4D/5D) and compare outputs between Python and Rust.

**Gaps / Notes**
- Python allows `-1` dims in bias shape; Rust currently assumes known sizes.
- Python uses `tf.nn.bias_add` for 2D and 4D `channels_last`; Rust currently uses reshape + add.

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

### `monolith/native_training/layers/add_bias_test.py`
<a id="monolith-native-training-layers-add-bias-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 65
- Purpose/role: Unit tests for `AddBias` instantiation, serialization, and forward call in TF v1 session mode.
- Key symbols/classes/functions: `AddBiasTest.test_ab_instantiate`, `test_ab_serde`, `test_ab_call`.
- External dependencies: TensorFlow v1 session runtime, NumPy.
- Side effects: Uses `tf.compat.v1.disable_eager_execution()` in main guard; runs session initializers.

**Required Behavior (Detailed)**
- `test_ab_instantiate`:
  - Builds `layer_template = AddBias.params()`.
  - Copies params, sets `initializer = tf.initializers.Zeros()`, calls `instantiate()`.
  - Also instantiates with `AddBias(initializer=tf.initializers.Zeros())`.
  - Both constructions must succeed.
- `test_ab_serde`:
  - Instantiates via params.
  - `cfg = ins1.get_config()`, then `AddBias.from_config(cfg)`.
  - No assertion; should complete without error.
- `test_ab_call`:
  - Creates layer via params, sets `name` and `initializer`.
  - Creates variable input shape `(100, 10)` and computes `tf.reduce_sum(layer(data))`.
  - Runs in a session after `global_variables_initializer()`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-layers/tests/add_bias_test.rs`.
- Rust public API surface: `AddBias` constructor/builder, serialization, forward call.
- Data model mapping:
  - `AddBias.params()` ↔ Rust config builder or `LayerParams` metadata.
  - `get_config`/`from_config` ↔ serde round-trip for `AddBias`.
- Feature gating: None.
- Integration points: `monolith_layers::AddBias`.

**Implementation Steps (Detailed)**
1. Add a constructor test for `AddBias::new()` and builder methods.
2. Add serde round-trip test for `AddBias` config.
3. Add forward pass test with deterministic input and bias; assert output sum.
4. Keep tests CPU-only; no TF runtime required.

**Tests (Detailed)**
- Python tests: `monolith/native_training/layers/add_bias_test.py`.
- Rust tests: `monolith-rs/crates/monolith-layers/tests/add_bias_test.rs` (new).
- Cross-language parity test:
  - Compare output sum for fixed input/bias between Python and Rust.

**Gaps / Notes**
- Python tests are smoke tests without explicit assertions; Rust should add asserts for deterministic behavior.

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
