<!--
Source: task/request.md
Lines: 12500-12648 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/layers/advanced_activations.py`
<a id="monolith-native-training-layers-advanced-activations-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 217
- Purpose/role: Defines activation layer wrappers, exports advanced activation classes, and provides `get/serialize/deserialize` helpers for activation identifiers.
- Key symbols/classes/functions:
  - Classes: `ReLU`, `LeakyReLU`, `ELU`, `Softmax`, `ThresholdedReLU`, `PReLU` (from TF), plus custom Layer wrappers `Tanh`, `Sigmoid`, `Sigmoid2`, `Linear`, `Gelu`, `Selu`, `Softsign`, `Softplus`, `Exponential`, `HardSigmoid`, `Swish`.
  - `get(identifier)`, `serialize(activation)`, `deserialize(identifier)`.
  - `__all__`, `__all_activations`, `ALL_ACTIVATION_NAMES`.
- External dependencies: TensorFlow Keras activations/layers, `types.MethodType`, Monolith `_params`, `monolith_export`.
- Side effects: Monkey-patches `params` method onto activation layer classes.

**Required Behavior (Detailed)**
- Class setup:
  - Defines lightweight `Layer` subclasses via `type(...)` for Tanh/Sigmoid/etc; each implements `call` with the corresponding TF activation function.
  - Adds `.params = MethodType(_params, cls)` for all activation classes listed (including TF-provided advanced activations).
- Export lists:
  - `__all__` includes names of all activation layers and wrappers.
  - `__all_activations` maps lowercase names (and synonyms like `hard_sigmoid`/`hardsigmoid`) to classes.
  - `ALL_ACTIVATION_NAMES = set(__all_activations.keys())`.
- `get(identifier)`:
  - `None` → `None`.
  - `str`:
    - If `identifier.lower()` in `__all_activations`: return a **new instance** of that class.
    - Else `eval(identifier)`; if dict, call `deserialize`; otherwise raise `TypeError`.
  - `dict`: call `deserialize`.
  - `callable`:
    - If has `params`, try `issubclass(identifier, Layer)`: if true, return new instance; else return identifier.
    - If `identifier` is a `Layer` instance: create new instance based on its class name.
    - Else try `identifier.__name__` and map to `__all_activations`; if not found return identifier.
  - Else: raise `TypeError('Could not interpret activation function identifier: ...')`.
- `serialize(activation)`:
  - Returns `repr(dict)` strings for known activation types; `None` otherwise.
  - Uses class-specific fields:
    - `LeakyReLU`/`ELU`: `alpha`.
    - `ReLU`: `max_value`, `negative_slope`, `threshold`.
    - `PReLU`: `alpha_initializer`, `alpha_regularizer`, `alpha_constraint`, `shared_axes` (uses `initializers.serialize` for all three in Python).
    - `Softmax`: `axis`.
    - `ThresholdedReLU`: `theta`.
- `deserialize(identifier)`:
  - Accepts dict or string repr of dict (via `eval`).
  - Requires `name` key; lowercases and looks up in `__all_activations`, pops `name`, and instantiates class with remaining kwargs.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-layers/src/activation.rs` and `activation_layer.rs`.
- Rust public API surface:
  - Activation structs (ReLU, LeakyReLU, ELU, Softmax, ThresholdedReLU, PReLU, Tanh, Sigmoid, Sigmoid2, Linear, GELU, SELU, Softplus, Softsign, Swish, Exponential, HardSigmoid).
  - `ActivationType` enum (in `mlp.rs`) and `ActivationLayer` wrapper for dynamic dispatch.
  - Add `activation::get`, `activation::serialize`, `activation::deserialize` or a dedicated registry module to mirror Python behavior.
- Data model mapping:
  - Python string identifiers ↔ Rust `ActivationType` (case-insensitive, include synonyms).
  - Python repr(dict) ↔ Rust serde JSON/YAML or explicit struct for config; if parity requires, accept Python-style dict strings.
- Feature gating: None.
- Integration points: `MLP` and any layer configs that accept activations.

**Implementation Steps (Detailed)**
1. Add a Rust activation registry mapping lowercased names and synonyms to constructors.
2. Implement `get(identifier)` variants:
   - Accept `&str`, `ActivationType`, or config struct; return `ActivationLayer`.
3. Implement `serialize` to return a Python-compatible dict representation (or document accepted Rust-native format and add translation).
4. Implement `deserialize` that can accept Python-style `repr(dict)` if needed for cross-language parity.
5. Ensure `params`-like metadata exists for activation classes (if using config builders).

**Tests (Detailed)**
- Python tests: `monolith/native_training/layers/advanced_activations_test.py`.
- Rust tests: `monolith-rs/crates/monolith-layers/tests/advanced_activations_test.rs` (new).
- Cross-language parity test:
  - For each name in `ALL_ACTIVATION_NAMES`, call `get` and compare forward outputs on fixed input.

**Gaps / Notes**
- Python uses `eval` on identifier strings for deserialize; Rust should avoid eval and instead accept a safe format, but parity requires handling Python-style repr strings.
- Python uses `initializers.serialize` for `alpha_regularizer`/`alpha_constraint` in `PReLU` serialization; verify whether this is a bug and whether Rust should mirror it.

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

### `monolith/native_training/layers/advanced_activations_test.py`
<a id="monolith-native-training-layers-advanced-activations-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 84
- Purpose/role: Exercises `advanced_activations.get`/`serialize` with identifiers and ensures activation layers run in a TF session.
- Key symbols/classes/functions: `serde`, `all_acts`, `raw_acts`, `lay_acts`, `ActivationsTest`.
- External dependencies: TensorFlow Keras activations/layers, TF v1 session mode.
- Side effects: Disables eager execution in main guard; runs session with variable initialization.

**Required Behavior (Detailed)**
- `serde(act)`:
  - `_act = get(act)`, `sered_act = serialize(_act)`, then `get(sered_act)` must succeed.
- `all_acts` list defines string identifiers for names to test.
- `raw_acts` list of Keras activation functions exists but is unused in tests.
- `lay_acts` list includes Keras layer instances (`ReLU`, `PReLU`, `ThresholdedReLU`, `ELU`, `Softmax`, `LeakyReLU`).
- Tests:
  - `test_get_from_str`: calls `serde` for each name in `all_acts`.
  - `test_get_from_layers`: calls `serde` for each layer instance in `lay_acts`.
  - `test_get_from_func`: loops `lay_acts` again (likely intended `raw_acts` but uses layers).
  - `test_params`: for each name, calls `cls = get(act).__class__` then `cls.params()`.
  - `test_call`: creates input `(100, 200)`, applies `get(act)` to each name, sums and evaluates in a session.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-layers/tests/advanced_activations_test.rs`.
- Rust public API surface: activation registry `get`, `serialize`, config params metadata.
- Data model mapping:
  - `all_acts` strings → Rust name lookup (`ActivationType` or registry).
  - `serialize` output → Rust serialization format (must accept Python repr strings if parity required).
- Feature gating: None.
- Integration points: `monolith_layers::activation` and `ActivationLayer`.

**Implementation Steps (Detailed)**
1. Add Rust tests to cover name lookup for all `all_acts` names.
2. Add serde round-trip test for each activation type.
3. Add test to ensure `params`/config metadata exists for each activation class.
4. Add forward test applying all activations to a fixed tensor and summing outputs.

**Tests (Detailed)**
- Python tests: `monolith/native_training/layers/advanced_activations_test.py`.
- Rust tests: `monolith-rs/crates/monolith-layers/tests/advanced_activations_test.rs` (new).
- Cross-language parity test:
  - Use identical inputs and compare output sums per activation name.

**Gaps / Notes**
- `test_get_from_func` likely intended to use `raw_acts` but currently uses `lay_acts`; Rust tests should mirror the current behavior, not the intent.

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
