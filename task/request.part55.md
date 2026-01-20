<!--
Source: task/request.md
Lines: 12649-12818 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/layers/agru.py`
<a id="monolith-native-training-layers-agru-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 295
- Purpose/role: Implements an Attention GRU (AGRU/AUGRU) cell and helper functions for static/dynamic RNN with attention scores.
- Key symbols/classes/functions: `AGRUCell`, `create_ta`, `static_rnn_with_attention`, `dynamic_rnn_with_attention`.
- External dependencies: TensorFlow internals (`rnn_cell_impl`, `tensor_array_ops`, `control_flow_ops`, `array_ops`, `math_ops`, `nn_ops`), Keras `Layer`, `InputSpec`, activations/initializers/regularizers, Monolith utils (`with_params`, `check_dim`, `dim_size`).
- Side effects: Creates trainable weights (`gates/*`, `candidate/*`) with initializer/regularizer; uses TensorArray and TF while_loop for dynamic RNN.

**Required Behavior (Detailed)**
- `AGRUCell` initialization:
  - `units` required; `att_type` must be `"AGRU"` or `"AUGRU"` (case-insensitive).
  - `activation = activations.get(activation or math_ops.tanh)`.
  - `initializer = tf.initializers.get(initializer) or tf.initializers.HeNormal()`.
  - `regularizer = regularizers.get(regularizer)`.
  - `input_spec` requires 3 inputs: `(x, state, att_score)`; `x` and `state` are 2D, `att_score` max_ndim=2.
- `build(inputs_shape)`:
  - `input_shape, state_shape, att_shape = inputs_shape`.
  - Assert `state_shape[-1] == units`.
  - `input_depth = check_dim(input_shape[-1])`; if `input_shape[-1] == -1`, raise `ValueError("Expected inputs.shape[-1] to be known, saw shape: ...")`.
  - Create weights:
    - `_gate_kernel`: shape `[input_depth + units, 2 * units]`.
    - `_gate_bias`: shape `[2 * units]`, initializer `Ones`.
    - `_candidate_kernel`: shape `[input_depth + units, units]`.
    - `_candidate_bias`: shape `[units]`, initializer `Ones`.
- `call((x, state, att_score))`:
  - `gate_inputs = matmul(concat([x, state], 1), _gate_kernel)`, bias add.
  - `value = sigmoid(gate_inputs)`; split into `r, u`.
  - `candidate = matmul(concat([x, r * state], 1), _candidate_kernel)`; bias add; `c = activation(candidate)`.
  - If `att_score is None`: standard GRU update: `(1 - u) * state + u * c`.
  - Else if `att_type == "AUGRU"`:
    - `u = (1 - att_score) * u`.
    - `new_h = u * state + (1 - u) * c`.
  - Else (`AGRU`):
    - `new_h = (1 - att_score) * state + att_score * c`.
  - Returns `(new_h, new_h)` (output and new state).
- `zero_state(batch_size, dtype)`:
  - In eager mode, caches last zero state to avoid recomputation.
  - Uses `_zero_state_tensors` from TF rnn_cell_impl with `backend.name_scope`.
- `get_config()` serializes `units`, `att_type`, `initializer`, `activation`, `regularizer`.
- `create_ta(name, size, dtype)` returns `TensorArray`.
- `static_rnn_with_attention(cell, inputs, att_scores, init_state=None)`:
  - `cell` must be `AGRUCell`.
  - If `init_state` is None, uses `cell.get_initial_state` if available, else `cell.zero_state`.
  - Transposes inputs to time-major, loops in Python, calls cell per time step with `att_scores[:, time]` reshaped to `(-1, 1)`.
  - Returns stacked outputs `(batch, time, hidden)` and final state.
  - Note: uses `dtype` variable in `get_initial_state` branch, but `dtype` is undefined (bug in Python).
- `dynamic_rnn_with_attention(cell, inputs, att_scores, parallel_iterations=1, swap_memory=True, init_state=None)`:
  - Same initialization rules as static (same undefined `dtype` issue).
  - Uses TensorArray + `control_flow_ops.while_loop`.
  - Outputs stacked and transposed back to batch-major.
  - Sets static shape `[None, time_steps, dim_size(outputs, -1)]`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-layers/src/agru.rs`.
- Rust public API surface:
  - Rust `AGRU` struct exists but is not a TF-style cell; needs parity with `AGRUCell` and with sequence outputs.
  - Add `AGRUCell`-like API (`forward_step`, `zero_state`, `state_size`, `output_size`).
  - Add `static_rnn_with_attention` / `dynamic_rnn_with_attention` helpers (pure Rust loops).
- Data model mapping:
  - Python `att_type` (`AGRU`/`AUGRU`) → Rust enum.
  - Python `activation` → Rust activation layer or function.
  - Python `initializer`/`regularizer` → Rust `Initializer`/`Regularizer`.
- Feature gating: None (pure Rust implementation).
- Integration points: DIEN/sequence models that expect AGRU outputs.

**Implementation Steps (Detailed)**
1. Extend `monolith_layers::agru` to support both AGRU and AUGRU update formulas and optional `att_score=None` (standard GRU).
2. Add `AGRUCell` struct mirroring Python weight shapes and bias initialization (gate/candidate splits).
3. Implement `static_rnn_with_attention`:
   - Accept inputs `[batch, time, dim]`, attention `[batch, time]`, optional initial state.
   - Return outputs for all timesteps and final state.
4. Implement `dynamic_rnn_with_attention`:
   - In Rust, this is a loop, but preserve behavior and output shape.
5. Match error semantics for unknown input depth; enforce input_dim known at build.
6. Add config serialization for `AGRUCell` (units, att_type, initializer, activation, regularizer).

**Tests (Detailed)**
- Python tests: `monolith/native_training/layers/agru_test.py`.
- Rust tests: `monolith-rs/crates/monolith-layers/tests/agru_test.rs` (new).
- Cross-language parity test:
  - Fix weights, input, attention; compare per-timestep outputs for AGRU and AUGRU modes.

**Gaps / Notes**
- Python `static_rnn_with_attention` and `dynamic_rnn_with_attention` reference `dtype` without definition; decide whether to mimic or correct in Rust.
- Rust `AGRU` currently returns only final hidden state and uses a different attention update formula; needs alignment.

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

### `monolith/native_training/layers/agru_test.py`
<a id="monolith-native-training-layers-agru-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 112
- Purpose/role: Smoke tests for `AGRUCell` instantiation, serde, and static/dynamic attention RNN helpers.
- Key symbols/classes/functions: `AGRUTest` methods `test_agru_instantiate`, `test_agru_serde`, `test_agru_call`, `test_agru_static_rnn_call`, `test_agru_dynamic_rnn_call`.
- External dependencies: TensorFlow v1 session mode, NumPy.
- Side effects: Disables eager execution and v2 behavior in main guard.

**Required Behavior (Detailed)**
- `test_agru_instantiate`:
  - Uses `AGRUCell.params()` to build `InstantiableParams`.
  - Sets `units=10`, `activation=sigmoid`, `initializer=GlorotNormal`, then `instantiate()`.
  - Also constructs directly with `AGRUCell(units=10, activation=sigmoid, initializer=HeUniform)`.
  - Both instantiations must succeed.
- `test_agru_serde`:
  - `cfg = AGRUCell(...).get_config()` then `AGRUCell.from_config(cfg)` must succeed.
- `test_agru_call`:
  - Inputs: `data` shape `(100, 100)`, `state` shape `(100, 10)`, `attr` shape `(100, 1)`.
  - Calls `layer((data, state, attr))`, gets `(output, new_state)`; sums `new_state`.
  - Runs in a session after variable initialization.
- `test_agru_static_rnn_call`:
  - Inputs: `data` shape `(100, 20, 10)`, `attr` shape `(100, 20)`.
  - Calls `static_rnn_with_attention`, receives `(outputs, final_state)`.
  - Sums `final_state` (not the full outputs).
- `test_agru_dynamic_rnn_call`:
  - Inputs: random `data` shape `(100, 20, 10)` and `attr` shape `(100, 20)`.
  - Calls `dynamic_rnn_with_attention`, receives `(outputs, final_state)` and sums `final_state`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-layers/tests/agru_test.rs`.
- Rust public API surface: `AGRUCell` (or equivalent), `static_rnn_with_attention`, `dynamic_rnn_with_attention`.
- Data model mapping:
  - Params-based instantiation ↔ Rust config/builder.
  - `get_config`/`from_config` ↔ serde round-trip.
- Feature gating: None.
- Integration points: `monolith_layers::agru`.

**Implementation Steps (Detailed)**
1. Add Rust tests for params/builder instantiation and serde round-trip.
2. Add forward step test for `AGRUCell` with fixed inputs, compare output sum.
3. Add static and dynamic RNN tests; validate final state sum against Python.
4. Mirror test input shapes and attention shapes from Python.

**Tests (Detailed)**
- Python tests: `monolith/native_training/layers/agru_test.py`.
- Rust tests: `monolith-rs/crates/monolith-layers/tests/agru_test.rs` (new).
- Cross-language parity test:
  - Fix weights and inputs and compare final state sums for static/dynamic helpers.

**Gaps / Notes**
- Python tests do not assert exact values; Rust tests should add deterministic assertions.

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
