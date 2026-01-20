<!--
Source: task/request.md
Lines: 13919-14188 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/layers/mlp_test.py`
<a id="monolith-native-training-layers-mlp-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 78
- Purpose/role: Smoke tests for MLP instantiation, serialization, and forward call with batch normalization and mixed activation list.
- Key symbols/classes/functions: `MLPTest` methods `test_mlp_instantiate`, `test_mlp_serde`, `test_mlp_call`.
- External dependencies: TensorFlow v1 session mode, NumPy.
- Side effects: Checks internal `_stacked_layers` length.

**Required Behavior (Detailed)**
- `test_mlp_instantiate`:
  - Params-based instantiate with `output_dims=[1,3,4,5]`, `activations=None`, `initializers=GlorotNormal`.
  - Direct constructor with same output dims and `initializers=HeUniform`.
- `test_mlp_serde`:
  - Instantiate via params, `get_config` and `MLP.from_config(cfg)` should succeed.
- `test_mlp_call`:
  - Params-based instantiate with:
    - `output_dims=[100,50,10,1]`,
    - `enable_batch_normalization=True`,
    - `activations=['relu', tanh, PReLU, None]`,
    - `initializers=GlorotNormal`.
  - Input shape `(100,100)`; sums output.
  - Asserts `len(layer._stacked_layers) == 11`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-layers/tests/mlp_test.rs`.
- Rust public API surface: `MLP`, `MLPConfig`, `ActivationType`.
- Data model mapping:
  - Activation list includes mixed string/function/class; Rust should accept equivalent `ActivationType`.
  - `_stacked_layers` count corresponds to Dense + optional BN + activation for each layer; ensure layering logic matches.
- Feature gating: None.
- Integration points: `monolith_layers::mlp`.

**Implementation Steps (Detailed)**
1. Add Rust tests for constructor and config serialization.
2. Add forward test with batch normalization and mixed activations; assert output shape/sum.
3. Add check on internal layer count if exposed (or infer via config).

**Tests (Detailed)**
- Python tests: `monolith/native_training/layers/mlp_test.py`.
- Rust tests: `monolith-rs/crates/monolith-layers/tests/mlp_test.rs` (new).
- Cross-language parity test:
  - Fix weights and inputs; compare output sums and layer counts.

**Gaps / Notes**
- Python uses `tf.keras.layers.PReLU` class in activations list; Rust must map to `ActivationType::PReLU` with default params.

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

### `monolith/native_training/layers/multi_task.py`
<a id="monolith-native-training-layers-multi-task-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 448
- Purpose/role: Multi-task learning layers: MMoE (multi-gate mixture of experts) and SNR (sub-network routing with hard-concrete gates).
- Key symbols/classes/functions: `MMoE`, `SNR`, `hard_concrete_ste`.
- External dependencies: TensorFlow/Keras (Layer, activations/initializers/regularizers/constraints), Monolith (`MLP`, `Dense`, `add_layer_loss`, `with_params`).
- Side effects: Adds loss terms for gate balancing (MMoE) and L0 penalty (SNR).

**Required Behavior (Detailed)**
- `MMoE`:
  - `gate_type` in `{softmax, topk, noise_topk}`; `top_k` default 1.
  - `num_experts` inferred from `expert_output_dims` or activations/initializers if not provided.
  - Experts are MLPs; all expert output dims must share same last dim.
  - Gate input dim inferred from `input_shape` (supports TF shape objects).
  - Gate weights shape `(gate_input_dim, num_experts * num_tasks)`; optional noise weights if `noise_topk`.
  - `calc_gate`:
    - Linear gate logits; optional noise.
    - Softmax over experts; if topk modes, zero out non-topk and renormalize.
    - Returns gates with shape `(batch, num_experts, num_tasks)`.
  - `call`:
    - If inputs is tuple, `(expert_input, gate_input)` else both are inputs.
    - `expert_outputs = stack([expert(x)], axis=2)` -> `(batch, output_dim, num_experts)`.
    - `mmoe_output = matmul(expert_outputs, gates)` -> `(batch, output_dim, num_tasks)`.
    - If gate_type != softmax: adds CV-squared loss over gate importance.
    - Returns list of per-task outputs via `unstack(axis=2)`.
- `hard_concrete_ste`:
  - Clamps to [0,1] in forward; gradient is identity (STE).
- `SNR`:
  - `snr_type` in `{trans, aver}`; `aver` requires `in_subnet_dim == out_subnet_dim`.
  - `build` infers `num_in_subnet` and `in_subnet_dim` from input list shapes.
  - `log_alpha` shape `(num_route, 1)` where `num_route = num_in_subnet * num_out_subnet`.
  - Adds L0 loss: `sum(sigmoid(log_alpha - factor))` with `factor = beta * log(-gamma/zeta)`.
  - If `snr_type=trans`: `weight` shape `(num_route, block_size)`; else identity tiled.
  - `sample`:
    - If training: sample `u`, `s = sigmoid((logit(u)+log_alpha)/beta)`.
    - Else: `s = sigmoid(log_alpha)`.
    - Stretch to `[gamma,zeta]`, then clamp to [0,1] (STE optional).
  - `call`:
    - Multiply `weight` by `z`, reshape to block matrix, matmul concat(inputs), split into outputs.

**Rust Mapping (Detailed)**
- Target crate/module:
  - `monolith-rs/crates/monolith-layers/src/mmoe.rs` (MMoE).
  - `monolith-rs/crates/monolith-layers/src/snr.rs` (SNR).
- Rust public API surface:
  - `MMoE`, `MMoEConfig`, `GateType`/`Gate`, `SNR`, `SNRConfig`, `SNRType`.
- Data model mapping:
  - Python `gate_type` → Rust enum.
  - `top_k` and noise_topk behavior → Rust gating implementation.
  - `use_bias`, batch norm flags, and initializers for experts.
- Feature gating: None.
- Integration points: MLP, Dense, activation registry.

**Implementation Steps (Detailed)**
1. Align expert construction and gate input dim inference with Python.
2. Implement noise_topk gating and CV-squared loss term for topk modes.
3. Ensure SNR sampling uses same hard-concrete bounds and STE behavior.
4. Add config serialization for MMoE and SNR (activations, initializers, gate_type, etc.).

**Tests (Detailed)**
- Python tests: `monolith/native_training/layers/multi_task_test.py`.
- Rust tests: `monolith-rs/crates/monolith-layers/tests/multi_task_test.rs` (new).
- Cross-language parity test:
  - Fix weights/inputs and compare outputs for MMoE (softmax/topk/noise_topk) and SNR (trans/aver).

**Gaps / Notes**
- Python uses `add_loss` for CV-squared and L0 penalty; Rust must decide how to expose these losses.

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

### `monolith/native_training/layers/multi_task_test.py`
<a id="monolith-native-training-layers-multi-task-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 128
- Purpose/role: Smoke tests for MMoE and SNR layers (instantiation, serde, forward).
- Key symbols/classes/functions: `MultiTaskTest` methods `test_mmoe_instantiate`, `test_mmoe_serde`, `test_mmoe_call`, `test_snr_instantiate`, `test_snr_serde`, `test_snr_call`.
- External dependencies: TensorFlow v1 session mode, NumPy.
- Side effects: Runs TF sessions for forward calls.

**Required Behavior (Detailed)**
- MMoE:
  - Instantiate via params with `num_tasks=2`, `num_experts=3`, `expert_output_dims=[128,64,64]`, `expert_activations='relu'`, `expert_initializers=GlorotNormal`.
  - Direct constructor with same settings.
  - `test_mmoe_call`: uses `gate_type='topk'`, `top_k=2`, expert dims list-of-lists, input `(100,128)`, sums output list.
- SNR:
  - Instantiate via params with `num_out_subnet=3`, `out_subnet_dim=128`, `use_ste=False`.
  - Direct constructor with same values.
  - `test_snr_call`: `snr_type='aver'`, `mode=PREDICT`, four inputs `(100,128)` each; sums outputs.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-layers/tests/multi_task_test.rs`.
- Rust public API surface: `MMoE`, `MMoEConfig`, `SNR`, `SNRConfig`.
- Data model mapping:
  - `gate_type` and `top_k` → Rust gate config.
  - `snr_type='aver'` → `SNRType::Aver`.
- Feature gating: None.
- Integration points: `monolith_layers::mmoe`, `monolith_layers::snr`.

**Implementation Steps (Detailed)**
1. Add Rust tests for config round-trip for MMoE and SNR.
2. Add forward tests with matching input shapes and configurations.
3. Add deterministic assertions on output shapes/sums.

**Tests (Detailed)**
- Python tests: `monolith/native_training/layers/multi_task_test.py`.
- Rust tests: `monolith-rs/crates/monolith-layers/tests/multi_task_test.rs` (new).
- Cross-language parity test:
  - Fix weights and inputs; compare outputs for MMoE (topk) and SNR (aver).

**Gaps / Notes**
- Python uses list-of-lists expert dims for MMoE; ensure Rust supports heterogeneous expert configs.

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

### `monolith/native_training/layers/norms.py`
<a id="monolith-native-training-layers-norms-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 343
- Purpose/role: Normalization and multi-task gradient balancing utilities: custom BatchNorm, LayerNorm, and GradNorm.
- Key symbols/classes/functions: `BatchNorm`, `LayerNorm`, `GradNorm`.
- External dependencies: TensorFlow/Keras (`Layer`, `InputSpec`, initializers/regularizers), Monolith `add_layer_loss`.
- Side effects: Emits TF summary scalars/histograms; adds losses for moving mean/variance and GradNorm.

**Required Behavior (Detailed)**
- `BatchNorm`:
  - Tracks moving_mean and moving_variance; optional center/scale with beta/gamma weights.
  - In TRAIN mode:
    - Computes batch mean/variance (optionally stop-grad).
    - Replaces gradient for moving stats with current batch values.
    - Adds losses for moving mean/variance and logs summaries.
    - If `training_use_global_dist`: blends moving stats with current stats using `global_dist_momentum`.
  - In EVAL mode:
    - Uses stopped moving stats; logs summaries.
  - Returns `tf.nn.batch_normalization` with epsilon.
- `LayerNorm`:
  - Normalizes across last dimension per sample; applies beta/gamma with epsilon 1e-6.
  - Same logic for train/eval.
- `GradNorm`:
  - Given `losses` and `shared_inputs`, computes gradients wrt shared inputs, concatenates, and computes norms.
  - Softmax weights over tasks; computes `gnorm_loss` using absolute or relative difference vs average.
  - Returns `(gnorm_loss, weighted_loss)`; logs weights and gnorms.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-layers/src/normalization.rs`.
- Rust public API surface: `BatchNorm`, `LayerNorm`, `GradNorm`.
- Data model mapping:
  - Momentum/epsilon/renorm settings map to Rust BatchNorm fields.
  - GradNorm computes losses from provided grads (Rust currently expects grads, not tensors).
- Feature gating: None.
- Integration points: MLP/MMoE/LHUC use BatchNorm; GradNorm used in multi-task setups.

**Implementation Steps (Detailed)**
1. Align BatchNorm behavior with Python: moving stats, training/eval paths, optional stop-grad, and global-dist blending.
2. Ensure LayerNorm uses epsilon=1e-6 and per-sample normalization.
3. Adjust GradNorm API to accept losses and grads consistent with Python (or document differences).
4. Add config serialization for BatchNorm/LayerNorm/GradNorm.

**Tests (Detailed)**
- Python tests: `monolith/native_training/layers/norms_test.py`.
- Rust tests: `monolith-rs/crates/monolith-layers/tests/norms_test.rs` (new).
- Cross-language parity test:
  - Compare BatchNorm/LayerNorm outputs and GradNorm loss for fixed inputs.

**Gaps / Notes**
- Python uses TF summaries and add_layer_loss for moving stats; Rust lacks equivalent side effects.

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
