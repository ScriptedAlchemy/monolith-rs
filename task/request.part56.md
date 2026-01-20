<!--
Source: task/request.md
Lines: 12819-12986 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/layers/dense.py`
<a id="monolith-native-training-layers-dense-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 307
- Purpose/role: Custom Dense layer with optional kernel normalization, optimizer attachment to variables, partitioned variable support, and inactive-ReLU monitoring.
- Key symbols/classes/functions: `Dense(Layer)` with `add_weight`, `get_variable`, `build`, `call`, `compute_output_shape`, `get_config`.
- External dependencies: TensorFlow/Keras internals (`core_ops.dense`, `variable_ops.PartitionedVariable`, `base_layer_utils`, `InputSpec`, `K.track_variable`, `tensor_shape`), Monolith utils (`with_params`, `get_uname`).
- Side effects:
  - Attaches `.optimizer` attribute to variables created.
  - Tracks split variables in layer trainable/non-trainable weights.
  - Emits summary histogram for inactive ReLU counts when enabled.

**Required Behavior (Detailed)**
- Initialization:
  - If `input_shape` not provided and `input_dim` is, convert to `input_shape=(input_dim,)`.
  - Sets `units`, `activation=activations.get(activation)`, `use_bias`, `kernel_initializer`, `bias_initializer`, `kernel_regularizer`, `bias_regularizer`, `allow_kernel_norm`, `kernel_norm_trainable`, `partitioner`, `inactive_relu_monitor`, `inactive_relu_monitor_decay`, `optimizer`.
  - `input_spec = InputSpec(min_ndim=2)`, `supports_masking=True`.
- `add_weight` override:
  - Calls `super().add_weight(...)`, then sets `var.optimizer = self.optimizer`.
  - If `PartitionedVariable`, set optimizer on each shard.
- `get_variable` helper:
  - Wraps `tf.compat.v1.get_variable` inside current name scope with AUTO_REUSE.
  - Sets optimizer on variable(s).
  - Manually tracks variables with `K.track_variable` and appends to `_trainable_weights` or `_non_trainable_weights`, including split/partitioned variables.
- `build(input_shape)`:
  - Ensures dtype is floating/complex; otherwise `TypeError("Unable to build `Dense` layer with non-floating point dtype %s")`.
  - Requires last dimension known; otherwise `ValueError("The last dimension of the inputs to `Dense` should be defined. Found `None`.")`.
  - Creates kernel variable using `get_variable` seeded by `kernel_initializer` output.
  - If `allow_kernel_norm`:
    - Normalize kernel with `tf.nn.l2_normalize(axis=0, epsilon=1e-6)`.
    - If `kernel_norm_trainable`, create `trainable_kernel_norm` initialized with `tf.linalg.norm(init_kernel, axis=0)` and multiply normalized kernel by it.
  - If `use_bias`, add bias weight; else `bias=None`.
  - If `inactive_relu_monitor` and activation is ReLU:
    - Create non-trainable `inactive_relu_count_moving_avg` variable under `METRIC_VARIABLES` and `GLOBAL_VARIABLES`.
- `call(inputs)`:
  - Uses `core_ops.dense(inputs, kernel, bias, activation, dtype=compute_dtype)`.
  - If `inactive_relu_monitor`:
    - `inactive_relu_count = units - count_nonzero(output, axis=0)`.
    - Logs histogram `inactive_relu_count_moving_avg`.
    - Updates moving average with decay and uses control dependencies.
- `compute_output_shape`:
  - Requires last dim defined; otherwise `ValueError("The innermost dimension of input_shape must be defined, but saw: %s")`.
  - Output shape = input_shape[:-1] + units.
- `get_config`:
  - Serializes units, activation, use_bias, initializers, regularizers, `allow_kernel_norm`, `kernel_norm_trainable`, `partitioner`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-layers/src/dense.rs`.
- Rust public API surface:
  - `Dense` currently implements a linear layer with optional kernel norm (no activation in the layer).
  - Add a `DenseConfig` or wrapper to include activation and inactive-ReLU monitoring.
- Data model mapping:
  - Python `activation` → Rust `ActivationType` or `ActivationLayer` (in `mlp.rs`/`activation_layer.rs`).
  - Python `kernel_norm_trainable` → Rust `kernel_norm_trainable`.
  - Python `partitioner` and `optimizer` do not exist in Rust; require explicit non-TF equivalents or document omission.
- Feature gating: None.
- Integration points: MLP and any model configs that reference `Dense` directly.

**Implementation Steps (Detailed)**
1. Decide parity approach:
   - Option A: Add activation inside `Dense` (match Python call signature).
   - Option B: Keep linear `Dense` and ensure all call sites add `ActivationLayer` explicitly (document difference).
2. Add kernel norm behavior to match TF:
   - Normalize weights along axis 0, epsilon 1e-6.
   - If trainable, scale by per-output norm.
3. Add inactive ReLU monitoring equivalent (optional):
   - Track per-unit zero counts and exponential moving average; integrate with Rust metrics/logging.
4. Add config serialization to include activation, kernel/bias initializers, regularizers, allow_kernel_norm, kernel_norm_trainable.
5. Mirror error messages for invalid input dtypes and missing last dimension (as close as Rust allows).

**Tests (Detailed)**
- Python tests: `monolith/native_training/layers/dense_test.py`.
- Rust tests: `monolith-rs/crates/monolith-layers/tests/dense_test.rs` (new).
- Cross-language parity test:
  - Fix weights/bias and compare outputs for activation on/off and kernel_norm modes.

**Gaps / Notes**
- Python attaches `.optimizer` to TF variables and supports partitioned variables; Rust has no equivalent.
- Python Dense includes activation inside the layer; Rust `Dense` is linear-only today.

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

### `monolith/native_training/layers/dense_test.py`
<a id="monolith-native-training-layers-dense-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 147
- Purpose/role: Tests Dense instantiation, serialization, forward, kernel norm, inactive-ReLU monitoring, and variable partitioning.
- Key symbols/classes/functions: `DenseTest` methods `test_dense_instantiate`, `test_dense_serde`, `test_dense_call`, `test_dense_kernel_norm_call`, `test_inactive_relu_monitor`, `test_dense_with_explicit_partition`, `test_dense_with_implicit_partition`.
- External dependencies: TensorFlow v1 session mode, NumPy.
- Side effects: Uses graph collections and variable partitioning.

**Required Behavior (Detailed)**
- `test_dense_instantiate`:
  - Builds `Dense.params()` template, sets `units=100`, `activation=sigmoid`, `kernel_initializer=GlorotNormal`, instantiates.
  - Also constructs `Dense(...)` directly; both must succeed.
- `test_dense_serde`:
  - Instantiates via params, calls `get_config`, and `Dense.from_config(cfg)`.
- `test_dense_call`:
  - Creates Dense with sigmoid activation, input `(100, 100)` ones; sums output and runs session.
- `test_dense_kernel_norm_call`:
  - Dense with `allow_kernel_norm=True`, `kernel_norm_trainable=True`; runs forward without errors.
- `test_inactive_relu_monitor`:
  - Dense with `activation=relu` and `inactive_relu_monitor=True`.
  - After calling on a constant input, asserts graph contains node name `Dense/inactive_relu_count_moving_avg_1`.
- `test_dense_with_explicit_partition`:
  - Dense with explicit `partitioner` and kernel_norm enabled.
  - Input shape `(100, 294)`; validates output shape `(100, 1024)`.
  - Collects per-shard kernel dims (expected `[59, 59, 59, 59, 58]` but not asserted).
- `test_dense_with_implicit_partition`:
  - Uses `variable_scope` partitioner; Dense with `partitioner=None` to inherit scope.
  - Verifies kernel shard dims equal `[59, 59, 59, 59, 58]`.
  - Validates output shape `(100, 1024)`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-layers/tests/dense_test.rs`.
- Rust public API surface: `Dense`, activation handling, kernel norm config.
- Data model mapping:
  - Params-based instantiation ↔ Rust builder/config.
  - `get_config`/`from_config` ↔ serde round-trip.
- Feature gating: None.
- Integration points: `monolith_layers::dense`.

**Implementation Steps (Detailed)**
1. Add Rust tests for constructor + config serde round-trip.
2. Add forward tests for base Dense and kernel_norm-enabled Dense.
3. If activation is moved out of Dense in Rust, adapt tests to apply activation separately but keep parity cases documented.
4. Decide how to mirror partitioner behavior:
   - If not supported, add explicit test that documents the unsupported feature.
5. Add inactive-ReLU monitoring metrics if implemented; otherwise document absence.

**Tests (Detailed)**
- Python tests: `monolith/native_training/layers/dense_test.py`.
- Rust tests: `monolith-rs/crates/monolith-layers/tests/dense_test.rs` (new).
- Cross-language parity test:
  - Fixed weights/bias and compare output sums for kernel_norm on/off.

**Gaps / Notes**
- Python partitioner behavior has no Rust equivalent; needs explicit parity plan or documented limitation.
- The expected partition shard sizes are implicit to TF partitioner; Rust may not replicate.

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
