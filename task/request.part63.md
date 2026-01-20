<!--
Source: task/request.md
Lines: 14520-14845 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/learning_rate_functions.py`
<a id="monolith-native-training-learning-rate-functions-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 112
- Purpose/role: Defines learning rate schedule function objects (base class + polynomial decay) for optimizers and embedding slice configs.
- Key symbols/classes/functions: `LearningRateFunction`, `PolynomialDecay`.
- External dependencies: TensorFlow v1 (`tf.compat.v1.train.polynomial_decay`, `get_or_create_global_step`), `abc`.
- Side effects: None; uses global step when called.

**Required Behavior (Detailed)**
- `LearningRateFunction`:
  - Abstract `__call__` that must be overridden.
  - `__str__` prints class name and sorted `__dict__` params.
- `PolynomialDecay`:
  - Stores init params: `initial_learning_rate`, `decay_steps`, `end_learning_rate`, `power`, `cycle`, `name`.
  - `__call__` fetches `global_step = tf.compat.v1.train.get_or_create_global_step()` and returns `tf.compat.v1.train.polynomial_decay(...)`.
  - Uses TF’s polynomial decay semantics (including `cycle`).

**Rust Mapping (Detailed)**
- Target crate/module: N/A (no Rust equivalent yet).
- Rust public API surface: None.
- Data model mapping: If implemented, use a trait + struct for polynomial decay tied to training step.
- Feature gating: None.
- Integration points: Optimizer configs and embedding slice configs.

**Implementation Steps (Detailed)**
1. Decide where to place LR schedules in Rust (optimizer module or training crate).
2. Implement `PolynomialDecay` with explicit step parameter (Rust lacks TF global_step).
3. Provide string formatting for config parity if required.

**Tests (Detailed)**
- Python tests: None in repo.
- Rust tests: Add unit tests for decay values at known steps.
- Cross-language parity test: Compare decay outputs for fixed steps.

**Gaps / Notes**
- Python relies on TF global step; Rust will need explicit step input or global trainer context.

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

### `monolith/native_training/learning_rate_functions_test.py`
<a id="monolith-native-training-learning-rate-functions-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 76
- Purpose/role: Tests PolynomialDecay schedule and its integration with an optimizer.
- Key symbols/classes/functions: `PolynomialDecayTest.test_basic`, `test_dense_optimizer`.
- External dependencies: TensorFlow v1 session/optimizers, NumPy.
- Side effects: Uses global_step and updates variables via Adagrad.

**Required Behavior (Detailed)**
- `test_basic`:
  - Creates global_step, increments twice, and checks decay outputs.
  - With `initial_learning_rate=0.01`, `decay_steps=10`, `end_learning_rate=0.11`:
    - At global_step=1: expects 0.02.
    - At global_step=2: expects 0.03.
  - Ensures `__str__` equality between two identical PolynomialDecay instances.
- `test_dense_optimizer`:
  - Uses PolynomialDecay as `learning_rate` for `AdagradOptimizer`.
  - Applies grads to two variables for 3 steps.
  - Verifies updated values match expected arrays.

**Rust Mapping (Detailed)**
- Target crate/module: N/A until learning rate schedules are implemented.
- Rust public API surface: PolynomialDecay schedule + optimizer integration.
- Data model mapping: global_step must be explicit in Rust.
- Feature gating: None.
- Integration points: Optimizer implementations (e.g., Adagrad).

**Implementation Steps (Detailed)**
1. Implement PolynomialDecay in Rust with explicit step input.
2. Add tests validating decay values for known steps (0,1,2).
3. If an optimizer exists, add integration test similar to Adagrad update.

**Tests (Detailed)**
- Python tests: `monolith/native_training/learning_rate_functions_test.py`.
- Rust tests: `monolith-rs/crates/monolith-optim/tests/learning_rate_functions_test.rs` (new) or similar.
- Cross-language parity test:
  - Compare decay values at fixed steps.

**Gaps / Notes**
- Python uses TF global_step; Rust needs explicit step or trainer context.

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

### `monolith/native_training/logging_ops.py`
<a id="monolith-native-training-logging-ops-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 56
- Purpose/role: Thin wrappers around custom logging/metrics TF ops (timestamps, timers, machine health).
- Key symbols/classes/functions: `tensors_timestamp`, `emit_timer`, `machine_info`, `check_machine_health`.
- External dependencies: TensorFlow, absl flags, custom ops `gen_monolith_ops`.
- Side effects: Registers a global flag `monolith_default_machine_info_mem_limit`.

**Required Behavior (Detailed)**
- `tensors_timestamp(tensors)`: returns `(tensors, timestamp)` via `monolith_tensors_timestamp`.
- `emit_timer(key, value, tags=None)`:
  - Formats tags as `"k=v|k2=v2"`, passes to `monolith_metric_v2`.
  - Returns TF op.
- `machine_info(mem_limit=None, shared_name=None)`:
  - Uses default flag if `mem_limit` is None.
  - Calls `monolith_machine_info` with `mem_limit`, `name`, `shared_name`.
- `check_machine_health(machine_info_tensor)`:
  - Returns scalar string tensor from `monolith_check_machine_health`.

**Rust Mapping (Detailed)**
- Target crate/module: N/A (custom TF ops not wired in Rust).
- Rust public API surface: None.
- Data model mapping: Would require TF runtime bindings.
- Feature gating: TF-runtime only if added.
- Integration points: metrics/logging pipeline.

**Implementation Steps (Detailed)**
1. Decide whether to expose these custom ops in Rust TF-runtime backend.
2. If yes, add FFI bindings and wrappers with identical signatures.
3. Provide a config/flag equivalent for default mem_limit.

**Tests (Detailed)**
- Python tests: none.
- Rust tests: add once bindings exist.
- Cross-language parity test: validate emitted tags and machine health output.

**Gaps / Notes**
- Requires custom TF ops; currently no Rust bindings.

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

### `monolith/native_training/logging_ops_test.py`
<a id="monolith-native-training-logging-ops-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 57
- Purpose/role: Tests logging ops custom TF wrappers (timestamp, timer, machine health).
- Key symbols/classes/functions: `LoggingOpsTest.test_tensors_timestamp`, `test_emit_timer`, `test_machine_health`, `test_machine_health_oom`.
- External dependencies: TensorFlow v1, absl flags, `logging_ops_pb2`.
- Side effects: Mutates global flag `monolith_default_machine_info_mem_limit`.

**Required Behavior (Detailed)**
- `test_tensors_timestamp`:
  - Calls `tensors_timestamp` twice and asserts newer timestamp >= old.
- `test_emit_timer`:
  - Calls `emit_timer("test", 0.0)` and evaluates op.
- `test_machine_health`:
  - Sets mem_limit high; `check_machine_health` returns empty bytes.
- `test_machine_health_oom`:
  - Sets mem_limit=0; `check_machine_health` returns serialized proto with status `OUT_OF_MEMORY`.

**Rust Mapping (Detailed)**
- Target crate/module: N/A (custom TF ops not implemented).
- Rust public API surface: None.
- Data model mapping: Would require TF runtime bindings and protobuf parsing.
- Feature gating: TF-runtime only.
- Integration points: logging/metrics pipeline.

**Implementation Steps (Detailed)**
1. Add Rust bindings for logging ops if TF runtime backend is enabled.
2. Add tests mirroring timestamp monotonicity and machine health outcomes.
3. Parse protobuf in Rust to validate OOM status.

**Tests (Detailed)**
- Python tests: `monolith/native_training/logging_ops_test.py`.
- Rust tests: N/A until bindings exist.
- Cross-language parity test: Validate proto outputs for machine health.

**Gaps / Notes**
- Depends on custom ops and protobufs not yet exposed in Rust.

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

### `monolith/native_training/losses/batch_softmax_loss.py`
<a id="monolith-native-training-losses-batch-softmax-loss-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 57
- Purpose/role: Computes batch softmax loss for retrieval-style training.
- Key symbols/classes/functions: `batch_softmax_loss`.
- External dependencies: TensorFlow.
- Side effects: None.

**Required Behavior (Detailed)**
- Inputs:
  - `query` shape `(batch_size, k)`, `item` shape `(batch_size, k)`.
  - `item_step_interval` shape `(batch_size,)`.
  - `r` weights (interest) same length as batch.
  - `normalize` (default True), `temperature` (default 1.0).
- Validation:
  - `temperature` must be > 0 else raise `ValueError("temperature should be positive, while got ...")`.
- Computation:
  - Optional L2-normalize query/item along axis 1.
  - `similarity = query @ item^T / temperature`.
  - Clamp `item_step_interval` to at least 1.0, compute `item_frequency = 1 / item_step_interval`.
  - Adjust similarity: `exp(similarity - log(item_frequency))`.
  - Loss: `-sum(r * log(diag(similarity) / reduce_sum(similarity, axis=1)))`.

**Rust Mapping (Detailed)**
- Target crate/module: N/A (no Rust loss implementation yet).
- Rust public API surface: loss function in training/optimizer crate.
- Data model mapping: Tensor ops for matmul, diag, log, exp.
- Feature gating: None.
- Integration points: training loss computation.

**Implementation Steps (Detailed)**
1. Implement batch_softmax_loss in Rust with the same math and shape checks.
2. Ensure numerical stability around log/exp and item_frequency.
3. Add input normalization option.

**Tests (Detailed)**
- Python tests: `monolith/native_training/losses/batch_softmax_loss_test.py`.
- Rust tests: new test in `monolith-rs/crates/monolith-training/tests`.
- Cross-language parity test: compare loss for fixed inputs.

**Gaps / Notes**
- Requires loss module placement decision in Rust.

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

### `monolith/native_training/losses/batch_softmax_loss_test.py`
<a id="monolith-native-training-losses-batch-softmax-loss-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 35
- Purpose/role: Single test for batch_softmax_loss numeric output.
- Key symbols/classes/functions: `BatchSoftmaxLossTest.test_batch_softmax_loss`.
- External dependencies: TensorFlow, NumPy.
- Side effects: None.

**Required Behavior (Detailed)**
- Creates random `query` and `item` tensors `(batch=4, dim=3)`.
- `item_step_interval` is random integers in `[1,10)`, `r` is ones.
- Calls `batch_softmax_loss` and asserts loss equals `6.5931373`.

**Rust Mapping (Detailed)**
- Target crate/module: N/A until loss is implemented.
- Rust public API surface: batch_softmax_loss.
- Data model mapping: Tensor operations and RNG.
- Feature gating: None.
- Integration points: training loss module.

**Implementation Steps (Detailed)**
1. Implement loss and a deterministic test by seeding RNG or using fixed inputs.
2. Match Python numeric output if using the same fixed inputs.

**Tests (Detailed)**
- Python tests: `monolith/native_training/losses/batch_softmax_loss_test.py`.
- Rust tests: add deterministic equivalent.
- Cross-language parity test: compare loss for fixed inputs.

**Gaps / Notes**
- Python test uses random inputs without setting a seed but asserts a fixed value; likely flaky. Prefer fixing inputs in Rust and note the discrepancy.

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
