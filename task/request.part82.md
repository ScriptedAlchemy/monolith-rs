<!--
Source: task/request.md
Lines: 18922-19224 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/optimizers/rmsprop_test.py`
<a id="monolith-native-training-optimizers-rmsprop-test-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 77
- Purpose/role: Tests for `RmspropOptimizer` (v1 behavior) including GPU/CPU consistency.
- Key symbols/classes/functions: `RmspropTest`, `build_graph`.
- External dependencies: TensorFlow, `tensorflow.python.framework.test_util`, `monolith.native_training.optimizers.rmsprop`.
- Side effects: Uses GPU if available and compares CPU/GPU results.

**Required Behavior (Detailed)**
- `build_graph()`:
  - Creates variable `v=[0.1]`, loss `0.12 * v`.
  - Optimizer: `RmspropOptimizer(learning_rate=0.1, weight_decay=1, beta1=0.9, beta2=0.9, epsilon=0.1)`.
  - Returns `opt.minimize(loss)`.
- `testBasic`:
  - Runs training once on GPU (if available) in a fresh graph with `test_util.use_gpu()`.
  - Checks that variables are placed on `/device:GPU:0` when GPU is available.
  - After one step, asserts:
    - `m` ≈ `0.06794526153774846`
    - `v` ≈ `0.00484`
    - variable `v` ≈ `0.03205473846225154`
    - exactly 3 variables (m, v, and the variable).
  - Runs the same graph on CPU (`test_util.force_cpu()`), checks `/device:CPU:0`.
  - Asserts CPU results equal GPU results.
- `__main__`: disables eager execution and runs `tf.test.main()`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-optimizer/src` (tests).
- Rust public API surface: `RmspropOptimizer` update step and slot access.
- Data model mapping: slot tensors accessible and comparable.
- Feature gating: GPU/CPU parity checks only if backend supports both.
- Integration points: optimizer correctness tests.

**Implementation Steps (Detailed)**
1. Reproduce one-step update with the same hyperparameters.
2. Assert slot and variable values match Python within tolerance.
3. If GPU backend exists, ensure CPU/GPU parity.

**Tests (Detailed)**
- Python tests: `RmspropTest` in this file.
- Rust tests: add `rmsprop_basic_update` and optional CPU/GPU parity test.
- Cross-language parity test: compare numeric results for the same input.

**Gaps / Notes**
- GPU/CPU parity is asserted; Rust must match this if both backends available.

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

### `monolith/native_training/optimizers/rmspropv2_test.py`
<a id="monolith-native-training-optimizers-rmspropv2-test-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 112
- Purpose/role: Tests for `RmspropOptimizer` with `use_v2=True` (alternate update rule).
- Key symbols/classes/functions: `RmspropTest`, `build_graph`.
- External dependencies: TensorFlow, `tensorflow.python.framework.test_util`, `monolith.native_training.optimizers.rmsprop`.
- Side effects: Uses GPU if available and compares CPU/GPU results.

**Required Behavior (Detailed)**
- `build_graph()`:
  - Creates variable `v=[0.1]`, loss `0.12 * v`.
  - Optimizer: `RmspropOptimizer(learning_rate=0.1, weight_decay=1, beta1=0.9, beta2=0.9, epsilon=0.1, use_v2=True)`.
  - Returns `opt.minimize(loss)`.
- `testBasic`:
  - Runs training on GPU (if available), checks `/device:GPU:0` placement.
  - After one step, asserts:
    - `m` ≈ `0.068750`
    - `v` ≈ `0.0484`
    - variable `v` ≈ `0.031250`
    - exactly 3 variables (m, v, variable).
  - Runs on CPU and asserts results equal GPU.
- `testWeightDecay`:
  - Duplicates `testBasic` (same graph and expectations).
- `__main__`: disables eager execution and runs `tf.test.main()`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-optimizer/src` (tests).
- Rust public API surface: `RmspropOptimizer` with `use_v2` flag.
- Data model mapping: slot tensors accessible and comparable.
- Feature gating: GPU/CPU parity checks conditional on backend support.
- Integration points: optimizer correctness tests.

**Implementation Steps (Detailed)**
1. Reproduce one-step update with `use_v2=True`.
2. Assert slot and variable values match Python within tolerance.
3. If GPU backend exists, ensure CPU/GPU parity.
4. Decide whether to keep duplicated `testWeightDecay` semantics or consolidate.

**Tests (Detailed)**
- Python tests: `RmspropTest` in this file.
- Rust tests: add `rmsprop_v2_basic_update` and optional CPU/GPU parity test.
- Cross-language parity test: compare numeric results for the same input.

**Gaps / Notes**
- `testWeightDecay` duplicates `testBasic` (same expectations).

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

### `monolith/native_training/optimizers/shampoo.py`
<a id="monolith-native-training-optimizers-shampoo-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 207
- Purpose/role: TODO (manual)
- Key symbols/classes/functions: TODO (manual)
- External dependencies: TODO (manual)
- Side effects: TODO (manual)

**Required Behavior (Detailed)**
- Define the **functional contract** (inputs → outputs) for every public function/class.
- Enumerate **error cases** and exact exception/messages that callers rely on.
- Capture **config + env var** behaviors (defaults, overrides, precedence).
- Document **I/O formats** used (proto shapes, TFRecord schemas, JSON, pbtxt).
- Note **threading/concurrency** assumptions (locks, async behavior, callbacks).
- Identify **determinism** requirements (seeds, ordering, float tolerances).
- Identify **performance characteristics** that must be preserved.
- Enumerate **metrics/logging** semantics (what is logged/when).

**Rust Mapping (Detailed)**
- Target crate/module: TODO (manual)
- Rust public API surface: TODO (manual)
- Data model mapping: TODO (manual)
- Feature gating: TODO (manual)
- Integration points: TODO (manual)

**Implementation Steps (Detailed)**
1. Extract all public symbols + docstrings; map to Rust equivalents.
2. Port pure logic first (helpers, utils), then stateful services.
3. Recreate exact input validation and error semantics.
4. Mirror side effects (files, env vars, sockets) in Rust.
5. Add config parsing and defaults matching Python behavior.
6. Add logging/metrics parity (field names, levels, cadence).
7. Integrate into call graph (link to downstream Rust modules).
8. Add tests and golden fixtures; compare outputs with Python.
9. Document deviations (if any) and mitigation plan.

**Tests (Detailed)**
- Python tests: TODO (manual)
- Rust tests: TODO (manual)
- Cross-language parity test: TODO (manual)

**Gaps / Notes**
- TODO (manual)

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

### `monolith/native_training/prefetch_queue.py`
<a id="monolith-native-training-prefetch-queue-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 379
- Purpose/role: TODO (manual)
- Key symbols/classes/functions: TODO (manual)
- External dependencies: TODO (manual)
- Side effects: TODO (manual)

**Required Behavior (Detailed)**
- Define the **functional contract** (inputs → outputs) for every public function/class.
- Enumerate **error cases** and exact exception/messages that callers rely on.
- Capture **config + env var** behaviors (defaults, overrides, precedence).
- Document **I/O formats** used (proto shapes, TFRecord schemas, JSON, pbtxt).
- Note **threading/concurrency** assumptions (locks, async behavior, callbacks).
- Identify **determinism** requirements (seeds, ordering, float tolerances).
- Identify **performance characteristics** that must be preserved.
- Enumerate **metrics/logging** semantics (what is logged/when).

**Rust Mapping (Detailed)**
- Target crate/module: TODO (manual)
- Rust public API surface: TODO (manual)
- Data model mapping: TODO (manual)
- Feature gating: TODO (manual)
- Integration points: TODO (manual)

**Implementation Steps (Detailed)**
1. Extract all public symbols + docstrings; map to Rust equivalents.
2. Port pure logic first (helpers, utils), then stateful services.
3. Recreate exact input validation and error semantics.
4. Mirror side effects (files, env vars, sockets) in Rust.
5. Add config parsing and defaults matching Python behavior.
6. Add logging/metrics parity (field names, levels, cadence).
7. Integrate into call graph (link to downstream Rust modules).
8. Add tests and golden fixtures; compare outputs with Python.
9. Document deviations (if any) and mitigation plan.

**Tests (Detailed)**
- Python tests: TODO (manual)
- Rust tests: TODO (manual)
- Cross-language parity test: TODO (manual)

**Gaps / Notes**
- TODO (manual)

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

### `monolith/native_training/prefetch_queue_test.py`
<a id="monolith-native-training-prefetch-queue-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 305
- Purpose/role: TODO (manual)
- Key symbols/classes/functions: TODO (manual)
- External dependencies: TODO (manual)
- Side effects: TODO (manual)

**Required Behavior (Detailed)**
- Define the **functional contract** (inputs → outputs) for every public function/class.
- Enumerate **error cases** and exact exception/messages that callers rely on.
- Capture **config + env var** behaviors (defaults, overrides, precedence).
- Document **I/O formats** used (proto shapes, TFRecord schemas, JSON, pbtxt).
- Note **threading/concurrency** assumptions (locks, async behavior, callbacks).
- Identify **determinism** requirements (seeds, ordering, float tolerances).
- Identify **performance characteristics** that must be preserved.
- Enumerate **metrics/logging** semantics (what is logged/when).

**Rust Mapping (Detailed)**
- Target crate/module: TODO (manual)
- Rust public API surface: TODO (manual)
- Data model mapping: TODO (manual)
- Feature gating: TODO (manual)
- Integration points: TODO (manual)

**Implementation Steps (Detailed)**
1. Extract all public symbols + docstrings; map to Rust equivalents.
2. Port pure logic first (helpers, utils), then stateful services.
3. Recreate exact input validation and error semantics.
4. Mirror side effects (files, env vars, sockets) in Rust.
5. Add config parsing and defaults matching Python behavior.
6. Add logging/metrics parity (field names, levels, cadence).
7. Integrate into call graph (link to downstream Rust modules).
8. Add tests and golden fixtures; compare outputs with Python.
9. Document deviations (if any) and mitigation plan.

**Tests (Detailed)**
- Python tests: TODO (manual)
- Rust tests: TODO (manual)
- Cross-language parity test: TODO (manual)

**Gaps / Notes**
- TODO (manual)

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
