<!--
Source: task/request.md
Lines: 18652-18921 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/net_utils_test.py`
<a id="monolith-native-training-net-utils-test-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 94
- Purpose/role: Tests for `net_utils` helpers and `NodeAliveChecker`.
- Key symbols/classes/functions: `NetUtilsTest`.
- External dependencies: `unittest`, `unittest.mock`, `random`, `time`, `absl.logging`, `monolith.native_training.net_utils`.
- Side effects: Uses randomized sleep; modifies module-level `_FAILED_TIME` and `_DEAD_SET`.

**Required Behavior (Detailed)**
- Custom `socket` class emulates `socket.socket` with:
  - `connect` sleeping random duration in `[0, 2 * timeout]`.
  - Marks a failure when sleep > timeout, increments `_FAILED_TIME`, and adds addr to `_DEAD_SET`.
- `test_basic`:
  - Mocks `net_utils.socket.socket` to return custom socket.
  - Creates `NodeAliveChecker` with 5 localhost addrs.
  - Asserts:
    - `get_addrs()` matches input set.
    - `len(alive) == 5 - _FAILED_TIME`, `len(dead) == _FAILED_TIME`.
    - Alive set equals input set minus `_DEAD_SET`; dead set equals `_DEAD_SET`.
    - `all_nodes_alive()` true only if `_FAILED_TIME == 0`.
- `test_concat_ip_and_port`:
  - Asserts IPv4 and hostname formatting (no brackets) and IPv6 formatting (`[::1]:10`).
- `test_get_local_server_addr`:
  - Asserts result is non-`None`.
- `__main__`: runs `unittest.main()`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/src` (network utils tests).
- Rust public API surface: tests for `concat_ip_and_port`, `get_local_server_addr`, and `NodeAliveChecker`.
- Data model mapping: mock TCP connection attempts and timeouts.
- Feature gating: none.
- Integration points: verify `NodeAliveChecker` behavior under simulated timeouts.

**Implementation Steps (Detailed)**
1. Add deterministic tests for `concat_ip_and_port` with IPv4 and IPv6.
2. Add a test for local server addr non-empty.
3. Implement a mockable connector for `NodeAliveChecker` to emulate timeouts without real sockets.

**Tests (Detailed)**
- Python tests: `NetUtilsTest` in this file.
- Rust tests: create `net_utils_concat`, `net_utils_local_addr`, `node_alive_checker_mocked`.
- Cross-language parity test: compare IPv6 formatting and default addr shapes.

**Gaps / Notes**
- Randomized sleeps can make the Python test non-deterministic; Rust tests should be deterministic.
- `_FAILED_TIME` and `_DEAD_SET` are global and not reset between tests.

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

### `monolith/native_training/optimizers/adamom.py`
<a id="monolith-native-training-optimizers-adamom-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 68
- Purpose/role: Defines `AdamomOptimizer`, a TF v1 optimizer backed by a custom Monolith op (`resource_apply_adamom`).
- Key symbols/classes/functions: `AdamomOptimizer`.
- External dependencies: TensorFlow v1 optimizer APIs, `monolith.native_training.runtime.ops.gen_monolith_ops`.
- Side effects: Creates optimizer slot variables (`m`, `v`, `c`) on first use.

**Required Behavior (Detailed)**
- **`AdamomOptimizer(tf.compat.v1.train.Optimizer)`**
  - `__init__(learning_rate=5e-6, ada_decay=0.9999, mom_decay=0.99, epsilon=1e-6, weight_decay=0.0, use_locking=False, name="Adamom")`:
    - Stores parameters on instance.
    - `_learning_rate_tensor` initialized to `None`.
  - `_create_slots(var_list)`:
    - For each variable `v`, creates zero slots:
      - `"m"` with name scope `self._name + "/m"`.
      - `"v"` with name scope `self._name + "/v"`.
      - `"c"` with name scope `self._name + "/c"`.
  - `_prepare()`:
    - Resolves learning rate via `_call_if_callable`.
    - Converts to tensor named `"learning_rate"`.
  - `_resource_apply_dense(grad, var)`:
    - Retrieves slots `m`, `v`, `c`.
    - Calls `training_ops.resource_apply_adamom` with:
      - `var.handle`, `m.handle`, `v.handle`, `c.handle`,
      - `learning_rate` cast to `grad.dtype.base_dtype`,
      - `ada_decay`, `mom_decay`, `epsilon`, `weight_decay`,
      - `grad`,
      - `use_locking=self._use_locking`.
  - Sparse gradients: no `_resource_apply_sparse` override; relies on base-class behavior (likely unsupported).

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-optimizer/src`.
- Rust public API surface:
  - `AdamomOptimizer` with identical hyperparameters.
  - Slot state for `m`, `v`, `c`.
- Data model mapping:
  - Slot tensors stored alongside parameters; update rule must match custom op semantics.
- Feature gating:
  - If TF runtime backend enabled, use TF custom op; otherwise implement in Rust.
- Integration points:
  - Used by training loop in `MonolithBaseModel.create_model_fn`.

**Implementation Steps (Detailed)**
1. Confirm exact math used by `resource_apply_adamom` (check TF custom op source).
2. Implement slot creation and zero-initialization matching TF names (`/m`, `/v`, `/c`).
3. Implement dense update rule and weight decay handling.
4. Decide behavior for sparse gradients (error or unsupported).
5. Add tests reproducing Python values in `adamom_test.py`.

**Tests (Detailed)**
- Python tests: `monolith/native_training/optimizers/adamom_test.py`.
- Rust tests: add deterministic numeric test for a single variable update.
- Cross-language parity test: compare slot values and updated var after one step.

**Gaps / Notes**
- Custom op semantics are not visible in this file; parity depends on `resource_apply_adamom` implementation.

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

### `monolith/native_training/optimizers/adamom_test.py`
<a id="monolith-native-training-optimizers-adamom-test-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 57
- Purpose/role: Validates `AdamomOptimizer` slot creation and update values.
- Key symbols/classes/functions: `AdamomTest`.
- External dependencies: TensorFlow v1 test APIs, `monolith.native_training.optimizers.adamom`.
- Side effects: Disables eager execution in `__main__`.

**Required Behavior (Detailed)**
- `testBasic`:
  - Creates variable `v=[0.1]` and loss `loss = 0.12 * v`.
  - Optimizer: `AdamomOptimizer(learning_rate=0.1, weight_decay=0.01, ada_decay=0.99, mom_decay=0.9)`.
  - Runs one `minimize` step.
  - Reads all variables and asserts:
    - slot `m` ≈ `0.0121`
    - slot `c` ≈ `1.0`
    - slot `v` ≈ `0.014641`
    - variable `v` ≈ `0.090000336`
  - Expects exactly 4 variables (`m`, `v`, `c`, and the original variable).
- `__main__`:
  - `tf.compat.v1.disable_eager_execution()`, then `tf.test.main()`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-optimizer/src` (tests).
- Rust public API surface: `AdamomOptimizer` update step and slot access.
- Data model mapping: slot tensors must be accessible and comparable.
- Feature gating: custom op parity when TF runtime backend used.
- Integration points: training loop tests or direct optimizer tests.

**Implementation Steps (Detailed)**
1. Reproduce the one-step update with a single scalar variable.
2. Assert slot values and updated var match Python within tolerance.
3. Ensure slot naming/ordering does not affect test outcomes.

**Tests (Detailed)**
- Python tests: `AdamomTest` in this file.
- Rust tests: add `adamom_basic_update` golden test.
- Cross-language parity test: compare slot/var values for the same input and hyperparameters.

**Gaps / Notes**
- The numeric expectations depend on custom op semantics; confirm via TF op source.

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

### `monolith/native_training/optimizers/rmsprop.py`
<a id="monolith-native-training-optimizers-rmsprop-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 102
- Purpose/role: Defines `RmspropOptimizer`, a TF v1 optimizer backed by the custom `resource_apply_rmsprop` op with optional v2 behavior.
- Key symbols/classes/functions: `RmspropOptimizer`.
- External dependencies: TensorFlow v1 optimizer APIs, `monolith.native_training.runtime.ops.gen_monolith_ops`.
- Side effects: Creates optimizer slot variables (`m`, `v`) on first use.

**Required Behavior (Detailed)**
- **`RmspropOptimizer(tf.compat.v1.train.Optimizer)`**
  - `__init__(learning_rate=5e-6, beta1=0.99, beta2=0.999, epsilon=1e-8, weight_decay=0.0, use_locking=False, use_v2=False, name="Rmsprop")`:
    - Stores parameters on instance, including `use_v2`.
  - `_create_slots(var_list)`:
    - For each variable `v`, creates zero slots:
      - `"m"` with name scope `self._name + "/m"`.
      - `"v"` with name scope `self._name + "/v"`.
  - `_prepare()`:
    - Resolves learning rate via `_call_if_callable`.
    - Converts to tensor named `"learning_rate"`.
  - `_apply_dense(grad, var)`:
    - Always raises `NotImplementedError("Please use tf.compat.v1.disable_eager_execution() instead of tf.compat.v1.disable_v2_behavior()")`.
  - `_resource_apply_dense(grad, var)`:
    - Retrieves slots `m`, `v`.
    - Calls `training_ops.resource_apply_rmsprop` with:
      - `var.handle`, `m.handle`, `v.handle`,
      - `learning_rate` cast to `grad.dtype.base_dtype`,
      - `beta1`, `beta2`, `epsilon`, `weight_decay`,
      - `grad`,
      - `use_locking=self._use_locking`, `use_v2=self._use_v2`.
  - Sparse gradients: no `_resource_apply_sparse` override.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-optimizer/src`.
- Rust public API surface:
  - `RmspropOptimizer` with `use_v2` toggle.
  - Slot state for `m` and `v`.
- Data model mapping:
  - Slot tensors stored alongside parameters; update rule must match custom op semantics.
- Feature gating:
  - If TF runtime backend enabled, use TF custom op; otherwise implement in Rust.
- Integration points:
  - Used by training loop in `MonolithBaseModel.create_model_fn` or optimizer registry.

**Implementation Steps (Detailed)**
1. Confirm exact math used by `resource_apply_rmsprop` and its `use_v2` branch.
2. Implement slot creation and zero-initialization matching TF names (`/m`, `/v`).
3. Implement dense update rule and weight decay handling.
4. Decide sparse gradient behavior (error or unsupported).
5. Add tests reproducing Python values from `rmsprop_test.py` and `rmspropv2_test.py`.

**Tests (Detailed)**
- Python tests: `monolith/native_training/optimizers/rmsprop_test.py`, `monolith/native_training/optimizers/rmspropv2_test.py`.
- Rust tests: add deterministic numeric tests for both v1 and v2 behavior.
- Cross-language parity test: compare slot/var values for one update step.

**Gaps / Notes**
- Custom op semantics are not visible in this file; parity depends on `resource_apply_rmsprop` implementation.
- `_apply_dense` raises a hard error; Rust should surface an equivalent error if similar path exists.

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
