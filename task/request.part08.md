<!--
Source: task/request.md
Lines: 19763-23700 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/nested_tensors.py`
<a id="monolith-native-training-nested-tensors-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 110
- Purpose/role: Utility for flattening arbitrarily nested structures containing Tensors/RaggedTensors/primitive constants into a flat tensor list and reconstructing the nested structure after computation.
- Key symbols/classes/functions: `_iterate`, `NestedTensors`.
- External dependencies: `tensorflow`, `itertools`, `copy`.
- Side effects:
  - Mutates dict/list/tuple structures passed into `_iterate` (and thus into `NestedTensors.__init__`) by replacing leaves with object IDs.

**Required Behavior (Detailed)**
- **`_iterate(nested, action)`** (internal helper):
  - Recurses through nested structures; for each leaf, applies `action(leaf)` and replaces leaf with the return value.
  - Handles:
    - `None`: no action, returns `None`.
    - `list`/`tuple`: maps recursively; preserves tuple vs list types.
    - `dict`: iterates `items()` and assigns `nested[k] = _iterate(v, action)` (mutates dict).
    - other: replaces with `action(nested)`.
- **`NestedTensors(nested)`**
  - Stores original `nested` reference as `self._nested`.
  - Initializes:
    - `_id_mapping: dict[obj_id -> (kind_idx, list_index)]`
    - `_ragged_tensors: List[tf.RaggedTensor]`
    - `_tensors: List[tf.Tensor]`
    - `_other_objs: List[Any]` for allowed constants/objects.
  - Calls `_iterate(self._nested, self._add_tensor)` so **all leaves become object IDs**.
- **`_add_tensor(tensor)`**:
  - For each unique object ID:
    - `tf.Tensor` → kind 0, appended to `_tensors`.
    - `tf.RaggedTensor` → requires `ragged_rank == 1`; else raises
      `ValueError("Nested tensor doesn't support nested RaggedTensor.")`; kind 1, appended to `_ragged_tensors`.
    - `(bool, int, str, tf.Variable, None)` → kind 2, appended to `_other_objs` (preserved).
    - Otherwise raises `ValueError("Tensor is not supported. {}".format(tensor))`.
  - Returns the object ID (an int), which replaces the leaf.
- **`get_tensors()`**
  - Flattens ragged tensors into `(values, row_splits)` pairs.
  - Returns `self._tensors + flatten_ragged_tensors` as a single list of `tf.Tensor`.
- **`get_nested_result(tensors)`**
  - Splits `tensors` into:
    - `tensors[:len(self._tensors)]` for dense tensors,
    - `flatten_ragged_tensors = tensors[len(self._tensors):]`.
  - Asserts `len(flatten_ragged_tensors) == len(self._ragged_tensors) * 2`.
  - Reconstructs ragged tensors via `_flatten_to_ragged`.
  - Uses `_id_mapping` to replace object IDs in a deep-copied nested structure with the
    corresponding tensor/ ragged tensor / other object.
  - Returns reconstructed nested structure with original container shapes.
- **`_convert_ragged_to_tensors(ragged)`**:
  - Returns `(ragged.values, ragged.row_splits)`.
- **`_convert_tensors_to_ragged(values, row_splits)`**:
  - Returns `tf.RaggedTensor.from_row_splits(values, row_splits, validate=False)`.
- **`_ragged_to_flatten(ragged_tensors)`**:
  - Returns flattened list `[values0, row_splits0, values1, row_splits1, ...]`.
- **`_flatten_to_ragged(tensors)`**:
  - Takes even/odd positions as `values` and `row_splits`; reconstructs list of ragged tensors.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-tensor/src` (nested tensor utilities).
- Rust public API surface:
  - `NestedTensors` struct with `get_tensors()` and `get_nested_result()`.
  - Internal recursion helper for nested structures.
- Data model mapping:
  - Support for dense tensors + ragged tensors with `values`/`row_splits`.
  - Preserve primitive constants and variables when rebuilding.
- Feature gating: none (unless ragged tensors are optional in backend).
- Integration points:
  - Use in async/queue or feature pipelines where nested structures must be flattened for execution.

**Implementation Steps (Detailed)**
1. Implement a `NestedValue` enum for nested structures (list/tuple/map/leaf).
2. Track object IDs and build mapping identical to Python (kind indices 0/1/2).
3. Support ragged tensors only with rank 1; error on higher rank.
4. Implement flatten/unflatten with identical ordering (values then row_splits).
5. Add round-trip tests mirroring Python coverage.

**Tests (Detailed)**
- Python tests: `monolith/native_training/nested_tensors_test.py`.
- Rust tests: add unit tests for:
  - basic dict/tuple nesting,
  - constants-only nesting (no tensors),
  - ragged tensor round-trip,
  - ragged rank error path.
- Cross-language parity test: round-trip nested structures and compare to Python evaluation.

**Gaps / Notes**
- `_iterate` mutates dicts in place; `NestedTensors` does **not** copy input `nested` before replacing leaves.
- Order of dict iteration affects reconstruction order; Python preserves insertion order.

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

### `monolith/native_training/nested_tensors_test.py`
<a id="monolith-native-training-nested-tensors-test-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 57
- Purpose/role: Unit tests for `NestedTensors` flatten/unflatten behavior.
- Key symbols/classes/functions: `NestedTensorTest`.
- External dependencies: TensorFlow, `monolith.native_training.nested_tensors`.
- Side effects: Uses TF test harness; disables eager execution in `__main__`.

**Required Behavior (Detailed)**
- `testBasic`:
  - Creates nested structure with tensors `{ "a": ones([]), "b": (ones([]), ones([])) }`.
  - Replaces flattened tensors with zeros; reconstructs and asserts dict equals `{"a": 0, "b": (0, 0)}`.
- `testConstant`:
  - Uses nested constants only (`{"a": {"b": 2}}`).
  - Expects `get_tensors()` returns empty list and reconstruction returns the same constants.
- `testRaggedTensor`:
  - Round-trip a ragged tensor `tf.ragged.constant([[], [1], [2, 3]])`.
  - Asserts reconstructed value equals original ragged list.
- `testRaggedTensorWithPlaceHolder`:
  - Creates ragged tensor, gets flattened tensors, builds placeholders for each flattened tensor.
  - Calls `get_nested_result(tensors)` (no explicit assert).
  - Implicitly validates that placeholder creation doesn't break `get_nested_result` call path.
- `__main__`:
  - Calls `tf.compat.v1.disable_eager_execution()` and runs `tf.test.main()`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-tensor/src` (tests alongside `NestedTensors`).
- Rust public API surface: test helpers to validate nested round-trip and ragged support.
- Data model mapping: ragged tensors must support `values` + `row_splits`.
- Feature gating: ragged tests conditional on ragged support.
- Integration points: ensure tests reflect the same ordering and reconstruction semantics as Python.

**Implementation Steps (Detailed)**
1. Port test cases to Rust equivalents (basic nesting, constants-only, ragged round-trip).
2. Add a placeholder/shape-only test if the backend supports placeholders or uninitialized tensors.
3. Ensure test runner matches eager/graph expectations (if applicable).

**Tests (Detailed)**
- Python tests: `NestedTensorTest` in this file.
- Rust tests: implement `nested_tensors_basic`, `nested_tensors_constants`, `nested_tensors_ragged`.
- Cross-language parity test: compare round-trip output for identical inputs.

**Gaps / Notes**
- `testRaggedTensorWithPlaceHolder` has no assertions; it only exercises the code path.

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

### `monolith/native_training/net_utils.py`
<a id="monolith-native-training-net-utils-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 133
- Purpose/role: Network utility helpers for address formatting, local IP discovery, and threaded liveness checks for host:port addresses.
- Key symbols/classes/functions: `NodeAliveChecker`, `is_ipv6_address`, `concat_ip_and_port`, `get_local_ip`, `is_ipv4_supported`, `get_local_server_addr`, `AddressFamily`.
- External dependencies: `socket`, `threading`, `queue.Queue`, `ipaddress`, `absl.logging`.
- Side effects:
  - Opens TCP sockets to check liveness.
  - Spawns threads in `NodeAliveChecker.__init__`.
  - Prints connection errors to stdout.

**Required Behavior (Detailed)**
- **`NodeAliveChecker(addrs, timeout=1, num_thread=10)`**
  - Initializes with:
    - `_addrs` list, `_timeout`, `_num_thread`.
    - `_lock`, `_alive` set, `_dead` set.
    - `_q = Queue()` populated with all `addrs`.
  - Immediately starts `_start()` (blocks until all threads finish).
- **`_ping(addr)`**
  - Expects `addr` string in `host:port` form; uses `addr.rsplit(':', 1)` and `ip.strip('[]')`.
  - Determines IPv6 via `is_ipv6_address`.
  - Creates `socket.socket(AF_INET6 if IPv6 else AF_INET, SOCK_STREAM)` and `settimeout(timeout)`.
  - On successful `connect`, adds `addr` to `_alive`.
  - On exception:
    - Prints `cannot connect to {addr}, because {err}`.
    - Adds `addr` to `_dead`.
  - Always closes socket if created.
- **`_check_open()`**
  - Drains `_q` using `get_nowait()` in a loop; for each addr calls `_ping`.
  - Exits on `queue.Empty`.
- **`_start()`**
  - Spawns `_num_thread` threads running `_check_open()`.
  - Joins all threads (synchronous completion).
- **`all_nodes_alive()`** returns `len(_dead) == 0`.
- **`get_dead_nodes()`** returns `list(_dead)` (order arbitrary).
- **`get_alive_nodes()`** returns `list(_alive)` (order arbitrary).
- **`get_addrs()`** returns `_addrs` (original list).
- **`is_ipv6_address(ip)`**
  - Uses `ipaddress.ip_address(ip)`; returns False on `ValueError`, else `ip_obj.version == 6`.
- **`concat_ip_and_port(ip, port)`**
  - If not IPv6, returns `"ip:port"`.
  - If IPv6, returns `"[ip]:port"`.
- **`get_local_ip()`**
  - Tries `socket.getaddrinfo(gethostname(), None)[0][4][0]`.
  - On `socket.gaierror`, retries with `family=AF_INET6`.
- **`is_ipv4_supported()`**
  - Returns `not is_ipv6_address(get_local_ip())`.
- **`get_local_server_addr(port)`**
  - Returns `concat_ip_and_port(get_local_ip(), port)`.
  - Docstring notes IPv4 hosts return `gethostbyname(gethostname())` equivalent.
- **`AddressFamily`**
  - Constants: `IPV4 = 'ipv4'`, `IPV6 = 'ipv6'`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/src` (network utils).
- Rust public API surface:
  - `NodeAliveChecker` equivalent (synchronous check on construction or explicit `run()`).
  - Helpers for IPv4/IPv6 detection and addr formatting.
- Data model mapping:
  - Socket API via `std::net` (`TcpStream::connect_timeout`).
  - Address parsing should accept bracketed IPv6 addresses.
- Feature gating: none.
- Integration points: used by distributed training/serving orchestration.

**Implementation Steps (Detailed)**
1. Implement `is_ipv6_address` using `std::net::IpAddr` parse.
2. Implement `concat_ip_and_port` with IPv6 bracket handling.
3. Implement `get_local_ip` with fallback IPv6 resolution.
4. Implement `NodeAliveChecker` with thread pool + shared sets protected by mutex.
5. Preserve behavior of printing on connection failure (stdout).
6. Add tests for IPv6 formatting and local addr retrieval.

**Tests (Detailed)**
- Python tests: `monolith/native_training/net_utils_test.py`.
- Rust tests: add unit tests for concat and local addr; add mocked checker tests.
- Cross-language parity test: compare formatting and IPv6 detection results.

**Gaps / Notes**
- `NodeAliveChecker` performs all checks in `__init__` and blocks until complete.
- Address parsing assumes `host:port` or `[ipv6]:port` format; raw IPv6 with port and no brackets may not parse correctly.

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

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 207
- Purpose/role: Implements a dense-only Shampoo optimizer variant with eigen-based preconditioning, warmup blending, and AdaGrad-style normalization.
- Key symbols/classes/functions: `eigen_inverse_root`, `apply_sparse_precond`, `ShampooOptimizer`.
- External dependencies: TensorFlow v1 optimizer APIs, `tf.linalg.eigh`, `tensorflow.python.ops.state_ops`, `io_ops`.
- Side effects: Creates multiple slot variables per tensor dimension and updates them on each step.

**Required Behavior (Detailed)**
- **`eigen_inverse_root(mat, p, head, tail, damping=1e-3)`** (`@tf.function`):
  - Computes eigen-decomposition `eval, evec = tf.linalg.eigh(mat)`.
  - `alpha = -1.0 / p`, `dim = mat.shape[0]`.
  - `non_zero = tf.where(eval > damping)`.
  - `zeros` is:
    - `min(non_zero)` if non_zero not empty,
    - else `0`.
  - `eval_p = pow(max(eval, damping), alpha)`.
  - Head/tail selection adjustments:
    - If `head + tail > dim`: set `zeros = 0`, `head = dim`, `tail = 0`.
    - Else if `zeros + head + tail > dim`: set `zeros = dim - head - tail`.
  - Selects eigenvalues/vectors:
    - `eval_ht = concat(eval_p[zeros:zeros+head], eval_p[dim-tail:])`
    - `evec_ht = concat(evec[:, zeros:zeros+head], evec[:, dim-tail:], axis=1)`
  - `offset`:
    - `0.0` if `zeros + head + tail == dim`,
    - else `mean(eval[zeros+head:dim-tail])`.
  - Returns `(evec_ht, eval_ht - offset, offset)`.
- **`apply_sparse_precond(tensor, pvec, pval, offset)`**
  - Applies preconditioner using `tensordot`:
    - `tensor_tmp_1 = tensordot(tensor, pvec, axes=[[0],[0]])`
    - `tensor_tmp_2 = tensor_tmp_1 * pval`
    - `tensor_tmp_3 = tensordot(tensor_tmp_2, pvec, axes=[[-1],[-1]])`
  - Computes `tensor_transpose = transpose(tensor, perm=[1..rank-1,0])`.
  - Returns `tensor_tmp_3 + tensor_transpose * offset`.
- **`ShampooOptimizer(tf.compat.v1.train.Optimizer)`**
  - `__init__(learning_rate=0.03, beta_1=0.9, beta_2=1.0, warmup=5000, tau_1=200, tau_2=20, eigen_head=100, eigen_tail=100, damping_epsilon=1e-3, use_locking=False, name="Shampoo", **kwargs)`:
    - Stores parameters; passes `**kwargs` to base optimizer.
  - `_create_slots(var_list)`:
    - For each variable and each dimension `i`:
      - Creates `s{i}` and `g{i}` slots of shape `[dim, dim]`.
      - Computes `eigens = min(dim, eigen_head + eigen_tail)` and creates:
        - `pvec{i}` `[dim, eigens]`
        - `pval{i}` `[eigens]`
        - `o{i}` scalar.
    - Creates zero slots `d`, `m`, `pm` for each variable.
  - `_resource_apply_dense(grad, var)`:
    - Computes:
      - `global_step = tf.cast(tf.train.get_global_step(), int32)`.
      - `if_update_stat = (global_step % tau_2 == 0)`.
      - `if_warmed_up = global_step > warmup`.
      - `if_update_precond = if_warmed_up AND (global_step % tau_1 == 0)`.
      - `warmup_rate = clamp(global_step / warmup - 1, 0, 1)`.
      - `if_stat_momentum = beta_2 < 1.0 - 1e-10`.
    - For each dimension `i`:
      - `g_t`: if_update_stat → assign `tensordot(grad, grad, axes=[axes, axes])`, else identity.
      - `s_t`: if_stat_momentum → `s = beta_2*s + (1-beta_2)*g_t`, else `s += g_t`.
      - `pvec/pval/offset`: if_update_precond → compute from `eigen_inverse_root(s_t, 2*rank, ...)` and assign; else identity.
      - Updates `grad_precond = apply_sparse_precond(grad_precond, pvec_t, pval_t, offset_t)`.
    - Accumulates `d_t = d + grad*grad`.
    - `m_t = beta_1*m + (1-beta_1)*grad*rsqrt(d_t + 1e-30)`.
    - `pm_t = beta_1*pm + (1-beta_1)*grad_precond`.
    - `update_diag = lr * m_t`.
    - `update_second = lr * norm(m_t) / (norm(pm_t) + 1e-10) * pm_t`.
    - `var_t`:
      - If warmed up: `var -= (1-warmup_rate)*update_diag + warmup_rate*update_second`.
      - Else: `var -= update_diag`.
    - Returns `tf.group(*ops)` of all state updates.
  - `_resource_apply_sparse(grad, var)`:
    - `raise tf.no_op()` (note: this raises an op, likely unintended but must match).

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-optimizer/src`.
- Rust public API surface:
  - `ShampooOptimizer` with identical hyperparameters.
  - Helper functions for eigen inverse root and preconditioning.
- Data model mapping:
  - Slot tensors for each dimension (`s`, `g`, `pvec`, `pval`, `o`, `d`, `m`, `pm`).
  - Requires eigen decomposition and matrix operations.
- Feature gating:
  - This optimizer is heavy; consider optional feature or backend requirement (eigen decomposition support).
- Integration points:
  - Training loop should only enable for dense tensors.

**Implementation Steps (Detailed)**
1. Implement `eigen_inverse_root` and `apply_sparse_precond` with identical axis semantics.
2. Implement slot creation per dimension and scalar slots `d/m/pm`.
3. Mirror update scheduling via `tau_1`, `tau_2`, `warmup`, and global step.
4. Implement warmup blending and normalization exactly.
5. Decide handling of sparse gradients (match Python error).
6. Add tests that validate slot shapes and a small numeric step.

**Tests (Detailed)**
- Python tests: none in this repo for Shampoo.
- Rust tests: add shape/slot tests and a small-step numeric sanity check.
- Cross-language parity test: optional, requires TF reference run.

**Gaps / Notes**
- `_resource_apply_sparse` raises an op (`tf.no_op()`), which likely throws; Rust should match behavior or document deviation.
- Eigen decomposition and slicing logic is subtle; deterministic ordering must be preserved.

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

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 379
- Purpose/role: Implements queue-based prefetching and async execution helpers that support mixed CPU/GPU tensors and nested structures.
- Key symbols/classes/functions: `_GPUCompatiblePaddingFIFOQueue`, `_FIFOQueue`, `_MultiFIFOQueue`, `MultiQueueRunner`, `EnqueueHook`, `enqueue_dicts_with_queue_return`, `AsyncPushHook`, `AsyncFunctionMgr`.
- External dependencies: TensorFlow queue APIs (`data_flow_ops`, `gen_data_flow_ops`), `nested_tensors`, `utils.check_ops_dependence`, `absl.logging`.
- Side effects:
  - Creates TF queue resources (CPU and/or GPU).
  - Starts QueueRunner threads via hooks.
  - Adds control dependencies to enforce async op order.

**Required Behavior (Detailed)**
- **`_GPUCompatiblePaddingFIFOQueue`** (QueueBase):
  - Wraps `padding_fifo_queue_v2` but allows placement on CPU or GPU.
  - Validates `dtypes`/`shapes` length; raises `ValueError` if mismatch.
  - `enqueue_many` / `dequeue_many` are **not supported** and raise `NotImplementedError`.
- **`_FIFOQueue(dense_list, capacity=2, queue_name="prefetch_queue")`**:
  - `dense_list` must be a list; raises:
    - `ValueError` if `dense_list` is `None`,
    - `TypeError` if not a list.
  - Creates `_GPUCompatiblePaddingFIFOQueue` inside `tf.init_scope()`.
  - `enqueue_op = queue.enqueue(flatten_tensor_list)`.
  - `dequeue()` returns list of tensors; if single element, returns `[tensor]`.
- **`_MultiFIFOQueue(dense_list, capacity=2, queue_name="prefetch_queue")`**:
  - Splits tensors by device (`"GPU"` substring in `tensor.device`).
  - Always creates CPU queue; creates GPU queue only if GPU tensors exist.
  - `enqueue_op` is `tf.group` of all queue `enqueue_op`s.
  - `queue` property:
    - If only one queue, returns it.
    - If multiple queues, raises `NotImplementedError`.
  - `dequeue()`:
    - If one queue, returns its dequeue.
    - Else dequeues all queues and merges tensors back to original order using saved indices.
  - `size()`:
    - If multiple queues, returns CPU queue size (assumes synchronized enqueue/dequeue).
  - `queues` property returns list of queue resources.
- **`MultiQueueRunner`**:
  - `_init_from_args` accepts `queue` as list:
    - Builds grouped `close_op` and `cancel_op`.
  - `queue` property raises `NotImplementedError` when multi-queue.
  - `name` property returns first queue name in multi-queue case.
- **`EnqueueHook(q)`**:
  - Wraps `MultiQueueRunner(q.queues, [q.enqueue_op])`.
  - Starts threads in `after_create_session`.
- **`enqueue_dicts_with_queue_return(tensors, capacity=1, queue_name="prefetch_queue")`**
  - If `capacity == 0`, returns `(tensors, None)` with no queue.
  - Flattens nested tensors via `NestedTensors`.
  - Creates `_MultiFIFOQueue` with flattened tensors.
  - Dequeues in `tf.init_scope()` and rebuilds original nested structure.
  - Returns `(nested_result, queue)`.
- **`AsyncPushHook(queue, ops)`**:
  - `begin`: stores `queue.size()`.
  - `before_run`: returns `SessionRunArgs(run_ops)` only after queue initialized.
  - `after_run`: sets `_queue_init` once queue size > 0.
  - `end`: drains queue by running `run_ops` while `queue_size > 0`.
- **`AsyncFunctionMgr(is_async=True)`**:
  - `add_async_function(target, args=None, kwargs=None, is_async=None, queue_name="async_queue")`:
    - If `is_async` is False, runs `target(*args, **kwargs)` synchronously.
    - If async:
      - Appends dummy constant tensor to `args` to avoid empty input lists.
      - Enqueues `(args, kwargs)` via `enqueue_dicts_with_queue_return`.
      - Builds `run_ops = target(*args[:-1], **kwargs)` under control deps on dummy tensor and dummy op.
      - Adds `AsyncPushHook(queue, run_ops)`.
      - Calls `utils.check_ops_dependence(queue.enqueue_op.name, dummy_op.name)` to validate dependencies.
      - Returns `queue.enqueue_op`.
  - `hooks` property returns list of hooks.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src` (queue + async helpers).
- Rust public API surface:
  - Queue wrapper that supports nested tensors and optional CPU/GPU split.
  - Async function manager that enqueues inputs and runs ops via hooks.
- Data model mapping:
  - Preserve nested tensor structure using Rust equivalent of `NestedTensors`.
  - Separate queues per device; merge outputs in original order.
- Feature gating:
  - GPU queue support depends on backend; may be optional.
- Integration points:
  - Used by `NativeContext.add_async_function` and estimator hooks.

**Implementation Steps (Detailed)**
1. Implement a queue abstraction with padding and fixed shapes.
2. Implement split/merge logic for CPU/GPU tensors.
3. Add EnqueueHook equivalents to manage background threads.
4. Port `enqueue_dicts_with_queue_return` to flatten nested structures and rebuild.
5. Implement AsyncFunctionMgr with dummy tensor injection and dependency checks.
6. Add tests for nested structures, capacity 0, and async behavior.

**Tests (Detailed)**
- Python tests: `monolith/native_training/prefetch_queue_test.py`.
- Rust tests: add unit tests for queue behavior, nesting, async manager, and control-dependency handling.
- Cross-language parity test: compare nested structures and async side effects.

**Gaps / Notes**
- `_MultiFIFOQueue` assumes CPU/GPU queues are dequeued together; this is noted as a limitation in comments.
- Device detection uses `"GPU" in tensor.device` string matching.

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

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 305
- Purpose/role: Tests for queue helpers, nested tensor enqueue/dequeue, and async function manager.
- Key symbols/classes/functions: `GPUCompatiblePaddingFIFOQueueTests`, `FIFOQueueTest`, `PrefetchTest`, `AsyncManagerTest`.
- External dependencies: TensorFlow, `tensorflow.python.framework.test_util`, `nested_tensors`, `prefetch_queue`.
- Side effects: Uses GPU if available; runs MonitoredSessions.

**Required Behavior (Detailed)**
- **GPUCompatiblePaddingFIFOQueueTests**
  - `testEnqueueAndDequeue`:
    - Enqueues three float scalars on GPU; dequeues and validates device placement and arithmetic results.
  - `testGPUQueueCPUTensor`:
    - Creates CPU tensors, enqueues to GPU queue, dequeues on CPU; validates results.
  - `testMultiEnqueueAndDequeue`:
    - Enqueues tuple `(int32, float32)` and checks values in order.
  - `testIdentityHelper`:
    - Ensures `tf.identity` on GPU and queue enqueue/dequeue work; checks value `2`.
- **FIFOQueueTest**
  - `test_fifo_queue_data`:
    - Enqueues dense + ragged tensors; verifies round-trip values.
  - `test_fifo_queue_capacity`:
    - Enqueues/dequeues 4 items with capacity 4; validates values.
- **PrefetchTest**
  - **NOTE**: There are two methods named `test_enqueue_dicts_with_queue_return`; the second overrides the first, so only the second runs.
  - First (shadowed) version:
    - Enqueues dense/ragged dicts with capacity 3, validates output across multiple enqueue/dequeue cycles.
  - Second (effective) version:
    - Enqueues nested dicts containing tensors, strings, and `None`.
    - Asserts string and None entries preserved before session run, then runs queue and validates tensor values.
  - `test_enqueue_dicts_with_control_flow`:
    - Uses control dependency (`v.assign_add(1)`) and ensures it executes when enqueuing.
  - `test_enqueue_with_zero_capacity`:
    - `capacity=0` returns original tensors; validates values.
  - `test_estimator_prefetch`:
    - Uses Estimator `predict` with enqueue hook; verifies predicted values 0..19.
- **AsyncManagerTest**
  - `testBasic`: async adds once even after two enqueue runs (value = 1).
  - `testSync`: synchronous add yields value = 1 after one run.
  - `testEmptyInput`: async function with no args still works (value = 1 after two runs).

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src` (tests).
- Rust public API surface: queue helpers, nested enqueue/dequeue, async manager hooks.
- Data model mapping: ragged tensor support and CPU/GPU queueing if backend supports it.
- Feature gating: GPU-specific tests conditional on backend support.
- Integration points: estimator-style predict loops or equivalent.

**Implementation Steps (Detailed)**
1. Port queue tests for enqueue/dequeue ordering and device placement.
2. Add nested tensor round-trip tests (dense + ragged).
3. Add tests for `capacity=0` passthrough.
4. Add async function manager tests for async vs sync behavior.
5. Document/handle duplicate test name if porting to Rust.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: mirror test classes where possible; use deterministic inputs.
- Cross-language parity test: compare nested results and async side effects.

**Gaps / Notes**
- Duplicate method name in `PrefetchTest` hides the first test; only the second runs in Python.

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

### `monolith/native_training/ps_benchmark.py`
<a id="monolith-native-training-ps-benchmark-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 273
- Purpose/role: Implements a parameter-server throughput benchmark task that measures PS performance and selects top PS nodes.
- Key symbols/classes/functions: `BenchmarkConfig`, `_BenchmarkWorkerHook`, `_DummyCheckpointSaverHook`, `PsBenchMarkTask`.
- External dependencies: TensorFlow, `native_task.NativeTask`, `utils.ps_device`, `cpu_training` (tests), `logging_ops`, `service_discovery` (imported, unused).
- Side effects:
  - Mutates `BenchmarkConfig.ps_list` after benchmark completion.
  - Writes TF variables and metrics; uses session hooks.

**Required Behavior (Detailed)**
- **`BenchmarkConfig` dataclass**
  - Fields:
    - `ps_list: List`
    - `num_ps_required: int`
    - `num_workers: int`
    - `index: int`
    - `benchmark_secs: float = 60.0`
    - `ps_str_overridden: str = ""` (comma-separated override list; skips benchmark).
- **`_BenchmarkWorkerHook(SessionRunHook)`**
  - Creates variables in scope `_SCOPE_NAME`:
    - `_result` (string), `_ready` (bool list), `_done` (bool list),
      `_make_ready`/`_make_done` assigns, `_result_placeholder` + `_result_assign`.
  - `after_create_session`:
    - Marks self ready; waits until `sum(ready) >= int(num_workers * 0.9)`.
    - Sleeps 1 second, records `_start_time`.
  - `before_run`:
    - If `ps_str_overridden` set, assigns `_result` to override value.
    - Reads `_result`; if non-empty, raises `tf.errors.OutOfRangeError` to stop.
  - `after_run`:
    - Logs duration and requests stop.
  - `end`:
    - Marks done; waits until all workers done (timeout 10s).
    - If `index == 0` and no override:
      - Reads `throughput_tensor` (mean tensor metric).
      - Sorts PS nodes by throughput.
      - Logs per-PS throughput and then adjusts throughput by IP (sums for same IP).
      - Selects top `num_ps_required` PS entries and writes comma-separated string to `_result`.
    - Waits until `_result` is non-empty, then replaces `bm_config.ps_list` with selected list.
  - `_wait(cond, timeout=3600)` polls condition every 0.5s.
- **`_DummyCheckpointSaverHook`**
  - Extends `CheckpointSaverHook`, but overrides lifecycle methods to no-op.
  - Default `checkpoint_dir` is `$HOME/tmp` if not provided.
  - `_save` returns False (prevents saving).
- **`PsBenchMarkTask(NativeTask)`**
  - `params()` adds `bm_config`.
  - `create_input_fn` returns dataset of constant vector `[0.12, 0.23, 0.34, 0.45]` repeating and prefetching.
  - `create_model_fn`:
    - For each PS in `bm_config.ps_list`, creates a 256×256 variable on that PS.
    - Builds a while loop that performs heavy math (splits/concat/sqrt) until `benchmark_secs` elapsed.
    - Computes throughput `j / (ts_now - ts_before)` per PS.
    - Uses `tf.compat.v1.metrics.mean_tensor(tf.stack(throughputs))`.
    - Adds `_BenchmarkWorkerHook` and `_DummyCheckpointSaverHook`.
    - `PREDICT` returns predictions `0.0`.
    - `TRAIN/EVAL` returns EstimatorSpec with loss 0 and train_op group of metric update + global step increment.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src` (benchmark task).
- Rust public API surface:
  - `BenchmarkConfig` struct.
  - Benchmark task or command that measures PS throughput.
- Data model mapping:
  - Use equivalent session hooks / callbacks for multi-worker coordination.
- Feature gating:
  - PS benchmark likely only for TF runtime backend; may be optional.
- Integration points:
  - PS device placement logic (`ps_device`) and distributed training orchestration.

**Implementation Steps (Detailed)**
1. Port `BenchmarkConfig` and its semantics (override skipping).
2. Implement a benchmark task that measures throughput per PS (or stub with TF backend only).
3. Implement worker coordination and result selection logic (including IP aggregation).
4. Ensure `ps_list` is mutated to selected subset after completion.
5. Add tests mirroring Python behavior for override and selection count.

**Tests (Detailed)**
- Python tests: `monolith/native_training/ps_benchmark_test.py`.
- Rust tests: add tests for override behavior and list truncation.
- Cross-language parity test: compare selected PS list with identical throughput inputs (if TF backend available).

**Gaps / Notes**
- Imports `logging_ops` and `service_discovery` but does not use them in this file.
- Benchmark uses heavy TF ops; not suitable for pure Candle backend without substantial rework.

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

### `monolith/native_training/ps_benchmark_test.py`
<a id="monolith-native-training-ps-benchmark-test-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 57
- Purpose/role: Tests `PsBenchMarkTask` behavior in local CPU training.
- Key symbols/classes/functions: `PsBenchmarkTest`.
- External dependencies: TensorFlow, `cpu_training.local_train`, `utils.get_test_tmp_dir`.
- Side effects: Runs local training which mutates `bm_config.ps_list`.

**Required Behavior (Detailed)**
- `testBasic`:
  - Creates `BenchmarkConfig` with `ps_list=["ps0","ps1"]`, `num_ps_required=1`, `num_workers=1`, `index=0`, `benchmark_secs=1.0`.
  - Calls `cpu_training.local_train` with `num_ps=2`.
  - Asserts `len(p.bm_config.ps_list) == 1` after run.
- `testSkipBenchmark`:
  - Same config but sets `ps_str_overridden="overridden"`.
  - After training, expects `p.bm_config.ps_list[0] == "overridden"`.
- `__main__`: disables eager execution and runs test via `absl.app`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src` (tests).
- Rust public API surface: PS benchmark task and local training harness.
- Data model mapping: `BenchmarkConfig` with override string.
- Feature gating: likely TF runtime only.
- Integration points: local training driver for PS benchmark.

**Implementation Steps (Detailed)**
1. Provide a local training harness for benchmark task (or mock).
2. Ensure `ps_list` is reduced to `num_ps_required` entries.
3. Ensure override string bypasses benchmark.

**Tests (Detailed)**
- Python tests: `PsBenchmarkTest` in this file.
- Rust tests: add unit tests for selection and override behavior.
- Cross-language parity test: compare resulting `ps_list` for identical config.

**Gaps / Notes**
- Test relies on `cpu_training.local_train` which is not yet mapped in Rust.

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

### `monolith/native_training/ragged_utils.py`
<a id="monolith-native-training-ragged-utils-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 29
- Purpose/role: Provides a faster cached alternative to `RaggedTensor.value_rowids()` using a custom op.
- Key symbols/classes/functions: `fused_value_rowids`.
- External dependencies: TensorFlow, `gen_monolith_ops.monolith_fused_value_rowids`.
- Side effects: Caches result on the `RaggedTensor` instance via `monolith_fused_value_rowids` attribute.

**Required Behavior (Detailed)**
- **`fused_value_rowids(rt)`**
  - Validates `rt` is a `tf.RaggedTensor`; otherwise raises `ValueError("rt must be RaggedTensor")`.
  - If `rt` lacks attribute `monolith_fused_value_rowids`, computes it once via:
    - `ops.monolith_fused_value_rowids(rt.row_splits)`.
  - Returns the cached tensor (same object on subsequent calls).

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-tensor/src`.
- Rust public API surface: `fused_value_rowids(ragged)` helper.
- Data model mapping:
  - Ragged tensor must expose `row_splits`.
  - Cache results on ragged tensor wrapper if possible.
- Feature gating: requires custom op or Rust implementation of row-id computation.
- Integration points: used in ragged pipelines and feature preprocessing.

**Implementation Steps (Detailed)**
1. Implement row-id computation or bind to TF custom op when TF backend enabled.
2. Add caching on ragged tensor wrapper to avoid recomputation.
3. Preserve error message for non-ragged input.

**Tests (Detailed)**
- Python tests: `monolith/native_training/ragged_utils_test.py`.
- Rust tests: add caching test and expected row-ids for a sample ragged tensor.
- Cross-language parity test: compare value_rowids output and object identity if applicable.

**Gaps / Notes**
- Attribute-based caching mutates the ragged tensor object; Rust should use a wrapper or external cache.

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

### `monolith/native_training/ragged_utils_test.py`
<a id="monolith-native-training-ragged-utils-test-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 32
- Purpose/role: Tests `fused_value_rowids` caching and output correctness.
- Key symbols/classes/functions: `RaggedUtilsTestCase`.
- External dependencies: TensorFlow, `ragged_utils`.
- Side effects: Disables eager execution in `__main__`.

**Required Behavior (Detailed)**
- `test_basic`:
  - Creates `rt = tf.ragged.constant([[], [1], [2, 3]])`.
  - Calls `fused_value_rowids` twice; asserts returned objects are identical (`is`).
  - Asserts values equal `[1, 2, 2]`.
- `__main__`: disables eager execution and runs `tf.test.main()`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-tensor/src` (tests).
- Rust public API surface: `fused_value_rowids` and caching behavior.
- Data model mapping: ragged tensor representation with `row_splits`.
- Feature gating: custom op or Rust implementation required.
- Integration points: ragged pipelines using row-id mapping.

**Implementation Steps (Detailed)**
1. Add a test that calls `fused_value_rowids` twice and checks cache hit semantics (if applicable).
2. Verify output row-ids for a sample ragged tensor.

**Tests (Detailed)**
- Python tests: `RaggedUtilsTestCase` in this file.
- Rust tests: add `ragged_utils_basic` test.
- Cross-language parity test: compare row-id outputs for identical ragged input.

**Gaps / Notes**
- Python relies on object identity for caching; Rust may need explicit cache handles.

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

### `monolith/native_training/remote_predict_ops.py`
<a id="monolith-native-training-remote-predict-ops-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 0
- Purpose/role: Empty placeholder module.
- Key symbols/classes/functions: none.
- External dependencies: none.
- Side effects: none.

**Required Behavior (Detailed)**
- This module is empty; importing it should have no side effects.

**Rust Mapping (Detailed)**
- Target crate/module: none required.
- Rust public API surface: none.
- Data model mapping: none.
- Feature gating: none.
- Integration points: none.

**Implementation Steps (Detailed)**
1. No implementation needed unless future Python adds content.

**Tests (Detailed)**
- Python tests: none.
- Rust tests: none.
- Cross-language parity test: not applicable.

**Gaps / Notes**
- Keep stub to preserve import paths if referenced elsewhere.

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

### `monolith/native_training/restore_test.py`
<a id="monolith-native-training-restore-test-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 240
- Purpose/role: Integration tests for partial and full checkpoint restore across parameter servers (with and without PS monitor).
- Key symbols/classes/functions: `_generate_config`, `_get_id_tensor`, `PartialRestoreTest`.
- External dependencies: TensorFlow distributed server APIs, `basic_restore_hook`, `hash_table_ops`, `save_utils`, `utils`.
- Side effects: Creates local TF servers, writes checkpoints to `TEST_TMPDIR`.

**Required Behavior (Detailed)**
- **`_generate_config(servers, job_name=utils.PS_JOB_NAME)`**
  - Builds a `ClusterDef` with one job and tasks derived from `server.target` (strip `grpc://`).
  - Returns `ConfigProto` with `experimental.share_session_state_in_clusterspec_propagation = True`.
- **`_get_id_tensor(x)`** returns `tf.constant(x, dtype=tf.int64)`.
- **`PartialRestoreTest.build_graph()`**
  - On `ps_device(0)`:
    - Creates `global_step`, increments, `v0`, `op0`, hash table 0, assign_add, lookup.
  - On `ps_device(1)`:
    - Creates `v1`, `op1`, hash table 1, assign_add, lookup.
  - Returns `(train_op, v0, v1, lookup0, lookup1)` where `train_op` groups ops.
- **`test_restore_with_ps_monitor`**
  - Uses `PartialRecoverySaver` with `PsMonitor(2)` and `NoFirstSaveCheckpointSaverHook`.
  - Uses `HashTableCheckpointSaverListener` and `HashTableCheckpointRestorerListener`.
  - Creates two local servers, saves checkpoint after one step.
  - Runs a second session without restore hooks to mutate variables.
  - Creates new servers and restores full checkpoint; asserts values back to 1.
  - Creates a cluster with only one of the original PS servers and restores partially; asserts v0/lookup0 restored to 2 while v1/lookup1 restored to 1.
- **`test_restore_without_ps_monitor`**
  - Same as above but without `PsMonitor`:
    - Save checkpoint after one step.
    - Mutate values in a second session.
    - Restore all variables in a third session; assert values back to 1.
- `__main__`: disables eager execution and runs `tf.test.main()`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src` (checkpoint restore tests).
- Rust public API surface: restore hooks, saver utilities, hash table checkpoint listeners.
- Data model mapping: distributed cluster config and PS device placement.
- Feature gating: TF runtime backend required.
- Integration points: `save_utils` and `hash_table_ops` parity.

**Implementation Steps (Detailed)**
1. Provide Rust equivalents for `PartialRecoverySaver`, `PsMonitor`, and restore hooks.
2. Implement hash table checkpoint save/restore listeners for embedding tables.
3. Build a local distributed test harness that supports multiple PS servers.
4. Port both tests: full restore and partial restore with PS monitor.

**Tests (Detailed)**
- Python tests: `PartialRestoreTest` in this file.
- Rust tests: add integration tests for checkpoint restore paths.
- Cross-language parity test: compare restored values across identical checkpoints.

**Gaps / Notes**
- Heavy reliance on TF distributed runtime and custom hash table ops.
- Partial restore logic depends on PS monitor and server set differences.

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

### `monolith/native_training/runner_utils.py`
<a id="monolith-native-training-runner-utils-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 396
- Purpose/role: Runner configuration, checkpoint override logic, and service discovery helpers for distributed training.
- Key symbols/classes/functions: `isabs`, `gen_get_checkpoint_state`, `ContainerType`, `RunnerConfig`, `get_discovery`, `monolith_discovery`.
- External dependencies: TensorFlow checkpoint APIs, `gflags_utils`, service discovery classes, `save_utils`, `mlp_utils`, protobuf `text_format`.
- Side effects:
  - Monkey-patches `os.path.isabs`, `checkpoint_management.get_checkpoint_state`, and `tf.train.get_checkpoint_state`.
  - Writes checkpoint and monolith checkpoint metadata files to `model_dir`.
  - Updates global `FLAGS` for kafka settings and restore checkpoints.

**Required Behavior (Detailed)**
- **`isabs(path)`**:
  - Returns True for `hdfs:/` prefix; otherwise uses original `os.path.isabs`.
  - Assigned to `os.path.isabs` at import time.
- **`gen_get_checkpoint_state()`**:
  - Wraps original `checkpoint_management.get_checkpoint_state` with retry and restore override logic.
  - Retries up to 5 times when checkpoint file exists but state is `None`; raises `Exception("read ckpt error!")` if exceeded.
  - If `FLAGS.restore_ckpt` set and `latest_filename == "checkpoint"`:
    - Builds `restore_ckpt` path using checkpoint state directory + basename of `FLAGS.restore_ckpt`.
    - If `restore_ckpt` exists in `all_model_checkpoint_paths`:
      - TRAIN: uses `restore_ckpt` only if `restore_ckpt` marker file does not exist.
      - Non-TRAIN: always uses `restore_ckpt`.
    - Writes `restore_ckpt` marker and updates checkpoint file when applying restore.
  - Logs warnings for missing/identical restore checkpoints.
  - Catches `UnparsedFlagAccessError` and logs other exceptions with traceback.
  - Assigned to `checkpoint_management.get_checkpoint_state` and `tf.train.get_checkpoint_state`.
- **`ContainerType`** enum: `DOCKER=1`, `NATIVE=2`.
- **`RunnerConfig(DistributedCpuTrainingConfig)`**:
  - Large dataclass with training/runtime flags (see source for full list).
  - `__post_init__`:
    - Calls `mlp_pass()` and `add_mpi_exception_hook()`.
    - Updates via `gflags_utils.update(self)` (logs on failure).
    - For GPU partial sync training: sets `index` from `OMPI_COMM_WORLD_RANK` when worker index unset.
    - Propagates kafka settings into global `FLAGS`.
    - Asserts `zk_watch_address_family` in {IPV4, IPV6}.
    - Sets `FLAGS.restore_ckpt` if unset.
    - If `restore_dir` set:
      - Chief (local or worker 0) runs `_copy_ckpt_file`.
      - Others wait for `monolith_checkpoint` file to appear (poll every 30s).
  - **`_copy_ckpt_file()`**:
    - Reads `checkpoint` from `restore_dir`.
    - Writes `checkpoint` file into `model_dir` if not present.
    - Writes `restore_ckpt` marker file.
    - Updates/creates `monolith_checkpoint` with restore paths added to `exempt_model_checkpoint_paths`.
    - Logs warnings when restore checkpoint missing or invalid.
- **`get_discovery(runner_conf, psm=None)`**:
  - Local → `None`.
  - PRIMUS → `TfConfigServiceDiscovery` from `tf_config` JSON; sets `server_type` and `index`.
  - CONSUL → `ConsulServiceDiscovery(psm)`.
  - MLP → `MLPServiceDiscovery()`.
  - Else → `ZKServiceDiscovery(deep_insight_name, zk_server)`.
- **`monolith_discovery(runner_conf)`**:
  - Context manager; for non-local creates `psm` via `env_utils.generate_psm_from_uuid(runner_conf.uuid)` and yields discovery.
  - Ensures `discovery.close()` on exit; logs enter/exit.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src`.
- Rust public API surface:
  - `RunnerConfig` struct with post-init behavior.
  - Checkpoint state wrapper for restore override logic.
  - Service discovery factory and context guard.
- Data model mapping:
  - CheckpointState + MonolithCheckpointState handling.
  - HDFS path handling in `isabs`.
- Feature gating:
  - TF checkpoint patching only under TF runtime backend.
  - Service discovery requires Consul/ZK/MLP client support.
- Integration points:
  - Training entrypoints, service discovery, and checkpoint restore flows.

**Implementation Steps (Detailed)**
1. Implement path `isabs` override for `hdfs:/`.
2. Implement checkpoint state override logic and restore marker handling.
3. Port `RunnerConfig` fields and `__post_init__` behavior.
4. Implement `_copy_ckpt_file` logic for checkpoint + monolith checkpoint.
5. Add service discovery factory and context manager.
6. Add unit tests for discovery selection and checkpoint copy behavior.

**Tests (Detailed)**
- Python tests: `monolith/native_training/runner_utils_test.py`.
- Rust tests: add unit tests for `get_discovery` and `_copy_ckpt_file` logic.
- Cross-language parity test: compare checkpoint files and discovery types.

**Gaps / Notes**
- Monkey-patching `tf.train.get_checkpoint_state` is process-global; Rust must replicate or document differences.

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

### `monolith/native_training/runner_utils_test.py`
<a id="monolith-native-training-runner-utils-test-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 108
- Purpose/role: Tests service discovery selection and checkpoint copy logic in `RunnerConfig`.
- Key symbols/classes/functions: `RunnerUtilsTest`.
- External dependencies: TensorFlow, `CheckpointState`, `KazooTimeoutError`, `RunnerConfig`, `get_discovery`.
- Side effects: Writes checkpoint files under `TEST_TMPDIR`.

**Required Behavior (Detailed)**
- `test_get_discovery_local`:
  - `RunnerConfig(is_local=True)` → `get_discovery` returns `None`.
  - After toggling `is_local=False`, still uses previous return (no assertion change).
- `test_get_discovery_primus`:
  - Builds `tf_config` JSON with ps/worker/chief.
  - Expects `get_discovery` returns `TfConfigServiceDiscovery`.
- `test_get_discovery_consul`:
  - Expects `ConsulServiceDiscovery` when `discovery_type=CONSUL` and `psm` provided.
- `test_get_discovery_zk`:
  - Attempts ZK discovery; catches `KazooTimeoutError`.
- `test_copy_ckpt`:
  - Creates `restore_dir` with a `checkpoint` file containing three checkpoints.
  - Instantiates `RunnerConfig` with `restore_dir`, `model_dir`, `restore_ckpt='model.ckpt-30'`.
  - Asserts:
    - `monolith_checkpoint` file exists in `model_dir`.
    - `restore_ckpt` marker file exists.
    - `tf.train.get_checkpoint_state(model_dir)` returns `model.ckpt-30`.
  - Instantiates a worker `RunnerConfig` (index=2) to verify non-chief path.
- `__main__`: disables eager execution and runs tests.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src` (tests).
- Rust public API surface: `RunnerConfig`, `get_discovery`.
- Data model mapping: checkpoint file format and monolith checkpoint metadata.
- Feature gating: ZK/Consul discovery tests conditional on clients.
- Integration points: runner initialization and restore logic.

**Implementation Steps (Detailed)**
1. Add tests for discovery selection by `discovery_type`.
2. Add checkpoint copy test: ensure checkpoint and monolith_checkpoint files are written.
3. Add test for override `restore_ckpt` selection.

**Tests (Detailed)**
- Python tests: `RunnerUtilsTest` in this file.
- Rust tests: add discovery tests and checkpoint copy test.
- Cross-language parity test: compare checkpoint file contents and selected discovery type.

**Gaps / Notes**
- `test_get_discovery_local` mutates `is_local` after calling `get_discovery` but does not re-run discovery.

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

### `monolith/native_training/runtime/ops/gen_monolith_ops.py`
<a id="monolith-native-training-runtime-ops-gen-monolith-ops-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 23
- Purpose/role: Loads the Monolith custom op shared library and exposes generated op wrappers.
- Key symbols/classes/functions: `gen_monolith_ops_base` imports, `tf.load_library(...)`.
- External dependencies: TensorFlow, `monolith.utils.get_libops_path`.
- Side effects: Loads shared library `libtfkernel_monolith_ops_for_load.so` at import time.

**Required Behavior (Detailed)**
- Imports all symbols from `gen_monolith_ops_base`.
- Calls `tf.load_library(utils.get_libops_path("monolith/native_training/runtime/ops/libtfkernel_monolith_ops_for_load.so"))` on import.
- This is required for custom ops used throughout native training (hash tables, optimizers, ragged utils, etc.).

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-tf/src` (TF runtime bindings).
- Rust public API surface: dynamic library loader or FFI bindings for custom ops.
- Data model mapping: expose wrappers or direct FFI calls matching generated ops.
- Feature gating: only available when TF runtime backend is enabled and custom ops library present.
- Integration points: required by optimizers, hash table ops, ragged utils, etc.

**Implementation Steps (Detailed)**
1. Provide a Rust wrapper that loads the custom op shared library at startup.
2. Ensure load is idempotent and errors are surfaced clearly.
3. Map or bind the generated op APIs used in Python.

**Tests (Detailed)**
- Python tests: none in this file; exercised by downstream tests.
- Rust tests: add a smoke test that loads the shared library when TF backend enabled.
- Cross-language parity test: verify ops are available and callable.

**Gaps / Notes**
- Import-time side effect means load failures are fatal early; Rust should handle similarly or fail fast.

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

### `monolith/native_training/save_utils.py`
<a id="monolith-native-training-save-utils-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 1309
- Purpose/role: Checkpoint save/restore utilities with partial recovery, PS monitoring, monolith checkpoint metadata, and tide-aware saving.
- Key symbols/classes/functions: `get_latest_checkpoint_state`, `get_monolith_checkpoint_state`, `SaveHelper`, `SecondOrStepTimerWithTideSetting`, `NoFirstSaveCheckpointSaverHook`, `PsMonitor`, `SaverBuilder`, `PartialRecoverySaver`.
- External dependencies: TensorFlow checkpoint APIs, `MonolithCheckpointState`, `cli` metrics, `tide_available_now`, `CUSTOM_RESTORE_OP`, `calc_feed_dict`.
- Side effects:
  - Writes `monolith_checkpoint` file.
  - Emits metrics via `cli`.
  - Uses custom restore hooks and may write `clear_nn` flag file.

**Required Behavior (Detailed)**
- **`get_latest_checkpoint_state(checkpoint_dir, global_step_value)`**
  - Caches checkpoint state per directory in `_ckpt_state_cache_map`.
  - Refreshes cache when `global_step_value` increases or cached state is None.
- **`get_monolith_checkpoint_state(checkpoint_dir, filename=None, remove_invalid_path=False)`**
  - Reads `monolith_checkpoint` (or `filename`) and parses `MonolithCheckpointState`.
  - If `remove_invalid_path`:
    - Converts relative paths to absolute by prepending `checkpoint_dir`.
    - Removes paths that do not exist on filesystem.
  - Returns None on `OpError` or `ParseError` with warnings.
- **`SaveHelper`**
  - `get_ckpt_prefix(basename, step)` → `"basename-step"`.
  - `get_ckpt_asset_dir(ckpt_prefix)` → `"ckpt_prefix.assets/"`.
  - `get_global_step_value(ckpt_prefix)` → parse suffix after `-`.
  - `get_existing_checkpoint_steps()` uses `tf.train.get_checkpoint_state` to list steps.
- **`SecondOrStepTimerWithTideSetting`**
  - Extends `SecondOrStepTimer` with tide availability checks.
  - If tide unavailable (via `tide_available_now`), uses `_tide_every_secs` instead of `every_secs`.
  - `enable()/disable()` toggles trigger behavior.
- **`NoFirstSaveCheckpointSaverHook`**
  - Wraps `CheckpointSaverHook` with:
    - Optional tide settings (`tide_*`).
    - `no_first_save` (skip initial save).
    - `_is_dense_only`, `_use_native_multi_hash_table`, `_guard_saver_listeners`.
  - `after_create_session`:
    - Exports meta graph if requested.
    - For `PartialRecoverySaver`, calls `setup_ps_initialized_state`.
    - Creates monolith checkpoint state file if needed.
  - `trigger_save(session, ignore_save_errors=False)`:
    - Resets timer and invokes save logic under lock.
  - `_save(session, step)`:
    - Skips first save when `no_first_save`.
    - Skips dense-only save if same step as last.
    - Calls guard listeners even on save failures.
    - Retries save on `OpError` up to 2 times; optionally ignores errors.
    - Emits `save_checkpoint` metrics and timing.
  - `_create_or_update_monolith_ckpt_state(do_update)`:
    - Writes `monolith_checkpoint` with hash table type and timestamp.
  - `end(session)`:
    - Forces save if dense-only or model dump mode flags set.
- **`PsMonitor`**
  - Creates per-PS FIFOQueue to detect initialization.
  - `is_ps_uninitialized(sess, device)` checks queue size.
  - `setup_ps_initialized_state(sess)` enqueues to mark initialized.
- **`SaverBuilder(BulkSaverBuilder)`**
  - Overrides `_AddShardedRestoreOps` to store grouped restore ops per device.
  - `restore_ops_per_device` property returns grouped ops.
- **`PartialRecoverySaver`**
  - Based on TF `Saver` with modifications:
    - Supports `ps_monitor`, `exempt_checkpoint_paths`, `skip_save`, `model_dir`.
    - `exempt_checkpoint_paths` property reads `monolith_checkpoint` to avoid deleting exempt ckpts.
    - `_RecordLastCheckpoint` excludes exempt paths from deletion accounting.
    - `_MaybeDeleteOldCheckpoints` respects `keep_checkpoint_every_n_hours`.
    - `save(...)` supports `save_relative_paths`, handles parent dir errors, writes meta graph optionally.
    - `_origin_restore` restores by device; if `ps_monitor` set, skips devices already initialized.
    - `restore(...)`:
      - Falls back to `_origin_restore` during export or when no graph.
      - Supports `CUSTOM_RESTORE_OP` collection:
        - If alias map present, uses `calc_feed_dict` and `NewCheckpointReader`.
        - If init ops present, handles `clear_nn` flag and optionally updates global_step.
        - Removes `CUSTOM_RESTORE_OP` from graph collections.
    - `setup_ps_initialized_state(sess)` delegates to `ps_monitor`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-checkpoint/src`.
- Rust public API surface:
  - `PartialRecoverySaver`, `NoFirstSaveCheckpointSaverHook`, `SecondOrStepTimerWithTideSetting`, `SaveHelper`.
  - Monolith checkpoint state parser/writer.
- Data model mapping:
  - `MonolithCheckpointState` protobuf and checkpoint metadata.
  - Device-specific restore ops and partial recovery logic.
- Feature gating:
  - TF runtime backend required for Saver-based logic.
  - Tide-based timing requires time availability checks.
- Integration points:
  - Used by restore hooks, PS monitor, and runner utilities.

**Implementation Steps (Detailed)**
1. Implement checkpoint state cache and monolith checkpoint state parsing.
2. Port SaveHelper and tide-aware timer.
3. Implement NoFirstSaveCheckpointSaverHook with retry and metrics.
4. Implement PsMonitor queue-based initialization state.
5. Implement PartialRecoverySaver with exempt checkpoint handling and custom restore logic.
6. Add tests to mirror `save_utils_test.py` coverage.

**Tests (Detailed)**
- Python tests: `monolith/native_training/save_utils_test.py`.
- Rust tests: extensive saver/restore tests, max_to_keep, tide timer, and partial restore.
- Cross-language parity test: save in TF, restore in Rust (TF backend), compare values.

**Gaps / Notes**
- Custom restore logic depends on `CUSTOM_RESTORE_OP` collection and `clear_nn` flag file semantics.

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

### `monolith/native_training/save_utils_test.py`
<a id="monolith-native-training-save-utils-test-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 1740
- Purpose/role: Comprehensive tests for `PartialRecoverySaver`, NoFirstSaveCheckpointSaverHook, checkpoint retention, and tide-aware timer.
- Key symbols/classes/functions: `SaveUtilsTest`, `SaverHookTest`, `SaverTest`, `SaveRestoreShardedTest`, `MaxToKeepTest`, `RecoverLastCheckpointsTest`, `KeepCheckpointEveryNHoursTest`, `SaveRestoreWithVariableNameMap`, `SecondOrStepTimerWithTideSettingTest`.
- External dependencies: TensorFlow, `freezegun`, `mock`, `saver_test_utils`, `resource_variable_ops`, `checkpoint_management`.
- Side effects: Writes checkpoint files under `TEST_TMPDIR`, uses eager/graph modes.

**Required Behavior (Detailed)**
- **SaveUtilsTest**
  - `test_get_ckpt_steps`: `SaveHelper.get_existing_checkpoint_steps` returns steps {10,20,300}.
  - `test_exempt_checkpoints`: validates `max_to_keep` with exempt checkpoints.
- **SaverHookTest**
  - `test_basic`: NoFirstSaveCheckpointSaverHook skips initial save, saves on session close.
  - `test_op_error`: guard listener `after_save` is called even when save fails.
  - `test_trigger_save`: `trigger_save` forces save within session.
- **SaverTest**
  - `basicSaveRestore`: save/restore of variables and `CheckpointedOp`.
  - `testSaveMaxToKeep`: max_to_keep with exempt path.
  - `testResourceColocation`: SaveV2 inputs colocated on CPU of same device.
  - `testResourceVariableReadOpsAddedDeterministically`: graph defs deterministic.
  - `testEagerBasic` and `testEagerGraphCompatibility`: eager/graph save/restore compatibility.
  - `testResourceSaveRestoreCachingDevice`: caching_device variable save/restore.
  - `testNoAdditionalOpsAddedBySaverForResourceVariablesOutsideSaveScope`: no extraneous ops.
  - `testSaveCopyRestoreWithSaveRelativePaths`: relative paths survive directory move.
  - `testFilenameTensor`, `testInvalidPath`, `testInt64`, `testSomeErrors`, `testSameName`.
  - `testBasicsWithListOfVariables`, `_SaveAndLoad` helper, `testCacheRereadsFile`.
  - `testAllowEmpty`, `testGPU`, `testSharedServerOnGPU`, `testVariables`.
  - `testVarListShouldBeEmptyInDeferredBuild`, `testBuildShouldBeCalledBeforeSaveInCaseOfDeferBuild`, `testDeferredBuild`.
  - `testReshape`: reshape restore behavior.
  - `testSaveWithGlobalStep`, `testSaveWithGlobalStepWithPadding`.
  - `testSaveToNonexistingPath`, `testSaveToURI`.
  - `testSaveRestoreAndValidateVariableDtype`, `testRestoreLargeTensors`.
- **SaveRestoreShardedTest**
  - `testIterators`: sharded save/restore of dataset iterators across devices.
  - `testIteratorsUnshardedRestore`: restore from sharded checkpoint when saver is unsharded.
- **MaxToKeepTest**
  - `testMaxToKeepEager`: checkpoint rotation in eager mode.
  - `testNonSharded`, `testSharded`, `testNoMaxToKeep`, `testNoMetaGraph`.
- **RecoverLastCheckpointsTest**
  - `test_recover_last_checkpoints`: recover last checkpoints, handle missing files.
- **KeepCheckpointEveryNHoursTest**
  - `testNonSharded`: uses mocked time to validate keep-every-N-hours behavior.
- **SaveRestoreWithVariableNameMap**
  - `testNonReshapeResourceVariable` / `testNonReshapeVariable`: name mapping restore.
- **SecondOrStepTimerWithTideSettingTest**
  - `testNoTideSetting`, `testTideAvailable`, `testTideNotAvailable` using `freezegun`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-checkpoint/src` (tests).
- Rust public API surface: saver, hook, timer, checkpoint retention.
- Data model mapping: checkpoint files + metadata; iterator saveables if supported.
- Feature gating: eager/graph tests depend on backend; GPU tests conditional.
- Integration points: checkpoint manager and saver semantics.

**Implementation Steps (Detailed)**
1. Port core saver tests: basic save/restore, max_to_keep, exempt checkpoints.
2. Add hook tests for NoFirstSaveCheckpointSaverHook and trigger_save.
3. Add timer tests for tide behavior (mock time).
4. Add sharded save/restore tests if supported.
5. Add deferred build and reshape tests.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: map major behaviors; use deterministic fixtures.
- Cross-language parity test: save in TF, restore in Rust/TF backend, compare values.

**Gaps / Notes**
- Some tests rely on TF-specific Saveable objects (`CheckpointedOp`, iterators).

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

### `monolith/native_training/service_discovery.py`
<a id="monolith-native-training-service-discovery-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 481
- Purpose/role: Service discovery abstraction with Consul, TF_CONFIG, Zookeeper, and MLP implementations.
- Key symbols/classes/functions: `ServiceDiscoveryType`, `ServiceDiscovery`, `ConsulServiceDiscovery`, `TfConfigServiceDiscovery`, `ZKServiceDiscovery`, `MLPServiceDiscovery`, `deregister_all`.
- External dependencies: Consul client, Kazoo (ZK), `MonolithKazooClient`, `MLPEnv`.
- Side effects: Registers/deregisters nodes in external systems; spawns periodic registration threads for ZK.

**Required Behavior (Detailed)**
- **`ServiceDiscoveryType`**: `PRIMUS`, `CONSUL`, `ZK`, `MLP`.
- **`ServiceDiscovery`** abstract methods: `register`, `deregister`, `query`; optional `close`.
- **`retry_with_socket_error`**:
  - Retries callable up to 5 times on `socket.error` with randomized backoff (≤ `_RETRY_MAX_BACKOFF_SECS`).
- **`ConsulServiceDiscovery(consul_id, retry_time_secs=3.0)`**
  - `register`: best-effort de-register existing index, register via tags `{index,name,ip}`, then poll until visible (or raise `OSError` after ~180s).
  - `deregister`: removes entry by port.
  - `query_all`: returns `{name: {index: "ip:port"}}` from Consul lookup.
  - `query(name)`: returns mapping for that name.
- **`TfConfigServiceDiscovery(tf_config)`**
  - `register`/`deregister`: no-ops.
  - `query('ps')` uses `cluster['ps']`.
  - `query('worker')` prepends `cluster['chief']` if present.
  - `server_type`: `'worker'` if task type is `'chief'`, else task type.
  - `addr`: address for current task.
  - `index`: if chief exists and task is worker, index+1.
- **`ZKServiceDiscovery(job_name, zk_server=None, max_tries=3, delay=5)`**
  - Uses `MonolithKazooClient`, `ZKListener`, and Children/Data watches to maintain `_cluster`.
  - Registers nodes under `/monolith/{job_name}/{server_type}.{index}` as ephemeral.
  - Periodically re-registers every `_ZK_REGISTRATION_PERIOD` using background threads.
  - `query(name)` returns mapping for that name.
  - `close()` stops client and threads.
- **`MLPServiceDiscovery`**
  - Validates registration against `MLPEnv` expected roles/ports.
  - Maintains `_filters` to hide deregistered entries.
  - `query(name, skip_port_check=False)` returns address map; for PS checks port connectivity unless skipped.
  - `deregister_all` filters all roles.
  - `query_all` returns maps for `ps/worker/chief`.
  - `server_type` and `index` derived from `MLPEnv`.
- **`deregister_all(consul_id)`**: deregisters all entries in Consul for id.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src`.
- Rust public API surface: discovery trait + Consul/ZK/TF_CONFIG/MLP implementations.
- Data model mapping: `name -> index -> addr`.
- Feature gating: Consul/ZK/MLP clients behind optional features.
- Integration points: runner utilities and distributed training orchestration.

**Implementation Steps (Detailed)**
1. Define discovery trait and type enum.
2. Implement TF_CONFIG discovery (pure JSON).
3. Implement Consul discovery with retry + blacklist behavior.
4. Implement ZK discovery with watchers and periodic registration threads.
5. Implement MLP discovery with filter logic and port checks.
6. Add `deregister_all`.

**Tests (Detailed)**
- Python tests: `monolith/native_training/service_discovery_test.py`.
- Rust tests: mock clients and validate mappings, duplicate registration, retry behavior.
- Cross-language parity test: compare mapping outputs.

**Gaps / Notes**
- ZK implementation is stateful and thread-based; Rust must emulate or adapt.

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

### `monolith/native_training/service_discovery_test.py`
<a id="monolith-native-training-service-discovery-test-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 407
- Purpose/role: Unit tests for Consul, TF_CONFIG, ZK service discovery and deregistration helpers.
- Key symbols/classes/functions: `FakeConsul`, `FakeKazooClient`, `ConsultServiceDiscovery`, `TfConfigServiceDiscoveryTest`, `ZKServiceDiscoveryTest`, `UtilsTest`.
- External dependencies: `unittest`, `mock`, `kazoo` exceptions, `service_discovery`.
- Side effects: Uses mocked clients; no real network.

**Required Behavior (Detailed)**
- **Consul tests**
  - `test_basic`: register two entries, query returns mapping, deregister removes all.
  - `test_duplicate_registration`: re-register same index replaces addr.
  - `test_multi_names`: independent maps for `ps` and `worker`.
  - `test_retry`: mocked socket timeout triggers retry and eventually raises.
  - `test_registration_failed`: blacklist causes `OSError`.
- **TF_CONFIG tests**
  - `test_tf_conf_sd`: verifies ps and worker lists (chief prepended), addr, server_type, and index behavior.
- **ZK tests**
  - `test_basic`: register, query, deregister works.
  - `test_duplicate_registration`: re-register same index updates addr.
  - `test_multi_names`: multi service names.
  - `test_periodic_registration`: with shortened period, corrupted data is repaired.
  - `test_listener`: LOST then CONNECTED triggers re-registration.
- **Utils test**
  - `test_deregister_all`: deregister_all removes all entries.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src` (tests).
- Rust public API surface: discovery implementations and deregister_all.
- Data model mapping: in-memory mocks for Consul/ZK.
- Feature gating: optional client dependencies.
- Integration points: service discovery unit tests.

**Implementation Steps (Detailed)**
1. Add mock Consul client with register/lookup/deregister.
2. Add mock ZK client with watches and ephemeral nodes.
3. Port tests for duplicate registration, retry, periodic registration, and listener behavior.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: replicate with mocks.
- Cross-language parity test: compare mapping outputs and failure behaviors.

**Gaps / Notes**
- No direct tests for `MLPServiceDiscovery` in Python.

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

### `monolith/native_training/serving_ps_test.py`
<a id="monolith-native-training-serving-ps-test-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 231
- Purpose/role: Generates example batches and feature configs for serving PS tests.
- Key symbols/classes/functions: `FeatMeta`, `TableMeta`, `ServingPSTest`.
- External dependencies: `distribution_ops` (imported), `idl.matrix.proto.example_pb2` protos.
- Side effects: Prints generated `ExampleBatch` and `FeatureConfigs` to stdout.

**Required Behavior (Detailed)**
- Defines feature/table metadata (`FeatMeta`, `TableMeta`) and a `features` map.
- `test_example_gen`:
  - Builds `ExampleBatch` with `batch_size=10`.
  - For each feature, fills `fid_v1_list` or `fid_v2_list` based on `fid_version`.
  - For SHARED feature lists, only adds first feature (breaks after one).
  - Prints the batch.
- `test_conf_gen`:
  - Builds `FeatureConfigs` with per-feature slice dims and pooling types.
  - Constructs `OutConfig` objects:
    - `bias` and `vec` as `CONCAT`.
    - `uffm`, `iffm`, `seq` as `NONE`.
    - `user_only` as `STACK`.
  - Configures slice ranges and shapes based on metadata.
  - Prints configs.
- `__main__`: disables eager execution and runs `tf.test.main()`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src` (tests or utilities).
- Rust public API surface: helper to generate example batches and feature configs.
- Data model mapping: ExampleBatch, FeatureConfigs, OutConfig protos.
- Feature gating: requires proto definitions and encoding.
- Integration points: serving PS pipelines or export utilities.

**Implementation Steps (Detailed)**
1. Port metadata structs and feature definitions.
2. Implement example batch generator with fid_v1/v2 bit packing.
3. Implement feature config generation and OutConfig creation.
4. Add tests to validate shapes and slice configs.

**Tests (Detailed)**
- Python tests: `ServingPSTest` in this file (no assertions besides prints).
- Rust tests: add assertions for generated proto contents.
- Cross-language parity test: compare serialized protos.

**Gaps / Notes**
- Tests mainly print outputs; no asserts in Python.

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

### `monolith/native_training/session_run_hooks.py`
<a id="monolith-native-training-session-run-hooks-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 171
- Purpose/role: SessionRunHooks for tide-aware stopping and delayed worker start based on global step.
- Key symbols/classes/functions: `before`, `tide_available_now`, `CustomGlobalStepWaiterHook`, `TideStoppingHook`.
- External dependencies: TensorFlow session hooks, `training_util`, `datetime`, `random`.
- Side effects: Sleeps and may request session stop.

**Required Behavior (Detailed)**
- **`before(hour1, minute1, hour2, minute2)`**:
  - Returns True if time1 < time2 (lexicographic hour/minute).
- **`tide_available_now(start_h, start_m, end_h, end_m)`**:
  - Determines if current UTC time is within tide window; handles wrap-around (start > end).
- **`CustomGlobalStepWaiterHook(wait_until_step, tide_*, max_non_tide_wait_minute=10)`**
  - `begin`: creates global step tensor; raises if missing.
  - `before_run`:
    - If already started or wait_until_step <= 0, returns immediately.
    - If tide window configured and tide not available, logs and requests stop.
    - Polls global step until >= wait_until_step; sets `_worker_is_started`.
    - Also starts a timer once global_step > 1; if wait exceeds `max_non_tide_wait_minute` (+ random 0–600s), starts anyway.
    - Sleeps 0.5s between checks and logs periodically.
- **`TideStoppingHook(tide_*)`**
  - `before_run`: if tide not available, logs and requests stop.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src`.
- Rust public API surface: session hooks for wait-until-step and tide stopping.
- Data model mapping: global step access and session stop request.
- Feature gating: none.
- Integration points: training runner hooks.

**Implementation Steps (Detailed)**
1. Implement time window logic (`tide_available_now`) with UTC time.
2. Implement wait hook with global step polling and timeout behavior.
3. Implement tide stopping hook for graceful shutdown.
4. Add tests using time freezing/mocking.

**Tests (Detailed)**
- Python tests: `monolith/native_training/session_run_hooks_test.py`.
- Rust tests: add tests for tide availability and wait logic.
- Cross-language parity test: compare tide window evaluations.

**Gaps / Notes**
- `CustomGlobalStepWaiterHook` uses random extra wait time (0–600s) in timeout calculation.

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

### `monolith/native_training/session_run_hooks_test.py`
<a id="monolith-native-training-session-run-hooks-test-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 144
- Purpose/role: Tests for tide availability logic and global step waiter hook.
- Key symbols/classes/functions: `GlobalStepWaiterHookTest`, `TideStoppingHookTest`.
- External dependencies: TensorFlow, `freezegun`, `time`, `session_run_hooks`.
- Side effects: Uses frozen time and mocked `time.sleep`.

**Required Behavior (Detailed)**
- `GlobalStepWaiterHookTest`:
  - `test_not_wait_for_step_zero`: `wait_until_step=0` returns immediately.
  - `test_not_wait_if_tide_not_available`: with tide window outside current time, hook returns without waiting.
  - `test_wait_for_step`: mocks `time.sleep` to advance global_step; expects hook to loop twice.
- `TideStoppingHookTest`:
  - `test_stop_if_tide_not_available`: when tide not available, `request_stop` is called.
  - `test_do_not_stop_if_tide_available`: when tide available, no stop requested.
- `__main__`: disables eager execution and runs tests.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src` (tests).
- Rust public API surface: wait hook and tide stopping hook.
- Data model mapping: global step mock and session context.
- Feature gating: none.
- Integration points: training runner.

**Implementation Steps (Detailed)**
1. Add tests for immediate return when wait_until_step=0.
2. Mock time and global step updates to test waiting loop.
3. Add tests for tide stopping behavior.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: mirror with mocked time and global step.
- Cross-language parity test: compare tide-window logic outputs.

**Gaps / Notes**
- Uses freezegun; Rust tests should use deterministic time control.

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

### `monolith/native_training/signal_utils.py`
<a id="monolith-native-training-signal-utils-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 37
- Purpose/role: Installs a SIGUSR1 handler that prints stack traces.
- Key symbols/classes/functions: `print_stack_trace`, `add_siguser1_handler`.
- External dependencies: `signal`, `traceback`.
- Side effects: Registers a SIGUSR1 handler at import time.

**Required Behavior (Detailed)**
- `print_stack_trace(sig, frame)`:
  - Prints each line from `traceback.format_stack(frame)`.
- `add_siguser1_handler()`:
  - Captures current SIGUSR1 handler (`signal.getsignal`).
  - Registers new handler that calls previous handler (if callable) then prints stack trace.
- Module import calls `add_siguser1_handler()` immediately.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/src`.
- Rust public API surface: signal handler registration and stack trace printing.
- Data model mapping: signal handling and backtrace capture.
- Feature gating: may be platform-specific (POSIX only).
- Integration points: runtime diagnostics.

**Implementation Steps (Detailed)**
1. Register SIGUSR1 handler on module init.
2. Chain previous handler if present.
3. Print backtrace to stdout/stderr.

**Tests (Detailed)**
- Python tests: `monolith/native_training/signal_utils_test.py`.
- Rust tests: add signal handler test (if supported).
- Cross-language parity test: verify handler chaining and output.

**Gaps / Notes**
- Signal handling is OS-specific; Rust may need conditional compilation.

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

### `monolith/native_training/signal_utils_test.py`
<a id="monolith-native-training-signal-utils-test-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 30
- Purpose/role: Verifies SIGUSR1 handler registration and chaining.
- Key symbols/classes/functions: `SignalUtilsTest`.
- External dependencies: `signal`, `signal_utils`.
- Side effects: Raises SIGUSR1 signal.

**Required Behavior (Detailed)**
- `testBasic`:
  - Calls `add_siguser1_handler()` twice to ensure chaining works.
  - Raises SIGUSR1 signal via `signal.raise_signal`.
  - Test passes if no exception is raised.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/src` (tests).
- Rust public API surface: signal handler registration.
- Data model mapping: SIGUSR1 handling.
- Feature gating: OS-specific support.
- Integration points: runtime diagnostics.

**Implementation Steps (Detailed)**
1. Add test that registers handler twice and raises SIGUSR1.
2. Ensure no panic/crash.

**Tests (Detailed)**
- Python tests: `SignalUtilsTest` in this file.
- Rust tests: add signal handler test if supported.
- Cross-language parity test: ensure handler chaining does not error.

**Gaps / Notes**
- Signal handling tests may be flaky on non-POSIX systems.

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

### `monolith/native_training/static_reshape_op.py`
<a id="monolith-native-training-static-reshape-op-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 58
- Purpose/role: Wrapper for custom static reshape op and a small builder utility.
- Key symbols/classes/functions: `static_reshape`, `StaticReshapeNBuilder`.
- External dependencies: `gen_monolith_ops.monolith_static_reshape_n`.
- Side effects: None beyond calling custom op.

**Required Behavior (Detailed)**
- **`static_reshape(inputs, shapes, enable_parallelism=True)`**
  - Calls `monolith_static_reshape_n` custom op.
  - Returns `(outputs, sizes)` where `sizes` are flattened output sizes.
- **`StaticReshapeNBuilder`**
  - Collects `inputs` and `shapes`.
  - `add(input, shape)` appends and returns index.
  - `build()` calls `static_reshape` with collected lists.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-tf/src`.
- Rust public API surface: `static_reshape` wrapper and builder.
- Data model mapping: list of tensors + shape tuples with optional `None`.
- Feature gating: requires custom op library.
- Integration points: preprocessing pipelines needing fast reshape.

**Implementation Steps (Detailed)**
1. Bind to custom op `monolith_static_reshape_n` or implement equivalent.
2. Implement builder helper with index mapping.
3. Add tests for sizes and shape constraints.

**Tests (Detailed)**
- Python tests: `monolith/native_training/static_reshape_op_test.py`.
- Rust tests: add tests for sizes and nested structure indexing.
- Cross-language parity test: compare output shapes and sizes.

**Gaps / Notes**
- Custom op semantics must match exactly; shapes include `None` placeholders.

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

### `monolith/native_training/static_reshape_op_test.py`
<a id="monolith-native-training-static-reshape-op-test-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 79
- Purpose/role: Tests static reshape custom op and builder.
- Key symbols/classes/functions: `StaticReshapeOpTest`.
- External dependencies: TensorFlow, NumPy, `static_reshape_op`.
- Side effects: Runs sessions to execute custom op.

**Required Behavior (Detailed)**
- `test_static_reshape_n`:
  - Reshapes three inputs to target shapes with `None` dimensions.
  - Asserts `sizes == [5, 40, 12]`.
  - Verifies each output shape matches specified non-`None` dims.
- `test_nested_reshape_n`:
  - Uses `StaticReshapeNBuilder` with nested structures.
  - Flattens tensors via `tf.nest.map_structure` and builds op.
  - Validates outputs match expected flattened arrays.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-tf/src` (tests).
- Rust public API surface: `static_reshape` and builder.
- Data model mapping: nested structure flattening.
- Feature gating: requires custom op.
- Integration points: custom op tests.

**Implementation Steps (Detailed)**
1. Add tests for sizes output and shape constraints.
2. Add nested structure test using builder indices.

**Tests (Detailed)**
- Python tests: `StaticReshapeOpTest` in this file.
- Rust tests: mirror both tests.
- Cross-language parity test: compare outputs for same inputs.

**Gaps / Notes**
- Relies on custom op `monolith_static_reshape_n`.

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

### `monolith/native_training/summary/summary_ops.py`
<a id="monolith-native-training-summary-summary-ops-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 78
- Purpose/role: Summary ops for NAS and feature insight data with custom metadata.
- Key symbols/classes/functions: `nas_data`, `feature_insight_data`.
- External dependencies: TensorFlow summary APIs, `summary.utils`, `feature_insight` op.
- Side effects: Adds tensor summaries to graph collections.

**Required Behavior (Detailed)**
- **`nas_data(weight, segment_names=None, segment_sizes=None, group_info=None, raw_tag=None, collections=None, description=None, name=None)`**
  - Uses `utils.prepare_head(..., out_type='bytes')` to build metadata.
  - Creates `tf.summary.tensor_summary` with tag `MONOLITH_NAS_DATA`.
  - Name defaults to `{summaty_type}` or `{name}_{summaty_type}`.
  - Description defaults to summary type.
- **`feature_insight_data(input_tensor, segment_names, segment_sizes, weight=None, group_info=None, label=None, collections=None, description=None, name=None)`**
  - If `weight` is provided, calls `feature_insight` op to aggregate, and adjusts `segment_sizes` to uniform dimension.
  - Determines `raw_tag`: direct vs train based on `label`.
  - If label provided:
    - Casts to float32, ensures rank 2, sets `label_size` in metadata.
    - Concatenates label to summary_data.
  - Writes `tf.summary.tensor_summary` with tag `MONOLITH_FI_DATA` and JSON metadata.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-tf/src`.
- Rust public API surface: summary helpers for NAS and feature insight.
- Data model mapping: summary metadata proto + tensor data.
- Feature gating: requires TensorBoard summary APIs; feature_insight op if used.
- Integration points: training summaries and analysis tooling.

**Implementation Steps (Detailed)**
1. Port `prepare_head` and metadata creation.
2. Implement summary writer helpers with matching tags and metadata.
3. Bind or reimplement `feature_insight` op if needed.
4. Add tests to validate metadata and tensor shapes.

**Tests (Detailed)**
- Python tests: `monolith/native_training/summary/summary_ops_test.py`.
- Rust tests: add metadata and shape validation tests.
- Cross-language parity test: compare serialized summary metadata.

**Gaps / Notes**
- Summary metadata uses JSON content; ensure exact formatting and ordering.

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

### `monolith/native_training/summary/summary_ops_test.py`
<a id="monolith-native-training-summary-summary-ops-test-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 122
- Purpose/role: Tests summary metadata and tensor outputs for NAS and feature insight summaries.
- Key symbols/classes/functions: `SummaryTest`.
- External dependencies: TensorBoard data provider APIs, `summary_ops`.
- Side effects: Writes summary logs to `demo_logs_v1` and reads them back.

**Required Behavior (Detailed)**
- `setUpClass`:
  - Builds summary ops (`nas_data`, `feature_insight_data`), writes summaries for 10 steps.
  - Initializes `EventMultiplexer` and `MultiplexerDataProvider`.
- `test_nas_data`:
  - Reads summary metadata and tensor values for tag `gating/monolith_nas_weight`.
  - Asserts plugin content JSON matches expected string and values equal weights.
- `test_feature_insight_data`:
  - Reads tag `fi_train/monolith_feature_insight`.
  - Asserts plugin content JSON (including `label_size`) matches expected.
  - Validates tensor shape `(3, 7)` when label_size=1.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-tf/src` (tests).
- Rust public API surface: summary ops and metadata.
- Data model mapping: TensorBoard summary metadata content.
- Feature gating: TensorBoard event parsing required.
- Integration points: summary writers and readers.

**Implementation Steps (Detailed)**
1. Add tests that emit summary data and read via event parsing.
2. Validate plugin metadata JSON and tensor shapes.

**Tests (Detailed)**
- Python tests: `SummaryTest` in this file.
- Rust tests: mirror event log parsing if supported.
- Cross-language parity test: compare plugin metadata content.

**Gaps / Notes**
- Uses TensorBoard data provider APIs; Rust may need alternative test approach.

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

### `monolith/native_training/summary/utils.py`
<a id="monolith-native-training-summary-utils-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 114
- Purpose/role: Summary metadata helpers and NAS weight extraction.
- Key symbols/classes/functions: `SummaryType`, `create_summary_metadata`, `prepare_head`, `get_nas_weight_json`.
- External dependencies: TensorFlow, TensorBoard summary protos.
- Side effects: None (pure helpers).

**Required Behavior (Detailed)**
- Constants:
  - `PLUGIN_NAME = 'monolith'`
  - `MONOLITH_NAS_DATA`, `MONOLITH_FI_DATA`, `KTYPE`, `KMETA`, `KDATA`.
- `SummaryType` values: `gating`, `selecting`, `mixed`, `simple`, `fi_direct`, `fi_train`.
- `create_summary_metadata(description=None, meta_content=b'')`:
  - Returns `SummaryMetadata` with plugin data and `DATA_CLASS_TENSOR`.
  - Accepts `meta_content` as bytes or str; encodes to UTF-8 if str.
- `_name_to_group_id(segment_names, group_info)`:
  - Builds mapping of segment names to group indices based on group_info.
  - Reorders group IDs by sorted group id.
- `prepare_head(segment_names, segment_sizes, group_info=None, raw_tag=None, out_type='tensor')`:
  - If no segments: returns empty tensor/bytes with raw_tag.
  - Determines `raw_tag`:
    - `gating` when all sizes are ints, else `selecting`.
  - Builds data dict with tag type, names, sizes, and optional `group_index`.
  - Returns tensor/JSON/bytes depending on out_type.
- `get_nas_weight_json(ckpt_dir_or_file, prefix=None)`:
  - Loads checkpoint, finds variable containing `prefix` (defaults to `ARCH_TENSOR_PREFIX`), returns list of values as strings.
  - Raises Exception if not found.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-tf/src`.
- Rust public API surface: metadata builder + NAS weight extraction.
- Data model mapping: Summary metadata proto and JSON content.
- Feature gating: checkpoint reader required for `get_nas_weight_json`.
- Integration points: summary ops.

**Implementation Steps (Detailed)**
1. Port constants and summary metadata creation.
2. Implement `prepare_head` with identical JSON ordering.
3. Implement checkpoint reader helper (or stub) for NAS weights.

**Tests (Detailed)**
- Python tests: `monolith/native_training/summary/utils_test.py`.
- Rust tests: add tests for `prepare_head` outputs.
- Cross-language parity test: compare JSON metadata strings.

**Gaps / Notes**
- `ARCH_TENSOR_PREFIX` is referenced but not defined in this file; locate source before porting.

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

### `monolith/native_training/summary/utils_test.py`
<a id="monolith-native-training-summary-utils-test-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 43
- Purpose/role: Tests `prepare_head` output for gating and selecting cases.
- Key symbols/classes/functions: `UtilsTest`.
- External dependencies: `summary.utils`.
- Side effects: None.

**Required Behavior (Detailed)**
- `test_read_head_gating`:
  - `segment_sizes` are ints → SummaryType.GATING.
  - Asserts returned tensor equals expected JSON bytes with `group_index`.
- `test_read_head_selecting`:
  - `segment_sizes` are lists → SummaryType.SELECTING.
  - Asserts returned tensor equals expected JSON bytes without `group_index`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-tf/src` (tests).
- Rust public API surface: `prepare_head` output.
- Data model mapping: JSON bytes in tensor.
- Feature gating: none.
- Integration points: summary metadata.

**Implementation Steps (Detailed)**
1. Add tests for gating/selecting outputs and expected JSON ordering.

**Tests (Detailed)**
- Python tests: `UtilsTest` in this file.
- Rust tests: mirror gating/selecting cases.
- Cross-language parity test: compare JSON bytes.

**Gaps / Notes**
- JSON ordering must match Python `json.dumps` output.

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

### `monolith/native_training/sync_hooks.py`
<a id="monolith-native-training-sync-hooks-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 176
- Purpose/role: Implements synchronization hooks between chief and workers.
- Key symbols/classes/functions: `SyncHelper`, `ChiefSyncHook`, `WorkerSyncHook`, `TrainingHooksHelper`.
- External dependencies: TensorFlow, `absl.logging`.
- Side effects: Uses TF variables to track worker status.

**Required Behavior (Detailed)**
- **`SyncHelper(num_workers, is_chief, var_device="/job:chief/task:0")`**
  - Creates `control_var` boolean vector length `num_workers`.
  - Index 0 = restore status; indices >0 = worker alive flags.
  - `mark_restore_done` sets index 0 True.
  - `start_worker`/`finish_worker` toggles worker index.
  - `get_alive_workers` returns indices with True.
  - `get_num_alive_workers` returns count of alive workers.
- **`ChiefSyncHook(sync_helper, timeout_seconds=1800)`**
  - `after_create_session`: marks restore done.
  - `end`: waits until no alive workers or timeout; logs remaining workers.
- **`WorkerSyncHook(worker_index, sync_helper)`**
  - `after_create_session`: marks worker alive and waits for restore status.
  - `end`: marks worker finished.
- **`TrainingHooksHelper(enable_sync, num_workers, worker_idx, chief_timeout_seconds)`**
  - If enabled, creates SyncHelper and attaches Chief/Worker hooks.
  - `training_hooks` and `training_chief_hooks` return tuples.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src`.
- Rust public API surface: sync helper and hooks.
- Data model mapping: shared status vector (e.g., in a distributed store).
- Feature gating: distributed training only.
- Integration points: training runner hooks.

**Implementation Steps (Detailed)**
1. Implement SyncHelper state tracking.
2. Implement Chief and Worker hooks with timeout.
3. Add helper to assemble hooks.
4. Add tests for synchronization flow.

**Tests (Detailed)**
- Python tests: `monolith/native_training/sync_hooks_test.py`.
- Rust tests: add multi-threaded hook tests.
- Cross-language parity test: compare wait/finish behavior.

**Gaps / Notes**
- `var_device` defaults to chief device; may differ in Rust backends.

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

### `monolith/native_training/sync_hooks_test.py`
<a id="monolith-native-training-sync-hooks-test-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 119
- Purpose/role: Tests sync helper and hook coordination.
- Key symbols/classes/functions: `SyncHooksTest`, `CountHook`.
- External dependencies: TensorFlow, threading.
- Side effects: Spawns threads.

**Required Behavior (Detailed)**
- `test_sync_process`:
  - Creates `SyncHelper`, Chief/Worker hooks, and count hooks.
  - Worker waits at after_create_session until chief marks restore done.
  - Chief waits at end until worker finishes; verifies counts.
- `test_hook_helper`:
  - Ensures TrainingHooksHelper returns empty tuples when disabled.
  - Creates enabled helper for grammar check.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src` (tests).
- Rust public API surface: sync helper and hooks.
- Data model mapping: thread synchronization.
- Feature gating: distributed training.
- Integration points: training runner.

**Implementation Steps (Detailed)**
1. Add multithreaded test for sync flow.
2. Add test for TrainingHooksHelper output.

**Tests (Detailed)**
- Python tests: `SyncHooksTest` in this file.
- Rust tests: mirror with threads.
- Cross-language parity test: compare hook sequencing.

**Gaps / Notes**
- Relies on timing; Rust tests should avoid flakiness.

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

### `monolith/native_training/sync_training_hooks.py`
<a id="monolith-native-training-sync-training-hooks-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 355
- Purpose/role: Hooks for synchronized training, parameter sync, forced dumps, and EOF-aware input wrapping.
- Key symbols/classes/functions: `SyncTrainingBarrierSaverListener`, `ParameterSyncHook`, `SyncTrainingForceDumpHook`, `SyncTrainingSaverControlHook`, `SyncTrainingInfoHook`, `ReqTimeControlDumpHook`, `EofAwareTask`.
- External dependencies: TensorFlow, Horovod (`hvd_lib`), distributed serving ops, hash table ops.
- Side effects: Broadcasts control flags, reads marker files, requests stop, modifies input pipeline.

**Required Behavior (Detailed)**
- **`SyncTrainingBarrierSaverListener`**
  - Uses `hvd_lib.broadcast` to sync after save.
- **`ParameterSyncHook(sync_backend, ps_index, refresh_interval=100)`**
  - Refreshes sync config periodically.
  - Calls `ParameterSyncClient.create_sync_op` and runs with config feed.
- **`SyncTrainingForceDumpHook(model_dir, target_timer, step_interval=100)`**
  - Every `step_interval`, rank 0 checks `dump_{step}` and `stop_{step}` files.
  - Broadcasts flags to all ranks; enables/disables target_timer based on UTC hour (18–20) and flags.
  - Requests stop if `should_stop`.
- **`SyncTrainingSaverControlHook(model_dir, target_timer, step_interval=100)`**
  - Toggles `target_timer` based on existence of `ONLINE` file.
- **`SyncTrainingInfoHook`**
  - Every 600s, logs hash table sizes collected from graph.
- **`ReqTimeControlDumpHook(model_dir, target_timer, step_interval=1000)`**
  - Rank 0 reads `req_time` collection and `limit_req_time` file.
  - Broadcasts values; requests stop if `req_time >= limit`.
- **`EofAwareTask(task, use_dataservice=False)`**
  - Wraps `NativeTask` to inject EOF flag into features and stop training across ranks when EOF is reached.
  - Adds `EofHook` to training hooks using `hvd_lib.allgather`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src`.
- Rust public API surface: training hooks for sync and EOF handling.
- Data model mapping: Horovod/allreduce equivalents if used.
- Feature gating: Horovod/distributed backend required for some hooks.
- Integration points: training runner and serving parameter sync.

**Implementation Steps (Detailed)**
1. Implement sync barrier and parameter sync hooks.
2. Implement dump/stop control via filesystem flags.
3. Implement EOF-aware task wrapper with distributed stop.
4. Add tests for EOF-aware task behavior.

**Tests (Detailed)**
- Python tests: `monolith/native_training/sync_training_hooks_test.py`.
- Rust tests: add EOF-aware task tests and hook smoke tests.
- Cross-language parity test: compare EOF stop behavior.

**Gaps / Notes**
- Several hooks depend on Horovod; Rust parity depends on distributed backend support.

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

### `monolith/native_training/sync_training_hooks_test.py`
<a id="monolith-native-training-sync-training-hooks-test-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 92
- Purpose/role: Tests EOF-aware task wrapper with simple datasets.
- Key symbols/classes/functions: `EofAwareTaskTest`.
- External dependencies: TensorFlow, `hvd_lib`, `sync_training_hooks`.
- Side effects: Initializes Horovod.

**Required Behavior (Detailed)**
- `test_basic`:
  - Defines simple NativeTask with dataset [1,2,3].
  - Wraps in `EofAwareTask` and trains estimator.
  - Expects `global_step == 6` (sum of 1+2+3).
- `test_dict`:
  - Similar but dataset yields dict `{"1": x}`.
  - Expects `global_step == 6`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src` (tests).
- Rust public API surface: EOF-aware task wrapper.
- Data model mapping: dataset + global_step.
- Feature gating: Horovod/distributed backend if used.
- Integration points: estimator or training loop.

**Implementation Steps (Detailed)**
1. Implement EOF-aware task wrapper.
2. Add tests for scalar and dict datasets.

**Tests (Detailed)**
- Python tests: `EofAwareTaskTest` in this file.
- Rust tests: mirror both cases.
- Cross-language parity test: compare global_step final value.

**Gaps / Notes**
- Tests call `hvd_lib.init()`; Rust may stub if Horovod not supported.

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

### `monolith/native_training/tensor_utils.py`
<a id="monolith-native-training-tensor-utils-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 162
- Purpose/role: Utilities for packing/unpacking tensors (including typed groups) and ragged squeeze helper.
- Key symbols/classes/functions: `maybe_squeeze_3d_tensor`, `pack_tensors`, `unpack_tensors`, `pack_typed_keyed_tensors`, `unpack_packed_tensors`, `split_tensors_with_type`, `merge_dicts`.
- External dependencies: TensorFlow, `static_reshape_op`.
- Side effects: None (pure tensor ops).

**Required Behavior (Detailed)**
- `maybe_squeeze_3d_tensor(x)`:
  - Accepts RaggedTensor with rank 2 or 3; squeezes axis=1 when rank 3.
  - Raises ValueError on non-ragged or unexpected rank.
- `pack_tensors(keyed_tensors)`:
  - Uses StaticReshapeNBuilder to flatten tensors to 1-D, concatenates in sorted key order.
  - Returns `(packed_tensor, sizes)` where sizes are per-tensor sizes.
- `unpack_tensors(keyed_shape, packed)`:
  - Splits packed tensor by sizes, reshapes to original shapes, returns dict by sorted key.
- `split_tensors_with_type` / `merge_dicts`:
  - Group tensors by dtype string; returns list of dicts.
  - `merge_dicts` flattens back into one dict.
- `pack_typed_keyed_tensors` / `unpack_packed_tensors`:
  - Packs list of dicts (already grouped by type) into list of packed tensors plus a final concat of sizes.
  - `unpack_packed_tensors` reconstructs list of dicts by shape metadata.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-tensor/src`.
- Rust public API surface: packing/unpacking helpers and ragged squeeze.
- Data model mapping: tensor shapes and sizes encoded as concatenated sizes.
- Feature gating: requires static reshape op or equivalent.
- Integration points: data transfer and async pipelines.

**Implementation Steps (Detailed)**
1. Port ragged squeeze helper with rank checks.
2. Implement pack/unpack with deterministic key ordering.
3. Implement typed packing and size concatenation.
4. Add tests for placeholder support and dtype grouping.

**Tests (Detailed)**
- Python tests: `monolith/native_training/tensor_utils_test.py`.
- Rust tests: add pack/unpack and typed pack/unpack tests.
- Cross-language parity test: compare packed tensors and reconstructed outputs.

**Gaps / Notes**
- `StaticReshapeNBuilder` relies on custom op; ensure Rust backend supports it.

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

### `monolith/native_training/tensor_utils_test.py`
<a id="monolith-native-training-tensor-utils-test-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 175
- Purpose/role: Tests for tensor packing/unpacking utilities.
- Key symbols/classes/functions: `TensorUtilsTest`.
- External dependencies: TensorFlow, `tensor_utils`.
- Side effects: None.

**Required Behavior (Detailed)**
- `test_maybe_squeeze_3d_tensor`: verifies ragged squeeze for rank 2/3.
- `test_pack_tensors`: pack/unpack dict of tensors; verifies sizes and values.
- `test_pack_typed_keyed_tensors`: pack/unpack multiple dtype dicts; verifies packed sizes and outputs.
- `test_pack_typed_keyed_tensors_with_placeholder`: supports placeholders with feed_dict.
- `test_split_tensors_with_type_and_merge_dicts`: dtype grouping and merge round-trip.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-tensor/src` (tests).
- Rust public API surface: pack/unpack helpers.
- Data model mapping: dtype grouping and shape handling.
- Feature gating: requires static reshape op support.
- Integration points: packing utilities.

**Implementation Steps (Detailed)**
1. Port tests for pack/unpack and typed packing.
2. Add placeholder-like tests if backend supports deferred shapes.

**Tests (Detailed)**
- Python tests: `TensorUtilsTest` in this file.
- Rust tests: mirror test cases.
- Cross-language parity test: compare packed sizes and reconstructed dicts.

**Gaps / Notes**
- Placeholder test relies on feed_dict semantics.

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

### `monolith/native_training/test_utils.py`
<a id="monolith-native-training-test-utils-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 65
- Purpose/role: Test helpers for hash table config and PS cluster setup.
- Key symbols/classes/functions: `generate_test_hash_table_config`, `create_test_ps_cluster`, `profile_it`.
- External dependencies: TensorFlow, `entry`, `utils`, `embedding_hash_table_pb2`.
- Side effects: Starts local TF servers; profiler writes to `/tmp/tests_profile`.

**Required Behavior (Detailed)**
- `generate_test_hash_table_config(dim=2, use_float16=False, learning_rate=1.0)`:
  - Builds `EmbeddingHashTableConfig` with cuckoo, one segment, SGD, zero init, fp32 compression.
  - Returns `entry.HashTableConfigInstance(config, [learning_rate])`.
- `create_test_ps_cluster(num_ps)`:
  - Creates `num_ps` local servers.
  - Builds `ClusterDef` with PS job and returns `(servers, ConfigProto)`.
- `profile_it(fn)` decorator:
  - Starts TensorFlow profiler with fixed options.
  - Stops profiler and sleeps 1s after run (note: `time` is not imported in file).

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src`.
- Rust public API surface: test helpers for hash table config and local cluster setup.
- Data model mapping: hash table config proto.
- Feature gating: TF runtime required for local server creation.
- Integration points: test infrastructure.

**Implementation Steps (Detailed)**
1. Port hash table config helper.
2. Implement local PS cluster helper if TF backend available.
3. Add profiling helper or stub.

**Tests (Detailed)**
- Python tests: none dedicated.
- Rust tests: use helpers in other test modules.
- Cross-language parity test: not applicable.

**Gaps / Notes**
- `profile_it` uses `time.sleep` but `time` is not imported; Python would error if called.

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

### `monolith/native_training/touched_key_set_ops.py`
<a id="monolith-native-training-touched-key-set-ops-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 61
- Purpose/role: Thin Python wrapper for TF custom ops that manage a thread-safe “touched key set” resource (insert IDs, steal+clear IDs).
- Key symbols/classes/functions: `TOUCHED_KEY_SET_CAPACITY`, `TOUCHED_KEY_SET_CONCURRENCY_LEVEL`, `create_touched_key_set`, `TouchedKeySet`.
- External dependencies: TensorFlow, `monolith.native_training.runtime.ops.gen_monolith_ops` (custom TF ops).
- Side effects:
  - Creates a stateful TF resource (`MonolithTouchedKeySet`) with `shared_name="MonolithTouchedKeySet" + name_suffix`.
  - `insert`/`steal` are stateful ops that mutate the underlying set.
  - `TouchedKeySet.__init__` ignores the `name_suffix` argument (potential bug / resource collision risk).

**Required Behavior (Detailed)**
- Constants:
  - `TOUCHED_KEY_SET_CAPACITY = 64 * 1024 * 1024 // (8 * 4)` → 2,097,152 (matches TF op default capacity).
  - `TOUCHED_KEY_SET_CONCURRENCY_LEVEL = 1024` (matches TF op default).
- `create_touched_key_set(capacity, concurrency_level, name_suffix="")`:
  - Calls TF custom op `MonolithTouchedKeySet` with `capacity`, `concurrency_level`, `shared_name="MonolithTouchedKeySet" + name_suffix`.
  - Returns a TF resource handle (scalar resource tensor).
- `TouchedKeySet.__init__(capacity=..., concurrency_level=..., name_suffix="")`:
  - Creates resource via `create_touched_key_set(capacity, concurrency_level)` **without** passing `name_suffix`.
  - Stores `_capacity`, `_concurrency_level`, and `_set` (resource handle).
- `TouchedKeySet.insert(ids)`:
  - Calls `monolith_touched_key_set_insert(handle, ids)`.
  - `ids` is `int64` tensor of any shape; op flattens via `NumElements()` and iterates in row-major order.
  - Returns `total_dropped_num` (int64) = sum of dropped keys across this call’s inserts.
  - Dropped keys semantics (from C++ hopscotch set):
    - On insert: if current size **> capacity**, the set is **cleared**, and `dropped_keys = size_before_clear`.
    - For each inserted ID, the per-key `Insert` returns `dropped_keys` (0 if no clear, `size_before_clear` if cleared during that insert).
    - Duplicate inserts return `dropped_keys` without increasing size.
  - The TF op allocates a 1-element tensor output; tests treat it as scalar.
- `TouchedKeySet.steal()`:
  - Calls `monolith_touched_key_set_steal(handle)`.
  - Returns all currently stored keys as a 1-D int64 tensor; **clears** the set.
  - Output order is **non-deterministic** (internal hopscotch table + `absl::flat_hash_set` iteration).
- Threading/concurrency:
  - Insert is thread-safe with per-bucket locks; `concurrency_level` controls lock count (rounded to power of two).
  - `steal` obtains a global clear lock and locks all buckets, blocking concurrent inserts while it drains.
- Determinism:
  - Output order of `steal` is unspecified; callers sort when comparing.
  - Overflow behavior is “clear-all” once size exceeds capacity (no partial eviction).
- Performance characteristics:
  - Average O(1) inserts; table size = next power of two of `capacity * 1.2`.
  - Uses extra overflow set (`absl::flat_hash_set`) when hopscotch insertion fails.
  - Lazy initialization to reduce memory until first insert.
- Metrics/logging: none.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-hash-table/src` (new `touched_key_set.rs` or equivalent).
- Rust public API surface:
  - `struct TouchedKeySet { capacity: u32, concurrency_level: u32, ... }`
  - `fn new(capacity: u32, concurrency_level: u32, name_suffix: Option<&str>) -> Self`
  - `fn insert(&self, ids: &[i64]) -> i64`
  - `fn steal(&self) -> Vec<i64>`
  - Accessors: `capacity()`, `concurrency_level()`.
- Data model mapping:
  - TF resource handle ↔ Rust in-process set instance.
  - If TF runtime backend is enabled, wrap the custom op handles instead of in-process implementation.
- Feature gating:
  - Default Rust-native implementation (no TF).
  - Optional `tf-runtime` feature: if a real `saved_model.pb` and custom ops are present, use TF-backed ops for parity.
- Integration points:
  - Parameter sync / hash table touched key tracking (see `runtime/ops/parameter_sync_tf_bridge.*` and `hash_table_op.cc` for Python-side wiring).
  - Rust hash-table or parameter-sync module should consume `TouchedKeySet` to report touched keys.

**Implementation Steps (Detailed)**
1. Define Rust `TouchedKeySet` API and decide backend selection (native vs TF runtime).
2. Port hopscotch hash set semantics (clear-all on `size > capacity`, duplicate-insert behavior, `steal` order nondeterminism).
3. Preserve lazy initialization (defer allocation until first insert) or document the deviation if eager.
4. Implement concurrency: shard locks by `concurrency_level`, match power-of-two rounding.
5. Ensure `insert` returns **total dropped count** per call (sum of per-key clears).
6. Implement `steal` to return all keys and clear; order undefined.
7. Add feature flag wiring in `monolith-rs` to switch to TF runtime custom ops when available.
8. Document the `name_suffix` bug in Python (ignored in `TouchedKeySet.__init__`) and decide whether Rust mirrors it or fixes it.

**Tests (Detailed)**
- Python tests: `monolith/native_training/touched_key_set_ops_test.py`.
- Rust tests:
  - `test_basic`: insert 0..999 into capacity 1000, `insert` returns 0, `steal` returns all keys (sorted match).
  - `test_overflow_clear`: insert 0..1004 into capacity 1000; `insert` returns 1001; `steal` returns {1001..1004}.
  - `test_duplicate_inserts`: repeated inserts do not increase size or return drops.
  - `test_thread_safety`: concurrent inserts + steal does not panic and preserves clear semantics.
- Cross-language parity test:
  - Run Python op + Rust implementation with same inserts; compare `total_dropped_num` and sorted `steal` output.

**Gaps / Notes**
- `TouchedKeySet.__init__` accepts `name_suffix` but never passes it to `create_touched_key_set` (likely unintended).
- TF op returns a 1-element tensor for `total_dropped_num` even though shape inference says scalar.

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

### `monolith/native_training/touched_key_set_ops_test.py`
<a id="monolith-native-training-touched-key-set-ops-test-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 51
- Purpose/role: Validates basic `TouchedKeySet` behavior (insert/steal) and overflow clear semantics.
- Key symbols/classes/functions: `TouchedKeySetOpsTest`, `test_touched_key_set_basic`, `test_touched_key_set_overflow`.
- External dependencies: TensorFlow, `TouchedKeySet` wrapper.
- Side effects: Creates TF resource-backed touched key set.

**Required Behavior (Detailed)**
- `test_touched_key_set_basic`:
  - Create `TouchedKeySet(1000, 1)`.
  - Insert `ids = [0..999]`, expect `total_dropped_num == 0`.
  - `steal()` returns exactly those IDs (order ignored; test sorts).
- `test_touched_key_set_overflow`:
  - Create `TouchedKeySet(1000, 1)`.
  - Insert `ids = [0..1004]`, expect `total_dropped_num == 1001` (clear-all on overflow).
  - `steal()` returns `[1001, 1002, 1003, 1004]` (sorted).
- Uses TF v1 session execution (`tf.test.TestCase` + `self.session()`).
- `tf.compat.v1.disable_eager_execution()` in `__main__`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-hash-table/src` (new touched key set tests).
- Rust public API surface: `TouchedKeySet::new`, `insert`, `steal`.
- Data model mapping: `Vec<i64>` ↔ TF int64 tensor equivalents.
- Feature gating: if TF backend is optional, tests should run against native implementation; add TF-backed parity tests under feature flag.
- Integration points: unit tests in hash table crate; optional integration test comparing to Python output.

**Implementation Steps (Detailed)**
1. Port `test_touched_key_set_basic` as Rust unit test (sort results before compare).
2. Port `test_touched_key_set_overflow` to validate clear-all behavior.
3. Add a Rust test for duplicate insert (not in Python, but covers insert semantics).
4. Add optional cross-language test (Python vs Rust) if CI can run TF custom ops.

**Tests (Detailed)**
- Python tests: `TouchedKeySetOpsTest.test_touched_key_set_basic`, `.test_touched_key_set_overflow`.
- Rust tests:
  - `touched_key_set_basic`
  - `touched_key_set_overflow`
  - `touched_key_set_duplicate_insert` (optional but recommended)
- Cross-language parity test:
  - Compare Python session outputs vs Rust outputs for the two scenarios above.

**Gaps / Notes**
- Output ordering from `steal` is non-deterministic; tests must sort before compare.

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

### `monolith/native_training/utils.py`
<a id="monolith-native-training-utils-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 320
- Purpose/role: Utility helpers for TF-native training (device strings, gradient backprop helpers, shape helpers, params introspection, graph dependency checks, TF collections, env-driven metric prefix).
- Key symbols/classes/functions: `PS_JOB_NAME`, `ps_device`, `propagate_back_gradients`, `propagate_back_dict_gradients`, `get_ndim`, `int_shape`, `extend_as_list`, `check_list`, `to_snake_case`, `to_list`, `_get_parameters`, `_get_all_parameters`, `_inverted_index`, `params`, `check_ops_dependence`, `with_params`, `get_local_host`, `get_test_tmp_dir`, `get_debugging_info_file_name`, `get_meta_graph_file_name`, `add_to_collections`, `get_collection`, `set_metric_prefix`, `get_metric_prefix`.
- External dependencies: TensorFlow (core + internal graph utils), `absl.logging`, `monolith.core.base_layer.get_uname`, `monolith.core.hyperparams` (`allowed_kwargs`, `InstantiableParams`, `Params`), stdlib (`os`, `platform`, `socket`, `re`, `inspect`, `copy`, `types`).
- Side effects:
  - Raises `RuntimeError`, `TypeError`, `ValueError`, generic `Exception` in validation helpers.
  - Mutates TF collections via `tf.compat.v1.add_to_collections`.
  - Sets environment variable `MONOLITH_METRIC_PREFIX`.

**Required Behavior (Detailed)**
- Constants:
  - `PS_JOB_NAME = "ps"`.
- `ps_device(index)`:
  - Returns `"/job:ps/task:{index}/device:CPU:0"` (string formatting only; no validation).
- `propagate_back_gradients(grads_and_vars, xs, valid_var_set=None)`:
  - Iterates `(grad, var)` pairs; if `valid_var_set` and `var` not in it → `RuntimeError("Invalid variables in the input", var, valid_var_set)`.
  - Accumulates `combined_vars` and `combined_grads` in input order.
  - Returns `tf.gradients(combined_vars, list(xs), combined_grads)` (list aligned with `xs` order).
- `propagate_back_dict_gradients(grads_and_vars, x_to_key, valid_var_set=None)`:
  - Calls `propagate_back_gradients` using `x_to_key.keys()`.
  - Zips returned `dxs` with `x_to_key.items()` (dict iteration order), grouping into `defaultdict(list)` keyed by `key` → list of `(dx, x)`.
- `get_ndim(x)`:
  - Returns `len(x.get_shape()._dims)` if `_dims` is not `None`; else `None`.
- `int_shape(x)`:
  - Uses `x.get_shape().as_list()`; maps `None` → `-1`, `int` as-is, `tf.compat.v1.Dimension` → `.value`.
  - Any other dim type raises `ValueError`; function catches `ValueError` and returns `None`.
- `extend_as_list(x, n)`:
  - If `x` is `list`/`tuple`:
    - If `len(x) < n`: return `x + [None] * (n - len(x))`.
    - Else: return `x` as-is (no copy).
  - Else: attempts `[x if i == 0 else deepcopy(x) for i in range(n)]`; if deepcopy fails, returns `[x] * n`.
- `check_list(candidate, length_checker, could_be_none=False)`:
  - If `candidate is None` and `not could_be_none`: `TypeError`.
  - Only accepts `None` or `list`; other types (including `tuple`) → `TypeError`.
  - If `candidate` is list and `length_checker(len(candidate))` is false → `ValueError`.
  - Returns `candidate` unchanged.
- `to_snake_case(name)`:
  - Inserts underscores between camel-case boundaries; lowercases.
  - If result starts with `_`, prefix with `"private"`.
- `to_list(x)`:
  - If `x` is a `list`, returns `x` (no copy).
  - Else returns `[x]` (tuples are not expanded).
- Parameter helpers:
  - `_get_parameters(cls, parameters)` collects `__init__` parameters except `self/cls` and varargs/kwargs.
  - `_get_all_parameters` walks base classes (excluding `object`) then calls `_get_parameters`.
  - `_inverted_index(ips, idx_dict)` recursively maps param names to their `InstantiableParams` container.
  - `params(cls)`:
    - Finds nearest base with `.params()`, otherwise `InstantiableParams(cls)`.
    - Sets `ips.cls = cls`.
    - Builds param list from `__init__` signature across the MRO.
    - Attempts to define `name` using `get_uname(cls.__name__)` (ignores exceptions).
    - For each parameter:
      - If name exists in inverted index: update default value if non-empty.
      - Else: `ips.define(name, default_or_None, name)`; if define fails and default is not empty/None, set via `ips[name] = default`.
    - Defines `allowed_kwargs` with default `None` (ignores exceptions).
    - Returns `ips`.
- `check_ops_dependence(op_names_1, op_names_2)`:
  - Extracts subgraph of `op_names_1` from default graph.
  - If any op in `op_names_2` appears in that subgraph, raises `Exception` with message:
    `"Checking ops dependence, the ops [%s] depend on ops [%s], which may cause ops [%s] to be run twice."`
- `with_params(cls)`:
  - Binds `params` as a method of `cls` and returns `cls`.
- Host/paths/env:
  - `get_local_host()`:
    - Windows/Linux: `socket.gethostbyname(socket.gethostname())`.
    - Other OS: `socket.gethostbyname(socket.gethostname() + ".local")`.
  - `get_test_tmp_dir()` reads `TEST_TMPDIR` env var, default `/tmp`.
  - `get_debugging_info_file_name(model_dir)` → `model_dir/debugging_info.pb`.
  - `get_meta_graph_file_name(model_dir)` → `model_dir/meta_graph_for_debugging.pb`.
  - `set_metric_prefix(prefix)` sets `MONOLITH_METRIC_PREFIX`.
  - `get_metric_prefix()` returns `MONOLITH_METRIC_PREFIX` or default `"monolith.training"`.
- Collections:
  - `add_to_collections(names, value)`:
    - If value is `bool|int|float|str`, always adds (even falsy).
    - Else if `value` is truthy, adds.
    - Else logs `value is {value}, skip`.
  - `get_collection(name)`:
    - If collection is `bool|int|float|str`, returns it directly.
    - Else if collection is truthy (non-empty list), returns it.
    - Else returns `None`.
- Threading/concurrency: no explicit locks here (TF graph-level ops only).
- Determinism: `propagate_back_dict_gradients` relies on dict iteration order (preserves insertion order in Python 3.7+).
- Performance: `check_ops_dependence` materializes graph def + subgraph; can be expensive on large graphs.
- Metrics/logging: only `add_to_collections` logs when skipping falsy values.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/utils.rs` (new).
- Rust public API surface:
  - `ps_device(index: usize) -> String`
  - `propagate_back_gradients(...)` / `propagate_back_dict_gradients(...)` (TF backend only)
  - `get_ndim`, `int_shape`, `extend_as_list`, `check_list`, `to_snake_case`, `to_list`
  - `params` analog for Rust config structs (likely a builder/derive macro)
  - `check_ops_dependence` (TF backend only)
  - `get_local_host`, `get_test_tmp_dir`, `get_debugging_info_file_name`, `get_meta_graph_file_name`
  - `add_to_collections` / `get_collection` (TF backend only)
  - `set_metric_prefix`, `get_metric_prefix`
- Data model mapping:
  - TF tensors/graph ops map only under `tf-runtime` feature.
  - Rust-native training uses native tensors and may skip TF-specific utilities.
- Feature gating:
  - `tf-runtime` for gradient propagation + collections + graph dependence checks.
  - Native-only implementations for string/path/env helpers.
- Integration points:
  - `monolith-training` modules referencing `utils.ps_device`, `utils.get_metric_prefix`, `utils.get_test_tmp_dir`.

**Implementation Steps (Detailed)**
1. Create `monolith-training/src/utils.rs` and port pure helpers (string/path/env/shape helpers).
2. Implement `ps_device` exactly (string formatting).
3. For `propagate_back_*`, gate behind TF runtime and map to TF gradient APIs (or document unsupported for Candle).
4. Recreate `params` behavior in Rust (likely a builder or derive macro). Preserve defaults + `allowed_kwargs` handling.
5. Implement `check_ops_dependence` for TF backend; otherwise return Ok/NotSupported.
6. Provide collection helpers under TF backend (or maintain a Rust-side registry if needed).
7. Mirror exceptions and error messages where externally visible (especially `RuntimeError` in `propagate_back_gradients`).

**Tests (Detailed)**
- Python tests: `monolith/native_training/utils_test.py`.
- Rust tests:
  - `test_propagate_back_dict_gradients` (TF backend only).
  - `test_check_ops_dependence` (TF backend only).
  - `test_collections` (TF backend only).
  - Unit tests for `ps_device`, `to_snake_case`, `extend_as_list`, `check_list`, `get_metric_prefix`.
- Cross-language parity test:
  - Compare `propagate_back_dict_gradients` outputs (dx, x) for the same graph.

**Gaps / Notes**
- `extend_as_list` will error for tuples when `len(x) < n` due to `tuple + list` (no guard).
- `to_list` treats tuples as a single element (not expanded).
- Uses private TF shape attribute `_dims`.
- Several imports are unused (`numpy.lib.arraysetops.isin`, `tensorflow.python.framework.ops`, `variables`, `six`).

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

### `monolith/native_training/utils_test.py`
<a id="monolith-native-training-utils-test-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 70
- Purpose/role: Tests gradient backprop grouping, graph dependence checks, and TF collection helpers.
- Key symbols/classes/functions: `UtilsTest`, `test_propagate_back_dict_gradients`, `test_check_ops_dependence`, `test_collections`.
- External dependencies: TensorFlow, `monolith.native_training.utils`.
- Side effects: Adds TF collections within the test graph.

**Required Behavior (Detailed)**
- `test_propagate_back_dict_gradients`:
  - Create `x = tf.Variable(8.0)`, `y = 2 * x`, `grad_y = 3 * y`.
  - `valid_vars = {y}`; call `propagate_back_dict_gradients(zip([grad_y], [y]), {x: "group1"}, valid_vars)`.
  - After session init, `grouped["group1"]` yields list `[(dx, x)]` with `dx == 96` and `x == 8`.
- `test_check_ops_dependence`:
  - `v.assign_add(1)` stored as `add`; create `t1`, `t2` under control dependency on `add`.
  - `check_ops_dependence(t1.op.name, add.name)` raises `Exception`.
  - `check_ops_dependence(t1.op.name, t2.op.name)` does not raise.
- `test_collections`:
  - Adds scalars, lists, and None to collections.
  - `get_collection('int')[-1] == 2`, `'str'[-1] == 'str'`, `'bool'[-1] == True`.
  - Lists: last entries are `[4,5,6]`, `['hello','world']`, `[False]` (empty list and None are skipped).
- Uses TF v1 session mode (`tf.test.TestCase` + `self.session()`).
- `tf.compat.v1.disable_eager_execution()` in `__main__`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests/utils.rs` (new).
- Rust public API surface: `propagate_back_dict_gradients`, `check_ops_dependence`, `add_to_collections`, `get_collection`.
- Data model mapping: Rust-side tensor or TF runtime tensor under `tf-runtime` feature.
- Feature gating: tests run only when TF backend is enabled.
- Integration points: `monolith-training` utils module.

**Implementation Steps (Detailed)**
1. Port `test_propagate_back_dict_gradients` using TF backend (or skip with explicit feature guard).
2. Port `test_check_ops_dependence` by constructing a simple TF graph and verifying exception behavior.
3. Port `test_collections` by adding values to TF collections and asserting last entries.
4. Add native-only tests for non-TF helpers (string/path/env) separately.

**Tests (Detailed)**
- Python tests: `UtilsTest.test_propagate_back_dict_gradients`, `.test_check_ops_dependence`, `.test_collections`.
- Rust tests:
  - `propagate_back_dict_gradients_matches_python` (TF backend)
  - `check_ops_dependence_raises_on_dependency` (TF backend)
  - `collections_add_get` (TF backend)
- Cross-language parity test:
  - Compare gradients and collection ordering between Python and Rust TF backends.

**Gaps / Notes**
- Collections ordering relies on TF collection append order; Rust must preserve the same semantics.

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

### `monolith/native_training/variables.py`
<a id="monolith-native-training-variables-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 147
- Purpose/role: Provides a cached-variable mechanism for distributed TF training: creates local cached copies of remote variables, updates them via fetch/assign ops, and supplies a session hook.
- Key symbols/classes/functions: `_CACHED_VARIABLES`, `CachedVariableAssociates`, `CachedVariableMeta`, `cached_value`, `cached_variable_creator`, `fetch_all_cached_variables`, `assign_all_cached_variables`, `FetchAllCachedVariablesHook`.
- External dependencies: TensorFlow (resource variables, custom gradients, estimator hooks), `graph_meta.get_meta`.
- Side effects:
  - Mutates variables’ private `_cached_value` attribute.
  - Adds variables to TF collections.
  - Creates local (worker) `LOCAL_VARIABLES` for cache/fetch.

**Required Behavior (Detailed)**
- `_CACHED_VARIABLES = "monolith_cached_variables"`: TF collection name storing original vars that are cached.
- Data classes:
  - `CachedVariableAssociates(async_fetched_var, async_cached_var)`.
  - `CachedVariableMeta(var_id_to_assoc: Dict[int, CachedVariableAssociates])`.
- `_get_meta()`:
  - Uses `graph_meta.get_meta("cached_variables_meta", CachedVariableMeta)` to create/retrieve per-graph metadata.
- `cached_value(var, async_cached_var)` (custom gradient):
  - Forward: returns `async_cached_var` (cached local value).
  - Backward: returns gradient `(dy, None)` → gradients flow to `var` only; cached var has no gradient.
- `_get_valid_op_name(name)`:
  - Replaces `":"` and `"/"` with `"_"` for safe op naming.
- `cached_variable_creator(next_creator, **kwargs)`:
  - Creates `var = next_creator(**kwargs)`.
  - Validates: `var` must be `ResourceVariable` else `ValueError("Only ResourceVariable is supported. Do you disable V2 behavior or use strategy?")`.
  - If `var._cached_value` already set: `ValueError("The variable has already been cached. Consider about removing cache_device.")`.
  - Creates `async_cached_var` and `async_fetched_var` under `tf.device(None)`:
    - `resource_variable_ops.ResourceVariable`
    - `initial_value=var.initial_value`, `trainable=False`
    - `collections=[tf.compat.v1.GraphKeys.LOCAL_VARIABLES]`
    - `shape=var.shape`, `dtype=var.dtype`
  - If `async_cached_var.device == var.device`, returns `var` unchanged (skip cache).
  - Else:
    - Adds `var` to `_CACHED_VARIABLES` collection.
    - Sets `var._cached_value = cached_value(var, async_cached_var)`.
    - Stores associates in meta: `var_id_to_assoc[id(var)] = CachedVariableAssociates(...)`.
  - Returns `var`.
- `fetch_all_cached_variables()`:
  - For each `var` in `_CACHED_VARIABLES`:
    - Gets `fetched_var = meta.var_id_to_assoc[id(var)].async_fetched_var`.
    - Adds assign op: `fetched_var.assign(var._read_variable_op(), name="fetch_from_{device}", read_value=False)`, where device is sanitized via `_get_valid_op_name`.
  - Returns `tf.group(ops)` (no name).
- `assign_all_cached_variables()`:
  - For each cached `var`:
    - Assigns `associates.async_cached_var.assign(associates.async_fetched_var, name="assign_cached_var", read_value=False)`.
  - Returns `tf.group(ops, name="assign_all_cached_variables")`.
- `FetchAllCachedVariablesHook`:
  - Initializes `_fetch_op`, `_assign_op`, `_first_run = True`.
  - `after_create_session`: resets `_first_run`.
  - `before_run`:
    - If first run: synchronously `session.run(_fetch_op)` then `session.run(_assign_op)`.
    - Returns `SessionRunArgs(_fetch_op)` to fetch each step.
  - `after_run`: runs `_assign_op` to update cached vars after fetch.
- Observed behavior:
  - `var * 1.0` uses cached value (`_cached_value`), while `var` reads the actual remote value.
  - Updates to `var` only appear in cached reads after fetch/assign.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/variables.rs` (new).
- Rust public API surface:
  - `cached_variable_creator(...)` equivalent for TF backend.
  - `fetch_all_cached_variables()`, `assign_all_cached_variables()`.
  - `FetchAllCachedVariablesHook` analog (or callback in training loop).
- Data model mapping:
  - TF resource variables ↔ TF runtime handles.
  - Cached local variables stored in a per-graph registry keyed by var identity.
- Feature gating:
  - Only meaningful under `tf-runtime` (Candle backend has no TF graph/collections).
- Integration points:
  - Distributed training loops that rely on cached reads (e.g., PS-based training).

**Implementation Steps (Detailed)**
1. Add TF-backend registry for cached variables (keyed by var identity/handle).
2. Implement cached read wrapper (custom gradient or TF graph rewrite) so reads use cached local vars but gradients flow to originals.
3. Create local (worker) cached + fetched variables with `LOCAL_VARIABLES` collection.
4. Implement fetch/assign ops; preserve op naming via `_get_valid_op_name`.
5. Add a hook/callback in training loop that mirrors `FetchAllCachedVariablesHook` scheduling.
6. Define behavior when cached vars colocate with originals (skip caching).
7. Gate all behavior behind TF runtime feature; document unsupported for Candle.

**Tests (Detailed)**
- Python tests: `monolith/native_training/variables_test.py`.
- Rust tests:
  - `test_basic_cached_variable` (TF backend).
  - `test_hook_cached_variable` (TF backend).
  - `test_gradient_cached_variable` (TF backend).
- Cross-language parity test:
  - Run Python and Rust graphs with same PS setup and compare cached vs direct reads after updates.

**Gaps / Notes**
- Uses private TF APIs: `var._cached_value`, `var._read_variable_op()`.
- Imports `variables_lib` and `core` but does not use them.

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

### `monolith/native_training/variables_test.py`
<a id="monolith-native-training-variables-test-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 89
- Purpose/role: Validates cached-variable behavior under PS training, hook scheduling, and gradient updates.
- Key symbols/classes/functions: `CachedVariableTest.testBasic`, `.testHook`, `.testGradient`.
- External dependencies: TensorFlow, `variables`, `test_utils.create_test_ps_cluster`.
- Side effects: Starts local PS servers for test sessions.

**Required Behavior (Detailed)**
- `testBasic`:
  - Create local PS cluster with 2 servers.
  - Use `tf.variable_creator_scope(cached_variable_creator)` and place `var` on `/job:ps/task:1`.
  - After init:
    - `var * 1.0` returns cached value (5.0).
    - After `var.assign_add(2.0)`: `var * 1.0` still 5.0; `var` reads 7.0.
  - After `fetch_all_cached_variables()` + `assign_all_cached_variables()`: `var * 1.0` becomes 7.0.
- `testHook`:
  - Build `var` on PS with cache creator; `var_cached = var * 1.0`.
  - Use `FetchAllCachedVariablesHook` with `SingularMonitoredSession`.
  - After `assign_sub(var, 1.0)`:
    - `var_cached` may take up to two runs to reflect update; final expected value is 4.0.
- `testGradient`:
  - Use SGD optimizer with `loss = var`.
  - After `opt.minimize(loss)`: `var` reads 4.0 (updated).
  - Cached read (`var * 1.0`) still 5.0 (not fetched yet).
- Tests run with TF v1 sessions; eager disabled in `__main__`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests/variables.rs` (new).
- Rust public API surface: `cached_variable_creator`, `fetch_all_cached_variables`, `assign_all_cached_variables`, `FetchAllCachedVariablesHook`.
- Data model mapping: TF runtime variables and sessions.
- Feature gating: TF backend only.
- Integration points: PS cluster creation helpers (Rust equivalent of `test_utils.create_test_ps_cluster`).

**Implementation Steps (Detailed)**
1. Port `testBasic` using a local PS cluster and cached variable creator.
2. Port `testHook` with a Rust equivalent of session hook (before_run/after_run callbacks).
3. Port `testGradient` to ensure gradients update original var but cached reads remain stale.
4. Ensure the cached read path (`var * 1.0`) uses cached value in Rust TF backend.

**Tests (Detailed)**
- Python tests: `CachedVariableTest.testBasic`, `.testHook`, `.testGradient`.
- Rust tests:
  - `cached_variable_basic` (TF backend)
  - `cached_variable_hook_updates` (TF backend)
  - `cached_variable_gradient_reads` (TF backend)
- Cross-language parity test:
  - Compare cached vs direct reads across the same update sequence in Python and Rust.

**Gaps / Notes**
- Hook behavior is sensitive to session scheduling; Rust must preserve the “fetch before run, assign after run” pattern.

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

### `monolith/native_training/yarn_runtime.py`
<a id="monolith-native-training-yarn-runtime-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 127
- Purpose/role: Yarn/Primus runtime helpers for local host resolution and AppMaster gRPC control (kill, finish, savepoint).
- Key symbols/classes/functions: `get_local_host`, `_get_primus_am_host`, `_get_channel`, `maybe_kill_application`, `maybe_finish_application`, `create_primus_save_point`.
- External dependencies: `grpc`, `primus_am_service_pb2(_grpc)`, `net_utils.get_local_ip`, env vars.
- Side effects:
  - Creates/caches gRPC channels.
  - Sends kill/succeed/savepoint requests to AppMaster.
  - Sleeps while waiting for savepoint completion.
  - Logs info/error messages.

**Required Behavior (Detailed)**
- `get_local_host()`:
  - If `CLOUDNATIVE_INET_ADDR` env var exists: use first entry before comma.
  - Else if `YARN_INET_ADDR` exists: use that value.
  - Else: `net_utils.get_local_ip()`.
  - Asserts non-empty result and returns it.
- `_get_primus_am_host()`:
  - If both `PRIMUS_AM_RPC_HOST` and `PRIMUS_AM_RPC_PORT` are set, returns `"host:port"`.
  - Else returns empty string.
- `_get_channel(addr)`:
  - Caches `grpc.insecure_channel(addr)` in `_CHANNEL_MAP`.
  - Returns cached channel for the same addr.
- `maybe_kill_application(reason) -> bool`:
  - If Primus AM host is available:
    - Builds `KillRequest` with `exit_code=1`, `diagnose=reason`, `graceful_shutdown_timeout_ms=20000`.
    - Calls `stub.kill(req, timeout=10)`.
    - Returns `True` on success; `False` on `grpc.RpcError`.
  - If no host: logs “Current framework doesn't support kill. Ignore killing...” and returns `False`.
- `maybe_finish_application()`:
  - If Primus AM host is available:
    - Builds `SucceedRequest` with `graceful_shutdown_timeout_ms=20000`.
    - Calls `stub.succeed(req, timeout=10)`.
    - Returns `True` on success; logs on `grpc.RpcError` (returns `None`).
  - If no host: returns `None` (no log).
- `create_primus_save_point(dst) -> bool`:
  - If Primus AM host is available:
    - Calls `createSavepoint` with `savepoint_dir=dst`.
    - If response `code != 0`, logs error and returns `False`.
    - Else polls `createSavepointStatus` every 5s:
      - `PENDING`/`RUNNING`: sleep and continue.
      - `SUCCEEDED`: log success and return `True`.
      - Any other state: log error and return `False`.
    - On `grpc.RpcError`: logs and returns `False`.
  - If no host: returns `None`.
- Threading/concurrency:
  - `_CHANNEL_MAP` is a global dict without a lock; concurrent callers can race.
- Performance:
  - Savepoint polling is blocking with 5s sleep intervals.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/yarn_runtime.rs` (new).
- Rust public API surface:
  - `get_local_host() -> String`
  - `maybe_kill_application(reason: &str) -> bool`
  - `maybe_finish_application() -> Option<bool>`
  - `create_primus_save_point(dst: &str) -> Option<bool>`
- Data model mapping:
  - gRPC service `primus_am_service.proto` already in `monolith-rs/proto`.
  - Use tonic/grpc-generated client stubs for `AppMasterService`.
- Feature gating:
  - Requires gRPC + proto build; optional if runtime doesn't use Primus.
- Integration points:
  - Distributed training orchestration that needs to terminate or checkpoint jobs.

**Implementation Steps (Detailed)**
1. Generate Rust gRPC client from `primus_am_service.proto`.
2. Implement env-var host resolution (`CLOUDNATIVE_INET_ADDR` > `YARN_INET_ADDR` > `net_utils` equivalent).
3. Add a channel cache (e.g., `DashMap` or `Mutex<HashMap<...>>`) to mirror `_CHANNEL_MAP`.
4. Implement `maybe_kill_application` and `maybe_finish_application` with the same timeout and fields.
5. Implement `create_primus_save_point` with polling loop + 5s sleep.
6. Mirror return semantics (False vs None) or document a deliberate normalization.

**Tests (Detailed)**
- Python tests: `monolith/native_training/yarn_runtime_test.py`.
- Rust tests:
  - `get_local_host_env_override` (CLOUDNATIVE_INET_ADDR + YARN_INET_ADDR cases).
  - gRPC kill/finish/savepoint flow using an in-process gRPC server.
- Cross-language parity test:
  - Simulate the same gRPC server and verify request fields + timeouts.

**Gaps / Notes**
- `maybe_finish_application` does not return `False` on errors; it logs and returns `None`.
- `create_primus_save_point` and `maybe_finish_application` return `None` when no Primus host is configured.

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

### `monolith/native_training/yarn_runtime_test.py`
<a id="monolith-native-training-yarn-runtime-test-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 133
- Purpose/role: Exercises env-based host resolution and gRPC AppMaster controls (kill/finish/savepoint).
- Key symbols/classes/functions: `YarnRuntimeTest`, `test_get_local_host_*`, `test_kill`, `test_finish`, `test_save_primus`.
- External dependencies: gRPC, Primus AM proto stubs, `unittest.mock` for env vars.
- Side effects: Starts in-process gRPC servers (unix: addresses).

**Required Behavior (Detailed)**
- `test_get_local_host_overwrite`:
  - With `YARN_INET_ADDR=1.2.3.4`, `get_local_host()` returns `1.2.3.4`.
- `test_get_local_host_overwrite_by_cloudnative`:
  - With `CLOUDNATIVE_INET_ADDR=1.2.3.4,5.6.7.8`, returns `1.2.3.4`.
- `test_get_local_host_basic`:
  - Calls `get_local_host()` without explicit env overrides (no assertion).
- `test_kill`:
  - Sets `PRIMUS_AM_RPC_HOST=unix`, `PRIMUS_AM_RPC_PORT=test_kill`.
  - Starts gRPC server with `kill` handler; validates `request.diagnose` equals reason.
  - Calls `maybe_kill_application(reason)` and asserts handler invoked.
- `test_finish`:
  - Similar setup; installs `succeed` handler; asserts invoked after `maybe_finish_application()`.
- `test_save_primus`:
  - Sets Primus host/port; installs `createSavepoint` and `createSavepointStatus` handlers.
  - Status handler returns `SUCCEEDED`; `create_primus_save_point(dst)` returns `True`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests/yarn_runtime.rs` (new).
- Rust public API surface: `get_local_host`, `maybe_kill_application`, `maybe_finish_application`, `create_primus_save_point`.
- Data model mapping: use Rust gRPC server/client for `AppMasterService`.
- Feature gating: tests require gRPC + proto build; optional for non-Primus builds.
- Integration points: gRPC server harness for tests (likely tokio + tonic).

**Implementation Steps (Detailed)**
1. Port env override tests for `get_local_host`.
2. Implement a local gRPC server with unary handlers for `kill`, `succeed`, `createSavepoint`, `createSavepointStatus`.
3. Validate request fields (e.g., `diagnose` for kill) and that calls occur.
4. Ensure unix-domain or loopback TCP addressing works in Rust test harness.

**Tests (Detailed)**
- Python tests: `YarnRuntimeTest.test_get_local_host_*`, `.test_kill`, `.test_finish`, `.test_save_primus`.
- Rust tests:
  - `get_local_host_env_overrides`
  - `maybe_kill_application_calls_stub`
  - `maybe_finish_application_calls_stub`
  - `create_primus_save_point_succeeds`
- Cross-language parity test:
  - Compare request fields and call ordering with a mock gRPC server.

**Gaps / Notes**
- Python tests use gRPC addresses like `unix:test_kill`; Rust must support unix-domain channels or adapt tests to TCP.

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

### `monolith/native_training/zk_utils.py`
<a id="monolith-native-training-zk-utils-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 96
- Purpose/role: Zookeeper utilities for host selection (IPv4 vs IPv6), default server list, authenticated Kazoo client, and cleanup of stale ZK paths.
- Key symbols/classes/functions: `_PORT`, `_HOSTS`, `_HOSTS_IPV6`, `is_ipv6_only`, `default_zk_servers`, `MonolithKazooClient`, `clear_zk_path`.
- External dependencies: `kazoo.client.KazooClient`, `env_utils.get_zk_auth_data`, `socket`, `datetime`.
- Side effects:
  - Connects to ZK, mutates nodes (delete paths).
  - Logs informational messages about IP resolution.

**Required Behavior (Detailed)**
- Globals:
  - `_PORT = 2181`.
  - `_HOSTS` and `_HOSTS_IPV6` are defined with IPs, then overwritten to empty lists later (current effective values are empty).
- `is_ipv6_only()`:
  - If any of `MY_HOST_IP`, `MY_POD_IP`, `MY_HOST_IPV6` env vars are set:
    - Treat as “tce/byterec env”.
    - `ipv4_addr` from `MY_HOST_IP` or `MY_POD_IP` (may be `None`).
  - Else:
    - `ipv4_addr = socket.gethostbyname(socket.gethostname())` (except → `None`).
  - Logs `ipv4_addr`, then sets `ipv6_only = not ipv4_addr` and logs the result.
  - Returns `ipv6_only`.
- `default_zk_servers(use_ipv6: bool = False)`:
  - If `use_ipv6` or `is_ipv6_only()`:
    - Returns comma-joined `"[ip]:port"` entries from `_HOSTS_IPV6`.
  - Else:
    - Returns comma-joined `"ip:port"` entries from `_HOSTS`.
  - With current `_HOSTS`/`_HOSTS_IPV6` emptied, returns empty string.
- `MonolithKazooClient`:
  - Subclasses `KazooClient`.
  - If `auth_data` not provided, injects `get_zk_auth_data()` into kwargs.
- `clear_zk_path(zk_server, job_name, force_clear_zk_path)`:
  - Connects with `MonolithKazooClient(zk_server or default_zk_servers())`.
  - `base_path = "/monolith"`, `delta = timedelta(weeks=9)`.
  - `ensure_path(base_path)`.
  - For each child under `/monolith`:
    - `get_children(path, include_data=True)` → stat; if `stat.mtime // 1000 + delta < now`, delete recursively (ignore errors).
  - For current `job_name`:
    - If exists and `force_clear_zk_path`: delete recursively.
    - Else: raise `ValueError("there are [<children>] in monolith zk path")` using current children list.
  - Always stops client in `finally`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/zk_utils.rs` (new).
- Rust public API surface:
  - `is_ipv6_only() -> bool`
  - `default_zk_servers(use_ipv6: bool) -> String`
  - `MonolithZkClient` wrapper (auth inject)
  - `clear_zk_path(zk_server: Option<&str>, job_name: &str, force: bool) -> Result<(), Error>`
- Data model mapping:
  - Kazoo client ↔ Rust ZK client (e.g., `zookeeper` crate).
  - `stat.mtime` (ms) ↔ Rust stat `mtime` (ms).
- Feature gating:
  - ZK client optional for environments without ZK.
- Integration points:
  - Any training components relying on ZK-based coordination or cleanup.

**Implementation Steps (Detailed)**
1. Decide Rust ZK client crate and auth mechanism matching `get_zk_auth_data()`.
2. Implement `is_ipv6_only` using env vars and hostname resolution.
3. Port `default_zk_servers` formatting (IPv6 `[ip]:port`).
4. Implement `clear_zk_path` with the same TTL logic (9 weeks) and error message.
5. Preserve the “ignore delete errors” behavior on stale node cleanup.

**Tests (Detailed)**
- Python tests: none in repo.
- Rust tests:
  - Unit tests for `default_zk_servers` formatting and env-driven `is_ipv6_only`.
  - Integration tests for `clear_zk_path` if ZK test container is available.
- Cross-language parity test:
  - Compare TTL deletion behavior with a mocked ZK server.

**Gaps / Notes**
- `_HOSTS` and `_HOSTS_IPV6` are overwritten to empty lists, so `default_zk_servers` returns empty string by default (likely a sanitization artifact).

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

### `monolith/path_utils.py`
<a id="monolith-path-utils-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 47
- Purpose/role: Locate monolith base directory and resolve paths to bundled libraries.
- Key symbols/classes/functions: `find_main`, `get_libops_path`.
- External dependencies: `os` only (explicitly avoids third-party imports).
- Side effects: raises ValueError when base directory cannot be found.

**Required Behavior (Detailed)**
- `find_main()`:
  - Uses `__file__` path; searches for split markers in order: `/__main__/`, `/site-packages/`, `/monolith/`.
  - If marker found:
    - For `/monolith/`: `main_dir` is path prefix before marker.
    - Else: `main_dir` is prefix joined with the marker path component.
  - Returns `main_dir` only if `main_dir/monolith` exists; else raises ValueError with full path in message.
- `get_libops_path(lib_name)`:
  - Returns `os.path.join(find_main(), lib_name)`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/src/path_utils.rs`.
- Rust public API surface: `find_main()` and `get_libops_path()`.

**Implementation Steps (Detailed)**
1. Implement path resolution with the same marker order and behavior.
2. Preserve error message details for diagnostics.

**Tests (Detailed)**
- Python tests: `monolith/utils_test.py` (find_main / get_libops_path).
- Rust tests: replicate `find_main` behavior under test harness.

**Gaps / Notes**
- Behavior depends on Bazel layout (`__main__`); Rust tests should simulate or adjust.

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

### `monolith/tpu_runner.py`
<a id="monolith-tpu-runner-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 429
- Purpose/role: TPU training/eval runner using `tf.estimator.tpu.TPUEstimator` with embedding support and optional CPU test mode.
- Key symbols/classes/functions: `TPURunner`, `create_tpu_estimator`, `create_tpu_estimator_on_cpu`, `run`, CLI `main`.
- External dependencies: `tensorflow.compat.v1`, `cloud_tpu_client`, `BaseEmbeddingTask`, `model_registry`.
- Side effects: configures TPU versions, launches training/eval, writes summaries.

**Required Behavior (Detailed)**
- CLI flags include TPU location (`tpu`, `gcp_project`, `tpu_zone`), mode, model_dir, checkpoint interval, iteration counts, embedding options, CPU test, partition strategy, overwrite_end_date.
- `TPURunner.__init__`:
  - Reads flags and task_param; sets accelerator to `tpu`.
  - Allows task_param to override save_checkpoints_steps.
  - Optionally overwrites `train.end_date` when flag provided.
- `create_tpu_estimator`:
  - Optionally configures TPU version and waits for healthy.
  - Builds TPU cluster resolver, computes total replicas and global batch size.
  - Sets TPUConfig with iterations_per_loop and host_call settings.
  - Uses `TPUInfeedOutfeedSessionWithEndOfStreamHandlingHook` when stopping signals enabled.
  - Builds embedding_config_spec if feature/table configs exist.
  - Returns TPUEstimator and total_replicas.
- `create_tpu_estimator_on_cpu`:
  - Creates TPUEstimator with `use_tpu=False` and small batch size; used for CPU test.
- `run()`:
  - Loads global step (or 0).
  - Instantiates task; if BaseEmbeddingTask, prepares feature/table configs.
  - Uses CPU test wrapper if enabled.
  - `train`: `est.train`.
  - `eval`: iterates checkpoints and evaluates; writes summaries and stops at max_steps.
  - `train_and_eval`: not supported (raises TypeError).
- `main`: logs FLAGS and task_param, runs runner.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/tpu_runner.rs`.
- Rust public API surface: runner struct + CLI entrypoint.
- Data model mapping: TPU/embedding configs likely require TF runtime bindings.
- Feature gating: `tf-runtime`, `tpu` features.

**Implementation Steps (Detailed)**
1. Port CLI flag set and task registry lookup.
2. Decide runtime strategy (native Rust vs TF bridge).
3. Preserve TPU version config and host_call/embedding config semantics if using TF.
4. Mirror evaluation-by-checkpoint loop and summary writing.

**Tests (Detailed)**
- Python tests: none specific.
- Rust tests: CLI parsing + control flow tests; TPU behavior likely gated/skipped in CI.

**Gaps / Notes**
- Full parity depends on TensorFlow TPU Estimator which is not available natively in Rust.

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

### `monolith/utils.py`
<a id="monolith-utils-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 81
- Purpose/role: Small utility helpers for TF monkey patching and recursive file copy with tf.io.gfile.
- Key symbols/classes/functions: `enable_monkey_patch`, `CopyFile`, `CopyRecursively`.
- External dependencies: `tensorflow`, `ThreadPoolExecutor`.
- Side effects: modifies `tensorflow.python.training.monitored_session` module attribute; copies files (possibly remote).

**Required Behavior (Detailed)**
- `enable_monkey_patch()`:
  - Imports `tensorflow.python.training.monitored_session` and sets `_PREEMPTION_ERRORS` to `(tf.errors.AbortedError,)`.
- `CopyFile(src, dst, overwrite=True, skip_nonexist=True, max_retries=5)`:
  - Uses `tf.io.gfile.copy` and retries on NotFoundError (skip if `skip_nonexist`).
- `CopyRecursively(src, dst, max_workers=1, skip_nonexist=True, max_retries=5)`:
  - Recursively copies a directory tree via `tf.io.gfile`.
  - If `src` missing and `skip_nonexist`, returns; else raises ValueError.
  - If `dst` exists, removes it then recreates.
  - If `max_workers > 1`, uses ThreadPoolExecutor to copy files in parallel.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-tf/src/utils.rs` (TF-dependent utilities) plus `monolith-core/src/path_utils.rs` for re-exports.
- Rust public API surface: equivalents for monkey patch (if bridging TF) and recursive copy.

**Implementation Steps (Detailed)**
1. Provide TF monkey patch equivalent when using Python/TF runtime; otherwise document unsupported.
2. Implement recursive copy with retries and optional parallelism.
3. Ensure semantics match tf.io.gfile for remote paths (HDFS/GCS) where needed.

**Tests (Detailed)**
- Python tests: `monolith/utils_test.py`
- Rust tests: port tests for `find_main`, `get_libops_path`, and CopyRecursively.

**Gaps / Notes**
- Full parity requires tf.io.gfile support; Rust may need a virtual filesystem abstraction.

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

### `monolith/utils_test.py`
<a id="monolith-utils-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 65
- Purpose/role: Tests path utilities, monkey patch, and recursive copy.
- Key symbols/classes/functions: `UtilsTest` test cases.
- External dependencies: `tensorflow`, `monitored_session`.
- Side effects: writes temp files in `/tmp`.

**Required Behavior (Detailed)**
- `testFindMain`: `utils.find_main()` base dir last path component is `__main__` (Bazel layout assumption).
- `testGetLibopsPath`: `utils.get_libops_path("monolith/utils_test.py")` exists.
- `testLoadMonitoredSession`: `_PREEMPTION_ERRORS` equals `(errors.AbortedError,)` after monkey patch.
- `testMultiThreadedCopy`: creates nested dirs/files and verifies copied content with `CopyRecursively(max_workers=2)`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-tf/tests/utils.rs`.
- Rust public API surface: path utils + recursive copy.

**Implementation Steps (Detailed)**
1. Port tests with temp dirs and file content checks.
2. If TF monkey patch is unsupported, document and adjust tests accordingly.

**Tests (Detailed)**
- Python tests: this file
- Rust tests: parity tests for path and copy semantics.

**Gaps / Notes**
- `find_main` behavior is tied to Bazel execution layout; Rust tests may need harness adjustments.

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
