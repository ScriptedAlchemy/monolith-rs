<!--
Source: task/request.md
Lines: 18392-18651 (1-based, inclusive)
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
