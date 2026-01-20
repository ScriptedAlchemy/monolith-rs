<!--
Source: task/request.md
Lines: 3914-4119 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/core/hyperparams.py`
<a id="monolith-core-hyperparams-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 439
- Purpose/role: Dynamic hyperparameter container with attribute + dotted-path access, nested parameter trees, immutability, stringification, and class instantiation support.
- Key symbols/classes/functions: `_is_named_tuple`, `_SortedDict`, `_Param`, `Params`, `InstantiableParams`, `copy_params_to`, `update_params`, `allowed_kwargs`.
- External dependencies: `copy`, `re`, `six`, `inspect.signature/Parameter`, `tensorflow` (Tensor detection for deepcopy).
- Side effects: error messages include similar-key hints; deepcopy of `tf.Tensor` intentionally keeps a reference (no deep copy).

**Required Behavior (Detailed)**
- `_is_named_tuple(x)`:
  - Returns true if `x` is a `tuple` and has `_fields` attribute.
- `_SortedDict.__repr__()`:
  - Always renders dict entries sorted by key.
- `_Param`:
  - Stores `name`, `value`, `description`.
  - `__eq__` compares only name + value (description ignored).
  - `__deepcopy__`: if value is a `tf.Tensor`, **do not copy** (keep ref); else deep-copy; memo updated.
  - `to_string(nested_depth)`:
    - For `Params`: delegate to nested `_to_string`.
    - For `dict`: produce `_SortedDict` with nested `GetRepr` on values.
    - For `list/tuple` (non-namedtuple): reconstruct same type from `GetRepr` values.
    - If value has `Repr` method, call it (used for function-like objects).
    - If value is `str`, wrap in double quotes.
    - Indent with 2 spaces per nesting level.
  - `set` stores value without copying.
  - `get` returns stored value.
- `Params`:
  - Internal storage: `_params` dict mapping name → `_Param`; `_immutable` bool.
  - `__setattr__` / `__setitem__`: if immutable -> `TypeError("This Params instance is immutable.")`. Otherwise set existing param; missing name -> `AttributeError` with `_key_error_string`.
  - `__getattr__` / `__getitem__`: missing name -> `AttributeError` with `_key_error_string`.
  - `__dir__` returns sorted param names. `__contains__`, `__len__` work on `_params`.
  - `__eq__`: only equal to another `Params` with equal `_params` (uses `_Param.__eq__`).
  - `__str__` / `_to_string`: emits multi-line block with params sorted by name; nested `Params` rendered recursively.
  - `_similar_keys(name)`:
    - Builds list of keys with >0.5 overlap of 3-char substrings from `name`.
  - `_key_error_string(name)`:
    - If similar keys exist: `"name (did you mean: [k1,k2])"` with keys sorted.
  - `define(name, default_value, description)`:
    - Reject if immutable → `TypeError("This Params instance is immutable.")`.
    - Assert `name` is `str` matching `^[a-z][a-z0-9_]*$` (raises `AssertionError`).
    - If already defined, `AttributeError("Parameter %s is already defined" % name)`.
  - `freeze()` sets `_immutable=True`; `is_immutable()` returns bool.
  - `_get_nested("a.b[0].c")`:
    - Supports dot navigation into nested `Params`.
    - Supports list indexing in a segment via `name[index]`.
    - If missing segment: `AttributeError` with partial path.
    - If intermediate is not `Params`: `AssertionError("Cannot introspect <type> for <path>")`.
  - `set(**kwargs)`:
    - If immutable -> `TypeError("This Params instance is immutable: %s" % self)`.
    - Dotted names traverse `_get_nested`; missing keys -> `AttributeError` with similar key hints.
    - Returns `self`.
  - `get(name)`:
    - Dotted names traverse `_get_nested`; missing -> `AttributeError` with similar key hints.
  - `delete(*names)`:
    - If immutable -> `TypeError("This Params instance is immutable.")`.
    - Dotted names traverse `_get_nested`; missing -> `AttributeError` with similar key hints.
    - Returns `self`.
  - `iter_params()` yields `(name, value)` for all params (dict iteration order).
  - `copy()` deep-copies `_params` and preserves `_immutable`.
- `copy_params_to(from_p, to_p, skip=None)`:
  - For each param in `from_p`, sets into `to_p`.
  - Nested `Params` are copied via `p.copy()`.
  - `skip` list omits names.
- `InstantiableParams(Params)`:
  - Defines param `cls`.
  - `instantiate()`:
    - If `cls.__init__` has exactly 2 parameters and `cls` has attribute `params` and `'params'` is a parameter: call `cls(self)`.
    - Otherwise, build inverted index of all leaf (non-Params) values; pass matching `__init__` parameter names (excluding `self`, `cls`, varargs/kwargs).
    - Additionally pass any `allowed_kwargs` present and not `None`.
- `update_params(params, args)`:
  - Recursively traverses params; if leaf key exists in `args`, sets and removes from `args`.
  - Unmatched keys remain in `args`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/src/hyperparams.rs`.
- Rust public API surface: `Params`, `ParamValue`, `InstantiableParams`, `copy_params_to`, `update_params`, `ParamsFactory`.
- Data model mapping:
  - Python `_Param` → Rust `Param` (name/value/description).
  - Python `Params` → Rust `Params` with `BTreeMap` storage.
  - Python raw values → Rust `ParamValue` (including `External` for TF tensors / opaque handles).
- Feature gating: none currently; TF handle storage likely lives in `ParamValue::External` (tie into `monolith-tf` if needed).
- Integration points: `base_layer.rs`, `base_task.rs`, `base_model_params.rs`, `model_registry.rs`.

**Implementation Steps (Detailed)**
1. Align Rust `Params` error messages with Python (TypeError/AttributeError strings).
2. Match `__str__` formatting (sorted keys, indent, list/dict rendering, quoting).
3. Implement `_similar_keys` overlap logic and error hints exactly.
4. Support list indexing in dotted paths with Python-compatible errors.
5. Preserve Tensor/opaque value deepcopy semantics (reference copy).
6. Implement `InstantiableParams.instantiate` reflection-style path or provide compatible factory shim.
7. Ensure `copy_params_to` surfaces missing keys (don’t silently drop errors).
8. Add tests mirroring `hyperparams_test.py` including exact error strings.

**Tests (Detailed)**
- Python tests: `monolith/core/hyperparams_test.py`.
- Rust tests: add `monolith-rs/crates/monolith-core/tests/hyperparams.rs`.
- Cross-language parity test: add fixture-driven JSON snapshot compare of `to_string`, errors, and nested access.

**Gaps / Notes**
- Rust currently raises `MonolithError::ConfigError` rather than `TypeError`/`AttributeError` equivalents; messages differ.
- Rust `Params` does not implement equality; Python `Params.__eq__` is relied on in tests.
- Rust `InstantiableParams` uses `ParamsFactory` (no reflection or `allowed_kwargs` handling).
- `copy_params_to` in Rust ignores `set` errors (should propagate to match Python exceptions).
- String rendering differs (Python uses sorted `_SortedDict` repr and special `Repr()` hook).
- Rust treats list navigation errors differently (`Expected list index` vs Python `Cannot introspect` assertion).

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

### `monolith/core/hyperparams_test.py`
<a id="monolith-core-hyperparams-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 277
- Purpose/role: Unit tests defining required semantics for `Params` behavior, error messages, and string rendering.
- Key symbols/classes/functions: `ParamsTest`, `TestEnum`.
- External dependencies: `unittest`, `tensorflow`, `absl.flags/logging`, `monolith.core.hyperparams`.
- Side effects: none (tests only).

**Required Behavior (Detailed)**
- `test_equals`:
  - `Params.__eq__` compares only name+value; nested `Params` must compare deeply; non-Params should not be equal.
- `test_deep_copy`:
  - `Params.copy()` deep-copies nested `Params`.
  - `tf.Tensor` values are **shared** (same object identity) in copied params.
- `test_copy_params_to`:
  - Copy respects `skip` list; `dest` only contains copied keys.
- `test_define_existing`:
  - Duplicate `define` raises `AttributeError` with "already defined".
- `test_legal_param_names`:
  - Invalid names (`None`, empty, `_foo`, `Foo`, `1foo`, `foo$`) raise `AssertionError`.
  - Valid examples: `foo_bar`, `foo9`.
- `test_set_and_get`:
  - Setting undefined param via `.set` or `setattr` raises `AttributeError` mentioning name.
  - `set/get` works; `delete` removes; subsequent access raises `AttributeError`.
- `test_set_and_get_nested_param`:
  - Dotted traversal with nested `Params` works for `get`, `set`, `delete`.
  - `set` on missing path raises `AttributeError` with dotted name.
  - Attempting to navigate into non-Params (e.g., dict) raises `AssertionError("Cannot introspect ...")`.
- `test_freeze`:
  - Frozen params reject `set`, `setattr`, `delete`, `define` with `TypeError`.
  - `get` on missing still raises `AttributeError`.
  - Nested params remain mutable after parent frozen.
  - Attempt to set `_immutable` attribute raises `TypeError`.
  - `copy()` of frozen params is still immutable.
- `test_to_string`:
  - `str(params)` yields exact multi-line formatting:
    - Sorted keys.
    - Strings quoted.
    - Dicts rendered with sorted keys and nested `Params` converted to dicts.
    - Lists render nested `Params` as dicts.
- `test_iter_params`:
  - Iteration yields all keys/values (order not asserted).
- `test_similar_keys`:
  - Error message for misspelled attribute includes sorted similar keys:
    - `actuvation (did you mean: [activation,activations])`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/src/hyperparams.rs`.
- Rust public API surface: `Params`, `ParamValue`, `InstantiableParams`, `copy_params_to`, `update_params`.
- Data model mapping: implement Rust tests mirroring Python assertions and error text.
- Feature gating: none.
- Integration points: Rust tests in `monolith-rs/crates/monolith-core/tests`.

**Implementation Steps (Detailed)**
1. Add Rust unit tests that mirror each Python test case.
2. Build helper to compare error strings for invalid operations.
3. Ensure `Params` equality is implemented for Rust tests.
4. Add string formatting parity checks using exact expected text.
5. Validate frozen behavior and nested mutability.

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
