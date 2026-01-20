<!--
Source: task/request.md
Lines: 7432-7674 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/data/training_instance/python/parser_utils.py`
<a id="monolith-native-training-data-training-instance-python-parser-utils-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 85
- Purpose/role: Utilities for parser pipelines, including queued extra-parse steps and ragged encoding expansion/contract helpers.
- Key symbols/classes/functions: `_extra_parse_steps`, `add_extra_parse_step`, `RaggedEncodingHelper.expand`, `RaggedEncodingHelper.contract`, `advanced_parse`.
- External dependencies: TensorFlow, `ragged_utils.fused_value_rowids`.
- Side effects: mutates global deque of extra parse steps; mutates RaggedTensor internal row partition caches during `contract`.

**Required Behavior (Detailed)**
- `_extra_parse_steps`:
  - Global `deque` used to store parse step callables.
- `add_extra_parse_step(parse_fn)`:
  - Appends parse_fn to `_extra_parse_steps`.
- `RaggedEncodingHelper.expand(name_to_ragged_ids, with_precomputed_nrows=True, with_precomputed_value_rowids=False)`:
  - For each RaggedTensor value, returns a dict with:
    - `values`, `row_splits`, optional `nrows` (if flag), optional `value_rowids` computed via `ragged_utils.fused_value_rowids` (if flag).
  - Non-ragged entries pass through unchanged.
- `RaggedEncodingHelper.contract(name_to_ragged_ids)`:
  - For dict entries with `values` and `row_splits`, rebuilds `tf.RaggedTensor.from_row_splits(..., validate=False)`.
  - If `nrows` present, asserts `_row_partition._nrows` is None before assigning.
  - If `value_rowids` present, asserts `_row_partition._value_rowids` is None before assigning.
  - Non-dict entries pass through unchanged.
- `advanced_parse(features)`:
  - Pops parse steps from `_extra_parse_steps` in FIFO order and applies each to `features`.
  - Returns final features dict.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/src` (parser utilities) + `monolith-tensor` for ragged.
- Rust public API surface: `add_extra_parse_step` and `advanced_parse` equivalents; ragged expand/contract helpers.
- Data model mapping: RaggedTensor internal encodings → Rust ragged structure with cached rowids/nrows.
- Feature gating: none.
- Integration points: `parse_instance_ops_test.py` uses `RaggedEncodingHelper`.

**Implementation Steps (Detailed)**
1. Implement a global queue of parse steps (with proper synchronization if used across threads).
2. Implement ragged expand/contract; ensure cached rowids/nrows are set only once.
3. Mirror `fused_value_rowids` behavior using Rust ragged utilities.

**Tests (Detailed)**
- Python tests: `parse_instance_ops_test.py` (`RaggedEncodingHelperTest`).
- Rust tests: add unit tests that expand, contract, and verify rowids/nrows caching.
- Cross-language parity test: compare ragged values and cached rowids against Python output.

**Gaps / Notes**
- Directly mutates internal ragged partition caches; Rust must provide an equivalent escape hatch.

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

### `monolith/native_training/data/training_instance/python/pb_datasource_ops.py`
<a id="monolith-native-training-data-training-instance-python-pb-datasource-ops-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 48
- Purpose/role: Thin wrappers around `gen_monolith_ops` for filtering and negative sampling on variant tensors in training_instance pipelines.
- Key symbols/classes/functions: `filter_by_fids`, `filter_by_value`, `negative_sample`, `variant_dummy`.
- External dependencies: TensorFlow, `gen_monolith_ops` custom kernels.
- Side effects: none beyond custom op invocation.

**Required Behavior (Detailed)**
- `filter_by_fids(variant, filter_fids, has_fids, select_fids, has_actions)`:
  - Passes list args (defaults to empty) to `pb_datasource_ops.set_filter`.
- `filter_by_value(variant, field_name, op, operand)`:
  - Calls `pb_datasource_ops.value_filter` with given field/op/operand.
- `negative_sample(variant, drop_rate, label_index, threshold)`:
  - Calls `pb_datasource_ops.negative_sample` with drop/threshold params.
- `variant_dummy(variant)`:
  - Calls `pb_datasource_ops.variant_dummy`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/src` (ops wrappers).
- Rust public API surface: minimal wrappers for filtering/negative sampling on variant streams.
- Data model mapping: variant tensor → Rust variant representation.
- Feature gating: custom op availability (TF backend).
- Integration points: training_instance datasets and tests.

**Implementation Steps (Detailed)**
1. Add Rust wrapper functions that call the underlying kernel backend.
2. Ensure default empty list behavior matches Python.
3. Expose in public API for dataset pipelines.

**Tests (Detailed)**
- Python tests: indirectly via `instance_negative_gen_dataset_op_test.py`.
- Rust tests: minimal unit tests for wrappers if backend available.
- Cross-language parity test: verify behavior using fixed fixtures.

**Gaps / Notes**
- This is a thin wrapper; underlying op semantics are defined in C++/TF kernels.

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

### `monolith/native_training/data/training_instance/python/test_data_utils.py`
<a id="monolith-native-training-data-training-instance-python-test-data-utils-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 15
- Purpose/role: Placeholder test utility module; currently only imports TensorFlow.
- Key symbols/classes/functions: none.
- External dependencies: TensorFlow.
- Side effects: none.

**Required Behavior (Detailed)**
- No runtime behavior beyond importing TensorFlow.

**Rust Mapping (Detailed)**
- Target crate/module: none.
- Rust public API surface: none.
- Data model mapping: none.
- Feature gating: none.
- Integration points: none.

**Implementation Steps (Detailed)**
1. No Rust port needed unless file is expanded in Python.

**Tests (Detailed)**
- Python tests: none.
- Rust tests: none.
- Cross-language parity test: none.

**Gaps / Notes**
- File is effectively empty; keep an eye on future changes.

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

### `monolith/native_training/data/transform/transforms.py`
<a id="monolith-native-training-data-transform-transforms-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 250
- Purpose/role: Declarative transform objects that serialize to `TransformConfig` proto for dataset transform pipelines (filters, label generation, logical composition).
- Key symbols/classes/functions: `Transform` (ABC), `Compose`, `FilterByFid`, `FilterByAction`, `FilterByLabel`, `FilterByValue`, `AddLabel`, `LogicalOr`.
- External dependencies: `transform_config_pb2`, `LineId` proto descriptor for field validation.
- Side effects: none; purely builds proto configs with validation asserts.

**Required Behavior (Detailed)**
- `Transform` abstract base:
  - `as_proto()` returns `transform_config_pb2.TransformConfig`.
  - `_is_leaf_node()` distinguishes leaf vs composite.
- `Compose(transforms)`:
  - Requires all items are `Transform` instances.
  - `as_proto` merges each transform’s proto into a single `TransformConfig` via `MergeFrom` in order.
  - `_is_leaf_node` returns False.
- `FilterByFid(has_fids, filter_fids, select_fids)`:
  - `as_proto` appends a `basic_config.filter_by_fid` entry with respective lists.
  - `_is_leaf_node` True.
- `FilterByAction(has_actions)`:
  - Adds `basic_config.filter_by_action.has_actions`.
  - `_is_leaf_node` True.
- `FilterByLabel(thresholds)`:
  - Adds `basic_config.filter_by_label.thresholds`.
  - `_is_leaf_node` True.
- `FilterByValue(field_name, op, operand, keep_empty=False)`:
  - Validates `op` in `{gt,ge,eq,lt,le,neq,between,in,not-in,all,any,diff,startswith,endswith}`.
  - Validates `field_name` exists in `LineId` descriptor; `operand` is not None.
  - Infers operand type based on field cpp_type and op:
    - Repeated fields (`field.has_options`): only `all/any/diff` allowed; only integer types; operand int or list of int.
    - Float/double: `between` uses list; otherwise single float.
    - Int types: `in/not-in/between` use list; otherwise single int.
    - String: operand is str or list of str; else `RuntimeError("params error!")`.
  - Stores `float_operand`, `int_operand`, `string_operand`, `keep_empty`.
  - `as_proto` fills `basic_config.filter_by_value` with operands and flags.
  - `_is_leaf_node` True.
- `AddLabel(config, negative_value, new_sample_rate)`:
  - Parses config `pos_actions:neg_actions:sample_rate` separated by `;` (skips empty parts).
  - Adds `basic_config.add_label` with negative value + new sample rate and a `task_label_config` entry per task.
  - `_is_leaf_node` True.
- `LogicalOr(x, y)`:
  - Requires both `x` and `y` are leaf nodes.
  - `as_proto` creates `logical_or_config` and copies `basic_config` from each side.
  - `_is_leaf_node` False.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/src` (transform config builders).
- Rust public API surface: `Transform` trait + concrete structs mirroring Python class names; `as_proto()` to `TransformConfig`.
- Data model mapping: transform structs → `transform_config_pb2::TransformConfig`.
- Feature gating: none; pure config serialization.
- Integration points: `TransformDataset` op uses serialized config (see `datasets.py`).

**Implementation Steps (Detailed)**
1. Create Rust transform trait with `as_proto` and `is_leaf` methods.
2. Implement concrete transforms with identical validation (asserts or Result errors).
3. Preserve `Compose` merge ordering and `LogicalOr` leaf-only requirement.
4. Implement `FilterByValue` operand parsing based on LineId descriptor in Rust.

**Tests (Detailed)**
- Python tests: `transforms_test.py`.
- Rust tests: unit tests for each transform’s proto encoding and validation.
- Cross-language parity test: serialize configs in Python and Rust and compare bytes.

**Gaps / Notes**
- `FilterByValue` uses `LineId` field metadata to infer types; Rust must mirror the same descriptor mapping.

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
