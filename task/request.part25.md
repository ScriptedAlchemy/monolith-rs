<!--
Source: task/request.md
Lines: 6306-6464 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/data/feature_utils.py`
<a id="monolith-native-training-data-feature-utils-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 1070
- Purpose/role: Feature/label filtering, transformation, and utility ops over variant tensors; mostly thin wrappers around custom `gen_monolith_ops` kernels with strict input validation and feature-registration side effects.
- Key symbols/classes/functions: `filter_by_fids`, `filter_by_feature_value`, `filter_by_value`, `add_action`, `add_label`, `scatter_label`, `filter_by_label`, `special_strategy`, `negative_sample`, `feature_combine`, `switch_slot`, `switch_slot_batch`, `label_upper_bound`, `label_normalization`, `use_field_as_label`, `create_item_pool`, `item_pool_random_fill`, `item_pool_check`, `save_item_pool`, `restore_item_pool`, `fill_multi_rank_output`, `use_f100_multi_head`, `map_id`, `multi_label_gen`, `string_to_variant`, `string_to_variant_with_transform`, `variant_to_zeros`, `kafka_resource_init`, `kafka_read_next`, `kafka_read_next_v2`, `has_variant`, `gen_fid_mask`, `tf_example_to_example`.
- External dependencies: TensorFlow, numpy, `idl.matrix.proto.line_id_pb2.LineId`, `data_op_config_pb2.LabelConf`/`TFRecordFeatureDescription`, `gen_monolith_ops` custom kernels.
- Side effects: calls `add_feature`/`add_feature_by_fids` to ensure downstream parsing includes required fields; validates/loads operand files via `tf.io.gfile.exists`; asserts on invalid inputs; builds TF ops that mutate or filter variant tensors.

**Required Behavior (Detailed)**
- `filter_by_fids(variant, filter_fids, has_fids, select_fids, has_actions, req_time_min, select_slots, variant_type)`:
  - Coerces `filter_fids`/`has_fids`/`select_fids` to `np.uint64` then `int64` list; defaults to empty lists.
  - `select_slots` defaults to empty; asserts all slots > 0.
  - If `variant_type != 'instance'`, calls `add_feature_by_fids` for all fid lists.
  - Calls `ragged_data_ops.set_filter(...)` with `has_actions or []`, `req_time_min`, `select_slots`, `variant_type` and returns variant tensor.
- `filter_by_feature_value(variant, field_name, op, operand, field_type, keep_empty, operand_filepath)`:
  - `op` must be in `{gt,ge,eq,lt,le,neq,between,in,not-in,all,any,diff,startswith,endswith}`.
  - Exactly one of `operand` or `operand_filepath` is provided; if filepath set, it must exist and `op` must be in `{in, not-in}`.
  - `field_type` must be in `{int64,float,double,bytes}`; builds `int_operand`, `float_operand`, `string_operand` based on type/op:
    - `all/any/diff` only for `int64` (operand int or list of int).
    - `between` uses a list of numbers (float/double) or ints for int64.
    - `bytes` accepts str or list of str; otherwise raises `RuntimeError("params error!")`.
  - Calls `ragged_data_ops.feature_value_filter(...)` with operands, file path, `keep_empty`, returns variant.
- `filter_by_value(variant, field_name, op, operand, variant_type, keep_empty, operand_filepath)`:
  - For `variant_type != 'instance'`, calls `add_feature('__LINE_ID__')`.
  - `field_name` must exist in `LineId` descriptor; uses proto field `cpp_type`/`has_options` to determine parsing rules.
  - Same operand vs operand_filepath exclusivity; operand file must exist; only `in/not-in` supported with filepath.
  - For repeated fields (`field.has_options`), only `all/any/diff` allowed and only integer types.
  - For `string` fields: operand must be str or list of str, else `RuntimeError("params error!")`.
  - Calls `ragged_data_ops.value_filter(...)` with `variant_type` and returns variant.
- `add_action(variant, field_name, op, operand, action, variant_type)`:
  - For `variant_type != 'instance'`, calls `add_feature('__LINE_ID__')`.
  - `op` in `{gt,ge,eq,lt,le,neq,between,in}`; field must exist in `LineId`.
  - Builds typed operands (float/int/string) based on field cpp_type; for `in/between` on integer types, operand is list of int.
  - Calls `ragged_data_ops.add_action(..., actions=[action], variant_type)`.
- `add_label(variant, config, negative_value, new_sample_rate, variant_type)`:
  - For `variant_type != 'instance'`, calls `add_feature('__LINE_ID__')`.
  - `config` is required; `new_sample_rate` must be in `(0, 1.0]`.
  - Parses `config` with `;` task separator; each task `pos_actions:neg_actions:sample_rate` (empty lists allowed). Skips empty trailing parts.
  - Builds `LabelConf` proto and calls `ragged_data_ops.add_label(..., negative_value, sample_rate=new_sample_rate)`.
- `scatter_label(variant, config, variant_type)`:
  - For `variant_type != 'instance'`, calls `add_feature('__LABEL__')` and `add_feature('__LINE_ID__')`.
  - `config` required; passes through to `ragged_data_ops.scatter_label`.
- `filter_by_label(variant, label_threshold, filter_equal, variant_type)`:
  - For `variant_type != 'instance'`, calls `add_feature('__LABEL__')`.
  - `label_threshold` must be non-empty list.
  - Calls `ragged_data_ops.filter_by_label(..., filter_equal, variant_type)` and returns boolean tensor.
- `special_strategy(variant, strategy_list, strategy_conf, variant_type, keep_empty_strategy)`:
  - For `variant_type != 'instance'`, calls `add_feature('__LABEL__')` and `add_feature('__LINE_ID__')`.
  - `strategy_conf` is optional; parses comma-separated `strategy:sample_rate` or `strategy:sample_rate:label` entries.
  - Ensures lengths consistent and each `sample_rate` in `[0,1]`.
  - Calls `ragged_data_ops.special_strategy(..., keep_empty_strategy, variant_type)`.
- `negative_sample(variant, drop_rate, label_index, threshold, variant_type, action_priority, per_action_drop_rate)`:
  - For `variant_type != 'instance'`, calls `add_feature('__LABEL__')`.
  - `action_priority` and `per_action_drop_rate` are optional strings; if both set, parse lists of actions and per-action drop rates.
  - Calls `ragged_data_ops.negative_sample(..., priorities, actions, per_action_drop_rate)`.
- `feature_combine(src1, src2, slot)`:
  - Requires `tf.RaggedTensor` inputs; calls `ragged_data_ops.feature_combine(..., fid_version=2)`.
  - If `splits[0]` is `float32`, uses `from_row_splits(values, splits[1])`; else `from_nested_row_splits`.
- `switch_slot(ragged, slot)`:
  - Requires `tf.RaggedTensor`; calls `ragged_data_ops.switch_slot(..., fid_version=2)`.
  - If `splits[0]` is `float32`, returns new ragged from row_splits; else returns `ragged.with_flat_values(values)`.
- `switch_slot_batch(variant, features, variant_type, suffix)`:
  - `features` maps feature name → `(inplace, new_slot)`; `variant_type` must be `example` or `example_batch`.
  - Builds `features`, `inplaces`, `slots` arrays; calls `ragged_data_ops.switch_slot_batch(..., suffix)`.
- `label_upper_bound(variant, label_upper_bounds, variant_type)`:
  - `label_upper_bounds` non-empty; calls `ragged_data_ops.label_upper_bound`.
- `label_normalization(variant, norm_methods, norm_values, variant_type)`:
  - `norm_methods` length must equal `norm_values`; calls `ragged_data_ops.label_normalization`.
- `use_field_as_label(variant, field_name, overwrite_invalid_value, label_threshold, variant_type)`:
  - Calls `ragged_data_ops.use_field_as_label` to overwrite labels from LineId field with optional clamping.
- Item pool ops:
  - `create_item_pool(start_num, max_item_num_per_channel, container, shared_name)` asserts `start_num >= 0`, `max_item_num_per_channel > 0`, calls `ItemPoolCreate`.
  - `item_pool_random_fill`, `item_pool_check(model_path, global_step, nshards, buffer_size)`, `save_item_pool`, `restore_item_pool` delegate to custom ops.
- `fill_multi_rank_output(variant, enable_draw_as_rank, enable_chnid_as_rank, enable_lineid_rank_as_rank, rank_num, variant_type)`:
  - For `variant_type != 'instance'`, calls `add_feature('__LINE_ID__')`.
  - Calls `ragged_data_ops.fill_multi_rank_output`.
- `use_f100_multi_head(variant, variant_type)`:
  - Pass-through to `ragged_data_ops.use_f100_multi_head`.
- `map_id(tensor, map_dict, default)`:
  - `map_dict` non-empty; passes `from_value`, `to_value`, `default` to `ragged_data_ops.MapId`.
- `multi_label_gen(variant, head_to_index, head_field, pos_actions, neg_actions, use_origin_label, pos_label, neg_label, action_priority, task_num, variant_type)`:
  - Builds `head_to_index` string (`head:idx`), computes `task_num` if unset; asserts `max_idx < task_num`.
  - If `use_origin_label`, `pos_actions` and `neg_actions` must be empty; otherwise `pos_actions` non-empty.
  - `head_field` must exist in `LineId` descriptor and be int or string.
  - Calls `ragged_data_ops.multi_label_gen(..., action_priority, pos/neg actions, labels, variant_type)`.
- `string_to_variant(...)`:
  - `variant_type` must be `instance|example|examplebatch|example_batch`; converts string tensor into variant using header flags and optional `chnids/datasources`.
- `string_to_variant_with_transform(...)`:
  - Similar to `string_to_variant` but accepts `input_type` and `output_type` for on-the-fly transforms.
- `variant_to_zeros(tensor)`:
  - Calls `ragged_data_ops.variant_to_zeros` to produce zeroed variant tensor.
- Kafka ops:
  - `kafka_resource_init(topics, metadata, input_pb_type, output_pb_type, has_sort_id, lagrangex_header, kafka_dump_prefix, kafka_dump, container, shared_name)` calls `KafkaGroupReadableInit`.
  - `kafka_read_next`/`kafka_read_next_v2` call `KafkaGroupReadableNext`/`NextV2` with poll/stream timeouts.
- `has_variant(input, variant_type)`:
  - Calls `ragged_data_ops.HasVariant`.
- `gen_fid_mask(ragged, fid)`:
  - Casts `fid` to `np.uint64` → `int64`; calls `monolith_gen_fid_mask` with row_splits and flat_values.
- `tf_example_to_example(serialized, sparse_features, dense_features, label, instance_weight)`:
  - Defaults: empty sparse/dense/label/instance_weight if None.
  - Validates no overlaps between sparse/dense/label/instance_weight; slot ids unique; each slot id in `[1, 32768)`.
  - Builds `TFRecordFeatureDescription` proto and calls `MonolithTFExampleToExample` op.
- Error semantics:
  - Many checks are `assert` (raising `AssertionError`), some raise `RuntimeError("params error!")` for invalid bytes operands.
  - Operand file must exist and be used only with `in/not-in` ops; callers rely on these preconditions.
- I/O formats:
  - Variant tensor format is custom monolith variant; string inputs for `string_to_variant*` are framed by headers (sort header or lagrangex header) and length-prefixed protos.
  - Operand file for `operand_filepath` is expected to contain serialized `example_pb2.FilterValues` (see tests).
  - `tf_example_to_example` expects TF Example serialized bytes and emits Monolith Example variant.
- Threading/concurrency:
  - No explicit threading here; concurrency behavior is inside custom ops (e.g., Kafka resources).
- Determinism/perf:
  - Performance relies on custom ops; callers expect these to be safe in `tf.data` pipelines (including parallel map/filter). Determinism depends on op implementations; keep semantics stable.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/src` for dataset/feature ops; optionally `monolith-rs/crates/monolith-tf` for TF-runtime-backed kernels.
- Rust public API surface: `feature_utils` module exposing the same function set (or a `FeatureOps` trait with backend-specific implementations).
- Data model mapping: TF Variant / RaggedTensor → Rust `Variant`/`Ragged` equivalents (likely in `monolith-data` or `monolith-tensor`).
- Feature gating: Kafka ops, TFExample conversion, item-pool ops, and label/negative sampling depend on custom kernels; gate behind a TF backend or feature flags.
- Integration points: parsing (`parsers.py`), datasets (`datasets.py`), hooks (`item_pool_hook.py`), and training pipelines/tests.

**Implementation Steps (Detailed)**
1. Enumerate all custom ops used here and decide per-op strategy: native Rust implementation vs TF runtime binding.
2. Port validation logic exactly (asserts, `RuntimeError("params error!")`, slot range checks).
3. Provide Rust equivalents for `LineId` field metadata (from `monolith-proto`) and reuse it for `filter_by_value`/`add_action`/`multi_label_gen`.
4. Implement operand file reading for `in/not-in` filters using the same `FilterValues` proto.
5. Implement Ragged feature transforms (`feature_combine`, `switch_slot`, `switch_slot_batch`) with fid-v2 rules.
6. Add feature registry side effects equivalent to `add_feature`/`add_feature_by_fids` so parsing includes required fields.
7. Implement item pool ops or wrap TF kernels; include save/restore/check.
8. Implement label ops (`add_label`, `scatter_label`, `filter_by_label`, `label_upper_bound`, `label_normalization`, `use_field_as_label`, `multi_label_gen`) with identical label-invalid sentinel values.
9. Implement `string_to_variant` framing rules (headers, length prefix, flags, chnids/datasources) and `tf_example_to_example` conversion.
10. Add Kafka resource wrappers with poll/stream timeouts and variant conversion.
11. Add Rust tests mirroring Python expectations (see `feature_utils_test.py`) and cross-language fixtures.

**Tests (Detailed)**
- Python tests: `monolith/native_training/data/feature_utils_test.py`, `data_ops_test.py`, `eager_mode_test.py` (feature_combine/switch_slot usage).
- Rust tests: `monolith-rs/crates/monolith-data/tests/feature_utils_*` for filtering, labels, switching slots, map_id, fid mask, string_to_variant, TFExample conversion.
- Cross-language parity test: run Python test fixtures and compare Rust outputs on identical serialized inputs (including FilterValues operand files).

**Gaps / Notes**
- Heavy reliance on `gen_monolith_ops`; Rust must either re-implement kernels or use TF runtime backend (optional per earlier requirement).
- `filter_by_value` and `filter_by_feature_value` behavior is used in parallel dataset filters; caching/parallel safety must match TF op semantics.

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
