<!--
Source: task/request.md
Lines: 6901-7016 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/data/parsers.py`
<a id="monolith-native-training-data-parsers-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 782
- Purpose/role: Parsing utilities that turn Monolith Instance/Example/ExampleBatch variant tensors into feature dicts, plus sharding sparse fid helpers and parser context management.
- Key symbols/classes/functions: `ParserCtx`, `ShardingSparseFidsOpParams`, `ProtoType`, `parse_instances`, `parse_examples`, `parse_example_batch`, `parse_example_batch_list`, `sharding_sparse_fids`, `sharding_sparse_fids_with_context`, `_add_dense_features`, `_add_extra_features`, `_assemble`.
- External dependencies: TensorFlow, `LineId` proto, `FeatureConfigs` proto, `LabelConf` proto, `FeatureList`, `gen_monolith_ops` custom kernels, `logging_ops`, `native_task_context`, `FLAGS.dataset_use_dataservice`.
- Side effects: populates TF collections via `add_to_collections`; writes to global parser context; logs timing metrics (`logging_ops.emit_timer`); registers required feature names via `add_feature` when example-batch parsing.

**Required Behavior (Detailed)**
- `ParserCtx` (context manager):
  - Global `_default_parser_ctx` is used if none exists; `get_default_parser_ctx()` creates `ParserCtx(False)` once.
  - `enable_resource_constrained_roughsort` (class-level flag) injects `item_id` into `extra_features` when parsing instances.
  - `enable_fused_layout` toggles v2 parsing ops and sharded sparse fid handling.
  - `parser_type` is set to `'instance'`, `'example'`, or `'examplebatch'` by parse functions.
  - `sharding_sparse_fids_op_params` holds op configuration (see below) and drives `sharding_sparse_fids_with_context` behavior.
  - `set/get` store arbitrary per-parse context values (e.g., `batch_size`).
  - `sharding_sparse_fids_features_insert_to_features` injects nested dict values into `features` with `__sharding_sparse_fids__` prefix; supports two-level dicts only.
  - `sharding_sparse_fids_features_parse_from_features` reverses the prefixing and removes those keys from `features`.
- `ShardingSparseFidsOpParams` dataclass:
  - Fields: `num_ps`, `use_native_multi_hash_table`, `unique` (callable), `transfer_float16`, `sub_table_name_to_config`, `feature_configs`, `enable_gpu_emb`, `use_gpu`.
- `ProtoType.get_tf_type(proto_type)`:
  - Maps proto field types to tf dtypes: INT → `tf.int64`, FLOAT → `tf.float32`, STRING → `tf.string`.
  - Raises `Exception('proto_type {} is not support'.format(proto_type))` for unknown types.
- `_add_dense_features(names, shapes, types, dense_features, dense_feature_shapes, dense_feature_types)`:
  - Requires `dense_features` and `dense_feature_shapes` non-null, same length, shapes > 0.
  - Defaults `dense_feature_types` to `[tf.float32] * len(dense_features)` if None; otherwise lengths must match.
  - Appends to `names`, `shapes`, `types`.
- `_add_extra_features(names, shapes, types, extra_features, extra_feature_shapes)`:
  - Requires `extra_features` and shapes non-null, same length, shapes > 0.
  - Resolves dtype from `LineId` descriptor per field; raises `Exception(f"{name} is not in line id, pls check!")` if missing.
- `_assemble(sparse_features, names, shapes, types, out_list, batch_size)`:
  - For sparse features: takes `split = out_list[i]` (reshaped to `(batch_size+1,)` if batch_size provided) and `value = out_list[i + len(names)]`; returns `tf.RaggedTensor.from_row_splits`.
  - For dense features: uses `out_list[i]` directly.
  - Returns dict of feature name → tensor/ragged tensor.
- `parse_instances(tensor, fidv1_features, fidv2_features, dense_features, dense_feature_shapes, dense_feature_types, extra_features, extra_feature_shapes)`:
  - If `ParserCtx.enable_resource_constrained_roughsort` is True, ensures `item_id` is in `extra_features` with shape 1.
  - Validates dense feature inputs and defaults types to `tf.float32`.
  - Sets parser context type `'instance'` and writes multiple lists to TF collections + context (fidv1/fidv2/dense/extra, shapes/types).
  - Non-fused layout:
    - For `fidv1_features`: adds feature names from slots via `get_feature_name_and_slot`; if all entries are strings, resolves slots via `FeatureList.parse()` and raises `RuntimeError("fidv1_features error")` on failure.
    - Adds `fidv2_features` names; sets shapes to `-1` and types to `tf.int64`.
    - Asserts no duplicate names.
    - Calls `parse_instance_ops.parse_instances(...)` and `_assemble` with sparse features.
  - Fused layout:
    - If no names, injects `__FAKE_FEATURE__` with shape 1/float32.
    - Calls `parse_instances_v2` and `_assemble` (no sparse features list).
    - If `sharding_sparse_fids_op_params` present and (`use_gpu` or `FLAGS.dataset_use_dataservice`), calls `sharding_sparse_fids_with_context(instances, features, ctx)`.
    - Else stores `instances` under `__sharding_sparse_fids__sparse_features` key.
    - Removes `__FAKE_FEATURE__` before returning.
- `parse_examples(...)` and `parse_example_batch(...)`:
  - Same dense/extra validation pattern as `parse_instances`.
  - Sets parser context type `'example'` or `'examplebatch'` and stores config in TF collections.
  - If `is_example_batch()` is True, registers required features via `add_feature`: sparse features, dense features (adds `__LABEL__` for label), and `__LINE_ID__` for extra features.
  - Non-fused: names from sparse features, shapes `-1`, types `tf.int64`, then calls `parse_examples`/`parse_example_batch` and `_assemble` (batch_size from context for example_batch).
  - Fused: same `__FAKE_FEATURE__` fallback, uses `parse_examples_v2`/`parse_example_batch_v2`, then `sharding_sparse_fids_with_context` or stores under `__sharding_sparse_fids__sparse_features`.
- `sharding_sparse_fids(tensor, ps_num, feature_cfgs, unique, input_type, parallel_flag, fid_list_ret_list, version)`:
  - Normalizes `input_type` (`example_batch` → `examplebatch`).
  - Builds sorted `table_name_list` from `feature_cfgs.feature_configs[*].table`; `ps_num=1` if 0; `table_count = len(table_name_list) * ps_num`.
  - Uses `logging_ops.tensors_timestamp` around op call and emits timer `sharding_sparse_fids` with tag `model_name` from `native_task_context`.
  - Calls versioned custom op (`sharding_sparse_fids_v5/v4/v3/v2` or legacy) returning fid lists, row splits, offsets, and sizes.
  - Asserts list lengths for versions 5/4; returns either raw lists (if `fid_list_ret_list` or `version==4`) or dicts keyed by `table:ps_index` with row splits and row_split_size.
- `sharding_sparse_fids_with_context(sparse_features, features, parser_ctx)`:
  - Calls `sharding_sparse_fids` with params from `parser_ctx.sharding_sparse_fids_op_params`.
  - If `enable_gpu_emb`: inserts `shards_value`, `shards_row_lengths`, `shards_table_row_lengths`, offsets, `batch_size`, `fid_list_emb_row_lenth` into `features` using prefixed keys.
  - Else inserts `shards`, offsets, `batch_size`, size stats; if `use_native_multi_hash_table`, also inserts `shards_row_split` and `shards_row_split_size`.
- `parse_example_batch_list(tensor_list, label_config, positive_label, negative_label, names, shapes, dtypes, extra_features)`:
  - Optionally parses `label_config` (semicolon-separated tasks, each `pos_actions:neg_actions`) into `LabelConf`, and adds `label` feature with shape `len(tasks)`.
  - Marks `shapes[i] == -1` as sparse, appends `tf.int64` to `dtypes` for sparse values (to match op output list shape).
  - Calls `parse_example_batch_list` op with serialized label conf, then `_assemble`.
- Error semantics:
  - Extensive `assert` checks for list lengths/shape values, duplicates, and supported types; specific exceptions for invalid LineId fields and fidv1_features name mapping.
- Metrics/logging:
  - `sharding_sparse_fids` emits a timer metric named `sharding_sparse_fids` with model_name tag.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/src` (parsing), `monolith-rs/crates/monolith-proto` (LineId/FeatureConfigs), optional TF backend for custom ops.
- Rust public API surface: `parsers` module with `parse_instances`, `parse_examples`, `parse_example_batch`, and sharding helpers; `ParserCtx` analog for context state.
- Data model mapping: TF Variant/RaggedTensor → Rust datasets/feature maps; need ragged representation and feature registry.
- Feature gating: fused layout parsing, sharding_sparse_fids, and GPU embedding paths behind feature flags.
- Integration points: datasets (`datasets.py`), feature registry (`feature_list.py`), training pipelines expecting collections metadata.

**Implementation Steps (Detailed)**
1. Implement a Rust `ParserCtx` with context manager semantics (scoped override) and a global default.
2. Port `_add_dense_features`, `_add_extra_features`, and `_assemble` with equivalent validation and ragged construction.
3. Implement `parse_instances`/`parse_examples`/`parse_example_batch` in Rust, honoring `enable_fused_layout` and `enable_resource_constrained_roughsort` behavior.
4. Provide `FeatureList` lookups for fidv1 slot-name mapping and raise equivalent errors on failure.
5. Persist metadata to a Rust collection registry mirroring `add_to_collections` semantics.
6. Implement sharding_sparse_fids and sharding_sparse_fids_with_context around native kernels or TF runtime bindings; preserve timing metric emission.
7. Implement parse_example_batch_list with label_config parsing and label feature insertion.
8. Add tests for parsing shape/type inference, ragged assembly, and sharding outputs using small fixture tensors.

**Tests (Detailed)**
- Python tests: `data_ops_test.py`, `parse_sparse_feature_test.py`, `feature_utils_test.py`, `tf_example_to_example_test.py` (parsing paths).
- Rust tests: parser unit tests for each parse_* function; sharding_sparse_fids smoke tests (if backend available).
- Cross-language parity test: parse the same fixture files and compare feature dict keys, shapes, and ragged values.

**Gaps / Notes**
- Fused layout paths depend on custom ops (`parse_*_v2` and `sharding_sparse_fids_*`); must be backed by TF runtime or re-implemented.
- `parse_example_batch_list` mutates dtypes length to match op outputs; replicate this behavior exactly.

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
