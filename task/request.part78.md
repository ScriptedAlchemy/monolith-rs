<!--
Source: task/request.md
Lines: 17901-18204 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/native_model.py`
<a id="monolith-native-training-native-model-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 1109
- Purpose/role: Core TF-native model base classes and helpers for loss/prediction, embedding slice management, device placement, file output, metrics (AUC/MSE + deep insight), and export hooks.
- Key symbols/classes/functions:
  - `get_sigmoid_loss_and_pred`, `get_softmax_loss_and_pred`
  - `DeviceCtxType`, `MonolithDeviceCtx`
  - `MonolithBaseModel`, `MonolithModel`
  - Key methods: `create_model_fn`, `create_input_fn`, `create_serving_input_receiver_fn`,
    `create_embedding_feature_column`, `lookup_embedding_slice`, `share_slot`, `add_extra_output`, `add_training_hook`
- External dependencies: TensorFlow (graph/Estimator APIs, metrics, summaries, SavedModel), `absl.flags/logging`,
  monolith internal modules (`feature`, `feature_utils`, `file_ops`, `metric_utils`, `export_context`,
  `layers.LogitCorrection`, `dump_utils`, `device_utils`, `distribution_utils`, `embedding_combiners`, etc.),
  `OutConfig/OutType/TensorShape` proto from `idl.matrix.proto.example_pb2`.
- Side effects:
  - Writes prediction/eval outputs to per-worker files.
  - Writes item embedding cache table files in resource-constrained roughsort predict mode.
  - Mutates TF graph collections and graph-attached lists (`__losses`, `__training_hooks`, `__export_outputs`).
  - Registers slots and switches feature slots (`register_slots`, `switch_slot`, `switch_slot_batch`).
  - Adds TF summary scalars and `tf.print` ops for metrics.
  - Adds custom hooks for Kafka/file metrics.
  - Enables TOB env (`enable_tob_env`) on init.

**Required Behavior (Detailed)**
- **`get_sigmoid_loss_and_pred(name, logits, label, batch_size, sample_rate=1.0, sample_bias=False, mode=TRAIN, instance_weight=None, mask=None, logit_clip_threshold=None, predict_before_correction=True)`**
  - Reshapes `logits` to `(-1,)` and **overrides** `batch_size` using `dim_size(logits, 0)` regardless of the caller-supplied `batch_size`.
  - If `mode != PREDICT`:
    - `sample_rate` handling:
      - If `float`, fill tensor of shape `(batch_size,)`.
      - If `None`, fill tensor with `1.0`.
    - Instantiate `LogitCorrection(activation=None, sample_bias=sample_bias, name='sample_rate_correction')`.
    - Compute `logits_biased = src((logits, sample_rate))`.
    - `pred` is sigmoid of **raw** `logits` when `predict_before_correction=True`, else sigmoid of `logits_biased`.
    - If `logit_clip_threshold` set:
      - Assert `0 < logit_clip_threshold < 1`.
      - Compute `threshold = log((1 - p) / p)` and clip `logits_biased` to `[-threshold, threshold]`.
    - Compute `losses = sigmoid_cross_entropy_with_logits(labels=label.reshape(-1), logits=logits_biased)`.
    - If `instance_weight` present, reshape to `(-1,)`.
    - If `mask` present, reshape to `(-1,)` and `boolean_mask` both `losses` and `instance_weight` (if present).
    - If `instance_weight` present, multiply `losses *= instance_weight`.
    - Final `loss = reduce_sum(losses)`.
  - If `mode == PREDICT`: `loss=None`, `pred = sigmoid(logits)` (no correction and no clipping).
  - Returns `(loss, pred)` with op names using `{name}_sigmoid_*`.
- **`get_softmax_loss_and_pred(name, logits, label, mode)`**
  - `pred = argmax(softmax(logits, name='{name}_softmax_pred'), axis=1)`.
  - If `mode != PREDICT`, `loss = softmax_cross_entropy_with_logits(labels=label, logits=logits, name='{name}_softmax_loss')`.
  - Else `loss=None`. Returns `(loss, pred)`.
- **`DeviceCtxType`**
  - Constants: `INPUT_FN`, `MODEL_FN`, `INPUT_RECEIVER_FN`, `OTHERS`.
  - `all_types()` returns set of all constants.
- **`MonolithDeviceCtx(ctx_type)`**
  - Context manager for device placement; asserts `ctx_type` in `DeviceCtxType.all_types()`.
  - `__enter__`:
    - No-op if `enable_sync_training()` is false or `export_context.is_exporting()` is true.
    - Selects device function:
      - `INPUT_FN` → `input_device_fn`
      - `MODEL_FN` → `model_device_fn`
      - `INPUT_RECEIVER_FN` → `serving_input_device_fn`
      - Otherwise no-op.
    - Calls `tf.compat.v1.device(self._device_fn)` and enters it.
  - `__exit__`:
    - If `ctx_type == MODEL_FN`, calls `ensure_variables_in_device()` before exiting device scope.
    - Resets `_current` and `_device_fn`.
  - `ensure_variables_in_device()`:
    - Iterates `graph.get_operations()` and for ops whose name starts with `global_step`, calls `graph._apply_device_functions(op)` (private TF API).
- **`MonolithBaseModel(NativeTask, ABC)`**
  - `params()` defines:
    - `output_path`, `output_fields`, `delimiter` (default `\t`), `file_name`,
      `enable_grads_and_vars_summary`, `dense_weight_decay`, `clip_norm` (default `1000.0`),
      `sparse_norm_warmup_steps`, `default_occurrence_threshold`.
  - `__init__`:
    - Calls `enable_tob_env()`.
    - Initializes dicts: `fs_dict`, `fc_dict`, `slice_dict`, `_layout_dict`, `_occurrence_threshold`, `_share_slot_mapping`.
    - `_use_dense_allreduce = FLAGS.enable_sync_training`.
  - `__getattr__`:
    - For attributes in `self.p`, returns param value.
    - Special case `batch_size`: returns `eval.per_replica_batch_size` if `p.mode == EVAL`, else `train.per_replica_batch_size`.
    - Falls back to property getters or base `__getattr__`.
  - `__setattr__`:
    - If `self.p` has attr, sets there.
    - Special case `batch_size`: sets both `train.per_replica_batch_size` and `eval.per_replica_batch_size`.
  - `__deepcopy__`:
    - Deep-copies all attributes except `dump_utils` (shared reference).
  - `_get_file_ops(features, pred)`:
    - Requires `p.output_fields` set.
    - Builds `output_path = p.output_path/part-{worker_index:05d}`; opens `file_ops.WritableFile`.
    - `op_fields` is `features[field]` for each `output_fields` (comma-separated).
    - Appends predictions:
      - list/tuple → extend
      - dict → sorted by key before extending
      - scalar → append
    - Attempts to `tf.squeeze` tensors where rank > 1 and last dim == 1.
    - Formats each row using `tf.strings.format` with delimiter-joined `{}` and `summarize=-1`.
    - Uses `tf.map_fn` with `fn_output_signature=tf.string` and `tf.stop_gradient`.
    - Returns `(op_file, write_op)` where `write_op = op_file.append(tf.strings.reduce_join(result))`.
  - `_dump_item_embedding_ops(features)`:
    - Assumes instance is `DeepRoughSortBaseModel` and features contain `item_id`, `item_bias`, `item_vec`.
    - Writes `MonolithHashTable_cached_item_embeddings-00000-of-00001` under `_cal_item_cache_table_path()`.
    - Uses `WritableFile.append_entry_dump` to write.
  - `_get_real_mode(mode)`:
    - If `mode == PREDICT`, returns `PREDICT`.
    - If `mode == TRAIN`, returns `self.mode` (not necessarily `TRAIN`).
    - Otherwise raises `ValueError('model error!')`.
  - `is_fused_layout()` returns `ctx.layout_factory is not None`.
  - `instantiate()` returns `self` (no cloning).
  - `add_loss(losses)` appends one or many losses to graph-level `__losses`.
  - `losses` property:
    - Stored on `tf.compat.v1.get_default_graph()` as `__losses` list.
  - `_global_step` property:
    - Inside `maybe_device_if_allowed('/device:GPU:0')`, returns `tf.compat.v1.train.get_or_create_global_step()`.
  - `_training_hooks` property:
    - Stored on graph as `__training_hooks` list.
  - `clean()` clears feature-slot caches: `fs_dict`, `fc_dict`, `slice_dict`, `_occurrence_threshold`.
  - `create_input_fn()`:
    - Returns closure that wraps `self.input_fn(mode)` in `MonolithDeviceCtx(INPUT_FN)`.
  - `create_model_fn()`:
    - Resets caches via `clean()`.
    - Defines `model_fn_internal(features, mode, config)`:
      - `global_step = _global_step`, `real_mode = _get_real_mode(mode)`.
      - Runs `self.model_fn(features, real_mode)` inside `MonolithDeviceCtx(MODEL_FN)`.
      - Accepts either `EstimatorSpec` or `(label, loss, pred)` tuple/list:
        - `EstimatorSpec`: extract `label`, `loss`, `pred`, optional `head_name`, `classification`.
          If `pred` is dict, `head_name` becomes keys (in insertion order).
        - Tuple/list: `label, loss, pred`; if `pred` dict, `head_name` from keys; else `head_name` from
          `self.metrics.deep_insight_target`; sets `is_classification=True` and emits a warning.
        - Otherwise raises `Exception("EstimatorSpec Error!")`.
      - Validates `head_name`, `label`, `pred` shapes and alignment.
      - Normalizes `label` to `tf.identity` (name `label_{_node_name(...)}` or dict keys).
      - Calls `dump_utils.add_model_fn(self, mode, features, label, loss, pred, head_name, is_classification)`.
      - Adds auxiliary losses: `loss += tf.add_n(self.losses)` when present.
      - **Resource-constrained roughsort predict path**:
        - When not exporting, `real_mode == PREDICT`, and `FLAGS.enable_resource_constrained_roughsort`,
          and `self` is `DeepRoughSortBaseModel`, it writes item cache table and returns
          `EstimatorSpec(PREDICT, loss=1, train_op=no_op, training_hooks=[FileCloseHook]+_training_hooks, predictions=identity(...))`.
      - **Predict path**:
        - `predictions = dict(zip(head_name, pred))` for list/tuple, else `pred`.
        - If exporting or `p.output_path` is `None`: returns `EstimatorSpec(PREDICT, predictions=..., training_hooks=_training_hooks)`.
        - Else writes per-worker file using `_get_file_ops` and wraps `predictions` with `tf.identity` under control deps.
        - If exporting and `_export_outputs` populated, merges via `spec._replace(export_outputs=...)`.
      - **Metrics accumulation for train/eval**:
        - Builds `targets`, `labels_list`, `preds_list` aligned with heads.
        - If `FLAGS.disable_native_metrics` is false:
          - Classification → `tf.compat.v1.metrics.auc`; regression → `tf.compat.v1.metrics.mean_squared_error`.
          - Adds `tf.print` of metric value to stderr and `tf.compat.v1.summary.scalar`.
          - Adds update op to `train_ops`.
      - **Deep insight metrics**:
        - If any of `metrics.enable_kafka_metrics`, `enable_file_metrics`, `enable_deep_insight` and
          `metrics.deep_insight_sample_ratio > 0`:
          - Calls `metric_utils.write_deep_insight` with features, labels, preds, model_name, target, etc.
          - Optionally uses `dump_filename = f\"{dump_filename}.part-{worker_index:05d}\"`.
          - Adds op to collection `"deep_insight_op"`.
          - Adds `KafkaMetricHook` or `FileMetricHook` (only one of each type allowed; see `add_training_hook`).
      - **Eval path**:
        - If exporting or no `output_path`: returns `EstimatorSpec(mode, loss=loss, train_op=tf.group(train_ops), training_hooks=_training_hooks)`,
          with `pred`/`preds` added into `train_ops`.
        - Else writes outputs to file (same as predict) and returns EstimatorSpec with close hook.
      - **Train path**:
        - Determines `dense_optimizer` from `local_spec.optimizer` or `self._default_dense_optimizer`, else raises `Exception("dense_optimizer not found!")`.
        - Calls `dump_utils.add_optimizer(dense_optimizer)`.
        - Adds `feature_utils.apply_gradients_with_var_optimizer` to `train_ops` with:
          - `clip_type=ClipByGlobalNorm`, `clip_norm=self.clip_norm`, `dense_weight_decay=self.dense_weight_decay`,
            `global_step=_global_step`, `grads_and_vars_summary`, `sparse_norm_warmup_steps`,
            `is_fused_layout`, `use_allreduce=_use_dense_allreduce`.
        - Calls `add_batch_norm_into_update_ops()` then groups `UPDATE_OPS` and returns EstimatorSpec with `train_op=tf.group(train_ops)`.
  - `create_serving_input_receiver_fn()`:
    - Wraps `self.serving_input_receiver_fn()` in `MonolithDeviceCtx(INPUT_RECEIVER_FN)` and
      passes through `dump_utils.record_receiver`.
  - Abstract methods:
    - `input_fn(mode) -> DatasetV2`, `model_fn(features, mode)`, `serving_input_receiver_fn() -> ServingInputReceiver`.
  - `_export_outputs` property:
    - Graph-attached dict `__export_outputs` (created lazily).
  - `add_extra_output(name, outputs, head_name=None, head_type=None)`:
    - Adds `name` to collection `'signature_name'`.
    - If exporting: inserts `PredictOutput(outputs)` into `_export_outputs`, else ignores.
    - Raises `KeyError` if `name` already exists.
  - `add_training_hook(hook)`:
    - Prevents multiple `KafkaMetricHook` or `FileMetricHook`.
  - `add_layout(name, slice_list, out_type, shape_list)`:
    - Builds `OutConfig` with `OutType` mapping (`concat/stack/addn/none`).
    - For each slice, adds `slice_configs` with `feature_name`, `start`, `end`.
    - For each shape, writes dims; first dim forced to `-1`, subsequent dims from int or `.value`.
    - Stores in `_layout_dict[name]`.
  - `layout_dict` property getter/setter.
- **`MonolithModel(MonolithBaseModel)`**
  - `params()` adds `feature_list` string path.
  - `__init__`:
    - Uses provided params or class params.
    - Sets `dump_utils.enable = FLAGS.enable_model_dump`.
  - `_get_fs_conf(shared_name, slot, occurrence_threshold, expire_time)`:
    - Returns `FeatureSlotConfig` with `has_bias=False`, `slot_id=slot`, `occurrence_threshold`, `expire_time`,
      and hash table config using `GpucucoHashTableConfig` if `self.p.train.use_gpu_emb_table` else `CuckooHashTableConfig`.
  - `_embedding_slice_lookup(fc, slice_name, slice_dim, initializer, optimizer, compressor, learning_rate_fn, slice_list)`:
    - Asserts non-fused layout.
    - Accepts `fc` as feature name or `FeatureColumn`.
    - Applies `_share_slot_mapping` for shared embedding names.
    - Creates or reuses `FeatureSlice` in `slice_dict[feature_name][slice_name]`.
    - Appends `(fc.feature_name, fc_slice)` to `slice_list`.
    - Returns `fc.embedding_lookup(fc_slice)`.
  - `create_embedding_feature_column(feature_name, occurrence_threshold=None, expire_time=36500, max_seq_length=0, shared_name=None, combiner=None)`:
    - Converts `combiner` string to `FeatureColumn.reduce_sum/reduce_mean/first_n`.
    - Resolves `feature_name` and `slot` via `get_feature_name_and_slot`.
    - If `feature_name` exists in `fc_dict`, returns it.
    - If `shared_name` provided:
      - Stores `_share_slot_mapping[feature_name] = shared_name`.
      - Reuses existing `fs_dict` or `fc_dict` for shared slot if present.
      - Else creates new `FeatureSlot` for `shared_name` and stores in `fs_dict`.
      - If shared slot not created first and `get_feature_name_and_slot` fails, raises exception with explicit message.
    - If not shared: creates new `FeatureSlot` via `ctx.create_feature_slot`.
    - Default `combiner`: `first_n(max_seq_length)` for sequence features, else `reduce_sum`.
    - Creates `FeatureColumn`, stores in `fc_dict`, returns it.
  - `lookup_embedding_slice(features, slice_name, slice_dim=None, initializer=None, optimizer=None, compressor=None, learning_rate_fn=None, group_out_type='add_n', out_type=None)`:
    - Computes `layout_name = f'{slice_name}_{md5(sorted(features)).hexdigest()}'`.
    - If fused layout:
      - If `features` is list/tuple and `slice_dim` int and contains group tuples/lists: raises `ValueError("group pool is not support when fused_layout")`.
      - Returns `ctx.layout_factory.get_layout(layout_name)`.
    - Otherwise builds `feature_embeddings` and `slice_list` from `features` in these cases:
      - `dict`: feature name → slice dim.
      - `list/tuple` + `slice_dim` int:
        - if all elements are (str|int|FeatureColumn): fixed-dim list.
        - if all elements are list/tuple groups: `group_out_type` must be `concat` or `add_n`;
          each group is a list of feature names; group embeddings are summed or concatenated.
        - else raises `ValueError("ValueError for features")`.
      - `list/tuple` of `(feature, dim)` pairs: variable dims.
      - Otherwise raises `ValueError("ValueError for features")`.
    - If `out_type is None`: records layout with `shape_list` from embeddings and returns list of embeddings.
    - Else `out_type` in `{concat, stack, add_n, addn}`:
      - `concat`: `tf.concat(axis=1)`
      - `stack`: `tf.stack(axis=1)`
      - `add_n/addn`: `tf.add_n`
      - Records layout and returns tensor.
  - `share_slot(features=None, share_meta=None, variant_type='example', suffix='share')`:
    - For each `name -> (inplace, slot)` in `share_meta`:
      - Registers slot mapping via `register_slots`, using `shared_name = f'{name}_{suffix}'` when not inplace.
    - If `features` is dict:
      - `inplace=True`: `features[name] = switch_slot(features[name], slot)`
      - Else: `features[shared_name] = switch_slot(features[name], slot)`
      - Returns modified dict.
    - Else returns `map_fn = lambda tensor: switch_slot_batch(tensor, share_meta, variant_type=variant_type, suffix=suffix)`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src` for model base + training loop glue; `monolith-rs/crates/monolith-layers` for logit correction; `monolith-rs/crates/monolith-data` for feature slots; `monolith-rs/crates/monolith-hash-table` for embedding tables; `monolith-rs/crates/monolith-serving` for export signatures.
- Rust public API surface:
  - `get_sigmoid_loss_and_pred` / `get_softmax_loss_and_pred` equivalents in a `losses` or `metrics` module.
  - `DeviceCtxType` + `MonolithDeviceCtx` analog for device placement (no-op in pure Candle; required for TF runtime backend).
  - `MonolithBaseModel` trait + `MonolithModel` struct that mirror param plumbing, embedding-slice helpers,
    training/eval/predict flow, file output, and metrics hooks.
  - Export signature registry mirroring `add_extra_output`.
- Data model mapping:
  - Feature slots/slices map to Rust `FeatureSlot`, `FeatureColumn`, `FeatureSlice` types.
  - Layout dictionary maps to Rust `OutConfig` protobufs (via `monolith-proto`) with consistent shape semantics (`-1` batch dim).
- Feature gating:
  - **Default**: Candle-native backend; `MonolithDeviceCtx` becomes a no-op.
  - **Optional**: TF runtime backend only when `saved_model.pb` + `libtensorflow` + custom ops present.
  - Metrics hooks (Kafka/file/deep_insight) should be feature-gated with optional dependencies.
- Integration points:
  - Training entrypoints (Estimator analog) must call into `create_model_fn` flow equivalents.
  - Export flow must honor `export_context`-like state to control output signatures and device placement.

**Implementation Steps (Detailed)**
1. Define Rust equivalents for loss helpers with identical shape handling, sample-rate correction, clipping, and mask/weight semantics.
2. Implement `MonolithDeviceCtx` abstraction; in Candle, no-op; in TF runtime, map to device placement APIs.
3. Build Rust `MonolithBaseModel` trait:
   - Store per-graph/per-run `losses`, `training_hooks`, and `export_outputs`.
   - Implement `create_input_fn`, `create_model_fn`, `create_serving_input_receiver_fn` analogs.
4. Implement file output writer with exact formatting and ordering (output_fields order, dict pred sorted by key, delimiter handling).
5. Port metrics collection:
   - AUC/MSE metrics with logging + summary behavior (or compatible replacements).
   - Deep insight pipeline with Kafka/file hooks and per-worker filename suffixing.
6. Implement embedding feature slot + slice machinery with shared slots and layout tracking.
7. Add export signature registry and merge semantics for extra outputs.
8. Add feature flags and error handling parity for unsupported paths (e.g., fused layout group pooling).
9. Add cross-language tests: compare output file contents, metric logging events, and embedding slice layouts.

**Tests (Detailed)**
- Python tests: `monolith/native_training/model_comp_test.py` (uses `MonolithModel`).
- Rust tests: none yet (needs new parity tests for loss helpers, file output formatting, and embedding slice layouts).
- Cross-language parity test:
  - Golden test that runs a minimal TF model with two heads and compares loss/pred outputs + output file formatting.
  - Embedding slice layout test comparing `OutConfig` shape/slice configs between Python and Rust.

**Gaps / Notes**
- `_get_real_mode` rejects `EVAL`; this is likely intentional in their training flow (only TRAIN/PREDICT). If Rust adds eval mode, define parity behavior explicitly.
- Prediction file output uses **sorted dict keys** only when `pred` is dict; list/tuple order is preserved.
- `create_model_fn` treats tuple/list return as classification and emits warning; exact warning text should be preserved if exposed.
- Uses private TF API `graph._apply_device_functions` for `global_step` ops.
- Metrics include `tf.print(..., output_stream=sys.stderr)` side effects.
- Deep insight hooks only attach when sample ratio > 0.

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
