<!--
Source: task/request.md
Lines: 16018-16209 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/model_comp_test.py`
<a id="monolith-native-training-model-comp-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 183
- Purpose/role: Integration test comparing TF embedding updates vs Monolith embedding updates under sync training (Horovod).
- Key symbols/classes/functions: `EmbeddingUpdateTask`, `CpuSyncTrainTest`, `lookup_tf_embedding`.
- External dependencies: TensorFlow, Horovod, Monolith CPU training stack, Keras layers.
- Side effects: Sets environment variables at import; runs distributed training; writes model checkpoints under `/tmp/<user>/monolith_test/...`.

**Required Behavior (Detailed)**
- Module-level env vars (set before TF/Horovod import):
  - `MONOLITH_WITH_HOROVOD=True`, `HOROVOD_AUTOTUNE=1`, `HOROVOD_CYCLE_TIME=0.1`,
    `MONOLITH_SYNC_EMPTY_RANK0_PS_SHARD=0`, `MONOLITH_WITH_ALLREDUCE_FUSION=one`,
    `MONOLITH_ROOT_LOG_INTERVAL=10`.
- Sets TF v1 random seed to 42.
- Global constants: `num_features=17`, `batch_size=455`, `emb_dim=15`, `fid_max_val=100000`.
- `lookup_tf_embedding(features, f_name, dim)`:
  - Builds `RaggedTensor` from `tf_<f_name>_p1`/`p2`.
  - Embedding lookup on a zeros-initialized variable.
  - Returns `segment_sum` by row ids.
- `EmbeddingUpdateTask(MonolithModel)`:
  - `__init__`: sets `train.max_steps=50`, `train.per_replica_batch_size=batch_size`.
  - `input_fn`:
    - Generates random feature vectors with variable length per feature (1..24).
    - Uses `dense_to_ragged_batch` with batch_size and `advanced_parse`.
    - Adds `tf_feature{i}_p1/p2` to features for TF embedding lookup.
  - `model_fn`:
    - Creates embedding feature columns and Monolith embeddings.
    - Computes TF embeddings via `lookup_tf_embedding`.
    - Asserts Monolith embeddings equal TF embeddings.
    - Builds parallel Keras MLPs for both embedding sets, computes MSE losses.
    - Returns `EstimatorSpec` with combined loss, predictions, labels, head names, and optimizer.
  - `serving_input_receiver_fn`: unimplemented (`pass`).
- `CpuSyncTrainTest`:
  - `_create_config(gpu, multi_hash_table)` builds `DistributedCpuTrainingConfig` with sync training enabled.
  - `test_embedding_update`:
    - Initializes Horovod, runs distributed sync training in 2 configurations (cpu/multi-hash on/off).
    - If GPU available, repeats with GPU enabled.
- `__main__`: disables eager execution and runs `tf.test.main()`.

**Rust Mapping (Detailed)**
- Target crate/module: N/A (TF/Horovod integration test only).
- Rust public API surface: none.
- Data model mapping: none.
- Feature gating: Horovod/TF runtime only.
- Integration points: training loop parity for embedding updates.

**Implementation Steps (Detailed)**
1. If Rust aims to match embedding update semantics, port the comparison into Rust unit tests using Candle/TF backend.
2. Provide deterministic random feature generation for repeatability.
3. Mirror embedding lookup + segment sum behavior.

**Tests (Detailed)**
- Python tests: `monolith/native_training/model_comp_test.py`.
- Rust tests: none.
- Cross-language parity test: compare embedding tensors and loss values on fixed seeds.

**Gaps / Notes**
- `serving_input_receiver_fn` is unimplemented.

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

### `monolith/native_training/model_dump/dump_utils.py`
<a id="monolith-native-training-model-dump-dump-utils-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 757
- Purpose/role: Central model dump utility that records feature/slice metadata, input/output tensors, signatures, and graph defs into `ModelDump` protobufs.
- Key symbols/classes/functions: `DumpUtils`, `parse_input_fn_result`, wrappers `record_feature`, `record_slice`, `record_receiver`.
- External dependencies: TensorFlow graph/ops internals, protobufs (`model_dump` protos), pickle, `tf.io.gfile`, export context, data parsers.
- Side effects: Monkeypatches `util.parse_input_fn_result`; stores `ProtoModel` on default graph; writes/reads dump files.

**Required Behavior (Detailed)**
- `DumpUtils` is a singleton (`_instance`), `__init__` initializes fields only once:
  - `enable`, `_params`, `_run_config`, `_user_params`, train/infer `ProtoModel` + graph defs, sub-model caches, table configs, slice dims, feature combiners.
- `model_dump` property:
  - Attaches `ProtoModel()` to default graph as `graph.monolith_model_dump`.
- `update_kwargs_with_default(func, kwargs)`:
  - Fills `kwargs` entries with function default values when `None`.
- `record_feature(func)` wrapper:
  - When `need_record`, appends to `model_dump.features`.
  - Copies args/kwargs into proto, converting integer `feature_name` via `get_feature_name_and_slot`.
  - Logs warnings if field missing in proto.
- `record_slice(func)` wrapper:
  - Forbids `learning_rate_fn` (raises `Exception`).
  - Records slice config into `model_dump.emb_slices` including initializer/optimizer/compressor protos.
  - Calls wrapped function and appends output tensor names.
- `record_receiver(func)` wrapper:
  - Records serving input receiver features and receiver tensors into `serving_input_receiver_fn`.
  - Stores ragged tensors as `{values,row_splits,is_ragged}`; dense tensors include dtype/last_dim.
- `record_params(model)`:
  - Captures non-callable, non-private attrs except a skip list into `_params`.
- `get_params_bytes(model)`:
  - Pickles model attributes (including deep-copied `p` and serialized `_layout_dict`).
  - Returns pickled bytes; used by `add_model_fn`.
- `add_signature` / `restore_signature`:
  - Syncs signatures with current export context, mapping tensor names.
- `add_model_fn(model, mode, features, label, loss, pred, head_name, is_classification)`:
  - Fills `model_fn` proto with labels, loss, predictions, head names, classification flags.
  - Adds user summaries from `GraphKeys.SUMMARIES`.
  - Records non-ragged features not already registered.
  - Records extra losses from graph `__losses`.
  - Records `export_outputs` as `extra_output.fetch_dict`.
  - Enforces that only `ItemPoolSaveRestoreHook` is allowed in `__training_hooks`.
  - Stores signatures and SaveSliceInfo for variables.
  - Snapshots graph_def for TRAIN vs INFER into `train_graph`/`infer_graph`.
- `add_input_fn(results)`:
  - Records input feature tensor names and ragged flags; records label if present.
  - Stores parser type and item pool name.
- `add_sub_model(sub_model_type, name, graph)` / `restore_sub_model(sub_model_type)`:
  - Stores/restore sub-graph defs for PS or dense submodels via export context subgraphs.
- `add_optimizer(optimizer)`:
  - Pickles optimizer into `model_dump.optimizer`.
- `dump(fname)`:
  - Builds `ModelDump` proto with run config, user params, train/infer graphs, sub-models, table configs, slice dims, combiners.
  - Writes serialized bytes to `fname` using `tf.io.gfile`.
- `load(fname)`:
  - Reads `ModelDump` and reconstructs train/infer graph defs, table configs, feature slices, combiners, user params, sub-models.
- `get_graph_helper(mode)`:
  - Builds `GraphDefHelper` with SaveSliceInfo from train/infer model dump.
  - Caches on graph as `graph.graph_def_helper`.
- `restore_params()`:
  - Unpickles model params; rebuilds `_layout_dict` from `OutConfig` proto bytes.
  - Deletes `_training_hooks` key if present; raises if layout_dict missing.
- `need_record`:
  - True when `enable` and graph does not have `DRY_RUN` attribute.
- `table_configs` property/setter:
  - Converts between proto configs and `entry.HashTableConfigInstance`.
  - Setter disallows non-numeric `learning_rate_fns`.
- `feature_slice_dims` property/setter:
  - Converts between proto list and dict of dims.
- `feature_combiners` property/setter:
  - Maps `ReduceSum/ReduceMean/FirstN` to/from proto enum `Combiner`.
- `get_slot_to_occurrence_threshold` / `get_slot_to_expire_time`:
  - Builds slot->value maps; warns if slot resolution fails.
- `has_collected`:
  - True if table configs, slice dims, combiners all non-empty; otherwise asserts they are empty.
- `parse_input_fn_result(result)`:
  - If `DatasetV2`, makes iterator + `_DatasetInitializerHook`.
  - Else uses `DatasetInitHook` from collection `mkiter`.
  - Calls `DumpUtils().add_input_fn` and returns parsed iterator result + input hooks.
  - Monkeypatches `util.parse_input_fn_result`.

**Rust Mapping (Detailed)**
- Target crate/module: N/A (TF graph/proto dump infrastructure not in Rust).
- Rust public API surface: if parity required, add a model-dump module capturing graph metadata and feature configs.
- Data model mapping: Protobuf `ModelDump` -> Rust structs; tensor names as strings.
- Feature gating: TF runtime only if applicable.
- Integration points: model export, serving input receivers, embedding table configs.

**Implementation Steps (Detailed)**
1. Decide whether Rust needs model dump/export parity; define equivalent data structures.
2. Implement feature/slice recording and tensor-name capture.
3. Implement save/load of graph metadata and signatures.
4. Mirror validation (learning_rate_fn disallow, training hooks restrictions).

**Tests (Detailed)**
- Python tests: none in repo.
- Rust tests: add serialization/deserialization tests if implemented.
- Cross-language parity test: compare dumped proto fields for a simple model.

**Gaps / Notes**
- `export_outputs` branch uses `ts` variable that may be undefined when outputs are not dict (possible bug).
- `parse_input_fn_result` monkeypatch changes global TF estimator behavior.

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
