<!--
Source: task/request.md
Lines: 17214-17479 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/model_export/saved_model_exporters.py`
<a id="monolith-native-training-model-export-saved-model-exporters-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 739
- Purpose/role: Implements SavedModel exporters for standalone and distributed export, including hashtable restore/assign signatures and warmup assets.
- Key symbols/classes/functions: `BaseExporter`, `StandaloneExporter`, `DistributedExporter`.
- External dependencies: TensorFlow SavedModel internals, Monolith hash table ops, export_context, DumpUtils.
- Side effects: Writes SavedModel directories, copies assets, modifies TF collections, restores variables/hashtables.

**Required Behavior (Detailed)**
- `BaseExporter`:
  - Stores model_fn, model_dir, export_dir_base, shared_embedding, warmup_file, and optional export_context_list.
  - `create_asset_base()`:
    - Adds a `ASSET_BASE` tensor and AssetFileDef to assets collection if not already.
    - Returns tensor with value `"./"`.
  - `add_ckpt_to_assets(ckpt_to_export, pattern="*")`:
    - Adds all matching ckpt asset files to `ASSET_FILEPATHS` collection.
  - `build_signature(input_tensor_dict, output_tensor_dict)`:
    - Wraps tensors or TensorInfo into `SignatureDef` for PREDICT.
  - `_freeze_dense_graph(graph_def, signature_def_map, session)`:
    - Collects all input/output nodes from signatures and uses `convert_variables_to_constants`.
    - Restores device placement in frozen graph.
  - `_export_saved_model_from_graph(...)`:
    - Requires export_dir or export_dir_base.
    - Builds signatures from export_ctx.
    - Optionally adds hashtable assign signatures.
    - Creates Session with soft placement + GPU updates.
    - Restores variables and hashtables (restore ops and assign ops).
    - Writes SavedModel via `Builder` to temp dir then renames.
    - Copies `assets_extra` to `assets.extra` if provided.
  - `_export_frozen_saved_model_from_graph(...)`:
    - Similar but freezes graph and re-imports into a new graph before export.
  - `create_hashtable_restore_ops` / `create_multi_hashtable_restore_ops`:
    - For each (multi) hash table in graph collections, builds restore ops.
    - If not shared_embedding, adds ckpt files to assets and uses asset base; else uses ckpt asset dir.
  - `build_hashtable_assign_inputs_outputs`:
    - Creates placeholder-based assign tensors for hashtable update signature.
  - `add_multi_hashtable_assign_signatures`:
    - Adds raw_assign signatures for multi-hash tables (ragged id + flat values).
  - `_model_fn_with_input_reveiver`:
    - Runs model_fn in PREDICT mode and registers signatures in export_context.
  - `export_saved_model(...)`:
    - Abstract.
  - `gen_warmup_assets()`:
    - Generates warmup TFRecord via `gen_warmup_file` if not present and returns assets dict.
- `StandaloneExporter.export_saved_model(...)`:
  - Enters export mode STANDALONE; clears `TF_CONFIG` temporarily.
  - Builds graph, runs model_fn, exports SavedModel with warmup assets.
  - Restores `TF_CONFIG` on exit.
- `DistributedExporter.export_saved_model(...)`:
  - Creates ExportContext with `with_remote_gpu` flag.
  - Enters DISTRIBUTED export mode; clears `TF_CONFIG` temporarily.
  - Exports entry graph (optionally with GPU device placement).
  - Exports dense subgraphs and ps subgraphs stored in export_ctx.
  - Supports `dense_only`, `include_graphs`, `global_step_as_timestamp`, `freeze_variable`.
  - Skips exporting if target dir already exists.

**Rust Mapping (Detailed)**
- Target crate/module: N/A.
- Rust public API surface: if implementing SavedModel export, mirror BaseExporter and specialized exporters.
- Data model mapping: SavedModel signatures, assets, and hashtable metadata.
- Feature gating: TF runtime + monolith hash table ops required.
- Integration points: export pipeline, checkpoint loader, hash table restore.

**Implementation Steps (Detailed)**
1. Implement export context signature collection in Rust if needed.
2. Add SavedModel builder wrappers and asset copying.
3. Implement hash table restore/assign signatures in Rust or document lack.
4. Mirror distributed export layout (entry + dense + ps submodels).

**Tests (Detailed)**
- Python tests: `saved_model_exporters_test.py`.
- Rust tests: add export smoke tests if implemented.
- Cross-language parity test: compare exported signature defs and asset layout.

**Gaps / Notes**
- `_model_fn_with_input_reveiver` typo in name (receiver misspelled) but used internally.
- Uses TF internal APIs; may be brittle across TF versions.

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

### `monolith/native_training/model_export/saved_model_exporters_test.py`
<a id="monolith-native-training-model-export-saved-model-exporters-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 153
- Purpose/role: Tests StandaloneExporter with hash tables and multi-hash tables, including shared embedding mode.
- Key symbols/classes/functions: `ModelFnCreator`, `SavedModelExportersTest`.
- External dependencies: TensorFlow Estimator, Monolith hash table ops, SavedModel exporter.
- Side effects: Creates checkpoints and exports SavedModels under `TEST_TMPDIR`.

**Required Behavior (Detailed)**
- `ModelFnCreator.create_model_fn()`:
  - Sets `_called_in_exported_mode` if `export_context.EXPORT_MODE != None`.
  - Builds hash table and multi-hash table.
  - In PREDICT mode:
    - Exports outputs for default signature, "table/lookup", and "mtable/lookup".
  - In TRAIN mode:
    - Adds assign_add ops for tables.
    - Adds `CheckpointSaverHook` with hash table saver listeners.
    - Returns `EstimatorSpec` with train_op and loss=0.
- `dummy_input_receiver_fn` returns empty features with a string placeholder.
- `SavedModelExportersTest`:
  - `run_pred(export_path, key=DEFAULT)` loads SavedModel and runs output tensor.
  - `testBasic`:
    - Trains one step to create checkpoint.
    - Exports SavedModel and asserts predictions for table and mtable lookups.
    - Asserts model_fn was called in export mode.
  - `testSharedEmebdding`:
    - Exports with `shared_embedding=True` and asserts predictions.

**Rust Mapping (Detailed)**
- Target crate/module: N/A.
- Rust public API surface: none.
- Data model mapping: hash table ops and SavedModel exports.
- Feature gating: TF runtime + hash table ops.
- Integration points: export pipeline and hash table checkpointing.

**Implementation Steps (Detailed)**
1. If Rust supports hash-table-backed exports, add equivalent tests.
2. Verify lookup outputs after export match expected values.
3. Cover shared embedding behavior if implemented.

**Tests (Detailed)**
- Python tests: `saved_model_exporters_test.py`.
- Rust tests: none.
- Cross-language parity test: compare exported predictions for fixed inputs.

**Gaps / Notes**
- Misspelling `testSharedEmebdding` in test name.

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

### `monolith/native_training/model_export/saved_model_visulizer.py`
<a id="monolith-native-training-model-export-saved-model-visulizer-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 89
- Purpose/role: CLI utility to import a SavedModel protobuf and write it to TensorBoard for visualization.
- Key symbols/classes/functions: `import_to_tensorboard`, `main`.
- External dependencies: TensorFlow SavedModel proto, TensorBoard summary writer.
- Side effects: Reads a SavedModel file, writes TensorBoard logdir.

**Required Behavior (Detailed)**
- `import_to_tensorboard(model_dir, log_dir)`:
  - Opens SavedModel file at `model_dir` as bytes.
  - Parses `SavedModel` proto; if more than one meta_graph, prints message and exits with code 1.
  - Imports the first graph_def into a new graph.
  - Writes graph to `log_dir` using `summary.FileWriter`.
  - Prints TensorBoard command.
- `main` invokes `import_to_tensorboard` with CLI flags.
- CLI parsing via `argparse`, requires `--model_dir` and `--log_dir`.
- Uses `app.run` from TF platform with parsed args.

**Rust Mapping (Detailed)**
- Target crate/module: N/A.
- Rust public API surface: none.
- Data model mapping: SavedModel proto parsing.
- Feature gating: TF runtime only.
- Integration points: tooling/debugging.

**Implementation Steps (Detailed)**
1. If Rust needs similar tool, parse SavedModel protobuf and emit graph for visualization.
2. Provide CLI for model_dir/log_dir.

**Tests (Detailed)**
- Python tests: none.
- Rust tests: none.
- Cross-language parity test: not applicable.

**Gaps / Notes**
- Uses TF internal APIs; depends on TensorFlow installation.

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

### `monolith/native_training/model_export/warmup_data_decoder.py`
<a id="monolith-native-training-model-export-warmup-data-decoder-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 55
- Purpose/role: CLI tool to decode TF Serving warmup TFRecord files and print sanitized requests.
- Key symbols/classes/functions: `main`.
- External dependencies: TensorFlow, TF Serving PredictionLog proto, `env_utils`.
- Side effects: Reads TFRecord file, logs decoded model specs.

**Required Behavior (Detailed)**
- Flag `file_name` specifies input TFRecord path.
- `main`:
  - Attempts `env_utils.setup_hdfs_env()`; ignores errors.
  - Enables eager execution and sets TF logging verbosity.
  - Defines `decode_fn` to parse `PredictionLog` from record bytes.
  - Iterates TFRecordDataset over `file_name`, decodes each log.
  - Extracts PredictRequest, replaces `string_val:.*` with `string_val: ...` in printed output.
  - Logs index and sanitized request string.
- Uses `app.run(main)`.

**Rust Mapping (Detailed)**
- Target crate/module: N/A.
- Rust public API surface: none.
- Data model mapping: TF Serving PredictionLog proto.
- Feature gating: TF Serving protos required.
- Integration points: tooling for warmup verification.

**Implementation Steps (Detailed)**
1. If Rust needs a decoder, parse TFRecord PredictionLogs and print sanitized requests.
2. Mirror regex sanitization for `string_val` fields.

**Tests (Detailed)**
- Python tests: none.
- Rust tests: none.
- Cross-language parity test: not applicable.

**Gaps / Notes**
- Eager execution is required; script is for inspection only.

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
