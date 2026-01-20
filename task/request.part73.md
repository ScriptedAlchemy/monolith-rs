<!--
Source: task/request.md
Lines: 16661-16937 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/model_export/demo_predictor.py`
<a id="monolith-native-training-model-export-demo-predictor-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 110
- Purpose/role: CLI demo to load a SavedModel and run prediction with randomly generated inputs.
- Key symbols/classes/functions: `make_fid_v1`, `generate_demo_instance`, `random_generate_instances`, `random_generate_int`, `random_generate_float`, `predict`, `main`.
- External dependencies: TensorFlow SavedModel, NumPy, `proto_parser_pb2.Instance`, TestFFMModel constants.
- Side effects: Loads SavedModel from disk; logs prediction outputs.

**Required Behavior (Detailed)**
- Flags:
  - `saved_model_path` (required path), `tag_set` (default "serve"), `signature` (default "serving_default"), `batch_size` (default 128).
- `make_fid_v1(slot_id, fid)`:
  - Encodes FID v1 as `(slot_id << 54) | fid`.
- `generate_demo_instance()`:
  - Creates `Instance` proto.
  - For each slot in `model._NUM_SLOTS`, generates 5 random fids in that slot based on `max_vocab`.
  - Returns serialized bytes.
- `random_generate_instances(bs)`:
  - Returns list of `bs` serialized Instance bytes.
- `random_generate_examples(bs)` (unused):
  - Returns list of serialized Example bytes using `model.generate_ffm_example`.
- `random_generate_int(shape)`:
  - Returns int64 array in `[0, max_vocab)` where `max_vocab = max(_VOCAB_SIZES) * _NUM_SLOTS`.
- `random_generate_float(shape)`:
  - Returns float array of `uniform(0,1)` values.
- `predict()`:
  - Loads SavedModel with `tf.compat.v1.saved_model.load`.
  - Reads signature inputs/outputs for `FLAGS.signature`.
  - For each input, builds a feed tensor based on dtype:
    - string -> list of serialized instances, shape length must be 1.
    - int64 -> random ints.
    - float32 -> random floats.
    - else raises `ValueError`.
  - Runs session and logs outputs.
- `main` calls `predict`; `__main__` sets logging verbosity to INFO and runs via absl.

**Rust Mapping (Detailed)**
- Target crate/module: N/A.
- Rust public API surface: none.
- Data model mapping: if implementing predictor in Rust, map SavedModel signature I/O to random data generation.
- Feature gating: TF runtime only.
- Integration points: serving validation / smoke tests.

**Implementation Steps (Detailed)**
1. If Rust can load SavedModels, implement a CLI to sample inputs and run predictions.
2. Mirror dtype-based generation (string -> serialized Instance, int64/float32 random arrays).
3. Match FID v1 encoding for feature IDs.

**Tests (Detailed)**
- Python tests: `demo_predictor_client.py` (manual) or none.
- Rust tests: none.
- Cross-language parity test: compare output shapes for identical inputs.

**Gaps / Notes**
- `random_generate_examples` is unused.

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

### `monolith/native_training/model_export/demo_predictor_client.py`
<a id="monolith-native-training-model-export-demo-predictor-client-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 93
- Purpose/role: gRPC client for TensorFlow Serving PredictionService using random inputs derived from SavedModel signature.
- Key symbols/classes/functions: `get_signature_def`, `main`.
- External dependencies: gRPC, TensorFlow Serving protos, TensorFlow.
- Side effects: Sends Predict RPC to remote server.

**Required Behavior (Detailed)**
- Flags:
  - `server` (default "localhost:8500"), `model_name` ("default"), `signature_name` ("serving_default"), `use_example` (bool).
  - Note: code references `FLAGS.batch_size` but flag is not defined in this file (bug).
- `get_signature_def(stub)`:
  - Requests signature_def metadata via `GetModelMetadata`.
  - Unpacks `SignatureDefMap` and returns signature by `FLAGS.signature_name`.
  - Prints available signature names.
- `main`:
  - Creates insecure gRPC channel and PredictionService stub.
  - Builds PredictRequest with model spec.
  - For each input in signature:
    - Computes shape, substituting `FLAGS.batch_size` for -1 dims.
    - Generates example/instance bytes for string inputs.
    - Generates random ints/floats for int64/float32 inputs.
    - Raises `ValueError` for unsupported dtype.
  - Calls `stub.Predict(request, timeout=30)` and logs result.
- Logging verbosity set to INFO in `__main__`.

**Rust Mapping (Detailed)**
- Target crate/module: N/A.
- Rust public API surface: none.
- Data model mapping: if implementing, use TF Serving gRPC protos in Rust.
- Feature gating: gRPC + TF Serving protos.
- Integration points: serving smoke tests.

**Implementation Steps (Detailed)**
1. Define missing `batch_size` flag or pass as CLI arg.
2. If Rust needs a client, implement signature discovery and random input generation.
3. Mirror example/instance encoding logic using demo_predictor helpers.

**Tests (Detailed)**
- Python tests: none.
- Rust tests: none.
- Cross-language parity test: compare request shapes and dtype handling.

**Gaps / Notes**
- `FLAGS.batch_size` is referenced but never defined (likely a bug).

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

### `monolith/native_training/model_export/export_context.py`
<a id="monolith-native-training-model-export-export-context-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 141
- Purpose/role: Manages export mode state and signatures for model export; provides context manager for export mode.
- Key symbols/classes/functions: `ExportMode`, `ExportContext`, `enter_export_mode`, `is_exporting*`, `get_current_export_ctx`, `is_dry_run_or_exporting`.
- External dependencies: TensorFlow, `tf_contextlib`, `monolith_export` decorator.
- Side effects: Global export mode state; stores signatures in TF collections.

**Required Behavior (Detailed)**
- `ExportMode` enum: `NONE`, `STANDALONE`, `DISTRIBUTED`.
- `SavedModelSignature` namedtuple (`name`, `inputs`, `outputs`).
- `ExportContext`:
  - Maintains `sub_graphs` and `dense_sub_graphs` as `defaultdict(tf.Graph)`.
  - Maintains `_signatures` keyed by graph id; each entry maps name -> SavedModelSignature.
  - `add_signature` adds to TF collection `signature_name` and stores signature.
  - `merge_signature` updates existing signature inputs/outputs or creates empty.
  - `signatures(graph)` returns signature values for given graph id.
  - `with_remote_gpu` property returns constructor flag.
  - `sub_graph_num` returns count of sub_graphs.
- Globals:
  - `EXPORT_MODE` starts as `NONE`.
  - `EXPORT_CTX` starts as `None`.
- `is_exporting` / `is_exporting_standalone` / `is_exporting_distributed`:
  - Compares `EXPORT_MODE` to enum values.
- `get_current_export_ctx`:
  - Returns `EXPORT_CTX`.
- `enter_export_mode(mode, export_ctx=None)`:
  - Asserts no nested export (`EXPORT_MODE is NONE` and `EXPORT_CTX is None`).
  - Creates new `ExportContext()` if not provided.
  - Sets globals, yields `export_ctx`, then resets globals to defaults in `finally`.
- `is_dry_run_or_exporting()`:
  - Returns True if export mode active or default graph has `dry_run` attribute.

**Rust Mapping (Detailed)**
- Target crate/module: N/A.
- Rust public API surface: optional export context struct with thread-local state.
- Data model mapping: signatures map, subgraph registry.
- Feature gating: export-only.
- Integration points: model export pipeline.

**Implementation Steps (Detailed)**
1. Implement export context state in Rust (thread-local/global).
2. Provide RAII guard for entering/exiting export mode.
3. Mirror signature tracking and graph association logic if needed.

**Tests (Detailed)**
- Python tests: `export_context` is exercised by export hooks and demo exporters.
- Rust tests: add unit tests for mode nesting and signature registry.
- Cross-language parity test: ensure signature names collected match.

**Gaps / Notes**
- Uses global mutable state; not thread-safe for parallel exports.

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

### `monolith/native_training/model_export/export_hooks.py`
<a id="monolith-native-training-model-export-export-hooks-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 137
- Purpose/role: Checkpoint saver listener that exports SavedModel on each checkpoint and prunes old exports.
- Key symbols/classes/functions: `get_global_step`, `ExportSaverListener`.
- External dependencies: TensorFlow, Monolith save_utils/export_state_utils, custom metrics client.
- Side effects: Writes export directories, deletes old ones, emits metrics.

**Required Behavior (Detailed)**
- `get_global_step(checkpoint_path)`:
  - Regex `model.ckpt-(\d+)`; asserts match; returns int.
- `ExportSaverListener`:
  - `__init__(save_path, serving_input_receiver_fn, exporter, exempt_checkpoint_paths=None, dense_only=False)`:
    - Stores serving_input_receiver_fn and exporter.
    - `self._helper = save_utils.SaveHelper(save_path)`.
    - Builds `self._exempt_checkpoint_steps` from `exempt_checkpoint_paths` by parsing global steps.
    - `dense_only` toggles special deletion logic.
    - Creates metric client via `cli.get_cli(utils.get_metric_prefix())`.
  - `after_save(session, global_step_value)`:
    - Uses SaveHelper to get checkpoint prefix for step.
    - Calls `exporter.export_saved_model(...)`.
    - Accepts export_dirs as bytes, list, or dict of values.
    - For each export_dir:
      - Adds entry to export state and prunes old entries.
  - `_add_entry_to_state(export_dir, global_step_value)`:
    - Decodes bytes; computes base/version.
    - Appends `ServingEntry(export_dir, global_step)` to state and overwrites state file.
    - Calls `_update_metrics`.
  - `_maybe_delete_old_entries(export_dir)`:
    - Loads existing state; computes `existing_steps` from current checkpoints plus exempt steps.
    - If `dense_only`, also loads full checkpoint state from model_dir and includes all steps.
    - Removes entries not in `existing_steps`, deleting directories via `tf.io.gfile.rmtree`.
  - `_update_metrics(export_dir_base, version)`:
    - Emits `export_models.latest_version` as int if version is numeric.
    - `version` uses `split(".")[0]` to handle float-like names.
    - Logs warning on exceptions every 1200 seconds.

**Rust Mapping (Detailed)**
- Target crate/module: N/A.
- Rust public API surface: if implementing export hooks, add a checkpoint listener trait.
- Data model mapping: export state protobufs -> Rust structs.
- Feature gating: export-only.
- Integration points: checkpoint saving, export pipeline, metrics client.

**Implementation Steps (Detailed)**
1. Implement checkpoint listener in Rust training loop if needed.
2. Mirror export directory state tracking and pruning rules.
3. Add metrics emission for latest export version.

**Tests (Detailed)**
- Python tests: `export_hooks_test.py`.
- Rust tests: add filesystem tests for pruning behavior.
- Cross-language parity test: compare export state entries after simulated checkpoints.

**Gaps / Notes**
- `get_global_step` asserts regex match; invalid checkpoint paths will raise AssertionError.

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
