<!--
Source: task/request.md
Lines: 16417-16660 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/model_export/data_gen_utils.py`
<a id="monolith-native-training-model-export-data-gen-utils-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 732
- Purpose/role: Generates synthetic Example/Instance/ExampleBatch data and PredictionLogs for model export, warmup, and testing.
- Key symbols/classes/functions: `FeatureMeta`, `ParserArgs`, `gen_fids_v1`, `gen_fids_v2`, `fill_features`, `fill_line_id`, `gen_example`, `gen_instance`, `gen_example_batch`, `gen_prediction_log`, `gen_warmup_file`, `gen_random_data_file`.
- External dependencies: TensorFlow, TF Serving protos, Monolith feature list, proto types (`Example`, `Instance`, `LineId`).
- Side effects: Writes TFRecord warmup files and binary data files.

**Required Behavior (Detailed)**
- Constants: `MASK_V1/MAX_SLOT_V1`, `MASK_V2/MAX_SLOT_V2` control fid encoding.
- `FeatureMeta`:
  - Infers dtype from LineId field descriptors or slot (defaults to float32 for dense, int64 for sparse).
  - `shape` defaults to 1 for dense, -1 for sparse.
- `ParserArgs` dataclass:
  - Reads defaults from TF collections via `get_collection`.
  - Ensures `DEFAULT_SERVING_SIGNATURE_DEF_KEY` is present in `signature_name`.
  - Attempts `FeatureList.parse()` if no feature_list provided.
- `gen_fids_v1(slot, size)` / `gen_fids_v2(slot, size)`:
  - Encodes slot in high bits and random low bits; v1 logs when slot > max, v2 asserts slot range.
- `fill_features` (singledispatch):
  - For `EFeature` (Example):
    - Sparse: generates fid_v2_list with drop_rate logic and slot-specific handling.
    - Dense: fills float/double/int64 lists with random values.
  - For `IFeature` (Instance):
    - Similar logic; uses `feature.fid`, `float_value`, `int64_value`.
- `fill_line_id(line_id, features, hash_len=48, actions=None)`:
  - Fills LineId fields based on metadata or defaults; handles repeated vs scalar fields.
- `lg_header(source)` / `sort_header(sort_id, kafka_dump, kafka_dump_prefix)`:
  - Produce binary headers for data files, including Java hash computation.
- `gen_example(...)`:
  - Builds Example with named_feature entries; uses `FeatureList` or slot lookup; fills labels and LineId.
- `gen_instance(...)`:
  - Builds Instance proto using fidv1/fidv2 features; fills labels and LineId.
- `gen_example_batch(...)`:
  - Builds ExampleBatch with per-feature lists, LineId list (`__LINE_ID__`), and labels (`__LABEL__`).
- `gen_prediction_log(args)`:
  - Generates PredictRequest logs using the appropriate variant type.
  - Supports multiple signatures; may emit multiple requests for multi-head outputs.
- `gen_warmup_file(warmup_file, drop_rate)`:
  - Builds ParserArgs, removes dense label if present, writes PredictionLog TFRecord to file.
  - Creates directories if needed; returns file path or None.
- `gen_random_data_file(...)`:
  - Writes binary file with headers and serialized instances for `num_batch`.

**Rust Mapping (Detailed)**
- Target crate/module: N/A (Python proto generators).
- Rust public API surface: if needed, add a data-gen utility module for tests/warmup.
- Data model mapping: map proto types (Example/Instance/ExampleBatch) to Rust protobufs.
- Feature gating: TF Serving protos required for PredictionLog generation.
- Integration points: model export/warmup pipeline.

**Implementation Steps (Detailed)**
1. Implement fid encoding and feature filling logic in Rust.
2. Mirror ParserArgs collection-based defaults if Rust uses similar collections.
3. Implement PredictionLog generation and TFRecord writing for warmup data.
4. Port binary data file format (headers + length + payload).

**Tests (Detailed)**
- Python tests: none (`data_gen_utils_test.py` is empty).
- Rust tests: add unit tests for fid encoding and example generation.
- Cross-language parity test: compare serialized outputs for fixed RNG seed.

**Gaps / Notes**
- Uses `eval` on stored representations and random data generation; not deterministic without seeding.

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

### `monolith/native_training/model_export/data_gen_utils_test.py`
<a id="monolith-native-training-model-export-data-gen-utils-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 0
- Purpose/role: Empty test placeholder.
- Key symbols/classes/functions: none.
- External dependencies: none.
- Side effects: none.

**Required Behavior (Detailed)**
- File is empty; no tests executed.

**Rust Mapping (Detailed)**
- Target crate/module: N/A.
- Rust public API surface: none.
- Data model mapping: none.
- Feature gating: none.
- Integration points: none.

**Implementation Steps (Detailed)**
1. Add tests if/when data generation is ported to Rust.

**Tests (Detailed)**
- Python tests: none.
- Rust tests: none.
- Cross-language parity test: N/A.

**Gaps / Notes**
- No coverage for data generation utilities.

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

### `monolith/native_training/model_export/demo_export.py`
<a id="monolith-native-training-model-export-demo-export-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 100
- Purpose/role: CLI demo that exports a saved model from the TestFFMModel using standalone or distributed exporter.
- Key symbols/classes/functions: `export_saved_model`, `main`.
- External dependencies: TensorFlow, Monolith CPU training, `parse_instances`, `StandaloneExporter`, `DistributedExporter`.
- Side effects: Writes SavedModel to disk under `export_base`; uses flags; disables eager execution.

**Required Behavior (Detailed)**
- Defines flags:
  - `num_ps` (default 5) for CPU training config.
  - `model_dir` and `export_base` default to `/tmp/<user>/monolith/native_training/demo/...`.
  - `export_mode` enum (Standalone or Distributed).
- `export_saved_model(model_dir, export_base, num_ps, export_mode)`:
  - Disables eager execution; sets TF logging verbosity to INFO.
  - Instantiates `TestFFMModel` params with name `"demo_export"` and batch size 64.
  - Creates `cpu_training.CpuTraining` with `CpuTrainingConfig(num_ps=num_ps)`.
  - Chooses exporter:
    - `StandaloneExporter` or `DistributedExporter` (with `shared_embedding=False`).
  - Defines `serving_input_receiver_fn`:
    - `instances` placeholder of dtype `tf.string` with shape `(None,)`.
    - Parses instances via `parse_instances`, with fidv1 features 0.._NUM_SLOTS-1.
    - Builds `features` dict with keys `feature_i` from `slot_i`.
    - Returns `tf.estimator.export.ServingInputReceiver`.
  - Calls `exporter.export_saved_model(serving_input_receiver_fn)`.
- `main(_)` calls `export_saved_model` with flags.
- `__main__` uses `absl.app.run`.

**Rust Mapping (Detailed)**
- Target crate/module: N/A (Python TF export demo).
- Rust public API surface: none.
- Data model mapping: if exporting in Rust, define equivalent serving input receiver.
- Feature gating: TF runtime only.
- Integration points: export pipeline.

**Implementation Steps (Detailed)**
1. If Rust export is desired, implement a demo exporter that mirrors TestFFMModel inputs.
2. Map parsing logic for FID v1 features to Rust serving inputs.
3. Preserve default paths and batch size for parity tests.

**Tests (Detailed)**
- Python tests: `monolith/native_training/model_export/demo_export_test.py`.
- Rust tests: none.
- Cross-language parity test: compare exported SavedModel signatures and input names.

**Gaps / Notes**
- Demo only; depends on TestFFMModel and CPU training stack.

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

### `monolith/native_training/model_export/demo_export_test.py`
<a id="monolith-native-training-model-export-demo-export-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 48
- Purpose/role: Integration test that trains TestFFMModel and verifies standalone/distributed export paths run without error.
- Key symbols/classes/functions: `DemoExportTest.test_demo_export`.
- External dependencies: TensorFlow, Monolith CPU training, `demo_export.export_saved_model`.
- Side effects: Creates training checkpoints and two SavedModel export directories under `TEST_TMPDIR`.

**Required Behavior (Detailed)**
- Disables eager execution at import time.
- `test_demo_export`:
  - Creates `model_dir = $TEST_TMPDIR/test_ffm_model`.
  - Trains TestFFMModel with `cpu_training.local_train(params, num_ps=5, model_dir=...)`.
  - Calls `demo_export.export_saved_model` twice:
    - Standalone export to `$TEST_TMPDIR/standalone_saved_model`.
    - Distributed export to `$TEST_TMPDIR/distributed_saved_model`.
  - Uses `ExportMode.STANDALONE` and `ExportMode.DISTRIBUTED`.
- No explicit assertions on contents; success is absence of errors.

**Rust Mapping (Detailed)**
- Target crate/module: N/A.
- Rust public API surface: none.
- Data model mapping: none.
- Feature gating: TF runtime only.
- Integration points: export pipeline.

**Implementation Steps (Detailed)**
1. If Rust adds export support, add a smoke test for export outputs.
2. Ensure deterministic temp paths and cleanup.

**Tests (Detailed)**
- Python tests: `monolith/native_training/model_export/demo_export_test.py`.
- Rust tests: none.
- Cross-language parity test: compare exported SavedModel signatures.

**Gaps / Notes**
- Test is heavy (trains and exports); may be slow.

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
