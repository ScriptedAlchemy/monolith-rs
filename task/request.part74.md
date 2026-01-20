<!--
Source: task/request.md
Lines: 16938-17213 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/model_export/export_hooks_test.py`
<a id="monolith-native-training-model-export-export-hooks-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 141
- Purpose/role: Tests ExportSaverListener behavior (state updates, dict export outputs, deletion of old exports).
- Key symbols/classes/functions: `ExportHookTest.testBasic`, `testExporterReturnsDict`, `testDeleted`.
- External dependencies: TensorFlow, `save_utils`, `export_state_utils`, `unittest.mock`.
- Side effects: Creates model/export dirs under `TEST_TMPDIR`.

**Required Behavior (Detailed)**
- `testBasic`:
  - Mocks exporter to return `export_dir` bytes and asserts checkpoint path format.
  - Runs `NoFirstSaveCheckpointSaverHook` and sets global_step to 10.
  - Verifies export state has one entry with correct dir and step.
- `testExporterReturnsDict`:
  - Mocks exporter to return dict of model names to export dirs.
  - Ensures no errors during export and state update.
- `testDeleted`:
  - Mocks exporter to create unique export dirs per step.
  - Uses `PartialRecoverySaver` with `max_to_keep=1`.
  - After two steps, verifies only latest export remains and old one deleted.
- Tests rely on `export_state_utils.get_export_saver_listener_state` and filesystem cleanup.

**Rust Mapping (Detailed)**
- Target crate/module: N/A.
- Rust public API surface: none.
- Data model mapping: export state proto to Rust struct if implemented.
- Feature gating: export-only.
- Integration points: export listener and checkpoint saver.

**Implementation Steps (Detailed)**
1. If Rust implements export hooks, add tests mirroring state entries and deletion.
2. Mock exporter to return bytes or dict.
3. Validate pruning when `max_to_keep` is 1.

**Tests (Detailed)**
- Python tests: `monolith/native_training/model_export/export_hooks_test.py`.
- Rust tests: none.
- Cross-language parity test: compare state entries after simulated checkpoints.

**Gaps / Notes**
- Uses real filesystem; may need cleanup to avoid test leakage.

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

### `monolith/native_training/model_export/export_state_utils.py`
<a id="monolith-native-training-model-export-export-state-utils-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 46
- Purpose/role: Reads/writes export state file (`ExportSaverListenerState`) containing `ServingModelState` proto.
- Key symbols/classes/functions: `get_export_saver_listener_state`, `overwrite_export_saver_listener_state`.
- External dependencies: TensorFlow gfile, protobuf text_format, `export_pb2`.
- Side effects: Reads and writes state files in export directory.

**Required Behavior (Detailed)**
- `_ExportSaverListenerStateFile = "ExportSaverListenerState"`.
- `get_export_saver_listener_state(export_dir_base)`:
  - Reads `<export_dir_base>/ExportSaverListenerState` as text proto.
  - If file missing, returns empty `ServingModelState`.
  - Parses using `text_format.Merge`.
- `overwrite_export_saver_listener_state(export_dir_base, state)`:
  - Ensures `export_dir_base` exists (`gfile.makedirs`).
  - Writes text proto to temp file `<filename>-tmp`.
  - Atomically renames temp to final file (overwrite=True).

**Rust Mapping (Detailed)**
- Target crate/module: N/A.
- Rust public API surface: if implemented, add export state read/write helpers.
- Data model mapping: `ServingModelState` proto in Rust.
- Feature gating: export-only.
- Integration points: export hooks.

**Implementation Steps (Detailed)**
1. Implement read/write of text-format proto in Rust (or switch to binary with parity notes).
2. Preserve temp-file rename semantics for atomic updates.

**Tests (Detailed)**
- Python tests: `export_state_utils_test.py`.
- Rust tests: add read/write round-trip tests.
- Cross-language parity test: compare serialized text output.

**Gaps / Notes**
- Uses text-format proto; any Rust implementation must match formatting if parity is required.

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

### `monolith/native_training/model_export/export_state_utils_test.py`
<a id="monolith-native-training-model-export-export-state-utils-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 36
- Purpose/role: Round-trip test for export state read/write.
- Key symbols/classes/functions: `ExportStateUtilsTest.test_basic`.
- External dependencies: `export_state_utils`, `export_pb2`, filesystem.
- Side effects: Writes state file under `TEST_TMPDIR`.

**Required Behavior (Detailed)**
- `test_basic`:
  - Creates `ServingModelState` with one entry (`export_dir="a"`, `global_step=1`).
  - Writes state to temp dir via `overwrite_export_saver_listener_state`.
  - Reads state back and asserts equality with original.
- Uses `unittest.TestCase`.

**Rust Mapping (Detailed)**
- Target crate/module: N/A.
- Rust public API surface: none.
- Data model mapping: ServingModelState proto.
- Feature gating: export-only.
- Integration points: export state utilities.

**Implementation Steps (Detailed)**
1. Implement Rust read/write helpers and add a round-trip test.
2. Ensure protobuf equality holds after text-format serialization.

**Tests (Detailed)**
- Python tests: `export_state_utils_test.py`.
- Rust tests: add round-trip test if implemented.
- Cross-language parity test: compare text serialization output.

**Gaps / Notes**
- Uses deprecated `assertEquals` (should be `assertEqual`).

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

### `monolith/native_training/model_export/export_utils.py`
<a id="monolith-native-training-model-export-export-utils-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 98
- Purpose/role: Helper for defining and invoking remote prediction signatures via `distributed_serving_ops.remote_predict`.
- Key symbols/classes/functions: `RemotePredictHelper`, `_get_tensor_signature_name`.
- External dependencies: TensorFlow, `nested_tensors`, `export_context`, `distributed_serving_ops`.
- Side effects: Registers SavedModel signatures with `ExportContext`.

**Required Behavior (Detailed)**
- `_get_tensor_signature_name(t)`:
  - Returns tensor name with ":" replaced by "_" (e.g., "foo:0" -> "foo_0").
- `RemotePredictHelper.__init__(name, input_tensors, remote_func)`:
  - Wraps inputs in `NestedTensors`, stores remote func, calls `_define_remote_func`.
- `_define_remote_func()`:
  - Creates placeholders matching flat input tensors (dtype/shape) with suffix `_remote_input_ph`.
  - Builds nested input structure from placeholders and calls `remote_func`.
  - Wraps outputs in `NestedTensors`.
  - Builds signature input/output dicts keyed by `_get_tensor_signature_name`.
  - Asserts no name conflicts (lengths match).
  - Registers signature in current `ExportContext` via `add_signature`.
- `call_remote_predict(model_name, input_tensors=None, old_model_name=None, task=0)`:
  - Uses provided `input_tensors` or original input tensors.
  - Calls `distributed_serving_ops.remote_predict` with signature name and I/O names.
  - Passes `output_types` from output tensor dtypes and `signature_name=self._name`.
  - Returns outputs in original nested structure.

**Rust Mapping (Detailed)**
- Target crate/module: N/A.
- Rust public API surface: if remote predict exists, implement a helper mirroring signature registration.
- Data model mapping: nested tensor structure + signature names.
- Feature gating: remote serving only.
- Integration points: distributed serving ops.

**Implementation Steps (Detailed)**
1. Implement nested tensor flattening and placeholder generation if Rust supports graph export.
2. Ensure signature name mapping replaces ":" with "_".
3. Provide remote predict wrapper matching arg ordering and output types.

**Tests (Detailed)**
- Python tests: `export_utils_test.py`.
- Rust tests: add unit test for signature name mapping and nested output reconstruction.
- Cross-language parity test: compare signature I/O names and remote_predict call args.

**Gaps / Notes**
- Relies on global export context; ensure one is active when constructing helper.

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

### `monolith/native_training/model_export/export_utils_test.py`
<a id="monolith-native-training-model-export-export-utils-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 43
- Purpose/role: Basic test for `RemotePredictHelper` signature definition and call path.
- Key symbols/classes/functions: `ExportUtilsTest.testBasic`.
- External dependencies: TensorFlow, `export_context`, `export_utils`.
- Side effects: Enters export mode (standalone).

**Required Behavior (Detailed)**
- `testBasic`:
  - Enters `export_context.enter_export_mode(EXPORT_MODE.STANDALONE)`.
  - Defines `remote_func(d)` returning `d["a"] * 3 + d["b"] * 4`.
  - Instantiates `RemotePredictHelper("test_func", {"a": tf.constant(1), "b": tf.constant(2)}, remote_func)`.
  - Calls `helper.call_remote_predict("model_name")`.
  - Asserts result is a `tf.Tensor`.
- Note: test intentionally only checks grammar due to missing TF Serving compilation.

**Rust Mapping (Detailed)**
- Target crate/module: N/A.
- Rust public API surface: none.
- Data model mapping: none.
- Feature gating: export-only.
- Integration points: remote predict.

**Implementation Steps (Detailed)**
1. If Rust implements RemotePredictHelper, add a similar smoke test.
2. Ensure export mode context is active for signature registration.

**Tests (Detailed)**
- Python tests: `export_utils_test.py`.
- Rust tests: none.
- Cross-language parity test: verify signature registration.

**Gaps / Notes**
- No actual remote serving is exercised.

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
