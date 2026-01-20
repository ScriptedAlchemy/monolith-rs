<!--
Source: task/request.md
Lines: 11465-11734 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/hooks/ckpt_info_test.py`
<a id="monolith-native-training-hooks-ckpt-info-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 45
- Purpose/role: Verifies that `FidSlotCountSaverListener` writes correct slot counts.
- Key symbols/classes/functions: `FidCountListener.test_basic`.
- External dependencies: TensorFlow, `hash_table_ops`, `ckpt_info`, `ckpt_info_pb2`.
- Side effects: Writes `ckpt.info-0` in temp dir.

**Required Behavior (Detailed)**
- Creates a hash table, assigns id 1, runs `before_save`.
- Reads `ckpt.info-0` and parses `CkptInfo`; `slot_counts[0] == 1`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests/ckpt_info_test.rs` (new).
- Rust public API surface: slot count saver listener.
- Feature gating: TF runtime required.

**Implementation Steps (Detailed)**
1. Build a small hash table and add one entry.
2. Invoke the saver listener and read output file.
3. Parse `CkptInfo` and assert slot_counts[0] == 1.

**Tests (Detailed)**
- Python tests: `monolith/native_training/hooks/ckpt_info_test.py`.
- Rust tests: `monolith-rs/crates/monolith-training/tests/ckpt_info_test.rs`.
- Cross-language parity test: compare text output files.

**Gaps / Notes**
- Slot extraction relies on `extract_slot_from_entry` custom op.

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

### `monolith/native_training/hooks/controller_hooks.py`
<a id="monolith-native-training-hooks-controller-hooks-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 170
- Purpose/role: Controller hooks for stop/save signaling, barrier coordination, and file-based action queries.
- Key symbols/classes/functions: `ControllerHook`, `StopHelper`, `QueryActionHook`.
- External dependencies: TensorFlow, `barrier_ops`, `controller_hooks_pb2`, `utils.ps_device`.
- Side effects: Writes/reads action files under model_dir; places/removes barriers; triggers save callback.

**Required Behavior (Detailed)**
- `ControllerHook`:
  - Creates local `control_var=[False, False]` on ps0 device if `num_ps>0`.
  - `stop_op` assigns True to index 0; `trigger_save_op` assigns True to index 1; `reset_trigger_save_op` assigns False.
  - `before_run` requests `control_var`.
  - `after_run`:
    - If stop flag set: place barrier (action `STOP_ACTION`), wait up to 30s for all workers blocked, then remove barrier.
    - If trigger_save flag set: reset flag and call `_trigger_save` callback if provided.
- `StopHelper`:
  - Barrier callback sets internal `_should_stop` on STOP action.
  - `create_stop_hook` returns a hook that calls `request_stop` when `_should_stop`.
- `QueryActionHook`:
  - Polls `<model_dir>/monolith_action` every `QUERY_INTERVAL` seconds in a background thread.
  - Parses `ControllerHooksProto` from text; on TRIGGER_SAVE runs `hook.trigger_save_op`, on STOP runs `hook.stop_op`.
  - Writes response to `<model_dir>/monolith_action_response` and deletes query file.
  - On parse error, writes error to response.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/hooks/controller_hooks.rs` (new).
- Rust public API surface: controller hook, stop helper, file-based action polling hook.
- Feature gating: TF runtime for SessionRunHook equivalents; polling logic can be generic.
- Integration points: barrier ops, checkpoint save trigger, model_dir actions.

**Implementation Steps (Detailed)**
1. Implement a control flag shared variable (or atomic state) to signal stop/save.
2. Add barrier coordination for STOP action and wait-until-blocked logic.
3. Implement file polling for `monolith_action` and action parsing.
4. Wire trigger-save callback to the training loop/hook.

**Tests (Detailed)**
- Python tests: `controller_hooks_test.py`.
- Rust tests: `monolith-rs/crates/monolith-training/tests/controller_hooks_test.rs` (new).
- Cross-language parity test: verify TRIGGER_SAVE and STOP paths behave as expected.

**Gaps / Notes**
- Python has a likely bug in `_write_resp` call with two args on unknown action; decide whether to fix or preserve behavior in Rust.

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

### `monolith/native_training/hooks/controller_hooks_test.py`
<a id="monolith-native-training-hooks-controller-hooks-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 82
- Purpose/role: Tests ControllerHook stop/save behavior and QueryActionHook file polling.
- Key symbols/classes/functions: `ControllerHookTest`, `QueryActionHookTest`.
- External dependencies: TensorFlow, `barrier_ops`, `controller_hooks`.
- Side effects: Creates action files under temp model_dir.

**Required Behavior (Detailed)**
- `testStop`:
  - Uses `StopHelper` + `BarrierOp` with callback; runs `stop_op` and ensures session stops after a subsequent run if needed.
- `testSave`:
  - Trigger save op should invoke `trigger_save` exactly once.
- `QueryActionHookTest.testStop`:
  - Writes `monolith_action` file with `action: TRIGGER_SAVE` and waits for processing.
  - Confirms `trigger_save` called once.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests/controller_hooks_test.rs` (new).
- Rust public API surface: controller hook and file polling hook.
- Feature gating: TF runtime if hooks are TF-based; otherwise simulate in test harness.

**Implementation Steps (Detailed)**
1. Implement a test harness to invoke stop/save flags and verify state transitions.
2. Add a file-based query test that writes `monolith_action` and checks callback execution.

**Tests (Detailed)**
- Python tests: `controller_hooks_test.py`.
- Rust tests: `monolith-rs/crates/monolith-training/tests/controller_hooks_test.rs`.
- Cross-language parity test: compare stop/save action handling.

**Gaps / Notes**
- Timing-based file polling tests may need retries or longer timeouts in Rust CI.

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

### `monolith/native_training/hooks/feature_engineering_hooks.py`
<a id="monolith-native-training-hooks-feature-engineering-hooks-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 99
- Purpose/role: Captures feature batches during training and dumps them as ExampleBatch protobuf files.
- Key symbols/classes/functions: `FeatureEngineeringSaveHook`.
- External dependencies: TensorFlow, `idl.matrix.proto.example_pb2` (ExampleBatch, FeatureListType).
- Side effects: Writes `.pb` files under `<model_dir>/features`.

**Required Behavior (Detailed)**
- `FeatureEngineeringSaveHook.__init__(config, nxt_elem, cap=100)`:
  - Stores config, next-element tensor, and buffer cap.
- `begin()`:
  - Initializes `_batch_list=[]` and `_steps=0`.
- `before_run`:
  - Increments step counter; returns `SessionRunArgs(nxt_elem)` after the first step (skips iterator init step).
- `after_run`:
  - Appends `run_values.results` to batch buffer; when buffer size reaches `cap`, calls `_save_features()` and clears buffer.
- `_save_features`:
  - Ensures `<model_dir>/features` exists.
  - Names output file as `chief_<uuid>.pb` for worker0, else `worker<index>_<uuid>.pb`.
  - Converts each batch dict into `ExampleBatch`:
    - Each key becomes a `named_feature_list` with type `INDIVIDUAL`.
    - RaggedTensorValue uses `to_list()`, ndarray uses `tolist()`.
    - If list entries are floats -> `float_list`, else `fid_v2_list`.
    - Sets `example_batch.batch_size` to len(lv).
  - Writes each serialized ExampleBatch preceded by two `<Q` headers: 0 (lagrange) and size.
- `end`:
  - Always attempts to save remaining batches (even empty list).

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/hooks/feature_engineering_hooks.rs` (new).
- Rust public API surface: session hook that records feature batches and writes ExampleBatch files.
- Feature gating: TF runtime for SessionRunHook integration; feature serialization can be shared.
- Integration points: dataset iterators and model_dir outputs.

**Implementation Steps (Detailed)**
1. Define a hook that buffers batch feature maps and serializes to ExampleBatch protos.
2. Implement ragged vs dense conversion and output naming conventions.
3. Write files with lagrange header + size + protobuf bytes.

**Tests (Detailed)**
- Python tests: none.
- Rust tests: add integration test with a small batch dict and validate file contents.
- Cross-language parity test: compare serialized ExampleBatch output for a fixed batch.

**Gaps / Notes**
- `end()` currently saves even when buffer is empty; decide whether to preserve this behavior in Rust.

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

### `monolith/native_training/hooks/hook_utils.py`
<a id="monolith-native-training-hooks-hook-utils-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 41
- Purpose/role: Thin wrappers to forward only before-save or after-save callbacks.
- Key symbols/classes/functions: `BeforeSaveListener`, `AfterSaveListener`.
- External dependencies: TensorFlow.
- Side effects: Delegates to wrapped listener.

**Required Behavior (Detailed)**
- `BeforeSaveListener`:
  - Stores a `CheckpointSaverListener` and only forwards `before_save`.
  - `__repr__` appends wrapped listener repr.
- `AfterSaveListener`:
  - Stores a listener and only forwards `after_save`.
  - `__repr__` appends wrapped listener repr.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/hooks/hook_utils.rs` (new).
- Rust public API surface: wrappers that forward only before/after save events.
- Feature gating: TF runtime only.

**Implementation Steps (Detailed)**
1. Implement wrapper types around saver listener traits.
2. Ensure only the intended callback is forwarded.

**Tests (Detailed)**
- Python tests: `hook_utils_test.py`.
- Rust tests: `monolith-rs/crates/monolith-training/tests/hook_utils_test.rs` (new).
- Cross-language parity test: verify callbacks fire only for intended phase.

**Gaps / Notes**
- Python allows calling non-forwarded method without error (it just inherits default); Rust should match behavior or document deviations.

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
