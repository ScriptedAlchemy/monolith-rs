<!--
Source: task/request.md
Lines: 4914-5209 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/barrier_ops.py`
<a id="monolith-native-training-barrier-ops-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 158
- Purpose/role: TF-based barrier primitive to block non-chief workers until chief releases barrier; includes SessionRunHook integration.
- Key symbols/classes/functions: `BarrierOp`, `BarrierHook`, `BarrierAlreadyPlacedError`.
- External dependencies: `tensorflow`, `absl.logging`, `threading`, `time`, `basic_restore_hook` (imported, unused).
- Side effects: creates TF variables/placeholders, sleeps, logs every 60 seconds.

**Required Behavior (Detailed)**
- `BarrierAlreadyPlacedError`: raised when attempting to place a barrier twice.
- `BarrierOp(capacity, is_chief=True, wait_seconds=1, name_prefix="default", barrier_callbacks=None)`:
  - Creates `barrier_var` bool vector of length `capacity`:
    - If `is_chief`: `LOCAL_VARIABLES`; else `VARIABLES`.
  - Creates index placeholder `_idx_ph`.
  - `_place_op` sets `barrier_var[idx] = True`; `_remove_op` sets `False`.
  - `barrier_placed_tensor` references element 0.
  - `barrier_op_action` string variable + placeholder, assign op.
  - Uses a threading lock for thread safety.
- `place_barrier(session, action="")`:
  - If barrier already placed (index 0 True), raises `BarrierAlreadyPlacedError`.
  - Sets barrier at index 0 and assigns action; runs callbacks `(action, session)`.
- `remove_barrier(session)`:
  - Clears barrier at index 0; no checks.
- `is_barrier_placed(session)`:
  - Returns `session.run(barrier_placed_tensor)`.
- `wait_until_barrier_removed(session, index)`:
  - Validates `index` in `(0, capacity)` else `ValueError`.
  - Sets barrier at `index`, reads action, runs callbacks.
  - Loops until `barrier_placed_tensor` becomes False, sleeping `wait_seconds` and logging every 60s.
  - Removes its own barrier index after release.
- `is_all_blocked`/`is_none_blocked`:
  - Reads `barrier_var` and checks count equals capacity or 0.
- `get_unblocked_indices`/`get_blocked_indices`:
  - Returns indices with False/True in barrier vector.
- `BarrierHook(index, barrier_op)`:
  - `before_run` requests `barrier_placed_tensor`.
  - `after_run`: if `index > 0` and barrier placed, calls `wait_until_barrier_removed`.
- Threading: lock guards state changes; waiting uses sleep polling.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/barrier.rs`.
- Rust public API surface: `Barrier` trait, `InProcessBarrier`, `PsBarrier` (async).
- Data model mapping: TF variable barrier → async barrier/coordination via PS; no SessionRunHook equivalent.
- Feature gating: none (used in distributed training).
- Integration points: `runner.rs`, `distributed_ps.rs`.

**Implementation Steps (Detailed)**
1. Decide whether a TF-variable barrier is needed in Rust (likely no).
2. Map barrier semantics to async barrier APIs (waiting, timeouts).
3. Provide hook-like integration if training loop expects per-step blocking.

**Tests (Detailed)**
- Python tests: `monolith/native_training/barrier_ops_test.py`.
- Rust tests: add async barrier tests for arrival/release semantics.
- Cross-language parity test: not applicable unless TF runtime added.

**Gaps / Notes**
- TF `basic_restore_hook` import unused.
- Python barrier uses polling on tensor 0; Rust barrier is event-based and may not mirror per-step hook semantics.

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

### `monolith/native_training/barrier_ops_test.py`
<a id="monolith-native-training-barrier-ops-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 104
- Purpose/role: Verifies BarrierOp placement/removal and BarrierHook blocking behavior.
- Key symbols/classes/functions: `BarrierOpsTest`.
- External dependencies: `tensorflow`, `monitored_session._HookedSession`.
- Side effects: spawns threads and TF sessions.

**Required Behavior (Detailed)**
- `test_basic`:
  - Place barrier; double place raises `BarrierAlreadyPlacedError`.
  - Remove barrier → `is_barrier_removed` true.
- `test_barrier_hook_not_blocked`:
  - Without barrier, hook does not block; global step reaches 5.
- `test_barrier_hook_blocked`:
  - Place barrier; worker thread blocks after 1 step.
  - Callback called with action, sets a variable to True.
  - Removing barrier allows training to finish; all barriers cleared.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/barrier.rs`.
- Rust public API surface: barrier trait/implementations.
- Data model mapping: no SessionRunHook analogue; tests should exercise async barrier directly.
- Feature gating: none.
- Integration points: training runner.

**Implementation Steps (Detailed)**
1. Add Rust tests for arrival/release semantics using in-process barrier.
2. Add integration test for PS barrier timeout if applicable.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: add async barrier tests for arrival/release semantics.
- Cross-language parity test: not applicable.

**Gaps / Notes**
- Rust barrier is async/event-based; no direct SessionRunHook equivalent.

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

### `monolith/native_training/basic_restore_hook.py`
<a id="monolith-native-training-basic-restore-hook-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 72
- Purpose/role: SessionRunHook that invokes listener callbacks around checkpoint restore (no actual restore logic).
- Key symbols/classes/functions: `CheckpointRestorerListener`, `CheckpointRestorerHook`.
- External dependencies: `tensorflow.python.training.session_run_hook`, `absl.logging`.
- Side effects: logs on creation and restore calls.

**Required Behavior (Detailed)**
- `CheckpointRestorerListener`:
  - Interface with `begin`, `before_restore(session)`, `after_restore(session)`, `end(session)`; all no-ops by default.
- `CheckpointRestorerHook(listeners=None)`:
  - Logs "Create CheckpointRestorerHook."
  - `begin()` calls `listener.begin()` for each listener.
  - `after_create_session(session, coord)` calls `_restore(session)`.
  - `_restore(session)`:
    - Logs "Calling checkpoint restorer listeners."
    - Calls `before_restore(session)` then `after_restore(session)` on each listener.
    - **No actual restore actions performed in hook**.
- No `end()` implementation in hook; listener `end()` is never invoked.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/hooks.rs`.
- Rust public API surface: `Hook` trait + `CheckpointHook` (save only).
- Data model mapping: listener callbacks map to hook lifecycle events.
- Feature gating: none.
- Integration points: training loop hook list.

**Implementation Steps (Detailed)**
1. Add a Rust hook that runs listener callbacks at session start.
2. If restore is implemented in Rust, wire callbacks before/after restore.
3. Ensure lifecycle ordering matches Python (begin → after_create_session → before/after restore).

**Tests (Detailed)**
- Python tests: `monolith/native_training/basic_restore_hook_test.py`.
- Rust tests: add hook lifecycle test in `monolith-training`.
- Cross-language parity test: compare callback ordering under identical steps.

**Gaps / Notes**
- Hook does not implement `end()`, so listener `end()` is never called.

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

### `monolith/native_training/basic_restore_hook_test.py`
<a id="monolith-native-training-basic-restore-hook-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 137
- Purpose/role: Verifies listener callbacks fire during `after_create_session` and do not repeat during training.
- Key symbols/classes/functions: `CountCheckpointRestorerListener`, `CountHook`, `CheckpointRestorerHookTest`.
- External dependencies: `tensorflow`, `session_run_hook`.
- Side effects: runs TF monitored sessions.

**Required Behavior (Detailed)**
- `test_restore_only_in_after_create_session`:
  - Listener `begin/before_restore/after_restore` called once at session creation.
  - Another hook receives `after_create_session` once before any `before_run/after_run`.
  - After training steps, listener counts remain unchanged; CountHook shows before_run/after_run/end increments.
- `test_two_listeners_with_restorer`:
  - Two listeners both receive begin/before/after restore once.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/hooks.rs`.
- Rust public API surface: hook lifecycle tests.
- Data model mapping: count callbacks during `on_start` or equivalent.
- Feature gating: none.
- Integration points: hook list execution order.

**Implementation Steps (Detailed)**
1. Add a Rust test that runs a hook list and validates callback order/counts.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: add CPU tensor tests for clipping behavior and immutability.
- Cross-language parity test: compare Rust outputs vs TF for fixed inputs.

**Gaps / Notes**
- GPU-only custom ops in Python; Rust needs equivalent or fallback path.

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

### `monolith/native_training/clip_ops.py`
<a id="monolith-native-training-clip-ops-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 80
- Purpose/role: Custom gradient clipping with Monolith GPU ops and optional fused path.
- Key symbols/classes/functions: `_global_norm`, `clip_by_global_norm`.
- External dependencies: `tensorflow`, `device_utils`, `gen_monolith_ops` (custom ops).
- Side effects: none (pure tensor ops), but relies on custom TF ops.

**Required Behavior (Detailed)**
- `_global_norm(t_list)`:
  - Returns `None` if list empty.
  - Uses `gen_monolith_ops.global_l2_reduce` to compute L2 sum, then `sqrt`.
- `clip_by_global_norm(t_list, clip_norm, use_norm=None)`:
  - Requires `t_list` to be a list; else raises `TypeError("t_list should be a list")`.
  - If `t_list` empty, returns `(t_list, 0)`.
  - If `use_norm` provided: returns `monolith_clip_by_global_norm(t_list, use_norm, clip_norm)` and `use_norm`.
  - If in GPU placement context: uses fused op `monolith_clip_by_global_norm_fused(t_list, clip_norm)` (returns list, norm).
  - Otherwise computes `global_norm` via:
    - `_global_norm` if GPU context, else `tf.linalg.global_norm`.
  - Applies `monolith_clip_by_global_norm(t_list, global_norm, clip_norm)` and returns `(list_clipped, global_norm)`.
- Expected semantics match `tf.clip_by_global_norm` including NaN on inf norms.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-optimizer/src` (or `monolith-tensor`).
- Rust public API surface: clip-by-global-norm helper in optimizer utilities.
- Data model mapping: custom TF ops → native tensor ops (Candle/TF runtime).
- Feature gating: GPU vs CPU paths should be explicit.
- Integration points: optimizer step and gradient pre-processing.

**Implementation Steps (Detailed)**
1. Implement global norm computation and clipping on tensor lists.
2. Provide GPU-optimized path or document as CPU-only if missing.
3. Ensure NaN propagation on inf norms matches TF.
4. Add compatibility test vs `tf.clip_by_global_norm`.

**Tests (Detailed)**
- Python tests: `monolith/native_training/clip_ops_test.py`.
- Rust tests: add numeric tests for clip_by_global_norm including NaN/inf cases.
- Cross-language parity test: compare outputs against TF for fixed inputs.

**Gaps / Notes**
- Custom ops (`monolith_clip_by_global_norm_*`) are not available in Rust.

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
