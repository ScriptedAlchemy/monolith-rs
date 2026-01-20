<!--
Source: task/request.md
Lines: 2954-3163 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/core/auto_checkpoint_feed_hook.py`
<a id="monolith-core-auto-checkpoint-feed-hook-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 376
- Purpose/role: TPU infeed/outfeed SessionRunHook with thread-managed queues, TPU init/shutdown, and end-of-stream stopping signals.
- Key symbols/classes/functions: `PeriodicLogger`, `_SIGNAL`, `_OpQueueContext`, `_OpSignalOnceQueueContext`, `TPUInfeedOutfeedSessionWithEndOfStreamHandlingHook`.
- External dependencies: `threading`, `time`, `os`, `six.moves.queue/xrange`, `tensorflow.compat.v1`, `config_pb2`, `tpu_compilation_result`, `summary_ops_v2` (plus implicit `ops`, `tpu_config`, `session_support`).
- Side effects: spawns threads, initializes/shuts down TPU, runs TF sessions, reads env vars, logs via TF logging.

**Required Behavior (Detailed)**
- `PeriodicLogger(seconds)`:
  - `log()` emits TF log only if elapsed time exceeds `seconds`.
- `_SIGNAL`:
  - `NEXT_BATCH = -1`, `STOP = -2` (negative values reserved for control).
- `_OpQueueContext`:
  - Starts a daemon thread running `target(self, *args)`.
  - `send_next_batch_signal(iterations)` enqueues integer iterations.
  - `read_iteration_counts()` yields iterations until `_SIGNAL.STOP` then returns.
  - `join()` logs, sends STOP, joins thread.
- `_OpSignalOnceQueueContext`:
  - Only allows the first `send_next_batch_signal`; subsequent calls ignored.
- `TPUInfeedOutfeedSessionWithEndOfStreamHandlingHook`:
  - `__init__`:
    - Stores enqueue/dequeue ops, rendezvous, master, session config, init ops, outfeed cadence.
    - Reads embedding config from `ctx` if present.
    - Sets `_should_initialize_tpu=False` when model-parallel + per-host input broadcast; else true.
    - Sets `stopping_signal=False`.
  - `_create_or_get_iterations_per_loop()`:
    - Uses graph collection `tpu_estimator_iterations_per_loop`.
    - If >1 existing var → `RuntimeError("Multiple iterations_per_loop_var in collection.")`.
    - Else creates resource variable in scope `tpu_estimator`, int32 scalar, non-trainable, in LOCAL_VARIABLES, colocated with global step.
  - `begin()`:
    - Records `_iterations_per_loop_var`.
    - Adds TPU shutdown op to `_finalize_ops` if `_should_initialize_tpu`.
    - Adds summary writer init ops to `_init_ops` and flush ops to `_finalize_ops`.
  - `_run_infeed(queue_ctx, session)`:
    - Optional sleep (`initial_infeed_sleep_secs`).
    - If `run_infeed_loop_on_coordinator`: for each iteration signal, runs `enqueue_ops` `steps` times.
    - Else runs `enqueue_ops` once per signal.
  - `_run_outfeed(queue_ctx, session)`:
    - Runs `dequeue_ops` every `outfeed_every_n_steps`.
    - If output includes `_USER_PROVIDED_SIGNAL_NAME`, expects a dict containing `stopping`.
    - When first stopping signal seen, sets `stopping_signals=True` and later flips `self.stopping_signal=True`.
  - `_assertCompilationSucceeded(result, coord)`:
    - Parses TPU compilation proto; if `status_error_message` set → log error + `coord.request_stop()`.
  - `after_create_session()`:
    - If `_should_initialize_tpu`, runs `tf.tpu.initialize_system` in a fresh graph/session.
    - Runs `_init_ops` with 30 minute timeout.
    - If `TPU_SPLIT_COMPILE_AND_EXECUTE=1`, runs `tpu_compile_op` and asserts compilation success.
    - Starts infeed/outfeed controller threads.
    - If `TF_TPU_WATCHDOG_TIMEOUT>0`, starts worker watchdog.
  - `before_run()`:
    - If `stopping_signal` is set, raises `tf.errors.OutOfRangeError`.
    - Reads `iterations_per_loop`, sends signals to infeed/outfeed controllers.
  - `end()`:
    - Joins infeed/outfeed threads; calls rendezvous `record_done`; runs finalize ops (flush + shutdown).
  - `get_stopping_signals_and_name(features)`:
    - If `_USER_PROVIDED_SIGNAL_NAME` in `features`, uses `tf.tpu.cross_replica_sum` to compute `stopping` boolean.
    - Returns `(stopping_signals, _USER_PROVIDED_SIGNAL_NAME)`.
- Env vars:
  - `TPU_SPLIT_COMPILE_AND_EXECUTE=1` triggers separate compile step.
  - `TF_TPU_WATCHDOG_TIMEOUT` enables worker watchdog.
- Threading: two daemon threads (infeed/outfeed) controlled via queues.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-tf/src` (TPU runtime integration).
- Rust public API surface: no equivalent; would need a TPU session hook abstraction.
- Data model mapping: TF session/run hooks have no direct Rust equivalent.
- Feature gating: only relevant if TF-runtime backend is enabled.
- Integration points: training loop / input pipeline in `monolith-training`.

**Implementation Steps (Detailed)**
1. Decide if TPU SessionRunHook is in-scope for Rust TF backend.
2. If in-scope, implement infeed/outfeed controller threads and queue signaling.
3. Expose env-var toggles for compile/execute split and watchdog timeout.
4. Provide cross-replica stopping signal handling for outfeed.

**Tests (Detailed)**
- Python tests: none dedicated (covered indirectly by TPU training flows).
- Rust tests: none yet; would need integration tests with TF runtime.
- Cross-language parity test: not applicable unless TF runtime implemented in Rust.

**Gaps / Notes**
- Missing imports in Python (`ops`, `tpu_config`, `session_support`) are referenced but not imported.
- Rust has no TPU SessionRunHook analog; full parity requires TF runtime + hook infrastructure.

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

### `monolith/core/base_embedding_host_call.py`
<a id="monolith-core-base-embedding-host-call-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 643
- Purpose/role: Host call logic for embedding tasks, including TPU variable caching and DeepInsight metrics.
- Key symbols/classes/functions: `BaseEmbeddingHostCall`, `update_tpu_variables_ops`, `generate_host_call_hook`.
- External dependencies: `tensorflow.compat.v1`, `tensorflow`, `BaseHostCall`, `ReplicatedVariable`.
- Side effects: creates TPU variables on all replicas; writes TF summaries; computes AUC.

**Required Behavior (Detailed)**
- Constants define metric names and TPU variable names.
- `TPUVariableRestoreHook` runs assign op after session creation.
- `BaseEmbeddingHostCall.__init__`:
  - Stores flags for host call, deepinsight, scalar metrics, caching mode.
  - Extracts `context` unless `cpu_test`.
  - Creates TPU variables if `enable_host_call` and caching enabled.
- `_create_tpu_var`:
  - Creates per-replica TPU variables across hosts with zeros, adds to `TPU_VAR` collection.
  - Wraps in `ReplicatedVariable`; registers restore hooks.
- `_compute_new_value(base, delta, update_offset)`:
  - Pads `delta` to base length, then `tf.roll` by offset, then adds to base.
- `update_tpu_variables_ops(...)`:
  - Clears TPU vars when `global_step % host_call_every_n_steps == 1`.
  - Writes labels/preds/uid_buckets and req_times/sample_rates into TPU vars at computed offsets.
  - Updates accumulated counter only after tensor updates.
- `record_summary_tpu_variables()` collects TPU vars into host call tensors.
- `record_summary_tensor` filters based on `enable_host_call_scalar_metrics` and deprecated metric names.
- `_write_summary_ops`:
  - Filters by uid_bucket and stopping_signals; computes AUC; writes scalar summaries and serialized tensors.
- `generate_host_call_hook()`:
  - Calls `compress_tensors()` and returns either `_host_call` or `_host_call_with_tpu` plus tensor list; or `None` if disabled.
  - Host call writes summaries under `output_dir/host_call` with `tf2.summary`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/src/base_embedding_host_call.rs`.
- Rust public API surface: host call builder with optional TPU caching mode.
- Data model mapping: summary writer and AUC metric implementation.

**Implementation Steps (Detailed)**
1. Recreate TPU variable caching logic (or provide equivalent buffer) with host-call step windowing.
2. Implement summary writing and AUC computation matching TF semantics.
3. Mirror filtering by UID buckets and stopping signals.
4. Provide compatibility with TPU/CPU test modes.

**Tests (Detailed)**
- Python tests: `monolith/core/base_embedding_host_call_test.py`
- Rust tests: unit tests for `_compute_new_value` and host call output shapes.

**Gaps / Notes**
- Full parity requires TF TPU runtime; Rust may need to stub or bridge this functionality.

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

### `monolith/core/base_embedding_host_call_test.py`
<a id="monolith-core-base-embedding-host-call-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 77
- Purpose/role: Tests `_compute_new_value` in BaseEmbeddingHostCall.
- Key symbols/classes/functions: `BaseEmbeddingHostCallTest.test_compute_new_value`.
- External dependencies: `tensorflow.compat.v1`.
- Side effects: none.

**Required Behavior (Detailed)**
- Constructs `BaseEmbeddingHostCall` with `enable_host_call=False` and `context=None`.
- Verifies `_compute_new_value` behavior with different offsets.
- Uses TF session to evaluate results and compares to expected tensors.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/tests/base_embedding_host_call.rs`.
- Rust public API surface: `_compute_new_value` equivalent.

**Implementation Steps (Detailed)**
1. Port `_compute_new_value` logic in Rust and add unit tests for offsets.

**Tests (Detailed)**
- Python tests: this file
- Rust tests: same assertions.

**Gaps / Notes**
- Needs tensor ops; may require a small tensor wrapper or TF binding in Rust tests.

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
