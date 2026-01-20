<!--
Source: task/request.md
Lines: 15384-15599 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/metric/kafka_utils.py`
<a id="monolith-native-training-metric-kafka-utils-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 119
- Purpose/role: Simple Kafka producer wrapper with a background thread + queue; tracks send counters.
- Key symbols/classes/functions: `KProducer`, `KProducer.send`, `KProducer.close`.
- External dependencies: `kafka.KafkaProducer`, `queue.Queue`, `threading.Thread/RLock`, `absl.logging`, `time`.
- Side effects: Starts a background thread on init; sends messages to Kafka.

**Required Behavior (Detailed)**
- `KProducer.__init__(brokers, topic)`:
  - Stores `brokers` and `topic`.
  - Creates `KafkaProducer(bootstrap_servers=brokers)`.
  - Initializes `_lock` (RLock), `_has_stopped=False`, `_msg_queue=Queue()`.
  - Initializes counters `_total`, `_success`, `_failed` to 0.
  - Spawns background thread targeting `_poll` and starts it.
- `send(msgs)`:
  - If `msgs` is `None` or empty, returns immediately.
  - If `msgs` is `str` or `bytes`, wraps into a list.
  - Else filters iterable to only non-`None` entries with `len(msg) > 0`.
  - If resulting list is non-empty:
    - Logs first message up to 10 times via `logging.log_first_n(INFO, msgs[0], n=10)`.
    - Increments `_total` by `len(msgs)`.
    - Enqueues the list into `_msg_queue`.
  - No encoding/conversion; message passed to KafkaProducer as-is.
- `_poll()` (background thread):
  - Loop: `msg_batch = _msg_queue.get(timeout=1)`.
  - On any exception (e.g., timeout), checks `_has_stopped` under lock:
    - If stopped: break; else continue.
  - If `msg_batch` non-empty: sends each message via `producer.send(topic, msg)` and attaches callbacks:
    - `_send_success` for success, `_send_failed` for error.
  - After processing a batch, exits if `_has_stopped` is True.
- `total()`, `success()`, `failed()`:
  - Return counters; not synchronized across threads (may race).
- `_flush()`:
  - Asserts `_has_stopped` is True.
  - Drains `_msg_queue` (timeout=1) until empty or exception.
  - Sends queued messages with callbacks (same as `_poll`).
- `close()`:
  - Sets `_has_stopped=True` under lock.
  - Joins background thread.
  - Calls `_flush()` and then `producer.close(timeout=1)`.
  - Logs warnings on any exception.
- `_send_success(...)`: increments `_success`.
- `_send_failed(...)`: sleeps 2 seconds, logs warning, increments `_failed`.
- Threading/concurrency: background thread + queue; counters are not locked.
- Determinism: none; dependent on Kafka/network.

**Rust Mapping (Detailed)**
- Target crate/module: N/A (no Rust Kafka wrapper).
- Rust public API surface: optional Kafka producer wrapper with background worker.
- Data model mapping: messages as `Vec<u8>` or `String`; track counters.
- Feature gating: requires a Rust Kafka client (e.g., `rdkafka`).
- Integration points: metrics emission pipeline.

**Implementation Steps (Detailed)**
1. Choose Kafka client crate and implement a background worker with channel/queue.
2. Mirror `send` behavior (filtering, logging first message, counters).
3. Implement graceful shutdown (stop flag, join thread, flush queue).
4. Provide success/failure counters and expose them.

**Tests (Detailed)**
- Python tests: none.
- Rust tests: add unit tests with a mocked producer or test broker.
- Cross-language parity test: compare counters and send filtering behavior.

**Gaps / Notes**
- Python implementation is not thread-safe for counters; preserve semantics unless explicitly improved.

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

### `monolith/native_training/metric/metric_hook.py`
<a id="monolith-native-training-metric-metric-hook-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 563
- Purpose/role: Collection of TensorFlow Estimator hooks for metrics, profiling, Kafka/file logging, and telemetry.
- Key symbols/classes/functions: `ThroughputMetricHook`, `StepLossMetricHook`, `CustomMetricHook`, `Tf2ProfilerHook`, `ByteCCLTelemetryHook`, `NVProfilerHook`, `KafkaMetricHook`, `FileMetricHook`, `WriteOnlyFileAndStat`, helper functions (`default_parse_fn`, `default_layout_fn`, `vepfs_layout_fn`, `vepfs_key_fn`).
- External dependencies: TensorFlow Estimator hooks, TF profiler, BytePS telemetry, Kafka (via `KProducer`), `tf.io.gfile`, `absl.flags/logging`, `alert_manager`, `alert_pb2`.
- Side effects: Registers exit hook via import (`exit_hook`), starts background threads, writes to Kafka/files, may start TF profiler server on port 6666.

**Required Behavior (Detailed)**
- Module globals:
  - `FLAGS = flags.FLAGS` used by `Tf2ProfilerHook`.
  - Importing `exit_hook` executes signal/atexit registration for metrics.
- `ThroughputMetricHook`:
  - `__init__(model_name, start_time_secs, cluster_type="stable", run_every_n_secs=30)`:
    - Initializes counters and `self._mcli = cli.get_cli(utils.get_metric_prefix())`.
    - If alert manager exists, creates `AlertProto` and registers rules with prefix.
  - `begin()`: sets `self._global_step_tensor = tf.compat.v1.train.get_global_step()`.
  - `before_run(run_context)`:
    - On first step, reads `global_step` via `session.run`.
    - Records `emit_time` (int seconds).
    - If `start_time_secs` provided, emits timer `run_start_elapsed_time.all` with tags `{model_name, cluster_type}`.
    - Returns `SessionRunArgs({"global_step": global_step_tensor})`.
  - `after_run(run_context, run_values)`:
    - If elapsed wall time >= `run_every_n_secs`, emits:
      - `run_steps.all` counter (step interval).
      - `run_steps_elapsed_time.all` timer (elapsed_time / step_interval).
    - Updates emit step/time. (No guard against `step_interval == 0`.)
- `StepLossMetricHook`:
  - `__init__(loss_tensor)` stores tensor and mcli.
  - `before_run`: requests loss tensor.
  - `after_run`: emits `step_loss` store with loss value.
- `CustomMetricHook`:
  - `__init__(metric_tensors)`:
    - Validates each tensor is scalar (rank 0) and dtype in `{tf.float32, tf.int32}`.
    - Raises `ValueError` if invalid or if metric list empty.
  - `before_run`: requests all metric tensors.
  - `after_run`: emits each metric as float via `emit_store`.
- `Tf2ProfilerHook`:
  - `__init__(logdir, init_step_range, save_steps=None, save_secs=None, options=None)`:
    - Validates `end_step > start_step` when provided.
    - Sets `delta = end_step - start_step` or default 10.
    - If `save_steps` provided and `<= delta`, raises `ValueError`.
    - Creates `SecondOrStepTimer(every_steps=save_steps, every_secs=save_secs)`.
  - `begin()`:
    - If `FLAGS.enable_sync_training` tries `tf.profiler.experimental.server.start(6666)`; logs warning on failure.
  - `before_run`:
    - If profiling, creates `_pywrap_traceme.TraceMe("TraceContext", graph_type="train", step_num=current_step)` for step-time graph fix.
    - Returns `SessionRunArgs(fetches=None)`.
  - `after_run`:
    - Increments `current_step`.
    - Stops TraceMe if active.
    - If `start_step` is None, defers profiling to `current_step + 500` with default delta.
    - Stops profiling when `current_step >= end_step`.
    - If timer triggers, starts profiling and sets new `[start_step, end_step)` window.
  - `end(sess)`: stops profiling if active.
  - `_start_profiling()`: `tf.profiler.experimental.start(logdir, options)`; ignores `AlreadyExistsError`.
  - `_stop_profiling()`: calls `tf.profiler.experimental.stop()`; ignores `UnavailableError`.
- `ByteCCLTelemetryHook`:
  - Requires global step tensor (`training_util._get_or_create_global_step_read()`), else `RuntimeError`.
  - Logs telemetry every `interval` steps by sampling BytePS ops on rank 0.
  - `_log_telemetry()` filters ops containing `alltoall` or first 3 `PushPull` entries.
- `NVProfilerHook`:
  - Subclass of `Tf2ProfilerHook` with `logdir=None`.
  - Loads `libcudart.so` and calls `cudaProfilerStart/Stop`.
- `KafkaMetricHook` (singleton):
  - Uses `KAFKA_BROKER_LIST` and `KAFKA_TOPIC_NAME` env vars to create `KProducer`.
  - `__init__`: loads `deep_insight_op` from TF collection if not provided; stores as tensor dict.
  - `after_run`: sends `deep_insight_op` messages to Kafka if any.
  - `end`: closes producer, logs success/failed counts.
- Helper functions:
  - `default_parse_fn`: JSON-decodes strings/bytes; otherwise returns input.
  - `default_layout_fn`: returns string or JSON dump; falls back to `repr` on error.
  - `vepfs_layout_fn`: formats deep insight record as `req_time;gid;uid;predict_scores;labels`.
  - `vepfs_key_fn`: builds path `base/model_name/date/worker_{id}`.
- `WriteOnlyFileAndStat`:
  - Holds buffered output; rotates partitions after `partition_size` lines (default 1e6).
  - Uses `tf.io.gfile` to write `part_XXXXXX.{file_ext}` under `key` directory.
  - `write()` buffers formatted strings; `flush()` writes and rotates; `close()` closes stream.
  - `is_available()` returns True if updated within last 24 hours.
  - Note: uses `List`/`Dict` typing annotations without importing them (potential NameError at runtime).
- `FileMetricHook` (singleton):
  - Initializes from `deep_insight_op` collection if not provided.
  - Requires `key_fn` for routing items; if `None`, `_send` will fail when called.
  - Spawns background thread on first `after_run`.
  - Enqueues messages (handles list/tuple/np.ndarray or scalar).
  - `_send` parses items, writes to per-key `WriteOnlyFileAndStat`, and cleans up inactive files every 10 minutes.
  - `end` waits for queue to drain, stops thread, closes open files.
- Threading/concurrency: multiple background threads; queue for metrics, RLock in file writer.
- Determinism: depends on timing, Kafka/network, filesystem.
- Logging/metrics: uses `mcli.emit_*`, absl logging, Kafka/file outputs.

**Rust Mapping (Detailed)**
- Target crate/module: N/A (TF Estimator hooks and Kafka/file hooks not present in Rust).
- Rust public API surface: if needed, add a training hooks module with metrics, profiling, and output sinks.
- Data model mapping: map TF hooks to Rust training loop callbacks; map `deep_insight_op` outputs to Rust equivalents.
- Feature gating: Kafka and profiler hooks should be optional (feature flags).
- Integration points: training loop, metrics client, optional BytePS/collective telemetry.

**Implementation Steps (Detailed)**
1. Decide which hooks are needed in Rust training (throughput, loss, custom metrics).
2. Implement throughput/loss hooks as callbacks in Rust training loop.
3. Provide profiling hooks only if profiling support exists (TF2/NV profilers likely N/A).
4. Implement Kafka/File output sinks if required; reuse Rust Kafka + filesystem abstractions.
5. Match environment-variable configuration for Kafka (`KAFKA_BROKER_LIST`, `KAFKA_TOPIC_NAME`).
6. Preserve thread/queue behavior and file partitioning semantics.
7. Add tests for validation errors (CustomMetricHook), file rotation, and queue draining.

**Tests (Detailed)**
- Python tests: `monolith/native_training/metric/metric_hook_test.py`.
- Rust tests: N/A until hooks exist.
- Cross-language parity test: validate emitted metrics names/tags and file output formatting.

**Gaps / Notes**
- `List`/`Dict` are used in annotations without import; may require adding `from typing import List, Dict` in Python for runtime use.
- `FileMetricHook` will fail if `key_fn` is not provided; ensure callers pass `vepfs_key_fn` or a custom function.

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
