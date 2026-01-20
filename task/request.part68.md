<!--
Source: task/request.md
Lines: 15600-15813 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/metric/metric_hook_test.py`
<a id="monolith-native-training-metric-metric-hook-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 189
- Purpose/role: Tests for `Tf2ProfilerHook` and `FileMetricHook` behaviors.
- Key symbols/classes/functions: `Tf2ProfilerHookTest`, `FileMetricHookTest`.
- External dependencies: TensorFlow, `os`, `time`, `json`, `random`, `datetime`.
- Side effects: Writes profiling data under `TEST_TMPDIR` and file metrics under `$HOME/tmp/file_metric_hook`.

**Required Behavior (Detailed)**
- `Tf2ProfilerHookTest`:
  - `setUp`:
    - `logdir = $TEST_TMPDIR/<test_name>`.
    - `filepattern = logdir/plugins/profile/*`.
    - Creates a graph with global_step and train_op (`assign_add` by 1).
  - `_count_files()` returns count of files matching pattern.
  - `test_steps`:
    - Hook: `Tf2ProfilerHook(logdir, init_step_range=[0,10], save_steps=50)`.
    - Runs one step in `SingularMonitoredSession`.
    - Expects exactly 1 profile file.
  - `test_multiple_steps_1`:
    - Hook with `save_steps=30`, runs 30 steps with 0.15s sleep.
    - Expects 1 file (profile only at 0~9).
  - `test_multiple_steps_2`:
    - Same hook, runs 31 steps with 0.15s sleep.
    - Expects 2 files (0~9 and step 30).
  - `test_secs_1`:
    - Hook with `save_secs=1`, runs 10 steps with 0.15s sleep.
    - Expects at least 1 file.
  - `test_secs_2`:
    - Hook with `save_secs=3`, runs 21 steps with 0.15s sleep.
    - Expects at least 2 files.
- `FileMetricHookTest`:
  - `setUpClass`:
    - `model_name='test_model'`, `base_name=$HOME/tmp/file_metric_hook`.
    - Creates `FileMetricHook(worker_id=0, key_fn=vepfs_key_fn, layout_fn=vepfs_layout_fn, batch_size=8, partition_size=32)`.
  - `tearDownClass`:
    - Calls `hook.end(None)` to flush/close.
    - For each of last 8 days, asserts:
      - date directory exists under `base_name/model_name/<YYYYMMDD>/worker_0/`.
      - exactly 2 files exist; each has 32 lines.
  - `test_vepfs_key_fn`:
    - Asserts path formatting for fixed data.
  - `test_vepfs_layout_fn`:
    - Asserts formatted string with predict/label JSON and fallback `gid`.
  - `test_after_run`:
    - Builds `RunValue` wrapper with `results={'deep_insight_op':[json.dumps(rv)]}`.
    - For last 8 days, sends 64 records/day with random predict/label values.
    - Calls `hook.after_run` to enqueue metrics; file writing validated in `tearDownClass`.

**Rust Mapping (Detailed)**
- Target crate/module: N/A (TF profiler and FileMetricHook not implemented in Rust).
- Rust public API surface: if implemented, provide equivalent tests for profiling triggers and file partitioning.
- Data model mapping: file output format must match `vepfs_layout_fn` and `vepfs_key_fn`.
- Feature gating: profiling/Kafka/file outputs should be optional.
- Integration points: Rust training hook system.

**Implementation Steps (Detailed)**
1. If Rust supports profiling hooks, add tests for step/second trigger behavior.
2. If file output hook is implemented, port these tests with deterministic data (no randomness).
3. Ensure file partitioning at 32 lines and 2 files per day for 64 records.

**Tests (Detailed)**
- Python tests: `monolith/native_training/metric/metric_hook_test.py`.
- Rust tests: N/A until hooks are implemented.
- Cross-language parity test: compare file outputs and profile dump counts if available.

**Gaps / Notes**
- Tests rely on filesystem and time sleeps; may be flaky or slow.

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

### `monolith/native_training/metric/utils.py`
<a id="monolith-native-training-metric-utils-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 104
- Purpose/role: Convenience wrapper to emit Deep Insight metrics using custom TF ops (v1 or v2).
- Key symbols/classes/functions: `write_deep_insight`.
- External dependencies: TensorFlow, `deep_insight_ops`, Python `logging`.
- Side effects: Calls custom ops that create clients and emit metrics; logs when disabling.

**Required Behavior (Detailed)**
- `write_deep_insight(features, sample_ratio, model_name, labels=None, preds=None, target=None, targets=None, labels_list=None, preds_list=None, sample_rates_list=None, extra_fields_keys=[], enable_deep_insight_metrics=True, enable_kafka_metrics=False, dump_filename=None)`:
  - Requires `features["req_time"]`; if missing:
    - Logs "Disabling deep_insight because req_time is absent".
    - Returns `tf.no_op()`.
  - `is_fake = enable_kafka_metrics or (dump_filename is not None and len(dump_filename) > 0)`.
  - Creates client: `deep_insight_ops.deep_insight_client(enable_deep_insight_metrics, is_fake, dump_filename)`.
  - `req_times = reshape(features["req_time"], [-1])`.
  - **Single-target path** (`not targets`):
    - `uids = reshape(features["uid"], [-1])`.
    - `sample_rates = reshape(features["sample_rate"], [-1])`.
    - Calls `deep_insight_ops.write_deep_insight` with `labels`, `preds`, `model_name`, `target`, `sample_ratio`, `return_msgs=is_fake`.
  - **Multi-target path** (`targets` truthy):
    - `labels = stack([label if rank==1 else reshape(label, (-1,)) for label in labels_list if label is not None])`.
    - `preds = stack([pred if rank==1 else reshape(pred, (-1,)) for pred in preds_list if pred is not None])`.
    - `sample_rates_list` handling:
      - If falsy: uses `features["sample_rate"]` reshaped to [-1] and repeats `len(targets)` times.
      - If list/tuple: reshapes each to rank 1; filters out None.
      - Else raises `Exception("sample_rates_list error!")`.
    - `sample_rates = stack(sample_rates_list)`.
    - Ensures `"uid"` in `extra_fields_keys` (mutates list default).
    - Builds `extra_fields_values` by reshaping each `features[key]` to [-1].
    - Calls `deep_insight_ops.write_deep_insight_v2` with `targets`, `extra_fields_*`, `return_msgs=is_fake`.
  - Returns the op tensor from deep_insight ops.
- Error cases:
  - Missing `uid`/`sample_rate`/extra fields -> `KeyError`.
  - Empty `labels_list`/`preds_list` -> `tf.stack` error.
  - `sample_rates_list` non-list and truthy -> raises generic `Exception`.
- Mutability note: `extra_fields_keys` default list is mutated when adding `"uid"`.
- No threading; deterministic aside from op behavior.

**Rust Mapping (Detailed)**
- Target crate/module: N/A (custom TF ops not bound in Rust).
- Rust public API surface: optional wrapper around TF runtime deep insight ops.
- Data model mapping: feature tensors, string targets, list-of-tensors for multi-target.
- Feature gating: TF runtime + custom ops only.
- Integration points: metrics pipeline in training.

**Implementation Steps (Detailed)**
1. Add TF runtime bindings for deep insight ops if needed.
2. Mirror single-target vs multi-target branching.
3. Preserve `is_fake` semantics and `return_msgs`.
4. Avoid mutable default pitfalls if porting (but keep behavior if parity requires).

**Tests (Detailed)**
- Python tests: `monolith/native_training/metric/utils_test.py`.
- Rust tests: N/A until ops are bound.
- Cross-language parity test: verify that v1/v2 calls receive the same tensors/flags.

**Gaps / Notes**
- `extra_fields_keys` uses a mutable default list; repeated calls may accumulate `"uid"`.

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

### `monolith/native_training/metric/utils_test.py`
<a id="monolith-native-training-metric-utils-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 50
- Purpose/role: Tests basic call path for `utils.write_deep_insight` using a mocked op.
- Key symbols/classes/functions: `DeepInsightTest.test_basic`.
- External dependencies: TensorFlow, `unittest.mock`.
- Side effects: None (deep insight op is mocked).

**Required Behavior (Detailed)**
- `test_basic`:
  - Patches `deep_insight_ops.write_deep_insight` and sets a side-effect function.
  - `fake_call` evaluates `uids` tensor in a session and asserts it equals `[1,2,3]`.
  - Constructs `features` with `uid`, `req_time`, `sample_rate` tensors.
  - Creates `labels`, `preds`, `model_name`, `target`.
  - Calls `utils.write_deep_insight(...)`.
  - Note: Call uses positional arguments in a non-obvious order; still exercises `uids` extraction.
- `__main__` disables eager execution and runs `tf.test.main()`.

**Rust Mapping (Detailed)**
- Target crate/module: N/A.
- Rust public API surface: none.
- Data model mapping: none.
- Feature gating: TF runtime only.
- Integration points: deep insight wrapper tests if implemented.

**Implementation Steps (Detailed)**
1. If deep insight ops are bound in Rust, add a unit test to ensure `uid` extraction and reshape behavior.
2. Prefer explicit keyword arguments to avoid positional mis-ordering.

**Tests (Detailed)**
- Python tests: `monolith/native_training/metric/utils_test.py`.
- Rust tests: N/A until implementation exists.
- Cross-language parity test: verify tensor shapes/values passed to op.

**Gaps / Notes**
- The test passes arguments positionally in a confusing order relative to the function signature; keep behavior but consider fixing in Python if allowed.

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
