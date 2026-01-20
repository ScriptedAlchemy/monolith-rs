<!--
Source: task/request.md
Lines: 8143-8388 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/device_utils.py`
<a id="monolith-native-training-device-utils-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 231
- Purpose/role: Device placement utilities for training/serving, including GPU gating, device functions, and MPI/PS placement logic.
- Key symbols/classes/functions: `enable_gpu_training`, `disable_gpu_training`, `is_gpu_training`, `get_visible_gpus`, `default_device_fn`, `maybe_device_if_allowed`, `within_placement_context_of`, `get_device_fn`, `input_device_fn`, `model_device_fn`, `serving_input_device_fn`, `skip_device`.
- External dependencies: TensorFlow DeviceSpec, `device_setter`, MPI rank helper `get_mpi_rank`, flags (`num_ps`, `enable_gpu_training`, `enable_sync_training`, `is_local`).
- Side effects: global `_GPU_PLACEMENT_ALLOWED` flag; influences device placement for ops.

**Required Behavior (Detailed)**
- GPU training flag:
  - `_GPU_PLACEMENT_ALLOWED` default False; `enable_gpu_training()` sets True; `disable_gpu_training()` sets False; `is_gpu_training()` returns it.
- `get_visible_gpus(local_rank, processes_per_gpu=1)`:
  - Ensures `processes_per_gpu` is int >= 1; returns string of `local_rank / processes_per_gpu` as GPU index.
- `_device_rule(device_name)`:
  - Returns `/device:CPU:0` when `device_name` is empty.
  - If assigned GPU but `_GPU_PLACEMENT_ALLOWED` is False or device type empty, merges with default CPU while keeping job/task/replica.
- `skip_device(op)`:
  - Returns True for summary ops (`Write*`, `*Summary`) or string `Const` ops.
- `default_device_fn(op)`:
  - Returns CPU for skipped ops; otherwise applies `_device_rule` to op device.
- `maybe_device_if_allowed(device_name)`:
  - Context manager that forces device via `_device_rule` to prevent unintended GPU placement.
- Placement context helpers:
  - `_FakeOp` and `within_placement_context_of(device_name)` check current placement via graph `_apply_device_functions`.
- `get_device_fn(cluster=None, task=None)`:
  - Determines MPI mode via `OMPI_COMM_WORLD_LOCAL_RANK`.
  - Chooses GPU vs CPU based on `FLAGS.enable_gpu_training` or `_GPU_PLACEMENT_ALLOWED`.
  - If sync training + MPI + PS: builds device spec for chief/worker based on rank and returns custom `_device_fn` that merges with op device.
  - If sync training but no PS: returns `default_device_fn`.
  - If async (no sync training):
    - Returns None for local mode or missing cluster/task.
    - Else uses `tf.compat.v1.train.replica_device_setter` with `ps_tasks=FLAGS.num_ps` and standard PS ops.
- `input_device_fn(op)`:
  - In MPI+PS+sync training returns `/job:chief|worker/replica:0/task:<idx>/device:CPU:0`, else CPU.
- `model_device_fn(op)`:
  - Similar to `_device_fn` but for model scope; uses GPU if enabled, else CPU; respects op.device and `_class` attr.
- `serving_input_device_fn(op)`:
  - Uses op.device if set, else CPU.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/device`.
- Rust public API surface: device placement utilities and device function factories.
- Data model mapping: TF DeviceSpec strings → Rust device strings used by TF runtime bindings.
- Feature gating: GPU placement gate, MPI/PS sync training, replica device setter behavior.
- Integration points: training config, session creation, input pipelines.

**Implementation Steps (Detailed)**
1. Implement GPU gating and visible GPU computation.
2. Implement device rule merging logic with default CPU and job/task retention.
3. Provide Rust equivalents of `get_device_fn`/`input_device_fn`/`model_device_fn` for sync/async modes.
4. Mirror skip-device rules for summary ops and string const.
5. Add placement-context helper or document unsupported if TF internals unavailable.

**Tests (Detailed)**
- Python tests: `device_utils_test.py`.
- Rust tests: unit tests for device rules, GPU gating, and MPI/PS device fn outputs.
- Cross-language parity test: compare device string outputs under fixed flag/env combinations.

**Gaps / Notes**
- Depends on TF internal device functions; Rust may need to mimic device strings rather than enforcing in graph.

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

### `monolith/native_training/device_utils_test.py`
<a id="monolith-native-training-device-utils-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 104
- Purpose/role: Tests device placement rules and GPU gating in `device_utils`.
- Key symbols/classes/functions: `DeviceUtilsTest` and methods `test_basic`, `test_cpu_only`, `test_str_context`, `test_str_nested_contexts`, `test_cpu_device_merge`, `test_gpu_device_merge`, `test_process_gpu_map`.
- External dependencies: TensorFlow, `device_utils`.
- Side effects: none.

**Required Behavior (Detailed)**
- `test_basic`: default device function places constants on `/device:CPU:0`.
- `test_cpu_only`: when GPU training disabled, explicit GPU device request is overridden to CPU.
- `test_str_context`: with GPU enabled, bare constants default to CPU, `tf.device("GPU:0")` forces GPU:0.
- `test_str_nested_contexts`: nested device contexts maintain correct placement for CPU/GPU overrides.
- `test_cpu_device_merge`: with GPU disabled, device job/task merged with CPU; `within_placement_context_of` reports CPU.
- `test_gpu_device_merge`: with GPU enabled, device job/task merged with GPU; `maybe_device_if_allowed` forces GPU:1 placement and context checks.
- `test_process_gpu_map`: `get_visible_gpus` returns expected indices for local_rank/processes_per_gpu combinations.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests`.
- Rust public API surface: device_utils functions.
- Data model mapping: device strings matching TF conventions.
- Feature gating: GPU training toggle.
- Integration points: training device placement.

**Implementation Steps (Detailed)**
1. Add Rust tests to assert device string outputs for each scenario.
2. Verify GPU gating overrides explicit GPU placement when disabled.
3. Validate visible GPU mapping logic.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: `device_utils_test.rs`.
- Cross-language parity test: compare device string outputs and placement context behavior.

**Gaps / Notes**
- Python tests rely on TF device placement; Rust tests may need to compare string outputs rather than actual TF graph placement.

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

### `monolith/native_training/distribute/distributed_dataset.py`
<a id="monolith-native-training-distribute-distributed-dataset-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 81
- Purpose/role: Builds a dynamic sharding dataset that expands glob patterns on demand using shared queues and a TF session-backed generator.
- Key symbols/classes/functions: `create_dynamic_sharding_dataset`.
- External dependencies: `str_queue.StrQueue`, `session_hooks.get_current_session`, `utils.ps_device`, `native_task_context`.
- Side effects: creates shared queues on PS0 or host; uses TF session to dequeue filenames.

**Required Behavior (Detailed)**
- `create_dynamic_sharding_dataset(glob_patterns, name)`:
  - Creates two shared string queues:
    - `glob_patterns_queue`: seeded with glob patterns.
    - `filenames_queue`: auto-enqueue filenames by expanding patterns.
  - Chooses device on PS0 if `num_ps > 0`, else default device.
  - `glob_pattern()` (tf.function): dequeues a pattern; if not out_of_range, calls `tf.io.matching_files`; else returns `""` and out_of_range.
  - `filenames_queue.dequeue()` returns `(filename_bytes, out_of_range)`.
  - `filename_generator()` runs dequeue via current session; raises `StopIteration` on out_of_range; else decodes bytes to string.
  - Builds `dataset_ops.MapDataset` over a dummy infinite dataset; maps to `tf.py_function(filename_generator)` with `preserve_cardinality=False`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/src`.
- Rust public API surface: dynamic sharding dataset builder for file patterns.
- Data model mapping: file pattern → stream of file paths.
- Feature gating: requires session hooks/queues or Rust equivalents.
- Integration points: `datasets.py` uses this for dynamic sharding.

**Implementation Steps (Detailed)**
1. Implement a Rust dynamic sharding iterator that expands patterns lazily.
2. Support shared queue semantics for multi-worker coordination (or document limitation).
3. Ensure out_of_range yields end of stream and map preserves non-cardinality.

**Tests (Detailed)**
- Python tests: `distributed_dataset_test.py`.
- Rust tests: unit tests for pattern expansion order and termination.
- Cross-language parity test: compare file lists produced for a given glob set.

**Gaps / Notes**
- Relies on TF session and custom `StrQueue`; Rust needs a coordinated queue for distributed use.

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

### `monolith/native_training/distribute/distributed_dataset_test.py`
<a id="monolith-native-training-distribute-distributed-dataset-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 124
- Purpose/role: Tests dynamic sharding dataset expansion, EOF handling, composition with TextLineDataset, and iterator save/restore.
- Key symbols/classes/functions: `DynamicShardingDatasetTest`, `gen_test_files`, `testBasic`, `testEof`, `testWithOtherDataset`, `testSaveRestore`.
- External dependencies: TensorFlow, `distributed_dataset.create_dynamic_sharding_dataset`, `session_hooks.SetCurrentSessionHook`.
- Side effects: writes temp files under `TEST_TMPDIR` and saves/loads iterator checkpoints.

**Required Behavior (Detailed)**
- `gen_test_files(files_dir)`:
  - Creates files `a_0.txt`..`e_1.txt` with two lines each: `a.0.0`, `a.0.1`, etc.
- `setUp`:
  - Uses `TEST_TMPDIR` and creates data dir + files if missing.
  - Builds glob patterns `a_*.txt`..`e_*.txt`.
- `get_test_session()`:
  - Returns `SingularMonitoredSession` with `SetCurrentSessionHook`.
- `testBasic`:
  - Reads 10 filenames from dynamic sharding dataset; expects ordered list of `a_0..e_1` full paths.
- `testEof`:
  - With empty patterns, iterator should raise `OutOfRangeError`; verifies dependent op does not mutate variable `v`.
- `testWithOtherDataset`:
  - `filename_dataset.flat_map(TextLineDataset)` yields lines; first three lines are `a.0.0`, `a.0.1`, `a.1.0`.
- `testSaveRestore`:
  - Creates saveable iterator; reads `a.0.0`, saves; reads `a.0.1`, restores; next read is still `a.0.1`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/tests`.
- Rust public API surface: dynamic sharding dataset + iterator save/restore (if supported).
- Data model mapping: file list → dataset → line dataset.
- Feature gating: iterator save/restore may depend on TF runtime.
- Integration points: `distributed_dataset` implementation.

**Implementation Steps (Detailed)**
1. Implement Rust tests that generate temp files and validate ordered filename emission.
2. Verify EOF behavior for empty patterns.
3. Test composition with line reader and save/restore semantics.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: `distributed_dataset_test.rs` with tempdir fixtures.
- Cross-language parity test: compare file order and resume position after restore.

**Gaps / Notes**
- `saveable` iterator behavior is TF-specific; Rust may need explicit checkpointing support.

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
