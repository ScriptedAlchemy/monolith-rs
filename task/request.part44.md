<!--
Source: task/request.md
Lines: 10118-10249 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/estimator_mode_test.py`
<a id="monolith-native-training-estimator-mode-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 417
- Purpose/role: End-to-end integration tests for multiple distributed modes (CPU, sparse+ dense GPU, full GPU) by launching the training binary with various env/config permutations.
- Key symbols/classes/functions: `DistributedTrainTest`, `_run_test`, `run_cpu`, `sparse_dense_run`, `full_gpu_run`.
- External dependencies: TensorFlow, `RunnerConfig`, training binary `monolith/native_training/tasks/sparse_dense_gpu/model`, `gen_input_file`, `MultiHeadModel`, `test_util`.
- Side effects: Creates temp dataset files, spawns multiple processes (including mpirun), sets many env vars, writes logs, deletes temp dirs.

**Required Behavior (Detailed)**
- `setUpClass`:
  - Generates dataset file via `gen_input_file` and creates symlinks for suffixes 0..9.
  - Updates `FLAGS.dataset_input_patterns` to include `{INT(0,99)}`.
- `find_free_port(count)`:
  - Finds `count` available local ports (no reuse).
- `_run_test(...)`:
  - Creates `cur_modir` under test tmp dir; removes existing.
  - Builds `args_tmpl` list for the training binary with flags:
    - mode=train, model_dir, num_ps/workers, uuid, dataset flags, discovery settings, timeouts, metrics disable, dataservice toggle, cluster type.
  - Populates MLP_* env vars per role via `fill_host_env`.
  - Allocates ports for PS/worker/dsworker/dispatcher and sets env accordingly.
  - `start_process`:
    - For `use_mpi_run=True`, writes a hostfile and uses `mpirun` with Horovod-related env exports.
    - Else, spawns subprocess per role with `MLP_ROLE`, `MLP_ROLE_INDEX`, `MLP_PORT`, and `MLP_SSH_PORT` envs, writing logs to files.
  - Starts dispatcher, dsworker, ps, worker processes.
  - `wait_for_process` enforces timeouts; may terminate on timeout when `ignore_timeout=True`.
  - Cleans up log files and removes `cur_modir`.
- `run_cpu(...)`:
  - Skips if GPU is available; runs CPU cases with `enable_gpu_training=False`, `enable_sync_training=False` and embedding prefetch/postpush flags.
- `sparse_dense_run(...)`:
  - Requires GPU; uses MPI run and sets sync training flags, partial sync, params_override, and dataset service.
- `full_gpu_run(...)`:
  - Requires GPU; uses MPI run with `enable_sync_training`, `reorder_fids_in_data_pipeline`, `filter_type=probabilistic_filter`, `enable_async_optimize=False`.
- Test variants:
  - CPU tests `test_cpu0..3` vary `enable_fused_layout` and `use_native_multi_hash_table`.
  - Sparse+Dense GPU tests `test_sparse_dense0..3` vary layout and native hash table.
  - Full GPU tests `test_full_gpu_0..3` vary layout and native hash table.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests/estimator_mode_test.rs` (new) or CI scripts.
- Rust public API surface: distributed training CLI entrypoint and env-based cluster discovery.
- Feature gating: GPU and MPI required; tests should be behind `gpu`/`mpi` feature flags and skipped in CI by default.
- Integration points: training binary CLI, dataset service, Horovod/BytePS integration.

**Implementation Steps (Detailed)**
1. Implement or stub the Rust training binary to accept similar CLI flags.
2. Add integration tests that spawn subprocesses with env roles for PS/worker/dsworker/dispatcher.
3. Add MPI-based test harness if Horovod/BytePS parity is required.
4. Ensure temp dirs and logs are cleaned up even on failure.

**Tests (Detailed)**
- Python tests: `monolith/native_training/estimator_mode_test.py`.
- Rust tests: integration tests in `monolith-rs/crates/monolith-training/tests/` or CI scripts.
- Cross-language parity test: compare training completion and exit codes across CPU/GPU modes.

**Gaps / Notes**
- Heavy integration tests require external binaries and GPU; likely to be skipped in Rust CI.
- Port allocation is fragile; may need reserved port ranges.

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

### `monolith/native_training/estimator_test.py`
<a id="monolith-native-training-estimator-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 112
- Purpose/role: Local-mode Estimator smoke tests for train/eval/predict/export/import flow.
- Key symbols/classes/functions: `EstimatorTrainTest`, `get_saved_model_path`.
- External dependencies: TensorFlow, `RunnerConfig`, `TestFFMModel`, `generate_ffm_example`, `import_saved_model`.
- Side effects: Writes checkpoints and exported models under temp dirs; performs inference loops.

**Required Behavior (Detailed)**
- `setUpClass`:
  - Removes existing model_dir if present.
  - Sets model params: deep insight disabled, batch size 64, export dir base, `shared_embedding=True`.
  - Creates `RunnerConfig(is_local=True, num_ps=0, model_dir=..., use_native_multi_hash_table=False)`.
- `train/eval/predict`:
  - Instantiate `Estimator` and call the respective method (steps=10 for train/eval).
- `export_saved_model`:
  - Calls `Estimator.export_saved_model()` with defaults.
- `import_saved_model`:
  - Uses latest saved model dir from `export_base`.
  - Runs inference through `import_saved_model` context for 10 iterations, with generated FFM examples.
- `test_local`:
  - Runs train, eval, predict, export, and import in sequence.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests/estimator_test.rs` (new).
- Rust public API surface: local Estimator train/eval/predict/export/import.
- Feature gating: SavedModel export/import requires `tf-runtime`; otherwise stub or skip.
- Integration points: training data generation, model definition, export path handling.

**Implementation Steps (Detailed)**
1. Implement a local-only Estimator test in Rust that runs train/eval/predict with a simple model.
2. Add SavedModel export/import tests behind `tf-runtime` feature.
3. Ensure temp dirs are cleaned up after tests.

**Tests (Detailed)**
- Python tests: `monolith/native_training/estimator_test.py`.
- Rust tests: `monolith-rs/crates/monolith-training/tests/estimator_test.rs`.
- Cross-language parity test: compare export/import outputs on fixed inputs.

**Gaps / Notes**
- Python import_saved_model uses TF sessions and custom ops; Rust parity likely requires TF runtime.

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
