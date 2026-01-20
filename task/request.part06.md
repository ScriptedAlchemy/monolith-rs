<!--
Source: task/request.md
Lines: 11759-15684 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/estimator_dist_test.py`
<a id="monolith-native-training-estimator-dist-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 166
- Purpose/role: Integration test for distributed training/eval using TF_CONFIG-style discovery and multi-process PS/worker setup.
- Key symbols/classes/functions: `EstimatorTrainTest`, `get_cluster`, `get_free_port`.
- External dependencies: `tensorflow`, `RunnerConfig`, `TestFFMModel`, `TfConfigServiceDiscovery`, `Estimator`.
- Side effects: Spawns multiple processes, binds local ports, writes checkpoints under tmp.

**Required Behavior (Detailed)**
- `get_free_port()`:
  - Binds a local socket on port 0 to find an available port; closes socket and returns port.
- `get_cluster(ps_num, worker_num)`:
  - Returns dict with `ps`, `worker`, and `chief` addresses on free ports (workers exclude chief).
- `EstimatorTrainTest.setUpClass`:
  - Removes existing `model_dir` if present.
  - Creates `TestFFMModel` params with deep insight disabled and batch size 64.
- `EstimatorTrainTest.train()`:
  - Spawns `ps_num` PS processes and `worker_num` worker/chief processes.
  - Each process builds `TF_CONFIG`-like dict, uses `TfConfigServiceDiscovery`, constructs `RunnerConfig`, and calls `Estimator.train(steps=10)`.
  - Waits for all processes; asserts exitcode 0 for each.
- `EstimatorTrainTest.eval()`:
  - Same as train but calls `Estimator.evaluate(steps=10)`.
- `test_dist`:
  - Runs `train()` then `eval()`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests/estimator_dist_test.rs` (new).
- Rust public API surface: distributed training harness and config discovery equivalent.
- Data model mapping: distributed cluster config (ps/worker/chief), runner config, model params.
- Feature gating: likely `tf-runtime` or `distributed` feature; skip if distributed stack not available.
- Integration points: service discovery and process orchestration.

**Implementation Steps (Detailed)**
1. Implement a Rust integration test that spawns multiple processes (or threads) for PS/worker roles.
2. Provide a discovery config equivalent to `TfConfigServiceDiscovery` and ensure `Estimator` can use it.
3. Run short train/eval steps and assert clean exit.
4. Add timeouts and cleanup for spawned processes and temp directories.

**Tests (Detailed)**
- Python tests: `monolith/native_training/estimator_dist_test.py`.
- Rust tests: `monolith-rs/crates/monolith-training/tests/estimator_dist_test.rs` (integration).
- Cross-language parity test: verify training/eval complete under equivalent cluster topology.

**Gaps / Notes**
- Uses real multi-process TF; Rust currently lacks distributed PS/worker runtime.
- Port binding is fragile; consider deterministic port assignment for CI.

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

### `monolith/native_training/feature.py`
<a id="monolith-native-training-feature-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 663
- Purpose/role: Defines feature slots/columns, embedding table interfaces, and embedding slice fusion logic for Monolith feature pipelines.
- Key symbols/classes/functions: `FeatureEmbTable`, `FeatureSlotConfig`, `FeatureSlot`, `FeatureColumn`, `FeatureFactory`, `DummyFeatureEmbTable`, `DummyFeatureFactory`, `_FeatureFactoryFusionHelper`, `create_embedding_slices`, `EmbeddingFeatureEmbTable`, `FeatureFactoryFromEmbeddings`, `EmbeddingLayoutFactory`.
- External dependencies: TensorFlow, `entry`, `embedding_combiners`, `distribution_ops`, `device_utils`, `ragged_utils`, `embedding_hash_table_pb2`, `prefetch_queue`, `is_exporting`.
- Side effects: Uses env var `MONOLITH_GPU_FEATURE_FACTORY_FUSION_LEVEL`; adds tensors to TF collection `monolith_reduced_embs`.

**Required Behavior (Detailed)**
- Constants:
  - `_FEATURE_STRAT_END_KEY = "{}:{}_{}"` used to key embedding slices by feature name and slice bounds.
  - `DEFAULT_EXPIRE_TIME = 36500` (days).
- `FeatureEmbTable` (abstract):
  - `add_feature_slice(segment, learning_rate_fn)` and `set_feature_metadata(feature_name, combiner)` are no-ops in base.
  - `embedding_lookup(feature_name, start, end)` is abstract.
- `FeatureSlice`: NamedTuple with `feature_slot`, `start`, `end`.
- `FeatureSlotConfig`:
  - Defaults for bias/default vector configs using `entry` builders; default expire time and occurrence threshold.
  - `__post_init__` sets `name` to `slot_id` string if not provided.
- `FeatureSlot`:
  - Holds table/config, current dim size, and registered feature columns.
  - If `has_bias` true, creates a bias slice of dim 1 using bias configs.
  - `add_feature_slice`:
    - Applies defaults for initializer/optimizer/compressor/learning_rate_fn.
    - Creates `EntryConfig.Segment` via `entry.CombineAsSegment` and registers with table.
    - Returns `FeatureSlice(start, end)` and updates `_current_dim_size`.
  - Registers feature columns via `_add_feature_column`, and updates table metadata.
  - `get_feature_columns`, `get_bias_slice`, `slot` (int of name), `name`.
- `FeatureColumn`:
  - Factory helpers: `reduce_sum`, `reduce_mean`, `first_n(seq_length)`.
  - `embedding_lookup(s)` asserts slice belongs to slot and delegates to table.
  - `get_all_embeddings_concat()` returns full embedding tensor for gradients (start/end None).
  - `get_all_embedding_slices()` returns per-slice tensors for this feature name.
  - `get_bias()` returns bias slice embeddings.
  - `set_size_tensor(row_lengths)`:
    - Only for `FirstN` combiner; builds boolean mask `[B, max_seq_length]` from row_lengths and stores as int32 `size_tensor`.
- `FeatureFactory` (abstract):
  - Manages `slot_to_occurrence_threshold` and `slot_to_expire_time`.
  - `apply_gradients` default raises `NotImplementedError`.
- `DummyFeatureEmbTable` (config collection):
  - `add_feature_slice` auto-infers `learning_rate_fn` from optimizer config:
    - If optimizer has `warmup_steps > 0`, uses `PolynomialDecay` from 0.0 to `learning_rate` over warmup_steps.
    - Else uses `opt_config.learning_rate` value directly.
  - `embedding_lookup` builds placeholders and combines via combiner; respects fixed batch size.
  - `get_table_config` merges slices via `_merge_slices` and returns `TableConfig` with:
    - `slice_configs` merged
    - `feature_names`
    - `unmerged_slice_dims` (original slice sizes)
    - `hashtable_config`
    - `feature_to_combiners`.
  - `_merge_slices` merges adjacent slices when proto config (excluding dim_size) and `learning_rate_fn` string match; sums dim_size.
- `DummyFeatureFactory`:
  - Ensures unique table name; registers slot thresholds/expire times by slot_id.
  - `apply_gradients` returns `tf.no_op()`.
  - `get_table_name_to_table_config` errors if a table has no slices.
- `EmbeddingFeatureEmbTable`:
  - Wraps actual embeddings and embedding_slices; returns full embedding when start/end None; otherwise uses `_FEATURE_STRAT_END_KEY`.
- `_FeatureFactoryFusionHelper`:
  - Collects ragged rows, value_rowids, embeddings, batch_size, and slice_dims.
  - `reduce_and_split`: CPU scatter_nd reduce and split; adds reduced tensor to collection.
  - `fused_reduce_and_split`: uses `distribution_ops.fused_reduce_sum_and_split` on CPU.
  - `fused_reduce_then_split`: GPU `distribution_ops.fused_reduce_and_split_gpu` and then manual split mapping.
- `create_embedding_slices(...)`:
  - For each feature:
    - If combiner is `ReduceSum`, uses helper for fused reduce+split.
    - Else uses combiner + `tf.split` on combined embeddings.
  - Chooses fusion path:
    - If not exporting and within GPU placement and `MONOLITH_GPU_FEATURE_FACTORY_FUSION_LEVEL==1` -> `fused_reduce_then_split`.
    - Else if not exporting -> `reduce_and_split` (CPU) or `fused_reduce_and_split` (CPU fused) depending on placement.
    - If exporting -> `reduce_and_split`.
  - Constructs `embedding_slices` dict keyed by name/start/end.
- `FeatureFactoryFromEmbeddings`: builds `EmbeddingFeatureEmbTable` from `name_to_embeddings` and `name_to_embedding_slices`.
- `EmbeddingLayoutFactory`:
  - Uses `PartitionedHashTable` to apply gradients with layout embeddings and optional async push.
  - `get_layout` and `flattened_layout` expose layout tensors.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/src/feature.rs` plus embedding combiner logic in `monolith-rs/crates/monolith-layers` and hash table config in `monolith-rs/crates/monolith-hash-table`.
- Rust public API surface:
  - `FeatureSlot`, `FeatureSlice`, `FeatureColumn` equivalents in `monolith-core`.
  - Combiner types (`ReduceSum`, `ReduceMean`, `FirstN`) in `monolith-layers`.
  - `create_embedding_slices` and fusion helpers in a new `monolith-training` or `monolith-layers` module.
- Data model mapping: represent ragged inputs as `(values, row_splits)` and carry slice dimensions explicitly.
- Feature gating: GPU fused ops require TF runtime or custom kernels; Candle backend should use CPU reduce+split.
- Integration points: `feature_utils`, embedding lookup, and hash table update paths.

**Implementation Steps (Detailed)**
1. Implement FeatureSlot/FeatureColumn config layering in Rust, mapping to existing `monolith-core` feature structs.
2. Implement `DummyFeatureEmbTable` and `DummyFeatureFactory` for config collection and tests.
3. Implement `create_embedding_slices` with reduce/split logic; add optional fused path when TF runtime is available.
4. Preserve learning rate warmup logic in `DummyFeatureEmbTable.add_feature_slice`.
5. Add `EmbeddingLayoutFactory` wrapper around Rust hash table gradient application (or stub if hash table not available).
6. Add tests matching Python expectations for slice merging, bias, combiner selection, and fused behavior.

**Tests (Detailed)**
- Python tests: `monolith/native_training/feature_test.py`.
- Rust tests: add `monolith-rs/crates/monolith-core/tests/feature_test.rs` and/or `monolith-rs/crates/monolith-layers/tests/feature_factory_test.rs`.
- Cross-language parity test: compare slice configs (serialized proto), embedding slice outputs, and combiners.

**Gaps / Notes**
- Many operations rely on TF ragged tensors and custom fused ops; Rust backend must choose between TF runtime or native reductions.
- The `_FEATURE_STRAT_END_KEY` key format must match exactly to ensure lookup parity.

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

### `monolith/native_training/feature_test.py`
<a id="monolith-native-training-feature-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 266
- Purpose/role: Tests for feature slot/column config collection, slice merging, and embedding slice creation (including fused paths and FirstN behavior).
- Key symbols/classes/functions: `CollectingConfigTest`, `EmbeddingTest`.
- External dependencies: TensorFlow, `entry`, `embedding_combiners`, `feature`, `learning_rate_functions`, `embedding_hash_table_pb2`, protobuf `text_format`.
- Side effects: None beyond TensorFlow graph execution.

**Required Behavior (Detailed)**
- `CollectingConfigTest.test_basic`:
  - Dummy table + segment dim_size=5 with sgd; reduce_sum combiner; embedding_lookup produces placeholder shape `[4, 5]`.
- `test_basic_with_seq_features`:
  - FirstN(10) combiner -> embedding_lookup placeholder shape `[4, 10, 5]`.
- `test_info`:
  - Adds segments with adagrad warmup and sgd; two adjacent sgd slices with same learning_rate_fn should merge to dim_size=4.
  - Ensures learning_rate_fn for warmup slice is a `LearningRateFunction`.
  - `feature_names` list contains `feature1`.
- `test_factory`:
  - DummyFeatureFactory creates slot and feature columns; table config includes feature names and slice dim_size=5.
- `test_factory_with_seq_features`:
  - FirstN combiners stored in `feature_to_combiners` map; verifies mapping.
- `test_factory_with_slot_occurrence_threshold`:
  - Factory stores occurrence thresholds keyed by slot_id.
- `test_factory_with_applying_gradients`:
  - Dummy factory apply_gradients accepts grads and returns no-op.
- `test_bias`:
  - Slot with `has_bias=True` exposes bias lookup without errors.
- `EmbeddingTest.test_factory`:
  - Uses `create_embedding_slices` with ReduceSum; lookup returns `[[1],[2]]` for ragged ids.
- `test_factory_with_seq_features`:
  - FirstN(2) returns sequence embeddings `[[[1],[3]], [[5],[7]]]`.
- `test_fused_factory`:
  - ReduceSum with ragged splits producing zeros in empty rows; verifies slice outputs for each slice.
- `test_fused_factory_with_seq_features_larger_than_max_seq_length`:
  - FirstN(2) truncates rows longer than max_seq_length; verifies outputs.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/tests/feature_test.rs` and/or `monolith-rs/crates/monolith-layers/tests/feature_factory_test.rs` (new).
- Rust public API surface: `FeatureSlot`, `FeatureColumn`, `DummyFeatureFactory`, `create_embedding_slices`, combiners.
- Data model mapping: ragged inputs as values + row_splits; test outputs should match Python arrays.
- Feature gating: fused GPU paths behind TF runtime or CUDA feature; CPU paths always available.
- Integration points: `entry` config builder and learning rate functions.

**Implementation Steps (Detailed)**
1. Port each test case with deterministic tensors.
2. Validate slice merging behavior using serialized proto bytes for segments.
3. Add tests for FirstN shape and truncation behavior.
4. Ensure `DummyFeatureFactory` tracks occurrence thresholds and bias slice creation.

**Tests (Detailed)**
- Python tests: `monolith/native_training/feature_test.py`.
- Rust tests: add parity tests under `monolith-rs/crates/monolith-core/tests`.
- Cross-language parity test: compare outputs and merged slice configs with Python reference.

**Gaps / Notes**
- Fused GPU behavior depends on custom ops; Rust tests should skip if TF runtime is unavailable.

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

### `monolith/native_training/feature_utils.py`
<a id="monolith-native-training-feature-utils-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 419
- Purpose/role: Applies gradients to dense variables and embedding tables with optional clipping, Horovod/BytePS allreduce, and async embedding updates.
- Key symbols/classes/functions: `allreduce_cond`, `GradClipType`, `_gen_norm_warmup`, `apply_gradients_with_var_optimizer`, `apply_gradients`.
- External dependencies: TensorFlow, `clip_ops`, `distribution_ops.gen_distribution_ops`, `device_utils`, `feature`, `NativeContext`, Horovod/BytePS (optional).
- Side effects: Reads env vars, performs allreduce, writes TF summaries, updates global step, mutates globals `control_ops` and `dense_opt_ops`.

**Required Behavior (Detailed)**
- Env flags read at import time:
  - `MONOLITH_WITH_HOROVOD`, `MONOLITH_WITH_BYTEPS`, `MONOLITH_WITH_BYTEPS_ALLREDUCE`, `MONOLITH_WITH_ALLREDUCE_FUSION`, `MONOLITH_WITH_ALLREDUCE_FP16`, `MONOLITH_SKIP_ALLREDUCE`.
  - If Horovod enabled, imports `horovod.tensorflow` and compression classes.
- `allreduce_cond(grads, scale=1)`:
  - Selects BytePS or Horovod compression (FP16 vs None) based on envs.
  - Filters `None` grads, allreduces only non-None grads, then maps results back into original positions.
  - Fusion modes:
    - `one`: uses `monolith_aligned_flat_concat` + allreduce + `monolith_aligned_flat_split`.
    - `grouped`: uses `hvd.grouped_allreduce` (not supported with BytePS).
    - `multi`: raises `RuntimeError` (dropped).
    - default: allreduces each grad individually with Average op.
- `GradClipType` enum: `ClipByNorm`, `ClipByGlobalNorm`, `ClipByValue`, `ClipByDenseAndSparse`, `NoClip`.
- `_gen_norm_warmup(clip_norm, global_step_var, warmup_step)`:
  - Returns `clip_norm` scaled linearly from 0 to 1 over `warmup_step` using `tf.cond`.
- `apply_gradients_with_var_optimizer(...)`:
  - Computes grads for dense variables + embedding tensors.
  - For fused layout, replaces missing grads with zeros.
  - Splits dense vs sparse grads and optionally applies UE conditional gradient check.
  - Supports clip by global norm (dense/sparse), value, or per-tensor norm; optional sparse warmup.
  - Defers global norm clipping to a scale factor when using GPU + allreduce (fused with later kernels).
  - Optionally writes gradient/variable histograms and norms to summaries.
  - Dense grads optionally allreduced and L2 weight-decayed.
  - Applies dense grads via custom per-variable optimizer or shared `var_opt` (async via `ctx.add_async_function`).
  - Applies embedding grads via `ctx.apply_embedding_gradients` (on CPU) with optional scale.
  - Increments `global_step` after optimize ops with control dependencies.
- `apply_gradients(...)`:
  - Similar flow for layout-based embeddings (`ctx.layout_factory.flattened_layout()`) with simpler clipping logic.
  - If no dense variables, still increments global_step.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/feature_utils.rs` (new) and `monolith-rs/crates/monolith-optimizer` for optimizer integration.
- Rust public API surface: gradient application helpers for dense + embedding params, clip modes, and allreduce hooks.
- Data model mapping: Candle tensors for dense grads; embedding grads routed through hash table/update API.
- Feature gating: Horovod/BytePS allreduce under `tf-runtime` or `distributed` feature; default backend uses local grads only.
- Integration points: `NativeContext`, `EmbeddingLayoutFactory`, async function manager, and training loop.

**Implementation Steps (Detailed)**
1. Implement `GradClipType` enum and clipping helpers in Rust.
2. Implement global norm computation and optional warmup scaling.
3. Implement dense vs sparse gradient separation (embedding tensors tracked separately).
4. Add optional allreduce hooks (no-op when disabled) and fusion strategy `one` if TF runtime is enabled.
5. Add weight decay for dense grads.
6. Wire into `NativeContext.apply_embedding_gradients` equivalent and async scheduling.
7. Add summary/logging equivalents where possible.

**Tests (Detailed)**
- Python tests: `monolith/native_training/feature_utils_test.py`.
- Rust tests: `monolith-rs/crates/monolith-training/tests/feature_utils_test.rs` (new).
- Cross-language parity test: verify gradient updates and global_step increments on identical toy graphs.

**Gaps / Notes**
- Fusion path depends on custom TF ops (`monolith_aligned_flat_concat/split`).
- UE gradient check logic depends on feature tensors and names; requires parity in Rust model representation.

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

### `monolith/native_training/feature_utils_test.py`
<a id="monolith-native-training-feature-utils-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 144
- Purpose/role: Tests gradient application for dense vars and embeddings, including fused allreduce path and async embedding push.
- Key symbols/classes/functions: `_setup_test_embedding`, `FeatureUtilsTest` cases.
- External dependencies: TensorFlow, `feature_utils`, `feature`, `embedding_combiners`, `NativeContext`, `prefetch_queue`.
- Side effects: Sets env `MONOLITH_WITH_ALLREDUCE_FUSION=one`.

**Required Behavior (Detailed)**
- `_setup_test_embedding(is_async=False)`:
  - Builds embedding var and ragged ids, creates embedding slices via `create_embedding_slices`.
  - Mocks `feature_factory.apply_gradients` to subtract gradients from embedding vars.
  - Returns `(ctx, fc, emb_var, emb)`.
- `test_apply_gradients_with_dense_optimizer`:
  - Loss includes dense var and embedding sum; clip_norm=1.0.
  - After one step: dense var becomes 0.5; embedding var becomes `[0.5,0.5,0.5,1.0]`; global_step=1.
- `test_apply_gradients_with_dense_optimizer_gpu` (GPU-only):
  - Same expectations with `use_allreduce=True` and no summary; tests deferred clip fusion path.
- `test_apply_gradients_with_dense_optimizer_post_push`:
  - Async embedding push enabled; running op three times triggers two async pushes.
  - Dense var becomes -1.0; embedding var becomes `[-2.0,-2.0,-2.0,1.0]`.
- `test_apply_gradients_without_dense_optimizer`:
  - Loss uses embeddings only; after step, embedding var becomes `[0.0,0.0,0.0,1.0]` and global_step=1.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests/feature_utils_test.rs` (new).
- Rust public API surface: gradient application helpers and async embedding push hooks.
- Feature gating: GPU tests behind `cuda`/`tf-runtime` feature; skip if unavailable.
- Integration points: `NativeContext` equivalent and embedding update interface.

**Implementation Steps (Detailed)**
1. Port `_setup_test_embedding` logic to create a small embedding table and feature column in Rust.
2. Implement tests for dense+embedding gradients with clipping and global_step increments.
3. Add GPU test for deferred clip + allreduce path (skip if no GPU).
4. Add async embedding push test verifying delayed updates.

**Tests (Detailed)**
- Python tests: `monolith/native_training/feature_utils_test.py`.
- Rust tests: `monolith-rs/crates/monolith-training/tests/feature_utils_test.rs`.
- Cross-language parity test: compare updated dense var and embedding values after a single step.

**Gaps / Notes**
- Tests assume deterministic gradients and initial values; Rust must mirror initialization and scaling exactly.

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

### `monolith/native_training/gen_seq_mask.py`
<a id="monolith-native-training-gen-seq-mask-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 26
- Purpose/role: Wrapper around custom op to generate sequence masks from row splits.
- Key symbols/classes/functions: `gen_seq_mask`.
- External dependencies: TensorFlow, `gen_monolith_ops`.
- Side effects: None.

**Required Behavior (Detailed)**
- Accepts `splits` as Tensor or RaggedTensor; uses `row_splits()` when ragged.
- Calls `ops.gen_seq_mask(splits=..., max_seq_length=...)`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-tf/src/gen_seq_mask.rs` (new).
- Rust public API surface: `gen_seq_mask` wrapper.
- Feature gating: TF runtime + custom ops.

**Implementation Steps (Detailed)**
1. Add binding for `gen_seq_mask` custom op.
2. Accept either row_splits tensor or ragged wrapper.

**Tests (Detailed)**
- Python tests: `gen_seq_mask_test.py`.
- Rust tests: `monolith-rs/crates/monolith-tf/tests/gen_seq_mask_test.rs`.
- Cross-language parity test: compare masks for fixed splits.

**Gaps / Notes**
- Requires custom ops library.

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

### `monolith/native_training/gen_seq_mask_test.py`
<a id="monolith-native-training-gen-seq-mask-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 42
- Purpose/role: Tests gen_seq_mask for int32 and int64 splits.
- Key symbols/classes/functions: `GenSeqMaskTest`.
- External dependencies: TensorFlow, `gen_seq_mask`.
- Side effects: None.

**Required Behavior (Detailed)**
- For splits `[0,5,7,9,13]` and `max_seq_length=6`, mask equals expected matrix for both int32 and int64.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-tf/tests/gen_seq_mask_test.rs`.
- Rust public API surface: gen_seq_mask wrapper.
- Feature gating: TF runtime + custom ops.

**Implementation Steps (Detailed)**
1. Add tests for int32 and int64 splits with expected outputs.

**Tests (Detailed)**
- Python tests: `gen_seq_mask_test.py`.
- Rust tests: `monolith-rs/crates/monolith-tf/tests/gen_seq_mask_test.rs`.
- Cross-language parity test: compare masks.

**Gaps / Notes**
- None.

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

### `monolith/native_training/gflags_utils.py`
<a id="monolith-native-training-gflags-utils-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 282
- Purpose/role: Utilities to extract flags from dataclass docstrings and link flags to dataclass defaults.
- Key symbols/classes/functions: `extract_help_info`, `extract_flags`, `extract_flags_decorator`, `update`, `LinkDataclassToFlags`, `update_by_flags`.
- External dependencies: `absl.flags`, `dataclasses`, `Enum`, `inspect`, `re`.
- Side effects: Defines gflags and mutates dataclass instances based on flags.

**Required Behavior (Detailed)**
- `extract_help_info` parses `:param` lines and returns normalized help strings.
- `extract_flags` defines flags for type-hinted fields (int/bool/str/float/enum) with defaults; skips missing help or skip list.
- `extract_flags_decorator` returns decorator that calls `extract_flags`.
- `get_flags_parser` returns parser that logs errors and exits on invalid flags.
- `update` applies flags to config when config field is default and flag is non-default.
- `LinkDataclassToFlags` validates fields/flags and records mappings.
- `update_by_flags` patches `__init__` to apply linked flags when field is default.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-cli/src/gflags_utils.rs` (new) or `monolith-core` config utilities.
- Rust public API surface: config flag extraction and update helpers.
- Feature gating: CLI only.

**Implementation Steps (Detailed)**
1. Implement help metadata parsing (or explicit metadata in Rust).
2. Provide flag registration for primitive and enum types.
3. Implement update logic that respects defaults vs overrides.
4. Provide helper to link flags to dataclass fields.

**Tests (Detailed)**
- Python tests: `gflags_utils_test.py`.
- Rust tests: unit tests for help parsing and update/flag linking behavior.
- Cross-language parity test: compare config updates for equivalent flags.

**Gaps / Notes**
- Python relies on docstring parsing; Rust likely needs explicit metadata.

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

### `monolith/native_training/gflags_utils_test.py`
<a id="monolith-native-training-gflags-utils-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 217
- Purpose/role: Tests help parsing, flag extraction, config update, and flag linking.
- Key symbols/classes/functions: `GflagUtilsTest`.
- External dependencies: `absl.flags`, `absltest`, `gflags_utils`.
- Side effects: Defines flags in test scope.

**Required Behavior (Detailed)**
- `test_extract_help_info`: parses `:param` lines and joins multi-line help.
- `test_update`: updates config fields only when default and flag is non-default.
- `test_extract_gflags_decorator`: ensures flags are defined for decorated dataclasses and skipped for removed/base fields.
- `test_link_flag` / `test_link_flag_inheritance`: validates linked flags override defaults and inheritance behavior.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-cli/tests/gflags_utils_test.rs` (new).
- Rust public API surface: flag extraction and linking utilities.
- Feature gating: CLI only.

**Implementation Steps (Detailed)**
1. Add tests for help parsing and update logic.
2. Add tests for linked flags and inheritance behavior.

**Tests (Detailed)**
- Python tests: `gflags_utils_test.py`.
- Rust tests: `monolith-rs/crates/monolith-cli/tests/gflags_utils_test.rs`.
- Cross-language parity test: compare flag update outcomes.

**Gaps / Notes**
- Flag registration order in Rust may differ; ensure deterministic behavior.

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

### `monolith/native_training/graph_meta.py`
<a id="monolith-native-training-graph-meta-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 30
- Purpose/role: Stores per-graph metadata in a TF collection.
- Key symbols/classes/functions: `get_meta`.
- External dependencies: TensorFlow.
- Side effects: Mutates graph collections.

**Required Behavior (Detailed)**
- Uses graph collection `monolith_graph_meta` to store a single dict.
- If key missing, calls `MetaFactory()` and stores result.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-tf/src/graph_meta.rs` (new).
- Rust public API surface: graph metadata helper.
- Feature gating: TF runtime only.

**Implementation Steps (Detailed)**
1. Implement get-or-create metadata storage keyed in graph collection.
2. Mirror single-dict behavior.

**Tests (Detailed)**
- Python tests: none.
- Rust tests: add unit test for get_meta caching.

**Gaps / Notes**
- None.

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

### `monolith/native_training/graph_utils.py`
<a id="monolith-native-training-graph-utils-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 26
- Purpose/role: Adds batch-norm moving average assign ops into TF UPDATE_OPS collection.
- Key symbols/classes/functions: `add_batch_norm_into_update_ops`.
- External dependencies: TensorFlow.
- Side effects: Mutates default graph collections.

**Required Behavior (Detailed)**
- `add_batch_norm_into_update_ops()`:
  - Scans default graph operations.
  - Selects ops where `"AssignMovingAvg"` is in the op name and `op.type == "AssignSubVariableOp"`.
  - Adds each to `tf.GraphKeys.UPDATE_OPS` collection.

**Rust Mapping (Detailed)**
- Target crate/module: TF runtime utility module (e.g., `monolith-rs/crates/monolith-tf/src/graph_utils.rs`).
- Rust public API surface: `add_batch_norm_into_update_ops` equivalent for TF graphs.
- Feature gating: `tf-runtime` only.
- Integration points: training graph construction when using batch norm layers.

**Implementation Steps (Detailed)**
1. Implement graph op scan in TF runtime bindings.
2. Filter ops by name substring and op type.
3. Add ops to UPDATE_OPS collection.

**Tests (Detailed)**
- Python tests: none.
- Rust tests: add a small TF graph with batch norm and verify UPDATE_OPS contains moving average assigns.
- Cross-language parity test: compare the count of added ops for a known graph.

**Gaps / Notes**
- No-op for Candle backend; document as TF-only behavior.

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

### `monolith/native_training/hash_filter_ops.py`
<a id="monolith-native-training-hash-filter-ops-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 217
- Purpose/role: Builds hash filter resources, intercepts gradients, and handles save/restore for hash filters.
- Key symbols/classes/functions: `FilterType`, `create_hash_filters`, `save_hash_filter`, `restore_hash_filter`, `intercept_gradient`, `HashFilterCheckpointSaverListener`, `HashFilterCheckpointRestorerListener`.
- External dependencies: TensorFlow custom ops `gen_monolith_ops`, `save_utils.SaveHelper`, `basic_restore_hook`, `utils.ps_device`.
- Side effects: Creates TF resources, writes checkpoint assets, registers gradient for custom op.

**Required Behavior (Detailed)**
- Constants:
  - `HASH_FILTER_CAPACITY=300000000`, `HASH_FILTER_SPLIT_NUM=7`, `_TIMEOUT_IN_MS=1800000`.
- `FilterType`: string constants `SLIDING_HASH_FILTER`, `PROBABILISTIC_FILTER`, `NO_FILTER`.
- `create_hash_filter(capacity, split_num, config, name_suffix)`:
  - Calls `MonolithHashFilter` custom op with shared_name `MonolithHashFilter<suffix>`.
- `create_probabilistic_filter(equal_probability, config, name_suffix)`:
  - Calls `MonolithProbabilisticFilter` op with shared_name `MonolithProbabilisticFilter<suffix>`.
- `create_dummy_hash_filter(name_suffix)`:
  - Calls `MonolithDummyHashFilter` op with shared_name `DummyHashFilter<suffix>`.
- `_create_hash_filter(...)`:
  - Selects real or dummy filter based on `enable_hash_filter` and `filter_type`; invalid type raises `ValueError`.
- `create_hash_filters(ps_num, enable_hash_filter, ...)`:
  - If `ps_num==0`, returns a single filter.
  - Else, for each PS index, creates filter on `utils.ps_device(i)` unless exporting standalone.
- `save_hash_filter(hash_filter, basename, enable_hash_filter)`:
  - If enabled, uses `monolith_hash_filter_save` custom op; else returns `tf.no_op()`.
- `restore_hash_filter(hash_filter, basename, enable_hash_filter)`:
  - If enabled, uses `monolith_hash_filter_restore` custom op; else returns `tf.no_op()`.
- `intercept_gradient(filter_tensor, ids, embeddings)`:
  - Calls `MonolithHashFilterInterceptGradient`; filters gradients based on ids.
- `HashFilterCheckpointSaverListener`:
  - Builds save graph with placeholders per hash filter; writes to asset dir using SaveHelper.
  - `before_save` writes to `hash_filter_<ps_idx>` files under asset dir and runs save op with timeout.
- `HashFilterCheckpointRestorerListener`:
  - Looks up latest checkpoint; restores from asset dir or legacy prefix.
  - Uses placeholders for hash_filter basenames and runs restore op with timeout.
- Registered gradient `MonolithHashFilterInterceptGradient`:
  - Uses `MonolithHashFilterInterceptGradientGradient` custom op to produce filtered gradients; returns `(None, None, filtered_grad)`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-tf/src/hash_filter_ops.rs` (new) or TF runtime adapter.
- Rust public API surface: hash filter creation, save/restore, and gradient intercept wrappers.
- Feature gating: `tf-runtime` + custom ops required; no-op in Candle backend.
- Integration points: hash filter use in embedding tables and training loops.

**Implementation Steps (Detailed)**
1. Wrap custom ops for hash filter creation and gradient intercept.
2. Implement save/restore helpers using TF session run with timeouts.
3. Add Rust equivalents of saver/restorer listeners if using TF Estimator.
4. Ensure asset dir layout matches Python (`hash_filter_<ps_idx>` files).

**Tests (Detailed)**
- Python tests: `monolith/native_training/hash_filter_ops_test.py`.
- Rust tests: integration tests under `monolith-rs/crates/monolith-tf/tests/hash_filter_ops_test.rs` (new).
- Cross-language parity test: compare gradient filtering behavior and save/restore contents.

**Gaps / Notes**
- Requires custom ops from `libmonolith_ops` and TF runtime; not supported on pure Candle backend.

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

### `monolith/native_training/hash_filter_ops_test.py`
<a id="monolith-native-training-hash-filter-ops-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 217
- Purpose/role: Tests hash filter gradient interception and save/restore behavior.
- Key symbols/classes/functions: `HashFilterOpsTest` cases.
- External dependencies: TensorFlow, `hash_filter_ops`, `embedding_hash_table_pb2`, TFRecord reader.
- Side effects: Writes TFRecord checkpoint shards under `TEST_TMPDIR`.

**Required Behavior (Detailed)**
- `test_hash_filter_basic`:
  - Creates hash filter with occurrence threshold 3; intercepts gradient.
  - Gradients are filtered progressively: first two runs zero out more ids, later runs allow gradients.
- `test_hash_filter_save_restore`:
  - Saves filter to basename; verifies 7 split files created.
  - Restores and verifies gradient filtering state persists across saves.
- `test_hash_filter_save_restore_across_multiple_filters`:
  - Creates filter with split_num=100; verifies each shard contains expected `HashFilterSplitMetaDump` fields.
  - After second save, first 4 shards contain 2 elements; remaining shards contain 0.
- `test_dummy_hash_filter_basic`:
  - Dummy filter should not filter gradients (all ones).
- `test_dummy_hash_filter_save_restore`:
  - Save/restore with dummy filter produces no files and no effect on gradients.
- `test_restore_not_found`:
  - Restoring from non-existent path raises exception.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-tf/tests/hash_filter_ops_test.rs` (new).
- Rust public API surface: hash filter creation, save/restore, intercept gradient ops.
- Feature gating: `tf-runtime` + custom ops; skip otherwise.
- Integration points: TFRecord parsing of `HashFilterSplitMetaDump`.

**Implementation Steps (Detailed)**
1. Port test cases to Rust TF runtime harness.
2. Implement TFRecord reader for `HashFilterSplitMetaDump` proto in Rust.
3. Validate gradient filtering sequence and save/restore shard counts.

**Tests (Detailed)**
- Python tests: `monolith/native_training/hash_filter_ops_test.py`.
- Rust tests: `monolith-rs/crates/monolith-tf/tests/hash_filter_ops_test.rs`.
- Cross-language parity test: compare shard metadata and gradient outputs for same ids.

**Gaps / Notes**
- Tests depend on custom ops and TFRecord writer/reader compatibility.

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

### `monolith/native_training/hash_table_ops.py`
<a id="monolith-native-training-hash-table-ops-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 1208
- Purpose/role: Core TensorFlow hash table wrapper with custom ops for lookup, update, save/restore, and fused operations.
- Key symbols/classes/functions: `BaseHashTable`, `HashTable`, `hash_table_from_config`, `test_hash_table`, `fused_lookup`, `fused_apply_gradient`, checkpoint saver/restorer listeners.
- External dependencies: TensorFlow, custom ops `gen_monolith_ops`, `embedding_hash_table_pb2`, `hash_filter_ops`, `distributed_serving_ops`, `save_utils`, `graph_meta`.
- Side effects: Registers hash tables in graph collection, writes/reads checkpoint assets, registers proto serialization hooks.

**Required Behavior (Detailed)**
- `BaseHashTable` abstract API: `assign`, `assign_add`, `lookup`, `apply_gradients`, `as_op`, `dim_size`.
- `_HASH_TABLE_GRAPH_KEY` collection stores `HashTable` instances for save/restore.
- `HashTable`:
  - Ensures unique `shared_name` via metadata; raises if duplicate.
  - Wraps `monolith_hash_table_*` custom ops for assign/assign_add/lookup/lookup_entry/optimize/save/restore/size.
  - `apply_gradients` uses `monolith_hash_table_optimize` and returns a new table with control dependency.
  - `save_as_tensor` dumps entries as serialized `EntryDump` strings with sharding and offsets.
  - `to_proto`/`from_proto` serialize state via `hash_table_ops_pb2.HashTableProto` and `_BOOL_MAP`.
- `fused_lookup`:
  - Calls `monolith_hash_table_fused_lookup` and returns embeddings, recv_splits, id_offsets, emb_offsets.
- `fused_apply_gradient`:
  - Calls `monolith_hash_table_fused_optimize` with ids, indices, fused_slot_size, grads, offsets, learning rates, req_time, global_step.
- `hash_table_from_config`:
  - Builds table op using `EmbeddingHashTableConfig` and `HashTableConfigInstance`.
  - Chooses GPU vs CPU based on table type; forces SERVING entry type when exporting.
  - Creates hash filter and sync client if not provided.
- `test_hash_table` and `vocab_hash_table` helpers create simple tables for tests.
- `HashTableCheckpointSaverListener`:
  - Builds save ops using placeholders; writes asset files with randomized sleep to reduce metadata pressure.
- `HashTableCheckpointRestorerListener`:
  - Restores from latest checkpoint assets; supports sparse-only assets; uses thread pool to resolve prefixes.
- `HashTableRestorerSaverListener`:
  - Triggers restore after save (used for evicting stale entries).
- Registers proto serialization with `ops.register_proto_function`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-hash-table` + TF runtime wrappers in `monolith-rs/crates/monolith-tf`.
- Rust public API surface: hash table struct with assign/lookup/optimize, fused lookup/optimize wrappers, save/restore helpers.
- Data model mapping: use `monolith::hash_table` protos for config and serialization.
- Feature gating: TF custom ops required; Candle backend may implement a native hash table for local use.
- Integration points: feature factory, embedding gradients, distributed serving.

**Implementation Steps (Detailed)**
1. Implement a Rust hash table wrapper that mirrors assign/assign_add/lookup semantics and returns updated handles.
2. Add proto serialization helpers and maintain a registry for save/restore discovery.
3. Wrap fused lookup/optimize in TF runtime; add no-op stubs otherwise.
4. Implement checkpoint saver/restorer listeners and asset dir layout matching Python.
5. Add thread-pool based restore prefix matching for extra restore names.

**Tests (Detailed)**
- Python tests: `hash_table_ops_test.py`.
- Rust tests: `monolith-rs/crates/monolith-tf/tests/hash_table_ops_test.rs` (TF runtime) and native hash table unit tests.
- Cross-language parity test: compare lookup/update outputs and serialized dumps.

**Gaps / Notes**
- Heavy reliance on custom TF ops; full parity requires TF runtime + compiled ops.
- Save/restore path handling must match asset dir naming to avoid silent failures.

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

### `monolith/native_training/hash_table_ops_benchmark.py`
<a id="monolith-native-training-hash-table-ops-benchmark-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 148
- Purpose/role: Benchmarks hash table lookup and optimize paths (single-thread vs multi-thread).
- Key symbols/classes/functions: `HashTableOpsBenchmark` tests.
- External dependencies: TensorFlow, `hash_table_ops`.
- Side effects: Prints timing to stdout.

**Required Behavior (Detailed)**
- `test_lookup` / `test_lookup_multi_thread`:
  - Build table with len=10000, dim=32; assign ones for all but last 5 IDs.
  - Run lookup in a loop and validate embeddings (ones vs zeros).
- `test_basic_optimize` / `test_multi_threads_optimize` / `test_multi_threads_optimize_with_dedup`:
  - Build table with len=1,000,000; lookup, compute grads, apply gradients (optionally MT/dedup), print timing.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-examples/src/bin/hash_table_ops_benchmark.rs` (new).
- Rust public API surface: hash table benchmark harness.
- Feature gating: TF runtime + custom ops for parity.

**Implementation Steps (Detailed)**
1. Port benchmark loops to Rust, mirroring data sizes and operations.
2. Add CLI flags for MT/dedup variants.
3. Validate outputs in debug mode.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: none; benchmark only.
- Cross-language parity test: compare output correctness and rough timing.

**Gaps / Notes**
- Uses `tf.test.TestCase` for benchmarking; Rust should use criterion or custom timers.

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

### `monolith/native_training/hash_table_ops_test.py`
<a id="monolith-native-training-hash-table-ops-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 1206
- Purpose/role: Comprehensive tests for hash table lookup, update, save/restore, eviction, fused ops, and hooks.
- Key symbols/classes/functions: `HashTableOpsTest` methods, helper `test_hash_table_with_hash_filters`.
- External dependencies: TensorFlow, `hash_table_ops`, `hash_filter_ops`, `embedding_hash_table_pb2`, `learning_rate_functions`.
- Side effects: Writes checkpoint assets under `TEST_TMPDIR`.

**Required Behavior (Detailed)**
- Basic ops:
  - `test_basic` assigns values and verifies lookup and size; table name auto-unique.
  - `test_assign` verifies assign overwrites values under control dependencies.
  - `test_lookup_entry` parses `EntryDump` strings; missing ids return empty bytes.
  - `test_save_as_tensor` ensures serialized entries can be parsed.
  - `testNameConflict` duplicate shared_name raises `ValueError`.
- Gradient updates:
  - `test_gradients` updates embeddings with SGD (learning_rate=0.1) -> `[[0.2],[0.1]]` for ids [0,1].
  - `test_gradients_with_learning_rate_fn` accepts callable LR.
  - `test_gradients_with_learning_rate_decay` uses PolynomialDecay; expected outputs `[[0.04],[0.02]]`.
  - `test_gradients_with_dedup` enables dedup; expected outputs `[[0.3...],[0.2...]]` for vec_dim=10.
  - `test_gradients_with_different_ids` applies grads with mismatched ids -> `[[0.1],[0.2]]`.
  - `test_gradients_with_hash_filter` verifies occurrence threshold gating across repeated updates.
- Save/restore:
  - `test_save_restore` round-trips assign_add values.
  - `test_restore_from_another_table` uses extra_restore_names to restore.
- Feature eviction / TTL:
  - `test_save_restore_with_feature_eviction_assign_add` and `...apply_gradients` evict entries older than expire_time.
  - `test_entry_ttl_zero` evicts all entries on restore.
  - `test_entry_ttl_not_zero` preserves entries when TTL positive.
  - `test_entry_ttl_by_slots` uses slot_expire_time_config for per-slot TTL.
- Hooks and restore flows:
  - `test_restore_not_found` raises on missing checkpoint.
  - `test_save_restore_hook` saver + restorer hook restores after sub_op.
  - Additional tests validate restore-after-save, feature eviction with hooks, and cleanup of save paths.
- Advanced/fused ops:
  - `test_fused_lookup` and `test_fused_optimize` cover fused operations.
  - `test_batch_softmax_optimizer` verifies BatchSoftmax behavior.
  - `test_extract_fid` checks `extract_slot_from_entry`.
  - `test_meta_graph_export` ensures proto export/import works.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-tf/tests/hash_table_ops_test.rs` (new).
- Rust public API surface: hash table operations, fused lookup/optimize, save/restore, eviction logic.
- Feature gating: TF runtime + custom ops required; skip otherwise.
- Integration points: hash filter ops, learning rate functions, save_utils equivalents.

**Implementation Steps (Detailed)**
1. Port basic lookup/assign/size tests to Rust TF runtime.
2. Implement gradient update tests for SGD and learning rate schedules.
3. Add save/restore tests with asset files; verify eviction by TTL and per-slot settings.
4. Add hook-based save/restore ordering tests.
5. Add fused op tests and meta-graph export/import tests.

**Tests (Detailed)**
- Python tests: `monolith/native_training/hash_table_ops_test.py`.
- Rust tests: `monolith-rs/crates/monolith-tf/tests/hash_table_ops_test.rs`.
- Cross-language parity test: compare serialized `EntryDump` bytes and lookup results.

**Gaps / Notes**
- Many tests depend on TF custom ops and checkpoint assets; may be too heavy for Rust CI.

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

### `monolith/native_training/hash_table_utils.py`
<a id="monolith-native-training-hash-table-utils-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 46
- Purpose/role: Utility helpers to iterate hash tables and infer embedding dim sizes from config.
- Key symbols/classes/functions: `iterate_table_and_apply`, `infer_dim_size`.
- External dependencies: TensorFlow, `embedding_hash_table_pb2`.
- Side effects: Iterates table via `save_as_tensor` and calls `apply_fn`.

**Required Behavior (Detailed)**
- `iterate_table_and_apply(table, apply_fn, limit=1000, nshards=4, name="IterateTable")`:
  - Runs in `tf.function`.
  - Iterates `nshards` shards; for each shard, repeatedly calls `table.save_as_tensor(i, nshards, limit, offset)` until dump size < limit and offset != 0.
  - Uses `tf.autograph.experimental.set_loop_options` with shape invariants to allow dynamic `dump` size.
  - Calls `apply_fn(dump)` for each dump batch (serialized EntryDump strings).
- `infer_dim_size(config)`:
  - Sums `segment.dim_size` across `config.entry_config.segments`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-hash-table/src/utils.rs` (new) or `monolith-rs/crates/monolith-hash-table/src/lib.rs`.
- Rust public API surface: iterator over table dumps and dim-size inference from proto config.
- Data model mapping: `EmbeddingHashTableConfig` from `monolith-proto`.
- Feature gating: table iteration requires TF runtime or native hash table implementation.
- Integration points: model dump utilities and table export.

**Implementation Steps (Detailed)**
1. Implement a Rust iterator that pages through table dumps with `limit` and `offset` semantics.
2. Provide a safe callback API for applying functions to each batch.
3. Implement `infer_dim_size` by summing segment dims from proto.

**Tests (Detailed)**
- Python tests: none.
- Rust tests: add unit tests for `infer_dim_size` and mock iteration semantics.
- Cross-language parity test: compare dim_size for a known config.

**Gaps / Notes**
- `iterate_table_and_apply` depends on `save_as_tensor` behavior; Rust must match sharding and offset semantics.

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

### `monolith/native_training/hash_table_utils_test.py`
<a id="monolith-native-training-hash-table-utils-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 45
- Purpose/role: Tests `iterate_table_and_apply` paging across shards.
- Key symbols/classes/functions: `HashTableUtilsTest.test_iterate_table_and_apply`.
- External dependencies: TensorFlow, `hash_table_utils`, `hash_table_ops`.
- Side effects: Creates a test hash table and updates a counter variable.

**Required Behavior (Detailed)**
- Creates a test hash table with 100 ids.
- Uses `iterate_table_and_apply` with `limit=2` and `nshards=10` to iterate.
- `count_fn` increments a counter by `tf.size(dump)` for each batch.
- Final count must equal 100.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-hash-table/tests/hash_table_utils_test.rs` (new).
- Rust public API surface: table iteration helper and callback support.
- Feature gating: TF runtime or native hash table required.
- Integration points: hash table test helper equivalent.

**Implementation Steps (Detailed)**
1. Implement a Rust test that fills a table with 100 entries.
2. Iterate with small limit and shard count; accumulate total dumped entries.
3. Assert total count equals 100.

**Tests (Detailed)**
- Python tests: `monolith/native_training/hash_table_utils_test.py`.
- Rust tests: `monolith-rs/crates/monolith-hash-table/tests/hash_table_utils_test.rs`.
- Cross-language parity test: compare counts for same config.

**Gaps / Notes**
- Depends on `save_as_tensor` semantics; ensure Rust matches offset/limit behavior.

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

### `monolith/native_training/hooks/ckpt_hooks.py`
<a id="monolith-native-training-hooks-ckpt-hooks-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 193
- Purpose/role: Checkpoint hooks for worker iterator state and barrier coordination during saves.
- Key symbols/classes/functions: `BarrierSaverListener`, `WorkerCkptHelper`, `assign_ckpt_info`, `get_ckpt_info`, `disable_iterator_save_restore`.
- External dependencies: TensorFlow, `barrier_ops`, `basic_restore_hook`, `graph_meta`, `ckpt_hooks_pb2`.
- Side effects: Creates local variables and placeholders, manipulates iterator saveables, blocks workers with barriers.

**Required Behavior (Detailed)**
- `_get_meta()`:
  - Lazily creates `WorkerCkptMetaInfo` local variable + placeholder + assign op.
- `assign_ckpt_info(session, info)`:
  - Assigns serialized `WorkerCkptInfo` to info_var.
- `get_ckpt_info(session)`:
  - Reads info_var and parses to `WorkerCkptInfo`.
- `BarrierSaverListener`:
  - On `before_save`, places a barrier and waits for workers to block (up to max_pending_seconds).
  - On `after_save`, removes barrier if it was placed by this listener.
- `WorkerCkptHelper`:
  - Creates iterator saveables (if enabled) and a Saver to save per-worker iterator state.
  - `create_save_iterator_callback` saves iterator checkpoints using current global_step from `WorkerCkptInfo`.
  - `create_restorer_hook` restores iterator state on session creation.
- `disable_iterator_save_restore()`:
  - Disables iterator save/restore globally (must be called before helper creation).

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/hooks/ckpt_hooks.rs` (new).
- Rust public API surface: iterator checkpoint helper and barrier-based saver listener.
- Feature gating: TF runtime required for iterator saveables and session hooks.
- Integration points: training loops and distributed barrier coordination.

**Implementation Steps (Detailed)**
1. Implement worker checkpoint metadata storage and serialization.
2. Implement barrier saver listener with wait/remove semantics.
3. Implement iterator save/restore helper for dataset iterators.
4. Provide a global toggle to disable iterator save/restore.

**Tests (Detailed)**
- Python tests: `ckpt_hooks_test.py`.
- Rust tests: `monolith-rs/crates/monolith-training/tests/ckpt_hooks_test.rs` (new).
- Cross-language parity test: verify iterator restore resumes from same element.

**Gaps / Notes**
- Uses TF iterator saveables and should be skipped if dataset iterators are not used in Rust.

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

### `monolith/native_training/hooks/ckpt_hooks_test.py`
<a id="monolith-native-training-hooks-ckpt-hooks-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 181
- Purpose/role: Tests worker iterator save/restore and barrier-based saver listener behavior.
- Key symbols/classes/functions: `WorkerCkptHooksTest`, `CountCheckpointSaverListener`.
- External dependencies: TensorFlow, `ckpt_hooks`, `barrier_ops`, `save_utils`.
- Side effects: Writes checkpoints under `TEST_TMPDIR`, spawns a thread to run a monitored session.

**Required Behavior (Detailed)**
- `testIteratorSaveRestore`:
  - Saves iterator state at global_step=10 and restores; next element after restore matches expected value.
- `testNoCkpt`:
  - Restorer hook is a no-op when no checkpoint exists.
- `testNoSaveables`:
  - If no saveables, saving iterator state is skipped without error.
- `testCkptDisabled`:
  - `disable_iterator_save_restore()` prevents restore; iterator restarts from beginning.
- `test_saver_with_barrier`:
  - Verifies barrier placement during save and that saver listener callbacks are invoked in order.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests/ckpt_hooks_test.rs` (new).
- Rust public API surface: barrier saver listener and worker iterator checkpoint helper.
- Feature gating: TF runtime required.
- Integration points: save_utils and barrier operations.

**Implementation Steps (Detailed)**
1. Add a Rust test dataset iterator and verify save/restore steps.
2. Add a test for disabled iterator restore.
3. Add a barrier coordination test ensuring save waits for workers.

**Tests (Detailed)**
- Python tests: `monolith/native_training/hooks/ckpt_hooks_test.py`.
- Rust tests: `monolith-rs/crates/monolith-training/tests/ckpt_hooks_test.rs`.
- Cross-language parity test: compare iterator position after restore.

**Gaps / Notes**
- Threaded monitored session in test may be hard to replicate in Rust; can approximate with explicit calls.

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

### `monolith/native_training/hooks/ckpt_info.py`
<a id="monolith-native-training-hooks-ckpt-info-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 98
- Purpose/role: Saves per-slot feature-id counts from hash tables into a `ckpt.info-*` file at checkpoint time.
- Key symbols/classes/functions: `FidSlotCountSaverListener`.
- External dependencies: TensorFlow, `hash_table_ops`, `hash_table_utils`, `ckpt_info_pb2`.
- Side effects: Writes `ckpt.info-<global_step>` to model_dir.

**Required Behavior (Detailed)**
- `FidSlotCountSaverListener.__init__(model_dir)`:
  - Collects hash tables from graph collections; errors if none exist.
  - Groups tables by device and allocates per-device count variables of size `_MAX_SLOT`.
  - Builds `iterate_table_and_apply` ops to accumulate slot counts.
- `before_save`:
  - Skips if multi-hash tables exist.
  - Initializes count vars, runs count op, sums counts across devices.
  - Writes `ckpt_info_pb2.CkptInfo` text to `ckpt.info-<step>`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/hooks/ckpt_info.rs` (new).
- Rust public API surface: saver listener that writes slot counts.
- Feature gating: TF runtime required for table iteration.
- Integration points: hash table iteration helper and checkpoint hooks.

**Implementation Steps (Detailed)**
1. Implement slot-count accumulation using table dump iteration.
2. Serialize `CkptInfo` via protobuf text format.
3. Write `ckpt.info-<step>` file during save.

**Tests (Detailed)**
- Python tests: `ckpt_info_test.py`.
- Rust tests: `monolith-rs/crates/monolith-training/tests/ckpt_info_test.rs` (new).
- Cross-language parity test: compare output file contents for the same table.

**Gaps / Notes**
- `_MAX_SLOT` constant must match; slot extraction uses `extract_slot_from_entry` custom op.

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

### `monolith/native_training/hooks/hook_utils_test.py`
<a id="monolith-native-training-hooks-hook-utils-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 35
- Purpose/role: Smoke test for BeforeSaveListener and AfterSaveListener wrappers.
- Key symbols/classes/functions: `HookUtilsTest.testBeforeAfterSaverListener`.
- External dependencies: TensorFlow, `hook_utils`.
- Side effects: None.

**Required Behavior (Detailed)**
- Wraps a base `CheckpointSaverListener` and calls before/after save methods to ensure no errors.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests/hook_utils_test.rs` (new).
- Rust public API surface: hook wrapper types.
- Feature gating: TF runtime.

**Implementation Steps (Detailed)**
1. Instantiate wrapper types around a dummy listener.
2. Invoke before/after save methods and ensure no panic.

**Tests (Detailed)**
- Python tests: `hook_utils_test.py`.
- Rust tests: `monolith-rs/crates/monolith-training/tests/hook_utils_test.rs`.
- Cross-language parity test: not required beyond smoke test.

**Gaps / Notes**
- This is a compile/smoke test only.

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

### `monolith/native_training/hooks/ps_check_hooks.py`
<a id="monolith-native-training-hooks-ps-check-hooks-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 97
- Purpose/role: Health-check hooks for PS machines, reporting failures and placing barriers.
- Key symbols/classes/functions: `PsHealthCheckerHook`, `Config`, `get_ps_machine_info_shared_name`.
- External dependencies: TensorFlow, `logging_ops`, `barrier_ops`, `logging_ops_pb2`.
- Side effects: Spawns background thread, places barrier on failure, logs error details.

**Required Behavior (Detailed)**
- `get_ps_machine_info_shared_name(index)` returns `"ps_machine_info_<index>"`.
- `_default_report(results)`:
  - Logs per-PS MachineHealthResult using text_format one-line strings.
- `Config`:
  - Contains `barrier_op`, `num_ps`, `ps_device_fn` (default `utils.ps_device`), `report_fn` (default `_default_report`).
- `_PsHealthChecker`:
  - Builds `machine_info` and `check_machine_health` ops per PS device.
  - Runs in a daemon thread registered with coordinator.
  - If any status is non-empty, parses `MachineHealthResult`, calls report_fn, places barrier, and waits for stop.
  - Sleeps/waits via `coord.wait_for_stop(timeout=30)` in loop.
- `PsHealthCheckerHook`:
  - Creates checker on `begin()` and starts thread in `after_create_session`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/hooks/ps_check_hooks.rs` (new).
- Rust public API surface: PS health checker hook with background polling.
- Feature gating: TF runtime/custom ops required for machine health ops.
- Integration points: barrier ops and logging/alerting system.

**Implementation Steps (Detailed)**
1. Wrap logging_ops machine_info and health check ops in Rust TF runtime.
2. Spawn a background thread to poll health status.
3. On failure, call report_fn and place barrier, then request stop.

**Tests (Detailed)**
- Python tests: `ps_check_hooks_test.py`.
- Rust tests: `monolith-rs/crates/monolith-training/tests/ps_check_hooks_test.rs` (new).
- Cross-language parity test: simulate healthy vs OOM conditions and verify report hook invocation.

**Gaps / Notes**
- Health status is encoded as serialized proto bytes; Rust must parse with `logging_ops_pb2` equivalent.

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

### `monolith/native_training/hooks/ps_check_hooks_test.py`
<a id="monolith-native-training-hooks-ps-check-hooks-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 112
- Purpose/role: Tests PS health checker hook and error handling.
- Key symbols/classes/functions: `PsCheckHooksTest` cases.
- External dependencies: TensorFlow, `ps_check_hooks`, `logging_ops`.
- Side effects: Uses monitored sessions and sleeps briefly.

**Required Behavior (Detailed)**
- `test_basic`:
  - Healthy machine info should not trigger report.
- `test_oom`:
  - mem_limit=0 triggers report_fn once.
- `test_raise_in_after_create_session` / `test_raise_in_before_run`:
  - Raising in hooks should propagate DeadlineExceededError.
- `test_default_report`:
  - Calls `_default_report` with a MachineHealthResult for smoke coverage.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests/ps_check_hooks_test.rs` (new).
- Rust public API surface: PS health checker hook.
- Feature gating: TF runtime/custom ops required.

**Implementation Steps (Detailed)**
1. Add a test harness that simulates healthy and unhealthy machine_info results.
2. Assert report function called under unhealthy case.
3. Verify exceptions propagate from hook callbacks.

**Tests (Detailed)**
- Python tests: `ps_check_hooks_test.py`.
- Rust tests: `monolith-rs/crates/monolith-training/tests/ps_check_hooks_test.rs`.
- Cross-language parity test: compare report invocation counts.

**Gaps / Notes**
- Tests rely on custom logging_ops; may need to skip if ops unavailable.

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

### `monolith/native_training/hooks/server/client_lib.py`
<a id="monolith-native-training-hooks-server-client-lib-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 30
- Purpose/role: gRPC client helper to connect to controller server from model_dir.
- Key symbols/classes/functions: `get_stub_from_model_dir`.
- External dependencies: `grpc`, TensorFlow gfile, generated `service_pb2_grpc`.
- Side effects: Reads controller server address file.

**Required Behavior (Detailed)**
- Reads `<model_dir>/controller_server_addr.txt`.
- Creates `grpc.insecure_channel(addr)` and returns `ControllerStub`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/hooks/server/client_lib.rs` (new).
- Rust public API surface: helper to read addr file and create gRPC client.
- Feature gating: gRPC required.

**Implementation Steps (Detailed)**
1. Read server addr file from model_dir.
2. Create gRPC channel and Controller client stub.

**Tests (Detailed)**
- Python tests: `server_lib_test.py`.
- Rust tests: `monolith-rs/crates/monolith-training/tests/server_lib_test.rs` (integration).
- Cross-language parity test: ensure client can connect to server hook.

**Gaps / Notes**
- Assumes address file is present and readable.

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

### `monolith/native_training/hooks/server/constants.py`
<a id="monolith-native-training-hooks-server-constants-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 15
- Purpose/role: Defines filename for controller server address.
- Key symbols/classes/functions: `SERVER_ADDR_FILENAME`.
- External dependencies: None.
- Side effects: None.

**Required Behavior (Detailed)**
- `SERVER_ADDR_FILENAME = "controller_server_addr.txt"`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/hooks/server/constants.rs` (new).
- Rust public API surface: constant for addr file name.

**Implementation Steps (Detailed)**
1. Define `SERVER_ADDR_FILENAME` constant.

**Tests (Detailed)**
- Python tests: covered indirectly in `server_lib_test.py`.
- Rust tests: none required beyond integration tests.

**Gaps / Notes**
- None.

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

### `monolith/native_training/hooks/server/server_lib.py`
<a id="monolith-native-training-hooks-server-server-lib-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 95
- Purpose/role: gRPC controller server for training control (stop/resume/save/status).
- Key symbols/classes/functions: `ControllerServicer`, `ServerHook`.
- External dependencies: gRPC, TensorFlow, `barrier_ops`, `save_utils`, `net_utils`.
- Side effects: Starts a gRPC server, writes address file, triggers barrier ops and checkpoints.

**Required Behavior (Detailed)**
- `ControllerServicer`:
  - `StopTraining`: places barrier; if already placed, aborts with ALREADY_EXISTS.
  - `ResumeTraining`: removes barrier.
  - `GetBlockStatus`: returns blocked and unblocked indices for barrier.
  - `SaveCheckpoint`: calls `saver_hook.trigger_save`.
  - `GetTrainingStatus`: returns global_step from session.
- `ServerHook`:
  - On `after_create_session`, starts gRPC server on ephemeral port, writes addr to `<model_dir>/controller_server_addr.txt`.
  - On `end`, stops server (grace 20s).

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/hooks/server/server_lib.rs` (new).
- Rust public API surface: Controller gRPC service + ServerHook.
- Feature gating: gRPC required; TF runtime for SessionRunHook lifecycle.
- Integration points: barrier ops, checkpoint saver hooks, training global_step.

**Implementation Steps (Detailed)**
1. Define gRPC service with Stop/Resume/Status/Save endpoints.
2. Start server in hook after session creation; write addr file.
3. Implement barrier operations and save trigger forwarding.

**Tests (Detailed)**
- Python tests: `server_lib_test.py`.
- Rust tests: `monolith-rs/crates/monolith-training/tests/server_lib_test.rs`.
- Cross-language parity test: issue gRPC commands and verify barrier state changes.

**Gaps / Notes**
- Uses `net_utils.get_local_server_addr` for address formatting; Rust should mirror semantics.

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

### `monolith/native_training/hooks/server/server_lib_test.py`
<a id="monolith-native-training-hooks-server-server-lib-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 54
- Purpose/role: Integration test for controller gRPC server and client helper.
- Key symbols/classes/functions: `ServerTest.test_basic`.
- External dependencies: TensorFlow, gRPC, `server_lib`, `client_lib`, `barrier_ops`, `save_utils`.
- Side effects: Starts server in monitored session.

**Required Behavior (Detailed)**
- Starts ServerHook and saver hook in a session.
- Uses client stub from model_dir to:
  - StopTraining (second StopTraining should raise RpcError).
  - GetBlockStatus shows blocked index then unblocked after ResumeTraining.
  - SaveCheckpoint and GetTrainingStatus succeed.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests/server_lib_test.rs` (new).
- Rust public API surface: controller server hook and client stub.
- Feature gating: gRPC required.

**Implementation Steps (Detailed)**
1. Start controller server hook and saver hook in test session.
2. Use client to call gRPC methods and assert barrier behavior.

**Tests (Detailed)**
- Python tests: `server_lib_test.py`.
- Rust tests: `monolith-rs/crates/monolith-training/tests/server_lib_test.rs`.
- Cross-language parity test: compare gRPC behavior and responses.

**Gaps / Notes**
- Needs a working saver hook to test SaveCheckpoint path.

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

### `monolith/native_training/hooks/session_hooks.py`
<a id="monolith-native-training-hooks-session-hooks-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 44
- Purpose/role: Tracks the current TF session via a hook and provides a helper to fetch it.
- Key symbols/classes/functions: `SetCurrentSessionHook`, `get_current_session`.
- External dependencies: TensorFlow.
- Side effects: Stores session in a module-level singleton during hook lifetime.

**Required Behavior (Detailed)**
- `SetCurrentSessionHook.after_create_session` sets `_INFO.session`.
- `SetCurrentSessionHook.end` clears `_INFO.session`.
- `get_current_session()` returns `_INFO.session` if set, else `tf.compat.v1.get_default_session()`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/hooks/session_hooks.rs` (new).
- Rust public API surface: session tracking hook + helper to fetch current session.
- Feature gating: TF runtime only.

**Implementation Steps (Detailed)**
1. Add a thread-safe global or TLS slot for current session.
2. Hook into session creation/end to set/clear.
3. Provide getter that falls back to default session.

**Tests (Detailed)**
- Python tests: `session_hooks_test.py`.
- Rust tests: `monolith-rs/crates/monolith-training/tests/session_hooks_test.rs`.
- Cross-language parity test: ensure session is available inside monitored session and cleared after.

**Gaps / Notes**
- Global session state should be scoped carefully in multi-session environments.

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

### `monolith/native_training/hooks/session_hooks_test.py`
<a id="monolith-native-training-hooks-session-hooks-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 33
- Purpose/role: Smoke test for current-session tracking.
- Key symbols/classes/functions: `SessionHooksTest.testBasic`.
- External dependencies: TensorFlow, `session_hooks`.
- Side effects: None.

**Required Behavior (Detailed)**
- Asserts `get_current_session()` is None outside a session.
- Inside MonitoredSession with `SetCurrentSessionHook`, `get_current_session()` returns non-None.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests/session_hooks_test.rs` (new).
- Rust public API surface: session tracking helper.
- Feature gating: TF runtime.

**Implementation Steps (Detailed)**
1. Add test that checks current session availability inside hook-managed session.
2. Ensure session is cleared after end.

**Tests (Detailed)**
- Python tests: `session_hooks_test.py`.
- Rust tests: `monolith-rs/crates/monolith-training/tests/session_hooks_test.rs`.
- Cross-language parity test: not required beyond smoke behavior.

**Gaps / Notes**
- None.

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

### `monolith/native_training/hvd_lib.py`
<a id="monolith-native-training-hvd-lib-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 65
- Purpose/role: Lazy import wrapper for Horovod or BytePS TensorFlow libraries.
- Key symbols/classes/functions: `_Lib`, module-level `__getattr__`.
- External dependencies: `byteps.tensorflow` or `horovod.tensorflow`.
- Side effects: Imports the chosen library on first use.

**Required Behavior (Detailed)**
- `_Lib.enable_bps` reads `MONOLITH_WITH_BYTEPS` env var.
- `lib` property imports BytePS if enabled, else Horovod.
- Provides passthrough methods: `init`, `rank`, `size`, `allgather`, `broadcast`, `BroadcastGlobalVariablesHook`.
- Module-level `__getattr__` forwards to `_Lib` methods.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/hvd_lib.rs` (new).
- Rust public API surface: wrapper interface with lazy initialization for Horovod/BytePS bindings.
- Feature gating: only available under TF runtime / distributed features.

**Implementation Steps (Detailed)**
1. Implement lazy initialization with mutex/once for BytePS or Horovod bindings.
2. Expose helper methods mirroring Python names.
3. Read `MONOLITH_WITH_BYTEPS` to choose backend.

**Tests (Detailed)**
- Python tests: none.
- Rust tests: add smoke tests to verify backend selection and lazy init.
- Cross-language parity test: ensure backend selection matches env.

**Gaps / Notes**
- Requires actual Horovod/BytePS bindings for Rust; otherwise should be stubbed with clear errors.

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

### `monolith/native_training/input.py`
<a id="monolith-native-training-input-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 45
- Purpose/role: Utility to generate random FFM training examples.
- Key symbols/classes/functions: `slot_to_key`, `generate_ffm_example`.
- External dependencies: NumPy, TensorFlow.
- Side effects: None.

**Required Behavior (Detailed)**
- `slot_to_key(slot)` returns `"feature_<slot>"`.
- `generate_ffm_example(vocab_sizes, length=5)`:
  - Creates label feature with random int in [0,1) (effectively 0).
  - For each vocab size, samples `num_ids` in `[1, length]` and ids in a range offset by `max_vocab * i`.
  - Constructs a `tf.train.Example` and returns serialized bytes.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/src/input.rs` (new) or `monolith-rs/crates/monolith-examples` helpers.
- Rust public API surface: helper to generate serialized Example protos.
- Data model mapping: `monolith::io::proto::Example` or TensorFlow Example proto.

**Implementation Steps (Detailed)**
1. Implement `slot_to_key` helper.
2. Implement random example generation with identical id ranges.
3. Serialize Example protobuf to bytes.

**Tests (Detailed)**
- Python tests: used indirectly in estimator tests.
- Rust tests: add deterministic test with fixed RNG seed.
- Cross-language parity test: compare serialized Example with fixed seed.

**Gaps / Notes**
- Python uses `np.random.randint(low=0, high=1)` which always yields 0; consider whether to preserve this.

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

### `monolith/native_training/layers/__init__.py`
<a id="monolith-native-training-layers-init-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 46
- Purpose/role: Aggregates and re-exports Keras layers plus Monolith custom layers, then patches Keras layer classes with a `params()` helper for InstantiableParams construction.
- Key symbols/classes/functions:
  - Re-exported custom layers: `MLP`, `Dense` (custom override), `AddBias`, `LHUCTower`, `LogitCorrection`, `LayerNorm`, `GradNorm`, `SumPooling`, `AvgPooling`, `MaxPooling`, `MergeType`, `DCNType`, `MMoE`, `SNR`, plus everything from `feature_cross`, `feature_trans`, `feature_seq`, `advanced_activations`.
  - `keras_layers` dict: name → Keras layer class with `params` monkey-patched.
- External dependencies: `tensorflow` (`tf.keras.layers`, `Layer`), `types.MethodType`, `monolith.native_training.utils.params` (`params` helper).
- Side effects: Module import-time monkey-patching of Keras layer classes to inject `params()`; removal of wildcard-imported `Dense` symbol to replace with custom Dense.

**Required Behavior (Detailed)**
- Import order and namespace behavior:
  - `from tensorflow.keras.layers import *` makes all Keras layer classes available in module namespace.
  - `del globals()['Dense']` removes the Keras `Dense` symbol created by the wildcard import.
  - `from monolith.native_training.layers.dense import Dense` inserts the custom Dense in its place.
- Custom layer re-exports:
  - Re-exports layer modules so downstream Python code can do `from monolith.native_training.layers import X` for both Keras layers and custom layers.
- Keras layer patching:
  - Creates `keras_layers = {}`.
  - Iterates `dir(tf.keras.layers)`, skipping names that start with `_` or are exactly `"Layer"`.
  - For each candidate:
    - Retrieves `cls = getattr(tf.keras.layers, name)`.
    - If `issubclass(cls, Layer)` and `cls` does **not** already have `params`, then attaches `cls.params = MethodType(_params, cls)` and inserts `keras_layers[name] = cls`.
    - All errors in this process are swallowed (`except: pass`) to avoid import-time failures for non-class attributes or invalid `issubclass` checks.
  - Result: `keras_layers` includes only Keras layer classes that were patched.
- Error handling and determinism:
  - No explicit errors thrown; all reflection errors are suppressed.
  - Determinism is not relevant; import-time logic is pure reflection/patching.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-layers/src/lib.rs` + `monolith-rs/crates/monolith-layers/src/prelude` (existing `prelude` module).
- Rust public API surface:
  - Provide a `prelude` module that re-exports Monolith layers analogous to Python's `__init__` aggregator.
  - Ensure `Dense` refers to the Monolith implementation (`monolith_layers::Dense`).
  - Expose `MergeType` and `DCNType` equivalents (`MergeType` already exists; map Python `DCNType` to Rust `DCNMode` or add an alias if necessary).
- Data model mapping:
  - Python's `params()` returns `InstantiableParams`; Rust should expose a `LayerParams`/`BuildConfig` trait that returns serializable config metadata per layer, or central registry entries.
- Feature gating:
  - If TF runtime backend is enabled (`cfg(feature = "tf-runtime")`), consider an optional registry for TensorFlow-native layers.
  - Otherwise, provide Monolith-only registry/prelude without TF/Keras dependencies.
- Integration points:
  - Downstream uses `monolith.native_training.layers` as a convenience import; in Rust this maps to `monolith_layers::prelude::*` or a `layers` module in the top-level crate.

**Implementation Steps (Detailed)**
1. Confirm `monolith_layers::prelude` exports all custom layers used in Python (`AddBias`, `MLP`, `LHUCTower`, `LogitCorrection`, `LayerNorm`, `GradNorm`, pooling layers, `MMoE`, `SNR`, feature cross/trans/seq layers).
2. Add missing exports or alias types in `monolith-rs/crates/monolith-layers/src/lib.rs` to mirror Python names (`DCNType` alias if needed).
3. Create an optional `LayerRegistry` (e.g., `HashMap<&'static str, LayerFactory>`) to mimic `keras_layers` if dynamic layer lookup is required; document that Python only registers patched Keras layers.
4. Implement a `LayerParams` trait or per-layer config builder to mirror Python `params()` (if used by config/codegen tooling).
5. Add a top-level `monolith-rs/crates/monolith/src/layers.rs` (or re-export from `monolith_layers`) to provide a single import path similar to Python.
6. Document that there is no exact Keras wildcard import equivalent; in Rust, this is replaced by explicit prelude exports and optional registry for dynamic construction.

**Tests (Detailed)**
- Python tests: None directly for `layers/__init__.py`.
- Rust tests:
  - Add a compile-time test that `monolith_layers::prelude::*` includes expected names (e.g., `Dense`, `MLP`, `AddBias`, pooling layers).
  - If a registry is added, include a test that registry contains expected layer names and no duplicates.
- Cross-language parity test:
  - (Optional) capture list of exported layer names in Python and compare to Rust prelude/registry list for overlap.

**Gaps / Notes**
- `keras_layers` is only populated for Keras classes missing `params`; if Rust introduces a registry, decide whether it should include only "patched" entries or the full set of Rust layers.
- Python explicitly suppresses all errors during reflection; Rust should avoid panicking on optional registry population.

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

### `monolith/native_training/layers/add_bias.py`
<a id="monolith-native-training-layers-add-bias-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 110
- Purpose/role: Keras layer that adds a learnable bias to inputs while handling `channels_first`/`channels_last` formats for 3D/4D/5D tensors.
- Key symbols/classes/functions: `AddBias(Layer)` with `build`, `call`, `get_config`.
- External dependencies: TensorFlow (`Layer`, `InputSpec`, `initializers`, `regularizers`, `tf.nn.bias_add`), Monolith utils (`get_ndim`, `int_shape`, `with_params`), layer utils (`check_dim`, `dim_size`), `monolith_export`.
- Side effects: Adds a trainable weight named `"bias"` during `build`; decorated with `@with_params` and `@monolith_export`.

**Required Behavior (Detailed)**
- Initialization:
  - `initializer = initializers.get(initializer) or tf.initializers.Zeros()`.
  - `regularizer = regularizers.get(regularizer)`.
  - `input_spec = InputSpec(min_ndim=2)`; `bias=None`.
- Build:
  - `shape = list(map(check_dim, input_shape[1:]))` (batch dim removed).
  - `check_dim(None) -> -1`, so unknown dims propagate to bias shape.
  - `self.add_weight(name='bias', shape=shape, dtype=tf.float32, initializer=initializer, regularizer=regularizer)`.
- Call:
  - `data_format = kwargs.get('data_format', 'channels_last')`.
  - Validate `data_format` is `"channels_first"` or `"channels_last"`; otherwise raise `ValueError('Unknown data_format: ' + str(data_format))`.
  - `bias_shape = int_shape(self.bias)` (tuple, `-1` for unknown dims).
  - If `len(bias_shape)` is not `1` and not `get_ndim(inputs) - 1`, raise:
    - `ValueError('Unexpected bias dimensions %d, expect to be 1 or %d dimensions' % (len(bias_shape), get_ndim(inputs)))`.
  - For `get_ndim(inputs) == 5`:
    - `channels_first`:
      - `len(bias_shape)==1`: reshape to `(1, C, 1, 1, 1)`.
      - `len(bias_shape)>1`: reshape to `(1, bias_shape[3]) + bias_shape[:3]`.
    - `channels_last`:
      - `len(bias_shape)==1`: reshape to `(1, 1, 1, C)`.
      - `len(bias_shape)>1`: reshape to `(1,) + bias_shape`.
  - For `get_ndim(inputs) == 4`:
    - `channels_first`:
      - `len(bias_shape)==1`: reshape to `(1, C, 1, 1)`.
      - `len(bias_shape)>1`: reshape to `(1, bias_shape[2]) + bias_shape[:2]`.
    - `channels_last`:
      - `len(bias_shape)==1`: `tf.nn.bias_add(inputs, bias, data_format='NHWC')`.
      - `len(bias_shape)>1`: reshape to `(1,) + bias_shape`.
  - For `get_ndim(inputs) == 3`:
    - `channels_first`:
      - `len(bias_shape)==1`: reshape to `(1, C, 1)`.
      - `len(bias_shape)>1`: reshape to `(1, bias_shape[1], bias_shape[0])`.
    - `channels_last`:
      - `len(bias_shape)==1`: reshape to `(1, 1, C)`.
      - `len(bias_shape)>1`: reshape to `(1,) + bias_shape`.
  - Else (2D or other): `tf.nn.bias_add(inputs, bias)`.
- Serialization:
  - `get_config()` serializes initializer/regularizer and merges with base config.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-layers/src/add_bias.rs`.
- Rust public API surface: `AddBias`, `DataFormat`, `forward_with_format`, builder-style setters.
- Data model mapping:
  - Python `initializer` → Rust `Initializer`.
  - Python `regularizer` → Rust `Regularizer`.
  - Python `data_format` string → Rust `DataFormat`.
- Feature gating: None; pure Rust.
- Integration points: Re-export `AddBias` from `monolith_layers::prelude` for parity with Python `layers` aggregator.

**Implementation Steps (Detailed)**
1. Enforce Python error cases in Rust:
   - Reject bias shapes not equal to `1` or `ndim-1` with the same message text.
   - Reject invalid `data_format` string on parse.
2. Verify reshape permutations match Python for 3D/4D/5D (channels-first permutations).
3. Match `tf.nn.bias_add` semantics for 4D channels-last and 2D inputs (broadcast + dtype behavior).
4. Decide how to handle unknown dimensions (`-1` in Python) in Rust, document and test.
5. Add `LayerParams` metadata or config serialization to mirror `with_params` and `get_config`.

**Tests (Detailed)**
- Python tests: `monolith/native_training/layers/add_bias_test.py`.
- Rust tests: `monolith-rs/crates/monolith-layers/tests/add_bias_test.rs` (new).
- Cross-language parity test:
  - Generate fixed bias + random tensors (3D/4D/5D) and compare outputs between Python and Rust.

**Gaps / Notes**
- Python allows `-1` dims in bias shape; Rust currently assumes known sizes.
- Python uses `tf.nn.bias_add` for 2D and 4D `channels_last`; Rust currently uses reshape + add.

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

### `monolith/native_training/layers/add_bias_test.py`
<a id="monolith-native-training-layers-add-bias-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 65
- Purpose/role: Unit tests for `AddBias` instantiation, serialization, and forward call in TF v1 session mode.
- Key symbols/classes/functions: `AddBiasTest.test_ab_instantiate`, `test_ab_serde`, `test_ab_call`.
- External dependencies: TensorFlow v1 session runtime, NumPy.
- Side effects: Uses `tf.compat.v1.disable_eager_execution()` in main guard; runs session initializers.

**Required Behavior (Detailed)**
- `test_ab_instantiate`:
  - Builds `layer_template = AddBias.params()`.
  - Copies params, sets `initializer = tf.initializers.Zeros()`, calls `instantiate()`.
  - Also instantiates with `AddBias(initializer=tf.initializers.Zeros())`.
  - Both constructions must succeed.
- `test_ab_serde`:
  - Instantiates via params.
  - `cfg = ins1.get_config()`, then `AddBias.from_config(cfg)`.
  - No assertion; should complete without error.
- `test_ab_call`:
  - Creates layer via params, sets `name` and `initializer`.
  - Creates variable input shape `(100, 10)` and computes `tf.reduce_sum(layer(data))`.
  - Runs in a session after `global_variables_initializer()`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-layers/tests/add_bias_test.rs`.
- Rust public API surface: `AddBias` constructor/builder, serialization, forward call.
- Data model mapping:
  - `AddBias.params()` ↔ Rust config builder or `LayerParams` metadata.
  - `get_config`/`from_config` ↔ serde round-trip for `AddBias`.
- Feature gating: None.
- Integration points: `monolith_layers::AddBias`.

**Implementation Steps (Detailed)**
1. Add a constructor test for `AddBias::new()` and builder methods.
2. Add serde round-trip test for `AddBias` config.
3. Add forward pass test with deterministic input and bias; assert output sum.
4. Keep tests CPU-only; no TF runtime required.

**Tests (Detailed)**
- Python tests: `monolith/native_training/layers/add_bias_test.py`.
- Rust tests: `monolith-rs/crates/monolith-layers/tests/add_bias_test.rs` (new).
- Cross-language parity test:
  - Compare output sum for fixed input/bias between Python and Rust.

**Gaps / Notes**
- Python tests are smoke tests without explicit assertions; Rust should add asserts for deterministic behavior.

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

### `monolith/native_training/layers/advanced_activations.py`
<a id="monolith-native-training-layers-advanced-activations-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 217
- Purpose/role: Defines activation layer wrappers, exports advanced activation classes, and provides `get/serialize/deserialize` helpers for activation identifiers.
- Key symbols/classes/functions:
  - Classes: `ReLU`, `LeakyReLU`, `ELU`, `Softmax`, `ThresholdedReLU`, `PReLU` (from TF), plus custom Layer wrappers `Tanh`, `Sigmoid`, `Sigmoid2`, `Linear`, `Gelu`, `Selu`, `Softsign`, `Softplus`, `Exponential`, `HardSigmoid`, `Swish`.
  - `get(identifier)`, `serialize(activation)`, `deserialize(identifier)`.
  - `__all__`, `__all_activations`, `ALL_ACTIVATION_NAMES`.
- External dependencies: TensorFlow Keras activations/layers, `types.MethodType`, Monolith `_params`, `monolith_export`.
- Side effects: Monkey-patches `params` method onto activation layer classes.

**Required Behavior (Detailed)**
- Class setup:
  - Defines lightweight `Layer` subclasses via `type(...)` for Tanh/Sigmoid/etc; each implements `call` with the corresponding TF activation function.
  - Adds `.params = MethodType(_params, cls)` for all activation classes listed (including TF-provided advanced activations).
- Export lists:
  - `__all__` includes names of all activation layers and wrappers.
  - `__all_activations` maps lowercase names (and synonyms like `hard_sigmoid`/`hardsigmoid`) to classes.
  - `ALL_ACTIVATION_NAMES = set(__all_activations.keys())`.
- `get(identifier)`:
  - `None` → `None`.
  - `str`:
    - If `identifier.lower()` in `__all_activations`: return a **new instance** of that class.
    - Else `eval(identifier)`; if dict, call `deserialize`; otherwise raise `TypeError`.
  - `dict`: call `deserialize`.
  - `callable`:
    - If has `params`, try `issubclass(identifier, Layer)`: if true, return new instance; else return identifier.
    - If `identifier` is a `Layer` instance: create new instance based on its class name.
    - Else try `identifier.__name__` and map to `__all_activations`; if not found return identifier.
  - Else: raise `TypeError('Could not interpret activation function identifier: ...')`.
- `serialize(activation)`:
  - Returns `repr(dict)` strings for known activation types; `None` otherwise.
  - Uses class-specific fields:
    - `LeakyReLU`/`ELU`: `alpha`.
    - `ReLU`: `max_value`, `negative_slope`, `threshold`.
    - `PReLU`: `alpha_initializer`, `alpha_regularizer`, `alpha_constraint`, `shared_axes` (uses `initializers.serialize` for all three in Python).
    - `Softmax`: `axis`.
    - `ThresholdedReLU`: `theta`.
- `deserialize(identifier)`:
  - Accepts dict or string repr of dict (via `eval`).
  - Requires `name` key; lowercases and looks up in `__all_activations`, pops `name`, and instantiates class with remaining kwargs.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-layers/src/activation.rs` and `activation_layer.rs`.
- Rust public API surface:
  - Activation structs (ReLU, LeakyReLU, ELU, Softmax, ThresholdedReLU, PReLU, Tanh, Sigmoid, Sigmoid2, Linear, GELU, SELU, Softplus, Softsign, Swish, Exponential, HardSigmoid).
  - `ActivationType` enum (in `mlp.rs`) and `ActivationLayer` wrapper for dynamic dispatch.
  - Add `activation::get`, `activation::serialize`, `activation::deserialize` or a dedicated registry module to mirror Python behavior.
- Data model mapping:
  - Python string identifiers ↔ Rust `ActivationType` (case-insensitive, include synonyms).
  - Python repr(dict) ↔ Rust serde JSON/YAML or explicit struct for config; if parity requires, accept Python-style dict strings.
- Feature gating: None.
- Integration points: `MLP` and any layer configs that accept activations.

**Implementation Steps (Detailed)**
1. Add a Rust activation registry mapping lowercased names and synonyms to constructors.
2. Implement `get(identifier)` variants:
   - Accept `&str`, `ActivationType`, or config struct; return `ActivationLayer`.
3. Implement `serialize` to return a Python-compatible dict representation (or document accepted Rust-native format and add translation).
4. Implement `deserialize` that can accept Python-style `repr(dict)` if needed for cross-language parity.
5. Ensure `params`-like metadata exists for activation classes (if using config builders).

**Tests (Detailed)**
- Python tests: `monolith/native_training/layers/advanced_activations_test.py`.
- Rust tests: `monolith-rs/crates/monolith-layers/tests/advanced_activations_test.rs` (new).
- Cross-language parity test:
  - For each name in `ALL_ACTIVATION_NAMES`, call `get` and compare forward outputs on fixed input.

**Gaps / Notes**
- Python uses `eval` on identifier strings for deserialize; Rust should avoid eval and instead accept a safe format, but parity requires handling Python-style repr strings.
- Python uses `initializers.serialize` for `alpha_regularizer`/`alpha_constraint` in `PReLU` serialization; verify whether this is a bug and whether Rust should mirror it.

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

### `monolith/native_training/layers/advanced_activations_test.py`
<a id="monolith-native-training-layers-advanced-activations-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 84
- Purpose/role: Exercises `advanced_activations.get`/`serialize` with identifiers and ensures activation layers run in a TF session.
- Key symbols/classes/functions: `serde`, `all_acts`, `raw_acts`, `lay_acts`, `ActivationsTest`.
- External dependencies: TensorFlow Keras activations/layers, TF v1 session mode.
- Side effects: Disables eager execution in main guard; runs session with variable initialization.

**Required Behavior (Detailed)**
- `serde(act)`:
  - `_act = get(act)`, `sered_act = serialize(_act)`, then `get(sered_act)` must succeed.
- `all_acts` list defines string identifiers for names to test.
- `raw_acts` list of Keras activation functions exists but is unused in tests.
- `lay_acts` list includes Keras layer instances (`ReLU`, `PReLU`, `ThresholdedReLU`, `ELU`, `Softmax`, `LeakyReLU`).
- Tests:
  - `test_get_from_str`: calls `serde` for each name in `all_acts`.
  - `test_get_from_layers`: calls `serde` for each layer instance in `lay_acts`.
  - `test_get_from_func`: loops `lay_acts` again (likely intended `raw_acts` but uses layers).
  - `test_params`: for each name, calls `cls = get(act).__class__` then `cls.params()`.
  - `test_call`: creates input `(100, 200)`, applies `get(act)` to each name, sums and evaluates in a session.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-layers/tests/advanced_activations_test.rs`.
- Rust public API surface: activation registry `get`, `serialize`, config params metadata.
- Data model mapping:
  - `all_acts` strings → Rust name lookup (`ActivationType` or registry).
  - `serialize` output → Rust serialization format (must accept Python repr strings if parity required).
- Feature gating: None.
- Integration points: `monolith_layers::activation` and `ActivationLayer`.

**Implementation Steps (Detailed)**
1. Add Rust tests to cover name lookup for all `all_acts` names.
2. Add serde round-trip test for each activation type.
3. Add test to ensure `params`/config metadata exists for each activation class.
4. Add forward test applying all activations to a fixed tensor and summing outputs.

**Tests (Detailed)**
- Python tests: `monolith/native_training/layers/advanced_activations_test.py`.
- Rust tests: `monolith-rs/crates/monolith-layers/tests/advanced_activations_test.rs` (new).
- Cross-language parity test:
  - Use identical inputs and compare output sums per activation name.

**Gaps / Notes**
- `test_get_from_func` likely intended to use `raw_acts` but currently uses `lay_acts`; Rust tests should mirror the current behavior, not the intent.

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

### `monolith/native_training/layers/agru.py`
<a id="monolith-native-training-layers-agru-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 295
- Purpose/role: Implements an Attention GRU (AGRU/AUGRU) cell and helper functions for static/dynamic RNN with attention scores.
- Key symbols/classes/functions: `AGRUCell`, `create_ta`, `static_rnn_with_attention`, `dynamic_rnn_with_attention`.
- External dependencies: TensorFlow internals (`rnn_cell_impl`, `tensor_array_ops`, `control_flow_ops`, `array_ops`, `math_ops`, `nn_ops`), Keras `Layer`, `InputSpec`, activations/initializers/regularizers, Monolith utils (`with_params`, `check_dim`, `dim_size`).
- Side effects: Creates trainable weights (`gates/*`, `candidate/*`) with initializer/regularizer; uses TensorArray and TF while_loop for dynamic RNN.

**Required Behavior (Detailed)**
- `AGRUCell` initialization:
  - `units` required; `att_type` must be `"AGRU"` or `"AUGRU"` (case-insensitive).
  - `activation = activations.get(activation or math_ops.tanh)`.
  - `initializer = tf.initializers.get(initializer) or tf.initializers.HeNormal()`.
  - `regularizer = regularizers.get(regularizer)`.
  - `input_spec` requires 3 inputs: `(x, state, att_score)`; `x` and `state` are 2D, `att_score` max_ndim=2.
- `build(inputs_shape)`:
  - `input_shape, state_shape, att_shape = inputs_shape`.
  - Assert `state_shape[-1] == units`.
  - `input_depth = check_dim(input_shape[-1])`; if `input_shape[-1] == -1`, raise `ValueError("Expected inputs.shape[-1] to be known, saw shape: ...")`.
  - Create weights:
    - `_gate_kernel`: shape `[input_depth + units, 2 * units]`.
    - `_gate_bias`: shape `[2 * units]`, initializer `Ones`.
    - `_candidate_kernel`: shape `[input_depth + units, units]`.
    - `_candidate_bias`: shape `[units]`, initializer `Ones`.
- `call((x, state, att_score))`:
  - `gate_inputs = matmul(concat([x, state], 1), _gate_kernel)`, bias add.
  - `value = sigmoid(gate_inputs)`; split into `r, u`.
  - `candidate = matmul(concat([x, r * state], 1), _candidate_kernel)`; bias add; `c = activation(candidate)`.
  - If `att_score is None`: standard GRU update: `(1 - u) * state + u * c`.
  - Else if `att_type == "AUGRU"`:
    - `u = (1 - att_score) * u`.
    - `new_h = u * state + (1 - u) * c`.
  - Else (`AGRU`):
    - `new_h = (1 - att_score) * state + att_score * c`.
  - Returns `(new_h, new_h)` (output and new state).
- `zero_state(batch_size, dtype)`:
  - In eager mode, caches last zero state to avoid recomputation.
  - Uses `_zero_state_tensors` from TF rnn_cell_impl with `backend.name_scope`.
- `get_config()` serializes `units`, `att_type`, `initializer`, `activation`, `regularizer`.
- `create_ta(name, size, dtype)` returns `TensorArray`.
- `static_rnn_with_attention(cell, inputs, att_scores, init_state=None)`:
  - `cell` must be `AGRUCell`.
  - If `init_state` is None, uses `cell.get_initial_state` if available, else `cell.zero_state`.
  - Transposes inputs to time-major, loops in Python, calls cell per time step with `att_scores[:, time]` reshaped to `(-1, 1)`.
  - Returns stacked outputs `(batch, time, hidden)` and final state.
  - Note: uses `dtype` variable in `get_initial_state` branch, but `dtype` is undefined (bug in Python).
- `dynamic_rnn_with_attention(cell, inputs, att_scores, parallel_iterations=1, swap_memory=True, init_state=None)`:
  - Same initialization rules as static (same undefined `dtype` issue).
  - Uses TensorArray + `control_flow_ops.while_loop`.
  - Outputs stacked and transposed back to batch-major.
  - Sets static shape `[None, time_steps, dim_size(outputs, -1)]`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-layers/src/agru.rs`.
- Rust public API surface:
  - Rust `AGRU` struct exists but is not a TF-style cell; needs parity with `AGRUCell` and with sequence outputs.
  - Add `AGRUCell`-like API (`forward_step`, `zero_state`, `state_size`, `output_size`).
  - Add `static_rnn_with_attention` / `dynamic_rnn_with_attention` helpers (pure Rust loops).
- Data model mapping:
  - Python `att_type` (`AGRU`/`AUGRU`) → Rust enum.
  - Python `activation` → Rust activation layer or function.
  - Python `initializer`/`regularizer` → Rust `Initializer`/`Regularizer`.
- Feature gating: None (pure Rust implementation).
- Integration points: DIEN/sequence models that expect AGRU outputs.

**Implementation Steps (Detailed)**
1. Extend `monolith_layers::agru` to support both AGRU and AUGRU update formulas and optional `att_score=None` (standard GRU).
2. Add `AGRUCell` struct mirroring Python weight shapes and bias initialization (gate/candidate splits).
3. Implement `static_rnn_with_attention`:
   - Accept inputs `[batch, time, dim]`, attention `[batch, time]`, optional initial state.
   - Return outputs for all timesteps and final state.
4. Implement `dynamic_rnn_with_attention`:
   - In Rust, this is a loop, but preserve behavior and output shape.
5. Match error semantics for unknown input depth; enforce input_dim known at build.
6. Add config serialization for `AGRUCell` (units, att_type, initializer, activation, regularizer).

**Tests (Detailed)**
- Python tests: `monolith/native_training/layers/agru_test.py`.
- Rust tests: `monolith-rs/crates/monolith-layers/tests/agru_test.rs` (new).
- Cross-language parity test:
  - Fix weights, input, attention; compare per-timestep outputs for AGRU and AUGRU modes.

**Gaps / Notes**
- Python `static_rnn_with_attention` and `dynamic_rnn_with_attention` reference `dtype` without definition; decide whether to mimic or correct in Rust.
- Rust `AGRU` currently returns only final hidden state and uses a different attention update formula; needs alignment.

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

### `monolith/native_training/layers/agru_test.py`
<a id="monolith-native-training-layers-agru-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 112
- Purpose/role: Smoke tests for `AGRUCell` instantiation, serde, and static/dynamic attention RNN helpers.
- Key symbols/classes/functions: `AGRUTest` methods `test_agru_instantiate`, `test_agru_serde`, `test_agru_call`, `test_agru_static_rnn_call`, `test_agru_dynamic_rnn_call`.
- External dependencies: TensorFlow v1 session mode, NumPy.
- Side effects: Disables eager execution and v2 behavior in main guard.

**Required Behavior (Detailed)**
- `test_agru_instantiate`:
  - Uses `AGRUCell.params()` to build `InstantiableParams`.
  - Sets `units=10`, `activation=sigmoid`, `initializer=GlorotNormal`, then `instantiate()`.
  - Also constructs directly with `AGRUCell(units=10, activation=sigmoid, initializer=HeUniform)`.
  - Both instantiations must succeed.
- `test_agru_serde`:
  - `cfg = AGRUCell(...).get_config()` then `AGRUCell.from_config(cfg)` must succeed.
- `test_agru_call`:
  - Inputs: `data` shape `(100, 100)`, `state` shape `(100, 10)`, `attr` shape `(100, 1)`.
  - Calls `layer((data, state, attr))`, gets `(output, new_state)`; sums `new_state`.
  - Runs in a session after variable initialization.
- `test_agru_static_rnn_call`:
  - Inputs: `data` shape `(100, 20, 10)`, `attr` shape `(100, 20)`.
  - Calls `static_rnn_with_attention`, receives `(outputs, final_state)`.
  - Sums `final_state` (not the full outputs).
- `test_agru_dynamic_rnn_call`:
  - Inputs: random `data` shape `(100, 20, 10)` and `attr` shape `(100, 20)`.
  - Calls `dynamic_rnn_with_attention`, receives `(outputs, final_state)` and sums `final_state`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-layers/tests/agru_test.rs`.
- Rust public API surface: `AGRUCell` (or equivalent), `static_rnn_with_attention`, `dynamic_rnn_with_attention`.
- Data model mapping:
  - Params-based instantiation ↔ Rust config/builder.
  - `get_config`/`from_config` ↔ serde round-trip.
- Feature gating: None.
- Integration points: `monolith_layers::agru`.

**Implementation Steps (Detailed)**
1. Add Rust tests for params/builder instantiation and serde round-trip.
2. Add forward step test for `AGRUCell` with fixed inputs, compare output sum.
3. Add static and dynamic RNN tests; validate final state sum against Python.
4. Mirror test input shapes and attention shapes from Python.

**Tests (Detailed)**
- Python tests: `monolith/native_training/layers/agru_test.py`.
- Rust tests: `monolith-rs/crates/monolith-layers/tests/agru_test.rs` (new).
- Cross-language parity test:
  - Fix weights and inputs and compare final state sums for static/dynamic helpers.

**Gaps / Notes**
- Python tests do not assert exact values; Rust tests should add deterministic assertions.

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

### `monolith/native_training/layers/dense.py`
<a id="monolith-native-training-layers-dense-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 307
- Purpose/role: Custom Dense layer with optional kernel normalization, optimizer attachment to variables, partitioned variable support, and inactive-ReLU monitoring.
- Key symbols/classes/functions: `Dense(Layer)` with `add_weight`, `get_variable`, `build`, `call`, `compute_output_shape`, `get_config`.
- External dependencies: TensorFlow/Keras internals (`core_ops.dense`, `variable_ops.PartitionedVariable`, `base_layer_utils`, `InputSpec`, `K.track_variable`, `tensor_shape`), Monolith utils (`with_params`, `get_uname`).
- Side effects:
  - Attaches `.optimizer` attribute to variables created.
  - Tracks split variables in layer trainable/non-trainable weights.
  - Emits summary histogram for inactive ReLU counts when enabled.

**Required Behavior (Detailed)**
- Initialization:
  - If `input_shape` not provided and `input_dim` is, convert to `input_shape=(input_dim,)`.
  - Sets `units`, `activation=activations.get(activation)`, `use_bias`, `kernel_initializer`, `bias_initializer`, `kernel_regularizer`, `bias_regularizer`, `allow_kernel_norm`, `kernel_norm_trainable`, `partitioner`, `inactive_relu_monitor`, `inactive_relu_monitor_decay`, `optimizer`.
  - `input_spec = InputSpec(min_ndim=2)`, `supports_masking=True`.
- `add_weight` override:
  - Calls `super().add_weight(...)`, then sets `var.optimizer = self.optimizer`.
  - If `PartitionedVariable`, set optimizer on each shard.
- `get_variable` helper:
  - Wraps `tf.compat.v1.get_variable` inside current name scope with AUTO_REUSE.
  - Sets optimizer on variable(s).
  - Manually tracks variables with `K.track_variable` and appends to `_trainable_weights` or `_non_trainable_weights`, including split/partitioned variables.
- `build(input_shape)`:
  - Ensures dtype is floating/complex; otherwise `TypeError("Unable to build `Dense` layer with non-floating point dtype %s")`.
  - Requires last dimension known; otherwise `ValueError("The last dimension of the inputs to `Dense` should be defined. Found `None`.")`.
  - Creates kernel variable using `get_variable` seeded by `kernel_initializer` output.
  - If `allow_kernel_norm`:
    - Normalize kernel with `tf.nn.l2_normalize(axis=0, epsilon=1e-6)`.
    - If `kernel_norm_trainable`, create `trainable_kernel_norm` initialized with `tf.linalg.norm(init_kernel, axis=0)` and multiply normalized kernel by it.
  - If `use_bias`, add bias weight; else `bias=None`.
  - If `inactive_relu_monitor` and activation is ReLU:
    - Create non-trainable `inactive_relu_count_moving_avg` variable under `METRIC_VARIABLES` and `GLOBAL_VARIABLES`.
- `call(inputs)`:
  - Uses `core_ops.dense(inputs, kernel, bias, activation, dtype=compute_dtype)`.
  - If `inactive_relu_monitor`:
    - `inactive_relu_count = units - count_nonzero(output, axis=0)`.
    - Logs histogram `inactive_relu_count_moving_avg`.
    - Updates moving average with decay and uses control dependencies.
- `compute_output_shape`:
  - Requires last dim defined; otherwise `ValueError("The innermost dimension of input_shape must be defined, but saw: %s")`.
  - Output shape = input_shape[:-1] + units.
- `get_config`:
  - Serializes units, activation, use_bias, initializers, regularizers, `allow_kernel_norm`, `kernel_norm_trainable`, `partitioner`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-layers/src/dense.rs`.
- Rust public API surface:
  - `Dense` currently implements a linear layer with optional kernel norm (no activation in the layer).
  - Add a `DenseConfig` or wrapper to include activation and inactive-ReLU monitoring.
- Data model mapping:
  - Python `activation` → Rust `ActivationType` or `ActivationLayer` (in `mlp.rs`/`activation_layer.rs`).
  - Python `kernel_norm_trainable` → Rust `kernel_norm_trainable`.
  - Python `partitioner` and `optimizer` do not exist in Rust; require explicit non-TF equivalents or document omission.
- Feature gating: None.
- Integration points: MLP and any model configs that reference `Dense` directly.

**Implementation Steps (Detailed)**
1. Decide parity approach:
   - Option A: Add activation inside `Dense` (match Python call signature).
   - Option B: Keep linear `Dense` and ensure all call sites add `ActivationLayer` explicitly (document difference).
2. Add kernel norm behavior to match TF:
   - Normalize weights along axis 0, epsilon 1e-6.
   - If trainable, scale by per-output norm.
3. Add inactive ReLU monitoring equivalent (optional):
   - Track per-unit zero counts and exponential moving average; integrate with Rust metrics/logging.
4. Add config serialization to include activation, kernel/bias initializers, regularizers, allow_kernel_norm, kernel_norm_trainable.
5. Mirror error messages for invalid input dtypes and missing last dimension (as close as Rust allows).

**Tests (Detailed)**
- Python tests: `monolith/native_training/layers/dense_test.py`.
- Rust tests: `monolith-rs/crates/monolith-layers/tests/dense_test.rs` (new).
- Cross-language parity test:
  - Fix weights/bias and compare outputs for activation on/off and kernel_norm modes.

**Gaps / Notes**
- Python attaches `.optimizer` to TF variables and supports partitioned variables; Rust has no equivalent.
- Python Dense includes activation inside the layer; Rust `Dense` is linear-only today.

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

### `monolith/native_training/layers/dense_test.py`
<a id="monolith-native-training-layers-dense-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 147
- Purpose/role: Tests Dense instantiation, serialization, forward, kernel norm, inactive-ReLU monitoring, and variable partitioning.
- Key symbols/classes/functions: `DenseTest` methods `test_dense_instantiate`, `test_dense_serde`, `test_dense_call`, `test_dense_kernel_norm_call`, `test_inactive_relu_monitor`, `test_dense_with_explicit_partition`, `test_dense_with_implicit_partition`.
- External dependencies: TensorFlow v1 session mode, NumPy.
- Side effects: Uses graph collections and variable partitioning.

**Required Behavior (Detailed)**
- `test_dense_instantiate`:
  - Builds `Dense.params()` template, sets `units=100`, `activation=sigmoid`, `kernel_initializer=GlorotNormal`, instantiates.
  - Also constructs `Dense(...)` directly; both must succeed.
- `test_dense_serde`:
  - Instantiates via params, calls `get_config`, and `Dense.from_config(cfg)`.
- `test_dense_call`:
  - Creates Dense with sigmoid activation, input `(100, 100)` ones; sums output and runs session.
- `test_dense_kernel_norm_call`:
  - Dense with `allow_kernel_norm=True`, `kernel_norm_trainable=True`; runs forward without errors.
- `test_inactive_relu_monitor`:
  - Dense with `activation=relu` and `inactive_relu_monitor=True`.
  - After calling on a constant input, asserts graph contains node name `Dense/inactive_relu_count_moving_avg_1`.
- `test_dense_with_explicit_partition`:
  - Dense with explicit `partitioner` and kernel_norm enabled.
  - Input shape `(100, 294)`; validates output shape `(100, 1024)`.
  - Collects per-shard kernel dims (expected `[59, 59, 59, 59, 58]` but not asserted).
- `test_dense_with_implicit_partition`:
  - Uses `variable_scope` partitioner; Dense with `partitioner=None` to inherit scope.
  - Verifies kernel shard dims equal `[59, 59, 59, 59, 58]`.
  - Validates output shape `(100, 1024)`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-layers/tests/dense_test.rs`.
- Rust public API surface: `Dense`, activation handling, kernel norm config.
- Data model mapping:
  - Params-based instantiation ↔ Rust builder/config.
  - `get_config`/`from_config` ↔ serde round-trip.
- Feature gating: None.
- Integration points: `monolith_layers::dense`.

**Implementation Steps (Detailed)**
1. Add Rust tests for constructor + config serde round-trip.
2. Add forward tests for base Dense and kernel_norm-enabled Dense.
3. If activation is moved out of Dense in Rust, adapt tests to apply activation separately but keep parity cases documented.
4. Decide how to mirror partitioner behavior:
   - If not supported, add explicit test that documents the unsupported feature.
5. Add inactive-ReLU monitoring metrics if implemented; otherwise document absence.

**Tests (Detailed)**
- Python tests: `monolith/native_training/layers/dense_test.py`.
- Rust tests: `monolith-rs/crates/monolith-layers/tests/dense_test.rs` (new).
- Cross-language parity test:
  - Fixed weights/bias and compare output sums for kernel_norm on/off.

**Gaps / Notes**
- Python partitioner behavior has no Rust equivalent; needs explicit parity plan or documented limitation.
- The expected partition shard sizes are implicit to TF partitioner; Rust may not replicate.

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

### `monolith/native_training/layers/feature_cross.py`
<a id="monolith-native-training-layers-feature-cross-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 805
- Purpose/role: Collection of feature-crossing layers: GroupInt/FFM, AllInt, CDot, CAN, DCN, CIN.
- Key symbols/classes/functions: `GroupInt` (alias `FFM`), `AllInt`, `CDot`, `CAN`, `DCN`, `CIN`.
- External dependencies: TensorFlow/Keras (`Layer`, `Conv1D`, activations/initializers/regularizers), Monolith layers (`MLP`), layer utils (`merge_tensor_list`, `DCNType`, `check_dim`, `dim_size`), `layer_ops.ffm`, TF internals (`variable_ops.PartitionedVariable`, `base_layer_utils`, `K.track_variable`).
- Side effects: Creates multiple trainable weights and nested Keras layers (MLP, Conv1D); uses TF variable tracking for split variables; logs via `absl.logging` (imported).

**Required Behavior (Detailed)**
- `GroupInt` (aka `FFM`):
  - Inputs: tuple `(left_fields, right_fields)` where each is list of tensors.
  - Concats left/right along axis=1, then calls `ffm(left, right, dim_size, int_type)`.
  - `interaction_type` in `{'multiply', 'dot'}`; `use_attention` only valid for `multiply`.
  - If `use_attention`: reshape to `(bs, num_feature, emb_dim)`, run MLP to get attention `(bs, num_feature, 1)`, apply elementwise weighting; output reshaped to `(bs, num_feature * emb_dim)`.
  - Returns list `[ffm_embeddings]` if `keep_list` else tensor.
  - Config includes interaction_type, attention_units, activation, initializer, regularizer, out_type, keep_list.
- `AllInt`:
  - Inputs: `embeddings` shape `[batch, num_feat, emb_size]`.
  - Builds kernel shape `(num_feat, cmp_dim)` and optional bias `(cmp_dim,)`.
  - Call: transposes embeddings to `[batch, emb_size, num_feat]`, computes `feature_comp = transposed @ kernel` (+bias), then `interaction = embeddings @ feature_comp` to get `[batch, num_feat, cmp_dim]`.
  - Returns `merge_tensor_list(interaction, merge_type=out_type, keep_list=keep_list)`.
- `CDot`:
  - Build: stores `_num_feature`, `_emd_size`, creates `project_weight` `(num_feature, project_dim)` and `compress_tower` MLP with output dims `compress_units + [emd_size * project_dim]`.
  - Call:
    - Project input: `(bs, emb_size, num_feature) @ project_weight` → `(bs, emb_size, project_dim)`.
    - Flatten and run compress MLP → `compressed` `(bs, emb_size * project_dim)`.
    - Cross: `inputs @ reshape(compressed, (bs, emb_size, project_dim))` → `(bs, num_feature, project_dim)`, flatten to `(bs, num_feature * project_dim)`.
    - Output: `concat([crossed, compressed], axis=1)`.
- `CAN`:
  - Inputs: `(user_emb, item_emb)`.
  - `item_emb` is split into alternating weight/bias tensors for `layer_num` layers; expects size `u_emb_size*(u_emb_size+1) * layer_num`.
  - Handles four shape cases based on `is_seq` and `is_stacked`, reshaping weights/bias accordingly.
  - Applies `layer_num` iterations of `user_emb = activation(user_emb @ weight + bias)` (or linear if activation is None).
  - Output reduces/squeezes based on `is_seq/is_stacked`.
- `DCN`:
  - Supports types: `Vector`, `Matrix`, `Mixed` (from `DCNType`).
  - `Vector`: kernel shape `(dim,1)` per layer; update `xl = x0 * (xl @ w) + b + xl`.
  - `Matrix`: kernel shape `(dim,dim)` per layer; update `xl = x0 * (xl @ W + b) + xl`.
  - `Mixed`: per layer, per-expert low-rank factors `U,V,C` (dims `dim x low_rank`), gating `G` (`dim x 1`), bias; computes expert outputs and softmax-gated mixture; adds residual `+ xl`.
  - Optional `allow_kernel_norm` in `get_variable`: normalizes var (axis=0, eps=1e-6) and multiplies by trainable norm initialized with `tf.norm(var_init, axis=0)`.
  - Optional dropout during `TRAIN` mode: `tf.nn.dropout(xl, rate=1-keep_prob)`.
- `CIN`:
  - Inputs: `[batch, num_feat, emb_size]`, uses `Conv1D` per layer (`hidden_uints`).
  - For each layer: compute `zl = einsum('bdh,bdm->bdhm', xl, x0)`, reshape to `(bs, emb_size, last_hidden_dim * num_feat)`, apply Conv1D.
  - Concatenate `reduce_sum` of each layer output along emb_size: `concat([sum(hi, axis=1) ...], axis=1)`.

**Rust Mapping (Detailed)**
- Target crate/module:
  - `monolith-rs/crates/monolith-layers/src/feature_cross.rs` (GroupInt/AllInt/CDot/CAN/CIN).
  - `monolith-rs/crates/monolith-layers/src/dcn.rs` (DCN).
- Rust public API surface:
  - `GroupInt`, `AllInt`, `CDot`, `CAN`, `CIN` in `feature_cross.rs`.
  - `CrossNetwork`/`CrossLayer` in `dcn.rs` for DCN variants.
- Data model mapping:
  - Python `DCNType` → Rust `DCNMode` (vector/matrix/mixed).
  - `interaction_type` → Rust `GroupIntType`.
  - `merge_tensor_list` → Rust `merge_tensor_list` / `MergeType`.
- Feature gating: GPU-accelerated paths in Rust when `cuda`/`metal` features enabled (if used).
- Integration points: MLP, merge utils, embedding and pooling layers.

**Implementation Steps (Detailed)**
1. Verify each Python layer has a Rust counterpart with matching defaults and shapes.
2. Align `GroupInt` attention path with Python’s MLP attention (last dim must be 1).
3. Ensure `AllInt` and `CDot` matmul/reshape orderings match Python exactly.
4. Implement CAN’s `is_seq` / `is_stacked` shape logic and weight/bias splitting.
5. Map DCN modes and kernel norm behavior; match gating and mixture logic for `Mixed`.
6. Ensure CIN’s einsum/reshape/Conv1D logic matches; if Conv1D is missing, emulate with 1x1 conv via matmul.
7. Add config serialization parity for each layer (activation, initializer, regularizer, units, dims).

**Tests (Detailed)**
- Python tests: `monolith/native_training/layers/feature_cross_test.py`.
- Rust tests: `monolith-rs/crates/monolith-layers/tests/feature_cross_test.rs` (new).
- Cross-language parity test:
  - Fix small inputs and compare outputs for each layer variant (GroupInt, AllInt, CDot, CAN, DCN, CIN).

**Gaps / Notes**
- Python uses TF internals for split variable tracking in DCN kernel_norm path; Rust does not have a direct analogue.
- Some layers (e.g., CDot/CIN) depend on Keras `Conv1D`; ensure Rust kernel shapes/stride/activation match.

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

### `monolith/native_training/layers/feature_cross_test.py`
<a id="monolith-native-training-layers-feature-cross-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 286
- Purpose/role: Smoke tests for feature crossing layers (GroupInt/AllInt/CDot/CAN/DCN/CIN).
- Key symbols/classes/functions: `FeatureCrossTest` methods for instantiate/serde/call per layer.
- External dependencies: TensorFlow v1 session mode, NumPy.
- Side effects: Disables v2 behavior in main guard; runs TF sessions.

**Required Behavior (Detailed)**
- GroupInt:
  - Instantiate via params and direct constructor.
  - `test_groupint_call`: left list of 5 tensors `(100,10)`, right list of 3 tensors `(100,10)`.
  - `test_groupint_attention_call`: same shapes with attention MLP.
- AllInt:
  - Instantiate/serde with `cmp_dim=4`.
  - Call on input `(100,10,10)`.
- CDot:
  - Instantiate/serde with `project_dim=8`, `compress_units=[128,256]`, `activation='tanh'`.
  - Call on input `(100,10,10)`.
- CAN:
  - Instantiate/serde with `layer_num=8`.
  - `test_can_seq_call`: user `(128,10,12,10)`, item `(128,220)`.
  - `test_can_call`: user `(128,10,10)`, item `(128,220)`.
- DCN:
  - Instantiate/serde for `dcn_type='matrix'`, `use_dropout=True`, `keep_prob=0.5`.
  - Call for vector/matrix/mixed modes; input `(128,10,10)`, kernel_norm enabled.
- CIN:
  - Instantiate/serde with `hidden_uints=[10,5]`, activation configured.
  - Call on input `(128,10,10)`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-layers/tests/feature_cross_test.rs`.
- Rust public API surface: `GroupInt`, `AllInt`, `CDot`, `CAN`, `CIN`, `DCN` equivalents.
- Data model mapping:
  - Params-based instantiation ↔ Rust config/builder.
  - `get_config`/`from_config` ↔ serde round-trip.
- Feature gating: None.
- Integration points: `monolith_layers::feature_cross` and `monolith_layers::dcn`.

**Implementation Steps (Detailed)**
1. Add Rust tests for each layer’s constructor and config serialization.
2. Add forward tests with same input shapes as Python.
3. For DCN, include tests for vector/matrix/mixed modes with kernel_norm on.
4. For CAN, enforce item size consistency with `layer_num`/`u_emb_size`.

**Tests (Detailed)**
- Python tests: `monolith/native_training/layers/feature_cross_test.py`.
- Rust tests: `monolith-rs/crates/monolith-layers/tests/feature_cross_test.rs` (new).
- Cross-language parity test:
  - Fix weights and inputs; compare output sums per layer.

**Gaps / Notes**
- Python tests are smoke tests; Rust should add deterministic assertions.

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

### `monolith/native_training/layers/feature_seq.py`
<a id="monolith-native-training-layers-feature-seq-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 361
- Purpose/role: Sequence feature models: DIN attention, DIEN interest evolution, DMR_U2I sequence matching.
- Key symbols/classes/functions: `DIN`, `DIEN`, `DMR_U2I`.
- External dependencies: TensorFlow/Keras (`Layer`, `Dense`, `GRUCell`, activations/initializers/regularizers), Monolith layers (`MLP`, `AGRUCell`, `dynamic_rnn_with_attention`), `monolith_export`, `with_params`.
- Side effects: Adds nested layer weights/regularization losses; uses TF summary in DIN when mask provided.

**Required Behavior (Detailed)**
- `DIN`:
  - Inputs: `queries` `[B,H]`, `keys` `[B,T,H]`; optional `mask` in kwargs.
  - Builds MLP (`dense_tower`) with `hidden_units` (last dim must be 1).
  - Call:
    - Tile `queries` to `[B,T,H]`.
    - `din_all = concat([q, k, q-k, q*k], axis=-1)` -> `[B,T,4H]`.
    - `attention_weight = dense_tower(din_all)` -> `[B,T,1]`.
    - If `decay`, divide by `sqrt(H)`.
    - If `mask`: zero out masked positions and emit summary histogram `{name}_attention_outputs`.
    - If `mode == 'sum'`: `attention_out = matmul(attention_weight, keys, transpose_a=True)` -> `[B,1,H]`, squeeze to `[B,H]`.
    - Else: elementwise `keys * attention_weight` -> `[B,T,H]`.
- `DIEN`:
  - Builds:
    - GRUCell for interest extraction (`gru_cell`).
    - AGRUCell (`augru_cell`) for interest evolution (note: `att_type` argument exists but build hard-codes `att_type='AGRU'`).
    - Attention weight matrix `weight` `(num_units, num_units)`.
  - `_attention(queries, keys)`:
    - `query_weight = matmul(queries, weight, transpose_b=True)` reshape to `[B, H, 1]`.
    - `logit = squeeze(matmul(keys, query_weight))` -> `[B,T]`.
    - `softmax(logit)` returns attention scores.
  - `call`:
    - Accepts `queries` and `keys` from args/kwargs (mask optional but unused).
    - `outputs = dynamic_rnn(gru_cell, keys)` -> `[B,T,H]`.
    - `attn_scores = _attention(queries, outputs)` -> `[B,T]`.
    - `_, final_state = dynamic_rnn_with_attention(augru_cell, outputs, attn_scores)`.
    - Returns `final_state` `[B,H]`.
- `DMR_U2I`:
  - Build: `pos_emb (seq_len, cmp_dim)`, `emb_weight (ue_size, cmp_dim)`, `z_weight (cmp_dim,1)`, `bias (cmp_dim)`, `linear Dense` to `ie_size`.
  - Call:
    - `emb_cmp = user_seq @ emb_weight`.
    - `comped = pos_emb + emb_cmp + bias`.
    - `alpha = softmax(comped @ z_weight, axis=1)` -> `[B,seq_len,1]`.
    - `user_seq_merged = squeeze(transpose(user_seq) @ alpha)` -> `[B, ue_size]`.
    - `user_seq_merged = linear(user_seq_merged)` -> `[B, ie_size]`.
    - Output `user_seq_merged * items` (elementwise).

**Rust Mapping (Detailed)**
- Target crate/module:
  - `monolith-rs/crates/monolith-layers/src/din.rs` (DIN).
  - `monolith-rs/crates/monolith-layers/src/dien.rs` (DIEN).
  - `monolith-rs/crates/monolith-layers/src/dmr.rs` (DMR_U2I).
- Rust public API surface:
  - `DINAttention`/`DINConfig`, `DIENLayer`/`DIENConfig`, `DMRU2I`.
- Data model mapping:
  - Python `mode` (`sum` vs elementwise) → Rust `DINOutputMode`.
  - Activation strings → Rust `ActivationType`.
  - AGRU/GRU cell configs → Rust GRU/AUGRU implementations.
- Feature gating: None.
- Integration points: AGRU, MLP, Dense, activation registry.

**Implementation Steps (Detailed)**
1. Verify DIN attention math and mask handling match Python (including decay scaling and summary logging).
2. Ensure DIEN uses the same attention formula; decide whether to respect Python’s `att_type` parameter (currently ignored).
3. Align DIEN to use AGRU vs AUGRU to match Python behavior.
4. Implement DMR_U2I using Dense + activation as in Python, including position embeddings.
5. Add config serialization for all three layers.

**Tests (Detailed)**
- Python tests: `monolith/native_training/layers/feature_seq_test.py`.
- Rust tests: `monolith-rs/crates/monolith-layers/tests/feature_seq_test.rs` (new).
- Cross-language parity test:
  - Fixed inputs and weights; compare outputs for DIN (sum/elementwise), DIEN, and DMR_U2I.

**Gaps / Notes**
- DIEN’s `att_type` argument is not used in build (hard-coded `'AGRU'`); decide whether to mirror this bug or fix with a flag.

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

### `monolith/native_training/layers/feature_seq_test.py`
<a id="monolith-native-training-layers-feature-seq-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 126
- Purpose/role: Smoke tests for DIN, DIEN, and DMR_U2I layers.
- Key symbols/classes/functions: `FeatureSeqTest` methods for instantiate/serde/call.
- External dependencies: TensorFlow v1 session mode, NumPy.
- Side effects: Runs TF sessions; eager disabled in main guard.

**Required Behavior (Detailed)**
- DIN:
  - Instantiate via params and direct constructor (`hidden_units=[10,1]`).
  - `test_din_call`: query `(100,10)`, keys `(100,15,10)`.
- DIEN:
  - Instantiate via params and direct constructor (`num_units=10`).
  - `test_dien_call`: query `(100,10)`, keys `(100,15,10)`.
- DMR_U2I:
  - Instantiate via params and direct constructor (`cmp_dim=10`, `activation='relu'`).
  - `test_dmr_call`: query `(100,10)`, keys `(100,15,10)`.
- All tests compute sum of outputs and run session initialization.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-layers/tests/feature_seq_test.rs`.
- Rust public API surface: `DINAttention`/`DINConfig`, `DIENLayer`, `DMRU2I`.
- Data model mapping:
  - Params-based instantiation ↔ Rust config/builder.
  - `get_config`/`from_config` ↔ serde round-trip.
- Feature gating: None.
- Integration points: `monolith_layers::din`, `monolith_layers::dien`, `monolith_layers::dmr`.

**Implementation Steps (Detailed)**
1. Add Rust tests for constructor and config serialization for each layer.
2. Add forward tests with the same input shapes.
3. Add deterministic assertions (output shapes and sums) for parity.

**Tests (Detailed)**
- Python tests: `monolith/native_training/layers/feature_seq_test.py`.
- Rust tests: `monolith-rs/crates/monolith-layers/tests/feature_seq_test.rs` (new).
- Cross-language parity test:
  - Fix weights and inputs; compare output sums for DIN, DIEN, DMR_U2I.

**Gaps / Notes**
- Python tests do not assert numeric values; Rust should add explicit assertions.

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

### `monolith/native_training/layers/feature_trans.py`
<a id="monolith-native-training-layers-feature-trans-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 340
- Purpose/role: Feature transformation layers: AutoInt (self-attention), iRazor (feature/embedding dimension selection), SeNet (feature re-weighting).
- Key symbols/classes/functions: `AutoInt`, `iRazor`, `SeNet`.
- External dependencies: TensorFlow/Keras (`Layer`, `InputSpec`, initializers/regularizers), Monolith (`MLP`, `add_layer_loss`, `merge_tensor_list`, `with_params`, `check_dim`/`dim_size`).
- Side effects: Emits TF summary histogram for iRazor NAS weights; adds auxiliary loss via `add_layer_loss`.

**Required Behavior (Detailed)**
- `AutoInt`:
  - Input shape `[B, num_feat, emb_dim]` (3D).
  - For `layer_num` iterations:
    - `attn = softmax(autoint_input @ autoint_input^T)` → `[B, num_feat, num_feat]`.
    - `autoint_input = attn @ autoint_input` → `[B, num_feat, emb_dim]`.
  - Output via `merge_tensor_list` with `out_type` and `keep_list`.
- `iRazor`:
  - `nas_space` defines embedding dimension groups; `rigid_masks` is a constant `[nas_len, emb_size]` with grouped 1s.
  - Build: `nas_logits` weight shape `(num_feat, nas_len)`.
  - Call:
    - `nas_weight = softmax(nas_logits / t)`; histogram summary `"nas_weight"`.
    - `soft_masks = nas_weight @ rigid_masks` → `[num_feat, emb_size]`.
    - If `feature_weight` provided, compute `nas_loss = feature_weight @ sum(soft_masks, axis=1)` and call `add_layer_loss`.
    - `out_embeds = embeds * soft_masks`.
    - Return `merge_tensor_list(out_embeds, out_type, keep_list)`.
- `SeNet`:
  - If inputs is a tensor `[B, num_feat, emb_dim]`: `sequeeze_embedding = reduce_mean(inputs, axis=2)`.
  - If inputs is list of tensors:
    - `on_gpu=True`: use `segment_sum` on concatenated embeddings and lens to compute means.
    - Else: `sequeeze_embedding = concat([reduce_mean(embed, axis=1)] ...)`.
  - `cmp_tower` MLP outputs feature weights of shape `[B, num_feat]`.
  - For tensor input: reshape weights to `[B, num_feat, 1]` and multiply.
  - For list input: split weights and multiply per tensor.
  - Output via `merge_tensor_list` with `num_feature`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-layers/src/feature_trans.rs` (AutoInt, IRazor, SeNet).
- Rust public API surface:
  - `AutoInt` + config, `IRazor`, `SENetLayer` (if present) or equivalent.
- Data model mapping:
  - `out_type` → `MergeType`.
  - `feature_weight` auxiliary loss → Rust loss registry or explicit return.
- Feature gating: None; GPU optimization optional.
- Integration points: merge utils, MLP, loss aggregation.

**Implementation Steps (Detailed)**
1. Align AutoInt attention math and merge semantics.
2. Implement iRazor rigid/soft mask logic and optional feature-weighted auxiliary loss.
3. Implement SeNet for both tensor and list inputs; include `on_gpu` optimization if possible.
4. Add config serialization for each layer (including `nas_space`, `t`, `cmp_dim`, `num_feature`).

**Tests (Detailed)**
- Python tests: `monolith/native_training/layers/feature_trans_test.py`.
- Rust tests: `monolith-rs/crates/monolith-layers/tests/feature_trans_test.rs` (new).
- Cross-language parity test:
  - Fix embeddings and weights; compare outputs for AutoInt, iRazor, SeNet.

**Gaps / Notes**
- Python iRazor uses `add_layer_loss` to register aux loss; Rust needs an equivalent or explicit API.

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

### `monolith/native_training/layers/feature_trans_test.py`
<a id="monolith-native-training-layers-feature-trans-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 140
- Purpose/role: Smoke tests for AutoInt, SeNet, and iRazor layers.
- Key symbols/classes/functions: `FeatureTransTest` methods for instantiate/serde/call.
- External dependencies: TensorFlow v1 session mode, NumPy.
- Side effects: Runs TF session after variable init.

**Required Behavior (Detailed)**
- AutoInt:
  - Instantiate via params and direct constructor (`layer_num=1`).
  - `test_autoint_call`: input `(100,10,10)`, `layer_num=2`.
- SeNet:
  - Instantiate/serde with `num_feature=10`, `cmp_dim=4`, custom initializers.
  - `test_senet_call`: input `(100,10,10)`.
- iRazor:
  - Instantiate/serde with `nas_space=[0,2,5,7,10]`, `t=0.08`.
  - `test_irazor_call`: input `(100,10,10)`.
- All tests sum outputs and run session initialization.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-layers/tests/feature_trans_test.rs`.
- Rust public API surface: `AutoInt`, `IRazor`, `SeNet` equivalents.
- Data model mapping:
  - Params-based instantiation ↔ Rust config/builder.
  - `get_config`/`from_config` ↔ serde round-trip.
- Feature gating: None.
- Integration points: `monolith_layers::feature_trans`.

**Implementation Steps (Detailed)**
1. Add Rust tests for constructor and config serialization for each layer.
2. Add forward tests with the same input shapes.
3. Add deterministic assertions on output shapes/sums.

**Tests (Detailed)**
- Python tests: `monolith/native_training/layers/feature_trans_test.py`.
- Rust tests: `monolith-rs/crates/monolith-layers/tests/feature_trans_test.rs` (new).
- Cross-language parity test:
  - Fix weights and inputs; compare output sums for AutoInt, SeNet, iRazor.

**Gaps / Notes**
- Python tests are smoke tests without numeric assertions; Rust should add explicit checks.

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

### `monolith/native_training/layers/layer_ops.py`
<a id="monolith-native-training-layers-layer-ops-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 131
- Purpose/role: Python wrappers around custom TensorFlow ops (FFM, FeatureInsight, MonolithFidCounter) and their gradients.
- Key symbols/classes/functions: `ffm`, `feature_insight`, `fid_counter` and registered gradients `_ffm_grad`, `_feature_insight`, `_fid_counter_grad`.
- External dependencies: `gen_monolith_ops` custom op library, TensorFlow gradient registry.
- Side effects: Registers gradients for custom ops; uses TF summary in downstream layers.

**Required Behavior (Detailed)**
- `ffm(left, right, dim_size, int_type='multiply')`:
  - Calls `layer_ops_lib.FFM` with given attrs and returns the op output.
  - `int_type` determines multiply vs dot behavior.
- `_ffm_grad` (gradient for `FFM`):
  - Calls `layer_ops_lib.FFMGrad` with `grad`, `left`, `right`, `dim_size`, `int_type`.
  - Returns `(left_grad, right_grad)`.
- `feature_insight(input_embedding, weight, segment_sizes, aggregate=False)`:
  - Asserts `segment_sizes` provided and `input_embedding.shape[-1] == weight.shape[0]`.
  - Calls `FeatureInsight` custom op.
  - If `aggregate=True`:
    - Builds `segment_ids` of length `k * num_feature` where `k = weight.shape[-1]`.
    - Returns `transpose(segment_sum(transpose(out * out), segment_ids))`.
  - Else returns `out` directly.
- `_feature_insight` (gradient for `FeatureInsight`):
  - Calls `FeatureInsightGrad` with `grad`, `input`, `weight`, `segment_sizes`, `K`.
  - Returns gradients for input_embedding and weight.
- `fid_counter(counter, counter_threshold, step=1.0)`:
  - Calls `MonolithFidCounter` op with `counter`, `step`, `counter_threshold`.
  - Adds `step` to counter, then clamps at threshold.
  - Docstring notes counter slice should use `SgdOptimizer(1.0)` and suggests `Fp32Compressor`.
- `_fid_counter_grad`:
  - Gradient is `-step` (as a constant) until `counter >= counter_threshold`, then zero.

**Rust Mapping (Detailed)**
- Target crate/module: custom ops would live in `monolith-rs/crates/monolith-tf` (TF runtime) or in native Rust layers (for pure Rust path).
- Rust public API surface:
  - Provide equivalents for `ffm`, `feature_insight`, `fid_counter` if needed by higher-level layers.
- Data model mapping:
  - `int_type` → Rust enum for multiply/dot.
  - `segment_sizes` and `aggregate` → explicit API arguments.
- Feature gating: TF runtime only for custom ops unless reimplemented in Rust.
- Integration points: `feature_cross.GroupInt` uses `ffm`; other code may rely on `feature_insight` or `fid_counter`.

**Implementation Steps (Detailed)**
1. Decide whether to reimplement `FFM`, `FeatureInsight`, and `MonolithFidCounter` in Rust or provide TF-runtime wrappers.
2. If TF runtime is used, expose safe Rust bindings and gradient equivalents.
3. For pure Rust path, implement `ffm` and its backward, plus feature_insight/ fid_counter logic.
4. Add tests for forward and backward (if training supported).

**Tests (Detailed)**
- Python tests: None dedicated in this file (covered by layer tests).
- Rust tests: `monolith-rs/crates/monolith-layers/tests/layer_ops_test.rs` (new, if implemented).
- Cross-language parity test:
  - Compare FFM outputs/gradients and fid_counter behavior between Python and Rust.

**Gaps / Notes**
- `feature_insight` and `fid_counter` depend on custom TF ops; Rust currently lacks equivalents.

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

### `monolith/native_training/layers/layer_ops_test.py`
<a id="monolith-native-training-layers-layer-ops-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 232
- Purpose/role: Validates custom ops (FFM, FeatureInsight, MonolithFidCounter) across CPU/GPU and checks gradients.
- Key symbols/classes/functions: `LayerOpsTest` methods `test_ffm_mul`, `test_ffm_mul_grad`, `test_ffm_dot`, `test_ffm_dot_grad`, `test_feature_insight`, `test_feature_insight_grad`, `test_fid_counter_grad`.
- External dependencies: TensorFlow GPU test utilities, custom ops via `layer_ops`.
- Side effects: Forces GPU contexts when available; uses global `tf.random.set_seed(0)`.

**Required Behavior (Detailed)**
- `test_ffm_mul`:
  - Uses `ffm(left, right, dim_size=4)` with `left` shape `(8,40)` and `right` `(8,48)` (10*4 and 12*4).
  - Checks GPU device placement if GPU available; compares CPU and GPU outputs.
  - Expects output shape `(8, 480)` for multiply mode.
- `test_ffm_mul_grad`:
  - Computes gradients of sum of FFM output wrt left/right.
  - Expects left_grad shape `(8,40)` and right_grad `(8,48)`; CPU and GPU grads equal.
- `test_ffm_dot`:
  - Uses `int_type='dot'`.
  - Expects output shape `(8,120)`; CPU and GPU outputs equal.
- `test_ffm_dot_grad`:
  - Same gradient checks as multiply, output dims unchanged.
- `test_feature_insight`:
  - Builds expected result by splitting input/weights per `segment_sizes=[3,2,4]`, matmul per segment, then optional aggregate using segment_sum of squared outputs.
  - Calls `layer_ops.feature_insight(..., aggregate=True)` and asserts close to expected.
- `test_feature_insight_grad`:
  - Compares gradients of `feature_insight` output vs explicit matmul concatenation.
  - Asserts outputs and gradients match.
- `test_fid_counter_grad`:
  - Verifies fid_counter increments and gradient = `-step` until threshold, then 0 at threshold.
  - Checks counter values for step=1, step=0.01, and threshold case at 1000.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-layers/tests/layer_ops_test.rs` (if ops reimplemented) or `monolith-rs/crates/monolith-tf/tests` for TF-runtime bindings.
- Rust public API surface: FFM op (multiply/dot), FeatureInsight, fid_counter equivalents.
- Data model mapping:
  - Output shapes match Python expectations (multiply: `B * (L*R*D)` flattened, dot: `B * (L*R)`).
  - Gradients should match analytic gradients of FFM/FeatureInsight.
- Feature gating: GPU tests optional; must be skipped if GPU backend unavailable.
- Integration points: `feature_cross` uses FFM.

**Implementation Steps (Detailed)**
1. Implement Rust tests mirroring CPU/GPU parity (skip GPU when not supported).
2. Add gradient checks for FFM and FeatureInsight if backward is implemented.
3. Add fid_counter unit test that verifies saturation and gradient behavior.
4. Ensure deterministic seeding for random tensors.

**Tests (Detailed)**
- Python tests: `monolith/native_training/layers/layer_ops_test.py`.
- Rust tests: `monolith-rs/crates/monolith-layers/tests/layer_ops_test.rs` (new).
- Cross-language parity test:
  - Compare outputs and gradients for FFM multiply/dot and FeatureInsight aggregate mode.

**Gaps / Notes**
- Python tests rely on custom TF ops and GPU placement; Rust needs an equivalent implementation or explicit skip/feature gate.

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

### `monolith/native_training/layers/lhuc.py`
<a id="monolith-native-training-layers-lhuc-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 296
- Purpose/role: LHUCTower: augments a dense tower with LHUC gating MLPs per layer; supports shared or per-layer LHUC configs, optional batch normalization, and weight norm.
- Key symbols/classes/functions: `LHUCTower`, `lhuc_params`, `build`, `call`, `get_config`, `from_config`.
- External dependencies: TensorFlow/Keras (`Layer`, `BatchNormalization`, `Sequential`, regularizers), Monolith layers (`MLP`, `Dense`), `extend_as_list`, `advanced_activations`.
- Side effects: Creates nested Dense/MLP layers and BatchNorm; collects trainable/non-trainable weights; supports LHUC-specific overrides via `lhuc_*` kwargs.

**Required Behavior (Detailed)**
- Initialization:
  - Splits kwargs into `_lhuc_kwargs` with `lhuc_` prefix; remaining kwargs passed to base Layer.
  - `output_dims` defines dense tower layers; `n_layers = len(output_dims)`.
  - `activations`:
    - None → `[relu]*(n_layers-1) + [None]`.
    - List/tuple length must match `n_layers`; maps via `ad_acts.get`.
    - Single activation string/function → same for all but last layer (None).
  - `initializers` expanded to list length `n_layers` via `extend_as_list`.
  - LHUC output dims:
    - If `lhuc_output_dims` is list of lists: each last dim must equal corresponding `output_dims[i]`.
    - If list of ints: applied to every layer and auto-append `[dim]`.
    - Else default: `[[dim] for dim in output_dims]`.
  - `lhuc_activations`: for each LHUC MLP, uses `relu` for all but last, last is `sigmoid2`.
- `build`:
  - Optional input BatchNorm if `enable_batch_normalization`.
  - For each layer:
    - Create `Sequential` block with Dense (custom monolith Dense) + optional BatchNorm + activation.
    - Dense uses weight norm options and regularizers.
    - Build LHUC MLP (`MLP`) with per-layer `lhuc_output_dims` and overrides via `lhuc_params`.
  - Extends trainable/non-trainable weights from sublayers.
- `call`:
  - If inputs is tuple/list: `(dense_input, lhuc_input)` else both are inputs.
  - Apply `extra_layers` (input BatchNorm) to dense_input.
  - For each layer and corresponding lhuc MLP:
    - `output_t = layer(dense_input) * lhuc_layer(lhuc_input)`.
    - Feed output to next layer.
- `get_config`:
  - Serializes activations via `ad_acts.serialize`, initializers via `tf.initializers.serialize`.
  - Includes batch norm settings and regularizers.
  - Adds `_lhuc_kwargs` into config.
- `from_config`:
  - Creates params via `params().copy()`, fills known keys, deserializes initializers/activations and regularizers.
  - Pops used keys from config and returns `p.instantiate()`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-layers/src/lhuc.rs`.
- Rust public API surface: `LHUCTower`, `LHUCConfig`, `LHUCOverrides`, `LHUCOutputDims`.
- Data model mapping:
  - Python activations list → Rust `ActivationType` list; last layer forced to `None`.
  - `sigmoid2` in LHUC MLP last layer → Rust `ActivationType::Sigmoid2`.
  - `lhuc_*` kwargs → `LHUCOverrides`.
- Feature gating: None.
- Integration points: `Dense`, `MLP`, `BatchNorm`.

**Implementation Steps (Detailed)**
1. Ensure Rust LHUCTower uses same activation and initializer expansion rules.
2. Mirror LHUC output dims logic (shared list vs per-layer list).
3. Add input BatchNorm and per-layer BatchNorm gating with same defaults.
4. Implement LHUCOverrides mapping to override LHUC MLP settings.
5. Ensure config serialization/deserialization matches Python (including `lhuc_*` fields).

**Tests (Detailed)**
- Python tests: `monolith/native_training/layers/lhuc_test.py`.
- Rust tests: `monolith-rs/crates/monolith-layers/tests/lhuc_test.rs` (new).
- Cross-language parity test:
  - Fix weights and inputs; compare outputs for single-input and `(dense_input, lhuc_input)` modes.

**Gaps / Notes**
- Python `from_config` ignores `kernel_regularizer` and `bias_regularizer` assignments (calls deserialize but does not set); decide whether to mirror or fix in Rust.

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

### `monolith/native_training/layers/lhuc_test.py`
<a id="monolith-native-training-layers-lhuc-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 73
- Purpose/role: Smoke tests for LHUCTower instantiation, config serialization, and forward call with separate dense/LHUC inputs.
- Key symbols/classes/functions: `LHUCTowerTest` methods `test_lhuc_instantiate`, `test_lhuc_serde`, `test_lhuc_call`.
- External dependencies: TensorFlow v1 session mode, NumPy.
- Side effects: Runs session after variable initialization.

**Required Behavior (Detailed)**
- `test_lhuc_instantiate`:
  - Params-based instantiate with `output_dims=[1,3,4,5]`, `activations=None`, `initializers=GlorotNormal`.
  - Direct constructor with same output dims and `initializers=HeUniform`.
- `test_lhuc_serde`:
  - Instantiates via params, `cfg = get_config()`, `LHUCTower.from_config(cfg)` should succeed.
- `test_lhuc_call`:
  - Builds LHUCTower with:
    - `output_dims=[50,20,1]`,
    - `lhuc_output_dims=[[50,50],[50,50,20],[100,1]]`,
    - `lhuc_use_bias=False`, `use_bias=True`, `activations=None`.
  - Inputs: `dense_data` `(100,100)` and `lhuc_data` `(100,50)`.
  - Calls layer with `[dense_data, lhuc_data]`, sums output, runs session.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-layers/tests/lhuc_test.rs`.
- Rust public API surface: `LHUCTower`, `LHUCConfig`, `LHUCOverrides`.
- Data model mapping:
  - `lhuc_*` kwargs ↔ `LHUCOverrides`.
  - Params-based instantiation ↔ Rust config builder.
- Feature gating: None.
- Integration points: `monolith_layers::lhuc`.

**Implementation Steps (Detailed)**
1. Add Rust tests for constructor/config round-trip.
2. Add forward test with separate dense/LHUC inputs and per-layer LHUC output dims.
3. Add assertions on output shape and deterministic sum.

**Tests (Detailed)**
- Python tests: `monolith/native_training/layers/lhuc_test.py`.
- Rust tests: `monolith-rs/crates/monolith-layers/tests/lhuc_test.rs` (new).
- Cross-language parity test:
  - Fix weights and inputs; compare output sums.

**Gaps / Notes**
- Python uses `lhuc_use_bias` override (via `lhuc_*` kwargs); ensure Rust override wiring matches.

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

### `monolith/native_training/layers/logit_correction.py`
<a id="monolith-native-training-layers-logit-correction-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 88
- Purpose/role: Logit correction layer to compensate for sampling bias during training or inference.
- Key symbols/classes/functions: `LogitCorrection`, `safe_log_sigmoid`, `get_sample_logits`.
- External dependencies: TensorFlow/Keras (`Layer`, `InputSpec`, activations), `with_params`.
- Side effects: None beyond computation.

**Required Behavior (Detailed)**
- Inputs: `(logits, sample_rate)` where both are max 2D tensors.
- `call`:
  - `corrected = get_sample_logits(logits, sample_rate, sample_bias)`.
  - If `activation` is set, apply it.
- `safe_log_sigmoid(logits)`:
  - Stable computation of `log(sigmoid(logits))` using `log1p(exp(neg_abs))` trick.
- `get_sample_logits`:
  - `sample_rate is None` and `sample_bias=True`: return `safe_log_sigmoid(logits)`.
  - `sample_rate not None` and `sample_bias=False`: return `logits - log(sample_rate)`.
  - `sample_rate not None` and `sample_bias=True`: return `safe_log_sigmoid(logits) - log(sample_rate)`.
  - Else: return `logits`.
- `compute_output_shape` returns a 1D shape `([None])` (questionable but part of API).
- `get_config` serializes `activation` and `sample_bias`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-layers/src/logit_correction.rs`.
- Rust public API surface: `LogitCorrection` with `forward_with_sample_rate`.
- Data model mapping:
  - Python activation → Rust `ActivationType`/`ActivationLayer`.
  - `sample_rate` optional tensor → `Option<&Tensor>`.
- Feature gating: None.
- Integration points: Used in training heads that correct logits for sampling.

**Implementation Steps (Detailed)**
1. Ensure `safe_log_sigmoid` matches TF numeric behavior.
2. Confirm `get_sample_logits` branch logic matches Python.
3. Add optional activation layer application.
4. Add config serialization to match `activation` and `sample_bias`.

**Tests (Detailed)**
- Python tests: `monolith/native_training/layers/logit_correction_test.py`.
- Rust tests: `monolith-rs/crates/monolith-layers/tests/logit_correction_test.rs` (new).
- Cross-language parity test:
  - Compare corrected logits for combinations of sample_rate present/absent and sample_bias true/false.

**Gaps / Notes**
- Python `compute_output_shape` always returns `[None]` regardless of input; Rust may not expose shape inference.

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

### `monolith/native_training/layers/logit_correction_test.py`
<a id="monolith-native-training-layers-logit-correction-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 65
- Purpose/role: Smoke tests for LogitCorrection instantiation, serialization, and forward call.
- Key symbols/classes/functions: `SailSpecialTest` methods `test_sr_instantiate`, `test_sr_serde`, `test_sr_call`.
- External dependencies: TensorFlow v1 session mode, NumPy.
- Side effects: Runs session after variable init.

**Required Behavior (Detailed)**
- `test_sr_instantiate`:
  - Params-based instantiation with `activation=relu`.
  - Direct constructor with `activation=relu`.
- `test_sr_serde`:
  - Instantiate with `activation=sigmoid`, then `get_config` and `from_config`.
- `test_sr_call`:
  - Instantiate via params with `activation=tanh`.
  - Inputs: logits `x` shape `(100,10)` and sample_rate `sr` shape `(100,1)` sampled in `(1e-10, 1)`.
  - Runs `layer((x, sr))` and sums outputs in session.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-layers/tests/logit_correction_test.rs`.
- Rust public API surface: `LogitCorrection` with optional activation.
- Data model mapping:
  - Params-based instantiation ↔ Rust config/builder.
  - `get_config`/`from_config` ↔ serde round-trip.
- Feature gating: None.
- Integration points: `monolith_layers::logit_correction`.

**Implementation Steps (Detailed)**
1. Add Rust tests for constructor/config round-trip.
2. Add forward test with logits + sample_rate; assert output shape and deterministic sum.

**Tests (Detailed)**
- Python tests: `monolith/native_training/layers/logit_correction_test.py`.
- Rust tests: `monolith-rs/crates/monolith-layers/tests/logit_correction_test.rs` (new).
- Cross-language parity test:
  - Fix logits/sample_rate and compare outputs for activation on/off.

**Gaps / Notes**
- Python test uses `activation=tanh`; ensure Rust supports this activation.

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

### `monolith/native_training/layers/mlp.py`
<a id="monolith-native-training-layers-mlp-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 211
- Purpose/role: Core MLP layer built from custom Dense layers with optional batch normalization, weight norm, and feature insight logging.
- Key symbols/classes/functions: `MLP`, `build`, `call`, `get_config`, `from_config`, `get_layer`.
- External dependencies: TensorFlow/Keras (`Layer`, `BatchNormalization`, regularizers), Monolith `Dense`, `extend_as_list`, `advanced_activations`, `feature_insight_data`.
- Side effects: Adds BN/Dense losses, uses `feature_insight_data` hook on first Dense when segment info provided.

**Required Behavior (Detailed)**
- Initialization:
  - `output_dims` defines layers; `use_weight_norm`, `use_learnable_weight_norm`, `use_bias`, regularizers and BN params stored.
  - `initializers` expanded to list of length `_n_layers` via `extend_as_list` and `tf.initializers.get`.
  - `activations`:
    - None → `[relu]*(n_layers-1) + [None]`.
    - List/tuple length must match `n_layers`; each mapped via `ad_acts.get`.
    - Single activation → applied to all but last (None).
- `build`:
  - If BN enabled, prepend input BN layer.
  - For each layer:
    - Create custom `Dense` with activation=None, bias/init/regularizer and kernel norm settings.
    - Optional BatchNorm between layers (not on final layer).
    - Append activation layer if not None.
  - Tracks trainable/non-trainable weights and adds sub-layer losses.
- `call`:
  - Sequentially applies `_stacked_layers`.
  - When a layer name ends with `dense_0` and kwargs provided, calls `feature_insight_data` with segment metadata and layer kernel.
- `get_config` serializes activations and initializers, regularizers, BN settings, weight norm flags.
- `from_config`:
  - Deserializes `initializers` and `activations`; others assigned directly.
  - Ignores leftover keys, then `instantiate()`.
- `get_layer(index)` returns `_stacked_layers[index]`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-layers/src/mlp.rs`.
- Rust public API surface: `MLP`, `MLPConfig`, `ActivationType`, `ActivationLayer`.
- Data model mapping:
  - Python activations list → Rust `ActivationType` list; last layer activation None.
  - `feature_insight_data` hook → optional instrumentation in Rust.
- Feature gating: None.
- Integration points: `Dense`, `BatchNorm`, activation registry.

**Implementation Steps (Detailed)**
1. Ensure MLPConfig defaults align with Python (weight norm on, BN off).
2. Add support for per-layer initializers and activations list expansion (1 or N).
3. Add optional input BatchNorm and per-layer BN (skip last).
4. Add optional feature insight hook (or document omission).
5. Add config serialization compatible with Python `get_config`.

**Tests (Detailed)**
- Python tests: `monolith/native_training/layers/mlp_test.py`.
- Rust tests: `monolith-rs/crates/monolith-layers/tests/mlp_test.rs` (new).
- Cross-language parity test:
  - Fix weights and inputs; compare output sums with/without BN and weight norm.

**Gaps / Notes**
- Python `feature_insight_data` side effect not yet mirrored in Rust.

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

### `monolith/native_training/layers/mlp_test.py`
<a id="monolith-native-training-layers-mlp-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 78
- Purpose/role: Smoke tests for MLP instantiation, serialization, and forward call with batch normalization and mixed activation list.
- Key symbols/classes/functions: `MLPTest` methods `test_mlp_instantiate`, `test_mlp_serde`, `test_mlp_call`.
- External dependencies: TensorFlow v1 session mode, NumPy.
- Side effects: Checks internal `_stacked_layers` length.

**Required Behavior (Detailed)**
- `test_mlp_instantiate`:
  - Params-based instantiate with `output_dims=[1,3,4,5]`, `activations=None`, `initializers=GlorotNormal`.
  - Direct constructor with same output dims and `initializers=HeUniform`.
- `test_mlp_serde`:
  - Instantiate via params, `get_config` and `MLP.from_config(cfg)` should succeed.
- `test_mlp_call`:
  - Params-based instantiate with:
    - `output_dims=[100,50,10,1]`,
    - `enable_batch_normalization=True`,
    - `activations=['relu', tanh, PReLU, None]`,
    - `initializers=GlorotNormal`.
  - Input shape `(100,100)`; sums output.
  - Asserts `len(layer._stacked_layers) == 11`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-layers/tests/mlp_test.rs`.
- Rust public API surface: `MLP`, `MLPConfig`, `ActivationType`.
- Data model mapping:
  - Activation list includes mixed string/function/class; Rust should accept equivalent `ActivationType`.
  - `_stacked_layers` count corresponds to Dense + optional BN + activation for each layer; ensure layering logic matches.
- Feature gating: None.
- Integration points: `monolith_layers::mlp`.

**Implementation Steps (Detailed)**
1. Add Rust tests for constructor and config serialization.
2. Add forward test with batch normalization and mixed activations; assert output shape/sum.
3. Add check on internal layer count if exposed (or infer via config).

**Tests (Detailed)**
- Python tests: `monolith/native_training/layers/mlp_test.py`.
- Rust tests: `monolith-rs/crates/monolith-layers/tests/mlp_test.rs` (new).
- Cross-language parity test:
  - Fix weights and inputs; compare output sums and layer counts.

**Gaps / Notes**
- Python uses `tf.keras.layers.PReLU` class in activations list; Rust must map to `ActivationType::PReLU` with default params.

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
