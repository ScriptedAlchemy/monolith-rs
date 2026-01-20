<!--
Source: task/request.md
Lines: 9450-9626 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/distribution_utils.py`
<a id="monolith-native-training-distribution-utils-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 443
- Purpose/role: BytePS/Horovod initialization, MPI helpers, sync training config updates, GPU session config tweaks, and BytePS micro-benchmarks.
- Key symbols/classes/functions: `bps_init`, `byteps_benchmark_ar`, `byteps_benchmark_a2a`, `bps_comm_benchmark`, `init_sync_train_and_update_conf`, `get_mpi_rank`, `get_mpi_local_rank`, `get_mpi_size`, `get_mpi_local_size`, `enable_sync_training`, `try_init_cuda`, `get_device_str`, `get_sync_run_hooks`, `update_session_config_for_gpu`.
- External dependencies: `absl.flags`, `absl.logging`, `tensorflow`, `byteps.tensorflow` (optional), `horovod.tensorflow` (optional), `monolith.native_training.metric.metric_hook.ByteCCLTelemetryHook`.
- Side effects: Sets many env vars, creates `/tmp/bps_<uuid>_socket_<id>` dir, runs `ip addr show` via shell, enables eager execution in benchmark funcs, initializes BytePS/Horovod, mutates config object fields.

**Required Behavior (Detailed)**
- Global state:
  - `_SYNC_TRAIN_INITED` gate to avoid repeated init.
  - `enable_bps = int(os.getenv("MONOLITH_WITH_BYTEPS", "0"))` evaluated at import time.
- `bps_init(uuid)`:
  - Ensures `BYTEPS_ALLTOALL_SESSION_SIZE` default `3`.
  - Mirrors `OMPI_COMM_WORLD_*` into `BYTEPS_LOCAL_SIZE`, uses `BYTEPS_LOCAL_SIZE` to compute `local_rank` and `phy_node_id`.
  - Computes `socket_path = /tmp/bps_<uuid>_socket_<phy_node_id>` and creates it.
  - Chooses network interface:
    - If `BYTEPS_GPU_NIC_BINDING_MODE=0`, uses `DMLC_INTERFACE` (default `eth0`).
    - Else, binds NIC by GPU index (`NUM_GPU_PER_NIC=2`), sets `CUDA_VISIBLE_DEVICES` and UCX/GDR envs when `MONOLITH_WITH_BYTEPS_FWD_GDR` or `MONOLITH_WITH_BYTEPS_BWD_GDR` is enabled.
    - If `BYTEPS_WITH_ALL_NICS=1`, sets `UCX_NET_DEVICES` to a list of mlx5 + eth; else only `mlx5_<nic_id>:1`.
  - Runs `ip addr show <interface>` to compute host IP; exports as `UCX_RDMA_CM_SOURCE_ADDRESS` and `DMLC_NODE_HOST`.
  - Sets required BytePS/PSLite env vars (role, worker/server counts, UUID, ranks, telemetry, log levels, perf knobs, partition sizes).
  - Ensures `BYTEPS_P2P_PARTITION_BYTES` and `BYTEPS_PARTITION_BYTES` defaults computed from `size`.
  - Imports `byteps.tensorflow` and calls `bps.init(lazy=False)`.
- `byteps_benchmark_ar(total_len, total_niter=10000, use_cpu=False, op='pushpull')`:
  - Enables eager execution; uses `bps.push_pull` by default.
  - Creates tensor of shape `[total_len, 1]` on CPU/GPU.
  - Runs `total_niter` iterations, logs latency/Goodput every 20 iterations, returns `goodputs[1:]`.
- `byteps_benchmark_a2a(total_len, total_niter=10000, dst_gpu=True, src_gpu=True)`:
  - Enables eager execution; if CPU-only (`dst_gpu=False` and `src_gpu=False`) reduces `total_len` by 8.
  - Builds splits and recv_splits; selects correct BytePS alltoall variant (`alltoall`, `alltoall_cpu2gpu`, `alltoall_gpu2cpu`).
  - Runs loop and returns `goodputs[1:]`.
- `bps_comm_benchmark()`:
  - Reads `MONOLITH_BENCHMARK_BPS`, `MONOLITH_BENCHMARK_ITERS`, and length env vars; sets TF memory growth for all GPUs.
  - Runs selected benchmarks and prints summary tuples `(total_len, avg_goodput)`.
- `init_sync_train_and_update_conf(dct_config)`:
  - Logs entry; imports BytePS or Horovod as needed; initializes once.
  - If not `merge_sync_training_ckpt`, updates `dct_config.model_dir` with `index-<rank>` suffix under `model_dir/uuid/`.
  - Sets `num_ps=0`, `reorder_fids_in_data_pipeline=True`, `index=hvd.rank()`, `num_workers=hvd.size()`, `enable_variable_partition=False`.
  - Catches ImportError/NotFoundError and logs warning.
- MPI helpers:
  - `get_mpi_rank/local_rank/size/local_size` pull from `OMPI_COMM_WORLD_*` envs; warn and use defaults (0/1) when missing.
- `enable_sync_training()`:
  - Returns `FLAGS.enable_sync_training and 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ`; returns False on exception.
- `try_init_cuda()`:
  - If `CUDA_VISIBLE_DEVICES` not set but MPI local rank present, set `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES=<local_rank>`.
  - If sync training enabled and not initialized, tries to import BytePS or Horovod (based on `MONOLITH_WITH_BYTEPS`/`MONOLITH_WITH_HOROVOD`) and `hvd.init()`; logs exceptions.
- `get_device_str(force_on_cpu=False)`:
  - Uses `FLAGS.enable_gpu_training` or `device_utils._GPU_PLACEMENT_ALLOWED` to choose GPU vs CPU.
  - For MPI + sync training:
    - In PS mode (`FLAGS.num_ps > 0`): returns `/job:chief` for rank 0 else `/job:worker` with `task` offsets and `/device:{GPU|CPU}:0`.
    - Without PS mode: returns empty string.
  - Otherwise returns `/device:{GPU|CPU}:0`.
- `get_sync_run_hooks(is_full_sync=False)`:
  - Returns empty list when not in sync mode.
  - Uses BytePS `BroadcastGlobalVariablesHook` when `MONOLITH_WITH_BYTEPS` and `MONOLITH_WITH_BYTEPS_BCAST` are set.
  - If `MONOLITH_WITH_BYTEPS_BCAST == -1`, returns empty list.
  - Adds `ByteCCLTelemetryHook(50)` when `is_full_sync` and using BytePS broadcast.
  - Falls back to Horovod `BroadcastGlobalVariablesHook` when not using BytePS.
- `update_session_config_for_gpu(session_config)`:
  - When sync training is enabled, sets `gpu_options.visible_device_list` to local rank.
  - If `MONOLITH_FORCE_GPU_COMPATIBLE=1`, sets `force_gpu_compatible=True`.
  - If BytePS GDR alltoall enabled (`MONOLITH_WITH_BYTEPS_FWD_GDR` or `MONOLITH_WITH_BYTEPS_BWD_GDR`), disables `allow_growth`, sets `per_process_gpu_memory_fraction=0.4` and visible device list to local rank.
  - Otherwise enables `allow_growth`.
  - When not in sync training, still sets `allow_growth=True`.

**Rust Mapping (Detailed)**
- Target crate/module: new `monolith-rs/crates/monolith-training/src/distribution_utils.rs` (sync training env + device helpers) and `monolith-rs/crates/monolith-training/src/distributed.rs` for MPI helpers.
- Rust public API surface: `init_sync_train_and_update_conf`, `get_mpi_*`, `enable_sync_training`, `try_init_cuda`, `get_device_str`, `get_sync_run_hooks`, `update_session_config_for_gpu` equivalents; optional TF-specific BytePS/Horovod bridge behind feature flags.
- Data model mapping: map Python `dct_config` mutation to Rust config struct (likely in `monolith-training` or `monolith-cli`).
- Feature gating: `tf-runtime` (BytePS/Horovod) and `cuda` (GPU-specific paths); default Candle backend should no-op or provide safe fallbacks.
- Integration points: `monolith/native_training/estimator.py`, `cpu_training.py`, `device_utils.py`, `model_export/saved_model_exporters.py`, and `data/datasets.py` equivalents in Rust.

**Implementation Steps (Detailed)**
1. Define a Rust config struct that mirrors `dct_config` fields used here (`uuid`, `model_dir`, `merge_sync_training_ckpt`, `num_ps`, `reorder_fids_in_data_pipeline`, `index`, `num_workers`, `enable_variable_partition`).
2. Implement env parsing for MPI (`OMPI_COMM_WORLD_*`) and BytePS/Horovod gating (`MONOLITH_WITH_BYTEPS`, `MONOLITH_WITH_HOROVOD`).
3. Add `get_device_str` logic to Rust; plumb in `enable_gpu_training` and `num_ps` flags (from CLI or config).
4. For TF runtime, implement BytePS/Horovod initialization and broadcast hooks; otherwise return empty hooks and log warnings.
5. Implement `try_init_cuda` that sets env vars before GPU runtime init (Rust side); keep `_SYNC_TRAIN_INITED` equivalent.
6. Implement `update_session_config_for_gpu` only when using TF sessions; in Candle backend, document as no-op.
7. Port benchmark helpers only if TF BytePS runtime is supported; otherwise document as unsupported.

**Tests (Detailed)**
- Python tests: none in-tree specific to this file.
- Rust tests: add unit tests for env parsing and `get_device_str` permutations; integration tests for config updates and no-op behavior when BytePS/Horovod missing.
- Cross-language parity test: compare outputs of MPI helpers and device string formatting for a matrix of env/flag combinations.

**Gaps / Notes**
- Uses shell command `ip addr show` to resolve interface IP; Rust port should use OS APIs or run the command for parity.
- Heavy BytePS/Horovod coupling means full parity likely only under TF runtime with custom ops.

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

### `monolith/native_training/embedding_combiners.py`
<a id="monolith-native-training-embedding-combiners-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 102
- Purpose/role: Defines embedding combiner strategies for ragged inputs (sum/mean pooling and FirstN sequence padding).
- Key symbols/classes/functions: `Combiner`, `ReduceSum`, `ReduceMean`, `FirstN`.
- External dependencies: `tensorflow`, `distribution_ops`, `ragged_utils`, `device_utils`.
- Side effects: None beyond device placement in `FirstN.combine`.

**Required Behavior (Detailed)**
- `Combiner`:
  - Stores `max_seq_length` and exposes `combine(...)` abstract method.
- `ReduceSum.combine(key, embedding, name=None)`:
  - Uses `ragged_utils.fused_value_rowids(key)` to map values to row ids.
  - Calls `distribution_ops.reduce_sum(expand_dims(rowids), embedding, expand_dims(key.nrows(), 0), name=name)`.
- `ReduceMean.combine(key, embedding, name=None)`:
  - Same as `ReduceSum` but calls `distribution_ops.reduce_mean`.
- `FirstN.__init__(seq_length)`:
  - Asserts `seq_length > 0`, sets `max_seq_length` to `seq_length`.
- `FirstN.combine(key, embedding, name=None)`:
  - If `embedding` is not a `tf.Tensor`, converts it.
  - Computes `batch_size_tensor = key.nrows()`.
  - Converts `key` to sparse (`key_sparse = key.to_sparse()`), uses `key_sparse.indices` to scatter.
  - Builds `shape = [batch_size, max(max_seq_length, key_sparse.dense_shape[1]), embedding_dim]` with `embedding.shape.as_list()[1]`.
  - Under `device_utils.maybe_device_if_allowed('/device:GPU:0')`, calls `tf.scatter_nd(indices, embedding, shape)`.
  - Returns `tf.slice(scattered, [0,0,0], [-1, max_seq_length, -1])` to enforce sequence length.
  - Rows with fewer embeddings are zero-padded by scatter.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-layers/src/embedding.rs` and/or new `monolith-rs/crates/monolith-layers/src/combiner.rs`.
- Rust public API surface:
  - `Combiner` trait with `combine(key, embedding) -> Tensor`.
  - `ReduceSum` and `ReduceMean` pooling for ragged sequences.
  - `FirstN` equivalent using `SequenceEmbeddingLookup` or a new combiner wrapper.
- Data model mapping: Ragged input represented as `(values, row_lengths)` or `(values, row_splits)`; must map to pooled or padded tensors.
- Feature gating: TF runtime path can call distribution_ops; Candle backend should implement native pooling and padding.
- Integration points: use in `feature.py`/`feature_utils.py` equivalents and embedding table lookup paths.

**Implementation Steps (Detailed)**
1. Define a Rust `Combiner` trait and enums for `ReduceSum`, `ReduceMean`, `FirstN`.
2. Implement pooling for ragged sequences using row lengths (sum/mean) with deterministic order.
3. Implement `FirstN` by zero-padding to `[batch, max_seq_length, dim]` and truncating when longer.
4. Preserve shape inference behavior: unknown batch size => dynamic dimension, but known `max_seq_length` and embedding dim.
5. If TF runtime is enabled, optionally route to distribution_ops to match TF kernels exactly.
6. Add device placement logic for GPU (if supported) and document when CPU is forced.

**Tests (Detailed)**
- Python tests: `monolith/native_training/embedding_combiners_test.py`.
- Rust tests: `monolith-rs/crates/monolith-layers/tests/embedding_combiners_test.rs` (new) or extend `monolith-layers/src/embedding.rs` tests.
- Cross-language parity test: compare pooled and padded outputs for the same ragged inputs and ensure shape inference matches.

**Gaps / Notes**
- Python uses `ragged_utils.fused_value_rowids` and custom reduce ops; Rust must replicate row-id logic exactly.
- `FirstN` uses `scatter_nd` behavior; ensure zero-fill for missing entries and correct truncation.

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
