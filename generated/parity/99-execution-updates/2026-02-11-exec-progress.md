# Python → Rust Parity Execution Update (2026-02-11)

## Completed in this execution slice

### 1) `monolith-cli export` moved from stub to functional path
- Implemented checkpoint resolution from either:
  - explicit checkpoint file, or
  - checkpoint directory (auto-picks latest `checkpoint-*.json`).
- Added real model restore + export flow via `monolith-checkpoint::ModelExporter`.
- Added optional quantization pass (8-bit / 16-bit simulation) and validation.
- Added optional warmup asset copy flow into export output.
- Added explicit `--model-version` support.
- Added explicit error behavior for unsupported `onnx` / `torchscript` formats.

### 2) `monolith-cli serve` lifecycle parity improved
- Replaced stub behavior with real startup/shutdown of `monolith_serving::Server`.
- Command now:
  - builds serving config from CLI flags,
  - starts the primary serving server,
  - blocks until ctrl-c,
  - performs graceful stop.

### 3) CLI test parity hardening
- Fixed `tf_runner` tests to use valid dotted task names matching model registry resolution semantics.
- Added/updated export tests covering:
  - file checkpoint export path,
  - latest checkpoint selection from directory,
  - unsupported format errors,
  - invalid quantization bit-width errors.

### 4) Linux workspace build/test lane hardening
- Updated `monolith-examples` default features to avoid Apple-only Metal dependency on Linux.
- Fixed rank-2 tensor concatenation bug in `monolith-layers::Tensor::cat`:
  - dim=0 now validates columns and sums rows,
  - dim=1 now validates rows and sums columns.
- Added regression tests for rank-2 concat on both dimensions.
- Removed strict DIEN hidden-size panic in stock example model constructor to keep forward path shape-safe.

### 5) Native-training parity improvements (entrypoint/runtime helpers)
- `hooks::CheckpointHook` now performs real persistence by default:
  - writes checkpoint payload files to `model_dir/checkpoint-<step>.json`,
  - supports retention with `max_to_keep`,
  - prunes oldest checkpoints once retention threshold is exceeded.
- `estimator::ModelFn` now supports input-aware prediction:
  - added default `predict(&[f32]) -> Vec<f32>` API,
  - `Estimator::predict_with_inputs` executes prediction over explicit feature vectors,
  - legacy `predict(num_examples)` remains compatible by delegating to empty feature vectors.
- `native_training::hvd_lib` upgraded from fixed stubs to env-driven parity helpers:
  - robust BytePS/Horovod backend detection from `MONOLITH_WITH_BYTEPS`,
  - rank/size/local_rank/local_size read from Horovod/OMPI/BytePS env conventions,
  - defensive fallbacks for missing/invalid values.
- `distributed` module now includes a `LocalCluster` runtime simulator:
  - starts/stops local PS + worker roles,
  - routes parameter registration by hash to PS shards,
  - applies worker gradient steps against routed parameters.
- Added `run_config` parity module:
  - introduces `RunConfig` and `RunnerConfig`,
  - implements Python-style merge semantics preserving CLI/base overrides,
  - maps discovery `Consul -> Zk` for runner compatibility,
  - enforces CPU defaults (`embedding_prefetch_capacity >= 1`, post-push enabled),
  - exposes explicit user override extraction.
- `runner_utils` parity expanded:
  - added discovery factory selection (`local/primus/consul/mlp`),
  - added context-style discovery guard with automatic `close()`,
  - added tests mirroring Python runner_utils discovery selection behavior.
- Added chief/non-chief restore synchronization helper:
  - chief performs checkpoint copy from `restore_dir`,
  - non-chief polls for synced checkpoint/monolith checkpoint artifacts with timeout handling.
- Added integration parity tests under `monolith-training/tests/runner_utils_parity.rs`
  for run-config merge + discovery selection + restore synchronization flow.
- Data pipeline parity improvement in `monolith-data`:
  - `TFRecordDataset::from_pattern` now supports:
    - comma-separated glob/path specs,
    - `@filelist` input with one path/glob per line,
    - deterministic de-duplication + ordering.
  - added regression tests for glob/csv/filelist/not-found behavior.

### 6) Additional parity hardening in file/prefetch/distributed flows
- Added integration parity tests for `monolith-training::file_ops`:
  - verifies `WritableFile::append_entry_dump` writes TFRecord-framed `EntryDump` protos
    that round-trip decode correctly,
  - validates input-shape errors for `append_entry_dump`,
  - validates post-close append behavior (`BrokenPipe`).
- Added integration parity tests for `monolith-training::prefetch_queue`:
  - zero-capacity passthrough behavior,
  - nested enqueue/dequeue preserving non-tensor leaves (`String`/`Null`),
  - async function manager end-of-run queue drain behavior.
- Upgraded `monolith-training::distributed::LocalCluster` synchronization semantics:
  - `Worker::sync_barrier` now enforces running-state precondition,
  - new non-blocking local barrier coordination in `LocalCluster::sync_barrier`,
  - adds explicit `BarrierStatus::{Waiting, Released}` outcomes per barrier epoch,
  - adds barrier release regression coverage across epochs.

### 7) Runner/discovery parity follow-up (ZK + path helper)
- `runner_utils::get_discovery` now supports `ServiceDiscoveryType::Zk` selection directly
  (returns a `RunnerDiscovery::Zk { deep_insight_name, zk_server }` backend descriptor)
  instead of failing with unsupported-backend errors.
- Added `RunnerDiscovery::zk_config()` helper for parity assertions and downstream wiring.
- Added/extended ZK selection tests in:
  - `monolith-training/src/runner_utils.rs` (`test_get_discovery_zk`)
  - `monolith-training/tests/runner_utils_parity.rs` (`test_run_config_to_discovery_selection_zk`)
- Added `runner_utils::isabs(...)` parity helper:
  - treats `hdfs:/...` paths as absolute (matching Python monkey-patched behavior),
  - retains native absolute-path checks for local filesystems.

### 8) Runner checkpoint-state override/retry parity helper
- Added `runner_utils::get_checkpoint_state_with_restore_override(...)` and `RunnerMode`:
  - retries checkpoint-state reads when the checkpoint file exists but parse fails,
  - returns `read ckpt error!` (`RunnerUtilsError::ReadCheckpointFailed`) after retry budget exhaustion,
  - applies `restore_ckpt` override behavior for `latest_filename == "checkpoint"`:
    - train mode only applies restore once when marker file is absent,
    - non-train modes always apply restore override,
    - writes updated `checkpoint` metadata and `restore_ckpt` marker when override is applied.
- Added unit and integration parity tests covering:
  - train-mode override,
  - marker-protected train-mode behavior,
  - non-train override behavior,
  - parse/retry failure path.

### 9) `monolith_discovery` context parity for Consul psm generation
- Updated `runner_utils::monolith_discovery(...)` to auto-generate a Consul `psm`
  identifier when caller does not explicitly provide one:
  - uses `env_utils::generate_psm_from_uuid(...)` over `runner_conf.deep_insight_name`,
  - mirrors Python context-manager flow where non-local discovery creates psm internally.
- Added unit + integration coverage ensuring Consul discovery context now succeeds
  without explicit `psm` argument.

### 10) RunnerConfig post-init style restore semantics
- Expanded `run_config::{RunConfig, RunnerConfig}` with restore/runtime parity fields:
  - `index`,
  - `restore_dir`,
  - `restore_ckpt`.
- Added `runner_utils::initialize_restore_checkpoint_from_runner(...)`:
  - no-op when `restore_dir` is absent,
  - chief path (`is_local || index == 0`) performs restore copy immediately,
  - non-chief path waits for chief sync artifacts through existing prepare helper.
- Added unit + integration tests for:
  - no-restore-dir passthrough,
  - chief/worker synchronization through config-driven initialization,
  - merge behavior carrying restore fields from `RunConfig` into `RunnerConfig`.

### 11) RunnerConfig env-export parity for estimator initialization
- Expanded `RunConfig`/`RunnerConfig` with optional env-export fields:
  - `tf_grpc_worker_cache_threads`,
  - `monolith_grpc_worker_service_handler_multiplier`.
- Added `RunConfig::apply_runtime_env_exports(&RunnerConfig)` helper to mirror
  Python estimator-side exports:
  - `TF_GRPC_WORKER_CACHE_THREADS`,
  - `MONOLITH_GRPC_WORKER_SERVICE_HANDLER_MULTIPLIER`.
- Added regression tests covering:
  - merge + user override visibility for these new fields,
  - concrete env var export behavior.

## Validation evidence (commands run)

1. `cargo test -p monolith-cli -q` ✅  
2. `cargo test -p monolith-layers -q` ✅  
3. `cargo test -p monolith-examples --bin stock_prediction test_model_forward -q` ✅  
4. `cargo test --workspace -q` ✅
5. `cargo test -p monolith-training -q` ✅ (re-run after each native-training change)
6. `cargo test --workspace -q` ✅ (post-native-training parity update verification)
7. `cargo test -p monolith-data -q` ✅ (post-pattern expansion regression run)
8. `cargo test -p monolith-training -q` ✅ (post file_ops + prefetch + distributed barrier updates)
9. `cargo test -p monolith-training -q` ✅ (post ZK discovery selection + `isabs` helper updates)
10. `cargo test -p monolith-training -q` ✅ (post checkpoint-state override/retry parity helper)
11. `cargo test -p monolith-training -q` ✅ (post monolith_discovery Consul auto-psm behavior)
12. `cargo test -p monolith-training -q` ✅ (post RunnerConfig restore-init parity helper updates)
13. `cargo test -p monolith-training -q` ✅ (post RunnerConfig env-export parity updates)

## Notes
- This update specifically closes major TODO/stub surfaces in CLI runtime flows and restores a reliable Linux workspace test command.
- Remaining parity workstreams continue in core/native-training/domain-specific modules listed in task definitions.
