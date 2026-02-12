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
- Wired env export application into config-driven runtime initialization
  (`initialize_restore_checkpoint_from_runner`) so runtime entrypoints can
  apply export semantics even when restore sync is a no-op.

### 12) Estimator train/eval API parity for step controls
- Enhanced `monolith-training::estimator::Estimator` with Python-like step control semantics:
  - added `train_with_limits(steps, max_steps)`:
    - `steps` interpreted as relative additional steps,
    - `max_steps` interpreted as absolute global-step cap,
    - when both are provided uses `min(global_step + steps, max_steps)`.
  - added `evaluate_with_steps(steps)` to override configured eval step count.
  - `train()` / `evaluate()` now delegate to these explicit helpers.
  - `train_and_evaluate()` now performs per-round relative training slices without mutating config.
- Added regression tests covering:
  - relative training slices,
  - absolute max-step caps,
  - combined `steps + max_steps` behavior,
  - explicit eval-step overrides.

### 13) Distributed runner integration with `RunnerConfig`
- Added `runner::distributed_config_from_runner(...)` to map `RunnerConfig` into
  `DistributedRunConfig` (role/index/cluster sizing + bind address).
- Added `runner::run_distributed_from_runner_config(...)`:
  - applies runner post-init restore/env semantics via `initialize_restore_checkpoint_from_runner`,
  - dispatches into existing role-based distributed runner.
- Exported new runner APIs at crate root for downstream runtime wiring.
- Added smoke coverage in both unit and integration tests for RunnerConfig-driven
  PS/worker distributed startup.

### 14) EstimatorSpec parity surface
- Added `EstimatorSpec` + `EstimatorSpecUpdate` to mirror Python estimator output API shape:
  - fields include `label`, `pred`, `head_name`, `loss`, `optimizer`, `classification`,
  - constructor defaults align with Python (`head_name/loss/optimizer=None`, `classification=true`).
- Added `EstimatorSpec::replace(...)` parity helper with mode immutability enforcement:
  - allows replacement only when requested mode equals existing mode,
  - returns explicit mode-change error on mismatched mode updates.
- Added regression tests for defaults, allowed replace updates, and forbidden mode mutation.

### 15) Distributed runner lifecycle cleanup parity
- Improved distributed orchestration cleanup semantics in `runner::run_distributed(...)`:
  - role startup errors now preserve cleanup attempts,
  - worker/ps registrations are explicitly deregistered on normal completion,
  - discovery disconnect is consistently attempted after role execution.
- Upgraded registration flow to propagate discovery registration errors
  instead of silently ignoring them.
- Added assertion coverage ensuring worker service entries are removed from
  discovery after worker completion in runner smoke tests.

### 16) RunConfig/RunnerConfig → EstimatorConfig integration
- Added direct config bridge helpers:
  - `RunConfig::to_estimator_config()`
  - `RunnerConfig::to_estimator_config()`
- Added `Estimator::from_runner_config(...)` constructor for runtime wiring parity.
- Mapping includes:
  - `model_dir`,
  - `log_step_count_steps`,
  - `warm_start_from` derived from `restore_ckpt` when set.
- Added tests for both run/runner config bridges and estimator construction from runner config.

### 17) Runner checkpoint override edge-case hardening
- Improved `get_checkpoint_state_with_restore_override` matching logic to support
  Python-compatible restore checkpoint matching across multiple forms:
  - exact full-path matches,
  - basename-only matches against full checkpoint paths,
  - derived parent-path fallback matches.
- Added additional guard semantics:
  - restore overrides are applied only when `latest_filename == "checkpoint"`.
- Added tests covering:
  - basename → full-path override resolution,
  - non-`checkpoint` filename bypass behavior.

### 18) Runner discovery helper integration
- Expanded `RunnerDiscovery` with backend-agnostic operational helpers:
  - `register(name, index, addr)`,
  - `deregister(name, index, addr)`,
  - `query(name)`.
- This enables runner-side discovery callsites to use a unified API across
  TF_CONFIG/MLP/Consul backends.
- ZK descriptor variant now returns explicit guidance errors for register/query
  calls, directing callers to use concrete ZK discovery implementation.
- Added tests to validate:
  - Primus discovery queries through `RunnerDiscovery`,
  - expected error behavior when calling query on descriptor-only ZK variant.

### 19) Estimator constructors from RunConfig/RunnerConfig
- Added estimator construction helpers aligned with runtime config flows:
  - `Estimator::from_runner_config(...)`
  - `Estimator::from_run_config(...)`
- `from_runner_config` now applies runtime env exports before constructing estimator config.
- `from_run_config` performs run→runner merge parity and then builds estimator.
- Added tests verifying:
  - estimator config fields propagated correctly from run/runner configs,
  - runtime env export side effects occur during runner-config based construction.

### 20) Configurable restore synchronization timing parity
- Added restore synchronization timing fields to both config layers:
  - `restore_sync_timeout_secs`
  - `restore_sync_poll_interval_ms`
- Wired new defaults through merge + override extraction paths.
- Added `initialize_restore_checkpoint_from_runner_defaults(...)` to consume
  runner-config timing settings and drive restore wait behavior.
- Updated distributed runner config-entrypoint path to use these config-driven
  restore synchronization timings instead of hard-coded values.
- Added regression tests for merge behavior and defaulted restore initialization.

### 21) Estimator distributed runtime integration helper
- Added `Estimator::run_distributed_runtime(...)` to bridge estimator flows into
  distributed role orchestration using `RunnerConfig`.
- The helper delegates to `run_distributed_from_runner_config` and maps runtime
  failures into estimator-level error semantics (`EstimatorError::Distributed`).
- Added async smoke coverage validating PS/worker distributed startup through the
  estimator-facing runtime helper.

### 22) Native-training integration parity test expansion
- Expanded `tests/native_training_parity.rs` with higher-level cross-module checks:
  - estimator construction from `RunConfig` with warm-start propagation assertions,
  - discovery roundtrip checks via `get_discovery` + `RunnerDiscovery::query` for Primus.
- This complements unit-level tests by validating the newer estimator/runner/discovery
  APIs through integration-level entrypoints.

### 23) `MonolithDiscoveryGuard` API ergonomics parity
- Expanded discovery guard surface with direct operational helpers:
  - `kind()`
  - `register(name, index, addr)`
  - `deregister(name, index, addr)`
  - `query(name)`
- Added explicit local-mode behavior (`LocalModeNoDiscovery`) for discovery operations
  attempted without a backend.
- Added tests covering:
  - local guard operation error semantics,
  - Primus query pass-through through guard-level API.

### 24) Runner-utils integration test hardening
- Expanded `tests/runner_utils_parity.rs` to cover newly added guard-level operations:
  - `monolith_discovery(...).query("ps")` in Primus mode,
  - local-mode register error path through guard API.
- This ensures guard ergonomics are validated not only at unit level but also
  at integration-test entrypoints.

### 25) Estimator runtime initialization from runner config
- Added `Estimator::initialize_runtime_from_runner_config(...)`:
  - applies runner post-init restore/env behavior via runner-utils defaults helper,
  - returns optional checkpoint-state evidence for runtime orchestration.
- Added estimator-level error mapping for runner-utils failures.
- Added regression coverage ensuring runtime initialization:
  - syncs restore checkpoints for chief/local mode,
  - creates expected restore marker / monolith checkpoint side effects.

### 26) Discovery selection helper directly from `RunConfig`
- Added `get_discovery_from_run_config(run_conf, base, psm)`:
  - applies RunConfig→RunnerConfig merge semantics first,
  - then resolves discovery backend using existing runner discovery factory.
- Exported helper from crate root for downstream runtime callsites.
- Added unit + integration coverage for Primus selection via run-config path.

### 27) Discovery guard construction helper directly from `RunConfig`
- Added `monolith_discovery_from_run_config(run_conf, base, psm)`:
  - applies RunConfig→RunnerConfig merge,
  - constructs `MonolithDiscoveryGuard` from merged runner config in one step.
- Exported helper at crate root for runtime callsites.
- Added unit + integration tests validating Primus guard creation + query behavior
  through run-config entrypoint.

### 28) Initialized estimator constructor from `RunConfig`
- Added `Estimator::from_run_config_initialized(...)`:
  - performs run→runner merge,
  - executes runner runtime initialization (restore/env side effects),
  - returns `(Estimator, Option<CheckpointState>)` for callers that need restore evidence.
- Added regression coverage validating:
  - restore synchronization side effects via initialized constructor path,
  - estimator config propagation from run config values.

### 29) Distributed runner entrypoint directly from `RunConfig`
- Added `runner::run_distributed_from_run_config(...)`:
  - applies RunConfig→RunnerConfig merge semantics,
  - dispatches through existing runner-config distributed entrypoint.
- Exported helper at crate root for runtime callsites.
- Added unit + integration smoke coverage for run-config driven PS/worker flows.

### 30) Estimator distributed runtime helper directly from `RunConfig`
- Added `Estimator::run_distributed_runtime_from_run_config(...)`:
  - routes through run-config distributed runner entrypoint,
  - preserves estimator-level distributed error mapping.
- Added estimator-level async smoke coverage for run-config driven PS/worker flow.

### 31) Run-config restore initialization helper parity
- Added runner-utils restore initialization helpers from `RunConfig`:
  - `initialize_restore_checkpoint_from_run_config(...)`
  - `initialize_restore_checkpoint_from_run_config_defaults(...)`
- Added explicit `RunnerUtilsError::RunConfig` variant for merge failures.
- Updated run-config wrappers to use run-config-specific error mapping.
- Added estimator convenience API:
  - `Estimator::initialize_runtime_from_run_config(...)`.
- Refactored `Estimator::from_run_config_initialized(...)` to reuse the run-config
  runtime init path directly.
- Added unit + integration coverage for run-config restore initialization behavior.

### 32) Initialized estimator constructor from `RunnerConfig`
- Added `Estimator::from_runner_config_initialized(...)`:
  - applies runner runtime initialization (restore/env semantics),
  - returns `(Estimator, Option<CheckpointState>)`.
- Added regression coverage validating restore side effects and estimator config
  propagation through the runner-config initialized constructor path.

### 33) Local-cluster blocking barrier parity helper
- Extended `distributed::LocalCluster` barrier semantics:
  - added `DistributedError::BarrierTimeout { epoch, timeout_ms }`,
  - added `wait_for_barrier(worker_index, timeout, poll_interval)` blocking helper,
  - preserved released-epoch state so repeated callers at same epoch observe
    `BarrierStatus::Released`.
- Added regression tests for:
  - release visibility on repeated barrier checks,
  - timeout behavior when not all workers arrive,
  - successful blocking wait release when peers arrive.

### 34) PS server latency stats parity
- Implemented average latency tracking in `distributed_ps` server stats:
  - added aggregate latency counters for lookup/apply RPC paths,
  - computes `avg_lookup_latency_us` and `avg_apply_latency_us` from counters.
- Wired latency measurement into lookup/apply handlers with microsecond granularity.
- Added async regression test to validate:
  - lookup/apply request counters,
  - non-zero average latency fields,
  - stats payload consistency after real RPC handler invocation.

### 35) PS stats parity for failed apply requests
- Updated apply-gradients error path to participate in stats accounting:
  - failed gradient-shape requests now increment `apply_gradients_count`,
  - failed requests contribute to `avg_apply_latency_us`.
- Added regression coverage validating failed apply requests are reflected in
  stats counters and latency fields.

### 36) PS barrier runtime semantics hardening
- Hardened `distributed_ps` barrier behavior:
  - validates `num_workers > 0`,
  - enforces consistent `num_workers` per `barrier_id`,
  - returns explicit mismatch diagnostics when barrier shape changes.
- Improved barrier round handling:
  - leader resets arrived-count after successful round,
  - timeout path decrements pending arrival count best-effort.
- Added async regression tests covering:
  - multi-round barrier reuse and successful reset behavior,
  - mismatched `num_workers` rejection semantics.

### 37) Generation-based PS barrier state + duplicate worker guard
- Reworked PS barrier internals from raw `tokio::Barrier` round assumptions to
  explicit generation-based state:
  - `generation` tracked per `barrier_id`,
  - `arrived_workers` tracked explicitly per generation.
- Added duplicate worker protection:
  - repeated `worker_id` arrival in same generation now returns explicit error.
- Improved timeout safety:
  - timed-out worker is removed from current generation arrival set,
  - stale timeout races detect generation advancement and return success.
- Added async tests for:
  - duplicate worker rejection while first request is pending,
  - timeout cleanup allowing subsequent successful retry round,
  - existing mismatch/reset semantics remain covered.

### 38) PS barrier worker-id validation parity
- Added explicit worker-id range validation for barrier requests:
  - rejects `worker_id < 0` and `worker_id >= num_workers`.
- Added regression test ensuring out-of-range worker IDs fail with clear
  diagnostics and do not mutate barrier state.

### 39) PS client true parallel fanout parity
- Upgraded `distributed_ps::PsClient` to execute shard RPCs concurrently:
  - lookup fanout now runs all shard requests via `try_join_all`,
  - apply-gradients fanout now runs all shard updates in parallel as well.
- This aligns runtime behavior with module design intent and Python parity notes
  around parallel shard fanout semantics.
- Added end-to-end async regression coverage:
  - starts two real PS gRPC servers,
  - verifies cross-shard lookup/create behavior,
  - verifies duplicate-gradient aggregation + update correctness across shards.

### 40) PS client no-shard guard semantics
- Hardened `PsClient` configuration/runtime guards:
  - `connect(&[])` now returns `InvalidConfig` instead of creating an unusable client.
  - `lookup` / `apply_gradients` / `barrier` now fail fast with `InvalidConfig`
    when no shard clients are configured.
- Added async regression coverage for:
  - empty-address connect rejection,
  - lookup behavior on an explicitly empty client instance.

### 41) PS client early input validation parity
- Added client-side validation before network fanout:
  - `lookup` now rejects `dim_size == 0`,
  - `apply_gradients` now rejects `dim_size == 0`,
  - `apply_gradients` now validates gradient tensor length upfront and returns
    `DimensionMismatch` instead of panicking downstream in aggregation paths,
  - `barrier` now validates `num_workers > 0` and `worker_id` range before RPC.
- Added regression tests for:
  - zero-dim lookup rejection,
  - gradient-size mismatch rejection in apply path,
  - invalid worker-range rejection in barrier path.

### 42) PS client typed barrier error mapping parity
- Improved barrier client error semantics:
  - validates `timeout_ms > 0` before RPC,
  - maps barrier timeout responses to `PsError::Timeout(Duration)`,
  - maps non-timeout status-code `1` responses to `PsError::InvalidConfig`,
  - preserves status-code-aware mapping for cancellation/internal cases.
- Added async regression coverage for:
  - non-positive timeout rejection,
  - timeout mapping from real server barrier timeout response,
  - server-side barrier shape mismatch mapping to `InvalidConfig`.

### 43) PS client health/stats API parity
- Added explicit PS client helper APIs for observability:
  - `health_check_shard(shard_id, component)`
  - `health_check_all(component)` (parallel fanout)
  - `get_stats_shard(shard_id, include_table_stats)`
  - `get_stats_all(include_table_stats)` (parallel fanout)
- Added shard-index validation for single-shard helper calls.
- Added end-to-end async coverage:
  - shard health check on live gRPC server,
  - per-shard stats retrieval after lookup traffic,
  - multi-shard health/stats fanout behavior.

### 44) PS client batch multi-table convenience APIs
- Added batched client helpers aligned with proto batch messages:
  - `batch_lookup(BatchLookupRequest) -> BatchLookupResponse`
  - `batch_apply_gradients(BatchApplyGradientsRequest) -> BatchApplyGradientsResponse`
- Refactored client internals to expose reusable per-request response helpers:
  - `lookup_response(...)`
  - `apply_gradients_response(...)`
- Batched calls preserve per-request status semantics:
  - successful sub-requests return status `0`,
  - failed sub-requests are encoded with status `1` and per-entry error messages
    rather than aborting the whole batch.
- Added async regression coverage for mixed-success batch lookup/apply behavior.

### 45) Local cluster barrier timeout cleanup parity hardening
- Hardened `LocalCluster::wait_for_barrier(...)` timeout semantics so a timed-out
  worker is removed from the epoch waiter set before returning
  `DistributedError::BarrierTimeout`.
- Added internal helper `remove_barrier_waiter(epoch, worker_index)` to ensure
  stale timeout participants do not poison follow-up synchronization attempts.
- Added regression test `test_local_cluster_wait_for_barrier_timeout_cleanup_allows_retry`
  to verify:
  - a timed-out worker does not spuriously count as an arrived participant later,
  - subsequent retry from both workers releases the barrier correctly.

### 46) Batch lookup duplicate-preservation regression coverage
- Added async test `test_ps_client_batch_lookup_preserves_duplicate_found_flags` to
  validate that `batch_lookup` preserves per-request duplicate semantics:
  - first `create_if_missing=true` call reports duplicate entries as initialized,
  - subsequent lookup reports duplicate entries as found,
  - `num_found` / `num_initialized` match duplicated request cardinality.

### 47) Discovery guard lifecycle API hardening
- Added `MonolithDiscoveryGuard::close(&mut self)` for explicit, manual discovery
  teardown without waiting for drop.
- Close behavior is idempotent:
  - first close detaches and closes the backend,
  - repeated close calls are safe no-ops.
- Updated `Drop` implementation to delegate to `close()` so explicit/manual and
  implicit/drop lifecycles stay behaviorally aligned.
- Added integration test `test_monolith_discovery_guard_manual_close_is_idempotent`
  validating:
  - manual close transitions guard to local/no-backend mode,
  - subsequent query returns `LocalModeNoDiscovery`,
  - repeated close remains successful.

### 48) Batched PS client per-entry validation + fanout hardening
- Hardened `PsClient` batch helper semantics:
  - `batch_lookup(...)` now validates each sub-request `dim_size > 0` before cast,
    preventing invalid i32→usize conversion edge cases.
  - `batch_apply_gradients(...)` now validates each sub-request `dim_size > 0`
    with the same per-entry error semantics.
- Refactored batch request processing to run per-entry calls concurrently via
  `futures::join_all` while preserving response order.
- Updated internal response helpers (`lookup_response`, `apply_gradients_response`)
  to shared immutable-client usage needed by concurrent batch orchestration.
- Added async regression coverage:
  - `test_ps_client_batch_lookup_validates_dim_size_per_entry`
  - `test_ps_client_batch_apply_validates_dim_size_per_entry`
  ensuring invalid entries fail without aborting valid peer entries.

### 49) Local cluster released-barrier lifecycle pruning
- Added `LocalCluster::prune_released_barriers()` to prevent unbounded growth of
  `released_barriers` across long-running training epochs.
- Pruning policy preserves correctness for lagging workers:
  - keep release markers for epochs `>= min(worker_step)` so a slow worker can
    still observe already-released epochs,
  - drop only epochs that all workers have advanced beyond.
- `sync_barrier(...)` now performs prune-on-entry before release checks.
- Added regression test
  `test_local_cluster_prunes_released_barriers_after_all_workers_advance`
  validating:
  - lagging worker can still observe released epoch before it advances,
  - obsolete epoch markers are eventually pruned once all workers move forward.

### 50) Immutable/concurrency-friendly PS client API surface
- Updated `PsClient` request methods to borrow immutably (`&self`) where mutation
  is unnecessary:
  - `lookup`, `apply_gradients`, `barrier`,
  - `health_check_shard`, `health_check_all`,
  - `get_stats_shard`, `get_stats_all`,
  - `batch_lookup`, `batch_apply_gradients`.
- Added `#[derive(Clone)]` on `PsClient` for ergonomic sharing in runtime orchestration.
- Refactored shard RPC calls to use cloned tonic clients inside each method, removing
  mutable self requirements while preserving behavior and result ordering.
- Added async regression `test_ps_client_supports_parallel_immutable_lookups` to
  validate concurrent lookup calls on a shared immutable client reference.
- Updated worker runtime call path (`runner.rs`) to align with immutable PS client API.

### 51) Detailed PS client response metadata APIs
- Added public detailed response variants on `PsClient`:
  - `lookup_detailed(...) -> LookupResponse`
  - `apply_gradients_detailed(...) -> ApplyGradientsResponse`
- Existing convenience APIs now delegate internally:
  - `lookup(...)` delegates to `lookup_detailed(...)` and returns embeddings.
  - `apply_gradients(...)` delegates to `apply_gradients_detailed(...)` and returns
    `(num_updated, num_not_found)`.
- Added async regression `test_ps_client_detailed_lookup_and_apply_metadata`
  verifying that detailed API calls expose:
  - found/initialized counters and duplicate semantics for lookup,
  - explicit update/not-found counts and success status for apply.

### 52) Barrier wrapper alignment with immutable PS client
- Updated `PsBarrier` to use immutable lock-guard binding when calling
  `PsClient::barrier(...)`, matching the refactored immutable `PsClient` API.
- Keeps barrier call semantics unchanged while removing stale mutable-borrow
  assumptions after client interface hardening.

### 53) `monolith-core` path_utils test/runtime robustness hardening
- Hardened `find_main()` source-file canonicalization:
  - resolves `file!()` across both crate-relative and workspace-relative forms
    before canonicalization, reducing dependency on process working directory.
- Added deterministic env-var mutation isolation in `path_utils` tests using a
  shared test mutex, preventing racy `MONOLITH_MAIN_DIR` mutations across
  parallel test execution.
- Addresses intermittent workspace-test flakiness observed in
  `path_utils::tests::test_get_libops_path_points_to_file`.

### 54) Lock-free PS barrier wrapper concurrency parity
- Refactored `PsBarrier` to store `PsClient` directly instead of wrapping it in
  `tokio::sync::Mutex<PsClient>`.
- Because `PsClient` request APIs are now immutable (`&self`), barrier waits no
  longer require serializing all callers through a client mutex.
- Added barrier-layer tests in `barrier.rs`:
  - `test_in_memory_barrier_waits_for_all_workers`
  - `test_ps_barrier_allows_parallel_waits`
  validating concurrent wait behavior for both in-memory and PS-backed barrier
  implementations.

### 55) Shard-selectable PS barrier coordinator API
- Added `PsClient::barrier_on_shard(shard_id, barrier_id, worker_id, num_workers, timeout_ms)`
  for explicit barrier coordinator selection in multi-shard deployments.
- `PsClient::barrier(...)` now delegates to `barrier_on_shard(0, ...)` for backward-compatible default behavior.
- Added regression tests:
  - `test_ps_client_barrier_on_shard_rejects_invalid_index`
  - `test_ps_client_barrier_on_shard_routes_to_selected_coordinator`
  covering shard-index validation and explicit coordinator routing semantics.

### 56) Default PS health/stats client helpers
- Added convenience APIs on `PsClient`:
  - `health_check(component)` -> default shard (index 0)
  - `get_stats(include_table_stats)` -> default shard (index 0)
- Hardened shard APIs with explicit no-client guard semantics:
  - `health_check_shard(...)` now returns `InvalidConfig("no PS clients configured")`
    when client list is empty.
  - `get_stats_shard(...)` now mirrors the same no-client guard.
- Added async regression `test_ps_client_default_health_and_stats_methods`
  validating default helper behavior and stats retrieval on live PS gRPC server.

### 57) PS lookup reconstruction index-tracking hardening
- Replaced ad-hoc encoded shard/local position bookkeeping in
  `PsClient::lookup_response(...)` with explicit tuple mapping:
  - from `HashMap<i64, usize>` encoded positions
  - to `HashMap<i64, (usize, usize)>` explicit `(shard_id, local_idx)` tuples.
- This removes potential encoded-position collision risks for very large
  per-shard batches while keeping remap behavior unchanged.

### 58) Typed barrier-layer error mapping parity
- Enhanced `barrier.rs` error semantics so `PsBarrier` maps `PsClient` errors
  into barrier-domain variants:
  - `PsError::Timeout(_)` -> `BarrierError::Timeout`
  - `PsError::InvalidConfig(msg)` -> `BarrierError::InvalidConfig(msg)`
  - all other PS errors -> `BarrierError::Rpc(PsError)`
- Added targeted barrier-layer regressions:
  - `test_ps_barrier_maps_timeout_to_barrier_timeout`
  - `test_ps_barrier_maps_invalid_config_error`
  ensuring callers can distinguish timeout/configuration failures from generic
  RPC failures.

### 59) PS barrier direct-connect convenience API
- Added `PsBarrier::connect(addrs, timeout_ms) -> PsResult<PsBarrier>` to build
  PS-backed barriers directly from shard address lists.
- Keeps existing `PsBarrier::new(PsClient, timeout_ms)` path unchanged while
  simplifying runner/bootstrap call sites that only have addresses.
- Added regression `test_ps_barrier_connect_requires_addresses` to verify
  connect-time config guard behavior remains typed (`PsError::InvalidConfig`).

### 60) Worker runtime PS connection reuse
- Optimized distributed worker runtime setup to reuse one `PsClient` instance
  for both:
  - direct lookup/apply training operations,
  - barrier synchronization via `PsBarrier`.
- Removed duplicate `PsClient::connect(...)` call in worker role startup,
  reducing unnecessary connection establishment overhead.

### 61) Configurable worker barrier timeout in distributed runner
- Added `barrier_timeout_ms` to `DistributedRunConfig` with default `10_000`.
- Worker runtime now passes `cfg.barrier_timeout_ms` into `PsBarrier` instead of
  hardcoded timeout constants.
- Extended runner config mapping test coverage to assert default barrier timeout
  wiring (`distributed_config_from_runner`).

### 62) Train CLI parity: barrier-timeout propagation
- Added distributed CLI option `--barrier-timeout-ms` to `monolith-cli train`.
- Wired the option through `TrainCommand` into
  `DistributedRunConfig.barrier_timeout_ms` so runtime barrier wait behavior can
  be tuned from command-line configuration.
- Updated train-command test fixture constructors to include the new field and
  keep compile/test parity coverage green.

### 63) RunConfig/RunnerConfig distributed runtime tuning propagation
- Added distributed runtime tuning fields to both config layers:
  - `connect_retries`
  - `retry_backoff_ms`
  - `barrier_timeout_ms`
- Extended `RunConfig::to_runner_config(...)` merge behavior and
  `RunConfig::user_overrides()` metadata export to include these fields.
- Updated `distributed_config_from_runner(...)` to propagate retry/backoff/barrier
  timeout knobs into `DistributedRunConfig` instead of always using hardcoded
  runner defaults.
- Added/updated coverage:
  - runner mapping test now verifies `connect_retries`, `retry_backoff_ms`,
    and `barrier_timeout_ms` transfer correctly.
  - run-config merge/user-overrides tests now assert explicit propagation and
    override emission for all new distributed runtime tuning fields.

### 64) PS discovery heartbeat lifecycle cleanup in runner
- Hardened `run_ps_role(...)` heartbeat management to avoid leaked background
  heartbeat tasks when PS role exits or is cancelled.
- Replaced fire-and-forget heartbeat spawn with a watch-channel controlled task:
  - PS role now signals stop on server shutdown path.
  - heartbeat task is awaited during shutdown for deterministic cleanup.
  - cancellation paths (e.g., task abort) are handled via sender drop + stop
    signal handling.
- Added async regression `test_ps_heartbeat_task_stops_after_ps_task_abort`
  using a counting discovery backend to verify heartbeat increments stop after
  PS task cancellation.

### 65) Local-cluster start/stop lifecycle guard semantics
- Hardened `distributed::LocalCluster` role lifecycle behavior:
  - `start()` now rejects reentrant starts when any PS/worker role is already
    running.
  - `stop()` now rejects calls when cluster roles are already fully stopped.
- Added regression coverage:
  - `test_local_cluster_start_is_not_reentrant`
  - `test_local_cluster_stop_requires_running_cluster`
  ensuring lifecycle misuse is surfaced as typed invalid-configuration errors
  rather than silently no-oping.

### 66) Local PS/worker lifecycle guard parity hardening
- Removed remaining "stub" lifecycle semantics from `distributed` local role
  implementations by enforcing explicit start/stop state checks:
  - `ParameterServer::start()` now rejects reentrant starts.
  - `ParameterServer::stop()` now rejects stop-when-stopped calls.
  - `Worker::start()` now rejects reentrant starts.
  - `Worker::stop()` now rejects stop-when-stopped calls.
- Added regression coverage:
  - `test_parameter_server_lifecycle_guards`
  - `test_worker_lifecycle_guards`
  to validate typed invalid-configuration behavior for lifecycle misuse.

### 67) Local-cluster running-state enforcement for register/train paths
- Added `LocalCluster::ensure_cluster_running()` and wired it into:
  - `register_parameter(...)`
  - `train_step(...)`
- This prevents parameter registration or training updates from executing against
  partially started / stopped cluster roles.
- Added regression coverage:
  - `test_local_cluster_register_parameter_requires_running_cluster`
  - `test_local_cluster_train_step_requires_running_cluster`
  to validate explicit invalid-configuration errors when callers invoke these
  operations outside the running lifecycle window.

### 68) Cluster configuration duplicate-address validation parity
- Hardened `ClusterConfig::validate()` to reject duplicate endpoint entries:
  - duplicate parameter-server addresses,
  - duplicate worker addresses.
- This prevents ambiguous in-process topology setup and mirrors stricter
  distributed runtime configuration hygiene.
- Extended `test_cluster_config_validation` with explicit duplicate-address
  negative cases for both PS and worker lists.

### 69) Deterministic PS discovery ordering by shard metadata
- Hardened worker-side PS discovery address selection in `runner.rs`:
  - introduced `ordered_ps_addrs(...)` helper,
  - prefers discovery metadata `index` to build shard-ordered PS address lists,
  - falls back to address-sort ordering when index metadata is missing/invalid.
- This ensures client shard ordering aligns with advertised PS shard indices in
  multi-PS deployments, improving distributed runtime determinism.
- Added unit coverage:
  - `test_ordered_ps_addrs_prefers_discovery_index_metadata`
  - `test_ordered_ps_addrs_falls_back_to_address_sort_without_index`

### 70) Contiguous PS shard-index enforcement during worker discovery
- Strengthened metadata-driven PS ordering to require a contiguous shard index
  set for the expected worker configuration:
  - when index metadata is present, worker discovery now requires shard indexes
    `0..num_ps-1` before connecting.
  - missing intermediate shard indexes (e.g. 0,2 without 1) now force retry
    instead of silently connecting to a gapped shard set.
- Added regression `test_ordered_ps_addrs_requires_contiguous_index_set`.

### 71) Conflicting duplicate shard-index discovery guard
- Hardened metadata-index ordering logic to reject conflicting duplicate shard
  index advertisements (same `index` with different endpoint addresses).
- Worker discovery now treats this as inconsistent discovery state and retries
  instead of selecting one address arbitrarily.
- Added regression `test_ordered_ps_addrs_rejects_conflicting_duplicate_index`.

### 72) RunConfig/RunnerConfig propagation for discovery role names + table settings
- Expanded distributed runtime config bridge fields in `run_config.rs`:
  - `discovery_service_type_ps`
  - `discovery_service_type_worker`
  - `table_name`
  - `dim`
- Added merge + user-override parity handling for all four fields in
  `RunConfig::to_runner_config(...)` and `RunConfig::user_overrides()`.
- Updated `distributed_config_from_runner(...)` to propagate these values into
  `DistributedRunConfig` so runner-config based launches no longer rely on
  hardcoded defaults for discovery role names and embedding table dimensions.
- Expanded tests in `run_config.rs` and `runner.rs` to assert end-to-end field
  propagation and override visibility.

### 73) Strict PS metadata-consistency gating during worker discovery
- Hardened `ordered_ps_addrs(...)` metadata mode with stricter consistency
  requirements:
  - if **no** services provide `index` metadata: fallback to address-sorted mode.
  - if metadata is present but **mixed missing**, **invalid parse**, or otherwise
    incomplete/inconsistent: return empty and force discovery retry.
- This prevents silently mixing metadata-ordered and address-ordered semantics in
  unstable discovery windows.
- Added regressions:
  - `test_ordered_ps_addrs_rejects_mixed_index_metadata_presence`
  - `test_ordered_ps_addrs_rejects_invalid_index_metadata`

### 74) Distributed runtime config preflight validation
- Added `DistributedRunConfig::validate()` and wired it into
  `run_distributed(...)` before discovery/role startup.
- Enforced preflight constraints:
  - `num_ps > 0`
  - `num_workers > 0`
  - `dim > 0`
  - `barrier_timeout_ms > 0`
- Added async regression
  `test_run_distributed_rejects_invalid_runtime_config` to assert immediate
  typed failure on invalid distributed launch configuration.

### 75) Typed PS discovery ordering diagnostics in worker retry loop
- Refactored PS address ordering helper to return typed ordering outcomes:
  - `PsAddrOrderError::MissingOrGappedIndexSet`
  - `PsAddrOrderError::ConflictingDuplicateIndex`
  - `PsAddrOrderError::MixedIndexMetadataPresence`
  - `PsAddrOrderError::InvalidIndexMetadata`
- Worker discovery retry loop now preserves and surfaces the latest ordering
  issue in timeout errors for improved runtime diagnostics instead of returning
  opaque “got 0 expected N” messages in metadata-inconsistent states.
- Updated ordering regressions to assert explicit typed errors.

### 76) Worker timeout diagnostics regression coverage
- Added async runtime regression
  `test_run_worker_role_timeout_reports_ordering_issue`.
- Test seeds discovery with mixed metadata presence (`index` on one PS, missing
  on another) and verifies worker timeout includes explicit
  `MixedIndexMetadataPresence` issue signal in the error string.
- This validates the end-to-end diagnostic path (not just helper-level ordering
  return values).

### 77) ParameterSync replicator lifecycle cleanup in PS runner
- Refactored `ParameterSyncReplicator::spawn(...)` to return a managed
  `ParameterSyncReplicatorTask` handle with explicit `stop().await` semantics.
- Added watch-channel based shutdown coordination to the replicator loop, so
  background replication exits cleanly on:
  - explicit stop,
  - owner-drop cancellation paths.
- Wired `run_ps_role(...)` to retain the spawned replicator task handle and
  stop/join it during server shutdown, preventing orphaned replication loops.
- Added regression `test_parameter_sync_replicator_task_stop` to validate
  stop/join lifecycle behavior.

### 78) Worker registration failure cleanup robustness
- Hardened `run_distributed(...)` worker-role startup flow so worker
  registration errors no longer early-return before cleanup.
- Registration failure now feeds into common role error handling, ensuring:
  - `deregister_async(service_id)` is still attempted,
  - `disconnect()` is still attempted.
- Added async regression
  `test_run_distributed_disconnects_when_worker_registration_fails` using a
  failing discovery backend to assert cleanup calls happen even when
  registration fails before worker loop startup.

### 79) Worker discovery timeout retains last ordering issue across retries
- Hardened `run_worker_role(...)` retry loop to preserve the last observed
  `PsAddrOrderError` across attempts rather than only reporting the final
  attempt’s ordering result.
- This prevents loss of diagnostics when an earlier attempt observed metadata
  inconsistency but the final attempt simply returned an empty service list.
- Added async regression
  `test_run_worker_role_preserves_last_ordering_issue_across_retries` using a
  sequenced discovery backend (inconsistent metadata first, empty results next)
  to verify timeout messages retain `MixedIndexMetadataPresence`.

### 80) PS registration failure cleanup regression coverage
- Added async regression
  `test_run_distributed_disconnects_when_ps_registration_fails`.
- Uses the same failing discovery backend as worker registration-failure tests
  to assert PS-role startup failures still execute common cleanup:
  - `deregister_async` attempted,
  - `disconnect` attempted.
- Ensures role-specific registration error paths remain aligned with distributed
  runner cleanup semantics.

### 81) Worker timeout diagnostics retain last discovery backend error
- Hardened worker discovery retry logic to preserve the last discovery backend
  error string across attempts and include it in timeout diagnostics.
- Timeout errors now surface richer context combinations:
  - ordering issue + discovery error,
  - ordering issue only,
  - discovery error only,
  - neither (legacy generic timeout).
- Added async regression
  `test_run_worker_role_preserves_last_discovery_error_across_retries` using a
  sequenced discovery backend (error first, empty result next) to verify the
  original discovery failure context is retained at timeout.

### 82) Combined timeout diagnostics coverage (ordering + discovery error)
- Added async regression
  `test_run_worker_role_timeout_reports_ordering_and_discovery_errors`.
- Uses sequenced discovery behavior where:
  - first attempt returns mixed metadata (ordering inconsistency),
  - second attempt returns discovery backend failure.
- Verifies timeout diagnostics preserve and emit both context signals in the
  final error:
  - `MixedIndexMetadataPresence`,
  - underlying discovery error (`forced discover failure`).

### 83) ParameterSync replicator drop-safety lifecycle guard
- Added `Drop` implementation for `ParameterSyncReplicatorTask`:
  - best-effort stop signal send,
  - background join handle abort as a safety net when explicit
    `stop().await` is not called.
- This prevents forgotten task handles from leaving detached replication loops
  alive indefinitely.
- Added regression `test_parameter_sync_replicator_task_drop_is_safe`.

### 84) ParameterSync replicator handle ownership refinement
- Refined `ParameterSyncReplicatorTask` internals to store join handle as
  `Option<JoinHandle<()>>` and consume it exactly once.
- `stop().await` now takes ownership of the handle and awaits completion without
  leaving a stale handle to be aborted again in `Drop`.
- `Drop` now aborts only when the handle was not already consumed by explicit
  stop, making stop/drop interaction deterministic and avoiding redundant abort
  calls.

### 85) Worker timeout diagnostics now include max observed PS count
- Enhanced worker discovery timeout diagnostics to track and report
  `max observed` PS count across retries.
- This helps distinguish:
  - total discovery failure (`max observed: 0`),
  - partial cluster visibility (e.g., `max observed: 1` when `num_ps=2`),
  even when the final attempt returns fewer services.
- Added async regression
  `test_run_worker_role_timeout_reports_max_observed_ps_count`.

### 86) Worker timeout diagnostics split raw vs usable PS visibility
- Refined worker discovery timeout diagnostics to report both:
  - `max raw observed` (services discovered before ordering/consistency checks),
  - `max usable observed` (services surviving ordering validation).
- This makes metadata-consistency failures more actionable; e.g. inconsistent
  shard metadata now clearly shows raw discovery visibility while usable count
  remains zero.
- Added regression
  `test_run_worker_role_reports_raw_vs_usable_observed_counts`.

### 87) Worker timeout diagnostics now include retry attempt count
- Extended worker discovery timeout diagnostics to report total attempts made
  (`connect_retries + 1`) in final timeout messages.
- This improves operational debuggability by explicitly distinguishing retry
  budget exhaustion from immediate failures.
- Added async regression
  `test_run_worker_role_timeout_reports_attempt_count`.

### 88) Stale discovery-error cleanup after successful rediscovery
- Refined worker discovery retry diagnostics so successful discover calls clear
  previously latched backend-discovery errors.
- Prevents stale error strings from polluting final timeout messages when later
  attempts do reach the discovery backend successfully but still do not meet PS
  quorum.
- Added async regression
  `test_run_worker_role_clears_stale_discovery_error_after_successful_discover`.

### 89) Stale ordering-issue cleanup after usable rediscovery
- Refined worker discovery retry diagnostics so successful ordering with at
  least one usable endpoint clears previously latched ordering inconsistency
  markers.
- Prevents stale `last ordering issue` context from being reported after
  topology recovery when retries still time out due to insufficient PS quorum.
- Added async regression
  `test_run_worker_role_clears_stale_ordering_issue_after_usable_discovery`.

### 90) Worker heartbeat lifecycle cleanup + deterministic shutdown
- Refactored discovery heartbeat orchestration in `runner.rs` into shared helper
  utilities used by both PS and worker roles:
  - `spawn_heartbeat_task(...)`
  - `stop_heartbeat_task(...)`
- Added worker-role heartbeat coverage:
  - worker now runs backend heartbeat while waiting for PS discovery retries,
  - heartbeat task is explicitly stopped/joined before worker role returns
    (success or timeout error) to avoid background task leaks.
- Updated PS role to use the same shared heartbeat helpers for consistent
  lifecycle behavior across both distributed roles.
- Added async regression
  `test_worker_heartbeat_task_stops_after_worker_timeout` validating:
  - heartbeats are emitted during retry wait,
  - heartbeat count becomes stable after worker timeout exit (no leak).

### 91) Discovery watch poller lifecycle cleanup
- Refactored polling watch-loop logic into shared helper:
  - `spawn_watch_poll_loop(...)`.
- Updated ZK and Consul async watch implementations to use the shared poll-loop
  helper while preserving existing add/remove event semantics.
- Added deterministic poller shutdown behavior:
  - watch task now exits when all broadcast receivers are dropped,
  - send failures now trigger loop termination to avoid background task leaks.
- Added async regressions:
  - `test_spawn_watch_poll_loop_emits_added_and_removed_events`
  - `test_spawn_watch_poll_loop_stops_after_receivers_drop`
  validating both event emission and cleanup-on-unsubscribe behavior.

### 92) In-memory discovery watcher sender cleanup
- Hardened `InMemoryDiscovery::notify_watchers(...)` lifecycle behavior:
  - removes sender entries when there are no active receivers,
  - also removes sender entries when broadcast send fails.
- This prevents stale per-service watch sender entries from accumulating after
  subscribers disconnect.
- Added regression
  `test_in_memory_removes_dead_watchers_after_notification`.

### 93) Consul registration replacement bounded retry semantics
- Hardened `native_training::service_discovery::ConsulServiceDiscovery::register(...)`
  replacement flow for existing `(name, index)` entries:
  - added bounded replacement retries (`max_replace_retries`, default 60),
  - returns explicit timeout-style error when stale existing registration never
    clears (`Timed out clearing existing consul registration ...`),
  - short-circuits idempotently when the desired address is already visible.
- Added builder helper:
  - `with_max_replace_retries(...)` for deterministic tuning in tests.
- Added regressions:
  - `consul_idempotent_registration_short_circuits_when_addr_already_visible`
  - `consul_register_times_out_when_old_registration_never_clears`.

### 94) ZK registration-thread lock/close lifecycle cleanup
- Hardened `native_training::service_discovery::ZkServiceDiscovery` thread-map
  lifecycle handling to avoid holding shared lock across thread joins:
  - `register(...)`: remove/rejoin old thread outside lock before insert,
  - `deregister(...)`: remove thread handle, then join outside lock,
  - `close(...)`: drain thread map under lock, then join all drained threads
    without lock contention.
- Added regression:
  - `zk_close_is_idempotent_after_deregister`
  to validate close/deregister lifecycle remains safe and idempotent.

### 95) MLP discovery parity expansion (env shape + host validation)
- Expanded `py_discovery::MlpServiceDiscovery` and env parsing parity:
  - `MlpEnv` now captures:
    - `MLP_ROLE`,
    - `MLP_ROLE_INDEX`,
    - optional `MLP_HOST`,
    - role replica counts (`MLP_<ROLE>_NUM`).
  - Added `MlpServiceDiscovery` helper surface:
    - `server_type()`,
    - `index()`,
    - `addr()`,
    - `query_all()`,
    - `deregister_all()`.
- Hardened Python-style validation in register/deregister paths:
  - explicit role-existence checks,
  - non-empty query name guard,
  - host allowlist validation (local aliases + role hosts + current role host candidates),
  - existing port consistency check retained.
- Added regression coverage:
  - `test_mlp_service_discovery_query_all_and_filters`
  - `test_mlp_register_rejects_unexpected_host`
  - `test_mlp_query_requires_non_empty_name`.

### 96) MLP discovery close semantics parity
- Extended `py_discovery::MlpServiceDiscovery` lifecycle behavior to mirror
  Python `close()` semantics:
  - added internal closed-state guard,
  - `close()` now clears runtime filters and transitions discovery into
    no-op/empty-query mode.
- Post-close behavior now mirrors Python-style inert discovery:
  - `register` / `deregister` become no-op success,
  - `query` / `query_all` return empty maps,
  - `server_type()` / `addr()` return `None`.
- Added regression:
  - `test_mlp_close_disables_discovery_and_clears_filters`.

### 97) Workspace test-lane stability hardening (`model_registry`)
- Addressed intermittent workspace failure in
  `model_registry::tests::test_register_duplicate_error_message`
  caused by global registry cross-test interference.
- Added test-level serialization lock for model-registry unit tests in
  `monolith-core/src/model_registry.rs` to prevent concurrent
  `clear_registry_for_test()` races.
- This preserves behavior while making the workspace test lane deterministic.

### 98) MLP `query_all` role-filter parity tightening
- Refined `MlpServiceDiscovery::query_all()` so output includes only:
  - configured roles from environment (`MLP_<ROLE>_NUM`), and
  - supported service-discovery names (`ps`, `worker`, `chief`).
- This removes previously emitted extra empty keys for unconfigured supported
  roles (e.g. omitted `chief` when not configured) and ignores unsupported role
  names even when present in env counts.
- Added regression:
  - `test_mlp_query_all_only_includes_supported_configured_roles`.

### 99) Consul query_all malformed-entry diagnostics hardening
- Hardened `native_training::service_discovery::ConsulServiceDiscovery::query_all()`:
  - now validates required lookup fields strictly per entry:
    - `Port`,
    - `Tags.name`,
    - `Tags.ip`,
    - `Tags.index` (string or integer parseable).
  - malformed entries now return explicit typed errors instead of being silently
    coerced to default/empty values.
- Added regression:
  - `consul_query_all_rejects_malformed_entries`.

### 100) ZK discovery closed-state operation guards
- Hardened `native_training::service_discovery::ZkServiceDiscovery` lifecycle:
  - added explicit closed-state tracking,
  - `close()` now idempotently marks backend closed and exits early on repeated
    calls,
  - `register`, `query`, and `deregister` now return explicit errors when
    invoked after close.
- Added regression:
  - `zk_operations_fail_after_close`.

### 101) Runner-utils MLP discovery guard lifecycle regression coverage
- Added regression coverage for `monolith_discovery` guard close-path using MLP
  discovery backend.
- Ensures:
  - MLP discovery guard is initialized and queryable from env wiring,
  - `MonolithDiscoveryGuard::close()` is idempotent for the MLP backend,
  - post-close query returns `LocalModeNoDiscovery` from cleared guard state.
- Added per-test environment snapshot + mutex in `runner_utils` tests to
  guarantee deterministic env-var test isolation.

### 102) Consul discovery closed-state lifecycle guards
- Hardened `native_training::service_discovery::ConsulServiceDiscovery` with
  explicit closed-state tracking:
  - added idempotent `close()` implementation,
  - `register`, `query`, and `deregister` now fail explicitly after close.
- Added regression:
  - `consul_close_is_idempotent_and_blocks_operations`.

### 103) Consul query_all close-state consistency guard
- Hardened `ConsulServiceDiscovery::query_all()` to honor close-state
  consistently with other operations.
- After `close()`, direct `query_all()` calls now fail with explicit closed-state
  error (previously only `register/query/deregister` were guarded).
- Extended regression:
  - `consul_close_is_idempotent_and_blocks_operations`.

### 104) Consul clone lifecycle close-state parity regression
- Added regression verifying close-state is shared across cloned
  `ConsulServiceDiscovery` handles:
  - closing one clone disables operations on peer clones.
- Added regression:
  - `consul_close_state_is_shared_across_clones`.

### 105) Worker heartbeat success-path shutdown regression coverage
- Extended runner heartbeat lifecycle coverage to validate worker success path:
  - added delayed-discovery support in test discovery backend to ensure worker
    heartbeat ticks while discovery waits,
  - added regression that verifies heartbeat stops after worker completes
    successfully (not just timeout path).
- Added regression:
  - `test_worker_heartbeat_task_stops_after_worker_success`.

### 106) Heartbeat shutdown robustness when heartbeat RPC blocks
- Hardened `stop_heartbeat_task(...)` with bounded join timeout:
  - sends stop signal first,
  - waits up to 100ms for heartbeat task exit,
  - logs timeout warning and avoids indefinite shutdown hang if heartbeat call
    is stuck.
- Added regression with a blocking heartbeat backend to verify worker shutdown
  still returns promptly:
  - `test_run_worker_role_does_not_hang_when_heartbeat_blocks`.

### 107) In-flight heartbeat cancellation via stop-aware select loop
- Refactored heartbeat task loop to make in-flight heartbeat calls stop-aware:
  - when interval fires, heartbeat RPC is now raced against stop-signal changes,
    enabling cancellation even while heartbeat RPC is blocked.
- Added regression for PS abort lifecycle:
  - verifies aborting PS role cancels an in-flight blocking heartbeat call and
    releases active-heartbeat state.
- Added regression:
  - `test_ps_abort_cancels_inflight_blocking_heartbeat`.

### 108) Connect-failure discovery cleanup in distributed runner
- Hardened `run_distributed(...)` initialization path:
  - if `discovery.connect()` fails, runner now attempts
    `discovery.disconnect()` best-effort before returning error.
- Added regression:
  - `test_run_distributed_attempts_disconnect_when_connect_fails`.

### 109) Forced heartbeat-task abort after shutdown timeout
- Tightened `stop_heartbeat_task(...)` timeout path:
  - if heartbeat task does not stop within timeout window, it is now explicitly
    aborted and awaited, preventing detached long-lived task leaks.
- Added regression:
  - `test_stop_heartbeat_task_aborts_nonterminating_task`.

### 110) Disconnect-after-success cleanup regression for deregister failures
- Added distributed-runner lifecycle regression ensuring disconnect still runs
  when post-success `deregister_async` fails:
  - worker role completes successful PS interactions against a live PS server,
  - backend is forced to fail deregistration,
  - runner still attempts `disconnect`.
- Added regression:
  - `test_run_distributed_attempts_disconnect_when_deregister_fails_after_success`.

### 111) Worker-timeout cleanup regression coverage after successful registration
- Added distributed-runner regression for worker timeout path where service
  registration already succeeded:
  - validates timeout error surface,
  - validates cleanup sequence still attempts both `deregister` and
    `disconnect`,
  - validates worker registration is removed from discovery state.
- Added regression:
  - `test_run_distributed_disconnects_when_worker_role_fails_after_registration`.

### 112) Disconnect-failure propagation after successful worker completion
- Added distributed-runner regression for success path where:
  - worker role succeeds against a live PS,
  - `deregister_async` succeeds,
  - `disconnect()` fails.
- Verifies runner:
  - still performs deregister before disconnect,
  - returns disconnect error explicitly.
- Added regression:
  - `test_run_distributed_surfaces_disconnect_failure_after_success`.

### 113) Combined post-success cleanup failure semantics hardening
- Hardened `run_distributed(...)` success-path cleanup handling:
  - when `deregister` fails, runner now still checks/disconnects and logs any
    disconnect failure before returning the primary deregister error.
- Added regressions for combined failure/error-priority and connect+disconnect
  dual-failure cleanup behavior:
  - `test_run_distributed_prefers_deregister_error_when_both_post_success_cleanup_steps_fail`
  - `test_run_distributed_returns_connect_error_when_connect_and_disconnect_fail`.

### 114) Disconnect-timeout cleanup regression coverage after successful run
- Added success-path cleanup regression where:
  - worker role completes successfully,
  - deregister succeeds,
  - disconnect blocks and times out.
- Verifies timeout error is surfaced with explicit cleanup context while
  preserving deterministic cleanup call ordering/counters.
- Added regression:
  - `test_run_distributed_surfaces_disconnect_timeout_after_success`.

### 115) Discovery-cleanup timeout wrapper + worker-error precedence coverage
- Introduced bounded cleanup helper for distributed runner teardown:
  - `await_discovery_cleanup(...)` wraps discovery cleanup ops
    (`deregister`/`disconnect`) with timeout and explicit timeout error context.
- Added cleanup-timeout regressions:
  - connect-failure path with blocking disconnect does not hang and preserves
    primary connect error (`test_run_distributed_connect_failure_does_not_hang_when_disconnect_blocks`),
  - post-success deregister timeout surfaces explicit cleanup timeout while still
    attempting disconnect (`test_run_distributed_surfaces_deregister_timeout_after_success`),
  - role-error precedence is preserved even when both cleanup steps time out
    (`test_run_distributed_preserves_worker_error_when_cleanup_steps_timeout`).

### 116) Parameter-sync replicator deterministic stop under stuck task conditions
- Hardened `ParameterSyncReplicatorTask::stop` to avoid indefinite awaits:
  - added bounded stop timeout and forced `abort()` fallback when join does not
    complete promptly,
  - emits warning with timeout context when forced abort path is used.
- Improved replicator spawn loop cancellation behavior:
  - each scheduled flush now races `flush_once()` against stop signal so stop can
    preempt in-flight flush execution points.
- Added regression:
  - `test_parameter_sync_replicator_task_stop_aborts_nonterminating_task`.

### 117) Connect/register blocking-operation hardening in distributed runner
- Added bounded timeout wrapper for discovery setup operations:
  - `await_discovery_operation(...)` now guards `connect` and `register` calls
    to prevent indefinite hangs before role execution.
- Updated distributed runner flow:
  - `run_distributed` now uses bounded `connect` and worker `register` calls.
  - `run_ps_role` now uses bounded `register` call before server startup.
- Added blocking-operation regressions:
  - `test_run_distributed_connect_timeout_does_not_hang_and_attempts_disconnect`
  - `test_run_distributed_worker_register_timeout_does_not_hang`
  - `test_run_distributed_ps_register_timeout_does_not_hang`
  - verifies timeout surfacing plus deterministic cleanup attempts
    (`deregister` + `disconnect`) after setup-stage timeout failures.

### 118) Configurable discovery setup timeout surface + CLI integration
- Made discovery setup timeout configurable per distributed run config:
  - added `discovery_operation_timeout` to `DistributedRunConfig`,
  - defaulted to `5s` to avoid overly aggressive setup-stage timeouts for
    non-test backends.
- Updated setup-timeout wrapper usage to consume config timeout for:
  - `connect` in `run_distributed`,
  - `register` in worker/PS setup paths.
- Updated distributed train CLI wiring to populate new config field when
  constructing `DistributedRunConfig` for runtime execution.

### 119) Configurable discovery cleanup timeout semantics + bounded-delay regression
- Extended distributed runner config with
  `discovery_cleanup_timeout: Duration`.
- Updated cleanup wrapper call sites to consume per-run timeout for:
  - `deregister`,
  - `disconnect`,
  across connect-failure and role-exit cleanup paths.
- Added regression to validate configured timeout is honored:
  - `test_run_distributed_honors_configured_cleanup_timeout`
  - verifies reduced cleanup timeout bounds total cleanup delay while preserving
    worker-role primary error precedence.
- Updated train CLI distributed config wiring to include new cleanup timeout
  field for end-to-end config completeness.

### 120) Worker discovery-query timeout resilience in retry loop
- Generalized discovery operation timeout helper to support typed operation
  results, then applied it to worker-side `discover_async` calls.
- Worker discovery loop now bounds each discover attempt using
  `discovery_operation_timeout` so blocked backends cannot hang retry logic.
- Added blocking discover regressions:
  - `test_run_worker_role_retries_when_discover_operation_times_out`
  - `test_run_distributed_worker_discover_timeout_does_not_hang_and_cleans_up`
- Coverage verifies:
  - discover-operation timeouts are reflected in worker timeout diagnostics,
  - retry budget still advances under discover timeouts,
  - role-level cleanup (`deregister` + `disconnect`) still runs when discovery
    operations block.

### 121) Train CLI parity exposure for discovery timeout controls
- Extended `monolith-cli train` flags with explicit discovery timeout controls:
  - `--discovery-operation-timeout-ms` (default `5000`)
  - `--discovery-cleanup-timeout-ms` (default `200`)
- Wired new flags through distributed config assembly into runner runtime:
  - `DistributedRunConfig::discovery_operation_timeout`
  - `DistributedRunConfig::discovery_cleanup_timeout`
- Updated train command default test fixtures to include new timeout fields so
  CLI parity defaults remain explicit and regression-protected.

### 122) RunConfig/RunnerConfig parity propagation for discovery timeout controls
- Extended native training config models with timeout fields:
  - `RunConfig::discovery_operation_timeout_ms`
  - `RunConfig::discovery_cleanup_timeout_ms`
  - `RunnerConfig::discovery_operation_timeout_ms`
  - `RunnerConfig::discovery_cleanup_timeout_ms`
- Added merge and override propagation for both timeout fields in:
  - `RunConfig::to_runner_config`
  - `RunConfig::user_overrides`
- Updated distributed runner config mapping so RunnerConfig-derived execution
  now carries timeout values into:
  - `DistributedRunConfig::discovery_operation_timeout`
  - `DistributedRunConfig::discovery_cleanup_timeout`
- Expanded parity regressions to assert timeout field merge/override visibility
  and runner mapping correctness.

### 123) Distributed timeout configuration validation hardening
- Strengthened `DistributedRunConfig::validate()` to reject invalid timeout
  configurations:
  - `discovery_operation_timeout == 0`,
  - `discovery_cleanup_timeout == 0`.
- Added validation regressions:
  - `test_distributed_config_validate_rejects_zero_discovery_operation_timeout`
  - `test_distributed_config_validate_rejects_zero_discovery_cleanup_timeout`
- Ensures invalid timeout config fails fast before runtime discovery orchestration
  starts.

### 124) Timeout diagnostics now include concrete timeout duration
- Enhanced timeout error messages for both wrappers to include elapsed budget:
  - `Timed out during discovery operation: <op> <service_id> after <N>ms`
  - `Timed out during discovery cleanup: <op> <service_id> after <N>ms`
- Expanded regressions to assert timeout duration visibility in returned errors
  for:
  - connect/register/discover operation timeout paths,
  - deregister/disconnect cleanup timeout paths.
- This improves runtime diagnosability for timeout tuning and environment
  debugging without changing error-precedence behavior.

### 125) Setup-timeout precedence hardened when cleanup also blocks
- Added blocked-cleanup precedence regressions for setup-stage timeout paths:
  - `test_run_distributed_connect_timeout_preserves_error_when_disconnect_cleanup_times_out`
  - `test_run_distributed_worker_register_timeout_preserves_error_when_cleanup_times_out`
- Added dedicated blocking backends to simulate:
  - connect timeout + blocking disconnect cleanup,
  - register timeout + blocking deregister/disconnect cleanup.
- Coverage verifies:
  - distributed runner remains non-hanging under compounded blocking conditions,
  - setup-stage timeout error remains primary even when cleanup operations also
    time out,
  - cleanup attempts still execute once (count-based assertions).

### 126) PS register timeout precedence hardened under blocked cleanup
- Added PS-role counterpart regression for setup-timeout precedence when cleanup
  operations also block:
  - `test_run_distributed_ps_register_timeout_preserves_error_when_cleanup_times_out`
- Confirms parity invariants for PS setup path:
  - runner does not hang under blocked register+deregister+disconnect,
  - PS register timeout error remains primary even when cleanup times out,
  - cleanup attempts still execute (count assertions).

### 127) Train CLI distributed-config builder parity hardening
- Refactored train command distributed config assembly into dedicated helper:
  - `TrainCommand::build_distributed_run_config()`
- Added focused unit coverage for CLI→runner config mapping semantics:
  - timeout flag propagation (`operation`/`cleanup` ms),
  - heartbeat enable/disable behavior,
  - non-distributed early-none path,
  - invalid bind-address validation behavior.
- Added PS blocked-cleanup precedence counterpart regression:
  - `test_run_distributed_ps_register_timeout_preserves_error_when_cleanup_times_out`
  - ensures setup-timeout primacy and non-hanging behavior across both worker
    and PS setup paths.

### 128) CLI timeout input guardrails + distributed-config builder validation
- Hardened train CLI distributed config builder with explicit input validation:
  - rejects `--discovery-operation-timeout-ms=0`,
  - rejects `--discovery-cleanup-timeout-ms=0`,
  - returns actionable CLI-facing validation errors before runtime dispatch.
- Expanded train-command builder test coverage:
  - timeout field mapping and heartbeat behavior,
  - non-distributed no-op path,
  - invalid bind address path,
  - zero-timeout rejection paths.
- Keeps runtime validation intact while surfacing misconfiguration earlier at
  CLI assembly boundaries.

### 129) Run-config timeout propagation integration coverage (worker discover path)
- Added integration-level regression ensuring `RunConfig` timeout controls
  propagate through `run_distributed_from_run_config(...)` into runtime worker
  discovery behavior:
  - `distributed_runner_from_run_config_honors_discover_timeout_controls`
- Uses blocking discovery backend to verify:
  - discover operation timeout surfaces with configured duration context,
  - run-config timeout fields (`operation` + `cleanup`) are honored end-to-end,
  - cleanup attempts still execute after timeout-driven worker failure.

### 130) Run-config connect-timeout precedence coverage under blocked cleanup
- Added integration-level regression for run-config connect path with blocked
  cleanup:
  - `distributed_runner_from_run_config_preserves_connect_timeout_when_cleanup_blocks`
- Uses backend where `connect` and cleanup `disconnect` both block to validate:
  - runtime does not hang,
  - connect timeout remains primary error (with configured duration),
  - cleanup attempt still occurs exactly once.
- Complements worker-discover integration coverage to cover setup-timeout
  precedence invariants through run-config entrypoint.

### 131) Runner-config connect-timeout precedence integration coverage
- Added runner-config entrypoint counterpart regression:
  - `distributed_runner_from_runner_config_preserves_connect_timeout_when_cleanup_blocks`
- Confirms through `run_distributed_from_runner_config(...)` that:
  - blocked connect + blocked cleanup disconnect does not hang,
  - configured connect timeout remains primary error with duration context,
  - cleanup disconnect attempt still executes once.
- Ensures setup-timeout precedence parity is now covered across both
  run-config and runner-config distributed entrypoints.

### 132) Run-config worker register-timeout precedence integration coverage
- Added integration regression for run-config entrypoint worker registration:
  - `distributed_runner_from_run_config_preserves_register_timeout_when_cleanup_blocks`
- Uses backend where worker `register_async` and cleanup operations
  (`deregister_async` + `disconnect`) block, validating:
  - runtime returns without hanging,
  - register timeout remains the primary surfaced error (with configured
    duration context),
  - cleanup attempts are still executed.
- Extends run-config integration timeout-precedence coverage from connect and
  discover paths to worker register path.

### 133) Runner-config worker register-timeout precedence integration coverage
- Added runner-config entrypoint counterpart regression:
  - `distributed_runner_from_runner_config_preserves_register_timeout_when_cleanup_blocks`
- Uses backend with blocked worker `register_async` and blocked cleanup
  (`deregister_async` + `disconnect`) to validate:
  - non-hanging runtime behavior,
  - worker register timeout remains the primary surfaced error (with configured
    duration context),
  - cleanup attempts are still executed.
- Completes worker register timeout-precedence integration coverage across both
  run-config and runner-config entrypoints.

### 134) PS register-timeout precedence integration coverage across config entrypoints
- Added PS-role integration regressions for both config-driven entrypoints:
  - `distributed_runner_from_run_config_preserves_ps_register_timeout_when_cleanup_blocks`
  - `distributed_runner_from_runner_config_preserves_ps_register_timeout_when_cleanup_blocks`
- Uses backend with blocked `register_async` and blocked cleanup operations
  (`deregister_async` + `disconnect`) to validate:
  - runtime non-hanging behavior in PS role setup failure path,
  - PS register timeout remains the primary surfaced error (with configured
    duration context),
  - cleanup attempts still execute.
- Completes setup-timeout precedence integration matrix for worker + PS register
  flows across run-config and runner-config entrypoints.

### 135) Runner-config discover-timeout propagation integration coverage
- Added runner-config entrypoint discover-timeout regression:
  - `distributed_runner_from_runner_config_honors_discover_timeout_controls`
- Uses blocking worker discover backend to validate through
  `run_distributed_from_runner_config(...)` that:
  - configured operation timeout propagates to discover operation diagnostics,
  - runtime remains non-hanging,
  - cleanup attempts still execute after discover timeout.
- Complements existing run-config discover-timeout integration coverage for
  entrypoint parity completeness.

### 136) Discover-retry propagation integration coverage across config entrypoints
- Added retry-focused discover-timeout integration regressions:
  - `distributed_runner_from_run_config_propagates_discover_retry_controls`
  - `distributed_runner_from_runner_config_propagates_discover_retry_controls`
- Validates with `connect_retries=2` and blocking discover backend that:
  - exactly three discover attempts are made,
  - timeout diagnostics still include configured operation timeout context,
  - cleanup attempts execute after retries are exhausted.
- Extends timeout-control parity coverage from value propagation to retry-loop
  semantics propagation in both config entrypoints.

### 137) Retry-backoff propagation integration coverage across config entrypoints
- Added backoff-focused discover retry regressions:
  - `distributed_runner_from_run_config_propagates_retry_backoff_controls`
  - `distributed_runner_from_runner_config_propagates_retry_backoff_controls`
- Uses empty-discovery backend with `connect_retries=2` and non-trivial
  `retry_backoff_ms` to validate:
  - retry loop performs expected number of discover attempts,
  - elapsed runtime reflects configured backoff delays,
  - failure remains deterministic (`Timed out waiting for PS discovery`) and
    cleanup attempts still execute.
- Complements retry-count and timeout-propagation tests by proving
  backoff-duration propagation from both config entrypoints into runtime retry
  behavior.

### 138) Custom discovery service-type propagation integration coverage
- Added service-type focused integration regressions:
  - `distributed_runner_from_run_config_propagates_custom_service_type_fields`
  - `distributed_runner_from_runner_config_propagates_custom_service_type_fields`
- Introduced recording discovery backend to capture runtime service-type usage
  across registration and discovery operations.
- Validates across both config entrypoints that:
  - worker registration uses `discovery_service_type_worker`,
  - worker discovery queries `discovery_service_type_ps`,
  - PS registration uses `discovery_service_type_ps`.
- Ensures config-level custom service-type controls are propagated end-to-end
  into runtime discovery lifecycle calls.

### 139) Config-surface service-type mapping assertion hardening
- Strengthened config-level mapping regressions for service-type fields:
  - train CLI distributed-config builder mapping test now asserts custom
    `discovery_service_type_ps` and `discovery_service_type_worker` values are
    preserved in generated `DistributedRunConfig`.
  - run-config user-overrides regression now asserts
    `discovery_service_type_worker` participates in explicit override reporting
    alongside existing PS service-type override checks.
- Complements runtime propagation integrations by ensuring upstream config and
  CLI mapping surfaces preserve custom service-type inputs deterministically.

### 140) PS connect-timeout precedence integration coverage across config entrypoints
- Added PS-role connect-timeout precedence regressions:
  - `distributed_runner_from_run_config_preserves_ps_connect_timeout_when_cleanup_blocks`
  - `distributed_runner_from_runner_config_preserves_ps_connect_timeout_when_cleanup_blocks`
- Uses backend with blocked `connect` and blocked cleanup `disconnect` to
  validate for both config entrypoints that:
  - runtime does not hang,
  - PS connect timeout remains the primary surfaced error with configured
    duration context,
  - cleanup disconnect attempt still executes.
- Completes connect-timeout precedence integration coverage for both worker and
  PS roles across run-config and runner-config entrypoints.

### 141) Zero-timeout validation integration coverage across config entrypoints
- Added validation-focused config entrypoint regressions:
  - `distributed_runner_from_run_config_rejects_zero_operation_timeout`
  - `distributed_runner_from_run_config_rejects_zero_cleanup_timeout`
  - `distributed_runner_from_runner_config_rejects_zero_operation_timeout`
  - `distributed_runner_from_runner_config_rejects_zero_cleanup_timeout`
- Validates that zero timeout values propagated from both run-config and
  runner-config surfaces are rejected by distributed config validation before
  runtime orchestration.
- Confirms entrypoint-level timeout propagation includes invalid-value rejection
  semantics in addition to previously covered timeout and retry behavior.

### 142) Cleanup-timeout propagation integration coverage across config entrypoints
- Added cleanup-timeout focused integration regressions:
  - `distributed_runner_from_run_config_honors_cleanup_timeout_with_blocked_cleanup`
  - `distributed_runner_from_runner_config_honors_cleanup_timeout_with_blocked_cleanup`
- Uses blocked connect + blocked cleanup disconnect backend with small configured
  operation/cleanup timeouts to validate that:
  - runtime remains non-hanging,
  - operation timeout diagnostics reflect configured values,
  - total elapsed runtime is bounded by configured cleanup timeout semantics
    rather than drifting toward long/default cleanup waits.
- Complements existing operation-timeout precedence and retry propagation tests
  by adding direct elapsed-time evidence for cleanup-timeout propagation through
  both config entrypoints.

### 143) Register-path cleanup-timeout propagation integration coverage
- Added register-path cleanup-timeout regressions:
  - `distributed_runner_from_run_config_honors_cleanup_timeout_after_register_timeout`
  - `distributed_runner_from_runner_config_honors_cleanup_timeout_after_register_timeout`
- Uses blocked worker `register_async` plus blocked cleanup operations with
  small configured operation/cleanup timeouts to validate:
  - runtime remains non-hanging,
  - register timeout diagnostics preserve configured operation timeout context,
  - total elapsed runtime stays bounded by configured cleanup-timeout behavior
    across both config entrypoints.
- Complements connect-path cleanup-timeout coverage to ensure cleanup-timeout
  propagation is validated for both setup failure families (connect + register).

### 144) Barrier-timeout propagation integration coverage across config entrypoints
- Added barrier-timeout focused integration regressions:
  - `distributed_runner_from_run_config_propagates_barrier_timeout_controls`
  - `distributed_runner_from_runner_config_propagates_barrier_timeout_controls`
- Uses real in-memory discovery + live PS role task with `num_workers=2` and a
  single active worker to force barrier wait timeout in worker role path.
- Validates across both config entrypoints that:
  - worker failure surfaces barrier-timeout semantics (`Barrier timeout`),
  - runtime returns promptly (bounded elapsed time) according to configured
    `barrier_timeout_ms` controls rather than default long waits.
- Extends config-entrypoint propagation coverage beyond discovery/cleanup
  controls to include synchronization timeout semantics.

### 145) Barrier-timeout input validation coverage across CLI/config entrypoints
- Added validation coverage for invalid barrier-timeout inputs:
  - train CLI distributed config builder now rejects `--barrier-timeout-ms <= 0`
    with actionable CLI-facing error.
  - Added CLI regression:
    - `test_build_distributed_run_config_rejects_zero_barrier_timeout`
  - Added run/runner config entrypoint integration regressions:
    - `distributed_runner_from_run_config_rejects_zero_barrier_timeout`
    - `distributed_runner_from_runner_config_rejects_zero_barrier_timeout`
- Confirms invalid barrier-timeout values are rejected consistently across both
  configuration assembly and distributed entrypoint runtime validation surfaces.

### 146) Negative barrier-timeout validation coverage completion
- Expanded barrier-timeout validation coverage to include negative values across
  all surfaced configuration paths:
  - CLI distributed-config builder rejects negative `--barrier-timeout-ms`
    values (`test_build_distributed_run_config_rejects_negative_barrier_timeout`).
  - run-config/runner-config entrypoint regressions assert negative
    `barrier_timeout_ms` values are rejected by distributed config validation:
    - `distributed_runner_from_run_config_rejects_negative_barrier_timeout`
    - `distributed_runner_from_runner_config_rejects_negative_barrier_timeout`
- Completes barrier-timeout input validation matrix for non-positive values
  (zero + negative) across CLI, run-config, and runner-config paths.

### 147) Distributed dim input validation coverage across CLI/config entrypoints
- Added distributed-mode dim validation in train CLI distributed-config builder:
  - rejects `--dim=0` with actionable CLI-facing error.
- Added CLI regression:
  - `test_build_distributed_run_config_rejects_zero_dim`
- Added run/runner config entrypoint integration regressions:
  - `distributed_runner_from_run_config_rejects_zero_dim`
  - `distributed_runner_from_runner_config_rejects_zero_dim`
- Confirms `dim > 0` validation is enforced consistently across CLI assembly
  and both config entrypoint runtime validation paths.

### 148) Worker-index timeout diagnostic propagation integration coverage
- Added worker-index diagnostic propagation regressions:
  - `distributed_runner_from_run_config_propagates_worker_index_into_connect_timeout_diagnostics`
  - `distributed_runner_from_runner_config_propagates_worker_index_into_connect_timeout_diagnostics`
- Uses blocked connect + blocked cleanup backend with non-zero worker indices
  to validate that timeout diagnostics include the expected role/index service-id
  (`worker-<index>`) propagated from run-config and runner-config entrypoints.
- Strengthens diagnostic parity by proving index propagation through config
  entrypoints into runtime timeout error contexts.

### 149) PS-index timeout diagnostic propagation integration coverage
- Added PS-index diagnostic propagation regressions:
  - `distributed_runner_from_run_config_propagates_ps_index_into_connect_timeout_diagnostics`
  - `distributed_runner_from_runner_config_propagates_ps_index_into_connect_timeout_diagnostics`
- Uses blocked connect + blocked cleanup backend with non-zero ps indices to
  verify that timeout diagnostics include `ps-<index>` service-id context from
  run-config and runner-config entrypoints.
- Extends diagnostic parity coverage to both role families (worker + ps) for
  connect-timeout error attribution.

### 150) Register-timeout diagnostic enrichment with service-type context
- Extended discovery operation timeout diagnostics to support richer operation
  descriptors (operation strings now carry full context instead of relying on a
  fixed `op + service_id` formatter).
- Updated worker/ps register timeout paths to include service-type context in
  timeout diagnostics:
  - `register <service_id> as <service_type>`
- Added integration regressions proving custom service-type propagation into
  register timeout diagnostics across run/runner config entrypoints:
  - `distributed_runner_from_run_config_propagates_worker_service_type_into_register_timeout_diagnostics`
  - `distributed_runner_from_runner_config_propagates_worker_service_type_into_register_timeout_diagnostics`
  - `distributed_runner_from_run_config_propagates_ps_service_type_into_register_timeout_diagnostics`
  - `distributed_runner_from_runner_config_propagates_ps_service_type_into_register_timeout_diagnostics`
- Updated existing timeout regressions to assert new default diagnostic form:
  - worker: `register worker-0 as worker ...`
  - ps: `register ps-0 as ps ...`

### 151) Discover-timeout diagnostic enrichment with queried service-type context
- Extended worker discover timeout diagnostics to include queried discovery
  service-type context:
  - operation descriptor now emits
    `discover <worker-id> for <discovery_service_type_ps>`.
- Updated internal distributed-runner discover-timeout regressions to assert
  default discover diagnostic form (`for ps`).
- Added run/runner config integration regressions proving custom
  `discovery_service_type_ps` propagation into discover-timeout diagnostics:
  - `distributed_runner_from_run_config_propagates_discover_service_type_into_timeout_diagnostics`
  - `distributed_runner_from_runner_config_propagates_discover_service_type_into_timeout_diagnostics`
- Strengthens diagnostic parity by surfacing both caller identity and queried
  service class during discover timeout failures.

### 152) Runner-level discover timeout service-type diagnostic regression
- Added direct runner-level regression:
  - `test_run_worker_role_discover_timeout_includes_service_type_context`
- Verifies `run_worker_role` timeout diagnostics retain queried service-type
  context (`discover worker-0 for <custom-type>`) even outside higher-level
  run-config/runner-config entrypoint wrappers.
- Complements integration coverage with a focused unit-level guard on worker
  discover timeout message fidelity.

### 153) Distributed-runner discover timeout custom service-type regression
- Added direct distributed-runner regression:
  - `test_run_distributed_worker_discover_timeout_includes_custom_service_type_context`
- Verifies end-to-end `run_distributed(...)` worker discover-timeout diagnostics
  retain custom queried service-type context and preserve cleanup behavior
  (`deregister` + `disconnect`) when discover blocks.
- Complements worker-role unit coverage with top-level distributed entrypoint
  coverage for custom service-type timeout message fidelity.

### 154) Distributed-runner register timeout custom service-type regressions
- Added direct distributed-runner regressions for custom register service-type
  timeout diagnostics:
  - `test_run_distributed_worker_register_timeout_includes_custom_service_type_context`
  - `test_run_distributed_ps_register_timeout_includes_custom_service_type_context`
- Tightened default register-timeout assertions in blocking register regressions
  to match enriched default context:
  - worker: `register worker-0 as worker ...`
  - ps: `register ps-0 as ps ...`
- Confirms both worker and ps top-level distributed registration timeout paths
  preserve custom service-type context while still enforcing bounded timeout
  behavior and cleanup attempts.

### 155) Connect-timeout diagnostic enrichment with service-type context
- Enriched connect-timeout operation diagnostics to include service-type context:
  - `connect <service_id> via <service_type>`
- Expanded integration coverage across config entrypoints:
  - run-config worker custom service-type connect timeout regression
  - run-config ps custom service-type connect timeout regression
  - runner-config worker custom service-type connect timeout regression
  - runner-config ps custom service-type connect timeout regression
- Expanded direct distributed-runner unit coverage:
  - `test_run_distributed_worker_connect_timeout_includes_custom_service_type_context`
  - `test_run_distributed_ps_connect_timeout_includes_custom_service_type_context`
- Updated existing connect-timeout assertions (run-config, runner-config, and
  direct distributed runner tests) to assert enriched default context:
  - worker: `connect worker-<index> via worker ...`
  - ps: `connect ps-<index> via ps ...`

### 156) Cleanup-timeout diagnostic enrichment with service-type context
- Refactored discovery cleanup timeout diagnostics to support full cleanup
  operation descriptors.
- Cleanup timeout diagnostics now include service-type context:
  - `deregister <service_id> from <service_type>`
  - `disconnect <service_id> via <service_type>`
- Added focused cleanup regressions for custom worker service type after
  successful worker role completion:
  - `test_run_distributed_surfaces_deregister_timeout_with_custom_service_type_after_success`
  - `test_run_distributed_surfaces_disconnect_timeout_with_custom_service_type_after_success`
- Updated existing cleanup-timeout assertions to require enriched default
  context (`from worker` / `via worker`) in post-success cleanup timeout paths.

### 157) Non-empty discovery service-type validation across CLI and runtime configs
- Added distributed config validation for non-empty discovery service-type
  fields:
  - `discovery_service_type_ps`
  - `discovery_service_type_worker`
- Added train CLI distributed-config builder validation for non-empty
  service-type flags:
  - `--discovery-service-type-ps`
  - `--discovery-service-type-worker`
- Added validation regressions:
  - CLI:
    - `test_build_distributed_run_config_rejects_empty_ps_service_type`
    - `test_build_distributed_run_config_rejects_empty_worker_service_type`
  - distributed config unit:
    - `test_distributed_config_validate_rejects_empty_ps_service_type`
    - `test_distributed_config_validate_rejects_empty_worker_service_type`
  - run/runner config integration:
    - `distributed_runner_from_run_config_rejects_empty_ps_service_type`
    - `distributed_runner_from_run_config_rejects_empty_worker_service_type`
    - `distributed_runner_from_runner_config_rejects_empty_ps_service_type`
    - `distributed_runner_from_runner_config_rejects_empty_worker_service_type`
- Ensures malformed empty discovery service-type configuration is rejected
  consistently across CLI assembly and both runtime config entrypoint layers.

### 158) Parameter-sync interval validation when replication targets are configured
- Added distributed config validation to reject zero parameter-sync interval
  when parameter-sync replication targets are configured:
  - `parameter_sync_targets` non-empty requires `parameter_sync_interval > 0`
- Added CLI distributed-config builder validation:
  - rejects `--parameter-sync-interval-ms=0` when any
    `--parameter-sync-target` is provided.
- Added regression coverage:
  - CLI:
    - `test_build_distributed_run_config_rejects_zero_parameter_sync_interval_with_targets`
  - distributed config unit:
    - `test_distributed_config_validate_rejects_zero_parameter_sync_interval_when_targets_configured`
- Prevents runtime panic risk from invalid zero replication interval in
  parameter-sync task setup paths.

### 159) Env-utils test isolation against ambient ZK auth environment leakage
- Hardened `native_training::env_utils` tests with serialized env mutation and
  automatic env snapshot/restore to prevent cross-test/process environment
  leakage from `ZK_AUTH`.
- Added explicit trim-empty regression:
  - `test_get_zk_auth_data_empty_after_trim_is_none`
- Verified full workspace remains green even when `ZK_AUTH` is explicitly set
  in the parent environment (`ZK_AUTH=user:pass`), eliminating prior
  environment-sensitive failure mode for
  `test_get_zk_auth_data_none`.

### 160) Parameter-sync target metadata validation hardening
- Expanded parameter-sync validation when replication targets are configured:
  - reject empty/whitespace `parameter_sync_targets` entries,
  - reject empty/whitespace `parameter_sync_model_name`,
  - reject empty/whitespace `parameter_sync_signature_name`.
- Added CLI distributed-config builder validation counterparts:
  - `--parameter-sync-target` entries must be non-empty,
  - `--parameter-sync-model-name` must be non-empty when targets are set,
  - `--parameter-sync-signature-name` must be non-empty when targets are set.
- Added regression coverage:
  - CLI:
    - `test_build_distributed_run_config_rejects_empty_parameter_sync_target_entry`
    - `test_build_distributed_run_config_rejects_empty_parameter_sync_model_name_with_targets`
    - `test_build_distributed_run_config_rejects_empty_parameter_sync_signature_name_with_targets`
  - distributed config unit:
    - `test_distributed_config_validate_rejects_empty_parameter_sync_target_entry`
    - `test_distributed_config_validate_rejects_empty_parameter_sync_model_name_when_targets_configured`
    - `test_distributed_config_validate_rejects_empty_parameter_sync_signature_when_targets_configured`
- Prevents malformed parameter-sync metadata from reaching runtime replication
  loops and failing late at push/connect boundaries.

### 161) Run/runner config parity wiring for parameter-sync replication fields
- Extended `RunConfig` and `RunnerConfig` with explicit parameter-sync
  replication fields:
  - `parameter_sync_targets`
  - `parameter_sync_interval_ms`
  - `parameter_sync_model_name`
  - `parameter_sync_signature_name`
- Updated run-config merge and user-override extraction to preserve/emit these
  fields using existing Python-parity merge semantics.
- Wired `distributed_config_from_runner` to propagate all parameter-sync
  replication fields into `DistributedRunConfig`, closing a runtime-config gap
  where these values previously always fell back to distributed defaults.
- Added/updated regression coverage:
  - run_config unit:
    - explicit merge assertions for all new fields
    - `user_overrides` assertions for all new fields
  - distributed config mapping unit:
    - `test_distributed_config_from_runner_maps_fields` now validates complete
      parameter-sync propagation
  - native-training parity integration:
    - reintroduced run/runner-config parameter-sync validation regressions now
      that fields are first-class:
      - zero interval with targets is rejected
      - empty target entries are rejected
      - empty model/signature names with targets are rejected
- Ensures parameter-sync replication settings are configurable and validated
  consistently across all distributed runtime entrypoint surfaces.

### 162) Run/runner config cleanup-timeout diagnostics parity after successful worker runs
- Added new integration-only discovery backend and helper PS server bootstrap to
  exercise **successful worker execution + blocked cleanup** scenarios via
  run/runner-config entrypoints.
- Added four parity regressions validating that post-success cleanup timeout
  diagnostics preserve configured custom worker service-type context:
  - run-config path:
    - `distributed_runner_from_run_config_surfaces_deregister_timeout_with_custom_service_type_after_success`
    - `distributed_runner_from_run_config_surfaces_disconnect_timeout_with_custom_service_type_after_success`
  - runner-config path:
    - `distributed_runner_from_runner_config_surfaces_deregister_timeout_with_custom_service_type_after_success`
    - `distributed_runner_from_runner_config_surfaces_disconnect_timeout_with_custom_service_type_after_success`
- These tests specifically prove parity for cleanup-timeout diagnostic strings:
  - `deregister worker-0 from <custom_worker_service_type>`
  - `disconnect worker-0 via <custom_worker_service_type>`
  after an otherwise successful distributed worker lifecycle.
- Closes a run/runner-config entrypoint parity gap where this custom service-type
  cleanup diagnostic behavior was only covered at lower-level distributed-runner
  unit test surfaces.

### 163) Run/runner config parity for worker-discovery error precedence under blocked cleanup
- Added integration-only discovery backend coverage for:
  - empty PS discovery results (worker timeout path),
  - blocked `deregister_async` and `disconnect` cleanup operations.
- Added run-config + runner-config parity regressions ensuring worker
  discovery-timeout errors remain primary when both cleanup steps time out:
  - `distributed_runner_from_run_config_preserves_worker_discovery_error_when_cleanup_times_out`
  - `distributed_runner_from_runner_config_preserves_worker_discovery_error_when_cleanup_times_out`
- Added elapsed-time bounds proving configured cleanup timeouts from run/runner
  config entrypoints are honored even when both cleanup phases block.
- Confirms parity for error-precedence semantics already validated at lower-level
  distributed-runner tests, now explicitly exercised through both
  high-level config entrypoint APIs.

### 164) Worker discovery-timeout diagnostics now include discovery service-type context
- Enhanced worker fallback discovery-timeout diagnostics (empty/insufficient PS
  results across retries) to include queried PS discovery service type:
  - `Timed out waiting for PS discovery (service type: <...>): ...`
- Added/updated regression coverage:
  - distributed runner unit:
    - extended `test_run_distributed_preserves_worker_error_when_cleanup_steps_timeout`
      to assert default service-type context (`ps`)
    - added
      `test_run_distributed_preserves_worker_error_with_custom_discovery_service_type_when_cleanup_steps_timeout`
      for custom service-type context propagation
  - native-training integration parity:
    - strengthened existing run/runner blocked-cleanup precedence tests to
      assert default service-type context
    - added custom service-type propagation parity regressions:
      - `distributed_runner_from_run_config_propagates_custom_discover_service_type_into_worker_discovery_error_when_cleanup_times_out`
      - `distributed_runner_from_runner_config_propagates_custom_discover_service_type_into_worker_discovery_error_when_cleanup_times_out`
- Improves timeout diagnosability while preserving existing error-precedence
  semantics and cleanup-timeout behavior.

### 165) Distributed role-index range validation parity across CLI/runtime entrypoints
- Added distributed runtime config validation for role-specific index bounds:
  - PS role requires `index < num_ps`
  - worker role requires `index < num_workers`
- Added CLI distributed-config builder validation counterparts:
  - reject `--num-ps=0`
  - reject `--num-workers-cluster=0`
  - reject `--index >= --num-ps` when `--role ps`
  - reject `--index >= --num-workers-cluster` when `--role worker`
- Added regression coverage across all layers:
  - CLI unit tests for all new numeric/range guards
  - distributed config unit tests for role-based index range checks
  - native-training integration parity tests for run-config and runner-config
    entrypoints rejecting out-of-range PS/worker indices
- Ensures invalid role/index topology is rejected deterministically before
  discovery/register loops, aligning diagnostics and lifecycle behavior across
  high-level and low-level distributed runtime surfaces.

### 166) Non-empty distributed table-name validation parity across CLI/runtime configs
- Added distributed runtime validation requiring non-empty/trimmed `table_name`
  in `DistributedRunConfig::validate`.
- Added CLI distributed-config builder counterpart requiring non-empty
  `--table-name` in distributed mode.
- Added regression coverage:
  - CLI unit:
    - `test_build_distributed_run_config_rejects_empty_table_name`
  - distributed config unit:
    - `test_distributed_config_validate_rejects_empty_table_name`
  - native-training integration parity:
    - `distributed_runner_from_run_config_rejects_empty_table_name`
    - `distributed_runner_from_runner_config_rejects_empty_table_name`
- Prevents malformed empty table names from reaching PS lookup/apply paths and
  failing later in distributed worker execution loops.

### 167) Strict zero-count propagation and validation for distributed cluster sizing
- Removed silent coercion in `distributed_config_from_runner` that previously
  transformed:
  - `num_ps = 0` → `1`
  - `num_workers = 0` → `1`
- This now preserves configured values and allows shared distributed validation
  to reject invalid cluster sizing deterministically.
- Added/expanded regression coverage:
  - distributed config unit:
    - `test_distributed_config_validate_rejects_zero_num_ps`
    - `test_distributed_config_validate_rejects_zero_num_workers`
  - native-training integration parity:
    - `distributed_runner_from_run_config_rejects_zero_num_ps`
    - `distributed_runner_from_run_config_rejects_zero_num_workers`
    - `distributed_runner_from_runner_config_rejects_zero_num_ps`
    - `distributed_runner_from_runner_config_rejects_zero_num_workers`
- Ensures run/runner-config entrypoints no longer mask malformed topology and
  now align with CLI-level strict cluster-size validation semantics.

### 168) Distinct PS/worker discovery service-type validation parity
- Added distributed runtime validation requiring
  `discovery_service_type_ps != discovery_service_type_worker` (trim-aware).
- Added CLI distributed-config builder counterpart requiring:
  - `--discovery-service-type-ps` and
  - `--discovery-service-type-worker`
  to be distinct in distributed mode.
- Added regression coverage:
  - CLI unit:
    - `test_build_distributed_run_config_rejects_identical_ps_and_worker_service_types`
  - distributed config unit:
    - `test_distributed_config_validate_rejects_identical_ps_and_worker_service_types`
  - native-training integration parity:
    - `distributed_runner_from_run_config_rejects_identical_ps_and_worker_service_types`
    - `distributed_runner_from_runner_config_rejects_identical_ps_and_worker_service_types`
- Prevents ambiguous cluster advertisement/discovery where PS and worker entries
  share the same service type and can pollute worker PS discovery resolution.

### 169) Whitespace-normalized discovery service-type validation parity
- Added distributed runtime validation to reject discovery service types with
  leading/trailing whitespace:
  - `discovery_service_type_ps`
  - `discovery_service_type_worker`
- Added CLI distributed-config builder parity checks to reject:
  - `--discovery-service-type-ps` with leading/trailing whitespace
  - `--discovery-service-type-worker` with leading/trailing whitespace
- Added regression coverage:
  - CLI unit:
    - `test_build_distributed_run_config_rejects_whitespace_padded_ps_service_type`
    - `test_build_distributed_run_config_rejects_whitespace_padded_worker_service_type`
  - distributed config unit:
    - `test_distributed_config_validate_rejects_whitespace_padded_ps_service_type`
    - `test_distributed_config_validate_rejects_whitespace_padded_worker_service_type`
  - native-training integration parity:
    - `distributed_runner_from_run_config_rejects_whitespace_padded_ps_service_type`
    - `distributed_runner_from_run_config_rejects_whitespace_padded_worker_service_type`
    - `distributed_runner_from_runner_config_rejects_whitespace_padded_ps_service_type`
    - `distributed_runner_from_runner_config_rejects_whitespace_padded_worker_service_type`
- Prevents subtle discovery mismatches caused by whitespace-tainted service-type
  keys while preserving explicit, actionable validation diagnostics.

### 170) Whitespace-normalized distributed table-name validation parity
- Added distributed runtime validation to reject table names with
  leading/trailing whitespace:
  - `table_name.trim() != table_name` is now rejected.
- Added CLI distributed-config builder parity check rejecting
  whitespace-padded `--table-name`.
- Added regression coverage:
  - CLI unit:
    - `test_build_distributed_run_config_rejects_whitespace_padded_table_name`
  - distributed config unit:
    - `test_distributed_config_validate_rejects_whitespace_padded_table_name`
  - native-training integration parity:
    - `distributed_runner_from_run_config_rejects_whitespace_padded_table_name`
    - `distributed_runner_from_runner_config_rejects_whitespace_padded_table_name`
- Prevents subtle table-name key mismatches between lookup/apply calls and PS
  table initialization caused by accidental whitespace in distributed configs.

### 171) Whitespace-normalized parameter-sync metadata validation parity
- Expanded distributed runtime validation (when parameter-sync targets are
  configured) to reject leading/trailing whitespace in:
  - `parameter_sync_targets` entries
  - `parameter_sync_model_name`
  - `parameter_sync_signature_name`
- Added CLI distributed-config builder parity checks rejecting the same
  whitespace-padded parameter-sync inputs.
- Added regression coverage:
  - CLI unit:
    - `test_build_distributed_run_config_rejects_whitespace_padded_parameter_sync_target_entry`
    - `test_build_distributed_run_config_rejects_whitespace_padded_parameter_sync_model_name_with_targets`
    - `test_build_distributed_run_config_rejects_whitespace_padded_parameter_sync_signature_name_with_targets`
  - distributed config unit:
    - `test_distributed_config_validate_rejects_whitespace_padded_parameter_sync_target_entry`
    - `test_distributed_config_validate_rejects_whitespace_padded_parameter_sync_model_name_when_targets_configured`
    - `test_distributed_config_validate_rejects_whitespace_padded_parameter_sync_signature_when_targets_configured`
  - native-training integration parity:
    - run-config + runner-config regressions for padded target/model/signature
      rejection through distributed runner entrypoints.
- Prevents subtle replication endpoint/model signature mismatches caused by
  whitespace-tainted metadata while preserving early, actionable diagnostics.

### 172) Unique parameter-sync target validation parity across CLI/runtime entrypoints
- Added distributed runtime validation requiring unique
  `parameter_sync_targets` entries.
- Added CLI distributed-config builder counterpart requiring unique
  `--parameter-sync-target` entries.
- Added regression coverage:
  - CLI unit:
    - `test_build_distributed_run_config_rejects_duplicate_parameter_sync_target_entry`
  - distributed config unit:
    - `test_distributed_config_validate_rejects_duplicate_parameter_sync_target_entries`
  - native-training integration parity:
    - `distributed_runner_from_run_config_rejects_duplicate_parameter_sync_target_entry`
    - `distributed_runner_from_runner_config_rejects_duplicate_parameter_sync_target_entry`
- Prevents duplicate online push fanout caused by repeated target endpoints and
  guarantees deterministic one-target-one-push replication intent.

### 173) Worker discovery-timeout diagnostics now include worker service-id context
- Enhanced worker fallback discovery-timeout diagnostics to include worker
  service id (`worker-{index}`) alongside discovery service-type context:
  - `Timed out waiting for PS discovery for worker-<index> (service type: <...>): ...`
- Updated distributed runner tests to assert service-id context in worker
  discovery-timeout paths (default/custom discovery service types and blocked
  cleanup variants).
- Expanded native-training integration parity coverage:
  - strengthened existing run/runner worker-discovery timeout regressions to
    assert `for worker-0` context,
  - added explicit worker-index propagation regressions:
    - `distributed_runner_from_run_config_propagates_worker_index_into_ps_discovery_timeout_diagnostics`
    - `distributed_runner_from_runner_config_propagates_worker_index_into_ps_discovery_timeout_diagnostics`
- Improves timeout diagnosability by tying fallback discovery failures directly
  to the failing worker identity while preserving existing error-precedence
  semantics.

### 174) Parameter-sync target uniqueness now normalizes optional `http://` prefix
- Tightened distributed config/CLI parameter-sync target uniqueness semantics to
  treat these as equivalent duplicates:
  - `127.0.0.1:8500`
  - `http://127.0.0.1:8500`
- Applied canonicalization in both entrypoints:
  - runtime distributed validation (`DistributedRunConfig::validate`)
  - CLI config builder validation (`TrainCommand::build_distributed_run_config`)
- Preserved existing uniqueness error surfaces while preventing duplicate
  replication fanout caused by mixed explicit-vs-implicit HTTP URL notation.
- Added regression coverage at all layers:
  - runner unit test: rejects duplicates after `http://` normalization
  - CLI unit test: rejects `--parameter-sync-target` duplicates after
    `http://` normalization
  - native-training integration tests (RunConfig + RunnerConfig): reject
    normalized duplicate targets before runtime execution.

### 175) Strict parameter-sync endpoint format validation at CLI/runtime boundaries
- Added explicit endpoint syntax validation for every parameter-sync target in:
  - distributed runtime validation (`DistributedRunConfig::validate`)
  - CLI distributed config builder (`TrainCommand::build_distributed_run_config`)
- Validation now canonicalizes implicit host:port targets to `http://...` and
  then checks URI validity through `tonic::transport::Endpoint::from_shared`,
  surfacing deterministic config errors before runtime replication starts.
- Rejects malformed entries like `http://` early with clear diagnostics:
  - runtime: `distributed config has invalid parameter_sync_targets entry ...`
  - CLI: `--parameter-sync-target contains invalid endpoint ...`
- Added parity coverage:
  - runner unit test for invalid parameter-sync endpoint entry
  - CLI unit test for invalid `--parameter-sync-target` endpoint
  - integration tests (RunConfig + RunnerConfig) verifying malformed endpoint
    rejection through distributed runner entrypoints.

### 176) Discovery service-type distinctness normalized case-insensitively
- Hardened distributed config validation to reject PS/worker discovery service
  types that differ only by ASCII letter case after trimming (e.g.
  `Service` vs `service`).
- Applied consistently in both entrypoint layers:
  - runtime distributed validation (`DistributedRunConfig::validate`)
  - CLI distributed config builder (`TrainCommand::build_distributed_run_config`)
- Prevents ambiguous discovery namespaces and accidental cross-role service
  collisions caused by case-variant configuration drift.
- Added regression coverage:
  - runner unit: rejects case-insensitive identical discovery service types
  - CLI unit: rejects case-insensitive identical service-type flags
  - native-training integration (RunConfig + RunnerConfig): rejects
    case-insensitive identical PS/worker service type pairs.

### 177) Parameter-sync target uniqueness now normalizes case-variant `http://` prefixes and host casing
- Strengthened duplicate-target detection to avoid case-based bypasses:
  - `EXAMPLE.com:8500`
  - `HtTp://example.COM:8500`
  are now treated as the same canonical target.
- Applied in both validation entrypoints:
  - runtime distributed validation (`DistributedRunConfig::validate`)
  - CLI distributed config builder (`TrainCommand::build_distributed_run_config`)
- Canonicalization now:
  - strips `http://` prefix case-insensitively,
  - compares normalized targets case-insensitively.
- Added regression coverage:
  - runner unit: rejects duplicates after case-insensitive HTTP-prefix/host normalization
  - CLI unit: rejects duplicate `--parameter-sync-target` entries with case variants
  - native-training integration (RunConfig + RunnerConfig): rejects normalized
    case-variant duplicate parameter-sync targets.

### 178) Parameter-sync endpoint parsing now recognizes case-variant HTTP/HTTPS schemes before canonicalization
- Hardened endpoint-preparation path for parameter-sync targets to detect
  `http://` and `https://` schemes case-insensitively before deciding whether to
  auto-prefix `http://`.
- Removes ambiguity where mixed-case scheme input (e.g. `HtTp://...`) could be
  double-prefixed before validation, causing parser-dependent behavior.
- Applied in both validation entrypoints:
  - runtime distributed validation (`DistributedRunConfig::validate`)
  - CLI distributed config builder (`TrainCommand::build_distributed_run_config`)
- Added regression coverage:
  - runner unit: accepts case-insensitive HTTP scheme targets
  - CLI unit: accepts case-insensitive HTTP scheme targets
  - existing duplicate normalization regressions continue validating
    case-insensitive scheme/host duplicate rejection semantics.

### 179) Integration entrypoints now assert case-insensitive parameter-sync scheme acceptance path reaches runtime discovery semantics
- Extended native-training integration parity suite to verify that mixed-case
  `http://` parameter-sync targets are accepted by both entrypoint layers
  (`RunConfig` and `RunnerConfig`) and do not fail early in distributed-config
  validation.
- New integration regressions assert the resulting runtime path reaches worker
  PS discovery timeout handling (proof that validation accepted endpoint syntax):
  - `distributed_runner_from_run_config_accepts_case_insensitive_http_scheme_parameter_sync_target`
  - `distributed_runner_from_runner_config_accepts_case_insensitive_http_scheme_parameter_sync_target`
- Locks in end-to-end parity for case-insensitive scheme handling across config
  normalization + runtime entrypoint wiring.

### 180) Parameter-sync endpoints now enforce authority-only URL shape and trailing-slash-normalized uniqueness
- Tightened parameter-sync target normalization in CLI/runtime validation:
  - endpoints must not include URL path/query payloads,
  - uniqueness canonicalizes by normalized `scheme://authority` form.
- Practical parity effects:
  - rejects malformed/ambiguous endpoints like
    `http://127.0.0.1:8500/v1?foo=bar`,
  - treats `127.0.0.1:8500` and `http://127.0.0.1:8500/` as duplicates.
- Applied across both entrypoint layers:
  - runtime distributed validation (`DistributedRunConfig::validate`)
  - CLI distributed config builder (`TrainCommand::build_distributed_run_config`)
- Regression expansion:
  - runner unit: reject path/query endpoints; reject duplicates after
    trailing-slash normalization
  - CLI unit: reject path/query endpoints; reject duplicates after trailing-slash normalization
  - integration tests (RunConfig + RunnerConfig): reject path/query endpoints
    and trailing-slash-normalized duplicate targets.

### 181) Parameter-sync endpoint schemes now explicitly constrained to HTTP(S)
- Added explicit scheme guard to reject non-HTTP transports in
  `parameter_sync_targets` before endpoint canonicalization:
  - accepted: `http://...`, `https://...`, and bare `host:port` (auto-prefixed to http)
  - rejected: e.g. `ftp://...`
- Applied consistently across entrypoints:
  - runtime distributed validation (`DistributedRunConfig::validate`)
  - CLI distributed config builder (`TrainCommand::build_distributed_run_config`)
- Expanded regression coverage at all layers:
  - runner unit + CLI unit: unsupported-scheme rejection
  - native-training integration (RunConfig + RunnerConfig): unsupported-scheme
    rejection through distributed entrypoint paths.

### 182) Parameter-sync endpoint canonicalization now rejects userinfo and normalizes default ports for uniqueness
- Hardened endpoint canonicalization rules in both validation entrypoints:
  - reject endpoints containing userinfo (e.g. `http://user@host:port`),
  - normalize implicit default ports for duplicate detection:
    - `http://host` == `http://host:80`
    - `https://host` == `https://host:443`.
- Applied in:
  - runtime distributed validation (`DistributedRunConfig::validate`)
  - CLI distributed config builder (`TrainCommand::build_distributed_run_config`)
- Added comprehensive regressions:
  - runner + CLI unit tests:
    - userinfo rejection
    - HTTP default-port duplicate normalization
    - HTTPS default-port duplicate normalization
  - native-training integration tests (RunConfig + RunnerConfig):
    - userinfo rejection
    - HTTP default-port duplicate normalization rejection.

### 183) Integration parity expanded for HTTPS default-port duplicate normalization
- Added native-training integration regressions proving default-port
  canonicalization parity for HTTPS endpoints through both distributed entrypoints:
  - `distributed_runner_from_run_config_rejects_duplicate_parameter_sync_target_entry_after_https_default_port_normalization`
  - `distributed_runner_from_runner_config_rejects_duplicate_parameter_sync_target_entry_after_https_default_port_normalization`
- Ensures `https://host` and `https://host:443` are treated as duplicates
  end-to-end (not just in unit-level validation coverage).

### 184) Strict internal-whitespace normalization across distributed service/table/sync-name identifiers
- Added internal-whitespace rejection (not only trim checks) for distributed
  identity fields:
  - `discovery_service_type_ps`
  - `discovery_service_type_worker`
  - `table_name`
  - `parameter_sync_model_name` (when parameter-sync targets configured)
  - `parameter_sync_signature_name` (when parameter-sync targets configured)
- Applied consistently at both entrypoint layers:
  - runtime distributed validation (`DistributedRunConfig::validate`)
  - CLI distributed config builder (`TrainCommand::build_distributed_run_config`)
- Expanded parity regressions comprehensively:
  - runner unit tests for each new internal-whitespace constraint,
  - CLI unit tests for matching flag-level constraints,
  - native-training integration tests for both RunConfig and RunnerConfig
    entrypoints, covering service-type/table-name/model/signature internal-whitespace rejection.

### 185) Role-error returns now include cleanup-failure diagnostics without losing primary failure semantics
- Improved distributed runner error reporting for role failures with blocked or
  failing cleanup steps:
  - preserves the primary role error text (e.g. worker PS-discovery timeout,
    register timeout),
  - appends cleanup issue diagnostics in the same returned error string:
    `discovery cleanup encountered issues after role error: ...`.
- Ensures callers can observe cleanup failures directly from returned errors
  while retaining deterministic primary-error precedence.
- Regression updates:
  - runner unit tests now assert appended cleanup diagnostics in worker discovery
    timeout + blocked cleanup scenarios.
  - native-training integration tests (RunConfig + RunnerConfig) now assert
    cleanup issue context presence and cleanup timeout operation diagnostics in
    worker discovery timeout + blocked cleanup flows.

### 186) Register-timeout parity now verifies appended cleanup diagnostics in both RunConfig and RunnerConfig paths
- Extended integration parity coverage for register-timeout-with-blocked-cleanup
  scenarios across worker and PS roles:
  - default + custom worker service types
  - default + custom PS service types
- New assertions verify returned errors include both:
  - primary register timeout diagnostics (error precedence preserved),
  - appended cleanup issue context with concrete cleanup operation timeout
    messages (`deregister ...`, `disconnect ...`).
- Completes end-to-end diagnostic parity for role error paths where cleanup also
  fails or times out, ensuring consistency between unit-level runner behavior
  and run/runner configuration entrypoints.

### 187) Connect-error parity now appends cleanup diagnostics when disconnect cleanup fails or times out
- Hardened `run_distributed` connect-failure handling:
  - when `connect` fails/timeouts and subsequent `disconnect` cleanup also
    fails/timeouts, return value now preserves primary connect error while
    appending cleanup operation diagnostics in the same error message.
- Expanded parity tests:
  - runner unit tests now assert appended cleanup context + disconnect operation
    diagnostics for:
    - connect timeout + disconnect cleanup timeout,
    - connect failure + disconnect cleanup timeout,
    - connect failure + disconnect cleanup failure.
  - native-training integration tests (RunConfig + RunnerConfig) now assert
    connect-timeout paths include cleanup issue context and disconnect timeout
    diagnostics across worker/ps roles, index propagation variants, custom
    discovery service types, and cleanup-timeout-bound paths.
- Result: connect/setup lifecycle now matches role-error lifecycle semantics for
  cleanup diagnostics, preserving deterministic error precedence while exposing
  actionable cleanup failure context.

### 188) Post-success cleanup parity now preserves deregister failure precedence while appending disconnect diagnostics
- Hardened `run_distributed` successful-role cleanup path:
  - when both cleanup steps fail (`deregister` + `disconnect`), returned error
    now preserves primary deregister failure while appending disconnect cleanup
    diagnostics in the same message.
- Expanded parity tests:
  - runner unit regression now asserts combined cleanup context + disconnect
    failure diagnostics when both post-success cleanup steps fail.
  - native-training integration regressions (RunConfig + RunnerConfig) now
    assert successful worker run + both blocked cleanup steps produce:
    - primary deregister timeout diagnostics,
    - appended cleanup issue context for successful-role completion,
    - appended disconnect timeout diagnostics with configured cleanup timeout.
- Result: setup, role-error, and post-success cleanup branches now all surface
  rich cleanup diagnostics with deterministic primary-error precedence.

### 189) Connect-failure integration parity now covers disconnect-failure (non-timeout) cleanup paths and restores missing native parity test doubles
- Added missing native-training integration discovery doubles for connect-failure
  cleanup scenarios:
  - `FailingConnectWithHangingDisconnectFromConfigDiscovery`
  - `FailingConnectWithFailingDisconnectFromConfigDiscovery`
- Expanded RunConfig + RunnerConfig integration parity with new regressions:
  - `distributed_runner_from_run_config_preserves_connect_failure_with_disconnect_failure_context`
  - `distributed_runner_from_runner_config_preserves_connect_failure_with_disconnect_failure_context`
- New assertions verify connect failure remains primary while returned errors
  append cleanup issue context with concrete disconnect-failure diagnostics and
  custom worker discovery-service-type operation context.
- Added direct `native_training_parity` binary run in validation lane to ensure
  integration test binary compilation/execution remains green with expanded
  connect-failure coverage.

### 190) Connect-failure cleanup-timeout diagnostics now verify custom worker service-type propagation in RunConfig/RunnerConfig paths
- Tightened existing connect-failure + blocked-disconnect integration assertions
  to validate custom worker discovery-service-type propagation in cleanup
  timeout operation diagnostics.
- Updated tests:
  - `distributed_runner_from_run_config_preserves_connect_failure_with_cleanup_timeout_context`
  - `distributed_runner_from_runner_config_preserves_connect_failure_with_cleanup_timeout_context`
- Both now configure `discovery_service_type_worker = "trainer_custom"` and
  assert timeout diagnostics include:
  - `disconnect worker-0 via trainer_custom after 20ms`
- This closes the remaining service-type-context parity gap for connect-failure
  cleanup-timeout diagnostics in run/runner configuration entrypoints.

### 191) PS-role connect-failure cleanup diagnostics parity expanded across timeout and failure cleanup modes
- Added PS-role integration parity regressions for RunConfig + RunnerConfig
  connect-failure branches with custom PS discovery service type:
  - `distributed_runner_from_run_config_preserves_ps_connect_failure_with_cleanup_timeout_context`
  - `distributed_runner_from_run_config_preserves_ps_connect_failure_with_disconnect_failure_context`
  - `distributed_runner_from_runner_config_preserves_ps_connect_failure_with_cleanup_timeout_context`
  - `distributed_runner_from_runner_config_preserves_ps_connect_failure_with_disconnect_failure_context`
- New assertions verify:
  - primary connect failure precedence is preserved for PS role,
  - cleanup issue context is appended when disconnect cleanup times out/fails,
  - disconnect operation diagnostics include custom PS service-type context
    (`parameter_server_custom`) in both timeout and failure modes.
- Result: connect-failure cleanup diagnostics now have symmetric worker/PS role
  integration coverage for run/runner entrypoints.

### 192) Successful-role disconnect-only cleanup failures now include explicit cleanup issue context
- Hardened `run_distributed` post-success cleanup tail path:
  - when `deregister` succeeds but `disconnect` fails/timeouts, returned error
    now appends explicit successful-role cleanup issue context instead of
    surfacing raw disconnect cleanup error alone.
- Expanded runner unit assertions for successful worker runs:
  - disconnect failure after success now asserts appended successful-role cleanup
    context,
  - disconnect timeout after success now asserts appended successful-role cleanup
    context,
  - custom worker service-type disconnect timeout after success now asserts
    appended successful-role cleanup context.
- Expanded native-training integration assertions (RunConfig + RunnerConfig)
  for successful worker run + custom disconnect timeout path to assert appended
  successful-role cleanup issue context.
- Result: all cleanup-failure result paths (connect-stage, role-error-stage,
  post-success-stage) now provide explicit cleanup-context framing while
  preserving primary error precedence.

### 193) Successful-role deregister-only cleanup failures now include explicit cleanup issue context
- Hardened `run_distributed` post-success cleanup branch for single-step
  deregister failures/timeouts:
  - when `deregister` fails/timeouts and `disconnect` succeeds, returned error
    now appends explicit successful-role cleanup issue context.
- Expanded runner unit assertions to require cleanup context for:
  - deregister failure after successful worker run,
  - deregister timeout after successful worker run,
  - custom worker service-type deregister timeout after successful run.
- Expanded native-training integration assertions (RunConfig + RunnerConfig)
  for custom-service-type deregister-timeout-after-success paths to require
  appended successful-role cleanup issue context.
- Result: both single-step and multi-step post-success cleanup failures now
  consistently provide successful-role cleanup-context diagnostics.

### 194) Default-service-type post-success cleanup timeout parity expanded for RunConfig/RunnerConfig integration paths
- Added native-training integration regressions for successful worker run with
  **default** discovery service types and blocked post-success cleanup:
  - `distributed_runner_from_run_config_surfaces_deregister_timeout_after_success`
  - `distributed_runner_from_run_config_surfaces_disconnect_timeout_after_success`
  - `distributed_runner_from_runner_config_surfaces_deregister_timeout_after_success`
  - `distributed_runner_from_runner_config_surfaces_disconnect_timeout_after_success`
- New assertions verify for both run/runner entrypoints:
  - default-service-type cleanup timeout diagnostics (`worker`) remain primary,
  - appended successful-role cleanup issue context is present,
  - connect/register/discover/deregister/disconnect call counts remain
    deterministic and bounded.
- Result: post-success cleanup-context parity coverage now spans both custom and
  default discovery service-type configurations.

### 195) Default-service-type post-success cleanup failure parity expanded for RunConfig/RunnerConfig integration paths
- Added a failing post-success cleanup integration backend:
  - `FailingCleanupAfterSuccessFromConfigDiscovery`
  - configurable forced deregister/disconnect failures with deterministic call
    counters and successful worker→PS discovery path.
- Added new native-training integration regressions:
  - `distributed_runner_from_run_config_surfaces_deregister_failure_after_success`
  - `distributed_runner_from_run_config_surfaces_disconnect_failure_after_success`
  - `distributed_runner_from_runner_config_surfaces_deregister_failure_after_success`
  - `distributed_runner_from_runner_config_surfaces_disconnect_failure_after_success`
- New assertions verify for default service types:
  - forced deregister/disconnect failures are preserved as primary cleanup
    errors after successful worker completion,
  - successful-role cleanup issue context remains appended,
  - lifecycle call counts remain deterministic (connect/register/discover/
    deregister/disconnect exactly once).
- Result: post-success cleanup parity coverage now includes both timeout and
  non-timeout failure modes for default-service-type run/runner paths.

### 196) Post-success single-step cleanup failure diagnostics now include operation context; custom-service-type integration parity expanded
- Refined `run_distributed` post-success single-step cleanup error wrapping:
  - when only deregister fails, appended successful-role cleanup context now
    includes explicit deregister operation context.
  - when only disconnect fails, appended successful-role cleanup context now
    includes explicit disconnect operation context.
- Expanded runner unit assertions to require operation context for:
  - deregister failure after successful run (`deregister worker-0 from worker`),
  - disconnect failure after successful run (`disconnect worker-0 via worker`).
- Expanded native-training integration parity:
  - default-service-type post-success failure tests now assert cleanup operation
    context presence.
  - added custom-worker post-success failure tests (RunConfig + RunnerConfig):
    - deregister failure operation context (`deregister ... from trainer_custom`)
    - disconnect failure operation context (`disconnect ... via trainer_custom`)
  - all assert preserved primary failure + appended successful-role cleanup
    context + deterministic lifecycle call counts.
- Result: post-success cleanup failure diagnostics are now richer and symmetric
  across default/custom service-type integration paths.

### 197) PS connect-failure cleanup diagnostics parity expanded for default service type and runner unit coverage
- Added runner unit regressions for PS connect-failure paths with custom PS
  service type:
  - `test_run_distributed_ps_connect_failure_does_not_hang_when_disconnect_blocks_with_custom_service_type`
  - `test_run_distributed_returns_ps_connect_error_when_connect_and_disconnect_fail_with_custom_service_type`
- Expanded native-training integration parity for **default PS service type**
  connect-failure cleanup diagnostics (RunConfig + RunnerConfig):
  - timeout cleanup mode:
    - `distributed_runner_from_run_config_preserves_default_ps_connect_failure_with_cleanup_timeout_context`
    - `distributed_runner_from_runner_config_preserves_default_ps_connect_failure_with_cleanup_timeout_context`
  - failure cleanup mode:
    - `distributed_runner_from_run_config_preserves_default_ps_connect_failure_with_disconnect_failure_context`
    - `distributed_runner_from_runner_config_preserves_default_ps_connect_failure_with_disconnect_failure_context`
- New assertions verify:
  - primary connect failure precedence,
  - appended cleanup issue context,
  - default-service-type cleanup operation diagnostics (`disconnect ps-0 via ps`)
    for timeout and failure cleanup outcomes.
- Result: connect-failure cleanup diagnostics now have explicit default-service
  integration parity for PS role plus reinforced runner-level unit coverage.

### 198) Post-success both-cleanup-failure diagnostics now preserve explicit deregister operation context end-to-end
- Hardened `run_distributed` post-success **both-cleanup-failure** branch so
  returned errors now include explicit deregister operation context even when
  deregister remains the primary failure:
  - `deregister <service-id> from <service-type>: <deregister-error>`
  - plus appended disconnect operation diagnostics in cleanup issue context.
- Expanded runner unit regression
  `test_run_distributed_prefers_deregister_error_when_both_post_success_cleanup_steps_fail`
  to assert explicit deregister operation context in addition to disconnect
  failure context.
- Expanded native-training integration parity:
  - added default-service-type both-failure tests (RunConfig + RunnerConfig):
    - `distributed_runner_from_run_config_preserves_deregister_failure_with_disconnect_failure_context_after_success`
    - `distributed_runner_from_runner_config_preserves_deregister_failure_with_disconnect_failure_context_after_success`
  - added custom-worker both-failure tests (RunConfig + RunnerConfig):
    - `distributed_runner_from_run_config_surfaces_custom_worker_deregister_failure_after_success`
    - `distributed_runner_from_run_config_surfaces_custom_worker_disconnect_failure_after_success`
    - `distributed_runner_from_runner_config_surfaces_custom_worker_deregister_failure_after_success`
    - `distributed_runner_from_runner_config_surfaces_custom_worker_disconnect_failure_after_success`
  - enriched existing default-service-type single-step failure assertions to
    require cleanup operation context presence.
- Result: post-success cleanup failure diagnostics now consistently carry
  operation-level context for both primary and secondary cleanup failures across
  default/custom run/runner entrypoint paths.

### 199) Runner-level PS connect-timeout+cleanup-timeout parity reinforced with explicit custom-service diagnostics
- Added new runner unit regression:
  - `test_run_distributed_ps_connect_timeout_preserves_error_when_disconnect_cleanup_times_out_with_custom_service_type`
- Coverage guarantees:
  - primary connect-timeout remains dominant for PS role,
  - appended cleanup issue context is present,
  - cleanup timeout diagnostics include explicit custom PS service-type operation
    context (`disconnect ps-0 via parameter_server_custom`).
- Result: runner-level timeout/error-precedence semantics now mirror existing
  worker-path guarantees for PS custom service-type connect-timeout cleanup flow.

### 200) Discover-error role-failure cleanup diagnostics parity expanded with explicit last-discovery-error preservation
- Added new runner mock + regression coverage for **discover returns error + cleanup hangs**:
  - new mock: `WorkerDiscoverErrorWithHangingCleanupDiscovery`
  - new regression:
    - `test_run_distributed_preserves_worker_discover_failure_when_cleanup_steps_timeout`
- Strengthened existing worker-timeout cleanup regressions to also assert discover
  attempt accounting (`discover_count == 1`) in hanging-cleanup paths.
- Added new config-entrypoint integration mock + regressions:
  - new mock: `DiscoverErrorWithHangingCleanupFromConfigDiscovery`
  - `distributed_runner_from_run_config_preserves_last_discover_error_when_cleanup_times_out`
  - `distributed_runner_from_runner_config_preserves_last_discover_error_when_cleanup_times_out`
- New assertions verify:
  - worker-role timeout remains primary,
  - `last discovery error` preserves forced discover failure details,
  - role-error cleanup context is appended,
  - both cleanup timeout operations are surfaced with explicit operation context.
- Result: discover-error diagnostics now have explicit cleanup-timeout parity
  across runner unit and RunConfig/RunnerConfig integration entrypoints.

### 201) Discover-error cleanup-timeout parity extended for custom service-type + worker-index propagation
- Added runner-level regression:
  - `test_run_distributed_preserves_worker_discover_failure_with_custom_service_types_and_index_when_cleanup_steps_timeout`
- Added RunConfig + RunnerConfig integration regressions:
  - `distributed_runner_from_run_config_preserves_last_discover_error_with_custom_service_types_and_index_when_cleanup_times_out`
  - `distributed_runner_from_runner_config_preserves_last_discover_error_with_custom_service_types_and_index_when_cleanup_times_out`
- New assertions verify that discover-error cleanup-timeout diagnostics include:
  - custom PS discover service-type context,
  - propagated worker index in role-timeout diagnostics,
  - custom worker service-type context in deregister/disconnect cleanup timeout
    operation names (`trainer_custom`),
  - preserved `last discovery error` details.
- Result: discover-error cleanup-timeout semantics now have deterministic
  custom-service/index parity across runner, RunConfig, and RunnerConfig paths.

### 202) Discover-error + cleanup-failure (non-timeout) parity expanded across runner and config entrypoints
- Added runner-level failing-cleanup discover-error regression:
  - new mock: `WorkerDiscoverErrorWithFailingCleanupDiscovery`
  - new test:
    - `test_run_distributed_preserves_worker_discover_failure_when_cleanup_steps_fail`
- Added integration-level failing-cleanup discover-error regressions:
  - new mock: `DiscoverErrorWithFailingCleanupFromConfigDiscovery`
  - `distributed_runner_from_run_config_preserves_last_discover_error_when_cleanup_fails`
  - `distributed_runner_from_runner_config_preserves_last_discover_error_when_cleanup_fails`
- New assertions verify:
  - worker role timeout remains primary,
  - `last discovery error` preserves discover failure details,
  - role-error cleanup context is appended,
  - explicit deregister/disconnect cleanup **failure** operation diagnostics are
    included (not only timeout diagnostics).
- Result: role-error cleanup diagnostics now have stronger parity for both
  timeout-based and immediate-failure cleanup outcomes in discover-error paths.

### 203) Discover-error cleanup-failure parity extended for custom service-type + worker-index propagation
- Added runner regression:
  - `test_run_distributed_preserves_worker_discover_failure_with_custom_service_types_and_index_when_cleanup_steps_fail`
- Added RunConfig + RunnerConfig integration regressions:
  - `distributed_runner_from_run_config_preserves_last_discover_error_with_custom_service_types_and_index_when_cleanup_fails`
  - `distributed_runner_from_runner_config_preserves_last_discover_error_with_custom_service_types_and_index_when_cleanup_fails`
- New assertions verify discover-error cleanup-failure diagnostics include:
  - custom PS discover service type context (`parameter_server_custom`),
  - propagated worker index in role-timeout diagnostics,
  - custom worker service type in cleanup operation failures
    (`deregister/disconnect worker-<idx> ... trainer_custom`),
  - preserved last discovery error details and role-error cleanup issue context.
- Result: discover-error cleanup-failure semantics now have consistent
  custom-service/index parity across runner and config-driven distributed
  entrypoints.

### 204) Worker-timeout (empty-discovery) cleanup-failure parity expanded for custom service-type/index paths
- Added runner mock + regressions for **empty discovery timeout + cleanup failure**:
  - new mock: `WorkerTimeoutWithFailingCleanupDiscovery`
  - `test_run_distributed_preserves_worker_timeout_when_cleanup_steps_fail`
  - `test_run_distributed_preserves_worker_timeout_with_custom_service_types_and_index_when_cleanup_steps_fail`
- Added integration mock + regressions:
  - new mock: `EmptyDiscoverWithFailingCleanupFromConfigDiscovery`
  - `distributed_runner_from_run_config_preserves_worker_timeout_with_custom_service_types_and_index_when_cleanup_fails`
  - `distributed_runner_from_runner_config_preserves_worker_timeout_with_custom_service_types_and_index_when_cleanup_fails`
- New assertions verify:
  - worker timeout remains primary for empty-discovery role error paths,
  - cleanup issue context is appended for failing deregister/disconnect,
  - custom PS service type and worker index propagate into role-timeout diagnostics,
  - custom worker service type propagates into cleanup operation failure context.
- Result: worker-timeout role-error cleanup-failure diagnostics now have parity
  with discover-error paths across runner and config-driven entrypoints.

### 205) Worker-timeout cleanup-failure parity completed for default RunConfig/RunnerConfig paths
- Added default-service integration regressions:
  - `distributed_runner_from_run_config_preserves_worker_timeout_when_cleanup_fails`
  - `distributed_runner_from_runner_config_preserves_worker_timeout_when_cleanup_fails`
- These complement existing custom service/index coverage and verify default-path
  role-error cleanup-failure behavior end-to-end.
- New assertions verify:
  - primary worker-timeout diagnostics remain dominant,
  - default service-type diagnostics are preserved (`service type: ps`,
    `worker-0`),
  - cleanup issue context is appended with explicit default operation failures:
    - `deregister worker-0 from worker` + `forced deregister failure`
    - `disconnect worker-0 via worker` + `forced disconnect failure`.
- Result: worker-timeout cleanup-failure parity is now complete across
  default/custom service-type matrices for runner, RunConfig, and RunnerConfig
  entrypoints.

### 206) Register-failure role-error cleanup diagnostics now explicitly validated for default/custom service-type runner paths
- Strengthened existing runner registration-failure regressions:
  - `test_run_distributed_disconnects_when_worker_registration_fails`
  - `test_run_distributed_disconnects_when_ps_registration_fails`
  with explicit assertions for:
  - primary register failure preservation,
  - appended role-error cleanup issue context,
  - concrete deregister operation/failure diagnostics.
- Added new runner regressions for custom service-type registration-failure
  cleanup diagnostics:
  - `test_run_distributed_worker_registration_failure_includes_custom_service_type_cleanup_context`
  - `test_run_distributed_ps_registration_failure_includes_custom_service_type_cleanup_context`
- New assertions verify cleanup diagnostics include service-type-specific
  operation context (`trainer_custom`, `parameter_server_custom`) when register
  fails and deregister cleanup reports failure.
- Result: register-failure role-error cleanup diagnostics now have explicit
  default/custom runner-level parity guarantees, matching the broader discovery
  lifecycle error-context hardening strategy.

### 207) Config-entrypoint register-failure cleanup diagnostics parity expanded for custom service types
- Added RunConfig regressions:
  - `distributed_runner_from_run_config_preserves_worker_register_failure_with_custom_service_type_cleanup_context`
  - `distributed_runner_from_run_config_preserves_ps_register_failure_with_custom_service_type_cleanup_context`
- Added RunnerConfig regressions:
  - `distributed_runner_from_runner_config_preserves_worker_register_failure_with_custom_service_type_cleanup_context`
  - `distributed_runner_from_runner_config_preserves_ps_register_failure_with_custom_service_type_cleanup_context`
- Added shared integration mock:
  - `FailingRegisterWithFailingCleanupFromConfigDiscovery`
  simulating register failure plus failing deregister/disconnect cleanup.
- New assertions verify for both worker/ps custom service types:
  - register failure remains primary,
  - role-error cleanup issue context is appended,
  - cleanup operation diagnostics carry exact custom service-type context for
    deregister/disconnect failures.
- Result: register-failure cleanup diagnostics now have parity across runner and
  both config-driven distributed entrypoints for custom service-type paths.

### 208) Default-worker connect-failure cleanup diagnostics parity expanded across RunConfig/RunnerConfig entrypoints
- Added RunConfig regressions:
  - `distributed_runner_from_run_config_preserves_default_worker_connect_failure_with_cleanup_timeout_context`
  - `distributed_runner_from_run_config_preserves_default_worker_connect_failure_with_disconnect_failure_context`
- Added RunnerConfig regressions:
  - `distributed_runner_from_runner_config_preserves_default_worker_connect_failure_with_cleanup_timeout_context`
  - `distributed_runner_from_runner_config_preserves_default_worker_connect_failure_with_disconnect_failure_context`
- New assertions verify default worker service-type cleanup diagnostics:
  - primary connect failure remains dominant,
  - role-error cleanup issue context is appended,
  - disconnect cleanup diagnostics include explicit default operation context
    (`disconnect worker-0 via worker`) for both timeout and failure modes.
- Result: connect-failure cleanup diagnostics now have complete default/custom
  worker parity across run/runner distributed entrypoints (matching existing PS
  and runner-level unit coverage).

### 209) Default-service register-failure cleanup diagnostics parity expanded across RunConfig/RunnerConfig entrypoints
- Added RunConfig default-service regressions:
  - `distributed_runner_from_run_config_preserves_worker_register_failure_with_default_service_type_cleanup_context`
  - `distributed_runner_from_run_config_preserves_ps_register_failure_with_default_service_type_cleanup_context`
- Added RunnerConfig default-service regressions:
  - `distributed_runner_from_runner_config_preserves_worker_register_failure_with_default_service_type_cleanup_context`
  - `distributed_runner_from_runner_config_preserves_ps_register_failure_with_default_service_type_cleanup_context`
- Reused `FailingRegisterWithFailingCleanupFromConfigDiscovery` to simulate
  register failure plus failing deregister/disconnect cleanup operations.
- New assertions verify default-service diagnostics include:
  - primary register-failure precedence,
  - role-error cleanup issue context,
  - explicit default cleanup operation failure context (`worker` / `ps` service
    types for deregister/disconnect operations).
- Result: register-failure cleanup diagnostics now have complete default/custom
  service-type parity across runner, RunConfig, and RunnerConfig entrypoints.

### 210) Register-failure + cleanup-timeout parity expanded across runner/config entrypoints
- Added runner-level hanging-cleanup register-failure mock + regressions:
  - new mock: `FailingRegisterWithHangingCleanupDiscovery`
  - `test_run_distributed_preserves_worker_register_failure_when_cleanup_steps_timeout`
  - `test_run_distributed_preserves_ps_register_failure_with_custom_service_type_when_cleanup_steps_timeout`
- Added integration hanging-cleanup register-failure mock + regressions:
  - new mock: `FailingRegisterWithHangingCleanupFromConfigDiscovery`
  - RunConfig:
    - `distributed_runner_from_run_config_preserves_worker_register_failure_with_cleanup_timeout_context`
    - `distributed_runner_from_run_config_preserves_ps_register_failure_with_custom_service_type_cleanup_timeout_context`
  - RunnerConfig:
    - `distributed_runner_from_runner_config_preserves_worker_register_failure_with_cleanup_timeout_context`
    - `distributed_runner_from_runner_config_preserves_ps_register_failure_with_custom_service_type_cleanup_timeout_context`
- New assertions verify:
  - primary register failure remains dominant,
  - role-error cleanup context is appended,
  - cleanup timeout diagnostics include explicit deregister/disconnect operation
    context with correct default/custom service types.
- Result: register-failure cleanup diagnostics now cover both cleanup-failure and
  cleanup-timeout modes across runner, RunConfig, and RunnerConfig paths.

### 211) Register-failure + cleanup-timeout default/custom matrix completed across runner/config entrypoints
- Added runner-level cleanup-timeout regressions to complete missing service-type
  combinations:
  - `test_run_distributed_preserves_worker_register_failure_with_custom_service_type_when_cleanup_steps_timeout`
  - `test_run_distributed_preserves_ps_register_failure_when_cleanup_steps_timeout`
- Added RunConfig integration regressions for missing default/custom combinations:
  - `distributed_runner_from_run_config_preserves_worker_register_failure_with_custom_service_type_cleanup_timeout_context`
  - `distributed_runner_from_run_config_preserves_ps_register_failure_with_cleanup_timeout_context`
- Added RunnerConfig integration regressions for matching combinations:
  - `distributed_runner_from_runner_config_preserves_worker_register_failure_with_custom_service_type_cleanup_timeout_context`
  - `distributed_runner_from_runner_config_preserves_ps_register_failure_with_cleanup_timeout_context`
- New assertions verify for all four matrices (worker/ps × default/custom):
  - register failure remains primary,
  - role-error cleanup context is appended,
  - cleanup-timeout diagnostics include explicit deregister/disconnect operation
    context with correct service-id and service-type.
- Result: register-failure cleanup-timeout diagnostics now have complete
  default/custom parity across runner + RunConfig + RunnerConfig entrypoints.

### 212) Indexed worker register-failure cleanup-timeout diagnostics parity expanded
- Added runner-level indexed worker regression:
  - `test_run_distributed_preserves_worker_register_failure_with_custom_service_type_and_index_when_cleanup_steps_timeout`
- Added RunConfig integration regression:
  - `distributed_runner_from_run_config_preserves_worker_register_failure_with_custom_service_type_and_index_cleanup_timeout_context`
- Added RunnerConfig integration regression:
  - `distributed_runner_from_runner_config_preserves_worker_register_failure_with_custom_service_type_and_index_cleanup_timeout_context`
- New assertions verify that non-zero worker index is preserved through
  register-failure cleanup-timeout diagnostics:
  - `deregister worker-3 from trainer_custom`
  - `disconnect worker-3 via trainer_custom`
  while preserving primary register-failure precedence and role-error cleanup
  issue context.
- Result: custom worker service-type cleanup-timeout diagnostics now include
  explicit worker-index parity across runner + RunConfig + RunnerConfig paths.

### 213) Indexed PS register-failure cleanup-timeout diagnostics parity expanded
- Added runner-level indexed PS regression:
  - `test_run_distributed_preserves_ps_register_failure_with_custom_service_type_and_index_when_cleanup_steps_timeout`
- Added RunConfig integration regression:
  - `distributed_runner_from_run_config_preserves_ps_register_failure_with_custom_service_type_and_index_cleanup_timeout_context`
- Added RunnerConfig integration regression:
  - `distributed_runner_from_runner_config_preserves_ps_register_failure_with_custom_service_type_and_index_cleanup_timeout_context`
- New assertions verify that non-zero PS index is preserved through
  register-failure cleanup-timeout diagnostics:
  - `deregister ps-2 from parameter_server_custom`
  - `disconnect ps-2 via parameter_server_custom`
  while preserving primary register-failure precedence and role-error cleanup
  issue context.
- Result: custom PS service-type cleanup-timeout diagnostics now include explicit
  PS-index parity across runner + RunConfig + RunnerConfig paths.

### 214) Indexed register-failure cleanup-failure diagnostics parity expanded
- Added runner-level indexed cleanup-failure regressions:
  - `test_run_distributed_worker_registration_failure_with_custom_service_type_and_index_includes_cleanup_context`
  - `test_run_distributed_ps_registration_failure_with_custom_service_type_and_index_includes_cleanup_context`
- Added RunConfig indexed integration regressions:
  - `distributed_runner_from_run_config_preserves_worker_register_failure_with_custom_service_type_and_index_cleanup_context`
  - `distributed_runner_from_run_config_preserves_ps_register_failure_with_custom_service_type_and_index_cleanup_context`
- Added RunnerConfig indexed integration regressions:
  - `distributed_runner_from_runner_config_preserves_worker_register_failure_with_custom_service_type_and_index_cleanup_context`
  - `distributed_runner_from_runner_config_preserves_ps_register_failure_with_custom_service_type_and_index_cleanup_context`
- New assertions verify indexed cleanup-failure diagnostics preserve:
  - primary register-failure precedence,
  - role-error cleanup issue context,
  - explicit indexed operation context for failing cleanup ops
    (`worker-3` / `ps-2` with custom service types).
- Result: register-failure cleanup-failure diagnostics now have indexed parity
  across runner + RunConfig + RunnerConfig custom service-type paths.

### 215) Default-service indexed register-failure cleanup parity expanded (failure + timeout)
- Added runner-level indexed default-service regressions for cleanup-failure:
  - `test_run_distributed_worker_registration_failure_with_default_service_type_and_index_includes_cleanup_context`
  - `test_run_distributed_ps_registration_failure_with_default_service_type_and_index_includes_cleanup_context`
- Added runner-level indexed default-service regressions for cleanup-timeout:
  - `test_run_distributed_preserves_worker_register_failure_with_default_service_type_and_index_when_cleanup_steps_timeout`
  - `test_run_distributed_preserves_ps_register_failure_with_default_service_type_and_index_when_cleanup_steps_timeout`
- Added RunConfig indexed default-service integration regressions:
  - cleanup-failure:
    - `distributed_runner_from_run_config_preserves_worker_register_failure_with_default_service_type_and_index_cleanup_context`
    - `distributed_runner_from_run_config_preserves_ps_register_failure_with_default_service_type_and_index_cleanup_context`
  - cleanup-timeout:
    - `distributed_runner_from_run_config_preserves_worker_register_failure_with_default_service_type_and_index_cleanup_timeout_context`
    - `distributed_runner_from_run_config_preserves_ps_register_failure_with_default_service_type_and_index_cleanup_timeout_context`
- Added RunnerConfig indexed default-service integration regressions:
  - cleanup-failure:
    - `distributed_runner_from_runner_config_preserves_worker_register_failure_with_default_service_type_and_index_cleanup_context`
    - `distributed_runner_from_runner_config_preserves_ps_register_failure_with_default_service_type_and_index_cleanup_context`
  - cleanup-timeout:
    - `distributed_runner_from_runner_config_preserves_worker_register_failure_with_default_service_type_and_index_cleanup_timeout_context`
    - `distributed_runner_from_runner_config_preserves_ps_register_failure_with_default_service_type_and_index_cleanup_timeout_context`
- New assertions verify indexed default-service operation diagnostics preserve:
  - worker index (`worker-3`) and ps index (`ps-2`),
  - primary register-failure precedence,
  - role-error cleanup issue context,
  - explicit deregister/disconnect operation context for both cleanup-failure
    and cleanup-timeout paths.
- Result: register-failure cleanup diagnostics now have complete
  default/custom + indexed parity across runner + RunConfig + RunnerConfig.

### 216) Default-service indexed connect-failure cleanup parity expanded (failure + timeout)
- Added runner-level indexed default-service connect-failure regressions:
  - cleanup-timeout:
    - `test_run_distributed_connect_failure_does_not_hang_when_disconnect_blocks_with_default_service_type_and_index`
    - `test_run_distributed_ps_connect_failure_does_not_hang_when_disconnect_blocks_with_default_service_type_and_index`
  - cleanup-failure:
    - `test_run_distributed_returns_connect_error_when_connect_and_disconnect_fail_with_default_service_type_and_index`
    - `test_run_distributed_returns_ps_connect_error_when_connect_and_disconnect_fail_with_default_service_type_and_index`
- Added RunConfig indexed default-service connect-failure integration regressions:
  - cleanup-timeout:
    - `distributed_runner_from_run_config_preserves_default_worker_connect_failure_with_index_cleanup_timeout_context`
    - `distributed_runner_from_run_config_preserves_default_ps_connect_failure_with_index_cleanup_timeout_context`
  - cleanup-failure:
    - `distributed_runner_from_run_config_preserves_default_worker_connect_failure_with_index_disconnect_failure_context`
    - `distributed_runner_from_run_config_preserves_default_ps_connect_failure_with_index_disconnect_failure_context`
- Added RunnerConfig indexed default-service connect-failure integration regressions:
  - cleanup-timeout:
    - `distributed_runner_from_runner_config_preserves_default_worker_connect_failure_with_index_cleanup_timeout_context`
    - `distributed_runner_from_runner_config_preserves_default_ps_connect_failure_with_index_cleanup_timeout_context`
  - cleanup-failure:
    - `distributed_runner_from_runner_config_preserves_default_worker_connect_failure_with_index_disconnect_failure_context`
    - `distributed_runner_from_runner_config_preserves_default_ps_connect_failure_with_index_disconnect_failure_context`
- New assertions verify indexed default-service connect-failure diagnostics preserve:
  - worker index (`worker-3`) and ps index (`ps-2`),
  - primary connect-failure precedence,
  - role-error cleanup issue context,
  - explicit indexed disconnect cleanup operation context for both timeout and
    failure paths.
- Result: connect-failure cleanup diagnostics now have complete
  default/custom + indexed parity across runner + RunConfig + RunnerConfig.

### 217) Custom-service indexed connect-failure cleanup parity expanded (failure + timeout)
- Added runner-level indexed custom-service connect-failure regressions:
  - cleanup-timeout:
    - `test_run_distributed_connect_failure_does_not_hang_when_disconnect_blocks_with_custom_service_type_and_index`
    - `test_run_distributed_ps_connect_failure_does_not_hang_when_disconnect_blocks_with_custom_service_type_and_index`
  - cleanup-failure:
    - `test_run_distributed_returns_connect_error_when_connect_and_disconnect_fail_with_custom_service_type_and_index`
    - `test_run_distributed_returns_ps_connect_error_when_connect_and_disconnect_fail_with_custom_service_type_and_index`
- Added RunConfig indexed custom-service connect-failure integration regressions:
  - cleanup-timeout:
    - `distributed_runner_from_run_config_preserves_connect_failure_with_custom_service_type_and_index_cleanup_timeout_context`
    - `distributed_runner_from_run_config_preserves_ps_connect_failure_with_custom_service_type_and_index_cleanup_timeout_context`
  - cleanup-failure:
    - `distributed_runner_from_run_config_preserves_connect_failure_with_custom_service_type_and_index_disconnect_failure_context`
    - `distributed_runner_from_run_config_preserves_ps_connect_failure_with_custom_service_type_and_index_disconnect_failure_context`
- Added RunnerConfig indexed custom-service connect-failure integration regressions:
  - cleanup-timeout:
    - `distributed_runner_from_runner_config_preserves_connect_failure_with_custom_service_type_and_index_cleanup_timeout_context`
    - `distributed_runner_from_runner_config_preserves_ps_connect_failure_with_custom_service_type_and_index_cleanup_timeout_context`
  - cleanup-failure:
    - `distributed_runner_from_runner_config_preserves_connect_failure_with_custom_service_type_and_index_disconnect_failure_context`
    - `distributed_runner_from_runner_config_preserves_ps_connect_failure_with_custom_service_type_and_index_disconnect_failure_context`
- New assertions verify indexed custom-service connect-failure diagnostics preserve:
  - worker index (`worker-3`) and ps index (`ps-2`),
  - primary connect-failure precedence,
  - role-error cleanup issue context,
  - explicit indexed disconnect cleanup operation context for both timeout and
    failure paths.
- Result: connect-failure cleanup diagnostics now have complete
  default/custom + indexed parity across runner + RunConfig + RunnerConfig.

### 218) Custom-service indexed connect-timeout diagnostics parity expanded
- Added runner-level indexed custom-service connect-timeout regressions:
  - `test_run_distributed_connect_timeout_preserves_error_when_disconnect_cleanup_times_out_with_custom_service_type_and_index`
  - `test_run_distributed_ps_connect_timeout_preserves_error_when_disconnect_cleanup_times_out_with_custom_service_type_and_index`
- Added RunConfig indexed custom-service connect-timeout integration regressions:
  - `distributed_runner_from_run_config_propagates_worker_service_type_and_index_into_connect_timeout_diagnostics`
  - `distributed_runner_from_run_config_propagates_ps_service_type_and_index_into_connect_timeout_diagnostics`
- Added RunnerConfig indexed custom-service connect-timeout integration regressions:
  - `distributed_runner_from_runner_config_propagates_worker_service_type_and_index_into_connect_timeout_diagnostics`
  - `distributed_runner_from_runner_config_propagates_ps_service_type_and_index_into_connect_timeout_diagnostics`
- New assertions verify timeout diagnostics preserve both custom service type and
  non-zero role index for operation and cleanup contexts:
  - worker path: `connect/disconnect worker-3 via trainer_custom`
  - ps path: `connect/disconnect ps-2 via parameter_server_custom`
- Result: connect-timeout diagnostics now have explicit service-type+index parity
  across runner + RunConfig + RunnerConfig paths.

### 219) Default-service indexed connect-timeout diagnostics parity expanded
- Added runner-level indexed default-service connect-timeout regressions:
  - `test_run_distributed_connect_timeout_preserves_error_when_disconnect_cleanup_times_out_with_default_service_type_and_index`
  - `test_run_distributed_ps_connect_timeout_preserves_error_when_disconnect_cleanup_times_out_with_default_service_type_and_index`
- New assertions verify default-service non-zero index timeout diagnostics preserve:
  - worker path: `connect/disconnect worker-3 via worker`
  - ps path: `connect/disconnect ps-2 via ps`
  while keeping connect-timeout precedence and appended role-error cleanup
  context.
- Result: runner-level connect-timeout diagnostics now have complete
  default/custom + indexed parity for worker and ps paths.

### 220) Connect-timeout + disconnect-failure precedence parity expanded
- Added runner-level regressions for blocked connect + failing disconnect cleanup:
  - `test_run_distributed_connect_timeout_preserves_error_when_disconnect_cleanup_fails`
  - `test_run_distributed_ps_connect_timeout_preserves_error_when_disconnect_cleanup_fails_with_custom_service_type_and_index`
- Added integration regression coverage across config entrypoints:
  - RunConfig:
    - `distributed_runner_from_run_config_preserves_connect_timeout_with_disconnect_failure_context`
    - `distributed_runner_from_run_config_preserves_ps_connect_timeout_with_disconnect_failure_context`
  - RunnerConfig:
    - `distributed_runner_from_runner_config_preserves_connect_timeout_with_disconnect_failure_context`
    - `distributed_runner_from_runner_config_preserves_ps_connect_timeout_with_disconnect_failure_context`
- Added/used dedicated integration mock:
  - `HangingConnectWithFailingDisconnectFromConfigDiscovery`
  (connect blocks to trigger operation timeout while disconnect fails immediately).
- New assertions verify:
  - primary connect-timeout error is preserved,
  - role-error cleanup issue context is appended,
  - cleanup diagnostics include explicit disconnect failure operation context with
    propagated role index and custom service type where configured.
- Result: connect-timeout error-precedence semantics now cover both cleanup-timeout
  and cleanup-failure paths across runner + RunConfig + RunnerConfig entrypoints.

### 221) Register-timeout + cleanup-failure precedence parity expanded
- Added runner-level regressions for blocked register + failing cleanup:
  - `test_run_distributed_worker_register_timeout_preserves_error_when_cleanup_fails`
  - `test_run_distributed_ps_register_timeout_preserves_error_when_cleanup_fails_with_custom_service_type_and_index`
- Added dedicated failing-cleanup register-timeout integration mock:
  - `HangingRegisterWithFailingCleanupFromConfigDiscovery`
  (register blocks to trigger operation timeout while deregister/disconnect fail).
- Added RunConfig integration regressions:
  - `distributed_runner_from_run_config_preserves_register_timeout_with_disconnect_failure_context`
  - `distributed_runner_from_run_config_preserves_ps_register_timeout_with_disconnect_failure_context`
- Added RunnerConfig integration regressions:
  - `distributed_runner_from_runner_config_preserves_register_timeout_with_disconnect_failure_context`
  - `distributed_runner_from_runner_config_preserves_ps_register_timeout_with_disconnect_failure_context`
- New assertions verify:
  - primary register-timeout error remains authoritative,
  - role-error cleanup issue context is appended,
  - cleanup diagnostics include explicit deregister/disconnect failure operation
    context with propagated custom service type and non-zero role index.
- Result: register-timeout parity now covers both cleanup-timeout and
  cleanup-failure branches across runner + RunConfig + RunnerConfig entrypoints.

### 222) Discover-operation-timeout cleanup diagnostics parity expanded
- Added runner-level discover-operation-timeout cleanup regressions:
  - `test_run_distributed_worker_discover_timeout_preserves_error_when_cleanup_times_out_with_custom_service_type_and_index`
  - `test_run_distributed_worker_discover_timeout_preserves_error_when_cleanup_fails_with_default_service_type_and_index`
- Added integration discovery mocks for blocked discover + cleanup issues:
  - `HangingDiscoverWithHangingCleanupFromConfigDiscovery`
  - `HangingDiscoverWithFailingCleanupFromConfigDiscovery`
- Added RunConfig integration regressions:
  - `distributed_runner_from_run_config_preserves_discover_timeout_with_custom_service_types_and_index_when_cleanup_times_out`
  - `distributed_runner_from_run_config_preserves_discover_timeout_with_default_service_type_and_index_when_cleanup_fails`
- Added RunnerConfig integration regressions:
  - `distributed_runner_from_runner_config_preserves_discover_timeout_with_custom_service_types_and_index_when_cleanup_times_out`
  - `distributed_runner_from_runner_config_preserves_discover_timeout_with_default_service_type_and_index_when_cleanup_fails`
- New assertions verify:
  - primary worker discovery-timeout error remains authoritative,
  - last discover operation-timeout diagnostics preserve queried PS service type
    and worker index context,
  - role-error cleanup issue context includes explicit deregister/disconnect
    timeout or failure diagnostics with propagated worker service type + index.
- Result: discover-operation-timeout paths now have explicit cleanup-timeout and
  cleanup-failure parity coverage across runner + RunConfig + RunnerConfig
  entrypoints.

### 223) Discover-timeout cleanup matrix parity completed
- Expanded runner discover-timeout regressions to cover remaining matrix cells:
  - `test_run_distributed_worker_discover_timeout_preserves_error_when_cleanup_times_out_with_default_service_type_and_index`
  - `test_run_distributed_worker_discover_timeout_preserves_error_when_cleanup_fails_with_custom_service_type_and_index`
- Expanded integration discover-timeout coverage across both config entrypoints:
  - RunConfig:
    - `distributed_runner_from_run_config_preserves_discover_timeout_with_default_service_type_and_index_when_cleanup_times_out`
    - `distributed_runner_from_run_config_preserves_discover_timeout_with_custom_service_types_and_index_when_cleanup_fails`
  - RunnerConfig:
    - `distributed_runner_from_runner_config_preserves_discover_timeout_with_default_service_type_and_index_when_cleanup_times_out`
    - `distributed_runner_from_runner_config_preserves_discover_timeout_with_custom_service_types_and_index_when_cleanup_fails`
- New assertions complete the discover-timeout cleanup matrix by validating:
  - default/custom PS service-type propagation in discovery-timeout diagnostics,
  - default/custom worker service-type propagation in cleanup diagnostics,
  - indexed worker identity propagation (`worker-2`, `worker-3`) across
    operation timeout and cleanup timeout/failure contexts.
- Result: discover-timeout diagnostics now have full default/custom ×
  cleanup-timeout/cleanup-failure parity across runner + RunConfig +
  RunnerConfig paths.

### 224) Register-timeout cleanup matrix parity completed
- Expanded runner register-timeout regressions to cover remaining matrix cells:
  - `test_run_distributed_worker_register_timeout_preserves_error_when_cleanup_times_out_with_custom_service_type_and_index`
  - `test_run_distributed_worker_register_timeout_preserves_error_when_cleanup_fails_with_custom_service_type_and_index`
  - `test_run_distributed_ps_register_timeout_preserves_error_when_cleanup_times_out_with_custom_service_type_and_index`
  - `test_run_distributed_ps_register_timeout_preserves_error_when_cleanup_fails_with_default_service_type_and_index`
- Expanded RunConfig integration coverage:
  - `distributed_runner_from_run_config_preserves_register_timeout_with_default_service_type_and_index_disconnect_failure_context`
  - `distributed_runner_from_run_config_preserves_register_timeout_with_custom_service_type_and_index_when_cleanup_blocks`
  - `distributed_runner_from_run_config_preserves_ps_register_timeout_with_default_service_type_and_index_disconnect_failure_context`
  - `distributed_runner_from_run_config_preserves_ps_register_timeout_with_custom_service_type_and_index_when_cleanup_blocks`
- Expanded RunnerConfig integration coverage:
  - `distributed_runner_from_runner_config_preserves_register_timeout_with_default_service_type_and_index_disconnect_failure_context`
  - `distributed_runner_from_runner_config_preserves_register_timeout_with_custom_service_type_and_index_when_cleanup_blocks`
  - `distributed_runner_from_runner_config_preserves_ps_register_timeout_with_default_service_type_and_index_disconnect_failure_context`
  - `distributed_runner_from_runner_config_preserves_ps_register_timeout_with_custom_service_type_and_index_when_cleanup_blocks`
- New assertions complete the register-timeout cleanup matrix by verifying:
  - default/custom service-type propagation in operation-timeout diagnostics,
  - default/custom service-type propagation in cleanup diagnostics,
  - indexed role identity propagation (`worker-2`, `worker-3`, `ps-2`) across
    cleanup-timeout and cleanup-failure contexts.
- Result: register-timeout diagnostics now have full default/custom ×
  cleanup-timeout/cleanup-failure parity across runner + RunConfig +
  RunnerConfig entrypoints.

### 225) Worker-timeout cleanup matrix parity expanded
- Added runner-level worker-timeout regressions:
  - `test_run_distributed_preserves_worker_error_with_custom_service_types_and_index_when_cleanup_steps_timeout`
  - `test_run_distributed_preserves_worker_timeout_with_default_service_type_and_index_when_cleanup_steps_fail`
- Added RunConfig integration regressions:
  - `distributed_runner_from_run_config_preserves_worker_timeout_with_custom_service_types_and_index_when_cleanup_times_out`
  - `distributed_runner_from_run_config_preserves_worker_timeout_with_default_service_type_and_index_when_cleanup_fails`
- Added RunnerConfig integration regressions:
  - `distributed_runner_from_runner_config_preserves_worker_timeout_with_custom_service_types_and_index_when_cleanup_times_out`
  - `distributed_runner_from_runner_config_preserves_worker_timeout_with_default_service_type_and_index_when_cleanup_fails`
- New assertions verify worker-timeout diagnostics preserve:
  - custom/default PS service-type context,
  - indexed worker identity propagation (`worker-2`, `worker-3`),
  - role-error cleanup issue context with custom/default worker service-type
    timeout/failure cleanup operation diagnostics.
- Result: worker-timeout cleanup diagnostics now explicitly cover indexed
  default/custom service-type timeout/failure branches across runner + RunConfig
  + RunnerConfig paths.

### 226) Discover-error cleanup matrix parity expanded for indexed defaults
- Added runner-level discover-error regressions:
  - `test_run_distributed_preserves_worker_discover_failure_with_default_service_type_and_index_when_cleanup_steps_timeout`
  - `test_run_distributed_preserves_worker_discover_failure_with_default_service_type_and_index_when_cleanup_steps_fail`
- Added RunConfig integration regressions:
  - `distributed_runner_from_run_config_preserves_last_discover_error_with_default_service_type_and_index_when_cleanup_times_out`
  - `distributed_runner_from_run_config_preserves_last_discover_error_with_default_service_type_and_index_when_cleanup_fails`
- Added RunnerConfig integration regressions:
  - `distributed_runner_from_runner_config_preserves_last_discover_error_with_default_service_type_and_index_when_cleanup_times_out`
  - `distributed_runner_from_runner_config_preserves_last_discover_error_with_default_service_type_and_index_when_cleanup_fails`
- New assertions verify indexed default-service discover-error diagnostics
  preserve:
  - worker index propagation (`worker-2`) in role-error context,
  - last discover error propagation (`forced discover failure`),
  - cleanup timeout/failure operation diagnostics for default worker service
    type (`worker`) on deregister/disconnect steps.
- Result: discover-error cleanup diagnostics now include explicit indexed
  default-service parity coverage across runner + RunConfig + RunnerConfig
  entrypoints.

### 227) Worker-timeout cleanup-timeout indexed default parity completed
- Added runner-level worker-timeout cleanup-timeout regression:
  - `test_run_distributed_preserves_worker_error_with_default_service_type_and_index_when_cleanup_steps_timeout`
- Added RunConfig integration regression:
  - `distributed_runner_from_run_config_preserves_worker_timeout_with_default_service_type_and_index_when_cleanup_times_out`
- Added RunnerConfig integration regression:
  - `distributed_runner_from_runner_config_preserves_worker_timeout_with_default_service_type_and_index_when_cleanup_times_out`
- New assertions verify indexed default-service worker-timeout diagnostics
  preserve:
  - default PS service type context (`service type: ps`),
  - indexed worker identity propagation (`for worker-2`),
  - cleanup-timeout operation diagnostics with default worker service type
    (`deregister/disconnect worker-2 ... worker`) under bounded cleanup timeout.
- Result: worker-timeout cleanup-timeout diagnostics now have explicit indexed
  default-service parity across runner + RunConfig + RunnerConfig entrypoints.

### 228) Connect-timeout cleanup-failure matrix parity expanded
- Added runner-level connect-timeout cleanup-failure regressions:
  - `test_run_distributed_connect_timeout_preserves_error_when_disconnect_cleanup_fails_with_custom_service_type_and_index`
  - `test_run_distributed_ps_connect_timeout_preserves_error_when_disconnect_cleanup_fails_with_default_service_type_and_index`
- Added RunConfig integration regressions:
  - `distributed_runner_from_run_config_preserves_default_connect_timeout_with_index_disconnect_failure_context`
  - `distributed_runner_from_run_config_preserves_default_ps_connect_timeout_with_index_disconnect_failure_context`
- Added RunnerConfig integration regressions:
  - `distributed_runner_from_runner_config_preserves_default_connect_timeout_with_index_disconnect_failure_context`
  - `distributed_runner_from_runner_config_preserves_default_ps_connect_timeout_with_index_disconnect_failure_context`
- New assertions verify connect-timeout diagnostics preserve:
  - indexed worker/ps identities (`worker-2`, `ps-2`),
  - default/custom service-type propagation in operation-timeout context,
  - appended cleanup issue context with explicit disconnect-failure operation
    diagnostics.
- Result: connect-timeout + disconnect-failure diagnostics now include explicit
  default/custom indexed parity across runner + RunConfig + RunnerConfig
  entrypoints.

### 229) Runner connect-timeout cleanup-failure indexed default-worker parity added
- Added runner-level regression:
  - `test_run_distributed_connect_timeout_preserves_error_when_disconnect_cleanup_fails_with_default_service_type_and_index`
- New assertions verify indexed default-worker connect-timeout diagnostics
  preserve:
  - operation-timeout context (`connect worker-2 via worker`),
  - cleanup issue context,
  - disconnect-failure diagnostics with indexed default-worker identity
    (`disconnect worker-2 via worker` + `forced disconnect failure`).
- Result: runner-level connect-timeout + cleanup-failure diagnostics now include
  explicit indexed default-worker parity alongside existing custom/default PS
  and custom worker coverage.

### 230) Register-timeout cleanup-timeout indexed default parity completed
- Added runner-level regressions:
  - `test_run_distributed_worker_register_timeout_preserves_error_when_cleanup_times_out_with_default_service_type_and_index`
  - `test_run_distributed_ps_register_timeout_preserves_error_when_cleanup_times_out_with_default_service_type_and_index`
- Added RunConfig integration regressions:
  - `distributed_runner_from_run_config_preserves_register_timeout_with_default_service_type_and_index_when_cleanup_blocks`
  - `distributed_runner_from_run_config_preserves_ps_register_timeout_with_default_service_type_and_index_when_cleanup_blocks`
- Added RunnerConfig integration regressions:
  - `distributed_runner_from_runner_config_preserves_register_timeout_with_default_service_type_and_index_when_cleanup_blocks`
  - `distributed_runner_from_runner_config_preserves_ps_register_timeout_with_default_service_type_and_index_when_cleanup_blocks`
- New assertions verify indexed default-service register-timeout diagnostics
  preserve:
  - operation-timeout context for worker/ps registration (`worker-2`, `ps-2`),
  - cleanup issue aggregation context,
  - cleanup timeout operation diagnostics for deregister/disconnect using
    default discovery service types (`worker`, `ps`).
- Result: register-timeout + cleanup-timeout diagnostics now have explicit
  indexed default-service parity across runner + RunConfig + RunnerConfig
  entrypoints.

### 231) Runner worker register-timeout cleanup-failure indexed default parity added
- Added runner-level regression:
  - `test_run_distributed_worker_register_timeout_preserves_error_when_cleanup_fails_with_default_service_type_and_index`
- New assertions verify indexed default-worker register-timeout diagnostics
  preserve:
  - operation-timeout context (`register worker-2 as worker`),
  - cleanup issue aggregation context,
  - cleanup-failure operation diagnostics with indexed default-worker identity
    (`deregister/disconnect worker-2 ... worker` + forced failures).
- Result: runner-level register-timeout + cleanup-failure diagnostics now
  include explicit indexed default-worker parity, aligning with existing
  indexed default PS and integration-level coverage.

### 232) Worker ordering-issue timeout cleanup diagnostics parity expanded
- Added runner-level ordering-issue timeout regressions:
  - `test_run_distributed_preserves_worker_ordering_issue_timeout_with_default_service_type_and_index_when_cleanup_steps_timeout`
  - `test_run_distributed_preserves_worker_ordering_issue_timeout_with_custom_service_types_and_index_when_cleanup_steps_fail`
- Added RunConfig integration regressions:
  - `distributed_runner_from_run_config_preserves_worker_ordering_issue_timeout_with_default_service_type_and_index_when_cleanup_times_out`
  - `distributed_runner_from_run_config_preserves_worker_ordering_issue_timeout_with_custom_service_types_and_index_when_cleanup_fails`
- Added RunnerConfig integration regressions:
  - `distributed_runner_from_runner_config_preserves_worker_ordering_issue_timeout_with_default_service_type_and_index_when_cleanup_times_out`
  - `distributed_runner_from_runner_config_preserves_worker_ordering_issue_timeout_with_custom_service_types_and_index_when_cleanup_fails`
- New assertions verify worker timeout diagnostics preserve:
  - ordering error payload (`last ordering issue: MixedIndexMetadataPresence`),
  - default/custom PS service-type context and indexed worker service-id propagation,
  - cleanup timeout/failure operation diagnostics on deregister/disconnect.
- Result: ordering-issue-driven worker timeout diagnostics now have explicit
  cleanup timeout/failure parity across runner + RunConfig + RunnerConfig
  entrypoints.

### 233) Worker ordering+discovery composite timeout cleanup parity expanded
- Added runner-level composite-timeout regressions:
  - `test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_with_default_service_type_and_index_when_cleanup_steps_timeout`
  - `test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_with_custom_service_types_and_index_when_cleanup_steps_fail`
- Added RunConfig integration regressions:
  - `distributed_runner_from_run_config_preserves_worker_ordering_and_discovery_error_timeout_with_default_service_type_and_index_when_cleanup_times_out`
  - `distributed_runner_from_run_config_preserves_worker_ordering_and_discovery_error_timeout_with_custom_service_types_and_index_when_cleanup_fails`
- Added RunnerConfig integration regressions:
  - `distributed_runner_from_runner_config_preserves_worker_ordering_and_discovery_error_timeout_with_default_service_type_and_index_when_cleanup_times_out`
  - `distributed_runner_from_runner_config_preserves_worker_ordering_and_discovery_error_timeout_with_custom_service_types_and_index_when_cleanup_fails`
- Added composite sequencing discovery mocks (ordering issue on first discover,
  discover failure on retry) for runner + integration parity paths.
- New assertions verify composite worker timeout diagnostics preserve both:
  - `last ordering issue: MixedIndexMetadataPresence`
  - `last discovery error: Internal error: forced discover failure`
  while still appending cleanup timeout/failure operation diagnostics.
- Result: composite ordering+discover-error timeout precedence and cleanup
  context propagation now have explicit parity coverage across runner + RunConfig
  + RunnerConfig entrypoints.

### 234) Worker ordering+discovery composite timeout cleanup matrix completed
- Added runner-level complementary composite regressions:
  - `test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_with_default_service_type_and_index_when_cleanup_steps_fail`
  - `test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_with_custom_service_types_and_index_when_cleanup_steps_timeout`
- Added RunConfig integration complementary regressions:
  - `distributed_runner_from_run_config_preserves_worker_ordering_and_discovery_error_timeout_with_default_service_type_and_index_when_cleanup_fails`
  - `distributed_runner_from_run_config_preserves_worker_ordering_and_discovery_error_timeout_with_custom_service_types_and_index_when_cleanup_times_out`
- Added RunnerConfig integration complementary regressions:
  - `distributed_runner_from_runner_config_preserves_worker_ordering_and_discovery_error_timeout_with_default_service_type_and_index_when_cleanup_fails`
  - `distributed_runner_from_runner_config_preserves_worker_ordering_and_discovery_error_timeout_with_custom_service_types_and_index_when_cleanup_times_out`
- New assertions complete the composite matrix by validating both remaining
  branches:
  - default service-type + cleanup-failure composite diagnostics,
  - custom service-type + cleanup-timeout composite diagnostics,
  while preserving ordering/discovery error precedence payloads.
- Result: worker ordering+discovery composite timeout cleanup diagnostics now
  have full default/custom × timeout/failure matrix parity across runner +
  RunConfig + RunnerConfig entrypoints.

### 235) Worker ordering-issue timeout cleanup matrix completed
- Added runner-level complementary ordering-issue regressions:
  - `test_run_distributed_preserves_worker_ordering_issue_timeout_with_default_service_type_and_index_when_cleanup_steps_fail`
  - `test_run_distributed_preserves_worker_ordering_issue_timeout_with_custom_service_types_and_index_when_cleanup_steps_timeout`
- Added RunConfig integration complementary regressions:
  - `distributed_runner_from_run_config_preserves_worker_ordering_issue_timeout_with_default_service_type_and_index_when_cleanup_fails`
  - `distributed_runner_from_run_config_preserves_worker_ordering_issue_timeout_with_custom_service_types_and_index_when_cleanup_times_out`
- Added RunnerConfig integration complementary regressions:
  - `distributed_runner_from_runner_config_preserves_worker_ordering_issue_timeout_with_default_service_type_and_index_when_cleanup_fails`
  - `distributed_runner_from_runner_config_preserves_worker_ordering_issue_timeout_with_custom_service_types_and_index_when_cleanup_times_out`
- New assertions complete the ordering-issue matrix by validating both
  remaining branches:
  - default service-type + cleanup-failure ordering diagnostics,
  - custom service-type + cleanup-timeout ordering diagnostics,
  while preserving `last ordering issue: MixedIndexMetadataPresence`.
- Result: worker ordering-issue timeout cleanup diagnostics now have full
  default/custom × timeout/failure matrix parity across runner + RunConfig +
  RunnerConfig entrypoints.

### 236) Worker ordering-issue custom non-index cleanup parity expanded
- Added runner-level custom non-index regressions:
  - `test_run_distributed_preserves_worker_ordering_issue_timeout_with_custom_service_type_when_cleanup_steps_timeout`
  - `test_run_distributed_preserves_worker_ordering_issue_timeout_with_custom_service_type_when_cleanup_steps_fail`
- Added RunConfig integration custom non-index regressions:
  - `distributed_runner_from_run_config_preserves_worker_ordering_issue_timeout_with_custom_service_type_when_cleanup_times_out`
  - `distributed_runner_from_run_config_preserves_worker_ordering_issue_timeout_with_custom_service_type_when_cleanup_fails`
- Added RunnerConfig integration custom non-index regressions:
  - `distributed_runner_from_runner_config_preserves_worker_ordering_issue_timeout_with_custom_service_type_when_cleanup_times_out`
  - `distributed_runner_from_runner_config_preserves_worker_ordering_issue_timeout_with_custom_service_type_when_cleanup_fails`
- New assertions verify custom service-type ordering-issue diagnostics for
  non-index workers (`worker-0`) preserve ordering issue precedence and append
  custom worker cleanup timeout/failure operation context.
- Result: ordering-issue cleanup diagnostics now explicitly cover both indexed
  and non-indexed custom service-type paths across runner + RunConfig +
  RunnerConfig entrypoints.

### 237) Worker ordering+discovery composite custom non-index cleanup parity expanded
- Added runner-level custom non-index composite regressions:
  - `test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_with_custom_service_type_when_cleanup_steps_timeout`
  - `test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_with_custom_service_type_when_cleanup_steps_fail`
- Added RunConfig integration custom non-index composite regressions:
  - `distributed_runner_from_run_config_preserves_worker_ordering_and_discovery_error_timeout_with_custom_service_type_when_cleanup_times_out`
  - `distributed_runner_from_run_config_preserves_worker_ordering_and_discovery_error_timeout_with_custom_service_type_when_cleanup_fails`
- Added RunnerConfig integration custom non-index composite regressions:
  - `distributed_runner_from_runner_config_preserves_worker_ordering_and_discovery_error_timeout_with_custom_service_type_when_cleanup_times_out`
  - `distributed_runner_from_runner_config_preserves_worker_ordering_and_discovery_error_timeout_with_custom_service_type_when_cleanup_fails`
- New assertions verify custom non-index worker (`worker-0`) composite timeout
  diagnostics preserve both ordering/discovery precedence payloads:
  - `last ordering issue: MixedIndexMetadataPresence`
  - `last discovery error: Internal error: forced discover failure`
  while appending custom worker cleanup timeout/failure operation context.
- Result: ordering+discovery composite timeout cleanup diagnostics now
  explicitly cover both indexed and non-indexed custom service-type paths across
  runner + RunConfig + RunnerConfig entrypoints.

### 238) Worker discover-timeout custom non-index cleanup parity expanded
- Added runner-level custom non-index discover-timeout regressions:
  - `test_run_distributed_worker_discover_timeout_preserves_error_when_cleanup_times_out_with_custom_service_type`
  - `test_run_distributed_worker_discover_timeout_preserves_error_when_cleanup_fails_with_custom_service_type`
- Added RunConfig integration custom non-index discover-timeout regressions:
  - `distributed_runner_from_run_config_preserves_discover_timeout_with_custom_service_type_when_cleanup_times_out`
  - `distributed_runner_from_run_config_preserves_discover_timeout_with_custom_service_type_when_cleanup_fails`
- Added RunnerConfig integration custom non-index discover-timeout regressions:
  - `distributed_runner_from_runner_config_preserves_discover_timeout_with_custom_service_type_when_cleanup_times_out`
  - `distributed_runner_from_runner_config_preserves_discover_timeout_with_custom_service_type_when_cleanup_fails`
- New assertions verify custom non-index worker (`worker-0`) discover-timeout
  diagnostics preserve:
  - primary timeout precedence (`Timed out waiting for PS discovery`)
  - custom discovery service-type operation context
    (`discover worker-0 for parameter_server_custom`)
  - cleanup timeout/failure operation diagnostics for custom worker service type
    (`trainer_custom`) on deregister + disconnect.
- Result: discover-timeout cleanup diagnostics now explicitly cover both indexed
  and non-indexed custom service-type paths across runner + RunConfig +
  RunnerConfig entrypoints.

### 239) Worker discover-error custom non-index cleanup parity expanded
- Added runner-level custom non-index discover-error regressions:
  - `test_run_distributed_preserves_worker_discover_failure_with_custom_service_type_when_cleanup_steps_timeout`
  - `test_run_distributed_preserves_worker_discover_failure_with_custom_service_type_when_cleanup_steps_fail`
- Added RunConfig integration custom non-index discover-error regressions:
  - `distributed_runner_from_run_config_preserves_last_discover_error_with_custom_service_type_when_cleanup_times_out`
  - `distributed_runner_from_run_config_preserves_last_discover_error_with_custom_service_type_when_cleanup_fails`
- Added RunnerConfig integration custom non-index discover-error regressions:
  - `distributed_runner_from_runner_config_preserves_last_discover_error_with_custom_service_type_when_cleanup_times_out`
  - `distributed_runner_from_runner_config_preserves_last_discover_error_with_custom_service_type_when_cleanup_fails`
- New assertions verify custom non-index worker (`worker-0`) discover-error
  diagnostics preserve:
  - primary timeout precedence (`Timed out waiting for PS discovery`)
  - preserved discover-error payload
    (`last discovery error: Internal error: forced discover failure`)
  - custom cleanup timeout/failure operation diagnostics for custom worker
    service type (`trainer_custom`) on deregister + disconnect.
- Result: discover-error cleanup diagnostics now explicitly cover both indexed
  and non-indexed custom service-type paths across runner + RunConfig +
  RunnerConfig entrypoints.

### 240) Worker timeout custom non-index cleanup parity expanded
- Added runner-level custom non-index worker-timeout regressions:
  - `test_run_distributed_preserves_worker_error_with_custom_service_type_when_cleanup_steps_timeout`
  - `test_run_distributed_preserves_worker_timeout_with_custom_service_type_when_cleanup_steps_fail`
- Added RunConfig integration custom non-index worker-timeout regressions:
  - `distributed_runner_from_run_config_preserves_worker_timeout_with_custom_service_type_when_cleanup_times_out`
  - `distributed_runner_from_run_config_preserves_worker_timeout_with_custom_service_type_when_cleanup_fails`
- Added RunnerConfig integration custom non-index worker-timeout regressions:
  - `distributed_runner_from_runner_config_preserves_worker_timeout_with_custom_service_type_when_cleanup_times_out`
  - `distributed_runner_from_runner_config_preserves_worker_timeout_with_custom_service_type_when_cleanup_fails`
- New assertions verify custom non-index worker (`worker-0`) timeout
  diagnostics preserve:
  - primary timeout precedence (`Timed out waiting for PS discovery`)
  - custom discovery service-type context (`parameter_server_custom`)
  - custom worker cleanup timeout/failure operation diagnostics
    (`trainer_custom`) on deregister + disconnect.
- Result: worker-timeout cleanup diagnostics now explicitly cover both indexed
  and non-indexed custom service-type paths across runner + RunConfig +
  RunnerConfig entrypoints.

### 241) Register-timeout custom non-index cleanup-failure parity expanded
- Added runner-level custom non-index register-timeout cleanup-failure
  regressions:
  - `test_run_distributed_worker_register_timeout_preserves_error_when_cleanup_fails_with_custom_service_type`
  - `test_run_distributed_ps_register_timeout_preserves_error_when_cleanup_fails_with_custom_service_type`
- Added runner-level custom non-index register-timeout cleanup-timeout
  complements:
  - `test_run_distributed_worker_register_timeout_preserves_error_when_cleanup_times_out_with_custom_service_type`
  - `test_run_distributed_ps_register_timeout_preserves_error_when_cleanup_times_out_with_custom_service_type`
- Added RunConfig integration custom non-index cleanup-failure regressions:
  - `distributed_runner_from_run_config_preserves_register_timeout_with_custom_service_type_disconnect_failure_context`
  - `distributed_runner_from_run_config_preserves_ps_register_timeout_with_custom_service_type_disconnect_failure_context`
- Added RunnerConfig integration custom non-index cleanup-failure regressions:
  - `distributed_runner_from_runner_config_preserves_register_timeout_with_custom_service_type_disconnect_failure_context`
  - `distributed_runner_from_runner_config_preserves_ps_register_timeout_with_custom_service_type_disconnect_failure_context`
- New assertions verify non-index custom worker/ps register-timeout diagnostics
  preserve primary operation-timeout precedence and append custom
  service-type-specific cleanup failure/timeout operation context on
  deregister + disconnect.
- Result: register-timeout cleanup diagnostics now explicitly cover both indexed
  and non-indexed custom service-type failure paths across runner + RunConfig +
  RunnerConfig entrypoints.

### 242) Register-timeout custom non-index cleanup-timeout integration parity expanded
- Added RunConfig integration custom non-index cleanup-timeout regressions:
  - `distributed_runner_from_run_config_preserves_register_timeout_with_custom_service_type_when_cleanup_blocks`
  - `distributed_runner_from_run_config_preserves_ps_register_timeout_with_custom_service_type_when_cleanup_blocks`
- Added RunnerConfig integration custom non-index cleanup-timeout regressions:
  - `distributed_runner_from_runner_config_preserves_register_timeout_with_custom_service_type_when_cleanup_blocks`
  - `distributed_runner_from_runner_config_preserves_ps_register_timeout_with_custom_service_type_when_cleanup_blocks`
- New assertions verify non-index custom worker/ps register-timeout diagnostics
  preserve primary operation-timeout precedence and append custom
  service-type-specific cleanup-timeout operation context on
  deregister + disconnect.
- Result: register-timeout cleanup-timeout diagnostics now explicitly cover both
  indexed and non-indexed custom service-type paths across RunConfig +
  RunnerConfig entrypoints.

### 243) Connect-timeout custom non-index cleanup-failure parity expanded
- Added runner-level custom non-index connect-timeout cleanup-failure
  regressions:
  - `test_run_distributed_connect_timeout_preserves_error_when_disconnect_cleanup_fails_with_custom_service_type`
  - `test_run_distributed_ps_connect_timeout_preserves_error_when_disconnect_cleanup_fails_with_custom_service_type`
- Added RunConfig integration custom non-index connect-timeout
  cleanup-failure regressions:
  - `distributed_runner_from_run_config_preserves_connect_timeout_with_custom_service_type_disconnect_failure_context`
  - `distributed_runner_from_run_config_preserves_ps_connect_timeout_with_custom_service_type_disconnect_failure_context`
- Added RunnerConfig integration custom non-index connect-timeout
  cleanup-failure regressions:
  - `distributed_runner_from_runner_config_preserves_connect_timeout_with_custom_service_type_disconnect_failure_context`
  - `distributed_runner_from_runner_config_preserves_ps_connect_timeout_with_custom_service_type_disconnect_failure_context`
- New assertions verify non-index custom worker/ps connect-timeout diagnostics
  preserve primary operation-timeout precedence and append custom
  service-type-specific disconnect-failure cleanup operation context.
- Result: connect-timeout cleanup-failure diagnostics now explicitly cover both
  indexed and non-indexed custom service-type paths across runner + RunConfig +
  RunnerConfig entrypoints.

### 244) Connect-timeout custom non-index cleanup-timeout parity expanded
- Added runner-level custom non-index connect-timeout cleanup-timeout
  regression:
  - `test_run_distributed_connect_timeout_preserves_error_when_disconnect_cleanup_times_out_with_custom_service_type`
- Added RunConfig integration custom non-index connect-timeout cleanup-timeout
  regressions:
  - `distributed_runner_from_run_config_preserves_connect_timeout_with_custom_service_type_when_cleanup_blocks`
  - `distributed_runner_from_run_config_preserves_ps_connect_timeout_with_custom_service_type_when_cleanup_blocks`
- Added RunnerConfig integration custom non-index connect-timeout
  cleanup-timeout regressions:
  - `distributed_runner_from_runner_config_preserves_connect_timeout_with_custom_service_type_when_cleanup_blocks`
  - `distributed_runner_from_runner_config_preserves_ps_connect_timeout_with_custom_service_type_when_cleanup_blocks`
- New assertions verify non-index custom worker/ps connect-timeout diagnostics
  preserve primary operation-timeout precedence and append custom
  service-type-specific disconnect-timeout cleanup operation context.
- Result: connect-timeout cleanup-timeout diagnostics now explicitly cover both
  indexed and non-indexed custom service-type paths across runner + RunConfig +
  RunnerConfig entrypoints.

### 245) Connect-failure custom non-index cleanup-context parity expanded
- Added runner-level custom non-index connect-failure cleanup-context
  regressions:
  - `test_run_distributed_connect_failure_does_not_hang_when_disconnect_blocks_with_custom_service_type`
  - `test_run_distributed_returns_connect_error_when_connect_and_disconnect_fail_with_custom_service_type`
- Added RunConfig integration custom non-index connect-failure cleanup-context
  regressions:
  - `distributed_runner_from_run_config_preserves_connect_failure_with_custom_service_type_cleanup_timeout_context`
  - `distributed_runner_from_run_config_preserves_connect_failure_with_custom_service_type_disconnect_failure_context`
  - `distributed_runner_from_run_config_preserves_ps_connect_failure_with_custom_service_type_cleanup_timeout_context`
  - `distributed_runner_from_run_config_preserves_ps_connect_failure_with_custom_service_type_disconnect_failure_context`
- Added RunnerConfig integration custom non-index connect-failure
  cleanup-context regressions:
  - `distributed_runner_from_runner_config_preserves_connect_failure_with_custom_service_type_cleanup_timeout_context`
  - `distributed_runner_from_runner_config_preserves_connect_failure_with_custom_service_type_disconnect_failure_context`
  - `distributed_runner_from_runner_config_preserves_ps_connect_failure_with_custom_service_type_cleanup_timeout_context`
  - `distributed_runner_from_runner_config_preserves_ps_connect_failure_with_custom_service_type_disconnect_failure_context`
- New assertions verify custom non-index worker/ps connect-failure diagnostics
  preserve primary connect-failure precedence and append custom service-type
  cleanup operation context for timeout and failure variants.
- Result: connect-failure cleanup diagnostics now explicitly cover indexed and
  non-indexed custom service-type paths across runner + RunConfig +
  RunnerConfig entrypoints.

### 246) Discover-timeout custom non-index custom-service-types parity expanded
- Added runner-level custom non-index discover-timeout cleanup regressions:
  - `test_run_distributed_preserves_worker_discover_failure_with_custom_service_types_when_cleanup_steps_timeout`
  - `test_run_distributed_preserves_worker_discover_failure_with_custom_service_types_when_cleanup_steps_fail`
- Added RunConfig integration custom non-index discover-timeout cleanup
  regressions:
  - `distributed_runner_from_run_config_preserves_discover_timeout_with_custom_service_types_when_cleanup_times_out`
  - `distributed_runner_from_run_config_preserves_discover_timeout_with_custom_service_types_when_cleanup_fails`
- Added RunnerConfig integration custom non-index discover-timeout cleanup
  regressions:
  - `distributed_runner_from_runner_config_preserves_discover_timeout_with_custom_service_types_when_cleanup_times_out`
  - `distributed_runner_from_runner_config_preserves_discover_timeout_with_custom_service_types_when_cleanup_fails`
- New assertions verify custom non-index worker discover-timeout diagnostics
  preserve primary timeout precedence and append cleanup timeout/failure
  operation context while carrying both custom PS + worker service-type context.
- Result: discover-timeout cleanup diagnostics now explicitly cover both indexed
  and non-indexed custom-service-types paths across runner + RunConfig +
  RunnerConfig entrypoints.

### 247) Worker-timeout custom-service-types non-index parity expanded
- Added runner-level custom non-index worker-timeout cleanup regressions:
  - `test_run_distributed_preserves_worker_timeout_with_custom_service_types_when_cleanup_steps_fail`
  - `test_run_distributed_preserves_worker_error_with_custom_service_types_when_cleanup_steps_timeout`
- Added RunConfig integration custom non-index worker-timeout cleanup
  regressions:
  - `distributed_runner_from_run_config_preserves_worker_timeout_with_custom_service_types_when_cleanup_fails`
  - `distributed_runner_from_run_config_preserves_worker_timeout_with_custom_service_types_when_cleanup_times_out`
- Added RunnerConfig integration custom non-index worker-timeout cleanup
  regressions:
  - `distributed_runner_from_runner_config_preserves_worker_timeout_with_custom_service_types_when_cleanup_fails`
  - `distributed_runner_from_runner_config_preserves_worker_timeout_with_custom_service_types_when_cleanup_times_out`
- New assertions verify custom non-index worker-timeout diagnostics preserve
  primary timeout precedence and append cleanup timeout/failure operation
  context while carrying both custom PS + worker service-type context.
- Result: worker-timeout cleanup diagnostics now explicitly cover indexed and
  non-indexed custom-service-types paths across runner + RunConfig +
  RunnerConfig entrypoints.

### 248) Ordering-issue custom-service-types non-index parity expanded
- Added runner-level custom-service-types non-index ordering-issue regressions:
  - `test_run_distributed_preserves_worker_ordering_issue_timeout_with_custom_service_types_when_cleanup_steps_timeout`
  - `test_run_distributed_preserves_worker_ordering_issue_timeout_with_custom_service_types_when_cleanup_steps_fail`
- Added RunConfig integration custom-service-types non-index ordering-issue
  regressions:
  - `distributed_runner_from_run_config_preserves_worker_ordering_issue_timeout_with_custom_service_types_when_cleanup_times_out`
  - `distributed_runner_from_run_config_preserves_worker_ordering_issue_timeout_with_custom_service_types_when_cleanup_fails`
- Added RunnerConfig integration custom-service-types non-index ordering-issue
  regressions:
  - `distributed_runner_from_runner_config_preserves_worker_ordering_issue_timeout_with_custom_service_types_when_cleanup_times_out`
  - `distributed_runner_from_runner_config_preserves_worker_ordering_issue_timeout_with_custom_service_types_when_cleanup_fails`
- New assertions verify custom-service-types non-index ordering-issue timeout
  diagnostics preserve primary timeout precedence and append cleanup
  timeout/failure operation context across runner/config entrypoints.
- Result: ordering-issue cleanup diagnostics now explicitly cover indexed and
  non-indexed custom-service-types paths across runner + RunConfig +
  RunnerConfig entrypoints.

### 249) Ordering+discover-error + last-discover-error custom-service-types non-index parity expanded
- Added runner-level custom-service-types non-index ordering+discover-error
  timeout cleanup regressions:
  - `test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_with_custom_service_types_when_cleanup_steps_timeout`
  - `test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_with_custom_service_types_when_cleanup_steps_fail`
- Added RunConfig integration custom-service-types non-index
  ordering+discover-error timeout cleanup regressions:
  - `distributed_runner_from_run_config_preserves_worker_ordering_and_discovery_error_timeout_with_custom_service_types_when_cleanup_times_out`
  - `distributed_runner_from_run_config_preserves_worker_ordering_and_discovery_error_timeout_with_custom_service_types_when_cleanup_fails`
- Added RunnerConfig integration custom-service-types non-index
  ordering+discover-error timeout cleanup regressions:
  - `distributed_runner_from_runner_config_preserves_worker_ordering_and_discovery_error_timeout_with_custom_service_types_when_cleanup_times_out`
  - `distributed_runner_from_runner_config_preserves_worker_ordering_and_discovery_error_timeout_with_custom_service_types_when_cleanup_fails`
- Added RunConfig + RunnerConfig custom-service-types non-index
  `last discovery error` cleanup regressions:
  - `distributed_runner_from_run_config_preserves_last_discover_error_with_custom_service_types_when_cleanup_times_out`
  - `distributed_runner_from_run_config_preserves_last_discover_error_with_custom_service_types_when_cleanup_fails`
  - `distributed_runner_from_runner_config_preserves_last_discover_error_with_custom_service_types_when_cleanup_times_out`
  - `distributed_runner_from_runner_config_preserves_last_discover_error_with_custom_service_types_when_cleanup_fails`
- Result: custom-service-types parity matrix for non-index worker paths now
  includes ordering+discover-error and last-discover-error cleanup timeout/fail
  diagnostics across runner + RunConfig + RunnerConfig entrypoints.

### 250) Runner register-failure custom-service-type non-index naming parity completed
- Added runner-level custom-service-type non-index register-failure aliases to
  complete indexed/non-index naming matrix coverage:
  - `test_run_distributed_worker_registration_failure_with_custom_service_type_includes_cleanup_context`
  - `test_run_distributed_ps_registration_failure_with_custom_service_type_includes_cleanup_context`
- New regressions mirror existing custom-service-type non-index semantics and
  assert register-failure primary precedence plus appended cleanup issue context
  (`deregister ... missing`) for worker and ps roles.
- Result: no remaining
  `with_custom_service_type_and_index -> with_custom_service_type` counterpart
  naming gaps remain in `runner.rs`.

### 251) Default-service non-index ordering+discover + last-discover parity expanded
- Added runner-level default-service non-index ordering+discover timeout
  cleanup regressions:
  - `test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_with_default_service_type_when_cleanup_steps_timeout`
  - `test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_with_default_service_type_when_cleanup_steps_fail`
- Added RunConfig integration default-service non-index regressions:
  - ordering+discover timeout cleanup:
    - `distributed_runner_from_run_config_preserves_worker_ordering_and_discovery_error_timeout_with_default_service_type_when_cleanup_times_out`
    - `distributed_runner_from_run_config_preserves_worker_ordering_and_discovery_error_timeout_with_default_service_type_when_cleanup_fails`
  - last-discover-error cleanup:
    - `distributed_runner_from_run_config_preserves_last_discover_error_with_default_service_type_when_cleanup_times_out`
    - `distributed_runner_from_run_config_preserves_last_discover_error_with_default_service_type_when_cleanup_fails`
- Added RunnerConfig integration default-service non-index regressions:
  - ordering+discover timeout cleanup:
    - `distributed_runner_from_runner_config_preserves_worker_ordering_and_discovery_error_timeout_with_default_service_type_when_cleanup_times_out`
    - `distributed_runner_from_runner_config_preserves_worker_ordering_and_discovery_error_timeout_with_default_service_type_when_cleanup_fails`
  - last-discover-error cleanup:
    - `distributed_runner_from_runner_config_preserves_last_discover_error_with_default_service_type_when_cleanup_times_out`
    - `distributed_runner_from_runner_config_preserves_last_discover_error_with_default_service_type_when_cleanup_fails`
- Result: default-service non-index parity now explicitly covers both
  ordering+discover composite timeout and last-discover-error cleanup
  timeout/failure paths across runner + RunConfig + RunnerConfig entrypoints.

### 252) Default-service non-index discover-timeout cleanup parity expanded
- Added runner-level default-service non-index discover-timeout regressions:
  - `test_run_distributed_worker_discover_timeout_preserves_error_when_cleanup_times_out_with_default_service_type`
  - `test_run_distributed_worker_discover_timeout_preserves_error_when_cleanup_fails_with_default_service_type`
- Added RunConfig integration default-service non-index discover-timeout
  regressions:
  - `distributed_runner_from_run_config_preserves_discover_timeout_with_default_service_type_when_cleanup_times_out`
  - `distributed_runner_from_run_config_preserves_discover_timeout_with_default_service_type_when_cleanup_fails`
- Added RunnerConfig integration default-service non-index discover-timeout
  regressions:
  - `distributed_runner_from_runner_config_preserves_discover_timeout_with_default_service_type_when_cleanup_times_out`
  - `distributed_runner_from_runner_config_preserves_discover_timeout_with_default_service_type_when_cleanup_fails`
- Result: discover-timeout cleanup timeout/failure diagnostics now explicitly
  cover default-service non-index worker paths across runner + RunConfig +
  RunnerConfig entrypoints.

### 253) Default-service non-index worker-timeout cleanup parity expanded
- Added runner-level default-service non-index worker-timeout regressions:
  - `test_run_distributed_preserves_worker_timeout_with_default_service_type_when_cleanup_steps_fail`
  - `test_run_distributed_preserves_worker_error_with_default_service_type_when_cleanup_steps_timeout`
- Added RunConfig integration default-service non-index worker-timeout
  regressions:
  - `distributed_runner_from_run_config_preserves_worker_timeout_with_default_service_type_when_cleanup_fails`
  - `distributed_runner_from_run_config_preserves_worker_timeout_with_default_service_type_when_cleanup_times_out`
- Added RunnerConfig integration default-service non-index worker-timeout
  regressions:
  - `distributed_runner_from_runner_config_preserves_worker_timeout_with_default_service_type_when_cleanup_fails`
  - `distributed_runner_from_runner_config_preserves_worker_timeout_with_default_service_type_when_cleanup_times_out`
- Result: worker-timeout cleanup timeout/failure diagnostics now explicitly
  cover default-service non-index worker paths across runner + RunConfig +
  RunnerConfig entrypoints.

### 254) Default-service non-index worker-ordering cleanup parity expanded
- Added runner-level default-service non-index worker-ordering-timeout
  regressions:
  - `test_run_distributed_preserves_worker_ordering_issue_timeout_with_default_service_type_when_cleanup_steps_timeout`
  - `test_run_distributed_preserves_worker_ordering_issue_timeout_with_default_service_type_when_cleanup_steps_fail`
- Added RunConfig integration default-service non-index worker-ordering-timeout
  regressions:
  - `distributed_runner_from_run_config_preserves_worker_ordering_issue_timeout_with_default_service_type_when_cleanup_times_out`
  - `distributed_runner_from_run_config_preserves_worker_ordering_issue_timeout_with_default_service_type_when_cleanup_fails`
- Added RunnerConfig integration default-service non-index
  worker-ordering-timeout regressions:
  - `distributed_runner_from_runner_config_preserves_worker_ordering_issue_timeout_with_default_service_type_when_cleanup_times_out`
  - `distributed_runner_from_runner_config_preserves_worker_ordering_issue_timeout_with_default_service_type_when_cleanup_fails`
- Result: worker-ordering timeout cleanup timeout/failure diagnostics now
  explicitly cover default-service non-index worker paths across runner +
  RunConfig + RunnerConfig entrypoints.

### 255) Default-service non-index register parity matrix completed
- Added runner-level default-service non-index register-parity regressions:
  - registration-failure cleanup context:
    - `test_run_distributed_worker_registration_failure_with_default_service_type_includes_cleanup_context`
    - `test_run_distributed_ps_registration_failure_with_default_service_type_includes_cleanup_context`
  - register-failure cleanup-timeout:
    - `test_run_distributed_preserves_worker_register_failure_with_default_service_type_when_cleanup_steps_timeout`
    - `test_run_distributed_preserves_ps_register_failure_with_default_service_type_when_cleanup_steps_timeout`
  - register-timeout cleanup-timeout/failure:
    - `test_run_distributed_worker_register_timeout_preserves_error_when_cleanup_times_out_with_default_service_type`
    - `test_run_distributed_worker_register_timeout_preserves_error_when_cleanup_fails_with_default_service_type`
    - `test_run_distributed_ps_register_timeout_preserves_error_when_cleanup_times_out_with_default_service_type`
    - `test_run_distributed_ps_register_timeout_preserves_error_when_cleanup_fails_with_default_service_type`
- Added RunConfig integration default-service non-index register-parity tests:
  - worker/ps register-failure cleanup-timeout context:
    - `distributed_runner_from_run_config_preserves_worker_register_failure_with_default_service_type_cleanup_timeout_context`
    - `distributed_runner_from_run_config_preserves_ps_register_failure_with_default_service_type_cleanup_timeout_context`
  - worker/ps register-timeout cleanup failure/timeout:
    - `distributed_runner_from_run_config_preserves_register_timeout_with_default_service_type_disconnect_failure_context`
    - `distributed_runner_from_run_config_preserves_register_timeout_with_default_service_type_when_cleanup_blocks`
    - `distributed_runner_from_run_config_preserves_ps_register_timeout_with_default_service_type_disconnect_failure_context`
    - `distributed_runner_from_run_config_preserves_ps_register_timeout_with_default_service_type_when_cleanup_blocks`
- Added RunnerConfig integration default-service non-index register-parity
  tests:
  - worker/ps register-failure cleanup-timeout context:
    - `distributed_runner_from_runner_config_preserves_worker_register_failure_with_default_service_type_cleanup_timeout_context`
    - `distributed_runner_from_runner_config_preserves_ps_register_failure_with_default_service_type_cleanup_timeout_context`
  - worker/ps register-timeout cleanup failure/timeout:
    - `distributed_runner_from_runner_config_preserves_register_timeout_with_default_service_type_disconnect_failure_context`
    - `distributed_runner_from_runner_config_preserves_register_timeout_with_default_service_type_when_cleanup_blocks`
    - `distributed_runner_from_runner_config_preserves_ps_register_timeout_with_default_service_type_disconnect_failure_context`
    - `distributed_runner_from_runner_config_preserves_ps_register_timeout_with_default_service_type_when_cleanup_blocks`
- Result: indexed→non-index register parity gaps are fully closed in
  `native_training_parity.rs`, and runner default-service indexed→non-index
  gaps are reduced to connect/discover-failure path aliases only.

### 256) Default-service non-index connect/discover parity aliases completed
- Added runner-level default-service non-index connect-timeout cleanup aliases:
  - `test_run_distributed_connect_timeout_preserves_error_when_disconnect_cleanup_fails_with_default_service_type`
  - `test_run_distributed_connect_timeout_preserves_error_when_disconnect_cleanup_times_out_with_default_service_type`
  - `test_run_distributed_ps_connect_timeout_preserves_error_when_disconnect_cleanup_fails_with_default_service_type`
  - `test_run_distributed_ps_connect_timeout_preserves_error_when_disconnect_cleanup_times_out_with_default_service_type`
- Added runner-level default-service non-index connect-failure cleanup aliases:
  - `test_run_distributed_connect_failure_does_not_hang_when_disconnect_blocks_with_default_service_type`
  - `test_run_distributed_ps_connect_failure_does_not_hang_when_disconnect_blocks_with_default_service_type`
  - `test_run_distributed_returns_connect_error_when_connect_and_disconnect_fail_with_default_service_type`
  - `test_run_distributed_returns_ps_connect_error_when_connect_and_disconnect_fail_with_default_service_type`
- Added runner-level default-service non-index worker-discover-failure cleanup
  aliases:
  - `test_run_distributed_preserves_worker_discover_failure_with_default_service_type_when_cleanup_steps_timeout`
  - `test_run_distributed_preserves_worker_discover_failure_with_default_service_type_when_cleanup_steps_fail`
- Result: indexed→non-index counterpart gap audit now reports zero missing
  aliases in both `runner.rs` and `native_training_parity.rs`.

### 257) Worker discovery error timeout/failure naming parity completed
- Added RunConfig integration cleanup-failure aliases for worker discovery error
  timeout semantics:
  - `distributed_runner_from_run_config_preserves_worker_discovery_error_when_cleanup_fails`
  - `distributed_runner_from_run_config_propagates_custom_discover_service_type_into_worker_discovery_error_when_cleanup_fails`
- Added RunnerConfig integration cleanup-failure aliases for worker discovery
  error timeout semantics:
  - `distributed_runner_from_runner_config_preserves_worker_discovery_error_when_cleanup_fails`
  - `distributed_runner_from_runner_config_propagates_custom_discover_service_type_into_worker_discovery_error_when_cleanup_fails`
- Added RunConfig + RunnerConfig integration cleanup-timeout aliases for generic
  worker timeout semantics:
  - `distributed_runner_from_run_config_preserves_worker_timeout_when_cleanup_times_out`
  - `distributed_runner_from_runner_config_preserves_worker_timeout_when_cleanup_times_out`
- Result: cleanup timeout↔failure naming symmetry for worker timeout/discovery
  integration tests is now complete (`_when_cleanup_times_out` and
  `_when_cleanup_fails` counterpart audit returns zero missing).

### 258) Runner worker-timeout cleanup-step timeout aliases completed
- Added runner-level worker-timeout cleanup-timeout aliases for all existing
  cleanup-failure naming variants:
  - `test_run_distributed_preserves_worker_timeout_when_cleanup_steps_timeout`
  - `test_run_distributed_preserves_worker_timeout_with_custom_service_types_and_index_when_cleanup_steps_timeout`
  - `test_run_distributed_preserves_worker_timeout_with_custom_service_type_when_cleanup_steps_timeout`
  - `test_run_distributed_preserves_worker_timeout_with_custom_service_types_when_cleanup_steps_timeout`
  - `test_run_distributed_preserves_worker_timeout_with_default_service_type_and_index_when_cleanup_steps_timeout`
  - `test_run_distributed_preserves_worker_timeout_with_default_service_type_when_cleanup_steps_timeout`
- Added shared helper
  `assert_worker_timeout_cleanup_timeout_case(...)` to keep assertions and
  diagnostics consistent across new timeout alias tests.
- Result: runner counterpart audit now reports zero missing aliases in the
  `_when_cleanup_steps_fail -> _when_cleanup_steps_timeout` direction.

### 259) Runner cleanup-step timeout/failure counterpart matrix fully closed
- Added runner-level cleanup-failure aliases for worker register-failure
  naming gaps:
  - `test_run_distributed_preserves_worker_register_failure_when_cleanup_steps_fail`
  - `test_run_distributed_preserves_worker_register_failure_with_custom_service_type_when_cleanup_steps_fail`
  - `test_run_distributed_preserves_worker_register_failure_with_custom_service_type_and_index_when_cleanup_steps_fail`
  - `test_run_distributed_preserves_worker_register_failure_with_default_service_type_and_index_when_cleanup_steps_fail`
  - `test_run_distributed_preserves_worker_register_failure_with_default_service_type_when_cleanup_steps_fail`
- Added runner-level cleanup-failure aliases for PS register-failure naming
  gaps:
  - `test_run_distributed_preserves_ps_register_failure_when_cleanup_steps_fail`
  - `test_run_distributed_preserves_ps_register_failure_with_custom_service_type_when_cleanup_steps_fail`
  - `test_run_distributed_preserves_ps_register_failure_with_custom_service_type_and_index_when_cleanup_steps_fail`
  - `test_run_distributed_preserves_ps_register_failure_with_default_service_type_and_index_when_cleanup_steps_fail`
  - `test_run_distributed_preserves_ps_register_failure_with_default_service_type_when_cleanup_steps_fail`
- Added runner-level cleanup-failure aliases for worker-timeout naming gaps:
  - `test_run_distributed_preserves_worker_error_when_cleanup_steps_fail`
  - `test_run_distributed_preserves_worker_error_with_custom_discovery_service_type_when_cleanup_steps_fail`
  - `test_run_distributed_preserves_worker_error_with_custom_service_types_and_index_when_cleanup_steps_fail`
  - `test_run_distributed_preserves_worker_error_with_custom_service_type_when_cleanup_steps_fail`
  - `test_run_distributed_preserves_worker_error_with_custom_service_types_when_cleanup_steps_fail`
  - `test_run_distributed_preserves_worker_error_with_default_service_type_and_index_when_cleanup_steps_fail`
  - `test_run_distributed_preserves_worker_error_with_default_service_type_when_cleanup_steps_fail`
- Added shared helper assertions to keep diagnostics/counter checks consistent:
  - `assert_register_failure_cleanup_fail_case(...)`
  - `assert_worker_error_cleanup_fail_case(...)`
- Result: runner `_when_cleanup_steps_timeout` ↔ `_when_cleanup_steps_fail`
  counterpart audit now reports zero missing aliases in both directions.

### 260) Runner PS register-timeout cleanup timeout/failure symmetry closed
- Added runner-level plain cleanup-failure counterpart:
  - `test_run_distributed_ps_register_timeout_preserves_error_when_cleanup_fails`
- This closes the final `_when_cleanup_times_out` naming gap for the plain
  PS register-timeout case (`ps-0` / default service type) alongside existing
  default/custom indexed/non-indexed cleanup-failure variants.
- Result: runner `_when_cleanup_times_out` ↔ `_when_cleanup_fails` counterpart
  audit now reports zero missing aliases in both directions.

### 261) Run/runner connect-timeout naming alias parity expanded
- Added RunConfig default connect-timeout alias:
  - `distributed_runner_from_run_config_preserves_connect_timeout_disconnect_failure_context`
- Added RunnerConfig default connect-timeout alias:
  - `distributed_runner_from_runner_config_preserves_connect_timeout_disconnect_failure_context`
- Result: both config entrypoints now expose non-`with_` alias coverage for the
  default worker connect-timeout cleanup-block path, matching surrounding
  naming conventions used by existing disconnect-failure-context variants.

### 262) Run/runner register-timeout and PS-register timeout alias parity completed
- Added RunConfig cleanup-block alias names:
  - `distributed_runner_from_run_config_preserves_register_timeout_with_custom_service_type_and_index_disconnect_failure_context`
  - `distributed_runner_from_run_config_preserves_ps_register_timeout_disconnect_failure_context`
  - `distributed_runner_from_run_config_preserves_ps_register_timeout_with_custom_service_type_and_index_disconnect_failure_context`
- Added RunnerConfig cleanup-block alias names:
  - `distributed_runner_from_runner_config_preserves_register_timeout_with_custom_service_type_and_index_disconnect_failure_context`
  - `distributed_runner_from_runner_config_preserves_ps_register_timeout_disconnect_failure_context`
  - `distributed_runner_from_runner_config_preserves_ps_register_timeout_with_custom_service_type_and_index_disconnect_failure_context`
- Result: `_when_cleanup_blocks -> _disconnect_failure_context` counterpart
  audit in `native_training_parity.rs` now reports zero missing aliases.

### 263) Connect-failure cleanup-context alias parity expanded
- Added RunConfig alias:
  - `distributed_runner_from_run_config_preserves_connect_failure_with_cleanup_context`
- Added RunnerConfig alias:
  - `distributed_runner_from_runner_config_preserves_connect_failure_with_cleanup_context`
- Result: default connect-failure cleanup-timeout-context naming now has direct
  cleanup-context aliases across both config entrypoints, reducing
  `_cleanup_timeout_context -> _cleanup_context` gaps in integration parity
  audits.

### 264) Custom connect-failure cleanup-context alias parity expanded
- Added RunConfig custom connect-failure cleanup-context aliases:
  - `distributed_runner_from_run_config_preserves_connect_failure_with_custom_service_type_cleanup_context`
  - `distributed_runner_from_run_config_preserves_connect_failure_with_custom_service_type_and_index_cleanup_context`
- Added RunnerConfig custom connect-failure cleanup-context aliases:
  - `distributed_runner_from_runner_config_preserves_connect_failure_with_custom_service_type_cleanup_context`
  - `distributed_runner_from_runner_config_preserves_connect_failure_with_custom_service_type_and_index_cleanup_context`
- Result: integration counterpart audit for
  `_cleanup_timeout_context -> _cleanup_context` improved from 22 missing to 18
  missing aliases.

### 265) Default connect-failure cleanup-context alias parity expanded
- Added RunConfig default connect-failure cleanup-context aliases:
  - `distributed_runner_from_run_config_preserves_default_worker_connect_failure_with_cleanup_context`
  - `distributed_runner_from_run_config_preserves_default_worker_connect_failure_with_index_cleanup_context`
  - `distributed_runner_from_run_config_preserves_default_ps_connect_failure_with_cleanup_context`
  - `distributed_runner_from_run_config_preserves_default_ps_connect_failure_with_index_cleanup_context`
- Added RunnerConfig default connect-failure cleanup-context aliases:
  - `distributed_runner_from_runner_config_preserves_default_worker_connect_failure_with_cleanup_context`
  - `distributed_runner_from_runner_config_preserves_default_worker_connect_failure_with_index_cleanup_context`
  - `distributed_runner_from_runner_config_preserves_default_ps_connect_failure_with_cleanup_context`
  - `distributed_runner_from_runner_config_preserves_default_ps_connect_failure_with_index_cleanup_context`
- Result: integration counterpart audit for
  `_cleanup_timeout_context -> _cleanup_context` improved from 18 missing to 10
  missing aliases.

### 266) PS connect/register cleanup-context alias parity completed
- Added RunConfig cleanup-context aliases:
  - `distributed_runner_from_run_config_preserves_ps_connect_failure_with_cleanup_context`
  - `distributed_runner_from_run_config_preserves_ps_connect_failure_with_custom_service_type_cleanup_context`
  - `distributed_runner_from_run_config_preserves_ps_connect_failure_with_custom_service_type_and_index_cleanup_context`
  - `distributed_runner_from_run_config_preserves_worker_register_failure_with_cleanup_context`
  - `distributed_runner_from_run_config_preserves_ps_register_failure_with_cleanup_context`
- Added RunnerConfig cleanup-context aliases:
  - `distributed_runner_from_runner_config_preserves_ps_connect_failure_with_cleanup_context`
  - `distributed_runner_from_runner_config_preserves_ps_connect_failure_with_custom_service_type_cleanup_context`
  - `distributed_runner_from_runner_config_preserves_ps_connect_failure_with_custom_service_type_and_index_cleanup_context`
  - `distributed_runner_from_runner_config_preserves_worker_register_failure_with_cleanup_context`
  - `distributed_runner_from_runner_config_preserves_ps_register_failure_with_cleanup_context`
- Result: `_cleanup_timeout_context -> _cleanup_context` integration alias audit
  is now fully closed (`TOTAL 0` missing counterparts).

### 267) Worker register disconnect-context alias parity started
- Added RunConfig alias:
  - `distributed_runner_from_run_config_preserves_worker_register_failure_with_disconnect_failure_context`
- Added RunnerConfig alias:
  - `distributed_runner_from_runner_config_preserves_worker_register_failure_with_disconnect_failure_context`
- Result: exploratory counterpart audit for
  `_cleanup_context -> _disconnect_failure_context` register-failure naming
  gaps reduced from 20 to 18.

### 268) PS register disconnect-context alias parity expanded
- Added RunConfig alias:
  - `distributed_runner_from_run_config_preserves_ps_register_failure_with_disconnect_failure_context`
- Added RunnerConfig alias:
  - `distributed_runner_from_runner_config_preserves_ps_register_failure_with_disconnect_failure_context`
- Result: exploratory counterpart audit for
  `_cleanup_context -> _disconnect_failure_context` register-failure naming
  gaps reduced from 18 to 16.

### 269) RunConfig worker-register disconnect-context variant parity expanded
- Added RunConfig aliases:
  - `distributed_runner_from_run_config_preserves_worker_register_failure_with_custom_service_type_disconnect_failure_context`
  - `distributed_runner_from_run_config_preserves_worker_register_failure_with_custom_service_type_and_index_disconnect_failure_context`
  - `distributed_runner_from_run_config_preserves_worker_register_failure_with_default_service_type_disconnect_failure_context`
  - `distributed_runner_from_run_config_preserves_worker_register_failure_with_default_service_type_and_index_disconnect_failure_context`
- Result: exploratory counterpart audit for
  `_cleanup_context -> _disconnect_failure_context` register-failure naming
  gaps reduced from 16 to 12.

### 270) RunnerConfig worker-register disconnect-context variant parity expanded
- Added RunnerConfig aliases:
  - `distributed_runner_from_runner_config_preserves_worker_register_failure_with_custom_service_type_disconnect_failure_context`
  - `distributed_runner_from_runner_config_preserves_worker_register_failure_with_custom_service_type_and_index_disconnect_failure_context`
  - `distributed_runner_from_runner_config_preserves_worker_register_failure_with_default_service_type_disconnect_failure_context`
  - `distributed_runner_from_runner_config_preserves_worker_register_failure_with_default_service_type_and_index_disconnect_failure_context`
- Result: exploratory counterpart audit for
  `_cleanup_context -> _disconnect_failure_context` register-failure naming
  gaps reduced from 12 to 8.

### 271) PS register disconnect-context variant parity completed
- Added RunConfig aliases:
  - `distributed_runner_from_run_config_preserves_ps_register_failure_with_custom_service_type_disconnect_failure_context`
  - `distributed_runner_from_run_config_preserves_ps_register_failure_with_custom_service_type_and_index_disconnect_failure_context`
  - `distributed_runner_from_run_config_preserves_ps_register_failure_with_default_service_type_disconnect_failure_context`
  - `distributed_runner_from_run_config_preserves_ps_register_failure_with_default_service_type_and_index_disconnect_failure_context`
- Added RunnerConfig aliases:
  - `distributed_runner_from_runner_config_preserves_ps_register_failure_with_custom_service_type_disconnect_failure_context`
  - `distributed_runner_from_runner_config_preserves_ps_register_failure_with_custom_service_type_and_index_disconnect_failure_context`
  - `distributed_runner_from_runner_config_preserves_ps_register_failure_with_default_service_type_disconnect_failure_context`
  - `distributed_runner_from_runner_config_preserves_ps_register_failure_with_default_service_type_and_index_disconnect_failure_context`
- Result: exploratory counterpart audit for
  `_cleanup_context -> _disconnect_failure_context` naming now fully closes
  (`missing 0`).

### 272) Disconnect-to-cleanup counterpart parity completed
- Added RunConfig cleanup-context counterparts for disconnect-failure-context
  naming family:
  - connect-timeout variants (`connect_timeout*`, `ps_connect_timeout*`)
  - register-timeout variants (`register_timeout*`, `ps_register_timeout*`)
  - post-success cleanup (`deregister_failure_with_*_after_success`)
- Added RunnerConfig cleanup-context counterparts for the same naming family.
- Implementation approach:
  - Added direct `#[test]` alias wrappers that invoke the existing
    `*_disconnect_failure_context*` test functions, avoiding async-nesting
    pitfalls while preserving validated execution paths.
- Result: `_disconnect_failure_context -> _cleanup_context` parity audit now
  fully closes (`missing 0`), and key cleanup naming families are symmetric in
  both directions.

### 273) `with_index` counterpart naming parity completed
- Added RunConfig no-index alias counterparts for default-* with-index test names:
  - default connect-timeout variants
  - default worker/ps connect-failure variants (`cleanup_context`,
    `cleanup_timeout_context`, `disconnect_failure_context`)
  - default ps connect-timeout variants
- Added RunnerConfig no-index alias counterparts for the same default-* with-index
  families.
- Implementation approach:
  - Added direct `#[test]` alias wrappers calling existing `*_with_index_*`
    tests to keep behavior identical and avoid async test nesting.
- Result: exploratory naming audit for `with_index -> (no with_index)` is fully
  closed (`missing 0`).

### 274) Cleanup-to-cleanup-timeout counterpart parity completed
- Added RunConfig `*_cleanup_timeout_context` alias wrappers for all missing
  `*_cleanup_context` counterparts across:
  - connect-timeout / ps-connect-timeout
  - register-timeout / ps-register-timeout
  - post-success deregister cleanup diagnostics
- Added RunnerConfig `*_cleanup_timeout_context` alias wrappers for the same
  families.
- Implementation approach:
  - Added direct `#[test]` wrappers invoking existing `*_cleanup_context` tests
    to preserve behavior and maintain async safety.
- Result: `_cleanup_context -> _cleanup_timeout_context` naming audit now fully
  closes (`missing 0`).

### 275) Runner registration-failure cleanup naming parity completed
- Added runner unit-test alias wrappers in `runner.rs` to close remaining
  registration-failure naming symmetry gaps:
  - `_cleanup_context -> _disconnect_failure_context`
  - `_cleanup_context -> _cleanup_timeout_context`
- Scope covered:
  - PS registration failure variants (default/custom service types, with/without
    index)
  - Worker registration failure variants (default/custom service types,
    with/without index)
- Result: runner unit-test cleanup naming audits are fully closed for both
  transformations (`missing 0`).

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
14. `cargo test -p monolith-training -q` ✅ (post runtime wiring of env-export helper)
15. `cargo test --workspace -q` ✅ (post latest runner-config + runner-utils parity updates)
16. `cargo test -p monolith-training -q` ✅ (post estimator `steps/max_steps` parity API updates)
17. `cargo test -p monolith-training -q` ✅ (post RunnerConfig-driven distributed runner entrypoint)
18. `cargo test -p monolith-training -q` ✅ (post `EstimatorSpec` parity surface + tests)
19. `cargo test -p monolith-training -q` ✅ (post distributed runner cleanup lifecycle updates)
20. `cargo test -p monolith-training -q` ✅ (post run/runner → estimator config bridge additions)
21. `cargo test -p monolith-training -q` ✅ (post checkpoint override edge-case hardening)
22. `cargo test -p monolith-training -q` ✅ (post `RunnerDiscovery` helper integration)
23. `cargo test -p monolith-training -q` ✅ (post estimator constructors from run/runner config)
24. `cargo test -p monolith-training -q` ✅ (post configurable restore sync timing integration)
25. `cargo test -p monolith-training -q` ✅ (post estimator distributed runtime helper)
26. `cargo test -p monolith-training -q` ✅ (post native-training integration parity test expansion)
27. `cargo test --workspace -q` ✅ (post latest estimator/runner parity and integration test additions)
28. `cargo test -p monolith-training -q` ✅ (post MonolithDiscoveryGuard operational helper expansion)
29. `cargo test -p monolith-training -q` ✅ (post runner_utils integration guard-operation tests)
30. `cargo test -p monolith-training -q` ✅ (post estimator runtime initialization helper)
31. `cargo test -p monolith-training -q` ✅ (post `get_discovery_from_run_config` helper integration)
32. `cargo test --workspace -q` ✅ (post latest runner/discovery and estimator runtime parity updates)
33. `cargo test -p monolith-training -q` ✅ (post `monolith_discovery_from_run_config` helper integration)
34. `cargo test -p monolith-training -q` ✅ (post initialized estimator constructor from run config)
35. `cargo test -p monolith-training -q` ✅ (post run-config distributed runner entrypoint)
36. `cargo test -p monolith-training -q` ✅ (post estimator run-config distributed runtime helper)
37. `cargo test --workspace -q` ✅ (post latest estimator/runner run-config convenience API additions)
38. `cargo test -p monolith-training -q` ✅ (post run-config restore initialization helper parity APIs)
39. `cargo test -p monolith-training -q` ✅ (post initialized estimator constructor from runner config)
40. `cargo test --workspace -q` ✅ (post latest run-config restore init and runner-config initializer parity updates)
41. `cargo test -p monolith-training -q` ✅ (post local-cluster blocking barrier parity helper)
42. `cargo test --workspace -q` ✅ (post local-cluster blocking barrier parity helper + repeated release semantics)
43. `cargo test -p monolith-training -q` ✅ (post PS lookup/apply latency stats tracking)
44. `cargo test -p monolith-training -q` ✅ (post failed-apply request accounting in PS stats)
45. `cargo test --workspace -q` ✅ (post PS latency + failed-apply stats parity updates)
46. `cargo test -p monolith-training -q` ✅ (post PS barrier semantics hardening)
47. `cargo test --workspace -q` ✅ (post PS barrier consistency + stats parity updates)
48. `cargo test -p monolith-training -q` ✅ (post generation-based PS barrier state + duplicate-worker guard)
49. `cargo test --workspace -q` ✅ (post generation-based PS barrier state and duplicate-worker timeout cleanup semantics)
50. `cargo test -p monolith-training -q` ✅ (post PS barrier worker-id range validation)
51. `cargo test --workspace -q` ✅ (post PS barrier worker-id validation and generation-state runtime checks)
52. `cargo test -p monolith-training -q` ✅ (post PS client parallel shard fanout implementation)
53. `cargo test --workspace -q` ✅ (post PS client parallel fanout runtime implementation and end-to-end shard test)
54. `cargo test -p monolith-training -q` ✅ (post PS client no-shard configuration guards)
55. `cargo test --workspace -q` ✅ (post PS client no-shard guards and additional client edge-case tests)
56. `cargo test -p monolith-training -q` ✅ (post PS client early input validation parity hardening)
57. `cargo test --workspace -q` ✅ (post PS client early input validation + dimension/barrier guard hardening)
58. `cargo test -p monolith-training -q` ✅ (post typed PS client barrier status-code error mapping)
59. `cargo test --workspace -q` ✅ (post typed PS client barrier error mapping and additional status-semantics tests)
60. `cargo test -p monolith-training -q` ✅ (post PS client shard/all health+stats API additions)
61. `cargo test --workspace -q` ✅ (post PS client health/stats API additions and shard validation coverage)
62. `cargo test -p monolith-training -q` ✅ (post PS client batch lookup/apply convenience API additions)
63. `cargo test --workspace -q` ✅ (post PS client batch lookup/apply additions and regression coverage)
64. `cargo test -p monolith-training -q` ✅ (post local-cluster barrier timeout cleanup + retry regression)
65. `cargo test -p monolith-training -q` ✅ (post batch-lookup duplicate found-flag regression coverage)
66. `cargo test -p monolith-training -q` ✅ (post explicit discovery-guard close lifecycle API + idempotence coverage)
67. `cargo test --workspace -q` ✅ (post distributed/runtime + discovery lifecycle parity hardening)
68. `cargo test -p monolith-training -q` ✅ (post batched PS per-entry dim validation + parallel batch fanout)
69. `cargo test --workspace -q` ✅ (post batched PS per-entry validation hardening and regression coverage)
70. `cargo test -p monolith-training -q` ✅ (post local-cluster released-barrier pruning + lagging-worker safety regression)
71. `cargo test --workspace -q` ✅ (post local-cluster released-barrier pruning and latest distributed/runtime parity hardening)
72. `cargo test -p monolith-training -q` ✅ (post immutable/concurrency-friendly PS client API refactor + parallel immutable lookup test)
73. `cargo test --workspace -q` ✅ (post immutable/concurrency-friendly PS client API refactor and caller cleanup)
74. `cargo test -p monolith-training -q` ✅ (post detailed PS client lookup/apply response metadata API additions)
76. `cargo test -p monolith-training -q` ✅ (post PsBarrier immutable-client wrapper alignment)
77. `cargo test -p monolith-core -q` ✅ (post path_utils source-path and env-race hardening)
78. `cargo test --workspace -q` ✅ (post path_utils robustness fix and full workspace regression)
79. `cargo test -p monolith-training -q` ✅ (post lock-free PsBarrier wrapper refactor and barrier-layer concurrency tests)
80. `cargo test -p monolith-training -q` ✅ (post shard-selectable PS barrier coordinator API + routing/index regressions)
81. `cargo test --workspace -q` ✅ (post shard-selectable barrier coordinator API and lock-free barrier wrapper updates)
82. `cargo test -p monolith-training -q` ✅ (post default health/stats client helpers + shard no-client guard parity)
83. `cargo test --workspace -q` ✅ (post default health/stats helper APIs and latest distributed/runtime parity updates)
84. `cargo test -p monolith-training -q` ✅ (post tuple-based lookup shard/local position mapping hardening)
85. `cargo test --workspace -q` ✅ (post latest PS client barrier/health/stats and lookup index hardening updates)
86. `cargo test -p monolith-training -q` ✅ (post typed barrier-layer error mapping and timeout/config regression coverage)
87. `cargo test --workspace -q` ✅ (post typed barrier-layer mapping and latest distributed/runtime PS client hardening)
88. `cargo test -p monolith-training -q` ✅ (post PsBarrier direct-connect convenience API + connect guard regression)
89. `cargo test --workspace -q` ✅ (post PsBarrier connect convenience API and latest barrier/distributed PS hardening)
90. `cargo test -p monolith-training -q` ✅ (post worker runtime PS client connection reuse optimization)
92. `cargo test -p monolith-training -q` ✅ (post configurable worker barrier timeout wiring in distributed runner)
91. `cargo test --workspace -q` ✅ (post worker PS client reuse optimization and latest barrier/distributed/runtime updates)
93. `cargo test -p monolith-cli -q` ✅ (post train CLI barrier-timeout flag plumbing for distributed runtime config)
94. `cargo test --workspace -q` ✅ (post train CLI/distributed runner barrier-timeout parity wiring and full regression rerun)
95. `cargo test -p monolith-training -q` ✅ (post run/runner-config propagation of distributed connect-retry/backoff/barrier-timeout fields)
96. `cargo test --workspace -q` ✅ (post run/runner-config distributed runtime tuning propagation and full workspace regression rerun)
97. `cargo test -p monolith-training -q` ✅ (post cancellable PS discovery heartbeat lifecycle management in distributed runner)
98. `cargo test --workspace -q` ✅ (post runner heartbeat lifecycle cancellation refactor and full workspace regression rerun)
99. `cargo test -p monolith-training -q` ✅ (post local-cluster start/stop lifecycle guard semantics and regression tests)
100. `cargo test --workspace -q` ✅ (post local-cluster lifecycle guard semantics and full workspace regression rerun)
101. `cargo test -p monolith-training -q` ✅ (post local PS/worker lifecycle guard parity hardening and new regression coverage)
102. `cargo test --workspace -q` ✅ (post local PS/worker lifecycle guard parity hardening and full workspace regression rerun)
103. `cargo test -p monolith-training -q` ✅ (post local-cluster running-state enforcement for register/train operations)
104. `cargo test --workspace -q` ✅ (post local-cluster running-state enforcement for register/train operations and full workspace regression rerun)
105. `cargo test -p monolith-training -q` ✅ (post cluster-config duplicate-address validation hardening)
106. `cargo test --workspace -q` ✅ (post cluster-config duplicate-address validation hardening and full workspace regression rerun)
107. `cargo test -p monolith-training -q` ✅ (post deterministic PS discovery ordering by shard index metadata + fallback sorting coverage)
108. `cargo test --workspace -q` ✅ (post deterministic PS discovery ordering by shard index metadata and full workspace regression rerun)
109. `cargo test -p monolith-training -q` ✅ (post contiguous PS shard-index enforcement for metadata-based worker discovery ordering)
110. `cargo test --workspace -q` ✅ (post contiguous PS shard-index enforcement and full workspace regression rerun)
111. `cargo test -p monolith-training -q` ✅ (post conflicting duplicate shard-index advertisement guard in worker discovery ordering)
112. `cargo test --workspace -q` ✅ (post conflicting duplicate shard-index advertisement guard and full workspace regression rerun)
113. `cargo test -p monolith-training -q` ✅ (post run/runner config propagation of discovery service-role names and table settings into distributed runtime config)
114. `cargo test --workspace -q` ✅ (post run/runner config discovery-role + table/dim propagation updates and full workspace regression rerun)
115. `cargo test -p monolith-training -q` ✅ (post strict mixed/invalid PS index-metadata consistency gating in worker discovery ordering)
116. `cargo test --workspace -q` ✅ (post strict mixed/invalid PS index-metadata consistency gating and full workspace regression rerun)
117. `cargo test -p monolith-training -q` ✅ (post distributed runtime config preflight validation guard and invalid-config regression)
118. `cargo test --workspace -q` ✅ (post distributed runtime preflight validation guard and full workspace regression rerun)
119. `cargo test -p monolith-training -q` ✅ (post typed PS discovery ordering diagnostics and retry-time ordering-issue propagation)
120. `cargo test --workspace -q` ✅ (post typed PS discovery ordering diagnostics and full workspace regression rerun)
121. `cargo test -p monolith-training -q` ✅ (post worker timeout diagnostics regression for mixed PS shard-index metadata inconsistency)
122. `cargo test --workspace -q` ✅ (post worker timeout diagnostics regression and full workspace rerun)
123. `cargo test -p monolith-training -q` ✅ (post ParameterSync replicator managed-task lifecycle cleanup and PS runner shutdown wiring)
124. `cargo test --workspace -q` ✅ (post ParameterSync replicator managed-task lifecycle cleanup and full workspace regression rerun)
125. `cargo test -p monolith-training -q` ✅ (post distributed runner cleanup hardening for worker registration failure path)
126. `cargo test --workspace -q` ✅ (post distributed runner cleanup hardening for worker registration failure path and full workspace rerun)
127. `cargo test -p monolith-training -q` ✅ (post worker discovery timeout diagnostic retention hardening across retry attempts)
128. `cargo test --workspace -q` ✅ (post worker discovery timeout diagnostic retention hardening and full workspace rerun)
129. `cargo test -p monolith-training -q` ✅ (post PS registration-failure cleanup regression coverage)
130. `cargo test --workspace -q` ✅ (post PS registration-failure cleanup regression coverage and full workspace rerun)
131. `cargo test -p monolith-training -q` ✅ (post worker timeout diagnostics enhancement retaining last discovery backend error across retries)
132. `cargo test --workspace -q` ✅ (post worker timeout diagnostics enhancement retaining last discovery backend error and full workspace rerun)
133. `cargo test -p monolith-training -q` ✅ (post combined ordering+discovery timeout diagnostics regression coverage)
134. `cargo test --workspace -q` ✅ (post combined ordering+discovery timeout diagnostics regression coverage and full workspace rerun)
135. `cargo test -p monolith-training -q` ✅ (post ParameterSync replicator drop-safety lifecycle guard and regression coverage)
136. `cargo test --workspace -q` ✅ (post ParameterSync replicator drop-safety lifecycle guard and full workspace rerun)
137. `cargo test -p monolith-training -q` ✅ (post one-shot join-handle ownership refinement for ParameterSync replicator task stop/drop lifecycle)
138. `cargo test --workspace -q` ✅ (post one-shot join-handle ownership refinement for ParameterSync replicator task stop/drop lifecycle and full workspace rerun)
139. `cargo test -p monolith-training -q` ✅ (post worker timeout diagnostics enhancement reporting max observed PS count across retries)
140. `cargo test --workspace -q` ✅ (post worker timeout diagnostics enhancement reporting max observed PS count and full workspace rerun)
141. `cargo test -p monolith-training -q` ✅ (post raw-vs-usable PS visibility diagnostics enhancement in worker discovery timeout path)
142. `cargo test --workspace -q` ✅ (post raw-vs-usable PS visibility diagnostics enhancement and full workspace rerun)
143. `cargo test -p monolith-training -q` ✅ (post worker timeout diagnostics enhancement reporting retry attempt count)
144. `cargo test --workspace -q` ✅ (post worker timeout diagnostics enhancement reporting retry attempt count and full workspace rerun)
145. `cargo test -p monolith-training -q` ✅ (post stale discovery-error cleanup on successful rediscovery attempts in worker timeout diagnostics)
146. `cargo test --workspace -q` ✅ (post stale discovery-error cleanup on successful rediscovery attempts and full workspace rerun)
147. `cargo test -p monolith-training -q` ✅ (post stale ordering-issue cleanup on usable rediscovery attempts in worker timeout diagnostics)
148. `cargo test --workspace -q` ✅ (post stale ordering-issue cleanup on usable rediscovery attempts and full workspace rerun)
149. `cargo test -p monolith-training -q` ✅ (post worker heartbeat lifecycle cleanup + deterministic worker timeout heartbeat-task shutdown)
150. `cargo test --workspace -q` ✅ (post worker heartbeat lifecycle cleanup and full workspace regression rerun)
151. `cargo test -p monolith-training -q` ✅ (post discovery watch poller lifecycle cleanup and unsubscribe-stop regressions)
152. `cargo test --workspace -q` ✅ (post discovery watch poller lifecycle cleanup and full workspace regression rerun)
153. `cargo test -p monolith-training -q` ✅ (post in-memory discovery dead-watcher sender cleanup)
154. `cargo test --workspace -q` ✅ (post in-memory discovery dead-watcher sender cleanup and full workspace regression rerun)
155. `cargo test -p monolith-training -q` ✅ (post bounded consul replacement-retry hardening and stale-registration timeout/idempotence regressions)
156. `cargo test --workspace -q` ✅ (post bounded consul replacement-retry hardening and full workspace regression rerun)
157. `cargo test -p monolith-training -q` ✅ (post ZK registration-thread lock cleanup and close-idempotence regression)
158. `cargo test --workspace -q` ✅ (post ZK registration-thread lock cleanup and full workspace regression rerun)
159. `cargo test -p monolith-training -q` ✅ (post MLP discovery parity expansion and host/query/filter regression coverage)
160. `cargo test --workspace -q` ✅ (post MLP discovery parity expansion and full workspace regression rerun)
161. `cargo test -p monolith-training -q` ✅ (post MLP discovery close-semantics lifecycle parity and close-state regression)
162. `cargo test -p monolith-core -q` ✅ (post model-registry test serialization lock against global-state races)
163. `cargo test --workspace -q` ✅ (post model-registry test-race stabilization and full workspace regression rerun)
164. `cargo test -p monolith-training -q` ✅ (post MLP query_all configured-role filtering parity tightening)
165. `cargo test --workspace -q` ✅ (post MLP query_all configured-role filtering parity tightening and full workspace regression rerun)
166. `cargo test -p monolith-training -q` ✅ (post strict Consul query_all malformed-entry validation and diagnostics hardening)
167. `cargo test --workspace -q` ✅ (post strict Consul query_all malformed-entry validation and full workspace regression rerun)
168. `cargo test -p monolith-training -q` ✅ (post ZK closed-state operation guards and after-close lifecycle regression)
169. `cargo test --workspace -q` ✅ (post ZK closed-state operation guards and full workspace regression rerun)
170. `cargo test -p monolith-training -q` ✅ (post runner-utils MLP guard close-lifecycle regression addition)
171. `cargo test --workspace -q` ✅ (post runner-utils MLP guard close-lifecycle regression addition and full workspace rerun)
172. `cargo test -p monolith-training -q` ✅ (post Consul closed-state lifecycle guard hardening)
173. `cargo test --workspace -q` ✅ (post Consul closed-state lifecycle guard hardening and full workspace regression rerun)
174. `cargo test -p monolith-training -q` ✅ (post Consul query_all close-state consistency guard)
175. `cargo test --workspace -q` ✅ (post Consul query_all close-state consistency guard and full workspace regression rerun)
176. `cargo test -p monolith-training -q` ✅ (post Consul clone close-state parity regression)
177. `cargo test --workspace -q` ✅ (post Consul clone close-state parity regression and full workspace regression rerun)
178. `cargo test -p monolith-training -q` ✅ (post worker-success heartbeat lifecycle shutdown regression)
179. `cargo test --workspace -q` ✅ (post worker-success heartbeat lifecycle shutdown regression and full workspace rerun)
180. `cargo test -p monolith-training -q` ✅ (post heartbeat stop timeout guard for blocking heartbeat calls)
181. `cargo test --workspace -q` ✅ (post heartbeat stop timeout guard for blocking heartbeat calls and full workspace rerun)
182. `cargo test -p monolith-training -q` ✅ (post stop-aware in-flight heartbeat cancellation refactor)
183. `cargo test --workspace -q` ✅ (post stop-aware in-flight heartbeat cancellation refactor and full workspace rerun)
184. `cargo test -p monolith-training -q` ✅ (post connect-failure discovery disconnect cleanup hardening)
185. `cargo test --workspace -q` ✅ (post connect-failure discovery disconnect cleanup hardening and full workspace rerun)
186. `cargo test -p monolith-training -q` ✅ (post forced heartbeat-task abort on shutdown-timeout path)
187. `cargo test --workspace -q` ✅ (post forced heartbeat-task abort on shutdown-timeout path and full workspace rerun)
188. `cargo test -p monolith-training -q` ✅ (post disconnect-after-success lifecycle regression for deregister-failure path)
189. `cargo test --workspace -q` ✅ (post disconnect-after-success lifecycle regression for deregister-failure path and full workspace rerun)
190. `cargo test -p monolith-training -q` ✅ (post worker-timeout cleanup regression after successful registration)
191. `cargo test --workspace -q` ✅ (post worker-timeout cleanup regression after successful registration and full workspace rerun)
192. `cargo test -p monolith-training -q` ✅ (post disconnect-failure propagation regression after successful worker completion)
193. `cargo test --workspace -q` ✅ (post disconnect-failure propagation regression after successful worker completion and full workspace rerun)
194. `cargo test -p monolith-training -q` ✅ (post combined cleanup-failure semantics hardening in distributed runner)
195. `cargo test --workspace -q` ✅ (post combined cleanup-failure semantics hardening in distributed runner and full workspace rerun)
196. `cargo test -p monolith-training -q` ✅ (post disconnect-timeout cleanup regression after successful worker completion)
197. `cargo test --workspace -q` ✅ (post disconnect-timeout cleanup regression after successful worker completion and full workspace rerun)
198. `cargo test -p monolith-training -q` ✅ (post worker-error precedence regression when cleanup steps time out)
199. `cargo test --workspace -q` ✅ (post worker-error precedence regression when cleanup steps time out and full workspace rerun)
200. `cargo test -p monolith-training -q` ✅ (post discovery-cleanup timeout wrapper + disconnect-timeout success-path regression additions)
201. `cargo test --workspace -q` ✅ (post discovery-cleanup timeout wrapper + disconnect-timeout success-path regression additions and full workspace rerun)
202. `cargo test -p monolith-training -q` ✅ (post parameter-sync replicator deterministic stop hardening + nonterminating-stop regression)
203. `cargo test --workspace -q` ✅ (post parameter-sync replicator deterministic stop hardening + nonterminating-stop regression and full workspace rerun)
204. `cargo test -p monolith-training -q` ✅ (post connect/register discovery-operation timeout hardening and blocking-operation regressions)
205. `cargo test --workspace -q` ✅ (post connect/register discovery-operation timeout hardening and blocking-operation regressions full workspace rerun)
206. `cargo test -p monolith-training -q` ✅ (post configurable discovery setup timeout field introduction and setup-timeout regression updates)
207. `cargo test -p monolith-cli -q` ✅ (post train CLI wiring for configurable discovery setup timeout)
208. `cargo test --workspace -q` ✅ (post configurable discovery setup timeout + CLI wiring updates full workspace rerun)
209. `cargo test -p monolith-training -q` ✅ (post configurable discovery cleanup-timeout support and bounded-delay regression)
210. `cargo test -p monolith-cli -q` ✅ (post CLI wiring for discovery cleanup-timeout field)
211. `cargo test --workspace -q` ✅ (post configurable discovery cleanup-timeout support + CLI wiring full workspace rerun)
212. `cargo test -p monolith-training -q` ✅ (post worker discover-operation timeout handling and retry-loop blocking regressions)
213. `cargo test --workspace -q` ✅ (post worker discover-operation timeout handling and retry-loop blocking regressions full workspace rerun)
214. `cargo test -p monolith-cli -q` ✅ (post train CLI discovery timeout flag exposure + config wiring updates)
215. `cargo test --workspace -q` ✅ (post train CLI discovery timeout flag exposure + config wiring updates full workspace rerun)
216. `cargo test -p monolith-training -q` ✅ (post run-config/runner-config discovery-timeout field propagation and mapping regressions)
217. `cargo test -p monolith-cli -q` ✅ (post run-config timeout-field expansion compatibility check against CLI command surfaces)
218. `cargo test --workspace -q` ✅ (post run-config/runner-config discovery-timeout field propagation full workspace rerun)
219. `cargo test -p monolith-training -q` ✅ (post distributed timeout validation guards and zero-timeout validation regressions)
220. `cargo test --workspace -q` ✅ (post distributed timeout validation guards and zero-timeout validation regressions full workspace rerun)
221. `cargo test -p monolith-training -q` ✅ (post timeout-diagnostics duration context enrichment across operation/cleanup timeout regressions)
222. `cargo test --workspace -q` ✅ (post timeout-diagnostics duration context enrichment full workspace rerun)
223. `cargo test -p monolith-training -q` ✅ (post blocked-cleanup precedence regressions for setup-stage connect/register timeout paths)
224. `cargo test --workspace -q` ✅ (post blocked-cleanup precedence regressions for setup-stage connect/register timeout paths full workspace rerun)
225. `cargo test -p monolith-training -q` ✅ (post PS register blocked-cleanup precedence regression addition)
226. `cargo test --workspace -q` ✅ (post PS register blocked-cleanup precedence regression addition full workspace rerun)
227. `cargo test -p monolith-cli -q` ✅ (post train distributed-config builder extraction + CLI mapping regressions)
228. `cargo test --workspace -q` ✅ (post train distributed-config builder extraction + CLI mapping regressions full workspace rerun)
229. `cargo test -p monolith-training -q` ✅ (post PS setup-timeout precedence regression for blocked cleanup path)
230. `cargo test --workspace -q` ✅ (post PS setup-timeout precedence regression for blocked cleanup path full workspace rerun)
231. `cargo test -p monolith-cli -q` ✅ (post CLI timeout input validation guardrails + builder regression expansion)
232. `cargo test --workspace -q` ✅ (post CLI timeout input validation guardrails + builder regression expansion full workspace rerun)
233. `cargo test -p monolith-training -q` ✅ (post run-config discover-timeout integration regression coverage)
234. `cargo test --workspace -q` ✅ (post run-config discover-timeout integration regression coverage full workspace rerun)
235. `cargo test -p monolith-training -q` ✅ (post run-config connect-timeout precedence integration regression under blocked cleanup)
236. `cargo test --workspace -q` ✅ (post run-config connect-timeout precedence integration regression under blocked cleanup full workspace rerun)
237. `cargo test -p monolith-training -q` ✅ (post runner-config connect-timeout precedence integration regression under blocked cleanup)
238. `cargo test --workspace -q` ✅ (post runner-config connect-timeout precedence integration regression under blocked cleanup full workspace rerun)
239. `cargo test -p monolith-training -q` ✅ (post run-config register-timeout precedence integration regression under blocked cleanup)
240. `cargo test --workspace -q` ✅ (post run-config register-timeout precedence integration regression under blocked cleanup full workspace rerun)
241. `cargo test -p monolith-training -q` ✅ (post runner-config register-timeout precedence integration regression under blocked cleanup)
242. `cargo test --workspace -q` ✅ (post runner-config register-timeout precedence integration regression under blocked cleanup full workspace rerun)
243. `cargo test -p monolith-training -q` ✅ (post PS register-timeout precedence integration regressions across run/runner config entrypoints)
244. `cargo test --workspace -q` ✅ (post PS register-timeout precedence integration regressions across run/runner config entrypoints full workspace rerun)
245. `cargo test -p monolith-training -q` ✅ (post runner-config discover-timeout propagation integration regression)
246. `cargo test --workspace -q` ✅ (post runner-config discover-timeout propagation integration regression full workspace rerun)
247. `cargo test -p monolith-training -q` ✅ (post discover-retry propagation integration regressions across run/runner config entrypoints)
248. `cargo test --workspace -q` ✅ (post discover-retry propagation integration regressions across run/runner config entrypoints full workspace rerun)
249. `cargo test -p monolith-training -q` ✅ (post retry-backoff propagation integration regressions across run/runner config entrypoints)
250. `cargo test --workspace -q` ✅ (post retry-backoff propagation integration regressions across run/runner config entrypoints full workspace rerun)
251. `cargo test -p monolith-training -q` ✅ (post custom discovery service-type propagation integration regressions across run/runner config entrypoints)
252. `cargo test --workspace -q` ✅ (post custom discovery service-type propagation integration regressions across run/runner config entrypoints full workspace rerun)
253. `cargo test -p monolith-cli -q && cargo test -p monolith-training -q` ✅ (post config-surface service-type mapping assertion hardening)
254. `cargo test --workspace -q` ✅ (post config-surface service-type mapping assertion hardening full workspace rerun)
255. `cargo test -p monolith-training -q` ✅ (post PS connect-timeout precedence integration regressions across run/runner config entrypoints)
256. `cargo test --workspace -q` ✅ (post PS connect-timeout precedence integration regressions across run/runner config entrypoints full workspace rerun)
257. `cargo test -p monolith-training -q` ✅ (post zero-timeout validation integration regressions across run/runner config entrypoints)
258. `cargo test --workspace -q` ✅ (post zero-timeout validation integration regressions across run/runner config entrypoints full workspace rerun)
259. `cargo test -p monolith-training -q` ✅ (post cleanup-timeout propagation integration regressions across run/runner config entrypoints)
260. `cargo test --workspace -q` ✅ (post cleanup-timeout propagation integration regressions across run/runner config entrypoints full workspace rerun)
261. `cargo test -p monolith-training -q` ✅ (post register-path cleanup-timeout propagation integration regressions across run/runner config entrypoints)
262. `cargo test --workspace -q` ✅ (post register-path cleanup-timeout propagation integration regressions across run/runner config entrypoints full workspace rerun)
263. `cargo test -p monolith-training -q` ✅ (post barrier-timeout propagation integration regressions across run/runner config entrypoints)
264. `cargo test --workspace -q` ✅ (post barrier-timeout propagation integration regressions across run/runner config entrypoints full workspace rerun)
265. `cargo test -p monolith-cli -q && cargo test -p monolith-training -q` ✅ (post barrier-timeout input validation coverage across CLI and config entrypoints)
266. `cargo test --workspace -q` ✅ (post barrier-timeout input validation coverage across CLI and config entrypoints full workspace rerun)
267. `cargo test -p monolith-cli -q && cargo test -p monolith-training -q` ✅ (post negative barrier-timeout validation coverage across CLI and config entrypoints)
268. `cargo test --workspace -q` ✅ (post negative barrier-timeout validation coverage across CLI and config entrypoints full workspace rerun)
269. `cargo test -p monolith-cli -q && cargo test -p monolith-training -q` ✅ (post distributed dim input validation coverage across CLI and config entrypoints)
270. `cargo test --workspace -q` ✅ (post distributed dim input validation coverage across CLI and config entrypoints full workspace rerun)
271. `cargo test -p monolith-training -q` ✅ (post worker-index timeout diagnostic propagation integrations across run/runner config entrypoints)
272. `cargo test --workspace -q` ✅ (post worker-index timeout diagnostic propagation integrations across run/runner config entrypoints full workspace rerun)
273. `cargo test -p monolith-training -q` ✅ (post ps-index timeout diagnostic propagation integrations across run/runner config entrypoints)
274. `cargo test --workspace -q` ✅ (post ps-index timeout diagnostic propagation integrations across run/runner config entrypoints full workspace rerun)
275. `cargo test -p monolith-training -q` ✅ (post register-timeout diagnostic enrichment with service-type context + integration regressions)
276. `cargo test --workspace -q` ✅ (post register-timeout diagnostic enrichment with service-type context full workspace rerun)
277. `cargo test -p monolith-training -q` ✅ (post discover-timeout diagnostic enrichment with queried service-type context + integration regressions)
278. `cargo test --workspace -q` ✅ (post discover-timeout diagnostic enrichment with queried service-type context full workspace rerun)
279. `cargo test -p monolith-training -q` ✅ (post runner-level discover timeout service-type diagnostic regression)
280. `cargo test --workspace -q` ✅ (post runner-level discover timeout service-type diagnostic regression full workspace rerun)
281. `cargo test -p monolith-training -q` ✅ (post distributed-runner discover timeout custom service-type regression)
282. `cargo test --workspace -q` ✅ (post distributed-runner discover timeout custom service-type regression full workspace rerun)
283. `cargo test -p monolith-training -q` ✅ (post distributed-runner register timeout custom service-type regressions)
284. `cargo test --workspace -q` ✅ (post distributed-runner register timeout custom service-type regressions full workspace rerun)
285. `cargo test -p monolith-training -q` ✅ (post connect-timeout diagnostic enrichment with service-type context + regression expansion)
286. `cargo test --workspace -q` ✅ (post connect-timeout diagnostic enrichment with service-type context full workspace rerun)
287. `cargo test -p monolith-training -q` ✅ (post cleanup-timeout diagnostic enrichment with service-type context + custom cleanup regressions)
288. `cargo test --workspace -q` ✅ (post cleanup-timeout diagnostic enrichment with service-type context full workspace rerun)
289. `cargo test -p monolith-cli -q && cargo test -p monolith-training -q` ✅ (post non-empty discovery service-type validation across CLI and runtime configs)
290. `cargo test --workspace -q` ⚠️ env-sensitive failure in `native_training::env_utils::tests::test_get_zk_auth_data_none` due inherited `ZK_AUTH`.
291. `unset ZK_AUTH && cargo test --workspace -q` ✅ (post non-empty discovery service-type validation across CLI/runtime configs full workspace rerun)
292. `cargo test -p monolith-cli -q && cargo test -p monolith-training -q` ✅ (post parameter-sync interval validation when targets are configured)
293. `unset ZK_AUTH && cargo test --workspace -q` ✅ (post parameter-sync interval validation full workspace rerun)
294. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post env-utils test isolation hardening)
295. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post env-utils test isolation hardening full workspace rerun under ambient ZK auth env)
296. `ZK_AUTH=user:pass cargo test -p monolith-cli -q && ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post parameter-sync target metadata validation hardening)
297. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post parameter-sync target metadata validation hardening full workspace rerun under ambient ZK auth env)
298. `ZK_AUTH=user:pass cargo test -p monolith-cli -q && ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post run/runner-config parameter-sync field parity wiring and validation regression restoration)
299. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post run/runner-config parameter-sync field parity wiring full workspace rerun under ambient ZK auth env)
300. `ZK_AUTH=user:pass cargo test -p monolith-cli -q && ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post run/runner-config cleanup-timeout custom service-type diagnostic parity regressions after successful worker runs)
301. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post run/runner-config cleanup-timeout custom service-type diagnostic parity updates full workspace rerun under ambient ZK auth env)
302. `ZK_AUTH=user:pass cargo test -p monolith-cli -q && ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post run/runner worker-discovery error precedence parity regressions under blocked cleanup)
303. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post run/runner worker-discovery error precedence parity regressions full workspace rerun under ambient ZK auth env)
304. `ZK_AUTH=user:pass cargo test -p monolith-cli -q && ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post worker discovery-timeout service-type diagnostic enrichment + run/runner propagation regressions)
305. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post worker discovery-timeout service-type diagnostic enrichment full workspace rerun under ambient ZK auth env)
306. `ZK_AUTH=user:pass cargo test -p monolith-cli -q && ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post distributed role-index range validation parity across CLI/runtime entrypoints)
307. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post distributed role-index range validation parity full workspace rerun under ambient ZK auth env)
308. `ZK_AUTH=user:pass cargo test -p monolith-cli -q && ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post non-empty distributed table-name validation parity across CLI/runtime config entrypoints)
309. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post non-empty distributed table-name validation parity full workspace rerun under ambient ZK auth env)
310. `ZK_AUTH=user:pass cargo test -p monolith-cli -q && ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post strict zero-count propagation in distributed config mapping + zero-size cluster validation regressions)
311. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post strict zero-count propagation and zero-size cluster validation full workspace rerun under ambient ZK auth env)
312. `ZK_AUTH=user:pass cargo test -p monolith-cli -q && ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post distinct PS/worker discovery service-type validation parity across CLI/runtime entrypoints)
313. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post distinct PS/worker discovery service-type validation full workspace rerun under ambient ZK auth env)
314. `ZK_AUTH=user:pass cargo test -p monolith-cli -q && ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post whitespace-padded discovery service-type validation parity across CLI/runtime entrypoints)
315. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post whitespace-padded discovery service-type validation full workspace rerun under ambient ZK auth env)
316. `ZK_AUTH=user:pass cargo test -p monolith-cli -q && ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post whitespace-padded distributed table-name validation parity across CLI/runtime entrypoints)
317. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post whitespace-padded distributed table-name validation full workspace rerun under ambient ZK auth env)
318. `ZK_AUTH=user:pass cargo test -p monolith-cli -q && ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post whitespace-padded parameter-sync metadata validation parity across CLI/runtime entrypoints)
319. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post whitespace-padded parameter-sync metadata validation full workspace rerun under ambient ZK auth env)
320. `ZK_AUTH=user:pass cargo test -p monolith-cli -q && ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post unique parameter-sync target validation parity across CLI/runtime entrypoints)
321. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post unique parameter-sync target validation parity full workspace rerun under ambient ZK auth env)
322. `ZK_AUTH=user:pass cargo test -p monolith-cli -q && ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post worker service-id enrichment in PS discovery-timeout diagnostics + run/runner index propagation regressions)
323. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post worker service-id discovery-timeout diagnostic enrichment full workspace rerun under ambient ZK auth env)
324. `ZK_AUTH=user:pass cargo test -p monolith-cli -q && ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post parameter-sync target uniqueness normalization for optional http-prefixes across CLI/runtime validation layers)
325. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post parameter-sync target http-prefix normalization full workspace rerun under ambient ZK auth env)
326. `ZK_AUTH=user:pass cargo test -p monolith-cli -q && ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post strict parameter-sync endpoint syntax validation and malformed-target regression expansion)
327. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post strict parameter-sync endpoint syntax validation full workspace rerun under ambient ZK auth env)
328. `ZK_AUTH=user:pass cargo test -p monolith-cli -q && ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post case-insensitive discovery service-type distinctness normalization across CLI/runtime validation layers)
329. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post case-insensitive discovery service-type distinctness normalization full workspace rerun under ambient ZK auth env)
330. `ZK_AUTH=user:pass cargo test -p monolith-cli -q && ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post case-insensitive http-prefix/host normalization for parameter-sync target uniqueness across CLI/runtime validation layers)
331. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post case-insensitive parameter-sync target uniqueness normalization full workspace rerun under ambient ZK auth env)
332. `ZK_AUTH=user:pass cargo test -p monolith-cli -q && ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post case-insensitive HTTP/HTTPS scheme recognition before parameter-sync endpoint auto-prefixing across CLI/runtime validation layers)
333. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post case-insensitive parameter-sync endpoint scheme recognition full workspace rerun under ambient ZK auth env)
334. `ZK_AUTH=user:pass cargo test -p monolith-cli -q && ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post native-training integration coverage expansion for case-insensitive parameter-sync scheme acceptance through RunConfig/RunnerConfig entrypoints)
335. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post integration parity expansion for case-insensitive parameter-sync scheme acceptance full workspace rerun under ambient ZK auth env)
336. `ZK_AUTH=user:pass cargo test -p monolith-cli -q && ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post authority-only parameter-sync endpoint validation + trailing-slash-normalized uniqueness hardening across CLI/runtime layers)
337. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post authority-only parameter-sync endpoint validation + trailing-slash-normalized uniqueness full workspace rerun under ambient ZK auth env)
338. `ZK_AUTH=user:pass cargo test -p monolith-cli -q && ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post explicit http/https-only parameter-sync scheme validation hardening across CLI/runtime layers)
339. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post explicit http/https-only parameter-sync scheme validation full workspace rerun under ambient ZK auth env)
340. `ZK_AUTH=user:pass cargo test -p monolith-cli -q && ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post userinfo rejection + default-port normalization canonicalization hardening for parameter-sync endpoints across CLI/runtime layers)
341. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post userinfo rejection + default-port normalization parameter-sync endpoint canonicalization full workspace rerun under ambient ZK auth env)
342. `ZK_AUTH=user:pass cargo test -p monolith-training -q && ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post native-training integration parity expansion for https default-port duplicate normalization)
343. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post native-training https default-port duplicate normalization integration parity expansion full workspace rerun under ambient ZK auth env)
344. `ZK_AUTH=user:pass cargo test -p monolith-cli -q && ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post strict internal-whitespace normalization for distributed service/table/parameter-sync identity fields across CLI/runtime validation layers)
345. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post strict internal-whitespace normalization for distributed service/table/parameter-sync identity fields full workspace rerun under ambient ZK auth env)
346. `ZK_AUTH=user:pass cargo test -p monolith-training -q && ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post role-error cleanup issue context propagation while preserving primary error text in distributed runner + parity test updates)
347. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post role-error cleanup issue context propagation parity expansion full workspace rerun under ambient ZK auth env)
348. `ZK_AUTH=user:pass cargo test -p monolith-training -q && ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post register-timeout integration parity expansion asserting appended cleanup diagnostics for run/runner worker+ps role paths with blocked cleanup)
349. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post register-timeout cleanup-diagnostic integration parity expansion full workspace rerun under ambient ZK auth env)
350. `ZK_AUTH=user:pass cargo test -p monolith-training -q && ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post connect-failure cleanup-diagnostic propagation in runner + run/runner connect-timeout integration parity expansion)
351. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post connect-failure cleanup-diagnostic propagation hardening full workspace rerun under ambient ZK auth env)
352. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post post-success cleanup diagnostic propagation hardening for combined deregister+disconnect failure paths)
353. `ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post post-success cleanup diagnostic propagation hardening targeted CLI regression check)
354. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post post-success cleanup diagnostic propagation hardening full workspace rerun under ambient ZK auth env)
355. `ZK_AUTH=user:pass cargo test -p monolith-training --test native_training_parity -q` ✅ (post connect-failure cleanup-disconnect-failure integration parity expansion and missing test-double restoration)
356. `ZK_AUTH=user:pass cargo test -p monolith-training -q && ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post native-training connect-failure cleanup-failure integration parity expansion targeted training/cli rerun)
357. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post connect-failure cleanup-failure integration parity expansion full workspace rerun under ambient ZK auth env)
358. `ZK_AUTH=user:pass cargo test -p monolith-training --test native_training_parity -q` ✅ (post custom worker service-type propagation assertions for connect-failure cleanup-timeout diagnostics in run/runner entrypoints)
359. `ZK_AUTH=user:pass cargo test -p monolith-training -q && ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post connect-failure cleanup-timeout custom-service-type parity assertion updates targeted training/cli rerun)
360. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post connect-failure cleanup-timeout custom-service-type parity assertion updates full workspace rerun under ambient ZK auth env)
361. `ZK_AUTH=user:pass cargo test -p monolith-training --test native_training_parity -q` ✅ (post PS-role connect-failure cleanup timeout/failure integration parity expansion with custom PS service-type diagnostics)
362. `ZK_AUTH=user:pass cargo test -p monolith-training -q && ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post PS-role connect-failure cleanup parity expansion targeted training/cli rerun)
363. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post PS-role connect-failure cleanup parity expansion full workspace rerun under ambient ZK auth env)
364. `ZK_AUTH=user:pass cargo test -p monolith-training -q && ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post successful-role disconnect-only cleanup context propagation hardening and related unit/integration parity assertion updates)
365. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post successful-role disconnect-only cleanup context propagation hardening full workspace rerun under ambient ZK auth env)
366. `ZK_AUTH=user:pass cargo test -p monolith-training -q && ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post successful-role deregister-only cleanup context propagation hardening and related unit/integration parity assertion updates)
367. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post successful-role deregister-only cleanup context propagation hardening full workspace rerun under ambient ZK auth env)
368. `ZK_AUTH=user:pass cargo test -p monolith-training --test native_training_parity -q` ✅ (post default-service-type post-success cleanup timeout integration parity expansion for run/runner entrypoints)
369. `ZK_AUTH=user:pass cargo test -p monolith-training -q && ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post default-service-type post-success cleanup-timeout integration parity expansion targeted training/cli rerun)
370. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post default-service-type post-success cleanup-timeout integration parity expansion full workspace rerun under ambient ZK auth env)
371. `ZK_AUTH=user:pass cargo test -p monolith-training --test native_training_parity -q` ✅ (post default-service-type post-success cleanup failure integration parity expansion with forced deregister/disconnect failures)
372. `ZK_AUTH=user:pass cargo test -p monolith-training -q && ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post default-service-type post-success cleanup failure integration parity expansion targeted training/cli rerun)
373. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post default-service-type post-success cleanup failure integration parity expansion full workspace rerun under ambient ZK auth env)
374. `ZK_AUTH=user:pass cargo test -p monolith-training --test native_training_parity -q` ✅ (post post-success single-step cleanup operation-context diagnostic wrapping and custom-worker failure integration parity expansion)
375. `ZK_AUTH=user:pass cargo test -p monolith-training -q && ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post post-success single-step cleanup operation-context parity expansion targeted training/cli rerun)
376. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post post-success single-step cleanup operation-context parity expansion full workspace rerun under ambient ZK auth env)
377. `ZK_AUTH=user:pass cargo test -p monolith-training -q && ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post PS connect-failure cleanup diagnostics parity expansion for default service type integration paths and PS unit coverage)
378. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post PS connect-failure cleanup diagnostics parity expansion full workspace rerun under ambient ZK auth env)
379. `ZK_AUTH=user:pass cargo test -p monolith-training -q && ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post post-success both-cleanup-failure diagnostic operation-context parity expansion across default/custom integration paths)
380. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post post-success both-cleanup-failure diagnostic operation-context parity expansion full workspace rerun under ambient ZK auth env)
381. `ZK_AUTH=user:pass cargo test -p monolith-training -q && ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post runner-level PS connect-timeout+cleanup-timeout custom service-type parity regression addition)
382. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post runner-level PS connect-timeout+cleanup-timeout custom service-type parity regression full workspace rerun under ambient ZK auth env)
383. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post discover-error + hanging-cleanup parity expansion; includes fixup for worker timeout mock discover-count accounting)
384. `ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post discover-error cleanup-timeout parity expansion compatibility verification)
385. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post discover-error cleanup-timeout parity expansion full workspace rerun under ambient ZK auth env)
386. `ZK_AUTH=user:pass cargo test -p monolith-training -q && ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post custom-service/index discover-error cleanup-timeout parity expansion across runner + config entrypoints)
387. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post custom-service/index discover-error cleanup-timeout parity expansion full workspace rerun under ambient ZK auth env)
388. `ZK_AUTH=user:pass cargo test -p monolith-training -q && ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post discover-error failing-cleanup diagnostics parity expansion across runner + config entrypoints)
389. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post discover-error failing-cleanup diagnostics parity expansion full workspace rerun under ambient ZK auth env)
390. `ZK_AUTH=user:pass cargo test -p monolith-training -q && ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post custom-service/index discover-error cleanup-failure diagnostics parity expansion across runner + config entrypoints)
391. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post custom-service/index discover-error cleanup-failure diagnostics parity expansion full workspace rerun under ambient ZK auth env)
392. `ZK_AUTH=user:pass cargo test -p monolith-training -q && ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post worker-timeout cleanup-failure parity expansion for custom service/index paths)
393. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post worker-timeout cleanup-failure parity expansion for custom service/index paths full workspace rerun under ambient ZK auth env)
394. `ZK_AUTH=user:pass cargo test -p monolith-training -q && ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post default worker-timeout cleanup-failure integration parity expansion for run/runner entrypoints)
395. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post default worker-timeout cleanup-failure integration parity expansion full workspace rerun under ambient ZK auth env)
396. `ZK_AUTH=user:pass cargo test -p monolith-training -q && ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post register-failure cleanup-context parity assertion expansion for default/custom runner service-type paths)
397. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post register-failure cleanup-context parity assertion expansion full workspace rerun under ambient ZK auth env)
398. `ZK_AUTH=user:pass cargo test -p monolith-training -q && ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post custom-service register-failure cleanup-context integration parity expansion across run/runner entrypoints)
399. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post custom-service register-failure cleanup-context integration parity expansion full workspace rerun under ambient ZK auth env)
400. `ZK_AUTH=user:pass cargo test -p monolith-training -q && ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post default-worker connect-failure cleanup-context integration parity expansion across run/runner entrypoints)
401. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post default-worker connect-failure cleanup-context integration parity expansion full workspace rerun under ambient ZK auth env)
402. `ZK_AUTH=user:pass cargo test -p monolith-training -q && ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post default-service register-failure cleanup-context integration parity expansion across run/runner entrypoints)
403. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post default-service register-failure cleanup-context integration parity expansion full workspace rerun under ambient ZK auth env)
404. `ZK_AUTH=user:pass cargo test -p monolith-training -q && ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post register-failure cleanup-timeout parity expansion across runner + config entrypoints)
405. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post register-failure cleanup-timeout parity expansion across runner + config entrypoints full workspace rerun under ambient ZK auth env)
406. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post register-failure cleanup-timeout default/custom matrix completion across runner + config entrypoints)
407. `ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post register-failure cleanup-timeout matrix completion compatibility verification)
408. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post register-failure cleanup-timeout default/custom matrix completion full workspace rerun under ambient ZK auth env)
409. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post indexed worker register-failure cleanup-timeout diagnostics parity expansion)
410. `ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post indexed worker register-failure cleanup-timeout diagnostics parity compatibility verification)
411. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post indexed worker register-failure cleanup-timeout diagnostics parity expansion full workspace rerun under ambient ZK auth env)
412. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post indexed PS register-failure cleanup-timeout diagnostics parity expansion)
413. `ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post indexed PS register-failure cleanup-timeout diagnostics parity compatibility verification)
414. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post indexed PS register-failure cleanup-timeout diagnostics parity expansion full workspace rerun under ambient ZK auth env)
415. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post indexed register-failure cleanup-failure diagnostics parity expansion)
416. `ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post indexed register-failure cleanup-failure diagnostics parity compatibility verification)
417. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post indexed register-failure cleanup-failure diagnostics parity expansion full workspace rerun under ambient ZK auth env)
418. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post default-service indexed register-failure cleanup failure/timeout parity expansion)
419. `ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post default-service indexed register-failure cleanup parity compatibility verification)
420. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post default-service indexed register-failure cleanup parity expansion full workspace rerun under ambient ZK auth env)
421. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post default-service indexed connect-failure cleanup failure/timeout parity expansion)
422. `ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post default-service indexed connect-failure cleanup parity compatibility verification)
423. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post default-service indexed connect-failure cleanup parity expansion full workspace rerun under ambient ZK auth env)
424. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post custom-service indexed connect-failure cleanup failure/timeout parity expansion)
425. `ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post custom-service indexed connect-failure cleanup parity compatibility verification)
426. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post custom-service indexed connect-failure cleanup parity expansion full workspace rerun under ambient ZK auth env)
427. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post custom-service indexed connect-timeout diagnostics parity expansion)
428. `ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post custom-service indexed connect-timeout diagnostics parity compatibility verification)
429. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post custom-service indexed connect-timeout diagnostics parity expansion full workspace rerun under ambient ZK auth env)
430. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post default-service indexed connect-timeout diagnostics parity expansion at runner level)
431. `ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post default-service indexed connect-timeout diagnostics parity compatibility verification)
432. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post default-service indexed connect-timeout diagnostics parity expansion full workspace rerun under ambient ZK auth env)
433. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post connect-timeout + disconnect-failure precedence parity expansion across runner/config entrypoints)
434. `ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post connect-timeout + disconnect-failure precedence parity compatibility verification)
435. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post connect-timeout + disconnect-failure precedence parity expansion full workspace rerun under ambient ZK auth env)
436. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post register-timeout + cleanup-failure precedence parity expansion across runner/config entrypoints)
437. `ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post register-timeout + cleanup-failure precedence parity compatibility verification)
438. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post register-timeout + cleanup-failure precedence parity expansion full workspace rerun under ambient ZK auth env)
439. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post discover-operation-timeout cleanup diagnostics parity expansion across runner/config entrypoints)
440. `ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post discover-operation-timeout cleanup diagnostics parity compatibility verification)
441. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post discover-operation-timeout cleanup diagnostics parity expansion full workspace rerun under ambient ZK auth env)
442. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post discover-timeout cleanup matrix parity completion across runner/config entrypoints)
443. `ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post discover-timeout cleanup matrix parity compatibility verification)
444. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post discover-timeout cleanup matrix parity completion full workspace rerun under ambient ZK auth env)
445. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post register-timeout cleanup matrix parity completion across runner/config entrypoints)
446. `ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post register-timeout cleanup matrix parity compatibility verification)
447. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post register-timeout cleanup matrix parity completion full workspace rerun under ambient ZK auth env)
448. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post worker-timeout cleanup matrix parity expansion across runner/config entrypoints)
449. `ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post worker-timeout cleanup matrix parity compatibility verification)
450. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post worker-timeout cleanup matrix parity expansion full workspace rerun under ambient ZK auth env)
451. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post discover-error indexed-default cleanup matrix parity expansion across runner/config entrypoints)
452. `ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post discover-error indexed-default cleanup matrix parity compatibility verification)
453. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post discover-error indexed-default cleanup matrix parity expansion full workspace rerun under ambient ZK auth env)
454. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post worker-timeout cleanup-timeout indexed-default parity completion across runner/config entrypoints)
455. `ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post worker-timeout cleanup-timeout indexed-default parity compatibility verification)
456. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post worker-timeout cleanup-timeout indexed-default parity completion full workspace rerun under ambient ZK auth env)
457. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post connect-timeout cleanup-failure matrix parity expansion across runner/config entrypoints)
458. `ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post connect-timeout cleanup-failure matrix parity compatibility verification)
459. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post connect-timeout cleanup-failure matrix parity expansion full workspace rerun under ambient ZK auth env)
460. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post runner-only indexed default-worker connect-timeout cleanup-failure parity addition)
461. `ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post runner-only indexed default-worker connect-timeout cleanup-failure compatibility verification)
462. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post runner-only indexed default-worker connect-timeout cleanup-failure parity addition full workspace rerun under ambient ZK auth env)
463. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post register-timeout cleanup-timeout indexed-default parity completion across runner/config entrypoints)
464. `ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post register-timeout cleanup-timeout indexed-default parity compatibility verification)
465. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post register-timeout cleanup-timeout indexed-default parity completion full workspace rerun under ambient ZK auth env)
466. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post runner-only worker register-timeout cleanup-failure indexed-default parity addition)
467. `ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post runner-only worker register-timeout cleanup-failure indexed-default compatibility verification)
468. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post runner-only worker register-timeout cleanup-failure indexed-default parity addition full workspace rerun under ambient ZK auth env)
469. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post worker ordering-issue timeout cleanup diagnostics parity expansion across runner/config entrypoints)
470. `ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post worker ordering-issue timeout cleanup diagnostics parity compatibility verification)
471. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post worker ordering-issue timeout cleanup diagnostics parity expansion full workspace rerun under ambient ZK auth env)
472. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post worker ordering+discover composite timeout cleanup parity expansion across runner/config entrypoints)
473. `ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post worker ordering+discover composite timeout cleanup parity compatibility verification)
474. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post worker ordering+discover composite timeout cleanup parity expansion full workspace rerun under ambient ZK auth env)
475. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post worker ordering+discover composite timeout cleanup matrix completion across runner/config entrypoints)
476. `ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post worker ordering+discover composite timeout cleanup matrix completion compatibility verification)
477. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post worker ordering+discover composite timeout cleanup matrix completion full workspace rerun under ambient ZK auth env)
478. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post worker ordering-issue timeout cleanup matrix completion across runner/config entrypoints)
479. `ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post worker ordering-issue timeout cleanup matrix completion compatibility verification)
480. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post worker ordering-issue timeout cleanup matrix completion full workspace rerun under ambient ZK auth env)
481. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post worker ordering-issue custom non-index cleanup parity expansion across runner/config entrypoints)
482. `ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post worker ordering-issue custom non-index cleanup parity compatibility verification)
483. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post worker ordering-issue custom non-index cleanup parity expansion full workspace rerun under ambient ZK auth env)
484. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post worker ordering+discover composite custom non-index cleanup parity expansion across runner/config entrypoints)
485. `ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post worker ordering+discover composite custom non-index cleanup parity compatibility verification)
486. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post worker ordering+discover composite custom non-index cleanup parity expansion full workspace rerun under ambient ZK auth env)
487. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post worker discover-timeout custom non-index cleanup parity expansion across runner/config entrypoints)
488. `ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post worker discover-timeout custom non-index cleanup parity compatibility verification)
489. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post worker discover-timeout custom non-index cleanup parity expansion full workspace rerun under ambient ZK auth env)
490. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post worker discover-error custom non-index cleanup parity expansion across runner/config entrypoints)
491. `ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post worker discover-error custom non-index cleanup parity compatibility verification)
492. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post worker discover-error custom non-index cleanup parity expansion full workspace rerun under ambient ZK auth env)
493. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post worker-timeout custom non-index cleanup parity expansion across runner/config entrypoints)
494. `ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post worker-timeout custom non-index cleanup parity compatibility verification)
495. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post worker-timeout custom non-index cleanup parity expansion full workspace rerun under ambient ZK auth env)
496. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post register-timeout custom non-index cleanup-failure parity expansion across runner/config entrypoints)
497. `ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post register-timeout custom non-index cleanup-failure parity compatibility verification)
498. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post register-timeout custom non-index cleanup-failure parity expansion full workspace rerun under ambient ZK auth env)
499. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post register-timeout custom non-index cleanup-timeout integration parity expansion across run/runner entrypoints)
500. `ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post register-timeout custom non-index cleanup-timeout integration parity compatibility verification)
501. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post register-timeout custom non-index cleanup-timeout integration parity expansion full workspace rerun under ambient ZK auth env)
502. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post connect-timeout custom non-index cleanup-failure parity expansion across runner/config entrypoints)
503. `ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post connect-timeout custom non-index cleanup-failure parity compatibility verification)
504. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post connect-timeout custom non-index cleanup-failure parity expansion full workspace rerun under ambient ZK auth env)
505. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post connect-timeout custom non-index cleanup-timeout parity expansion across runner/config entrypoints)
506. `ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post connect-timeout custom non-index cleanup-timeout parity compatibility verification)
507. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post connect-timeout custom non-index cleanup-timeout parity expansion full workspace rerun under ambient ZK auth env)
508. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post connect-failure custom non-index cleanup-context parity expansion across runner/config entrypoints)
509. `ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post connect-failure custom non-index cleanup-context parity compatibility verification)
510. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post connect-failure custom non-index cleanup-context parity expansion full workspace rerun under ambient ZK auth env)
511. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post discover-timeout custom non-index custom-service-types parity expansion across runner/config entrypoints)
512. `ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post discover-timeout custom non-index custom-service-types parity compatibility verification)
513. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post discover-timeout custom non-index custom-service-types parity expansion full workspace rerun under ambient ZK auth env)
514. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post worker-timeout custom-service-types non-index parity expansion across runner/config entrypoints)
515. `ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post worker-timeout custom-service-types non-index parity compatibility verification)
516. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post worker-timeout custom-service-types non-index parity expansion full workspace rerun under ambient ZK auth env)
517. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post ordering-issue custom-service-types non-index parity expansion across runner/config entrypoints)
518. `ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post ordering-issue custom-service-types non-index parity compatibility verification)
519. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post ordering-issue custom-service-types non-index parity expansion full workspace rerun under ambient ZK auth env)
520. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post ordering+discover-error + last-discover-error custom-service-types non-index parity expansion across runner/config entrypoints)
521. `ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post ordering+discover-error + last-discover-error custom-service-types non-index parity compatibility verification)
522. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post ordering+discover-error + last-discover-error custom-service-types non-index parity expansion full workspace rerun under ambient ZK auth env)
523. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post runner register-failure custom-service-type non-index naming parity alias expansion)
524. `ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post runner register-failure custom-service-type non-index naming parity compatibility verification)
525. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post runner register-failure custom-service-type non-index naming parity expansion full workspace rerun under ambient ZK auth env)
526. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post default-service non-index ordering+discover + last-discover cleanup parity expansion across runner/config entrypoints)
527. `ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post default-service non-index ordering+discover + last-discover parity compatibility verification)
528. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post default-service non-index ordering+discover + last-discover cleanup parity expansion full workspace rerun under ambient ZK auth env)
529. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post default-service non-index discover-timeout cleanup parity expansion across runner/config entrypoints)
530. `ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post default-service non-index discover-timeout cleanup parity compatibility verification)
531. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post default-service non-index discover-timeout cleanup parity expansion full workspace rerun under ambient ZK auth env)
532. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post default-service non-index worker-timeout cleanup parity expansion across runner/config entrypoints)
533. `ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post default-service non-index worker-timeout cleanup parity compatibility verification)
534. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post default-service non-index worker-timeout cleanup parity expansion full workspace rerun under ambient ZK auth env)
535. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post default-service non-index worker-ordering cleanup parity expansion across runner/config entrypoints)
536. `ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post default-service non-index worker-ordering cleanup parity compatibility verification)
537. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post default-service non-index worker-ordering cleanup parity expansion full workspace rerun under ambient ZK auth env)
538. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post default-service non-index register parity matrix completion across runner/config entrypoints)
539. `ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post default-service non-index register parity matrix compatibility verification)
540. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post default-service non-index register parity matrix completion full workspace rerun under ambient ZK auth env)
541. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post default-service non-index connect/discover parity alias completion in runner)
542. `ZK_AUTH=user:pass cargo test -p monolith-cli -q` ✅ (post default-service non-index connect/discover parity alias compatibility verification)
543. `ZK_AUTH=user:pass cargo test --workspace -q` ✅ (post default-service non-index connect/discover parity alias completion full workspace rerun under ambient ZK auth env)
544. `ZK_AUTH=user:pass cargo test -p monolith-training worker_discovery_error_when_cleanup_fails -- --nocapture` ✅ (post worker discovery error cleanup-failure naming parity alias expansion across RunConfig/RunnerConfig entrypoints)
545. `ZK_AUTH=user:pass cargo test -p monolith-training worker_timeout_when_cleanup_times_out -- --nocapture` ✅ (post worker-timeout cleanup-timeout generic alias expansion across RunConfig/RunnerConfig entrypoints)
546. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post worker discovery/timeout naming parity expansion full monolith-training regression rerun)
547. `ZK_AUTH=user:pass cargo test -p monolith-training worker_timeout_with_ -- --nocapture` ✅ (post runner worker-timeout cleanup-step timeout alias expansion across custom/default/indexed variants)
548. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post runner worker-timeout cleanup-step timeout alias expansion full monolith-training regression rerun)
549. `ZK_AUTH=user:pass cargo test -p monolith-training cleanup_steps_fail -- --nocapture` ✅ (post runner cleanup-step cleanup-failure alias expansion across worker/ps register-failure and worker-timeout naming variants)
550. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post runner cleanup-step timeout/failure matrix closure full monolith-training regression rerun)
551. `python3` cleanup-step counterpart audit (`runner.rs`) ✅ (reports zero missing aliases for both `_when_cleanup_steps_timeout -> _when_cleanup_steps_fail` and reverse direction)
552. `ZK_AUTH=user:pass cargo test -p monolith-training ps_register_timeout_preserves_error_when_cleanup_ -- --nocapture` ✅ (post runner plain PS register-timeout cleanup-failure counterpart addition)
553. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post runner plain PS register-timeout cleanup timeout/failure symmetry closure full monolith-training rerun)
554. `python3` cleanup-timeout/failure counterpart audit (`runner.rs`) ✅ (reports zero missing aliases for both `_when_cleanup_times_out -> _when_cleanup_fails` and reverse direction)
555. `ZK_AUTH=user:pass cargo test -p monolith-training connect_timeout_ -- --nocapture` ✅ (post run/runner default connect-timeout disconnect-failure-context alias additions)
556. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post run/runner connect-timeout alias expansion full monolith-training regression rerun)
557. `python3` naming audit (`native_training_parity.rs`) ✅ (verifies newly added connect-timeout alias names are present and executable)
558. `ZK_AUTH=user:pass cargo test -p monolith-training "preserves_ps_register_timeout_" -- --nocapture` ✅ (post run/runner ps-register-timeout disconnect-failure alias additions)
559. `ZK_AUTH=user:pass cargo test -p monolith-training "preserves_register_timeout_with_custom_service_type_and_index" -- --nocapture` ✅ (post run/runner custom-index register-timeout disconnect-failure alias additions)
560. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post register-timeout alias parity completion full monolith-training regression rerun)
561. `python3` cleanup-block alias audit (`native_training_parity.rs`) ✅ (reports zero missing `_when_cleanup_blocks -> _disconnect_failure_context` aliases)
562. `ZK_AUTH=user:pass cargo test -p monolith-training "connect_failure_with_cleanup" -- --nocapture` ✅ (post run/runner connect-failure cleanup-context alias additions)
563. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post connect-failure cleanup-context alias expansion full monolith-training regression rerun)
564. `python3` cleanup-timeout-context alias audit (`native_training_parity.rs`) ✅ (remaining `_cleanup_timeout_context -> _cleanup_context` gaps reduced after new run/runner connect-failure aliases)
565. `ZK_AUTH=user:pass cargo test -p monolith-training "connect_failure_with_custom_service_type" -- --nocapture` ✅ (post run/runner custom connect-failure cleanup-context alias additions)
566. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post custom connect-failure cleanup-context alias expansion full monolith-training regression rerun)
567. `python3` cleanup-timeout-context alias audit (`native_training_parity.rs`) ✅ (remaining `_cleanup_timeout_context -> _cleanup_context` gaps reduced from 22 to 18 after custom connect-failure alias additions)
568. `ZK_AUTH=user:pass cargo test -p monolith-training default_worker_connect_failure_with_cleanup_context -- --nocapture` ✅
569. `ZK_AUTH=user:pass cargo test -p monolith-training default_ps_connect_failure_with_cleanup_context -- --nocapture` ✅
570. `ZK_AUTH=user:pass cargo test -p monolith-training default_worker_connect_failure_with_index_cleanup_context -- --nocapture` ✅
571. `ZK_AUTH=user:pass cargo test -p monolith-training default_ps_connect_failure_with_index_cleanup_context -- --nocapture` ✅
572. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post default connect-failure cleanup-context alias expansion full monolith-training regression rerun)
573. `python3` cleanup-timeout-context alias audit (`native_training_parity.rs`) ✅ (remaining `_cleanup_timeout_context -> _cleanup_context` gaps reduced from 18 to 10 after default connect-failure alias additions)
574. `ZK_AUTH=user:pass cargo test -p monolith-training ps_connect_failure_with_cleanup_context -- --nocapture` ✅
575. `ZK_AUTH=user:pass cargo test -p monolith-training ps_connect_failure_with_custom_service_type_cleanup_context -- --nocapture` ✅
576. `ZK_AUTH=user:pass cargo test -p monolith-training ps_connect_failure_with_custom_service_type_and_index_cleanup_context -- --nocapture` ✅
577. `ZK_AUTH=user:pass cargo test -p monolith-training worker_register_failure_with_cleanup_context -- --nocapture` ✅
578. `ZK_AUTH=user:pass cargo test -p monolith-training ps_register_failure_with_cleanup_context -- --nocapture` ✅
579. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post PS connect/register cleanup-context alias expansion full monolith-training regression rerun)
580. `python3` cleanup-timeout-context alias audit (`native_training_parity.rs`) ✅ (`TOTAL 0` missing `_cleanup_timeout_context -> _cleanup_context` counterparts)
581. `ZK_AUTH=user:pass cargo test -p monolith-training worker_register_failure_with_disconnect_failure_context -- --nocapture` ✅
582. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post worker register disconnect-context alias additions full monolith-training regression rerun)
583. `python3` cleanup-context/disconnect-context alias audit (`native_training_parity.rs`) ✅ (remaining `_cleanup_context -> _disconnect_failure_context` gaps reduced from 20 to 18)
584. `ZK_AUTH=user:pass cargo test -p monolith-training ps_register_failure_with_disconnect_failure_context -- --nocapture` ✅
585. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post ps register disconnect-context alias additions full monolith-training regression rerun)
586. `python3` cleanup-context/disconnect-context alias audit (`native_training_parity.rs`) ✅ (remaining `_cleanup_context -> _disconnect_failure_context` gaps reduced from 18 to 16)
587. `ZK_AUTH=user:pass cargo test -p monolith-training worker_register_failure_with_custom_service_type_disconnect_failure_context -- --nocapture` ✅
588. `ZK_AUTH=user:pass cargo test -p monolith-training worker_register_failure_with_custom_service_type_and_index_disconnect_failure_context -- --nocapture` ✅
589. `ZK_AUTH=user:pass cargo test -p monolith-training worker_register_failure_with_default_service_type_disconnect_failure_context -- --nocapture` ✅
590. `ZK_AUTH=user:pass cargo test -p monolith-training worker_register_failure_with_default_service_type_and_index_disconnect_failure_context -- --nocapture` ✅
591. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post RunConfig worker-register disconnect-context variant additions full monolith-training regression rerun)
592. `python3` cleanup-context/disconnect-context alias audit (`native_training_parity.rs`) ✅ (remaining `_cleanup_context -> _disconnect_failure_context` gaps reduced from 16 to 12)
593. `ZK_AUTH=user:pass cargo test -p monolith-training runner_config_preserves_worker_register_failure_with_custom_service_type_disconnect_failure_context -- --nocapture` ✅
594. `ZK_AUTH=user:pass cargo test -p monolith-training runner_config_preserves_worker_register_failure_with_custom_service_type_and_index_disconnect_failure_context -- --nocapture` ✅
595. `ZK_AUTH=user:pass cargo test -p monolith-training runner_config_preserves_worker_register_failure_with_default_service_type_disconnect_failure_context -- --nocapture` ✅
596. `ZK_AUTH=user:pass cargo test -p monolith-training runner_config_preserves_worker_register_failure_with_default_service_type_and_index_disconnect_failure_context -- --nocapture` ✅
597. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post RunnerConfig worker-register disconnect-context variant additions full monolith-training regression rerun)
598. `python3` cleanup-context/disconnect-context alias audit (`native_training_parity.rs`) ✅ (remaining `_cleanup_context -> _disconnect_failure_context` gaps reduced from 12 to 8)
599. `ZK_AUTH=user:pass cargo test -p monolith-training run_config_preserves_ps_register_failure_with_custom_service_type_disconnect_failure_context -- --nocapture` ✅
600. `ZK_AUTH=user:pass cargo test -p monolith-training run_config_preserves_ps_register_failure_with_custom_service_type_and_index_disconnect_failure_context -- --nocapture` ✅
601. `ZK_AUTH=user:pass cargo test -p monolith-training run_config_preserves_ps_register_failure_with_default_service_type_disconnect_failure_context -- --nocapture` ✅
602. `ZK_AUTH=user:pass cargo test -p monolith-training run_config_preserves_ps_register_failure_with_default_service_type_and_index_disconnect_failure_context -- --nocapture` ✅
603. `ZK_AUTH=user:pass cargo test -p monolith-training runner_config_preserves_ps_register_failure_with_custom_service_type_disconnect_failure_context -- --nocapture` ✅
604. `ZK_AUTH=user:pass cargo test -p monolith-training runner_config_preserves_ps_register_failure_with_custom_service_type_and_index_disconnect_failure_context -- --nocapture` ✅
605. `ZK_AUTH=user:pass cargo test -p monolith-training runner_config_preserves_ps_register_failure_with_default_service_type_disconnect_failure_context -- --nocapture` ✅
606. `ZK_AUTH=user:pass cargo test -p monolith-training runner_config_preserves_ps_register_failure_with_default_service_type_and_index_disconnect_failure_context -- --nocapture` ✅
607. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post PS-register disconnect-context variant additions full monolith-training regression rerun)
608. `python3` cleanup-context/disconnect-context alias audit (`native_training_parity.rs`) ✅ (`missing 0` for `_cleanup_context -> _disconnect_failure_context`)
609. `ZK_AUTH=user:pass cargo test -p monolith-training run_config_preserves_connect_timeout_cleanup_context -- --nocapture` ✅
610. `ZK_AUTH=user:pass cargo test -p monolith-training run_config_preserves_ps_register_timeout_with_default_service_type_cleanup_context -- --nocapture` ✅
611. `ZK_AUTH=user:pass cargo test -p monolith-training runner_config_preserves_connect_timeout_cleanup_context -- --nocapture` ✅
612. `ZK_AUTH=user:pass cargo test -p monolith-training runner_config_preserves_ps_register_timeout_with_default_service_type_cleanup_context -- --nocapture` ✅
613. `ZK_AUTH=user:pass cargo test -p monolith-training -q` ✅ (post disconnect->cleanup wrapper alias expansion full monolith-training regression rerun)
614. `python3` multi-family naming audit (`native_training_parity.rs`) ✅ (`_disconnect_failure_context -> _cleanup_context` and related cleanup symmetry families all `missing 0`)
615. `ZK_AUTH=user:pass cargo test -p monolith-training run_config_preserves_default_connect_timeout_cleanup_context -- --nocapture` ✅
616. `ZK_AUTH=user:pass cargo test -p monolith-training runner_config_preserves_default_ps_connect_failure_cleanup_timeout_context -- --nocapture` ✅
617. `ZK_AUTH=user:pass cargo test -p monolith-training -q && python3 with_index counterpart audit` ✅ (`with_index -> no-with_index` `missing 0`)
618. `ZK_AUTH=user:pass cargo test -p monolith-training run_config_preserves_connect_timeout_cleanup_timeout_context -- --nocapture` ✅
619. `ZK_AUTH=user:pass cargo test -p monolith-training run_config_preserves_ps_register_timeout_with_custom_service_type_cleanup_timeout_context -- --nocapture` ✅
620. `ZK_AUTH=user:pass cargo test -p monolith-training runner_config_preserves_connect_timeout_cleanup_timeout_context -- --nocapture` ✅
621. `ZK_AUTH=user:pass cargo test -p monolith-training runner_config_preserves_ps_register_timeout_with_custom_service_type_cleanup_timeout_context -- --nocapture` ✅
622. `ZK_AUTH=user:pass cargo test -p monolith-training -q && python3 cleanup-to-cleanup-timeout audit` ✅ (`_cleanup_context -> _cleanup_timeout_context` `missing 0`)
623. `ZK_AUTH=user:pass cargo test -p monolith-training test_run_distributed_ps_registration_failure_with_default_service_type_includes_disconnect_failure_context -- --nocapture` ✅
624. `ZK_AUTH=user:pass cargo test -p monolith-training test_run_distributed_worker_registration_failure_with_default_service_type_includes_cleanup_timeout_context -- --nocapture` ✅
625. `ZK_AUTH=user:pass cargo test -p monolith-training -q && python3 runner cleanup naming audit` ✅ (`runner.rs` `_cleanup_context -> _disconnect_failure_context` and `_cleanup_context -> _cleanup_timeout_context` both `missing 0`)
75. `cargo test --workspace -q` ✅ (post detailed PS client response metadata additions and distributed/runtime regression rerun)

## Notes
- This update specifically closes major TODO/stub surfaces in CLI runtime flows and restores a reliable Linux workspace test command.
- Remaining parity workstreams continue in core/native-training/domain-specific modules listed in task definitions.
