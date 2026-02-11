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
75. `cargo test --workspace -q` ✅ (post detailed PS client response metadata additions and distributed/runtime regression rerun)

## Notes
- This update specifically closes major TODO/stub surfaces in CLI runtime flows and restores a reliable Linux workspace test command.
- Remaining parity workstreams continue in core/native-training/domain-specific modules listed in task definitions.
