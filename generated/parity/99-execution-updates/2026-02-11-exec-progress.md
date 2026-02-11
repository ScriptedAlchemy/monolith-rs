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

## Notes
- This update specifically closes major TODO/stub surfaces in CLI runtime flows and restores a reliable Linux workspace test command.
- Remaining parity workstreams continue in core/native-training/domain-specific modules listed in task definitions.
