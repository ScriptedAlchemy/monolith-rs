<!--
Source: task/request.md
Lines: 5511-5708 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/cpu_sync_training_test.py`
<a id="monolith-native-training-cpu-sync-training-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 360
- Purpose/role: End-to-end CPU sync training tests with Horovod for features, embeddings, sequence features, and distributed sync training.
- Key symbols/classes/functions: `FeatureTask`, `EmbeddingUpdateTask`, `FloatFeatureTask`, `SequenceFeatureTask`, `NonFeatureTask`, `CpuSyncTrainTest`, `DistributedSyncTrainTest`.
- External dependencies: `horovod.tensorflow`, `cpu_training`, `feature`, `embedding_combiners`, `device_utils`, `NativeTask`, `entry`, `advanced_parse`.
- Side effects: sets `MONOLITH_WITH_HOROVOD=True` env var; runs TF estimators and training loops.

**Required Behavior (Detailed)**
- Environment:
  - `MONOLITH_WITH_HOROVOD` must be set **before** importing `monolith.native_training`.
- `FeatureTask`:
  - Input: ragged int64 feature `[1,2,3,4]` repeated 5.
  - Model uses `FeatureSlotConfig`, one slice (dim=5), embedding lookup.
  - For TRAIN: computes loss on embedding, applies gradients via feature factory.
- `EmbeddingUpdateTask`:
  - Compares monolith embedding updates vs TF embedding lookup.
  - Uses `ConstantsInitializer(0)` and `AdagradOptimizer(0.1, accum=1)`.
  - Asserts equality between monolith embedding and TF embedding; increments global step.
- `FloatFeatureTask`:
  - Includes float feature; predictions from float feature sum.
  - Training uses ragged embedding for gradients; float feature only for predictions.
- `SequenceFeatureTask`:
  - Ragged sequence feature; uses `embedding_combiners.FirstN(2)`.
  - Loss from embeddings; predictions from sequence feature sum.
- `NonFeatureTask`:
  - Input dataset yields scalar; model returns constant loss and uses input as train op.
- `CpuSyncTrainTest`:
  - `test_cpu_training_feature/float_feature/sequence_feature/non_feature` run `CpuTraining` with `enable_sync_training=True`.
  - `test_embedding_update` trains 10 steps, compares embedding updates to TF.
- `DistributedSyncTrainTest`:
  - `test_basic` and `test_sparse_pipelining` invoke `distributed_sync_train` with config toggles (pipelined a2a, embedding_postpush).

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests`.
- Rust public API surface: distributed training harness + Horovod equivalent (if any).
- Data model mapping: TF estimator + feature factory → Rust training loop abstractions.
- Feature gating: requires Horovod/TF runtime; Rust likely lacks direct support.
- Integration points: `CpuTraining`, feature pipeline, embedding update logic.

**Implementation Steps (Detailed)**
1. Determine whether Rust will support Horovod-like sync training.
2. If yes, add integration tests for feature pipeline and embedding updates.
3. If no, document tests as Python-only and provide alternative sync tests.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: TBD (requires distributed training support).
- Cross-language parity test: compare embedding update equivalence on a tiny synthetic dataset.

**Gaps / Notes**
- Tests assume Horovod is available and initialize `hvd` in-process.
- Uses `entry.ConstantsInitializer` and `entry.AdagradOptimizer` (must exist in Rust).

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

### `monolith/native_training/cpu_training.py`
<a id="monolith-native-training-cpu-training-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 2449
- Purpose/role: TODO (manual)
- Key symbols/classes/functions: TODO (manual)
- External dependencies: TODO (manual)
- Side effects: TODO (manual)

**Required Behavior (Detailed)**
- Define the **functional contract** (inputs → outputs) for every public function/class.
- Enumerate **error cases** and exact exception/messages that callers rely on.
- Capture **config + env var** behaviors (defaults, overrides, precedence).
- Document **I/O formats** used (proto shapes, TFRecord schemas, JSON, pbtxt).
- Note **threading/concurrency** assumptions (locks, async behavior, callbacks).
- Identify **determinism** requirements (seeds, ordering, float tolerances).
- Identify **performance characteristics** that must be preserved.
- Enumerate **metrics/logging** semantics (what is logged/when).

**Rust Mapping (Detailed)**
- Target crate/module: TODO (manual)
- Rust public API surface: TODO (manual)
- Data model mapping: TODO (manual)
- Feature gating: TODO (manual)
- Integration points: TODO (manual)

**Implementation Steps (Detailed)**
1. Extract all public symbols + docstrings; map to Rust equivalents.
2. Port pure logic first (helpers, utils), then stateful services.
3. Recreate exact input validation and error semantics.
4. Mirror side effects (files, env vars, sockets) in Rust.
5. Add config parsing and defaults matching Python behavior.
6. Add logging/metrics parity (field names, levels, cadence).
7. Integrate into call graph (link to downstream Rust modules).
8. Add tests and golden fixtures; compare outputs with Python.
9. Document deviations (if any) and mitigation plan.

**Tests (Detailed)**
- Python tests: TODO (manual)
- Rust tests: TODO (manual)
- Cross-language parity test: TODO (manual)

**Gaps / Notes**
- TODO (manual)

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

### `monolith/native_training/cpu_training_distributed_test_binary.py`
<a id="monolith-native-training-cpu-training-distributed-test-binary-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 226
- Purpose/role: Distributed CPU training integration test binary with host-based service discovery.
- Key symbols/classes/functions: `SyncHook`, `FeatureTask`, `HostServiceDiscovery`, `test0/1/2`, `test_run`.
- External dependencies: `absl.flags/app`, `tensorflow`, `cpu_training`, `cluster_manager`, `service_discovery`, `feature`.
- Side effects: overrides retry/backoff globals, sets `_shutdown_ps`, writes discovery files, spawns barrier sync.

**Required Behavior (Detailed)**
- Flags:
  - `test_case`, `test_dir`, `server_type` (`ps`/`worker`), `index`, `num_ps`, `num_workers`, `num_extra_ps`, `num_redundant_ps`, `uuid`, `use_native_multi_hash_table`.
- Overrides:
  - `cluster_manager._cluster_query_failure_handler = _sleep_short` (0.1s).
  - `cpu_training._EXTRA_PS_BENCHMARK_SECS = 0.5`.
- `SyncHook`:
  - Creates per-worker boolean var in local variables (chief) or global variables (workers).
  - After session creation, sets its index to True; chief waits until all workers set.
- `FeatureTask`:
  - Defines `training_hooks` param.
  - Model builds feature slot, embedding lookup, applies gradients.
  - Training hooks include `SyncHook` and any provided hooks.
- `HostServiceDiscovery`:
  - Registers by writing files `<base>/<name>/<index>` with address.
  - Query reads files into `{index: addr}` map.
- `test_run(params)`:
  - Builds `DistributedCpuTrainingConfig` using flags and a per-test model dir.
  - Sets `params.train.max_pending_seconds_for_barrier = 2`.
  - Uses `HostServiceDiscovery` and runs `cpu_training.distributed_train`.
- `test0`: normal run.
- `test1`: overrides `_shutdown_ps` to never exit.
- `test2`: adds `RaiseErrorHook` that throws `DeadlineExceededError` on first `before_run`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests`.
- Rust public API surface: distributed training integration tests + file-based discovery.
- Data model mapping: HostServiceDiscovery → file-backed discovery in Rust (not present).
- Feature gating: requires distributed training + PS/worker runner.
- Integration points: `distributed_train` analog, barrier sync.

**Implementation Steps (Detailed)**
1. Add file-backed discovery helper for integration tests.
2. Add hook to block chief until all workers register.
3. Add test cases that simulate non-shutdown and deadline errors.

**Tests (Detailed)**
- Python tests: this binary test (invoked by integration harness).
- Rust tests: integration tests if Rust distributed runner exists.
- Cross-language parity test: verify barrier synchronization semantics.

**Gaps / Notes**
- This script mutates module-level globals (`_EXTRA_PS_BENCHMARK_SECS`, `_shutdown_ps`).

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
