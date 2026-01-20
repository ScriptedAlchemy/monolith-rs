<!--
Source: task/request.md
Lines: 15814-16017 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/mlp_utils.py`
<a id="monolith-native-training-mlp-utils-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 444
- Purpose/role: Utilities for MLP/YARN distributed training setup, tf.data service orchestration, MPI exception handling, and TF profiler control.
- Key symbols/classes/functions: `check_port`, `MLPEnv`, `add_mpi_exception_hook`, `mlp_pass`, `begin`, `after_create_session`, `EXTRA_DSWORKERS`.
- External dependencies: TensorFlow, tf.data service (`dsvc`), MPI/Horovod/BytePS, `yarn_runtime`, `distribution_utils`, `absl.flags/logging`, `socket`, `signal`, `subprocess`.
- Side effects: Opens sockets, starts TF DataService dispatcher/worker servers, starts TF profiler server, sets `sys.excepthook`, calls `os._exit(0)`, monkeypatches `_DatasetInitializerHook` methods.

**Required Behavior (Detailed)**
- `check_port(host, port, timeout=1)`:
  - Opens IPv4 or IPv6 socket based on host string (`':' in host.strip('[]')`).
  - Attempts to connect within `timeout` seconds; returns True if connected.
  - On timeout returns False.
  - On socket error, retries until timeout expires; returns False otherwise.
- `MLPEnv.__init__()`:
  - Collects env vars starting with `MLP_` or `MPI_`.
  - Reads `MLP_FRAMEWORK`, `MLP_SSH_PORT`, `MLP_LOG_PATH`, `MLP_DEBUG_PORT`, `MLP_ENTRYPOINT_DIR`, `MLP_TASK_CMD`, `MLP_ROLE`.
  - Builds `all_roles` from env vars like `MLP_<ROLE>_NUM`.
  - If `enable_mpi` (OMPI rank exists and role is WORKER):
    - `index = get_mpi_rank()`, `all_roles['WORKER'] = get_mpi_size()`.
    - `port = int(MLP_PORT) + index`.
  - Else uses `MLP_ROLE_INDEX` and `MLP_PORT`.
  - `avaiable` is True if MLP env and roles exist.
  - Records `cpu`, `gpu`, `gpu_type`, `mem` from env.
  - `host = yarn_runtime.get_local_host()`.
- `MLPEnv.enable_mpi`:
  - True when `OMPI_COMM_WORLD_RANK` exists and role is WORKER.
- `MLPEnv._get(name, default=None)`:
  - Reads from `_mlp_env`, strips quotes; returns default if missing.
- `num_replicas(role=None)`:
  - If MPI worker, returns `get_mpi_size()`; else reads `MLP_<ROLE>_NUM`.
- `get_all_host(role=None, is_primary=True)`:
  - Returns `MLP_<ROLE>_ALL_PRIMARY_HOSTS` or `MLP_<ROLE>_ALL_HOSTS` value.
- `get_all_addrs(role=None, is_primary=True)`:
  - Reads `MLP_<ROLE>_ALL_PRIMARY_ADDRS`/`ALL_ADDRS`, splits by comma, else [].
- `get_host(role=None, index=None, is_primary=True)`:
  - Uses MPI/local index logic; returns `MLP_<ROLE>_<index>_PRIMARY_HOST` or `_HOST`.
- `get_addr(role=None, index=None, is_primary=True)`:
  - Computes host and port:
    - MPI worker uses `MLP_<ROLE>_0_PORT` + index.
    - Otherwise `MLP_<ROLE>_<index>_PORT`.
  - Returns `"host:port"` or None.
- `get_port(role=None, index=None)`:
  - MPI worker: `MLP_<ROLE>_0_PORT` (default 2222) + index.
  - Else: `MLP_<ROLE>_<index>_PORT` (default 2222).
  - Note: `_get` returns strings; may require int conversion for correctness.
- `dispatcher_target()`:
  - Returns `grpc://{dispatcher_addr}` or `grpc://localhost:5050`.
- `dispatcher_addr(role=None)`:
  - Uses role default `'dispatcher'` and `get_addr`.
- `wait(role=None, index=0, timeout=-1, use_ssh=True)`:
  - Repeatedly calls `check_port` on `host:port`, sleeping 5s until ready or timeout.
  - Uses `ssh_port` when `use_ssh=True`.
- `join(role='worker', index=0, use_ssh=True)`:
  - Waits for host to come up, then loops until port stops responding (timeout=60 per check).
  - Stops TF profiler if started and calls `os._exit(0)`.
- `queue_device` property:
  - Returns `/job:ps/task:0/device:CPU:0` if PS exists; else worker CPU or local CPU.
- `start_profiler(port=6666)`:
  - Starts `tf.profiler.experimental.server` (port offset by MPI index).
- `profiler_trace(...)`:
  - Builds `ProfilerOptions` and calls `tf.profiler.experimental.client.trace`.
  - Uses all addresses if `index < 0`, otherwise specific address.
- `add_mpi_exception_hook()`:
  - If OMPI rank not set, returns.
  - Installs `sys.excepthook` that prints error details and calls `mpi4py.MPI.COMM_WORLD.Abort(1)`.
- `EXTRA_DSWORKERS`: global list of extra WorkerServer handles.
- `mlp_pass(dispatcher_role='dispatcher', dsworker_role='dsworker', worker_role='worker', ps_role='ps')`:
  - Uppercases roles; if `FLAGS.dataset_use_dataservice`, monkeypatches `_DatasetInitializerHook.begin` and `.after_create_session`.
  - Creates `MLPEnv`. If available:
    - Dispatcher role: starts `dsvc.DispatchServer` on `mlp_env.port`, then `mlp_env.join()`.
    - Dsworker role: waits for dispatcher, starts `dsvc.WorkerServer`, starts profiler, then join.
    - Worker role:
      - If dataset service enabled, waits for dispatcher and dsworkers, sets `FLAGS.data_service_dispatcher`.
      - Starts extra dsworkers on GPU worker based on `FLAGS.num_extra_dsworker_on_gpu_worker`.
      - Logs worker start info and roles.
- `begin(self)` (monkeypatched into `_DatasetInitializerHook`):
  - Stores iterator initializer; sets `_broadcast_dataset_id=None`.
  - If sync training enabled, tries to import BytePS or Horovod (based on `MONOLITH_WITH_BYTEPS`).
  - If `registed_dataset_id` collection exists, broadcasts dataset_id from rank 0 and stores `_broadcast_dataset_id`.
  - Calls `graph.clear_collection(...)` (graph is undefined; potential bug).
- `after_create_session(self, session, coord)`:
  - If `_broadcast_dataset_id` present, runs it to log dataset ids and clears it.
  - Runs iterator initializer.
- Threading/concurrency: uses tf.data service servers and background processes; no explicit threads here.
- Determinism: depends on environment, networking, MPI rank.
- Logging/metrics: extensive absl logging.

**Rust Mapping (Detailed)**
- Target crate/module: N/A (TF data service + MPI orchestration not in Rust).
- Rust public API surface: if needed, separate runtime module for distributed training env/launcher.
- Data model mapping: env var parsing, host/port resolution, dataset service endpoints.
- Feature gating: MPI/BytePS/TF data service features optional.
- Integration points: training launcher, profiler integration.

**Implementation Steps (Detailed)**
1. Determine whether Rust training needs MLP/YARN orchestration features.
2. If yes, implement env parsing and host/port logic, plus dispatcher/worker lifecycle.
3. Implement MPI exception handling with `mpi` crate equivalents.
4. Provide profiler server/trace integration if applicable.
5. Mirror dataset initializer hook behavior in Rust data input pipeline.

**Tests (Detailed)**
- Python tests: none in repo.
- Rust tests: add unit tests for env parsing and host/port computation.
- Cross-language parity test: compare computed addresses/ports for fixed env maps.

**Gaps / Notes**
- `begin()` references `graph.clear_collection` but `graph` is undefined (likely bug).
- `get_port` uses `_get` without int conversion; may return strings.

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

### `monolith/native_training/model.py`
<a id="monolith-native-training-model-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 182
- Purpose/role: Defines a small FFM (field-aware factorization machine) test model and input pipeline for native training.
- Key symbols/classes/functions: `_parse_example`, `TestFFMModel`, `FFMParams`, constants `_NUM_SLOTS`, `_FFM_SLOT`, `_VOCAB_SIZES`, `_NUM_EXAMPLES`.
- External dependencies: TensorFlow, NumPy, Monolith feature/entry APIs, `deep_insight_ops`.
- Side effects: Sets NumPy seed in input function; emits deep insight metrics in training; logs model info.

**Required Behavior (Detailed)**
- Constants:
  - `_NUM_SLOTS = 6`, `_VOCAB_SIZES = [5,5,5,5,5,5]`, `_NUM_EXAMPLES = 64`.
  - `_FFM_SLOT` defines tuple pairs and embedding dim for dot products.
- `_parse_example(example: str) -> Dict[str, tf.Tensor]`:
  - Builds feature map with fixed label and VarLenFeature for each slot.
  - Parses examples with `tf.io.parse_example`.
  - Converts any `SparseTensor` to `RaggedTensor`.
  - Returns feature dict including `"label"` and slot features.
- `TestFFMModel(NativeTask)`:
  - `create_input_fn(mode)` returns `input_fn()`:
    - Sets `np.random.seed(0)` for deterministic example generation.
    - Generates `_NUM_EXAMPLES` examples via `generate_ffm_example(_VOCAB_SIZES)`.
    - Builds dataset: batch to `per_replica_batch_size`, map `_parse_example`, cache, repeat, prefetch.
  - `create_model_fn()` returns `model_fn(features, mode, config)`:
    - Creates feature slots/columns for each slot with FTRL bias + SGD default optimizer.
    - Collects bias embeddings for first half of slots.
    - Computes FFM dot products for each `(user,item,dim)` in `_FFM_SLOT`.
    - `ffm_out = add_n(dot_res) + sum_bias`, `pred = sigmoid(ffm_out)`.
    - If `mode == PREDICT`: returns `EstimatorSpec(predictions=pred)`.
    - Loss: `reduce_sum(binary_crossentropy(features["label"], pred))`.
    - If deep insight metrics enabled and sample ratio > 0:
      - Creates deep insight client, builds uids/req_times/sample_rates, calls `write_deep_insight`.
      - Logs model_name/target.
    - Else uses `tf.no_op()`.
    - Increments global step and applies embedding gradients via `ctx.apply_embedding_gradients`.
    - Returns `EstimatorSpec(loss=loss, train_op=group(apply_grads, deep_insight_op))`.
  - `create_serving_input_receiver_fn()`:
    - Creates string placeholder `instances`, parses via `_parse_example`, returns `ServingInputReceiver`.
- `FFMParams(SingleTaskModelParams)`:
  - `task()` returns `TestFFMModel.params()` with `per_replica_batch_size = 64`.

**Rust Mapping (Detailed)**
- Target crate/module: N/A (test model; no Rust equivalent).
- Rust public API surface: none.
- Data model mapping: if ported, map to Rust embedding feature slots and FFM dot products.
- Feature gating: none.
- Integration points: training pipeline and deep insight metrics.

**Implementation Steps (Detailed)**
1. If needed for parity demos, implement a Rust FFM example model and parser.
2. Mirror dataset generation determinism (`np.random.seed(0)`) with fixed RNG.
3. Match bias/slot construction and FFM dot-product structure.
4. Add deep insight metrics only if TF runtime backend exists.

**Tests (Detailed)**
- Python tests: none in repo.
- Rust tests: optional integration test for forward pass + loss.
- Cross-language parity test: compare forward outputs on fixed synthetic batch.

**Gaps / Notes**
- This is a test/sample model; may not be required for production parity.

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
