<!--
Source: task/request.md
Lines: 18205-18391 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/native_task.py`
<a id="monolith-native-training-native-task-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 213
- Purpose/role: Defines the `NativeTask` base class (TF-native training/eval/serving) and the `NativeContext` helper for feature slot creation, embedding gradient application, and async functions.
- Key symbols/classes/functions: `NativeContext`, `NativeTask`.
- External dependencies: TensorFlow, `monolith.core.base_task.BaseTask`, `monolith.core.hyperparams`,
  `monolith.native_training.feature`, `monolith.native_training.prefetch_queue`,
  `monolith.native_training.model_export.export_context.ExportMode`.
- Side effects:
  - Raises `ValueError` if both `feature_factory` and `layout_factory` are provided in `NativeContext`.
  - Delegates to async function manager (may enqueue TF ops).

**Required Behavior (Detailed)**
- **`NativeContext(feature_factory=None, async_function_mgr=None, layout_factory=None)`**
  - Stores `feature_factory`, `async_function_mgr`, `layout_factory`.
  - If both `feature_factory` and `layout_factory` are set, raises:
    - `ValueError("Cannot set feature_factory and layout_factory in the same time")`.
- **`NativeContext.create_feature_slot(config)`**
  - If `layout_factory` present, delegates to `layout_factory.create_feature_slot(config)`.
  - Else uses `feature_factory.create_feature_slot(config)`.
  - No TF ops are created by this call (per docstring).
- **`NativeContext.apply_embedding_gradients(grads_and_vars, scale=1)`**
  - If `layout_factory` present, delegates to `layout_factory.apply_gradients(grads_and_vars)`.
  - Else delegates to `feature_factory.apply_gradients(grads_and_vars, scale=scale)`.
  - Expects `grads_and_vars` from `FeatureColumn.get_all_embeddings_concatenated`.
- **`NativeContext.add_async_function(target, args=None, kwargs=None, is_async=None, queue_name="async_queue")`**
  - Delegates to `async_function_mgr.add_async_function(...)`.
  - Returns enqueue op if async, else result of `target`.
  - Semantic contract (documented): tensors used by async function should be passed via args/kwargs only.
- **`NativeTask(BaseTask, abc.ABC)`**
  - `params()` extends BaseTask params with:
    - `metrics.*`:
      - `enable_deep_insight` (default False), `deep_insight_target` ("ctr_head"),
        `deep_insight_name` (None), `deep_insight_sample_ratio` (0.01),
        `extra_fields_keys` (list),
        `enable_throughput_hook` (True),
        `enable_kafka_metrics` (False),
        `enable_tf2_profiler_hook` (True),
        `enable_file_metrics` (False),
        `file_base_name` ("/vepfs/jaguar_deepinsight_results"),
        `file_ext` ("txt"),
        `parse_fn`/`key_fn`/`layout_fn` (None),
        `dump_filename` (""),
        `use_data_service` (False).
    - `mode`: `tf.estimator.ModeKeys.TRAIN` (temporary; doc says will be removed).
    - `train.*`:
      - `max_pending_seconds_for_barrier` (30),
      - `slow_start_steps` (0),
      - `sample_bias` (0.0),
      - `use_gpu_emb_table` (False),
      - `use_fountain` (False),
      - `fountain_zk_host` (""), `fountain_model_name` (""), `fountain_parse_on_server` (False),
        `fountain_precompute_value_rowids` (False).
    - `serving.*`:
      - `export_with_gpu_allowed` (False),
      - `export_with_cleared_entry_devices` (False),
      - `export_when_saving` (False),
      - `export_dir_base` ("exported_models"),
      - `export_mode` (`ExportMode.DISTRIBUTED`),
      - `shared_embedding` (True),
      - `with_remote_gpu` (False).
  - `__init__(params)`:
    - Calls `BaseTask.__init__` and sets `self._ctx = NativeContext()` and `self.p = params`.
  - `ctx` property returns `self._ctx`.
  - Abstract methods:
    - `create_input_fn(self, mode)`
    - `create_model_fn(self)`
  - `create_serving_input_receiver_fn()`:
    - Returns `None` by default; callers must override when `serving.export_when_saving` is enabled.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src` (task base + context), `monolith-rs/crates/monolith-data` (feature factories/layouts).
- Rust public API surface:
  - `NativeContext` struct with `create_feature_slot`, `apply_embedding_gradients`, `add_async_function`.
  - `NativeTask` trait with `params()`, `create_input_fn`, `create_model_fn`, `create_serving_input_receiver_fn`.
- Data model mapping:
  - `feature_factory` ↔ `FeatureFactory` trait.
  - `layout_factory` ↔ `EmbeddingLayoutFactory` trait (optional, mutually exclusive).
- Feature gating:
  - Async function manager (prefetch queue) behind feature flag if not supported by backend.
  - Export-related params gated by serving feature.
- Integration points:
  - `NativeTask` becomes base for `MonolithBaseModel` in Rust; param defaults must match Python.

**Implementation Steps (Detailed)**
1. Port `NativeContext` with mutual exclusion validation and delegation to feature/layout factories.
2. Implement an async function manager interface in Rust (or stubs if backend lacks it).
3. Port `NativeTask.params()` with identical defaults and nested param groups.
4. Implement `NativeTask` base that stores `params` and `ctx`.
5. Add validation hooks so `create_serving_input_receiver_fn` must be overridden when export is enabled.
6. Add unit tests for parameter defaults and mutual exclusion errors.

**Tests (Detailed)**
- Python tests: none dedicated; covered indirectly by model/task usage.
- Rust tests: add unit tests for `NativeContext` validation and param defaults.
- Cross-language parity test: compare serialized defaults from Python vs Rust `params()` output.

**Gaps / Notes**
- `OutConfig`, `OutType`, `TensorShape` are imported but unused in this file (no runtime effect).
- `create_serving_input_receiver_fn` returning `None` is explicitly invalid when export is enabled; Rust should enforce or document this.

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

### `monolith/native_training/native_task_context.py`
<a id="monolith-native-training-native-task-context-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 58
- Purpose/role: Defines a global per-process task context (`NativeTaskContext`) and helper functions to set/get it.
- Key symbols/classes/functions: `NativeTaskContext`, `with_ctx`, `get`.
- External dependencies: `contextlib`, `typing.NamedTuple`, `monolith.agent_service.backends.SyncBackend`.
- Side effects: Mutates module-level `_CTX` global.

**Required Behavior (Detailed)**
- **`NativeTaskContext(NamedTuple)`** with fields:
  - `num_ps: int`
  - `ps_index: int`
  - `num_workers: int`
  - `worker_index: int`
  - `model_name: str`
  - `sync_backend: SyncBackend`
  - `server_type: str`
- **`with_ctx(ctx)`** (context manager):
  - Stores previous `_CTX` as `old_ctx`, sets `_CTX = ctx`, yields.
  - On exit:
    - If `old_ctx` is not `None`, restores `_CTX = old_ctx`.
    - If `old_ctx` is `None`, leaves `_CTX` set to `ctx` (no reset to `None`).
- **`get()`**:
  - If `_CTX is None`, returns a new `NativeTaskContext` with defaults:
    - `num_ps=0`, `ps_index=0`, `num_workers=1`, `worker_index=0`,
      `server_type=""`, `model_name=""`, `sync_backend=None`.
  - Else returns `_CTX` object as-is (no copy).

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src` (task context).
- Rust public API surface:
  - `struct NativeTaskContext { ... }`
  - `with_ctx(ctx, |...| { ... })` or RAII guard to set/restore.
  - `get_ctx()` returns current context or default.
- Data model mapping:
  - `sync_backend` should be an enum or trait object mirroring `SyncBackend`.
- Feature gating: none.
- Integration points:
  - Training/serving flows should call `get_ctx()` for worker index and model name.

**Implementation Steps (Detailed)**
1. Implement a thread-local or global context storage in Rust (match Python semantics).
2. Implement context guard that **only** restores prior context if it existed.
3. Provide `get_ctx()` returning default values when unset.
4. Add unit tests for default context and nesting behavior.

**Tests (Detailed)**
- Python tests: none dedicated.
- Rust tests: add unit tests for `with_ctx` nesting and default values.
- Cross-language parity test: compare defaults and nesting semantics.

**Gaps / Notes**
- The context manager does **not** clear `_CTX` when exiting outermost scope; Rust should match this behavior exactly.

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
