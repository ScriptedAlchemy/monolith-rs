# Monolith Python -> Monolith-RS Parity Master Plan: Manifest Notes

## 1) Crisp Summary Of The Request

Design an Unpack task graph (manifest.json) that drives an exhaustive, process-heavy porting effort: port every Python file under `monolith/` into `monolith-rs/` with максимально close parity. The plan must:

- Account for every Python file (334 discovered) with a concrete Rust destination or justified N/A.
- Preserve behavior/protocol/config/ops/I/O semantics; deviations must be tracked as explicit parity gaps.
- Prioritize `monolith/agent_service/**` serving parity first, then `monolith/core/**`, then `monolith/native_training/**`.
- Keep TensorFlow runtime optional (dynamic load; no vendoring), Candle default.
- Maintain/extend existing artifacts: `monolith-rs/PYTHON_PARITY_INDEX.md` and per-file checklists under `monolith-rs/parity/**`.

This manifest should be a runnable orchestration that (a) validates inventory + checklist coverage, (b) builds and maintains a Python->Rust mapping table, and (c) provides repeatable per-module port execution + parity validation steps.

## 2) Proposed Module Graph Outline

Note: module ids below are suggestions for `task/*.ai.tsx` files; keep each module small and composable. The orchestrator should import phase modules to enforce ordering.

### Phase 0: Repo/Artifact Sanity + Inputs

1) `task/00-discover.ai.tsx`
   - Responsibility:
     - Load/read existing inventory artifacts (Python file list, counts) and the Rust parity checklist tree.
     - Verify required base directories exist (`monolith/`, `monolith-rs/`).
   - Why:
     - Early failure with actionable errors; establishes stable inputs for later phases.

2) `task/01-validate-inventory.ai.tsx`
   - Responsibility:
     - Validate: 334 python files <-> 334 checklist files; detect missing/extra.
     - Validate index row count matches checklist file count.
     - Produce a short validation report and machine-readable summary (counts, diffs).
   - Why:
     - Enforces "nothing is left untracked" non-negotiable before porting starts.

### Phase 1: Canonical Mapping Table (Python -> Rust Targets)

3) `task/10-generate-mapping-table.ai.tsx`
   - Responsibility:
     - Produce/refresh a canonical mapping table for ALL python files:
       - `monolith/**.py` -> Rust crate/module path(s) + status + notes
     - Start by materializing the phase-1 mapping sections already drafted (agent_service) and then extend for core/native_training.
   - Why:
     - Turns per-file checklists into an executable, queryable roadmap and gate for progress reporting.

4) `task/11-normalize-mapping.ai.tsx`
   - Responsibility:
     - Normalize mapping conventions:
       - crate naming, module naming, location of tests, fixtures, proto modules
       - required "N/A justification" format for unsupported behaviors
   - Why:
     - Prevents drift and inconsistent destinations across many contributors.

### Phase 2: Agent Service Parity (Highest Priority)

5) `task/20-agent-service-plan.ai.tsx`
   - Responsibility:
     - Extract a sub-plan for `monolith/agent_service/**`:
       - group files into implementable clusters (data_def/backends/zk_mirror/tfs_* / process supervision / CLI)
       - define target Rust modules and test harness strategy (fake ZK, fake TFServing)
   - Why:
     - Provides a concrete sequence for the most cross-cutting and operationally critical area.

6) `task/21-agent-service-impl.ai.tsx` (optional in initial manifest; can be a placeholder)
   - Responsibility:
     - Execute porting tasks for a selected slice (e.g., data_def + fake_zk + backends serialization), or at least produce implementation-ready tickets.
   - Why:
     - Converts plan into actionable steps, but keep initial manifest focused if implementation is out of scope.

### Phase 3: Core Parity

7) `task/30-core-plan.ai.tsx`
   - Responsibility:
     - Focus on `monolith/core/**` critical infra: `hyperparams`, `feature`, `model_registry`, base_layer/task/model_params.
     - Identify Rust-side gaps and required tests (string rendering parity, error text parity, nested map semantics).
   - Why:
     - Core infra correctness unlocks many downstream ports.

### Phase 4: Native Training Parity (Largest Surface)

8) `task/40-native-training-plan.ai.tsx`
   - Responsibility:
     - Segment `monolith/native_training/**` into subdomains (data pipeline, distribution/runtime, export/checkpoint, hooks, metrics).
     - Explicitly flag TF-dependent vs TF-optional implementations and map to Rust crates.
   - Why:
     - Avoids a single overwhelming plan; enables incremental milestones.

### Phase 5: Cross-Cutting Runtime + TF Optionality

9) `task/50-tf-runtime-plan.ai.tsx`
   - Responsibility:
     - Define optional TF runtime integration contract:
       - dynamic libtensorflow loading, custom op loading, signature mapping, SavedModel parsing
     - Identify which parity gaps require TF runtime vs Candle-only feasibility.
   - Why:
     - Keeps TF concerns explicit and gated, as required by non-negotiables.

### Phase 6: Validation + Reporting (Always Last)

10) `task/90-parity-dashboard.ai.tsx`
   - Responsibility:
     - Produce an end-to-end report:
       - coverage status by module/crate
       - open parity gaps summary
       - next suggested work queue (top N TODOs with highest dependency value)
   - Why:
     - Converts artifacts into a single "what to do next" output for stakeholders.

11) `task/index.ai.tsx` (orchestrator)
   - Responsibility:
     - Imports: 00/01 -> 10/11 -> (20/30/40/50) -> 90
     - Adds a final summarizer agent that writes a single deliverable (likely a doc) used by humans to drive the next steps.
   - Why:
     - Ensures deterministic phase ordering without intricate agent-level dependencies.

## 3) Cross-Module Wiring Notes (assetRef + external_needs)

Recommended exported assets (via `assetRef(...)`) and how downstream modules should pin them:

- From `task/01-validate-inventory.ai.tsx`
  - Export: `inventory_validation_report` (doc) and `inventory_validation_json` (data)
  - Downstream usage:
    - `external_needs: [{ alias: 'inventory_validation_json', agent: 'validate' }]`
    - Include with `<Context>{inventory_validation_json}</Context>`

- From `task/10-generate-mapping-table.ai.tsx`
  - Export: `mapping_table_md` (doc) and `mapping_table_json` (data)
  - Downstream usage:
    - Agent-service/core/native-training plan modules depend on mapping normalization.

- From `task/11-normalize-mapping.ai.tsx`
  - Export: `mapping_conventions` (doc) and `mapping_conventions_json` (data)
  - Downstream usage:
    - Ensures consistent crate/module naming and consistent N/A justification fields.

- From domain planners (`task/20-*`, `task/30-*`, `task/40-*`, `task/50-*`)
  - Export: `domain_plan_md` and `domain_plan_json` for each domain.
  - Downstream usage:
    - The final dashboard consumes all domain plans and the inventory/mapping artifacts.

Wiring pattern:

- Prefer module-level ordering via ESM imports for phase sequencing.
- Use `external_needs` only when a downstream agent must read a specific upstream agent output deterministically (especially when an upstream module contains multiple agents).
- Keep exported assets small (summaries / structured JSON), not large file trees, to avoid prompt bloat.

## 4) Concurrency / Threading Notes (No <Loop>/<Parallel> In Generator)

The generator should avoid `<Loop>/<Parallel>` for repo-wide enumeration because:

- Planning-time action evaluation + loop expansion will be heavy for 334 files, increasing plan latency.
- Many per-file prompts would be large and would compete for context budget; better to aggregate and operate per-domain.

Instead:

- Concurrency should be applied at the module/domain level:
  - Agent service plan and core plan can run in parallel only after inventory/mapping phases are complete.
- Keep thread grouping default; only override threads for:
  - A single "rollup" summary agent that merges domain plans into the final dashboard output.

If later manifests add per-file execution:

- Use chunking (e.g., 20-50 files per unit) rather than one agent per file.
- Use a supervisor/reviewer step for high-risk modules (agent_service, hyperparams, ZK fakes, TFServing fakes).

## 5) Next-Agent Implementation Checklist (For Writing manifest.json)

This note is intentionally a blueprint for the *next* agent that will write the actual Unpack module files and generate `manifest.json`.

- Create the phase modules listed in section (2) under `task/*.ai.tsx` and keep them small (one primary agent each).
- In each producer module, export `assetRef(...)` bindings for the assets that should be consumed elsewhere (section 3).
- In each consumer module, import the upstream `assetRef` bindings and pin them with `external_needs` to the correct upstream agent id(s).
- In `task/index.ai.tsx`, import all phase modules in the intended order to establish module-level ordering.
- Keep outputs bounded: emit md/json summaries for validation/mapping/domain plans; avoid pushing 334-file raw tables into prompt context.

## Part: task/request.part01.md

Lines: 1-320 (1-based, inclusive)

- "This is the **single source of truth** for **porting every line of Python in `monolith/`** into Rust (`monolith-rs/`) to reach **full or максимально close parity**."
- "We are targeting **line-for-line parity** in behavior and public surface area, not necessarily 1:1 API style."
- "**Every Python file in `monolith/` must be accounted for** in this plan with a concrete Rust destination or an explicit, justified N/A."
- "We must **preserve semantics** even if the Rust implementation uses different internal structures."
- "**No vendoring** of TensorFlow runtime libraries. TF runtime must be optional and dynamic (Linux x86_64 only, best effort elsewhere)."
- "**Candle remains the default** inference backend unless a true TF SavedModel runtime is available."
- "**All parity gaps must be listed**, tracked, and closed or explicitly justified."
- "Parity means:"
- "**Behavior:** Rust outputs match Python outputs given the same inputs and configuration."
- "**Protocol:** gRPC, protobuf, and disk formats remain compatible."
- "**Config:** CLI flags, config files, and environment variables behave the same."
- "**Ops:** Custom TensorFlow ops and side effects are supported or mapped."
- "**I/O:** File formats (TFRecord, SavedModel, checkpointing) are compatible."
- "**Tests:** Rust tests cover the same scenarios as Python tests."
- "We will port **all** of:"
- "`monolith/agent_service/**`"
- "`monolith/core/**`"
- "`monolith/native_training/**`"
- "`monolith/monolith_workspace.bzl`, `monolith/tf_serving_workspace.bzl`"
- "`monolith/utils.py`, `monolith/path_utils.py`, `monolith/base_runner.py`, `monolith/tpu_runner.py`, `monolith/gpu_runner.py`"
- "Any Python entry points under `monolith/**` and `monolith/native_training/**`"
- "We will also account for:"
- "`third_party/org_tensorflow/**` (patches that affect behavior)"
- "Any Python-dependent tooling or code-gen (feature lists, ops, TF Serving config)"
- "### 3.1 Generate a **Line-Level Inventory**"
- "Build a script to enumerate:"
- "File path"
- "Line count"
- "Top-level symbols (classes, functions)"
- "Create `monolith-rs/PYTHON_PARITY_INDEX.md` containing:"
- "One row per file with line count and status"
- "A link to a per-file checklist"
- "**Status:** DONE"
- "**Artifacts:**"
- "`monolith-rs/PYTHON_PARITY_INDEX.md` (334 files enumerated)"
- "### 3.2 Per-File Parity Checklist"
- "For every Python file, create a matching checklist file:"
- "`monolith-rs/parity/<path>.md` (mirrors Python path)"
- "Each checklist includes:"
- "Function/class list"
- "Behavior notes"
- "Rust mapping (file + symbol)"
- "Test coverage"
- "Open gaps"
- "**Status:** DONE"
- "**Artifacts:**"
- "`monolith-rs/parity/**` (per-file checklist for every `monolith/**/*.py`)"
- "Rule: No file is considered “done” until:"
- "Feature parity verified against Python tests or equivalent."
- "Rust tests added."
- "Input/output formats validated."
- "Performance regressions documented."
- "## 6) TensorFlow Runtime Integration (Optional, but Required for Full Parity)"
- "Key Requirements:"
- "Dynamic `libtensorflow` loading (no vendoring)"
- "Custom op loading (`TF_LoadLibrary`)"
- "Signature-based tensor mapping"
- "Output decoding consistent with Python"
- "### 13.1 `monolith/agent_service/**` → `monolith-rs/crates/monolith-serving`"
- "| `monolith/agent_service/agent_service.py` | `monolith-rs/crates/monolith-serving/src/grpc_agent.rs` (gRPC), `monolith-rs/crates/monolith-serving/src/server.rs` (runtime) | IN PROGRESS | Split responsibilities between coordination gRPC and serving runtime. |"
- "| `monolith/agent_service/tfs_client.py` | `monolith-rs/crates/monolith-serving/src/tfserving.rs` | IN PROGRESS | Map client utilities to Rust TF Serving client. |"

## Part: task/request.part02.md

Lines: 321-661 (1-based, inclusive)

- "## Line-Level Inventory (All Python Files)"
- "This table enumerates **every** Python file under `monolith/` with line counts and a direct link to its checklist section."
- "| Python File | Lines | Status | Rust Mapping | Notes |"

## Part: task/request.part03.md

Lines: 662-4913 (1-based, inclusive)

- "Every file listed below must be fully mapped to Rust with parity behavior verified."
- "`monolith/__init__.py`" on import: "Calls `enable_monkey_patch()`; on exception, logs `enable_monkey_patch failed`."
- "`monolith/__init__.py`" module wiring: "Registers module in `sys.modules` under `monolith.<name>`."
- "`monolith/__init__.py`" Rust guidance: "Avoid heavy side effects on import; if unavoidable, document and feature-gate."
- "`monolith/agent_service/__init__.py`": "Module is intentionally empty; importing it must have no side effects."
- "`monolith/agent_service/agent.py`" startup: "Initializes HDFS env via `env_utils.setup_hdfs_env()`."
- "`monolith/agent_service/agent.py`" dense multi-process: "handles `DeployType.DENSE` + `dense_service_num > 1` by spawning multiple processes."
- "`monolith/agent_service/agent_base.py`": "`AgentBase` abstract base with `start()` and `wait_for_termination()`."
- "`monolith/agent_service/agent_client.py`" host selection: "Host is `MY_HOST_IP` env var or `socket.gethostbyname(gethostname())`."
- "`monolith/agent_service/agent_client.py`" command dispatch: "`FLAGS.cmd_type` dispatch:"
- "`monolith/agent_service/agent_controller.py`" SavedModel detection: "scans graph nodes with `op == 'TfServingRemotePredict'`."
- "`monolith/agent_service/agent_service.py`" overloads: "`AgentServiceImpl.__init__` is **overloaded** via `@singledispatchmethod`:"
- "`monolith/agent_service/agent_service.py`" v3 parity note: "Else: `NotImplementedError` for v3."
- "`monolith/agent_service/mocked_zkclient.py`": "In-memory fake Kazoo/ZooKeeper client with watches, nodes, and basic CRUD."
- "`monolith/agent_service/model_manager.py`" markers/locks: "`WRITE_DONE = '.write.done'`, `READ_LOCK = '.read.lock'`."
- "`monolith/agent_service/replica_manager.py`": "Maintains replica registration and status updates in ZK; watches for replica changes and exposes lookup APIs."
- "`monolith/agent_service/resource_utils.py`" model sizing: "`cal_model_info_v2(exported_models_path, ckpt=None, version=None)`:"
- "`monolith/agent_service/run.py`" dispatch: "Unknown value raises `ValueError`."
- "`monolith/agent_service/tfs_client.py`" record IO: "`read_data` reads size-prefixed payload (8-byte little-endian length)."
- "`monolith/agent_service/tfs_monitor.py`" port selection: "`get_addr(sub_model_name)` chooses port based on deploy type and sub_model type."
- "`monolith/core/auto_checkpoint_feed_hook.py`": "TPU infeed/outfeed SessionRunHook with thread-managed queues, TPU init/shutdown, and end-of-stream stopping signals."
- "`monolith/core/base_layer.py`" abstract fprop: "`fprop()` is abstract: raises `NotImplementedError('Abstract method of %s' % self)`."
- "`monolith/core/hyperparams.py`": "Dynamic hyperparameter container with attribute + dotted-path access, nested parameter trees, immutability, stringification, and class instantiation support."
- "`monolith/core/hyperparams.py`" deepcopy rule: "`__deepcopy__`: if value is a `tf.Tensor`, **do not copy** (keep ref); else deep-copy; memo updated."

## Part: task/request.part04.md

Lines: 4914-8613 (1-based, inclusive)

- "### `monolith/native_training/barrier_ops.py`"
- "**Status:** IN PROGRESS (manual)"
- "`BarrierAlreadyPlacedError`: raised when attempting to place a barrier twice."
- "`BarrierOp(capacity, is_chief=True, wait_seconds=1, name_prefix=\"default\", barrier_callbacks=None)`:"
- "`place_barrier(session, action=\"\")`: - If barrier already placed (index 0 True), raises `BarrierAlreadyPlacedError`."
- "`wait_until_barrier_removed(session, index)`: - Validates `index` in `(0, capacity)` else `ValueError`."
- "`BarrierHook(index, barrier_op)`: - `after_run`: if `index > 0` and barrier placed, calls `wait_until_barrier_removed`."
- "Python tests: `monolith/native_training/barrier_ops_test.py`."
- "Target crate/module: `monolith-rs/crates/monolith-training/src/barrier.rs`."
- "### `monolith/native_training/basic_restore_hook.py`"
- "`CheckpointRestorerHook(listeners=None)`: - Logs \"Create CheckpointRestorerHook.\""
- "`_restore(session)`: - **No actual restore actions performed in hook**."
- "Gaps / Notes - Hook does not implement `end()`, so listener `end()` is never called."
- "Target crate/module: `monolith-rs/crates/monolith-training/src/hooks.rs`."
- "### `monolith/native_training/clip_ops.py`"
- "`clip_by_global_norm(t_list, clip_norm, use_norm=None)`: - Requires `t_list` to be a list; else raises `TypeError(\"t_list should be a list\")`."
- "`clip_by_global_norm(t_list, clip_norm, use_norm=None)`: - If `t_list` empty, returns `(t_list, 0)`."
- "Gaps / Notes - Custom ops (`monolith_clip_by_global_norm_*`) are not available in Rust."
- "### `monolith/native_training/cluster_manager.py`"
- "`generate_session_config(cluster_and_task=None)`: - If `None`, returns `ConfigProto(allow_soft_placement=True)`."
- "`generate_session_config(cluster_and_task=None)`: - Disables Grappler meta optimizer."
- "`_get_ps_cluster_file_name(model_dir, uuid)`: - Path: `<model_dir>/ps_cluster_dir/<uuid or \"ps_info\">`."
- "`_save_ps_cluster_to_file(file_name, ps_addrs)`: - Writes comma-separated list to temp file and atomically renames."
- "### `monolith/native_training/consul.py`"
- "`Client.__init__()`: - Determines consul host: - `CONSUL_HTTP_HOST` or `TCE_HOST_IP` env vars. - Else uses `/opt/tmp/sock/consul.sock` if file exists. - Else defaults to `\"127.0.0.1\"`."
- "`lookup(name, timeout=3, cachetime=0)`: - If `cachetime>0` and cached entry is fresh, returns it."
- "`_lookup(name, timeout)`: - GET `/v1/lookup/name?name=<name>&addr-family=dual-stack`."
- "Gaps / Notes - Python uses a ByteDance-specific `/v1/lookup/name` API, not stock Consul."
- "### `monolith/native_training/cpu_sync_training_test.py`"
- "Environment: - `MONOLITH_WITH_HOROVOD` must be set **before** importing `monolith.native_training`."
- "### `monolith/native_training/data/__init__.py`"
- "Importing package exposes symbols listed above at `monolith.native_training.data.*`."
- "### `monolith/native_training/data/datasets.py`"
- "Flags defined: `data_service_dispatcher`, `dataset_use_dataservice`, `dataset_input_patterns`, `dataset_input_use_snappy`, `dataset_input_compression_type`, `dataset_input_use_parquet`, `dataset_input_use_tfrecord`, `dataset_worker_idx`, `dataset_num_workers`, `kafka_other_metadata`."
- "Gaps / Notes - Heavy reliance on custom TF ops; Rust needs replacements or TF runtime backend."
- "### `monolith/native_training/data/feature_utils.py`"
- "`filter_by_feature_value(variant, field_name, op, operand, field_type, keep_empty, operand_filepath)`: - `op` must be in `{gt,ge,eq,lt,le,neq,between,in,not-in,all,any,diff,startswith,endswith}`."
- "`filter_by_feature_value(variant, field_name, op, operand, field_type, keep_empty, operand_filepath)`: - Exactly one of `operand` or `operand_filepath` is provided; if filepath set, it must exist and `op` must be in `{in, not-in}`."
- "`filter_by_value(variant, field_name, op, operand, variant_type, keep_empty, operand_filepath)`: - For `variant_type != 'instance'`, calls `add_feature('__LINE_ID__')`."
- "Error semantics: - Many checks are `assert` (raising `AssertionError`), some raise `RuntimeError(\"params error!\")` for invalid bytes operands."
- "### `monolith/native_training/data/parsers.py`"
- "`ProtoType.get_tf_type(proto_type)`: - Raises `Exception('proto_type {} is not support'.format(proto_type))` for unknown types."
- "Metrics/logging: - `sharding_sparse_fids` emits a timer metric named `sharding_sparse_fids` with model_name tag."
- "### `monolith/native_training/debugging/debugging_client.py`"
- "`--type` must be `debugging_variables` or `debugging_features`."
- "### `monolith/native_training/debugging/debugging_server.py`"
- "`create_app()`: - On exceptions, returns `status=fail` with traceback in msg."
- "### `monolith/native_training/device_utils.py`"
- "`enable_gpu_training()` sets True; `disable_gpu_training()` sets False; `is_gpu_training()` returns it."

## Part: task/request.part05.md

Lines: 8614-12264 (1-based, inclusive)

- "### `monolith/native_training/distributed_ps_benchmark.py`"
- "`_generate_config(servers, job_name=utils.PS_JOB_NAME)`: - Builds `ClusterDef` with job tasks derived from server targets; returns `ConfigProto`."
- "`_get_vocab_hash_table_factory(dim)`: - Returns factory that builds a hash table with `EmbeddingHashTableConfig` using cuckoo + SGD(1.0) + zeros init and segment dim `dim`."
- "`DistributedHashTableTest.lookup(enable_dedup, real_run=True)`: - Creates `ps_num=10` local servers; uses server0 with cluster config."
- "`apply_gradients(real_run=True)`: - Similar setup; assigns ones to embeddings, looks up, computes `loss=0.3*embeddings`, grads; applies gradients."
- "Target crate/module: `monolith-rs/crates/monolith-training/benches`."
- "### `monolith/native_training/distributed_ps_factory.py`"
- "`MultiHashTableFactory`: - Caches converted configs via `multi_hash_table_ops.convert_to_cached_config` keyed by `id(slot_to_config)`."
- "`create_multi_type_hash_table(num_ps, slot_to_config, hash_filters, sync_clients, reduce_network_packets, max_rpc_deadline_millis)`: - `num_ps==0`: returns local `MergedMultiTypeHashTable` backed by `MultiTypeHashTable` and local hash tables."
- "`create_multi_type_hash_table(num_ps, slot_to_config, hash_filters, sync_clients, reduce_network_packets, max_rpc_deadline_millis)`: - `reduce_network_packets=True`: uses `DistributedMultiTypeHashTable` (multi-type on PS) to reduce RPC count."
- "`create_partitioned_hash_table(num_ps, use_native_multi_hash_table, max_rpc_deadline_millis, hash_filters, sync_clients, enable_gpu_emb, queue_configs)`: - Returns `distributed_ps.PartitionedHashTable` with queue configs."
- "Target crate/module: `monolith-rs/crates/monolith-training/src/ps`."
- "### `monolith/native_training/distributed_ps_factory_test.py`"
- "Side effects: sets `MONOLITH_WITH_HOROVOD=True`; may initialize Horovod."
- "`test_create_multi_type_hash_table_2_ps_with_reduced_packets`: same with `reduce_network_packets=True`."
- "Target crate/module: `monolith-rs/crates/monolith-training/tests`."
- "### `monolith/native_training/distributed_ps_sync.py`"
- "Environment flags: - `MONOLITH_WITH_HOROVOD`, `MONOLITH_WITH_OPTIMIZED_HOROVOD`, `MONOLITH_WITH_BYTEPS` and related G2G/GDR flags determine alltoall backend and GPU paths."
- "`DistributedMultiTypeHashTableMpi.lookup(slot_to_id, auxiliary_bundle, early_reorder_indicies_res_pack)`: - Requires `early_reorder_indicies_res_pack` (support for `reorder_fids_in_data_pipeline=False` dropped)."
- "`assign/assign_add/reinitialize`: - Not implemented (raises `NotImplementedError`)."
- "Target crate/module: `monolith-rs/crates/monolith-training/src/ps`."
- "### `monolith/native_training/distributed_ps_sync_test.py`"
- "Side effects: sets `MONOLITH_WITH_HOROVOD=True` and initializes Horovod."
- "First lookup returns zeros."
- "Second lookup returns negative values scaled by `hvd.size()`."
- "### `monolith/native_training/distributed_serving_ops.py`"
- "`remote_predict(...)`: - Validates `model_name` non-null."
- "`remote_predict(...)`: - Calls `tf_serving_remote_predict` custom op with input/output aliases, model name, task, version, deadline, signature; returns output tensors (index 2 of op result)."
- "`refresh_sync_config(sync_backend, ps_index)`: - Fetches sync targets; populates `ClientConfig` with targets and extra info; sets model name, signature `hashtable_assign`, timeout 3000ms; returns serialized bytes."
- "Target crate/module: `monolith-rs/crates/monolith-training/src/serving` or `monolith-serving`."
- "### `monolith/native_training/distribution_ops.py`"
- "Purpose/role: Wrapper utilities around custom distribution ops for sharding, embedding layout transforms, and gradient backprop helpers."
- "`split_by_indices(indices, tensor, num_splits)`: - Calls `monolith_split_by_indices` custom op; gradient registered via `monolith_split_by_indices_gradient`."
- "`fused_embedding_to_layout(embeddings_list, fid_offset, feature_offset, nfl_offset, batch_size, ...)`: - Converts flattened embeddings into layout tensors using `FeatureConfigs` and offsets; supports multiple versions and GPU paths."
- "Target crate/module: `monolith-rs/crates/monolith-training/src/ops` (or `monolith-tensor` for ragged)."
- "### `monolith/native_training/distribution_ops_test.py`"
- "Feature gating: `tf-runtime` feature for these tests; GPU-only tests gated on CUDA availability."
- "Integration points: custom op library load (libmonolith_ops) before creating the TF graph/session."
- "### `monolith/native_training/distribution_utils.py`"
- "Side effects: Sets many env vars, creates `/tmp/bps_<uuid>_socket_<id>` dir, runs `ip addr show` via shell, enables eager execution in benchmark funcs, initializes BytePS/Horovod, mutates config object fields."
- "`bps_init(uuid)`: - Runs `ip addr show <interface>` to compute host IP; exports as `UCX_RDMA_CM_SOURCE_ADDRESS` and `DMLC_NODE_HOST`."
- "`try_init_cuda()`: - If `CUDA_VISIBLE_DEVICES` not set but MPI local rank present, set `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES=<local_rank>`."
- "Gaps / Notes - Uses shell command `ip addr show` to resolve interface IP; Rust port should use OS APIs or run the command for parity."
- "### `monolith/native_training/embedding_combiners.py`"
- "`ReduceSum.combine(key, embedding, name=None)`: - Uses `ragged_utils.fused_value_rowids(key)` to map values to row ids."
- "`FirstN.combine(key, embedding, name=None)`: - Under `device_utils.maybe_device_if_allowed('/device:GPU:0')`, calls `tf.scatter_nd(indices, embedding, shape)`."
- "### `monolith/native_training/entry.py`"
- "`_convert_to_proto(obj, proto)`: - Calls `proto.SetInParent()`."
- "`HashTableConfigInstance.__str__` returns `TableConfigPB:<serialized>, LearningRateFns:[<fn_strs>]` where proto is `SerializeToString()` and each fn uses `str(fn)`."
- "Gaps / Notes - Proto fields must match Python names exactly; default handling must skip `None` to avoid overwriting proto defaults."
- "### `monolith/native_training/env_utils.py`"
- "`get_zk_auth_data()`: - Reads `ZK_AUTH` env var."
- "`get_zk_auth_data()`: - If set, prints `\"ZK_AUTH <value>\"` and returns `[('digest', ZK_AUTH)]`."

## Part: task/request.part03.md

Lines: 662-4581 (1-based, inclusive)

- "## Per-File Parity Checklists (All Python Files)"
- "Every file listed below must be fully mapped to Rust with parity behavior verified."
- "Avoid heavy side effects on import; if unavoidable, document and feature-gate."
- "### `monolith/__init__.py`"
- "`add_module(module)`: - Accepts a module object or a module string."
- "If `name == 'native_model'`, rename to `'base_model'`."
- "Registers module in `sys.modules` under `monolith.<name>`."
- "Calls `enable_monkey_patch()`; on exception, logs `enable_monkey_patch failed`."
- "### `monolith/native_training/file_ops.py`"
- "`WritableFile.append(content)`: - Appends a 0-D string tensor to the file (via `monolith_writable_file_append`)."
- "`FileCloseHook(files)`: - `files` must be a `list` of `WritableFile`."
- "Feature gating: - TF runtime required for graph-op parity; native mode may use std::fs writes."
- "### `monolith/native_training/file_ops_test.py`"
- "Tests rely on `TEST_TMPDIR` env var; Rust tests should mirror temp dir handling."
- "### `monolith/native_training/fused_embedding_to_layout_test.py`"
- "For `op_version >= 3` without `shard_op_version`, the test raises `TypeError('Not imple')`."
- "GPU tests require `test_util.use_gpu()` and only run for versions 3/4."
- "### `monolith/agent_service/agent.py`"
- "Mutates `REPLICA_ID` and `DENSE_SERVICE_IDX` when `use_mps` is true."
- "If `model_manager.start()` returns False: log error, `os.kill(os.getpid(), SIGKILL)`."
- "### `monolith/agent_service/agent_base.py`"
- "Command string: `{proxy_binary} --port={config.proxy_port} --grpc_target=localhost:{config.tfs_entry_port} [--conf_file=proxy.conf] &`."
- "`ServingLog` context manager:"
- "On exit: `chdir` back to previous cwd (does not close file handle itself)."
- "### `monolith/agent_service/agent_client.py`"
- "Uses regex `TASK = r'^(\\w+):(\\d+)$'` to detect `server_type:task` nodes; otherwise treats as `idc:cluster` prefix."
- "Handle `NoNodeError` by printing \"{model_name} has not load !\" and returning."
- "### `monolith/agent_service/agent_controller.py`"
- "Reads `saved_model.pb`, parses `SavedModel`, scans graph nodes with `op == 'TfServingRemotePredict'`."
- "Build `SavedModelDeployConfig(model_base_path=..., version_policy='latest' for entry, else 'latest_once')`."
- "For `pub/unpub`, layout path is `/{bzid}/layouts/{layout}`."
- "### `monolith/agent_service/agent_service.py`"
- "`AgentServiceImpl.__init__` is **overloaded** via `@singledispatchmethod`:"
- "Else: `NotImplementedError` for v3."
- "HeartBeat behavior:"
- "`agent_version == 3`: use `AgentDataProvider` callback map to fill addresses."
- "`GetResource` behavior:"
- "Else: fill address with `get_local_ip()` + `agent_port`, plus memory via `cal_available_memory_v2()`."
- "### `monolith/agent_service/backends.py`"
- "Creates ephemeral binding node at"
- "`/{bzid}/binding/{model}/{sub_graph}:{container}` (makepath=True)."
- "If `_is_lost` set: clears available set, calls `_zk.restart()`, returns."
- "### `monolith/agent_service/data_def.py`"
- "`ReplicaMeta.get_address(use_archon=False, address_family=AddressFamily.IPV4)`:"
- "Treats `0.0.0.0*` and `[::]*` as invalid (set to None)."
- "### `monolith/agent_service/mocked_zkclient.py`"
- "`create(path, value=b'', ephemeral=False, makepath=False, sequence=False)`:"
- "If `sequence`: appends zero-padded 10-digit counter (starts at 0000000000)."
- "### `monolith/agent_service/model_manager.py`"
- "Constants: `WRITE_DONE=\".write.done\"`, `READ_LOCK=\".read.lock\"`."
- "Selects latest version by **string comparison** (`old_data[0] < version`)."
- "Gaps / Notes - `remove_read_lock` uses `os.join` (likely typo) when deleting stray locks."
- "### `monolith/agent_service/resource_utils.py`"
- "`open_hdfs(fname)`: - Retries up to 3 times; logs exceptions; asserts output list is not None."
- "`cal_model_info_v2(exported_models_path, ckpt=None, version=None)`:"
- "Returns `{sub_model_name: (size, version_path)}`."
- "### `monolith/agent_service/run.py`"
- "Flag `bin_name` selects:"
- "Unknown value raises `ValueError`."
- "### `monolith/agent_service/tfs_wrapper.py`"
- "Runs `strings $TFS_BINARY | grep PredictionServiceGrpc` (shell=True) to detect gRPC remote op support."
- "### `monolith/agent_service/utils.py`"
- "Overrides `os.path.isabs` to treat paths starting with `hdfs:/` as absolute."
- "`AgentConfig.__post_init__`:"
- "Port allocation (uses `find_free_port` and env overrides):"
- "`agent_port` from `PORT2` or free."
- "### `monolith/base_runner.py`"
- "`write_summary(logs, summary_writer, current_step)`: - Creates a new TF v1 Graph context."
- "### `monolith/core/base_host_call.py`"
- "`compress_tensors()`: - Groups tensors by dtype; concatenates each dtype list along axis 0, then `expand_dims` to add batch dimension."
- "`decompress_tensors(tensors)`:"
- "Asserts first tensor name is `global_step` (error message uses first character only)."
- "### `monolith/core/base_embedding_task.py`"
- "`download_vocab_size_file_from_hdfs()`:"
- "Runs `hadoop fs -copyToLocal` to temp folder."
- "`create_input_fn(mode=TRAIN)`:"
- "If `enable_stopping_signals`:"
- "`dataset = user_provided_dataset.concatenate(final_batch_dataset)`."
- "`qr_multi_hashing` and `vocab_size > qr_hashing_threshold`:"
- "Deletes original `example[f\"{fc_name}_0\"]`."
- "### `monolith/core/base_layer.py`"
- "`__getattr__`:"
- "Else raises `\"<name> is not a sub-layer of <self>.\"`."

## Part: task/request.part04.md

Lines: 4582-8387 (1-based, inclusive)

- "`params()`:"
- "- `accelerator` (None/\"tpu\"/\"horovod\")"
- "- `create_input_fn(mode)` and `create_model_fn()` are abstract and must be overridden."
- "`runWithCPU(task_name)`:"
- "`runMergeVectorTestOnCPU(task_name)`:"
- "suite() creates a `unittest.TestSuite` containing the four test classes."
- "Ensures dtype is floating/complex, else `TypeError`."
- "Requires last_dim defined; else `ValueError`."
- "`FeatureSlice`:"
- "`__repr__` returns `[FeatureSlice][slot_{slot_id}][{slice_index}]` (used as dict key)."
- "`__hash__` uses `(feature_slot.slot_id(), slice_index)`."
- "`Env.is_finalized` method recurses (`return self.is_finalized`) instead of returning `_is_finalized`."
- "`_embedding_lookup` uses undefined `slot_id` in QR branch; should likely be `feature_column._feature_slot.slot_id()`."
- "`record_summary_tensor(name, tensor)`:"
- "Asserts name unique and tensor rank <=1."
- "`compress_tensors()` / `decompress_tensors()`:"
- "Asserts first name is `\"global_step\"` (same message quirk with `[0][0]`)."
- "`Params`:"
- "`__setattr__` / `__setitem__`: if immutable -> `TypeError(\"This Params instance is immutable.\")`."
- "`_get_nested(\"a.b[0].c\")`:"
- "Supports list indexing in a segment via `name[index]`."
- "`test_similar_keys`:"
- "`actuvation (did you mean: [activation,activations])`."
- "`teacher_embedding_sizes_list` is ignored in `__init__` (always `None`)."
- "`call()` returns `mixed_embedding` in teacher path, but it is never defined."
- "Note: code defines `module_str` with `path` but `path` is undefined (bug); actual import uses `.params.params`."
- "Shortcut key behavior must be preserved for compatibility with existing names."
- "`optimizers` contains:"
- "`BarrierAlreadyPlacedError`: raised when attempting to place a barrier twice."
- "`generate_session_config(cluster_and_task=None)`:"
- "If `None`, returns `ConfigProto(allow_soft_placement=True)`."
- "`Client.__init__()`:"
- "Port from `CONSUL_HTTP_PORT` or `2280`."
- "`MONOLITH_WITH_HOROVOD` must be set **before** importing `monolith.native_training`."
- "Defines `monolith_chief_alert_proto` flag; chief maps it to `FLAGS.monolith_alert_proto`."
- "Flags defined: `data_service_dispatcher`, `dataset_use_dataservice`, `dataset_input_patterns`, `dataset_input_use_snappy`, `dataset_input_compression_type`, `dataset_input_use_parquet`, `dataset_input_use_tfrecord`, `dataset_worker_idx`, `dataset_num_workers`, `kafka_other_metadata`."

## Part: task/request.part05.md

Lines: 8388-11812 (1-based, inclusive)

- "End-to-end test that converts TF Example records to Monolith Example variants via `tf_example_to_example`, then parses with `parse_examples` and asserts fid/dense defaults."
- "writes `/tmp/test.tfrecord` with 10k TF Examples; uses TF1 session."
- "Disables TF2 behavior (`tf.compat.v1.disable_v2_behavior()`), uses TF1 session graph."
- "`calc_hash_value(val)` returns `int(log2(abs(val)+1))`."
- "Dataset pipeline:"
- "`TFRecordDataset` \u2192 `map(tf_example_to_example)` with:"
- "`sparse_features={'feature0':1,'feature1':2,'feature4':3}` (fid_v2 slots)"
- "`dense_features=['feature2']`"
- "`label='feature3'`"
- "`instance_weight=None`"
- "`instance_weight` defaults to `[1.0,1.0]`."
- "The hash for float sparse feature4 is `int(log2(abs(val)+1))`; must match exactly."
- "Rust tests should use tempdir paths."
- "If `file_name` is empty string, treats input as stdin and calls `ckpt_hooks.disable_iterator_save_restore()`."
- "files_list=None defaults to `['']` (stdin)."
- "If a single file and no glob expansion/sharding/dynamic sharding, returns `PBInstanceDatasetV2` directly; validates existence when file is non-empty and logs fatal on missing file."
- "Final dataset uses `interleave` with `cycle_length`, `block_length`, `num_parallel_calls`, `deterministic=False`."
- "Missing file uses `logging.fatal` in TF; decide equivalent behavior in Rust (panic or error)."
- "Creates `PBInstanceDataset(file_name='', has_sort_id=True, kafka_dump_prefix=True)` (stdin path)."
- "`batch(32)` and `map(parse)`."
- "Builds one-shot iterator, fetches one batch, logs `elements['sample_rate']`."
- "Add a Rust test that simulates stdin input (e.g., pipe fixture data into the dataset reader)."
- "Applies `dataset.negative_gen` with:"
- "`neg_num=7`, `channel_slot`, `group_slots`, `per_channel_sample=True`, `start_num=0`, `max_group_num_per_channel=10000`, `label_field='actions'`, `label_index=0`, `negative_label=-2`, `use_neg_ins=True`."
- "Asserts in first batch that `channel_res[0][0] == channel_res[0][i]` for i in 1..7 (negatives share channel), and label at index 1 equals `NEGATIVE_LABEL`."
- "Collects ~1024 samples; groups by channel and verifies ring buffer behavior:"
- "Depends on `instance.pb` fixture and custom ops; ensure Rust has compatible dataset and parse op support."
- "Adds defaults: `misc_float_features=['sample_rate']`, `misc_int64_features=['req_time','uid']`, `misc_repeated_float_features=['label']`."
- "Reshapes non-repeated misc float/int64 features to 1-D (`tf.reshape(features[key], [-1])`)."
- "Builds `ragged_keys` from `fidv1_features` (via `get_slot_feature_name`) plus `fidv2_features`."
- "Missing ragged fid slot \u2192 empty ragged (`[[]]`)."
- "Global `deque` used to store parse step callables."
- "Pops parse steps from `_extra_parse_steps` in FIFO order and applies each to `features`."
- "Validates `op` in `{gt,ge,eq,lt,le,neq,between,in,not-in,all,any,diff,startswith,endswith}`."
- "Validates `field_name` exists in `LineId` descriptor; `operand` is not None."
- "Compose(transforms):"
- "`as_proto` merges each transform\u2019s proto into a single `TransformConfig` via `MergeFrom` in order."
- "LogicalOr(x, y):"
- "Requires both `x` and `y` are leaf nodes."
- "`TOBENV` default `False` toggles slot name prefix (`slot_` vs `fc_slot_`)."
- "Else use `USED_FREATUE_NAMES`: assign a new slot id (`len+1`) if missing and return it."
- "Uses global mutable state; Rust must be careful about concurrency or test isolation."
- "`--type` must be `debugging_variables` or `debugging_features`."
- "Buckets by PS index using `fid % num_ps`."
- "`node_name(name)`:"
- "Strips whitespace, trailing `/`, leading `^`, and `:0` suffix if numeric."
- "For partitioned vars (multiple new names):"
- "Splits old tensor by first-dimension sizes from placeholders (`np.split`) and assigns each split to its placeholder."
- "`_GPU_PLACEMENT_ALLOWED` default False; `enable_gpu_training()` sets True; `disable_gpu_training()` sets False; `is_gpu_training()` returns it."
- "skip_device(op):"
- "Returns True for summary ops (`Write*`, `*Summary`) or string `Const` ops."
- "get_visible_gpus(local_rank, processes_per_gpu=1):"
- "Ensures `processes_per_gpu` is int >= 1; returns string of `local_rank / processes_per_gpu` as GPU index."
- "Builds a dynamic sharding dataset that expands glob patterns on demand using shared queues and a TF session-backed generator."
- "Builds `dataset_ops.MapDataset` over a dummy infinite dataset; maps to `tf.py_function(filename_generator)` with `preserve_cardinality=False`."
- "If still empty, returns `(\"\", True)`."
- "Runs `ip addr show <interface>` to compute host IP; exports as `UCX_RDMA_CM_SOURCE_ADDRESS` and `DMLC_NODE_HOST`."
- "If `CUDA_VISIBLE_DEVICES` not set but MPI local rank present, set `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES=<local_rank>`."
- "Returns `tf.slice(scattered, [0,0,0], [-1, max_seq_length, -1])` to enforce sequence length."

## Part: task/request.part06.md

Lines: 11813-15822 (1-based, inclusive)

- "Env flags read at import time:"
- "`MONOLITH_WITH_HOROVOD`, `MONOLITH_WITH_BYTEPS`, `MONOLITH_WITH_BYTEPS_ALLREDUCE`, `MONOLITH_WITH_ALLREDUCE_FUSION`, `MONOLITH_WITH_ALLREDUCE_FP16`, `MONOLITH_SKIP_ALLREDUCE`."
- "`allreduce_cond(grads, scale=1)`:"
- "Fusion modes:"
- "`one`: uses `monolith_aligned_flat_concat` + allreduce + `monolith_aligned_flat_split`."
- "`grouped`: uses `hvd.grouped_allreduce` (not supported with BytePS)."
- "`multi`: raises `RuntimeError` (dropped)."
- "`GradClipType` enum: `ClipByNorm`, `ClipByGlobalNorm`, `ClipByValue`, `ClipByDenseAndSparse`, `NoClip`."
- "`_gen_norm_warmup(clip_norm, global_step_var, warmup_step)`:"
- "Returns `clip_norm` scaled linearly from 0 to 1 over `warmup_step` using `tf.cond`."
- "`apply_gradients_with_var_optimizer(...)`:"
- "Applies dense grads via custom per-variable optimizer or shared `var_opt` (async via `ctx.add_async_function`)."
- "Applies embedding grads via `ctx.apply_embedding_gradients` (on CPU) with optional scale."
- "Increments `global_step` after optimize ops with control dependencies."
- "`apply_gradients(...)`:"
- "If no dense variables, still increments global_step."
- "Python tests: `monolith/native_training/feature_utils_test.py`."
- "`test_apply_gradients_with_dense_optimizer`:"
- "After one step: dense var becomes 0.5; embedding var becomes `[0.5,0.5,0.5,1.0]`; global_step=1."
- "`test_apply_gradients_with_dense_optimizer_post_push`:"
- "Async embedding push enabled; running op three times triggers two async pushes."
- "Dense var becomes -1.0; embedding var becomes `[-2.0,-2.0,-2.0,1.0]`."
- "`gen_seq_mask`:"
- "Calls `ops.gen_seq_mask(splits=..., max_seq_length=...)`."
- "For splits `[0,5,7,9,13]` and `max_seq_length=6`, mask equals expected matrix for both int32 and int64."
- "`gflags_utils.py`:"
- "`extract_help_info` parses `:param` lines and returns normalized help strings."
- "`extract_flags` defines flags for type-hinted fields (int/bool/str/float/enum) with defaults; skips missing help or skip list."
- "`update` applies flags to config when config field is default and flag is non-default."
- "`update_by_flags` patches `__init__` to apply linked flags when field is default."
- "`graph_meta.py`:"
- "Uses graph collection `monolith_graph_meta` to store a single dict."
- "`graph_utils.py`:"
- "Selects ops where `\"AssignMovingAvg\"` is in the op name and `op.type == \"AssignSubVariableOp\"`."
- "`hash_filter_ops.py` constants:"
- "`HASH_FILTER_CAPACITY=300000000`, `HASH_FILTER_SPLIT_NUM=7`, `_TIMEOUT_IN_MS=1800000`."
- "`FilterType`: string constants `SLIDING_HASH_FILTER`, `PROBABILISTIC_FILTER`, `NO_FILTER`."
- "`save_hash_filter(hash_filter, basename, enable_hash_filter)`:"
- "If enabled, uses `monolith_hash_filter_save` custom op; else returns `tf.no_op()`."
- "Registered gradient `MonolithHashFilterInterceptGradient`:"
- "Returns `(None, None, filtered_grad)`."
- "`hash_table_ops.py`:"
- "`BaseHashTable` abstract API: `assign`, `assign_add`, `lookup`, `apply_gradients`, `as_op`, `dim_size`."
- "`fused_lookup`:"
- "Calls `monolith_hash_table_fused_lookup` and returns embeddings, recv_splits, id_offsets, emb_offsets."
- "`fused_apply_gradient`:"
- "Calls `monolith_hash_table_fused_optimize` with ids, indices, fused_slot_size, grads, offsets, learning rates, req_time, global_step."
- "`hash_table_utils.py`:"
- "`iterate_table_and_apply(table, apply_fn, limit=1000, nshards=4, name=\"IterateTable\")`:"
- "Repeatedly calls `table.save_as_tensor(i, nshards, limit, offset)` until dump size < limit and offset != 0."
- "`infer_dim_size(config)`:"
- "Sums `segment.dim_size` across `config.entry_config.segments`."
- "`hooks/ckpt_hooks.py`:"
- "`BarrierSaverListener`:"
- "On `before_save`, places a barrier and waits for workers to block (up to max_pending_seconds)."
- "`disable_iterator_save_restore()`:"
- "Disables iterator save/restore globally (must be called before helper creation)."
- "`hooks/controller_hooks.py`:"
- "`QueryActionHook`:"
- "Polls `<model_dir>/monolith_action` every `QUERY_INTERVAL` seconds in a background thread."
- "Writes response to `<model_dir>/monolith_action_response` and deletes query file."
- "`hooks/feature_engineering_hooks.py`:"
- "Writes each serialized ExampleBatch preceded by two `<Q` headers: 0 (lagrange) and size."
- "`hooks/session_hooks.py`:"
- "`get_current_session()` returns `_INFO.session` if set, else `tf.compat.v1.get_default_session()`."
- "`hvd_lib.py`:"
- "Lazy import wrapper for Horovod or BytePS TensorFlow libraries."

## Part: task/request.part07.md

Lines: 15823-19762 (1-based, inclusive)

- "`MergeType`: string constants `concat`, `stack`, `None`."
- "`DCNType`: string constants `vector`, `matrix`, `mixed`."
- "`check_dim(dim)`: - `None` → `-1`, `int` → itself, `tf.compat.v1.Dimension` → `.value`, else raise."
- "`dim_size(inputs, axis)`: - Uses static shape; if unknown (`-1`), returns dynamic `array_ops.shape(inputs)[axis]`."
- "`merge_tensor_list(tensor_list, merge_type='concat', num_feature=None, axis=1, keep_list=False)`:"
- "Accepts tensor or list; if single tensor, uses shape to decide:"
- "3D: `stack` returns `[tensor]` or tensor; `concat` reshapes to `[B, num_feat*emb]`; `None` unstack on axis."
- "2D with `num_feature>1`: `stack` reshapes to `[B, num_feature, emb]`; `concat` returns as-is; `None` unstack."
- "2D without `num_feature`: returns as-is."
- "Else: raise shape error."
- "For list length >1: `stack`, `concat`, or return list."
- "`gumbel_keys(w)`: samples Gumbel noise and adds to `w`."
- "`continuous_topk(w, k, t, separate=False)`: - Iteratively computes soft top-k masks; returns sum or list."
- "`sample_subset(w, k, t=0.1)`: - `w = gumbel_keys(w)` then `continuous_topk`."
- "Rust mapping hint: \"Target crate/module: `monolith-rs/crates/monolith-layers/src/merge.rs` for merge utilities; DCNType maps to `monolith-layers/src/dcn.rs` (`DCNMode`).\""
- "Rust mapping hint: \"`MergeType::None` corresponds to `MergeOutput::List`.\""
- "Gap note: \"Gumbel subset sampling utilities are missing in Rust; add if used elsewhere.\""
- "`LearningRateFunction`: - Abstract `__call__` that must be overridden. - `__str__` prints class name and sorted `__dict__` params."
- "`PolynomialDecay`:"
- "Stores init params: `initial_learning_rate`, `decay_steps`, `end_learning_rate`, `power`, `cycle`, `name`."
- "`__call__` fetches `global_step = tf.compat.v1.train.get_or_create_global_step()` and returns `tf.compat.v1.train.polynomial_decay(...)`."
- "Gap note: \"Python relies on TF global step; Rust will need explicit step input or global trainer context.\""
- "`LoggingOps` wrappers:"
- "`tensors_timestamp(tensors)`: returns `(tensors, timestamp)` via `monolith_tensors_timestamp`."
- "`emit_timer(key, value, tags=None)`: - Formats tags as `\"k=v|k2=v2\"`, passes to `monolith_metric_v2`."
- "`machine_info(mem_limit=None, shared_name=None)`: - Uses default flag if `mem_limit` is None."
- "`check_machine_health(machine_info_tensor)`: - Returns scalar string tensor from `monolith_check_machine_health`."
- "Custom-op dependency note: \"Requires custom TF ops; currently no Rust bindings.\""
- "`batch_softmax_loss` validation: \"`temperature` must be > 0 else raise `ValueError(\"temperature should be positive, while got ...\")`.\""
- "`batch_softmax_loss` computation:"
- "Optional L2-normalize query/item along axis 1."
- "`similarity = query @ item^T / temperature`."
- "Clamp `item_step_interval` to at least 1.0, compute `item_frequency = 1 / item_step_interval`."
- "Adjust similarity: `exp(similarity - log(item_frequency))`."
- "Loss: `-sum(r * log(diag(similarity) / reduce_sum(similarity, axis=1)))`."
- "Test note: \"Python test uses random inputs without setting a seed but asserts a fixed value; likely flaky.\""
- "`inbatch_auc_loss` op wrapper: \"Wrapper for custom TF op `InbatchAucLoss` and its gradient registration.\""
- "Gradient note: \"`InbatchAucLoss` gradient returns `None` for label and computed gradient for logit via `inbatch_auc_loss_grad`.\""
- "`ltr_losses.py` constant: \"`_EPSILON = 1e-10` used to set invalid logits to a log probability for softmax/ListMLE.\""
- "`label_valid_fn(labels)`: \"Returns boolean tensor for label validity: `labels >= 0`.\""
- "`sort_by_scores(scores, features_list, topn=None)`: \"`scores` must be rank-2 `[batch_size, list_size]`; asserts rank 2.\""
- "`organize_valid_indices(is_valid, shuffle=True, seed=None)`: \"Invalid entries get sentinel score `-1e-6` so they appear last.\""
- "`approx_ranks(logits, alpha=10.)`: \"Computes approximate ranks via generalized sigmoid of pairwise differences\" and \"O(list_size^2) memory.\""
- "`inverse_max_dcg(...)`: \"Returns `1/discounted_gain` when `discounted_gain > 0`, else 0; shape `[batch,1]`.\""
- "`get_batch_idx_size(...)`: \"Keep exact scatter shapes and `-1e-6` offset; even if unused elsewhere.\""
- "`make_loss_fn(...)` validation: \"Validates `reduction` is in `tf.losses.Reduction.all()` and not `NONE`, else raises `ValueError(\"Invalid reduction: ...\")`.\""
- "`make_loss_fn(...)` validation: \"Raises `ValueError` if `loss_keys` empty or `loss_weights` length mismatch.\""
- "`make_loss_fn(...)` error: \"Unknown `loss_key` raises `ValueError(\"Invalid loss_key: ...\")`.\""
- "`_list_mle_loss(...)` invalid-logit sentinel: \"Invalid logits -> `log(_EPSILON)`.\""
- "`_approx_ndcg_loss(...)` invalid-logit sentinel: \"Invalid logits -> `min(logits) - 1e3`.\""
- "Doc/code discrepancy note: \"`_approx_ndcg_loss` docstring says weights default to label sum; code uses `ones_like(label_sum)`. Preserve code behavior.\""
- "`metric/cli.py` stub behavior: \"`Client.__getattr__(name)`: - Returns a function `method(*args, **kwargs)` that does nothing and returns `None`.\""
- "`deep_insight_client(..., container=socket.gethostname())`: \"Default `container` is evaluated at import time (module load), not at call time.\""
- "`metric/exit_hook.py` import-side effects: \"On import, installs handlers\" for `SIGHUP`, `SIGINT`, `SIGTERM` and registers `exit_hook()` via `@atexit.register`."
- "`exit_hook()` condition: \"Only emits counter if `sig_no is not None`: `mcli.emit_counter(\"exit_hook\", 1, tags)`.\""
- "`metric/kafka_utils.py` concurrency: \"Starts a background thread on init; sends messages to Kafka.\""
- "`KProducer.send(msgs)` filtering: \"Else filters iterable to only non-`None` entries with `len(msg) > 0`.\""
- "`KProducer.close()`: \"Sets `_has_stopped=True` under lock. - Joins background thread. - Calls `_flush()` and then `producer.close(timeout=1)`.\""
- "`metric/metric_hook.py` global side effect: \"Importing `exit_hook` executes signal/atexit registration for metrics.\""
- "Potential runtime issue note: \"`WriteOnlyFileAndStat`\" - \"uses `List`/`Dict` typing annotations without importing them (potential NameError at runtime).\""
- "Potential caller requirement note: \"`FileMetricHook` will fail if `key_fn` is not provided; ensure callers pass `vepfs_key_fn` or a custom function.\""
- "`metric/utils.py` deep insight disable path: \"Requires `features[\"req_time\"]`; if missing: - Logs \"Disabling deep_insight because req_time is absent\". - Returns `tf.no_op()`.\""
- "Mutability note: \"`extra_fields_keys` default list is mutated when adding `\"uid\"`.\""
- "`mlp_utils.py` bug note: \"`begin()` references `graph.clear_collection` but `graph` is undefined (likely bug).\""
- "`model_export/demo_predictor_client.py` bug note: \"code references `FLAGS.batch_size` but flag is not defined in this file (bug).\""
- "`model_export/warmup_data_gen.py` bug note: \"**bug**: returns `tf.int46` for int64 cases (invalid dtype).\""
- "`native_task_context.with_ctx(ctx)` semantics: \"If `old_ctx` is `None`, leaves `_CTX` set to `ctx` (no reset to `None`).\""

## Part: task/request.part08.md

Lines: 19763-23700 (1-based, inclusive)

- "`monolith/native_training/nested_tensors.py` side effects: \"Mutates dict/list/tuple structures passed into `_iterate` (and thus into `NestedTensors.__init__`) by replacing leaves with object IDs.\""
- "`monolith/native_training/nested_tensors.py` `_iterate(nested, action)`: \"Recurses through nested structures; for each leaf, applies `action(leaf)` and replaces leaf with the return value.\""
- "`monolith/native_training/nested_tensors.py` `NestedTensors(nested)`: \"Calls `_iterate(self._nested, self._add_tensor)` so **all leaves become object IDs**.\""
- "`monolith/native_training/nested_tensors.py` ragged constraint: \"`tf.RaggedTensor` → requires `ragged_rank == 1`; else raises `ValueError(\\\"Nested tensor doesn't support nested RaggedTensor.\\\")`.\""
- "`monolith/native_training/nested_tensors.py` dict ordering note: \"Order of dict iteration affects reconstruction order; Python preserves insertion order.\""
- "`monolith/native_training/net_utils.py` constructor behavior: \"Immediately starts `_start()` (blocks until all threads finish).\""
- "`monolith/native_training/net_utils.py` error side effect: \"Prints connection errors to stdout.\""
- "`monolith/native_training/runtime/ops/gen_monolith_ops.py` import-time side effect: \"Loads the Monolith custom op shared library\" and \"Calls `tf.load_library(utils.get_libops_path(\\\"monolith/native_training/runtime/ops/libtfkernel_monolith_ops_for_load.so\\\"))` on import.\""
- "`monolith/native_training/signal_utils.py` import-time side effect: \"Module import calls `add_siguser1_handler()` immediately.\""
- "`monolith/native_training/ragged_utils.py` caching semantics: \"Caches result on the `RaggedTensor` instance via `monolith_fused_value_rowids` attribute.\""
- "`monolith/native_training/prefetch_queue.py` capacity=0 passthrough: \"If `capacity == 0`, returns `(tensors, None)` with no queue.\""
- "`monolith/native_training/prefetch_queue_test.py` test gotcha: \"There are two methods named `test_enqueue_dicts_with_queue_return`; the second overrides the first, so only the second runs.\""
- "`monolith/native_training/touched_key_set_ops.py` potential bug: \"`TouchedKeySet.__init__` ignores the `name_suffix` argument (potential bug / resource collision risk).\""
- "`monolith/native_training/touched_key_set_ops.py` overflow behavior: \"Overflow behavior is \\\"clear-all\\\" once size exceeds capacity (no partial eviction).\""
- "`monolith/native_training/yarn_runtime.py` env precedence: \"If `CLOUDNATIVE_INET_ADDR` env var exists: use first entry before comma.\" and \"Else if `YARN_INET_ADDR` exists: use that value.\""
