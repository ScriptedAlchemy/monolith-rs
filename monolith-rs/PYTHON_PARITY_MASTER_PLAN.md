# Monolith Python → Monolith-RS Parity Master Plan

**Last updated:** 2026-01-19
**Owner:** Monolith-RS
**Status:** IN PROGRESS (living document)

This is the **single source of truth** for **porting every line of Python in `monolith/`** into Rust (`monolith-rs/`) to reach **full or максимально close parity**. It is intentionally exhaustive and process-heavy so nothing is missed.

---

## 0) Non-Negotiables (Read First)

- We are targeting **line-for-line parity** in behavior and public surface area, not necessarily 1:1 API style.
- **Every Python file in `monolith/` must be accounted for** in this plan with a concrete Rust destination or an explicit, justified N/A.
- We must **preserve semantics** even if the Rust implementation uses different internal structures.
- **No vendoring** of TensorFlow runtime libraries. TF runtime must be optional and dynamic (Linux x86_64 only, best effort elsewhere).
- **Candle remains the default** inference backend unless a true TF SavedModel runtime is available.
- **All parity gaps must be listed**, tracked, and closed or explicitly justified.

---

## 1) Definition of “Parity”

Parity means:
- **Behavior:** Rust outputs match Python outputs given the same inputs and configuration.
- **Protocol:** gRPC, protobuf, and disk formats remain compatible.
- **Config:** CLI flags, config files, and environment variables behave the same.
- **Ops:** Custom TensorFlow ops and side effects are supported or mapped.
- **I/O:** File formats (TFRecord, SavedModel, checkpointing) are compatible.
- **Tests:** Rust tests cover the same scenarios as Python tests.

Parity **does NOT require**:
- Identical code structure.
- Identical APIs (Rust idioms allowed if external behavior is the same).

---

## 2) Scope Inventory (Every Python Area)

We will port **all** of:
- `monolith/agent_service/**`
- `monolith/core/**`
- `monolith/native_training/**`
- `monolith/monolith_workspace.bzl`, `monolith/tf_serving_workspace.bzl`
- `monolith/utils.py`, `monolith/path_utils.py`, `monolith/base_runner.py`, `monolith/tpu_runner.py`, `monolith/gpu_runner.py`
- Any Python entry points under `monolith/**` and `monolith/native_training/**`

We will also account for:
- `third_party/org_tensorflow/**` (patches that affect behavior)
- Any Python-dependent tooling or code-gen (feature lists, ops, TF Serving config)

---

## 3) Master Porting Workflow (Every Line Tracked)

### 3.1 Generate a **Line-Level Inventory**
**Goal:** Track every Python line and its Rust destination.

**Tasks:**
- Build a script to enumerate:
  - File path
  - Line count
  - Top-level symbols (classes, functions)
- Create `monolith-rs/PYTHON_PARITY_INDEX.md` containing:
  - One row per file with line count and status
  - A link to a per-file checklist

**Status:** DONE\n\n**Artifacts:**\n- `monolith-rs/PYTHON_PARITY_INDEX.md` (334 files enumerated)

### 3.2 Per-File Parity Checklist
**Goal:** Each Python file gets a dedicated Rust parity checklist.

**Tasks:**
- For every Python file, create a matching checklist file:
  - `monolith-rs/parity/<path>.md` (mirrors Python path)
  - Each checklist includes:
    - Function/class list
    - Behavior notes
    - Rust mapping (file + symbol)
    - Test coverage
    - Open gaps

**Status:** DONE\n\n**Artifacts:**\n- `monolith-rs/parity/**` (per-file checklist for every `monolith/**/*.py`)

### 3.3 Porting Discipline
**Rule:** No file is considered “done” until:
- Feature parity verified against Python tests or equivalent.
- Rust tests added.
- Input/output formats validated.
- Performance regressions documented.

---

## 4) Cross-Cutting Infrastructure to Port First

These enable all module ports:

### 4.1 Config & CLI Parity
- CLI flags match Python entry points.
- Environment variable behavior identical.
- JSON/YAML config parsing for training/serving.

### 4.2 Protobufs & gRPC
- Ensure all `.proto` files compiled and exposed.
- TFS prediction and model service compatibility.

### 4.3 Data Formats
- TFRecord read/write
- Example / ExampleBatch / Instance formats
- Feature list and feature parsing rules

### 4.4 Checkpoint & Export
- SavedModel or SavedModel-like export
- Embeddings and optimizer states
- Manifest compatibility

### 4.5 Runtime & Scheduling
- Distributed training runtime hooks
- Service discovery (ZK/Consul)
- Parameter sync

---

## 5) Module-by-Module Port Plan (Exhaustive)

### 5.1 `monolith/agent_service/**`
**Goal:** Full parity with agent service, model manager, replica management, TF Serving integration.

**Key Submodules:**
- `agent_service.py`
- `agent_controller.py`
- `agent_v1.py`, `agent_v3.py`
- `backends.py`, `replica_manager.py`, `model_manager.py`
- `tfs_client.py`, `tfs_wrapper.py`, `tfs_monitor.py`
- `utils.py`, `constants.py`

**Porting Steps (per file):**
1. Map gRPC service behaviors to `monolith-serving`.
2. Implement config + environment behavior.
3. Recreate client utilities for Predict/ModelStatus/Metadata.
4. Validate with Python test vectors.

**Status:** IN PROGRESS (mapping phase)

### 5.2 `monolith/core/**`
**Goal:** Full parity with model registry, task base classes, hyperparams, core layers.

**Key Files:**
- `base_layer.py`
- `base_task.py`
- `base_model_params.py`
- `hyperparams.py`
- `model.py`
- `model_registry.py`

**Status:** TODO (not started)

### 5.3 `monolith/native_training/**`
**Goal:** Full parity with estimator, data pipeline, distributed runtime, export.

**Subareas:**
- `data/**` (datasets, feature extraction, parsers)
- `runtime/**` (hash tables, parameter sync)
- `model_export/**`
- `distribution_ops.py`, `distributed_ps.py`

**Status:** TODO (not started)

### 5.4 `monolith/utils.py` + helpers
- `path_utils.py`
- `base_runner.py`, `gpu_runner.py`, `tpu_runner.py`

**Status:** TODO (not started)

---

## 6) TensorFlow Runtime Integration (Optional, but Required for Full Parity)

**Purpose:** Enable running true TF SavedModel graphs when Candle cannot match.

**Key Requirements:**
- Dynamic `libtensorflow` loading (no vendoring)
- Custom op loading (`TF_LoadLibrary`)
- Signature-based tensor mapping
- Output decoding consistent with Python

**Status:** TODO (tracked separately with sub-plan embedded into this doc)

---

## 7) Mapping Strategy (Python → Rust)

For each Python file:
- Choose a destination Rust crate/module.
- Define equivalent Rust API(s).
- Record any unavoidable deviations.
- Add bridging adapters if required to preserve call patterns.

---

## 8) Test Parity Strategy

- Mirror Python tests in Rust where possible.
- For graph/TF behavior, validate against actual SavedModel outputs.
- For gRPC/proto compatibility, use golden fixtures.
- Maintain a parity test suite that runs Python and Rust against the same inputs.

---

## 9) Validation & Auditing (Triple-Check)

Every milestone must include:
- **Check 1:** File-level parity checklist complete.
- **Check 2:** Integration test across module boundaries.
- **Check 3:** End-to-end run of a full workflow.

We will not declare parity until all three checks pass.

---

## 10) Live Update Protocol (Incremental Updates)

Every time we progress:
- Update this file with:
  - Added files mapped
  - Closed gaps
  - New blockers
- Update `PYTHON_PARITY_INDEX.md` (line inventory)
- Update per-file checklists

**Triple-check ritual for each update:**
1. Verify all files touched are listed in `PYTHON_PARITY_INDEX.md`.
2. Verify any new Rust code has mapped Python source lines.
3. Re-run or re-validate the relevant tests.

---

## 11) Immediate Next Actions (Start Here)

1. ✅ **Generate full Python file inventory** with line counts.
2. ✅ **Create `PYTHON_PARITY_INDEX.md`** with one row per file.
3. ✅ **Create per-file checklists** under `monolith-rs/parity/`.
4. ⏳ **Define module mapping table** (Python → Rust crate/module).
5. ⏳ **Start with `monolith/agent_service`** (serving parity is highest priority).

---

## 12) Triple-Check Log (Do Not Skip)

**Latest verification (2026-01-19):**
- Python files discovered: **334**
- Parity checklist files created: **334**
- Missing checklist files: **0**
- Extra checklist files: **0**

**How to re-check:**
- Re-run the inventory generator and count parity files.
- Ensure `PYTHON_PARITY_INDEX.md` row count equals checklist file count.

---

## 13) Initial Mapping Table (Phase 1)

This is the first incremental mapping pass. It will be expanded **file-by-file** until every Python file has an explicit Rust destination.

### 13.1 `monolith/agent_service/**` → `monolith-rs/crates/monolith-serving`

| Python File | Rust Target | Status | Notes |
|---|---|---|---|
| `monolith/agent_service/agent_service.py` | `monolith-rs/crates/monolith-serving/src/grpc_agent.rs` (gRPC), `monolith-rs/crates/monolith-serving/src/server.rs` (runtime) | TODO | Split responsibilities between coordination gRPC and serving runtime. |
| `monolith/agent_service/agent_controller.py` | `monolith-rs/crates/monolith-serving/src/*` | TODO | Needs model layout/config orchestration parity. |
| `monolith/agent_service/agent_v1.py` | `monolith-rs/crates/monolith-serving/src/*` | TODO | Defines agent logic; map to server + manager abstractions. |
| `monolith/agent_service/agent_v3.py` | `monolith-rs/crates/monolith-serving/src/*` | TODO | Advanced routing / layout logic; likely new Rust module. |
| `monolith/agent_service/backends.py` | `monolith-rs/crates/monolith-serving/src/*` | TODO | Backend abstraction for layout + storage. |
| `monolith/agent_service/replica_manager.py` | `monolith-rs/crates/monolith-serving/src/*` | TODO | Replica discovery + updates. |
| `monolith/agent_service/model_manager.py` | `monolith-rs/crates/monolith-serving/src/*` | TODO | Model lifecycle + watcher loops. |
| `monolith/agent_service/tfs_client.py` | `monolith-rs/crates/monolith-serving/src/tfserving.rs` | TODO | Map client utilities to Rust TF Serving client. |
| `monolith/agent_service/tfs_wrapper.py` | `monolith-rs/crates/monolith-serving/src/tfserving.rs` | TODO | Wrapper logic around Predict/ModelStatus. |
| `monolith/agent_service/tfs_monitor.py` | `monolith-rs/crates/monolith-serving/src/*` | TODO | Monitoring + metrics; may need new module. |
| `monolith/agent_service/utils.py` | `monolith-rs/crates/monolith-serving/src/*` | TODO | Config parsing, model specs, helper utilities. |
| `monolith/agent_service/constants.py` | `monolith-rs/crates/monolith-serving/src/*` | TODO | Constants + enums. |
| `monolith/agent_service/data_def.py` | `monolith-rs/crates/monolith-serving/src/*` | TODO | Data definition structures. |
| Tests in `monolith/agent_service/*_test.py` | `monolith-rs/crates/monolith-serving/tests/*` | TODO | Port test cases and fixtures. |

---

### 13.2 `monolith/core/**` → `monolith-rs/crates/monolith-core` + `monolith-rs/crates/monolith-layers`

**Status:** TODO (mapping pending)
**Notes:** Use per-file checklists under `monolith-rs/parity/monolith/core/*.md` as the canonical mapping source until this table is populated.

---

### 13.3 `monolith/native_training/**` → `monolith-rs/crates/monolith-training`, `monolith-rs/crates/monolith-data`, `monolith-rs/crates/monolith-checkpoint`, `monolith-rs/crates/monolith-serving`

**Status:** TODO (mapping pending)
**Notes:** Use per-file checklists under `monolith-rs/parity/monolith/native_training/**/*.md` as the canonical mapping source until this table is populated.

---

## 14) Notes / Open Questions

- Which subset of Python behaviors should be prioritized for the first parity milestone?
- Are we targeting full parity for GPU/TPU runtime semantics, or “functional parity” only?
- Should we keep Python code synced into `monolith-rs/python/` or treat it as external?

---

## 15) Appendix: Placeholders for Future Expansion

- **Detailed TF runtime integration steps**
- **Custom op inventory + build steps**
- **Data pipeline micro-parity checklists**
- **CLI command mapping table**
- **ZK/Consul parity matrix**

---

> This document will be expanded with concrete, line-level checklists and mapping tables next. No file is left untracked.

## Line-Level Inventory (All Python Files)

This table enumerates **every** Python file under `monolith/` with line counts and a direct link to its checklist section.

| Python File | Lines | Status | Rust Mapping | Notes |
|---|---:|---|---|---|
| [`monolith/__init__.py`](#monolith-init-py) | 55 | TODO | TODO (manual) |  |
| [`monolith/agent_service/__init__.py`](#monolith-agent-service-init-py) | 0 | IN PROGRESS | monolith-rs/crates/monolith-serving/src | |
| [`monolith/agent_service/agent.py`](#monolith-agent-service-agent-py) | 100 | IN PROGRESS | monolith-rs/crates/monolith-cli, monolith-rs/crates/monolith-serving/src | |
| [`monolith/agent_service/agent_base.py`](#monolith-agent-service-agent-base-py) | 88 | IN PROGRESS | monolith-rs/crates/monolith-serving/src | |
| [`monolith/agent_service/agent_client.py`](#monolith-agent-service-agent-client-py) | 216 | IN PROGRESS | monolith-rs/crates/monolith-cli/src/bin/agent_client.rs | |
| [`monolith/agent_service/agent_controller.py`](#monolith-agent-service-agent-controller-py) | 145 | IN PROGRESS | monolith-rs/crates/monolith-cli/src/bin/agent_controller.rs | |
| [`monolith/agent_service/agent_controller_test.py`](#monolith-agent-service-agent-controller-test-py) | 95 | IN PROGRESS | monolith-rs/crates/monolith-cli/tests | |
| [`monolith/agent_service/agent_service.py`](#monolith-agent-service-agent-service-py) | 155 | IN PROGRESS | monolith-rs/crates/monolith-serving/src | |
| [`monolith/agent_service/agent_service_test.py`](#monolith-agent-service-agent-service-test-py) | 107 | IN PROGRESS | monolith-rs/crates/monolith-serving/tests | |
| [`monolith/agent_service/agent_v1.py`](#monolith-agent-service-agent-v1-py) | 390 | IN PROGRESS | monolith-rs/crates/monolith-serving/src | |
| [`monolith/agent_service/agent_v3.py`](#monolith-agent-service-agent-v3-py) | 210 | IN PROGRESS | monolith-rs/crates/monolith-serving/src | |
| [`monolith/agent_service/agent_v3_test.py`](#monolith-agent-service-agent-v3-test-py) | 114 | IN PROGRESS | monolith-rs/crates/monolith-serving/tests | |
| [`monolith/agent_service/backends.py`](#monolith-agent-service-backends-py) | 518 | IN PROGRESS | monolith-rs/crates/monolith-serving/src | |
| [`monolith/agent_service/backends_test.py`](#monolith-agent-service-backends-test-py) | 134 | IN PROGRESS | monolith-rs/crates/monolith-serving/tests | |
| [`monolith/agent_service/client.py`](#monolith-agent-service-client-py) | 126 | IN PROGRESS | monolith-rs/crates/monolith-cli/src/bin/serving_client.rs | |
| [`monolith/agent_service/constants.py`](#monolith-agent-service-constants-py) | 15 | IN PROGRESS | monolith-rs/crates/monolith-serving/src | |
| [`monolith/agent_service/data_def.py`](#monolith-agent-service-data-def-py) | 171 | IN PROGRESS | monolith-rs/crates/monolith-serving/src | |
| [`monolith/agent_service/data_def_test.py`](#monolith-agent-service-data-def-test-py) | 52 | IN PROGRESS | monolith-rs/crates/monolith-serving/tests | |
| [`monolith/agent_service/mocked_tfserving.py`](#monolith-agent-service-mocked-tfserving-py) | 399 | IN PROGRESS | monolith-rs/crates/monolith-serving/tests/support | |
| [`monolith/agent_service/mocked_tfserving_test.py`](#monolith-agent-service-mocked-tfserving-test-py) | 92 | IN PROGRESS | monolith-rs/crates/monolith-serving/tests | |
| [`monolith/agent_service/mocked_zkclient.py`](#monolith-agent-service-mocked-zkclient-py) | 377 | IN PROGRESS | monolith-rs/crates/monolith-serving/tests/support | |
| [`monolith/agent_service/mocked_zkclient_test.py`](#monolith-agent-service-mocked-zkclient-test-py) | 130 | IN PROGRESS | monolith-rs/crates/monolith-serving/tests | |
| [`monolith/agent_service/model_manager.py`](#monolith-agent-service-model-manager-py) | 371 | IN PROGRESS | monolith-rs/crates/monolith-serving/src | |
| [`monolith/agent_service/model_manager_test.py`](#monolith-agent-service-model-manager-test-py) | 113 | IN PROGRESS | monolith-rs/crates/monolith-serving/tests | |
| [`monolith/agent_service/replica_manager.py`](#monolith-agent-service-replica-manager-py) | 835 | IN PROGRESS | monolith-rs/crates/monolith-serving/src | |
| [`monolith/agent_service/replica_manager_test.py`](#monolith-agent-service-replica-manager-test-py) | 126 | IN PROGRESS | monolith-rs/crates/monolith-serving/tests | |
| [`monolith/agent_service/resource_utils.py`](#monolith-agent-service-resource-utils-py) | 269 | IN PROGRESS | monolith-rs/crates/monolith-serving/src | |
| [`monolith/agent_service/resource_utils_test.py`](#monolith-agent-service-resource-utils-test-py) | 36 | IN PROGRESS | monolith-rs/crates/monolith-serving/tests | |
| [`monolith/agent_service/run.py`](#monolith-agent-service-run-py) | 39 | IN PROGRESS | monolith-rs/crates/monolith-cli/src/bin | |
| [`monolith/agent_service/svr_client.py`](#monolith-agent-service-svr-client-py) | 70 | IN PROGRESS | monolith-rs/crates/monolith-cli/src | |
| [`monolith/agent_service/tfs_client.py`](#monolith-agent-service-tfs-client-py) | 503 | IN PROGRESS | monolith-rs/crates/monolith-cli/src/bin/tfs_client.rs | |
| [`monolith/agent_service/tfs_client_test.py`](#monolith-agent-service-tfs-client-test-py) | 50 | IN PROGRESS | monolith-rs/crates/monolith-cli/tests | |
| [`monolith/agent_service/tfs_monitor.py`](#monolith-agent-service-tfs-monitor-py) | 303 | IN PROGRESS | monolith-rs/crates/monolith-serving/src | |
| [`monolith/agent_service/tfs_monitor_test.py`](#monolith-agent-service-tfs-monitor-test-py) | 182 | IN PROGRESS | monolith-rs/crates/monolith-serving/tests | |
| [`monolith/agent_service/tfs_wrapper.py`](#monolith-agent-service-tfs-wrapper-py) | 202 | IN PROGRESS | monolith-rs/crates/monolith-serving/src | |
| [`monolith/agent_service/utils.py`](#monolith-agent-service-utils-py) | 1167 | IN PROGRESS | monolith-rs/crates/monolith-serving/src | |
| [`monolith/agent_service/utils_test.py`](#monolith-agent-service-utils-test-py) | 170 | IN PROGRESS | monolith-rs/crates/monolith-serving/tests | |
| [`monolith/agent_service/zk_mirror.py`](#monolith-agent-service-zk-mirror-py) | 672 | IN PROGRESS | monolith-rs/crates/monolith-serving/src | |
| [`monolith/agent_service/zk_mirror_test.py`](#monolith-agent-service-zk-mirror-test-py) | 229 | IN PROGRESS | monolith-rs/crates/monolith-serving/tests | |
| [`monolith/base_runner.py`](#monolith-base-runner-py) | 46 | TODO | TODO (manual) |  |
| [`monolith/common/python/mem_profiling.py`](#monolith-common-python-mem-profiling-py) | 51 | TODO | TODO (manual) |  |
| [`monolith/core/__init__.py`](#monolith-core-init-py) | 0 | TODO | TODO (manual) |  |
| [`monolith/core/auto_checkpoint_feed_hook.py`](#monolith-core-auto-checkpoint-feed-hook-py) | 376 | TODO | TODO (manual) |  |
| [`monolith/core/base_embedding_host_call.py`](#monolith-core-base-embedding-host-call-py) | 643 | TODO | TODO (manual) |  |
| [`monolith/core/base_embedding_host_call_test.py`](#monolith-core-base-embedding-host-call-test-py) | 77 | TODO | TODO (manual) |  |
| [`monolith/core/base_embedding_task.py`](#monolith-core-base-embedding-task-py) | 611 | TODO | TODO (manual) |  |
| [`monolith/core/base_host_call.py`](#monolith-core-base-host-call-py) | 145 | TODO | TODO (manual) |  |
| [`monolith/core/base_layer.py`](#monolith-core-base-layer-py) | 161 | TODO | TODO (manual) |  |
| [`monolith/core/base_layer_test.py`](#monolith-core-base-layer-test-py) | 41 | TODO | TODO (manual) |  |
| [`monolith/core/base_model_params.py`](#monolith-core-base-model-params-py) | 25 | TODO | TODO (manual) |  |
| [`monolith/core/base_task.py`](#monolith-core-base-task-py) | 95 | TODO | TODO (manual) |  |
| [`monolith/core/base_tpu_test.py`](#monolith-core-base-tpu-test-py) | 73 | TODO | TODO (manual) |  |
| [`monolith/core/core_test_suite.py`](#monolith-core-core-test-suite-py) | 35 | TODO | TODO (manual) |  |
| [`monolith/core/dense.py`](#monolith-core-dense-py) | 179 | TODO | TODO (manual) |  |
| [`monolith/core/dense_test.py`](#monolith-core-dense-test-py) | 108 | TODO | TODO (manual) |  |
| [`monolith/core/feature.py`](#monolith-core-feature-py) | 611 | TODO | TODO (manual) |  |
| [`monolith/core/feature_test.py`](#monolith-core-feature-test-py) | 178 | TODO | TODO (manual) |  |
| [`monolith/core/host_call.py`](#monolith-core-host-call-py) | 248 | TODO | TODO (manual) |  |
| [`monolith/core/hyperparams.py`](#monolith-core-hyperparams-py) | 439 | TODO | TODO (manual) |  |
| [`monolith/core/hyperparams_test.py`](#monolith-core-hyperparams-test-py) | 277 | TODO | TODO (manual) |  |
| [`monolith/core/mixed_emb_op_comb_nws.py`](#monolith-core-mixed-emb-op-comb-nws-py) | 421 | TODO | TODO (manual) |  |
| [`monolith/core/model.py`](#monolith-core-model-py) | 320 | TODO | TODO (manual) |  |
| [`monolith/core/model_imports.py`](#monolith-core-model-imports-py) | 104 | TODO | TODO (manual) |  |
| [`monolith/core/model_registry.py`](#monolith-core-model-registry-py) | 174 | TODO | TODO (manual) |  |
| [`monolith/core/optimizers.py`](#monolith-core-optimizers-py) | 25 | TODO | TODO (manual) |  |
| [`monolith/core/py_utils.py`](#monolith-core-py-utils-py) | 313 | TODO | TODO (manual) |  |
| [`monolith/core/testing_utils.py`](#monolith-core-testing-utils-py) | 203 | TODO | TODO (manual) |  |
| [`monolith/core/tpu_variable.py`](#monolith-core-tpu-variable-py) | 214 | TODO | TODO (manual) |  |
| [`monolith/core/util.py`](#monolith-core-util-py) | 269 | TODO | TODO (manual) |  |
| [`monolith/core/util_test.py`](#monolith-core-util-test-py) | 150 | TODO | TODO (manual) |  |
| [`monolith/core/variance_scaling.py`](#monolith-core-variance-scaling-py) | 188 | TODO | TODO (manual) |  |
| [`monolith/gpu_runner.py`](#monolith-gpu-runner-py) | 226 | TODO | TODO (manual) |  |
| [`monolith/native_training/alert/alert_manager.py`](#monolith-native-training-alert-alert-manager-py) | 31 | TODO | TODO (manual) |  |
| [`monolith/native_training/alert/alert_manager_test.py`](#monolith-native-training-alert-alert-manager-test-py) | 32 | TODO | TODO (manual) |  |
| [`monolith/native_training/barrier_ops.py`](#monolith-native-training-barrier-ops-py) | 158 | TODO | TODO (manual) |  |
| [`monolith/native_training/barrier_ops_test.py`](#monolith-native-training-barrier-ops-test-py) | 104 | TODO | TODO (manual) |  |
| [`monolith/native_training/basic_restore_hook.py`](#monolith-native-training-basic-restore-hook-py) | 72 | TODO | TODO (manual) |  |
| [`monolith/native_training/basic_restore_hook_test.py`](#monolith-native-training-basic-restore-hook-test-py) | 137 | TODO | TODO (manual) |  |
| [`monolith/native_training/clip_ops.py`](#monolith-native-training-clip-ops-py) | 80 | TODO | TODO (manual) |  |
| [`monolith/native_training/clip_ops_test.py`](#monolith-native-training-clip-ops-test-py) | 92 | TODO | TODO (manual) |  |
| [`monolith/native_training/cluster_manager.py`](#monolith-native-training-cluster-manager-py) | 184 | TODO | TODO (manual) |  |
| [`monolith/native_training/cluster_manager_test.py`](#monolith-native-training-cluster-manager-test-py) | 35 | TODO | TODO (manual) |  |
| [`monolith/native_training/consul.py`](#monolith-native-training-consul-py) | 149 | TODO | TODO (manual) |  |
| [`monolith/native_training/consul_test.py`](#monolith-native-training-consul-test-py) | 59 | TODO | TODO (manual) |  |
| [`monolith/native_training/cpu_sync_training_test.py`](#monolith-native-training-cpu-sync-training-test-py) | 360 | TODO | TODO (manual) |  |
| [`monolith/native_training/cpu_training.py`](#monolith-native-training-cpu-training-py) | 2449 | TODO | TODO (manual) |  |
| [`monolith/native_training/cpu_training_distributed_test_binary.py`](#monolith-native-training-cpu-training-distributed-test-binary-py) | 226 | TODO | TODO (manual) |  |
| [`monolith/native_training/cpu_training_test.py`](#monolith-native-training-cpu-training-test-py) | 597 | TODO | TODO (manual) |  |
| [`monolith/native_training/data/__init__.py`](#monolith-native-training-data-init-py) | 20 | TODO | TODO (manual) |  |
| [`monolith/native_training/data/data_ops_test.py`](#monolith-native-training-data-data-ops-test-py) | 502 | TODO | TODO (manual) |  |
| [`monolith/native_training/data/data_service_parquet_test.py`](#monolith-native-training-data-data-service-parquet-test-py) | 145 | TODO | TODO (manual) |  |
| [`monolith/native_training/data/data_service_test.py`](#monolith-native-training-data-data-service-test-py) | 98 | TODO | TODO (manual) |  |
| [`monolith/native_training/data/datasets.py`](#monolith-native-training-data-datasets-py) | 1642 | TODO | TODO (manual) |  |
| [`monolith/native_training/data/eager_mode_test.py`](#monolith-native-training-data-eager-mode-test-py) | 186 | TODO | TODO (manual) |  |
| [`monolith/native_training/data/extract_fid_test.py`](#monolith-native-training-data-extract-fid-test-py) | 30 | TODO | TODO (manual) |  |
| [`monolith/native_training/data/feature_list.py`](#monolith-native-training-data-feature-list-py) | 409 | TODO | TODO (manual) |  |
| [`monolith/native_training/data/feature_list_test.py`](#monolith-native-training-data-feature-list-test-py) | 0 | TODO | TODO (manual) |  |
| [`monolith/native_training/data/feature_utils.py`](#monolith-native-training-data-feature-utils-py) | 1070 | TODO | TODO (manual) |  |
| [`monolith/native_training/data/feature_utils_test.py`](#monolith-native-training-data-feature-utils-test-py) | 1414 | TODO | TODO (manual) |  |
| [`monolith/native_training/data/item_pool_hook.py`](#monolith-native-training-data-item-pool-hook-py) | 109 | TODO | TODO (manual) |  |
| [`monolith/native_training/data/item_pool_test.py`](#monolith-native-training-data-item-pool-test-py) | 58 | TODO | TODO (manual) |  |
| [`monolith/native_training/data/kafka_dataset_test.py`](#monolith-native-training-data-kafka-dataset-test-py) | 239 | TODO | TODO (manual) |  |
| [`monolith/native_training/data/multi_flow_test.py`](#monolith-native-training-data-multi-flow-test-py) | 125 | TODO | TODO (manual) |  |
| [`monolith/native_training/data/negative_gen_test.py`](#monolith-native-training-data-negative-gen-test-py) | 253 | TODO | TODO (manual) |  |
| [`monolith/native_training/data/parse_sparse_feature_test.py`](#monolith-native-training-data-parse-sparse-feature-test-py) | 1833 | TODO | TODO (manual) |  |
| [`monolith/native_training/data/parsers.py`](#monolith-native-training-data-parsers-py) | 782 | TODO | TODO (manual) |  |
| [`monolith/native_training/data/tf_example_to_example_test.py`](#monolith-native-training-data-tf-example-to-example-test-py) | 183 | TODO | TODO (manual) |  |
| [`monolith/native_training/data/training_instance/python/instance_dataset_op.py`](#monolith-native-training-data-training-instance-python-instance-dataset-op-py) | 166 | TODO | TODO (manual) |  |
| [`monolith/native_training/data/training_instance/python/instance_dataset_op_test_stdin.py`](#monolith-native-training-data-training-instance-python-instance-dataset-op-test-stdin-py) | 58 | TODO | TODO (manual) |  |
| [`monolith/native_training/data/training_instance/python/instance_negative_gen_dataset_op_test.py`](#monolith-native-training-data-training-instance-python-instance-negative-gen-dataset-op-test-py) | 283 | TODO | TODO (manual) |  |
| [`monolith/native_training/data/training_instance/python/parse_instance_ops.py`](#monolith-native-training-data-training-instance-python-parse-instance-ops-py) | 245 | TODO | TODO (manual) |  |
| [`monolith/native_training/data/training_instance/python/parse_instance_ops_test.py`](#monolith-native-training-data-training-instance-python-parse-instance-ops-test-py) | 185 | TODO | TODO (manual) |  |
| [`monolith/native_training/data/training_instance/python/parser_utils.py`](#monolith-native-training-data-training-instance-python-parser-utils-py) | 85 | TODO | TODO (manual) |  |
| [`monolith/native_training/data/training_instance/python/pb_datasource_ops.py`](#monolith-native-training-data-training-instance-python-pb-datasource-ops-py) | 48 | TODO | TODO (manual) |  |
| [`monolith/native_training/data/training_instance/python/test_data_utils.py`](#monolith-native-training-data-training-instance-python-test-data-utils-py) | 15 | TODO | TODO (manual) |  |
| [`monolith/native_training/data/transform/transforms.py`](#monolith-native-training-data-transform-transforms-py) | 250 | TODO | TODO (manual) |  |
| [`monolith/native_training/data/transform/transforms_test.py`](#monolith-native-training-data-transform-transforms-test-py) | 70 | TODO | TODO (manual) |  |
| [`monolith/native_training/data/transform_dataset_test.py`](#monolith-native-training-data-transform-dataset-test-py) | 168 | TODO | TODO (manual) |  |
| [`monolith/native_training/data/utils.py`](#monolith-native-training-data-utils-py) | 55 | TODO | TODO (manual) |  |
| [`monolith/native_training/debugging/debugging_client.py`](#monolith-native-training-debugging-debugging-client-py) | 98 | TODO | TODO (manual) |  |
| [`monolith/native_training/debugging/debugging_server.py`](#monolith-native-training-debugging-debugging-server-py) | 217 | TODO | TODO (manual) |  |
| [`monolith/native_training/demo.py`](#monolith-native-training-demo-py) | 57 | TODO | TODO (manual) |  |
| [`monolith/native_training/dense_reload_utils.py`](#monolith-native-training-dense-reload-utils-py) | 457 | TODO | TODO (manual) |  |
| [`monolith/native_training/dense_reload_utils_test.py`](#monolith-native-training-dense-reload-utils-test-py) | 192 | TODO | TODO (manual) |  |
| [`monolith/native_training/device_utils.py`](#monolith-native-training-device-utils-py) | 231 | TODO | TODO (manual) |  |
| [`monolith/native_training/device_utils_test.py`](#monolith-native-training-device-utils-test-py) | 104 | TODO | TODO (manual) |  |
| [`monolith/native_training/distribute/distributed_dataset.py`](#monolith-native-training-distribute-distributed-dataset-py) | 81 | TODO | TODO (manual) |  |
| [`monolith/native_training/distribute/distributed_dataset_test.py`](#monolith-native-training-distribute-distributed-dataset-test-py) | 124 | TODO | TODO (manual) |  |
| [`monolith/native_training/distribute/str_queue.py`](#monolith-native-training-distribute-str-queue-py) | 114 | TODO | TODO (manual) |  |
| [`monolith/native_training/distribute/str_queue_test.py`](#monolith-native-training-distribute-str-queue-test-py) | 67 | TODO | TODO (manual) |  |
| [`monolith/native_training/distributed_ps.py`](#monolith-native-training-distributed-ps-py) | 2108 | TODO | TODO (manual) |  |
| [`monolith/native_training/distributed_ps_benchmark.py`](#monolith-native-training-distributed-ps-benchmark-py) | 168 | TODO | TODO (manual) |  |
| [`monolith/native_training/distributed_ps_factory.py`](#monolith-native-training-distributed-ps-factory-py) | 262 | TODO | TODO (manual) |  |
| [`monolith/native_training/distributed_ps_factory_test.py`](#monolith-native-training-distributed-ps-factory-test-py) | 87 | TODO | TODO (manual) |  |
| [`monolith/native_training/distributed_ps_sync.py`](#monolith-native-training-distributed-ps-sync-py) | 531 | TODO | TODO (manual) |  |
| [`monolith/native_training/distributed_ps_sync_test.py`](#monolith-native-training-distributed-ps-sync-test-py) | 109 | TODO | TODO (manual) |  |
| [`monolith/native_training/distributed_ps_test.py`](#monolith-native-training-distributed-ps-test-py) | 979 | TODO | TODO (manual) |  |
| [`monolith/native_training/distributed_serving_ops.py`](#monolith-native-training-distributed-serving-ops-py) | 160 | TODO | TODO (manual) |  |
| [`monolith/native_training/distributed_serving_ops_test.py`](#monolith-native-training-distributed-serving-ops-test-py) | 142 | TODO | TODO (manual) |  |
| [`monolith/native_training/distribution_ops.py`](#monolith-native-training-distribution-ops-py) | 889 | TODO | TODO (manual) |  |
| [`monolith/native_training/distribution_ops_benchmark.py`](#monolith-native-training-distribution-ops-benchmark-py) | 118 | TODO | TODO (manual) |  |
| [`monolith/native_training/distribution_ops_fused_benchmark.py`](#monolith-native-training-distribution-ops-fused-benchmark-py) | 61 | TODO | TODO (manual) |  |
| [`monolith/native_training/distribution_ops_fused_test.py`](#monolith-native-training-distribution-ops-fused-test-py) | 148 | TODO | TODO (manual) |  |
| [`monolith/native_training/distribution_ops_test.py`](#monolith-native-training-distribution-ops-test-py) | 536 | TODO | TODO (manual) |  |
| [`monolith/native_training/distribution_utils.py`](#monolith-native-training-distribution-utils-py) | 443 | TODO | TODO (manual) |  |
| [`monolith/native_training/embedding_combiners.py`](#monolith-native-training-embedding-combiners-py) | 102 | TODO | TODO (manual) |  |
| [`monolith/native_training/embedding_combiners_test.py`](#monolith-native-training-embedding-combiners-test-py) | 47 | TODO | TODO (manual) |  |
| [`monolith/native_training/entry.py`](#monolith-native-training-entry-py) | 630 | TODO | TODO (manual) |  |
| [`monolith/native_training/entry_test.py`](#monolith-native-training-entry-test-py) | 84 | TODO | TODO (manual) |  |
| [`monolith/native_training/env_utils.py`](#monolith-native-training-env-utils-py) | 32 | TODO | TODO (manual) |  |
| [`monolith/native_training/env_utils_test.py`](#monolith-native-training-env-utils-test-py) | 23 | TODO | TODO (manual) |  |
| [`monolith/native_training/estimator.py`](#monolith-native-training-estimator-py) | 667 | TODO | TODO (manual) |  |
| [`monolith/native_training/estimator_dist_test.py`](#monolith-native-training-estimator-dist-test-py) | 166 | TODO | TODO (manual) |  |
| [`monolith/native_training/estimator_mode_test.py`](#monolith-native-training-estimator-mode-test-py) | 417 | TODO | TODO (manual) |  |
| [`monolith/native_training/estimator_test.py`](#monolith-native-training-estimator-test-py) | 112 | TODO | TODO (manual) |  |
| [`monolith/native_training/feature.py`](#monolith-native-training-feature-py) | 663 | TODO | TODO (manual) |  |
| [`monolith/native_training/feature_test.py`](#monolith-native-training-feature-test-py) | 266 | TODO | TODO (manual) |  |
| [`monolith/native_training/feature_utils.py`](#monolith-native-training-feature-utils-py) | 419 | TODO | TODO (manual) |  |
| [`monolith/native_training/feature_utils_test.py`](#monolith-native-training-feature-utils-test-py) | 144 | TODO | TODO (manual) |  |
| [`monolith/native_training/file_ops.py`](#monolith-native-training-file-ops-py) | 51 | TODO | TODO (manual) |  |
| [`monolith/native_training/file_ops_test.py`](#monolith-native-training-file-ops-test-py) | 56 | TODO | TODO (manual) |  |
| [`monolith/native_training/fused_embedding_to_layout_test.py`](#monolith-native-training-fused-embedding-to-layout-test-py) | 1333 | TODO | TODO (manual) |  |
| [`monolith/native_training/gen_seq_mask.py`](#monolith-native-training-gen-seq-mask-py) | 26 | TODO | TODO (manual) |  |
| [`monolith/native_training/gen_seq_mask_test.py`](#monolith-native-training-gen-seq-mask-test-py) | 42 | TODO | TODO (manual) |  |
| [`monolith/native_training/gflags_utils.py`](#monolith-native-training-gflags-utils-py) | 282 | TODO | TODO (manual) |  |
| [`monolith/native_training/gflags_utils_test.py`](#monolith-native-training-gflags-utils-test-py) | 217 | TODO | TODO (manual) |  |
| [`monolith/native_training/graph_meta.py`](#monolith-native-training-graph-meta-py) | 30 | TODO | TODO (manual) |  |
| [`monolith/native_training/graph_utils.py`](#monolith-native-training-graph-utils-py) | 26 | TODO | TODO (manual) |  |
| [`monolith/native_training/hash_filter_ops.py`](#monolith-native-training-hash-filter-ops-py) | 326 | TODO | TODO (manual) |  |
| [`monolith/native_training/hash_filter_ops_test.py`](#monolith-native-training-hash-filter-ops-test-py) | 228 | TODO | TODO (manual) |  |
| [`monolith/native_training/hash_table_ops.py`](#monolith-native-training-hash-table-ops-py) | 738 | TODO | TODO (manual) |  |
| [`monolith/native_training/hash_table_ops_benchmark.py`](#monolith-native-training-hash-table-ops-benchmark-py) | 148 | TODO | TODO (manual) |  |
| [`monolith/native_training/hash_table_ops_test.py`](#monolith-native-training-hash-table-ops-test-py) | 1200 | TODO | TODO (manual) |  |
| [`monolith/native_training/hash_table_utils.py`](#monolith-native-training-hash-table-utils-py) | 50 | TODO | TODO (manual) |  |
| [`monolith/native_training/hash_table_utils_test.py`](#monolith-native-training-hash-table-utils-test-py) | 45 | TODO | TODO (manual) |  |
| [`monolith/native_training/hooks/ckpt_hooks.py`](#monolith-native-training-hooks-ckpt-hooks-py) | 193 | TODO | TODO (manual) |  |
| [`monolith/native_training/hooks/ckpt_hooks_test.py`](#monolith-native-training-hooks-ckpt-hooks-test-py) | 181 | TODO | TODO (manual) |  |
| [`monolith/native_training/hooks/ckpt_info.py`](#monolith-native-training-hooks-ckpt-info-py) | 98 | TODO | TODO (manual) |  |
| [`monolith/native_training/hooks/ckpt_info_test.py`](#monolith-native-training-hooks-ckpt-info-test-py) | 45 | TODO | TODO (manual) |  |
| [`monolith/native_training/hooks/controller_hooks.py`](#monolith-native-training-hooks-controller-hooks-py) | 170 | TODO | TODO (manual) |  |
| [`monolith/native_training/hooks/controller_hooks_test.py`](#monolith-native-training-hooks-controller-hooks-test-py) | 82 | TODO | TODO (manual) |  |
| [`monolith/native_training/hooks/feature_engineering_hooks.py`](#monolith-native-training-hooks-feature-engineering-hooks-py) | 99 | TODO | TODO (manual) |  |
| [`monolith/native_training/hooks/hook_utils.py`](#monolith-native-training-hooks-hook-utils-py) | 41 | TODO | TODO (manual) |  |
| [`monolith/native_training/hooks/hook_utils_test.py`](#monolith-native-training-hooks-hook-utils-test-py) | 35 | TODO | TODO (manual) |  |
| [`monolith/native_training/hooks/ps_check_hooks.py`](#monolith-native-training-hooks-ps-check-hooks-py) | 97 | TODO | TODO (manual) |  |
| [`monolith/native_training/hooks/ps_check_hooks_test.py`](#monolith-native-training-hooks-ps-check-hooks-test-py) | 112 | TODO | TODO (manual) |  |
| [`monolith/native_training/hooks/server/client_lib.py`](#monolith-native-training-hooks-server-client-lib-py) | 30 | TODO | TODO (manual) |  |
| [`monolith/native_training/hooks/server/constants.py`](#monolith-native-training-hooks-server-constants-py) | 15 | TODO | TODO (manual) |  |
| [`monolith/native_training/hooks/server/server_lib.py`](#monolith-native-training-hooks-server-server-lib-py) | 95 | TODO | TODO (manual) |  |
| [`monolith/native_training/hooks/server/server_lib_test.py`](#monolith-native-training-hooks-server-server-lib-test-py) | 54 | TODO | TODO (manual) |  |
| [`monolith/native_training/hooks/session_hooks.py`](#monolith-native-training-hooks-session-hooks-py) | 44 | TODO | TODO (manual) |  |
| [`monolith/native_training/hooks/session_hooks_test.py`](#monolith-native-training-hooks-session-hooks-test-py) | 33 | TODO | TODO (manual) |  |
| [`monolith/native_training/hvd_lib.py`](#monolith-native-training-hvd-lib-py) | 65 | TODO | TODO (manual) |  |
| [`monolith/native_training/input.py`](#monolith-native-training-input-py) | 45 | TODO | TODO (manual) |  |
| [`monolith/native_training/layers/__init__.py`](#monolith-native-training-layers-init-py) | 46 | TODO | TODO (manual) |  |
| [`monolith/native_training/layers/add_bias.py`](#monolith-native-training-layers-add-bias-py) | 110 | TODO | TODO (manual) |  |
| [`monolith/native_training/layers/add_bias_test.py`](#monolith-native-training-layers-add-bias-test-py) | 65 | TODO | TODO (manual) |  |
| [`monolith/native_training/layers/advanced_activations.py`](#monolith-native-training-layers-advanced-activations-py) | 217 | TODO | TODO (manual) |  |
| [`monolith/native_training/layers/advanced_activations_test.py`](#monolith-native-training-layers-advanced-activations-test-py) | 84 | TODO | TODO (manual) |  |
| [`monolith/native_training/layers/agru.py`](#monolith-native-training-layers-agru-py) | 295 | TODO | TODO (manual) |  |
| [`monolith/native_training/layers/agru_test.py`](#monolith-native-training-layers-agru-test-py) | 112 | TODO | TODO (manual) |  |
| [`monolith/native_training/layers/dense.py`](#monolith-native-training-layers-dense-py) | 307 | TODO | TODO (manual) |  |
| [`monolith/native_training/layers/dense_test.py`](#monolith-native-training-layers-dense-test-py) | 147 | TODO | TODO (manual) |  |
| [`monolith/native_training/layers/feature_cross.py`](#monolith-native-training-layers-feature-cross-py) | 805 | TODO | TODO (manual) |  |
| [`monolith/native_training/layers/feature_cross_test.py`](#monolith-native-training-layers-feature-cross-test-py) | 286 | TODO | TODO (manual) |  |
| [`monolith/native_training/layers/feature_seq.py`](#monolith-native-training-layers-feature-seq-py) | 361 | TODO | TODO (manual) |  |
| [`monolith/native_training/layers/feature_seq_test.py`](#monolith-native-training-layers-feature-seq-test-py) | 126 | TODO | TODO (manual) |  |
| [`monolith/native_training/layers/feature_trans.py`](#monolith-native-training-layers-feature-trans-py) | 340 | TODO | TODO (manual) |  |
| [`monolith/native_training/layers/feature_trans_test.py`](#monolith-native-training-layers-feature-trans-test-py) | 140 | TODO | TODO (manual) |  |
| [`monolith/native_training/layers/layer_ops.py`](#monolith-native-training-layers-layer-ops-py) | 131 | TODO | TODO (manual) |  |
| [`monolith/native_training/layers/layer_ops_test.py`](#monolith-native-training-layers-layer-ops-test-py) | 232 | TODO | TODO (manual) |  |
| [`monolith/native_training/layers/lhuc.py`](#monolith-native-training-layers-lhuc-py) | 296 | TODO | TODO (manual) |  |
| [`monolith/native_training/layers/lhuc_test.py`](#monolith-native-training-layers-lhuc-test-py) | 73 | TODO | TODO (manual) |  |
| [`monolith/native_training/layers/logit_correction.py`](#monolith-native-training-layers-logit-correction-py) | 88 | TODO | TODO (manual) |  |
| [`monolith/native_training/layers/logit_correction_test.py`](#monolith-native-training-layers-logit-correction-test-py) | 65 | TODO | TODO (manual) |  |
| [`monolith/native_training/layers/mlp.py`](#monolith-native-training-layers-mlp-py) | 211 | TODO | TODO (manual) |  |
| [`monolith/native_training/layers/mlp_test.py`](#monolith-native-training-layers-mlp-test-py) | 78 | TODO | TODO (manual) |  |
| [`monolith/native_training/layers/multi_task.py`](#monolith-native-training-layers-multi-task-py) | 448 | TODO | TODO (manual) |  |
| [`monolith/native_training/layers/multi_task_test.py`](#monolith-native-training-layers-multi-task-test-py) | 128 | TODO | TODO (manual) |  |
| [`monolith/native_training/layers/norms.py`](#monolith-native-training-layers-norms-py) | 343 | TODO | TODO (manual) |  |
| [`monolith/native_training/layers/norms_test.py`](#monolith-native-training-layers-norms-test-py) | 84 | TODO | TODO (manual) |  |
| [`monolith/native_training/layers/pooling.py`](#monolith-native-training-layers-pooling-py) | 101 | TODO | TODO (manual) |  |
| [`monolith/native_training/layers/pooling_test.py`](#monolith-native-training-layers-pooling-test-py) | 141 | TODO | TODO (manual) |  |
| [`monolith/native_training/layers/sparse_nas.py`](#monolith-native-training-layers-sparse-nas-py) | 31 | TODO | TODO (manual) |  |
| [`monolith/native_training/layers/sparse_nas_test.py`](#monolith-native-training-layers-sparse-nas-test-py) | 23 | TODO | TODO (manual) |  |
| [`monolith/native_training/layers/utils.py`](#monolith-native-training-layers-utils-py) | 159 | TODO | TODO (manual) |  |
| [`monolith/native_training/learning_rate_functions.py`](#monolith-native-training-learning-rate-functions-py) | 112 | TODO | TODO (manual) |  |
| [`monolith/native_training/learning_rate_functions_test.py`](#monolith-native-training-learning-rate-functions-test-py) | 76 | TODO | TODO (manual) |  |
| [`monolith/native_training/logging_ops.py`](#monolith-native-training-logging-ops-py) | 56 | TODO | TODO (manual) |  |
| [`monolith/native_training/logging_ops_test.py`](#monolith-native-training-logging-ops-test-py) | 57 | TODO | TODO (manual) |  |
| [`monolith/native_training/losses/batch_softmax_loss.py`](#monolith-native-training-losses-batch-softmax-loss-py) | 57 | TODO | TODO (manual) |  |
| [`monolith/native_training/losses/batch_softmax_loss_test.py`](#monolith-native-training-losses-batch-softmax-loss-test-py) | 35 | TODO | TODO (manual) |  |
| [`monolith/native_training/losses/inbatch_auc_loss.py`](#monolith-native-training-losses-inbatch-auc-loss-py) | 41 | TODO | TODO (manual) |  |
| [`monolith/native_training/losses/inbatch_auc_loss_test.py`](#monolith-native-training-losses-inbatch-auc-loss-test-py) | 71 | TODO | TODO (manual) |  |
| [`monolith/native_training/losses/ltr_losses.py`](#monolith-native-training-losses-ltr-losses-py) | 1233 | TODO | TODO (manual) |  |
| [`monolith/native_training/metric/cli.py`](#monolith-native-training-metric-cli-py) | 28 | TODO | TODO (manual) |  |
| [`monolith/native_training/metric/deep_insight_ops.py`](#monolith-native-training-metric-deep-insight-ops-py) | 134 | TODO | TODO (manual) |  |
| [`monolith/native_training/metric/deep_insight_ops_test.py`](#monolith-native-training-metric-deep-insight-ops-test-py) | 33 | TODO | TODO (manual) |  |
| [`monolith/native_training/metric/exit_hook.py`](#monolith-native-training-metric-exit-hook-py) | 48 | TODO | TODO (manual) |  |
| [`monolith/native_training/metric/kafka_utils.py`](#monolith-native-training-metric-kafka-utils-py) | 119 | TODO | TODO (manual) |  |
| [`monolith/native_training/metric/metric_hook.py`](#monolith-native-training-metric-metric-hook-py) | 563 | TODO | TODO (manual) |  |
| [`monolith/native_training/metric/metric_hook_test.py`](#monolith-native-training-metric-metric-hook-test-py) | 189 | TODO | TODO (manual) |  |
| [`monolith/native_training/metric/utils.py`](#monolith-native-training-metric-utils-py) | 104 | TODO | TODO (manual) |  |
| [`monolith/native_training/metric/utils_test.py`](#monolith-native-training-metric-utils-test-py) | 50 | TODO | TODO (manual) |  |
| [`monolith/native_training/mlp_utils.py`](#monolith-native-training-mlp-utils-py) | 444 | TODO | TODO (manual) |  |
| [`monolith/native_training/model.py`](#monolith-native-training-model-py) | 182 | TODO | TODO (manual) |  |
| [`monolith/native_training/model_comp_test.py`](#monolith-native-training-model-comp-test-py) | 183 | TODO | TODO (manual) |  |
| [`monolith/native_training/model_dump/dump_utils.py`](#monolith-native-training-model-dump-dump-utils-py) | 757 | TODO | TODO (manual) |  |
| [`monolith/native_training/model_dump/graph_utils.py`](#monolith-native-training-model-dump-graph-utils-py) | 845 | TODO | TODO (manual) |  |
| [`monolith/native_training/model_dump/graph_utils_test.py`](#monolith-native-training-model-dump-graph-utils-test-py) | 86 | TODO | TODO (manual) |  |
| [`monolith/native_training/model_export/__init__.py`](#monolith-native-training-model-export-init-py) | 22 | TODO | TODO (manual) |  |
| [`monolith/native_training/model_export/data_gen_utils.py`](#monolith-native-training-model-export-data-gen-utils-py) | 732 | TODO | TODO (manual) |  |
| [`monolith/native_training/model_export/data_gen_utils_test.py`](#monolith-native-training-model-export-data-gen-utils-test-py) | 0 | TODO | TODO (manual) |  |
| [`monolith/native_training/model_export/demo_export.py`](#monolith-native-training-model-export-demo-export-py) | 100 | TODO | TODO (manual) |  |
| [`monolith/native_training/model_export/demo_export_test.py`](#monolith-native-training-model-export-demo-export-test-py) | 48 | TODO | TODO (manual) |  |
| [`monolith/native_training/model_export/demo_predictor.py`](#monolith-native-training-model-export-demo-predictor-py) | 110 | TODO | TODO (manual) |  |
| [`monolith/native_training/model_export/demo_predictor_client.py`](#monolith-native-training-model-export-demo-predictor-client-py) | 93 | TODO | TODO (manual) |  |
| [`monolith/native_training/model_export/export_context.py`](#monolith-native-training-model-export-export-context-py) | 141 | TODO | TODO (manual) |  |
| [`monolith/native_training/model_export/export_hooks.py`](#monolith-native-training-model-export-export-hooks-py) | 137 | TODO | TODO (manual) |  |
| [`monolith/native_training/model_export/export_hooks_test.py`](#monolith-native-training-model-export-export-hooks-test-py) | 141 | TODO | TODO (manual) |  |
| [`monolith/native_training/model_export/export_state_utils.py`](#monolith-native-training-model-export-export-state-utils-py) | 46 | TODO | TODO (manual) |  |
| [`monolith/native_training/model_export/export_state_utils_test.py`](#monolith-native-training-model-export-export-state-utils-test-py) | 36 | TODO | TODO (manual) |  |
| [`monolith/native_training/model_export/export_utils.py`](#monolith-native-training-model-export-export-utils-py) | 98 | TODO | TODO (manual) |  |
| [`monolith/native_training/model_export/export_utils_test.py`](#monolith-native-training-model-export-export-utils-test-py) | 43 | TODO | TODO (manual) |  |
| [`monolith/native_training/model_export/saved_model_exporters.py`](#monolith-native-training-model-export-saved-model-exporters-py) | 739 | TODO | TODO (manual) |  |
| [`monolith/native_training/model_export/saved_model_exporters_test.py`](#monolith-native-training-model-export-saved-model-exporters-test-py) | 153 | TODO | TODO (manual) |  |
| [`monolith/native_training/model_export/saved_model_visulizer.py`](#monolith-native-training-model-export-saved-model-visulizer-py) | 89 | TODO | TODO (manual) |  |
| [`monolith/native_training/model_export/warmup_data_decoder.py`](#monolith-native-training-model-export-warmup-data-decoder-py) | 55 | TODO | TODO (manual) |  |
| [`monolith/native_training/model_export/warmup_data_gen.py`](#monolith-native-training-model-export-warmup-data-gen-py) | 253 | TODO | TODO (manual) |  |
| [`monolith/native_training/model_export/warmup_example_batch.py`](#monolith-native-training-model-export-warmup-example-batch-py) | 57 | TODO | TODO (manual) |  |
| [`monolith/native_training/monolith_export.py`](#monolith-native-training-monolith-export-py) | 18 | TODO | TODO (manual) |  |
| [`monolith/native_training/multi_hash_table_ops.py`](#monolith-native-training-multi-hash-table-ops-py) | 695 | TODO | TODO (manual) |  |
| [`monolith/native_training/multi_hash_table_ops_test.py`](#monolith-native-training-multi-hash-table-ops-test-py) | 249 | TODO | TODO (manual) |  |
| [`monolith/native_training/multi_type_hash_table.py`](#monolith-native-training-multi-type-hash-table-py) | 435 | TODO | TODO (manual) |  |
| [`monolith/native_training/multi_type_hash_table_test.py`](#monolith-native-training-multi-type-hash-table-test-py) | 326 | TODO | TODO (manual) |  |
| [`monolith/native_training/native_model.py`](#monolith-native-training-native-model-py) | 1109 | TODO | TODO (manual) |  |
| [`monolith/native_training/native_task.py`](#monolith-native-training-native-task-py) | 213 | TODO | TODO (manual) |  |
| [`monolith/native_training/native_task_context.py`](#monolith-native-training-native-task-context-py) | 58 | TODO | TODO (manual) |  |
| [`monolith/native_training/nested_tensors.py`](#monolith-native-training-nested-tensors-py) | 110 | TODO | TODO (manual) |  |
| [`monolith/native_training/nested_tensors_test.py`](#monolith-native-training-nested-tensors-test-py) | 57 | TODO | TODO (manual) |  |
| [`monolith/native_training/net_utils.py`](#monolith-native-training-net-utils-py) | 133 | TODO | TODO (manual) |  |
| [`monolith/native_training/net_utils_test.py`](#monolith-native-training-net-utils-test-py) | 94 | TODO | TODO (manual) |  |
| [`monolith/native_training/optimizers/adamom.py`](#monolith-native-training-optimizers-adamom-py) | 68 | TODO | TODO (manual) |  |
| [`monolith/native_training/optimizers/adamom_test.py`](#monolith-native-training-optimizers-adamom-test-py) | 57 | TODO | TODO (manual) |  |
| [`monolith/native_training/optimizers/rmsprop.py`](#monolith-native-training-optimizers-rmsprop-py) | 102 | TODO | TODO (manual) |  |
| [`monolith/native_training/optimizers/rmsprop_test.py`](#monolith-native-training-optimizers-rmsprop-test-py) | 77 | TODO | TODO (manual) |  |
| [`monolith/native_training/optimizers/rmspropv2_test.py`](#monolith-native-training-optimizers-rmspropv2-test-py) | 112 | TODO | TODO (manual) |  |
| [`monolith/native_training/optimizers/shampoo.py`](#monolith-native-training-optimizers-shampoo-py) | 207 | TODO | TODO (manual) |  |
| [`monolith/native_training/prefetch_queue.py`](#monolith-native-training-prefetch-queue-py) | 379 | TODO | TODO (manual) |  |
| [`monolith/native_training/prefetch_queue_test.py`](#monolith-native-training-prefetch-queue-test-py) | 305 | TODO | TODO (manual) |  |
| [`monolith/native_training/ps_benchmark.py`](#monolith-native-training-ps-benchmark-py) | 273 | TODO | TODO (manual) |  |
| [`monolith/native_training/ps_benchmark_test.py`](#monolith-native-training-ps-benchmark-test-py) | 57 | TODO | TODO (manual) |  |
| [`monolith/native_training/ragged_utils.py`](#monolith-native-training-ragged-utils-py) | 29 | TODO | TODO (manual) |  |
| [`monolith/native_training/ragged_utils_test.py`](#monolith-native-training-ragged-utils-test-py) | 32 | TODO | TODO (manual) |  |
| [`monolith/native_training/remote_predict_ops.py`](#monolith-native-training-remote-predict-ops-py) | 0 | TODO | TODO (manual) |  |
| [`monolith/native_training/restore_test.py`](#monolith-native-training-restore-test-py) | 240 | TODO | TODO (manual) |  |
| [`monolith/native_training/runner_utils.py`](#monolith-native-training-runner-utils-py) | 396 | TODO | TODO (manual) |  |
| [`monolith/native_training/runner_utils_test.py`](#monolith-native-training-runner-utils-test-py) | 108 | TODO | TODO (manual) |  |
| [`monolith/native_training/runtime/ops/gen_monolith_ops.py`](#monolith-native-training-runtime-ops-gen-monolith-ops-py) | 23 | TODO | TODO (manual) |  |
| [`monolith/native_training/save_utils.py`](#monolith-native-training-save-utils-py) | 1309 | TODO | TODO (manual) |  |
| [`monolith/native_training/save_utils_test.py`](#monolith-native-training-save-utils-test-py) | 1740 | TODO | TODO (manual) |  |
| [`monolith/native_training/service_discovery.py`](#monolith-native-training-service-discovery-py) | 481 | TODO | TODO (manual) |  |
| [`monolith/native_training/service_discovery_test.py`](#monolith-native-training-service-discovery-test-py) | 407 | TODO | TODO (manual) |  |
| [`monolith/native_training/serving_ps_test.py`](#monolith-native-training-serving-ps-test-py) | 231 | TODO | TODO (manual) |  |
| [`monolith/native_training/session_run_hooks.py`](#monolith-native-training-session-run-hooks-py) | 171 | TODO | TODO (manual) |  |
| [`monolith/native_training/session_run_hooks_test.py`](#monolith-native-training-session-run-hooks-test-py) | 144 | TODO | TODO (manual) |  |
| [`monolith/native_training/signal_utils.py`](#monolith-native-training-signal-utils-py) | 37 | TODO | TODO (manual) |  |
| [`monolith/native_training/signal_utils_test.py`](#monolith-native-training-signal-utils-test-py) | 30 | TODO | TODO (manual) |  |
| [`monolith/native_training/static_reshape_op.py`](#monolith-native-training-static-reshape-op-py) | 58 | TODO | TODO (manual) |  |
| [`monolith/native_training/static_reshape_op_test.py`](#monolith-native-training-static-reshape-op-test-py) | 79 | TODO | TODO (manual) |  |
| [`monolith/native_training/summary/summary_ops.py`](#monolith-native-training-summary-summary-ops-py) | 78 | TODO | TODO (manual) |  |
| [`monolith/native_training/summary/summary_ops_test.py`](#monolith-native-training-summary-summary-ops-test-py) | 122 | TODO | TODO (manual) |  |
| [`monolith/native_training/summary/utils.py`](#monolith-native-training-summary-utils-py) | 114 | TODO | TODO (manual) |  |
| [`monolith/native_training/summary/utils_test.py`](#monolith-native-training-summary-utils-test-py) | 43 | TODO | TODO (manual) |  |
| [`monolith/native_training/sync_hooks.py`](#monolith-native-training-sync-hooks-py) | 176 | TODO | TODO (manual) |  |
| [`monolith/native_training/sync_hooks_test.py`](#monolith-native-training-sync-hooks-test-py) | 119 | TODO | TODO (manual) |  |
| [`monolith/native_training/sync_training_hooks.py`](#monolith-native-training-sync-training-hooks-py) | 355 | TODO | TODO (manual) |  |
| [`monolith/native_training/sync_training_hooks_test.py`](#monolith-native-training-sync-training-hooks-test-py) | 92 | TODO | TODO (manual) |  |
| [`monolith/native_training/tensor_utils.py`](#monolith-native-training-tensor-utils-py) | 162 | TODO | TODO (manual) |  |
| [`monolith/native_training/tensor_utils_test.py`](#monolith-native-training-tensor-utils-test-py) | 175 | TODO | TODO (manual) |  |
| [`monolith/native_training/test_utils.py`](#monolith-native-training-test-utils-py) | 65 | TODO | TODO (manual) |  |
| [`monolith/native_training/touched_key_set_ops.py`](#monolith-native-training-touched-key-set-ops-py) | 61 | TODO | TODO (manual) |  |
| [`monolith/native_training/touched_key_set_ops_test.py`](#monolith-native-training-touched-key-set-ops-test-py) | 51 | TODO | TODO (manual) |  |
| [`monolith/native_training/utils.py`](#monolith-native-training-utils-py) | 320 | TODO | TODO (manual) |  |
| [`monolith/native_training/utils_test.py`](#monolith-native-training-utils-test-py) | 70 | TODO | TODO (manual) |  |
| [`monolith/native_training/variables.py`](#monolith-native-training-variables-py) | 147 | TODO | TODO (manual) |  |
| [`monolith/native_training/variables_test.py`](#monolith-native-training-variables-test-py) | 89 | TODO | TODO (manual) |  |
| [`monolith/native_training/yarn_runtime.py`](#monolith-native-training-yarn-runtime-py) | 127 | TODO | TODO (manual) |  |
| [`monolith/native_training/yarn_runtime_test.py`](#monolith-native-training-yarn-runtime-test-py) | 133 | TODO | TODO (manual) |  |
| [`monolith/native_training/zk_utils.py`](#monolith-native-training-zk-utils-py) | 96 | TODO | TODO (manual) |  |
| [`monolith/path_utils.py`](#monolith-path-utils-py) | 47 | TODO | TODO (manual) |  |
| [`monolith/tpu_runner.py`](#monolith-tpu-runner-py) | 429 | TODO | TODO (manual) |  |
| [`monolith/utils.py`](#monolith-utils-py) | 81 | TODO | TODO (manual) |  |
| [`monolith/utils_test.py`](#monolith-utils-test-py) | 65 | TODO | TODO (manual) |  |

## Per-File Parity Checklists (All Python Files)

Every file listed below must be fully mapped to Rust with parity behavior verified.

### `monolith/__init__.py`
<a id="monolith-init-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 55
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

### `monolith/agent_service/__init__.py`
<a id="monolith-agent-service-init-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 0
- Purpose/role: Empty package initializer for `monolith.agent_service`.
- Key symbols/classes/functions: none
- External dependencies: none
- Side effects: none

**Required Behavior (Detailed)**
- Module is intentionally empty; importing it must have no side effects.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/src/lib.rs` (module boundary only).
- Rust public API surface: none required.
- Data model mapping: none.

**Implementation Steps (Detailed)**
1. Ensure the Rust crate module layout mirrors the Python package (no runtime behavior needed).

**Tests (Detailed)**
- Python tests: none
- Rust tests: none (module boundary only)

**Gaps / Notes**
- None; this file is a no-op.

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

### `monolith/agent_service/agent.py`
<a id="monolith-agent-service-agent-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 96
- Purpose/role: Main entrypoint for running agent processes; selects AgentV1/AgentV3 and manages dense multi-process mode.
- Key symbols/classes/functions: `run_agent`, `main`
- External dependencies: `absl`, `multiprocessing`, `subprocess`, `signal`, `ModelManager`, `AgentV1`, `AgentV3`
- Side effects: env var mutations, process spawning, logging, OS signals.

**Required Behavior (Detailed)**
- `run_agent(conf_path, tfs_log, use_mps, replica_id, dense_service_index)`:
  - Mutates `REPLICA_ID` and `DENSE_SERVICE_IDX` when `use_mps` is true.
  - Loads `AgentConfig` from file and instantiates AgentV1 or AgentV3 based on `agent_version`.
  - Starts `ModelManager` for rough sort model; terminates self on failure.
- `main()`:
  - Initializes HDFS env via `env_utils.setup_hdfs_env()`.
  - Loads AgentConfig, handles `DeployType.DENSE` + `dense_service_num > 1` by spawning multiple processes.
  - Otherwise runs single agent.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-cli` or `monolith-rs/crates/monolith-serving/src/bin/*` (new binary).
- Rust public API surface: CLI command mirroring flags `--conf` and `--tfs_log`.
- Integration points: use Rust `AgentConfig` loader, start AgentV1/V3 ported classes.

**Implementation Steps (Detailed)**
1. Add Rust CLI command equivalent to Python `agent.py` entrypoint.
2. Implement env var behavior for `REPLICA_ID` and `DENSE_SERVICE_IDX`.
3. Add multiprocessing behavior for dense service count.
4. Port `ModelManager` startup and failure semantics.
5. Wire into AgentV1/AgentV3 implementations.

**Tests (Detailed)**
- Python tests: none specific.
- Rust tests: add integration test to verify process launch plan and env var behavior.

**Gaps / Notes**
- Rust currently lacks equivalent entrypoint for agent daemonization.

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

### `monolith/agent_service/agent_base.py`
<a id="monolith-agent-service-agent-base-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 86
- Purpose/role: Agent base class + helpers for launching TFS/proxy binaries and log redirection.
- Key symbols/classes/functions: `get_cmd_path`, `get_cmd_and_port`, `ServingLog`, `AgentBase`
- External dependencies: `os`, `abc`, `AgentConfig`, `TFS_HOME`, `TFSServerType`
- Side effects: resolves binary paths, builds shell commands, changes CWD in context manager.

**Required Behavior (Detailed)**
- `get_cmd_path()` returns the absolute path to this Python file.
- `get_cmd_and_port(...)`:
  - For PS/ENTRY/DENSE: delegates to `AgentConfig.get_cmd_and_port` with `tfs_binary`.
  - Else (proxy): builds proxy command string, uses `proxy.conf` if present.
- `ServingLog` context manager:
  - Builds log filename prefixed with `log_prefix` and switches CWD to `$TFS_HOME/bin`.
- `AgentBase` abstract base with `start()` and `wait_for_termination()`.

**Rust Mapping (Detailed)**
- Target crate/module: new module in `monolith-rs/crates/monolith-serving/src/agent_base.rs` (or similar).
- Rust public API surface: `AgentBase` trait, `get_cmd_and_port`, `ServingLog` equivalent.
- Data model mapping: `AgentConfig` in Rust config module.

**Implementation Steps (Detailed)**
1. Port command construction rules exactly (including proxy.conf behavior).
2. Add Rust CWD + log file handling in a scoped guard.
3. Define a trait for Agent lifecycle parity (`start`, `wait_for_termination`).
4. Ensure paths match Python defaults (`TFS_HOME`, binary names).

**Tests (Detailed)**
- Python tests: none specific.
- Rust tests: add unit tests for command generation + log filename behavior.

**Gaps / Notes**
- Rust currently lacks explicit AgentBase and ServingLog helpers.

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

### `monolith/agent_service/agent_client.py`
<a id="monolith-agent-service-agent-client-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 216
- Purpose/role: CLI for AgentService + ZooKeeper operations (heartbeat, replica lookup, publish/load/unload, resource and portal inspection).
- Key symbols/classes/functions: `main`
- External dependencies: `grpc`, `kazoo`, `monolith.agent_service.*`, `monolith.native_training.env_utils`
- Side effects: gRPC calls, ZooKeeper reads/writes/deletes, prints to stdout, reads env vars.

**Required Behavior (Detailed)**
- Startup:
  - `env_utils.setup_hdfs_env()` is called.
  - `AgentConfig.from_file(FLAGS.conf)` is loaded; if `FLAGS.port != 0` overrides `agent_port`.
  - Host is `MY_HOST_IP` env var or `socket.gethostbyname(gethostname())`.
  - gRPC channel to `{host}:{agent_port}`; `AgentServiceStub` created.
  - `model_name` resolved as `agent_conf.base_name` or `FLAGS.model_name`.
- `FLAGS.server_type` mapping: `ps` -> `ServerType.PS`, `dense` -> `ServerType.DENSE`, else `ServerType.ENTRY`.
- `FLAGS.cmd_type` dispatch:
  - `hb`: send `HeartBeatRequest(server_type=...)`; print each key with number of addrs and list.
  - `gr`: assert `model_name`; send `GetReplicasRequest(server_type, task, model_name)`; print reply address list.
  - `addr` or (`get` + `args=addr`):
    - Connect `MonolithKazooClient` with `agent_conf.zk_servers`.
    - Traverse `/{bzid}/service/{model_name}`; support dc-aware layout (`idc:cluster/server_type:task`) and non-dc (`server_type:task`).
    - For each replica node, read `ReplicaMeta`, print path + `archon_address`, `address`, and `ModelState.Name`.
    - Sort output; handle `NoNodeError` by printing "{model_name} has not load !" and returning.
  - `get` + `args=info`: print `cal_model_info_v2(model_dir, ckpt)`.
  - `get` + `args in {res,pub,portal,lock,elect}`:
    - Select ZK path prefix: `/bzid/resource|publish|portal|lock|election`.
    - If path missing, print `no {args} found !`.
    - Otherwise list children, read data for each, print sorted keys with values.
  - `load`: assert `model_name`; create `ModelMeta(model_name, model_dir, ckpt, num_shard)`; write to `/bzid/portal/{model_name}` (create or set).
  - `unload`: delete `/bzid/portal/{model_name}`; ignore errors.
  - `clean`: delete all nodes under portal/publish/service/resource based on `FLAGS.args`.
- All ZK clients are started and stopped per operation.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-cli/src/bin/agent_client.rs` (or CLI subcommand).
- Rust public API surface: CLI entrypoint + small helper functions for each `cmd_type`.
- Data model mapping: use Rust protobuf types for `AgentService` and `ReplicaMeta`/`ModelMeta` equivalents.
- Feature gating: `tf-runtime` or `zk` features for ZK access; gRPC feature for AgentService.
- Integration points: reuse Rust ZK client + gRPC client modules.

**Implementation Steps (Detailed)**
1. Recreate CLI flags: `port`, `args`, `server_type`, `task`, `model_dir`, `ckpt`, `num_shard`, plus shared `FLAGS` from `client.py`.
2. Implement gRPC client calls (`HeartBeat`, `GetReplicas`) with identical formatting to stdout.
3. Implement ZK traversal for `addr` respecting dc-aware and non-dc-aware layouts.
4. Implement portal/publish/resource/lock/election reads with exact error messages.
5. Implement `load/unload/clean` ZK mutations with same node paths.
6. Mirror env var usage for host selection and config overrides.

**Tests (Detailed)**
- Python tests: none specific
- Rust tests: add CLI integration tests with fake ZK + fake gRPC stub
- Cross-language parity test: run Python CLI and Rust CLI against same fake ZK/gRPC environment and compare output.

**Gaps / Notes**
- This CLI hard-depends on ZK and gRPC; Rust implementation needs compatible test doubles.

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

### `monolith/agent_service/agent_controller.py`
<a id="monolith-agent-service-agent-controller-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 145
- Purpose/role: CLI to declare SavedModel configs in ZK, and publish/unpublish layouts.
- Key symbols/classes/functions: `find_model_name`, `declare_saved_model`, `map_model_to_layout`, `bzid_info`, `main`
- External dependencies: `tensorflow`, `saved_model_pb2`, `compat`, `monolith.agent_service.backends.*`, ZK backend.
- Side effects: reads `saved_model.pb`, writes/updates ZK nodes, prints JSON.

**Required Behavior (Detailed)**
- `find_model_name(exported_models_path)`:
  - Uses `entry/` subdirectory and picks `sorted(tf.io.gfile.listdir(entry_path))[0]` as timestamp.
  - Reads `saved_model.pb`, parses `SavedModel`, scans graph nodes with `op == 'TfServingRemotePredict'`.
  - Returns first `model_name` attribute (decoded, before `:`), or `None` if not present.
- `declare_saved_model(bd, export_base, model_name=None, overwrite=False, arch='entry_ps')`:
  - Asserts `arch == 'entry_ps'`.
  - Determines `model_name` via `find_model_name` if not supplied.
  - Logs mismatch if supplied name differs from export name; asserts non-None.
  - Asserts no existing saved_models unless `overwrite`.
  - For each subgraph in `export_base`:
    - Build `SavedModelDeployConfig(model_base_path=..., version_policy='latest' for entry, else 'latest_once')`.
    - Call `bd.decl_saved_model(SavedModel(model_name, sub_graph), deploy_config)`.
  - Logs success; returns `model_name`.
- `map_model_to_layout(bd, model_pattern, layout_path, action)`:
  - Parses `model_pattern` as `model_name:sub_graph_pattern`.
  - `fnmatch.filter` on available subgraphs.
  - `pub` => `bd.add_to_layout`, `unpub` => `bd.remove_from_layout`.
- `bzid_info(bd)`: `print(json.dumps(bd.bzid_info(), indent=2))`.
- `main`:
  - Validates `FLAGS.cmd` in `decl|pub|unpub|bzid_info`.
  - Creates `ZKBackend`, `start()` then executes command; `stop()` in `finally`.
  - For `pub/unpub`, layout path is `/{bzid}/layouts/{layout}`.
  - Uses `env_utils.setup_hdfs_env()` in `__main__` guard; sets logging.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-cli/src/bin/agent_controller.rs`.
- Rust public API surface: CLI entrypoint + helpers for `declare_saved_model` and layout mapping.
- Data model mapping: `SavedModel`, `SavedModelDeployConfig`, `ZKBackend` analogs in Rust.
- Feature gating: `tf-runtime`/`saved-model` parsing feature for `find_model_name` (optional if using tf runtime).
- Integration points: ZK backend + layout manager.

**Implementation Steps (Detailed)**
1. Port CLI flags (`cmd`, `bzid`, `zk_servers`, `export_base`, `model_name`, `layout`, `arch`, `overwrite`).
2. Implement saved_model.pb parsing in Rust (protobuf decode + graph scan for `TfServingRemotePredict`).
3. Recreate `declare_saved_model` and `map_model_to_layout` logic exactly.
4. Ensure `latest` vs `latest_once` policy mapping is preserved.
5. Port `bzid_info` output formatting to JSON.

**Tests (Detailed)**
- Python tests: `monolith/agent_service/agent_controller_test.py`
- Rust tests: add equivalent tests using fake ZK and test saved_model fixture.
- Cross-language parity test: compare `bzid_info`/layout changes after publish/unpublish.

**Gaps / Notes**
- Rust needs SavedModel graph parsing for remote predict op discovery.

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

### `monolith/agent_service/agent_controller_test.py`
<a id="monolith-agent-service-agent-controller-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 95
- Purpose/role: Tests ZKBackend + agent_controller declare/publish/unpublish flows using FakeKazooClient.
- Key symbols/classes/functions: `AgentControllerTest.test_decl_saved_models`, `.test_pub`
- External dependencies: `FakeKazooClient`, `ZKBackend`, test saved_model fixtures.
- Side effects: creates ZK nodes in fake client.

**Required Behavior (Detailed)**
- `setUpClass`:
  - Create `ZKBackend` with fake ZK; `start()`.
- `test_decl_saved_models`:
  - Call `declare_saved_model` with test saved_model directory.
  - Verify `bd.list_saved_models('test_ffm_model')` matches `{ps_0..ps_4, entry}`.
- `test_pub`:
  - Declare saved model again.
  - `map_model_to_layout(..., "entry", action="pub")` adds entry only.
  - `map_model_to_layout(..., "ps_*", action="pub")` adds all ps subgraphs.
  - `map_model_to_layout(..., "ps_*", action="unpub")` removes ps subgraphs.
  - `map_model_to_layout(..., "entry", action="unpub")` results in empty layout list.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-cli/tests/agent_controller.rs` (or serving tests if backend lives there).
- Rust public API surface: ZK backend + controller helpers.
- Data model mapping: `SavedModel` and `SavedModelDeployConfig` equality comparisons.

**Implementation Steps (Detailed)**
1. Port fake ZK backend for tests.
2. Port `declare_saved_model` and `map_model_to_layout` in Rust.
3. Add fixtures for saved_model testdata or stub saved_model parsing.

**Tests (Detailed)**
- Python tests: this file
- Rust tests: exact parity assertions for saved_models and layout contents.

**Gaps / Notes**
- Requires test saved_model fixture to exist and be accessible in Rust tests.

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

### `monolith/agent_service/agent_service.py`
<a id="monolith-agent-service-agent-service-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 155
- Purpose/role: gRPC AgentService implementation + server wrapper for replica discovery and heartbeat.
- Key symbols/classes/functions: `AgentDataProvider`, `AgentServiceImpl`, `AgentService`
- External dependencies: `grpc`, `concurrent.futures`, `functools.singledispatchmethod`, `monolith.agent_service.*`
- Side effects: gRPC server binds to ports; reads env/config; logs heartbeat.

**Required Behavior (Detailed)**
- `AgentServiceImpl.__init__` is **overloaded** via `@singledispatchmethod`:
  - `ReplicaWatcher` + optional `AgentConfig`
  - `ZKMirror` + `AgentConfig`
  - `AgentDataProvider` + `AgentConfig`
- `GetReplicas` behavior:
  - If `conf is None` or `agent_version == 1`, fetch replicas via `ReplicaWatcher` with `idc/cluster`.
  - If `agent_version == 2`, fetch via ZKMirror (`get_task_replicas`).
  - Else: `NotImplementedError` for v3.
- `HeartBeat` behavior:
  - `agent_version == 1`: call `ReplicaWatcher.get_all_replicas`, optionally strip DC prefix when `dc_aware`.
  - `agent_version == 2`: call `ZKMirror.get_all_replicas`.
  - `agent_version == 3`: use `AgentDataProvider` callback map to fill addresses.
- `GetResource` behavior:
  - `agent_version == 1`: return empty `GetResourceResponse`.
  - Else: fill address with `get_local_ip()` + `agent_port`, plus memory via `cal_available_memory_v2()`.
- `AgentService` wrapper:
  - Constructs grpc server with `ThreadPoolExecutor`, registers `AgentServiceImpl`, binds to port.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/src/grpc_agent.rs` (gRPC), `monolith-rs/crates/monolith-serving/src/server.rs` (server lifecycle), plus new module for AgentConfig integration.
- Rust public API surface: `AgentServiceRealImpl` + `AgentGrpcServer` must be extended to mirror multi-backend behaviors and `AgentServiceImpl` logic.
- Data model mapping: use `monolith_proto::monolith::serving::agent_service::*` for request/response.
- Feature gating: always available with `grpc` feature.
- Integration points: `monolith-serving::server::Server` should optionally host AgentService parity endpoints.

**Implementation Steps (Detailed)**
1. Add AgentConfig wiring to Rust AgentService (support v1/v2/v3 equivalent selection).
2. Port `ReplicaWatcher` + `ZKMirror` behaviors or stub with clear TODOs and guards.
3. Implement `dc_aware` key rewriting in HeartBeat response.
4. Implement `GetResource` response with local IP and memory (port parity).
5. Add AgentDataProvider path for v3-style maps.
6. Mirror Python error semantics (NotImplemented) when agent_version == 3 in GetReplicas.
7. Add tests mirroring `agent_service_test.py`.

**Tests (Detailed)**
- Python tests: `monolith/agent_service/agent_service_test.py`
- Rust tests: add to `monolith-rs/crates/monolith-serving/tests/*` covering GetReplicas/HeartBeat/GetResource.
- Cross-language parity test: run Python client against Rust server for v1/v2 paths.

**Gaps / Notes**
- Rust currently provides minimal AgentService; needs full v1/v2/v3 parity.

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

### `monolith/agent_service/agent_service_test.py`
<a id="monolith-agent-service-agent-service-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 107
- Purpose/role: Tests gRPC AgentService heartbeats and replica lookup using FakeKazooClient + ReplicaWatcher.
- Key symbols/classes/functions: `AgentServiceTest.setUpClass`, `test_heart_beat`, `test_get_replicas`
- External dependencies: Fake ZK, `ReplicaWatcher`, `AgentService`, gRPC client stubs.
- Side effects: starts gRPC server, creates ephemeral ZK nodes.

**Required Behavior (Detailed)**
- `setUpClass`:
  - Set IDC/cluster env vars.
  - Create `FakeKazooClient` and `ReplicaWatcher` with `AgentConfig` (dc-aware, ps deploy).
  - Register ZK nodes for PS replicas and entry replicas with `ReplicaMeta(stat=AVAILABLE)`.
  - Start watcher, start `AgentService`, and create `SvrClient`.
- `test_heart_beat`: `heart_beat(ServerType.PS)` should return addresses map with `num_ps` entries.
- `test_get_replicas`: `get_replicas(ServerType.PS, task=NUM_PS_REPLICAS-1)` returns `NUM_PS_REPLICAS` addresses.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/tests/agent_service.rs`.
- Rust public API surface: gRPC AgentService + ReplicaWatcher equivalent.
- Data model mapping: `ReplicaMeta`, `ModelState`, and gRPC proto types.

**Implementation Steps (Detailed)**
1. Implement fake ZK and replica registration helpers in Rust tests.
2. Start Rust AgentService server on a free port.
3. Call HeartBeat and GetReplicas via gRPC client; assert counts.

**Tests (Detailed)**
- Python tests: this file
- Rust tests: parity test with fake ZK + gRPC client.

**Gaps / Notes**
- Rust needs fake ZK client and ReplicaWatcher equivalent to reproduce dc-aware paths.

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

### `monolith/agent_service/agent_v1.py`
<a id="monolith-agent-service-agent-v1-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 390
- Purpose/role: v1 agent orchestrator that launches TensorFlow Serving processes (ps/entry/dense) and hosts AgentService.
- Key symbols/classes/functions: `ProcessType`, `ProcessNode`, `ProcessMgr`, `AgentV1`
- External dependencies: `subprocess`, `signal`, `threading`, `ReplicaManager`, `AgentService`, `AgentBase`
- Side effects: starts and kills subprocesses; registers signal handlers; logs to files.

**Required Behavior (Detailed)**
- `ProcessNode.__init__`:
  - Chooses command and port via `get_cmd_and_port` based on process type.
  - For ENTRY sets env `PORT2` to agent port.
  - Tracks `_sub_procs` map and `_popen` handle.
- `ProcessNode.run()`:
  - If ENTRY: waits for PS replicas (and dense if `dense_alone`) via `ReplicaManager` before starting; timeout 3600s.
  - Launches subprocess with `ServingLog` output redirection (unless `MLP_POD_NAME` env set).
  - Waits for port open; starts sub-procs recursively; returns success boolean.
- `ProcessNode.wait_for_started()`: polls `check_port_open` every 10s up to 3600s; returns True on success.
- `ProcessNode.kill()`: kills sub-procs then self with retries; uses `poll`/`returncode` guards.
- `ProcessMgr`:
  - Registers SIGTERM/SIGINT handler to kill all processes.
  - Background `_poll` thread watches all processes; if any exits, kills all.
  - `start()` runs `ProcessNode.run()` for each and starts poll thread.
- `AgentV1`:
  - Starts ZK client, ReplicaManager (watcher/updater), AgentService, and ProcessMgr in that order.
  - Build process graph based on `DeployType` (MIXED/ENTRY/PS/DENSE) and `dense_alone`.
  - `stop()` shuts down processes, AgentService, ReplicaManager, ZK.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/src/agent_v1.rs` + process supervisor module.
- Rust public API surface: `AgentV1` struct implementing `AgentBase` trait.
- Data model mapping: `AgentConfig`, `DeployType`, `TFSServerType` in Rust config module.
- Integration points: gRPC AgentService + ReplicaManager + process management.

**Implementation Steps (Detailed)**
1. Implement process supervisor with child processes and log redirection.
2. Port ENTRY wait-for-PS/dense logic with same timeouts and intervals.
3. Port signal handling to trigger shutdown of all children.
4. Port startup order and error handling semantics.
5. Ensure `PORT2` env var is injected for ENTRY.

**Tests (Detailed)**
- Python tests: none directly
- Rust tests: integration test that spawns dummy processes and validates kill-on-exit behavior.
- Cross-language parity test: simulate ReplicaManager readiness and verify ENTRY wait logic.

**Gaps / Notes**
- Requires replacement for `check_port_open` and process supervision in Rust.

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

### `monolith/agent_service/agent_v3.py`
<a id="monolith-agent-service-agent-v3-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 210
- Purpose/role: v3 unified agent that manages TFS config via layouts, registers service info, and serves AgentService addrs.
- Key symbols/classes/functions: `AgentV3`, `gen_empty_model_config_file`
- External dependencies: `TFSWrapper`, `ZKBackend`, `Container`, protobuf text/json, `tensorflow_serving.config`.
- Side effects: writes model_config file, starts TFS process, registers ZK service info, background threads.

**Required Behavior (Detailed)**
- Enforces `DeployType.UNIFIED` and `agent_version == 3`.
- Generates empty model config file with `model_config_list {}`.
- Creates `TFSWrapper` with archon/grpc/http ports and config file; used to query model status.
- Layout filters:
  - `config.layout_filters` strings may include `${shard_id}` and `${shard_num}` substitutions.
  - Each filter is `match;cond`, where `match` becomes regex via `normalize_regex`.
  - `cond` is evaluated with regex groupdict values cast to int.
- Builds `ContainerServiceInfo` using local IP and ports; includes debug JSON with layout path + filters.
- `_gen_addrs_map()`:
  - Reads `backend.get_service_map()` and returns `{model:sub_graph: [grpc/archon addrs]}`.
  - Uses gRPC addr if `TFSWrapper.is_grpc_remote_op` else archon addr.
- `layout_update_callback(saved_models)`:
  - Applies layout filters to SavedModel list.
  - Writes new `ModelServerConfig` protobuf text to `_model_config_path`.
- `sync_available_saved_models()`:
  - Reads `TFSWrapper.list_saved_models_status()` and syncs AVAILABLE models to backend.
- `start()`:
  - Starts TFSWrapper, ZKBackend, AgentService.
  - Background thread every 60s: `report_service_info`.
  - Background thread every 30s: `sync_available_saved_models`.
  - Registers layout callback on `config.layout_path`.
- `wait_for_termination()` polls TFS process; if exit, stops and kills self.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/src/agent_v3.rs`.
- Rust public API surface: `AgentV3` implementing `AgentBase` with background tasks.
- Data model mapping: `Container`, `SavedModel`, `SavedModelDeployConfig`, `ContainerServiceInfo`.
- Feature gating: `tf-runtime` and `zk` features (TFS process and ZK backend).

**Implementation Steps (Detailed)**
1. Port layout filter parsing + evaluation semantics (regex + eval).
2. Implement model_config file generation for TensorFlow Serving.
3. Port service info reporting and available saved_model syncing to ZK.
4. Implement addrs map provider for AgentService v3.
5. Ensure shutdown semantics (set exit event, join threads, kill process).

**Tests (Detailed)**
- Python tests: `monolith/agent_service/agent_v3_test.py`
- Rust tests: use FakeTFSWrapper + FakeZK to verify model publishing and service map updates.
- Cross-language parity test: run layout update and verify config file contents.

**Gaps / Notes**
- Requires Rust implementation of ZKBackend and TFSWrapper equivalents.

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

### `monolith/agent_service/agent_v3_test.py`
<a id="monolith-agent-service-agent-v3-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 114
- Purpose/role: Tests AgentV3 layout/publish flow with FakeTFSWrapper and FakeKazooClient.
- Key symbols/classes/functions: `AgentV3Test.test_service_info`, `.test_publish_models`
- External dependencies: Fake TFS wrapper, fake ZK backend.
- Side effects: starts AgentV3 (with fake components), writes model_config file.

**Required Behavior (Detailed)**
- Setup:
  - Create `AgentV3` with `deploy_type=unified`, `agent_version=3`.
  - Replace `_tfs_wrapper` with `FakeTFSWrapper` and `_backend._zk` with `FakeKazooClient`.
  - Start agent; populate `/gip/saved_models/test_ffm_model/*` nodes with deploy configs.
- `test_service_info`: backend `get_service_info(container)` equals agent's `_service_info`.
- `test_publish_models`:
  - Add layout nodes (`/gip/layout/test_ffm_model:entry`, `/ps_0`).
  - Verify FakeTFSWrapper sees both models.
  - Call `sync_available_saved_models()` and verify service_map binding.
  - Delete one layout node, verify updated list and service_map.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/tests/agent_v3.rs`.
- Rust public API surface: AgentV3, FakeTFSWrapper, Fake ZK backend.

**Implementation Steps (Detailed)**
1. Implement FakeTFSWrapper in Rust with model_config file parsing.
2. Implement Fake ZK backend with nodes under saved_models/layouts.
3. Port tests for service_info equality and publish/unpublish behavior.

**Tests (Detailed)**
- Python tests: this file
- Rust tests: direct parity test using fake components.

**Gaps / Notes**
- Requires deterministic model_config parsing in Rust to match FakeTFSWrapper behavior.

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

### `monolith/agent_service/backends.py`
<a id="monolith-agent-service-backends-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 402
- Purpose/role: Backend abstractions + ZK implementation for layouts, saved models, and service info.
- Key symbols/classes/functions: `SavedModel`, `SavedModelDeployConfig`, `Container`, `ContainerServiceInfo`, `AgentBackend`, `CtrlBackend`, `SyncBackend`, `ZKBackend`
- External dependencies: `kazoo`, `dataclasses_json`, `monolith.native_training.zk_utils`
- Side effects: ZK reads/writes, watches, retry loops, thread sync.

**Required Behavior (Detailed)**
- Dataclasses serialize/deserialize to JSON bytes.
- ZKBackend implements:
  - Layout watchers (ChildrenWatch) and callbacks
  - Saved model registration & layout updates
  - Service info reporting + retrieval
  - Sync target subscription
- Correct handling of ZK errors (NodeExists, NoNode, ConnectionClosed, etc.).

**Rust Mapping (Detailed)**
- Target crate/module: new `monolith-rs/crates/monolith-serving/src/agent_backend.rs` + ZK integration (feature gated).
- Rust public API surface: trait equivalents for AgentBackend/CtrlBackend/SyncBackend and ZK implementation.
- Integration points: `ReplicaManager`, `ModelManager`, agent controllers.

**Implementation Steps (Detailed)**
1. Define Rust structs for SavedModel, SavedModelDeployConfig, Container, ContainerServiceInfo with serde JSON.
2. Define backend traits mirroring Python abstract base classes.
3. Implement ZK backend with watchers; match callback semantics and payload formats.
4. Port error handling and retry behaviors.

**Tests (Detailed)**
- Python tests: `monolith/agent_service/backends_test.py`
- Rust tests: add ZK mock tests for layout watches and serialization.

**Gaps / Notes**
- No Rust ZK backend yet; needs full parity implementation.

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

### `monolith/agent_service/backends_test.py`
<a id="monolith-agent-service-backends-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 134
- Purpose/role: Tests ZKBackend saved_model declaration, layout callbacks, bindings, and sync targets.
- Key symbols/classes/functions: `ZKBackendTest` methods
- External dependencies: `FakeKazooClient`, `ZKBackend`, `SavedModel`, `SavedModelDeployConfig`.
- Side effects: ZK node creation in fake client.

**Required Behavior (Detailed)**
- `setUpClass`:
  - Initialize ZKBackend with FakeKazooClient, report service info, register layout callback.
- `test_register_service`: `get_service_info(container)` equals originally reported service info.
- `test_layout_callback`:
  - Declare saved_models and add to layout; callback receives list of `(SavedModel, DeployConfig)` in order.
  - Removing entry updates callback list.
- `test_sync_available_models`: `sync_available_saved_models` creates binding nodes under `/bzid/binding/...`.
- `test_service_map`: `get_service_map` returns service info list for entry/ps_0.
- `test_sync_backend`: `subscribe_model` + `sync_available_saved_models` produces `get_sync_targets` for a subgraph.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/tests/backends.rs`.
- Rust public API surface: ZKBackend + SavedModel + DeployConfig.

**Implementation Steps (Detailed)**
1. Port Fake ZK backend + ZKBackend logic.
2. Port tests for layout callback and binding map parity.
3. Ensure ordering of saved_models matches Python list order.

**Tests (Detailed)**
- Python tests: this file
- Rust tests: parity tests mirroring the same assertions.

**Gaps / Notes**
- Requires deterministic layout callback ordering in Rust.

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

### `monolith/agent_service/client.py`
<a id="monolith-agent-service-client-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 126
- Purpose/role: Lightweight CLI client for load/unload/status via ZKMirror (portal/publish/service inspection).
- Key symbols/classes/functions: `ServingClient`, `LoadSate`, `main`
- External dependencies: `MonolithKazooClient`, `ZKMirror`, `ModelMeta`, `ReplicaMeta`, TF Serving `ModelVersionStatus`.
- Side effects: ZK reads/writes, prints status, env setup.

**Required Behavior (Detailed)**
- `LoadSate` dataclass: `portal: bool`, `publish: bool`, `service: dict`.
- `ServingClient.__init__`:
  - Creates `MonolithKazooClient` and `ZKMirror(zk, bzid)`; starts mirror with `is_client=True`.
- `load(model_name, model_dir, ckpt=None, num_shard=-1)`:
  - Creates `ModelMeta`, computes path under `portal_base_path`.
  - If path exists, raise `RuntimeError('{model_name} has exists')`.
  - Otherwise create node with serialized meta.
- `unload(model_name)`:
  - Delete portal node if exists; else log warning.
- `get_status(model_name)`:
  - `portal` True if `/bzid/portal/{model_name}` exists.
  - `publish` True if any `/bzid/publish/{shard}:{replica}:{model_name}` exists.
  - `service` map from `server_type:task:replica` to `ReplicaMeta.stat` for all replicas under `/bzid/service/{model_name}`.
- `main`:
  - Requires `zk_servers` and `bzid` flags.
  - `cmd_type` `load` or `unload`, otherwise prints `get_status`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-cli/src/bin/serving_client.rs`.
- Rust public API surface: CLI entrypoint + `ServingClient` struct.
- Data model mapping: `ModelMeta`, `ReplicaMeta`, `ModelState` enums.

**Implementation Steps (Detailed)**
1. Port `ServingClient` with ZKMirror client mode.
2. Implement portal node create/delete semantics and error message parity.
3. Implement status inspection across portal/publish/service paths.
4. Port CLI flags and default cmd_type handling.

**Tests (Detailed)**
- Python tests: none specific
- Rust tests: add fake ZK tests for `load/unload/get_status`.

**Gaps / Notes**
- Rust needs ZKMirror equivalent and a way to read/write serialized ModelMeta.

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

### `monolith/agent_service/constants.py`
<a id="monolith-agent-service-constants-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 3
- Purpose/role: constant definitions for agent service.
- Key symbols/classes/functions: `HOST_SHARD_ENV`
- External dependencies: none
- Side effects: none

**Required Behavior (Detailed)**
- Export string constant `MONOLITH_HOST_SHARD_N`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/src/constants.rs` (new) or existing config module.

**Implementation Steps (Detailed)**
1. Add Rust constant with identical name/value.
2. Ensure all config code references same constant.

**Tests (Detailed)**
- Python tests: none.
- Rust tests: optional constant presence test.

**Gaps / Notes**
- Trivial but must be mirrored.

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

### `monolith/agent_service/data_def.py`
<a id="monolith-agent-service-data-def-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 173
- Purpose/role: Data model definitions for agent service (ModelMeta, ResourceSpec, PublishMeta, ReplicaMeta, Event, enums).
- Key symbols/classes/functions: `ModelMeta`, `ResourceSpec`, `PublishMeta`, `ReplicaMeta`, `Event`, `PublishType`, `EventType`.
- External dependencies: `dataclasses_json`, `tensorflow_serving` protos, `AddressFamily`.
- Side effects: none; pure data serialization + address selection logic.

**Required Behavior (Detailed)**
- JSON serialization/deserialization exactly matches dataclasses_json output.
- `ReplicaMeta.get_address` resolves IPv4/IPv6 with archon preference and filters `0.0.0.0` / `[::]`.
- `get_path` methods build correct ZK paths.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/src/data_def.rs` (new) or `monolith-rs/crates/monolith-core/src`.
- Rust public API surface: structs with serde JSON; enums for state.
- Data model mapping: use `monolith_proto` for ModelState where appropriate.

**Implementation Steps (Detailed)**
1. Port all dataclasses to Rust structs with serde JSON.
2. Implement `serialize`/`deserialize` helpers to match Python bytes encoding.
3. Port `ReplicaMeta.get_address` and path helpers exactly.

**Tests (Detailed)**
- Python tests: `monolith/agent_service/data_def_test.py`
- Rust tests: add roundtrip JSON + address selection tests.

**Gaps / Notes**
- No Rust equivalents yet.

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

### `monolith/agent_service/data_def_test.py`
<a id="monolith-agent-service-data-def-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 52
- Purpose/role: Tests serialization/deserialization roundtrip for `ModelMeta`, `ResourceSpec`, `ReplicaMeta`.
- Key symbols/classes/functions: `DataDefTest.serde` and tests.
- External dependencies: `monolith.agent_service.data_def`.
- Side effects: none.

**Required Behavior (Detailed)**
- `serde(item)`:
  - `serialized = item.serialize()`
  - `recom = cls.deserialize(serialized)`
  - Asserts equality.
- Tests:
  - `ModelMeta` with `model_name`, `num_shard`, `model_dir`, `ckpt`.
  - `ResourceSpec` with `address`, `shard_id`, `replica_id`, `memory`, `cpu`.
  - `ReplicaMeta` with `address`, `model_name`, `server_type`, `task`, `replica`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/tests/data_def.rs`.
- Rust public API surface: serialization/deserialization methods for the same structs.

**Implementation Steps (Detailed)**
1. Port `serialize`/`deserialize` format (JSON or bytes) to Rust.
2. Ensure equality comparisons are field-wise identical.
3. Add tests for roundtrip parity.

**Tests (Detailed)**
- Python tests: this file
- Rust tests: same data roundtrip assertions.

**Gaps / Notes**
- Serialization format must match Python exactly (field names, defaults).

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

### `monolith/agent_service/mocked_tfserving.py`
<a id="monolith-agent-service-mocked-tfserving-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 399
- Purpose/role: In-process fake TensorFlow Serving gRPC server for tests.
- Key symbols/classes/functions: `ModelConf`, `ModelVersion`, `ModelMeta`, `Event`, `ModelMgr`, `ModelServiceImpl`, `PredictionServiceImpl`, `FakeTFServing`.
- External dependencies: `grpc`, `tensorflow_serving.apis.*`, `model_server_config_pb2`, utils for status/model_config.
- Side effects: starts gRPC server; spawns background thread to update model states.

**Required Behavior (Detailed)**
- `ModelMgr`:
  - Manages models and versions, with `load`, `remove`, `get_status`, `get_metadata`.
  - `load()`:
    - For each ModelConfig, derive version policy (`latest`, `all`, or `specific`) and create `ModelVersion` list.
    - Enqueue `Event(state=START)` for each version.
  - `remove()`:
    - Marks model as unloading, enqueue `UNLOADING` events for each version.
  - `_poll()` thread:
    - Processes queued events and transitions version state: UNKNOWN -> START -> LOADING -> AVAILABLE -> UNLOADING -> END.
    - For `latest` policy, unloads oldest when max exceeded.
    - Periodically adds new version for non-specific policies.
  - `get_status(model_spec)`:
    - Returns version statuses (or NOT_FOUND status if missing).
  - `get_metadata(model_spec, metadata_field)`:
    - Returns requested metadata from model_conf and version fields.
- `ModelServiceImpl`:
  - `GetModelStatus` returns `GetModelStatusResponse` with version statuses.
  - `HandleReloadConfigRequest`:
    - Removes old models not in new config; loads new ones; returns OK status.
- `PredictionServiceImpl.GetModelMetadata`:
  - Populates metadata `Any` values (byte-encoded repr of fields).
- `FakeTFServing`:
  - Can be constructed from a single model config, a config file, or a ModelServerConfig.
  - Starts gRPC server on `port` with ModelService + PredictionService.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/tests/support/fake_tfserving.rs`.
- Rust public API surface: Fake gRPC server that emulates ModelService and PredictionService.
- Data model mapping: ModelVersionStatus, ModelSpec, ReloadConfigRequest, MetadataResponse.

**Implementation Steps (Detailed)**
1. Implement in-memory model manager with same state transitions and policies.
2. Implement gRPC services for ModelService and PredictionService.
3. Parse ModelServerConfig text/proto to initialize models.
4. Provide `start()`/`stop()` APIs for tests.

**Tests (Detailed)**
- Python tests: `monolith/agent_service/mocked_tfserving_test.py`
- Rust tests: ensure fake server supports metadata/status/reload as in Python.

**Gaps / Notes**
- Must preserve timing semantics for state transitions used by other tests.

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

### `monolith/agent_service/mocked_tfserving_test.py`
<a id="monolith-agent-service-mocked-tfserving-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 92
- Purpose/role: Tests FakeTFServing gRPC endpoints for metadata, status, and reload config.
- Key symbols/classes/functions: `MockedTFSTest` methods.
- External dependencies: gRPC stubs for ModelService and PredictionService.
- Side effects: starts FakeTFServing in background thread.

**Required Behavior (Detailed)**
- `setUpClass`: start FakeTFServing on a free port, wait for readiness.
- `test_get_model_metadata`: call `GetModelMetadata` and assert response type.
- `test_get_model_status`: call `GetModelStatus` and assert response type.
- `test_handle_reload_config_request`: send `ReloadConfigRequest` with two model configs; assert OK response type.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/tests/mocked_tfserving.rs`.
- Rust public API surface: FakeTFServing server for test; gRPC client stubs.

**Implementation Steps (Detailed)**
1. Recreate FakeTFServing server in Rust tests.
2. Add client requests for metadata/status/reload.
3. Assert response types and basic fields.

**Tests (Detailed)**
- Python tests: this file
- Rust tests: parity tests for gRPC responses.

**Gaps / Notes**
- Ensure server thread lifecycle (start/stop) matches Python behavior.

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

### `monolith/agent_service/mocked_zkclient.py`
<a id="monolith-agent-service-mocked-zkclient-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 377
- Purpose/role: In-memory fake Kazoo/ZooKeeper client with watches, nodes, and basic CRUD.
- Key symbols/classes/functions: `ChildrenWatch`, `DataWatch`, `Election`, `Node`, `Catalog`, `FakeKazooClient`.
- External dependencies: `kazoo.protocol.states`, `kazoo.exceptions`.
- Side effects: in-memory state updates and watch callbacks.

**Required Behavior (Detailed)**
- `Node`:
  - Tracks `path`, `value`, `ephemeral`, `children`, timestamps, and version.
  - `set()` updates value/version and triggers data watch with CHANGED event.
  - `create_child()` creates child Node and triggers parent children watch.
  - `remove_child()` handles recursive delete and triggers children watch; raises NotEmptyError if needed.
  - `__del__` triggers DELETED events and cleans children.
- `Catalog`:
  - Holds root node; maintains data/children watches and sequence counters.
  - `ensure_path()` creates nodes along path.
  - `create()` supports `makepath` and `sequence` numbering; raises NodeExistsError if exists.
  - `delete()`, `set()`, `get()` operate on nodes.
- `FakeKazooClient`:
  - API compatible subset: `start`, `stop`, `create`, `delete`, `set`, `get`, `exists`, `get_children`, `ensure_path`, `retry`.
  - Provides `DataWatch`, `ChildrenWatch`, `Election` constructors via `partial`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/tests/support/fake_zk.rs`.
- Rust public API surface: fake ZK client with watches and CRUD for tests.

**Implementation Steps (Detailed)**
1. Implement Node tree with watch callbacks and version tracking.
2. Implement create/delete/set/get semantics and exceptions.
3. Support `sequence` numbering for `create`.
4. Provide compatible DataWatch/ChildrenWatch interfaces for tests.

**Tests (Detailed)**
- Python tests: `monolith/agent_service/mocked_zkclient_test.py`
- Rust tests: add unit tests for CRUD and watches.

**Gaps / Notes**
- Must mimic Kazoo event types and ordering as closely as possible.

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

### `monolith/agent_service/mocked_zkclient_test.py`
<a id="monolith-agent-service-mocked-zkclient-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 130
- Purpose/role: Tests FakeKazooClient CRUD and watch behaviors.
- Key symbols/classes/functions: `MockedZKClientTest` methods.
- External dependencies: `FakeKazooClient`, kazoo exceptions.
- Side effects: creates and deletes in-memory nodes; prints watch outputs.

**Required Behavior (Detailed)**
- `test_create`: create path with `makepath=True` and assert returned path.
- `test_set_get`: create node, set/get, verify NoNodeError on invalid path.
- `test_delete`: delete node and parent path.
- `test_data_watch`: register DataWatch and ensure callback is invoked.
- `test_children_watch`: register ChildrenWatch with `send_event=True` and ensure callback is invoked on child creation.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/tests/mocked_zkclient.rs`.
- Rust public API surface: Fake ZK client and watch callbacks.

**Implementation Steps (Detailed)**
1. Port FakeKazooClient tests to Rust.
2. Assert correct errors for missing nodes.
3. Ensure watch callbacks are invoked with expected events.

**Tests (Detailed)**
- Python tests: this file
- Rust tests: direct port with similar assertions.

**Gaps / Notes**
- Rust watch callback API may need adaptation; keep behavior parity.

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

### `monolith/agent_service/model_manager.py`
<a id="monolith-agent-service-model-manager-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 371
- Purpose/role: Copies latest model versions from a source path to a local receive path with lock/marker semantics.
- Key symbols/classes/functions: `ModelManager` and methods.
- External dependencies: `os`, `shutil`, `threading`, `monolith.native_training.metric.cli`.
- Side effects: filesystem reads/writes, directory copies, lock files, metrics emission.

**Required Behavior (Detailed)**
- Constants:
  - `WRITE_DONE = '.write.done'`, `READ_LOCK = '.read.lock'`.
- `start()`:
  - If `model_name` is None, return True.
  - Delete receive path.
  - Wait for source path to exist and for `*.write.done` marker matching model.
  - Run `loop_once()` until a model copy succeeds; then remove read locks and start background thread.
- `loop_once()`:
  - Reads `source_data` via `get_source_data` (latest version per model).
  - For each model, if new version > last, copy model data via `copy_model`.
  - Updates `_models` and `_latest_models` (version + update time).
- `copy_model()`:
  - Copies each sub_model to a `-temp` dir, then atomically renames.
  - If any copy fails, cleans temp dirs and returns failure.
- `get_source_data()`:
  - Walks source dir; detects model dirs named `model@version` and `.write.done` files.
  - Creates read locks under each `model@version` dir.
  - For each model, selects latest version and builds list of `(sub_model/version, path)` entries.
- `remove_read_lock()`:
  - Deletes local lock files and any stray `*.read.lock` in source path.
- Metrics:
  - `check_model_update_time()` emits `version.delay` and `update.delay` for latest model.
  - `_check_version` not present here; only per-loop metrics.
- `remove_old_file()` keeps last `remain_version_num` versions.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/src/model_manager.rs`.
- Rust public API surface: `ModelManager` struct with `start/stop` and loop logic.
- Data model mapping: filesystem semantics only.

**Implementation Steps (Detailed)**
1. Port lock/marker file semantics (`.write.done`, `.read.lock`).
2. Port `copy_model` to use temp dirs + atomic rename.
3. Implement background loop with configurable intervals/timeouts.
4. Port metrics emission (optional feature flag).

**Tests (Detailed)**
- Python tests: `monolith/agent_service/model_manager_test.py`
- Rust tests: replicate file layout and verify copy + ignore old versions.

**Gaps / Notes**
- Ensure delete semantics are safe and match Python (file vs dir removal).

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

### `monolith/agent_service/model_manager_test.py`
<a id="monolith-agent-service-model-manager-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 113
- Purpose/role: Tests ModelManager copying behavior and ignoring older versions.
- Key symbols/classes/functions: `ModelManagerTest.test_start`, `.test_ignore_old`
- External dependencies: filesystem, `ModelManager`.
- Side effects: creates temp directories and files.

**Required Behavior (Detailed)**
- `create_file(model_name, timestamp, p2p_data_path)`:
  - Creates `model@timestamp/model/ps_item_embedding_*/timestamp` directories.
  - Writes `model@timestamp.write.done` marker.
- `test_start`:
  - Creates p2p data with one version.
  - Starts ModelManager and asserts model copies exist under receive path.
- `test_ignore_old`:
  - Creates new version, starts manager, then creates older version.
  - Verifies older version is not copied.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/tests/model_manager.rs`.
- Rust public API surface: `ModelManager` with configurable timeouts/intervals.

**Implementation Steps (Detailed)**
1. Port test directory setup helpers.
2. Validate copy results and ignore-old behavior.

**Tests (Detailed)**
- Python tests: this file
- Rust tests: parity tests with temp dirs.

**Gaps / Notes**
- Ensure Rust tests use temp directories and cleanup to avoid flakiness.

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

### `monolith/agent_service/replica_manager.py`
<a id="monolith-agent-service-replica-manager-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 835
- Purpose/role: Maintains replica registration and status updates in ZK; watches for replica changes and exposes lookup APIs.
- Key symbols/classes/functions: `ReplicaWatcher`, `ReplicaUpdater`, `ZKListener`, `ReplicaManager`, `SyncBackendWrapper`.
- External dependencies: `kazoo`, `TFSMonitor`, `AgentConfig`, `ReplicaMeta`, `ModelState`, metrics.
- Side effects: ZK watches, ephemeral node creation, periodic polling, metrics emission.

**Required Behavior (Detailed)**
- `ReplicaWatcher`:
  - Watches ZK paths under `/{bzid}/service/{base_name}`.
  - Supports dc-aware paths (`idc:cluster/server_type:task`) and non-dc paths.
  - Uses ChildrenWatch/DataWatch to maintain `replicas` dict: `{task_path: {replica_id: ReplicaMeta}}`.
  - `_poll()` resyncs periodically (every 60s) and re-registers missing ephemeral nodes (when local replica was removed).
  - `get_all_replicas`, `get_replicas`, `get_replica`, `get_replicas_with_extra_info` filter by server type, idc/cluster, and ModelState.AVAILABLE.
- `ReplicaUpdater`:
  - `register()` creates ephemeral nodes for entry/ps/dense; supports `replica_id == -1` (sequence node) for entry.
  - `_do_update()` queries `TFSMonitor.get_model_status` and updates ZK with new ModelState.
  - `_updater` loop updates each model_name; handles exceptions by setting UNKNOWN.
  - `_check_version` emits metrics for latest model version and update timestamps.
  - `_reregister` loop re-registers after ZK connection loss.
- `ZKListener`:
  - On LOST: disables polling and updating; on reconnect: triggers reregister.
- `ReplicaManager`:
  - Combines watcher and updater; exposes lookup APIs and `is_ps_set_started` / `is_dense_set_started`.
- `SyncBackendWrapper`:
  - Implements `SyncBackend` for parameter sync; returns replicas with extra info.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/src/replica_manager.rs`.
- Rust public API surface: `ReplicaManager`, `ReplicaWatcher`, `ReplicaUpdater`, `SyncBackendWrapper`.
- Data model mapping: `ReplicaMeta`, `ModelState`, `TFSServerType`, `ServerType`.
- Integration points: ZK client + TFSMonitor + metrics client.

**Implementation Steps (Detailed)**
1. Implement watcher with ZK watches and periodic polling for reconciliation.
2. Implement updater with registration, status updates, and metrics.
3. Port dc-aware path parsing (`ZKPath`) and address family handling.
4. Port connection loss handling and re-registration semantics.
5. Provide SyncBackend wrapper for parameter sync features.

**Tests (Detailed)**
- Python tests: `monolith/agent_service/replica_manager_test.py`
- Rust tests: fake ZK + fake TFS server to validate registration and updates.

**Gaps / Notes**
- This module depends on TFSMonitor (gRPC) and ZK; ensure fakes are available for tests.

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

### `monolith/agent_service/replica_manager_test.py`
<a id="monolith-agent-service-replica-manager-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 126
- Purpose/role: Partial setup for ReplicaManager tests; builds FakeTFServing and helper registration.
- Key symbols/classes/functions: `ReplicaMgrTest.setUpClass`, `register` helper.
- External dependencies: FakeTFServing, FakeKazooClient, ReplicaWatcher/Updater.
- Side effects: starts FakeTFServing servers on entry/ps ports.

**Required Behavior (Detailed)**
- `setUpClass`:
  - Sets env vars for shard/replica/idc/cluster.
  - Constructs `AgentConfig` (deploy_type mixed, dc_aware=True).
  - Parses command strings to find model_config_file paths.
  - Starts FakeTFServing for entry and ps; runs in background threads.
- `register(zk)`:
  - Creates ReplicaMeta nodes for PS tasks and entry replicas (excluding current shard/replica).
- Note: file currently has no actual test methods beyond setup.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/tests/replica_manager.rs`.
- Rust public API surface: FakeTFServing + ReplicaManager registration.

**Implementation Steps (Detailed)**
1. Port setup to Rust tests (FakeTFServing, AgentConfig).
2. Add explicit test cases in Rust (missing in Python) for registration and lookup.

**Tests (Detailed)**
- Python tests: this file (incomplete)
- Rust tests: implement meaningful coverage for ReplicaManager behavior.

**Gaps / Notes**
- Python test file is incomplete; Rust should add assertions to cover behaviors.

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

### `monolith/agent_service/resource_utils.py`
<a id="monolith-agent-service-resource-utils-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 269
- Purpose/role: Resource and model-size utilities (memory, CPU, model size computation, HDFS helpers).
- Key symbols/classes/functions: `cal_model_info_v2`, `total_memory`, `cal_available_memory`, `CPU`, `num_cpu`, `cal_cpu_usage`.
- External dependencies: `tensorflow.io.gfile`, `psutil`, `subprocess`, `export_state_utils`.
- Side effects: reads cgroup files, runs shell commands, reads filesystem/HDFS.

**Required Behavior (Detailed)**
- `_get_pod_cgroup_path()` reads `/proc/1/cgroup` and extracts memory cgroup path.
- `exists(dirname)` uses `tf.io.gfile.isdir` or `tf.io.gfile.exists`.
- `open_hdfs(fname)` runs `${_HADOOP_BIN} fs -text` and yields non-empty lines.
- `cal_model_info_v2(exported_models_path, ckpt=None, version=None)`:
  - Resolves absolute path; asserts exists.
  - Lists sub_model names under exported_models_path.
  - Determines `ckpt` from `tf.train.get_checkpoint_state` if missing; derives `global_step`.
  - Determines version: uses `export_state_utils` or lists numeric dirs; intersects across sub_models; uses latest if not specified.
  - Sums file sizes under each sub_model/version path.
  - Adds assets size from `{ckpt}.assets` files matching regex `^.+_(\d+)-\d+-of-\d+$`.
  - Returns map `{sub_model_name: (size, version_path)}`.
- `total_memory`/`cal_available_memory`: read cgroup memory limit/usage; fall back to `MY_MEM_LIMIT` env.
- `total_memory_v2`/`cal_available_memory_v2`: use `psutil.virtual_memory()`.
- `CPU` helper reads cpuacct usage and computes usage delta.
- `num_cpu` uses cgroup `cpu.cfs_quota_us` and `cpu.cfs_period_us`, fallback `MY_CPU_LIMIT`.
- `cal_cpu_usage` samples CPU usage over 5 seconds and averages; `cal_cpu_usage_v2` uses `psutil.cpu_percent()`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/src/resource_utils.rs`.
- Rust public API surface: functions for model size + resource metrics.
- Data model mapping: `SubModelName`, `SubModelSize`, `VersionPath` equivalents.

**Implementation Steps (Detailed)**
1. Implement cgroup parsing for memory/cpu or provide platform-specific fallbacks.
2. Port `cal_model_info_v2` using filesystem traversal and checkpoint state parsing.
3. Replace `tf.io.gfile` with Rust FS/HDFS abstraction as needed.
4. Provide `psutil`-equivalent data via `sysinfo` or `/proc` parsing.

**Tests (Detailed)**
- Python tests: `monolith/agent_service/resource_utils_test.py`
- Rust tests: validate memory/cpu functions and model info on fixture dirs.

**Gaps / Notes**
- Requires HDFS support or a stub for `open_hdfs`.

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

### `monolith/agent_service/resource_utils_test.py`
<a id="monolith-agent-service-resource-utils-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 36
- Purpose/role: Tests resource utility functions (memory/cpu).
- Key symbols/classes/functions: `UtilTest.test_cal_avaiable_memory_v2`, `.test_cal_cpu_usage_v2`
- External dependencies: `resource_utils`.
- Side effects: reads system stats.

**Required Behavior (Detailed)**
- `test_cal_avaiable_memory_v2`: asserts `0 < available < total`.
- `test_cal_cpu_usage_v2`: asserts CPU usage is in `[0, 100]`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/tests/resource_utils.rs`.
- Rust public API surface: resource utility functions.

**Implementation Steps (Detailed)**
1. Port tests with `sysinfo` or `/proc` sources.
2. Ensure numeric bounds match Python behavior.

**Tests (Detailed)**
- Python tests: this file
- Rust tests: same assertions.

**Gaps / Notes**
- Tests may be flaky on constrained CI; consider tolerances.

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

### `monolith/agent_service/run.py`
<a id="monolith-agent-service-run-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 39
- Purpose/role: Entry-point multiplexer for `agent`, `agent_client`, and `tfs_client` binaries.
- Key symbols/classes/functions: `main`
- External dependencies: `absl.app`, `flags`, agent and client entrypoints.
- Side effects: dispatches into selected CLI.

**Required Behavior (Detailed)**
- Flag `bin_name` selects:
  - `agent` -> `monolith.agent_service.agent.main`
  - `agent_client` -> `monolith.agent_service.agent_client.main`
  - `tfs_client` -> `monolith.agent_service.tfs_client.main`
- Unknown value raises `ValueError`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-cli/src/bin/monolith.rs` (dispatcher) or separate binaries.
- Rust public API surface: CLI dispatch or multi-bin setup.

**Implementation Steps (Detailed)**
1. Decide whether to provide a dispatcher binary or separate `cargo` bins.
2. If dispatcher, implement `bin_name` flag with same choices.
3. Forward args to target subcommand.

**Tests (Detailed)**
- Python tests: none
- Rust tests: optional CLI smoke test.

**Gaps / Notes**
- Rust may prefer multiple bins; document any deviations.

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

### `monolith/agent_service/svr_client.py`
<a id="monolith-agent-service-svr-client-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 70
- Purpose/role: Thin gRPC client wrapper for AgentService (heart_beat and get_replicas).
- Key symbols/classes/functions: `SvrClient`, `heart_beat`, `get_replicas`
- External dependencies: `grpc`, `AgentServiceStub`, `AgentConfig`.
- Side effects: gRPC calls, prints responses.

**Required Behavior (Detailed)**
- `__init__`: accepts config path or `AgentConfig` object; defers stub creation.
- `stub` property:
  - Uses `MY_HOST_IP` env or local hostname; connects to `{host}:{agent_port}`.
- `get_server_type`:
  - If input is string, maps `ps/entry/dense` to enum using `FLAGS.server_type` (note: uses global flags).
- `heart_beat`: sends `HeartBeatRequest` and prints addresses.
- `get_replicas`: sends `GetReplicasRequest` and prints address list.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-cli/src/agent_svr_client.rs` or library module.
- Rust public API surface: `SvrClient` with `heart_beat` and `get_replicas`.

**Implementation Steps (Detailed)**
1. Implement gRPC stub creation with env host selection.
2. Port enum mapping for server types.
3. Preserve stdout printing behavior (for CLI usage).

**Tests (Detailed)**
- Python tests: none
- Rust tests: add gRPC stub tests using a fake AgentService.

**Gaps / Notes**
- The string mapping uses global FLAGS; ensure Rust CLI has equivalent.

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

### `monolith/agent_service/tfs_client.py`
<a id="monolith-agent-service-tfs-client-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 503
- Purpose/role: CLI for TensorFlow Serving status/metadata/load/predict/profile; includes data generation and format conversion.
- Key symbols/classes/functions: `read_header`, `read_data`, `generate_random_instance`, `get_instance_proto`, `get_example_batch_proto`, `get_example_batch_to_instance`, `ProfileThread`, `main`.
- External dependencies: TF Serving protos, matrix `ExampleBatch`/`Instance`, `FeatureList`, `data_gen_utils`.
- Side effects: reads/writes files, makes gRPC requests, spawns threads, prints output.

**Required Behavior (Detailed)**
- Input parsing helpers:
  - `read_header` consumes Kafka/LagrangeX header bytes depending on flags.
  - `read_data` reads size-prefixed payload (8-byte little-endian length).
- Data generation:
  - `generate_random_instance` produces random fid values based on slots/vocab sizes.
  - `generate_random_example_batch` builds ExampleBatch with random fid values and LineId.
  - `gen_random_file` uses `data_gen_utils` to create random data files.
- Tensor proto conversion:
  - `get_instance_proto`: returns TensorProto from list of serialized Instance bytes.
  - `get_example_batch_proto`: reads ExampleBatch from file or generates random; returns TensorProto.
  - `get_example_batch_to_instance`: converts ExampleBatch to list of Instance messages (handles fid_v1/fid_v2/float/int/bytes fields).
- CLI (`main`):
  - Uses `AgentConfig.from_file` and env host to pick target port (entry/ps/dense).
  - `status`: GetModelStatus request.
  - `meta`: GetModelMetadata request.
  - `load`: ReloadConfigRequest using a ModelServerConfig pbtxt file.
  - `profile`: generates/loads ExampleBatch data, spawns `ProfileThread`s, computes avg latency, p99, QPS.
  - Default: send PredictRequest with input type (instance/example_batch) and print response.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-cli/src/bin/tfs_client.rs`.
- Rust public API surface: CLI entrypoint + helper functions for data parsing and proto conversion.
- Data model mapping: matrix ExampleBatch/Instance protos + TF Serving PredictRequest.
- Feature gating: `tf-runtime` and `matrix-protos` features.

**Implementation Steps (Detailed)**
1. Port file header parsing and size-prefixed record reading.
2. Implement ExampleBatch/Instance generation and conversion logic.
3. Implement gRPC clients for ModelService and PredictionService.
4. Port profiling logic with multi-threaded request loop and latency stats.
5. Mirror CLI flags and defaults exactly.

**Tests (Detailed)**
- Python tests: `monolith/agent_service/tfs_client_test.py`
- Rust tests: port get_instance_proto and get_example_batch_to_instance tests.

**Gaps / Notes**
- Python references `user_features` in `get_example_batch_proto_v2` without local definition; confirm source of this global in Rust.

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

### `monolith/agent_service/tfs_client_test.py`
<a id="monolith-agent-service-tfs-client-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 50
- Purpose/role: Tests tensor proto generation and ExampleBatch-to-Instance conversion.
- Key symbols/classes/functions: `TFSClientTest.test_get_instance_proto`, `.test_get_example_batch_to_instance_*`
- External dependencies: `tfs_client` helpers.
- Side effects: reads test data files.

**Required Behavior (Detailed)**
- `test_get_instance_proto`: asserts dtype and tensor shape for random instance batch.
- `test_get_example_batch_to_instance_from_pb`: reads binary examplebatch file with header.
- `test_get_example_batch_to_instance_from_pbtxt`: reads pbtxt example batch file.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-cli/tests/tfs_client.rs`.
- Rust public API surface: helper functions for instance/example batch parsing.

**Implementation Steps (Detailed)**
1. Port tests using fixture files (`examplebatch.data`, `example_batch.pbtxt`).
2. Assert dtype and shapes match Python behavior.

**Tests (Detailed)**
- Python tests: this file
- Rust tests: parity tests for parsing and tensor construction.

**Gaps / Notes**
- Ensure Rust uses the same byte-order and header handling as Python.

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

### `monolith/agent_service/tfs_monitor.py`
<a id="monolith-agent-service-tfs-monitor-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 303
- Purpose/role: gRPC monitor for TensorFlow Serving model status and config reload.
- Key symbols/classes/functions: `TFSMonitor`, `get_model_status` (singledispatch), `gen_model_config`, `handle_reload_config_request`.
- External dependencies: TF Serving gRPC stubs, `ModelServerConfig`, `PublishMeta`.
- Side effects: gRPC calls to TFS servers.

**Required Behavior (Detailed)**
- Maintains gRPC stubs for ENTRY/PS/DENSE based on deploy_type and ports.
- `get_addr(sub_model_name)` chooses port based on deploy type and sub_model type.
- `get_service_type(sub_model_name)` returns TFSServerType or None.
- `get_model_status(PublishMeta)`:
  - For each sub_model, builds `GetModelStatusRequest`.
  - For dense nodes (entry when dense-along-entry), may omit version unless `fix_dense_version`.
  - On RPC errors, returns UNKNOWN with StatusProto error code/details.
  - Returns map `{tfs_model_name: (version_path, ModelVersionStatus)}`.
- `get_model_status(name, version=None, signature_name=None)`:
  - Returns list of ModelVersionStatus for a model via `GetModelStatus`.
- `gen_model_config(pms)`:
  - Builds ModelServerConfig per service type from PublishMeta list.
  - For dense nodes: use `latest` policy unless `fix_dense_version`.
  - For ps/entry: `specific` policy with version number.
- `handle_reload_config_request(service_type, model_configs)`:
  - Ensures default model config is present.
  - Sends ReloadConfigRequest to appropriate service.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/src/tfs_monitor.rs`.
- Rust public API surface: `TFSMonitor` struct with status and reload APIs.
- Data model mapping: TF Serving protos and `PublishMeta` equivalents.

**Implementation Steps (Detailed)**
1. Port gRPC client setup for entry/ps/dense.
2. Implement singledispatch-like overloads (Rust traits/enum arguments).
3. Port model config generation logic and dense version policy.
4. Add default model config injection.

**Tests (Detailed)**
- Python tests: `monolith/agent_service/tfs_monitor_test.py`
- Rust tests: start fake TFS servers and verify reload/status.

**Gaps / Notes**
- Requires TF Serving protos + stubs in Rust.

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

### `monolith/agent_service/tfs_monitor_test.py`
<a id="monolith-agent-service-tfs-monitor-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 182
- Purpose/role: Tests TFSMonitor reload and remove config with FakeTFServing.
- Key symbols/classes/functions: `TFSMonitorTest.test_reload_config`, `.test_remove_config`
- External dependencies: FakeTFServing, ModelServerConfig, PublishMeta.
- Side effects: starts fake TF serving servers.

**Required Behavior (Detailed)**
- Setup:
  - Start FakeTFServing for entry and ps ports; wait for readiness.
  - Create `TFSMonitor` and connect.
- `test_reload_config`:
  - Generate PublishMeta list with random ps counts and entry submodel.
  - Call `gen_model_config` then `handle_reload_config_request` per service type.
- `test_remove_config`:
  - Similar to reload config but with different models; ensures reload path can remove models.
- `tearDown`: compares before/after status; ensures NOT_FOUND responses for unloaded models.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/tests/tfs_monitor.rs`.
- Rust public API surface: TFSMonitor + FakeTFServing.

**Implementation Steps (Detailed)**
1. Port fake TFS server setup.
2. Port PublishMeta-based config generation and reload requests.
3. Assert status responses match Python expectations (NOT_FOUND or version numbers).

**Tests (Detailed)**
- Python tests: this file
- Rust tests: parity test for reload/remove behavior.

**Gaps / Notes**
- Requires deterministic fake TFS server behavior for version states.

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

### `monolith/agent_service/tfs_wrapper.py`
<a id="monolith-agent-service-tfs-wrapper-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 202
- Purpose/role: Wraps TensorFlow Serving process launch, config file handling, and model status queries.
- Key symbols/classes/functions: `TFSWrapper`, `FakeTFSWrapper`
- External dependencies: `subprocess`, `grpc`, TF Serving protos.
- Side effects: launches external process, writes logs, opens gRPC channel.

**Required Behavior (Detailed)**
- `TFSWrapper.__init__`:
  - Saves ports, config file, binary config, log path.
  - Uses `strings $TFS_BINARY | grep PredictionServiceGrpc` to detect grpc remote op support.
- `_prepare_cmd()`:
  - Builds CLI flags: model_config_file, ports, poll interval, archon settings, metrics prefix.
  - If grpc remote op absent, adds `archon_entry_to_ps_rpc_timeout`.
  - Fills in defaults from `TfServingConfig` (incl. platform_config_file).
- `start()`:
  - `os.chdir(find_main())` and `subprocess.Popen` with stdout to log file.
  - Creates gRPC channel to `localhost:grpc_port` and ModelServiceStub.
- `stop()`:
  - Closes channel, closes log, kills process.
- `list_saved_models()`:
  - Parses model config file text into `ModelServerConfig` and returns model names.
- `list_saved_models_status()`:
  - For each saved model, calls `GetModelStatus`, selects available version or last, handles RPC errors.
- `FakeTFSWrapper`:
  - No process; reads model_config file and returns AVAILABLE for all models.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/src/tfs_wrapper.rs`.
- Rust public API surface: `TFSWrapper` + `FakeTFSWrapper` for tests.
- Feature gating: `tf-runtime` and `grpc` features.

**Implementation Steps (Detailed)**
1. Port command building logic and TfServingConfig mapping.
2. Implement process spawn + logging.
3. Implement gRPC status queries and model_config parsing.
4. Implement FakeTFSWrapper for tests.

**Tests (Detailed)**
- Python tests: used indirectly by `agent_v3_test`.
- Rust tests: use FakeTFSWrapper to validate list_saved_models/STATUS.

**Gaps / Notes**
- `TFS_BINARY` path and `find_main()` must map correctly in Rust.

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

### `monolith/agent_service/utils.py`
<a id="monolith-agent-service-utils-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: ~1000+
- Purpose/role: Core config + helper utilities for agent service, TF Serving configs, TensorProto creation, network utilities, and config file generation.
- Key symbols/classes/functions: `AgentConfig`, `DeployType`, `TFSServerType`, `gen_model_spec`, `gen_model_config`, `make_tensor_proto`, `get_local_ip`, many helpers.
- External dependencies: `tensorflow`, `tensorflow_serving` protos, `protobuf.text_format`, `json`, `socket`, `os`.
- Side effects: overrides `os.path.isabs`; writes platform config files; reads/writes files; inspects env; opens sockets.

**Required Behavior (Detailed)**
- Must preserve ALL defaults in `AgentConfig` and flag parsing (`flags.DEFINE_string('conf', ...)`).
- `AgentConfig.__post_init__` logic is critical for port allocation and layout config.
- `gen_model_spec` and `gen_model_config` must match TF Serving proto semantics.
- `make_tensor_proto` must mirror TF string tensor encoding for PredictRequest inputs.
- `get_local_ip` and port helpers must match network selection logic (IPv4/IPv6).

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/src/config.rs` + new `utils.rs`.
- Rust public API surface: `AgentConfig` struct + helpers for model spec and TensorProto assembly.
- Data model mapping: use `monolith_proto::tensorflow_serving::apis` for ModelSpec and `tensorflow_core::TensorProto`.

**Implementation Steps (Detailed)**
1. Port `AgentConfig` with all fields + defaults + env overrides.
2. Recreate port allocation logic, deploy type handling, and platform config file generation.
3. Port `gen_model_spec` and `gen_model_config` helpers with identical proto fields.
4. Implement `make_tensor_proto` for DT_STRING using TF Serving proto types.
5. Port network/IP helper methods (`get_local_ip`, `find_free_port`, etc.).
6. Mirror all file I/O (platform config) and text_format behavior.

**Tests (Detailed)**
- Python tests: `monolith/agent_service/utils_test.py`
- Rust tests: add unit tests for config defaults, model spec generation, and TensorProto creation.

**Gaps / Notes**
- This file is high-risk; many behaviors are implicit and must be traced manually.

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

### `monolith/agent_service/utils_test.py`
<a id="monolith-agent-service-utils-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 170
- Purpose/role: Tests utility helpers (model spec/config, status proto, AgentConfig parsing, instance parsing, ZKPath parsing).
- Key symbols/classes/functions: `ServingUtilsTest` methods.
- External dependencies: `monolith.agent_service.utils`.
- Side effects: reads config and test data files.

**Required Behavior (Detailed)**
- `gen_model_spec` and `gen_model_config` must set fields correctly (name, version, signature).
- `gen_status_proto` should preserve error_code and message.
- `gen_model_version_status` should match version and state.
- `AgentConfig.from_file` should load `agent.conf` and expose values (e.g., `stand_alone_serving`, `layout_filters`).
- `InstanceFormater` should parse json/pbtext/dump and produce TensorProto of correct dtype and shape.
- `get_cmd_and_port` should include `model_config_file_poll_wait_seconds` for agent_version 2.
- `ZKPath` parsing:
  - Full dc-aware path, partial path, and old (non-dc) paths must parse bzid/base_name/idc/cluster/server_type/index/replica_id correctly.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/tests/utils.rs`.
- Rust public API surface: utils module equivalents in Rust.

**Implementation Steps (Detailed)**
1. Port utilities and add tests for every helper above.
2. Ensure AgentConfig parsing matches Python (including list parsing for layout filters).
3. Port InstanceFormater-like parsing for JSON/pbtext/dump inputs.
4. Implement ZKPath parser with dc-aware and legacy formats.

**Tests (Detailed)**
- Python tests: this file
- Rust tests: parity tests with same fixtures.

**Gaps / Notes**
- InstanceFormater depends on test_data fixtures; ensure Rust can read them.

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

### `monolith/agent_service/zk_mirror.py`
<a id="monolith-agent-service-zk-mirror-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 672
- Purpose/role: ZK mirror/cache with watches for portal/publish/resource/service paths; coordinates scheduling and replica updates.
- Key symbols/classes/functions: `ZKMirror` and methods (watch_portal, watch_publish, expected_loading, update_service, election).
- External dependencies: `MonolithKazooClient`, `kazoo` watchers/election, `PublishMeta`, `ReplicaMeta`, `ResourceSpec`.
- Side effects: ZK watch registration, ZK CRUD, background threads, queue events.

**Required Behavior (Detailed)**
- Maintains in-memory `_data` cache of ZK paths to bytes.
- Defines base paths: `resource`, `portal`, `publish`, `service`, `locks`, `election`.
- CRUD helpers (`create`, `set`, `delete`, `exists`, `ensure_path`) wrap ZK and fall back to cache on errors.
- `report_resource`: writes ResourceSpec as ephemeral node.
- `resources` property: returns list of ResourceSpec from cached paths.
- `num_tce_replica`: waits until every replica id appears for all shards; returns count.
- `tce_replica_id`: uses env `REPLICA_ID` or derives from pod name.
- `publish_loadding`: writes PublishMeta entries to publish path; updates cache.
- `expected_loading`:
  - Groups PublishMeta by model_name; selects when all publish nodes have arrived.
  - Adjusts shard_id/replica_id for autoscaler and entry cases; filters sub_models to entry when needed.
- `update_service(replicas)`:
  - Computes paths for local replicas; removes outdated nodes; creates/updates current replicas.
- Replica queries: `get_all_replicas`, `get_model_replicas`, `get_task_replicas`, `get_replica` (AVAILABLE only).
- Watchers:
  - `watch_portal`: ensures portal/publish consistency; installs DataWatch for model meta; emits Event(PORTAL).
  - `watch_publish`: installs watches on publish nodes; emits Event(PUBLISH) when all publish nodes arrive.
  - `watch_resource`: watches resource nodes into cache.
  - `watch_service`: watches service hierarchy into cache.
- `election`: uses Kazoo Election to run leader function; handles reconnects.
- `start(is_client=False)`: starts ZK, watches service, and optionally publish.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/src/zk_mirror.rs`.
- Rust public API surface: `ZKMirror` with same watch APIs and cache.
- Data model mapping: `PublishMeta`, `ResourceSpec`, `ReplicaMeta`, `Event`.

**Implementation Steps (Detailed)**
1. Implement ZK cache and watch registration with thread-safe locks.
2. Port expected_loading logic (publish count and shard/replica overrides).
3. Port update_service and replica query helpers.
4. Implement leader election and reconnect behavior.
5. Provide Queue/Event mechanism for scheduler integration.

**Tests (Detailed)**
- Python tests: `monolith/agent_service/zk_mirror_test.py`
- Rust tests: use Fake ZK to validate watches, publish flow, and replica queries.

**Gaps / Notes**
- Requires a robust fake ZK in Rust to test watch behavior.

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

### `monolith/agent_service/zk_mirror_test.py`
<a id="monolith-agent-service-zk-mirror-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 229
- Purpose/role: End-to-end test of ZKMirror portal/publish/service/resource flows using FakeKazooClient.
- Key symbols/classes/functions: `ZKMirrorTest.test_crud`, `.test_zk_mirror`
- External dependencies: FakeZK, FakeTFServing, data_def types.
- Side effects: ZK node creation, event queue handling.

**Required Behavior (Detailed)**
- `test_crud`:
  - `ensure_path`, `exists`, `create`, `get/set`, `delete` operations.
  - Checks derived properties: `num_tce_shard`, `tce_replica_id`, `tce_shard_id`.
- `test_zk_mirror`:
  - `watch_portal` + `watch_resource`.
  - Portal event should be emitted for new ModelMeta.
  - Scheduler simulation publishes PublishMeta to all shards/replicas.
  - `expected_loading` selects correct PublishMeta for current shard.
  - `update_service` writes ReplicaMeta nodes; verify replica query APIs.
  - `report_resource` and `resources` should roundtrip ResourceSpec.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-serving/tests/zk_mirror.rs`.
- Rust public API surface: ZKMirror with fake ZK.

**Implementation Steps (Detailed)**
1. Port fake ZK + ZKMirror to Rust.
2. Implement scheduler simulation and verify queue events.
3. Validate replica query helpers and resource roundtrip.

**Tests (Detailed)**
- Python tests: this file
- Rust tests: direct parity tests.

**Gaps / Notes**
- Requires deterministic ordering of publish nodes and queue events.

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

### `monolith/base_runner.py`
<a id="monolith-base-runner-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 46
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

### `monolith/common/python/mem_profiling.py`
<a id="monolith-common-python-mem-profiling-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 51
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

### `monolith/core/__init__.py`
<a id="monolith-core-init-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 0
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

### `monolith/core/auto_checkpoint_feed_hook.py`
<a id="monolith-core-auto-checkpoint-feed-hook-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 376
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

### `monolith/core/base_embedding_host_call.py`
<a id="monolith-core-base-embedding-host-call-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 643
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

### `monolith/core/base_embedding_host_call_test.py`
<a id="monolith-core-base-embedding-host-call-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 77
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

### `monolith/core/base_embedding_task.py`
<a id="monolith-core-base-embedding-task-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 611
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

### `monolith/core/base_host_call.py`
<a id="monolith-core-base-host-call-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 145
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

### `monolith/core/base_layer.py`
<a id="monolith-core-base-layer-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 161
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

### `monolith/core/base_layer_test.py`
<a id="monolith-core-base-layer-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 41
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

### `monolith/core/base_model_params.py`
<a id="monolith-core-base-model-params-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 25
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

### `monolith/core/base_task.py`
<a id="monolith-core-base-task-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 95
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

### `monolith/core/base_tpu_test.py`
<a id="monolith-core-base-tpu-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 73
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

### `monolith/core/core_test_suite.py`
<a id="monolith-core-core-test-suite-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 35
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

### `monolith/core/dense.py`
<a id="monolith-core-dense-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 179
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

### `monolith/core/dense_test.py`
<a id="monolith-core-dense-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 108
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

### `monolith/core/feature.py`
<a id="monolith-core-feature-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 611
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

### `monolith/core/feature_test.py`
<a id="monolith-core-feature-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 178
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

### `monolith/core/host_call.py`
<a id="monolith-core-host-call-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 248
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

### `monolith/core/hyperparams.py`
<a id="monolith-core-hyperparams-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 439
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

### `monolith/core/hyperparams_test.py`
<a id="monolith-core-hyperparams-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 277
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

### `monolith/core/mixed_emb_op_comb_nws.py`
<a id="monolith-core-mixed-emb-op-comb-nws-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 421
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

### `monolith/core/model.py`
<a id="monolith-core-model-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 320
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

### `monolith/core/model_imports.py`
<a id="monolith-core-model-imports-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 104
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

### `monolith/core/model_registry.py`
<a id="monolith-core-model-registry-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 174
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

### `monolith/core/optimizers.py`
<a id="monolith-core-optimizers-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 25
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

### `monolith/core/py_utils.py`
<a id="monolith-core-py-utils-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 313
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

### `monolith/core/testing_utils.py`
<a id="monolith-core-testing-utils-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 203
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

### `monolith/core/tpu_variable.py`
<a id="monolith-core-tpu-variable-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 214
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

### `monolith/core/util.py`
<a id="monolith-core-util-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 269
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

### `monolith/core/util_test.py`
<a id="monolith-core-util-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 150
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

### `monolith/core/variance_scaling.py`
<a id="monolith-core-variance-scaling-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 188
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

### `monolith/gpu_runner.py`
<a id="monolith-gpu-runner-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 226
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

### `monolith/native_training/alert/alert_manager.py`
<a id="monolith-native-training-alert-alert-manager-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 31
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

### `monolith/native_training/alert/alert_manager_test.py`
<a id="monolith-native-training-alert-alert-manager-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 32
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

### `monolith/native_training/barrier_ops.py`
<a id="monolith-native-training-barrier-ops-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 158
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

### `monolith/native_training/barrier_ops_test.py`
<a id="monolith-native-training-barrier-ops-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 104
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

### `monolith/native_training/basic_restore_hook.py`
<a id="monolith-native-training-basic-restore-hook-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 72
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

### `monolith/native_training/basic_restore_hook_test.py`
<a id="monolith-native-training-basic-restore-hook-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 137
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

### `monolith/native_training/clip_ops.py`
<a id="monolith-native-training-clip-ops-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 80
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

### `monolith/native_training/clip_ops_test.py`
<a id="monolith-native-training-clip-ops-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 92
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

### `monolith/native_training/cluster_manager.py`
<a id="monolith-native-training-cluster-manager-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 184
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

### `monolith/native_training/cluster_manager_test.py`
<a id="monolith-native-training-cluster-manager-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 35
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

### `monolith/native_training/consul.py`
<a id="monolith-native-training-consul-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 149
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

### `monolith/native_training/consul_test.py`
<a id="monolith-native-training-consul-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 59
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

### `monolith/native_training/cpu_sync_training_test.py`
<a id="monolith-native-training-cpu-sync-training-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 360
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

### `monolith/native_training/cpu_training.py`
<a id="monolith-native-training-cpu-training-py"></a>

**Status:** TODO (manual review required)

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

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 226
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

### `monolith/native_training/cpu_training_test.py`
<a id="monolith-native-training-cpu-training-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 597
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

### `monolith/native_training/data/__init__.py`
<a id="monolith-native-training-data-init-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 20
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

### `monolith/native_training/data/data_ops_test.py`
<a id="monolith-native-training-data-data-ops-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 502
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

### `monolith/native_training/data/data_service_parquet_test.py`
<a id="monolith-native-training-data-data-service-parquet-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 145
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

### `monolith/native_training/data/data_service_test.py`
<a id="monolith-native-training-data-data-service-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 98
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

### `monolith/native_training/data/datasets.py`
<a id="monolith-native-training-data-datasets-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 1642
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

### `monolith/native_training/data/eager_mode_test.py`
<a id="monolith-native-training-data-eager-mode-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 186
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

### `monolith/native_training/data/extract_fid_test.py`
<a id="monolith-native-training-data-extract-fid-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 30
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

### `monolith/native_training/data/feature_list.py`
<a id="monolith-native-training-data-feature-list-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 409
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

### `monolith/native_training/data/feature_list_test.py`
<a id="monolith-native-training-data-feature-list-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 0
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

### `monolith/native_training/data/feature_utils.py`
<a id="monolith-native-training-data-feature-utils-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 1070
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

### `monolith/native_training/data/feature_utils_test.py`
<a id="monolith-native-training-data-feature-utils-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 1414
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

### `monolith/native_training/data/item_pool_hook.py`
<a id="monolith-native-training-data-item-pool-hook-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 109
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

### `monolith/native_training/data/item_pool_test.py`
<a id="monolith-native-training-data-item-pool-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 58
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

### `monolith/native_training/data/kafka_dataset_test.py`
<a id="monolith-native-training-data-kafka-dataset-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 239
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

### `monolith/native_training/data/multi_flow_test.py`
<a id="monolith-native-training-data-multi-flow-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 125
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

### `monolith/native_training/data/negative_gen_test.py`
<a id="monolith-native-training-data-negative-gen-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 253
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

### `monolith/native_training/data/parse_sparse_feature_test.py`
<a id="monolith-native-training-data-parse-sparse-feature-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 1833
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

### `monolith/native_training/data/parsers.py`
<a id="monolith-native-training-data-parsers-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 782
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

### `monolith/native_training/data/tf_example_to_example_test.py`
<a id="monolith-native-training-data-tf-example-to-example-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 183
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

### `monolith/native_training/data/training_instance/python/instance_dataset_op.py`
<a id="monolith-native-training-data-training-instance-python-instance-dataset-op-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 166
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

### `monolith/native_training/data/training_instance/python/instance_dataset_op_test_stdin.py`
<a id="monolith-native-training-data-training-instance-python-instance-dataset-op-test-stdin-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 58
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

### `monolith/native_training/data/training_instance/python/instance_negative_gen_dataset_op_test.py`
<a id="monolith-native-training-data-training-instance-python-instance-negative-gen-dataset-op-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 283
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

### `monolith/native_training/data/training_instance/python/parse_instance_ops.py`
<a id="monolith-native-training-data-training-instance-python-parse-instance-ops-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 245
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

### `monolith/native_training/data/training_instance/python/parse_instance_ops_test.py`
<a id="monolith-native-training-data-training-instance-python-parse-instance-ops-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 185
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

### `monolith/native_training/data/training_instance/python/parser_utils.py`
<a id="monolith-native-training-data-training-instance-python-parser-utils-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 85
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

### `monolith/native_training/data/training_instance/python/pb_datasource_ops.py`
<a id="monolith-native-training-data-training-instance-python-pb-datasource-ops-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 48
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

### `monolith/native_training/data/training_instance/python/test_data_utils.py`
<a id="monolith-native-training-data-training-instance-python-test-data-utils-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 15
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

### `monolith/native_training/data/transform/transforms.py`
<a id="monolith-native-training-data-transform-transforms-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 250
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

### `monolith/native_training/data/transform/transforms_test.py`
<a id="monolith-native-training-data-transform-transforms-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 70
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

### `monolith/native_training/data/transform_dataset_test.py`
<a id="monolith-native-training-data-transform-dataset-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 168
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

### `monolith/native_training/data/utils.py`
<a id="monolith-native-training-data-utils-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 55
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

### `monolith/native_training/debugging/debugging_client.py`
<a id="monolith-native-training-debugging-debugging-client-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 98
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

### `monolith/native_training/debugging/debugging_server.py`
<a id="monolith-native-training-debugging-debugging-server-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 217
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

### `monolith/native_training/demo.py`
<a id="monolith-native-training-demo-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 57
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

### `monolith/native_training/dense_reload_utils.py`
<a id="monolith-native-training-dense-reload-utils-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 457
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

### `monolith/native_training/dense_reload_utils_test.py`
<a id="monolith-native-training-dense-reload-utils-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 192
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

### `monolith/native_training/device_utils.py`
<a id="monolith-native-training-device-utils-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 231
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

### `monolith/native_training/device_utils_test.py`
<a id="monolith-native-training-device-utils-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 104
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

### `monolith/native_training/distribute/distributed_dataset.py`
<a id="monolith-native-training-distribute-distributed-dataset-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 81
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

### `monolith/native_training/distribute/distributed_dataset_test.py`
<a id="monolith-native-training-distribute-distributed-dataset-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 124
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

### `monolith/native_training/distribute/str_queue.py`
<a id="monolith-native-training-distribute-str-queue-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 114
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

### `monolith/native_training/distribute/str_queue_test.py`
<a id="monolith-native-training-distribute-str-queue-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 67
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

### `monolith/native_training/distributed_ps.py`
<a id="monolith-native-training-distributed-ps-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 2108
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

### `monolith/native_training/distributed_ps_benchmark.py`
<a id="monolith-native-training-distributed-ps-benchmark-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 168
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

### `monolith/native_training/distributed_ps_factory.py`
<a id="monolith-native-training-distributed-ps-factory-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 262
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

### `monolith/native_training/distributed_ps_factory_test.py`
<a id="monolith-native-training-distributed-ps-factory-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 87
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

### `monolith/native_training/distributed_ps_sync.py`
<a id="monolith-native-training-distributed-ps-sync-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 531
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

### `monolith/native_training/distributed_ps_sync_test.py`
<a id="monolith-native-training-distributed-ps-sync-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 109
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

### `monolith/native_training/distributed_ps_test.py`
<a id="monolith-native-training-distributed-ps-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 979
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

### `monolith/native_training/distributed_serving_ops.py`
<a id="monolith-native-training-distributed-serving-ops-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 160
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

### `monolith/native_training/distributed_serving_ops_test.py`
<a id="monolith-native-training-distributed-serving-ops-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 142
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

### `monolith/native_training/distribution_ops.py`
<a id="monolith-native-training-distribution-ops-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 889
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

### `monolith/native_training/distribution_ops_benchmark.py`
<a id="monolith-native-training-distribution-ops-benchmark-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 118
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

### `monolith/native_training/distribution_ops_fused_benchmark.py`
<a id="monolith-native-training-distribution-ops-fused-benchmark-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 61
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

### `monolith/native_training/distribution_ops_fused_test.py`
<a id="monolith-native-training-distribution-ops-fused-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 148
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

### `monolith/native_training/distribution_ops_test.py`
<a id="monolith-native-training-distribution-ops-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 536
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

### `monolith/native_training/distribution_utils.py`
<a id="monolith-native-training-distribution-utils-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 443
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

### `monolith/native_training/embedding_combiners.py`
<a id="monolith-native-training-embedding-combiners-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 102
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

### `monolith/native_training/embedding_combiners_test.py`
<a id="monolith-native-training-embedding-combiners-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 47
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

### `monolith/native_training/entry.py`
<a id="monolith-native-training-entry-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 630
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

### `monolith/native_training/entry_test.py`
<a id="monolith-native-training-entry-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 84
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

### `monolith/native_training/env_utils.py`
<a id="monolith-native-training-env-utils-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 32
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

### `monolith/native_training/env_utils_test.py`
<a id="monolith-native-training-env-utils-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 23
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

### `monolith/native_training/estimator.py`
<a id="monolith-native-training-estimator-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 667
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

### `monolith/native_training/estimator_dist_test.py`
<a id="monolith-native-training-estimator-dist-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 166
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

### `monolith/native_training/estimator_mode_test.py`
<a id="monolith-native-training-estimator-mode-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 417
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

### `monolith/native_training/estimator_test.py`
<a id="monolith-native-training-estimator-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 112
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

### `monolith/native_training/feature.py`
<a id="monolith-native-training-feature-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 663
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

### `monolith/native_training/feature_test.py`
<a id="monolith-native-training-feature-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 266
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

### `monolith/native_training/feature_utils.py`
<a id="monolith-native-training-feature-utils-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 419
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

### `monolith/native_training/feature_utils_test.py`
<a id="monolith-native-training-feature-utils-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 144
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

### `monolith/native_training/file_ops.py`
<a id="monolith-native-training-file-ops-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 51
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

### `monolith/native_training/file_ops_test.py`
<a id="monolith-native-training-file-ops-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 56
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

### `monolith/native_training/fused_embedding_to_layout_test.py`
<a id="monolith-native-training-fused-embedding-to-layout-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 1333
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

### `monolith/native_training/gen_seq_mask.py`
<a id="monolith-native-training-gen-seq-mask-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 26
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

### `monolith/native_training/gen_seq_mask_test.py`
<a id="monolith-native-training-gen-seq-mask-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 42
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

### `monolith/native_training/gflags_utils.py`
<a id="monolith-native-training-gflags-utils-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 282
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

### `monolith/native_training/gflags_utils_test.py`
<a id="monolith-native-training-gflags-utils-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 217
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

### `monolith/native_training/graph_meta.py`
<a id="monolith-native-training-graph-meta-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 30
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

### `monolith/native_training/graph_utils.py`
<a id="monolith-native-training-graph-utils-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 26
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

### `monolith/native_training/hash_filter_ops.py`
<a id="monolith-native-training-hash-filter-ops-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 326
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

### `monolith/native_training/hash_filter_ops_test.py`
<a id="monolith-native-training-hash-filter-ops-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 228
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

### `monolith/native_training/hash_table_ops.py`
<a id="monolith-native-training-hash-table-ops-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 738
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

### `monolith/native_training/hash_table_ops_benchmark.py`
<a id="monolith-native-training-hash-table-ops-benchmark-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 148
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

### `monolith/native_training/hash_table_ops_test.py`
<a id="monolith-native-training-hash-table-ops-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 1200
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

### `monolith/native_training/hash_table_utils.py`
<a id="monolith-native-training-hash-table-utils-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 50
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

### `monolith/native_training/hash_table_utils_test.py`
<a id="monolith-native-training-hash-table-utils-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 45
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

### `monolith/native_training/hooks/ckpt_hooks.py`
<a id="monolith-native-training-hooks-ckpt-hooks-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 193
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

### `monolith/native_training/hooks/ckpt_hooks_test.py`
<a id="monolith-native-training-hooks-ckpt-hooks-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 181
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

### `monolith/native_training/hooks/ckpt_info.py`
<a id="monolith-native-training-hooks-ckpt-info-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 98
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

### `monolith/native_training/hooks/ckpt_info_test.py`
<a id="monolith-native-training-hooks-ckpt-info-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 45
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

### `monolith/native_training/hooks/controller_hooks.py`
<a id="monolith-native-training-hooks-controller-hooks-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 170
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

### `monolith/native_training/hooks/controller_hooks_test.py`
<a id="monolith-native-training-hooks-controller-hooks-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 82
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

### `monolith/native_training/hooks/feature_engineering_hooks.py`
<a id="monolith-native-training-hooks-feature-engineering-hooks-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 99
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

### `monolith/native_training/hooks/hook_utils.py`
<a id="monolith-native-training-hooks-hook-utils-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 41
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

### `monolith/native_training/hooks/hook_utils_test.py`
<a id="monolith-native-training-hooks-hook-utils-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 35
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

### `monolith/native_training/hooks/ps_check_hooks.py`
<a id="monolith-native-training-hooks-ps-check-hooks-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 97
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

### `monolith/native_training/hooks/ps_check_hooks_test.py`
<a id="monolith-native-training-hooks-ps-check-hooks-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 112
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

### `monolith/native_training/hooks/server/client_lib.py`
<a id="monolith-native-training-hooks-server-client-lib-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 30
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

### `monolith/native_training/hooks/server/constants.py`
<a id="monolith-native-training-hooks-server-constants-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 15
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

### `monolith/native_training/hooks/server/server_lib.py`
<a id="monolith-native-training-hooks-server-server-lib-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 95
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

### `monolith/native_training/hooks/server/server_lib_test.py`
<a id="monolith-native-training-hooks-server-server-lib-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 54
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

### `monolith/native_training/hooks/session_hooks.py`
<a id="monolith-native-training-hooks-session-hooks-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 44
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

### `monolith/native_training/hooks/session_hooks_test.py`
<a id="monolith-native-training-hooks-session-hooks-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 33
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

### `monolith/native_training/hvd_lib.py`
<a id="monolith-native-training-hvd-lib-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 65
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

### `monolith/native_training/input.py`
<a id="monolith-native-training-input-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 45
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

### `monolith/native_training/layers/__init__.py`
<a id="monolith-native-training-layers-init-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 46
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

### `monolith/native_training/layers/add_bias.py`
<a id="monolith-native-training-layers-add-bias-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 110
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

### `monolith/native_training/layers/add_bias_test.py`
<a id="monolith-native-training-layers-add-bias-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 65
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

### `monolith/native_training/layers/advanced_activations.py`
<a id="monolith-native-training-layers-advanced-activations-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 217
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

### `monolith/native_training/layers/advanced_activations_test.py`
<a id="monolith-native-training-layers-advanced-activations-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 84
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

### `monolith/native_training/layers/agru.py`
<a id="monolith-native-training-layers-agru-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 295
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

### `monolith/native_training/layers/agru_test.py`
<a id="monolith-native-training-layers-agru-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 112
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

### `monolith/native_training/layers/dense.py`
<a id="monolith-native-training-layers-dense-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 307
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

### `monolith/native_training/layers/dense_test.py`
<a id="monolith-native-training-layers-dense-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 147
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

### `monolith/native_training/layers/feature_cross.py`
<a id="monolith-native-training-layers-feature-cross-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 805
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

### `monolith/native_training/layers/feature_cross_test.py`
<a id="monolith-native-training-layers-feature-cross-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 286
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

### `monolith/native_training/layers/feature_seq.py`
<a id="monolith-native-training-layers-feature-seq-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 361
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

### `monolith/native_training/layers/feature_seq_test.py`
<a id="monolith-native-training-layers-feature-seq-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 126
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

### `monolith/native_training/layers/feature_trans.py`
<a id="monolith-native-training-layers-feature-trans-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 340
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

### `monolith/native_training/layers/feature_trans_test.py`
<a id="monolith-native-training-layers-feature-trans-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 140
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

### `monolith/native_training/layers/layer_ops.py`
<a id="monolith-native-training-layers-layer-ops-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 131
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

### `monolith/native_training/layers/layer_ops_test.py`
<a id="monolith-native-training-layers-layer-ops-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 232
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

### `monolith/native_training/layers/lhuc.py`
<a id="monolith-native-training-layers-lhuc-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 296
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

### `monolith/native_training/layers/lhuc_test.py`
<a id="monolith-native-training-layers-lhuc-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 73
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

### `monolith/native_training/layers/logit_correction.py`
<a id="monolith-native-training-layers-logit-correction-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 88
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

### `monolith/native_training/layers/logit_correction_test.py`
<a id="monolith-native-training-layers-logit-correction-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 65
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

### `monolith/native_training/layers/mlp.py`
<a id="monolith-native-training-layers-mlp-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 211
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

### `monolith/native_training/layers/mlp_test.py`
<a id="monolith-native-training-layers-mlp-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 78
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

### `monolith/native_training/layers/multi_task.py`
<a id="monolith-native-training-layers-multi-task-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 448
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

### `monolith/native_training/layers/multi_task_test.py`
<a id="monolith-native-training-layers-multi-task-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 128
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

### `monolith/native_training/layers/norms.py`
<a id="monolith-native-training-layers-norms-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 343
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

### `monolith/native_training/layers/norms_test.py`
<a id="monolith-native-training-layers-norms-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 84
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

### `monolith/native_training/layers/pooling.py`
<a id="monolith-native-training-layers-pooling-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 101
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

### `monolith/native_training/layers/pooling_test.py`
<a id="monolith-native-training-layers-pooling-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 141
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

### `monolith/native_training/layers/sparse_nas.py`
<a id="monolith-native-training-layers-sparse-nas-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 31
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

### `monolith/native_training/layers/sparse_nas_test.py`
<a id="monolith-native-training-layers-sparse-nas-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 23
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

### `monolith/native_training/layers/utils.py`
<a id="monolith-native-training-layers-utils-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 159
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

### `monolith/native_training/learning_rate_functions.py`
<a id="monolith-native-training-learning-rate-functions-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 112
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

### `monolith/native_training/learning_rate_functions_test.py`
<a id="monolith-native-training-learning-rate-functions-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 76
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

### `monolith/native_training/logging_ops.py`
<a id="monolith-native-training-logging-ops-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 56
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

### `monolith/native_training/logging_ops_test.py`
<a id="monolith-native-training-logging-ops-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 57
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

### `monolith/native_training/losses/batch_softmax_loss.py`
<a id="monolith-native-training-losses-batch-softmax-loss-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 57
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

### `monolith/native_training/losses/batch_softmax_loss_test.py`
<a id="monolith-native-training-losses-batch-softmax-loss-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 35
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

### `monolith/native_training/losses/inbatch_auc_loss.py`
<a id="monolith-native-training-losses-inbatch-auc-loss-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 41
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

### `monolith/native_training/losses/inbatch_auc_loss_test.py`
<a id="monolith-native-training-losses-inbatch-auc-loss-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 71
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

### `monolith/native_training/losses/ltr_losses.py`
<a id="monolith-native-training-losses-ltr-losses-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 1233
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

### `monolith/native_training/metric/cli.py`
<a id="monolith-native-training-metric-cli-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 28
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

### `monolith/native_training/metric/deep_insight_ops.py`
<a id="monolith-native-training-metric-deep-insight-ops-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 134
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

### `monolith/native_training/metric/deep_insight_ops_test.py`
<a id="monolith-native-training-metric-deep-insight-ops-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 33
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

### `monolith/native_training/metric/exit_hook.py`
<a id="monolith-native-training-metric-exit-hook-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 48
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

### `monolith/native_training/metric/kafka_utils.py`
<a id="monolith-native-training-metric-kafka-utils-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 119
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

### `monolith/native_training/metric/metric_hook.py`
<a id="monolith-native-training-metric-metric-hook-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 563
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

### `monolith/native_training/metric/metric_hook_test.py`
<a id="monolith-native-training-metric-metric-hook-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 189
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

### `monolith/native_training/metric/utils.py`
<a id="monolith-native-training-metric-utils-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 104
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

### `monolith/native_training/metric/utils_test.py`
<a id="monolith-native-training-metric-utils-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 50
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

### `monolith/native_training/mlp_utils.py`
<a id="monolith-native-training-mlp-utils-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 444
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

### `monolith/native_training/model.py`
<a id="monolith-native-training-model-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 182
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

### `monolith/native_training/model_comp_test.py`
<a id="monolith-native-training-model-comp-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 183
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

### `monolith/native_training/model_dump/dump_utils.py`
<a id="monolith-native-training-model-dump-dump-utils-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 757
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

### `monolith/native_training/model_dump/graph_utils.py`
<a id="monolith-native-training-model-dump-graph-utils-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 845
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

### `monolith/native_training/model_dump/graph_utils_test.py`
<a id="monolith-native-training-model-dump-graph-utils-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 86
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

### `monolith/native_training/model_export/__init__.py`
<a id="monolith-native-training-model-export-init-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 22
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

### `monolith/native_training/model_export/data_gen_utils.py`
<a id="monolith-native-training-model-export-data-gen-utils-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 732
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

### `monolith/native_training/model_export/data_gen_utils_test.py`
<a id="monolith-native-training-model-export-data-gen-utils-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 0
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

### `monolith/native_training/model_export/demo_export.py`
<a id="monolith-native-training-model-export-demo-export-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 100
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

### `monolith/native_training/model_export/demo_export_test.py`
<a id="monolith-native-training-model-export-demo-export-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 48
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

### `monolith/native_training/model_export/demo_predictor.py`
<a id="monolith-native-training-model-export-demo-predictor-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 110
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

### `monolith/native_training/model_export/demo_predictor_client.py`
<a id="monolith-native-training-model-export-demo-predictor-client-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 93
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

### `monolith/native_training/model_export/export_context.py`
<a id="monolith-native-training-model-export-export-context-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 141
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

### `monolith/native_training/model_export/export_hooks.py`
<a id="monolith-native-training-model-export-export-hooks-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 137
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

### `monolith/native_training/model_export/export_hooks_test.py`
<a id="monolith-native-training-model-export-export-hooks-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 141
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

### `monolith/native_training/model_export/export_state_utils.py`
<a id="monolith-native-training-model-export-export-state-utils-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 46
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

### `monolith/native_training/model_export/export_state_utils_test.py`
<a id="monolith-native-training-model-export-export-state-utils-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 36
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

### `monolith/native_training/model_export/export_utils.py`
<a id="monolith-native-training-model-export-export-utils-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 98
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

### `monolith/native_training/model_export/export_utils_test.py`
<a id="monolith-native-training-model-export-export-utils-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 43
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

### `monolith/native_training/model_export/saved_model_exporters.py`
<a id="monolith-native-training-model-export-saved-model-exporters-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 739
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

### `monolith/native_training/model_export/saved_model_exporters_test.py`
<a id="monolith-native-training-model-export-saved-model-exporters-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 153
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

### `monolith/native_training/model_export/saved_model_visulizer.py`
<a id="monolith-native-training-model-export-saved-model-visulizer-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 89
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

### `monolith/native_training/model_export/warmup_data_decoder.py`
<a id="monolith-native-training-model-export-warmup-data-decoder-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 55
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

### `monolith/native_training/model_export/warmup_data_gen.py`
<a id="monolith-native-training-model-export-warmup-data-gen-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 253
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

### `monolith/native_training/model_export/warmup_example_batch.py`
<a id="monolith-native-training-model-export-warmup-example-batch-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 57
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

### `monolith/native_training/monolith_export.py`
<a id="monolith-native-training-monolith-export-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 18
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

### `monolith/native_training/multi_hash_table_ops.py`
<a id="monolith-native-training-multi-hash-table-ops-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 695
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

### `monolith/native_training/multi_hash_table_ops_test.py`
<a id="monolith-native-training-multi-hash-table-ops-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 249
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

### `monolith/native_training/multi_type_hash_table.py`
<a id="monolith-native-training-multi-type-hash-table-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 435
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

### `monolith/native_training/multi_type_hash_table_test.py`
<a id="monolith-native-training-multi-type-hash-table-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 326
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

### `monolith/native_training/native_model.py`
<a id="monolith-native-training-native-model-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 1109
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

### `monolith/native_training/native_task.py`
<a id="monolith-native-training-native-task-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 213
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

### `monolith/native_training/native_task_context.py`
<a id="monolith-native-training-native-task-context-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 58
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

### `monolith/native_training/nested_tensors.py`
<a id="monolith-native-training-nested-tensors-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 110
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

### `monolith/native_training/nested_tensors_test.py`
<a id="monolith-native-training-nested-tensors-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 57
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

### `monolith/native_training/net_utils.py`
<a id="monolith-native-training-net-utils-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 133
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

### `monolith/native_training/net_utils_test.py`
<a id="monolith-native-training-net-utils-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 94
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

### `monolith/native_training/optimizers/adamom.py`
<a id="monolith-native-training-optimizers-adamom-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 68
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

### `monolith/native_training/optimizers/adamom_test.py`
<a id="monolith-native-training-optimizers-adamom-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 57
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

### `monolith/native_training/optimizers/rmsprop.py`
<a id="monolith-native-training-optimizers-rmsprop-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 102
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

### `monolith/native_training/optimizers/rmsprop_test.py`
<a id="monolith-native-training-optimizers-rmsprop-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 77
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

### `monolith/native_training/optimizers/rmspropv2_test.py`
<a id="monolith-native-training-optimizers-rmspropv2-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 112
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

### `monolith/native_training/optimizers/shampoo.py`
<a id="monolith-native-training-optimizers-shampoo-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 207
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

### `monolith/native_training/prefetch_queue.py`
<a id="monolith-native-training-prefetch-queue-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 379
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

### `monolith/native_training/prefetch_queue_test.py`
<a id="monolith-native-training-prefetch-queue-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 305
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

### `monolith/native_training/ps_benchmark.py`
<a id="monolith-native-training-ps-benchmark-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 273
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

### `monolith/native_training/ps_benchmark_test.py`
<a id="monolith-native-training-ps-benchmark-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 57
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

### `monolith/native_training/ragged_utils.py`
<a id="monolith-native-training-ragged-utils-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 29
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

### `monolith/native_training/ragged_utils_test.py`
<a id="monolith-native-training-ragged-utils-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 32
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

### `monolith/native_training/remote_predict_ops.py`
<a id="monolith-native-training-remote-predict-ops-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 0
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

### `monolith/native_training/restore_test.py`
<a id="monolith-native-training-restore-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 240
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

### `monolith/native_training/runner_utils.py`
<a id="monolith-native-training-runner-utils-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 396
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

### `monolith/native_training/runner_utils_test.py`
<a id="monolith-native-training-runner-utils-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 108
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

### `monolith/native_training/runtime/ops/gen_monolith_ops.py`
<a id="monolith-native-training-runtime-ops-gen-monolith-ops-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 23
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

### `monolith/native_training/save_utils.py`
<a id="monolith-native-training-save-utils-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 1309
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

### `monolith/native_training/save_utils_test.py`
<a id="monolith-native-training-save-utils-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 1740
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

### `monolith/native_training/service_discovery.py`
<a id="monolith-native-training-service-discovery-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 481
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

### `monolith/native_training/service_discovery_test.py`
<a id="monolith-native-training-service-discovery-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 407
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

### `monolith/native_training/serving_ps_test.py`
<a id="monolith-native-training-serving-ps-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 231
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

### `monolith/native_training/session_run_hooks.py`
<a id="monolith-native-training-session-run-hooks-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 171
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

### `monolith/native_training/session_run_hooks_test.py`
<a id="monolith-native-training-session-run-hooks-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 144
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

### `monolith/native_training/signal_utils.py`
<a id="monolith-native-training-signal-utils-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 37
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

### `monolith/native_training/signal_utils_test.py`
<a id="monolith-native-training-signal-utils-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 30
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

### `monolith/native_training/static_reshape_op.py`
<a id="monolith-native-training-static-reshape-op-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 58
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

### `monolith/native_training/static_reshape_op_test.py`
<a id="monolith-native-training-static-reshape-op-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 79
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

### `monolith/native_training/summary/summary_ops.py`
<a id="monolith-native-training-summary-summary-ops-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 78
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

### `monolith/native_training/summary/summary_ops_test.py`
<a id="monolith-native-training-summary-summary-ops-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 122
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

### `monolith/native_training/summary/utils.py`
<a id="monolith-native-training-summary-utils-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 114
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

### `monolith/native_training/summary/utils_test.py`
<a id="monolith-native-training-summary-utils-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 43
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

### `monolith/native_training/sync_hooks.py`
<a id="monolith-native-training-sync-hooks-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 176
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

### `monolith/native_training/sync_hooks_test.py`
<a id="monolith-native-training-sync-hooks-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 119
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

### `monolith/native_training/sync_training_hooks.py`
<a id="monolith-native-training-sync-training-hooks-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 355
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

### `monolith/native_training/sync_training_hooks_test.py`
<a id="monolith-native-training-sync-training-hooks-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 92
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

### `monolith/native_training/tensor_utils.py`
<a id="monolith-native-training-tensor-utils-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 162
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

### `monolith/native_training/tensor_utils_test.py`
<a id="monolith-native-training-tensor-utils-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 175
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

### `monolith/native_training/test_utils.py`
<a id="monolith-native-training-test-utils-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 65
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

### `monolith/native_training/touched_key_set_ops.py`
<a id="monolith-native-training-touched-key-set-ops-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 61
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

### `monolith/native_training/touched_key_set_ops_test.py`
<a id="monolith-native-training-touched-key-set-ops-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 51
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

### `monolith/native_training/utils.py`
<a id="monolith-native-training-utils-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 320
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

### `monolith/native_training/utils_test.py`
<a id="monolith-native-training-utils-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 70
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

### `monolith/native_training/variables.py`
<a id="monolith-native-training-variables-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 147
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

### `monolith/native_training/variables_test.py`
<a id="monolith-native-training-variables-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 89
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

### `monolith/native_training/yarn_runtime.py`
<a id="monolith-native-training-yarn-runtime-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 127
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

### `monolith/native_training/yarn_runtime_test.py`
<a id="monolith-native-training-yarn-runtime-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 133
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

### `monolith/native_training/zk_utils.py`
<a id="monolith-native-training-zk-utils-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 96
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

### `monolith/path_utils.py`
<a id="monolith-path-utils-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 47
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

### `monolith/tpu_runner.py`
<a id="monolith-tpu-runner-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 429
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

### `monolith/utils.py`
<a id="monolith-utils-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 81
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

### `monolith/utils_test.py`
<a id="monolith-utils-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 65
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