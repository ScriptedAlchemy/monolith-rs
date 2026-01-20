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

**Status:** IN PROGRESS (manual)

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

**Status:** IN PROGRESS (manual)

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
| [`monolith/__init__.py`](#monolith-init-py) | 55 | IN PROGRESS | monolith-rs/crates/monolith-core | |
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
| [`monolith/base_runner.py`](#monolith-base-runner-py) | 46 | IN PROGRESS | monolith-rs/crates/monolith-training/src | |
| [`monolith/common/python/mem_profiling.py`](#monolith-common-python-mem-profiling-py) | 51 | IN PROGRESS | monolith-rs/crates/monolith-training/src | |
| [`monolith/core/__init__.py`](#monolith-core-init-py) | 0 | IN PROGRESS | monolith-rs/crates/monolith-core/src | |
| [`monolith/core/auto_checkpoint_feed_hook.py`](#monolith-core-auto-checkpoint-feed-hook-py) | 376 | IN PROGRESS | monolith-rs/crates/monolith-tf/src |  |
| [`monolith/core/base_embedding_host_call.py`](#monolith-core-base-embedding-host-call-py) | 643 | IN PROGRESS | monolith-rs/crates/monolith-core/src | |
| [`monolith/core/base_embedding_host_call_test.py`](#monolith-core-base-embedding-host-call-test-py) | 77 | IN PROGRESS | monolith-rs/crates/monolith-core/tests | |
| [`monolith/core/base_embedding_task.py`](#monolith-core-base-embedding-task-py) | 611 | IN PROGRESS | monolith-rs/crates/monolith-core/src | |
| [`monolith/core/base_host_call.py`](#monolith-core-base-host-call-py) | 145 | IN PROGRESS | monolith-rs/crates/monolith-core/src | |
| [`monolith/core/base_layer.py`](#monolith-core-base-layer-py) | 161 | IN PROGRESS | monolith-rs/crates/monolith-core/src | |
| [`monolith/core/base_layer_test.py`](#monolith-core-base-layer-test-py) | 41 | IN PROGRESS | monolith-rs/crates/monolith-core/tests | |
| [`monolith/core/base_model_params.py`](#monolith-core-base-model-params-py) | 25 | IN PROGRESS | monolith-rs/crates/monolith-core/src | |
| [`monolith/core/base_task.py`](#monolith-core-base-task-py) | 95 | IN PROGRESS | monolith-rs/crates/monolith-core/src | |
| [`monolith/core/base_tpu_test.py`](#monolith-core-base-tpu-test-py) | 73 | IN PROGRESS | monolith-rs/crates/monolith-training/tests | |
| [`monolith/core/core_test_suite.py`](#monolith-core-core-test-suite-py) | 35 | IN PROGRESS | monolith-rs/crates/monolith-core/tests | |
| [`monolith/core/dense.py`](#monolith-core-dense-py) | 179 | IN PROGRESS | monolith-rs/crates/monolith-layers/src | |
| [`monolith/core/dense_test.py`](#monolith-core-dense-test-py) | 108 | IN PROGRESS | monolith-rs/crates/monolith-layers/tests | |
| [`monolith/core/feature.py`](#monolith-core-feature-py) | 611 | IN PROGRESS | monolith-rs/crates/monolith-core/src/feature.rs |  |
| [`monolith/core/feature_test.py`](#monolith-core-feature-test-py) | 178 | IN PROGRESS | monolith-rs/crates/monolith-core/tests |  |
| [`monolith/core/host_call.py`](#monolith-core-host-call-py) | 248 | IN PROGRESS | monolith-rs/crates/monolith-core/src | |
| [`monolith/core/hyperparams.py`](#monolith-core-hyperparams-py) | 439 | IN PROGRESS | monolith-rs/crates/monolith-core/src/hyperparams.rs |  |
| [`monolith/core/hyperparams_test.py`](#monolith-core-hyperparams-test-py) | 277 | IN PROGRESS | monolith-rs/crates/monolith-core/tests |  |
| [`monolith/core/mixed_emb_op_comb_nws.py`](#monolith-core-mixed-emb-op-comb-nws-py) | 421 | IN PROGRESS | monolith-rs/crates/monolith-layers/src |  |
| [`monolith/core/model.py`](#monolith-core-model-py) | 320 | IN PROGRESS | monolith-rs/crates/monolith-training/src |  |
| [`monolith/core/model_imports.py`](#monolith-core-model-imports-py) | 104 | IN PROGRESS | monolith-rs/crates/monolith-core/src | |
| [`monolith/core/model_registry.py`](#monolith-core-model-registry-py) | 174 | IN PROGRESS | monolith-rs/crates/monolith-core/src | |
| [`monolith/core/optimizers.py`](#monolith-core-optimizers-py) | 25 | IN PROGRESS | monolith-rs/crates/monolith-optimizer/src | |
| [`monolith/core/py_utils.py`](#monolith-core-py-utils-py) | 313 | IN PROGRESS | monolith-rs/crates/monolith-core/src | |
| [`monolith/core/testing_utils.py`](#monolith-core-testing-utils-py) | 203 | IN PROGRESS | monolith-rs/crates/monolith-core/tests | |
| [`monolith/core/tpu_variable.py`](#monolith-core-tpu-variable-py) | 214 | IN PROGRESS | monolith-rs/crates/monolith-tf/src | |
| [`monolith/core/util.py`](#monolith-core-util-py) | 269 | IN PROGRESS | monolith-rs/crates/monolith-core/src | |
| [`monolith/core/util_test.py`](#monolith-core-util-test-py) | 149 | IN PROGRESS | monolith-rs/crates/monolith-core/tests | |
| [`monolith/core/variance_scaling.py`](#monolith-core-variance-scaling-py) | 188 | IN PROGRESS | monolith-rs/crates/monolith-core/src | |
| [`monolith/gpu_runner.py`](#monolith-gpu-runner-py) | 226 | IN PROGRESS | monolith-rs/crates/monolith-training/src | |
| [`monolith/native_training/alert/alert_manager.py`](#monolith-native-training-alert-alert-manager-py) | 31 | IN PROGRESS | monolith-rs/crates/monolith-training/src |  |
| [`monolith/native_training/alert/alert_manager_test.py`](#monolith-native-training-alert-alert-manager-test-py) | 32 | IN PROGRESS | monolith-rs/crates/monolith-training/tests |  |
| [`monolith/native_training/barrier_ops.py`](#monolith-native-training-barrier-ops-py) | 158 | IN PROGRESS | monolith-rs/crates/monolith-training/src/barrier.rs |  |
| [`monolith/native_training/barrier_ops_test.py`](#monolith-native-training-barrier-ops-test-py) | 104 | IN PROGRESS | monolith-rs/crates/monolith-training/tests |  |
| [`monolith/native_training/basic_restore_hook.py`](#monolith-native-training-basic-restore-hook-py) | 72 | IN PROGRESS | monolith-rs/crates/monolith-training/src |  |
| [`monolith/native_training/basic_restore_hook_test.py`](#monolith-native-training-basic-restore-hook-test-py) | 137 | IN PROGRESS | monolith-rs/crates/monolith-training/tests |  |
| [`monolith/native_training/clip_ops.py`](#monolith-native-training-clip-ops-py) | 80 | IN PROGRESS | monolith-rs/crates/monolith-optimizer/src |  |
| [`monolith/native_training/clip_ops_test.py`](#monolith-native-training-clip-ops-test-py) | 92 | IN PROGRESS | monolith-rs/crates/monolith-optimizer/tests |  |
| [`monolith/native_training/cluster_manager.py`](#monolith-native-training-cluster-manager-py) | 184 | IN PROGRESS | monolith-rs/crates/monolith-training/src/distributed.rs |  |
| [`monolith/native_training/cluster_manager_test.py`](#monolith-native-training-cluster-manager-test-py) | 35 | IN PROGRESS | monolith-rs/crates/monolith-training/tests |  |
| [`monolith/native_training/consul.py`](#monolith-native-training-consul-py) | 149 | IN PROGRESS | monolith-rs/crates/monolith-training/src/discovery.rs |  |
| [`monolith/native_training/consul_test.py`](#monolith-native-training-consul-test-py) | 59 | IN PROGRESS | monolith-rs/crates/monolith-training/tests |  |
| [`monolith/native_training/cpu_sync_training_test.py`](#monolith-native-training-cpu-sync-training-test-py) | 360 | IN PROGRESS | monolith-rs/crates/monolith-training/tests |  |
| [`monolith/native_training/cpu_training.py`](#monolith-native-training-cpu-training-py) | 2449 | TODO | TODO (manual) |  |
| [`monolith/native_training/cpu_training_distributed_test_binary.py`](#monolith-native-training-cpu-training-distributed-test-binary-py) | 226 | IN PROGRESS | monolith-rs/crates/monolith-training/tests |  |
| [`monolith/native_training/cpu_training_test.py`](#monolith-native-training-cpu-training-test-py) | 597 | IN PROGRESS | monolith-rs/crates/monolith-training/tests |  |
| [`monolith/native_training/data/__init__.py`](#monolith-native-training-data-init-py) | 20 | IN PROGRESS | monolith-rs/crates/monolith-data/src |  |
| [`monolith/native_training/data/data_ops_test.py`](#monolith-native-training-data-data-ops-test-py) | 502 | IN PROGRESS | monolith-rs/crates/monolith-data/tests |  |
| [`monolith/native_training/data/data_service_parquet_test.py`](#monolith-native-training-data-data-service-parquet-test-py) | 145 | IN PROGRESS | monolith-rs/crates/monolith-data/tests |  |
| [`monolith/native_training/data/data_service_test.py`](#monolith-native-training-data-data-service-test-py) | 98 | IN PROGRESS | monolith-rs/crates/monolith-data/tests |  |
| [`monolith/native_training/data/datasets.py`](#monolith-native-training-data-datasets-py) | 1642 | IN PROGRESS | monolith-rs/crates/monolith-data/src |  |
| [`monolith/native_training/data/eager_mode_test.py`](#monolith-native-training-data-eager-mode-test-py) | 186 | IN PROGRESS | monolith-rs/crates/monolith-data/tests |  |
| [`monolith/native_training/data/extract_fid_test.py`](#monolith-native-training-data-extract-fid-test-py) | 30 | IN PROGRESS | monolith-rs/crates/monolith-data/tests |  |
| [`monolith/native_training/data/feature_list.py`](#monolith-native-training-data-feature-list-py) | 409 | IN PROGRESS | monolith-rs/crates/monolith-data/src/feature_list.rs |  |
| [`monolith/native_training/data/feature_list_test.py`](#monolith-native-training-data-feature-list-test-py) | 0 | IN PROGRESS | monolith-rs/crates/monolith-data/tests |  |
| [`monolith/native_training/data/feature_utils.py`](#monolith-native-training-data-feature-utils-py) | 1070 | IN PROGRESS | monolith-rs/crates/monolith-data/src |  |
| [`monolith/native_training/data/feature_utils_test.py`](#monolith-native-training-data-feature-utils-test-py) | 1414 | IN PROGRESS | monolith-rs/crates/monolith-data/tests |  |
| [`monolith/native_training/data/item_pool_hook.py`](#monolith-native-training-data-item-pool-hook-py) | 109 | IN PROGRESS | monolith-rs/crates/monolith-data/src |  |
| [`monolith/native_training/data/item_pool_test.py`](#monolith-native-training-data-item-pool-test-py) | 58 | IN PROGRESS | monolith-rs/crates/monolith-data/tests |  |
| [`monolith/native_training/data/kafka_dataset_test.py`](#monolith-native-training-data-kafka-dataset-test-py) | 239 | IN PROGRESS | monolith-rs/crates/monolith-data/tests |  |
| [`monolith/native_training/data/multi_flow_test.py`](#monolith-native-training-data-multi-flow-test-py) | 125 | IN PROGRESS | monolith-rs/crates/monolith-data/tests |  |
| [`monolith/native_training/data/negative_gen_test.py`](#monolith-native-training-data-negative-gen-test-py) | 253 | IN PROGRESS | monolith-rs/crates/monolith-data/tests |  |
| [`monolith/native_training/data/parse_sparse_feature_test.py`](#monolith-native-training-data-parse-sparse-feature-test-py) | 1833 | IN PROGRESS | monolith-rs/crates/monolith-data/tests |  |
| [`monolith/native_training/data/parsers.py`](#monolith-native-training-data-parsers-py) | 782 | IN PROGRESS | monolith-rs/crates/monolith-data/src |  |
| [`monolith/native_training/data/tf_example_to_example_test.py`](#monolith-native-training-data-tf-example-to-example-test-py) | 183 | IN PROGRESS | monolith-rs/crates/monolith-data/tests |  |
| [`monolith/native_training/data/training_instance/python/instance_dataset_op.py`](#monolith-native-training-data-training-instance-python-instance-dataset-op-py) | 166 | IN PROGRESS | monolith-rs/crates/monolith-data/src |  |
| [`monolith/native_training/data/training_instance/python/instance_dataset_op_test_stdin.py`](#monolith-native-training-data-training-instance-python-instance-dataset-op-test-stdin-py) | 58 | IN PROGRESS | monolith-rs/crates/monolith-data/tests |  |
| [`monolith/native_training/data/training_instance/python/instance_negative_gen_dataset_op_test.py`](#monolith-native-training-data-training-instance-python-instance-negative-gen-dataset-op-test-py) | 283 | IN PROGRESS | monolith-rs/crates/monolith-data/tests |  |
| [`monolith/native_training/data/training_instance/python/parse_instance_ops.py`](#monolith-native-training-data-training-instance-python-parse-instance-ops-py) | 245 | IN PROGRESS | monolith-rs/crates/monolith-data/src |  |
| [`monolith/native_training/data/training_instance/python/parse_instance_ops_test.py`](#monolith-native-training-data-training-instance-python-parse-instance-ops-test-py) | 185 | IN PROGRESS | monolith-rs/crates/monolith-data/tests |  |
| [`monolith/native_training/data/training_instance/python/parser_utils.py`](#monolith-native-training-data-training-instance-python-parser-utils-py) | 85 | IN PROGRESS | monolith-rs/crates/monolith-data/src |  |
| [`monolith/native_training/data/training_instance/python/pb_datasource_ops.py`](#monolith-native-training-data-training-instance-python-pb-datasource-ops-py) | 48 | IN PROGRESS | monolith-rs/crates/monolith-data/src |  |
| [`monolith/native_training/data/training_instance/python/test_data_utils.py`](#monolith-native-training-data-training-instance-python-test-data-utils-py) | 15 | IN PROGRESS | none |  |
| [`monolith/native_training/data/transform/transforms.py`](#monolith-native-training-data-transform-transforms-py) | 250 | IN PROGRESS | monolith-rs/crates/monolith-data/src |  |
| [`monolith/native_training/data/transform/transforms_test.py`](#monolith-native-training-data-transform-transforms-test-py) | 70 | IN PROGRESS | monolith-rs/crates/monolith-data/tests |  |
| [`monolith/native_training/data/transform_dataset_test.py`](#monolith-native-training-data-transform-dataset-test-py) | 168 | IN PROGRESS | monolith-rs/crates/monolith-data/tests |  |
| [`monolith/native_training/data/utils.py`](#monolith-native-training-data-utils-py) | 55 | IN PROGRESS | monolith-rs/crates/monolith-data/src |  |
| [`monolith/native_training/debugging/debugging_client.py`](#monolith-native-training-debugging-debugging-client-py) | 98 | IN PROGRESS | monolith-rs/crates/monolith-training/src/debugging |  |
| [`monolith/native_training/debugging/debugging_server.py`](#monolith-native-training-debugging-debugging-server-py) | 217 | IN PROGRESS | monolith-rs/crates/monolith-training/src/debugging |  |
| [`monolith/native_training/demo.py`](#monolith-native-training-demo-py) | 57 | IN PROGRESS | monolith-rs/crates/monolith-training/examples |  |
| [`monolith/native_training/dense_reload_utils.py`](#monolith-native-training-dense-reload-utils-py) | 457 | IN PROGRESS | monolith-rs/crates/monolith-training/src/checkpoint |  |
| [`monolith/native_training/dense_reload_utils_test.py`](#monolith-native-training-dense-reload-utils-test-py) | 192 | IN PROGRESS | monolith-rs/crates/monolith-training/tests |  |
| [`monolith/native_training/device_utils.py`](#monolith-native-training-device-utils-py) | 231 | IN PROGRESS | monolith-rs/crates/monolith-training/src/device |  |
| [`monolith/native_training/device_utils_test.py`](#monolith-native-training-device-utils-test-py) | 104 | IN PROGRESS | monolith-rs/crates/monolith-training/tests |  |
| [`monolith/native_training/distribute/distributed_dataset.py`](#monolith-native-training-distribute-distributed-dataset-py) | 81 | IN PROGRESS | monolith-rs/crates/monolith-data/src |  |
| [`monolith/native_training/distribute/distributed_dataset_test.py`](#monolith-native-training-distribute-distributed-dataset-test-py) | 124 | IN PROGRESS | monolith-rs/crates/monolith-data/tests |  |
| [`monolith/native_training/distribute/str_queue.py`](#monolith-native-training-distribute-str-queue-py) | 114 | IN PROGRESS | monolith-rs/crates/monolith-data/src |  |
| [`monolith/native_training/distribute/str_queue_test.py`](#monolith-native-training-distribute-str-queue-test-py) | 67 | IN PROGRESS | monolith-rs/crates/monolith-data/tests |  |
| [`monolith/native_training/distributed_ps.py`](#monolith-native-training-distributed-ps-py) | 2108 | IN PROGRESS | monolith-rs/crates/monolith-training/src/ps |  |
| [`monolith/native_training/distributed_ps_benchmark.py`](#monolith-native-training-distributed-ps-benchmark-py) | 168 | IN PROGRESS | monolith-rs/crates/monolith-training/benches |  |
| [`monolith/native_training/distributed_ps_factory.py`](#monolith-native-training-distributed-ps-factory-py) | 262 | IN PROGRESS | monolith-rs/crates/monolith-training/src/ps |  |
| [`monolith/native_training/distributed_ps_factory_test.py`](#monolith-native-training-distributed-ps-factory-test-py) | 87 | IN PROGRESS | monolith-rs/crates/monolith-training/tests |  |
| [`monolith/native_training/distributed_ps_sync.py`](#monolith-native-training-distributed-ps-sync-py) | 531 | IN PROGRESS | monolith-rs/crates/monolith-training/src/ps |  |
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
| [`monolith/path_utils.py`](#monolith-path-utils-py) | 47 | IN PROGRESS | monolith-rs/crates/monolith-core/src | |
| [`monolith/tpu_runner.py`](#monolith-tpu-runner-py) | 429 | IN PROGRESS | monolith-rs/crates/monolith-training/src | |
| [`monolith/utils.py`](#monolith-utils-py) | 81 | IN PROGRESS | monolith-rs/crates/monolith-tf/src | |
| [`monolith/utils_test.py`](#monolith-utils-test-py) | 65 | IN PROGRESS | monolith-rs/crates/monolith-tf/tests | |

## Per-File Parity Checklists (All Python Files)

Every file listed below must be fully mapped to Rust with parity behavior verified.

### `monolith/__init__.py`
<a id="monolith-init-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 55
- Purpose/role: Package bootstrap that re-exports key submodules and enables TensorFlow monkey patching.
- Key symbols/classes/functions: `add_module`, side-effect imports of `data`, `layers`, `model_export`, `entry`, `native_model` (as `base_model`), `estimator`.
- External dependencies: `tensorflow.python.tools.module_util` (imported), `absl.logging`, `importlib`, `monolith.utils.enable_monkey_patch`.
- Side effects: imports training modules on import; injects modules into `sys.modules`; may modify TensorFlow monitored_session behavior.

**Required Behavior (Detailed)**
- `add_module(module)`:
  - Accepts a module object or a module string.
  - If a string, imports it; derives `name` from the last path component.
  - If `name == 'native_model'`, rename to `'base_model'`.
  - Registers module in `sys.modules` under `monolith.<name>`.
- On import:
  - Calls `add_module` for `data`, `layers`, `model_export`, `entry`, `native_model` (as `base_model`), `estimator`.
  - Calls `enable_monkey_patch()`; on exception, logs `enable_monkey_patch failed`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/src/lib.rs` (crate-level re-exports), plus `monolith-rs/crates/monolith-training` for training APIs.
- Rust public API surface: `pub use` re-exports of subcrates/modules to mimic `monolith.*` namespace.
- Data model mapping: N/A (module wiring only).

**Implementation Steps (Detailed)**
1. Define a top-level Rust crate that re-exports equivalent submodules (data/layers/training/export/estimator).
2. Ensure any TF monkey patch logic is represented (or explicitly documented as unsupported) in Rust.
3. Avoid heavy side effects on import; if unavoidable, document and feature-gate.

**Tests (Detailed)**
- Python tests: covered indirectly by `monolith/utils_test.py` monkey patch test.
- Rust tests: add a smoke test that `monolith` re-exports are accessible.

**Gaps / Notes**
- Python import side effects are heavy; Rust should provide explicit init if needed.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 46
- Purpose/role: Base runner class with TensorFlow summary writing helper.
- Key symbols/classes/functions: `BaseRunner`, `write_summary`.
- External dependencies: `tensorflow`.
- Side effects: writes TensorFlow summary events to a writer.

**Required Behavior (Detailed)**
- `BaseRunner.__init__` accepts `*args, **kwargs` (no stored fields in base).
- `run()` is abstract and raises `NotImplementedError`.
- `write_summary(logs, summary_writer, current_step)`:
  - Creates a new TF v1 Graph context.
  - Builds `tf.compat.v1.Summary.Value` entries from `logs` dict.
  - Writes to `summary_writer.add_summary(tf_summary, current_step)`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/base_runner.rs`.
- Rust public API surface: `BaseRunner` trait/struct with `run` and `write_summary` helper.
- Data model mapping: summary writer abstraction (likely TB event writer).

**Implementation Steps (Detailed)**
1. Implement a minimal runner trait with `run()`.
2. Provide a summary writer abstraction compatible with TF event files (or document absence).
3. Mirror log-to-summary conversion semantics.

**Tests (Detailed)**
- Python tests: none specific.
- Rust tests: unit test for summary serialization if supported.

**Gaps / Notes**
- Rust likely needs a TensorBoard writer crate to match TF summary outputs.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 51
- Purpose/role: Configure tcmalloc heap profiling via environment variables.
- Key symbols/classes/functions: `enable_tcmalloc`, `setup_heap_profile`.
- External dependencies: `MLPEnv`, `monolith.utils`.
- Side effects: modifies `LD_PRELOAD` and `HEAP_PROFILE_*` env vars.

**Required Behavior (Detailed)**
- `enable_tcmalloc()`:
  - Appends `../gperftools/libtcmalloc/lib/libtcmalloc.so` (resolved via `utils.get_libops_path`) to `LD_PRELOAD`.
- `setup_heap_profile(...)`:
  - Calls `enable_tcmalloc()`.
  - Uses `MLPEnv().index` to name heap profile file `hprof_<index>` in `heap_pro_file` or `utils.find_main()`.
  - Sets env vars:
    - `HEAPPROFILE` path
    - `HEAP_PROFILE_INUSE_INTERVAL` and `HEAP_PROFILE_ALLOCATION_INTERVAL` scaled by `1/sample_ratio`
    - `HEAP_PROFILE_SAMPLE_RATIO`, `HEAP_PROFILE_TIME_INTERVAL`, `HEAP_PROFILE_MMAP`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/mem_profiling.rs`.
- Rust public API surface: functions to set environment variables and preload tcmalloc.

**Implementation Steps (Detailed)**
1. Implement environment variable setup with the same names and scaling.
2. Provide an MLPEnv equivalent or configuration injection for index.
3. Ensure LD_PRELOAD modification is appended (not overwritten).

**Tests (Detailed)**
- Python tests: none
- Rust tests: unit tests for env var values given sample_ratio.

**Gaps / Notes**
- Rust cannot preload shared libs at runtime on all platforms; document OS-specific behavior.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 0
- Purpose/role: Empty initializer for `monolith.core`.
- Key symbols/classes/functions: none
- External dependencies: none
- Side effects: none

**Required Behavior (Detailed)**
- Importing this module must not execute any side effects.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/src/lib.rs`.
- Rust public API surface: module boundary only.

**Implementation Steps (Detailed)**
1. Ensure the Rust crate exposes `monolith-core` without implicit side effects.

**Tests (Detailed)**
- Python tests: none
- Rust tests: none

**Gaps / Notes**
- None.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 376
- Purpose/role: TPU infeed/outfeed SessionRunHook with thread-managed queues, TPU init/shutdown, and end-of-stream stopping signals.
- Key symbols/classes/functions: `PeriodicLogger`, `_SIGNAL`, `_OpQueueContext`, `_OpSignalOnceQueueContext`, `TPUInfeedOutfeedSessionWithEndOfStreamHandlingHook`.
- External dependencies: `threading`, `time`, `os`, `six.moves.queue/xrange`, `tensorflow.compat.v1`, `config_pb2`, `tpu_compilation_result`, `summary_ops_v2` (plus implicit `ops`, `tpu_config`, `session_support`).
- Side effects: spawns threads, initializes/shuts down TPU, runs TF sessions, reads env vars, logs via TF logging.

**Required Behavior (Detailed)**
- `PeriodicLogger(seconds)`:
  - `log()` emits TF log only if elapsed time exceeds `seconds`.
- `_SIGNAL`:
  - `NEXT_BATCH = -1`, `STOP = -2` (negative values reserved for control).
- `_OpQueueContext`:
  - Starts a daemon thread running `target(self, *args)`.
  - `send_next_batch_signal(iterations)` enqueues integer iterations.
  - `read_iteration_counts()` yields iterations until `_SIGNAL.STOP` then returns.
  - `join()` logs, sends STOP, joins thread.
- `_OpSignalOnceQueueContext`:
  - Only allows the first `send_next_batch_signal`; subsequent calls ignored.
- `TPUInfeedOutfeedSessionWithEndOfStreamHandlingHook`:
  - `__init__`:
    - Stores enqueue/dequeue ops, rendezvous, master, session config, init ops, outfeed cadence.
    - Reads embedding config from `ctx` if present.
    - Sets `_should_initialize_tpu=False` when model-parallel + per-host input broadcast; else true.
    - Sets `stopping_signal=False`.
  - `_create_or_get_iterations_per_loop()`:
    - Uses graph collection `tpu_estimator_iterations_per_loop`.
    - If >1 existing var → `RuntimeError("Multiple iterations_per_loop_var in collection.")`.
    - Else creates resource variable in scope `tpu_estimator`, int32 scalar, non-trainable, in LOCAL_VARIABLES, colocated with global step.
  - `begin()`:
    - Records `_iterations_per_loop_var`.
    - Adds TPU shutdown op to `_finalize_ops` if `_should_initialize_tpu`.
    - Adds summary writer init ops to `_init_ops` and flush ops to `_finalize_ops`.
  - `_run_infeed(queue_ctx, session)`:
    - Optional sleep (`initial_infeed_sleep_secs`).
    - If `run_infeed_loop_on_coordinator`: for each iteration signal, runs `enqueue_ops` `steps` times.
    - Else runs `enqueue_ops` once per signal.
  - `_run_outfeed(queue_ctx, session)`:
    - Runs `dequeue_ops` every `outfeed_every_n_steps`.
    - If output includes `_USER_PROVIDED_SIGNAL_NAME`, expects a dict containing `stopping`.
    - When first stopping signal seen, sets `stopping_signals=True` and later flips `self.stopping_signal=True`.
  - `_assertCompilationSucceeded(result, coord)`:
    - Parses TPU compilation proto; if `status_error_message` set → log error + `coord.request_stop()`.
  - `after_create_session()`:
    - If `_should_initialize_tpu`, runs `tf.tpu.initialize_system` in a fresh graph/session.
    - Runs `_init_ops` with 30 minute timeout.
    - If `TPU_SPLIT_COMPILE_AND_EXECUTE=1`, runs `tpu_compile_op` and asserts compilation success.
    - Starts infeed/outfeed controller threads.
    - If `TF_TPU_WATCHDOG_TIMEOUT>0`, starts worker watchdog.
  - `before_run()`:
    - If `stopping_signal` is set, raises `tf.errors.OutOfRangeError`.
    - Reads `iterations_per_loop`, sends signals to infeed/outfeed controllers.
  - `end()`:
    - Joins infeed/outfeed threads; calls rendezvous `record_done`; runs finalize ops (flush + shutdown).
  - `get_stopping_signals_and_name(features)`:
    - If `_USER_PROVIDED_SIGNAL_NAME` in `features`, uses `tf.tpu.cross_replica_sum` to compute `stopping` boolean.
    - Returns `(stopping_signals, _USER_PROVIDED_SIGNAL_NAME)`.
- Env vars:
  - `TPU_SPLIT_COMPILE_AND_EXECUTE=1` triggers separate compile step.
  - `TF_TPU_WATCHDOG_TIMEOUT` enables worker watchdog.
- Threading: two daemon threads (infeed/outfeed) controlled via queues.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-tf/src` (TPU runtime integration).
- Rust public API surface: no equivalent; would need a TPU session hook abstraction.
- Data model mapping: TF session/run hooks have no direct Rust equivalent.
- Feature gating: only relevant if TF-runtime backend is enabled.
- Integration points: training loop / input pipeline in `monolith-training`.

**Implementation Steps (Detailed)**
1. Decide if TPU SessionRunHook is in-scope for Rust TF backend.
2. If in-scope, implement infeed/outfeed controller threads and queue signaling.
3. Expose env-var toggles for compile/execute split and watchdog timeout.
4. Provide cross-replica stopping signal handling for outfeed.

**Tests (Detailed)**
- Python tests: none dedicated (covered indirectly by TPU training flows).
- Rust tests: none yet; would need integration tests with TF runtime.
- Cross-language parity test: not applicable unless TF runtime implemented in Rust.

**Gaps / Notes**
- Missing imports in Python (`ops`, `tpu_config`, `session_support`) are referenced but not imported.
- Rust has no TPU SessionRunHook analog; full parity requires TF runtime + hook infrastructure.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 643
- Purpose/role: Host call logic for embedding tasks, including TPU variable caching and DeepInsight metrics.
- Key symbols/classes/functions: `BaseEmbeddingHostCall`, `update_tpu_variables_ops`, `generate_host_call_hook`.
- External dependencies: `tensorflow.compat.v1`, `tensorflow`, `BaseHostCall`, `ReplicatedVariable`.
- Side effects: creates TPU variables on all replicas; writes TF summaries; computes AUC.

**Required Behavior (Detailed)**
- Constants define metric names and TPU variable names.
- `TPUVariableRestoreHook` runs assign op after session creation.
- `BaseEmbeddingHostCall.__init__`:
  - Stores flags for host call, deepinsight, scalar metrics, caching mode.
  - Extracts `context` unless `cpu_test`.
  - Creates TPU variables if `enable_host_call` and caching enabled.
- `_create_tpu_var`:
  - Creates per-replica TPU variables across hosts with zeros, adds to `TPU_VAR` collection.
  - Wraps in `ReplicatedVariable`; registers restore hooks.
- `_compute_new_value(base, delta, update_offset)`:
  - Pads `delta` to base length, then `tf.roll` by offset, then adds to base.
- `update_tpu_variables_ops(...)`:
  - Clears TPU vars when `global_step % host_call_every_n_steps == 1`.
  - Writes labels/preds/uid_buckets and req_times/sample_rates into TPU vars at computed offsets.
  - Updates accumulated counter only after tensor updates.
- `record_summary_tpu_variables()` collects TPU vars into host call tensors.
- `record_summary_tensor` filters based on `enable_host_call_scalar_metrics` and deprecated metric names.
- `_write_summary_ops`:
  - Filters by uid_bucket and stopping_signals; computes AUC; writes scalar summaries and serialized tensors.
- `generate_host_call_hook()`:
  - Calls `compress_tensors()` and returns either `_host_call` or `_host_call_with_tpu` plus tensor list; or `None` if disabled.
  - Host call writes summaries under `output_dir/host_call` with `tf2.summary`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/src/base_embedding_host_call.rs`.
- Rust public API surface: host call builder with optional TPU caching mode.
- Data model mapping: summary writer and AUC metric implementation.

**Implementation Steps (Detailed)**
1. Recreate TPU variable caching logic (or provide equivalent buffer) with host-call step windowing.
2. Implement summary writing and AUC computation matching TF semantics.
3. Mirror filtering by UID buckets and stopping signals.
4. Provide compatibility with TPU/CPU test modes.

**Tests (Detailed)**
- Python tests: `monolith/core/base_embedding_host_call_test.py`
- Rust tests: unit tests for `_compute_new_value` and host call output shapes.

**Gaps / Notes**
- Full parity requires TF TPU runtime; Rust may need to stub or bridge this functionality.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 77
- Purpose/role: Tests `_compute_new_value` in BaseEmbeddingHostCall.
- Key symbols/classes/functions: `BaseEmbeddingHostCallTest.test_compute_new_value`.
- External dependencies: `tensorflow.compat.v1`.
- Side effects: none.

**Required Behavior (Detailed)**
- Constructs `BaseEmbeddingHostCall` with `enable_host_call=False` and `context=None`.
- Verifies `_compute_new_value` behavior with different offsets.
- Uses TF session to evaluate results and compares to expected tensors.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/tests/base_embedding_host_call.rs`.
- Rust public API surface: `_compute_new_value` equivalent.

**Implementation Steps (Detailed)**
1. Port `_compute_new_value` logic in Rust and add unit tests for offsets.

**Tests (Detailed)**
- Python tests: this file
- Rust tests: same assertions.

**Gaps / Notes**
- Needs tensor ops; may require a small tensor wrapper or TF binding in Rust tests.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 611
- Purpose/role: Base class for TPU embedding tasks; builds input pipeline, vocab sizing, and TPU embedding configs.
- Key symbols/classes/functions: `BaseEmbeddingTask.params`, `create_input_fn`, `_post_process_example`, `create_feature_and_table_config_dict`, `process_features_for_cpu_test`.
- External dependencies: `tensorflow.compat.v1`, `tpu_embedding`, `FeatureSlot/FeatureColumn`, `auto_checkpoint_feed_hook`, `util`.
- Side effects: reads vocab file (possibly from HDFS), constructs TF datasets and embedding configs.

**Required Behavior (Detailed)**
- `params()` defines many embedding-related flags, including vocab sizing, QR hashing, deepinsight, host call metrics, file input ranges, and stopping signals.
- `__init__`:
  - Sets flags, builds vocab dict (from file or downloaded from HDFS), constructs `Env`.
- `download_vocab_size_file_from_hdfs()`:
  - Downloads a single `part*.csv` from HDFS into temp folder; updates `p.vocab_file_path` if successful.
- `_create_vocab_dict()`:
  - Reads vocab file (tsv slot_id -> count), applies overrides and offsets, returns dict.
- `create_input_fn(mode)`:
  - Only supports TRAIN.
  - If `file_pattern` provided, uses `tf.data.Dataset.list_files` (no shuffle); else uses `file_folder` + `date_and_file_name_format` and `util.range_dateset` with `start_date/end_date`.
  - Shards files per TPU host call index.
  - Interleaves files with `cycle_length` and parses examples via `_get_feature_map` and `_post_process_example`.
  - If `enable_stopping_signals`, appends a final batch with stop signal flag via `auto_checkpoint_feed_hook`.
  - Prefetches with AUTOTUNE.
- `_post_process_example`:
  - Converts embedding tensors to SparseTensor; applies vocab size mods, QR hashing, and FeatureColumn3D row_lengths.
  - Adds UID bucket for AUC sampling if `_UID` exists.
- `create_feature_and_table_config_dict()`:
  - Builds `tpu_embedding.TableConfig` and `FeatureConfig` per slot/feature slice, including QR hashing tables.
- `process_features_for_cpu_test()`:
  - Creates embedding variables with random init and uses `safe_embedding_lookup_sparse`.
  - Clears internal feature/table config dicts after processing.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/src/base_embedding_task.rs`.
- Rust public API surface: base embedding task trait/struct with dataset and embedding config helpers.
- Data model mapping: TF dataset pipeline and TPU embedding configs (likely via TF runtime bridge).

**Implementation Steps (Detailed)**
1. Port parameter schema and vocab file parsing.
2. Implement dataset pipeline generation or equivalent data loader.
3. Recreate embedding config generation and QR hashing logic.
4. Implement CPU test path for embeddings.

**Tests (Detailed)**
- Python tests: none direct; used by TPU runner tests.
- Rust tests: unit tests for vocab dict creation and QR hashing config creation.

**Gaps / Notes**
- Full parity depends on TF TPU embedding APIs; Rust likely needs a bridge or compatibility layer.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 145
- Purpose/role: Collects tensors for TPU host calls, with compression/decompression by dtype.
- Key symbols/classes/functions: `BaseHostCall.record_summary_tensor`, `compress_tensors`, `decompress_tensors`.
- External dependencies: `tensorflow`, `absl.logging`.
- Side effects: builds internal tensor lists; uses global_step tensor.

**Required Behavior (Detailed)**
- `__init__(output_dir, enable_host_call)`:
  - Initializes `_tensor_names` with `"global_step"` and `_tensors` with reshaped global step.
  - `_lists_tensor_sizes` tracks original sizes for decompression.
- `record_summary_tensor(name, tensor)`:
  - No-op if host call disabled.
  - Asserts unique name, asserts tensor rank <= 1, reshapes to `[-1]`, appends.
- `compress_tensors()`:
  - Groups tensors by dtype; concatenates each dtype list along axis 0, then `expand_dims` to add batch dimension.
  - Stores per-group tensor sizes in `_lists_tensor_sizes`.
  - Replaces `_tensor_names` and `_tensors` with compressed versions.
- `decompress_tensors(tensors)`:
  - Splits each compressed tensor by recorded sizes; squeezes to 1D.
  - Asserts first tensor name is `global_step` (error message uses first character only).
  - Returns `(global_step_scalar, decompressed_tensor_list)` where global step is `decompressed_tensor_list[0][0]`.
- `generate_host_call_hook()` default returns `None` (to be overridden).

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/src/base_host_call.rs`.
- Rust public API surface: `BaseHostCall` struct with tensor collection and (de)compression.

**Implementation Steps (Detailed)**
1. Implement tensor collection and dtype grouping logic.
2. Preserve compression/decompression shapes and per-dtype grouping semantics.
3. Mirror global_step positioning and return shape.

**Tests (Detailed)**
- Python tests: none specific (used indirectly by embedding host call tests).
- Rust tests: unit tests for compress/decompress roundtrip.

**Gaps / Notes**
- Error message in assert uses `self._tensor_names[0][0]` (first char); keep for parity.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 161
- Purpose/role: Base class for layers with child management, name assignment, and per-graph layer loss tracking.
- Key symbols/classes/functions: `BaseLayer`, `get_uname`, `add_layer_loss`, `get_layer_loss`.
- External dependencies: `tensorflow`, `InstantiableParams`, `NestedMap`.
- Side effects: global registries `_layer_loss` and `_name_inuse` are mutated.

**Required Behavior (Detailed)**
- `params()`:
  - Returns `InstantiableParams(cls)` and defines `name` using `get_uname(cls.__name__)`.
- `__init__(params)`:
  - Asserts `params.name` is set.
  - Initializes `_private_children` as `NestedMap`.
- `children` property returns `_private_children`.
- `__getattr__`:
  - Raises AttributeError if `_private_children` not created.
  - If `name` in children, returns it.
  - If class has a property with that name, calls the property's getter to surface the same AttributeError.
  - Else raises `"<name> is not a sub-layer of <self>."`.
- `__call__` forwards to `fprop()`.
- `fprop()` is abstract: raises `NotImplementedError('Abstract method of %s' % self)`.
- `create_child(name, params)`:
  - If `params.name` empty, assigns `self.p.name` (assumes BaseLayer has `p` attribute set by InstantiableParams).
  - Instantiates child via `params.instantiate()` and stores under `_private_children[name]`.
- `create_children(name, params_list)`:
  - Creates list; for each param, sets `param.name = f"{name}_{index}"` if missing; instantiates and appends.
- `get_uname(name)`:
  - Uses `_name_inuse` defaultdict; **note**: code checks membership but never inserts, so it currently always returns `name` (no suffix) unless keys are inserted elsewhere.
- `add_layer_loss(name, loss)`:
  - Adds loss into `_layer_loss` keyed by default graph and layer name.
- `get_layer_loss()`:
  - Returns the dict for the current default graph.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/src/base_layer.rs`.
- Rust public API surface: `BaseLayer` trait/struct, child map, `get_uname`, layer loss registry.
- Data model mapping: `NestedMap` equivalent and parameter instantiation mechanism.

**Implementation Steps (Detailed)**
1. Implement a base layer trait with child management and `fprop` entrypoint.
2. Provide a `NestedMap` equivalent with attribute-like access.
3. Mirror `get_uname` behavior (including current no-op uniqueness unless explicitly fixed).
4. Implement per-graph loss registry or document deviation if Rust lacks TF graphs.

**Tests (Detailed)**
- Python tests: `monolith/core/base_layer_test.py`
- Rust tests: verify create_child/create_children and `__getattr__`-like behavior.

**Gaps / Notes**
- BaseLayer relies on `self.p` being set by hyperparams instantiation; mirror this in Rust or document.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 41
- Purpose/role: Tests child creation APIs in `BaseLayer`.
- Key symbols/classes/functions: `BaseLayerTest.test_create_child`, `.test_create_children`.
- External dependencies: `base_layer`.
- Side effects: none.

**Required Behavior (Detailed)**
- `test_create_child`:
  - Instantiates BaseLayer params, sets name, creates layer, calls `create_child`, and asserts child exists.
- `test_create_children`:
  - Creates two child layers and asserts list length is 2.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/tests/base_layer.rs`.
- Rust public API surface: BaseLayer child creation.

**Implementation Steps (Detailed)**
1. Port tests using Rust BaseLayer + params instantiation.
2. Assert child map/list presence and lengths.

**Tests (Detailed)**
- Python tests: this file
- Rust tests: same assertions.

**Gaps / Notes**
- Python tests access `_disable_create_child` which is not defined in BaseLayer; confirm if Rust needs similar flag (likely no-op).

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 25
- Purpose/role: Defines abstract params holder for single-task models.
- Key symbols/classes/functions: `SingleTaskModelParams.task`.
- External dependencies: none.
- Side effects: none.

**Required Behavior (Detailed)**
- `SingleTaskModelParams.task()` is abstract and must be overridden to return task params; raises `NotImplementedError('Abstract method')` by default.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/src/base_model_params.rs`.
- Rust public API surface: trait or struct with `task()`/`task_params()` abstract method.

**Implementation Steps (Detailed)**
1. Create a Rust trait for model params with `task()` returning task config.
2. Ensure errors are explicit when called on base type.

**Tests (Detailed)**
- Python tests: none specific.
- Rust tests: optional compile-time trait enforcement or runtime panic test.

**Gaps / Notes**
- None.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 95
- Purpose/role: Base class for a single training task, defining standard hyperparams and abstract input/model functions.
- Key symbols/classes/functions: `BaseTask.params`, `create_input_fn`, `create_model_fn`.
- External dependencies: `base_layer.BaseLayer`, `hyperparams.Params`.
- Side effects: none.

**Required Behavior (Detailed)**
- `params()`:
  - Extends `BaseLayer.params()` with:
    - `accelerator` (None/"tpu"/"horovod")
    - `input.eval_examples`, `input.train_examples`
    - `eval.per_replica_batch_size`, `eval.steps_per_eval`, `eval.steps`
    - `train.steps`, `train.max_steps`, `train.per_replica_batch_size`, `train.file_pattern`, `train.repeat`, `train.label_key`, `train.save_checkpoints_steps`, `train.save_checkpoints_secs`, `train.dense_only_save_checkpoints_secs`, `train.dense_only_save_checkpoints_steps`
- `create_input_fn(mode)` and `create_model_fn()` are abstract and must be overridden.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/src/base_task.rs`.
- Rust public API surface: base task trait with hyperparams schema and abstract hooks.
- Data model mapping: `Params` equivalent (typed config or map).

**Implementation Steps (Detailed)**
1. Implement base task trait/struct with default parameter schema.
2. Provide typed config structures with the same field names.
3. Ensure overridden methods are required (trait methods).

**Tests (Detailed)**
- Python tests: none specific.
- Rust tests: verify default params include the expected keys.

**Gaps / Notes**
- This depends on the Rust `Params`/hyperparams implementation to mirror Python semantics.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 73
- Purpose/role: Base TPU tests for running TPU tasks on CPU and validating merged vector behavior.
- Key symbols/classes/functions: `BaseTPUTest.runWithCPU`, `runMergeVectorTestOnCPU`.
- External dependencies: `model_registry`, `TPURunner`.
- Side effects: runs TPU runner in CPU test mode.

**Required Behavior (Detailed)**
- `runWithCPU(task_name)`:
  - Retrieves task params, instantiates TPURunner, sets `_cpu_test=True` and `_host_call_every_n_steps=0`, runs.
- `runMergeVectorTestOnCPU(task_name)`:
  - Enables `merge_vector` on task params; runs in CPU test mode.
  - Validates merged slot dims and embedding dims in runner task env.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests/base_tpu.rs`.
- Rust public API surface: TPU runner CPU-test mode.

**Implementation Steps (Detailed)**
1. Provide CPU test mode in Rust TPU runner.
2. Expose task env state for merge_vector assertions.

**Tests (Detailed)**
- Python tests: used as base class in TPU-related tests.
- Rust tests: implement equivalent helper assertions.

**Gaps / Notes**
- Requires task env structures in Rust to match slot/dim semantics.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 35
- Purpose/role: Aggregates core unit tests into a suite.
- Key symbols/classes/functions: `suite()`.
- External dependencies: `unittest`, `ParamsTest`, `BaseLayerTest`, `BaseEmbeddingHostCallTest`, `UtilTest`.
- Side effects: runs test suite when invoked as main.

**Required Behavior (Detailed)**
- `suite()` creates a `unittest.TestSuite` containing the four test classes.
- `__main__` runs suite with `TextTestRunner(verbosity=2)`.

**Rust Mapping (Detailed)**
- Target crate/module: N/A (Rust test harness handles suites).
- Rust public API surface: none; map to Rust test module organization.

**Implementation Steps (Detailed)**
1. Ensure Rust tests cover the same components.
2. Document that Python-style suite is not required in Rust.

**Tests (Detailed)**
- Python tests: this file
- Rust tests: already covered by individual test modules.

**Gaps / Notes**
- None.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 179
- Purpose/role: Custom Dense layer combining TF Keras Dense with BaseLayer and optional kernel normalization.
- Key symbols/classes/functions: `Dense.params`, `build`, `fprop`.
- External dependencies: `tensorflow`, `VarianceScaling`.
- Side effects: creates TF variables (kernel/bias/trainable norms).

**Required Behavior (Detailed)**
- `params()`:
  - Defines units, activation, use_bias, kernel/bias initializers, kernel norm options, and partitioner.
- `__init__`:
  - Initializes BaseLayer and tf.keras.layers.Dense with given params.
  - Sets attributes: `allow_kernel_norm`, `kernel_norm_trainable`, `var_name_prefix`, `partitioner`.
- `build(input_shape)`:
  - Validates dtype (float/complex).
  - Uses `VarianceScaling` initializer to create kernel; uses `tf.compat.v1.get_variable` (partitioner optional).
  - If `allow_kernel_norm`: L2-normalizes kernel and optionally multiplies by trainable norm variable.
  - Creates bias if `use_bias`.
- `get_config()` merges base config with custom fields.
- `fprop(inputs)` calls `self.call(inputs)`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-layers/src/dense.rs`.
- Rust public API surface: Dense layer with optional kernel normalization and partitioner hooks.

**Implementation Steps (Detailed)**
1. Implement Dense forward pass and parameter initializers.
2. Add optional kernel normalization and trainable scaling.
3. Preserve config serialization fields.

**Tests (Detailed)**
- Python tests: `monolith/core/dense_test.py`
- Rust tests: layer instantiation, dtype handling, partitioner behavior.

**Gaps / Notes**
- Partitioned variables are TF-specific; Rust may need abstraction or ignore.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 108
- Purpose/role: Tests Dense layer instantiation, dtype handling, and partitioner.
- Key symbols/classes/functions: `DenseTest` methods.
- External dependencies: `testing_utils.layer_test`, `Dense`.
- Side effects: creates TF variables and runs sessions.

**Required Behavior (Detailed)**
- `test_dense_instantiate`: runs `layer_test` for different input shapes.
- `test_dense_dtype`: ensures output dtype is float32 when specified.
- `test_dense`: checks output shape and runs session to initialize vars.
- `test_dense_with_partitioner`: ensures Dense works with variable partitioner.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-layers/tests/dense.rs`.
- Rust public API surface: Dense layer tests.

**Implementation Steps (Detailed)**
1. Port tests to Rust layer API.
2. Validate shapes and dtype outputs.
3. Provide a mock partitioner or skip if unsupported (document).

**Tests (Detailed)**
- Python tests: this file
- Rust tests: parity tests.

**Gaps / Notes**
- Some TF-specific behaviors may require stubbing in Rust.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 611
- Purpose/role: Sail-like feature API for defining feature slots/slices/columns and embedding lookups with optional merged vector handling.
- Key symbols/classes/functions: `FeatureSlice`, `FeatureSlot`, `FeatureColumnV1`, `FeatureColumn3D`, `Env`.
- External dependencies: `tensorflow`, `absl.logging`, `collections.namedtuple` (imported; unused).
- Side effects: creates TF placeholders, mutates shared `Env` state, writes to `_tpu_features`, logs via `absl.logging`.

**Required Behavior (Detailed)**
- `FeatureSlice`:
  - Constructor stores `feature_slot`, `dim`, `slice_index`, `optimizer`, `initializer`, `learning_rate_fn`.
  - `__repr__` returns `[FeatureSlice][slot_{slot_id}][{slice_index}]` (used as dict key).
  - `__hash__` uses `(feature_slot.slot_id(), slice_index)`.
  - Read-only properties: `dim`, `slice_index`, `optimizer`, `initializer`, `learning_rate_fn`.
- `FeatureSlot`:
  - Constructor registers itself into `Env` via `env.set_feature_slot(slot_id, self)`.
  - If `has_bias=True`, creates bias `FeatureSlice` with `dim=1`, `slice_index=0` using bias optimizer/initializer/lr and appends to `_feature_slices`.
  - `add_feature_slice(dim, optimizer=None, initializer=None, learning_rate_fn=None)`:
    - Defaults to `default_vec_*` settings when args are `None`.
    - Creates `FeatureSlice` with `slice_index=len(_feature_slices)`; appends and returns it.
  - `add_merged_feature_slice(...)`: same as `add_feature_slice` but appends to `_merged_feature_slices`.
  - `_add_feature_column(feature_column)`:
    - Adds to `_feature_columns`.
    - If `has_bias=True`, sets `_bias` for **all** feature columns by calling `feature_column.embedding_lookup(feature_slices[0])`.
  - Properties expose `bias_*`, `default_vec_*`, `feature_slices`, `merged_feature_slices`, `feature_columns`.
- `FeatureColumnV1`:
  - Constructor stores `feature_slot`, `fc_name`; initializes placeholder dicts and `_bias=None`; registers with `FeatureSlot`.
  - `embedding_lookup(feature_slice, init_minval_for_oov=None, init_maxval_for_oov=None)`:
    - Delegates to `Env._embedding_lookup`.
  - `get_bias()` asserts `_bias is not None`.
  - `feature_slice_to_tf_placeholder`:
    - Asserts `env.is_finalized` (note: Python has a bug, method recurses).
    - Returns merged or non-merged placeholder map based on `env._merge_vector`.
- `FeatureColumn3D`:
  - Constructor sets `max_seq_length`, logs it, registers with slot.
  - `embedding_lookup(...)` delegates to `Env._seq_embedding_lookup`.
  - `size_tensor_lookup()` delegates to `Env._size_tensor_lookup`.
  - `feature_slice_to_tf_placeholder` returns 3D placeholder map.
- `Env`:
  - Constructor initializes `vocab_size_dict`, `_slot_id_to_feature_slot`, `_tpu_features=None`, `_is_finalized=False`, then calls `set_params(params)`.
  - `set_params(params)` reads:
    - `qr_multi_hashing`, `qr_hashing_threshold`, `qr_collision_rate`,
      `use_random_init_embedding_for_oov`, `merge_vector`.
  - `set_feature_slot(slot_id, feature_slot)`:
    - If already finalized, returns immediately.
    - Asserts `slot_id` uniqueness.
  - `set_tpu_features(tpu_features)`:
    - Assigns `_tpu_features`; if `_merge_vector` is true, calls `_split_merged_embedding` on each feature slot.
  - `_embedding_lookup(feature_column, feature_slice, init_minval_for_oov=None, init_maxval_for_oov=None)`:
    - Asserts slot IDs match.
    - If `_tpu_features` exists:
      - If `qr_multi_hashing` and `vocab_size_dict[slot_id] > qr_hashing_threshold`, uses quotient/remainder features (`fc_name_slice_0`, `fc_name_slice_1`) and returns their sum.
      - Otherwise uses key `"{fc_name}_{slice_index}"`.
      - If `use_random_init_embedding_for_oov` and `init_minval_for_oov` provided:
        - Computes `norm` across axis=1, replaces near-zero rows with `tf.random.uniform` values.
    - If no `_tpu_features`:
      - Creates `tf.compat.v1.placeholder(tf.float32, [None, dim])` keyed by `FeatureSlice` object.
  - `_seq_embedding_lookup(...)`:
    - Same structure as `_embedding_lookup` but uses placeholder shape `[None, max_seq_length, dim]`.
    - Random OOV initialization uses `tf.random.uniform` and `tf.norm(axis=1)` (3D norms).
  - `_size_tensor_lookup(feature_column)`:
    - If `_tpu_features` exist, reads `"{fc_name}_0_row_lengths"` and builds `[B, max_seq_length]` boolean mask (cast to `int32`).
    - Else returns placeholder `tf.float32` with shape `[None, max_seq_length]` named `"{fc_name}_size"`.
  - `finalize()`:
    - Asserts not already finalized; sets `_is_finalized=True`.
    - If `_merge_vector`, calls `_merge_vector_in_same_slot()`.
  - `_merge_vector_in_same_slot()`:
    - For each slot, keeps bias slice separate (assert `dim==1`).
    - Merges remaining slices into a single merged `FeatureSlice` whose dim is sum of vector dims.
    - For each feature column, creates placeholder for merged slice and updates `_merged_feature_slice_to_tf_placeholder`.
  - `_split_merged_embedding(feature_slot)`:
    - For each feature column, finds merged embedding from `_tpu_features`, splits by per-slice dims, and writes back individual embeddings into `_tpu_features` by `"{fc_name}_{slice_index}"`.
  - Properties:
    - `vocab_size_dict` returns stored dict.
    - `slot_id_to_feature_slot` asserts `is_finalized` (buggy in Python) then returns map.
    - `features` returns `_tpu_features`.
- Determinism: random OOV embeddings depend on TF RNG.
- Logging: `logging.info` in `FeatureColumn3D.__init__`, `logging.vlog` in lookup methods.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/src/feature.rs`.
- Rust public API surface: `FeatureSlot`, `FeatureSlice`, `SparseFeatureColumn`, `DenseFeatureColumn`, `FeatureColumn` trait.
- Data model mapping: Python Sail-like slot/slice/column + `Env` **do not exist** in Rust; current Rust types represent data containers, not TF graph wiring.
- Feature gating: TF placeholder/graph APIs are Python-only; Rust needs an explicit TF backend or a compat layer.
- Integration points: Python `Env` expected by embedding tasks; Rust `feature.rs` is not integrated with embedding task flow.

**Implementation Steps (Detailed)**
1. Decide whether to port the Sail-like API or provide a shim around existing Rust feature structs.
2. If porting, add `Env`, `FeatureSlot`, `FeatureSlice`, `FeatureColumnV1/3D` in Rust with TF backend hooks.
3. Implement placeholder / feature tensor registry for training/serving use cases.
4. Add merge/split vector logic with deterministic slice ordering.
5. Match OOV random initialization (norm threshold 1e-10) and QR hashing behavior.
6. Capture and fix Python bugs when porting (e.g., `is_finalized` recursion, undefined `slot_id`).

**Tests (Detailed)**
- Python tests: `monolith/core/feature_test.py`.
- Rust tests: add `monolith-rs/crates/monolith-core/tests/feature.rs` mirroring merge/split behavior and bias handling.
- Cross-language parity test: compare split/merge outputs and placeholder shapes.

**Gaps / Notes**
- Python `Env.is_finalized` method recurses (`return self.is_finalized`) instead of returning `_is_finalized`.
- `_embedding_lookup` uses undefined `slot_id` in QR branch; should likely be `feature_column._feature_slot.slot_id()`.
- `_seq_embedding_lookup` references `feature_slice.init_minval_for_oov`/`init_maxval_for_oov` which are not defined.
- Rust `feature.rs` implements data structures, not the TF embedding/placeholder semantics used in Python.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 178
- Purpose/role: Tests bias slice creation, feature slice indexing, feature column registration, and merged vector split/merge behavior.
- Key symbols/classes/functions: `FeatureSlotTest`, `FeatureColumnV1Test`.
- External dependencies: `tensorflow`, `monolith.core.hyperparams`, `FeatureSlot`, `FeatureColumnV1`, `Env`.
- Side effects: builds TF placeholders/tensors; uses TF session for split verification.

**Required Behavior (Detailed)**
- `FeatureSlotTest.test_has_bias`:
  - `FeatureSlot(has_bias=True)` creates one bias slice with `dim=1`, `slice_index=0`.
- `FeatureSlotTest.test_add_feature_slice`:
  - Additional slices get incrementing `slice_index` and correct dims.
- `FeatureColumnV1Test.test_add_feature_column`:
  - Creating a feature column appends to `FeatureSlot._feature_columns`.
- `FeatureColumnV1Test.test_merge_split_vector_in_same_slot`:
  - With `merge_vector=True`:
    - `_merge_vector_in_same_slot()` populates `_merged_feature_slices` and merged placeholder maps per slot and column.
    - Expected merged dims:
      - slot 1 (bias+2) -> merged dims `[1,2]`
      - slot 2 (bias only) -> `[1]`
      - slot 3 (no bias, 2+3) -> `[5]`
      - slot 4 (bias+2+3+4) -> `[1,9]`
    - `_split_merged_embedding()` splits merged embeddings into per-slice tensors with exact contents as asserted.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/src/feature.rs`.
- Rust public API surface: currently missing Env + merge/split logic.
- Data model mapping: add tests once feature API exists in Rust.
- Feature gating: TF backend likely required for placeholder/graph ops.
- Integration points: embedding tasks and feature pipeline.

**Implementation Steps (Detailed)**
1. Implement Rust equivalents for `Env`, `FeatureSlot`, `FeatureColumnV1`.
2. Add merge/split vector unit tests mirroring Python.
3. Verify placeholder/tensor semantics for no-feature and TPU-feature cases.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: `monolith-rs/crates/monolith-core/tests/feature.rs`.
- Cross-language parity test: compare split/merge outputs and bias dims.

**Gaps / Notes**
- Rust feature module is currently a different abstraction; merge/split tests are not implemented.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 248
- Purpose/role: Host call implementation for non-TPU-variable caching; collects tensors and writes summaries (AUC, deepinsight).
- Key symbols/classes/functions: `HostCall.record_summary_tensor`, `compress_tensors`, `generate_host_call_hook`.
- External dependencies: `tensorflow`, `absl.logging`.
- Side effects: creates summary writers and writes scalar/text summaries.

**Required Behavior (Detailed)**
- Similar tensor collection/compression logic to `BaseHostCall`:
  - Global step tensor is always first.
  - Tensors grouped by dtype, concatenated, expanded to batch dimension.
- `generate_host_call_hook()`:
  - If disabled, returns None.
  - Otherwise returns `_host_call` and compressed tensors.
  - `_host_call` decompresses tensors, writes scalar summaries and AUC; optional deepinsight text summaries via `_serialize_messages`.
- `_serialize_messages`:
  - Verifies shapes/dtypes, flattens tensors, writes serialized tensors as text summaries.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/src/host_call.rs`.
- Rust public API surface: host call builder with summary writer integration.

**Implementation Steps (Detailed)**
1. Port tensor compression/decompression semantics.
2. Implement summary writer outputs (scalar + text) and AUC calculation.
3. Support optional deepinsight serialization.

**Tests (Detailed)**
- Python tests: none specific.
- Rust tests: unit tests for compress/decompress and summary output formatting.

**Gaps / Notes**
- Reuses logic duplicated in BaseHostCall; consider shared helper in Rust.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 439
- Purpose/role: Dynamic hyperparameter container with attribute + dotted-path access, nested parameter trees, immutability, stringification, and class instantiation support.
- Key symbols/classes/functions: `_is_named_tuple`, `_SortedDict`, `_Param`, `Params`, `InstantiableParams`, `copy_params_to`, `update_params`, `allowed_kwargs`.
- External dependencies: `copy`, `re`, `six`, `inspect.signature/Parameter`, `tensorflow` (Tensor detection for deepcopy).
- Side effects: error messages include similar-key hints; deepcopy of `tf.Tensor` intentionally keeps a reference (no deep copy).

**Required Behavior (Detailed)**
- `_is_named_tuple(x)`:
  - Returns true if `x` is a `tuple` and has `_fields` attribute.
- `_SortedDict.__repr__()`:
  - Always renders dict entries sorted by key.
- `_Param`:
  - Stores `name`, `value`, `description`.
  - `__eq__` compares only name + value (description ignored).
  - `__deepcopy__`: if value is a `tf.Tensor`, **do not copy** (keep ref); else deep-copy; memo updated.
  - `to_string(nested_depth)`:
    - For `Params`: delegate to nested `_to_string`.
    - For `dict`: produce `_SortedDict` with nested `GetRepr` on values.
    - For `list/tuple` (non-namedtuple): reconstruct same type from `GetRepr` values.
    - If value has `Repr` method, call it (used for function-like objects).
    - If value is `str`, wrap in double quotes.
    - Indent with 2 spaces per nesting level.
  - `set` stores value without copying.
  - `get` returns stored value.
- `Params`:
  - Internal storage: `_params` dict mapping name → `_Param`; `_immutable` bool.
  - `__setattr__` / `__setitem__`: if immutable -> `TypeError("This Params instance is immutable.")`. Otherwise set existing param; missing name -> `AttributeError` with `_key_error_string`.
  - `__getattr__` / `__getitem__`: missing name -> `AttributeError` with `_key_error_string`.
  - `__dir__` returns sorted param names. `__contains__`, `__len__` work on `_params`.
  - `__eq__`: only equal to another `Params` with equal `_params` (uses `_Param.__eq__`).
  - `__str__` / `_to_string`: emits multi-line block with params sorted by name; nested `Params` rendered recursively.
  - `_similar_keys(name)`:
    - Builds list of keys with >0.5 overlap of 3-char substrings from `name`.
  - `_key_error_string(name)`:
    - If similar keys exist: `"name (did you mean: [k1,k2])"` with keys sorted.
  - `define(name, default_value, description)`:
    - Reject if immutable → `TypeError("This Params instance is immutable.")`.
    - Assert `name` is `str` matching `^[a-z][a-z0-9_]*$` (raises `AssertionError`).
    - If already defined, `AttributeError("Parameter %s is already defined" % name)`.
  - `freeze()` sets `_immutable=True`; `is_immutable()` returns bool.
  - `_get_nested("a.b[0].c")`:
    - Supports dot navigation into nested `Params`.
    - Supports list indexing in a segment via `name[index]`.
    - If missing segment: `AttributeError` with partial path.
    - If intermediate is not `Params`: `AssertionError("Cannot introspect <type> for <path>")`.
  - `set(**kwargs)`:
    - If immutable -> `TypeError("This Params instance is immutable: %s" % self)`.
    - Dotted names traverse `_get_nested`; missing keys -> `AttributeError` with similar key hints.
    - Returns `self`.
  - `get(name)`:
    - Dotted names traverse `_get_nested`; missing -> `AttributeError` with similar key hints.
  - `delete(*names)`:
    - If immutable -> `TypeError("This Params instance is immutable.")`.
    - Dotted names traverse `_get_nested`; missing -> `AttributeError` with similar key hints.
    - Returns `self`.
  - `iter_params()` yields `(name, value)` for all params (dict iteration order).
  - `copy()` deep-copies `_params` and preserves `_immutable`.
- `copy_params_to(from_p, to_p, skip=None)`:
  - For each param in `from_p`, sets into `to_p`.
  - Nested `Params` are copied via `p.copy()`.
  - `skip` list omits names.
- `InstantiableParams(Params)`:
  - Defines param `cls`.
  - `instantiate()`:
    - If `cls.__init__` has exactly 2 parameters and `cls` has attribute `params` and `'params'` is a parameter: call `cls(self)`.
    - Otherwise, build inverted index of all leaf (non-Params) values; pass matching `__init__` parameter names (excluding `self`, `cls`, varargs/kwargs).
    - Additionally pass any `allowed_kwargs` present and not `None`.
- `update_params(params, args)`:
  - Recursively traverses params; if leaf key exists in `args`, sets and removes from `args`.
  - Unmatched keys remain in `args`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/src/hyperparams.rs`.
- Rust public API surface: `Params`, `ParamValue`, `InstantiableParams`, `copy_params_to`, `update_params`, `ParamsFactory`.
- Data model mapping:
  - Python `_Param` → Rust `Param` (name/value/description).
  - Python `Params` → Rust `Params` with `BTreeMap` storage.
  - Python raw values → Rust `ParamValue` (including `External` for TF tensors / opaque handles).
- Feature gating: none currently; TF handle storage likely lives in `ParamValue::External` (tie into `monolith-tf` if needed).
- Integration points: `base_layer.rs`, `base_task.rs`, `base_model_params.rs`, `model_registry.rs`.

**Implementation Steps (Detailed)**
1. Align Rust `Params` error messages with Python (TypeError/AttributeError strings).
2. Match `__str__` formatting (sorted keys, indent, list/dict rendering, quoting).
3. Implement `_similar_keys` overlap logic and error hints exactly.
4. Support list indexing in dotted paths with Python-compatible errors.
5. Preserve Tensor/opaque value deepcopy semantics (reference copy).
6. Implement `InstantiableParams.instantiate` reflection-style path or provide compatible factory shim.
7. Ensure `copy_params_to` surfaces missing keys (don’t silently drop errors).
8. Add tests mirroring `hyperparams_test.py` including exact error strings.

**Tests (Detailed)**
- Python tests: `monolith/core/hyperparams_test.py`.
- Rust tests: add `monolith-rs/crates/monolith-core/tests/hyperparams.rs`.
- Cross-language parity test: add fixture-driven JSON snapshot compare of `to_string`, errors, and nested access.

**Gaps / Notes**
- Rust currently raises `MonolithError::ConfigError` rather than `TypeError`/`AttributeError` equivalents; messages differ.
- Rust `Params` does not implement equality; Python `Params.__eq__` is relied on in tests.
- Rust `InstantiableParams` uses `ParamsFactory` (no reflection or `allowed_kwargs` handling).
- `copy_params_to` in Rust ignores `set` errors (should propagate to match Python exceptions).
- String rendering differs (Python uses sorted `_SortedDict` repr and special `Repr()` hook).
- Rust treats list navigation errors differently (`Expected list index` vs Python `Cannot introspect` assertion).

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 277
- Purpose/role: Unit tests defining required semantics for `Params` behavior, error messages, and string rendering.
- Key symbols/classes/functions: `ParamsTest`, `TestEnum`.
- External dependencies: `unittest`, `tensorflow`, `absl.flags/logging`, `monolith.core.hyperparams`.
- Side effects: none (tests only).

**Required Behavior (Detailed)**
- `test_equals`:
  - `Params.__eq__` compares only name+value; nested `Params` must compare deeply; non-Params should not be equal.
- `test_deep_copy`:
  - `Params.copy()` deep-copies nested `Params`.
  - `tf.Tensor` values are **shared** (same object identity) in copied params.
- `test_copy_params_to`:
  - Copy respects `skip` list; `dest` only contains copied keys.
- `test_define_existing`:
  - Duplicate `define` raises `AttributeError` with "already defined".
- `test_legal_param_names`:
  - Invalid names (`None`, empty, `_foo`, `Foo`, `1foo`, `foo$`) raise `AssertionError`.
  - Valid examples: `foo_bar`, `foo9`.
- `test_set_and_get`:
  - Setting undefined param via `.set` or `setattr` raises `AttributeError` mentioning name.
  - `set/get` works; `delete` removes; subsequent access raises `AttributeError`.
- `test_set_and_get_nested_param`:
  - Dotted traversal with nested `Params` works for `get`, `set`, `delete`.
  - `set` on missing path raises `AttributeError` with dotted name.
  - Attempting to navigate into non-Params (e.g., dict) raises `AssertionError("Cannot introspect ...")`.
- `test_freeze`:
  - Frozen params reject `set`, `setattr`, `delete`, `define` with `TypeError`.
  - `get` on missing still raises `AttributeError`.
  - Nested params remain mutable after parent frozen.
  - Attempt to set `_immutable` attribute raises `TypeError`.
  - `copy()` of frozen params is still immutable.
- `test_to_string`:
  - `str(params)` yields exact multi-line formatting:
    - Sorted keys.
    - Strings quoted.
    - Dicts rendered with sorted keys and nested `Params` converted to dicts.
    - Lists render nested `Params` as dicts.
- `test_iter_params`:
  - Iteration yields all keys/values (order not asserted).
- `test_similar_keys`:
  - Error message for misspelled attribute includes sorted similar keys:
    - `actuvation (did you mean: [activation,activations])`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/src/hyperparams.rs`.
- Rust public API surface: `Params`, `ParamValue`, `InstantiableParams`, `copy_params_to`, `update_params`.
- Data model mapping: implement Rust tests mirroring Python assertions and error text.
- Feature gating: none.
- Integration points: Rust tests in `monolith-rs/crates/monolith-core/tests`.

**Implementation Steps (Detailed)**
1. Add Rust unit tests that mirror each Python test case.
2. Build helper to compare error strings for invalid operations.
3. Ensure `Params` equality is implemented for Rust tests.
4. Add string formatting parity checks using exact expected text.
5. Validate frozen behavior and nested mutability.

**Tests (Detailed)**
- Python tests: TODO (manual)
- Rust tests: TODO (manual)
- Cross-language parity test: TODO (manual)

**Gaps / Notes**
- TODO (manual)

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 421
- Purpose/role: Keras layer for neural width search (NWS) over embedding sizes, with optional teacher distillation transform.
- Key symbols/classes/functions: `TeacherEmbeddingTransform`, `MixedEmbedOpComb`.
- External dependencies: `numpy`, `tensorflow`, `tensorflow.keras.layers.Layer/InputSpec`.
- Side effects: creates trainable TF variables, prints to stdout, uses TF random sampling.

**Required Behavior (Detailed)**
- `TeacherEmbeddingTransform(max_choice_per_embedding, teacher_embedding_sizes_list)`:
  - Stores arrays; asserts same length.
  - `build(input_shape)`:
    - Input is 2D; last dim equals sum(teacher sizes).
    - Creates `teacher_embedding_transform_weight` with shape `[sum(max_choice * size), 1]` initialized with `TruncatedNormal(stddev=0.15)`.
    - Creates `teacher_embedding_transform_bias` with shape `[sum(max_choice)]` initialized with zeros.
    - Calls `_snapshot_for_serving` on both (method not defined in this class).
  - `call(inputs)`:
    - Slices teacher embedding per size; for each slice, reshapes weight to `[size, max_choice]` and matmul.
    - Concats results along axis 1 and adds bias.
  - `compute_output_shape` raises `NotImplementedError`.
  - `get_config` returns max choices + teacher sizes.
- `MixedEmbedOpComb(slot_names, embedding_size_choices_list, warmup_steps, pretraining_steps, teacher_embedding_sizes_list=None, distillation_mask=False)`:
  - Asserts slot names length matches choices list; prints lengths.
  - Computes per-slot num choices, per-slot max embedding choice (sum of sizes), total embedding size, max num choices.
  - **Note:** `teacher_embedding_sizes_list` is ignored (set to `None`), so teacher path is disabled.
  - `build(input_shape)`:
    - Asserts input is 2D; verifies total input dim matches total embedding size.
    - Creates `arch_embedding_weights` variable of length sum(num_choices), init uniform(-1e-3, 1e-3).
    - For each slot:
      - Softmax over weights slice; compute entropy (softmax_cross_entropy_with_logits_v2).
      - Compute expected embedding dims as weighted sum of choice sizes.
      - If first choice size is 0, scale its prob by warmup schedule based on global step.
      - Create per-choice masks (ranges over max_emb_choice), pad to max_num_choices.
      - Pretraining: if global_step < pretraining_steps, probability hard-coded to `[0.5, 0.5]` (assumes 2 choices).
      - Sample choice via `tf.random.categorical`, one-hot, select mask.
      - Apply straight-through trick: mask * (1 + w - stop_gradient(w)).
    - Concatenates per-slot masks into `_arch_embedding_masks_multipler`.
    - Stores `_arch_entropy`, `_expected_emb_dims`, `_expected_zero_embedding_size_weights`,
      `_arch_embedding_weights_after_softmax_list`.
  - `call(inputs)`:
    - Computes `masked_embedding = embedding * _arch_embedding_masks_multipler`.
    - If teacher path active:
      - Builds `TeacherEmbeddingTransform`, transforms teacher embedding.
      - Computes distillation MSE against masked embedding (optionally with mask).
      - Returns `(mixed_embedding, distillation_loss, teacher_embedding_transform.name)` but `mixed_embedding` is undefined (bug).
    - Else returns `masked_embedding`.
  - `get_config` includes `slot_names`, `embedding_size_choices_list`, `warmup_steps`, `teacher_embedding_sizes_list` (pretraining_steps and distillation flag omitted).
  - `get_arch_embedding_weights()` returns variable; `get_summaries()` returns entropy, expected dims, and weight list.
- Determinism: sampling uses TF RNG; dependent on global step and random categorical.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-layers/src` (no existing equivalent).
- Rust public API surface: none; would need a layer module for NWS + distillation.
- Data model mapping: Keras layers + TF ops → Rust tensor ops (Candle/TF runtime).
- Feature gating: requires TF runtime or compatible tensor ops.
- Integration points: embedding layer selection / search in training pipeline.

**Implementation Steps (Detailed)**
1. Decide whether to support NWS (sampling-based) in Rust backend.
2. If yes, implement mask sampling, expected dims, and entropy summaries.
3. Add teacher distillation transform and MSE loss path.
4. Ensure global step and warmup/pretraining schedules are wired.
5. Fix Python bugs when porting (teacher path, mixed_embedding).

**Tests (Detailed)**
- Python tests: none in repo.
- Rust tests: add unit tests for mask construction, sampling shapes, and expected dims.
- Cross-language parity test: compare mask selection behavior under fixed RNG seeds.

**Gaps / Notes**
- `teacher_embedding_sizes_list` is ignored in `__init__` (always `None`).
- `call()` returns `mixed_embedding` in teacher path, but it is never defined.
- `pretraining_steps` only supports 2 choices (`[0.5, 0.5]` hard-coded).
- `_snapshot_for_serving` is called but not defined on `Layer` (likely missing mixin).

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 320
- Purpose/role: Deprecated Sail-like TPU Model wrapper for configuring TPU embedding tables, input pipelines, and pooling.
- Key symbols/classes/functions: `Model`, `create_input_fn`, `create_feature_and_table_config_dict`, `sum_pooling`, `init_slot_to_dims`.
- External dependencies: `tensorflow`, `tpu_embedding`, `absl.logging`, `math`, `FeatureSlot/FeatureColumnV1/Env`.
- Side effects: reads vocab file, logs size info, builds TF datasets/TPU table configs.

**Required Behavior (Detailed)**
- `Model.__init__(params)`:
  - Reads `vocab_size_per_slot`, logs if fixed.
  - Builds vocab dict from file; constructs `Env` and runs `init_slot_to_dims()`.
- `_create_vocab_dict(file_path, vocab_size_per_slot=None)`:
  - Reads TSV with `slot_id<TAB>count`.
  - Skips non-digit slot IDs; uses fixed vocab size if provided.
  - Returns `{slot_id: vocab_size}`.
- `_get_feature_map()`: abstract; must return TF parse feature map.
- `_post_process_example(example)`:
  - For each slot in `env.slot_to_dims`:
    - If `vocab_size_per_slot` set, mod values in `slot_{id}_0`.
    - Duplicates `slot_{id}_0` into `slot_{id}_{i}` for each additional dim.
- `create_input_fn(file_pattern, repeat=True)`:
  - Returns `input_fn(params)` using TF Dataset:
    - `list_files(shuffle=False)`, shard by `context.current_input_fn_deployment()`.
    - Optional per-shard skip from `params["shard_skip_file_number"]`.
    - `interleave(TFRecordDataset, cycle_length, num_parallel_calls, deterministic=False)`.
    - `batch(drop_remainder=True).map(parse_example, num_parallel_calls=AUTOTUNE, deterministic=False)`.
    - `repeat()` if requested; `prefetch(AUTOTUNE)`.
- `_padding_8(dim)`:
  - Rounds up to multiple of 8.
- `_get_slot_number(optimizer, use_gradient_accumulation)`:
  - Maps TPU optimizer class → slot count:
    - FTRL: 3, Adagrad: 2, Adam: 3, SGD: 1; adds 1 if gradient accumulation.
  - Else asserts unsupported optimizer (assert uses truthy string; ineffective).
- `_get_max_slot_number()`:
  - Iterates env slots and dims; chooses bias vs vec optimizer per dim; returns max slot count.
- `create_feature_and_table_config_dict()`:
  - Requires `env.is_finalized()`.
  - For each slot/dim: builds `tpu_embedding.TableConfig` and `FeatureConfig`.
  - Computes and logs embedding table sizes (raw, padded-to-8, padded+max-slot).
  - Returns `(feature_to_config_dict, table_to_config_dict)`.
- `sum_pooling(fc_dict, input_map, features, dim, total_embeddings, add_into_embeddings=True)`:
  - For each slot in `features`, calls `fc_dict[slot].add_vector(dim)`, appends to totals, populates `input_map` with unique keys.
  - Returns single embedding if one feature; else `tf.add_n`.
- `logits_fn()`, `create_model_fn()` are abstract.
- `init_slot_to_dims()` calls `logits_fn()`, `env.finalize()`, logs slot dims.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src` (no direct equivalent).
- Rust public API surface: none; core training uses different abstractions.
- Data model mapping: TF TPU embedding configs have no Rust analog.
- Feature gating: requires TF runtime backend.
- Integration points: input pipelines, embedding table config, TPU training loop.

**Implementation Steps (Detailed)**
1. Decide whether to port deprecated Model API or replace with `base_embedding_task` parity.
2. If ported, implement vocab dict parsing, dataset sharding, and TFRecord pipeline equivalents.
3. Add TPU embedding config generator or document unsupported feature in Rust.
4. Implement pooling helper and slot/embedding table sizing logs.

**Tests (Detailed)**
- Python tests: none in repo.
- Rust tests: none (integration tests would be required with TF runtime).
- Cross-language parity test: compare table config dicts and size calculations.

**Gaps / Notes**
- File is marked deprecated; likely superseded by `base_embedding_task.py`.
- `Env` constructor here omits `params` argument (signature mismatch with current `Env`).
- `_get_slot_number` uses `assert("message")` which never fails; should raise.
- Depends on `env.slot_to_dims` and `env.slot_to_config` which do not exist in current `Env` implementation.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 104
- Purpose/role: Dynamic import helpers for task/model parameter modules.
- Key symbols/classes/functions: `_Import`, `ImportAllParams`, `ImportParams`.
- External dependencies: `importlib`, `absl.logging`.
- Side effects: imports Python modules dynamically; logs import results.

**Required Behavior (Detailed)**
- `_Import(name)`:
  - Logs attempt; imports module; logs success; returns True on success.
  - On ImportError, logs error and returns False.
- `ImportAllParams(task_root=_ROOT, task_dirs=_DIRS, require_success=False)`:
  - For each `task` in `task_dirs`, attempts to import `{task_root}.{task}.params.params`.
  - If `require_success` and nothing imported, raises `LookupError`.
  - Note: code defines `module_str` with `path` but `path` is undefined (bug); actual import uses `.params.params`.
- `ImportParams(model_name, task_root, task_dirs, require_success=True)`:
  - Expects `model_name` to contain a dot; else ValueError.
  - Extracts `model_module` and attempts to import it directly.
  - For built-in tasks, if `model_module` starts with `{task}.`, builds module path `{task_root}.{task}.params.{path}` and attempts import.
  - If `require_success` and no import succeeded, raises LookupError with guidance.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/src/model_imports.rs`.
- Rust public API surface: helper functions to load/register model param types (likely via registry rather than dynamic import).

**Implementation Steps (Detailed)**
1. Provide a Rust registry-based import mechanism (explicit registration, no dynamic import).
2. Mirror error messages and logging at call sites.
3. Document that Python dynamic import is replaced by static registration.

**Tests (Detailed)**
- Python tests: none specific.
- Rust tests: verify registry lookup error messages when missing.

**Gaps / Notes**
- Python uses dynamic import; Rust should use explicit registration or plugin loading if required.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 174
- Purpose/role: Global registry for `SingleTaskModelParams` classes with dynamic import helpers.
- Key symbols/classes/functions: `_ModelRegistryHelper`, `RegisterSingleTaskModel`, `GetAllRegisteredClasses`, `GetClass`, `GetParams`.
- External dependencies: `absl.logging`, `model_imports`, `SingleTaskModelParams`.
- Side effects: stores classes in global dict; logs registration; raises on duplicates.

**Required Behavior (Detailed)**
- Registry maps keys to param classes. Key format: `<module>.<ClassName>`; also a shortcut key with `monolith.tasks.` prefix stripped and `params.` removed.
- `_RegisterModel(src_cls)`:
  - Registers both full and shortcut keys.
  - Raises ValueError on duplicate key.
  - Logs module registration once per module.
- `RegisterSingleTaskModel` decorator:
  - Validates subclass of `SingleTaskModelParams` else TypeError.
  - Registers class and returns it.
- `GetAllRegisteredClasses()`:
  - Logs warning if none registered.
- `GetClass(class_key)`:
  - Raises LookupError if missing; logs known models.
- `GetParams(class_key)`:
  - Instantiates model params class, calls `.task()` to get task config.
- Module-level helpers call `model_imports.ImportAllParams` or `ImportParams` before returning registry results.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/src/model_registry.rs`.
- Rust public API surface: registry with registration macro/trait and lookup by string key.

**Implementation Steps (Detailed)**
1. Implement a global registry (static map) for model param factories.
2. Provide a registration macro that enforces trait bounds.
3. Implement lookup with detailed error listing known keys.
4. Replace dynamic import with explicit registration in Rust.

**Tests (Detailed)**
- Python tests: none specific.
- Rust tests: register dummy models, verify duplicate detection and lookup errors.

**Gaps / Notes**
- Shortcut key behavior must be preserved for compatibility with existing names.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 25
- Purpose/role: Maps string names to TensorFlow v1 optimizer classes.
- Key symbols/classes/functions: `optimizers` dict.
- External dependencies: `tf.compat.v1.train` optimizers.
- Side effects: none.

**Required Behavior (Detailed)**
- `optimizers` contains:
  - `'adagrad'`: `AdagradOptimizer`
  - `'momentum'`: `MomentumOptimizer`
  - `'rmsprop'`: `RMSPropOptimizer`
  - `'adam'`: `AdamOptimizer`

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-optimizer/src/lib.rs`.
- Rust public API surface: map from string to optimizer factory or enum.

**Implementation Steps (Detailed)**
1. Define optimizer registry in Rust with equivalent names.
2. Map to native implementations or TF bindings if used.

**Tests (Detailed)**
- Python tests: none.
- Rust tests: verify lookup table matches expected keys.

**Gaps / Notes**
- Rust may not support all TF optimizers natively; document fallbacks.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 313
- Purpose/role: NestedMap container with attribute access, flatten/pack utilities, and validation.
- Key symbols/classes/functions: `NestedMap` and its methods.
- External dependencies: `six`, `re`, `numpy` (imported but unused here).
- Side effects: enforces key validation on set.

**Required Behavior (Detailed)**
- Keys must be valid identifiers (regex `[A-Za-z_][A-Za-z0-9_]*`) and not reserved dict attributes.
- Attribute access maps to items; missing keys raise AttributeError listing available attributes.
- `GetItem` and `Get` support dotted key paths; no list indexing.
- `Set` creates nested maps along dotted path; raises if intermediate is not map/dict.
- `Flatten`, `FlattenItems`, `Pack`, `Transform`, `IsCompatible`, `Filter`, `FilterKeyVal`, `DebugString`, `VLog` mirror Lingvo-style NestedMap.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/src/py_utils.rs`.
- Rust public API surface: `NestedMap` with typed key validation and traversal helpers.

**Implementation Steps (Detailed)**
1. Implement NestedMap with key validation and attribute-like access (maybe via methods).
2. Provide flatten/pack utilities preserving order (sorted keys).
3. Implement filter logic with delete sentinel behavior.

**Tests (Detailed)**
- Python tests: indirect via core layers.
- Rust tests: unit tests for key validation, Get/Set, Flatten/Pack compatibility.

**Gaps / Notes**
- Python uses dynamic attributes; Rust should expose explicit methods.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 203
- Purpose/role: Test helper for Keras/BaseLayer compatibility; adapted from TF Keras testing_utils.
- Key symbols/classes/functions: `layer_test`.
- External dependencies: TF internal Keras/test utilities.
- Side effects: builds models, runs predict, validates shapes/dtypes.

**Required Behavior (Detailed)**
- `layer_test`:
  - Generates `input_data` if not provided; uses `input_shape` and `input_dtype` defaults.
  - Instantiates layer with `kwargs`; optionally calls `adapt`.
  - Tests `get_weights`/`set_weights` and re-instantiation with `weights` kwarg.
  - Builds functional model `y = layer(x)` and checks output dtype.
  - Checks expected output shape and computed output signature.
  - Runs `model.predict` and compares output to expected if provided.
  - Chooses assertion method based on dtype (string vs numeric).

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/tests/testing_utils.rs`.
- Rust public API surface: test helper for layer validation (if Rust layers exist).

**Implementation Steps (Detailed)**
1. Implement a minimal test harness for Rust layers with shape/dtype checks.
2. If Rust uses different layer API, map semantics accordingly.

**Tests (Detailed)**
- Python tests: used by layer tests in core.
- Rust tests: provide helper functions for layer unit tests.

**Gaps / Notes**
- TF internal APIs used here are not available in Rust; might require simplified harness.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 214
- Purpose/role: ReplicatedVariable wrapper for TPU context handling and resource variable ops.
- Key symbols/classes/functions: `ReplicatedVariable`, `_enclosing_tpu_context`, `_tensor_conversion`.
- External dependencies: TF internal ops.
- Side effects: registers tensor conversion function globally.

**Required Behavior (Detailed)**
- `_enclosing_tpu_context()` walks control flow contexts to find XLA TPU context.
- `ReplicatedVariable`:
  - Wraps list of per-replica variables; exposes `handle` that is replicated in TPU context.
  - Implements `assign`, `assign_add`, `assign_sub`, `read_value` using resource ops.
  - Provides `initializer`, `dtype`, `shape`, `get_shape`, `to_proto` by delegating to primary var.
  - `_should_act_as_resource_variable` is a no-op placeholder.
- Registers tensor conversion function and dense tensor like type (pre-TF2.3).

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-tf/src/tpu_variable.rs`.
- Rust public API surface: replicated variable wrapper if TF TPU is supported.

**Implementation Steps (Detailed)**
1. Provide wrapper around per-replica variables with TPU context awareness.
2. Implement assign/read operations using TF runtime or stub.
3. Ensure tensor conversion registration or equivalent behavior in Rust.

**Tests (Detailed)**
- Python tests: none.
- Rust tests: optional integration tests if TPU runtime is available.

**Gaps / Notes**
- TF TPU context is highly specific; Rust likely needs a bridge or stubs.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 269
- Purpose/role: GCS utilities, checkpoint reload helpers, and dataset date filtering.
- Key symbols/classes/functions: `download_gcs_file`, `update_params`, `range_dateset`.
- External dependencies: `google.cloud.storage`, `tensorflow.compat.v1`, `gsutil` via subprocess.
- Side effects: network calls to GCS, subprocess execution, logging.

**Required Behavior (Detailed)**
- `get_bucket_name_and_relavite_path(gs_file_path)` parses `gs://bucket/path` into bucket + relative path.
- `download_gcs_file` and `download_gcs_file_with_relative_path` fetch blobs to local filename.
- `list_gcs_files_with_prefix` returns bucket and blob iterator for prefix.
- `parse_example_number_meta_file` reads `file,count` lines (comma separated) and ensures file names are ascending.
- `calculate_shard_skip_file_number` computes per-shard skip counts based on completed steps and batch sizes.
- `get_checkpoint_completed_step_number` scans GCS checkpoint `.meta` files to find max step.
- `update_params`:
  - Computes batch sizes based on shard count (TPU workers) and validates consistency.
  - Uses checkpoint step to compute `shard_skip_file_number` based on meta file.
- `get_per_file_example_numbers_for_checkpoint_reload`:
  - Uses `gsutil ls` on dataset path, checks ordering, and matches against meta list.
- `range_dateset` filters dataset elements by date substring between start/end dates.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/src/util.rs`.
- Rust public API surface: GCS helpers and dataset date filtering.

**Implementation Steps (Detailed)**
1. Implement GCS helpers (use Google Cloud Storage SDK or `gsutil` fallback).
2. Port checkpoint reload logic and shard skip calculations.
3. Implement dataset date range filter (likely for TF dataset bridge).

**Tests (Detailed)**
- Python tests: `monolith/core/util_test.py`
- Rust tests: replicate range_dateset filtering behavior.

**Gaps / Notes**
- `gsutil` subprocess usage may need replacement or explicit dependency in Rust.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 149
- Purpose/role: Tests `range_dateset` date filtering behavior.
- Key symbols/classes/functions: `UtilTest.test_range_dataset_*`.
- External dependencies: `tensorflow.compat.v1`.
- Side effects: none.

**Required Behavior (Detailed)**
- Tests that `range_dateset` filters dataset elements based on date substring in path.
- Covers single date, multiple dates, out-of-bound ranges, missing start or end date.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/tests/util.rs`.
- Rust public API surface: `range_dateset` equivalent.

**Implementation Steps (Detailed)**
1. Port tests using a mock dataset list and filter function.
2. Ensure output order and content match Python.

**Tests (Detailed)**
- Python tests: this file
- Rust tests: same scenarios.

**Gaps / Notes**
- Requires a dataset abstraction; may be tested with plain lists in Rust.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 188
- Purpose/role: Numpy-based variance scaling initializer (truncated/untruncated normal or uniform).
- Key symbols/classes/functions: `_compute_fans`, `VarianceScaling.__call__`, `get_config`.
- External dependencies: `numpy`, `scipy.stats.truncnorm`.
- Side effects: uses NumPy RNG (seeded per call).

**Required Behavior (Detailed)**
- `_compute_fans(shape, data_format)` computes fan_in/fan_out for dense or conv shapes; supports `channels_first/last`.
- `VarianceScaling.__init__` validates `scale > 0`, `mode` in `fan_in/fan_out/fan_avg`, distribution in `truncated_normal/untruncated_normal/uniform`.
- `__call__(shape, dtype)`:
  - Computes scaled variance based on mode.
  - Seeds NumPy RNG with `self.seed` each call.
  - For `truncated_normal`: uses scipy truncnorm with cutoff ±2 stddev, stddev adjusted by constant 0.87962566103423978.
  - For `untruncated_normal`: uses `np.random.normal`.
  - For `uniform`: uses `np.random.uniform` with limit `sqrt(3*scale)`.
- `get_config()` returns dict with scale/mode/distribution/seed.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/src/variance_scaling.rs`.
- Rust public API surface: `VarianceScaling` struct implementing initializer to produce arrays.

**Implementation Steps (Detailed)**
1. Port `_compute_fans` logic and mode handling.
2. Implement truncated normal sampling (use rand_distr or custom truncation) matching SciPy behavior.
3. Ensure seeding behavior matches (seed per call).

**Tests (Detailed)**
- Python tests: none.
- Rust tests: sample shape and distribution sanity checks; seed determinism.

**Gaps / Notes**
- Matching SciPy truncnorm exactly may require careful implementation.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 226
- Purpose/role: GPU training/eval runner for TF Estimator, with optional Horovod multi-GPU support.
- Key symbols/classes/functions: `GPURunner`, `create_estimator`, `run`, CLI `main`.
- External dependencies: `tensorflow`, `horovod`, `mpi4py`, `absl.flags`, `model_registry`.
- Side effects: initializes Horovod, configures GPU visibility, trains/evaluates model, writes summaries.

**Required Behavior (Detailed)**
- CLI flags: `task`, `model_dir`, `save_checkpoints_steps`, `mode` (`train|eval|train_and_eval`).
- `GPURunner.__init__`:
  - Reads flags and task_param; sets `_mode`.
- `create_estimator(model_fn)`:
  - If `task_param.accelerator == 'horovod'`:
    - `hvd.rank()` controls checkpoint saving (only rank 0).
    - Configures `tf.compat.v1.ConfigProto` with XLA JIT ON and GPU memory growth.
    - Sets `visible_device_list` to local rank.
    - `num_gpus = hvd.size()`.
  - Else: `num_gpus = 1` and uses `tf.compat.v1.estimator.RunConfig`.
  - Returns `tf.compat.v1.estimator.Estimator` with params: train/eval batch sizes, accelerator, num_replicas, hvd_rank.
- `run()`:
  - Loads global step (or 0).
  - Instantiates task; builds input_fn_train/eval and model_fn.
  - If horovod: `hvd.init()`, sets visible GPU.
  - For `train`:
    - Horovod: uses `BroadcastGlobalVariablesHook(0)`.
    - Non-horovod: `est.train`.
  - For `eval`:
    - Computes `num_steps` from `eval_examples` and batch size.
    - Runs `est.evaluate` and writes summary under `model_dir/eval`.
  - For `train_and_eval`:
    - Loop train for `steps_per_eval` up to max_steps, evaluate and write summary.
    - Horovod uses MPI barrier after each eval cycle.
- `main`: fetches task params from registry and runs `GPURunner`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/gpu_runner.rs` (or CLI bin).
- Rust public API surface: runner struct + CLI entrypoint.
- Data model mapping: task params, estimator equivalent (likely still Python-backed or TensorFlow runtime binding).
- Feature gating: `tf-runtime` and `horovod` (if supported) features.

**Implementation Steps (Detailed)**
1. Recreate CLI flags and task registry lookup in Rust.
2. Decide how to execute training: native Rust pipeline or Python/TF bridge.
3. If bridging TF, maintain Horovod initialization and GPU pinning logic.
4. Mirror evaluation loop, summary writing, and MPI barrier behavior.

**Tests (Detailed)**
- Python tests: none specific.
- Rust tests: integration tests for CLI argument parsing and control flow (mocked task).

**Gaps / Notes**
- Full parity likely requires TF Estimator execution; consider keeping Python runner or embedding TF runtime.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 31
- Purpose/role: Placeholder alert manager module defining a flag and stub accessor.
- Key symbols/classes/functions: `get_default_alert_manager()`.
- External dependencies: `absl.flags`, `absl.logging`, `google.protobuf.text_format`, `threading`, `time`, `traceback`.
- Side effects: defines `--monolith_alert_proto` flag at import time.

**Required Behavior (Detailed)**
- Flag definition:
  - `monolith_alert_proto` (string), default `""`, description: "The text format of alert proto."
- `get_default_alert_manager()`:
  - Returns `None`.
- No other behavior implemented.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src` (if alerts are ported).
- Rust public API surface: optional alert manager accessor + config flag.
- Data model mapping: map CLI flag to config struct; no runtime behavior.
- Feature gating: none.
- Integration points: training runner or alert subsystem (not present).

**Implementation Steps (Detailed)**
1. Add a config flag or CLI option mirroring `monolith_alert_proto`.
2. Provide a stub `get_default_alert_manager` returning `None`/`Option::None`.

**Tests (Detailed)**
- Python tests: none.
- Rust tests: none required (flag parsing only).
- Cross-language parity test: not applicable.

**Gaps / Notes**
- Module is largely empty; functionality likely lives elsewhere (or removed).

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 32
- Purpose/role: Placeholder test module; no actual tests defined.
- Key symbols/classes/functions: none (only `__main__` entrypoint).
- External dependencies: `absl.testing`, `absl.flags/app`, `google.protobuf.text_format`.
- Side effects: running as main executes `absltest.main()`.

**Required Behavior (Detailed)**
- No tests; `absltest.main()` called if executed.

**Rust Mapping (Detailed)**
- Target crate/module: none.
- Rust public API surface: none.
- Data model mapping: none.
- Feature gating: none.
- Integration points: none.

**Implementation Steps (Detailed)**
1. No Rust tests required unless alert manager gains functionality.

**Tests (Detailed)**
- Python tests: none.
- Rust tests: none.
- Cross-language parity test: not applicable.

**Gaps / Notes**
- No tests defined; consider removing or adding real coverage if alerting is implemented.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 158
- Purpose/role: TF-based barrier primitive to block non-chief workers until chief releases barrier; includes SessionRunHook integration.
- Key symbols/classes/functions: `BarrierOp`, `BarrierHook`, `BarrierAlreadyPlacedError`.
- External dependencies: `tensorflow`, `absl.logging`, `threading`, `time`, `basic_restore_hook` (imported, unused).
- Side effects: creates TF variables/placeholders, sleeps, logs every 60 seconds.

**Required Behavior (Detailed)**
- `BarrierAlreadyPlacedError`: raised when attempting to place a barrier twice.
- `BarrierOp(capacity, is_chief=True, wait_seconds=1, name_prefix="default", barrier_callbacks=None)`:
  - Creates `barrier_var` bool vector of length `capacity`:
    - If `is_chief`: `LOCAL_VARIABLES`; else `VARIABLES`.
  - Creates index placeholder `_idx_ph`.
  - `_place_op` sets `barrier_var[idx] = True`; `_remove_op` sets `False`.
  - `barrier_placed_tensor` references element 0.
  - `barrier_op_action` string variable + placeholder, assign op.
  - Uses a threading lock for thread safety.
- `place_barrier(session, action="")`:
  - If barrier already placed (index 0 True), raises `BarrierAlreadyPlacedError`.
  - Sets barrier at index 0 and assigns action; runs callbacks `(action, session)`.
- `remove_barrier(session)`:
  - Clears barrier at index 0; no checks.
- `is_barrier_placed(session)`:
  - Returns `session.run(barrier_placed_tensor)`.
- `wait_until_barrier_removed(session, index)`:
  - Validates `index` in `(0, capacity)` else `ValueError`.
  - Sets barrier at `index`, reads action, runs callbacks.
  - Loops until `barrier_placed_tensor` becomes False, sleeping `wait_seconds` and logging every 60s.
  - Removes its own barrier index after release.
- `is_all_blocked`/`is_none_blocked`:
  - Reads `barrier_var` and checks count equals capacity or 0.
- `get_unblocked_indices`/`get_blocked_indices`:
  - Returns indices with False/True in barrier vector.
- `BarrierHook(index, barrier_op)`:
  - `before_run` requests `barrier_placed_tensor`.
  - `after_run`: if `index > 0` and barrier placed, calls `wait_until_barrier_removed`.
- Threading: lock guards state changes; waiting uses sleep polling.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/barrier.rs`.
- Rust public API surface: `Barrier` trait, `InProcessBarrier`, `PsBarrier` (async).
- Data model mapping: TF variable barrier → async barrier/coordination via PS; no SessionRunHook equivalent.
- Feature gating: none (used in distributed training).
- Integration points: `runner.rs`, `distributed_ps.rs`.

**Implementation Steps (Detailed)**
1. Decide whether a TF-variable barrier is needed in Rust (likely no).
2. Map barrier semantics to async barrier APIs (waiting, timeouts).
3. Provide hook-like integration if training loop expects per-step blocking.

**Tests (Detailed)**
- Python tests: `monolith/native_training/barrier_ops_test.py`.
- Rust tests: add async barrier tests for arrival/release semantics.
- Cross-language parity test: not applicable unless TF runtime added.

**Gaps / Notes**
- TF `basic_restore_hook` import unused.
- Python barrier uses polling on tensor 0; Rust barrier is event-based and may not mirror per-step hook semantics.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 104
- Purpose/role: Verifies BarrierOp placement/removal and BarrierHook blocking behavior.
- Key symbols/classes/functions: `BarrierOpsTest`.
- External dependencies: `tensorflow`, `monitored_session._HookedSession`.
- Side effects: spawns threads and TF sessions.

**Required Behavior (Detailed)**
- `test_basic`:
  - Place barrier; double place raises `BarrierAlreadyPlacedError`.
  - Remove barrier → `is_barrier_removed` true.
- `test_barrier_hook_not_blocked`:
  - Without barrier, hook does not block; global step reaches 5.
- `test_barrier_hook_blocked`:
  - Place barrier; worker thread blocks after 1 step.
  - Callback called with action, sets a variable to True.
  - Removing barrier allows training to finish; all barriers cleared.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/barrier.rs`.
- Rust public API surface: barrier trait/implementations.
- Data model mapping: no SessionRunHook analogue; tests should exercise async barrier directly.
- Feature gating: none.
- Integration points: training runner.

**Implementation Steps (Detailed)**
1. Add Rust tests for arrival/release semantics using in-process barrier.
2. Add integration test for PS barrier timeout if applicable.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: add async barrier tests for arrival/release semantics.
- Cross-language parity test: not applicable.

**Gaps / Notes**
- Rust barrier is async/event-based; no direct SessionRunHook equivalent.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 72
- Purpose/role: SessionRunHook that invokes listener callbacks around checkpoint restore (no actual restore logic).
- Key symbols/classes/functions: `CheckpointRestorerListener`, `CheckpointRestorerHook`.
- External dependencies: `tensorflow.python.training.session_run_hook`, `absl.logging`.
- Side effects: logs on creation and restore calls.

**Required Behavior (Detailed)**
- `CheckpointRestorerListener`:
  - Interface with `begin`, `before_restore(session)`, `after_restore(session)`, `end(session)`; all no-ops by default.
- `CheckpointRestorerHook(listeners=None)`:
  - Logs "Create CheckpointRestorerHook."
  - `begin()` calls `listener.begin()` for each listener.
  - `after_create_session(session, coord)` calls `_restore(session)`.
  - `_restore(session)`:
    - Logs "Calling checkpoint restorer listeners."
    - Calls `before_restore(session)` then `after_restore(session)` on each listener.
    - **No actual restore actions performed in hook**.
- No `end()` implementation in hook; listener `end()` is never invoked.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/hooks.rs`.
- Rust public API surface: `Hook` trait + `CheckpointHook` (save only).
- Data model mapping: listener callbacks map to hook lifecycle events.
- Feature gating: none.
- Integration points: training loop hook list.

**Implementation Steps (Detailed)**
1. Add a Rust hook that runs listener callbacks at session start.
2. If restore is implemented in Rust, wire callbacks before/after restore.
3. Ensure lifecycle ordering matches Python (begin → after_create_session → before/after restore).

**Tests (Detailed)**
- Python tests: `monolith/native_training/basic_restore_hook_test.py`.
- Rust tests: add hook lifecycle test in `monolith-training`.
- Cross-language parity test: compare callback ordering under identical steps.

**Gaps / Notes**
- Hook does not implement `end()`, so listener `end()` is never called.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 137
- Purpose/role: Verifies listener callbacks fire during `after_create_session` and do not repeat during training.
- Key symbols/classes/functions: `CountCheckpointRestorerListener`, `CountHook`, `CheckpointRestorerHookTest`.
- External dependencies: `tensorflow`, `session_run_hook`.
- Side effects: runs TF monitored sessions.

**Required Behavior (Detailed)**
- `test_restore_only_in_after_create_session`:
  - Listener `begin/before_restore/after_restore` called once at session creation.
  - Another hook receives `after_create_session` once before any `before_run/after_run`.
  - After training steps, listener counts remain unchanged; CountHook shows before_run/after_run/end increments.
- `test_two_listeners_with_restorer`:
  - Two listeners both receive begin/before/after restore once.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/hooks.rs`.
- Rust public API surface: hook lifecycle tests.
- Data model mapping: count callbacks during `on_start` or equivalent.
- Feature gating: none.
- Integration points: hook list execution order.

**Implementation Steps (Detailed)**
1. Add a Rust test that runs a hook list and validates callback order/counts.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: add CPU tensor tests for clipping behavior and immutability.
- Cross-language parity test: compare Rust outputs vs TF for fixed inputs.

**Gaps / Notes**
- GPU-only custom ops in Python; Rust needs equivalent or fallback path.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 80
- Purpose/role: Custom gradient clipping with Monolith GPU ops and optional fused path.
- Key symbols/classes/functions: `_global_norm`, `clip_by_global_norm`.
- External dependencies: `tensorflow`, `device_utils`, `gen_monolith_ops` (custom ops).
- Side effects: none (pure tensor ops), but relies on custom TF ops.

**Required Behavior (Detailed)**
- `_global_norm(t_list)`:
  - Returns `None` if list empty.
  - Uses `gen_monolith_ops.global_l2_reduce` to compute L2 sum, then `sqrt`.
- `clip_by_global_norm(t_list, clip_norm, use_norm=None)`:
  - Requires `t_list` to be a list; else raises `TypeError("t_list should be a list")`.
  - If `t_list` empty, returns `(t_list, 0)`.
  - If `use_norm` provided: returns `monolith_clip_by_global_norm(t_list, use_norm, clip_norm)` and `use_norm`.
  - If in GPU placement context: uses fused op `monolith_clip_by_global_norm_fused(t_list, clip_norm)` (returns list, norm).
  - Otherwise computes `global_norm` via:
    - `_global_norm` if GPU context, else `tf.linalg.global_norm`.
  - Applies `monolith_clip_by_global_norm(t_list, global_norm, clip_norm)` and returns `(list_clipped, global_norm)`.
- Expected semantics match `tf.clip_by_global_norm` including NaN on inf norms.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-optimizer/src` (or `monolith-tensor`).
- Rust public API surface: clip-by-global-norm helper in optimizer utilities.
- Data model mapping: custom TF ops → native tensor ops (Candle/TF runtime).
- Feature gating: GPU vs CPU paths should be explicit.
- Integration points: optimizer step and gradient pre-processing.

**Implementation Steps (Detailed)**
1. Implement global norm computation and clipping on tensor lists.
2. Provide GPU-optimized path or document as CPU-only if missing.
3. Ensure NaN propagation on inf norms matches TF.
4. Add compatibility test vs `tf.clip_by_global_norm`.

**Tests (Detailed)**
- Python tests: `monolith/native_training/clip_ops_test.py`.
- Rust tests: add numeric tests for clip_by_global_norm including NaN/inf cases.
- Cross-language parity test: compare outputs against TF for fixed inputs.

**Gaps / Notes**
- Custom ops (`monolith_clip_by_global_norm_*`) are not available in Rust.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 92
- Purpose/role: Validates custom clip-by-global-norm against TF behavior and input immutability.
- Key symbols/classes/functions: `ClipOpsTest`, `NormOpsTest`.
- External dependencies: `tensorflow`, `numpy`, `test_util`.
- Side effects: runs GPU-only tests if available.

**Required Behavior (Detailed)**
- `ClipOpsTest._test_clip_by_global_norm`:
  - Runs op on GPU; compares output to TF `clip_by_global_norm` (unless expected provided).
  - Asserts input tensors are not modified in-place.
- `test_clip_by_global_norm`:
  - Covers simple, uneven shapes, no clipping, zero norm, inf -> NaN, and large random grads.
- `NormOpsTest`:
  - `test_it` (GPU only) checks `_global_norm` for inf and known values.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-optimizer/tests`.
- Rust public API surface: clip-by-global-norm tests.
- Data model mapping: use CPU tensors unless GPU backend exists.
- Feature gating: GPU-only tests optional.
- Integration points: optimizer utilities.

**Implementation Steps (Detailed)**
1. Add CPU tests matching the Python expected outputs.
2. Add NaN/inf propagation test.
3. Add randomized large-shape test if memory allows.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: add temp-dir PS file round-trip test.
- Cross-language parity test: not applicable.

**Gaps / Notes**
- Rust uses different discovery stack; PS file persistence may not be needed.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 184
- Purpose/role: Build TF cluster specs for distributed training and manage PS discovery persistence.
- Key symbols/classes/functions: `generate_session_config`, `get_training_cluster`, `_query_ps_cluster`, `_query_chief_addr`, `_save_ps_cluster_to_file`, `_fetch_ps_cluster_from_file`.
- External dependencies: `tensorflow`, `absl.logging`, `ServiceDiscovery`, `metric.cli`.
- Side effects: reads/writes PS cluster file via `tf.io.gfile`, sleeps on retries, emits metrics.

**Required Behavior (Detailed)**
- `emit_store(name, value, tagkv=None)`:
  - Delegates to `_MCLI.emit_store` (metrics).
- `generate_session_config(cluster_and_task=None)`:
  - If `None`, returns `ConfigProto(allow_soft_placement=True)`.
  - Else builds `ClusterSpec` and `ConfigProto(cluster_def=...)` with:
    - `device_filters` including `/job:ps` and `/job:chief`; non-chief adds its own job/task filter.
  - Sets `share_cluster_devices_in_session=True`.
  - Sets `experimental.share_session_state_in_clusterspec_propagation=True`.
  - Disables Grappler meta optimizer.
- `get_training_cluster(...)`:
  - If `index == 0` (chief):
    - If `num_redundant_ps`: try reading PS addrs from file (timeout=0); if insufficient, query discovery then save to file.
    - Else query discovery for PS addrs.
    - Builds cluster with `chief=[worker_addr]`, `worker=fake_worker_list`, `ps=ps_addrs`.
    - `task={"type":"chief","index":0}`.
  - Else (worker):
    - Gets `chief_addr` via `_query_chief_addr`.
    - Builds `fake_worker_list` of size `num_workers-1` and assigns `worker_addr` at `index-1`.
    - PS addrs from file (if redundant) or discovery.
    - `task={"type":"worker","index":index-1}`.
  - Asserts PS count equals `num_required_ps`.
- `_query_chief_addr(discovery)`:
  - Polls `discovery.query("worker")` until index 0 present; sleeps 5s between retries.
- `_query_ps_cluster(discovery, num_required_ps, model_name=None, cluster_type="stable")`:
  - Polls `discovery.query("ps")` until enough PS; logs count and emits metrics if model_name provided.
  - Returns sorted PS addresses by index, truncated to `num_required_ps`.
- `_save_ps_cluster_to_file(file_name, ps_addrs)`:
  - Writes comma-separated list to temp file and atomically renames.
- `_fetch_ps_cluster_from_file(file_name, timeout=1800)`:
  - Repeatedly attempts read; returns list if found, else empty after timeout.
- `_get_ps_cluster_file_name(model_dir, uuid)`:
  - Path: `<model_dir>/ps_cluster_dir/<uuid or "ps_info">`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/distributed.rs`, `py_discovery.rs`.
- Rust public API surface: `ClusterConfig`, `ServiceDiscovery` implementations.
- Data model mapping: TF `ClusterSpec` → Rust `ClusterConfig`; discovery polling → async discovery APIs.
- Feature gating: none.
- Integration points: distributed runner and service discovery.

**Implementation Steps (Detailed)**
1. Map discovery polling to Rust async discovery with retry/backoff.
2. Add PS cluster file persistence helpers (optional).
3. Provide cluster config builder mirroring fake worker list behavior if needed for TF_CONFIG parity.

**Tests (Detailed)**
- Python tests: `monolith/native_training/cluster_manager_test.py`.
- Rust tests: add unit test for PS file round-trip and cluster config validation.
- Cross-language parity test: compare generated cluster dictionaries for sample inputs.

**Gaps / Notes**
- Python uses fake worker list due to TF_CONFIG limitation; Rust cluster config may not need this.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 35
- Purpose/role: Tests PS cluster file persistence helpers.
- Key symbols/classes/functions: `ClusterManagerTest.testBasic`.
- External dependencies: `os`, `cluster_manager`.
- Side effects: writes temp files under `TEST_TMPDIR`.

**Required Behavior (Detailed)**
- `testBasic`:
  - Writes PS addrs to file via `_save_ps_cluster_to_file`.
  - Reads via `_fetch_ps_cluster_from_file` and asserts equality.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests`.
- Rust public API surface: PS cluster file helper tests.
- Data model mapping: same string list round-trip.
- Feature gating: none.
- Integration points: optional in discovery integration.

**Implementation Steps (Detailed)**
1. Add Rust test that writes/reads PS cluster file in a temp dir.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: TODO (manual)
- Cross-language parity test: TODO (manual)

**Gaps / Notes**
- TODO (manual)

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 149
- Purpose/role: Minimal Consul client for service lookup/register/deregister using HTTP or Unix socket.
- Key symbols/classes/functions: `Client`, `UnixHTTPConnection`, `ConsulException`.
- External dependencies: `six.moves.http_client.HTTPConnection`, `socket`, `threading`, `json`, `os`.
- Side effects: spawns a health-check thread on register; caches lookup results.

**Required Behavior (Detailed)**
- `UnixHTTPConnection(path)`:
  - Connects via UNIX domain socket to Consul.
- `Client.__init__()`:
  - Determines consul host:
    - `CONSUL_HTTP_HOST` or `TCE_HOST_IP` env vars.
    - Else uses `/opt/tmp/sock/consul.sock` if file exists.
    - Else defaults to `"127.0.0.1"`.
  - Port from `CONSUL_HTTP_PORT` or `2280`.
  - Initializes `_cache` and `_lock`.
- `lookup(name, timeout=3, cachetime=0)`:
  - If `cachetime>0` and cached entry is fresh, returns it.
  - Else uses `_lookup`, with longer timeout (30s) on cache miss.
  - Caches result with timestamp.
- `_lookup(name, timeout)`:
  - Uses Unix socket if host starts with `/`, else TCP.
  - GET `/v1/lookup/name?name=<name>&addr-family=dual-stack`.
  - If status != 200: logs error and returns `[]`.
  - Otherwise returns JSON-decoded list.
- `register(name, port, tags=None, check_script=None, host=None)`:
  - Builds payload with id `<name>-<port>`, TTL check (60s).
  - Adds tags as `["k:v"]`.
  - If `check_script` provided: replaces check with `interval=30s, script=...`.
  - Registers via PUT `/v1/agent/service/register`.
  - On non-200 → raises `ConsulException`.
  - Spawns daemon thread that periodically `GET /v1/agent/check/pass/service:<name>-<port>`; on socket error sleeps 2s and retries.
- `deregister(name, port, host=None)`:
  - PUT `/v1/agent/service/deregister/<name>-<port>`.
  - On non-200 → raises `ConsulException`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/discovery.rs`.
- Rust public API surface: `ConsulDiscovery` (stub), `ServiceDiscovery` traits.
- Data model mapping: Python client → Rust Consul discovery abstraction.
- Feature gating: `consul` feature.
- Integration points: distributed runner discovery.

**Implementation Steps (Detailed)**
1. Implement Consul HTTP client or keep stub and document missing features.
2. Add cache semantics and Unix socket support if parity required.
3. Add optional background health-check ticker.

**Tests (Detailed)**
- Python tests: `monolith/native_training/consul_test.py`.
- Rust tests: add mock HTTP tests for lookup/register/deregister (feature-gated).
- Cross-language parity test: compare HTTP request paths and payloads.

**Gaps / Notes**
- Python uses a ByteDance-specific `/v1/lookup/name` API, not stock Consul.
- Rust `ConsulDiscovery` targets standard Consul catalog APIs; endpoint mismatch.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 59
- Purpose/role: Unit tests for Consul client lookup/register/deregister using mocked HTTPConnection.
- Key symbols/classes/functions: `ConsulTest`.
- External dependencies: `unittest.mock`, `six.moves.http_client.OK`.
- Side effects: none (network mocked).

**Required Behavior (Detailed)**
- `test_lookup`:
  - Mock HTTP 200 with JSON payload; `Client.lookup` returns decoded list.
- `test_register` / `test_deregister`:
  - Mock HTTP 200; call methods without error.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests`.
- Rust public API surface: Consul discovery tests.
- Data model mapping: mock HTTP client behavior.
- Feature gating: `consul`.
- Integration points: discovery subsystem.

**Implementation Steps (Detailed)**
1. Add mocked HTTP tests for lookup/register/deregister under `consul` feature.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: TODO (manual)
- Cross-language parity test: TODO (manual)

**Gaps / Notes**
- TODO (manual)

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

### `monolith/native_training/cpu_training_test.py`
<a id="monolith-native-training-cpu-training-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 597
- Purpose/role: Comprehensive CPU training tests covering feature slots, export modes, occurrence/expire configs, distributed training, debugging server, and local train.
- Key symbols/classes/functions: `FeatureTask`, `FloatFeatureTask`, `SequenceFeatureTask`, `FeatureWithSlotOccurrenceThresholdTask`, `FeatureWithExpireTimeTask`, `NonFeatureTask`, `CpuTrainTest`, `DistributedTrainTest`, `LocalTrainTest`.
- External dependencies: `tensorflow`, `cpu_training`, `entry`, `feature`, `utils`, `debugging_server`, `saved_model_exporters`, `ExportMode`, `debugging_info_pb2`, `embedding_hash_table_pb2`, `ServiceDiscovery`.
- Side effects: spawns subprocesses for distributed tests; writes checkpoints and export dirs; reads debugging info files.

**Required Behavior (Detailed)**
- Shared helpers:
  - `inc_global_step_op()` increments global step and returns grouped op.
  - `FLAGS.use_native_multi_hash_table` controls hash table implementation.
- `FeatureTask`:
  - Input: ragged feature tensor.
  - Model: create slot/slice, embedding lookup; grads applied via feature factory; predict returns sum.
  - Serving input receiver uses ragged constant + placeholder for serialized input.
- `FloatFeatureTask`:
  - Uses ragged embedding + float feature; predictions from float feature; training uses embedding grads.
- `SequenceFeatureTask`:
  - Uses combiner `FeatureColumnV1.first_n(2)`; predictions from sequence feature sum.
- `FeatureWithSlotOccurrenceThresholdTask`:
  - Creates slot with `slot_id=2021`, `occurrence_threshold=3`; asserts training captures threshold map.
- `FeatureWithExpireTimeTask`:
  - Two slots with `expire_time=0` and `1`; uses `ZerosInitializer`.
  - After training, checks `_slot_to_expire_time` map and prediction values.
- `NonFeatureTask`:
  - Input dataset yields scalar; train op uses input with global step increment.
- `CpuTrainTest`:
  - `test_cpu_training_feature` basic training.
  - `test_with_misc_features`: `feature_eviction_on_save=True`.
  - `test_with_export_when_saving`: `serving.export_when_saving=True`.
  - `test_dense_only_export`: export mode `DISTRIBUTED` + `dense_only_save_checkpoints_steps=10`.
  - `test_with_prefetch_postpush`: enables variable prefetch/postpush and embedding postpush.
  - `test_cpu_training_float_feature` and `test_cpu_training_sequence_feature` run those tasks.
  - `test_cpu_training_with_slot_occurrence_threshold` checks internal threshold map.
  - `test_cpu_training_with_expire_time` checks expire time map and prediction values.
  - `test_cpu_training_non_feature` runs non-feature task.
  - `test_gpu_export`: exports saved model with remote GPU.
- `DistributedTrainTest`:
  - Spawns `cpu_training_distributed_test_binary` processes for PS/worker.
  - Tests: basic, extra_ps, redundant_ps, debugging server (case=1), temporary error (case=2).
  - `test1_with_debugging_server` waits for checkpoints then reads debugging info proto; checks variable/feature fetch via debugging server.
- `LocalTrainTest`:
  - `local_train` with and without PS.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests`.
- Rust public API surface: training loop, export pipeline, debug info tooling.
- Data model mapping: TF Estimator/FeatureFactory → Rust training/feature abstractions.
- Feature gating: depends on TF runtime and distributed runner.
- Integration points: `TrainingConfig`, export utilities, discovery and debugging.

**Implementation Steps (Detailed)**
1. Build Rust integration tests that cover: basic training, feature thresholds, expire time, and export flow.
2. Add distributed runner test harness or mark as Python-only.
3. Provide debug info export and retrieval parity tests.

**Tests (Detailed)**
- Python tests: this file plus `cpu_training_distributed_test_binary`.
- Rust tests: integration tests for training/export/debug info.
- Cross-language parity test: compare embedding predictions and debug info outputs.

**Gaps / Notes**
- Uses private internals (`training._slot_to_occurrence_threshold`, `_slot_to_expire_time`).
- Debugging server depends on proto + embedding hash table dumps not present in Rust.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 20
- Purpose/role: Package exports for dataset builders and feature utilities.
- Key symbols/classes/functions: re-exports `PBDataset`, `InstanceReweightDataset`, `NegativeGenDataset`, `PbType`, `parse_examples`, `parse_instances`, `parse_example_batch`, `filter_by_*`, `feature_combine`, `negative_sample`, `switch_slot`, `special_strategy`.
- External dependencies: internal modules `datasets`, `parsers`, `feature_utils`.
- Side effects: imports modules at package import time.

**Required Behavior (Detailed)**
- Importing package exposes symbols listed above at `monolith.native_training.data.*`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/src`.
- Rust public API surface: module re-exports for dataset and feature utils.
- Data model mapping: match Python export surface.
- Feature gating: none.
- Integration points: data pipeline and training input.

**Implementation Steps (Detailed)**
1. Re-export dataset and parser APIs in Rust modules.

**Tests (Detailed)**
- Python tests: none specific.
- Rust tests: module visibility tests.
- Cross-language parity test: not applicable.

**Gaps / Notes**
- None.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 502
- Purpose/role: End-to-end tests for PB datasets, parsing, filtering, compression, and dataset variants.
- Key symbols/classes/functions: `DataOpsTest`, `CahceOneDatasetTest`, `DecompressTest`, helper parsers.
- External dependencies: `tensorflow`, `PBDataset`, `parse_examples/instances/example_batch`, `feature_utils` filters, `ExampleBatch`, `gen_random_data_file`, `session_hooks`.
- Side effects: creates temp files under `tmp_data`; invokes external `zstd` binary; writes compressed files.

**Required Behavior (Detailed)**
- `DataOpsTest.setUpClass`:
  - Generates random instance data files (3 parts) using `gen_random_data_file`.
- `pb_dataset_target(input_pb_type, output_pb_type, filter_fn=None)`:
  - Builds `PBDataset` for Instance/Example/ExampleBatch with appropriate flags.
  - For ExampleBatch, applies `instance_reweight`.
  - Applies optional `filter_fn`.
  - Batches and maps to parser (`parse_inst_exam`/`parse_eb`).
  - Iterates through dataset and counts elements (logs count).
- Tests validate permutations:
  - Instance→Instance/Example, Example→Example/Instance, ExampleBatch→Example/Instance.
  - PLAINTEXT output for Instance/Example.
  - `filter_by_fids`, `filter_by_value` (ge/in/eq/between/str/any/all/diff), `special_strategy`.
  - `parse_example_batch` for scalar and batch inputs.
  - `PBDataset` resolves to `FilePBDataset` or `KafkaDataset` based on flags/args.
  - `testCreateInstanceDatasetHdfs` reads generated files via `PBDataset`.
  - `PBDataset.gen_patterns` returns correct date range size.
- `CahceOneDatasetTest`:
  - `CacheOneDataset` wraps dataset; second element flagged `True`.
- `DecompressTest`:
  - Creates copies of examplebatch data and tests `CompressType.ZSTD/ZLIB/GZIP`.
  - Uses `parse_example_batch` to validate decompression.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/tests`.
- Rust public API surface: dataset parsing, compression, filter utils.
- Data model mapping: TF datasets → Rust data pipeline equivalents.
- Feature gating: compression codecs and Kafka support.
- Integration points: `monolith-data` crate and training input.

**Implementation Steps (Detailed)**
1. Add Rust tests for parsing Instance/Example/ExampleBatch formats.
2. Add filter/feature utility tests mirroring Python cases.
3. Add compression decode tests (zstd/zlib/gzip).

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: dataset parsing + compression + filter tests.
- Cross-language parity test: compare parsed feature dict shapes and counts.

**Gaps / Notes**
- Requires external `zstd` binary for decompression test.
- Relies on hard-coded proto files under `monolith/native_training/data/training_instance`.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 145
- Purpose/role: Integration test for TF data service reading Parquet via `PBDataset`.
- Key symbols/classes/functions: `DataServiceTest2.testDataServiceWithParquetDataset`.
- External dependencies: `tensorflow.data.experimental.service`, `PBDataset`, `PbType`, `ExampleBatch`, `json`.
- Side effects: starts dispatcher/workers on port 7080, reads local files.

**Required Behavior (Detailed)**
- `setUpClass`/`tearDownClass`: create and destroy dispatcher + workers.
- `testDataServiceWithParquetDataset`:
  - Reads `META_JSON_PATH` and `PARQUET_DIR` env vars (defaults under `$HOME/temp`).
  - Loads meta JSON, builds column names/types mapping.
  - Creates `PBDataset` with `use_data_service=True`, `use_parquet=True`, `output_pb_type=PLAINTEXT`.
  - Registers dataset and reads from two consumers (distributed_epoch).
  - Parses `ExampleBatch` from bytes, accumulates `batch_size`.
  - Prints row count (assertion against parquet row count is commented out).

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/tests`.
- Rust public API surface: data service + parquet ingestion tests (if supported).
- Data model mapping: `PBDataset` + Parquet pipeline.
- Feature gating: Parquet and data service support.
- Integration points: dataset registration/consumers.

**Implementation Steps (Detailed)**
1. Add Rust integration test only if parquet + data service are implemented.
2. Otherwise, document as Python-only system test.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: optional integration tests.
- Cross-language parity test: compare total row counts.

**Gaps / Notes**
- Test depends on local parquet files and meta JSON; not hermetic.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 98
- Purpose/role: Tests data service split provider with `DynamicMatchingFilesDataset`.
- Key symbols/classes/functions: `DataServiceTest.testSplitProvider`.
- External dependencies: `tensorflow.data.experimental.service`, `DynamicMatchingFilesDataset`.
- Side effects: starts dispatcher/workers on port 7080.

**Required Behavior (Detailed)**
- Registers `DynamicMatchingFilesDataset` and consumes it via two distributed_epoch consumers.
- Alternates pulling items until both consumers are exhausted.
- Asserts total count equals 19.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/tests`.
- Rust public API surface: data service dataset tests.
- Data model mapping: dynamic file matching dataset.
- Feature gating: TF data service equivalent required.
- Integration points: dataset registry + consumer.

**Implementation Steps (Detailed)**
1. Only port if Rust data service is implemented.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: TODO (manual)
- Cross-language parity test: TODO (manual)

**Gaps / Notes**
- TODO (manual)

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 1642
- Purpose/role: Dataset factory and dataset classes for PB files, Kafka streams, Parquet, TFRecord, transforms, and data service.
- Key symbols/classes/functions: `PBDataset`, `FilePBDataset`, `DistributedFilePBDataset`, `KafkaDataset`, `ParquetDataset`, `InstanceReweightDataset`, `NegativeGenDataset`, `SplitFlowDataset`, `MergeFlowDataset`, `TransformDataset`, `DatasetMetaclass`, `distribute`, `merged_window`.
- External dependencies: TensorFlow internals, custom ops `gen_monolith_ops`, Kafka consumer, data service APIs, feature utils.
- Side effects: defines many flags; registers graph collections; disables iterator save/restore; spawns Kafka polling thread.

**Required Behavior (Detailed)**
- Flags defined: `data_service_dispatcher`, `dataset_use_dataservice`, `dataset_input_patterns`, `dataset_input_use_snappy`, `dataset_input_compression_type`, `dataset_input_use_parquet`, `dataset_input_use_tfrecord`, `dataset_worker_idx`, `dataset_num_workers`, `kafka_other_metadata`.
- `DatasetMetaclass.__call__`:
  - Normalizes shorthand kwargs (`topics_or_files`, `buffer_size_or_group_id`, `input_pb_type_or_servers`).
  - Expands `dataset_input_patterns` into file patterns (DATE/INT range syntax); overwrites `patterns`; removes `file_name`.
  - Forces parquet/tfrecord flags from global FLAGS; prevents both true.
  - If Kafka args present, returns `KafkaDataset`.
  - Otherwise returns `DistributedFilePBDataset`, `ParquetDataset`, `TFRecordDatasetWrapper`, or `FilePBDataset` based on args.
- `PBDataset`: empty init (factory via metaclass).
- `PBDataset.gen_patterns(...)`:
  - Expands date/hour ranges into path patterns.
- `DynamicMatchingFilesDataset`: uses custom op `dynamic_matching_files_dataset`.
- `ParquetDataset`: validates columns/types, sets `OUTPUT_PB_TYPE_GRAPH_KEY`, uses custom parquet op.
- `FilePBDataset`:
  - Determines input/output pb types, uses FeatureList to prune if possible.
  - Configures flags: has_sort_id, kafka_dump, lagrangex_header, compression, snappy.
  - Adds output pb type to collection; calls `pb_dataset` op.
- `DistributedFilePBDataset`:
  - Creates file list, optional dynamic sharding, data service or matching_files.
  - Supports parquet/tfrecord mapping; handles sharding by worker or explicit shard_num.
- `InstanceReweightDataset`: wraps custom op `instance_reweight_dataset` based on action priorities.
- `NegativeGenDataset`: wraps custom op `instance_negative_gen_dataset`, creates item pool.
- `SplitFlowDataset` / `MergeFlowDataset`: custom ops for flow splitting/merging.
- `KafkaDataset`:
  - Initializes Kafka resource via custom ops; uses `kafka_read_next_v2`, unbatch.
  - Sets output pb type collection and flags (sort_id, dump, lagrangex_header).
- `PyKafkaDataset`:
  - Python KafkaConsumer with background polling thread; converts strings to variant via `string_to_variant`.
- `register_dataset` / `from_dataset_id`:
  - Data service register/uncompress; external_state_policy preserved.
- `merged_window`: window and reshape dataset elements.
- `distribute`: data service integration with sync training / ps-worker queue; handles Horovod/BytePS.
- `TransformDataset`: wraps `transform_dataset` op with serialized Transform config.
- Monkey patches: `Dataset.instance_reweight`, `.negative_gen`, `.split_flow`, `.merge_flow`, `.distribute`, `.merged_window`, `.transform`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/src`.
- Rust public API surface: dataset factory + dataset operators.
- Data model mapping: TF Dataset/Variant → Rust streaming dataset abstractions.
- Feature gating: Kafka, Parquet, TFRecord, DataService, custom ops.
- Integration points: data ingestion and training input pipeline.

**Implementation Steps (Detailed)**
1. Decide which dataset sources are in-scope for Rust (file/kafka/parquet).
2. Implement dataset factory/dispatch logic and pattern expansion.
3. Provide custom ops or pure-Rust replacements for instance reweight, negative gen, transform.
4. Add data service integration or document unsupported.

**Tests (Detailed)**
- Python tests: `data_ops_test.py`, `data_service_test.py`, `negative_gen_test.py`, etc.
- Rust tests: dataset factory, pattern expansion, ops parity tests.
- Cross-language parity test: compare dataset counts and parsed outputs for fixtures.

**Gaps / Notes**
- Heavy reliance on custom TF ops; Rust needs replacements or TF runtime backend.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 186
- Purpose/role: Eager-mode dataset parsing tests for Instance/Example/ExampleBatch.
- Key symbols/classes/functions: `DataOpsTest.target`, test methods.
- External dependencies: `tensorflow`, `PBDataset`, `parse_instances/parse_examples/parse_example_batch`, `switch_slot`, `feature_combine`.
- Side effects: reads training_instance fixtures.

**Required Behavior (Detailed)**
- `target(input_pb_type, output_pb_type)`:
  - Builds `PBDataset` for Instance/Example/ExampleBatch and applies instance reweight for ExampleBatch.
  - Parses via `parse_inst_exam` or `parse_eb` depending on pb types.
  - Batches then iterates `dataset.take(5)` and asserts feature dict length ∈ {26,27}.
- `testExampleBatch2Instance`, `testExample2Instance`, `testInstance2Instance` call `target`.
- `testExampleBatch`:
  - Parses ExampleBatch to ExampleBatch via `parse_example_batch`, asserts len ∈ {26,27}.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/tests`.
- Rust public API surface: dataset parsing tests (eager mode equivalent).
- Data model mapping: dataset pipelines in Rust.
- Feature gating: none.
- Integration points: `monolith-data` parsing pipeline.

**Implementation Steps (Detailed)**
1. Add parsing tests for all pb types with fixed fixtures.
2. Ensure eager-mode behavior maps to Rust pipeline semantics.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: TODO (manual)
- Cross-language parity test: TODO (manual)

**Gaps / Notes**
- TODO (manual)

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 30
- Purpose/role: Tests custom op `extract_fid`.
- Key symbols/classes/functions: `ExtraFidTest.test_parse_search`.
- External dependencies: `tensorflow`, `gen_monolith_ops.extract_fid`.
- Side effects: none.

**Required Behavior (Detailed)**
- `extract_fid(185, 4)` must return `1153447759131936`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/tests` (or runtime ops crate).
- Rust public API surface: `extract_fid` equivalent.
- Data model mapping: custom op to Rust function.
- Feature gating: requires runtime ops.
- Integration points: feature parsing pipeline.

**Implementation Steps (Detailed)**
1. Implement `extract_fid` in Rust runtime ops.
2. Add a unit test for the exact constant output.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: add exact-match test.
- Cross-language parity test: compare output for fixed inputs.

**Gaps / Notes**
- Relies on custom TF op; no Rust implementation yet.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 409
- Purpose/role: Parses feature_list configuration and provides feature lookup, slot mapping, and feature filtering utilities.
- Key symbols/classes/functions: `Feed`, `Cache`, `Feature`, `FeatureList`, `get_feature_name_and_slot`, `add_feature`, `add_feature_by_fids`, `get_valid_features`, `is_example_batch`.
- External dependencies: `absl.flags`, `tensorflow`, `numpy`, `inspect`, `dataclasses`, `monolith.native_training.data.utils`.
- Side effects: populates global `_cache` and `_VALID_FNAMES`; adds to TF collections via `add_to_collections`.

**Required Behavior (Detailed)**
- `new_instance(cls, args)`:
  - Inspects `__init__` signature and passes only matching args.
- `Feed` dataclass:
  - `shared` parsed from truthy strings; `feature_id` cast to int.
  - `name` property returns `feed_name`.
- `Cache` dataclass:
  - `capacity`, `timeout` cast to int if string.
  - `name` property uses `cache_name` > `cache_key_class` > `cache_column` else raises.
- `Feature` dataclass:
  - Parses comma-separated string fields into lists.
  - Casts slot/shared/need_raw/feature_id/version to proper types.
  - `__str__` renders key=value terms based on type hints; bools only if True.
  - `name` strips `fc_`/`f_` prefixes and lowercases terms.
  - `depend_strip_prefix` strips prefixes for dependencies.
- `FeatureList`:
  - Stores column names, feeds, caches, features; builds slot→feature mapping.
  - Adds itself to TF collections `feature_list`.
  - `__getitem__` resolves by slot id or feature name variants (`f_`, `fc_`, dashed names).
  - `get_with_slot(slot)` returns list or empty.
  - `__contains__` supports names and slots.
  - `parse(fname=None, use_old_name=True)`:
    - Reads config file; caches results in `_cache`.
    - Parses `column_name:`, `cache_column:` and `feed/cache/feature` lines into dataclasses.
    - `use_old_name` chooses between raw feature_name or normalized name as key.
- `get_feature_name_and_slot(item)`:
  - Handles int, str, or FeatureColumn-like objects.
  - Uses `FeatureList.parse()` with fallback to slot name utility.
- `is_example_batch()`:
  - Checks `FLAGS.data_type` for example_batch.
- `add_feature` / `add_feature_by_fids`:
  - Maintains `_VALID_FNAMES`; for example_batch fids, resolves feature list by slot and version.
  - Raises if fid cannot be mapped.
- `get_valid_features()` returns list of collected features.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/src/feature_list.rs` (if implemented).
- Rust public API surface: feature list parser + lookup utilities.
- Data model mapping: config parsing + slot/feature mapping.
- Feature gating: none.
- Integration points: data parsing and column pruning.

**Implementation Steps (Detailed)**
1. Implement feature_list.conf parser with the same key parsing rules.
2. Add slot/name resolution helpers.
3. Implement example_batch feature filtering via fid decoding.

**Tests (Detailed)**
- Python tests: none (feature_list_test.py is empty).
- Rust tests: add unit tests for parsing, slot lookup, fid mapping.
- Cross-language parity test: compare parsed feature list for a fixed config file.

**Gaps / Notes**
- Uses global caches and TF collections; Rust should provide equivalent global registry if required.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 0
- Purpose/role: Empty file (no tests).
- Key symbols/classes/functions: none.
- External dependencies: none.
- Side effects: none.

**Required Behavior (Detailed)**
- None.

**Rust Mapping (Detailed)**
- Target crate/module: none.
- Rust public API surface: none.
- Data model mapping: none.
- Feature gating: none.
- Integration points: none.

**Implementation Steps (Detailed)**
1. No Rust tests required unless feature list gains tests.

**Tests (Detailed)**
- Python tests: none.
- Rust tests: TODO (manual)
- Cross-language parity test: TODO (manual)

**Gaps / Notes**
- TODO (manual)

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 1070
- Purpose/role: Feature/label filtering, transformation, and utility ops over variant tensors; mostly thin wrappers around custom `gen_monolith_ops` kernels with strict input validation and feature-registration side effects.
- Key symbols/classes/functions: `filter_by_fids`, `filter_by_feature_value`, `filter_by_value`, `add_action`, `add_label`, `scatter_label`, `filter_by_label`, `special_strategy`, `negative_sample`, `feature_combine`, `switch_slot`, `switch_slot_batch`, `label_upper_bound`, `label_normalization`, `use_field_as_label`, `create_item_pool`, `item_pool_random_fill`, `item_pool_check`, `save_item_pool`, `restore_item_pool`, `fill_multi_rank_output`, `use_f100_multi_head`, `map_id`, `multi_label_gen`, `string_to_variant`, `string_to_variant_with_transform`, `variant_to_zeros`, `kafka_resource_init`, `kafka_read_next`, `kafka_read_next_v2`, `has_variant`, `gen_fid_mask`, `tf_example_to_example`.
- External dependencies: TensorFlow, numpy, `idl.matrix.proto.line_id_pb2.LineId`, `data_op_config_pb2.LabelConf`/`TFRecordFeatureDescription`, `gen_monolith_ops` custom kernels.
- Side effects: calls `add_feature`/`add_feature_by_fids` to ensure downstream parsing includes required fields; validates/loads operand files via `tf.io.gfile.exists`; asserts on invalid inputs; builds TF ops that mutate or filter variant tensors.

**Required Behavior (Detailed)**
- `filter_by_fids(variant, filter_fids, has_fids, select_fids, has_actions, req_time_min, select_slots, variant_type)`:
  - Coerces `filter_fids`/`has_fids`/`select_fids` to `np.uint64` then `int64` list; defaults to empty lists.
  - `select_slots` defaults to empty; asserts all slots > 0.
  - If `variant_type != 'instance'`, calls `add_feature_by_fids` for all fid lists.
  - Calls `ragged_data_ops.set_filter(...)` with `has_actions or []`, `req_time_min`, `select_slots`, `variant_type` and returns variant tensor.
- `filter_by_feature_value(variant, field_name, op, operand, field_type, keep_empty, operand_filepath)`:
  - `op` must be in `{gt,ge,eq,lt,le,neq,between,in,not-in,all,any,diff,startswith,endswith}`.
  - Exactly one of `operand` or `operand_filepath` is provided; if filepath set, it must exist and `op` must be in `{in, not-in}`.
  - `field_type` must be in `{int64,float,double,bytes}`; builds `int_operand`, `float_operand`, `string_operand` based on type/op:
    - `all/any/diff` only for `int64` (operand int or list of int).
    - `between` uses a list of numbers (float/double) or ints for int64.
    - `bytes` accepts str or list of str; otherwise raises `RuntimeError("params error!")`.
  - Calls `ragged_data_ops.feature_value_filter(...)` with operands, file path, `keep_empty`, returns variant.
- `filter_by_value(variant, field_name, op, operand, variant_type, keep_empty, operand_filepath)`:
  - For `variant_type != 'instance'`, calls `add_feature('__LINE_ID__')`.
  - `field_name` must exist in `LineId` descriptor; uses proto field `cpp_type`/`has_options` to determine parsing rules.
  - Same operand vs operand_filepath exclusivity; operand file must exist; only `in/not-in` supported with filepath.
  - For repeated fields (`field.has_options`), only `all/any/diff` allowed and only integer types.
  - For `string` fields: operand must be str or list of str, else `RuntimeError("params error!")`.
  - Calls `ragged_data_ops.value_filter(...)` with `variant_type` and returns variant.
- `add_action(variant, field_name, op, operand, action, variant_type)`:
  - For `variant_type != 'instance'`, calls `add_feature('__LINE_ID__')`.
  - `op` in `{gt,ge,eq,lt,le,neq,between,in}`; field must exist in `LineId`.
  - Builds typed operands (float/int/string) based on field cpp_type; for `in/between` on integer types, operand is list of int.
  - Calls `ragged_data_ops.add_action(..., actions=[action], variant_type)`.
- `add_label(variant, config, negative_value, new_sample_rate, variant_type)`:
  - For `variant_type != 'instance'`, calls `add_feature('__LINE_ID__')`.
  - `config` is required; `new_sample_rate` must be in `(0, 1.0]`.
  - Parses `config` with `;` task separator; each task `pos_actions:neg_actions:sample_rate` (empty lists allowed). Skips empty trailing parts.
  - Builds `LabelConf` proto and calls `ragged_data_ops.add_label(..., negative_value, sample_rate=new_sample_rate)`.
- `scatter_label(variant, config, variant_type)`:
  - For `variant_type != 'instance'`, calls `add_feature('__LABEL__')` and `add_feature('__LINE_ID__')`.
  - `config` required; passes through to `ragged_data_ops.scatter_label`.
- `filter_by_label(variant, label_threshold, filter_equal, variant_type)`:
  - For `variant_type != 'instance'`, calls `add_feature('__LABEL__')`.
  - `label_threshold` must be non-empty list.
  - Calls `ragged_data_ops.filter_by_label(..., filter_equal, variant_type)` and returns boolean tensor.
- `special_strategy(variant, strategy_list, strategy_conf, variant_type, keep_empty_strategy)`:
  - For `variant_type != 'instance'`, calls `add_feature('__LABEL__')` and `add_feature('__LINE_ID__')`.
  - `strategy_conf` is optional; parses comma-separated `strategy:sample_rate` or `strategy:sample_rate:label` entries.
  - Ensures lengths consistent and each `sample_rate` in `[0,1]`.
  - Calls `ragged_data_ops.special_strategy(..., keep_empty_strategy, variant_type)`.
- `negative_sample(variant, drop_rate, label_index, threshold, variant_type, action_priority, per_action_drop_rate)`:
  - For `variant_type != 'instance'`, calls `add_feature('__LABEL__')`.
  - `action_priority` and `per_action_drop_rate` are optional strings; if both set, parse lists of actions and per-action drop rates.
  - Calls `ragged_data_ops.negative_sample(..., priorities, actions, per_action_drop_rate)`.
- `feature_combine(src1, src2, slot)`:
  - Requires `tf.RaggedTensor` inputs; calls `ragged_data_ops.feature_combine(..., fid_version=2)`.
  - If `splits[0]` is `float32`, uses `from_row_splits(values, splits[1])`; else `from_nested_row_splits`.
- `switch_slot(ragged, slot)`:
  - Requires `tf.RaggedTensor`; calls `ragged_data_ops.switch_slot(..., fid_version=2)`.
  - If `splits[0]` is `float32`, returns new ragged from row_splits; else returns `ragged.with_flat_values(values)`.
- `switch_slot_batch(variant, features, variant_type, suffix)`:
  - `features` maps feature name → `(inplace, new_slot)`; `variant_type` must be `example` or `example_batch`.
  - Builds `features`, `inplaces`, `slots` arrays; calls `ragged_data_ops.switch_slot_batch(..., suffix)`.
- `label_upper_bound(variant, label_upper_bounds, variant_type)`:
  - `label_upper_bounds` non-empty; calls `ragged_data_ops.label_upper_bound`.
- `label_normalization(variant, norm_methods, norm_values, variant_type)`:
  - `norm_methods` length must equal `norm_values`; calls `ragged_data_ops.label_normalization`.
- `use_field_as_label(variant, field_name, overwrite_invalid_value, label_threshold, variant_type)`:
  - Calls `ragged_data_ops.use_field_as_label` to overwrite labels from LineId field with optional clamping.
- Item pool ops:
  - `create_item_pool(start_num, max_item_num_per_channel, container, shared_name)` asserts `start_num >= 0`, `max_item_num_per_channel > 0`, calls `ItemPoolCreate`.
  - `item_pool_random_fill`, `item_pool_check(model_path, global_step, nshards, buffer_size)`, `save_item_pool`, `restore_item_pool` delegate to custom ops.
- `fill_multi_rank_output(variant, enable_draw_as_rank, enable_chnid_as_rank, enable_lineid_rank_as_rank, rank_num, variant_type)`:
  - For `variant_type != 'instance'`, calls `add_feature('__LINE_ID__')`.
  - Calls `ragged_data_ops.fill_multi_rank_output`.
- `use_f100_multi_head(variant, variant_type)`:
  - Pass-through to `ragged_data_ops.use_f100_multi_head`.
- `map_id(tensor, map_dict, default)`:
  - `map_dict` non-empty; passes `from_value`, `to_value`, `default` to `ragged_data_ops.MapId`.
- `multi_label_gen(variant, head_to_index, head_field, pos_actions, neg_actions, use_origin_label, pos_label, neg_label, action_priority, task_num, variant_type)`:
  - Builds `head_to_index` string (`head:idx`), computes `task_num` if unset; asserts `max_idx < task_num`.
  - If `use_origin_label`, `pos_actions` and `neg_actions` must be empty; otherwise `pos_actions` non-empty.
  - `head_field` must exist in `LineId` descriptor and be int or string.
  - Calls `ragged_data_ops.multi_label_gen(..., action_priority, pos/neg actions, labels, variant_type)`.
- `string_to_variant(...)`:
  - `variant_type` must be `instance|example|examplebatch|example_batch`; converts string tensor into variant using header flags and optional `chnids/datasources`.
- `string_to_variant_with_transform(...)`:
  - Similar to `string_to_variant` but accepts `input_type` and `output_type` for on-the-fly transforms.
- `variant_to_zeros(tensor)`:
  - Calls `ragged_data_ops.variant_to_zeros` to produce zeroed variant tensor.
- Kafka ops:
  - `kafka_resource_init(topics, metadata, input_pb_type, output_pb_type, has_sort_id, lagrangex_header, kafka_dump_prefix, kafka_dump, container, shared_name)` calls `KafkaGroupReadableInit`.
  - `kafka_read_next`/`kafka_read_next_v2` call `KafkaGroupReadableNext`/`NextV2` with poll/stream timeouts.
- `has_variant(input, variant_type)`:
  - Calls `ragged_data_ops.HasVariant`.
- `gen_fid_mask(ragged, fid)`:
  - Casts `fid` to `np.uint64` → `int64`; calls `monolith_gen_fid_mask` with row_splits and flat_values.
- `tf_example_to_example(serialized, sparse_features, dense_features, label, instance_weight)`:
  - Defaults: empty sparse/dense/label/instance_weight if None.
  - Validates no overlaps between sparse/dense/label/instance_weight; slot ids unique; each slot id in `[1, 32768)`.
  - Builds `TFRecordFeatureDescription` proto and calls `MonolithTFExampleToExample` op.
- Error semantics:
  - Many checks are `assert` (raising `AssertionError`), some raise `RuntimeError("params error!")` for invalid bytes operands.
  - Operand file must exist and be used only with `in/not-in` ops; callers rely on these preconditions.
- I/O formats:
  - Variant tensor format is custom monolith variant; string inputs for `string_to_variant*` are framed by headers (sort header or lagrangex header) and length-prefixed protos.
  - Operand file for `operand_filepath` is expected to contain serialized `example_pb2.FilterValues` (see tests).
  - `tf_example_to_example` expects TF Example serialized bytes and emits Monolith Example variant.
- Threading/concurrency:
  - No explicit threading here; concurrency behavior is inside custom ops (e.g., Kafka resources).
- Determinism/perf:
  - Performance relies on custom ops; callers expect these to be safe in `tf.data` pipelines (including parallel map/filter). Determinism depends on op implementations; keep semantics stable.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/src` for dataset/feature ops; optionally `monolith-rs/crates/monolith-tf` for TF-runtime-backed kernels.
- Rust public API surface: `feature_utils` module exposing the same function set (or a `FeatureOps` trait with backend-specific implementations).
- Data model mapping: TF Variant / RaggedTensor → Rust `Variant`/`Ragged` equivalents (likely in `monolith-data` or `monolith-tensor`).
- Feature gating: Kafka ops, TFExample conversion, item-pool ops, and label/negative sampling depend on custom kernels; gate behind a TF backend or feature flags.
- Integration points: parsing (`parsers.py`), datasets (`datasets.py`), hooks (`item_pool_hook.py`), and training pipelines/tests.

**Implementation Steps (Detailed)**
1. Enumerate all custom ops used here and decide per-op strategy: native Rust implementation vs TF runtime binding.
2. Port validation logic exactly (asserts, `RuntimeError("params error!")`, slot range checks).
3. Provide Rust equivalents for `LineId` field metadata (from `monolith-proto`) and reuse it for `filter_by_value`/`add_action`/`multi_label_gen`.
4. Implement operand file reading for `in/not-in` filters using the same `FilterValues` proto.
5. Implement Ragged feature transforms (`feature_combine`, `switch_slot`, `switch_slot_batch`) with fid-v2 rules.
6. Add feature registry side effects equivalent to `add_feature`/`add_feature_by_fids` so parsing includes required fields.
7. Implement item pool ops or wrap TF kernels; include save/restore/check.
8. Implement label ops (`add_label`, `scatter_label`, `filter_by_label`, `label_upper_bound`, `label_normalization`, `use_field_as_label`, `multi_label_gen`) with identical label-invalid sentinel values.
9. Implement `string_to_variant` framing rules (headers, length prefix, flags, chnids/datasources) and `tf_example_to_example` conversion.
10. Add Kafka resource wrappers with poll/stream timeouts and variant conversion.
11. Add Rust tests mirroring Python expectations (see `feature_utils_test.py`) and cross-language fixtures.

**Tests (Detailed)**
- Python tests: `monolith/native_training/data/feature_utils_test.py`, `data_ops_test.py`, `eager_mode_test.py` (feature_combine/switch_slot usage).
- Rust tests: `monolith-rs/crates/monolith-data/tests/feature_utils_*` for filtering, labels, switching slots, map_id, fid mask, string_to_variant, TFExample conversion.
- Cross-language parity test: run Python test fixtures and compare Rust outputs on identical serialized inputs (including FilterValues operand files).

**Gaps / Notes**
- Heavy reliance on `gen_monolith_ops`; Rust must either re-implement kernels or use TF runtime backend (optional per earlier requirement).
- `filter_by_value` and `filter_by_feature_value` behavior is used in parallel dataset filters; caching/parallel safety must match TF op semantics.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 1414
- Purpose/role: End-to-end tests for feature_utils ops over `PBDataset` pipelines, including label ops, filters, slot switching, fid masks, string-to-variant framing, and negative sampling.
- Key symbols/classes/functions: `DataOpsTest`, `pb_dataset_target`, helper generators (`generate_instance`, `write_instance_into_file`), tests for `add_action`, `add_label`, `scatter_label`, `filter_by_*`, `switch_slot_batch`, `map_id`, `multi_label_gen`, `string_to_variant`, `negative_sample`.
- External dependencies: TensorFlow, `PBDataset`, `parsers` (`parse_instances`, `parse_examples`), `example_pb2.FilterValues`, `proto_parser_pb2.Instance`, temporary file IO.
- Side effects: creates/deletes temp files, writes serialized Instance files, writes FilterValues proto files, logs sample counts.

**Required Behavior (Detailed)**
- `pb_dataset_target(...)` helper:
  - Chooses input file based on `input_pb_type` (instance/example/examplebatch fixtures under `monolith/native_training/data/training_instance`).
  - Builds `PBDataset` with header flags; optional `add_action_fn`, optional filter.
  - For ExampleBatch: applies `instance_reweight` with fixed `action_priority` and `reweight` config, `variant_type` based on output.
  - Batches, parses via `parse_instance_or_example` or `parse_example_batch`, and returns a list of `return_result_key` slices.
- Action/format conversions:
  - `test_input_instance_output_instance`: actions are `[[1,0], ...]` for two batches.
  - `test_input_instance_output_instance_add_action`: adding action 2 yields `[[1,2], ...]`.
  - `test_input_instance_output_example`: output actions `[[1,0,0], ...]`.
  - `test_input_instance_output_example_add_action`: `req_time between [1622667900,1622667911]` adds action 2 in some rows.
  - `test_input_example_output_instance` and `*_add_action` mirror the above for Example input.
  - `test_input_example_output_example` and `*_add_action` expect action arrays with 3 columns.
  - `test_input_example_batch_output_instance` and `*_add_action` expect action arrays starting with `2` and optionally `3`.
  - `test_input_example_batch_output_example` and `*_add_action` expect action arrays with 3 columns.
- `test_input_instance_output_instance_add_label`:
  - Builds a temp Instance file with deterministic action patterns.
  - Applies `add_label` with config `1,2:3:1.0;4::0.5` and then `filter_by_label`.
  - Expects total valid instances in range `[340, 360]` for `mock_batch_num=100`.
- `test_input_instance_output_instance_label_upper_bound`:
  - `label_upper_bounds=[0.5,0.5]` clamps labels to `[[0,0.5], ...]`.
- `test_input_instance_output_instance_label_normalization`:
  - `norm_methods=['scale','repow']`, `norm_values=[0.5,3]` results in labels `[[0,8], ...]`.
- `test_input_examplebatch_output_instance_use_field_as_label`:
  - Uses `sample_rate` field as label; with `overwrite_invalid_value` and `label_threshold` combinations expects:
    - threshold 0 → labels `[[1,1], ...]`.
    - threshold 1.1 with prior `label_upper_bound` → labels `[[1,1], ...]`.
    - threshold 0.9 with prior `label_upper_bound` → labels `[[0,0.5], ...]`.
- `test_input_instance_output_instance_filter_by_label_equals`:
  - With `filter_equal=False`, expects 100 batches and labels `[[0,1], ...]`.
  - With `filter_equal=True`, expects 49 batches and labels `[[0,2], ...]`.
- `test_input_instance_output_instance_scatter_label`:
  - `scatter_label_config = '100:3,200:1,300:4'` and `filter_by_label` yields 2 valid instances.
  - Labels contain invalid sentinel `-3.4028235e+38` with the selected index set to original label value.
- `test_filter_by_bytes_value`:
  - Filters `req_id` using `endswith` with `filter_by_value` (LineId) and `filter_by_feature_value` (feature list).
  - Expects 4 outputs `[[b'abckjhfjh'], [b'kjhfjh'], ...]`.
  - Parallel filter path (`dataset.map(...).filter(...)`) must preserve correctness and use cached feature index.
- `test_filter_by_float_value`:
  - Filters `video_play_time > 2.5` using `filter_by_feature_value` (`field_type='float'`).
  - Expects req_id outputs `[[b'huggfyfixyz'], [b'mbzc'], ...]`.
- `test_filter_by_value_not_in`:
  - Writes `FilterValues` proto files (bytes + int64) and uses `operand_filepath` with `in/not-in`.
  - For bytes: `not-in` filters out `hello/world`, expects `excluded/300/400` (or `300/400` when using file).
  - For int64: `in` keeps chnid `[20,30,666]`, expects did values `world/excluded/400`.
  - Both `filter_by_value` and `filter_by_feature_value` must match.
- `test_filter_by_value_all`:
  - Uses `filter_by_feature_value` with `op='all'` on `chnids` list; only `did='excluded'` passes.
- `test_map_id`:
  - `map_id({123:0,456:1,789:2}, default=-1)` transforms `[123,456,789,912]` → `[0,1,2,-1]`.
- `test_filter_by_fids`:
  - Filters instances that contain both slots 2 and 3; verifies resulting ragged values match `get_fid_v1` for indices 1..4.
- `test_multi_label_gen`:
  - Builds labels based on `head_to_index` mapping and action rules; expects label vectors with INVALID_LABEL sentinel except for the matched task.
- `test_string_to_variant`:
  - Builds framed Instance bytes (with headers); one empty record allowed; `string_to_variant` preserves shape; `variant_to_zeros` callable.
- `test_has_variant`:
  - `has_variant` returns `True` for a valid variant tensor.
- `test_switch_slot_batch`:
  - `switch_slot_batch` with mix of in-place and copy-to-suffix behavior; verifies slot IDs in resulting ragged tensors (`>> 48` equals shared slot when expected).
- `test_gen_fid_mask_int64` / `test_gen_fid_mask_int32`:
  - `gen_fid_mask(ragged, fid=3)` yields `[1.,1.,0.,0.]` for both row_splits dtypes.
- `test_negative_sample_with_positive_actions`:
  - Iterates 1000 synthetic samples, applies `negative_sample` with action priority and per-action drop rates.
  - Asserts deterministic outcomes for positive labels and specific action cases; logs drop-rate ratios for matched/mismatched actions.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/tests` plus any ops crates that implement feature_utils kernels.
- Rust public API surface: tests should call Rust equivalents of `feature_utils` functions and dataset helpers.
- Data model mapping: use Rust dataset pipelines to parse the same fixtures and assert outputs.
- Feature gating: many tests require TF-runtime-backed custom ops (string/variant parsing, label ops, switch-slot, negative sampling).
- Integration points: `monolith-data` parsing, `monolith-proto` for Instance/Example, and dataset fixtures under `monolith/native_training/data/training_instance`.

**Implementation Steps (Detailed)**
1. Port helpers to Rust test utilities: serialize `Instance` protos, write framed records (headers + lengths), and parse with Rust pipelines.
2. Copy the Python expected outputs into Rust assertions (actions arrays, label arrays, invalid label sentinel values).
3. Implement FilterValues file generation in Rust to test `operand_filepath` parity.
4. Recreate dataset pipelines for each test case (including ExampleBatch `instance_reweight`).
5. Add parallel filter test to validate thread-safety/caching behavior.
6. Keep negative-sample test deterministic for mandatory branches; log or assert ratios only if stable.

**Tests (Detailed)**
- Python tests: this file (primary reference).
- Rust tests: new `feature_utils_tests.rs` with one test per Python case + helpers.
- Cross-language parity test: run Python and Rust on same temp fixtures; compare arrays and variant validity.

**Gaps / Notes**
- Many tests depend on `monolith/native_training/data/training_instance/*.pb` fixtures; ensure these are accessible to Rust tests.
- INVALID_LABEL sentinel appears as `-3.4028235e+38` in Python output; Rust must match exact float.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 109
- Purpose/role: SessionRunHook to save/restore item pool state during training.
- Key symbols/classes/functions: `ItemPoolSaveRestoreHook`.
- External dependencies: `tensorflow`, `save_item_pool`, `restore_item_pool`, `POOL_KEY`.
- Side effects: writes/reads item pool checkpoints; logs progress.

**Required Behavior (Detailed)**
- `begin()`:
  - Retrieves pools from TF collection `POOL_KEY`.
  - Creates placeholders for save/restore steps.
  - Reads checkpoint state from `model_dir`; if present, builds `restore_item_pool` op.
  - Builds `save_item_pool` op.
- `after_create_session()`:
  - If not PREDICT:
    - Reads global step, restores item pool from checkpoint (if available) using step parsed from checkpoint path.
- `after_run()`:
  - If TRAIN and `save_steps>0`: saves pool when global step advances by `save_steps`.
- `end()`:
  - If TRAIN: saves pool once more if global step advanced.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/src` (item pool ops).
- Rust public API surface: item pool save/restore hook (if training hooks exist).
- Data model mapping: TF collections → Rust registry.
- Feature gating: item pool feature.
- Integration points: training hook lifecycle.

**Implementation Steps (Detailed)**
1. Implement save/restore of item pool state in Rust.
2. Add training hook for periodic save and restore on startup.

**Tests (Detailed)**
- Python tests: `monolith/native_training/data/item_pool_test.py`.
- Rust tests: add item pool save/restore tests.
- Cross-language parity test: compare pool state serialization.

**Gaps / Notes**
- Requires TF collection `POOL_KEY` and custom item pool ops.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 58
- Purpose/role: Tests create/save/restore/check of item pool ops.
- Key symbols/classes/functions: `ItemPoolTest.test_create_item_pool`.
- External dependencies: `tensorflow`, `feature_utils` item pool ops.
- Side effects: writes item pool files under `$HOME/<user>/tmp/monolith/data/test`.

**Required Behavior (Detailed)**
- `setUpClass`:
  - Creates item pool, randomly fills, and saves to model path with `nshards=2`.
- `test_create_item_pool`:
  - Restores pool and checks it using `item_pool_check`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/tests`.
- Rust public API surface: item pool ops tests.
- Data model mapping: same save/restore workflow.
- Feature gating: item pool support.
- Integration points: data utils.

**Implementation Steps (Detailed)**
1. Add Rust unit test that saves/restores item pool and validates contents.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: TODO (manual)
- Cross-language parity test: TODO (manual)

**Gaps / Notes**
- TODO (manual)

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 239
- Purpose/role: Integration test for KafkaDataset ingestion and label parsing.
- Key symbols/classes/functions: `start_producer`, `KafkaDatasetTest.test_kafka_dataset`.
- External dependencies: `kafka.KafkaProducer`, `tensorflow`, `KafkaDataset`, `parse_instances/parse_examples`, `add_label`.
- Side effects: produces Kafka messages to a real cluster; sleeps and joins producer thread.

**Required Behavior (Detailed)**
- Flags control Kafka connection, topic, and data generation.
- `start_producer(input_type)`:
  - Generates Example/Instance/ExampleBatch protos and writes to Kafka with length-prefixed encoding.
  - Uses hard-coded SASL credentials and sleeps 10s before production.
- `test_kafka_dataset(input_type, output_type)`:
  - Starts producer thread.
  - Creates `KafkaDataset` with given variant/output types.
  - Applies `add_label` with config string (click head optional).
  - Batches, parses into features, splits label vector into task labels.
  - Iterates for `num_batch` and prints results.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/tests`.
- Rust public API surface: Kafka dataset ingestion tests.
- Data model mapping: Kafka stream → dataset parser.
- Feature gating: Kafka support.
- Integration points: data pipeline.

**Implementation Steps (Detailed)**
1. Provide integration tests only in environments with Kafka.
2. Mock Kafka for unit tests to avoid hard-coded credentials.

**Tests (Detailed)**
- Python tests: this file (integration).
- Rust tests: TODO (manual)
- Cross-language parity test: TODO (manual)

**Gaps / Notes**
- TODO (manual)

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 125
- Purpose/role: Tests split/merge flow on instance dataset using lagrangex headers.
- Key symbols/classes/functions: `MultiFlowTest.test_data_flow`.
- External dependencies: `tensorflow`, `PBDataset`, `parse_instances`, `Instance` proto.
- Side effects: writes/reads `data.pb` under `TEST_TMPDIR`.

**Required Behavior (Detailed)**
- `setUpClass`:
  - Generates `NUM_INSTANCE` Instance protos with random fids, line_id fields.
  - Writes lagrangex header and length-prefixed data to `data.pb`.
- `mk_kgx_header(dataflow)`:
  - Computes Java hash code for `dataflow`, writes 4-byte header.
- `test_data_flow`:
  - Reads dataset with `lagrangex_header=True`.
  - Splits into flows by device_types and merges back.
  - Parses instances and batches; expects 8 batches of size 512.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/tests`.
- Rust public API surface: split_flow / merge_flow dataset operations.
- Data model mapping: lagrangex header parsing.
- Feature gating: none.
- Integration points: data pipeline.

**Implementation Steps (Detailed)**
1. Add lagrangex header parsing and flow split/merge in Rust datasets.
2. Add test for split/merge on synthetic instance data.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: TODO (manual)
- Cross-language parity test: TODO (manual)

**Gaps / Notes**
- TODO (manual)

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 253
- Purpose/role: Tests negative sampling generation for Instance/Example datasets.
- Key symbols/classes/functions: `NegativeGenTest.test_dataset_target`.
- External dependencies: `tensorflow`, `PBDataset`, `negative_gen`, `parse_instances/parse_examples`.
- Side effects: writes a temporary `{variant_type}.pb` file.

**Required Behavior (Detailed)**
- `setUpClass`:
  - Generates sample data with random FIDs and labels; writes length-prefixed protos.
  - Tracks per-channel pos/neg counts and per-gid counts.
- `test_dataset_target`:
  - Reads PBDataset and applies `negative_gen` with configured params:
    - `neg_num`, `start_num`, `max_item_num`, `cache_only_pos`, `per_channel`, `throw_origin`, `throw_origin_neg`.
  - Parses dataset and counts pos/neg labels; verifies counts and expected ranges.
  - Ensures total count equals pos+neg.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/tests`.
- Rust public API surface: negative sampling dataset transform.
- Data model mapping: `negative_gen` functionality in Rust.
- Feature gating: none.
- Integration points: dataset pipeline.

**Implementation Steps (Detailed)**
1. Implement negative sampling logic in Rust datasets.
2. Add tests for per-channel and non-channel sampling boundaries.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: TODO (manual)
- Cross-language parity test: TODO (manual)

**Gaps / Notes**
- TODO (manual)

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 1833
- Purpose/role: Validates sparse feature sharding logic and fused layout parsing across ExampleBatch/Example/Instance.
- Key symbols/classes/functions: `DataOpsV2Test`, `DataOpsV2TestFitPreV2`, `DataOpsV2Testv4`, `DataOpsV2TestFitPre`.
- External dependencies: `tensorflow`, `parse_instances/parse_examples/parse_example_batch`, `sharding_sparse_fids`, proto `FeatureConfigs`.
- Side effects: reads training_instance fixtures; prints debug output.

**Required Behavior (Detailed)**
- Implements reference sharding calculations for multiple versions (v2/v3/v4).
- Validates that `sharding_sparse_fids` outputs (`fid_map`, offsets, row splits) match manually computed results.
- Tests for:
  - ExampleBatch sharding with shared features.
  - Example sharding with generated v2 features.
  - Instance sharding with v1+v2 features.
  - Pre-v2 compatibility path (`DataOpsV2TestFitPre`).
- Uses `ParserCtx.enable_fused_layout` toggle to compare base vs fused outputs.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/tests`.
- Rust public API surface: sparse sharding utilities and fused layout parser.
- Data model mapping: feature configs → shard maps and offsets.
- Feature gating: none.
- Integration points: parsing pipeline for distributed embedding.

**Implementation Steps (Detailed)**
1. Implement sharding_sparse_fids equivalent in Rust.
2. Port the reference sharding calculations to Rust tests.
3. Compare fused vs non-fused parsing outputs.

**Tests (Detailed)**
- Python tests: this file (extensive).
- Rust tests: TODO (manual)
- Cross-language parity test: TODO (manual)

**Gaps / Notes**
- TODO (manual)

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 782
- Purpose/role: Parsing utilities that turn Monolith Instance/Example/ExampleBatch variant tensors into feature dicts, plus sharding sparse fid helpers and parser context management.
- Key symbols/classes/functions: `ParserCtx`, `ShardingSparseFidsOpParams`, `ProtoType`, `parse_instances`, `parse_examples`, `parse_example_batch`, `parse_example_batch_list`, `sharding_sparse_fids`, `sharding_sparse_fids_with_context`, `_add_dense_features`, `_add_extra_features`, `_assemble`.
- External dependencies: TensorFlow, `LineId` proto, `FeatureConfigs` proto, `LabelConf` proto, `FeatureList`, `gen_monolith_ops` custom kernels, `logging_ops`, `native_task_context`, `FLAGS.dataset_use_dataservice`.
- Side effects: populates TF collections via `add_to_collections`; writes to global parser context; logs timing metrics (`logging_ops.emit_timer`); registers required feature names via `add_feature` when example-batch parsing.

**Required Behavior (Detailed)**
- `ParserCtx` (context manager):
  - Global `_default_parser_ctx` is used if none exists; `get_default_parser_ctx()` creates `ParserCtx(False)` once.
  - `enable_resource_constrained_roughsort` (class-level flag) injects `item_id` into `extra_features` when parsing instances.
  - `enable_fused_layout` toggles v2 parsing ops and sharded sparse fid handling.
  - `parser_type` is set to `'instance'`, `'example'`, or `'examplebatch'` by parse functions.
  - `sharding_sparse_fids_op_params` holds op configuration (see below) and drives `sharding_sparse_fids_with_context` behavior.
  - `set/get` store arbitrary per-parse context values (e.g., `batch_size`).
  - `sharding_sparse_fids_features_insert_to_features` injects nested dict values into `features` with `__sharding_sparse_fids__` prefix; supports two-level dicts only.
  - `sharding_sparse_fids_features_parse_from_features` reverses the prefixing and removes those keys from `features`.
- `ShardingSparseFidsOpParams` dataclass:
  - Fields: `num_ps`, `use_native_multi_hash_table`, `unique` (callable), `transfer_float16`, `sub_table_name_to_config`, `feature_configs`, `enable_gpu_emb`, `use_gpu`.
- `ProtoType.get_tf_type(proto_type)`:
  - Maps proto field types to tf dtypes: INT → `tf.int64`, FLOAT → `tf.float32`, STRING → `tf.string`.
  - Raises `Exception('proto_type {} is not support'.format(proto_type))` for unknown types.
- `_add_dense_features(names, shapes, types, dense_features, dense_feature_shapes, dense_feature_types)`:
  - Requires `dense_features` and `dense_feature_shapes` non-null, same length, shapes > 0.
  - Defaults `dense_feature_types` to `[tf.float32] * len(dense_features)` if None; otherwise lengths must match.
  - Appends to `names`, `shapes`, `types`.
- `_add_extra_features(names, shapes, types, extra_features, extra_feature_shapes)`:
  - Requires `extra_features` and shapes non-null, same length, shapes > 0.
  - Resolves dtype from `LineId` descriptor per field; raises `Exception(f"{name} is not in line id, pls check!")` if missing.
- `_assemble(sparse_features, names, shapes, types, out_list, batch_size)`:
  - For sparse features: takes `split = out_list[i]` (reshaped to `(batch_size+1,)` if batch_size provided) and `value = out_list[i + len(names)]`; returns `tf.RaggedTensor.from_row_splits`.
  - For dense features: uses `out_list[i]` directly.
  - Returns dict of feature name → tensor/ragged tensor.
- `parse_instances(tensor, fidv1_features, fidv2_features, dense_features, dense_feature_shapes, dense_feature_types, extra_features, extra_feature_shapes)`:
  - If `ParserCtx.enable_resource_constrained_roughsort` is True, ensures `item_id` is in `extra_features` with shape 1.
  - Validates dense feature inputs and defaults types to `tf.float32`.
  - Sets parser context type `'instance'` and writes multiple lists to TF collections + context (fidv1/fidv2/dense/extra, shapes/types).
  - Non-fused layout:
    - For `fidv1_features`: adds feature names from slots via `get_feature_name_and_slot`; if all entries are strings, resolves slots via `FeatureList.parse()` and raises `RuntimeError("fidv1_features error")` on failure.
    - Adds `fidv2_features` names; sets shapes to `-1` and types to `tf.int64`.
    - Asserts no duplicate names.
    - Calls `parse_instance_ops.parse_instances(...)` and `_assemble` with sparse features.
  - Fused layout:
    - If no names, injects `__FAKE_FEATURE__` with shape 1/float32.
    - Calls `parse_instances_v2` and `_assemble` (no sparse features list).
    - If `sharding_sparse_fids_op_params` present and (`use_gpu` or `FLAGS.dataset_use_dataservice`), calls `sharding_sparse_fids_with_context(instances, features, ctx)`.
    - Else stores `instances` under `__sharding_sparse_fids__sparse_features` key.
    - Removes `__FAKE_FEATURE__` before returning.
- `parse_examples(...)` and `parse_example_batch(...)`:
  - Same dense/extra validation pattern as `parse_instances`.
  - Sets parser context type `'example'` or `'examplebatch'` and stores config in TF collections.
  - If `is_example_batch()` is True, registers required features via `add_feature`: sparse features, dense features (adds `__LABEL__` for label), and `__LINE_ID__` for extra features.
  - Non-fused: names from sparse features, shapes `-1`, types `tf.int64`, then calls `parse_examples`/`parse_example_batch` and `_assemble` (batch_size from context for example_batch).
  - Fused: same `__FAKE_FEATURE__` fallback, uses `parse_examples_v2`/`parse_example_batch_v2`, then `sharding_sparse_fids_with_context` or stores under `__sharding_sparse_fids__sparse_features`.
- `sharding_sparse_fids(tensor, ps_num, feature_cfgs, unique, input_type, parallel_flag, fid_list_ret_list, version)`:
  - Normalizes `input_type` (`example_batch` → `examplebatch`).
  - Builds sorted `table_name_list` from `feature_cfgs.feature_configs[*].table`; `ps_num=1` if 0; `table_count = len(table_name_list) * ps_num`.
  - Uses `logging_ops.tensors_timestamp` around op call and emits timer `sharding_sparse_fids` with tag `model_name` from `native_task_context`.
  - Calls versioned custom op (`sharding_sparse_fids_v5/v4/v3/v2` or legacy) returning fid lists, row splits, offsets, and sizes.
  - Asserts list lengths for versions 5/4; returns either raw lists (if `fid_list_ret_list` or `version==4`) or dicts keyed by `table:ps_index` with row splits and row_split_size.
- `sharding_sparse_fids_with_context(sparse_features, features, parser_ctx)`:
  - Calls `sharding_sparse_fids` with params from `parser_ctx.sharding_sparse_fids_op_params`.
  - If `enable_gpu_emb`: inserts `shards_value`, `shards_row_lengths`, `shards_table_row_lengths`, offsets, `batch_size`, `fid_list_emb_row_lenth` into `features` using prefixed keys.
  - Else inserts `shards`, offsets, `batch_size`, size stats; if `use_native_multi_hash_table`, also inserts `shards_row_split` and `shards_row_split_size`.
- `parse_example_batch_list(tensor_list, label_config, positive_label, negative_label, names, shapes, dtypes, extra_features)`:
  - Optionally parses `label_config` (semicolon-separated tasks, each `pos_actions:neg_actions`) into `LabelConf`, and adds `label` feature with shape `len(tasks)`.
  - Marks `shapes[i] == -1` as sparse, appends `tf.int64` to `dtypes` for sparse values (to match op output list shape).
  - Calls `parse_example_batch_list` op with serialized label conf, then `_assemble`.
- Error semantics:
  - Extensive `assert` checks for list lengths/shape values, duplicates, and supported types; specific exceptions for invalid LineId fields and fidv1_features name mapping.
- Metrics/logging:
  - `sharding_sparse_fids` emits a timer metric named `sharding_sparse_fids` with model_name tag.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/src` (parsing), `monolith-rs/crates/monolith-proto` (LineId/FeatureConfigs), optional TF backend for custom ops.
- Rust public API surface: `parsers` module with `parse_instances`, `parse_examples`, `parse_example_batch`, and sharding helpers; `ParserCtx` analog for context state.
- Data model mapping: TF Variant/RaggedTensor → Rust datasets/feature maps; need ragged representation and feature registry.
- Feature gating: fused layout parsing, sharding_sparse_fids, and GPU embedding paths behind feature flags.
- Integration points: datasets (`datasets.py`), feature registry (`feature_list.py`), training pipelines expecting collections metadata.

**Implementation Steps (Detailed)**
1. Implement a Rust `ParserCtx` with context manager semantics (scoped override) and a global default.
2. Port `_add_dense_features`, `_add_extra_features`, and `_assemble` with equivalent validation and ragged construction.
3. Implement `parse_instances`/`parse_examples`/`parse_example_batch` in Rust, honoring `enable_fused_layout` and `enable_resource_constrained_roughsort` behavior.
4. Provide `FeatureList` lookups for fidv1 slot-name mapping and raise equivalent errors on failure.
5. Persist metadata to a Rust collection registry mirroring `add_to_collections` semantics.
6. Implement sharding_sparse_fids and sharding_sparse_fids_with_context around native kernels or TF runtime bindings; preserve timing metric emission.
7. Implement parse_example_batch_list with label_config parsing and label feature insertion.
8. Add tests for parsing shape/type inference, ragged assembly, and sharding outputs using small fixture tensors.

**Tests (Detailed)**
- Python tests: `data_ops_test.py`, `parse_sparse_feature_test.py`, `feature_utils_test.py`, `tf_example_to_example_test.py` (parsing paths).
- Rust tests: parser unit tests for each parse_* function; sharding_sparse_fids smoke tests (if backend available).
- Cross-language parity test: parse the same fixture files and compare feature dict keys, shapes, and ragged values.

**Gaps / Notes**
- Fused layout paths depend on custom ops (`parse_*_v2` and `sharding_sparse_fids_*`); must be backed by TF runtime or re-implemented.
- `parse_example_batch_list` mutates dtypes length to match op outputs; replicate this behavior exactly.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 183
- Purpose/role: End-to-end test that converts TF Example records to Monolith Example variants via `tf_example_to_example`, then parses with `parse_examples` and asserts fid/dense defaults.
- Key symbols/classes/functions: `serialize_example`, `get_fid_v2`, `calc_hash_value`, `TFExampleToExampleTest.test_tf_example_to_example`.
- External dependencies: TensorFlow TFRecord, numpy RNG, `tf_example_to_example`, `parse_examples`.
- Side effects: writes `/tmp/test.tfrecord` with 10k TF Examples; uses TF1 session.

**Required Behavior (Detailed)**
- Helper functions:
  - `_bytes_feature`, `_float_feature`, `_int64_feature` wrap values into `tf.train.Feature` list types.
  - `serialize_example(feature0, feature1, feature2, feature3, feature4)` builds a `tf.train.Example` with:
    - `feature0`: int64 (bool values allowed)
    - `feature1`: int64
    - `feature2`: bytes
    - `feature3`: float
    - `feature4`: float
  - `get_fid_v2(slot, signature)` uses `fid_v2_mask=(1<<48)-1` and returns `(slot<<48) | (signature & mask)`.
  - `calc_hash_value(val)` returns `int(log2(abs(val)+1))`.
- `test_tf_example_to_example`:
  - Disables TF2 behavior (`tf.compat.v1.disable_v2_behavior()`), uses TF1 session graph.
  - Generates 10k samples:
    - `feature0`: random bools
    - `feature1`: random ints in [0,4]
    - `feature2`: bytes from `strings[feature1]`
    - `feature3`: random normal float
    - `feature4`: random normal float
  - Writes TFRecord file `/tmp/test.tfrecord` with serialized Examples.
  - Dataset pipeline:
    - `TFRecordDataset` → `map(tf_example_to_example)` with:
      - `sparse_features={'feature0':1,'feature1':2,'feature4':3}` (fid_v2 slots)
      - `dense_features=['feature2']`
      - `label='feature3'`
      - `instance_weight=None`
    - Batch size 2 → `map(parse_examples)` with:
      - `sparse_features=['not_existed1','feature0','feature1','feature4']`
      - `dense_features=['label','feature2','feature3','not_existed2','instance_weight']`
      - `dense_feature_types=[float32,string,float32,float32,float32]`
      - `dense_feature_shapes=[1,1,1,1,1]`
  - In session loop (5k batches):
    - `not_existed1` ragged has zero values.
    - `feature0/feature1` fids equal `get_fid_v2(slot, original int/bool)` per batch.
    - `feature4` fid uses slot 3 and `calc_hash_value` of float value (log2(abs(val)+1)).
    - `label` equals original `feature3` (float) per batch.
    - `feature3` dense output is `[0,0]` (missing in conversion), `not_existed2` is `[0,0]`.
    - `instance_weight` defaults to `[1.0,1.0]`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/tests` (TFExample conversion) + `monolith-proto` for Example parsing.
- Rust public API surface: test helper for TF Example serialization; conversion op `tf_example_to_example` or equivalent.
- Data model mapping: TF Example bytes → Monolith Example variant → parsed feature dict.
- Feature gating: TFRecord read/write and TFExample conversion backend.
- Integration points: `feature_utils.tf_example_to_example` and `parsers.parse_examples` parity.

**Implementation Steps (Detailed)**
1. Implement TF Example serialization helper in Rust tests (or load TFRecord fixtures generated in Python).
2. Provide `tf_example_to_example` conversion in Rust with the same slot/fid-v2 behavior and hashing for float feature4.
3. Ensure missing sparse features emit empty ragged values; missing dense features emit zeros; `instance_weight` defaults to 1.0.
4. Add a Rust test that mirrors batch size 2 with deterministic input, validating fid values and dense defaults.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: `tf_example_to_example.rs` (new) that asserts identical fid/dense outputs.
- Cross-language parity test: generate a fixed TFRecord in Python and run Rust conversion+parse on it; compare outputs.

**Gaps / Notes**
- Uses `/tmp/test.tfrecord`; Rust tests should use tempdir paths.
- The hash for float sparse feature4 is `int(log2(abs(val)+1))`; must match exactly.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 166
- Purpose/role: TF DatasetSource wrapper around custom `instance_dataset` op for reading serialized Instance records from PB files or stdin, with optional sharding/interleave utilities.
- Key symbols/classes/functions: `_PBInstanceDataset`, `PBInstanceDatasetV2`, `create_instance_dataset`, alias `PBInstanceDataset`.
- External dependencies: TensorFlow Dataset internals, `gen_monolith_ops.instance_dataset`, `distributed_dataset.create_dynamic_sharding_dataset`, `ckpt_hooks.disable_iterator_save_restore`, TF matching_files.
- Side effects: disables iterator save/restore when reading from stdin; logs initialization; uses TF fatal logging on missing file.

**Required Behavior (Detailed)**
- `_PBInstanceDataset(file_name, use_snappy, has_sort_id, kafka_dump, kafka_dump_prefix)`:
  - Calls custom op `instance_dataset` with tensors for file name, snappy, and header flags.
  - `element_spec` is scalar string `TensorSpec([], tf.string)`.
- `PBInstanceDatasetV2`:
  - If `file_name` is empty string, treats input as stdin and calls `ckpt_hooks.disable_iterator_save_restore()`.
  - Creates `_PBInstanceDataset` internally and forwards variant tensor into `DatasetV2`.
  - `_clone` merges kwargs with stored defaults.
  - `_inputs()` returns `[]`.
- `create_instance_dataset(...)`:
  - `files_list=None` defaults to `['']` (stdin).
  - If a single file and no glob expansion/sharding/dynamic sharding, returns `PBInstanceDatasetV2` directly; validates existence when file is non-empty and logs fatal on missing file.
  - `enable_dynamic_sharding=True`:
    - Converts to dataset via `distributed_dataset.create_dynamic_sharding_dataset` and `flat_map` with `PBInstanceDatasetV2`.
  - `enable_sharding=True`:
    - Requires a single file pattern; uses `MatchingFilesDataset`, shards by `shard_num/shard_index`, logs shard info; forces `use_snappy=True`.
  - Else:
    - Uses `MatchingFilesDataset` if `expand_glob_path=True`, otherwise `Dataset.from_tensor_slices`.
  - Final dataset uses `interleave` with `cycle_length`, `block_length`, `num_parallel_calls`, `deterministic=False`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/src` for dataset creation and file reader.
- Rust public API surface: `pb_instance_dataset` (or similar) and `create_instance_dataset` with matching options.
- Data model mapping: custom op variant tensor → Rust stream of serialized Instance bytes.
- Feature gating: stdin mode, dynamic sharding, and TF MatchingFiles behavior.
- Integration points: datasets (`datasets.py`), training pipelines that expect PBInstanceDataset semantics.

**Implementation Steps (Detailed)**
1. Implement a Rust dataset source that wraps Instance file reading with flags for sort_id/kafka headers and snappy.
2. Mirror stdin special case and disable iterator save/restore in Rust equivalents.
3. Implement glob expansion, sharding, and dynamic sharding (or document unsupported) with identical defaults.
4. Preserve interleave behavior and `deterministic=False` semantics.

**Tests (Detailed)**
- Python tests: `instance_dataset_op_test_stdin.py`, other dataset tests using PBInstanceDataset.
- Rust tests: dataset source tests for stdin vs file, sharding path, missing file handling.
- Cross-language parity test: read a fixture PB file in Python and Rust and compare record sequence.

**Gaps / Notes**
- Uses TF Dataset internals; Rust must define a similar streaming abstraction.
- Missing file uses `logging.fatal` in TF; decide equivalent behavior in Rust (panic or error).

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 58
- Purpose/role: Smoke test for `PBInstanceDataset` reading from stdin (empty file_name), batching, and parsing with `parse_instances`.
- Key symbols/classes/functions: `PBInstanceDataset`, `parse_instances`, `testInstanceDataset`.
- External dependencies: TensorFlow v1 session, `instance_dataset_ops`, `parse_instance_ops`.
- Side effects: expects stdin data stream; logs warnings; runs one batch read.

**Required Behavior (Detailed)**
- Defines feature lists:
  - `FIDV1_FEATURES = [1..9]`
  - `FIDV2_FEATURES = ['fc_360d_ml_convert_cid', 'fc_360d_ml_convert_advertiser_id']`
  - `FLOAT_FEATURES = ['fc_muse_finish_rough_10168_uid_d128']` with dim `[128]`
  - `INT64_FEATURES = ['fc_dense_external_action']` with dim `[1]`
- `parse(serialized)` calls `parse_instances(serialized, fidv1, fidv2, float_feats, float_dims, int64_feats, int64_dims)`.
- `testInstanceDataset()`:
  - Creates `PBInstanceDataset(file_name='', has_sort_id=True, kafka_dump_prefix=True)` (stdin path).
  - `batch(32)` and `map(parse)`.
  - Builds one-shot iterator, fetches one batch, logs `elements['sample_rate']`.
- Script mode: disables eager and runs `testInstanceDataset()`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/tests`.
- Rust public API surface: stdin dataset reader + parse_instances in Rust.
- Data model mapping: stream of framed Instance records from stdin.
- Feature gating: stdin support in dataset source; parse_instances.
- Integration points: dataset source in `instance_dataset_op.py` parity.

**Implementation Steps (Detailed)**
1. Add a Rust test that simulates stdin input (e.g., pipe fixture data into the dataset reader).
2. Ensure parsing handles fidv1/fidv2 and dense features as in Python.
3. Validate batch size 32 returns expected keys including `sample_rate`.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: add stdin dataset smoke test with a small fixture.
- Cross-language parity test: compare parsed batch fields from the same stdin fixture.

**Gaps / Notes**
- This test assumes stdin provides valid Instance records; Rust tests should supply a controlled fixture stream.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 283
- Purpose/role: Tests negative sample generation dataset (`negative_gen` and `InstanceNegativeGenDataset`) over Instance PB data, including per-channel ring buffer behavior.
- Key symbols/classes/functions: `InsNegativeDatasetTest`, `parse1`, `testNegativeGen`, `testRingBufferCache`, `testIgnoreReaNegInstance`, `testUseNegInstance`.
- External dependencies: TensorFlow, `PBDataset`, `InstanceNegativeGenDataset`, `PbType`, custom parse ops `parse_variant_instances`.
- Side effects: reads fixture `monolith/native_training/data/training_instance/instance.pb`.

**Required Behavior (Detailed)**
- Constants:
  - `FILE_NAME` fixture file for Instance PB.
  - `CHANNEL_SLOT=357`, `GROUP_SLOTS=[200..242]`, `LABEL_FIELD='actions'`, `LABEL_INDEX=0`.
  - Negative labels `NEGATIVE_LABEL=-2`, `NEGATIVE_LABEL2=-1`.
  - `GID='gid'` used as misc int64 feature.
- `parse1(pb_variant)`:
  - Uses a fixed `FIDV1_FEATURES` list and `parse_variant_instances` with `misc_int64_features=[GID]`.
- `testNegativeGen`:
  - Builds `PBDataset` (Instance → Instance) with headers; applies `dataset.negative_gen` with:
    - `neg_num=7`, `channel_slot`, `group_slots`, `per_channel_sample=True`, `start_num=0`, `max_group_num_per_channel=10000`, `label_field='actions'`, `label_index=0`, `negative_label=-2`, `use_neg_ins=True`.
  - Batches 8 and parses.
  - Asserts in first batch that `channel_res[0][0] == channel_res[0][i]` for i in 1..7 (negatives share channel), and label at index 1 equals `NEGATIVE_LABEL`.
- `testRingBufferCache`:
  - Same negative_gen config except `max_group_num_per_channel=2`.
  - Collects ~1024 samples; groups by channel and verifies ring buffer behavior:
    - For channels with >2 samples, checks that group fids from later samples are not present in the first sample when gids differ.
  - Logs `valid_count` of checked non-overlapping group features.
- `testIgnoreReaNegInstance`:
  - First applies `dataset.negative_gen(..., negative_label=-2, use_neg_ins=True)`.
  - Then wraps with `InstanceNegativeGenDataset(..., negative_label=-1, use_neg_ins=False)`.
  - Asserts label at index 1 equals `NEGATIVE_LABEL2` (real negatives ignored).
- `testUseNegInstance`:
  - Same as previous but `use_neg_ins=True` in wrapper.
  - Asserts labels: index1/index2 are `NEGATIVE_LABEL2`, index3/index4 are `NEGATIVE_LABEL` (mix of generated vs real negatives).

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/tests` with negative_gen dataset implementation in `monolith-data`.
- Rust public API surface: `negative_gen` dataset operator and `InstanceNegativeGenDataset` wrapper.
- Data model mapping: Instance variant streams with channel/group fid slots and label field `actions`.
- Feature gating: requires negative generation ops + item pool or group cache implementation.
- Integration points: dataset pipeline in `datasets.py` and negative-gen custom ops in Rust/TF backend.

**Implementation Steps (Detailed)**
1. Implement `negative_gen` dataset operator and wrapper in Rust with identical parameters.
2. Ensure per-channel sampling and ring buffer cache behavior match Python semantics.
3. Expose `use_neg_ins` toggle to include/exclude existing negatives.
4. Add Rust tests that load the same fixture PB file and verify label/channel/group constraints.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: new `instance_negative_gen_dataset_op_test.rs` mirroring each test case.
- Cross-language parity test: compare label distributions and channel/group assignments on identical fixtures.

**Gaps / Notes**
- Depends on `instance.pb` fixture and custom ops; ensure Rust has compatible dataset and parse op support.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 245
- Purpose/role: Instance parsing helpers that wrap custom ops to extract fid and dense features into Ragged/Dense tensors, including LineId fields and repeated fields handling.
- Key symbols/classes/functions: `_parse_instance_impl`, `parse_instances2`, `parse_instances`, `monolith_raw_parse_instance`.
- External dependencies: TensorFlow ragged internals (`RowPartition`), `gen_monolith_ops` custom kernels, `get_slot_feature_name`, parser_utils hooks (imported but not used here).
- Side effects: Builds RaggedTensor with precomputed row ids/nrows; uses default misc feature lists.

**Required Behavior (Detailed)**
- `_parse_instance_impl(serialized, fidv1_features, fidv2_features, float_features, float_feature_dims, int64_features, int64_feature_dims, string_features, string_feature_dims, misc_float_features, misc_float_dims, misc_int64_features, misc_int64_dims, misc_string_features, misc_string_dims, cc_op)`:
  - Normalizes all list args to empty lists if None.
  - Calls `cc_op` (custom op) with counts `N/M/O/P/Q/R/S` and all feature lists/dims.
  - Builds `ragged_keys` from `fidv1_features` (via `get_slot_feature_name`) plus `fidv2_features`.
  - For each ragged split/value pair, constructs `RowPartition` with precomputed `value_rowids` and `nrows`, then `tf.RaggedTensor(values, row_partition, internal=True)`.
  - Returns dict mapping ragged + float + int64 + string + misc_* features to their tensors in order.
- `parse_instances2(...)`:
  - Thin wrapper that calls `_parse_instance_impl` with `parse_instance_ops.monolith_parse_instances`.
- `parse_instances(...)`:
  - Adds defaults: `misc_float_features=['sample_rate']`, `misc_int64_features=['req_time','uid']`, `misc_repeated_float_features=['label']`.
  - Normalizes list args to empty lists and sets default dims (1) for misc features.
  - Calls `parse_instances2` with concatenated misc+repeated feature lists/dims.
  - Reshapes non-repeated misc float/int64 features to 1-D (`tf.reshape(features[key], [-1])`).
  - Returns feature dict.
- `monolith_raw_parse_instance`:
  - Exposes `parse_instance_ops.MonolithRawParseInstance` for testing only.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/src` for instance parsing; ragged support in `monolith-tensor`.
- Rust public API surface: `parse_instances`/`parse_instances2` equivalents returning `HashMap<String, Tensor>` with ragged types.
- Data model mapping: custom op outputs (splits/values) → Rust ragged tensors with cached row ids/nrows.
- Feature gating: depends on custom parsing kernels or TF runtime.
- Integration points: dataset parsing pipelines, tests in `parse_instance_ops_test.py` and other training_instance tests.

**Implementation Steps (Detailed)**
1. Implement Rust wrappers for `monolith_parse_instances` (or bind to TF op) that return splits/values arrays.
2. Build ragged tensors with cached row metadata to match TF `RowPartition` behavior.
3. Match default misc feature lists and reshape semantics in `parse_instances`.
4. Preserve feature key ordering in output map to match downstream expectations.

**Tests (Detailed)**
- Python tests: `parse_instance_ops_test.py`, `instance_dataset_op_test_stdin.py`, `instance_negative_gen_dataset_op_test.py`.
- Rust tests: parser unit tests for ragged vs dense outputs; ensure misc defaults applied.
- Cross-language parity test: parse a fixture instance and compare fid/dense outputs.

**Gaps / Notes**
- Uses TF internal ragged APIs; Rust must provide equivalent row-partition caching to avoid perf regressions.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 185
- Purpose/role: Validates instance parsing ops and ragged encoding helpers, including missing field defaults and raw parse concat behavior.
- Key symbols/classes/functions: `RaggedEncodingHelperTest`, `ParseInstancesTest`, `RawParseInstanceTest`, helper `generate_instance`, `make_fid_v1`, `make_fid_v2`.
- External dependencies: TensorFlow, `proto_parser_pb2.Instance`, `parse_instance_ops`, `parser_utils.RaggedEncodingHelper`.
- Side effects: none.

**Required Behavior (Detailed)**
- `generate_instance()` builds an Instance with:
  - fidv1 list `[make_fid_v1(i,i) for i in range(10)]`.
  - fidv2 feature `name='fidv2'` with `make_fid_v2(100,i)`.
  - float feature `ue` length 16 (i * 1e-5), int64 feature `int64_feature=100`, string feature `string_feature='test_string'`.
  - label `[1.1,2.2,3.3]`, line_id fields: uid=110, sample_rate=0.5, req_time=64, actions=[0,100], user_id='123'.
- `RaggedEncodingHelperTest.testExpandContract`:
  - Builds a ragged tensor, expands with `RaggedEncodingHelper.expand(..., with_precomputed_value_rowids=True)` and verifies `value_rowids` equals TF-computed.
  - `contract` returns original ragged values and preserves cached value_rowids.
- `ParseInstancesTest.testParseInstance`:
  - Calls `parse_instances2` with explicit fidv1/fidv2/float/int64/string/misc fields and dims.
  - Asserts:
    - 10 fidv1 slots returned (`slot_*`).
    - `slot_1` uses fid_v2 encoding for v1 slot values.
    - `fidv2` ragged equals `get_test_fidv2()`.
    - dense features: `int64_feature=[[100]]`, `string_feature=[[b'test_string']]`, `ue` length 16, `sample_rate=[[0.5]]`, `label=[[1.1,2.2,3.3]]`, `uid=[[110]]`, `actions=[[0,100]]`, `user_id=[['123']]`.
- `ParseInstancesTest.testParseInstanceV1Only`:
  - `parse_instances2` with `fidv1_features=[1]` yields `slot_1` with fid_v1 encoding.
- `ParseInstancesTest.testParseInstanceWithMissingFields`:
  - Requests extra missing fields; expects:
    - Missing ragged fid slot → empty ragged (`[[]]`).
    - Missing fidv2 → empty ragged.
    - Missing float → zeros of specified dim.
    - Missing int64/string → zeros/empty strings of specified dim.
- `RawParseInstanceTest.test_concat`:
  - Calls `monolith_raw_parse_instance` with `fid_output_type='CONCAT'`.
  - Expects first tensor offsets `[0,1,2,len(fidv2)+2]` and second tensor concatenated fids `[fidv1 slot0, fidv1 slot1] + fidv2 list`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/tests` and ragged utils in `monolith-tensor`.
- Rust public API surface: ragged encoding helper, `parse_instances2`, raw parse op if exposed.
- Data model mapping: same fid encoding rules (v1/v2), missing-field defaults.
- Feature gating: raw parse op requires custom kernel support.
- Integration points: parse_instance_ops implementation and parser_utils utilities.

**Implementation Steps (Detailed)**
1. Implement ragged encoding helper in Rust and verify cached rowids/nrows behavior.
2. Port `parse_instances2` tests with the same synthetic Instance fixture.
3. Ensure missing fields return empty ragged or zero-filled dense tensors as specified.
4. If raw parse op is supported, add concat mode test for offsets + fid list order.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: new `parse_instance_ops_test.rs` matching each test case.
- Cross-language parity test: compare parsed feature dicts for identical serialized Instance.

**Gaps / Notes**
- `slot_*` fidv1 encoding differs between v1-only path and v2 conversion; Rust must replicate both behaviors.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 85
- Purpose/role: Utilities for parser pipelines, including queued extra-parse steps and ragged encoding expansion/contract helpers.
- Key symbols/classes/functions: `_extra_parse_steps`, `add_extra_parse_step`, `RaggedEncodingHelper.expand`, `RaggedEncodingHelper.contract`, `advanced_parse`.
- External dependencies: TensorFlow, `ragged_utils.fused_value_rowids`.
- Side effects: mutates global deque of extra parse steps; mutates RaggedTensor internal row partition caches during `contract`.

**Required Behavior (Detailed)**
- `_extra_parse_steps`:
  - Global `deque` used to store parse step callables.
- `add_extra_parse_step(parse_fn)`:
  - Appends parse_fn to `_extra_parse_steps`.
- `RaggedEncodingHelper.expand(name_to_ragged_ids, with_precomputed_nrows=True, with_precomputed_value_rowids=False)`:
  - For each RaggedTensor value, returns a dict with:
    - `values`, `row_splits`, optional `nrows` (if flag), optional `value_rowids` computed via `ragged_utils.fused_value_rowids` (if flag).
  - Non-ragged entries pass through unchanged.
- `RaggedEncodingHelper.contract(name_to_ragged_ids)`:
  - For dict entries with `values` and `row_splits`, rebuilds `tf.RaggedTensor.from_row_splits(..., validate=False)`.
  - If `nrows` present, asserts `_row_partition._nrows` is None before assigning.
  - If `value_rowids` present, asserts `_row_partition._value_rowids` is None before assigning.
  - Non-dict entries pass through unchanged.
- `advanced_parse(features)`:
  - Pops parse steps from `_extra_parse_steps` in FIFO order and applies each to `features`.
  - Returns final features dict.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/src` (parser utilities) + `monolith-tensor` for ragged.
- Rust public API surface: `add_extra_parse_step` and `advanced_parse` equivalents; ragged expand/contract helpers.
- Data model mapping: RaggedTensor internal encodings → Rust ragged structure with cached rowids/nrows.
- Feature gating: none.
- Integration points: `parse_instance_ops_test.py` uses `RaggedEncodingHelper`.

**Implementation Steps (Detailed)**
1. Implement a global queue of parse steps (with proper synchronization if used across threads).
2. Implement ragged expand/contract; ensure cached rowids/nrows are set only once.
3. Mirror `fused_value_rowids` behavior using Rust ragged utilities.

**Tests (Detailed)**
- Python tests: `parse_instance_ops_test.py` (`RaggedEncodingHelperTest`).
- Rust tests: add unit tests that expand, contract, and verify rowids/nrows caching.
- Cross-language parity test: compare ragged values and cached rowids against Python output.

**Gaps / Notes**
- Directly mutates internal ragged partition caches; Rust must provide an equivalent escape hatch.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 48
- Purpose/role: Thin wrappers around `gen_monolith_ops` for filtering and negative sampling on variant tensors in training_instance pipelines.
- Key symbols/classes/functions: `filter_by_fids`, `filter_by_value`, `negative_sample`, `variant_dummy`.
- External dependencies: TensorFlow, `gen_monolith_ops` custom kernels.
- Side effects: none beyond custom op invocation.

**Required Behavior (Detailed)**
- `filter_by_fids(variant, filter_fids, has_fids, select_fids, has_actions)`:
  - Passes list args (defaults to empty) to `pb_datasource_ops.set_filter`.
- `filter_by_value(variant, field_name, op, operand)`:
  - Calls `pb_datasource_ops.value_filter` with given field/op/operand.
- `negative_sample(variant, drop_rate, label_index, threshold)`:
  - Calls `pb_datasource_ops.negative_sample` with drop/threshold params.
- `variant_dummy(variant)`:
  - Calls `pb_datasource_ops.variant_dummy`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/src` (ops wrappers).
- Rust public API surface: minimal wrappers for filtering/negative sampling on variant streams.
- Data model mapping: variant tensor → Rust variant representation.
- Feature gating: custom op availability (TF backend).
- Integration points: training_instance datasets and tests.

**Implementation Steps (Detailed)**
1. Add Rust wrapper functions that call the underlying kernel backend.
2. Ensure default empty list behavior matches Python.
3. Expose in public API for dataset pipelines.

**Tests (Detailed)**
- Python tests: indirectly via `instance_negative_gen_dataset_op_test.py`.
- Rust tests: minimal unit tests for wrappers if backend available.
- Cross-language parity test: verify behavior using fixed fixtures.

**Gaps / Notes**
- This is a thin wrapper; underlying op semantics are defined in C++/TF kernels.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 15
- Purpose/role: Placeholder test utility module; currently only imports TensorFlow.
- Key symbols/classes/functions: none.
- External dependencies: TensorFlow.
- Side effects: none.

**Required Behavior (Detailed)**
- No runtime behavior beyond importing TensorFlow.

**Rust Mapping (Detailed)**
- Target crate/module: none.
- Rust public API surface: none.
- Data model mapping: none.
- Feature gating: none.
- Integration points: none.

**Implementation Steps (Detailed)**
1. No Rust port needed unless file is expanded in Python.

**Tests (Detailed)**
- Python tests: none.
- Rust tests: none.
- Cross-language parity test: none.

**Gaps / Notes**
- File is effectively empty; keep an eye on future changes.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 250
- Purpose/role: Declarative transform objects that serialize to `TransformConfig` proto for dataset transform pipelines (filters, label generation, logical composition).
- Key symbols/classes/functions: `Transform` (ABC), `Compose`, `FilterByFid`, `FilterByAction`, `FilterByLabel`, `FilterByValue`, `AddLabel`, `LogicalOr`.
- External dependencies: `transform_config_pb2`, `LineId` proto descriptor for field validation.
- Side effects: none; purely builds proto configs with validation asserts.

**Required Behavior (Detailed)**
- `Transform` abstract base:
  - `as_proto()` returns `transform_config_pb2.TransformConfig`.
  - `_is_leaf_node()` distinguishes leaf vs composite.
- `Compose(transforms)`:
  - Requires all items are `Transform` instances.
  - `as_proto` merges each transform’s proto into a single `TransformConfig` via `MergeFrom` in order.
  - `_is_leaf_node` returns False.
- `FilterByFid(has_fids, filter_fids, select_fids)`:
  - `as_proto` appends a `basic_config.filter_by_fid` entry with respective lists.
  - `_is_leaf_node` True.
- `FilterByAction(has_actions)`:
  - Adds `basic_config.filter_by_action.has_actions`.
  - `_is_leaf_node` True.
- `FilterByLabel(thresholds)`:
  - Adds `basic_config.filter_by_label.thresholds`.
  - `_is_leaf_node` True.
- `FilterByValue(field_name, op, operand, keep_empty=False)`:
  - Validates `op` in `{gt,ge,eq,lt,le,neq,between,in,not-in,all,any,diff,startswith,endswith}`.
  - Validates `field_name` exists in `LineId` descriptor; `operand` is not None.
  - Infers operand type based on field cpp_type and op:
    - Repeated fields (`field.has_options`): only `all/any/diff` allowed; only integer types; operand int or list of int.
    - Float/double: `between` uses list; otherwise single float.
    - Int types: `in/not-in/between` use list; otherwise single int.
    - String: operand is str or list of str; else `RuntimeError("params error!")`.
  - Stores `float_operand`, `int_operand`, `string_operand`, `keep_empty`.
  - `as_proto` fills `basic_config.filter_by_value` with operands and flags.
  - `_is_leaf_node` True.
- `AddLabel(config, negative_value, new_sample_rate)`:
  - Parses config `pos_actions:neg_actions:sample_rate` separated by `;` (skips empty parts).
  - Adds `basic_config.add_label` with negative value + new sample rate and a `task_label_config` entry per task.
  - `_is_leaf_node` True.
- `LogicalOr(x, y)`:
  - Requires both `x` and `y` are leaf nodes.
  - `as_proto` creates `logical_or_config` and copies `basic_config` from each side.
  - `_is_leaf_node` False.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/src` (transform config builders).
- Rust public API surface: `Transform` trait + concrete structs mirroring Python class names; `as_proto()` to `TransformConfig`.
- Data model mapping: transform structs → `transform_config_pb2::TransformConfig`.
- Feature gating: none; pure config serialization.
- Integration points: `TransformDataset` op uses serialized config (see `datasets.py`).

**Implementation Steps (Detailed)**
1. Create Rust transform trait with `as_proto` and `is_leaf` methods.
2. Implement concrete transforms with identical validation (asserts or Result errors).
3. Preserve `Compose` merge ordering and `LogicalOr` leaf-only requirement.
4. Implement `FilterByValue` operand parsing based on LineId descriptor in Rust.

**Tests (Detailed)**
- Python tests: `transforms_test.py`.
- Rust tests: unit tests for each transform’s proto encoding and validation.
- Cross-language parity test: serialize configs in Python and Rust and compare bytes.

**Gaps / Notes**
- `FilterByValue` uses `LineId` field metadata to infer types; Rust must mirror the same descriptor mapping.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 70
- Purpose/role: Smoke tests that build transform configs and log the resulting protobufs; no assertions beyond successful construction.
- Key symbols/classes/functions: `TransformsTest`, test methods for each transform type.
- External dependencies: `transforms` module, `absl.logging`, `unittest`.
- Side effects: logs serialized proto configs.

**Required Behavior (Detailed)**
- `test_filter_by_fid`: builds `FilterByFid(has_fids=[1], filter_fids=[2,3], select_fids=None)` and logs proto.
- `test_filter_by_action`: builds `FilterByAction(has_actions=[4])` and logs proto.
- `test_filter_by_label`: builds `FilterByLabel(thresholds=[-100, -100])` and logs proto.
- `test_add_label`: builds `AddLabel(config='1,2:3:1.0;4::0.5', negative_value=0.0, new_sample_rate=0.3)` and logs proto.
- `test_logical_or`: builds `LogicalOr(FilterByAction([1,2]), FilterByFid([10000000]))` and logs proto.
- `test_compose`: builds `Compose([...])` with multiple transforms including `LogicalOr`, logs proto.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/tests`.
- Rust public API surface: transform builders and `as_proto` serialization.
- Data model mapping: Rust transforms → TransformConfig protobufs.
- Feature gating: none.
- Integration points: verifies transform config serialization used by `TransformDataset`.

**Implementation Steps (Detailed)**
1. Add Rust tests that construct equivalent transforms and ensure `as_proto` succeeds.
2. Optionally compare serialized proto bytes to Python output for deterministic configs.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: `transforms_test.rs` with equivalent constructions.
- Cross-language parity test: serialize each config in both languages and compare bytes.

**Gaps / Notes**
- Tests are smoke-only; Rust should at least mirror construction and serialization.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 55
- Purpose/role: Simple slot/feature-name helpers for training_instance parsing; global mapping of feature names to slots with TOB env toggle.
- Key symbols/classes/functions: `enable_tob_env`, `get_slot_feature_name`, `get_slot_from_feature_name`, `register_slots`, globals `TOBENV`, `USED_FREATUE_NAMES`, `NAME_TO_SLOT`.
- External dependencies: none.
- Side effects: mutates global dictionaries for name/slot mapping.

**Required Behavior (Detailed)**
- Globals:
  - `TOBENV` default `False` toggles slot name prefix (`slot_` vs `fc_slot_`).
  - `USED_FREATUE_NAMES` maps arbitrary feature names to assigned slot ids (incrementing).
  - `NAME_TO_SLOT` maps feature name → slot id (explicit).
- `enable_tob_env()`:
  - Sets `TOBENV = True` globally.
- `get_slot_feature_name(slot)`:
  - Returns `"fc_slot_{slot}"` if `TOBENV` else `"slot_{slot}"`.
- `get_slot_from_feature_name(feature_name)`:
  - If in `NAME_TO_SLOT`, return mapped slot.
  - Else if name starts with `slot_` or `fc_slot_`, parse suffix int; return int or `None` if non-numeric.
  - Else use `USED_FREATUE_NAMES`: assign a new slot id (`len+1`) if missing and return it.
- `register_slots(sparse_features)`:
  - Accepts list/tuple of ints or dict name→slot.
  - For list: asserts ints and converts to dict via `get_slot_feature_name`.
  - Updates `NAME_TO_SLOT` with provided mapping.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/src` (feature utils).
- Rust public API surface: slot/feature-name mapping utilities with global registry or context-bound mapping.
- Data model mapping: feature names → slots used by parsing/feature extraction.
- Feature gating: TOB env toggle.
- Integration points: `parse_instance_ops` (fidv1 slot naming), feature list parsing.

**Implementation Steps (Detailed)**
1. Implement a global or context-local registry for `NAME_TO_SLOT` and `USED_FEATURE_NAMES` with deterministic assignment.
2. Provide `enable_tob_env` toggle and `get_slot_feature_name` logic.
3. Mirror `get_slot_from_feature_name` fallback behavior for unknown names.
4. Implement `register_slots` with list/dict handling and type checks.

**Tests (Detailed)**
- Python tests: none explicit.
- Rust tests: add unit tests for TOB/non-TOB naming and deterministic slot assignment.
- Cross-language parity test: compare mapping outputs for a fixed sequence of names.

**Gaps / Notes**
- Uses global mutable state; Rust must be careful about concurrency or test isolation.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 98
- Purpose/role: CLI client to query debugging server endpoints for variable values or feature embeddings.
- Key symbols/classes/functions: `main`, CLI flags `type`, `variable_names`, `feature_ids`, `feature_name`, `feature_names`.
- External dependencies: `requests`, `json`, protobuf `text_format`, `embedding_hash_table_pb2.EntryDump`.
- Side effects: HTTP POSTs to local debugging server; logs results; raises exceptions on invalid flag combos.

**Required Behavior (Detailed)**
- Flags:
  - `--type` must be `debugging_variables` or `debugging_features`.
  - `--variable_names` list for variable lookup.
  - `--feature_ids` list for feature lookup.
  - `--feature_name` single name to pair with all ids.
  - `--feature_names` list of names; must be same length as `feature_ids` if provided.
- `debugging_variables` flow:
  - If `variable_names` empty → log and return.
  - POST JSON `{"variable_names": [...]}` to `http://127.0.0.1:<port>/debugging/variables`.
  - Response JSON contains `STATUS`, `SUCCESS/FAIL`, `MSG` keys; on FAIL log reason and return.
  - `MSG` is JSON-encoded dict name→value; logs each variable value or "Not exist".
- `debugging_features` flow:
  - Disallow providing both `feature_name` and `feature_names`.
  - If `feature_ids` empty → log and return.
  - If `feature_name` set, expand to list same length as ids.
  - Validate `len(feature_names) == len(feature_ids)` else raise.
  - POST JSON `{"feature_names": [...], "feature_ids": [...]}` to `/debugging/features`.
  - On FAIL log reason and return.
  - `MSG` is JSON-encoded dict name→id→textproto of `EntryDump`.
  - If present, parse textproto into `EntryDump` and log; else log "Not exist".
- Script mode: sets logging verbosity INFO, disables eager, and runs app.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/debugging` (or CLI crate).
- Rust public API surface: CLI command for debugging server queries.
- Data model mapping: JSON request/response; `EntryDump` textproto parsing.
- Feature gating: requires debugging server running locally.
- Integration points: `debugging_server.py` endpoints.

**Implementation Steps (Detailed)**
1. Implement a Rust CLI that mirrors flags and validation.
2. POST to `/debugging/variables` and `/debugging/features` with identical JSON payloads.
3. Parse response JSON; for features, parse textproto into `EntryDump` (protobuf text format parser).
4. Match logging output patterns and error handling (exceptions for invalid flags).

**Tests (Detailed)**
- Python tests: none.
- Rust tests: integration tests with a mocked debugging server (or golden responses).
- Cross-language parity test: compare outputs against Python client for same server responses.

**Gaps / Notes**
- Depends on `requests` and protobuf text parsing; Rust needs equivalent libraries.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 217
- Purpose/role: Flask server that exposes debugging endpoints to fetch variable values and feature embeddings from a running training cluster using saved graph metadata.
- Key symbols/classes/functions: `DebuggingWorker`, `create_app`, `/debugging/variables`, `/debugging/features`, `main`.
- External dependencies: Flask, TensorFlow server/meta_graph import, `debugging_info_pb2`, `embedding_hash_table_pb2`, custom ops `monolith_hash_table_lookup_entry`.
- Side effects: spins up TF server, reads model_dir debugging info, imports meta graph, starts Flask server.

**Required Behavior (Detailed)**
- Flags:
  - `--host`, `--port`, `--model_dir` required to bind server and load debugging info.
- `DebuggingWorker(model_dir)`:
  - Reads `DebuggingInfo` proto from `utils.get_debugging_info_file_name(model_dir)`.
  - Starts a local TF server and builds a fake worker cluster where the last worker is the local server; chief/ps addresses come from debugging info.
  - Builds `feature_name_config_map` from debugging info and initializes a `MergedMultiTypeHashTable` with a dummy factory.
  - Creates session config via `cluster_manager.generate_session_config` and imports the meta graph from `utils.get_meta_graph_file_name(model_dir)`.
- `fetch_variables(variable_names)`:
  - Filters requested variables to those present in imported graph; returns dict name → stringified value.
- `fetch_features(feature_names, feature_ids)`:
  - Maps feature names to merged table names via `slot_mapping` in merged table.
  - Buckets by PS index using `fid % num_ps`.
  - For each table/ps index, fetches EntryDump via `monolith_hash_table_lookup_entry`.
  - Parses EntryDump bytes to text proto and returns dict `feature_name -> {fid: textproto}`.
- `create_app()`:
  - Constructs Flask app and a `DebuggingWorker`.
  - `/debugging/variables`: expects JSON `variable_names`; returns `{status, msg}` with msg JSON string.
  - `/debugging/features`: expects JSON `feature_names` + `feature_ids` (same length); returns `{status, msg}`.
  - On exceptions, returns `status=fail` with traceback in msg.
- `main`:
  - Calls `env_utils.setup_hdfs_env()` and runs Flask app.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/debugging` (server) + TF backend bindings for graph/variables.
- Rust public API surface: debugging server binary exposing `/debugging/variables` and `/debugging/features`.
- Data model mapping: DebuggingInfo proto, EntryDump parsing, table lookup by PS index.
- Feature gating: requires TF runtime + custom ops for hash table lookup.
- Integration points: debugging_client CLI, model_dir metadata generation.

**Implementation Steps (Detailed)**
1. Implement a Rust server (e.g., axum/warp) with matching endpoints and JSON payloads.
2. Load DebuggingInfo and meta graph; create TF session with cluster config matching Python.
3. Implement variable lookup and table lookup logic with PS sharding by `fid % num_ps`.
4. Return responses with identical JSON structure and error handling.

**Tests (Detailed)**
- Python tests: none.
- Rust tests: integration tests with mocked debugging info + TF session (if feasible).
- Cross-language parity test: compare responses for same model_dir and queries.

**Gaps / Notes**
- Requires TF runtime and custom ops; Rust implementation may need to shell out or depend on TF C API.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 57
- Purpose/role: Minimal demo entrypoint to run a local CPU training job for `TestFFMModel` with export enabled.
- Key symbols/classes/functions: `main`, CLI flags `num_ps`, `model_dir`.
- External dependencies: `cpu_training.local_train`, `TestFFMModel`, `ExportMode`.
- Side effects: launches a training run, writes checkpoints/exports to `model_dir`.

**Required Behavior (Detailed)**
- CLI flags:
  - `--num_ps`: number of parameter servers; `0` runs locally.
  - `--model_dir`: output directory.
- `main`:
  - Builds params via `TestFFMModel.params()` and sets:
    - `params.name = 'test_ffm_model'`
    - `params.train.per_replica_batch_size = 64`
    - `params.serving.export_when_saving = True`
    - `params.serving.export_mode = ExportMode.DISTRIBUTED`
  - Calls `cpu_training.local_train(..., steps=100, save_checkpoints_steps=50)`.
- Script mode: enables INFO logging and disables eager execution.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/examples` (or CLI).
- Rust public API surface: demo binary that runs a comparable CPU training flow.
- Data model mapping: params/config to Rust training config.
- Feature gating: requires training pipeline parity with Python.
- Integration points: `cpu_training` equivalent and model definition in Rust.

**Implementation Steps (Detailed)**
1. Implement a Rust demo that configures an equivalent model and training loop.
2. Mirror flags (`num_ps`, `model_dir`) and default values.
3. Ensure checkpoint/export cadence matches (`steps=100`, `save_checkpoints_steps=50`).

**Tests (Detailed)**
- Python tests: none.
- Rust tests: optional smoke test that runs a short training stub.
- Cross-language parity test: compare produced artifacts for a short run (if feasible).

**Gaps / Notes**
- Depends on `TestFFMModel` and `cpu_training` parity in Rust.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 457
- Purpose/role: Custom checkpoint restore logic for dense variables, including aliasing/mapping between old and new variable names and partitioned variable splitting.
- Key symbols/classes/functions: `CustomRestoreListener`, `add_mapping_rules`, `node_name`, `get_new_name`, `get_guess_name`, `split_name`, `calc_reorder_info`, `get_full_prefix`, `update_var_name_mapping_for_dense`, `infer_variable_name`, `calc_feed_dict`.
- External dependencies: TensorFlow checkpoint reader, `CheckpointRestorerListener`, `is_exporting`, numpy, regex patterns.
- Side effects: inspects checkpoint files, builds custom restore ops in graph collections, logs extensive info, may create `clear_nn` flag file logic.

**Required Behavior (Detailed)**
- Globals/regex:
  - `CUSTOM_RESTORE_OP` collection key and `CustomRestoreListenerKey` name.
  - `PAT` matches `.../part_<num>/...` for partitioned vars.
  - `DensePat` matches dense layer names for bias/kernel/trainable_kernel_norm.
  - `_NameMapping` regex rules for special-case name conversions; `add_mapping_rules` merges additional regex patterns.
- `node_name(name)`:
  - Strips whitespace, trailing `/`, leading `^`, and `:0` suffix if numeric.
- `get_new_name(name)`:
  - Deduplicates repeated path terms in a name (preserving order) and rejoins with `/`.
- `get_guess_name(name)`:
  - Applies `_NameMapping` regex patterns; returns formatted guess if matched, else original.
- `split_name(name)`:
  - Splits trailing digits; returns `(base, int_suffix)` or `(name, 0)` if none.
- `calc_reorder_info(names, is_ordered=True)`:
  - Optionally sorts by numeric suffix.
  - Returns `(need_reorder, base)` where base is `dense_` for base name `dense` else base name; `need_reorder` when suffix sequence isn't contiguous starting at 0/1 or when multiple names.
- `get_full_prefix(short_prefix, prefix_set)`:
  - Chooses the longest prefix in `prefix_set` that ends with `short_prefix`.
- `update_var_name_mapping_for_dense(var_name_mapping)`:
  - Groups dense layer vars by prefix/dense_name/bias; uses `DensePat` to normalize names.
  - For dense layers with multiple indices, may reorder and rename to `dense_{i}` or base name.
  - Ensures bias entries are present; fills missing entries into `var_name_mapping`.
- `CustomRestoreListener`:
  - `__init__`: accepts `alias_map`, `clear_nn`, `continue_training`, `model_dir`, `enable_alias_map_auto_gen` (defaults True).
  - `begin()`:
    - Skip if `is_exporting()`.
    - Loads checkpoint state from `model_dir`; sets `ckpt_name`.
    - If `clear_nn`:
      - Uses `clear_nn` flag file to skip if present.
      - Adds `global_variables_initializer` to `CUSTOM_RESTORE_OP`; if `continue_training`, adds placeholder + assign op for global_step.
    - Else if `_need_build_custom_init_graph(variables)`:
      - Creates placeholders and assign ops for each variable; stores placeholders + alias map into `CUSTOM_RESTORE_OP`.
  - `_need_build_custom_init_graph(variables)`:
    - Auto-generates alias_map when not provided and enabled:
      - Reads ckpt var names; checks compatibility by removing `/part_<n>`.
      - Builds `var_name_mapping` from `get_new_name(old_name)` to `old_name` and refines via `update_var_name_mapping_for_dense`.
      - Builds `alias_map` for each variable; handles missing dense names with `miss_dense_names` / `miss_dense_map`.
      - For unresolved names, uses `get_guess_name` or `miss_dense_map`; if still missing, logs warning and returns False.
    - Returns True if any variable name is not covered by alias_map values.
- `infer_variable_name(names)`:
  - Removes `/part_<n>` segments to infer merged variable names.
- `calc_feed_dict(ckpt, alias_map, placeholders)`:
  - Builds reverse map old_name → list of new variable names.
  - If inferred new names all exist in checkpoint, returns None (no alias restore needed).
  - Otherwise, builds feed dict mapping placeholders to ckpt tensors.
  - For partitioned vars (multiple new names):
    - Handles dense name grouping and ordering.
    - Sorts by partition index extracted via `PAT`.
    - Splits old tensor by first-dimension sizes from placeholders (`np.split`) and assigns each split to its placeholder.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/checkpoint` (restore hooks) + `monolith-checkpoint` utilities.
- Rust public API surface: custom restore listener/hook that can build alias maps and feed dicts.
- Data model mapping: checkpoint variable names → current graph variable names, including partitioned tensors.
- Feature gating: requires TensorFlow checkpoint reader or compatible reader in Rust.
- Integration points: `basic_restore_hook` and training session initialization.

**Implementation Steps (Detailed)**
1. Implement name normalization helpers (`node_name`, `get_new_name`, `split_name`, `get_guess_name`) in Rust.
2. Port dense name mapping logic (`update_var_name_mapping_for_dense`) including reorder rules and prefix resolution.
3. Implement alias-map auto generation using checkpoint metadata and dense mappings.
4. Build custom restore ops/feeds with placeholders and assign ops; support `clear_nn` + `continue_training` global step update.
5. Implement partitioned variable splitting logic equivalent to `calc_feed_dict`.

**Tests (Detailed)**
- Python tests: `dense_reload_utils_test.py`.
- Rust tests: unit tests for name mapping, alias generation, and feed dict splitting.
- Cross-language parity test: use a sample ckpt with renamed vars and ensure alias restore works identically.

**Gaps / Notes**
- Heavy TF internals: requires checkpoint reader and graph variable manipulation in Rust.
- Auto alias mapping may be fragile; parity requires matching regex and reorder heuristics exactly.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 192
- Purpose/role: Tests for dense reload utilities: variable name inference, feed dict splitting for partitioned vars, and custom restore listener modes.
- Key symbols/classes/functions: `DenseReloadUtilsTest`, `setUpClass`, `test_infer_variable_name`, `test_calc_feed_dict`, `test_alias_map_listener`, `test_clear_nn_listener`.
- External dependencies: TensorFlow, `GlorotNormal`, `Ones`, `infer_variable_name`, `calc_feed_dict`, `CustomRestoreListener`.
- Side effects: creates and deletes checkpoint files under `./ckpt`.

**Required Behavior (Detailed)**
- `setUpClass`:
  - Builds a graph with `global_step`, a partitioned variable `partition` (shape 1280x512), and `small_var`.
  - Saves checkpoint `ckpt/test-<global_step>` in cwd.
- `tearDownClass`:
  - Removes `./ckpt` directory if exists.
- `test_infer_variable_name`:
  - Creates a partitioned variable and checks `infer_variable_name` removes `/part_xx` to yield `{partition_var.name:0}`.
- `test_calc_feed_dict`:
  - Creates partitioned `partition2` and `small_var2`.
  - Builds `alias_map` mapping new names to old checkpoint names (`small_var2` → `small_var`, `partition2 parts` → `partition`).
  - Creates placeholders with `origin_name` for each var/partition.
  - `calc_feed_dict` returns mapping for each alias; asserts shapes match partition shapes.
- `test_alias_map_listener`:
  - Builds same alias_map/placeholders and calls `CustomRestoreListener(alias_map=..., model_dir=./ckpt).begin()` (no asserts, just should not error).
- `test_clear_nn_listener`:
  - Creates `CustomRestoreListener(clear_nn=True, model_dir=./ckpt)` and calls `begin()`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests`.
- Rust public API surface: dense reload utilities and custom restore listener.
- Data model mapping: checkpoint vars/partitioned vars to Rust checkpoint reader and feed dict logic.
- Feature gating: requires checkpoint reader and graph variable introspection.
- Integration points: `dense_reload_utils.py` implementation.

**Implementation Steps (Detailed)**
1. Build Rust tests that create a checkpoint with partitioned variables (or mock the reader).
2. Verify `infer_variable_name` removes partition suffixes.
3. Validate `calc_feed_dict` splitting behavior for partitioned variables.
4. Ensure custom restore listener handles alias_map and clear_nn without error.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: `dense_reload_utils_test.rs` analog with temp directories.
- Cross-language parity test: compare feed dict splits on a shared checkpoint.

**Gaps / Notes**
- The Python tests rely on TF checkpoint creation; Rust tests may need to use Python-generated checkpoints.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 231
- Purpose/role: Device placement utilities for training/serving, including GPU gating, device functions, and MPI/PS placement logic.
- Key symbols/classes/functions: `enable_gpu_training`, `disable_gpu_training`, `is_gpu_training`, `get_visible_gpus`, `default_device_fn`, `maybe_device_if_allowed`, `within_placement_context_of`, `get_device_fn`, `input_device_fn`, `model_device_fn`, `serving_input_device_fn`, `skip_device`.
- External dependencies: TensorFlow DeviceSpec, `device_setter`, MPI rank helper `get_mpi_rank`, flags (`num_ps`, `enable_gpu_training`, `enable_sync_training`, `is_local`).
- Side effects: global `_GPU_PLACEMENT_ALLOWED` flag; influences device placement for ops.

**Required Behavior (Detailed)**
- GPU training flag:
  - `_GPU_PLACEMENT_ALLOWED` default False; `enable_gpu_training()` sets True; `disable_gpu_training()` sets False; `is_gpu_training()` returns it.
- `get_visible_gpus(local_rank, processes_per_gpu=1)`:
  - Ensures `processes_per_gpu` is int >= 1; returns string of `local_rank / processes_per_gpu` as GPU index.
- `_device_rule(device_name)`:
  - Returns `/device:CPU:0` when `device_name` is empty.
  - If assigned GPU but `_GPU_PLACEMENT_ALLOWED` is False or device type empty, merges with default CPU while keeping job/task/replica.
- `skip_device(op)`:
  - Returns True for summary ops (`Write*`, `*Summary`) or string `Const` ops.
- `default_device_fn(op)`:
  - Returns CPU for skipped ops; otherwise applies `_device_rule` to op device.
- `maybe_device_if_allowed(device_name)`:
  - Context manager that forces device via `_device_rule` to prevent unintended GPU placement.
- Placement context helpers:
  - `_FakeOp` and `within_placement_context_of(device_name)` check current placement via graph `_apply_device_functions`.
- `get_device_fn(cluster=None, task=None)`:
  - Determines MPI mode via `OMPI_COMM_WORLD_LOCAL_RANK`.
  - Chooses GPU vs CPU based on `FLAGS.enable_gpu_training` or `_GPU_PLACEMENT_ALLOWED`.
  - If sync training + MPI + PS: builds device spec for chief/worker based on rank and returns custom `_device_fn` that merges with op device.
  - If sync training but no PS: returns `default_device_fn`.
  - If async (no sync training):
    - Returns None for local mode or missing cluster/task.
    - Else uses `tf.compat.v1.train.replica_device_setter` with `ps_tasks=FLAGS.num_ps` and standard PS ops.
- `input_device_fn(op)`:
  - In MPI+PS+sync training returns `/job:chief|worker/replica:0/task:<idx>/device:CPU:0`, else CPU.
- `model_device_fn(op)`:
  - Similar to `_device_fn` but for model scope; uses GPU if enabled, else CPU; respects op.device and `_class` attr.
- `serving_input_device_fn(op)`:
  - Uses op.device if set, else CPU.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/device`.
- Rust public API surface: device placement utilities and device function factories.
- Data model mapping: TF DeviceSpec strings → Rust device strings used by TF runtime bindings.
- Feature gating: GPU placement gate, MPI/PS sync training, replica device setter behavior.
- Integration points: training config, session creation, input pipelines.

**Implementation Steps (Detailed)**
1. Implement GPU gating and visible GPU computation.
2. Implement device rule merging logic with default CPU and job/task retention.
3. Provide Rust equivalents of `get_device_fn`/`input_device_fn`/`model_device_fn` for sync/async modes.
4. Mirror skip-device rules for summary ops and string const.
5. Add placement-context helper or document unsupported if TF internals unavailable.

**Tests (Detailed)**
- Python tests: `device_utils_test.py`.
- Rust tests: unit tests for device rules, GPU gating, and MPI/PS device fn outputs.
- Cross-language parity test: compare device string outputs under fixed flag/env combinations.

**Gaps / Notes**
- Depends on TF internal device functions; Rust may need to mimic device strings rather than enforcing in graph.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 104
- Purpose/role: Tests device placement rules and GPU gating in `device_utils`.
- Key symbols/classes/functions: `DeviceUtilsTest` and methods `test_basic`, `test_cpu_only`, `test_str_context`, `test_str_nested_contexts`, `test_cpu_device_merge`, `test_gpu_device_merge`, `test_process_gpu_map`.
- External dependencies: TensorFlow, `device_utils`.
- Side effects: none.

**Required Behavior (Detailed)**
- `test_basic`: default device function places constants on `/device:CPU:0`.
- `test_cpu_only`: when GPU training disabled, explicit GPU device request is overridden to CPU.
- `test_str_context`: with GPU enabled, bare constants default to CPU, `tf.device("GPU:0")` forces GPU:0.
- `test_str_nested_contexts`: nested device contexts maintain correct placement for CPU/GPU overrides.
- `test_cpu_device_merge`: with GPU disabled, device job/task merged with CPU; `within_placement_context_of` reports CPU.
- `test_gpu_device_merge`: with GPU enabled, device job/task merged with GPU; `maybe_device_if_allowed` forces GPU:1 placement and context checks.
- `test_process_gpu_map`: `get_visible_gpus` returns expected indices for local_rank/processes_per_gpu combinations.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests`.
- Rust public API surface: device_utils functions.
- Data model mapping: device strings matching TF conventions.
- Feature gating: GPU training toggle.
- Integration points: training device placement.

**Implementation Steps (Detailed)**
1. Add Rust tests to assert device string outputs for each scenario.
2. Verify GPU gating overrides explicit GPU placement when disabled.
3. Validate visible GPU mapping logic.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: `device_utils_test.rs`.
- Cross-language parity test: compare device string outputs and placement context behavior.

**Gaps / Notes**
- Python tests rely on TF device placement; Rust tests may need to compare string outputs rather than actual TF graph placement.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 81
- Purpose/role: Builds a dynamic sharding dataset that expands glob patterns on demand using shared queues and a TF session-backed generator.
- Key symbols/classes/functions: `create_dynamic_sharding_dataset`.
- External dependencies: `str_queue.StrQueue`, `session_hooks.get_current_session`, `utils.ps_device`, `native_task_context`.
- Side effects: creates shared queues on PS0 or host; uses TF session to dequeue filenames.

**Required Behavior (Detailed)**
- `create_dynamic_sharding_dataset(glob_patterns, name)`:
  - Creates two shared string queues:
    - `glob_patterns_queue`: seeded with glob patterns.
    - `filenames_queue`: auto-enqueue filenames by expanding patterns.
  - Chooses device on PS0 if `num_ps > 0`, else default device.
  - `glob_pattern()` (tf.function): dequeues a pattern; if not out_of_range, calls `tf.io.matching_files`; else returns `""` and out_of_range.
  - `filenames_queue.dequeue()` returns `(filename_bytes, out_of_range)`.
  - `filename_generator()` runs dequeue via current session; raises `StopIteration` on out_of_range; else decodes bytes to string.
  - Builds `dataset_ops.MapDataset` over a dummy infinite dataset; maps to `tf.py_function(filename_generator)` with `preserve_cardinality=False`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/src`.
- Rust public API surface: dynamic sharding dataset builder for file patterns.
- Data model mapping: file pattern → stream of file paths.
- Feature gating: requires session hooks/queues or Rust equivalents.
- Integration points: `datasets.py` uses this for dynamic sharding.

**Implementation Steps (Detailed)**
1. Implement a Rust dynamic sharding iterator that expands patterns lazily.
2. Support shared queue semantics for multi-worker coordination (or document limitation).
3. Ensure out_of_range yields end of stream and map preserves non-cardinality.

**Tests (Detailed)**
- Python tests: `distributed_dataset_test.py`.
- Rust tests: unit tests for pattern expansion order and termination.
- Cross-language parity test: compare file lists produced for a given glob set.

**Gaps / Notes**
- Relies on TF session and custom `StrQueue`; Rust needs a coordinated queue for distributed use.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 124
- Purpose/role: Tests dynamic sharding dataset expansion, EOF handling, composition with TextLineDataset, and iterator save/restore.
- Key symbols/classes/functions: `DynamicShardingDatasetTest`, `gen_test_files`, `testBasic`, `testEof`, `testWithOtherDataset`, `testSaveRestore`.
- External dependencies: TensorFlow, `distributed_dataset.create_dynamic_sharding_dataset`, `session_hooks.SetCurrentSessionHook`.
- Side effects: writes temp files under `TEST_TMPDIR` and saves/loads iterator checkpoints.

**Required Behavior (Detailed)**
- `gen_test_files(files_dir)`:
  - Creates files `a_0.txt`..`e_1.txt` with two lines each: `a.0.0`, `a.0.1`, etc.
- `setUp`:
  - Uses `TEST_TMPDIR` and creates data dir + files if missing.
  - Builds glob patterns `a_*.txt`..`e_*.txt`.
- `get_test_session()`:
  - Returns `SingularMonitoredSession` with `SetCurrentSessionHook`.
- `testBasic`:
  - Reads 10 filenames from dynamic sharding dataset; expects ordered list of `a_0..e_1` full paths.
- `testEof`:
  - With empty patterns, iterator should raise `OutOfRangeError`; verifies dependent op does not mutate variable `v`.
- `testWithOtherDataset`:
  - `filename_dataset.flat_map(TextLineDataset)` yields lines; first three lines are `a.0.0`, `a.0.1`, `a.1.0`.
- `testSaveRestore`:
  - Creates saveable iterator; reads `a.0.0`, saves; reads `a.0.1`, restores; next read is still `a.0.1`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/tests`.
- Rust public API surface: dynamic sharding dataset + iterator save/restore (if supported).
- Data model mapping: file list → dataset → line dataset.
- Feature gating: iterator save/restore may depend on TF runtime.
- Integration points: `distributed_dataset` implementation.

**Implementation Steps (Detailed)**
1. Implement Rust tests that generate temp files and validate ordered filename emission.
2. Verify EOF behavior for empty patterns.
3. Test composition with line reader and save/restore semantics.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: `distributed_dataset_test.rs` with tempdir fixtures.
- Cross-language parity test: compare file order and resume position after restore.

**Gaps / Notes**
- `saveable` iterator behavior is TF-specific; Rust may need explicit checkpointing support.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 114
- Purpose/role: A TF-based string queue with save/restore support, critical section synchronization, and optional auto-enqueue when empty.
- Key symbols/classes/functions: `StrQueue`, `enqueue_many`, `dequeue`, `_raw_enqueue_many`, `_raw_dequeue`.
- External dependencies: TensorFlow `CriticalSection`, variables, tf.function.
- Side effects: maintains internal TF variables (`_arr`, `_offset`, `_arr_size`), uses critical section for synchronization.

**Required Behavior (Detailed)**
- `StrQueue.__init__(initial_elements, critical_section, auto_enqueue_fn, capacity, name)`:
  - Creates a shared `CriticalSection` (or reuses provided).
  - Initializes `_arr` (string array of size `capacity`), `_offset`, `_arr_size` as variables.
  - Enqueues `initial_elements` during initialization via control deps.
  - Uses `_var_for_init` dummy variable to ensure initial enqueue runs.
- `enqueue_many(elements)`:
  - Converts to string tensor and calls `_raw_enqueue_many` inside critical section.
- `dequeue()`:
  - Executes `_raw_dequeue` inside critical section; returns `(element, out_of_range)`.
- `_raw_enqueue_many(elements)` (tf.function):
  - Computes `old_arr_size = _arr_size - _offset`, `new_arr_size = old_arr_size + size(elements)`.
  - Asserts `new_arr_size <= capacity`.
  - Compacts array by shifting remaining elements to front, appends new elements, resets `_offset` to 0, updates `_arr_size`.
- `_raw_dequeue()` (tf.function):
  - Asserts `_offset <= _arr_size`.
  - If `auto_enqueue_fn` provided, loops while empty: calls auto fn to get `(elements, out_of_range)`; enqueues elements unless out_of_range.
  - If still empty, returns `("", True)`.
  - Else returns element at `_offset` and increments `_offset`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/src` (distributed queues).
- Rust public API surface: string queue with enqueue/dequeue and optional auto-fill.
- Data model mapping: TF variables → Rust in-memory or shared queue state.
- Feature gating: requires distributed synchronization if used across workers.
- Integration points: `distributed_dataset.create_dynamic_sharding_dataset`.

**Implementation Steps (Detailed)**
1. Implement a thread-safe queue with capacity and offset/size semantics.
2. Provide auto-enqueue hook that is called when empty.
3. Match out_of_range behavior and empty return value (`""`).
4. If using TF runtime, preserve CriticalSection semantics for shared state.

**Tests (Detailed)**
- Python tests: `str_queue_test.py`.
- Rust tests: queue enqueue/dequeue, auto-enqueue loop, capacity assert.
- Cross-language parity test: compare sequence of dequeued elements for the same auto-enqueue function.

**Gaps / Notes**
- TF CriticalSection semantics may need a custom mutex + barrier if implemented in Rust.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 67
- Purpose/role: Tests basic enqueue/dequeue behavior, initialization, out-of-range handling, and auto-enqueue logic for `StrQueue`.
- Key symbols/classes/functions: `QueueTest` with `testBasic`, `testInit`, `testOutOfRange`, `testAutoEnqueue`.
- External dependencies: TensorFlow, `str_queue.StrQueue`.
- Side effects: none.

**Required Behavior (Detailed)**
- `testBasic`:
  - Enqueues `test1`, `test2` and dequeues in order.
- `testInit`:
  - Initializes queue with `initial_elements=['test1']` and dequeues `test1`.
- `testOutOfRange`:
  - Dequeue from empty queue returns `out_of_range=True`.
- `testAutoEnqueue`:
  - `auto_enqueue` increments variable `v` and enqueues stringified values until `v > 2`, then returns out_of_range.
  - Dequeues yield `"1"`, `"2"`, then `out_of_range=True` for subsequent dequeues.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/tests`.
- Rust public API surface: `StrQueue` equivalent.
- Data model mapping: string queue semantics.
- Feature gating: none.
- Integration points: `distributed_dataset` uses StrQueue.

**Implementation Steps (Detailed)**
1. Add Rust tests that validate enqueue/dequeue ordering and init behavior.
2. Implement auto-enqueue hook test with controlled counter.
3. Verify out_of_range behavior persists after exhaustion.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: `str_queue_test.rs`.
- Cross-language parity test: compare dequeued sequences for fixed auto-enqueue behavior.

**Gaps / Notes**
- TensorFlow session semantics are not required; Rust can implement a pure queue test.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 2108
- Purpose/role: Distributed parameter server embedding hash table implementation (single-type and multi-type), sharded lookup/apply gradients, fused layout embedding pipelines, and GPU/Horovod/BytePS all-to-all paths.
- Key symbols/classes/functions: `ps_device`, `DistributedHashTable`, `DistributedMultiTypeHashTable`, `PartitionedHashTable`, `get_sub_table_name`, `PartitionedHashTable.gen_feature_configs`, `merge_feature_config`, `lookup`, `apply_gradients`, `_lookup_gpu`, `_apply_gradients_gpu`.
- External dependencies: TensorFlow, custom ops (`distribution_ops`, `multi_hash_table_ops`), `export_context`, `prefetch_queue`, `hvd`/`bps` (if enabled), FeatureConfigs protos.
- Side effects: creates PS-side graphs/signatures during export; emits lookup timers; enqueues prefetch queues; uses global parser context for sharding configs.

**Required Behavior (Detailed)**
- `ps_device(i)`:
  - Context manager that clears device stack (`colocate_with(None, True)`) and sets device to `utils.ps_device(i)` for PS ops.
- `DistributedHashTable` (single-type table):
  - Constructor builds per-PS tables; sends learning-rate tensors to each PS (unless exporting standalone).
  - `lookup(ids)`:
    - `tf.unique` ids, shard by `id % ps_num`, lookup on each PS, and `map_id_to_embedding` back to original order.
    - Tracks input/output tensors for backprop.
  - `assign/assign_add`: split ids/values by PS and call underlying table method.
  - `apply_gradients`: unique ids, split gradients with `map_id_to_embedding_gradient_back_prop`, apply on each PS (dedup disabled).
  - `as_op`: aggregates PS table ops.
- `DistributedMultiTypeHashTable` (multi-slot table):
  - Builds per-PS multi-type tables; supports raw API if tables are `RawMultiTypeHashTable`.
  - Export mode builds PS subgraphs with `lookup` and optional `raw_lookup` signatures.
  - `lookup(slot_to_id)`:
    - Raw API path: uses ragged IDs and `unique_key_with_value_and_offset` to reduce duplicate lookups, splits by PS, uses raw lookup and `fill_with_offset_map`, then reconstructs embeddings; returns only requested slots.
    - Non-raw path: per-slot unique/split, PS lookup (remote_predict if exporting distributed); maps back by `map_id_to_embedding`.
  - `assign/assign_add`: per-slot split by PS and call underlying table methods.
  - `reinitialize(slot, ids)`: raw-only; splits ids and concatenates status.
  - `apply_gradients`: raw path uses `raw_apply_gradients` with fused flat grads; non-raw path packs keyed tensors, optional float16 transfer.
  - `as_op` combines PS tables; `get_table_dim_sizes` delegates to cc dims.
- `get_sub_table_name(strs)`:
  - Returns `(concat, md5(concat))` for merged table naming.
- `PartitionedHashTable`:
  - `gen_feature_configs`: builds `FeatureConfigs` and `ShardingSparseFidsOpParams` based on feature configs and combiners; supports native multi-hash-table and GPU embedding modes.
  - `merge_feature_config` / `no_merge_feature_config`: compute merged sub-table names (with md5) or keep per-feature tables; handles `fc_slot_` → `slot_` extra restore names.
  - Constructor:
    - Reads `parser_ctx.sharding_sparse_fids_op_params` for PS count, native multi-table mode, feature configs, and GPU options.
    - Creates per-PS tables or GPU table; builds export signatures for lookup/raw_lookup when exporting.
    - Sets up learning-rate tensors for each sub-table.
  - `lookup(features, auxiliary_bundle, ...)`:
    - If GPU embedding enabled, delegates to `_lookup_gpu` and optionally returns callable.
    - Otherwise obtains sharded fids via `sharding_sparse_fids` or `ParserCtx`-encoded features, stores offsets and sizes in `auxiliary_bundle`.
    - Optionally returns `lookup_callable_fn` or `fused_layout_callable_fn` for two-phase lookup.
    - `call_lookup`:
      - Uses raw/native lookup or packed lookup; remote_predict in export mode.
      - Stores per-PS embeddings and optional fids/row_splits in `auxiliary_bundle`.
      - Optionally moves auxiliary tensors to GPU and enqueues prefetch queues.
    - `fused_layout_callable_fn`:
      - Calls `distribution_ops.fused_embedding_to_layout` to reconstruct layout embeddings (CPU/GPU depending on export or `_use_gpu`).
      - Uses `nest_layout` to produce output dict.
  - `apply_gradients(layout_grads_and_vars, global_step, req_time, auxiliary_bundle, async_function_mgr, async_push, grad_scale)`:
    - For non-GPU path: uses `fused_embedding_to_layout_grad` to compute per-PS grads, then applies via raw or packed update; supports async push queues.
    - Includes tensor move CPU helper for GPU-derived tensors.
    - For GPU path, delegates to `_apply_gradients_gpu`.
  - `_lookup_gpu`:
    - Uses all-to-all (HVD/BPS/custom) to exchange ids and embeddings; calls `fused_lookup` on GPU table; then all-to-all embeddings back; finally `fused_embedding_to_layout` (version 4) and `nest_layout`.
    - Populates `auxiliary_bundle` with many intermediate tensors (id_flat_t, splits, offsets, recv embeddings, etc.) and optional pipeline queues.
  - `_apply_gradients_gpu`:
    - Computes `fused_embedding_to_layout_grad` on GPU, performs all-to-all backprop (HVD/BPS/custom), and calls `fused_apply_gradient` on GPU table; supports async optimize queues.
  - `assign/assign_add`:
    - Non-GPU only; routes to `_update` or `_native_hash_table_update` depending on native multi-table mode.
  - `flatten_layout` / `nest_layout`:
    - Deterministic ordering by `feature_configs.out_configs` (sorted names); `OutType.NONE` yields list per slices.
  - Queue hooks:
    - `add_queue_hook` stores local hooks; `get_queue_hooks` collects hooks from tables.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/ps` + `monolith-hash-table` + `monolith-data`.
- Rust public API surface: distributed hash table abstractions, partitioned multi-type tables, lookup/apply_gradients APIs.
- Data model mapping: FeatureConfigs and sharded fids, embedding layouts, and fused layout conversions.
- Feature gating: export mode, raw API support, GPU embedding, Horovod/BytePS all-to-all.
- Integration points: `parsers.sharding_sparse_fids`, `embedding_combiners`, `prefetch_queue`, export signatures.

**Implementation Steps (Detailed)**
1. Implement Rust equivalents for `DistributedHashTable` and `DistributedMultiTypeHashTable` with sharding by `id % ps_num`.
2. Recreate packed tensor transfer and optional float16 transport.
3. Implement `PartitionedHashTable` with sharding feature configs and `fused_embedding_to_layout`/`_grad` equivalents.
4. Add GPU embedding path + all-to-all (if supported); otherwise gate behind features.
5. Mirror export signatures for PS-side lookups.
6. Port queue hook logic for pipelined execution.

**Tests (Detailed)**
- Python tests: `distributed_ps_test.py`, `distributed_ps_sync_test.py`, `distribution_ops_test.py`.
- Rust tests: integration tests for lookup/apply_gradients with small sharded tables and layout configs.
- Cross-language parity test: compare embedding outputs for fixed ids across PS shards.

**Gaps / Notes**
- This module is large and deeply tied to TF custom ops; full parity likely requires a TF backend or substantial Rust kernel work.
- GPU/Horovod/BytePS paths are specialized; may need staged parity plan.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 168
- Purpose/role: Benchmark tests for distributed hash table lookup and apply_gradients performance, optionally with profiling.
- Key symbols/classes/functions: `_generate_config`, `_get_vocab_hash_table_factory`, `DistributedHashTableTest.lookup`, `DistributedHashTableTest.apply_gradients`.
- External dependencies: TensorFlow local servers, `distributed_ps.DistributedHashTable`, `hash_filter_ops`, `hash_table_ops`, `embedding_hash_table_pb2`.
- Side effects: creates local PS servers, may write profiler logs under `/tmp/distributed_ps_benchmark`.

**Required Behavior (Detailed)**
- `_generate_config(servers, job_name=utils.PS_JOB_NAME)`:
  - Builds `ClusterDef` with job tasks derived from server targets; returns `ConfigProto`.
- `_get_vocab_hash_table_factory(dim)`:
  - Returns factory that builds a hash table with `EmbeddingHashTableConfig` using cuckoo + SGD(1.0) + zeros init and segment dim `dim`.
- `DistributedHashTableTest.lookup(enable_dedup, real_run=True)`:
  - Creates `ps_num=10` local servers; uses server0 with cluster config.
  - Builds hash filters and a `DistributedHashTable`, assigns add for ids 0..num_elements-1.
  - If `real_run`: lookup ids `x//2`, check values equal `x//2` repeated per dim; prints wall time; optional profiler.
  - If `real_run=False`: just runs `hash_table.as_op()` to measure overhead.
- `apply_gradients(real_run=True)`:
  - Similar setup; assigns ones to embeddings, looks up, computes `loss=0.3*embeddings`, grads; applies gradients.
  - If `real_run`: after apply, looks up and expects values `0.4` (1.0 + 0.3*?); prints timing.
  - If `real_run=False`: checks grads equal `0.3` if not profiling.
- Tests invoke both real and overhead modes.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/benches`.
- Rust public API surface: benchmark harness for distributed PS table lookup/apply gradients.
- Data model mapping: hash table config, embedding values.
- Feature gating: requires distributed PS runtime and hash table ops.
- Integration points: `distributed_ps` implementation.

**Implementation Steps (Detailed)**
1. Implement a Rust benchmark that spins up local PS servers (or mock) and measures lookup/apply_gradients.
2. Mirror data sizes (1e6 ids, dim=16) and expected outputs.
3. Optionally add profiling hooks matching Python behavior.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: bench-only; optional correctness assertions for small sizes.
- Cross-language parity test: compare outputs for small benchmark sizes.

**Gaps / Notes**
- Uses TF local servers and profiling; Rust may need a simplified harness.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 262
- Purpose/role: Factory helpers to build distributed or local multi-type hash tables and partitioned hash tables with different network/packet-reduction strategies.
- Key symbols/classes/functions: `MultiHashTableFactory`, `create_in_worker_multi_type_hash_table`, `create_multi_type_hash_table`, `create_native_multi_hash_table`, `create_in_worker_native_multi_hash_table`, `create_partitioned_hash_table`.
- External dependencies: `distributed_ps`, `distributed_ps_sync`, `hash_table_ops`, `hash_filter_ops`, `multi_type_hash_table`, `multi_hash_table_ops`, `entry.HashTableConfigInstance`.
- Side effects: none beyond table creation.

**Required Behavior (Detailed)**
- `MultiHashTableFactory`:
  - Caches converted configs via `multi_hash_table_ops.convert_to_cached_config` keyed by `id(slot_to_config)`.
  - `__call__(idx, slot_to_config)` returns `MultiHashTable.from_cached_config` using hash_filter and sync_client for shard `idx`.
- `create_in_worker_multi_type_hash_table(shard_num, slot_to_config, hash_filter, sync_client, queue_configs)`:
  - Builds a `MergedMultiTypeHashTable` whose underlying factory is `DistributedMultiTypeHashTableMpi` (alltoall) created from a per-worker `MultiTypeHashTable` factory.
- `create_multi_type_hash_table(num_ps, slot_to_config, hash_filters, sync_clients, reduce_network_packets, max_rpc_deadline_millis)`:
  - Validates sync_clients length; fills with None if missing.
  - `num_ps==0`: returns local `MergedMultiTypeHashTable` backed by `MultiTypeHashTable` and local hash tables.
  - `reduce_network_packets=False`: uses `DistributedHashTable` per slot within `MultiTypeHashTable` (dedup on worker, distribute to PS).
  - `reduce_network_packets=True`: uses `DistributedMultiTypeHashTable` (multi-type on PS) to reduce RPC count.
- `create_native_multi_hash_table(num_ps, slot_to_config, hash_filters, sync_clients, max_rpc_deadline_millis)`:
  - `num_ps==0`: returns local `MultiHashTable.from_configs`.
  - Else returns `DistributedMultiTypeHashTable` with `MultiHashTableFactory`.
- `create_in_worker_native_multi_hash_table(shard_num, slot_to_config, hash_filter, sync_client, queue_configs)`:
  - Returns `DistributedMultiTypeHashTableMpi` with a local native `MultiHashTable` per shard.
- `create_partitioned_hash_table(num_ps, use_native_multi_hash_table, max_rpc_deadline_millis, hash_filters, sync_clients, enable_gpu_emb, queue_configs)`:
  - Normalizes hash_filters/sync_clients lists.
  - Chooses `multi_type_factory` based on native vs non-native multi-hash table:
    - Native: `MultiHashTableFactory`.
    - Non-native: `MultiTypeHashTable` with hash tables created per PS.
  - Returns `distributed_ps.PartitionedHashTable` with queue configs.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/ps`.
- Rust public API surface: factory functions for hash table backends (local vs distributed).
- Data model mapping: slot configs → hash tables; shard selection logic.
- Feature gating: `reduce_network_packets`, native multi-hash table, GPU embedding.
- Integration points: used by training setup to instantiate embedding tables.

**Implementation Steps (Detailed)**
1. Implement Rust factories mirroring the three strategies (local, distributed per slot, distributed multi-type).
2. Preserve caching for expensive config conversion (if needed).
3. Ensure `num_ps==0` uses local tables and no RPC.
4. Add `PartitionedHashTable` factory with queue config propagation.

**Tests (Detailed)**
- Python tests: `distributed_ps_factory_test.py`.
- Rust tests: unit tests for factory selection logic and returned table types.
- Cross-language parity test: compare selected strategy for given flags.

**Gaps / Notes**
- Depends on TF custom ops for hash tables; Rust must provide equivalent backends.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 87
- Purpose/role: Smoke tests for distributed hash table factory functions; primarily checks that constructors run without errors.
- Key symbols/classes/functions: `_get_test_slot_to_config`, `_get_test_hash_filters`, `FactoryTest`.
- External dependencies: Horovod (enabled via env), TensorFlow PS cluster utilities, `test_utils.generate_test_hash_table_config`.
- Side effects: sets `MONOLITH_WITH_HOROVOD=True`; may initialize Horovod.

**Required Behavior (Detailed)**
- `_get_test_slot_to_config()`:
  - Uses `test_utils.generate_test_hash_table_config(4, learning_rate=0.1)`; returns slot map with keys `"1"`, `"2"`.
- `_get_test_hash_filters(num)`:
  - Returns `hash_filter_ops.create_hash_filters(num, False)`.
- Tests:
  - `test_create_in_worker_multi_type_hash_table*`: calls `create_in_worker_multi_type_hash_table` with hvd initialized.
  - `test_create_multi_type_hash_table_0_ps`: local (no PS) creation.
  - `test_create_multi_type_hash_table_2_ps`: creates PS cluster and calls factory under a session.
  - `test_create_multi_type_hash_table_2_ps_with_reduced_packets`: same with `reduce_network_packets=True`.
  - `test_create_native_multi_hash_table_0_ps` and `_2_ps`: native multi-hash table creation.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests`.
- Rust public API surface: factory functions in `distributed_ps_factory`.
- Data model mapping: slot configs to table instances.
- Feature gating: Horovod/PS cluster support.
- Integration points: distributed PS creation.

**Implementation Steps (Detailed)**
1. Add Rust smoke tests to ensure factory functions are callable in local/PS modes.
2. If Horovod not supported, gate tests accordingly.
3. Verify hash filter creation and config plumbing.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: `distributed_ps_factory_test.rs` (smoke only).
- Cross-language parity test: not required beyond constructor success.

**Gaps / Notes**
- These are grammar/smoke tests, not functional correctness tests.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 531
- Purpose/role: Horovod/BytePS synchronous all-to-all embedding lookup and update for distributed multi-type hash tables.
- Key symbols/classes/functions: `DistributedMultiTypeHashTableMpi.lookup`, `.apply_gradients`, `.as_op`.
- External dependencies: Horovod/BytePS env flags, `distribution_ops`, `feature_utils` (control/dense_opt ops), `prefetch_queue`.
- Side effects: uses enqueue queues for pipelined execution; emits alltoall metrics summaries when enabled.

**Required Behavior (Detailed)**
- Environment flags:
  - `MONOLITH_WITH_HOROVOD`, `MONOLITH_WITH_OPTIMIZED_HOROVOD`, `MONOLITH_WITH_BYTEPS` and related G2G/GDR flags determine alltoall backend and GPU paths.
  - `FLAGS.enable_alltoall_metrics` + `enable_alltoall_metrics_for_slot` control summary emission.
- `DistributedMultiTypeHashTableMpi.__init__(shard_num, table_factory, queue_configs)`:
  - Determines rank from BytePS or Horovod; builds local shard table via `table_factory`.
  - Stores output dims, queue configs, and dependency ops.
- `lookup(slot_to_id, auxiliary_bundle, early_reorder_indicies_res_pack)`:
  - Requires `early_reorder_indicies_res_pack` (support for `reorder_fids_in_data_pipeline=False` dropped).
  - Unpacks `(all_fids, shard_sizes, sharded_slot_sizes, emb_offset_sz, fused_embedding_offsets, req_time)`.
  - Performs alltoall on fids and per-slot sizes via BPS/HVD/custom optimized HVD.
  - Stores key tensors in `auxiliary_bundle` (id_flat_t, id_size_flat_t, emb offsets, recv splits, etc.).
  - Calls `self._table.fused_lookup(...)` on GPU, yielding `fused_embeddings`, splits, offsets, indices.
  - Performs embedding alltoall (fwd) and queues prefetch if configured.
  - Uses `distribution_ops.fused_gather_embeddings_by_input` to assemble per-slot embeddings on GPU.
  - Returns `(slot_to_embedding, auxiliary_bundle)`.
- `apply_gradients(slot_to_grad, auxiliary_bundle, global_step, req_time, scale)`:
  - Uses `feature_utils.control_ops` dependency.
  - Computes `grad_flat` via `fused_gather_embeddings_by_input_gradient`.
  - Optionally casts for BPS bwd.
  - Enqueues async optimize queue if configured.
  - Performs backward alltoall using BPS/HVD/custom optimized HVD.
  - Emits alltoall metrics summaries when enabled.
  - Calls `self._table.fused_apply_gradient` with id/grad buffers and offsets.
  - Supports async optimize queue via `AsyncPushHook`.
- `assign/assign_add/reinitialize`:
  - Not implemented (raises `NotImplementedError`).
- `as_op`:
  - Returns `self._table.as_op` with dependency ops.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/ps`.
- Rust public API surface: synchronous alltoall embedding lookup/update for multi-type tables.
- Data model mapping: packed fid buffers and fused embedding offsets.
- Feature gating: Horovod/BytePS support; GPU alltoall paths.
- Integration points: `distributed_ps_factory.create_in_worker_multi_type_hash_table`.

**Implementation Steps (Detailed)**
1. Implement Rust backend selection for alltoall (HVD/BPS equivalents) or gate feature.
2. Port `fused_lookup` + `fused_gather_embeddings_by_input` and gradient counterparts.
3. Preserve auxiliary_bundle keys and queue-based pipelining.
4. Mirror alltoall metric summaries (if logging/metrics available in Rust).

**Tests (Detailed)**
- Python tests: `distributed_ps_sync_test.py`.
- Rust tests: integration tests with small shard_num and deterministic ids.
- Cross-language parity test: compare embeddings and gradients for small fixtures.

**Gaps / Notes**
- Requires GPU kernels and alltoall comms; may need staged parity.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 47
- Purpose/role: Locate monolith base directory and resolve paths to bundled libraries.
- Key symbols/classes/functions: `find_main`, `get_libops_path`.
- External dependencies: `os` only (explicitly avoids third-party imports).
- Side effects: raises ValueError when base directory cannot be found.

**Required Behavior (Detailed)**
- `find_main()`:
  - Uses `__file__` path; searches for split markers in order: `/__main__/`, `/site-packages/`, `/monolith/`.
  - If marker found:
    - For `/monolith/`: `main_dir` is path prefix before marker.
    - Else: `main_dir` is prefix joined with the marker path component.
  - Returns `main_dir` only if `main_dir/monolith` exists; else raises ValueError with full path in message.
- `get_libops_path(lib_name)`:
  - Returns `os.path.join(find_main(), lib_name)`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/src/path_utils.rs`.
- Rust public API surface: `find_main()` and `get_libops_path()`.

**Implementation Steps (Detailed)**
1. Implement path resolution with the same marker order and behavior.
2. Preserve error message details for diagnostics.

**Tests (Detailed)**
- Python tests: `monolith/utils_test.py` (find_main / get_libops_path).
- Rust tests: replicate `find_main` behavior under test harness.

**Gaps / Notes**
- Behavior depends on Bazel layout (`__main__`); Rust tests should simulate or adjust.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 429
- Purpose/role: TPU training/eval runner using `tf.estimator.tpu.TPUEstimator` with embedding support and optional CPU test mode.
- Key symbols/classes/functions: `TPURunner`, `create_tpu_estimator`, `create_tpu_estimator_on_cpu`, `run`, CLI `main`.
- External dependencies: `tensorflow.compat.v1`, `cloud_tpu_client`, `BaseEmbeddingTask`, `model_registry`.
- Side effects: configures TPU versions, launches training/eval, writes summaries.

**Required Behavior (Detailed)**
- CLI flags include TPU location (`tpu`, `gcp_project`, `tpu_zone`), mode, model_dir, checkpoint interval, iteration counts, embedding options, CPU test, partition strategy, overwrite_end_date.
- `TPURunner.__init__`:
  - Reads flags and task_param; sets accelerator to `tpu`.
  - Allows task_param to override save_checkpoints_steps.
  - Optionally overwrites `train.end_date` when flag provided.
- `create_tpu_estimator`:
  - Optionally configures TPU version and waits for healthy.
  - Builds TPU cluster resolver, computes total replicas and global batch size.
  - Sets TPUConfig with iterations_per_loop and host_call settings.
  - Uses `TPUInfeedOutfeedSessionWithEndOfStreamHandlingHook` when stopping signals enabled.
  - Builds embedding_config_spec if feature/table configs exist.
  - Returns TPUEstimator and total_replicas.
- `create_tpu_estimator_on_cpu`:
  - Creates TPUEstimator with `use_tpu=False` and small batch size; used for CPU test.
- `run()`:
  - Loads global step (or 0).
  - Instantiates task; if BaseEmbeddingTask, prepares feature/table configs.
  - Uses CPU test wrapper if enabled.
  - `train`: `est.train`.
  - `eval`: iterates checkpoints and evaluates; writes summaries and stops at max_steps.
  - `train_and_eval`: not supported (raises TypeError).
- `main`: logs FLAGS and task_param, runs runner.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/tpu_runner.rs`.
- Rust public API surface: runner struct + CLI entrypoint.
- Data model mapping: TPU/embedding configs likely require TF runtime bindings.
- Feature gating: `tf-runtime`, `tpu` features.

**Implementation Steps (Detailed)**
1. Port CLI flag set and task registry lookup.
2. Decide runtime strategy (native Rust vs TF bridge).
3. Preserve TPU version config and host_call/embedding config semantics if using TF.
4. Mirror evaluation-by-checkpoint loop and summary writing.

**Tests (Detailed)**
- Python tests: none specific.
- Rust tests: CLI parsing + control flow tests; TPU behavior likely gated/skipped in CI.

**Gaps / Notes**
- Full parity depends on TensorFlow TPU Estimator which is not available natively in Rust.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 81
- Purpose/role: Small utility helpers for TF monkey patching and recursive file copy with tf.io.gfile.
- Key symbols/classes/functions: `enable_monkey_patch`, `CopyFile`, `CopyRecursively`.
- External dependencies: `tensorflow`, `ThreadPoolExecutor`.
- Side effects: modifies `tensorflow.python.training.monitored_session` module attribute; copies files (possibly remote).

**Required Behavior (Detailed)**
- `enable_monkey_patch()`:
  - Imports `tensorflow.python.training.monitored_session` and sets `_PREEMPTION_ERRORS` to `(tf.errors.AbortedError,)`.
- `CopyFile(src, dst, overwrite=True, skip_nonexist=True, max_retries=5)`:
  - Uses `tf.io.gfile.copy` and retries on NotFoundError (skip if `skip_nonexist`).
- `CopyRecursively(src, dst, max_workers=1, skip_nonexist=True, max_retries=5)`:
  - Recursively copies a directory tree via `tf.io.gfile`.
  - If `src` missing and `skip_nonexist`, returns; else raises ValueError.
  - If `dst` exists, removes it then recreates.
  - If `max_workers > 1`, uses ThreadPoolExecutor to copy files in parallel.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-tf/src/utils.rs` (TF-dependent utilities) plus `monolith-core/src/path_utils.rs` for re-exports.
- Rust public API surface: equivalents for monkey patch (if bridging TF) and recursive copy.

**Implementation Steps (Detailed)**
1. Provide TF monkey patch equivalent when using Python/TF runtime; otherwise document unsupported.
2. Implement recursive copy with retries and optional parallelism.
3. Ensure semantics match tf.io.gfile for remote paths (HDFS/GCS) where needed.

**Tests (Detailed)**
- Python tests: `monolith/utils_test.py`
- Rust tests: port tests for `find_main`, `get_libops_path`, and CopyRecursively.

**Gaps / Notes**
- Full parity requires tf.io.gfile support; Rust may need a virtual filesystem abstraction.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
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

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 65
- Purpose/role: Tests path utilities, monkey patch, and recursive copy.
- Key symbols/classes/functions: `UtilsTest` test cases.
- External dependencies: `tensorflow`, `monitored_session`.
- Side effects: writes temp files in `/tmp`.

**Required Behavior (Detailed)**
- `testFindMain`: `utils.find_main()` base dir last path component is `__main__` (Bazel layout assumption).
- `testGetLibopsPath`: `utils.get_libops_path("monolith/utils_test.py")` exists.
- `testLoadMonitoredSession`: `_PREEMPTION_ERRORS` equals `(errors.AbortedError,)` after monkey patch.
- `testMultiThreadedCopy`: creates nested dirs/files and verifies copied content with `CopyRecursively(max_workers=2)`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-tf/tests/utils.rs`.
- Rust public API surface: path utils + recursive copy.

**Implementation Steps (Detailed)**
1. Port tests with temp dirs and file content checks.
2. If TF monkey patch is unsupported, document and adjust tests accordingly.

**Tests (Detailed)**
- Python tests: this file
- Rust tests: parity tests for path and copy semantics.

**Gaps / Notes**
- `find_main` behavior is tied to Bazel execution layout; Rust tests may need harness adjustments.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed
