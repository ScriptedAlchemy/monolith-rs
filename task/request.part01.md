<!--
Source: task/request.md
Lines: 1-320 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
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
| `monolith/agent_service/agent_service.py` | `monolith-rs/crates/monolith-serving/src/grpc_agent.rs` (gRPC), `monolith-rs/crates/monolith-serving/src/server.rs` (runtime) | IN PROGRESS | Split responsibilities between coordination gRPC and serving runtime. |
| `monolith/agent_service/agent_controller.py` | `monolith-rs/crates/monolith-serving/src/*` | IN PROGRESS | Needs model layout/config orchestration parity. |
| `monolith/agent_service/agent_v1.py` | `monolith-rs/crates/monolith-serving/src/*` | IN PROGRESS | Defines agent logic; map to server + manager abstractions. |
| `monolith/agent_service/agent_v3.py` | `monolith-rs/crates/monolith-serving/src/*` | IN PROGRESS | Advanced routing / layout logic; likely new Rust module. |
| `monolith/agent_service/backends.py` | `monolith-rs/crates/monolith-serving/src/*` | IN PROGRESS | Backend abstraction for layout + storage. |
| `monolith/agent_service/replica_manager.py` | `monolith-rs/crates/monolith-serving/src/*` | IN PROGRESS | Replica discovery + updates. |
| `monolith/agent_service/model_manager.py` | `monolith-rs/crates/monolith-serving/src/*` | IN PROGRESS | Model lifecycle + watcher loops. |
| `monolith/agent_service/tfs_client.py` | `monolith-rs/crates/monolith-serving/src/tfserving.rs` | IN PROGRESS | Map client utilities to Rust TF Serving client. |
| `monolith/agent_service/tfs_wrapper.py` | `monolith-rs/crates/monolith-serving/src/tfserving.rs` | IN PROGRESS | Wrapper logic around Predict/ModelStatus. |
| `monolith/agent_service/tfs_monitor.py` | `monolith-rs/crates/monolith-serving/src/*` | IN PROGRESS | Monitoring + metrics; may need new module. |
| `monolith/agent_service/utils.py` | `monolith-rs/crates/monolith-serving/src/*` | IN PROGRESS | Config parsing, model specs, helper utilities. |
| `monolith/agent_service/constants.py` | `monolith-rs/crates/monolith-serving/src/*` | IN PROGRESS | Constants + enums. |
| `monolith/agent_service/data_def.py` | `monolith-rs/crates/monolith-serving/src/*` | IN PROGRESS | Data definition structures. |
| Tests in `monolith/agent_service/*_test.py` | `monolith-rs/crates/monolith-serving/tests/*` | IN PROGRESS | Port test cases and fixtures. |

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
