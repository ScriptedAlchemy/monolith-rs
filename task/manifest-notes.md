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
