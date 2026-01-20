# Monolith Python -> Monolith-RS Parity Master Plan (Request Summary)

## 1) What success looks like
Monolith-RS reaches full (or максимально close) behavioral parity with the existing Python code under `monolith/`, with every Python file explicitly accounted for (mapped to Rust destination or justified N/A), and parity validated via mirrored tests, golden fixtures, and cross-language comparisons while keeping TensorFlow runtime optional (no vendoring) and Candle as the default inference backend.

## 2) Hard requirements
- Track *every* Python file in `monolith/` (and relevant tooling/patches) with a concrete Rust target or an explicit justified N/A; no untracked files.
- Parity definition includes: behavior/output matching, protocol compatibility (protobuf/gRPC/disk formats), CLI/config/env var compatibility, ops/side effects mapping, I/O format compatibility (TFRecord/SavedModel/checkpoint), and comparable test coverage.
- Preserve semantics even if Rust internals differ; prioritize line-for-line behavioral parity over idiomatic API shape.
- Do not vendor TensorFlow runtime libraries; TF runtime must be optional and dynamically loaded (Linux x86_64 best-effort elsewhere).
- Candle remains the default inference backend unless a true TF SavedModel runtime is available.
- List and track all parity gaps; close them or explicitly justify deviations.
- A file is not "done" until: checklist complete, parity verified against Python tests or equivalent, Rust tests added, formats validated, and perf regressions documented.
- Maintain/update the inventory index and per-file parity checklists as progress is made.

## 3) Non-goals / out of scope
- Identical code structure or identical API style between Python and Rust (idiomatic Rust allowed if external behavior matches).
- Declaring parity without passing file-level + integration + end-to-end verification checks.

## 4) Deliverables the task graph should produce (files/artifacts)
- `monolith-rs/PYTHON_PARITY_INDEX.md`: master table of all Python files with line counts, status, and Rust mapping.
- `monolith-rs/parity/**`: one checklist per Python file mirroring path; includes symbol list, behavior notes, Rust mapping, tests, gaps.
- Expanded “module mapping table” (Python -> Rust crate/module) covering all Python files (incrementally).
- Parity test suite: Rust tests mirroring Python tests where feasible + cross-language parity harness that runs Python and Rust against shared inputs/fixtures.
- Optional TF runtime integration plan + implementation artifacts (dynamic libtensorflow loading, custom op loading, signature mapping) gated behind features.

## 5) Suggested phases / modules (high-level)
- Phase A: Cross-cutting infrastructure parity (CLI/config/env precedence; protobuf/gRPC; data formats; checkpoint/export; runtime/scheduling).
- Phase B: High-priority serving: `monolith/agent_service/**` -> `monolith-rs/crates/monolith-serving` (gRPC AgentService, ZK backends, TF Serving client/monitor/wrapper, replica/model managers, CLIs).
- Phase C: Core modeling: `monolith/core/**` -> `monolith-rs/crates/monolith-core` + `monolith-layers` (+ `monolith-tf` where TF-specific).
- Phase D: Training: `monolith/native_training/**` -> `monolith-rs/crates/monolith-training`/`monolith-data`/`monolith-checkpoint` (data pipeline, distributed runtime, export).
- Phase E: Optional TF runtime: dynamic loading + custom ops for parity gaps Candle cannot cover.
- Phase F: Validation/auditing: per-file completion -> integration tests -> end-to-end workflow runs; update logs and indices.

## 6) Risks / gotchas
- Extremely large scope (hundreds of files): risk of missing files unless inventory/checklists remain canonical and continuously updated.
- Hidden behavior in side effects (imports, env var mutations, process spawning, ZK watches, TF graph/session hooks) is hard to replicate; may require explicit Rust init APIs + feature gating.
- TF/runtime constraints: no vendoring and optional dynamic loading complicate cross-platform support and test coverage.
- Many Python modules depend on custom TF ops (`gen_monolith_ops`) and TF Dataset internals; Rust likely needs either TF-runtime-backed kernels or re-implementations.
- Test parity complexity: some Python tests are incomplete or integration-heavy (Kafka, TPU, data service); need clear strategy for what becomes Rust tests vs documented Python-only tests.
- Output/protocol compatibility: protobuf/gRPC, SavedModel/TFRecord/checkpoint formats must remain compatible; regression risk is high without golden fixtures.

## Open questions (from the request)
- Which subset of behaviors/modules is the first parity milestone (beyond “agent_service first”)?
- Are GPU/TPU runtime semantics targeted for full parity or “functional parity” only?
- Should Python code be vendored/synced into `monolith-rs/python/` or treated as external reference?
