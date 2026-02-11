# Agent Service Rust Parity Plan (monolith/agent_service only)

This document is the implementation-oriented plan for porting the Python module `monolith/agent_service/**` to Rust (`monolith-rs/**`) with behavioral and protocol parity.

Inventory + generator context lives at `{{assets.agent_service_plan_json.path}}` (do not edit by hand). Current generation notes: normalized mapping (`generated/parity/11-normalize-mapping/normalized_mapping.json`) is missing, so this plan is based on direct source inspection and the inventory list (38 Python files under `monolith/agent_service/**`).

## Purpose And Scope

- **In scope:** Only the runtime and tooling under `monolith/agent_service/**`: Agent gRPC service (AgentService proto), ZK-backed discovery/control plane, TF Serving integration, model config/status monitoring, and the process/orchestration logic (agent v1 + v3).
- **Out of scope:** Training/runtime outside agent_service (e.g. `monolith/native_training/**`) except where agent_service depends on interfaces/protos/paths; those dependencies are treated as external contracts.

## Target Rust Crate / Module Layout

Use existing crates where they already match the boundary, and add one orchestration crate for the agent runtime.

- **Protos + gRPC contract types**
  - `monolith-rs/crates/monolith-proto/`
  - Key modules:
    - `monolith-rs/crates/monolith-proto/src/lib.rs` (exports `monolith::serving::agent_service` and TF Serving protos)
    - Source proto: `monolith-rs/proto/agent_service.proto`
- **Serving-side functionality (prediction + TF Serving compatibility)**
  - `monolith-rs/crates/monolith-serving/`
  - Key modules:
    - `monolith-rs/crates/monolith-serving/src/agent_service.rs` (embedding/inference agent; separate from upstream AgentService discovery)
    - `monolith-rs/crates/monolith-serving/src/tfserving.rs` (TF Serving client utilities; pbtxt parsing)
    - `monolith-rs/crates/monolith-serving/src/tfserving_server.rs` (TF Serving PredictionService + ModelService subset)
    - `monolith-rs/crates/monolith-serving/src/grpc_agent.rs` (tonic server for upstream AgentService proto; extend for parity)
- **New: agent runtime / orchestration (process supervision + ZK control plane)**
  - Create `monolith-rs/crates/monolith-agent/`
  - Proposed module skeleton (stable paths to implement against):
    - `monolith-rs/crates/monolith-agent/src/lib.rs`
    - `monolith-rs/crates/monolith-agent/src/config.rs` (AgentConfig, parsing `agent.conf`, env overrides)
    - `monolith-rs/crates/monolith-agent/src/data_def.rs` (ModelMeta/PublishMeta/ReplicaMeta/ResourceSpec equivalents)
    - `monolith-rs/crates/monolith-agent/src/zk/mod.rs` (ZK client abstraction + path helpers)
    - `monolith-rs/crates/monolith-agent/src/zk/fake.rs` (in-memory Fake ZK for deterministic tests; watcher + election support)
    - `monolith-rs/crates/monolith-agent/src/zk/real.rs` (real ZK implementation behind a feature flag)
    - `monolith-rs/crates/monolith-agent/src/zk_mirror.rs` (mirror + event queue + leader election logic)
    - `monolith-rs/crates/monolith-agent/src/backends.rs` (ZKBackend: layouts, bindings, service info, sync targets)
    - `monolith-rs/crates/monolith-agent/src/replica_manager.rs` (ReplicaWatcher + ReplicaManager: discovery + status)
    - `monolith-rs/crates/monolith-agent/src/tfs_wrapper.rs` (optional process wrapper; always testable via fake)
    - `monolith-rs/crates/monolith-agent/src/tfs_monitor.rs` (reload config + status polling + model config generation)
    - `monolith-rs/crates/monolith-agent/src/model_manager.rs` (filesystem copy/retention logic)
    - `monolith-rs/crates/monolith-agent/src/agent_v1.rs` (process graph: ps/entry/dense/proxy)
    - `monolith-rs/crates/monolith-agent/src/agent_v3.rs` (layout-driven unified serving + bindings)
    - `monolith-rs/crates/monolith-agent/src/bin/monolith-agent.rs` (entrypoint parity with `monolith/agent_service/agent.py`)
- **CLI tooling parity**
  - Extend `monolith-rs/crates/monolith-cli/` with subcommands rather than adding many ad-hoc binaries.
  - Proposed new commands:
    - `monolith-rs/crates/monolith-cli/src/commands/agent.rs` (run agent)
    - `monolith-rs/crates/monolith-cli/src/commands/agent_ctl.rs` (declare/publish/unpublish layouts)
    - `monolith-rs/crates/monolith-cli/src/commands/agent_client.rs` (GetReplicas/HeartBeat + ZK inspection helpers)
    - `monolith-rs/crates/monolith-cli/src/commands/tfs_client.rs` (TF Serving Predict + status utilities; supports pbtxt/json inputs)

## Execution Order (Workstreams)

1) **gRPC contract and boundary correctness (dependency: monolith-proto builds)**
   - Ensure all Rust uses of AgentService are via `monolith_proto::monolith::serving::agent_service::*`.
   - Extend `monolith-rs/crates/monolith-serving/src/grpc_agent.rs` to cover Python behaviors:
     - `GetReplicas`: task routing + model_name handling consistent with v1/v2/v3 modes.
     - `HeartBeat`: key formatting (dc-aware path keying vs enum name keying).
     - `GetResource`: address/shard/replica + resource fields.
   - Deliverable: a parity-focused server implementation (still can be backed by fake ZK in tests).

2) **Core data definitions + config parsing (dependency: none; unblocks almost everything)**
   - Implement Rust equivalents of `data_def.py` and the minimal subset of `utils.py` needed for config and ZK paths.
   - Parse `monolith/agent_service/agent.conf` fields into a stable `AgentConfig`:
     - env overrides for `SHARD_ID`, `REPLICA_ID`, and `MONOLITH_HOST_SHARD_N` semantics
     - deploy_type + agent_version routing (v1/v3)
   - Deliverable: `monolith-agent` crate compiles with unit tests for serde roundtrips and config parsing.

3) **ZK abstraction + Fake ZK (dependency: step 2)**
   - Define a small trait (sync + async as needed) for the ZK operations used in agent_service:
     - create/set/get/delete, get_children, exists, ensure_path
     - data watches + children watches
     - leader election primitive (single-leader deterministically in Fake ZK)
     - ephemeral node lifecycle tied to a client session handle
   - Implement `zk::fake` first; gate `zk::real` behind a feature flag so the crate remains buildable without ZK.
   - Deliverable: deterministic tests ported from `mocked_zkclient*.py` behavior (watch triggers, NotEmpty, NoNode).

4) **Replica discovery (dependency: step 3)**
   - Port `replica_manager.py` behavior:
     - `ReplicaWatcher.watch_data()` with dc-aware and non-dc-aware path layouts
     - `get_replicas(server_type, task, idc, cluster)` and `get_all_replicas(server_type, idc, cluster)`
     - polling-based reconciliation loop (keep as optional; tests use watch-triggered updates)
   - Wire `grpc_agent` service methods to the watcher for v1 behavior.
   - Deliverable: Rust tests mirroring `monolith/agent_service/agent_service_test.py`.

5) **ZKMirror + control plane (dependency: step 3; used by v2/v3 patterns)**
   - Port the path model and state machine in `zk_mirror.py`:
     - portal/publish/service/resource trees
     - publish expectations and selection rules (shortest_sub_model_pm logic, replica-specific override)
     - event queue semantics (only the subset required for agent_v3 + tooling)
   - Deliverable: unit tests for publish selection and path building.

6) **Backend APIs for layout + bindings + service info (dependency: step 3; consumes step 5 if sharing path helpers)**
   - Port `backends.py`:
     - declare/list saved models + deploy config encoding
     - layout callback registration and deterministic ordering of (SavedModel, DeployConfig)
     - available saved model binding (ephemeral nodes) and `get_service_map()`
     - sync target resolution for parameter sync endpoints (if used by your deployment)
   - Deliverable: Rust tests mirroring `monolith/agent_service/backends_test.py` and `agent_controller_test.py` (using Fake ZK).

7) **TF Serving integration (dependency: monolith-serving tfserving client/server; step 2 for config wiring)**
   - Port `tfs_monitor.py` logic in Rust, using:
     - existing `monolith_serving::TfServingClient` for RPCs
     - existing pbtxt parsing helper for `ModelServerConfig`
   - Implement Rust `FakeTFServing` for tests:
     - ModelService: GetModelStatus + HandleReloadConfigRequest
     - PredictionService: optional (needed only if you want parity for `tfs_client.py` flows)
   - Deliverable: Rust tests similar to `monolith/agent_service/tfs_monitor_test.py`.

8) **Process orchestration (dependency: step 2; optional dependency on step 7)**
   - Port the process graph and lifecycle:
     - v1: process nodes for PS/ENTRY/DENSE/PROXY and wait-for-started/failover semantics (`agent_v1.py` + `agent_base.py`)
     - v3: unified mode layout-driven config updates, bindings, and service info heartbeat (`agent_v3.py`)
   - Keep process launching behind feature flags; always keep a pure in-process mode for tests (fake TF Serving).
   - Deliverable: integration test that exercises v3 layout updates end-to-end with Fake ZK + Fake TF Serving.

9) **CLI parity + operational tooling (dependency: core modules stable)**
   - Implement Rust CLI commands to cover the commonly-used workflows in:
     - `agent.py`, `agent_client.py`, `client.py`, `agent_controller.py`, `tfs_client.py`
   - Focus on deterministic output ordering and explicit exit codes (for automation).
   - Deliverable: CLI smoke tests (golden output snapshots) with Fake ZK.

## Test / Harness Strategy

- **Fake ZK (required)**
  - Implement a deterministic in-memory ZK (`monolith-rs/crates/monolith-agent/src/zk/fake.rs`) supporting:
    - children/data watches that fire synchronously on mutations
    - ephemeral nodes tied to a client handle
    - leader election that is deterministic under single-process tests
  - Port Python tests:
    - `monolith/agent_service/backends_test.py` -> `monolith-rs/crates/monolith-agent/tests/backends.rs`
    - `monolith/agent_service/agent_service_test.py` -> `monolith-rs/crates/monolith-agent/tests/agent_service.rs`

- **Fake TFServing (required)**
  - Implement a tonic server providing the subset used by agent_service:
    - ModelService/GetModelStatus
    - ModelService/HandleReloadConfigRequest
    - (optional) PredictionService/Predict for request-shape testing
  - Port Python tests:
    - `monolith/agent_service/tfs_monitor_test.py` -> `monolith-rs/crates/monolith-agent/tests/tfs_monitor.rs`

- **Cross-language parity harness (required)**
  - Add a small harness that runs the same scenario against Python and Rust and compares:
    - serialized proto responses (bytes) for AgentService RPCs
    - pbtxt generated `ModelServerConfig` text normalized by parsing + re-encoding
  - Proposed location:
    - `monolith-rs/scripts/parity_agent_service/` (inputs + runner)
    - `monolith-rs/scripts/parity_agent_service/vectors/` (golden requests/responses)
  - Determinism rule: vectors are checked into the repo; the harness must be order-stable and not embed timestamps.

- **gRPC contract checks (required)**
  - Rust-only: ensure `monolith_proto::monolith::serving::agent_service` server can round-trip using tonic client (already present; extend coverage).
    - Existing: `monolith-rs/crates/monolith-serving/tests/agent_service_tonic.rs`
  - Cross-language: add a contract test that spins up the Rust server and calls it using a Python-generated stub (or `grpcurl`) as part of CI where Python is available.

## File Accountability Table (Python -> Rust)

Status legend:
- `Done`: parity implemented + Rust tests.
- `Partial`: some Rust exists, but gaps remain.
- `Planned`: not yet ported.
- `N/A`: intentionally not ported; justification required.

| Python file (inventory order) | Status | Rust target(s) | Notes / N/A justification |
| --- | --- | --- | --- |
| `monolith/agent_service/__init__.py` | N/A | N/A | Package marker only (0 LOC). |
| `monolith/agent_service/agent.py` | Planned | `monolith-rs/crates/monolith-agent/src/bin/monolith-agent.rs` | Entry-point parity: selects agent version, starts model manager, runs agent. |
| `monolith/agent_service/agent_base.py` | Planned | `monolith-rs/crates/monolith-agent/src/agent_v1.rs` `monolith-rs/crates/monolith-agent/src/agent_v3.rs` | Shared command building + logging context + base trait. |
| `monolith/agent_service/agent_client.py` | Planned | `monolith-rs/crates/monolith-cli/src/commands/agent_client.rs` | CLI for AgentService RPCs + ZK inspection. |
| `monolith/agent_service/agent_controller.py` | Planned | `monolith-rs/crates/monolith-cli/src/commands/agent_ctl.rs` | Layout/controller tooling: declare saved models, publish/unpublish. |
| `monolith/agent_service/agent_controller_test.py` | Planned | `monolith-rs/crates/monolith-agent/tests/agent_controller.rs` | Port using Fake ZK; verify layout contents ordering. |
| `monolith/agent_service/agent_service.py` | Partial | `monolith-rs/crates/monolith-serving/src/grpc_agent.rs` | Rust has a minimal server; must add v1/v2/v3 behaviors + dc-aware keying + GetResource fields. |
| `monolith/agent_service/agent_service_test.py` | Planned | `monolith-rs/crates/monolith-agent/tests/agent_service.rs` | Port: watcher-backed discovery + HeartBeat response shape. |
| `monolith/agent_service/agent_v1.py` | Planned | `monolith-rs/crates/monolith-agent/src/agent_v1.rs` | Process graph + failover + startup ordering (PS/DENSE before ENTRY). |
| `monolith/agent_service/agent_v3.py` | Planned | `monolith-rs/crates/monolith-agent/src/agent_v3.rs` | Unified deploy: layout callback -> config update, bindings, service info, AgentService address map. |
| `monolith/agent_service/agent_v3_test.py` | Planned | `monolith-rs/crates/monolith-agent/tests/agent_v3.rs` | End-to-end with Fake ZK + Fake TFServing; validate layout filter behavior. |
| `monolith/agent_service/backends.py` | Planned | `monolith-rs/crates/monolith-agent/src/backends.rs` | ZKBackend: saved model registry, layout management, service map, sync targets. |
| `monolith/agent_service/backends_test.py` | Planned | `monolith-rs/crates/monolith-agent/tests/backends.rs` | Port with Fake ZK; assert callback ordering + bindings. |
| `monolith/agent_service/client.py` | Planned | `monolith-rs/crates/monolith-cli/src/commands/agent_client.rs` | Lightweight wrapper around ZKMirror (load/unload/status). |
| `monolith/agent_service/constants.py` | Planned | `monolith-rs/crates/monolith-agent/src/constants.rs` | Small env var constants; keep centralized for config behavior. |
| `monolith/agent_service/data_def.py` | Planned | `monolith-rs/crates/monolith-agent/src/data_def.rs` | Dataclass-json equivalents with stable serialization. |
| `monolith/agent_service/data_def_test.py` | Planned | `monolith-rs/crates/monolith-agent/tests/data_def.rs` | Roundtrip serde tests; byte-compat is not required but structure parity is. |
| `monolith/agent_service/mocked_tfserving.py` | Planned | `monolith-rs/crates/monolith-agent/src/fakes/tfserving.rs` | Deterministic fake ModelService (+ optional PredictionService). |
| `monolith/agent_service/mocked_tfserving_test.py` | Planned | `monolith-rs/crates/monolith-agent/tests/fake_tfserving.rs` | Validate fake status transitions and reload behavior used by monitor tests. |
| `monolith/agent_service/mocked_zkclient.py` | Planned | `monolith-rs/crates/monolith-agent/src/zk/fake.rs` | In-memory ZK with watches + errors; mirror FakeKazooClient semantics. |
| `monolith/agent_service/mocked_zkclient_test.py` | Planned | `monolith-rs/crates/monolith-agent/tests/fake_zk.rs` | Port: create/set/get/delete, recursive delete, watch triggers. |
| `monolith/agent_service/model_manager.py` | Planned | `monolith-rs/crates/monolith-agent/src/model_manager.rs` | File copy + retention + lockfile semantics (READ_LOCK/WRITE_DONE). |
| `monolith/agent_service/model_manager_test.py` | Planned | `monolith-rs/crates/monolith-agent/tests/model_manager.rs` | Use tempdirs; deterministic version selection; no real HDFS dependency. |
| `monolith/agent_service/replica_manager.py` | Planned | `monolith-rs/crates/monolith-agent/src/replica_manager.rs` | ReplicaWatcher core + polling reconciliation; used by gRPC AgentService v1 behavior. |
| `monolith/agent_service/replica_manager_test.py` | Planned | `monolith-rs/crates/monolith-agent/tests/replica_manager.rs` | Port the minimal scenarios; use Fake ZK for watches + path layouts. |
| `monolith/agent_service/resource_utils.py` | Planned | `monolith-rs/crates/monolith-agent/src/resource_utils.rs` | Resource probes (memory/cpu) behind OS-gated impls; keep placeholders where needed. |
| `monolith/agent_service/resource_utils_test.py` | Planned | `monolith-rs/crates/monolith-agent/tests/resource_utils.rs` | Test pure functions only; avoid asserting host-specific metrics. |
| `monolith/agent_service/run.py` | Planned | `monolith-rs/crates/monolith-cli/src/commands/agent.rs` | One-shot runner wrapper; prefer CLI subcommand over a separate binary. |
| `monolith/agent_service/svr_client.py` | Planned | `monolith-rs/crates/monolith-agent/src/agent_client.rs` | Programmatic AgentService client used by tests and higher-level tooling. |
| `monolith/agent_service/tfs_client.py` | Partial | `monolith-rs/crates/monolith-serving/src/tfserving.rs` `monolith-rs/crates/monolith-cli/src/commands/tfs_client.rs` | Rust has TF Serving RPC client; needs parity for input generators and file formats. |
| `monolith/agent_service/tfs_client_test.py` | Planned | `monolith-rs/crates/monolith-cli/tests/tfs_client.rs` | Port minimal tensor construction + ExampleBatch->Instance conversion checks. |
| `monolith/agent_service/tfs_monitor.py` | Planned | `monolith-rs/crates/monolith-agent/src/tfs_monitor.rs` | Implement reload config + status aggregation; reuse `TfServingClient`. |
| `monolith/agent_service/tfs_monitor_test.py` | Planned | `monolith-rs/crates/monolith-agent/tests/tfs_monitor.rs` | Port with Fake TFServing; avoid sleeping-based flakiness (use controlled event queue). |
| `monolith/agent_service/tfs_wrapper.py` | Planned | `monolith-rs/crates/monolith-agent/src/tfs_wrapper.rs` | Process wrapper is feature-gated; always provide Fake wrapper for tests. |
| `monolith/agent_service/utils.py` | Planned | `monolith-rs/crates/monolith-agent/src/config.rs` `monolith-rs/crates/monolith-agent/src/utils.rs` | Split: config parsing + pb/pbtxt helpers + path parsing. |
| `monolith/agent_service/utils_test.py` | Planned | `monolith-rs/crates/monolith-agent/tests/utils.rs` | Port: config parsing, ZKPath parsing, tensor proto builders (subset). |
| `monolith/agent_service/zk_mirror.py` | Planned | `monolith-rs/crates/monolith-agent/src/zk_mirror.rs` | Mirror state + event emission; integrate with backends + agent_v3. |
| `monolith/agent_service/zk_mirror_test.py` | Planned | `monolith-rs/crates/monolith-agent/tests/zk_mirror.rs` | Port selection logic + event ordering; use Fake ZK for determinism. |

## Top Risks And Mitigations

- **ZK watch semantics mismatch (ordering, re-entrancy, ephemeral lifetimes).**
  - Mitigation: define a narrow `ZkClient` trait based on actual call sites; fully specify watch trigger behavior in Fake ZK tests before implementing real ZK.
- **TF Serving pbtxt/config behavior drift (Python text_format vs Rust parsing).**
  - Mitigation: always compare configs by parsing into `ModelServerConfig` and re-encoding; avoid direct string comparisons.
- **Process supervision parity is OS- and deployment-specific (signals, ports, env).**
  - Mitigation: isolate process launching behind feature flags and keep a pure in-process mode (Fake TFServing) so the core logic is testable everywhere.
- **AgentService vs "serving agent" naming collision (Python AgentService is discovery; Rust AgentServiceImpl is inference).**
  - Mitigation: keep discovery gRPC service in a clearly named module (e.g. `grpc_agent` / `agent_discovery`) and document it in code + CLI help.
- **Cross-language harness availability (Python may not be available in all environments).**
  - Mitigation: check in deterministic golden vectors; run cross-language comparisons only in CI jobs that have Python, but keep Rust-only contract tests always-on.
