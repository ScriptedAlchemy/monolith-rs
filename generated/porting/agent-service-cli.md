Agent-Service CLI/Controller/Service Port (Rust)

This workspace ports the Python agent-service discovery/controller pieces under
`monolith/agent_service/` to Rust, primarily for parity tests and for a minimal
CLI surface.

Rust entry points

- Library: `monolith-rs/crates/monolith-serving`
  - Discovery gRPC service (Python parity: `agent_service.py`):
    `monolith_serving::agent_service_discovery`
  - Unified container agent (Python parity: `agent_v3.py`):
    `monolith_serving::agent_v3::AgentV3`
  - Controller helpers (Python parity: `agent_controller.py`):
    `monolith_serving::agent_controller`
  - Shared utilities (Python parity: `utils.py` subset used by tests):
    `monolith_serving::utils`
- CLI: `monolith-rs/crates/monolith-cli`
  - Command group: `monolith agent-service ...`

CLI usage (parity-style)

- Run agent v3 (fake ZooKeeper; used for local testing):
  - `monolith agent-service agent --conf /path/to/agent.conf --fake-zk`
- Controller:
  - `monolith agent-service controller --cmd decl --export-base /path/to/export --fake-zk`
  - `monolith agent-service controller --cmd pub --layout foo --model-name model:entry --fake-zk`
- Client:
  - `monolith agent-service client --target 127.0.0.1:PORT --cmd-type hb`
  - `monolith agent-service client --target 127.0.0.1:PORT --cmd-type gr --task 0 --model-name model`

Notes

- The CLI currently wires `--fake-zk` only. Production ZooKeeper integration is
  represented by the `ZkClient` trait; tests use `FakeKazooClient`.
- Parity tests live in `monolith-rs/crates/monolith-serving/tests/*_parity.rs`
  and mirror the Python tests listed in `task/15-agent-service-cli.ai.tsx`.
