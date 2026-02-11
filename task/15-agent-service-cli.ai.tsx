import { Agent, Instructions, Program, Prompt, System, Asset } from '@unpack/ai';

export default (
  <Program
    id="agent-service-cli"
    target={{ language: 'rust' }}
    description="Port agent_service CLI/controller/service behavior into monolith-cli and monolith-serving."
  >
    <Asset id="log" kind="doc" path="generated/monolith/task/agent-service-cli.md" />
    <Agent id="apply-agent-service-cli">
      <Prompt>
        <System>
          You are porting Python agent-service CLI/controller code to Rust. Edit
          the codebase directly; do not write plans, mapping tables, or JSON.
        </System>
        <Instructions>
Implement parity for these Python files:
- monolith/agent_service/agent.py
- monolith/agent_service/agent_base.py
- monolith/agent_service/agent_service.py
- monolith/agent_service/agent_service_test.py
- monolith/agent_service/agent_service.proto
- monolith/agent_service/agent_controller.py
- monolith/agent_service/agent_controller_test.py
- monolith/agent_service/agent_v1.py
- monolith/agent_service/agent_v3.py
- monolith/agent_service/agent_v3_test.py
- monolith/agent_service/client.py
- monolith/agent_service/agent_client.py
- monolith/agent_service/svr_client.py
- monolith/agent_service/run.py
- monolith/agent_service/utils.py
- monolith/agent_service/utils_test.py

Rust targets:
- monolith-rs/crates/monolith-cli
- monolith-rs/crates/monolith-serving
- monolith-rs/crates/monolith-proto (for gRPC/proto definitions)

Requirements:
- Match CLI flags, env var handling, and config precedence.
- Preserve gRPC/service behavior and controller lifecycles.
- Add or update Rust tests that mirror the Python tests listed above.

Do not emit mapping docs or JSON. Focus on code edits and tests.
        </Instructions>
      </Prompt>
    </Agent>
  </Program>
);
