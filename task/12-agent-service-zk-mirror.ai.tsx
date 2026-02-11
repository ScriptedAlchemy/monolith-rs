import { Agent, Instructions, Program, Prompt, System, Asset } from '@unpack/ai';

export default (
  <Program
    id="agent-service-zk-mirror"
    target={{ language: 'rust' }}
    description="Port ZK mirror + fake ZK client behavior for agent_service into monolith-serving."
  >
    <Asset id="log" kind="doc" path="generated/monolith/task/agent-service-zk-mirror.md" />
    <Agent id="apply-agent-service-zk-mirror">
      <Prompt>
        <System>
          You are porting Python ZK mirror logic to Rust. Edit the Rust
          codebase directly; do not write plans, mapping tables, or JSON.
        </System>
        <Instructions>
Implement parity for these Python files:
- monolith/agent_service/zk_mirror.py
- monolith/agent_service/zk_mirror_test.py
- monolith/agent_service/mocked_zkclient.py
- monolith/agent_service/mocked_zkclient_test.py

Rust target:
- monolith-rs/crates/monolith-serving

Requirements:
- Preserve ZK watch semantics, state transitions, and error behaviors.
- Implement a fake/mock ZK client in Rust to support parity tests.
- Add or update Rust tests that mirror the Python tests listed above.

Do not emit mapping docs or JSON. Focus on code edits and tests.
        </Instructions>
      </Prompt>
    </Agent>
  </Program>
);
