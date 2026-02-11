import { Agent, Instructions, Program, Prompt, System, Asset } from '@unpack/ai';

export default (
  <Program
    id="agent-service-replica-manager"
    target={{ language: 'rust' }}
    description="Port replica/model manager logic for agent_service into monolith-serving."
  >
    <Asset id="log" kind="doc" path="generated/monolith/task/agent-service-replica-manager.md" />
    <Agent id="apply-agent-service-replica-manager">
      <Prompt>
        <System>
          You are porting Python replica/model manager logic to Rust. Edit the
          codebase directly; do not write plans, mapping tables, or JSON.
        </System>
        <Instructions>
Implement parity for these Python files:
- monolith/agent_service/replica_manager.py
- monolith/agent_service/replica_manager_test.py
- monolith/agent_service/model_manager.py
- monolith/agent_service/model_manager_test.py

Rust target:
- monolith-rs/crates/monolith-serving

Requirements:
- Preserve state machines, reconciliation loops, and error semantics.
- Match timing and retry behavior from Python.
- Add or update Rust tests that mirror the Python tests listed above.

Do not emit mapping docs or JSON. Focus on code edits and tests.
        </Instructions>
      </Prompt>
    </Agent>
  </Program>
);
