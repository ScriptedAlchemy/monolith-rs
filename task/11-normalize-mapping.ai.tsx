import { Agent, Instructions, Program, Prompt, System, Asset } from '@unpack/ai';

export default (
  <Program
    id="agent-service-backends"
    target={{ language: 'rust' }}
    description="Port monolith/agent_service backend plumbing into monolith-serving with parity tests."
  >
    <Asset id="log" kind="doc" path="generated/monolith/task/agent-service-backends.md" />
    <Agent id="apply-agent-service-backends">
      <Prompt>
        <System>
          You are porting Python agent-service backends to Rust. Edit the Rust
          codebase directly; do not write plans, mapping tables, or JSON.
        </System>
        <Instructions>
Implement parity for these Python files:
- monolith/agent_service/backends.py
- monolith/agent_service/backends_test.py
- monolith/agent_service/resource_utils.py
- monolith/agent_service/resource_utils_test.py

Rust target:
- monolith-rs/crates/monolith-serving

Requirements:
- Mirror backend selection logic, defaults, and error handling.
- Preserve serialization formats used by backends.
- Add or update Rust tests that mirror the Python tests listed above.

Do not emit mapping docs or JSON. Focus on code edits and tests.
        </Instructions>
      </Prompt>
    </Agent>
  </Program>
);
