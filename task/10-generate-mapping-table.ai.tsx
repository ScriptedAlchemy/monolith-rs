import { Agent, Instructions, Program, Prompt, System, Asset } from '@unpack/ai';

export default (
  <Program
    id="agent-service-data-def"
    target={{ language: 'rust' }}
    description="Port monolith/agent_service data definitions and constants into monolith-serving with parity tests."
  >
    <Asset id="log" kind="doc" path="generated/monolith/task/agent-service-data-def.md" />
    <Agent id="apply-agent-service-data-def">
      <Prompt>
        <System>
          You are porting Python agent-service code to Rust. Edit the Rust
          codebase directly; do not write plans, mapping tables, or JSON.
        </System>
        <Instructions>
Implement parity for these Python files:
- monolith/agent_service/data_def.py
- monolith/agent_service/data_def_test.py
- monolith/agent_service/constants.py

Rust target:
- monolith-rs/crates/monolith-serving

Requirements:
- Mirror data structures, defaults, and error semantics.
- Preserve any serialization or on-wire formats.
- Add or update Rust tests that mirror the Python tests listed above.

Do not emit mapping docs or JSON. Focus on code edits and tests.
        </Instructions>
      </Prompt>
    </Agent>
  </Program>
);
