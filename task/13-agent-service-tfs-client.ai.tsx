import { Agent, Instructions, Program, Prompt, System, Asset } from '@unpack/ai';

export default (
  <Program
    id="agent-service-tfs-client"
    target={{ language: 'rust' }}
    description="Port TFServing client/monitor/wrapper + mocks into monolith-serving."
  >
    <Asset id="log" kind="doc" path="generated/monolith/task/agent-service-tfs-client.md" />
    <Agent id="apply-agent-service-tfs-client">
      <Prompt>
        <System>
          You are porting Python TFServing client logic to Rust. Edit the Rust
          codebase directly; do not write plans, mapping tables, or JSON.
        </System>
        <Instructions>
Implement parity for these Python files:
- monolith/agent_service/tfs_client.py
- monolith/agent_service/tfs_client_test.py
- monolith/agent_service/tfs_monitor.py
- monolith/agent_service/tfs_monitor_test.py
- monolith/agent_service/tfs_wrapper.py
- monolith/agent_service/mocked_tfserving.py
- monolith/agent_service/mocked_tfserving_test.py

Rust target:
- monolith-rs/crates/monolith-serving

Requirements:
- Preserve TFServing request/response shaping and error handling.
- Match timeout, retry, and monitoring semantics from Python.
- Provide Rust mocks/fakes that mirror the Python mock behavior.
- Add or update Rust tests that mirror the Python tests listed above.

Do not emit mapping docs or JSON. Focus on code edits and tests.
        </Instructions>
      </Prompt>
    </Agent>
  </Program>
);
