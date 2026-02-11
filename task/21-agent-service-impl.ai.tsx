import { Agent, Instructions, Program, Prompt, System, Asset } from '@unpack/ai';

export default (
  <Program
    id="native-training-service-discovery"
    target={{ language: 'rust' }}
    description="Port native_training service discovery and environment helpers into monolith-training."
  >
    <Asset id="log" kind="doc" path="generated/monolith/task/native-training-service-discovery.md" />
    <Agent id="apply-native-training-service-discovery">
      <Prompt>
        <System>
          You are porting native_training service discovery and env utilities to
          Rust. Edit the Rust codebase directly; do not write plans, mapping
          tables, or JSON.
        </System>
        <Instructions>
Implement parity for these Python files:
- monolith/native_training/consul.py
- monolith/native_training/consul_test.py
- monolith/native_training/service_discovery.py
- monolith/native_training/service_discovery_test.py
- monolith/native_training/env_utils.py
- monolith/native_training/env_utils_test.py
- monolith/native_training/device_utils.py
- monolith/native_training/device_utils_test.py
- monolith/native_training/hvd_lib.py
- monolith/native_training/graph_meta.py
- monolith/native_training/graph_utils.py

Rust targets:
- monolith-rs/crates/monolith-training
- monolith-rs/crates/monolith-core (shared utils if needed)

Requirements:
- Preserve service discovery semantics and error messages.
- Match env var handling and default resolution logic.
- Add or update Rust tests that mirror the Python tests listed above.

Do not emit mapping docs or JSON. Focus on code edits and tests.
        </Instructions>
      </Prompt>
    </Agent>
  </Program>
);
