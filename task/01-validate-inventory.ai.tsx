import { Agent, Instructions, Program, Prompt, System, Asset } from '@unpack/ai';

export default (
  <Program
    id="core-embeddings-layers"
    target={{ language: 'rust' }}
    description="Port core embedding/layer utilities into monolith-core/monolith-layers with parity tests."
  >
    <Asset id="log" kind="doc" path="generated/monolith/task/core-embeddings-layers.md" />
    <Agent id="apply-core-embeddings-layers">
      <Prompt>
        <System>
          You are porting Python monolith core code to Rust. Edit the Rust
          codebase directly; do not write plans, mapping tables, or JSON.
        </System>
        <Instructions>
Implement parity for these Python files:
- monolith/core/base_embedding_task.py
- monolith/core/base_embedding_host_call.py
- monolith/core/base_embedding_host_call_test.py
- monolith/core/dense.py
- monolith/core/dense_test.py
- monolith/core/variance_scaling.py
- monolith/core/mixed_emb_op_comb_nws.py

Rust targets:
- monolith-rs/crates/monolith-core
- monolith-rs/crates/monolith-layers

Requirements:
- Preserve embedding/layer math and initialization semantics.
- Match error handling and shape validations from Python.
- Add or update Rust tests that mirror the Python tests listed above.

Do not emit mapping docs or JSON. Focus on code edits and tests.
        </Instructions>
      </Prompt>
    </Agent>
  </Program>
);
