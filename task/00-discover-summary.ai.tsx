import { Agent, Instructions, Program, Prompt, System, Asset } from '@unpack/ai';

export default (
  <Program
    id="core-hyperparams-base-layer"
    target={{ language: 'rust' }}
    description="Port monolith/core hyperparams + base layer/task params into monolith-core with parity tests."
  >
    <Asset id="log" kind="doc" path="generated/monolith/task/core-hyperparams-base-layer.md" />
    <Agent id="apply-core-hyperparams-base-layer">
      <Prompt>
        <System>
          You are porting Python monolith core code to Rust. Edit the Rust
          codebase directly; do not write plans, mapping tables, or JSON.
        </System>
        <Instructions>
Implement parity for these Python files:
- monolith/core/hyperparams.py
- monolith/core/hyperparams_test.py
- monolith/core/base_layer.py
- monolith/core/base_layer_test.py
- monolith/core/base_task.py
- monolith/core/base_model_params.py
- monolith/core/base_tpu_test.py

Rust targets:
- monolith-rs/crates/monolith-core
- monolith-rs/crates/monolith-layers (if needed)

Requirements:
- Mirror attribute access, immutability, string formatting, and error semantics.
- Preserve public API shape where it affects callers or tests.
- Add or update Rust tests that mirror the Python tests listed above.

Do not emit mapping docs or JSON. Focus on code edits and tests.
        </Instructions>
      </Prompt>
    </Agent>
  </Program>
);
