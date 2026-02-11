import { Agent, Instructions, Program, Prompt, System, Asset } from '@unpack/ai';

export default (
  <Program
    id="core-feature-model-registry"
    target={{ language: 'rust' }}
    description="Port monolith/core feature + model registry surface into monolith-core with parity tests."
  >
    <Asset id="log" kind="doc" path="generated/monolith/task/core-feature-model-registry.md" />
    <Agent id="apply-core-feature-model-registry">
      <Prompt>
        <System>
          You are porting Python monolith core code to Rust. Edit the Rust
          codebase directly; do not write plans, mapping tables, or JSON.
        </System>
        <Instructions>
Implement parity for these Python files:
- monolith/core/feature.py
- monolith/core/feature_test.py
- monolith/core/model_registry.py
- monolith/core/model.py
- monolith/core/model_imports.py
- monolith/core/core_test_suite.py

Rust targets:
- monolith-rs/crates/monolith-core

Requirements:
- Preserve feature definitions, registry behavior, and error messages.
- Keep serialization and string rendering parity where tests depend on it.
- Add or update Rust tests to cover the Python test cases listed above.

Do not emit mapping docs or JSON. Focus on code edits and tests.
        </Instructions>
      </Prompt>
    </Agent>
  </Program>
);
