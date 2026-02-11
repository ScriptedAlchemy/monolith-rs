import { Agent, Instructions, Program, Prompt, System, Asset } from '@unpack/ai';

export default (
  <Program
    id="core-utils-optimizers"
    target={{ language: 'rust' }}
    description="Port core utilities, optimizers, and host-call helpers into monolith-core with parity tests."
  >
    <Asset id="log" kind="doc" path="generated/monolith/task/core-utils-optimizers.md" />
    <Agent id="apply-core-utils-optimizers">
      <Prompt>
        <System>
          You are porting Python monolith core code to Rust. Edit the Rust
          codebase directly; do not write plans, mapping tables, or JSON.
        </System>
        <Instructions>
Implement parity for these Python files:
- monolith/core/util.py
- monolith/core/util_test.py
- monolith/core/py_utils.py
- monolith/core/testing_utils.py
- monolith/core/optimizers.py
- monolith/core/host_call.py
- monolith/core/base_host_call.py
- monolith/core/auto_checkpoint_feed_hook.py
- monolith/core/tpu_variable.py

Rust targets:
- monolith-rs/crates/monolith-core

Requirements:
- Preserve helper function behavior, defaults, and error messages.
- Match optimizer math and parameter validation.
- Add or update Rust tests that mirror the Python tests listed above.

Do not emit mapping docs or JSON. Focus on code edits and tests.
        </Instructions>
      </Prompt>
    </Agent>
  </Program>
);
