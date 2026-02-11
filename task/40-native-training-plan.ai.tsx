import { Agent, Instructions, Program, Prompt, System, Asset } from '@unpack/ai';

export default (
  <Program
    id="native-training-data-pipeline"
    target={{ language: 'rust' }}
    description="Port native_training data pipeline into monolith-data with parity tests."
  >
    <Asset id="log" kind="doc" path="generated/monolith/task/native-training-data-pipeline.md" />
    <Agent id="apply-native-training-data-pipeline">
      <Prompt>
        <System>
          You are porting native_training data pipeline code to Rust. Edit the
          Rust codebase directly; do not write plans, mapping tables, or JSON.
        </System>
        <Instructions>
Implement parity for these Python files and directories:
- monolith/native_training/data/**
- monolith/native_training/input.py
- monolith/native_training/file_ops.py
- monolith/native_training/file_ops_test.py
- monolith/native_training/prefetch_queue.py
- monolith/native_training/prefetch_queue_test.py
- monolith/native_training/gflags_utils.py
- monolith/native_training/gflags_utils_test.py
- monolith/native_training/fountain/**

Rust targets:
- monolith-rs/crates/monolith-data
- monolith-rs/crates/monolith-training (if needed)

Requirements:
- Preserve dataset wiring, parsing rules, compression settings, and defaults.
- Keep TF runtime optional; implement Candle-first behavior where possible.
- Add or update Rust tests that mirror the Python tests listed above.

Do not emit mapping docs or JSON. Focus on code edits and tests.
        </Instructions>
      </Prompt>
    </Agent>
  </Program>
);
