import { Agent, Instructions, Program, Prompt, System, Asset } from '@unpack/ai';

export default (
  <Program
    id="native-training-entrypoints"
    target={{ language: 'rust' }}
    description="Port native_training entrypoints and training loops into monolith-training with parity tests."
  >
    <Asset id="log" kind="doc" path="generated/monolith/task/native-training-entrypoints.md" />
    <Agent id="apply-native-training-entrypoints">
      <Prompt>
        <System>
          You are porting native_training entrypoints to Rust. Edit the Rust
          codebase directly; do not write plans, mapping tables, or JSON.
        </System>
        <Instructions>
Implement parity for these Python files:
- monolith/native_training/entry.py
- monolith/native_training/entry_test.py
- monolith/native_training/estimator.py
- monolith/native_training/estimator_test.py
- monolith/native_training/estimator_mode_test.py
- monolith/native_training/estimator_dist_test.py
- monolith/native_training/cpu_training.py
- monolith/native_training/cpu_training_test.py
- monolith/native_training/cpu_sync_training_test.py
- monolith/native_training/cpu_training_distributed_test_binary.py
- monolith/native_training/native_model.py
- monolith/native_training/native_task.py
- monolith/native_training/native_task_context.py
- monolith/native_training/model.py
- monolith/native_training/model_comp_test.py
- monolith/native_training/runner_utils.py
- monolith/native_training/runner_utils_test.py
- monolith/native_training/demo.py

Rust targets:
- monolith-rs/crates/monolith-training
- monolith-rs/crates/monolith-data

Requirements:
- Preserve training loop semantics, flags, and error behavior.
- Keep TensorFlow runtime optional; do not vendor TF binaries.
- Add or update Rust tests that mirror the Python tests listed above.

Do not emit mapping docs or JSON. Focus on code edits and tests.
        </Instructions>
      </Prompt>
    </Agent>
  </Program>
);
