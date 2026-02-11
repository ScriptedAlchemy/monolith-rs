import { Agent, Instructions, Program, Prompt, System, Asset } from '@unpack/ai';

export default (
  <Program
    id="native-training-export-checkpoint"
    target={{ language: 'rust' }}
    description="Port native_training export/checkpoint paths into monolith-checkpoint with parity tests."
  >
    <Asset id="log" kind="doc" path="generated/monolith/task/native-training-export-checkpoint.md" />
    <Agent id="apply-native-training-export-checkpoint">
      <Prompt>
        <System>
          You are porting native_training export/checkpoint logic to Rust. Edit
          the Rust codebase directly; do not write plans, mapping tables, or JSON.
        </System>
        <Instructions>
Implement parity for these Python files and directories:
- monolith/native_training/save_utils.py
- monolith/native_training/save_utils_test.py
- monolith/native_training/restore_test.py
- monolith/native_training/dense_reload_utils.py
- monolith/native_training/dense_reload_utils_test.py
- monolith/native_training/monolith_export.py
- monolith/native_training/model_export/**
- monolith/native_training/model_dump/**
- monolith/native_training/monolith_checkpoint_state.proto

Rust targets:
- monolith-rs/crates/monolith-checkpoint
- monolith-rs/crates/monolith-training
- monolith-rs/crates/monolith-proto (for checkpoint protos)

Requirements:
- Preserve checkpoint format, file layout, and error behavior.
- Keep SavedModel/TFRecord compatibility where used.
- Add or update Rust tests that mirror the Python tests listed above.

Do not emit mapping docs or JSON. Focus on code edits and tests.
        </Instructions>
      </Prompt>
    </Agent>
  </Program>
);
