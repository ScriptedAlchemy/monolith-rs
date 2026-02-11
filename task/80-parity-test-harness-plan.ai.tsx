import { Agent, Instructions, Program, Prompt, System, Asset } from '@unpack/ai';

export default (
  <Program
    id="native-training-layers-optimizers-hooks"
    target={{ language: 'rust' }}
    description="Port native_training layers, optimizers, hooks, and metrics into monolith-training with parity tests."
  >
    <Asset id="log" kind="doc" path="generated/monolith/task/native-training-layers-optimizers-hooks.md" />
    <Agent id="apply-native-training-layers-optimizers-hooks">
      <Prompt>
        <System>
          You are porting native_training layers/optimizers/hooks to Rust. Edit
          the Rust codebase directly; do not write plans, mapping tables, or JSON.
        </System>
        <Instructions>
Implement parity for these Python files and directories:
- monolith/native_training/layers/**
- monolith/native_training/losses/**
- monolith/native_training/optimizers/**
- monolith/native_training/hooks/**
- monolith/native_training/logging_ops.py
- monolith/native_training/logging_ops_test.py
- monolith/native_training/learning_rate_functions.py
- monolith/native_training/learning_rate_functions_test.py
- monolith/native_training/alert/**
- monolith/native_training/metric/**
- monolith/native_training/debugging/**

Rust targets:
- monolith-rs/crates/monolith-training
- monolith-rs/crates/monolith-layers
- monolith-rs/crates/monolith-optimizer

Requirements:
- Preserve numerical behavior, config defaults, and error handling.
- Keep TF runtime optional; do not vendor TF binaries.
- Add or update Rust tests that mirror the Python tests listed above.

Do not emit mapping docs or JSON. Focus on code edits and tests.
        </Instructions>
      </Prompt>
    </Agent>
  </Program>
);
