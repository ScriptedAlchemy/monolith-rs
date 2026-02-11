import { Agent, Instructions, Program, Prompt, System, Asset } from '@unpack/ai';

export default (
  <Program
    id="native-training-distributed-runtime"
    target={{ language: 'rust' }}
    description="Port native_training distributed runtime into monolith-training with parity tests."
  >
    <Asset id="log" kind="doc" path="generated/monolith/task/native-training-distributed-runtime.md" />
    <Agent id="apply-native-training-distributed-runtime">
      <Prompt>
        <System>
          You are porting native_training distributed runtime code to Rust.
          Edit the Rust codebase directly; do not write plans, mapping tables,
          or JSON.
        </System>
        <Instructions>
Implement parity for these Python files and directories:
- monolith/native_training/distributed_ps.py
- monolith/native_training/distributed_ps_test.py
- monolith/native_training/distributed_ps_sync.py
- monolith/native_training/distributed_ps_sync_test.py
- monolith/native_training/distributed_ps_factory.py
- monolith/native_training/distributed_ps_factory_test.py
- monolith/native_training/distributed_ps_benchmark.py
- monolith/native_training/distributed_serving_ops.py
- monolith/native_training/distributed_serving_ops_test.py
- monolith/native_training/distribution_ops.py
- monolith/native_training/distribution_ops_test.py
- monolith/native_training/distribution_ops_benchmark.py
- monolith/native_training/distribution_ops_fused_benchmark.py
- monolith/native_training/distribution_ops_fused_test.py
- monolith/native_training/distribution_utils.py
- monolith/native_training/distribute/**
- monolith/native_training/runtime/**
- monolith/native_training/ps_benchmark.py
- monolith/native_training/ps_benchmark_test.py

Rust targets:
- monolith-rs/crates/monolith-training

Requirements:
- Preserve distributed coordination, sharding, and error semantics.
- Keep TF runtime optional; do not vendor TF binaries.
- Add or update Rust tests that mirror the Python tests listed above.

Do not emit mapping docs or JSON. Focus on code edits and tests.
        </Instructions>
      </Prompt>
    </Agent>
  </Program>
);
