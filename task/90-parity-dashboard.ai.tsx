import { Agent, Instructions, Program, Prompt, System, Asset } from '@unpack/ai';

export default (
  <Program
    id="native-training-feature-ops"
    target={{ language: 'rust' }}
    description="Port native_training feature/hash ops into monolith-training with parity tests."
  >
    <Asset id="log" kind="doc" path="generated/monolith/task/native-training-feature-ops.md" />
    <Agent id="apply-native-training-feature-ops">
      <Prompt>
        <System>
          You are porting native_training feature and hash-table ops to Rust.
          Edit the Rust codebase directly; do not write plans, mapping tables,
          or JSON.
        </System>
        <Instructions>
Implement parity for these Python files and directories:
- monolith/native_training/feature.py
- monolith/native_training/feature_test.py
- monolith/native_training/feature_utils.py
- monolith/native_training/feature_utils_test.py
- monolith/native_training/embedding_combiners.py
- monolith/native_training/embedding_combiners_test.py
- monolith/native_training/hash_table_ops.py
- monolith/native_training/hash_table_ops_test.py
- monolith/native_training/hash_filter_ops.py
- monolith/native_training/hash_filter_ops_test.py
- monolith/native_training/hash_table_utils.py
- monolith/native_training/hash_table_utils_test.py
- monolith/native_training/multi_hash_table_ops.py
- monolith/native_training/multi_hash_table_ops_test.py
- monolith/native_training/multi_type_hash_table.py
- monolith/native_training/multi_type_hash_table_test.py
- monolith/native_training/ragged_utils.py
- monolith/native_training/ragged_utils_test.py
- monolith/native_training/nested_tensors.py
- monolith/native_training/nested_tensors_test.py
- monolith/native_training/gen_seq_mask.py
- monolith/native_training/gen_seq_mask_test.py
- monolith/native_training/net_utils.py
- monolith/native_training/net_utils_test.py
- monolith/native_training/mlp_utils.py
- monolith/native_training/hash_table_ops.proto
- monolith/native_training/multi_hash_table_ops.proto

Rust targets:
- monolith-rs/crates/monolith-training
- monolith-rs/crates/monolith-tensor
- monolith-rs/crates/monolith-proto (for hash-table protos)

Requirements:
- Preserve hashing/embedding semantics and error behavior.
- Keep TF runtime optional; do not vendor TF binaries.
- Add or update Rust tests that mirror the Python tests listed above.

Do not emit mapping docs or JSON. Focus on code edits and tests.
        </Instructions>
      </Prompt>
    </Agent>
  </Program>
);
