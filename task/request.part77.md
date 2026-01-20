<!--
Source: task/request.md
Lines: 17772-17900 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/multi_type_hash_table.py`
<a id="monolith-native-training-multi-type-hash-table-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 435
- Purpose/role: Abstractions for multi-type hash tables and a merged-table wrapper that deduplicates configs across slots.
- Key symbols/classes/functions: `BaseMultiTypeHashTable`, `MultiTypeHashTable`, `MergedMultiTypeHashTable`.
- External dependencies: TensorFlow, hash_table_ops, distribution_ops, prefetch_queue.
- Side effects: Uses device placement; may register queue hooks for pipelined execution.

**Required Behavior (Detailed)**
- `BaseMultiTypeHashTable`:
  - Abstract API: `lookup`, `assign`, `assign_add`, `reinitialize`, `apply_gradients`, `as_op`, `get_table_dim_sizes`.
  - Supports queue hook aggregation via `add_queue_hook` and `get_queue_hooks`.
- `MultiTypeHashTable`:
  - Builds per-slot hash tables using a factory; maintains resources and learning_rate tensors.
  - `lookup` returns per-slot embeddings.
  - `assign` / `assign_add` / `apply_gradients` delegate to per-slot tables and return updated copy.
  - `as_op` returns no-op dependent on all table ops.
  - Supports fused lookup/optimize via custom ops using flattened learning rate tensor.
  - `reinitialize` not supported (raises NotImplementedError).
- `_IndexedValues` dataclass: records merged slots, index, and value tensor for merged operations.
- `MergedMultiTypeHashTable`:
  - Deduplicates slots with identical config (stringified config as key).
  - Builds merged slot names using MD5; tracks slot->merged_slot mapping.
  - If old naming mismatch, adds `extra_restore_names`.
  - `lookup`:
    - Merges slot ids by merged slot, calls underlying table lookup.
    - Splits embeddings back to original slots using sizes.
    - Supports optional early reorder results via `auxiliary_bundle`.
  - `assign` / `assign_add` / `apply_gradients`:
    - Merges ids and values before delegating.
    - `skip_merge_id` option in `_update` to bypass merge for certain paths.
  - `reinitialize` not supported.
  - `get_table_dim_sizes` returns inferred sizes for merged configs.

**Rust Mapping (Detailed)**
- Target crate/module: N/A (TF hash table ops).
- Rust public API surface: if embedding tables are implemented in Rust, add multi-type table abstraction and merged wrapper.
- Data model mapping: slot->embedding tensors, per-slot configs.
- Feature gating: embedding/hash table feature only.
- Integration points: embedding lookup and optimizer updates.

**Implementation Steps (Detailed)**
1. Implement BaseMultiTypeHashTable trait in Rust with lookup/assign APIs.
2. Add merged-table wrapper to reduce redundant configs.
3. Preserve slot ordering and size-based splitting for merged lookups.

**Tests (Detailed)**
- Python tests: `multi_type_hash_table_test.py`.
- Rust tests: add tests for merged slot mapping and lookup correctness.
- Cross-language parity test: compare embeddings for identical configs.

**Gaps / Notes**
- Merging uses `str(config)`; any changes in string representation alter merge behavior.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed

### `monolith/native_training/multi_type_hash_table_test.py`
<a id="monolith-native-training-multi-type-hash-table-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 326
- Purpose/role: Tests MultiTypeHashTable and MergedMultiTypeHashTable behaviors, including fused ops and name stability.
- Key symbols/classes/functions: `MultiTypeHashTableTest.*`, `MergedMultiTypeHashTable.*`.
- External dependencies: TensorFlow, hash_table_ops custom ops.
- Side effects: None beyond TF session operations.

**Required Behavior (Detailed)**
- `test_basic`: assign_add + lookup values per slot.
- `test_apply_gradients`: applies gradients; expected negative embeddings.
- `test_apply_gradients_with_learning_rate_decay`: uses PolynomialDecay learning rate; checks scaled updates.
- `test_apply_gradients_without_lookup`: gradient updates without prior lookup.
- `test_fused_lookup` / `test_fused_lookup_multi_shards`:
  - Validate fused lookup outputs (embeddings, splits, offsets).
- `test_fused_apply_gradients` / `test_fused_apply_gradients_missing_tables`:
  - Validate fused optimize updates and resulting embeddings.
- `MergedMultiTypeHashTable.testBasic`:
  - Merges slots 1/2; verifies combined updates and gradients.
- `testNameStability`:
  - Ensures merged slot name (MD5) deterministic; factory called with single merged key.
- `testRestoreName`:
  - Verifies `extra_restore_names` for old naming convention `fc_slot_*`.

**Rust Mapping (Detailed)**
- Target crate/module: N/A.
- Rust public API surface: none.
- Data model mapping: custom ops for hash tables.
- Feature gating: TF runtime + custom ops.
- Integration points: embedding table implementation.

**Implementation Steps (Detailed)**
1. If Rust binds custom ops, port tests for assign/lookup and fused ops.
2. Validate merged slot mapping and restore name behavior.

**Tests (Detailed)**
- Python tests: `multi_type_hash_table_test.py`.
- Rust tests: none.
- Cross-language parity test: compare embeddings and offsets for fused ops.

**Gaps / Notes**
- Tests rely on custom ops and fixed learning rate semantics.

**Verification Checklist (Must be Checked Off)**
- [ ] All public functions/classes mapped to Rust
- [ ] Behavior matches Python on normal inputs
- [ ] Error handling parity confirmed
- [ ] Config/env precedence parity confirmed
- [ ] I/O formats identical (proto/JSON/TFRecord/pbtxt)
- [ ] Threading/concurrency semantics preserved
- [ ] Logging/metrics parity confirmed
- [ ] Performance risks documented
- [ ] Rust tests added and passing
- [ ] Cross-language parity test completed
