<!--
Source: task/request.md
Lines: 3681-3913 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/core/feature.py`
<a id="monolith-core-feature-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 611
- Purpose/role: Sail-like feature API for defining feature slots/slices/columns and embedding lookups with optional merged vector handling.
- Key symbols/classes/functions: `FeatureSlice`, `FeatureSlot`, `FeatureColumnV1`, `FeatureColumn3D`, `Env`.
- External dependencies: `tensorflow`, `absl.logging`, `collections.namedtuple` (imported; unused).
- Side effects: creates TF placeholders, mutates shared `Env` state, writes to `_tpu_features`, logs via `absl.logging`.

**Required Behavior (Detailed)**
- `FeatureSlice`:
  - Constructor stores `feature_slot`, `dim`, `slice_index`, `optimizer`, `initializer`, `learning_rate_fn`.
  - `__repr__` returns `[FeatureSlice][slot_{slot_id}][{slice_index}]` (used as dict key).
  - `__hash__` uses `(feature_slot.slot_id(), slice_index)`.
  - Read-only properties: `dim`, `slice_index`, `optimizer`, `initializer`, `learning_rate_fn`.
- `FeatureSlot`:
  - Constructor registers itself into `Env` via `env.set_feature_slot(slot_id, self)`.
  - If `has_bias=True`, creates bias `FeatureSlice` with `dim=1`, `slice_index=0` using bias optimizer/initializer/lr and appends to `_feature_slices`.
  - `add_feature_slice(dim, optimizer=None, initializer=None, learning_rate_fn=None)`:
    - Defaults to `default_vec_*` settings when args are `None`.
    - Creates `FeatureSlice` with `slice_index=len(_feature_slices)`; appends and returns it.
  - `add_merged_feature_slice(...)`: same as `add_feature_slice` but appends to `_merged_feature_slices`.
  - `_add_feature_column(feature_column)`:
    - Adds to `_feature_columns`.
    - If `has_bias=True`, sets `_bias` for **all** feature columns by calling `feature_column.embedding_lookup(feature_slices[0])`.
  - Properties expose `bias_*`, `default_vec_*`, `feature_slices`, `merged_feature_slices`, `feature_columns`.
- `FeatureColumnV1`:
  - Constructor stores `feature_slot`, `fc_name`; initializes placeholder dicts and `_bias=None`; registers with `FeatureSlot`.
  - `embedding_lookup(feature_slice, init_minval_for_oov=None, init_maxval_for_oov=None)`:
    - Delegates to `Env._embedding_lookup`.
  - `get_bias()` asserts `_bias is not None`.
  - `feature_slice_to_tf_placeholder`:
    - Asserts `env.is_finalized` (note: Python has a bug, method recurses).
    - Returns merged or non-merged placeholder map based on `env._merge_vector`.
- `FeatureColumn3D`:
  - Constructor sets `max_seq_length`, logs it, registers with slot.
  - `embedding_lookup(...)` delegates to `Env._seq_embedding_lookup`.
  - `size_tensor_lookup()` delegates to `Env._size_tensor_lookup`.
  - `feature_slice_to_tf_placeholder` returns 3D placeholder map.
- `Env`:
  - Constructor initializes `vocab_size_dict`, `_slot_id_to_feature_slot`, `_tpu_features=None`, `_is_finalized=False`, then calls `set_params(params)`.
  - `set_params(params)` reads:
    - `qr_multi_hashing`, `qr_hashing_threshold`, `qr_collision_rate`,
      `use_random_init_embedding_for_oov`, `merge_vector`.
  - `set_feature_slot(slot_id, feature_slot)`:
    - If already finalized, returns immediately.
    - Asserts `slot_id` uniqueness.
  - `set_tpu_features(tpu_features)`:
    - Assigns `_tpu_features`; if `_merge_vector` is true, calls `_split_merged_embedding` on each feature slot.
  - `_embedding_lookup(feature_column, feature_slice, init_minval_for_oov=None, init_maxval_for_oov=None)`:
    - Asserts slot IDs match.
    - If `_tpu_features` exists:
      - If `qr_multi_hashing` and `vocab_size_dict[slot_id] > qr_hashing_threshold`, uses quotient/remainder features (`fc_name_slice_0`, `fc_name_slice_1`) and returns their sum.
      - Otherwise uses key `"{fc_name}_{slice_index}"`.
      - If `use_random_init_embedding_for_oov` and `init_minval_for_oov` provided:
        - Computes `norm` across axis=1, replaces near-zero rows with `tf.random.uniform` values.
    - If no `_tpu_features`:
      - Creates `tf.compat.v1.placeholder(tf.float32, [None, dim])` keyed by `FeatureSlice` object.
  - `_seq_embedding_lookup(...)`:
    - Same structure as `_embedding_lookup` but uses placeholder shape `[None, max_seq_length, dim]`.
    - Random OOV initialization uses `tf.random.uniform` and `tf.norm(axis=1)` (3D norms).
  - `_size_tensor_lookup(feature_column)`:
    - If `_tpu_features` exist, reads `"{fc_name}_0_row_lengths"` and builds `[B, max_seq_length]` boolean mask (cast to `int32`).
    - Else returns placeholder `tf.float32` with shape `[None, max_seq_length]` named `"{fc_name}_size"`.
  - `finalize()`:
    - Asserts not already finalized; sets `_is_finalized=True`.
    - If `_merge_vector`, calls `_merge_vector_in_same_slot()`.
  - `_merge_vector_in_same_slot()`:
    - For each slot, keeps bias slice separate (assert `dim==1`).
    - Merges remaining slices into a single merged `FeatureSlice` whose dim is sum of vector dims.
    - For each feature column, creates placeholder for merged slice and updates `_merged_feature_slice_to_tf_placeholder`.
  - `_split_merged_embedding(feature_slot)`:
    - For each feature column, finds merged embedding from `_tpu_features`, splits by per-slice dims, and writes back individual embeddings into `_tpu_features` by `"{fc_name}_{slice_index}"`.
  - Properties:
    - `vocab_size_dict` returns stored dict.
    - `slot_id_to_feature_slot` asserts `is_finalized` (buggy in Python) then returns map.
    - `features` returns `_tpu_features`.
- Determinism: random OOV embeddings depend on TF RNG.
- Logging: `logging.info` in `FeatureColumn3D.__init__`, `logging.vlog` in lookup methods.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/src/feature.rs`.
- Rust public API surface: `FeatureSlot`, `FeatureSlice`, `SparseFeatureColumn`, `DenseFeatureColumn`, `FeatureColumn` trait.
- Data model mapping: Python Sail-like slot/slice/column + `Env` **do not exist** in Rust; current Rust types represent data containers, not TF graph wiring.
- Feature gating: TF placeholder/graph APIs are Python-only; Rust needs an explicit TF backend or a compat layer.
- Integration points: Python `Env` expected by embedding tasks; Rust `feature.rs` is not integrated with embedding task flow.

**Implementation Steps (Detailed)**
1. Decide whether to port the Sail-like API or provide a shim around existing Rust feature structs.
2. If porting, add `Env`, `FeatureSlot`, `FeatureSlice`, `FeatureColumnV1/3D` in Rust with TF backend hooks.
3. Implement placeholder / feature tensor registry for training/serving use cases.
4. Add merge/split vector logic with deterministic slice ordering.
5. Match OOV random initialization (norm threshold 1e-10) and QR hashing behavior.
6. Capture and fix Python bugs when porting (e.g., `is_finalized` recursion, undefined `slot_id`).

**Tests (Detailed)**
- Python tests: `monolith/core/feature_test.py`.
- Rust tests: add `monolith-rs/crates/monolith-core/tests/feature.rs` mirroring merge/split behavior and bias handling.
- Cross-language parity test: compare split/merge outputs and placeholder shapes.

**Gaps / Notes**
- Python `Env.is_finalized` method recurses (`return self.is_finalized`) instead of returning `_is_finalized`.
- `_embedding_lookup` uses undefined `slot_id` in QR branch; should likely be `feature_column._feature_slot.slot_id()`.
- `_seq_embedding_lookup` references `feature_slice.init_minval_for_oov`/`init_maxval_for_oov` which are not defined.
- Rust `feature.rs` implements data structures, not the TF embedding/placeholder semantics used in Python.

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

### `monolith/core/feature_test.py`
<a id="monolith-core-feature-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 178
- Purpose/role: Tests bias slice creation, feature slice indexing, feature column registration, and merged vector split/merge behavior.
- Key symbols/classes/functions: `FeatureSlotTest`, `FeatureColumnV1Test`.
- External dependencies: `tensorflow`, `monolith.core.hyperparams`, `FeatureSlot`, `FeatureColumnV1`, `Env`.
- Side effects: builds TF placeholders/tensors; uses TF session for split verification.

**Required Behavior (Detailed)**
- `FeatureSlotTest.test_has_bias`:
  - `FeatureSlot(has_bias=True)` creates one bias slice with `dim=1`, `slice_index=0`.
- `FeatureSlotTest.test_add_feature_slice`:
  - Additional slices get incrementing `slice_index` and correct dims.
- `FeatureColumnV1Test.test_add_feature_column`:
  - Creating a feature column appends to `FeatureSlot._feature_columns`.
- `FeatureColumnV1Test.test_merge_split_vector_in_same_slot`:
  - With `merge_vector=True`:
    - `_merge_vector_in_same_slot()` populates `_merged_feature_slices` and merged placeholder maps per slot and column.
    - Expected merged dims:
      - slot 1 (bias+2) -> merged dims `[1,2]`
      - slot 2 (bias only) -> `[1]`
      - slot 3 (no bias, 2+3) -> `[5]`
      - slot 4 (bias+2+3+4) -> `[1,9]`
    - `_split_merged_embedding()` splits merged embeddings into per-slice tensors with exact contents as asserted.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/src/feature.rs`.
- Rust public API surface: currently missing Env + merge/split logic.
- Data model mapping: add tests once feature API exists in Rust.
- Feature gating: TF backend likely required for placeholder/graph ops.
- Integration points: embedding tasks and feature pipeline.

**Implementation Steps (Detailed)**
1. Implement Rust equivalents for `Env`, `FeatureSlot`, `FeatureColumnV1`.
2. Add merge/split vector unit tests mirroring Python.
3. Verify placeholder/tensor semantics for no-feature and TPU-feature cases.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: `monolith-rs/crates/monolith-core/tests/feature.rs`.
- Cross-language parity test: compare split/merge outputs and bias dims.

**Gaps / Notes**
- Rust feature module is currently a different abstraction; merge/split tests are not implemented.

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

### `monolith/core/host_call.py`
<a id="monolith-core-host-call-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 248
- Purpose/role: Host call implementation for non-TPU-variable caching; collects tensors and writes summaries (AUC, deepinsight).
- Key symbols/classes/functions: `HostCall.record_summary_tensor`, `compress_tensors`, `generate_host_call_hook`.
- External dependencies: `tensorflow`, `absl.logging`.
- Side effects: creates summary writers and writes scalar/text summaries.

**Required Behavior (Detailed)**
- Similar tensor collection/compression logic to `BaseHostCall`:
  - Global step tensor is always first.
  - Tensors grouped by dtype, concatenated, expanded to batch dimension.
- `generate_host_call_hook()`:
  - If disabled, returns None.
  - Otherwise returns `_host_call` and compressed tensors.
  - `_host_call` decompresses tensors, writes scalar summaries and AUC; optional deepinsight text summaries via `_serialize_messages`.
- `_serialize_messages`:
  - Verifies shapes/dtypes, flattens tensors, writes serialized tensors as text summaries.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/src/host_call.rs`.
- Rust public API surface: host call builder with summary writer integration.

**Implementation Steps (Detailed)**
1. Port tensor compression/decompression semantics.
2. Implement summary writer outputs (scalar + text) and AUC calculation.
3. Support optional deepinsight serialization.

**Tests (Detailed)**
- Python tests: none specific.
- Rust tests: unit tests for compress/decompress and summary output formatting.

**Gaps / Notes**
- Reuses logic duplicated in BaseHostCall; consider shared helper in Rust.

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
