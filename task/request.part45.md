<!--
Source: task/request.md
Lines: 10250-10441 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/feature.py`
<a id="monolith-native-training-feature-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 663
- Purpose/role: Defines feature slots/columns, embedding table interfaces, and embedding slice fusion logic for Monolith feature pipelines.
- Key symbols/classes/functions: `FeatureEmbTable`, `FeatureSlotConfig`, `FeatureSlot`, `FeatureColumn`, `FeatureFactory`, `DummyFeatureEmbTable`, `DummyFeatureFactory`, `_FeatureFactoryFusionHelper`, `create_embedding_slices`, `EmbeddingFeatureEmbTable`, `FeatureFactoryFromEmbeddings`, `EmbeddingLayoutFactory`.
- External dependencies: TensorFlow, `entry`, `embedding_combiners`, `distribution_ops`, `device_utils`, `ragged_utils`, `embedding_hash_table_pb2`, `prefetch_queue`, `is_exporting`.
- Side effects: Uses env var `MONOLITH_GPU_FEATURE_FACTORY_FUSION_LEVEL`; adds tensors to TF collection `monolith_reduced_embs`.

**Required Behavior (Detailed)**
- Constants:
  - `_FEATURE_STRAT_END_KEY = "{}:{}_{}"` used to key embedding slices by feature name and slice bounds.
  - `DEFAULT_EXPIRE_TIME = 36500` (days).
- `FeatureEmbTable` (abstract):
  - `add_feature_slice(segment, learning_rate_fn)` and `set_feature_metadata(feature_name, combiner)` are no-ops in base.
  - `embedding_lookup(feature_name, start, end)` is abstract.
- `FeatureSlice`: NamedTuple with `feature_slot`, `start`, `end`.
- `FeatureSlotConfig`:
  - Defaults for bias/default vector configs using `entry` builders; default expire time and occurrence threshold.
  - `__post_init__` sets `name` to `slot_id` string if not provided.
- `FeatureSlot`:
  - Holds table/config, current dim size, and registered feature columns.
  - If `has_bias` true, creates a bias slice of dim 1 using bias configs.
  - `add_feature_slice`:
    - Applies defaults for initializer/optimizer/compressor/learning_rate_fn.
    - Creates `EntryConfig.Segment` via `entry.CombineAsSegment` and registers with table.
    - Returns `FeatureSlice(start, end)` and updates `_current_dim_size`.
  - Registers feature columns via `_add_feature_column`, and updates table metadata.
  - `get_feature_columns`, `get_bias_slice`, `slot` (int of name), `name`.
- `FeatureColumn`:
  - Factory helpers: `reduce_sum`, `reduce_mean`, `first_n(seq_length)`.
  - `embedding_lookup(s)` asserts slice belongs to slot and delegates to table.
  - `get_all_embeddings_concat()` returns full embedding tensor for gradients (start/end None).
  - `get_all_embedding_slices()` returns per-slice tensors for this feature name.
  - `get_bias()` returns bias slice embeddings.
  - `set_size_tensor(row_lengths)`:
    - Only for `FirstN` combiner; builds boolean mask `[B, max_seq_length]` from row_lengths and stores as int32 `size_tensor`.
- `FeatureFactory` (abstract):
  - Manages `slot_to_occurrence_threshold` and `slot_to_expire_time`.
  - `apply_gradients` default raises `NotImplementedError`.
- `DummyFeatureEmbTable` (config collection):
  - `add_feature_slice` auto-infers `learning_rate_fn` from optimizer config:
    - If optimizer has `warmup_steps > 0`, uses `PolynomialDecay` from 0.0 to `learning_rate` over warmup_steps.
    - Else uses `opt_config.learning_rate` value directly.
  - `embedding_lookup` builds placeholders and combines via combiner; respects fixed batch size.
  - `get_table_config` merges slices via `_merge_slices` and returns `TableConfig` with:
    - `slice_configs` merged
    - `feature_names`
    - `unmerged_slice_dims` (original slice sizes)
    - `hashtable_config`
    - `feature_to_combiners`.
  - `_merge_slices` merges adjacent slices when proto config (excluding dim_size) and `learning_rate_fn` string match; sums dim_size.
- `DummyFeatureFactory`:
  - Ensures unique table name; registers slot thresholds/expire times by slot_id.
  - `apply_gradients` returns `tf.no_op()`.
  - `get_table_name_to_table_config` errors if a table has no slices.
- `EmbeddingFeatureEmbTable`:
  - Wraps actual embeddings and embedding_slices; returns full embedding when start/end None; otherwise uses `_FEATURE_STRAT_END_KEY`.
- `_FeatureFactoryFusionHelper`:
  - Collects ragged rows, value_rowids, embeddings, batch_size, and slice_dims.
  - `reduce_and_split`: CPU scatter_nd reduce and split; adds reduced tensor to collection.
  - `fused_reduce_and_split`: uses `distribution_ops.fused_reduce_sum_and_split` on CPU.
  - `fused_reduce_then_split`: GPU `distribution_ops.fused_reduce_and_split_gpu` and then manual split mapping.
- `create_embedding_slices(...)`:
  - For each feature:
    - If combiner is `ReduceSum`, uses helper for fused reduce+split.
    - Else uses combiner + `tf.split` on combined embeddings.
  - Chooses fusion path:
    - If not exporting and within GPU placement and `MONOLITH_GPU_FEATURE_FACTORY_FUSION_LEVEL==1` -> `fused_reduce_then_split`.
    - Else if not exporting -> `reduce_and_split` (CPU) or `fused_reduce_and_split` (CPU fused) depending on placement.
    - If exporting -> `reduce_and_split`.
  - Constructs `embedding_slices` dict keyed by name/start/end.
- `FeatureFactoryFromEmbeddings`: builds `EmbeddingFeatureEmbTable` from `name_to_embeddings` and `name_to_embedding_slices`.
- `EmbeddingLayoutFactory`:
  - Uses `PartitionedHashTable` to apply gradients with layout embeddings and optional async push.
  - `get_layout` and `flattened_layout` expose layout tensors.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/src/feature.rs` plus embedding combiner logic in `monolith-rs/crates/monolith-layers` and hash table config in `monolith-rs/crates/monolith-hash-table`.
- Rust public API surface:
  - `FeatureSlot`, `FeatureSlice`, `FeatureColumn` equivalents in `monolith-core`.
  - Combiner types (`ReduceSum`, `ReduceMean`, `FirstN`) in `monolith-layers`.
  - `create_embedding_slices` and fusion helpers in a new `monolith-training` or `monolith-layers` module.
- Data model mapping: represent ragged inputs as `(values, row_splits)` and carry slice dimensions explicitly.
- Feature gating: GPU fused ops require TF runtime or custom kernels; Candle backend should use CPU reduce+split.
- Integration points: `feature_utils`, embedding lookup, and hash table update paths.

**Implementation Steps (Detailed)**
1. Implement FeatureSlot/FeatureColumn config layering in Rust, mapping to existing `monolith-core` feature structs.
2. Implement `DummyFeatureEmbTable` and `DummyFeatureFactory` for config collection and tests.
3. Implement `create_embedding_slices` with reduce/split logic; add optional fused path when TF runtime is available.
4. Preserve learning rate warmup logic in `DummyFeatureEmbTable.add_feature_slice`.
5. Add `EmbeddingLayoutFactory` wrapper around Rust hash table gradient application (or stub if hash table not available).
6. Add tests matching Python expectations for slice merging, bias, combiner selection, and fused behavior.

**Tests (Detailed)**
- Python tests: `monolith/native_training/feature_test.py`.
- Rust tests: add `monolith-rs/crates/monolith-core/tests/feature_test.rs` and/or `monolith-rs/crates/monolith-layers/tests/feature_factory_test.rs`.
- Cross-language parity test: compare slice configs (serialized proto), embedding slice outputs, and combiners.

**Gaps / Notes**
- Many operations rely on TF ragged tensors and custom fused ops; Rust backend must choose between TF runtime or native reductions.
- The `_FEATURE_STRAT_END_KEY` key format must match exactly to ensure lookup parity.

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

### `monolith/native_training/feature_test.py`
<a id="monolith-native-training-feature-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 266
- Purpose/role: Tests for feature slot/column config collection, slice merging, and embedding slice creation (including fused paths and FirstN behavior).
- Key symbols/classes/functions: `CollectingConfigTest`, `EmbeddingTest`.
- External dependencies: TensorFlow, `entry`, `embedding_combiners`, `feature`, `learning_rate_functions`, `embedding_hash_table_pb2`, protobuf `text_format`.
- Side effects: None beyond TensorFlow graph execution.

**Required Behavior (Detailed)**
- `CollectingConfigTest.test_basic`:
  - Dummy table + segment dim_size=5 with sgd; reduce_sum combiner; embedding_lookup produces placeholder shape `[4, 5]`.
- `test_basic_with_seq_features`:
  - FirstN(10) combiner -> embedding_lookup placeholder shape `[4, 10, 5]`.
- `test_info`:
  - Adds segments with adagrad warmup and sgd; two adjacent sgd slices with same learning_rate_fn should merge to dim_size=4.
  - Ensures learning_rate_fn for warmup slice is a `LearningRateFunction`.
  - `feature_names` list contains `feature1`.
- `test_factory`:
  - DummyFeatureFactory creates slot and feature columns; table config includes feature names and slice dim_size=5.
- `test_factory_with_seq_features`:
  - FirstN combiners stored in `feature_to_combiners` map; verifies mapping.
- `test_factory_with_slot_occurrence_threshold`:
  - Factory stores occurrence thresholds keyed by slot_id.
- `test_factory_with_applying_gradients`:
  - Dummy factory apply_gradients accepts grads and returns no-op.
- `test_bias`:
  - Slot with `has_bias=True` exposes bias lookup without errors.
- `EmbeddingTest.test_factory`:
  - Uses `create_embedding_slices` with ReduceSum; lookup returns `[[1],[2]]` for ragged ids.
- `test_factory_with_seq_features`:
  - FirstN(2) returns sequence embeddings `[[[1],[3]], [[5],[7]]]`.
- `test_fused_factory`:
  - ReduceSum with ragged splits producing zeros in empty rows; verifies slice outputs for each slice.
- `test_fused_factory_with_seq_features_larger_than_max_seq_length`:
  - FirstN(2) truncates rows longer than max_seq_length; verifies outputs.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/tests/feature_test.rs` and/or `monolith-rs/crates/monolith-layers/tests/feature_factory_test.rs` (new).
- Rust public API surface: `FeatureSlot`, `FeatureColumn`, `DummyFeatureFactory`, `create_embedding_slices`, combiners.
- Data model mapping: ragged inputs as values + row_splits; test outputs should match Python arrays.
- Feature gating: fused GPU paths behind TF runtime or CUDA feature; CPU paths always available.
- Integration points: `entry` config builder and learning rate functions.

**Implementation Steps (Detailed)**
1. Port each test case with deterministic tensors.
2. Validate slice merging behavior using serialized proto bytes for segments.
3. Add tests for FirstN shape and truncation behavior.
4. Ensure `DummyFeatureFactory` tracks occurrence thresholds and bias slice creation.

**Tests (Detailed)**
- Python tests: `monolith/native_training/feature_test.py`.
- Rust tests: add parity tests under `monolith-rs/crates/monolith-core/tests`.
- Cross-language parity test: compare outputs and merged slice configs with Python reference.

**Gaps / Notes**
- Fused GPU behavior depends on custom ops; Rust tests should skip if TF runtime is unavailable.

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
