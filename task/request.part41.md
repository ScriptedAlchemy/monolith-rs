<!--
Source: task/request.md
Lines: 9627-9785 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/embedding_combiners_test.py`
<a id="monolith-native-training-embedding-combiners-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 47
- Purpose/role: Validates `ReduceSum` and `FirstN` combiners, including unknown shape handling.
- Key symbols/classes/functions: `CombinerTest` test cases.
- External dependencies: `tensorflow`, `embedding_combiners`.
- Side effects: Uses TF v1 graph mode when run as main.

**Required Behavior (Detailed)**
- `testReduceSum`:
  - `key = RaggedTensor.from_row_lengths([1,2,3], [1,2])` and `emb=[[1.0],[2.0],[3.0]]`.
  - `ReduceSum.combine` returns `[[1.0],[5.0]]`.
- `testFirstN`:
  - `key = RaggedTensor.from_row_lengths([1,2,3,4,5,6], [1,2,3])`, `emb` 6x1.
  - `FirstN(2)` returns `[[[1.0],[0.0]], [[2.0],[3.0]], [[4.0],[5.0]]]` (zero-padded for row 0).
- `testFirstNUnknownShape`:
  - `key` is ragged placeholder, `emb` placeholder `[None,6]`.
  - `FirstN(2)` result shape is `[None, 2, 6]`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-layers/tests/embedding_combiners_test.rs` (new) or `monolith-rs/crates/monolith-layers/src/embedding.rs` unit tests.
- Rust public API surface: `ReduceSum`, `FirstN` combiners or equivalent pooling + sequence embedding logic.
- Data model mapping: ragged input modeled as `(values, row_lengths)`; tests should construct identical ragged cases.
- Feature gating: none for Candle backend; TF runtime tests optional.
- Integration points: `embedding_combiners` module or embedded in `embedding` layers.

**Implementation Steps (Detailed)**
1. Add Rust tests mirroring the three cases above.
2. Ensure `FirstN` produces zero-padded outputs for short rows.
3. Verify output shape inference for unknown batch size, but fixed `max_seq_length` and embedding dim.

**Tests (Detailed)**
- Python tests: `monolith/native_training/embedding_combiners_test.py`.
- Rust tests: add parity tests in `monolith-rs/crates/monolith-layers/tests/embedding_combiners_test.rs`.
- Cross-language parity test: compare outputs for the same ragged inputs.

**Gaps / Notes**
- Python uses ragged tensors; Rust tests must define an equivalent ragged representation.

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

### `monolith/native_training/entry.py`
<a id="monolith-native-training-entry-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 630
- Purpose/role: Defines optimizer/initializer/compressor wrappers and hash table config helpers that emit `embedding_hash_table_pb2` proto configs for Monolith hash tables.
- Key symbols/classes/functions: `Optimizer`, `SgdOptimizer`, `AdagradOptimizer`, `AdadeltaOptimizer`, `AdamOptimizer`, `AmsgradOptimizer`, `BatchSoftmaxOptimizer`, `MomentumOptimizer`, `MovingAverageOptimizer`, `RmspropOptimizer`, `RmspropV2Optimizer`, `FTRLWithGroupSparsityOptimizer`, `AdaGradWithGroupLassoOptimizer`, `DynamicWdAdagradOptimizer`, `FtrlOptimizer`, `Initializer`, `ZerosInitializer`, `ConstantsInitializer`, `RandomUniformInitializer`, `BatchSoftmaxInitializer`, `Compressor`, `OneBitCompressor`, `FixedR8Compressor`, `Fp16Compressor`, `Fp32Compressor`, `CombineAsSegment`, `HashTableConfig`, `CuckooHashTableConfig`, `HashTableConfigInstance`.
- External dependencies: `tensorflow`, `monolith_export`, `embedding_hash_table_pb2` (package `monolith.hash_table`).
- Side effects: None (pure config assembly), except for exceptions in constructors and learning-rate helpers.

**Required Behavior (Detailed)**
- `_convert_to_proto(obj, proto)`:
  - Calls `proto.SetInParent()`.
  - Iterates `obj.__dict__` and assigns any non-`None` field values to the proto fields with the same name.
- `Optimizer` (abstract): `as_proto()` returns `embedding_hash_table_pb2.OptimizerConfig`.
- `StochasticRoundingFloat16OptimizerWrapper(optimizer)`:
  - Wraps any optimizer; `as_proto()` calls inner optimizer then sets `stochastic_rounding_float16 = True` on the returned config.
- Optimizers (all call `_convert_to_proto` on their respective `OptimizerConfig` sub-message):
  - `SgdOptimizer(learning_rate=None)` -> `opt.sgd`.
  - `AdagradOptimizer(learning_rate=None, initial_accumulator_value=None, hessian_compression_times=1, warmup_steps=0, weight_decay_factor=0.0)` -> `opt.adagrad`.
  - `AdadeltaOptimizer(learning_rate=None, weight_decay_factor=0.0, averaging_ratio=0.9, epsilon=0.01, warmup_steps=0)` -> `opt.adadelta`.
  - `AdamOptimizer(learning_rate=None, beta1=0.9, beta2=0.99, use_beta1_warmup=False, weight_decay_factor=0.0, use_nesterov=False, epsilon=0.01, warmup_steps=0)` -> `opt.adam`.
  - `AmsgradOptimizer(learning_rate=None, beta1=0.9, beta2=0.99, weight_decay_factor=0.0, use_nesterov=False, epsilon=0.01, warmup_steps=0)` -> `opt.amsgrad` (not `monolith_export`).
  - `BatchSoftmaxOptimizer(learning_rate=None)` -> `opt.batch_softmax`.
  - `MomentumOptimizer(learning_rate=None, weight_decay_factor=0.0, use_nesterov=False, momentum=0.9, warmup_steps=0)` -> `opt.momentum`.
  - `MovingAverageOptimizer(momentum=0.9)` -> `opt.moving_average` (not `monolith_export`).
  - `RmspropOptimizer(learning_rate=None, weight_decay_factor=0.0, momentum=0.9)` -> `opt.rmsprop`.
  - `RmspropV2Optimizer(learning_rate=None, weight_decay_factor=0.0, momentum=0.9)` -> `opt.rmspropv2`.
  - `FTRLWithGroupSparsityOptimizer(learning_rate=None, initial_accumulator_value=None, beta=None, warmup_steps=0, l1_regularization=None, l2_regularization=None)` -> `opt.group_ftrl` with `l1_regularization_strength` and `l2_regularization_strength` fields set.
  - `AdaGradWithGroupLassoOptimizer(learning_rate=None, beta=None, initial_accumulator_value=None, l2_regularization=None, weight_decay_factor=0.0, warmup_steps=0)` -> `opt.group_adagrad` with `l2_regularization_strength` set.
  - `DynamicWdAdagradOptimizer(learning_rate=None, initial_accumulator_value=None, hessian_compression_times=1, warmup_steps=0, weight_decay_factor=0.0, decouple_weight_decay=True, enable_dynamic_wd=True, flip_direction=True, dynamic_wd_temperature=1.0)` -> `opt.dynamic_wd_adagrad`.
  - `FtrlOptimizer(learning_rate=None, initial_accumulator_value=None, beta=None, warmup_steps=0, l1_regularization=None, l2_regularization=None)` -> `opt.ftrl` with `l1_regularization_strength` and `l2_regularization_strength` fields set.
- `Initializer` (abstract): `as_proto()` returns `embedding_hash_table_pb2.InitializerConfig`.
  - `ZerosInitializer()` -> `init.zeros`.
  - `ConstantsInitializer(constant)` -> `init.constants` with `constant` set.
  - `RandomUniformInitializer(minval=None, maxval=None)` -> `init.random_uniform`.
  - `BatchSoftmaxInitializer(init_step_interval)`:
    - Raises `ValueError` if `init_step_interval < 1`.
    - Stores `constant = init_step_interval` and returns `init.constants`.
- `Compressor` (abstract): `as_proto()` returns `embedding_hash_table_pb2.FloatCompressorConfig`.
  - `OneBitCompressor(step_size=200, amplitude=0.05)` -> `comp.one_bit` with `step_size` and `amplitude`.
  - `FixedR8Compressor(fixed_range=1.0)` -> `comp.fixed_r8` with `r` field set.
  - `Fp16Compressor()` -> `comp.fp16`.
  - `Fp32Compressor()` -> `comp.fp32`.
- `CombineAsSegment(dim_size, initializer, optimizer, compressor)`:
  - Accepts either wrapper objects or raw proto configs.
  - Creates `EntryConfig.Segment`, sets `dim_size`, and `CopyFrom` for init/opt/comp configs.
- `HashTableConfig` (abstract): `mutate_table(table_config)`.
- `CuckooHashTableConfig(initial_capacity=1, feature_evict_every_n_hours=0)`:
  - `mutate_table` sets `table_config.initial_capacity` and `table_config.cuckoo.SetInParent()`.
  - If `feature_evict_every_n_hours > 0`, sets `enable_feature_eviction=True` and `feature_evict_every_n_hours`.
- `HashTableConfigInstance(table_config, learning_rate_fns, extra_restore_names=None)`:
  - Stores a copy of `extra_restore_names` (default `[]`).
  - `__str__` returns `TableConfigPB:<serialized>, LearningRateFns:[<fn_strs>]` where proto is `SerializeToString()` and each fn uses `str(fn)`.
  - `call_learning_rate_fns()`:
    - Under name scope `learning_rate`, calls each fn if callable, else casts to `tf.float32`.
    - Returns `tf.stack(learning_rates)`; raises `Exception` if list is empty.
  - `call_learning_rate_fns_fewer_ops()`:
    - Same call rules but returns raw list (no `tf.cast` for non-callables) and raises if empty.
  - `set_learning_rate_tensor()` stores computed tensor; `learning_rate_tensor` property exposes it.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-hash-table/src` plus proto types in `monolith-rs/crates/monolith-proto` (`monolith::hash_table::*`).
- Rust public API surface:
  - Builder structs mirroring the optimizer/initializer/compressor wrappers (e.g., `SgdOptimizerConfig`, `AdamOptimizerConfig`, `RandomUniformInitializerConfig`, `OneBitCompressorConfig`).
  - `CombineAsSegment` equivalent that produces `monolith::hash_table::EntryConfig::Segment`.
  - `HashTableConfig` trait and `CuckooHashTableConfig` implementation.
  - `HashTableConfigInstance` struct holding a `EmbeddingHashTableConfig`, learning-rate fn list, and extra restore names.
- Data model mapping: Use `monolith::hash_table::OptimizerConfig`, `InitializerConfig`, `FloatCompressorConfig`, `EmbeddingHashTableConfig` from `monolith-proto`.
- Feature gating: none for Candle backend; TF runtime should reuse the same proto configs.
- Integration points: `feature.py`, `hash_table_ops.py`, `multi_hash_table_ops.py`, and `cpu_training.py` equivalents.

**Implementation Steps (Detailed)**
1. Add Rust config builder types that mirror field names and defaults from Python (including `None`-skip semantics).
2. Implement a `_convert_to_proto` equivalent that only sets fields when they are `Some(...)`.
3. Implement `StochasticRoundingFloat16OptimizerWrapper` as a decorator that toggles `stochastic_rounding_float16` on `OptimizerConfig`.
4. Implement `CombineAsSegment` with enum inputs to accept either builder or direct proto.
5. Port `HashTableConfigInstance.__str__` to a deterministic `Display` implementation using serialized proto bytes + fn string signatures.
6. Implement `call_learning_rate_fns` and `call_learning_rate_fns_fewer_ops` using Candle/Tensor APIs; preserve error messages when list is empty.
7. Add unit tests for each builder and for `CombineAsSegment` output.

**Tests (Detailed)**
- Python tests: `monolith/native_training/entry_test.py`.
- Rust tests: `monolith-rs/crates/monolith-hash-table/tests/entry_test.rs` (new) to mirror optimizer/initializer/compressor config creation and `HashTableConfigInstance.__str__` behavior.
- Cross-language parity test: compare serialized proto bytes produced by Python and Rust for each optimizer/initializer/compressor config.

**Gaps / Notes**
- Proto fields must match Python names exactly; default handling must skip `None` to avoid overwriting proto defaults.
- `HashTableConfigInstance.__str__` depends on `SerializeToString()` ordering; ensure Rust uses the same proto serialization (protobuf binary) for parity.

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
