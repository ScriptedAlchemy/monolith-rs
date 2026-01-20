<!--
Source: task/request.md
Lines: 4604-8388 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/core/base_task.py`
<a id="monolith-core-base-task-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 95
- Purpose/role: Base class for a single training task, defining standard hyperparams and abstract input/model functions.
- Key symbols/classes/functions: `BaseTask.params`, `create_input_fn`, `create_model_fn`.
- External dependencies: `base_layer.BaseLayer`, `hyperparams.Params`.
- Side effects: none.

**Required Behavior (Detailed)**
- `params()`:
  - Extends `BaseLayer.params()` with:
    - `accelerator` (None/"tpu"/"horovod")
    - `input.eval_examples`, `input.train_examples`
    - `eval.per_replica_batch_size`, `eval.steps_per_eval`, `eval.steps`
    - `train.steps`, `train.max_steps`, `train.per_replica_batch_size`, `train.file_pattern`, `train.repeat`, `train.label_key`, `train.save_checkpoints_steps`, `train.save_checkpoints_secs`, `train.dense_only_save_checkpoints_secs`, `train.dense_only_save_checkpoints_steps`
- `create_input_fn(mode)` and `create_model_fn()` are abstract and must be overridden.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/src/base_task.rs`.
- Rust public API surface: base task trait with hyperparams schema and abstract hooks.
- Data model mapping: `Params` equivalent (typed config or map).

**Implementation Steps (Detailed)**
1. Implement base task trait/struct with default parameter schema.
2. Provide typed config structures with the same field names.
3. Ensure overridden methods are required (trait methods).

**Tests (Detailed)**
- Python tests: none specific.
- Rust tests: verify default params include the expected keys.

**Gaps / Notes**
- This depends on the Rust `Params`/hyperparams implementation to mirror Python semantics.

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

### `monolith/core/base_tpu_test.py`
<a id="monolith-core-base-tpu-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 73
- Purpose/role: Base TPU tests for running TPU tasks on CPU and validating merged vector behavior.
- Key symbols/classes/functions: `BaseTPUTest.runWithCPU`, `runMergeVectorTestOnCPU`.
- External dependencies: `model_registry`, `TPURunner`.
- Side effects: runs TPU runner in CPU test mode.

**Required Behavior (Detailed)**
- `runWithCPU(task_name)`:
  - Retrieves task params, instantiates TPURunner, sets `_cpu_test=True` and `_host_call_every_n_steps=0`, runs.
- `runMergeVectorTestOnCPU(task_name)`:
  - Enables `merge_vector` on task params; runs in CPU test mode.
  - Validates merged slot dims and embedding dims in runner task env:
    - `env = runner._task._env`.
    - Asserts number of slots equals number of merged slots.
    - For each slot, checks merged dims:
      - If original dims start with bias 1, bias retained and other dims summed.
      - Else all dims summed into single entry.
    - For each TPU feature named `slot_{slot_id}_{index}`:
      - Asserts embedding dim equals original `env._slot_to_dims[slot_id][index]`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests/base_tpu.rs`.
- Rust public API surface: TPU runner CPU-test mode.

**Implementation Steps (Detailed)**
1. Provide CPU test mode in Rust TPU runner.
2. Expose task env state for merge_vector assertions.

**Tests (Detailed)**
- Python tests: used as base class in TPU-related tests.
- Rust tests: implement equivalent helper assertions.

**Gaps / Notes**
- Requires task env structures in Rust to match slot/dim semantics.

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

### `monolith/core/core_test_suite.py`
<a id="monolith-core-core-test-suite-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 35
- Purpose/role: Aggregates core unit tests into a suite.
- Key symbols/classes/functions: `suite()`.
- External dependencies: `unittest`, `ParamsTest`, `BaseLayerTest`, `BaseEmbeddingHostCallTest`, `UtilTest`.
- Side effects: runs test suite when invoked as main.

**Required Behavior (Detailed)**
- `suite()` creates a `unittest.TestSuite` containing the four test classes.
- `__main__` runs suite with `TextTestRunner(verbosity=2)`.

**Rust Mapping (Detailed)**
- Target crate/module: N/A (Rust test harness handles suites).
- Rust public API surface: none; map to Rust test module organization.

**Implementation Steps (Detailed)**
1. Ensure Rust tests cover the same components.
2. Document that Python-style suite is not required in Rust.

**Tests (Detailed)**
- Python tests: this file
- Rust tests: already covered by individual test modules.

**Gaps / Notes**
- None.

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

### `monolith/core/dense.py`
<a id="monolith-core-dense-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 179
- Purpose/role: Custom Dense layer combining TF Keras Dense with BaseLayer and optional kernel normalization.
- Key symbols/classes/functions: `Dense.params`, `build`, `fprop`.
- External dependencies: `tensorflow`, `VarianceScaling`.
- Side effects: creates TF variables (kernel/bias/trainable norms).

**Required Behavior (Detailed)**
- `params()` defaults:
  - `units=512`, `activation=None`, `use_bias=True`.
  - `kernel_initializer=VarianceScaling(mode='fan_avg', distribution='uniform')`.
  - `bias_initializer='zeros'`.
  - `allow_kernel_norm=True`, `kernel_norm_trainable=True`.
  - `partitioner=None`.
- `__init__(params, **kwargs)`:
  - If `input_dim` provided and `input_shape` missing, sets `input_shape=(input_dim,)`.
  - Calls `BaseLayer.__init__`, then `tf.keras.layers.Dense.__init__` with params (no regularizers/constraints).
  - Sets `self.p=params`, `self.units=int(params.units)` (forces int), activation via `activations.get`.
  - Resolves `bias_initializer` via `initializers.get`.
  - Sets `supports_masking=True`, `input_spec=InputSpec(min_ndim=2)`.
  - Sets `allow_kernel_norm`, `kernel_norm_trainable`, `var_name_prefix=params.name`, `partitioner`.
- `build(input_shape)`:
  - Ensures dtype is floating/complex, else `TypeError`.
  - Requires last_dim defined; else `ValueError`.
  - `kernel_shape=[last_dim, units]`; `init_kernel = kernel_initializer(shape, dtype)`.
  - If `partitioner is None`: `kernel_initializer = lambda shape,dtype: init_kernel` (constant init); else use `init_kernel` directly.
  - Creates `self.kernel` via `tf.compat.v1.get_variable` with name `{var_name_prefix}/kernel`, partitioner optional.
  - If `allow_kernel_norm`:
    - L2-normalize kernel along axis 0 with `epsilon=1e-6`.
    - If `kernel_norm_trainable`:
      - `init_trainable_kernel_norm = np.linalg.norm(init_kernel, axis=0)`.
      - If no partitioner: `norm_initializer = lambda shape,dtype: init_trainable_kernel_norm`; else use array directly.
      - Creates `trainable_kernel_norm` variable `{var_name_prefix}/trainable_kernel_norm`.
      - `kernel = kernel * trainable_kernel_norm`.
  - If `use_bias`: `self.bias = add_weight(name='{prefix}/bias', shape=[units], initializer=bias_initializer)`, else `None`.
  - Sets `self.built = True`.
- `get_config()`:
  - Adds custom fields (`allow_kernel_norm`, `kernel_norm_trainable`, `partitioner`) and serialized initializers/activation.
- `fprop(inputs)` simply calls `self.call(inputs)`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-layers/src/dense.rs`.
- Rust public API surface: Dense layer with optional kernel normalization and partitioner hooks.

**Implementation Steps (Detailed)**
1. Implement Dense forward pass and parameter initializers.
2. Add optional kernel normalization and trainable scaling.
3. Preserve config serialization fields.

**Tests (Detailed)**
- Python tests: `monolith/core/dense_test.py`
- Rust tests: layer instantiation, dtype handling, partitioner behavior.

**Gaps / Notes**
- Partitioned variables are TF-specific; Rust may need abstraction or ignore.

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

### `monolith/core/dense_test.py`
<a id="monolith-core-dense-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 108
- Purpose/role: Tests Dense layer instantiation, dtype handling, and partitioner.
- Key symbols/classes/functions: `DenseTest` methods.
- External dependencies: `testing_utils.layer_test`, `Dense`.
- Side effects: creates TF variables and runs sessions.

**Required Behavior (Detailed)**
- `test_dense_instantiate`:
  - Uses `Dense.params()` template; creates four params with names `test_dense0..3`, `units=3`.
  - `testing_utils.layer_test(Dense, kwargs={'params': p}, input_shape=...)` for shapes `(3,2)`, `(3,4,2)`, `(None,None,2)`, `(3,4,5,2)`.
- `test_dense_dtype`:
  - Builds Dense with `dtype='float32'`, input tensor from random ints.
  - Asserts `outputs.dtype == 'float32'`.
- `test_dense`:
  - Creates Dense with `units=3`, feeds `(2,4)` ones.
  - Asserts output shape `(2,3)`.
  - Runs session with global variable init.
- `test_dense_with_partitioner`:
  - Sets `partitioner = tf.compat.v1.variable_axis_size_partitioner(1024)`, units=5.
  - Input `(2,4096)` ones; output shape `(2,5)`; runs session init + output.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-layers/tests/dense.rs`.
- Rust public API surface: Dense layer tests.

**Implementation Steps (Detailed)**
1. Port tests to Rust layer API.
2. Validate shapes and dtype outputs.
3. Provide a mock partitioner or skip if unsupported (document).

**Tests (Detailed)**
- Python tests: this file
- Rust tests: parity tests.

**Gaps / Notes**
- Some TF-specific behaviors may require stubbing in Rust.

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
  - `embedding_lookup(...)` delegates to `Env._seq_embedding_lookup` using `self._max_seq_length` (ignores passed `max_seq_length` arg).
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
  - Builds `_params.Params()` and defines required fields (`qr_multi_hashing`, `qr_hashing_threshold`, `qr_collision_rate`, `use_random_init_embedding_for_oov`, `merge_vector`).
  - `env = Env({}, params)`, `FeatureSlot(env, slot_id=1, has_bias=True)`.
  - Asserts `len(feature_slices)==1`, dim=1, slice_index=0.
- `FeatureSlotTest.test_add_feature_slice`:
  - Same params/env setup; `FeatureSlot(..., has_bias=True)`.
  - `add_feature_slice(dim=10)` results in 2 slices: bias (dim 1, idx 0) and new slice (dim 10, idx 1).
- `FeatureColumnV1Test.test_add_feature_column`:
  - Same params/env setup; `FeatureSlot(has_bias=True)` then `add_feature_slice(dim=10)`.
  - `FeatureColumnV1(fs_1, 'fc_name_1')` registers in `_feature_columns` (len==1).
- `FeatureColumnV1Test.test_merge_split_vector_in_same_slot`:
  - With `merge_vector=True`:
    - Creates slots/slices:
      - slot1 has_bias True + slice dim 2.
      - slot2 has_bias True only.
      - slot3 has_bias False + slices dim 2 and 3.
      - slot4 has_bias True + slices dim 2,3,4.
    - Creates FeatureColumnV1s for slots; calls `embedding_lookup` to populate placeholders.
    - Calls `env._merge_vector_in_same_slot()` and asserts merged slice counts and dims:
      - fs1 merged dims [1,2]; fs2 [1]; fs3 [5]; fs4 [1,9].
      - Each FeatureColumn has merged placeholders for expected slices.
    - Split test:
      - Sets `env._tpu_features` with merged tensors: `fc_name_1_0`, `fc_name_1_1`, `fc_name_2_0`, `fc_name_3_0`, `fc_name_4_0`, `fc_name_4_1`, `fc_name_5_0`, `fc_name_5_1`.
      - Runs `_split_merged_embedding` for fs1..fs4 inside session.
      - Asserts split tensors match expected per-slice values for fc_name_3 (2+3 split), fc_name_4/5 (bias + 2 + 3 + 4 split).

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
- Initialization mirrors `BaseHostCall`:
  - `tensor_names=["global_step"]`, `tensors=[reshape(global_step, [-1])]`, `_lists_tensor_sizes=[]`.
- `record_summary_tensor(name, tensor)`:
  - Asserts name unique and tensor rank <=1.
  - Reshapes to `[-1]` and appends (no enable_host_call guard here).
- `compress_tensors()` / `decompress_tensors()`:
  - Same dtype grouping and concat/expand logic as `BaseHostCall`.
  - Uses `tensor.shape[0].value` for sizes and `tf.split(..., axis=1)` for decompression.
  - Asserts first name is `"global_step"` (same message quirk with `[0][0]`).
- `_verify_shape_and_dtype(tensor, shape_list, dtype)`:
  - Asserts tensor is not None, shape matches, dtype matches.
- `_serialize_messages(labels, y_preds, sample_rates, req_times, gs)`:
  - Expects each tensor shape `[num_cores, batch]` (rank 2).
  - Verifies y_preds/sample_rates float32 and req_times int64.
  - Flattens each to 1D via `tf.reshape(..., [-1])`.
  - Writes serialized tensors as text summaries with keys:
    - `di_example_sample_rates`, `di_labels`, `di_preds`, `di_req_times`.
- `generate_host_call_hook()`:
  - If `_enable_host_call` True:
    - Calls `compress_tensors()` then returns `(_host_call, self._tensors)`.
  - Else logs "host_call has been disabled" and returns `None`.
  - `_host_call(*args)`:
    - `gs, tensors = decompress_tensors(args)`.
    - Creates summary writer at `{output_dir}/host_call` with `flush_millis=10000`, `max_queue=5000`.
    - Iterates over tensors (skips index 0):
      - If name contains `_avg`: `reduce_mean`.
      - If name contains `_max`: `reduce_max`.
      - If name contains labels/preds/req_time/sample_rate keys: capture in local variables.
      - Else uses `t[0]` as scalar.
      - Writes scalar summary for any `data`.
    - If labels and preds present: adds `tf.metrics.auc` and summary `auc`.
    - If `enable_deepinsight` and labels present: calls `_serialize_messages(...)`.
    - Returns `tf.group(all_v2_summary_ops, auc_op)` if auc_op exists, else `all_v2_summary_ops`.

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

### `monolith/core/hyperparams.py`
<a id="monolith-core-hyperparams-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 439
- Purpose/role: Dynamic hyperparameter container with attribute + dotted-path access, nested parameter trees, immutability, stringification, and class instantiation support.
- Key symbols/classes/functions: `_is_named_tuple`, `_SortedDict`, `_Param`, `Params`, `InstantiableParams`, `copy_params_to`, `update_params`, `allowed_kwargs`.
- External dependencies: `copy`, `re`, `six`, `inspect.signature/Parameter`, `tensorflow` (Tensor detection for deepcopy).
- Side effects: error messages include similar-key hints; deepcopy of `tf.Tensor` intentionally keeps a reference (no deep copy).

**Required Behavior (Detailed)**
- `_is_named_tuple(x)`:
  - Returns true if `x` is a `tuple` and has `_fields` attribute.
- `_SortedDict.__repr__()`:
  - Always renders dict entries sorted by key.
- `_Param`:
  - Stores `name`, `value`, `description`.
  - `__eq__` compares only name + value (description ignored).
  - `__deepcopy__`: if value is a `tf.Tensor`, **do not copy** (keep ref); else deep-copy; memo updated.
  - `to_string(nested_depth)`:
    - For `Params`: delegate to nested `_to_string`.
    - For `dict`: produce `_SortedDict` with nested `GetRepr` on values.
    - For `list/tuple` (non-namedtuple): reconstruct same type from `GetRepr` values.
    - If value has `Repr` method, call it (used for function-like objects).
    - If value is `str`, wrap in double quotes.
    - Indent with 2 spaces per nesting level.
  - `set` stores value without copying.
  - `get` returns stored value.
- `Params`:
  - Internal storage: `_params` dict mapping name → `_Param`; `_immutable` bool.
  - `__setattr__` / `__setitem__`: if immutable -> `TypeError("This Params instance is immutable.")`. Otherwise set existing param; missing name -> `AttributeError` with `_key_error_string`.
  - `__getattr__` / `__getitem__`: missing name -> `AttributeError` with `_key_error_string`.
  - `__dir__` returns sorted param names. `__contains__`, `__len__` work on `_params`.
  - `__eq__`: only equal to another `Params` with equal `_params` (uses `_Param.__eq__`).
  - `__str__` / `_to_string`: emits multi-line block with params sorted by name; nested `Params` rendered recursively.
  - `_similar_keys(name)`:
    - Builds list of keys with >0.5 overlap of 3-char substrings from `name`.
  - `_key_error_string(name)`:
    - If similar keys exist: `"name (did you mean: [k1,k2])"` with keys sorted.
  - `define(name, default_value, description)`:
    - Reject if immutable → `TypeError("This Params instance is immutable.")`.
    - Assert `name` is `str` matching `^[a-z][a-z0-9_]*$` (raises `AssertionError`).
    - If already defined, `AttributeError("Parameter %s is already defined" % name)`.
  - `freeze()` sets `_immutable=True`; `is_immutable()` returns bool.
  - `_get_nested("a.b[0].c")`:
    - Supports dot navigation into nested `Params`.
    - Supports list indexing in a segment via `name[index]`.
    - If missing segment: `AttributeError` with partial path.
    - If intermediate is not `Params`: `AssertionError("Cannot introspect <type> for <path>")`.
  - `set(**kwargs)`:
    - If immutable -> `TypeError("This Params instance is immutable: %s" % self)`.
    - Dotted names traverse `_get_nested`; missing keys -> `AttributeError` with similar key hints.
    - Returns `self`.
  - `get(name)`:
    - Dotted names traverse `_get_nested`; missing -> `AttributeError` with similar key hints.
  - `delete(*names)`:
    - If immutable -> `TypeError("This Params instance is immutable.")`.
    - Dotted names traverse `_get_nested`; missing -> `AttributeError` with similar key hints.
    - Returns `self`.
  - `iter_params()` yields `(name, value)` for all params (dict iteration order).
  - `copy()` deep-copies `_params` and preserves `_immutable`.
- `copy_params_to(from_p, to_p, skip=None)`:
  - For each param in `from_p`, sets into `to_p`.
  - Nested `Params` are copied via `p.copy()`.
  - `skip` list omits names.
- `InstantiableParams(Params)`:
  - Defines param `cls`.
  - `instantiate()`:
    - If `cls.__init__` has exactly 2 parameters and `cls` has attribute `params` and `'params'` is a parameter: call `cls(self)`.
    - Otherwise, build inverted index of all leaf (non-Params) values; pass matching `__init__` parameter names (excluding `self`, `cls`, varargs/kwargs).
    - Additionally pass any `allowed_kwargs` present and not `None`.
- `update_params(params, args)`:
  - Recursively traverses params; if leaf key exists in `args`, sets and removes from `args`.
  - Unmatched keys remain in `args`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/src/hyperparams.rs`.
- Rust public API surface: `Params`, `ParamValue`, `InstantiableParams`, `copy_params_to`, `update_params`, `ParamsFactory`.
- Data model mapping:
  - Python `_Param` → Rust `Param` (name/value/description).
  - Python `Params` → Rust `Params` with `BTreeMap` storage.
  - Python raw values → Rust `ParamValue` (including `External` for TF tensors / opaque handles).
- Feature gating: none currently; TF handle storage likely lives in `ParamValue::External` (tie into `monolith-tf` if needed).
- Integration points: `base_layer.rs`, `base_task.rs`, `base_model_params.rs`, `model_registry.rs`.

**Implementation Steps (Detailed)**
1. Align Rust `Params` error messages with Python (TypeError/AttributeError strings).
2. Match `__str__` formatting (sorted keys, indent, list/dict rendering, quoting).
3. Implement `_similar_keys` overlap logic and error hints exactly.
4. Support list indexing in dotted paths with Python-compatible errors.
5. Preserve Tensor/opaque value deepcopy semantics (reference copy).
6. Implement `InstantiableParams.instantiate` reflection-style path or provide compatible factory shim.
7. Ensure `copy_params_to` surfaces missing keys (don’t silently drop errors).
8. Add tests mirroring `hyperparams_test.py` including exact error strings.

**Tests (Detailed)**
- Python tests: `monolith/core/hyperparams_test.py`.
- Rust tests: add `monolith-rs/crates/monolith-core/tests/hyperparams.rs`.
- Cross-language parity test: add fixture-driven JSON snapshot compare of `to_string`, errors, and nested access.

**Gaps / Notes**
- Rust currently raises `MonolithError::ConfigError` rather than `TypeError`/`AttributeError` equivalents; messages differ.
- Rust `Params` does not implement equality; Python `Params.__eq__` is relied on in tests.
- Rust `InstantiableParams` uses `ParamsFactory` (no reflection or `allowed_kwargs` handling).
- `copy_params_to` in Rust ignores `set` errors (should propagate to match Python exceptions).
- String rendering differs (Python uses sorted `_SortedDict` repr and special `Repr()` hook).
- Rust treats list navigation errors differently (`Expected list index` vs Python `Cannot introspect` assertion).

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

### `monolith/core/hyperparams_test.py`
<a id="monolith-core-hyperparams-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 277
- Purpose/role: Unit tests defining required semantics for `Params` behavior, error messages, and string rendering.
- Key symbols/classes/functions: `ParamsTest`, `TestEnum`.
- External dependencies: `unittest`, `tensorflow`, `absl.flags/logging`, `monolith.core.hyperparams`.
- Side effects: none (tests only).

**Required Behavior (Detailed)**
- `test_equals`:
  - `Params.__eq__` compares only name+value; nested `Params` must compare deeply; non-Params should not be equal.
- `test_deep_copy`:
  - `Params.copy()` deep-copies nested `Params`.
  - `tf.Tensor` values are **shared** (same object identity) in copied params.
- `test_copy_params_to`:
  - Copy respects `skip` list; `dest` only contains copied keys.
- `test_define_existing`:
  - Duplicate `define` raises `AttributeError` with "already defined".
- `test_legal_param_names`:
  - Invalid names (`None`, empty, `_foo`, `Foo`, `1foo`, `foo$`) raise `AssertionError`.
  - Valid examples: `foo_bar`, `foo9`.
- `test_set_and_get`:
  - Setting undefined param via `.set` or `setattr` raises `AttributeError` mentioning name.
  - `set/get` works; `delete` removes; subsequent access raises `AttributeError`.
- `test_set_and_get_nested_param`:
  - Dotted traversal with nested `Params` works for `get`, `set`, `delete`.
  - `set` on missing path raises `AttributeError` with dotted name.
  - Attempting to navigate into non-Params (e.g., dict) raises `AssertionError("Cannot introspect ...")`.
- `test_freeze`:
  - Frozen params reject `set`, `setattr`, `delete`, `define` with `TypeError`.
  - `get` on missing still raises `AttributeError`.
  - Nested params remain mutable after parent frozen.
  - Attempt to set `_immutable` attribute raises `TypeError`.
  - `copy()` of frozen params is still immutable.
- `test_to_string`:
  - `str(params)` yields exact multi-line formatting:
    - Sorted keys.
    - Strings quoted.
    - Dicts rendered with sorted keys and nested `Params` converted to dicts.
    - Lists render nested `Params` as dicts.
- `test_iter_params`:
  - Iteration yields all keys/values (order not asserted).
- `test_similar_keys`:
  - Error message for misspelled attribute includes sorted similar keys:
    - `actuvation (did you mean: [activation,activations])`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/src/hyperparams.rs`.
- Rust public API surface: `Params`, `ParamValue`, `InstantiableParams`, `copy_params_to`, `update_params`.
- Data model mapping: implement Rust tests mirroring Python assertions and error text.
- Feature gating: none.
- Integration points: Rust tests in `monolith-rs/crates/monolith-core/tests`.

**Implementation Steps (Detailed)**
1. Add Rust unit tests that mirror each Python test case.
2. Build helper to compare error strings for invalid operations.
3. Ensure `Params` equality is implemented for Rust tests.
4. Add string formatting parity checks using exact expected text.
5. Validate frozen behavior and nested mutability.

**Tests (Detailed)**
- Python tests: `monolith/core/hyperparams_test.py`.
- Rust tests: add unit tests in `monolith-rs/crates/monolith-core/tests/hyperparams.rs` mirroring each Python test case and expected error strings.
- Cross-language parity test: run Python tests to generate expected strings/structures, then compare Rust outputs to those fixtures.

**Gaps / Notes**
- String formatting parity is brittle; capture exact expected text from Python tests as golden fixtures.

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

### `monolith/core/mixed_emb_op_comb_nws.py`
<a id="monolith-core-mixed-emb-op-comb-nws-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 421
- Purpose/role: Keras layer for neural width search (NWS) over embedding sizes, with optional teacher distillation transform.
- Key symbols/classes/functions: `TeacherEmbeddingTransform`, `MixedEmbedOpComb`.
- External dependencies: `numpy`, `tensorflow`, `tensorflow.keras.layers.Layer/InputSpec`.
- Side effects: creates trainable TF variables, prints to stdout, uses TF random sampling.

**Required Behavior (Detailed)**
- `TeacherEmbeddingTransform(max_choice_per_embedding, teacher_embedding_sizes_list)`:
  - Stores arrays; asserts same length.
  - `build(input_shape)`:
    - Input is 2D; last dim equals sum(teacher sizes).
    - Creates `teacher_embedding_transform_weight` with shape `[sum(max_choice * size), 1]` initialized with `TruncatedNormal(stddev=0.15)`.
    - Creates `teacher_embedding_transform_bias` with shape `[sum(max_choice)]` initialized with zeros.
    - Calls `_snapshot_for_serving` on both (method not defined in this class).
  - `call(inputs)`:
    - Slices teacher embedding per size; for each slice, reshapes weight to `[size, max_choice]` and matmul.
    - Concats results along axis 1 and adds bias.
  - `compute_output_shape` raises `NotImplementedError`.
  - `get_config` returns max choices + teacher sizes.
- `MixedEmbedOpComb(slot_names, embedding_size_choices_list, warmup_steps, pretraining_steps, teacher_embedding_sizes_list=None, distillation_mask=False)`:
  - Asserts slot names length matches choices list; prints lengths.
  - Computes per-slot num choices, per-slot max embedding choice (sum of sizes), total embedding size, max num choices.
  - **Note:** `teacher_embedding_sizes_list` is ignored (set to `None`), so teacher path is disabled.
  - `build(input_shape)`:
    - Asserts input is 2D; verifies total input dim matches total embedding size.
    - Creates `arch_embedding_weights` variable of length sum(num_choices), init uniform(-1e-3, 1e-3).
    - For each slot:
      - Softmax over weights slice; compute entropy (softmax_cross_entropy_with_logits_v2).
      - Compute expected embedding dims as weighted sum of choice sizes.
      - If first choice size is 0, scale its prob by warmup schedule based on global step.
      - Create per-choice masks (ranges over max_emb_choice), pad to max_num_choices.
      - Pretraining: if global_step < pretraining_steps, probability hard-coded to `[0.5, 0.5]` (assumes 2 choices).
      - Sample choice via `tf.random.categorical`, one-hot, select mask.
      - Apply straight-through trick: mask * (1 + w - stop_gradient(w)).
    - Concatenates per-slot masks into `_arch_embedding_masks_multipler`.
    - Stores `_arch_entropy`, `_expected_emb_dims`, `_expected_zero_embedding_size_weights`,
      `_arch_embedding_weights_after_softmax_list`.
  - `call(inputs)`:
    - Computes `masked_embedding = embedding * _arch_embedding_masks_multipler`.
    - If teacher path active:
      - Builds `TeacherEmbeddingTransform`, transforms teacher embedding.
      - Computes distillation MSE against masked embedding (optionally with mask).
      - Returns `(mixed_embedding, distillation_loss, teacher_embedding_transform.name)` but `mixed_embedding` is undefined (bug).
    - Else returns `masked_embedding`.
  - `get_config` includes `slot_names`, `embedding_size_choices_list`, `warmup_steps`, `teacher_embedding_sizes_list` (pretraining_steps and distillation flag omitted).
  - `get_arch_embedding_weights()` returns variable; `get_summaries()` returns entropy, expected dims, and weight list.
- Determinism: sampling uses TF RNG; dependent on global step and random categorical.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-layers/src` (no existing equivalent).
- Rust public API surface: none; would need a layer module for NWS + distillation.
- Data model mapping: Keras layers + TF ops → Rust tensor ops (Candle/TF runtime).
- Feature gating: requires TF runtime or compatible tensor ops.
- Integration points: embedding layer selection / search in training pipeline.

**Implementation Steps (Detailed)**
1. Decide whether to support NWS (sampling-based) in Rust backend.
2. If yes, implement mask sampling, expected dims, and entropy summaries.
3. Add teacher distillation transform and MSE loss path.
4. Ensure global step and warmup/pretraining schedules are wired.
5. Fix Python bugs when porting (teacher path, mixed_embedding).

**Tests (Detailed)**
- Python tests: none in repo.
- Rust tests: add unit tests for mask construction, sampling shapes, and expected dims.
- Cross-language parity test: compare mask selection behavior under fixed RNG seeds.

**Gaps / Notes**
- `teacher_embedding_sizes_list` is ignored in `__init__` (always `None`).
- `call()` returns `mixed_embedding` in teacher path, but it is never defined.
- `pretraining_steps` only supports 2 choices (`[0.5, 0.5]` hard-coded).
- `_snapshot_for_serving` is called but not defined on `Layer` (likely missing mixin).

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

### `monolith/core/model.py`
<a id="monolith-core-model-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 320
- Purpose/role: Deprecated Sail-like TPU Model wrapper for configuring TPU embedding tables, input pipelines, and pooling.
- Key symbols/classes/functions: `Model`, `create_input_fn`, `create_feature_and_table_config_dict`, `sum_pooling`, `init_slot_to_dims`.
- External dependencies: `tensorflow`, `tpu_embedding`, `absl.logging`, `math`, `FeatureSlot/FeatureColumnV1/Env`.
- Side effects: reads vocab file, logs size info, builds TF datasets/TPU table configs.

**Required Behavior (Detailed)**
- `Model.__init__(params)`:
  - Reads `vocab_size_per_slot`, logs if fixed.
  - Builds vocab dict from file; constructs `Env` and runs `init_slot_to_dims()`.
- `_create_vocab_dict(file_path, vocab_size_per_slot=None)`:
  - Reads TSV with `slot_id<TAB>count`.
  - Skips non-digit slot IDs; uses fixed vocab size if provided.
  - Returns `{slot_id: vocab_size}`.
- `_get_feature_map()`: abstract; must return TF parse feature map.
- `_post_process_example(example)`:
  - For each slot in `env.slot_to_dims`:
    - If `vocab_size_per_slot` set, mod values in `slot_{id}_0`.
    - Duplicates `slot_{id}_0` into `slot_{id}_{i}` for each additional dim.
- `create_input_fn(file_pattern, repeat=True)`:
  - Returns `input_fn(params)` using TF Dataset:
    - `list_files(shuffle=False)`, shard by `context.current_input_fn_deployment()`.
    - Optional per-shard skip from `params["shard_skip_file_number"]`.
    - `interleave(TFRecordDataset, cycle_length, num_parallel_calls, deterministic=False)`.
    - `batch(drop_remainder=True).map(parse_example, num_parallel_calls=AUTOTUNE, deterministic=False)`.
    - `repeat()` if requested; `prefetch(AUTOTUNE)`.
- `_padding_8(dim)`:
  - Rounds up to multiple of 8.
- `_get_slot_number(optimizer, use_gradient_accumulation)`:
  - Maps TPU optimizer class → slot count:
    - FTRL: 3, Adagrad: 2, Adam: 3, SGD: 1; adds 1 if gradient accumulation.
  - Else asserts unsupported optimizer (assert uses truthy string; ineffective).
- `_get_max_slot_number()`:
  - Iterates env slots and dims; chooses bias vs vec optimizer per dim; returns max slot count.
- `create_feature_and_table_config_dict()`:
  - Requires `env.is_finalized()`.
  - For each slot/dim: builds `tpu_embedding.TableConfig` and `FeatureConfig`.
  - Computes and logs embedding table sizes (raw, padded-to-8, padded+max-slot).
  - Returns `(feature_to_config_dict, table_to_config_dict)`.
- `sum_pooling(fc_dict, input_map, features, dim, total_embeddings, add_into_embeddings=True)`:
  - For each slot in `features`, calls `fc_dict[slot].add_vector(dim)`, appends to totals, populates `input_map` with unique keys.
  - Returns single embedding if one feature; else `tf.add_n`.
- `logits_fn()`, `create_model_fn()` are abstract.
- `init_slot_to_dims()` calls `logits_fn()`, `env.finalize()`, logs slot dims.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src` (no direct equivalent).
- Rust public API surface: none; core training uses different abstractions.
- Data model mapping: TF TPU embedding configs have no Rust analog.
- Feature gating: requires TF runtime backend.
- Integration points: input pipelines, embedding table config, TPU training loop.

**Implementation Steps (Detailed)**
1. Decide whether to port deprecated Model API or replace with `base_embedding_task` parity.
2. If ported, implement vocab dict parsing, dataset sharding, and TFRecord pipeline equivalents.
3. Add TPU embedding config generator or document unsupported feature in Rust.
4. Implement pooling helper and slot/embedding table sizing logs.

**Tests (Detailed)**
- Python tests: none in repo.
- Rust tests: none (integration tests would be required with TF runtime).
- Cross-language parity test: compare table config dicts and size calculations.

**Gaps / Notes**
- File is marked deprecated; likely superseded by `base_embedding_task.py`.
- `Env` constructor here omits `params` argument (signature mismatch with current `Env`).
- `_get_slot_number` uses `assert("message")` which never fails; should raise.
- Depends on `env.slot_to_dims` and `env.slot_to_config` which do not exist in current `Env` implementation.

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

### `monolith/core/model_imports.py`
<a id="monolith-core-model-imports-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 104
- Purpose/role: Dynamic import helpers for task/model parameter modules.
- Key symbols/classes/functions: `_Import`, `ImportAllParams`, `ImportParams`.
- External dependencies: `importlib`, `absl.logging`.
- Side effects: imports Python modules dynamically; logs import results.

**Required Behavior (Detailed)**
- `_Import(name)`:
  - Logs attempt; imports module; logs success; returns True on success.
  - On ImportError, logs error and returns False.
- `ImportAllParams(task_root=_ROOT, task_dirs=_DIRS, require_success=False)`:
  - For each `task` in `task_dirs`, attempts to import `{task_root}.{task}.params.params`.
  - If `require_success` and nothing imported, raises `LookupError`.
  - Note: code defines `module_str` with `path` but `path` is undefined (bug); actual import uses `.params.params`.
- `ImportParams(model_name, task_root, task_dirs, require_success=True)`:
  - Expects `model_name` to contain a dot; else ValueError.
  - Extracts `model_module` and attempts to import it directly.
  - For built-in tasks, if `model_module` starts with `{task}.`, builds module path `{task_root}.{task}.params.{path}` and attempts import.
  - If `require_success` and no import succeeded, raises LookupError with guidance.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/src/model_imports.rs`.
- Rust public API surface: helper functions to load/register model param types (likely via registry rather than dynamic import).

**Implementation Steps (Detailed)**
1. Provide a Rust registry-based import mechanism (explicit registration, no dynamic import).
2. Mirror error messages and logging at call sites.
3. Document that Python dynamic import is replaced by static registration.

**Tests (Detailed)**
- Python tests: none specific.
- Rust tests: verify registry lookup error messages when missing.

**Gaps / Notes**
- Python uses dynamic import; Rust should use explicit registration or plugin loading if required.

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

### `monolith/core/model_registry.py`
<a id="monolith-core-model-registry-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 174
- Purpose/role: Global registry for `SingleTaskModelParams` classes with dynamic import helpers.
- Key symbols/classes/functions: `_ModelRegistryHelper`, `RegisterSingleTaskModel`, `GetAllRegisteredClasses`, `GetClass`, `GetParams`.
- External dependencies: `absl.logging`, `model_imports`, `SingleTaskModelParams`.
- Side effects: stores classes in global dict; logs registration; raises on duplicates.

**Required Behavior (Detailed)**
- Registry maps keys to param classes. Key format: `<module>.<ClassName>`; also a shortcut key with `monolith.tasks.` prefix stripped and `params.` removed.
- `_RegisterModel(src_cls)`:
  - Registers both full and shortcut keys.
  - Raises ValueError on duplicate key.
  - Logs module registration once per module.
- `RegisterSingleTaskModel` decorator:
  - Validates subclass of `SingleTaskModelParams` else TypeError.
  - Registers class and returns it.
- `GetAllRegisteredClasses()`:
  - Logs warning if none registered.
- `GetClass(class_key)`:
  - Raises LookupError if missing; logs known models.
- `GetParams(class_key)`:
  - Instantiates model params class, calls `.task()` to get task config.
- Module-level helpers call `model_imports.ImportAllParams` or `ImportParams` before returning registry results.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/src/model_registry.rs`.
- Rust public API surface: registry with registration macro/trait and lookup by string key.

**Implementation Steps (Detailed)**
1. Implement a global registry (static map) for model param factories.
2. Provide a registration macro that enforces trait bounds.
3. Implement lookup with detailed error listing known keys.
4. Replace dynamic import with explicit registration in Rust.

**Tests (Detailed)**
- Python tests: none specific.
- Rust tests: register dummy models, verify duplicate detection and lookup errors.

**Gaps / Notes**
- Shortcut key behavior must be preserved for compatibility with existing names.

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

### `monolith/core/optimizers.py`
<a id="monolith-core-optimizers-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 25
- Purpose/role: Maps string names to TensorFlow v1 optimizer classes.
- Key symbols/classes/functions: `optimizers` dict.
- External dependencies: `tf.compat.v1.train` optimizers.
- Side effects: none.

**Required Behavior (Detailed)**
- `optimizers` contains:
  - `'adagrad'`: `AdagradOptimizer`
  - `'momentum'`: `MomentumOptimizer`
  - `'rmsprop'`: `RMSPropOptimizer`
  - `'adam'`: `AdamOptimizer`

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-optimizer/src/lib.rs`.
- Rust public API surface: map from string to optimizer factory or enum.

**Implementation Steps (Detailed)**
1. Define optimizer registry in Rust with equivalent names.
2. Map to native implementations or TF bindings if used.

**Tests (Detailed)**
- Python tests: none.
- Rust tests: verify lookup table matches expected keys.

**Gaps / Notes**
- Rust may not support all TF optimizers natively; document fallbacks.

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

### `monolith/core/py_utils.py`
<a id="monolith-core-py-utils-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 313
- Purpose/role: NestedMap container with attribute access, flatten/pack utilities, and validation.
- Key symbols/classes/functions: `NestedMap` and its methods.
- External dependencies: `six`, `re`, `numpy` (imported but unused here).
- Side effects: enforces key validation on set.

**Required Behavior (Detailed)**
- `_NAME_PATTERN` is compiled from `'[A-Za-z_][A-Za-z0-9_]*'`.
- `NestedMap` is a `dict` subclass with:
  - `_HAS_DYNAMIC_ATTRIBUTES = True`.
  - `_RESERVED_KEYS = set(dir(dict))` (reserved attributes disallowed as keys).
  - `_DELETE = object()` sentinel used for filter/delete in `_RecursiveMap`.
- `__init__`:
  - Calls `dict.__init__` then validates every existing key.
  - Asserts keys are `six.string_types`, match `_NAME_PATTERN`, and are not reserved.
- `__setitem__`:
  - Same validation as `__init__`.
  - Uses `super().__setitem__` after validation.
- `__setattr__` delegates to `__setitem__` (attribute assignment adds/overwrites a key).
- `__getattr__`:
  - Returns `self[name]`.
  - On `KeyError`, raises `AttributeError` with message that includes sorted available keys.
- `__delattr__`:
  - Deletes `self[name]`.
  - On `KeyError`, raises `AttributeError` with sorted available keys.
- `copy()` returns `NestedMap(self)` (not a raw dict copy).
- `__deepcopy__` and `DeepCopy`:
  - Deep-copy the structure but **not** leaf objects.
  - Implemented as `self.Pack(self.Flatten())`.
- `FromNestedDict(x)`:
  - Recursively converts **dicts** into `NestedMap`.
  - Recurses through **lists/tuples**, preserving container type.
  - Leaves other values unchanged.
- `CheckKey(key)`:
  - Raises `ValueError("Invalid NestedMap key '...'" )` if not a string or regex mismatch.
- `GetItem(key)`:
  - Splits `key` on `.` and traverses nested maps.
  - **No list indexing support** (explicitly documented).
  - Raises `KeyError` if a key is missing.
- `Get(key, default=None)`:
  - Returns `GetItem(key)` or `default` on `KeyError` **or** `TypeError`
    (TypeError occurs when an intermediate is a list and a string key is used).
- `Set(key, value)`:
  - Splits on `.` and validates each sub-key.
  - Creates nested `NestedMap()` as needed.
  - Raises `ValueError` if a sub-key exists but is **not** `dict` or `NestedMap`.
  - Sets terminal value at the last sub-key.
- `_RecursiveMap(fn, flatten=False)`:
  - Recurses through **NestedMap** (sorted keys) and **list** (index order only).
  - Builds key paths like `foo.bar[10].baz`.
  - If `fn` returns `_DELETE`, that entry is dropped.
  - If all children of a container are deleted, returns `_DELETE`.
  - If root resolves to `_DELETE`, returns `[]` (flatten) or empty `NestedMap`.
- `Flatten()`:
  - Returns a flat list of leaf values.
  - Descends only into `NestedMap` and `list` (NOT dicts/tuples/namedtuples).
- `FlattenItems()`:
  - Returns list of `(key_path, value)` tuples using dotted paths and `[i]`.
- `Pack(lst)`:
  - Asserts `len(self.FlattenItems()) == len(lst)`.
  - Iterates through `lst` in order, replacing each leaf value.
  - If `lst` is too short, `StopIteration` will propagate.
- `Transform(fn)`:
  - Applies `fn(value)` to each leaf; preserves structure.
- `IsCompatible(other)`:
  - Flattens **keys only** for self and other and compares for equality.
- `Filter(fn)`:
  - Keeps entries for which `fn(entry)` is True (delegates to `FilterKeyVal`).
- `FilterKeyVal(fn)`:
  - Applies `fn(key, value)`.
  - Uses `_DELETE` to prune entries; can delete entire subtrees.
- `_ToStrings()`:
  - Builds a list of strings `"key<spaces>value"` with padding based on max key length.
  - Uses 4 spaces of padding after the longest key.
  - Returns lexicographically sorted list of strings.
- `DebugString()` returns `'\n'.join(_ToStrings())`.
- `VLog(level=None, prefix=None)`:
  - Defaults `level=0`, `prefix='nmap: '`.
  - Logs each `_ToStrings()` line via `tf.logging.vlog(level, '%s %s', prefix, l)`.
  - Note: `tf` is **not imported** in this module; if `VLog` is called without
    external injection of `tf`, it will raise `NameError`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/src/py_utils.rs`.
- Rust public API surface: `NestedMap` with typed key validation and traversal helpers.

**Implementation Steps (Detailed)**
1. Implement NestedMap with key validation and attribute-like access (maybe via methods).
2. Provide flatten/pack utilities preserving order (sorted keys).
3. Implement filter logic with delete sentinel behavior.

**Tests (Detailed)**
- Python tests: indirect via core layers.
- Rust tests: unit tests for key validation, Get/Set, Flatten/Pack compatibility.

**Gaps / Notes**
- Python uses dynamic attributes; Rust should expose explicit methods.
- `_RecursiveMap` only descends into `NestedMap` + `list`; any dict/tuple stored
  as a value is treated as a leaf (parity requires that behavior).

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

### `monolith/core/testing_utils.py`
<a id="monolith-core-testing-utils-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 203
- Purpose/role: Test helper for Keras/BaseLayer compatibility; adapted from TF Keras testing_utils.
- Key symbols/classes/functions: `layer_test`.
- External dependencies: TF internal Keras/test utilities.
- Side effects: builds models, runs predict, validates shapes/dtypes.

**Required Behavior (Detailed)**
- Decorator: `@test_util.disable_cudnn_autotune` wraps `layer_test`.
- `layer_test(...)`:
  - If `input_data is None`:
    - Requires `input_shape` or raises `ValueError('input_shape is None')`.
    - Default `input_dtype = 'float32'`.
    - Replaces `None` dimensions in `input_shape` with random ints in `[1, 3]`.
    - Creates `input_data = 10 * np.random.random(input_data_shape)`.
    - If dtype starts with `'float'`, subtracts `0.5`.
    - Casts to `input_dtype`.
  - If `input_data` provided and `input_shape is None`, sets `input_shape = input_data.shape`.
  - If `input_dtype is None`, sets to `input_data.dtype`.
  - If `expected_output_dtype is None`, sets to `input_dtype`.
  - Selects assertion:
    - If `expected_output_dtype` is `string` (`dtypes.as_dtype(...) == dtypes.string`):
      - If `test_harness` provided: uses `test_harness.assertAllEqual`.
      - Else: uses `string_test` (not defined in this file; assumed TF test util).
    - Else (numeric):
      - If `test_harness` provided: uses `test_harness.assertAllClose`.
      - Else: uses `tensorflow.python.keras.testing_utils.numeric_test`.
  - Instantiation: `kwargs = kwargs or {}` then `layer = layer_cls(**kwargs)`.
  - If `adapt_data` provided: `layer.adapt(adapt_data)` is called.
  - Weights round-trip:
    - `weights = layer.get_weights()`, `layer.set_weights(weights)`.
    - If `'weights'` is in `layer_cls.__init__` signature:
      - Adds `weights` to `kwargs` and re-instantiates `layer = layer_cls(**kwargs)`.
  - Functional API:
    - `x = layers.Input(shape=input_shape[1:], dtype=input_dtype)` (drops batch dim).
    - `y = layer(x)`.
    - If `backend.dtype(y) != expected_output_dtype`, raises `AssertionError`
      including layer name, input, actual dtype, expected dtype, kwargs.
  - Output shape check (if `expected_output_shape` provided):
    - Uses helper `assert_shapes_equal`:
      - Checks rank equality.
      - For each dim, if `tensor_shape.Dimension`, uses `.value`.
      - If expected is not None and differs, raises `AssertionError`.
    - Compares `tensor_shape.TensorShape(expected_output_shape)` vs `y.shape`.
  - Shape inference checks:
    - `model = models.Model(x, y)`.
    - `computed_output_shape = tuple(layer.compute_output_shape(TensorShape(input_shape)).as_list())`.
    - `computed_output_signature = layer.compute_output_signature(TensorSpec(shape=input_shape, dtype=input_dtype))`.
    - `actual_output = model.predict(input_data)`.
    - Asserts `computed_output_shape` == `actual_output.shape`.
    - Asserts `computed_output_signature.shape` == `actual_output.shape`.
    - If `computed_output_signature.dtype != actual_output.dtype`, raises `AssertionError`.
  - If `expected_output` provided: `assert_equal(actual_output, expected_output)`.
  - NOTE: `validate_training`, `custom_objects`, and `test_harness` params are mostly unused
    (only `test_harness` affects assertions).
  - NOTE: Despite docstring, the function **does not return** `actual_output`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/tests/testing_utils.rs`.
- Rust public API surface: test helper for layer validation (if Rust layers exist).

**Implementation Steps (Detailed)**
1. Implement a minimal test harness for Rust layers with shape/dtype checks.
2. If Rust uses different layer API, map semantics accordingly.

**Tests (Detailed)**
- Python tests: used by layer tests in core.
- Rust tests: provide helper functions for layer unit tests.

**Gaps / Notes**
- TF internal APIs used here are not available in Rust; might require simplified harness.
- `string_test` is referenced but not defined in this file (relies on TF test utils).
- `validate_training` is unused in this implementation.

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

### `monolith/core/tpu_variable.py`
<a id="monolith-core-tpu-variable-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 214
- Purpose/role: ReplicatedVariable wrapper for TPU context handling and resource variable ops.
- Key symbols/classes/functions: `ReplicatedVariable`, `_enclosing_tpu_context`, `_tensor_conversion`.
- External dependencies: TF internal ops.
- Side effects: registers tensor conversion function globally.

**Required Behavior (Detailed)**
- TF version detection:
  - Tries `from tensorflow.python.types import core`; if import succeeds `TF_23=True`.
  - If `TF_23`: `VariableBase = core.Tensor`, else `VariableBase = object`.
- `_handle_graph(handle)` context manager:
  - Enters `handle.graph.as_default()` to run assign ops in the handle’s graph.
- `_enclosing_tpu_context()`:
  - Starts at `ops.get_default_graph()._get_control_flow_context()`.
  - Walks `outer_context` until `control_flow_ops.XLAControlFlowContext` or `None`.
- `ReplicatedVariable(name, variables)`:
  - Stores `_name`, `_vars`, `_primary_var = variables[0]`, `_cached_value = None`,
    `_dtype = variables[0].dtype`.
  - `handle` property:
    - If no TPU context: returns `_primary_var.handle`.
    - Else: `tpu_context.get_replicated_var_handle(self._name, self._vars)`.
  - `_assign_dependencies()`:
    - If `_cached_value` is not None, wraps ops in `control_dependencies([_cached_value])`.
  - `initializer`: `control_flow_ops.group([v.initializer for v in self._vars])`.
  - `graph`: `_primary_var.graph`.
  - `_shared_name`: returns `_common_name` (attribute never defined here).
  - `_unique_id`: delegates to `_primary_var._unique_id` (protected access).
  - `name`, `dtype`, `shape`, `get_shape`, `to_proto` delegate to primary var.
  - `constraint`: always `None`.
  - `op`: `self.get().op`.
  - `_read_variable_op()`:
    - If no TPU context: `self._primary_var.read_value()`.
    - Else: `gen_resource_variable_ops.read_variable_op(self.handle, self._dtype)`.
  - `read_value()` returns `_read_variable_op()`.
  - `assign(value, use_locking=None, name=None, read_value=False)`:
    - Ignores `use_locking`.
    - Converts value to tensor with `dtype=self.dtype`.
    - Uses `assign_variable_op(self.handle, value_tensor)`.
    - If `read_value` True returns `_read_variable_op()`, else returns assign op.
  - `assign_add(delta, ..., read_value=True)` / `assign_sub(...)`:
    - Same pattern using `assign_add_variable_op` / `assign_sub_variable_op`.
    - Defaults `read_value=True`.
  - `get()` returns `_primary_var`.
  - `_in_graph_mode` delegates to `_primary_var._in_graph_mode`.
  - `_should_act_as_resource_variable()` is a no-op `pass`.
  - `_dense_var_to_tensor(dtype=None, name=None, as_ref=False)`:
    - If no TPU context:
      - If `_primary_var` has `_dense_var_to_tensor`, call it.
      - Else `ops.convert_to_tensor(_primary_var)`.
    - If dtype is not None and differs from `self.dtype`, returns `NotImplemented`.
    - If `as_ref` True: return `self.handle`; else return `self.read_value()`.
- Tensor conversion registration:
  - `_tensor_conversion` calls `var._dense_var_to_tensor(...)`.
  - `ops.register_tensor_conversion_function(ReplicatedVariable, _tensor_conversion)`.
  - If `not TF_23`: `ops.register_dense_tensor_like_type(ReplicatedVariable)`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-tf/src/tpu_variable.rs`.
- Rust public API surface: replicated variable wrapper if TF TPU is supported.

**Implementation Steps (Detailed)**
1. Provide wrapper around per-replica variables with TPU context awareness.
2. Implement assign/read operations using TF runtime or stub.
3. Ensure tensor conversion registration or equivalent behavior in Rust.

**Tests (Detailed)**
- Python tests: none.
- Rust tests: optional integration tests if TPU runtime is available.

**Gaps / Notes**
- TF TPU context is highly specific; Rust likely needs a bridge or stubs.
- `_shared_name` references `_common_name` which is never set in this class.
- `_cached_value` is never assigned within this module; only affects ordering if set externally.

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

### `monolith/core/util.py`
<a id="monolith-core-util-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 269
- Purpose/role: GCS utilities, checkpoint reload helpers, and dataset date filtering.
- Key symbols/classes/functions: `download_gcs_file`, `update_params`, `range_dateset`.
- External dependencies: `google.cloud.storage`, `tensorflow.compat.v1`, `gsutil` via subprocess.
- Side effects: network calls to GCS, subprocess execution, logging.

**Required Behavior (Detailed)**
- Constants:
  - `_GS_PREFIX = "gs://"`; `_CORE_NUMBER_PER_HOST = 8`.
  - `_DATE_FORMAT_LEN = 8`, `_MIN_DATE = "00000000"`, `_MAX_DATE = "99999999"`.
- `get_bucket_name_and_relavite_path(gs_file_path)`:
  - Asserts input contains `_GS_PREFIX`.
  - Parses bucket between `gs://` and first `/`.
  - Returns `(bucket_name, relative_path_after_bucket)`.
- `download_gcs_file(gs_file_path, local_file_name)`:
  - Logs start; calls `get_bucket_name_and_relavite_path`.
  - Delegates to `download_gcs_file_with_relative_path`.
- `download_gcs_file_with_relative_path(bucket, relative_path, local_file_name)`:
  - Uses `google.cloud.storage.Client()`.
  - `bucket = storage_client.bucket(bucket_name)`.
  - `blob = bucket.blob(relative_path)` then `blob.download_to_filename(local_file_name)`.
- `list_gcs_files_with_prefix(gs_path_prefix)`:
  - Uses storage client + `list_blobs(bucket_name, prefix=relative_path_prefix)`.
  - Returns `(bucket_name, blob_iterator)`.
- `parse_example_number_meta_file(meta_file, seperator)`:
  - Reads all lines, ignores any line without a comma.
  - Splits on **comma** (the `seperator` arg is unused).
  - Enforces lexicographic ascending `file_name` via `assert previous_file_name < file_name`.
  - Parses `count = int(split_str[1])` and appends `(file_name, count)` list.
- `calculate_shard_skip_file_number(file_example_number, shard_num, completed_steps_number, batch_size_per_core)`:
  - `processed_example_number_per_host = batch_size_per_core * completed_steps_number * _CORE_NUMBER_PER_HOST`.
  - Iterates counts in round-robin shard order (`shard_index = (shard_index + 1) % shard_num`).
  - If `example_number + shard_accumulated_example_count[shard_index] <= processed_example_number_per_host`:
    - Accumulates count and increments `shard_skip_file_number` for that shard.
  - Returns `shard_skip_file_number` list length `shard_num`.
- `get_checkpoint_completed_step_number(checkpoint_path)`:
  - Lists blobs with prefix `path.join(checkpoint_path, "model.ckpt")`.
  - Considers only `*.meta` files.
  - Extracts step from blob name between `"-"` and `".meta"`.
  - Returns max step (0 if none).
- `update_params(params, tpu_cluster_resolver)`:
  - `shard_num = tpu_cluster_resolver.cluster_spec().num_tasks("worker")`.
  - Requires either `batch_size_per_core` or `global_batch_size` not None.
  - If only `global_batch_size`: sets `batch_size_per_core = global_batch_size / shard_num / _CORE_NUMBER_PER_HOST`.
  - If only `batch_size_per_core`: sets `global_batch_size = batch_size_per_core * shard_num * _CORE_NUMBER_PER_HOST`.
  - If both: asserts equality.
  - Logs batch sizes.
  - Calls `get_checkpoint_completed_step_number(params["model_dir"])`.
  - If `completed_step_number > 0`:
    - Calls `get_per_file_example_numbers_for_checkpoint_reload(...)`.
    - Computes `shard_skip_file_number` and stores in `params["shard_skip_file_number"]`.
    - Logs the computed list.
  - NOTE: uses Python `/` for division, so `batch_size_per_core` may be float.
- `get_per_file_example_numbers_for_checkpoint_reload(train_dataset_path, file_example_number_meta, seperator)`:
  - Runs `gsutil ls train_dataset_path` via `subprocess.Popen`.
  - Reads stdout lines; for each:
    - Decodes UTF-8, strips newline.
    - Uses `get_bucket_name_and_relavite_path` to get `relative_path`.
    - Enforces lexicographic ascending relative path.
  - Loads `file_example_number_list` via `parse_example_number_meta_file(...)`.
  - Asserts `train_file_path_list` non-empty.
  - Finds first index in meta list where `train_file_path_list[0] <= file_path` (lexicographic).
  - Asserts remaining meta length can cover training list.
  - Iterates training list:
    - Asserts each train file matches meta file name at current index.
    - Appends the corresponding count to `example_number_list`.
  - Logs completion and returns `example_number_list` (list of counts).
- `range_dateset(dataset, root_path, start_date=None, end_date=None)`:
  - Defaults `start_date` to `_MIN_DATE`, `end_date` to `_MAX_DATE`.
  - Logs start/end.
  - `filter_fn(x)`:
    - `path_prefix_len = len(root_path)`.
    - Extracts `date_str = tf.strings.substr(x, path_prefix_len, _DATE_FORMAT_LEN)`.
    - Converts to `int32` and compares with `start_date`/`end_date` (inclusive).
  - Returns `dataset.filter(filter_fn)`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/src/util.rs`.
- Rust public API surface: GCS helpers and dataset date filtering.

**Implementation Steps (Detailed)**
1. Implement GCS helpers (use Google Cloud Storage SDK or `gsutil` fallback).
2. Port checkpoint reload logic and shard skip calculations.
3. Implement dataset date range filter (likely for TF dataset bridge).

**Tests (Detailed)**
- Python tests: `monolith/core/util_test.py`
- Rust tests: replicate range_dateset filtering behavior.

**Gaps / Notes**
- `gsutil` subprocess usage may need replacement or explicit dependency in Rust.

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

### `monolith/core/util_test.py`
<a id="monolith-core-util-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 149
- Purpose/role: Tests `range_dateset` date filtering behavior.
- Key symbols/classes/functions: `UtilTest.test_range_dataset_*`.
- External dependencies: `tensorflow.compat.v1`.
- Side effects: none.

**Required Behavior (Detailed)**
- Uses TF1 graph/session APIs (`tf.compat.v1`, `self.session()`).
- `root_path = "gs://test_folder/unzipped_tf_records_corrected_repartitioned/"`.
- `test_range_dataset_single`:
  - Input dataset has dates 20200501, 20200502, 20200503 (single part each).
  - Filters start/end = "20200502".
  - Expects only the 20200502 item.
- `test_range_dataset_multiple`:
  - Input dataset includes 20200501, 20200502, 20200503 (two parts), 20200504.
  - Filters start="20200502", end="20200503".
  - Expects 20200502 and both 20200503 parts.
- `test_range_dataset_out_of_boundary`:
  - Input dataset contains 20200501, 20200502.
  - Filters start="20200401", end="20200505".
  - Expects both items (range fully covers).
- `test_range_dataset_no_start_date`:
  - Filters with `start_date=None`, `end_date="20200505"`.
  - Expects all items (uses `_MIN_DATE`).
- `test_range_dataset_no_end_date`:
  - Filters with `start_date="20200502"`, `end_date=None`.
  - Expects only 20200502 (uses `_MAX_DATE`).
- Each test:
  - Uses `tf.compat.v1.data.make_one_shot_iterator(dataset)`.
  - Iterates `next_element` in a `try` loop; catches `tf.errors.OutOfRangeError`.
  - Asserts each output equals expected list; verifies count matches.
- `__main__` path disables eager execution then `tf.test.main()`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/tests/util.rs`.
- Rust public API surface: `range_dateset` equivalent.

**Implementation Steps (Detailed)**
1. Port tests using a mock dataset list and filter function.
2. Ensure output order and content match Python.

**Tests (Detailed)**
- Python tests: this file
- Rust tests: same scenarios.

**Gaps / Notes**
- Requires a dataset abstraction; may be tested with plain lists in Rust.

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

### `monolith/core/variance_scaling.py`
<a id="monolith-core-variance-scaling-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 188
- Purpose/role: Numpy-based variance scaling initializer (truncated/untruncated normal or uniform).
- Key symbols/classes/functions: `_compute_fans`, `VarianceScaling.__call__`, `get_config`.
- External dependencies: `numpy`, `scipy.stats.truncnorm`.
- Side effects: uses NumPy RNG (seeded per call).

**Required Behavior (Detailed)**
- `_compute_fans(shape, data_format='channels_last')`:
  - `len(shape)==2`: `fan_in = shape[0]`, `fan_out = shape[1]`.
  - `len(shape) in {3,4,5}` (conv kernels):
    - If `channels_first`:
      - `receptive_field_size = np.prod(shape[2:])`.
      - `fan_in = shape[1] * receptive_field_size`.
      - `fan_out = shape[0] * receptive_field_size`.
    - If `channels_last`:
      - `receptive_field_size = np.prod(shape[:-2])`.
      - `fan_in = shape[-2] * receptive_field_size`.
      - `fan_out = shape[-1] * receptive_field_size`.
    - Else raises `ValueError('Invalid data_format: ' + data_format)`.
  - Else: `fan_in = fan_out = np.sqrt(np.prod(shape))`.
- `VarianceScaling.__init__(scale=1.0, mode='fan_in', distribution='truncated_normal', seed=None)`:
  - Validates `scale > 0` else `ValueError`.
  - `mode = mode.lower()` and must be in `{'fan_in','fan_out','fan_avg'}` else `ValueError`.
  - `distribution = distribution.lower()` and must be in
    `{'truncated_normal','untruncated_normal','uniform'}` else `ValueError`.
  - Stores `scale/mode/distribution/seed`.
- `__call__(shape, dtype=np.float32)`:
  - Computes `fan_in, fan_out = _compute_fans(shape)`.
  - Adjusts `scale`:
    - `fan_in`: `scale /= max(1., fan_in)`.
    - `fan_out`: `scale /= max(1., fan_out)`.
    - `fan_avg`: `scale /= max(1., float(fan_in + fan_out) / 2)`.
  - Seeds NumPy RNG with `np.random.seed(self.seed)` on every call.
  - `distribution == 'truncated_normal'`:
    - `mean = 0.0`.
    - `stddev = sqrt(scale) / 0.87962566103423978` (constant from truncnorm std).
    - Clips at ±2*stddev; computes `a`, `b` for `stats.truncnorm`.
    - Returns `stats.truncnorm.rvs(..., size=shape).astype(dtype)`.
  - `distribution == 'untruncated_normal'`:
    - Uses `np.random.normal(loc=0, scale=sqrt(scale), size=shape)`.
    - **Always** casts to `'float32'` (ignores `dtype` parameter).
  - `distribution == 'uniform'`:
    - `limit = sqrt(3. * scale)`.
    - `np.random.uniform(low=-limit, high=limit, size=shape).astype(dtype)`.
- `get_config()` returns dict with `scale`, `mode`, `distribution`, `seed`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/src/variance_scaling.rs`.
- Rust public API surface: `VarianceScaling` struct implementing initializer to produce arrays.

**Implementation Steps (Detailed)**
1. Port `_compute_fans` logic and mode handling.
2. Implement truncated normal sampling (use rand_distr or custom truncation) matching SciPy behavior.
3. Ensure seeding behavior matches (seed per call).

**Tests (Detailed)**
- Python tests: none.
- Rust tests: sample shape and distribution sanity checks; seed determinism.

**Gaps / Notes**
- Matching SciPy truncnorm exactly may require careful implementation.
- `untruncated_normal` ignores `dtype` and forces `'float32'`; parity should preserve.

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

### `monolith/gpu_runner.py`
<a id="monolith-gpu-runner-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 226
- Purpose/role: GPU training/eval runner for TF Estimator, with optional Horovod multi-GPU support.
- Key symbols/classes/functions: `GPURunner`, `create_estimator`, `run`, CLI `main`.
- External dependencies: `tensorflow`, `horovod`, `mpi4py`, `absl.flags`, `model_registry`.
- Side effects: initializes Horovod, configures GPU visibility, trains/evaluates model, writes summaries.

**Required Behavior (Detailed)**
- CLI flags (absl):
  - `task` (string, required by caller).
  - `model_dir` (string, model + summaries).
  - `save_checkpoints_steps` (int, None means no checkpoints).
  - `mode` enum: `"train_and_eval" | "train" | "eval"` (default `"train"`).
- `GPURunner.__init__(task_param, ...)`:
  - Calls `BaseRunner.__init__`.
  - Reads flags into `_model_dir`, `_save_checkpoints_steps`, `_mode`.
  - Stores `_task_param`.
- `create_estimator(model_fn)`:
  - If `self._task_param.accelerator == "horovod"`:
    - `model_dir = self._model_dir` (comment notes same dir for all ranks).
    - `save_checkpoints_steps = self._save_checkpoints_steps` **only if** `hvd.rank() == 0`, else `None`.
    - Builds `tf.compat.v1.ConfigProto()`:
      - `config.graph_options.optimizer_options.global_jit_level = tf.compat.v1.OptimizerOptions.ON_1`.
      - `config.gpu_options.allow_growth = True`.
      - `config.gpu_options.visible_device_list = str(hvd.local_rank())`.
    - Wraps in `tf.estimator.RunConfig(model_dir=..., save_checkpoints_steps=..., session_config=config)`.
    - Sets `num_gpus = hvd.size()`.
  - Else:
    - `num_gpus = 1`.
    - Uses `tf.compat.v1.estimator.RunConfig(model_dir=..., save_checkpoints_steps=...)`.
  - Returns `tf.compat.v1.estimator.Estimator` with `params`:
    - `"train_batch_size"`: `task_param.train.per_replica_batch_size`.
    - `"eval_batch_size"`: `task_param.eval.per_replica_batch_size`.
    - `"accelerator"`: `task_param.accelerator`.
    - `"num_replicas"`: `num_gpus`.
    - `"hvd_rank"`: `hvd.rank()` if horovod else `0`.
- `run()`:
  - Loads `current_step` via `tf.train.load_variable(model_dir, GLOBAL_STEP)`;
    on `TypeError`, `ValueError`, or `tf.errors.NotFoundError`, uses `0`.
  - Instantiates task: `task = task_param.instantiate()`.
  - Creates `input_fn_train`, `input_fn_eval`, and `model_fn`.
  - If horovod:
    - `hvd.init()`.
    - Lists GPUs via `tf.config.experimental.list_physical_devices('GPU')`.
    - For each GPU:
      - `set_memory_growth(gpu, True)`.
      - `set_visible_devices(gpus[hvd.local_rank()], 'GPU')` (inside loop).
  - Creates estimator and starts timer (`start_timestamp = time.time()`).
  - `mode == 'train'`:
    - If horovod: uses `hvd.BroadcastGlobalVariablesHook(0)` and passes to `est.train`.
    - Else: `est.train(input_fn_train, max_steps=task_param.train.max_steps)`.
  - `mode == 'eval'`:
    - `eval_output_dir = os.path.join(model_dir, 'eval')`, `tf.io.gfile.makedirs`.
    - `total_examples = task_param.input.eval_examples`.
    - `eval_batch_size = task_param.eval.per_replica_batch_size`.
    - `num_steps = total_examples // eval_batch_size` (floor).
    - `eval_results = est.evaluate(input_fn_eval, steps=num_steps)`.
    - Writes summaries via `tf.compat.v1.summary.FileWriter(eval_output_dir)` and
      `self.write_summary(eval_results, summary_writer, current_step)`.
  - `mode == 'train_and_eval'`:
    - `steps_per_eval = task_param.eval.steps_per_eval`, `max_steps = task_param.train.max_steps`.
    - Creates `eval_output_dir` and loops while `current_step < max_steps`:
      - `next_checkpoint = min(current_step + steps_per_eval, max_steps)`.
      - Trains to `next_checkpoint` (horovod uses broadcast hook).
      - Sets `current_step = next_checkpoint`.
      - Logs elapsed seconds since `start_timestamp`.
      - Computes `num_steps = total_examples // eval_batch_size`.
      - If not horovod **or** `hvd.rank() == 0`:
        - Logs "Starting to evaluate.", sleeps 10 seconds, then `est.evaluate`.
        - Writes eval summary with `current_step`.
      - If horovod: `MPI.COMM_WORLD.barrier()` (sync all workers).
    - After loop: logs total elapsed time as `int(time.time() - start_timestamp)`.
- `main(unused_argv)`:
  - Looks up task params via `model_registry.GetParams(FLAGS.task)`.
  - Logs task params, creates `GPURunner`, and calls `run()`.
- `__main__`:
  - `logging.set_verbosity(logging.INFO)`.
  - `tf.compat.v1.disable_v2_behavior()`.
  - `app.run(main)`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/gpu_runner.rs` (or CLI bin).
- Rust public API surface: runner struct + CLI entrypoint.
- Data model mapping: task params, estimator equivalent (likely still Python-backed or TensorFlow runtime binding).
- Feature gating: `tf-runtime` and `horovod` (if supported) features.

**Implementation Steps (Detailed)**
1. Recreate CLI flags and task registry lookup in Rust.
2. Decide how to execute training: native Rust pipeline or Python/TF bridge.
3. If bridging TF, maintain Horovod initialization and GPU pinning logic.
4. Mirror evaluation loop, summary writing, and MPI barrier behavior.

**Tests (Detailed)**
- Python tests: none specific.
- Rust tests: integration tests for CLI argument parsing and control flow (mocked task).

**Gaps / Notes**
- Full parity likely requires TF Estimator execution; consider keeping Python runner or embedding TF runtime.

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

### `monolith/native_training/alert/alert_manager.py`
<a id="monolith-native-training-alert-alert-manager-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 31
- Purpose/role: Placeholder alert manager module defining a flag and stub accessor.
- Key symbols/classes/functions: `get_default_alert_manager()`.
- External dependencies: `absl.flags`, `absl.logging`, `google.protobuf.text_format`, `threading`, `time`, `traceback`.
- Side effects: defines `--monolith_alert_proto` flag at import time.

**Required Behavior (Detailed)**
- Flag definition:
  - `monolith_alert_proto` (string), default `""`, description: "The text format of alert proto."
- `get_default_alert_manager()`:
  - Returns `None`.
- No other behavior implemented.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src` (if alerts are ported).
- Rust public API surface: optional alert manager accessor + config flag.
- Data model mapping: map CLI flag to config struct; no runtime behavior.
- Feature gating: none.
- Integration points: training runner or alert subsystem (not present).

**Implementation Steps (Detailed)**
1. Add a config flag or CLI option mirroring `monolith_alert_proto`.
2. Provide a stub `get_default_alert_manager` returning `None`/`Option::None`.

**Tests (Detailed)**
- Python tests: none.
- Rust tests: none required (flag parsing only).
- Cross-language parity test: not applicable.

**Gaps / Notes**
- Module is largely empty; functionality likely lives elsewhere (or removed).

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

### `monolith/native_training/alert/alert_manager_test.py`
<a id="monolith-native-training-alert-alert-manager-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 32
- Purpose/role: Placeholder test module; no actual tests defined.
- Key symbols/classes/functions: none (only `__main__` entrypoint).
- External dependencies: `absl.testing`, `absl.flags/app`, `google.protobuf.text_format`.
- Side effects: running as main executes `absltest.main()`.

**Required Behavior (Detailed)**
- No tests; `absltest.main()` called if executed.

**Rust Mapping (Detailed)**
- Target crate/module: none.
- Rust public API surface: none.
- Data model mapping: none.
- Feature gating: none.
- Integration points: none.

**Implementation Steps (Detailed)**
1. No Rust tests required unless alert manager gains functionality.

**Tests (Detailed)**
- Python tests: none.
- Rust tests: none.
- Cross-language parity test: not applicable.

**Gaps / Notes**
- No tests defined; consider removing or adding real coverage if alerting is implemented.

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

### `monolith/native_training/barrier_ops.py`
<a id="monolith-native-training-barrier-ops-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 158
- Purpose/role: TF-based barrier primitive to block non-chief workers until chief releases barrier; includes SessionRunHook integration.
- Key symbols/classes/functions: `BarrierOp`, `BarrierHook`, `BarrierAlreadyPlacedError`.
- External dependencies: `tensorflow`, `absl.logging`, `threading`, `time`, `basic_restore_hook` (imported, unused).
- Side effects: creates TF variables/placeholders, sleeps, logs every 60 seconds.

**Required Behavior (Detailed)**
- `BarrierAlreadyPlacedError`: raised when attempting to place a barrier twice.
- `BarrierOp(capacity, is_chief=True, wait_seconds=1, name_prefix="default", barrier_callbacks=None)`:
  - Creates `barrier_var` bool vector of length `capacity`:
    - If `is_chief`: `LOCAL_VARIABLES`; else `VARIABLES`.
  - Creates index placeholder `_idx_ph`.
  - `_place_op` sets `barrier_var[idx] = True`; `_remove_op` sets `False`.
  - `barrier_placed_tensor` references element 0.
  - `barrier_op_action` string variable + placeholder, assign op.
  - Uses a threading lock for thread safety.
- `place_barrier(session, action="")`:
  - If barrier already placed (index 0 True), raises `BarrierAlreadyPlacedError`.
  - Sets barrier at index 0 and assigns action; runs callbacks `(action, session)`.
- `remove_barrier(session)`:
  - Clears barrier at index 0; no checks.
- `is_barrier_placed(session)`:
  - Returns `session.run(barrier_placed_tensor)`.
- `wait_until_barrier_removed(session, index)`:
  - Validates `index` in `(0, capacity)` else `ValueError`.
  - Sets barrier at `index`, reads action, runs callbacks.
  - Loops until `barrier_placed_tensor` becomes False, sleeping `wait_seconds` and logging every 60s.
  - Removes its own barrier index after release.
- `is_all_blocked`/`is_none_blocked`:
  - Reads `barrier_var` and checks count equals capacity or 0.
- `get_unblocked_indices`/`get_blocked_indices`:
  - Returns indices with False/True in barrier vector.
- `BarrierHook(index, barrier_op)`:
  - `before_run` requests `barrier_placed_tensor`.
  - `after_run`: if `index > 0` and barrier placed, calls `wait_until_barrier_removed`.
- Threading: lock guards state changes; waiting uses sleep polling.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/barrier.rs`.
- Rust public API surface: `Barrier` trait, `InProcessBarrier`, `PsBarrier` (async).
- Data model mapping: TF variable barrier → async barrier/coordination via PS; no SessionRunHook equivalent.
- Feature gating: none (used in distributed training).
- Integration points: `runner.rs`, `distributed_ps.rs`.

**Implementation Steps (Detailed)**
1. Decide whether a TF-variable barrier is needed in Rust (likely no).
2. Map barrier semantics to async barrier APIs (waiting, timeouts).
3. Provide hook-like integration if training loop expects per-step blocking.

**Tests (Detailed)**
- Python tests: `monolith/native_training/barrier_ops_test.py`.
- Rust tests: add async barrier tests for arrival/release semantics.
- Cross-language parity test: not applicable unless TF runtime added.

**Gaps / Notes**
- TF `basic_restore_hook` import unused.
- Python barrier uses polling on tensor 0; Rust barrier is event-based and may not mirror per-step hook semantics.

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

### `monolith/native_training/barrier_ops_test.py`
<a id="monolith-native-training-barrier-ops-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 104
- Purpose/role: Verifies BarrierOp placement/removal and BarrierHook blocking behavior.
- Key symbols/classes/functions: `BarrierOpsTest`.
- External dependencies: `tensorflow`, `monitored_session._HookedSession`.
- Side effects: spawns threads and TF sessions.

**Required Behavior (Detailed)**
- `test_basic`:
  - Place barrier; double place raises `BarrierAlreadyPlacedError`.
  - Remove barrier → `is_barrier_removed` true.
- `test_barrier_hook_not_blocked`:
  - Without barrier, hook does not block; global step reaches 5.
- `test_barrier_hook_blocked`:
  - Place barrier; worker thread blocks after 1 step.
  - Callback called with action, sets a variable to True.
  - Removing barrier allows training to finish; all barriers cleared.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/barrier.rs`.
- Rust public API surface: barrier trait/implementations.
- Data model mapping: no SessionRunHook analogue; tests should exercise async barrier directly.
- Feature gating: none.
- Integration points: training runner.

**Implementation Steps (Detailed)**
1. Add Rust tests for arrival/release semantics using in-process barrier.
2. Add integration test for PS barrier timeout if applicable.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: add async barrier tests for arrival/release semantics.
- Cross-language parity test: not applicable.

**Gaps / Notes**
- Rust barrier is async/event-based; no direct SessionRunHook equivalent.

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

### `monolith/native_training/basic_restore_hook.py`
<a id="monolith-native-training-basic-restore-hook-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 72
- Purpose/role: SessionRunHook that invokes listener callbacks around checkpoint restore (no actual restore logic).
- Key symbols/classes/functions: `CheckpointRestorerListener`, `CheckpointRestorerHook`.
- External dependencies: `tensorflow.python.training.session_run_hook`, `absl.logging`.
- Side effects: logs on creation and restore calls.

**Required Behavior (Detailed)**
- `CheckpointRestorerListener`:
  - Interface with `begin`, `before_restore(session)`, `after_restore(session)`, `end(session)`; all no-ops by default.
- `CheckpointRestorerHook(listeners=None)`:
  - Logs "Create CheckpointRestorerHook."
  - `begin()` calls `listener.begin()` for each listener.
  - `after_create_session(session, coord)` calls `_restore(session)`.
  - `_restore(session)`:
    - Logs "Calling checkpoint restorer listeners."
    - Calls `before_restore(session)` then `after_restore(session)` on each listener.
    - **No actual restore actions performed in hook**.
- No `end()` implementation in hook; listener `end()` is never invoked.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/hooks.rs`.
- Rust public API surface: `Hook` trait + `CheckpointHook` (save only).
- Data model mapping: listener callbacks map to hook lifecycle events.
- Feature gating: none.
- Integration points: training loop hook list.

**Implementation Steps (Detailed)**
1. Add a Rust hook that runs listener callbacks at session start.
2. If restore is implemented in Rust, wire callbacks before/after restore.
3. Ensure lifecycle ordering matches Python (begin → after_create_session → before/after restore).

**Tests (Detailed)**
- Python tests: `monolith/native_training/basic_restore_hook_test.py`.
- Rust tests: add hook lifecycle test in `monolith-training`.
- Cross-language parity test: compare callback ordering under identical steps.

**Gaps / Notes**
- Hook does not implement `end()`, so listener `end()` is never called.

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

### `monolith/native_training/basic_restore_hook_test.py`
<a id="monolith-native-training-basic-restore-hook-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 137
- Purpose/role: Verifies listener callbacks fire during `after_create_session` and do not repeat during training.
- Key symbols/classes/functions: `CountCheckpointRestorerListener`, `CountHook`, `CheckpointRestorerHookTest`.
- External dependencies: `tensorflow`, `session_run_hook`.
- Side effects: runs TF monitored sessions.

**Required Behavior (Detailed)**
- `test_restore_only_in_after_create_session`:
  - Listener `begin/before_restore/after_restore` called once at session creation.
  - Another hook receives `after_create_session` once before any `before_run/after_run`.
  - After training steps, listener counts remain unchanged; CountHook shows before_run/after_run/end increments.
- `test_two_listeners_with_restorer`:
  - Two listeners both receive begin/before/after restore once.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/hooks.rs`.
- Rust public API surface: hook lifecycle tests.
- Data model mapping: count callbacks during `on_start` or equivalent.
- Feature gating: none.
- Integration points: hook list execution order.

**Implementation Steps (Detailed)**
1. Add a Rust test that runs a hook list and validates callback order/counts.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: add CPU tensor tests for clipping behavior and immutability.
- Cross-language parity test: compare Rust outputs vs TF for fixed inputs.

**Gaps / Notes**
- GPU-only custom ops in Python; Rust needs equivalent or fallback path.

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

### `monolith/native_training/clip_ops.py`
<a id="monolith-native-training-clip-ops-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 80
- Purpose/role: Custom gradient clipping with Monolith GPU ops and optional fused path.
- Key symbols/classes/functions: `_global_norm`, `clip_by_global_norm`.
- External dependencies: `tensorflow`, `device_utils`, `gen_monolith_ops` (custom ops).
- Side effects: none (pure tensor ops), but relies on custom TF ops.

**Required Behavior (Detailed)**
- `_global_norm(t_list)`:
  - Returns `None` if list empty.
  - Uses `gen_monolith_ops.global_l2_reduce` to compute L2 sum, then `sqrt`.
- `clip_by_global_norm(t_list, clip_norm, use_norm=None)`:
  - Requires `t_list` to be a list; else raises `TypeError("t_list should be a list")`.
  - If `t_list` empty, returns `(t_list, 0)`.
  - If `use_norm` provided: returns `monolith_clip_by_global_norm(t_list, use_norm, clip_norm)` and `use_norm`.
  - If in GPU placement context: uses fused op `monolith_clip_by_global_norm_fused(t_list, clip_norm)` (returns list, norm).
  - Otherwise computes `global_norm` via:
    - `_global_norm` if GPU context, else `tf.linalg.global_norm`.
  - Applies `monolith_clip_by_global_norm(t_list, global_norm, clip_norm)` and returns `(list_clipped, global_norm)`.
- Expected semantics match `tf.clip_by_global_norm` including NaN on inf norms.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-optimizer/src` (or `monolith-tensor`).
- Rust public API surface: clip-by-global-norm helper in optimizer utilities.
- Data model mapping: custom TF ops → native tensor ops (Candle/TF runtime).
- Feature gating: GPU vs CPU paths should be explicit.
- Integration points: optimizer step and gradient pre-processing.

**Implementation Steps (Detailed)**
1. Implement global norm computation and clipping on tensor lists.
2. Provide GPU-optimized path or document as CPU-only if missing.
3. Ensure NaN propagation on inf norms matches TF.
4. Add compatibility test vs `tf.clip_by_global_norm`.

**Tests (Detailed)**
- Python tests: `monolith/native_training/clip_ops_test.py`.
- Rust tests: add numeric tests for clip_by_global_norm including NaN/inf cases.
- Cross-language parity test: compare outputs against TF for fixed inputs.

**Gaps / Notes**
- Custom ops (`monolith_clip_by_global_norm_*`) are not available in Rust.

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

### `monolith/native_training/clip_ops_test.py`
<a id="monolith-native-training-clip-ops-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 92
- Purpose/role: Validates custom clip-by-global-norm against TF behavior and input immutability.
- Key symbols/classes/functions: `ClipOpsTest`, `NormOpsTest`.
- External dependencies: `tensorflow`, `numpy`, `test_util`.
- Side effects: runs GPU-only tests if available.

**Required Behavior (Detailed)**
- `ClipOpsTest._test_clip_by_global_norm`:
  - Runs op on GPU; compares output to TF `clip_by_global_norm` (unless expected provided).
  - Asserts input tensors are not modified in-place.
- `test_clip_by_global_norm`:
  - Covers simple, uneven shapes, no clipping, zero norm, inf -> NaN, and large random grads.
- `NormOpsTest`:
  - `test_it` (GPU only) checks `_global_norm` for inf and known values.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-optimizer/tests`.
- Rust public API surface: clip-by-global-norm tests.
- Data model mapping: use CPU tensors unless GPU backend exists.
- Feature gating: GPU-only tests optional.
- Integration points: optimizer utilities.

**Implementation Steps (Detailed)**
1. Add CPU tests matching the Python expected outputs.
2. Add NaN/inf propagation test.
3. Add randomized large-shape test if memory allows.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: add temp-dir PS file round-trip test.
- Cross-language parity test: not applicable.

**Gaps / Notes**
- Rust uses different discovery stack; PS file persistence may not be needed.

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

### `monolith/native_training/cluster_manager.py`
<a id="monolith-native-training-cluster-manager-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 184
- Purpose/role: Build TF cluster specs for distributed training and manage PS discovery persistence.
- Key symbols/classes/functions: `generate_session_config`, `get_training_cluster`, `_query_ps_cluster`, `_query_chief_addr`, `_save_ps_cluster_to_file`, `_fetch_ps_cluster_from_file`.
- External dependencies: `tensorflow`, `absl.logging`, `ServiceDiscovery`, `metric.cli`.
- Side effects: reads/writes PS cluster file via `tf.io.gfile`, sleeps on retries, emits metrics.

**Required Behavior (Detailed)**
- `emit_store(name, value, tagkv=None)`:
  - Delegates to `_MCLI.emit_store` (metrics).
- `generate_session_config(cluster_and_task=None)`:
  - If `None`, returns `ConfigProto(allow_soft_placement=True)`.
  - Else builds `ClusterSpec` and `ConfigProto(cluster_def=...)` with:
    - `device_filters` including `/job:ps` and `/job:chief`; non-chief adds its own job/task filter.
  - Sets `share_cluster_devices_in_session=True`.
  - Sets `experimental.share_session_state_in_clusterspec_propagation=True`.
  - Disables Grappler meta optimizer.
- `get_training_cluster(...)`:
  - If `index == 0` (chief):
    - If `num_redundant_ps`: try reading PS addrs from file (timeout=0); if insufficient, query discovery then save to file.
    - Else query discovery for PS addrs.
    - Builds cluster with `chief=[worker_addr]`, `worker=fake_worker_list`, `ps=ps_addrs`.
    - `task={"type":"chief","index":0}`.
  - Else (worker):
    - Gets `chief_addr` via `_query_chief_addr`.
    - Builds `fake_worker_list` of size `num_workers-1` and assigns `worker_addr` at `index-1`.
    - PS addrs from file (if redundant) or discovery.
    - `task={"type":"worker","index":index-1}`.
  - Asserts PS count equals `num_required_ps`.
- `_query_chief_addr(discovery)`:
  - Polls `discovery.query("worker")` until index 0 present; sleeps 5s between retries.
- `_query_ps_cluster(discovery, num_required_ps, model_name=None, cluster_type="stable")`:
  - Polls `discovery.query("ps")` until enough PS; logs count and emits metrics if model_name provided.
  - Returns sorted PS addresses by index, truncated to `num_required_ps`.
- `_save_ps_cluster_to_file(file_name, ps_addrs)`:
  - Writes comma-separated list to temp file and atomically renames.
- `_fetch_ps_cluster_from_file(file_name, timeout=1800)`:
  - Repeatedly attempts read; returns list if found, else empty after timeout.
- `_get_ps_cluster_file_name(model_dir, uuid)`:
  - Path: `<model_dir>/ps_cluster_dir/<uuid or "ps_info">`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/distributed.rs`, `py_discovery.rs`.
- Rust public API surface: `ClusterConfig`, `ServiceDiscovery` implementations.
- Data model mapping: TF `ClusterSpec` → Rust `ClusterConfig`; discovery polling → async discovery APIs.
- Feature gating: none.
- Integration points: distributed runner and service discovery.

**Implementation Steps (Detailed)**
1. Map discovery polling to Rust async discovery with retry/backoff.
2. Add PS cluster file persistence helpers (optional).
3. Provide cluster config builder mirroring fake worker list behavior if needed for TF_CONFIG parity.

**Tests (Detailed)**
- Python tests: `monolith/native_training/cluster_manager_test.py`.
- Rust tests: add unit test for PS file round-trip and cluster config validation.
- Cross-language parity test: compare generated cluster dictionaries for sample inputs.

**Gaps / Notes**
- Python uses fake worker list due to TF_CONFIG limitation; Rust cluster config may not need this.

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

### `monolith/native_training/cluster_manager_test.py`
<a id="monolith-native-training-cluster-manager-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 35
- Purpose/role: Tests PS cluster file persistence helpers.
- Key symbols/classes/functions: `ClusterManagerTest.testBasic`.
- External dependencies: `os`, `cluster_manager`.
- Side effects: writes temp files under `TEST_TMPDIR`.

**Required Behavior (Detailed)**
- `testBasic`:
  - Writes PS addrs to file via `_save_ps_cluster_to_file`.
  - Reads via `_fetch_ps_cluster_from_file` and asserts equality.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests`.
- Rust public API surface: PS cluster file helper tests.
- Data model mapping: same string list round-trip.
- Feature gating: none.
- Integration points: optional in discovery integration.

**Implementation Steps (Detailed)**
1. Add Rust test that writes/reads PS cluster file in a temp dir.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: add `cluster_manager_test` in `monolith-rs/crates/monolith-training/tests` that writes/reads a temp PS cluster file and asserts exact content.
- Cross-language parity test: run Python test to produce the file format and compare Rust read results against that output.

**Gaps / Notes**
- Ensure temp dir handling mirrors `TEST_TMPDIR` semantics used by TF tests.

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

### `monolith/native_training/consul.py`
<a id="monolith-native-training-consul-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 149
- Purpose/role: Minimal Consul client for service lookup/register/deregister using HTTP or Unix socket.
- Key symbols/classes/functions: `Client`, `UnixHTTPConnection`, `ConsulException`.
- External dependencies: `six.moves.http_client.HTTPConnection`, `socket`, `threading`, `json`, `os`.
- Side effects: spawns a health-check thread on register; caches lookup results.

**Required Behavior (Detailed)**
- `UnixHTTPConnection(path)`:
  - Connects via UNIX domain socket to Consul.
- `Client.__init__()`:
  - Determines consul host:
    - `CONSUL_HTTP_HOST` or `TCE_HOST_IP` env vars.
    - Else uses `/opt/tmp/sock/consul.sock` if file exists.
    - Else defaults to `"127.0.0.1"`.
  - Port from `CONSUL_HTTP_PORT` or `2280`.
  - Initializes `_cache` and `_lock`.
- `lookup(name, timeout=3, cachetime=0)`:
  - If `cachetime>0` and cached entry is fresh, returns it.
  - Else uses `_lookup`, with longer timeout (30s) on cache miss.
  - Caches result with timestamp.
- `_lookup(name, timeout)`:
  - Uses Unix socket if host starts with `/`, else TCP.
  - GET `/v1/lookup/name?name=<name>&addr-family=dual-stack`.
  - If status != 200: logs error and returns `[]`.
  - Otherwise returns JSON-decoded list.
- `register(name, port, tags=None, check_script=None, host=None)`:
  - Builds payload with id `<name>-<port>`, TTL check (60s).
  - Adds tags as `["k:v"]`.
  - If `check_script` provided: replaces check with `interval=30s, script=...`.
  - Registers via PUT `/v1/agent/service/register`.
  - On non-200 → raises `ConsulException`.
  - Spawns daemon thread that periodically `GET /v1/agent/check/pass/service:<name>-<port>`; on socket error sleeps 2s and retries.
- `deregister(name, port, host=None)`:
  - PUT `/v1/agent/service/deregister/<name>-<port>`.
  - On non-200 → raises `ConsulException`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/discovery.rs`.
- Rust public API surface: `ConsulDiscovery` (stub), `ServiceDiscovery` traits.
- Data model mapping: Python client → Rust Consul discovery abstraction.
- Feature gating: `consul` feature.
- Integration points: distributed runner discovery.

**Implementation Steps (Detailed)**
1. Implement Consul HTTP client or keep stub and document missing features.
2. Add cache semantics and Unix socket support if parity required.
3. Add optional background health-check ticker.

**Tests (Detailed)**
- Python tests: `monolith/native_training/consul_test.py`.
- Rust tests: add mock HTTP tests for lookup/register/deregister (feature-gated).
- Cross-language parity test: compare HTTP request paths and payloads.

**Gaps / Notes**
- Python uses a ByteDance-specific `/v1/lookup/name` API, not stock Consul.
- Rust `ConsulDiscovery` targets standard Consul catalog APIs; endpoint mismatch.

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

### `monolith/native_training/consul_test.py`
<a id="monolith-native-training-consul-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 59
- Purpose/role: Unit tests for Consul client lookup/register/deregister using mocked HTTPConnection.
- Key symbols/classes/functions: `ConsulTest`.
- External dependencies: `unittest.mock`, `six.moves.http_client.OK`.
- Side effects: none (network mocked).

**Required Behavior (Detailed)**
- `test_lookup`:
  - Mock HTTP 200 with JSON payload; `Client.lookup` returns decoded list.
- `test_register` / `test_deregister`:
  - Mock HTTP 200; call methods without error.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests`.
- Rust public API surface: Consul discovery tests.
- Data model mapping: mock HTTP client behavior.
- Feature gating: `consul`.
- Integration points: discovery subsystem.

**Implementation Steps (Detailed)**
1. Add mocked HTTP tests for lookup/register/deregister under `consul` feature.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: add mocked HTTP client tests in `monolith-rs/crates/monolith-training/tests/consul.rs` to cover lookup/register/deregister success paths and error status handling.
- Cross-language parity test: capture Python mocked payloads and compare Rust decoding output shapes and fields.

**Gaps / Notes**
- Python test uses `six.moves.http_client.OK`; ensure Rust test uses numeric 200 status and equivalent JSON decoding.

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

### `monolith/native_training/cpu_sync_training_test.py`
<a id="monolith-native-training-cpu-sync-training-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 360
- Purpose/role: End-to-end CPU sync training tests with Horovod for features, embeddings, sequence features, and distributed sync training.
- Key symbols/classes/functions: `FeatureTask`, `EmbeddingUpdateTask`, `FloatFeatureTask`, `SequenceFeatureTask`, `NonFeatureTask`, `CpuSyncTrainTest`, `DistributedSyncTrainTest`.
- External dependencies: `horovod.tensorflow`, `cpu_training`, `feature`, `embedding_combiners`, `device_utils`, `NativeTask`, `entry`, `advanced_parse`.
- Side effects: sets `MONOLITH_WITH_HOROVOD=True` env var; runs TF estimators and training loops.

**Required Behavior (Detailed)**
- Environment:
  - `MONOLITH_WITH_HOROVOD` must be set **before** importing `monolith.native_training`.
- `FeatureTask`:
  - Input: ragged int64 feature `[1,2,3,4]` repeated 5.
  - Model uses `FeatureSlotConfig`, one slice (dim=5), embedding lookup.
  - For TRAIN: computes loss on embedding, applies gradients via feature factory.
- `EmbeddingUpdateTask`:
  - Compares monolith embedding updates vs TF embedding lookup.
  - Uses `ConstantsInitializer(0)` and `AdagradOptimizer(0.1, accum=1)`.
  - Asserts equality between monolith embedding and TF embedding; increments global step.
- `FloatFeatureTask`:
  - Includes float feature; predictions from float feature sum.
  - Training uses ragged embedding for gradients; float feature only for predictions.
- `SequenceFeatureTask`:
  - Ragged sequence feature; uses `embedding_combiners.FirstN(2)`.
  - Loss from embeddings; predictions from sequence feature sum.
- `NonFeatureTask`:
  - Input dataset yields scalar; model returns constant loss and uses input as train op.
- `CpuSyncTrainTest`:
  - `test_cpu_training_feature/float_feature/sequence_feature/non_feature` run `CpuTraining` with `enable_sync_training=True`.
  - `test_embedding_update` trains 10 steps, compares embedding updates to TF.
- `DistributedSyncTrainTest`:
  - `test_basic` and `test_sparse_pipelining` invoke `distributed_sync_train` with config toggles (pipelined a2a, embedding_postpush).

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests`.
- Rust public API surface: distributed training harness + Horovod equivalent (if any).
- Data model mapping: TF estimator + feature factory → Rust training loop abstractions.
- Feature gating: requires Horovod/TF runtime; Rust likely lacks direct support.
- Integration points: `CpuTraining`, feature pipeline, embedding update logic.

**Implementation Steps (Detailed)**
1. Determine whether Rust will support Horovod-like sync training.
2. If yes, add integration tests for feature pipeline and embedding updates.
3. If no, document tests as Python-only and provide alternative sync tests.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: TBD (requires distributed training support).
- Cross-language parity test: compare embedding update equivalence on a tiny synthetic dataset.

**Gaps / Notes**
- Tests assume Horovod is available and initialize `hvd` in-process.
- Uses `entry.ConstantsInitializer` and `entry.AdagradOptimizer` (must exist in Rust).

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

### `monolith/native_training/cpu_training.py`
<a id="monolith-native-training-cpu-training-py"></a>

**Status:** IN PROGRESS (manual review complete)

**Python Summary**
- Lines: 2449
- Purpose/role: Main CPU training orchestrator for `NativeTask` (local + distributed), covering feature table construction, embedding lookups/prefetch, checkpoint/restore, export, sync/async training, PS lifecycle, and metrics.
- Key symbols/classes/functions: `_combine_slices_as_table`, `_lookup_embedding_ids`, `_convert_parquets_to_instance`, `create_exporter`, `_CpuFeatureFactory`, `_FusedCpuFeatureFactory`, `_MetricsHeartBeatThread`, `CpuTrainingConfig`, `CpuTraining`, `DistributedCpuTrainingConfig`, `_prepare_server`, `_shutdown_ps`, `_join_ps`, `_do_worker_train`, `_do_worker_feature_engineering`, `distributed_train`, `distributed_sync_train`, `local_train_internal`, `local_feature_engineering_internal`, `local_train`.
- External dependencies: TensorFlow (graph/session/estimator, internal file_io/summary), absl flags/logging, numpy, gRPC-free but many Monolith modules (hash tables, export, sync hooks, dataset distribution, service discovery), and optional parquet/CityHash for roughsort.
- Side effects:
  - Sets/clears `TF_CONFIG` env var.
  - Starts TF servers (PS/worker), threads (metrics heartbeat, PS sync).
  - Writes files (`debugging_info.pb`, checkpoints, dense-only checkpoints, candidate item pb).
  - Alters global flags (`FLAGS.dataset_worker_idx`, `FLAGS.dataset_num_workers`, `FLAGS.monolith_alert_proto`).
  - Pushes tensors into TF collections for dequeued features and hooks.

**Required Behavior (Detailed)**
- Flags:
  - Defines `monolith_chief_alert_proto` flag; chief maps it to `FLAGS.monolith_alert_proto`.
- Utility helpers:
  - `_combine_slices_as_table(slices, hashtable_config)`:
    - Builds `EmbeddingHashTableConfig`, appends all slice segments.
    - If exporting, sets `EntryConfig.entry_type = SERVING`.
    - Collects `learning_rate_fn` per slice.
    - Calls `hashtable_config.mutate_table(table_config)`.
    - Returns `HashTableConfigInstance(table_config, learning_rate_fns)`.
  - `_lookup_embedding_ids(hash_table, name_to_embedding_ids)`:
    - Converts ragged tensors to `values` and calls `hash_table.lookup`.
  - `_convert_parquets_to_instance(parquets_path, instance_path)`:
    - Requires `parquets_path` to be a directory; selects **latest date subdir** (YYYYMMDD).
    - Requires `.snappy.parquet` files; reads `item_id` and `fids`.
    - Uses `CityHash64(str(item_id)) & ((1<<63)-1)` for item_id.
    - Writes `Instance` protobufs as `[u64_len][bytes]` records to `instance_path`.
  - `create_exporter(...)`:
    - `ExportMode.STANDALONE` → `StandaloneExporter`.
    - `ExportMode.DISTRIBUTED` → `DistributedExporter` with `dense_only`, `allow_gpu`, `clear_entry_devices`, `include_graphs`, `global_step_as_timestamp=config.enable_sync_training`, `with_remote_gpu`.
    - Else raises `ValueError("Invalid export_mode: ...")`.
- Feature factories:
  - `_CpuFeatureFactory.apply_gradients`:
    - `emb_grads = utils.propagate_back_gradients(grads_and_vars, embeddings.values())`.
    - Maps slot→ids and slot→grads; `global_step = tf.identity(get_or_create_global_step())`.
    - Enqueues `_push` via `AsyncFunctionMgr.add_async_function`, queue `postpush_queue`.
  - `_FusedCpuFeatureFactory.apply_gradients`:
    - Computes gradients on `/device:GPU:0`.
    - Calls `hash_table.apply_gradients(..., auxiliary_bundle, scale, skip_merge_id)` depending on `use_native_multi_hash_table`.
- Metrics:
  - `_MetricsHeartBeatThread` emits `training_heart_beat` with `{type: running/stopped}` every interval; flushes each time.
- `get_req_time(features)`:
  - Returns `features["req_time"][0]` if present; else `None`.
- `CpuTrainingConfig` (dataclass, gflags linked `use_dataservice`):
  - Fields/defaults (non-None):
    - `server_type="worker"`, `index=0`, `num_ps=0`, `num_workers=1`, `model_name=""`.
    - `filter_capacity=300000000`, `filter_split_num=7`, `filter_type=FilterType.SLIDING_HASH_FILTER`, `filter_equal_probability=True`, `hashtable_init_capacity=0`.
    - `embedding_prefetch_capacity=0`, `enable_embedding_postpush=False`, `enable_variable_prefetch=False`, `enable_variable_postpush=False`.
    - `enable_sync_training=False`, `enable_partial_sync_training=False`, `enable_gpu_training=False`, `processes_per_gpu=1`, `merge_sync_training_ckpt=True`.
    - `mode=tf.estimator.ModeKeys.TRAIN`, `enable_realtime_training=False`, `enable_async_optimize=False`.
    - `enable_pipelined_fwda2a=False`, `enable_pipelined_bwda2a=False`.
    - `profile_save_steps_interval=5000`, `reorder_fids_in_data_pipeline=False`.
    - `chief_timeout_secs=1800`, `dense_only_stop_training_when_save=False`.
    - `warmup_file="./warmup_file"`, `skip_zero_embedding_when_serving=False`, `max_rpc_deadline_millis=30000`.
    - `checkpoints_max_to_keep=10`, `cluster_type="stable"`, `max_slow_start_wait_minute=10`.
    - `enable_model_ckpt_info=False`, `feature_eviction_on_save=False`, `only_feature_engineering=False`.
    - `enable_variable_partition=True`, `enable_fused_layout=False`, `force_shutdown_ps=False`.
    - `clear_nn=False`, `continue_training=False`, `enable_model_dump=False`.
    - `enable_resource_constrained_roughsort=False`, `roughsort_items_use_parquet=False`.
    - `items_input_lagrangex_header=False`, `items_input_has_sort_id=False`.
    - `items_input_kafka_dump=False`, `items_input_kafka_dump_prefix=False`.
    - `num_extra_dsworker_on_gpu_worker=0`, `save_summary_steps=100`, `log_step_count_steps=100`.
  - Fields/defaults (None):
    - `use_native_multi_hash_table=None`, `partial_recovery=None`.
    - `tide_start_hour=None`, `tide_start_minute=None`, `tide_end_hour=None`, `tide_end_minute=None`, `tide_save_secs=None`.
    - `profile_some_steps_from=None`, `profile_with_nvprof_from_to=None`.
    - `save_checkpoints_secs=None`, `save_checkpoints_steps=None`, `dense_only_save_checkpoints_secs=None`, `dense_only_save_checkpoints_steps=None`.
    - `submit_time_secs=None`, `containers_ready_time_secs=None`.
    - `reload_alias_map=None`, `enable_alias_map_auto_gen=None`.
    - `roughsort_candidate_items_path=None`.
    - `device_fn=None`, `use_dataservice=None`.
  - `enable_full_sync_training` = `enable_sync_training and not enable_partial_sync_training`.
- Serving-config conversion:
  - `_make_serving_config_from_training_config` disables prefetch/postpush and sync flags; if sync training, disables sync and may set `num_ps = num_workers`.
  - `_make_serving_feature_configs_from_training_configs` sets entry_type to SERVING, sets `skip_zero_embedding`, and calls `cuckoo.SetInParent()`.
- Context helpers:
  - `make_native_task_context` builds `NativeTaskContext` with PS/worker indices and sync backend.
  - `is_chief`: MPI rank 0 for sync/partial sync; otherwise worker index 0.
- `CpuTraining.__init__`:
  - Requires `server_type == "worker"`.
  - Derives `model_name` from `task.p.metrics.deep_insight_name` or class name.
  - For realtime training: sets `partial_recovery=True` and `dense_only_save_checkpoints_secs=1800` if unset.
  - Defaults `use_native_multi_hash_table=True` if None.
  - Updates parser contexts and dataset flags.
  - Collects feature configs via `DumpUtils` (if dumped) or `_collect_feature_name_to_table_config`.
  - Builds serving feature configs and optional dummy merged table (if not native multi hash table).
  - Initializes fused-layout params and export context list.
- `CpuTraining.create_input_fn`:
  - Wraps task input_fn; optionally reorders fids via `distribution_ops.fused_reorder_by_indices`.
  - Uses `datasets.distribute` when `use_dataservice` and not exporting.
  - Always `prefetch(tf.data.AUTOTUNE)`.
- `CpuTraining.create_model_fn`:
  - Builds hash tables + hash filters (and sync clients).
  - Uses GPU device for hash filters if `use_gpu_emb_table`.
  - For exporting: returns hash_filters = `[None]`.
  - For sync training, uses in-worker hash table creation and queue configs.
  - Returns pipelined model_fn via `_get_pipelined_model_fn`.
- `CpuTraining._get_pipelined_model_fn`:
  - If no embedding features: disable feature_factory and return raw model_fn.
  - Defines restore hooks (`CheckpointRestorerHook`) for hash tables, filters, and `CustomRestoreListener`.
  - Save hooks:
    - `PartialRecoverySaver` with `max_to_keep`, `exempt_checkpoint_paths`, `skip_save` if not root node.
    - `HashTableCheckpointSaverListener`, `MultiHashTableCheckpointSaverListener`, `HashFilterCheckpointSaverListener`.
    - Optional `FidSlotCountSaverListener`, feature eviction listeners.
    - Dense-only saver hooks and optional `SyncTrainingSaverControlHook`.
    - Errors if both `save_checkpoints_secs` and `save_checkpoints_steps` are set.
  - Training hooks:
    - `SetCurrentSessionHook` first.
    - Barrier hooks + PS health check if async training.
    - Queue hooks, cached-variable hooks, async function hooks.
  - `model_fn` path:
    - Creates hash_table, hash_filters and `AsyncFunctionMgr`.
    - Handles EOF signal in features (keys `"2"` or `EofAwareTask.EOF_KEY`).
    - Fused layout path uses `hash_table.lookup(..., ret_lookup_callable_fn=True)` and sets `EmbeddingLayoutFactory`.
    - Non-fused path does embedding lookup + optional prefetch queue; records dequeued features into TF collections.
    - Builds `CpuFeatureFactory` or `FusedCpuFeatureFactory` depending on sync training.
    - If exporting with remote GPU, uses `RemotePredictHelper` to call a subgraph; builds prediction-only EstimatorSpec.
    - Adds restore/save/metrics hooks when not exporting.
    - For partial sync training and non-root worker, constructs custom Scaffold with local init ops.
    - Adds dequeued EOF to collection.
- `DistributedCpuTrainingConfig` extends `CpuTrainingConfig` with `model_dir`, thread counts, retry/timeout, redundant PS, uuid, and fountain config.
- PS/worker helpers:
  - `_prepare_server` initializes PS machine info via `logging_ops.machine_info`.
  - `_shutdown_ps` enqueues a shutdown token into each PS queue.
  - `_join_ps` blocks on PS queue; optionally runs a parameter sync thread using `ParameterSyncClient`.
  - `_get_blocked_addrs` extracts all cluster addresses except ignored jobs.
  - `NodeAliveCheckerError` raised when nodes unreachable.
- Worker execution:
  - `_do_worker_train`:
    - Validates task is `NativeTask`, forbids `enable_model_dump` with `with_remote_gpu`.
    - Builds `RunConfig` with `save_summary_steps/log_step_count_steps * num_workers`.
    - Optionally wraps with `EofAwareTask` if partial sync or dataservice.
    - Calls `estimator.train(..., max_steps=params.train.max_steps)`.
    - For `enable_resource_constrained_roughsort`: runs an extra training pass on candidate items.
  - `_do_worker_feature_engineering`:
    - Runs dataset iterator and `FeatureEngineeringSaveHook` in `MonitoredTrainingSession`.
  - `_run_ps_benchmark`: runs extra PS benchmark task and restores cluster.
  - `_save_debugging_info`: writes `debugging_info.pb` with cluster + feature configs.
- `distributed_train`:
  - Validates config; sets alert proto on chief.
  - Creates TF server, registers with discovery, starts PS or worker flow.
  - Worker flow handles retries, node health checks, metrics heartbeat thread, benchmark PS, and graceful shutdown (kill/finish application via yarn_runtime).
- `distributed_sync_train`:
  - MPI-based sync training; sets `use_gpu_emb_table` when GPU training enabled.
  - Adjusts session config (memory optimization, optional XLA).
  - Only rank 0 writes summaries; non-root uses NOP summary writer.
  - Adds sync hooks (`ParameterSyncHook`, `SyncTrainingInfoHook`).
- Local entry points:
  - `local_train_internal`:
    - Creates local PS servers if requested and sets `TF_CONFIG`.
    - Disables meta optimizer (ragged tensors).
    - Runs estimator.train for `steps`.
    - Optionally runs roughsort item pass.
  - `local_feature_engineering_internal`:
    - Similar to distributed FE path but for local graph; supports profiler start/stop.
  - `local_train`:
    - Builds `CpuTrainingConfig`, optionally removes model_dir, calls local_* based on `only_feature_engineering`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/cpu_training.rs` (new), `monolith-rs/crates/monolith-training/src/distributed.rs` (new), `monolith-rs/crates/monolith-training/src/local.rs` (new).
- Rust public API surface:
  - `CpuTrainingConfig`, `DistributedCpuTrainingConfig`, `CpuTraining` wrapper.
  - `distributed_train`, `distributed_sync_train`, `local_train`, `local_feature_engineering`.
  - Exporter selection and hook wiring.
- Data model mapping:
  - TF Estimator-based pipeline ↔ Rust training loop (Candle or TF runtime).
  - Hash table ops ↔ `monolith-hash-table` + distributed PS equivalents.
  - TF collections/hook APIs ↔ Rust hook/callback system.
- Feature gating:
  - TF runtime required for estimator compatibility, saved_model export, and custom ops.
  - MPI/horovod for sync training (`get_mpi_rank/size`).
  - gRPC + service discovery for distributed orchestration.
- Integration points:
  - Hash tables (`hash_table_ops`, `multi_hash_table_ops`), dataset distribution (`datasets`), exporter (`saved_model_exporters`), hooks (`session_run_hooks`, `sync_training_hooks`), metrics (`metric/cli`).

**Implementation Steps (Detailed)**
1. Define Rust config structs mirroring all `CpuTrainingConfig` and `DistributedCpuTrainingConfig` fields and defaults.
2. Implement feature config collection (`DummyFeatureFactory` analogue) or load from dump metadata.
3. Port exporter creation (standalone/distributed), including dense-only export path.
4. Implement hash table creation logic for async vs sync training and fused layout.
5. Recreate embedding prefetch queue + async postpush pipeline.
6. Implement hook system that mirrors restore/save/metrics/barrier hook ordering.
7. Port distributed orchestration (service discovery, PS lifecycle, retries, heartbeat).
8. Implement local training utilities and roughsort item conversion.
9. Provide TF runtime shims for cached-variable handling and partitioned variables if TF backend is used.

**Tests (Detailed)**
- Python tests: `cpu_training_test.py`, `cpu_training_distributed_test_binary.py` + integration harness.
- Rust tests:
  - Unit tests for config defaults and serving-config conversion.
  - Integration tests for local training and distributed orchestration once TF backend exists.
- Cross-language parity test:
  - Compare checkpoint layout, export artifacts, and hash table update semantics on a minimal model.

**Gaps / Notes**
- Uses several TF private APIs and graph collections; Rust will need a custom hook framework.
- `should_do_first_save` is forced `False`, diverging from intended partial recovery behavior.
- Relies on `TF_CONFIG` env var and estimator graph semantics; Candle backend will require a new control-plane.

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

### `monolith/native_training/cpu_training_distributed_test_binary.py`
<a id="monolith-native-training-cpu-training-distributed-test-binary-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 226
- Purpose/role: Distributed CPU training integration test binary with host-based service discovery.
- Key symbols/classes/functions: `SyncHook`, `FeatureTask`, `HostServiceDiscovery`, `test0/1/2`, `test_run`.
- External dependencies: `absl.flags/app`, `tensorflow`, `cpu_training`, `cluster_manager`, `service_discovery`, `feature`.
- Side effects: overrides retry/backoff globals, sets `_shutdown_ps`, writes discovery files, spawns barrier sync.

**Required Behavior (Detailed)**
- Flags:
  - `test_case`, `test_dir`, `server_type` (`ps`/`worker`), `index`, `num_ps`, `num_workers`, `num_extra_ps`, `num_redundant_ps`, `uuid`, `use_native_multi_hash_table`.
- Overrides:
  - `cluster_manager._cluster_query_failure_handler = _sleep_short` (0.1s).
  - `cpu_training._EXTRA_PS_BENCHMARK_SECS = 0.5`.
- `SyncHook`:
  - Creates per-worker boolean var in local variables (chief) or global variables (workers).
  - After session creation, sets its index to True; chief waits until all workers set.
- `FeatureTask`:
  - Defines `training_hooks` param.
  - Model builds feature slot, embedding lookup, applies gradients.
  - Training hooks include `SyncHook` and any provided hooks.
- `HostServiceDiscovery`:
  - Registers by writing files `<base>/<name>/<index>` with address.
  - Query reads files into `{index: addr}` map.
- `test_run(params)`:
  - Builds `DistributedCpuTrainingConfig` using flags and a per-test model dir.
  - Sets `params.train.max_pending_seconds_for_barrier = 2`.
  - Uses `HostServiceDiscovery` and runs `cpu_training.distributed_train`.
- `test0`: normal run.
- `test1`: overrides `_shutdown_ps` to never exit.
- `test2`: adds `RaiseErrorHook` that throws `DeadlineExceededError` on first `before_run`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests`.
- Rust public API surface: distributed training integration tests + file-based discovery.
- Data model mapping: HostServiceDiscovery → file-backed discovery in Rust (not present).
- Feature gating: requires distributed training + PS/worker runner.
- Integration points: `distributed_train` analog, barrier sync.

**Implementation Steps (Detailed)**
1. Add file-backed discovery helper for integration tests.
2. Add hook to block chief until all workers register.
3. Add test cases that simulate non-shutdown and deadline errors.

**Tests (Detailed)**
- Python tests: this binary test (invoked by integration harness).
- Rust tests: integration tests if Rust distributed runner exists.
- Cross-language parity test: verify barrier synchronization semantics.

**Gaps / Notes**
- This script mutates module-level globals (`_EXTRA_PS_BENCHMARK_SECS`, `_shutdown_ps`).

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

### `monolith/native_training/cpu_training_test.py`
<a id="monolith-native-training-cpu-training-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 597
- Purpose/role: Comprehensive CPU training tests covering feature slots, export modes, occurrence/expire configs, distributed training, debugging server, and local train.
- Key symbols/classes/functions: `FeatureTask`, `FloatFeatureTask`, `SequenceFeatureTask`, `FeatureWithSlotOccurrenceThresholdTask`, `FeatureWithExpireTimeTask`, `NonFeatureTask`, `CpuTrainTest`, `DistributedTrainTest`, `LocalTrainTest`.
- External dependencies: `tensorflow`, `cpu_training`, `entry`, `feature`, `utils`, `debugging_server`, `saved_model_exporters`, `ExportMode`, `debugging_info_pb2`, `embedding_hash_table_pb2`, `ServiceDiscovery`.
- Side effects: spawns subprocesses for distributed tests; writes checkpoints and export dirs; reads debugging info files.

**Required Behavior (Detailed)**
- Shared helpers:
  - `inc_global_step_op()` increments global step and returns grouped op.
  - `FLAGS.use_native_multi_hash_table` controls hash table implementation.
- `FeatureTask`:
  - Input: ragged feature tensor.
  - Model: create slot/slice, embedding lookup; grads applied via feature factory; predict returns sum.
  - Serving input receiver uses ragged constant + placeholder for serialized input.
- `FloatFeatureTask`:
  - Uses ragged embedding + float feature; predictions from float feature; training uses embedding grads.
- `SequenceFeatureTask`:
  - Uses combiner `FeatureColumnV1.first_n(2)`; predictions from sequence feature sum.
- `FeatureWithSlotOccurrenceThresholdTask`:
  - Creates slot with `slot_id=2021`, `occurrence_threshold=3`; asserts training captures threshold map.
- `FeatureWithExpireTimeTask`:
  - Two slots with `expire_time=0` and `1`; uses `ZerosInitializer`.
  - After training, checks `_slot_to_expire_time` map and prediction values.
- `NonFeatureTask`:
  - Input dataset yields scalar; train op uses input with global step increment.
- `CpuTrainTest`:
  - `test_cpu_training_feature` basic training.
  - `test_with_misc_features`: `feature_eviction_on_save=True`.
  - `test_with_export_when_saving`: `serving.export_when_saving=True`.
  - `test_dense_only_export`: export mode `DISTRIBUTED` + `dense_only_save_checkpoints_steps=10`.
  - `test_with_prefetch_postpush`: enables variable prefetch/postpush and embedding postpush.
  - `test_cpu_training_float_feature` and `test_cpu_training_sequence_feature` run those tasks.
  - `test_cpu_training_with_slot_occurrence_threshold` checks internal threshold map.
  - `test_cpu_training_with_expire_time` checks expire time map and prediction values.
  - `test_cpu_training_non_feature` runs non-feature task.
  - `test_gpu_export`: exports saved model with remote GPU.
- `DistributedTrainTest`:
  - Spawns `cpu_training_distributed_test_binary` processes for PS/worker.
  - Tests: basic, extra_ps, redundant_ps, debugging server (case=1), temporary error (case=2).
  - `test1_with_debugging_server` waits for checkpoints then reads debugging info proto; checks variable/feature fetch via debugging server.
- `LocalTrainTest`:
  - `local_train` with and without PS.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests`.
- Rust public API surface: training loop, export pipeline, debug info tooling.
- Data model mapping: TF Estimator/FeatureFactory → Rust training/feature abstractions.
- Feature gating: depends on TF runtime and distributed runner.
- Integration points: `TrainingConfig`, export utilities, discovery and debugging.

**Implementation Steps (Detailed)**
1. Build Rust integration tests that cover: basic training, feature thresholds, expire time, and export flow.
2. Add distributed runner test harness or mark as Python-only.
3. Provide debug info export and retrieval parity tests.

**Tests (Detailed)**
- Python tests: this file plus `cpu_training_distributed_test_binary`.
- Rust tests: integration tests for training/export/debug info.
- Cross-language parity test: compare embedding predictions and debug info outputs.

**Gaps / Notes**
- Uses private internals (`training._slot_to_occurrence_threshold`, `_slot_to_expire_time`).
- Debugging server depends on proto + embedding hash table dumps not present in Rust.

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

### `monolith/native_training/data/__init__.py`
<a id="monolith-native-training-data-init-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 20
- Purpose/role: Package exports for dataset builders and feature utilities.
- Key symbols/classes/functions: re-exports `PBDataset`, `InstanceReweightDataset`, `NegativeGenDataset`, `PbType`, `parse_examples`, `parse_instances`, `parse_example_batch`, `filter_by_*`, `feature_combine`, `negative_sample`, `switch_slot`, `special_strategy`.
- External dependencies: internal modules `datasets`, `parsers`, `feature_utils`.
- Side effects: imports modules at package import time.

**Required Behavior (Detailed)**
- Importing package exposes symbols listed above at `monolith.native_training.data.*`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/src`.
- Rust public API surface: module re-exports for dataset and feature utils.
- Data model mapping: match Python export surface.
- Feature gating: none.
- Integration points: data pipeline and training input.

**Implementation Steps (Detailed)**
1. Re-export dataset and parser APIs in Rust modules.

**Tests (Detailed)**
- Python tests: none specific.
- Rust tests: module visibility tests.
- Cross-language parity test: not applicable.

**Gaps / Notes**
- Bugs/quirks to preserve or fix explicitly:
  - `Env.is_finalized()` returns `self.is_finalized` (recursive), and is referenced without call in some asserts, so the assert always passes.
  - `_embedding_lookup` references `slot_id` without defining it (should be `feature_column.feature_slot.slot_id()`), may raise `NameError` if QR path is executed.
  - `_seq_embedding_lookup` uses `feature_slice.init_minval_for_oov/init_maxval_for_oov` which are not defined on `FeatureSlice`.
  - `collections.namedtuple` imported but unused.

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

### `monolith/native_training/data/data_ops_test.py`
<a id="monolith-native-training-data-data-ops-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 502
- Purpose/role: End-to-end tests for PB datasets, parsing, filtering, compression, and dataset variants.
- Key symbols/classes/functions: `DataOpsTest`, `CahceOneDatasetTest`, `DecompressTest`, helper parsers.
- External dependencies: `tensorflow`, `PBDataset`, `parse_examples/instances/example_batch`, `feature_utils` filters, `ExampleBatch`, `gen_random_data_file`, `session_hooks`.
- Side effects: creates temp files under `tmp_data`; invokes external `zstd` binary; writes compressed files.

**Required Behavior (Detailed)**
- `DataOpsTest.setUpClass`:
  - Generates random instance data files (3 parts) using `gen_random_data_file`.
- `pb_dataset_target(input_pb_type, output_pb_type, filter_fn=None)`:
  - Builds `PBDataset` for Instance/Example/ExampleBatch with appropriate flags.
  - For ExampleBatch, applies `instance_reweight`.
  - Applies optional `filter_fn`.
  - Batches and maps to parser (`parse_inst_exam`/`parse_eb`).
  - Iterates through dataset and counts elements (logs count).
- Tests validate permutations:
  - Instance→Instance/Example, Example→Example/Instance, ExampleBatch→Example/Instance.
  - PLAINTEXT output for Instance/Example.
  - `filter_by_fids`, `filter_by_value` (ge/in/eq/between/str/any/all/diff), `special_strategy`.
  - `parse_example_batch` for scalar and batch inputs.
  - `PBDataset` resolves to `FilePBDataset` or `KafkaDataset` based on flags/args.
  - `testCreateInstanceDatasetHdfs` reads generated files via `PBDataset`.
  - `PBDataset.gen_patterns` returns correct date range size.
- `CahceOneDatasetTest`:
  - `CacheOneDataset` wraps dataset; second element flagged `True`.
- `DecompressTest`:
  - Creates copies of examplebatch data and tests `CompressType.ZSTD/ZLIB/GZIP`.
  - Uses `parse_example_batch` to validate decompression.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/tests`.
- Rust public API surface: dataset parsing, compression, filter utils.
- Data model mapping: TF datasets → Rust data pipeline equivalents.
- Feature gating: compression codecs and Kafka support.
- Integration points: `monolith-data` crate and training input.

**Implementation Steps (Detailed)**
1. Add Rust tests for parsing Instance/Example/ExampleBatch formats.
2. Add filter/feature utility tests mirroring Python cases.
3. Add compression decode tests (zstd/zlib/gzip).

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: dataset parsing + compression + filter tests.
- Cross-language parity test: compare parsed feature dict shapes and counts.

**Gaps / Notes**
- Requires external `zstd` binary for decompression test.
- Relies on hard-coded proto files under `monolith/native_training/data/training_instance`.

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

### `monolith/native_training/data/data_service_parquet_test.py`
<a id="monolith-native-training-data-data-service-parquet-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 145
- Purpose/role: Integration test for TF data service reading Parquet via `PBDataset`.
- Key symbols/classes/functions: `DataServiceTest2.testDataServiceWithParquetDataset`.
- External dependencies: `tensorflow.data.experimental.service`, `PBDataset`, `PbType`, `ExampleBatch`, `json`.
- Side effects: starts dispatcher/workers on port 7080, reads local files.

**Required Behavior (Detailed)**
- `setUpClass`/`tearDownClass`: create and destroy dispatcher + workers.
- `testDataServiceWithParquetDataset`:
  - Reads `META_JSON_PATH` and `PARQUET_DIR` env vars (defaults under `$HOME/temp`).
  - Loads meta JSON, builds column names/types mapping.
  - Creates `PBDataset` with `use_data_service=True`, `use_parquet=True`, `output_pb_type=PLAINTEXT`.
  - Registers dataset and reads from two consumers (distributed_epoch).
  - Parses `ExampleBatch` from bytes, accumulates `batch_size`.
  - Prints row count (assertion against parquet row count is commented out).

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/tests`.
- Rust public API surface: data service + parquet ingestion tests (if supported).
- Data model mapping: `PBDataset` + Parquet pipeline.
- Feature gating: Parquet and data service support.
- Integration points: dataset registration/consumers.

**Implementation Steps (Detailed)**
1. Add Rust integration test only if parquet + data service are implemented.
2. Otherwise, document as Python-only system test.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: optional integration tests.
- Cross-language parity test: compare total row counts.

**Gaps / Notes**
- Test depends on local parquet files and meta JSON; not hermetic.

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

### `monolith/native_training/data/data_service_test.py`
<a id="monolith-native-training-data-data-service-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 98
- Purpose/role: Tests data service split provider with `DynamicMatchingFilesDataset`.
- Key symbols/classes/functions: `DataServiceTest.testSplitProvider`.
- External dependencies: `tensorflow.data.experimental.service`, `DynamicMatchingFilesDataset`.
- Side effects: starts dispatcher/workers on port 7080.

**Required Behavior (Detailed)**
- Registers `DynamicMatchingFilesDataset` and consumes it via two distributed_epoch consumers.
- Alternates pulling items until both consumers are exhausted.
- Asserts total count equals 19.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/tests`.
- Rust public API surface: data service dataset tests.
- Data model mapping: dynamic file matching dataset.
- Feature gating: TF data service equivalent required.
- Integration points: dataset registry + consumer.

**Implementation Steps (Detailed)**
1. Only port if Rust data service is implemented.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: add integration test in `monolith-rs/crates/monolith-data/tests/data_service.rs` that spins a local dispatcher/worker (or mocked service) and verifies dual consumer exhaustion with total count 19.
- Cross-language parity test: use the same file list and expected count in Python and Rust, compare emitted element sequence length.

**Gaps / Notes**
- Requires a Rust equivalent of TF data service; otherwise gate the test behind a feature and document as unsupported.

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

### `monolith/native_training/data/datasets.py`
<a id="monolith-native-training-data-datasets-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 1642
- Purpose/role: Dataset factory and dataset classes for PB files, Kafka streams, Parquet, TFRecord, transforms, and data service.
- Key symbols/classes/functions: `PBDataset`, `FilePBDataset`, `DistributedFilePBDataset`, `KafkaDataset`, `ParquetDataset`, `InstanceReweightDataset`, `NegativeGenDataset`, `SplitFlowDataset`, `MergeFlowDataset`, `TransformDataset`, `DatasetMetaclass`, `distribute`, `merged_window`.
- External dependencies: TensorFlow internals, custom ops `gen_monolith_ops`, Kafka consumer, data service APIs, feature utils.
- Side effects: defines many flags; registers graph collections; disables iterator save/restore; spawns Kafka polling thread.

**Required Behavior (Detailed)**
- Flags defined: `data_service_dispatcher`, `dataset_use_dataservice`, `dataset_input_patterns`, `dataset_input_use_snappy`, `dataset_input_compression_type`, `dataset_input_use_parquet`, `dataset_input_use_tfrecord`, `dataset_worker_idx`, `dataset_num_workers`, `kafka_other_metadata`.
- `DatasetMetaclass.__call__`:
  - Normalizes shorthand kwargs (`topics_or_files`, `buffer_size_or_group_id`, `input_pb_type_or_servers`).
  - Expands `dataset_input_patterns` into file patterns (DATE/INT range syntax); overwrites `patterns`; removes `file_name`.
  - Forces parquet/tfrecord flags from global FLAGS; prevents both true.
  - If Kafka args present, returns `KafkaDataset`.
  - Otherwise returns `DistributedFilePBDataset`, `ParquetDataset`, `TFRecordDatasetWrapper`, or `FilePBDataset` based on args.
- `PBDataset`: empty init (factory via metaclass).
- `PBDataset.gen_patterns(...)`:
  - Expands date/hour ranges into path patterns.
- `DynamicMatchingFilesDataset`: uses custom op `dynamic_matching_files_dataset`.
- `ParquetDataset`: validates columns/types, sets `OUTPUT_PB_TYPE_GRAPH_KEY`, uses custom parquet op.
- `FilePBDataset`:
  - Determines input/output pb types, uses FeatureList to prune if possible.
  - Configures flags: has_sort_id, kafka_dump, lagrangex_header, compression, snappy.
  - Adds output pb type to collection; calls `pb_dataset` op.
- `DistributedFilePBDataset`:
  - Creates file list, optional dynamic sharding, data service or matching_files.
  - Supports parquet/tfrecord mapping; handles sharding by worker or explicit shard_num.
- `InstanceReweightDataset`: wraps custom op `instance_reweight_dataset` based on action priorities.
- `NegativeGenDataset`: wraps custom op `instance_negative_gen_dataset`, creates item pool.
- `SplitFlowDataset` / `MergeFlowDataset`: custom ops for flow splitting/merging.
- `KafkaDataset`:
  - Initializes Kafka resource via custom ops; uses `kafka_read_next_v2`, unbatch.
  - Sets output pb type collection and flags (sort_id, dump, lagrangex_header).
- `PyKafkaDataset`:
  - Python KafkaConsumer with background polling thread; converts strings to variant via `string_to_variant`.
- `register_dataset` / `from_dataset_id`:
  - Data service register/uncompress; external_state_policy preserved.
- `merged_window`: window and reshape dataset elements.
- `distribute`: data service integration with sync training / ps-worker queue; handles Horovod/BytePS.
- `TransformDataset`: wraps `transform_dataset` op with serialized Transform config.
- Monkey patches: `Dataset.instance_reweight`, `.negative_gen`, `.split_flow`, `.merge_flow`, `.distribute`, `.merged_window`, `.transform`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/src`.
- Rust public API surface: dataset factory + dataset operators.
- Data model mapping: TF Dataset/Variant → Rust streaming dataset abstractions.
- Feature gating: Kafka, Parquet, TFRecord, DataService, custom ops.
- Integration points: data ingestion and training input pipeline.

**Implementation Steps (Detailed)**
1. Decide which dataset sources are in-scope for Rust (file/kafka/parquet).
2. Implement dataset factory/dispatch logic and pattern expansion.
3. Provide custom ops or pure-Rust replacements for instance reweight, negative gen, transform.
4. Add data service integration or document unsupported.

**Tests (Detailed)**
- Python tests: `data_ops_test.py`, `data_service_test.py`, `negative_gen_test.py`, etc.
- Rust tests: dataset factory, pattern expansion, ops parity tests.
- Cross-language parity test: compare dataset counts and parsed outputs for fixtures.

**Gaps / Notes**
- Heavy reliance on custom TF ops; Rust needs replacements or TF runtime backend.

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

### `monolith/native_training/data/eager_mode_test.py`
<a id="monolith-native-training-data-eager-mode-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 186
- Purpose/role: Eager-mode dataset parsing tests for Instance/Example/ExampleBatch.
- Key symbols/classes/functions: `DataOpsTest.target`, test methods.
- External dependencies: `tensorflow`, `PBDataset`, `parse_instances/parse_examples/parse_example_batch`, `switch_slot`, `feature_combine`.
- Side effects: reads training_instance fixtures.

**Required Behavior (Detailed)**
- `target(input_pb_type, output_pb_type)`:
  - Builds `PBDataset` for Instance/Example/ExampleBatch and applies instance reweight for ExampleBatch.
  - Parses via `parse_inst_exam` or `parse_eb` depending on pb types.
  - Batches then iterates `dataset.take(5)` and asserts feature dict length ∈ {26,27}.
- `testExampleBatch2Instance`, `testExample2Instance`, `testInstance2Instance` call `target`.
- `testExampleBatch`:
  - Parses ExampleBatch to ExampleBatch via `parse_example_batch`, asserts len ∈ {26,27}.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/tests`.
- Rust public API surface: dataset parsing tests (eager mode equivalent).
- Data model mapping: dataset pipelines in Rust.
- Feature gating: none.
- Integration points: `monolith-data` parsing pipeline.

**Implementation Steps (Detailed)**
1. Add parsing tests for all pb types with fixed fixtures.
2. Ensure eager-mode behavior maps to Rust pipeline semantics.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: create fixture-based parsing tests in `monolith-rs/crates/monolith-data/tests` that load the same serialized examples used by Python and assert parsed feature values/types.
- Cross-language parity test: generate golden parsed outputs from Python, then compare Rust parsing outputs field-by-field.

**Gaps / Notes**
- Requires deterministic fixture generation for each PB type (Example/ExampleBatch/Instance).

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

### `monolith/native_training/data/extract_fid_test.py`
<a id="monolith-native-training-data-extract-fid-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 30
- Purpose/role: Tests custom op `extract_fid`.
- Key symbols/classes/functions: `ExtraFidTest.test_parse_search`.
- External dependencies: `tensorflow`, `gen_monolith_ops.extract_fid`.
- Side effects: none.

**Required Behavior (Detailed)**
- `extract_fid(185, 4)` must return `1153447759131936`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/tests` (or runtime ops crate).
- Rust public API surface: `extract_fid` equivalent.
- Data model mapping: custom op to Rust function.
- Feature gating: requires runtime ops.
- Integration points: feature parsing pipeline.

**Implementation Steps (Detailed)**
1. Implement `extract_fid` in Rust runtime ops.
2. Add a unit test for the exact constant output.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: add exact-match test.
- Cross-language parity test: compare output for fixed inputs.

**Gaps / Notes**
- Relies on custom TF op; no Rust implementation yet.

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

### `monolith/native_training/data/feature_list.py`
<a id="monolith-native-training-data-feature-list-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 409
- Purpose/role: Parses feature_list configuration and provides feature lookup, slot mapping, and feature filtering utilities.
- Key symbols/classes/functions: `Feed`, `Cache`, `Feature`, `FeatureList`, `get_feature_name_and_slot`, `add_feature`, `add_feature_by_fids`, `get_valid_features`, `is_example_batch`.
- External dependencies: `absl.flags`, `tensorflow`, `numpy`, `inspect`, `dataclasses`, `monolith.native_training.data.utils`.
- Side effects: populates global `_cache` and `_VALID_FNAMES`; adds to TF collections via `add_to_collections`.

**Required Behavior (Detailed)**
- `new_instance(cls, args)`:
  - Inspects `__init__` signature and passes only matching args.
- `Feed` dataclass:
  - `shared` parsed from truthy strings; `feature_id` cast to int.
  - `name` property returns `feed_name`.
- `Cache` dataclass:
  - `capacity`, `timeout` cast to int if string.
  - `name` property uses `cache_name` > `cache_key_class` > `cache_column` else raises.
- `Feature` dataclass:
  - Parses comma-separated string fields into lists.
  - Casts slot/shared/need_raw/feature_id/version to proper types.
  - `__str__` renders key=value terms based on type hints; bools only if True.
  - `name` strips `fc_`/`f_` prefixes and lowercases terms.
  - `depend_strip_prefix` strips prefixes for dependencies.
- `FeatureList`:
  - Stores column names, feeds, caches, features; builds slot→feature mapping.
  - Adds itself to TF collections `feature_list`.
  - `__getitem__` resolves by slot id or feature name variants (`f_`, `fc_`, dashed names).
  - `get_with_slot(slot)` returns list or empty.
  - `__contains__` supports names and slots.
  - `parse(fname=None, use_old_name=True)`:
    - Reads config file; caches results in `_cache`.
    - Parses `column_name:`, `cache_column:` and `feed/cache/feature` lines into dataclasses.
    - `use_old_name` chooses between raw feature_name or normalized name as key.
- `get_feature_name_and_slot(item)`:
  - Handles int, str, or FeatureColumn-like objects.
  - Uses `FeatureList.parse()` with fallback to slot name utility.
- `is_example_batch()`:
  - Checks `FLAGS.data_type` for example_batch.
- `add_feature` / `add_feature_by_fids`:
  - Maintains `_VALID_FNAMES`; for example_batch fids, resolves feature list by slot and version.
  - Raises if fid cannot be mapped.
- `get_valid_features()` returns list of collected features.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/src/feature_list.rs` (if implemented).
- Rust public API surface: feature list parser + lookup utilities.
- Data model mapping: config parsing + slot/feature mapping.
- Feature gating: none.
- Integration points: data parsing and column pruning.

**Implementation Steps (Detailed)**
1. Implement feature_list.conf parser with the same key parsing rules.
2. Add slot/name resolution helpers.
3. Implement example_batch feature filtering via fid decoding.

**Tests (Detailed)**
- Python tests: none (feature_list_test.py is empty).
- Rust tests: add unit tests for parsing, slot lookup, fid mapping.
- Cross-language parity test: compare parsed feature list for a fixed config file.

**Gaps / Notes**
- Uses global caches and TF collections; Rust should provide equivalent global registry if required.

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

### `monolith/native_training/data/feature_list_test.py`
<a id="monolith-native-training-data-feature-list-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 0
- Purpose/role: Empty file (no tests).
- Key symbols/classes/functions: none.
- External dependencies: none.
- Side effects: none.

**Required Behavior (Detailed)**
- None.

**Rust Mapping (Detailed)**
- Target crate/module: none.
- Rust public API surface: none.
- Data model mapping: none.
- Feature gating: none.
- Integration points: none.

**Implementation Steps (Detailed)**
1. No Rust tests required unless feature list gains tests.

**Tests (Detailed)**
- Python tests: none.
- Rust tests: N/A (no Rust surface; no behavior to test).
- Cross-language parity test: N/A.

**Gaps / Notes**
- Keep this section in case future feature-list tests are added; no current parity work required.

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

### `monolith/native_training/data/feature_utils.py`
<a id="monolith-native-training-data-feature-utils-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 1070
- Purpose/role: Feature/label filtering, transformation, and utility ops over variant tensors; mostly thin wrappers around custom `gen_monolith_ops` kernels with strict input validation and feature-registration side effects.
- Key symbols/classes/functions: `filter_by_fids`, `filter_by_feature_value`, `filter_by_value`, `add_action`, `add_label`, `scatter_label`, `filter_by_label`, `special_strategy`, `negative_sample`, `feature_combine`, `switch_slot`, `switch_slot_batch`, `label_upper_bound`, `label_normalization`, `use_field_as_label`, `create_item_pool`, `item_pool_random_fill`, `item_pool_check`, `save_item_pool`, `restore_item_pool`, `fill_multi_rank_output`, `use_f100_multi_head`, `map_id`, `multi_label_gen`, `string_to_variant`, `string_to_variant_with_transform`, `variant_to_zeros`, `kafka_resource_init`, `kafka_read_next`, `kafka_read_next_v2`, `has_variant`, `gen_fid_mask`, `tf_example_to_example`.
- External dependencies: TensorFlow, numpy, `idl.matrix.proto.line_id_pb2.LineId`, `data_op_config_pb2.LabelConf`/`TFRecordFeatureDescription`, `gen_monolith_ops` custom kernels.
- Side effects: calls `add_feature`/`add_feature_by_fids` to ensure downstream parsing includes required fields; validates/loads operand files via `tf.io.gfile.exists`; asserts on invalid inputs; builds TF ops that mutate or filter variant tensors.

**Required Behavior (Detailed)**
- `filter_by_fids(variant, filter_fids, has_fids, select_fids, has_actions, req_time_min, select_slots, variant_type)`:
  - Coerces `filter_fids`/`has_fids`/`select_fids` to `np.uint64` then `int64` list; defaults to empty lists.
  - `select_slots` defaults to empty; asserts all slots > 0.
  - If `variant_type != 'instance'`, calls `add_feature_by_fids` for all fid lists.
  - Calls `ragged_data_ops.set_filter(...)` with `has_actions or []`, `req_time_min`, `select_slots`, `variant_type` and returns variant tensor.
- `filter_by_feature_value(variant, field_name, op, operand, field_type, keep_empty, operand_filepath)`:
  - `op` must be in `{gt,ge,eq,lt,le,neq,between,in,not-in,all,any,diff,startswith,endswith}`.
  - Exactly one of `operand` or `operand_filepath` is provided; if filepath set, it must exist and `op` must be in `{in, not-in}`.
  - `field_type` must be in `{int64,float,double,bytes}`; builds `int_operand`, `float_operand`, `string_operand` based on type/op:
    - `all/any/diff` only for `int64` (operand int or list of int).
    - `between` uses a list of numbers (float/double) or ints for int64.
    - `bytes` accepts str or list of str; otherwise raises `RuntimeError("params error!")`.
  - Calls `ragged_data_ops.feature_value_filter(...)` with operands, file path, `keep_empty`, returns variant.
- `filter_by_value(variant, field_name, op, operand, variant_type, keep_empty, operand_filepath)`:
  - For `variant_type != 'instance'`, calls `add_feature('__LINE_ID__')`.
  - `field_name` must exist in `LineId` descriptor; uses proto field `cpp_type`/`has_options` to determine parsing rules.
  - Same operand vs operand_filepath exclusivity; operand file must exist; only `in/not-in` supported with filepath.
  - For repeated fields (`field.has_options`), only `all/any/diff` allowed and only integer types.
  - For `string` fields: operand must be str or list of str, else `RuntimeError("params error!")`.
  - Calls `ragged_data_ops.value_filter(...)` with `variant_type` and returns variant.
- `add_action(variant, field_name, op, operand, action, variant_type)`:
  - For `variant_type != 'instance'`, calls `add_feature('__LINE_ID__')`.
  - `op` in `{gt,ge,eq,lt,le,neq,between,in}`; field must exist in `LineId`.
  - Builds typed operands (float/int/string) based on field cpp_type; for `in/between` on integer types, operand is list of int.
  - Calls `ragged_data_ops.add_action(..., actions=[action], variant_type)`.
- `add_label(variant, config, negative_value, new_sample_rate, variant_type)`:
  - For `variant_type != 'instance'`, calls `add_feature('__LINE_ID__')`.
  - `config` is required; `new_sample_rate` must be in `(0, 1.0]`.
  - Parses `config` with `;` task separator; each task `pos_actions:neg_actions:sample_rate` (empty lists allowed). Skips empty trailing parts.
  - Builds `LabelConf` proto and calls `ragged_data_ops.add_label(..., negative_value, sample_rate=new_sample_rate)`.
- `scatter_label(variant, config, variant_type)`:
  - For `variant_type != 'instance'`, calls `add_feature('__LABEL__')` and `add_feature('__LINE_ID__')`.
  - `config` required; passes through to `ragged_data_ops.scatter_label`.
- `filter_by_label(variant, label_threshold, filter_equal, variant_type)`:
  - For `variant_type != 'instance'`, calls `add_feature('__LABEL__')`.
  - `label_threshold` must be non-empty list.
  - Calls `ragged_data_ops.filter_by_label(..., filter_equal, variant_type)` and returns boolean tensor.
- `special_strategy(variant, strategy_list, strategy_conf, variant_type, keep_empty_strategy)`:
  - For `variant_type != 'instance'`, calls `add_feature('__LABEL__')` and `add_feature('__LINE_ID__')`.
  - `strategy_conf` is optional; parses comma-separated `strategy:sample_rate` or `strategy:sample_rate:label` entries.
  - Ensures lengths consistent and each `sample_rate` in `[0,1]`.
  - Calls `ragged_data_ops.special_strategy(..., keep_empty_strategy, variant_type)`.
- `negative_sample(variant, drop_rate, label_index, threshold, variant_type, action_priority, per_action_drop_rate)`:
  - For `variant_type != 'instance'`, calls `add_feature('__LABEL__')`.
  - `action_priority` and `per_action_drop_rate` are optional strings; if both set, parse lists of actions and per-action drop rates.
  - Calls `ragged_data_ops.negative_sample(..., priorities, actions, per_action_drop_rate)`.
- `feature_combine(src1, src2, slot)`:
  - Requires `tf.RaggedTensor` inputs; calls `ragged_data_ops.feature_combine(..., fid_version=2)`.
  - If `splits[0]` is `float32`, uses `from_row_splits(values, splits[1])`; else `from_nested_row_splits`.
- `switch_slot(ragged, slot)`:
  - Requires `tf.RaggedTensor`; calls `ragged_data_ops.switch_slot(..., fid_version=2)`.
  - If `splits[0]` is `float32`, returns new ragged from row_splits; else returns `ragged.with_flat_values(values)`.
- `switch_slot_batch(variant, features, variant_type, suffix)`:
  - `features` maps feature name → `(inplace, new_slot)`; `variant_type` must be `example` or `example_batch`.
  - Builds `features`, `inplaces`, `slots` arrays; calls `ragged_data_ops.switch_slot_batch(..., suffix)`.
- `label_upper_bound(variant, label_upper_bounds, variant_type)`:
  - `label_upper_bounds` non-empty; calls `ragged_data_ops.label_upper_bound`.
- `label_normalization(variant, norm_methods, norm_values, variant_type)`:
  - `norm_methods` length must equal `norm_values`; calls `ragged_data_ops.label_normalization`.
- `use_field_as_label(variant, field_name, overwrite_invalid_value, label_threshold, variant_type)`:
  - Calls `ragged_data_ops.use_field_as_label` to overwrite labels from LineId field with optional clamping.
- Item pool ops:
  - `create_item_pool(start_num, max_item_num_per_channel, container, shared_name)` asserts `start_num >= 0`, `max_item_num_per_channel > 0`, calls `ItemPoolCreate`.
  - `item_pool_random_fill`, `item_pool_check(model_path, global_step, nshards, buffer_size)`, `save_item_pool`, `restore_item_pool` delegate to custom ops.
- `fill_multi_rank_output(variant, enable_draw_as_rank, enable_chnid_as_rank, enable_lineid_rank_as_rank, rank_num, variant_type)`:
  - For `variant_type != 'instance'`, calls `add_feature('__LINE_ID__')`.
  - Calls `ragged_data_ops.fill_multi_rank_output`.
- `use_f100_multi_head(variant, variant_type)`:
  - Pass-through to `ragged_data_ops.use_f100_multi_head`.
- `map_id(tensor, map_dict, default)`:
  - `map_dict` non-empty; passes `from_value`, `to_value`, `default` to `ragged_data_ops.MapId`.
- `multi_label_gen(variant, head_to_index, head_field, pos_actions, neg_actions, use_origin_label, pos_label, neg_label, action_priority, task_num, variant_type)`:
  - Builds `head_to_index` string (`head:idx`), computes `task_num` if unset; asserts `max_idx < task_num`.
  - If `use_origin_label`, `pos_actions` and `neg_actions` must be empty; otherwise `pos_actions` non-empty.
  - `head_field` must exist in `LineId` descriptor and be int or string.
  - Calls `ragged_data_ops.multi_label_gen(..., action_priority, pos/neg actions, labels, variant_type)`.
- `string_to_variant(...)`:
  - `variant_type` must be `instance|example|examplebatch|example_batch`; converts string tensor into variant using header flags and optional `chnids/datasources`.
- `string_to_variant_with_transform(...)`:
  - Similar to `string_to_variant` but accepts `input_type` and `output_type` for on-the-fly transforms.
- `variant_to_zeros(tensor)`:
  - Calls `ragged_data_ops.variant_to_zeros` to produce zeroed variant tensor.
- Kafka ops:
  - `kafka_resource_init(topics, metadata, input_pb_type, output_pb_type, has_sort_id, lagrangex_header, kafka_dump_prefix, kafka_dump, container, shared_name)` calls `KafkaGroupReadableInit`.
  - `kafka_read_next`/`kafka_read_next_v2` call `KafkaGroupReadableNext`/`NextV2` with poll/stream timeouts.
- `has_variant(input, variant_type)`:
  - Calls `ragged_data_ops.HasVariant`.
- `gen_fid_mask(ragged, fid)`:
  - Casts `fid` to `np.uint64` → `int64`; calls `monolith_gen_fid_mask` with row_splits and flat_values.
- `tf_example_to_example(serialized, sparse_features, dense_features, label, instance_weight)`:
  - Defaults: empty sparse/dense/label/instance_weight if None.
  - Validates no overlaps between sparse/dense/label/instance_weight; slot ids unique; each slot id in `[1, 32768)`.
  - Builds `TFRecordFeatureDescription` proto and calls `MonolithTFExampleToExample` op.
- Error semantics:
  - Many checks are `assert` (raising `AssertionError`), some raise `RuntimeError("params error!")` for invalid bytes operands.
  - Operand file must exist and be used only with `in/not-in` ops; callers rely on these preconditions.
- I/O formats:
  - Variant tensor format is custom monolith variant; string inputs for `string_to_variant*` are framed by headers (sort header or lagrangex header) and length-prefixed protos.
  - Operand file for `operand_filepath` is expected to contain serialized `example_pb2.FilterValues` (see tests).
  - `tf_example_to_example` expects TF Example serialized bytes and emits Monolith Example variant.
- Threading/concurrency:
  - No explicit threading here; concurrency behavior is inside custom ops (e.g., Kafka resources).
- Determinism/perf:
  - Performance relies on custom ops; callers expect these to be safe in `tf.data` pipelines (including parallel map/filter). Determinism depends on op implementations; keep semantics stable.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/src` for dataset/feature ops; optionally `monolith-rs/crates/monolith-tf` for TF-runtime-backed kernels.
- Rust public API surface: `feature_utils` module exposing the same function set (or a `FeatureOps` trait with backend-specific implementations).
- Data model mapping: TF Variant / RaggedTensor → Rust `Variant`/`Ragged` equivalents (likely in `monolith-data` or `monolith-tensor`).
- Feature gating: Kafka ops, TFExample conversion, item-pool ops, and label/negative sampling depend on custom kernels; gate behind a TF backend or feature flags.
- Integration points: parsing (`parsers.py`), datasets (`datasets.py`), hooks (`item_pool_hook.py`), and training pipelines/tests.

**Implementation Steps (Detailed)**
1. Enumerate all custom ops used here and decide per-op strategy: native Rust implementation vs TF runtime binding.
2. Port validation logic exactly (asserts, `RuntimeError("params error!")`, slot range checks).
3. Provide Rust equivalents for `LineId` field metadata (from `monolith-proto`) and reuse it for `filter_by_value`/`add_action`/`multi_label_gen`.
4. Implement operand file reading for `in/not-in` filters using the same `FilterValues` proto.
5. Implement Ragged feature transforms (`feature_combine`, `switch_slot`, `switch_slot_batch`) with fid-v2 rules.
6. Add feature registry side effects equivalent to `add_feature`/`add_feature_by_fids` so parsing includes required fields.
7. Implement item pool ops or wrap TF kernels; include save/restore/check.
8. Implement label ops (`add_label`, `scatter_label`, `filter_by_label`, `label_upper_bound`, `label_normalization`, `use_field_as_label`, `multi_label_gen`) with identical label-invalid sentinel values.
9. Implement `string_to_variant` framing rules (headers, length prefix, flags, chnids/datasources) and `tf_example_to_example` conversion.
10. Add Kafka resource wrappers with poll/stream timeouts and variant conversion.
11. Add Rust tests mirroring Python expectations (see `feature_utils_test.py`) and cross-language fixtures.

**Tests (Detailed)**
- Python tests: `monolith/native_training/data/feature_utils_test.py`, `data_ops_test.py`, `eager_mode_test.py` (feature_combine/switch_slot usage).
- Rust tests: `monolith-rs/crates/monolith-data/tests/feature_utils_*` for filtering, labels, switching slots, map_id, fid mask, string_to_variant, TFExample conversion.
- Cross-language parity test: run Python test fixtures and compare Rust outputs on identical serialized inputs (including FilterValues operand files).

**Gaps / Notes**
- Heavy reliance on `gen_monolith_ops`; Rust must either re-implement kernels or use TF runtime backend (optional per earlier requirement).
- `filter_by_value` and `filter_by_feature_value` behavior is used in parallel dataset filters; caching/parallel safety must match TF op semantics.

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

### `monolith/native_training/data/feature_utils_test.py`
<a id="monolith-native-training-data-feature-utils-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 1414
- Purpose/role: End-to-end tests for feature_utils ops over `PBDataset` pipelines, including label ops, filters, slot switching, fid masks, string-to-variant framing, and negative sampling.
- Key symbols/classes/functions: `DataOpsTest`, `pb_dataset_target`, helper generators (`generate_instance`, `write_instance_into_file`), tests for `add_action`, `add_label`, `scatter_label`, `filter_by_*`, `switch_slot_batch`, `map_id`, `multi_label_gen`, `string_to_variant`, `negative_sample`.
- External dependencies: TensorFlow, `PBDataset`, `parsers` (`parse_instances`, `parse_examples`), `example_pb2.FilterValues`, `proto_parser_pb2.Instance`, temporary file IO.
- Side effects: creates/deletes temp files, writes serialized Instance files, writes FilterValues proto files, logs sample counts.

**Required Behavior (Detailed)**
- `pb_dataset_target(...)` helper:
  - Chooses input file based on `input_pb_type` (instance/example/examplebatch fixtures under `monolith/native_training/data/training_instance`).
  - Builds `PBDataset` with header flags; optional `add_action_fn`, optional filter.
  - For ExampleBatch: applies `instance_reweight` with fixed `action_priority` and `reweight` config, `variant_type` based on output.
  - Batches, parses via `parse_instance_or_example` or `parse_example_batch`, and returns a list of `return_result_key` slices.
- Action/format conversions:
  - `test_input_instance_output_instance`: actions are `[[1,0], ...]` for two batches.
  - `test_input_instance_output_instance_add_action`: adding action 2 yields `[[1,2], ...]`.
  - `test_input_instance_output_example`: output actions `[[1,0,0], ...]`.
  - `test_input_instance_output_example_add_action`: `req_time between [1622667900,1622667911]` adds action 2 in some rows.
  - `test_input_example_output_instance` and `*_add_action` mirror the above for Example input.
  - `test_input_example_output_example` and `*_add_action` expect action arrays with 3 columns.
  - `test_input_example_batch_output_instance` and `*_add_action` expect action arrays starting with `2` and optionally `3`.
  - `test_input_example_batch_output_example` and `*_add_action` expect action arrays with 3 columns.
- `test_input_instance_output_instance_add_label`:
  - Builds a temp Instance file with deterministic action patterns.
  - Applies `add_label` with config `1,2:3:1.0;4::0.5` and then `filter_by_label`.
  - Expects total valid instances in range `[340, 360]` for `mock_batch_num=100`.
- `test_input_instance_output_instance_label_upper_bound`:
  - `label_upper_bounds=[0.5,0.5]` clamps labels to `[[0,0.5], ...]`.
- `test_input_instance_output_instance_label_normalization`:
  - `norm_methods=['scale','repow']`, `norm_values=[0.5,3]` results in labels `[[0,8], ...]`.
- `test_input_examplebatch_output_instance_use_field_as_label`:
  - Uses `sample_rate` field as label; with `overwrite_invalid_value` and `label_threshold` combinations expects:
    - threshold 0 → labels `[[1,1], ...]`.
    - threshold 1.1 with prior `label_upper_bound` → labels `[[1,1], ...]`.
    - threshold 0.9 with prior `label_upper_bound` → labels `[[0,0.5], ...]`.
- `test_input_instance_output_instance_filter_by_label_equals`:
  - With `filter_equal=False`, expects 100 batches and labels `[[0,1], ...]`.
  - With `filter_equal=True`, expects 49 batches and labels `[[0,2], ...]`.
- `test_input_instance_output_instance_scatter_label`:
  - `scatter_label_config = '100:3,200:1,300:4'` and `filter_by_label` yields 2 valid instances.
  - Labels contain invalid sentinel `-3.4028235e+38` with the selected index set to original label value.
- `test_filter_by_bytes_value`:
  - Filters `req_id` using `endswith` with `filter_by_value` (LineId) and `filter_by_feature_value` (feature list).
  - Expects 4 outputs `[[b'abckjhfjh'], [b'kjhfjh'], ...]`.
  - Parallel filter path (`dataset.map(...).filter(...)`) must preserve correctness and use cached feature index.
- `test_filter_by_float_value`:
  - Filters `video_play_time > 2.5` using `filter_by_feature_value` (`field_type='float'`).
  - Expects req_id outputs `[[b'huggfyfixyz'], [b'mbzc'], ...]`.
- `test_filter_by_value_not_in`:
  - Writes `FilterValues` proto files (bytes + int64) and uses `operand_filepath` with `in/not-in`.
  - For bytes: `not-in` filters out `hello/world`, expects `excluded/300/400` (or `300/400` when using file).
  - For int64: `in` keeps chnid `[20,30,666]`, expects did values `world/excluded/400`.
  - Both `filter_by_value` and `filter_by_feature_value` must match.
- `test_filter_by_value_all`:
  - Uses `filter_by_feature_value` with `op='all'` on `chnids` list; only `did='excluded'` passes.
- `test_map_id`:
  - `map_id({123:0,456:1,789:2}, default=-1)` transforms `[123,456,789,912]` → `[0,1,2,-1]`.
- `test_filter_by_fids`:
  - Filters instances that contain both slots 2 and 3; verifies resulting ragged values match `get_fid_v1` for indices 1..4.
- `test_multi_label_gen`:
  - Builds labels based on `head_to_index` mapping and action rules; expects label vectors with INVALID_LABEL sentinel except for the matched task.
- `test_string_to_variant`:
  - Builds framed Instance bytes (with headers); one empty record allowed; `string_to_variant` preserves shape; `variant_to_zeros` callable.
- `test_has_variant`:
  - `has_variant` returns `True` for a valid variant tensor.
- `test_switch_slot_batch`:
  - `switch_slot_batch` with mix of in-place and copy-to-suffix behavior; verifies slot IDs in resulting ragged tensors (`>> 48` equals shared slot when expected).
- `test_gen_fid_mask_int64` / `test_gen_fid_mask_int32`:
  - `gen_fid_mask(ragged, fid=3)` yields `[1.,1.,0.,0.]` for both row_splits dtypes.
- `test_negative_sample_with_positive_actions`:
  - Iterates 1000 synthetic samples, applies `negative_sample` with action priority and per-action drop rates.
  - Asserts deterministic outcomes for positive labels and specific action cases; logs drop-rate ratios for matched/mismatched actions.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/tests` plus any ops crates that implement feature_utils kernels.
- Rust public API surface: tests should call Rust equivalents of `feature_utils` functions and dataset helpers.
- Data model mapping: use Rust dataset pipelines to parse the same fixtures and assert outputs.
- Feature gating: many tests require TF-runtime-backed custom ops (string/variant parsing, label ops, switch-slot, negative sampling).
- Integration points: `monolith-data` parsing, `monolith-proto` for Instance/Example, and dataset fixtures under `monolith/native_training/data/training_instance`.

**Implementation Steps (Detailed)**
1. Port helpers to Rust test utilities: serialize `Instance` protos, write framed records (headers + lengths), and parse with Rust pipelines.
2. Copy the Python expected outputs into Rust assertions (actions arrays, label arrays, invalid label sentinel values).
3. Implement FilterValues file generation in Rust to test `operand_filepath` parity.
4. Recreate dataset pipelines for each test case (including ExampleBatch `instance_reweight`).
5. Add parallel filter test to validate thread-safety/caching behavior.
6. Keep negative-sample test deterministic for mandatory branches; log or assert ratios only if stable.

**Tests (Detailed)**
- Python tests: this file (primary reference).
- Rust tests: new `feature_utils_tests.rs` with one test per Python case + helpers.
- Cross-language parity test: run Python and Rust on same temp fixtures; compare arrays and variant validity.

**Gaps / Notes**
- Many tests depend on `monolith/native_training/data/training_instance/*.pb` fixtures; ensure these are accessible to Rust tests.
- INVALID_LABEL sentinel appears as `-3.4028235e+38` in Python output; Rust must match exact float.

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

### `monolith/native_training/data/item_pool_hook.py`

<a id="monolith-native-training-data-item-pool-hook-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 109
- Purpose/role: SessionRunHook to save/restore item pool state during training.
- Key symbols/classes/functions: `ItemPoolSaveRestoreHook`.
- External dependencies: `tensorflow`, `save_item_pool`, `restore_item_pool`, `POOL_KEY`.
- Side effects: writes/reads item pool checkpoints; logs progress.

**Required Behavior (Detailed)**
- `begin()`:
  - Retrieves pools from TF collection `POOL_KEY`.
  - Creates placeholders for save/restore steps.
  - Reads checkpoint state from `model_dir`; if present, builds `restore_item_pool` op.
  - Builds `save_item_pool` op.
- `after_create_session()`:
  - If not PREDICT:
    - Reads global step, restores item pool from checkpoint (if available) using step parsed from checkpoint path.
- `after_run()`:
  - If TRAIN and `save_steps>0`: saves pool when global step advances by `save_steps`.
- `end()`:
  - If TRAIN: saves pool once more if global step advanced.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/src` (item pool ops).
- Rust public API surface: item pool save/restore hook (if training hooks exist).
- Data model mapping: TF collections → Rust registry.
- Feature gating: item pool feature.
- Integration points: training hook lifecycle.

**Implementation Steps (Detailed)**
1. Implement save/restore of item pool state in Rust.
2. Add training hook for periodic save and restore on startup.

**Tests (Detailed)**
- Python tests: `monolith/native_training/data/item_pool_test.py`.
- Rust tests: add item pool save/restore tests.
- Cross-language parity test: compare pool state serialization.

**Gaps / Notes**
- Requires TF collection `POOL_KEY` and custom item pool ops.

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

### `monolith/native_training/data/item_pool_test.py`
<a id="monolith-native-training-data-item-pool-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 58
- Purpose/role: Tests create/save/restore/check of item pool ops.
- Key symbols/classes/functions: `ItemPoolTest.test_create_item_pool`.
- External dependencies: `tensorflow`, `feature_utils` item pool ops.
- Side effects: writes item pool files under `$HOME/<user>/tmp/monolith/data/test`.

**Required Behavior (Detailed)**
- `setUpClass`:
  - Creates item pool, randomly fills, and saves to model path with `nshards=2`.
- `test_create_item_pool`:
  - Restores pool and checks it using `item_pool_check`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/tests`.
- Rust public API surface: item pool ops tests.
- Data model mapping: same save/restore workflow.
- Feature gating: item pool support.
- Integration points: data utils.

**Implementation Steps (Detailed)**
1. Add Rust unit test that saves/restores item pool and validates contents.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: add save/restore test in `monolith-rs/crates/monolith-data/tests/item_pool.rs` that mirrors pool size, shard count, and deterministic content checks.
- Cross-language parity test: generate a Python item-pool shard set and verify Rust restore yields identical items.

**Gaps / Notes**
- Requires a deterministic RNG seed to keep pool contents stable across languages.

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
