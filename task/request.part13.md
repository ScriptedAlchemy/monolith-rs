<!--
Source: task/request.md
Lines: 3442-3680 (1-based, inclusive)
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
  - Validates merged slot dims and embedding dims in runner task env.

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
- `params()`:
  - Defines units, activation, use_bias, kernel/bias initializers, kernel norm options, and partitioner.
- `__init__`:
  - Initializes BaseLayer and tf.keras.layers.Dense with given params.
  - Sets attributes: `allow_kernel_norm`, `kernel_norm_trainable`, `var_name_prefix`, `partitioner`.
- `build(input_shape)`:
  - Validates dtype (float/complex).
  - Uses `VarianceScaling` initializer to create kernel; uses `tf.compat.v1.get_variable` (partitioner optional).
  - If `allow_kernel_norm`: L2-normalizes kernel and optionally multiplies by trainable norm variable.
  - Creates bias if `use_bias`.
- `get_config()` merges base config with custom fields.
- `fprop(inputs)` calls `self.call(inputs)`.

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
- `test_dense_instantiate`: runs `layer_test` for different input shapes.
- `test_dense_dtype`: ensures output dtype is float32 when specified.
- `test_dense`: checks output shape and runs session to initialize vars.
- `test_dense_with_partitioner`: ensures Dense works with variable partitioner.

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
