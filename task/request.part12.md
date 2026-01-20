<!--
Source: task/request.md
Lines: 3164-3441 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/core/base_embedding_task.py`
<a id="monolith-core-base-embedding-task-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 611
- Purpose/role: Base class for TPU embedding tasks; builds input pipeline, vocab sizing, and TPU embedding configs.
- Key symbols/classes/functions: `BaseEmbeddingTask.params`, `create_input_fn`, `_post_process_example`, `create_feature_and_table_config_dict`, `process_features_for_cpu_test`.
- External dependencies: `tensorflow.compat.v1`, `tpu_embedding`, `FeatureSlot/FeatureColumn`, `auto_checkpoint_feed_hook`, `util`.
- Side effects: reads vocab file (possibly from HDFS), constructs TF datasets and embedding configs.

**Required Behavior (Detailed)**
- `params()` defines many embedding-related flags, including vocab sizing, QR hashing, deepinsight, host call metrics, file input ranges, and stopping signals.
- `__init__`:
  - Sets flags, builds vocab dict (from file or downloaded from HDFS), constructs `Env`.
- `download_vocab_size_file_from_hdfs()`:
  - Downloads a single `part*.csv` from HDFS into temp folder; updates `p.vocab_file_path` if successful.
- `_create_vocab_dict()`:
  - Reads vocab file (tsv slot_id -> count), applies overrides and offsets, returns dict.
- `create_input_fn(mode)`:
  - Only supports TRAIN.
  - If `file_pattern` provided, uses `tf.data.Dataset.list_files` (no shuffle); else uses `file_folder` + `date_and_file_name_format` and `util.range_dateset` with `start_date/end_date`.
  - Shards files per TPU host call index.
  - Interleaves files with `cycle_length` and parses examples via `_get_feature_map` and `_post_process_example`.
  - If `enable_stopping_signals`, appends a final batch with stop signal flag via `auto_checkpoint_feed_hook`.
  - Prefetches with AUTOTUNE.
- `_post_process_example`:
  - Converts embedding tensors to SparseTensor; applies vocab size mods, QR hashing, and FeatureColumn3D row_lengths.
  - Adds UID bucket for AUC sampling if `_UID` exists.
- `create_feature_and_table_config_dict()`:
  - Builds `tpu_embedding.TableConfig` and `FeatureConfig` per slot/feature slice, including QR hashing tables.
- `process_features_for_cpu_test()`:
  - Creates embedding variables with random init and uses `safe_embedding_lookup_sparse`.
  - Clears internal feature/table config dicts after processing.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/src/base_embedding_task.rs`.
- Rust public API surface: base embedding task trait/struct with dataset and embedding config helpers.
- Data model mapping: TF dataset pipeline and TPU embedding configs (likely via TF runtime bridge).

**Implementation Steps (Detailed)**
1. Port parameter schema and vocab file parsing.
2. Implement dataset pipeline generation or equivalent data loader.
3. Recreate embedding config generation and QR hashing logic.
4. Implement CPU test path for embeddings.

**Tests (Detailed)**
- Python tests: none direct; used by TPU runner tests.
- Rust tests: unit tests for vocab dict creation and QR hashing config creation.

**Gaps / Notes**
- Full parity depends on TF TPU embedding APIs; Rust likely needs a bridge or compatibility layer.

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

### `monolith/core/base_host_call.py`
<a id="monolith-core-base-host-call-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 145
- Purpose/role: Collects tensors for TPU host calls, with compression/decompression by dtype.
- Key symbols/classes/functions: `BaseHostCall.record_summary_tensor`, `compress_tensors`, `decompress_tensors`.
- External dependencies: `tensorflow`, `absl.logging`.
- Side effects: builds internal tensor lists; uses global_step tensor.

**Required Behavior (Detailed)**
- `__init__(output_dir, enable_host_call)`:
  - Initializes `_tensor_names` with `"global_step"` and `_tensors` with reshaped global step.
  - `_lists_tensor_sizes` tracks original sizes for decompression.
- `record_summary_tensor(name, tensor)`:
  - No-op if host call disabled.
  - Asserts unique name, asserts tensor rank <= 1, reshapes to `[-1]`, appends.
- `compress_tensors()`:
  - Groups tensors by dtype; concatenates each dtype list along axis 0, then `expand_dims` to add batch dimension.
  - Stores per-group tensor sizes in `_lists_tensor_sizes`.
  - Replaces `_tensor_names` and `_tensors` with compressed versions.
- `decompress_tensors(tensors)`:
  - Splits each compressed tensor by recorded sizes; squeezes to 1D.
  - Asserts first tensor name is `global_step` (error message uses first character only).
  - Returns `(global_step_scalar, decompressed_tensor_list)` where global step is `decompressed_tensor_list[0][0]`.
- `generate_host_call_hook()` default returns `None` (to be overridden).

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/src/base_host_call.rs`.
- Rust public API surface: `BaseHostCall` struct with tensor collection and (de)compression.

**Implementation Steps (Detailed)**
1. Implement tensor collection and dtype grouping logic.
2. Preserve compression/decompression shapes and per-dtype grouping semantics.
3. Mirror global_step positioning and return shape.

**Tests (Detailed)**
- Python tests: none specific (used indirectly by embedding host call tests).
- Rust tests: unit tests for compress/decompress roundtrip.

**Gaps / Notes**
- Error message in assert uses `self._tensor_names[0][0]` (first char); keep for parity.

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

### `monolith/core/base_layer.py`
<a id="monolith-core-base-layer-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 161
- Purpose/role: Base class for layers with child management, name assignment, and per-graph layer loss tracking.
- Key symbols/classes/functions: `BaseLayer`, `get_uname`, `add_layer_loss`, `get_layer_loss`.
- External dependencies: `tensorflow`, `InstantiableParams`, `NestedMap`.
- Side effects: global registries `_layer_loss` and `_name_inuse` are mutated.

**Required Behavior (Detailed)**
- `params()`:
  - Returns `InstantiableParams(cls)` and defines `name` using `get_uname(cls.__name__)`.
- `__init__(params)`:
  - Asserts `params.name` is set.
  - Initializes `_private_children` as `NestedMap`.
- `children` property returns `_private_children`.
- `__getattr__`:
  - Raises AttributeError if `_private_children` not created.
  - If `name` in children, returns it.
  - If class has a property with that name, calls the property's getter to surface the same AttributeError.
  - Else raises `"<name> is not a sub-layer of <self>."`.
- `__call__` forwards to `fprop()`.
- `fprop()` is abstract: raises `NotImplementedError('Abstract method of %s' % self)`.
- `create_child(name, params)`:
  - If `params.name` empty, assigns `self.p.name` (assumes BaseLayer has `p` attribute set by InstantiableParams).
  - Instantiates child via `params.instantiate()` and stores under `_private_children[name]`.
- `create_children(name, params_list)`:
  - Creates list; for each param, sets `param.name = f"{name}_{index}"` if missing; instantiates and appends.
- `get_uname(name)`:
  - Uses `_name_inuse` defaultdict; **note**: code checks membership but never inserts, so it currently always returns `name` (no suffix) unless keys are inserted elsewhere.
- `add_layer_loss(name, loss)`:
  - Adds loss into `_layer_loss` keyed by default graph and layer name.
- `get_layer_loss()`:
  - Returns the dict for the current default graph.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/src/base_layer.rs`.
- Rust public API surface: `BaseLayer` trait/struct, child map, `get_uname`, layer loss registry.
- Data model mapping: `NestedMap` equivalent and parameter instantiation mechanism.

**Implementation Steps (Detailed)**
1. Implement a base layer trait with child management and `fprop` entrypoint.
2. Provide a `NestedMap` equivalent with attribute-like access.
3. Mirror `get_uname` behavior (including current no-op uniqueness unless explicitly fixed).
4. Implement per-graph loss registry or document deviation if Rust lacks TF graphs.

**Tests (Detailed)**
- Python tests: `monolith/core/base_layer_test.py`
- Rust tests: verify create_child/create_children and `__getattr__`-like behavior.

**Gaps / Notes**
- BaseLayer relies on `self.p` being set by hyperparams instantiation; mirror this in Rust or document.

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

### `monolith/core/base_layer_test.py`
<a id="monolith-core-base-layer-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 41
- Purpose/role: Tests child creation APIs in `BaseLayer`.
- Key symbols/classes/functions: `BaseLayerTest.test_create_child`, `.test_create_children`.
- External dependencies: `base_layer`.
- Side effects: none.

**Required Behavior (Detailed)**
- `test_create_child`:
  - Instantiates BaseLayer params, sets name, creates layer, calls `create_child`, and asserts child exists.
- `test_create_children`:
  - Creates two child layers and asserts list length is 2.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/tests/base_layer.rs`.
- Rust public API surface: BaseLayer child creation.

**Implementation Steps (Detailed)**
1. Port tests using Rust BaseLayer + params instantiation.
2. Assert child map/list presence and lengths.

**Tests (Detailed)**
- Python tests: this file
- Rust tests: same assertions.

**Gaps / Notes**
- Python tests access `_disable_create_child` which is not defined in BaseLayer; confirm if Rust needs similar flag (likely no-op).

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

### `monolith/core/base_model_params.py`
<a id="monolith-core-base-model-params-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 25
- Purpose/role: Defines abstract params holder for single-task models.
- Key symbols/classes/functions: `SingleTaskModelParams.task`.
- External dependencies: none.
- Side effects: none.

**Required Behavior (Detailed)**
- `SingleTaskModelParams.task()` is abstract and must be overridden to return task params; raises `NotImplementedError('Abstract method')` by default.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-core/src/base_model_params.rs`.
- Rust public API surface: trait or struct with `task()`/`task_params()` abstract method.

**Implementation Steps (Detailed)**
1. Create a Rust trait for model params with `task()` returning task config.
2. Ensure errors are explicit when called on base type.

**Tests (Detailed)**
- Python tests: none specific.
- Rust tests: optional compile-time trait enforcement or runtime panic test.

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
