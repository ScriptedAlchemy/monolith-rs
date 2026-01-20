<!--
Source: task/request.md
Lines: 7925-8142 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/demo.py`
<a id="monolith-native-training-demo-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 57
- Purpose/role: Minimal demo entrypoint to run a local CPU training job for `TestFFMModel` with export enabled.
- Key symbols/classes/functions: `main`, CLI flags `num_ps`, `model_dir`.
- External dependencies: `cpu_training.local_train`, `TestFFMModel`, `ExportMode`.
- Side effects: launches a training run, writes checkpoints/exports to `model_dir`.

**Required Behavior (Detailed)**
- CLI flags:
  - `--num_ps`: number of parameter servers; `0` runs locally.
  - `--model_dir`: output directory.
- `main`:
  - Builds params via `TestFFMModel.params()` and sets:
    - `params.name = 'test_ffm_model'`
    - `params.train.per_replica_batch_size = 64`
    - `params.serving.export_when_saving = True`
    - `params.serving.export_mode = ExportMode.DISTRIBUTED`
  - Calls `cpu_training.local_train(..., steps=100, save_checkpoints_steps=50)`.
- Script mode: enables INFO logging and disables eager execution.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/examples` (or CLI).
- Rust public API surface: demo binary that runs a comparable CPU training flow.
- Data model mapping: params/config to Rust training config.
- Feature gating: requires training pipeline parity with Python.
- Integration points: `cpu_training` equivalent and model definition in Rust.

**Implementation Steps (Detailed)**
1. Implement a Rust demo that configures an equivalent model and training loop.
2. Mirror flags (`num_ps`, `model_dir`) and default values.
3. Ensure checkpoint/export cadence matches (`steps=100`, `save_checkpoints_steps=50`).

**Tests (Detailed)**
- Python tests: none.
- Rust tests: optional smoke test that runs a short training stub.
- Cross-language parity test: compare produced artifacts for a short run (if feasible).

**Gaps / Notes**
- Depends on `TestFFMModel` and `cpu_training` parity in Rust.

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

### `monolith/native_training/dense_reload_utils.py`
<a id="monolith-native-training-dense-reload-utils-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 457
- Purpose/role: Custom checkpoint restore logic for dense variables, including aliasing/mapping between old and new variable names and partitioned variable splitting.
- Key symbols/classes/functions: `CustomRestoreListener`, `add_mapping_rules`, `node_name`, `get_new_name`, `get_guess_name`, `split_name`, `calc_reorder_info`, `get_full_prefix`, `update_var_name_mapping_for_dense`, `infer_variable_name`, `calc_feed_dict`.
- External dependencies: TensorFlow checkpoint reader, `CheckpointRestorerListener`, `is_exporting`, numpy, regex patterns.
- Side effects: inspects checkpoint files, builds custom restore ops in graph collections, logs extensive info, may create `clear_nn` flag file logic.

**Required Behavior (Detailed)**
- Globals/regex:
  - `CUSTOM_RESTORE_OP` collection key and `CustomRestoreListenerKey` name.
  - `PAT` matches `.../part_<num>/...` for partitioned vars.
  - `DensePat` matches dense layer names for bias/kernel/trainable_kernel_norm.
  - `_NameMapping` regex rules for special-case name conversions; `add_mapping_rules` merges additional regex patterns.
- `node_name(name)`:
  - Strips whitespace, trailing `/`, leading `^`, and `:0` suffix if numeric.
- `get_new_name(name)`:
  - Deduplicates repeated path terms in a name (preserving order) and rejoins with `/`.
- `get_guess_name(name)`:
  - Applies `_NameMapping` regex patterns; returns formatted guess if matched, else original.
- `split_name(name)`:
  - Splits trailing digits; returns `(base, int_suffix)` or `(name, 0)` if none.
- `calc_reorder_info(names, is_ordered=True)`:
  - Optionally sorts by numeric suffix.
  - Returns `(need_reorder, base)` where base is `dense_` for base name `dense` else base name; `need_reorder` when suffix sequence isn't contiguous starting at 0/1 or when multiple names.
- `get_full_prefix(short_prefix, prefix_set)`:
  - Chooses the longest prefix in `prefix_set` that ends with `short_prefix`.
- `update_var_name_mapping_for_dense(var_name_mapping)`:
  - Groups dense layer vars by prefix/dense_name/bias; uses `DensePat` to normalize names.
  - For dense layers with multiple indices, may reorder and rename to `dense_{i}` or base name.
  - Ensures bias entries are present; fills missing entries into `var_name_mapping`.
- `CustomRestoreListener`:
  - `__init__`: accepts `alias_map`, `clear_nn`, `continue_training`, `model_dir`, `enable_alias_map_auto_gen` (defaults True).
  - `begin()`:
    - Skip if `is_exporting()`.
    - Loads checkpoint state from `model_dir`; sets `ckpt_name`.
    - If `clear_nn`:
      - Uses `clear_nn` flag file to skip if present.
      - Adds `global_variables_initializer` to `CUSTOM_RESTORE_OP`; if `continue_training`, adds placeholder + assign op for global_step.
    - Else if `_need_build_custom_init_graph(variables)`:
      - Creates placeholders and assign ops for each variable; stores placeholders + alias map into `CUSTOM_RESTORE_OP`.
  - `_need_build_custom_init_graph(variables)`:
    - Auto-generates alias_map when not provided and enabled:
      - Reads ckpt var names; checks compatibility by removing `/part_<n>`.
      - Builds `var_name_mapping` from `get_new_name(old_name)` to `old_name` and refines via `update_var_name_mapping_for_dense`.
      - Builds `alias_map` for each variable; handles missing dense names with `miss_dense_names` / `miss_dense_map`.
      - For unresolved names, uses `get_guess_name` or `miss_dense_map`; if still missing, logs warning and returns False.
    - Returns True if any variable name is not covered by alias_map values.
- `infer_variable_name(names)`:
  - Removes `/part_<n>` segments to infer merged variable names.
- `calc_feed_dict(ckpt, alias_map, placeholders)`:
  - Builds reverse map old_name → list of new variable names.
  - If inferred new names all exist in checkpoint, returns None (no alias restore needed).
  - Otherwise, builds feed dict mapping placeholders to ckpt tensors.
  - For partitioned vars (multiple new names):
    - Handles dense name grouping and ordering.
    - Sorts by partition index extracted via `PAT`.
    - Splits old tensor by first-dimension sizes from placeholders (`np.split`) and assigns each split to its placeholder.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/checkpoint` (restore hooks) + `monolith-checkpoint` utilities.
- Rust public API surface: custom restore listener/hook that can build alias maps and feed dicts.
- Data model mapping: checkpoint variable names → current graph variable names, including partitioned tensors.
- Feature gating: requires TensorFlow checkpoint reader or compatible reader in Rust.
- Integration points: `basic_restore_hook` and training session initialization.

**Implementation Steps (Detailed)**
1. Implement name normalization helpers (`node_name`, `get_new_name`, `split_name`, `get_guess_name`) in Rust.
2. Port dense name mapping logic (`update_var_name_mapping_for_dense`) including reorder rules and prefix resolution.
3. Implement alias-map auto generation using checkpoint metadata and dense mappings.
4. Build custom restore ops/feeds with placeholders and assign ops; support `clear_nn` + `continue_training` global step update.
5. Implement partitioned variable splitting logic equivalent to `calc_feed_dict`.

**Tests (Detailed)**
- Python tests: `dense_reload_utils_test.py`.
- Rust tests: unit tests for name mapping, alias generation, and feed dict splitting.
- Cross-language parity test: use a sample ckpt with renamed vars and ensure alias restore works identically.

**Gaps / Notes**
- Heavy TF internals: requires checkpoint reader and graph variable manipulation in Rust.
- Auto alias mapping may be fragile; parity requires matching regex and reorder heuristics exactly.

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

### `monolith/native_training/dense_reload_utils_test.py`
<a id="monolith-native-training-dense-reload-utils-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 192
- Purpose/role: Tests for dense reload utilities: variable name inference, feed dict splitting for partitioned vars, and custom restore listener modes.
- Key symbols/classes/functions: `DenseReloadUtilsTest`, `setUpClass`, `test_infer_variable_name`, `test_calc_feed_dict`, `test_alias_map_listener`, `test_clear_nn_listener`.
- External dependencies: TensorFlow, `GlorotNormal`, `Ones`, `infer_variable_name`, `calc_feed_dict`, `CustomRestoreListener`.
- Side effects: creates and deletes checkpoint files under `./ckpt`.

**Required Behavior (Detailed)**
- `setUpClass`:
  - Builds a graph with `global_step`, a partitioned variable `partition` (shape 1280x512), and `small_var`.
  - Saves checkpoint `ckpt/test-<global_step>` in cwd.
- `tearDownClass`:
  - Removes `./ckpt` directory if exists.
- `test_infer_variable_name`:
  - Creates a partitioned variable and checks `infer_variable_name` removes `/part_xx` to yield `{partition_var.name:0}`.
- `test_calc_feed_dict`:
  - Creates partitioned `partition2` and `small_var2`.
  - Builds `alias_map` mapping new names to old checkpoint names (`small_var2` → `small_var`, `partition2 parts` → `partition`).
  - Creates placeholders with `origin_name` for each var/partition.
  - `calc_feed_dict` returns mapping for each alias; asserts shapes match partition shapes.
- `test_alias_map_listener`:
  - Builds same alias_map/placeholders and calls `CustomRestoreListener(alias_map=..., model_dir=./ckpt).begin()` (no asserts, just should not error).
- `test_clear_nn_listener`:
  - Creates `CustomRestoreListener(clear_nn=True, model_dir=./ckpt)` and calls `begin()`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests`.
- Rust public API surface: dense reload utilities and custom restore listener.
- Data model mapping: checkpoint vars/partitioned vars to Rust checkpoint reader and feed dict logic.
- Feature gating: requires checkpoint reader and graph variable introspection.
- Integration points: `dense_reload_utils.py` implementation.

**Implementation Steps (Detailed)**
1. Build Rust tests that create a checkpoint with partitioned variables (or mock the reader).
2. Verify `infer_variable_name` removes partition suffixes.
3. Validate `calc_feed_dict` splitting behavior for partitioned variables.
4. Ensure custom restore listener handles alias_map and clear_nn without error.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: `dense_reload_utils_test.rs` analog with temp directories.
- Cross-language parity test: compare feed dict splits on a shared checkpoint.

**Gaps / Notes**
- The Python tests rely on TF checkpoint creation; Rust tests may need to use Python-generated checkpoints.

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
