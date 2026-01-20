<!--
Source: task/request.md
Lines: 6465-6684 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
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
- Rust tests: TODO (manual)
- Cross-language parity test: TODO (manual)

**Gaps / Notes**
- TODO (manual)

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
