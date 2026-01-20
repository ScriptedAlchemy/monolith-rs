<!--
Source: task/request.md
Lines: 7225-7431 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/data/training_instance/python/instance_negative_gen_dataset_op_test.py`
<a id="monolith-native-training-data-training-instance-python-instance-negative-gen-dataset-op-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 283
- Purpose/role: Tests negative sample generation dataset (`negative_gen` and `InstanceNegativeGenDataset`) over Instance PB data, including per-channel ring buffer behavior.
- Key symbols/classes/functions: `InsNegativeDatasetTest`, `parse1`, `testNegativeGen`, `testRingBufferCache`, `testIgnoreReaNegInstance`, `testUseNegInstance`.
- External dependencies: TensorFlow, `PBDataset`, `InstanceNegativeGenDataset`, `PbType`, custom parse ops `parse_variant_instances`.
- Side effects: reads fixture `monolith/native_training/data/training_instance/instance.pb`.

**Required Behavior (Detailed)**
- Constants:
  - `FILE_NAME` fixture file for Instance PB.
  - `CHANNEL_SLOT=357`, `GROUP_SLOTS=[200..242]`, `LABEL_FIELD='actions'`, `LABEL_INDEX=0`.
  - Negative labels `NEGATIVE_LABEL=-2`, `NEGATIVE_LABEL2=-1`.
  - `GID='gid'` used as misc int64 feature.
- `parse1(pb_variant)`:
  - Uses a fixed `FIDV1_FEATURES` list and `parse_variant_instances` with `misc_int64_features=[GID]`.
- `testNegativeGen`:
  - Builds `PBDataset` (Instance → Instance) with headers; applies `dataset.negative_gen` with:
    - `neg_num=7`, `channel_slot`, `group_slots`, `per_channel_sample=True`, `start_num=0`, `max_group_num_per_channel=10000`, `label_field='actions'`, `label_index=0`, `negative_label=-2`, `use_neg_ins=True`.
  - Batches 8 and parses.
  - Asserts in first batch that `channel_res[0][0] == channel_res[0][i]` for i in 1..7 (negatives share channel), and label at index 1 equals `NEGATIVE_LABEL`.
- `testRingBufferCache`:
  - Same negative_gen config except `max_group_num_per_channel=2`.
  - Collects ~1024 samples; groups by channel and verifies ring buffer behavior:
    - For channels with >2 samples, checks that group fids from later samples are not present in the first sample when gids differ.
  - Logs `valid_count` of checked non-overlapping group features.
- `testIgnoreReaNegInstance`:
  - First applies `dataset.negative_gen(..., negative_label=-2, use_neg_ins=True)`.
  - Then wraps with `InstanceNegativeGenDataset(..., negative_label=-1, use_neg_ins=False)`.
  - Asserts label at index 1 equals `NEGATIVE_LABEL2` (real negatives ignored).
- `testUseNegInstance`:
  - Same as previous but `use_neg_ins=True` in wrapper.
  - Asserts labels: index1/index2 are `NEGATIVE_LABEL2`, index3/index4 are `NEGATIVE_LABEL` (mix of generated vs real negatives).

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/tests` with negative_gen dataset implementation in `monolith-data`.
- Rust public API surface: `negative_gen` dataset operator and `InstanceNegativeGenDataset` wrapper.
- Data model mapping: Instance variant streams with channel/group fid slots and label field `actions`.
- Feature gating: requires negative generation ops + item pool or group cache implementation.
- Integration points: dataset pipeline in `datasets.py` and negative-gen custom ops in Rust/TF backend.

**Implementation Steps (Detailed)**
1. Implement `negative_gen` dataset operator and wrapper in Rust with identical parameters.
2. Ensure per-channel sampling and ring buffer cache behavior match Python semantics.
3. Expose `use_neg_ins` toggle to include/exclude existing negatives.
4. Add Rust tests that load the same fixture PB file and verify label/channel/group constraints.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: new `instance_negative_gen_dataset_op_test.rs` mirroring each test case.
- Cross-language parity test: compare label distributions and channel/group assignments on identical fixtures.

**Gaps / Notes**
- Depends on `instance.pb` fixture and custom ops; ensure Rust has compatible dataset and parse op support.

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

### `monolith/native_training/data/training_instance/python/parse_instance_ops.py`
<a id="monolith-native-training-data-training-instance-python-parse-instance-ops-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 245
- Purpose/role: Instance parsing helpers that wrap custom ops to extract fid and dense features into Ragged/Dense tensors, including LineId fields and repeated fields handling.
- Key symbols/classes/functions: `_parse_instance_impl`, `parse_instances2`, `parse_instances`, `monolith_raw_parse_instance`.
- External dependencies: TensorFlow ragged internals (`RowPartition`), `gen_monolith_ops` custom kernels, `get_slot_feature_name`, parser_utils hooks (imported but not used here).
- Side effects: Builds RaggedTensor with precomputed row ids/nrows; uses default misc feature lists.

**Required Behavior (Detailed)**
- `_parse_instance_impl(serialized, fidv1_features, fidv2_features, float_features, float_feature_dims, int64_features, int64_feature_dims, string_features, string_feature_dims, misc_float_features, misc_float_dims, misc_int64_features, misc_int64_dims, misc_string_features, misc_string_dims, cc_op)`:
  - Normalizes all list args to empty lists if None.
  - Calls `cc_op` (custom op) with counts `N/M/O/P/Q/R/S` and all feature lists/dims.
  - Builds `ragged_keys` from `fidv1_features` (via `get_slot_feature_name`) plus `fidv2_features`.
  - For each ragged split/value pair, constructs `RowPartition` with precomputed `value_rowids` and `nrows`, then `tf.RaggedTensor(values, row_partition, internal=True)`.
  - Returns dict mapping ragged + float + int64 + string + misc_* features to their tensors in order.
- `parse_instances2(...)`:
  - Thin wrapper that calls `_parse_instance_impl` with `parse_instance_ops.monolith_parse_instances`.
- `parse_instances(...)`:
  - Adds defaults: `misc_float_features=['sample_rate']`, `misc_int64_features=['req_time','uid']`, `misc_repeated_float_features=['label']`.
  - Normalizes list args to empty lists and sets default dims (1) for misc features.
  - Calls `parse_instances2` with concatenated misc+repeated feature lists/dims.
  - Reshapes non-repeated misc float/int64 features to 1-D (`tf.reshape(features[key], [-1])`).
  - Returns feature dict.
- `monolith_raw_parse_instance`:
  - Exposes `parse_instance_ops.MonolithRawParseInstance` for testing only.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/src` for instance parsing; ragged support in `monolith-tensor`.
- Rust public API surface: `parse_instances`/`parse_instances2` equivalents returning `HashMap<String, Tensor>` with ragged types.
- Data model mapping: custom op outputs (splits/values) → Rust ragged tensors with cached row ids/nrows.
- Feature gating: depends on custom parsing kernels or TF runtime.
- Integration points: dataset parsing pipelines, tests in `parse_instance_ops_test.py` and other training_instance tests.

**Implementation Steps (Detailed)**
1. Implement Rust wrappers for `monolith_parse_instances` (or bind to TF op) that return splits/values arrays.
2. Build ragged tensors with cached row metadata to match TF `RowPartition` behavior.
3. Match default misc feature lists and reshape semantics in `parse_instances`.
4. Preserve feature key ordering in output map to match downstream expectations.

**Tests (Detailed)**
- Python tests: `parse_instance_ops_test.py`, `instance_dataset_op_test_stdin.py`, `instance_negative_gen_dataset_op_test.py`.
- Rust tests: parser unit tests for ragged vs dense outputs; ensure misc defaults applied.
- Cross-language parity test: parse a fixture instance and compare fid/dense outputs.

**Gaps / Notes**
- Uses TF internal ragged APIs; Rust must provide equivalent row-partition caching to avoid perf regressions.

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

### `monolith/native_training/data/training_instance/python/parse_instance_ops_test.py`
<a id="monolith-native-training-data-training-instance-python-parse-instance-ops-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 185
- Purpose/role: Validates instance parsing ops and ragged encoding helpers, including missing field defaults and raw parse concat behavior.
- Key symbols/classes/functions: `RaggedEncodingHelperTest`, `ParseInstancesTest`, `RawParseInstanceTest`, helper `generate_instance`, `make_fid_v1`, `make_fid_v2`.
- External dependencies: TensorFlow, `proto_parser_pb2.Instance`, `parse_instance_ops`, `parser_utils.RaggedEncodingHelper`.
- Side effects: none.

**Required Behavior (Detailed)**
- `generate_instance()` builds an Instance with:
  - fidv1 list `[make_fid_v1(i,i) for i in range(10)]`.
  - fidv2 feature `name='fidv2'` with `make_fid_v2(100,i)`.
  - float feature `ue` length 16 (i * 1e-5), int64 feature `int64_feature=100`, string feature `string_feature='test_string'`.
  - label `[1.1,2.2,3.3]`, line_id fields: uid=110, sample_rate=0.5, req_time=64, actions=[0,100], user_id='123'.
- `RaggedEncodingHelperTest.testExpandContract`:
  - Builds a ragged tensor, expands with `RaggedEncodingHelper.expand(..., with_precomputed_value_rowids=True)` and verifies `value_rowids` equals TF-computed.
  - `contract` returns original ragged values and preserves cached value_rowids.
- `ParseInstancesTest.testParseInstance`:
  - Calls `parse_instances2` with explicit fidv1/fidv2/float/int64/string/misc fields and dims.
  - Asserts:
    - 10 fidv1 slots returned (`slot_*`).
    - `slot_1` uses fid_v2 encoding for v1 slot values.
    - `fidv2` ragged equals `get_test_fidv2()`.
    - dense features: `int64_feature=[[100]]`, `string_feature=[[b'test_string']]`, `ue` length 16, `sample_rate=[[0.5]]`, `label=[[1.1,2.2,3.3]]`, `uid=[[110]]`, `actions=[[0,100]]`, `user_id=[['123']]`.
- `ParseInstancesTest.testParseInstanceV1Only`:
  - `parse_instances2` with `fidv1_features=[1]` yields `slot_1` with fid_v1 encoding.
- `ParseInstancesTest.testParseInstanceWithMissingFields`:
  - Requests extra missing fields; expects:
    - Missing ragged fid slot → empty ragged (`[[]]`).
    - Missing fidv2 → empty ragged.
    - Missing float → zeros of specified dim.
    - Missing int64/string → zeros/empty strings of specified dim.
- `RawParseInstanceTest.test_concat`:
  - Calls `monolith_raw_parse_instance` with `fid_output_type='CONCAT'`.
  - Expects first tensor offsets `[0,1,2,len(fidv2)+2]` and second tensor concatenated fids `[fidv1 slot0, fidv1 slot1] + fidv2 list`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/tests` and ragged utils in `monolith-tensor`.
- Rust public API surface: ragged encoding helper, `parse_instances2`, raw parse op if exposed.
- Data model mapping: same fid encoding rules (v1/v2), missing-field defaults.
- Feature gating: raw parse op requires custom kernel support.
- Integration points: parse_instance_ops implementation and parser_utils utilities.

**Implementation Steps (Detailed)**
1. Implement ragged encoding helper in Rust and verify cached rowids/nrows behavior.
2. Port `parse_instances2` tests with the same synthetic Instance fixture.
3. Ensure missing fields return empty ragged or zero-filled dense tensors as specified.
4. If raw parse op is supported, add concat mode test for offsets + fid list order.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: new `parse_instance_ops_test.rs` matching each test case.
- Cross-language parity test: compare parsed feature dicts for identical serialized Instance.

**Gaps / Notes**
- `slot_*` fidv1 encoding differs between v1-only path and v2 conversion; Rust must replicate both behaviors.

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
