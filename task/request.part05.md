<!--
Source: task/request.md
Lines: 8388-11812 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/data/tf_example_to_example_test.py`
<a id="monolith-native-training-data-tf-example-to-example-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 183
- Purpose/role: End-to-end test that converts TF Example records to Monolith Example variants via `tf_example_to_example`, then parses with `parse_examples` and asserts fid/dense defaults.
- Key symbols/classes/functions: `serialize_example`, `get_fid_v2`, `calc_hash_value`, `TFExampleToExampleTest.test_tf_example_to_example`.
- External dependencies: TensorFlow TFRecord, numpy RNG, `tf_example_to_example`, `parse_examples`.
- Side effects: writes `/tmp/test.tfrecord` with 10k TF Examples; uses TF1 session.

**Required Behavior (Detailed)**
- Helper functions:
  - `_bytes_feature`, `_float_feature`, `_int64_feature` wrap values into `tf.train.Feature` list types.
  - `serialize_example(feature0, feature1, feature2, feature3, feature4)` builds a `tf.train.Example` with:
    - `feature0`: int64 (bool values allowed)
    - `feature1`: int64
    - `feature2`: bytes
    - `feature3`: float
    - `feature4`: float
  - `get_fid_v2(slot, signature)` uses `fid_v2_mask=(1<<48)-1` and returns `(slot<<48) | (signature & mask)`.
  - `calc_hash_value(val)` returns `int(log2(abs(val)+1))`.
- `test_tf_example_to_example`:
  - Disables TF2 behavior (`tf.compat.v1.disable_v2_behavior()`), uses TF1 session graph.
  - Generates 10k samples:
    - `feature0`: random bools
    - `feature1`: random ints in [0,4]
    - `feature2`: bytes from `strings[feature1]`
    - `feature3`: random normal float
    - `feature4`: random normal float
  - Writes TFRecord file `/tmp/test.tfrecord` with serialized Examples.
  - Dataset pipeline:
    - `TFRecordDataset` → `map(tf_example_to_example)` with:
      - `sparse_features={'feature0':1,'feature1':2,'feature4':3}` (fid_v2 slots)
      - `dense_features=['feature2']`
      - `label='feature3'`
      - `instance_weight=None`
    - Batch size 2 → `map(parse_examples)` with:
      - `sparse_features=['not_existed1','feature0','feature1','feature4']`
      - `dense_features=['label','feature2','feature3','not_existed2','instance_weight']`
      - `dense_feature_types=[float32,string,float32,float32,float32]`
      - `dense_feature_shapes=[1,1,1,1,1]`
  - In session loop (5k batches):
    - `not_existed1` ragged has zero values.
    - `feature0/feature1` fids equal `get_fid_v2(slot, original int/bool)` per batch.
    - `feature4` fid uses slot 3 and `calc_hash_value` of float value (log2(abs(val)+1)).
    - `label` equals original `feature3` (float) per batch.
    - `feature3` dense output is `[0,0]` (missing in conversion), `not_existed2` is `[0,0]`.
    - `instance_weight` defaults to `[1.0,1.0]`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/tests` (TFExample conversion) + `monolith-proto` for Example parsing.
- Rust public API surface: test helper for TF Example serialization; conversion op `tf_example_to_example` or equivalent.
- Data model mapping: TF Example bytes → Monolith Example variant → parsed feature dict.
- Feature gating: TFRecord read/write and TFExample conversion backend.
- Integration points: `feature_utils.tf_example_to_example` and `parsers.parse_examples` parity.

**Implementation Steps (Detailed)**
1. Implement TF Example serialization helper in Rust tests (or load TFRecord fixtures generated in Python).
2. Provide `tf_example_to_example` conversion in Rust with the same slot/fid-v2 behavior and hashing for float feature4.
3. Ensure missing sparse features emit empty ragged values; missing dense features emit zeros; `instance_weight` defaults to 1.0.
4. Add a Rust test that mirrors batch size 2 with deterministic input, validating fid values and dense defaults.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: `tf_example_to_example.rs` (new) that asserts identical fid/dense outputs.
- Cross-language parity test: generate a fixed TFRecord in Python and run Rust conversion+parse on it; compare outputs.

**Gaps / Notes**
- Uses `/tmp/test.tfrecord`; Rust tests should use tempdir paths.
- The hash for float sparse feature4 is `int(log2(abs(val)+1))`; must match exactly.

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

### `monolith/native_training/data/training_instance/python/instance_dataset_op.py`
<a id="monolith-native-training-data-training-instance-python-instance-dataset-op-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 166
- Purpose/role: TF DatasetSource wrapper around custom `instance_dataset` op for reading serialized Instance records from PB files or stdin, with optional sharding/interleave utilities.
- Key symbols/classes/functions: `_PBInstanceDataset`, `PBInstanceDatasetV2`, `create_instance_dataset`, alias `PBInstanceDataset`.
- External dependencies: TensorFlow Dataset internals, `gen_monolith_ops.instance_dataset`, `distributed_dataset.create_dynamic_sharding_dataset`, `ckpt_hooks.disable_iterator_save_restore`, TF matching_files.
- Side effects: disables iterator save/restore when reading from stdin; logs initialization; uses TF fatal logging on missing file.

**Required Behavior (Detailed)**
- `_PBInstanceDataset(file_name, use_snappy, has_sort_id, kafka_dump, kafka_dump_prefix)`:
  - Calls custom op `instance_dataset` with tensors for file name, snappy, and header flags.
  - `element_spec` is scalar string `TensorSpec([], tf.string)`.
- `PBInstanceDatasetV2`:
  - If `file_name` is empty string, treats input as stdin and calls `ckpt_hooks.disable_iterator_save_restore()`.
  - Creates `_PBInstanceDataset` internally and forwards variant tensor into `DatasetV2`.
  - `_clone` merges kwargs with stored defaults.
  - `_inputs()` returns `[]`.
- `create_instance_dataset(...)`:
  - `files_list=None` defaults to `['']` (stdin).
  - If a single file and no glob expansion/sharding/dynamic sharding, returns `PBInstanceDatasetV2` directly; validates existence when file is non-empty and logs fatal on missing file.
  - `enable_dynamic_sharding=True`:
    - Converts to dataset via `distributed_dataset.create_dynamic_sharding_dataset` and `flat_map` with `PBInstanceDatasetV2`.
  - `enable_sharding=True`:
    - Requires a single file pattern; uses `MatchingFilesDataset`, shards by `shard_num/shard_index`, logs shard info; forces `use_snappy=True`.
  - Else:
    - Uses `MatchingFilesDataset` if `expand_glob_path=True`, otherwise `Dataset.from_tensor_slices`.
  - Final dataset uses `interleave` with `cycle_length`, `block_length`, `num_parallel_calls`, `deterministic=False`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/src` for dataset creation and file reader.
- Rust public API surface: `pb_instance_dataset` (or similar) and `create_instance_dataset` with matching options.
- Data model mapping: custom op variant tensor → Rust stream of serialized Instance bytes.
- Feature gating: stdin mode, dynamic sharding, and TF MatchingFiles behavior.
- Integration points: datasets (`datasets.py`), training pipelines that expect PBInstanceDataset semantics.

**Implementation Steps (Detailed)**
1. Implement a Rust dataset source that wraps Instance file reading with flags for sort_id/kafka headers and snappy.
2. Mirror stdin special case and disable iterator save/restore in Rust equivalents.
3. Implement glob expansion, sharding, and dynamic sharding (or document unsupported) with identical defaults.
4. Preserve interleave behavior and `deterministic=False` semantics.

**Tests (Detailed)**
- Python tests: `instance_dataset_op_test_stdin.py`, other dataset tests using PBInstanceDataset.
- Rust tests: dataset source tests for stdin vs file, sharding path, missing file handling.
- Cross-language parity test: read a fixture PB file in Python and Rust and compare record sequence.

**Gaps / Notes**
- Uses TF Dataset internals; Rust must define a similar streaming abstraction.
- Missing file uses `logging.fatal` in TF; decide equivalent behavior in Rust (panic or error).

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

### `monolith/native_training/data/training_instance/python/instance_dataset_op_test_stdin.py`
<a id="monolith-native-training-data-training-instance-python-instance-dataset-op-test-stdin-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 58
- Purpose/role: Smoke test for `PBInstanceDataset` reading from stdin (empty file_name), batching, and parsing with `parse_instances`.
- Key symbols/classes/functions: `PBInstanceDataset`, `parse_instances`, `testInstanceDataset`.
- External dependencies: TensorFlow v1 session, `instance_dataset_ops`, `parse_instance_ops`.
- Side effects: expects stdin data stream; logs warnings; runs one batch read.

**Required Behavior (Detailed)**
- Defines feature lists:
  - `FIDV1_FEATURES = [1..9]`
  - `FIDV2_FEATURES = ['fc_360d_ml_convert_cid', 'fc_360d_ml_convert_advertiser_id']`
  - `FLOAT_FEATURES = ['fc_muse_finish_rough_10168_uid_d128']` with dim `[128]`
  - `INT64_FEATURES = ['fc_dense_external_action']` with dim `[1]`
- `parse(serialized)` calls `parse_instances(serialized, fidv1, fidv2, float_feats, float_dims, int64_feats, int64_dims)`.
- `testInstanceDataset()`:
  - Creates `PBInstanceDataset(file_name='', has_sort_id=True, kafka_dump_prefix=True)` (stdin path).
  - `batch(32)` and `map(parse)`.
  - Builds one-shot iterator, fetches one batch, logs `elements['sample_rate']`.
- Script mode: disables eager and runs `testInstanceDataset()`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/tests`.
- Rust public API surface: stdin dataset reader + parse_instances in Rust.
- Data model mapping: stream of framed Instance records from stdin.
- Feature gating: stdin support in dataset source; parse_instances.
- Integration points: dataset source in `instance_dataset_op.py` parity.

**Implementation Steps (Detailed)**
1. Add a Rust test that simulates stdin input (e.g., pipe fixture data into the dataset reader).
2. Ensure parsing handles fidv1/fidv2 and dense features as in Python.
3. Validate batch size 32 returns expected keys including `sample_rate`.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: add stdin dataset smoke test with a small fixture.
- Cross-language parity test: compare parsed batch fields from the same stdin fixture.

**Gaps / Notes**
- This test assumes stdin provides valid Instance records; Rust tests should supply a controlled fixture stream.

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

### `monolith/native_training/data/training_instance/python/parser_utils.py`
<a id="monolith-native-training-data-training-instance-python-parser-utils-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 85
- Purpose/role: Utilities for parser pipelines, including queued extra-parse steps and ragged encoding expansion/contract helpers.
- Key symbols/classes/functions: `_extra_parse_steps`, `add_extra_parse_step`, `RaggedEncodingHelper.expand`, `RaggedEncodingHelper.contract`, `advanced_parse`.
- External dependencies: TensorFlow, `ragged_utils.fused_value_rowids`.
- Side effects: mutates global deque of extra parse steps; mutates RaggedTensor internal row partition caches during `contract`.

**Required Behavior (Detailed)**
- `_extra_parse_steps`:
  - Global `deque` used to store parse step callables.
- `add_extra_parse_step(parse_fn)`:
  - Appends parse_fn to `_extra_parse_steps`.
- `RaggedEncodingHelper.expand(name_to_ragged_ids, with_precomputed_nrows=True, with_precomputed_value_rowids=False)`:
  - For each RaggedTensor value, returns a dict with:
    - `values`, `row_splits`, optional `nrows` (if flag), optional `value_rowids` computed via `ragged_utils.fused_value_rowids` (if flag).
  - Non-ragged entries pass through unchanged.
- `RaggedEncodingHelper.contract(name_to_ragged_ids)`:
  - For dict entries with `values` and `row_splits`, rebuilds `tf.RaggedTensor.from_row_splits(..., validate=False)`.
  - If `nrows` present, asserts `_row_partition._nrows` is None before assigning.
  - If `value_rowids` present, asserts `_row_partition._value_rowids` is None before assigning.
  - Non-dict entries pass through unchanged.
- `advanced_parse(features)`:
  - Pops parse steps from `_extra_parse_steps` in FIFO order and applies each to `features`.
  - Returns final features dict.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/src` (parser utilities) + `monolith-tensor` for ragged.
- Rust public API surface: `add_extra_parse_step` and `advanced_parse` equivalents; ragged expand/contract helpers.
- Data model mapping: RaggedTensor internal encodings → Rust ragged structure with cached rowids/nrows.
- Feature gating: none.
- Integration points: `parse_instance_ops_test.py` uses `RaggedEncodingHelper`.

**Implementation Steps (Detailed)**
1. Implement a global queue of parse steps (with proper synchronization if used across threads).
2. Implement ragged expand/contract; ensure cached rowids/nrows are set only once.
3. Mirror `fused_value_rowids` behavior using Rust ragged utilities.

**Tests (Detailed)**
- Python tests: `parse_instance_ops_test.py` (`RaggedEncodingHelperTest`).
- Rust tests: add unit tests that expand, contract, and verify rowids/nrows caching.
- Cross-language parity test: compare ragged values and cached rowids against Python output.

**Gaps / Notes**
- Directly mutates internal ragged partition caches; Rust must provide an equivalent escape hatch.

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

### `monolith/native_training/data/training_instance/python/pb_datasource_ops.py`
<a id="monolith-native-training-data-training-instance-python-pb-datasource-ops-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 48
- Purpose/role: Thin wrappers around `gen_monolith_ops` for filtering and negative sampling on variant tensors in training_instance pipelines.
- Key symbols/classes/functions: `filter_by_fids`, `filter_by_value`, `negative_sample`, `variant_dummy`.
- External dependencies: TensorFlow, `gen_monolith_ops` custom kernels.
- Side effects: none beyond custom op invocation.

**Required Behavior (Detailed)**
- `filter_by_fids(variant, filter_fids, has_fids, select_fids, has_actions)`:
  - Passes list args (defaults to empty) to `pb_datasource_ops.set_filter`.
- `filter_by_value(variant, field_name, op, operand)`:
  - Calls `pb_datasource_ops.value_filter` with given field/op/operand.
- `negative_sample(variant, drop_rate, label_index, threshold)`:
  - Calls `pb_datasource_ops.negative_sample` with drop/threshold params.
- `variant_dummy(variant)`:
  - Calls `pb_datasource_ops.variant_dummy`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/src` (ops wrappers).
- Rust public API surface: minimal wrappers for filtering/negative sampling on variant streams.
- Data model mapping: variant tensor → Rust variant representation.
- Feature gating: custom op availability (TF backend).
- Integration points: training_instance datasets and tests.

**Implementation Steps (Detailed)**
1. Add Rust wrapper functions that call the underlying kernel backend.
2. Ensure default empty list behavior matches Python.
3. Expose in public API for dataset pipelines.

**Tests (Detailed)**
- Python tests: indirectly via `instance_negative_gen_dataset_op_test.py`.
- Rust tests: minimal unit tests for wrappers if backend available.
- Cross-language parity test: verify behavior using fixed fixtures.

**Gaps / Notes**
- This is a thin wrapper; underlying op semantics are defined in C++/TF kernels.

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

### `monolith/native_training/data/training_instance/python/test_data_utils.py`
<a id="monolith-native-training-data-training-instance-python-test-data-utils-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 15
- Purpose/role: Placeholder test utility module; currently only imports TensorFlow.
- Key symbols/classes/functions: none.
- External dependencies: TensorFlow.
- Side effects: none.

**Required Behavior (Detailed)**
- No runtime behavior beyond importing TensorFlow.

**Rust Mapping (Detailed)**
- Target crate/module: none.
- Rust public API surface: none.
- Data model mapping: none.
- Feature gating: none.
- Integration points: none.

**Implementation Steps (Detailed)**
1. No Rust port needed unless file is expanded in Python.

**Tests (Detailed)**
- Python tests: none.
- Rust tests: none.
- Cross-language parity test: none.

**Gaps / Notes**
- File is effectively empty; keep an eye on future changes.

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

### `monolith/native_training/data/transform/transforms.py`
<a id="monolith-native-training-data-transform-transforms-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 250
- Purpose/role: Declarative transform objects that serialize to `TransformConfig` proto for dataset transform pipelines (filters, label generation, logical composition).
- Key symbols/classes/functions: `Transform` (ABC), `Compose`, `FilterByFid`, `FilterByAction`, `FilterByLabel`, `FilterByValue`, `AddLabel`, `LogicalOr`.
- External dependencies: `transform_config_pb2`, `LineId` proto descriptor for field validation.
- Side effects: none; purely builds proto configs with validation asserts.

**Required Behavior (Detailed)**
- `Transform` abstract base:
  - `as_proto()` returns `transform_config_pb2.TransformConfig`.
  - `_is_leaf_node()` distinguishes leaf vs composite.
- `Compose(transforms)`:
  - Requires all items are `Transform` instances.
  - `as_proto` merges each transform’s proto into a single `TransformConfig` via `MergeFrom` in order.
  - `_is_leaf_node` returns False.
- `FilterByFid(has_fids, filter_fids, select_fids)`:
  - `as_proto` appends a `basic_config.filter_by_fid` entry with respective lists.
  - `_is_leaf_node` True.
- `FilterByAction(has_actions)`:
  - Adds `basic_config.filter_by_action.has_actions`.
  - `_is_leaf_node` True.
- `FilterByLabel(thresholds)`:
  - Adds `basic_config.filter_by_label.thresholds`.
  - `_is_leaf_node` True.
- `FilterByValue(field_name, op, operand, keep_empty=False)`:
  - Validates `op` in `{gt,ge,eq,lt,le,neq,between,in,not-in,all,any,diff,startswith,endswith}`.
  - Validates `field_name` exists in `LineId` descriptor; `operand` is not None.
  - Infers operand type based on field cpp_type and op:
    - Repeated fields (`field.has_options`): only `all/any/diff` allowed; only integer types; operand int or list of int.
    - Float/double: `between` uses list; otherwise single float.
    - Int types: `in/not-in/between` use list; otherwise single int.
    - String: operand is str or list of str; else `RuntimeError("params error!")`.
  - Stores `float_operand`, `int_operand`, `string_operand`, `keep_empty`.
  - `as_proto` fills `basic_config.filter_by_value` with operands and flags.
  - `_is_leaf_node` True.
- `AddLabel(config, negative_value, new_sample_rate)`:
  - Parses config `pos_actions:neg_actions:sample_rate` separated by `;` (skips empty parts).
  - Adds `basic_config.add_label` with negative value + new sample rate and a `task_label_config` entry per task.
  - `_is_leaf_node` True.
- `LogicalOr(x, y)`:
  - Requires both `x` and `y` are leaf nodes.
  - `as_proto` creates `logical_or_config` and copies `basic_config` from each side.
  - `_is_leaf_node` False.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/src` (transform config builders).
- Rust public API surface: `Transform` trait + concrete structs mirroring Python class names; `as_proto()` to `TransformConfig`.
- Data model mapping: transform structs → `transform_config_pb2::TransformConfig`.
- Feature gating: none; pure config serialization.
- Integration points: `TransformDataset` op uses serialized config (see `datasets.py`).

**Implementation Steps (Detailed)**
1. Create Rust transform trait with `as_proto` and `is_leaf` methods.
2. Implement concrete transforms with identical validation (asserts or Result errors).
3. Preserve `Compose` merge ordering and `LogicalOr` leaf-only requirement.
4. Implement `FilterByValue` operand parsing based on LineId descriptor in Rust.

**Tests (Detailed)**
- Python tests: `transforms_test.py`.
- Rust tests: unit tests for each transform’s proto encoding and validation.
- Cross-language parity test: serialize configs in Python and Rust and compare bytes.

**Gaps / Notes**
- `FilterByValue` uses `LineId` field metadata to infer types; Rust must mirror the same descriptor mapping.

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

### `monolith/native_training/data/transform/transforms_test.py`
<a id="monolith-native-training-data-transform-transforms-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 70
- Purpose/role: Smoke tests that build transform configs and log the resulting protobufs; no assertions beyond successful construction.
- Key symbols/classes/functions: `TransformsTest`, test methods for each transform type.
- External dependencies: `transforms` module, `absl.logging`, `unittest`.
- Side effects: logs serialized proto configs.

**Required Behavior (Detailed)**
- `test_filter_by_fid`: builds `FilterByFid(has_fids=[1], filter_fids=[2,3], select_fids=None)` and logs proto.
- `test_filter_by_action`: builds `FilterByAction(has_actions=[4])` and logs proto.
- `test_filter_by_label`: builds `FilterByLabel(thresholds=[-100, -100])` and logs proto.
- `test_add_label`: builds `AddLabel(config='1,2:3:1.0;4::0.5', negative_value=0.0, new_sample_rate=0.3)` and logs proto.
- `test_logical_or`: builds `LogicalOr(FilterByAction([1,2]), FilterByFid([10000000]))` and logs proto.
- `test_compose`: builds `Compose([...])` with multiple transforms including `LogicalOr`, logs proto.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/tests`.
- Rust public API surface: transform builders and `as_proto` serialization.
- Data model mapping: Rust transforms → TransformConfig protobufs.
- Feature gating: none.
- Integration points: verifies transform config serialization used by `TransformDataset`.

**Implementation Steps (Detailed)**
1. Add Rust tests that construct equivalent transforms and ensure `as_proto` succeeds.
2. Optionally compare serialized proto bytes to Python output for deterministic configs.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: `transforms_test.rs` with equivalent constructions.
- Cross-language parity test: serialize each config in both languages and compare bytes.

**Gaps / Notes**
- Tests are smoke-only; Rust should at least mirror construction and serialization.

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

### `monolith/native_training/data/utils.py`
<a id="monolith-native-training-data-utils-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 55
- Purpose/role: Simple slot/feature-name helpers for training_instance parsing; global mapping of feature names to slots with TOB env toggle.
- Key symbols/classes/functions: `enable_tob_env`, `get_slot_feature_name`, `get_slot_from_feature_name`, `register_slots`, globals `TOBENV`, `USED_FREATUE_NAMES`, `NAME_TO_SLOT`.
- External dependencies: none.
- Side effects: mutates global dictionaries for name/slot mapping.

**Required Behavior (Detailed)**
- Globals:
  - `TOBENV` default `False` toggles slot name prefix (`slot_` vs `fc_slot_`).
  - `USED_FREATUE_NAMES` maps arbitrary feature names to assigned slot ids (incrementing).
  - `NAME_TO_SLOT` maps feature name → slot id (explicit).
- `enable_tob_env()`:
  - Sets `TOBENV = True` globally.
- `get_slot_feature_name(slot)`:
  - Returns `"fc_slot_{slot}"` if `TOBENV` else `"slot_{slot}"`.
- `get_slot_from_feature_name(feature_name)`:
  - If in `NAME_TO_SLOT`, return mapped slot.
  - Else if name starts with `slot_` or `fc_slot_`, parse suffix int; return int or `None` if non-numeric.
  - Else use `USED_FREATUE_NAMES`: assign a new slot id (`len+1`) if missing and return it.
- `register_slots(sparse_features)`:
  - Accepts list/tuple of ints or dict name→slot.
  - For list: asserts ints and converts to dict via `get_slot_feature_name`.
  - Updates `NAME_TO_SLOT` with provided mapping.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/src` (feature utils).
- Rust public API surface: slot/feature-name mapping utilities with global registry or context-bound mapping.
- Data model mapping: feature names → slots used by parsing/feature extraction.
- Feature gating: TOB env toggle.
- Integration points: `parse_instance_ops` (fidv1 slot naming), feature list parsing.

**Implementation Steps (Detailed)**
1. Implement a global or context-local registry for `NAME_TO_SLOT` and `USED_FEATURE_NAMES` with deterministic assignment.
2. Provide `enable_tob_env` toggle and `get_slot_feature_name` logic.
3. Mirror `get_slot_from_feature_name` fallback behavior for unknown names.
4. Implement `register_slots` with list/dict handling and type checks.

**Tests (Detailed)**
- Python tests: none explicit.
- Rust tests: add unit tests for TOB/non-TOB naming and deterministic slot assignment.
- Cross-language parity test: compare mapping outputs for a fixed sequence of names.

**Gaps / Notes**
- Uses global mutable state; Rust must be careful about concurrency or test isolation.

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

### `monolith/native_training/debugging/debugging_client.py`
<a id="monolith-native-training-debugging-debugging-client-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 98
- Purpose/role: CLI client to query debugging server endpoints for variable values or feature embeddings.
- Key symbols/classes/functions: `main`, CLI flags `type`, `variable_names`, `feature_ids`, `feature_name`, `feature_names`.
- External dependencies: `requests`, `json`, protobuf `text_format`, `embedding_hash_table_pb2.EntryDump`.
- Side effects: HTTP POSTs to local debugging server; logs results; raises exceptions on invalid flag combos.

**Required Behavior (Detailed)**
- Flags:
  - `--type` must be `debugging_variables` or `debugging_features`.
  - `--variable_names` list for variable lookup.
  - `--feature_ids` list for feature lookup.
  - `--feature_name` single name to pair with all ids.
  - `--feature_names` list of names; must be same length as `feature_ids` if provided.
- `debugging_variables` flow:
  - If `variable_names` empty → log and return.
  - POST JSON `{"variable_names": [...]}` to `http://127.0.0.1:<port>/debugging/variables`.
  - Response JSON contains `STATUS`, `SUCCESS/FAIL`, `MSG` keys; on FAIL log reason and return.
  - `MSG` is JSON-encoded dict name→value; logs each variable value or "Not exist".
- `debugging_features` flow:
  - Disallow providing both `feature_name` and `feature_names`.
  - If `feature_ids` empty → log and return.
  - If `feature_name` set, expand to list same length as ids.
  - Validate `len(feature_names) == len(feature_ids)` else raise.
  - POST JSON `{"feature_names": [...], "feature_ids": [...]}` to `/debugging/features`.
  - On FAIL log reason and return.
  - `MSG` is JSON-encoded dict name→id→textproto of `EntryDump`.
  - If present, parse textproto into `EntryDump` and log; else log "Not exist".
- Script mode: sets logging verbosity INFO, disables eager, and runs app.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/debugging` (or CLI crate).
- Rust public API surface: CLI command for debugging server queries.
- Data model mapping: JSON request/response; `EntryDump` textproto parsing.
- Feature gating: requires debugging server running locally.
- Integration points: `debugging_server.py` endpoints.

**Implementation Steps (Detailed)**
1. Implement a Rust CLI that mirrors flags and validation.
2. POST to `/debugging/variables` and `/debugging/features` with identical JSON payloads.
3. Parse response JSON; for features, parse textproto into `EntryDump` (protobuf text format parser).
4. Match logging output patterns and error handling (exceptions for invalid flags).

**Tests (Detailed)**
- Python tests: none.
- Rust tests: integration tests with a mocked debugging server (or golden responses).
- Cross-language parity test: compare outputs against Python client for same server responses.

**Gaps / Notes**
- Depends on `requests` and protobuf text parsing; Rust needs equivalent libraries.

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

### `monolith/native_training/debugging/debugging_server.py`
<a id="monolith-native-training-debugging-debugging-server-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 217
- Purpose/role: Flask server that exposes debugging endpoints to fetch variable values and feature embeddings from a running training cluster using saved graph metadata.
- Key symbols/classes/functions: `DebuggingWorker`, `create_app`, `/debugging/variables`, `/debugging/features`, `main`.
- External dependencies: Flask, TensorFlow server/meta_graph import, `debugging_info_pb2`, `embedding_hash_table_pb2`, custom ops `monolith_hash_table_lookup_entry`.
- Side effects: spins up TF server, reads model_dir debugging info, imports meta graph, starts Flask server.

**Required Behavior (Detailed)**
- Flags:
  - `--host`, `--port`, `--model_dir` required to bind server and load debugging info.
- `DebuggingWorker(model_dir)`:
  - Reads `DebuggingInfo` proto from `utils.get_debugging_info_file_name(model_dir)`.
  - Starts a local TF server and builds a fake worker cluster where the last worker is the local server; chief/ps addresses come from debugging info.
  - Builds `feature_name_config_map` from debugging info and initializes a `MergedMultiTypeHashTable` with a dummy factory.
  - Creates session config via `cluster_manager.generate_session_config` and imports the meta graph from `utils.get_meta_graph_file_name(model_dir)`.
- `fetch_variables(variable_names)`:
  - Filters requested variables to those present in imported graph; returns dict name → stringified value.
- `fetch_features(feature_names, feature_ids)`:
  - Maps feature names to merged table names via `slot_mapping` in merged table.
  - Buckets by PS index using `fid % num_ps`.
  - For each table/ps index, fetches EntryDump via `monolith_hash_table_lookup_entry`.
  - Parses EntryDump bytes to text proto and returns dict `feature_name -> {fid: textproto}`.
- `create_app()`:
  - Constructs Flask app and a `DebuggingWorker`.
  - `/debugging/variables`: expects JSON `variable_names`; returns `{status, msg}` with msg JSON string.
  - `/debugging/features`: expects JSON `feature_names` + `feature_ids` (same length); returns `{status, msg}`.
  - On exceptions, returns `status=fail` with traceback in msg.
- `main`:
  - Calls `env_utils.setup_hdfs_env()` and runs Flask app.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/debugging` (server) + TF backend bindings for graph/variables.
- Rust public API surface: debugging server binary exposing `/debugging/variables` and `/debugging/features`.
- Data model mapping: DebuggingInfo proto, EntryDump parsing, table lookup by PS index.
- Feature gating: requires TF runtime + custom ops for hash table lookup.
- Integration points: debugging_client CLI, model_dir metadata generation.

**Implementation Steps (Detailed)**
1. Implement a Rust server (e.g., axum/warp) with matching endpoints and JSON payloads.
2. Load DebuggingInfo and meta graph; create TF session with cluster config matching Python.
3. Implement variable lookup and table lookup logic with PS sharding by `fid % num_ps`.
4. Return responses with identical JSON structure and error handling.

**Tests (Detailed)**
- Python tests: none.
- Rust tests: integration tests with mocked debugging info + TF session (if feasible).
- Cross-language parity test: compare responses for same model_dir and queries.

**Gaps / Notes**
- Requires TF runtime and custom ops; Rust implementation may need to shell out or depend on TF C API.

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

### `monolith/native_training/device_utils.py`
<a id="monolith-native-training-device-utils-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 231
- Purpose/role: Device placement utilities for training/serving, including GPU gating, device functions, and MPI/PS placement logic.
- Key symbols/classes/functions: `enable_gpu_training`, `disable_gpu_training`, `is_gpu_training`, `get_visible_gpus`, `default_device_fn`, `maybe_device_if_allowed`, `within_placement_context_of`, `get_device_fn`, `input_device_fn`, `model_device_fn`, `serving_input_device_fn`, `skip_device`.
- External dependencies: TensorFlow DeviceSpec, `device_setter`, MPI rank helper `get_mpi_rank`, flags (`num_ps`, `enable_gpu_training`, `enable_sync_training`, `is_local`).
- Side effects: global `_GPU_PLACEMENT_ALLOWED` flag; influences device placement for ops.

**Required Behavior (Detailed)**
- GPU training flag:
  - `_GPU_PLACEMENT_ALLOWED` default False; `enable_gpu_training()` sets True; `disable_gpu_training()` sets False; `is_gpu_training()` returns it.
- `get_visible_gpus(local_rank, processes_per_gpu=1)`:
  - Ensures `processes_per_gpu` is int >= 1; returns string of `local_rank / processes_per_gpu` as GPU index.
- `_device_rule(device_name)`:
  - Returns `/device:CPU:0` when `device_name` is empty.
  - If assigned GPU but `_GPU_PLACEMENT_ALLOWED` is False or device type empty, merges with default CPU while keeping job/task/replica.
- `skip_device(op)`:
  - Returns True for summary ops (`Write*`, `*Summary`) or string `Const` ops.
- `default_device_fn(op)`:
  - Returns CPU for skipped ops; otherwise applies `_device_rule` to op device.
- `maybe_device_if_allowed(device_name)`:
  - Context manager that forces device via `_device_rule` to prevent unintended GPU placement.
- Placement context helpers:
  - `_FakeOp` and `within_placement_context_of(device_name)` check current placement via graph `_apply_device_functions`.
- `get_device_fn(cluster=None, task=None)`:
  - Determines MPI mode via `OMPI_COMM_WORLD_LOCAL_RANK`.
  - Chooses GPU vs CPU based on `FLAGS.enable_gpu_training` or `_GPU_PLACEMENT_ALLOWED`.
  - If sync training + MPI + PS: builds device spec for chief/worker based on rank and returns custom `_device_fn` that merges with op device.
  - If sync training but no PS: returns `default_device_fn`.
  - If async (no sync training):
    - Returns None for local mode or missing cluster/task.
    - Else uses `tf.compat.v1.train.replica_device_setter` with `ps_tasks=FLAGS.num_ps` and standard PS ops.
- `input_device_fn(op)`:
  - In MPI+PS+sync training returns `/job:chief|worker/replica:0/task:<idx>/device:CPU:0`, else CPU.
- `model_device_fn(op)`:
  - Similar to `_device_fn` but for model scope; uses GPU if enabled, else CPU; respects op.device and `_class` attr.
- `serving_input_device_fn(op)`:
  - Uses op.device if set, else CPU.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/device`.
- Rust public API surface: device placement utilities and device function factories.
- Data model mapping: TF DeviceSpec strings → Rust device strings used by TF runtime bindings.
- Feature gating: GPU placement gate, MPI/PS sync training, replica device setter behavior.
- Integration points: training config, session creation, input pipelines.

**Implementation Steps (Detailed)**
1. Implement GPU gating and visible GPU computation.
2. Implement device rule merging logic with default CPU and job/task retention.
3. Provide Rust equivalents of `get_device_fn`/`input_device_fn`/`model_device_fn` for sync/async modes.
4. Mirror skip-device rules for summary ops and string const.
5. Add placement-context helper or document unsupported if TF internals unavailable.

**Tests (Detailed)**
- Python tests: `device_utils_test.py`.
- Rust tests: unit tests for device rules, GPU gating, and MPI/PS device fn outputs.
- Cross-language parity test: compare device string outputs under fixed flag/env combinations.

**Gaps / Notes**
- Depends on TF internal device functions; Rust may need to mimic device strings rather than enforcing in graph.

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

### `monolith/native_training/device_utils_test.py`
<a id="monolith-native-training-device-utils-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 104
- Purpose/role: Tests device placement rules and GPU gating in `device_utils`.
- Key symbols/classes/functions: `DeviceUtilsTest` and methods `test_basic`, `test_cpu_only`, `test_str_context`, `test_str_nested_contexts`, `test_cpu_device_merge`, `test_gpu_device_merge`, `test_process_gpu_map`.
- External dependencies: TensorFlow, `device_utils`.
- Side effects: none.

**Required Behavior (Detailed)**
- `test_basic`: default device function places constants on `/device:CPU:0`.
- `test_cpu_only`: when GPU training disabled, explicit GPU device request is overridden to CPU.
- `test_str_context`: with GPU enabled, bare constants default to CPU, `tf.device("GPU:0")` forces GPU:0.
- `test_str_nested_contexts`: nested device contexts maintain correct placement for CPU/GPU overrides.
- `test_cpu_device_merge`: with GPU disabled, device job/task merged with CPU; `within_placement_context_of` reports CPU.
- `test_gpu_device_merge`: with GPU enabled, device job/task merged with GPU; `maybe_device_if_allowed` forces GPU:1 placement and context checks.
- `test_process_gpu_map`: `get_visible_gpus` returns expected indices for local_rank/processes_per_gpu combinations.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests`.
- Rust public API surface: device_utils functions.
- Data model mapping: device strings matching TF conventions.
- Feature gating: GPU training toggle.
- Integration points: training device placement.

**Implementation Steps (Detailed)**
1. Add Rust tests to assert device string outputs for each scenario.
2. Verify GPU gating overrides explicit GPU placement when disabled.
3. Validate visible GPU mapping logic.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: `device_utils_test.rs`.
- Cross-language parity test: compare device string outputs and placement context behavior.

**Gaps / Notes**
- Python tests rely on TF device placement; Rust tests may need to compare string outputs rather than actual TF graph placement.

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

### `monolith/native_training/distribute/distributed_dataset.py`
<a id="monolith-native-training-distribute-distributed-dataset-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 81
- Purpose/role: Builds a dynamic sharding dataset that expands glob patterns on demand using shared queues and a TF session-backed generator.
- Key symbols/classes/functions: `create_dynamic_sharding_dataset`.
- External dependencies: `str_queue.StrQueue`, `session_hooks.get_current_session`, `utils.ps_device`, `native_task_context`.
- Side effects: creates shared queues on PS0 or host; uses TF session to dequeue filenames.

**Required Behavior (Detailed)**
- `create_dynamic_sharding_dataset(glob_patterns, name)`:
  - Creates two shared string queues:
    - `glob_patterns_queue`: seeded with glob patterns.
    - `filenames_queue`: auto-enqueue filenames by expanding patterns.
  - Chooses device on PS0 if `num_ps > 0`, else default device.
  - `glob_pattern()` (tf.function): dequeues a pattern; if not out_of_range, calls `tf.io.matching_files`; else returns `""` and out_of_range.
  - `filenames_queue.dequeue()` returns `(filename_bytes, out_of_range)`.
  - `filename_generator()` runs dequeue via current session; raises `StopIteration` on out_of_range; else decodes bytes to string.
  - Builds `dataset_ops.MapDataset` over a dummy infinite dataset; maps to `tf.py_function(filename_generator)` with `preserve_cardinality=False`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/src`.
- Rust public API surface: dynamic sharding dataset builder for file patterns.
- Data model mapping: file pattern → stream of file paths.
- Feature gating: requires session hooks/queues or Rust equivalents.
- Integration points: `datasets.py` uses this for dynamic sharding.

**Implementation Steps (Detailed)**
1. Implement a Rust dynamic sharding iterator that expands patterns lazily.
2. Support shared queue semantics for multi-worker coordination (or document limitation).
3. Ensure out_of_range yields end of stream and map preserves non-cardinality.

**Tests (Detailed)**
- Python tests: `distributed_dataset_test.py`.
- Rust tests: unit tests for pattern expansion order and termination.
- Cross-language parity test: compare file lists produced for a given glob set.

**Gaps / Notes**
- Relies on TF session and custom `StrQueue`; Rust needs a coordinated queue for distributed use.

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

### `monolith/native_training/distribute/distributed_dataset_test.py`
<a id="monolith-native-training-distribute-distributed-dataset-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 124
- Purpose/role: Tests dynamic sharding dataset expansion, EOF handling, composition with TextLineDataset, and iterator save/restore.
- Key symbols/classes/functions: `DynamicShardingDatasetTest`, `gen_test_files`, `testBasic`, `testEof`, `testWithOtherDataset`, `testSaveRestore`.
- External dependencies: TensorFlow, `distributed_dataset.create_dynamic_sharding_dataset`, `session_hooks.SetCurrentSessionHook`.
- Side effects: writes temp files under `TEST_TMPDIR` and saves/loads iterator checkpoints.

**Required Behavior (Detailed)**
- `gen_test_files(files_dir)`:
  - Creates files `a_0.txt`..`e_1.txt` with two lines each: `a.0.0`, `a.0.1`, etc.
- `setUp`:
  - Uses `TEST_TMPDIR` and creates data dir + files if missing.
  - Builds glob patterns `a_*.txt`..`e_*.txt`.
- `get_test_session()`:
  - Returns `SingularMonitoredSession` with `SetCurrentSessionHook`.
- `testBasic`:
  - Reads 10 filenames from dynamic sharding dataset; expects ordered list of `a_0..e_1` full paths.
- `testEof`:
  - With empty patterns, iterator should raise `OutOfRangeError`; verifies dependent op does not mutate variable `v`.
- `testWithOtherDataset`:
  - `filename_dataset.flat_map(TextLineDataset)` yields lines; first three lines are `a.0.0`, `a.0.1`, `a.1.0`.
- `testSaveRestore`:
  - Creates saveable iterator; reads `a.0.0`, saves; reads `a.0.1`, restores; next read is still `a.0.1`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/tests`.
- Rust public API surface: dynamic sharding dataset + iterator save/restore (if supported).
- Data model mapping: file list → dataset → line dataset.
- Feature gating: iterator save/restore may depend on TF runtime.
- Integration points: `distributed_dataset` implementation.

**Implementation Steps (Detailed)**
1. Implement Rust tests that generate temp files and validate ordered filename emission.
2. Verify EOF behavior for empty patterns.
3. Test composition with line reader and save/restore semantics.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: `distributed_dataset_test.rs` with tempdir fixtures.
- Cross-language parity test: compare file order and resume position after restore.

**Gaps / Notes**
- `saveable` iterator behavior is TF-specific; Rust may need explicit checkpointing support.

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

### `monolith/native_training/distribute/str_queue.py`
<a id="monolith-native-training-distribute-str-queue-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 114
- Purpose/role: A TF-based string queue with save/restore support, critical section synchronization, and optional auto-enqueue when empty.
- Key symbols/classes/functions: `StrQueue`, `enqueue_many`, `dequeue`, `_raw_enqueue_many`, `_raw_dequeue`.
- External dependencies: TensorFlow `CriticalSection`, variables, tf.function.
- Side effects: maintains internal TF variables (`_arr`, `_offset`, `_arr_size`), uses critical section for synchronization.

**Required Behavior (Detailed)**
- `StrQueue.__init__(initial_elements, critical_section, auto_enqueue_fn, capacity, name)`:
  - Creates a shared `CriticalSection` (or reuses provided).
  - Initializes `_arr` (string array of size `capacity`), `_offset`, `_arr_size` as variables.
  - Enqueues `initial_elements` during initialization via control deps.
  - Uses `_var_for_init` dummy variable to ensure initial enqueue runs.
- `enqueue_many(elements)`:
  - Converts to string tensor and calls `_raw_enqueue_many` inside critical section.
- `dequeue()`:
  - Executes `_raw_dequeue` inside critical section; returns `(element, out_of_range)`.
- `_raw_enqueue_many(elements)` (tf.function):
  - Computes `old_arr_size = _arr_size - _offset`, `new_arr_size = old_arr_size + size(elements)`.
  - Asserts `new_arr_size <= capacity`.
  - Compacts array by shifting remaining elements to front, appends new elements, resets `_offset` to 0, updates `_arr_size`.
- `_raw_dequeue()` (tf.function):
  - Asserts `_offset <= _arr_size`.
  - If `auto_enqueue_fn` provided, loops while empty: calls auto fn to get `(elements, out_of_range)`; enqueues elements unless out_of_range.
  - If still empty, returns `("", True)`.
  - Else returns element at `_offset` and increments `_offset`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/src` (distributed queues).
- Rust public API surface: string queue with enqueue/dequeue and optional auto-fill.
- Data model mapping: TF variables → Rust in-memory or shared queue state.
- Feature gating: requires distributed synchronization if used across workers.
- Integration points: `distributed_dataset.create_dynamic_sharding_dataset`.

**Implementation Steps (Detailed)**
1. Implement a thread-safe queue with capacity and offset/size semantics.
2. Provide auto-enqueue hook that is called when empty.
3. Match out_of_range behavior and empty return value (`""`).
4. If using TF runtime, preserve CriticalSection semantics for shared state.

**Tests (Detailed)**
- Python tests: `str_queue_test.py`.
- Rust tests: queue enqueue/dequeue, auto-enqueue loop, capacity assert.
- Cross-language parity test: compare sequence of dequeued elements for the same auto-enqueue function.

**Gaps / Notes**
- TF CriticalSection semantics may need a custom mutex + barrier if implemented in Rust.

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

### `monolith/native_training/distribute/str_queue_test.py`
<a id="monolith-native-training-distribute-str-queue-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 67
- Purpose/role: Tests basic enqueue/dequeue behavior, initialization, out-of-range handling, and auto-enqueue logic for `StrQueue`.
- Key symbols/classes/functions: `QueueTest` with `testBasic`, `testInit`, `testOutOfRange`, `testAutoEnqueue`.
- External dependencies: TensorFlow, `str_queue.StrQueue`.
- Side effects: none.

**Required Behavior (Detailed)**
- `testBasic`:
  - Enqueues `test1`, `test2` and dequeues in order.
- `testInit`:
  - Initializes queue with `initial_elements=['test1']` and dequeues `test1`.
- `testOutOfRange`:
  - Dequeue from empty queue returns `out_of_range=True`.
- `testAutoEnqueue`:
  - `auto_enqueue` increments variable `v` and enqueues stringified values until `v > 2`, then returns out_of_range.
  - Dequeues yield `"1"`, `"2"`, then `out_of_range=True` for subsequent dequeues.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/tests`.
- Rust public API surface: `StrQueue` equivalent.
- Data model mapping: string queue semantics.
- Feature gating: none.
- Integration points: `distributed_dataset` uses StrQueue.

**Implementation Steps (Detailed)**
1. Add Rust tests that validate enqueue/dequeue ordering and init behavior.
2. Implement auto-enqueue hook test with controlled counter.
3. Verify out_of_range behavior persists after exhaustion.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: `str_queue_test.rs`.
- Cross-language parity test: compare dequeued sequences for fixed auto-enqueue behavior.

**Gaps / Notes**
- TensorFlow session semantics are not required; Rust can implement a pure queue test.

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

### `monolith/native_training/distributed_ps.py`
<a id="monolith-native-training-distributed-ps-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 2108
- Purpose/role: Distributed parameter server embedding hash table implementation (single-type and multi-type), sharded lookup/apply gradients, fused layout embedding pipelines, and GPU/Horovod/BytePS all-to-all paths.
- Key symbols/classes/functions: `ps_device`, `DistributedHashTable`, `DistributedMultiTypeHashTable`, `PartitionedHashTable`, `get_sub_table_name`, `PartitionedHashTable.gen_feature_configs`, `merge_feature_config`, `lookup`, `apply_gradients`, `_lookup_gpu`, `_apply_gradients_gpu`.
- External dependencies: TensorFlow, custom ops (`distribution_ops`, `multi_hash_table_ops`), `export_context`, `prefetch_queue`, `hvd`/`bps` (if enabled), FeatureConfigs protos.
- Side effects: creates PS-side graphs/signatures during export; emits lookup timers; enqueues prefetch queues; uses global parser context for sharding configs.

**Required Behavior (Detailed)**
- `ps_device(i)`:
  - Context manager that clears device stack (`colocate_with(None, True)`) and sets device to `utils.ps_device(i)` for PS ops.
- `DistributedHashTable` (single-type table):
  - Constructor builds per-PS tables; sends learning-rate tensors to each PS (unless exporting standalone).
  - `lookup(ids)`:
    - `tf.unique` ids, shard by `id % ps_num`, lookup on each PS, and `map_id_to_embedding` back to original order.
    - Tracks input/output tensors for backprop.
  - `assign/assign_add`: split ids/values by PS and call underlying table method.
  - `apply_gradients`: unique ids, split gradients with `map_id_to_embedding_gradient_back_prop`, apply on each PS (dedup disabled).
  - `as_op`: aggregates PS table ops.
- `DistributedMultiTypeHashTable` (multi-slot table):
  - Builds per-PS multi-type tables; supports raw API if tables are `RawMultiTypeHashTable`.
  - Export mode builds PS subgraphs with `lookup` and optional `raw_lookup` signatures.
  - `lookup(slot_to_id)`:
    - Raw API path: uses ragged IDs and `unique_key_with_value_and_offset` to reduce duplicate lookups, splits by PS, uses raw lookup and `fill_with_offset_map`, then reconstructs embeddings; returns only requested slots.
    - Non-raw path: per-slot unique/split, PS lookup (remote_predict if exporting distributed); maps back by `map_id_to_embedding`.
  - `assign/assign_add`: per-slot split by PS and call underlying table methods.
  - `reinitialize(slot, ids)`: raw-only; splits ids and concatenates status.
  - `apply_gradients`: raw path uses `raw_apply_gradients` with fused flat grads; non-raw path packs keyed tensors, optional float16 transfer.
  - `as_op` combines PS tables; `get_table_dim_sizes` delegates to cc dims.
- `get_sub_table_name(strs)`:
  - Returns `(concat, md5(concat))` for merged table naming.
- `PartitionedHashTable`:
  - `gen_feature_configs`: builds `FeatureConfigs` and `ShardingSparseFidsOpParams` based on feature configs and combiners; supports native multi-hash-table and GPU embedding modes.
  - `merge_feature_config` / `no_merge_feature_config`: compute merged sub-table names (with md5) or keep per-feature tables; handles `fc_slot_` → `slot_` extra restore names.
  - Constructor:
    - Reads `parser_ctx.sharding_sparse_fids_op_params` for PS count, native multi-table mode, feature configs, and GPU options.
    - Creates per-PS tables or GPU table; builds export signatures for lookup/raw_lookup when exporting.
    - Sets up learning-rate tensors for each sub-table.
  - `lookup(features, auxiliary_bundle, ...)`:
    - If GPU embedding enabled, delegates to `_lookup_gpu` and optionally returns callable.
    - Otherwise obtains sharded fids via `sharding_sparse_fids` or `ParserCtx`-encoded features, stores offsets and sizes in `auxiliary_bundle`.
    - Optionally returns `lookup_callable_fn` or `fused_layout_callable_fn` for two-phase lookup.
    - `call_lookup`:
      - Uses raw/native lookup or packed lookup; remote_predict in export mode.
      - Stores per-PS embeddings and optional fids/row_splits in `auxiliary_bundle`.
      - Optionally moves auxiliary tensors to GPU and enqueues prefetch queues.
    - `fused_layout_callable_fn`:
      - Calls `distribution_ops.fused_embedding_to_layout` to reconstruct layout embeddings (CPU/GPU depending on export or `_use_gpu`).
      - Uses `nest_layout` to produce output dict.
  - `apply_gradients(layout_grads_and_vars, global_step, req_time, auxiliary_bundle, async_function_mgr, async_push, grad_scale)`:
    - For non-GPU path: uses `fused_embedding_to_layout_grad` to compute per-PS grads, then applies via raw or packed update; supports async push queues.
    - Includes tensor move CPU helper for GPU-derived tensors.
    - For GPU path, delegates to `_apply_gradients_gpu`.
  - `_lookup_gpu`:
    - Uses all-to-all (HVD/BPS/custom) to exchange ids and embeddings; calls `fused_lookup` on GPU table; then all-to-all embeddings back; finally `fused_embedding_to_layout` (version 4) and `nest_layout`.
    - Populates `auxiliary_bundle` with many intermediate tensors (id_flat_t, splits, offsets, recv embeddings, etc.) and optional pipeline queues.
  - `_apply_gradients_gpu`:
    - Computes `fused_embedding_to_layout_grad` on GPU, performs all-to-all backprop (HVD/BPS/custom), and calls `fused_apply_gradient` on GPU table; supports async optimize queues.
  - `assign/assign_add`:
    - Non-GPU only; routes to `_update` or `_native_hash_table_update` depending on native multi-table mode.
  - `flatten_layout` / `nest_layout`:
    - Deterministic ordering by `feature_configs.out_configs` (sorted names); `OutType.NONE` yields list per slices.
  - Queue hooks:
    - `add_queue_hook` stores local hooks; `get_queue_hooks` collects hooks from tables.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/ps` + `monolith-hash-table` + `monolith-data`.
- Rust public API surface: distributed hash table abstractions, partitioned multi-type tables, lookup/apply_gradients APIs.
- Data model mapping: FeatureConfigs and sharded fids, embedding layouts, and fused layout conversions.
- Feature gating: export mode, raw API support, GPU embedding, Horovod/BytePS all-to-all.
- Integration points: `parsers.sharding_sparse_fids`, `embedding_combiners`, `prefetch_queue`, export signatures.

**Implementation Steps (Detailed)**
1. Implement Rust equivalents for `DistributedHashTable` and `DistributedMultiTypeHashTable` with sharding by `id % ps_num`.
2. Recreate packed tensor transfer and optional float16 transport.
3. Implement `PartitionedHashTable` with sharding feature configs and `fused_embedding_to_layout`/`_grad` equivalents.
4. Add GPU embedding path + all-to-all (if supported); otherwise gate behind features.
5. Mirror export signatures for PS-side lookups.
6. Port queue hook logic for pipelined execution.

**Tests (Detailed)**
- Python tests: `distributed_ps_test.py`, `distributed_ps_sync_test.py`, `distribution_ops_test.py`.
- Rust tests: integration tests for lookup/apply_gradients with small sharded tables and layout configs.
- Cross-language parity test: compare embedding outputs for fixed ids across PS shards.

**Gaps / Notes**
- This module is large and deeply tied to TF custom ops; full parity likely requires a TF backend or substantial Rust kernel work.
- GPU/Horovod/BytePS paths are specialized; may need staged parity plan.

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

### `monolith/native_training/distributed_ps_benchmark.py`
<a id="monolith-native-training-distributed-ps-benchmark-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 168
- Purpose/role: Benchmark tests for distributed hash table lookup and apply_gradients performance, optionally with profiling.
- Key symbols/classes/functions: `_generate_config`, `_get_vocab_hash_table_factory`, `DistributedHashTableTest.lookup`, `DistributedHashTableTest.apply_gradients`.
- External dependencies: TensorFlow local servers, `distributed_ps.DistributedHashTable`, `hash_filter_ops`, `hash_table_ops`, `embedding_hash_table_pb2`.
- Side effects: creates local PS servers, may write profiler logs under `/tmp/distributed_ps_benchmark`.

**Required Behavior (Detailed)**
- `_generate_config(servers, job_name=utils.PS_JOB_NAME)`:
  - Builds `ClusterDef` with job tasks derived from server targets; returns `ConfigProto`.
- `_get_vocab_hash_table_factory(dim)`:
  - Returns factory that builds a hash table with `EmbeddingHashTableConfig` using cuckoo + SGD(1.0) + zeros init and segment dim `dim`.
- `DistributedHashTableTest.lookup(enable_dedup, real_run=True)`:
  - Creates `ps_num=10` local servers; uses server0 with cluster config.
  - Builds hash filters and a `DistributedHashTable`, assigns add for ids 0..num_elements-1.
  - If `real_run`: lookup ids `x//2`, check values equal `x//2` repeated per dim; prints wall time; optional profiler.
  - If `real_run=False`: just runs `hash_table.as_op()` to measure overhead.
- `apply_gradients(real_run=True)`:
  - Similar setup; assigns ones to embeddings, looks up, computes `loss=0.3*embeddings`, grads; applies gradients.
  - If `real_run`: after apply, looks up and expects values `0.4` (1.0 + 0.3*?); prints timing.
  - If `real_run=False`: checks grads equal `0.3` if not profiling.
- Tests invoke both real and overhead modes.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/benches`.
- Rust public API surface: benchmark harness for distributed PS table lookup/apply gradients.
- Data model mapping: hash table config, embedding values.
- Feature gating: requires distributed PS runtime and hash table ops.
- Integration points: `distributed_ps` implementation.

**Implementation Steps (Detailed)**
1. Implement a Rust benchmark that spins up local PS servers (or mock) and measures lookup/apply_gradients.
2. Mirror data sizes (1e6 ids, dim=16) and expected outputs.
3. Optionally add profiling hooks matching Python behavior.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: bench-only; optional correctness assertions for small sizes.
- Cross-language parity test: compare outputs for small benchmark sizes.

**Gaps / Notes**
- Uses TF local servers and profiling; Rust may need a simplified harness.

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

### `monolith/native_training/distributed_ps_factory.py`
<a id="monolith-native-training-distributed-ps-factory-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 262
- Purpose/role: Factory helpers to build distributed or local multi-type hash tables and partitioned hash tables with different network/packet-reduction strategies.
- Key symbols/classes/functions: `MultiHashTableFactory`, `create_in_worker_multi_type_hash_table`, `create_multi_type_hash_table`, `create_native_multi_hash_table`, `create_in_worker_native_multi_hash_table`, `create_partitioned_hash_table`.
- External dependencies: `distributed_ps`, `distributed_ps_sync`, `hash_table_ops`, `hash_filter_ops`, `multi_type_hash_table`, `multi_hash_table_ops`, `entry.HashTableConfigInstance`.
- Side effects: none beyond table creation.

**Required Behavior (Detailed)**
- `MultiHashTableFactory`:
  - Caches converted configs via `multi_hash_table_ops.convert_to_cached_config` keyed by `id(slot_to_config)`.
  - `__call__(idx, slot_to_config)` returns `MultiHashTable.from_cached_config` using hash_filter and sync_client for shard `idx`.
- `create_in_worker_multi_type_hash_table(shard_num, slot_to_config, hash_filter, sync_client, queue_configs)`:
  - Builds a `MergedMultiTypeHashTable` whose underlying factory is `DistributedMultiTypeHashTableMpi` (alltoall) created from a per-worker `MultiTypeHashTable` factory.
- `create_multi_type_hash_table(num_ps, slot_to_config, hash_filters, sync_clients, reduce_network_packets, max_rpc_deadline_millis)`:
  - Validates sync_clients length; fills with None if missing.
  - `num_ps==0`: returns local `MergedMultiTypeHashTable` backed by `MultiTypeHashTable` and local hash tables.
  - `reduce_network_packets=False`: uses `DistributedHashTable` per slot within `MultiTypeHashTable` (dedup on worker, distribute to PS).
  - `reduce_network_packets=True`: uses `DistributedMultiTypeHashTable` (multi-type on PS) to reduce RPC count.
- `create_native_multi_hash_table(num_ps, slot_to_config, hash_filters, sync_clients, max_rpc_deadline_millis)`:
  - `num_ps==0`: returns local `MultiHashTable.from_configs`.
  - Else returns `DistributedMultiTypeHashTable` with `MultiHashTableFactory`.
- `create_in_worker_native_multi_hash_table(shard_num, slot_to_config, hash_filter, sync_client, queue_configs)`:
  - Returns `DistributedMultiTypeHashTableMpi` with a local native `MultiHashTable` per shard.
- `create_partitioned_hash_table(num_ps, use_native_multi_hash_table, max_rpc_deadline_millis, hash_filters, sync_clients, enable_gpu_emb, queue_configs)`:
  - Normalizes hash_filters/sync_clients lists.
  - Chooses `multi_type_factory` based on native vs non-native multi-hash table:
    - Native: `MultiHashTableFactory`.
    - Non-native: `MultiTypeHashTable` with hash tables created per PS.
  - Returns `distributed_ps.PartitionedHashTable` with queue configs.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/ps`.
- Rust public API surface: factory functions for hash table backends (local vs distributed).
- Data model mapping: slot configs → hash tables; shard selection logic.
- Feature gating: `reduce_network_packets`, native multi-hash table, GPU embedding.
- Integration points: used by training setup to instantiate embedding tables.

**Implementation Steps (Detailed)**
1. Implement Rust factories mirroring the three strategies (local, distributed per slot, distributed multi-type).
2. Preserve caching for expensive config conversion (if needed).
3. Ensure `num_ps==0` uses local tables and no RPC.
4. Add `PartitionedHashTable` factory with queue config propagation.

**Tests (Detailed)**
- Python tests: `distributed_ps_factory_test.py`.
- Rust tests: unit tests for factory selection logic and returned table types.
- Cross-language parity test: compare selected strategy for given flags.

**Gaps / Notes**
- Depends on TF custom ops for hash tables; Rust must provide equivalent backends.

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

### `monolith/native_training/distributed_ps_factory_test.py`
<a id="monolith-native-training-distributed-ps-factory-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 87
- Purpose/role: Smoke tests for distributed hash table factory functions; primarily checks that constructors run without errors.
- Key symbols/classes/functions: `_get_test_slot_to_config`, `_get_test_hash_filters`, `FactoryTest`.
- External dependencies: Horovod (enabled via env), TensorFlow PS cluster utilities, `test_utils.generate_test_hash_table_config`.
- Side effects: sets `MONOLITH_WITH_HOROVOD=True`; may initialize Horovod.

**Required Behavior (Detailed)**
- `_get_test_slot_to_config()`:
  - Uses `test_utils.generate_test_hash_table_config(4, learning_rate=0.1)`; returns slot map with keys `"1"`, `"2"`.
- `_get_test_hash_filters(num)`:
  - Returns `hash_filter_ops.create_hash_filters(num, False)`.
- Tests:
  - `test_create_in_worker_multi_type_hash_table*`: calls `create_in_worker_multi_type_hash_table` with hvd initialized.
  - `test_create_multi_type_hash_table_0_ps`: local (no PS) creation.
  - `test_create_multi_type_hash_table_2_ps`: creates PS cluster and calls factory under a session.
  - `test_create_multi_type_hash_table_2_ps_with_reduced_packets`: same with `reduce_network_packets=True`.
  - `test_create_native_multi_hash_table_0_ps` and `_2_ps`: native multi-hash table creation.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests`.
- Rust public API surface: factory functions in `distributed_ps_factory`.
- Data model mapping: slot configs to table instances.
- Feature gating: Horovod/PS cluster support.
- Integration points: distributed PS creation.

**Implementation Steps (Detailed)**
1. Add Rust smoke tests to ensure factory functions are callable in local/PS modes.
2. If Horovod not supported, gate tests accordingly.
3. Verify hash filter creation and config plumbing.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: `distributed_ps_factory_test.rs` (smoke only).
- Cross-language parity test: not required beyond constructor success.

**Gaps / Notes**
- These are grammar/smoke tests, not functional correctness tests.

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

### `monolith/native_training/distributed_ps_sync.py`
<a id="monolith-native-training-distributed-ps-sync-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 531
- Purpose/role: Horovod/BytePS synchronous all-to-all embedding lookup and update for distributed multi-type hash tables.
- Key symbols/classes/functions: `DistributedMultiTypeHashTableMpi.lookup`, `.apply_gradients`, `.as_op`.
- External dependencies: Horovod/BytePS env flags, `distribution_ops`, `feature_utils` (control/dense_opt ops), `prefetch_queue`.
- Side effects: uses enqueue queues for pipelined execution; emits alltoall metrics summaries when enabled.

**Required Behavior (Detailed)**
- Environment flags:
  - `MONOLITH_WITH_HOROVOD`, `MONOLITH_WITH_OPTIMIZED_HOROVOD`, `MONOLITH_WITH_BYTEPS` and related G2G/GDR flags determine alltoall backend and GPU paths.
  - `FLAGS.enable_alltoall_metrics` + `enable_alltoall_metrics_for_slot` control summary emission.
- `DistributedMultiTypeHashTableMpi.__init__(shard_num, table_factory, queue_configs)`:
  - Determines rank from BytePS or Horovod; builds local shard table via `table_factory`.
  - Stores output dims, queue configs, and dependency ops.
- `lookup(slot_to_id, auxiliary_bundle, early_reorder_indicies_res_pack)`:
  - Requires `early_reorder_indicies_res_pack` (support for `reorder_fids_in_data_pipeline=False` dropped).
  - Unpacks `(all_fids, shard_sizes, sharded_slot_sizes, emb_offset_sz, fused_embedding_offsets, req_time)`.
  - Performs alltoall on fids and per-slot sizes via BPS/HVD/custom optimized HVD.
  - Stores key tensors in `auxiliary_bundle` (id_flat_t, id_size_flat_t, emb offsets, recv splits, etc.).
  - Calls `self._table.fused_lookup(...)` on GPU, yielding `fused_embeddings`, splits, offsets, indices.
  - Performs embedding alltoall (fwd) and queues prefetch if configured.
  - Uses `distribution_ops.fused_gather_embeddings_by_input` to assemble per-slot embeddings on GPU.
  - Returns `(slot_to_embedding, auxiliary_bundle)`.
- `apply_gradients(slot_to_grad, auxiliary_bundle, global_step, req_time, scale)`:
  - Uses `feature_utils.control_ops` dependency.
  - Computes `grad_flat` via `fused_gather_embeddings_by_input_gradient`.
  - Optionally casts for BPS bwd.
  - Enqueues async optimize queue if configured.
  - Performs backward alltoall using BPS/HVD/custom optimized HVD.
  - Emits alltoall metrics summaries when enabled.
  - Calls `self._table.fused_apply_gradient` with id/grad buffers and offsets.
  - Supports async optimize queue via `AsyncPushHook`.
- `assign/assign_add/reinitialize`:
  - Not implemented (raises `NotImplementedError`).
- `as_op`:
  - Returns `self._table.as_op` with dependency ops.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/ps`.
- Rust public API surface: synchronous alltoall embedding lookup/update for multi-type tables.
- Data model mapping: packed fid buffers and fused embedding offsets.
- Feature gating: Horovod/BytePS support; GPU alltoall paths.
- Integration points: `distributed_ps_factory.create_in_worker_multi_type_hash_table`.

**Implementation Steps (Detailed)**
1. Implement Rust backend selection for alltoall (HVD/BPS equivalents) or gate feature.
2. Port `fused_lookup` + `fused_gather_embeddings_by_input` and gradient counterparts.
3. Preserve auxiliary_bundle keys and queue-based pipelining.
4. Mirror alltoall metric summaries (if logging/metrics available in Rust).

**Tests (Detailed)**
- Python tests: `distributed_ps_sync_test.py`.
- Rust tests: integration tests with small shard_num and deterministic ids.
- Cross-language parity test: compare embeddings and gradients for small fixtures.

**Gaps / Notes**
- Requires GPU kernels and alltoall comms; may need staged parity.

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

### `monolith/native_training/distributed_ps_sync_test.py`
<a id="monolith-native-training-distributed-ps-sync-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 109
- Purpose/role: Validates synchronous alltoall distributed multi-type hash table lookup and gradient updates under Horovod.
- Key symbols/classes/functions: `DistributedMultiTypeHashTableMpiTest.testBasic`, `gen_test_configs`.
- External dependencies: Horovod, `distribution_ops.fused_reorder_by_indices`, `distributed_ps_sync.DistributedMultiTypeHashTableMpi`.
- Side effects: sets `MONOLITH_WITH_HOROVOD=True` and initializes Horovod.

**Required Behavior (Detailed)**
- `gen_test_configs()`:
  - Builds two test hash table configs: slot "1" dim=1 lr=1.0; slot "2" dim=2 with PolynomialDecay LR.
- `testBasic(use_native_multi_hash_table=False)`:
  - Initializes Horovod, global_step=0.
  - Creates table with `DistributedMultiTypeHashTableMpi(hvd.size(), table_factory)`.
  - `slot_to_ids = {"1": [1,1], "2": [2]}`.
  - Uses `distribution_ops.fused_reorder_by_indices` to produce `reordred` pack (plus None timestamp).
  - First lookup returns zeros.
  - Applies gradients `{1: [[0.5],[0.5]], 2: [[0.5,1.0]]}` with `global_step=0`.
  - Second lookup returns negative values scaled by `hvd.size()`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests`.
- Rust public API surface: `DistributedMultiTypeHashTableMpi` and reorder helpers.
- Data model mapping: slot configs, id arrays, gradient arrays.
- Feature gating: Horovod alltoall support.
- Integration points: `distributed_ps_sync` implementation.

**Implementation Steps (Detailed)**
1. Add Rust test that initializes the sync table and performs lookup + apply_gradients.
2. Implement fused reorder (or provide equivalent packed inputs).
3. Validate outputs match expected scaled negatives.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: `distributed_ps_sync_test.rs` with small fixed ids.
- Cross-language parity test: compare outputs for same ids and gradients.

**Gaps / Notes**
- Requires Horovod or equivalent alltoall backend; gate test if unavailable.

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

### `monolith/native_training/distributed_ps_test.py`
<a id="monolith-native-training-distributed-ps-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 979
- Purpose/role: Comprehensive tests for distributed hash tables, multi-type hash tables, export behavior, and partitioned hash table lookup/apply gradients (CPU/GPU).
- Key symbols/classes/functions: `DistributedHashTableTest`, `DistributedMultiTypeHashTableTest`, `DistributedMultiTypeHashTableServingTest`, `PartitionedHashTableTest`.
- External dependencies: TF PS clusters, Horovod env, `distribution_ops`, `export_context`, `sharding_sparse_fids_with_context`.
- Side effects: sets `MONOLITH_WITH_HOROVOD=1` and uses test clusters.

**Required Behavior (Detailed)**
- `DistributedHashTableTest`:
  - `test_basic`: assign_add then lookup equals assigned values.
  - `test_assign`: second assign overwrites ids; lookup after control dependency yields updated values.
  - `test_lookup_dedup`: duplicate ids return repeated embeddings.
  - `test_apply_gradients`: gradients with loss `2*values` updates to `-2` for dim=1.
  - `test_apply_gradients_with_learning_rate_function`: polynomial decay learning rate affects updates; after global_step increment, values change to `-4.2`.
  - `test_apply_gradients_with_duplicates`: duplicate ids produce accumulated gradient; expected `-4` for duplicate id.
  - `test_apply_gradients_with_different_ids`: bp_ids differ from ids; updates only bp ids.
- `DistributedMultiTypeHashTableTest` (param native vs non-native):
  - `testBasic`: assign_add per slot, lookup values, apply_gradients halves values.
  - `test_assign_and_reinitialize`: assign then assign with half values; native mode tests `reinitialize` status and zeros for slot.
  - `test_apply_gradients_with_learning_rate_function`: similar to single-table with polynomial decay; values update with global_step.
  - `test_apply_gradients_float16`: transfer_float16 path; verifies lookup output after apply gradients.
- `DistributedMultiTypeHashTableServingTest`:
  - `test_export_model`: export distributed/standalone/normal training and ensure lookup shapes; verifies `export_ctx.sub_graph_num`.
- `PartitionedHashTableTest`:
  - Helpers: `gen_table_config`, `gen_out_config`, `get_parser_ctx`, `gen_data`, `gen_variant_tensor`.
  - `_test_basic`: assign + assign_add and `_lookup_raw` should return sum of embeddings; runs CPU and GPU variants.
  - `_test_lookup`: assigns const embeddings, sharding sparse fids + lookup yields expected layout tensors (`bias`, `vec`, `deep`).
  - `_test_apply_gradients`: assigns const values, lookup+apply_gradients; verifies updated embeddings against expected FTRL/AdaGrad formulas.
  - `test_apply_gradients_for_gpu_emb`: compares GPU embedding path with CPU path using same gradients; outputs must match.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests`.
- Rust public API surface: distributed hash tables, multi-type tables, partitioned hash table API.
- Data model mapping: fids/embeddings, layout configs, and gradient updates.
- Feature gating: PS clusters, Horovod, GPU embedding path.
- Integration points: `distributed_ps`, `distributed_ps_sync`, `distribution_ops`.

**Implementation Steps (Detailed)**
1. Port test utilities to build PS clusters and configs in Rust (or provide Python-driven fixtures).
2. Recreate expected numeric outputs for assign/lookup/apply_gradients.
3. Implement layout config generation and sharding for partitioned hash table tests.
4. Add GPU embedding parity tests comparing CPU and GPU paths.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: `distributed_ps_test.rs` covering key cases above.
- Cross-language parity test: compare lookup/apply_gradients outputs for small fixed inputs.

**Gaps / Notes**
- File is extensive; ensure Rust tests focus on correctness for representative cases if full coverage is too costly.

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

### `monolith/native_training/distributed_serving_ops.py`
<a id="monolith-native-training-distributed-serving-ops-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 160
- Purpose/role: Remote predict and parameter sync client/server utilities for distributed serving; wraps custom ops for TF Serving RPC and sync.
- Key symbols/classes/functions: `remote_predict`, `create_parameter_sync_clients`, `parameter_sync_client_from_config`, `refresh_sync_config`, `ParameterSyncClient`, `DummySyncServer`.
- External dependencies: `gen_monolith_ops`, `parameter_sync_pb2`, `SyncBackend`, `ServerType`.
- Side effects: creates sync clients/servers on PS devices; uses RPC to remote predict.

**Required Behavior (Detailed)**
- `remote_predict(...)`:
  - Validates `model_name` non-null.
  - Calls `tf_serving_remote_predict` custom op with input/output aliases, model name, task, version, deadline, signature; returns output tensors (index 2 of op result).
- `create_parameter_sync_clients(ps_num)`:
  - For `ps_num==0`, returns single client.
  - Else creates one client per PS on PS device (unless exporting standalone).
- `parameter_sync_client_from_config(config, name_suffix)`:
  - Creates `MonolithParameterSyncClient` op with serialized config and shared_name.
- `refresh_sync_config(sync_backend, ps_index)`:
  - Fetches sync targets; populates `ClientConfig` with targets and extra info; sets model name, signature `hashtable_assign`, timeout 3000ms; returns serialized bytes.
- `create_dummy_sync_client` / `create_dummy_sync_server`:
  - Wrap dummy sync ops.
- `ParameterSyncClient`:
  - `create_sync_op` calls `monolith_parameter_sync` op with client handle and config string.
  - `as_op` wraps client handle with `tf.group`.
- `DummySyncServer`:
  - `shutdown` and `get_port` wrap dummy server ops; `as_op` groups server handle.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/serving` or `monolith-serving`.
- Rust public API surface: remote predict wrapper + parameter sync client/server wrappers.
- Data model mapping: `ClientConfig` protobuf and sync target lists.
- Feature gating: TF runtime + custom ops for remote predict and sync.
- Integration points: distributed PS and export paths.

**Implementation Steps (Detailed)**
1. Implement RPC wrapper for remote predict (TF Serving or custom stub).
2. Implement parameter sync client creation and config refresh logic.
3. Provide dummy client/server for tests.

**Tests (Detailed)**
- Python tests: `distributed_serving_ops_test.py`.
- Rust tests: integration tests for config building and dummy sync ops.
- Cross-language parity test: compare serialized config bytes.

**Gaps / Notes**
- `remote_predict` relies on custom op; Rust likely needs TF C API bindings.

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

### `monolith/native_training/distributed_serving_ops_test.py`
<a id="monolith-native-training-distributed-serving-ops-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 142
- Purpose/role: Tests parameter sync client/server ops and sync config generation for ZK backend and replica watcher.
- Key symbols/classes/functions: `ParameterSyncOpsTest`, `test_parameter_sync_client`, `test_refresh_sync_config_1`, `test_refresh_sync_config_2`.
- External dependencies: `DummySyncServer`, `ParameterSyncClient`, agent_service backend mocks, FakeKazooClient, `parameter_sync_pb2`.
- Side effects: creates dummy sync servers and ZK backend state; uses fake clients.

**Required Behavior (Detailed)**
- `test_parameter_sync_client`:
  - Creates two `DummySyncServer`s, gets ports.
  - Builds `ParameterSyncClient` with targets; creates hash table with `sync_client`.
  - Runs lookup + apply_gradients; expects embeddings `[[0.2,0.2,0.2],[0.1,0.1,0.1]]`.
  - Calls `client.create_sync_op` with config; prints JSON; shuts down servers.
- `test_refresh_sync_config_1`:
  - Mocks `ReplicaWatcher` with FakeKazooClient; sets replica meta with address `localhost:8500`.
  - `refresh_sync_config` should set `model_name='ps_1'` and targets `['localhost:8500']`.
- `test_refresh_sync_config_2`:
  - Sets up ZK backend with container services; syncs available saved models.
  - `refresh_sync_config` with ps_index 1 yields `model_name='test_ffm_model:ps_1'` and targets `['localhost:8888']`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests`.
- Rust public API surface: parameter sync client/server wrappers + config refresh.
- Data model mapping: ZK backend or equivalent; ClientConfig proto.
- Feature gating: requires sync backend mocks or fixtures.
- Integration points: distributed_serving_ops + agent_service backends.

**Implementation Steps (Detailed)**
1. Implement dummy sync server/client wrappers in Rust for test harness.
2. Add tests that apply gradients and verify embedding updates.
3. Add mock backend tests for `refresh_sync_config` with ZK-style data.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: `distributed_serving_ops_test.rs` with mocks.
- Cross-language parity test: compare ClientConfig bytes and target lists.

**Gaps / Notes**
- Depends on agent_service backends; Rust will need lightweight mocks.

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

### `monolith/native_training/distribution_ops.py`
<a id="monolith-native-training-distribution-ops-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 889
- Purpose/role: Wrapper utilities around custom distribution ops for sharding, embedding layout transforms, and gradient backprop helpers.
- Key symbols/classes/functions: `split_by_indices`, `ragged_split_by_indices`, `unique_key_with_value_and_offset`, `fill_with_offset_map`, `finalize_shared_tensor`, `reorder_by_indices`, `fused_reorder_by_indices`, `map_id_to_embedding`, `fused_embedding_to_layout` (+ grads), `map_id_to_embedding_gradient_back_prop`, `fused_gather_embeddings_by_input`, `fused_gather_embeddings_by_input_gradient`, reduce/sorted-segment ops.
- External dependencies: `gen_monolith_ops` custom kernels, `FeatureConfigs` proto.
- Side effects: registers custom gradients for several ops.

**Required Behavior (Detailed)**
- `split_by_indices(indices, tensor, num_splits)`:
  - Calls `monolith_split_by_indices` custom op; gradient registered via `monolith_split_by_indices_gradient`.
- `ragged_split_by_indices(indices, num, num_splits)`:
  - Splits ragged tensor by indices; returns list of ragged tensors + list of corresponding positions.
- `unique_key_with_value_and_offset(key, dims, generate_buffer=True)`:
  - Deduplicates ragged keys and returns `unique_key`, `value_offset` (ragged) and `value_buffer` sized by dims.
- `fill_with_offset_map(pos, value, value_offset_map, value_buffer, dims)`:
  - Fills `value_buffer` positions from offsets; gradient registered via `fill_with_offset_map_gradient`.
- `finalize_shared_tensor(shared_tensor_handles, dtype, shape)`:
  - Finalizes shared tensor handles; gradient returns upstream grad (identity).
- `reorder_by_indices` / `fused_reorder_by_indices`:
  - Reorders input ids by shard indices; `fused_reorder_by_indices` returns packed tensors and offsets for fused pipelines.
- `map_id_to_embedding(ids, embeddings, input)`:
  - Maps sharded embeddings back to original id order; gradient hook registered.
- `fused_embedding_to_layout(embeddings_list, fid_offset, feature_offset, nfl_offset, batch_size, ...)`:
  - Converts flattened embeddings into layout tensors using `FeatureConfigs` and offsets; supports multiple versions and GPU paths.
  - Gradient functions `_fused_embedding_to_layout_grad_v{1..5}` and `fused_embedding_to_layout_grad` wrap custom ops.
- `map_id_to_embedding_gradient_back_prop(ids, input, grads)`:
  - Builds gradients back to sharded embeddings.
- `gather_embeddings_by_input` / `fused_gather_embeddings_by_input`:
  - Gathers embedding vectors using ids and offsets; fused variants use offsets and sizes.
- Reduce ops:
  - `reduce_mean`, `reduce_sum`, `reduce_sqrtn` with custom gradients.
  - `fused_sorted_segment_sum`, `fused_reduce_sum_and_split`, `fused_reduce_and_split_gpu` for GPU fused reductions with gradients.
- `normalize_merged_split(row_split, size)`:
  - Normalizes row split sizes for merged ragged splits.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/ops` (or `monolith-tensor` for ragged).
- Rust public API surface: distribution ops wrappers, gradient-friendly functions if TF backend.
- Data model mapping: ragged tensors, embedding offsets, FeatureConfigs.
- Feature gating: requires custom kernel bindings or reimplementation.
- Integration points: distributed_ps, partitioned hash table, sharding pipelines.

**Implementation Steps (Detailed)**
1. Bind or reimplement each custom op with identical signatures and gradient behavior.
2. Implement ragged split/unique/offset map logic in Rust if not using TF.
3. Support fused embedding to layout and gradient versions used by PartitionedHashTable.
4. Add tests for each op with small deterministic tensors.

**Tests (Detailed)**
- Python tests: `distribution_ops_test.py`, `distribution_ops_fused_test.py`.
- Rust tests: unit tests for each op wrapper; integration tests with PartitionedHashTable.
- Cross-language parity test: compare outputs for fixed inputs and gradient checks.

**Gaps / Notes**
- Many ops are custom kernels; full parity requires substantial backend work.

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

### `monolith/native_training/distribution_ops_benchmark.py`
<a id="monolith-native-training-distribution-ops-benchmark-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 118
- Purpose/role: Benchmarks `map_id_to_embedding` and `gather_embeddings_by_input` with/without multi-threading; optional profiler output.
- Key symbols/classes/functions: `DistributionOpsBenchmarkTest.map_id_to_embedding`, `test_gather_embeddings_by_ids_basic`, `test_gather_embeddings_by_ids_multi_threads`.
- External dependencies: TensorFlow profiler, `distribution_ops`.
- Side effects: writes profiler logs under `/tmp/distribution_ops_benchmark/*`.

**Required Behavior (Detailed)**
- `map_id_to_embedding(use_multi_threads)`:
  - Creates 1e6 ids, dim=16, ps_num=10; splits ids/embeddings and maps back.
  - Asserts mapped embeddings equal original; starts/stops TF profiler in log dir.
- `test_gather_embeddings_by_ids_basic`:
  - Benchmarks gather with 100k features; runs for dim=32 and dim=256, different input lengths.
- `test_gather_embeddings_by_ids_multi_threads`:
  - Same as above but `use_multi_threads=True`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/benches`.
- Rust public API surface: bench harness for distribution ops.
- Data model mapping: ids/embeddings tensors.
- Feature gating: multi-threaded execution support.
- Integration points: `distribution_ops` implementation.

**Implementation Steps (Detailed)**
1. Implement Rust benchmarks for `map_id_to_embedding` and `gather_embeddings_by_input`.
2. Mirror tensor sizes and check correctness for small sizes; run benchmarks for large sizes.
3. Add optional profiling hooks.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: benches or microbench harness.
- Cross-language parity test: not required beyond correctness checks.

**Gaps / Notes**
- Pure benchmark; not a correctness test.

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

### `monolith/native_training/distribution_ops_fused_benchmark.py`
<a id="monolith-native-training-distribution-ops-fused-benchmark-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 61
- Purpose/role: Benchmarks `fused_reorder_by_indices` performance on large random IDs.
- Key symbols/classes/functions: `run_fused_reorder_by_indicies`.
- External dependencies: numpy, TensorFlow, `distribution_ops`.
- Side effects: none; prints average wall time.

**Required Behavior (Detailed)**
- Generates ~1e6 unique int64 IDs, 30 slots, 256 shards.
- For each slot, duplicates IDs to force duplicates and shuffles.
- Runs `distribution_ops.fused_reorder_by_indices(ids_list, num_of_shards=256)` in a session and times execution.
- Main prints average wall time over 5 runs.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/benches`.
- Rust public API surface: fused reorder benchmark.
- Data model mapping: list of id tensors, shard count.
- Feature gating: fused reorder op implementation.
- Integration points: `distribution_ops` fused reorder.

**Implementation Steps (Detailed)**
1. Implement Rust bench that generates similar random IDs and runs fused reorder.
2. Use consistent shard count and slot count for comparability.

**Tests (Detailed)**
- Python tests: this file (benchmark).
- Rust tests: bench only.
- Cross-language parity test: not required beyond output correctness.

**Gaps / Notes**
- Pure benchmark; no correctness assertions.

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

### `monolith/native_training/distribution_ops_fused_test.py`
<a id="monolith-native-training-distribution-ops-fused-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 148
- Purpose/role: Tests for `fused_reorder_by_indices` correctness and embedding offset outputs.
- Key symbols/classes/functions: `_test_fused_reorder_by_indices`, `test_fused_reorder_by_indices`, `test_ragged_tensor_workflow`.
- External dependencies: TensorFlow, `distribution_ops`.
- Side effects: none.

**Required Behavior (Detailed)**
- `test_benchmark`: runs a large random `fused_reorder_by_indices` to smoke test.
- `_test_fused_reorder_by_indices`:
  - Calls `fused_reorder_by_indices(ids_list, num_of_shards, dim_sizes)`.
  - Asserts output order, split sizes, and sharded slot sizes; optionally checks embedding offsets.
- `test_fused_reorder_by_indices`:
  - Multiple cases: single slot, extra empty slot, plus offset ids, empty slots, different shard counts, and dim_sizes for offsets.
- `test_ragged_tensor_workflow`:
  - Builds merged slot values from ragged tensors and validates fused reorder output.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests`.
- Rust public API surface: `fused_reorder_by_indices` op.
- Data model mapping: list of id tensors → reordered ids + split sizes.
- Feature gating: fused distribution ops.
- Integration points: partitioned hash table lookup pipeline.

**Implementation Steps (Detailed)**
1. Implement Rust tests mirroring each expected output case.
2. Validate embedding offsets for dim_sizes cases.
3. Include ragged workflow test for merged slots.

**Tests (Detailed)**
- Python tests: this file.
- Rust tests: `distribution_ops_fused_test.rs`.
- Cross-language parity test: compare outputs and offsets for fixed inputs.

**Gaps / Notes**
- Uses Python list inputs; Rust tests should use deterministic tensors.

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

### `monolith/native_training/distribution_ops_test.py`

<a id="monolith-native-training-distribution-ops-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 536
- Purpose/role: TensorFlow test coverage for custom distribution ops (split/reorder, ragged routing, embedding gather, reduction, fused GPU ops).
- Key symbols/classes/functions: `DistributionOpsTest` test cases.
- External dependencies: `numpy`, `tensorflow`, `tensorflow.python.framework.test_util`, `random`, `monolith.native_training.distribution_ops`.
- Side effects: Requires GPU for `@test_util.run_gpu_only` tests; uses TF v1 sessions and graph mode.

**Required Behavior (Detailed)**
- `test_split_by_indices`:
  - `ids=[0,1,2,2,3]`, `indices=ids % 3`, `split_by_indices(..., num_splits=3)` -> `[[0,3],[1],[2,2]]`.
- `test_reorder_by_indices`:
  - `ids=[0,1,2,2,3,5]`, `indices=ids % 3`, `reorder_by_indices(..., num_of_shards=3)` -> `output=[3,0,1,5,2]`, `split_sizes=[2,1,2]`.
- `test_split_by_indices_gradient`:
  - Gradient of split over `tensor=[[0,0],[1,1],[2,2]]` returns all ones.
- `test_split_by_indices_empty_gradient`:
  - Empty inputs return empty gradient `[]`.
- `test_ragged_split_by_indices`:
  - Ragged `num=[[],[],[4,3,2],[1],[],[]]`, `indices=[0,1,0,1]` -> `splits` and `pos` arrays match expected nested ragged values.
- `test_unique_key_with_value_and_offset_and_fill_with_offset_map`:
  - `unique_key_with_value_and_offset` over ragged keys returns:
    - `unique_key=[[],[0,1,2],[0,1],[]]`
    - `value_offset=[[],[[0,8],[2,6],[4]],[[10,16],[13]],[]]`
  - `fill_with_offset_map` + `finalize_shared_tensor` yields `buffer=[0,1,2,3,4,5,2,3,0,1,6,7,8,9,10,11,6,7,8]`.
  - Gradient of `buffer` wrt `value` equals `[8,10,8,10,4,5,26,28,30,13,14,15]`.
- `test_fill_with_offset_map_error_case`:
  - When `value` length too small (10 vs expected 12), evaluating `filled_tensor` raises `InvalidArgumentError`.
- `test_unique_key_with_value_and_offset_empty`:
  - Empty ragged keys -> empty `unique_key`/`value_offset`.
- `test_map_id_to_embedding`:
  - Map ids `[1]`,`[2]` to embeddings `[[1,1]]`,`[[2,2]]` and input `[[1],[2]]` -> output `[[[1,1]],[[2,2]]]`.
- `test_map_id_to_embedding_multi_threads`:
  - 1k ids, 16-dim embeddings, `ps_num=10` -> multi-threaded mapping returns exact original embeddings.
- `test_map_id_to_embedding_gradient`:
  - Loss vs target `[[2,2],[2,2],[2,2]]` yields gradients `embeddings1=[[-2,-2]]`, `embeddings2=[[-1,-1]]`.
- `test_gather_embeddings_by_ids`:
  - `ids=[1,2,3]`, `embeddings=[[1,1],[2,2],[3,3]]`, input `[[2],[1],[2]]` -> output `[[[2,2]],[[1,1]],[[2,2]]]`, `index_mapping=[[1],[0],[1]]`.
- `test_gather_embeddings_by_ids_gradient`:
  - Gradient wrt embeddings equals `[[-2,-2],[-1,-1],[0,0]]`.
- `test_gather_embeddings_by_ids_gradient_back_prop`:
  - `ids=[2,3,1]`, `grads` + `index_mapping=[1,0,1,2]` -> output `[[2,2],[5,5],[8,8]]`.
- `test_fused_gather_embeddings_by_input` (GPU only):
  - Uses fused embeddings + offsets with large SCALE; expects exact outputs per slot (repeated SCALE times).
- `test_fused_gather_embeddings_by_input_gradient` (GPU only):
  - `fused_embeddings_size=22`, `embedding_dims=[3,2]`, SCALE=888 -> output length 22 and expected sums scaled; tolerance `rtol=1e-7 * SCALE`.
- `test_reduce_mean` and `test_reduce_mean_gradient`:
  - Mean reductions produce expected values; gradients are `[-1,-1]` per row.
- `test_reduce_sum` and `test_reduce_sum_gradient`:
  - Sum reductions produce expected values; gradients are `[-1,-1]` per row.
- `test_reduce_sqrtn`, `test_reduce_sqrtn_gradient`, `test_reduce_sqrtn_gradient_zero`:
  - Sqrt-N reductions and gradients match expected numeric values; zero inputs yield zero gradients.
- `test_fused_reduce_sum_and_split`:
  - CPU-only; verifies split sizes `[2,1]` and `[1,2]` for consecutive/non-consecutive indices, with zero-filled rows for gaps.
- `test_fused_reduce_sum_and_split_grad`:
  - Gradient wrt id_values is all ones.
- `test_fused_reduce_scatter` (GPU only):
  - `fused_sorted_segment_sum` matches `scatter_nd` output and gradient across multiple shapes (includes empty tensor case).
- `test_fused_reduce_and_split_gpu` (GPU only):
  - For ragged rows and many embedding lengths, outputs match scatter+split and gradients match for all outputs.
- `test_aligned_concat_split` (GPU only):
  - Random tensors round-trip through `monolith_aligned_flat_concat`/`monolith_aligned_flat_split`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-tf` (TF runtime adapter) + new tests.
- Rust public API surface: wrappers in `monolith-rs/crates/monolith-tf/src/distribution_ops.rs` (new) and tests in `monolith-rs/crates/monolith-tf/tests/distribution_ops_test.rs` and `monolith-rs/crates/monolith-tf/tests/distribution_ops_gpu_test.rs`.
- Data model mapping: TF tensors (dense and ragged), custom op handles, gradient support for TF runtime.
- Feature gating: `tf-runtime` feature for these tests; GPU-only tests gated on CUDA availability.
- Integration points: custom op library load (libmonolith_ops) before creating the TF graph/session.

**Implementation Steps (Detailed)**
1. Add TF runtime harness in Rust: build graph/session, load `libmonolith_ops`, wrap op invocation.
2. Implement CPU tests for `split_by_indices`, `reorder_by_indices`, `ragged_split_by_indices`, `unique_key_with_value_and_offset`, `fill_with_offset_map`, `finalize_shared_tensor`, `map_id_to_embedding`, `gather_embeddings_by_input`, and `reduce_*` ops.
3. Add gradient checks using TF gradient API; if Rust bindings do not expose gradients, run parity via Python harness and document skip in Rust.
4. Add GPU-only tests for fused gather, fused reduce scatter, fused reduce+split GPU, and aligned concat/split; skip when CUDA/custom ops missing.
5. Seed RNG or replace random tensors with deterministic values to avoid flakiness (especially `test_aligned_concat_split`).
6. Validate error handling for `fill_with_offset_map` with invalid input sizes (InvalidArgumentError).
7. Document Candle backend deviations: these ops require TF custom kernels and are only supported under the TF runtime feature.

**Tests (Detailed)**
- Python tests: `monolith/native_training/distribution_ops_test.py`.
- Rust tests: `monolith-rs/crates/monolith-tf/tests/distribution_ops_test.rs` (CPU) and `monolith-rs/crates/monolith-tf/tests/distribution_ops_gpu_test.rs` (GPU).
- Cross-language parity test: run Python and Rust TF tests on identical inputs; compare tensors within tolerance and verify gradients.

**Gaps / Notes**
- Requires TF custom ops build + dynamic loading; without this, Rust tests must be skipped.
- GPU tests are sensitive to CUDA availability and may need CI skips.

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

### `monolith/native_training/distribution_utils.py`
<a id="monolith-native-training-distribution-utils-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 443
- Purpose/role: BytePS/Horovod initialization, MPI helpers, sync training config updates, GPU session config tweaks, and BytePS micro-benchmarks.
- Key symbols/classes/functions: `bps_init`, `byteps_benchmark_ar`, `byteps_benchmark_a2a`, `bps_comm_benchmark`, `init_sync_train_and_update_conf`, `get_mpi_rank`, `get_mpi_local_rank`, `get_mpi_size`, `get_mpi_local_size`, `enable_sync_training`, `try_init_cuda`, `get_device_str`, `get_sync_run_hooks`, `update_session_config_for_gpu`.
- External dependencies: `absl.flags`, `absl.logging`, `tensorflow`, `byteps.tensorflow` (optional), `horovod.tensorflow` (optional), `monolith.native_training.metric.metric_hook.ByteCCLTelemetryHook`.
- Side effects: Sets many env vars, creates `/tmp/bps_<uuid>_socket_<id>` dir, runs `ip addr show` via shell, enables eager execution in benchmark funcs, initializes BytePS/Horovod, mutates config object fields.

**Required Behavior (Detailed)**
- Global state:
  - `_SYNC_TRAIN_INITED` gate to avoid repeated init.
  - `enable_bps = int(os.getenv("MONOLITH_WITH_BYTEPS", "0"))` evaluated at import time.
- `bps_init(uuid)`:
  - Ensures `BYTEPS_ALLTOALL_SESSION_SIZE` default `3`.
  - Mirrors `OMPI_COMM_WORLD_*` into `BYTEPS_LOCAL_SIZE`, uses `BYTEPS_LOCAL_SIZE` to compute `local_rank` and `phy_node_id`.
  - Computes `socket_path = /tmp/bps_<uuid>_socket_<phy_node_id>` and creates it.
  - Chooses network interface:
    - If `BYTEPS_GPU_NIC_BINDING_MODE=0`, uses `DMLC_INTERFACE` (default `eth0`).
    - Else, binds NIC by GPU index (`NUM_GPU_PER_NIC=2`), sets `CUDA_VISIBLE_DEVICES` and UCX/GDR envs when `MONOLITH_WITH_BYTEPS_FWD_GDR` or `MONOLITH_WITH_BYTEPS_BWD_GDR` is enabled.
    - If `BYTEPS_WITH_ALL_NICS=1`, sets `UCX_NET_DEVICES` to a list of mlx5 + eth; else only `mlx5_<nic_id>:1`.
  - Runs `ip addr show <interface>` to compute host IP; exports as `UCX_RDMA_CM_SOURCE_ADDRESS` and `DMLC_NODE_HOST`.
  - Sets required BytePS/PSLite env vars (role, worker/server counts, UUID, ranks, telemetry, log levels, perf knobs, partition sizes).
  - Ensures `BYTEPS_P2P_PARTITION_BYTES` and `BYTEPS_PARTITION_BYTES` defaults computed from `size`.
  - Imports `byteps.tensorflow` and calls `bps.init(lazy=False)`.
- `byteps_benchmark_ar(total_len, total_niter=10000, use_cpu=False, op='pushpull')`:
  - Enables eager execution; uses `bps.push_pull` by default.
  - Creates tensor of shape `[total_len, 1]` on CPU/GPU.
  - Runs `total_niter` iterations, logs latency/Goodput every 20 iterations, returns `goodputs[1:]`.
- `byteps_benchmark_a2a(total_len, total_niter=10000, dst_gpu=True, src_gpu=True)`:
  - Enables eager execution; if CPU-only (`dst_gpu=False` and `src_gpu=False`) reduces `total_len` by 8.
  - Builds splits and recv_splits; selects correct BytePS alltoall variant (`alltoall`, `alltoall_cpu2gpu`, `alltoall_gpu2cpu`).
  - Runs loop and returns `goodputs[1:]`.
- `bps_comm_benchmark()`:
  - Reads `MONOLITH_BENCHMARK_BPS`, `MONOLITH_BENCHMARK_ITERS`, and length env vars; sets TF memory growth for all GPUs.
  - Runs selected benchmarks and prints summary tuples `(total_len, avg_goodput)`.
- `init_sync_train_and_update_conf(dct_config)`:
  - Logs entry; imports BytePS or Horovod as needed; initializes once.
  - If not `merge_sync_training_ckpt`, updates `dct_config.model_dir` with `index-<rank>` suffix under `model_dir/uuid/`.
  - Sets `num_ps=0`, `reorder_fids_in_data_pipeline=True`, `index=hvd.rank()`, `num_workers=hvd.size()`, `enable_variable_partition=False`.
  - Catches ImportError/NotFoundError and logs warning.
- MPI helpers:
  - `get_mpi_rank/local_rank/size/local_size` pull from `OMPI_COMM_WORLD_*` envs; warn and use defaults (0/1) when missing.
- `enable_sync_training()`:
  - Returns `FLAGS.enable_sync_training and 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ`; returns False on exception.
- `try_init_cuda()`:
  - If `CUDA_VISIBLE_DEVICES` not set but MPI local rank present, set `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES=<local_rank>`.
  - If sync training enabled and not initialized, tries to import BytePS or Horovod (based on `MONOLITH_WITH_BYTEPS`/`MONOLITH_WITH_HOROVOD`) and `hvd.init()`; logs exceptions.
- `get_device_str(force_on_cpu=False)`:
  - Uses `FLAGS.enable_gpu_training` or `device_utils._GPU_PLACEMENT_ALLOWED` to choose GPU vs CPU.
  - For MPI + sync training:
    - In PS mode (`FLAGS.num_ps > 0`): returns `/job:chief` for rank 0 else `/job:worker` with `task` offsets and `/device:{GPU|CPU}:0`.
    - Without PS mode: returns empty string.
  - Otherwise returns `/device:{GPU|CPU}:0`.
- `get_sync_run_hooks(is_full_sync=False)`:
  - Returns empty list when not in sync mode.
  - Uses BytePS `BroadcastGlobalVariablesHook` when `MONOLITH_WITH_BYTEPS` and `MONOLITH_WITH_BYTEPS_BCAST` are set.
  - If `MONOLITH_WITH_BYTEPS_BCAST == -1`, returns empty list.
  - Adds `ByteCCLTelemetryHook(50)` when `is_full_sync` and using BytePS broadcast.
  - Falls back to Horovod `BroadcastGlobalVariablesHook` when not using BytePS.
- `update_session_config_for_gpu(session_config)`:
  - When sync training is enabled, sets `gpu_options.visible_device_list` to local rank.
  - If `MONOLITH_FORCE_GPU_COMPATIBLE=1`, sets `force_gpu_compatible=True`.
  - If BytePS GDR alltoall enabled (`MONOLITH_WITH_BYTEPS_FWD_GDR` or `MONOLITH_WITH_BYTEPS_BWD_GDR`), disables `allow_growth`, sets `per_process_gpu_memory_fraction=0.4` and visible device list to local rank.
  - Otherwise enables `allow_growth`.
  - When not in sync training, still sets `allow_growth=True`.

**Rust Mapping (Detailed)**
- Target crate/module: new `monolith-rs/crates/monolith-training/src/distribution_utils.rs` (sync training env + device helpers) and `monolith-rs/crates/monolith-training/src/distributed.rs` for MPI helpers.
- Rust public API surface: `init_sync_train_and_update_conf`, `get_mpi_*`, `enable_sync_training`, `try_init_cuda`, `get_device_str`, `get_sync_run_hooks`, `update_session_config_for_gpu` equivalents; optional TF-specific BytePS/Horovod bridge behind feature flags.
- Data model mapping: map Python `dct_config` mutation to Rust config struct (likely in `monolith-training` or `monolith-cli`).
- Feature gating: `tf-runtime` (BytePS/Horovod) and `cuda` (GPU-specific paths); default Candle backend should no-op or provide safe fallbacks.
- Integration points: `monolith/native_training/estimator.py`, `cpu_training.py`, `device_utils.py`, `model_export/saved_model_exporters.py`, and `data/datasets.py` equivalents in Rust.

**Implementation Steps (Detailed)**
1. Define a Rust config struct that mirrors `dct_config` fields used here (`uuid`, `model_dir`, `merge_sync_training_ckpt`, `num_ps`, `reorder_fids_in_data_pipeline`, `index`, `num_workers`, `enable_variable_partition`).
2. Implement env parsing for MPI (`OMPI_COMM_WORLD_*`) and BytePS/Horovod gating (`MONOLITH_WITH_BYTEPS`, `MONOLITH_WITH_HOROVOD`).
3. Add `get_device_str` logic to Rust; plumb in `enable_gpu_training` and `num_ps` flags (from CLI or config).
4. For TF runtime, implement BytePS/Horovod initialization and broadcast hooks; otherwise return empty hooks and log warnings.
5. Implement `try_init_cuda` that sets env vars before GPU runtime init (Rust side); keep `_SYNC_TRAIN_INITED` equivalent.
6. Implement `update_session_config_for_gpu` only when using TF sessions; in Candle backend, document as no-op.
7. Port benchmark helpers only if TF BytePS runtime is supported; otherwise document as unsupported.

**Tests (Detailed)**
- Python tests: none in-tree specific to this file.
- Rust tests: add unit tests for env parsing and `get_device_str` permutations; integration tests for config updates and no-op behavior when BytePS/Horovod missing.
- Cross-language parity test: compare outputs of MPI helpers and device string formatting for a matrix of env/flag combinations.

**Gaps / Notes**
- Uses shell command `ip addr show` to resolve interface IP; Rust port should use OS APIs or run the command for parity.
- Heavy BytePS/Horovod coupling means full parity likely only under TF runtime with custom ops.

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

### `monolith/native_training/embedding_combiners.py`
<a id="monolith-native-training-embedding-combiners-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 102
- Purpose/role: Defines embedding combiner strategies for ragged inputs (sum/mean pooling and FirstN sequence padding).
- Key symbols/classes/functions: `Combiner`, `ReduceSum`, `ReduceMean`, `FirstN`.
- External dependencies: `tensorflow`, `distribution_ops`, `ragged_utils`, `device_utils`.
- Side effects: None beyond device placement in `FirstN.combine`.

**Required Behavior (Detailed)**
- `Combiner`:
  - Stores `max_seq_length` and exposes `combine(...)` abstract method.
- `ReduceSum.combine(key, embedding, name=None)`:
  - Uses `ragged_utils.fused_value_rowids(key)` to map values to row ids.
  - Calls `distribution_ops.reduce_sum(expand_dims(rowids), embedding, expand_dims(key.nrows(), 0), name=name)`.
- `ReduceMean.combine(key, embedding, name=None)`:
  - Same as `ReduceSum` but calls `distribution_ops.reduce_mean`.
- `FirstN.__init__(seq_length)`:
  - Asserts `seq_length > 0`, sets `max_seq_length` to `seq_length`.
- `FirstN.combine(key, embedding, name=None)`:
  - If `embedding` is not a `tf.Tensor`, converts it.
  - Computes `batch_size_tensor = key.nrows()`.
  - Converts `key` to sparse (`key_sparse = key.to_sparse()`), uses `key_sparse.indices` to scatter.
  - Builds `shape = [batch_size, max(max_seq_length, key_sparse.dense_shape[1]), embedding_dim]` with `embedding.shape.as_list()[1]`.
  - Under `device_utils.maybe_device_if_allowed('/device:GPU:0')`, calls `tf.scatter_nd(indices, embedding, shape)`.
  - Returns `tf.slice(scattered, [0,0,0], [-1, max_seq_length, -1])` to enforce sequence length.
  - Rows with fewer embeddings are zero-padded by scatter.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-layers/src/embedding.rs` and/or new `monolith-rs/crates/monolith-layers/src/combiner.rs`.
- Rust public API surface:
  - `Combiner` trait with `combine(key, embedding) -> Tensor`.
  - `ReduceSum` and `ReduceMean` pooling for ragged sequences.
  - `FirstN` equivalent using `SequenceEmbeddingLookup` or a new combiner wrapper.
- Data model mapping: Ragged input represented as `(values, row_lengths)` or `(values, row_splits)`; must map to pooled or padded tensors.
- Feature gating: TF runtime path can call distribution_ops; Candle backend should implement native pooling and padding.
- Integration points: use in `feature.py`/`feature_utils.py` equivalents and embedding table lookup paths.

**Implementation Steps (Detailed)**
1. Define a Rust `Combiner` trait and enums for `ReduceSum`, `ReduceMean`, `FirstN`.
2. Implement pooling for ragged sequences using row lengths (sum/mean) with deterministic order.
3. Implement `FirstN` by zero-padding to `[batch, max_seq_length, dim]` and truncating when longer.
4. Preserve shape inference behavior: unknown batch size => dynamic dimension, but known `max_seq_length` and embedding dim.
5. If TF runtime is enabled, optionally route to distribution_ops to match TF kernels exactly.
6. Add device placement logic for GPU (if supported) and document when CPU is forced.

**Tests (Detailed)**
- Python tests: `monolith/native_training/embedding_combiners_test.py`.
- Rust tests: `monolith-rs/crates/monolith-layers/tests/embedding_combiners_test.rs` (new) or extend `monolith-layers/src/embedding.rs` tests.
- Cross-language parity test: compare pooled and padded outputs for the same ragged inputs and ensure shape inference matches.

**Gaps / Notes**
- Python uses `ragged_utils.fused_value_rowids` and custom reduce ops; Rust must replicate row-id logic exactly.
- `FirstN` uses `scatter_nd` behavior; ensure zero-fill for missing entries and correct truncation.

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

### `monolith/native_training/entry_test.py`
<a id="monolith-native-training-entry-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 84
- Purpose/role: Smoke tests for optimizer/initializer/compressor config builders and `HashTableConfigInstance` string equality.
- Key symbols/classes/functions: `EntryTest` test cases.
- External dependencies: `entry`, `learning_rate_functions`, `embedding_hash_table_pb2`.
- Side effects: None.

**Required Behavior (Detailed)**
- `test_optimizers`:
  - Instantiates each optimizer class and calls `as_proto()` without error.
  - Covers: `SgdOptimizer`, `AdagradOptimizer`, `FtrlOptimizer`, `DynamicWdAdagradOptimizer`, `AdadeltaOptimizer`, `AdamOptimizer`, `AmsgradOptimizer`, `MomentumOptimizer`, `MovingAverageOptimizer`, `RmspropOptimizer`, `RmspropV2Optimizer`, `BatchSoftmaxOptimizer`.
- `test_initializer`:
  - Calls `as_proto()` for `ZerosInitializer`, `RandomUniformInitializer(-0.5,0.5)`, `BatchSoftmaxInitializer(1.0)`.
- `test_compressor`:
  - Calls `as_proto()` for `Fp16Compressor`, `Fp32Compressor`, `FixedR8Compressor`, `OneBitCompressor`.
- `test_combine`:
  - Calls `CombineAsSegment(5, ZerosInitializer(), SgdOptimizer(), Fp16Compressor())` and expects no error.
- `test_hashtable_config`:
  - Instantiates `CuckooHashTableConfig`.
- `test_hashtable_config_entrance`:
  - Creates `EmbeddingHashTableConfig` instances and `HashTableConfigInstance` wrappers.
  - Validates `str(config1) == str(config2)` for same numeric learning rate.
  - Validates `str(config3) == str(config4)` for same callable learning-rate function.
  - Validates `str(config1) != str(config3)`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-hash-table/tests/entry_test.rs` (new).
- Rust public API surface: config builder types mirroring the Python constructors and `as_proto` equivalents.
- Data model mapping: `monolith::hash_table::EmbeddingHashTableConfig` from `monolith-proto`.
- Feature gating: none.
- Integration points: `HashTableConfigInstance` display and equality semantics.

**Implementation Steps (Detailed)**
1. Add Rust tests that call each builder and assert proto creation succeeds.
2. Verify `CombineAsSegment` builds a segment with `dim_size=5` and correct config types.
3. Implement `HashTableConfigInstance` string or equality behavior to match Python `__str__` semantics.
4. Add test cases that compare string outputs for numeric vs callable learning rate functions.

**Tests (Detailed)**
- Python tests: `monolith/native_training/entry_test.py`.
- Rust tests: `monolith-rs/crates/monolith-hash-table/tests/entry_test.rs`.
- Cross-language parity test: compare serialized proto bytes and `__str__` outputs for the same configs.

**Gaps / Notes**
- Python uses `learning_rate_functions.PolynomialDecay` for callable learning-rate fns; Rust needs an equivalent or a stub to produce deterministic string output.

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

### `monolith/native_training/env_utils.py`
<a id="monolith-native-training-env-utils-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 32
- Purpose/role: Minimal environment utility stubs for HDFS setup, UUID->PSM conversion, and ZooKeeper auth data.
- Key symbols/classes/functions: `setup_hdfs_env`, `generate_psm_from_uuid`, `get_zk_auth_data`.
- External dependencies: `os`, `absl.logging` (unused), `contextlib`, `hashlib`, `subprocess`, `socket` (unused).
- Side effects: `get_zk_auth_data` prints `ZK_AUTH` to stdout when set.

**Required Behavior (Detailed)**
- `setup_hdfs_env()`:
  - Currently a no-op (`pass`).
- `generate_psm_from_uuid(s)`:
  - Returns the input string unchanged.
- `get_zk_auth_data()`:
  - Reads `ZK_AUTH` env var.
  - If set, prints `"ZK_AUTH <value>"` and returns `[('digest', ZK_AUTH)]`.
  - If unset, returns `None`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/env_utils.rs` (new) or `monolith-rs/crates/monolith-core/src/env_utils.rs`.
- Rust public API surface: `setup_hdfs_env`, `generate_psm_from_uuid`, `get_zk_auth_data` equivalents.
- Data model mapping: `get_zk_auth_data` returns `Option<Vec<(String, String)>>` or a domain-specific auth struct.
- Feature gating: none; functions are pure env utilities.
- Integration points: any ZooKeeper or HDFS setup codepaths in Rust equivalents.

**Implementation Steps (Detailed)**
1. Implement `setup_hdfs_env` as a no-op in Rust until actual HDFS setup logic is defined.
2. Implement `generate_psm_from_uuid` as identity function.
3. Implement `get_zk_auth_data` to read `ZK_AUTH` and return digest tuple; log/print similarly.
4. Add tests to verify behavior with `ZK_AUTH` set/unset.

**Tests (Detailed)**
- Python tests: none (see `env_utils_test.py`, empty).
- Rust tests: add unit tests in `monolith-rs/crates/monolith-training/tests/env_utils_test.rs`.
- Cross-language parity test: set `ZK_AUTH` env and compare return value.

**Gaps / Notes**
- Many imports are unused; indicates incomplete implementation in Python.
- If upstream has a richer implementation, this file should be revisited for parity.

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

### `monolith/native_training/env_utils_test.py`
<a id="monolith-native-training-env-utils-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 23
- Purpose/role: Placeholder test file; no tests implemented.
- Key symbols/classes/functions: None.
- External dependencies: `os`, `unittest`, `mock`, `env_utils` (unused).
- Side effects: None.

**Required Behavior (Detailed)**
- No runtime behavior; file only defines imports and `unittest.main()` when run as main.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests/env_utils_test.rs` (new).
- Rust public API surface: tests for `env_utils` functions.
- Feature gating: none.
- Integration points: ensure env parsing helpers are covered by unit tests.

**Implementation Steps (Detailed)**
1. Implement Rust tests for `get_zk_auth_data` with and without `ZK_AUTH`.
2. Add a smoke test for `generate_psm_from_uuid` identity behavior.
3. Keep `setup_hdfs_env` as no-op test (ensures no panic).

**Tests (Detailed)**
- Python tests: none.
- Rust tests: `monolith-rs/crates/monolith-training/tests/env_utils_test.rs`.
- Cross-language parity test: not required beyond env value checks.

**Gaps / Notes**
- This file has no actual tests; Rust should still cover behavior to keep parity validated.

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

### `monolith/native_training/estimator.py`
<a id="monolith-native-training-estimator-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 667
- Purpose/role: High-level Estimator API for local/distributed training, evaluation, prediction, and saved_model export/import.
- Key symbols/classes/functions: `EstimatorSpec`, `RunConfig`, `Estimator`, `import_saved_model`.
- External dependencies: TensorFlow Estimator, Kazoo/ZK, AgentService, CpuTraining, RunnerConfig, service discovery, DumpUtils, device_utils, distribution_utils.
- Side effects: mutates env vars, initializes ZK clients/backends, logs metrics, writes model dumps, may start/stop distributed services.

**Required Behavior (Detailed)**
- `EstimatorSpec` (namedtuple):
  - Fields: `label`, `pred`, `head_name`, `loss`, `optimizer`, `classification`.
  - `__new__` sets defaults `head_name=None`, `loss=None`, `optimizer=None`, `classification=True`.
  - `_replace` forbids changing `mode` if present in kwargs and different from existing (defensive check).
- `RunConfig` (dataclass_json):
  - Fields include: `is_local`, `num_ps`, `num_workers`, timeout settings, layout flags, retry settings, parameter-sync fields, checkpoint/export settings, profiling flags, alias map settings, kafka settings, metrics flags, summary/log cadence.
  - `to_runner_config()`:
    - Builds `RunnerConfig` with key fields.
    - For each RunConfig field, if value differs from RunConfig default and differs from current conf value, updates conf (preserves CLI overrides).
    - Converts `ServiceDiscoveryType.CONSUL` to `ServiceDiscoveryType.ZK`.
    - If `enable_gpu_training` is False, ensures `embedding_prefetch_capacity >= 1` and `enable_embedding_postpush=True`.
    - If `enable_parameter_sync` is True, sets `enable_realtime_training` or `enable_parameter_sync` on `RunnerConfig` (raises if neither exists).
  - `__post_init__`:
    - Serializes to JSON and records config in `DumpUtils`.
    - Records user params that differ from defaults to `DumpUtils`.
- `Estimator.__init__(model, conf, warm_start_from=None)`:
  - Converts `RunConfig` to `RunnerConfig` if needed.
  - Sets deep-insight metrics on the model based on local vs distributed and runner_conf values.
  - If realtime training on PS, initializes sync backend via ZK (either `ZKBackend` or `ReplicaWatcher` with `MonolithKazooClient`).
  - Applies `params_override` JSON to model `.p` or `.params` when present.
  - Attempts `env_utils.setup_hdfs_env()` if `HADOOP_HDFS_HOME` missing (logs errors).
  - Exports env vars `TF_GRPC_WORKER_CACHE_THREADS` and `MONOLITH_GRPC_WORKER_SERVICE_HANDLER_MULTIPLIER` from runner_conf.
- `Estimator._est` (lazy property):
  - Deep-copies model; sets mode `PREDICT` and instantiates `CpuTraining` task.
  - Deletes `TF_CONF` env var if present.
  - Constructs `tf.estimator.Estimator` with `model_fn`, `model_dir`, and `RunConfig(log_step_count_steps=...)`, with `warm_start_from` if provided.
- `Estimator.train(steps=None, max_steps=None, hooks=None)`:
  - Validates hooks are `tf.estimator.SessionRunHook`.
  - Sets metric prefix `monolith.training.<deep_insight_name>`.
  - Deep-copies model, sets mode `TRAIN`, overrides steps/max_steps if provided.
  - If local: choose model_dir (default `/tmp/<user>/<model>`), call `local_train_internal`, and write `DumpUtils` to `model_dump`.
  - If distributed: disable DumpUtils; start sync backend and subscribe model; log env + flags + params.
    - If `enable_full_sync_training`: call `init_sync_train_and_update_conf` then `distributed_sync_train`.
    - Else: use `monolith_discovery` context; if `enable_gpu_training` -> `device_utils.enable_gpu_training()` and disable `use_gpu_emb_table`; if partial sync and worker -> `try_init_cuda()` and set `device_fn`.
    - Call `distributed_train`.
  - Calls `close()` at end.
- `Estimator.evaluate(steps=None, hooks=None)`:
  - Mirrors `train()` but uses mode `EVAL` and `distributed_train` (no user hooks in distributed eval except full sync).
- `Estimator.predict(...)`:
  - Creates estimator via `_est`, builds input_fn, calls `est.predict(...)`, then `close()`.
- `Estimator.export_saved_model(batch_size=64, name=None, dense_only=False, enable_fused_layout=False)`:
  - Copies model/conf; sets `enable_fused_layout`, model name, batch size, mode `PREDICT`.
  - Creates `CpuTraining` task and exporter; uses `ParserCtx(enable_fused_layout=...)`.
  - Calls `exporter.export_saved_model` with `serving_input_receiver_fn`.
- `import_saved_model(saved_model_path, input_name='instances', output_name='output', signature=None)`:
  - Context manager that resolves latest numeric version directory if `saved_model_path` not numeric.
  - Loads SavedModel with `tf.saved_model.load`, chooses signature (default serving).
  - Builds placeholders dict from requested inputs.
  - Builds output dict from requested outputs or all outputs if none provided.
  - Returns `infer(features)` callable that runs session and maps output tensor names to output names.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/estimator.rs` plus new `run_config.rs`/`runner_config.rs` and export/import helpers.
- Rust public API surface:
  - `EstimatorSpec` struct analog for model outputs.
  - `RunConfig` struct with JSON serialization + `to_runner_config` merge semantics.
  - `Estimator` wrapper that orchestrates training/eval/predict, plus saved-model import/export if TF runtime is enabled.
- Data model mapping: use `monolith-training` model traits (`ModelFn`) for Candle backend; optional TF runtime path for SavedModel import/export.
- Feature gating: `tf-runtime` feature for SavedModel and TF Estimator parity; default Candle backend implements local training only.
- Integration points: `cpu_training`/`distributed_train` equivalents, service discovery, device utils, dump utilities, parameter sync backend.

**Implementation Steps (Detailed)**
1. Implement `RunConfig` in Rust with all fields and defaults; add JSON serialization to match Python `dataclass_json`.
2. Implement `to_runner_config` merge logic (only override when RunConfig value differs from default and from current runner config).
3. Implement `Estimator` struct with local training/eval/predict flows mirroring Python.
4. Add optional ZK/AgentService integration or stub with clear errors when unavailable.
5. Implement env var exports (`TF_GRPC_WORKER_CACHE_THREADS`, `MONOLITH_GRPC_WORKER_SERVICE_HANDLER_MULTIPLIER`).
6. Implement SavedModel export/import only when TF runtime is enabled; otherwise document as unsupported.
7. Add parity tests for RunConfig merging and local train/eval/predict call flow.

**Tests (Detailed)**
- Python tests: `estimator_test.py`, `estimator_dist_test.py`, `estimator_mode_test.py`.
- Rust tests: add `estimator_test.rs` for local flow and config merging; integration tests for distributed modes as available.
- Cross-language parity test: compare config merge outputs, model_dir resolution, and SavedModel import/export behavior.

**Gaps / Notes**
- Python relies on TF Estimator and distributed training stack; Rust currently has only stubs for distributed execution.
- SavedModel import/export likely requires TF runtime and custom ops.

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

### `monolith/native_training/estimator_dist_test.py`
<a id="monolith-native-training-estimator-dist-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 166
- Purpose/role: Integration test for distributed training/eval using TF_CONFIG-style discovery and multi-process PS/worker setup.
- Key symbols/classes/functions: `EstimatorTrainTest`, `get_cluster`, `get_free_port`.
- External dependencies: `tensorflow`, `RunnerConfig`, `TestFFMModel`, `TfConfigServiceDiscovery`, `Estimator`.
- Side effects: Spawns multiple processes, binds local ports, writes checkpoints under tmp.

**Required Behavior (Detailed)**
- `get_free_port()`:
  - Binds a local socket on port 0 to find an available port; closes socket and returns port.
- `get_cluster(ps_num, worker_num)`:
  - Returns dict with `ps`, `worker`, and `chief` addresses on free ports (workers exclude chief).
- `EstimatorTrainTest.setUpClass`:
  - Removes existing `model_dir` if present.
  - Creates `TestFFMModel` params with deep insight disabled and batch size 64.
- `EstimatorTrainTest.train()`:
  - Spawns `ps_num` PS processes and `worker_num` worker/chief processes.
  - Each process builds `TF_CONFIG`-like dict, uses `TfConfigServiceDiscovery`, constructs `RunnerConfig`, and calls `Estimator.train(steps=10)`.
  - Waits for all processes; asserts exitcode 0 for each.
- `EstimatorTrainTest.eval()`:
  - Same as train but calls `Estimator.evaluate(steps=10)`.
- `test_dist`:
  - Runs `train()` then `eval()`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests/estimator_dist_test.rs` (new).
- Rust public API surface: distributed training harness and config discovery equivalent.
- Data model mapping: distributed cluster config (ps/worker/chief), runner config, model params.
- Feature gating: likely `tf-runtime` or `distributed` feature; skip if distributed stack not available.
- Integration points: service discovery and process orchestration.

**Implementation Steps (Detailed)**
1. Implement a Rust integration test that spawns multiple processes (or threads) for PS/worker roles.
2. Provide a discovery config equivalent to `TfConfigServiceDiscovery` and ensure `Estimator` can use it.
3. Run short train/eval steps and assert clean exit.
4. Add timeouts and cleanup for spawned processes and temp directories.

**Tests (Detailed)**
- Python tests: `monolith/native_training/estimator_dist_test.py`.
- Rust tests: `monolith-rs/crates/monolith-training/tests/estimator_dist_test.rs` (integration).
- Cross-language parity test: verify training/eval complete under equivalent cluster topology.

**Gaps / Notes**
- Uses real multi-process TF; Rust currently lacks distributed PS/worker runtime.
- Port binding is fragile; consider deterministic port assignment for CI.

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

### `monolith/native_training/estimator_mode_test.py`
<a id="monolith-native-training-estimator-mode-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 417
- Purpose/role: End-to-end integration tests for multiple distributed modes (CPU, sparse+ dense GPU, full GPU) by launching the training binary with various env/config permutations.
- Key symbols/classes/functions: `DistributedTrainTest`, `_run_test`, `run_cpu`, `sparse_dense_run`, `full_gpu_run`.
- External dependencies: TensorFlow, `RunnerConfig`, training binary `monolith/native_training/tasks/sparse_dense_gpu/model`, `gen_input_file`, `MultiHeadModel`, `test_util`.
- Side effects: Creates temp dataset files, spawns multiple processes (including mpirun), sets many env vars, writes logs, deletes temp dirs.

**Required Behavior (Detailed)**
- `setUpClass`:
  - Generates dataset file via `gen_input_file` and creates symlinks for suffixes 0..9.
  - Updates `FLAGS.dataset_input_patterns` to include `{INT(0,99)}`.
- `find_free_port(count)`:
  - Finds `count` available local ports (no reuse).
- `_run_test(...)`:
  - Creates `cur_modir` under test tmp dir; removes existing.
  - Builds `args_tmpl` list for the training binary with flags:
    - mode=train, model_dir, num_ps/workers, uuid, dataset flags, discovery settings, timeouts, metrics disable, dataservice toggle, cluster type.
  - Populates MLP_* env vars per role via `fill_host_env`.
  - Allocates ports for PS/worker/dsworker/dispatcher and sets env accordingly.
  - `start_process`:
    - For `use_mpi_run=True`, writes a hostfile and uses `mpirun` with Horovod-related env exports.
    - Else, spawns subprocess per role with `MLP_ROLE`, `MLP_ROLE_INDEX`, `MLP_PORT`, and `MLP_SSH_PORT` envs, writing logs to files.
  - Starts dispatcher, dsworker, ps, worker processes.
  - `wait_for_process` enforces timeouts; may terminate on timeout when `ignore_timeout=True`.
  - Cleans up log files and removes `cur_modir`.
- `run_cpu(...)`:
  - Skips if GPU is available; runs CPU cases with `enable_gpu_training=False`, `enable_sync_training=False` and embedding prefetch/postpush flags.
- `sparse_dense_run(...)`:
  - Requires GPU; uses MPI run and sets sync training flags, partial sync, params_override, and dataset service.
- `full_gpu_run(...)`:
  - Requires GPU; uses MPI run with `enable_sync_training`, `reorder_fids_in_data_pipeline`, `filter_type=probabilistic_filter`, `enable_async_optimize=False`.
- Test variants:
  - CPU tests `test_cpu0..3` vary `enable_fused_layout` and `use_native_multi_hash_table`.
  - Sparse+Dense GPU tests `test_sparse_dense0..3` vary layout and native hash table.
  - Full GPU tests `test_full_gpu_0..3` vary layout and native hash table.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests/estimator_mode_test.rs` (new) or CI scripts.
- Rust public API surface: distributed training CLI entrypoint and env-based cluster discovery.
- Feature gating: GPU and MPI required; tests should be behind `gpu`/`mpi` feature flags and skipped in CI by default.
- Integration points: training binary CLI, dataset service, Horovod/BytePS integration.

**Implementation Steps (Detailed)**
1. Implement or stub the Rust training binary to accept similar CLI flags.
2. Add integration tests that spawn subprocesses with env roles for PS/worker/dsworker/dispatcher.
3. Add MPI-based test harness if Horovod/BytePS parity is required.
4. Ensure temp dirs and logs are cleaned up even on failure.

**Tests (Detailed)**
- Python tests: `monolith/native_training/estimator_mode_test.py`.
- Rust tests: integration tests in `monolith-rs/crates/monolith-training/tests/` or CI scripts.
- Cross-language parity test: compare training completion and exit codes across CPU/GPU modes.

**Gaps / Notes**
- Heavy integration tests require external binaries and GPU; likely to be skipped in Rust CI.
- Port allocation is fragile; may need reserved port ranges.

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

### `monolith/native_training/estimator_test.py`
<a id="monolith-native-training-estimator-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 112
- Purpose/role: Local-mode Estimator smoke tests for train/eval/predict/export/import flow.
- Key symbols/classes/functions: `EstimatorTrainTest`, `get_saved_model_path`.
- External dependencies: TensorFlow, `RunnerConfig`, `TestFFMModel`, `generate_ffm_example`, `import_saved_model`.
- Side effects: Writes checkpoints and exported models under temp dirs; performs inference loops.

**Required Behavior (Detailed)**
- `setUpClass`:
  - Removes existing model_dir if present.
  - Sets model params: deep insight disabled, batch size 64, export dir base, `shared_embedding=True`.
  - Creates `RunnerConfig(is_local=True, num_ps=0, model_dir=..., use_native_multi_hash_table=False)`.
- `train/eval/predict`:
  - Instantiate `Estimator` and call the respective method (steps=10 for train/eval).
- `export_saved_model`:
  - Calls `Estimator.export_saved_model()` with defaults.
- `import_saved_model`:
  - Uses latest saved model dir from `export_base`.
  - Runs inference through `import_saved_model` context for 10 iterations, with generated FFM examples.
- `test_local`:
  - Runs train, eval, predict, export, and import in sequence.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests/estimator_test.rs` (new).
- Rust public API surface: local Estimator train/eval/predict/export/import.
- Feature gating: SavedModel export/import requires `tf-runtime`; otherwise stub or skip.
- Integration points: training data generation, model definition, export path handling.

**Implementation Steps (Detailed)**
1. Implement a local-only Estimator test in Rust that runs train/eval/predict with a simple model.
2. Add SavedModel export/import tests behind `tf-runtime` feature.
3. Ensure temp dirs are cleaned up after tests.

**Tests (Detailed)**
- Python tests: `monolith/native_training/estimator_test.py`.
- Rust tests: `monolith-rs/crates/monolith-training/tests/estimator_test.rs`.
- Cross-language parity test: compare export/import outputs on fixed inputs.

**Gaps / Notes**
- Python import_saved_model uses TF sessions and custom ops; Rust parity likely requires TF runtime.

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
