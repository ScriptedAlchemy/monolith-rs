<!--
Source: task/request.md
Lines: 7017-7224 (1-based, inclusive)
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
