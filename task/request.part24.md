<!--
Source: task/request.md
Lines: 6182-6305 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
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
