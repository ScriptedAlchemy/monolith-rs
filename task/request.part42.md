<!--
Source: task/request.md
Lines: 9786-9950 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
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
