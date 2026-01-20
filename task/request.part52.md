<!--
Source: task/request.md
Lines: 12028-12264 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/hooks/server/server_lib_test.py`
<a id="monolith-native-training-hooks-server-server-lib-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 54
- Purpose/role: Integration test for controller gRPC server and client helper.
- Key symbols/classes/functions: `ServerTest.test_basic`.
- External dependencies: TensorFlow, gRPC, `server_lib`, `client_lib`, `barrier_ops`, `save_utils`.
- Side effects: Starts server in monitored session.

**Required Behavior (Detailed)**
- Starts ServerHook and saver hook in a session.
- Uses client stub from model_dir to:
  - StopTraining (second StopTraining should raise RpcError).
  - GetBlockStatus shows blocked index then unblocked after ResumeTraining.
  - SaveCheckpoint and GetTrainingStatus succeed.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests/server_lib_test.rs` (new).
- Rust public API surface: controller server hook and client stub.
- Feature gating: gRPC required.

**Implementation Steps (Detailed)**
1. Start controller server hook and saver hook in test session.
2. Use client to call gRPC methods and assert barrier behavior.

**Tests (Detailed)**
- Python tests: `server_lib_test.py`.
- Rust tests: `monolith-rs/crates/monolith-training/tests/server_lib_test.rs`.
- Cross-language parity test: compare gRPC behavior and responses.

**Gaps / Notes**
- Needs a working saver hook to test SaveCheckpoint path.

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

### `monolith/native_training/hooks/session_hooks.py`
<a id="monolith-native-training-hooks-session-hooks-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 44
- Purpose/role: Tracks the current TF session via a hook and provides a helper to fetch it.
- Key symbols/classes/functions: `SetCurrentSessionHook`, `get_current_session`.
- External dependencies: TensorFlow.
- Side effects: Stores session in a module-level singleton during hook lifetime.

**Required Behavior (Detailed)**
- `SetCurrentSessionHook.after_create_session` sets `_INFO.session`.
- `SetCurrentSessionHook.end` clears `_INFO.session`.
- `get_current_session()` returns `_INFO.session` if set, else `tf.compat.v1.get_default_session()`.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/hooks/session_hooks.rs` (new).
- Rust public API surface: session tracking hook + helper to fetch current session.
- Feature gating: TF runtime only.

**Implementation Steps (Detailed)**
1. Add a thread-safe global or TLS slot for current session.
2. Hook into session creation/end to set/clear.
3. Provide getter that falls back to default session.

**Tests (Detailed)**
- Python tests: `session_hooks_test.py`.
- Rust tests: `monolith-rs/crates/monolith-training/tests/session_hooks_test.rs`.
- Cross-language parity test: ensure session is available inside monitored session and cleared after.

**Gaps / Notes**
- Global session state should be scoped carefully in multi-session environments.

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

### `monolith/native_training/hooks/session_hooks_test.py`
<a id="monolith-native-training-hooks-session-hooks-test-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 33
- Purpose/role: Smoke test for current-session tracking.
- Key symbols/classes/functions: `SessionHooksTest.testBasic`.
- External dependencies: TensorFlow, `session_hooks`.
- Side effects: None.

**Required Behavior (Detailed)**
- Asserts `get_current_session()` is None outside a session.
- Inside MonitoredSession with `SetCurrentSessionHook`, `get_current_session()` returns non-None.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/tests/session_hooks_test.rs` (new).
- Rust public API surface: session tracking helper.
- Feature gating: TF runtime.

**Implementation Steps (Detailed)**
1. Add test that checks current session availability inside hook-managed session.
2. Ensure session is cleared after end.

**Tests (Detailed)**
- Python tests: `session_hooks_test.py`.
- Rust tests: `monolith-rs/crates/monolith-training/tests/session_hooks_test.rs`.
- Cross-language parity test: not required beyond smoke behavior.

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

### `monolith/native_training/hvd_lib.py`
<a id="monolith-native-training-hvd-lib-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 65
- Purpose/role: Lazy import wrapper for Horovod or BytePS TensorFlow libraries.
- Key symbols/classes/functions: `_Lib`, module-level `__getattr__`.
- External dependencies: `byteps.tensorflow` or `horovod.tensorflow`.
- Side effects: Imports the chosen library on first use.

**Required Behavior (Detailed)**
- `_Lib.enable_bps` reads `MONOLITH_WITH_BYTEPS` env var.
- `lib` property imports BytePS if enabled, else Horovod.
- Provides passthrough methods: `init`, `rank`, `size`, `allgather`, `broadcast`, `BroadcastGlobalVariablesHook`.
- Module-level `__getattr__` forwards to `_Lib` methods.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-training/src/hvd_lib.rs` (new).
- Rust public API surface: wrapper interface with lazy initialization for Horovod/BytePS bindings.
- Feature gating: only available under TF runtime / distributed features.

**Implementation Steps (Detailed)**
1. Implement lazy initialization with mutex/once for BytePS or Horovod bindings.
2. Expose helper methods mirroring Python names.
3. Read `MONOLITH_WITH_BYTEPS` to choose backend.

**Tests (Detailed)**
- Python tests: none.
- Rust tests: add smoke tests to verify backend selection and lazy init.
- Cross-language parity test: ensure backend selection matches env.

**Gaps / Notes**
- Requires actual Horovod/BytePS bindings for Rust; otherwise should be stubbed with clear errors.

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

### `monolith/native_training/input.py`
<a id="monolith-native-training-input-py"></a>

**Status:** IN PROGRESS (manual)

**Python Summary**
- Lines: 45
- Purpose/role: Utility to generate random FFM training examples.
- Key symbols/classes/functions: `slot_to_key`, `generate_ffm_example`.
- External dependencies: NumPy, TensorFlow.
- Side effects: None.

**Required Behavior (Detailed)**
- `slot_to_key(slot)` returns `"feature_<slot>"`.
- `generate_ffm_example(vocab_sizes, length=5)`:
  - Creates label feature with random int in [0,1) (effectively 0).
  - For each vocab size, samples `num_ids` in `[1, length]` and ids in a range offset by `max_vocab * i`.
  - Constructs a `tf.train.Example` and returns serialized bytes.

**Rust Mapping (Detailed)**
- Target crate/module: `monolith-rs/crates/monolith-data/src/input.rs` (new) or `monolith-rs/crates/monolith-examples` helpers.
- Rust public API surface: helper to generate serialized Example protos.
- Data model mapping: `monolith::io::proto::Example` or TensorFlow Example proto.

**Implementation Steps (Detailed)**
1. Implement `slot_to_key` helper.
2. Implement random example generation with identical id ranges.
3. Serialize Example protobuf to bytes.

**Tests (Detailed)**
- Python tests: used indirectly in estimator tests.
- Rust tests: add deterministic test with fixed RNG seed.
- Cross-language parity test: compare serialized Example with fixed seed.

**Gaps / Notes**
- Python uses `np.random.randint(low=0, high=1)` which always yields 0; consider whether to preserve this.

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
