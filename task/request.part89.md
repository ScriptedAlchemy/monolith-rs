<!--
Source: task/request.md
Lines: 21025-21324 (1-based, inclusive)
Note: This file is auto-generated to keep prompt context bounded.
-->
### `monolith/native_training/test_utils.py`
<a id="monolith-native-training-test-utils-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 65
- Purpose/role: TODO (manual)
- Key symbols/classes/functions: TODO (manual)
- External dependencies: TODO (manual)
- Side effects: TODO (manual)

**Required Behavior (Detailed)**
- Define the **functional contract** (inputs → outputs) for every public function/class.
- Enumerate **error cases** and exact exception/messages that callers rely on.
- Capture **config + env var** behaviors (defaults, overrides, precedence).
- Document **I/O formats** used (proto shapes, TFRecord schemas, JSON, pbtxt).
- Note **threading/concurrency** assumptions (locks, async behavior, callbacks).
- Identify **determinism** requirements (seeds, ordering, float tolerances).
- Identify **performance characteristics** that must be preserved.
- Enumerate **metrics/logging** semantics (what is logged/when).

**Rust Mapping (Detailed)**
- Target crate/module: TODO (manual)
- Rust public API surface: TODO (manual)
- Data model mapping: TODO (manual)
- Feature gating: TODO (manual)
- Integration points: TODO (manual)

**Implementation Steps (Detailed)**
1. Extract all public symbols + docstrings; map to Rust equivalents.
2. Port pure logic first (helpers, utils), then stateful services.
3. Recreate exact input validation and error semantics.
4. Mirror side effects (files, env vars, sockets) in Rust.
5. Add config parsing and defaults matching Python behavior.
6. Add logging/metrics parity (field names, levels, cadence).
7. Integrate into call graph (link to downstream Rust modules).
8. Add tests and golden fixtures; compare outputs with Python.
9. Document deviations (if any) and mitigation plan.

**Tests (Detailed)**
- Python tests: TODO (manual)
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

### `monolith/native_training/touched_key_set_ops.py`
<a id="monolith-native-training-touched-key-set-ops-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 61
- Purpose/role: TODO (manual)
- Key symbols/classes/functions: TODO (manual)
- External dependencies: TODO (manual)
- Side effects: TODO (manual)

**Required Behavior (Detailed)**
- Define the **functional contract** (inputs → outputs) for every public function/class.
- Enumerate **error cases** and exact exception/messages that callers rely on.
- Capture **config + env var** behaviors (defaults, overrides, precedence).
- Document **I/O formats** used (proto shapes, TFRecord schemas, JSON, pbtxt).
- Note **threading/concurrency** assumptions (locks, async behavior, callbacks).
- Identify **determinism** requirements (seeds, ordering, float tolerances).
- Identify **performance characteristics** that must be preserved.
- Enumerate **metrics/logging** semantics (what is logged/when).

**Rust Mapping (Detailed)**
- Target crate/module: TODO (manual)
- Rust public API surface: TODO (manual)
- Data model mapping: TODO (manual)
- Feature gating: TODO (manual)
- Integration points: TODO (manual)

**Implementation Steps (Detailed)**
1. Extract all public symbols + docstrings; map to Rust equivalents.
2. Port pure logic first (helpers, utils), then stateful services.
3. Recreate exact input validation and error semantics.
4. Mirror side effects (files, env vars, sockets) in Rust.
5. Add config parsing and defaults matching Python behavior.
6. Add logging/metrics parity (field names, levels, cadence).
7. Integrate into call graph (link to downstream Rust modules).
8. Add tests and golden fixtures; compare outputs with Python.
9. Document deviations (if any) and mitigation plan.

**Tests (Detailed)**
- Python tests: TODO (manual)
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

### `monolith/native_training/touched_key_set_ops_test.py`
<a id="monolith-native-training-touched-key-set-ops-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 51
- Purpose/role: TODO (manual)
- Key symbols/classes/functions: TODO (manual)
- External dependencies: TODO (manual)
- Side effects: TODO (manual)

**Required Behavior (Detailed)**
- Define the **functional contract** (inputs → outputs) for every public function/class.
- Enumerate **error cases** and exact exception/messages that callers rely on.
- Capture **config + env var** behaviors (defaults, overrides, precedence).
- Document **I/O formats** used (proto shapes, TFRecord schemas, JSON, pbtxt).
- Note **threading/concurrency** assumptions (locks, async behavior, callbacks).
- Identify **determinism** requirements (seeds, ordering, float tolerances).
- Identify **performance characteristics** that must be preserved.
- Enumerate **metrics/logging** semantics (what is logged/when).

**Rust Mapping (Detailed)**
- Target crate/module: TODO (manual)
- Rust public API surface: TODO (manual)
- Data model mapping: TODO (manual)
- Feature gating: TODO (manual)
- Integration points: TODO (manual)

**Implementation Steps (Detailed)**
1. Extract all public symbols + docstrings; map to Rust equivalents.
2. Port pure logic first (helpers, utils), then stateful services.
3. Recreate exact input validation and error semantics.
4. Mirror side effects (files, env vars, sockets) in Rust.
5. Add config parsing and defaults matching Python behavior.
6. Add logging/metrics parity (field names, levels, cadence).
7. Integrate into call graph (link to downstream Rust modules).
8. Add tests and golden fixtures; compare outputs with Python.
9. Document deviations (if any) and mitigation plan.

**Tests (Detailed)**
- Python tests: TODO (manual)
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

### `monolith/native_training/utils.py`
<a id="monolith-native-training-utils-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 320
- Purpose/role: TODO (manual)
- Key symbols/classes/functions: TODO (manual)
- External dependencies: TODO (manual)
- Side effects: TODO (manual)

**Required Behavior (Detailed)**
- Define the **functional contract** (inputs → outputs) for every public function/class.
- Enumerate **error cases** and exact exception/messages that callers rely on.
- Capture **config + env var** behaviors (defaults, overrides, precedence).
- Document **I/O formats** used (proto shapes, TFRecord schemas, JSON, pbtxt).
- Note **threading/concurrency** assumptions (locks, async behavior, callbacks).
- Identify **determinism** requirements (seeds, ordering, float tolerances).
- Identify **performance characteristics** that must be preserved.
- Enumerate **metrics/logging** semantics (what is logged/when).

**Rust Mapping (Detailed)**
- Target crate/module: TODO (manual)
- Rust public API surface: TODO (manual)
- Data model mapping: TODO (manual)
- Feature gating: TODO (manual)
- Integration points: TODO (manual)

**Implementation Steps (Detailed)**
1. Extract all public symbols + docstrings; map to Rust equivalents.
2. Port pure logic first (helpers, utils), then stateful services.
3. Recreate exact input validation and error semantics.
4. Mirror side effects (files, env vars, sockets) in Rust.
5. Add config parsing and defaults matching Python behavior.
6. Add logging/metrics parity (field names, levels, cadence).
7. Integrate into call graph (link to downstream Rust modules).
8. Add tests and golden fixtures; compare outputs with Python.
9. Document deviations (if any) and mitigation plan.

**Tests (Detailed)**
- Python tests: TODO (manual)
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

### `monolith/native_training/utils_test.py`
<a id="monolith-native-training-utils-test-py"></a>

**Status:** TODO (manual review required)

**Python Summary**
- Lines: 70
- Purpose/role: TODO (manual)
- Key symbols/classes/functions: TODO (manual)
- External dependencies: TODO (manual)
- Side effects: TODO (manual)

**Required Behavior (Detailed)**
- Define the **functional contract** (inputs → outputs) for every public function/class.
- Enumerate **error cases** and exact exception/messages that callers rely on.
- Capture **config + env var** behaviors (defaults, overrides, precedence).
- Document **I/O formats** used (proto shapes, TFRecord schemas, JSON, pbtxt).
- Note **threading/concurrency** assumptions (locks, async behavior, callbacks).
- Identify **determinism** requirements (seeds, ordering, float tolerances).
- Identify **performance characteristics** that must be preserved.
- Enumerate **metrics/logging** semantics (what is logged/when).

**Rust Mapping (Detailed)**
- Target crate/module: TODO (manual)
- Rust public API surface: TODO (manual)
- Data model mapping: TODO (manual)
- Feature gating: TODO (manual)
- Integration points: TODO (manual)

**Implementation Steps (Detailed)**
1. Extract all public symbols + docstrings; map to Rust equivalents.
2. Port pure logic first (helpers, utils), then stateful services.
3. Recreate exact input validation and error semantics.
4. Mirror side effects (files, env vars, sockets) in Rust.
5. Add config parsing and defaults matching Python behavior.
6. Add logging/metrics parity (field names, levels, cadence).
7. Integrate into call graph (link to downstream Rust modules).
8. Add tests and golden fixtures; compare outputs with Python.
9. Document deviations (if any) and mitigation plan.

**Tests (Detailed)**
- Python tests: TODO (manual)
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
