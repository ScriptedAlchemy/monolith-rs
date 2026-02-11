# core-utils-optimizers

Ported a small, unit-testable subset of Monolith core utilities and host-call helpers into Rust `monolith-core`, matching Python behavior where it is meaningful outside of TensorFlow.

## util.py + util_test.py

- Added `monolith-rs/crates/monolith-core/src/util.rs`:
  - `get_bucket_name_and_relavite_path()` with Python-style assertion message (`"File name: ..."`).
  - `range_dateset()` implemented as a pure path filter; added Rust tests mirroring `monolith/core/util_test.py`.
  - Also ported small pure helpers (`parse_example_number_meta_file`, `calculate_shard_skip_file_number`) for reuse.

## py_utils.py

- Tightened Python-parity error messaging in `monolith-rs/crates/monolith-core/src/nested_map.rs`:
  - Invalid key -> `Invalid NestedMap key '...'`
  - Reserved key -> `... is a reserved key`
  - `Set()` into a non-map intermediate -> matches Python ValueError string.
  - Adjusted `FilterKeyVal()` empty-container deletion semantics to match Python `_DELETE` behavior.

## base_host_call.py + host_call.py

- Added `monolith-rs/crates/monolith-core/src/base_host_call.rs`:
  - Pure data implementation of tensor name tracking, dtype-group compression, and decompression (split + squeeze).
  - Includes unit tests for round-trip packing/decompression behavior.

## optimizers.py

- Added `monolith-rs/crates/monolith-core/src/optimizers.rs`:
  - Minimal registry-like parsing for Python keys (`adam`, `adagrad`, `momentum`, `rmsprop`).

## Notes / Not Ported

- `testing_utils.py`, `auto_checkpoint_feed_hook.py`, `tpu_variable.py`, and the TF-specific parts of `host_call.py` are tightly coupled to TensorFlow/TPU runtime and are not meaningfully portable into this Rust crate without a TF runtime. Only the pure helper logic was ported.

