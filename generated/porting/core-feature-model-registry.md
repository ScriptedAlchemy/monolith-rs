# core-feature-model-registry

Implemented Rust parity coverage for Python core feature + model registry surfaces.

## Feature parity (monolith/core/feature.py + feature_test.py)

- Added Sail-like feature types in `monolith-rs/crates/monolith-core/src/feature.rs`:
  - `SailEnv`
  - `SailFeatureSlot`
  - `SailFeatureSlice` (with Python `__repr__` parity via `Display`)
  - `FeatureColumnV1`
  - `EmbeddingTensor` (minimal 2D tensor for merge/split tests)
- Implemented merge/split behavior used by Python tests:
  - `SailEnv::merge_vector_in_same_slot`
  - `SailEnv::split_merged_embedding`
- Added Rust unit tests mirroring `monolith/core/feature_test.py` assertions.

## Model registry parity (monolith/core/model_registry.py)

- Updated `monolith-rs/crates/monolith-core/src/model_registry.rs` to use Python-parity error messages:
  - Duplicate registration: `Duplicate model registered for key {key}: {module}.{class_name}`
  - Missing model: `Model {key} not found from list of above known models.`
- Added a test-only helper `clear_registry_for_test()` and unit tests to lock error message parity.

## Notes

- The existing Rust batch feature types (`FeatureSlot`, `FeatureSlice`, `SparseFeatureColumn`, `DenseFeatureColumn`) are kept intact for downstream crates.
- Python `monolith/core/model.py` and `model_imports.py` are TensorFlow/import-system specific; no additional Rust runtime implementation was required for the current `monolith-core` crate tests.
