# Agent: apply-agent-service-tfs-client

Ported a Python TFServing tooling subset into `monolith-rs/crates/monolith-serving`.

- Added client-side shaping helpers for `TensorProto(DT_STRING)` batches and
  `ExampleBatch` (column-major) -> `Instance` conversion.
- Added a TFServing monitor (`GetModelStatus`, `HandleReloadConfigRequest`,
  `ModelServerConfig` generation) matching Python semantics used by tests.
- Added an in-process fake TFServing gRPC server implementing:
  `ModelService/GetModelStatus`, `ModelService/HandleReloadConfigRequest`,
  `PredictionService/GetModelMetadata`.
- Added a lightweight wrapper around `model_config_file` pbtxt parsing and
  saved-model status listing.
- Added parity tests mirroring:
  `tfs_client_test.py`, `mocked_tfserving_test.py`, `tfs_monitor_test.py`.

Notes:
- Some repos may not include `examplebatch.data`; the Rust parity test generates
  a minimal framed file on the fly if missing.
