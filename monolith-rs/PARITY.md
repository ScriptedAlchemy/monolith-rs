# Monolith Python -> Monolith-RS Parity Tracker

This repo contains both:
- `monolith/` (Python + TensorFlow implementation)
- `monolith-rs/` (Rust implementation)

"Port all Python capabilities" is a multi-phase effort. This file tracks what
exists in Rust today vs what exists in Python, focusing on **common user flows**
(train -> export -> serve -> monitor) rather than 1:1 TensorFlow API surface.

## Status Legend

- DONE: implemented in Rust with tests/examples
- PARTIAL: implemented but missing key behaviors / gaps / limited scope
- PLANNED: tracked but not implemented yet
- N/A: Python-only (TensorFlow graph/runtime) with no Rust equivalent planned

## Top-Level Python Areas (Inventory)

### 1) Data pipeline + parsing (`monolith/native_training/data`, `monolith/native_training/feature.py`, etc.)

- Example/Instance protobuf formats: DONE (Rust `monolith-proto`, `monolith-data`)
- TFRecord read/write: DONE (Rust `monolith-data`)
- Kafka stream ingest: PARTIAL (Rust `monolith-data` has a mockable Kafka source; no rdkafka integration yet)
- Parquet ingest: PARTIAL (Rust `monolith-data` supports parquet behind feature flags)

### 2) Training / Estimator (`monolith/native_training/estimator.py`, runners)

- Estimator-style orchestration: PARTIAL (Rust `monolith-training`)
- Distributed PS/worker simulation: PARTIAL (Rust `monolith-training`, `monolith-rs/examples/distributed_training.rs`)
- TPU/GPU runner parity: N/A (Python is TensorFlow-backed; Rust uses its own tensor backends)

### 3) Hash tables + embeddings (`monolith/native_training/runtime/hash_table/**`)

- Embedding hash table core ops: PARTIAL (Rust `monolith-hash-table`)
- Checkpointing/serialization: PARTIAL (Rust `monolith-checkpoint`)
- Optimizer state for embeddings: PARTIAL (Rust `monolith-optimizer`, `monolith-hash-table`)

### 4) Model export + loading (`monolith/native_training/model_export/**`)

- Exported manifest + loading: PARTIAL (Rust `monolith-checkpoint`, `monolith-serving`, examples: `model_export`, `local_predictor`)
- TensorFlow SavedModel export parity: N/A (Rust does not emit TF SavedModel)

### 5) Serving + agent (`monolith/agent_service/**`, TF Serving integration)

- Monolith AgentService gRPC (agent_service.proto): DONE (Rust `monolith-serving` + `monolith-proto`)
- ParameterSync RPC (embedding delta sync): PARTIAL (Rust `monolith-serving`, `monolith-training`)
- TF Serving ModelService / PredictionService (GetModelStatus, Predict, pbtxt ModelServerConfig parsing):
  - PLANNED (needs TF Serving protos + pbtxt parsing + gRPC client)
- Replica manager / model manager / monitoring loops (ZK/Consul + filesystem watcher):
  - PLANNED

### 6) Service discovery (`monolith/native_training/service_discovery.py`, ZK/Consul)

- In-memory discovery: DONE (Rust `monolith-training`)
- TF_CONFIG discovery parity: PARTIAL (Rust `monolith-training`)
- ZooKeeper / Consul real backends: PARTIAL/PLANNED (depends on crate choices and operational requirements)

## Recommended Porting Phases

### Phase 1 (Most common local workflows)

- Data ingestion parity (TFRecord + ExampleBatch/Instance parsing) + tests (DONE/PARTIAL)
- Single-node training demo parity (DONE/PARTIAL)
- Export + local predictor parity (PARTIAL)

### Phase 2 (Production-serving workflows)

- TF Serving API client parity:
  - compile TF Serving protos into `monolith-proto` (or separate crate)
  - add pbtxt parsing for `ModelServerConfig` (text-format)
  - implement `GetModelStatus` + `Predict` clients
- Agent workflows parity:
  - model config generation
  - watcher/registrar loops

### Phase 3 (Distributed & ops)

- Real Kafka integration (rdkafka)
- ParameterSync end-to-end against real servers
- ZK/Consul discovery parity
- Observability parity (metrics/logging; dashboards not in scope)

## Next Step (Pick One)

To proceed efficiently, decide the highest-priority parity target:

1) TF Serving integration (ModelService/GetModelStatus + pbtxt ModelServerConfig parsing)
2) Kafka real integration (rdkafka-based consumer/producer for Example protos)
3) Distributed training runtime parity (PS/worker with discovery + sync/checkpoint)

