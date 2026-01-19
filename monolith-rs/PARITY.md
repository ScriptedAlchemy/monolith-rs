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

## Reality Check (January 2026)

This file was originally written early in the port and is now partially stale.
The Rust implementation has recently added real distributed PS gRPC, TF Serving
PredictionService support, and Candle-backed inference scaffolding.

If you are reading this for "what is still missing?", focus on the **Remaining
Gaps** section near the end.

## Top-Level Python Areas (Inventory)

### 1) Data pipeline + parsing (`monolith/native_training/data`, `monolith/native_training/feature.py`, etc.)

- Example/Instance protobuf formats: DONE (Rust `monolith-proto`, `monolith-data`)
- TFRecord read/write: DONE (Rust `monolith-data`)
- Kafka stream ingest: PARTIAL (Rust `monolith-data` has a mockable Kafka source; no rdkafka integration yet)
- Parquet ingest: PARTIAL (Rust `monolith-data` supports parquet behind feature flags)

### 2) Training / Estimator (`monolith/native_training/estimator.py`, runners)

- Estimator-style orchestration: PARTIAL (Rust `monolith-training`)
- Distributed PS/worker runtime (gRPC + discovery): DONE/PARTIAL
  - DONE: PS gRPC server + worker client with dedup/routing/aggregation (`crates/monolith-training/src/distributed_ps.rs`)
  - DONE: discovery backends for in-memory, TF_CONFIG, and MLP env (`crates/monolith-training/src/discovery.rs`, `crates/monolith-training/src/py_discovery.rs`)
  - PARTIAL: ZK/Consul parity (feature-gated, polling watch semantics; no TTL/ACL/ephemeral-node parity yet)
- TPU/GPU runner parity: N/A (Python is TensorFlow-backed; Rust uses its own tensor backends)

### 3) Hash tables + embeddings (`monolith/native_training/runtime/hash_table/**`)

- Embedding hash table core ops: PARTIAL (Rust `monolith-hash-table`)
- Checkpointing/serialization: PARTIAL (Rust `monolith-checkpoint`)
- Optimizer state for embeddings: PARTIAL (Rust `monolith-optimizer`, `monolith-hash-table`)

### 4) Model export + loading (`monolith/native_training/model_export/**`)

- Exported manifest + loading: PARTIAL (Rust `monolith-checkpoint`, `monolith-serving`, examples: `model_export`, `local_predictor`)
- "SavedModel-like" export dir for Rust serving: DONE (Rust `monolith-checkpoint` `ExportFormat::SavedModel`)
  - Writes `dense/params.json`, `embeddings/*.json`, `manifest.json`
  - Writes `model_spec.json` when ModelState metadata contains model spec fields
- TensorFlow SavedModel export parity: N/A (Rust does not emit TF GraphDef/SavedModel)

### 5) Serving + agent (`monolith/agent_service/**`, TF Serving integration)

- Monolith AgentService gRPC (agent_service.proto): DONE (Rust `monolith-serving` + `monolith-proto`)
- ParameterSync RPC (embedding delta sync): DONE/PARTIAL
  - DONE: ParameterSync gRPC server + client + "dirty" replication hooks
  - PARTIAL: production semantics (backpressure, retry, checkpoints for deltas)
- TF Serving PredictionService (Predict + Example decoding): DONE
  - Server: `crates/monolith-serving/src/tfserving_server.rs`
  - Client + pbtxt ModelServerConfig parsing: DONE/PARTIAL (enough for tests and basic config flows)
- TF Serving ModelService parity: PARTIAL (not all RPCs implemented)
- Replica manager / model manager / monitoring loops (ZK/Consul + filesystem watcher): PLANNED/PARTIAL

### 6) Service discovery (`monolith/native_training/service_discovery.py`, ZK/Consul)

- In-memory discovery: DONE (Rust `monolith-training`)
- TF_CONFIG discovery parity: PARTIAL (Rust `monolith-training`)
- ZooKeeper / Consul real backends: PARTIAL (feature-gated in `crates/monolith-training/src/discovery.rs`)
  - Missing: ephemeral nodes + session lifecycle for ZK; TTL/health checks/ACL tokens + blocking queries for Consul

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

## Remaining Gaps (Most Important For "Python Parity")

This is the current (Jan 2026) short list of what still blocks "Rust can do what
Python Monolith can do" for typical production workflows:

1) **Export CLI is still a stub**
   - `crates/monolith-cli/src/commands/export.rs` does not yet load a checkpoint and call
     `monolith_checkpoint::ModelExporter`.
   - Workaround today: call exporter from Rust code/tests; CLI needs wiring.

2) **Serving model-family parity**
   - Candle inference currently supports `mlp`, `dcn`, `mmoe` via `model_spec.json`.
   - Python Monolith commonly uses DIN/DIEN patterns; Rust has DIN/DIEN in `monolith-layers`,
     but serving does not yet load/execute them via Candle.

3) **Distributed training semantics vs Python TensorFlow**
   - We have a real PS gRPC implementation with dedup/routing/gradient aggregation + barrier,
     but it is not a full replacement for TensorFlow's distributed runtime:
     allreduce, fault tolerance, and TF graph/device semantics are out of scope.

4) **ZK/Consul production parity**
   - Feature-gated backends exist, but missing core operational semantics:
     ZK ephemeral nodes/watches, Consul TTL keepalive/health checks/ACLs/blocking queries.

5) **GPU/Metal test stability**
   - Candle/Metal can crash under parallel test execution; we default to CPU in tests and
     allow `MONOLITH_FORCE_CPU=1` to force CPU in any build.

If you want the fastest path to "feels like Python", implement (1) export CLI wiring,
then (2) DIN/DIEN Candle inference + export naming conventions, then (3) ops hardening.
