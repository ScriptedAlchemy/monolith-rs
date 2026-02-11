# Workspace/Third-Party Notes (Rust)

This repository also contains a Bazel-based Python/TensorFlow tree (`monolith/` +
`third_party/`). The Rust workspace (`monolith-rs/`) intentionally does **not**
vendor TensorFlow runtime binaries.

What we do map:
- TensorFlow / TF Serving protos: `monolith-rs/proto/tensorflow/**`,
  `monolith-rs/proto/tensorflow_serving/**` compiled by `crates/monolith-proto`.
- ZooKeeper client support: feature-gated in `crates/monolith-training`
  (`--features zookeeper`) using `zookeeper-client` crate.
- TF Serving control-plane compatibility: `crates/monolith-serving` implements
  TF Serving gRPC APIs against Rust backends.

What we do not map:
- `third_party/org_tensorflow/**` and `third_party/org_tensorflow_serving/**`
  Bazel patches are not applied to Rust builds.
- No `libtensorflow.so` / SavedModel execution is embedded. Any TF runtime usage
  must be external and optional.

