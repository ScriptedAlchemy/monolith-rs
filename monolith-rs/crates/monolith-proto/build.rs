//! Protobuf / gRPC code generation for Monolith.
//!
//! This crate compiles the `.proto` files in `monolith-rs/proto/` using `prost`
//! and (when the `grpc` feature is enabled) generates tonic service stubs.

use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Re-run if any proto changes
    println!("cargo:rerun-if-changed=../../proto");
    println!("cargo:rerun-if-changed=../../proto/tensorflow_serving");
    println!("cargo:rerun-if-changed=../../proto/tensorflow");
    println!("cargo:rerun-if-changed=../../proto/google");

    let proto_root = PathBuf::from("../../proto");

    // Note: Some protos import `idl/matrix/proto/{feature,line_id}.proto`.
    // We provide those paths via symlinks under `proto/idl/matrix/proto/`.
    let includes = [proto_root.clone()];

    let protos = vec![
        // Core feature/instance formats
        //
        // IMPORTANT: Do NOT compile `feature.proto` or `line_id.proto` directly here.
        // They are imported via `idl/matrix/proto/{feature,line_id}.proto` and we
        // provide those import paths as symlinks. Compiling both the "root" path
        // and the imported path causes protoc to treat them as two different
        // files, resulting in duplicate symbol errors.
        proto_root.join("example.proto"),
        proto_root.join("proto_parser.proto"),
        // Hash table configs.
        //
        // IMPORTANT: Do NOT compile `optimizer.proto`, `initializer_config.proto`,
        // or `float_compressor.proto` directly here. They are imported via
        // `monolith/native_training/runtime/hash_table/**` paths and we provide
        // those import paths as symlinks. Compiling both paths triggers duplicate
        // symbol errors in protoc.
        proto_root.join("embedding_hash_table.proto"),
        proto_root.join("hash_table_ops.proto"),
        proto_root.join("multi_hash_table_ops.proto"),
        // Training / controller / export
        proto_root.join("data_op_config.proto"),
        proto_root.join("transform_config.proto"),
        proto_root.join("monolith_checkpoint_state.proto"),
        proto_root.join("export.proto"),
        proto_root.join("ckpt_hooks.proto"),
        proto_root.join("controller_hooks.proto"),
        proto_root.join("service.proto"),
        // Serving / sync / misc services
        proto_root.join("agent_service.proto"),
        proto_root.join("parameter_sync.proto"),
        proto_root.join("parameter_sync_rpc.proto"),
        // PS Training RPC service (for distributed training)
        proto_root.join("ps_training.proto"),
        proto_root.join("logging_ops.proto"),
        proto_root.join("alert.proto"),
        proto_root.join("ckpt_info.proto"),
        proto_root.join("debugging_info.proto"),
        proto_root.join("monolith_model.proto"),
        // NOTE: `primus_am_service.proto` imports google/protobuf wrappers; we
        // currently don't expose a `google` module from `monolith-proto` (and
        // Primus isn't used by the Rust parity work yet), so skip compiling it
        // to keep the crate building cleanly.

        // TensorFlow Serving (tonic/prost)
        proto_root.join("tensorflow_serving/apis/model_service.proto"),
        proto_root.join("tensorflow_serving/apis/get_model_status.proto"),
        proto_root.join("tensorflow_serving/apis/get_model_metadata.proto"),
        proto_root.join("tensorflow_serving/apis/predict.proto"),
        proto_root.join("tensorflow_serving/apis/prediction_service.proto"),
        proto_root.join("tensorflow_serving/apis/model_management.proto"),
        proto_root.join("tensorflow_serving/config/model_server_config.proto"),
    ];

    // Make sure changes to indirectly imported protos trigger rebuilds too.
    println!("cargo:rerun-if-changed=../../proto/feature.proto");
    println!("cargo:rerun-if-changed=../../proto/line_id.proto");
    println!("cargo:rerun-if-changed=../../proto/float_compressor.proto");
    println!("cargo:rerun-if-changed=../../proto/initializer_config.proto");
    println!("cargo:rerun-if-changed=../../proto/optimizer.proto");
    println!("cargo:rerun-if-changed=../../proto/xla/tsl/protobuf/error_codes.proto");

    // Convert to string paths for tonic-build.
    let proto_strs: Vec<_> = protos
        .iter()
        .map(|p| p.to_string_lossy().to_string())
        .collect();
    let include_strs: Vec<_> = includes
        .iter()
        .map(|p| p.to_string_lossy().to_string())
        .collect();

    // Use tonic-build so we get both prost messages and tonic service stubs.
    tonic_build::configure()
        .build_client(true)
        .build_server(true)
        // Emit a descriptor set so other crates can do reflection-based pbtxt parsing.
        .file_descriptor_set_path(std::env::var("OUT_DIR")? + "/descriptor.bin")
        // Generate WKTs locally so wrapper types like BoolValue exist.
        .compile_well_known_types(false)
        // Keep compilation robust even if proto3 optional appears.
        .protoc_arg("--experimental_allow_proto3_optional")
        .compile(&proto_strs, &include_strs)?;

    Ok(())
}
