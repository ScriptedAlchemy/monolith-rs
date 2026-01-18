//! Generated protobuf and gRPC code for Monolith.
//!
//! This crate compiles the `.proto` files under `monolith-rs/proto/` and
//! exposes the generated Rust types (prost messages) and tonic service stubs.
//!
//! The source `.proto` files are a direct copy of the upstream Monolith
//! repository, so these types are intended to match the Python-generated
//! protobuf classes 1:1.

#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(clippy::large_enum_variant)]

// NOTE: The generated modules are keyed by `package` in the `.proto` files and
// mapped to snake_case module paths by prost/tonic.

pub mod idl {
    pub mod matrix {
        pub mod proto {
            tonic::include_proto!("idl.matrix.proto");
        }
    }
}

pub mod monolith {
    pub mod io {
        pub mod proto {
            tonic::include_proto!("monolith.io.proto");
        }
    }

    pub mod hash_table {
        tonic::include_proto!("monolith.hash_table");
    }

    pub mod parameter_sync {
        tonic::include_proto!("monolith.parameter_sync");
    }

    pub mod parameter_sync_rpc {
        tonic::include_proto!("monolith.parameter_sync_rpc");
    }

    pub mod ps_training {
        tonic::include_proto!("monolith.ps_training");
    }

    pub mod serving {
        pub mod agent_service {
            tonic::include_proto!("monolith.serving.agent_service");
        }
    }

    pub mod native_training {
        tonic::include_proto!("monolith.native_training");

        pub mod data {
            tonic::include_proto!("monolith.native_training.data");

            pub mod config {
                tonic::include_proto!("monolith.native_training.data.config");
            }
        }

        pub mod model_dump {
            tonic::include_proto!("monolith.native_training.model_dump");
        }
    }

    pub mod model_export {
        tonic::include_proto!("monolith.model_export");
    }

    pub mod hooks {
        tonic::include_proto!("monolith.hooks");
    }
}

pub mod parser {
    pub mod proto {
        tonic::include_proto!("parser.proto");
    }
}

pub mod tensorflow {
    pub mod monolith_tf {
        tonic::include_proto!("tensorflow.monolith_tf");
    }
}

// Primus orchestration protos (not required for current Rust parity work).
// These import google/protobuf wrappers and would require additional module
// wiring, so keep them disabled for now.

// Convenience re-exports for commonly used message types.
pub use idl::matrix::proto::LineId;
pub use monolith::io::proto::{Example, ExampleBatch, ExampleBatchRowMajor, Feature, NamedFeature};

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;

    #[test]
    fn test_example_roundtrip_prost() {
        // Minimal Example: empty features/labels; line_id omitted.
        let ex = Example::default();
        let bytes = ex.encode_to_vec();
        let decoded = Example::decode(bytes.as_slice()).unwrap();
        assert_eq!(ex, decoded);
    }

    #[test]
    fn test_agent_service_types_exist() {
        // Just ensure generated types are present and linked.
        use monolith::serving::agent_service::{GetReplicasRequest, ServerType};
        let req = GetReplicasRequest {
            server_type: ServerType::Ps as i32,
            task: 0,
            model_name: "m".to_string(),
        };
        assert_eq!(req.task, 0);
    }
}
