#![cfg(feature = "grpc")]

use monolith_proto::idl::matrix::proto as matrix_proto;
use monolith_proto::monolith::io::proto as monolith_io;
use monolith_proto::tensorflow_core::DataType;
use monolith_serving::tfs_client::{
    get_example_batch_to_instance, get_instance_proto, FramedFileFlags,
};
use monolith_serving::ServingResult;
use prost::Message;
use std::path::Path;
use tempfile::tempdir;

// Mirrors `monolith/agent_service/tfs_client_test.py`.

fn repo_root() -> std::path::PathBuf {
    // monolith-serving crate lives at `monolith-rs/crates/monolith-serving`.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .expect("repo root")
}

#[test]
fn test_get_instance_proto_default() -> ServingResult<()> {
    let tp = get_instance_proto(None, 256)?;
    assert_eq!(tp.dtype, DataType::DtString as i32);
    assert_eq!(tp.tensor_shape.as_ref().unwrap().dim[0].size, 256);
    Ok(())
}

#[test]
fn test_get_example_batch_to_instance_from_pb() -> ServingResult<()> {
    // The Python test uses a framed binary file. Some checkouts may not include it,
    // so generate a minimal framed file if missing.
    let file_name =
        repo_root().join("monolith/native_training/data/training_instance/examplebatch.data");
    let flags = FramedFileFlags {
        lagrangex_header: true,
        ..Default::default()
    };

    let path = if file_name.exists() {
        file_name
    } else {
        let dir = tempdir().unwrap();
        let tmp = dir.path().join("examplebatch.data");

        // Build a single-record framed file: [8 dummy bytes][u64 size][payload].
        let eb = monolith_io::ExampleBatch {
            named_feature_list: vec![
                monolith_io::NamedFeatureList {
                    id: 0,
                    name: "user_id".to_string(),
                    feature: vec![monolith_io::Feature {
                        r#type: Some(monolith_io::feature::Type::FidV2List(
                            monolith_io::FidList { value: vec![123] },
                        )),
                    }],
                    r#type: monolith_io::FeatureListType::Individual as i32,
                },
                monolith_io::NamedFeatureList {
                    id: 0,
                    name: "__LINE_ID__".to_string(),
                    feature: vec![monolith_io::Feature {
                        r#type: Some(monolith_io::feature::Type::BytesList(
                            monolith_io::BytesList {
                                value: vec![matrix_proto::LineId::default().encode_to_vec()],
                            },
                        )),
                    }],
                    r#type: monolith_io::FeatureListType::Individual as i32,
                },
            ],
            named_raw_feature_list: vec![],
            batch_size: 1,
            data_source_key: 0,
        };
        let payload = eb.encode_to_vec();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&payload);
        std::fs::write(&tmp, bytes).unwrap();

        // Keep tempdir alive by leaking it; this is test-only and tiny.
        std::mem::forget(dir);
        tmp
    };

    let _ = get_example_batch_to_instance(&path, "pb", flags)?;
    Ok(())
}

#[test]
fn test_get_example_batch_to_instance_from_pbtxt() -> ServingResult<()> {
    let file_name = repo_root().join("monolith/agent_service/example_batch.pbtxt");
    let flags = FramedFileFlags {
        lagrangex_header: false,
        ..Default::default()
    };
    let _ = get_example_batch_to_instance(&file_name, "pbtxt", flags)?;
    Ok(())
}
