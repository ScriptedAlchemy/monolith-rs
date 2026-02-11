//! TFServing client-side data shaping helpers (Python parity).
//!
//! This module ports the small subset of `monolith/agent_service/tfs_client.py`
//! that is exercised by unit tests:
//! - building `TensorProto(DT_STRING)` batches for `Instance` payloads
//! - converting `ExampleBatch` (column-major) into a batch of `Instance` protos
//! - reading the Python "examplebatch.data" framed file format

use crate::error::{ServingError, ServingResult};
use monolith_proto::descriptor_pool::descriptor_pool;
use monolith_proto::idl::matrix::proto as matrix_proto;
use monolith_proto::monolith::io::proto as monolith_io;
use monolith_proto::parser::proto as parser_proto;
use monolith_proto::tensorflow_core as tf_core;
use prost::Message;
use prost_reflect::prost::Message as ReflectMessage;
use prost_reflect::{DynamicMessage, MessageDescriptor};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// Flags controlling the framing used by some training dumps.
#[derive(Debug, Clone, Copy, Default)]
pub struct FramedFileFlags {
    /// Python parity: `FLAGS.lagrangex_header`.
    pub lagrangex_header: bool,
    /// Python parity: `FLAGS.has_sort_id`.
    pub has_sort_id: bool,
    /// Python parity: `FLAGS.kafka_dump`.
    pub kafka_dump: bool,
    /// Python parity: `FLAGS.kafka_dump_prefix`.
    pub kafka_dump_prefix: bool,
}

fn read_u64_le<R: Read>(r: &mut R) -> std::io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_header<R: Read>(r: &mut R, flags: FramedFileFlags) -> std::io::Result<()> {
    const INT_SIZE: usize = 8;
    if flags.lagrangex_header {
        // Python parity: skip a single int64 header.
        let mut buf = [0u8; INT_SIZE];
        r.read_exact(&mut buf)?;
        return Ok(());
    }

    let mut aggregate_page_sortid_size: u64 = 0;
    if flags.kafka_dump_prefix {
        let mut size = read_u64_le(r)?;
        if size == 0 {
            size = read_u64_le(r)?;
        } else {
            aggregate_page_sortid_size = size;
        }
    }

    if flags.has_sort_id {
        let size = if aggregate_page_sortid_size == 0 {
            read_u64_le(r)?
        } else {
            aggregate_page_sortid_size
        };
        // Skip sort_id bytes.
        let mut discard = vec![0u8; size as usize];
        r.read_exact(&mut discard)?;
    }

    if flags.kafka_dump {
        let mut buf = [0u8; INT_SIZE];
        r.read_exact(&mut buf)?;
    }

    Ok(())
}

fn read_framed_message<R: Read>(r: &mut R, flags: FramedFileFlags) -> std::io::Result<Vec<u8>> {
    read_header(r, flags)?;
    let size = read_u64_le(r)? as usize;
    let mut buf = vec![0u8; size];
    r.read_exact(&mut buf)?;
    Ok(buf)
}

fn make_string_tensor(values: Vec<Vec<u8>>) -> tf_core::TensorProto {
    let shape = tf_core::TensorShapeProto {
        dim: vec![tf_core::tensor_shape_proto::Dim {
            size: values.len() as i64,
            name: "".to_string(),
        }],
        ..Default::default()
    };

    tf_core::TensorProto {
        dtype: tf_core::DataType::DtString as i32,
        tensor_shape: Some(shape),
        string_val: values,
        ..Default::default()
    }
}

/// Python parity for `get_instance_proto()`.
pub fn get_instance_proto(
    input_file: Option<&Path>,
    batch_size: usize,
) -> ServingResult<tf_core::TensorProto> {
    let mut instances: Vec<Vec<u8>> = Vec::with_capacity(batch_size);
    if let Some(path) = input_file {
        let mut f = File::open(path)?;
        for _ in 0..batch_size {
            let bytes = read_framed_message(&mut f, FramedFileFlags::default())
                .map_err(|e| ServingError::IoError(e))?;
            let inst = parser_proto::Instance::decode(bytes.as_slice()).map_err(|e| {
                ServingError::InvalidRequest(format!("Failed to decode Instance: {e}"))
            })?;
            instances.push(inst.encode_to_vec());
        }
    } else {
        for i in 0..batch_size {
            // Deterministic, "random-enough" instance payload for parity tests.
            let inst = parser_proto::Instance {
                fid: vec![(1u64 << 54) | (i as u64 + 1)],
                ..Default::default()
            };
            instances.push(inst.encode_to_vec());
        }
    }

    Ok(make_string_tensor(instances))
}

fn parse_pbtxt_with_pool<T: Message + Default>(full_name: &str, pbtxt: &str) -> ServingResult<T> {
    let pool = descriptor_pool();
    let msg_desc: MessageDescriptor = pool.get_message_by_name(full_name).ok_or_else(|| {
        ServingError::ConfigError(format!("DescriptorPool missing message {full_name}"))
    })?;
    let dyn_msg = DynamicMessage::parse_text_format(msg_desc, pbtxt).map_err(|e| {
        ServingError::ConfigError(format!("Failed to parse {full_name} pbtxt: {e}"))
    })?;
    let bytes = ReflectMessage::encode_to_vec(&dyn_msg);
    T::decode(bytes.as_slice()).map_err(|e| {
        ServingError::ConfigError(format!("Failed to decode {full_name} from pbtxt: {e}"))
    })
}

/// Python parity for `get_example_batch_to_instance(input_file, file_type)`.
pub fn get_example_batch_to_instance(
    input_file: &Path,
    file_type: &str,
    flags: FramedFileFlags,
) -> ServingResult<tf_core::TensorProto> {
    let eb: monolith_io::ExampleBatch = if file_type == "pb" {
        let mut f = File::open(input_file)?;
        let bytes = read_framed_message(&mut f, flags).map_err(|e| ServingError::IoError(e))?;
        monolith_io::ExampleBatch::decode(bytes.as_slice()).map_err(|e| {
            ServingError::InvalidRequest(format!("Failed to decode ExampleBatch: {e}"))
        })?
    } else {
        let pbtxt = std::fs::read_to_string(input_file)?;
        parse_pbtxt_with_pool::<monolith_io::ExampleBatch>(
            "monolith.io.proto.ExampleBatch",
            &pbtxt,
        )?
    };

    let batch_size = eb.batch_size.max(0) as usize;
    let mut inst_list: Vec<Vec<u8>> = Vec::with_capacity(batch_size);
    let mask: u64 = (1u64 << 48) - 1;

    for i in 0..batch_size {
        let mut inst = parser_proto::Instance::default();
        for nfl in &eb.named_feature_list {
            let use_shared = nfl.r#type == monolith_io::FeatureListType::Shared as i32;
            let feat = if use_shared {
                nfl.feature.first()
            } else {
                nfl.feature.get(i)
            };
            let Some(feat) = feat else { continue };

            let name = nfl.name.as_str();

            // Special pseudo-columns.
            if name == "__LABEL__" {
                match &feat.r#type {
                    Some(monolith_io::feature::Type::FloatList(fl)) => {
                        inst.label.extend_from_slice(&fl.value);
                    }
                    Some(monolith_io::feature::Type::DoubleList(dl)) => {
                        inst.label.extend(dl.value.iter().map(|v| *v as f32));
                    }
                    _ => {}
                }
                continue;
            }
            if name == "__LINE_ID__" {
                if let Some(monolith_io::feature::Type::BytesList(bl)) = &feat.r#type {
                    if let Some(first) = bl.value.first() {
                        let line_id =
                            matrix_proto::LineId::decode(first.as_slice()).map_err(|e| {
                                ServingError::InvalidRequest(format!(
                                    "Failed to decode LineId: {e}"
                                ))
                            })?;
                        inst.line_id = Some(line_id);
                    }
                }
                continue;
            }

            // Regular features become `parser.proto.Instance.feature[]` entries (idl.matrix.proto.Feature).
            let mut out_feat = matrix_proto::Feature {
                name: Some(name.to_string()),
                ..Default::default()
            };

            match &feat.r#type {
                Some(monolith_io::feature::Type::FidV1List(fl)) if !fl.value.is_empty() => {
                    let slot_id = fl.value[0] >> 54;
                    out_feat
                        .fid
                        .extend(fl.value.iter().map(|v| (slot_id << 48) | (mask & *v)));
                }
                Some(monolith_io::feature::Type::FidV2List(fl)) if !fl.value.is_empty() => {
                    out_feat.fid.extend_from_slice(&fl.value);
                }
                Some(monolith_io::feature::Type::FloatList(fl)) if !fl.value.is_empty() => {
                    out_feat.float_value.extend_from_slice(&fl.value);
                }
                Some(monolith_io::feature::Type::DoubleList(dl)) if !dl.value.is_empty() => {
                    out_feat
                        .float_value
                        .extend(dl.value.iter().map(|v| *v as f32));
                }
                Some(monolith_io::feature::Type::Int64List(il)) if !il.value.is_empty() => {
                    out_feat.int64_value.extend_from_slice(&il.value);
                }
                Some(monolith_io::feature::Type::BytesList(bl)) if !bl.value.is_empty() => {
                    out_feat.bytes_value.extend_from_slice(&bl.value);
                }
                _ => continue,
            }

            inst.feature.push(out_feat);
        }

        inst_list.push(inst.encode_to_vec());
    }

    Ok(make_string_tensor(inst_list))
}

/// Convenience for callers that need to repeatedly read framed messages from a file.
///
/// This intentionally matches the Python framing where each record is:
/// `[header][u64_le size][payload bytes...]`.
pub fn read_framed_example_batch_from_file(
    path: &Path,
    flags: FramedFileFlags,
) -> ServingResult<monolith_io::ExampleBatch> {
    let mut f = File::open(path)?;
    let bytes = read_framed_message(&mut f, flags).map_err(|e| ServingError::IoError(e))?;
    monolith_io::ExampleBatch::decode(bytes.as_slice())
        .map_err(|e| ServingError::InvalidRequest(format!("Failed to decode ExampleBatch: {e}")))
}

/// Utility for tests: if the file does not start at a framed message boundary, rewind.
pub fn rewind_file_to_start(f: &mut File) -> std::io::Result<()> {
    f.seek(SeekFrom::Start(0))?;
    Ok(())
}
