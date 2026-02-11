//! Protobuf text-format (pbtxt) helpers.
//!
//! Python Monolith uses `google.protobuf.text_format` to read/write a handful of
//! small state protos (e.g. `monolith_checkpoint`, export state files, TF Serving
//! configs). In Rust we rely on `prost-reflect` to parse pbtxt into a
//! `DynamicMessage`, then decode into the concrete `prost`-generated message.

use crate::descriptor_pool::descriptor_pool;
use prost::Message;
use prost_reflect::prost::Message as ReflectMessage;
use prost_reflect::DynamicMessage;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PbtxtError {
    #[error("DescriptorPool missing message {0}")]
    MissingMessage(String),
    #[error("Failed to parse {full_name} pbtxt: {source}")]
    ParseTextFormat {
        full_name: String,
        source: prost_reflect::text_format::ParseError,
    },
    #[error("Failed to decode {full_name} after pbtxt parse: {source}")]
    DecodeAfterParse {
        full_name: String,
        source: prost::DecodeError,
    },
    #[error("Failed to decode {full_name} into DynamicMessage: {source}")]
    DecodeDynamic {
        full_name: String,
        source: prost::DecodeError,
    },
}

/// Parse a pbtxt string into a concrete `prost` message.
///
/// `full_name` must be the fully qualified protobuf message name,
/// e.g. `monolith.native_training.MonolithCheckpointState`.
pub fn parse_pbtxt<T: Message + Default>(full_name: &str, pbtxt: &str) -> Result<T, PbtxtError> {
    let pool = descriptor_pool();
    let msg_desc = pool
        .get_message_by_name(full_name)
        .ok_or_else(|| PbtxtError::MissingMessage(full_name.to_string()))?;
    let dyn_msg = DynamicMessage::parse_text_format(msg_desc, pbtxt).map_err(|e| {
        PbtxtError::ParseTextFormat {
            full_name: full_name.to_string(),
            source: e,
        }
    })?;
    let bytes = ReflectMessage::encode_to_vec(&dyn_msg);
    T::decode(bytes.as_slice()).map_err(|e| PbtxtError::DecodeAfterParse {
        full_name: full_name.to_string(),
        source: e,
    })
}

/// Convert a concrete `prost` message to pbtxt.
pub fn to_pbtxt<T: Message>(full_name: &str, msg: &T) -> Result<String, PbtxtError> {
    let pool = descriptor_pool();
    let msg_desc = pool
        .get_message_by_name(full_name)
        .ok_or_else(|| PbtxtError::MissingMessage(full_name.to_string()))?;

    let dyn_msg =
        DynamicMessage::decode(msg_desc, msg.encode_to_vec().as_slice()).map_err(|e| {
            PbtxtError::DecodeDynamic {
                full_name: full_name.to_string(),
                source: e,
            }
        })?;

    Ok(dyn_msg.to_text_format())
}
