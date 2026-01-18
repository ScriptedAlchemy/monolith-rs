//! Small helpers for working with the generated Monolith protos.
//!
//! The upstream protobuf schemas support many feature encodings (fid_v1_list,
//! fid_v2_list, float_list, etc.). These helpers provide a common, ergonomic
//! way to extract the common cases used throughout the Rust port.

use crate::monolith::io::proto::feature;
use crate::monolith::io::proto::Feature;

/// Returns the first fid from a `Feature` if it is encoded as fid_v1_list or fid_v2_list.
pub fn first_fid(feature_msg: &Feature) -> Option<i64> {
    match &feature_msg.r#type {
        Some(feature::Type::FidV2List(l)) => l.value.first().copied().map(|v| v as i64),
        Some(feature::Type::FidV1List(l)) => l.value.first().copied().map(|v| v as i64),
        _ => None,
    }
}

/// Returns all fids from a `Feature` if it is encoded as fid_v1_list or fid_v2_list.
pub fn fids(feature_msg: &Feature) -> Option<Vec<i64>> {
    match &feature_msg.r#type {
        Some(feature::Type::FidV2List(l)) => Some(l.value.iter().map(|&v| v as i64).collect()),
        Some(feature::Type::FidV1List(l)) => Some(l.value.iter().map(|&v| v as i64).collect()),
        _ => None,
    }
}

/// Returns the first float from a `Feature` if it is encoded as float_list.
pub fn first_float(feature_msg: &Feature) -> Option<f32> {
    match &feature_msg.r#type {
        Some(feature::Type::FloatList(l)) => l.value.first().copied(),
        Some(feature::Type::FloatValue(v)) => Some(*v),
        _ => None,
    }
}
