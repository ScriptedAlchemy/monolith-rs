//! Feature ID (Fid) utilities and types.
//!
//! This module provides types and functions for working with feature IDs in Monolith.
//! A feature ID (Fid) encodes both the slot ID and the feature hash value into a single
//! 64-bit integer for efficient storage and lookup.
//!
//! # Fid Encoding
//!
//! The Fid is encoded as follows:
//! - Upper bits (configurable): Slot ID
//! - Lower bits: Feature hash value
//!
//! This encoding allows for efficient slot extraction without additional lookups.

use crate::error::{MonolithError, Result};

/// A feature ID that encodes both slot and feature information.
///
/// The Fid is a 64-bit signed integer that combines the slot ID in the upper bits
/// and the feature hash in the lower bits.
pub type Fid = i64;

/// A slot identifier for feature slots.
///
/// Slot IDs are 32-bit signed integers that identify different feature categories
/// or groups in the model.
pub type SlotId = i32;

/// Number of bits used for the slot ID in the Fid encoding.
pub const SLOT_BITS: u32 = 16;

/// Number of bits used for the feature hash in the Fid encoding.
pub const FEATURE_BITS: u32 = 64 - SLOT_BITS;

/// Mask for extracting the feature hash from a Fid.
pub const FEATURE_MASK: i64 = (1i64 << FEATURE_BITS) - 1;

/// Mask for extracting the slot ID from a Fid.
pub const SLOT_MASK: i64 = !FEATURE_MASK;

/// Maximum valid slot ID.
pub const MAX_SLOT_ID: SlotId = (1 << SLOT_BITS) - 1;

/// Maximum valid feature hash value.
pub const MAX_FEATURE_VALUE: i64 = (1i64 << FEATURE_BITS) - 1;

/// Extracts the slot ID from a feature ID.
///
/// The slot ID is stored in the upper bits of the Fid. This function
/// performs a right shift to extract those bits.
///
/// # Arguments
///
/// * `fid` - The feature ID to extract the slot from.
///
/// # Returns
///
/// The slot ID encoded in the upper bits of the Fid.
///
/// # Examples
///
/// ```
/// use monolith_core::fid::{extract_slot, make_fid};
///
/// let fid = make_fid(42, 12345).unwrap();
/// let slot = extract_slot(fid);
/// assert_eq!(slot, 42);
/// ```
#[inline]
pub fn extract_slot(fid: Fid) -> SlotId {
    ((fid >> FEATURE_BITS) & (MAX_SLOT_ID as i64)) as SlotId
}

/// Extracts the feature hash value from a feature ID.
///
/// The feature hash is stored in the lower bits of the Fid. This function
/// applies a mask to extract those bits.
///
/// # Arguments
///
/// * `fid` - The feature ID to extract the feature hash from.
///
/// # Returns
///
/// The feature hash value encoded in the lower bits of the Fid.
///
/// # Examples
///
/// ```
/// use monolith_core::fid::{extract_feature, make_fid};
///
/// let fid = make_fid(42, 12345).unwrap();
/// let feature = extract_feature(fid);
/// assert_eq!(feature, 12345);
/// ```
#[inline]
pub fn extract_feature(fid: Fid) -> i64 {
    fid & FEATURE_MASK
}

/// Creates a feature ID from a slot ID and feature hash.
///
/// This function combines a slot ID and feature hash into a single Fid
/// by placing the slot ID in the upper bits and the feature hash in the lower bits.
///
/// # Arguments
///
/// * `slot` - The slot ID (must be non-negative and fit in SLOT_BITS).
/// * `feature` - The feature hash value (must fit in FEATURE_BITS).
///
/// # Returns
///
/// A `Result` containing the combined Fid, or an error if the inputs are invalid.
///
/// # Errors
///
/// Returns `MonolithError::InvalidSlotId` if the slot ID is negative or too large.
/// Returns `MonolithError::InvalidFid` if the feature hash is negative or too large.
///
/// # Examples
///
/// ```
/// use monolith_core::fid::make_fid;
///
/// let fid = make_fid(10, 999).unwrap();
/// assert!(fid > 0);
/// ```
#[inline]
pub fn make_fid(slot: SlotId, feature: i64) -> Result<Fid> {
    if !(0..=MAX_SLOT_ID).contains(&slot) {
        return Err(MonolithError::InvalidSlotId { slot_id: slot });
    }

    if !(0..=MAX_FEATURE_VALUE).contains(&feature) {
        return Err(MonolithError::InvalidFid { fid: feature });
    }

    Ok(((slot as i64) << FEATURE_BITS) | feature)
}

/// Creates a feature ID from a slot ID and feature hash without validation.
///
/// This is an unchecked version of `make_fid` that skips validation for performance.
/// Use this only when you are certain the inputs are valid.
///
/// # Safety
///
/// The caller must ensure:
/// - `slot` is non-negative and less than or equal to `MAX_SLOT_ID`
/// - `feature` is non-negative and less than or equal to `MAX_FEATURE_VALUE`
///
/// # Arguments
///
/// * `slot` - The slot ID.
/// * `feature` - The feature hash value.
///
/// # Returns
///
/// The combined Fid.
#[inline]
pub fn make_fid_unchecked(slot: SlotId, feature: i64) -> Fid {
    ((slot as i64) << FEATURE_BITS) | (feature & FEATURE_MASK)
}

/// Validates a feature ID.
///
/// Checks that the Fid has a valid slot ID and feature hash.
///
/// # Arguments
///
/// * `fid` - The feature ID to validate.
///
/// # Returns
///
/// `true` if the Fid is valid, `false` otherwise.
#[inline]
pub fn is_valid_fid(fid: Fid) -> bool {
    let slot = extract_slot(fid);
    (0..=MAX_SLOT_ID).contains(&slot)
}

/// Validates a slot ID.
///
/// Checks that the slot ID is within the valid range.
///
/// # Arguments
///
/// * `slot` - The slot ID to validate.
///
/// # Returns
///
/// `true` if the slot ID is valid, `false` otherwise.
#[inline]
pub fn is_valid_slot(slot: SlotId) -> bool {
    (0..=MAX_SLOT_ID).contains(&slot)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_make_fid_and_extract() {
        let slot = 42;
        let feature = 12345;

        let fid = make_fid(slot, feature).unwrap();
        assert_eq!(extract_slot(fid), slot);
        assert_eq!(extract_feature(fid), feature);
    }

    #[test]
    fn test_make_fid_zero_values() {
        let fid = make_fid(0, 0).unwrap();
        assert_eq!(extract_slot(fid), 0);
        assert_eq!(extract_feature(fid), 0);
    }

    #[test]
    fn test_make_fid_max_values() {
        let fid = make_fid(MAX_SLOT_ID, MAX_FEATURE_VALUE).unwrap();
        assert_eq!(extract_slot(fid), MAX_SLOT_ID);
        assert_eq!(extract_feature(fid), MAX_FEATURE_VALUE);
    }

    #[test]
    fn test_make_fid_invalid_slot() {
        let result = make_fid(-1, 0);
        assert!(matches!(result, Err(MonolithError::InvalidSlotId { .. })));

        let result = make_fid(MAX_SLOT_ID + 1, 0);
        assert!(matches!(result, Err(MonolithError::InvalidSlotId { .. })));
    }

    #[test]
    fn test_make_fid_invalid_feature() {
        let result = make_fid(0, -1);
        assert!(matches!(result, Err(MonolithError::InvalidFid { .. })));
    }

    #[test]
    fn test_make_fid_unchecked() {
        let slot = 100;
        let feature = 9999;

        let fid = make_fid_unchecked(slot, feature);
        assert_eq!(extract_slot(fid), slot);
        assert_eq!(extract_feature(fid), feature);
    }

    #[test]
    fn test_is_valid_fid() {
        let valid_fid = make_fid(10, 500).unwrap();
        assert!(is_valid_fid(valid_fid));
    }

    #[test]
    fn test_is_valid_slot() {
        assert!(is_valid_slot(0));
        assert!(is_valid_slot(MAX_SLOT_ID));
        assert!(!is_valid_slot(-1));
        assert!(!is_valid_slot(MAX_SLOT_ID + 1));
    }

    #[test]
    fn test_roundtrip_multiple_values() {
        let test_cases = vec![
            (0, 0),
            (1, 1),
            (100, 50000),
            (MAX_SLOT_ID, 0),
            (0, MAX_FEATURE_VALUE),
            (1000, 1000000),
        ];

        for (slot, feature) in test_cases {
            let fid = make_fid(slot, feature).unwrap();
            assert_eq!(
                extract_slot(fid),
                slot,
                "Slot mismatch for ({}, {})",
                slot,
                feature
            );
            assert_eq!(
                extract_feature(fid),
                feature,
                "Feature mismatch for ({}, {})",
                slot,
                feature
            );
        }
    }
}
