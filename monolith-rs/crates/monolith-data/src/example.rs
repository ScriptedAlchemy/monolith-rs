//! Example utilities for working with Monolith `monolith.io.proto.Example` protos.
//!
//! This module provides helper functions to create, manipulate, and query
//! [`Example`] protobuf messages used throughout the Monolith data pipeline.
//!
//! # Example
//!
//! ```
//! use monolith_data::example::{create_example, add_feature, get_feature};
//! use monolith_proto::monolith::io::proto::feature;
//!
//! let mut example = create_example();
//! add_feature(&mut example, "user_id", vec![12345], vec![1.0]);
//!
//! if let Some(feature) = get_feature(&example, "user_id") {
//!     match &feature.r#type {
//!         Some(feature::Type::FidV2List(l)) => assert_eq!(l.value, vec![12345]),
//!         other => panic!("unexpected feature type: {:?}", other),
//!     }
//! }
//! ```

use monolith_proto::idl::matrix::proto::LineId;
use monolith_proto::monolith::io::proto::{feature, FidList, FloatList};
use monolith_proto::{Example, Feature, NamedFeature};

/// Simple struct for accessing feature data with direct `fid` and `value` fields.
///
/// This provides a simpler interface than the raw proto `Feature` type.
#[derive(Debug, Clone, Default)]
pub struct FeatureData {
    /// Feature IDs (FIDs) for sparse features.
    pub fid: Vec<i64>,
    /// Float values associated with features.
    pub value: Vec<f32>,
}

impl FeatureData {
    /// Creates a new FeatureData from FIDs and values.
    pub fn new(fid: Vec<i64>, value: Vec<f32>) -> Self {
        Self { fid, value }
    }
}

/// Extracts [`FeatureData`] from a proto [`Feature`].
///
/// This helper function extracts FIDs and float values from the Feature's
/// oneof type into a simple struct with `fid` and `value` fields.
///
/// # Arguments
///
/// * `feature` - The Feature to extract data from
///
/// # Returns
///
/// A [`FeatureData`] struct with the extracted fids and values.
pub fn extract_feature_data(feature: &Feature) -> FeatureData {
    let mut fid = Vec::new();
    let mut value = Vec::new();

    if let Some(ref t) = feature.r#type {
        match t {
            feature::Type::FidV1List(l) | feature::Type::FidV2List(l) => {
                fid = l.value.iter().map(|&v| v as i64).collect();
            }
            feature::Type::FloatList(l) => {
                value = l.value.clone();
            }
            feature::Type::Int64List(l) => {
                fid = l.value.clone();
            }
            feature::Type::FidV1Lists(ls) | feature::Type::FidV2Lists(ls) => {
                for list in &ls.list {
                    fid.extend(list.value.iter().map(|&v| v as i64));
                }
            }
            feature::Type::FloatLists(ls) => {
                for list in &ls.list {
                    value.extend(&list.value);
                }
            }
            _ => {}
        }
    }

    FeatureData { fid, value }
}

/// Gets feature data by name from an [`Example`].
///
/// This is a convenience function that combines [`get_feature`] and
/// [`extract_feature_data`] into a single call.
///
/// # Arguments
///
/// * `example` - The example to search
/// * `name` - The name of the feature to find
///
/// # Returns
///
/// An `Option` containing a [`FeatureData`] if the feature is found.
pub fn get_feature_data(example: &Example, name: &str) -> Option<FeatureData> {
    get_feature(example, name).map(extract_feature_data)
}

/// Creates a new empty [`Example`].
///
/// # Returns
///
/// A new `Example` with no features and no line ID.
pub fn create_example() -> Example {
    Example {
        named_feature: Vec::new(),
        named_raw_feature: Vec::new(),
        line_id: None,
        label: Vec::new(),
        instance_weight: 1.0,
        data_source_key: 0,
    }
}

/// Creates a new [`Example`] with the specified `LineId`.
///
/// # Arguments
///
/// * `line_id` - The LineId proto to associate with this example
///
/// # Returns
///
/// A new `Example` with the given line ID.
pub fn create_example_with_line_id(line_id: LineId) -> Example {
    Example {
        named_feature: Vec::new(),
        line_id: Some(line_id),
        named_raw_feature: Vec::new(),
        label: Vec::new(),
        instance_weight: 1.0,
        data_source_key: 0,
    }
}

/// Adds a named feature to an [`Example`].
///
/// # Arguments
///
/// * `example` - The example to add the feature to
/// * `name` - The name of the feature
/// * `fids` - Sparse IDs (written as `fid_v2_list` when non-empty)
/// * `values` - Dense float values (written as `float_list` when `fids` is empty)
///
/// The upstream proto supports many feature encodings. For the Rust port we
/// keep this helper intentionally simple:
///
/// - If `fids` is non-empty, we store a sparse feature as `fid_v2_list` and
///   ignore `values`.
/// - Otherwise, if `values` is non-empty, we store a dense feature as
///   `float_list`.
pub fn add_feature(example: &mut Example, name: &str, fids: Vec<i64>, values: Vec<f32>) {
    let feature = if !fids.is_empty() {
        let fid_list = FidList {
            value: fids.into_iter().map(|v| v as u64).collect(),
        };
        Feature {
            r#type: Some(feature::Type::FidV2List(fid_list)),
        }
    } else {
        let float_list = FloatList { value: values };
        Feature {
            r#type: Some(feature::Type::FloatList(float_list)),
        }
    };
    let named_feature = NamedFeature {
        id: 0,
        name: name.to_string(),
        feature: Some(feature),
        sorted_id: 0,
    };
    example.named_feature.push(named_feature);
}

/// Gets a feature by name from an [`Example`].
///
/// # Arguments
///
/// * `example` - The example to search
/// * `name` - The name of the feature to find
///
/// # Returns
///
/// An `Option` containing a reference to the [`Feature`] if found.
pub fn get_feature<'a>(example: &'a Example, name: &str) -> Option<&'a Feature> {
    example
        .named_feature
        .iter()
        .find(|nf| nf.name == name)
        .and_then(|nf| nf.feature.as_ref())
}

/// Gets a mutable feature by name from an [`Example`].
///
/// # Arguments
///
/// * `example` - The example to search
/// * `name` - The name of the feature to find
///
/// # Returns
///
/// An `Option` containing a mutable reference to the [`Feature`] if found.
pub fn get_feature_mut<'a>(example: &'a mut Example, name: &str) -> Option<&'a mut Feature> {
    example
        .named_feature
        .iter_mut()
        .find(|nf| nf.name == name)
        .and_then(|nf| nf.feature.as_mut())
}

/// Returns the names of all features in an [`Example`].
///
/// # Arguments
///
/// * `example` - The example to get feature names from
///
/// # Returns
///
/// A vector of feature names.
pub fn feature_names(example: &Example) -> Vec<&str> {
    example
        .named_feature
        .iter()
        .map(|nf| nf.name.as_str())
        .collect()
}

/// Returns the number of features in an [`Example`].
///
/// # Arguments
///
/// * `example` - The example to count features in
///
/// # Returns
///
/// The number of named features.
pub fn feature_count(example: &Example) -> usize {
    example.named_feature.len()
}

/// Checks if an [`Example`] contains a feature with the given name.
///
/// # Arguments
///
/// * `example` - The example to search
/// * `name` - The name of the feature to find
///
/// # Returns
///
/// `true` if the feature exists, `false` otherwise.
pub fn has_feature(example: &Example, name: &str) -> bool {
    example.named_feature.iter().any(|nf| nf.name == name)
}

/// Removes a feature by name from an [`Example`].
///
/// # Arguments
///
/// * `example` - The example to remove the feature from
/// * `name` - The name of the feature to remove
///
/// # Returns
///
/// `true` if a feature was removed, `false` if no feature with that name existed.
pub fn remove_feature(example: &mut Example, name: &str) -> bool {
    let initial_len = example.named_feature.len();
    example.named_feature.retain(|nf| nf.name != name);
    example.named_feature.len() < initial_len
}

/// Merges features from one [`Example`] into another.
///
/// Features with the same name in `other` will overwrite those in `target`.
///
/// # Arguments
///
/// * `target` - The example to merge features into
/// * `other` - The example to take features from
pub fn merge_examples(target: &mut Example, other: &Example) {
    for nf in &other.named_feature {
        // Remove existing feature with same name if present
        remove_feature(target, &nf.name);
        // Add the new feature
        target.named_feature.push(nf.clone());
    }
}

/// Creates a deep clone of an [`Example`].
///
/// # Arguments
///
/// * `example` - The example to clone
///
/// # Returns
///
/// A new `Example` that is a deep copy of the input.
pub fn clone_example(example: &Example) -> Example {
    example.clone()
}

/// Gets the total number of feature IDs across all features in an [`Example`].
///
/// # Arguments
///
/// * `example` - The example to count feature IDs in
///
/// # Returns
///
/// The total count of feature IDs.
pub fn total_fid_count(example: &Example) -> usize {
    example
        .named_feature
        .iter()
        .filter_map(|nf| nf.feature.as_ref())
        .map(|f| match &f.r#type {
            Some(feature::Type::FidV2List(l)) => l.value.len(),
            Some(feature::Type::FidV1List(l)) => l.value.len(),
            Some(feature::Type::FidV2Lists(ls)) => ls.list.iter().map(|l| l.value.len()).sum(),
            Some(feature::Type::FidV1Lists(ls)) => ls.list.iter().map(|l| l.value.len()).sum(),
            _ => 0,
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use monolith_proto::monolith::io::proto::feature;
    use monolith_proto::monolith::io::proto::FidList;
    use monolith_proto::monolith::io::proto::FloatList;

    #[test]
    fn test_create_example() {
        let example = create_example();
        assert!(example.named_feature.is_empty());
        assert!(example.line_id.is_none());
    }

    #[test]
    fn test_create_example_with_line_id() {
        let line_id = LineId::default();
        let example = create_example_with_line_id(line_id.clone());
        assert_eq!(example.line_id, Some(line_id));
    }

    #[test]
    fn test_add_and_get_feature() {
        let mut example = create_example();
        add_feature(&mut example, "user_id", vec![100, 200], vec![1.0, 2.0]);

        let feature = get_feature(&example, "user_id").unwrap();
        match &feature.r#type {
            Some(feature::Type::FidV2List(FidList { value })) => {
                assert_eq!(value, &vec![100u64, 200u64]);
            }
            other => panic!("Expected FidV2List, got {:?}", other),
        }
    }

    #[test]
    fn test_add_and_get_dense_feature_float_list() {
        let mut example = create_example();
        add_feature(&mut example, "embedding", vec![], vec![0.1, 0.2, 0.3]);

        let feature = get_feature(&example, "embedding").unwrap();
        match &feature.r#type {
            Some(feature::Type::FloatList(FloatList { value })) => {
                assert_eq!(value, &vec![0.1, 0.2, 0.3]);
            }
            other => panic!("Expected FloatList, got {:?}", other),
        }
    }

    #[test]
    fn test_get_feature_not_found() {
        let example = create_example();
        assert!(get_feature(&example, "nonexistent").is_none());
    }

    #[test]
    fn test_get_feature_mut() {
        let mut example = create_example();
        add_feature(&mut example, "score", vec![1], vec![0.5]);

        if let Some(feature) = get_feature_mut(&mut example, "score") {
            feature.r#type = Some(feature::Type::FidV2List(FidList { value: vec![9] }));
        }

        let feature = get_feature(&example, "score").unwrap();
        match &feature.r#type {
            Some(feature::Type::FidV2List(FidList { value })) => assert_eq!(value, &vec![9u64]),
            other => panic!("Expected FidV2List, got {:?}", other),
        }
    }

    #[test]
    fn test_feature_names() {
        let mut example = create_example();
        add_feature(&mut example, "a", vec![1], vec![1.0]);
        add_feature(&mut example, "b", vec![2], vec![2.0]);
        add_feature(&mut example, "c", vec![3], vec![3.0]);

        let names = feature_names(&example);
        assert_eq!(names, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_feature_count() {
        let mut example = create_example();
        assert_eq!(feature_count(&example), 0);

        add_feature(&mut example, "x", vec![1], vec![1.0]);
        assert_eq!(feature_count(&example), 1);

        add_feature(&mut example, "y", vec![2], vec![2.0]);
        assert_eq!(feature_count(&example), 2);
    }

    #[test]
    fn test_has_feature() {
        let mut example = create_example();
        add_feature(&mut example, "exists", vec![1], vec![1.0]);

        assert!(has_feature(&example, "exists"));
        assert!(!has_feature(&example, "missing"));
    }

    #[test]
    fn test_remove_feature() {
        let mut example = create_example();
        add_feature(&mut example, "to_remove", vec![1], vec![1.0]);
        add_feature(&mut example, "to_keep", vec![2], vec![2.0]);

        assert!(remove_feature(&mut example, "to_remove"));
        assert!(!has_feature(&example, "to_remove"));
        assert!(has_feature(&example, "to_keep"));

        // Removing non-existent feature returns false
        assert!(!remove_feature(&mut example, "nonexistent"));
    }

    #[test]
    fn test_merge_examples() {
        let mut target = create_example();
        add_feature(&mut target, "a", vec![1], vec![1.0]);
        add_feature(&mut target, "b", vec![2], vec![2.0]);

        let mut other = create_example();
        add_feature(&mut other, "b", vec![20], vec![20.0]); // Override
        add_feature(&mut other, "c", vec![3], vec![3.0]); // New

        merge_examples(&mut target, &other);

        assert_eq!(feature_count(&target), 3);
        let a = get_feature(&target, "a").unwrap();
        let b = get_feature(&target, "b").unwrap();
        let c = get_feature(&target, "c").unwrap();
        match &a.r#type {
            Some(feature::Type::FidV2List(l)) => assert_eq!(l.value, vec![1]),
            other => panic!("Expected FidV2List, got {:?}", other),
        }
        match &b.r#type {
            Some(feature::Type::FidV2List(l)) => assert_eq!(l.value, vec![20]),
            other => panic!("Expected FidV2List, got {:?}", other),
        }
        match &c.r#type {
            Some(feature::Type::FidV2List(l)) => assert_eq!(l.value, vec![3]),
            other => panic!("Expected FidV2List, got {:?}", other),
        }
    }

    #[test]
    fn test_total_fid_count() {
        let mut example = create_example();
        add_feature(&mut example, "a", vec![1, 2, 3], vec![1.0, 2.0, 3.0]);
        add_feature(&mut example, "b", vec![4, 5], vec![4.0, 5.0]);

        assert_eq!(total_fid_count(&example), 5);
    }
}
