//! Transform traits and implementations for data pipelines.
//!
//! This module provides the [`Transform`] trait and common transform implementations
//! for filtering, mapping, and otherwise transforming examples in a data pipeline.
//!
//! # Example
//!
//! ```
//! use monolith_data::transform::{Transform, FilterTransform, MapTransform};
//! use monolith_data::{has_feature, add_feature, create_example};
//!
//! // Filter examples that have a specific feature
//! let filter = FilterTransform::new(|ex| has_feature(ex, "user_id"));
//!
//! // Map examples to add a new feature
//! let mapper = MapTransform::new(|mut ex| {
//!     add_feature(&mut ex, "processed", vec![1], vec![1.0]);
//!     ex
//! });
//!
//! // Test: example without user_id is filtered out
//! let example = create_example();
//! assert!(filter.apply(example).is_none());
//! ```

use monolith_proto::Example;

/// A transform that can be applied to examples.
///
/// Transforms are the building blocks of data pipelines. They can filter,
/// modify, or otherwise process examples as they flow through the pipeline.
pub trait Transform: Send + Sync {
    /// Applies the transform to an example.
    ///
    /// # Arguments
    ///
    /// * `example` - The example to transform
    ///
    /// # Returns
    ///
    /// An `Option` containing the transformed example, or `None` if the
    /// example should be filtered out.
    fn apply(&self, example: Example) -> Option<Example>;

    /// Returns the name of this transform for debugging purposes.
    fn name(&self) -> &str {
        "Transform"
    }
}

/// A transform that filters examples based on a predicate.
///
/// Examples for which the predicate returns `false` are removed from the pipeline.
///
/// # Type Parameters
///
/// * `F` - The predicate function type
pub struct FilterTransform<F>
where
    F: Fn(&Example) -> bool + Send + Sync,
{
    predicate: F,
    name: String,
}

impl<F> FilterTransform<F>
where
    F: Fn(&Example) -> bool + Send + Sync,
{
    /// Creates a new filter transform with the given predicate.
    ///
    /// # Arguments
    ///
    /// * `predicate` - A function that returns `true` for examples to keep
    pub fn new(predicate: F) -> Self {
        Self {
            predicate,
            name: "FilterTransform".to_string(),
        }
    }

    /// Sets a custom name for this transform.
    ///
    /// # Arguments
    ///
    /// * `name` - The name to use for this transform
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
}

impl<F> Transform for FilterTransform<F>
where
    F: Fn(&Example) -> bool + Send + Sync,
{
    fn apply(&self, example: Example) -> Option<Example> {
        if (self.predicate)(&example) {
            Some(example)
        } else {
            None
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// A transform that maps examples using a function.
///
/// The mapping function can modify examples in any way, including changing
/// features, line IDs, etc.
///
/// # Type Parameters
///
/// * `F` - The mapping function type
pub struct MapTransform<F>
where
    F: Fn(Example) -> Example + Send + Sync,
{
    mapper: F,
    name: String,
}

impl<F> MapTransform<F>
where
    F: Fn(Example) -> Example + Send + Sync,
{
    /// Creates a new map transform with the given mapping function.
    ///
    /// # Arguments
    ///
    /// * `mapper` - A function that transforms examples
    pub fn new(mapper: F) -> Self {
        Self {
            mapper,
            name: "MapTransform".to_string(),
        }
    }

    /// Sets a custom name for this transform.
    ///
    /// # Arguments
    ///
    /// * `name` - The name to use for this transform
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
}

impl<F> Transform for MapTransform<F>
where
    F: Fn(Example) -> Example + Send + Sync,
{
    fn apply(&self, example: Example) -> Option<Example> {
        Some((self.mapper)(example))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// A transform that optionally maps examples, allowing filtering.
///
/// This combines the functionality of [`FilterTransform`] and [`MapTransform`],
/// allowing the function to both transform and optionally filter examples.
///
/// # Type Parameters
///
/// * `F` - The filter-map function type
pub struct FilterMapTransform<F>
where
    F: Fn(Example) -> Option<Example> + Send + Sync,
{
    filter_mapper: F,
    name: String,
}

impl<F> FilterMapTransform<F>
where
    F: Fn(Example) -> Option<Example> + Send + Sync,
{
    /// Creates a new filter-map transform.
    ///
    /// # Arguments
    ///
    /// * `filter_mapper` - A function that optionally transforms examples
    pub fn new(filter_mapper: F) -> Self {
        Self {
            filter_mapper,
            name: "FilterMapTransform".to_string(),
        }
    }

    /// Sets a custom name for this transform.
    ///
    /// # Arguments
    ///
    /// * `name` - The name to use for this transform
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
}

impl<F> Transform for FilterMapTransform<F>
where
    F: Fn(Example) -> Option<Example> + Send + Sync,
{
    fn apply(&self, example: Example) -> Option<Example> {
        (self.filter_mapper)(example)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// A chain of transforms applied in sequence.
///
/// This allows composing multiple transforms into a single transform
/// that applies them in order.
pub struct TransformChain {
    transforms: Vec<Box<dyn Transform>>,
    name: String,
}

impl TransformChain {
    /// Creates a new empty transform chain.
    pub fn new() -> Self {
        Self {
            transforms: Vec::new(),
            name: "TransformChain".to_string(),
        }
    }

    /// Adds a transform to the chain.
    ///
    /// # Arguments
    ///
    /// * `transform` - The transform to add
    #[allow(clippy::should_implement_trait)]
    pub fn add<T: Transform + 'static>(mut self, transform: T) -> Self {
        self.transforms.push(Box::new(transform));
        self
    }

    /// Sets a custom name for this transform chain.
    ///
    /// # Arguments
    ///
    /// * `name` - The name to use for this transform chain
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Returns the number of transforms in the chain.
    pub fn len(&self) -> usize {
        self.transforms.len()
    }

    /// Returns `true` if the chain contains no transforms.
    pub fn is_empty(&self) -> bool {
        self.transforms.is_empty()
    }
}

impl Default for TransformChain {
    fn default() -> Self {
        Self::new()
    }
}

impl Transform for TransformChain {
    fn apply(&self, mut example: Example) -> Option<Example> {
        for transform in &self.transforms {
            example = transform.apply(example)?;
        }
        Some(example)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// A dataset wrapper that applies transforms to examples.
///
/// # Type Parameters
///
/// * `I` - The underlying iterator type
/// * `T` - The transform type
pub struct TransformedDataset<I, T> {
    inner: I,
    transform: T,
}

impl<I, T> TransformedDataset<I, T>
where
    I: Iterator<Item = Example>,
    T: Transform,
{
    /// Creates a new transformed dataset.
    ///
    /// # Arguments
    ///
    /// * `inner` - The underlying iterator of examples
    /// * `transform` - The transform to apply
    pub fn new(inner: I, transform: T) -> Self {
        Self { inner, transform }
    }

    /// Returns an iterator over transformed examples.
    pub fn iter(self) -> TransformedIterator<I, T> {
        TransformedIterator {
            inner: self.inner,
            transform: self.transform,
        }
    }
}

/// Iterator that applies a transform to examples.
pub struct TransformedIterator<I, T> {
    inner: I,
    transform: T,
}

impl<I, T> Iterator for TransformedIterator<I, T>
where
    I: Iterator<Item = Example>,
    T: Transform,
{
    type Item = Example;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let example = self.inner.next()?;
            if let Some(transformed) = self.transform.apply(example) {
                return Some(transformed);
            }
            // If transform filtered out the example, continue to next
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::example::{add_feature, create_example, get_feature, has_feature};
    use monolith_proto::monolith::io::proto::feature;

    fn make_example_with_value(val: i64) -> Example {
        let mut ex = create_example();
        add_feature(&mut ex, "value", vec![val], vec![val as f32]);
        ex
    }

    #[test]
    fn test_filter_transform_keep() {
        let filter = FilterTransform::new(|_| true);
        let example = create_example();
        assert!(filter.apply(example).is_some());
    }

    #[test]
    fn test_filter_transform_remove() {
        let filter = FilterTransform::new(|_| false);
        let example = create_example();
        assert!(filter.apply(example).is_none());
    }

    #[test]
    fn test_filter_transform_with_predicate() {
        let filter = FilterTransform::new(|ex| has_feature(ex, "important"));

        let mut ex_with = create_example();
        add_feature(&mut ex_with, "important", vec![1], vec![1.0]);

        let ex_without = create_example();

        assert!(filter.apply(ex_with).is_some());
        assert!(filter.apply(ex_without).is_none());
    }

    #[test]
    fn test_filter_transform_name() {
        let filter = FilterTransform::new(|_| true).with_name("MyFilter");
        assert_eq!(filter.name(), "MyFilter");
    }

    #[test]
    fn test_map_transform() {
        let mapper = MapTransform::new(|mut ex| {
            add_feature(&mut ex, "added", vec![42], vec![42.0]);
            ex
        });

        let example = create_example();
        let result = mapper.apply(example).unwrap();
        assert!(has_feature(&result, "added"));
    }

    #[test]
    fn test_map_transform_name() {
        let mapper = MapTransform::new(|ex| ex).with_name("MyMapper");
        assert_eq!(mapper.name(), "MyMapper");
    }

    #[test]
    fn test_filter_map_transform() {
        let filter_map = FilterMapTransform::new(|ex| {
            if has_feature(&ex, "keep") {
                let mut result = ex;
                add_feature(&mut result, "processed", vec![1], vec![1.0]);
                Some(result)
            } else {
                None
            }
        });

        let mut ex_keep = create_example();
        add_feature(&mut ex_keep, "keep", vec![1], vec![1.0]);

        let ex_drop = create_example();

        let result = filter_map.apply(ex_keep).unwrap();
        assert!(has_feature(&result, "processed"));
        assert!(filter_map.apply(ex_drop).is_none());
    }

    #[test]
    fn test_transform_chain() {
        let chain = TransformChain::new()
            .add(FilterTransform::new(|ex| {
                get_feature(ex, "value")
                    .and_then(|f| match &f.r#type {
                        Some(feature::Type::FidV2List(l)) => {
                            l.value.first().copied().map(|v| v as i64)
                        }
                        Some(feature::Type::FidV1List(l)) => {
                            l.value.first().copied().map(|v| v as i64)
                        }
                        _ => None,
                    })
                    .map(|v| v > 5)
                    .unwrap_or(false)
            }))
            .add(MapTransform::new(|mut ex| {
                add_feature(&mut ex, "passed_filter", vec![1], vec![1.0]);
                ex
            }));

        assert_eq!(chain.len(), 2);

        let ex_pass = make_example_with_value(10);
        let ex_fail = make_example_with_value(3);

        let result = chain.apply(ex_pass).unwrap();
        assert!(has_feature(&result, "passed_filter"));
        assert!(chain.apply(ex_fail).is_none());
    }

    #[test]
    fn test_transform_chain_empty() {
        let chain = TransformChain::new();
        assert!(chain.is_empty());

        let example = create_example();
        assert!(chain.apply(example).is_some());
    }

    #[test]
    fn test_transformed_dataset() {
        let examples: Vec<Example> = (0..10).map(make_example_with_value).collect();

        let filter = FilterTransform::new(|ex| {
            get_feature(ex, "value")
                .and_then(|f| match &f.r#type {
                    Some(feature::Type::FidV2List(l)) => l.value.first().copied().map(|v| v as i64),
                    Some(feature::Type::FidV1List(l)) => l.value.first().copied().map(|v| v as i64),
                    _ => None,
                })
                .map(|v| v % 2 == 0)
                .unwrap_or(false)
        });

        let dataset = TransformedDataset::new(examples.into_iter(), filter);
        let results: Vec<_> = dataset.iter().collect();

        // Should only have even values: 0, 2, 4, 6, 8
        assert_eq!(results.len(), 5);
    }
}
