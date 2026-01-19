//! Layer trait definition for neural network layers.
//!
//! This module defines the core [`Layer`] trait that all neural network layers
//! must implement, providing a unified interface for forward and backward passes.

use crate::error::LayerError;
use crate::tensor::Tensor;

/// A neural network layer that supports forward and backward propagation.
///
/// This trait defines the fundamental interface for all neural network layers
/// in Monolith. Each layer must be able to:
/// - Perform a forward pass to compute outputs from inputs
/// - Perform a backward pass to compute gradients
/// - Expose its learnable parameters
///
/// # Example
///
/// ```
/// use monolith_layers::dense::Dense;
/// use monolith_layers::layer::Layer;
/// use monolith_layers::tensor::Tensor;
///
/// let layer = Dense::new(128, 64);
/// let input = Tensor::zeros(&[32, 128]); // batch of 32, input dim 128
/// let output = layer.forward(&input).unwrap();
/// assert_eq!(output.shape(), &[32, 64]);
/// ```
pub trait Layer: Send + Sync {
    /// Performs a forward pass through the layer.
    ///
    /// Takes an input tensor and produces an output tensor by applying
    /// the layer's transformation.
    ///
    /// # Arguments
    ///
    /// * `input` - The input tensor to the layer
    ///
    /// # Returns
    ///
    /// The output tensor after applying the layer's transformation
    ///
    /// # Errors
    ///
    /// Returns a [`LayerError`] if the input shape is incompatible with the layer
    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError>;

    /// Performs a backward pass through the layer.
    ///
    /// Takes the gradient of the loss with respect to the layer's output
    /// and computes the gradient with respect to the layer's input.
    /// This also updates internal gradient accumulators for the layer's parameters.
    ///
    /// # Arguments
    ///
    /// * `grad` - The gradient of the loss with respect to the layer's output
    ///
    /// # Returns
    ///
    /// The gradient of the loss with respect to the layer's input
    ///
    /// # Errors
    ///
    /// Returns a [`LayerError`] if the gradient shape is incompatible
    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError>;

    /// Returns references to the layer's learnable parameters.
    ///
    /// This is used by optimizers to update the layer's weights during training.
    ///
    /// # Returns
    ///
    /// A vector of references to the layer's parameter tensors
    fn parameters(&self) -> Vec<&Tensor>;

    /// Returns mutable references to the layer's learnable parameters.
    ///
    /// This is used by optimizers to update the layer's weights during training.
    ///
    /// # Returns
    ///
    /// A vector of mutable references to the layer's parameter tensors
    fn parameters_mut(&mut self) -> Vec<&mut Tensor>;

    /// Returns the regularization loss contributed by this layer.
    ///
    /// Default implementation returns 0.0.
    fn regularization_loss(&self) -> f32 {
        0.0
    }

    /// Returns the name of the layer for debugging and logging purposes.
    fn name(&self) -> &str {
        "Layer"
    }

    /// Returns whether the layer is in training mode.
    ///
    /// Some layers behave differently during training vs inference
    /// (e.g., Dropout, BatchNorm).
    fn is_training(&self) -> bool {
        true
    }

    /// Sets the layer's training mode.
    ///
    /// # Arguments
    ///
    /// * `training` - Whether to enable training mode
    fn set_training(&mut self, _training: bool) {
        // Default implementation does nothing
    }

    /// Applies parameter constraints (if any).
    ///
    /// Default implementation does nothing.
    fn apply_constraints(&mut self) {
        // Default implementation does nothing
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock layer for testing
    struct MockLayer {
        weight: Tensor,
        training: bool,
    }

    impl MockLayer {
        fn new() -> Self {
            Self {
                weight: Tensor::zeros(&[10, 10]),
                training: true,
            }
        }
    }

    impl Layer for MockLayer {
        fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
            Ok(input.clone())
        }

        fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
            Ok(grad.clone())
        }

        fn parameters(&self) -> Vec<&Tensor> {
            vec![&self.weight]
        }

        fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
            vec![&mut self.weight]
        }

        fn name(&self) -> &str {
            "MockLayer"
        }

        fn is_training(&self) -> bool {
            self.training
        }

        fn set_training(&mut self, training: bool) {
            self.training = training;
        }
    }

    #[test]
    fn test_layer_trait() {
        let mut layer = MockLayer::new();
        let input = Tensor::zeros(&[2, 10]);

        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape(), input.shape());

        let grad = Tensor::ones(&[2, 10]);
        let input_grad = layer.backward(&grad).unwrap();
        assert_eq!(input_grad.shape(), grad.shape());

        assert_eq!(layer.parameters().len(), 1);
        assert_eq!(layer.name(), "MockLayer");
    }

    #[test]
    fn test_training_mode() {
        let mut layer = MockLayer::new();
        assert!(layer.is_training());

        layer.set_training(false);
        assert!(!layer.is_training());

        layer.set_training(true);
        assert!(layer.is_training());
    }
}
