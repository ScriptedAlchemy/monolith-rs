//! AddBias layer to add a learnable bias with data format handling.

use crate::error::LayerError;
use crate::initializer::Initializer;
use crate::layer::Layer;
use crate::regularizer::Regularizer;
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};

/// Data format for channel position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataFormat {
    ChannelsFirst,
    ChannelsLast,
}

impl Default for DataFormat {
    fn default() -> Self {
        DataFormat::ChannelsLast
    }
}

/// AddBias layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddBias {
    initializer: Initializer,
    regularizer: Regularizer,
    data_format: DataFormat,
    bias: Option<Tensor>,
    bias_grad: Option<Tensor>,
    cached_input_shape: Option<Vec<usize>>,
}

impl AddBias {
    /// Creates a new AddBias layer with zero initializer.
    pub fn new() -> Self {
        Self {
            initializer: Initializer::Zeros,
            regularizer: Regularizer::None,
            data_format: DataFormat::ChannelsLast,
            bias: None,
            bias_grad: None,
            cached_input_shape: None,
        }
    }

    /// Sets initializer.
    pub fn with_initializer(mut self, initializer: Initializer) -> Self {
        self.initializer = initializer;
        self
    }

    /// Sets bias regularizer.
    pub fn with_regularizer(mut self, regularizer: Regularizer) -> Self {
        self.regularizer = regularizer;
        self
    }

    /// Sets data format.
    pub fn with_data_format(mut self, data_format: DataFormat) -> Self {
        self.data_format = data_format;
        self
    }

    fn ensure_bias(&mut self, input_shape: &[usize]) {
        if self.bias.is_some() {
            return;
        }
        let bias_shape = input_shape[1..].to_vec();
        let bias = self.initializer.initialize(&bias_shape);
        self.bias = Some(bias);
    }

    fn reshape_bias_for_input(
        &self,
        input_shape: &[usize],
        data_format: DataFormat,
    ) -> Result<Tensor, LayerError> {
        let bias = self.bias.as_ref().ok_or(LayerError::NotInitialized)?;
        let bias_shape = bias.shape().to_vec();
        let ndim = input_shape.len();

        if bias_shape.len() == 1 {
            let c = bias_shape[0];
            let mut shape = vec![1usize; ndim];
            match data_format {
                DataFormat::ChannelsLast => {
                    shape[ndim - 1] = c;
                }
                DataFormat::ChannelsFirst => {
                    if ndim < 2 {
                        return Err(LayerError::ForwardError {
                            message: "ChannelsFirst requires ndim >= 2".to_string(),
                        });
                    }
                    shape[1] = c;
                }
            }
            return Ok(bias.reshape(&shape));
        }

        if bias_shape.len() == ndim - 1 {
            match data_format {
                DataFormat::ChannelsLast => {
                    let mut shape = Vec::with_capacity(ndim);
                    shape.push(1);
                    shape.extend_from_slice(&bias_shape);
                    return Ok(bias.reshape(&shape));
                }
                DataFormat::ChannelsFirst => {
                    let shape = match ndim {
                        3 => vec![1, bias_shape[1], bias_shape[0]],
                        4 => vec![1, bias_shape[2], bias_shape[0], bias_shape[1]],
                        5 => vec![
                            1,
                            bias_shape[3],
                            bias_shape[0],
                            bias_shape[1],
                            bias_shape[2],
                        ],
                        _ => {
                            let mut s = Vec::with_capacity(ndim);
                            s.push(1);
                            s.extend_from_slice(&bias_shape);
                            s
                        }
                    };
                    return Ok(bias.reshape(&shape));
                }
            }
        }

        let mut shape = vec![1usize; ndim - bias_shape.len()];
        shape.extend_from_slice(&bias_shape);
        Ok(bias.reshape(&shape))
    }

    /// Forward with explicit data format.
    pub fn forward_with_format(
        &mut self,
        input: &Tensor,
        data_format: DataFormat,
    ) -> Result<Tensor, LayerError> {
        if input.ndim() < 2 {
            return Err(LayerError::ForwardError {
                message: format!("AddBias expects >=2D input, got {}D", input.ndim()),
            });
        }
        self.ensure_bias(input.shape());
        self.cached_input_shape = Some(input.shape().to_vec());
        let bias_view = self.reshape_bias_for_input(input.shape(), data_format)?;
        Ok(input.add(&bias_view))
    }

    /// Returns bias tensor.
    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

impl Default for AddBias {
    fn default() -> Self {
        Self::new()
    }
}

impl Layer for AddBias {
    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        let mut layer = self.clone();
        layer.forward_with_format(input, self.data_format)
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let bias = self.bias.as_ref().ok_or(LayerError::NotInitialized)?;
        let input_shape = self
            .cached_input_shape
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;

        let ndim = input_shape.len();
        let bias_shape = bias.shape().to_vec();

        let grad_bias = if bias_shape.len() == 1 {
            let channel_axis = match self.data_format {
                DataFormat::ChannelsLast => ndim - 1,
                DataFormat::ChannelsFirst => 1,
            };
            let axes: Vec<usize> = (0..ndim).filter(|&i| i != channel_axis).collect();
            grad.sum_axes(&axes)
        } else {
            grad.sum_axes(&[0])
        };

        self.bias_grad = Some(grad_bias);
        Ok(grad.clone())
    }

    fn parameters(&self) -> Vec<&Tensor> {
        self.bias.as_ref().map(|b| vec![b]).unwrap_or_default()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        self.bias.as_mut().map(|b| vec![b]).unwrap_or_default()
    }

    fn name(&self) -> &str {
        "AddBias"
    }

    fn regularization_loss(&self) -> f32 {
        if let Some(bias) = &self.bias {
            self.regularizer.loss(bias)
        } else {
            0.0
        }
    }
}
