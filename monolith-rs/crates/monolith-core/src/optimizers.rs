//! Optimizer registry mirroring Python `monolith/core/optimizers.py`.
//!
//! The Python file provides a simple mapping from string keys to TensorFlow
//! optimizer classes. In Rust, optimizers live in `monolith-optimizer`, so we
//! provide:
//! - an enum for the supported names
//! - string parsing with Python-like keys
//! - conversion into `monolith_optimizer::OptimizerConfig` where possible

use crate::error::{MonolithError, Result};

/// Supported optimizer names from Python `optimizers` dict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizerName {
    Adagrad,
    Momentum,
    Rmsprop,
    Adam,
}

impl OptimizerName {
    /// Parses a Python optimizer key (e.g. `"adam"`).
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "adagrad" => Ok(Self::Adagrad),
            "momentum" => Ok(Self::Momentum),
            "rmsprop" => Ok(Self::Rmsprop),
            "adam" => Ok(Self::Adam),
            _ => Err(MonolithError::PyValueError {
                message: format!("Unknown optimizer: {}", s),
            }),
        }
    }

    /// Returns the canonical Python key.
    pub fn as_key(&self) -> &'static str {
        match self {
            OptimizerName::Adagrad => "adagrad",
            OptimizerName::Momentum => "momentum",
            OptimizerName::Rmsprop => "rmsprop",
            OptimizerName::Adam => "adam",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_optimizer_keys() {
        assert_eq!(OptimizerName::parse("adam").unwrap(), OptimizerName::Adam);
        assert_eq!(
            OptimizerName::parse("adagrad").unwrap(),
            OptimizerName::Adagrad
        );
        assert_eq!(
            OptimizerName::parse("momentum").unwrap(),
            OptimizerName::Momentum
        );
        assert_eq!(
            OptimizerName::parse("rmsprop").unwrap(),
            OptimizerName::Rmsprop
        );

        let err = OptimizerName::parse("nope").unwrap_err();
        assert_eq!(err.to_string(), "Unknown optimizer: nope");
    }
}
