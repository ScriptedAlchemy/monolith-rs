//! Entry types for embedding storage.

use serde::{Deserialize, Serialize};

/// State maintained by optimizers for each embedding.
///
/// Different optimizers maintain different state:
/// - SGD: No additional state
/// - Adam: First and second moment estimates
/// - Adagrad: Sum of squared gradients
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizerState {
    /// No optimizer state (used for SGD without momentum).
    None,

    /// State for Adam optimizer.
    Adam {
        /// First moment estimate (mean of gradients).
        m: Vec<f32>,
        /// Second moment estimate (variance of gradients).
        v: Vec<f32>,
        /// Number of updates applied.
        t: u64,
    },

    /// State for Adagrad optimizer.
    Adagrad {
        /// Sum of squared gradients.
        accumulator: Vec<f32>,
    },

    /// State for SGD with momentum.
    Momentum {
        /// Velocity vector.
        velocity: Vec<f32>,
    },

    /// State for FTRL optimizer.
    Ftrl {
        /// Accumulated gradient.
        z: Vec<f32>,
        /// Accumulated squared gradient.
        n: Vec<f32>,
    },
}

impl OptimizerState {
    /// Creates a new Adam optimizer state with the given dimension.
    pub fn new_adam(dim: usize) -> Self {
        Self::Adam {
            m: vec![0.0; dim],
            v: vec![0.0; dim],
            t: 0,
        }
    }

    /// Creates a new Adagrad optimizer state with the given dimension.
    pub fn new_adagrad(dim: usize) -> Self {
        Self::Adagrad {
            accumulator: vec![0.0; dim],
        }
    }

    /// Creates a new momentum optimizer state with the given dimension.
    pub fn new_momentum(dim: usize) -> Self {
        Self::Momentum {
            velocity: vec![0.0; dim],
        }
    }

    /// Creates a new FTRL optimizer state with the given dimension.
    pub fn new_ftrl(dim: usize) -> Self {
        Self::Ftrl {
            z: vec![0.0; dim],
            n: vec![0.0; dim],
        }
    }

    /// Returns the memory size of this optimizer state in bytes.
    pub fn memory_size(&self) -> usize {
        match self {
            Self::None => 0,
            Self::Adam { m, v, .. } => (m.len() + v.len()) * std::mem::size_of::<f32>() + 8,
            Self::Adagrad { accumulator } => accumulator.len() * std::mem::size_of::<f32>(),
            Self::Momentum { velocity } => velocity.len() * std::mem::size_of::<f32>(),
            Self::Ftrl { z, n } => (z.len() + n.len()) * std::mem::size_of::<f32>(),
        }
    }
}

impl Default for OptimizerState {
    fn default() -> Self {
        Self::None
    }
}

/// An entry in the embedding hash table.
///
/// Each entry contains:
/// - The feature ID
/// - The embedding vector
/// - Optional optimizer state for training
/// - Last update timestamp for eviction
///
/// # Example
///
/// ```
/// use monolith_hash_table::{EmbeddingEntry, OptimizerState};
///
/// let entry = EmbeddingEntry::new(42, vec![0.1, 0.2, 0.3, 0.4]);
/// assert_eq!(entry.id(), 42);
/// assert_eq!(entry.embedding().len(), 4);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingEntry {
    /// The feature ID.
    id: i64,
    /// The embedding vector.
    embedding: Vec<f32>,
    /// Optimizer state for this embedding.
    optimizer_state: OptimizerState,
    /// Last update timestamp (Unix timestamp in seconds).
    last_update_ts: u64,
}

impl EmbeddingEntry {
    /// Creates a new embedding entry with no optimizer state.
    ///
    /// The last update timestamp is initialized to 0.
    pub fn new(id: i64, embedding: Vec<f32>) -> Self {
        Self {
            id,
            embedding,
            optimizer_state: OptimizerState::None,
            last_update_ts: 0,
        }
    }

    /// Creates a new embedding entry with the specified optimizer state.
    ///
    /// The last update timestamp is initialized to 0.
    pub fn with_optimizer_state(
        id: i64,
        embedding: Vec<f32>,
        optimizer_state: OptimizerState,
    ) -> Self {
        Self {
            id,
            embedding,
            optimizer_state,
            last_update_ts: 0,
        }
    }

    /// Creates a new embedding entry with a specific timestamp.
    pub fn with_timestamp(id: i64, embedding: Vec<f32>, timestamp: u64) -> Self {
        Self {
            id,
            embedding,
            optimizer_state: OptimizerState::None,
            last_update_ts: timestamp,
        }
    }

    /// Returns the feature ID.
    #[inline]
    pub fn id(&self) -> i64 {
        self.id
    }

    /// Returns a reference to the embedding vector.
    #[inline]
    pub fn embedding(&self) -> &[f32] {
        &self.embedding
    }

    /// Returns a mutable reference to the embedding vector.
    #[inline]
    pub fn embedding_mut(&mut self) -> &mut [f32] {
        &mut self.embedding
    }

    /// Sets the embedding vector.
    pub fn set_embedding(&mut self, embedding: Vec<f32>) {
        self.embedding = embedding;
    }

    /// Copies the embedding data from a slice.
    pub fn copy_embedding_from(&mut self, data: &[f32]) {
        self.embedding.clear();
        self.embedding.extend_from_slice(data);
    }

    /// Returns a reference to the optimizer state.
    #[inline]
    pub fn optimizer_state(&self) -> &OptimizerState {
        &self.optimizer_state
    }

    /// Returns a mutable reference to the optimizer state.
    #[inline]
    pub fn optimizer_state_mut(&mut self) -> &mut OptimizerState {
        &mut self.optimizer_state
    }

    /// Sets the optimizer state.
    pub fn set_optimizer_state(&mut self, state: OptimizerState) {
        self.optimizer_state = state;
    }

    /// Returns the last update timestamp.
    #[inline]
    pub fn get_timestamp(&self) -> u64 {
        self.last_update_ts
    }

    /// Sets the last update timestamp.
    #[inline]
    pub fn set_timestamp(&mut self, timestamp: u64) {
        self.last_update_ts = timestamp;
    }

    /// Returns the dimension of the embedding.
    #[inline]
    pub fn dim(&self) -> usize {
        self.embedding.len()
    }

    /// Returns the total memory size of this entry in bytes.
    pub fn memory_size(&self) -> usize {
        std::mem::size_of::<i64>()
            + self.embedding.len() * std::mem::size_of::<f32>()
            + self.optimizer_state.memory_size()
            + std::mem::size_of::<u64>() // last_update_ts
    }

    /// Applies a gradient update to this entry's embedding using the stored optimizer state.
    ///
    /// This method handles both the embedding update and optimizer state update in a single
    /// call to avoid borrow checker issues.
    ///
    /// # Arguments
    ///
    /// * `gradients` - The gradient values for each embedding dimension
    /// * `learning_rate` - The learning rate to use for the update
    pub fn apply_gradient_update(&mut self, gradients: &[f32], learning_rate: f32) {
        match &mut self.optimizer_state {
            OptimizerState::None => {
                // Simple SGD
                for (emb, &grad) in self.embedding.iter_mut().zip(gradients.iter()) {
                    *emb -= learning_rate * grad;
                }
            }
            OptimizerState::Adam { m, v, t } => {
                const BETA1: f32 = 0.9;
                const BETA2: f32 = 0.999;
                const EPSILON: f32 = 1e-8;

                *t += 1;
                let t_f = *t as f32;

                for (j, &grad) in gradients.iter().enumerate() {
                    m[j] = BETA1 * m[j] + (1.0 - BETA1) * grad;
                    v[j] = BETA2 * v[j] + (1.0 - BETA2) * grad * grad;

                    let m_hat = m[j] / (1.0 - BETA1.powf(t_f));
                    let v_hat = v[j] / (1.0 - BETA2.powf(t_f));

                    self.embedding[j] -= learning_rate * m_hat / (v_hat.sqrt() + EPSILON);
                }
            }
            OptimizerState::Adagrad { accumulator } => {
                const EPSILON: f32 = 1e-8;

                for (j, &grad) in gradients.iter().enumerate() {
                    accumulator[j] += grad * grad;
                    self.embedding[j] -= learning_rate * grad / (accumulator[j].sqrt() + EPSILON);
                }
            }
            OptimizerState::Momentum { velocity } => {
                const MOMENTUM: f32 = 0.9;

                for (j, &grad) in gradients.iter().enumerate() {
                    velocity[j] = MOMENTUM * velocity[j] + learning_rate * grad;
                    self.embedding[j] -= velocity[j];
                }
            }
            OptimizerState::Ftrl { z, n } => {
                const ALPHA: f32 = 1.0;
                const BETA: f32 = 1.0;
                const LAMBDA1: f32 = 0.0;
                const LAMBDA2: f32 = 0.0;

                for (j, &grad) in gradients.iter().enumerate() {
                    let sigma = (n[j] + grad * grad).sqrt() / ALPHA - n[j].sqrt() / ALPHA;
                    z[j] += grad - sigma * self.embedding[j];
                    n[j] += grad * grad;

                    if z[j].abs() <= LAMBDA1 {
                        self.embedding[j] = 0.0;
                    } else {
                        let sign = if z[j] > 0.0 { 1.0 } else { -1.0 };
                        self.embedding[j] =
                            -(z[j] - sign * LAMBDA1) / ((BETA + n[j].sqrt()) / ALPHA + LAMBDA2);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_entry_creation() {
        let entry = EmbeddingEntry::new(42, vec![1.0, 2.0, 3.0, 4.0]);

        assert_eq!(entry.id(), 42);
        assert_eq!(entry.embedding(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(entry.dim(), 4);
        assert!(matches!(entry.optimizer_state(), OptimizerState::None));
    }

    #[test]
    fn test_embedding_entry_with_adam() {
        let entry =
            EmbeddingEntry::with_optimizer_state(1, vec![0.5, 0.5], OptimizerState::new_adam(2));

        assert_eq!(entry.id(), 1);
        assert!(matches!(
            entry.optimizer_state(),
            OptimizerState::Adam { .. }
        ));

        if let OptimizerState::Adam { m, v, t } = entry.optimizer_state() {
            assert_eq!(m.len(), 2);
            assert_eq!(v.len(), 2);
            assert_eq!(*t, 0);
        }
    }

    #[test]
    fn test_embedding_entry_mutation() {
        let mut entry = EmbeddingEntry::new(1, vec![1.0, 2.0]);

        entry.embedding_mut()[0] = 10.0;
        assert_eq!(entry.embedding(), &[10.0, 2.0]);

        entry.copy_embedding_from(&[3.0, 4.0]);
        assert_eq!(entry.embedding(), &[3.0, 4.0]);
    }

    #[test]
    fn test_optimizer_state_memory_size() {
        assert_eq!(OptimizerState::None.memory_size(), 0);

        let adam = OptimizerState::new_adam(4);
        assert!(adam.memory_size() > 0);

        let adagrad = OptimizerState::new_adagrad(4);
        assert!(adagrad.memory_size() > 0);
    }
}
