//! Base model parameter traits mirroring Python `SingleTaskModelParams`.

use crate::hyperparams::Params;

/// Parameters for a single-task model.
///
/// Implementors should return task configuration via [`task`].
pub trait SingleTaskModelParams: Send + Sync {
    /// Returns task params for the model.
    fn task(&self) -> Params;
}
