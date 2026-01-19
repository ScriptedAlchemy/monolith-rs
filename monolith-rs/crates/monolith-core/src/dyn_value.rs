//! Shared dynamic value trait for heterogeneous containers.

use std::any::Any;
use std::fmt;

/// Trait for dynamically-typed values stored in containers.
pub trait DynValue: Any + fmt::Debug + Send + Sync {
    fn as_any(&self) -> &dyn Any;
}

impl<T> DynValue for T
where
    T: Any + fmt::Debug + Send + Sync,
{
    fn as_any(&self) -> &dyn Any {
        self
    }
}
