//! Internal utilities for oxicode

/// Sealed trait to prevent external implementations
pub trait Sealed {}

impl<T> Sealed for &mut T where T: Sealed {}
