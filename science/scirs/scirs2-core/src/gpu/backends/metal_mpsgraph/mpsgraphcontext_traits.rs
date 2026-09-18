//! # MPSGraphContext - Trait Implementations
//!
//! This module contains trait implementations for `MPSGraphContext`.
//!
//! ## Implemented Traits
//!
//! - `Send`
//! - `Sync`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::MPSGraphContext;

unsafe impl Send for MPSGraphContext {}

unsafe impl Sync for MPSGraphContext {}
