//! # MemoryLimits - Trait Implementations
//!
//! This module contains trait implementations for `MemoryLimits`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::MemoryLimits;

impl Default for MemoryLimits {
    fn default() -> Self {
        Self::auto_detect()
    }
}
