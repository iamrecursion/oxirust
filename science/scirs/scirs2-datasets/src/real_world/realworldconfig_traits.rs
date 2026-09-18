//! # RealWorldConfig - Trait Implementations
//!
//! This module contains trait implementations for `RealWorldConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::RealWorldConfig;

impl Default for RealWorldConfig {
    fn default() -> Self {
        Self {
            use_cache: true,
            download_if_missing: true,
            data_home: None,
            return_preprocessed: false,
            subset: None,
            random_state: None,
        }
    }
}
