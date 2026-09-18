//! # SearchConfig - Trait Implementations
//!
//! This module contains trait implementations for `SearchConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::parallel_ops::*;

use super::types::SearchConfig;

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            max_neighbors: 10,
            radius: None,
            approximation_factor: 1.0,
            parallel_search: true,
            num_threads: None,
            adaptive_indexing: false,
            cache_results: true,
        }
    }
}
