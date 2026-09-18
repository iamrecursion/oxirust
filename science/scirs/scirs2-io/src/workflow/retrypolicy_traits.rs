//! # RetryPolicy - Trait Implementations
//!
//! This module contains trait implementations for `RetryPolicy`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::RetryPolicy;

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            backoff_seconds: 60,
            exponential_backoff: true,
        }
    }
}
