//! # ArimaParams - Trait Implementations
//!
//! This module contains trait implementations for `ArimaParams`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ArimaParams;

impl Default for ArimaParams {
    fn default() -> Self {
        Self {
            p: 1,
            d: 0,
            q: 0,
            seasonal_p: None,
            seasonal_d: None,
            seasonal_q: None,
            seasonal_period: None,
            fit_intercept: true,
            trend: None,
        }
    }
}
