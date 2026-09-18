//! # ExpSmoothingParams - Trait Implementations
//!
//! This module contains trait implementations for `ExpSmoothingParams`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ExpSmoothingParams;

impl Default for ExpSmoothingParams {
    fn default() -> Self {
        Self {
            alpha: 0.3,
            beta: None,
            gamma: None,
            seasonal_period: None,
            multiplicative_trend: false,
            multiplicative_seasonality: false,
            damped_trend: false,
            phi: None,
        }
    }
}
