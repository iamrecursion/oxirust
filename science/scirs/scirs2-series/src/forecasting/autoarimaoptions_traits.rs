//! # AutoArimaOptions - Trait Implementations
//!
//! This module contains trait implementations for `AutoArimaOptions`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::AutoArimaOptions;

impl Default for AutoArimaOptions {
    fn default() -> Self {
        Self {
            max_p: 5,
            max_d: 2,
            max_q: 5,
            seasonal: false,
            seasonal_period: None,
            max_seasonal_p: 2,
            max_seasonal_d: 1,
            max_seasonal_q: 2,
            auto_diff: true,
            with_constant: true,
            information_criterion: "aic".to_string(),
            stepwise: true,
            max_order: 10,
        }
    }
}
