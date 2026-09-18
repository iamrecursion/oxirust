//! # LossConfig - Trait Implementations
//!
//! This module contains trait implementations for `LossConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`

use super::types::LossConfig;

impl Default for LossConfig {
    fn default() -> Self {
        Self {
            supervised_weight: 1.0,
            constraint_weight: 1.0,
            rule_weight: 1.0,
            temperature: 1.0,
        }
    }
}
