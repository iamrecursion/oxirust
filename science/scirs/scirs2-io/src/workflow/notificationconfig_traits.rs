//! # NotificationConfig - Trait Implementations
//!
//! This module contains trait implementations for `NotificationConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::NotificationConfig;

impl Default for NotificationConfig {
    fn default() -> Self {
        Self {
            on_success: false,
            on_failure: true,
            on_start: false,
            channels: Vec::new(),
        }
    }
}
