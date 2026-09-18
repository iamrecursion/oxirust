//! # ExecutorConfig - Trait Implementations
//!
//! This module contains trait implementations for `ExecutorConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use chrono::{DateTime, Datelike, Duration, Utc};

use super::types::ExecutorConfig;

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            max_concurrentworkflows: 10,
            task_timeout: Duration::hours(1),
            checkpoint_interval: Duration::minutes(5),
        }
    }
}
