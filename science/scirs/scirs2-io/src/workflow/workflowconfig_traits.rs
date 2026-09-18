//! # WorkflowConfig - Trait Implementations
//!
//! This module contains trait implementations for `WorkflowConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{NotificationConfig, RetryPolicy, WorkflowConfig};

impl Default for WorkflowConfig {
    fn default() -> Self {
        Self {
            max_parallel_tasks: 4,
            retry_policy: RetryPolicy::default(),
            timeout: None,
            checkpoint_dir: None,
            notifications: NotificationConfig::default(),
            scheduling: None,
        }
    }
}
