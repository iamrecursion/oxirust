//! # LearningStatistics - Trait Implementations
//!
//! This module contains trait implementations for `LearningStatistics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::parallel_ops::*;
use scirs2_core::random::prelude::*;

use super::types::{LearningStatistics, LearningTrend};

impl Default for LearningStatistics {
    fn default() -> Self {
        Self {
            average_error: 0.0,
            learning_trend: LearningTrend::Unknown,
            total_episodes: 0,
            architecture_changes: 0,
            current_learning_rate: 0.001,
        }
    }
}
