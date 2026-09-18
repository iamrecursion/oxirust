//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::parallel_ops::*;
use scirs2_core::random::prelude::*;

use super::types::{AdaptiveStreamConfig, AdaptiveStreamingEngine, QualityAlert};

/// Alert callback function type alias
pub(super) type AlertCallback = Box<dyn Fn(&QualityAlert) + Send + Sync>;
/// Convenience function to create a new adaptive streaming engine
#[allow(dead_code)]
pub fn create_adaptive_engine() -> AdaptiveStreamingEngine {
    AdaptiveStreamingEngine::new(AdaptiveStreamConfig::default())
}
/// Convenience function to create a streaming engine with custom config
#[allow(dead_code)]
pub fn create_adaptive_engine_with_config(
    _config: AdaptiveStreamConfig,
) -> AdaptiveStreamingEngine {
    AdaptiveStreamingEngine::new(_config)
}
