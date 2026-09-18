//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::{Float, NumCast, One, Zero};
use scirs2_core::{
    parallel_ops::*,
    simd_ops::{PlatformCapabilities, SimdUnifiedOps},
};

use super::types::{AdaptiveMemoryConfig, AdaptiveMemoryManager, TrainingExample};

/// Predictive model trait
pub trait PredictiveModel: Send + Sync {
    fn predict(&self, features: &[f64]) -> f64;
    fn train(&mut self, trainingdata: &[TrainingExample]) -> Result<(), String>;
    fn get_confidence(&self) -> f64;
    fn get_feature_importance(&self) -> Vec<f64>;
}
/// Compressor trait
pub trait Compressor: Send + Sync {
    fn compress(&self, data: &[u8]) -> Result<Vec<u8>, String>;
    fn decompress(&self, data: &[u8]) -> Result<Vec<u8>, String>;
    fn compression_ratio(&self, originalsize: usize, compressedsize: usize) -> f64;
}
/// Convenient type aliases
pub type F64AdaptiveMemoryManager = AdaptiveMemoryManager<f64>;
pub type F32AdaptiveMemoryManager = AdaptiveMemoryManager<f32>;
/// Factory functions
#[allow(dead_code)]
pub fn create_adaptive_memory_manager<F>() -> AdaptiveMemoryManager<F>
where
    F: Float
        + NumCast
        + SimdUnifiedOps
        + Zero
        + One
        + PartialOrd
        + Copy
        + Send
        + Sync
        + 'static
        + std::fmt::Display,
{
    AdaptiveMemoryManager::new()
}
#[allow(dead_code)]
pub fn create_optimized_memory_manager<F>(config: AdaptiveMemoryConfig) -> AdaptiveMemoryManager<F>
where
    F: Float
        + NumCast
        + SimdUnifiedOps
        + Zero
        + One
        + PartialOrd
        + Copy
        + Send
        + Sync
        + 'static
        + std::fmt::Display,
{
    AdaptiveMemoryManager::with_config(config)
}
