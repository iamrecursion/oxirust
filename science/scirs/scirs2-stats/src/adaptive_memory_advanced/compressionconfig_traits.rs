//! # CompressionConfig - Trait Implementations
//!
//! This module contains trait implementations for `CompressionConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::{
    parallel_ops::*,
    simd_ops::{PlatformCapabilities, SimdUnifiedOps},
};

use super::types::{CompressionAlgorithm, CompressionConfig};

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            enable_compression: true,
            algorithm: CompressionAlgorithm::Zstd,
            compression_level: 3,
            compression_threshold: 1024,
            adaptive_compression: true,
        }
    }
}
