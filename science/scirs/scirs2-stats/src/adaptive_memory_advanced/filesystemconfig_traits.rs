//! # FileSystemConfig - Trait Implementations
//!
//! This module contains trait implementations for `FileSystemConfig`.
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

use super::types::{FileSystemConfig, IOScheduler};

impl Default for FileSystemConfig {
    fn default() -> Self {
        Self {
            io_scheduler: IOScheduler::MQ,
            read_ahead: 128 * 1024,
            write_behind: true,
            direct_io: false,
            async_io: true,
        }
    }
}
