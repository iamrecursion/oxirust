//! GPU compute kernels library
//!
//! This module provides a comprehensive library of GPU compute kernels
//! for common image and signal processing operations.

pub mod color;
pub mod filter;
pub mod reduce;
pub mod resize;
pub mod transform;

pub use color::{ColorConversionKernel, ColorSpace as KernelColorSpace};
pub use filter::{ConvolutionKernel, FilterKernel};
pub use reduce::{ReduceKernel, ReduceOp};
pub use resize::{ResizeFilter, ResizeKernel};
pub use transform::{TransformKernel, TransformType};

/// Kernel execution statistics
#[derive(Debug, Clone, Copy, Default)]
pub struct KernelStats {
    /// Number of workgroups dispatched
    pub workgroups: (u32, u32, u32),
    /// Total number of threads
    pub total_threads: u64,
    /// Estimated FLOPS (floating-point operations)
    pub estimated_flops: u64,
}

impl KernelStats {
    /// Create new kernel statistics
    #[must_use]
    pub fn new(workgroups: (u32, u32, u32), threads_per_workgroup: u32) -> Self {
        let total_threads = u64::from(workgroups.0)
            * u64::from(workgroups.1)
            * u64::from(workgroups.2)
            * u64::from(threads_per_workgroup);

        Self {
            workgroups,
            total_threads,
            estimated_flops: 0,
        }
    }

    /// Set the estimated FLOPS
    #[must_use]
    pub fn with_flops(mut self, flops: u64) -> Self {
        self.estimated_flops = flops;
        self
    }
}
