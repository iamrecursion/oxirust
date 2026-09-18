//! Image resize kernels with various interpolation methods

use crate::{GpuDevice, Result};

/// Resize interpolation filter
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResizeFilter {
    /// Nearest neighbor - fastest, lowest quality
    Nearest,
    /// Bilinear interpolation - balanced quality/speed
    Bilinear,
    /// Bicubic interpolation - high quality, slower
    Bicubic,
    /// Lanczos resampling - highest quality, slowest
    Lanczos,
    /// Area averaging - good for downscaling
    Area,
}

impl ResizeFilter {
    /// Get the filter ID for shader dispatch
    #[must_use]
    pub fn to_id(self) -> u32 {
        match self {
            Self::Nearest => 0,
            Self::Bilinear => 1,
            Self::Bicubic => 2,
            Self::Lanczos => 3,
            Self::Area => 4,
        }
    }

    /// Get the filter name
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Nearest => "Nearest",
            Self::Bilinear => "Bilinear",
            Self::Bicubic => "Bicubic",
            Self::Lanczos => "Lanczos",
            Self::Area => "Area",
        }
    }

    /// Get the filter kernel radius
    #[must_use]
    pub fn kernel_radius(self) -> u32 {
        match self {
            Self::Nearest => 0,
            Self::Bilinear => 1,
            Self::Bicubic => 2,
            Self::Lanczos => 3,
            Self::Area => 1,
        }
    }
}

/// Image resize kernel
pub struct ResizeKernel {
    filter: ResizeFilter,
}

impl ResizeKernel {
    /// Create a new resize kernel
    #[must_use]
    pub fn new(filter: ResizeFilter) -> Self {
        Self { filter }
    }

    /// Execute the resize operation
    ///
    /// # Arguments
    ///
    /// * `device` - GPU device
    /// * `input` - Input image buffer
    /// * `src_width` - Source image width
    /// * `src_height` - Source image height
    /// * `output` - Output image buffer
    /// * `dst_width` - Destination image width
    /// * `dst_height` - Destination image height
    ///
    /// # Errors
    ///
    /// Returns an error if the resize operation fails.
    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &self,
        device: &GpuDevice,
        input: &[u8],
        _src_width: u32,
        _src_height: u32,
        output: &mut [u8],
        dst_width: u32,
        dst_height: u32,
    ) -> Result<()> {
        // Delegate to the ops module implementation
        crate::ops::ScaleOperation::scale(
            device,
            input,
            _src_width,
            _src_height,
            output,
            dst_width,
            dst_height,
            self.filter.into(),
        )
    }

    /// Get the filter type
    #[must_use]
    pub fn filter(&self) -> ResizeFilter {
        self.filter
    }

    /// Calculate output buffer size
    #[must_use]
    pub fn output_size(dst_width: u32, dst_height: u32, channels: u32) -> usize {
        (dst_width * dst_height * channels) as usize
    }

    /// Estimate FLOPS for the resize operation
    #[must_use]
    pub fn estimate_flops(
        _src_width: u32,
        _src_height: u32,
        dst_width: u32,
        dst_height: u32,
        filter: ResizeFilter,
    ) -> u64 {
        let output_pixels = u64::from(dst_width) * u64::from(dst_height);
        let kernel_size = u64::from(filter.kernel_radius());
        let samples_per_pixel = (kernel_size * 2 + 1) * (kernel_size * 2 + 1);

        // Each sample requires interpolation (4-8 ops per sample)
        output_pixels * samples_per_pixel * 6
    }
}

impl From<ResizeFilter> for crate::ops::ScaleFilter {
    fn from(filter: ResizeFilter) -> Self {
        match filter {
            ResizeFilter::Nearest => Self::Nearest,
            ResizeFilter::Bilinear => Self::Bilinear,
            ResizeFilter::Bicubic => Self::Bicubic,
            ResizeFilter::Area => Self::Area,
            ResizeFilter::Lanczos => Self::Bicubic, // Fallback to bicubic for Lanczos
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resize_filter_properties() {
        assert_eq!(ResizeFilter::Nearest.to_id(), 0);
        assert_eq!(ResizeFilter::Bilinear.to_id(), 1);
        assert_eq!(ResizeFilter::Bicubic.to_id(), 2);

        assert_eq!(ResizeFilter::Nearest.kernel_radius(), 0);
        assert_eq!(ResizeFilter::Bilinear.kernel_radius(), 1);
        assert_eq!(ResizeFilter::Bicubic.kernel_radius(), 2);
        assert_eq!(ResizeFilter::Lanczos.kernel_radius(), 3);
    }

    #[test]
    fn test_output_size_calculation() {
        assert_eq!(ResizeKernel::output_size(1920, 1080, 4), 1920 * 1080 * 4);
        assert_eq!(ResizeKernel::output_size(640, 480, 3), 640 * 480 * 3);
    }

    #[test]
    fn test_flops_estimation() {
        let flops = ResizeKernel::estimate_flops(1920, 1080, 960, 540, ResizeFilter::Bilinear);
        assert!(flops > 0);

        let flops_bicubic =
            ResizeKernel::estimate_flops(1920, 1080, 960, 540, ResizeFilter::Bicubic);
        assert!(flops_bicubic > flops); // Bicubic should be more expensive
    }
}
