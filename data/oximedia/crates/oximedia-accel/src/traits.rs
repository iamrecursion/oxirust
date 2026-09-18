//! Traits for hardware acceleration.

use crate::error::AccelResult;
use oximedia_core::PixelFormat;

/// Scaling filter quality/algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleFilter {
    /// Nearest neighbor (fastest, lowest quality).
    Nearest,
    /// Bilinear interpolation (good speed/quality balance).
    Bilinear,
    /// Bicubic interpolation (slower, higher quality).
    Bicubic,
    /// Lanczos resampling (slowest, highest quality).
    Lanczos,
}

/// Hardware acceleration trait for video processing operations.
///
/// This trait provides a unified interface for GPU and CPU implementations
/// of common video processing operations.
pub trait HardwareAccel: Send + Sync {
    /// Scales an image from source dimensions to destination dimensions.
    ///
    /// # Arguments
    ///
    /// * `input` - Input image data
    /// * `src_width` - Source image width
    /// * `src_height` - Source image height
    /// * `dst_width` - Destination image width
    /// * `dst_height` - Destination image height
    /// * `format` - Pixel format of the image
    /// * `filter` - Scaling filter to use
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Input buffer size doesn't match dimensions
    /// - GPU operation fails
    /// - Format is unsupported
    fn scale_image(
        &self,
        input: &[u8],
        src_width: u32,
        src_height: u32,
        dst_width: u32,
        dst_height: u32,
        format: PixelFormat,
        filter: ScaleFilter,
    ) -> AccelResult<Vec<u8>>;

    /// Converts image from one pixel format to another.
    ///
    /// # Arguments
    ///
    /// * `input` - Input image data
    /// * `width` - Image width
    /// * `height` - Image height
    /// * `src_format` - Source pixel format
    /// * `dst_format` - Destination pixel format
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Input buffer size doesn't match dimensions
    /// - GPU operation fails
    /// - Format conversion is unsupported
    fn convert_color(
        &self,
        input: &[u8],
        width: u32,
        height: u32,
        src_format: PixelFormat,
        dst_format: PixelFormat,
    ) -> AccelResult<Vec<u8>>;

    /// Performs block-based motion estimation between two frames.
    ///
    /// Returns a vector of motion vectors (dx, dy) for each block.
    /// Blocks are scanned left-to-right, top-to-bottom.
    ///
    /// # Arguments
    ///
    /// * `reference` - Reference frame data
    /// * `current` - Current frame data
    /// * `width` - Frame width
    /// * `height` - Frame height
    /// * `block_size` - Size of motion estimation blocks (typically 8 or 16)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Input buffer sizes don't match dimensions
    /// - GPU operation fails
    /// - Block size is invalid
    fn motion_estimation(
        &self,
        reference: &[u8],
        current: &[u8],
        width: u32,
        height: u32,
        block_size: u32,
    ) -> AccelResult<Vec<(i16, i16)>>;
}
