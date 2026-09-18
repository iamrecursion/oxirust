//! Image preprocessing for ML inference.
//!
//! [`ImagePreprocessor`] is a builder for converting raw pixel data
//! (RGB/BGR, u8 or f32) into a normalised `f32` tensor in either NCHW
//! or NHWC layout. The design is deliberately minimal — it covers the
//! overwhelmingly common case of ImageNet-style classifiers and shot-
//! boundary detectors, where input is "u8 RGB, scaled to a fixed size,
//! normalised by per-channel mean/std".
//!
//! Scaling is done with nearest-neighbour (Pure-Rust, zero deps); richer
//! interpolation is intentionally left to `oximedia-scaling` so the
//! default build stays small.
//!
//! ## Builder flow
//!
//! 1. Start with [`ImagePreprocessor::new`] to fix the output size.
//! 2. Chain [`with_pixel_layout`](ImagePreprocessor::with_pixel_layout),
//!    [`with_tensor_layout`](ImagePreprocessor::with_tensor_layout), and
//!    [`with_input_range`](ImagePreprocessor::with_input_range) to match
//!    the source data.
//! 3. Apply per-channel normalisation via
//!    [`with_mean`](ImagePreprocessor::with_mean) /
//!    [`with_std`](ImagePreprocessor::with_std), or use the
//!    [`with_imagenet_normalization`](ImagePreprocessor::with_imagenet_normalization)
//!    shortcut.
//! 4. Call [`process_u8_rgb`](ImagePreprocessor::process_u8_rgb) to get
//!    a flattened `Vec<f32>` ready to feed into ONNX.
//!
//! ## Example
//!
//! ```
//! use oximedia_ml::{ImagePreprocessor, TensorLayout};
//!
//! # fn main() -> oximedia_ml::MlResult<()> {
//! let preproc = ImagePreprocessor::new(224, 224)
//!     .with_tensor_layout(TensorLayout::Nchw)
//!     .with_imagenet_normalization();
//!
//! assert_eq!(preproc.batch_shape(), vec![1, 3, 224, 224]);
//!
//! let white = vec![255_u8; 224 * 224 * 3];
//! let tensor = preproc.process_u8_rgb(&white, 224, 224)?;
//! assert_eq!(tensor.len(), 3 * 224 * 224);
//! # Ok(())
//! # }
//! ```

use crate::error::{MlError, MlResult};

/// Pixel layout of the source image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PixelLayout {
    /// Red, Green, Blue channel order (default).
    Rgb,
    /// Blue, Green, Red channel order (OpenCV convention).
    Bgr,
}

/// Tensor memory layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorLayout {
    /// Batch, Channel, Height, Width.
    Nchw,
    /// Batch, Height, Width, Channel.
    Nhwc,
}

/// Scalar range of the source image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputRange {
    /// Values are in `[0, 255]` u8 range.
    U8,
    /// Values are already in `[0.0, 1.0]` f32 range.
    UnitFloat,
}

/// Builder for an image preprocessing pipeline.
///
/// See the [module-level docs][self] for the intended flow. An instance
/// is cheap to clone so a single pre-built preprocessor can be shared
/// across threads.
#[derive(Clone, Debug)]
pub struct ImagePreprocessor {
    target_width: u32,
    target_height: u32,
    pixel_layout: PixelLayout,
    tensor_layout: TensorLayout,
    input_range: InputRange,
    mean: [f32; 3],
    std: [f32; 3],
    swap_to_rgb: bool,
}

impl ImagePreprocessor {
    /// Create a new preprocessor for a given output size.
    #[must_use]
    pub fn new(target_width: u32, target_height: u32) -> Self {
        Self {
            target_width,
            target_height,
            pixel_layout: PixelLayout::Rgb,
            tensor_layout: TensorLayout::Nchw,
            input_range: InputRange::U8,
            mean: [0.0, 0.0, 0.0],
            std: [1.0, 1.0, 1.0],
            swap_to_rgb: false,
        }
    }

    /// Set the input pixel layout. Default is [`PixelLayout::Rgb`].
    #[must_use]
    pub fn with_pixel_layout(mut self, layout: PixelLayout) -> Self {
        self.pixel_layout = layout;
        self.swap_to_rgb = layout == PixelLayout::Bgr;
        self
    }

    /// Set the output tensor layout. Default is [`TensorLayout::Nchw`].
    #[must_use]
    pub fn with_tensor_layout(mut self, layout: TensorLayout) -> Self {
        self.tensor_layout = layout;
        self
    }

    /// Set the scalar range of the source image. Default is [`InputRange::U8`].
    #[must_use]
    pub fn with_input_range(mut self, range: InputRange) -> Self {
        self.input_range = range;
        self
    }

    /// Set the per-channel normalisation mean.
    #[must_use]
    pub fn with_mean(mut self, mean: [f32; 3]) -> Self {
        self.mean = mean;
        self
    }

    /// Set the per-channel normalisation std-dev.
    #[must_use]
    pub fn with_std(mut self, std: [f32; 3]) -> Self {
        self.std = std;
        self
    }

    /// Apply ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.
    #[must_use]
    pub fn with_imagenet_normalization(self) -> Self {
        self.with_mean([0.485, 0.456, 0.406])
            .with_std([0.229, 0.224, 0.225])
    }

    /// Target width in pixels.
    #[must_use]
    pub fn target_width(&self) -> u32 {
        self.target_width
    }

    /// Target height in pixels.
    #[must_use]
    pub fn target_height(&self) -> u32 {
        self.target_height
    }

    /// Process a raw u8 RGB(3-channel) buffer of size `src_w * src_h * 3`.
    ///
    /// The output is a flattened `Vec<f32>` containing a single image
    /// (no batch dimension). Call [`ImagePreprocessor::batch_shape`] to
    /// learn the logical shape.
    ///
    /// # Errors
    ///
    /// Returns [`MlError::Preprocess`] if:
    ///
    /// * `pixels.len() != src_w * src_h * 3`,
    /// * either dimension (source or target) is zero.
    pub fn process_u8_rgb(&self, pixels: &[u8], src_w: u32, src_h: u32) -> MlResult<Vec<f32>> {
        let expected = (src_w as usize) * (src_h as usize) * 3;
        if pixels.len() != expected {
            return Err(MlError::preprocess(format!(
                "expected {expected} bytes for {src_w}x{src_h} RGB, got {}",
                pixels.len()
            )));
        }
        if src_w == 0 || src_h == 0 {
            return Err(MlError::preprocess("source image has zero extent"));
        }
        if self.target_width == 0 || self.target_height == 0 {
            return Err(MlError::preprocess("target size has zero extent"));
        }

        let tw = self.target_width as usize;
        let th = self.target_height as usize;
        let mut out = vec![0.0_f32; tw * th * 3];

        let x_ratio = (src_w as f32) / (self.target_width as f32);
        let y_ratio = (src_h as f32) / (self.target_height as f32);

        for y in 0..th {
            let src_y = ((y as f32) * y_ratio) as usize;
            let src_y = src_y.min((src_h as usize).saturating_sub(1));
            for x in 0..tw {
                let src_x = ((x as f32) * x_ratio) as usize;
                let src_x = src_x.min((src_w as usize).saturating_sub(1));
                let src_idx = (src_y * (src_w as usize) + src_x) * 3;
                let (r_src, g_src, b_src) =
                    (pixels[src_idx], pixels[src_idx + 1], pixels[src_idx + 2]);
                let (r_raw, g_raw, b_raw) = if self.swap_to_rgb {
                    (b_src, g_src, r_src)
                } else {
                    (r_src, g_src, b_src)
                };

                let (r, g, b) = match self.input_range {
                    InputRange::U8 => (
                        (r_raw as f32) / 255.0,
                        (g_raw as f32) / 255.0,
                        (b_raw as f32) / 255.0,
                    ),
                    InputRange::UnitFloat => (r_raw as f32, g_raw as f32, b_raw as f32),
                };

                let r = (r - self.mean[0]) / self.std[0];
                let g = (g - self.mean[1]) / self.std[1];
                let b = (b - self.mean[2]) / self.std[2];

                match self.tensor_layout {
                    TensorLayout::Nhwc => {
                        let dst = (y * tw + x) * 3;
                        out[dst] = r;
                        out[dst + 1] = g;
                        out[dst + 2] = b;
                    }
                    TensorLayout::Nchw => {
                        let plane = tw * th;
                        let pixel = y * tw + x;
                        out[pixel] = r;
                        out[plane + pixel] = g;
                        out[(plane * 2) + pixel] = b;
                    }
                }
            }
        }

        Ok(out)
    }

    /// Return the logical shape of the output tensor with a leading
    /// batch dim of 1. Matches the flat buffer returned by
    /// [`ImagePreprocessor::process_u8_rgb`].
    #[must_use]
    pub fn batch_shape(&self) -> Vec<usize> {
        let tw = self.target_width as usize;
        let th = self.target_height as usize;
        match self.tensor_layout {
            TensorLayout::Nchw => vec![1, 3, th, tw],
            TensorLayout::Nhwc => vec![1, th, tw, 3],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_defaults() {
        let p = ImagePreprocessor::new(224, 224);
        assert_eq!(p.target_width(), 224);
        assert_eq!(p.target_height(), 224);
        assert_eq!(p.batch_shape(), vec![1, 3, 224, 224]);
    }

    #[test]
    fn nhwc_batch_shape() {
        let p = ImagePreprocessor::new(64, 32).with_tensor_layout(TensorLayout::Nhwc);
        assert_eq!(p.batch_shape(), vec![1, 32, 64, 3]);
    }

    #[test]
    fn mismatched_buffer_errors() {
        let p = ImagePreprocessor::new(4, 4);
        let pixels = vec![0u8; 10];
        let err = p.process_u8_rgb(&pixels, 2, 2).expect_err("must fail");
        assert!(matches!(err, MlError::Preprocess(_)));
    }

    #[test]
    fn zero_target_errors() {
        let p = ImagePreprocessor::new(0, 4);
        let pixels = vec![0u8; 4 * 4 * 3];
        let err = p.process_u8_rgb(&pixels, 4, 4).expect_err("must fail");
        assert!(matches!(err, MlError::Preprocess(_)));
    }

    #[test]
    fn imagenet_white_pixel_is_normalized() {
        // white pixel, 1×1 input, 1×1 target, ImageNet normalisation.
        let p = ImagePreprocessor::new(1, 1).with_imagenet_normalization();
        let pixels = vec![255u8, 255u8, 255u8];
        let out = p.process_u8_rgb(&pixels, 1, 1).expect("ok");
        assert_eq!(out.len(), 3);
        let expected_r = (1.0 - 0.485) / 0.229;
        let expected_g = (1.0 - 0.456) / 0.224;
        let expected_b = (1.0 - 0.406) / 0.225;
        assert!((out[0] - expected_r).abs() < 1e-5);
        assert!((out[1] - expected_g).abs() < 1e-5);
        assert!((out[2] - expected_b).abs() < 1e-5);
    }

    #[test]
    fn bgr_swaps_to_rgb() {
        let p = ImagePreprocessor::new(1, 1)
            .with_pixel_layout(PixelLayout::Bgr)
            .with_input_range(InputRange::U8);
        let pixels = vec![10u8, 20u8, 30u8];
        let out = p.process_u8_rgb(&pixels, 1, 1).expect("ok");
        // BGR→RGB swap => R=30/255, G=20/255, B=10/255 (no mean/std).
        assert!((out[0] - 30.0 / 255.0).abs() < 1e-5);
        assert!((out[1] - 20.0 / 255.0).abs() < 1e-5);
        assert!((out[2] - 10.0 / 255.0).abs() < 1e-5);
    }

    #[test]
    fn nchw_layout_plane_major() {
        let p = ImagePreprocessor::new(2, 1).with_input_range(InputRange::UnitFloat);
        // 2×1 image: two pixels with distinct channel values.
        // Pixel 0: (0.1, 0.2, 0.3); Pixel 1: (0.4, 0.5, 0.6)
        // raw u8 encoding: values 0..=255 via .process_u8_rgb expects u8 in [0..=255]; switch to UnitFloat below.
        // Hack: reinterpret u8 values as already-unit floats.
        let pixels = vec![25u8, 51, 76, 102, 128, 153];
        let out = p.process_u8_rgb(&pixels, 2, 1).expect("ok");
        // With InputRange::UnitFloat, r/g/b treated as f32 directly (u8 → f32 cast).
        assert_eq!(out.len(), 2 * 1 * 3);
        // Plane 0 = R: [25, 102]
        assert!((out[0] - 25.0).abs() < 1e-5);
        assert!((out[1] - 102.0).abs() < 1e-5);
        // Plane 1 = G: [51, 128]
        assert!((out[2] - 51.0).abs() < 1e-5);
        assert!((out[3] - 128.0).abs() < 1e-5);
        // Plane 2 = B: [76, 153]
        assert!((out[4] - 76.0).abs() < 1e-5);
        assert!((out[5] - 153.0).abs() < 1e-5);
    }
}
