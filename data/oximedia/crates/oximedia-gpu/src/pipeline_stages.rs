//! GPU-style image processing pipeline stage abstraction.
//!
//! This module defines the [`ImagePipelineStage`] trait and a set of built-in
//! concrete stages.  Stages are chained in an [`ImageComputePipeline`] that
//! validates pixel-format compatibility before execution.
//!
//! # Example
//! ```no_run
//! use oximedia_gpu::pipeline_stages::{
//!     ImageComputePipeline, GrayscaleStage, GaussianBlurStage, SobelStage,
//! };
//!
//! let mut pipeline = ImageComputePipeline::new(4, 4);
//! pipeline.add_stage(Box::new(GrayscaleStage)).expect("add grayscale");
//! pipeline.add_stage(Box::new(GaussianBlurStage { sigma: 1.0 })).expect("add blur");
//! pipeline.add_stage(Box::new(SobelStage)).expect("add sobel");
//!
//! let rgba: Vec<u8> = (0..16).flat_map(|i: u8| [i * 4, i * 4, i * 4, 255]).collect();
//! let result = pipeline.execute(&rgba).expect("execute");
//! assert_eq!(result.len(), 4 * 4); // Gray8 output
//! ```

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_lossless)]

// ---------------------------------------------------------------------------
// PixelFormat
// ---------------------------------------------------------------------------

/// Pixel-format descriptor used to type-check pipeline stage connections.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PixelFormat {
    /// 8-bit RGBA, 4 bytes per pixel.
    Rgba8,
    /// 8-bit RGB, 3 bytes per pixel.
    Rgb8,
    /// 8-bit grayscale, 1 byte per pixel.
    Gray8,
    /// Planar YCbCr 4:2:0.
    Yuv420,
    /// Planar YCbCr 4:2:2.
    Yuv422,
    /// 8-bit BGRA, 4 bytes per pixel.
    Bgra8,
    /// 32-bit float RGBA, 16 bytes per pixel.
    F32Rgba,
}

impl PixelFormat {
    /// Human-readable name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Rgba8 => "RGBA8",
            Self::Rgb8 => "RGB8",
            Self::Gray8 => "Gray8",
            Self::Yuv420 => "YUV420",
            Self::Yuv422 => "YUV422",
            Self::Bgra8 => "BGRA8",
            Self::F32Rgba => "F32RGBA",
        }
    }

    /// Bytes per pixel (for packed formats; approximate for planar).
    #[must_use]
    pub fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Rgba8 => 4,
            Self::Rgb8 => 3,
            Self::Gray8 => 1,
            Self::Yuv420 => 2, // approximate (1.5 but rounded up)
            Self::Yuv422 => 2,
            Self::Bgra8 => 4,
            Self::F32Rgba => 16,
        }
    }
}

impl std::fmt::Display for PixelFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

// ---------------------------------------------------------------------------
// ImagePipelineStage trait
// ---------------------------------------------------------------------------

/// A single processing stage in an [`ImageComputePipeline`].
///
/// Implementations must be `Send + Sync` so that pipelines can eventually
/// be executed in parallel contexts.
pub trait ImagePipelineStage: Send + Sync {
    /// Human-readable stage name.
    fn name(&self) -> &str;

    /// Transform `input` and return the resulting buffer.
    fn process(&self, input: &[u8], width: u32, height: u32) -> Vec<u8>;

    /// Pixel format this stage expects as input.
    fn input_format(&self) -> PixelFormat;

    /// Pixel format this stage produces as output.
    fn output_format(&self) -> PixelFormat;
}

// ---------------------------------------------------------------------------
// Built-in stages
// ---------------------------------------------------------------------------

// --- GrayscaleStage ---------------------------------------------------------

/// Convert RGBA to grayscale using the BT.601 luma formula.
///
/// Input: [`PixelFormat::Rgba8`], Output: [`PixelFormat::Gray8`].
pub struct GrayscaleStage;

impl ImagePipelineStage for GrayscaleStage {
    fn name(&self) -> &str {
        "Grayscale"
    }

    fn process(&self, input: &[u8], width: u32, height: u32) -> Vec<u8> {
        let n = width as usize * height as usize;
        let expected = n * 4;
        if input.len() != expected {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(n);
        // Process 4 pixels per iteration for auto-vectorisation.
        let chunk4 = n / 4;
        let rem = n % 4;
        for i in 0..chunk4 {
            let base = i * 16;
            for offset in [0_usize, 4, 8, 12] {
                let b = base + offset;
                let y = luma_bt601(input[b], input[b + 1], input[b + 2]);
                out.push(y);
            }
        }
        let rem_start = chunk4 * 16;
        for p in 0..rem {
            let b = rem_start + p * 4;
            out.push(luma_bt601(input[b], input[b + 1], input[b + 2]));
        }
        out
    }

    fn input_format(&self) -> PixelFormat {
        PixelFormat::Rgba8
    }
    fn output_format(&self) -> PixelFormat {
        PixelFormat::Gray8
    }
}

// --- GaussianBlurStage ------------------------------------------------------

/// Apply a separable Gaussian blur.
///
/// Supports both [`PixelFormat::Gray8`] and [`PixelFormat::Rgba8`] input
/// (configured at construction).  The `input_fmt` field selects the mode;
/// for Gray8 each byte is a single luma sample, for Rgba8 each channel is
/// blurred independently.
pub struct GaussianBlurStage {
    /// Standard deviation of the Gaussian kernel.
    pub sigma: f32,
}

impl ImagePipelineStage for GaussianBlurStage {
    fn name(&self) -> &str {
        "GaussianBlur"
    }

    fn process(&self, input: &[u8], width: u32, height: u32) -> Vec<u8> {
        let w = width as usize;
        let h = height as usize;
        let n = w * h;

        // Determine channel count from input length.
        let channels = if input.len() == n * 4 {
            4usize
        } else if input.len() == n {
            1
        } else {
            return Vec::new();
        };

        if self.sigma <= 0.0 {
            return input.to_vec();
        }

        let radius = (3.0 * self.sigma).ceil() as usize;
        let kernel = build_1d_gaussian(radius, self.sigma);

        // Separate into per-channel f32 planes, blur each, recombine.
        let mut out = vec![0u8; input.len()];
        for ch in 0..channels {
            let plane: Vec<f32> = (0..n).map(|i| input[i * channels + ch] as f32).collect();
            let blurred = gaussian_pass_2d(&plane, w, h, &kernel, radius);
            for i in 0..n {
                out[i * channels + ch] = blurred[i].round().clamp(0.0, 255.0) as u8;
            }
        }
        out
    }

    fn input_format(&self) -> PixelFormat {
        PixelFormat::Gray8
    }
    fn output_format(&self) -> PixelFormat {
        PixelFormat::Gray8
    }
}

// --- SobelStage -------------------------------------------------------------

/// Detect edges using the Sobel gradient magnitude operator.
///
/// Input: [`PixelFormat::Gray8`], Output: [`PixelFormat::Gray8`].
pub struct SobelStage;

impl ImagePipelineStage for SobelStage {
    fn name(&self) -> &str {
        "Sobel"
    }

    fn process(&self, input: &[u8], width: u32, height: u32) -> Vec<u8> {
        let w = width as usize;
        let h = height as usize;
        if input.len() != w * h {
            return Vec::new();
        }

        let gray_f: Vec<f32> = input.iter().map(|&b| b as f32).collect();
        let mut out = vec![0u8; w * h];

        for row in 1..h.saturating_sub(1) {
            let rb = row * w;
            for col in 1..w.saturating_sub(1) {
                let tl = gray_f[(row - 1) * w + col - 1];
                let tc = gray_f[(row - 1) * w + col];
                let tr = gray_f[(row - 1) * w + col + 1];
                let ml = gray_f[row * w + col - 1];
                let mr = gray_f[row * w + col + 1];
                let bl = gray_f[(row + 1) * w + col - 1];
                let bc = gray_f[(row + 1) * w + col];
                let br = gray_f[(row + 1) * w + col + 1];

                let gx = -tl + tr - 2.0 * ml + 2.0 * mr - bl + br;
                let gy = -tl - 2.0 * tc - tr + bl + 2.0 * bc + br;
                let mag = (gx * gx + gy * gy).sqrt();
                out[rb + col] = mag.round().clamp(0.0, 255.0) as u8;
            }
        }
        out
    }

    fn input_format(&self) -> PixelFormat {
        PixelFormat::Gray8
    }
    fn output_format(&self) -> PixelFormat {
        PixelFormat::Gray8
    }
}

// --- ThresholdStage ---------------------------------------------------------

/// Binary threshold: pixels ≥ threshold → 255, otherwise 0.
///
/// Input: [`PixelFormat::Gray8`], Output: [`PixelFormat::Gray8`].
pub struct ThresholdStage {
    /// Threshold value (inclusive).
    pub threshold: u8,
}

impl ImagePipelineStage for ThresholdStage {
    fn name(&self) -> &str {
        "Threshold"
    }

    fn process(&self, input: &[u8], width: u32, height: u32) -> Vec<u8> {
        if input.len() != width as usize * height as usize {
            return Vec::new();
        }
        input
            .iter()
            .map(|&px| if px >= self.threshold { 255 } else { 0 })
            .collect()
    }

    fn input_format(&self) -> PixelFormat {
        PixelFormat::Gray8
    }
    fn output_format(&self) -> PixelFormat {
        PixelFormat::Gray8
    }
}

// --- ColorConvertStage ------------------------------------------------------

/// Convert between pixel formats.
///
/// Currently supported conversions:
/// - `Rgba8 → Gray8` (BT.601 luma)
/// - `Rgba8 → Bgra8` (channel swap)
/// - `Bgra8 → Rgba8` (channel swap)
/// - `Gray8 → Rgba8` (broadcast luma, alpha = 255)
/// - Identity (same format → passthrough)
///
/// All other combinations return an empty vector.
pub struct ColorConvertStage {
    /// Source pixel format.
    pub from: PixelFormat,
    /// Target pixel format.
    pub to: PixelFormat,
}

impl ImagePipelineStage for ColorConvertStage {
    fn name(&self) -> &str {
        "ColorConvert"
    }

    fn process(&self, input: &[u8], width: u32, height: u32) -> Vec<u8> {
        let n = width as usize * height as usize;

        if self.from == self.to {
            return input.to_vec();
        }

        match (self.from, self.to) {
            (PixelFormat::Rgba8, PixelFormat::Gray8) => {
                if input.len() != n * 4 {
                    return Vec::new();
                }
                (0..n)
                    .map(|i| {
                        let b = i * 4;
                        luma_bt601(input[b], input[b + 1], input[b + 2])
                    })
                    .collect()
            }
            (PixelFormat::Gray8, PixelFormat::Rgba8) => {
                if input.len() != n {
                    return Vec::new();
                }
                input.iter().flat_map(|&px| [px, px, px, 255]).collect()
            }
            (PixelFormat::Rgba8, PixelFormat::Bgra8) | (PixelFormat::Bgra8, PixelFormat::Rgba8) => {
                if input.len() != n * 4 {
                    return Vec::new();
                }
                let mut out = input.to_vec();
                for i in 0..n {
                    let b = i * 4;
                    out.swap(b, b + 2); // swap R and B
                }
                out
            }
            _ => Vec::new(), // unsupported conversion
        }
    }

    fn input_format(&self) -> PixelFormat {
        self.from
    }
    fn output_format(&self) -> PixelFormat {
        self.to
    }
}

// --- OverlayStage -----------------------------------------------------------

/// Composite a pre-loaded overlay image over the input frame.
///
/// The overlay is blended using `alpha` as a uniform opacity multiplier
/// (0.0 = invisible, 1.0 = opaque overlay) in addition to the overlay's
/// own alpha channel.
///
/// Input: [`PixelFormat::Rgba8`], Output: [`PixelFormat::Rgba8`].
pub struct OverlayStage {
    /// Overlay image data (RGBA8, same dimensions as the pipeline).
    pub overlay: Vec<u8>,
    /// Uniform opacity for the overlay (0.0 – 1.0).
    pub alpha: f32,
}

impl ImagePipelineStage for OverlayStage {
    fn name(&self) -> &str {
        "Overlay"
    }

    fn process(&self, input: &[u8], width: u32, height: u32) -> Vec<u8> {
        let n = width as usize * height as usize;
        let expected = n * 4;
        if input.len() != expected || self.overlay.len() != expected {
            return input.to_vec();
        }

        let alpha_clamp = self.alpha.clamp(0.0, 1.0);
        let mut out = vec![0u8; expected];

        for i in 0..n {
            let b = i * 4;
            let bg_r = input[b] as f32;
            let bg_g = input[b + 1] as f32;
            let bg_b = input[b + 2] as f32;
            let bg_a = input[b + 3] as f32 / 255.0;

            let ov_r = self.overlay[b] as f32;
            let ov_g = self.overlay[b + 1] as f32;
            let ov_b = self.overlay[b + 2] as f32;
            let ov_a = (self.overlay[b + 3] as f32 / 255.0) * alpha_clamp;

            // Porter-Duff "over".
            let out_a = ov_a + bg_a * (1.0 - ov_a);
            if out_a <= 0.0 {
                continue;
            }
            let inv = 1.0 / out_a;
            out[b] = ((ov_r * ov_a + bg_r * bg_a * (1.0 - ov_a)) * inv)
                .round()
                .clamp(0.0, 255.0) as u8;
            out[b + 1] = ((ov_g * ov_a + bg_g * bg_a * (1.0 - ov_a)) * inv)
                .round()
                .clamp(0.0, 255.0) as u8;
            out[b + 2] = ((ov_b * ov_a + bg_b * bg_a * (1.0 - ov_a)) * inv)
                .round()
                .clamp(0.0, 255.0) as u8;
            out[b + 3] = (out_a * 255.0).round().clamp(0.0, 255.0) as u8;
        }
        out
    }

    fn input_format(&self) -> PixelFormat {
        PixelFormat::Rgba8
    }
    fn output_format(&self) -> PixelFormat {
        PixelFormat::Rgba8
    }
}

// ---------------------------------------------------------------------------
// ImageComputePipeline
// ---------------------------------------------------------------------------

/// A linear sequence of [`ImagePipelineStage`]s.
///
/// Before execution, the pipeline validates that each stage's output format
/// matches the next stage's input format.
pub struct ImageComputePipeline {
    stages: Vec<Box<dyn ImagePipelineStage>>,
    /// Width in pixels for all frames processed by this pipeline.
    pub width: u32,
    /// Height in pixels for all frames processed by this pipeline.
    pub height: u32,
}

impl ImageComputePipeline {
    /// Create an empty pipeline for frames of size `width × height`.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            stages: Vec::new(),
            width,
            height,
        }
    }

    /// Append a stage to the pipeline.
    ///
    /// Validates that the new stage's input format matches the previous
    /// stage's output format.  If the pipeline is empty any format is
    /// accepted.
    ///
    /// # Errors
    ///
    /// Returns an error string describing the format mismatch.
    pub fn add_stage(&mut self, stage: Box<dyn ImagePipelineStage>) -> Result<(), String> {
        if let Some(prev) = self.stages.last() {
            let prev_out = prev.output_format();
            let next_in = stage.input_format();
            if prev_out != next_in {
                return Err(format!(
                    "Format mismatch between '{}' (output: {}) and '{}' (input: {})",
                    prev.name(),
                    prev_out,
                    stage.name(),
                    next_in,
                ));
            }
        }
        self.stages.push(stage);
        Ok(())
    }

    /// Run all stages in sequence and return the final output.
    ///
    /// The `input` slice is passed to the first stage.  Each subsequent
    /// stage receives the output of the previous stage.
    ///
    /// # Errors
    ///
    /// Returns an error string if any stage produces an empty output
    /// (which indicates a dimension mismatch at runtime).
    pub fn execute(&self, input: &[u8]) -> Result<Vec<u8>, String> {
        if self.stages.is_empty() {
            return Ok(input.to_vec());
        }

        let mut current: Vec<u8> = input.to_vec();
        for stage in &self.stages {
            let next = stage.process(&current, self.width, self.height);
            if next.is_empty() {
                return Err(format!(
                    "Stage '{}' returned empty output (possible dimension mismatch)",
                    stage.name()
                ));
            }
            current = next;
        }
        Ok(current)
    }

    /// Number of stages in this pipeline.
    #[must_use]
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }

    /// Validate format compatibility for all consecutive stage pairs.
    ///
    /// # Errors
    ///
    /// Returns the first format mismatch found, if any.
    pub fn validate(&self) -> Result<(), String> {
        for pair in self.stages.windows(2) {
            let a = &pair[0];
            let b = &pair[1];
            if a.output_format() != b.input_format() {
                return Err(format!(
                    "Stage '{}' outputs {} but '{}' expects {}",
                    a.name(),
                    a.output_format(),
                    b.name(),
                    b.input_format(),
                ));
            }
        }
        Ok(())
    }

    /// Names of all stages, in execution order.
    #[must_use]
    pub fn stage_names(&self) -> Vec<&str> {
        self.stages.iter().map(|s| s.name()).collect()
    }
}

impl std::fmt::Debug for ImageComputePipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImageComputePipeline")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("stage_count", &self.stages.len())
            .field("stages", &self.stage_names())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

#[inline(always)]
fn luma_bt601(r: u8, g: u8, b: u8) -> u8 {
    let y = 0.299_f32 * r as f32 + 0.587_f32 * g as f32 + 0.114_f32 * b as f32;
    y.round().clamp(0.0, 255.0) as u8
}

fn build_1d_gaussian(radius: usize, sigma: f32) -> Vec<f32> {
    let len = 2 * radius + 1;
    let two_sigma_sq = 2.0 * sigma * sigma;
    let mut k: Vec<f32> = (0..len)
        .map(|i| {
            let x = (i as isize - radius as isize) as f32;
            (-x * x / two_sigma_sq).exp()
        })
        .collect();
    let sum: f32 = k.iter().sum();
    if sum > 0.0 {
        k.iter_mut().for_each(|v| *v /= sum);
    }
    k
}

fn gaussian_pass_2d(plane: &[f32], w: usize, h: usize, kernel: &[f32], radius: usize) -> Vec<f32> {
    // Horizontal pass.
    let mut tmp = vec![0.0_f32; w * h];
    for row in 0..h {
        let rs = row * w;
        for col in 0..w {
            let (mut acc, mut wsum) = (0.0_f32, 0.0_f32);
            for (ki, &kv) in kernel.iter().enumerate() {
                let src_col = col as isize + ki as isize - radius as isize;
                if src_col >= 0 && src_col < w as isize {
                    acc += plane[rs + src_col as usize] * kv;
                    wsum += kv;
                }
            }
            tmp[rs + col] = if wsum > 0.0 { acc / wsum } else { 0.0 };
        }
    }
    // Vertical pass.
    let mut out = vec![0.0_f32; w * h];
    for col in 0..w {
        for row in 0..h {
            let (mut acc, mut wsum) = (0.0_f32, 0.0_f32);
            for (ki, &kv) in kernel.iter().enumerate() {
                let src_row = row as isize + ki as isize - radius as isize;
                if src_row >= 0 && src_row < h as isize {
                    acc += tmp[src_row as usize * w + col] * kv;
                    wsum += kv;
                }
            }
            out[row * w + col] = if wsum > 0.0 { acc / wsum } else { 0.0 };
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn rgba_frame(w: u32, h: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
        let n = w as usize * h as usize;
        (0..n).flat_map(|_| [r, g, b, 255]).collect()
    }

    fn gray_frame(w: u32, h: u32, v: u8) -> Vec<u8> {
        vec![v; w as usize * h as usize]
    }

    // --- PixelFormat ---

    #[test]
    fn test_pixel_format_display() {
        assert_eq!(PixelFormat::Rgba8.to_string(), "RGBA8");
        assert_eq!(PixelFormat::Gray8.to_string(), "Gray8");
    }

    // --- GrayscaleStage ---

    #[test]
    fn test_grayscale_white() {
        let stage = GrayscaleStage;
        let input = rgba_frame(4, 4, 255, 255, 255);
        let out = stage.process(&input, 4, 4);
        assert_eq!(out.len(), 16);
        assert!(
            out.iter().all(|&v| v > 250),
            "white should map to ~255 gray"
        );
    }

    #[test]
    fn test_grayscale_format() {
        let stage = GrayscaleStage;
        assert_eq!(stage.input_format(), PixelFormat::Rgba8);
        assert_eq!(stage.output_format(), PixelFormat::Gray8);
    }

    // --- GaussianBlurStage ---

    #[test]
    fn test_gaussian_blur_constant_gray() {
        let stage = GaussianBlurStage { sigma: 1.5 };
        let input = gray_frame(8, 8, 100);
        let out = stage.process(&input, 8, 8);
        assert_eq!(out.len(), 64);
        for &v in &out {
            assert!(
                (v as i32 - 100).unsigned_abs() <= 2,
                "constant image should remain ~100, got {v}"
            );
        }
    }

    #[test]
    fn test_gaussian_blur_wrong_size() {
        let stage = GaussianBlurStage { sigma: 1.0 };
        let out = stage.process(&[0u8; 3], 4, 4);
        assert!(
            out.is_empty(),
            "wrong-size input should produce empty output"
        );
    }

    // --- SobelStage ---

    #[test]
    fn test_sobel_flat_is_zero() {
        let stage = SobelStage;
        let input = gray_frame(8, 8, 128);
        let out = stage.process(&input, 8, 8);
        for row in 1..7_usize {
            for col in 1..7_usize {
                assert_eq!(out[row * 8 + col], 0, "flat image interior should be 0");
            }
        }
    }

    #[test]
    fn test_sobel_output_format() {
        let stage = SobelStage;
        assert_eq!(stage.input_format(), PixelFormat::Gray8);
        assert_eq!(stage.output_format(), PixelFormat::Gray8);
    }

    // --- ThresholdStage ---

    #[test]
    fn test_threshold_binary() {
        let stage = ThresholdStage { threshold: 128 };
        let input = vec![100u8, 128, 200, 50, 128, 255];
        let out = stage.process(&input, 6, 1);
        assert_eq!(out, vec![0, 255, 255, 0, 255, 255]);
    }

    // --- ColorConvertStage ---

    #[test]
    fn test_color_convert_identity() {
        let stage = ColorConvertStage {
            from: PixelFormat::Rgba8,
            to: PixelFormat::Rgba8,
        };
        let input = rgba_frame(2, 2, 10, 20, 30);
        let out = stage.process(&input, 2, 2);
        assert_eq!(out, input);
    }

    #[test]
    fn test_color_convert_rgba_to_gray() {
        let stage = ColorConvertStage {
            from: PixelFormat::Rgba8,
            to: PixelFormat::Gray8,
        };
        let input = rgba_frame(2, 2, 255, 255, 255);
        let out = stage.process(&input, 2, 2);
        assert_eq!(out.len(), 4);
        assert!(out.iter().all(|&v| v > 250));
    }

    #[test]
    fn test_color_convert_rgba_to_bgra_swap() {
        let stage = ColorConvertStage {
            from: PixelFormat::Rgba8,
            to: PixelFormat::Bgra8,
        };
        let input = vec![255u8, 0, 0, 255]; // red RGBA
        let out = stage.process(&input, 1, 1);
        assert_eq!(&out[0..4], &[0u8, 0, 255, 255]); // should be blue BGRA
    }

    // --- OverlayStage ---

    #[test]
    fn test_overlay_transparent_overlay() {
        let bg = rgba_frame(2, 2, 0, 0, 255);
        let overlay_data: Vec<u8> = (0..4).flat_map(|_| [255u8, 0, 0, 0u8]).collect(); // fully transparent red
        let stage = OverlayStage {
            overlay: overlay_data,
            alpha: 1.0,
        };
        let out = stage.process(&bg, 2, 2);
        // Fully transparent overlay → output ≈ bg.
        assert_eq!(&out[0..3], &[0u8, 0, 255]);
    }

    // --- ImageComputePipeline ---

    #[test]
    fn test_pipeline_empty_passthrough() {
        let pipeline = ImageComputePipeline::new(4, 4);
        let input = gray_frame(4, 4, 77);
        let out = pipeline.execute(&input).expect("execute");
        assert_eq!(out, input);
    }

    #[test]
    fn test_pipeline_add_stage_format_mismatch() {
        let mut pipeline = ImageComputePipeline::new(4, 4);
        pipeline
            .add_stage(Box::new(GrayscaleStage))
            .expect("add grayscale");
        // GrayscaleStage outputs Gray8, but GrayscaleStage itself accepts Rgba8 — mismatch.
        let result = pipeline.add_stage(Box::new(GrayscaleStage));
        assert!(result.is_err(), "should detect format mismatch");
    }

    #[test]
    fn test_pipeline_validate_ok() {
        let mut pipeline = ImageComputePipeline::new(4, 4);
        pipeline
            .add_stage(Box::new(GrayscaleStage))
            .expect("grayscale");
        pipeline.add_stage(Box::new(SobelStage)).expect("sobel");
        assert!(pipeline.validate().is_ok());
    }

    #[test]
    fn test_pipeline_stage_count() {
        let mut pipeline = ImageComputePipeline::new(4, 4);
        assert_eq!(pipeline.stage_count(), 0);
        pipeline.add_stage(Box::new(GrayscaleStage)).expect("add");
        assert_eq!(pipeline.stage_count(), 1);
        pipeline.add_stage(Box::new(SobelStage)).expect("add");
        assert_eq!(pipeline.stage_count(), 2);
    }

    #[test]
    fn test_pipeline_full_rgba_to_binary() {
        // RGBA → Gray8 → Threshold
        let mut pipeline = ImageComputePipeline::new(4, 4);
        pipeline.add_stage(Box::new(GrayscaleStage)).expect("gray");
        pipeline
            .add_stage(Box::new(ThresholdStage { threshold: 128 }))
            .expect("thresh");

        let input = rgba_frame(4, 4, 200, 200, 200); // bright grey → luma ≈ 200
        let out = pipeline.execute(&input).expect("execute");
        assert_eq!(out.len(), 16);
        assert!(
            out.iter().all(|&v| v == 255),
            "all pixels should be above threshold"
        );
    }
}
