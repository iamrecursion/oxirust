//! Focus stacking: combining images at different focus distances into an all-in-focus result.
//!
//! Focus stacking is used in macro photography, microscopy, and landscape photography
//! where depth of field is limited. Multiple images are captured at different focus
//! distances and blended using focus quality measures.
//!
//! # Algorithm
//!
//! 1. For each input image, compute a per-pixel focus measure (Laplacian energy)
//! 2. Optionally smooth the focus maps with a Gaussian to reduce noise
//! 3. For each pixel, select the source image with the highest focus measure
//! 4. Blend using weighted average based on focus measures for smooth transitions
//!
//! # Example
//!
//! ```rust
//! use oximedia_image::focus_stack::{FocusStacker, FocusMethod, BlendMode};
//!
//! let stacker = FocusStacker::new(FocusMethod::Laplacian, BlendMode::Weighted);
//! // Images are f32 grayscale, row-major, all same dimensions
//! let img1 = vec![0.5_f32; 16]; // 4x4 image focused on foreground
//! let img2 = vec![0.7_f32; 16]; // 4x4 image focused on background
//! let images: Vec<&[f32]> = vec![&img1, &img2];
//! let result = stacker.stack(&images, 4, 4).expect("stacking should succeed");
//! assert_eq!(result.len(), 16);
//! ```

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]

use crate::error::{ImageError, ImageResult};

/// Focus measure algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FocusMethod {
    /// Laplacian energy: sum of squared second derivatives.
    /// Good general-purpose focus measure with edge sensitivity.
    Laplacian,
    /// Gradient magnitude: sum of squared first derivatives (Sobel-like).
    /// More sensitive to edges, less to noise.
    GradientMagnitude,
    /// Local variance: variance within a window.
    /// Good for textured regions.
    LocalVariance,
}

/// Blending mode for combining focused regions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendMode {
    /// Winner-take-all: pixel from image with highest focus measure.
    /// Sharp but can produce seam artifacts.
    Maximum,
    /// Weighted average based on focus measures.
    /// Smoother transitions, reduced artifacts.
    Weighted,
}

/// Focus stacker configuration and executor.
#[derive(Debug, Clone)]
pub struct FocusStacker {
    /// Focus measurement method.
    method: FocusMethod,
    /// Blending mode.
    blend_mode: BlendMode,
    /// Gaussian smoothing radius for focus maps (0 = no smoothing).
    smooth_radius: usize,
}

impl FocusStacker {
    /// Creates a new focus stacker with the given method and blend mode.
    #[must_use]
    pub fn new(method: FocusMethod, blend_mode: BlendMode) -> Self {
        Self {
            method,
            blend_mode,
            smooth_radius: 2,
        }
    }

    /// Sets the Gaussian smoothing radius for focus maps.
    ///
    /// Larger radius produces smoother transitions but may lose fine detail.
    /// Set to 0 to disable smoothing.
    #[must_use]
    pub fn with_smooth_radius(mut self, radius: usize) -> Self {
        self.smooth_radius = radius;
        self
    }

    /// Stacks multiple images into a single all-in-focus result.
    ///
    /// All images must be single-channel f32, row-major, with the same dimensions.
    ///
    /// # Errors
    ///
    /// Returns an error if fewer than 2 images are provided or dimensions are invalid.
    pub fn stack(&self, images: &[&[f32]], width: usize, height: usize) -> ImageResult<Vec<f32>> {
        let n = width * height;
        if images.len() < 2 {
            return Err(ImageError::invalid_format(
                "Focus stacking requires at least 2 images",
            ));
        }
        if n == 0 {
            return Err(ImageError::InvalidDimensions(width as u32, height as u32));
        }
        for (i, img) in images.iter().enumerate() {
            if img.len() < n {
                return Err(ImageError::invalid_format(format!(
                    "Image {i} has {} pixels, expected {n}",
                    img.len()
                )));
            }
        }

        // Compute focus maps
        let mut focus_maps: Vec<Vec<f32>> = Vec::with_capacity(images.len());
        for img in images {
            let fm = self.compute_focus_map(img, width, height);
            let smoothed = if self.smooth_radius > 0 {
                gaussian_smooth(&fm, width, height, self.smooth_radius)
            } else {
                fm
            };
            focus_maps.push(smoothed);
        }

        // Blend
        let result = match self.blend_mode {
            BlendMode::Maximum => self.blend_maximum(images, &focus_maps, n),
            BlendMode::Weighted => self.blend_weighted(images, &focus_maps, n),
        };

        Ok(result)
    }

    /// Stacks multiple RGB images (3-channel interleaved f32).
    ///
    /// Each image has `width * height * 3` elements in RGB order.
    ///
    /// # Errors
    ///
    /// Returns an error if fewer than 2 images are provided or dimensions are invalid.
    pub fn stack_rgb(
        &self,
        images: &[&[f32]],
        width: usize,
        height: usize,
    ) -> ImageResult<Vec<f32>> {
        let n = width * height;
        let n3 = n * 3;
        if images.len() < 2 {
            return Err(ImageError::invalid_format(
                "Focus stacking requires at least 2 images",
            ));
        }
        if n == 0 {
            return Err(ImageError::InvalidDimensions(width as u32, height as u32));
        }
        for (i, img) in images.iter().enumerate() {
            if img.len() < n3 {
                return Err(ImageError::invalid_format(format!(
                    "RGB image {i} has {} elements, expected {n3}",
                    img.len()
                )));
            }
        }

        // Compute focus maps from luminance
        let luminances: Vec<Vec<f32>> = images.iter().map(|img| rgb_to_luminance(img, n)).collect();

        let mut focus_maps: Vec<Vec<f32>> = Vec::with_capacity(images.len());
        for lum in &luminances {
            let fm = self.compute_focus_map(lum, width, height);
            let smoothed = if self.smooth_radius > 0 {
                gaussian_smooth(&fm, width, height, self.smooth_radius)
            } else {
                fm
            };
            focus_maps.push(smoothed);
        }

        // Blend RGB channels
        let mut result = vec![0.0_f32; n3];
        match self.blend_mode {
            BlendMode::Maximum => {
                for px in 0..n {
                    let best_idx = focus_maps
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            a[px]
                                .partial_cmp(&b[px])
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    result[px * 3] = images[best_idx][px * 3];
                    result[px * 3 + 1] = images[best_idx][px * 3 + 1];
                    result[px * 3 + 2] = images[best_idx][px * 3 + 2];
                }
            }
            BlendMode::Weighted => {
                for px in 0..n {
                    let total_weight: f32 = focus_maps.iter().map(|fm| fm[px]).sum();
                    if total_weight < f32::EPSILON {
                        // Equal weight fallback
                        let inv = 1.0 / images.len() as f32;
                        for img in images {
                            result[px * 3] += img[px * 3] * inv;
                            result[px * 3 + 1] += img[px * 3 + 1] * inv;
                            result[px * 3 + 2] += img[px * 3 + 2] * inv;
                        }
                    } else {
                        let inv = 1.0 / total_weight;
                        for (fi, img) in images.iter().enumerate() {
                            let w = focus_maps[fi][px] * inv;
                            result[px * 3] += img[px * 3] * w;
                            result[px * 3 + 1] += img[px * 3 + 1] * w;
                            result[px * 3 + 2] += img[px * 3 + 2] * w;
                        }
                    }
                }
            }
        }

        Ok(result)
    }

    /// Computes per-pixel focus measure for a single-channel image.
    fn compute_focus_map(&self, image: &[f32], width: usize, height: usize) -> Vec<f32> {
        match self.method {
            FocusMethod::Laplacian => laplacian_focus_map(image, width, height),
            FocusMethod::GradientMagnitude => gradient_focus_map(image, width, height),
            FocusMethod::LocalVariance => variance_focus_map(image, width, height),
        }
    }

    /// Winner-take-all blending.
    fn blend_maximum(&self, images: &[&[f32]], focus_maps: &[Vec<f32>], n: usize) -> Vec<f32> {
        let mut result = vec![0.0_f32; n];
        for px in 0..n {
            let best_idx = focus_maps
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    a[px]
                        .partial_cmp(&b[px])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap_or(0);
            result[px] = images[best_idx][px];
        }
        result
    }

    /// Weighted average blending based on focus measures.
    fn blend_weighted(&self, images: &[&[f32]], focus_maps: &[Vec<f32>], n: usize) -> Vec<f32> {
        let mut result = vec![0.0_f32; n];
        for px in 0..n {
            let total_weight: f32 = focus_maps.iter().map(|fm| fm[px]).sum();
            if total_weight < f32::EPSILON {
                // Equal weight fallback
                let inv = 1.0 / images.len() as f32;
                for img in images {
                    result[px] += img[px] * inv;
                }
            } else {
                let inv = 1.0 / total_weight;
                for (fi, img) in images.iter().enumerate() {
                    result[px] += img[px] * (focus_maps[fi][px] * inv);
                }
            }
        }
        result
    }
}

/// Computes Laplacian focus map: |L(x,y)|^2 using discrete Laplacian kernel.
///
/// Laplacian kernel: [0, 1, 0; 1, -4, 1; 0, 1, 0]
fn laplacian_focus_map(image: &[f32], width: usize, height: usize) -> Vec<f32> {
    let n = width * height;
    let mut focus = vec![0.0_f32; n];
    let w = width;

    for y in 1..height.saturating_sub(1) {
        for x in 1..width.saturating_sub(1) {
            let idx = y * w + x;
            let center = image[idx];
            let up = image[(y - 1) * w + x];
            let down = image[(y + 1) * w + x];
            let left = image[y * w + (x - 1)];
            let right = image[y * w + (x + 1)];

            let laplacian = up + down + left + right - 4.0 * center;
            focus[idx] = laplacian * laplacian;
        }
    }

    focus
}

/// Computes gradient magnitude focus map: |Gx|^2 + |Gy|^2.
fn gradient_focus_map(image: &[f32], width: usize, height: usize) -> Vec<f32> {
    let n = width * height;
    let mut focus = vec![0.0_f32; n];
    let w = width;

    for y in 1..height.saturating_sub(1) {
        for x in 1..width.saturating_sub(1) {
            let idx = y * w + x;
            let gx = image[y * w + (x + 1)] - image[y * w + (x - 1)];
            let gy = image[(y + 1) * w + x] - image[(y - 1) * w + x];
            focus[idx] = gx * gx + gy * gy;
        }
    }

    focus
}

/// Computes local variance focus map using a 3x3 window.
fn variance_focus_map(image: &[f32], width: usize, height: usize) -> Vec<f32> {
    let n = width * height;
    let mut focus = vec![0.0_f32; n];
    let w = width;

    for y in 1..height.saturating_sub(1) {
        for x in 1..width.saturating_sub(1) {
            let mut sum = 0.0_f32;
            let mut sum_sq = 0.0_f32;
            let mut count = 0.0_f32;

            for dy in 0..3usize {
                for dx in 0..3usize {
                    let py = y + dy - 1;
                    let px = x + dx - 1;
                    let val = image[py * w + px];
                    sum += val;
                    sum_sq += val * val;
                    count += 1.0;
                }
            }

            let mean = sum / count;
            let variance = (sum_sq / count) - (mean * mean);
            focus[y * w + x] = variance.max(0.0);
        }
    }

    focus
}

/// Simple box-blur Gaussian approximation for smoothing focus maps.
fn gaussian_smooth(map: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
    let n = width * height;
    let mut temp = vec![0.0_f32; n];
    let mut result = vec![0.0_f32; n];
    let r = radius as isize;
    let kernel_size = (2 * r + 1) as f32;

    // Horizontal pass
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0_f32;
            let mut count = 0.0_f32;
            for dx in -r..=r {
                let sx = x as isize + dx;
                if sx >= 0 && (sx as usize) < width {
                    sum += map[y * width + sx as usize];
                    count += 1.0;
                }
            }
            temp[y * width + x] = if count > 0.0 { sum / count } else { 0.0 };
        }
    }

    // Vertical pass
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0_f32;
            let mut count = 0.0_f32;
            for dy in -r..=r {
                let sy = y as isize + dy;
                if sy >= 0 && (sy as usize) < height {
                    sum += temp[sy as usize * width + x];
                    count += 1.0;
                }
            }
            result[y * width + x] = if count > 0.0 { sum / count } else { 0.0 };
        }
    }

    let _ = kernel_size; // used conceptually
    result
}

/// Converts interleaved RGB to luminance using BT.709 coefficients.
fn rgb_to_luminance(rgb: &[f32], pixel_count: usize) -> Vec<f32> {
    let mut lum = vec![0.0_f32; pixel_count];
    for i in 0..pixel_count {
        let r = rgb.get(i * 3).copied().unwrap_or(0.0);
        let g = rgb.get(i * 3 + 1).copied().unwrap_or(0.0);
        let b = rgb.get(i * 3 + 2).copied().unwrap_or(0.0);
        lum[i] = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    }
    lum
}

/// Compute a per-pixel focus quality score for a single image.
///
/// Useful for evaluating which regions are in focus without performing a full stack.
#[must_use]
pub fn compute_focus_quality(
    image: &[f32],
    width: usize,
    height: usize,
    method: FocusMethod,
) -> Vec<f32> {
    match method {
        FocusMethod::Laplacian => laplacian_focus_map(image, width, height),
        FocusMethod::GradientMagnitude => gradient_focus_map(image, width, height),
        FocusMethod::LocalVariance => variance_focus_map(image, width, height),
    }
}

/// Compute overall focus score (mean focus measure) for an image.
///
/// Higher values indicate a more in-focus image overall.
#[must_use]
pub fn overall_focus_score(image: &[f32], width: usize, height: usize) -> f32 {
    let fm = laplacian_focus_map(image, width, height);
    let n = fm.len();
    if n == 0 {
        return 0.0;
    }
    let sum: f32 = fm.iter().sum();
    sum / n as f32
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_focused_foreground(width: usize, height: usize) -> Vec<f32> {
        // Sharp edges in top half, blurry bottom half
        let mut img = vec![0.5_f32; width * height];
        for y in 0..height / 2 {
            for x in 0..width {
                img[y * width + x] = if (x + y) % 3 == 0 { 1.0 } else { 0.0 };
            }
        }
        img
    }

    fn make_focused_background(width: usize, height: usize) -> Vec<f32> {
        // Blurry top half, sharp edges in bottom half
        let mut img = vec![0.5_f32; width * height];
        for y in height / 2..height {
            for x in 0..width {
                img[y * width + x] = if (x + y) % 3 == 0 { 1.0 } else { 0.0 };
            }
        }
        img
    }

    #[test]
    fn test_focus_stack_basic() {
        let w = 8;
        let h = 8;
        let img1 = make_focused_foreground(w, h);
        let img2 = make_focused_background(w, h);
        let images: Vec<&[f32]> = vec![&img1, &img2];

        let stacker = FocusStacker::new(FocusMethod::Laplacian, BlendMode::Weighted);
        let result = stacker.stack(&images, w, h).expect("stack should succeed");
        assert_eq!(result.len(), w * h);
        assert!(result.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_focus_stack_maximum_blend() {
        let w = 8;
        let h = 8;
        let img1 = make_focused_foreground(w, h);
        let img2 = make_focused_background(w, h);
        let images: Vec<&[f32]> = vec![&img1, &img2];

        let stacker = FocusStacker::new(FocusMethod::Laplacian, BlendMode::Maximum);
        let result = stacker.stack(&images, w, h).expect("stack should succeed");
        assert_eq!(result.len(), w * h);
    }

    #[test]
    fn test_focus_stack_gradient_method() {
        let w = 8;
        let h = 8;
        let img1 = make_focused_foreground(w, h);
        let img2 = make_focused_background(w, h);
        let images: Vec<&[f32]> = vec![&img1, &img2];

        let stacker = FocusStacker::new(FocusMethod::GradientMagnitude, BlendMode::Weighted);
        let result = stacker.stack(&images, w, h).expect("stack should succeed");
        assert_eq!(result.len(), w * h);
    }

    #[test]
    fn test_focus_stack_variance_method() {
        let w = 8;
        let h = 8;
        let img1 = make_focused_foreground(w, h);
        let img2 = make_focused_background(w, h);
        let images: Vec<&[f32]> = vec![&img1, &img2];

        let stacker = FocusStacker::new(FocusMethod::LocalVariance, BlendMode::Weighted);
        let result = stacker.stack(&images, w, h).expect("stack should succeed");
        assert_eq!(result.len(), w * h);
    }

    #[test]
    fn test_focus_stack_no_smoothing() {
        let w = 8;
        let h = 8;
        let img1 = make_focused_foreground(w, h);
        let img2 = make_focused_background(w, h);
        let images: Vec<&[f32]> = vec![&img1, &img2];

        let stacker =
            FocusStacker::new(FocusMethod::Laplacian, BlendMode::Weighted).with_smooth_radius(0);
        let result = stacker.stack(&images, w, h).expect("stack should succeed");
        assert_eq!(result.len(), w * h);
    }

    #[test]
    fn test_focus_stack_three_images() {
        let w = 8;
        let h = 8;
        let img1 = vec![0.3_f32; w * h];
        let img2 = make_focused_foreground(w, h);
        let img3 = make_focused_background(w, h);
        let images: Vec<&[f32]> = vec![&img1, &img2, &img3];

        let stacker = FocusStacker::new(FocusMethod::Laplacian, BlendMode::Weighted);
        let result = stacker.stack(&images, w, h).expect("stack should succeed");
        assert_eq!(result.len(), w * h);
    }

    #[test]
    fn test_focus_stack_too_few_images() {
        let img1 = vec![0.5_f32; 16];
        let images: Vec<&[f32]> = vec![&img1];

        let stacker = FocusStacker::new(FocusMethod::Laplacian, BlendMode::Weighted);
        assert!(stacker.stack(&images, 4, 4).is_err());
    }

    #[test]
    fn test_focus_stack_zero_dimensions() {
        let img1 = vec![0.5_f32; 16];
        let img2 = vec![0.5_f32; 16];
        let images: Vec<&[f32]> = vec![&img1, &img2];

        let stacker = FocusStacker::new(FocusMethod::Laplacian, BlendMode::Weighted);
        assert!(stacker.stack(&images, 0, 0).is_err());
    }

    #[test]
    fn test_focus_stack_mismatched_size() {
        let img1 = vec![0.5_f32; 16];
        let img2 = vec![0.5_f32; 8]; // Too small
        let images: Vec<&[f32]> = vec![&img1, &img2];

        let stacker = FocusStacker::new(FocusMethod::Laplacian, BlendMode::Weighted);
        assert!(stacker.stack(&images, 4, 4).is_err());
    }

    #[test]
    fn test_focus_stack_uniform_images() {
        let w = 6;
        let h = 6;
        let img1 = vec![0.5_f32; w * h];
        let img2 = vec![0.5_f32; w * h];
        let images: Vec<&[f32]> = vec![&img1, &img2];

        let stacker = FocusStacker::new(FocusMethod::Laplacian, BlendMode::Weighted);
        let result = stacker.stack(&images, w, h).expect("stack should succeed");
        // Uniform images have zero focus, should fallback to equal weight
        for &v in &result {
            assert!(
                (v - 0.5).abs() < 0.01,
                "Uniform image should produce ~0.5: {v}"
            );
        }
    }

    #[test]
    fn test_focus_stack_rgb() {
        let w = 6;
        let h = 6;
        let n = w * h * 3;
        // Image 1: sharp red channel pattern
        let mut img1 = vec![0.5_f32; n];
        for y in 0..h / 2 {
            for x in 0..w {
                let idx = (y * w + x) * 3;
                img1[idx] = if x % 2 == 0 { 1.0 } else { 0.0 };
            }
        }
        // Image 2: sharp pattern in bottom half
        let mut img2 = vec![0.5_f32; n];
        for y in h / 2..h {
            for x in 0..w {
                let idx = (y * w + x) * 3;
                img2[idx] = if x % 2 == 0 { 1.0 } else { 0.0 };
            }
        }

        let images: Vec<&[f32]> = vec![&img1, &img2];
        let stacker = FocusStacker::new(FocusMethod::Laplacian, BlendMode::Weighted);
        let result = stacker
            .stack_rgb(&images, w, h)
            .expect("stack_rgb should succeed");
        assert_eq!(result.len(), n);
        assert!(result.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_compute_focus_quality() {
        let w = 6;
        let h = 6;
        let img = make_focused_foreground(w, h);
        let quality = compute_focus_quality(&img, w, h, FocusMethod::Laplacian);
        assert_eq!(quality.len(), w * h);
        // Sharp regions should have non-zero focus measure
        assert!(quality.iter().any(|&v| v > 0.0));
    }

    #[test]
    fn test_overall_focus_score() {
        let w = 8;
        let h = 8;
        let sharp = make_focused_foreground(w, h);
        let uniform = vec![0.5_f32; w * h];

        let sharp_score = overall_focus_score(&sharp, w, h);
        let uniform_score = overall_focus_score(&uniform, w, h);

        assert!(
            sharp_score > uniform_score,
            "Sharp image should have higher focus score: {sharp_score} vs {uniform_score}"
        );
    }

    #[test]
    fn test_laplacian_focus_map_center() {
        // A single bright pixel in center should produce high Laplacian
        let w = 5;
        let h = 5;
        let mut img = vec![0.0_f32; w * h];
        img[2 * w + 2] = 1.0; // center
        let fm = laplacian_focus_map(&img, w, h);
        assert!(fm[2 * w + 2] > 0.0, "Center should have high focus");
    }

    #[test]
    fn test_gradient_focus_map_edge() {
        // Horizontal edge: left half 0, right half 1
        let w = 6;
        let h = 4;
        let mut img = vec![0.0_f32; w * h];
        for y in 0..h {
            for x in w / 2..w {
                img[y * w + x] = 1.0;
            }
        }
        let fm = gradient_focus_map(&img, w, h);
        // Edge column should have high gradient
        let edge_col = w / 2;
        for y in 1..h - 1 {
            assert!(
                fm[y * w + edge_col] > 0.0,
                "Edge should have gradient at ({edge_col}, {y})"
            );
        }
    }

    #[test]
    fn test_variance_focus_map_textured() {
        let w = 6;
        let h = 6;
        let mut img = vec![0.5_f32; w * h];
        // Add texture in center
        for y in 2..4 {
            for x in 2..4 {
                img[y * w + x] = if (x + y) % 2 == 0 { 1.0 } else { 0.0 };
            }
        }
        let fm = variance_focus_map(&img, w, h);
        // Textured region should have higher variance than uniform region
        let center_var = fm[2 * w + 2];
        assert!(center_var > 0.0, "Textured region should have variance");
    }

    #[test]
    fn test_gaussian_smooth() {
        let w = 5;
        let h = 5;
        let mut map = vec![0.0_f32; w * h];
        map[2 * w + 2] = 100.0; // impulse at center
        let smoothed = gaussian_smooth(&map, w, h, 1);
        // Smoothed center should be less than original
        assert!(smoothed[2 * w + 2] < 100.0);
        // Neighbors should have some energy
        assert!(smoothed[2 * w + 1] > 0.0);
        assert!(smoothed[1 * w + 2] > 0.0);
    }

    #[test]
    fn test_rgb_to_luminance() {
        // Pure white
        let rgb = vec![1.0_f32, 1.0, 1.0, 0.0, 0.0, 0.0];
        let lum = rgb_to_luminance(&rgb, 2);
        assert!((lum[0] - 1.0).abs() < 0.01);
        assert!((lum[1] - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_focus_method_eq() {
        assert_eq!(FocusMethod::Laplacian, FocusMethod::Laplacian);
        assert_ne!(FocusMethod::Laplacian, FocusMethod::GradientMagnitude);
    }

    #[test]
    fn test_blend_mode_eq() {
        assert_eq!(BlendMode::Maximum, BlendMode::Maximum);
        assert_ne!(BlendMode::Maximum, BlendMode::Weighted);
    }
}
