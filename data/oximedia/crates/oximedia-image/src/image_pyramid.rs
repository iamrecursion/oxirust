//! Advanced image pyramid operations: Gaussian, Laplacian, and multi-resolution blending.
//!
//! Uses a proper 5x5 Gaussian kernel `[1,4,6,4,1]/16` for high-quality downsampling,
//! supports Laplacian pyramids for reconstruction, and provides multi-resolution
//! blending of two images with a mask pyramid.

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// Gaussian kernel
// ---------------------------------------------------------------------------

/// 1-D Gaussian kernel weights: [1, 4, 6, 4, 1] / 16.
const GAUSS_KERNEL_1D: [f32; 5] = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];

/// Apply a separable 5x5 Gaussian blur on single-channel f32 image.
#[must_use]
#[allow(clippy::cast_precision_loss)]
fn gaussian_blur_5x5(image: &[f32], width: u32, height: u32) -> Vec<f32> {
    let w = width as usize;
    let h = height as usize;
    if w == 0 || h == 0 || image.len() != w * h {
        return image.to_vec();
    }

    // Horizontal pass
    let mut temp = vec![0.0_f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0_f32;
            for (ki, &kv) in GAUSS_KERNEL_1D.iter().enumerate() {
                let sx = (x as i64 + ki as i64 - 2).clamp(0, w as i64 - 1) as usize;
                sum += image[y * w + sx] * kv;
            }
            temp[y * w + x] = sum;
        }
    }

    // Vertical pass
    let mut result = vec![0.0_f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0_f32;
            for (ki, &kv) in GAUSS_KERNEL_1D.iter().enumerate() {
                let sy = (y as i64 + ki as i64 - 2).clamp(0, h as i64 - 1) as usize;
                sum += temp[sy * w + x] * kv;
            }
            result[y * w + x] = sum;
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Downsample / Upsample
// ---------------------------------------------------------------------------

/// Downsample a single-channel f32 image by factor 2 using 5x5 Gaussian blur then subsampling.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn downsample_2x(image: &[f32], width: u32, height: u32) -> (Vec<f32>, u32, u32) {
    let w = width as usize;
    let h = height as usize;
    if w <= 1 && h <= 1 {
        return (image.to_vec(), width, height);
    }

    let blurred = gaussian_blur_5x5(image, width, height);
    let new_w = ((w + 1) / 2).max(1);
    let new_h = ((h + 1) / 2).max(1);

    let mut dst = vec![0.0_f32; new_w * new_h];
    for dy in 0..new_h {
        let sy = (dy * 2).min(h - 1);
        for dx in 0..new_w {
            let sx = (dx * 2).min(w - 1);
            dst[dy * new_w + dx] = blurred[sy * w + sx];
        }
    }
    (dst, new_w as u32, new_h as u32)
}

/// Upsample a single-channel f32 image by factor 2 using bilinear interpolation.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn upsample_2x(
    image: &[f32],
    width: u32,
    height: u32,
    target_w: u32,
    target_h: u32,
) -> Vec<f32> {
    let sw = width as usize;
    let sh = height as usize;
    let tw = target_w as usize;
    let th = target_h as usize;

    if sw == 0 || sh == 0 || tw == 0 || th == 0 {
        return vec![0.0_f32; tw * th];
    }

    let mut dst = vec![0.0_f32; tw * th];
    for y in 0..th {
        let sy_f = y as f32 / 2.0;
        let sy0 = (sy_f.floor() as usize).min(sh - 1);
        let sy1 = (sy0 + 1).min(sh - 1);
        let fy = sy_f - sy_f.floor();

        for x in 0..tw {
            let sx_f = x as f32 / 2.0;
            let sx0 = (sx_f.floor() as usize).min(sw - 1);
            let sx1 = (sx0 + 1).min(sw - 1);
            let fx = sx_f - sx_f.floor();

            let v00 = image[sy0 * sw + sx0];
            let v10 = image[sy0 * sw + sx1];
            let v01 = image[sy1 * sw + sx0];
            let v11 = image[sy1 * sw + sx1];

            let val = v00 * (1.0 - fx) * (1.0 - fy)
                + v10 * fx * (1.0 - fy)
                + v01 * (1.0 - fx) * fy
                + v11 * fx * fy;
            dst[y * tw + x] = val;
        }
    }
    dst
}

// ---------------------------------------------------------------------------
// GaussianPyramid
// ---------------------------------------------------------------------------

/// Gaussian pyramid: each level is a progressively lower-resolution version of the image.
#[derive(Debug, Clone)]
pub struct GaussianPyramid {
    /// Levels stored as (data, width, height). Level 0 is the original.
    pub levels: Vec<(Vec<f32>, u32, u32)>,
}

impl GaussianPyramid {
    /// Build a Gaussian pyramid with `num_levels` levels.
    ///
    /// Level 0 is the original image. Each subsequent level is blurred with a
    /// 5x5 Gaussian kernel and downsampled by 2.
    #[must_use]
    pub fn build(image: &[f32], width: u32, height: u32, num_levels: u32) -> Self {
        let mut levels = Vec::with_capacity(num_levels as usize);
        if width == 0 || height == 0 || num_levels == 0 {
            return Self { levels };
        }

        levels.push((image.to_vec(), width, height));

        for _ in 1..num_levels {
            let (prev_data, prev_w, prev_h) = levels
                .last()
                .map(|(d, w, h)| (d.as_slice(), *w, *h))
                .unwrap_or((&[], 0, 0));
            if prev_w <= 1 && prev_h <= 1 {
                break;
            }
            let (down, nw, nh) = downsample_2x(prev_data, prev_w, prev_h);
            levels.push((down, nw, nh));
        }

        Self { levels }
    }

    /// Number of levels in the pyramid.
    #[must_use]
    pub fn depth(&self) -> usize {
        self.levels.len()
    }

    /// Get a specific level (data, width, height).
    #[must_use]
    pub fn get(&self, level: usize) -> Option<(&[f32], u32, u32)> {
        self.levels
            .get(level)
            .map(|(d, w, h)| (d.as_slice(), *w, *h))
    }
}

// ---------------------------------------------------------------------------
// LaplacianPyramid
// ---------------------------------------------------------------------------

/// Laplacian pyramid for band-pass decomposition and reconstruction.
///
/// Each level stores the difference between the Gaussian level and the upsampled
/// next-coarser Gaussian level.  The coarsest level is the low-pass residual.
#[derive(Debug, Clone)]
pub struct LaplacianPyramid {
    /// Laplacian levels (band-pass residuals). The last element is the coarsest
    /// Gaussian level (low-pass).
    pub levels: Vec<(Vec<f32>, u32, u32)>,
}

impl LaplacianPyramid {
    /// Build a Laplacian pyramid from a Gaussian pyramid.
    ///
    /// Each Laplacian level `i` = `gaussian[i] - upsample(gaussian[i+1])`.
    /// The coarsest Gaussian level is stored as the last Laplacian "level".
    #[must_use]
    pub fn from_gaussian(gp: &GaussianPyramid) -> Self {
        let n = gp.levels.len();
        if n == 0 {
            return Self { levels: Vec::new() };
        }
        if n == 1 {
            return Self {
                levels: vec![gp.levels[0].clone()],
            };
        }

        let mut levels = Vec::with_capacity(n);

        for i in 0..n - 1 {
            let (ref g_data, g_w, g_h) = gp.levels[i];
            let (ref g_next, next_w, next_h) = gp.levels[i + 1];

            // Upsample the coarser level to match current level size
            let upsampled = upsample_2x(g_next, next_w, next_h, g_w, g_h);

            // Laplacian = current - upsampled
            let lap: Vec<f32> = g_data
                .iter()
                .zip(upsampled.iter())
                .map(|(&a, &b)| a - b)
                .collect();

            levels.push((lap, g_w, g_h));
        }

        // Last level is the coarsest Gaussian
        levels.push(gp.levels[n - 1].clone());

        Self { levels }
    }

    /// Reconstruct the original image from the Laplacian pyramid.
    ///
    /// Starts from the coarsest level and adds back each Laplacian band.
    #[must_use]
    pub fn reconstruct(&self) -> Vec<f32> {
        let n = self.levels.len();
        if n == 0 {
            return Vec::new();
        }
        if n == 1 {
            return self.levels[0].0.clone();
        }

        // Start with the coarsest level
        let (mut current, mut cw, mut ch) = self.levels[n - 1].clone();

        // Add back Laplacian levels from coarse to fine
        for i in (0..n - 1).rev() {
            let (ref lap, lw, lh) = self.levels[i];

            // Upsample current to match the Laplacian level size
            let upsampled = upsample_2x(&current, cw, ch, lw, lh);

            // Add Laplacian band
            current = upsampled
                .iter()
                .zip(lap.iter())
                .map(|(&u, &l)| u + l)
                .collect();
            cw = lw;
            ch = lh;
        }

        current
    }

    /// Number of levels (including the low-pass base).
    #[must_use]
    pub fn depth(&self) -> usize {
        self.levels.len()
    }
}

// ---------------------------------------------------------------------------
// Multi-resolution blending
// ---------------------------------------------------------------------------

/// Blend two images using multi-resolution pyramid blending.
///
/// Given Gaussian pyramids `a` and `b`, and a Gaussian pyramid of a blending
/// mask (values 0.0–1.0), produce a blended Laplacian pyramid and reconstruct.
///
/// The formula at each level: `blended[i] = mask[i] * lap_a[i] + (1 - mask[i]) * lap_b[i]`
#[must_use]
pub fn blend_pyramids(
    a: &GaussianPyramid,
    b: &GaussianPyramid,
    mask_gaussian: &GaussianPyramid,
) -> Vec<f32> {
    let lap_a = LaplacianPyramid::from_gaussian(a);
    let lap_b = LaplacianPyramid::from_gaussian(b);

    let n = lap_a
        .levels
        .len()
        .min(lap_b.levels.len())
        .min(mask_gaussian.levels.len());
    if n == 0 {
        return Vec::new();
    }

    let mut blended_levels: Vec<(Vec<f32>, u32, u32)> = Vec::with_capacity(n);

    for i in 0..n {
        let (ref la, la_w, la_h) = lap_a.levels[i];
        let (ref lb, _, _) = lap_b.levels[i];
        let (ref mask_data, _, _) = mask_gaussian.levels[i];

        let size = la.len().min(lb.len()).min(mask_data.len());
        let mut blended = vec![0.0_f32; size];
        for j in 0..size {
            let m = mask_data[j].clamp(0.0, 1.0);
            blended[j] = m * la[j] + (1.0 - m) * lb[j];
        }

        blended_levels.push((blended, la_w, la_h));
    }

    let blended_lap = LaplacianPyramid {
        levels: blended_levels,
    };
    blended_lap.reconstruct()
}

// ---------------------------------------------------------------------------
// Utility: convert u8 image to/from f32
// ---------------------------------------------------------------------------

/// Convert grayscale u8 image to f32 (0.0–255.0 range).
#[must_use]
pub fn u8_to_f32(image: &[u8]) -> Vec<f32> {
    image.iter().map(|&v| v as f32).collect()
}

/// Convert f32 image to grayscale u8 (clamped to 0–255).
#[must_use]
pub fn f32_to_u8(image: &[f32]) -> Vec<u8> {
    image
        .iter()
        .map(|&v| v.round().clamp(0.0, 255.0) as u8)
        .collect()
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- Gaussian blur ---

    #[test]
    fn test_gaussian_blur_uniform() {
        let img = vec![100.0_f32; 8 * 8];
        let blurred = gaussian_blur_5x5(&img, 8, 8);
        for &v in &blurred {
            assert!((v - 100.0).abs() < 0.1, "Expected ~100, got {v}");
        }
    }

    #[test]
    fn test_gaussian_blur_empty() {
        let blurred = gaussian_blur_5x5(&[], 0, 0);
        assert!(blurred.is_empty());
    }

    // --- Downsample ---

    #[test]
    fn test_downsample_2x_dimensions() {
        let img = vec![50.0_f32; 8 * 8];
        let (down, w, h) = downsample_2x(&img, 8, 8);
        assert_eq!(w, 4);
        assert_eq!(h, 4);
        assert_eq!(down.len(), 16);
    }

    #[test]
    fn test_downsample_2x_uniform_value() {
        let img = vec![42.0_f32; 16 * 16];
        let (down, _, _) = downsample_2x(&img, 16, 16);
        for &v in &down {
            assert!((v - 42.0).abs() < 0.5, "Expected ~42, got {v}");
        }
    }

    #[test]
    fn test_downsample_2x_odd_dimensions() {
        let img = vec![1.0_f32; 7 * 5];
        let (_, w, h) = downsample_2x(&img, 7, 5);
        assert_eq!(w, 4); // (7+1)/2 = 4
        assert_eq!(h, 3); // (5+1)/2 = 3
    }

    // --- Upsample ---

    #[test]
    fn test_upsample_2x_dimensions() {
        let img = vec![1.0_f32; 4 * 4];
        let up = upsample_2x(&img, 4, 4, 8, 8);
        assert_eq!(up.len(), 64);
    }

    #[test]
    fn test_upsample_2x_uniform() {
        let img = vec![77.0_f32; 3 * 3];
        let up = upsample_2x(&img, 3, 3, 6, 6);
        for &v in &up {
            assert!((v - 77.0).abs() < 0.1, "Expected ~77, got {v}");
        }
    }

    // --- GaussianPyramid ---

    #[test]
    fn test_gaussian_pyramid_build() {
        let img = vec![128.0_f32; 32 * 32];
        let gp = GaussianPyramid::build(&img, 32, 32, 4);
        assert_eq!(gp.depth(), 4);
    }

    #[test]
    fn test_gaussian_pyramid_level_sizes() {
        let img = vec![0.0_f32; 64 * 64];
        let gp = GaussianPyramid::build(&img, 64, 64, 5);
        let sizes: Vec<(u32, u32)> = gp.levels.iter().map(|(_, w, h)| (*w, *h)).collect();
        assert_eq!(sizes[0], (64, 64));
        assert_eq!(sizes[1], (32, 32));
        assert_eq!(sizes[2], (16, 16));
        assert_eq!(sizes[3], (8, 8));
        assert_eq!(sizes[4], (4, 4));
    }

    #[test]
    fn test_gaussian_pyramid_empty() {
        let gp = GaussianPyramid::build(&[], 0, 0, 3);
        assert_eq!(gp.depth(), 0);
    }

    #[test]
    fn test_gaussian_pyramid_single_level() {
        let img = vec![1.0_f32; 4];
        let gp = GaussianPyramid::build(&img, 2, 2, 1);
        assert_eq!(gp.depth(), 1);
    }

    #[test]
    fn test_gaussian_pyramid_get() {
        let img = vec![10.0_f32; 16];
        let gp = GaussianPyramid::build(&img, 4, 4, 2);
        assert!(gp.get(0).is_some());
        assert!(gp.get(1).is_some());
        assert!(gp.get(5).is_none());
    }

    // --- LaplacianPyramid ---

    #[test]
    fn test_laplacian_from_gaussian() {
        let img = vec![100.0_f32; 16 * 16];
        let gp = GaussianPyramid::build(&img, 16, 16, 3);
        let lp = LaplacianPyramid::from_gaussian(&gp);
        assert_eq!(lp.depth(), 3);
    }

    #[test]
    fn test_laplacian_reconstruct_uniform() {
        let img = vec![150.0_f32; 32 * 32];
        let gp = GaussianPyramid::build(&img, 32, 32, 4);
        let lp = LaplacianPyramid::from_gaussian(&gp);
        let recon = lp.reconstruct();
        assert_eq!(recon.len(), img.len());
        for (i, &v) in recon.iter().enumerate() {
            assert!((v - 150.0).abs() < 2.0, "Pixel {i}: expected ~150, got {v}");
        }
    }

    #[test]
    fn test_laplacian_reconstruct_gradient() {
        // A gradient image should be approximately reconstructed
        let w = 16u32;
        let h = 16u32;
        let img: Vec<f32> = (0..w * h)
            .map(|i| (i % w) as f32 * (255.0 / 15.0))
            .collect();
        let gp = GaussianPyramid::build(&img, w, h, 3);
        let lp = LaplacianPyramid::from_gaussian(&gp);
        let recon = lp.reconstruct();
        let max_err: f32 = img
            .iter()
            .zip(recon.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_err < 10.0,
            "Max reconstruction error {max_err} too large"
        );
    }

    #[test]
    fn test_laplacian_empty() {
        let gp = GaussianPyramid { levels: Vec::new() };
        let lp = LaplacianPyramid::from_gaussian(&gp);
        assert_eq!(lp.depth(), 0);
        assert!(lp.reconstruct().is_empty());
    }

    #[test]
    fn test_laplacian_single_level() {
        let img = vec![42.0_f32; 4];
        let gp = GaussianPyramid::build(&img, 2, 2, 1);
        let lp = LaplacianPyramid::from_gaussian(&gp);
        assert_eq!(lp.depth(), 1);
        let recon = lp.reconstruct();
        assert_eq!(recon.len(), 4);
    }

    // --- Blending ---

    #[test]
    fn test_blend_pyramids_uniform() {
        let w = 16u32;
        let h = 16u32;
        let n = (w * h) as usize;
        let img_a = vec![200.0_f32; n];
        let img_b = vec![50.0_f32; n];
        let mask = vec![0.5_f32; n]; // 50/50 blend

        let gp_a = GaussianPyramid::build(&img_a, w, h, 3);
        let gp_b = GaussianPyramid::build(&img_b, w, h, 3);
        let gp_mask = GaussianPyramid::build(&mask, w, h, 3);

        let blended = blend_pyramids(&gp_a, &gp_b, &gp_mask);
        assert_eq!(blended.len(), n);
        // Expect roughly 125 everywhere
        for (i, &v) in blended.iter().enumerate() {
            assert!(
                (v - 125.0).abs() < 10.0,
                "Pixel {i}: expected ~125, got {v}"
            );
        }
    }

    #[test]
    fn test_blend_pyramids_mask_zero() {
        let w = 8u32;
        let h = 8u32;
        let n = (w * h) as usize;
        let img_a = vec![255.0_f32; n];
        let img_b = vec![0.0_f32; n];
        let mask = vec![0.0_f32; n]; // All B

        let gp_a = GaussianPyramid::build(&img_a, w, h, 2);
        let gp_b = GaussianPyramid::build(&img_b, w, h, 2);
        let gp_mask = GaussianPyramid::build(&mask, w, h, 2);

        let blended = blend_pyramids(&gp_a, &gp_b, &gp_mask);
        for &v in &blended {
            assert!((v).abs() < 5.0, "Expected ~0, got {v}");
        }
    }

    // --- Conversion utilities ---

    #[test]
    fn test_u8_to_f32() {
        let input = vec![0u8, 128, 255];
        let f = u8_to_f32(&input);
        assert_eq!(f, vec![0.0, 128.0, 255.0]);
    }

    #[test]
    fn test_f32_to_u8() {
        let input = vec![-10.0_f32, 128.5, 300.0];
        let u = f32_to_u8(&input);
        assert_eq!(u, vec![0, 129, 255]);
    }

    #[test]
    fn test_roundtrip_u8_f32_u8() {
        let input = vec![0u8, 42, 128, 200, 255];
        let converted = f32_to_u8(&u8_to_f32(&input));
        assert_eq!(converted, input);
    }
}
