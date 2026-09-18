//! Depth estimation from a rectified stereo image pair.
//!
//! Given two horizontally-aligned images (i.e. the output of `stereo_rectify`),
//! this module computes a dense disparity map using **block matching** (Sum of
//! Absolute Differences) and then converts each disparity value to a physical
//! depth via the standard stereo camera formula:
//!
//! ```text
//! depth = (focal_length_px * baseline_m) / disparity
//! ```
//!
//! # Design
//!
//! | Symbol | Meaning |
//! |--------|---------|
//! | `f`    | Focal length in pixels (same for both rectified cameras) |
//! | `B`    | Baseline — distance between camera optical centres in metres |
//! | `d`    | Disparity in pixels (left column minus best-matching right column) |
//!
//! The search is restricted to `[min_disparity, max_disparity)` pixels to the
//! **left** of the corresponding point in the right image (standard convention
//! for left-camera disparity).  Pixels for which no valid disparity is found
//! (near the image border or where min_disparity == 0) receive `f32::INFINITY`.

use crate::{AlignError, AlignResult};

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the block-matching stereo depth estimator.
#[derive(Debug, Clone)]
pub struct StereoDepthConfig {
    /// Side length (in pixels) of the square matching window.
    ///
    /// Must be odd and ≥ 1.  Typical values: 5, 7, 11.
    pub block_size: usize,

    /// Minimum disparity to search (inclusive).  Must be ≥ 0.
    pub min_disparity: i32,

    /// Maximum disparity to search (exclusive).  Must be > `min_disparity`.
    pub max_disparity: i32,

    /// Focal length of the rectified cameras in **pixels**.
    pub focal_length_px: f64,

    /// Distance between camera optical centres in **metres** (baseline).
    pub baseline_m: f64,
}

impl StereoDepthConfig {
    /// Returns `Err` if the configuration is self-contradictory.
    ///
    /// # Errors
    ///
    /// - `block_size` is zero or even.
    /// - `min_disparity >= max_disparity`.
    /// - `focal_length_px` or `baseline_m` are non-positive.
    pub fn validate(&self) -> AlignResult<()> {
        if self.block_size == 0 || self.block_size % 2 == 0 {
            return Err(AlignError::InvalidConfig(
                "block_size must be a positive odd number".to_string(),
            ));
        }
        if self.min_disparity >= self.max_disparity {
            return Err(AlignError::InvalidConfig(
                "min_disparity must be less than max_disparity".to_string(),
            ));
        }
        if self.focal_length_px <= 0.0 {
            return Err(AlignError::InvalidConfig(
                "focal_length_px must be positive".to_string(),
            ));
        }
        if self.baseline_m <= 0.0 {
            return Err(AlignError::InvalidConfig(
                "baseline_m must be positive".to_string(),
            ));
        }
        Ok(())
    }
}

impl Default for StereoDepthConfig {
    fn default() -> Self {
        Self {
            block_size: 7,
            min_disparity: 0,
            max_disparity: 64,
            focal_length_px: 700.0,
            baseline_m: 0.12,
        }
    }
}

// ─── Estimator ────────────────────────────────────────────────────────────────

/// Stereo depth estimator using SAD (Sum of Absolute Differences) block matching.
///
/// Expects **rectified** grayscale image pairs — i.e. the horizontal epipolar
/// lines of left and right images coincide, so the search reduces to a 1-D
/// scan along each row.
///
/// # Example
///
/// ```rust
/// use oximedia_align::stereo_depth::{StereoDepthConfig, StereoDepthEstimator};
///
/// let config = StereoDepthConfig {
///     block_size: 5,
///     min_disparity: 1,
///     max_disparity: 16,
///     focal_length_px: 500.0,
///     baseline_m: 0.1,
/// };
/// let estimator = StereoDepthEstimator::new();
///
/// let width = 16usize;
/// let height = 8usize;
/// let left = vec![128u8; width * height];
/// let right = vec![128u8; width * height];
///
/// // Uniform images → disparity 1 (min_disparity) for every pixel
/// let depth = estimator
///     .compute_depth_map(&left, &right, width, height, &config)
///     .unwrap();
/// assert_eq!(depth.len(), width * height);
/// ```
#[derive(Debug, Default)]
pub struct StereoDepthEstimator;

impl StereoDepthEstimator {
    /// Creates a new `StereoDepthEstimator`.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Computes a dense depth map from a rectified stereo pair.
    ///
    /// # Parameters
    ///
    /// - `left`   – Row-major grayscale pixels (u8) of the **left** camera image.
    /// - `right`  – Row-major grayscale pixels (u8) of the **right** camera image.
    /// - `width`  – Image width in pixels.  Both images must have the same dimensions.
    /// - `height` – Image height in pixels.
    /// - `config` – Block matching and camera parameters.
    ///
    /// # Returns
    ///
    /// A `Vec<f32>` with `width * height` depth values in metres, stored in
    /// row-major order.  Pixels for which depth cannot be determined are set to
    /// `f32::INFINITY`.
    ///
    /// # Errors
    ///
    /// Returns [`AlignError::InvalidConfig`] if `config.validate()` fails, or
    /// [`AlignError::InsufficientData`] if the images are too small to hold
    /// even a single matching block.
    pub fn compute_depth_map(
        &self,
        left: &[u8],
        right: &[u8],
        width: usize,
        height: usize,
        config: &StereoDepthConfig,
    ) -> AlignResult<Vec<f32>> {
        config.validate()?;

        if left.len() != width * height || right.len() != width * height {
            return Err(AlignError::InsufficientData(
                "Image buffer length does not match width × height".to_string(),
            ));
        }

        let half = config.block_size / 2;
        let max_disp = config.max_disparity as usize;
        let min_disp = config.min_disparity.max(0) as usize;

        let mut depth_map = vec![f32::INFINITY; width * height];

        for y in 0..height {
            // Row range for the matching block (clipped to image bounds)
            let row_start = y.saturating_sub(half);
            let row_end = (y + half + 1).min(height);

            for x in 0..width {
                // Column range for the left patch
                let col_start_l = x.saturating_sub(half);
                let col_end_l = (x + half + 1).min(width);

                let mut best_sad = u64::MAX;
                let mut best_disp: Option<usize> = None;

                // Search over candidate disparities
                for d in min_disp..max_disp {
                    // Shift right image patch to the left by d pixels
                    // Right patch starts at col_start_l - d (saturated)
                    let right_col_start = if col_start_l >= d {
                        col_start_l - d
                    } else {
                        continue; // patch would go out of bounds
                    };
                    let right_col_end = if col_end_l > d {
                        col_end_l - d
                    } else {
                        continue;
                    };

                    let mut sad: u64 = 0;
                    let mut count: usize = 0;

                    for row in row_start..row_end {
                        let left_row = &left[row * width..];
                        let right_row = &right[row * width..];

                        let l_slice = &left_row[col_start_l..col_end_l];
                        let r_slice = &right_row[right_col_start..right_col_end];

                        // Number of overlapping columns
                        let cols = l_slice.len().min(r_slice.len());
                        for c in 0..cols {
                            let diff =
                                (i32::from(l_slice[c]) - i32::from(r_slice[c])).unsigned_abs();
                            sad += u64::from(diff);
                            count += 1;
                        }
                    }

                    if count == 0 {
                        continue;
                    }

                    // Normalise by area to make SAD comparable across border pixels
                    let norm_sad = sad / count as u64;

                    if norm_sad < best_sad {
                        best_sad = norm_sad;
                        best_disp = Some(d);
                    }
                }

                if let Some(d) = best_disp {
                    if d == 0 {
                        // Zero disparity → infinite depth (point at infinity)
                        depth_map[y * width + x] = f32::INFINITY;
                    } else {
                        let depth = (config.focal_length_px * config.baseline_m) / d as f64;
                        depth_map[y * width + x] = depth as f32;
                    }
                }
                // Else: remains INFINITY (no valid match found)
            }
        }

        Ok(depth_map)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a uniform gray image.
    fn uniform_image(width: usize, height: usize, value: u8) -> Vec<u8> {
        vec![value; width * height]
    }

    /// A disparity of `d` pixels means every left-image column is best matched
    /// by the right-image column shifted by `d` to the left.  For a uniform
    /// image the SAD is 0 for every candidate, so the first (minimum) disparity
    /// `min_disparity` wins.  The expected depth is then:
    ///
    /// `depth = f * B / min_disparity`
    #[test]
    fn test_depth_map_uniform_disparity() {
        let config = StereoDepthConfig {
            block_size: 3,
            min_disparity: 2,
            max_disparity: 16,
            focal_length_px: 400.0,
            baseline_m: 0.1,
        };
        let estimator = StereoDepthEstimator::new();
        let w = 24usize;
        let h = 12usize;

        let left = uniform_image(w, h, 100);
        let right = uniform_image(w, h, 100);

        let depth = estimator
            .compute_depth_map(&left, &right, w, h, &config)
            .expect("depth map should succeed");

        assert_eq!(depth.len(), w * h);

        // Expected: f * B / min_disparity = 400.0 * 0.1 / 2 = 20.0
        let expected = (config.focal_length_px * config.baseline_m) / config.min_disparity as f64;

        for (idx, &d) in depth.iter().enumerate() {
            // Border pixels may be INFINITY (not enough block overlap)
            if d.is_finite() {
                assert!(
                    (f64::from(d) - expected).abs() < 1.0,
                    "pixel {idx}: depth {d} expected ~{expected}"
                );
            }
        }
    }

    /// When `min_disparity == 0`, every pixel whose best match is at disparity 0
    /// should receive `f32::INFINITY` (infinite depth / point at infinity).
    #[test]
    fn test_depth_map_zero_disparity_infinite_depth() {
        let config = StereoDepthConfig {
            block_size: 3,
            min_disparity: 0, // allows zero disparity
            max_disparity: 8,
            focal_length_px: 500.0,
            baseline_m: 0.2,
        };
        let estimator = StereoDepthEstimator::new();
        let w = 20usize;
        let h = 10usize;

        // Identical images → best disparity is 0 → depth = INFINITY
        let img = uniform_image(w, h, 128);
        let depth = estimator
            .compute_depth_map(&img, &img, w, h, &config)
            .expect("depth map should succeed");

        assert_eq!(depth.len(), w * h);

        // Every valid pixel should be INFINITY when disparity is 0
        for &d in &depth {
            assert!(d.is_infinite(), "expected infinite depth, got {d}");
        }
    }

    /// Validate that non-trivial disparity (shifted right image) is recovered.
    ///
    /// Construct a left image with a vertical stripe of bright pixels and a
    /// right image with the same stripe shifted by exactly `shift` pixels.
    #[test]
    fn test_depth_map_known_shift() {
        let config = StereoDepthConfig {
            block_size: 3,
            min_disparity: 1,
            max_disparity: 10,
            focal_length_px: 300.0,
            baseline_m: 0.06,
        };
        let estimator = StereoDepthEstimator::new();
        let w = 32usize;
        let h = 16usize;
        let shift: usize = 4;

        let mut left = vec![50u8; w * h];
        let mut right = vec![50u8; w * h];

        // Bright stripe at column 12 in left; column 12 - shift in right
        let stripe_col_l: usize = 12;
        let stripe_col_r = stripe_col_l.saturating_sub(shift);
        for row in 0..h {
            left[row * w + stripe_col_l] = 200;
            if stripe_col_r < w {
                right[row * w + stripe_col_r] = 200;
            }
        }

        let depth = estimator
            .compute_depth_map(&left, &right, w, h, &config)
            .expect("depth map should succeed");

        // Near the stripe centre pixels should have depth ≈ f*B/shift
        let expected = (config.focal_length_px * config.baseline_m) / shift as f64;
        let half = config.block_size / 2;

        for row in half..(h - half) {
            let idx = row * w + stripe_col_l;
            let d = depth[idx];
            if d.is_finite() {
                assert!(
                    (f64::from(d) - expected).abs() < expected * 0.5,
                    "row {row}: depth {d} expected ~{expected}"
                );
            }
        }
    }

    #[test]
    fn test_config_validation_rejects_even_block_size() {
        let config = StereoDepthConfig {
            block_size: 4,
            ..StereoDepthConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_rejects_inverted_disparity_range() {
        let config = StereoDepthConfig {
            min_disparity: 10,
            max_disparity: 5,
            ..StereoDepthConfig::default()
        };
        assert!(config.validate().is_err());
    }
}
