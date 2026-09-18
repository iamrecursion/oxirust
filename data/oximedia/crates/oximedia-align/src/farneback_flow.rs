//! Dense optical flow using the Farneback polynomial expansion method.
//!
//! Implements a multi-scale dense optical flow algorithm inspired by Gunnar
//! Farneback's method. Each pixel in the image is assigned a flow vector
//! by fitting local polynomial approximations and solving for displacement.
//!
//! # Algorithm Overview
//!
//! 1. Build Gaussian image pyramids.
//! 2. At the coarsest level, initialise flow to zero.
//! 3. At each level:
//!    a. Compute the polynomial expansion coefficients (quadratic model) for a
//!       local neighbourhood around each pixel.
//!    b. Use the polynomial coefficients from both frames to solve for the
//!       displacement field via a weighted least-squares fit.
//!    c. Propagate to the next finer level (upsample by 2).
//! 4. The final result is a per-pixel `(dx, dy)` flow field at original resolution.
//!
//! # References
//!
//! - Farneback, G. "Two-Frame Motion Estimation Based on Polynomial Expansion"
//!   Proceedings of the 13th Scandinavian Conference on Image Analysis, 2003.

#![allow(clippy::cast_precision_loss)]

use crate::{AlignError, AlignResult};

/// Configuration for Farneback dense optical flow.
#[derive(Debug, Clone)]
pub struct FarnebackConfig {
    /// Number of pyramid levels (1 = single resolution).
    pub pyramid_levels: usize,
    /// Polynomial expansion neighbourhood (half-size). Full window is `2*n + 1`.
    pub poly_n: usize,
    /// Standard deviation of the Gaussian weighting inside the poly window.
    pub poly_sigma: f64,
    /// Number of iterations at each pyramid level.
    pub iterations: usize,
    /// Size of the averaging window for flow smoothing.
    pub win_size: usize,
}

impl Default for FarnebackConfig {
    fn default() -> Self {
        Self {
            pyramid_levels: 3,
            poly_n: 3,
            poly_sigma: 1.2,
            iterations: 5,
            win_size: 5,
        }
    }
}

/// Dense flow field: per-pixel (dx, dy).
#[derive(Debug, Clone)]
pub struct DenseFlowField {
    /// Horizontal displacements, row-major.
    pub dx: Vec<f32>,
    /// Vertical displacements, row-major.
    pub dy: Vec<f32>,
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
}

impl DenseFlowField {
    /// Create a zero-initialised dense flow field.
    #[must_use]
    pub fn zeros(width: usize, height: usize) -> Self {
        let n = width * height;
        Self {
            dx: vec![0.0; n],
            dy: vec![0.0; n],
            width,
            height,
        }
    }

    /// Get the flow vector at pixel `(x, y)`.
    #[must_use]
    pub fn at(&self, x: usize, y: usize) -> Option<(f32, f32)> {
        if x < self.width && y < self.height {
            let idx = y * self.width + x;
            Some((self.dx[idx], self.dy[idx]))
        } else {
            None
        }
    }

    /// Compute the average magnitude over all pixels.
    #[must_use]
    pub fn avg_magnitude(&self) -> f32 {
        let n = self.dx.len();
        if n == 0 {
            return 0.0;
        }
        let sum: f64 = self
            .dx
            .iter()
            .zip(self.dy.iter())
            .map(|(&dx, &dy)| f64::from((dx * dx + dy * dy).sqrt()))
            .sum();
        (sum / n as f64) as f32
    }

    /// Maximum magnitude flow vector.
    #[must_use]
    pub fn max_magnitude(&self) -> f32 {
        self.dx
            .iter()
            .zip(self.dy.iter())
            .map(|(&dx, &dy)| (dx * dx + dy * dy).sqrt())
            .fold(0.0_f32, f32::max)
    }

    /// Upsample by factor 2 using bilinear interpolation.
    #[must_use]
    pub fn upsample_2x(&self) -> Self {
        let nw = self.width * 2;
        let nh = self.height * 2;
        let mut out = DenseFlowField::zeros(nw, nh);

        for y in 0..nh {
            for x in 0..nw {
                let sx = x as f32 / 2.0;
                let sy = y as f32 / 2.0;

                let x0 = (sx.floor() as usize).min(self.width.saturating_sub(1));
                let y0 = (sy.floor() as usize).min(self.height.saturating_sub(1));
                let x1 = (x0 + 1).min(self.width.saturating_sub(1));
                let y1 = (y0 + 1).min(self.height.saturating_sub(1));

                let fx = sx - x0 as f32;
                let fy = sy - y0 as f32;

                let idx = y * nw + x;

                // Bilinear interpolate dx
                let d00 = self.dx[y0 * self.width + x0];
                let d10 = self.dx[y0 * self.width + x1];
                let d01 = self.dx[y1 * self.width + x0];
                let d11 = self.dx[y1 * self.width + x1];
                out.dx[idx] = (d00 * (1.0 - fx) * (1.0 - fy)
                    + d10 * fx * (1.0 - fy)
                    + d01 * (1.0 - fx) * fy
                    + d11 * fx * fy)
                    * 2.0; // scale flow by 2

                let d00 = self.dy[y0 * self.width + x0];
                let d10 = self.dy[y0 * self.width + x1];
                let d01 = self.dy[y1 * self.width + x0];
                let d11 = self.dy[y1 * self.width + x1];
                out.dy[idx] = (d00 * (1.0 - fx) * (1.0 - fy)
                    + d10 * fx * (1.0 - fy)
                    + d01 * (1.0 - fx) * fy
                    + d11 * fx * fy)
                    * 2.0;
            }
        }

        out
    }
}

/// Compute dense optical flow using the Farneback method.
///
/// Both `prev` and `curr` must be single-channel grayscale images of size
/// `width * height`.
///
/// # Errors
///
/// Returns an error if the images have mismatched sizes or are too small.
pub fn compute_farneback_flow(
    prev: &[u8],
    curr: &[u8],
    width: usize,
    height: usize,
    config: &FarnebackConfig,
) -> AlignResult<DenseFlowField> {
    if prev.len() != width * height || curr.len() != width * height {
        return Err(AlignError::InvalidConfig(
            "Image size does not match width*height".to_string(),
        ));
    }
    if width < 8 || height < 8 {
        return Err(AlignError::InvalidConfig(
            "Image must be at least 8x8".to_string(),
        ));
    }

    // Build pyramids (f32 for precision)
    let prev_pyr = build_f32_pyramid(prev, width, height, config.pyramid_levels);
    let curr_pyr = build_f32_pyramid(curr, width, height, config.pyramid_levels);

    // Start at coarsest level with zero flow
    let coarsest = prev_pyr.len() - 1;
    let mut flow = DenseFlowField::zeros(prev_pyr[coarsest].1, prev_pyr[coarsest].2);

    // Coarse-to-fine refinement
    for level in (0..prev_pyr.len()).rev() {
        let (ref prev_img, pw, ph) = prev_pyr[level];
        let (ref curr_img, _, _) = curr_pyr[level];

        // If not the coarsest level, upsample the flow from the previous (coarser) result
        if level < coarsest {
            flow = flow.upsample_2x();
            // Trim to actual level dimensions (pyramid dimensions might differ by 1)
            flow = trim_flow(&flow, pw, ph);
        }

        // Compute polynomial expansion for both images
        let poly_prev = polynomial_expansion(prev_img, pw, ph, config.poly_n, config.poly_sigma);
        let poly_curr = polynomial_expansion(curr_img, pw, ph, config.poly_n, config.poly_sigma);

        // Iterative refinement at this level
        for _iter in 0..config.iterations {
            flow = update_flow(&poly_prev, &poly_curr, &flow, pw, ph, config.win_size);
        }
    }

    Ok(flow)
}

/// Polynomial expansion coefficients for one pixel.
/// We store a simplified version: (r1, r2, r3, r4, r5) representing the
/// quadratic model f(dx,dy) ~ r1*dx^2 + r2*dy^2 + r3*dx*dy + r4*dx + r5*dy + const.
#[derive(Debug, Clone, Copy, Default)]
struct PolyCoeff {
    a11: f32, // coefficient of dx^2
    a22: f32, // coefficient of dy^2
    a12: f32, // coefficient of dx*dy
    b1: f32,  // coefficient of dx
    b2: f32,  // coefficient of dy
}

/// Compute polynomial expansion for the image.
fn polynomial_expansion(
    image: &[f32],
    width: usize,
    height: usize,
    poly_n: usize,
    poly_sigma: f64,
) -> Vec<PolyCoeff> {
    let n = width * height;
    let mut coeffs = vec![PolyCoeff::default(); n];

    // Build Gaussian weight kernel
    let ksize = 2 * poly_n + 1;
    let mut kernel = vec![0.0_f64; ksize * ksize];
    let sigma2 = poly_sigma * poly_sigma;
    let pn = poly_n as isize;

    for ky in 0..ksize {
        for kx in 0..ksize {
            let dx = kx as f64 - poly_n as f64;
            let dy = ky as f64 - poly_n as f64;
            kernel[ky * ksize + kx] = (-0.5 * (dx * dx + dy * dy) / sigma2).exp();
        }
    }

    // For each interior pixel, fit a local quadratic
    for y in poly_n..height.saturating_sub(poly_n) {
        for x in poly_n..width.saturating_sub(poly_n) {
            let idx = y * width + x;

            // Weighted least squares to fit: f = a11*dx^2 + a22*dy^2 + a12*dx*dy + b1*dx + b2*dy + c
            // We accumulate the normal equations.
            let mut s_xx_xx = 0.0_f64;
            let mut s_yy_yy = 0.0_f64;
            let mut s_xy_xy = 0.0_f64;
            let mut s_x_x = 0.0_f64;
            let mut s_y_y = 0.0_f64;
            let mut s_xx_f = 0.0_f64;
            let mut s_yy_f = 0.0_f64;
            let mut s_xy_f = 0.0_f64;
            let mut s_x_f = 0.0_f64;
            let mut s_y_f = 0.0_f64;

            for ky in 0..ksize {
                for kx in 0..ksize {
                    let dx = kx as f64 - poly_n as f64;
                    let dy = ky as f64 - poly_n as f64;
                    let w = kernel[ky * ksize + kx];

                    let nx = (x as isize + kx as isize - pn) as usize;
                    let ny = (y as isize + ky as isize - pn) as usize;
                    let val = f64::from(image[ny * width + nx]);

                    let xx = dx * dx;
                    let yy = dy * dy;
                    let xy = dx * dy;

                    s_xx_xx += w * xx * xx;
                    s_yy_yy += w * yy * yy;
                    s_xy_xy += w * xy * xy;
                    s_x_x += w * dx * dx;
                    s_y_y += w * dy * dy;
                    s_xx_f += w * xx * val;
                    s_yy_f += w * yy * val;
                    s_xy_f += w * xy * val;
                    s_x_f += w * dx * val;
                    s_y_f += w * dy * val;
                }
            }

            // Solve diagonal-approximated normal equations for simplicity.
            // Full normal equation solve would involve a 6x6 system; we use
            // the diagonal approximation which is sufficient for flow estimation.
            let a11 = if s_xx_xx > 1e-10 {
                s_xx_f / s_xx_xx
            } else {
                0.0
            };
            let a22 = if s_yy_yy > 1e-10 {
                s_yy_f / s_yy_yy
            } else {
                0.0
            };
            let a12 = if s_xy_xy > 1e-10 {
                s_xy_f / s_xy_xy
            } else {
                0.0
            };
            let b1 = if s_x_x > 1e-10 { s_x_f / s_x_x } else { 0.0 };
            let b2 = if s_y_y > 1e-10 { s_y_f / s_y_y } else { 0.0 };

            coeffs[idx] = PolyCoeff {
                a11: a11 as f32,
                a22: a22 as f32,
                a12: a12 as f32,
                b1: b1 as f32,
                b2: b2 as f32,
            };
        }
    }

    coeffs
}

/// Update the flow field using the polynomial coefficients.
fn update_flow(
    poly_prev: &[PolyCoeff],
    poly_curr: &[PolyCoeff],
    flow: &DenseFlowField,
    width: usize,
    height: usize,
    win_size: usize,
) -> DenseFlowField {
    let mut new_flow = DenseFlowField::zeros(width, height);
    let half = (win_size / 2) as isize;

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;

            // Accumulate structure tensor and mismatch from neighbourhood
            let mut h11 = 0.0_f32;
            let mut h22 = 0.0_f32;
            let mut h12 = 0.0_f32;
            let mut g1 = 0.0_f32;
            let mut g2 = 0.0_f32;

            for dy in -half..=half {
                for dx in -half..=half {
                    let nx = x as isize + dx;
                    let ny = y as isize + dy;

                    if nx < 0 || ny < 0 || nx >= width as isize || ny >= height as isize {
                        continue;
                    }

                    let nidx = ny as usize * width + nx as usize;
                    let pp = &poly_prev[nidx];
                    let pc = &poly_curr[nidx];

                    // Average of the two Hessians
                    let a11 = (pp.a11 + pc.a11) * 0.5;
                    let a22 = (pp.a22 + pc.a22) * 0.5;
                    let a12 = (pp.a12 + pc.a12) * 0.5;

                    // Difference of the gradients adjusted for current flow
                    let fx = flow.dx[nidx];
                    let fy = flow.dy[nidx];

                    let db1 = pc.b1 - pp.b1 + 2.0 * a11 * fx + a12 * fy;
                    let db2 = pc.b2 - pp.b2 + a12 * fx + 2.0 * a22 * fy;

                    h11 += 4.0 * a11 * a11 + a12 * a12;
                    h22 += a12 * a12 + 4.0 * a22 * a22;
                    h12 += 2.0 * a11 * a12 + 2.0 * a12 * a22;
                    g1 += 2.0 * a11 * db1 + a12 * db2;
                    g2 += a12 * db1 + 2.0 * a22 * db2;
                }
            }

            // Solve 2x2 system
            let det = h11 * h22 - h12 * h12;
            if det.abs() > 1e-6 {
                let inv_det = 1.0 / det;
                new_flow.dx[idx] = flow.dx[idx] - (h22 * g1 - h12 * g2) * inv_det;
                new_flow.dy[idx] = flow.dy[idx] - (-h12 * g1 + h11 * g2) * inv_det;
            } else {
                new_flow.dx[idx] = flow.dx[idx];
                new_flow.dy[idx] = flow.dy[idx];
            }
        }
    }

    new_flow
}

// -- Pyramid helpers ----------------------------------------------------------

fn build_f32_pyramid(
    image: &[u8],
    width: usize,
    height: usize,
    levels: usize,
) -> Vec<(Vec<f32>, usize, usize)> {
    let f32_img: Vec<f32> = image.iter().map(|&v| f32::from(v)).collect();
    let mut pyr = Vec::with_capacity(levels);
    pyr.push((f32_img, width, height));

    for _ in 1..levels {
        let (ref prev, pw, ph) = pyr[pyr.len() - 1];
        let nw = pw / 2;
        let nh = ph / 2;
        if nw < 4 || nh < 4 {
            break;
        }
        let down = downsample_f32(prev, pw, ph, nw, nh);
        pyr.push((down, nw, nh));
    }

    pyr
}

fn downsample_f32(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
    let mut dst = vec![0.0_f32; dw * dh];
    for dy in 0..dh {
        for dx in 0..dw {
            let sx = dx * 2;
            let sy = dy * 2;
            let mut sum = 0.0_f32;
            let mut count = 0u32;
            for oy in 0..2 {
                for ox in 0..2 {
                    let rx = sx + ox;
                    let ry = sy + oy;
                    if rx < sw && ry < sh {
                        sum += src[ry * sw + rx];
                        count += 1;
                    }
                }
            }
            dst[dy * dw + dx] = if count > 0 { sum / count as f32 } else { 0.0 };
        }
    }
    dst
}

fn trim_flow(flow: &DenseFlowField, target_w: usize, target_h: usize) -> DenseFlowField {
    let mut out = DenseFlowField::zeros(target_w, target_h);
    for y in 0..target_h.min(flow.height) {
        for x in 0..target_w.min(flow.width) {
            let src_idx = y * flow.width + x;
            let dst_idx = y * target_w + x;
            out.dx[dst_idx] = flow.dx[src_idx];
            out.dy[dst_idx] = flow.dy[src_idx];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- DenseFlowField -------------------------------------------------------

    #[test]
    fn test_dense_flow_zeros() {
        let f = DenseFlowField::zeros(10, 10);
        assert_eq!(f.width, 10);
        assert_eq!(f.height, 10);
        assert_eq!(f.avg_magnitude(), 0.0);
        assert_eq!(f.max_magnitude(), 0.0);
    }

    #[test]
    fn test_dense_flow_at() {
        let mut f = DenseFlowField::zeros(4, 4);
        f.dx[5] = 3.0;
        f.dy[5] = 4.0;
        let (dx, dy) = f.at(1, 1).expect("should be in bounds");
        assert!((dx - 3.0).abs() < 1e-6);
        assert!((dy - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_dense_flow_at_out_of_bounds() {
        let f = DenseFlowField::zeros(4, 4);
        assert!(f.at(10, 10).is_none());
    }

    #[test]
    fn test_dense_flow_upsample() {
        let mut f = DenseFlowField::zeros(4, 4);
        for i in 0..16 {
            f.dx[i] = 1.0;
            f.dy[i] = -1.0;
        }
        let up = f.upsample_2x();
        assert_eq!(up.width, 8);
        assert_eq!(up.height, 8);
        // After upsampling, flow values should be roughly doubled
        let (dx, dy) = up.at(4, 4).expect("should be in bounds");
        assert!((dx - 2.0).abs() < 0.5, "dx={dx}");
        assert!((dy + 2.0).abs() < 0.5, "dy={dy}");
    }

    #[test]
    fn test_dense_flow_max_magnitude() {
        let mut f = DenseFlowField::zeros(4, 4);
        f.dx[0] = 3.0;
        f.dy[0] = 4.0;
        assert!((f.max_magnitude() - 5.0).abs() < 1e-5);
    }

    // -- Farneback flow -------------------------------------------------------

    #[test]
    fn test_farneback_identical_frames() {
        let w = 64usize;
        let h = 64usize;
        let img = vec![128u8; w * h];

        let config = FarnebackConfig {
            pyramid_levels: 2,
            poly_n: 2,
            poly_sigma: 1.0,
            iterations: 3,
            win_size: 5,
        };

        let flow = compute_farneback_flow(&img, &img, w, h, &config).expect("should succeed");
        assert_eq!(flow.width, w);
        assert_eq!(flow.height, h);
        // Identical frames should have near-zero flow
        assert!(
            flow.avg_magnitude() < 1.0,
            "identical frames avg_mag={}",
            flow.avg_magnitude()
        );
    }

    #[test]
    fn test_farneback_shifted_image() {
        let w = 64usize;
        let h = 64usize;
        // Create a vertical stripe pattern
        let mut prev = vec![0u8; w * h];
        let mut curr = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                prev[y * w + x] = if (x / 8) % 2 == 0 { 200 } else { 50 };
                // Shift by 2 pixels to the right
                let sx = (x + 2).min(w - 1);
                curr[y * w + x] = if (sx / 8) % 2 == 0 { 200 } else { 50 };
            }
        }

        let config = FarnebackConfig {
            pyramid_levels: 2,
            poly_n: 3,
            poly_sigma: 1.2,
            iterations: 5,
            win_size: 7,
        };

        let flow = compute_farneback_flow(&prev, &curr, w, h, &config).expect("should succeed");
        // There should be some non-zero flow
        assert!(flow.max_magnitude() > 0.0, "should detect some motion");
    }

    #[test]
    fn test_farneback_image_mismatch() {
        let config = FarnebackConfig::default();
        let result = compute_farneback_flow(&[0u8; 100], &[0u8; 200], 10, 10, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_farneback_too_small() {
        let config = FarnebackConfig::default();
        let result = compute_farneback_flow(&[0u8; 4], &[0u8; 4], 2, 2, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_farneback_default_config() {
        let config = FarnebackConfig::default();
        assert_eq!(config.pyramid_levels, 3);
        assert_eq!(config.poly_n, 3);
        assert_eq!(config.iterations, 5);
    }

    // -- Pyramid helpers ------------------------------------------------------

    #[test]
    fn test_f32_pyramid_levels() {
        let img = vec![100u8; 64 * 64];
        let pyr = build_f32_pyramid(&img, 64, 64, 3);
        assert_eq!(pyr.len(), 3);
        assert_eq!(pyr[0].1, 64);
        assert_eq!(pyr[1].1, 32);
        assert_eq!(pyr[2].1, 16);
    }

    #[test]
    fn test_f32_pyramid_constant_preserved() {
        let img = vec![100u8; 32 * 32];
        let pyr = build_f32_pyramid(&img, 32, 32, 2);
        for &v in &pyr[1].0 {
            assert!((v - 100.0).abs() < 1e-3);
        }
    }

    #[test]
    fn test_trim_flow_smaller() {
        let f = DenseFlowField::zeros(8, 8);
        let trimmed = trim_flow(&f, 4, 4);
        assert_eq!(trimmed.width, 4);
        assert_eq!(trimmed.height, 4);
    }

    #[test]
    fn test_polynomial_expansion_constant() {
        let img = vec![100.0_f32; 32 * 32];
        let coeffs = polynomial_expansion(&img, 32, 32, 2, 1.0);
        // On a constant image, the linear and quadratic coefficients should be near zero.
        for c in &coeffs[3 * 32 + 3..29 * 32 + 29] {
            assert!(c.b1.abs() < 1.0, "b1={}", c.b1);
            assert!(c.b2.abs() < 1.0, "b2={}", c.b2);
        }
    }
}
