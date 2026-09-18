#![allow(dead_code)]

//! Blur kernel generation and application for VFX compositing.
//!
//! Supports Gaussian, box, and disc (bokeh-style) blur kernels with
//! configurable radius and normalisation.

use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Kernel types
// ---------------------------------------------------------------------------

/// Shape of the blur kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelShape {
    /// Gaussian (bell-curve) blur.
    Gaussian,
    /// Uniform box blur.
    Box,
    /// Disc / circular (bokeh-like) blur.
    Disc,
    /// Tent (triangular) filter.
    Tent,
}

/// A 2-D convolution kernel stored in row-major order.
#[derive(Debug, Clone)]
pub struct BlurKernel {
    /// Kernel width (always odd).
    pub width: usize,
    /// Kernel height (always odd).
    pub height: usize,
    /// Weight data in row-major order (length = width * height).
    pub weights: Vec<f64>,
    /// Shape that was used to generate this kernel.
    pub shape: KernelShape,
}

impl BlurKernel {
    // -- constructors -------------------------------------------------------

    /// Generate a Gaussian blur kernel.
    ///
    /// `radius` determines the half-size; the kernel side is `2 * radius + 1`.
    /// `sigma` controls the spread. If `sigma <= 0`, it defaults to `radius / 2`.
    #[allow(clippy::cast_precision_loss)]
    pub fn gaussian(radius: usize, sigma: f64) -> Self {
        let sigma = if sigma <= 0.0 {
            (radius.max(1) as f64) / 2.0
        } else {
            sigma
        };
        let side = 2 * radius + 1;
        let mut weights = vec![0.0_f64; side * side];
        let center = radius as f64;
        let two_sigma_sq = 2.0 * sigma * sigma;

        let mut sum = 0.0_f64;
        for y in 0..side {
            for x in 0..side {
                let dx = x as f64 - center;
                let dy = y as f64 - center;
                let w = (-(dx * dx + dy * dy) / two_sigma_sq).exp();
                weights[y * side + x] = w;
                sum += w;
            }
        }
        // normalise
        if sum > 0.0 {
            for w in &mut weights {
                *w /= sum;
            }
        }

        Self {
            width: side,
            height: side,
            weights,
            shape: KernelShape::Gaussian,
        }
    }

    /// Generate a uniform box-blur kernel.
    #[allow(clippy::cast_precision_loss)]
    pub fn box_blur(radius: usize) -> Self {
        let side = 2 * radius + 1;
        let n = (side * side) as f64;
        let weights = vec![1.0 / n; side * side];
        Self {
            width: side,
            height: side,
            weights,
            shape: KernelShape::Box,
        }
    }

    /// Generate a disc (circular) blur kernel.
    #[allow(clippy::cast_precision_loss)]
    pub fn disc(radius: usize) -> Self {
        let side = 2 * radius + 1;
        let center = radius as f64;
        let r2 = (radius as f64 + 0.5) * (radius as f64 + 0.5);
        let mut weights = vec![0.0_f64; side * side];
        let mut sum = 0.0_f64;

        for y in 0..side {
            for x in 0..side {
                let dx = x as f64 - center;
                let dy = y as f64 - center;
                if dx * dx + dy * dy <= r2 {
                    weights[y * side + x] = 1.0;
                    sum += 1.0;
                }
            }
        }
        if sum > 0.0 {
            for w in &mut weights {
                *w /= sum;
            }
        }
        Self {
            width: side,
            height: side,
            weights,
            shape: KernelShape::Disc,
        }
    }

    /// Generate a tent (triangular) kernel.
    #[allow(clippy::cast_precision_loss)]
    pub fn tent(radius: usize) -> Self {
        let side = 2 * radius + 1;
        let center = radius as f64;
        let max_dist = center + 1.0;
        let mut weights = vec![0.0_f64; side * side];
        let mut sum = 0.0_f64;

        for y in 0..side {
            for x in 0..side {
                let dx = (x as f64 - center).abs();
                let dy = (y as f64 - center).abs();
                let dist = dx.max(dy);
                let w = (1.0 - dist / max_dist).max(0.0);
                weights[y * side + x] = w;
                sum += w;
            }
        }
        if sum > 0.0 {
            for w in &mut weights {
                *w /= sum;
            }
        }
        Self {
            width: side,
            height: side,
            weights,
            shape: KernelShape::Tent,
        }
    }

    // -- queries ------------------------------------------------------------

    /// Return the kernel radius (half-size).
    pub fn radius(&self) -> usize {
        self.width / 2
    }

    /// Return total number of weights.
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// Check if the kernel is empty (should never be).
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    /// Sum of all weights (should be ~1.0 for normalised kernels).
    pub fn weight_sum(&self) -> f64 {
        self.weights.iter().sum()
    }

    /// Get weight at (x, y) in kernel space.
    pub fn get(&self, x: usize, y: usize) -> Option<f64> {
        if x < self.width && y < self.height {
            Some(self.weights[y * self.width + x])
        } else {
            None
        }
    }

    /// Return the center weight.
    pub fn center_weight(&self) -> f64 {
        let cx = self.width / 2;
        let cy = self.height / 2;
        self.weights[cy * self.width + cx]
    }

    /// Apply the kernel to a single-channel f64 image buffer.
    ///
    /// `src` is row-major with dimensions `(img_w, img_h)`.
    /// Returns a new buffer of the same size.
    #[allow(clippy::cast_precision_loss)]
    pub fn convolve(&self, src: &[f64], img_w: usize, img_h: usize) -> Vec<f64> {
        let mut dst = vec![0.0_f64; img_w * img_h];
        let r = self.radius();

        for y in 0..img_h {
            for x in 0..img_w {
                let mut acc = 0.0_f64;
                for ky in 0..self.height {
                    for kx in 0..self.width {
                        let sy = (y as isize + ky as isize - r as isize)
                            .clamp(0, (img_h - 1) as isize)
                            as usize;
                        let sx = (x as isize + kx as isize - r as isize)
                            .clamp(0, (img_w - 1) as isize)
                            as usize;
                        acc += src[sy * img_w + sx] * self.weights[ky * self.width + kx];
                    }
                }
                dst[y * img_w + x] = acc;
            }
        }
        dst
    }

    /// Generate a 1-D Gaussian kernel for separable convolution.
    #[allow(clippy::cast_precision_loss)]
    pub fn gaussian_1d(radius: usize, sigma: f64) -> Vec<f64> {
        let sigma = if sigma <= 0.0 {
            (radius.max(1) as f64) / 2.0
        } else {
            sigma
        };
        let side = 2 * radius + 1;
        let center = radius as f64;
        let two_sigma_sq = 2.0 * sigma * sigma;
        let norm = 1.0 / (two_sigma_sq * PI).sqrt();

        let mut k: Vec<f64> = (0..side)
            .map(|i| {
                let d = i as f64 - center;
                norm * (-d * d / two_sigma_sq).exp()
            })
            .collect();

        let sum: f64 = k.iter().sum();
        if sum > 0.0 {
            for v in &mut k {
                *v /= sum;
            }
        }
        k
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_kernel_size() {
        let k = BlurKernel::gaussian(3, 1.5);
        assert_eq!(k.width, 7);
        assert_eq!(k.height, 7);
        assert_eq!(k.len(), 49);
    }

    #[test]
    fn test_gaussian_kernel_normalised() {
        let k = BlurKernel::gaussian(4, 2.0);
        let sum = k.weight_sum();
        assert!((sum - 1.0).abs() < 1e-9, "sum was {sum}");
    }

    #[test]
    fn test_gaussian_center_is_max() {
        let k = BlurKernel::gaussian(3, 1.5);
        let center = k.center_weight();
        for &w in &k.weights {
            assert!(w <= center + 1e-12);
        }
    }

    #[test]
    fn test_box_blur_uniform() {
        let k = BlurKernel::box_blur(2);
        let expected = 1.0 / 25.0;
        for &w in &k.weights {
            assert!((w - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn test_box_blur_normalised() {
        let k = BlurKernel::box_blur(5);
        let sum = k.weight_sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_disc_kernel_normalised() {
        let k = BlurKernel::disc(4);
        let sum = k.weight_sum();
        assert!((sum - 1.0).abs() < 1e-9, "disc sum was {sum}");
    }

    #[test]
    fn test_disc_center_nonzero() {
        let k = BlurKernel::disc(3);
        assert!(k.center_weight() > 0.0);
    }

    #[test]
    fn test_tent_kernel_normalised() {
        let k = BlurKernel::tent(3);
        let sum = k.weight_sum();
        assert!((sum - 1.0).abs() < 1e-9, "tent sum was {sum}");
    }

    #[test]
    fn test_tent_center_is_max() {
        let k = BlurKernel::tent(4);
        let cw = k.center_weight();
        for &w in &k.weights {
            assert!(w <= cw + 1e-12);
        }
    }

    #[test]
    fn test_kernel_get() {
        let k = BlurKernel::box_blur(1);
        assert!(k.get(0, 0).is_some());
        assert!(k.get(10, 10).is_none());
    }

    #[test]
    fn test_convolve_identity() {
        // radius 0 kernel (1x1 with weight 1) should return the same image
        let k = BlurKernel::box_blur(0);
        let src = vec![1.0, 2.0, 3.0, 4.0];
        let dst = k.convolve(&src, 2, 2);
        for (a, b) in src.iter().zip(dst.iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_convolve_box_smoothing() {
        let k = BlurKernel::box_blur(1);
        // 3x3 image with a spike in the centre
        let src = vec![0.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0];
        let dst = k.convolve(&src, 3, 3);
        // centre should be 9/9 = 1.0
        assert!((dst[4] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_gaussian_1d_normalised() {
        let k = BlurKernel::gaussian_1d(5, 2.0);
        let sum: f64 = k.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "1d sum was {sum}");
    }

    #[test]
    fn test_gaussian_1d_symmetric() {
        let k = BlurKernel::gaussian_1d(4, 2.0);
        let n = k.len();
        for i in 0..n / 2 {
            assert!((k[i] - k[n - 1 - i]).abs() < 1e-12, "asymmetry at {i}");
        }
    }

    #[test]
    fn test_radius_accessor() {
        let k = BlurKernel::gaussian(5, 2.5);
        assert_eq!(k.radius(), 5);
    }
}
