//! Image filtering operations: convolution, blur, sharpen.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

/// Convolution kernel size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelSize {
    /// 3x3 kernel.
    K3x3,
    /// 5x5 kernel.
    K5x5,
    /// 7x7 kernel.
    K7x7,
}

impl KernelSize {
    /// Returns the side length of the kernel.
    #[must_use]
    pub const fn size(self) -> usize {
        match self {
            Self::K3x3 => 3,
            Self::K5x5 => 5,
            Self::K7x7 => 7,
        }
    }

    /// Returns the half-size (floor(size/2)) used for border calculations.
    #[must_use]
    pub const fn half_size(self) -> usize {
        match self {
            Self::K3x3 => 1,
            Self::K5x5 => 2,
            Self::K7x7 => 3,
        }
    }
}

/// Compute a normalized 2-D Gaussian kernel returned as a flat row-major `Vec<f32>`.
///
/// The kernel has dimension `size.size() x size.size()` and sums to 1.0.
#[must_use]
pub fn gaussian_kernel(size: KernelSize, sigma: f32) -> Vec<f32> {
    let n = size.size();
    let half = size.half_size() as i32;
    let mut kernel = vec![0.0f32; n * n];
    let two_sigma_sq = 2.0 * sigma * sigma;

    let mut sum = 0.0f32;
    for ky in 0..n {
        for kx in 0..n {
            let dy = ky as i32 - half;
            let dx = kx as i32 - half;
            let val = (-(dx * dx + dy * dy) as f32 / two_sigma_sq).exp();
            kernel[ky * n + kx] = val;
            sum += val;
        }
    }

    // Normalize so the kernel sums to 1.
    for v in &mut kernel {
        *v /= sum;
    }
    kernel
}

/// Apply a square `kernel_size x kernel_size` convolution to a grayscale `pixels` slice.
///
/// `pixels` must have exactly `width * height` bytes. Returns a new `Vec<u8>` of the same
/// length. Border pixels use clamped (edge-replicate) addressing. Output values are clamped
/// to [0, 255].
#[must_use]
pub fn apply_convolution(
    pixels: &[u8],
    width: usize,
    height: usize,
    kernel: &[f32],
    kernel_size: usize,
) -> Vec<u8> {
    assert_eq!(pixels.len(), width * height, "pixel buffer size mismatch");
    assert_eq!(
        kernel.len(),
        kernel_size * kernel_size,
        "kernel size mismatch"
    );

    let half = (kernel_size / 2) as i64;
    let mut out = vec![0u8; width * height];

    for y in 0..height {
        for x in 0..width {
            let mut acc = 0.0f32;
            for ky in 0..kernel_size {
                for kx in 0..kernel_size {
                    let sy = (y as i64 + ky as i64 - half).clamp(0, height as i64 - 1) as usize;
                    let sx = (x as i64 + kx as i64 - half).clamp(0, width as i64 - 1) as usize;
                    acc += pixels[sy * width + sx] as f32 * kernel[ky * kernel_size + kx];
                }
            }
            out[y * width + x] = acc.clamp(0.0, 255.0).round() as u8;
        }
    }
    out
}

/// Apply a simple box blur of the given `radius` to a grayscale image.
///
/// A radius of 1 means a 3x3 average, radius 2 means 5x5, etc.
#[must_use]
pub fn box_blur(pixels: &[u8], width: usize, height: usize, radius: u32) -> Vec<u8> {
    assert_eq!(pixels.len(), width * height, "pixel buffer size mismatch");

    let r = radius as i64;
    let diameter = (2 * r + 1) as f32;
    let area = diameter * diameter;
    let mut out = vec![0u8; width * height];

    for y in 0..height {
        for x in 0..width {
            let mut sum = 0u32;
            for dy in -r..=r {
                for dx in -r..=r {
                    let sy = (y as i64 + dy).clamp(0, height as i64 - 1) as usize;
                    let sx = (x as i64 + dx).clamp(0, width as i64 - 1) as usize;
                    sum += pixels[sy * width + sx] as u32;
                }
            }
            out[y * width + x] = ((sum as f32 / area).round() as u32).min(255) as u8;
        }
    }
    out
}

/// Sharpen a grayscale image using a Laplacian sharpening kernel.
///
/// The kernel is:
/// ```text
///  0 -1  0
/// -1  5 -1
///  0 -1  0
/// ```
#[must_use]
pub fn sharpen(pixels: &[u8], width: usize, height: usize) -> Vec<u8> {
    assert_eq!(pixels.len(), width * height, "pixel buffer size mismatch");

    #[rustfmt::skip]
    let kernel: [f32; 9] = [
         0.0, -1.0,  0.0,
        -1.0,  5.0, -1.0,
         0.0, -1.0,  0.0,
    ];
    apply_convolution(pixels, width, height, &kernel, 3)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------- KernelSize tests ----------

    #[test]
    fn test_kernel_size_values() {
        assert_eq!(KernelSize::K3x3.size(), 3);
        assert_eq!(KernelSize::K5x5.size(), 5);
        assert_eq!(KernelSize::K7x7.size(), 7);
    }

    #[test]
    fn test_kernel_half_size() {
        assert_eq!(KernelSize::K3x3.half_size(), 1);
        assert_eq!(KernelSize::K5x5.half_size(), 2);
        assert_eq!(KernelSize::K7x7.half_size(), 3);
    }

    // ---------- gaussian_kernel tests ----------

    #[test]
    fn test_gaussian_kernel_length() {
        let k = gaussian_kernel(KernelSize::K3x3, 1.0);
        assert_eq!(k.len(), 9);

        let k5 = gaussian_kernel(KernelSize::K5x5, 1.0);
        assert_eq!(k5.len(), 25);

        let k7 = gaussian_kernel(KernelSize::K7x7, 1.5);
        assert_eq!(k7.len(), 49);
    }

    #[test]
    fn test_gaussian_kernel_sums_to_one() {
        let k = gaussian_kernel(KernelSize::K3x3, 1.0);
        let sum: f32 = k.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "kernel sum = {sum}");
    }

    #[test]
    fn test_gaussian_kernel_5x5_sums_to_one() {
        let k = gaussian_kernel(KernelSize::K5x5, 2.0);
        let sum: f32 = k.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "kernel sum = {sum}");
    }

    #[test]
    fn test_gaussian_kernel_center_is_max() {
        let k = gaussian_kernel(KernelSize::K3x3, 1.0);
        let center = k[4]; // (1,1) for 3x3
        for &v in &k {
            assert!(v <= center + 1e-6, "center should be maximum");
        }
    }

    #[test]
    fn test_gaussian_kernel_all_positive() {
        let k = gaussian_kernel(KernelSize::K7x7, 2.0);
        for &v in &k {
            assert!(v > 0.0);
        }
    }

    // ---------- apply_convolution tests ----------

    #[test]
    fn test_convolution_identity() {
        // Identity kernel should return the same image.
        let pixels = vec![100u8; 9]; // 3x3 all-100
        let kernel = [0.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let out = apply_convolution(&pixels, 3, 3, &kernel, 3);
        assert_eq!(out, pixels);
    }

    #[test]
    fn test_convolution_output_length() {
        let pixels = vec![128u8; 20 * 10];
        let kernel = gaussian_kernel(KernelSize::K3x3, 1.0);
        let out = apply_convolution(&pixels, 20, 10, &kernel, 3);
        assert_eq!(out.len(), 200);
    }

    #[test]
    fn test_convolution_uniform_image_unchanged() {
        // Convolving a uniform image with any normalized kernel should return the same value.
        let pixels = vec![200u8; 5 * 5];
        let kernel = gaussian_kernel(KernelSize::K3x3, 1.0);
        let out = apply_convolution(&pixels, 5, 5, &kernel, 3);
        for &v in &out {
            assert!((v as i16 - 200).abs() <= 1, "expected ~200 got {v}");
        }
    }

    // ---------- box_blur tests ----------

    #[test]
    fn test_box_blur_uniform() {
        let pixels = vec![128u8; 6 * 6];
        let out = box_blur(&pixels, 6, 6, 1);
        assert_eq!(out.len(), 36);
        for &v in &out {
            assert_eq!(v, 128);
        }
    }

    #[test]
    fn test_box_blur_output_length() {
        let pixels = vec![0u8; 8 * 8];
        let out = box_blur(&pixels, 8, 8, 2);
        assert_eq!(out.len(), 64);
    }

    #[test]
    fn test_box_blur_radius_zero() {
        // Radius 0 means a 1x1 kernel – identity.
        let pixels: Vec<u8> = (0..16).map(|i| (i * 10) as u8).collect();
        let out = box_blur(&pixels, 4, 4, 0);
        assert_eq!(out, pixels);
    }

    // ---------- sharpen tests ----------

    #[test]
    fn test_sharpen_output_length() {
        let pixels = vec![100u8; 4 * 4];
        let out = sharpen(&pixels, 4, 4);
        assert_eq!(out.len(), 16);
    }

    #[test]
    fn test_sharpen_uniform_image_unchanged() {
        // Sharpening a uniform image should leave it unchanged (kernel sums to 1).
        let pixels = vec![80u8; 5 * 5];
        let out = sharpen(&pixels, 5, 5);
        for &v in &out {
            assert_eq!(v, 80, "uniform image should be unchanged after sharpen");
        }
    }
}
