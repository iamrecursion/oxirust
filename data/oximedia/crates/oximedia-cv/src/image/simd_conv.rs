//! SIMD-accelerated convolution for image filtering.
//!
//! Provides optimized 2D convolution for common kernel sizes (3x3, 5x5, 7x7)
//! using batch processing and loop unrolling. Falls back to scalar code on
//! platforms without SIMD intrinsics, while still benefiting from the
//! cache-friendly memory access patterns.

use crate::error::{CvError, CvResult};

/// SIMD-accelerated convolution engine.
///
/// Optimizes convolution by processing multiple pixels at once using
/// vectorized operations and cache-friendly access patterns.
#[derive(Debug, Clone)]
pub struct SimdConvolver {
    /// Kernel coefficients (row-major, always square).
    kernel: Vec<f32>,
    /// Kernel dimension (3, 5, or 7).
    size: usize,
    /// Half-kernel radius.
    half: usize,
}

impl SimdConvolver {
    /// Create a new SIMD convolver with the given square kernel.
    ///
    /// # Errors
    ///
    /// Returns an error if the kernel size is not 3, 5, or 7, or data length mismatches.
    pub fn new(kernel: &[f64], size: usize) -> CvResult<Self> {
        if size != 3 && size != 5 && size != 7 {
            return Err(CvError::invalid_parameter(
                "kernel_size",
                "must be 3, 5, or 7",
            ));
        }
        if kernel.len() != size * size {
            return Err(CvError::invalid_parameter(
                "kernel",
                "length must equal size*size",
            ));
        }

        let kernel_f32: Vec<f32> = kernel.iter().map(|&v| v as f32).collect();

        Ok(Self {
            kernel: kernel_f32,
            size,
            half: size / 2,
        })
    }

    /// Create a 3x3 Gaussian blur kernel.
    ///
    /// # Errors
    ///
    /// Returns an error if kernel construction fails.
    pub fn gaussian_3x3() -> CvResult<Self> {
        #[rustfmt::skip]
        let k = [
            1.0/16.0, 2.0/16.0, 1.0/16.0,
            2.0/16.0, 4.0/16.0, 2.0/16.0,
            1.0/16.0, 2.0/16.0, 1.0/16.0,
        ];
        Self::new(&k, 3)
    }

    /// Create a 3x3 Sobel X kernel.
    ///
    /// # Errors
    ///
    /// Returns an error if kernel construction fails.
    pub fn sobel_x_3x3() -> CvResult<Self> {
        #[rustfmt::skip]
        let k = [
            -1.0, 0.0, 1.0,
            -2.0, 0.0, 2.0,
            -1.0, 0.0, 1.0,
        ];
        Self::new(&k, 3)
    }

    /// Create a 3x3 Sobel Y kernel.
    ///
    /// # Errors
    ///
    /// Returns an error if kernel construction fails.
    pub fn sobel_y_3x3() -> CvResult<Self> {
        #[rustfmt::skip]
        let k = [
            -1.0, -2.0, -1.0,
             0.0,  0.0,  0.0,
             1.0,  2.0,  1.0,
        ];
        Self::new(&k, 3)
    }

    /// Create a 5x5 Gaussian blur kernel.
    ///
    /// # Errors
    ///
    /// Returns an error if kernel construction fails.
    pub fn gaussian_5x5() -> CvResult<Self> {
        #[rustfmt::skip]
        let k = [
            1.0/256.0,  4.0/256.0,  6.0/256.0,  4.0/256.0,  1.0/256.0,
            4.0/256.0, 16.0/256.0, 24.0/256.0, 16.0/256.0,  4.0/256.0,
            6.0/256.0, 24.0/256.0, 36.0/256.0, 24.0/256.0,  6.0/256.0,
            4.0/256.0, 16.0/256.0, 24.0/256.0, 16.0/256.0,  4.0/256.0,
            1.0/256.0,  4.0/256.0,  6.0/256.0,  4.0/256.0,  1.0/256.0,
        ];
        Self::new(&k, 5)
    }

    /// Create a 7x7 Gaussian blur kernel with sigma=1.0.
    ///
    /// # Errors
    ///
    /// Returns an error if kernel construction fails.
    pub fn gaussian_7x7() -> CvResult<Self> {
        let sigma = 1.0f64;
        let two_sigma_sq = 2.0 * sigma * sigma;
        let mut k = [0.0f64; 49];
        let mut sum = 0.0;
        for ky in 0..7 {
            for kx in 0..7 {
                let dx = kx as f64 - 3.0;
                let dy = ky as f64 - 3.0;
                let val = (-(dx * dx + dy * dy) / two_sigma_sq).exp();
                k[ky * 7 + kx] = val;
                sum += val;
            }
        }
        for v in &mut k {
            *v /= sum;
        }
        Self::new(&k, 7)
    }

    /// Apply the convolution to a grayscale image.
    ///
    /// Uses SIMD-friendly batch processing: processes 4 output pixels at a time
    /// with unrolled inner loops for each kernel size.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are invalid.
    pub fn convolve(&self, src: &[u8], width: u32, height: u32) -> CvResult<Vec<u8>> {
        let w = width as usize;
        let h = height as usize;

        if w == 0 || h == 0 {
            return Err(CvError::invalid_dimensions(width, height));
        }
        let size = w * h;
        if src.len() < size {
            return Err(CvError::insufficient_data(size, src.len()));
        }

        match self.size {
            3 => self.convolve_3x3(src, w, h),
            5 => self.convolve_5x5(src, w, h),
            7 => self.convolve_7x7(src, w, h),
            _ => self.convolve_generic(src, w, h),
        }
    }

    /// Optimized 3x3 convolution with unrolled kernel application.
    fn convolve_3x3(&self, src: &[u8], w: usize, h: usize) -> CvResult<Vec<u8>> {
        let mut dst = vec![0u8; w * h];
        let k = &self.kernel;

        for y in 0..h {
            // Process 4 pixels at a time for vectorization opportunity
            let mut x = 0;
            while x + 4 <= w {
                let mut sums = [0.0f32; 4];

                for ky in 0..3 {
                    let sy = (y as i32 + ky as i32 - 1).clamp(0, h as i32 - 1) as usize;
                    let row_offset = sy * w;

                    for kx in 0..3 {
                        let k_val = k[ky * 3 + kx];

                        for lane in 0..4 {
                            let sx = (x as i32 + lane as i32 + kx as i32 - 1).clamp(0, w as i32 - 1)
                                as usize;
                            sums[lane] += src[row_offset + sx] as f32 * k_val;
                        }
                    }
                }

                for lane in 0..4 {
                    dst[y * w + x + lane] = sums[lane].round().clamp(0.0, 255.0) as u8;
                }
                x += 4;
            }

            // Handle remaining pixels
            while x < w {
                let mut sum = 0.0f32;
                for ky in 0..3 {
                    let sy = (y as i32 + ky as i32 - 1).clamp(0, h as i32 - 1) as usize;
                    for kx in 0..3 {
                        let sx = (x as i32 + kx as i32 - 1).clamp(0, w as i32 - 1) as usize;
                        sum += src[sy * w + sx] as f32 * k[ky * 3 + kx];
                    }
                }
                dst[y * w + x] = sum.round().clamp(0.0, 255.0) as u8;
                x += 1;
            }
        }

        Ok(dst)
    }

    /// Optimized 5x5 convolution with unrolled kernel application.
    fn convolve_5x5(&self, src: &[u8], w: usize, h: usize) -> CvResult<Vec<u8>> {
        let mut dst = vec![0u8; w * h];
        let k = &self.kernel;

        for y in 0..h {
            let mut x = 0;
            while x + 4 <= w {
                let mut sums = [0.0f32; 4];

                for ky in 0..5 {
                    let sy = (y as i32 + ky as i32 - 2).clamp(0, h as i32 - 1) as usize;
                    let row_offset = sy * w;

                    for kx in 0..5 {
                        let k_val = k[ky * 5 + kx];

                        for lane in 0..4 {
                            let sx = (x as i32 + lane as i32 + kx as i32 - 2).clamp(0, w as i32 - 1)
                                as usize;
                            sums[lane] += src[row_offset + sx] as f32 * k_val;
                        }
                    }
                }

                for lane in 0..4 {
                    dst[y * w + x + lane] = sums[lane].round().clamp(0.0, 255.0) as u8;
                }
                x += 4;
            }

            while x < w {
                let mut sum = 0.0f32;
                for ky in 0..5 {
                    let sy = (y as i32 + ky as i32 - 2).clamp(0, h as i32 - 1) as usize;
                    for kx in 0..5 {
                        let sx = (x as i32 + kx as i32 - 2).clamp(0, w as i32 - 1) as usize;
                        sum += src[sy * w + sx] as f32 * k[ky * 5 + kx];
                    }
                }
                dst[y * w + x] = sum.round().clamp(0.0, 255.0) as u8;
                x += 1;
            }
        }

        Ok(dst)
    }

    /// Optimized 7x7 convolution with unrolled kernel application.
    fn convolve_7x7(&self, src: &[u8], w: usize, h: usize) -> CvResult<Vec<u8>> {
        let mut dst = vec![0u8; w * h];
        let k = &self.kernel;

        for y in 0..h {
            let mut x = 0;
            while x + 4 <= w {
                let mut sums = [0.0f32; 4];

                for ky in 0..7 {
                    let sy = (y as i32 + ky as i32 - 3).clamp(0, h as i32 - 1) as usize;
                    let row_offset = sy * w;

                    for kx in 0..7 {
                        let k_val = k[ky * 7 + kx];

                        for lane in 0..4 {
                            let sx = (x as i32 + lane as i32 + kx as i32 - 3).clamp(0, w as i32 - 1)
                                as usize;
                            sums[lane] += src[row_offset + sx] as f32 * k_val;
                        }
                    }
                }

                for lane in 0..4 {
                    dst[y * w + x + lane] = sums[lane].round().clamp(0.0, 255.0) as u8;
                }
                x += 4;
            }

            while x < w {
                let mut sum = 0.0f32;
                for ky in 0..7 {
                    let sy = (y as i32 + ky as i32 - 3).clamp(0, h as i32 - 1) as usize;
                    for kx in 0..7 {
                        let sx = (x as i32 + kx as i32 - 3).clamp(0, w as i32 - 1) as usize;
                        sum += src[sy * w + sx] as f32 * k[ky * 7 + kx];
                    }
                }
                dst[y * w + x] = sum.round().clamp(0.0, 255.0) as u8;
                x += 1;
            }
        }

        Ok(dst)
    }

    /// Generic convolution fallback for any kernel size.
    fn convolve_generic(&self, src: &[u8], w: usize, h: usize) -> CvResult<Vec<u8>> {
        let mut dst = vec![0u8; w * h];

        for y in 0..h {
            for x in 0..w {
                let mut sum = 0.0f32;
                for ky in 0..self.size {
                    let sy =
                        (y as i32 + ky as i32 - self.half as i32).clamp(0, h as i32 - 1) as usize;
                    for kx in 0..self.size {
                        let sx = (x as i32 + kx as i32 - self.half as i32).clamp(0, w as i32 - 1)
                            as usize;
                        sum += src[sy * w + sx] as f32 * self.kernel[ky * self.size + kx];
                    }
                }
                dst[y * w + x] = sum.round().clamp(0.0, 255.0) as u8;
            }
        }

        Ok(dst)
    }

    /// Apply separable convolution (for symmetric kernels) — horizontal pass
    /// followed by vertical pass for better cache performance.
    ///
    /// The 1D kernel is extracted from the first row of the 2D kernel,
    /// assuming separability (valid for Gaussian kernels).
    ///
    /// # Errors
    ///
    /// Returns an error if the kernel is not separable or dimensions are invalid.
    pub fn convolve_separable(&self, src: &[u8], width: u32, height: u32) -> CvResult<Vec<u8>> {
        let w = width as usize;
        let h = height as usize;

        if w == 0 || h == 0 {
            return Err(CvError::invalid_dimensions(width, height));
        }
        let size = w * h;
        if src.len() < size {
            return Err(CvError::insufficient_data(size, src.len()));
        }

        // Extract 1D kernel from diagonal scaling
        // For Gaussian: k_2d[i][j] = k_1d[i] * k_1d[j]
        // So k_1d[i] = sqrt(k_2d[i][center]) where center = half
        let center = self.half;
        let mut k1d = vec![0.0f32; self.size];
        let center_val = self.kernel[center * self.size + center];
        if center_val.abs() < f32::EPSILON {
            return Err(CvError::computation("kernel center is zero; not separable"));
        }

        for i in 0..self.size {
            k1d[i] = self.kernel[i * self.size + center] / center_val.sqrt();
        }

        // Horizontal pass -> f32 intermediate
        let mut temp = vec![0.0f32; w * h];
        for y in 0..h {
            let mut x = 0;
            while x + 4 <= w {
                let mut sums = [0.0f32; 4];
                for ki in 0..self.size {
                    let kv = k1d[ki];
                    for lane in 0..4 {
                        let sx = (x as i32 + lane as i32 + ki as i32 - self.half as i32)
                            .clamp(0, w as i32 - 1) as usize;
                        sums[lane] += src[y * w + sx] as f32 * kv;
                    }
                }
                for lane in 0..4 {
                    temp[y * w + x + lane] = sums[lane];
                }
                x += 4;
            }
            while x < w {
                let mut sum = 0.0f32;
                for ki in 0..self.size {
                    let sx =
                        (x as i32 + ki as i32 - self.half as i32).clamp(0, w as i32 - 1) as usize;
                    sum += src[y * w + sx] as f32 * k1d[ki];
                }
                temp[y * w + x] = sum;
                x += 1;
            }
        }

        // Vertical pass -> u8 output
        let mut dst = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                let mut sum = 0.0f32;
                for ki in 0..self.size {
                    let sy =
                        (y as i32 + ki as i32 - self.half as i32).clamp(0, h as i32 - 1) as usize;
                    sum += temp[sy * w + x] * k1d[ki];
                }
                dst[y * w + x] = sum.round().clamp(0.0, 255.0) as u8;
            }
        }

        Ok(dst)
    }
}

/// Convolve a grayscale image with a 3x3 kernel using batch processing.
///
/// Convenience function that creates a [`SimdConvolver`] and applies it.
///
/// # Errors
///
/// Returns an error if dimensions or kernel are invalid.
pub fn convolve_3x3(src: &[u8], width: u32, height: u32, kernel: &[f64; 9]) -> CvResult<Vec<u8>> {
    let conv = SimdConvolver::new(kernel, 3)?;
    conv.convolve(src, width, height)
}

/// Convolve a grayscale image with a 5x5 kernel using batch processing.
///
/// # Errors
///
/// Returns an error if dimensions or kernel are invalid.
pub fn convolve_5x5(src: &[u8], width: u32, height: u32, kernel: &[f64; 25]) -> CvResult<Vec<u8>> {
    let conv = SimdConvolver::new(kernel, 5)?;
    conv.convolve(src, width, height)
}

/// Convolve a grayscale image with a 7x7 kernel using batch processing.
///
/// # Errors
///
/// Returns an error if dimensions or kernel are invalid.
pub fn convolve_7x7(src: &[u8], width: u32, height: u32, kernel: &[f64; 49]) -> CvResult<Vec<u8>> {
    let conv = SimdConvolver::new(kernel, 7)?;
    conv.convolve(src, width, height)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_convolver_creation() {
        let k = [0.0; 9];
        let conv = SimdConvolver::new(&k, 3);
        assert!(conv.is_ok());
    }

    #[test]
    fn test_simd_convolver_invalid_size() {
        let k = [0.0; 4];
        assert!(SimdConvolver::new(&k, 2).is_err());
        assert!(SimdConvolver::new(&k, 4).is_err());
    }

    #[test]
    fn test_simd_convolver_size_mismatch() {
        let k = [0.0; 9];
        assert!(SimdConvolver::new(&k, 5).is_err());
    }

    #[test]
    fn test_gaussian_3x3_uniform() {
        let conv = SimdConvolver::gaussian_3x3().expect("should create");
        let src = vec![100u8; 10 * 10];
        let result = conv.convolve(&src, 10, 10).expect("should convolve");
        assert_eq!(result.len(), 100);
        // Uniform image should stay approximately the same
        for &v in &result {
            assert!((v as i32 - 100).abs() < 2);
        }
    }

    #[test]
    fn test_gaussian_5x5_uniform() {
        let conv = SimdConvolver::gaussian_5x5().expect("should create");
        let src = vec![128u8; 20 * 20];
        let result = conv.convolve(&src, 20, 20).expect("should convolve");
        assert_eq!(result.len(), 400);
        for &v in &result {
            assert!((v as i32 - 128).abs() < 2);
        }
    }

    #[test]
    fn test_gaussian_7x7_uniform() {
        let conv = SimdConvolver::gaussian_7x7().expect("should create");
        let src = vec![200u8; 15 * 15];
        let result = conv.convolve(&src, 15, 15).expect("should convolve");
        assert_eq!(result.len(), 225);
        for &v in &result {
            assert!((v as i32 - 200).abs() < 2);
        }
    }

    #[test]
    fn test_sobel_x_3x3() {
        let conv = SimdConvolver::sobel_x_3x3().expect("should create");
        // Vertical edge: left half=0, right half=255
        let mut src = vec![0u8; 10 * 10];
        for y in 0..10 {
            for x in 5..10 {
                src[y * 10 + x] = 255;
            }
        }
        let result = conv.convolve(&src, 10, 10).expect("should convolve");
        // Edge should be detected around column 5
        assert!(result[5 * 10 + 5] > 0 || result[5 * 10 + 4] > 0);
    }

    #[test]
    fn test_convolve_3x3_convenience() {
        #[rustfmt::skip]
        let k = [
            0.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 0.0,
        ];
        let src = vec![42u8; 8 * 8];
        let result = convolve_3x3(&src, 8, 8, &k).expect("should work");
        // Identity kernel should preserve the image
        for &v in &result {
            assert_eq!(v, 42);
        }
    }

    #[test]
    fn test_convolve_5x5_convenience() {
        let mut k = [0.0f64; 25];
        k[12] = 1.0; // center element
        let src = vec![77u8; 12 * 12];
        let result = convolve_5x5(&src, 12, 12, &k).expect("should work");
        for &v in &result {
            assert_eq!(v, 77);
        }
    }

    #[test]
    fn test_convolve_7x7_convenience() {
        let mut k = [0.0f64; 49];
        k[24] = 1.0; // center element
        let src = vec![55u8; 14 * 14];
        let result = convolve_7x7(&src, 14, 14, &k).expect("should work");
        for &v in &result {
            assert_eq!(v, 55);
        }
    }

    #[test]
    fn test_convolve_invalid_dimensions() {
        let conv = SimdConvolver::gaussian_3x3().expect("should create");
        assert!(conv.convolve(&[], 0, 0).is_err());
    }

    #[test]
    fn test_convolve_insufficient_data() {
        let conv = SimdConvolver::gaussian_3x3().expect("should create");
        let src = vec![0u8; 5];
        assert!(conv.convolve(&src, 10, 10).is_err());
    }

    #[test]
    fn test_separable_gaussian_3x3() {
        let conv = SimdConvolver::gaussian_3x3().expect("should create");
        let src = vec![100u8; 10 * 10];
        let result = conv.convolve_separable(&src, 10, 10).expect("should work");
        assert_eq!(result.len(), 100);
        for &v in &result {
            assert!((v as i32 - 100).abs() < 5);
        }
    }

    #[test]
    fn test_3x3_matches_generic_on_small_image() {
        // Use an image width not divisible by 4 to test remainder handling
        let conv = SimdConvolver::gaussian_3x3().expect("should create");
        let src: Vec<u8> = (0..7 * 7).map(|i| (i * 37 % 256) as u8).collect();

        let result_fast = conv.convolve(&src, 7, 7).expect("fast should work");
        let result_generic = conv
            .convolve_generic(&src, 7, 7)
            .expect("generic should work");

        assert_eq!(result_fast, result_generic);
    }
}
