//! SIMD convolution operations
//!
//! 1D and 2D convolution kernels, separable filters, and border handling
//! for image processing pipelines.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]

/// Border handling strategy for convolution
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BorderMode {
    /// Clamp to nearest edge pixel
    Clamp,
    /// Reflect about border (mirror)
    Reflect,
    /// Wrap around (tiling)
    Wrap,
    /// Use constant value (default: 0)
    Constant(u8),
}

/// A fixed-size convolution kernel stored as `f32` weights.
#[derive(Debug, Clone)]
pub struct Kernel {
    weights: Vec<f32>,
    width: usize,
    height: usize,
}

impl Kernel {
    /// Create a new kernel from row-major weights.
    ///
    /// # Errors
    /// Returns an error if `weights.len() != width * height`.
    pub fn new(weights: Vec<f32>, width: usize, height: usize) -> Result<Self, String> {
        if weights.len() != width * height {
            return Err(format!(
                "weights.len()={} != width*height={}",
                weights.len(),
                width * height
            ));
        }
        Ok(Self {
            weights,
            width,
            height,
        })
    }

    /// 3x3 Gaussian blur kernel (normalized)
    #[must_use]
    pub fn gaussian_3x3() -> Self {
        Self {
            weights: vec![
                1.0 / 16.0,
                2.0 / 16.0,
                1.0 / 16.0,
                2.0 / 16.0,
                4.0 / 16.0,
                2.0 / 16.0,
                1.0 / 16.0,
                2.0 / 16.0,
                1.0 / 16.0,
            ],
            width: 3,
            height: 3,
        }
    }

    /// 3x3 Laplacian edge detection kernel
    #[must_use]
    pub fn laplacian_3x3() -> Self {
        Self {
            weights: vec![0.0, -1.0, 0.0, -1.0, 4.0, -1.0, 0.0, -1.0, 0.0],
            width: 3,
            height: 3,
        }
    }

    /// 3x3 Sharpen kernel
    #[must_use]
    pub fn sharpen_3x3() -> Self {
        Self {
            weights: vec![0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0],
            width: 3,
            height: 3,
        }
    }

    /// Identity kernel (1x1)
    #[must_use]
    pub fn identity() -> Self {
        Self {
            weights: vec![1.0],
            width: 1,
            height: 1,
        }
    }

    /// Returns the kernel width
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Returns the kernel height
    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Returns the kernel weights
    #[must_use]
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }
}

/// Get a pixel value with border handling
#[allow(clippy::too_many_arguments)]
fn get_pixel(src: &[u8], width: usize, height: usize, x: i32, y: i32, border: BorderMode) -> u8 {
    let clamped_x = clamp_coord(x, width as i32, border);
    let clamped_y = clamp_coord(y, height as i32, border);
    match (clamped_x, clamped_y) {
        (Some(cx), Some(cy)) => src[cy as usize * width + cx as usize],
        _ => {
            if let BorderMode::Constant(v) = border {
                v
            } else {
                0
            }
        }
    }
}

fn clamp_coord(v: i32, size: i32, border: BorderMode) -> Option<i32> {
    if v >= 0 && v < size {
        return Some(v);
    }
    match border {
        BorderMode::Clamp => Some(v.clamp(0, size - 1)),
        BorderMode::Reflect => {
            let mut cv = v;
            if cv < 0 {
                cv = -cv - 1;
            } else {
                cv = 2 * size - cv - 1;
            }
            Some(cv.clamp(0, size - 1))
        }
        BorderMode::Wrap => Some(v.rem_euclid(size)),
        BorderMode::Constant(_) => None,
    }
}

/// Apply a 2D convolution kernel to a grayscale image.
///
/// # Errors
/// Returns an error if buffer lengths don't equal `width * height`.
pub fn convolve2d(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    kernel: &Kernel,
    border: BorderMode,
) -> Result<(), String> {
    if src.len() != width * height || dst.len() != width * height {
        return Err("Buffer length must equal width * height".to_string());
    }
    let kw = kernel.width as i32;
    let kh = kernel.height as i32;
    let kw_half = kw / 2;
    let kh_half = kh / 2;

    for y in 0..height {
        for x in 0..width {
            let mut acc: f32 = 0.0;
            for ky in 0..kh {
                for kx in 0..kw {
                    let py = y as i32 + ky - kh_half;
                    let px = x as i32 + kx - kw_half;
                    let weight = kernel.weights[(ky * kw + kx) as usize];
                    let pixel = get_pixel(src, width, height, px, py, border);
                    acc += f32::from(pixel) * weight;
                }
            }
            dst[y * width + x] = acc.clamp(0.0, 255.0) as u8;
        }
    }
    Ok(())
}

/// Separable 1D convolution along the horizontal axis.
///
/// # Errors
/// Returns an error if buffer lengths don't equal `width * height`.
pub fn convolve1d_horizontal(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    kernel: &[f32],
    border: BorderMode,
) -> Result<(), String> {
    if src.len() != width * height || dst.len() != width * height {
        return Err("Buffer length must equal width * height".to_string());
    }
    let klen = kernel.len() as i32;
    let half = klen / 2;
    for y in 0..height {
        for x in 0..width {
            let mut acc: f32 = 0.0;
            for k in 0..klen {
                let px = x as i32 + k - half;
                let pixel = get_pixel(src, width, height, px, y as i32, border);
                acc += f32::from(pixel) * kernel[k as usize];
            }
            dst[y * width + x] = acc.clamp(0.0, 255.0) as u8;
        }
    }
    Ok(())
}

/// Separable 1D convolution along the vertical axis.
///
/// # Errors
/// Returns an error if buffer lengths don't equal `width * height`.
pub fn convolve1d_vertical(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    kernel: &[f32],
    border: BorderMode,
) -> Result<(), String> {
    if src.len() != width * height || dst.len() != width * height {
        return Err("Buffer length must equal width * height".to_string());
    }
    let klen = kernel.len() as i32;
    let half = klen / 2;
    for y in 0..height {
        for x in 0..width {
            let mut acc: f32 = 0.0;
            for k in 0..klen {
                let py = y as i32 + k - half;
                let pixel = get_pixel(src, width, height, x as i32, py, border);
                acc += f32::from(pixel) * kernel[k as usize];
            }
            dst[y * width + x] = acc.clamp(0.0, 255.0) as u8;
        }
    }
    Ok(())
}

/// Apply a separable Gaussian blur via two 1D passes.
///
/// # Errors
/// Returns an error if dimensions are inconsistent.
pub fn separable_gaussian_blur(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    border: BorderMode,
) -> Result<(), String> {
    // [1, 2, 1] / 4 normalized Gaussian 1D kernel
    let kernel = [0.25f32, 0.5, 0.25];
    let mut tmp = vec![0u8; width * height];
    convolve1d_horizontal(src, &mut tmp, width, height, &kernel, border)?;
    convolve1d_vertical(&tmp, dst, width, height, &kernel, border)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_new_valid() {
        let k = Kernel::new(vec![1.0; 9], 3, 3).expect("should succeed in test");
        assert_eq!(k.width(), 3);
        assert_eq!(k.height(), 3);
    }

    #[test]
    fn test_kernel_new_invalid() {
        assert!(Kernel::new(vec![1.0; 8], 3, 3).is_err());
    }

    #[test]
    fn test_identity_kernel() {
        let src = vec![100u8, 150, 200, 50];
        let mut dst = vec![0u8; 4];
        let k = Kernel::identity();
        convolve2d(&src, &mut dst, 2, 2, &k, BorderMode::Clamp).expect("should succeed in test");
        assert_eq!(src, dst);
    }

    #[test]
    fn test_gaussian_3x3_preserves_uniform() {
        // A uniform image should be unchanged by Gaussian blur
        let src = vec![128u8; 9];
        let mut dst = vec![0u8; 9];
        let k = Kernel::gaussian_3x3();
        convolve2d(&src, &mut dst, 3, 3, &k, BorderMode::Clamp).expect("should succeed in test");
        for &v in &dst {
            assert!((i32::from(v) - 128).abs() <= 1);
        }
    }

    #[test]
    fn test_laplacian_detects_edge() {
        // Flat areas should produce 0 with Laplacian; we just check it runs without error
        let src = vec![0u8; 25];
        let mut dst = vec![0u8; 25];
        let k = Kernel::laplacian_3x3();
        convolve2d(&src, &mut dst, 5, 5, &k, BorderMode::Constant(0))
            .expect("should succeed in test");
        assert!(dst.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_sharpen_kernel_predefined() {
        let k = Kernel::sharpen_3x3();
        assert_eq!(k.weights().len(), 9);
        // Center weight should be 5.0
        assert!((k.weights()[4] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_convolve2d_buffer_mismatch() {
        let src = vec![0u8; 6];
        let mut dst = vec![0u8; 9];
        let k = Kernel::identity();
        assert!(convolve2d(&src, &mut dst, 3, 3, &k, BorderMode::Clamp).is_err());
    }

    #[test]
    fn test_convolve1d_horizontal_identity() {
        let src = vec![10u8, 20, 30, 40, 50, 60, 70, 80, 90];
        let mut dst = vec![0u8; 9];
        // Identity 1D kernel: [1.0]
        convolve1d_horizontal(&src, &mut dst, 3, 3, &[1.0], BorderMode::Clamp)
            .expect("should succeed in test");
        assert_eq!(src, dst);
    }

    #[test]
    fn test_convolve1d_vertical_identity() {
        let src = vec![10u8, 20, 30, 40, 50, 60, 70, 80, 90];
        let mut dst = vec![0u8; 9];
        convolve1d_vertical(&src, &mut dst, 3, 3, &[1.0], BorderMode::Clamp)
            .expect("should succeed in test");
        assert_eq!(src, dst);
    }

    #[test]
    fn test_separable_gaussian_uniform() {
        let src = vec![200u8; 25];
        let mut dst = vec![0u8; 25];
        separable_gaussian_blur(&src, &mut dst, 5, 5, BorderMode::Clamp)
            .expect("should succeed in test");
        for &v in &dst {
            assert!((i32::from(v) - 200).abs() <= 1);
        }
    }

    #[test]
    fn test_border_clamp() {
        // Getting pixel at x=-1 with clamp should return x=0
        let src = vec![42u8; 4];
        let v = get_pixel(&src, 2, 2, -1, 0, BorderMode::Clamp);
        assert_eq!(v, 42);
    }

    #[test]
    fn test_border_constant() {
        let src = vec![42u8; 4];
        let v = get_pixel(&src, 2, 2, -1, 0, BorderMode::Constant(99));
        assert_eq!(v, 99);
    }

    #[test]
    fn test_border_wrap() {
        let src = vec![10u8, 20, 30, 40];
        // x=2 wraps to x=0 in width=2
        let v = get_pixel(&src, 2, 2, 2, 0, BorderMode::Wrap);
        assert_eq!(v, 10);
    }

    #[test]
    fn test_convolve1d_h_buffer_mismatch() {
        let src = vec![0u8; 4];
        let mut dst = vec![0u8; 9];
        assert!(convolve1d_horizontal(&src, &mut dst, 3, 3, &[1.0], BorderMode::Clamp).is_err());
    }

    #[test]
    fn test_convolve2d_reflect_border() {
        let src = vec![100u8; 9];
        let mut dst = vec![0u8; 9];
        let k = Kernel::gaussian_3x3();
        convolve2d(&src, &mut dst, 3, 3, &k, BorderMode::Reflect).expect("should succeed in test");
        for &v in &dst {
            assert!((i32::from(v) - 100).abs() <= 2);
        }
    }
}
