//! Flat-vector 2D frame type for internal algorithm processing.
//!
//! `Frame2D` replaces any ndarray dependency with a simple stride-based
//! `Vec<f32>` layout compatible with SIMD and zero-copy slice views.

use crate::{DenoiseError, DenoiseResult};

/// A 2D single-channel floating-point frame stored in row-major order.
///
/// Pixels are stored densely: `data[y * width + x]`.
/// This eliminates any external ndarray dependency while remaining
/// compatible with SIMD code paths that require contiguous memory.
#[derive(Clone, Debug)]
pub struct Frame2D {
    /// Row-major pixel data.
    pub data: Vec<f32>,
    /// Frame width in pixels.
    pub width: usize,
    /// Frame height in pixels.
    pub height: usize,
}

impl Frame2D {
    /// Construct a zero-filled `Frame2D`.
    ///
    /// # Errors
    ///
    /// Returns [`DenoiseError::InvalidConfig`] if `width` or `height` is zero.
    pub fn new(width: usize, height: usize) -> DenoiseResult<Self> {
        if width == 0 || height == 0 {
            return Err(DenoiseError::InvalidConfig(
                "Frame2D dimensions must be non-zero".to_string(),
            ));
        }
        Ok(Self {
            data: vec![0.0f32; width * height],
            width,
            height,
        })
    }

    /// Construct a `Frame2D` filled with `value`.
    ///
    /// # Errors
    ///
    /// Returns [`DenoiseError::InvalidConfig`] if dimensions are zero.
    pub fn filled(width: usize, height: usize, value: f32) -> DenoiseResult<Self> {
        if width == 0 || height == 0 {
            return Err(DenoiseError::InvalidConfig(
                "Frame2D dimensions must be non-zero".to_string(),
            ));
        }
        Ok(Self {
            data: vec![value; width * height],
            width,
            height,
        })
    }

    /// Construct from an existing `Vec<f32>`.
    ///
    /// # Errors
    ///
    /// Returns [`DenoiseError::InvalidConfig`] if `data.len() != width * height`
    /// or if dimensions are zero.
    pub fn from_vec(data: Vec<f32>, width: usize, height: usize) -> DenoiseResult<Self> {
        if width == 0 || height == 0 {
            return Err(DenoiseError::InvalidConfig(
                "Frame2D dimensions must be non-zero".to_string(),
            ));
        }
        if data.len() != width * height {
            return Err(DenoiseError::InvalidConfig(format!(
                "data length {} does not match {}x{}={}",
                data.len(),
                width,
                height,
                width * height
            )));
        }
        Ok(Self {
            data,
            width,
            height,
        })
    }

    /// Construct from a raw `u8` byte plane (e.g. from `VideoFrame`).
    ///
    /// Each `u8` is converted to `f32` in the range `[0.0, 255.0]`.
    /// `stride` is the number of bytes per row (may differ from `width`).
    ///
    /// # Errors
    ///
    /// Returns [`DenoiseError::InvalidConfig`] if dimensions are zero or the
    /// slice is too short for the given `stride`.
    pub fn from_u8_plane(
        data: &[u8],
        width: usize,
        height: usize,
        stride: usize,
    ) -> DenoiseResult<Self> {
        if width == 0 || height == 0 {
            return Err(DenoiseError::InvalidConfig(
                "Frame2D dimensions must be non-zero".to_string(),
            ));
        }
        let required = stride * height;
        if data.len() < required {
            return Err(DenoiseError::InvalidConfig(format!(
                "plane slice too short: need {required} bytes, got {}",
                data.len()
            )));
        }
        let mut out = vec![0.0f32; width * height];
        for y in 0..height {
            for x in 0..width {
                out[y * width + x] = f32::from(data[y * stride + x]);
            }
        }
        Ok(Self {
            data: out,
            width,
            height,
        })
    }

    /// Export back to a `u8` byte plane with given `stride`.
    ///
    /// Values are clamped to `[0.0, 255.0]` and rounded before conversion.
    ///
    /// # Errors
    ///
    /// Returns [`DenoiseError::ProcessingError`] if `out.len()` is too short.
    pub fn to_u8_plane(&self, out: &mut [u8], stride: usize) -> DenoiseResult<()> {
        let required = stride * self.height;
        if out.len() < required {
            return Err(DenoiseError::ProcessingError(format!(
                "output slice too short: need {required}, got {}",
                out.len()
            )));
        }
        for y in 0..self.height {
            for x in 0..self.width {
                let v = self.data[y * self.width + x];
                out[y * stride + x] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
        Ok(())
    }

    /// Get pixel value at `(y, x)`.
    ///
    /// Returns `0.0` if coordinates are out of bounds.
    #[inline]
    #[must_use]
    pub fn get(&self, y: usize, x: usize) -> f32 {
        if y < self.height && x < self.width {
            self.data[y * self.width + x]
        } else {
            0.0
        }
    }

    /// Get pixel value with clamped boundary — useful for filter kernels.
    #[inline]
    #[must_use]
    pub fn get_clamped(&self, y: i32, x: i32) -> f32 {
        let cy = y.clamp(0, (self.height as i32) - 1) as usize;
        let cx = x.clamp(0, (self.width as i32) - 1) as usize;
        self.data[cy * self.width + cx]
    }

    /// Set pixel value at `(y, x)`.
    ///
    /// No-op if coordinates are out of bounds.
    #[inline]
    pub fn set(&mut self, y: usize, x: usize, value: f32) {
        if y < self.height && x < self.width {
            self.data[y * self.width + x] = value;
        }
    }

    /// Accumulate `value` into pixel at `(y, x)`.
    #[inline]
    pub fn add(&mut self, y: usize, x: usize, value: f32) {
        if y < self.height && x < self.width {
            self.data[y * self.width + x] += value;
        }
    }

    /// Number of pixels.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.width * self.height
    }

    /// Returns `true` if the frame contains no pixels.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Compute per-pixel PSNR (in dB) between `self` and `other`.
    ///
    /// Uses 8-bit dynamic range (MAX = 255).
    ///
    /// Returns `f64::INFINITY` if frames are identical (MSE == 0).
    ///
    /// # Errors
    ///
    /// Returns [`DenoiseError::InvalidConfig`] if dimensions do not match.
    pub fn psnr(&self, other: &Frame2D) -> DenoiseResult<f64> {
        if self.width != other.width || self.height != other.height {
            return Err(DenoiseError::InvalidConfig(
                "PSNR: frame dimensions must match".to_string(),
            ));
        }
        let mse: f64 = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| {
                let d = f64::from(a) - f64::from(b);
                d * d
            })
            .sum::<f64>()
            / (self.len() as f64);

        if mse < f64::EPSILON {
            return Ok(f64::INFINITY);
        }
        Ok(10.0 * (255.0_f64 * 255.0 / mse).log10())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame2d_new() {
        let f = Frame2D::new(8, 4).expect("valid dims");
        assert_eq!(f.width, 8);
        assert_eq!(f.height, 4);
        assert_eq!(f.len(), 32);
        assert!(f.data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_frame2d_zero_dims_error() {
        assert!(Frame2D::new(0, 4).is_err());
        assert!(Frame2D::new(4, 0).is_err());
    }

    #[test]
    fn test_frame2d_filled() {
        let f = Frame2D::filled(4, 4, 128.0).expect("valid dims");
        assert!(f.data.iter().all(|&v| (v - 128.0).abs() < f32::EPSILON));
    }

    #[test]
    fn test_frame2d_from_vec_length_mismatch() {
        let result = Frame2D::from_vec(vec![1.0; 10], 4, 4);
        assert!(result.is_err());
    }

    #[test]
    fn test_frame2d_get_set() {
        let mut f = Frame2D::new(4, 4).expect("valid dims");
        f.set(2, 3, 42.0);
        assert!((f.get(2, 3) - 42.0).abs() < f32::EPSILON);
        assert!((f.get(0, 0) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_frame2d_get_out_of_bounds() {
        let f = Frame2D::new(4, 4).expect("valid dims");
        assert!((f.get(10, 10) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_frame2d_get_clamped() {
        let mut f = Frame2D::new(4, 4).expect("valid dims");
        f.set(0, 0, 99.0);
        // Negative coords clamp to (0,0)
        assert!((f.get_clamped(-1, -1) - 99.0).abs() < f32::EPSILON);
        // Large coords clamp to last pixel
        f.set(3, 3, 77.0);
        assert!((f.get_clamped(100, 100) - 77.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_frame2d_u8_roundtrip() {
        let data: Vec<u8> = (0u8..64).collect();
        let f = Frame2D::from_u8_plane(&data, 8, 8, 8).expect("valid");
        let mut out = vec![0u8; 8 * 8];
        f.to_u8_plane(&mut out, 8).expect("export ok");
        assert_eq!(out, data);
    }

    #[test]
    fn test_frame2d_u8_plane_too_short() {
        let data = vec![0u8; 5]; // too short for 4x4 stride=4
        let result = Frame2D::from_u8_plane(&data, 4, 4, 4);
        assert!(result.is_err());
    }

    #[test]
    fn test_frame2d_psnr_identical() {
        let a = Frame2D::filled(8, 8, 128.0).expect("valid");
        let b = Frame2D::filled(8, 8, 128.0).expect("valid");
        let p = a.psnr(&b).expect("psnr ok");
        assert!(p.is_infinite(), "identical frames => infinite PSNR");
    }

    #[test]
    fn test_frame2d_psnr_different() {
        let a = Frame2D::filled(8, 8, 128.0).expect("valid");
        let b = Frame2D::filled(8, 8, 100.0).expect("valid");
        let p = a.psnr(&b).expect("psnr ok");
        assert!(p.is_finite());
        assert!(p > 0.0);
    }

    #[test]
    fn test_frame2d_psnr_dim_mismatch() {
        let a = Frame2D::new(4, 4).expect("valid");
        let b = Frame2D::new(8, 4).expect("valid");
        assert!(a.psnr(&b).is_err());
    }

    #[test]
    fn test_frame2d_add() {
        let mut f = Frame2D::filled(4, 4, 10.0).expect("valid");
        f.add(2, 2, 5.0);
        assert!((f.get(2, 2) - 15.0).abs() < f32::EPSILON);
    }
}
