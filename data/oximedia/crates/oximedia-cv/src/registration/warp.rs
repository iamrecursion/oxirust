//! Image warping utilities for registration.
//!
//! Provides bilinear and bicubic interpolation for image warping.

use super::TransformMatrix;
use crate::error::{CvError, CvResult};

/// Interpolation method for image warping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationMethod {
    /// Nearest neighbor (fastest, lowest quality).
    NearestNeighbor,
    /// Bilinear interpolation (good balance).
    Bilinear,
    /// Bicubic interpolation (highest quality).
    Bicubic,
}

/// Warp a grayscale image using a transform matrix.
///
/// # Errors
///
/// Returns an error if dimensions are invalid.
pub fn warp_grayscale(
    image: &[u8],
    width: u32,
    height: u32,
    transform: &TransformMatrix,
    method: InterpolationMethod,
) -> CvResult<Vec<u8>> {
    if width == 0 || height == 0 {
        return Err(CvError::invalid_dimensions(width, height));
    }

    let size = (width as usize) * (height as usize);
    if image.len() < size {
        return Err(CvError::insufficient_data(size, image.len()));
    }

    let inv = transform
        .inverse()
        .unwrap_or_else(|_| TransformMatrix::identity());

    let w = width as usize;
    let h = height as usize;
    let mut output = vec![0u8; size];

    for y in 0..h {
        for x in 0..w {
            let (sx, sy) = inv.transform_point(x as f64, y as f64);

            let val = match method {
                InterpolationMethod::NearestNeighbor => {
                    let sx = sx.round() as i64;
                    let sy = sy.round() as i64;
                    if sx >= 0 && sx < w as i64 && sy >= 0 && sy < h as i64 {
                        image[sy as usize * w + sx as usize] as f64
                    } else {
                        0.0
                    }
                }
                InterpolationMethod::Bilinear => bilinear_sample(image, w, h, sx, sy),
                InterpolationMethod::Bicubic => bicubic_sample(image, w, h, sx, sy),
            };

            output[y * w + x] = val.round().clamp(0.0, 255.0) as u8;
        }
    }

    Ok(output)
}

/// Bilinear sampling.
fn bilinear_sample(image: &[u8], width: usize, height: usize, x: f64, y: f64) -> f64 {
    if x < 0.0 || x >= (width - 1) as f64 || y < 0.0 || y >= (height - 1) as f64 {
        return 0.0;
    }

    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let fx = x - x0 as f64;
    let fy = y - y0 as f64;

    let v00 = image[y0 * width + x0] as f64;
    let v01 = image[y0 * width + x1] as f64;
    let v10 = image[y1 * width + x0] as f64;
    let v11 = image[y1 * width + x1] as f64;

    v00 * (1.0 - fx) * (1.0 - fy) + v01 * fx * (1.0 - fy) + v10 * (1.0 - fx) * fy + v11 * fx * fy
}

/// Bicubic interpolation kernel.
fn cubic_weight(t: f64) -> f64 {
    let t = t.abs();
    if t <= 1.0 {
        (1.5 * t - 2.5) * t * t + 1.0
    } else if t <= 2.0 {
        ((-0.5 * t + 2.5) * t - 4.0) * t + 2.0
    } else {
        0.0
    }
}

/// Bicubic sampling.
fn bicubic_sample(image: &[u8], width: usize, height: usize, x: f64, y: f64) -> f64 {
    if x < 1.0 || x >= (width - 2) as f64 || y < 1.0 || y >= (height - 2) as f64 {
        return bilinear_sample(image, width, height, x, y);
    }

    let xi = x.floor() as i64;
    let yi = y.floor() as i64;
    let fx = x - xi as f64;
    let fy = y - yi as f64;

    let mut result = 0.0;

    for dy in -1i64..=2 {
        let wy = cubic_weight(fy - dy as f64);
        let sy = (yi + dy) as usize;

        for dx in -1i64..=2 {
            let wx = cubic_weight(fx - dx as f64);
            let sx = (xi + dx) as usize;

            if sx < width && sy < height {
                result += image[sy * width + sx] as f64 * wx * wy;
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_warp_identity() {
        let image = vec![128u8; 50 * 50];
        let result = warp_grayscale(
            &image,
            50,
            50,
            &TransformMatrix::identity(),
            InterpolationMethod::Bilinear,
        )
        .expect("should succeed");

        // Center pixels should be preserved
        assert_eq!(result[25 * 50 + 25], 128);
    }

    #[test]
    fn test_warp_nearest_neighbor() {
        let image = vec![200u8; 50 * 50];
        let result = warp_grayscale(
            &image,
            50,
            50,
            &TransformMatrix::identity(),
            InterpolationMethod::NearestNeighbor,
        )
        .expect("should succeed");
        assert_eq!(result[25 * 50 + 25], 200);
    }

    #[test]
    fn test_warp_bicubic() {
        let image = vec![100u8; 50 * 50];
        let result = warp_grayscale(
            &image,
            50,
            50,
            &TransformMatrix::identity(),
            InterpolationMethod::Bicubic,
        )
        .expect("should succeed");
        assert_eq!(result.len(), 50 * 50);
    }

    #[test]
    fn test_warp_invalid_dimensions() {
        let result = warp_grayscale(
            &[],
            0,
            0,
            &TransformMatrix::identity(),
            InterpolationMethod::Bilinear,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_cubic_weight() {
        assert!((cubic_weight(0.0) - 1.0).abs() < 1e-6);
        assert!(cubic_weight(2.5).abs() < 1e-6);
    }
}
