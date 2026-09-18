//! Enhanced Correlation Coefficient (ECC) registration.
//!
//! Iterative optimization of image alignment using the ECC criterion.

use super::{RegistrationQuality, TransformMatrix, TransformationType};
use crate::error::{CvError, CvResult};

/// Register two images using the ECC algorithm.
///
/// # Errors
///
/// Returns an error if registration fails.
pub fn register_ecc(
    reference: &[u8],
    target: &[u8],
    width: u32,
    height: u32,
    transform_type: TransformationType,
    max_iterations: usize,
    convergence_threshold: f64,
) -> CvResult<(TransformMatrix, RegistrationQuality)> {
    let size = (width as usize) * (height as usize);
    if reference.len() < size || target.len() < size {
        return Err(CvError::insufficient_data(
            size,
            reference.len().min(target.len()),
        ));
    }

    // Normalize images to f64
    let ref_f: Vec<f64> = reference[..size]
        .iter()
        .map(|&v| v as f64 / 255.0)
        .collect();
    let tgt_f: Vec<f64> = target[..size].iter().map(|&v| v as f64 / 255.0).collect();

    let mut transform = TransformMatrix::identity();
    let mut prev_ecc = 0.0;
    let mut iterations = 0;

    for iter in 0..max_iterations {
        iterations = iter + 1;

        // Compute warped target
        let warped = warp_image(&tgt_f, width, height, &transform);

        // Compute ECC criterion
        let ecc = compute_ecc(&ref_f, &warped);

        // Check convergence
        if (ecc - prev_ecc).abs() < convergence_threshold && iter > 0 {
            break;
        }
        prev_ecc = ecc;

        // Compute gradient-based update
        let update = compute_ecc_update(&ref_f, &warped, width, height, transform_type);
        transform = transform.compose(&update);
    }

    let quality = RegistrationQuality {
        success: prev_ecc > 0.5,
        rmse: 1.0 - prev_ecc,
        inliers: size,
        confidence: prev_ecc.clamp(0.0, 1.0),
        iterations,
    };

    Ok((transform, quality))
}

/// Compute ECC criterion between two images.
fn compute_ecc(reference: &[f64], warped: &[f64]) -> f64 {
    let n = reference.len().min(warped.len());
    if n == 0 {
        return 0.0;
    }

    let mean_ref: f64 = reference[..n].iter().sum::<f64>() / n as f64;
    let mean_warp: f64 = warped[..n].iter().sum::<f64>() / n as f64;

    let mut num = 0.0;
    let mut den_ref = 0.0;
    let mut den_warp = 0.0;

    for i in 0..n {
        let dr = reference[i] - mean_ref;
        let dw = warped[i] - mean_warp;
        num += dr * dw;
        den_ref += dr * dr;
        den_warp += dw * dw;
    }

    let denom = (den_ref * den_warp).sqrt();
    if denom < 1e-12 {
        return 0.0;
    }

    (num / denom).clamp(-1.0, 1.0)
}

/// Compute ECC parameter update.
fn compute_ecc_update(
    reference: &[f64],
    warped: &[f64],
    width: u32,
    height: u32,
    _transform_type: TransformationType,
) -> TransformMatrix {
    let w = width as usize;
    let h = height as usize;
    let n = w * h;

    // Compute image gradients
    let mut dx_sum = 0.0;
    let mut dy_sum = 0.0;
    let mut weight_sum = 0.0;

    for y in 1..(h - 1) {
        for x in 1..(w - 1) {
            let idx = y * w + x;
            if idx >= n || idx >= reference.len() || idx >= warped.len() {
                continue;
            }

            let gx = (warped[idx + 1] - warped[idx - 1]) / 2.0;
            let gy = (warped[idx + w] - warped[idx - w]) / 2.0;
            let diff = reference[idx] - warped[idx];

            let weight = diff.abs().min(1.0);
            dx_sum += gx * diff * weight;
            dy_sum += gy * diff * weight;
            weight_sum += weight;
        }
    }

    let scale = if weight_sum > 1e-6 {
        0.1 / weight_sum
    } else {
        0.0
    };

    TransformMatrix::translation(dx_sum * scale, dy_sum * scale)
}

/// Warp image using transform matrix.
fn warp_image(image: &[f64], width: u32, height: u32, transform: &TransformMatrix) -> Vec<f64> {
    let w = width as usize;
    let h = height as usize;
    let mut output = vec![0.0; w * h];

    for y in 0..h {
        for x in 0..w {
            let (sx, sy) = transform.transform_point(x as f64, y as f64);
            let sx = sx.round() as i64;
            let sy = sy.round() as i64;

            if sx >= 0 && sx < w as i64 && sy >= 0 && sy < h as i64 {
                let src_idx = sy as usize * w + sx as usize;
                if src_idx < image.len() {
                    output[y * w + x] = image[src_idx];
                }
            }
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_ecc_identical() {
        // Use a gradient image so ECC correlation is meaningful (uniform images have zero variance)
        let mut image = vec![0u8; 100 * 100];
        for y in 0..100 {
            for x in 0..100 {
                image[y * 100 + x] = ((x * 2 + y) % 256) as u8;
            }
        }
        let (transform, quality) = register_ecc(
            &image,
            &image,
            100,
            100,
            TransformationType::Translation,
            10,
            1e-6,
        )
        .expect("should succeed");

        assert!(quality.success);
        let (tx, ty) = transform.get_translation();
        assert!(tx.abs() < 5.0);
        assert!(ty.abs() < 5.0);
    }

    #[test]
    fn test_compute_ecc_identical() {
        let img = vec![0.5; 100];
        let ecc = compute_ecc(&img, &img);
        // Constant images have zero variance, ECC = 0
        assert!(ecc.abs() < 0.01 || (ecc - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_compute_ecc_correlated() {
        let a: Vec<f64> = (0..100).map(|i| (i as f64) / 100.0).collect();
        let b: Vec<f64> = (0..100).map(|i| (i as f64) / 100.0 + 0.1).collect();
        let ecc = compute_ecc(&a, &b);
        assert!(ecc > 0.9);
    }

    #[test]
    fn test_warp_image_identity() {
        let image: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        let warped = warp_image(&image, 10, 10, &TransformMatrix::identity());
        assert_eq!(warped.len(), 100);
        for i in 0..100 {
            assert!((warped[i] - image[i]).abs() < 1e-6);
        }
    }
}
