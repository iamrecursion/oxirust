//! Optical flow-based registration.
//!
//! Uses dense optical flow estimation to register images.

use super::{RegistrationQuality, TransformMatrix, TransformationType};
use crate::error::{CvError, CvResult};

/// Register two images using optical flow.
///
/// # Errors
///
/// Returns an error if registration fails.
pub fn register_optical_flow(
    reference: &[u8],
    target: &[u8],
    width: u32,
    height: u32,
    transform_type: TransformationType,
) -> CvResult<(TransformMatrix, RegistrationQuality)> {
    let size = (width as usize) * (height as usize);
    if reference.len() < size || target.len() < size {
        return Err(CvError::insufficient_data(
            size,
            reference.len().min(target.len()),
        ));
    }

    let w = width as usize;
    let h = height as usize;

    // Compute dense flow using block matching
    let block_size = 16;
    let search_range = 32;
    let mut flow_x = Vec::new();
    let mut flow_y = Vec::new();
    let mut weights = Vec::new();

    for by in (0..h).step_by(block_size) {
        for bx in (0..w).step_by(block_size) {
            let bw = block_size.min(w - bx);
            let bh = block_size.min(h - by);

            if let Some((dx, dy, sad)) =
                match_block(reference, target, w, h, bx, by, bw, bh, search_range)
            {
                let weight = 1.0 / (1.0 + sad as f64);
                flow_x.push(dx as f64);
                flow_y.push(dy as f64);
                weights.push(weight);
            }
        }
    }

    if flow_x.is_empty() {
        return Err(CvError::matrix_error("no flow vectors computed"));
    }

    // Estimate transform from flow vectors
    let transform = match transform_type {
        TransformationType::Translation => {
            let total_w: f64 = weights.iter().sum();
            let tx: f64 = flow_x
                .iter()
                .zip(weights.iter())
                .map(|(dx, w)| dx * w)
                .sum::<f64>()
                / total_w;
            let ty: f64 = flow_y
                .iter()
                .zip(weights.iter())
                .map(|(dy, w)| dy * w)
                .sum::<f64>()
                / total_w;
            TransformMatrix::translation(tx, ty)
        }
        _ => {
            let total_w: f64 = weights.iter().sum();
            let tx: f64 = flow_x
                .iter()
                .zip(weights.iter())
                .map(|(dx, w)| dx * w)
                .sum::<f64>()
                / total_w;
            let ty: f64 = flow_y
                .iter()
                .zip(weights.iter())
                .map(|(dy, w)| dy * w)
                .sum::<f64>()
                / total_w;
            TransformMatrix::translation(tx, ty)
        }
    };

    let confidence = if weights.is_empty() {
        0.0
    } else {
        let avg_weight = weights.iter().sum::<f64>() / weights.len() as f64;
        (avg_weight * 100.0).clamp(0.0, 1.0)
    };

    let quality = RegistrationQuality {
        success: confidence > 0.3,
        rmse: 0.0,
        inliers: flow_x.len(),
        confidence,
        iterations: 1,
    };

    Ok((transform, quality))
}

/// Match a block using SAD (Sum of Absolute Differences).
fn match_block(
    reference: &[u8],
    target: &[u8],
    width: usize,
    height: usize,
    bx: usize,
    by: usize,
    bw: usize,
    bh: usize,
    search_range: usize,
) -> Option<(i32, i32, u32)> {
    let mut best_dx = 0i32;
    let mut best_dy = 0i32;
    let mut best_sad = u32::MAX;

    let sr = search_range as i32;

    for dy in -sr..=sr {
        for dx in -sr..=sr {
            let tx = bx as i32 + dx;
            let ty = by as i32 + dy;

            if tx < 0 || ty < 0 || (tx as usize + bw) > width || (ty as usize + bh) > height {
                continue;
            }

            let mut sad = 0u32;
            for row in 0..bh {
                for col in 0..bw {
                    let ref_idx = (by + row) * width + (bx + col);
                    let tgt_idx = (ty as usize + row) * width + (tx as usize + col);

                    if ref_idx < reference.len() && tgt_idx < target.len() {
                        sad += (reference[ref_idx] as i32 - target[tgt_idx] as i32).unsigned_abs();
                    }
                }
            }

            // Prefer smaller displacement on tie (favors zero motion)
            let curr_dist = dx.abs() + dy.abs();
            let best_dist = best_dx.abs() + best_dy.abs();
            if sad < best_sad || (sad == best_sad && curr_dist < best_dist) {
                best_sad = sad;
                best_dx = dx;
                best_dy = dy;
            }
        }
    }

    if best_sad < u32::MAX {
        Some((best_dx, best_dy, best_sad))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_identical() {
        let image = vec![128u8; 64 * 64];
        let (transform, quality) =
            register_optical_flow(&image, &image, 64, 64, TransformationType::Translation)
                .expect("should succeed");

        let (tx, ty) = transform.get_translation();
        assert!(tx.abs() < 2.0);
        assert!(ty.abs() < 2.0);
        assert!(quality.inliers > 0);
    }

    #[test]
    fn test_register_insufficient_data() {
        let result = register_optical_flow(
            &[0u8; 10],
            &[0u8; 10],
            100,
            100,
            TransformationType::Translation,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_match_block_identical() {
        let image = vec![128u8; 64 * 64];
        let result = match_block(&image, &image, 64, 64, 16, 16, 16, 16, 8);
        assert!(result.is_some());
        let (dx, dy, sad) = result.expect("should have match");
        assert_eq!(dx, 0);
        assert_eq!(dy, 0);
        assert_eq!(sad, 0);
    }
}
