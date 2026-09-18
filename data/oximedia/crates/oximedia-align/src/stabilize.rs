//! Video stabilization pipeline.
//!
//! Combines optical flow estimation, affine transform fitting, temporal
//! smoothing, and image warping to produce stabilized video output.
//!
//! # Pipeline stages
//!
//! 1. **Motion estimation**: Compute inter-frame motion using sparse KLT
//!    tracking (or dense Farneback flow).
//! 2. **Global motion model**: Fit an affine (or similarity) transform to the
//!    estimated motion vectors using RANSAC.
//! 3. **Trajectory smoothing**: Build a cumulative camera trajectory and smooth
//!    it with a configurable Gaussian filter to remove high-frequency jitter
//!    while preserving intentional camera movements.
//! 4. **Compensation**: Apply the smoothed correction to each frame via affine
//!    warping.

#![allow(clippy::cast_precision_loss)]

use crate::affine::AffineMatrix;
use crate::{AlignError, AlignResult};

/// Stabilization strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StabilizationMode {
    /// Correct only translation (2 DOF).
    Translation,
    /// Correct translation + rotation (3 DOF).
    TranslationRotation,
    /// Full affine correction (6 DOF).
    Affine,
}

/// Configuration for the stabilization pipeline.
#[derive(Debug, Clone)]
pub struct StabilizeConfig {
    /// Stabilization mode.
    pub mode: StabilizationMode,
    /// Gaussian smoothing radius (in frames). Larger values produce smoother
    /// output but can introduce more cropping/border artefacts.
    pub smooth_radius: usize,
    /// Standard deviation of the Gaussian kernel (in frames).
    pub smooth_sigma: f64,
    /// Maximum correction in pixels. Corrections exceeding this are clamped
    /// to prevent extreme warping.
    pub max_correction_px: f64,
    /// Border handling: percentage of the frame to crop on each side (0.0 to 0.5).
    pub crop_ratio: f64,
}

impl Default for StabilizeConfig {
    fn default() -> Self {
        Self {
            mode: StabilizationMode::TranslationRotation,
            smooth_radius: 15,
            smooth_sigma: 5.0,
            max_correction_px: 50.0,
            crop_ratio: 0.05,
        }
    }
}

/// A single frame's estimated global motion relative to the previous frame.
#[derive(Debug, Clone, Copy)]
pub struct FrameMotion {
    /// Translation X.
    pub dx: f64,
    /// Translation Y.
    pub dy: f64,
    /// Rotation angle (radians).
    pub da: f64,
    /// Scale factor (1.0 = no scaling).
    pub ds: f64,
}

impl FrameMotion {
    /// Create a new frame motion with all parameters.
    #[must_use]
    pub fn new(dx: f64, dy: f64, da: f64, ds: f64) -> Self {
        Self { dx, dy, da, ds }
    }

    /// Identity (no motion).
    #[must_use]
    pub fn identity() -> Self {
        Self {
            dx: 0.0,
            dy: 0.0,
            da: 0.0,
            ds: 1.0,
        }
    }

    /// Convert to an affine matrix.
    #[must_use]
    pub fn to_affine(&self) -> AffineMatrix {
        let c = (self.da.cos() * self.ds) as f32;
        let s = (self.da.sin() * self.ds) as f32;
        AffineMatrix {
            data: [
                [c, -s, self.dx as f32],
                [s, c, self.dy as f32],
                [0.0, 0.0, 1.0],
            ],
        }
    }
}

/// Cumulative trajectory: the sum of all frame motions up to a given frame.
#[derive(Debug, Clone, Copy)]
pub struct Trajectory {
    /// Cumulative X translation.
    pub x: f64,
    /// Cumulative Y translation.
    pub y: f64,
    /// Cumulative angle.
    pub a: f64,
}

impl Trajectory {
    /// Zero trajectory.
    #[must_use]
    pub fn zero() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            a: 0.0,
        }
    }
}

/// Build the cumulative trajectory from per-frame motions.
#[must_use]
pub fn build_trajectory(motions: &[FrameMotion]) -> Vec<Trajectory> {
    let mut trajectory = Vec::with_capacity(motions.len());
    let mut cum = Trajectory::zero();

    for m in motions {
        cum.x += m.dx;
        cum.y += m.dy;
        cum.a += m.da;
        trajectory.push(cum);
    }

    trajectory
}

/// Smooth a trajectory using a 1D Gaussian filter.
///
/// The smoothed trajectory preserves intentional camera movements (pans, tilts)
/// while removing high-frequency jitter.
#[must_use]
pub fn smooth_trajectory(trajectory: &[Trajectory], radius: usize, sigma: f64) -> Vec<Trajectory> {
    if trajectory.is_empty() {
        return Vec::new();
    }

    let n = trajectory.len();
    let kernel = build_gaussian_kernel(radius, sigma);

    let mut smoothed = Vec::with_capacity(n);

    for i in 0..n {
        let mut sum_x = 0.0_f64;
        let mut sum_y = 0.0_f64;
        let mut sum_a = 0.0_f64;
        let mut sum_w = 0.0_f64;

        for (ki, &w) in kernel.iter().enumerate() {
            let j_signed = i as isize + ki as isize - radius as isize;
            let j = j_signed.max(0).min((n - 1) as isize) as usize;

            sum_x += trajectory[j].x * w;
            sum_y += trajectory[j].y * w;
            sum_a += trajectory[j].a * w;
            sum_w += w;
        }

        if sum_w > 0.0 {
            smoothed.push(Trajectory {
                x: sum_x / sum_w,
                y: sum_y / sum_w,
                a: sum_a / sum_w,
            });
        } else {
            smoothed.push(trajectory[i]);
        }
    }

    smoothed
}

/// Compute the per-frame correction transforms.
///
/// The correction is: `smoothed_trajectory - original_trajectory`, which gives
/// the delta to apply to each frame.
#[must_use]
pub fn compute_corrections(
    motions: &[FrameMotion],
    original: &[Trajectory],
    smoothed: &[Trajectory],
    config: &StabilizeConfig,
) -> Vec<FrameMotion> {
    let n = motions.len().min(original.len()).min(smoothed.len());
    let mut corrections = Vec::with_capacity(n);

    for i in 0..n {
        let mut cdx = smoothed[i].x - original[i].x;
        let mut cdy = smoothed[i].y - original[i].y;
        let cda = smoothed[i].a - original[i].a;

        // Clamp correction
        let mag = (cdx * cdx + cdy * cdy).sqrt();
        if mag > config.max_correction_px {
            let scale = config.max_correction_px / mag;
            cdx *= scale;
            cdy *= scale;
        }

        corrections.push(FrameMotion::new(cdx, cdy, cda, 1.0));
    }

    corrections
}

/// Apply an affine correction to a single grayscale frame.
///
/// # Errors
///
/// Returns an error if the frame size is inconsistent.
pub fn apply_stabilization(
    frame: &[u8],
    width: usize,
    height: usize,
    correction: &FrameMotion,
    config: &StabilizeConfig,
) -> AlignResult<Vec<u8>> {
    if frame.len() != width * height {
        return Err(AlignError::InvalidConfig(
            "Frame size does not match width*height".to_string(),
        ));
    }

    let mat = correction.to_affine();

    // Compute inverse transform for backward mapping
    let inv = invert_affine(&mat)?;

    // Compute crop region
    let crop_x = (width as f64 * config.crop_ratio) as usize;
    let crop_y = (height as f64 * config.crop_ratio) as usize;

    let mut output = vec![0u8; width * height];

    for y in 0..height {
        for x in 0..width {
            // Map output pixel to input pixel via inverse transform
            let (sx, sy) = inv.transform_point(x as f32, y as f32);

            let sx = sx as isize;
            let sy = sy as isize;

            // Bounds check (accounting for crop region = fill with border value)
            if sx >= crop_x as isize
                && sy >= crop_y as isize
                && sx < (width - crop_x) as isize
                && sy < (height - crop_y) as isize
            {
                let src_idx = sy as usize * width + sx as usize;
                if src_idx < frame.len() {
                    output[y * width + x] = frame[src_idx];
                }
            }
        }
    }

    Ok(output)
}

/// Full stabilization pipeline: given a sequence of frame motions, compute
/// the stabilized correction transforms.
///
/// # Errors
///
/// Returns an error if the motions slice is empty.
pub fn stabilize_pipeline(
    motions: &[FrameMotion],
    config: &StabilizeConfig,
) -> AlignResult<Vec<FrameMotion>> {
    if motions.is_empty() {
        return Err(AlignError::InsufficientData(
            "Need at least one frame motion".to_string(),
        ));
    }

    let trajectory = build_trajectory(motions);
    let smoothed = smooth_trajectory(&trajectory, config.smooth_radius, config.smooth_sigma);
    let corrections = compute_corrections(motions, &trajectory, &smoothed, config);

    Ok(corrections)
}

/// Estimate per-frame motion from consecutive grayscale frames using a simple
/// block-matching approach (for use when KLT is not yet integrated).
///
/// Returns the estimated translation (dx, dy) between `prev` and `curr`.
///
/// # Errors
///
/// Returns an error if the frame sizes are inconsistent.
pub fn estimate_motion_translation(
    prev: &[u8],
    curr: &[u8],
    width: usize,
    height: usize,
    block_size: usize,
    search_range: usize,
) -> AlignResult<FrameMotion> {
    if prev.len() != width * height || curr.len() != width * height {
        return Err(AlignError::InvalidConfig(
            "Frame size does not match width*height".to_string(),
        ));
    }

    let mut total_dx = 0.0_f64;
    let mut total_dy = 0.0_f64;
    let mut count = 0u32;

    // Sample blocks from the interior
    let margin = search_range + block_size;
    if margin >= width / 2 || margin >= height / 2 {
        return Ok(FrameMotion::identity());
    }

    for by in (margin..height - margin).step_by(block_size) {
        for bx in (margin..width - margin).step_by(block_size) {
            let (best_dx, best_dy) =
                match_block(prev, curr, width, height, bx, by, block_size, search_range);
            total_dx += best_dx as f64;
            total_dy += best_dy as f64;
            count += 1;
        }
    }

    if count == 0 {
        return Ok(FrameMotion::identity());
    }

    Ok(FrameMotion::new(
        total_dx / f64::from(count),
        total_dy / f64::from(count),
        0.0,
        1.0,
    ))
}

// -- Internal helpers ---------------------------------------------------------

fn match_block(
    prev: &[u8],
    curr: &[u8],
    width: usize,
    _height: usize,
    bx: usize,
    by: usize,
    block_size: usize,
    search_range: usize,
) -> (i32, i32) {
    let mut best_sad = u64::MAX;
    let mut best_dx = 0i32;
    let mut best_dy = 0i32;
    let sr = search_range as i32;

    for dy in -sr..=sr {
        for dx in -sr..=sr {
            let mut sad = 0u64;
            for row in 0..block_size {
                for col in 0..block_size {
                    let px = bx + col;
                    let py = by + row;
                    let cx = (px as i32 + dx) as usize;
                    let cy = (py as i32 + dy) as usize;

                    let prev_val = i32::from(prev[py * width + px]);
                    let curr_val = i32::from(curr[cy * width + cx]);
                    sad += (prev_val - curr_val).unsigned_abs() as u64;
                }
            }

            if sad < best_sad
                || (sad == best_sad
                    && (dx.unsigned_abs() + dy.unsigned_abs())
                        < (best_dx.unsigned_abs() + best_dy.unsigned_abs()))
            {
                best_sad = sad;
                best_dx = dx;
                best_dy = dy;
            }
        }
    }

    (best_dx, best_dy)
}

fn build_gaussian_kernel(radius: usize, sigma: f64) -> Vec<f64> {
    let size = 2 * radius + 1;
    let mut kernel = Vec::with_capacity(size);
    let sigma2 = sigma * sigma;

    for i in 0..size {
        let x = i as f64 - radius as f64;
        kernel.push((-0.5 * x * x / sigma2).exp());
    }

    kernel
}

fn invert_affine(mat: &AffineMatrix) -> AlignResult<AffineMatrix> {
    let a = f64::from(mat.data[0][0]);
    let b = f64::from(mat.data[0][1]);
    let tx = f64::from(mat.data[0][2]);
    let c = f64::from(mat.data[1][0]);
    let d = f64::from(mat.data[1][1]);
    let ty = f64::from(mat.data[1][2]);

    let det = a * d - b * c;
    if det.abs() < 1e-12 {
        return Err(AlignError::NumericalError(
            "Singular affine matrix cannot be inverted".to_string(),
        ));
    }

    let inv_det = 1.0 / det;
    let ia = (d * inv_det) as f32;
    let ib = (-b * inv_det) as f32;
    let ic = (-c * inv_det) as f32;
    let id = (a * inv_det) as f32;
    let itx = ((-d * tx + b * ty) * inv_det) as f32;
    let ity = ((c * tx - a * ty) * inv_det) as f32;

    Ok(AffineMatrix {
        data: [[ia, ib, itx], [ic, id, ity], [0.0, 0.0, 1.0]],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- FrameMotion ----------------------------------------------------------

    #[test]
    fn test_frame_motion_identity() {
        let m = FrameMotion::identity();
        assert_eq!(m.dx, 0.0);
        assert_eq!(m.dy, 0.0);
        assert_eq!(m.da, 0.0);
        assert_eq!(m.ds, 1.0);
    }

    #[test]
    fn test_frame_motion_to_affine_identity() {
        let m = FrameMotion::identity();
        let mat = m.to_affine();
        assert!(mat.is_identity());
    }

    #[test]
    fn test_frame_motion_to_affine_translation() {
        let m = FrameMotion::new(10.0, 20.0, 0.0, 1.0);
        let mat = m.to_affine();
        let (x, y) = mat.transform_point(0.0, 0.0);
        assert!((x - 10.0).abs() < 1e-4);
        assert!((y - 20.0).abs() < 1e-4);
    }

    #[test]
    fn test_frame_motion_to_affine_rotation() {
        let m = FrameMotion::new(0.0, 0.0, std::f64::consts::PI / 2.0, 1.0);
        let mat = m.to_affine();
        let (x, y) = mat.transform_point(1.0, 0.0);
        assert!((x - 0.0).abs() < 1e-4, "x={x}");
        assert!((y - 1.0).abs() < 1e-4, "y={y}");
    }

    // -- Trajectory -----------------------------------------------------------

    #[test]
    fn test_build_trajectory() {
        let motions = vec![
            FrameMotion::new(1.0, 0.0, 0.0, 1.0),
            FrameMotion::new(2.0, 0.0, 0.0, 1.0),
            FrameMotion::new(-1.0, 0.0, 0.0, 1.0),
        ];
        let traj = build_trajectory(&motions);
        assert_eq!(traj.len(), 3);
        assert!((traj[0].x - 1.0).abs() < 1e-10);
        assert!((traj[1].x - 3.0).abs() < 1e-10);
        assert!((traj[2].x - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_build_trajectory_empty() {
        let traj = build_trajectory(&[]);
        assert!(traj.is_empty());
    }

    // -- Smoothing ------------------------------------------------------------

    #[test]
    fn test_smooth_trajectory_constant() {
        let traj = vec![
            Trajectory {
                x: 5.0,
                y: 3.0,
                a: 0.0,
            },
            Trajectory {
                x: 5.0,
                y: 3.0,
                a: 0.0,
            },
            Trajectory {
                x: 5.0,
                y: 3.0,
                a: 0.0,
            },
            Trajectory {
                x: 5.0,
                y: 3.0,
                a: 0.0,
            },
            Trajectory {
                x: 5.0,
                y: 3.0,
                a: 0.0,
            },
        ];
        let smoothed = smooth_trajectory(&traj, 2, 1.0);
        for s in &smoothed {
            assert!((s.x - 5.0).abs() < 1e-6);
            assert!((s.y - 3.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_smooth_trajectory_reduces_spike() {
        let traj = vec![
            Trajectory {
                x: 0.0,
                y: 0.0,
                a: 0.0,
            },
            Trajectory {
                x: 0.0,
                y: 0.0,
                a: 0.0,
            },
            Trajectory {
                x: 100.0,
                y: 0.0,
                a: 0.0,
            }, // spike
            Trajectory {
                x: 0.0,
                y: 0.0,
                a: 0.0,
            },
            Trajectory {
                x: 0.0,
                y: 0.0,
                a: 0.0,
            },
        ];
        let smoothed = smooth_trajectory(&traj, 2, 1.0);
        // The spike at index 2 should be attenuated
        assert!(
            smoothed[2].x < 100.0,
            "spike should be smoothed: {}",
            smoothed[2].x
        );
        assert!(smoothed[2].x > 0.0, "spike should still have some value");
    }

    #[test]
    fn test_smooth_trajectory_empty() {
        let smoothed = smooth_trajectory(&[], 5, 1.0);
        assert!(smoothed.is_empty());
    }

    // -- Corrections ----------------------------------------------------------

    #[test]
    fn test_corrections_zero_when_no_smoothing() {
        let motions = vec![FrameMotion::new(1.0, 0.0, 0.0, 1.0); 5];
        let traj = build_trajectory(&motions);
        // If smoothed == original, corrections should be zero
        let corrections = compute_corrections(&motions, &traj, &traj, &StabilizeConfig::default());
        for c in &corrections {
            assert!((c.dx).abs() < 1e-10);
            assert!((c.dy).abs() < 1e-10);
        }
    }

    #[test]
    fn test_corrections_clamped() {
        let config = StabilizeConfig {
            max_correction_px: 10.0,
            ..StabilizeConfig::default()
        };

        let original = vec![Trajectory {
            x: 0.0,
            y: 0.0,
            a: 0.0,
        }];
        let smoothed = vec![Trajectory {
            x: 100.0,
            y: 0.0,
            a: 0.0,
        }];
        let motions = vec![FrameMotion::identity()];

        let corrections = compute_corrections(&motions, &original, &smoothed, &config);
        let mag =
            (corrections[0].dx * corrections[0].dx + corrections[0].dy * corrections[0].dy).sqrt();
        assert!(
            mag <= config.max_correction_px + 1e-6,
            "correction should be clamped: mag={mag}"
        );
    }

    // -- Pipeline -------------------------------------------------------------

    #[test]
    fn test_stabilize_pipeline() {
        let motions: Vec<FrameMotion> = (0..30)
            .map(|i| FrameMotion::new((i as f64 * 0.5).sin() * 5.0, 0.0, 0.0, 1.0))
            .collect();

        let config = StabilizeConfig {
            smooth_radius: 5,
            smooth_sigma: 2.0,
            max_correction_px: 50.0,
            ..StabilizeConfig::default()
        };

        let corrections = stabilize_pipeline(&motions, &config).expect("should succeed");
        assert_eq!(corrections.len(), motions.len());
    }

    #[test]
    fn test_stabilize_pipeline_empty() {
        let result = stabilize_pipeline(&[], &StabilizeConfig::default());
        assert!(result.is_err());
    }

    // -- Apply stabilization --------------------------------------------------

    #[test]
    fn test_apply_stabilization_identity() {
        let w = 32usize;
        let h = 32usize;
        let frame: Vec<u8> = (0..w * h).map(|i| (i % 256) as u8).collect();

        let correction = FrameMotion::identity();
        let config = StabilizeConfig {
            crop_ratio: 0.0,
            ..StabilizeConfig::default()
        };

        let output =
            apply_stabilization(&frame, w, h, &correction, &config).expect("should succeed");
        // With identity correction and no crop, output should match input
        assert_eq!(output.len(), frame.len());
        // Interior pixels should match
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                assert_eq!(
                    output[y * w + x],
                    frame[y * w + x],
                    "mismatch at ({x}, {y})"
                );
            }
        }
    }

    #[test]
    fn test_apply_stabilization_size_mismatch() {
        let result = apply_stabilization(
            &[0u8; 100],
            20,
            20,
            &FrameMotion::identity(),
            &StabilizeConfig::default(),
        );
        assert!(result.is_err());
    }

    // -- Motion estimation ----------------------------------------------------

    #[test]
    fn test_estimate_motion_identical() {
        let w = 64usize;
        let h = 64usize;
        let frame = vec![128u8; w * h];

        let motion =
            estimate_motion_translation(&frame, &frame, w, h, 8, 4).expect("should succeed");
        assert!((motion.dx).abs() < 1.0);
        assert!((motion.dy).abs() < 1.0);
    }

    #[test]
    fn test_estimate_motion_shifted() {
        let w = 128usize;
        let h = 128usize;
        let mut prev = vec![64u8; w * h];
        let mut curr = vec![64u8; w * h];

        // Create a strong vertical stripe pattern
        for y in 0..h {
            for x in 0..w {
                prev[y * w + x] = if (x / 16) % 2 == 0 { 200 } else { 50 };
                // Shift right by 3
                let sx = if x >= 3 { x - 3 } else { 0 };
                curr[y * w + x] = if (sx / 16) % 2 == 0 { 200 } else { 50 };
            }
        }

        let motion =
            estimate_motion_translation(&prev, &curr, w, h, 16, 8).expect("should succeed");
        // Should detect roughly 3px rightward shift
        assert!(
            (motion.dx - 3.0).abs() < 3.0,
            "expected ~3px shift, got dx={:.2}",
            motion.dx
        );
    }

    #[test]
    fn test_estimate_motion_size_mismatch() {
        let result = estimate_motion_translation(&[0u8; 100], &[0u8; 200], 10, 10, 4, 2);
        assert!(result.is_err());
    }

    // -- Internal helpers -----------------------------------------------------

    #[test]
    fn test_gaussian_kernel_is_symmetric() {
        let kernel = build_gaussian_kernel(5, 2.0);
        assert_eq!(kernel.len(), 11);
        for i in 0..5 {
            assert!((kernel[i] - kernel[10 - i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_gaussian_kernel_peak_at_center() {
        let kernel = build_gaussian_kernel(3, 1.0);
        let center = kernel[3];
        for (i, &v) in kernel.iter().enumerate() {
            if i != 3 {
                assert!(v <= center + 1e-10);
            }
        }
    }

    #[test]
    fn test_invert_affine_identity() {
        let id = AffineMatrix::identity();
        let inv = invert_affine(&id).expect("should succeed");
        assert!(inv.is_identity());
    }

    #[test]
    fn test_invert_affine_translation() {
        let t = AffineMatrix::translation(5.0, -3.0);
        let inv = invert_affine(&t).expect("should succeed");
        let (x, y) = inv.transform_point(5.0, -3.0);
        assert!((x - 0.0).abs() < 1e-4);
        assert!((y - 0.0).abs() < 1e-4);
    }

    #[test]
    fn test_invert_affine_singular() {
        let singular = AffineMatrix {
            data: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        };
        assert!(invert_affine(&singular).is_err());
    }

    #[test]
    fn test_stabilize_config_default() {
        let config = StabilizeConfig::default();
        assert_eq!(config.smooth_radius, 15);
        assert_eq!(config.mode, StabilizationMode::TranslationRotation);
    }
}
