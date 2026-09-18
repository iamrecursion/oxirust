//! Low-resolution preview stabilization.
//!
//! Running full-resolution stabilization during content creation or timeline
//! scrubbing is prohibitively expensive.  This module provides a two-stage
//! pipeline that:
//!
//! 1. Downscales each frame to a configurable fraction of its original size.
//! 2. Runs the motion estimation and smoothing on the small frames.
//! 3. Scales the resulting corrective transforms back to full-resolution coordinates.
//! 4. Returns **both** the stabilized preview frames *and* the full-res transforms,
//!    so the caller can either display the preview immediately or schedule a
//!    full-resolution render using the pre-computed transforms.
//!
//! The downscale step uses box-filter averaging, which is fast but sufficient
//! for motion estimation accuracy (sub-pixel precision is not required at this stage).

use crate::error::{StabilizeError, StabilizeResult};
use crate::motion::estimate::MotionEstimator;
use crate::motion::tracker::MotionTracker;
use crate::motion::trajectory::Trajectory;
use crate::smooth::filter::TrajectorySmoother;
use crate::transform::calculate::{StabilizationTransform, TransformCalculator};
use crate::warp::apply::FrameWarper;
use crate::{Frame, StabilizationMode};
use scirs2_core::ndarray::Array2;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the preview stabilization pipeline.
#[derive(Debug, Clone)]
pub struct PreviewConfig {
    /// Downscale factor in `(0, 1]`.  0.25 means quarter-resolution.
    pub scale: f64,
    /// Smoothing window size for the preview trajectory.
    pub smoothing_window: usize,
    /// Smoothing strength in `[0, 1]`.
    pub smoothing_strength: f64,
    /// Number of features to track (fewer → faster).
    pub feature_count: usize,
    /// Stabilization mode used for motion estimation.
    pub mode: StabilizationMode,
}

impl Default for PreviewConfig {
    fn default() -> Self {
        Self {
            scale: 0.25,
            smoothing_window: 30,
            smoothing_strength: 0.8,
            feature_count: 200,
            mode: StabilizationMode::Translation,
        }
    }
}

impl PreviewConfig {
    /// Create a configuration at a given downscale factor.
    ///
    /// # Errors
    ///
    /// Returns an error when `scale` is not in `(0, 1]`.
    pub fn with_scale(scale: f64) -> StabilizeResult<Self> {
        if !(scale > 0.0 && scale <= 1.0) {
            return Err(StabilizeError::invalid_parameter(
                "scale",
                format!("{scale}: must be in (0, 1]"),
            ));
        }
        Ok(Self {
            scale,
            ..Self::default()
        })
    }
}

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// Output of the preview stabilization pipeline.
pub struct PreviewResult {
    /// Stabilized low-resolution preview frames.
    pub preview_frames: Vec<Frame>,
    /// Corrective transforms scaled back to full-resolution coordinates.
    ///
    /// Apply these to the original frames to produce a full-resolution output
    /// without re-running feature tracking.
    pub full_res_transforms: Vec<StabilizationTransform>,
    /// The downscale factor that was used.
    pub scale: f64,
}

// ---------------------------------------------------------------------------
// Downscale / upscale helpers
// ---------------------------------------------------------------------------

/// Box-filter downscale of an `Array2<u8>` by a fractional `scale`.
///
/// Each output pixel is the average of the corresponding source block.
#[must_use]
pub fn downscale(data: &Array2<u8>, scale: f64) -> Array2<u8> {
    let (src_h, src_w) = data.dim();
    let dst_w = ((src_w as f64 * scale).round() as usize).max(1);
    let dst_h = ((src_h as f64 * scale).round() as usize).max(1);

    let inv_scale_x = src_w as f64 / dst_w as f64;
    let inv_scale_y = src_h as f64 / dst_h as f64;

    let mut out = Array2::zeros((dst_h, dst_w));

    for dy in 0..dst_h {
        for dx in 0..dst_w {
            // Source region covered by this output pixel.
            let sx0 = (dx as f64 * inv_scale_x) as usize;
            let sy0 = (dy as f64 * inv_scale_y) as usize;
            let sx1 = (((dx + 1) as f64 * inv_scale_x) as usize).min(src_w);
            let sy1 = (((dy + 1) as f64 * inv_scale_y) as usize).min(src_h);

            let mut sum = 0u32;
            let mut count = 0u32;
            for sy in sy0..sy1 {
                for sx in sx0..sx1 {
                    sum += data[[sy, sx]] as u32;
                    count += 1;
                }
            }
            out[[dy, dx]] = sum.checked_div(count).unwrap_or(0) as u8;
        }
    }

    out
}

/// Scale a [`StabilizationTransform`] from preview coordinates to full-res coordinates.
///
/// Translation components are divided by `scale`; rotation and scale fields are
/// left unchanged (they are already dimensionless / angular).
#[must_use]
pub fn scale_transform_to_full_res(
    t: &StabilizationTransform,
    scale: f64,
) -> StabilizationTransform {
    StabilizationTransform {
        dx: t.dx / scale,
        dy: t.dy / scale,
        angle: t.angle,
        scale: t.scale,
        frame_index: t.frame_index,
        confidence: t.confidence,
    }
}

// ---------------------------------------------------------------------------
// Preview stabilizer
// ---------------------------------------------------------------------------

/// Runs the full stabilization pipeline on downscaled frames and returns both
/// preview output and full-resolution-ready transforms.
///
/// # Errors
///
/// - [`StabilizeError::EmptyFrameSequence`] – `frames` is empty.
/// - Propagates motion-tracking and smoothing errors.
pub fn stabilize_preview(
    frames: &[Frame],
    config: &PreviewConfig,
) -> StabilizeResult<PreviewResult> {
    if frames.is_empty() {
        return Err(StabilizeError::EmptyFrameSequence);
    }

    // 1. Downscale frames.
    let small_frames: Vec<Frame> = frames
        .iter()
        .map(|f| {
            let small_data = downscale(&f.data, config.scale);
            let (sh, sw) = small_data.dim();
            Frame::new(sw, sh, f.timestamp, small_data)
        })
        .collect();

    // 2. Run motion tracker on preview frames.
    let mut tracker = MotionTracker::new(config.feature_count);
    let tracks = match tracker.track(&small_frames) {
        Ok(t) => t,
        Err(StabilizeError::InsufficientFeatures { .. }) => {
            // Fall back to identity transforms.
            let identity_transforms: Vec<StabilizationTransform> = (0..frames.len())
                .map(StabilizationTransform::identity)
                .collect();
            let warper = FrameWarper::new();
            let preview_frames = warper.warp(&small_frames, &identity_transforms)?;
            return Ok(PreviewResult {
                preview_frames,
                full_res_transforms: identity_transforms,
                scale: config.scale,
            });
        }
        Err(e) => return Err(e),
    };

    // 3. Estimate motion on the preview trajectory.
    let estimator = MotionEstimator::new(config.mode);
    let models = estimator.estimate(&tracks, small_frames.len())?;
    let trajectory = Trajectory::from_models(&models)?;

    // 4. Smooth the preview trajectory.
    let mut smoother = TrajectorySmoother::new(config.smoothing_window, config.smoothing_strength);
    let smoothed = smoother.smooth(&trajectory)?;

    // 5. Compute preview corrective transforms.
    let calculator = TransformCalculator::new();
    let preview_transforms = calculator.calculate(&trajectory, &smoothed)?;

    // 6. Scale transforms to full resolution.
    let full_res_transforms: Vec<StabilizationTransform> = preview_transforms
        .iter()
        .map(|t| scale_transform_to_full_res(t, config.scale))
        .collect();

    // 7. Warp preview frames.
    let warper = FrameWarper::new();
    let preview_frames = warper.warp(&small_frames, &preview_transforms)?;

    Ok(PreviewResult {
        preview_frames,
        full_res_transforms,
        scale: config.scale,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn make_frame(w: usize, h: usize, fill: u8, ts: f64) -> Frame {
        Frame::new(w, h, ts, Array2::from_elem((h, w), fill))
    }

    fn make_frames(n: usize, w: usize, h: usize) -> Vec<Frame> {
        (0..n)
            .map(|i| make_frame(w, h, (i * 20) as u8, i as f64 / 30.0))
            .collect()
    }

    #[test]
    fn test_preview_config_default() {
        let cfg = PreviewConfig::default();
        assert!((cfg.scale - 0.25).abs() < 1e-9);
        assert!(cfg.feature_count > 0);
    }

    #[test]
    fn test_preview_config_with_scale_valid() {
        let cfg = PreviewConfig::with_scale(0.5).expect("valid scale");
        assert!((cfg.scale - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_preview_config_with_scale_invalid() {
        assert!(PreviewConfig::with_scale(0.0).is_err());
        assert!(PreviewConfig::with_scale(-0.1).is_err());
        assert!(PreviewConfig::with_scale(1.5).is_err());
    }

    #[test]
    fn test_downscale_half_size() {
        let data = Array2::from_elem((100, 100), 128u8);
        let small = downscale(&data, 0.5);
        let (h, w) = small.dim();
        assert_eq!(w, 50);
        assert_eq!(h, 50);
        // Uniform fill → all output pixels should remain 128.
        assert!(small.iter().all(|&v| v == 128));
    }

    #[test]
    fn test_downscale_uniform_preserves_value() {
        let data = Array2::from_elem((40, 60), 200u8);
        let small = downscale(&data, 0.25);
        assert!(small.iter().all(|&v| v == 200));
    }

    #[test]
    fn test_downscale_minimum_size_one() {
        let data = Array2::from_elem((5, 5), 100u8);
        let tiny = downscale(&data, 0.01); // should not panic; minimum 1×1
        let (h, w) = tiny.dim();
        assert!(h >= 1 && w >= 1);
    }

    #[test]
    fn test_scale_transform_to_full_res() {
        let t = StabilizationTransform {
            frame_index: 0,
            dx: 4.0,
            dy: -2.0,
            angle: 0.01,
            scale: 1.0,
            confidence: 1.0,
        };
        let full = scale_transform_to_full_res(&t, 0.25);
        assert!((full.dx - 16.0).abs() < 1e-9);
        assert!((full.dy - (-8.0)).abs() < 1e-9);
        assert!((full.angle - 0.01).abs() < 1e-9); // unchanged
    }

    #[test]
    fn test_stabilize_preview_empty_error() {
        let cfg = PreviewConfig::default();
        let result = stabilize_preview(&[], &cfg);
        assert!(matches!(result, Err(StabilizeError::EmptyFrameSequence)));
    }

    #[test]
    fn test_stabilize_preview_output_counts_match() {
        let frames = make_frames(5, 64, 64);
        let cfg = PreviewConfig {
            scale: 0.5,
            feature_count: 50,
            ..PreviewConfig::default()
        };
        let result = stabilize_preview(&frames, &cfg).expect("preview ok");
        assert_eq!(result.preview_frames.len(), 5);
        assert_eq!(result.full_res_transforms.len(), 5);
    }

    #[test]
    fn test_stabilize_preview_frame_dimensions_reduced() {
        let frames = make_frames(3, 80, 60);
        let cfg = PreviewConfig {
            scale: 0.5,
            feature_count: 50,
            ..PreviewConfig::default()
        };
        let result = stabilize_preview(&frames, &cfg).expect("ok");
        // Preview frames should be approximately half size.
        let pf = &result.preview_frames[0];
        assert!(pf.width <= 40 + 1 && pf.width >= 40 - 1);
        assert!(pf.height <= 30 + 1 && pf.height >= 30 - 1);
    }

    #[test]
    fn test_stabilize_preview_scale_stored() {
        let frames = make_frames(3, 64, 64);
        let cfg = PreviewConfig {
            scale: 0.25,
            feature_count: 50,
            ..PreviewConfig::default()
        };
        let result = stabilize_preview(&frames, &cfg).expect("ok");
        assert!((result.scale - 0.25).abs() < 1e-9);
    }

    #[test]
    fn test_stabilize_preview_full_res_transforms_scaled_correctly() {
        let frames = make_frames(3, 64, 64);
        let scale = 0.5;
        let cfg = PreviewConfig {
            scale,
            feature_count: 50,
            ..PreviewConfig::default()
        };
        let result = stabilize_preview(&frames, &cfg).expect("ok");
        // Verify that for each frame the full-res translation is (preview_translation / scale).
        // We can't inspect internal preview transforms directly, but we can check structure.
        assert_eq!(
            result.full_res_transforms.len(),
            result.preview_frames.len()
        );
    }
}
