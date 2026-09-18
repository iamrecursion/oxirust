//! Motion-based scene detection.
//!
//! This module provides scene detection based on motion analysis.
//! It uses motion vectors, optical flow patterns, and frame differencing
//! to detect scene boundaries.

use crate::error::{CvError, CvResult};
use oximedia_codec::VideoFrame;
use oximedia_core::PixelFormat;

use super::{ChangeType, SceneChange, SceneConfig, SceneMetadata};

/// Configuration for motion-based detection.
#[derive(Debug, Clone)]
pub struct MotionConfig {
    /// Block size for motion estimation.
    pub block_size: usize,
    /// Search range for motion vectors.
    pub search_range: i32,
    /// Threshold for considering a pixel as changed.
    pub diff_threshold: u8,
    /// Use block matching (true) or simple frame differencing (false).
    pub use_block_matching: bool,
    /// Motion magnitude threshold.
    pub motion_threshold: f64,
}

impl Default for MotionConfig {
    fn default() -> Self {
        Self {
            block_size: 16,
            search_range: 8,
            diff_threshold: 30,
            use_block_matching: false, // Simple diff is faster and sufficient
            motion_threshold: 0.3,
        }
    }
}

/// Motion vector.
#[derive(Debug, Clone, Copy)]
pub struct MotionVector {
    /// Horizontal displacement.
    pub dx: i32,
    /// Vertical displacement.
    pub dy: i32,
    /// Motion magnitude.
    pub magnitude: f64,
}

impl MotionVector {
    /// Create a new motion vector.
    #[must_use]
    pub fn new(dx: i32, dy: i32) -> Self {
        let magnitude = ((dx * dx + dy * dy) as f64).sqrt();
        Self { dx, dy, magnitude }
    }

    /// Check if this is a zero motion vector.
    #[must_use]
    pub const fn is_zero(&self) -> bool {
        self.dx == 0 && self.dy == 0
    }
}

/// Motion field containing motion vectors for a frame pair.
#[derive(Debug, Clone)]
pub struct MotionField {
    /// Motion vectors in a grid.
    pub vectors: Vec<MotionVector>,
    /// Number of blocks in X direction.
    pub blocks_x: usize,
    /// Number of blocks in Y direction.
    pub blocks_y: usize,
    /// Block size used.
    pub block_size: usize,
}

impl MotionField {
    /// Create a new motion field.
    #[must_use]
    pub fn new(blocks_x: usize, blocks_y: usize, block_size: usize) -> Self {
        let count = blocks_x * blocks_y;
        Self {
            vectors: vec![MotionVector::new(0, 0); count],
            blocks_x,
            blocks_y,
            block_size,
        }
    }

    /// Compute average motion magnitude.
    #[must_use]
    pub fn average_magnitude(&self) -> f64 {
        if self.vectors.is_empty() {
            return 0.0;
        }

        let sum: f64 = self.vectors.iter().map(|v| v.magnitude).sum();
        sum / self.vectors.len() as f64
    }

    /// Compute motion consistency (how similar are motion vectors).
    #[must_use]
    pub fn consistency(&self) -> f64 {
        if self.vectors.len() < 2 {
            return 1.0;
        }

        let avg_dx: f64 =
            self.vectors.iter().map(|v| v.dx as f64).sum::<f64>() / self.vectors.len() as f64;
        let avg_dy: f64 =
            self.vectors.iter().map(|v| v.dy as f64).sum::<f64>() / self.vectors.len() as f64;

        let variance: f64 = self
            .vectors
            .iter()
            .map(|v| {
                let dx_diff = v.dx as f64 - avg_dx;
                let dy_diff = v.dy as f64 - avg_dy;
                dx_diff * dx_diff + dy_diff * dy_diff
            })
            .sum::<f64>()
            / self.vectors.len() as f64;

        let std_dev = variance.sqrt();

        // Normalize to [0, 1], where 1 is highly consistent
        if std_dev < f64::EPSILON {
            1.0
        } else {
            (1.0 / (1.0 + std_dev / 10.0)).clamp(0.0, 1.0)
        }
    }

    /// Count non-zero motion vectors.
    #[must_use]
    pub fn non_zero_count(&self) -> usize {
        self.vectors.iter().filter(|v| !v.is_zero()).count()
    }
}

/// Extract grayscale data from a video frame.
fn extract_grayscale(frame: &VideoFrame) -> CvResult<Vec<u8>> {
    match frame.format {
        PixelFormat::Yuv420p | PixelFormat::Yuv422p | PixelFormat::Yuv444p => {
            if frame.planes.is_empty() {
                return Err(CvError::insufficient_data(1, 0));
            }
            Ok(frame.planes[0].data.clone())
        }
        PixelFormat::Rgb24 => {
            if frame.planes.is_empty() {
                return Err(CvError::insufficient_data(1, 0));
            }

            let data = &frame.planes[0].data;
            let size = (frame.width * frame.height) as usize;
            let mut gray = Vec::with_capacity(size);

            for chunk in data.chunks_exact(3) {
                let luma = (chunk[0] as f64 * 0.299
                    + chunk[1] as f64 * 0.587
                    + chunk[2] as f64 * 0.114) as u8;
                gray.push(luma);
            }

            Ok(gray)
        }
        _ => Err(CvError::unsupported_format(format!("{:?}", frame.format))),
    }
}

/// Compute frame difference ratio.
fn compute_frame_diff(frame1: &[u8], frame2: &[u8], threshold: u8) -> f64 {
    if frame1.len() != frame2.len() {
        return 1.0;
    }

    let mut diff_count = 0;

    for (p1, p2) in frame1.iter().zip(frame2.iter()) {
        let diff = (*p1 as i32 - *p2 as i32).abs();
        if diff >= threshold as i32 {
            diff_count += 1;
        }
    }

    diff_count as f64 / frame1.len() as f64
}

/// Compute Sum of Absolute Differences (SAD) for a block.
#[allow(clippy::too_many_arguments)]
fn compute_sad(
    src: &[u8],
    ref_data: &[u8],
    src_x: usize,
    src_y: usize,
    ref_x: usize,
    ref_y: usize,
    block_size: usize,
    width: usize,
) -> u32 {
    let mut sad = 0u32;

    for by in 0..block_size {
        for bx in 0..block_size {
            let src_idx = (src_y + by) * width + src_x + bx;
            let ref_idx = (ref_y + by) * width + ref_x + bx;

            if src_idx < src.len() && ref_idx < ref_data.len() {
                let diff = (src[src_idx] as i32 - ref_data[ref_idx] as i32).abs();
                sad += diff as u32;
            }
        }
    }

    sad
}

/// Estimate motion between two frames using block matching.
fn estimate_motion_block_matching(
    frame1: &[u8],
    frame2: &[u8],
    width: u32,
    height: u32,
    config: &MotionConfig,
) -> CvResult<MotionField> {
    let w = width as usize;
    let h = height as usize;
    let bs = config.block_size;
    let sr = config.search_range;

    let blocks_x = w.div_ceil(bs);
    let blocks_y = h.div_ceil(bs);

    let mut field = MotionField::new(blocks_x, blocks_y, bs);

    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let src_x = bx * bs;
            let src_y = by * bs;

            // Skip if block goes out of bounds
            if src_x + bs > w || src_y + bs > h {
                continue;
            }

            let mut best_dx = 0i32;
            let mut best_dy = 0i32;
            let mut best_sad = u32::MAX;

            // Search in the search range
            for dy in -sr..=sr {
                for dx in -sr..=sr {
                    let ref_x = (src_x as i32 + dx).max(0) as usize;
                    let ref_y = (src_y as i32 + dy).max(0) as usize;

                    // Check bounds
                    if ref_x + bs > w || ref_y + bs > h {
                        continue;
                    }

                    let sad = compute_sad(frame2, frame1, src_x, src_y, ref_x, ref_y, bs, w);

                    if sad < best_sad {
                        best_sad = sad;
                        best_dx = dx;
                        best_dy = dy;
                    }
                }
            }

            let idx = by * blocks_x + bx;
            field.vectors[idx] = MotionVector::new(best_dx, best_dy);
        }
    }

    Ok(field)
}

/// Compute motion score between two frames.
pub fn compute_motion_score(
    frame1: &VideoFrame,
    frame2: &VideoFrame,
    config: &MotionConfig,
) -> CvResult<f64> {
    if frame1.width != frame2.width || frame1.height != frame2.height {
        return Err(CvError::invalid_parameter(
            "frames",
            "dimensions must match",
        ));
    }

    let gray1 = extract_grayscale(frame1)?;
    let gray2 = extract_grayscale(frame2)?;

    if config.use_block_matching {
        let motion_field =
            estimate_motion_block_matching(&gray1, &gray2, frame1.width, frame1.height, config)?;

        let avg_magnitude = motion_field.average_magnitude();
        let consistency = motion_field.consistency();

        // High motion with low consistency suggests scene change
        let motion_score = avg_magnitude / (config.block_size as f64 * 2.0);
        let scene_change_score = motion_score * (1.0 - consistency);

        Ok(1.0 - scene_change_score.clamp(0.0, 1.0))
    } else {
        // Simple frame differencing
        let diff_ratio = compute_frame_diff(&gray1, &gray2, config.diff_threshold);

        // High difference suggests scene change
        Ok(1.0 - diff_ratio)
    }
}

/// Detect motion-based scene changes.
pub fn detect_motion_changes(
    frames: &[VideoFrame],
    config: &SceneConfig,
) -> CvResult<Vec<SceneChange>> {
    let mut changes = Vec::new();

    for i in 1..frames.len() {
        let similarity = compute_motion_score(&frames[i - 1], &frames[i], &config.motion_config)?;
        let diff = 1.0 - similarity;

        if diff > config.threshold {
            changes.push(SceneChange {
                frame_number: i,
                timestamp: frames[i].timestamp,
                confidence: diff,
                change_type: ChangeType::Cut,
                metadata: SceneMetadata {
                    motion_score: Some(diff),
                    ..Default::default()
                },
            });
        }
    }

    Ok(changes)
}

/// Compute motion intensity map.
pub fn compute_motion_intensity(
    frame1: &VideoFrame,
    frame2: &VideoFrame,
    threshold: u8,
) -> CvResult<Vec<u8>> {
    if frame1.width != frame2.width || frame1.height != frame2.height {
        return Err(CvError::invalid_parameter(
            "frames",
            "dimensions must match",
        ));
    }

    let gray1 = extract_grayscale(frame1)?;
    let gray2 = extract_grayscale(frame2)?;

    let mut intensity = Vec::with_capacity(gray1.len());

    for (p1, p2) in gray1.iter().zip(gray2.iter()) {
        let diff = (*p1 as i32 - *p2 as i32).abs();
        let motion = if diff >= threshold as i32 {
            diff.min(255) as u8
        } else {
            0
        };
        intensity.push(motion);
    }

    Ok(intensity)
}

/// Compute motion histogram.
pub fn compute_motion_histogram(
    frame1: &VideoFrame,
    frame2: &VideoFrame,
    bins: usize,
) -> CvResult<Vec<u32>> {
    let intensity = compute_motion_intensity(frame1, frame2, 0)?;

    let mut histogram = vec![0u32; bins];
    let bin_scale = bins as f64 / 256.0;

    for &val in &intensity {
        let bin = ((val as f64 * bin_scale) as usize).min(bins - 1);
        histogram[bin] += 1;
    }

    Ok(histogram)
}

/// Analyze motion patterns in a sequence of frames.
pub fn analyze_motion_pattern(frames: &[VideoFrame], window: usize) -> CvResult<Vec<f64>> {
    if frames.len() < 2 {
        return Ok(Vec::new());
    }

    let config = MotionConfig::default();
    let mut motion_scores = Vec::new();

    for i in 1..frames.len() {
        let score = compute_motion_score(&frames[i - 1], &frames[i], &config)?;
        motion_scores.push(1.0 - score); // Convert similarity to motion amount
    }

    // Apply smoothing with sliding window
    if window > 1 && motion_scores.len() >= window {
        let mut smoothed = Vec::new();

        for i in 0..motion_scores.len() {
            let start = i.saturating_sub(window / 2);
            let end = (i + window / 2 + 1).min(motion_scores.len());

            let sum: f64 = motion_scores[start..end].iter().sum();
            let avg = sum / (end - start) as f64;
            smoothed.push(avg);
        }

        return Ok(smoothed);
    }

    Ok(motion_scores)
}

/// Detect camera motion (pan, tilt, zoom).
pub fn detect_camera_motion(
    frame1: &VideoFrame,
    frame2: &VideoFrame,
    config: &MotionConfig,
) -> CvResult<CameraMotion> {
    let gray1 = extract_grayscale(frame1)?;
    let gray2 = extract_grayscale(frame2)?;

    let motion_field =
        estimate_motion_block_matching(&gray1, &gray2, frame1.width, frame1.height, config)?;

    let avg_dx: f64 = motion_field
        .vectors
        .iter()
        .map(|v| v.dx as f64)
        .sum::<f64>()
        / motion_field.vectors.len() as f64;
    let avg_dy: f64 = motion_field
        .vectors
        .iter()
        .map(|v| v.dy as f64)
        .sum::<f64>()
        / motion_field.vectors.len() as f64;

    let consistency = motion_field.consistency();

    // Determine motion type based on average motion and consistency
    let motion_type = if consistency > 0.7 {
        if avg_dx.abs() > avg_dy.abs() && avg_dx.abs() > 2.0 {
            if avg_dx > 0.0 {
                CameraMotionType::PanRight
            } else {
                CameraMotionType::PanLeft
            }
        } else if avg_dy.abs() > 2.0 {
            if avg_dy > 0.0 {
                CameraMotionType::TiltDown
            } else {
                CameraMotionType::TiltUp
            }
        } else {
            CameraMotionType::Static
        }
    } else if motion_field.average_magnitude() > 5.0 {
        CameraMotionType::Complex
    } else {
        CameraMotionType::Static
    };

    Ok(CameraMotion {
        motion_type,
        dx: avg_dx,
        dy: avg_dy,
        consistency,
        magnitude: motion_field.average_magnitude(),
    })
}

/// Camera motion type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraMotionType {
    /// No significant camera motion.
    Static,
    /// Camera panning left.
    PanLeft,
    /// Camera panning right.
    PanRight,
    /// Camera tilting up.
    TiltUp,
    /// Camera tilting down.
    TiltDown,
    /// Zoom in or out.
    Zoom,
    /// Complex motion (multiple types).
    Complex,
}

/// Camera motion analysis result.
#[derive(Debug, Clone)]
pub struct CameraMotion {
    /// Type of camera motion detected.
    pub motion_type: CameraMotionType,
    /// Average horizontal displacement.
    pub dx: f64,
    /// Average vertical displacement.
    pub dy: f64,
    /// Motion consistency (0-1).
    pub consistency: f64,
    /// Average motion magnitude.
    pub magnitude: f64,
}

impl CameraMotion {
    /// Check if camera is mostly static.
    #[must_use]
    pub fn is_static(&self) -> bool {
        matches!(self.motion_type, CameraMotionType::Static)
    }

    /// Check if camera is in motion.
    #[must_use]
    pub fn is_moving(&self) -> bool {
        !self.is_static()
    }
}
