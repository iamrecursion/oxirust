//! Rolling shutter correction.
//!
//! This module provides tools for correcting rolling shutter artifacts:
//!
//! - Motion estimation per scanline
//! - Wobble correction
//! - Skew removal
//! - Global shutter simulation

use crate::{AlignError, AlignResult};
// Vector2 removed - unused

/// Rolling shutter parameters
#[derive(Debug, Clone)]
pub struct RollingShutterParams {
    /// Readout time in seconds (time to read entire frame)
    pub readout_time: f64,
    /// Frame rate
    pub frame_rate: f64,
    /// Readout direction
    pub direction: ReadoutDirection,
}

/// Readout direction for rolling shutter
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadoutDirection {
    /// Top to bottom
    TopToBottom,
    /// Bottom to top
    BottomToTop,
    /// Left to right
    LeftToRight,
    /// Right to left
    RightToLeft,
}

impl RollingShutterParams {
    /// Create new rolling shutter parameters
    #[must_use]
    pub fn new(readout_time: f64, frame_rate: f64, direction: ReadoutDirection) -> Self {
        Self {
            readout_time,
            frame_rate,
            direction,
        }
    }

    /// Compute time offset for a given scanline
    #[must_use]
    pub fn compute_scanline_time(&self, scanline: usize, total_lines: usize) -> f64 {
        let progress = match self.direction {
            ReadoutDirection::TopToBottom => scanline as f64 / total_lines as f64,
            ReadoutDirection::BottomToTop => 1.0 - (scanline as f64 / total_lines as f64),
            ReadoutDirection::LeftToRight => scanline as f64 / total_lines as f64,
            ReadoutDirection::RightToLeft => 1.0 - (scanline as f64 / total_lines as f64),
        };

        progress * self.readout_time
    }
}

/// Motion vector for a scanline
#[derive(Debug, Clone, Copy)]
pub struct MotionVector {
    /// Horizontal displacement
    pub dx: f32,
    /// Vertical displacement
    pub dy: f32,
    /// Confidence (0.0 to 1.0)
    pub confidence: f32,
}

impl MotionVector {
    /// Create a new motion vector
    #[must_use]
    pub fn new(dx: f32, dy: f32, confidence: f32) -> Self {
        Self { dx, dy, confidence }
    }

    /// Create zero motion vector
    #[must_use]
    pub fn zero() -> Self {
        Self {
            dx: 0.0,
            dy: 0.0,
            confidence: 1.0,
        }
    }

    /// Magnitude of motion
    #[must_use]
    pub fn magnitude(&self) -> f32 {
        (self.dx * self.dx + self.dy * self.dy).sqrt()
    }
}

/// Rolling shutter motion estimator
pub struct RollingShutterEstimator {
    /// Block size for motion estimation
    pub block_size: usize,
    /// Search range for motion
    pub search_range: isize,
}

impl Default for RollingShutterEstimator {
    fn default() -> Self {
        Self {
            block_size: 16,
            search_range: 16,
        }
    }
}

impl RollingShutterEstimator {
    /// Create a new motion estimator
    #[must_use]
    pub fn new(block_size: usize, search_range: isize) -> Self {
        Self {
            block_size,
            search_range,
        }
    }

    /// Estimate motion vectors for each scanline
    ///
    /// # Errors
    /// Returns error if frames are invalid
    pub fn estimate_motion(
        &self,
        frame1: &[u8],
        frame2: &[u8],
        width: usize,
        height: usize,
    ) -> AlignResult<Vec<MotionVector>> {
        if frame1.len() != width * height * 3 || frame2.len() != width * height * 3 {
            return Err(AlignError::InvalidConfig("Frame size mismatch".to_string()));
        }

        let mut motion_vectors = Vec::new();

        // Estimate motion for each row
        for y in (0..height).step_by(self.block_size) {
            let mv = self.estimate_row_motion(frame1, frame2, width, height, y);
            motion_vectors.push(mv);
        }

        Ok(motion_vectors)
    }

    /// Estimate motion for a single row
    fn estimate_row_motion(
        &self,
        frame1: &[u8],
        frame2: &[u8],
        width: usize,
        height: usize,
        y: usize,
    ) -> MotionVector {
        let mut best_dx = 0;
        let mut best_dy = 0;
        let mut best_sad = u32::MAX;

        // Search in a window around the current position
        for dy in -self.search_range..=self.search_range {
            for dx in -self.search_range..=self.search_range {
                let sad = self.compute_sad(frame1, frame2, width, height, 0, y, dx, dy);

                if sad < best_sad {
                    best_sad = sad;
                    best_dx = dx;
                    best_dy = dy;
                }
            }
        }

        // Compute confidence based on SAD
        let confidence = if best_sad == 0 {
            1.0
        } else {
            1.0 / (1.0 + (best_sad as f32 / 1000.0))
        };

        MotionVector::new(best_dx as f32, best_dy as f32, confidence)
    }

    /// Compute sum of absolute differences
    #[allow(clippy::too_many_arguments)]
    fn compute_sad(
        &self,
        frame1: &[u8],
        frame2: &[u8],
        width: usize,
        height: usize,
        x: usize,
        y: usize,
        dx: isize,
        dy: isize,
    ) -> u32 {
        let mut sad = 0u32;
        let block_height = self.block_size.min(height - y);

        for by in 0..block_height {
            for bx in 0..self.block_size.min(width) {
                let x1 = x + bx;
                let y1 = y + by;

                let x2 = (x1 as isize + dx).max(0).min((width - 1) as isize) as usize;
                let y2 = (y1 as isize + dy).max(0).min((height - 1) as isize) as usize;

                let idx1 = (y1 * width + x1) * 3;
                let idx2 = (y2 * width + x2) * 3;

                if idx1 + 2 < frame1.len() && idx2 + 2 < frame2.len() {
                    for c in 0..3 {
                        sad += u32::from(
                            (i16::from(frame1[idx1 + c]) - i16::from(frame2[idx2 + c]))
                                .unsigned_abs(),
                        );
                    }
                }
            }
        }

        sad
    }
}

/// Rolling shutter corrector
pub struct RollingShutterCorrector {
    /// Camera parameters
    pub params: RollingShutterParams,
    /// Motion estimator
    estimator: RollingShutterEstimator,
}

impl RollingShutterCorrector {
    /// Create a new rolling shutter corrector
    #[must_use]
    pub fn new(params: RollingShutterParams) -> Self {
        Self {
            params,
            estimator: RollingShutterEstimator::default(),
        }
    }

    /// Correct rolling shutter in a frame
    ///
    /// # Errors
    /// Returns error if correction fails
    pub fn correct(
        &self,
        frame: &[u8],
        motion_vectors: &[MotionVector],
        width: usize,
        height: usize,
    ) -> AlignResult<Vec<u8>> {
        if frame.len() != width * height * 3 {
            return Err(AlignError::InvalidConfig("Frame size mismatch".to_string()));
        }

        let mut corrected = vec![0u8; width * height * 3];

        // Apply motion compensation per scanline
        for (block_idx, mv) in motion_vectors.iter().enumerate() {
            let y_start = block_idx * self.estimator.block_size;
            let y_end = (y_start + self.estimator.block_size).min(height);

            for y in y_start..y_end {
                self.correct_scanline(frame, &mut corrected, width, y, mv);
            }
        }

        Ok(corrected)
    }

    /// Correct a single scanline
    fn correct_scanline(
        &self,
        input: &[u8],
        output: &mut [u8],
        width: usize,
        y: usize,
        mv: &MotionVector,
    ) {
        for x in 0..width {
            let src_x = (x as f32 - mv.dx).round() as isize;
            let src_y = (y as f32 - mv.dy).round() as isize;

            if src_x >= 0 && src_x < width as isize && src_y >= 0 {
                let src_idx = (src_y as usize * width + src_x as usize) * 3;
                let dst_idx = (y * width + x) * 3;

                if src_idx + 2 < input.len() && dst_idx + 2 < output.len() {
                    output[dst_idx..dst_idx + 3].copy_from_slice(&input[src_idx..src_idx + 3]);
                }
            }
        }
    }

    /// Estimate and correct rolling shutter in one step
    ///
    /// # Errors
    /// Returns error if correction fails
    pub fn estimate_and_correct(
        &self,
        frame1: &[u8],
        frame2: &[u8],
        width: usize,
        height: usize,
    ) -> AlignResult<Vec<u8>> {
        let motion_vectors = self
            .estimator
            .estimate_motion(frame1, frame2, width, height)?;
        self.correct(frame2, &motion_vectors, width, height)
    }
}

/// Wobble detector for rolling shutter artifacts
pub struct WobbleDetector {
    /// Threshold for wobble detection
    pub threshold: f32,
}

impl Default for WobbleDetector {
    fn default() -> Self {
        Self { threshold: 5.0 }
    }
}

impl WobbleDetector {
    /// Create a new wobble detector
    #[must_use]
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }

    /// Detect wobble in motion vectors
    #[must_use]
    pub fn detect_wobble(&self, motion_vectors: &[MotionVector]) -> bool {
        if motion_vectors.len() < 3 {
            return false;
        }

        // Check for oscillating motion
        let mut sign_changes = 0;

        for i in 2..motion_vectors.len() {
            let d1 = motion_vectors[i - 1].dx - motion_vectors[i - 2].dx;
            let d2 = motion_vectors[i].dx - motion_vectors[i - 1].dx;

            if d1 * d2 < 0.0 && d1.abs() > self.threshold {
                sign_changes += 1;
            }
        }

        // If motion changes direction frequently, it's wobble
        sign_changes > motion_vectors.len() / 4
    }

    /// Compute wobble metric (0.0 = no wobble, 1.0 = severe wobble)
    #[must_use]
    pub fn compute_wobble_metric(&self, motion_vectors: &[MotionVector]) -> f32 {
        if motion_vectors.len() < 2 {
            return 0.0;
        }

        let mut total_variation = 0.0f32;

        for i in 1..motion_vectors.len() {
            let ddx = motion_vectors[i].dx - motion_vectors[i - 1].dx;
            let ddy = motion_vectors[i].dy - motion_vectors[i - 1].dy;
            total_variation += (ddx * ddx + ddy * ddy).sqrt();
        }

        let avg_variation = total_variation / (motion_vectors.len() - 1) as f32;

        // Normalize to 0-1 range (assuming max variation of 20 pixels)
        (avg_variation / 20.0).min(1.0)
    }
}

/// Skew corrector for rolling shutter-induced distortion
pub struct SkewCorrector {
    /// Angular velocity (radians per second)
    pub angular_velocity: f64,
}

impl SkewCorrector {
    /// Create a new skew corrector
    #[must_use]
    pub fn new(angular_velocity: f64) -> Self {
        Self { angular_velocity }
    }

    /// Correct skew in image
    ///
    /// # Errors
    /// Returns error if correction fails
    pub fn correct(
        &self,
        frame: &[u8],
        width: usize,
        height: usize,
        params: &RollingShutterParams,
    ) -> AlignResult<Vec<u8>> {
        if frame.len() != width * height * 3 {
            return Err(AlignError::InvalidConfig("Frame size mismatch".to_string()));
        }

        let mut corrected = vec![0u8; width * height * 3];

        for y in 0..height {
            let time = params.compute_scanline_time(y, height);
            let angle = self.angular_velocity * time;

            // Compute horizontal offset due to rotation
            let offset = (angle * (height as f64 / 2.0)) as isize;

            self.shift_scanline(frame, &mut corrected, width, y, offset);
        }

        Ok(corrected)
    }

    /// Shift a scanline horizontally
    fn shift_scanline(
        &self,
        input: &[u8],
        output: &mut [u8],
        width: usize,
        y: usize,
        offset: isize,
    ) {
        for x in 0..width {
            let src_x = (x as isize - offset).max(0).min((width - 1) as isize) as usize;

            let src_idx = (y * width + src_x) * 3;
            let dst_idx = (y * width + x) * 3;

            if src_idx + 2 < input.len() && dst_idx + 2 < output.len() {
                output[dst_idx..dst_idx + 3].copy_from_slice(&input[src_idx..src_idx + 3]);
            }
        }
    }
}

/// Temporal interpolator for global shutter simulation
pub struct GlobalShutterSimulator {
    /// Number of virtual sub-frames
    pub sub_frames: usize,
}

impl Default for GlobalShutterSimulator {
    fn default() -> Self {
        Self { sub_frames: 10 }
    }
}

impl GlobalShutterSimulator {
    /// Create a new global shutter simulator
    #[must_use]
    pub fn new(sub_frames: usize) -> Self {
        Self { sub_frames }
    }

    /// Simulate global shutter by averaging virtual sub-frames
    ///
    /// # Errors
    /// Returns error if simulation fails
    pub fn simulate(
        &self,
        frames: &[&[u8]],
        width: usize,
        height: usize,
        params: &RollingShutterParams,
    ) -> AlignResult<Vec<u8>> {
        if frames.is_empty() {
            return Err(AlignError::InsufficientData(
                "Need at least one frame".to_string(),
            ));
        }

        let mut output = vec![0u32; width * height * 3];

        // For each scanline, average contributions from multiple frames
        for y in 0..height {
            let _time = params.compute_scanline_time(y, height);

            for frame in frames {
                if frame.len() != width * height * 3 {
                    continue;
                }

                for x in 0..width {
                    let idx = (y * width + x) * 3;
                    if idx + 2 < frame.len() {
                        output[idx] += u32::from(frame[idx]);
                        output[idx + 1] += u32::from(frame[idx + 1]);
                        output[idx + 2] += u32::from(frame[idx + 2]);
                    }
                }
            }
        }

        // Average
        let n = frames.len() as u32;
        let result = output.iter().map(|&v| (v / n) as u8).collect();

        Ok(result)
    }
}

/// Temporal smoother for rolling shutter motion vectors.
///
/// Applies an exponentially weighted moving average (EWMA) across consecutive
/// frames to suppress frame-to-frame jitter in the per-scanline motion
/// estimates. This is critical for preventing flickering artifacts that occur
/// when raw per-frame motion vectors vary erratically.
///
/// # Algorithm
///
/// For each block index `i`, the smoother maintains a running estimate:
///
/// ```text
/// mv_smoothed[i] = alpha * mv_new[i] + (1 - alpha) * mv_prev[i]
/// ```
///
/// A lower `alpha` produces more temporal smoothing (more lag), while a higher
/// `alpha` responds faster to genuine motion changes.
pub struct TemporalSmoother {
    /// Smoothing factor in (0, 1].  Lower = smoother.
    alpha: f64,
    /// Previous smoothed motion vectors (one per block).
    state: Vec<MotionVector>,
}

impl TemporalSmoother {
    /// Create a new temporal smoother.
    ///
    /// `alpha` is clamped to `[0.01, 1.0]`.
    #[must_use]
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha: alpha.clamp(0.01, 1.0),
            state: Vec::new(),
        }
    }

    /// Smooth a new frame's motion vectors against the running average.
    ///
    /// On the first call the input is returned as-is (there is no history).
    /// Subsequent calls blend the new vectors with the accumulated state.
    ///
    /// If the number of blocks changes between calls (e.g. resolution change)
    /// the state is reset.
    pub fn smooth(&mut self, motion_vectors: &[MotionVector]) -> Vec<MotionVector> {
        if self.state.len() != motion_vectors.len() {
            // First frame or resolution change: initialise state
            self.state = motion_vectors.to_vec();
            return motion_vectors.to_vec();
        }

        let alpha = self.alpha as f32;
        let one_minus = 1.0 - alpha;

        let mut result = Vec::with_capacity(motion_vectors.len());
        for (prev, new) in self.state.iter_mut().zip(motion_vectors.iter()) {
            let dx = alpha * new.dx + one_minus * prev.dx;
            let dy = alpha * new.dy + one_minus * prev.dy;
            let conf = alpha * new.confidence + one_minus * prev.confidence;

            prev.dx = dx;
            prev.dy = dy;
            prev.confidence = conf;

            result.push(MotionVector::new(dx, dy, conf));
        }

        result
    }

    /// Reset the internal state so the next call starts fresh.
    pub fn reset(&mut self) {
        self.state.clear();
    }

    /// Current smoothing factor.
    #[must_use]
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Number of blocks tracked.
    #[must_use]
    pub fn num_blocks(&self) -> usize {
        self.state.len()
    }
}

/// Gaussian temporal smoother that keeps a window of past frames and applies
/// a weighted average across time.
///
/// This is more expensive than EWMA but produces less phase lag because the
/// kernel is symmetric (it uses future context when available via lookahead).
pub struct GaussianTemporalSmoother {
    /// Kernel half-size (total window = 2 * radius + 1).
    radius: usize,
    /// Precomputed 1-D Gaussian kernel weights.
    kernel: Vec<f64>,
    /// Ring buffer of recent motion vector frames.
    history: Vec<Vec<MotionVector>>,
    /// Maximum number of frames to store (= 2 * radius + 1).
    capacity: usize,
}

impl GaussianTemporalSmoother {
    /// Create a new Gaussian temporal smoother.
    ///
    /// * `radius` -- half-size of the Gaussian kernel.
    /// * `sigma` -- standard deviation (in frames).
    #[must_use]
    pub fn new(radius: usize, sigma: f64) -> Self {
        let sigma = sigma.max(0.1);
        let cap = 2 * radius + 1;
        let mut kernel = Vec::with_capacity(cap);
        for i in 0..cap {
            let x = i as f64 - radius as f64;
            kernel.push((-0.5 * x * x / (sigma * sigma)).exp());
        }
        // Normalise
        let sum: f64 = kernel.iter().sum();
        if sum > 1e-15 {
            for v in &mut kernel {
                *v /= sum;
            }
        }

        Self {
            radius,
            kernel,
            history: Vec::with_capacity(cap),
            capacity: cap,
        }
    }

    /// Push a new frame of motion vectors and return the smoothed result for
    /// the centre frame (i.e. with `radius` frames of look-ahead/look-behind
    /// when available).
    ///
    /// Until the buffer is full the result uses whatever history is available.
    pub fn push(&mut self, motion_vectors: &[MotionVector]) -> Vec<MotionVector> {
        self.history.push(motion_vectors.to_vec());
        if self.history.len() > self.capacity {
            self.history.remove(0);
        }

        let num_blocks = motion_vectors.len();
        let num_frames = self.history.len();

        // The "centre" index in the available history
        let centre = if num_frames > self.radius {
            num_frames - 1 - self.radius.min(num_frames - 1)
        } else {
            0
        };

        let mut result = Vec::with_capacity(num_blocks);
        for block_idx in 0..num_blocks {
            let mut sum_dx = 0.0_f64;
            let mut sum_dy = 0.0_f64;
            let mut sum_conf = 0.0_f64;
            let mut weight_total = 0.0_f64;

            for (frame_offset, frame) in self.history.iter().enumerate() {
                if block_idx >= frame.len() {
                    continue;
                }
                // Map frame_offset to kernel index relative to centre
                let ki = frame_offset as isize - centre as isize + self.radius as isize;
                if ki < 0 || ki >= self.kernel.len() as isize {
                    continue;
                }
                let w = self.kernel[ki as usize];
                let mv = &frame[block_idx];
                sum_dx += f64::from(mv.dx) * w;
                sum_dy += f64::from(mv.dy) * w;
                sum_conf += f64::from(mv.confidence) * w;
                weight_total += w;
            }

            if weight_total > 1e-15 {
                result.push(MotionVector::new(
                    (sum_dx / weight_total) as f32,
                    (sum_dy / weight_total) as f32,
                    (sum_conf / weight_total) as f32,
                ));
            } else {
                result.push(
                    motion_vectors
                        .get(block_idx)
                        .copied()
                        .unwrap_or(MotionVector::zero()),
                );
            }
        }

        result
    }

    /// Reset the history buffer.
    pub fn reset(&mut self) {
        self.history.clear();
    }

    /// Number of frames currently in the history buffer.
    #[must_use]
    pub fn history_len(&self) -> usize {
        self.history.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rolling_shutter_params() {
        let params = RollingShutterParams::new(0.033, 30.0, ReadoutDirection::TopToBottom);
        assert_eq!(params.readout_time, 0.033);
        assert_eq!(params.frame_rate, 30.0);
    }

    #[test]
    fn test_scanline_time() {
        let params = RollingShutterParams::new(0.01, 100.0, ReadoutDirection::TopToBottom);
        let time = params.compute_scanline_time(500, 1000);
        assert!((time - 0.005).abs() < 1e-10);
    }

    #[test]
    fn test_motion_vector() {
        let mv = MotionVector::new(10.0, 20.0, 0.9);
        assert_eq!(mv.dx, 10.0);
        assert_eq!(mv.dy, 20.0);
        assert_eq!(mv.confidence, 0.9);

        let mag = mv.magnitude();
        assert!((mag - (10.0f32 * 10.0 + 20.0 * 20.0).sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_zero_motion_vector() {
        let mv = MotionVector::zero();
        assert_eq!(mv.dx, 0.0);
        assert_eq!(mv.dy, 0.0);
        assert_eq!(mv.magnitude(), 0.0);
    }

    #[test]
    fn test_wobble_detector() {
        let detector = WobbleDetector::new(5.0);
        assert_eq!(detector.threshold, 5.0);
    }

    #[test]
    fn test_wobble_metric() {
        let detector = WobbleDetector::default();
        let vectors = vec![
            MotionVector::new(0.0, 0.0, 1.0),
            MotionVector::new(10.0, 0.0, 1.0),
            MotionVector::new(0.0, 0.0, 1.0),
            MotionVector::new(10.0, 0.0, 1.0),
        ];

        let metric = detector.compute_wobble_metric(&vectors);
        assert!(metric > 0.0);
    }

    #[test]
    fn test_skew_corrector() {
        let corrector = SkewCorrector::new(1.0);
        assert_eq!(corrector.angular_velocity, 1.0);
    }

    #[test]
    fn test_global_shutter_simulator() {
        let simulator = GlobalShutterSimulator::new(10);
        assert_eq!(simulator.sub_frames, 10);
    }

    #[test]
    fn test_readout_direction() {
        assert_eq!(ReadoutDirection::TopToBottom, ReadoutDirection::TopToBottom);
        assert_ne!(ReadoutDirection::TopToBottom, ReadoutDirection::BottomToTop);
    }

    // ── TemporalSmoother (EWMA) ─────────────────────────────────────────────

    #[test]
    fn test_temporal_smoother_first_frame_passthrough() {
        let mut smoother = TemporalSmoother::new(0.5);
        let mvs = vec![
            MotionVector::new(10.0, 5.0, 0.9),
            MotionVector::new(-3.0, 2.0, 0.8),
        ];
        let result = smoother.smooth(&mvs);
        assert_eq!(result.len(), 2);
        assert!((result[0].dx - 10.0).abs() < 1e-5);
        assert!((result[1].dy - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_temporal_smoother_convergence() {
        let mut smoother = TemporalSmoother::new(0.3);
        // Feed constant motion: smoother should converge to it
        let mvs = vec![MotionVector::new(4.0, -2.0, 1.0)];
        for _ in 0..50 {
            let _ = smoother.smooth(&mvs);
        }
        let result = smoother.smooth(&mvs);
        assert!(
            (result[0].dx - 4.0).abs() < 0.01,
            "should converge to 4.0, got {}",
            result[0].dx
        );
        assert!(
            (result[0].dy + 2.0).abs() < 0.01,
            "should converge to -2.0, got {}",
            result[0].dy
        );
    }

    #[test]
    fn test_temporal_smoother_dampens_jitter() {
        let mut smoother = TemporalSmoother::new(0.2);
        // Alternate between +10 and -10 (high-frequency jitter)
        let _ = smoother.smooth(&[MotionVector::new(10.0, 0.0, 1.0)]);
        for _ in 0..20 {
            let _ = smoother.smooth(&[MotionVector::new(-10.0, 0.0, 1.0)]);
            let _ = smoother.smooth(&[MotionVector::new(10.0, 0.0, 1.0)]);
        }
        let result = smoother.smooth(&[MotionVector::new(-10.0, 0.0, 1.0)]);
        // After many oscillations, the smoothed result should be near zero
        assert!(
            result[0].dx.abs() < 5.0,
            "jitter should be dampened, got {}",
            result[0].dx
        );
    }

    #[test]
    fn test_temporal_smoother_alpha_clamping() {
        let s1 = TemporalSmoother::new(0.0);
        assert!((s1.alpha() - 0.01).abs() < 1e-10);

        let s2 = TemporalSmoother::new(2.0);
        assert!((s2.alpha() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_temporal_smoother_reset() {
        let mut smoother = TemporalSmoother::new(0.5);
        let _ = smoother.smooth(&[MotionVector::new(5.0, 5.0, 1.0)]);
        assert_eq!(smoother.num_blocks(), 1);
        smoother.reset();
        assert_eq!(smoother.num_blocks(), 0);
    }

    // ── GaussianTemporalSmoother ────────────────────────────────────────────

    #[test]
    fn test_gaussian_smoother_constant_input() {
        let mut smoother = GaussianTemporalSmoother::new(2, 1.0);
        let mvs = vec![MotionVector::new(3.0, -1.0, 0.9)];
        for _ in 0..10 {
            let result = smoother.push(&mvs);
            assert_eq!(result.len(), 1);
            // With constant input, output should converge to input
            assert!((result[0].dx - 3.0).abs() < 0.5, "dx={}", result[0].dx);
        }
    }

    #[test]
    fn test_gaussian_smoother_dampens_spike() {
        let mut smoother = GaussianTemporalSmoother::new(2, 1.0);
        let normal = vec![MotionVector::new(0.0, 0.0, 1.0)];
        let spike = vec![MotionVector::new(100.0, 0.0, 1.0)];

        let _ = smoother.push(&normal);
        let _ = smoother.push(&normal);
        let result = smoother.push(&spike); // spike at the most recent frame
                                            // The spike should be attenuated because it's averaged with past zeros
        assert!(
            result[0].dx < 100.0,
            "spike should be dampened: dx={}",
            result[0].dx
        );
    }

    #[test]
    fn test_gaussian_smoother_history_len() {
        let mut smoother = GaussianTemporalSmoother::new(1, 0.5);
        assert_eq!(smoother.history_len(), 0);
        let mvs = vec![MotionVector::zero()];
        let _ = smoother.push(&mvs);
        assert_eq!(smoother.history_len(), 1);
        let _ = smoother.push(&mvs);
        let _ = smoother.push(&mvs);
        // Capacity is 2*1+1 = 3
        assert_eq!(smoother.history_len(), 3);
        let _ = smoother.push(&mvs);
        // Should evict oldest
        assert_eq!(smoother.history_len(), 3);
    }

    #[test]
    fn test_gaussian_smoother_reset() {
        let mut smoother = GaussianTemporalSmoother::new(2, 1.0);
        let _ = smoother.push(&[MotionVector::zero()]);
        assert_eq!(smoother.history_len(), 1);
        smoother.reset();
        assert_eq!(smoother.history_len(), 0);
    }
}
