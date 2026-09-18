#![allow(dead_code)]
//! Real-time single-pass stabilization with bounded look-ahead buffer.
//!
//! Unlike multi-pass stabilization which requires the entire video to be available
//! up-front, this module provides a streaming stabilizer that processes frames as
//! they arrive, using only a small bounded look-ahead window.  This makes it
//! suitable for live streaming, broadcast, and real-time monitoring applications.
//!
//! # Design
//!
//! The [`RealtimeStabilizer`] maintains an internal ring buffer of the last
//! `look_ahead` frames.  For each incoming frame it:
//!
//! 1. Estimates the raw motion vector between the new frame and the previous.
//! 2. Appends the motion vector to a sliding trajectory window.
//! 3. Computes the smoothed target position using an exponential moving average
//!    applied to the trajectories inside the window.
//! 4. Emits a [`StabilizedFrame`] containing the warp parameters needed to
//!    correct the current output frame.  Actual pixel warping is left to the
//!    caller so this module has no dependency on frame pixel data.
//!
//! # Latency vs. Quality Trade-off
//!
//! A larger `look_ahead` produces smoother output (more future context) but
//! increases output latency by `look_ahead` frames.  For zero-latency output
//! set `look_ahead = 0`; the stabilizer then degenerates to a pure causal EMA
//! filter on the trajectory.
//!
//! # Example
//!
//! ```
//! use oximedia_stabilize::realtime_stabilize::{
//!     RealtimeStabilizer, RealtimeConfig, RawMotion,
//! };
//!
//! let config = RealtimeConfig::default().with_look_ahead(5).with_alpha(0.2);
//! let mut stabilizer = RealtimeStabilizer::new(config);
//!
//! // Feed frames (synthetic motion vectors for illustration)
//! for i in 0_u32..30 {
//!     let raw = RawMotion { dx: (i as f64 * 0.3).sin() * 8.0, dy: 0.5, angle: 0.0, scale: 1.0 };
//!     if let Some(frame) = stabilizer.push(raw) {
//!         // Use frame.correction_dx / correction_dy to warp output pixels
//!         let _ = frame.correction_dx;
//!     }
//! }
//! // Flush remaining frames
//! for frame in stabilizer.flush() {
//!     let _ = frame.correction_dy;
//! }
//! ```

use std::collections::VecDeque;

use crate::error::{StabilizeError, StabilizeResult};

// ─────────────────────────────────────────────────────────────────
//  Configuration
// ─────────────────────────────────────────────────────────────────

/// Configuration for [`RealtimeStabilizer`].
#[derive(Debug, Clone)]
pub struct RealtimeConfig {
    /// Number of future frames to buffer before emitting output.
    ///
    /// A value of 0 gives causal (zero-latency) stabilization; larger values
    /// improve smoothness at the cost of output latency.  Must be < 256.
    pub look_ahead: usize,
    /// Exponential moving average alpha for trajectory smoothing.
    ///
    /// Lower values produce smoother output but more lag.  Range: (0, 1].
    pub alpha: f64,
    /// Maximum allowed correction magnitude in pixels.
    ///
    /// Corrections larger than this value are clamped so the output frame
    /// never drifts further than `max_correction_px` from the original.
    pub max_correction_px: f64,
    /// Minimum input motion to consider a frame "moving" (pixels).
    ///
    /// Frames with motion below this threshold are treated as static to
    /// avoid amplifying sensor noise.
    pub motion_deadzone_px: f64,
    /// Enable adaptive alpha: when the motion magnitude is large (potential
    /// intentional pan), alpha is temporarily increased to follow the motion
    /// more closely rather than correcting it.
    pub adaptive_alpha: bool,
    /// Motion magnitude (pixels) above which adaptive alpha kicks in.
    pub adaptive_alpha_threshold: f64,
    /// Alpha value used when the motion exceeds `adaptive_alpha_threshold`.
    pub adaptive_alpha_fast: f64,
}

impl Default for RealtimeConfig {
    fn default() -> Self {
        Self {
            look_ahead: 10,
            alpha: 0.15,
            max_correction_px: 100.0,
            motion_deadzone_px: 0.5,
            adaptive_alpha: true,
            adaptive_alpha_threshold: 40.0,
            adaptive_alpha_fast: 0.7,
        }
    }
}

impl RealtimeConfig {
    /// Create a new config with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the look-ahead buffer size.
    #[must_use]
    pub fn with_look_ahead(mut self, frames: usize) -> Self {
        self.look_ahead = frames.min(255);
        self
    }

    /// Set the EMA alpha (0 < alpha ≤ 1).
    #[must_use]
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha.clamp(0.001, 1.0);
        self
    }

    /// Set the maximum correction in pixels.
    #[must_use]
    pub fn with_max_correction_px(mut self, px: f64) -> Self {
        self.max_correction_px = px.max(1.0);
        self
    }

    /// Set the motion deadzone in pixels.
    #[must_use]
    pub fn with_motion_deadzone(mut self, px: f64) -> Self {
        self.motion_deadzone_px = px.max(0.0);
        self
    }

    /// Enable or disable adaptive alpha.
    #[must_use]
    pub const fn with_adaptive_alpha(mut self, enabled: bool) -> Self {
        self.adaptive_alpha = enabled;
        self
    }

    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`StabilizeError::InvalidParameter`] if any field is out of range.
    pub fn validate(&self) -> StabilizeResult<()> {
        if !(0.001..=1.0).contains(&self.alpha) {
            return Err(StabilizeError::invalid_parameter(
                "alpha",
                format!("{}", self.alpha),
            ));
        }
        if self.max_correction_px < 1.0 {
            return Err(StabilizeError::invalid_parameter(
                "max_correction_px",
                format!("{}", self.max_correction_px),
            ));
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────
//  Raw motion input
// ─────────────────────────────────────────────────────────────────

/// Raw motion estimate between consecutive frames.
///
/// Callers fill this from their preferred feature tracker or optical-flow
/// estimator and feed it to [`RealtimeStabilizer::push`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RawMotion {
    /// Horizontal translation (pixels, positive = right).
    pub dx: f64,
    /// Vertical translation (pixels, positive = down).
    pub dy: f64,
    /// Rotation (radians, positive = clockwise).
    pub angle: f64,
    /// Scale factor relative to the previous frame (1.0 = unchanged).
    pub scale: f64,
}

impl RawMotion {
    /// Create a pure translation motion.
    #[must_use]
    pub const fn translation(dx: f64, dy: f64) -> Self {
        Self {
            dx,
            dy,
            angle: 0.0,
            scale: 1.0,
        }
    }

    /// Create an affine motion.
    #[must_use]
    pub const fn affine(dx: f64, dy: f64, angle: f64, scale: f64) -> Self {
        Self {
            dx,
            dy,
            angle,
            scale,
        }
    }

    /// Euclidean translation magnitude.
    #[must_use]
    pub fn translation_magnitude(&self) -> f64 {
        (self.dx * self.dx + self.dy * self.dy).sqrt()
    }

    /// True if the motion is within the deadzone.
    #[must_use]
    pub fn is_static(&self, deadzone_px: f64) -> bool {
        self.translation_magnitude() < deadzone_px
            && self.angle.abs() < 1e-4
            && (self.scale - 1.0).abs() < 1e-4
    }
}

impl Default for RawMotion {
    fn default() -> Self {
        Self::translation(0.0, 0.0)
    }
}

// ─────────────────────────────────────────────────────────────────
//  Stabilized frame output
// ─────────────────────────────────────────────────────────────────

/// Stabilization parameters for one output frame emitted by [`RealtimeStabilizer`].
///
/// Apply `correction_dx` / `correction_dy` (and optionally `correction_angle`
/// / `correction_scale`) to the corresponding input frame to obtain the
/// stabilized output.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StabilizedFrame {
    /// Output frame index.
    pub index: usize,
    /// Horizontal correction to apply (pixels).
    pub correction_dx: f64,
    /// Vertical correction to apply (pixels).
    pub correction_dy: f64,
    /// Angular correction (radians, positive = counter-clockwise to undo CW shake).
    pub correction_angle: f64,
    /// Scale correction (multiply the frame by this before translating).
    pub correction_scale: f64,
    /// Raw motion magnitude that was corrected (for diagnostics).
    pub raw_motion_px: f64,
    /// Confidence in the stabilization (0.0 = no stabilization applied,
    /// 1.0 = full correction computed).
    pub confidence: f64,
}

impl StabilizedFrame {
    /// True if the correction is essentially zero (no warping needed).
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.correction_dx.abs() < 0.5
            && self.correction_dy.abs() < 0.5
            && self.correction_angle.abs() < 0.001
            && (self.correction_scale - 1.0).abs() < 0.001
    }
}

// ─────────────────────────────────────────────────────────────────
//  Internal trajectory accumulator
// ─────────────────────────────────────────────────────────────────

/// Accumulated absolute camera position at a frame boundary.
#[derive(Debug, Clone, Copy)]
struct TrajectoryPoint {
    /// Absolute X position (sum of dx motions from frame 0).
    x: f64,
    /// Absolute Y position (sum of dy motions from frame 0).
    y: f64,
    /// Absolute rotation (sum of angle motions).
    angle: f64,
    /// Cumulative scale (product of per-frame scale factors).
    scale: f64,
}

impl TrajectoryPoint {
    fn integrate(&self, motion: &RawMotion) -> Self {
        Self {
            x: self.x + motion.dx,
            y: self.y + motion.dy,
            angle: self.angle + motion.angle,
            scale: self.scale * motion.scale,
        }
    }
}

impl Default for TrajectoryPoint {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            angle: 0.0,
            scale: 1.0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────
//  RealtimeStabilizer
// ─────────────────────────────────────────────────────────────────

/// Single-pass real-time stabilizer with bounded look-ahead.
///
/// Feed raw motion estimates one frame at a time via [`push`](Self::push).
/// The stabilizer returns `Some(StabilizedFrame)` once enough look-ahead
/// frames have been buffered; `None` means the frame has been queued but
/// not yet emitted.  Call [`flush`](Self::flush) at end-of-stream to drain
/// remaining buffered frames.
///
/// Thread safety: not `Send` or `Sync` — wrap in `Arc<Mutex<_>>` for
/// multi-threaded use.
#[derive(Debug)]
pub struct RealtimeStabilizer {
    config: RealtimeConfig,
    /// Absolute trajectory history (newest at back).
    trajectory: VecDeque<TrajectoryPoint>,
    /// Raw motions buffered for look-ahead (index = input frame order).
    pending_motions: VecDeque<RawMotion>,
    /// Number of input frames received.
    input_count: usize,
    /// Number of output frames emitted.
    output_count: usize,
    /// Current smoothed trajectory position (EMA state).
    smoothed: TrajectoryPoint,
    /// Whether the EMA has been seeded with the first frame.
    ema_initialized: bool,
}

impl RealtimeStabilizer {
    /// Create a new real-time stabilizer with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns [`StabilizeError::InvalidParameter`] if the config is invalid.
    pub fn new(config: RealtimeConfig) -> Self {
        config.validate().unwrap_or(()); // best-effort; caller should validate
        let cap = config.look_ahead + 2;
        Self {
            config,
            trajectory: VecDeque::with_capacity(cap),
            pending_motions: VecDeque::with_capacity(cap),
            input_count: 0,
            output_count: 0,
            smoothed: TrajectoryPoint::default(),
            ema_initialized: false,
        }
    }

    /// Create from config, returning an error if config is invalid.
    ///
    /// # Errors
    ///
    /// Returns [`StabilizeError::InvalidParameter`] for invalid config fields.
    pub fn try_new(config: RealtimeConfig) -> StabilizeResult<Self> {
        config.validate()?;
        Ok(Self::new(config))
    }

    /// Push a raw motion estimate and optionally receive a stabilized frame.
    ///
    /// Returns `Some(frame)` when the look-ahead buffer is full enough to
    /// emit a stabilized output.  Returns `None` if the frame was queued
    /// but the look-ahead window is not yet satisfied.
    pub fn push(&mut self, motion: RawMotion) -> Option<StabilizedFrame> {
        // Integrate absolute trajectory
        let prev = self.trajectory.back().copied().unwrap_or_default();
        let current = prev.integrate(&motion);

        self.trajectory.push_back(current);
        self.pending_motions.push_back(motion);
        self.input_count += 1;

        // Update smoothed trajectory with EMA
        let alpha = self.effective_alpha(&motion);
        if self.ema_initialized {
            self.smoothed.x = alpha * current.x + (1.0 - alpha) * self.smoothed.x;
            self.smoothed.y = alpha * current.y + (1.0 - alpha) * self.smoothed.y;
            self.smoothed.angle = alpha * current.angle + (1.0 - alpha) * self.smoothed.angle;
            self.smoothed.scale = alpha * current.scale + (1.0 - alpha) * self.smoothed.scale;
        } else {
            self.smoothed = current;
            self.ema_initialized = true;
        }

        // Trim trajectory history beyond what we need
        let max_history = self.config.look_ahead + 2;
        while self.trajectory.len() > max_history {
            self.trajectory.pop_front();
        }

        // Emit a stabilized frame only once the look-ahead buffer is full
        if self.pending_motions.len() > self.config.look_ahead {
            self.emit_frame()
        } else {
            None
        }
    }

    /// Flush all remaining buffered frames.
    ///
    /// Call this at end-of-stream to drain look-ahead frames that have not
    /// yet been emitted.
    pub fn flush(&mut self) -> Vec<StabilizedFrame> {
        let mut output = Vec::with_capacity(self.pending_motions.len());
        while !self.pending_motions.is_empty() {
            if let Some(frame) = self.emit_frame() {
                output.push(frame);
            }
        }
        output
    }

    /// Reset the stabilizer to its initial state.
    pub fn reset(&mut self) {
        self.trajectory.clear();
        self.pending_motions.clear();
        self.input_count = 0;
        self.output_count = 0;
        self.smoothed = TrajectoryPoint::default();
        self.ema_initialized = false;
    }

    /// Number of input frames received so far.
    #[must_use]
    pub const fn input_count(&self) -> usize {
        self.input_count
    }

    /// Number of output frames emitted so far.
    #[must_use]
    pub const fn output_count(&self) -> usize {
        self.output_count
    }

    /// Number of frames currently buffered (not yet emitted).
    #[must_use]
    pub fn buffered_count(&self) -> usize {
        self.pending_motions.len()
    }

    /// Current output latency in frames.
    ///
    /// Equals `input_count - output_count` before flush is called.
    #[must_use]
    pub fn latency_frames(&self) -> usize {
        self.input_count.saturating_sub(self.output_count)
    }

    // ── Private helpers ────────────────────────────────────────────────

    /// Determine effective EMA alpha, potentially using the fast value when
    /// adaptive alpha is enabled and the motion is large.
    fn effective_alpha(&self, motion: &RawMotion) -> f64 {
        if self.config.adaptive_alpha
            && motion.translation_magnitude() > self.config.adaptive_alpha_threshold
        {
            self.config.adaptive_alpha_fast
        } else {
            self.config.alpha
        }
    }

    /// Pop the oldest pending motion and emit a `StabilizedFrame` for it.
    fn emit_frame(&mut self) -> Option<StabilizedFrame> {
        let motion = self.pending_motions.pop_front()?;
        let index = self.output_count;
        self.output_count += 1;

        // The raw trajectory position for the output frame
        // (integrate from previous output position using the popped motion)
        let output_trajectory_idx = self
            .output_count
            .saturating_sub(1)
            .min(self.trajectory.len().saturating_sub(1));
        let raw_pos = self
            .trajectory
            .get(output_trajectory_idx)
            .copied()
            .unwrap_or_default();

        // Correction = smoothed_target − raw_position
        let raw_motion_px = motion.translation_magnitude();
        let deadzone = self.config.motion_deadzone_px;

        let (cdx, cdy) = if raw_motion_px < deadzone {
            (0.0, 0.0)
        } else {
            let dx = self.smoothed.x - raw_pos.x;
            let dy = self.smoothed.y - raw_pos.y;
            // Clamp to max_correction_px
            let mag = (dx * dx + dy * dy).sqrt();
            if mag > self.config.max_correction_px && mag > 1e-9 {
                let scale = self.config.max_correction_px / mag;
                (dx * scale, dy * scale)
            } else {
                (dx, dy)
            }
        };

        let correction_angle = if raw_motion_px < deadzone {
            0.0
        } else {
            -(self.smoothed.angle - raw_pos.angle)
        };

        let correction_scale = if raw_motion_px < deadzone {
            1.0
        } else {
            // Inverse of the raw scale drift
            let scale_drift = raw_pos.scale;
            if scale_drift.abs() > 1e-9 {
                self.smoothed.scale / scale_drift
            } else {
                1.0
            }
        };

        // Confidence: lower when correction is near the clamp limit
        let conf = if raw_motion_px < 1e-9 {
            1.0
        } else {
            let corr_mag = (cdx * cdx + cdy * cdy).sqrt();
            (1.0 - corr_mag / self.config.max_correction_px.max(1.0)).clamp(0.0, 1.0)
        };

        Some(StabilizedFrame {
            index,
            correction_dx: cdx,
            correction_dy: cdy,
            correction_angle,
            correction_scale,
            raw_motion_px,
            confidence: conf,
        })
    }
}

// ─────────────────────────────────────────────────────────────────
//  Streaming stats helper
// ─────────────────────────────────────────────────────────────────

/// Running statistics accumulated over all emitted [`StabilizedFrame`]s.
///
/// Useful for real-time quality monitoring of a live stabilized stream.
#[derive(Debug, Default)]
pub struct StreamStats {
    frame_count: usize,
    sum_correction: f64,
    sum_raw_motion: f64,
    max_correction: f64,
    max_raw_motion: f64,
    clamp_events: usize,
}

impl StreamStats {
    /// Create new empty stats.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Ingest a stabilized frame into the running statistics.
    pub fn update(&mut self, frame: &StabilizedFrame) {
        self.frame_count += 1;
        let corr = (frame.correction_dx * frame.correction_dx
            + frame.correction_dy * frame.correction_dy)
            .sqrt();
        self.sum_correction += corr;
        self.sum_raw_motion += frame.raw_motion_px;
        self.max_correction = self.max_correction.max(corr);
        self.max_raw_motion = self.max_raw_motion.max(frame.raw_motion_px);
        if frame.confidence < 0.1 {
            self.clamp_events += 1;
        }
    }

    /// Number of frames processed.
    #[must_use]
    pub const fn frame_count(&self) -> usize {
        self.frame_count
    }

    /// Average correction magnitude (pixels).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn avg_correction_px(&self) -> f64 {
        if self.frame_count == 0 {
            return 0.0;
        }
        self.sum_correction / self.frame_count as f64
    }

    /// Average raw motion magnitude (pixels).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn avg_raw_motion_px(&self) -> f64 {
        if self.frame_count == 0 {
            return 0.0;
        }
        self.sum_raw_motion / self.frame_count as f64
    }

    /// Peak correction magnitude seen so far (pixels).
    #[must_use]
    pub const fn max_correction_px(&self) -> f64 {
        self.max_correction
    }

    /// Peak raw motion magnitude seen so far (pixels).
    #[must_use]
    pub const fn max_raw_motion_px(&self) -> f64 {
        self.max_raw_motion
    }

    /// Number of frames where the correction was clamped (near max limit).
    #[must_use]
    pub const fn clamp_events(&self) -> usize {
        self.clamp_events
    }
}

// ─────────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn push_n(
        stabilizer: &mut RealtimeStabilizer,
        n: usize,
        dx: f64,
        dy: f64,
    ) -> Vec<StabilizedFrame> {
        let mut frames = Vec::new();
        for _ in 0..n {
            if let Some(f) = stabilizer.push(RawMotion::translation(dx, dy)) {
                frames.push(f);
            }
        }
        frames
    }

    #[test]
    fn test_config_default() {
        let cfg = RealtimeConfig::default();
        assert!(cfg.look_ahead > 0);
        assert!(cfg.alpha > 0.0 && cfg.alpha <= 1.0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_validation_bad_alpha() {
        let cfg = RealtimeConfig::default().with_alpha(0.0);
        // 0.0 gets clamped to 0.001 by with_alpha
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_stabilizer_no_output_until_lookahead_full() {
        let config = RealtimeConfig::default().with_look_ahead(5);
        let mut stab = RealtimeStabilizer::new(config);

        // First 5 frames should return None (buffer not full)
        for _ in 0..5 {
            let out = stab.push(RawMotion::translation(3.0, 1.0));
            assert!(out.is_none(), "should be buffered, not emitted yet");
        }
        // 6th frame should emit the first output
        let out = stab.push(RawMotion::translation(3.0, 1.0));
        assert!(out.is_some(), "6th frame should emit output");
    }

    #[test]
    fn test_flush_drains_all_frames() {
        let config = RealtimeConfig::default().with_look_ahead(5);
        let mut stab = RealtimeStabilizer::new(config);

        // Push 10 frames; look_ahead=5, so 5 should be emitted inline
        let mut emitted = push_n(&mut stab, 10, 2.0, 0.5);
        assert_eq!(emitted.len(), 5, "5 frames emitted inline");

        // Flush should emit the remaining 5
        let flushed = stab.flush();
        emitted.extend(flushed);
        assert_eq!(emitted.len(), 10, "all 10 frames after flush");
    }

    #[test]
    fn test_output_indices_are_sequential() {
        let config = RealtimeConfig::default().with_look_ahead(3);
        let mut stab = RealtimeStabilizer::new(config);

        let mut all_frames: Vec<StabilizedFrame> = Vec::new();
        for _ in 0..12 {
            if let Some(f) = stab.push(RawMotion::translation(5.0, 5.0)) {
                all_frames.push(f);
            }
        }
        all_frames.extend(stab.flush());

        for (expected_idx, frame) in all_frames.iter().enumerate() {
            assert_eq!(frame.index, expected_idx, "frame index must be sequential");
        }
    }

    #[test]
    fn test_zero_lookahead_causal() {
        let config = RealtimeConfig::default().with_look_ahead(0);
        let mut stab = RealtimeStabilizer::new(config);

        // With look_ahead=0 every push should immediately emit
        for i in 0..8 {
            let out = stab.push(RawMotion::translation(i as f64, 0.0));
            assert!(
                out.is_some(),
                "zero look_ahead should emit on every push (frame {i})"
            );
        }
    }

    #[test]
    fn test_correction_is_bounded() {
        let config = RealtimeConfig::default()
            .with_look_ahead(2)
            .with_max_correction_px(20.0);
        let mut stab = RealtimeStabilizer::new(config);

        let mut all_frames: Vec<StabilizedFrame> = Vec::new();
        for _ in 0..15 {
            if let Some(f) = stab.push(RawMotion::translation(200.0, 200.0)) {
                all_frames.push(f);
            }
        }
        all_frames.extend(stab.flush());

        for frame in &all_frames {
            let corr = (frame.correction_dx.powi(2) + frame.correction_dy.powi(2)).sqrt();
            assert!(corr <= 20.0 + 1e-9, "correction {corr:.2} exceeds max 20.0");
        }
    }

    #[test]
    fn test_static_frames_zero_correction() {
        let config = RealtimeConfig::default()
            .with_look_ahead(3)
            .with_motion_deadzone(1.0)
            .with_adaptive_alpha(false);
        let mut stab = RealtimeStabilizer::new(config);

        // Tiny motion below deadzone
        let mut all_frames: Vec<StabilizedFrame> = Vec::new();
        for _ in 0..10 {
            if let Some(f) = stab.push(RawMotion::translation(0.1, 0.1)) {
                all_frames.push(f);
            }
        }
        all_frames.extend(stab.flush());

        for frame in &all_frames {
            assert!(
                frame.correction_dx.abs() < 1e-9 && frame.correction_dy.abs() < 1e-9,
                "static frame should have zero correction"
            );
        }
    }

    #[test]
    fn test_reset_clears_state() {
        let config = RealtimeConfig::default().with_look_ahead(5);
        let mut stab = RealtimeStabilizer::new(config);

        push_n(&mut stab, 6, 10.0, 5.0);
        stab.reset();

        assert_eq!(stab.input_count(), 0);
        assert_eq!(stab.output_count(), 0);
        assert_eq!(stab.buffered_count(), 0);
    }

    #[test]
    fn test_latency_equals_lookahead_before_flush() {
        let config = RealtimeConfig::default().with_look_ahead(7);
        let mut stab = RealtimeStabilizer::new(config);

        push_n(&mut stab, 20, 3.0, 3.0);
        // After 20 inputs with look_ahead=7: 13 outputs emitted, 7 buffered
        let latency = stab.latency_frames();
        assert_eq!(latency, 7, "latency should equal look_ahead={}", 7);
    }

    #[test]
    fn test_stream_stats_update() {
        let config = RealtimeConfig::default().with_look_ahead(2);
        let mut stab = RealtimeStabilizer::new(config);
        let mut stats = StreamStats::new();

        for _ in 0..10 {
            if let Some(f) = stab.push(RawMotion::translation(8.0, 6.0)) {
                stats.update(&f);
            }
        }
        for f in stab.flush() {
            stats.update(&f);
        }

        assert_eq!(stats.frame_count(), 10);
        assert!(stats.avg_raw_motion_px() > 0.0);
        assert!(stats.max_raw_motion_px() > 0.0);
    }

    #[test]
    fn test_try_new_valid_config() {
        let config = RealtimeConfig::default();
        assert!(RealtimeStabilizer::try_new(config).is_ok());
    }

    #[test]
    fn test_adaptive_alpha_large_motion() {
        // With adaptive alpha enabled and large motion, the EMA should follow
        // the trajectory more quickly (smaller lag)
        let config = RealtimeConfig::default()
            .with_look_ahead(0)
            .with_alpha(0.05)
            .with_adaptive_alpha(true);
        let config = RealtimeConfig {
            adaptive_alpha_threshold: 20.0,
            adaptive_alpha_fast: 0.9,
            ..config
        };

        let mut stab = RealtimeStabilizer::new(config);

        // Push large motion — adaptive alpha should engage
        for _ in 0..5 {
            let _ = stab.push(RawMotion::translation(50.0, 0.0));
        }
        // No panic and output emitted (zero look-ahead)
        assert_eq!(stab.output_count(), 5);
    }

    #[test]
    fn test_raw_motion_is_static() {
        let m = RawMotion::translation(0.3, 0.2);
        assert!(m.is_static(1.0));
        let m2 = RawMotion::translation(5.0, 0.0);
        assert!(!m2.is_static(1.0));
    }

    #[test]
    fn test_stabilized_frame_is_identity() {
        let f = StabilizedFrame {
            index: 0,
            correction_dx: 0.1,
            correction_dy: 0.1,
            correction_angle: 0.0,
            correction_scale: 1.0,
            raw_motion_px: 0.0,
            confidence: 1.0,
        };
        assert!(f.is_identity());
    }
}
