//! Clip speed and reverse control for timeline clips.
//!
//! Supports normal, reverse, and variable-speed playback with
//! keyframe-based speed envelopes, freeze frames, and speed ramps.

#![allow(dead_code)]

/// The speed playback mode of a clip.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpeedMode {
    /// Normal forward playback at the source frame rate.
    Normal,
    /// Reversed playback at the source frame rate.
    Reverse,
    /// Variable speed driven by keyframes.
    Variable,
}

impl SpeedMode {
    /// Returns `true` if this is a variable-speed mode.
    #[must_use]
    pub fn is_variable(self) -> bool {
        matches!(self, Self::Variable)
    }

    /// Returns `true` if this is reverse playback.
    #[must_use]
    pub fn is_reverse(self) -> bool {
        matches!(self, Self::Reverse)
    }
}

/// Speed configuration for a clip.
#[derive(Debug, Clone)]
pub struct ClipSpeed {
    /// The speed mode.
    pub mode: SpeedMode,
    /// Speed multiplier (e.g. 0.5 = half speed, 2.0 = double speed).
    /// Ignored when `mode` is `Variable`.
    pub multiplier: f64,
    /// Source duration in frames at 1× speed.
    pub source_frames: u64,
}

impl ClipSpeed {
    /// Create a new `ClipSpeed` with the given mode, multiplier, and source frame count.
    #[must_use]
    pub fn new(mode: SpeedMode, multiplier: f64, source_frames: u64) -> Self {
        Self {
            mode,
            multiplier: multiplier.max(0.001),
            source_frames,
        }
    }

    /// Compute the output duration in frames after speed adjustment.
    ///
    /// For normal/reverse modes: `ceil(source_frames / multiplier)`.
    /// For variable mode: returns `source_frames` unchanged (envelope not evaluated here).
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation
    )]
    pub fn output_duration_frames(&self) -> u64 {
        if self.mode == SpeedMode::Variable {
            self.source_frames
        } else {
            let out = (self.source_frames as f64 / self.multiplier).ceil();
            out as u64
        }
    }

    /// Returns `true` if the clip is playing slower than real-time.
    #[must_use]
    pub fn is_slow_motion(&self) -> bool {
        self.multiplier < 1.0 && !matches!(self.mode, SpeedMode::Variable)
    }

    /// Returns `true` if playing at exactly normal speed and not reversed.
    #[must_use]
    pub fn is_normal(&self) -> bool {
        matches!(self.mode, SpeedMode::Normal) && (self.multiplier - 1.0).abs() < f64::EPSILON
    }
}

/// A keyframe that defines speed at a specific output frame.
#[derive(Debug, Clone, Copy)]
pub struct SpeedKeyframe {
    /// Output frame at which this keyframe applies.
    pub frame: u64,
    /// Speed multiplier at this keyframe.
    pub speed: f64,
}

impl SpeedKeyframe {
    /// Create a new `SpeedKeyframe`.
    #[must_use]
    pub fn new(frame: u64, speed: f64) -> Self {
        Self { frame, speed }
    }

    /// Linearly interpolate speed toward `other` at `t` (0.0–1.0).
    ///
    /// Returns the interpolated speed value.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn interpolate_to(&self, other: &Self, t: f64) -> f64 {
        let t = t.clamp(0.0, 1.0);
        self.speed + (other.speed - self.speed) * t
    }

    /// Compute `t` at a given output `frame` between `self` and `other`.
    ///
    /// Returns `None` if the keyframes are at the same frame.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn t_at_frame(&self, other: &Self, frame: u64) -> Option<f64> {
        let span = other.frame as f64 - self.frame as f64;
        if span.abs() < f64::EPSILON {
            return None;
        }
        Some(((frame as f64 - self.frame as f64) / span).clamp(0.0, 1.0))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SpeedEffect — high-level speed effect descriptor
// ─────────────────────────────────────────────────────────────────────────────

/// High-level description of a speed effect applied to a clip.
///
/// Use [`ClipSpeedController`] to map playhead frame numbers to source frame
/// numbers under a given `SpeedEffect`.
#[derive(Debug, Clone, PartialEq)]
pub enum SpeedEffect {
    /// Normal forward playback at 1× speed.
    Normal,
    /// Freeze the output at the source frame `at_frame` for the entire clip duration.
    FreezeFrame {
        /// Source frame index to hold.
        at_frame: u64,
    },
    /// Constant speed change (factor > 1.0 = faster, < 1.0 = slower).
    ConstantSpeed {
        /// Speed multiplier.  Must be > 0.
        factor: f32,
    },
    /// Variable (ramping) speed driven by a list of `(output_frame, speed_factor)` keyframes.
    ///
    /// Speed is linearly interpolated between consecutive keyframes.
    /// Keyframes are assumed to be sorted by `output_frame`.
    VariableSpeed {
        /// Keyframes as `(output_frame, speed_factor)` pairs, sorted ascending.
        keyframes: Vec<(u64, f32)>,
    },
}

impl Default for SpeedEffect {
    fn default() -> Self {
        Self::Normal
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ClipSpeedController — maps playhead frame → source frame
// ─────────────────────────────────────────────────────────────────────────────

/// Maps playhead time (in output frames) to source media frame numbers under a
/// given [`SpeedEffect`].
///
/// # Example
///
/// ```
/// use oximedia_edit::clip_speed::{SpeedEffect, ClipSpeedController};
///
/// let ctrl = ClipSpeedController::new(SpeedEffect::ConstantSpeed { factor: 2.0 }, 0);
/// // At playhead frame 10, the source should be at frame 20 (2× faster).
/// assert_eq!(ctrl.frame_at_playhead(10), 20);
/// ```
#[derive(Debug, Clone)]
pub struct ClipSpeedController {
    /// The speed effect to evaluate.
    pub effect: SpeedEffect,
    /// Source frame offset: the source frame index corresponding to output frame 0.
    pub source_offset: u64,
}

impl ClipSpeedController {
    /// Create a new controller for the given effect and source start frame.
    #[must_use]
    pub fn new(effect: SpeedEffect, source_offset: u64) -> Self {
        Self {
            effect,
            source_offset,
        }
    }

    /// Compute the source-media frame index that corresponds to `playhead_frame`
    /// (the zero-based output frame index within the clip).
    ///
    /// - [`SpeedEffect::Normal`]: source frame = `playhead_frame + source_offset`.
    /// - [`SpeedEffect::FreezeFrame`]: always returns `at_frame`.
    /// - [`SpeedEffect::ConstantSpeed`]: source frame = `floor(playhead_frame × factor) + source_offset`.
    /// - [`SpeedEffect::VariableSpeed`]: integrates the piecewise-linear speed envelope up to
    ///   `playhead_frame` and returns `floor(integral) + source_offset`.
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation
    )]
    pub fn frame_at_playhead(&self, playhead_frame: u64) -> u64 {
        match &self.effect {
            SpeedEffect::Normal => self.source_offset + playhead_frame,

            SpeedEffect::FreezeFrame { at_frame } => *at_frame,

            SpeedEffect::ConstantSpeed { factor } => {
                let source_delta = (playhead_frame as f64 * f64::from(*factor)) as u64;
                self.source_offset + source_delta
            }

            SpeedEffect::VariableSpeed { keyframes } => {
                if keyframes.is_empty() {
                    // No keyframes — treat as Normal.
                    return self.source_offset + playhead_frame;
                }

                let pf = playhead_frame as f64;
                let mut accumulated: f64 = 0.0;
                let mut prev_output: f64 = 0.0;
                let mut prev_speed: f64 = f64::from(keyframes[0].1);

                for &(kf_frame, kf_speed) in keyframes {
                    let kf_out = kf_frame as f64;
                    let kf_sp = f64::from(kf_speed);

                    if pf <= kf_out {
                        // Interpolate speed between prev and this keyframe.
                        let span = kf_out - prev_output;
                        let t = if span > 0.0 {
                            (pf - prev_output) / span
                        } else {
                            1.0
                        };
                        let interp_speed = prev_speed + (kf_sp - prev_speed) * t;
                        // Area under the trapezoid up to playhead_frame.
                        let seg_delta = pf - prev_output;
                        let avg_speed = (prev_speed + interp_speed) / 2.0;
                        accumulated += seg_delta * avg_speed;
                        let source_frame = (accumulated as u64) + self.source_offset;
                        return source_frame;
                    }

                    // Accumulate entire segment.
                    let seg_delta = kf_out - prev_output;
                    let avg_speed = (prev_speed + kf_sp) / 2.0;
                    accumulated += seg_delta * avg_speed;

                    prev_output = kf_out;
                    prev_speed = kf_sp;
                }

                // Playhead is beyond the last keyframe — continue at last speed.
                let remaining = pf - prev_output;
                accumulated += remaining * prev_speed;
                (accumulated as u64) + self.source_offset
            }
        }
    }

    /// Return the source frame held during a freeze, or `None` if the effect is not a freeze.
    #[must_use]
    pub fn freeze_frame(&self) -> Option<u64> {
        match &self.effect {
            SpeedEffect::FreezeFrame { at_frame } => Some(*at_frame),
            _ => None,
        }
    }

    /// Return the constant speed factor, or `None` for non-constant effects.
    #[must_use]
    pub fn constant_factor(&self) -> Option<f32> {
        match &self.effect {
            SpeedEffect::ConstantSpeed { factor } => Some(*factor),
            _ => None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speed_mode_is_variable() {
        assert!(SpeedMode::Variable.is_variable());
        assert!(!SpeedMode::Normal.is_variable());
        assert!(!SpeedMode::Reverse.is_variable());
    }

    #[test]
    fn test_speed_mode_is_reverse() {
        assert!(SpeedMode::Reverse.is_reverse());
        assert!(!SpeedMode::Normal.is_reverse());
    }

    #[test]
    fn test_clip_speed_output_duration_normal() {
        let cs = ClipSpeed::new(SpeedMode::Normal, 2.0, 100);
        assert_eq!(cs.output_duration_frames(), 50);
    }

    #[test]
    fn test_clip_speed_output_duration_slow_motion() {
        let cs = ClipSpeed::new(SpeedMode::Normal, 0.5, 100);
        assert_eq!(cs.output_duration_frames(), 200);
    }

    #[test]
    fn test_clip_speed_output_duration_reverse() {
        let cs = ClipSpeed::new(SpeedMode::Reverse, 1.0, 60);
        assert_eq!(cs.output_duration_frames(), 60);
    }

    #[test]
    fn test_clip_speed_output_duration_variable() {
        let cs = ClipSpeed::new(SpeedMode::Variable, 2.0, 120);
        // Variable mode returns source_frames unchanged
        assert_eq!(cs.output_duration_frames(), 120);
    }

    #[test]
    fn test_clip_speed_is_slow_motion_true() {
        let cs = ClipSpeed::new(SpeedMode::Normal, 0.25, 100);
        assert!(cs.is_slow_motion());
    }

    #[test]
    fn test_clip_speed_is_slow_motion_false_fast() {
        let cs = ClipSpeed::new(SpeedMode::Normal, 2.0, 100);
        assert!(!cs.is_slow_motion());
    }

    #[test]
    fn test_clip_speed_is_slow_motion_false_variable() {
        let cs = ClipSpeed::new(SpeedMode::Variable, 0.1, 100);
        assert!(!cs.is_slow_motion());
    }

    #[test]
    fn test_clip_speed_is_normal() {
        let cs = ClipSpeed::new(SpeedMode::Normal, 1.0, 50);
        assert!(cs.is_normal());
    }

    #[test]
    fn test_clip_speed_not_normal_when_reverse() {
        let cs = ClipSpeed::new(SpeedMode::Reverse, 1.0, 50);
        assert!(!cs.is_normal());
    }

    #[test]
    fn test_speed_keyframe_interpolate_midpoint() {
        let a = SpeedKeyframe::new(0, 1.0);
        let b = SpeedKeyframe::new(100, 2.0);
        let mid = a.interpolate_to(&b, 0.5);
        assert!((mid - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_speed_keyframe_interpolate_start() {
        let a = SpeedKeyframe::new(0, 0.5);
        let b = SpeedKeyframe::new(100, 4.0);
        assert!((a.interpolate_to(&b, 0.0) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_speed_keyframe_interpolate_end() {
        let a = SpeedKeyframe::new(0, 0.5);
        let b = SpeedKeyframe::new(100, 4.0);
        assert!((a.interpolate_to(&b, 1.0) - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_speed_keyframe_t_at_frame() {
        let a = SpeedKeyframe::new(0, 1.0);
        let b = SpeedKeyframe::new(100, 2.0);
        let t = a.t_at_frame(&b, 50).expect("t should be valid");
        assert!((t - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_speed_keyframe_t_at_frame_same_frame_returns_none() {
        let a = SpeedKeyframe::new(10, 1.0);
        let b = SpeedKeyframe::new(10, 2.0);
        assert!(a.t_at_frame(&b, 10).is_none());
    }

    // ── SpeedEffect / ClipSpeedController tests ─────────────────────────────

    #[test]
    fn test_speed_effect_normal() {
        let ctrl = ClipSpeedController::new(SpeedEffect::Normal, 0);
        assert_eq!(ctrl.frame_at_playhead(0), 0);
        assert_eq!(ctrl.frame_at_playhead(10), 10);
        assert_eq!(ctrl.frame_at_playhead(100), 100);
    }

    #[test]
    fn test_speed_effect_normal_with_source_offset() {
        let ctrl = ClipSpeedController::new(SpeedEffect::Normal, 50);
        assert_eq!(ctrl.frame_at_playhead(0), 50);
        assert_eq!(ctrl.frame_at_playhead(10), 60);
    }

    /// The key test required by the task specification.
    #[test]
    fn test_speed_freeze_frame() {
        let ctrl = ClipSpeedController::new(SpeedEffect::FreezeFrame { at_frame: 42 }, 0);
        // Regardless of playhead position, always returns the frozen source frame.
        assert_eq!(ctrl.frame_at_playhead(0), 42);
        assert_eq!(ctrl.frame_at_playhead(100), 42);
        assert_eq!(ctrl.frame_at_playhead(9999), 42);
        assert_eq!(ctrl.freeze_frame(), Some(42));
    }

    #[test]
    fn test_speed_effect_constant_double() {
        let ctrl = ClipSpeedController::new(SpeedEffect::ConstantSpeed { factor: 2.0 }, 0);
        assert_eq!(ctrl.frame_at_playhead(0), 0);
        assert_eq!(ctrl.frame_at_playhead(10), 20);
        assert_eq!(ctrl.frame_at_playhead(50), 100);
        assert_eq!(ctrl.constant_factor(), Some(2.0));
    }

    #[test]
    fn test_speed_effect_constant_half() {
        let ctrl = ClipSpeedController::new(SpeedEffect::ConstantSpeed { factor: 0.5 }, 0);
        // At playhead 10, source should be 5 (half speed).
        assert_eq!(ctrl.frame_at_playhead(10), 5);
        assert_eq!(ctrl.frame_at_playhead(20), 10);
    }

    #[test]
    fn test_speed_effect_constant_with_offset() {
        let ctrl = ClipSpeedController::new(SpeedEffect::ConstantSpeed { factor: 2.0 }, 10);
        // source_offset=10, so at playhead 5 → 10 + floor(5*2.0) = 20.
        assert_eq!(ctrl.frame_at_playhead(5), 20);
    }

    #[test]
    fn test_speed_effect_variable_empty_keyframes() {
        let ctrl = ClipSpeedController::new(
            SpeedEffect::VariableSpeed {
                keyframes: Vec::new(),
            },
            0,
        );
        // Falls back to Normal behaviour.
        assert_eq!(ctrl.frame_at_playhead(10), 10);
    }

    #[test]
    fn test_speed_effect_variable_single_constant_segment() {
        // One keyframe at frame 100 with speed 2.0.
        // Before that keyframe, speed linearly goes from 2.0 (first keyframe speed) to 2.0.
        let ctrl = ClipSpeedController::new(
            SpeedEffect::VariableSpeed {
                keyframes: vec![(100, 2.0)],
            },
            0,
        );
        // At playhead 50 (half of the segment), source = 50*2.0 = 100.
        assert_eq!(ctrl.frame_at_playhead(50), 100);
    }

    #[test]
    fn test_speed_effect_variable_ramp() {
        // Speed ramps from 1.0 at frame 0 to 2.0 at frame 100.
        // Use two keyframes: (0, 1.0) and (100, 2.0).
        let ctrl = ClipSpeedController::new(
            SpeedEffect::VariableSpeed {
                keyframes: vec![(0, 1.0), (100, 2.0)],
            },
            0,
        );
        // At playhead 0: source = 0.
        assert_eq!(ctrl.frame_at_playhead(0), 0);
        // At playhead beyond last keyframe: uses last speed (2.0).
        // At pf=150: integral(0..100 of lerp 1→2) + (150-100)*2.0
        //   = area of trapezoid with parallel sides 1 and 2, width 100: (1+2)/2*100 = 150
        //   + 50*2 = 100  →  total = 250
        assert_eq!(ctrl.frame_at_playhead(150), 250);
    }

    #[test]
    fn test_speed_effect_default_is_normal() {
        assert_eq!(SpeedEffect::default(), SpeedEffect::Normal);
    }

    #[test]
    fn test_freeze_frame_returns_none_for_non_freeze() {
        let ctrl = ClipSpeedController::new(SpeedEffect::Normal, 0);
        assert!(ctrl.freeze_frame().is_none());

        let ctrl2 = ClipSpeedController::new(SpeedEffect::ConstantSpeed { factor: 1.5 }, 0);
        assert!(ctrl2.freeze_frame().is_none());
    }

    #[test]
    fn test_constant_factor_returns_none_for_non_constant() {
        let ctrl = ClipSpeedController::new(SpeedEffect::Normal, 0);
        assert!(ctrl.constant_factor().is_none());
    }
}
