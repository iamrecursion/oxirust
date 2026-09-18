//! Timecode validation rules and violation reporting.
//!
//! This module provides structured validation of SMPTE timecodes against a
//! configurable set of rules, producing typed violation reports rather than
//! bare errors so callers can decide how to handle each issue.

#![allow(dead_code)]

use crate::Timecode;

// ── Validation rules ──────────────────────────────────────────────────────────

/// A single validation rule that can be applied to a timecode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationRule {
    /// Hours must be in 0–23.
    HoursInRange,
    /// Minutes must be in 0–59.
    MinutesInRange,
    /// Seconds must be in 0–59.
    SecondsInRange,
    /// Frame count must be in 0–(fps-1).
    FramesInRange,
    /// Frames 0 and 1 are illegal at the start of non-tenth minutes (DF only).
    DropFramePositions,
    /// Timecode must lie within an explicit allowed range \[start, end\].
    WithinRange,
}

impl std::fmt::Display for ValidationRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HoursInRange => write!(f, "hours-in-range"),
            Self::MinutesInRange => write!(f, "minutes-in-range"),
            Self::SecondsInRange => write!(f, "seconds-in-range"),
            Self::FramesInRange => write!(f, "frames-in-range"),
            Self::DropFramePositions => write!(f, "drop-frame-positions"),
            Self::WithinRange => write!(f, "within-range"),
        }
    }
}

// ── Violations ────────────────────────────────────────────────────────────────

/// A validation violation: which rule failed and a human-readable message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TcViolation {
    /// The rule that was violated.
    pub rule: ValidationRule,
    /// A description of the problem.
    pub message: String,
}

impl TcViolation {
    /// Create a new violation.
    pub fn new(rule: ValidationRule, message: impl Into<String>) -> Self {
        Self {
            rule,
            message: message.into(),
        }
    }
}

impl std::fmt::Display for TcViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}", self.rule, self.message)
    }
}

// ── Validator ─────────────────────────────────────────────────────────────────

/// Configuration for `TimecodeValidator`.
#[derive(Debug, Clone)]
pub struct ValidatorConfig {
    /// Rules to check. Defaults to all rules except `WithinRange`.
    pub rules: Vec<ValidationRule>,
    /// Optional allowed range `[start_frames, end_frames]` (inclusive).
    /// Only checked when `ValidationRule::WithinRange` is enabled.
    pub allowed_range: Option<(u64, u64)>,
}

impl Default for ValidatorConfig {
    fn default() -> Self {
        Self {
            rules: vec![
                ValidationRule::HoursInRange,
                ValidationRule::MinutesInRange,
                ValidationRule::SecondsInRange,
                ValidationRule::FramesInRange,
                ValidationRule::DropFramePositions,
            ],
            allowed_range: None,
        }
    }
}

/// Validates timecodes against a configurable set of rules.
///
/// # Example
/// ```
/// use oximedia_timecode::{Timecode, FrameRate};
/// use oximedia_timecode::tc_validator::{TimecodeValidator, ValidatorConfig};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let validator = TimecodeValidator::new(ValidatorConfig::default());
/// let tc = Timecode::new(1, 0, 0, 0, FrameRate::Fps25)?;
/// assert!(validator.validate(&tc).is_empty());
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct TimecodeValidator {
    config: ValidatorConfig,
}

impl TimecodeValidator {
    /// Create a new validator with the given configuration.
    pub fn new(config: ValidatorConfig) -> Self {
        Self { config }
    }

    /// Create a validator with default rules.
    pub fn default_validator() -> Self {
        Self::new(ValidatorConfig::default())
    }

    /// Validate a `Timecode` and return all violations found.
    /// An empty `Vec` means the timecode is valid under the configured rules.
    pub fn validate(&self, tc: &Timecode) -> Vec<TcViolation> {
        let mut violations = Vec::new();
        for &rule in &self.config.rules {
            match rule {
                ValidationRule::HoursInRange => {
                    if tc.hours > 23 {
                        violations.push(TcViolation::new(
                            rule,
                            format!("hours {} exceeds maximum of 23", tc.hours),
                        ));
                    }
                }
                ValidationRule::MinutesInRange => {
                    if tc.minutes > 59 {
                        violations.push(TcViolation::new(
                            rule,
                            format!("minutes {} exceeds maximum of 59", tc.minutes),
                        ));
                    }
                }
                ValidationRule::SecondsInRange => {
                    if tc.seconds > 59 {
                        violations.push(TcViolation::new(
                            rule,
                            format!("seconds {} exceeds maximum of 59", tc.seconds),
                        ));
                    }
                }
                ValidationRule::FramesInRange => {
                    if tc.frames >= tc.frame_rate.fps {
                        violations.push(TcViolation::new(
                            rule,
                            format!("frames {} >= fps {}", tc.frames, tc.frame_rate.fps),
                        ));
                    }
                }
                ValidationRule::DropFramePositions => {
                    if tc.frame_rate.drop_frame
                        && tc.seconds == 0
                        && tc.frames < 2
                        && !tc.minutes.is_multiple_of(10)
                    {
                        violations.push(TcViolation::new(
                            rule,
                            format!(
                                "frames {f} at {m}:00 is an illegal drop-frame position",
                                f = tc.frames,
                                m = tc.minutes,
                            ),
                        ));
                    }
                }
                ValidationRule::WithinRange => {
                    if let Some((start, end)) = self.config.allowed_range {
                        let pos = tc.to_frames();
                        if pos < start || pos > end {
                            violations.push(TcViolation::new(
                                rule,
                                format!(
                                    "frame position {pos} is outside allowed range [{start}, {end}]"
                                ),
                            ));
                        }
                    }
                }
            }
        }
        violations
    }

    /// Validate a range of consecutive timecodes for continuity.
    /// Returns violations for any timecode in the slice that fails validation.
    pub fn validate_range(&self, timecodes: &[Timecode]) -> Vec<(usize, TcViolation)> {
        let mut out = Vec::new();
        for (i, tc) in timecodes.iter().enumerate() {
            for v in self.validate(tc) {
                out.push((i, v));
            }
        }
        out
    }

    /// Return `true` if the timecode passes all configured rules.
    pub fn is_valid(&self, tc: &Timecode) -> bool {
        self.validate(tc).is_empty()
    }
}

// ── Helper: build a raw Timecode bypassing constructor checks ─────────────────

// ── Non-monotonic sequence detection ──────────────────────────────────────────

/// A single non-monotonic event detected in a timecode sequence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NonMonotonicEvent {
    /// Index of the timecode in the sequence that caused the event.
    pub frame_index: usize,
    /// The previous timecode (at `frame_index - 1`).
    pub prev_tc: Timecode,
    /// The current timecode (at `frame_index`).
    pub curr_tc: Timecode,
    /// Signed frame jump: `curr_tc.to_frames() - prev_tc.to_frames()`.
    /// Negative means backwards; very large positive means a forward skip.
    pub jump_frames: i64,
}

/// Scans a timecode sequence and reports positions where the timecode
/// does **not** advance monotonically by the expected one frame per step.
///
/// Only jumps whose absolute value exceeds `threshold_frames` are reported,
/// so callers can ignore minor jitter (e.g. `threshold_frames = 0` reports
/// every non-unit step; `threshold_frames = 1` only reports jumps ≥ 2 frames).
///
/// # Example
/// ```
/// use oximedia_timecode::{Timecode, FrameRate};
/// use oximedia_timecode::tc_validator::NonMonotonicDetector;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let tc0 = Timecode::new(0, 0, 0, 0, FrameRate::Fps25)?;
/// let tc1 = Timecode::new(0, 0, 2, 0, FrameRate::Fps25)?; // 2-second jump
/// let events = NonMonotonicDetector::new(1).scan_sequence(&[tc0, tc1]);
/// assert_eq!(events.len(), 1);
/// assert_eq!(events[0].frame_index, 1);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct NonMonotonicDetector {
    /// Only report absolute jumps strictly greater than this value.
    /// A threshold of `0` reports every non-unit step (including backwards).
    /// A threshold of `1` reports jumps whose absolute magnitude is > 1 frame
    /// (i.e. gaps of ≥ 2 or backwards movement of ≥ 2 frames).
    threshold_frames: i64,
}

impl NonMonotonicDetector {
    /// Create a new detector with the given threshold.
    ///
    /// `threshold_frames` is the *exclusive* lower bound on `|jump|` for
    /// events to be emitted. Set to `0` to report every non-unit step.
    pub fn new(threshold_frames: i64) -> Self {
        Self { threshold_frames }
    }

    /// Scan `timecodes` and return all non-monotonic events.
    ///
    /// The slice must contain at least 2 elements for any events to be
    /// produced; a slice of 0 or 1 elements always returns an empty `Vec`.
    pub fn scan_sequence(self, timecodes: &[Timecode]) -> Vec<NonMonotonicEvent> {
        let mut events = Vec::new();

        for i in 1..timecodes.len() {
            let prev = timecodes[i - 1];
            let curr = timecodes[i];

            let prev_f = prev.to_frames() as i64;
            let curr_f = curr.to_frames() as i64;
            let jump = curr_f - prev_f;

            // Expected monotonic step is +1; any other step is non-monotonic.
            // Only emit if |jump - 1| > threshold.
            let deviation = (jump - 1).abs();
            if deviation > self.threshold_frames {
                events.push(NonMonotonicEvent {
                    frame_index: i,
                    prev_tc: prev,
                    curr_tc: curr,
                    jump_frames: jump,
                });
            }
        }

        events
    }
}

// ── Helper: build a raw Timecode bypassing constructor checks ─────────────────

/// Build a `Timecode` directly without the safe constructor (for tests).
fn raw_timecode(hours: u8, minutes: u8, seconds: u8, frames: u8, fps: u8, drop: bool) -> Timecode {
    Timecode::from_raw_fields(hours, minutes, seconds, frames, fps, drop, 0)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FrameRate;

    fn valid_25fps() -> Timecode {
        Timecode::new(1, 30, 0, 12, FrameRate::Fps25).expect("valid timecode")
    }

    #[test]
    fn test_valid_timecode_no_violations() {
        let v = TimecodeValidator::default_validator();
        assert!(v.validate(&valid_25fps()).is_empty());
    }

    #[test]
    fn test_is_valid_returns_true_for_good_tc() {
        let v = TimecodeValidator::default_validator();
        assert!(v.is_valid(&valid_25fps()));
    }

    #[test]
    fn test_hours_out_of_range() {
        let tc = raw_timecode(24, 0, 0, 0, 25, false);
        let v = TimecodeValidator::default_validator();
        let vios = v.validate(&tc);
        assert!(vios.iter().any(|x| x.rule == ValidationRule::HoursInRange));
    }

    #[test]
    fn test_minutes_out_of_range() {
        let tc = raw_timecode(0, 60, 0, 0, 25, false);
        let v = TimecodeValidator::default_validator();
        let vios = v.validate(&tc);
        assert!(vios
            .iter()
            .any(|x| x.rule == ValidationRule::MinutesInRange));
    }

    #[test]
    fn test_seconds_out_of_range() {
        let tc = raw_timecode(0, 0, 60, 0, 25, false);
        let v = TimecodeValidator::default_validator();
        let vios = v.validate(&tc);
        assert!(vios
            .iter()
            .any(|x| x.rule == ValidationRule::SecondsInRange));
    }

    #[test]
    fn test_frames_out_of_range() {
        let tc = raw_timecode(0, 0, 0, 25, 25, false);
        let v = TimecodeValidator::default_validator();
        let vios = v.validate(&tc);
        assert!(vios.iter().any(|x| x.rule == ValidationRule::FramesInRange));
    }

    #[test]
    fn test_drop_frame_illegal_position_detected() {
        // Frames 0 at minute 1, second 0 — illegal in 29.97 DF
        let tc = raw_timecode(0, 1, 0, 0, 30, true);
        let v = TimecodeValidator::default_validator();
        let vios = v.validate(&tc);
        assert!(vios
            .iter()
            .any(|x| x.rule == ValidationRule::DropFramePositions));
    }

    #[test]
    fn test_drop_frame_tenth_minute_is_ok() {
        // Minute 10 is a "keep" minute for DF — frames 0 is legal
        let tc = raw_timecode(0, 10, 0, 0, 30, true);
        let v = TimecodeValidator::default_validator();
        let vios = v.validate(&tc);
        assert!(!vios
            .iter()
            .any(|x| x.rule == ValidationRule::DropFramePositions));
    }

    #[test]
    fn test_within_range_pass() {
        let tc = Timecode::new(0, 0, 1, 0, FrameRate::Fps25).expect("valid timecode"); // 25 frames
        let cfg = ValidatorConfig {
            rules: vec![ValidationRule::WithinRange],
            allowed_range: Some((0, 100)),
        };
        let v = TimecodeValidator::new(cfg);
        assert!(v.validate(&tc).is_empty());
    }

    #[test]
    fn test_within_range_fail() {
        let tc = Timecode::new(0, 0, 10, 0, FrameRate::Fps25).expect("valid timecode"); // 250 frames
        let cfg = ValidatorConfig {
            rules: vec![ValidationRule::WithinRange],
            allowed_range: Some((0, 100)),
        };
        let v = TimecodeValidator::new(cfg);
        let vios = v.validate(&tc);
        assert!(vios.iter().any(|x| x.rule == ValidationRule::WithinRange));
    }

    #[test]
    fn test_validate_range_empty_slice() {
        let v = TimecodeValidator::default_validator();
        assert!(v.validate_range(&[]).is_empty());
    }

    #[test]
    fn test_validate_range_all_valid() {
        let tcs: Vec<Timecode> = (0u8..5)
            .map(|f| Timecode::new(0, 0, 0, f, FrameRate::Fps25).expect("valid timecode"))
            .collect();
        let v = TimecodeValidator::default_validator();
        assert!(v.validate_range(&tcs).is_empty());
    }

    #[test]
    fn test_validate_range_with_violation() {
        let good = Timecode::new(0, 0, 0, 0, FrameRate::Fps25).expect("valid timecode");
        let bad = raw_timecode(0, 0, 0, 25, 25, false); // frames == fps
        let v = TimecodeValidator::default_validator();
        let results = v.validate_range(&[good, bad]);
        assert!(!results.is_empty());
        assert_eq!(results[0].0, 1); // index of bad timecode
    }

    #[test]
    fn test_rule_display() {
        assert_eq!(ValidationRule::HoursInRange.to_string(), "hours-in-range");
        assert_eq!(
            ValidationRule::DropFramePositions.to_string(),
            "drop-frame-positions"
        );
        assert_eq!(ValidationRule::WithinRange.to_string(), "within-range");
    }

    #[test]
    fn test_violation_display() {
        let v = TcViolation::new(ValidationRule::FramesInRange, "frames 30 >= fps 30");
        let s = v.to_string();
        assert!(s.contains("frames-in-range"));
        assert!(s.contains("frames 30"));
    }

    #[test]
    fn test_no_rules_produces_no_violations() {
        let tc = raw_timecode(99, 99, 99, 99, 25, false); // everything out of range
        let cfg = ValidatorConfig {
            rules: vec![],
            allowed_range: None,
        };
        let v = TimecodeValidator::new(cfg);
        assert!(v.validate(&tc).is_empty());
    }

    #[test]
    fn test_multiple_violations_accumulate() {
        let tc = raw_timecode(24, 60, 60, 25, 25, false);
        let v = TimecodeValidator::default_validator();
        let vios = v.validate(&tc);
        // Should find at least hours, minutes, seconds, frames violations
        assert!(vios.len() >= 4);
    }

    // ── NonMonotonicDetector tests ──────────────────────────────────────────

    #[test]
    fn test_non_monotonic_empty_slice() {
        let events = NonMonotonicDetector::new(0).scan_sequence(&[]);
        assert!(events.is_empty());
    }

    #[test]
    fn test_non_monotonic_single_element() {
        let tc = Timecode::new(0, 0, 0, 0, FrameRate::Fps25).expect("valid");
        let events = NonMonotonicDetector::new(0).scan_sequence(&[tc]);
        assert!(events.is_empty());
    }

    #[test]
    fn test_non_monotonic_normal_sequence_no_events() {
        // Build a perfectly sequential 25-frame sequence at 25fps.
        let tcs: Vec<Timecode> = (0u8..25)
            .map(|f| Timecode::new(0, 0, 0, f, FrameRate::Fps25).expect("valid"))
            .collect();
        let events = NonMonotonicDetector::new(0).scan_sequence(&tcs);
        assert!(
            events.is_empty(),
            "sequential sequence should produce no events, got: {:?}",
            events
        );
    }

    #[test]
    fn test_non_monotonic_2_second_jump_detected() {
        // Jump from 00:00:00:00 to 00:00:02:00 = +50 frames at 25fps.
        let tc0 = Timecode::new(0, 0, 0, 0, FrameRate::Fps25).expect("valid");
        let tc1 = Timecode::new(0, 0, 2, 0, FrameRate::Fps25).expect("valid");
        let events = NonMonotonicDetector::new(1).scan_sequence(&[tc0, tc1]);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].frame_index, 1);
        assert_eq!(events[0].jump_frames, 50);
    }

    #[test]
    fn test_non_monotonic_backwards_detected() {
        let tc0 = Timecode::new(0, 0, 1, 0, FrameRate::Fps25).expect("valid");
        let tc1 = Timecode::new(0, 0, 0, 0, FrameRate::Fps25).expect("valid"); // backwards
        let events = NonMonotonicDetector::new(0).scan_sequence(&[tc0, tc1]);
        assert_eq!(events.len(), 1);
        assert!(events[0].jump_frames < 0);
    }

    #[test]
    fn test_non_monotonic_threshold_filters_small_jumps() {
        // Jump of exactly 2 frames: with threshold=1, |2-1|=1 which is NOT > 1, so no event.
        let tc0 = Timecode::new(0, 0, 0, 0, FrameRate::Fps25).expect("valid");
        let tc1 = Timecode::new(0, 0, 0, 2, FrameRate::Fps25).expect("valid"); // +2 frame jump
                                                                               // threshold=1: deviation = |2-1| = 1, NOT > 1 → no event
        let events = NonMonotonicDetector::new(1).scan_sequence(&[tc0, tc1]);
        assert!(
            events.is_empty(),
            "jump of 2 should be filtered by threshold=1"
        );

        // threshold=0: deviation = 1 > 0 → event emitted
        let tc0b = Timecode::new(0, 0, 0, 0, FrameRate::Fps25).expect("valid");
        let tc1b = Timecode::new(0, 0, 0, 2, FrameRate::Fps25).expect("valid");
        let events2 = NonMonotonicDetector::new(0).scan_sequence(&[tc0b, tc1b]);
        assert_eq!(events2.len(), 1);
    }

    #[test]
    fn test_non_monotonic_multiple_events() {
        // A sequence: normal, then jump, then normal again, then backwards.
        let tcs = vec![
            Timecode::new(0, 0, 0, 0, FrameRate::Fps25).expect("valid"),
            Timecode::new(0, 0, 0, 1, FrameRate::Fps25).expect("valid"), // +1 ok
            Timecode::new(0, 0, 1, 0, FrameRate::Fps25).expect("valid"), // +24 jump
            Timecode::new(0, 0, 1, 1, FrameRate::Fps25).expect("valid"), // +1 ok
            Timecode::new(0, 0, 0, 5, FrameRate::Fps25).expect("valid"), // backwards
        ];
        let events = NonMonotonicDetector::new(1).scan_sequence(&tcs);
        // Should detect index 2 (jump) and index 4 (backwards)
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].frame_index, 2);
        assert_eq!(events[1].frame_index, 4);
    }
}
