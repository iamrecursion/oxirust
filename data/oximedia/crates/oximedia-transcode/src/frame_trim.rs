//! Frame-accurate trim and cut support for the transcoding pipeline.
//!
//! This module provides utilities for specifying frame-accurate trim points,
//! validating trim ranges, computing output durations, and assembling multi-cut
//! edit lists.  The actual frame-dropping during encode is handled by the
//! pipeline; these types describe *what* to cut.

#![allow(dead_code)]

use crate::{Result, TranscodeError};

// ─── Time representation ──────────────────────────────────────────────────────

/// A precise media timecode expressed in both frame number and milliseconds.
///
/// Storing both avoids lossy round-trips between the two representations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameTimecode {
    /// Zero-based frame index.
    pub frame: u64,
    /// Timestamp in milliseconds (derived from `frame` and `fps`).
    pub timestamp_ms: u64,
}

impl FrameTimecode {
    /// Creates a `FrameTimecode` from a frame number and a frame rate (num/den).
    ///
    /// # Panics-safe
    ///
    /// Returns `None` if `fps_den` is zero.
    #[must_use]
    pub fn from_frame(frame: u64, fps_num: u32, fps_den: u32) -> Option<Self> {
        if fps_den == 0 || fps_num == 0 {
            return None;
        }
        let timestamp_ms = frame * 1_000 * u64::from(fps_den) / u64::from(fps_num);
        Some(Self {
            frame,
            timestamp_ms,
        })
    }

    /// Creates a `FrameTimecode` from a millisecond timestamp, snapped to the
    /// nearest frame boundary.
    ///
    /// Returns `None` if `fps_den` is zero.
    #[must_use]
    pub fn from_ms(timestamp_ms: u64, fps_num: u32, fps_den: u32) -> Option<Self> {
        if fps_den == 0 || fps_num == 0 {
            return None;
        }
        let frame = timestamp_ms * u64::from(fps_num) / (1_000 * u64::from(fps_den));
        // Re-snap the ms to the actual frame boundary.
        let snapped_ms = frame * 1_000 * u64::from(fps_den) / u64::from(fps_num);
        Some(Self {
            frame,
            timestamp_ms: snapped_ms,
        })
    }

    /// Returns the duration in milliseconds between `self` and `other`.
    ///
    /// Returns `None` if `other` is before `self`.
    #[must_use]
    pub fn duration_to(&self, other: &Self) -> Option<u64> {
        other.timestamp_ms.checked_sub(self.timestamp_ms)
    }

    /// Returns the number of frames between `self` and `other`.
    ///
    /// Returns `None` if `other` is before `self`.
    #[must_use]
    pub fn frames_to(&self, other: &Self) -> Option<u64> {
        other.frame.checked_sub(self.frame)
    }
}

// ─── TrimPoint ────────────────────────────────────────────────────────────────

/// A single inclusive trim point (in-point or out-point).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrimPoint {
    /// Trim at a frame number.
    Frame(u64),
    /// Trim at a millisecond timestamp (snapped to nearest frame during validation).
    Milliseconds(u64),
}

impl TrimPoint {
    /// Resolves the trim point to a `FrameTimecode` given the source frame rate.
    ///
    /// # Errors
    ///
    /// Returns an error if the frame rate is zero.
    pub fn resolve(&self, fps_num: u32, fps_den: u32) -> Result<FrameTimecode> {
        match self {
            Self::Frame(f) => FrameTimecode::from_frame(*f, fps_num, fps_den).ok_or_else(|| {
                TranscodeError::ValidationError(crate::ValidationError::Unsupported(
                    "Invalid frame rate: fps_num or fps_den is zero".into(),
                ))
            }),
            Self::Milliseconds(ms) => {
                FrameTimecode::from_ms(*ms, fps_num, fps_den).ok_or_else(|| {
                    TranscodeError::ValidationError(crate::ValidationError::Unsupported(
                        "Invalid frame rate: fps_num or fps_den is zero".into(),
                    ))
                })
            }
        }
    }
}

// ─── TrimRange ────────────────────────────────────────────────────────────────

/// A single contiguous inclusive trim range `[in_point, out_point]`.
///
/// The `in_point` is the first frame to *keep*; `out_point` is the last frame
/// to *keep* (both inclusive).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrimRange {
    /// First frame (inclusive) to keep.
    pub in_point: TrimPoint,
    /// Last frame (inclusive) to keep.
    pub out_point: TrimPoint,
}

impl TrimRange {
    /// Creates a frame-accurate trim range.
    #[must_use]
    pub fn frames(in_frame: u64, out_frame: u64) -> Self {
        Self {
            in_point: TrimPoint::Frame(in_frame),
            out_point: TrimPoint::Frame(out_frame),
        }
    }

    /// Creates a millisecond-based trim range.
    #[must_use]
    pub fn milliseconds(in_ms: u64, out_ms: u64) -> Self {
        Self {
            in_point: TrimPoint::Milliseconds(in_ms),
            out_point: TrimPoint::Milliseconds(out_ms),
        }
    }

    /// Validates that `in_point < out_point` and optionally that both points
    /// lie within `total_frames`.
    ///
    /// # Errors
    ///
    /// Returns an error if the range is invalid.
    pub fn validate(&self, fps_num: u32, fps_den: u32, total_frames: Option<u64>) -> Result<()> {
        let in_tc = self.in_point.resolve(fps_num, fps_den)?;
        let out_tc = self.out_point.resolve(fps_num, fps_den)?;

        if in_tc.frame >= out_tc.frame {
            return Err(TranscodeError::ValidationError(
                crate::ValidationError::Unsupported(format!(
                    "Trim in-point frame {} must be less than out-point frame {}",
                    in_tc.frame, out_tc.frame
                )),
            ));
        }

        if let Some(total) = total_frames {
            if out_tc.frame >= total {
                return Err(TranscodeError::ValidationError(
                    crate::ValidationError::Unsupported(format!(
                        "Trim out-point frame {} exceeds total frames {}",
                        out_tc.frame, total
                    )),
                ));
            }
        }

        Ok(())
    }

    /// Resolves the trim range to resolved `FrameTimecode` values.
    ///
    /// # Errors
    ///
    /// Returns an error if the frame rate is invalid.
    pub fn resolve(&self, fps_num: u32, fps_den: u32) -> Result<ResolvedTrimRange> {
        let in_tc = self.in_point.resolve(fps_num, fps_den)?;
        let out_tc = self.out_point.resolve(fps_num, fps_den)?;
        Ok(ResolvedTrimRange {
            in_point: in_tc,
            out_point: out_tc,
        })
    }
}

/// A `TrimRange` whose points have been resolved to concrete `FrameTimecode` values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedTrimRange {
    /// Resolved in-point.
    pub in_point: FrameTimecode,
    /// Resolved out-point.
    pub out_point: FrameTimecode,
}

impl ResolvedTrimRange {
    /// Returns the duration in frames (inclusive).
    #[must_use]
    pub fn frame_count(&self) -> u64 {
        self.out_point.frame.saturating_sub(self.in_point.frame) + 1
    }

    /// Returns the duration in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.out_point
            .timestamp_ms
            .saturating_sub(self.in_point.timestamp_ms)
    }

    /// Returns `true` if `frame` falls within this trim range (inclusive).
    #[must_use]
    pub fn contains_frame(&self, frame: u64) -> bool {
        frame >= self.in_point.frame && frame <= self.out_point.frame
    }
}

// ─── FrameTrimConfig ──────────────────────────────────────────────────────────

/// Complete frame-accurate trim / multi-cut configuration for a transcode job.
///
/// Multiple `TrimRange` entries create a *cut list* — the pipeline concatenates
/// only the segments that correspond to each range.
#[derive(Debug, Clone)]
pub struct FrameTrimConfig {
    /// Source frame rate numerator.
    pub fps_num: u32,
    /// Source frame rate denominator.
    pub fps_den: u32,
    /// Total frame count of the source (used for bounds checking).
    pub total_source_frames: Option<u64>,
    /// Ordered list of ranges to include in the output.
    pub ranges: Vec<TrimRange>,
}

impl FrameTrimConfig {
    /// Creates a new trim config for the given frame rate.
    #[must_use]
    pub fn new(fps_num: u32, fps_den: u32) -> Self {
        Self {
            fps_num,
            fps_den,
            total_source_frames: None,
            ranges: Vec::new(),
        }
    }

    /// Sets the total frame count for bounds checking.
    #[must_use]
    pub fn total_frames(mut self, n: u64) -> Self {
        self.total_source_frames = Some(n);
        self
    }

    /// Adds a trim range.
    #[must_use]
    pub fn add_range(mut self, range: TrimRange) -> Self {
        self.ranges.push(range);
        self
    }

    /// Validates all ranges and returns resolved ranges sorted by in-point.
    ///
    /// # Errors
    ///
    /// Returns an error if any range is invalid or if ranges overlap.
    pub fn validate_and_resolve(&self) -> Result<Vec<ResolvedTrimRange>> {
        if self.ranges.is_empty() {
            return Err(TranscodeError::ValidationError(
                crate::ValidationError::Unsupported("Trim config has no ranges".into()),
            ));
        }

        let mut resolved: Vec<ResolvedTrimRange> = self
            .ranges
            .iter()
            .map(|r| {
                r.validate(self.fps_num, self.fps_den, self.total_source_frames)?;
                r.resolve(self.fps_num, self.fps_den)
            })
            .collect::<Result<Vec<_>>>()?;

        // Sort by in-point frame
        resolved.sort_by_key(|r| r.in_point.frame);

        // Check for overlapping ranges
        for pair in resolved.windows(2) {
            let a = &pair[0];
            let b = &pair[1];
            if b.in_point.frame <= a.out_point.frame {
                return Err(TranscodeError::ValidationError(
                    crate::ValidationError::Unsupported(format!(
                        "Trim ranges overlap: [{}, {}] and [{}, {}]",
                        a.in_point.frame, a.out_point.frame, b.in_point.frame, b.out_point.frame
                    )),
                ));
            }
        }

        Ok(resolved)
    }

    /// Returns the total output frame count across all ranges.
    ///
    /// # Errors
    ///
    /// Returns an error if validation fails.
    pub fn total_output_frames(&self) -> Result<u64> {
        let resolved = self.validate_and_resolve()?;
        Ok(resolved.iter().map(|r| r.frame_count()).sum())
    }

    /// Returns the total output duration in milliseconds.
    ///
    /// # Errors
    ///
    /// Returns an error if validation fails.
    pub fn total_output_duration_ms(&self) -> Result<u64> {
        let resolved = self.validate_and_resolve()?;
        Ok(resolved.iter().map(|r| r.duration_ms()).sum())
    }

    /// Returns `true` if the given source frame should be included in the output.
    ///
    /// Pre-resolves ranges lazily; panics-safe (returns `false` on error).
    #[must_use]
    pub fn should_include_frame(&self, frame: u64) -> bool {
        match self.validate_and_resolve() {
            Ok(resolved) => resolved.iter().any(|r| r.contains_frame(frame)),
            Err(_) => false,
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_timecode_from_frame_30fps() {
        let tc = FrameTimecode::from_frame(30, 30, 1).expect("valid");
        assert_eq!(tc.frame, 30);
        assert_eq!(tc.timestamp_ms, 1000);
    }

    #[test]
    fn test_frame_timecode_from_ms_30fps() {
        let tc = FrameTimecode::from_ms(1000, 30, 1).expect("valid");
        assert_eq!(tc.frame, 30);
        assert_eq!(tc.timestamp_ms, 1000);
    }

    #[test]
    fn test_frame_timecode_zero_fps_returns_none() {
        assert!(FrameTimecode::from_frame(5, 0, 1).is_none());
        assert!(FrameTimecode::from_frame(5, 30, 0).is_none());
    }

    #[test]
    fn test_frame_timecode_duration_to() {
        let a = FrameTimecode::from_frame(0, 30, 1).expect("valid");
        let b = FrameTimecode::from_frame(30, 30, 1).expect("valid");
        assert_eq!(a.duration_to(&b), Some(1000));
    }

    #[test]
    fn test_frame_timecode_duration_to_reverse_is_none() {
        let a = FrameTimecode::from_frame(30, 30, 1).expect("valid");
        let b = FrameTimecode::from_frame(0, 30, 1).expect("valid");
        assert_eq!(a.duration_to(&b), None);
    }

    #[test]
    fn test_trim_range_validate_ok() {
        let range = TrimRange::frames(0, 59);
        assert!(range.validate(30, 1, Some(120)).is_ok());
    }

    #[test]
    fn test_trim_range_validate_in_ge_out_fails() {
        let range = TrimRange::frames(60, 30);
        assert!(range.validate(30, 1, None).is_err());
    }

    #[test]
    fn test_trim_range_validate_out_exceeds_total_fails() {
        let range = TrimRange::frames(0, 200);
        assert!(range.validate(30, 1, Some(100)).is_err());
    }

    #[test]
    fn test_trim_range_resolve_ms() {
        let range = TrimRange::milliseconds(0, 1000);
        let resolved = range.resolve(30, 1).expect("valid");
        assert_eq!(resolved.in_point.frame, 0);
        assert_eq!(resolved.out_point.frame, 30);
    }

    #[test]
    fn test_resolved_range_frame_count() {
        let range = TrimRange::frames(10, 19);
        let r = range.resolve(30, 1).expect("valid");
        assert_eq!(r.frame_count(), 10);
    }

    #[test]
    fn test_resolved_range_contains_frame() {
        let range = TrimRange::frames(10, 19);
        let r = range.resolve(30, 1).expect("valid");
        assert!(r.contains_frame(10));
        assert!(r.contains_frame(15));
        assert!(r.contains_frame(19));
        assert!(!r.contains_frame(9));
        assert!(!r.contains_frame(20));
    }

    #[test]
    fn test_trim_config_total_output_frames() {
        let cfg = FrameTrimConfig::new(30, 1)
            .total_frames(300)
            .add_range(TrimRange::frames(0, 29)) // 30 frames
            .add_range(TrimRange::frames(60, 89)); // 30 frames
        assert_eq!(cfg.total_output_frames().expect("valid"), 60);
    }

    #[test]
    fn test_trim_config_overlapping_ranges_fails() {
        let cfg = FrameTrimConfig::new(30, 1)
            .add_range(TrimRange::frames(0, 59))
            .add_range(TrimRange::frames(30, 89)); // overlaps
        assert!(cfg.validate_and_resolve().is_err());
    }

    #[test]
    fn test_trim_config_no_ranges_fails() {
        let cfg = FrameTrimConfig::new(30, 1);
        assert!(cfg.validate_and_resolve().is_err());
    }

    #[test]
    fn test_should_include_frame() {
        let cfg = FrameTrimConfig::new(30, 1)
            .add_range(TrimRange::frames(10, 19))
            .add_range(TrimRange::frames(30, 39));

        assert!(cfg.should_include_frame(10));
        assert!(cfg.should_include_frame(15));
        assert!(cfg.should_include_frame(19));
        assert!(cfg.should_include_frame(30));
        assert!(!cfg.should_include_frame(9));
        assert!(!cfg.should_include_frame(20));
        assert!(!cfg.should_include_frame(29));
        assert!(!cfg.should_include_frame(40));
    }

    #[test]
    fn test_total_output_duration_ms() {
        // Two 1-second clips at 30 fps
        let cfg = FrameTrimConfig::new(30, 1)
            .add_range(TrimRange::milliseconds(0, 1000))
            .add_range(TrimRange::milliseconds(2000, 3000));
        let total_ms = cfg.total_output_duration_ms().expect("valid");
        assert_eq!(total_ms, 2000);
    }
}
