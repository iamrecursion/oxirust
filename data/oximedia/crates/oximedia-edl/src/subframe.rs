//! Sub-frame timecode precision for high frame rate content (120fps+).
//!
//! Standard EDL timecodes are limited to integer frame counts. For high
//! frame rate (HFR) content at 120fps, 240fps, or higher, sub-frame
//! precision may be needed to represent exact positions within a frame.
//!
//! This module provides a `SubFrameTimecode` that extends the standard
//! EDL timecode with fractional frame information.

#![allow(dead_code)]

use crate::error::{EdlError, EdlResult};
use crate::timecode::{EdlFrameRate, EdlTimecode};

/// High frame rate enumeration for sub-frame timecode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum HighFrameRate {
    /// 96 fps.
    Fps96,
    /// 100 fps.
    Fps100,
    /// 119.88 fps (NTSC 120).
    Fps11988,
    /// 120 fps.
    Fps120,
    /// 240 fps.
    Fps240,
    /// Custom HFR with nominal integer rate.
    Custom(u32),
}

impl HighFrameRate {
    /// Get the nominal frames per second.
    #[must_use]
    pub const fn nominal_fps(&self) -> u32 {
        match self {
            Self::Fps96 => 96,
            Self::Fps100 => 100,
            Self::Fps11988 => 120,
            Self::Fps120 => 120,
            Self::Fps240 => 240,
            Self::Custom(fps) => *fps,
        }
    }

    /// Get the exact fps as a float.
    #[must_use]
    pub fn exact_fps(&self) -> f64 {
        match self {
            Self::Fps96 => 96.0,
            Self::Fps100 => 100.0,
            Self::Fps11988 => 120_000.0 / 1001.0,
            Self::Fps120 => 120.0,
            Self::Fps240 => 240.0,
            Self::Custom(fps) => *fps as f64,
        }
    }

    /// Get the ratio of this HFR to a base EDL frame rate.
    /// For example, 120fps / 24fps = 5 sub-frames per base frame.
    #[must_use]
    pub fn sub_frame_ratio(&self, base: EdlFrameRate) -> u32 {
        let base_fps = base.fps();
        if base_fps == 0 {
            return 1;
        }
        let ratio = self.nominal_fps() / base_fps;
        ratio.max(1)
    }

    /// Check if this is an NTSC-style rate (needs drop-frame consideration).
    #[must_use]
    pub const fn is_ntsc(&self) -> bool {
        matches!(self, Self::Fps11988)
    }
}

impl std::fmt::Display for HighFrameRate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fps96 => write!(f, "96"),
            Self::Fps100 => write!(f, "100"),
            Self::Fps11988 => write!(f, "119.88"),
            Self::Fps120 => write!(f, "120"),
            Self::Fps240 => write!(f, "240"),
            Self::Custom(fps) => write!(f, "{fps}"),
        }
    }
}

/// Sub-frame timecode extending standard timecode with fractional precision.
///
/// The `sub_frame` field represents a position within a single base frame,
/// ranging from 0 to `sub_frame_divisor - 1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubFrameTimecode {
    /// Base timecode (at the EDL frame rate).
    pub base: EdlTimecode,
    /// Sub-frame position within the base frame (0-indexed).
    pub sub_frame: u32,
    /// Number of sub-frame divisions per base frame.
    pub sub_frame_divisor: u32,
}

impl SubFrameTimecode {
    /// Create a new sub-frame timecode.
    ///
    /// # Errors
    ///
    /// Returns an error if `sub_frame >= sub_frame_divisor`.
    pub fn new(base: EdlTimecode, sub_frame: u32, sub_frame_divisor: u32) -> EdlResult<Self> {
        if sub_frame_divisor == 0 {
            return Err(EdlError::validation("sub_frame_divisor must be > 0"));
        }
        if sub_frame >= sub_frame_divisor {
            return Err(EdlError::validation(format!(
                "sub_frame ({sub_frame}) must be < sub_frame_divisor ({sub_frame_divisor})"
            )));
        }
        Ok(Self {
            base,
            sub_frame,
            sub_frame_divisor,
        })
    }

    /// Create from a base timecode with no sub-frame offset.
    #[must_use]
    pub fn from_base(base: EdlTimecode) -> Self {
        Self {
            base,
            sub_frame: 0,
            sub_frame_divisor: 1,
        }
    }

    /// Create from a high frame rate and total HFR frame count.
    ///
    /// # Errors
    ///
    /// Returns an error if timecodes cannot be computed.
    pub fn from_hfr_frames(
        hfr_frames: u64,
        hfr: HighFrameRate,
        base_rate: EdlFrameRate,
    ) -> EdlResult<Self> {
        let ratio = hfr.sub_frame_ratio(base_rate);
        let base_frames = hfr_frames / ratio as u64;
        let sub = (hfr_frames % ratio as u64) as u32;

        let base = EdlTimecode::from_frames(base_frames, base_rate)?;
        Ok(Self {
            base,
            sub_frame: sub,
            sub_frame_divisor: ratio,
        })
    }

    /// Convert to total HFR frames.
    #[must_use]
    pub fn to_hfr_frames(&self) -> u64 {
        self.base.to_frames() * self.sub_frame_divisor as u64 + self.sub_frame as u64
    }

    /// Get the sub-frame fraction (0.0 to <1.0).
    #[must_use]
    pub fn sub_frame_fraction(&self) -> f64 {
        if self.sub_frame_divisor == 0 {
            return 0.0;
        }
        self.sub_frame as f64 / self.sub_frame_divisor as f64
    }

    /// Convert to seconds with sub-frame precision.
    #[must_use]
    pub fn to_seconds(&self, base_rate: EdlFrameRate) -> f64 {
        let base_seconds = self.base.to_frames() as f64 / base_rate.as_float();
        let sub_seconds = self.sub_frame_fraction() / base_rate.as_float();
        base_seconds + sub_seconds
    }

    /// Check if this timecode has a sub-frame offset.
    #[must_use]
    pub const fn has_sub_frame(&self) -> bool {
        self.sub_frame > 0
    }
}

impl PartialOrd for SubFrameTimecode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SubFrameTimecode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.to_hfr_frames().cmp(&other.to_hfr_frames())
    }
}

impl std::fmt::Display for SubFrameTimecode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.sub_frame > 0 {
            write!(
                f,
                "{}.{}/{}",
                self.base, self.sub_frame, self.sub_frame_divisor
            )
        } else {
            write!(f, "{}", self.base)
        }
    }
}

/// Sub-frame event range for HFR editing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubFrameRange {
    /// Start timecode.
    pub start: SubFrameTimecode,
    /// End timecode.
    pub end: SubFrameTimecode,
}

impl SubFrameRange {
    /// Create a new sub-frame range.
    ///
    /// # Errors
    ///
    /// Returns an error if start >= end.
    pub fn new(start: SubFrameTimecode, end: SubFrameTimecode) -> EdlResult<Self> {
        if start >= end {
            return Err(EdlError::validation("SubFrame range: start must be < end"));
        }
        Ok(Self { start, end })
    }

    /// Duration in HFR frames.
    #[must_use]
    pub fn duration_hfr_frames(&self) -> u64 {
        self.end
            .to_hfr_frames()
            .saturating_sub(self.start.to_hfr_frames())
    }

    /// Duration in seconds.
    #[must_use]
    pub fn duration_seconds(&self, base_rate: EdlFrameRate) -> f64 {
        self.end.to_seconds(base_rate) - self.start.to_seconds(base_rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tc(h: u8, m: u8, s: u8, f: u8) -> EdlTimecode {
        EdlTimecode::new(h, m, s, f, EdlFrameRate::Fps24).expect("failed to create")
    }

    #[test]
    fn test_high_frame_rate_nominal() {
        assert_eq!(HighFrameRate::Fps120.nominal_fps(), 120);
        assert_eq!(HighFrameRate::Fps240.nominal_fps(), 240);
        assert_eq!(HighFrameRate::Fps96.nominal_fps(), 96);
        assert_eq!(HighFrameRate::Custom(144).nominal_fps(), 144);
    }

    #[test]
    fn test_high_frame_rate_exact() {
        assert!((HighFrameRate::Fps120.exact_fps() - 120.0).abs() < f64::EPSILON);
        assert!((HighFrameRate::Fps11988.exact_fps() - 119.88011988).abs() < 0.001);
    }

    #[test]
    fn test_sub_frame_ratio() {
        assert_eq!(
            HighFrameRate::Fps120.sub_frame_ratio(EdlFrameRate::Fps24),
            5
        );
        assert_eq!(
            HighFrameRate::Fps240.sub_frame_ratio(EdlFrameRate::Fps24),
            10
        );
        assert_eq!(
            HighFrameRate::Fps120.sub_frame_ratio(EdlFrameRate::Fps60),
            2
        );
    }

    #[test]
    fn test_subframe_timecode_new() {
        let base = make_tc(1, 0, 0, 0);
        let tc = SubFrameTimecode::new(base, 3, 5).expect("should succeed");
        assert_eq!(tc.sub_frame, 3);
        assert_eq!(tc.sub_frame_divisor, 5);
        assert!(tc.has_sub_frame());
    }

    #[test]
    fn test_subframe_timecode_no_subframe() {
        let base = make_tc(1, 0, 0, 0);
        let tc = SubFrameTimecode::from_base(base);
        assert!(!tc.has_sub_frame());
        assert_eq!(tc.sub_frame, 0);
    }

    #[test]
    fn test_subframe_invalid_sub_frame() {
        let base = make_tc(1, 0, 0, 0);
        assert!(SubFrameTimecode::new(base, 5, 5).is_err());
        assert!(SubFrameTimecode::new(base, 10, 5).is_err());
    }

    #[test]
    fn test_subframe_zero_divisor() {
        let base = make_tc(1, 0, 0, 0);
        assert!(SubFrameTimecode::new(base, 0, 0).is_err());
    }

    #[test]
    fn test_subframe_fraction() {
        let base = make_tc(1, 0, 0, 0);
        let tc = SubFrameTimecode::new(base, 2, 5).expect("should succeed");
        assert!((tc.sub_frame_fraction() - 0.4).abs() < f64::EPSILON);
    }

    #[test]
    fn test_hfr_frame_roundtrip() {
        let base = make_tc(0, 0, 1, 0); // 1 second = 24 frames at 24fps
        let tc = SubFrameTimecode::new(base, 3, 5).expect("should succeed");
        let hfr_frames = tc.to_hfr_frames();
        // 24 base frames * 5 + 3 = 123
        assert_eq!(hfr_frames, 123);
    }

    #[test]
    fn test_from_hfr_frames() {
        let tc = SubFrameTimecode::from_hfr_frames(123, HighFrameRate::Fps120, EdlFrameRate::Fps24)
            .expect("should succeed");
        // 123 HFR frames / 5 ratio = 24 base frames, 3 sub-frames
        assert_eq!(tc.base.to_frames(), 24);
        assert_eq!(tc.sub_frame, 3);
        assert_eq!(tc.sub_frame_divisor, 5);
    }

    #[test]
    fn test_subframe_comparison() {
        let base1 = make_tc(0, 0, 0, 10);
        let base2 = make_tc(0, 0, 0, 10);

        let tc1 = SubFrameTimecode::new(base1, 1, 5).expect("should succeed");
        let tc2 = SubFrameTimecode::new(base2, 3, 5).expect("should succeed");
        assert!(tc1 < tc2);
    }

    #[test]
    fn test_subframe_display_with_sub() {
        let base = make_tc(1, 0, 0, 5);
        let tc = SubFrameTimecode::new(base, 2, 5).expect("should succeed");
        let s = tc.to_string();
        assert!(s.contains("2/5"));
    }

    #[test]
    fn test_subframe_display_without_sub() {
        let base = make_tc(1, 0, 0, 5);
        let tc = SubFrameTimecode::from_base(base);
        let s = tc.to_string();
        assert!(!s.contains('/'));
    }

    #[test]
    fn test_subframe_range() {
        let base1 = make_tc(0, 0, 0, 0);
        let base2 = make_tc(0, 0, 1, 0);
        let tc1 = SubFrameTimecode::new(base1, 0, 5).expect("should succeed");
        let tc2 = SubFrameTimecode::new(base2, 0, 5).expect("should succeed");
        let range = SubFrameRange::new(tc1, tc2).expect("should succeed");
        assert_eq!(range.duration_hfr_frames(), 120); // 24 base frames * 5
    }

    #[test]
    fn test_subframe_range_invalid() {
        let base = make_tc(0, 0, 1, 0);
        let tc = SubFrameTimecode::from_base(base);
        assert!(SubFrameRange::new(tc, tc).is_err());
    }

    #[test]
    fn test_high_frame_rate_is_ntsc() {
        assert!(HighFrameRate::Fps11988.is_ntsc());
        assert!(!HighFrameRate::Fps120.is_ntsc());
    }

    #[test]
    fn test_high_frame_rate_display() {
        assert_eq!(HighFrameRate::Fps120.to_string(), "120");
        assert_eq!(HighFrameRate::Fps11988.to_string(), "119.88");
        assert_eq!(HighFrameRate::Custom(144).to_string(), "144");
    }

    #[test]
    fn test_to_seconds() {
        let base = make_tc(0, 0, 1, 0); // 1 second at 24fps
        let tc = SubFrameTimecode::from_base(base);
        let secs = tc.to_seconds(EdlFrameRate::Fps24);
        assert!((secs - 1.0).abs() < 0.01);
    }
}
