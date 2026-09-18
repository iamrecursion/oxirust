//! Frame-accurate trim engine with SMPTE timecode support.
//!
//! Provides frame-level trimming of media segments and conversion between
//! frame counts and SMPTE timecode strings (drop-frame and non-drop-frame).
//!
//! # Drop-Frame Timecode
//!
//! For 29.97 fps (30000/1001) and 59.94 fps (60000/1001), the SMPTE drop-frame
//! convention skips frame numbers (not actual frames) to stay aligned with wall
//! clock time.
//!
//! - 29.97 fps: drop 2 frame numbers at the start of each minute, except every
//!   10th minute. Uses `;` as the separator (e.g. `01:00:00;00`).
//! - 59.94 fps: drop 4 frame numbers under the same rule.
//! - All other rates: non-drop-frame, uses `:` as separator.
//!
//! # Example
//!
//! ```rust
//! use oximedia_playout::frame_trim::{FrameTrimConfig, FrameTrimmer, MediaSegment};
//!
//! let config = FrameTrimConfig::new(0, 25, 25, 1).expect("valid config");
//! let mut seg = MediaSegment::new(25, 1);
//! for i in 0..50u64 {
//!     seg.add_frame(vec![i as u8; 100], i % 5 == 0);
//! }
//! let trimmed = FrameTrimmer::trim(&seg, &config).expect("trim ok");
//! assert_eq!(trimmed.total_frames, 25);
//! ```

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

// ── Error ─────────────────────────────────────────────────────────────────────

/// Error type for frame trimming operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrameTrimError {
    /// The config parameters are structurally invalid.
    InvalidConfig(String),
    /// `in_point_frames` is beyond the end of the segment.
    InPointOutOfRange { in_point: u64, total_frames: u64 },
    /// `out_point_frames` is beyond the end of the segment.
    OutPointOutOfRange { out_point: u64, total_frames: u64 },
    /// `in_point_frames` >= `out_point_frames`.
    InPointAfterOutPoint { in_point: u64, out_point: u64 },
    /// The trim produces zero frames.
    ZeroDuration,
    /// Frame rate is zero or otherwise invalid.
    InvalidFrameRate { fps_num: u32, fps_den: u32 },
    /// The source segment contains no frames.
    EmptySegment,
    /// Timecode string could not be parsed.
    InvalidTimecode(String),
}

impl std::fmt::Display for FrameTrimError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid trim config: {}", msg),
            Self::InPointOutOfRange {
                in_point,
                total_frames,
            } => write!(
                f,
                "in-point {} is out of range for segment with {} frames",
                in_point, total_frames
            ),
            Self::OutPointOutOfRange {
                out_point,
                total_frames,
            } => write!(
                f,
                "out-point {} is out of range for segment with {} frames",
                out_point, total_frames
            ),
            Self::InPointAfterOutPoint {
                in_point,
                out_point,
            } => write!(
                f,
                "in-point {} >= out-point {} — invalid trim range",
                in_point, out_point
            ),
            Self::ZeroDuration => write!(f, "trim results in zero-duration segment"),
            Self::InvalidFrameRate { fps_num, fps_den } => {
                write!(f, "invalid frame rate {}/{}", fps_num, fps_den)
            }
            Self::EmptySegment => write!(f, "source segment contains no frames"),
            Self::InvalidTimecode(s) => write!(f, "invalid timecode: {}", s),
        }
    }
}

impl std::error::Error for FrameTrimError {}

// ── FrameTrimConfig ───────────────────────────────────────────────────────────

/// Configuration for a frame-accurate trim operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameTrimConfig {
    /// In-point (inclusive, 0-based frame index).
    pub in_point_frames: u64,
    /// Out-point (exclusive: the trim includes frames [in, out)).
    pub out_point_frames: u64,
    /// Frame rate numerator (e.g. 30000 for 29.97).
    pub fps_num: u32,
    /// Frame rate denominator (e.g. 1001 for 29.97).
    pub fps_den: u32,
}

impl FrameTrimConfig {
    /// Creates and validates a new config.
    ///
    /// # Errors
    ///
    /// Returns an error if the frame rate is zero or the in/out points are
    /// logically inconsistent.
    pub fn new(
        in_point_frames: u64,
        out_point_frames: u64,
        fps_num: u32,
        fps_den: u32,
    ) -> Result<Self, FrameTrimError> {
        if fps_num == 0 || fps_den == 0 {
            return Err(FrameTrimError::InvalidFrameRate { fps_num, fps_den });
        }
        if in_point_frames >= out_point_frames {
            return Err(FrameTrimError::InPointAfterOutPoint {
                in_point: in_point_frames,
                out_point: out_point_frames,
            });
        }
        Ok(Self {
            in_point_frames,
            out_point_frames,
            fps_num,
            fps_den,
        })
    }

    /// Number of frames in the trimmed range.
    pub fn duration_frames(&self) -> u64 {
        self.out_point_frames.saturating_sub(self.in_point_frames)
    }

    /// PTS (in microseconds) of the in-point.
    pub fn in_point_pts_us(&self) -> u64 {
        frame_to_pts_us(self.in_point_frames, self.fps_num, self.fps_den)
    }

    /// PTS (in microseconds) of the out-point.
    pub fn out_point_pts_us(&self) -> u64 {
        frame_to_pts_us(self.out_point_frames, self.fps_num, self.fps_den)
    }

    /// Frame rate as a floating-point value.
    pub fn fps_as_f64(&self) -> f64 {
        f64::from(self.fps_num) / f64::from(self.fps_den)
    }
}

// ── FrameData ─────────────────────────────────────────────────────────────────

/// A single frame's data and metadata.
#[derive(Debug, Clone)]
pub struct FrameData {
    /// 0-based frame index within its parent segment.
    pub index: u64,
    /// Presentation timestamp in microseconds.
    pub pts_us: u64,
    /// Raw frame payload bytes.
    pub data: Vec<u8>,
    /// Whether this frame is an intra (keyframe / I-frame).
    pub keyframe: bool,
    /// Duration of this frame in microseconds.
    pub duration_us: u64,
}

// ── MediaSegment ─────────────────────────────────────────────────────────────

/// A sequence of frames forming a contiguous media segment.
#[derive(Debug, Clone)]
pub struct MediaSegment {
    /// All frames in presentation order.
    pub frames: Vec<FrameData>,
    /// PTS of the first frame in microseconds.
    pub start_pts_us: u64,
    /// Frame rate numerator.
    pub fps_num: u32,
    /// Frame rate denominator.
    pub fps_den: u32,
    /// Total number of frames (mirrors `frames.len()`).
    pub total_frames: u64,
}

impl MediaSegment {
    /// Creates an empty media segment with the given frame rate.
    pub fn new(fps_num: u32, fps_den: u32) -> Self {
        Self {
            frames: Vec::new(),
            start_pts_us: 0,
            fps_num,
            fps_den,
            total_frames: 0,
        }
    }

    /// Creates an empty media segment starting at `start_pts_us`.
    pub fn with_start_pts(fps_num: u32, fps_den: u32, start_pts_us: u64) -> Self {
        Self {
            frames: Vec::new(),
            start_pts_us,
            fps_num,
            fps_den,
            total_frames: 0,
        }
    }

    /// Appends a frame with auto-computed PTS and duration.
    pub fn add_frame(&mut self, data: Vec<u8>, keyframe: bool) {
        let index = self.total_frames;
        let pts_us = self.start_pts_us + frame_to_pts_us(index, self.fps_num, self.fps_den);
        let duration_us = frame_to_pts_us(1, self.fps_num, self.fps_den);
        self.frames.push(FrameData {
            index,
            pts_us,
            data,
            keyframe,
            duration_us,
        });
        self.total_frames += 1;
    }

    /// Returns the number of frames.
    pub fn frame_count(&self) -> u64 {
        self.total_frames
    }

    /// Returns the total duration in microseconds.
    pub fn duration_us(&self) -> u64 {
        frame_to_pts_us(self.total_frames, self.fps_num, self.fps_den)
    }
}

// ── TrimReport ────────────────────────────────────────────────────────────────

/// Summary information about a trim operation.
#[derive(Debug, Clone)]
pub struct TrimReport {
    /// In-point (0-based, inclusive).
    pub in_point_frames: u64,
    /// Out-point (0-based, exclusive).
    pub out_point_frames: u64,
    /// Number of frames in the trimmed segment.
    pub duration_frames: u64,
    /// SMPTE timecode of the in-point.
    pub in_point_timecode: String,
    /// SMPTE timecode of the out-point.
    pub out_point_timecode: String,
    /// PTS of the in-point in microseconds.
    pub in_point_pts_us: u64,
    /// PTS of the out-point in microseconds.
    pub out_point_pts_us: u64,
    /// Nearest keyframe at or before the in-point.
    pub nearest_keyframe_in: Option<u64>,
    /// Nearest keyframe at or after the out-point (within segment).
    pub nearest_keyframe_out: Option<u64>,
    /// Total bytes in the trimmed segment frames.
    pub trimmed_size_bytes: usize,
}

// ── FrameTrimmer ──────────────────────────────────────────────────────────────

/// Frame-accurate trim engine.
pub struct FrameTrimmer;

impl FrameTrimmer {
    /// Creates a new trimmer instance.
    pub fn new() -> Self {
        Self
    }

    /// Trims `media` to the frame range `[in_point, out_point)`.
    ///
    /// Returns a new `MediaSegment` containing only the selected frames,
    /// with re-indexed frame metadata.
    ///
    /// # Errors
    ///
    /// - [`FrameTrimError::EmptySegment`] if `media` contains no frames.
    /// - [`FrameTrimError::InPointOutOfRange`] if `in_point >= total_frames`.
    /// - [`FrameTrimError::OutPointOutOfRange`] if `out_point > total_frames`.
    /// - [`FrameTrimError::InPointAfterOutPoint`] if `in_point >= out_point`.
    /// - [`FrameTrimError::ZeroDuration`] if the resulting segment is empty.
    pub fn trim(
        media: &MediaSegment,
        config: &FrameTrimConfig,
    ) -> Result<MediaSegment, FrameTrimError> {
        let total = media.total_frames;
        if total == 0 {
            return Err(FrameTrimError::EmptySegment);
        }
        if config.in_point_frames >= total {
            return Err(FrameTrimError::InPointOutOfRange {
                in_point: config.in_point_frames,
                total_frames: total,
            });
        }
        if config.out_point_frames > total {
            return Err(FrameTrimError::OutPointOutOfRange {
                out_point: config.out_point_frames,
                total_frames: total,
            });
        }
        if config.in_point_frames >= config.out_point_frames {
            return Err(FrameTrimError::InPointAfterOutPoint {
                in_point: config.in_point_frames,
                out_point: config.out_point_frames,
            });
        }

        let in_idx = config.in_point_frames as usize;
        let out_idx = config.out_point_frames as usize;
        let selected = &media.frames[in_idx..out_idx];

        if selected.is_empty() {
            return Err(FrameTrimError::ZeroDuration);
        }

        let new_start_pts = selected[0].pts_us;
        let mut trimmed = MediaSegment::with_start_pts(media.fps_num, media.fps_den, new_start_pts);

        for (new_idx, frame) in selected.iter().enumerate() {
            let new_pts =
                new_start_pts + frame_to_pts_us(new_idx as u64, media.fps_num, media.fps_den);
            trimmed.frames.push(FrameData {
                index: new_idx as u64,
                pts_us: new_pts,
                data: frame.data.clone(),
                keyframe: frame.keyframe,
                duration_us: frame.duration_us,
            });
            trimmed.total_frames += 1;
        }

        Ok(trimmed)
    }

    /// Converts a 0-based frame count to a SMPTE timecode string.
    ///
    /// - Drop-frame (29.97 fps = 30000/1001, 59.94 fps = 60000/1001): uses `;`
    /// - All other rates: uses `:`
    pub fn to_timecode(frame: u64, fps_num: u32, fps_den: u32) -> String {
        let (nominal_fps, drop_frames) = nominal_fps_and_drop(fps_num, fps_den);
        if drop_frames > 0 {
            to_drop_frame_timecode(frame, nominal_fps, drop_frames)
        } else {
            to_nondrop_timecode(frame, nominal_fps)
        }
    }

    /// Parses a SMPTE timecode string back to a 0-based frame count.
    ///
    /// Both `:` (non-drop) and `;` (drop-frame) separators are accepted.
    ///
    /// # Errors
    ///
    /// Returns [`FrameTrimError::InvalidTimecode`] if the string is malformed.
    pub fn from_timecode(tc: &str, fps_num: u32, fps_den: u32) -> Result<u64, FrameTrimError> {
        let (nominal_fps, drop_frames) = nominal_fps_and_drop(fps_num, fps_den);
        parse_timecode(tc, nominal_fps, drop_frames)
    }

    /// Returns the duration of the trim in microseconds.
    pub fn trim_duration_us(config: &FrameTrimConfig) -> u64 {
        config.out_point_pts_us() - config.in_point_pts_us()
    }

    /// Finds the nearest keyframe at or before `frame_index` in `media`.
    pub fn nearest_keyframe_before(media: &MediaSegment, frame_index: u64) -> Option<u64> {
        let cap = (frame_index as usize).min(media.frames.len().saturating_sub(1));
        for i in (0..=cap).rev() {
            if media.frames[i].keyframe {
                return Some(i as u64);
            }
        }
        None
    }

    /// Finds the nearest keyframe at or after `frame_index` in `media`.
    pub fn nearest_keyframe_after(media: &MediaSegment, frame_index: u64) -> Option<u64> {
        let start = frame_index as usize;
        for i in start..media.frames.len() {
            if media.frames[i].keyframe {
                return Some(i as u64);
            }
        }
        None
    }

    /// Generates a comprehensive trim report.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`trim`][Self::trim].
    pub fn trim_report(
        media: &MediaSegment,
        config: &FrameTrimConfig,
    ) -> Result<TrimReport, FrameTrimError> {
        // Validate before computing anything
        let total = media.total_frames;
        if total == 0 {
            return Err(FrameTrimError::EmptySegment);
        }
        if config.in_point_frames >= total {
            return Err(FrameTrimError::InPointOutOfRange {
                in_point: config.in_point_frames,
                total_frames: total,
            });
        }
        if config.out_point_frames > total {
            return Err(FrameTrimError::OutPointOutOfRange {
                out_point: config.out_point_frames,
                total_frames: total,
            });
        }
        if config.in_point_frames >= config.out_point_frames {
            return Err(FrameTrimError::InPointAfterOutPoint {
                in_point: config.in_point_frames,
                out_point: config.out_point_frames,
            });
        }

        let in_tc = Self::to_timecode(config.in_point_frames, config.fps_num, config.fps_den);
        let out_tc = Self::to_timecode(config.out_point_frames, config.fps_num, config.fps_den);

        let kf_in = Self::nearest_keyframe_before(media, config.in_point_frames);
        let kf_out = if config.out_point_frames < total {
            Self::nearest_keyframe_after(media, config.out_point_frames)
        } else {
            None
        };

        let trimmed_size: usize = media.frames
            [config.in_point_frames as usize..config.out_point_frames as usize]
            .iter()
            .map(|f| f.data.len())
            .sum();

        Ok(TrimReport {
            in_point_frames: config.in_point_frames,
            out_point_frames: config.out_point_frames,
            duration_frames: config.duration_frames(),
            in_point_timecode: in_tc,
            out_point_timecode: out_tc,
            in_point_pts_us: config.in_point_pts_us(),
            out_point_pts_us: config.out_point_pts_us(),
            nearest_keyframe_in: kf_in,
            nearest_keyframe_out: kf_out,
            trimmed_size_bytes: trimmed_size,
        })
    }
}

impl Default for FrameTrimmer {
    fn default() -> Self {
        Self::new()
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Converts a frame count to microseconds.
fn frame_to_pts_us(frames: u64, fps_num: u32, fps_den: u32) -> u64 {
    if fps_num == 0 {
        return 0;
    }
    // frames * (fps_den / fps_num) * 1_000_000
    // Use u128 to avoid overflow for large frame counts
    let us = (frames as u128 * fps_den as u128 * 1_000_000u128) / fps_num as u128;
    us as u64
}

/// Returns `(nominal_fps, drop_frame_count)`.
///
/// Drop-frame rates:
/// - 30000/1001 (29.97) → nominal 30, drop 2
/// - 60000/1001 (59.94) → nominal 60, drop 4
fn nominal_fps_and_drop(fps_num: u32, fps_den: u32) -> (u32, u32) {
    match (fps_num, fps_den) {
        (30000, 1001) => (30, 2),
        (60000, 1001) => (60, 4),
        _ => {
            // Non-drop: nominal fps is the rounded integer
            let nominal = (fps_num + fps_den / 2) / fps_den;
            (nominal, 0)
        }
    }
}

/// Converts a frame count to non-drop-frame timecode `HH:MM:SS:FF`.
fn to_nondrop_timecode(frame: u64, fps: u32) -> String {
    if fps == 0 {
        return "00:00:00:00".to_string();
    }
    let fps = fps as u64;
    let ff = frame % fps;
    let total_secs = frame / fps;
    let ss = total_secs % 60;
    let total_mins = total_secs / 60;
    let mm = total_mins % 60;
    let hh = total_mins / 60;
    format!("{:02}:{:02}:{:02}:{:02}", hh, mm, ss, ff)
}

/// Converts a frame count to SMPTE drop-frame timecode `HH:MM:SS;FF`.
///
/// Algorithm:
/// 1. `d` = frames-per-10-minutes = `fps_nominal * 600 - drop_frames * 9`
/// 2. Compute which 10-minute block we're in and the remainder.
/// 3. Within the remainder, adjust for per-minute drops.
fn to_drop_frame_timecode(frame: u64, nominal_fps: u32, drop_frames: u32) -> String {
    let fps = nominal_fps as u64;
    let df = drop_frames as u64;

    // Frames per 10-minute block (no drop at minute 0, 10, 20, ...)
    let frames_per_10min = fps * 60 * 10 - df * 9;
    let frames_per_min = fps * 60 - df;

    let d = frame / frames_per_10min;
    let m = frame % frames_per_10min;

    // Adjust minutes within the 10-min block
    let adjusted = if m < df {
        m
    } else {
        // 0th minute in block has no drop; subsequent minutes do
        let extra_mins = (m - df) / frames_per_min + 1;
        m + df * extra_mins
    };

    let ff = adjusted % fps;
    let total_secs = adjusted / fps;
    let ss = total_secs % 60;
    let mm_inner = total_secs / 60 % 6; // within the 10-minute block
    let mm = d % 6 * 10 + mm_inner;
    let hh_total = d / 6;

    format!("{:02}:{:02}:{:02};{:02}", hh_total, mm, ss, ff)
}

/// Parses a SMPTE timecode string into a frame count.
///
/// Accepts `HH:MM:SS:FF` (non-drop) or `HH:MM:SS;FF` (drop-frame).
fn parse_timecode(tc: &str, nominal_fps: u32, drop_frames: u32) -> Result<u64, FrameTrimError> {
    // Replace ';' with ':' to normalise, then split
    let normalised = tc.replace(';', ":");
    let parts: Vec<&str> = normalised.splitn(4, ':').collect();
    if parts.len() != 4 {
        return Err(FrameTrimError::InvalidTimecode(tc.to_string()));
    }
    let parse = |s: &str| -> Result<u64, FrameTrimError> {
        s.trim()
            .parse::<u64>()
            .map_err(|_| FrameTrimError::InvalidTimecode(tc.to_string()))
    };
    let hh = parse(parts[0])?;
    let mm = parse(parts[1])?;
    let ss = parse(parts[2])?;
    let ff = parse(parts[3])?;

    let fps = nominal_fps as u64;
    let df = drop_frames as u64;

    if drop_frames == 0 {
        // Non-drop: straightforward conversion
        let total_frames = ff + ss * fps + mm * fps * 60 + hh * fps * 3600;
        Ok(total_frames)
    } else {
        // Drop-frame: reverse the drop-frame calculation
        let total_mins = hh * 60 + mm;
        let drop_count = df * (total_mins - total_mins / 10);
        let total_frames = ff + ss * fps + total_mins * fps * 60 + hh * fps * 3600 - drop_count;
        Ok(total_frames)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_segment(frame_count: u64, fps_num: u32, fps_den: u32) -> MediaSegment {
        let mut seg = MediaSegment::new(fps_num, fps_den);
        for i in 0..frame_count {
            seg.add_frame(vec![i as u8; 16], i % 5 == 0);
        }
        seg
    }

    // ── FrameTrimConfig ───────────────────────────────────────────────────────

    #[test]
    fn test_config_new_valid() {
        let cfg = FrameTrimConfig::new(0, 25, 25, 1).expect("should be valid");
        assert_eq!(cfg.in_point_frames, 0);
        assert_eq!(cfg.out_point_frames, 25);
    }

    #[test]
    fn test_config_in_after_out_returns_error() {
        let err = FrameTrimConfig::new(10, 5, 25, 1).unwrap_err();
        assert!(matches!(
            err,
            FrameTrimError::InPointAfterOutPoint {
                in_point: 10,
                out_point: 5
            }
        ));
    }

    #[test]
    fn test_config_equal_in_out_returns_error() {
        let err = FrameTrimConfig::new(5, 5, 25, 1).unwrap_err();
        assert!(matches!(err, FrameTrimError::InPointAfterOutPoint { .. }));
    }

    #[test]
    fn test_config_zero_fps_num_returns_error() {
        let err = FrameTrimConfig::new(0, 10, 0, 1).unwrap_err();
        assert!(matches!(
            err,
            FrameTrimError::InvalidFrameRate {
                fps_num: 0,
                fps_den: 1
            }
        ));
    }

    #[test]
    fn test_config_zero_fps_den_returns_error() {
        let err = FrameTrimConfig::new(0, 10, 25, 0).unwrap_err();
        assert!(matches!(
            err,
            FrameTrimError::InvalidFrameRate {
                fps_num: 25,
                fps_den: 0
            }
        ));
    }

    #[test]
    fn test_config_duration_frames() {
        let cfg = FrameTrimConfig::new(10, 35, 25, 1).expect("valid");
        assert_eq!(cfg.duration_frames(), 25);
    }

    #[test]
    fn test_config_fps_as_f64_25() {
        let cfg = FrameTrimConfig::new(0, 1, 25, 1).expect("valid");
        assert!((cfg.fps_as_f64() - 25.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_fps_as_f64_2997() {
        let cfg = FrameTrimConfig::new(0, 1, 30000, 1001).expect("valid");
        assert!((cfg.fps_as_f64() - 29.97).abs() < 0.01);
    }

    #[test]
    fn test_config_in_point_pts_us() {
        // At 25 fps, 1 frame = 40_000 µs
        let cfg = FrameTrimConfig::new(5, 10, 25, 1).expect("valid");
        assert_eq!(cfg.in_point_pts_us(), 5 * 40_000);
    }

    #[test]
    fn test_config_out_point_pts_us() {
        let cfg = FrameTrimConfig::new(0, 25, 25, 1).expect("valid");
        // 25 frames at 25 fps = 1_000_000 µs = 1 second
        assert_eq!(cfg.out_point_pts_us(), 1_000_000);
    }

    // ── MediaSegment ─────────────────────────────────────────────────────────

    #[test]
    fn test_segment_add_frame_increments_count() {
        let mut seg = MediaSegment::new(25, 1);
        assert_eq!(seg.frame_count(), 0);
        seg.add_frame(vec![0u8; 10], true);
        assert_eq!(seg.frame_count(), 1);
    }

    #[test]
    fn test_segment_pts_increments_correctly() {
        let mut seg = MediaSegment::new(25, 1);
        seg.add_frame(vec![0u8], true);
        seg.add_frame(vec![0u8], false);
        assert_eq!(seg.frames[0].pts_us, 0);
        assert_eq!(seg.frames[1].pts_us, 40_000); // 1/25 s = 40 ms = 40_000 µs
    }

    #[test]
    fn test_segment_duration_us() {
        let seg = make_segment(25, 25, 1);
        assert_eq!(seg.duration_us(), 1_000_000);
    }

    // ── FrameTrimmer::trim ───────────────────────────────────────────────────

    #[test]
    fn test_trim_basic() {
        let seg = make_segment(50, 25, 1);
        let cfg = FrameTrimConfig::new(10, 35, 25, 1).expect("valid");
        let trimmed = FrameTrimmer::trim(&seg, &cfg).expect("trim ok");
        assert_eq!(trimmed.total_frames, 25);
    }

    #[test]
    fn test_trim_re_indexes_frames() {
        let seg = make_segment(50, 25, 1);
        let cfg = FrameTrimConfig::new(10, 20, 25, 1).expect("valid");
        let trimmed = FrameTrimmer::trim(&seg, &cfg).expect("trim ok");
        assert_eq!(trimmed.frames[0].index, 0);
        assert_eq!(trimmed.frames[9].index, 9);
    }

    #[test]
    fn test_trim_out_of_range_in_point() {
        let seg = make_segment(10, 25, 1);
        let cfg = FrameTrimConfig::new(10, 15, 25, 1).expect("valid");
        let err = FrameTrimmer::trim(&seg, &cfg).unwrap_err();
        assert!(matches!(err, FrameTrimError::InPointOutOfRange { .. }));
    }

    #[test]
    fn test_trim_out_of_range_out_point() {
        let seg = make_segment(10, 25, 1);
        let cfg = FrameTrimConfig::new(5, 15, 25, 1).expect("valid");
        let err = FrameTrimmer::trim(&seg, &cfg).unwrap_err();
        assert!(matches!(err, FrameTrimError::OutPointOutOfRange { .. }));
    }

    #[test]
    fn test_trim_empty_segment() {
        let seg = MediaSegment::new(25, 1);
        let cfg = FrameTrimConfig::new(0, 5, 25, 1).expect("valid");
        let err = FrameTrimmer::trim(&seg, &cfg).unwrap_err();
        assert!(matches!(err, FrameTrimError::EmptySegment));
    }

    // ── FrameTrimmer::to_timecode ────────────────────────────────────────────

    #[test]
    fn test_timecode_25fps_zero() {
        assert_eq!(FrameTrimmer::to_timecode(0, 25, 1), "00:00:00:00");
    }

    #[test]
    fn test_timecode_25fps_one_second() {
        assert_eq!(FrameTrimmer::to_timecode(25, 25, 1), "00:00:01:00");
    }

    #[test]
    fn test_timecode_25fps_one_minute() {
        assert_eq!(FrameTrimmer::to_timecode(25 * 60, 25, 1), "00:01:00:00");
    }

    #[test]
    fn test_timecode_30fps_last_frame_of_second() {
        assert_eq!(FrameTrimmer::to_timecode(29, 30, 1), "00:00:00:29");
    }

    #[test]
    fn test_timecode_30fps_one_hour() {
        assert_eq!(FrameTrimmer::to_timecode(30 * 3600, 30, 1), "01:00:00:00");
    }

    #[test]
    fn test_timecode_2997_uses_semicolon() {
        let tc = FrameTrimmer::to_timecode(0, 30000, 1001);
        assert!(
            tc.contains(';'),
            "29.97 timecode should use ';' separator: {}",
            tc
        );
    }

    #[test]
    fn test_timecode_5994_uses_semicolon() {
        let tc = FrameTrimmer::to_timecode(0, 60000, 1001);
        assert!(
            tc.contains(';'),
            "59.94 timecode should use ';' separator: {}",
            tc
        );
    }

    // ── FrameTrimmer::from_timecode (round-trip) ──────────────────────────────

    #[test]
    fn test_from_timecode_25fps_roundtrip() {
        for frame in [0u64, 1, 24, 25, 100, 1501] {
            let tc = FrameTrimmer::to_timecode(frame, 25, 1);
            let back = FrameTrimmer::from_timecode(&tc, 25, 1).expect("parse ok");
            assert_eq!(back, frame, "round-trip failed for frame {frame}: tc={tc}");
        }
    }

    #[test]
    fn test_from_timecode_30fps_roundtrip() {
        for frame in [0u64, 29, 30, 1800] {
            let tc = FrameTrimmer::to_timecode(frame, 30, 1);
            let back = FrameTrimmer::from_timecode(&tc, 30, 1).expect("parse ok");
            assert_eq!(back, frame, "round-trip failed for frame {frame}");
        }
    }

    #[test]
    fn test_from_timecode_invalid_returns_error() {
        let err = FrameTrimmer::from_timecode("not:a:timecode", 25, 1).unwrap_err();
        assert!(matches!(err, FrameTrimError::InvalidTimecode(_)));
    }

    // ── Nearest keyframe ─────────────────────────────────────────────────────

    #[test]
    fn test_nearest_keyframe_before() {
        let seg = make_segment(20, 25, 1); // keyframes at 0, 5, 10, 15
                                           // Nearest keyframe before frame 12 → frame 10
        let kf = FrameTrimmer::nearest_keyframe_before(&seg, 12);
        assert_eq!(kf, Some(10));
    }

    #[test]
    fn test_nearest_keyframe_before_at_keyframe() {
        let seg = make_segment(20, 25, 1);
        let kf = FrameTrimmer::nearest_keyframe_before(&seg, 10);
        assert_eq!(kf, Some(10));
    }

    #[test]
    fn test_nearest_keyframe_after() {
        let seg = make_segment(20, 25, 1);
        // Nearest keyframe at or after frame 11 → frame 15
        let kf = FrameTrimmer::nearest_keyframe_after(&seg, 11);
        assert_eq!(kf, Some(15));
    }

    #[test]
    fn test_nearest_keyframe_after_none() {
        let seg = make_segment(20, 25, 1);
        // No keyframe after frame 16 (last keyframe is 15)
        let kf = FrameTrimmer::nearest_keyframe_after(&seg, 16);
        assert!(kf.is_none());
    }

    // ── TrimReport ────────────────────────────────────────────────────────────

    #[test]
    fn test_trim_report_basic() {
        let seg = make_segment(50, 25, 1);
        let cfg = FrameTrimConfig::new(5, 30, 25, 1).expect("valid");
        let report = FrameTrimmer::trim_report(&seg, &cfg).expect("report ok");
        assert_eq!(report.duration_frames, 25);
        assert!(!report.in_point_timecode.is_empty());
        assert!(!report.out_point_timecode.is_empty());
        assert_eq!(report.in_point_pts_us, 5 * 40_000);
        assert_eq!(report.out_point_pts_us, 30 * 40_000);
    }

    #[test]
    fn test_trim_report_size_bytes() {
        let seg = make_segment(20, 25, 1);
        let cfg = FrameTrimConfig::new(0, 10, 25, 1).expect("valid");
        let report = FrameTrimmer::trim_report(&seg, &cfg).expect("report ok");
        // Each frame is 16 bytes (from make_segment), 10 frames = 160 bytes
        assert_eq!(report.trimmed_size_bytes, 10 * 16);
    }

    #[test]
    fn test_trim_duration_us() {
        let cfg = FrameTrimConfig::new(0, 25, 25, 1).expect("valid");
        assert_eq!(FrameTrimmer::trim_duration_us(&cfg), 1_000_000);
    }
}
