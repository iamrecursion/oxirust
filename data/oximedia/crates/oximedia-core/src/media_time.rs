//! Media timing primitives: `MediaTime`, `TimeRange`, and `MediaTimeCalc`.
//!
//! Provides lightweight time representation using integer numerator/denominator
//! pairs for frame-accurate arithmetic without floating-point rounding errors.

#![allow(dead_code)]

use std::fmt;

/// A media timestamp expressed as `ticks / time_base`.
///
/// Avoids floating-point representation to preserve frame accuracy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MediaTime {
    /// Tick count (numerator).
    pub ticks: i64,
    /// Time base denominator (e.g. 90000 for MPEG, 48000 for audio).
    pub time_base: u64,
}

impl MediaTime {
    /// Creates a `MediaTime` at exactly zero.
    #[must_use]
    pub const fn zero(time_base: u64) -> Self {
        Self {
            ticks: 0,
            time_base,
        }
    }

    /// Creates a `MediaTime` from whole seconds.
    ///
    /// # Panics
    ///
    /// Panics if `time_base` is zero.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    pub fn from_secs(secs: f64, time_base: u64) -> Self {
        assert!(time_base > 0, "time_base must be non-zero");
        let ticks = (secs * time_base as f64).round() as i64;
        Self { ticks, time_base }
    }

    /// Converts this `MediaTime` to seconds as a 64-bit float.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn to_secs(&self) -> f64 {
        self.ticks as f64 / self.time_base as f64
    }

    /// Adds an offset in ticks (same time base) and returns a new `MediaTime`.
    #[must_use]
    pub const fn add_offset(&self, offset_ticks: i64) -> Self {
        Self {
            ticks: self.ticks + offset_ticks,
            time_base: self.time_base,
        }
    }

    /// Returns `true` if this time is strictly before `other`.
    ///
    /// Both times must share the same time base; otherwise this compares
    /// the raw tick values which may be misleading.
    #[must_use]
    pub fn is_before(&self, other: &Self) -> bool {
        if self.time_base == other.time_base {
            self.ticks < other.ticks
        } else {
            self.to_secs() < other.to_secs()
        }
    }

    /// Returns `true` if this time is strictly after `other`.
    #[must_use]
    pub fn is_after(&self, other: &Self) -> bool {
        other.is_before(self)
    }

    /// Returns the absolute difference between two `MediaTime` values in seconds.
    #[must_use]
    pub fn abs_diff_secs(&self, other: &Self) -> f64 {
        (self.to_secs() - other.to_secs()).abs()
    }

    /// Rescales this `MediaTime` to a different time base.
    ///
    /// # Panics
    ///
    /// Panics if `new_base` is zero.
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    #[must_use]
    pub fn rescale(&self, new_base: u64) -> Self {
        assert!(new_base > 0, "time_base must be non-zero");
        let new_ticks =
            (self.ticks as f64 * new_base as f64 / self.time_base as f64).round() as i64;
        Self {
            ticks: new_ticks,
            time_base: new_base,
        }
    }
}

impl fmt::Display for MediaTime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.6}s", self.to_secs())
    }
}

/// A half-open time range `[start, end)` in the same time base.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeRange {
    /// Inclusive start.
    pub start: MediaTime,
    /// Exclusive end.
    pub end: MediaTime,
}

impl TimeRange {
    /// Creates a new `TimeRange`.
    ///
    /// # Panics
    /// Panics if `start` is after `end`.
    #[must_use]
    pub fn new(start: MediaTime, end: MediaTime) -> Self {
        assert!(
            !end.is_before(&start),
            "TimeRange: start must not be after end"
        );
        Self { start, end }
    }

    /// Returns the duration of this range in seconds.
    #[must_use]
    pub fn duration(&self) -> f64 {
        self.end.to_secs() - self.start.to_secs()
    }

    /// Returns `true` if `time` falls within `[start, end)`.
    #[must_use]
    pub fn contains(&self, time: &MediaTime) -> bool {
        !time.is_before(&self.start) && time.is_before(&self.end)
    }

    /// Returns `true` if this range overlaps with `other`.
    #[must_use]
    pub fn overlaps(&self, other: &Self) -> bool {
        self.start.is_before(&other.end) && other.start.is_before(&self.end)
    }

    /// Returns the overlap range between `self` and `other`, if any.
    #[must_use]
    pub fn intersection(&self, other: &Self) -> Option<Self> {
        let start_secs = self.start.to_secs().max(other.start.to_secs());
        let end_secs = self.end.to_secs().min(other.end.to_secs());
        if end_secs <= start_secs {
            return None;
        }
        let tb = self.start.time_base;
        Some(Self::new(
            MediaTime::from_secs(start_secs, tb),
            MediaTime::from_secs(end_secs, tb),
        ))
    }
}

impl fmt::Display for TimeRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {})", self.start, self.end)
    }
}

/// Helper for common PTS/DTS calculations.
///
/// Converts presentation timestamps to decode timestamps given a constant
/// B-frame delay (measured in codec time-base ticks).
#[derive(Debug, Clone, Copy)]
pub struct MediaTimeCalc {
    /// Number of B-frames in the codec's lookahead (determines PTS→DTS offset).
    pub b_frame_delay: u32,
    /// Codec time base.
    pub time_base: u64,
}

impl MediaTimeCalc {
    /// Creates a `MediaTimeCalc` with the given parameters.
    #[must_use]
    pub const fn new(b_frame_delay: u32, time_base: u64) -> Self {
        Self {
            b_frame_delay,
            time_base,
        }
    }

    /// Converts a PTS (presentation timestamp in ticks) to a DTS (decode
    /// timestamp in ticks) by subtracting the B-frame delay.
    #[must_use]
    pub fn pts_to_dts(&self, pts: MediaTime) -> MediaTime {
        let delay = i64::from(self.b_frame_delay);
        pts.add_offset(-delay)
    }

    /// Converts a DTS back to a PTS by adding the B-frame delay.
    #[must_use]
    pub fn dts_to_pts(&self, dts: MediaTime) -> MediaTime {
        let delay = i64::from(self.b_frame_delay);
        dts.add_offset(delay)
    }

    /// Returns the minimum DTS offset in seconds caused by the B-frame delay.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn dts_offset_secs(&self) -> f64 {
        f64::from(self.b_frame_delay) / self.time_base as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero() {
        let t = MediaTime::zero(90_000);
        assert_eq!(t.ticks, 0);
        assert_eq!(t.to_secs(), 0.0);
    }

    #[test]
    fn test_from_secs_to_secs() {
        let t = MediaTime::from_secs(1.0, 90_000);
        assert!((t.to_secs() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_from_secs_half() {
        let t = MediaTime::from_secs(0.5, 90_000);
        assert!((t.to_secs() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_add_offset() {
        let t = MediaTime::from_secs(1.0, 90_000);
        let t2 = t.add_offset(90_000);
        assert!((t2.to_secs() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_is_before() {
        let t1 = MediaTime::from_secs(1.0, 90_000);
        let t2 = MediaTime::from_secs(2.0, 90_000);
        assert!(t1.is_before(&t2));
        assert!(!t2.is_before(&t1));
    }

    #[test]
    fn test_is_after() {
        let t1 = MediaTime::from_secs(1.0, 90_000);
        let t2 = MediaTime::from_secs(2.0, 90_000);
        assert!(t2.is_after(&t1));
        assert!(!t1.is_after(&t2));
    }

    #[test]
    fn test_abs_diff_secs() {
        let t1 = MediaTime::from_secs(1.0, 90_000);
        let t2 = MediaTime::from_secs(3.0, 90_000);
        assert!((t1.abs_diff_secs(&t2) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_rescale() {
        let t = MediaTime::from_secs(1.0, 90_000);
        let t2 = t.rescale(48_000);
        assert!((t2.to_secs() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_display() {
        let t = MediaTime::from_secs(1.5, 90_000);
        let s = format!("{t}");
        assert!(s.contains("1.5"));
    }

    #[test]
    fn test_time_range_duration() {
        let s = MediaTime::from_secs(0.0, 90_000);
        let e = MediaTime::from_secs(2.0, 90_000);
        let r = TimeRange::new(s, e);
        assert!((r.duration() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_time_range_contains() {
        let s = MediaTime::from_secs(1.0, 90_000);
        let e = MediaTime::from_secs(5.0, 90_000);
        let r = TimeRange::new(s, e);
        let mid = MediaTime::from_secs(3.0, 90_000);
        assert!(r.contains(&mid));
        assert!(!r.contains(&MediaTime::from_secs(0.5, 90_000)));
        // End is exclusive
        assert!(!r.contains(&e));
    }

    #[test]
    fn test_time_range_overlaps() {
        let r1 = TimeRange::new(
            MediaTime::from_secs(0.0, 90_000),
            MediaTime::from_secs(4.0, 90_000),
        );
        let r2 = TimeRange::new(
            MediaTime::from_secs(2.0, 90_000),
            MediaTime::from_secs(6.0, 90_000),
        );
        assert!(r1.overlaps(&r2));
    }

    #[test]
    fn test_time_range_no_overlap() {
        let r1 = TimeRange::new(
            MediaTime::from_secs(0.0, 90_000),
            MediaTime::from_secs(2.0, 90_000),
        );
        let r2 = TimeRange::new(
            MediaTime::from_secs(3.0, 90_000),
            MediaTime::from_secs(5.0, 90_000),
        );
        assert!(!r1.overlaps(&r2));
    }

    #[test]
    fn test_time_range_intersection() {
        let r1 = TimeRange::new(
            MediaTime::from_secs(0.0, 90_000),
            MediaTime::from_secs(4.0, 90_000),
        );
        let r2 = TimeRange::new(
            MediaTime::from_secs(2.0, 90_000),
            MediaTime::from_secs(6.0, 90_000),
        );
        let inter = r1.intersection(&r2).expect("intersection should exist");
        assert!((inter.start.to_secs() - 2.0).abs() < 1e-4);
        assert!((inter.end.to_secs() - 4.0).abs() < 1e-4);
    }

    #[test]
    fn test_time_range_intersection_none() {
        let r1 = TimeRange::new(
            MediaTime::from_secs(0.0, 90_000),
            MediaTime::from_secs(2.0, 90_000),
        );
        let r2 = TimeRange::new(
            MediaTime::from_secs(3.0, 90_000),
            MediaTime::from_secs(5.0, 90_000),
        );
        assert!(r1.intersection(&r2).is_none());
    }

    #[test]
    fn test_pts_to_dts_no_delay() {
        let calc = MediaTimeCalc::new(0, 90_000);
        let pts = MediaTime::from_secs(1.0, 90_000);
        let dts = calc.pts_to_dts(pts);
        assert_eq!(dts, pts);
    }

    #[test]
    fn test_pts_to_dts_with_delay() {
        let calc = MediaTimeCalc::new(2, 90_000);
        let pts = MediaTime {
            ticks: 100,
            time_base: 90_000,
        };
        let dts = calc.pts_to_dts(pts);
        assert_eq!(dts.ticks, 98);
    }

    #[test]
    fn test_dts_to_pts_roundtrip() {
        let calc = MediaTimeCalc::new(3, 90_000);
        let pts = MediaTime::from_secs(5.0, 90_000);
        let dts = calc.pts_to_dts(pts);
        let pts2 = calc.dts_to_pts(dts);
        assert_eq!(pts, pts2);
    }

    #[test]
    fn test_dts_offset_secs() {
        let calc = MediaTimeCalc::new(2, 90_000);
        let offset = calc.dts_offset_secs();
        assert!((offset - 2.0 / 90_000.0).abs() < 1e-12);
    }
}

// ---------------------------------------------------------------------------
// Rich media time using Rational time base
// ---------------------------------------------------------------------------

use crate::types::Rational;

/// Standard time base: 1/90000 (MPEG / H.264 / VP9 default).
pub const TB_90K: Rational = Rational {
    num: 1,
    den: 90_000,
};

/// Standard time base: 1/44100 (CD-quality audio).
pub const TB_44100: Rational = Rational {
    num: 1,
    den: 44_100,
};

/// Standard time base: 1/48000 (broadcast audio).
pub const TB_48K: Rational = Rational {
    num: 1,
    den: 48_000,
};

/// Standard time base: 1/1000 (millisecond precision).
pub const TB_1K: Rational = Rational { num: 1, den: 1_000 };

/// A media presentation timestamp with an associated [`Rational`] time base.
///
/// `pts` is measured in units of `time_base`; the wall-clock time is
/// `pts * time_base.num / time_base.den` seconds.
///
/// # Examples
///
/// ```
/// use oximedia_core::media_time::{PtsMediaTime, TB_90K};
///
/// let t = PtsMediaTime::new(90_000, TB_90K);
/// assert!((t.to_secs() - 1.0).abs() < 1e-9);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PtsMediaTime {
    /// Raw tick count (presentation timestamp).
    pub pts: i64,
    /// Time base as a rational fraction (e.g. `1/90000`).
    pub time_base: Rational,
}

impl PtsMediaTime {
    /// The canonical zero timestamp in a 1/90000 time base.
    pub const ZERO: Self = Self {
        pts: 0,
        time_base: TB_90K,
    };

    /// Creates a new `PtsMediaTime`.
    #[must_use]
    pub fn new(pts: i64, time_base: Rational) -> Self {
        Self { pts, time_base }
    }

    /// Converts seconds to ticks in the given time base and returns the
    /// resulting `PtsMediaTime`.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    pub fn from_secs(secs: f64, time_base: Rational) -> Self {
        // ticks = secs / (num/den) = secs * den / num
        let ticks = (secs * time_base.den as f64 / time_base.num as f64).round() as i64;
        Self {
            pts: ticks,
            time_base,
        }
    }

    /// Returns the wall-clock time in seconds.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn to_secs(&self) -> f64 {
        self.pts as f64 * self.time_base.to_f64()
    }

    /// Returns the wall-clock time in whole milliseconds (truncated toward
    /// zero).
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    pub fn to_ms(&self) -> i64 {
        (self.to_secs() * 1000.0) as i64
    }

    /// Rescales this timestamp to `new_base`, preserving the wall-clock
    /// instant as accurately as possible.
    ///
    /// Uses 128-bit intermediate arithmetic to prevent overflow.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn rebase(&self, new_base: Rational) -> Self {
        // new_pts = pts * (old_num / old_den) / (new_num / new_den)
        //         = pts * old_num * new_den / (old_den * new_num)
        let scale_num = i128::from(self.time_base.num) * i128::from(new_base.den);
        let scale_den = i128::from(self.time_base.den) * i128::from(new_base.num);
        let half = scale_den / 2;
        let new_pts = ((i128::from(self.pts) * scale_num + half) / scale_den) as i64;
        Self {
            pts: new_pts,
            time_base: new_base,
        }
    }

    /// Returns `true` if the presentation timestamp is non-negative (valid
    /// for presentation).
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.pts >= 0
    }

    /// Adds a [`MediaDuration`] to this timestamp.
    ///
    /// Both operands must share the same time base; if they differ the
    /// duration is first rebased.
    #[must_use]
    pub fn add_duration(&self, d: MediaDuration) -> Self {
        let d_rebased = d.rebase(self.time_base);
        Self {
            pts: self.pts + d_rebased.ticks,
            time_base: self.time_base,
        }
    }

    /// Returns the difference `self - other` as a [`MediaDuration`] in
    /// `self`'s time base.
    #[must_use]
    pub fn subtract(&self, other: &Self) -> MediaDuration {
        let other_rebased = other.rebase(self.time_base);
        MediaDuration {
            ticks: self.pts - other_rebased.pts,
            time_base: self.time_base,
        }
    }
}

/// A duration expressed in [`Rational`] time-base ticks.
#[derive(Debug, Clone, Copy)]
pub struct MediaDuration {
    /// Raw tick count.
    pub ticks: i64,
    /// Time base.
    pub time_base: Rational,
}

impl MediaDuration {
    /// Converts seconds to ticks and returns the resulting `MediaDuration`.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    pub fn from_secs(secs: f64, time_base: Rational) -> Self {
        let ticks = (secs * time_base.den as f64 / time_base.num as f64).round() as i64;
        Self { ticks, time_base }
    }

    /// Returns the duration in seconds.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn to_secs(&self) -> f64 {
        self.ticks as f64 * self.time_base.to_f64()
    }

    /// Rescales this duration to `new_base`.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn rebase(&self, new_base: Rational) -> Self {
        let scale_num = i128::from(self.time_base.num) * i128::from(new_base.den);
        let scale_den = i128::from(self.time_base.den) * i128::from(new_base.num);
        let half = scale_den / 2;
        let new_ticks = ((i128::from(self.ticks) * scale_num + half) / scale_den) as i64;
        Self {
            ticks: new_ticks,
            time_base: new_base,
        }
    }
}

#[cfg(test)]
mod tests_pts_media_time {
    use super::*;

    #[test]
    fn test_zero_constant() {
        assert_eq!(PtsMediaTime::ZERO.pts, 0);
        assert_eq!(PtsMediaTime::ZERO.time_base, TB_90K);
    }

    #[test]
    fn test_new_and_to_secs() {
        let t = PtsMediaTime::new(90_000, TB_90K);
        assert!((t.to_secs() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_from_secs() {
        let t = PtsMediaTime::from_secs(2.5, TB_90K);
        assert!((t.to_secs() - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_to_ms() {
        let t = PtsMediaTime::from_secs(1.5, TB_1K);
        assert_eq!(t.to_ms(), 1500);
    }

    #[test]
    fn test_rebase_90k_to_1k() {
        let t = PtsMediaTime::new(90_000, TB_90K);
        let r = t.rebase(TB_1K);
        assert_eq!(r.pts, 1000);
        assert!((r.to_secs() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_rebase_1k_to_44100() {
        let t = PtsMediaTime::from_secs(1.0, TB_1K);
        let r = t.rebase(TB_44100);
        assert_eq!(r.pts, 44_100);
    }

    #[test]
    fn test_is_valid() {
        assert!(PtsMediaTime::new(0, TB_90K).is_valid());
        assert!(PtsMediaTime::new(100, TB_90K).is_valid());
        assert!(!PtsMediaTime::new(-1, TB_90K).is_valid());
    }

    #[test]
    fn test_add_duration() {
        let t = PtsMediaTime::new(0, TB_1K);
        let d = MediaDuration::from_secs(1.5, TB_1K);
        let t2 = t.add_duration(d);
        assert_eq!(t2.pts, 1500);
    }

    #[test]
    fn test_add_duration_different_base() {
        let t = PtsMediaTime::new(90_000, TB_90K); // 1 second
        let d = MediaDuration::from_secs(0.5, TB_1K); // 0.5 seconds in 1k base
        let t2 = t.add_duration(d);
        assert!((t2.to_secs() - 1.5).abs() < 1e-4);
    }

    #[test]
    fn test_subtract() {
        let a = PtsMediaTime::new(3000, TB_1K);
        let b = PtsMediaTime::new(1000, TB_1K);
        let d = a.subtract(&b);
        assert!((d.to_secs() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_media_duration_from_secs_to_secs() {
        let d = MediaDuration::from_secs(0.25, TB_90K);
        assert!((d.to_secs() - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_media_duration_rebase() {
        let d = MediaDuration::from_secs(1.0, TB_90K);
        let r = d.rebase(TB_48K);
        assert_eq!(r.ticks, 48_000);
    }

    #[test]
    fn test_tb_constants() {
        assert_eq!(TB_90K.den, 90_000);
        assert_eq!(TB_44100.den, 44_100);
        assert_eq!(TB_48K.den, 48_000);
        assert_eq!(TB_1K.den, 1_000);
    }
}

// ---------------------------------------------------------------------------
// Bitrate / duration / size estimation utilities
// ---------------------------------------------------------------------------

/// Constant: nanoseconds per second.
const NS_PER_SEC: u64 = 1_000_000_000;

/// Estimates bitrate in bits per second from a byte count and a duration.
///
/// `duration_ns` must be non-zero; returns `None` otherwise.  All arithmetic
/// is checked so `None` is returned on overflow rather than panicking.
///
/// # Formula
///
/// ```text
/// bitrate_bps = bytes * 8 * 1_000_000_000 / duration_ns
/// ```
///
/// # Examples
///
/// ```
/// use oximedia_core::media_time::estimate_bitrate_bps;
///
/// // 100 KB in exactly 1 second → 800 kbps
/// let bps = estimate_bitrate_bps(100_000, 1_000_000_000).unwrap();
/// assert_eq!(bps, 800_000);
/// ```
#[must_use]
pub fn estimate_bitrate_bps(bytes: u64, duration_ns: u64) -> Option<u64> {
    if duration_ns == 0 {
        return None;
    }
    // Use u128 to avoid overflow in the intermediate product
    let bits_ns = (bytes as u128)
        .checked_mul(8)?
        .checked_mul(NS_PER_SEC as u128)?;
    let bps = bits_ns.checked_div(duration_ns as u128)?;
    // Downcast back to u64 — if it overflows u64 the result is None
    u64::try_from(bps).ok()
}

/// Estimates total file size in bytes for a given bitrate and duration.
///
/// Returns `None` on overflow or if `bitrate_bps` is zero.
///
/// # Formula
///
/// ```text
/// size_bytes = bitrate_bps * duration_ns / (8 * 1_000_000_000)
/// ```
///
/// # Examples
///
/// ```
/// use oximedia_core::media_time::estimate_size_bytes;
///
/// // 1 Mbps × 8 seconds = 1 MB
/// let bytes = estimate_size_bytes(1_000_000, 8_000_000_000).unwrap();
/// assert_eq!(bytes, 1_000_000);
/// ```
#[must_use]
pub fn estimate_size_bytes(bitrate_bps: u64, duration_ns: u64) -> Option<u64> {
    if bitrate_bps == 0 {
        return None;
    }
    let divisor = (8u128).checked_mul(NS_PER_SEC as u128)?;
    let numerator = (bitrate_bps as u128).checked_mul(duration_ns as u128)?;
    let size = numerator.checked_div(divisor)?;
    u64::try_from(size).ok()
}

/// Estimates duration in nanoseconds for a given file size and bitrate.
///
/// `bitrate_bps` must be non-zero; returns `None` otherwise.  All arithmetic
/// is checked so `None` is returned on overflow rather than panicking.
///
/// # Formula
///
/// ```text
/// duration_ns = bytes * 8 * 1_000_000_000 / bitrate_bps
/// ```
///
/// # Examples
///
/// ```
/// use oximedia_core::media_time::estimate_duration_ns;
///
/// // 100 KB at 800 kbps → exactly 1 second
/// let ns = estimate_duration_ns(100_000, 800_000).unwrap();
/// assert_eq!(ns, 1_000_000_000);
/// ```
#[must_use]
pub fn estimate_duration_ns(bytes: u64, bitrate_bps: u64) -> Option<u64> {
    if bitrate_bps == 0 {
        return None;
    }
    let bits_ns = (bytes as u128)
        .checked_mul(8)?
        .checked_mul(NS_PER_SEC as u128)?;
    let duration = bits_ns.checked_div(bitrate_bps as u128)?;
    u64::try_from(duration).ok()
}

// ---------------------------------------------------------------------------
// Tests — estimation utilities
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests_estimation {
    use super::*;

    /// 100 KB in exactly 1 second → 800 kbps; verify round-trip with
    /// `estimate_duration_ns`.
    #[test]
    fn test_bitrate_roundtrip() {
        let bytes: u64 = 100_000;
        let duration_ns: u64 = 1_000_000_000; // 1 second

        let bps = estimate_bitrate_bps(bytes, duration_ns).expect("should not overflow");
        assert_eq!(bps, 800_000, "100 KB in 1 s = 800 kbps");

        // Round-trip: recover duration from the computed bitrate
        let recovered_ns =
            estimate_duration_ns(bytes, bps).expect("round-trip should not overflow");
        assert_eq!(
            recovered_ns, duration_ns,
            "round-trip duration must match original"
        );
    }

    /// 1 Mbps × 8 seconds = 1 000 000 bytes.
    #[test]
    fn test_size_estimate() {
        let bitrate_bps: u64 = 1_000_000;
        let duration_ns: u64 = 8_000_000_000; // 8 seconds

        let size = estimate_size_bytes(bitrate_bps, duration_ns).expect("should not overflow");
        assert_eq!(size, 1_000_000, "1 Mbps × 8 s = 1 000 000 bytes");
    }

    /// Extremely large inputs must not panic — they return `None` on overflow.
    #[test]
    fn test_estimation_overflow_safe() {
        // bytes = u64::MAX, duration = 1 ns → intermediate product overflows
        // (u64::MAX * 8 overflows u128 when multiplied by 1e9 further on)
        // At minimum, the function must not panic.
        let _ = estimate_bitrate_bps(u64::MAX, 1);

        // bitrate = u64::MAX, duration = u64::MAX → numerator overflows u128
        let _ = estimate_size_bytes(u64::MAX, u64::MAX);

        // bytes = u64::MAX, bitrate = 1 → intermediate u128 may still fit,
        // but the final u64 cast may fail; must not panic.
        let _ = estimate_duration_ns(u64::MAX, 1);
    }

    /// Zero denominators must return `None`, not panic.
    #[test]
    fn test_estimation_zero_denominator() {
        assert!(
            estimate_bitrate_bps(1_000, 0).is_none(),
            "duration=0 must return None"
        );
        assert!(
            estimate_duration_ns(1_000, 0).is_none(),
            "bitrate=0 must return None"
        );
        assert!(
            estimate_size_bytes(0, 1_000_000_000).is_none(),
            "bitrate=0 must return None from estimate_size_bytes"
        );
    }
}
