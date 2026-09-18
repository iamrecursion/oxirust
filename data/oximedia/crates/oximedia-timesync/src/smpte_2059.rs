//! SMPTE ST 2059 — PTP Profile for Professional Media.
//!
//! SMPTE ST 2059-2 defines a PTP profile for professional broadcast and
//! production environments.  Key characteristics:
//!
//! * Domain 127 (configurable).
//! * `priority1` = 128, `priority2` = 128 (configurable).
//! * `logAnnounceInterval` = 0 (1 message per second).
//! * SMPTE Epoch = UNIX Epoch (1970-01-01 00:00:00 UTC).
//! * Frame-alignment: timestamps are aligned to the nearest video frame edge.

use std::fmt;

/// Rational number (numerator / denominator) used for frame rates.
///
/// Examples: 24/1 = 24 fps, 30000/1001 ≈ 29.97 fps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rational {
    /// Numerator.
    pub num: u64,
    /// Denominator (must be non-zero).
    pub den: u64,
}

impl Rational {
    /// Creates a new rational number.
    ///
    /// # Panics
    /// Does **not** panic; returns `1/1` if `den == 0` to avoid division by
    /// zero in downstream code.
    #[must_use]
    pub fn new(num: u64, den: u64) -> Self {
        if den == 0 {
            Self { num: 1, den: 1 }
        } else {
            Self { num, den }
        }
    }

    /// Converts to an `f64` value.
    #[must_use]
    pub fn to_f64(self) -> f64 {
        self.num as f64 / self.den as f64
    }

    /// Returns the frame duration in nanoseconds (rounded down).
    ///
    /// `frame_duration_ns = 1_000_000_000 × den / num`
    #[must_use]
    pub fn frame_duration_ns(self) -> u64 {
        if self.num == 0 {
            return 0;
        }
        1_000_000_000u128
            .saturating_mul(self.den as u128)
            .checked_div(self.num as u128)
            .unwrap_or(0) as u64
    }
}

impl fmt::Display for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.num, self.den)
    }
}

/// SMPTE ST 2059 PTP profile configuration.
///
/// Defines the PTP dataset fields required by ST 2059-2.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Smpte2059Profile {
    /// PTP domain number (default: 127).
    pub domain: u8,
    /// Priority 1 (default: 128).
    pub priority1: u8,
    /// Priority 2 (default: 128).
    pub priority2: u8,
    /// Log₂ of announce interval in seconds (default: 0 → 1 msg/s).
    pub announce_interval_log2: i8,
    /// Log₂ of sync interval in seconds (default: −3 → 8 msg/s).
    pub sync_interval_log2: i8,
    /// Log₂ of delay-request interval in seconds (default: 0).
    pub delay_req_interval_log2: i8,
    /// Announce receipt timeout (default: 3 intervals).
    pub announce_receipt_timeout: u8,
}

impl Smpte2059Profile {
    /// Returns the ST 2059-2 default profile.
    ///
    /// Domain 127, priority1 = 128, priority2 = 128,
    /// logAnnounceInterval = 0, logSyncInterval = −3.
    #[must_use]
    pub fn default_profile() -> Self {
        Self {
            domain: 127,
            priority1: 128,
            priority2: 128,
            announce_interval_log2: 0,
            sync_interval_log2: -3,
            delay_req_interval_log2: 0,
            announce_receipt_timeout: 3,
        }
    }

    /// Returns the announce interval in nanoseconds.
    #[must_use]
    pub fn announce_interval_ns(&self) -> u64 {
        log2_interval_to_ns(self.announce_interval_log2)
    }

    /// Returns the sync interval in nanoseconds.
    #[must_use]
    pub fn sync_interval_ns(&self) -> u64 {
        log2_interval_to_ns(self.sync_interval_log2)
    }

    /// Returns the announce timeout duration in nanoseconds.
    #[must_use]
    pub fn announce_timeout_ns(&self) -> u64 {
        self.announce_interval_ns()
            .saturating_mul(u64::from(self.announce_receipt_timeout))
    }

    /// Validates that this profile conforms to ST 2059-2 constraints.
    ///
    /// Returns `Ok(())` or an error string describing the first violation.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.domain > 127 {
            return Err("ST 2059 domain must be 0–127");
        }
        if self.announce_interval_log2 < -3 || self.announce_interval_log2 > 4 {
            return Err("ST 2059 logAnnounceInterval must be in [−3, 4]");
        }
        if self.announce_receipt_timeout < 2 || self.announce_receipt_timeout > 10 {
            return Err("ST 2059 announce receipt timeout must be in [2, 10]");
        }
        Ok(())
    }
}

impl Default for Smpte2059Profile {
    fn default() -> Self {
        Self::default_profile()
    }
}

/// Convert a log₂ interval value to a nanosecond duration.
///
/// For positive values: `2^n × 10^9` ns.
/// For negative values: `10^9 / 2^|n|` ns.
fn log2_interval_to_ns(log2: i8) -> u64 {
    const NS_PER_S: u64 = 1_000_000_000;
    if log2 >= 0 {
        NS_PER_S.saturating_mul(1u64 << (log2 as u32))
    } else {
        let divisor = 1u64 << ((-log2) as u32);
        NS_PER_S / divisor
    }
}

/// A PTP timestamp interpreted within the SMPTE ST 2059 context.
///
/// SMPTE Epoch = UNIX Epoch (1970-01-01 00:00:00 UTC), so no epoch conversion
/// is required.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Smpte2059Timestamp {
    /// Whole seconds since SMPTE Epoch (= UNIX Epoch).
    pub seconds: u64,
    /// Sub-second nanoseconds [0, 999_999_999].
    pub nanoseconds: u32,
}

impl Smpte2059Timestamp {
    /// Creates a timestamp, clamping `nanoseconds` to [0, 999_999_999].
    #[must_use]
    pub fn new(seconds: u64, nanoseconds: u32) -> Self {
        let ns = nanoseconds.min(999_999_999);
        Self {
            seconds,
            nanoseconds: ns,
        }
    }

    /// Converts from a PTP timestamp (seconds + nanoseconds since UNIX Epoch).
    ///
    /// Since the SMPTE Epoch equals the UNIX / PTP Epoch, this is a direct
    /// copy.
    #[must_use]
    pub fn from_ptp(seconds: u64, nanoseconds: u32) -> Self {
        Self::new(seconds, nanoseconds)
    }

    /// Total nanoseconds since the SMPTE Epoch.
    #[must_use]
    pub fn total_ns(&self) -> u128 {
        u128::from(self.seconds) * 1_000_000_000 + u128::from(self.nanoseconds)
    }

    /// Returns the difference `self - other` in nanoseconds (signed).
    ///
    /// Saturates at `i128::MIN` / `i128::MAX`.
    #[must_use]
    pub fn diff_ns(&self, other: &Self) -> i128 {
        let a = self.total_ns() as i128;
        let b = other.total_ns() as i128;
        a.saturating_sub(b)
    }

    /// Adds a signed nanosecond offset.
    ///
    /// Returns `None` if the result would be before the epoch.
    #[must_use]
    pub fn add_ns(&self, delta_ns: i64) -> Option<Self> {
        let total = self.total_ns() as i128 + i128::from(delta_ns);
        if total < 0 {
            return None;
        }
        let total = total as u128;
        let seconds = (total / 1_000_000_000) as u64;
        let nanoseconds = (total % 1_000_000_000) as u32;
        Some(Self {
            seconds,
            nanoseconds,
        })
    }
}

impl fmt::Display for Smpte2059Timestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{:09}", self.seconds, self.nanoseconds)
    }
}

/// Aligns a [`Smpte2059Timestamp`] to the nearest video frame boundary.
///
/// Per ST 2059-1, the grandmaster clock epoch is aligned to integer frame
/// boundaries relative to the SMPTE Epoch.  This function snaps `ts` to the
/// nearest frame edge (not just floor) for the given frame rate.
///
/// # Arguments
/// * `ts` — timestamp to align.
/// * `fps` — video frame rate as a [`Rational`] (e.g. `Rational::new(25, 1)`).
///
/// # Returns
/// A new timestamp aligned to the nearest frame boundary.
///
/// If `fps.num == 0` or `fps.frame_duration_ns() == 0`, the original timestamp
/// is returned unchanged.
#[must_use]
pub fn align_to_frame_rate(ts: &Smpte2059Timestamp, fps: Rational) -> Smpte2059Timestamp {
    let frame_dur_ns = fps.frame_duration_ns();
    if frame_dur_ns == 0 {
        return *ts;
    }
    let total = ts.total_ns();
    let frame_number = total / u128::from(frame_dur_ns);
    let frame_start = frame_number * u128::from(frame_dur_ns);
    let frame_end = frame_start + u128::from(frame_dur_ns);

    // Choose nearest boundary
    let dist_start = total - frame_start;
    let dist_end = frame_end - total;

    let aligned = if dist_start <= dist_end {
        frame_start
    } else {
        frame_end
    };

    let seconds = (aligned / 1_000_000_000) as u64;
    let nanoseconds = (aligned % 1_000_000_000) as u32;
    Smpte2059Timestamp {
        seconds,
        nanoseconds,
    }
}

/// Returns the SMPTE ST 2059 frame number for a given timestamp and frame rate.
///
/// Frame 0 starts at the SMPTE Epoch.
#[must_use]
pub fn frame_number(ts: &Smpte2059Timestamp, fps: Rational) -> u128 {
    let frame_dur_ns = fps.frame_duration_ns();
    if frame_dur_ns == 0 {
        return 0;
    }
    ts.total_ns() / u128::from(frame_dur_ns)
}

/// Returns the timestamp corresponding to the start of a given frame number.
///
/// Returns `None` if the resulting timestamp would overflow a `u64` seconds.
#[must_use]
pub fn frame_to_timestamp(frame: u128, fps: Rational) -> Option<Smpte2059Timestamp> {
    let frame_dur_ns = fps.frame_duration_ns();
    if frame_dur_ns == 0 {
        return None;
    }
    let total_ns = frame.checked_mul(u128::from(frame_dur_ns))?;
    let seconds = total_ns / 1_000_000_000;
    if seconds > u128::from(u64::MAX) {
        return None;
    }
    let nanoseconds = (total_ns % 1_000_000_000) as u32;
    Some(Smpte2059Timestamp {
        seconds: seconds as u64,
        nanoseconds,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Rational tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_rational_to_f64() {
        let r = Rational::new(30_000, 1001);
        let v = r.to_f64();
        assert!((v - 29.970_029_97).abs() < 0.000_001);
    }

    #[test]
    fn test_rational_frame_duration_25fps() {
        let fps = Rational::new(25, 1);
        // 1_000_000_000 / 25 = 40_000_000 ns
        assert_eq!(fps.frame_duration_ns(), 40_000_000);
    }

    #[test]
    fn test_rational_frame_duration_2997() {
        let fps = Rational::new(30_000, 1001);
        // 1e9 × 1001 / 30000 = 33_366_666.66... → floor = 33_366_666
        assert_eq!(fps.frame_duration_ns(), 33_366_666);
    }

    #[test]
    fn test_rational_zero_den_safe() {
        let r = Rational::new(30, 0);
        assert_eq!(r.den, 1);
        assert_eq!(r.frame_duration_ns(), 1_000_000_000 / 1);
    }

    // -----------------------------------------------------------------------
    // Smpte2059Profile tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_default_profile_values() {
        let p = Smpte2059Profile::default_profile();
        assert_eq!(p.domain, 127);
        assert_eq!(p.priority1, 128);
        assert_eq!(p.priority2, 128);
        assert_eq!(p.announce_interval_log2, 0);
    }

    #[test]
    fn test_profile_validate_ok() {
        let p = Smpte2059Profile::default_profile();
        assert!(p.validate().is_ok());
    }

    #[test]
    fn test_profile_validate_bad_domain() {
        let p = Smpte2059Profile {
            domain: 200,
            ..Smpte2059Profile::default_profile()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_announce_interval_ns() {
        let p = Smpte2059Profile::default_profile(); // log2=0 → 1 s
        assert_eq!(p.announce_interval_ns(), 1_000_000_000);
    }

    #[test]
    fn test_sync_interval_ns() {
        let p = Smpte2059Profile::default_profile(); // log2=−3 → 125 ms
        assert_eq!(p.sync_interval_ns(), 125_000_000);
    }

    #[test]
    fn test_announce_timeout_ns() {
        let p = Smpte2059Profile::default_profile(); // 3 × 1 s = 3 s
        assert_eq!(p.announce_timeout_ns(), 3_000_000_000);
    }

    // -----------------------------------------------------------------------
    // Smpte2059Timestamp tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_from_ptp_identity() {
        let ts = Smpte2059Timestamp::from_ptp(1_000_000, 500_000_000);
        assert_eq!(ts.seconds, 1_000_000);
        assert_eq!(ts.nanoseconds, 500_000_000);
    }

    #[test]
    fn test_total_ns() {
        let ts = Smpte2059Timestamp::new(1, 500_000_000);
        assert_eq!(ts.total_ns(), 1_500_000_000u128);
    }

    #[test]
    fn test_diff_ns() {
        let a = Smpte2059Timestamp::new(10, 0);
        let b = Smpte2059Timestamp::new(9, 0);
        assert_eq!(a.diff_ns(&b), 1_000_000_000i128);
        assert_eq!(b.diff_ns(&a), -1_000_000_000i128);
    }

    #[test]
    fn test_add_ns_positive() {
        let ts = Smpte2059Timestamp::new(0, 999_000_000);
        let ts2 = ts.add_ns(1_000_000).expect("should succeed");
        assert_eq!(ts2.seconds, 1);
        assert_eq!(ts2.nanoseconds, 0);
    }

    #[test]
    fn test_add_ns_negative() {
        let ts = Smpte2059Timestamp::new(1, 0);
        let ts2 = ts.add_ns(-500_000_000).expect("should succeed");
        assert_eq!(ts2.seconds, 0);
        assert_eq!(ts2.nanoseconds, 500_000_000);
    }

    #[test]
    fn test_add_ns_before_epoch_returns_none() {
        let ts = Smpte2059Timestamp::new(0, 0);
        assert!(ts.add_ns(-1).is_none());
    }

    // -----------------------------------------------------------------------
    // align_to_frame_rate tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_align_25fps_exact() {
        // Timestamp exactly at a 25 fps frame boundary (frame 100 = 4 000 000 000 ns)
        let ts = Smpte2059Timestamp::new(4, 0); // 4.000000000 s
        let fps = Rational::new(25, 1); // 40 ms per frame
        let aligned = align_to_frame_rate(&ts, fps);
        assert_eq!(aligned, ts);
    }

    #[test]
    fn test_align_25fps_snaps_forward() {
        let fps = Rational::new(25, 1); // 40_000_000 ns per frame
                                        // Slightly past frame 100 (4 s) but closer to frame 101 (4.04 s)
        let ts = Smpte2059Timestamp::new(4, 30_000_000); // 4.030 s
        let aligned = align_to_frame_rate(&ts, fps);
        // Closest boundary: frame 101 at 4.040 s
        assert_eq!(aligned.seconds, 4);
        assert_eq!(aligned.nanoseconds, 40_000_000);
    }

    #[test]
    fn test_align_25fps_snaps_backward() {
        let fps = Rational::new(25, 1);
        // Slightly past frame 100, closer to frame 100 start (4.000 s)
        let ts = Smpte2059Timestamp::new(4, 5_000_000); // 4.005 s
        let aligned = align_to_frame_rate(&ts, fps);
        assert_eq!(aligned.seconds, 4);
        assert_eq!(aligned.nanoseconds, 0);
    }

    #[test]
    fn test_frame_number_25fps() {
        let fps = Rational::new(25, 1);
        let ts = Smpte2059Timestamp::new(1, 0); // second 1 = frame 25
        assert_eq!(frame_number(&ts, fps), 25);
    }

    #[test]
    fn test_frame_to_timestamp_25fps() {
        let fps = Rational::new(25, 1);
        let ts = frame_to_timestamp(25, fps).expect("should succeed");
        assert_eq!(ts.seconds, 1);
        assert_eq!(ts.nanoseconds, 0);
    }

    #[test]
    fn test_frame_to_timestamp_zero_fps_returns_none() {
        let fps = Rational::new(0, 1);
        assert!(frame_to_timestamp(1, fps).is_none());
    }

    #[test]
    fn test_display_timestamp() {
        let ts = Smpte2059Timestamp::new(12, 34);
        let s = ts.to_string();
        assert!(s.starts_with("12."), "expected seconds prefix, got {s}");
    }
}
