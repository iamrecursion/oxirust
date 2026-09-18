//! NTP clock filter, stratum model, and client state for simulation.

/// An NTP-style timestamp with 32-bit seconds and 32-bit sub-second fraction.
///
/// The 32-bit fraction represents sub-seconds: `fraction / 2^32` seconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NtpFilterTimestamp {
    /// Seconds since the NTP epoch (1 January 1900).
    pub seconds: u64,
    /// Sub-second fraction (units of `2^-32` seconds).
    pub fraction: u32,
}

impl NtpFilterTimestamp {
    /// Construct a new `NtpFilterTimestamp`.
    #[must_use]
    pub fn new(seconds: u64, fraction: u32) -> Self {
        Self { seconds, fraction }
    }

    /// Convert to whole milliseconds (truncating sub-millisecond precision).
    #[must_use]
    pub fn to_ms(&self) -> u64 {
        self.seconds * 1_000 + u64::from(self.fraction) * 1_000 / u64::from(u32::MAX)
    }

    /// Construct from a millisecond value.
    #[must_use]
    pub fn from_ms(ms: u64) -> Self {
        let seconds = ms / 1_000;
        let rem_ms = ms % 1_000;
        // fraction = rem_ms * 2^32 / 1000
        let fraction = (rem_ms * u64::from(u32::MAX) / 1_000) as u32;
        Self { seconds, fraction }
    }

    /// Signed difference between `self` and `other` expressed in milliseconds.
    ///
    /// Positive means `self` is later than `other`.
    #[must_use]
    pub fn diff_ms(&self, other: &Self) -> f64 {
        let self_ms = self.seconds as f64 * 1_000.0
            + f64::from(self.fraction) / f64::from(u32::MAX) * 1_000.0;
        let other_ms = other.seconds as f64 * 1_000.0
            + f64::from(other.fraction) / f64::from(u32::MAX) * 1_000.0;
        self_ms - other_ms
    }
}

/// NTP stratum level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NtpStratum {
    /// Stratum 1 – directly connected to a reference clock (GPS, atomic).
    Primary,
    /// Stratum 2-15 – synchronized via network to a Primary or Secondary source.
    Secondary(u8),
    /// Stratum 0 or 16 – not synchronized.
    Unsynchronized,
}

impl NtpStratum {
    /// Returns the numeric stratum level (1 = Primary, 2-15 = Secondary, 16 = Unsynchronized).
    #[must_use]
    pub fn level(&self) -> u8 {
        match self {
            Self::Primary => 1,
            Self::Secondary(n) => (*n).clamp(2, 15),
            Self::Unsynchronized => 16,
        }
    }

    /// Returns `true` for Primary and Secondary strata (1-15).
    #[must_use]
    pub fn is_valid(&self) -> bool {
        matches!(self, Self::Primary | Self::Secondary(_))
    }
}

/// NTP clock filter: maintains a sliding window of offset/delay samples.
pub struct NtpClockFilter {
    /// Circular buffer of offset samples (milliseconds).
    pub offsets: Vec<f64>,
    /// Circular buffer of delay samples (milliseconds).
    pub delays: Vec<f64>,
}

impl NtpClockFilter {
    /// Create an empty clock filter.
    #[must_use]
    pub fn new() -> Self {
        Self {
            offsets: Vec::new(),
            delays: Vec::new(),
        }
    }

    /// Add a new offset/delay sample pair.
    pub fn add_sample(&mut self, offset: f64, delay: f64) {
        self.offsets.push(offset);
        self.delays.push(delay);
    }

    /// Return the offset associated with the minimum-delay sample.
    ///
    /// The NTP spec selects the sample with the lowest delay as the best
    /// estimate because it has the least measurement uncertainty.
    #[must_use]
    pub fn best_offset(&self) -> Option<f64> {
        if self.delays.is_empty() {
            return None;
        }
        let (min_idx, _) = self
            .delays
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))?;
        self.offsets.get(min_idx).copied()
    }

    /// Compute the RMS dispersion of offset samples around their mean.
    #[must_use]
    pub fn dispersion(&self) -> f64 {
        if self.offsets.is_empty() {
            return 0.0;
        }
        let mean = self.offsets.iter().sum::<f64>() / self.offsets.len() as f64;
        let variance = self
            .offsets
            .iter()
            .map(|&o| (o - mean).powi(2))
            .sum::<f64>()
            / self.offsets.len() as f64;
        variance.sqrt()
    }
}

impl Default for NtpClockFilter {
    fn default() -> Self {
        Self::new()
    }
}

/// Simplified NTP client state tracking offset, delay, and jitter.
pub struct NtpClientState {
    /// Stratum of this client.
    pub stratum: NtpStratum,
    /// Current best offset estimate in milliseconds.
    pub offset_ms: f64,
    /// Current round-trip delay in milliseconds.
    pub delay_ms: f64,
    /// Clock jitter estimate in milliseconds.
    pub jitter_ms: f64,
}

impl NtpClientState {
    /// Create a new `NtpClientState` in an unsynchronized state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            stratum: NtpStratum::Unsynchronized,
            offset_ms: 0.0,
            delay_ms: 0.0,
            jitter_ms: 0.0,
        }
    }

    /// Process a set of NTP timestamps and update offset, delay, and jitter.
    ///
    /// | Symbol | Meaning |
    /// |--------|---------|
    /// | t1     | Client transmit time (ms) |
    /// | t2     | Server receive time (ms) |
    /// | t3     | Server transmit time (ms) |
    /// | t4     | Client receive time (ms) |
    ///
    /// - `delay = (t4 - t1) - (t3 - t2)`
    /// - `offset = ((t2 - t1) + (t3 - t4)) / 2`
    pub fn update(&mut self, t1: u64, t2: u64, t3: u64, t4: u64) {
        let t1 = t1 as f64;
        let t2 = t2 as f64;
        let t3 = t3 as f64;
        let t4 = t4 as f64;

        let new_delay = (t4 - t1) - (t3 - t2);
        let new_offset = ((t2 - t1) + (t3 - t4)) / 2.0;

        // Exponential moving average for jitter
        let prev_offset = self.offset_ms;
        self.delay_ms = new_delay;
        self.offset_ms = new_offset;
        self.jitter_ms = (self.jitter_ms * 0.875 + (new_offset - prev_offset).abs() * 0.125).abs();

        if self.stratum == NtpStratum::Unsynchronized {
            self.stratum = NtpStratum::Secondary(2);
        }
    }

    /// Returns `true` if the client is synchronized to a valid stratum.
    #[must_use]
    pub fn is_synchronized(&self) -> bool {
        self.stratum.is_valid()
    }
}

impl Default for NtpClientState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ntp_filter_timestamp_to_ms() {
        let ts = NtpFilterTimestamp::new(1, 0);
        assert_eq!(ts.to_ms(), 1000);
    }

    #[test]
    fn test_ntp_filter_timestamp_from_ms() {
        let ts = NtpFilterTimestamp::from_ms(2000);
        assert_eq!(ts.seconds, 2);
    }

    #[test]
    fn test_ntp_filter_timestamp_round_trip() {
        let ms = 1500u64;
        let ts = NtpFilterTimestamp::from_ms(ms);
        // Round-trip to ms should be very close (within 1 ms due to integer rounding)
        let back = ts.to_ms();
        assert!((back as i64 - ms as i64).abs() <= 1);
    }

    #[test]
    fn test_ntp_filter_timestamp_diff_ms_positive() {
        let a = NtpFilterTimestamp::new(2, 0);
        let b = NtpFilterTimestamp::new(1, 0);
        let diff = a.diff_ms(&b);
        assert!((diff - 1000.0).abs() < 1.0);
    }

    #[test]
    fn test_ntp_filter_timestamp_diff_ms_negative() {
        let a = NtpFilterTimestamp::new(1, 0);
        let b = NtpFilterTimestamp::new(2, 0);
        let diff = a.diff_ms(&b);
        assert!(diff < 0.0);
    }

    #[test]
    fn test_ntp_stratum_level() {
        assert_eq!(NtpStratum::Primary.level(), 1);
        assert_eq!(NtpStratum::Secondary(4).level(), 4);
        assert_eq!(NtpStratum::Unsynchronized.level(), 16);
    }

    #[test]
    fn test_ntp_stratum_is_valid() {
        assert!(NtpStratum::Primary.is_valid());
        assert!(NtpStratum::Secondary(3).is_valid());
        assert!(!NtpStratum::Unsynchronized.is_valid());
    }

    #[test]
    fn test_ntp_clock_filter_empty() {
        let filter = NtpClockFilter::new();
        assert!(filter.best_offset().is_none());
        assert_eq!(filter.dispersion(), 0.0);
    }

    #[test]
    fn test_ntp_clock_filter_best_offset_min_delay() {
        let mut filter = NtpClockFilter::new();
        filter.add_sample(10.0, 5.0); // high delay
        filter.add_sample(20.0, 2.0); // low delay → this should be selected
        filter.add_sample(15.0, 8.0); // high delay
        let best = filter.best_offset().expect("should succeed in test");
        assert!((best - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_ntp_clock_filter_dispersion_zero() {
        let mut filter = NtpClockFilter::new();
        filter.add_sample(5.0, 1.0);
        filter.add_sample(5.0, 2.0);
        // All offsets equal → zero dispersion
        assert!(filter.dispersion() < 0.01);
    }

    #[test]
    fn test_ntp_client_state_initial() {
        let client = NtpClientState::new();
        assert!(!client.is_synchronized());
        assert_eq!(client.stratum, NtpStratum::Unsynchronized);
    }

    #[test]
    fn test_ntp_client_state_update_offset() {
        // t1=1000, t2=1020, t3=1025, t4=1050
        // delay = (1050-1000) - (1025-1020) = 50 - 5 = 45
        // offset = ((1020-1000) + (1025-1050)) / 2 = (20 - 25) / 2 = -2.5
        let mut client = NtpClientState::new();
        client.update(1000, 1020, 1025, 1050);
        assert!((client.delay_ms - 45.0).abs() < 0.01);
        assert!((client.offset_ms - (-2.5)).abs() < 0.01);
        assert!(client.is_synchronized());
    }

    #[test]
    fn test_ntp_client_state_default() {
        let client = NtpClientState::default();
        assert_eq!(client.stratum, NtpStratum::Unsynchronized);
    }
}
