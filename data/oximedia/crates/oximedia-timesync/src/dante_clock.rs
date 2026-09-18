//! Dante audio clock domain management.
//!
//! Dante (by Audinate) is a proprietary audio over IP protocol widely used
//! in professional audio. This module models Dante clock domains and
//! provides clock mismatch detection.

/// Clock source types supported by Dante devices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum ClockSource {
    /// Internal crystal oscillator
    Internal,
    /// External clock input (word clock or similar)
    External,
    /// Word clock (BNC connector, standard in professional audio)
    WordClock,
    /// AES3 (AES/EBU) digital audio clock
    Aes3,
    /// S/PDIF (Sony/Philips Digital Interface)
    Spdif,
    /// Network clock (PTP/IEEE 1588)
    Network,
}

impl ClockSource {
    /// Returns the display name for this clock source.
    #[must_use]
    pub fn display_name(&self) -> &'static str {
        match self {
            ClockSource::Internal => "Internal",
            ClockSource::External => "External",
            ClockSource::WordClock => "Word Clock",
            ClockSource::Aes3 => "AES3",
            ClockSource::Spdif => "S/PDIF",
            ClockSource::Network => "Network (PTP)",
        }
    }
}

/// Dante clock domain descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub struct DanteClockDomain {
    /// Domain identifier (0-255)
    pub domain_id: u8,
    /// Name of the master device controlling this domain
    pub master_device: String,
    /// Source of the master clock
    pub clock_source: ClockSource,
}

impl DanteClockDomain {
    /// Create a new Dante clock domain.
    #[must_use]
    pub fn new(domain_id: u8, master_device: String, clock_source: ClockSource) -> Self {
        Self {
            domain_id,
            master_device,
            clock_source,
        }
    }
}

/// Pull factor for sample rate adjustment.
///
/// Pull factors are used to convert between 24/25 fps film and 30 fps NTSC
/// video standards. They adjust the sample rate by a fractional amount.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum PullFactor {
    /// No pull, exact sample rate
    None,
    /// -0.1% pull-down (e.g., 47952 Hz from 48000 Hz)
    PullDown001,
    /// +0.1% pull-up (e.g., 48048 Hz from 48000 Hz)
    PullUp001,
    /// -4% pull-down (rare, used for 25→24 fps conversion)
    PullDown04,
    /// +4% pull-up (rare, used for 24→25 fps conversion)
    PullUp04,
}

impl PullFactor {
    /// Get the rate multiplier for this pull factor.
    ///
    /// Returns the exact ratio by which the nominal sample rate is multiplied.
    #[must_use]
    pub fn rate_multiplier(&self) -> f64 {
        match self {
            PullFactor::None => 1.0,
            PullFactor::PullDown001 => 1000.0 / 1001.0, // 0.999000999...
            PullFactor::PullUp001 => 1001.0 / 1000.0,   // 1.001
            PullFactor::PullDown04 => 0.96,
            PullFactor::PullUp04 => 1.04,
        }
    }

    /// Get the pull factor as a signed PPM (parts per million) offset.
    #[must_use]
    pub fn ppm(&self) -> f64 {
        (self.rate_multiplier() - 1.0) * 1_000_000.0
    }
}

/// Dante clock configuration for a device.
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub struct DanteClock {
    /// Clock domain this device belongs to
    pub domain: DanteClockDomain,
    /// Nominal sample rate in Hz
    pub sample_rate: u32,
    /// Pull factor applied to the sample rate
    pub pull_factor: PullFactor,
}

impl DanteClock {
    /// Create a new Dante clock configuration.
    #[must_use]
    pub fn new(domain: DanteClockDomain, sample_rate: u32, pull_factor: PullFactor) -> Self {
        Self {
            domain,
            sample_rate,
            pull_factor,
        }
    }

    /// Get the actual (pulled) sample rate.
    #[must_use]
    pub fn actual_rate(&self) -> f64 {
        DanteSampleRate::actual_rate(self.sample_rate, self.pull_factor)
    }
}

/// Sample rate utilities for Dante clocks.
#[allow(dead_code)]
pub struct DanteSampleRate;

impl DanteSampleRate {
    /// Calculate the actual sample rate after applying a pull factor.
    #[must_use]
    pub fn actual_rate(sample_rate: u32, pull: PullFactor) -> f64 {
        f64::from(sample_rate) * pull.rate_multiplier()
    }

    /// Check if two sample rates are compatible (same nominal rate).
    #[must_use]
    pub fn are_compatible(rate_a: u32, rate_b: u32) -> bool {
        // Compatible if they share the same base family (44.1k or 48k family)
        let family_a = if rate_a % 44100 == 0 {
            44100u32
        } else {
            48000u32
        };
        let family_b = if rate_b % 44100 == 0 {
            44100u32
        } else {
            48000u32
        };
        family_a == family_b
    }
}

/// Description of a detected clock mismatch between two Dante domains.
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub struct ClockMismatch {
    /// Frequency difference in parts per million
    pub ppm_difference: f64,
    /// Whether the clocks will drift apart over time
    pub will_drift: bool,
    /// Rate of drift in seconds per hour
    pub drift_rate_secs_per_hour: f64,
}

impl ClockMismatch {
    /// Create a new clock mismatch descriptor.
    #[must_use]
    pub fn new(ppm_difference: f64) -> Self {
        // Drift rate: ppm_difference ppm = ppm_difference us per second
        // Per hour: ppm_difference * 3600 microseconds = ppm_difference * 3600 / 1_000_000 seconds
        let drift_rate_secs_per_hour = ppm_difference.abs() * 3600.0 / 1_000_000.0;
        let will_drift = ppm_difference.abs() > 1.0; // More than 1 ppm is audible

        Self {
            ppm_difference,
            will_drift,
            drift_rate_secs_per_hour,
        }
    }

    /// Check if the mismatch is severe enough to cause audible artifacts.
    #[must_use]
    pub fn is_audible(&self) -> bool {
        self.ppm_difference.abs() > 10.0
    }
}

/// Detector for clock domain mismatches between Dante devices.
#[allow(dead_code)]
pub struct ClockMismatchDetector;

impl ClockMismatchDetector {
    /// Detect any clock mismatch between two Dante clocks.
    ///
    /// Returns `Some(ClockMismatch)` if the clocks are running at different
    /// effective rates, or `None` if they are synchronized.
    #[must_use]
    pub fn detect(domain_a: &DanteClock, domain_b: &DanteClock) -> Option<ClockMismatch> {
        let rate_a = domain_a.actual_rate();
        let rate_b = domain_b.actual_rate();

        if rate_a == 0.0 || rate_b == 0.0 {
            return None;
        }

        // Calculate PPM difference
        let ppm = (rate_a - rate_b) / rate_b * 1_000_000.0;

        // No mismatch if within 0.1 PPM (essentially identical clocks)
        if ppm.abs() < 0.1 {
            return None;
        }

        Some(ClockMismatch::new(ppm))
    }

    /// Check if two clocks are in the same domain (no mismatch possible).
    #[must_use]
    pub fn same_domain(domain_a: &DanteClock, domain_b: &DanteClock) -> bool {
        domain_a.domain.domain_id == domain_b.domain.domain_id
            && domain_a.domain.master_device == domain_b.domain.master_device
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_domain(id: u8) -> DanteClockDomain {
        DanteClockDomain::new(id, format!("device_{id}"), ClockSource::Network)
    }

    #[test]
    fn test_clock_source_display() {
        assert_eq!(ClockSource::Internal.display_name(), "Internal");
        assert_eq!(ClockSource::WordClock.display_name(), "Word Clock");
        assert_eq!(ClockSource::Network.display_name(), "Network (PTP)");
    }

    #[test]
    fn test_pull_factor_none() {
        assert!((PullFactor::None.rate_multiplier() - 1.0).abs() < f64::EPSILON);
        assert!(PullFactor::None.ppm().abs() < 0.001);
    }

    #[test]
    fn test_pull_factor_down_001() {
        let r = PullFactor::PullDown001.rate_multiplier();
        // Should be 1000/1001 ≈ 0.999000999
        assert!((r - 1000.0 / 1001.0).abs() < 1e-10);
        let ppm = PullFactor::PullDown001.ppm();
        // Should be approximately -999 ppm
        assert!(ppm < -900.0 && ppm > -1100.0);
    }

    #[test]
    fn test_pull_factor_up_001() {
        let r = PullFactor::PullUp001.rate_multiplier();
        assert!((r - 1.001).abs() < 1e-10);
        let ppm = PullFactor::PullUp001.ppm();
        assert!((ppm - 1000.0).abs() < 1.0);
    }

    #[test]
    fn test_dante_sample_rate_actual() {
        let rate = DanteSampleRate::actual_rate(48000, PullFactor::None);
        assert!((rate - 48000.0).abs() < 0.001);

        let rate_down = DanteSampleRate::actual_rate(48000, PullFactor::PullDown001);
        // Should be 48000 * 1000/1001 ≈ 47952.048
        assert!((rate_down - 47952.048).abs() < 0.1);
    }

    #[test]
    fn test_dante_sample_rate_compatible() {
        assert!(DanteSampleRate::are_compatible(48000, 48000));
        assert!(DanteSampleRate::are_compatible(48000, 96000));
        assert!(!DanteSampleRate::are_compatible(48000, 44100));
    }

    #[test]
    fn test_clock_mismatch_creation() {
        let mismatch = ClockMismatch::new(100.0);
        assert!((mismatch.ppm_difference - 100.0).abs() < f64::EPSILON);
        assert!(mismatch.will_drift);
        // drift rate: 100 * 3600 / 1_000_000 = 0.36 s/hour
        assert!((mismatch.drift_rate_secs_per_hour - 0.36).abs() < 0.001);
    }

    #[test]
    fn test_clock_mismatch_no_drift() {
        let mismatch = ClockMismatch::new(0.5);
        assert!(!mismatch.will_drift); // Less than 1 PPM
    }

    #[test]
    fn test_clock_mismatch_audible() {
        assert!(ClockMismatch::new(100.0).is_audible());
        assert!(!ClockMismatch::new(5.0).is_audible());
    }

    #[test]
    fn test_mismatch_detector_same_rate() {
        let clock_a = DanteClock::new(make_domain(0), 48000, PullFactor::None);
        let clock_b = DanteClock::new(make_domain(1), 48000, PullFactor::None);
        // Same actual rates - no mismatch
        assert!(ClockMismatchDetector::detect(&clock_a, &clock_b).is_none());
    }

    #[test]
    fn test_mismatch_detector_pull_mismatch() {
        let clock_a = DanteClock::new(make_domain(0), 48000, PullFactor::None);
        let clock_b = DanteClock::new(make_domain(1), 48000, PullFactor::PullDown001);
        let mismatch = ClockMismatchDetector::detect(&clock_a, &clock_b);
        assert!(mismatch.is_some());
        let m = mismatch.expect("should succeed in test");
        // About 999 ppm difference
        assert!(m.ppm_difference > 900.0);
        assert!(m.will_drift);
    }

    #[test]
    fn test_mismatch_detector_same_domain() {
        let domain = DanteClockDomain::new(0, "master_device".to_string(), ClockSource::Network);
        let clock_a = DanteClock::new(domain.clone(), 48000, PullFactor::None);
        let clock_b = DanteClock::new(domain, 48000, PullFactor::None);
        assert!(ClockMismatchDetector::same_domain(&clock_a, &clock_b));
    }

    #[test]
    fn test_mismatch_detector_different_domain() {
        let clock_a = DanteClock::new(make_domain(0), 48000, PullFactor::None);
        let clock_b = DanteClock::new(make_domain(1), 48000, PullFactor::None);
        assert!(!ClockMismatchDetector::same_domain(&clock_a, &clock_b));
    }
}
