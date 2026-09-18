//! Clock quality monitoring: MTIE and TDEV computation.
//!
//! Implements ITU-T G.8260 / G.8262 metrics for assessing clock performance:
//!
//! * **MTIE** (Maximum Time Interval Error) — the maximum peak-to-peak range
//!   of the phase deviation within an observation window of length τ.
//! * **TDEV** (Time Deviation) — a measure of phase noise spectral density
//!   based on the mean-squared second difference of the phase.
//!
//! Both metrics are used to classify clocks into ITU-T categories.

/// ITU-T clock quality class based on MTIE / TDEV thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ClockQualityClass {
    /// ITU-T G.811 — primary reference clock (PRC / PRTC).
    /// MTIE ≤ 100 ns over any 1 s window.
    G811,
    /// ITU-T G.812 — transit node clock (SSU/SETS).
    /// MTIE ≤ 1 µs over any 1 s window.
    G812,
    /// Synchronous Ethernet (SyncE / ITU-T G.8262).
    /// MTIE ≤ 100 µs over any 1 s window.
    SyncE,
    /// IEEE 1588 PTP slave / telecom profile (ITU-T G.8273).
    /// MTIE ≤ 1 ms over any 1 s window.
    Ptp,
    /// Does not meet any of the above thresholds.
    Unclassified,
}

impl ClockQualityClass {
    /// Returns the MTIE threshold (in nanoseconds) for this class at τ = 1 s.
    #[must_use]
    pub const fn mtie_threshold_ns(&self) -> f64 {
        match self {
            Self::G811 => 100.0,
            Self::G812 => 1_000.0,
            Self::SyncE => 100_000.0,
            Self::Ptp => 1_000_000.0,
            Self::Unclassified => f64::INFINITY,
        }
    }

    /// Returns a human-readable description.
    #[must_use]
    pub const fn description(&self) -> &'static str {
        match self {
            Self::G811 => "G.811 Primary Reference Clock (MTIE ≤ 100 ns)",
            Self::G812 => "G.812 Transit Node Clock (MTIE ≤ 1 µs)",
            Self::SyncE => "SyncE / G.8262 (MTIE ≤ 100 µs)",
            Self::Ptp => "PTP / G.8273 (MTIE ≤ 1 ms)",
            Self::Unclassified => "Unclassified (exceeds PTP threshold)",
        }
    }

    /// Classify based on an observed MTIE value (nanoseconds).
    #[must_use]
    pub fn from_mtie_ns(mtie_ns: f64) -> Self {
        if mtie_ns <= 100.0 {
            Self::G811
        } else if mtie_ns <= 1_000.0 {
            Self::G812
        } else if mtie_ns <= 100_000.0 {
            Self::SyncE
        } else if mtie_ns <= 1_000_000.0 {
            Self::Ptp
        } else {
            Self::Unclassified
        }
    }
}

/// A complete clock quality report for one observation period.
#[derive(Debug, Clone)]
pub struct ClockQualityReport {
    /// Maximum Time Interval Error in nanoseconds.
    pub mtie_ns: f64,
    /// Time Deviation in nanoseconds.
    pub tdev_ns: f64,
    /// Derived quality class based on `mtie_ns`.
    pub quality_class: ClockQualityClass,
    /// Number of phase samples used.
    pub sample_count: usize,
    /// Observation window length τ (in sample-interval units).
    pub tau: usize,
}

/// Computes MTIE and TDEV from a phase time series.
///
/// # MTIE definition
/// For each window of length `τ+1` samples, MTIE(τ) is:
/// ```text
/// MTIE(τ) = max_over_windows( max(x[i..i+τ]) − min(x[i..i+τ]) )
/// ```
///
/// # TDEV definition (simplified, per ITU-T G.8260)
/// ```text
/// TDEV²(n) = 1/(6·N) × Σ_i (x[i+2n] − 2·x[i+n] + x[i])²
/// ```
/// where n is the averaging factor and N is the count of complete triples.
pub struct ClockQualityMonitor {
    /// Accumulated phase deviation samples (nanoseconds).
    samples: Vec<f64>,
    /// Maximum number of samples to retain.
    max_samples: usize,
    /// Default observation window τ for report generation (sample units).
    default_tau: usize,
}

impl ClockQualityMonitor {
    /// Creates a new monitor.
    ///
    /// # Arguments
    /// * `max_samples` — ring-buffer capacity for phase samples.
    /// * `default_tau` — default window length τ used in `report()`.
    #[must_use]
    pub fn new(max_samples: usize, default_tau: usize) -> Self {
        Self {
            samples: Vec::with_capacity(max_samples.min(65_536)),
            max_samples: max_samples.max(3),
            default_tau: default_tau.max(1),
        }
    }

    /// Creates a monitor suitable for broadcast-grade PTP monitoring.
    ///
    /// 1 000 samples, τ = 10 s (assuming 1 Hz sample rate).
    #[must_use]
    pub fn broadcast_grade() -> Self {
        Self::new(1_000, 10)
    }

    /// Adds a new phase deviation sample (nanoseconds).
    pub fn add_sample(&mut self, phase_ns: f64) {
        if self.samples.len() >= self.max_samples {
            self.samples.remove(0);
        }
        self.samples.push(phase_ns);
    }

    /// Returns the current number of stored samples.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Computes MTIE(τ) from the stored samples.
    ///
    /// Returns `None` if there are fewer than `τ + 1` samples.
    #[must_use]
    pub fn mtie(&self, tau: usize) -> Option<f64> {
        let n = self.samples.len();
        if n < tau + 1 {
            return None;
        }
        let window_len = tau + 1;
        let mut max_range = 0.0_f64;

        for i in 0..=(n - window_len) {
            let window = &self.samples[i..i + window_len];
            let (mn, mx) = window
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
                    (lo.min(v), hi.max(v))
                });
            let range = mx - mn;
            if range > max_range {
                max_range = range;
            }
        }
        Some(max_range)
    }

    /// Computes TDEV(n) from the stored samples using the ITU-T definition.
    ///
    /// Returns `None` if there are fewer than `2·n + 1` samples.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn tdev(&self, n: usize) -> Option<f64> {
        let total = self.samples.len();
        if total < 2 * n + 1 {
            return None;
        }
        let count = total - 2 * n;
        let mut sum_sq = 0.0_f64;
        for i in 0..count {
            let diff = self.samples[i + 2 * n] - 2.0 * self.samples[i + n] + self.samples[i];
            sum_sq += diff * diff;
        }
        // TDEV = sqrt( sum_sq / (6 × count) )
        let variance = sum_sq / (6.0 * count as f64);
        Some(variance.sqrt())
    }

    /// Generates a full [`ClockQualityReport`] using `default_tau`.
    ///
    /// Returns `None` if insufficient samples are available.
    #[must_use]
    pub fn report(&self) -> Option<ClockQualityReport> {
        let tau = self.default_tau;
        let mtie_ns = self.mtie(tau)?;
        let tdev_ns = self.tdev(tau).unwrap_or(0.0);
        let quality_class = ClockQualityClass::from_mtie_ns(mtie_ns);
        Some(ClockQualityReport {
            mtie_ns,
            tdev_ns,
            quality_class,
            sample_count: self.samples.len(),
            tau,
        })
    }

    /// Generates a report for an explicit τ value.
    ///
    /// Returns `None` if insufficient samples are available.
    #[must_use]
    pub fn report_for_tau(&self, tau: usize) -> Option<ClockQualityReport> {
        let mtie_ns = self.mtie(tau)?;
        let tdev_ns = self.tdev(tau).unwrap_or(0.0);
        let quality_class = ClockQualityClass::from_mtie_ns(mtie_ns);
        Some(ClockQualityReport {
            mtie_ns,
            tdev_ns,
            quality_class,
            sample_count: self.samples.len(),
            tau,
        })
    }

    /// Clears all stored samples.
    pub fn reset(&mut self) {
        self.samples.clear();
    }

    /// Returns a reference to the raw sample buffer.
    #[must_use]
    pub fn samples(&self) -> &[f64] {
        &self.samples
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a monitor pre-loaded with a constant-phase signal (all zeros).
    fn monitor_const(n: usize) -> ClockQualityMonitor {
        let mut m = ClockQualityMonitor::new(n + 10, n / 2);
        for _ in 0..n {
            m.add_sample(0.0);
        }
        m
    }

    #[test]
    fn test_clock_quality_class_ordering() {
        assert!(ClockQualityClass::G811 < ClockQualityClass::G812);
        assert!(ClockQualityClass::G812 < ClockQualityClass::SyncE);
        assert!(ClockQualityClass::SyncE < ClockQualityClass::Ptp);
        assert!(ClockQualityClass::Ptp < ClockQualityClass::Unclassified);
    }

    #[test]
    fn test_from_mtie_thresholds() {
        assert_eq!(
            ClockQualityClass::from_mtie_ns(50.0),
            ClockQualityClass::G811
        );
        assert_eq!(
            ClockQualityClass::from_mtie_ns(100.0),
            ClockQualityClass::G811
        );
        assert_eq!(
            ClockQualityClass::from_mtie_ns(500.0),
            ClockQualityClass::G812
        );
        assert_eq!(
            ClockQualityClass::from_mtie_ns(1_000.0),
            ClockQualityClass::G812
        );
        assert_eq!(
            ClockQualityClass::from_mtie_ns(50_000.0),
            ClockQualityClass::SyncE
        );
        assert_eq!(
            ClockQualityClass::from_mtie_ns(100_000.0),
            ClockQualityClass::SyncE
        );
        assert_eq!(
            ClockQualityClass::from_mtie_ns(500_000.0),
            ClockQualityClass::Ptp
        );
        assert_eq!(
            ClockQualityClass::from_mtie_ns(1_000_000.0),
            ClockQualityClass::Ptp
        );
        assert_eq!(
            ClockQualityClass::from_mtie_ns(2_000_000.0),
            ClockQualityClass::Unclassified
        );
    }

    #[test]
    fn test_mtie_constant_signal() {
        let m = monitor_const(50);
        let mtie = m.mtie(10).expect("should have enough samples");
        assert!(mtie.abs() < 1e-12, "constant phase → MTIE = 0, got {mtie}");
    }

    #[test]
    fn test_mtie_known_value() {
        let mut m = ClockQualityMonitor::new(20, 2);
        // Phase alternates +100, -100 → peak-to-peak = 200 ns for any τ ≥ 1
        for i in 0..10 {
            m.add_sample(if i % 2 == 0 { 100.0 } else { -100.0 });
        }
        let mtie = m.mtie(1).expect("enough samples");
        assert!(
            (mtie - 200.0).abs() < 1e-9,
            "expected 200 ns MTIE, got {mtie}"
        );
    }

    #[test]
    fn test_mtie_insufficient_samples() {
        let mut m = ClockQualityMonitor::new(100, 5);
        m.add_sample(10.0);
        m.add_sample(20.0);
        // τ = 5 requires 6 samples → None
        assert!(m.mtie(5).is_none());
    }

    #[test]
    fn test_tdev_constant_signal() {
        let m = monitor_const(50);
        let tdev = m.tdev(5).expect("enough samples");
        assert!(tdev.abs() < 1e-12, "constant phase → TDEV = 0, got {tdev}");
    }

    #[test]
    fn test_tdev_positive_for_noisy_signal() {
        let mut m = ClockQualityMonitor::new(200, 5);
        // Pseudo-random phase noise
        let mut state: u64 = 0xABCD_EF01_2345_6789;
        for _ in 0..100 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let v = ((state >> 33) as f64 / (u32::MAX as f64) - 0.5) * 200.0;
            m.add_sample(v);
        }
        let tdev = m.tdev(5).expect("enough samples");
        assert!(tdev > 0.0, "noisy signal → TDEV > 0");
    }

    #[test]
    fn test_report_generation() {
        let mut m = ClockQualityMonitor::new(100, 5);
        for i in 0..20i64 {
            m.add_sample(i as f64 * 10.0 % 50.0);
        }
        let report = m.report().expect("should produce report");
        assert_eq!(report.tau, 5);
        assert!(report.sample_count > 0);
        assert!(report.mtie_ns >= 0.0);
        assert!(report.tdev_ns >= 0.0);
    }

    #[test]
    fn test_report_for_tau() {
        let mut m = ClockQualityMonitor::new(100, 1);
        for v in [10.0_f64, 20.0, 30.0, 40.0, 50.0] {
            m.add_sample(v);
        }
        let r = m.report_for_tau(2).expect("should produce report");
        assert_eq!(r.tau, 2);
    }

    #[test]
    fn test_sample_eviction_respects_max() {
        let mut m = ClockQualityMonitor::new(5, 1);
        for i in 0..10 {
            m.add_sample(i as f64);
        }
        assert_eq!(m.sample_count(), 5);
        // Most recent 5 samples: 5,6,7,8,9
        assert!((m.samples()[4] - 9.0).abs() < 1e-12);
    }

    #[test]
    fn test_reset_clears_samples() {
        let mut m = monitor_const(20);
        assert_eq!(m.sample_count(), 20);
        m.reset();
        assert_eq!(m.sample_count(), 0);
        assert!(m.mtie(1).is_none());
    }

    #[test]
    fn test_quality_class_descriptions() {
        assert!(ClockQualityClass::G811.description().contains("G.811"));
        assert!(ClockQualityClass::Ptp.description().contains("PTP"));
        assert!(ClockQualityClass::Unclassified
            .description()
            .contains("Unclassified"));
    }

    #[test]
    fn test_mtie_threshold_values() {
        assert!((ClockQualityClass::G811.mtie_threshold_ns() - 100.0).abs() < 1e-9);
        assert!((ClockQualityClass::G812.mtie_threshold_ns() - 1_000.0).abs() < 1e-9);
        assert!((ClockQualityClass::SyncE.mtie_threshold_ns() - 100_000.0).abs() < 1e-9);
        assert!((ClockQualityClass::Ptp.mtie_threshold_ns() - 1_000_000.0).abs() < 1e-9);
        assert!(ClockQualityClass::Unclassified
            .mtie_threshold_ns()
            .is_infinite());
    }
}
