//! Watermark robustness testing — attack simulation and threshold reporting.
#![allow(dead_code)]

/// Common signal-processing attacks applied to watermarked content.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttackType {
    /// Spatial rescaling (e.g., 50 % downscale then upscale).
    Resize,
    /// Lossy compression (e.g., JPEG/MP3 at a low quality factor).
    Compress,
    /// Region-of-interest crop (removes peripheral pixels/samples).
    Crop,
    /// Additive white Gaussian noise at moderate SNR.
    Noise,
    /// Low-pass filtering to remove high-frequency components.
    LowPassFilter,
    /// Time-scale / pitch shifting.
    TimeScale,
}

impl AttackType {
    /// Expected SNR loss in dB after applying this attack under default parameters.
    ///
    /// Higher values mean the attack is more damaging.
    #[must_use]
    pub fn expected_snr_loss_db(&self) -> f32 {
        match self {
            AttackType::Resize => 6.0,
            AttackType::Compress => 12.0,
            AttackType::Crop => 9.0,
            AttackType::Noise => 4.0,
            AttackType::LowPassFilter => 8.0,
            AttackType::TimeScale => 5.0,
        }
    }

    /// Human-readable name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            AttackType::Resize => "resize",
            AttackType::Compress => "compress",
            AttackType::Crop => "crop",
            AttackType::Noise => "noise",
            AttackType::LowPassFilter => "low_pass_filter",
            AttackType::TimeScale => "time_scale",
        }
    }
}

/// A single robustness test result for one attack.
#[derive(Debug, Clone)]
pub struct RobustnessTest {
    /// The attack that was simulated.
    pub attack: AttackType,
    /// SNR of the attacked signal relative to the original watermarked signal (dB).
    pub snr_db: f32,
    /// Bit-error rate after watermark re-extraction (0.0 – 0.5).
    pub ber: f32,
}

impl RobustnessTest {
    /// Create a new robustness test result.
    #[must_use]
    pub fn new(attack: AttackType, snr_db: f32, ber: f32) -> Self {
        Self {
            attack,
            snr_db,
            ber: ber.clamp(0.0, 0.5),
        }
    }

    /// Simulate the attack on `signal` using a simple analytical model.
    ///
    /// Returns a `RobustnessTest` populated with model-estimated SNR and BER.
    /// This is a lightweight simulation — real tests would apply actual codecs.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn simulate_attack(attack: AttackType, signal: &[f32]) -> Self {
        let base_power: f32 = signal.iter().map(|s| s * s).sum::<f32>() / signal.len() as f32;
        let noise_scale = attack.expected_snr_loss_db() / 20.0; // dB → linear (approx)
        let noise_power = base_power * noise_scale;
        // Estimated SNR after attack.
        let snr = if noise_power > 0.0 {
            10.0 * (base_power / noise_power).log10()
        } else {
            60.0
        };
        // Crude BER model: BER ~ 0.5 * erfc(sqrt(snr_linear/2))
        // Approximated as BER = 0.5 * exp(-snr_linear / 4).
        let snr_linear = 10.0_f32.powf(snr / 10.0);
        let ber = (0.5 * (-snr_linear / 4.0).exp()).clamp(0.0, 0.5);
        Self::new(attack, snr, ber)
    }

    /// Returns `true` if BER is below the given threshold (watermark survives).
    #[must_use]
    pub fn passes_ber_threshold(&self, max_ber: f32) -> bool {
        self.ber < max_ber
    }

    /// Returns `true` if SNR loss is within the expected model range.
    #[must_use]
    pub fn snr_plausible(&self) -> bool {
        self.snr_db > -20.0 && self.snr_db < 120.0
    }
}

/// Aggregated robustness report across multiple attacks.
#[derive(Debug, Clone, Default)]
pub struct RobustnessReport {
    /// Individual test results.
    pub tests: Vec<RobustnessTest>,
    /// Required BER threshold to pass (e.g., 0.1 = 10 % max BER).
    pub ber_threshold: f32,
}

impl RobustnessReport {
    /// Create an empty report with the given BER pass threshold.
    #[must_use]
    pub fn new(ber_threshold: f32) -> Self {
        Self {
            tests: Vec::new(),
            ber_threshold: ber_threshold.clamp(0.0, 0.5),
        }
    }

    /// Add a test result to the report.
    pub fn add(&mut self, test: RobustnessTest) {
        self.tests.push(test);
    }

    /// Run all standard attacks on `signal` and populate the report.
    #[must_use]
    pub fn run_all(signal: &[f32], ber_threshold: f32) -> Self {
        let attacks = [
            AttackType::Resize,
            AttackType::Compress,
            AttackType::Crop,
            AttackType::Noise,
            AttackType::LowPassFilter,
            AttackType::TimeScale,
        ];
        let mut report = Self::new(ber_threshold);
        for attack in &attacks {
            let result = RobustnessTest::simulate_attack(*attack, signal);
            report.add(result);
        }
        report
    }

    /// Returns `true` if every test in the report passes the BER threshold.
    #[must_use]
    pub fn passes_threshold(&self) -> bool {
        self.tests
            .iter()
            .all(|t| t.passes_ber_threshold(self.ber_threshold))
    }

    /// Number of tests that failed the BER threshold.
    #[must_use]
    pub fn failure_count(&self) -> usize {
        self.tests
            .iter()
            .filter(|t| !t.passes_ber_threshold(self.ber_threshold))
            .count()
    }

    /// Mean BER across all tests.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn mean_ber(&self) -> f32 {
        if self.tests.is_empty() {
            return 0.0;
        }
        self.tests.iter().map(|t| t.ber).sum::<f32>() / self.tests.len() as f32
    }

    /// The test with the worst (highest) BER.
    #[must_use]
    pub fn worst_test(&self) -> Option<&RobustnessTest> {
        self.tests.iter().max_by(|a, b| {
            a.ber
                .partial_cmp(&b.ber)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_signal(len: usize) -> Vec<f32> {
        vec![1.0_f32; len]
    }

    #[test]
    fn test_attack_type_snr_loss_positive() {
        for attack in [
            AttackType::Resize,
            AttackType::Compress,
            AttackType::Crop,
            AttackType::Noise,
            AttackType::LowPassFilter,
            AttackType::TimeScale,
        ] {
            assert!(attack.expected_snr_loss_db() > 0.0);
        }
    }

    #[test]
    fn test_attack_type_names_non_empty() {
        assert!(!AttackType::Compress.name().is_empty());
        assert_eq!(AttackType::Noise.name(), "noise");
    }

    #[test]
    fn test_compress_more_damaging_than_noise() {
        assert!(
            AttackType::Compress.expected_snr_loss_db() > AttackType::Noise.expected_snr_loss_db()
        );
    }

    #[test]
    fn test_robustness_test_ber_clamped() {
        let t = RobustnessTest::new(AttackType::Noise, 30.0, 0.9);
        assert_eq!(t.ber, 0.5);
    }

    #[test]
    fn test_simulate_attack_snr_plausible() {
        let signal = unit_signal(1024);
        let t = RobustnessTest::simulate_attack(AttackType::Noise, &signal);
        assert!(t.snr_plausible(), "SNR={} is implausible", t.snr_db);
    }

    #[test]
    fn test_simulate_attack_ber_in_range() {
        let signal = unit_signal(1024);
        for attack in [AttackType::Resize, AttackType::Compress, AttackType::Crop] {
            let t = RobustnessTest::simulate_attack(attack, &signal);
            assert!(t.ber >= 0.0 && t.ber <= 0.5, "BER={} out of range", t.ber);
        }
    }

    #[test]
    fn test_passes_ber_threshold_true() {
        let t = RobustnessTest::new(AttackType::Noise, 40.0, 0.02);
        assert!(t.passes_ber_threshold(0.1));
    }

    #[test]
    fn test_passes_ber_threshold_false() {
        let t = RobustnessTest::new(AttackType::Compress, 15.0, 0.4);
        assert!(!t.passes_ber_threshold(0.1));
    }

    #[test]
    fn test_report_empty_passes_threshold() {
        let report = RobustnessReport::new(0.1);
        assert!(report.passes_threshold()); // vacuously true
    }

    #[test]
    fn test_report_run_all_count() {
        let signal = unit_signal(2048);
        let report = RobustnessReport::run_all(&signal, 0.2);
        assert_eq!(report.tests.len(), 6);
    }

    #[test]
    fn test_report_mean_ber_empty() {
        let report = RobustnessReport::new(0.1);
        assert_eq!(report.mean_ber(), 0.0);
    }

    #[test]
    fn test_report_mean_ber_computed() {
        let mut report = RobustnessReport::new(0.1);
        report.add(RobustnessTest::new(AttackType::Noise, 30.0, 0.1));
        report.add(RobustnessTest::new(AttackType::Resize, 20.0, 0.3));
        assert!((report.mean_ber() - 0.2).abs() < 1e-5);
    }

    #[test]
    fn test_report_failure_count() {
        let mut report = RobustnessReport::new(0.15);
        report.add(RobustnessTest::new(AttackType::Noise, 35.0, 0.05));
        report.add(RobustnessTest::new(AttackType::Compress, 18.0, 0.4));
        assert_eq!(report.failure_count(), 1);
    }

    #[test]
    fn test_report_worst_test() {
        let mut report = RobustnessReport::new(0.1);
        report.add(RobustnessTest::new(AttackType::Noise, 30.0, 0.05));
        report.add(RobustnessTest::new(AttackType::Compress, 15.0, 0.35));
        let worst = report.worst_test().expect("should succeed in test");
        assert_eq!(worst.attack, AttackType::Compress);
    }

    #[test]
    fn test_report_worst_test_empty() {
        let report = RobustnessReport::new(0.1);
        assert!(report.worst_test().is_none());
    }
}
