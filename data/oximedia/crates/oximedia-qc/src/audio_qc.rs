//! Audio quality control checks.
//!
//! This module validates audio streams against loudness, true-peak,
//! channel balance, and other broadcast/streaming specifications.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// The type of audio QC test being performed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AudioQcTest {
    /// Integrated loudness per EBU R128.
    LoudnessEbuR128,
    /// True-peak level check.
    TruePeakLimit,
    /// Left/right channel balance check.
    ChannelBalance,
    /// Sample rate validation.
    SampleRateCheck,
    /// Clipping detection.
    ClippingDetect,
    /// Silence detection.
    SilenceCheck,
}

impl AudioQcTest {
    /// Returns `true` if this test is a loudness-related check.
    #[must_use]
    pub fn is_loudness(&self) -> bool {
        matches!(self, Self::LoudnessEbuR128 | Self::TruePeakLimit)
    }
}

/// The result of a single audio QC test.
#[derive(Debug, Clone)]
pub struct AudioQcResult {
    /// The test that was run.
    pub test: AudioQcTest,
    /// Whether the test passed.
    pub passed: bool,
    /// The measured value (units depend on the test).
    pub measured_value: f32,
    /// The pass/fail threshold for this test.
    pub threshold: f32,
    /// Pre-computed margin (positive = headroom, negative = exceeded threshold).
    pub margin: f32,
}

impl AudioQcResult {
    /// Creates a new result, computing `margin` automatically.
    #[must_use]
    pub fn new(test: AudioQcTest, passed: bool, measured_value: f32, threshold: f32) -> Self {
        let margin = threshold - measured_value;
        Self {
            test,
            passed,
            measured_value,
            threshold,
            margin,
        }
    }

    /// Returns the difference between the threshold and the measured value.
    ///
    /// A positive value indicates headroom; a negative value indicates excess.
    #[must_use]
    pub fn margin_from_threshold(&self) -> f32 {
        self.threshold - self.measured_value
    }

    /// Returns `true` if the test failed AND the measured value exceeds `threshold` by
    /// more than the given critical margin.
    #[must_use]
    pub fn is_critical_fail(&self, critical_margin: f32) -> bool {
        !self.passed && (self.measured_value - self.threshold).abs() > critical_margin
    }
}

/// Audio QC specification defining target values and limits.
#[derive(Debug, Clone)]
pub struct AudioQcSpec {
    /// Target integrated loudness in LUFS.
    pub target_lufs: f32,
    /// Maximum true-peak level in dBFS.
    pub max_tp_dbfs: f32,
    /// Maximum loudness range (LRA) in LU.
    pub max_lra: f32,
    /// Required sample rate in Hz.
    pub sample_rate: u32,
    /// Required channel count.
    pub channels: u8,
}

impl AudioQcSpec {
    /// Returns an `AudioQcSpec` compliant with EBU R128 broadcast requirements.
    ///
    /// - Target: -23.0 LUFS integrated
    /// - True peak: -1.0 dBFS
    /// - LRA: 20 LU max
    /// - Sample rate: 48000 Hz
    /// - Channels: 2
    #[must_use]
    pub fn ebu_r128_broadcast() -> Self {
        Self {
            target_lufs: -23.0,
            max_tp_dbfs: -1.0,
            max_lra: 20.0,
            sample_rate: 48000,
            channels: 2,
        }
    }

    /// Returns an `AudioQcSpec` compliant with ATSC A/85 requirements.
    ///
    /// - Target: -24.0 LUFS integrated
    /// - True peak: -2.0 dBFS
    /// - LRA: 15 LU max
    /// - Sample rate: 48000 Hz
    /// - Channels: 2
    #[must_use]
    pub fn atsc_a85() -> Self {
        Self {
            target_lufs: -24.0,
            max_tp_dbfs: -2.0,
            max_lra: 15.0,
            sample_rate: 48000,
            channels: 2,
        }
    }
}

/// Runs audio QC checks against a given `AudioQcSpec`.
#[derive(Debug, Clone)]
pub struct AudioQcRunner {
    /// The specification to check against.
    pub spec: AudioQcSpec,
}

impl AudioQcRunner {
    /// Creates a new runner with the provided spec.
    #[must_use]
    pub fn new(spec: AudioQcSpec) -> Self {
        Self { spec }
    }

    /// Checks whether the integrated loudness is within ±1 LU of the target.
    #[must_use]
    pub fn check_loudness(&self, integrated: f32) -> AudioQcResult {
        let tolerance = 1.0_f32;
        let passed = (integrated - self.spec.target_lufs).abs() <= tolerance;
        AudioQcResult::new(
            AudioQcTest::LoudnessEbuR128,
            passed,
            integrated,
            self.spec.target_lufs,
        )
    }

    /// Checks whether the true-peak level does not exceed the maximum.
    #[must_use]
    pub fn check_true_peak(&self, tp: f32) -> AudioQcResult {
        let passed = tp <= self.spec.max_tp_dbfs;
        AudioQcResult::new(
            AudioQcTest::TruePeakLimit,
            passed,
            tp,
            self.spec.max_tp_dbfs,
        )
    }

    /// Checks whether the provided sample rate matches the spec.
    #[must_use]
    pub fn check_sample_rate(&self, rate: u32) -> AudioQcResult {
        let passed = rate == self.spec.sample_rate;
        AudioQcResult::new(
            AudioQcTest::SampleRateCheck,
            passed,
            rate as f32,
            self.spec.sample_rate as f32,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- AudioQcTest tests ---

    #[test]
    fn test_loudness_ebu_is_loudness() {
        assert!(AudioQcTest::LoudnessEbuR128.is_loudness());
    }

    #[test]
    fn test_true_peak_is_loudness() {
        assert!(AudioQcTest::TruePeakLimit.is_loudness());
    }

    #[test]
    fn test_channel_balance_not_loudness() {
        assert!(!AudioQcTest::ChannelBalance.is_loudness());
    }

    #[test]
    fn test_sample_rate_not_loudness() {
        assert!(!AudioQcTest::SampleRateCheck.is_loudness());
    }

    #[test]
    fn test_clipping_detect_not_loudness() {
        assert!(!AudioQcTest::ClippingDetect.is_loudness());
    }

    // --- AudioQcResult tests ---

    #[test]
    fn test_result_margin_positive() {
        let r = AudioQcResult::new(AudioQcTest::LoudnessEbuR128, true, -24.0, -23.0);
        // threshold - measured = -23 - (-24) = 1.0
        assert!((r.margin_from_threshold() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_result_margin_negative() {
        let r = AudioQcResult::new(AudioQcTest::LoudnessEbuR128, false, -22.0, -23.0);
        // threshold - measured = -23 - (-22) = -1.0
        assert!((r.margin_from_threshold() - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn test_result_is_critical_fail_true() {
        let r = AudioQcResult::new(AudioQcTest::TruePeakLimit, false, 0.0, -1.0);
        // |0.0 - (-1.0)| = 1.0 > 0.5 => critical
        assert!(r.is_critical_fail(0.5));
    }

    #[test]
    fn test_result_is_critical_fail_false_when_passed() {
        let r = AudioQcResult::new(AudioQcTest::TruePeakLimit, true, -2.0, -1.0);
        assert!(!r.is_critical_fail(0.0));
    }

    // --- AudioQcSpec tests ---

    #[test]
    fn test_ebu_r128_target() {
        let spec = AudioQcSpec::ebu_r128_broadcast();
        assert!((spec.target_lufs - (-23.0)).abs() < 1e-5);
    }

    #[test]
    fn test_ebu_r128_sample_rate() {
        let spec = AudioQcSpec::ebu_r128_broadcast();
        assert_eq!(spec.sample_rate, 48000);
    }

    #[test]
    fn test_atsc_a85_target() {
        let spec = AudioQcSpec::atsc_a85();
        assert!((spec.target_lufs - (-24.0)).abs() < 1e-5);
    }

    #[test]
    fn test_atsc_a85_max_tp() {
        let spec = AudioQcSpec::atsc_a85();
        assert!((spec.max_tp_dbfs - (-2.0)).abs() < 1e-5);
    }

    // --- AudioQcRunner tests ---

    #[test]
    fn test_runner_loudness_pass() {
        let runner = AudioQcRunner::new(AudioQcSpec::ebu_r128_broadcast());
        let result = runner.check_loudness(-23.0);
        assert!(result.passed);
    }

    #[test]
    fn test_runner_loudness_fail() {
        let runner = AudioQcRunner::new(AudioQcSpec::ebu_r128_broadcast());
        let result = runner.check_loudness(-20.0);
        assert!(!result.passed);
    }

    #[test]
    fn test_runner_loudness_within_tolerance() {
        let runner = AudioQcRunner::new(AudioQcSpec::ebu_r128_broadcast());
        // -23 ± 1.0 => -22.5 should pass
        let result = runner.check_loudness(-22.5);
        assert!(result.passed);
    }

    #[test]
    fn test_runner_true_peak_pass() {
        let runner = AudioQcRunner::new(AudioQcSpec::ebu_r128_broadcast());
        let result = runner.check_true_peak(-3.0);
        assert!(result.passed);
    }

    #[test]
    fn test_runner_true_peak_fail() {
        let runner = AudioQcRunner::new(AudioQcSpec::ebu_r128_broadcast());
        let result = runner.check_true_peak(0.0);
        assert!(!result.passed);
    }

    #[test]
    fn test_runner_sample_rate_pass() {
        let runner = AudioQcRunner::new(AudioQcSpec::ebu_r128_broadcast());
        let result = runner.check_sample_rate(48000);
        assert!(result.passed);
    }

    #[test]
    fn test_runner_sample_rate_fail() {
        let runner = AudioQcRunner::new(AudioQcSpec::ebu_r128_broadcast());
        let result = runner.check_sample_rate(44100);
        assert!(!result.passed);
    }
}
