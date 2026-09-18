//! Two-pass loudness analysis for normalization.
//!
//! This module provides comprehensive loudness analysis used to determine
//! the optimal gain adjustment for normalization.

use crate::NormalizeResult;
use oximedia_metering::{ComplianceResult, LoudnessMeter, LoudnessMetrics, MeterConfig, Standard};

/// Loudness analysis result.
///
/// Contains all measurements and recommendations needed for normalization.
#[derive(Clone, Debug)]
pub struct AnalysisResult {
    /// Measured integrated loudness in LUFS.
    pub integrated_lufs: f64,

    /// Measured loudness range in LU.
    pub loudness_range: f64,

    /// Measured true peak in dBTP.
    pub true_peak_dbtp: f64,

    /// Target loudness for the standard.
    pub target_lufs: f64,

    /// Maximum allowed true peak for the standard.
    pub max_peak_dbtp: f64,

    /// Recommended gain adjustment in dB.
    pub recommended_gain_db: f64,

    /// Gain that would cause the true peak to reach exactly max_peak_dbtp.
    pub safe_gain_db: f64,

    /// Is the audio compliant without normalization?
    pub is_compliant: bool,

    /// Detailed compliance information.
    pub compliance: ComplianceResult,

    /// Full loudness metrics.
    pub metrics: LoudnessMetrics,

    /// Standard being analyzed against.
    pub standard: Standard,
}

impl AnalysisResult {
    /// Check if normalization is needed.
    pub fn needs_normalization(&self) -> bool {
        !self.is_compliant
    }

    /// Check if gain would cause clipping.
    pub fn would_clip(&self, gain_db: f64) -> bool {
        let gain_linear = db_to_linear(gain_db);
        let new_peak = linear_to_db(self.metrics.true_peak_linear * gain_linear);
        new_peak > self.max_peak_dbtp
    }

    /// Get the maximum safe gain that won't cause clipping.
    pub fn max_safe_gain_db(&self) -> f64 {
        if self.metrics.true_peak_linear > 0.0 {
            self.max_peak_dbtp - linear_to_db(self.metrics.true_peak_linear)
        } else {
            60.0 // Very quiet audio, allow large gain
        }
    }

    /// Get recommended gain clamped to safe range.
    pub fn clamped_gain_db(&self, max_gain: f64) -> f64 {
        self.recommended_gain_db
            .min(self.safe_gain_db)
            .clamp(-60.0, max_gain)
    }
}

/// Loudness analyzer for normalization.
///
/// Performs comprehensive loudness analysis to determine optimal normalization parameters.
pub struct LoudnessAnalyzer {
    meter: LoudnessMeter,
    standard: Standard,
}

impl LoudnessAnalyzer {
    /// Create a new loudness analyzer.
    pub fn new(standard: Standard, sample_rate: f64, channels: usize) -> NormalizeResult<Self> {
        let config = MeterConfig::new(standard, sample_rate, channels);
        let meter = LoudnessMeter::new(config)?;

        Ok(Self { meter, standard })
    }

    /// Process f32 audio samples for analysis.
    pub fn process_f32(&mut self, samples: &[f32]) {
        self.meter.process_f32(samples);
    }

    /// Process f64 audio samples for analysis.
    pub fn process_f64(&mut self, samples: &[f64]) {
        self.meter.process_f64(samples);
    }

    /// Get the current analysis result.
    pub fn result(&mut self) -> AnalysisResult {
        let metrics = self.meter.metrics();
        let compliance = self.meter.check_compliance();

        let target_lufs = self.standard.target_lufs();
        let max_peak_dbtp = self.standard.max_true_peak_dbtp();

        // Calculate recommended gain
        let recommended_gain_db = if metrics.integrated_lufs.is_finite() {
            target_lufs - metrics.integrated_lufs
        } else {
            0.0
        };

        // Calculate safe gain (won't exceed peak limit)
        let safe_gain_db = if metrics.true_peak_linear > 0.0 {
            max_peak_dbtp - linear_to_db(metrics.true_peak_linear)
        } else {
            60.0
        };

        AnalysisResult {
            integrated_lufs: metrics.integrated_lufs,
            loudness_range: metrics.loudness_range,
            true_peak_dbtp: metrics.true_peak_dbtp,
            target_lufs,
            max_peak_dbtp,
            recommended_gain_db,
            safe_gain_db,
            is_compliant: compliance.is_compliant(),
            compliance,
            metrics,
            standard: self.standard,
        }
    }

    /// Get current loudness metrics.
    pub fn metrics(&mut self) -> LoudnessMetrics {
        self.meter.metrics()
    }

    /// Reset the analyzer.
    pub fn reset(&mut self) {
        self.meter.reset();
    }

    /// Get the underlying meter.
    pub fn meter(&self) -> &LoudnessMeter {
        &self.meter
    }

    /// Get the underlying meter (mutable).
    pub fn meter_mut(&mut self) -> &mut LoudnessMeter {
        &mut self.meter
    }
}

/// Convert dB to linear gain.
#[inline]
pub fn db_to_linear(db: f64) -> f64 {
    10.0_f64.powf(db / 20.0)
}

/// Convert linear gain to dB.
#[inline]
pub fn linear_to_db(linear: f64) -> f64 {
    20.0 * linear.log10()
}

/// Calculate peak headroom in dB.
#[inline]
pub fn peak_headroom_db(peak_dbtp: f64, max_peak_dbtp: f64) -> f64 {
    max_peak_dbtp - peak_dbtp
}

/// Calculate loudness deviation from target.
#[inline]
pub fn loudness_deviation_lu(measured_lufs: f64, target_lufs: f64) -> f64 {
    measured_lufs - target_lufs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_db_linear_conversion() {
        assert!((db_to_linear(0.0) - 1.0).abs() < 1e-10);
        assert!((db_to_linear(6.0) - 1.9952623149688797).abs() < 1e-6);
        assert!((db_to_linear(-6.0) - 0.5011872336272722).abs() < 1e-6);

        assert!((linear_to_db(1.0) - 0.0).abs() < 1e-10);
        assert!((linear_to_db(2.0) - 6.020599913279624).abs() < 1e-6);
    }

    #[test]
    fn test_peak_headroom() {
        assert_eq!(peak_headroom_db(-3.0, -1.0), 2.0);
        assert_eq!(peak_headroom_db(-10.0, -2.0), 8.0);
    }

    #[test]
    fn test_loudness_deviation() {
        assert_eq!(loudness_deviation_lu(-20.0, -23.0), 3.0);
        assert_eq!(loudness_deviation_lu(-26.0, -23.0), -3.0);
    }

    #[test]
    fn test_analyzer_creation() {
        let analyzer = LoudnessAnalyzer::new(Standard::EbuR128, 48000.0, 2);
        assert!(analyzer.is_ok());
    }

    #[test]
    fn test_analysis_result_methods() {
        let result = AnalysisResult {
            integrated_lufs: -20.0,
            loudness_range: 10.0,
            true_peak_dbtp: -3.0,
            target_lufs: -23.0,
            max_peak_dbtp: -1.0,
            recommended_gain_db: -3.0,
            safe_gain_db: 2.0,
            is_compliant: false,
            compliance: ComplianceResult {
                standard: Standard::EbuR128,
                loudness_compliant: false,
                peak_compliant: true,
                lra_acceptable: true,
                integrated_lufs: -20.0,
                true_peak_dbtp: -3.0,
                loudness_range: 10.0,
                target_lufs: -23.0,
                max_peak_dbtp: -1.0,
                deviation_lu: 3.0,
            },
            metrics: LoudnessMetrics::default(),
            standard: Standard::EbuR128,
        };

        assert!(result.needs_normalization());
        // Clamped to safe_gain_db (2.0), which is less than recommended (-3.0)
        assert_eq!(result.clamped_gain_db(10.0), -3.0);
    }
}
