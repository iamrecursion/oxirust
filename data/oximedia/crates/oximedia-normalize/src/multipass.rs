//! Multi-pass processing controller.
//!
//! Coordinates multiple passes for accurate loudness normalization with
//! iterative refinement and quality optimization.

use crate::{
    AnalysisResult, LoudnessAnalyzer, NormalizationProcessor, NormalizeError, NormalizeResult,
    ProcessorConfig,
};
use oximedia_metering::Standard;

/// Multi-pass processing configuration.
#[derive(Clone, Debug)]
pub struct MultiPassConfig {
    /// Target loudness standard.
    pub standard: Standard,

    /// Sample rate in Hz.
    pub sample_rate: f64,

    /// Number of channels.
    pub channels: usize,

    /// Maximum number of passes.
    pub max_passes: usize,

    /// Convergence tolerance in LU.
    pub tolerance_lu: f64,

    /// Enable limiter in processing passes.
    pub enable_limiter: bool,

    /// Enable DRC in processing passes.
    pub enable_drc: bool,

    /// Lookahead time in milliseconds.
    pub lookahead_ms: f64,
}

impl MultiPassConfig {
    /// Create a new multi-pass configuration.
    pub fn new(standard: Standard, sample_rate: f64, channels: usize) -> Self {
        Self {
            standard,
            sample_rate,
            channels,
            max_passes: 3,
            tolerance_lu: 0.1,
            enable_limiter: true,
            enable_drc: false,
            lookahead_ms: 5.0,
        }
    }

    /// Create a high-precision configuration.
    pub fn high_precision(standard: Standard, sample_rate: f64, channels: usize) -> Self {
        Self {
            standard,
            sample_rate,
            channels,
            max_passes: 5,
            tolerance_lu: 0.05,
            enable_limiter: true,
            enable_drc: false,
            lookahead_ms: 10.0,
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> NormalizeResult<()> {
        if self.sample_rate < 8000.0 || self.sample_rate > 192_000.0 {
            return Err(NormalizeError::InvalidConfig(
                "Sample rate out of range".to_string(),
            ));
        }

        if self.channels == 0 || self.channels > 16 {
            return Err(NormalizeError::InvalidConfig(
                "Channel count out of range".to_string(),
            ));
        }

        if self.max_passes == 0 || self.max_passes > 10 {
            return Err(NormalizeError::InvalidConfig(
                "Max passes out of range (1-10)".to_string(),
            ));
        }

        if self.tolerance_lu <= 0.0 {
            return Err(NormalizeError::InvalidConfig(
                "Tolerance must be positive".to_string(),
            ));
        }

        Ok(())
    }
}

/// Multi-pass processing result.
#[derive(Clone, Debug)]
pub struct MultiPassResult {
    /// Final analysis result.
    pub final_analysis: AnalysisResult,

    /// Number of passes performed.
    pub passes_performed: usize,

    /// Converged to target within tolerance.
    pub converged: bool,

    /// Cumulative gain applied in dB.
    pub total_gain_db: f64,

    /// Per-pass analysis results.
    pub pass_results: Vec<PassResult>,
}

impl MultiPassResult {
    /// Check if the result meets the target.
    pub fn is_compliant(&self) -> bool {
        self.final_analysis.is_compliant
    }

    /// Get the final loudness deviation from target.
    pub fn final_deviation_lu(&self) -> f64 {
        self.final_analysis.integrated_lufs - self.final_analysis.target_lufs
    }
}

/// Result for a single pass.
#[derive(Clone, Debug)]
pub struct PassResult {
    /// Pass number (1-indexed).
    pub pass_number: usize,

    /// Analysis before this pass.
    pub before_analysis: AnalysisResult,

    /// Gain applied in this pass.
    pub applied_gain_db: f64,

    /// Analysis after this pass.
    pub after_analysis: AnalysisResult,

    /// Deviation from target after this pass.
    pub deviation_lu: f64,
}

/// Multi-pass processor.
///
/// Performs iterative normalization with analysis and refinement.
pub struct MultiPassProcessor {
    config: MultiPassConfig,
}

impl MultiPassProcessor {
    /// Create a new multi-pass processor.
    pub fn new(config: MultiPassConfig) -> NormalizeResult<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Process audio with multiple analysis/normalization passes.
    ///
    /// This is a conceptual implementation that shows the multi-pass structure.
    /// In practice, this would operate on actual audio data.
    pub fn process(&self, initial_analysis: AnalysisResult) -> NormalizeResult<MultiPassResult> {
        let mut pass_results = Vec::new();
        let mut current_analysis = initial_analysis.clone();
        let mut total_gain_db = 0.0;
        let mut converged = false;

        for pass_num in 1..=self.config.max_passes {
            // Check if we've converged
            let deviation = (current_analysis.integrated_lufs - current_analysis.target_lufs).abs();
            if deviation <= self.config.tolerance_lu {
                converged = true;
                break;
            }

            // Calculate gain for this pass
            let gain_db = current_analysis.recommended_gain_db;

            // Ensure we don't exceed safe gain
            let safe_gain = gain_db.min(current_analysis.safe_gain_db);

            // In a real implementation, we would:
            // 1. Apply the gain to the audio
            // 2. Re-analyze the processed audio
            // 3. Check convergence

            // Simulate post-processing analysis
            let after_analysis = self.simulate_after_analysis(&current_analysis, safe_gain);

            let pass_result = PassResult {
                pass_number: pass_num,
                before_analysis: current_analysis.clone(),
                applied_gain_db: safe_gain,
                after_analysis: after_analysis.clone(),
                deviation_lu: after_analysis.integrated_lufs - after_analysis.target_lufs,
            };

            pass_results.push(pass_result);

            total_gain_db += safe_gain;
            current_analysis = after_analysis;
        }

        Ok(MultiPassResult {
            final_analysis: current_analysis,
            passes_performed: pass_results.len(),
            converged,
            total_gain_db,
            pass_results,
        })
    }

    /// Simulate analysis after applying gain.
    ///
    /// In a real implementation, this would process actual audio and re-analyze.
    fn simulate_after_analysis(&self, before: &AnalysisResult, gain_db: f64) -> AnalysisResult {
        // Simulate the effect of applying gain
        let new_lufs = before.integrated_lufs + gain_db;

        // Calculate new peak
        let gain_linear = 10.0_f64.powf(gain_db / 20.0);
        let new_peak_linear = before.metrics.true_peak_linear * gain_linear;
        let new_peak_dbtp = 20.0 * new_peak_linear.log10();

        // Create updated analysis
        let mut updated = before.clone();
        updated.integrated_lufs = new_lufs;
        updated.true_peak_dbtp = new_peak_dbtp;
        updated.recommended_gain_db = updated.target_lufs - new_lufs;

        updated
    }

    /// Get the processor configuration.
    pub fn config(&self) -> &MultiPassConfig {
        &self.config
    }

    /// Create an analyzer for a pass.
    fn create_analyzer(&self) -> NormalizeResult<LoudnessAnalyzer> {
        LoudnessAnalyzer::new(
            self.config.standard,
            self.config.sample_rate,
            self.config.channels,
        )
    }

    /// Create a processor for a pass.
    fn create_processor(&self) -> NormalizeResult<NormalizationProcessor> {
        let processor_config = ProcessorConfig {
            sample_rate: self.config.sample_rate,
            channels: self.config.channels,
            enable_limiter: self.config.enable_limiter,
            enable_drc: self.config.enable_drc,
            lookahead_ms: self.config.lookahead_ms,
        };

        NormalizationProcessor::new(processor_config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_metering::{ComplianceResult, LoudnessMetrics};

    #[test]
    fn test_multipass_config_validation() {
        let config = MultiPassConfig::new(Standard::EbuR128, 48000.0, 2);
        assert!(config.validate().is_ok());

        let bad_config = MultiPassConfig {
            max_passes: 0,
            ..config
        };
        assert!(bad_config.validate().is_err());
    }

    #[test]
    fn test_multipass_processor_creation() {
        let config = MultiPassConfig::new(Standard::EbuR128, 48000.0, 2);
        let processor = MultiPassProcessor::new(config);
        assert!(processor.is_ok());
    }

    #[test]
    fn test_high_precision_config() {
        let config = MultiPassConfig::high_precision(Standard::EbuR128, 48000.0, 2);
        assert_eq!(config.max_passes, 5);
        assert_eq!(config.tolerance_lu, 0.05);
    }

    #[test]
    fn test_multipass_processing() {
        let config = MultiPassConfig::new(Standard::EbuR128, 48000.0, 2);
        let processor = MultiPassProcessor::new(config).expect("should succeed in test");

        let initial_analysis = create_test_analysis();
        let result = processor.process(initial_analysis);
        assert!(result.is_ok());

        let result = result.expect("should succeed in test");
        assert!(result.passes_performed > 0);
        assert!(result.passes_performed <= 3);
    }

    fn create_test_analysis() -> AnalysisResult {
        AnalysisResult {
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
        }
    }
}
