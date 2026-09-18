//! Dynamic range analysis.

/// Dynamic range analyzer.
pub struct DynamicsAnalyzer;

impl DynamicsAnalyzer {
    /// Create a new dynamics analyzer.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Compute crest factor (peak to RMS ratio).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn crest_factor(&self, signal: &[f32]) -> f32 {
        if signal.is_empty() {
            return 0.0;
        }

        let peak = signal.iter().map(|s| s.abs()).fold(0.0_f32, f32::max);
        let rms = {
            let sum_squares: f32 = signal.iter().map(|s| s * s).sum();
            (sum_squares / signal.len() as f32).sqrt()
        };

        if rms > 0.0 {
            peak / rms
        } else {
            0.0
        }
    }

    /// Compute dynamic range in dB.
    #[must_use]
    pub fn dynamic_range_db(&self, signal: &[f32]) -> f32 {
        let crest = self.crest_factor(signal);
        if crest > 0.0 {
            20.0 * crest.log10()
        } else {
            0.0
        }
    }
}

impl Default for DynamicsAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamics_analyzer_creation() {
        let _analyzer = DynamicsAnalyzer::new();
    }

    #[test]
    fn test_crest_factor() {
        let analyzer = DynamicsAnalyzer::new();
        let signal = vec![1.0, 0.5, 0.3, 0.2];
        let crest = analyzer.crest_factor(&signal);
        assert!(crest > 0.0);
    }
}
