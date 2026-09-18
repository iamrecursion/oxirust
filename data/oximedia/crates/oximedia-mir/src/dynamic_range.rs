#![allow(dead_code)]
//! Dynamic range analysis — LRA (Loudness Range), crest factor, compression detection.

/// Loudness Range value (EBU R 128 LRA) and basic statistics.
#[derive(Debug, Clone)]
pub struct LoudnessRange {
    /// LRA value in LU (Loudness Units).
    pub lra: f32,
    /// Short-term loudness percentile at 10 % (dBFS, approximately).
    pub low_percentile: f32,
    /// Short-term loudness percentile at 95 % (dBFS, approximately).
    pub high_percentile: f32,
}

impl LoudnessRange {
    /// Create a new `LoudnessRange`.
    #[must_use]
    pub fn new(lra: f32, low_percentile: f32, high_percentile: f32) -> Self {
        Self {
            lra,
            low_percentile,
            high_percentile,
        }
    }

    /// Returns `true` when the LRA suggests heavy dynamic compression (< 6 LU).
    #[must_use]
    pub fn is_compressed(&self) -> bool {
        self.lra < 6.0
    }

    /// Returns `true` when the LRA is within the EBU recommended broadcast range (6–20 LU).
    #[must_use]
    pub fn is_broadcast_compliant(&self) -> bool {
        self.lra >= 6.0 && self.lra <= 20.0
    }
}

/// Per-frame RMS and peak data used for dynamic range analysis.
#[derive(Debug, Clone, Copy)]
pub struct DynamicFrame {
    /// RMS energy of this frame (linear).
    pub rms: f32,
    /// Peak sample magnitude (linear, absolute).
    pub peak: f32,
}

/// Accumulates audio frames and computes dynamic-range metrics.
#[derive(Debug, Clone, Default)]
pub struct DynamicRangeAnalyzer {
    frames: Vec<DynamicFrame>,
    sample_rate: f32,
    frame_size: usize,
}

impl DynamicRangeAnalyzer {
    /// Create a new analyser.
    #[must_use]
    pub fn new(sample_rate: f32, frame_size: usize) -> Self {
        Self {
            frames: Vec::new(),
            sample_rate,
            frame_size,
        }
    }

    /// Add a single pre-computed frame.
    pub fn add_frame(&mut self, frame: DynamicFrame) {
        self.frames.push(frame);
    }

    /// Ingest a slice of audio samples, splitting into frames of `frame_size`.
    #[allow(clippy::cast_precision_loss)]
    pub fn ingest(&mut self, samples: &[f32]) {
        let fs = self.frame_size.max(1);
        for chunk in samples.chunks(fs) {
            let rms = {
                let sum: f32 = chunk.iter().map(|s| s * s).sum();
                (sum / chunk.len() as f32).sqrt()
            };
            let peak = chunk.iter().copied().map(f32::abs).fold(0.0_f32, f32::max);
            self.add_frame(DynamicFrame { rms, peak });
        }
    }

    /// Number of frames accumulated so far.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Compute the Loudness Range (LRA) approximation in LU.
    ///
    /// Uses the interquartile range of RMS-dB values across frames (10th–95th percentile).
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation
    )]
    #[must_use]
    pub fn lra(&self) -> f32 {
        if self.frames.is_empty() {
            return 0.0;
        }
        let mut db_values: Vec<f32> = self
            .frames
            .iter()
            .map(|f| {
                let rms = f.rms.max(1e-9);
                20.0 * rms.log10()
            })
            .collect();
        db_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = db_values.len();
        let lo_idx = (n as f32 * 0.10) as usize;
        let hi_idx = ((n as f32 * 0.95) as usize).min(n - 1);
        db_values[hi_idx] - db_values[lo_idx]
    }

    /// Crest factor (peak-to-RMS ratio in dB).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn crest_factor(&self) -> f32 {
        if self.frames.is_empty() {
            return 0.0;
        }
        let avg_rms: f32 =
            self.frames.iter().map(|f| f.rms).sum::<f32>() / self.frames.len() as f32;
        let max_peak = self.frames.iter().map(|f| f.peak).fold(0.0_f32, f32::max);
        if avg_rms < 1e-9 {
            return 0.0;
        }
        20.0 * (max_peak / avg_rms).log10()
    }

    /// Build a `DynamicRangeReport` from accumulated data.
    #[must_use]
    pub fn report(&self) -> DynamicRangeReport {
        let lra = self.lra();
        let crest = self.crest_factor();
        let lr = LoudnessRange::new(lra, 0.0, lra);
        DynamicRangeReport {
            loudness_range: lr,
            crest_factor_db: crest,
            frame_count: self.frame_count(),
        }
    }
}

/// Summary report produced by `DynamicRangeAnalyzer`.
#[derive(Debug, Clone)]
pub struct DynamicRangeReport {
    /// Loudness range statistics.
    pub loudness_range: LoudnessRange,
    /// Crest factor in dB.
    pub crest_factor_db: f32,
    /// Number of frames used for analysis.
    pub frame_count: usize,
}

impl DynamicRangeReport {
    /// Returns `true` when the content appears over-compressed (LRA < 3 LU and CF < 6 dB).
    #[must_use]
    pub fn is_over_compressed(&self) -> bool {
        self.loudness_range.lra < 3.0 && self.crest_factor_db < 6.0
    }

    /// Returns `true` when the crest factor indicates clipping risk (< 3 dB).
    #[must_use]
    pub fn is_clipping_risk(&self) -> bool {
        self.crest_factor_db < 3.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_signal(value: f32, len: usize) -> Vec<f32> {
        vec![value; len]
    }

    #[test]
    fn test_loudness_range_is_compressed() {
        let lr = LoudnessRange::new(4.0, -30.0, -26.0);
        assert!(lr.is_compressed());
    }

    #[test]
    fn test_loudness_range_not_compressed() {
        let lr = LoudnessRange::new(10.0, -35.0, -25.0);
        assert!(!lr.is_compressed());
    }

    #[test]
    fn test_broadcast_compliant() {
        let lr = LoudnessRange::new(12.0, -30.0, -18.0);
        assert!(lr.is_broadcast_compliant());
    }

    #[test]
    fn test_broadcast_non_compliant_too_low() {
        let lr = LoudnessRange::new(2.0, -25.0, -23.0);
        assert!(!lr.is_broadcast_compliant());
    }

    #[test]
    fn test_analyzer_no_frames() {
        let analyzer = DynamicRangeAnalyzer::new(48000.0, 1024);
        assert_eq!(analyzer.frame_count(), 0);
        assert!((analyzer.lra() - 0.0).abs() < 1e-6);
        assert!((analyzer.crest_factor() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_analyzer_add_frame() {
        let mut a = DynamicRangeAnalyzer::new(48000.0, 1024);
        a.add_frame(DynamicFrame {
            rms: 0.1,
            peak: 0.3,
        });
        assert_eq!(a.frame_count(), 1);
    }

    #[test]
    fn test_analyzer_ingest_splits_frames() {
        let mut a = DynamicRangeAnalyzer::new(48000.0, 512);
        let sig = flat_signal(0.5, 2048);
        a.ingest(&sig);
        assert_eq!(a.frame_count(), 4);
    }

    #[test]
    fn test_crest_factor_flat_signal() {
        let mut a = DynamicRangeAnalyzer::new(48000.0, 512);
        // Flat signal → peak == RMS → crest factor ≈ 0 dB.
        a.ingest(&flat_signal(0.5, 512));
        let cf = a.crest_factor();
        assert!(cf.abs() < 1.0, "crest factor unexpectedly large: {cf}");
    }

    #[test]
    fn test_report_is_over_compressed_false_for_wide_dr() {
        let mut a = DynamicRangeAnalyzer::new(48000.0, 256);
        // Build several frames with varied RMS to create wide LRA.
        for v in [0.01, 0.1, 0.5, 0.9, 0.02, 0.8, 0.03, 0.7] {
            a.add_frame(DynamicFrame { rms: v, peak: v });
        }
        let report = a.report();
        // Wide dynamic range should NOT be over-compressed.
        assert!(!report.is_over_compressed());
    }

    #[test]
    fn test_report_frame_count() {
        let mut a = DynamicRangeAnalyzer::new(48000.0, 512);
        a.ingest(&flat_signal(0.3, 1024));
        let report = a.report();
        assert_eq!(report.frame_count, 2);
    }

    #[test]
    fn test_report_clipping_risk_false_for_healthy_signal() {
        let mut a = DynamicRangeAnalyzer::new(48000.0, 512);
        // Low RMS, moderate peak → high crest factor.
        a.add_frame(DynamicFrame {
            rms: 0.01,
            peak: 0.9,
        });
        let report = a.report();
        assert!(!report.is_clipping_risk());
    }
}
