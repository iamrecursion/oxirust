#![allow(dead_code)]
//! Multi-algorithm watermark detection pipeline.
//!
//! This module provides a detection framework that can scan audio for
//! watermarks embedded via multiple algorithms simultaneously, rank
//! detection candidates by confidence, and report the most likely
//! embedded payload.

use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// Detection method
// ---------------------------------------------------------------------------

/// Available detection methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DetectionMethod {
    /// Correlation-based spread-spectrum detection.
    Correlation,
    /// Autocorrelation-based echo detection.
    EchoAutocorrelation,
    /// Phase analysis detection.
    PhaseAnalysis,
    /// LSB extraction.
    LsbExtract,
    /// Statistical patchwork detection.
    PatchworkStatistical,
    /// QIM lattice detection.
    QimLattice,
}

impl DetectionMethod {
    /// Return all available detection methods.
    #[must_use]
    pub fn all() -> &'static [DetectionMethod] {
        &[
            Self::Correlation,
            Self::EchoAutocorrelation,
            Self::PhaseAnalysis,
            Self::LsbExtract,
            Self::PatchworkStatistical,
            Self::QimLattice,
        ]
    }

    /// Human-readable name of this detection method.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Correlation => "Spread-Spectrum Correlation",
            Self::EchoAutocorrelation => "Echo Autocorrelation",
            Self::PhaseAnalysis => "Phase Analysis",
            Self::LsbExtract => "LSB Extraction",
            Self::PatchworkStatistical => "Patchwork Statistical",
            Self::QimLattice => "QIM Lattice",
        }
    }
}

// ---------------------------------------------------------------------------
// Detection candidate
// ---------------------------------------------------------------------------

/// A single detection candidate produced by one method.
#[derive(Debug, Clone)]
pub struct DetectionCandidate {
    /// The method that produced this candidate.
    pub method: DetectionMethod,
    /// Confidence score in [0.0, 1.0].
    pub confidence: f64,
    /// Detected raw bits (may be noisy).
    pub bits: Vec<bool>,
    /// Correlation peak value (method-specific metric).
    pub peak_value: f64,
    /// Frame offset where the strongest detection occurred.
    pub frame_offset: usize,
}

impl DetectionCandidate {
    /// Convert detected bits to bytes (MSB-first, zero-padded).
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        bits_to_bytes(&self.bits)
    }

    /// Number of detected bits.
    #[must_use]
    pub fn num_bits(&self) -> usize {
        self.bits.len()
    }
}

// ---------------------------------------------------------------------------
// Pipeline config
// ---------------------------------------------------------------------------

/// Configuration for the detection pipeline.
#[derive(Debug, Clone)]
pub struct DetectionPipelineConfig {
    /// Methods to attempt (in order).
    pub methods: Vec<DetectionMethod>,
    /// Frame size for analysis.
    pub frame_size: usize,
    /// Hop size between frames.
    pub hop_size: usize,
    /// Minimum confidence to accept a candidate.
    pub min_confidence: f64,
    /// Maximum number of candidates to report.
    pub max_candidates: usize,
    /// Key seed for keyed detection methods.
    pub key_seed: u64,
    /// Expected payload bit count (0 = auto-detect).
    pub expected_bits: usize,
}

impl Default for DetectionPipelineConfig {
    fn default() -> Self {
        Self {
            methods: DetectionMethod::all().to_vec(),
            frame_size: 2048,
            hop_size: 1024,
            min_confidence: 0.3,
            max_candidates: 10,
            key_seed: 0,
            expected_bits: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Pipeline result
// ---------------------------------------------------------------------------

/// Aggregated result of running the detection pipeline.
#[derive(Debug, Clone)]
pub struct DetectionResult {
    /// All candidates found, sorted by confidence descending.
    pub candidates: Vec<DetectionCandidate>,
    /// Per-method confidence map.
    pub method_confidences: BTreeMap<DetectionMethod, f64>,
    /// Whether any candidate exceeds the minimum confidence threshold.
    pub detected: bool,
    /// Total frames analysed.
    pub frames_analysed: usize,
}

impl DetectionResult {
    /// Return the best candidate (highest confidence), if any.
    #[must_use]
    pub fn best(&self) -> Option<&DetectionCandidate> {
        self.candidates.first()
    }

    /// Return all candidates from a specific method.
    #[must_use]
    pub fn by_method(&self, method: DetectionMethod) -> Vec<&DetectionCandidate> {
        self.candidates
            .iter()
            .filter(|c| c.method == method)
            .collect()
    }

    /// Return the overall confidence (best candidate confidence or 0).
    #[must_use]
    pub fn confidence(&self) -> f64 {
        self.best().map_or(0.0, |c| c.confidence)
    }
}

// ---------------------------------------------------------------------------
// Detection pipeline
// ---------------------------------------------------------------------------

/// Multi-method watermark detection pipeline.
#[derive(Debug, Clone)]
pub struct DetectionPipeline {
    /// Pipeline configuration.
    pub config: DetectionPipelineConfig,
}

impl DetectionPipeline {
    /// Create a new pipeline with the given config.
    #[must_use]
    pub fn new(config: DetectionPipelineConfig) -> Self {
        Self { config }
    }

    /// Run detection on the given audio samples.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn detect(&self, samples: &[f64]) -> DetectionResult {
        let num_frames = if samples.len() >= self.config.frame_size {
            (samples.len() - self.config.frame_size) / self.config.hop_size + 1
        } else {
            0
        };

        let mut candidates = Vec::new();
        let mut method_confidences = BTreeMap::new();

        for &method in &self.config.methods {
            let candidate = self.run_method(method, samples, num_frames);
            method_confidences.insert(method, candidate.confidence);
            if candidate.confidence >= self.config.min_confidence {
                candidates.push(candidate);
            }
        }

        candidates.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(self.config.max_candidates);

        let detected = !candidates.is_empty();

        DetectionResult {
            candidates,
            method_confidences,
            detected,
            frames_analysed: num_frames,
        }
    }

    /// Run a single detection method.
    #[allow(clippy::cast_precision_loss)]
    fn run_method(
        &self,
        method: DetectionMethod,
        samples: &[f64],
        num_frames: usize,
    ) -> DetectionCandidate {
        match method {
            DetectionMethod::Correlation => self.detect_correlation(samples, num_frames),
            DetectionMethod::EchoAutocorrelation => self.detect_echo(samples, num_frames),
            DetectionMethod::PhaseAnalysis => self.detect_phase(samples, num_frames),
            DetectionMethod::LsbExtract => self.detect_lsb(samples, num_frames),
            DetectionMethod::PatchworkStatistical => self.detect_patchwork(samples, num_frames),
            DetectionMethod::QimLattice => self.detect_qim(samples, num_frames),
        }
    }

    /// Correlation-based detection (spread-spectrum).
    #[allow(clippy::cast_precision_loss)]
    fn detect_correlation(&self, samples: &[f64], num_frames: usize) -> DetectionCandidate {
        let spread = self.generate_pn_sequence(self.config.frame_size);
        let mut max_corr = 0.0f64;
        let mut best_frame = 0usize;
        let mut bits = Vec::new();

        for f in 0..num_frames {
            let start = f * self.config.hop_size;
            let mut dot = 0.0f64;
            for (i, &s) in spread.iter().enumerate() {
                if start + i < samples.len() {
                    dot += samples[start + i] * s;
                }
            }
            let corr = dot / self.config.frame_size as f64;
            bits.push(corr > 0.0);
            let abs_corr = corr.abs();
            if abs_corr > max_corr {
                max_corr = abs_corr;
                best_frame = f;
            }
        }

        let confidence = (max_corr * 100.0).min(1.0);
        if self.config.expected_bits > 0 {
            bits.truncate(self.config.expected_bits);
        }

        DetectionCandidate {
            method: DetectionMethod::Correlation,
            confidence,
            bits,
            peak_value: max_corr,
            frame_offset: best_frame,
        }
    }

    /// Echo-based detection via autocorrelation.
    #[allow(clippy::cast_precision_loss)]
    fn detect_echo(&self, samples: &[f64], num_frames: usize) -> DetectionCandidate {
        let delay_0 = 50usize;
        let delay_1 = 100usize;
        let mut bits = Vec::new();
        let mut max_peak = 0.0f64;

        for f in 0..num_frames {
            let start = f * self.config.hop_size;
            let end = (start + self.config.frame_size).min(samples.len());
            if end <= start {
                continue;
            }
            let frame = &samples[start..end];

            let corr_0 = autocorrelation_at(frame, delay_0);
            let corr_1 = autocorrelation_at(frame, delay_1);
            bits.push(corr_1 > corr_0);

            let peak = corr_0.abs().max(corr_1.abs());
            if peak > max_peak {
                max_peak = peak;
            }
        }

        if self.config.expected_bits > 0 {
            bits.truncate(self.config.expected_bits);
        }

        DetectionCandidate {
            method: DetectionMethod::EchoAutocorrelation,
            confidence: (max_peak * 10.0).min(1.0),
            bits,
            peak_value: max_peak,
            frame_offset: 0,
        }
    }

    /// Phase-based detection.
    #[allow(clippy::cast_precision_loss)]
    fn detect_phase(&self, samples: &[f64], num_frames: usize) -> DetectionCandidate {
        let mut bits = Vec::new();
        let mut max_energy = 0.0f64;

        for f in 0..num_frames {
            let start = f * self.config.hop_size;
            let end = (start + self.config.frame_size).min(samples.len());
            if end <= start {
                continue;
            }
            let frame = &samples[start..end];
            // Simplified: use sign of the sum of the first half vs second half
            let half = frame.len() / 2;
            let sum_first: f64 = frame[..half].iter().sum();
            let sum_second: f64 = frame[half..].iter().sum();
            bits.push(sum_first > sum_second);

            let e = (sum_first.abs() + sum_second.abs()) / frame.len() as f64;
            if e > max_energy {
                max_energy = e;
            }
        }

        if self.config.expected_bits > 0 {
            bits.truncate(self.config.expected_bits);
        }

        DetectionCandidate {
            method: DetectionMethod::PhaseAnalysis,
            confidence: (max_energy * 5.0).min(1.0),
            bits,
            peak_value: max_energy,
            frame_offset: 0,
        }
    }

    /// LSB extraction.
    #[allow(clippy::cast_precision_loss)]
    fn detect_lsb(&self, samples: &[f64], _num_frames: usize) -> DetectionCandidate {
        let target_bits = if self.config.expected_bits > 0 {
            self.config.expected_bits
        } else {
            samples.len().min(256)
        };

        let mut bits = Vec::with_capacity(target_bits);
        for &s in samples.iter().take(target_bits) {
            let quantized = (s * 32768.0) as i32;
            bits.push(quantized & 1 == 1);
        }

        // Confidence: measure how many bits are non-random (simple entropy check)
        let ones = bits.iter().filter(|&&b| b).count();
        let ratio = ones as f64 / bits.len().max(1) as f64;
        let deviation = (ratio - 0.5).abs() * 2.0;

        // Scale by signal energy: all-zero input is not a valid watermark
        let energy: f64 = samples
            .iter()
            .take(target_bits)
            .map(|&s| s * s)
            .sum::<f64>()
            / target_bits.max(1) as f64;
        let energy_scale = (energy * 1000.0).min(1.0);
        let final_conf = (deviation * energy_scale).min(1.0);

        DetectionCandidate {
            method: DetectionMethod::LsbExtract,
            confidence: final_conf,
            bits,
            peak_value: final_conf,
            frame_offset: 0,
        }
    }

    /// Patchwork statistical detection.
    #[allow(clippy::cast_precision_loss)]
    fn detect_patchwork(&self, samples: &[f64], num_frames: usize) -> DetectionCandidate {
        let pair_distance = 10usize;
        let pairs_per_bit = 50usize;
        let mut bits = Vec::new();
        let mut max_stat = 0.0f64;

        let total_bits = num_frames.min(samples.len() / (pairs_per_bit * 2 + pair_distance).max(1));
        for b in 0..total_bits {
            let base = b * pairs_per_bit * 2;
            let mut diff_sum = 0.0f64;
            for p in 0..pairs_per_bit {
                let idx_a = base + p * 2;
                let idx_b = idx_a + pair_distance;
                if idx_b < samples.len() {
                    diff_sum += samples[idx_a] - samples[idx_b];
                }
            }
            let stat = diff_sum / pairs_per_bit as f64;
            bits.push(stat > 0.0);
            if stat.abs() > max_stat {
                max_stat = stat.abs();
            }
        }

        if self.config.expected_bits > 0 {
            bits.truncate(self.config.expected_bits);
        }

        DetectionCandidate {
            method: DetectionMethod::PatchworkStatistical,
            confidence: (max_stat * 50.0).min(1.0),
            bits,
            peak_value: max_stat,
            frame_offset: 0,
        }
    }

    /// QIM lattice detection.
    #[allow(clippy::cast_precision_loss)]
    fn detect_qim(&self, samples: &[f64], num_frames: usize) -> DetectionCandidate {
        let step = 0.01f64;
        let mut bits = Vec::new();
        let mut total_confidence = 0.0f64;

        for f in 0..num_frames {
            let start = f * self.config.hop_size;
            let end = (start + self.config.frame_size).min(samples.len());
            if end <= start {
                continue;
            }
            let frame = &samples[start..end];
            let mean: f64 = frame.iter().sum::<f64>() / frame.len() as f64;
            let quantized = (mean / step).round();
            let bit = (quantized as i64) % 2 != 0;
            bits.push(bit);

            let dist = (mean - quantized * step).abs();
            total_confidence += 1.0 - (dist / step).min(1.0);
        }

        if self.config.expected_bits > 0 {
            bits.truncate(self.config.expected_bits);
        }

        let avg_conf = if num_frames > 0 {
            total_confidence / num_frames as f64
        } else {
            0.0
        };

        // Scale confidence by signal energy: silence should not produce
        // high confidence even when quantization aligns perfectly.
        let energy: f64 = samples.iter().map(|&s| s * s).sum::<f64>() / samples.len().max(1) as f64;
        let energy_scale = (energy * 1000.0).min(1.0);
        let final_conf = (avg_conf * energy_scale).min(1.0);

        DetectionCandidate {
            method: DetectionMethod::QimLattice,
            confidence: final_conf,
            bits,
            peak_value: final_conf,
            frame_offset: 0,
        }
    }

    /// Generate a pseudo-random +/-1 sequence.
    fn generate_pn_sequence(&self, len: usize) -> Vec<f64> {
        let mut seq = Vec::with_capacity(len);
        let mut state = self.config.key_seed.wrapping_add(0xDEAD_BEEF);
        for _ in 0..len {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            seq.push(if state & 1 == 0 { 1.0 } else { -1.0 });
        }
        seq
    }
}

// ---------------------------------------------------------------------------
// Confidence combiner
// ---------------------------------------------------------------------------

/// Strategy for combining confidence scores from multiple methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CombineStrategy {
    /// Take the maximum confidence.
    Max,
    /// Average all confidences.
    Mean,
    /// Weighted average (weights proportional to confidence itself).
    WeightedMean,
}

/// Combine multiple detection results into a single confidence score.
#[allow(clippy::cast_precision_loss)]
pub fn combine_confidences(confidences: &[f64], strategy: CombineStrategy) -> f64 {
    if confidences.is_empty() {
        return 0.0;
    }
    match strategy {
        CombineStrategy::Max => confidences.iter().copied().fold(0.0f64, f64::max),
        CombineStrategy::Mean => confidences.iter().sum::<f64>() / confidences.len() as f64,
        CombineStrategy::WeightedMean => {
            let weight_sum: f64 = confidences.iter().sum();
            if weight_sum <= 0.0 {
                return 0.0;
            }
            let weighted: f64 = confidences.iter().map(|&c| c * c).sum();
            weighted / weight_sum
        }
    }
}

// ---------------------------------------------------------------------------
// Scan report
// ---------------------------------------------------------------------------

/// A summarised scan report suitable for logging or display.
#[derive(Debug, Clone)]
pub struct ScanReport {
    /// Whether a watermark was detected.
    pub watermark_found: bool,
    /// Best method name.
    pub best_method: String,
    /// Best confidence.
    pub best_confidence: f64,
    /// Number of candidate detections.
    pub num_candidates: usize,
    /// Combined confidence across all methods.
    pub combined_confidence: f64,
    /// Number of frames scanned.
    pub frames_scanned: usize,
}

/// Generate a scan report from a detection result.
#[must_use]
pub fn generate_report(result: &DetectionResult) -> ScanReport {
    let confs: Vec<f64> = result.method_confidences.values().copied().collect();
    let combined = combine_confidences(&confs, CombineStrategy::WeightedMean);

    let (best_method, best_confidence) = result.best().map_or_else(
        || ("None".to_string(), 0.0),
        |c| (c.method.name().to_string(), c.confidence),
    );

    ScanReport {
        watermark_found: result.detected,
        best_method,
        best_confidence,
        num_candidates: result.candidates.len(),
        combined_confidence: combined,
        frames_scanned: result.frames_analysed,
    }
}

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

/// Autocorrelation at a given lag.
#[allow(clippy::cast_precision_loss)]
fn autocorrelation_at(frame: &[f64], lag: usize) -> f64 {
    if lag >= frame.len() {
        return 0.0;
    }
    let n = frame.len() - lag;
    let mut sum = 0.0f64;
    for i in 0..n {
        sum += frame[i] * frame[i + lag];
    }
    sum / n as f64
}

/// Convert bits to bytes (MSB first).
fn bits_to_bytes(bits: &[bool]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(bits.len().div_ceil(8));
    for chunk in bits.chunks(8) {
        let mut byte = 0u8;
        for (i, &b) in chunk.iter().enumerate() {
            if b {
                byte |= 1 << (7 - i);
            }
        }
        bytes.push(byte);
    }
    bytes
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detection_method_all() {
        let all = DetectionMethod::all();
        assert_eq!(all.len(), 6);
    }

    #[test]
    fn test_detection_method_names() {
        assert_eq!(
            DetectionMethod::Correlation.name(),
            "Spread-Spectrum Correlation"
        );
        assert_eq!(DetectionMethod::LsbExtract.name(), "LSB Extraction");
    }

    #[test]
    fn test_default_config() {
        let cfg = DetectionPipelineConfig::default();
        assert_eq!(cfg.methods.len(), 6);
        assert_eq!(cfg.frame_size, 2048);
        assert!((cfg.min_confidence - 0.3).abs() < 1e-9);
    }

    #[test]
    fn test_pipeline_empty_signal() {
        let pipeline = DetectionPipeline::new(DetectionPipelineConfig::default());
        let result = pipeline.detect(&[]);
        assert_eq!(result.frames_analysed, 0);
    }

    #[test]
    fn test_pipeline_silence() {
        let config = DetectionPipelineConfig {
            frame_size: 256,
            hop_size: 128,
            ..Default::default()
        };
        let pipeline = DetectionPipeline::new(config);
        let samples = vec![0.0f64; 2048];
        let result = pipeline.detect(&samples);
        assert!(result.frames_analysed > 0);
        // Silence should produce low confidence
        assert!(result.confidence() < 0.5);
    }

    #[test]
    fn test_candidate_to_bytes() {
        let candidate = DetectionCandidate {
            method: DetectionMethod::Correlation,
            confidence: 0.9,
            bits: vec![true, false, true, false, false, false, false, true],
            peak_value: 0.5,
            frame_offset: 0,
        };
        let bytes = candidate.to_bytes();
        assert_eq!(bytes, vec![0b10100001]);
        assert_eq!(candidate.num_bits(), 8);
    }

    #[test]
    fn test_result_best() {
        let result = DetectionResult {
            candidates: vec![
                DetectionCandidate {
                    method: DetectionMethod::Correlation,
                    confidence: 0.8,
                    bits: vec![true],
                    peak_value: 0.5,
                    frame_offset: 0,
                },
                DetectionCandidate {
                    method: DetectionMethod::LsbExtract,
                    confidence: 0.3,
                    bits: vec![false],
                    peak_value: 0.1,
                    frame_offset: 0,
                },
            ],
            method_confidences: BTreeMap::new(),
            detected: true,
            frames_analysed: 10,
        };
        assert_eq!(
            result.best().expect("should succeed in test").method,
            DetectionMethod::Correlation
        );
        assert!((result.confidence() - 0.8).abs() < 1e-9);
    }

    #[test]
    fn test_result_by_method() {
        let result = DetectionResult {
            candidates: vec![
                DetectionCandidate {
                    method: DetectionMethod::Correlation,
                    confidence: 0.8,
                    bits: vec![],
                    peak_value: 0.0,
                    frame_offset: 0,
                },
                DetectionCandidate {
                    method: DetectionMethod::LsbExtract,
                    confidence: 0.5,
                    bits: vec![],
                    peak_value: 0.0,
                    frame_offset: 0,
                },
            ],
            method_confidences: BTreeMap::new(),
            detected: true,
            frames_analysed: 5,
        };
        let corr = result.by_method(DetectionMethod::Correlation);
        assert_eq!(corr.len(), 1);
        let echo = result.by_method(DetectionMethod::EchoAutocorrelation);
        assert!(echo.is_empty());
    }

    #[test]
    fn test_combine_max() {
        let confs = vec![0.1, 0.8, 0.3];
        assert!((combine_confidences(&confs, CombineStrategy::Max) - 0.8).abs() < 1e-9);
    }

    #[test]
    fn test_combine_mean() {
        let confs = vec![0.2, 0.4, 0.6];
        assert!((combine_confidences(&confs, CombineStrategy::Mean) - 0.4).abs() < 1e-9);
    }

    #[test]
    fn test_combine_weighted_mean() {
        let confs = vec![0.0, 1.0];
        let wm = combine_confidences(&confs, CombineStrategy::WeightedMean);
        // weighted = (0*0 + 1*1) / (0+1) = 1.0
        assert!((wm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_combine_empty() {
        assert!((combine_confidences(&[], CombineStrategy::Max)).abs() < 1e-9);
    }

    #[test]
    fn test_generate_report() {
        let mut method_confidences = BTreeMap::new();
        method_confidences.insert(DetectionMethod::Correlation, 0.7);
        method_confidences.insert(DetectionMethod::LsbExtract, 0.2);

        let result = DetectionResult {
            candidates: vec![DetectionCandidate {
                method: DetectionMethod::Correlation,
                confidence: 0.7,
                bits: vec![true, false],
                peak_value: 0.5,
                frame_offset: 3,
            }],
            method_confidences,
            detected: true,
            frames_analysed: 20,
        };

        let report = generate_report(&result);
        assert!(report.watermark_found);
        assert_eq!(report.best_method, "Spread-Spectrum Correlation");
        assert!((report.best_confidence - 0.7).abs() < 1e-9);
        assert_eq!(report.num_candidates, 1);
        assert_eq!(report.frames_scanned, 20);
        assert!(report.combined_confidence > 0.0);
    }

    #[test]
    fn test_autocorrelation_at() {
        let frame = vec![1.0, 0.5, 1.0, 0.5, 1.0, 0.5];
        let ac0 = autocorrelation_at(&frame, 0);
        let ac2 = autocorrelation_at(&frame, 2);
        // Lag-0 should be highest
        assert!(ac0 >= ac2);
    }

    #[test]
    fn test_bits_to_bytes() {
        let bits = vec![true, true, false, false, false, false, false, false];
        assert_eq!(bits_to_bytes(&bits), vec![0b11000000]);
    }
}
