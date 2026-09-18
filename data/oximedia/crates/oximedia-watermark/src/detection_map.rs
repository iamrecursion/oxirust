//! Watermark detection map: spatial detection confidence and threshold tuning.
//!
//! Provides:
//! - Per-segment confidence scores for spatial detection
//! - Confidence map aggregation and peak detection
//! - Threshold tuning via ROC curve analysis
//! - Overlap-add detection window

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// A single detection cell: one time/frequency segment.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct DetectionCell {
    /// Start sample index of this segment.
    pub start: usize,
    /// End sample index (exclusive).
    pub end: usize,
    /// Raw correlation value.
    pub correlation: f64,
    /// Normalized confidence score (0.0–1.0).
    pub confidence: f64,
    /// Whether this cell exceeds the detection threshold.
    pub detected: bool,
}

/// Configuration for detection map computation.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct DetectionMapConfig {
    /// Number of samples per detection segment.
    pub segment_size: usize,
    /// Overlap between consecutive segments (0.0–1.0).
    pub overlap: f64,
    /// Detection threshold for confidence score.
    pub threshold: f64,
    /// Smoothing window length for confidence map (cells).
    pub smooth_window: usize,
}

impl Default for DetectionMapConfig {
    fn default() -> Self {
        Self {
            segment_size: 2048,
            overlap: 0.5,
            threshold: 0.6,
            smooth_window: 3,
        }
    }
}

/// Detection map: a sequence of per-segment detection cells.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct DetectionMap {
    /// Ordered cells covering the audio signal.
    pub cells: Vec<DetectionCell>,
    /// Configuration used to build this map.
    pub config: DetectionMapConfig,
}

impl DetectionMap {
    /// Build a detection map from correlation scores.
    ///
    /// `correlations` contains one raw correlation value per segment.
    /// Confidence is computed by normalizing against the maximum observed correlation.
    #[must_use]
    pub fn from_correlations(correlations: &[f64], config: DetectionMapConfig) -> Self {
        if correlations.is_empty() {
            return Self {
                cells: Vec::new(),
                config,
            };
        }

        let hop = ((1.0 - config.overlap) * config.segment_size as f64) as usize;
        let hop = hop.max(1);

        let max_corr = correlations
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
            .max(1e-12);

        let cells: Vec<DetectionCell> = correlations
            .iter()
            .enumerate()
            .map(|(i, &corr)| {
                let start = i * hop;
                let end = start + config.segment_size;
                let confidence = (corr / max_corr).clamp(0.0, 1.0);
                DetectionCell {
                    start,
                    end,
                    correlation: corr,
                    confidence,
                    detected: corr >= config.threshold,
                }
            })
            .collect();

        Self { cells, config }
    }

    /// Apply a smoothing window (box filter) to confidence scores.
    #[must_use]
    pub fn smoothed_confidence(&self) -> Vec<f64> {
        let n = self.cells.len();
        let w = self.config.smooth_window.max(1);
        (0..n)
            .map(|i| {
                let start = i.saturating_sub(w / 2);
                let end = (i + w / 2 + 1).min(n);
                let sum: f64 = self.cells[start..end].iter().map(|c| c.confidence).sum();
                sum / (end - start) as f64
            })
            .collect()
    }

    /// Find peaks in the confidence map.
    ///
    /// A cell is a peak if its confidence exceeds both neighbours.
    #[must_use]
    pub fn find_peaks(&self) -> Vec<usize> {
        let smooth = self.smoothed_confidence();
        let n = smooth.len();
        if n < 3 {
            return (0..n)
                .filter(|&i| smooth[i] >= self.config.threshold)
                .collect();
        }
        (1..n - 1)
            .filter(|&i| {
                smooth[i] > smooth[i - 1]
                    && smooth[i] > smooth[i + 1]
                    && smooth[i] >= self.config.threshold
            })
            .collect()
    }

    /// Fraction of cells that are above the detection threshold.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn detection_rate(&self) -> f64 {
        if self.cells.is_empty() {
            return 0.0;
        }
        let detected = self.cells.iter().filter(|c| c.detected).count();
        detected as f64 / self.cells.len() as f64
    }

    /// Maximum confidence score across all cells.
    #[must_use]
    pub fn max_confidence(&self) -> f64 {
        self.cells
            .iter()
            .map(|c| c.confidence)
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Mean confidence score across all cells.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn mean_confidence(&self) -> f64 {
        if self.cells.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.cells.iter().map(|c| c.confidence).sum();
        sum / self.cells.len() as f64
    }

    /// Tune the detection threshold using a simple target false-positive rate.
    ///
    /// Given a set of "noise-only" confidence scores (no watermark),
    /// returns the threshold that keeps the false-positive rate below `max_fpr`.
    #[must_use]
    pub fn tune_threshold(noise_scores: &[f64], max_fpr: f64) -> f64 {
        if noise_scores.is_empty() {
            return 0.5;
        }
        let mut sorted = noise_scores.to_vec();
        sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        // Threshold = percentile at (1 - max_fpr)
        let idx = ((1.0 - max_fpr) * sorted.len() as f64) as usize;
        let idx = idx.min(sorted.len() - 1);
        sorted[idx]
    }

    /// Number of cells.
    #[must_use]
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    /// Whether the map has no cells.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }
}

/// Compute correlation between a reference sequence and a signal segment.
///
/// Returns the normalized cross-correlation peak value.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn normalized_xcorr(reference: &[f64], segment: &[f64]) -> f64 {
    let n = reference.len().min(segment.len());
    if n == 0 {
        return 0.0;
    }
    let dot: f64 = reference[..n]
        .iter()
        .zip(segment[..n].iter())
        .map(|(&a, &b)| a * b)
        .sum();
    let ref_norm: f64 = reference[..n].iter().map(|&a| a * a).sum::<f64>().sqrt();
    let seg_norm: f64 = segment[..n].iter().map(|&b| b * b).sum::<f64>().sqrt();
    let denom = ref_norm * seg_norm;
    if denom < 1e-12 {
        0.0
    } else {
        (dot / denom).clamp(-1.0, 1.0)
    }
}

/// Split a sample array into overlapping segments.
#[must_use]
pub fn split_into_segments(samples: &[f32], segment_size: usize, overlap: f64) -> Vec<Vec<f32>> {
    if segment_size == 0 || samples.is_empty() {
        return Vec::new();
    }
    let hop = ((1.0 - overlap) * segment_size as f64).max(1.0) as usize;
    let mut segments = Vec::new();
    let mut pos = 0usize;
    while pos + segment_size <= samples.len() {
        segments.push(samples[pos..pos + segment_size].to_vec());
        pos += hop;
    }
    segments
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detection_map_empty_correlations() {
        let config = DetectionMapConfig::default();
        let map = DetectionMap::from_correlations(&[], config);
        assert!(map.is_empty());
    }

    #[test]
    fn test_detection_map_cell_count() {
        let correlations = vec![0.1, 0.5, 0.8, 0.3, 0.9];
        let config = DetectionMapConfig::default();
        let map = DetectionMap::from_correlations(&correlations, config);
        assert_eq!(map.len(), 5);
    }

    #[test]
    fn test_detection_map_max_confidence_is_one() {
        let correlations = vec![0.2, 0.6, 1.0, 0.4];
        let config = DetectionMapConfig::default();
        let map = DetectionMap::from_correlations(&correlations, config);
        assert!((map.max_confidence() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_detection_map_confidence_normalized() {
        let correlations = vec![0.5, 1.0, 2.0];
        let config = DetectionMapConfig::default();
        let map = DetectionMap::from_correlations(&correlations, config);
        for cell in &map.cells {
            assert!(cell.confidence >= 0.0 && cell.confidence <= 1.0);
        }
    }

    #[test]
    fn test_detection_map_threshold_applied() {
        let correlations = vec![0.1, 0.5, 0.9, 0.3];
        let config = DetectionMapConfig {
            threshold: 0.6,
            ..Default::default()
        };
        let map = DetectionMap::from_correlations(&correlations, config);
        // max = 0.9, conf = [0.11, 0.55, 1.0, 0.33] → only cell[2] detected
        assert!(map.cells[2].detected);
        assert!(!map.cells[0].detected);
        assert!(!map.cells[3].detected);
    }

    #[test]
    fn test_detection_rate_all_detected() {
        let correlations = vec![1.0, 1.0, 1.0];
        let config = DetectionMapConfig {
            threshold: 0.5,
            ..Default::default()
        };
        let map = DetectionMap::from_correlations(&correlations, config);
        assert!((map.detection_rate() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_detection_rate_none_detected() {
        let correlations = vec![0.1, 0.1, 0.1];
        let config = DetectionMapConfig {
            threshold: 0.9,
            ..Default::default()
        };
        let map = DetectionMap::from_correlations(&correlations, config);
        assert_eq!(map.detection_rate(), 0.0);
    }

    #[test]
    fn test_mean_confidence_empty() {
        let config = DetectionMapConfig::default();
        let map = DetectionMap::from_correlations(&[], config);
        assert_eq!(map.mean_confidence(), 0.0);
    }

    #[test]
    fn test_mean_confidence_uniform() {
        let correlations = vec![0.5, 0.5, 0.5, 0.5];
        let config = DetectionMapConfig::default();
        let map = DetectionMap::from_correlations(&correlations, config);
        assert!((map.mean_confidence() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_smoothed_confidence_length() {
        let correlations = vec![0.2, 0.5, 0.8, 0.3, 0.6, 0.9, 0.1];
        let config = DetectionMapConfig::default();
        let map = DetectionMap::from_correlations(&correlations, config);
        let smooth = map.smoothed_confidence();
        assert_eq!(smooth.len(), correlations.len());
    }

    #[test]
    fn test_find_peaks_single_peak() {
        let correlations = vec![0.1, 0.3, 1.0, 0.3, 0.1];
        let config = DetectionMapConfig {
            threshold: 0.5,
            smooth_window: 1,
            ..Default::default()
        };
        let map = DetectionMap::from_correlations(&correlations, config);
        let peaks = map.find_peaks();
        assert_eq!(peaks.len(), 1);
        assert_eq!(peaks[0], 2);
    }

    #[test]
    fn test_tune_threshold_empty() {
        let t = DetectionMap::tune_threshold(&[], 0.05);
        assert!((t - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_tune_threshold_low_fpr() {
        let noise: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        let t = DetectionMap::tune_threshold(&noise, 0.05);
        // Should be near 0.95
        assert!(t >= 0.9);
    }

    #[test]
    fn test_normalized_xcorr_identical() {
        let v = vec![1.0, -1.0, 0.5, -0.5, 0.3];
        let corr = normalized_xcorr(&v, &v);
        assert!((corr - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_normalized_xcorr_orthogonal() {
        let a = vec![1.0, 0.0, 1.0, 0.0];
        let b = vec![0.0, 1.0, 0.0, 1.0];
        let corr = normalized_xcorr(&a, &b);
        assert!(corr.abs() < 1e-9);
    }

    #[test]
    fn test_normalized_xcorr_empty() {
        let corr = normalized_xcorr(&[], &[]);
        assert_eq!(corr, 0.0);
    }

    #[test]
    fn test_split_into_segments_no_overlap() {
        let samples: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let segs = split_into_segments(&samples, 10, 0.0);
        assert_eq!(segs.len(), 10);
        assert_eq!(segs[0].len(), 10);
    }

    #[test]
    fn test_split_into_segments_with_overlap() {
        let samples: Vec<f32> = vec![0.0; 200];
        let segs = split_into_segments(&samples, 20, 0.5);
        // hop = 10, so we get floor((200 - 20) / 10) + 1 = 19 segments
        assert!(segs.len() > 10);
    }

    #[test]
    fn test_split_into_segments_empty() {
        let segs = split_into_segments(&[], 10, 0.5);
        assert!(segs.is_empty());
    }
}
