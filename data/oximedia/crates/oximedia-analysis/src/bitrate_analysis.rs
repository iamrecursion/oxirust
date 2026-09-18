#![allow(dead_code)]

//! Bitrate distribution and encoding complexity analysis.
//!
//! This module provides tools for analyzing bitrate patterns across video
//! segments, detecting encoding complexity spikes, and computing statistical
//! summaries of bitrate usage. It is useful for evaluating encoder performance,
//! planning ABR ladder decisions, and detecting problematic encoding regions.

use std::collections::VecDeque;

/// Minimum number of samples required for statistical analysis.
const MIN_SAMPLES_FOR_STATS: usize = 2;

/// A single bitrate measurement for one frame or chunk.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BitrateSample {
    /// Frame index or chunk index.
    pub index: u64,
    /// Measured bitrate in bits per second.
    pub bitrate_bps: f64,
    /// Whether the frame is a keyframe (I-frame).
    pub is_keyframe: bool,
}

impl BitrateSample {
    /// Create a new bitrate sample.
    pub fn new(index: u64, bitrate_bps: f64, is_keyframe: bool) -> Self {
        Self {
            index,
            bitrate_bps,
            is_keyframe,
        }
    }
}

/// Aggregate statistics for a set of bitrate samples.
#[derive(Debug, Clone, PartialEq)]
pub struct BitrateStats {
    /// Number of samples.
    pub count: usize,
    /// Mean bitrate in bps.
    pub mean_bps: f64,
    /// Minimum bitrate in bps.
    pub min_bps: f64,
    /// Maximum bitrate in bps.
    pub max_bps: f64,
    /// Standard deviation of bitrate in bps.
    pub std_dev_bps: f64,
    /// Coefficient of variation (std_dev / mean).
    pub coefficient_of_variation: f64,
    /// Median bitrate in bps.
    pub median_bps: f64,
    /// 95th percentile bitrate in bps.
    pub p95_bps: f64,
}

impl Default for BitrateStats {
    fn default() -> Self {
        Self {
            count: 0,
            mean_bps: 0.0,
            min_bps: 0.0,
            max_bps: 0.0,
            std_dev_bps: 0.0,
            coefficient_of_variation: 0.0,
            median_bps: 0.0,
            p95_bps: 0.0,
        }
    }
}

/// A detected bitrate spike region.
#[derive(Debug, Clone, PartialEq)]
pub struct BitrateSpike {
    /// Starting frame index.
    pub start_index: u64,
    /// Ending frame index.
    pub end_index: u64,
    /// Peak bitrate in the spike region (bps).
    pub peak_bps: f64,
    /// Ratio of peak to the overall mean bitrate.
    pub ratio_to_mean: f64,
}

/// Classification of bitrate variability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitrateVariability {
    /// Low variability, CBR-like behavior.
    Constant,
    /// Moderate variability, typical VBR.
    Moderate,
    /// High variability, aggressive VBR or scene-complexity-driven.
    High,
    /// Extreme variability, may indicate encoding issues.
    Extreme,
}

impl BitrateVariability {
    /// Classify variability from the coefficient of variation.
    #[allow(clippy::cast_precision_loss)]
    pub fn from_cv(cv: f64) -> Self {
        if cv < 0.1 {
            Self::Constant
        } else if cv < 0.3 {
            Self::Moderate
        } else if cv < 0.6 {
            Self::High
        } else {
            Self::Extreme
        }
    }

    /// Returns a human-readable description.
    pub fn description(self) -> &'static str {
        match self {
            Self::Constant => "constant (CBR-like)",
            Self::Moderate => "moderate (typical VBR)",
            Self::High => "high (aggressive VBR)",
            Self::Extreme => "extreme (potential encoding issues)",
        }
    }
}

/// Analyzer for bitrate distribution and complexity.
#[derive(Debug)]
pub struct BitrateAnalyzer {
    /// All collected samples.
    samples: Vec<BitrateSample>,
    /// Sliding window for running statistics.
    window: VecDeque<f64>,
    /// Window size for running average.
    window_size: usize,
    /// Threshold multiplier for spike detection (times the mean).
    spike_threshold: f64,
}

impl BitrateAnalyzer {
    /// Create a new bitrate analyzer.
    ///
    /// `window_size` controls the sliding window for running stats.
    /// `spike_threshold` is a multiplier of the mean (e.g. 2.0 means spikes
    /// are detected when bitrate exceeds 2x the mean).
    pub fn new(window_size: usize, spike_threshold: f64) -> Self {
        let ws = if window_size == 0 { 30 } else { window_size };
        let st = if spike_threshold <= 0.0 {
            2.0
        } else {
            spike_threshold
        };
        Self {
            samples: Vec::new(),
            window: VecDeque::with_capacity(ws),
            window_size: ws,
            spike_threshold: st,
        }
    }

    /// Create with default settings (window=30, spike_threshold=2.0).
    pub fn with_defaults() -> Self {
        Self::new(30, 2.0)
    }

    /// Add a bitrate sample.
    pub fn add_sample(&mut self, sample: BitrateSample) {
        self.samples.push(sample);
        self.window.push_back(sample.bitrate_bps);
        if self.window.len() > self.window_size {
            self.window.pop_front();
        }
    }

    /// Get the current running average bitrate.
    #[allow(clippy::cast_precision_loss)]
    pub fn running_average(&self) -> f64 {
        if self.window.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.window.iter().sum();
        sum / self.window.len() as f64
    }

    /// Compute overall statistics from all samples.
    #[allow(clippy::cast_precision_loss)]
    pub fn compute_stats(&self) -> BitrateStats {
        if self.samples.is_empty() {
            return BitrateStats::default();
        }

        let count = self.samples.len();
        let values: Vec<f64> = self.samples.iter().map(|s| s.bitrate_bps).collect();

        let sum: f64 = values.iter().sum();
        let mean = sum / count as f64;

        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        let variance = if count >= MIN_SAMPLES_FOR_STATS {
            values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (count - 1) as f64
        } else {
            0.0
        };
        let std_dev = variance.sqrt();

        let cv = if mean > 0.0 { std_dev / mean } else { 0.0 };

        let mut sorted = values;
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median = if count % 2 == 0 {
            (sorted[count / 2 - 1] + sorted[count / 2]) / 2.0
        } else {
            sorted[count / 2]
        };

        let p95_idx = ((count as f64 * 0.95).ceil() as usize).min(count) - 1;
        let p95 = sorted[p95_idx];

        BitrateStats {
            count,
            mean_bps: mean,
            min_bps: min,
            max_bps: max,
            std_dev_bps: std_dev,
            coefficient_of_variation: cv,
            median_bps: median,
            p95_bps: p95,
        }
    }

    /// Compute separate statistics for keyframes and non-keyframes.
    #[allow(clippy::cast_precision_loss)]
    pub fn compute_keyframe_stats(&self) -> (BitrateStats, BitrateStats) {
        let keyframes: Vec<BitrateSample> = self
            .samples
            .iter()
            .filter(|s| s.is_keyframe)
            .copied()
            .collect();
        let non_keyframes: Vec<BitrateSample> = self
            .samples
            .iter()
            .filter(|s| !s.is_keyframe)
            .copied()
            .collect();

        let kf_analyzer = {
            let mut a = BitrateAnalyzer::with_defaults();
            for s in &keyframes {
                a.add_sample(*s);
            }
            a
        };

        let nkf_analyzer = {
            let mut a = BitrateAnalyzer::with_defaults();
            for s in &non_keyframes {
                a.add_sample(*s);
            }
            a
        };

        (kf_analyzer.compute_stats(), nkf_analyzer.compute_stats())
    }

    /// Detect bitrate spikes (regions where bitrate exceeds threshold * mean).
    #[allow(clippy::cast_precision_loss)]
    pub fn detect_spikes(&self) -> Vec<BitrateSpike> {
        let stats = self.compute_stats();
        if stats.count == 0 || stats.mean_bps <= 0.0 {
            return Vec::new();
        }

        let threshold = stats.mean_bps * self.spike_threshold;
        let mut spikes = Vec::new();
        let mut in_spike = false;
        let mut start_idx: u64 = 0;
        let mut peak: f64 = 0.0;

        for sample in &self.samples {
            if sample.bitrate_bps > threshold {
                if !in_spike {
                    in_spike = true;
                    start_idx = sample.index;
                    peak = sample.bitrate_bps;
                } else if sample.bitrate_bps > peak {
                    peak = sample.bitrate_bps;
                }
            } else if in_spike {
                spikes.push(BitrateSpike {
                    start_index: start_idx,
                    end_index: sample.index.saturating_sub(1),
                    peak_bps: peak,
                    ratio_to_mean: peak / stats.mean_bps,
                });
                in_spike = false;
            }
        }

        // Close any trailing spike.
        if in_spike {
            if let Some(last) = self.samples.last() {
                spikes.push(BitrateSpike {
                    start_index: start_idx,
                    end_index: last.index,
                    peak_bps: peak,
                    ratio_to_mean: peak / stats.mean_bps,
                });
            }
        }

        spikes
    }

    /// Classify the overall bitrate variability.
    pub fn classify_variability(&self) -> BitrateVariability {
        let stats = self.compute_stats();
        BitrateVariability::from_cv(stats.coefficient_of_variation)
    }

    /// Compute a histogram of bitrate values with the given number of bins.
    #[allow(clippy::cast_precision_loss)]
    pub fn bitrate_histogram(&self, num_bins: usize) -> Vec<(f64, f64, usize)> {
        if self.samples.is_empty() || num_bins == 0 {
            return Vec::new();
        }

        let stats = self.compute_stats();
        let range = stats.max_bps - stats.min_bps;
        if range <= 0.0 {
            return vec![(stats.min_bps, stats.max_bps, self.samples.len())];
        }

        let bin_width = range / num_bins as f64;
        let mut bins = vec![0usize; num_bins];

        for sample in &self.samples {
            let bin = ((sample.bitrate_bps - stats.min_bps) / bin_width).floor() as usize;
            let bin = bin.min(num_bins - 1);
            bins[bin] += 1;
        }

        bins.iter()
            .enumerate()
            .map(|(i, &count)| {
                let low = stats.min_bps + i as f64 * bin_width;
                let high = low + bin_width;
                (low, high, count)
            })
            .collect()
    }

    /// Return the total number of samples.
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Clear all samples.
    pub fn clear(&mut self) {
        self.samples.clear();
        self.window.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sample(index: u64, bps: f64, kf: bool) -> BitrateSample {
        BitrateSample::new(index, bps, kf)
    }

    #[test]
    fn test_sample_creation() {
        let s = BitrateSample::new(0, 5_000_000.0, true);
        assert_eq!(s.index, 0);
        assert!((s.bitrate_bps - 5_000_000.0).abs() < 0.1);
        assert!(s.is_keyframe);
    }

    #[test]
    fn test_empty_stats() {
        let analyzer = BitrateAnalyzer::with_defaults();
        let stats = analyzer.compute_stats();
        assert_eq!(stats.count, 0);
        assert!((stats.mean_bps).abs() < f64::EPSILON);
    }

    #[test]
    fn test_single_sample_stats() {
        let mut analyzer = BitrateAnalyzer::with_defaults();
        analyzer.add_sample(make_sample(0, 1_000_000.0, false));
        let stats = analyzer.compute_stats();
        assert_eq!(stats.count, 1);
        assert!((stats.mean_bps - 1_000_000.0).abs() < 0.1);
        assert!((stats.median_bps - 1_000_000.0).abs() < 0.1);
    }

    #[test]
    fn test_mean_and_median() {
        let mut analyzer = BitrateAnalyzer::with_defaults();
        for i in 0..5 {
            analyzer.add_sample(make_sample(i, (i as f64 + 1.0) * 1_000_000.0, false));
        }
        let stats = analyzer.compute_stats();
        assert_eq!(stats.count, 5);
        assert!((stats.mean_bps - 3_000_000.0).abs() < 0.1);
        assert!((stats.median_bps - 3_000_000.0).abs() < 0.1);
    }

    #[test]
    fn test_min_max() {
        let mut analyzer = BitrateAnalyzer::with_defaults();
        analyzer.add_sample(make_sample(0, 100.0, false));
        analyzer.add_sample(make_sample(1, 500.0, false));
        analyzer.add_sample(make_sample(2, 300.0, false));
        let stats = analyzer.compute_stats();
        assert!((stats.min_bps - 100.0).abs() < 0.1);
        assert!((stats.max_bps - 500.0).abs() < 0.1);
    }

    #[test]
    fn test_std_dev() {
        let mut analyzer = BitrateAnalyzer::with_defaults();
        // All same value => std_dev ~= 0
        for i in 0..10 {
            analyzer.add_sample(make_sample(i, 1000.0, false));
        }
        let stats = analyzer.compute_stats();
        assert!(stats.std_dev_bps < 0.001);
        assert!(stats.coefficient_of_variation < 0.001);
    }

    #[test]
    fn test_running_average() {
        let mut analyzer = BitrateAnalyzer::new(3, 2.0);
        analyzer.add_sample(make_sample(0, 100.0, false));
        analyzer.add_sample(make_sample(1, 200.0, false));
        analyzer.add_sample(make_sample(2, 300.0, false));
        assert!((analyzer.running_average() - 200.0).abs() < 0.1);
        // Adding a 4th should evict the first (100)
        analyzer.add_sample(make_sample(3, 400.0, false));
        assert!((analyzer.running_average() - 300.0).abs() < 0.1);
    }

    #[test]
    fn test_spike_detection() {
        let mut analyzer = BitrateAnalyzer::new(30, 2.0);
        // 10 samples at ~1000, then a spike at 5000
        for i in 0..10 {
            analyzer.add_sample(make_sample(i, 1000.0, false));
        }
        analyzer.add_sample(make_sample(10, 5000.0, false));
        analyzer.add_sample(make_sample(11, 1000.0, false));
        let spikes = analyzer.detect_spikes();
        assert!(!spikes.is_empty());
        assert_eq!(spikes[0].start_index, 10);
    }

    #[test]
    fn test_no_spikes_on_constant() {
        let mut analyzer = BitrateAnalyzer::new(30, 2.0);
        for i in 0..20 {
            analyzer.add_sample(make_sample(i, 1000.0, false));
        }
        let spikes = analyzer.detect_spikes();
        assert!(spikes.is_empty());
    }

    #[test]
    fn test_variability_constant() {
        let mut analyzer = BitrateAnalyzer::with_defaults();
        for i in 0..50 {
            analyzer.add_sample(make_sample(i, 1000.0, false));
        }
        assert_eq!(
            analyzer.classify_variability(),
            BitrateVariability::Constant
        );
    }

    #[test]
    fn test_variability_description() {
        assert_eq!(
            BitrateVariability::Constant.description(),
            "constant (CBR-like)"
        );
        assert_eq!(
            BitrateVariability::Extreme.description(),
            "extreme (potential encoding issues)"
        );
    }

    #[test]
    fn test_keyframe_stats() {
        let mut analyzer = BitrateAnalyzer::with_defaults();
        analyzer.add_sample(make_sample(0, 5000.0, true));
        analyzer.add_sample(make_sample(1, 1000.0, false));
        analyzer.add_sample(make_sample(2, 1200.0, false));
        analyzer.add_sample(make_sample(3, 4800.0, true));
        let (kf, nkf) = analyzer.compute_keyframe_stats();
        assert_eq!(kf.count, 2);
        assert_eq!(nkf.count, 2);
        assert!(kf.mean_bps > nkf.mean_bps);
    }

    #[test]
    fn test_histogram() {
        let mut analyzer = BitrateAnalyzer::with_defaults();
        for i in 0..100 {
            analyzer.add_sample(make_sample(i, (i as f64) * 10.0, false));
        }
        let hist = analyzer.bitrate_histogram(10);
        assert_eq!(hist.len(), 10);
        let total: usize = hist.iter().map(|(_, _, c)| c).sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn test_clear() {
        let mut analyzer = BitrateAnalyzer::with_defaults();
        analyzer.add_sample(make_sample(0, 1000.0, false));
        assert_eq!(analyzer.sample_count(), 1);
        analyzer.clear();
        assert_eq!(analyzer.sample_count(), 0);
    }
}
