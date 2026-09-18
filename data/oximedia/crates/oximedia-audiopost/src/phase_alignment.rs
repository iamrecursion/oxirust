#![allow(dead_code)]
//! Multi-microphone phase alignment and correction for audio post-production.
//!
//! When recording with multiple microphones, small differences in distance to the
//! source create phase offsets. This module provides tools to detect and correct
//! these offsets, improving clarity and avoiding comb-filtering artifacts.

use std::collections::HashMap;

/// A single audio channel identified by name.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ChannelId(String);

impl ChannelId {
    /// Create a new channel identifier.
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// Get the channel name.
    pub fn name(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ChannelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Phase relationship between two channels.
#[derive(Debug, Clone, PartialEq)]
pub struct PhaseRelation {
    /// Reference channel.
    pub reference: ChannelId,
    /// Target channel to be aligned.
    pub target: ChannelId,
    /// Detected delay of target relative to reference (in samples).
    pub delay_samples: i64,
    /// Detected delay in seconds (derived from sample rate).
    pub delay_seconds: f64,
    /// Correlation coefficient (0..1) at the detected offset.
    pub correlation: f64,
    /// Whether the target is phase-inverted relative to the reference.
    pub inverted: bool,
}

impl PhaseRelation {
    /// Create a new phase relation.
    #[allow(clippy::cast_precision_loss)]
    pub fn new(
        reference: ChannelId,
        target: ChannelId,
        delay_samples: i64,
        sample_rate: f64,
        correlation: f64,
        inverted: bool,
    ) -> Self {
        let delay_seconds = delay_samples as f64 / sample_rate;
        Self {
            reference,
            target,
            delay_samples,
            delay_seconds,
            correlation,
            inverted,
        }
    }

    /// Check if the correlation is strong enough to be considered valid.
    pub fn is_confident(&self, threshold: f64) -> bool {
        self.correlation >= threshold
    }

    /// Estimated distance offset in meters (assuming speed of sound = 343 m/s).
    pub fn distance_offset_meters(&self) -> f64 {
        self.delay_seconds.abs() * 343.0
    }
}

/// Cross-correlation result for a pair of signals.
#[derive(Debug, Clone)]
pub struct CrossCorrelation {
    /// The lag values (in samples).
    pub lags: Vec<i64>,
    /// The normalized correlation values corresponding to each lag.
    pub values: Vec<f64>,
}

impl CrossCorrelation {
    /// Compute cross-correlation between two signals.
    ///
    /// Uses a simple time-domain cross-correlation up to `max_lag` samples.
    #[allow(clippy::cast_precision_loss)]
    pub fn compute(signal_a: &[f64], signal_b: &[f64], max_lag: usize) -> Self {
        let n = signal_a.len().min(signal_b.len());
        if n == 0 {
            return Self {
                lags: vec![0],
                values: vec![0.0],
            };
        }

        let energy_a: f64 = signal_a.iter().take(n).map(|x| x * x).sum();
        let energy_b: f64 = signal_b.iter().take(n).map(|x| x * x).sum();
        let norm = (energy_a * energy_b).sqrt().max(1e-30);

        let mut lags = Vec::new();
        let mut values = Vec::new();

        let max = max_lag.min(n.saturating_sub(1));
        for lag_i in 0..=max {
            let lag = lag_i as i64;
            // Positive lag: signal_b is delayed.
            let corr_pos = compute_corr_at_lag(signal_a, signal_b, n, lag_i) / norm;
            lags.push(lag);
            values.push(corr_pos);

            if lag_i > 0 {
                // Negative lag: signal_a is delayed.
                let corr_neg = compute_corr_at_lag(signal_b, signal_a, n, lag_i) / norm;
                lags.push(-lag);
                values.push(corr_neg);
            }
        }

        Self { lags, values }
    }

    /// Find the lag with the highest absolute correlation.
    pub fn peak(&self) -> (i64, f64) {
        self.lags
            .iter()
            .zip(self.values.iter())
            .max_by(|(_, a), (_, b)| {
                a.abs()
                    .partial_cmp(&b.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(&lag, &val)| (lag, val))
            .unwrap_or((0, 0.0))
    }
}

/// Helper: compute normalized cross-correlation at a specific lag.
fn compute_corr_at_lag(a: &[f64], b: &[f64], n: usize, lag: usize) -> f64 {
    let mut sum = 0.0;
    let end = n.saturating_sub(lag);
    for i in 0..end {
        sum += a[i] * b[i + lag];
    }
    sum
}

/// Phase alignment engine that processes multiple channels.
#[derive(Debug)]
pub struct PhaseAligner {
    /// Sample rate of all channels.
    pub sample_rate: f64,
    /// Maximum search window in samples.
    pub max_lag: usize,
    /// Confidence threshold for accepting a detected offset.
    pub confidence_threshold: f64,
    /// Detected phase relations.
    relations: Vec<PhaseRelation>,
}

impl PhaseAligner {
    /// Create a new phase aligner.
    pub fn new(sample_rate: f64, max_lag: usize) -> Self {
        Self {
            sample_rate,
            max_lag,
            confidence_threshold: 0.5,
            relations: Vec::new(),
        }
    }

    /// Set the confidence threshold.
    pub fn with_confidence(mut self, threshold: f64) -> Self {
        self.confidence_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Detect phase relation between a reference and target signal.
    pub fn detect(
        &mut self,
        reference_id: ChannelId,
        target_id: ChannelId,
        reference_signal: &[f64],
        target_signal: &[f64],
    ) -> PhaseRelation {
        let xcorr = CrossCorrelation::compute(reference_signal, target_signal, self.max_lag);
        let (peak_lag, peak_val) = xcorr.peak();
        let inverted = peak_val < 0.0;
        let correlation = peak_val.abs();

        let relation = PhaseRelation::new(
            reference_id,
            target_id,
            peak_lag,
            self.sample_rate,
            correlation,
            inverted,
        );

        self.relations.push(relation.clone());
        relation
    }

    /// Return all detected phase relations.
    pub fn relations(&self) -> &[PhaseRelation] {
        &self.relations
    }

    /// Clear all detected relations.
    pub fn clear(&mut self) {
        self.relations.clear();
    }

    /// Apply detected delays to a set of signals, returning aligned signals.
    ///
    /// The `signals` map must contain entries for each `target` in the detected
    /// relations. Signals are zero-padded as needed.
    pub fn apply_corrections(
        &self,
        signals: &HashMap<ChannelId, Vec<f64>>,
    ) -> HashMap<ChannelId, Vec<f64>> {
        let mut result: HashMap<ChannelId, Vec<f64>> = signals.clone();

        for rel in &self.relations {
            if !rel.is_confident(self.confidence_threshold) {
                continue;
            }
            if let Some(signal) = result.get(&rel.target).cloned() {
                let corrected = apply_delay(&signal, -rel.delay_samples, rel.inverted);
                result.insert(rel.target.clone(), corrected);
            }
        }

        result
    }
}

/// Apply a sample delay (and optional inversion) to a signal.
fn apply_delay(signal: &[f64], delay: i64, invert: bool) -> Vec<f64> {
    let n = signal.len();
    let mut output = vec![0.0; n];
    let sign = if invert { -1.0 } else { 1.0 };

    for (i, out_sample) in output.iter_mut().enumerate() {
        let src_idx = i as i64 - delay;
        if src_idx >= 0 && (src_idx as usize) < n {
            *out_sample = signal[src_idx as usize] * sign;
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_id() {
        let ch = ChannelId::new("boom_mic");
        assert_eq!(ch.name(), "boom_mic");
        assert_eq!(ch.to_string(), "boom_mic");
    }

    #[test]
    fn test_phase_relation_distance() {
        let rel = PhaseRelation::new(
            ChannelId::new("a"),
            ChannelId::new("b"),
            48,
            48000.0,
            0.95,
            false,
        );
        // 48 samples at 48kHz = 1ms = 0.343m
        let dist = rel.distance_offset_meters();
        assert!((dist - 0.343).abs() < 0.01);
    }

    #[test]
    fn test_phase_relation_confidence() {
        let rel = PhaseRelation::new(
            ChannelId::new("a"),
            ChannelId::new("b"),
            10,
            48000.0,
            0.8,
            false,
        );
        assert!(rel.is_confident(0.5));
        assert!(!rel.is_confident(0.9));
    }

    #[test]
    fn test_cross_correlation_identical() {
        let signal: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let xcorr = CrossCorrelation::compute(&signal, &signal, 10);
        let (lag, val) = xcorr.peak();
        assert_eq!(lag, 0);
        assert!((val - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_cross_correlation_delayed() {
        let n = 200;
        let delay = 5;
        let original: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin()).collect();
        let mut delayed = vec![0.0; n];
        for i in delay..n {
            delayed[i] = original[i - delay];
        }

        let xcorr = CrossCorrelation::compute(&original, &delayed, 20);
        let (lag, _) = xcorr.peak();
        assert_eq!(lag, delay as i64);
    }

    #[test]
    fn test_cross_correlation_empty() {
        let xcorr = CrossCorrelation::compute(&[], &[], 10);
        let (lag, val) = xcorr.peak();
        assert_eq!(lag, 0);
        assert_eq!(val, 0.0);
    }

    #[test]
    fn test_phase_aligner_detect() {
        let signal: Vec<f64> = (0..500).map(|i| (i as f64 * 0.2).sin()).collect();
        let mut aligner = PhaseAligner::new(48000.0, 50);
        let rel = aligner.detect(
            ChannelId::new("ref"),
            ChannelId::new("target"),
            &signal,
            &signal,
        );
        assert_eq!(rel.delay_samples, 0);
        assert!(rel.correlation > 0.9);
    }

    #[test]
    fn test_phase_aligner_relations() {
        let signal: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let mut aligner = PhaseAligner::new(48000.0, 10);
        aligner.detect(ChannelId::new("a"), ChannelId::new("b"), &signal, &signal);
        assert_eq!(aligner.relations().len(), 1);
        aligner.clear();
        assert_eq!(aligner.relations().len(), 0);
    }

    #[test]
    fn test_apply_delay_no_invert() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = apply_delay(&signal, 2, false);
        // delay = 2 means src_idx = i - 2
        // i=0: src=-2 -> 0, i=1: src=-1 -> 0, i=2: src=0 -> 1.0, ...
        assert_eq!(result, vec![0.0, 0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_apply_delay_with_invert() {
        let signal = vec![1.0, 2.0, 3.0];
        let result = apply_delay(&signal, 0, true);
        assert_eq!(result, vec![-1.0, -2.0, -3.0]);
    }

    #[test]
    fn test_apply_corrections() {
        let signal: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let mut aligner = PhaseAligner::new(48000.0, 10).with_confidence(0.0);
        aligner.detect(
            ChannelId::new("ref"),
            ChannelId::new("target"),
            &signal,
            &signal,
        );

        let mut signals = HashMap::new();
        signals.insert(ChannelId::new("ref"), signal.clone());
        signals.insert(ChannelId::new("target"), signal.clone());

        let aligned = aligner.apply_corrections(&signals);
        assert_eq!(aligned.len(), 2);
        assert!(aligned.contains_key(&ChannelId::new("target")));
    }

    #[test]
    fn test_confidence_threshold() {
        let mut aligner = PhaseAligner::new(48000.0, 10).with_confidence(0.99);
        // Very weak correlation.
        let a: Vec<f64> = (0..50).map(|i| (i as f64 * 0.1).sin()).collect();
        let b: Vec<f64> = (0..50).map(|i| (i as f64 * 0.7).cos()).collect();
        aligner.detect(ChannelId::new("ref"), ChannelId::new("t"), &a, &b);

        let mut signals = HashMap::new();
        signals.insert(ChannelId::new("t"), b.clone());

        // With high threshold, weak relations should not be applied.
        let aligned = aligner.apply_corrections(&signals);
        // Signal should remain unchanged because correlation is below threshold.
        assert_eq!(aligned[&ChannelId::new("t")], b);
    }

    #[test]
    fn test_phase_relation_inverted() {
        let n = 200;
        let original: Vec<f64> = (0..n).map(|i| (i as f64 * 0.2).sin()).collect();
        let inverted: Vec<f64> = original.iter().map(|x| -x).collect();

        let mut aligner = PhaseAligner::new(48000.0, 10);
        let rel = aligner.detect(
            ChannelId::new("ref"),
            ChannelId::new("inv"),
            &original,
            &inverted,
        );
        assert!(rel.inverted);
    }
}
