#![allow(dead_code)]

//! Shot transition detection and classification utilities.
//!
//! Provides frame-level transition metrics, transition type identification,
//! and transition quality scoring for video editing workflows.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Transition types
// ---------------------------------------------------------------------------

/// Kind of transition between two shots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransitionKind {
    /// Hard cut (instantaneous change).
    Cut,
    /// Cross-dissolve (gradual blend).
    Dissolve,
    /// Fade to black or fade from black.
    Fade,
    /// Wipe (geometric pattern reveal).
    Wipe,
    /// Push (one frame pushes the other off-screen).
    Push,
    /// Morph (content-aware blend).
    Morph,
    /// Unknown or undetected transition type.
    Unknown,
}

impl TransitionKind {
    /// Return a human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            TransitionKind::Cut => "cut",
            TransitionKind::Dissolve => "dissolve",
            TransitionKind::Fade => "fade",
            TransitionKind::Wipe => "wipe",
            TransitionKind::Push => "push",
            TransitionKind::Morph => "morph",
            TransitionKind::Unknown => "unknown",
        }
    }
}

// ---------------------------------------------------------------------------
// Transition descriptor
// ---------------------------------------------------------------------------

/// Describes a detected transition between two shots.
#[derive(Debug, Clone)]
pub struct TransitionDescriptor {
    /// Frame index where the transition begins.
    pub start_frame: usize,
    /// Frame index where the transition ends.
    pub end_frame: usize,
    /// Duration of the transition in frames.
    pub duration_frames: usize,
    /// The kind of transition.
    pub kind: TransitionKind,
    /// Confidence of the classification (0.0 to 1.0).
    pub confidence: f64,
    /// Mean frame difference during the transition.
    pub mean_diff: f64,
}

impl TransitionDescriptor {
    /// Create a new transition descriptor.
    pub fn new(
        start_frame: usize,
        end_frame: usize,
        kind: TransitionKind,
        confidence: f64,
    ) -> Self {
        let duration = if end_frame > start_frame {
            end_frame - start_frame
        } else {
            1
        };
        Self {
            start_frame,
            end_frame,
            duration_frames: duration,
            kind,
            confidence,
            mean_diff: 0.0,
        }
    }

    /// Set the mean difference metric.
    pub fn with_mean_diff(mut self, mean_diff: f64) -> Self {
        self.mean_diff = mean_diff;
        self
    }

    /// Check whether this is an instantaneous cut.
    pub fn is_cut(&self) -> bool {
        self.kind == TransitionKind::Cut
    }

    /// Check whether this is a gradual transition (dissolve, fade, wipe, etc.).
    pub fn is_gradual(&self) -> bool {
        matches!(
            self.kind,
            TransitionKind::Dissolve
                | TransitionKind::Fade
                | TransitionKind::Wipe
                | TransitionKind::Push
                | TransitionKind::Morph
        )
    }
}

// ---------------------------------------------------------------------------
// Frame difference metrics
// ---------------------------------------------------------------------------

/// Compute the mean absolute difference between two frames
/// represented as flat f64 pixel arrays.
pub fn mean_absolute_diff(frame_a: &[f64], frame_b: &[f64]) -> f64 {
    if frame_a.is_empty() || frame_b.is_empty() {
        return 0.0;
    }
    let len = frame_a.len().min(frame_b.len());
    let sum: f64 = frame_a[..len]
        .iter()
        .zip(frame_b[..len].iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    sum / len as f64
}

/// Compute the histogram correlation between two grayscale histograms.
///
/// Each histogram should have 256 bins. Returns a value in [-1.0, 1.0].
pub fn histogram_correlation(hist_a: &[f64; 256], hist_b: &[f64; 256]) -> f64 {
    let mean_a: f64 = hist_a.iter().sum::<f64>() / 256.0;
    let mean_b: f64 = hist_b.iter().sum::<f64>() / 256.0;

    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;

    for i in 0..256 {
        let da = hist_a[i] - mean_a;
        let db = hist_b[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    let denom = (var_a * var_b).sqrt();
    if denom.abs() < f64::EPSILON {
        return 0.0;
    }
    cov / denom
}

// ---------------------------------------------------------------------------
// Transition classifier
// ---------------------------------------------------------------------------

/// Configuration for transition classification.
#[derive(Debug, Clone)]
pub struct TransitionClassifierConfig {
    /// Threshold for detecting a hard cut (high frame difference).
    pub cut_threshold: f64,
    /// Threshold for detecting a dissolve (moderate, sustained difference).
    pub dissolve_threshold: f64,
    /// Threshold for detecting a fade (luminance drop to near-zero).
    pub fade_threshold: f64,
    /// Minimum number of frames for a gradual transition.
    pub min_gradual_frames: usize,
}

impl Default for TransitionClassifierConfig {
    fn default() -> Self {
        Self {
            cut_threshold: 0.35,
            dissolve_threshold: 0.15,
            fade_threshold: 0.08,
            min_gradual_frames: 3,
        }
    }
}

/// Classifies transitions based on frame-level difference scores.
#[derive(Debug)]
pub struct TransitionClassifier {
    /// Configuration.
    config: TransitionClassifierConfig,
}

impl TransitionClassifier {
    /// Create a new classifier with the given config.
    pub fn new(config: TransitionClassifierConfig) -> Self {
        Self { config }
    }

    /// Create a classifier with default parameters.
    pub fn with_defaults() -> Self {
        Self {
            config: TransitionClassifierConfig::default(),
        }
    }

    /// Classify a transition given a sequence of inter-frame differences.
    ///
    /// `diffs` is a slice of frame-level difference scores (0.0 to 1.0)
    /// spanning the suspected transition region.
    pub fn classify(&self, diffs: &[f64]) -> TransitionDescriptor {
        if diffs.is_empty() {
            return TransitionDescriptor::new(0, 0, TransitionKind::Unknown, 0.0);
        }

        let max_diff = diffs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean_diff = diffs.iter().sum::<f64>() / diffs.len() as f64;

        // Single-frame spike => cut
        if diffs.len() == 1 && max_diff >= self.config.cut_threshold {
            return TransitionDescriptor::new(0, 1, TransitionKind::Cut, max_diff)
                .with_mean_diff(mean_diff);
        }

        // Multi-frame: check for fade (low mean luminance diff)
        if diffs.len() >= self.config.min_gradual_frames && mean_diff < self.config.fade_threshold {
            let confidence = 1.0 - (mean_diff / self.config.fade_threshold).min(1.0);
            return TransitionDescriptor::new(0, diffs.len(), TransitionKind::Fade, confidence)
                .with_mean_diff(mean_diff);
        }

        // Multi-frame: dissolve (moderate sustained difference)
        if diffs.len() >= self.config.min_gradual_frames
            && mean_diff >= self.config.fade_threshold
            && mean_diff < self.config.cut_threshold
        {
            let confidence = ((mean_diff - self.config.fade_threshold)
                / (self.config.cut_threshold - self.config.fade_threshold))
                .clamp(0.0, 1.0);
            return TransitionDescriptor::new(0, diffs.len(), TransitionKind::Dissolve, confidence)
                .with_mean_diff(mean_diff);
        }

        // Hard cut (spike in multi-frame window)
        if max_diff >= self.config.cut_threshold {
            return TransitionDescriptor::new(0, 1, TransitionKind::Cut, max_diff)
                .with_mean_diff(mean_diff);
        }

        TransitionDescriptor::new(0, diffs.len(), TransitionKind::Unknown, 0.0)
            .with_mean_diff(mean_diff)
    }

    /// Return the config.
    pub fn config(&self) -> &TransitionClassifierConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Transition summary
// ---------------------------------------------------------------------------

/// Summary statistics for a set of transitions.
#[derive(Debug, Clone)]
pub struct TransitionSummary {
    /// Total number of transitions.
    pub total: usize,
    /// Count per transition kind.
    pub counts: HashMap<TransitionKind, usize>,
    /// Average confidence across all transitions.
    pub avg_confidence: f64,
    /// Average transition duration in frames.
    pub avg_duration_frames: f64,
}

/// Compute a summary from a list of transition descriptors.
pub fn summarize_transitions(transitions: &[TransitionDescriptor]) -> TransitionSummary {
    if transitions.is_empty() {
        return TransitionSummary {
            total: 0,
            counts: HashMap::new(),
            avg_confidence: 0.0,
            avg_duration_frames: 0.0,
        };
    }

    let mut counts: HashMap<TransitionKind, usize> = HashMap::new();
    let mut total_conf = 0.0;
    let mut total_dur = 0usize;

    for t in transitions {
        *counts.entry(t.kind).or_insert(0) += 1;
        total_conf += t.confidence;
        total_dur += t.duration_frames;
    }

    let n = transitions.len() as f64;
    TransitionSummary {
        total: transitions.len(),
        counts,
        avg_confidence: total_conf / n,
        avg_duration_frames: total_dur as f64 / n,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- TransitionKind tests --

    #[test]
    fn test_transition_kind_label() {
        assert_eq!(TransitionKind::Cut.label(), "cut");
        assert_eq!(TransitionKind::Dissolve.label(), "dissolve");
        assert_eq!(TransitionKind::Fade.label(), "fade");
        assert_eq!(TransitionKind::Unknown.label(), "unknown");
    }

    // -- TransitionDescriptor tests --

    #[test]
    fn test_descriptor_new() {
        let d = TransitionDescriptor::new(10, 15, TransitionKind::Dissolve, 0.8);
        assert_eq!(d.start_frame, 10);
        assert_eq!(d.end_frame, 15);
        assert_eq!(d.duration_frames, 5);
        assert!((d.confidence - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_descriptor_is_cut() {
        let d = TransitionDescriptor::new(0, 1, TransitionKind::Cut, 0.9);
        assert!(d.is_cut());
        assert!(!d.is_gradual());
    }

    #[test]
    fn test_descriptor_is_gradual() {
        let d = TransitionDescriptor::new(0, 10, TransitionKind::Dissolve, 0.7);
        assert!(!d.is_cut());
        assert!(d.is_gradual());
    }

    #[test]
    fn test_descriptor_with_mean_diff() {
        let d = TransitionDescriptor::new(0, 5, TransitionKind::Fade, 0.6).with_mean_diff(0.12);
        assert!((d.mean_diff - 0.12).abs() < f64::EPSILON);
    }

    // -- Frame difference tests --

    #[test]
    fn test_mean_absolute_diff_identical() {
        let f = vec![0.5, 0.5, 0.5];
        assert!((mean_absolute_diff(&f, &f) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_mean_absolute_diff_different() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 1.0, 1.0];
        assert!((mean_absolute_diff(&a, &b) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_mean_absolute_diff_empty() {
        let a: Vec<f64> = Vec::new();
        let b = vec![1.0];
        assert!((mean_absolute_diff(&a, &b) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_histogram_correlation_identical() {
        let mut hist = [0.0_f64; 256];
        for i in 0..256 {
            hist[i] = i as f64;
        }
        let corr = histogram_correlation(&hist, &hist);
        assert!((corr - 1.0).abs() < 1e-9);
    }

    // -- TransitionClassifier tests --

    #[test]
    fn test_classify_cut() {
        let classifier = TransitionClassifier::with_defaults();
        let diffs = vec![0.8];
        let result = classifier.classify(&diffs);
        assert_eq!(result.kind, TransitionKind::Cut);
    }

    #[test]
    fn test_classify_dissolve() {
        let classifier = TransitionClassifier::with_defaults();
        let diffs = vec![0.15, 0.20, 0.18, 0.16, 0.14];
        let result = classifier.classify(&diffs);
        assert_eq!(result.kind, TransitionKind::Dissolve);
    }

    #[test]
    fn test_classify_fade() {
        let classifier = TransitionClassifier::with_defaults();
        let diffs = vec![0.02, 0.03, 0.01, 0.02];
        let result = classifier.classify(&diffs);
        assert_eq!(result.kind, TransitionKind::Fade);
    }

    #[test]
    fn test_classify_empty() {
        let classifier = TransitionClassifier::with_defaults();
        let result = classifier.classify(&[]);
        assert_eq!(result.kind, TransitionKind::Unknown);
    }

    // -- Transition summary tests --

    #[test]
    fn test_summarize_empty() {
        let summary = summarize_transitions(&[]);
        assert_eq!(summary.total, 0);
        assert!(summary.counts.is_empty());
    }

    #[test]
    fn test_summarize_basic() {
        let transitions = vec![
            TransitionDescriptor::new(0, 1, TransitionKind::Cut, 0.9),
            TransitionDescriptor::new(10, 20, TransitionKind::Dissolve, 0.7),
            TransitionDescriptor::new(30, 31, TransitionKind::Cut, 0.85),
        ];
        let summary = summarize_transitions(&transitions);
        assert_eq!(summary.total, 3);
        assert_eq!(summary.counts.get(&TransitionKind::Cut), Some(&2));
        assert_eq!(summary.counts.get(&TransitionKind::Dissolve), Some(&1));
        assert!(summary.avg_confidence > 0.0);
    }
}
