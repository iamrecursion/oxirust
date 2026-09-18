//! Sync quality scoring for multi-stream alignment in `OxiMedia`.
//!
//! Provides [`SyncScorer`] which evaluates alignment quality and produces
//! [`SyncReport`] summaries with per-stream metrics.

#![allow(dead_code)]

/// Quality level of a synchronization result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SyncQuality {
    /// Sync is unusable; offset error is too large.
    Poor,
    /// Sync is marginal; may cause audible/visible drift.
    Fair,
    /// Sync is acceptable for most purposes.
    Good,
    /// Sync is excellent; sub-frame accuracy achieved.
    Excellent,
}

impl SyncQuality {
    /// Return a numeric score in the range `[0.0, 1.0]`.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn score(&self) -> f64 {
        match self {
            Self::Poor => 0.1,
            Self::Fair => 0.4,
            Self::Good => 0.75,
            Self::Excellent => 1.0,
        }
    }

    /// Human-readable label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Poor => "poor",
            Self::Fair => "fair",
            Self::Good => "good",
            Self::Excellent => "excellent",
        }
    }
}

/// Result of evaluating sync between two streams.
#[derive(Debug, Clone)]
pub struct SyncResult {
    /// Quality classification.
    pub quality: SyncQuality,
    /// Measured offset in milliseconds (positive = stream B is ahead).
    pub offset_ms: f64,
    /// Cross-correlation confidence `[0.0, 1.0]`.
    pub confidence: f64,
    /// Stream identifier string.
    pub stream_id: String,
}

impl SyncResult {
    /// Create a new sync result.
    #[must_use]
    pub fn new(quality: SyncQuality, offset_ms: f64, confidence: f64, stream_id: &str) -> Self {
        Self {
            quality,
            offset_ms,
            confidence,
            stream_id: stream_id.to_owned(),
        }
    }

    /// Return `true` if quality is [`SyncQuality::Good`] or better.
    #[must_use]
    pub fn is_good_sync(&self) -> bool {
        self.quality >= SyncQuality::Good
    }

    /// Absolute value of the offset.
    #[must_use]
    pub fn abs_offset_ms(&self) -> f64 {
        self.offset_ms.abs()
    }
}

/// Configuration for the sync scorer.
#[derive(Debug, Clone)]
pub struct SyncScorerConfig {
    /// Offset threshold in ms below which sync is considered Excellent.
    pub excellent_threshold_ms: f64,
    /// Offset threshold in ms below which sync is considered Good.
    pub good_threshold_ms: f64,
    /// Offset threshold in ms below which sync is considered Fair.
    pub fair_threshold_ms: f64,
    /// Minimum confidence required for Good classification.
    pub min_confidence_good: f64,
}

impl Default for SyncScorerConfig {
    fn default() -> Self {
        Self {
            excellent_threshold_ms: 0.5,
            good_threshold_ms: 5.0,
            fair_threshold_ms: 33.0,
            min_confidence_good: 0.70,
        }
    }
}

/// Evaluates stream sync results and produces quality ratings.
#[derive(Debug)]
pub struct SyncScorer {
    config: SyncScorerConfig,
}

impl SyncScorer {
    /// Create a new scorer with the given configuration.
    #[must_use]
    pub fn new(config: SyncScorerConfig) -> Self {
        Self { config }
    }

    /// Create a scorer with default thresholds.
    #[must_use]
    pub fn default_scorer() -> Self {
        Self::new(SyncScorerConfig::default())
    }

    /// Evaluate a raw offset + confidence measurement for a stream.
    #[must_use]
    pub fn evaluate(&self, stream_id: &str, offset_ms: f64, confidence: f64) -> SyncResult {
        let abs_off = offset_ms.abs();
        let quality = if confidence < self.config.min_confidence_good {
            SyncQuality::Poor
        } else if abs_off <= self.config.excellent_threshold_ms {
            SyncQuality::Excellent
        } else if abs_off <= self.config.good_threshold_ms {
            SyncQuality::Good
        } else if abs_off <= self.config.fair_threshold_ms {
            SyncQuality::Fair
        } else {
            SyncQuality::Poor
        };
        SyncResult::new(quality, offset_ms, confidence, stream_id)
    }
}

/// Aggregated report from evaluating multiple streams.
#[derive(Debug, Default)]
pub struct SyncReport {
    /// Individual results, one per stream.
    pub results: Vec<SyncResult>,
}

impl SyncReport {
    /// Create an empty report.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a result to the report.
    pub fn add(&mut self, result: SyncResult) {
        self.results.push(result);
    }

    /// Return the worst quality across all results.
    #[must_use]
    pub fn worst_quality(&self) -> Option<SyncQuality> {
        self.results.iter().map(|r| r.quality).min()
    }

    /// Compute the average absolute offset in milliseconds.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn avg_offset_ms(&self) -> f64 {
        if self.results.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.results.iter().map(SyncResult::abs_offset_ms).sum();
        sum / self.results.len() as f64
    }

    /// Return the number of streams with good or better sync.
    #[must_use]
    pub fn good_count(&self) -> usize {
        self.results.iter().filter(|r| r.is_good_sync()).count()
    }

    /// Return `true` if every stream achieved good or better sync.
    #[must_use]
    pub fn all_good(&self) -> bool {
        !self.results.is_empty() && self.results.iter().all(SyncResult::is_good_sync)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── SyncQuality ──────────────────────────────────────────────────────────

    #[test]
    fn test_quality_order() {
        assert!(SyncQuality::Excellent > SyncQuality::Good);
        assert!(SyncQuality::Good > SyncQuality::Fair);
        assert!(SyncQuality::Fair > SyncQuality::Poor);
    }

    #[test]
    fn test_quality_score_monotone() {
        assert!(SyncQuality::Excellent.score() > SyncQuality::Good.score());
        assert!(SyncQuality::Good.score() > SyncQuality::Fair.score());
        assert!(SyncQuality::Fair.score() > SyncQuality::Poor.score());
    }

    #[test]
    fn test_quality_score_range() {
        for q in [
            SyncQuality::Poor,
            SyncQuality::Fair,
            SyncQuality::Good,
            SyncQuality::Excellent,
        ] {
            let s = q.score();
            assert!((0.0..=1.0).contains(&s), "score {s} out of [0,1]");
        }
    }

    #[test]
    fn test_quality_labels_non_empty() {
        for q in [
            SyncQuality::Poor,
            SyncQuality::Fair,
            SyncQuality::Good,
            SyncQuality::Excellent,
        ] {
            assert!(!q.label().is_empty());
        }
    }

    // ── SyncResult ───────────────────────────────────────────────────────────

    #[test]
    fn test_sync_result_is_good() {
        let good = SyncResult::new(SyncQuality::Good, 2.0, 0.9, "cam1");
        let poor = SyncResult::new(SyncQuality::Poor, 50.0, 0.3, "cam2");
        assert!(good.is_good_sync());
        assert!(!poor.is_good_sync());
    }

    #[test]
    fn test_sync_result_excellent_is_good() {
        let r = SyncResult::new(SyncQuality::Excellent, 0.1, 0.99, "cam1");
        assert!(r.is_good_sync());
    }

    #[test]
    fn test_abs_offset_negative() {
        let r = SyncResult::new(SyncQuality::Good, -8.0, 0.85, "cam3");
        assert!((r.abs_offset_ms() - 8.0).abs() < f64::EPSILON);
    }

    // ── SyncScorer ───────────────────────────────────────────────────────────

    #[test]
    fn test_scorer_excellent() {
        let scorer = SyncScorer::default_scorer();
        let r = scorer.evaluate("cam1", 0.3, 0.95);
        assert_eq!(r.quality, SyncQuality::Excellent);
    }

    #[test]
    fn test_scorer_good() {
        let scorer = SyncScorer::default_scorer();
        let r = scorer.evaluate("cam1", 3.0, 0.80);
        assert_eq!(r.quality, SyncQuality::Good);
    }

    #[test]
    fn test_scorer_fair() {
        let scorer = SyncScorer::default_scorer();
        let r = scorer.evaluate("cam1", 15.0, 0.75);
        assert_eq!(r.quality, SyncQuality::Fair);
    }

    #[test]
    fn test_scorer_poor_large_offset() {
        let scorer = SyncScorer::default_scorer();
        let r = scorer.evaluate("cam1", 100.0, 0.90);
        assert_eq!(r.quality, SyncQuality::Poor);
    }

    #[test]
    fn test_scorer_poor_low_confidence() {
        let scorer = SyncScorer::default_scorer();
        let r = scorer.evaluate("cam1", 0.1, 0.2);
        assert_eq!(r.quality, SyncQuality::Poor);
    }

    // ── SyncReport ───────────────────────────────────────────────────────────

    #[test]
    fn test_report_empty() {
        let report = SyncReport::new();
        assert!(report.worst_quality().is_none());
        assert!((report.avg_offset_ms()).abs() < f64::EPSILON);
        assert!(!report.all_good());
    }

    #[test]
    fn test_report_worst_quality() {
        let scorer = SyncScorer::default_scorer();
        let mut report = SyncReport::new();
        report.add(scorer.evaluate("cam1", 0.2, 0.99));
        report.add(scorer.evaluate("cam2", 20.0, 0.80));
        assert_eq!(report.worst_quality(), Some(SyncQuality::Fair));
    }

    #[test]
    fn test_report_avg_offset() {
        let mut report = SyncReport::new();
        report.add(SyncResult::new(SyncQuality::Good, 4.0, 0.9, "a"));
        report.add(SyncResult::new(SyncQuality::Good, 6.0, 0.9, "b"));
        assert!((report.avg_offset_ms() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_report_all_good() {
        let scorer = SyncScorer::default_scorer();
        let mut report = SyncReport::new();
        report.add(scorer.evaluate("cam1", 0.3, 0.95));
        report.add(scorer.evaluate("cam2", 2.0, 0.80));
        assert!(report.all_good());
    }

    #[test]
    fn test_report_good_count() {
        let scorer = SyncScorer::default_scorer();
        let mut report = SyncReport::new();
        report.add(scorer.evaluate("cam1", 0.3, 0.95)); // Excellent
        report.add(scorer.evaluate("cam2", 2.0, 0.80)); // Good
        report.add(scorer.evaluate("cam3", 50.0, 0.8)); // Poor
        assert_eq!(report.good_count(), 2);
    }
}
