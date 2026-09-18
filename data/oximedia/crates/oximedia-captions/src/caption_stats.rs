//! Caption statistics and quality scoring for `OxiMedia`.
//!
//! Computes words-per-minute, characters-per-cue, density classification,
//! and broadcast-compliance scores for a caption track.

#![allow(dead_code)]

/// Classifies the textual density of a caption track.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CaptionDensity {
    /// Very sparse – well below typical broadcast rates.
    Sparse,
    /// Normal density within broadcast-acceptable range.
    Normal,
    /// Slightly dense but still within tolerance.
    Dense,
    /// Exceeds recommended caption rates; compliance risk.
    OverLimit,
}

impl CaptionDensity {
    /// Human-readable label for display or reporting.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Sparse => "sparse",
            Self::Normal => "normal",
            Self::Dense => "dense",
            Self::OverLimit => "over-limit",
        }
    }

    /// Returns `true` when density is within broadcast-acceptable bounds.
    #[must_use]
    pub fn is_compliant(self) -> bool {
        matches!(self, Self::Sparse | Self::Normal | Self::Dense)
    }

    /// Classify a words-per-minute value into a density level.
    #[must_use]
    pub fn from_wpm(wpm: f64) -> Self {
        if wpm < 100.0 {
            Self::Sparse
        } else if wpm <= 160.0 {
            Self::Normal
        } else if wpm <= 200.0 {
            Self::Dense
        } else {
            Self::OverLimit
        }
    }
}

/// Aggregated statistics for a caption track.
#[derive(Debug, Clone, Default)]
pub struct CaptionStats {
    /// Total number of cues in the track.
    pub cue_count: usize,
    /// Total word count across all cues.
    pub total_words: usize,
    /// Total character count across all cues (excluding whitespace).
    pub total_chars: usize,
    /// Total covered duration of all cues in milliseconds.
    pub total_duration_ms: u64,
    /// Number of cues that exceed the per-cue character limit.
    pub over_limit_cues: usize,
}

impl CaptionStats {
    /// Calculates words per minute based on total words and covered duration.
    ///
    /// Returns `0.0` when there is no duration data.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn words_per_minute(&self) -> f64 {
        if self.total_duration_ms == 0 {
            return 0.0;
        }
        let minutes = self.total_duration_ms as f64 / 60_000.0;
        self.total_words as f64 / minutes
    }

    /// Calculates average characters per cue.
    ///
    /// Returns `0.0` when there are no cues.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn avg_chars_per_cue(&self) -> f64 {
        if self.cue_count == 0 {
            return 0.0;
        }
        self.total_chars as f64 / self.cue_count as f64
    }

    /// Returns the density classification for this track.
    #[must_use]
    pub fn density(&self) -> CaptionDensity {
        CaptionDensity::from_wpm(self.words_per_minute())
    }

    /// Builds `CaptionStats` from a slice of `(text, duration_ms)` tuples.
    #[must_use]
    pub fn from_cues(cues: &[(&str, u64)], chars_per_cue_limit: usize) -> Self {
        let mut stats = Self::default();
        stats.cue_count = cues.len();
        for (text, dur) in cues {
            let words = text.split_whitespace().count();
            let chars = text.chars().filter(|c| !c.is_whitespace()).count();
            stats.total_words += words;
            stats.total_chars += chars;
            stats.total_duration_ms += dur;
            if chars > chars_per_cue_limit {
                stats.over_limit_cues += 1;
            }
        }
        stats
    }
}

/// A composite quality score for a caption track.
#[derive(Debug, Clone)]
pub struct CaptionQualityScore {
    /// Words-per-minute of the track (lower is generally better for accessibility).
    pub wpm: f64,
    /// Fraction of cues within the per-cue character limit (0.0–1.0).
    pub chars_compliance_ratio: f64,
    /// Overall density classification.
    pub density: CaptionDensity,
    /// Numeric score from 0 (worst) to 100 (best).
    pub score: u8,
}

impl CaptionQualityScore {
    /// Computes a quality score from `CaptionStats`.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn from_stats(stats: &CaptionStats) -> Self {
        let wpm = stats.words_per_minute();
        let density = stats.density();

        let chars_compliance_ratio = if stats.cue_count == 0 {
            1.0
        } else {
            let compliant = stats.cue_count.saturating_sub(stats.over_limit_cues);
            compliant as f64 / stats.cue_count as f64
        };

        // Score heuristic: start at 100, deduct for density and non-compliance
        let density_penalty: f64 = match density {
            CaptionDensity::Sparse => 5.0,
            CaptionDensity::Normal => 0.0,
            CaptionDensity::Dense => 15.0,
            CaptionDensity::OverLimit => 40.0,
        };
        let compliance_penalty = (1.0 - chars_compliance_ratio) * 30.0;
        let raw = 100.0 - density_penalty - compliance_penalty;
        let score = raw.clamp(0.0, 100.0) as u8;

        Self {
            wpm,
            chars_compliance_ratio,
            density,
            score,
        }
    }

    /// Returns `true` when the track meets broadcast compliance thresholds.
    #[must_use]
    pub fn is_broadcast_compliant(&self) -> bool {
        self.density.is_compliant() && self.chars_compliance_ratio >= 0.95
    }

    /// Letter grade (A–F) corresponding to the numeric score.
    #[must_use]
    pub fn grade(&self) -> char {
        match self.score {
            90..=100 => 'A',
            80..=89 => 'B',
            70..=79 => 'C',
            60..=69 => 'D',
            _ => 'F',
        }
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn normal_cues() -> Vec<(&'static str, u64)> {
        vec![
            ("Hello world this is a test", 5000),
            ("Another caption line here", 4000),
            ("Short line", 2000),
        ]
    }

    #[test]
    fn test_density_label_sparse() {
        assert_eq!(CaptionDensity::Sparse.label(), "sparse");
    }

    #[test]
    fn test_density_label_over_limit() {
        assert_eq!(CaptionDensity::OverLimit.label(), "over-limit");
    }

    #[test]
    fn test_density_compliance() {
        assert!(CaptionDensity::Normal.is_compliant());
        assert!(!CaptionDensity::OverLimit.is_compliant());
    }

    #[test]
    fn test_density_from_wpm_sparse() {
        assert_eq!(CaptionDensity::from_wpm(50.0), CaptionDensity::Sparse);
    }

    #[test]
    fn test_density_from_wpm_normal() {
        assert_eq!(CaptionDensity::from_wpm(140.0), CaptionDensity::Normal);
    }

    #[test]
    fn test_density_from_wpm_over_limit() {
        assert_eq!(CaptionDensity::from_wpm(250.0), CaptionDensity::OverLimit);
    }

    #[test]
    fn test_stats_from_cues_word_count() {
        let cues = normal_cues();
        let stats = CaptionStats::from_cues(&cues, 60);
        // "Hello world this is a test" = 6, "Another caption line here" = 4, "Short line" = 2
        assert_eq!(stats.total_words, 12);
    }

    #[test]
    fn test_stats_cue_count() {
        let cues = normal_cues();
        let stats = CaptionStats::from_cues(&cues, 60);
        assert_eq!(stats.cue_count, 3);
    }

    #[test]
    fn test_stats_wpm_positive() {
        let cues = normal_cues();
        let stats = CaptionStats::from_cues(&cues, 60);
        assert!(stats.words_per_minute() > 0.0);
    }

    #[test]
    fn test_stats_wpm_zero_duration() {
        let stats = CaptionStats::default();
        assert_eq!(stats.words_per_minute(), 0.0);
    }

    #[test]
    fn test_stats_avg_chars_per_cue_zero_cues() {
        let stats = CaptionStats::default();
        assert_eq!(stats.avg_chars_per_cue(), 0.0);
    }

    #[test]
    fn test_stats_avg_chars_per_cue() {
        let cues = vec![("abc", 1000u64), ("defgh", 1000)];
        let stats = CaptionStats::from_cues(&cues, 60);
        // chars: 3 + 5 = 8, avg = 4.0
        assert!((stats.avg_chars_per_cue() - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_quality_score_broadcast_compliant() {
        // Slow, sparse caption track – should be compliant
        let stats = CaptionStats {
            cue_count: 10,
            total_words: 100,
            total_chars: 400,
            total_duration_ms: 120_000, // 2 minutes → 50 wpm
            over_limit_cues: 0,
        };
        let score = CaptionQualityScore::from_stats(&stats);
        assert!(score.is_broadcast_compliant());
    }

    #[test]
    fn test_quality_score_not_compliant_wpm() {
        let stats = CaptionStats {
            cue_count: 10,
            total_words: 600,
            total_chars: 2400,
            total_duration_ms: 60_000, // 1 minute → 600 wpm
            over_limit_cues: 0,
        };
        let score = CaptionQualityScore::from_stats(&stats);
        assert!(!score.is_broadcast_compliant());
    }

    #[test]
    fn test_quality_score_grade() {
        let stats = CaptionStats {
            cue_count: 5,
            total_words: 50,
            total_chars: 200,
            total_duration_ms: 60_000,
            over_limit_cues: 0,
        };
        let score = CaptionQualityScore::from_stats(&stats);
        // Should be A or B (well within normal density, no over-limit cues)
        assert!(matches!(score.grade(), 'A' | 'B'));
    }

    #[test]
    fn test_over_limit_cues_tracked() {
        let cues = vec![
            (
                "This is a very very very long line that exceeds sixty characters in count yes",
                2000u64,
            ),
            ("Short", 1000),
        ];
        let stats = CaptionStats::from_cues(&cues, 30);
        assert_eq!(stats.over_limit_cues, 1);
    }
}
