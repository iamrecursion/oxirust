//! A/B quality comparison for evaluating transcode settings.
//!
//! Provides types for representing encode candidates and comparing their
//! quality (PSNR, SSIM) vs. bitrate trade-offs.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// A candidate encode configuration to be compared.
#[derive(Debug, Clone, PartialEq)]
pub struct EncodeCandidate {
    /// Human-readable name for this candidate.
    pub name: String,
    /// Constant Rate Factor (0 = lossless, higher = lower quality).
    pub crf: u8,
    /// Encoder speed preset name (e.g. "slow", "medium", "fast").
    pub preset: String,
    /// Estimated output bitrate in kbps.
    pub estimated_kbps: u32,
}

impl EncodeCandidate {
    /// Creates a new `EncodeCandidate`.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        crf: u8,
        preset: impl Into<String>,
        estimated_kbps: u32,
    ) -> Self {
        Self {
            name: name.into(),
            crf,
            preset: preset.into(),
            estimated_kbps,
        }
    }

    /// Returns `true` if this candidate uses lossless encoding (CRF == 0).
    #[must_use]
    pub fn is_lossless(&self) -> bool {
        self.crf == 0
    }
}

/// Quality metrics comparison between two encode candidates.
#[derive(Debug, Clone)]
pub struct QualityComparison {
    /// First candidate.
    pub candidate_a: EncodeCandidate,
    /// Second candidate.
    pub candidate_b: EncodeCandidate,
    /// PSNR difference (A - B) in dB. Positive means A is better.
    pub psnr_diff: f32,
    /// SSIM difference (A - B). Positive means A is better.
    pub ssim_diff: f32,
    /// Bitrate difference as a percentage: `(bitrate_a - bitrate_b) / bitrate_b * 100`.
    pub bitrate_diff_pct: f32,
}

impl QualityComparison {
    /// Creates a new `QualityComparison`.
    #[must_use]
    pub fn new(
        candidate_a: EncodeCandidate,
        candidate_b: EncodeCandidate,
        psnr_diff: f32,
        ssim_diff: f32,
        bitrate_diff_pct: f32,
    ) -> Self {
        Self {
            candidate_a,
            candidate_b,
            psnr_diff,
            ssim_diff,
            bitrate_diff_pct,
        }
    }

    /// Returns the name of the candidate with higher PSNR.
    ///
    /// Returns `"tie"` if the difference is within 0.01 dB.
    #[must_use]
    pub fn winner_by_psnr(&self) -> &str {
        if self.psnr_diff.abs() < 0.01 {
            "tie"
        } else if self.psnr_diff > 0.0 {
            &self.candidate_a.name
        } else {
            &self.candidate_b.name
        }
    }

    /// Returns the name of the candidate with the better quality/bitrate trade-off.
    ///
    /// "Efficiency" is defined as PSNR gain relative to bitrate increase.
    /// If A has higher PSNR but also higher bitrate, efficiency is
    /// `psnr_diff / bitrate_diff_pct`. If the score is ≥ 0 A wins; otherwise B wins.
    /// Returns `"tie"` if both PSNR and bitrate differences are negligible.
    #[must_use]
    pub fn winner_by_efficiency(&self) -> &str {
        // If bitrate is essentially equal, fall back to pure PSNR
        if self.bitrate_diff_pct.abs() < 0.1 {
            return self.winner_by_psnr();
        }
        // A is better if it delivers more quality per kbps
        // Efficiency score: positive → A is more efficient
        let score = if self.bitrate_diff_pct > 0.0 {
            // A costs more bitrate; only wins if PSNR gain is worth it
            self.psnr_diff / self.bitrate_diff_pct
        } else {
            // A costs less bitrate; wins unless its PSNR is notably worse
            -self.psnr_diff / self.bitrate_diff_pct
        };

        if score >= 0.0 {
            &self.candidate_a.name
        } else {
            &self.candidate_b.name
        }
    }
}

/// A suite of A/B comparisons that can identify the overall best candidate.
#[derive(Debug, Clone, Default)]
pub struct AbTestSuite {
    /// All comparisons in this suite.
    pub comparisons: Vec<QualityComparison>,
}

impl AbTestSuite {
    /// Creates an empty suite.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a comparison to the suite.
    pub fn add(&mut self, comparison: QualityComparison) {
        self.comparisons.push(comparison);
    }

    /// Returns the number of comparisons in the suite.
    #[must_use]
    pub fn comparison_count(&self) -> usize {
        self.comparisons.len()
    }

    /// Returns the name of the candidate that wins the most comparisons by PSNR.
    ///
    /// Returns `None` if the suite is empty.
    #[must_use]
    pub fn best_candidate_by_psnr(&self) -> Option<String> {
        if self.comparisons.is_empty() {
            return None;
        }
        let mut scores: std::collections::HashMap<&str, i32> = std::collections::HashMap::new();
        for cmp in &self.comparisons {
            let winner = cmp.winner_by_psnr();
            if winner != "tie" {
                *scores.entry(winner).or_insert(0) += 1;
            }
        }
        scores
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(name, _)| name.to_string())
    }

    /// Returns the name of the candidate that wins the most comparisons by efficiency.
    ///
    /// Returns `None` if the suite is empty.
    #[must_use]
    pub fn best_candidate_by_efficiency(&self) -> Option<String> {
        if self.comparisons.is_empty() {
            return None;
        }
        let mut scores: std::collections::HashMap<&str, i32> = std::collections::HashMap::new();
        for cmp in &self.comparisons {
            let winner = cmp.winner_by_efficiency();
            if winner != "tie" {
                *scores.entry(winner).or_insert(0) += 1;
            }
        }
        scores
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(name, _)| name.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candidate(name: &str, crf: u8, kbps: u32) -> EncodeCandidate {
        EncodeCandidate::new(name, crf, "medium", kbps)
    }

    // --- EncodeCandidate ---

    #[test]
    fn test_is_lossless_true() {
        let c = make_candidate("lossless", 0, 50000);
        assert!(c.is_lossless());
    }

    #[test]
    fn test_is_lossless_false() {
        let c = make_candidate("lossy", 23, 3000);
        assert!(!c.is_lossless());
    }

    #[test]
    fn test_candidate_name() {
        let c = make_candidate("my-encoder", 18, 5000);
        assert_eq!(c.name, "my-encoder");
    }

    // --- QualityComparison ---

    #[test]
    fn test_winner_by_psnr_a_wins() {
        let cmp = QualityComparison::new(
            make_candidate("A", 18, 4000),
            make_candidate("B", 23, 3000),
            2.0,
            0.01,
            33.0,
        );
        assert_eq!(cmp.winner_by_psnr(), "A");
    }

    #[test]
    fn test_winner_by_psnr_b_wins() {
        let cmp = QualityComparison::new(
            make_candidate("A", 23, 3000),
            make_candidate("B", 18, 4000),
            -2.0,
            -0.01,
            -25.0,
        );
        assert_eq!(cmp.winner_by_psnr(), "B");
    }

    #[test]
    fn test_winner_by_psnr_tie() {
        let cmp = QualityComparison::new(
            make_candidate("A", 18, 4000),
            make_candidate("B", 18, 4000),
            0.005,
            0.0,
            0.0,
        );
        assert_eq!(cmp.winner_by_psnr(), "tie");
    }

    #[test]
    fn test_winner_by_efficiency_same_bitrate_falls_back_to_psnr() {
        let cmp = QualityComparison::new(
            make_candidate("A", 18, 4000),
            make_candidate("B", 23, 4000),
            3.0,
            0.02,
            0.05, // essentially same bitrate
        );
        assert_eq!(cmp.winner_by_efficiency(), "A");
    }

    #[test]
    fn test_winner_by_efficiency_a_cheaper_and_better() {
        // A has lower bitrate (negative bitrate_diff_pct) and better PSNR
        let cmp = QualityComparison::new(
            make_candidate("A", 20, 3000),
            make_candidate("B", 18, 4000),
            1.0, // A is 1 dB better
            0.01,
            -25.0, // A costs 25% less
        );
        assert_eq!(cmp.winner_by_efficiency(), "A");
    }

    // --- AbTestSuite ---

    #[test]
    fn test_suite_empty() {
        let suite = AbTestSuite::new();
        assert_eq!(suite.comparison_count(), 0);
        assert!(suite.best_candidate_by_psnr().is_none());
        assert!(suite.best_candidate_by_efficiency().is_none());
    }

    #[test]
    fn test_suite_add_increments_count() {
        let mut suite = AbTestSuite::new();
        let cmp = QualityComparison::new(
            make_candidate("A", 18, 4000),
            make_candidate("B", 23, 3000),
            2.0,
            0.01,
            33.0,
        );
        suite.add(cmp);
        assert_eq!(suite.comparison_count(), 1);
    }

    #[test]
    fn test_suite_best_by_psnr() {
        let mut suite = AbTestSuite::new();
        // A wins twice
        for _ in 0..2 {
            suite.add(QualityComparison::new(
                make_candidate("A", 18, 4000),
                make_candidate("B", 23, 3000),
                2.0,
                0.01,
                33.0,
            ));
        }
        assert_eq!(suite.best_candidate_by_psnr(), Some("A".to_string()));
    }

    #[test]
    fn test_suite_best_by_efficiency() {
        let mut suite = AbTestSuite::new();
        // B wins by efficiency: lower bitrate, comparable quality
        for _ in 0..2 {
            suite.add(QualityComparison::new(
                make_candidate("A", 18, 5000),
                make_candidate("B", 20, 3500),
                -0.5, // B is 0.5 dB better
                -0.002,
                43.0, // A costs 43% more
            ));
        }
        assert_eq!(suite.best_candidate_by_efficiency(), Some("B".to_string()));
    }
}
