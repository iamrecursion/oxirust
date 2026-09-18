#![allow(dead_code)]
//! Multi-factor search result ranking for `oximedia-search`.
//!
//! Provides a flexible ranking system that combines text relevance,
//! recency, popularity, and user-defined boosts to produce a final
//! ranked list of search results.

/// A factor that contributes to a document's final ranking score.
#[derive(Debug, Clone, PartialEq)]
pub enum RankingFactor {
    /// Text relevance from the BM25 / TF-IDF score.
    TextRelevance(f32),
    /// Boost applied to recently-created or recently-modified assets.
    Recency(f32),
    /// Boost from high view / access counts (popularity signal).
    Popularity(f32),
    /// Manual editorial boost applied to specific assets.
    EditorialBoost(f32),
    /// Penalty for very short (likely low-quality) assets.
    DurationPenalty(f32),
}

impl RankingFactor {
    /// Returns the raw weight of this factor (positive = boost, negative = penalty).
    #[must_use]
    pub fn weight(&self) -> f32 {
        match self {
            Self::TextRelevance(w)
            | Self::Recency(w)
            | Self::Popularity(w)
            | Self::EditorialBoost(w)
            | Self::DurationPenalty(w) => *w,
        }
    }

    /// Returns `true` if this factor reduces the score.
    #[must_use]
    pub fn is_penalty(&self) -> bool {
        self.weight() < 0.0
    }
}

/// A search result together with its computed ranking factors.
#[derive(Debug, Clone)]
pub struct RankedResult {
    /// Asset identifier string (e.g. UUID).
    pub asset_id: String,
    /// The factors that contributed to the final score.
    pub factors: Vec<RankingFactor>,
    /// Final combined score.
    pub final_score: f32,
}

impl RankedResult {
    /// Create a `RankedResult` by summing all factor weights.
    pub fn new(asset_id: impl Into<String>, factors: Vec<RankingFactor>) -> Self {
        let final_score = factors
            .iter()
            .map(RankingFactor::weight)
            .sum::<f32>()
            .clamp(0.0, 1.0);
        Self {
            asset_id: asset_id.into(),
            factors,
            final_score,
        }
    }

    /// Returns the contribution of a specific factor variant (sum if multiple).
    #[must_use]
    pub fn factor_contribution(&self, variant: &str) -> f32 {
        self.factors
            .iter()
            .filter(|f| {
                matches!(
                    (f, variant),
                    (RankingFactor::TextRelevance(_), "text")
                        | (RankingFactor::Recency(_), "recency")
                        | (RankingFactor::Popularity(_), "popularity")
                        | (RankingFactor::EditorialBoost(_), "editorial")
                        | (RankingFactor::DurationPenalty(_), "duration")
                )
            })
            .map(RankingFactor::weight)
            .sum()
    }
}

/// Weights used when combining ranking factors.
#[derive(Debug, Clone)]
pub struct RankingWeights {
    /// Weight applied to text relevance factor (default 0.5).
    pub text_weight: f32,
    /// Weight applied to recency factor (default 0.2).
    pub recency_weight: f32,
    /// Weight applied to popularity factor (default 0.2).
    pub popularity_weight: f32,
    /// Weight applied to editorial boost (default 0.1).
    pub editorial_weight: f32,
}

impl Default for RankingWeights {
    fn default() -> Self {
        Self {
            text_weight: 0.5,
            recency_weight: 0.2,
            popularity_weight: 0.2,
            editorial_weight: 0.1,
        }
    }
}

/// A raw candidate result fed into the ranker.
#[derive(Debug, Clone)]
pub struct RankCandidate {
    /// Asset identifier.
    pub asset_id: String,
    /// Raw text relevance in `[0.0, 1.0]`.
    pub text_score: f32,
    /// Unix timestamp of creation (seconds).
    pub created_at: i64,
    /// Number of times the asset has been viewed.
    pub view_count: u64,
    /// Duration in seconds (used for duration penalty).
    pub duration_secs: f64,
    /// Optional editorial boost value (0.0 = none).
    pub editorial_boost: f32,
}

/// Multi-factor search result ranker.
///
/// Takes a list of raw candidates and produces a ranked list where each
/// result has an explainable `RankedResult` with individual factor scores.
#[derive(Debug)]
pub struct SearchRanker {
    weights: RankingWeights,
    /// Unix timestamp used as "now" for recency calculations (defaults to current time).
    now_secs: i64,
    /// Duration below which a duration penalty is applied (seconds).
    short_duration_threshold: f64,
}

impl SearchRanker {
    /// Create a ranker with default weights and the current system time.
    #[must_use]
    pub fn new() -> Self {
        let now_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        Self {
            weights: RankingWeights::default(),
            now_secs,
            short_duration_threshold: 5.0,
        }
    }

    /// Create a ranker with custom weights.
    #[must_use]
    pub fn with_weights(weights: RankingWeights) -> Self {
        let mut r = Self::new();
        r.weights = weights;
        r
    }

    /// Override the reference "now" timestamp (useful for deterministic tests).
    #[must_use]
    pub fn with_now(mut self, now_secs: i64) -> Self {
        self.now_secs = now_secs;
        self
    }

    /// Set the short-duration penalty threshold (seconds).
    #[must_use]
    pub fn with_short_duration_threshold(mut self, secs: f64) -> Self {
        self.short_duration_threshold = secs;
        self
    }

    /// Rank a list of raw candidates and return sorted `RankedResult`s.
    ///
    /// Results are sorted by `final_score` descending.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn rank_results(&self, candidates: Vec<RankCandidate>) -> Vec<RankedResult> {
        let max_views = candidates
            .iter()
            .map(|c| c.view_count)
            .max()
            .unwrap_or(1)
            .max(1);

        let mut ranked: Vec<RankedResult> = candidates
            .into_iter()
            .map(|c| {
                let mut factors = Vec::new();

                // Text relevance.
                let text = c.text_score * self.weights.text_weight;
                factors.push(RankingFactor::TextRelevance(text));

                // Recency: exponential decay over ~30 days.
                let age_days = (self.now_secs - c.created_at).max(0) as f32 / 86_400.0;
                let recency = (-age_days / 30.0).exp() * self.weights.recency_weight;
                factors.push(RankingFactor::Recency(recency));

                // Popularity: normalised view count.
                let popularity =
                    (c.view_count as f32 / max_views as f32) * self.weights.popularity_weight;
                factors.push(RankingFactor::Popularity(popularity));

                // Editorial boost.
                if c.editorial_boost > 0.0 {
                    factors.push(RankingFactor::EditorialBoost(
                        c.editorial_boost * self.weights.editorial_weight,
                    ));
                }

                // Duration penalty for very short assets.
                if c.duration_secs < self.short_duration_threshold && c.duration_secs > 0.0 {
                    factors.push(RankingFactor::DurationPenalty(-0.05));
                }

                RankedResult::new(c.asset_id, factors)
            })
            .collect();

        ranked.sort_by(|a, b| {
            b.final_score
                .partial_cmp(&a.final_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        ranked
    }

    /// Returns the configured weights.
    #[must_use]
    pub fn weights(&self) -> &RankingWeights {
        &self.weights
    }
}

impl Default for SearchRanker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candidate(id: &str, text_score: f32, view_count: u64) -> RankCandidate {
        RankCandidate {
            asset_id: id.to_string(),
            text_score,
            created_at: 1_700_000_000, // fixed past timestamp
            view_count,
            duration_secs: 120.0,
            editorial_boost: 0.0,
        }
    }

    fn ranker() -> SearchRanker {
        // Use a fixed "now" for determinism.
        SearchRanker::new().with_now(1_700_000_000 + 86_400 * 10) // 10 days after created_at
    }

    #[test]
    fn test_rank_results_returns_sorted() {
        let r = ranker();
        let candidates = vec![
            make_candidate("a", 0.3, 10),
            make_candidate("b", 0.9, 100),
            make_candidate("c", 0.6, 50),
        ];
        let ranked = r.rank_results(candidates);
        assert!(ranked[0].final_score >= ranked[1].final_score);
        assert!(ranked[1].final_score >= ranked[2].final_score);
    }

    #[test]
    fn test_rank_results_count_preserved() {
        let r = ranker();
        let candidates: Vec<_> = (0..5)
            .map(|i| make_candidate(&i.to_string(), 0.5, 100))
            .collect();
        let ranked = r.rank_results(candidates);
        assert_eq!(ranked.len(), 5);
    }

    #[test]
    fn test_high_text_score_ranks_higher() {
        let r = ranker();
        let candidates = vec![
            make_candidate("low", 0.1, 0),
            make_candidate("high", 1.0, 0),
        ];
        let ranked = r.rank_results(candidates);
        assert_eq!(ranked[0].asset_id, "high");
    }

    #[test]
    fn test_editorial_boost_applied() {
        let r = ranker();
        let mut boosted = make_candidate("boosted", 0.5, 50);
        boosted.editorial_boost = 1.0;
        let normal = make_candidate("normal", 0.5, 50);
        let ranked = r.rank_results(vec![normal, boosted]);
        assert_eq!(ranked[0].asset_id, "boosted");
    }

    #[test]
    fn test_duration_penalty_for_short_asset() {
        let r = ranker().with_short_duration_threshold(10.0);
        let mut short = make_candidate("short", 0.8, 100);
        short.duration_secs = 2.0;
        let ranked = r.rank_results(vec![short]);
        let has_penalty = ranked[0]
            .factors
            .iter()
            .any(|f| matches!(f, RankingFactor::DurationPenalty(_)));
        assert!(has_penalty);
    }

    #[test]
    fn test_no_duration_penalty_for_normal_asset() {
        let r = ranker();
        let ranked = r.rank_results(vec![make_candidate("normal", 0.8, 100)]);
        let has_penalty = ranked[0]
            .factors
            .iter()
            .any(|f| matches!(f, RankingFactor::DurationPenalty(_)));
        assert!(!has_penalty);
    }

    #[test]
    fn test_final_score_clamped_to_one() {
        let r = ranker();
        let mut c = make_candidate("x", 1.0, u64::MAX);
        c.editorial_boost = 1.0;
        let ranked = r.rank_results(vec![c]);
        assert!(ranked[0].final_score <= 1.0);
    }

    #[test]
    fn test_ranking_factor_weight_positive() {
        let f = RankingFactor::TextRelevance(0.4);
        assert!((f.weight() - 0.4).abs() < 1e-5);
        assert!(!f.is_penalty());
    }

    #[test]
    fn test_ranking_factor_penalty() {
        let f = RankingFactor::DurationPenalty(-0.05);
        assert!(f.is_penalty());
    }

    #[test]
    fn test_ranked_result_factor_contribution() {
        let result = RankedResult::new(
            "asset1",
            vec![
                RankingFactor::TextRelevance(0.3),
                RankingFactor::Recency(0.1),
            ],
        );
        let text_contrib = result.factor_contribution("text");
        assert!((text_contrib - 0.3).abs() < 1e-5);
    }

    #[test]
    fn test_empty_candidates() {
        let r = ranker();
        let ranked = r.rank_results(vec![]);
        assert!(ranked.is_empty());
    }

    #[test]
    fn test_weights_default_sum_to_one() {
        let w = RankingWeights::default();
        let total = w.text_weight + w.recency_weight + w.popularity_weight + w.editorial_weight;
        assert!((total - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_with_weights_applied() {
        let weights = RankingWeights {
            text_weight: 1.0,
            recency_weight: 0.0,
            popularity_weight: 0.0,
            editorial_weight: 0.0,
        };
        let r = SearchRanker::with_weights(weights.clone()).with_now(1_700_000_000);
        assert!((r.weights().text_weight - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_popularity_normalised_by_max() {
        let r = ranker();
        let candidates = vec![
            make_candidate("low_views", 0.5, 10),
            make_candidate("max_views", 0.5, 1000),
        ];
        let ranked = r.rank_results(candidates);
        // max_views asset should have higher popularity contribution.
        let max_pop = ranked
            .iter()
            .find(|r| r.asset_id == "max_views")
            .expect("should succeed in test")
            .factor_contribution("popularity");
        let low_pop = ranked
            .iter()
            .find(|r| r.asset_id == "low_views")
            .expect("should succeed in test")
            .factor_contribution("popularity");
        assert!(max_pop > low_pop);
    }

    #[test]
    fn test_recency_decays_over_time() {
        // Two assets identical except created_at; more recent should score higher on recency.
        let now = 1_700_000_000_i64;
        let r = SearchRanker::new().with_now(now);
        let mut recent = make_candidate("recent", 0.5, 50);
        recent.created_at = now - 86_400; // 1 day ago
        let mut old = make_candidate("old", 0.5, 50);
        old.created_at = now - 86_400 * 60; // 60 days ago
        let ranked = r.rank_results(vec![old, recent]);
        assert_eq!(ranked[0].asset_id, "recent");
    }
}
