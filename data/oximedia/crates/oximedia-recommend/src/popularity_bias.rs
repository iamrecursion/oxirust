#![allow(dead_code)]
//! Popularity bias detection and correction for recommendation systems.
//!
//! Identifies when recommendations are disproportionately skewed toward
//! popular items and applies correction strategies to ensure long-tail
//! content gets fair exposure.

use std::collections::HashMap;

/// Popularity tier classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PopularityTier {
    /// Top 1% most popular items.
    Head,
    /// 1%-20% popularity range.
    Torso,
    /// Bottom 80% of items (long tail).
    LongTail,
}

/// Statistics about an item's popularity.
#[derive(Debug, Clone)]
pub struct ItemPopularity {
    /// Item identifier.
    pub item_id: String,
    /// Total interaction count.
    pub interaction_count: u64,
    /// Popularity percentile (0.0 = least popular, 1.0 = most popular).
    pub percentile: f64,
    /// Assigned popularity tier.
    pub tier: PopularityTier,
}

impl ItemPopularity {
    /// Create a new item popularity record.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn new(item_id: &str, interaction_count: u64, percentile: f64) -> Self {
        let tier = if percentile >= 0.99 {
            PopularityTier::Head
        } else if percentile >= 0.80 {
            PopularityTier::Torso
        } else {
            PopularityTier::LongTail
        };
        Self {
            item_id: item_id.to_string(),
            interaction_count,
            percentile,
            tier,
        }
    }
}

/// Metrics quantifying the degree of popularity bias.
#[derive(Debug, Clone)]
pub struct BiasMetrics {
    /// Gini coefficient of recommendation scores (0 = perfect equality, 1 = max inequality).
    pub gini_coefficient: f64,
    /// Fraction of recommendations from the head tier.
    pub head_fraction: f64,
    /// Fraction of recommendations from the long-tail tier.
    pub long_tail_fraction: f64,
    /// Average popularity percentile across recommendations.
    pub avg_percentile: f64,
    /// Number of unique items recommended.
    pub unique_items: usize,
    /// Catalog coverage (fraction of total catalog recommended).
    pub catalog_coverage: f64,
}

impl BiasMetrics {
    /// Return true if the recommendations are heavily biased toward popular items.
    #[must_use]
    pub fn is_heavily_biased(&self) -> bool {
        self.gini_coefficient > 0.7 || self.head_fraction > 0.5
    }

    /// Return true if long-tail coverage is adequate (at least the given threshold).
    #[must_use]
    pub fn has_adequate_long_tail(&self, min_fraction: f64) -> bool {
        self.long_tail_fraction >= min_fraction
    }
}

/// Correction strategy for reducing popularity bias.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorrectionStrategy {
    /// Inverse propensity scoring: boost items inversely proportional to popularity.
    InversePropensity,
    /// Calibrated: adjust scores to match target tier distribution.
    Calibrated,
    /// Re-ranking: post-process to enforce diversity quotas.
    ReRanking,
    /// No correction applied.
    None,
}

/// Configuration for the popularity bias corrector.
#[derive(Debug, Clone)]
pub struct BiasConfig {
    /// Correction strategy to use.
    pub strategy: CorrectionStrategy,
    /// Target fraction of results from the long-tail tier.
    pub target_long_tail: f64,
    /// Target fraction of results from the head tier.
    pub target_head: f64,
    /// Smoothing factor for inverse propensity scoring (higher = less aggressive).
    pub smoothing: f64,
    /// Whether to log correction details.
    pub log_corrections: bool,
}

impl Default for BiasConfig {
    fn default() -> Self {
        Self {
            strategy: CorrectionStrategy::InversePropensity,
            target_long_tail: 0.3,
            target_head: 0.2,
            smoothing: 1.0,
            log_corrections: false,
        }
    }
}

/// A scored recommendation item for bias correction.
#[derive(Debug, Clone)]
pub struct ScoredItem {
    /// Item identifier.
    pub item_id: String,
    /// Original recommendation score.
    pub original_score: f64,
    /// Corrected score after bias adjustment.
    pub corrected_score: f64,
    /// Popularity percentile.
    pub percentile: f64,
    /// Popularity tier.
    pub tier: PopularityTier,
}

impl ScoredItem {
    /// Create a new scored item.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn new(item_id: &str, score: f64, percentile: f64) -> Self {
        let tier = if percentile >= 0.99 {
            PopularityTier::Head
        } else if percentile >= 0.80 {
            PopularityTier::Torso
        } else {
            PopularityTier::LongTail
        };
        Self {
            item_id: item_id.to_string(),
            original_score: score,
            corrected_score: score,
            percentile,
            tier,
        }
    }

    /// Score change from correction.
    #[must_use]
    pub fn score_delta(&self) -> f64 {
        self.corrected_score - self.original_score
    }
}

/// Detects and corrects popularity bias in recommendation lists.
pub struct PopularityBiasCorrector {
    /// Configuration.
    config: BiasConfig,
    /// Popularity data by item ID.
    popularity_data: HashMap<String, ItemPopularity>,
}

impl PopularityBiasCorrector {
    /// Create a new bias corrector with the given configuration.
    #[must_use]
    pub fn new(config: BiasConfig) -> Self {
        Self {
            config,
            popularity_data: HashMap::new(),
        }
    }

    /// Create a corrector with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(BiasConfig::default())
    }

    /// Register popularity data for an item.
    pub fn add_item_popularity(&mut self, item: ItemPopularity) {
        self.popularity_data.insert(item.item_id.clone(), item);
    }

    /// Return the number of items with popularity data.
    #[must_use]
    pub fn item_count(&self) -> usize {
        self.popularity_data.len()
    }

    /// Measure bias in a list of scored items.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn measure_bias(&self, items: &[ScoredItem]) -> BiasMetrics {
        if items.is_empty() {
            return BiasMetrics {
                gini_coefficient: 0.0,
                head_fraction: 0.0,
                long_tail_fraction: 0.0,
                avg_percentile: 0.0,
                unique_items: 0,
                catalog_coverage: 0.0,
            };
        }

        let n = items.len() as f64;
        let head_count = items
            .iter()
            .filter(|i| i.tier == PopularityTier::Head)
            .count() as f64;
        let tail_count = items
            .iter()
            .filter(|i| i.tier == PopularityTier::LongTail)
            .count() as f64;
        let avg_pct: f64 = items.iter().map(|i| i.percentile).sum::<f64>() / n;

        let mut scores: Vec<f64> = items.iter().map(|i| i.original_score).collect();
        scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let gini = compute_gini(&scores);

        let unique = items
            .iter()
            .map(|i| &i.item_id)
            .collect::<std::collections::HashSet<_>>()
            .len();

        let catalog_size = if self.popularity_data.is_empty() {
            unique
        } else {
            self.popularity_data.len()
        };
        let coverage = if catalog_size > 0 {
            unique as f64 / catalog_size as f64
        } else {
            0.0
        };

        BiasMetrics {
            gini_coefficient: gini,
            head_fraction: head_count / n,
            long_tail_fraction: tail_count / n,
            avg_percentile: avg_pct,
            unique_items: unique,
            catalog_coverage: coverage,
        }
    }

    /// Apply bias correction to a list of scored items.
    #[allow(clippy::cast_precision_loss)]
    pub fn correct(&self, items: &mut [ScoredItem]) {
        match self.config.strategy {
            CorrectionStrategy::InversePropensity => {
                self.apply_inverse_propensity(items);
            }
            CorrectionStrategy::Calibrated => {
                self.apply_calibrated(items);
            }
            CorrectionStrategy::ReRanking => {
                self.apply_reranking(items);
            }
            CorrectionStrategy::None => {}
        }
    }

    /// Apply inverse propensity scoring.
    #[allow(clippy::cast_precision_loss)]
    fn apply_inverse_propensity(&self, items: &mut [ScoredItem]) {
        for item in items.iter_mut() {
            let propensity = item.percentile.max(0.01) + self.config.smoothing;
            item.corrected_score = item.original_score / propensity;
        }
    }

    /// Apply calibrated correction toward target distribution.
    #[allow(clippy::cast_precision_loss)]
    fn apply_calibrated(&self, items: &mut [ScoredItem]) {
        for item in items.iter_mut() {
            let boost = match item.tier {
                PopularityTier::Head => 1.0 - (1.0 - self.config.target_head) * 0.5,
                PopularityTier::Torso => 1.0,
                PopularityTier::LongTail => 1.0 + self.config.target_long_tail,
            };
            item.corrected_score = item.original_score * boost;
        }
    }

    /// Apply re-ranking correction.
    #[allow(clippy::cast_precision_loss)]
    fn apply_reranking(&self, items: &mut [ScoredItem]) {
        // Simple approach: penalize head items slightly, boost long-tail
        for item in items.iter_mut() {
            let factor = match item.tier {
                PopularityTier::Head => 0.8,
                PopularityTier::Torso => 1.0,
                PopularityTier::LongTail => 1.2,
            };
            item.corrected_score = item.original_score * factor;
        }
    }
}

/// Compute the Gini coefficient for a sorted list of non-negative values.
#[allow(clippy::cast_precision_loss)]
fn compute_gini(sorted_values: &[f64]) -> f64 {
    let n = sorted_values.len();
    if n == 0 {
        return 0.0;
    }
    let total: f64 = sorted_values.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    let mut cumulative = 0.0_f64;
    let mut area_under = 0.0_f64;
    for &val in sorted_values {
        cumulative += val;
        area_under += cumulative;
    }
    let n_f = n as f64;
    let gini = (n_f + 1.0) / n_f - (2.0 * area_under) / (n_f * total);
    gini.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_item_popularity_head() {
        let pop = ItemPopularity::new("item1", 10000, 0.99);
        assert_eq!(pop.tier, PopularityTier::Head);
    }

    #[test]
    fn test_item_popularity_torso() {
        let pop = ItemPopularity::new("item2", 500, 0.85);
        assert_eq!(pop.tier, PopularityTier::Torso);
    }

    #[test]
    fn test_item_popularity_long_tail() {
        let pop = ItemPopularity::new("item3", 10, 0.30);
        assert_eq!(pop.tier, PopularityTier::LongTail);
    }

    #[test]
    fn test_bias_metrics_heavily_biased() {
        let metrics = BiasMetrics {
            gini_coefficient: 0.8,
            head_fraction: 0.6,
            long_tail_fraction: 0.05,
            avg_percentile: 0.9,
            unique_items: 10,
            catalog_coverage: 0.01,
        };
        assert!(metrics.is_heavily_biased());
    }

    #[test]
    fn test_bias_metrics_not_biased() {
        let metrics = BiasMetrics {
            gini_coefficient: 0.3,
            head_fraction: 0.1,
            long_tail_fraction: 0.5,
            avg_percentile: 0.5,
            unique_items: 50,
            catalog_coverage: 0.5,
        };
        assert!(!metrics.is_heavily_biased());
    }

    #[test]
    fn test_bias_metrics_adequate_long_tail() {
        let metrics = BiasMetrics {
            gini_coefficient: 0.4,
            head_fraction: 0.2,
            long_tail_fraction: 0.35,
            avg_percentile: 0.5,
            unique_items: 20,
            catalog_coverage: 0.1,
        };
        assert!(metrics.has_adequate_long_tail(0.3));
        assert!(!metrics.has_adequate_long_tail(0.5));
    }

    #[test]
    fn test_scored_item_delta() {
        let mut item = ScoredItem::new("a", 0.8, 0.5);
        item.corrected_score = 0.6;
        assert!((item.score_delta() - (-0.2)).abs() < 1e-10);
    }

    #[test]
    fn test_inverse_propensity_correction() {
        let corrector = PopularityBiasCorrector::new(BiasConfig {
            strategy: CorrectionStrategy::InversePropensity,
            smoothing: 0.0,
            ..BiasConfig::default()
        });
        let mut items = vec![
            ScoredItem::new("head", 0.9, 0.99),
            ScoredItem::new("tail", 0.5, 0.10),
        ];
        corrector.correct(&mut items);
        // Head item should be penalized more (divided by higher propensity)
        assert!(items[1].corrected_score > items[0].corrected_score);
    }

    #[test]
    fn test_calibrated_correction() {
        let corrector = PopularityBiasCorrector::new(BiasConfig {
            strategy: CorrectionStrategy::Calibrated,
            target_long_tail: 0.5,
            target_head: 0.1,
            ..BiasConfig::default()
        });
        let mut items = vec![ScoredItem::new("tail", 0.5, 0.10)];
        corrector.correct(&mut items);
        // Long-tail should be boosted
        assert!(items[0].corrected_score > items[0].original_score);
    }

    #[test]
    fn test_reranking_correction() {
        let corrector = PopularityBiasCorrector::new(BiasConfig {
            strategy: CorrectionStrategy::ReRanking,
            ..BiasConfig::default()
        });
        let mut items = vec![
            ScoredItem::new("head", 1.0, 0.99),
            ScoredItem::new("tail", 1.0, 0.10),
        ];
        corrector.correct(&mut items);
        assert!(items[1].corrected_score > items[0].corrected_score);
    }

    #[test]
    fn test_no_correction() {
        let corrector = PopularityBiasCorrector::new(BiasConfig {
            strategy: CorrectionStrategy::None,
            ..BiasConfig::default()
        });
        let mut items = vec![ScoredItem::new("x", 0.7, 0.5)];
        corrector.correct(&mut items);
        assert!((items[0].corrected_score - items[0].original_score).abs() < 1e-10);
    }

    #[test]
    fn test_measure_bias_empty() {
        let corrector = PopularityBiasCorrector::with_defaults();
        let metrics = corrector.measure_bias(&[]);
        assert_eq!(metrics.unique_items, 0);
        assert!((metrics.gini_coefficient).abs() < 1e-10);
    }

    #[test]
    fn test_measure_bias_single_item() {
        let corrector = PopularityBiasCorrector::with_defaults();
        let items = vec![ScoredItem::new("a", 0.5, 0.5)];
        let metrics = corrector.measure_bias(&items);
        assert_eq!(metrics.unique_items, 1);
    }

    #[test]
    fn test_gini_equal_values() {
        let values = vec![1.0, 1.0, 1.0, 1.0];
        let g = compute_gini(&values);
        assert!(g < 0.01, "Expected near 0 for equal values, got {g}");
    }

    #[test]
    fn test_gini_unequal_values() {
        let values = vec![0.0, 0.0, 0.0, 100.0];
        let g = compute_gini(&values);
        assert!(g > 0.5, "Expected high Gini for skewed values, got {g}");
    }

    #[test]
    fn test_add_item_popularity() {
        let mut corrector = PopularityBiasCorrector::with_defaults();
        corrector.add_item_popularity(ItemPopularity::new("a", 100, 0.5));
        corrector.add_item_popularity(ItemPopularity::new("b", 200, 0.8));
        assert_eq!(corrector.item_count(), 2);
    }
}
