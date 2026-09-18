//! Weighted scoring primitives for composing recommendation relevance signals.
//!
//! Combines multiple heterogeneous score components (collaborative, content,
//! popularity, freshness, …) into a single normalised relevance score for
//! ranking candidates.

#![allow(dead_code)]

/// Individual signal contributing to a recommendation score.
#[derive(Debug, Clone, PartialEq)]
pub enum ScoreComponent {
    /// Score from collaborative filtering (user–user or item–item).
    Collaborative(f64),
    /// Score from content-based similarity.
    ContentBased(f64),
    /// Score derived from aggregate popularity / view counts.
    Popularity(f64),
    /// Score rewarding recently published content.
    Freshness(f64),
    /// Score based on explicit or implicit user rating.
    UserRating(f64),
    /// Score from trending detection algorithms.
    Trending(f64),
    /// Arbitrary named score for extensibility.
    Custom {
        /// Custom score name.
        name: String,
        /// Custom score value.
        value: f64,
    },
}

impl ScoreComponent {
    /// The raw numeric value of this component.
    #[must_use]
    pub fn value(&self) -> f64 {
        match self {
            Self::Collaborative(v)
            | Self::ContentBased(v)
            | Self::Popularity(v)
            | Self::Freshness(v)
            | Self::UserRating(v)
            | Self::Trending(v) => *v,
            Self::Custom { value, .. } => *value,
        }
    }

    /// Human-readable name of the component type.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Collaborative(_) => "collaborative",
            Self::ContentBased(_) => "content_based",
            Self::Popularity(_) => "popularity",
            Self::Freshness(_) => "freshness",
            Self::UserRating(_) => "user_rating",
            Self::Trending(_) => "trending",
            Self::Custom { name, .. } => name.as_str(),
        }
    }
}

/// A weighted combination of score components.
///
/// Each component is assigned an independent weight; the total score is the
/// sum of `weight × component_value` across all added components.
#[derive(Debug, Clone, Default)]
pub struct WeightedScore {
    entries: Vec<(ScoreComponent, f64)>,
}

impl WeightedScore {
    /// Create an empty weighted score.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a score component with the given weight.
    pub fn add_component(&mut self, component: ScoreComponent, weight: f64) {
        self.entries.push((component, weight));
    }

    /// Compute the weighted sum of all components.
    #[must_use]
    pub fn total_score(&self) -> f64 {
        self.entries.iter().map(|(comp, w)| comp.value() * w).sum()
    }

    /// Sum of all weights (used for normalisation).
    #[must_use]
    pub fn total_weight(&self) -> f64 {
        self.entries.iter().map(|(_, w)| *w).sum()
    }

    /// Normalised score in `[0, 1]` relative to maximum achievable value.
    ///
    /// Assumes each component value is already in `[0, 1]`.  Returns `0.0`
    /// when no components have been added or total weight is zero.
    #[must_use]
    pub fn normalize(&self) -> f64 {
        let tw = self.total_weight();
        if tw < f64::EPSILON {
            return 0.0;
        }
        (self.total_score() / tw).clamp(0.0, 1.0)
    }

    /// Number of components currently tracked.
    #[must_use]
    pub fn component_count(&self) -> usize {
        self.entries.len()
    }

    /// Returns an iterator over `(component, weight)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = &(ScoreComponent, f64)> {
        self.entries.iter()
    }
}

/// A candidate item paired with its composite relevance score.
#[derive(Debug, Clone)]
pub struct ScoredItem {
    /// Identifier of the candidate item.
    pub item_id: String,
    /// Composite weighted score for this item.
    pub score: WeightedScore,
}

impl ScoredItem {
    /// Create a scored item with an empty score.
    #[must_use]
    pub fn new(item_id: impl Into<String>) -> Self {
        Self {
            item_id: item_id.into(),
            score: WeightedScore::new(),
        }
    }

    /// Convenience: create with a pre-built score.
    #[must_use]
    pub fn with_score(item_id: impl Into<String>, score: WeightedScore) -> Self {
        Self {
            item_id: item_id.into(),
            score,
        }
    }

    /// Normalised relevance in `[0, 1]`.
    #[must_use]
    pub fn relevance(&self) -> f64 {
        self.score.normalize()
    }

    /// Compare this item's relevance to another, returning `true` if this
    /// item scores strictly higher.
    #[must_use]
    pub fn compare_scores(&self, other: &Self) -> bool {
        self.relevance() > other.relevance()
    }
}

/// Sort a mutable slice of `ScoredItem`s by relevance, highest first.
pub fn rank_scored_items(items: &mut [ScoredItem]) {
    items.sort_by(|a, b| {
        b.relevance()
            .partial_cmp(&a.relevance())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_score_component_value_collaborative() {
        let c = ScoreComponent::Collaborative(0.75);
        assert!((c.value() - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn test_score_component_value_custom() {
        let c = ScoreComponent::Custom {
            name: "my_signal".into(),
            value: 0.42,
        };
        assert!((c.value() - 0.42).abs() < f64::EPSILON);
    }

    #[test]
    fn test_score_component_name_freshness() {
        let c = ScoreComponent::Freshness(0.5);
        assert_eq!(c.name(), "freshness");
    }

    #[test]
    fn test_score_component_name_custom() {
        let c = ScoreComponent::Custom {
            name: "special".into(),
            value: 0.1,
        };
        assert_eq!(c.name(), "special");
    }

    #[test]
    fn test_weighted_score_empty_total() {
        let ws = WeightedScore::new();
        assert!((ws.total_score() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_weighted_score_add_and_total() {
        let mut ws = WeightedScore::new();
        ws.add_component(ScoreComponent::Collaborative(0.8), 1.0);
        ws.add_component(ScoreComponent::Popularity(0.6), 0.5);
        // 0.8*1.0 + 0.6*0.5 = 1.1
        assert!((ws.total_score() - 1.1).abs() < 1e-10);
    }

    #[test]
    fn test_total_weight() {
        let mut ws = WeightedScore::new();
        ws.add_component(ScoreComponent::Freshness(0.5), 0.3);
        ws.add_component(ScoreComponent::Trending(0.9), 0.7);
        assert!((ws.total_weight() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_zero_weight_returns_zero() {
        let ws = WeightedScore::new();
        assert!((ws.normalize() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_normalize_single_component() {
        let mut ws = WeightedScore::new();
        ws.add_component(ScoreComponent::ContentBased(0.6), 1.0);
        assert!((ws.normalize() - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_weighted_average() {
        let mut ws = WeightedScore::new();
        ws.add_component(ScoreComponent::Collaborative(1.0), 0.5);
        ws.add_component(ScoreComponent::ContentBased(0.0), 0.5);
        // (1.0*0.5 + 0.0*0.5) / 1.0 = 0.5
        assert!((ws.normalize() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_component_count() {
        let mut ws = WeightedScore::new();
        ws.add_component(ScoreComponent::UserRating(0.9), 1.0);
        ws.add_component(ScoreComponent::Trending(0.4), 0.2);
        assert_eq!(ws.component_count(), 2);
    }

    #[test]
    fn test_scored_item_relevance() {
        let mut ws = WeightedScore::new();
        ws.add_component(ScoreComponent::Collaborative(0.8), 1.0);
        let item = ScoredItem::with_score("item1", ws);
        assert!((item.relevance() - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_compare_scores_higher_wins() {
        let mut ws1 = WeightedScore::new();
        ws1.add_component(ScoreComponent::Collaborative(0.9), 1.0);
        let mut ws2 = WeightedScore::new();
        ws2.add_component(ScoreComponent::Collaborative(0.4), 1.0);
        let a = ScoredItem::with_score("a", ws1);
        let b = ScoredItem::with_score("b", ws2);
        assert!(a.compare_scores(&b));
        assert!(!b.compare_scores(&a));
    }

    #[test]
    fn test_rank_scored_items_sorts_descending() {
        let make = |id: &str, val: f64| {
            let mut ws = WeightedScore::new();
            ws.add_component(ScoreComponent::Popularity(val), 1.0);
            ScoredItem::with_score(id, ws)
        };
        let mut items = vec![make("c", 0.3), make("a", 0.9), make("b", 0.6)];
        rank_scored_items(&mut items);
        assert_eq!(items[0].item_id, "a");
        assert_eq!(items[1].item_id, "b");
        assert_eq!(items[2].item_id, "c");
    }

    #[test]
    fn test_scored_item_new_empty_score() {
        let item = ScoredItem::new("x");
        assert_eq!(item.score.component_count(), 0);
    }
}
