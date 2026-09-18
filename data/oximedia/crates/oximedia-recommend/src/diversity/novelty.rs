//! Novelty promotion for serendipitous recommendations.

use std::collections::HashSet;
use uuid::Uuid;

/// Novelty scorer
pub struct NoveltyScorer {
    /// Seen content IDs
    seen_content: HashSet<Uuid>,
}

impl NoveltyScorer {
    /// Create a new novelty scorer
    #[must_use]
    pub fn new() -> Self {
        Self {
            seen_content: HashSet::new(),
        }
    }

    /// Mark content as seen
    pub fn mark_seen(&mut self, content_id: Uuid) {
        self.seen_content.insert(content_id);
    }

    /// Calculate novelty score
    #[must_use]
    pub fn calculate_novelty(&self, content_id: Uuid, popularity: f32) -> f32 {
        let unseen_bonus = if self.seen_content.contains(&content_id) {
            0.0
        } else {
            1.0
        };

        // Novelty is higher for unseen and less popular content
        let popularity_factor = 1.0 - popularity.min(1.0);

        unseen_bonus * 0.6 + popularity_factor * 0.4
    }
}

impl Default for NoveltyScorer {
    fn default() -> Self {
        Self::new()
    }
}

/// Serendipity calculator
pub struct SerendipityCalculator;

impl SerendipityCalculator {
    /// Calculate serendipity score
    ///
    /// Serendipity = Unexpectedness × Relevance
    #[must_use]
    pub fn calculate(unexpectedness: f32, relevance: f32) -> f32 {
        unexpectedness * relevance
    }

    /// Calculate unexpectedness based on category overlap
    #[must_use]
    pub fn calculate_unexpectedness(user_categories: &[String], item_categories: &[String]) -> f32 {
        if user_categories.is_empty() || item_categories.is_empty() {
            return 1.0;
        }

        let user_set: HashSet<_> = user_categories.iter().collect();
        let item_set: HashSet<_> = item_categories.iter().collect();

        let overlap = user_set.intersection(&item_set).count();
        let union = user_set.union(&item_set).count();

        if union == 0 {
            return 1.0;
        }

        1.0 - (overlap as f32 / union as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_novelty_scorer() {
        let mut scorer = NoveltyScorer::new();
        let content_id = Uuid::new_v4();

        let novelty_before = scorer.calculate_novelty(content_id, 0.5);
        scorer.mark_seen(content_id);
        let novelty_after = scorer.calculate_novelty(content_id, 0.5);

        assert!(novelty_before > novelty_after);
    }

    #[test]
    fn test_serendipity() {
        let score = SerendipityCalculator::calculate(0.8, 0.7);
        assert!((score - 0.56).abs() < 0.01);
    }

    #[test]
    fn test_unexpectedness() {
        let user_cats = vec![String::from("Action"), String::from("Thriller")];
        let item_cats = vec![String::from("Comedy"), String::from("Romance")];

        let unexpected = SerendipityCalculator::calculate_unexpectedness(&user_cats, &item_cats);
        assert!(unexpected > 0.5); // No overlap, should be unexpected
    }
}
