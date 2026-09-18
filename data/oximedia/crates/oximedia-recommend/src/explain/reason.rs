//! Recommendation reasoning.

use crate::RecommendationReason;
use serde::{Deserialize, Serialize};

/// Explanation builder for recommendations
pub struct ExplanationBuilder {
    /// Template for explanations
    templates: ExplanationTemplates,
}

/// Explanation templates
#[derive(Debug, Clone)]
pub struct ExplanationTemplates {
    /// Similar to liked template
    pub similar_to_liked: String,
    /// Collaborative filtering template
    pub collaborative: String,
    /// Trending template
    pub trending: String,
    /// Matches profile template
    pub matches_profile: String,
    /// Fresh content template
    pub fresh_content: String,
    /// Popular template
    pub popular: String,
    /// Continue watching template
    pub continue_watching: String,
}

impl Default for ExplanationTemplates {
    fn default() -> Self {
        Self {
            similar_to_liked: String::from("Similar to content you enjoyed"),
            collaborative: String::from("Users with similar tastes also liked this"),
            trending: String::from("Trending in your area"),
            matches_profile: String::from("Matches your interests in {}"),
            fresh_content: String::from("New content from {} days ago"),
            popular: String::from("Popular with {} views"),
            continue_watching: String::from("Continue watching ({}% complete)"),
        }
    }
}

impl ExplanationBuilder {
    /// Create a new explanation builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            templates: ExplanationTemplates::default(),
        }
    }

    /// Build explanation from reason
    #[must_use]
    pub fn build_explanation(&self, reason: &RecommendationReason) -> String {
        match reason {
            RecommendationReason::SimilarToLiked { similarity, .. } => {
                format!(
                    "{} ({:.0}% match)",
                    self.templates.similar_to_liked,
                    similarity * 100.0
                )
            }
            RecommendationReason::CollaborativeFiltering { confidence } => {
                format!(
                    "{} ({:.0}% confidence)",
                    self.templates.collaborative,
                    confidence * 100.0
                )
            }
            RecommendationReason::Trending { trending_score } => {
                format!("{} (score: {:.1})", self.templates.trending, trending_score)
            }
            RecommendationReason::MatchesProfile { categories } => {
                let cats = categories.join(", ");
                self.templates.matches_profile.replace("{}", &cats)
            }
            RecommendationReason::FreshContent { published_days_ago } => self
                .templates
                .fresh_content
                .replace("{}", &published_days_ago.to_string()),
            RecommendationReason::Popular { view_count } => self
                .templates
                .popular
                .replace("{}", &view_count.to_string()),
            RecommendationReason::ContinueWatching { progress } => self
                .templates
                .continue_watching
                .replace("{}", &format!("{:.0}", progress * 100.0)),
        }
    }

    /// Combine multiple reasons into a single explanation
    #[must_use]
    pub fn combine_reasons(&self, reasons: &[RecommendationReason]) -> String {
        if reasons.is_empty() {
            return String::from("Recommended for you");
        }

        if reasons.len() == 1 {
            return self.build_explanation(&reasons[0]);
        }

        let explanations: Vec<String> = reasons.iter().map(|r| self.build_explanation(r)).collect();

        explanations.join("; ")
    }
}

impl Default for ExplanationBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Explanation metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplanationMetadata {
    /// Primary reason
    pub primary_reason: String,
    /// Secondary reasons
    pub secondary_reasons: Vec<String>,
    /// Confidence score
    pub confidence: f32,
}

impl Default for ExplanationMetadata {
    fn default() -> Self {
        Self {
            primary_reason: String::from("Recommended for you"),
            secondary_reasons: Vec::new(),
            confidence: 0.5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn test_explanation_builder() {
        let builder = ExplanationBuilder::new();
        let reason = RecommendationReason::SimilarToLiked {
            content_id: Uuid::new_v4(),
            similarity: 0.85,
        };

        let explanation = builder.build_explanation(&reason);
        assert!(explanation.contains("Similar to"));
    }

    #[test]
    fn test_combine_reasons() {
        let builder = ExplanationBuilder::new();
        let reasons = vec![
            RecommendationReason::Trending {
                trending_score: 0.9,
            },
            RecommendationReason::Popular { view_count: 1000 },
        ];

        let explanation = builder.combine_reasons(&reasons);
        assert!(explanation.contains("Trending"));
        assert!(explanation.contains("Popular"));
    }
}
