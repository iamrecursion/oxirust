//! Explanation generation.

use crate::error::RecommendResult;
use crate::{Recommendation, RecommendationReason};

/// Direction of a feature's contribution to the recommendation score.
#[derive(Debug, Clone, PartialEq)]
pub enum ImportanceDirection {
    /// Feature boosted the recommendation score.
    Positive,
    /// Feature reduced the recommendation score.
    Negative,
    /// Feature had no directional effect (neutral signal).
    Neutral,
}

/// A single feature's measured contribution to a recommendation score.
///
/// Used to power per-feature bar charts and visual explanations of why
/// an item was recommended.
#[derive(Debug, Clone)]
pub struct FeatureImportance {
    /// Human-readable name of the feature (e.g. "content_similarity", "trending").
    pub feature_name: String,
    /// Absolute contribution magnitude (always >= 0).
    pub contribution_score: f32,
    /// Whether this feature pushed the score up, down, or had no effect.
    pub direction: ImportanceDirection,
}

/// Decompose a [`Recommendation`]'s reasons into a list of [`FeatureImportance`].
///
/// Each [`RecommendationReason`] variant maps to one feature entry whose
/// `contribution_score` is drawn from the variant's own numeric sub-field.
/// The sum of contributions is not guaranteed to equal `rec.score` exactly
/// because multiple reasons can co-exist and the final score is the engine's
/// own blend; the values are still proportional and meaningful for charting.
fn decompose_reasons(rec: &Recommendation) -> Vec<FeatureImportance> {
    let mut importances: Vec<FeatureImportance> = rec
        .reasons
        .iter()
        .map(|reason| match reason {
            RecommendationReason::SimilarToLiked { similarity, .. } => FeatureImportance {
                feature_name: String::from("content_similarity"),
                contribution_score: *similarity,
                direction: ImportanceDirection::Positive,
            },
            RecommendationReason::CollaborativeFiltering { confidence } => FeatureImportance {
                feature_name: String::from("collaborative_filtering"),
                contribution_score: *confidence,
                direction: ImportanceDirection::Positive,
            },
            RecommendationReason::Trending { trending_score } => FeatureImportance {
                feature_name: String::from("trending_boost"),
                contribution_score: *trending_score,
                direction: ImportanceDirection::Positive,
            },
            RecommendationReason::MatchesProfile { categories } => FeatureImportance {
                feature_name: String::from("profile_match"),
                // Each matching category contributes equally; clamp to [0,1].
                contribution_score: (categories.len() as f32 * 0.1_f32).min(1.0_f32),
                direction: ImportanceDirection::Positive,
            },
            RecommendationReason::FreshContent { published_days_ago } => {
                // Newer content gets a stronger recency boost.
                let recency = 1.0_f32 / (1.0_f32 + *published_days_ago as f32 * 0.1_f32);
                FeatureImportance {
                    feature_name: String::from("recency_boost"),
                    contribution_score: recency,
                    direction: ImportanceDirection::Positive,
                }
            }
            RecommendationReason::Popular { view_count } => {
                // Log-scale popularity score capped at 1.0.
                let pop = ((*view_count as f32).ln_1p() / 20.0_f32).min(1.0_f32);
                FeatureImportance {
                    feature_name: String::from("popularity_boost"),
                    contribution_score: pop,
                    direction: ImportanceDirection::Positive,
                }
            }
            RecommendationReason::ContinueWatching { progress } => FeatureImportance {
                feature_name: String::from("continue_watching"),
                contribution_score: *progress,
                direction: ImportanceDirection::Positive,
            },
        })
        .collect();

    // Add a metadata-derived rating feature when available.
    if let Some(rating) = rec.metadata.avg_rating {
        let normalised = (rating / 5.0_f32).clamp(0.0_f32, 1.0_f32);
        importances.push(FeatureImportance {
            feature_name: String::from("avg_rating"),
            contribution_score: normalised,
            direction: if normalised >= 0.5 {
                ImportanceDirection::Positive
            } else {
                ImportanceDirection::Negative
            },
        });
    }

    importances
}

/// Generate explanation for a recommendation
///
/// # Errors
///
/// Returns an error if explanation generation fails
pub fn generate_explanation(recommendation: &Recommendation) -> RecommendResult<String> {
    let builder = super::reason::ExplanationBuilder::new();

    if recommendation.reasons.is_empty() {
        return Ok(String::from("Recommended for you"));
    }

    Ok(builder.combine_reasons(&recommendation.reasons))
}

/// Detailed explanation generator
pub struct DetailedExplanationGenerator {
    /// Include technical details
    include_technical: bool,
}

impl DetailedExplanationGenerator {
    /// Create a new detailed explanation generator
    #[must_use]
    pub fn new(include_technical: bool) -> Self {
        Self { include_technical }
    }

    /// Generate detailed explanation
    #[must_use]
    pub fn generate(&self, recommendation: &Recommendation) -> String {
        let mut explanation = String::new();

        // Add title
        explanation.push_str(&format!(
            "\"{}\" recommended because:\n\n",
            recommendation.metadata.title
        ));

        // Add reasons
        for (idx, reason) in recommendation.reasons.iter().enumerate() {
            let builder = super::reason::ExplanationBuilder::new();
            let reason_text = builder.build_explanation(reason);
            explanation.push_str(&format!("{}. {}\n", idx + 1, reason_text));
        }

        // Add technical details if requested
        if self.include_technical {
            explanation.push_str(&format!("\nRelevance Score: {:.2}\n", recommendation.score));
            explanation.push_str(&format!("Rank: #{}\n", recommendation.rank));
        }

        explanation
    }

    /// Generate a detailed explanation alongside per-feature importance scores.
    ///
    /// Returns `(explanation_text, importances)` where `importances` contains one
    /// entry per detected signal (content similarity, trending, rating, …).
    #[must_use]
    pub fn generate_with_importance(
        &self,
        recommendation: &Recommendation,
    ) -> (String, Vec<FeatureImportance>) {
        let explanation = self.generate(recommendation);
        let importances = decompose_reasons(recommendation);
        (explanation, importances)
    }

    /// Generate concise explanation
    #[must_use]
    pub fn generate_concise(&self, recommendation: &Recommendation) -> String {
        if recommendation.reasons.is_empty() {
            return String::from("Recommended for you");
        }

        let builder = super::reason::ExplanationBuilder::new();
        let primary_reason = builder.build_explanation(&recommendation.reasons[0]);

        if recommendation.reasons.len() > 1 {
            format!(
                "{} and {} more reasons",
                primary_reason,
                recommendation.reasons.len() - 1
            )
        } else {
            primary_reason
        }
    }
}

impl Default for DetailedExplanationGenerator {
    fn default() -> Self {
        Self::new(false)
    }
}

/// Explanation style
#[derive(Debug, Clone, Copy)]
pub enum ExplanationStyle {
    /// Brief, single-line explanation
    Brief,
    /// Detailed, multi-line explanation
    Detailed,
    /// Technical explanation with scores
    Technical,
}

/// Configurable explanation generator
pub struct ExplanationGenerator {
    /// Style of explanations
    style: ExplanationStyle,
    /// Detailed generator
    detailed_generator: DetailedExplanationGenerator,
}

impl ExplanationGenerator {
    /// Create a new explanation generator
    #[must_use]
    pub fn new(style: ExplanationStyle) -> Self {
        let include_technical = matches!(style, ExplanationStyle::Technical);

        Self {
            style,
            detailed_generator: DetailedExplanationGenerator::new(include_technical),
        }
    }

    /// Generate explanation based on configured style
    #[must_use]
    pub fn generate(&self, recommendation: &Recommendation) -> String {
        match self.style {
            ExplanationStyle::Brief => self.detailed_generator.generate_concise(recommendation),
            ExplanationStyle::Detailed | ExplanationStyle::Technical => {
                self.detailed_generator.generate(recommendation)
            }
        }
    }
}

impl Default for ExplanationGenerator {
    fn default() -> Self {
        Self::new(ExplanationStyle::Brief)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ContentMetadata, RecommendationReason};
    use uuid::Uuid;

    fn create_test_recommendation() -> Recommendation {
        Recommendation {
            content_id: Uuid::new_v4(),
            score: 0.85,
            rank: 1,
            reasons: vec![
                RecommendationReason::SimilarToLiked {
                    content_id: Uuid::new_v4(),
                    similarity: 0.9,
                },
                RecommendationReason::Trending {
                    trending_score: 0.8,
                },
            ],
            metadata: ContentMetadata {
                title: String::from("Test Content"),
                description: None,
                categories: vec![String::from("Action")],
                duration_ms: Some(7200000),
                thumbnail_url: None,
                created_at: 0,
                avg_rating: Some(4.5),
                view_count: 1000,
            },
            explanation: None,
        }
    }

    // ---- Feature importance tests ----

    #[test]
    fn test_feature_importance_non_empty() {
        let rec = create_test_recommendation();
        let generator = DetailedExplanationGenerator::new(false);
        let (_explanation, importances) = generator.generate_with_importance(&rec);
        // We have 2 reasons + avg_rating → at least 1 entry
        assert!(
            !importances.is_empty(),
            "expected non-empty importance list"
        );
    }

    #[test]
    fn test_feature_importance_scores_positive() {
        let rec = create_test_recommendation();
        let generator = DetailedExplanationGenerator::new(false);
        let (_explanation, importances) = generator.generate_with_importance(&rec);
        for imp in &importances {
            assert!(
                imp.contribution_score >= 0.0,
                "contribution_score must be non-negative, got {} for {}",
                imp.contribution_score,
                imp.feature_name
            );
        }
    }

    #[test]
    fn test_explanation_still_generated() {
        let rec = create_test_recommendation();
        let generator = DetailedExplanationGenerator::new(false);
        let (explanation, importances) = generator.generate_with_importance(&rec);
        assert!(
            !explanation.is_empty(),
            "explanation text must not be empty"
        );
        assert!(
            !importances.is_empty(),
            "importances must not be empty alongside explanation"
        );
    }

    #[test]
    fn test_feature_importance_direction_positive_for_similarity() {
        let rec = Recommendation {
            content_id: Uuid::new_v4(),
            score: 0.9,
            rank: 1,
            reasons: vec![RecommendationReason::SimilarToLiked {
                content_id: Uuid::new_v4(),
                similarity: 0.95,
            }],
            metadata: ContentMetadata {
                title: String::from("High Sim"),
                description: None,
                categories: vec![],
                duration_ms: None,
                thumbnail_url: None,
                created_at: 0,
                avg_rating: None,
                view_count: 0,
            },
            explanation: None,
        };
        let generator = DetailedExplanationGenerator::new(false);
        let (_exp, importances) = generator.generate_with_importance(&rec);
        let sim_imp = importances
            .iter()
            .find(|i| i.feature_name == "content_similarity")
            .expect("content_similarity entry must exist");
        assert_eq!(sim_imp.direction, ImportanceDirection::Positive);
        assert!((sim_imp.contribution_score - 0.95).abs() < 1e-5);
    }

    #[test]
    fn test_generate_explanation() {
        let rec = create_test_recommendation();
        let explanation = generate_explanation(&rec);

        assert!(explanation.is_ok());
        assert!(explanation
            .expect("should succeed in test")
            .contains("Similar to"));
    }

    #[test]
    fn test_detailed_explanation() {
        let generator = DetailedExplanationGenerator::new(false);
        let rec = create_test_recommendation();
        let explanation = generator.generate(&rec);

        assert!(explanation.contains("Test Content"));
        assert!(explanation.contains("recommended because"));
    }

    #[test]
    fn test_concise_explanation() {
        let generator = DetailedExplanationGenerator::new(false);
        let rec = create_test_recommendation();
        let explanation = generator.generate_concise(&rec);

        assert!(explanation.contains("Similar to"));
    }

    #[test]
    fn test_explanation_generator_styles() {
        let rec = create_test_recommendation();

        let brief_gen = ExplanationGenerator::new(ExplanationStyle::Brief);
        let brief = brief_gen.generate(&rec);
        assert!(!brief.is_empty());

        let detailed_gen = ExplanationGenerator::new(ExplanationStyle::Detailed);
        let detailed = detailed_gen.generate(&rec);
        assert!(detailed.len() > brief.len());
    }
}
