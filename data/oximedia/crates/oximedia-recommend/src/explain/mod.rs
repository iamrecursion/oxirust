//! Recommendation explanation generation.
//!
//! Provides human-readable explanations for why a particular item was
//! recommended to a user.

pub mod generate;
pub mod reason;

use std::fmt;

// ---------------------------------------------------------------------------
// Factor type
// ---------------------------------------------------------------------------

/// The category of factor that contributed to a recommendation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum FactorType {
    /// Recommended because it is similar to content the user has watched
    SimilarToWatched,
    /// Recommended because it is currently trending
    TrendingNow,
    /// Recommended because the user explicitly liked related content
    BecauseYouLiked,
    /// Recommended because it was recently released
    NewRelease,
    /// Recommended because it is popular within the user's preferred category
    PopularInCategory,
    /// Recommended via personalised model signals
    PersonalizedForYou,
}

impl fmt::Display for FactorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SimilarToWatched => write!(f, "Similar to what you watched"),
            Self::TrendingNow => write!(f, "Trending now"),
            Self::BecauseYouLiked => write!(f, "Because you liked"),
            Self::NewRelease => write!(f, "New release"),
            Self::PopularInCategory => write!(f, "Popular in category"),
            Self::PersonalizedForYou => write!(f, "Personalized for you"),
        }
    }
}

// ---------------------------------------------------------------------------
// Explanation factor
// ---------------------------------------------------------------------------

/// A single factor that contributed to the recommendation with its weight.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ExplanationFactor {
    /// The type of factor
    pub factor_type: FactorType,
    /// Importance weight in the range `[0.0, 1.0]`
    pub weight: f32,
    /// Human-readable description of this factor
    pub description: String,
}

impl ExplanationFactor {
    /// Create a new explanation factor.
    #[must_use]
    pub fn new(factor_type: FactorType, weight: f32, description: impl Into<String>) -> Self {
        Self {
            factor_type,
            weight: weight.clamp(0.0, 1.0),
            description: description.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// Full explanation
// ---------------------------------------------------------------------------

/// Complete explanation for a recommendation.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Explanation {
    /// The dominant reason for this recommendation
    pub primary_factor: ExplanationFactor,
    /// Additional contributing factors
    pub supporting_factors: Vec<ExplanationFactor>,
    /// Overall confidence in the recommendation (0.0–1.0)
    pub confidence: f32,
}

impl Explanation {
    /// Create a new explanation.
    #[must_use]
    pub fn new(
        primary_factor: ExplanationFactor,
        supporting_factors: Vec<ExplanationFactor>,
        confidence: f32,
    ) -> Self {
        Self {
            primary_factor,
            supporting_factors,
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Generate a natural-language sentence describing this recommendation.
    #[must_use]
    pub fn to_human_string(&self) -> String {
        let primary = match self.primary_factor.factor_type {
            FactorType::SimilarToWatched => {
                "Recommended because it is similar to content you have watched.".to_string()
            }
            FactorType::TrendingNow => "This is trending right now — don't miss out!".to_string(),
            FactorType::BecauseYouLiked => {
                format!(
                    "Because you liked similar content: {}",
                    self.primary_factor.description
                )
            }
            FactorType::NewRelease => "A brand-new release you might enjoy.".to_string(),
            FactorType::PopularInCategory => {
                format!(
                    "Popular in {}: a favourite among viewers like you.",
                    self.primary_factor.description
                )
            }
            FactorType::PersonalizedForYou => {
                "Picked just for you based on your taste profile.".to_string()
            }
        };

        if self.supporting_factors.is_empty() {
            primary
        } else {
            let supporting: Vec<&str> = self
                .supporting_factors
                .iter()
                .map(|f| f.description.as_str())
                .collect();
            format!("{} Also: {}", primary, supporting.join("; "))
        }
    }
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

/// Context used by [`ExplanationGenerator`] to build explanations.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct ExplainContext {
    /// IDs of media items the user has watched
    pub watch_history: Vec<u64>,
    /// Categories / genres the user has expressed a preference for
    pub liked_genres: Vec<String>,
    /// How strongly the item is trending (0.0–1.0)
    pub trending_score: f32,
    /// Content-based similarity score to the user's taste profile (0.0–1.0)
    pub similarity_score: f32,
}

// ---------------------------------------------------------------------------
// Explanation generator
// ---------------------------------------------------------------------------

/// Generates [`Explanation`]s for recommended media items.
#[derive(Debug, Default)]
#[allow(dead_code)]
pub struct ExplanationGenerator;

impl ExplanationGenerator {
    /// Create a new generator.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Generate an explanation for `media_id` being recommended.
    ///
    /// The primary factor is chosen by examining `context`; supporting factors
    /// are added when multiple signals are present.
    #[must_use]
    pub fn explain(
        &self,
        media_id: u64,
        user_id: Option<u64>,
        context: &ExplainContext,
    ) -> Explanation {
        let _ = media_id; // could be used for content lookup in a real system
        let _ = user_id;

        let mut factors: Vec<ExplanationFactor> = Vec::new();

        // Build candidate factors ordered by strength
        if context.trending_score >= 0.7 {
            factors.push(ExplanationFactor::new(
                FactorType::TrendingNow,
                context.trending_score,
                "Currently trending".to_string(),
            ));
        }

        if context.similarity_score >= 0.6 {
            factors.push(ExplanationFactor::new(
                FactorType::SimilarToWatched,
                context.similarity_score,
                "Matches your watch history".to_string(),
            ));
        }

        if !context.liked_genres.is_empty() {
            let genre_desc = context.liked_genres.join(", ");
            factors.push(ExplanationFactor::new(
                FactorType::PopularInCategory,
                0.65,
                genre_desc,
            ));
        }

        if !context.watch_history.is_empty() {
            factors.push(ExplanationFactor::new(
                FactorType::BecauseYouLiked,
                0.5,
                "Based on your liked content".to_string(),
            ));
        }

        // Always have at least a personalisation factor
        if factors.is_empty() {
            factors.push(ExplanationFactor::new(
                FactorType::PersonalizedForYou,
                0.4,
                "Personalised recommendation".to_string(),
            ));
        }

        // Sort by weight descending so the strongest factor is primary
        factors.sort_by(|a, b| {
            b.weight
                .partial_cmp(&a.weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let confidence = factors.first().map_or(0.4, |f| f.weight);
        let primary = factors.remove(0);
        Explanation::new(primary, factors, confidence)
    }
}

// ---------------------------------------------------------------------------
// Explanation template
// ---------------------------------------------------------------------------

/// A simple string template for rendering explanation text.
///
/// Placeholders in the template string:
/// - `{primary}` → description of the primary factor
/// - `{confidence}` → confidence percentage (0–100)
/// - `{factors}` → comma-separated descriptions of supporting factors
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ExplanationTemplate {
    /// Template string with `{primary}`, `{confidence}`, `{factors}` placeholders
    pub template: String,
}

impl ExplanationTemplate {
    /// Create a new template.
    #[must_use]
    pub fn new(template: impl Into<String>) -> Self {
        Self {
            template: template.into(),
        }
    }

    /// Render the template using the provided factors.
    ///
    /// The first factor in `factors` is treated as primary.
    #[must_use]
    pub fn render(&self, factors: &[ExplanationFactor]) -> String {
        let primary = factors
            .first()
            .map_or("your preferences", |f| f.description.as_str());

        let confidence = factors
            .first()
            .map_or(0, |f| (f.weight * 100.0).round() as u32);

        let supporting: Vec<&str> = factors
            .iter()
            .skip(1)
            .map(|f| f.description.as_str())
            .collect();
        let factors_str = if supporting.is_empty() {
            String::new()
        } else {
            supporting.join(", ")
        };

        self.template
            .replace("{primary}", primary)
            .replace("{confidence}", &confidence.to_string())
            .replace("{factors}", &factors_str)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factor_type_display() {
        assert_eq!(FactorType::TrendingNow.to_string(), "Trending now");
        assert_eq!(FactorType::NewRelease.to_string(), "New release");
    }

    #[test]
    fn test_explanation_factor_weight_clamp() {
        let f = ExplanationFactor::new(FactorType::TrendingNow, 2.5, "test");
        assert!((f.weight - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_explanation_new() {
        let primary = ExplanationFactor::new(FactorType::TrendingNow, 0.9, "Trending");
        let explanation = Explanation::new(primary, vec![], 0.9);
        assert!((explanation.confidence - 0.9).abs() < 1e-6);
        assert!(explanation.supporting_factors.is_empty());
    }

    #[test]
    fn test_explanation_to_human_string_trending() {
        let primary = ExplanationFactor::new(FactorType::TrendingNow, 0.9, "Trending");
        let explanation = Explanation::new(primary, vec![], 0.9);
        let s = explanation.to_human_string();
        assert!(s.contains("trending"));
    }

    #[test]
    fn test_explanation_to_human_string_with_supporting() {
        let primary = ExplanationFactor::new(FactorType::PersonalizedForYou, 0.8, "Personalized");
        let supporting = vec![ExplanationFactor::new(
            FactorType::TrendingNow,
            0.5,
            "Also trending",
        )];
        let explanation = Explanation::new(primary, supporting, 0.8);
        let s = explanation.to_human_string();
        assert!(s.contains("Also:"));
    }

    #[test]
    fn test_explanation_to_human_string_popular_in_category() {
        let primary = ExplanationFactor::new(FactorType::PopularInCategory, 0.7, "action");
        let explanation = Explanation::new(primary, vec![], 0.7);
        let s = explanation.to_human_string();
        assert!(s.contains("action"));
    }

    #[test]
    fn test_explanation_generator_trending() {
        let gen = ExplanationGenerator::new();
        let ctx = ExplainContext {
            trending_score: 0.9,
            ..Default::default()
        };
        let explanation = gen.explain(1, None, &ctx);
        assert_eq!(
            explanation.primary_factor.factor_type,
            FactorType::TrendingNow
        );
    }

    #[test]
    fn test_explanation_generator_similarity() {
        let gen = ExplanationGenerator::new();
        let ctx = ExplainContext {
            similarity_score: 0.85,
            ..Default::default()
        };
        let explanation = gen.explain(1, None, &ctx);
        assert_eq!(
            explanation.primary_factor.factor_type,
            FactorType::SimilarToWatched
        );
    }

    #[test]
    fn test_explanation_generator_fallback() {
        let gen = ExplanationGenerator::new();
        let ctx = ExplainContext::default();
        let explanation = gen.explain(42, Some(7), &ctx);
        assert_eq!(
            explanation.primary_factor.factor_type,
            FactorType::PersonalizedForYou
        );
    }

    #[test]
    fn test_explanation_template_render() {
        let tmpl = ExplanationTemplate::new("Why: {primary} ({confidence}% confident)");
        let factors = vec![ExplanationFactor::new(
            FactorType::TrendingNow,
            0.80,
            "Trending",
        )];
        let rendered = tmpl.render(&factors);
        assert!(rendered.contains("Trending"));
        assert!(rendered.contains("80%"));
    }

    #[test]
    fn test_explanation_template_with_factors() {
        let tmpl = ExplanationTemplate::new("{primary}. Also: {factors}");
        let factors = vec![
            ExplanationFactor::new(FactorType::TrendingNow, 0.9, "Trending"),
            ExplanationFactor::new(FactorType::NewRelease, 0.6, "New release"),
        ];
        let rendered = tmpl.render(&factors);
        assert!(rendered.contains("Trending"));
        assert!(rendered.contains("New release"));
    }
}
