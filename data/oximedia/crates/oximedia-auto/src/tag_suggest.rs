//! Automatic tag suggestion for media assets.
//!
//! Provides keyword extraction, entity recognition, and category mapping
//! to generate descriptive tags from media content.

#![allow(dead_code)]

use std::collections::HashMap;

/// A suggested tag with confidence score.
#[derive(Debug, Clone, PartialEq)]
pub struct TagSuggestion {
    /// The tag text.
    pub tag: String,
    /// Confidence in range [0.0, 1.0].
    pub confidence: f32,
    /// Category of the tag.
    pub category: TagCategory,
}

impl TagSuggestion {
    /// Create a new tag suggestion.
    pub fn new(tag: impl Into<String>, confidence: f32, category: TagCategory) -> Self {
        Self {
            tag: tag.into(),
            confidence: confidence.clamp(0.0, 1.0),
            category,
        }
    }
}

/// Category of a tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TagCategory {
    /// Subject matter.
    Subject,
    /// Named entity (person, org, location).
    Entity,
    /// Visual style or aesthetic.
    Style,
    /// Action or event.
    Action,
    /// Technical attribute.
    Technical,
    /// Emotion or mood.
    Mood,
    /// Genre classification.
    Genre,
}

/// Input features for tag suggestion.
#[derive(Debug, Clone, Default)]
pub struct TagInputFeatures {
    /// Free-form description text (e.g. from ASR or title).
    pub description: String,
    /// Scene labels detected in video.
    pub scene_labels: Vec<String>,
    /// Object labels detected in video.
    pub object_labels: Vec<String>,
    /// Dominant mood score (positive 0-1, negative 0-1).
    pub mood_positive: f32,
    /// Dominant mood score.
    pub mood_negative: f32,
    /// Detected audio events.
    pub audio_events: Vec<String>,
}

/// Configuration for tag suggestion.
#[derive(Debug, Clone)]
pub struct TagSuggestConfig {
    /// Minimum confidence threshold.
    pub min_confidence: f32,
    /// Maximum tags to return per category.
    pub max_per_category: usize,
    /// Maximum tags to return in total.
    pub max_total: usize,
    /// Whether to deduplicate similar tags.
    pub deduplicate: bool,
}

impl Default for TagSuggestConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.3,
            max_per_category: 5,
            max_total: 20,
            deduplicate: true,
        }
    }
}

/// Tag suggester engine.
#[derive(Debug, Clone)]
pub struct TagSuggester {
    config: TagSuggestConfig,
    /// Static keyword-to-category mapping.
    keyword_map: HashMap<String, TagCategory>,
}

impl TagSuggester {
    /// Create a new tag suggester.
    pub fn new(config: TagSuggestConfig) -> Self {
        let mut keyword_map = HashMap::new();
        // Populate with common keywords
        for kw in &["interview", "talk", "speech", "lecture", "presentation"] {
            keyword_map.insert((*kw).to_string(), TagCategory::Genre);
        }
        for kw in &["outdoor", "indoor", "studio", "landscape", "city"] {
            keyword_map.insert((*kw).to_string(), TagCategory::Subject);
        }
        for kw in &["running", "jumping", "walking", "dancing", "driving"] {
            keyword_map.insert((*kw).to_string(), TagCategory::Action);
        }
        for kw in &["happy", "sad", "excited", "calm", "tense"] {
            keyword_map.insert((*kw).to_string(), TagCategory::Mood);
        }
        for kw in &["4k", "hd", "slow-motion", "timelapse", "drone"] {
            keyword_map.insert((*kw).to_string(), TagCategory::Technical);
        }
        Self {
            config,
            keyword_map,
        }
    }

    /// Suggest tags from input features.
    pub fn suggest(&self, features: &TagInputFeatures) -> Vec<TagSuggestion> {
        let mut suggestions: Vec<TagSuggestion> = Vec::new();

        // Extract from description
        suggestions.extend(self.extract_from_text(&features.description));

        // Convert scene labels
        for label in &features.scene_labels {
            let category = self.categorize(label);
            suggestions.push(TagSuggestion::new(label.to_lowercase(), 0.85, category));
        }

        // Convert object labels
        for label in &features.object_labels {
            suggestions.push(TagSuggestion::new(
                label.to_lowercase(),
                0.75,
                TagCategory::Subject,
            ));
        }

        // Mood tags
        if features.mood_positive > 0.6 {
            suggestions.push(TagSuggestion::new(
                "upbeat",
                features.mood_positive,
                TagCategory::Mood,
            ));
        }
        if features.mood_negative > 0.6 {
            suggestions.push(TagSuggestion::new(
                "somber",
                features.mood_negative,
                TagCategory::Mood,
            ));
        }

        // Audio events
        for event in &features.audio_events {
            suggestions.push(TagSuggestion::new(
                event.to_lowercase(),
                0.70,
                TagCategory::Subject,
            ));
        }

        self.filter_and_rank(suggestions)
    }

    /// Extract keywords from free-form text.
    pub fn extract_from_text(&self, text: &str) -> Vec<TagSuggestion> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut tags = Vec::new();

        for word in &words {
            let lower = word.to_lowercase();
            let lower = lower.trim_matches(|c: char| !c.is_alphanumeric());
            if lower.len() >= 3 {
                if let Some(&cat) = self.keyword_map.get(lower) {
                    tags.push(TagSuggestion::new(lower.to_string(), 0.65, cat));
                }
            }
        }

        tags
    }

    /// Map a label to a tag category.
    fn categorize(&self, label: &str) -> TagCategory {
        let lower = label.to_lowercase();
        self.keyword_map
            .get(&lower)
            .copied()
            .unwrap_or(TagCategory::Subject)
    }

    /// Filter by confidence, deduplicate, and limit counts.
    fn filter_and_rank(&self, mut suggestions: Vec<TagSuggestion>) -> Vec<TagSuggestion> {
        // Remove low confidence
        suggestions.retain(|s| s.confidence >= self.config.min_confidence);

        // Sort by confidence descending
        suggestions.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Deduplicate by tag text
        if self.config.deduplicate {
            let mut seen = std::collections::HashSet::new();
            suggestions.retain(|s| seen.insert(s.tag.clone()));
        }

        // Limit per category
        let mut category_counts: HashMap<TagCategory, usize> = HashMap::new();
        suggestions.retain(|s| {
            let count = category_counts.entry(s.category).or_insert(0);
            if *count < self.config.max_per_category {
                *count += 1;
                true
            } else {
                false
            }
        });

        // Limit total
        suggestions.truncate(self.config.max_total);
        suggestions
    }
}

impl Default for TagSuggester {
    fn default() -> Self {
        Self::new(TagSuggestConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_suggester() -> TagSuggester {
        TagSuggester::default()
    }

    #[test]
    fn test_tag_suggestion_clamps_confidence() {
        let tag = TagSuggestion::new("test", 1.5, TagCategory::Subject);
        assert_eq!(tag.confidence, 1.0);

        let tag2 = TagSuggestion::new("test2", -0.5, TagCategory::Subject);
        assert_eq!(tag2.confidence, 0.0);
    }

    #[test]
    fn test_suggest_from_scene_labels() {
        let suggester = make_suggester();
        let features = TagInputFeatures {
            scene_labels: vec!["outdoor".to_string(), "landscape".to_string()],
            ..Default::default()
        };
        let tags = suggester.suggest(&features);
        assert!(!tags.is_empty());
        assert!(tags
            .iter()
            .any(|t| t.tag == "outdoor" || t.tag == "landscape"));
    }

    #[test]
    fn test_suggest_from_object_labels() {
        let suggester = make_suggester();
        let features = TagInputFeatures {
            object_labels: vec!["car".to_string(), "tree".to_string()],
            ..Default::default()
        };
        let tags = suggester.suggest(&features);
        assert!(tags.iter().any(|t| t.tag == "car" || t.tag == "tree"));
    }

    #[test]
    fn test_suggest_mood_positive() {
        let suggester = make_suggester();
        let features = TagInputFeatures {
            mood_positive: 0.9,
            ..Default::default()
        };
        let tags = suggester.suggest(&features);
        assert!(tags.iter().any(|t| t.tag == "upbeat"));
    }

    #[test]
    fn test_suggest_mood_negative() {
        let suggester = make_suggester();
        let features = TagInputFeatures {
            mood_negative: 0.8,
            ..Default::default()
        };
        let tags = suggester.suggest(&features);
        assert!(tags.iter().any(|t| t.tag == "somber"));
    }

    #[test]
    fn test_extract_from_text_keywords() {
        let suggester = make_suggester();
        let tags = suggester.extract_from_text("This is an outdoor interview");
        // "outdoor" maps to Subject, "interview" maps to Genre
        let has_outdoor = tags.iter().any(|t| t.tag == "outdoor");
        let has_interview = tags.iter().any(|t| t.tag == "interview");
        assert!(has_outdoor || has_interview);
    }

    #[test]
    fn test_deduplication() {
        let suggester = make_suggester();
        let features = TagInputFeatures {
            scene_labels: vec!["outdoor".to_string()],
            object_labels: vec!["outdoor".to_string()],
            ..Default::default()
        };
        let tags = suggester.suggest(&features);
        let outdoor_count = tags.iter().filter(|t| t.tag == "outdoor").count();
        assert_eq!(outdoor_count, 1, "Duplicates should be removed");
    }

    #[test]
    fn test_confidence_threshold() {
        let config = TagSuggestConfig {
            min_confidence: 0.9,
            ..Default::default()
        };
        let suggester = TagSuggester::new(config);
        let features = TagInputFeatures {
            mood_positive: 0.5, // below threshold
            ..Default::default()
        };
        let tags = suggester.suggest(&features);
        assert!(tags.iter().all(|t| t.confidence >= 0.9));
    }

    #[test]
    fn test_max_total_limit() {
        let config = TagSuggestConfig {
            max_total: 3,
            min_confidence: 0.0,
            ..Default::default()
        };
        let suggester = TagSuggester::new(config);
        let features = TagInputFeatures {
            scene_labels: vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string(),
                "e".to_string(),
            ],
            ..Default::default()
        };
        let tags = suggester.suggest(&features);
        assert!(tags.len() <= 3);
    }

    #[test]
    fn test_max_per_category() {
        let config = TagSuggestConfig {
            max_per_category: 2,
            max_total: 100,
            min_confidence: 0.0,
            deduplicate: true,
        };
        let suggester = TagSuggester::new(config);
        let features = TagInputFeatures {
            scene_labels: vec![
                "outdoor".to_string(),
                "indoor".to_string(),
                "studio".to_string(),
            ],
            ..Default::default()
        };
        let tags = suggester.suggest(&features);
        let subject_count = tags
            .iter()
            .filter(|t| t.category == TagCategory::Subject)
            .count();
        // "outdoor", "indoor", "studio" all map to Subject (via keyword_map), limit is 2
        assert!(subject_count <= 2);
    }

    #[test]
    fn test_audio_events_included() {
        let suggester = make_suggester();
        let features = TagInputFeatures {
            audio_events: vec!["applause".to_string()],
            ..Default::default()
        };
        let tags = suggester.suggest(&features);
        assert!(tags.iter().any(|t| t.tag == "applause"));
    }

    #[test]
    fn test_empty_features_returns_empty() {
        let suggester = make_suggester();
        let features = TagInputFeatures::default();
        let tags = suggester.suggest(&features);
        // No positive/negative mood triggers, no labels
        // mood_positive and mood_negative are 0.0 so no mood tags either
        assert!(tags.is_empty());
    }

    #[test]
    fn test_tags_sorted_by_confidence_descending() {
        let suggester = make_suggester();
        let features = TagInputFeatures {
            scene_labels: vec!["outdoor".to_string()],
            object_labels: vec!["car".to_string()],
            mood_positive: 0.95,
            ..Default::default()
        };
        let tags = suggester.suggest(&features);
        for w in tags.windows(2) {
            assert!(w[0].confidence >= w[1].confidence);
        }
    }

    #[test]
    fn test_categorize_unknown_label() {
        let suggester = make_suggester();
        let cat = suggester.categorize("something_unusual_xyz");
        assert_eq!(cat, TagCategory::Subject);
    }

    #[test]
    fn test_tag_category_equality() {
        assert_eq!(TagCategory::Subject, TagCategory::Subject);
        assert_ne!(TagCategory::Subject, TagCategory::Mood);
    }
}
