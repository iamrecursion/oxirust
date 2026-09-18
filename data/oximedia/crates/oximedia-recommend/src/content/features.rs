//! Content feature extraction for similarity calculation.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Content features for similarity calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentFeatures {
    /// Categories/genres
    pub categories: Vec<String>,
    /// Tags/keywords
    pub tags: Vec<String>,
    /// Duration (milliseconds)
    pub duration_ms: Option<i64>,
    /// Language
    pub language: Option<String>,
    /// Production year
    pub year: Option<u16>,
    /// Content type (video, audio, image)
    pub content_type: String,
    /// Quality indicators
    pub quality_features: QualityFeatures,
    /// Engagement features
    pub engagement_features: EngagementFeatures,
    /// Textual features
    pub text_features: TextFeatures,
    /// Custom features
    pub custom_features: HashMap<String, f32>,
}

/// Quality-related features
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QualityFeatures {
    /// Video resolution (if applicable)
    pub resolution: Option<String>,
    /// Bitrate (kbps)
    pub bitrate: Option<u32>,
    /// Frame rate
    pub framerate: Option<f32>,
    /// Has HDR
    pub has_hdr: bool,
    /// Audio quality
    pub audio_quality: Option<String>,
}

/// Engagement-related features
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EngagementFeatures {
    /// Average rating
    pub avg_rating: Option<f32>,
    /// Total views
    pub view_count: u64,
    /// Like count
    pub like_count: u64,
    /// Comment count
    pub comment_count: u64,
    /// Share count
    pub share_count: u64,
    /// Completion rate
    pub completion_rate: Option<f32>,
}

/// Text-based features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextFeatures {
    /// Title
    pub title: String,
    /// Description
    pub description: Option<String>,
    /// Transcript (if available)
    pub transcript: Option<String>,
    /// Named entities
    pub entities: Vec<String>,
}

impl ContentFeatures {
    /// Create new content features
    #[must_use]
    pub fn new(title: String, categories: Vec<String>) -> Self {
        Self {
            categories,
            tags: Vec::new(),
            duration_ms: None,
            language: None,
            year: None,
            content_type: String::from("video"),
            quality_features: QualityFeatures::default(),
            engagement_features: EngagementFeatures::default(),
            text_features: TextFeatures {
                title,
                description: None,
                transcript: None,
                entities: Vec::new(),
            },
            custom_features: HashMap::new(),
        }
    }

    /// Extract categorical features as a set
    #[must_use]
    pub fn categorical_features(&self) -> Vec<String> {
        let mut features = Vec::new();
        features.extend(self.categories.clone());
        features.extend(self.tags.clone());
        if let Some(ref lang) = self.language {
            features.push(format!("lang:{lang}"));
        }
        if let Some(year) = self.year {
            features.push(format!("year:{year}"));
        }
        features.push(format!("type:{}", self.content_type));
        features
    }

    /// Extract numerical features as a vector
    #[must_use]
    pub fn numerical_features(&self) -> Vec<f32> {
        let mut features = Vec::new();

        // Duration (normalized to hours)
        features.push(self.duration_ms.unwrap_or(0) as f32 / 3_600_000.0);

        // Engagement metrics (normalized)
        features.push(self.engagement_features.avg_rating.unwrap_or(0.0) / 5.0);
        features.push((self.engagement_features.view_count as f32).ln().max(0.0) / 20.0);
        features.push((self.engagement_features.like_count as f32).ln().max(0.0) / 15.0);
        features.push(self.engagement_features.completion_rate.unwrap_or(0.0));

        // Quality metrics
        if let Some(bitrate) = self.quality_features.bitrate {
            features.push((bitrate as f32).ln() / 15.0);
        } else {
            features.push(0.0);
        }

        features.push(f32::from(self.quality_features.has_hdr));

        // Add custom features
        for value in self.custom_features.values() {
            features.push(*value);
        }

        features
    }
}

/// Feature extractor for content
pub struct FeatureExtractor;

impl FeatureExtractor {
    /// Extract features from content metadata
    #[must_use]
    pub fn extract(_metadata: &crate::ContentMetadata) -> ContentFeatures {
        // In a real implementation, this would extract features from metadata
        ContentFeatures::new(String::from("Sample"), vec![String::from("category")])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_content_features_creation() {
        let features = ContentFeatures::new(
            String::from("Test Video"),
            vec![String::from("Action"), String::from("Adventure")],
        );
        assert_eq!(features.categories.len(), 2);
        assert_eq!(features.text_features.title, "Test Video");
    }

    #[test]
    fn test_categorical_features() {
        let mut features = ContentFeatures::new(String::from("Test"), vec![String::from("Drama")]);
        features.language = Some(String::from("en"));
        features.year = Some(2024);

        let categorical = features.categorical_features();
        assert!(!categorical.is_empty());
        assert!(categorical.contains(&String::from("Drama")));
    }

    #[test]
    fn test_numerical_features() {
        let features = ContentFeatures::new(String::from("Test"), vec![]);
        let numerical = features.numerical_features();
        assert!(!numerical.is_empty());
    }

    #[test]
    fn test_quality_features_default() {
        let quality = QualityFeatures::default();
        assert!(!quality.has_hdr);
        assert!(quality.resolution.is_none());
    }

    #[test]
    fn test_engagement_features_default() {
        let engagement = EngagementFeatures::default();
        assert_eq!(engagement.view_count, 0);
        assert_eq!(engagement.like_count, 0);
    }
}
