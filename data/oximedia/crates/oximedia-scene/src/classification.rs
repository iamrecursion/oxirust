//! Scene classification: genre detection, mood estimation, and location tags.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use serde::{Deserialize, Serialize};

/// Top-level genre of a scene.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Genre {
    /// Action / adventure content.
    Action,
    /// Documentary style.
    Documentary,
    /// Drama / narrative film.
    Drama,
    /// Comedy with bright tones.
    Comedy,
    /// Horror / suspense content.
    Horror,
    /// News / broadcast content.
    News,
    /// Sports broadcast.
    Sports,
    /// Nature / wildlife footage.
    Nature,
    /// Unknown genre.
    Unknown,
}

/// Estimated mood of a scene.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Mood {
    /// Happy / uplifting.
    Happy,
    /// Sad / melancholic.
    Sad,
    /// Tense / suspenseful.
    Tense,
    /// Calm / peaceful.
    Calm,
    /// Exciting / energetic.
    Exciting,
    /// Dark / ominous.
    Dark,
    /// Neutral.
    Neutral,
}

/// Broad location category for a scene.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LocationTag {
    /// Interior location.
    Indoor,
    /// Exterior location.
    Outdoor,
    /// Urban environment.
    Urban,
    /// Rural / countryside.
    Rural,
    /// Studio setting.
    Studio,
    /// Natural landscape.
    Nature,
    /// Underwater setting.
    Underwater,
    /// Aerial / airborne.
    Aerial,
    /// Unknown location.
    Unknown,
}

/// Result produced by scene classification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationResult {
    /// Detected genre with confidence.
    pub genre: GenreScore,
    /// Estimated mood.
    pub mood: MoodScore,
    /// Location tags, ordered by confidence.
    pub location_tags: Vec<LocationScore>,
}

/// Genre with an associated confidence score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenreScore {
    /// The genre label.
    pub genre: Genre,
    /// Confidence in [0, 1].
    pub confidence: f32,
}

/// Mood with an associated confidence score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoodScore {
    /// The mood label.
    pub mood: Mood,
    /// Confidence in [0, 1].
    pub confidence: f32,
}

/// Location tag with confidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocationScore {
    /// The location tag.
    pub tag: LocationTag,
    /// Confidence in [0, 1].
    pub confidence: f32,
}

/// Feature vector used for classification.
#[derive(Debug, Clone)]
pub struct SceneFeatures {
    /// Mean luminance of the frame (0–255).
    pub mean_luminance: f32,
    /// Mean saturation of the frame (0–255).
    pub mean_saturation: f32,
    /// Dominant hue angle in degrees (0–360).
    pub dominant_hue: f32,
    /// Estimated motion magnitude (0–1).
    pub motion_magnitude: f32,
    /// Ratio of sky pixels detected (0–1).
    pub sky_ratio: f32,
    /// Ratio of skin-tone pixels (0–1).
    pub skin_ratio: f32,
}

impl SceneFeatures {
    /// Create a new `SceneFeatures` instance.
    #[must_use]
    pub fn new(
        mean_luminance: f32,
        mean_saturation: f32,
        dominant_hue: f32,
        motion_magnitude: f32,
        sky_ratio: f32,
        skin_ratio: f32,
    ) -> Self {
        Self {
            mean_luminance,
            mean_saturation,
            dominant_hue,
            motion_magnitude,
            sky_ratio,
            skin_ratio,
        }
    }
}

/// Classifier that estimates genre, mood, and location tags from scene features.
#[derive(Debug, Default)]
pub struct SceneClassifier {
    /// Confidence threshold below which results are discarded.
    threshold: f32,
}

impl SceneClassifier {
    /// Create a classifier with a default confidence threshold of 0.3.
    #[must_use]
    pub fn new() -> Self {
        Self { threshold: 0.3 }
    }

    /// Create a classifier with a custom confidence threshold.
    #[must_use]
    pub fn with_threshold(threshold: f32) -> Self {
        Self {
            threshold: threshold.clamp(0.0, 1.0),
        }
    }

    /// Classify a scene from its extracted features.
    #[must_use]
    pub fn classify(&self, features: &SceneFeatures) -> ClassificationResult {
        ClassificationResult {
            genre: self.detect_genre(features),
            mood: self.estimate_mood(features),
            location_tags: self.tag_location(features),
        }
    }

    /// Detect the primary genre from feature heuristics.
    fn detect_genre(&self, f: &SceneFeatures) -> GenreScore {
        // High motion + high saturation → Action
        if f.motion_magnitude > 0.6 && f.mean_saturation > 150.0 {
            return GenreScore {
                genre: Genre::Action,
                confidence: 0.7_f32.max(self.threshold),
            };
        }
        // High sky ratio + low saturation → Documentary / Nature
        if f.sky_ratio > 0.4 {
            return GenreScore {
                genre: Genre::Nature,
                confidence: 0.65,
            };
        }
        // Dark scene with low saturation → Horror
        if f.mean_luminance < 60.0 && f.mean_saturation < 80.0 {
            return GenreScore {
                genre: Genre::Horror,
                confidence: 0.6,
            };
        }
        GenreScore {
            genre: Genre::Unknown,
            confidence: 0.4,
        }
    }

    /// Estimate the scene mood.
    fn estimate_mood(&self, f: &SceneFeatures) -> MoodScore {
        if f.mean_luminance > 180.0 && f.mean_saturation > 120.0 {
            return MoodScore {
                mood: Mood::Happy,
                confidence: 0.75,
            };
        }
        if f.mean_luminance < 80.0 {
            return MoodScore {
                mood: Mood::Dark,
                confidence: 0.7,
            };
        }
        if f.motion_magnitude > 0.7 {
            return MoodScore {
                mood: Mood::Exciting,
                confidence: 0.65,
            };
        }
        if f.sky_ratio > 0.3 && f.motion_magnitude < 0.2 {
            return MoodScore {
                mood: Mood::Calm,
                confidence: 0.6,
            };
        }
        MoodScore {
            mood: Mood::Neutral,
            confidence: 0.5,
        }
    }

    /// Produce location tags sorted by confidence.
    fn tag_location(&self, f: &SceneFeatures) -> Vec<LocationScore> {
        let mut tags: Vec<LocationScore> = Vec::new();

        // Outdoor heuristic: significant sky
        if f.sky_ratio > 0.2 {
            tags.push(LocationScore {
                tag: LocationTag::Outdoor,
                confidence: (0.5 + f.sky_ratio * 0.5).min(1.0),
            });
        } else {
            tags.push(LocationScore {
                tag: LocationTag::Indoor,
                confidence: 0.5 + (1.0 - f.sky_ratio) * 0.3,
            });
        }

        // Nature: high sky + low skin
        if f.sky_ratio > 0.3 && f.skin_ratio < 0.1 {
            tags.push(LocationScore {
                tag: LocationTag::Nature,
                confidence: 0.6,
            });
        }

        // Aerial: very high sky ratio
        if f.sky_ratio > 0.7 {
            tags.push(LocationScore {
                tag: LocationTag::Aerial,
                confidence: 0.55,
            });
        }

        // Studio: no sky + skin present
        if f.sky_ratio < 0.05 && f.skin_ratio > 0.15 {
            tags.push(LocationScore {
                tag: LocationTag::Studio,
                confidence: 0.55,
            });
        }

        // Filter by threshold and sort
        tags.retain(|t| t.confidence >= self.threshold);
        tags.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        tags
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn action_features() -> SceneFeatures {
        SceneFeatures::new(140.0, 180.0, 30.0, 0.8, 0.1, 0.2)
    }

    fn dark_features() -> SceneFeatures {
        SceneFeatures::new(50.0, 60.0, 200.0, 0.1, 0.05, 0.05)
    }

    fn nature_features() -> SceneFeatures {
        SceneFeatures::new(160.0, 100.0, 120.0, 0.15, 0.55, 0.02)
    }

    fn bright_features() -> SceneFeatures {
        SceneFeatures::new(200.0, 140.0, 60.0, 0.1, 0.1, 0.3)
    }

    #[test]
    fn test_classifier_new() {
        let c = SceneClassifier::new();
        assert!((c.threshold - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn test_classifier_with_threshold() {
        let c = SceneClassifier::with_threshold(0.5);
        assert!((c.threshold - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_threshold_clamped_high() {
        let c = SceneClassifier::with_threshold(2.0);
        assert!((c.threshold - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_threshold_clamped_low() {
        let c = SceneClassifier::with_threshold(-1.0);
        assert!((c.threshold - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_genre_action_detected() {
        let c = SceneClassifier::new();
        let r = c.classify(&action_features());
        assert_eq!(r.genre.genre, Genre::Action);
        assert!(r.genre.confidence >= 0.3);
    }

    #[test]
    fn test_genre_horror_detected() {
        let c = SceneClassifier::new();
        let r = c.classify(&dark_features());
        assert_eq!(r.genre.genre, Genre::Horror);
    }

    #[test]
    fn test_genre_nature_detected() {
        let c = SceneClassifier::new();
        let r = c.classify(&nature_features());
        assert_eq!(r.genre.genre, Genre::Nature);
    }

    #[test]
    fn test_mood_dark() {
        let c = SceneClassifier::new();
        let r = c.classify(&dark_features());
        assert_eq!(r.mood.mood, Mood::Dark);
    }

    #[test]
    fn test_mood_happy_bright() {
        let c = SceneClassifier::new();
        let r = c.classify(&bright_features());
        assert_eq!(r.mood.mood, Mood::Happy);
    }

    #[test]
    fn test_mood_exciting_action() {
        let c = SceneClassifier::new();
        let r = c.classify(&action_features());
        assert_eq!(r.mood.mood, Mood::Exciting);
    }

    #[test]
    fn test_location_tags_not_empty() {
        let c = SceneClassifier::new();
        let r = c.classify(&nature_features());
        assert!(!r.location_tags.is_empty());
    }

    #[test]
    fn test_location_outdoor_nature() {
        let c = SceneClassifier::new();
        let r = c.classify(&nature_features());
        let has_outdoor = r
            .location_tags
            .iter()
            .any(|t| t.tag == LocationTag::Outdoor);
        assert!(has_outdoor);
    }

    #[test]
    fn test_location_tags_sorted_by_confidence() {
        let c = SceneClassifier::new();
        let r = c.classify(&nature_features());
        let confs: Vec<f32> = r.location_tags.iter().map(|t| t.confidence).collect();
        for w in confs.windows(2) {
            assert!(w[0] >= w[1]);
        }
    }

    #[test]
    fn test_scene_features_new() {
        let f = SceneFeatures::new(100.0, 80.0, 180.0, 0.3, 0.2, 0.1);
        assert!((f.mean_luminance - 100.0).abs() < f32::EPSILON);
        assert!((f.dominant_hue - 180.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_confidence_in_range() {
        let c = SceneClassifier::new();
        let r = c.classify(&action_features());
        assert!(r.genre.confidence >= 0.0 && r.genre.confidence <= 1.0);
        assert!(r.mood.confidence >= 0.0 && r.mood.confidence <= 1.0);
    }
}
