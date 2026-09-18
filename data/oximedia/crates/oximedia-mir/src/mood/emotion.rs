//! Emotion classification.

/// Emotion classifier.
pub struct EmotionClassifier;

impl EmotionClassifier {
    /// Create a new emotion classifier.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Classify emotions from valence-arousal values.
    #[must_use]
    pub fn classify(&self, valence: f32, arousal: f32) -> Vec<String> {
        let mut emotions = Vec::new();

        if valence > 0.3 && arousal > 0.6 {
            emotions.push("joy".to_string());
        }
        if valence < -0.3 && arousal > 0.6 {
            emotions.push("anger".to_string());
        }
        if valence < -0.3 && arousal < 0.4 {
            emotions.push("sadness".to_string());
        }
        if valence > 0.3 && arousal < 0.4 {
            emotions.push("contentment".to_string());
        }
        if valence.abs() < 0.3 && arousal < 0.4 {
            emotions.push("neutral".to_string());
        }

        if emotions.is_empty() {
            emotions.push("neutral".to_string());
        }

        emotions
    }
}

impl Default for EmotionClassifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emotion_classifier_creation() {
        let _classifier = EmotionClassifier::new();
    }

    #[test]
    fn test_classify_joy() {
        let classifier = EmotionClassifier::new();
        let emotions = classifier.classify(0.8, 0.9);
        assert!(emotions.contains(&"joy".to_string()));
    }
}
