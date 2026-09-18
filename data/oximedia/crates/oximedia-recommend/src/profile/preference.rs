//! User preference learning.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Preference learner
pub struct PreferenceLearner {
    /// Learning rate
    learning_rate: f32,
    /// Decay factor for old preferences
    decay_factor: f32,
}

impl PreferenceLearner {
    /// Create a new preference learner
    #[must_use]
    pub fn new(learning_rate: f32, decay_factor: f32) -> Self {
        Self {
            learning_rate,
            decay_factor,
        }
    }

    /// Learn from positive interaction
    pub fn learn_positive(
        &self,
        preferences: &mut UserPreferences,
        features: &HashMap<String, f32>,
    ) {
        for (feature, &value) in features {
            let current = preferences.feature_weights.get(feature).unwrap_or(&0.0);
            let updated = current + self.learning_rate * value;
            preferences.feature_weights.insert(feature.clone(), updated);
        }
        preferences.apply_decay(self.decay_factor);
    }

    /// Learn from negative interaction
    pub fn learn_negative(
        &self,
        preferences: &mut UserPreferences,
        features: &HashMap<String, f32>,
    ) {
        for (feature, &value) in features {
            let current = preferences.feature_weights.get(feature).unwrap_or(&0.0);
            let updated = current - self.learning_rate * value;
            preferences.feature_weights.insert(feature.clone(), updated);
        }
        preferences.apply_decay(self.decay_factor);
    }

    /// Learn from implicit feedback
    pub fn learn_implicit(
        &self,
        preferences: &mut UserPreferences,
        features: &HashMap<String, f32>,
        strength: f32,
    ) {
        for (feature, &value) in features {
            let current = preferences.feature_weights.get(feature).unwrap_or(&0.0);
            let updated = current + self.learning_rate * value * strength;
            preferences.feature_weights.insert(feature.clone(), updated);
        }
        preferences.apply_decay(self.decay_factor);
    }
}

impl Default for PreferenceLearner {
    fn default() -> Self {
        Self::new(0.1, 0.99)
    }
}

/// User preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPreferences {
    /// User ID
    pub user_id: Uuid,
    /// Feature weights
    pub feature_weights: HashMap<String, f32>,
    /// Category preferences
    pub category_preferences: HashMap<String, f32>,
    /// Tag preferences
    pub tag_preferences: HashMap<String, f32>,
    /// Temporal preferences
    pub temporal_preferences: TemporalPreferences,
    /// Last updated
    pub last_updated: i64,
}

/// Temporal preferences (time-based patterns)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalPreferences {
    /// Hour of day preferences (0-23)
    pub hourly_weights: Vec<f32>,
    /// Day of week preferences (0-6)
    pub daily_weights: Vec<f32>,
}

impl Default for TemporalPreferences {
    fn default() -> Self {
        Self {
            hourly_weights: vec![1.0; 24],
            daily_weights: vec![1.0; 7],
        }
    }
}

impl UserPreferences {
    /// Create new user preferences
    #[must_use]
    pub fn new(user_id: Uuid) -> Self {
        Self {
            user_id,
            feature_weights: HashMap::new(),
            category_preferences: HashMap::new(),
            tag_preferences: HashMap::new(),
            temporal_preferences: TemporalPreferences::default(),
            last_updated: chrono::Utc::now().timestamp(),
        }
    }

    /// Apply decay to all preferences
    pub fn apply_decay(&mut self, decay_factor: f32) {
        for weight in self.feature_weights.values_mut() {
            *weight *= decay_factor;
        }
        for weight in self.category_preferences.values_mut() {
            *weight *= decay_factor;
        }
        for weight in self.tag_preferences.values_mut() {
            *weight *= decay_factor;
        }
        self.last_updated = chrono::Utc::now().timestamp();
    }

    /// Get preference score for features
    #[must_use]
    pub fn score_features(&self, features: &HashMap<String, f32>) -> f32 {
        let mut score = 0.0;
        for (feature, &value) in features {
            if let Some(&weight) = self.feature_weights.get(feature) {
                score += weight * value;
            }
        }
        score
    }

    /// Update category preference
    pub fn update_category(&mut self, category: &str, weight: f32) {
        *self
            .category_preferences
            .entry(category.to_string())
            .or_insert(0.0) += weight;
        self.last_updated = chrono::Utc::now().timestamp();
    }

    /// Update tag preference
    pub fn update_tag(&mut self, tag: &str, weight: f32) {
        *self.tag_preferences.entry(tag.to_string()).or_insert(0.0) += weight;
        self.last_updated = chrono::Utc::now().timestamp();
    }

    /// Update temporal preference
    pub fn update_temporal(&mut self, hour: u8, day: u8, weight: f32) {
        if (hour as usize) < self.temporal_preferences.hourly_weights.len() {
            self.temporal_preferences.hourly_weights[hour as usize] += weight;
        }
        if (day as usize) < self.temporal_preferences.daily_weights.len() {
            self.temporal_preferences.daily_weights[day as usize] += weight;
        }
        self.last_updated = chrono::Utc::now().timestamp();
    }

    /// Get top features
    #[must_use]
    pub fn get_top_features(&self, limit: usize) -> Vec<(String, f32)> {
        let mut features: Vec<(String, f32)> = self
            .feature_weights
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();

        features.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        features.truncate(limit);
        features
    }
}

/// Preference aggregator for combining multiple preference signals
pub struct PreferenceAggregator {
    /// Weights for different signals
    signal_weights: SignalWeights,
}

/// Weights for different preference signals
#[derive(Debug, Clone)]
pub struct SignalWeights {
    /// Explicit rating weight
    pub explicit_rating: f32,
    /// View completion weight
    pub view_completion: f32,
    /// Repeat view weight
    pub repeat_view: f32,
    /// Share weight
    pub share: f32,
    /// Like weight
    pub like: f32,
}

impl Default for SignalWeights {
    fn default() -> Self {
        Self {
            explicit_rating: 1.0,
            view_completion: 0.6,
            repeat_view: 0.8,
            share: 0.9,
            like: 0.7,
        }
    }
}

impl PreferenceAggregator {
    /// Create a new preference aggregator
    #[must_use]
    pub fn new(weights: SignalWeights) -> Self {
        Self {
            signal_weights: weights,
        }
    }

    /// Aggregate multiple signals into a single preference score
    #[must_use]
    pub fn aggregate(&self, signals: &PreferenceSignals) -> f32 {
        let mut score = 0.0;
        let mut total_weight = 0.0;

        if let Some(rating) = signals.explicit_rating {
            score += rating * self.signal_weights.explicit_rating;
            total_weight += self.signal_weights.explicit_rating;
        }

        if let Some(completion) = signals.view_completion {
            score += completion * self.signal_weights.view_completion;
            total_weight += self.signal_weights.view_completion;
        }

        if signals.repeat_view {
            score += self.signal_weights.repeat_view;
            total_weight += self.signal_weights.repeat_view;
        }

        if signals.shared {
            score += self.signal_weights.share;
            total_weight += self.signal_weights.share;
        }

        if signals.liked {
            score += self.signal_weights.like;
            total_weight += self.signal_weights.like;
        }

        if total_weight > 0.0 {
            score / total_weight
        } else {
            0.0
        }
    }
}

impl Default for PreferenceAggregator {
    fn default() -> Self {
        Self::new(SignalWeights::default())
    }
}

/// Preference signals from user interactions
#[derive(Debug, Clone, Default)]
pub struct PreferenceSignals {
    /// Explicit rating (0-5)
    pub explicit_rating: Option<f32>,
    /// View completion rate (0-1)
    pub view_completion: Option<f32>,
    /// Repeat view
    pub repeat_view: bool,
    /// Shared
    pub shared: bool,
    /// Liked
    pub liked: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preference_learner() {
        let learner = PreferenceLearner::new(0.1, 0.99);
        let mut prefs = UserPreferences::new(Uuid::new_v4());

        let mut features = HashMap::new();
        features.insert(String::from("action"), 1.0);

        learner.learn_positive(&mut prefs, &features);
        assert!(prefs.feature_weights.get("action").unwrap_or(&0.0) > &0.0);
    }

    #[test]
    fn test_preference_decay() {
        let mut prefs = UserPreferences::new(Uuid::new_v4());
        prefs.feature_weights.insert(String::from("test"), 1.0);

        prefs.apply_decay(0.9);
        assert!(
            (prefs
                .feature_weights
                .get("test")
                .expect("should succeed in test")
                - 0.9)
                .abs()
                < f32::EPSILON
        );
    }

    #[test]
    fn test_score_features() {
        let mut prefs = UserPreferences::new(Uuid::new_v4());
        prefs.feature_weights.insert(String::from("action"), 2.0);

        let mut features = HashMap::new();
        features.insert(String::from("action"), 1.0);

        let score = prefs.score_features(&features);
        assert!((score - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_preference_aggregator() {
        let aggregator = PreferenceAggregator::default();
        let signals = PreferenceSignals {
            explicit_rating: Some(4.0),
            view_completion: Some(0.8),
            repeat_view: true,
            shared: false,
            liked: true,
        };

        let score = aggregator.aggregate(&signals);
        assert!(score > 0.0 && score <= 5.0);
    }
}
