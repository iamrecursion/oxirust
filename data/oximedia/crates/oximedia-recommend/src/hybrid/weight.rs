//! Method weighting for hybrid recommendations.

use serde::{Deserialize, Serialize};

/// Dynamic weight calculator for hybrid methods
pub struct DynamicWeightCalculator {
    /// Base weights
    base_weights: WeightConfig,
}

/// Weight configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightConfig {
    /// Content-based weight
    pub content_based: f32,
    /// Collaborative weight
    pub collaborative: f32,
    /// Trending weight
    pub trending: f32,
    /// Context-aware adjustment
    pub context_adjustment: bool,
}

impl Default for WeightConfig {
    fn default() -> Self {
        Self {
            content_based: 0.5,
            collaborative: 0.4,
            trending: 0.1,
            context_adjustment: true,
        }
    }
}

impl DynamicWeightCalculator {
    /// Create a new dynamic weight calculator
    #[must_use]
    pub fn new(config: WeightConfig) -> Self {
        Self {
            base_weights: config,
        }
    }

    /// Calculate weights based on context
    #[must_use]
    pub fn calculate_weights(&self, context: &WeightContext) -> CalculatedWeights {
        let mut weights = CalculatedWeights {
            content_based: self.base_weights.content_based,
            collaborative: self.base_weights.collaborative,
            trending: self.base_weights.trending,
        };

        if self.base_weights.context_adjustment {
            self.adjust_for_context(&mut weights, context);
        }

        self.normalize(&mut weights);
        weights
    }

    /// Adjust weights based on context
    fn adjust_for_context(&self, weights: &mut CalculatedWeights, context: &WeightContext) {
        // Adjust based on user history length
        if context.user_history_length < 5 {
            // New user: prefer content-based and trending
            weights.content_based *= 1.5;
            weights.trending *= 1.3;
            weights.collaborative *= 0.5;
        }

        // Adjust based on time of day
        if context.is_peak_hours {
            weights.trending *= 1.2;
        }

        // Adjust based on cold start
        if context.is_cold_start {
            weights.content_based *= 1.5;
            weights.collaborative *= 0.3;
        }
    }

    /// Normalize weights to sum to 1.0
    fn normalize(&self, weights: &mut CalculatedWeights) {
        let total = weights.content_based + weights.collaborative + weights.trending;
        if total > f32::EPSILON {
            weights.content_based /= total;
            weights.collaborative /= total;
            weights.trending /= total;
        }
    }
}

/// Context for weight calculation
#[derive(Debug, Clone)]
pub struct WeightContext {
    /// User history length
    pub user_history_length: usize,
    /// Is peak hours
    pub is_peak_hours: bool,
    /// Is cold start scenario
    pub is_cold_start: bool,
    /// User engagement level
    pub engagement_level: f32,
}

impl Default for WeightContext {
    fn default() -> Self {
        Self {
            user_history_length: 0,
            is_peak_hours: false,
            is_cold_start: true,
            engagement_level: 0.5,
        }
    }
}

/// Calculated weights for different methods
#[derive(Debug, Clone)]
pub struct CalculatedWeights {
    /// Content-based weight
    pub content_based: f32,
    /// Collaborative weight
    pub collaborative: f32,
    /// Trending weight
    pub trending: f32,
}

/// Weight learning from user feedback
pub struct WeightLearner {
    /// Learning rate
    learning_rate: f32,
    /// Current weights
    weights: WeightConfig,
}

impl WeightLearner {
    /// Create a new weight learner
    #[must_use]
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            weights: WeightConfig::default(),
        }
    }

    /// Update weights based on user feedback
    pub fn update_from_feedback(&mut self, method: &str, positive: bool) {
        let adjustment = if positive {
            self.learning_rate
        } else {
            -self.learning_rate
        };

        match method {
            "content_based" => {
                self.weights.content_based = (self.weights.content_based + adjustment).max(0.0);
            }
            "collaborative" => {
                self.weights.collaborative = (self.weights.collaborative + adjustment).max(0.0);
            }
            "trending" => {
                self.weights.trending = (self.weights.trending + adjustment).max(0.0);
            }
            _ => {}
        }

        self.normalize_weights();
    }

    /// Normalize weights
    fn normalize_weights(&mut self) {
        let total = self.weights.content_based + self.weights.collaborative + self.weights.trending;
        if total > f32::EPSILON {
            self.weights.content_based /= total;
            self.weights.collaborative /= total;
            self.weights.trending /= total;
        }
    }

    /// Get current weights
    #[must_use]
    pub fn get_weights(&self) -> &WeightConfig {
        &self.weights
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_config_default() {
        let config = WeightConfig::default();
        assert!((config.content_based - 0.5).abs() < f32::EPSILON);
        assert!(config.context_adjustment);
    }

    #[test]
    fn test_dynamic_weight_calculator() {
        let calculator = DynamicWeightCalculator::new(WeightConfig::default());
        let context = WeightContext::default();
        let weights = calculator.calculate_weights(&context);

        let total = weights.content_based + weights.collaborative + weights.trending;
        assert!((total - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_weight_normalization() {
        let calculator = DynamicWeightCalculator::new(WeightConfig::default());
        let context = WeightContext {
            user_history_length: 0,
            is_peak_hours: false,
            is_cold_start: true,
            engagement_level: 0.5,
        };

        let weights = calculator.calculate_weights(&context);
        let total = weights.content_based + weights.collaborative + weights.trending;
        assert!((total - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_weight_learner() {
        let mut learner = WeightLearner::new(0.1);
        learner.update_from_feedback("content_based", true);

        let weights = learner.get_weights();
        assert!(weights.content_based > 0.5);
    }
}
