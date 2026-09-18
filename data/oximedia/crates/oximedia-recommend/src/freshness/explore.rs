//! Exploration vs exploitation balance.

use serde::{Deserialize, Serialize};

/// Epsilon-greedy exploration strategy
pub struct EpsilonGreedy {
    /// Exploration rate (0-1)
    epsilon: f32,
}

impl EpsilonGreedy {
    /// Create a new epsilon-greedy strategy
    #[must_use]
    pub fn new(epsilon: f32) -> Self {
        Self {
            epsilon: epsilon.clamp(0.0, 1.0),
        }
    }

    /// Decide whether to explore or exploit
    #[must_use]
    pub fn should_explore(&self) -> bool {
        // In a real implementation, would use proper random number generation
        // For now, use a deterministic approach
        self.epsilon > 0.5
    }

    /// Get exploration probability
    #[must_use]
    pub fn exploration_probability(&self) -> f32 {
        self.epsilon
    }

    /// Decay epsilon over time (for adaptive exploration)
    pub fn decay(&mut self, decay_rate: f32) {
        self.epsilon *= 1.0 - decay_rate;
        self.epsilon = self.epsilon.max(0.01); // Minimum exploration
    }
}

impl Default for EpsilonGreedy {
    fn default() -> Self {
        Self::new(0.1) // 10% exploration by default
    }
}

/// Upper Confidence Bound (UCB) strategy
pub struct UpperConfidenceBound {
    /// Exploration parameter
    c: f32,
}

impl UpperConfidenceBound {
    /// Create a new UCB strategy
    #[must_use]
    pub fn new(c: f32) -> Self {
        Self { c }
    }

    /// Calculate UCB score
    #[must_use]
    pub fn calculate_score(
        &self,
        avg_reward: f32,
        total_selections: u32,
        item_selections: u32,
    ) -> f32 {
        if item_selections == 0 {
            return f32::INFINITY; // Explore items never selected
        }

        let exploration_term = ((total_selections as f32).ln() / item_selections as f32).sqrt();
        avg_reward + self.c * exploration_term
    }
}

impl Default for UpperConfidenceBound {
    fn default() -> Self {
        Self::new(1.0)
    }
}

/// Thompson Sampling strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThompsonSampling {
    /// Alpha parameter (successes + 1)
    alpha: f32,
    /// Beta parameter (failures + 1)
    beta: f32,
}

impl ThompsonSampling {
    /// Create a new Thompson Sampling strategy
    #[must_use]
    pub fn new() -> Self {
        Self {
            alpha: 1.0,
            beta: 1.0,
        }
    }

    /// Update with observation
    pub fn update(&mut self, success: bool) {
        if success {
            self.alpha += 1.0;
        } else {
            self.beta += 1.0;
        }
    }

    /// Get expected value
    #[must_use]
    pub fn expected_value(&self) -> f32 {
        self.alpha / (self.alpha + self.beta)
    }
}

impl Default for ThompsonSampling {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_epsilon_greedy() {
        let mut strategy = EpsilonGreedy::new(0.2);
        assert!((strategy.exploration_probability() - 0.2).abs() < f32::EPSILON);

        strategy.decay(0.1);
        assert!(strategy.exploration_probability() < 0.2);
    }

    #[test]
    fn test_ucb() {
        let ucb = UpperConfidenceBound::new(1.0);
        let score = ucb.calculate_score(0.5, 100, 10);
        assert!(score > 0.5);
    }

    #[test]
    fn test_thompson_sampling() {
        let mut ts = ThompsonSampling::new();
        ts.update(true);
        ts.update(true);
        ts.update(false);

        let ev = ts.expected_value();
        assert!(ev > 0.0 && ev < 1.0);
    }
}
