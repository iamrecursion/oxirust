#![allow(dead_code)]
//! Time-decay models for aging user preferences and interactions.
//!
//! Over time, user interests shift. This module provides decay functions
//! that reduce the weight of older interactions, allowing the recommendation
//! engine to prioritize recent behavior. Supports exponential decay,
//! linear decay, step-function decay, and custom half-life configurations.

use std::collections::HashMap;

/// Type of decay function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecayType {
    /// Exponential decay: weight = e^(-lambda * age).
    Exponential,
    /// Linear decay: weight = max(0, 1 - age/lifetime).
    Linear,
    /// Step function: full weight until cutoff, then zero.
    Step,
    /// Logarithmic decay: weight = 1 / (1 + log(1 + age)).
    Logarithmic,
}

impl std::fmt::Display for DecayType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Exponential => write!(f, "Exponential"),
            Self::Linear => write!(f, "Linear"),
            Self::Step => write!(f, "Step"),
            Self::Logarithmic => write!(f, "Logarithmic"),
        }
    }
}

/// Configuration for a decay model.
#[derive(Debug, Clone)]
pub struct DecayConfig {
    /// Type of decay function.
    pub decay_type: DecayType,
    /// Half-life in seconds (for exponential: time until weight = 0.5).
    pub half_life_secs: f64,
    /// Minimum weight threshold (below this, interactions are discarded).
    pub min_weight: f64,
    /// Maximum age in seconds (for step/linear: hard cutoff).
    pub max_age_secs: f64,
}

impl Default for DecayConfig {
    fn default() -> Self {
        Self {
            decay_type: DecayType::Exponential,
            half_life_secs: 7.0 * 24.0 * 3600.0, // 1 week
            min_weight: 0.01,
            max_age_secs: 90.0 * 24.0 * 3600.0, // 90 days
        }
    }
}

/// Computes the decay weight for a given age (in seconds).
#[allow(clippy::cast_precision_loss)]
fn compute_weight(config: &DecayConfig, age_secs: f64) -> f64 {
    if age_secs < 0.0 {
        return 1.0;
    }
    if age_secs > config.max_age_secs {
        return 0.0;
    }

    let w = match config.decay_type {
        DecayType::Exponential => {
            if config.half_life_secs <= 0.0 {
                return 0.0;
            }
            let lambda = (2.0_f64).ln() / config.half_life_secs;
            (-lambda * age_secs).exp()
        }
        DecayType::Linear => {
            if config.max_age_secs <= 0.0 {
                return 0.0;
            }
            (1.0 - age_secs / config.max_age_secs).max(0.0)
        }
        DecayType::Step => {
            if age_secs <= config.max_age_secs {
                1.0
            } else {
                0.0
            }
        }
        DecayType::Logarithmic => 1.0 / (1.0 + (1.0 + age_secs).ln()),
    };

    if w < config.min_weight {
        0.0
    } else {
        w
    }
}

/// A single interaction record with timestamp.
#[derive(Debug, Clone)]
pub struct Interaction {
    /// Item identifier.
    pub item_id: String,
    /// Interaction timestamp (seconds since epoch).
    pub timestamp_secs: f64,
    /// Raw interaction score (e.g. rating, click weight).
    pub raw_score: f64,
    /// Category of the item.
    pub category: String,
}

/// A weighted interaction after decay has been applied.
#[derive(Debug, Clone)]
pub struct WeightedInteraction {
    /// Item identifier.
    pub item_id: String,
    /// Decayed weight (0.0-1.0).
    pub weight: f64,
    /// Weighted score (`raw_score` * weight).
    pub weighted_score: f64,
    /// Category of the item.
    pub category: String,
    /// Age of the interaction in seconds.
    pub age_secs: f64,
}

/// The decay model engine.
///
/// Applies time-decay to user interaction histories, producing weighted
/// scores that reflect the recency of each interaction.
#[derive(Debug)]
pub struct DecayModel {
    /// Configuration.
    config: DecayConfig,
    /// Per-user interaction history.
    user_interactions: HashMap<String, Vec<Interaction>>,
    /// Total interactions processed.
    total_processed: u64,
    /// Total interactions discarded (below `min_weight`).
    total_discarded: u64,
}

impl DecayModel {
    /// Create a new decay model.
    #[must_use]
    pub fn new(config: DecayConfig) -> Self {
        Self {
            config,
            user_interactions: HashMap::new(),
            total_processed: 0,
            total_discarded: 0,
        }
    }

    /// Record an interaction for a user.
    pub fn record_interaction(&mut self, user_id: impl Into<String>, interaction: Interaction) {
        self.user_interactions
            .entry(user_id.into())
            .or_default()
            .push(interaction);
    }

    /// Apply decay to all interactions for a user at the given current time.
    pub fn apply_decay(&mut self, user_id: &str, now_secs: f64) -> Vec<WeightedInteraction> {
        let interactions = match self.user_interactions.get(user_id) {
            Some(v) => v,
            None => return Vec::new(),
        };

        let mut results = Vec::new();
        for interaction in interactions {
            self.total_processed += 1;
            let age = now_secs - interaction.timestamp_secs;
            let weight = compute_weight(&self.config, age);
            if weight <= 0.0 {
                self.total_discarded += 1;
                continue;
            }
            results.push(WeightedInteraction {
                item_id: interaction.item_id.clone(),
                weight,
                weighted_score: interaction.raw_score * weight,
                category: interaction.category.clone(),
                age_secs: age,
            });
        }

        // Sort by weighted score descending
        results.sort_by(|a, b| {
            b.weighted_score
                .partial_cmp(&a.weighted_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }

    /// Compute aggregate category preferences with decay for a user.
    pub fn category_preferences(&mut self, user_id: &str, now_secs: f64) -> Vec<(String, f64)> {
        let weighted = self.apply_decay(user_id, now_secs);
        let mut category_scores: HashMap<String, f64> = HashMap::new();

        for w in &weighted {
            *category_scores.entry(w.category.clone()).or_default() += w.weighted_score;
        }

        let mut sorted: Vec<(String, f64)> = category_scores.into_iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted
    }

    /// Prune interactions older than `max_age_secs` for a user.
    pub fn prune_old(&mut self, user_id: &str, now_secs: f64) -> usize {
        let max_age = self.config.max_age_secs;
        let interactions = match self.user_interactions.get_mut(user_id) {
            Some(v) => v,
            None => return 0,
        };
        let before = interactions.len();
        interactions.retain(|i| (now_secs - i.timestamp_secs) <= max_age);
        before - interactions.len()
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &DecayConfig {
        &self.config
    }

    /// Get total interactions processed.
    #[must_use]
    pub fn total_processed(&self) -> u64 {
        self.total_processed
    }

    /// Get total interactions discarded.
    #[must_use]
    pub fn total_discarded(&self) -> u64 {
        self.total_discarded
    }

    /// Number of tracked users.
    #[must_use]
    pub fn user_count(&self) -> usize {
        self.user_interactions.len()
    }

    /// Number of interactions for a specific user.
    pub fn interaction_count(&self, user_id: &str) -> usize {
        self.user_interactions.get(user_id).map_or(0, Vec::len)
    }
}

impl Default for DecayModel {
    fn default() -> Self {
        Self::new(DecayConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn secs(days: f64) -> f64 {
        days * 24.0 * 3600.0
    }

    fn make_interaction(item_id: &str, ts: f64, score: f64, category: &str) -> Interaction {
        Interaction {
            item_id: item_id.to_string(),
            timestamp_secs: ts,
            raw_score: score,
            category: category.to_string(),
        }
    }

    #[test]
    fn test_decay_type_display() {
        assert_eq!(DecayType::Exponential.to_string(), "Exponential");
        assert_eq!(DecayType::Linear.to_string(), "Linear");
        assert_eq!(DecayType::Step.to_string(), "Step");
        assert_eq!(DecayType::Logarithmic.to_string(), "Logarithmic");
    }

    #[test]
    fn test_exponential_decay_half_life() {
        let config = DecayConfig {
            decay_type: DecayType::Exponential,
            half_life_secs: 100.0,
            min_weight: 0.001,
            max_age_secs: 10000.0,
        };
        let w_at_half = compute_weight(&config, 100.0);
        assert!((w_at_half - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_exponential_decay_zero_age() {
        let config = DecayConfig::default();
        let w = compute_weight(&config, 0.0);
        assert!((w - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_linear_decay() {
        let config = DecayConfig {
            decay_type: DecayType::Linear,
            half_life_secs: 100.0,
            min_weight: 0.0,
            max_age_secs: 100.0,
        };
        assert!((compute_weight(&config, 0.0) - 1.0).abs() < f64::EPSILON);
        assert!((compute_weight(&config, 50.0) - 0.5).abs() < f64::EPSILON);
        assert!((compute_weight(&config, 100.0) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_step_decay() {
        let config = DecayConfig {
            decay_type: DecayType::Step,
            half_life_secs: 100.0,
            min_weight: 0.0,
            max_age_secs: 50.0,
        };
        assert!((compute_weight(&config, 30.0) - 1.0).abs() < f64::EPSILON);
        assert!((compute_weight(&config, 60.0) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_logarithmic_decay() {
        let config = DecayConfig {
            decay_type: DecayType::Logarithmic,
            half_life_secs: 100.0,
            min_weight: 0.0,
            max_age_secs: 100000.0,
        };
        let w0 = compute_weight(&config, 0.0);
        let w100 = compute_weight(&config, 100.0);
        // At age 0, ln(1+0)=0, weight = 1/(1+0) = 1.0
        assert!((w0 - 1.0).abs() < 0.01);
        // Should decrease with age
        assert!(w100 < w0);
    }

    #[test]
    fn test_decay_beyond_max_age() {
        let config = DecayConfig {
            decay_type: DecayType::Exponential,
            half_life_secs: 100.0,
            min_weight: 0.01,
            max_age_secs: 200.0,
        };
        assert!((compute_weight(&config, 250.0) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_decay_model_apply() {
        let config = DecayConfig {
            decay_type: DecayType::Exponential,
            half_life_secs: secs(7.0),
            min_weight: 0.01,
            max_age_secs: secs(90.0),
        };
        let mut model = DecayModel::new(config);
        let now = secs(100.0);
        model.record_interaction(
            "user1",
            make_interaction("item1", now - secs(1.0), 5.0, "action"),
        );
        model.record_interaction(
            "user1",
            make_interaction("item2", now - secs(30.0), 4.0, "comedy"),
        );

        let results = model.apply_decay("user1", now);
        assert_eq!(results.len(), 2);
        // Recent item should have higher weighted score
        assert!(results[0].weighted_score > results[1].weighted_score);
    }

    #[test]
    fn test_decay_model_discards_old() {
        let config = DecayConfig {
            decay_type: DecayType::Step,
            half_life_secs: 100.0,
            min_weight: 0.0,
            max_age_secs: 100.0,
        };
        let mut model = DecayModel::new(config);
        model.record_interaction("u1", make_interaction("i1", 0.0, 5.0, "action"));

        let results = model.apply_decay("u1", 200.0);
        assert!(results.is_empty());
        assert_eq!(model.total_discarded(), 1);
    }

    #[test]
    fn test_category_preferences() {
        let config = DecayConfig {
            decay_type: DecayType::Step,
            half_life_secs: 100.0,
            min_weight: 0.0,
            max_age_secs: 1000.0,
        };
        let mut model = DecayModel::new(config);
        model.record_interaction("u1", make_interaction("i1", 900.0, 5.0, "action"));
        model.record_interaction("u1", make_interaction("i2", 900.0, 3.0, "action"));
        model.record_interaction("u1", make_interaction("i3", 900.0, 4.0, "comedy"));

        let prefs = model.category_preferences("u1", 950.0);
        assert_eq!(prefs[0].0, "action");
        assert!(prefs[0].1 > prefs[1].1);
    }

    #[test]
    fn test_prune_old_interactions() {
        let config = DecayConfig {
            max_age_secs: 100.0,
            ..Default::default()
        };
        let mut model = DecayModel::new(config);
        model.record_interaction("u1", make_interaction("i1", 0.0, 5.0, "a"));
        model.record_interaction("u1", make_interaction("i2", 50.0, 3.0, "b"));
        model.record_interaction("u1", make_interaction("i3", 150.0, 4.0, "c"));

        let pruned = model.prune_old("u1", 200.0);
        assert_eq!(pruned, 2); // i1 and i2 are >100 secs old
        assert_eq!(model.interaction_count("u1"), 1);
    }

    #[test]
    fn test_decay_model_user_count() {
        let mut model = DecayModel::default();
        model.record_interaction("u1", make_interaction("i1", 0.0, 1.0, "a"));
        model.record_interaction("u2", make_interaction("i2", 0.0, 1.0, "b"));
        assert_eq!(model.user_count(), 2);
    }

    #[test]
    fn test_no_interactions_returns_empty() {
        let mut model = DecayModel::default();
        let results = model.apply_decay("nonexistent", 1000.0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_negative_age_returns_full_weight() {
        let config = DecayConfig::default();
        let w = compute_weight(&config, -10.0);
        assert!((w - 1.0).abs() < f64::EPSILON);
    }
}
