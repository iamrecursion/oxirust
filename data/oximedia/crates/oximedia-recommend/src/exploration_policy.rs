#![allow(dead_code)]
//! Exploration vs exploitation policies for the recommendation system.
//!
//! Pure exploitation always recommends the highest-scoring items, which
//! can create filter bubbles. This module provides exploration strategies
//! that inject a controlled amount of novel or uncertain items into the
//! recommendation list: epsilon-greedy, softmax (Boltzmann), upper
//! confidence bound (UCB), and Thompson sampling approximations.

use std::collections::HashMap;

/// Type of exploration policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PolicyType {
    /// Epsilon-greedy: with probability epsilon, pick a random item.
    EpsilonGreedy,
    /// Softmax/Boltzmann: sample proportional to exponentiated scores.
    Softmax,
    /// Upper Confidence Bound: pick the item with highest score + uncertainty.
    Ucb,
    /// Decaying epsilon-greedy: epsilon decreases over time.
    DecayingEpsilon,
}

impl std::fmt::Display for PolicyType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EpsilonGreedy => write!(f, "EpsilonGreedy"),
            Self::Softmax => write!(f, "Softmax"),
            Self::Ucb => write!(f, "UCB"),
            Self::DecayingEpsilon => write!(f, "DecayingEpsilon"),
        }
    }
}

/// Configuration for an exploration policy.
#[derive(Debug, Clone)]
pub struct PolicyConfig {
    /// Policy type.
    pub policy_type: PolicyType,
    /// Epsilon for epsilon-greedy (0.0-1.0).
    pub epsilon: f64,
    /// Temperature for softmax (higher = more exploration).
    pub temperature: f64,
    /// Exploration coefficient for UCB.
    pub ucb_coefficient: f64,
    /// Decay factor for decaying epsilon (multiplied each round).
    pub decay_factor: f64,
    /// Minimum epsilon for decaying strategy.
    pub min_epsilon: f64,
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self {
            policy_type: PolicyType::EpsilonGreedy,
            epsilon: 0.1,
            temperature: 1.0,
            ucb_coefficient: 2.0,
            decay_factor: 0.995,
            min_epsilon: 0.01,
        }
    }
}

/// An item candidate with its exploitation score and exploration metadata.
#[derive(Debug, Clone)]
pub struct ScoredCandidate {
    /// Item identifier.
    pub item_id: String,
    /// Exploitation score (predicted relevance, 0.0-1.0).
    pub score: f64,
    /// Number of times this item has been shown.
    pub impression_count: u64,
    /// Number of times this item has been clicked.
    pub click_count: u64,
    /// Uncertainty estimate (standard deviation of score estimate).
    pub uncertainty: f64,
}

impl ScoredCandidate {
    /// Create a new scored candidate.
    #[must_use]
    pub fn new(item_id: &str, score: f64) -> Self {
        Self {
            item_id: item_id.to_string(),
            score,
            impression_count: 0,
            click_count: 0,
            uncertainty: 1.0,
        }
    }

    /// Observed click-through rate.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn observed_ctr(&self) -> f64 {
        if self.impression_count == 0 {
            return 0.0;
        }
        self.click_count as f64 / self.impression_count as f64
    }

    /// UCB score: score + c * sqrt(ln(total) / impressions).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn ucb_score(&self, total_impressions: u64, coefficient: f64) -> f64 {
        if self.impression_count == 0 {
            return f64::MAX;
        }
        let exploration =
            coefficient * ((total_impressions as f64).ln() / self.impression_count as f64).sqrt();
        self.score + exploration
    }

    /// Softmax weight: exp(score / temperature).
    #[must_use]
    pub fn softmax_weight(&self, temperature: f64) -> f64 {
        if temperature <= 0.0 {
            return if self.score > 0.0 { f64::MAX } else { 0.0 };
        }
        (self.score / temperature).exp()
    }
}

/// Result of applying an exploration policy.
#[derive(Debug, Clone)]
pub struct ExplorationResult {
    /// Re-ranked item IDs (in final recommended order).
    pub ranked_items: Vec<String>,
    /// Number of items selected via exploration (not pure exploitation).
    pub explored_count: usize,
    /// Number of items selected via exploitation.
    pub exploited_count: usize,
    /// Effective epsilon used.
    pub effective_epsilon: f64,
}

/// Exploration policy engine.
#[derive(Debug)]
pub struct ExplorationPolicy {
    /// Configuration.
    config: PolicyConfig,
    /// Current epsilon (for decaying strategies).
    current_epsilon: f64,
    /// Total rounds applied.
    rounds: u64,
    /// Per-item impression tracking.
    item_impressions: HashMap<String, u64>,
    /// Total impressions across all items.
    total_impressions: u64,
}

impl ExplorationPolicy {
    /// Create a new exploration policy with the given config.
    #[must_use]
    pub fn new(config: PolicyConfig) -> Self {
        let current_epsilon = config.epsilon;
        Self {
            config,
            current_epsilon,
            rounds: 0,
            item_impressions: HashMap::new(),
            total_impressions: 0,
        }
    }

    /// Create an epsilon-greedy policy.
    #[must_use]
    pub fn epsilon_greedy(epsilon: f64) -> Self {
        let mut config = PolicyConfig::default();
        config.policy_type = PolicyType::EpsilonGreedy;
        config.epsilon = epsilon.clamp(0.0, 1.0);
        Self::new(config)
    }

    /// Create a softmax policy.
    #[must_use]
    pub fn softmax(temperature: f64) -> Self {
        let mut config = PolicyConfig::default();
        config.policy_type = PolicyType::Softmax;
        config.temperature = temperature.max(0.01);
        Self::new(config)
    }

    /// Create a UCB policy.
    #[must_use]
    pub fn ucb(coefficient: f64) -> Self {
        let mut config = PolicyConfig::default();
        config.policy_type = PolicyType::Ucb;
        config.ucb_coefficient = coefficient.max(0.0);
        Self::new(config)
    }

    /// Get the current effective epsilon.
    #[must_use]
    pub fn effective_epsilon(&self) -> f64 {
        self.current_epsilon
    }

    /// Get the number of rounds applied.
    #[must_use]
    pub fn rounds(&self) -> u64 {
        self.rounds
    }

    /// Apply the exploration policy to rank candidates.
    ///
    /// Uses a deterministic approximation: for epsilon-greedy, the first
    /// `(1-epsilon)*limit` items come from exploitation, the rest from
    /// exploration (least-seen items). For UCB, items are sorted by UCB score.
    /// For softmax, items are sorted by softmax weight.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn apply(&mut self, candidates: &[ScoredCandidate], limit: usize) -> ExplorationResult {
        if candidates.is_empty() {
            return ExplorationResult {
                ranked_items: Vec::new(),
                explored_count: 0,
                exploited_count: 0,
                effective_epsilon: self.current_epsilon,
            };
        }

        self.rounds += 1;

        let result = match self.config.policy_type {
            PolicyType::EpsilonGreedy | PolicyType::DecayingEpsilon => {
                self.apply_epsilon_greedy(candidates, limit)
            }
            PolicyType::Softmax => self.apply_softmax(candidates, limit),
            PolicyType::Ucb => self.apply_ucb(candidates, limit),
        };

        // Decay epsilon if using decaying strategy.
        if self.config.policy_type == PolicyType::DecayingEpsilon {
            self.current_epsilon =
                (self.current_epsilon * self.config.decay_factor).max(self.config.min_epsilon);
        }

        result
    }

    /// Epsilon-greedy selection.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn apply_epsilon_greedy(
        &self,
        candidates: &[ScoredCandidate],
        limit: usize,
    ) -> ExplorationResult {
        let actual_limit = limit.min(candidates.len());
        let exploit_count = ((1.0 - self.current_epsilon) * actual_limit as f64).round() as usize;
        let explore_count = actual_limit.saturating_sub(exploit_count);

        // Sort by score descending for exploitation.
        let mut sorted: Vec<&ScoredCandidate> = candidates.iter().collect();
        sorted.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut ranked_items: Vec<String> = Vec::with_capacity(actual_limit);

        // Take top exploit_count by score.
        for c in sorted.iter().take(exploit_count) {
            ranked_items.push(c.item_id.clone());
        }

        // For exploration, pick least-seen items from the remainder.
        let mut remainder: Vec<&ScoredCandidate> =
            sorted.iter().skip(exploit_count).copied().collect();
        remainder.sort_by_key(|c| c.impression_count);
        for c in remainder.iter().take(explore_count) {
            ranked_items.push(c.item_id.clone());
        }

        ExplorationResult {
            ranked_items,
            explored_count: explore_count,
            exploited_count: exploit_count,
            effective_epsilon: self.current_epsilon,
        }
    }

    /// Softmax (Boltzmann) selection.
    fn apply_softmax(&self, candidates: &[ScoredCandidate], limit: usize) -> ExplorationResult {
        let actual_limit = limit.min(candidates.len());
        let mut weighted: Vec<(&ScoredCandidate, f64)> = candidates
            .iter()
            .map(|c| (c, c.softmax_weight(self.config.temperature)))
            .collect();
        weighted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let ranked_items: Vec<String> = weighted
            .iter()
            .take(actual_limit)
            .map(|(c, _)| c.item_id.clone())
            .collect();

        // Count how many items differ from pure exploitation order.
        let mut by_score: Vec<&str> = candidates.iter().map(|c| c.item_id.as_str()).collect();
        by_score.sort_by(|a, b| {
            let sa = candidates
                .iter()
                .find(|c| c.item_id == *a)
                .map_or(0.0, |c| c.score);
            let sb = candidates
                .iter()
                .find(|c| c.item_id == *b)
                .map_or(0.0, |c| c.score);
            sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
        });

        let explored = ranked_items
            .iter()
            .enumerate()
            .filter(|(i, id)| by_score.get(*i).map_or(true, |s| *s != id.as_str()))
            .count();

        ExplorationResult {
            ranked_items,
            explored_count: explored,
            exploited_count: actual_limit.saturating_sub(explored),
            effective_epsilon: self.current_epsilon,
        }
    }

    /// UCB selection.
    fn apply_ucb(&self, candidates: &[ScoredCandidate], limit: usize) -> ExplorationResult {
        let actual_limit = limit.min(candidates.len());
        let mut scored: Vec<(&ScoredCandidate, f64)> = candidates
            .iter()
            .map(|c| {
                (
                    c,
                    c.ucb_score(self.total_impressions.max(1), self.config.ucb_coefficient),
                )
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let ranked_items: Vec<String> = scored
            .iter()
            .take(actual_limit)
            .map(|(c, _)| c.item_id.clone())
            .collect();

        // Items with 0 impressions are definitely exploration.
        let explored = ranked_items
            .iter()
            .filter(|id| {
                candidates
                    .iter()
                    .find(|c| c.item_id == **id)
                    .is_some_and(|c| c.impression_count == 0)
            })
            .count();

        ExplorationResult {
            ranked_items,
            explored_count: explored,
            exploited_count: actual_limit.saturating_sub(explored),
            effective_epsilon: self.current_epsilon,
        }
    }

    /// Record that an item was shown (for UCB tracking).
    pub fn record_impression(&mut self, item_id: &str) {
        *self
            .item_impressions
            .entry(item_id.to_string())
            .or_insert(0) += 1;
        self.total_impressions += 1;
    }

    /// Get the policy type.
    #[must_use]
    pub fn policy_type(&self) -> PolicyType {
        self.config.policy_type
    }
}

impl Default for ExplorationPolicy {
    fn default() -> Self {
        Self::new(PolicyConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candidates(n: usize) -> Vec<ScoredCandidate> {
        (0..n)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let score = 1.0 - (i as f64 * 0.1);
                ScoredCandidate::new(&format!("item_{i}"), score.max(0.05))
            })
            .collect()
    }

    #[test]
    fn test_policy_type_display() {
        assert_eq!(PolicyType::EpsilonGreedy.to_string(), "EpsilonGreedy");
        assert_eq!(PolicyType::Ucb.to_string(), "UCB");
        assert_eq!(PolicyType::Softmax.to_string(), "Softmax");
        assert_eq!(PolicyType::DecayingEpsilon.to_string(), "DecayingEpsilon");
    }

    #[test]
    fn test_default_config() {
        let cfg = PolicyConfig::default();
        assert!((cfg.epsilon - 0.1).abs() < f64::EPSILON);
        assert!((cfg.temperature - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_scored_candidate_ctr() {
        let mut c = ScoredCandidate::new("x", 0.8);
        assert_eq!(c.observed_ctr(), 0.0);
        c.impression_count = 10;
        c.click_count = 3;
        assert!((c.observed_ctr() - 0.3).abs() < f64::EPSILON);
    }

    #[test]
    fn test_scored_candidate_ucb_no_impressions() {
        let c = ScoredCandidate::new("y", 0.5);
        assert_eq!(c.ucb_score(100, 2.0), f64::MAX);
    }

    #[test]
    fn test_scored_candidate_ucb_with_impressions() {
        let mut c = ScoredCandidate::new("z", 0.5);
        c.impression_count = 10;
        let ucb = c.ucb_score(100, 2.0);
        assert!(ucb > 0.5);
        assert!(ucb < 3.0);
    }

    #[test]
    fn test_softmax_weight_positive_temperature() {
        let c = ScoredCandidate::new("a", 0.8);
        let w = c.softmax_weight(1.0);
        assert!(w > 0.0);
    }

    #[test]
    fn test_epsilon_greedy_basic() {
        let mut policy = ExplorationPolicy::epsilon_greedy(0.2);
        let candidates = make_candidates(10);
        let result = policy.apply(&candidates, 5);
        assert_eq!(result.ranked_items.len(), 5);
        assert_eq!(result.explored_count + result.exploited_count, 5);
    }

    #[test]
    fn test_epsilon_greedy_zero_epsilon() {
        let mut policy = ExplorationPolicy::epsilon_greedy(0.0);
        let candidates = make_candidates(5);
        let result = policy.apply(&candidates, 3);
        // With epsilon=0, all should be exploited (top by score).
        assert_eq!(result.exploited_count, 3);
        assert_eq!(result.explored_count, 0);
        // First item should be highest score.
        assert_eq!(result.ranked_items[0], "item_0");
    }

    #[test]
    fn test_softmax_policy() {
        let mut policy = ExplorationPolicy::softmax(0.5);
        let candidates = make_candidates(8);
        let result = policy.apply(&candidates, 4);
        assert_eq!(result.ranked_items.len(), 4);
    }

    #[test]
    fn test_ucb_policy() {
        let mut policy = ExplorationPolicy::ucb(2.0);
        let candidates = make_candidates(6);
        let result = policy.apply(&candidates, 3);
        assert_eq!(result.ranked_items.len(), 3);
        // All candidates have 0 impressions, so all are explored.
        assert_eq!(result.explored_count, 3);
    }

    #[test]
    fn test_empty_candidates() {
        let mut policy = ExplorationPolicy::epsilon_greedy(0.1);
        let result = policy.apply(&[], 5);
        assert!(result.ranked_items.is_empty());
        assert_eq!(result.explored_count, 0);
    }

    #[test]
    fn test_decaying_epsilon() {
        let mut config = PolicyConfig::default();
        config.policy_type = PolicyType::DecayingEpsilon;
        config.epsilon = 0.5;
        config.decay_factor = 0.5;
        config.min_epsilon = 0.01;

        let mut policy = ExplorationPolicy::new(config);
        assert!((policy.effective_epsilon() - 0.5).abs() < f64::EPSILON);

        let candidates = make_candidates(5);
        policy.apply(&candidates, 3);
        assert!((policy.effective_epsilon() - 0.25).abs() < f64::EPSILON);

        policy.apply(&candidates, 3);
        assert!((policy.effective_epsilon() - 0.125).abs() < f64::EPSILON);
    }

    #[test]
    fn test_record_impression() {
        let mut policy = ExplorationPolicy::ucb(2.0);
        policy.record_impression("item_0");
        policy.record_impression("item_0");
        policy.record_impression("item_1");
        assert_eq!(policy.total_impressions, 3);
    }

    #[test]
    fn test_rounds_counter() {
        let mut policy = ExplorationPolicy::epsilon_greedy(0.1);
        assert_eq!(policy.rounds(), 0);
        let candidates = make_candidates(3);
        policy.apply(&candidates, 2);
        assert_eq!(policy.rounds(), 1);
        policy.apply(&candidates, 2);
        assert_eq!(policy.rounds(), 2);
    }
}
