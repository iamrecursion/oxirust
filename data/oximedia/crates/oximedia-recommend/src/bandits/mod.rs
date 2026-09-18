//! Multi-armed bandit algorithms for exploration/exploitation in recommendations.

use std::collections::HashMap;

/// A single bandit arm tracking pull count and accumulated rewards.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct BanditArm {
    /// Unique identifier for this arm
    pub id: String,
    /// Number of times this arm has been pulled
    pub pulls: u64,
    /// Cumulative reward received from this arm
    pub rewards: f64,
    /// Timestamp (in seconds) of last update
    pub last_updated: u64,
}

impl BanditArm {
    /// Create a new bandit arm with the given ID.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            pulls: 0,
            rewards: 0.0,
            last_updated: 0,
        }
    }

    /// Return the mean reward (0.0 if never pulled).
    #[must_use]
    pub fn mean_reward(&self) -> f64 {
        if self.pulls == 0 {
            0.0
        } else {
            self.rewards / self.pulls as f64
        }
    }
}

// ---------------------------------------------------------------------------
// Epsilon-greedy bandit
// ---------------------------------------------------------------------------

/// Epsilon-greedy multi-armed bandit.
///
/// With probability `epsilon` a random arm is selected (exploration); otherwise
/// the arm with the highest mean reward is selected (exploitation).
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct EpsilonGreedy {
    /// Exploration rate (0.0 = always greedy, 1.0 = always random)
    pub epsilon: f32,
    /// Arms managed by this bandit
    pub arms: Vec<BanditArm>,
}

impl EpsilonGreedy {
    /// Create a new epsilon-greedy bandit with the given arms.
    #[must_use]
    pub fn new(epsilon: f32, arms: Vec<BanditArm>) -> Self {
        Self {
            epsilon: epsilon.clamp(0.0, 1.0),
            arms,
        }
    }

    /// Select an arm index.
    ///
    /// A simple LCG is used to avoid depending on an external RNG crate.
    /// `seed` is consumed internally; pass a different value each call for
    /// meaningful exploration.
    #[must_use]
    pub fn select(&self, seed: u64) -> usize {
        if self.arms.is_empty() {
            return 0;
        }
        // Explore with probability epsilon
        let rand_val = lcg_f64(seed);
        if rand_val < f64::from(self.epsilon) {
            // Random arm
            let rand_idx = lcg_u64(seed.wrapping_add(1)) % self.arms.len() as u64;
            rand_idx as usize
        } else {
            self.best_arm()
        }
    }

    /// Update the reward for a given arm index.
    pub fn update(&mut self, arm_idx: usize, reward: f64) {
        if let Some(arm) = self.arms.get_mut(arm_idx) {
            arm.pulls += 1;
            arm.rewards += reward;
        }
    }

    /// Return the index of the arm with the highest mean reward.
    #[must_use]
    pub fn best_arm(&self) -> usize {
        self.arms
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.mean_reward()
                    .partial_cmp(&b.mean_reward())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map_or(0, |(idx, _)| idx)
    }
}

// ---------------------------------------------------------------------------
// UCB1 bandit
// ---------------------------------------------------------------------------

/// Upper-Confidence-Bound 1 (UCB1) bandit.
///
/// The UCB1 score for arm `i` is:
/// ```text
/// score_i = mean_i + sqrt(2 * ln(N) / n_i)
/// ```
/// where `N` is the total number of pulls and `n_i` is the number of pulls of
/// arm `i`.  Unpulled arms are always selected first.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Ucb1Bandit {
    /// Arms managed by this bandit
    pub arms: Vec<BanditArm>,
    /// Total number of pulls across all arms
    pub total_pulls: u64,
}

impl Ucb1Bandit {
    /// Create a new UCB1 bandit with the given arms.
    #[must_use]
    pub fn new(arms: Vec<BanditArm>) -> Self {
        Self {
            arms,
            total_pulls: 0,
        }
    }

    /// Compute the UCB1 score for arm `i`.
    #[must_use]
    fn ucb1_score(&self, arm: &BanditArm) -> f64 {
        if arm.pulls == 0 {
            return f64::INFINITY;
        }
        let exploration = ((2.0 * (self.total_pulls as f64).ln()) / arm.pulls as f64).sqrt();
        arm.mean_reward() + exploration
    }

    /// Select the arm with the highest UCB1 score.
    #[must_use]
    pub fn select(&self) -> usize {
        self.arms
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                self.ucb1_score(a)
                    .partial_cmp(&self.ucb1_score(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map_or(0, |(idx, _)| idx)
    }

    /// Update the reward for a given arm index.
    pub fn update(&mut self, arm_idx: usize, reward: f64) {
        if let Some(arm) = self.arms.get_mut(arm_idx) {
            arm.pulls += 1;
            arm.rewards += reward;
            self.total_pulls += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Thompson Sampling
// ---------------------------------------------------------------------------

/// Thompson sampling bandit using a Beta distribution approximation.
///
/// Each arm maintains `alpha` (success count + 1) and `beta` (failure count +
/// 1) parameters.  At selection time a sample is drawn from Beta(α, β) for
/// each arm and the arm with the highest sample is chosen.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ThompsonSampling {
    /// Alpha parameters (successes + 1) for each arm
    pub alpha: Vec<f64>,
    /// Beta parameters (failures + 1) for each arm
    pub beta: Vec<f64>,
}

impl ThompsonSampling {
    /// Create a new Thompson sampling bandit with `n` arms.
    ///
    /// All arms start with uniform Beta(1, 1).
    #[must_use]
    pub fn new(n: usize) -> Self {
        Self {
            alpha: vec![1.0; n],
            beta: vec![1.0; n],
        }
    }

    /// Sample a value from Beta(alpha, beta) using a simple LCG-based method.
    ///
    /// This uses the Johnk method which approximates Beta sampling without
    /// requiring an external statistics library.
    #[must_use]
    pub fn sample_beta(alpha: f64, beta: f64, seed: u64) -> f64 {
        // Use the relation: Beta(a, b) ≈ Gamma(a) / (Gamma(a) + Gamma(b))
        // Approximated via Johnk's method with uniform samples from LCG.
        let mut s = seed;
        let gamma_a = sample_gamma(alpha, &mut s);
        let gamma_b = sample_gamma(beta, &mut s);
        if gamma_a + gamma_b == 0.0 {
            return 0.5;
        }
        gamma_a / (gamma_a + gamma_b)
    }

    /// Select the arm with the highest Beta sample.
    #[must_use]
    pub fn select(&self, seed: u64) -> usize {
        self.alpha
            .iter()
            .zip(self.beta.iter())
            .enumerate()
            .map(|(i, (&a, &b))| {
                let s = Self::sample_beta(
                    a,
                    b,
                    seed.wrapping_add((i as u64).wrapping_mul(6364136223846793005)),
                );
                (i, s)
            })
            .max_by(|(_, s1), (_, s2)| s1.partial_cmp(s2).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i)
    }

    /// Record a success for the given arm (increments alpha).
    pub fn update_success(&mut self, arm: usize) {
        if arm < self.alpha.len() {
            self.alpha[arm] += 1.0;
        }
    }

    /// Record a failure for the given arm (increments beta).
    pub fn update_failure(&mut self, arm: usize) {
        if arm < self.beta.len() {
            self.beta[arm] += 1.0;
        }
    }
}

// ---------------------------------------------------------------------------
// Content bandit
// ---------------------------------------------------------------------------

/// A bandit that maps content IDs to arms for content exploration.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ContentBandit {
    /// Inner bandit algorithm
    inner: EpsilonGreedy,
    /// Map from content ID to arm index
    content_to_arm: HashMap<u64, usize>,
    /// Map from arm index to content ID
    arm_to_content: Vec<u64>,
}

impl ContentBandit {
    /// Create a new content bandit with the given content IDs and epsilon.
    #[must_use]
    pub fn new(content_ids: Vec<u64>, epsilon: f32) -> Self {
        let arms: Vec<BanditArm> = content_ids
            .iter()
            .map(|id| BanditArm::new(id.to_string()))
            .collect();
        let content_to_arm: HashMap<u64, usize> = content_ids
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, i))
            .collect();
        Self {
            inner: EpsilonGreedy::new(epsilon, arms),
            content_to_arm,
            arm_to_content: content_ids,
        }
    }

    /// Select a content ID to show.
    #[must_use]
    pub fn select_content(&self, seed: u64) -> Option<u64> {
        let arm_idx = self.inner.select(seed);
        self.arm_to_content.get(arm_idx).copied()
    }

    /// Record a reward for a given content ID.
    pub fn update(&mut self, content_id: u64, reward: f64) {
        if let Some(&arm_idx) = self.content_to_arm.get(&content_id) {
            self.inner.update(arm_idx, reward);
        }
    }

    /// Return the content ID with the highest mean reward.
    #[must_use]
    pub fn best_content(&self) -> Option<u64> {
        let best = self.inner.best_arm();
        self.arm_to_content.get(best).copied()
    }

    /// Return the number of arms.
    #[must_use]
    pub fn arm_count(&self) -> usize {
        self.inner.arms.len()
    }
}

// ---------------------------------------------------------------------------
// Internal helpers – Linear Congruential Generator
// ---------------------------------------------------------------------------

/// Advance a 64-bit LCG and return the next state.
#[inline]
fn lcg_next(state: u64) -> u64 {
    state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
}

/// Map an LCG state to [0.0, 1.0).
#[inline]
fn lcg_f64(seed: u64) -> f64 {
    let s = lcg_next(seed);
    (s >> 11) as f64 / (1u64 << 53) as f64
}

/// Map an LCG state to a u64 for indexing.
#[inline]
fn lcg_u64(seed: u64) -> u64 {
    lcg_next(seed)
}

/// Very simple Gamma(shape, 1) sampler via Marsaglia–Tsang (truncated).
///
/// This gives an approximate sample sufficient for Thompson sampling
/// without external dependencies.
fn sample_gamma(shape: f64, state: &mut u64) -> f64 {
    // For shape >= 1 use Marsaglia–Tsang; for shape < 1 use Ahrens–Dieter
    if shape < 1.0 {
        let boost = sample_gamma(1.0 + shape, state);
        let u = {
            *state = lcg_next(*state);
            (*state >> 11) as f64 / (1u64 << 53) as f64
        };
        return boost * u.powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        // Draw a normal variate via Box-Muller from LCG uniforms
        *state = lcg_next(*state);
        let u1 = (*state >> 11) as f64 / (1u64 << 53) as f64;
        *state = lcg_next(*state);
        let u2 = (*state >> 11) as f64 / (1u64 << 53) as f64;
        let x = (-2.0 * (u1 + 1e-10).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        let v = 1.0 + c * x;
        if v <= 0.0 {
            continue;
        }
        let v3 = v * v * v;
        *state = lcg_next(*state);
        let u = (*state >> 11) as f64 / (1u64 << 53) as f64;
        if u < 1.0 - 0.0331 * (x * x) * (x * x) {
            return d * v3;
        }
        if u.ln() < 0.5 * x * x + d * (1.0 - v3 + v3.ln()) {
            return d * v3;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_arms(n: usize) -> Vec<BanditArm> {
        (0..n).map(|i| BanditArm::new(format!("arm-{i}"))).collect()
    }

    #[test]
    fn test_bandit_arm_mean_reward_no_pulls() {
        let arm = BanditArm::new("test");
        assert_eq!(arm.mean_reward(), 0.0);
    }

    #[test]
    fn test_bandit_arm_mean_reward_after_update() {
        let mut arm = BanditArm::new("test");
        arm.pulls = 2;
        arm.rewards = 1.6;
        assert!((arm.mean_reward() - 0.8).abs() < 1e-9);
    }

    #[test]
    fn test_epsilon_greedy_best_arm() {
        let mut eg = EpsilonGreedy::new(0.0, make_arms(3));
        eg.update(1, 0.9);
        eg.update(1, 0.9);
        assert_eq!(eg.best_arm(), 1);
    }

    #[test]
    fn test_epsilon_greedy_greedy_select() {
        // With epsilon=0 we should always select the best arm
        let mut eg = EpsilonGreedy::new(0.0, make_arms(3));
        eg.update(2, 1.0);
        eg.update(2, 1.0);
        let selected = eg.select(42);
        assert_eq!(selected, 2);
    }

    #[test]
    fn test_epsilon_greedy_update() {
        let mut eg = EpsilonGreedy::new(0.1, make_arms(3));
        eg.update(0, 0.5);
        assert_eq!(eg.arms[0].pulls, 1);
        assert!((eg.arms[0].rewards - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_epsilon_greedy_explore() {
        // With epsilon=1 we always explore (random), so result should be in range
        let eg = EpsilonGreedy::new(1.0, make_arms(5));
        let selected = eg.select(12345);
        assert!(selected < 5);
    }

    #[test]
    fn test_ucb1_select_unpulled_first() {
        let mut bandit = Ucb1Bandit::new(make_arms(3));
        // Pull arm 0 twice so arm 1 and 2 should still have infinity score
        bandit.update(0, 0.5);
        bandit.update(0, 0.5);
        let selected = bandit.select();
        // Arms 1 and 2 are unpulled → score = infinity, either one may be selected
        assert!(selected == 1 || selected == 2);
    }

    #[test]
    fn test_ucb1_update_counts() {
        let mut bandit = Ucb1Bandit::new(make_arms(2));
        bandit.update(0, 1.0);
        bandit.update(1, 0.0);
        assert_eq!(bandit.total_pulls, 2);
        assert_eq!(bandit.arms[0].pulls, 1);
        assert_eq!(bandit.arms[1].pulls, 1);
    }

    #[test]
    fn test_ucb1_selects_higher_reward() {
        let mut bandit = Ucb1Bandit::new(make_arms(2));
        // Pull each arm many times so exploration term is small
        for _ in 0..50 {
            bandit.update(0, 0.9);
            bandit.update(1, 0.1);
        }
        let selected = bandit.select();
        assert_eq!(selected, 0);
    }

    #[test]
    fn test_thompson_sampling_select_range() {
        let ts = ThompsonSampling::new(5);
        let selected = ts.select(9999);
        assert!(selected < 5);
    }

    #[test]
    fn test_thompson_sampling_update_success() {
        let mut ts = ThompsonSampling::new(3);
        ts.update_success(0);
        assert!((ts.alpha[0] - 2.0).abs() < 1e-9);
        assert!((ts.beta[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_thompson_sampling_update_failure() {
        let mut ts = ThompsonSampling::new(3);
        ts.update_failure(2);
        assert!((ts.beta[2] - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_sample_beta_range() {
        for seed in 0..20u64 {
            let v = ThompsonSampling::sample_beta(2.0, 5.0, seed * 1000);
            assert!((0.0..=1.0).contains(&v), "value {v} out of range");
        }
    }

    #[test]
    fn test_content_bandit_select() {
        let cb = ContentBandit::new(vec![10, 20, 30], 0.0);
        let content = cb.select_content(42);
        assert!(content.is_some());
    }

    #[test]
    fn test_content_bandit_update_and_best() {
        let mut cb = ContentBandit::new(vec![10, 20, 30], 0.0);
        cb.update(20, 1.0);
        cb.update(20, 1.0);
        assert_eq!(cb.best_content(), Some(20));
    }

    #[test]
    fn test_content_bandit_arm_count() {
        let cb = ContentBandit::new(vec![1, 2, 3, 4], 0.1);
        assert_eq!(cb.arm_count(), 4);
    }
}
