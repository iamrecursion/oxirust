//! Multi-armed bandit algorithms for adaptive media experiments.
//!
//! Implements two exploration/exploitation strategies:
//!
//! * **Epsilon-greedy** — with probability ε explore uniformly; otherwise
//!   exploit the arm with the highest empirical mean reward.
//! * **Thompson sampling** — maintain Beta(α, β) posteriors for each arm
//!   and select the arm whose posterior sample is highest.
//!
//! Both algorithms are designed for Bernoulli rewards (click / no-click,
//! view / no-view) and are therefore modelled with Beta-Binomial conjugacy.
//!
//! All operations are deterministic given the same seed, which makes
//! experiments reproducible.

use crate::error::AnalyticsError;

// ─── Arm model ────────────────────────────────────────────────────────────────

/// One arm in a multi-armed bandit experiment.
#[derive(Debug, Clone)]
pub struct BanditArm {
    /// Unique identifier matching a variant ID in the parent experiment.
    pub id: String,
    /// Total number of times this arm has been pulled (impressions).
    pub pulls: u64,
    /// Total reward accumulated (successes / conversions).
    pub reward_sum: u64,
    /// Bayesian posterior α = prior_α + reward_sum.
    pub beta_alpha: f64,
    /// Bayesian posterior β = prior_β + (pulls − reward_sum).
    pub beta_beta: f64,
}

impl BanditArm {
    /// Create a new arm with a Jeffrey's uninformative prior Beta(0.5, 0.5).
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            pulls: 0,
            reward_sum: 0,
            beta_alpha: 0.5,
            beta_beta: 0.5,
        }
    }

    /// Record a Bernoulli outcome: `reward = 1` for success, `0` for failure.
    pub fn record_outcome(&mut self, reward: u32) {
        self.pulls += 1;
        let r = reward.min(1) as u64;
        self.reward_sum += r;
        self.beta_alpha += r as f64;
        self.beta_beta += (1 - reward.min(1)) as f64;
    }

    /// Empirical mean reward rate (0.0 if no pulls yet).
    pub fn empirical_mean(&self) -> f64 {
        if self.pulls == 0 {
            return 0.0;
        }
        self.reward_sum as f64 / self.pulls as f64
    }

    /// Posterior mean of the Beta distribution: α / (α + β).
    pub fn posterior_mean(&self) -> f64 {
        self.beta_alpha / (self.beta_alpha + self.beta_beta)
    }
}

// ─── Bandit state machine ─────────────────────────────────────────────────────

/// Configuration and state of a multi-armed bandit experiment.
#[derive(Debug, Clone)]
pub struct MultiArmedBandit {
    /// All arms (variants) in this experiment.
    pub arms: Vec<BanditArm>,
    /// Total pulls across all arms.
    pub total_pulls: u64,
    /// Active exploration strategy.
    pub strategy: BanditStrategy,
    /// Reproducible RNG state (xoshiro256**).
    rng_state: [u64; 4],
}

/// Strategy used for arm selection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BanditStrategy {
    /// Explore uniformly with probability `epsilon`; otherwise exploit best arm.
    EpsilonGreedy { epsilon: f64 },
    /// Sample a reward probability from each arm's Beta posterior and select
    /// the arm with the highest sample.
    ThompsonSampling,
}

impl MultiArmedBandit {
    /// Create a new bandit with the given arm IDs and strategy.
    ///
    /// Returns an error if `arm_ids` is empty.
    pub fn new(
        arm_ids: &[&str],
        strategy: BanditStrategy,
        seed: u64,
    ) -> Result<Self, AnalyticsError> {
        if arm_ids.is_empty() {
            return Err(AnalyticsError::ConfigError(
                "bandit must have at least one arm".to_string(),
            ));
        }
        let arms = arm_ids.iter().map(|id| BanditArm::new(*id)).collect();
        // Initialise the xoshiro256** state from the seed.
        let rng_state = [
            seed.wrapping_add(0x9e37_79b9_7f4a_7c15),
            seed.wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407),
            seed ^ 0xdeadbeef_cafebabe,
            seed.rotate_left(17).wrapping_add(0x0123_4567_89ab_cdef),
        ];
        Ok(Self {
            arms,
            total_pulls: 0,
            strategy,
            rng_state,
        })
    }

    /// Select an arm to pull according to the configured strategy.
    ///
    /// Returns the index (and id) of the selected arm.
    pub fn select_arm(&mut self) -> Result<usize, AnalyticsError> {
        if self.arms.is_empty() {
            return Err(AnalyticsError::ConfigError(
                "bandit has no arms".to_string(),
            ));
        }
        match self.strategy {
            BanditStrategy::EpsilonGreedy { epsilon } => self.select_epsilon_greedy(epsilon),
            BanditStrategy::ThompsonSampling => self.select_thompson(),
        }
    }

    /// Record an outcome for the arm at `arm_index`.
    pub fn record_outcome(&mut self, arm_index: usize, reward: u32) -> Result<(), AnalyticsError> {
        let n = self.arms.len();
        if arm_index >= n {
            return Err(AnalyticsError::ConfigError(format!(
                "arm index {} out of range (len={})",
                arm_index, n
            )));
        }
        self.arms[arm_index].record_outcome(reward);
        self.total_pulls += 1;
        Ok(())
    }

    /// Return the index of the arm with the highest posterior mean.
    pub fn best_arm_index(&self) -> Option<usize> {
        self.arms
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.posterior_mean()
                    .partial_cmp(&b.posterior_mean())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
    }

    /// Return a slice of (arm_id, posterior_mean) pairs sorted descending by mean.
    pub fn arm_rankings(&self) -> Vec<(&str, f64)> {
        let mut ranked: Vec<_> = self
            .arms
            .iter()
            .map(|a| (a.id.as_str(), a.posterior_mean()))
            .collect();
        ranked.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    fn select_epsilon_greedy(&mut self, epsilon: f64) -> Result<usize, AnalyticsError> {
        let u = self.next_f64();
        if u < epsilon {
            // Explore: uniform random arm.
            let r = self.next_f64();
            let idx = (r * self.arms.len() as f64) as usize;
            Ok(idx.min(self.arms.len() - 1))
        } else {
            // Exploit: arm with highest empirical mean; break ties by index.
            self.arms
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    a.empirical_mean()
                        .partial_cmp(&b.empirical_mean())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .ok_or_else(|| AnalyticsError::ConfigError("no arms to exploit".to_string()))
        }
    }

    fn select_thompson(&mut self) -> Result<usize, AnalyticsError> {
        // Collect (alpha, beta) pairs first to avoid borrowing `self.arms` and
        // `self` simultaneously in the closure.
        let params: Vec<(f64, f64)> = self
            .arms
            .iter()
            .map(|arm| (arm.beta_alpha, arm.beta_beta))
            .collect();

        let best = params
            .into_iter()
            .enumerate()
            .map(|(i, (alpha, beta))| {
                let sample = self.sample_beta_internal(alpha, beta);
                (i, sample)
            })
            .max_by(|(_, s1), (_, s2)| s1.partial_cmp(s2).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i);
        best.ok_or_else(|| AnalyticsError::ConfigError("no arms for Thompson sampling".to_string()))
    }

    /// Sample one value from Beta(alpha, beta).
    ///
    /// Uses Johnk's method for α, β ≤ 1 and Cheng's BB algorithm otherwise.
    fn sample_beta_internal(&mut self, alpha: f64, beta: f64) -> f64 {
        if alpha <= 1.0 && beta <= 1.0 {
            // Johnk's method.
            loop {
                let u = self.next_f64();
                let v = self.next_f64();
                let x = u.powf(1.0 / alpha);
                let y = v.powf(1.0 / beta);
                let s = x + y;
                if s <= 1.0 && s > 0.0 {
                    return x / s;
                }
            }
        }
        // Cheng's BB algorithm.
        let (a, b) = if alpha < beta {
            (alpha, beta)
        } else {
            (beta, alpha)
        };
        let lambda = ((a + b) / (2.0 * a * b)).sqrt();
        let c = a + 1.0 / lambda;
        loop {
            let u1 = self.next_f64();
            let v = (u1 / (1.0 - u1 + f64::EPSILON)).ln() / lambda;
            let w = a * v.exp();
            let u2 = self.next_f64();
            let z = u1 * u1 * u2;
            let r = c * v - 4.0_f64.ln();
            let s_val = a + r - w;
            let x = w / (b + w);
            if s_val + (a + b + 1.0).ln() >= 5.0 * (a + b).ln() + z.ln() {
                return if alpha < beta { x } else { 1.0 - x };
            }
            if r >= z.ln() {
                return if alpha < beta { x } else { 1.0 - x };
            }
        }
    }

    fn next_u64(&mut self) -> u64 {
        let [s0, s1, s2, s3] = self.rng_state;
        let result = s1.wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s1 << 17;
        self.rng_state[2] ^= s0;
        self.rng_state[3] ^= s1;
        self.rng_state[1] ^= s2;
        self.rng_state[0] ^= s3;
        self.rng_state[2] ^= t;
        self.rng_state[3] = s3.rotate_left(45);
        result
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

// ─── Regret tracking ──────────────────────────────────────────────────────────

/// Tracks cumulative regret for a bandit simulation.
///
/// Regret at step t = Σ_i (μ* − μ_{a_i}) where μ* is the best arm's true
/// reward rate and μ_{a_i} is the rate of the arm actually chosen at step i.
#[derive(Debug, Clone, Default)]
pub struct RegretTracker {
    /// Cumulative pseudo-regret accumulated so far.
    pub cumulative_regret: f64,
    /// Number of steps recorded.
    pub steps: u64,
    /// Running history of per-step regret values.
    pub regret_history: Vec<f64>,
}

impl RegretTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one step: `true_best_rate` is μ*, `chosen_arm_rate` is μ_{a_t}.
    pub fn record_step(&mut self, true_best_rate: f64, chosen_arm_rate: f64) {
        let step_regret = (true_best_rate - chosen_arm_rate).max(0.0);
        self.cumulative_regret += step_regret;
        self.steps += 1;
        self.regret_history.push(step_regret);
    }

    /// Average regret per step.
    pub fn average_regret(&self) -> f64 {
        if self.steps == 0 {
            return 0.0;
        }
        self.cumulative_regret / self.steps as f64
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn three_arm_bandit(strategy: BanditStrategy) -> MultiArmedBandit {
        MultiArmedBandit::new(&["arm_a", "arm_b", "arm_c"], strategy, 42)
            .expect("new should succeed")
    }

    // ── BanditArm ─────────────────────────────────────────────────────────────

    #[test]
    fn arm_new_has_jeffreys_prior() {
        let arm = BanditArm::new("test");
        assert!((arm.beta_alpha - 0.5).abs() < 1e-10);
        assert!((arm.beta_beta - 0.5).abs() < 1e-10);
        assert_eq!(arm.pulls, 0);
    }

    #[test]
    fn arm_record_outcome_updates_stats() {
        let mut arm = BanditArm::new("a");
        arm.record_outcome(1);
        arm.record_outcome(0);
        arm.record_outcome(1);
        assert_eq!(arm.pulls, 3);
        assert_eq!(arm.reward_sum, 2);
        assert!((arm.beta_alpha - 2.5).abs() < 1e-10);
        assert!((arm.beta_beta - 1.5).abs() < 1e-10);
    }

    #[test]
    fn arm_empirical_mean_zero_pulls() {
        let arm = BanditArm::new("a");
        assert_eq!(arm.empirical_mean(), 0.0);
    }

    #[test]
    fn arm_empirical_mean_correct() {
        let mut arm = BanditArm::new("a");
        for _ in 0..6 {
            arm.record_outcome(1);
        }
        for _ in 0..4 {
            arm.record_outcome(0);
        }
        assert!((arm.empirical_mean() - 0.6).abs() < 1e-10);
    }

    // ── MultiArmedBandit construction ─────────────────────────────────────────

    #[test]
    fn bandit_new_empty_arms_returns_error() {
        let result = MultiArmedBandit::new(&[], BanditStrategy::EpsilonGreedy { epsilon: 0.1 }, 0);
        assert!(result.is_err());
    }

    #[test]
    fn bandit_new_single_arm() {
        let bandit = MultiArmedBandit::new(&["only"], BanditStrategy::ThompsonSampling, 1);
        assert!(bandit.is_ok());
    }

    // ── Epsilon-greedy ────────────────────────────────────────────────────────

    #[test]
    fn epsilon_greedy_always_exploits_at_zero_epsilon() {
        // With ε = 0 the bandit always exploits; after seeding one arm as best,
        // it should always return that arm.
        let mut bandit = MultiArmedBandit::new(
            &["low", "high"],
            BanditStrategy::EpsilonGreedy { epsilon: 0.0 },
            99,
        )
        .expect("value should be present should succeed");
        // Give arm[1] many successes so it dominates.
        for _ in 0..100 {
            bandit
                .record_outcome(1, 1)
                .expect("record outcome should succeed");
        }
        let selected = bandit.select_arm().expect("select arm should succeed");
        assert_eq!(selected, 1, "should always pick the best arm (index 1)");
    }

    #[test]
    fn epsilon_greedy_explores_at_full_epsilon() {
        // With ε = 1.0 the bandit always explores uniformly.
        let mut bandit = MultiArmedBandit::new(
            &["a", "b", "c"],
            BanditStrategy::EpsilonGreedy { epsilon: 1.0 },
            7,
        )
        .expect("value should be present should succeed");
        // Over 300 pulls every arm should be chosen at least once.
        let mut counts = [0usize; 3];
        for _ in 0..300 {
            let idx = bandit.select_arm().expect("select arm should succeed");
            counts[idx] += 1;
        }
        for (i, &c) in counts.iter().enumerate() {
            assert!(c > 0, "arm {i} was never explored");
        }
    }

    #[test]
    fn epsilon_greedy_converges_to_best_arm() {
        // Simulate a 3-arm bandit: arm 0 rate=0.1, arm 1 rate=0.5, arm 2 rate=0.2.
        // With ε = 0.1 and enough rounds, arm 1 should dominate.
        let true_rates = [0.1, 0.5, 0.2];
        let mut bandit = MultiArmedBandit::new(
            &["arm0", "arm1", "arm2"],
            BanditStrategy::EpsilonGreedy { epsilon: 0.1 },
            123,
        )
        .expect("value should be present should succeed");
        // Use a deterministic oracle for reward generation.
        let mut oracle_rng = Xoshiro256Helper::new(42);

        for _ in 0..2000 {
            let idx = bandit.select_arm().expect("select arm should succeed");
            let reward = if oracle_rng.next_f64() < true_rates[idx] {
                1
            } else {
                0
            };
            bandit
                .record_outcome(idx, reward)
                .expect("record outcome should succeed");
        }
        let best = bandit
            .best_arm_index()
            .expect("best arm index should succeed");
        assert_eq!(best, 1, "arm 1 (rate=0.5) should be identified as best");
    }

    // ── Thompson sampling ────────────────────────────────────────────────────

    #[test]
    fn thompson_sampling_returns_valid_arm_index() {
        let mut bandit = three_arm_bandit(BanditStrategy::ThompsonSampling);
        let idx = bandit.select_arm().expect("select arm should succeed");
        assert!(idx < 3);
    }

    #[test]
    fn thompson_sampling_converges_to_best_arm() {
        let true_rates = [0.05, 0.40, 0.15];
        let mut bandit = MultiArmedBandit::new(
            &["arm0", "arm1", "arm2"],
            BanditStrategy::ThompsonSampling,
            999,
        )
        .expect("value should be present should succeed");
        let mut oracle = Xoshiro256Helper::new(77);

        for _ in 0..1000 {
            let idx = bandit.select_arm().expect("select arm should succeed");
            let reward = if oracle.next_f64() < true_rates[idx] {
                1
            } else {
                0
            };
            bandit
                .record_outcome(idx, reward)
                .expect("record outcome should succeed");
        }
        let best = bandit
            .best_arm_index()
            .expect("best arm index should succeed");
        assert_eq!(best, 1, "arm 1 (rate=0.40) should be best");
    }

    // ── RegretTracker ────────────────────────────────────────────────────────

    #[test]
    fn regret_tracker_no_steps() {
        let tracker = RegretTracker::new();
        assert_eq!(tracker.average_regret(), 0.0);
        assert_eq!(tracker.cumulative_regret, 0.0);
    }

    #[test]
    fn regret_tracker_accumulates_correctly() {
        let mut tracker = RegretTracker::new();
        tracker.record_step(0.5, 0.4); // regret = 0.1
        tracker.record_step(0.5, 0.5); // regret = 0.0
        tracker.record_step(0.5, 0.2); // regret = 0.3
        assert!((tracker.cumulative_regret - 0.4).abs() < 1e-10);
        assert!((tracker.average_regret() - 0.4 / 3.0).abs() < 1e-10);
        assert_eq!(tracker.regret_history.len(), 3);
    }

    #[test]
    fn regret_never_negative() {
        let mut tracker = RegretTracker::new();
        // Chosen arm better than "best" — regret clamped to 0.
        tracker.record_step(0.3, 0.9);
        assert_eq!(tracker.cumulative_regret, 0.0);
    }

    // ── arm_rankings ────────────────────────────────────────────────────────

    #[test]
    fn arm_rankings_sorted_descending() {
        let mut bandit =
            MultiArmedBandit::new(&["a", "b", "c"], BanditStrategy::ThompsonSampling, 0)
                .expect("new should succeed");
        // arm 0: 1/10, arm 1: 8/10, arm 2: 4/10
        for _ in 0..1 {
            bandit
                .record_outcome(0, 1)
                .expect("record outcome should succeed");
        }
        for _ in 0..9 {
            bandit
                .record_outcome(0, 0)
                .expect("record outcome should succeed");
        }
        for _ in 0..8 {
            bandit
                .record_outcome(1, 1)
                .expect("record outcome should succeed");
        }
        for _ in 0..2 {
            bandit
                .record_outcome(1, 0)
                .expect("record outcome should succeed");
        }
        for _ in 0..4 {
            bandit
                .record_outcome(2, 1)
                .expect("record outcome should succeed");
        }
        for _ in 0..6 {
            bandit
                .record_outcome(2, 0)
                .expect("record outcome should succeed");
        }
        let rankings = bandit.arm_rankings();
        assert_eq!(rankings[0].0, "b");
        assert_eq!(rankings[2].0, "a");
    }
}

// ── Helper RNG for tests only ────────────────────────────────────────────────

#[cfg(test)]
struct Xoshiro256Helper([u64; 4]);

#[cfg(test)]
impl Xoshiro256Helper {
    fn new(seed: u64) -> Self {
        let s = [
            seed.wrapping_add(0x9e37_79b9_7f4a_7c15),
            seed.wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407),
            seed ^ 0xdeadbeef_cafebabe,
            seed.rotate_left(17).wrapping_add(0x0123_4567_89ab_cdef),
        ];
        Self(s)
    }

    fn next_u64(&mut self) -> u64 {
        let [s0, s1, s2, s3] = self.0;
        let result = s1.wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s1 << 17;
        self.0[2] ^= s0;
        self.0[3] ^= s1;
        self.0[1] ^= s2;
        self.0[0] ^= s3;
        self.0[2] ^= t;
        self.0[3] = s3.rotate_left(45);
        result
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}
