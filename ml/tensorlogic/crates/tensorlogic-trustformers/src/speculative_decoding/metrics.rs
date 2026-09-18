//! Runtime metrics for speculative decoding.
//!
//! The engine updates a [`SpeculativeMetrics`] struct at the end of each
//! speculative step.  The three headline values are:
//!
//! | Field                   | Meaning                                            |
//! |-------------------------|----------------------------------------------------|
//! | `accept_rate`           | Fraction of drafted tokens accepted by the target. |
//! | `tokens_per_step_avg`   | Expected accepted tokens per round (≤ `k + 1`).    |
//! | `speedup_estimate`      | Modeled wall-clock speedup over pure target decode.|
//!
//! The speedup model follows Leviathan et al. §4: with per-round cost
//! `c_draft * k + c_target`, and on average `E[accepted+1]` tokens per round,
//! the naive serial baseline decodes 1 token per `c_target`, so the speedup
//! simplifies (with the common `c_draft ≪ c_target` assumption) to
//! `E[accepted+1] / (1 + k * (c_draft / c_target))`.  We expose the `c_ratio`
//! as a configurable field so callers can plug in whatever cost ratio the
//! deployment suggests.

/// Running metrics maintained inside
/// [`SpeculativeDecoder::generate`](crate::speculative_decoding::engine::SpeculativeDecoder::generate).
#[derive(Debug, Clone, PartialEq)]
pub struct SpeculativeMetrics {
    /// Fraction of individual draft tokens that the target accepted.
    pub accept_rate: f32,
    /// Average number of committed tokens per round (≤ `k + 1`).
    pub tokens_per_step_avg: f32,
    /// Analytical speedup estimate vs. pure target-model decoding.
    pub speedup_estimate: f32,
    /// Total number of speculative rounds executed so far.
    pub rounds: u64,
    /// Total draft tokens proposed.
    pub total_drafted: u64,
    /// Total draft tokens accepted.
    pub total_accepted: u64,
    /// Total tokens committed (accepted + resamples + bonus picks).
    pub total_committed: u64,
    /// `c_draft / c_target` ratio used for the speedup estimate.  Defaults
    /// to `0.125` per the 8× cost gap typical of small-draft + big-target
    /// pairings in Leviathan et al.
    pub cost_ratio: f32,
}

impl Default for SpeculativeMetrics {
    fn default() -> Self {
        Self {
            accept_rate: 0.0,
            tokens_per_step_avg: 0.0,
            speedup_estimate: 1.0,
            rounds: 0,
            total_drafted: 0,
            total_accepted: 0,
            total_committed: 0,
            cost_ratio: 0.125,
        }
    }
}

impl SpeculativeMetrics {
    /// Create a fresh metrics container; see [`Self::with_cost_ratio`] to
    /// tune the speedup model.
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure `c_draft / c_target` for the speedup estimate.
    pub fn with_cost_ratio(mut self, cost_ratio: f32) -> Self {
        self.cost_ratio = cost_ratio.max(0.0);
        self
    }

    /// Update the running statistics with the outcome of a single round.
    ///
    /// * `drafted` — number of draft tokens proposed (typically `k`).
    /// * `accepted` — number of those that survived the rejection test.
    /// * `committed` — total tokens appended to the output this round
    ///   (`accepted + 1` on rejection *or* `k + 1` on full acceptance).
    pub fn record_round(&mut self, drafted: u32, accepted: u32, committed: u32, k: u32) {
        self.rounds = self.rounds.saturating_add(1);
        self.total_drafted = self.total_drafted.saturating_add(drafted as u64);
        self.total_accepted = self.total_accepted.saturating_add(accepted as u64);
        self.total_committed = self.total_committed.saturating_add(committed as u64);

        if self.total_drafted > 0 {
            self.accept_rate = self.total_accepted as f32 / self.total_drafted as f32;
        }
        if self.rounds > 0 {
            self.tokens_per_step_avg = self.total_committed as f32 / self.rounds as f32;
        }

        // Cost of one round ≈ c_target * (1 + k * cost_ratio).
        // Average tokens per round = tokens_per_step_avg.
        // Baseline decodes 1 token per c_target round.
        let round_cost = 1.0 + (k as f32) * self.cost_ratio;
        self.speedup_estimate = if round_cost > 0.0 {
            self.tokens_per_step_avg / round_cost
        } else {
            1.0
        };
    }

    /// Reset counters to zero while keeping the configured `cost_ratio`.
    pub fn reset(&mut self) {
        let keep = self.cost_ratio;
        *self = Self::default();
        self.cost_ratio = keep;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_metrics_default() {
        let m = SpeculativeMetrics::new();
        assert_eq!(m.accept_rate, 0.0);
        assert_eq!(m.tokens_per_step_avg, 0.0);
        assert!((m.speedup_estimate - 1.0).abs() < 1e-6);
        assert_eq!(m.rounds, 0);
    }

    #[test]
    fn record_round_updates_everything() {
        let mut m = SpeculativeMetrics::new();
        m.record_round(4, 3, 4, 4);
        assert_eq!(m.rounds, 1);
        assert_eq!(m.total_drafted, 4);
        assert_eq!(m.total_accepted, 3);
        assert_eq!(m.total_committed, 4);
        assert!((m.accept_rate - 0.75).abs() < 1e-6);
        assert!((m.tokens_per_step_avg - 4.0).abs() < 1e-6);
        // speedup = tokens_per_step / (1 + k * cost_ratio) = 4 / (1 + 4*0.125) = 4/1.5.
        assert!((m.speedup_estimate - 4.0 / 1.5).abs() < 1e-4);
    }

    #[test]
    fn rolling_averages_accumulate() {
        let mut m = SpeculativeMetrics::new();
        m.record_round(4, 4, 5, 4); // full accept ⇒ k+1 committed.
        m.record_round(4, 0, 1, 4); // full reject ⇒ 1 committed.
        assert_eq!(m.rounds, 2);
        assert_eq!(m.total_accepted, 4);
        assert!((m.accept_rate - 0.5).abs() < 1e-6);
        assert!((m.tokens_per_step_avg - 3.0).abs() < 1e-6);
    }

    #[test]
    fn reset_keeps_cost_ratio() {
        let mut m = SpeculativeMetrics::new().with_cost_ratio(0.05);
        m.record_round(4, 3, 4, 4);
        m.reset();
        assert_eq!(m.rounds, 0);
        assert!((m.cost_ratio - 0.05).abs() < 1e-6);
    }

    #[test]
    fn with_cost_ratio_clamps_negative() {
        let m = SpeculativeMetrics::new().with_cost_ratio(-0.5);
        assert_eq!(m.cost_ratio, 0.0);
    }
}
