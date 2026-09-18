#![allow(dead_code)]
//! Render job priority scoring and scheduling order.
//!
//! Computes a composite priority score for render jobs based on multiple
//! weighted factors: explicit priority level, deadline proximity, job age,
//! estimated render cost, and artist seniority. The resulting score determines
//! the scheduling order in the render farm queue.

use std::fmt;

// ─── Priority Factor ────────────────────────────────────────────────────────

/// Individual factor contributing to the composite priority score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PriorityFactor {
    /// Explicit priority level set by the user.
    ExplicitLevel,
    /// How close the job is to its deadline.
    DeadlineProximity,
    /// How long the job has been waiting in the queue.
    QueueAge,
    /// Estimated render cost (inverse — cheaper jobs get slight boost).
    RenderCost,
    /// Artist / department seniority weight.
    ArtistWeight,
}

impl fmt::Display for PriorityFactor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::ExplicitLevel => "Explicit Level",
            Self::DeadlineProximity => "Deadline Proximity",
            Self::QueueAge => "Queue Age",
            Self::RenderCost => "Render Cost",
            Self::ArtistWeight => "Artist Weight",
        };
        f.write_str(label)
    }
}

// ─── Priority Weights ───────────────────────────────────────────────────────

/// Weights for each priority factor.
#[derive(Debug, Clone, PartialEq)]
pub struct PriorityWeights {
    /// Weight for explicit priority level [0, 1].
    pub explicit_level: f64,
    /// Weight for deadline proximity [0, 1].
    pub deadline_proximity: f64,
    /// Weight for queue age [0, 1].
    pub queue_age: f64,
    /// Weight for render cost [0, 1].
    pub render_cost: f64,
    /// Weight for artist seniority [0, 1].
    pub artist_weight: f64,
}

impl Default for PriorityWeights {
    fn default() -> Self {
        Self {
            explicit_level: 0.40,
            deadline_proximity: 0.25,
            queue_age: 0.15,
            render_cost: 0.10,
            artist_weight: 0.10,
        }
    }
}

impl PriorityWeights {
    /// Create new weights. Values are clamped to `[0, 1]`.
    #[must_use]
    pub fn new(
        explicit_level: f64,
        deadline_proximity: f64,
        queue_age: f64,
        render_cost: f64,
        artist_weight: f64,
    ) -> Self {
        Self {
            explicit_level: explicit_level.clamp(0.0, 1.0),
            deadline_proximity: deadline_proximity.clamp(0.0, 1.0),
            queue_age: queue_age.clamp(0.0, 1.0),
            render_cost: render_cost.clamp(0.0, 1.0),
            artist_weight: artist_weight.clamp(0.0, 1.0),
        }
    }

    /// Sum of all weights (useful for normalisation checks).
    #[must_use]
    pub fn total(&self) -> f64 {
        self.explicit_level
            + self.deadline_proximity
            + self.queue_age
            + self.render_cost
            + self.artist_weight
    }

    /// Returns `true` if all weights sum to approximately 1.0.
    #[must_use]
    pub fn is_normalised(&self) -> bool {
        (self.total() - 1.0).abs() < 1e-6
    }
}

// ─── Priority Input ─────────────────────────────────────────────────────────

/// Normalised input values for priority scoring (all in `[0, 1]`).
#[derive(Debug, Clone, PartialEq)]
pub struct PriorityInput {
    /// Normalised explicit priority (0 = lowest, 1 = urgent).
    pub explicit_level: f64,
    /// Normalised deadline proximity (0 = far away, 1 = imminent).
    pub deadline_proximity: f64,
    /// Normalised queue age (0 = just submitted, 1 = long wait).
    pub queue_age: f64,
    /// Normalised render cost factor (0 = expensive, 1 = cheap).
    pub render_cost: f64,
    /// Normalised artist weight (0 = junior, 1 = senior lead).
    pub artist_weight: f64,
}

impl Default for PriorityInput {
    fn default() -> Self {
        Self {
            explicit_level: 0.5,
            deadline_proximity: 0.0,
            queue_age: 0.0,
            render_cost: 0.5,
            artist_weight: 0.5,
        }
    }
}

impl PriorityInput {
    /// Create inputs with all factors at zero.
    #[must_use]
    pub fn zeros() -> Self {
        Self {
            explicit_level: 0.0,
            deadline_proximity: 0.0,
            queue_age: 0.0,
            render_cost: 0.0,
            artist_weight: 0.0,
        }
    }
}

// ─── Priority Score ─────────────────────────────────────────────────────────

/// Composite priority score with per-factor breakdown.
#[derive(Debug, Clone, PartialEq)]
pub struct PriorityScore {
    /// Final composite score (higher = more urgent).
    pub total: f64,
    /// Individual weighted contributions.
    pub breakdown: Vec<(PriorityFactor, f64)>,
}

impl PriorityScore {
    /// Returns the dominant factor (the one contributing the most).
    #[must_use]
    pub fn dominant_factor(&self) -> Option<PriorityFactor> {
        self.breakdown
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(f, _)| *f)
    }
}

impl fmt::Display for PriorityScore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "score={:.4}", self.total)
    }
}

// ─── Priority Scorer ────────────────────────────────────────────────────────

/// Computes composite priority scores for render jobs.
#[derive(Debug, Clone)]
pub struct PriorityScorer {
    /// Factor weights.
    pub weights: PriorityWeights,
    /// Optional boost multiplier for jobs past their soft deadline.
    pub overdue_boost: f64,
}

impl Default for PriorityScorer {
    fn default() -> Self {
        Self {
            weights: PriorityWeights::default(),
            overdue_boost: 1.5,
        }
    }
}

impl PriorityScorer {
    /// Create a scorer with default weights.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a scorer with custom weights.
    #[must_use]
    pub fn with_weights(weights: PriorityWeights) -> Self {
        Self {
            weights,
            ..Self::default()
        }
    }

    /// Set the overdue boost multiplier.
    #[must_use]
    pub fn with_overdue_boost(mut self, boost: f64) -> Self {
        self.overdue_boost = boost.max(1.0);
        self
    }

    /// Compute the priority score for a single job's inputs.
    #[must_use]
    pub fn score(&self, input: &PriorityInput) -> PriorityScore {
        let w = &self.weights;
        let contributions = vec![
            (
                PriorityFactor::ExplicitLevel,
                w.explicit_level * input.explicit_level,
            ),
            (
                PriorityFactor::DeadlineProximity,
                w.deadline_proximity * input.deadline_proximity,
            ),
            (PriorityFactor::QueueAge, w.queue_age * input.queue_age),
            (
                PriorityFactor::RenderCost,
                w.render_cost * input.render_cost,
            ),
            (
                PriorityFactor::ArtistWeight,
                w.artist_weight * input.artist_weight,
            ),
        ];
        let mut total: f64 = contributions.iter().map(|(_, v)| v).sum();
        // Apply overdue boost if deadline proximity is maxed out
        if input.deadline_proximity >= 1.0 {
            total *= self.overdue_boost;
        }
        PriorityScore {
            total,
            breakdown: contributions,
        }
    }

    /// Score a batch of jobs and return scores sorted descending (highest first).
    #[must_use]
    pub fn rank(&self, inputs: &[PriorityInput]) -> Vec<(usize, PriorityScore)> {
        let mut scored: Vec<(usize, PriorityScore)> = inputs
            .iter()
            .enumerate()
            .map(|(i, inp)| (i, self.score(inp)))
            .collect();
        scored.sort_by(|a, b| {
            b.1.total
                .partial_cmp(&a.1.total)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored
    }
}

/// Convert an explicit priority enum level (0..3) to a normalised `[0, 1]` value.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn normalise_priority_level(level: u32, max_level: u32) -> f64 {
    if max_level == 0 {
        return 0.0;
    }
    (f64::from(level) / f64::from(max_level)).clamp(0.0, 1.0)
}

/// Convert a deadline offset in seconds to a normalised proximity value.
///
/// 0 or negative seconds → 1.0 (imminent / overdue).
/// `>= horizon_secs` → 0.0.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn normalise_deadline(seconds_remaining: i64, horizon_secs: i64) -> f64 {
    if seconds_remaining <= 0 {
        return 1.0;
    }
    if horizon_secs <= 0 {
        return 0.0;
    }
    (1.0 - seconds_remaining as f64 / horizon_secs as f64).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_priority_weights_default_normalised() {
        let w = PriorityWeights::default();
        assert!(
            w.is_normalised(),
            "default weights should sum to 1.0, got {}",
            w.total()
        );
    }

    #[test]
    fn test_priority_weights_clamped() {
        let w = PriorityWeights::new(2.0, -1.0, 0.5, 0.5, 0.5);
        assert!((w.explicit_level - 1.0).abs() < 1e-12);
        assert!((w.deadline_proximity - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_priority_input_default() {
        let inp = PriorityInput::default();
        assert!((inp.explicit_level - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_priority_input_zeros() {
        let inp = PriorityInput::zeros();
        assert!((inp.explicit_level - 0.0).abs() < 1e-12);
        assert!((inp.queue_age - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_scorer_all_zeros() {
        let scorer = PriorityScorer::new();
        let score = scorer.score(&PriorityInput::zeros());
        assert!((score.total - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_scorer_max_explicit() {
        let scorer = PriorityScorer::new();
        let mut input = PriorityInput::zeros();
        input.explicit_level = 1.0;
        let score = scorer.score(&input);
        assert!(
            (score.total - 0.40).abs() < 1e-9,
            "expected 0.40 for max explicit, got {}",
            score.total
        );
    }

    #[test]
    fn test_scorer_overdue_boost() {
        let scorer = PriorityScorer::new().with_overdue_boost(2.0);
        let mut input = PriorityInput::zeros();
        input.deadline_proximity = 1.0;
        input.explicit_level = 1.0;
        let score = scorer.score(&input);
        // Without boost: 0.40 + 0.25 = 0.65; with 2x: 1.30
        assert!(
            (score.total - 1.30).abs() < 1e-9,
            "expected 1.30 with overdue boost, got {}",
            score.total
        );
    }

    #[test]
    fn test_scorer_dominant_factor() {
        let scorer = PriorityScorer::new();
        let mut input = PriorityInput::zeros();
        input.queue_age = 1.0;
        let score = scorer.score(&input);
        assert_eq!(score.dominant_factor(), Some(PriorityFactor::QueueAge));
    }

    #[test]
    fn test_scorer_rank_order() {
        let scorer = PriorityScorer::new();
        let inputs = vec![
            {
                let mut i = PriorityInput::zeros();
                i.explicit_level = 0.2;
                i
            },
            {
                let mut i = PriorityInput::zeros();
                i.explicit_level = 0.9;
                i
            },
            {
                let mut i = PriorityInput::zeros();
                i.explicit_level = 0.5;
                i
            },
        ];
        let ranked = scorer.rank(&inputs);
        assert_eq!(ranked[0].0, 1); // highest explicit
        assert_eq!(ranked[1].0, 2);
        assert_eq!(ranked[2].0, 0); // lowest explicit
    }

    #[test]
    fn test_normalise_priority_level() {
        assert!((normalise_priority_level(0, 3) - 0.0).abs() < 1e-12);
        assert!((normalise_priority_level(3, 3) - 1.0).abs() < 1e-12);
        assert!((normalise_priority_level(1, 2) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_normalise_priority_level_zero_max() {
        assert!((normalise_priority_level(5, 0) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_normalise_deadline_imminent() {
        assert!((normalise_deadline(0, 3600) - 1.0).abs() < 1e-12);
        assert!((normalise_deadline(-100, 3600) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_normalise_deadline_far() {
        assert!((normalise_deadline(3600, 3600) - 0.0).abs() < 1e-12);
        assert!((normalise_deadline(7200, 3600) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_normalise_deadline_mid() {
        let v = normalise_deadline(1800, 3600);
        assert!((v - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_priority_score_display() {
        let score = PriorityScore {
            total: 0.75,
            breakdown: vec![],
        };
        let s = format!("{score}");
        assert!(s.contains("0.75"));
    }

    #[test]
    fn test_priority_factor_display() {
        assert_eq!(
            format!("{}", PriorityFactor::DeadlineProximity),
            "Deadline Proximity"
        );
    }

    #[test]
    fn test_custom_weights() {
        let w = PriorityWeights::new(1.0, 0.0, 0.0, 0.0, 0.0);
        let scorer = PriorityScorer::with_weights(w);
        let mut input = PriorityInput::zeros();
        input.explicit_level = 0.8;
        let score = scorer.score(&input);
        assert!((score.total - 0.8).abs() < 1e-9);
    }
}
