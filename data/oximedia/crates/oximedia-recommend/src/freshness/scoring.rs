//! Content freshness scoring with configurable decay functions.
//!
//! Provides [`FreshnessDecay`], [`ContentAge`], [`FreshnessScorer`], and
//! [`BoostSchedule`] for flexible freshness-aware ranking.

// ---------------------------------------------------------------------------
// FreshnessDecay
// ---------------------------------------------------------------------------

/// Decay function applied to content age when computing freshness scores.
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(dead_code)]
pub enum FreshnessDecay {
    /// Score decays as `exp(-ln(2) * age / half_life)`.
    Exponential,
    /// Score decays linearly: `max(0, 1 - age / (2 * half_life))`.
    Linear,
    /// Score is 1.0 while `age <= half_life`, then 0.0.
    Step,
    /// Score decays as `1 / (1 + log2(1 + age / half_life))`.
    Logarithmic,
}

impl FreshnessDecay {
    /// Compute the decay factor for content with the given age.
    ///
    /// # Arguments
    ///
    /// * `age_days`       – how old the content is, in days
    /// * `half_life_days` – the age (in days) at which the score halves
    ///                      (or changes state for `Step`)
    ///
    /// Returns a value in `[0.0, 1.0]`.
    #[must_use]
    pub fn compute(self, age_days: f64, half_life_days: f64) -> f64 {
        if half_life_days <= 0.0 || age_days < 0.0 {
            return 0.0;
        }
        match self {
            Self::Exponential => {
                let decay_rate = std::f64::consts::LN_2 / half_life_days;
                (-decay_rate * age_days).exp()
            }
            Self::Linear => {
                // Reaches 0 at 2 * half_life
                (1.0 - age_days / (2.0 * half_life_days)).max(0.0)
            }
            Self::Step => {
                if age_days <= half_life_days {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Logarithmic => {
                let ratio = age_days / half_life_days;
                1.0 / (1.0 + (1.0 + ratio).log2())
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ContentAge
// ---------------------------------------------------------------------------

/// Metadata describing the age and update activity of a piece of content.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct ContentAge {
    /// Number of days since the content was first published
    pub published_at_days_ago: u32,
    /// Number of days since the content was last updated/edited
    pub last_updated_days_ago: u32,
    /// Current view velocity (views per day)
    pub view_velocity: f32,
}

impl ContentAge {
    /// Create a new `ContentAge`.
    #[must_use]
    pub fn new(published_at_days_ago: u32, last_updated_days_ago: u32, view_velocity: f32) -> Self {
        Self {
            published_at_days_ago,
            last_updated_days_ago,
            view_velocity: view_velocity.max(0.0),
        }
    }

    /// Return the effective age used for scoring.
    ///
    /// Uses `last_updated_days_ago` when the content has been updated more
    /// recently than it was published (which is always the case if the update
    /// is within the same recency window).
    #[must_use]
    pub fn effective_age_days(&self) -> f64 {
        f64::from(self.last_updated_days_ago.min(self.published_at_days_ago))
    }
}

// ---------------------------------------------------------------------------
// FreshnessScorer
// ---------------------------------------------------------------------------

/// Computes a freshness score for content items.
///
/// The score blends the age-based decay with a view-velocity boost:
///
/// ```text
/// score = decay(age) * (1 + velocity_boost)
/// ```
///
/// where `velocity_boost = view_velocity / max_velocity` capped at 1.0.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct FreshnessScorer {
    /// Half-life in days used for all decay computations
    pub half_life_days: f64,
    /// Maximum expected view velocity (views/day) for normalisation
    pub max_velocity: f32,
    /// Weight given to velocity boost (0.0 = age only)
    pub velocity_weight: f32,
}

impl FreshnessScorer {
    /// Create a new scorer.
    #[must_use]
    pub fn new(half_life_days: f64, max_velocity: f32, velocity_weight: f32) -> Self {
        Self {
            half_life_days: half_life_days.max(0.001),
            max_velocity: max_velocity.max(1.0),
            velocity_weight: velocity_weight.clamp(0.0, 1.0),
        }
    }

    /// Score content based on its age metadata and the chosen decay function.
    ///
    /// Returns a value in `[0.0, 2.0]` (can exceed 1.0 due to velocity boost).
    /// Callers should normalise if a `[0.0, 1.0]` range is required.
    #[must_use]
    pub fn score(&self, age: &ContentAge, decay: FreshnessDecay) -> f32 {
        let base = decay.compute(age.effective_age_days(), self.half_life_days) as f32;

        // Velocity boost: normalised to [0, 1] then scaled by weight
        let velocity_norm = (age.view_velocity / self.max_velocity).min(1.0);
        let boost = velocity_norm * self.velocity_weight;

        (base + boost).min(1.0)
    }
}

impl Default for FreshnessScorer {
    fn default() -> Self {
        Self::new(7.0, 1000.0, 0.2)
    }
}

// ---------------------------------------------------------------------------
// BoostSchedule
// ---------------------------------------------------------------------------

/// A time-based schedule of freshness boosts.
///
/// Each entry is `(day_offset, boost_factor)`.  The boost factor for a given
/// content age is determined by finding the first schedule entry whose
/// `day_offset` is ≥ the content's age in days.  If no entry matches, a
/// factor of `1.0` is returned.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct BoostSchedule {
    /// Sorted list of `(day_offset, boost_factor)` pairs
    pub boosts: Vec<(u32, f32)>,
}

impl BoostSchedule {
    /// Create a new boost schedule from the given pairs.
    ///
    /// The pairs are automatically sorted by `day_offset` ascending.
    #[must_use]
    pub fn new(mut boosts: Vec<(u32, f32)>) -> Self {
        boosts.sort_by_key(|(d, _)| *d);
        Self { boosts }
    }

    /// Look up the boost factor for content that is `age_days` old.
    ///
    /// Returns the factor from the first entry with `day_offset >= age_days`,
    /// or `1.0` if no such entry exists.
    #[must_use]
    pub fn factor_for(&self, age_days: u32) -> f32 {
        for &(day_offset, factor) in &self.boosts {
            if age_days <= day_offset {
                return factor;
            }
        }
        1.0
    }

    /// Apply the boost schedule to a base freshness score.
    #[must_use]
    pub fn apply(&self, base_score: f32, age_days: u32) -> f32 {
        (base_score * self.factor_for(age_days)).min(1.0)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- FreshnessDecay::Exponential --

    #[test]
    fn test_exponential_decay_zero_age() {
        let score = FreshnessDecay::Exponential.compute(0.0, 7.0);
        assert!((score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_exponential_decay_half_life() {
        let score = FreshnessDecay::Exponential.compute(7.0, 7.0);
        assert!((score - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_exponential_decay_decreasing() {
        let s0 = FreshnessDecay::Exponential.compute(0.0, 7.0);
        let s7 = FreshnessDecay::Exponential.compute(7.0, 7.0);
        let s14 = FreshnessDecay::Exponential.compute(14.0, 7.0);
        assert!(s0 > s7 && s7 > s14);
    }

    // -- FreshnessDecay::Linear --

    #[test]
    fn test_linear_decay_zero_age() {
        let score = FreshnessDecay::Linear.compute(0.0, 7.0);
        assert!((score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_linear_decay_after_two_half_lives() {
        let score = FreshnessDecay::Linear.compute(14.0, 7.0);
        assert!(score <= 0.0 + 1e-9);
    }

    // -- FreshnessDecay::Step --

    #[test]
    fn test_step_decay_within() {
        assert!((FreshnessDecay::Step.compute(3.0, 7.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_step_decay_beyond() {
        assert!((FreshnessDecay::Step.compute(8.0, 7.0) - 0.0).abs() < 1e-9);
    }

    // -- FreshnessDecay::Logarithmic --

    #[test]
    fn test_logarithmic_decay_decreasing() {
        let s0 = FreshnessDecay::Logarithmic.compute(0.0, 7.0);
        let s7 = FreshnessDecay::Logarithmic.compute(7.0, 7.0);
        let s70 = FreshnessDecay::Logarithmic.compute(70.0, 7.0);
        assert!(s0 > s7 && s7 > s70);
    }

    // -- ContentAge --

    #[test]
    fn test_content_age_effective_age_uses_minimum() {
        let age = ContentAge::new(30, 5, 100.0);
        assert_eq!(age.effective_age_days(), 5.0);
    }

    // -- FreshnessScorer --

    #[test]
    fn test_freshness_scorer_new_content() {
        let scorer = FreshnessScorer::default();
        let age = ContentAge::new(0, 0, 0.0);
        let score = scorer.score(&age, FreshnessDecay::Exponential);
        assert!(score > 0.9);
    }

    #[test]
    fn test_freshness_scorer_old_content() {
        let scorer = FreshnessScorer::default();
        let age = ContentAge::new(365, 365, 0.0);
        let score = scorer.score(&age, FreshnessDecay::Exponential);
        assert!(score < 0.1);
    }

    // -- BoostSchedule --

    #[test]
    fn test_boost_schedule_first_day() {
        let schedule = BoostSchedule::new(vec![(1, 2.0), (7, 1.5), (30, 1.1)]);
        let factor = schedule.factor_for(0);
        assert!((factor - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_boost_schedule_after_all_windows() {
        let schedule = BoostSchedule::new(vec![(1, 2.0), (7, 1.5)]);
        assert!((schedule.factor_for(30) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_boost_schedule_apply_clamps() {
        let schedule = BoostSchedule::new(vec![(7, 3.0)]);
        let result = schedule.apply(0.8, 3);
        assert!(result <= 1.0);
    }
}
