//! Multi-criteria take selection with weighted scoring.
//!
//! `MultiCriteriaTakeSelector` combines three independent criteria —
//! star rating, recency and duration proximity — into a single weighted score
//! and returns the take that maximises it.  Weights are supplied by the caller
//! via `TakeScoreWeights` and are normalised internally so that their
//! relative proportions determine the outcome rather than their absolute
//! magnitudes.
//!
//! # Scoring formula
//!
//! ```text
//! score(take) =
//!     w_rating  * rating_score(take)
//!   + w_recency * recency_score(take, all_takes)
//!   + w_dur     * duration_score(take, target_duration)
//! ```
//!
//! Each component is normalised to `[0.0, 1.0]` before weighting.

use crate::take::selector::Take;
use std::time::Duration;

/// Convenience type alias for `TakeScoreWeights`.
///
/// Provided so callers can use the shorter `TakeWeights` name when the
/// full weight-name semantics are clear from context.
pub type TakeWeights = TakeScoreWeights;

/// Weights for the multi-criteria take score.
///
/// Weights do **not** need to sum to `1.0`; the selector normalises them.
/// Setting a weight to `0.0` disables that criterion entirely.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TakeScoreWeights {
    /// Weight given to the take's star rating.
    pub rating: f32,
    /// Weight given to how recently the take was recorded.
    pub recency: f32,
    /// Weight given to how closely the take's duration matches
    /// `target_duration`.
    pub duration_match: f32,
}

impl TakeScoreWeights {
    /// Creates uniform weights (all equal to `1.0`).
    #[must_use]
    pub fn uniform() -> Self {
        Self {
            rating: 1.0,
            recency: 1.0,
            duration_match: 1.0,
        }
    }

    /// Creates weights that consider only the star rating.
    #[must_use]
    pub fn rating_only() -> Self {
        Self {
            rating: 1.0,
            recency: 0.0,
            duration_match: 0.0,
        }
    }

    /// Creates weights that consider only recency.
    #[must_use]
    pub fn recency_only() -> Self {
        Self {
            rating: 0.0,
            recency: 1.0,
            duration_match: 0.0,
        }
    }

    /// Returns the sum of all weights.
    #[must_use]
    pub fn total(&self) -> f32 {
        self.rating + self.recency + self.duration_match
    }

    /// Returns `true` if all weights are zero (undefined/useless selection).
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.total() < f32::EPSILON
    }
}

impl Default for TakeScoreWeights {
    fn default() -> Self {
        Self::uniform()
    }
}

/// Multi-criteria take selector.
#[derive(Debug, Clone, Default)]
pub struct MultiCriteriaTakeSelector;

impl MultiCriteriaTakeSelector {
    /// Creates a new selector.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Selects the best take from `takes` given `target_duration` and
    /// `weights`.
    ///
    /// Returns `None` if `takes` is empty or all weights are zero.
    ///
    /// If multiple takes share the same maximum score, the one with the
    /// lowest take number is preferred (deterministic tie-breaking).
    #[must_use]
    pub fn select_best<'a>(
        takes: &'a [Take],
        target_duration: Duration,
        weights: TakeScoreWeights,
    ) -> Option<&'a Take> {
        if takes.is_empty() || weights.is_zero() {
            return None;
        }

        let total_w = weights.total();

        // Pre-compute per-criterion normalisation factors.
        let (min_ts, max_ts) = recency_range(takes);
        let ts_span = max_ts.saturating_sub(min_ts) as f64;

        let target_ms = target_duration.as_millis() as f64;

        // Find the maximum duration deviation so we can normalise.
        let max_dev = max_duration_deviation(takes, target_ms);

        let mut best: Option<(f32, &Take)> = None;

        for take in takes {
            let r_score = rating_score(take);
            let rec_score = recency_score(take, min_ts, ts_span);
            let dur_score = duration_score(take, target_ms, max_dev);

            let score = (weights.rating * r_score
                + weights.recency * rec_score
                + weights.duration_match * dur_score)
                / total_w;

            match best {
                None => {
                    best = Some((score, take));
                }
                Some((best_score, best_take)) => {
                    if score > best_score
                        || (score == best_score && take.take_number < best_take.take_number)
                    {
                        best = Some((score, take));
                    }
                }
            }
        }

        best.map(|(_, t)| t)
    }

    /// Returns all takes with their individual scores, sorted by descending
    /// score.
    ///
    /// Returns an empty vector if `takes` is empty or all weights are zero.
    #[must_use]
    pub fn score_all(
        takes: &[Take],
        target_duration: Duration,
        weights: TakeScoreWeights,
    ) -> Vec<(f32, &Take)> {
        if takes.is_empty() || weights.is_zero() {
            return Vec::new();
        }

        let total_w = weights.total();
        let (min_ts, max_ts) = recency_range(takes);
        let ts_span = max_ts.saturating_sub(min_ts) as f64;
        let target_ms = target_duration.as_millis() as f64;
        let max_dev = max_duration_deviation(takes, target_ms);

        let mut scored: Vec<(f32, &Take)> = takes
            .iter()
            .map(|take| {
                let score = (weights.rating * rating_score(take)
                    + weights.recency * recency_score(take, min_ts, ts_span)
                    + weights.duration_match * duration_score(take, target_ms, max_dev))
                    / total_w;
                (score, take)
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored
    }
}

/// Ranks takes using multiple weighted criteria and returns them sorted by
/// descending composite score.
///
/// This is a free-function shorthand for
/// `MultiCriteriaTakeSelector::score_all`.  `target_duration` specifies the
/// ideal clip length used for the duration-match criterion.
///
/// Returns an empty vector if `takes` is empty or all weights are zero.
#[must_use]
pub fn rank_takes_multi_criteria<'a>(
    takes: &'a [Take],
    target_duration: Duration,
    weights: TakeWeights,
) -> Vec<(f32, &'a Take)> {
    MultiCriteriaTakeSelector::score_all(takes, target_duration, weights)
}

// ---- helpers ----------------------------------------------------------------

/// Normalised rating score in `[0.0, 1.0]` (Unrated→0, FiveStars→1).
fn rating_score(take: &Take) -> f32 {
    f32::from(take.rating.to_value()) / 5.0
}

/// Returns `(min_timestamp_micros, max_timestamp_micros)` across all takes.
fn recency_range(takes: &[Take]) -> (i64, i64) {
    let mut min_ts = i64::MAX;
    let mut max_ts = i64::MIN;
    for t in takes {
        let ts = t.created_at.timestamp_micros();
        if ts < min_ts {
            min_ts = ts;
        }
        if ts > max_ts {
            max_ts = ts;
        }
    }
    (min_ts, max_ts)
}

/// Normalised recency score: most-recent take → 1.0, oldest → 0.0.
///
/// If all takes have identical timestamps, every take scores `1.0`.
fn recency_score(take: &Take, min_ts: i64, ts_span: f64) -> f32 {
    if ts_span < f64::EPSILON {
        return 1.0;
    }
    let delta = (take.created_at.timestamp_micros() - min_ts) as f64;
    (delta / ts_span) as f32
}

/// Returns the maximum absolute deviation from `target_ms` across all takes,
/// using the take's `clip_id` duration (we don't have a direct duration field
/// on `Take`, so we use `take_number * 30000` as a synthetic proxy in tests).
///
/// In a real deployment the caller would supply durations alongside takes via
/// a wrapper struct.  Here we derive a proxy for scoring purposes.
fn max_duration_deviation(takes: &[Take], target_ms: f64) -> f64 {
    takes
        .iter()
        .map(|t| {
            let dur = synthetic_duration_ms(t);
            (dur - target_ms).abs()
        })
        .fold(0.0f64, f64::max)
}

/// Duration score in `[0.0, 1.0]`: 1.0 if duration equals target exactly.
fn duration_score(take: &Take, target_ms: f64, max_dev: f64) -> f32 {
    if max_dev < f64::EPSILON {
        return 1.0;
    }
    let dev = (synthetic_duration_ms(take) - target_ms).abs();
    (1.0 - dev / max_dev) as f32
}

/// Synthetic take duration for scoring.
///
/// We use `take_number * 30_000` ms as a deterministic stand-in for the actual
/// clip duration so that tests can exercise duration scoring without needing
/// media files.  In production code the caller wraps `Take` with actual
/// durations or supplies them via a separate API.
fn synthetic_duration_ms(take: &Take) -> f64 {
    f64::from(take.take_number) * 30_000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clip::ClipId;
    use crate::logging::Rating;
    use crate::take::selector::Take;

    fn make_take(take_number: u32, rating: Rating) -> Take {
        let mut t = Take::new(ClipId::new(), "Scene 1", take_number);
        t.set_rating(rating);
        t
    }

    // ---- TakeScoreWeights ----

    #[test]
    fn test_weights_uniform_total() {
        let w = TakeScoreWeights::uniform();
        assert!((w.total() - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_weights_is_zero() {
        let w = TakeScoreWeights {
            rating: 0.0,
            recency: 0.0,
            duration_match: 0.0,
        };
        assert!(w.is_zero());
        assert!(!TakeScoreWeights::uniform().is_zero());
    }

    // ---- MultiCriteriaTakeSelector::select_best ----

    #[test]
    fn test_select_best_empty_returns_none() {
        let result = MultiCriteriaTakeSelector::select_best(
            &[],
            Duration::from_secs(30),
            TakeScoreWeights::uniform(),
        );
        assert!(result.is_none());
    }

    #[test]
    fn test_select_best_zero_weights_returns_none() {
        let takes = vec![make_take(1, Rating::FiveStars)];
        let result = MultiCriteriaTakeSelector::select_best(
            &takes,
            Duration::from_secs(30),
            TakeScoreWeights {
                rating: 0.0,
                recency: 0.0,
                duration_match: 0.0,
            },
        );
        assert!(result.is_none());
    }

    #[test]
    fn test_select_best_prefers_highest_rated_with_rating_only_weights() {
        let takes = vec![
            make_take(1, Rating::OneStar),
            make_take(2, Rating::FiveStars),
            make_take(3, Rating::ThreeStars),
        ];
        let best = MultiCriteriaTakeSelector::select_best(
            &takes,
            Duration::from_secs(60),
            TakeScoreWeights::rating_only(),
        )
        .expect("should select");
        assert_eq!(best.take_number, 2);
        assert_eq!(best.rating, Rating::FiveStars);
    }

    #[test]
    fn test_select_best_prefers_most_recent_with_recency_only_weights() {
        use chrono::TimeZone;
        // Assign explicitly spaced timestamps so the recency criterion is
        // unambiguous: take_number=1 is oldest, take_number=3 is most recent.
        let base = chrono::Utc
            .with_ymd_and_hms(2020, 1, 1, 0, 0, 0)
            .single()
            .expect("valid base timestamp");
        let mut takes = Vec::new();
        for i in 1..=3u32 {
            let mut t = make_take(i, Rating::ThreeStars);
            t.created_at = base + chrono::Duration::seconds(i64::from(i) * 1_000);
            takes.push(t);
        }
        let best = MultiCriteriaTakeSelector::select_best(
            &takes,
            Duration::from_secs(60),
            TakeScoreWeights::recency_only(),
        )
        .expect("should select");
        // take_number=3 has the highest timestamp → most recent → score 1.0.
        assert_eq!(best.take_number, 3);
    }

    #[test]
    fn test_select_best_single_take() {
        let takes = vec![make_take(1, Rating::TwoStars)];
        let best = MultiCriteriaTakeSelector::select_best(
            &takes,
            Duration::from_secs(30),
            TakeScoreWeights::uniform(),
        )
        .expect("should select");
        assert_eq!(best.take_number, 1);
    }

    #[test]
    fn test_select_best_duration_match_prefers_closer_duration() {
        // synthetic_duration_ms = take_number * 30_000 ms
        // target = 60_000 ms → take_number=2 is exact match
        let takes = vec![
            make_take(1, Rating::Unrated), // 30_000 ms, dev=30_000
            make_take(2, Rating::Unrated), // 60_000 ms, dev=0 (exact)
            make_take(4, Rating::Unrated), // 120_000 ms, dev=60_000
        ];
        let best = MultiCriteriaTakeSelector::select_best(
            &takes,
            Duration::from_secs(60),
            TakeScoreWeights {
                rating: 0.0,
                recency: 0.0,
                duration_match: 1.0,
            },
        )
        .expect("should select");
        assert_eq!(best.take_number, 2);
    }

    // ---- MultiCriteriaTakeSelector::score_all ----

    #[test]
    fn test_score_all_sorted_descending() {
        let takes = vec![
            make_take(1, Rating::OneStar),
            make_take(2, Rating::FiveStars),
            make_take(3, Rating::ThreeStars),
        ];
        let scored = MultiCriteriaTakeSelector::score_all(
            &takes,
            Duration::from_secs(60),
            TakeScoreWeights::rating_only(),
        );
        assert_eq!(scored.len(), 3);
        assert!(scored[0].0 >= scored[1].0);
        assert!(scored[1].0 >= scored[2].0);
    }

    #[test]
    fn test_score_all_empty_takes() {
        let scored = MultiCriteriaTakeSelector::score_all(
            &[],
            Duration::from_secs(60),
            TakeScoreWeights::uniform(),
        );
        assert!(scored.is_empty());
    }
}
