//! Constraint-based rules for broadcast playlist validation.
//!
//! Provides a `PlaylistRule` enum representing individual playlist constraints,
//! a `RuleEvaluator` that checks a proposed item against a rule, and a
//! `PlaylistConstraintSet` that aggregates multiple rules.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;

/// An individual playlist rule / constraint.
#[derive(Debug, Clone)]
pub enum PlaylistRule {
    /// No item may appear twice within the given window (count of items).
    NoRepeat {
        /// How many items back to check for repeats.
        window: usize,
    },
    /// Total playlist duration must not exceed `max_seconds`.
    MaxDuration {
        /// Maximum total duration in seconds.
        max_seconds: f64,
    },
    /// The same artist must not appear within `gap` items of the previous appearance.
    RequireArtistGap {
        /// Minimum number of items between the same artist.
        gap: usize,
    },
    /// No more than `limit` items from the same genre may appear consecutively.
    MaxSameGenre {
        /// Maximum number of consecutive same-genre items.
        limit: usize,
    },
    /// Each item must be at least `min_s` seconds long.
    MinItemDuration {
        /// Minimum item duration in seconds.
        min_s: f64,
    },
}

impl PlaylistRule {
    /// Return the canonical name of this rule.
    #[must_use]
    pub fn rule_name(&self) -> &'static str {
        match self {
            Self::NoRepeat { .. } => "no_repeat",
            Self::MaxDuration { .. } => "max_duration",
            Self::RequireArtistGap { .. } => "require_artist_gap",
            Self::MaxSameGenre { .. } => "max_same_genre",
            Self::MinItemDuration { .. } => "min_item_duration",
        }
    }

    /// Human-readable description of this rule.
    #[must_use]
    pub fn description(&self) -> String {
        match self {
            Self::NoRepeat { window } => {
                format!("No track may repeat within the last {window} items")
            }
            Self::MaxDuration { max_seconds } => {
                format!("Playlist total duration must not exceed {max_seconds:.0}s")
            }
            Self::RequireArtistGap { gap } => {
                format!("Same artist must be separated by at least {gap} items")
            }
            Self::MaxSameGenre { limit } => {
                format!("No more than {limit} consecutive items from the same genre")
            }
            Self::MinItemDuration { min_s } => {
                format!("Each item must be at least {min_s:.1}s long")
            }
        }
    }
}

/// Context provided to a rule evaluator when checking whether a candidate
/// item can be appended to the current playlist.
#[derive(Debug, Clone)]
pub struct EvalContext {
    /// IDs of items already in the playlist, in order.
    pub existing_ids: Vec<String>,
    /// Artist name for each item (by index, same length as existing_ids).
    pub existing_artists: Vec<String>,
    /// Genre for each item (by index).
    pub existing_genres: Vec<String>,
    /// Duration of each item in seconds (by index).
    pub existing_durations: Vec<f64>,
    /// Total accumulated duration so far.
    pub total_duration_s: f64,
}

impl EvalContext {
    /// Create a new empty evaluation context.
    #[must_use]
    pub fn new() -> Self {
        Self {
            existing_ids: Vec::new(),
            existing_artists: Vec::new(),
            existing_genres: Vec::new(),
            existing_durations: Vec::new(),
            total_duration_s: 0.0,
        }
    }

    /// Append a new item to the context.
    pub fn push(&mut self, id: &str, artist: &str, genre: &str, duration_s: f64) {
        self.existing_ids.push(id.to_owned());
        self.existing_artists.push(artist.to_owned());
        self.existing_genres.push(genre.to_owned());
        self.existing_durations.push(duration_s);
        self.total_duration_s += duration_s;
    }
}

impl Default for EvalContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a rule evaluation.
#[derive(Debug, Clone)]
pub struct RuleCheckResult {
    /// Whether the candidate passes this rule.
    pub passed: bool,
    /// Human-readable reason (set when `passed == false`).
    pub reason: String,
}

impl RuleCheckResult {
    fn ok() -> Self {
        Self {
            passed: true,
            reason: String::new(),
        }
    }

    fn fail(reason: impl Into<String>) -> Self {
        Self {
            passed: false,
            reason: reason.into(),
        }
    }
}

/// Evaluates a single `PlaylistRule` against a candidate item and context.
#[derive(Debug, Clone)]
pub struct RuleEvaluator;

impl RuleEvaluator {
    /// Check whether `candidate_id` (with the given metadata) is allowed
    /// to be appended to the playlist described by `ctx`.
    #[must_use]
    pub fn check(
        rule: &PlaylistRule,
        ctx: &EvalContext,
        candidate_id: &str,
        candidate_artist: &str,
        candidate_genre: &str,
        candidate_duration_s: f64,
    ) -> RuleCheckResult {
        match rule {
            PlaylistRule::NoRepeat { window } => {
                let recent = ctx.existing_ids.iter().rev().take(*window);
                for id in recent {
                    if id == candidate_id {
                        return RuleCheckResult::fail(format!(
                            "Track '{candidate_id}' appeared within the last {window} items"
                        ));
                    }
                }
                RuleCheckResult::ok()
            }
            PlaylistRule::MaxDuration { max_seconds } => {
                if ctx.total_duration_s + candidate_duration_s > *max_seconds {
                    RuleCheckResult::fail(format!(
                        "Adding {candidate_duration_s:.0}s would exceed max duration {max_seconds:.0}s"
                    ))
                } else {
                    RuleCheckResult::ok()
                }
            }
            PlaylistRule::RequireArtistGap { gap } => {
                let recent: Vec<_> = ctx.existing_artists.iter().rev().take(*gap).collect();
                if recent.iter().any(|a| a.as_str() == candidate_artist) {
                    RuleCheckResult::fail(format!(
                        "Artist '{candidate_artist}' appeared within the last {gap} items"
                    ))
                } else {
                    RuleCheckResult::ok()
                }
            }
            PlaylistRule::MaxSameGenre { limit } => {
                let consecutive = ctx
                    .existing_genres
                    .iter()
                    .rev()
                    .take_while(|g| g.as_str() == candidate_genre)
                    .count();
                if consecutive >= *limit {
                    RuleCheckResult::fail(format!(
                        "Already have {consecutive} consecutive items in genre '{candidate_genre}'"
                    ))
                } else {
                    RuleCheckResult::ok()
                }
            }
            PlaylistRule::MinItemDuration { min_s } => {
                if candidate_duration_s < *min_s {
                    RuleCheckResult::fail(format!(
                        "Item duration {candidate_duration_s:.1}s is shorter than minimum {min_s:.1}s"
                    ))
                } else {
                    RuleCheckResult::ok()
                }
            }
        }
    }
}

/// A collection of playlist rules that must all pass for an item to be added.
#[derive(Debug, Clone, Default)]
pub struct PlaylistConstraintSet {
    rules: Vec<PlaylistRule>,
}

impl PlaylistConstraintSet {
    /// Create an empty constraint set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a rule to the constraint set.
    pub fn add_rule(&mut self, rule: PlaylistRule) {
        self.rules.push(rule);
    }

    /// Return the number of rules.
    #[must_use]
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Validate a candidate item against all rules in the set.
    ///
    /// Returns a map from rule name to `RuleCheckResult`.
    #[must_use]
    pub fn validate(
        &self,
        ctx: &EvalContext,
        candidate_id: &str,
        candidate_artist: &str,
        candidate_genre: &str,
        candidate_duration_s: f64,
    ) -> HashMap<String, RuleCheckResult> {
        self.rules
            .iter()
            .map(|rule| {
                let result = RuleEvaluator::check(
                    rule,
                    ctx,
                    candidate_id,
                    candidate_artist,
                    candidate_genre,
                    candidate_duration_s,
                );
                (rule.rule_name().to_owned(), result)
            })
            .collect()
    }

    /// Returns `true` if all rules pass for the given candidate.
    #[must_use]
    pub fn all_pass(
        &self,
        ctx: &EvalContext,
        candidate_id: &str,
        candidate_artist: &str,
        candidate_genre: &str,
        candidate_duration_s: f64,
    ) -> bool {
        self.rules.iter().all(|rule| {
            RuleEvaluator::check(
                rule,
                ctx,
                candidate_id,
                candidate_artist,
                candidate_genre,
                candidate_duration_s,
            )
            .passed
        })
    }

    /// Return names of all rules currently registered.
    #[must_use]
    pub fn rule_names(&self) -> Vec<&'static str> {
        self.rules.iter().map(PlaylistRule::rule_name).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_ctx() -> EvalContext {
        EvalContext::new()
    }

    fn ctx_with_tracks() -> EvalContext {
        let mut ctx = EvalContext::new();
        ctx.push("track_a", "Artist1", "pop", 210.0);
        ctx.push("track_b", "Artist2", "rock", 180.0);
        ctx.push("track_c", "Artist1", "pop", 195.0);
        ctx
    }

    // PlaylistRule tests

    #[test]
    fn test_rule_names_non_empty() {
        let rules = [
            PlaylistRule::NoRepeat { window: 5 },
            PlaylistRule::MaxDuration {
                max_seconds: 3600.0,
            },
            PlaylistRule::RequireArtistGap { gap: 3 },
            PlaylistRule::MaxSameGenre { limit: 2 },
            PlaylistRule::MinItemDuration { min_s: 30.0 },
        ];
        for r in &rules {
            assert!(!r.rule_name().is_empty());
            assert!(!r.description().is_empty());
        }
    }

    #[test]
    fn test_no_repeat_passes_new_track() {
        let rule = PlaylistRule::NoRepeat { window: 5 };
        let ctx = ctx_with_tracks();
        let r = RuleEvaluator::check(&rule, &ctx, "track_d", "Artist3", "jazz", 200.0);
        assert!(r.passed);
    }

    #[test]
    fn test_no_repeat_fails_recent_track() {
        let rule = PlaylistRule::NoRepeat { window: 5 };
        let ctx = ctx_with_tracks();
        let r = RuleEvaluator::check(&rule, &ctx, "track_a", "Artist1", "pop", 200.0);
        assert!(!r.passed);
        assert!(!r.reason.is_empty());
    }

    #[test]
    fn test_max_duration_passes_within_limit() {
        let rule = PlaylistRule::MaxDuration {
            max_seconds: 1000.0,
        };
        let ctx = ctx_with_tracks(); // total ~585s
        let r = RuleEvaluator::check(&rule, &ctx, "new", "X", "pop", 200.0);
        assert!(r.passed);
    }

    #[test]
    fn test_max_duration_fails_over_limit() {
        let rule = PlaylistRule::MaxDuration { max_seconds: 600.0 };
        let ctx = ctx_with_tracks(); // total ~585s
        let r = RuleEvaluator::check(&rule, &ctx, "new", "X", "pop", 200.0);
        assert!(!r.passed);
    }

    #[test]
    fn test_artist_gap_fails_when_recent() {
        let rule = PlaylistRule::RequireArtistGap { gap: 5 };
        let ctx = ctx_with_tracks(); // Artist1 is at positions 0 and 2
        let r = RuleEvaluator::check(&rule, &ctx, "track_x", "Artist1", "pop", 200.0);
        assert!(!r.passed);
    }

    #[test]
    fn test_artist_gap_passes_unknown_artist() {
        let rule = PlaylistRule::RequireArtistGap { gap: 5 };
        let ctx = ctx_with_tracks();
        let r = RuleEvaluator::check(&rule, &ctx, "track_x", "NewArtist", "pop", 200.0);
        assert!(r.passed);
    }

    #[test]
    fn test_max_same_genre_fails_on_consecutive() {
        let rule = PlaylistRule::MaxSameGenre { limit: 2 };
        let mut ctx = EvalContext::new();
        ctx.push("a", "Artist1", "pop", 200.0);
        ctx.push("b", "Artist2", "pop", 200.0);
        let r = RuleEvaluator::check(&rule, &ctx, "c", "Artist3", "pop", 200.0);
        assert!(!r.passed);
    }

    #[test]
    fn test_min_item_duration_fails_short() {
        let rule = PlaylistRule::MinItemDuration { min_s: 60.0 };
        let ctx = empty_ctx();
        let r = RuleEvaluator::check(&rule, &ctx, "jingle", "X", "advert", 15.0);
        assert!(!r.passed);
    }

    #[test]
    fn test_min_item_duration_passes_long_enough() {
        let rule = PlaylistRule::MinItemDuration { min_s: 60.0 };
        let ctx = empty_ctx();
        let r = RuleEvaluator::check(&rule, &ctx, "track", "X", "pop", 180.0);
        assert!(r.passed);
    }

    // PlaylistConstraintSet tests

    #[test]
    fn test_constraint_set_empty_all_pass() {
        let cs = PlaylistConstraintSet::new();
        let ctx = ctx_with_tracks();
        assert!(cs.all_pass(&ctx, "new", "NewArtist", "jazz", 200.0));
    }

    #[test]
    fn test_constraint_set_add_rule_count() {
        let mut cs = PlaylistConstraintSet::new();
        cs.add_rule(PlaylistRule::NoRepeat { window: 5 });
        cs.add_rule(PlaylistRule::MaxDuration {
            max_seconds: 3600.0,
        });
        assert_eq!(cs.rule_count(), 2);
    }

    #[test]
    fn test_validate_returns_all_rules() {
        let mut cs = PlaylistConstraintSet::new();
        cs.add_rule(PlaylistRule::NoRepeat { window: 5 });
        cs.add_rule(PlaylistRule::MinItemDuration { min_s: 30.0 });
        let ctx = empty_ctx();
        let results = cs.validate(&ctx, "t", "A", "pop", 180.0);
        assert_eq!(results.len(), 2);
    }
}
