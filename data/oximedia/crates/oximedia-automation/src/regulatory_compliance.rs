//! Automated content rating insertion per jurisdiction.
//!
//! This module provides a [`ComplianceEngine`] that:
//!
//! - Stores jurisdiction-specific rating rules (which content ratings require
//!   which compliance actions and within which time windows).
//! - Evaluates a content item against the applicable rules for a target
//!   jurisdiction.
//! - Returns a [`ComplianceDirective`] describing what ratings/advisories must
//!   be inserted before and/or during the item.
//!
//! # Supported Jurisdictions
//!
//! | Code | Name | System |
//! |------|------|--------|
//! | `US` | United States | TV Parental Guidelines |
//! | `CA` | Canada | Canadian Ratings |
//! | `AU` | Australia | Australian Classification |
//! | `GB` | United Kingdom | BBFC / Ofcom |
//! | `EU` | European Union (generic) | Pan-European Game Information (PEGI-TV) |
//!
//! New jurisdictions can be registered at runtime via
//! [`ComplianceEngine::register_rule`].

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info};

// ─────────────────────────────────────────────────────────────────────────────
// Content rating
// ─────────────────────────────────────────────────────────────────────────────

/// A content rating from any jurisdiction's classification system.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ContentRating {
    /// ISO 3166-1 alpha-2 jurisdiction code (e.g. `"US"`, `"AU"`).
    pub jurisdiction: String,
    /// Rating label as used in the jurisdiction (e.g. `"TV-14"`, `"M"`, `"15"`).
    pub rating: String,
    /// Optional content descriptors (e.g. `["V", "L"]` for violence/language).
    pub descriptors: Vec<String>,
}

impl ContentRating {
    /// Create a new content rating.
    pub fn new(
        jurisdiction: impl Into<String>,
        rating: impl Into<String>,
        descriptors: Vec<&str>,
    ) -> Self {
        Self {
            jurisdiction: jurisdiction.into(),
            rating: rating.into(),
            descriptors: descriptors.into_iter().map(str::to_string).collect(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Compliance action
// ─────────────────────────────────────────────────────────────────────────────

/// An action the automation system must take to comply with broadcast
/// regulations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComplianceAction {
    /// Insert a rating bug/overlay at the start of the item and at regular
    /// intervals.  `interval_secs`: how often the rating should re-appear
    /// (0 = once at start only).
    InsertRatingBug { interval_secs: u32 },
    /// Play a verbal content advisory before the item.
    PlayVoiceAdvisory,
    /// Display a text advisory card before the item for `duration_secs`.
    DisplayTextAdvisory { duration_secs: u32 },
    /// Block the content from airing (schedule gap filler / alternative).
    BlockContent,
    /// Restrict to a specific broadcast time window (e.g. watershed hours).
    EnforceTimeWindow { start_hour: u8, end_hour: u8 },
}

// ─────────────────────────────────────────────────────────────────────────────
// Compliance rule
// ─────────────────────────────────────────────────────────────────────────────

/// A rule that maps a content rating to the required compliance actions for a
/// specific jurisdiction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceRule {
    /// Jurisdiction code.
    pub jurisdiction: String,
    /// The rating label this rule applies to (exact match).
    pub rating: String,
    /// Required compliance actions.
    pub actions: Vec<ComplianceAction>,
    /// Optional human-readable notes about the regulation.
    pub notes: Option<String>,
}

impl ComplianceRule {
    /// Create a new compliance rule.
    pub fn new(
        jurisdiction: impl Into<String>,
        rating: impl Into<String>,
        actions: Vec<ComplianceAction>,
    ) -> Self {
        Self {
            jurisdiction: jurisdiction.into(),
            rating: rating.into(),
            actions,
            notes: None,
        }
    }

    /// Attach explanatory notes.
    pub fn with_notes(mut self, notes: impl Into<String>) -> Self {
        self.notes = Some(notes.into());
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Compliance directive
// ─────────────────────────────────────────────────────────────────────────────

/// The result of evaluating a content item's rating against compliance rules.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceDirective {
    /// Jurisdiction the directive applies to.
    pub jurisdiction: String,
    /// The rating that was evaluated.
    pub rating: ContentRating,
    /// Required actions (empty = no special action required).
    pub actions: Vec<ComplianceAction>,
    /// Whether the content is cleared to air (no `BlockContent` action).
    pub cleared_to_air: bool,
}

impl ComplianceDirective {
    /// Returns `true` if `BlockContent` is in the actions list.
    pub fn is_blocked(&self) -> bool {
        self.actions
            .iter()
            .any(|a| *a == ComplianceAction::BlockContent)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Engine
// ─────────────────────────────────────────────────────────────────────────────

/// Automated content rating compliance engine.
///
/// Stores jurisdiction rules and evaluates content items against them.
#[derive(Debug, Default)]
pub struct ComplianceEngine {
    /// Rules keyed by `(jurisdiction, rating)`.
    rules: HashMap<(String, String), ComplianceRule>,
}

impl ComplianceEngine {
    /// Create an empty compliance engine.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an engine pre-loaded with built-in rules for common jurisdictions.
    pub fn with_defaults() -> Self {
        let mut engine = Self::new();

        // ── US TV Parental Guidelines ─────────────────────────────────────────
        engine.register_rule(ComplianceRule::new(
            "US",
            "TV-Y",
            vec![ComplianceAction::InsertRatingBug { interval_secs: 0 }],
        ));
        engine.register_rule(ComplianceRule::new(
            "US",
            "TV-G",
            vec![ComplianceAction::InsertRatingBug { interval_secs: 0 }],
        ));
        engine.register_rule(ComplianceRule::new(
            "US",
            "TV-PG",
            vec![
                ComplianceAction::InsertRatingBug { interval_secs: 0 },
                ComplianceAction::DisplayTextAdvisory { duration_secs: 5 },
            ],
        ));
        engine.register_rule(
            ComplianceRule::new(
                "US",
                "TV-14",
                vec![
                    ComplianceAction::InsertRatingBug { interval_secs: 0 },
                    ComplianceAction::PlayVoiceAdvisory,
                    ComplianceAction::DisplayTextAdvisory { duration_secs: 5 },
                    ComplianceAction::EnforceTimeWindow {
                        start_hour: 22,
                        end_hour: 6,
                    },
                ],
            )
            .with_notes("FCC Children's Television Act: restricted to late night"),
        );
        engine.register_rule(
            ComplianceRule::new(
                "US",
                "TV-MA",
                vec![
                    ComplianceAction::InsertRatingBug { interval_secs: 0 },
                    ComplianceAction::PlayVoiceAdvisory,
                    ComplianceAction::DisplayTextAdvisory { duration_secs: 8 },
                    ComplianceAction::EnforceTimeWindow {
                        start_hour: 22,
                        end_hour: 6,
                    },
                ],
            )
            .with_notes("Mature content: after watershed only"),
        );

        // ── UK Ofcom watershed ────────────────────────────────────────────────
        engine.register_rule(ComplianceRule::new(
            "GB",
            "PG",
            vec![ComplianceAction::InsertRatingBug { interval_secs: 0 }],
        ));
        engine.register_rule(ComplianceRule::new(
            "GB",
            "12",
            vec![
                ComplianceAction::InsertRatingBug { interval_secs: 0 },
                ComplianceAction::EnforceTimeWindow {
                    start_hour: 20,
                    end_hour: 6,
                },
            ],
        ));
        engine.register_rule(
            ComplianceRule::new(
                "GB",
                "15",
                vec![
                    ComplianceAction::InsertRatingBug { interval_secs: 0 },
                    ComplianceAction::EnforceTimeWindow {
                        start_hour: 21,
                        end_hour: 5,
                    },
                ],
            )
            .with_notes("Ofcom Broadcasting Code Section 1.4 watershed"),
        );
        engine.register_rule(ComplianceRule::new(
            "GB",
            "18",
            vec![
                ComplianceAction::InsertRatingBug { interval_secs: 0 },
                ComplianceAction::PlayVoiceAdvisory,
                ComplianceAction::EnforceTimeWindow {
                    start_hour: 22,
                    end_hour: 5,
                },
            ],
        ));

        // ── Australia ─────────────────────────────────────────────────────────
        engine.register_rule(ComplianceRule::new(
            "AU",
            "G",
            vec![ComplianceAction::InsertRatingBug { interval_secs: 0 }],
        ));
        engine.register_rule(ComplianceRule::new(
            "AU",
            "PG",
            vec![
                ComplianceAction::InsertRatingBug { interval_secs: 0 },
                ComplianceAction::DisplayTextAdvisory { duration_secs: 5 },
            ],
        ));
        engine.register_rule(ComplianceRule::new(
            "AU",
            "M",
            vec![
                ComplianceAction::InsertRatingBug { interval_secs: 0 },
                ComplianceAction::DisplayTextAdvisory { duration_secs: 5 },
                ComplianceAction::EnforceTimeWindow {
                    start_hour: 20,
                    end_hour: 6,
                },
            ],
        ));
        engine.register_rule(ComplianceRule::new(
            "AU",
            "MA15+",
            vec![
                ComplianceAction::InsertRatingBug { interval_secs: 0 },
                ComplianceAction::PlayVoiceAdvisory,
                ComplianceAction::EnforceTimeWindow {
                    start_hour: 21,
                    end_hour: 5,
                },
            ],
        ));
        engine.register_rule(
            ComplianceRule::new("AU", "R18+", vec![ComplianceAction::BlockContent])
                .with_notes("R18+ content is not permitted on free-to-air broadcast television"),
        );

        engine
    }

    // ── Rule management ───────────────────────────────────────────────────────

    /// Register or replace a compliance rule.
    pub fn register_rule(&mut self, rule: ComplianceRule) {
        info!(
            "Compliance rule registered: {}/{} → {} actions",
            rule.jurisdiction,
            rule.rating,
            rule.actions.len()
        );
        let key = (rule.jurisdiction.clone(), rule.rating.clone());
        self.rules.insert(key, rule);
    }

    /// Remove a rule.  Returns `true` if a rule was removed.
    pub fn remove_rule(&mut self, jurisdiction: &str, rating: &str) -> bool {
        self.rules
            .remove(&(jurisdiction.to_string(), rating.to_string()))
            .is_some()
    }

    /// Return the number of registered rules.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    // ── Evaluation ────────────────────────────────────────────────────────────

    /// Evaluate a content item's rating for a given jurisdiction.
    ///
    /// If no rule exists for the `(jurisdiction, rating)` combination, returns
    /// a directive with no actions (cleared to air with no special requirements).
    pub fn evaluate(&self, rating: &ContentRating) -> ComplianceDirective {
        let key = (rating.jurisdiction.clone(), rating.rating.clone());
        debug!(
            "Evaluating compliance: {}/{}",
            rating.jurisdiction, rating.rating
        );

        let actions = self
            .rules
            .get(&key)
            .map(|r| r.actions.clone())
            .unwrap_or_default();

        let cleared_to_air = !actions.contains(&ComplianceAction::BlockContent);

        ComplianceDirective {
            jurisdiction: rating.jurisdiction.clone(),
            rating: rating.clone(),
            actions,
            cleared_to_air,
        }
    }

    /// Evaluate for a list of jurisdictions, returning one directive per
    /// matching rule.  Useful for syndicated content that airs in multiple
    /// territories.
    pub fn evaluate_multi(&self, ratings: &[ContentRating]) -> Vec<ComplianceDirective> {
        ratings.iter().map(|r| self.evaluate(r)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_defaults_loaded() {
        let engine = ComplianceEngine::with_defaults();
        assert!(engine.rule_count() > 0);
    }

    #[test]
    fn test_us_tv_ma_requires_watershed() {
        let engine = ComplianceEngine::with_defaults();
        let rating = ContentRating::new("US", "TV-MA", vec!["V", "L"]);
        let directive = engine.evaluate(&rating);
        assert!(directive.cleared_to_air);
        assert!(directive.actions.iter().any(|a| matches!(
            a,
            ComplianceAction::EnforceTimeWindow { start_hour: 22, .. }
        )));
    }

    #[test]
    fn test_au_r18_blocked() {
        let engine = ComplianceEngine::with_defaults();
        let rating = ContentRating::new("AU", "R18+", vec![]);
        let directive = engine.evaluate(&rating);
        assert!(!directive.cleared_to_air);
        assert!(directive.is_blocked());
    }

    #[test]
    fn test_unknown_rating_cleared() {
        let engine = ComplianceEngine::with_defaults();
        // No rule for this rating
        let rating = ContentRating::new("ZZ", "CUSTOM", vec![]);
        let directive = engine.evaluate(&rating);
        assert!(directive.cleared_to_air);
        assert!(directive.actions.is_empty());
    }

    #[test]
    fn test_register_custom_rule() {
        let mut engine = ComplianceEngine::new();
        engine.register_rule(ComplianceRule::new(
            "XX",
            "ADULT",
            vec![ComplianceAction::BlockContent],
        ));
        let rating = ContentRating::new("XX", "ADULT", vec![]);
        let directive = engine.evaluate(&rating);
        assert!(!directive.cleared_to_air);
    }

    #[test]
    fn test_remove_rule() {
        let mut engine = ComplianceEngine::with_defaults();
        let before = engine.rule_count();
        assert!(engine.remove_rule("US", "TV-Y"));
        assert_eq!(engine.rule_count(), before - 1);
    }

    #[test]
    fn test_evaluate_multi() {
        let engine = ComplianceEngine::with_defaults();
        let ratings = vec![
            ContentRating::new("US", "TV-G", vec![]),
            ContentRating::new("AU", "G", vec![]),
        ];
        let directives = engine.evaluate_multi(&ratings);
        assert_eq!(directives.len(), 2);
        assert!(directives.iter().all(|d| d.cleared_to_air));
    }

    #[test]
    fn test_gb_15_watershed() {
        let engine = ComplianceEngine::with_defaults();
        let rating = ContentRating::new("GB", "15", vec![]);
        let directive = engine.evaluate(&rating);
        assert!(directive.cleared_to_air);
        assert!(directive.actions.iter().any(|a| matches!(
            a,
            ComplianceAction::EnforceTimeWindow { start_hour: 21, .. }
        )));
    }
}
