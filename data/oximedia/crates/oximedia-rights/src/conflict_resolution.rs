//! Automatic rights conflict resolution suggestions.
//!
//! When overlapping or conflicting rights are detected, this module generates
//! actionable resolution suggestions ranked by feasibility, cost, and urgency.
//! It builds on `rights_conflict::ConflictType` and provides a higher-level
//! strategy engine for conflict triage and automated recommendation.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;

use crate::rights_conflict::{ConflictType, RightsConflict};

// ── ResolutionStrategy ──────────────────────────────────────────────────────

/// A recommended strategy for resolving a rights conflict.
#[derive(Debug, Clone, PartialEq)]
pub enum ResolutionStrategy {
    /// Revoke the newer / lower-priority license.
    RevokeLicense {
        /// ID of the license to revoke.
        license_id: String,
    },
    /// Narrow the territory of one or both licenses to eliminate overlap.
    NarrowTerritory {
        /// ID of the license to narrow.
        license_id: String,
        /// Territories to remove from the license.
        remove_territories: Vec<String>,
    },
    /// Adjust the time windows so the licenses become sequential, not
    /// overlapping.
    AdjustTimeWindow {
        /// ID of the license to adjust.
        license_id: String,
        /// Suggested new start epoch.
        new_start_ms: u64,
        /// Suggested new end epoch.
        new_end_ms: Option<u64>,
    },
    /// Convert the exclusive license to non-exclusive.
    ConvertToNonExclusive {
        /// ID of the license to convert.
        license_id: String,
    },
    /// Request a renegotiation between the parties.
    Renegotiate {
        /// IDs of the involved parties.
        party_ids: Vec<String>,
        /// Suggested negotiation topic.
        topic: String,
    },
    /// Issue a payment to settle the conflict.
    SettleWithPayment {
        /// ID of the payee.
        payee_id: String,
        /// Suggested amount in the agreement currency.
        suggested_amount: f64,
        /// Reason for the payment.
        reason: String,
    },
    /// Escalate to legal review.
    LegalReview {
        /// Brief for the legal team.
        brief: String,
    },
    /// Immediately cease usage of the asset.
    CeaseUsage {
        /// ID of the asset.
        asset_id: String,
        /// Reason for cessation.
        reason: String,
    },
}

impl ResolutionStrategy {
    /// Human-readable label for this strategy category.
    pub fn label(&self) -> &str {
        match self {
            Self::RevokeLicense { .. } => "Revoke License",
            Self::NarrowTerritory { .. } => "Narrow Territory",
            Self::AdjustTimeWindow { .. } => "Adjust Time Window",
            Self::ConvertToNonExclusive { .. } => "Convert to Non-Exclusive",
            Self::Renegotiate { .. } => "Renegotiate",
            Self::SettleWithPayment { .. } => "Settle with Payment",
            Self::LegalReview { .. } => "Legal Review",
            Self::CeaseUsage { .. } => "Cease Usage",
        }
    }

    /// Estimated complexity on a 1-5 scale (5 = most complex/costly).
    pub fn complexity(&self) -> u8 {
        match self {
            Self::RevokeLicense { .. } => 2,
            Self::NarrowTerritory { .. } => 3,
            Self::AdjustTimeWindow { .. } => 2,
            Self::ConvertToNonExclusive { .. } => 3,
            Self::Renegotiate { .. } => 4,
            Self::SettleWithPayment { .. } => 3,
            Self::LegalReview { .. } => 5,
            Self::CeaseUsage { .. } => 1,
        }
    }
}

// ── ResolutionSuggestion ────────────────────────────────────────────────────

/// A ranked suggestion for resolving a specific conflict.
#[derive(Debug, Clone)]
pub struct ResolutionSuggestion {
    /// The conflict this suggestion addresses.
    pub conflict_id: String,
    /// The recommended strategy.
    pub strategy: ResolutionStrategy,
    /// Confidence score from 0.0 (wild guess) to 1.0 (very confident).
    pub confidence: f64,
    /// Urgency score from 0.0 (can wait) to 1.0 (immediate action needed).
    pub urgency: f64,
    /// Human-readable explanation of why this strategy is suggested.
    pub rationale: String,
}

impl ResolutionSuggestion {
    /// Composite priority score (higher = more urgent and confident).
    pub fn priority_score(&self) -> f64 {
        self.confidence * 0.4 + self.urgency * 0.6
    }
}

// ── ConflictResolutionEngine ────────────────────────────────────────────────

/// Engine that analyzes `RightsConflict` instances and produces ranked
/// `ResolutionSuggestion` lists.
#[derive(Debug, Default)]
pub struct ConflictResolutionEngine {
    /// Custom weights per conflict type (defaults are built-in).
    urgency_weights: HashMap<String, f64>,
}

impl ConflictResolutionEngine {
    /// Create a new engine with default weights.
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the urgency weight for a conflict type label.
    pub fn set_urgency_weight(&mut self, conflict_type_label: &str, weight: f64) {
        self.urgency_weights
            .insert(conflict_type_label.to_string(), weight.clamp(0.0, 1.0));
    }

    /// Base urgency for a conflict type (normalised 0-1 from severity).
    fn base_urgency(conflict_type: &ConflictType) -> f64 {
        let severity = conflict_type.severity();
        f64::from(severity) / 10.0
    }

    /// Generate resolution suggestions for a single conflict.
    ///
    /// Returns suggestions sorted by descending `priority_score`.
    pub fn suggest(&self, conflict: &RightsConflict) -> Vec<ResolutionSuggestion> {
        let base_urgency = self
            .urgency_weights
            .get(&format!("{:?}", conflict.conflict_type))
            .copied()
            .unwrap_or_else(|| Self::base_urgency(&conflict.conflict_type));

        let mut suggestions = match &conflict.conflict_type {
            ConflictType::OverlappingExclusive => {
                self.suggest_overlapping_exclusive(conflict, base_urgency)
            }
            ConflictType::ExpiredLicense => self.suggest_expired_license(conflict, base_urgency),
            ConflictType::TerritoryBreach => self.suggest_territory_breach(conflict, base_urgency),
            ConflictType::ScopeBreach => self.suggest_scope_breach(conflict, base_urgency),
            ConflictType::UnauthorizedSublicense => {
                self.suggest_unauthorized_sublicense(conflict, base_urgency)
            }
            ConflictType::RoyaltyDefault => self.suggest_royalty_default(conflict, base_urgency),
        };

        suggestions.sort_by(|a, b| {
            b.priority_score()
                .partial_cmp(&a.priority_score())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        suggestions
    }

    /// Generate suggestions for all unresolved conflicts, merged and sorted.
    pub fn suggest_all(&self, conflicts: &[RightsConflict]) -> Vec<ResolutionSuggestion> {
        let mut all: Vec<ResolutionSuggestion> = conflicts
            .iter()
            .filter(|c| !c.is_resolved())
            .flat_map(|c| self.suggest(c))
            .collect();

        all.sort_by(|a, b| {
            b.priority_score()
                .partial_cmp(&a.priority_score())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        all
    }

    // ── Per-type suggestion generators ──────────────────────────────────────

    fn suggest_overlapping_exclusive(
        &self,
        conflict: &RightsConflict,
        urgency: f64,
    ) -> Vec<ResolutionSuggestion> {
        let mut out = Vec::new();

        // Primary: convert one license to non-exclusive
        if let Some(first_id) = conflict.involved_ids.first() {
            out.push(ResolutionSuggestion {
                conflict_id: conflict.id.clone(),
                strategy: ResolutionStrategy::ConvertToNonExclusive {
                    license_id: first_id.clone(),
                },
                confidence: 0.7,
                urgency,
                rationale: "Converting one exclusive license to non-exclusive eliminates the \
                    overlap while preserving both parties' rights."
                    .to_string(),
            });
        }

        // Secondary: adjust time windows
        if conflict.involved_ids.len() >= 2 {
            out.push(ResolutionSuggestion {
                conflict_id: conflict.id.clone(),
                strategy: ResolutionStrategy::AdjustTimeWindow {
                    license_id: conflict.involved_ids[1].clone(),
                    new_start_ms: 0,
                    new_end_ms: None,
                },
                confidence: 0.6,
                urgency,
                rationale: "Adjusting time windows to make the licenses sequential removes \
                    the temporal overlap."
                    .to_string(),
            });
        }

        // Tertiary: renegotiate
        out.push(ResolutionSuggestion {
            conflict_id: conflict.id.clone(),
            strategy: ResolutionStrategy::Renegotiate {
                party_ids: conflict.involved_ids.clone(),
                topic: "Overlapping exclusive territorial rights".to_string(),
            },
            confidence: 0.5,
            urgency: urgency * 0.8,
            rationale: "Renegotiation may yield a mutually acceptable territory split.".to_string(),
        });

        out
    }

    fn suggest_expired_license(
        &self,
        conflict: &RightsConflict,
        urgency: f64,
    ) -> Vec<ResolutionSuggestion> {
        let mut out = Vec::new();

        // Primary: cease usage immediately
        if let Some(first_id) = conflict.involved_ids.first() {
            out.push(ResolutionSuggestion {
                conflict_id: conflict.id.clone(),
                strategy: ResolutionStrategy::CeaseUsage {
                    asset_id: first_id.clone(),
                    reason: "License has expired; continued usage is unauthorized.".to_string(),
                },
                confidence: 0.9,
                urgency,
                rationale: "The most straightforward resolution: stop using the asset until \
                    the license is renewed."
                    .to_string(),
            });
        }

        // Secondary: renegotiate renewal
        out.push(ResolutionSuggestion {
            conflict_id: conflict.id.clone(),
            strategy: ResolutionStrategy::Renegotiate {
                party_ids: conflict.involved_ids.clone(),
                topic: "License renewal".to_string(),
            },
            confidence: 0.7,
            urgency: urgency * 0.7,
            rationale: "Renewing the license lets usage continue legally.".to_string(),
        });

        out
    }

    fn suggest_territory_breach(
        &self,
        conflict: &RightsConflict,
        urgency: f64,
    ) -> Vec<ResolutionSuggestion> {
        let mut out = Vec::new();

        // Primary: narrow territory
        if let Some(first_id) = conflict.involved_ids.first() {
            out.push(ResolutionSuggestion {
                conflict_id: conflict.id.clone(),
                strategy: ResolutionStrategy::NarrowTerritory {
                    license_id: first_id.clone(),
                    remove_territories: vec!["BREACHED_TERRITORY".to_string()],
                },
                confidence: 0.8,
                urgency,
                rationale: "Removing the breached territory from the license scope prevents \
                    further violations."
                    .to_string(),
            });
        }

        // Secondary: legal review
        out.push(ResolutionSuggestion {
            conflict_id: conflict.id.clone(),
            strategy: ResolutionStrategy::LegalReview {
                brief: format!("Territory breach detected: {}", conflict.description),
            },
            confidence: 0.6,
            urgency: urgency * 0.9,
            rationale: "Territory breaches may have legal consequences that require \
                professional review."
                .to_string(),
        });

        out
    }

    fn suggest_scope_breach(
        &self,
        conflict: &RightsConflict,
        urgency: f64,
    ) -> Vec<ResolutionSuggestion> {
        let mut out = Vec::new();

        // Primary: cease usage
        if let Some(first_id) = conflict.involved_ids.first() {
            out.push(ResolutionSuggestion {
                conflict_id: conflict.id.clone(),
                strategy: ResolutionStrategy::CeaseUsage {
                    asset_id: first_id.clone(),
                    reason: "Usage exceeds permitted scope.".to_string(),
                },
                confidence: 0.85,
                urgency,
                rationale: "Stop the out-of-scope usage to prevent further liability.".to_string(),
            });
        }

        // Secondary: renegotiate to expand scope
        out.push(ResolutionSuggestion {
            conflict_id: conflict.id.clone(),
            strategy: ResolutionStrategy::Renegotiate {
                party_ids: conflict.involved_ids.clone(),
                topic: "Expand usage scope".to_string(),
            },
            confidence: 0.6,
            urgency: urgency * 0.7,
            rationale: "Renegotiating the license to include the desired scope is the \
                long-term fix."
                .to_string(),
        });

        out
    }

    fn suggest_unauthorized_sublicense(
        &self,
        conflict: &RightsConflict,
        urgency: f64,
    ) -> Vec<ResolutionSuggestion> {
        let mut out = Vec::new();

        // Primary: revoke the sublicense
        if let Some(first_id) = conflict.involved_ids.first() {
            out.push(ResolutionSuggestion {
                conflict_id: conflict.id.clone(),
                strategy: ResolutionStrategy::RevokeLicense {
                    license_id: first_id.clone(),
                },
                confidence: 0.9,
                urgency,
                rationale: "An unauthorized sublicense must be revoked immediately.".to_string(),
            });
        }

        // Secondary: legal review
        out.push(ResolutionSuggestion {
            conflict_id: conflict.id.clone(),
            strategy: ResolutionStrategy::LegalReview {
                brief: "Unauthorized sublicense granted; assess liability exposure.".to_string(),
            },
            confidence: 0.8,
            urgency,
            rationale: "Legal review is warranted to assess damages and next steps.".to_string(),
        });

        out
    }

    fn suggest_royalty_default(
        &self,
        conflict: &RightsConflict,
        urgency: f64,
    ) -> Vec<ResolutionSuggestion> {
        let mut out = Vec::new();

        // Primary: settle with payment
        if let Some(payee_id) = conflict.involved_ids.first() {
            out.push(ResolutionSuggestion {
                conflict_id: conflict.id.clone(),
                strategy: ResolutionStrategy::SettleWithPayment {
                    payee_id: payee_id.clone(),
                    suggested_amount: 0.0, // caller must set actual amount
                    reason: "Overdue royalty payment".to_string(),
                },
                confidence: 0.85,
                urgency,
                rationale: "Paying the overdue royalty is the simplest resolution.".to_string(),
            });
        }

        // Secondary: renegotiate terms
        out.push(ResolutionSuggestion {
            conflict_id: conflict.id.clone(),
            strategy: ResolutionStrategy::Renegotiate {
                party_ids: conflict.involved_ids.clone(),
                topic: "Restructure royalty payment schedule".to_string(),
            },
            confidence: 0.5,
            urgency: urgency * 0.6,
            rationale: "If payment is untenable, restructuring the schedule may help.".to_string(),
        });

        out
    }
}

// ── ConflictTriageReport ────────────────────────────────────────────────────

/// Aggregated triage report for a set of conflicts.
#[derive(Debug)]
pub struct ConflictTriageReport {
    /// Total conflicts analyzed.
    pub total_conflicts: usize,
    /// Number of critical conflicts (severity >= 8).
    pub critical_count: usize,
    /// Number of non-critical conflicts.
    pub non_critical_count: usize,
    /// Sorted suggestions (highest priority first).
    pub suggestions: Vec<ResolutionSuggestion>,
    /// Breakdown of conflict count per type.
    pub type_breakdown: HashMap<String, usize>,
}

impl ConflictTriageReport {
    /// Generate a triage report from a slice of conflicts.
    pub fn from_conflicts(engine: &ConflictResolutionEngine, conflicts: &[RightsConflict]) -> Self {
        let unresolved: Vec<&RightsConflict> =
            conflicts.iter().filter(|c| !c.is_resolved()).collect();

        let total_conflicts = unresolved.len();
        let critical_count = unresolved
            .iter()
            .filter(|c| c.conflict_type.is_critical())
            .count();
        let non_critical_count = total_conflicts - critical_count;

        let mut type_breakdown = HashMap::new();
        for c in &unresolved {
            let key = format!("{:?}", c.conflict_type);
            *type_breakdown.entry(key).or_insert(0) += 1;
        }

        let suggestions = engine.suggest_all(conflicts);

        Self {
            total_conflicts,
            critical_count,
            non_critical_count,
            suggestions,
            type_breakdown,
        }
    }

    /// Returns `true` if there are zero unresolved critical conflicts.
    pub fn is_safe(&self) -> bool {
        self.critical_count == 0
    }

    /// Top N suggestions by priority.
    pub fn top_suggestions(&self, n: usize) -> &[ResolutionSuggestion] {
        let end = n.min(self.suggestions.len());
        &self.suggestions[..end]
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_conflict(id: &str, kind: ConflictType) -> RightsConflict {
        RightsConflict::new(
            id,
            kind,
            "Test conflict",
            vec!["rights-A".to_string(), "rights-B".to_string()],
        )
    }

    // ── ResolutionStrategy ──────────────────────────────────────────────────

    #[test]
    fn test_strategy_label_revoke() {
        let s = ResolutionStrategy::RevokeLicense {
            license_id: "L1".to_string(),
        };
        assert_eq!(s.label(), "Revoke License");
    }

    #[test]
    fn test_strategy_label_narrow_territory() {
        let s = ResolutionStrategy::NarrowTerritory {
            license_id: "L1".to_string(),
            remove_territories: vec![],
        };
        assert_eq!(s.label(), "Narrow Territory");
    }

    #[test]
    fn test_strategy_complexity_revoke() {
        let s = ResolutionStrategy::RevokeLicense {
            license_id: "L1".to_string(),
        };
        assert_eq!(s.complexity(), 2);
    }

    #[test]
    fn test_strategy_complexity_legal_review() {
        let s = ResolutionStrategy::LegalReview {
            brief: "test".to_string(),
        };
        assert_eq!(s.complexity(), 5);
    }

    #[test]
    fn test_strategy_complexity_cease_usage() {
        let s = ResolutionStrategy::CeaseUsage {
            asset_id: "a1".to_string(),
            reason: "test".to_string(),
        };
        assert_eq!(s.complexity(), 1);
    }

    // ── ResolutionSuggestion ────────────────────────────────────────────────

    #[test]
    fn test_suggestion_priority_score() {
        let s = ResolutionSuggestion {
            conflict_id: "c1".to_string(),
            strategy: ResolutionStrategy::CeaseUsage {
                asset_id: "a".to_string(),
                reason: "r".to_string(),
            },
            confidence: 1.0,
            urgency: 1.0,
            rationale: "test".to_string(),
        };
        assert!((s.priority_score() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_suggestion_priority_score_weighted() {
        let s = ResolutionSuggestion {
            conflict_id: "c1".to_string(),
            strategy: ResolutionStrategy::CeaseUsage {
                asset_id: "a".to_string(),
                reason: "r".to_string(),
            },
            confidence: 0.5,
            urgency: 0.5,
            rationale: "test".to_string(),
        };
        // 0.5*0.4 + 0.5*0.6 = 0.5
        assert!((s.priority_score() - 0.5).abs() < 1e-9);
    }

    // ── ConflictResolutionEngine ────────────────────────────────────────────

    #[test]
    fn test_engine_suggest_overlapping_exclusive() {
        let engine = ConflictResolutionEngine::new();
        let conflict = make_conflict("c1", ConflictType::OverlappingExclusive);
        let suggestions = engine.suggest(&conflict);
        assert!(suggestions.len() >= 2);
        // First suggestion should be highest priority
        assert!(suggestions[0].priority_score() >= suggestions[1].priority_score());
    }

    #[test]
    fn test_engine_suggest_expired_license() {
        let engine = ConflictResolutionEngine::new();
        let conflict = make_conflict("c2", ConflictType::ExpiredLicense);
        let suggestions = engine.suggest(&conflict);
        assert!(!suggestions.is_empty());
        // Primary should be CeaseUsage
        assert_eq!(suggestions[0].strategy.label(), "Cease Usage");
    }

    #[test]
    fn test_engine_suggest_territory_breach() {
        let engine = ConflictResolutionEngine::new();
        let conflict = make_conflict("c3", ConflictType::TerritoryBreach);
        let suggestions = engine.suggest(&conflict);
        assert!(!suggestions.is_empty());
        assert_eq!(suggestions[0].strategy.label(), "Narrow Territory");
    }

    #[test]
    fn test_engine_suggest_scope_breach() {
        let engine = ConflictResolutionEngine::new();
        let conflict = make_conflict("c4", ConflictType::ScopeBreach);
        let suggestions = engine.suggest(&conflict);
        assert!(!suggestions.is_empty());
        assert_eq!(suggestions[0].strategy.label(), "Cease Usage");
    }

    #[test]
    fn test_engine_suggest_unauthorized_sublicense() {
        let engine = ConflictResolutionEngine::new();
        let conflict = make_conflict("c5", ConflictType::UnauthorizedSublicense);
        let suggestions = engine.suggest(&conflict);
        assert!(suggestions.len() >= 2);
        assert_eq!(suggestions[0].strategy.label(), "Revoke License");
    }

    #[test]
    fn test_engine_suggest_royalty_default() {
        let engine = ConflictResolutionEngine::new();
        let conflict = make_conflict("c6", ConflictType::RoyaltyDefault);
        let suggestions = engine.suggest(&conflict);
        assert!(!suggestions.is_empty());
        assert_eq!(suggestions[0].strategy.label(), "Settle with Payment");
    }

    #[test]
    fn test_engine_suggest_all_skips_resolved() {
        let engine = ConflictResolutionEngine::new();
        let mut c1 = make_conflict("c1", ConflictType::ExpiredLicense);
        c1.mark_resolved();
        let c2 = make_conflict("c2", ConflictType::RoyaltyDefault);
        let suggestions = engine.suggest_all(&[c1, c2]);
        // Only c2 should produce suggestions
        assert!(suggestions.iter().all(|s| s.conflict_id == "c2"));
    }

    #[test]
    fn test_engine_suggest_all_sorted_by_priority() {
        let engine = ConflictResolutionEngine::new();
        let c1 = make_conflict("c1", ConflictType::RoyaltyDefault);
        let c2 = make_conflict("c2", ConflictType::UnauthorizedSublicense);
        let suggestions = engine.suggest_all(&[c1, c2]);
        for w in suggestions.windows(2) {
            assert!(w[0].priority_score() >= w[1].priority_score());
        }
    }

    #[test]
    fn test_engine_custom_urgency_weight() {
        let mut engine = ConflictResolutionEngine::new();
        engine.set_urgency_weight("RoyaltyDefault", 1.0);
        let conflict = make_conflict("c1", ConflictType::RoyaltyDefault);
        let suggestions = engine.suggest(&conflict);
        assert!(!suggestions.is_empty());
        assert!((suggestions[0].urgency - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_engine_urgency_weight_clamped() {
        let mut engine = ConflictResolutionEngine::new();
        engine.set_urgency_weight("RoyaltyDefault", 5.0);
        let conflict = make_conflict("c1", ConflictType::RoyaltyDefault);
        let suggestions = engine.suggest(&conflict);
        assert!(!suggestions.is_empty());
        assert!((suggestions[0].urgency - 1.0).abs() < 1e-9);
    }

    // ── ConflictTriageReport ────────────────────────────────────────────────

    #[test]
    fn test_triage_report_counts() {
        let engine = ConflictResolutionEngine::new();
        let conflicts = vec![
            make_conflict("c1", ConflictType::OverlappingExclusive), // critical
            make_conflict("c2", ConflictType::RoyaltyDefault),       // non-critical
            make_conflict("c3", ConflictType::TerritoryBreach),      // critical
        ];
        let report = ConflictTriageReport::from_conflicts(&engine, &conflicts);
        assert_eq!(report.total_conflicts, 3);
        assert_eq!(report.critical_count, 2);
        assert_eq!(report.non_critical_count, 1);
    }

    #[test]
    fn test_triage_report_is_safe_false() {
        let engine = ConflictResolutionEngine::new();
        let conflicts = vec![make_conflict("c1", ConflictType::OverlappingExclusive)];
        let report = ConflictTriageReport::from_conflicts(&engine, &conflicts);
        assert!(!report.is_safe());
    }

    #[test]
    fn test_triage_report_is_safe_true() {
        let engine = ConflictResolutionEngine::new();
        let conflicts = vec![make_conflict("c1", ConflictType::RoyaltyDefault)];
        let report = ConflictTriageReport::from_conflicts(&engine, &conflicts);
        assert!(report.is_safe());
    }

    #[test]
    fn test_triage_report_excludes_resolved() {
        let engine = ConflictResolutionEngine::new();
        let mut c1 = make_conflict("c1", ConflictType::OverlappingExclusive);
        c1.mark_resolved();
        let report = ConflictTriageReport::from_conflicts(&engine, &[c1]);
        assert_eq!(report.total_conflicts, 0);
        assert!(report.is_safe());
    }

    #[test]
    fn test_triage_report_type_breakdown() {
        let engine = ConflictResolutionEngine::new();
        let conflicts = vec![
            make_conflict("c1", ConflictType::RoyaltyDefault),
            make_conflict("c2", ConflictType::RoyaltyDefault),
            make_conflict("c3", ConflictType::ScopeBreach),
        ];
        let report = ConflictTriageReport::from_conflicts(&engine, &conflicts);
        assert_eq!(
            report.type_breakdown.get("RoyaltyDefault").copied(),
            Some(2)
        );
        assert_eq!(report.type_breakdown.get("ScopeBreach").copied(), Some(1));
    }

    #[test]
    fn test_triage_report_top_suggestions() {
        let engine = ConflictResolutionEngine::new();
        let conflicts = vec![
            make_conflict("c1", ConflictType::OverlappingExclusive),
            make_conflict("c2", ConflictType::RoyaltyDefault),
        ];
        let report = ConflictTriageReport::from_conflicts(&engine, &conflicts);
        let top2 = report.top_suggestions(2);
        assert_eq!(top2.len(), 2);
    }

    #[test]
    fn test_triage_report_top_suggestions_more_than_available() {
        let engine = ConflictResolutionEngine::new();
        let conflicts = vec![make_conflict("c1", ConflictType::ExpiredLicense)];
        let report = ConflictTriageReport::from_conflicts(&engine, &conflicts);
        let top10 = report.top_suggestions(10);
        assert!(top10.len() <= 10);
        assert_eq!(top10.len(), report.suggestions.len());
    }

    #[test]
    fn test_triage_report_empty_conflicts() {
        let engine = ConflictResolutionEngine::new();
        let report = ConflictTriageReport::from_conflicts(&engine, &[]);
        assert_eq!(report.total_conflicts, 0);
        assert!(report.is_safe());
        assert!(report.suggestions.is_empty());
    }

    #[test]
    fn test_suggest_no_involved_ids_does_not_panic() {
        let engine = ConflictResolutionEngine::new();
        let conflict = RightsConflict::new(
            "c1",
            ConflictType::OverlappingExclusive,
            "Empty involved",
            vec![],
        );
        let suggestions = engine.suggest(&conflict);
        // Should still produce the renegotiate suggestion at minimum
        assert!(!suggestions.is_empty());
    }
}
