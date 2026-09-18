//! Edit conflict resolution for collaborative video editing.
//!
//! Provides last-write-wins, merge strategies, and conflict detection
//! for concurrent edits to timeline regions.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// A logical clock value for ordering operations.
pub type LogicalClock = u64;

/// An identifier for an edit operation.
pub type EditId = Uuid;

/// The region of the timeline affected by an edit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimelineRegion {
    /// Track identifier.
    pub track_id: String,
    /// Start time in milliseconds.
    pub start_ms: i64,
    /// End time in milliseconds.
    pub end_ms: i64,
}

impl TimelineRegion {
    /// Create a new timeline region.
    pub fn new(track_id: impl Into<String>, start_ms: i64, end_ms: i64) -> Self {
        Self {
            track_id: track_id.into(),
            start_ms,
            end_ms,
        }
    }

    /// Check whether this region overlaps another.
    pub fn overlaps(&self, other: &TimelineRegion) -> bool {
        self.track_id == other.track_id
            && self.start_ms < other.end_ms
            && other.start_ms < self.end_ms
    }
}

/// Type of edit operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EditKind {
    /// Insert content at a position.
    Insert,
    /// Delete a region.
    Delete,
    /// Move a region.
    Move,
    /// Modify properties (e.g. gain, color).
    Modify,
    /// Replace a clip.
    Replace,
}

/// An edit operation submitted by a user.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditOperation {
    /// Unique operation ID.
    pub id: EditId,
    /// Author user ID.
    pub author: Uuid,
    /// Logical clock at submission time.
    pub clock: LogicalClock,
    /// Affected timeline region.
    pub region: TimelineRegion,
    /// Type of edit.
    pub kind: EditKind,
    /// Serialized payload (kind-specific).
    pub payload: serde_json::Value,
}

impl EditOperation {
    /// Create a new edit operation.
    pub fn new(
        author: Uuid,
        clock: LogicalClock,
        region: TimelineRegion,
        kind: EditKind,
        payload: serde_json::Value,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            author,
            clock,
            region,
            kind,
            payload,
        }
    }
}

/// How to resolve a conflict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolutionStrategy {
    /// The operation with the higher clock wins.
    LastWriteWins,
    /// The operation with the lower clock (earlier) wins.
    FirstWriteWins,
    /// Keep both operations (requires manual review).
    KeepBoth,
    /// Reject the incoming operation.
    RejectIncoming,
}

/// A detected conflict between two operations.
#[derive(Debug, Clone)]
pub struct Conflict {
    /// The existing (accepted) operation.
    pub existing: EditOperation,
    /// The incoming (challenger) operation.
    pub incoming: EditOperation,
    /// Chosen resolution strategy.
    pub resolution: ResolutionStrategy,
}

impl Conflict {
    /// Determine which operation wins under the chosen strategy.
    ///
    /// Returns `Some(op)` for the winner or `None` if both are kept.
    pub fn winner(&self) -> Option<&EditOperation> {
        match self.resolution {
            ResolutionStrategy::LastWriteWins => {
                if self.incoming.clock >= self.existing.clock {
                    Some(&self.incoming)
                } else {
                    Some(&self.existing)
                }
            }
            ResolutionStrategy::FirstWriteWins => {
                if self.existing.clock <= self.incoming.clock {
                    Some(&self.existing)
                } else {
                    Some(&self.incoming)
                }
            }
            ResolutionStrategy::KeepBoth => None,
            ResolutionStrategy::RejectIncoming => Some(&self.existing),
        }
    }
}

/// Conflict resolver that maintains applied operations.
#[derive(Debug)]
pub struct ConflictResolver {
    strategy: ResolutionStrategy,
    /// Applied operations keyed by region track.
    applied: HashMap<String, Vec<EditOperation>>,
}

impl ConflictResolver {
    /// Create a new resolver with the given strategy.
    pub fn new(strategy: ResolutionStrategy) -> Self {
        Self {
            strategy,
            applied: HashMap::new(),
        }
    }

    /// Detect conflicts between an incoming operation and applied operations.
    pub fn detect_conflicts(&self, incoming: &EditOperation) -> Vec<Conflict> {
        let track_ops = match self.applied.get(&incoming.region.track_id) {
            Some(ops) => ops,
            None => return vec![],
        };

        track_ops
            .iter()
            .filter(|existing| existing.region.overlaps(&incoming.region))
            .map(|existing| Conflict {
                existing: existing.clone(),
                incoming: incoming.clone(),
                resolution: self.strategy,
            })
            .collect()
    }

    /// Apply an operation, resolving conflicts as configured.
    ///
    /// Returns the list of conflicts that were resolved.
    pub fn apply(&mut self, incoming: EditOperation) -> Vec<Conflict> {
        let conflicts = self.detect_conflicts(&incoming);

        let accept_incoming = if conflicts.is_empty() {
            true
        } else {
            // Check if any conflict results in rejecting the incoming op
            conflicts.iter().all(|c| {
                !matches!(c.winner(), Some(w) if std::ptr::eq(w, &c.existing))
                    || matches!(self.strategy, ResolutionStrategy::LastWriteWins
                        if incoming.clock >= c.existing.clock)
                    || matches!(self.strategy, ResolutionStrategy::KeepBoth)
            });
            // Simplified: accept if strategy is not RejectIncoming
            !matches!(self.strategy, ResolutionStrategy::RejectIncoming)
        };

        if accept_incoming {
            // Under LastWriteWins, remove conflicting existing ops
            if matches!(self.strategy, ResolutionStrategy::LastWriteWins) {
                let incoming_region = incoming.region.clone();
                let track_id = incoming.region.track_id.clone();
                if let Some(ops) = self.applied.get_mut(&track_id) {
                    ops.retain(|op| !op.region.overlaps(&incoming_region));
                }
            }

            self.applied
                .entry(incoming.region.track_id.clone())
                .or_default()
                .push(incoming);
        }

        conflicts
    }

    /// Get all applied operations for a track.
    pub fn operations_for_track(&self, track_id: &str) -> &[EditOperation] {
        self.applied.get(track_id).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Total number of applied operations.
    pub fn operation_count(&self) -> usize {
        self.applied.values().map(Vec::len).sum()
    }
}

// ---------------------------------------------------------------------------
// Visual diff presentation for conflicting edits
// ---------------------------------------------------------------------------

/// A single field-level difference between two operations.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldDiff {
    /// Field name (e.g. "region.start_ms", "kind", "payload.volume").
    pub field: String,
    /// Value in the existing (accepted) operation, serialized as a string.
    pub existing_value: String,
    /// Value in the incoming (challenger) operation, serialized as a string.
    pub incoming_value: String,
    /// Severity of the difference.
    pub severity: DiffSeverity,
}

/// How significant a field-level difference is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DiffSeverity {
    /// Cosmetic / metadata change.
    Info,
    /// Timing or spatial shift.
    Warning,
    /// Structural change (kind mismatch, deletion vs modification).
    Critical,
}

impl std::fmt::Display for DiffSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Info => write!(f, "info"),
            Self::Warning => write!(f, "warning"),
            Self::Critical => write!(f, "critical"),
        }
    }
}

/// A visual diff between two conflicting operations.
#[derive(Debug, Clone)]
pub struct ConflictDiff {
    /// ID of the existing operation.
    pub existing_id: EditId,
    /// ID of the incoming operation.
    pub incoming_id: EditId,
    /// Per-field differences.
    pub diffs: Vec<FieldDiff>,
    /// Overall severity (the maximum of all field severities).
    pub overall_severity: DiffSeverity,
    /// Human-readable summary of the conflict.
    pub summary: String,
}

impl ConflictDiff {
    /// Whether this diff contains any critical differences.
    #[must_use]
    pub fn has_critical(&self) -> bool {
        self.diffs
            .iter()
            .any(|d| d.severity == DiffSeverity::Critical)
    }

    /// Number of field-level differences.
    #[must_use]
    pub fn diff_count(&self) -> usize {
        self.diffs.len()
    }

    /// Render the diff as a multi-line text report.
    #[must_use]
    pub fn render_text(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "Conflict: {} vs {} ({} differences)",
            self.existing_id,
            self.incoming_id,
            self.diffs.len()
        ));
        lines.push(format!("Severity: {}", self.overall_severity));
        lines.push(format!("Summary: {}", self.summary));
        lines.push(String::new());
        for diff in &self.diffs {
            lines.push(format!(
                "  [{severity}] {field}:",
                severity = diff.severity,
                field = diff.field,
            ));
            lines.push(format!("    existing: {}", diff.existing_value));
            lines.push(format!("    incoming: {}", diff.incoming_value));
        }
        lines.join("\n")
    }
}

/// Generate a visual diff for a conflict.
pub fn diff_conflict(conflict: &Conflict) -> ConflictDiff {
    let existing = &conflict.existing;
    let incoming = &conflict.incoming;
    let mut diffs = Vec::new();

    // Compare kind
    if existing.kind != incoming.kind {
        diffs.push(FieldDiff {
            field: "kind".to_string(),
            existing_value: format!("{:?}", existing.kind),
            incoming_value: format!("{:?}", incoming.kind),
            severity: DiffSeverity::Critical,
        });
    }

    // Compare region.track_id
    if existing.region.track_id != incoming.region.track_id {
        diffs.push(FieldDiff {
            field: "region.track_id".to_string(),
            existing_value: existing.region.track_id.clone(),
            incoming_value: incoming.region.track_id.clone(),
            severity: DiffSeverity::Warning,
        });
    }

    // Compare region.start_ms
    if existing.region.start_ms != incoming.region.start_ms {
        diffs.push(FieldDiff {
            field: "region.start_ms".to_string(),
            existing_value: existing.region.start_ms.to_string(),
            incoming_value: incoming.region.start_ms.to_string(),
            severity: DiffSeverity::Warning,
        });
    }

    // Compare region.end_ms
    if existing.region.end_ms != incoming.region.end_ms {
        diffs.push(FieldDiff {
            field: "region.end_ms".to_string(),
            existing_value: existing.region.end_ms.to_string(),
            incoming_value: incoming.region.end_ms.to_string(),
            severity: DiffSeverity::Warning,
        });
    }

    // Compare clock
    if existing.clock != incoming.clock {
        diffs.push(FieldDiff {
            field: "clock".to_string(),
            existing_value: existing.clock.to_string(),
            incoming_value: incoming.clock.to_string(),
            severity: DiffSeverity::Info,
        });
    }

    // Compare author
    if existing.author != incoming.author {
        diffs.push(FieldDiff {
            field: "author".to_string(),
            existing_value: existing.author.to_string(),
            incoming_value: incoming.author.to_string(),
            severity: DiffSeverity::Info,
        });
    }

    // Compare payload (as JSON strings)
    let existing_payload = existing.payload.to_string();
    let incoming_payload = incoming.payload.to_string();
    if existing_payload != incoming_payload {
        diffs.push(FieldDiff {
            field: "payload".to_string(),
            existing_value: existing_payload,
            incoming_value: incoming_payload,
            severity: DiffSeverity::Warning,
        });
    }

    let overall_severity = diffs
        .iter()
        .map(|d| d.severity)
        .max()
        .unwrap_or(DiffSeverity::Info);

    let summary = build_conflict_summary(existing, incoming, &diffs);

    ConflictDiff {
        existing_id: existing.id,
        incoming_id: incoming.id,
        diffs,
        overall_severity,
        summary,
    }
}

/// Build a human-readable summary of a conflict.
fn build_conflict_summary(
    existing: &EditOperation,
    incoming: &EditOperation,
    diffs: &[FieldDiff],
) -> String {
    let kind_conflict = diffs.iter().any(|d| d.field == "kind");
    let timing_conflict = diffs
        .iter()
        .any(|d| d.field.starts_with("region.") && d.severity >= DiffSeverity::Warning);

    if kind_conflict {
        format!(
            "Structural conflict: {:?} vs {:?} on {} [{}-{}ms]",
            existing.kind,
            incoming.kind,
            existing.region.track_id,
            existing.region.start_ms,
            existing.region.end_ms,
        )
    } else if timing_conflict {
        format!(
            "Timing conflict on {}: [{}-{}ms] vs [{}-{}ms]",
            existing.region.track_id,
            existing.region.start_ms,
            existing.region.end_ms,
            incoming.region.start_ms,
            incoming.region.end_ms,
        )
    } else {
        format!(
            "Parameter conflict on {} at [{}-{}ms] (clock {} vs {})",
            existing.region.track_id,
            existing.region.start_ms,
            existing.region.end_ms,
            existing.clock,
            incoming.clock,
        )
    }
}

/// Batch-diff all conflicts from a resolver.
pub fn diff_all_conflicts(conflicts: &[Conflict]) -> Vec<ConflictDiff> {
    conflicts.iter().map(diff_conflict).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user() -> Uuid {
        Uuid::new_v4()
    }

    fn region(start: i64, end: i64) -> TimelineRegion {
        TimelineRegion::new("track_1", start, end)
    }

    fn op(author: Uuid, clock: u64, r: TimelineRegion) -> EditOperation {
        EditOperation::new(author, clock, r, EditKind::Modify, serde_json::Value::Null)
    }

    #[test]
    fn test_region_overlaps() {
        let a = region(0, 1000);
        let b = region(500, 1500);
        assert!(a.overlaps(&b));
    }

    #[test]
    fn test_region_no_overlap_adjacent() {
        let a = region(0, 1000);
        let b = region(1000, 2000);
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn test_region_different_track_no_overlap() {
        let a = TimelineRegion::new("track_1", 0, 1000);
        let b = TimelineRegion::new("track_2", 0, 1000);
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn test_apply_non_conflicting() {
        let mut resolver = ConflictResolver::new(ResolutionStrategy::LastWriteWins);
        let o = op(user(), 1, region(0, 1000));
        let conflicts = resolver.apply(o);
        assert!(conflicts.is_empty());
        assert_eq!(resolver.operation_count(), 1);
    }

    #[test]
    fn test_detect_conflict_overlapping_ops() {
        let mut resolver = ConflictResolver::new(ResolutionStrategy::LastWriteWins);
        let u1 = user();
        let o1 = op(u1, 1, region(0, 1000));
        resolver.apply(o1);

        let o2 = op(user(), 2, region(500, 1500));
        let conflicts = resolver.detect_conflicts(&o2);
        assert_eq!(conflicts.len(), 1);
    }

    #[test]
    fn test_last_write_wins_removes_old_op() {
        let mut resolver = ConflictResolver::new(ResolutionStrategy::LastWriteWins);
        let o1 = op(user(), 1, region(0, 1000));
        resolver.apply(o1);

        let o2 = op(user(), 5, region(0, 1000));
        resolver.apply(o2);

        let ops = resolver.operations_for_track("track_1");
        // Under LastWriteWins, old conflicting op should be replaced
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].clock, 5);
    }

    #[test]
    fn test_reject_incoming_strategy() {
        let mut resolver = ConflictResolver::new(ResolutionStrategy::RejectIncoming);
        let o1 = op(user(), 1, region(0, 1000));
        resolver.apply(o1);

        let o2 = op(user(), 5, region(0, 1000));
        resolver.apply(o2);

        let ops = resolver.operations_for_track("track_1");
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].clock, 1);
    }

    #[test]
    fn test_conflict_winner_last_write_wins() {
        let existing = op(user(), 1, region(0, 1000));
        let incoming = op(user(), 5, region(0, 1000));
        let conflict = Conflict {
            existing: existing.clone(),
            incoming: incoming.clone(),
            resolution: ResolutionStrategy::LastWriteWins,
        };
        let winner = conflict
            .winner()
            .expect("collab test operation should succeed");
        assert_eq!(winner.clock, 5);
    }

    #[test]
    fn test_conflict_winner_first_write_wins() {
        let existing = op(user(), 1, region(0, 1000));
        let incoming = op(user(), 5, region(0, 1000));
        let conflict = Conflict {
            existing: existing.clone(),
            incoming: incoming.clone(),
            resolution: ResolutionStrategy::FirstWriteWins,
        };
        let winner = conflict
            .winner()
            .expect("collab test operation should succeed");
        assert_eq!(winner.clock, 1);
    }

    #[test]
    fn test_conflict_winner_keep_both_returns_none() {
        let existing = op(user(), 1, region(0, 1000));
        let incoming = op(user(), 5, region(0, 1000));
        let conflict = Conflict {
            existing,
            incoming,
            resolution: ResolutionStrategy::KeepBoth,
        };
        assert!(conflict.winner().is_none());
    }

    #[test]
    fn test_conflict_winner_reject_incoming() {
        let existing = op(user(), 1, region(0, 1000));
        let incoming = op(user(), 5, region(0, 1000));
        let conflict = Conflict {
            existing: existing.clone(),
            incoming,
            resolution: ResolutionStrategy::RejectIncoming,
        };
        let winner = conflict
            .winner()
            .expect("collab test operation should succeed");
        assert_eq!(winner.clock, 1);
    }

    #[test]
    fn test_non_overlapping_ops_accumulate() {
        let mut resolver = ConflictResolver::new(ResolutionStrategy::LastWriteWins);
        resolver.apply(op(user(), 1, region(0, 500)));
        resolver.apply(op(user(), 2, region(500, 1000)));
        assert_eq!(resolver.operation_count(), 2);
    }

    #[test]
    fn test_operations_for_unknown_track() {
        let resolver = ConflictResolver::new(ResolutionStrategy::LastWriteWins);
        assert!(resolver.operations_for_track("nonexistent").is_empty());
    }

    #[test]
    fn test_edit_operation_new_generates_unique_ids() {
        let u = user();
        let o1 = EditOperation::new(
            u,
            1,
            region(0, 100),
            EditKind::Insert,
            serde_json::Value::Null,
        );
        let o2 = EditOperation::new(
            u,
            2,
            region(0, 100),
            EditKind::Insert,
            serde_json::Value::Null,
        );
        assert_ne!(o1.id, o2.id);
    }

    #[test]
    fn test_region_duration() {
        let r = region(1000, 4000);
        assert_eq!(r.end_ms - r.start_ms, 3000);
    }

    // ---- Visual diff tests ----

    #[test]
    fn test_diff_severity_ordering() {
        assert!(DiffSeverity::Critical > DiffSeverity::Warning);
        assert!(DiffSeverity::Warning > DiffSeverity::Info);
    }

    #[test]
    fn test_diff_severity_display() {
        assert_eq!(DiffSeverity::Info.to_string(), "info");
        assert_eq!(DiffSeverity::Warning.to_string(), "warning");
        assert_eq!(DiffSeverity::Critical.to_string(), "critical");
    }

    #[test]
    fn test_diff_identical_ops_only_clock_and_author_differ() {
        let u1 = user();
        let u2 = user();
        let existing = op(u1, 1, region(0, 1000));
        let incoming = op(u2, 2, region(0, 1000));
        let conflict = Conflict {
            existing,
            incoming,
            resolution: ResolutionStrategy::LastWriteWins,
        };
        let diff = diff_conflict(&conflict);
        // clock and author differ
        assert!(diff.diffs.iter().any(|d| d.field == "clock"));
        assert!(diff.diffs.iter().any(|d| d.field == "author"));
        assert_eq!(diff.overall_severity, DiffSeverity::Info);
    }

    #[test]
    fn test_diff_different_kind_is_critical() {
        let u = user();
        let mut existing = op(u, 1, region(0, 1000));
        existing.kind = EditKind::Insert;
        let mut incoming = op(u, 2, region(0, 1000));
        incoming.kind = EditKind::Delete;
        let conflict = Conflict {
            existing,
            incoming,
            resolution: ResolutionStrategy::KeepBoth,
        };
        let diff = diff_conflict(&conflict);
        assert!(diff.has_critical());
        assert_eq!(diff.overall_severity, DiffSeverity::Critical);
    }

    #[test]
    fn test_diff_timing_conflict_is_warning() {
        let u = user();
        let existing = op(u, 1, region(0, 1000));
        let incoming = op(u, 1, region(500, 1500));
        let conflict = Conflict {
            existing,
            incoming,
            resolution: ResolutionStrategy::LastWriteWins,
        };
        let diff = diff_conflict(&conflict);
        let timing_diffs: Vec<_> = diff
            .diffs
            .iter()
            .filter(|d| d.field.starts_with("region."))
            .collect();
        assert!(!timing_diffs.is_empty());
        assert!(timing_diffs
            .iter()
            .all(|d| d.severity == DiffSeverity::Warning));
    }

    #[test]
    fn test_diff_payload_difference() {
        let u = user();
        let mut existing = EditOperation::new(
            u,
            1,
            region(0, 1000),
            EditKind::Modify,
            serde_json::json!({"volume": 0.5}),
        );
        let incoming = EditOperation::new(
            u,
            1,
            region(0, 1000),
            EditKind::Modify,
            serde_json::json!({"volume": 0.8}),
        );
        // Make ids differ to avoid id equality
        existing.id = Uuid::new_v4();
        let conflict = Conflict {
            existing,
            incoming,
            resolution: ResolutionStrategy::KeepBoth,
        };
        let diff = diff_conflict(&conflict);
        assert!(diff.diffs.iter().any(|d| d.field == "payload"));
    }

    #[test]
    fn test_diff_render_text_contains_fields() {
        let u1 = user();
        let u2 = user();
        let existing = op(u1, 1, region(0, 1000));
        let incoming = op(u2, 5, region(0, 1000));
        let conflict = Conflict {
            existing,
            incoming,
            resolution: ResolutionStrategy::LastWriteWins,
        };
        let diff = diff_conflict(&conflict);
        let text = diff.render_text();
        assert!(text.contains("clock"));
        assert!(text.contains("existing:"));
        assert!(text.contains("incoming:"));
        assert!(text.contains("Severity:"));
    }

    #[test]
    fn test_diff_count() {
        let u = user();
        let existing = op(u, 1, region(0, 1000));
        let incoming = op(u, 5, region(500, 1500));
        let conflict = Conflict {
            existing,
            incoming,
            resolution: ResolutionStrategy::KeepBoth,
        };
        let diff = diff_conflict(&conflict);
        assert!(diff.diff_count() >= 2); // at least clock + region changes
    }

    #[test]
    fn test_diff_all_conflicts_batch() {
        let mut resolver = ConflictResolver::new(ResolutionStrategy::KeepBoth);
        let o1 = op(user(), 1, region(0, 1000));
        resolver.apply(o1);
        let o2 = op(user(), 2, region(500, 1500));
        let conflicts = resolver.apply(o2);
        let diffs = diff_all_conflicts(&conflicts);
        assert_eq!(diffs.len(), conflicts.len());
    }

    #[test]
    fn test_conflict_summary_structural() {
        let u = user();
        let mut existing = op(u, 1, region(0, 1000));
        existing.kind = EditKind::Insert;
        let mut incoming = op(u, 2, region(0, 1000));
        incoming.kind = EditKind::Delete;
        let conflict = Conflict {
            existing,
            incoming,
            resolution: ResolutionStrategy::KeepBoth,
        };
        let diff = diff_conflict(&conflict);
        assert!(diff.summary.contains("Structural conflict"));
    }

    #[test]
    fn test_conflict_summary_timing() {
        let u = user();
        let existing = op(u, 1, region(0, 1000));
        let incoming = op(u, 1, region(500, 1500));
        let conflict = Conflict {
            existing,
            incoming,
            resolution: ResolutionStrategy::KeepBoth,
        };
        let diff = diff_conflict(&conflict);
        assert!(diff.summary.contains("Timing conflict"));
    }
}
