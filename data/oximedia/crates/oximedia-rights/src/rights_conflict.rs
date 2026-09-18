//! Rights conflict detection and resolution.

#![allow(dead_code)]

/// Categories of conflict that can arise between rights holders or licenses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConflictType {
    /// Two exclusive licenses covering the same territory and usage.
    OverlappingExclusive,
    /// A license has expired but is still being exercised.
    ExpiredLicense,
    /// Usage extends beyond the permitted territory.
    TerritoryBreach,
    /// Usage exceeds the permitted scope (e.g. commercial use on a non-commercial license).
    ScopeBreach,
    /// A sublicense was granted without authority.
    UnauthorizedSublicense,
    /// Royalty payments are overdue.
    RoyaltyDefault,
}

impl ConflictType {
    /// Severity score on a 1–10 scale (10 = most severe).
    pub fn severity(&self) -> u8 {
        match self {
            ConflictType::OverlappingExclusive => 9,
            ConflictType::ExpiredLicense => 7,
            ConflictType::TerritoryBreach => 8,
            ConflictType::ScopeBreach => 8,
            ConflictType::UnauthorizedSublicense => 9,
            ConflictType::RoyaltyDefault => 6,
        }
    }

    /// Returns `true` when the severity is high enough to be considered critical (>= 8).
    pub fn is_critical(&self) -> bool {
        self.severity() >= 8
    }
}

/// A detected conflict between rights or licenses.
#[derive(Debug, Clone)]
pub struct RightsConflict {
    /// Unique identifier for this conflict instance.
    pub id: String,
    /// The kind of conflict.
    pub conflict_type: ConflictType,
    /// Human-readable description of the conflict.
    pub description: String,
    /// IDs of the rights or licenses involved.
    pub involved_ids: Vec<String>,
    /// Whether this conflict has been resolved.
    pub resolved: bool,
}

impl RightsConflict {
    /// Create a new unresolved conflict.
    pub fn new(
        id: impl Into<String>,
        conflict_type: ConflictType,
        description: impl Into<String>,
        involved_ids: Vec<String>,
    ) -> Self {
        Self {
            id: id.into(),
            conflict_type,
            description: description.into(),
            involved_ids,
            resolved: false,
        }
    }

    /// Return a reference to the conflict's description string.
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Mark this conflict as resolved.
    pub fn mark_resolved(&mut self) {
        self.resolved = true;
    }

    /// Returns `true` when the conflict has been resolved.
    pub fn is_resolved(&self) -> bool {
        self.resolved
    }
}

/// Detects and resolves `RightsConflict` instances from a set of rights data.
#[derive(Debug, Default)]
pub struct ConflictResolver {
    conflicts: Vec<RightsConflict>,
}

impl ConflictResolver {
    /// Create a new resolver with an empty conflict list.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a manually detected conflict.
    pub fn detect_conflicts(&mut self, conflict: RightsConflict) {
        self.conflicts.push(conflict);
    }

    /// Attempt to resolve a conflict by ID. Returns `true` if found and marked resolved.
    pub fn resolve(&mut self, conflict_id: &str) -> bool {
        if let Some(c) = self.conflicts.iter_mut().find(|c| c.id == conflict_id) {
            c.mark_resolved();
            true
        } else {
            false
        }
    }

    /// Generate a report summarising all known conflicts.
    pub fn report(&self) -> ConflictReport {
        ConflictReport {
            total: self.conflicts.len(),
            resolved: self.conflicts.iter().filter(|c| c.resolved).count(),
            critical_unresolved: self
                .conflicts
                .iter()
                .filter(|c| !c.resolved && c.conflict_type.is_critical())
                .count(),
            conflicts: self.conflicts.clone(),
        }
    }

    /// Return all unresolved conflicts.
    pub fn unresolved(&self) -> Vec<&RightsConflict> {
        self.conflicts.iter().filter(|c| !c.resolved).collect()
    }
}

/// Summary report produced by `ConflictResolver`.
#[derive(Debug)]
pub struct ConflictReport {
    /// Total number of conflicts tracked.
    pub total: usize,
    /// Number that have been resolved.
    pub resolved: usize,
    /// Number of critical conflicts that remain unresolved.
    pub critical_unresolved: usize,
    /// Full list of conflicts.
    pub conflicts: Vec<RightsConflict>,
}

impl ConflictReport {
    /// Return the count of critical (unresolved) conflicts.
    pub fn critical_count(&self) -> usize {
        self.critical_unresolved
    }

    /// Returns `true` when all conflicts have been resolved.
    pub fn all_clear(&self) -> bool {
        self.critical_unresolved == 0 && self.resolved == self.total
    }
}

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

    #[test]
    fn test_severity_overlapping_exclusive() {
        assert_eq!(ConflictType::OverlappingExclusive.severity(), 9);
    }

    #[test]
    fn test_severity_royalty_default() {
        assert_eq!(ConflictType::RoyaltyDefault.severity(), 6);
    }

    #[test]
    fn test_is_critical_territory_breach() {
        assert!(ConflictType::TerritoryBreach.is_critical());
    }

    #[test]
    fn test_is_not_critical_royalty_default() {
        assert!(!ConflictType::RoyaltyDefault.is_critical());
    }

    #[test]
    fn test_conflict_description() {
        let c = make_conflict("c1", ConflictType::ExpiredLicense);
        assert_eq!(c.description(), "Test conflict");
    }

    #[test]
    fn test_conflict_initial_not_resolved() {
        let c = make_conflict("c1", ConflictType::ScopeBreach);
        assert!(!c.is_resolved());
    }

    #[test]
    fn test_conflict_mark_resolved() {
        let mut c = make_conflict("c1", ConflictType::ScopeBreach);
        c.mark_resolved();
        assert!(c.is_resolved());
    }

    #[test]
    fn test_resolver_detect_and_resolve() {
        let mut resolver = ConflictResolver::new();
        resolver.detect_conflicts(make_conflict("c1", ConflictType::TerritoryBreach));
        assert!(resolver.resolve("c1"));
    }

    #[test]
    fn test_resolver_resolve_unknown_returns_false() {
        let mut resolver = ConflictResolver::new();
        assert!(!resolver.resolve("no-such-id"));
    }

    #[test]
    fn test_report_critical_count() {
        let mut resolver = ConflictResolver::new();
        resolver.detect_conflicts(make_conflict("c1", ConflictType::OverlappingExclusive));
        resolver.detect_conflicts(make_conflict("c2", ConflictType::RoyaltyDefault));
        let report = resolver.report();
        assert_eq!(report.critical_count(), 1);
    }

    #[test]
    fn test_report_all_clear_after_resolution() {
        let mut resolver = ConflictResolver::new();
        resolver.detect_conflicts(make_conflict("c1", ConflictType::ScopeBreach));
        resolver.resolve("c1");
        let report = resolver.report();
        assert!(report.all_clear());
    }

    #[test]
    fn test_report_totals() {
        let mut resolver = ConflictResolver::new();
        resolver.detect_conflicts(make_conflict("c1", ConflictType::ExpiredLicense));
        resolver.detect_conflicts(make_conflict("c2", ConflictType::TerritoryBreach));
        resolver.resolve("c1");
        let report = resolver.report();
        assert_eq!(report.total, 2);
        assert_eq!(report.resolved, 1);
    }

    #[test]
    fn test_unresolved_list() {
        let mut resolver = ConflictResolver::new();
        resolver.detect_conflicts(make_conflict("c1", ConflictType::RoyaltyDefault));
        resolver.detect_conflicts(make_conflict("c2", ConflictType::ScopeBreach));
        resolver.resolve("c1");
        assert_eq!(resolver.unresolved().len(), 1);
    }

    #[test]
    fn test_unauthorized_sublicense_is_critical() {
        assert!(ConflictType::UnauthorizedSublicense.is_critical());
    }
}
