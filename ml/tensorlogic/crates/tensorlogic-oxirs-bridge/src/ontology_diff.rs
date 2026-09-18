//! Ontology diff: compare two [`SymbolTable`]s and report differences.
//!
//! The primary entry point is [`compare_symbol_tables`], which returns an
//! [`OntologyDiff`] describing every domain and predicate that was added,
//! removed, or modified between two symbol tables.

use serde::{Deserialize, Serialize};
use tensorlogic_adapters::SymbolTable;

// ── DiffEntry ─────────────────────────────────────────────────────────────────

/// A single change entry recorded in an [`OntologyDiff`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DiffEntry {
    /// The symbol exists in `b` but not in `a`.
    Added(String),
    /// The symbol exists in `a` but not in `b`.
    Removed(String),
    /// The symbol exists in both but its definition changed.
    Modified { before: String, after: String },
}

impl DiffEntry {
    /// Return the name of the symbol this entry refers to.
    pub fn name(&self) -> &str {
        match self {
            DiffEntry::Added(n) => n.as_str(),
            DiffEntry::Removed(n) => n.as_str(),
            DiffEntry::Modified { before, .. } => before.as_str(),
        }
    }

    /// Returns `true` if this is an [`DiffEntry::Added`] variant.
    pub fn is_addition(&self) -> bool {
        matches!(self, DiffEntry::Added(_))
    }

    /// Returns `true` if this is a [`DiffEntry::Removed`] variant.
    pub fn is_removal(&self) -> bool {
        matches!(self, DiffEntry::Removed(_))
    }

    /// Returns `true` if this is a [`DiffEntry::Modified`] variant.
    pub fn is_modification(&self) -> bool {
        matches!(self, DiffEntry::Modified { .. })
    }
}

// ── OntologyDiff ──────────────────────────────────────────────────────────────

/// Result of comparing two [`SymbolTable`]s.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OntologyDiff {
    /// Changes in domain (class) definitions.
    pub domain_diffs: Vec<DiffEntry>,
    /// Changes in predicate (property) definitions.
    pub predicate_diffs: Vec<DiffEntry>,
}

impl OntologyDiff {
    /// Create a new, empty [`OntologyDiff`].
    pub fn new() -> Self {
        OntologyDiff::default()
    }

    /// Returns `true` when no domain or predicate differences were found.
    pub fn is_empty(&self) -> bool {
        self.domain_diffs.is_empty() && self.predicate_diffs.is_empty()
    }

    /// Total number of change entries across domains and predicates.
    pub fn total_changes(&self) -> usize {
        self.domain_diffs.len() + self.predicate_diffs.len()
    }

    /// Human-readable multi-line report.
    pub fn report(&self) -> String {
        let mut out = String::new();
        out.push_str("OntologyDiff Report\n");
        out.push_str("===================\n");

        if self.domain_diffs.is_empty() {
            out.push_str("Domains: no changes\n");
        } else {
            out.push_str("Domains:\n");
            for entry in &self.domain_diffs {
                match entry {
                    DiffEntry::Added(n) => {
                        out.push_str(&format!("  + Added: {}\n", n));
                    }
                    DiffEntry::Removed(n) => {
                        out.push_str(&format!("  - Removed: {}\n", n));
                    }
                    DiffEntry::Modified { before, after } => {
                        out.push_str(&format!("  ~ Modified: {} -> {}\n", before, after));
                    }
                }
            }
        }

        if self.predicate_diffs.is_empty() {
            out.push_str("Predicates: no changes\n");
        } else {
            out.push_str("Predicates:\n");
            for entry in &self.predicate_diffs {
                match entry {
                    DiffEntry::Added(n) => {
                        out.push_str(&format!("  + Added: {}\n", n));
                    }
                    DiffEntry::Removed(n) => {
                        out.push_str(&format!("  - Removed: {}\n", n));
                    }
                    DiffEntry::Modified { before, after } => {
                        out.push_str(&format!("  ~ Modified: {} -> {}\n", before, after));
                    }
                }
            }
        }

        out
    }

    /// Short one-line summary.
    pub fn summary(&self) -> String {
        let added = self.domain_diffs.iter().filter(|e| e.is_addition()).count()
            + self
                .predicate_diffs
                .iter()
                .filter(|e| e.is_addition())
                .count();
        let removed = self.domain_diffs.iter().filter(|e| e.is_removal()).count()
            + self
                .predicate_diffs
                .iter()
                .filter(|e| e.is_removal())
                .count();
        let modified = self
            .domain_diffs
            .iter()
            .filter(|e| e.is_modification())
            .count()
            + self
                .predicate_diffs
                .iter()
                .filter(|e| e.is_modification())
                .count();
        format!(
            "OntologyDiff: {} added, {} removed, {} modified ({} total)",
            added,
            removed,
            modified,
            self.total_changes()
        )
    }
}

// ── compare_symbol_tables ─────────────────────────────────────────────────────

/// Compare two [`SymbolTable`]s and return an [`OntologyDiff`] describing the
/// differences.
///
/// # Domain comparison
///
/// - A domain present in `b` but absent from `a` → [`DiffEntry::Added`].
/// - A domain present in `a` but absent from `b` → [`DiffEntry::Removed`].
/// - A domain present in both: if its `cardinality` changed → [`DiffEntry::Modified`].
///
/// # Predicate comparison
///
/// - A predicate present in `b` but absent from `a` → [`DiffEntry::Added`].
/// - A predicate present in `a` but absent from `b` → [`DiffEntry::Removed`].
/// - A predicate present in both: if its `arity` or `arg_domains` changed
///   → [`DiffEntry::Modified`].
pub fn compare_symbol_tables(a: &SymbolTable, b: &SymbolTable) -> OntologyDiff {
    let mut diff = OntologyDiff::new();

    // ── Domains ───────────────────────────────────────────────────────────────
    for (name, b_info) in &b.domains {
        match a.domains.get(name) {
            None => {
                diff.domain_diffs.push(DiffEntry::Added(name.clone()));
            }
            Some(a_info) => {
                if a_info.cardinality != b_info.cardinality {
                    diff.domain_diffs.push(DiffEntry::Modified {
                        before: format!("{}(cardinality={})", name, a_info.cardinality),
                        after: format!("{}(cardinality={})", name, b_info.cardinality),
                    });
                }
            }
        }
    }
    for name in a.domains.keys() {
        if !b.domains.contains_key(name) {
            diff.domain_diffs.push(DiffEntry::Removed(name.clone()));
        }
    }

    // ── Predicates ────────────────────────────────────────────────────────────
    for (name, b_pred) in &b.predicates {
        match a.predicates.get(name) {
            None => {
                diff.predicate_diffs.push(DiffEntry::Added(name.clone()));
            }
            Some(a_pred) => {
                let arity_changed = a_pred.arity != b_pred.arity;
                let domains_changed = a_pred.arg_domains != b_pred.arg_domains;
                if arity_changed || domains_changed {
                    diff.predicate_diffs.push(DiffEntry::Modified {
                        before: format!(
                            "{}(arity={}, domains=[{}])",
                            name,
                            a_pred.arity,
                            a_pred.arg_domains.join(", ")
                        ),
                        after: format!(
                            "{}(arity={}, domains=[{}])",
                            name,
                            b_pred.arity,
                            b_pred.arg_domains.join(", ")
                        ),
                    });
                }
            }
        }
    }
    for name in a.predicates.keys() {
        if !b.predicates.contains_key(name) {
            diff.predicate_diffs.push(DiffEntry::Removed(name.clone()));
        }
    }

    diff
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_adapters::{DomainInfo, PredicateInfo};

    fn make_table_with_domain(name: &str) -> SymbolTable {
        let mut t = SymbolTable::new();
        t.add_domain(DomainInfo::new(name, 10))
            .expect("add_domain should succeed");
        t
    }

    fn make_table_with_predicate(domain: &str, pred: &str) -> SymbolTable {
        let mut t = SymbolTable::new();
        t.add_domain(DomainInfo::new(domain, 10))
            .expect("add_domain should succeed");
        t.add_predicate(PredicateInfo::new(pred, vec![domain.to_string()]))
            .expect("add_predicate should succeed");
        t
    }

    #[test]
    fn test_diff_identical_tables() {
        let a = SymbolTable::new();
        let b = SymbolTable::new();
        let diff = compare_symbol_tables(&a, &b);
        assert!(diff.is_empty());
    }

    #[test]
    fn test_diff_added_domain() {
        let a = SymbolTable::new();
        let b = make_table_with_domain("Person");
        let diff = compare_symbol_tables(&a, &b);
        assert_eq!(diff.domain_diffs.len(), 1);
        assert!(diff.domain_diffs[0].is_addition());
        assert_eq!(diff.domain_diffs[0].name(), "Person");
    }

    #[test]
    fn test_diff_removed_domain() {
        let a = make_table_with_domain("Animal");
        let b = SymbolTable::new();
        let diff = compare_symbol_tables(&a, &b);
        assert_eq!(diff.domain_diffs.len(), 1);
        assert!(diff.domain_diffs[0].is_removal());
        assert_eq!(diff.domain_diffs[0].name(), "Animal");
    }

    #[test]
    fn test_diff_added_predicate() {
        let a = make_table_with_domain("Person");
        let b = make_table_with_predicate("Person", "knows");
        let diff = compare_symbol_tables(&a, &b);
        assert_eq!(diff.predicate_diffs.len(), 1);
        assert!(diff.predicate_diffs[0].is_addition());
        assert_eq!(diff.predicate_diffs[0].name(), "knows");
    }

    #[test]
    fn test_diff_removed_predicate() {
        let a = make_table_with_predicate("Person", "knows");
        let b = make_table_with_domain("Person");
        let diff = compare_symbol_tables(&a, &b);
        assert_eq!(diff.predicate_diffs.len(), 1);
        assert!(diff.predicate_diffs[0].is_removal());
        assert_eq!(diff.predicate_diffs[0].name(), "knows");
    }

    #[test]
    fn test_diff_is_empty_on_empty() {
        assert!(OntologyDiff::new().is_empty());
    }

    #[test]
    fn test_diff_total_changes() {
        let mut diff = OntologyDiff::new();
        diff.domain_diffs.push(DiffEntry::Added("A".to_string()));
        diff.domain_diffs.push(DiffEntry::Added("B".to_string()));
        diff.predicate_diffs
            .push(DiffEntry::Removed("p".to_string()));
        assert_eq!(diff.total_changes(), 3);
    }

    #[test]
    fn test_diff_report_nonempty() {
        let mut a = SymbolTable::new();
        a.add_domain(DomainInfo::new("OldDomain", 5))
            .expect("add_domain should succeed");
        let mut b = SymbolTable::new();
        b.add_domain(DomainInfo::new("NewDomain", 5))
            .expect("add_domain should succeed");
        let diff = compare_symbol_tables(&a, &b);
        let report = diff.report();
        assert!(report.contains("Added") || report.contains("Removed"));
    }

    #[test]
    fn test_diff_summary_format() {
        let diff = OntologyDiff::new();
        let summary = diff.summary();
        assert!(summary.starts_with("OntologyDiff:"));
    }

    #[test]
    fn test_diff_entry_helpers() {
        let added = DiffEntry::Added("x".to_string());
        assert!(added.is_addition());
        assert!(!added.is_removal());
        assert!(!added.is_modification());

        let removed = DiffEntry::Removed("y".to_string());
        assert!(!removed.is_addition());
        assert!(removed.is_removal());
        assert!(!removed.is_modification());

        let modified = DiffEntry::Modified {
            before: "z(arity=1)".to_string(),
            after: "z(arity=2)".to_string(),
        };
        assert!(!modified.is_addition());
        assert!(!modified.is_removal());
        assert!(modified.is_modification());
    }
}
