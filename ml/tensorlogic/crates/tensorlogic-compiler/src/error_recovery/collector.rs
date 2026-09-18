//! Thread-safe diagnostic collector used by the tolerant compiler.
//!
//! The collector is a shareable sink for [`Diagnostic`] values produced during
//! tolerant compilation. It stores diagnostics in **insertion order** and
//! allows:
//!
//! * concurrent push from multiple worker threads via [`Arc<Mutex<_>>`];
//! * querying by severity (`errors()`, `warnings()`, `fatals()`, ...);
//! * filtering by expression index (`for_expression()`);
//! * clearing and replacing the buffer wholesale (for test harnesses).
//!
//! # Example
//!
//! ```
//! use tensorlogic_compiler::error_recovery::{
//!     Diagnostic, DiagnosticCollector, Severity,
//! };
//!
//! let c = DiagnosticCollector::new();
//! c.push(Diagnostic::warning("deprecated"));
//! c.push(Diagnostic::error("arity mismatch").with_expression_index(1));
//!
//! assert_eq!(c.len(), 2);
//! assert_eq!(c.count_of(Severity::Error), 1);
//! assert!(c.has_blocking());
//! ```

use std::sync::{Arc, Mutex};

use super::diagnostic::{Diagnostic, Severity};

/// Shared, thread-safe collector for [`Diagnostic`] values.
///
/// Cloning a `DiagnosticCollector` yields another handle to the *same*
/// underlying buffer (cheap `Arc` clone) — this is intentional: it allows
/// passing the collector into worker tasks without ceremony.
#[derive(Debug, Clone, Default)]
pub struct DiagnosticCollector {
    inner: Arc<Mutex<Vec<Diagnostic>>>,
}

impl DiagnosticCollector {
    /// Construct an empty collector.
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct a collector preseeded with the given diagnostics.
    pub fn with_diagnostics(initial: Vec<Diagnostic>) -> Self {
        Self {
            inner: Arc::new(Mutex::new(initial)),
        }
    }

    /// Push a single diagnostic. On a poisoned mutex the diagnostic is still
    /// pushed using the recovered guard — partial progress is always
    /// preserved.
    pub fn push(&self, diag: Diagnostic) {
        let mut guard = match self.inner.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        guard.push(diag);
    }

    /// Push multiple diagnostics in one shot, preserving their order.
    pub fn extend<I: IntoIterator<Item = Diagnostic>>(&self, diags: I) {
        let mut guard = match self.inner.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        guard.extend(diags);
    }

    /// Total number of diagnostics collected so far.
    pub fn len(&self) -> usize {
        self.snapshot_raw().len()
    }

    /// Returns `true` when no diagnostics have been collected.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clone out the full diagnostic list in insertion order.
    pub fn snapshot(&self) -> Vec<Diagnostic> {
        self.snapshot_raw()
    }

    fn snapshot_raw(&self) -> Vec<Diagnostic> {
        match self.inner.lock() {
            Ok(g) => g.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        }
    }

    /// All diagnostics of the given severity, in insertion order.
    pub fn of_severity(&self, severity: Severity) -> Vec<Diagnostic> {
        self.snapshot_raw()
            .into_iter()
            .filter(|d| d.severity == severity)
            .collect()
    }

    /// All diagnostics whose severity is at least `min` (inclusive), in
    /// insertion order.
    pub fn at_least(&self, min: Severity) -> Vec<Diagnostic> {
        self.snapshot_raw()
            .into_iter()
            .filter(|d| d.severity >= min)
            .collect()
    }

    /// Count of diagnostics of a specific severity.
    pub fn count_of(&self, severity: Severity) -> usize {
        self.snapshot_raw()
            .iter()
            .filter(|d| d.severity == severity)
            .count()
    }

    /// Convenience accessor: all [`Severity::Error`] diagnostics.
    pub fn errors(&self) -> Vec<Diagnostic> {
        self.of_severity(Severity::Error)
    }

    /// Convenience accessor: all [`Severity::Warning`] diagnostics.
    pub fn warnings(&self) -> Vec<Diagnostic> {
        self.of_severity(Severity::Warning)
    }

    /// Convenience accessor: all [`Severity::Fatal`] diagnostics.
    pub fn fatals(&self) -> Vec<Diagnostic> {
        self.of_severity(Severity::Fatal)
    }

    /// Convenience accessor: all [`Severity::Info`] diagnostics.
    pub fn infos(&self) -> Vec<Diagnostic> {
        self.of_severity(Severity::Info)
    }

    /// Returns `true` iff at least one diagnostic is blocking
    /// (Error or Fatal).
    pub fn has_blocking(&self) -> bool {
        self.snapshot_raw().iter().any(Diagnostic::is_blocking)
    }

    /// Returns `true` iff at least one Fatal diagnostic exists.
    pub fn has_fatal(&self) -> bool {
        self.snapshot_raw().iter().any(Diagnostic::is_fatal)
    }

    /// All diagnostics attached to the given expression index, in insertion
    /// order.
    pub fn for_expression(&self, idx: usize) -> Vec<Diagnostic> {
        self.snapshot_raw()
            .into_iter()
            .filter(|d| d.expression_index == Some(idx))
            .collect()
    }

    /// Drain the buffer, returning all diagnostics and leaving the collector
    /// empty.
    pub fn drain(&self) -> Vec<Diagnostic> {
        let mut guard = match self.inner.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        std::mem::take(&mut *guard)
    }

    /// Reset the collector to empty.
    pub fn clear(&self) {
        let _ = self.drain();
    }
}

#[cfg(test)]
mod tests {
    use super::super::diagnostic::{Diagnostic, Severity};
    use super::*;

    #[test]
    fn push_preserves_insertion_order() {
        let c = DiagnosticCollector::new();
        c.push(Diagnostic::info("a"));
        c.push(Diagnostic::warning("b"));
        c.push(Diagnostic::error("c"));
        let snap = c.snapshot();
        assert_eq!(snap.len(), 3);
        assert_eq!(snap[0].message, "a");
        assert_eq!(snap[1].message, "b");
        assert_eq!(snap[2].message, "c");
    }

    #[test]
    fn severity_filters() {
        let c = DiagnosticCollector::new();
        c.push(Diagnostic::info("i"));
        c.push(Diagnostic::warning("w1"));
        c.push(Diagnostic::warning("w2"));
        c.push(Diagnostic::error("e"));
        c.push(Diagnostic::fatal("f"));

        assert_eq!(c.len(), 5);
        assert_eq!(c.count_of(Severity::Warning), 2);
        assert_eq!(c.errors().len(), 1);
        assert_eq!(c.warnings().len(), 2);
        assert_eq!(c.fatals().len(), 1);
        assert_eq!(c.infos().len(), 1);
        assert!(c.has_blocking());
        assert!(c.has_fatal());
        assert_eq!(c.at_least(Severity::Error).len(), 2);
    }

    #[test]
    fn for_expression_filter() {
        let c = DiagnosticCollector::new();
        c.push(Diagnostic::error("e0").with_expression_index(0));
        c.push(Diagnostic::error("e1").with_expression_index(1));
        c.push(Diagnostic::warning("w1").with_expression_index(1));
        c.push(Diagnostic::info("no-idx"));

        assert_eq!(c.for_expression(0).len(), 1);
        assert_eq!(c.for_expression(1).len(), 2);
        assert_eq!(c.for_expression(99).len(), 0);
    }

    #[test]
    fn clone_shares_underlying_buffer() {
        let c1 = DiagnosticCollector::new();
        let c2 = c1.clone();
        c1.push(Diagnostic::error("shared"));
        assert_eq!(c2.len(), 1);
    }

    #[test]
    fn drain_empties_collector() {
        let c = DiagnosticCollector::with_diagnostics(vec![
            Diagnostic::error("a"),
            Diagnostic::warning("b"),
        ]);
        assert_eq!(c.len(), 2);
        let drained = c.drain();
        assert_eq!(drained.len(), 2);
        assert!(c.is_empty());
    }

    #[test]
    fn extend_appends_multiple() {
        let c = DiagnosticCollector::new();
        c.extend(vec![
            Diagnostic::info("x"),
            Diagnostic::warning("y"),
            Diagnostic::error("z"),
        ]);
        assert_eq!(c.len(), 3);
        assert_eq!(c.snapshot()[2].message, "z");
    }
}
