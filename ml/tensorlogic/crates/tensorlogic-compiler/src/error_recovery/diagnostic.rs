//! Diagnostic primitives used by the tolerant compiler.
//!
//! A [`Diagnostic`] is a single, self-contained report describing a compilation
//! issue. Each diagnostic carries a [`Severity`], a human-readable message, an
//! optional source span, and an optional *expression index* — i.e. the index
//! of the offending expression inside the slice passed to
//! [`crate::error_recovery::TolerantCompiler::compile_program`].
//!
//! # Severity semantics
//!
//! * [`Severity::Fatal`] — Unrecoverable failure. Under the default
//!   [`crate::error_recovery::RecoveryStrategy::SkipOnFatal`] strategy the
//!   remaining expressions are NOT compiled.
//! * [`Severity::Error`] — The current expression is dropped (its slot becomes
//!   `None`) but compilation of the rest of the program continues.
//! * [`Severity::Warning`] — The current expression still compiles; the
//!   warning is merely recorded.
//! * [`Severity::Info`] — Informational message; compilation proceeds.
//!
//! # Example
//!
//! ```
//! use tensorlogic_compiler::error_recovery::{Diagnostic, Severity};
//!
//! let d = Diagnostic::error("arity mismatch").with_expression_index(2);
//! assert!(d.is_blocking());
//! assert_eq!(d.severity, Severity::Error);
//! ```

use std::fmt;

/// Lightweight span into a source program — a byte range `[start, end)` plus an
/// optional filename or logical label.
///
/// This type deliberately does not depend on any particular parser — it is a
/// minimal, self-contained substitute that can be produced by any upstream
/// frontend. The compiler never dereferences the span; it is only propagated
/// for diagnostic printing.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SourceSpan {
    /// Byte offset of the first character (inclusive).
    pub start: usize,
    /// Byte offset one past the last character (exclusive).
    pub end: usize,
    /// Logical source label (e.g. a file path or `"<stdin>"`); optional.
    pub source: Option<String>,
}

impl SourceSpan {
    /// Construct a span without a source label.
    pub fn new(start: usize, end: usize) -> Self {
        Self {
            start,
            end,
            source: None,
        }
    }

    /// Construct a span carrying a source label (e.g. a filename).
    pub fn with_source(start: usize, end: usize, source: impl Into<String>) -> Self {
        Self {
            start,
            end,
            source: Some(source.into()),
        }
    }

    /// Length of the span in bytes (`end − start`, saturating to zero).
    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    /// Returns `true` when the span is empty (length zero).
    pub fn is_empty(&self) -> bool {
        self.end <= self.start
    }
}

impl fmt::Display for SourceSpan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.source {
            Some(src) => write!(f, "{}:{}..{}", src, self.start, self.end),
            None => write!(f, "{}..{}", self.start, self.end),
        }
    }
}

/// Diagnostic severity, ordered from least to most serious.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Severity {
    /// Informational; does not affect compilation.
    Info,
    /// Non-blocking; the current expression still compiles successfully.
    Warning,
    /// Blocking for *this expression only*; siblings still compile.
    Error,
    /// Unrecoverable; the tolerant driver aborts (under default strategy).
    Fatal,
}

impl Severity {
    /// Returns `true` when a diagnostic of this severity drops the current
    /// expression (or the whole program).
    pub fn is_blocking(self) -> bool {
        matches!(self, Severity::Error | Severity::Fatal)
    }

    /// Short, user-facing label (`"info"`, `"warning"`, `"error"`, `"fatal"`).
    pub fn label(self) -> &'static str {
        match self {
            Severity::Info => "info",
            Severity::Warning => "warning",
            Severity::Error => "error",
            Severity::Fatal => "fatal",
        }
    }
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// A single diagnostic produced during tolerant compilation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    /// Severity of the diagnostic.
    pub severity: Severity,
    /// Human-readable message.
    pub message: String,
    /// Optional source location.
    pub location: Option<SourceSpan>,
    /// Index into the slice passed to the tolerant compiler, if the
    /// diagnostic is tied to one particular expression.
    pub expression_index: Option<usize>,
}

impl Diagnostic {
    /// Construct a diagnostic from scratch.
    pub fn new(severity: Severity, message: impl Into<String>) -> Self {
        Self {
            severity,
            message: message.into(),
            location: None,
            expression_index: None,
        }
    }

    /// Convenience constructor for [`Severity::Info`] diagnostics.
    pub fn info(message: impl Into<String>) -> Self {
        Self::new(Severity::Info, message)
    }

    /// Convenience constructor for [`Severity::Warning`] diagnostics.
    pub fn warning(message: impl Into<String>) -> Self {
        Self::new(Severity::Warning, message)
    }

    /// Convenience constructor for [`Severity::Error`] diagnostics.
    pub fn error(message: impl Into<String>) -> Self {
        Self::new(Severity::Error, message)
    }

    /// Convenience constructor for [`Severity::Fatal`] diagnostics.
    pub fn fatal(message: impl Into<String>) -> Self {
        Self::new(Severity::Fatal, message)
    }

    /// Attach an expression index (builder-style).
    pub fn with_expression_index(mut self, idx: usize) -> Self {
        self.expression_index = Some(idx);
        self
    }

    /// Attach a source span (builder-style).
    pub fn with_location(mut self, span: SourceSpan) -> Self {
        self.location = Some(span);
        self
    }

    /// Returns `true` when this diagnostic is blocking
    /// (i.e. [`Severity::Error`] or [`Severity::Fatal`]).
    pub fn is_blocking(&self) -> bool {
        self.severity.is_blocking()
    }

    /// Returns `true` when this diagnostic is fatal
    /// (i.e. [`Severity::Fatal`]).
    pub fn is_fatal(&self) -> bool {
        self.severity == Severity::Fatal
    }
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}]", self.severity)?;
        if let Some(idx) = self.expression_index {
            write!(f, "[expr#{}]", idx)?;
        }
        if let Some(span) = &self.location {
            write!(f, " ({})", span)?;
        }
        write!(f, " {}", self.message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn severity_ordering() {
        assert!(Severity::Info < Severity::Warning);
        assert!(Severity::Warning < Severity::Error);
        assert!(Severity::Error < Severity::Fatal);
    }

    #[test]
    fn severity_is_blocking() {
        assert!(!Severity::Info.is_blocking());
        assert!(!Severity::Warning.is_blocking());
        assert!(Severity::Error.is_blocking());
        assert!(Severity::Fatal.is_blocking());
    }

    #[test]
    fn diagnostic_builders() {
        let d = Diagnostic::error("boom").with_expression_index(7);
        assert_eq!(d.severity, Severity::Error);
        assert_eq!(d.message, "boom");
        assert_eq!(d.expression_index, Some(7));
        assert!(d.is_blocking());
        assert!(!d.is_fatal());
    }

    #[test]
    fn diagnostic_display_includes_index_and_span() {
        let d = Diagnostic::warning("possible issue")
            .with_expression_index(3)
            .with_location(SourceSpan::with_source(10, 20, "main.tl"));
        let s = format!("{}", d);
        assert!(s.contains("warning"));
        assert!(s.contains("expr#3"));
        assert!(s.contains("main.tl:10..20"));
        assert!(s.contains("possible issue"));
    }

    #[test]
    fn source_span_basics() {
        let sp = SourceSpan::new(3, 7);
        assert_eq!(sp.len(), 4);
        assert!(!sp.is_empty());
        let empty = SourceSpan::new(5, 5);
        assert!(empty.is_empty());
    }
}
