//! A uniform, structured diagnostic type shared by every subcommand that reports problems found
//! in a document rather than in the command line itself (`lint`, `validate`, and the syntax
//! errors surfaced by `fmt`, `transform`, `query`, `canon`, `convert`, `hdt` and `sparql`).
//!
//! Rendered one per line as `path:line:col: severity: [code] message`, with every part that is
//! unknown (no position, no machine-readable code) simply omitted — this is deliberately close
//! to the `file:line:col: message` convention `xmllint`, `rustc` and most other command-line
//! tools share, so that editors and CI log scrapers that already understand that shape understand
//! `oxixml` too.

use std::fmt;

/// How serious a [`Diagnostic`] is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// Worth reporting, but not by itself a reason to fail.
    Warning,
    /// A well-formedness or validity violation.
    Error,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Warning => "warning",
            Self::Error => "error",
        })
    }
}

/// One reported problem, with as much location information as the producing engine tracked.
#[derive(Debug, Clone)]
pub struct Diagnostic {
    /// The input's display path (its argument as given on the command line, or `<stdin>`).
    pub path: String,
    /// The 1-based line number, when the producing engine tracks positions.
    pub line: Option<u64>,
    /// The 1-based column number, when the producing engine tracks positions.
    pub column: Option<u64>,
    /// How serious this is.
    pub severity: Severity,
    /// A machine-readable identifier for the rule that was violated (a SAX/XML error has none; a
    /// schema-validity error has a `cvc-…` code; a DTD or RELAX NG error may name its own kind).
    pub code: Option<String>,
    /// A human-readable explanation.
    pub message: String,
}

impl Diagnostic {
    /// Builds an error-severity diagnostic with no known position and no code — the common case
    /// for engines (RELAX NG, DTD, Schematron) that report a message and, at best, a node path
    /// rather than a line and column.
    #[must_use]
    pub fn error(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            line: None,
            column: None,
            severity: Severity::Error,
            code: None,
            message: message.into(),
        }
    }

    /// Builds a warning-severity diagnostic: worth reporting, but not by itself a reason for the
    /// producing subcommand to report failure.
    #[must_use]
    pub fn warning(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            line: None,
            column: None,
            severity: Severity::Warning,
            code: None,
            message: message.into(),
        }
    }

    /// Whether this diagnostic is severe enough to fail the check that produced it.
    #[must_use]
    pub const fn is_error(&self) -> bool {
        matches!(self.severity, Severity::Error)
    }

    /// Builder-style: attaches a 1-based line/column position.
    #[must_use]
    pub fn at(mut self, line: u64, column: u64) -> Self {
        self.line = Some(line);
        self.column = Some(column);
        self
    }

    /// Builder-style: attaches a machine-readable rule identifier.
    #[must_use]
    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.code = Some(code.into());
        self
    }
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.path)?;
        if let (Some(line), Some(column)) = (self.line, self.column) {
            write!(f, ":{line}:{column}")?;
        }
        write!(f, ": {}: ", self.severity)?;
        if let Some(code) = &self.code {
            write!(f, "[{code}] ")?;
        }
        write!(f, "{}", self.message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_every_part_when_present() {
        let diagnostic = Diagnostic::error("a.xml", "duplicate attribute")
            .at(3, 12)
            .with_code("cvc-complex-type.2.4.a");
        assert_eq!(
            diagnostic.to_string(),
            "a.xml:3:12: error: [cvc-complex-type.2.4.a] duplicate attribute"
        );
    }

    #[test]
    fn omits_position_and_code_when_absent() {
        let diagnostic = Diagnostic::error("a.xml", "not valid");
        assert_eq!(diagnostic.to_string(), "a.xml: error: not valid");
    }

    #[test]
    fn warnings_render_their_own_severity() {
        let diagnostic = Diagnostic::warning("a.xml", "deprecated construct");
        assert_eq!(
            diagnostic.to_string(),
            "a.xml: warning: deprecated construct"
        );
        assert!(!diagnostic.is_error());
    }

    #[test]
    fn errors_are_errors() {
        assert!(Diagnostic::error("a.xml", "not valid").is_error());
    }
}
