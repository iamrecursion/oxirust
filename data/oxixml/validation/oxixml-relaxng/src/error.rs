//! Errors reported while compiling a RELAX NG schema and while validating an
//! instance against it.
//!
//! The crate separates the two phases the way [ISO/IEC 19757-2][spec] does:
//!
//! * a schema is either *correct* or not — an incorrect schema yields a
//!   [`SchemaError`], and no validation can be attempted;
//! * an instance is either *valid* or not with respect to a correct schema —
//!   an invalid instance yields a list of [`RelaxNgError`]s.
//!
//! [spec]: https://relaxng.org/spec-20011203.html

use std::fmt;

/// The position a diagnostic refers to, when one is available.
///
/// Positions are 1-based, as in every XML error report. A position of
/// `(0, 0)` means "unknown".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, PartialOrd, Ord, Hash)]
pub struct Position {
    /// The 1-based line number, or `0` when unknown.
    pub line: u32,
    /// The 1-based column number, or `0` when unknown.
    pub column: u32,
}

impl Position {
    /// The "unknown position" marker.
    pub const UNKNOWN: Self = Self { line: 0, column: 0 };

    /// Builds a position from a 1-based line and column.
    #[must_use]
    pub const fn new(line: u32, column: u32) -> Self {
        Self { line, column }
    }

    /// Returns `true` when neither the line nor the column is known.
    #[must_use]
    pub const fn is_unknown(self) -> bool {
        self.line == 0 && self.column == 0
    }
}

impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_unknown() {
            f.write_str("<unknown position>")
        } else if self.column == 0 {
            write!(f, "line {}", self.line)
        } else {
            write!(f, "line {}, column {}", self.line, self.column)
        }
    }
}

/// A reason why a document is not a *correct* RELAX NG schema.
///
/// This covers every failure of the schema-compilation pipeline: the XML (or
/// compact) syntax itself, the full-syntax grammar of
/// [§3](https://relaxng.org/spec-20011203.html#syntax), the constraints attached
/// to the simplification rules of
/// [§4](https://relaxng.org/spec-20011203.html#simplification), and the
/// restrictions of [§7](https://relaxng.org/spec-20011203.html#restrictions).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchemaError {
    kind: SchemaErrorKind,
    message: String,
    position: Position,
    /// The section of the specification the failed rule belongs to, e.g. `"7.3"`.
    section: &'static str,
}

impl SchemaError {
    pub(crate) fn new(
        kind: SchemaErrorKind,
        section: &'static str,
        message: impl Into<String>,
    ) -> Self {
        Self {
            kind,
            message: message.into(),
            position: Position::UNKNOWN,
            section,
        }
    }

    pub(crate) fn at(mut self, position: Position) -> Self {
        if self.position.is_unknown() {
            self.position = position;
        }
        self
    }

    /// The broad category the error falls into.
    #[must_use]
    pub const fn kind(&self) -> SchemaErrorKind {
        self.kind
    }

    /// A human-readable description of what is wrong.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Where in the schema the problem was detected.
    #[must_use]
    pub const fn position(&self) -> Position {
        self.position
    }

    /// The specification section whose rule was violated (`"3"`, `"4.16"`, …).
    #[must_use]
    pub const fn section(&self) -> &'static str {
        self.section
    }
}

impl fmt::Display for SchemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.position.is_unknown() {
            write!(f, "[RELAX NG \u{a7}{}] {}", self.section, self.message)
        } else {
            write!(
                f,
                "{}: [RELAX NG \u{a7}{}] {}",
                self.position, self.section, self.message
            )
        }
    }
}

impl std::error::Error for SchemaError {}

/// The broad category a [`SchemaError`] falls into.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SchemaErrorKind {
    /// The schema is not well-formed XML, or the compact syntax did not parse.
    Syntax,
    /// The schema does not match the full syntax of §3.
    Grammar,
    /// A constraint attached to one of the simplification rules of §4 failed.
    Simplification,
    /// One of the restrictions of §7 is violated.
    Restriction,
    /// A datatype library, datatype or parameter is unknown or misused.
    Datatype,
    /// An `include` or `externalRef` could not be resolved or retrieved.
    Resource,
}

impl fmt::Display for SchemaErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Syntax => "syntax error",
            Self::Grammar => "grammar error",
            Self::Simplification => "simplification error",
            Self::Restriction => "restriction violation",
            Self::Datatype => "datatype error",
            Self::Resource => "resource error",
        })
    }
}

/// A reason why an instance document is not valid against a schema.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelaxNgError {
    kind: ValidationErrorKind,
    message: String,
    position: Position,
}

impl RelaxNgError {
    pub(crate) fn new(kind: ValidationErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            position: Position::UNKNOWN,
        }
    }

    /// The broad category the error falls into.
    #[must_use]
    pub const fn kind(&self) -> ValidationErrorKind {
        self.kind
    }

    /// A human-readable description of what is wrong.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Where in the instance the problem was detected.
    #[must_use]
    pub const fn position(&self) -> Position {
        self.position
    }
}

impl fmt::Display for RelaxNgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.position.is_unknown() {
            f.write_str(&self.message)
        } else {
            write!(f, "{}: {}", self.position, self.message)
        }
    }
}

impl std::error::Error for RelaxNgError {}

/// The broad category a [`RelaxNgError`] falls into.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ValidationErrorKind {
    /// The instance is not well-formed XML.
    NotWellFormed,
    /// An element appeared where the schema does not allow it.
    UnexpectedElement,
    /// An attribute appeared where the schema does not allow it.
    UnexpectedAttribute,
    /// Character data appeared where the schema does not allow it.
    UnexpectedText,
    /// The content of an element ended while the schema required more.
    IncompleteContent,
    /// A string did not match the datatype (or value) required by the schema.
    InvalidValue,
}

impl fmt::Display for ValidationErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::NotWellFormed => "not well-formed",
            Self::UnexpectedElement => "unexpected element",
            Self::UnexpectedAttribute => "unexpected attribute",
            Self::UnexpectedText => "unexpected text",
            Self::IncompleteContent => "incomplete content",
            Self::InvalidValue => "invalid value",
        })
    }
}

pub(crate) fn syntax(message: impl Into<String>) -> SchemaError {
    SchemaError::new(SchemaErrorKind::Syntax, "2", message)
}

pub(crate) fn grammar(message: impl Into<String>) -> SchemaError {
    SchemaError::new(SchemaErrorKind::Grammar, "3", message)
}

pub(crate) fn simplify(section: &'static str, message: impl Into<String>) -> SchemaError {
    SchemaError::new(SchemaErrorKind::Simplification, section, message)
}

pub(crate) fn restriction(section: &'static str, message: impl Into<String>) -> SchemaError {
    SchemaError::new(SchemaErrorKind::Restriction, section, message)
}

pub(crate) fn datatype(section: &'static str, message: impl Into<String>) -> SchemaError {
    SchemaError::new(SchemaErrorKind::Datatype, section, message)
}

pub(crate) fn resource(section: &'static str, message: impl Into<String>) -> SchemaError {
    SchemaError::new(SchemaErrorKind::Resource, section, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn position_display() {
        assert_eq!(Position::UNKNOWN.to_string(), "<unknown position>");
        assert_eq!(Position::new(3, 0).to_string(), "line 3");
        assert_eq!(Position::new(3, 7).to_string(), "line 3, column 7");
    }

    #[test]
    fn schema_error_display_carries_section() {
        let err = grammar("bad").at(Position::new(1, 2));
        assert!(err.to_string().contains("\u{a7}3"));
        assert_eq!(err.kind(), SchemaErrorKind::Grammar);
        assert_eq!(err.message(), "bad");
    }

    #[test]
    fn first_position_wins() {
        let err = grammar("bad")
            .at(Position::new(1, 2))
            .at(Position::new(9, 9));
        assert_eq!(err.position(), Position::new(1, 2));
    }
}
