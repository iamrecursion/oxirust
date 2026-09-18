//! Errors: schema-assembly failures and instance-validation failures.
//!
//! Both carry the *constraint identifier* the W3C Recommendation gives the rule
//! that was broken (`src-element.1`, `cvc-complex-type.2.4.a`, …), so a caller
//! can key behaviour off the exact rule rather than off message text.

use std::fmt;

/// A schema-assembly error: the schema document is not a conforming schema
/// document, or the components it maps onto do not satisfy a *schema component
/// constraint*.
///
/// The [`code`](SchemaError::code) is the identifier used by the
/// Recommendation — `src-…` for a "schema representation constraint",
/// `st-props-…`/`ct-props-…`/`cos-…`/`derivation-ok-…` for a "schema component
/// constraint" — which is also what Xerces reports.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchemaError {
    code: String,
    message: String,
    location: Option<String>,
}

impl SchemaError {
    /// Builds an error for the constraint `code`.
    #[must_use]
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            location: None,
        }
    }

    /// Attaches a human-readable location (usually a document URI plus an
    /// element path).
    #[must_use]
    pub fn at(mut self, location: impl Into<String>) -> Self {
        self.location = Some(location.into());
        self
    }

    /// Attaches `location` only if none is set yet.
    #[must_use]
    pub fn or_at(self, location: impl Into<String>) -> Self {
        if self.location.is_some() {
            self
        } else {
            self.at(location)
        }
    }

    /// The constraint identifier, for example `src-resolve`.
    #[must_use]
    pub fn code(&self) -> &str {
        &self.code
    }

    /// The human-readable explanation.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Where the problem is, when known.
    #[must_use]
    pub fn location(&self) -> Option<&str> {
        self.location.as_deref()
    }
}

impl fmt::Display for SchemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.code, self.message)?;
        if let Some(location) = &self.location {
            write!(f, " (at {location})")?;
        }
        Ok(())
    }
}

impl std::error::Error for SchemaError {}

/// A single instance-validation failure.
///
/// The [`code`](ValidationError::code) is the `cvc-…` identifier of the
/// *validation rule* that failed, and [`path`](ValidationError::path) is a
/// simplified XPath locating the offending node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationError {
    code: String,
    message: String,
    path: String,
}

impl ValidationError {
    /// Builds a validation error.
    #[must_use]
    pub fn new(
        code: impl Into<String>,
        path: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            path: path.into(),
        }
    }

    /// The `cvc-…` identifier of the rule that failed.
    #[must_use]
    pub fn code(&self) -> &str {
        &self.code
    }

    /// The human-readable explanation.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }

    /// A simplified XPath to the offending node, such as `/a/b[2]/@c`.
    #[must_use]
    pub fn path(&self) -> &str {
        &self.path
    }
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: [{}] {}", self.path, self.code, self.message)
    }
}

impl std::error::Error for ValidationError {}

/// The result type of schema assembly.
pub type SchemaResult<T> = Result<T, SchemaError>;
