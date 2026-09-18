//! [`XsltError`] — every failure this crate reports, under the error code a conforming
//! processor would raise.
//!
//! [XSLT 3.0 §2.10](https://www.w3.org/TR/xslt-30/#errors) names each condition with a
//! `QName` in the `err:` namespace — `XTSE0010` for a malformed stylesheet, `XTDE0640` for a
//! circular reference, `XTTE0505` for a type mismatch. The code is the part a test suite and a
//! user's `xsl:try` both key on, so it is carried structurally rather than folded into the
//! message.
//!
//! Errors raised by [`oxixml_xpath`], [`oxixml_fn`] and [`oxixml_infoset`] keep their own
//! codes when they are converted: an `XPTY0004` raised inside a `select` expression stays
//! `XPTY0004`.

use oxixml_fn::FnError;
use oxixml_infoset::{Sequence, SourcePosition, XdmError};
use oxixml_xpath::XPathError;
use std::error::Error;
use std::fmt;

/// The namespace every XSLT and XPath error code lives in.
pub const ERROR_NAMESPACE: &str = "http://www.w3.org/2005/xqt-errors";

/// Whether a failure was detected while compiling the stylesheet or while running it.
///
/// # Examples
///
/// ```
/// use oxixml_xslt::{ErrorPhase, XsltError};
///
/// assert_eq!(XsltError::stat1c("XTSE0010", "bad element").phase(), ErrorPhase::Static);
/// assert_eq!(XsltError::dynamic("XTDE0700", "unbound").phase(), ErrorPhase::Dynamic);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorPhase {
    /// Detected while compiling — a static error.
    Static,
    /// Detected while transforming — a dynamic error.
    Dynamic,
}

impl fmt::Display for ErrorPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Static => "static",
            Self::Dynamic => "dynamic",
        })
    }
}

/// A failure to compile or to run a stylesheet.
///
/// # Examples
///
/// ```
/// use oxixml_xslt::{ErrorPhase, XsltError};
///
/// let failure = XsltError::stat1c("XTSE0010", "`xsl:nonesuch` is not an XSLT element");
/// assert_eq!(failure.code(), Some("XTSE0010"));
/// assert_eq!(failure.phase(), ErrorPhase::Static);
/// assert!(failure.to_string().starts_with("XTSE0010"));
/// ```
#[derive(Debug, Clone)]
pub struct XsltError {
    code: Option<String>,
    namespace: Option<String>,
    message: String,
    phase: ErrorPhase,
    /// The value the failing construct associated with the error, which an `xsl:catch` body
    /// reads as `$err:value` — [XSLT 3.0 §8.3](https://www.w3.org/TR/xslt-30/#element-catch).
    /// Only a construct that *has* such a value carries one: a terminating `xsl:message` passes
    /// its content, and every condition the specification names on its own passes nothing.
    value: Option<Sequence>,
    /// Where in the stylesheet the failing instruction is written, which an `xsl:catch` body
    /// reads as `$err:line-number` and `$err:column-number`. It is present only when the
    /// stylesheet was parsed with source positions recorded and the failure travelled through
    /// an instruction that carries them.
    position: Option<SourcePosition>,
}

impl XsltError {
    /// Builds an error under `code`, detected at compile time.
    ///
    /// The method is spelled `stat1c` because `static` is a Rust keyword.
    #[must_use]
    pub fn stat1c(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: Some(code.into()),
            namespace: Some(ERROR_NAMESPACE.to_owned()),
            message: message.into(),
            phase: ErrorPhase::Static,
            value: None,
            position: None,
        }
    }

    /// Builds an error under `code`, detected at run time.
    #[must_use]
    pub fn dynamic(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: Some(code.into()),
            namespace: Some(ERROR_NAMESPACE.to_owned()),
            message: message.into(),
            phase: ErrorPhase::Dynamic,
            value: None,
            position: None,
        }
    }

    /// Builds an error whose name is not in the `err:` namespace — what `xsl:message` with
    /// `terminate="yes"` and a user's `fn:error` call produce.
    #[must_use]
    pub fn named(
        namespace: Option<String>,
        local: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            code: Some(local.into()),
            namespace,
            message: message.into(),
            phase: ErrorPhase::Dynamic,
            value: None,
            position: None,
        }
    }

    /// The local part of the error code, such as `XTSE0010`.
    #[must_use]
    pub fn code(&self) -> Option<&str> {
        self.code.as_deref()
    }

    /// The namespace of the error code.
    #[must_use]
    pub fn namespace(&self) -> Option<&str> {
        self.namespace.as_deref()
    }

    /// The human-readable description.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Whether the failure was static or dynamic.
    #[must_use]
    pub const fn phase(&self) -> ErrorPhase {
        self.phase
    }

    /// Whether the error carries exactly this code.
    #[must_use]
    pub fn is(&self, code: &str) -> bool {
        self.code.as_deref() == Some(code)
    }

    /// Re-labels the error as static.
    #[must_use]
    pub fn as_static(mut self) -> Self {
        self.phase = ErrorPhase::Static;
        self
    }

    /// Attaches the value the failing construct associated with the error.
    ///
    /// [XSLT 3.0 §8.3](https://www.w3.org/TR/xslt-30/#element-catch) hands it to an `xsl:catch`
    /// body as `$err:value`; a terminating `xsl:message` supplies the sequence its content
    /// constructed, which the suite's message-0501 reads back as an element with a comment
    /// child, so the nodes have to survive rather than being flattened into the description.
    #[must_use]
    pub(crate) fn with_value(mut self, value: Sequence) -> Self {
        self.value = Some(value);
        self
    }

    /// The value an `xsl:catch` body sees as `$err:value`, when the failing construct had one.
    #[must_use]
    pub(crate) fn value(&self) -> Option<&Sequence> {
        self.value.as_ref()
    }

    /// Records that the error passed out of an instruction written at `position`.
    ///
    /// The *first* attachment wins, because an error travels outwards: the instruction that
    /// raised it attaches its own position before any of the instructions containing it get the
    /// chance, so keeping the first is what makes the position the innermost one. The position an
    /// `xsl:catch` body is meant to see is that of the failing instruction rather than of the
    /// construct that happens to catch it, which the suite's try-022 settles: with an
    /// `xsl:value-of` that divides by zero inside an `xsl:result-document` inside an `xsl:try`,
    /// the line it asserts is the `xsl:value-of`'s.
    #[must_use]
    pub(crate) fn at(mut self, position: SourcePosition) -> Self {
        if self.position.is_none() {
            self.position = Some(position);
        }
        self
    }

    /// Where the failing instruction is written, when the stylesheet recorded source positions.
    #[must_use]
    pub(crate) const fn position(&self) -> Option<SourcePosition> {
        self.position
    }

    /// Prefixes the message with `context`, keeping the code.
    #[must_use]
    pub fn contextualize(mut self, context: &str) -> Self {
        self.message = format!("{context}: {}", self.message);
        self
    }
}

impl fmt::Display for XsltError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.code {
            Some(code) => write!(f, "{code}: {}", self.message),
            None => f.write_str(&self.message),
        }
    }
}

impl Error for XsltError {}

impl From<XPathError> for XsltError {
    fn from(error: XPathError) -> Self {
        let phase = if error.is_static() {
            ErrorPhase::Static
        } else {
            ErrorPhase::Dynamic
        };
        Self {
            code: Some(error.name().local_name().to_owned()),
            namespace: error.name().namespace().map(str::to_owned),
            message: error.message().to_owned(),
            phase,
            value: None,
            position: None,
        }
    }
}

impl From<FnError> for XsltError {
    fn from(error: FnError) -> Self {
        Self {
            code: Some(error.name().local_name().to_owned()),
            namespace: error.name().namespace().map(str::to_owned),
            message: error.description().to_owned(),
            phase: ErrorPhase::Dynamic,
            value: None,
            position: None,
        }
    }
}

impl From<XdmError> for XsltError {
    fn from(error: XdmError) -> Self {
        Self {
            code: Some(error.code().to_string()),
            namespace: Some(ERROR_NAMESPACE.to_owned()),
            message: error.message().to_owned(),
            phase: ErrorPhase::Dynamic,
            value: None,
            position: None,
        }
    }
}

/// The result of every fallible operation in this crate.
pub type XsltResult<T> = Result<T, XsltError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_and_phases_survive_construction() {
        let failure = XsltError::stat1c("XTSE0010", "bad");
        assert!(failure.is("XTSE0010"));
        assert_eq!(failure.namespace(), Some(ERROR_NAMESPACE));
        assert_eq!(failure.phase(), ErrorPhase::Static);
        assert_eq!(failure.message(), "bad");

        let renamed = XsltError::named(Some("urn:x".to_owned()), "oops", "user error");
        assert_eq!(renamed.namespace(), Some("urn:x"));
        assert_eq!(renamed.code(), Some("oops"));
    }

    #[test]
    fn xpath_errors_keep_their_code() {
        let error = XPathError::new(oxixml_xpath::XPathErrorCode::Xpty0004, "type");
        let converted = XsltError::from(error);
        assert_eq!(converted.code(), Some("XPTY0004"));
        assert_eq!(converted.phase(), ErrorPhase::Dynamic);
    }

    #[test]
    fn context_is_prefixed_without_losing_the_code() {
        let failure = XsltError::dynamic("XTDE0700", "unbound").contextualize("in xsl:value-of");
        assert_eq!(failure.code(), Some("XTDE0700"));
        assert_eq!(failure.message(), "in xsl:value-of: unbound");
        assert_eq!(failure.to_string(), "XTDE0700: in xsl:value-of: unbound");
    }
}
