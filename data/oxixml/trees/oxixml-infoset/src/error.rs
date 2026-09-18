//! [`XdmError`] and the [`ErrorCode`]s of the XPath/XQuery error vocabulary that the data
//! model itself can raise.

use std::error::Error;
use std::fmt;

/// One of the [XPath and XQuery error codes](https://www.w3.org/TR/xpath-functions-31/#error-summary)
/// that operations on the data model can raise.
///
/// The data-model layer deliberately raises only the handful of codes that are *properties of
/// the model* rather than of an expression language: type errors on atomization, cast failures,
/// out-of-bounds array subscripts, and so on. Everything an evaluator adds on top (static
/// errors, dynamic context errors, …) belongs to the query engine, not here.
///
/// # Examples
///
/// ```
/// use oxixml_infoset::ErrorCode;
///
/// assert_eq!(ErrorCode::Xpty0004.as_str(), "XPTY0004");
/// assert_eq!(ErrorCode::Foay0001.to_string(), "FOAY0001");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ErrorCode {
    /// `XPTY0004` — the value supplied has the wrong type for the context it is used in.
    Xpty0004,
    /// `FORG0001` — invalid value for a cast or constructor function.
    Forg0001,
    /// `FORG0006` — invalid argument type; raised by the effective boolean value rules for a
    /// sequence that is neither empty, nor node-headed, nor a suitable singleton.
    Forg0006,
    /// `FOTY0012` — the argument node does not have a typed value (an element with a complex
    /// type of element-only content, or a nilled element).
    Foty0012,
    /// `FOTY0013` — the argument to `fn:data` contains a function item, which cannot be
    /// atomized.
    Foty0013,
    /// `FOTY0014` — the argument to `fn:string` is a function item, which has no string value.
    Foty0014,
    /// `FOTY0015` — an argument to `fn:deep-equal` contains a function item that is neither a
    /// map nor an array; such items have no equality beyond identity.
    Foty0015,
    /// `FOAY0001` — array index out of bounds.
    Foay0001,
    /// `FOCH0002` — unsupported collation.
    Foch0002,
    /// `XQDY0137` — no two keys in a map may be the same key.
    Xqdy0137,
    /// `FODC0006` — the string supplied is not a well-formed XML document, or the tree under
    /// construction violates the data model's structural constraints.
    Fodc0006,
    /// `FODT0001` — overflow or underflow in a date/time operation: a lexical form that is
    /// syntactically a date, time or Gregorian value but names a point outside the timeline
    /// this implementation can represent, or an arithmetic result that leaves it.
    ///
    /// This is deliberately distinct from [`ErrorCode::Forg0001`]: F&O 3.1 §10.1.1 makes a
    /// *lexically invalid* value `FORG0001` and an *out-of-range* one `FODT0001`.
    Fodt0001,
    /// `FODT0002` — overflow or underflow in a duration operation, the duration counterpart of
    /// [`ErrorCode::Fodt0001`].
    Fodt0002,
    /// `FOCA0002` — invalid lexical value: the string is not in the lexical space of the type a
    /// *function* (rather than a cast) requires, as `fn:QName` and `fn:resolve-QName` do for
    /// the `xs:QName` production.
    Foca0002,
}

impl ErrorCode {
    /// The bare error code, as it appears in the specification's error summary.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Xpty0004 => "XPTY0004",
            Self::Forg0001 => "FORG0001",
            Self::Forg0006 => "FORG0006",
            Self::Foty0012 => "FOTY0012",
            Self::Foty0013 => "FOTY0013",
            Self::Foty0014 => "FOTY0014",
            Self::Foty0015 => "FOTY0015",
            Self::Foay0001 => "FOAY0001",
            Self::Foch0002 => "FOCH0002",
            Self::Xqdy0137 => "XQDY0137",
            Self::Fodc0006 => "FODC0006",
            Self::Fodt0001 => "FODT0001",
            Self::Fodt0002 => "FODT0002",
            Self::Foca0002 => "FOCA0002",
        }
    }

    /// The error's [`ExpandedName`](oxixml_qname::ExpandedName)-shaped namespace name, which is
    /// always the standard error namespace `http://www.w3.org/2005/xqt-errors`.
    #[must_use]
    pub const fn namespace(self) -> &'static str {
        "http://www.w3.org/2005/xqt-errors"
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// An error raised by an operation on the XDM data model.
///
/// Every `XdmError` carries the specification [`ErrorCode`] an XPath or XQuery processor would
/// report, plus a human-readable explanation. The code is what a `try`/`catch` in the query
/// language matches on; the message is for diagnostics only and is not part of the contract.
///
/// # Examples
///
/// ```
/// use oxixml_infoset::{AtomicType, AtomicValue, ErrorCode};
///
/// let error = AtomicValue::parse(AtomicType::Integer, "not-a-number")
///     .expect_err("`not-a-number` is not an integer");
/// assert_eq!(error.code(), ErrorCode::Forg0001);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XdmError {
    code: ErrorCode,
    message: String,
}

impl XdmError {
    /// Builds an error with the given code and message.
    #[must_use]
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    /// The specification error code.
    #[must_use]
    pub const fn code(&self) -> ErrorCode {
        self.code
    }

    /// The human-readable diagnostic message.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for XdmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.code, self.message)
    }
}

impl Error for XdmError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_includes_the_code_in_brackets() {
        let error = XdmError::new(ErrorCode::Xpty0004, "wrong type");
        assert_eq!(error.to_string(), "[XPTY0004] wrong type");
        assert_eq!(error.code(), ErrorCode::Xpty0004);
        assert_eq!(error.message(), "wrong type");
    }

    #[test]
    fn every_code_renders_as_its_specification_spelling() {
        assert_eq!(ErrorCode::Forg0001.as_str(), "FORG0001");
        assert_eq!(ErrorCode::Forg0006.as_str(), "FORG0006");
        assert_eq!(ErrorCode::Foty0012.as_str(), "FOTY0012");
        assert_eq!(ErrorCode::Foty0013.as_str(), "FOTY0013");
        assert_eq!(ErrorCode::Foty0014.as_str(), "FOTY0014");
        assert_eq!(ErrorCode::Foty0015.as_str(), "FOTY0015");
        assert_eq!(ErrorCode::Foay0001.as_str(), "FOAY0001");
        assert_eq!(ErrorCode::Foch0002.as_str(), "FOCH0002");
        assert_eq!(ErrorCode::Xqdy0137.as_str(), "XQDY0137");
        assert_eq!(ErrorCode::Fodc0006.as_str(), "FODC0006");
        assert_eq!(ErrorCode::Fodt0001.as_str(), "FODT0001");
        assert_eq!(ErrorCode::Fodt0002.as_str(), "FODT0002");
        assert_eq!(ErrorCode::Foca0002.as_str(), "FOCA0002");
        assert_eq!(
            ErrorCode::Xpty0004.namespace(),
            "http://www.w3.org/2005/xqt-errors"
        );
    }
}
