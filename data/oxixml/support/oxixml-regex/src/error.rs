//! Errors raised while compiling or evaluating a regular expression.
//!
//! Every error carries a human-readable message and, where the situation maps
//! onto one of the four W3C *XPath and XQuery Functions and Operators* error
//! codes, the corresponding [`ForxCode`]. The mapping is:
//!
//! | Variant | Code | Meaning |
//! |---|---|---|
//! | [`RegexError::InvalidFlags`] | `FORX0001` | the `flags` string is invalid |
//! | [`RegexError::InvalidPattern`] | `FORX0002` | the pattern is syntactically invalid |
//! | [`RegexError::MatchesEmptyString`] | `FORX0003` | the pattern matches a zero-length string |
//! | [`RegexError::InvalidReplacement`] | `FORX0004` | the replacement string is invalid |
//! | [`RegexError::BudgetExceeded`] | — | the back-tracking budget was exhausted |
//!
//! [`RegexError::BudgetExceeded`] has no `FORX` counterpart: it is an
//! implementation safety valve that stops pathological back-tracking (for
//! example `(a+)+\1$` against a long non-matching string) instead of hanging.
//! The budget is cumulative across every starting position tried while
//! looking for *one* match, so a back-reference pattern that is merely
//! expensive (never catastrophic) at each of many positions is bounded too,
//! not only the single-position worst case; [`Regex::replace`]
//! (crate::Regex::replace) and [`Regex::tokenize`](crate::Regex::tokenize)
//! extend that same cumulative budget across every match found during their
//! one call, since each is a single operation with one result. See
//! `crate::backtrack::DEFAULT_BUDGET` for the exact scope in each case,
//! including why [`Regex::find_iter`](crate::Regex::find_iter) does not
//! extend it across the matches it yields.

use core::fmt;

/// One of the four regular-expression error codes defined by
/// [*XPath and XQuery Functions and Operators 3.1*][fo], section 5.6.
///
/// [fo]: https://www.w3.org/TR/xpath-functions-31/#regex-syntax
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ForxCode {
    /// `FORX0001` — invalid regular-expression flags.
    Forx0001,
    /// `FORX0002` — invalid regular expression.
    Forx0002,
    /// `FORX0003` — the regular expression matches a zero-length string.
    Forx0003,
    /// `FORX0004` — invalid replacement string (raised only by `replace`).
    Forx0004,
}

impl ForxCode {
    /// Returns the canonical spelling of the code, such as `"FORX0001"`.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_regex::ForxCode;
    ///
    /// assert_eq!(ForxCode::Forx0002.as_str(), "FORX0002");
    /// ```
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            ForxCode::Forx0001 => "FORX0001",
            ForxCode::Forx0002 => "FORX0002",
            ForxCode::Forx0003 => "FORX0003",
            ForxCode::Forx0004 => "FORX0004",
        }
    }
}

impl fmt::Display for ForxCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// An error produced while building or running a [`Regex`](crate::Regex).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegexError {
    /// The `flags` argument to [`Regex::xpath`](crate::Regex::xpath) contained a
    /// character other than `s`, `m`, `i`, `x`, or `q` (`FORX0001`).
    InvalidFlags(String),
    /// The pattern is not a valid regular expression for its dialect
    /// (`FORX0002`). The message describes the specific problem.
    InvalidPattern(String),
    /// A `replace` or `tokenize` operation was attempted with a pattern that is
    /// capable of matching a zero-length string (`FORX0003`).
    MatchesEmptyString,
    /// The replacement string passed to [`Regex::replace`](crate::Regex::replace)
    /// is invalid (`FORX0004`): a `$` not followed by a digit and not escaped,
    /// or a `\` that is not part of a `\\` or `\$` pair.
    InvalidReplacement(String),
    /// Back-tracking evaluation of a pattern that contains back-references
    /// consumed the entire step budget without terminating. This protects
    /// against catastrophic back-tracking; it has no `FORX` code.
    BudgetExceeded,
}

impl RegexError {
    /// Returns the W3C `FORX` error code this error maps onto, or `None` for
    /// [`RegexError::BudgetExceeded`], which is implementation-specific.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_regex::{ForxCode, Regex, RegexError};
    ///
    /// let err = Regex::xpath("(", "").unwrap_err();
    /// assert_eq!(err.forx_code(), Some(ForxCode::Forx0002));
    /// ```
    #[must_use]
    pub const fn forx_code(&self) -> Option<ForxCode> {
        match self {
            RegexError::InvalidFlags(_) => Some(ForxCode::Forx0001),
            RegexError::InvalidPattern(_) => Some(ForxCode::Forx0002),
            RegexError::MatchesEmptyString => Some(ForxCode::Forx0003),
            RegexError::InvalidReplacement(_) => Some(ForxCode::Forx0004),
            RegexError::BudgetExceeded => None,
        }
    }
}

impl fmt::Display for RegexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegexError::InvalidFlags(m) => {
                write!(f, "FORX0001: invalid regular-expression flags: {m}")
            }
            RegexError::InvalidPattern(m) => write!(f, "FORX0002: invalid regular expression: {m}"),
            RegexError::MatchesEmptyString => {
                f.write_str("FORX0003: regular expression matches a zero-length string")
            }
            RegexError::InvalidReplacement(m) => {
                write!(f, "FORX0004: invalid replacement string: {m}")
            }
            RegexError::BudgetExceeded => {
                f.write_str("regular-expression back-tracking budget exceeded")
            }
        }
    }
}

impl std::error::Error for RegexError {}
