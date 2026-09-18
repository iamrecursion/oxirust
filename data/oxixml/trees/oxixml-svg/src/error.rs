//! Error types shared by every module of the crate.
//!
//! Two families exist: [`ValueError`] for failures inside a single attribute
//! microsyntax (path data, transform lists, colors, lengths, …), carrying a
//! byte offset into the attribute value, and [`SvgError`] for failures while
//! turning a whole input into an SVG document.

use std::fmt;

/// An error produced while parsing an SVG attribute microsyntax value.
///
/// Offsets are 0-based byte offsets into the attribute value string that was
/// handed to the parser, not into the surrounding document.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ValueError {
    /// The input ended before the value was complete.
    UnexpectedEnd {
        /// Byte offset of the end of the input.
        offset: usize,
    },
    /// A character that cannot start or continue the expected token was found.
    UnexpectedChar {
        /// Byte offset of the offending character.
        offset: usize,
        /// The character that was found.
        found: char,
    },
    /// A numeric component was malformed.
    InvalidNumber {
        /// Byte offset where the malformed number starts.
        offset: usize,
    },
    /// A component parsed but is outside the range the grammar permits.
    OutOfRange {
        /// Byte offset where the offending component starts.
        offset: usize,
        /// A short description of the constraint that was violated.
        what: &'static str,
    },
    /// A keyword was read but is not one the grammar accepts.
    UnknownKeyword {
        /// Byte offset where the keyword starts.
        offset: usize,
        /// The keyword that was found.
        keyword: Box<str>,
    },
    /// The value was empty (or whitespace-only) where content is required.
    Empty,
    /// A well-formed value was parsed but unexpected content followed it.
    TrailingData {
        /// Byte offset where the trailing content starts.
        offset: usize,
    },
}

impl ValueError {
    /// The 0-based byte offset the error points at, if it has one.
    #[must_use]
    pub const fn offset(&self) -> Option<usize> {
        match self {
            Self::UnexpectedEnd { offset }
            | Self::UnexpectedChar { offset, .. }
            | Self::InvalidNumber { offset }
            | Self::OutOfRange { offset, .. }
            | Self::UnknownKeyword { offset, .. }
            | Self::TrailingData { offset } => Some(*offset),
            Self::Empty => None,
        }
    }
}

impl fmt::Display for ValueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEnd { offset } => {
                write!(f, "unexpected end of value at byte offset {offset}")
            }
            Self::UnexpectedChar { offset, found } => {
                write!(f, "unexpected character {found:?} at byte offset {offset}")
            }
            Self::InvalidNumber { offset } => {
                write!(f, "malformed number at byte offset {offset}")
            }
            Self::OutOfRange { offset, what } => {
                write!(f, "value out of range at byte offset {offset}: {what}")
            }
            Self::UnknownKeyword { offset, keyword } => {
                write!(f, "unknown keyword {keyword:?} at byte offset {offset}")
            }
            Self::Empty => f.write_str("empty value where content is required"),
            Self::TrailingData { offset } => {
                write!(
                    f,
                    "trailing content after a complete value at byte offset {offset}"
                )
            }
        }
    }
}

impl std::error::Error for ValueError {}

/// An error produced while parsing an input into an SVG document.
#[derive(Debug)]
#[non_exhaustive]
pub enum SvgError {
    /// The input is not well-formed XML and could not be built into a tree.
    Xml(oxixml_dom::BuildError),
    /// The input parsed as XML but its document element is not
    /// `{http://www.w3.org/2000/svg}svg`.
    NotSvg {
        /// The qualified name of the document element that was found, if any.
        found: Option<Box<str>>,
    },
}

impl fmt::Display for SvgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Xml(error) => write!(f, "the input is not a well-formed XML document: {error}"),
            Self::NotSvg { found: Some(name) } => write!(
                f,
                "the document element {name:?} is not an `svg` element in the SVG namespace"
            ),
            Self::NotSvg { found: None } => {
                f.write_str("the document has no element in the SVG namespace")
            }
        }
    }
}

impl std::error::Error for SvgError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Xml(error) => Some(error),
            Self::NotSvg { .. } => None,
        }
    }
}

impl From<oxixml_dom::BuildError> for SvgError {
    fn from(error: oxixml_dom::BuildError) -> Self {
        Self::Xml(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_error_offset_is_reported() {
        let error = ValueError::UnexpectedChar {
            offset: 4,
            found: 'q',
        };
        assert_eq!(error.offset(), Some(4));
        assert_eq!(
            error.to_string(),
            "unexpected character 'q' at byte offset 4"
        );
        assert_eq!(ValueError::Empty.offset(), None);
    }

    #[test]
    fn svg_error_wraps_build_error() {
        let build = oxixml_dom::build::parse_str("<a>").expect_err("unclosed element is an error");
        let error = SvgError::from(build);
        assert!(matches!(error, SvgError::Xml(_)));
        assert!(
            error
                .to_string()
                .starts_with("the input is not a well-formed XML document")
        );
    }
}
