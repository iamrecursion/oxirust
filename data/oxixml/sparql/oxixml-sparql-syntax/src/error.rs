//! Error type returned during SPARQL parsing.

use std::error::Error;
use std::fmt;
use std::ops::Range;

/// Error returned during SPARQL parsing.
///
/// It carries a human readable [`message`](Self::message) and the [`location`](Self::location)
/// of the error inside the source text.
#[derive(Debug, Clone)]
pub struct SparqlSyntaxError {
    pub(crate) message: String,
    pub(crate) location: Range<TextPosition>,
}

impl SparqlSyntaxError {
    pub(crate) fn new(message: impl Into<String>, location: Range<TextPosition>) -> Self {
        Self {
            message: message.into(),
            location,
        }
    }

    /// Builds an error from a byte-offset range inside `text`.
    pub(crate) fn from_offsets(
        message: impl Into<String>,
        text: &str,
        offsets: Range<usize>,
    ) -> Self {
        Self::new(
            message,
            TextPosition::from_offset(text, offsets.start)
                ..TextPosition::from_offset(text, offsets.end),
        )
    }

    /// The location of the error inside the file.
    #[inline]
    pub fn location(&self) -> Range<TextPosition> {
        self.location.clone()
    }

    /// The error message.
    #[inline]
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for SparqlSyntaxError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.location.start.offset + 1 >= self.location.end.offset {
            write!(
                f,
                "Syntax error at line {} column {}: {}",
                self.location.start.line + 1,
                self.location.start.column + 1,
                self.message
            )
        } else if self.location.start.line == self.location.end.line {
            write!(
                f,
                "Syntax error at line {} between columns {} and {}: {}",
                self.location.start.line + 1,
                self.location.start.column + 1,
                self.location.end.column + 1,
                self.message
            )
        } else {
            write!(
                f,
                "Syntax error between line {} column {} and line {} column {}: {}",
                self.location.start.line + 1,
                self.location.start.column + 1,
                self.location.end.line + 1,
                self.location.end.column + 1,
                self.message
            )
        }
    }
}

impl Error for SparqlSyntaxError {}

/// A position in a text.
///
/// It is a `line` number starting from 0, a `column` number starting from 0 (in number of code
/// points) and a global file `offset` starting from 0 (in number of bytes).
#[derive(Eq, PartialEq, Debug, Copy, Clone)]
pub struct TextPosition {
    /// The line number, starting from 0.
    pub line: u64,
    /// The column number (in code points), starting from 0.
    pub column: u64,
    /// The byte offset, starting from 0.
    pub offset: u64,
}

impl TextPosition {
    fn from_offset(text: &str, offset: usize) -> Self {
        let offset = offset.min(text.len());
        // Snap to a char boundary so slicing never panics.
        let mut safe = offset;
        while safe > 0 && !text.is_char_boundary(safe) {
            safe -= 1;
        }
        let mut line = 0_u64;
        let mut column = 0_u64;
        for c in text[..safe].chars() {
            if c == '\n' {
                line += 1;
                column = 0;
            } else {
                column += 1;
            }
        }
        Self {
            line,
            column,
            offset: safe as u64,
        }
    }
}
