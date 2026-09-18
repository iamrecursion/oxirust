use std::ops::Range;
use std::{fmt, io};

/// A position in a text i.e. a `line` number starting from 0, a `column` number starting from 0 (in number of code points) and a global file `offset` starting from 0 (in number of bytes).
#[derive(Eq, PartialEq, Debug, Copy, Clone)]
pub struct TextPosition {
    /// The line number, starting from 0.
    pub line: u64,
    /// The column number (in Unicode code points), starting from 0.
    pub column: u64,
    /// The global byte offset in the input, starting from 0.
    pub offset: u64,
}

/// An error in the syntax of the parsed file.
///
/// It is composed of a message and a byte range in the input.
#[derive(Debug)]
pub struct TurtleSyntaxError {
    location: Range<TextPosition>,
    message: String,
}

impl TurtleSyntaxError {
    pub(crate) fn new(location: Range<TextPosition>, message: impl Into<String>) -> Self {
        Self {
            location,
            message: message.into(),
        }
    }

    /// The location of the error inside of the file.
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

impl fmt::Display for TurtleSyntaxError {
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

impl std::error::Error for TurtleSyntaxError {}

impl From<TurtleSyntaxError> for io::Error {
    #[inline]
    fn from(error: TurtleSyntaxError) -> Self {
        Self::new(io::ErrorKind::InvalidData, error)
    }
}

/// A parsing error.
///
/// It is the union of [`TurtleSyntaxError`] and [`io::Error`].
#[derive(Debug)]
pub enum TurtleParseError {
    /// I/O error during parsing (file not found...).
    Io(io::Error),
    /// An error in the file syntax.
    Syntax(TurtleSyntaxError),
}

impl fmt::Display for TurtleParseError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => e.fmt(f),
            Self::Syntax(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for TurtleParseError {
    #[inline]
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Syntax(e) => Some(e),
        }
    }
}

impl From<io::Error> for TurtleParseError {
    #[inline]
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<TurtleSyntaxError> for TurtleParseError {
    #[inline]
    fn from(error: TurtleSyntaxError) -> Self {
        Self::Syntax(error)
    }
}

impl From<TurtleParseError> for io::Error {
    #[inline]
    fn from(error: TurtleParseError) -> Self {
        match error {
            TurtleParseError::Syntax(e) => e.into(),
            TurtleParseError::Io(e) => e,
        }
    }
}
