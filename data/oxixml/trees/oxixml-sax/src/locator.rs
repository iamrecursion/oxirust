//! Document position tracking: [`Locator`] and [`SaxParseException`].
//!
//! Mirrors the role of `org.xml.sax.Locator` and `org.xml.sax.SAXParseException`
//! from the Java SAX2 API: a [`Locator`] is a snapshot of "where in the document"
//! a callback fired, and a [`SaxParseException`] pairs a human-readable message
//! with such a snapshot for [`ErrorHandler`](crate::ErrorHandler).

use std::fmt;

/// A snapshot of a position within the document being parsed: a 1-based line and
/// column (in Unicode scalar values), a 0-based byte offset, and the identifiers
/// of the entity being read.
///
/// Unlike the Java SAX2 `Locator`, which is a "live" object that the reader
/// mutates in place as parsing proceeds, this type is an immutable snapshot
/// captured at the point of the call: [`XMLReader::parse`](crate::XMLReader::parse)
/// passes a fresh one to
/// [`ContentHandler::set_document_locator`](crate::ContentHandler::set_document_locator)
/// once at the start of the document, and a fresh one is embedded in every
/// [`SaxParseException`] delivered to an [`ErrorHandler`](crate::ErrorHandler).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Locator {
    line: u64,
    column: u64,
    offset: u64,
    public_id: Option<String>,
    system_id: Option<String>,
}

impl Locator {
    /// Creates a new locator at the given line/column/byte-offset, with the given
    /// system and public identifiers.
    #[must_use]
    pub const fn new(
        line: u64,
        column: u64,
        offset: u64,
        public_id: Option<String>,
        system_id: Option<String>,
    ) -> Self {
        Self {
            line,
            column,
            offset,
            public_id,
            system_id,
        }
    }

    /// A locator at the very start of a document (line 1, column 1, offset 0)
    /// with no public or system identifier.
    #[must_use]
    pub const fn start() -> Self {
        Self::new(1, 1, 0, None, None)
    }

    /// The 1-based line number, or `0` if unknown.
    #[must_use]
    pub const fn line_number(&self) -> u64 {
        self.line
    }

    /// The 1-based column number (in Unicode scalar values), or `0` if unknown.
    #[must_use]
    pub const fn column_number(&self) -> u64 {
        self.column
    }

    /// The 0-based byte offset from the start of the entity.
    #[must_use]
    pub const fn byte_offset(&self) -> u64 {
        self.offset
    }

    /// The public identifier of the entity currently being parsed, if known.
    #[must_use]
    pub fn public_id(&self) -> Option<&str> {
        self.public_id.as_deref()
    }

    /// The system identifier (URI or file path) of the entity currently being
    /// parsed, if known.
    #[must_use]
    pub fn system_id(&self) -> Option<&str> {
        self.system_id.as_deref()
    }
}

impl fmt::Display for Locator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(system_id) = &self.system_id {
            write!(f, "{}:", system_id)?;
        }
        write!(f, "{}:{}", self.line, self.column)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// A parse error or warning, paired with the [`Locator`] describing where in the
/// document it was detected.
///
/// Mirrors `org.xml.sax.SAXParseException`. Delivered to
/// [`ErrorHandler::warning`](crate::ErrorHandler::warning),
/// [`ErrorHandler::error`](crate::ErrorHandler::error) and
/// [`ErrorHandler::fatal_error`](crate::ErrorHandler::fatal_error).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SaxParseException {
    message: String,
    locator: Locator,
}

impl SaxParseException {
    /// Creates a new exception from a message and the locator snapshot describing
    /// where it occurred.
    #[must_use]
    pub const fn new(message: String, locator: Locator) -> Self {
        Self { message, locator }
    }

    /// A human-readable description of the problem.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }

    /// The position at which the problem was detected.
    #[must_use]
    pub const fn locator(&self) -> &Locator {
        &self.locator
    }

    /// Convenience for `self.locator().line_number()`.
    #[must_use]
    pub const fn line_number(&self) -> u64 {
        self.locator.line_number()
    }

    /// Convenience for `self.locator().column_number()`.
    #[must_use]
    pub const fn column_number(&self) -> u64 {
        self.locator.column_number()
    }

    /// Convenience for `self.locator().system_id()`.
    #[must_use]
    pub fn system_id(&self) -> Option<&str> {
        self.locator.system_id()
    }
}

impl fmt::Display for SaxParseException {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.locator, self.message)
    }
}

impl std::error::Error for SaxParseException {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn start_locator_is_line_one_column_one() {
        let locator = Locator::start();
        assert_eq!(locator.line_number(), 1);
        assert_eq!(locator.column_number(), 1);
        assert_eq!(locator.byte_offset(), 0);
        assert_eq!(locator.system_id(), None);
    }

    #[test]
    fn display_includes_system_id_when_present() {
        let locator = Locator::new(3, 5, 20, None, Some("doc.xml".to_string()));
        assert_eq!(locator.to_string(), "doc.xml:3:5");

        let locator = Locator::new(3, 5, 20, None, None);
        assert_eq!(locator.to_string(), "3:5");
    }

    #[test]
    fn exception_display_combines_locator_and_message() {
        let locator = Locator::new(2, 7, 10, None, Some("a.xml".to_string()));
        let exception = SaxParseException::new("unexpected token".to_string(), locator);
        assert_eq!(exception.to_string(), "a.xml:2:7: unexpected token");
        assert_eq!(exception.message(), "unexpected token");
        assert_eq!(exception.line_number(), 2);
        assert_eq!(exception.column_number(), 7);
        assert_eq!(exception.system_id(), Some("a.xml"));
    }
}
