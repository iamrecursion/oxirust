//! The crate-wide error type, [`SaxError`].

use std::fmt;

/// The error type returned by [`XMLReader::parse`](crate::XMLReader::parse) and by
/// every fallible handler callback.
///
/// A handler callback returns `Err` to abort parsing early (for example, a
/// [`ContentHandler`](crate::ContentHandler) that only cares about the first
/// matching element can stop as soon as it has found it); the driving loop
/// propagates that error straight back out of `parse`, without further calls to
/// [`ErrorHandler`](crate::ErrorHandler).
#[derive(Debug, Clone)]
pub enum SaxError {
    /// The underlying [`oxixml_xml`] reader reported a fatal error (the document
    /// does not conform to the XML grammar, or violates a well-formedness
    /// constraint that could not be recovered from).
    Xml(oxixml_xml::errors::Error),
    /// A handler callback aborted parsing with a custom message.
    Handler(String),
    /// [`XMLReader::set_feature_by_name`](crate::XMLReader::set_feature_by_name)
    /// was called with a URI that does not name a feature this reader supports.
    UnknownFeature(String),
    /// The document violates an XML 1.0 well-formedness constraint that the
    /// underlying [`oxixml_xml`] reader does not itself diagnose — an element
    /// left open at the end of the entity, or character data outside the
    /// document element (XML 1.0 §2.1 `document ::= prolog element Misc*`).
    NotWellFormed(String),
}

impl SaxError {
    /// Builds a [`SaxError::Handler`] from any [`Display`](fmt::Display)-able
    /// message. Handy for handler implementations that want to abort parsing.
    #[must_use]
    pub fn handler<M: fmt::Display>(message: M) -> Self {
        Self::Handler(message.to_string())
    }
}

impl fmt::Display for SaxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Xml(e) => write!(f, "XML parse error: {}", e),
            Self::Handler(message) => write!(f, "handler error: {}", message),
            Self::UnknownFeature(uri) => write!(f, "unknown SAX feature: {}", uri),
            Self::NotWellFormed(message) => write!(f, "not well-formed: {}", message),
        }
    }
}

impl std::error::Error for SaxError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Xml(e) => Some(e),
            Self::Handler(_) | Self::UnknownFeature(_) | Self::NotWellFormed(_) => None,
        }
    }
}

impl From<oxixml_xml::errors::Error> for SaxError {
    #[inline]
    fn from(error: oxixml_xml::errors::Error) -> Self {
        Self::Xml(error)
    }
}

/// A specialized [`Result`](std::result::Result) with the error hard-wired to
/// [`SaxError`].
pub type Result<T> = std::result::Result<T, SaxError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_variants() {
        let handler = SaxError::handler("stop");
        assert_eq!(handler.to_string(), "handler error: stop");

        let unknown = SaxError::UnknownFeature("http://example.com/feature".to_string());
        assert_eq!(
            unknown.to_string(),
            "unknown SAX feature: http://example.com/feature"
        );
    }

    #[test]
    fn xml_error_has_source() {
        let xml_error: oxixml_xml::errors::Error =
            oxixml_xml::errors::SyntaxError::UnclosedTag.into();
        let error: SaxError = xml_error.into();
        assert!(std::error::Error::source(&error).is_some());
    }
}
