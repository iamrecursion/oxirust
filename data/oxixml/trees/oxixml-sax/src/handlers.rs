//! [`Handlers`], the bundle of optional callback objects passed to
//! [`XMLReader::parse`](crate::XMLReader::parse), and the [`Feature`] flags that
//! control how it drives them.

use crate::content_handler::ContentHandler;
use crate::decl_handler::DeclHandler;
use crate::dtd_handler::DtdHandler;
use crate::entity_resolver::EntityResolver;
use crate::error_handler::ErrorHandler;
use crate::lexical_handler::LexicalHandler;

/// A bundle of optional handler objects that [`XMLReader::parse`](crate::XMLReader::parse)
/// dispatches to.
///
/// Every field defaults to `None` (via [`Handlers::new`] / [`Default`]); a
/// missing handler is simply never called. Java SAX2 instead has an `XMLReader`
/// hold `setContentHandler`/`setErrorHandler`/… setters as mutable state; this
/// crate takes the handlers as one borrow per [`parse`](crate::XMLReader::parse)
/// call instead, which plays better with Rust's borrow checker (no need to give
/// up ownership of your handler, or reach for interior mutability, just to read
/// it back out after parsing finishes).
///
/// # Example
///
/// ```
/// use oxixml_sax::{ContentHandler, Handlers, XMLReader};
///
/// #[derive(Default)]
/// struct CountElements(u32);
/// impl ContentHandler for CountElements {
///     fn start_element(
///         &mut self,
///         _uri: Option<&str>,
///         _local_name: &str,
///         _qname: &str,
///         _attributes: &oxixml_sax::Attributes,
///     ) -> Result<(), oxixml_sax::SaxError> {
///         self.0 += 1;
///         Ok(())
///     }
/// }
///
/// let mut content = CountElements::default();
/// let mut handlers = Handlers::new().with_content(&mut content);
/// XMLReader::new().parse_str("<a><b/><b/></a>", &mut handlers)?;
/// assert_eq!(content.0, 3);
/// # Ok::<(), oxixml_sax::SaxError>(())
/// ```
#[derive(Default)]
pub struct Handlers<'h> {
    /// Receives element, character-data and prefix-mapping events.
    pub content: Option<&'h mut dyn ContentHandler>,
    /// Receives comment, `CDATA` and DTD-boundary events.
    pub lexical: Option<&'h mut dyn LexicalHandler>,
    /// Receives notation and unparsed-entity declarations.
    pub dtd: Option<&'h mut dyn DtdHandler>,
    /// Receives element, attribute-list and entity declarations.
    pub decl: Option<&'h mut dyn DeclHandler>,
    /// Receives warnings and (fatal) errors.
    pub error: Option<&'h mut dyn ErrorHandler>,
    /// Resolves external identifiers.
    pub entity_resolver: Option<&'h mut dyn EntityResolver>,
}

impl<'h> Handlers<'h> {
    /// An empty bundle: every handler is `None`.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            content: None,
            lexical: None,
            dtd: None,
            decl: None,
            error: None,
            entity_resolver: None,
        }
    }

    /// Installs a [`ContentHandler`].
    #[must_use]
    pub fn with_content(mut self, handler: &'h mut dyn ContentHandler) -> Self {
        self.content = Some(handler);
        self
    }

    /// Installs a [`LexicalHandler`].
    #[must_use]
    pub fn with_lexical(mut self, handler: &'h mut dyn LexicalHandler) -> Self {
        self.lexical = Some(handler);
        self
    }

    /// Installs a [`DtdHandler`].
    #[must_use]
    pub fn with_dtd(mut self, handler: &'h mut dyn DtdHandler) -> Self {
        self.dtd = Some(handler);
        self
    }

    /// Installs a [`DeclHandler`].
    #[must_use]
    pub fn with_decl(mut self, handler: &'h mut dyn DeclHandler) -> Self {
        self.decl = Some(handler);
        self
    }

    /// Installs an [`ErrorHandler`].
    #[must_use]
    pub fn with_error(mut self, handler: &'h mut dyn ErrorHandler) -> Self {
        self.error = Some(handler);
        self
    }

    /// Installs an [`EntityResolver`].
    #[must_use]
    pub fn with_entity_resolver(mut self, resolver: &'h mut dyn EntityResolver) -> Self {
        self.entity_resolver = Some(resolver);
        self
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// The SAX2 core feature URI for namespace processing.
pub const FEATURE_NAMESPACES: &str = "http://xml.org/sax/features/namespaces";
/// The SAX2 core feature URI controlling whether namespace declaration
/// attributes and prefixed qualified names are retained.
pub const FEATURE_NAMESPACE_PREFIXES: &str = "http://xml.org/sax/features/namespace-prefixes";

/// A toggle recognized by [`XMLReader::set_feature`](crate::XMLReader::set_feature),
/// mirroring the two SAX2 core features every conforming `XMLReader` must
/// support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Feature {
    /// Whether element and attribute names are resolved against namespace
    /// declarations, and [`ContentHandler::start_prefix_mapping`]/
    /// [`end_prefix_mapping`](crate::ContentHandler::end_prefix_mapping) are
    /// called. Default: `true`.
    ///
    /// [`ContentHandler::start_prefix_mapping`]: crate::ContentHandler::start_prefix_mapping
    Namespaces,
    /// Whether `xmlns`/`xmlns:*` attributes are retained in the
    /// [`Attributes`](crate::Attributes) list passed to
    /// [`ContentHandler::start_element`](crate::ContentHandler::start_element),
    /// alongside namespace-resolved names. Default: `false`.
    NamespacePrefixes,
}

impl Feature {
    /// The SAX2 feature URI naming this flag.
    #[must_use]
    pub const fn uri(self) -> &'static str {
        match self {
            Self::Namespaces => FEATURE_NAMESPACES,
            Self::NamespacePrefixes => FEATURE_NAMESPACE_PREFIXES,
        }
    }

    /// Looks up a [`Feature`] by its SAX2 URI, returning `None` if `uri` does not
    /// name a feature this crate supports.
    #[must_use]
    pub fn from_uri(uri: &str) -> Option<Self> {
        match uri {
            FEATURE_NAMESPACES => Some(Self::Namespaces),
            FEATURE_NAMESPACE_PREFIXES => Some(Self::NamespacePrefixes),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_uri_round_trip() {
        assert_eq!(
            Feature::from_uri(FEATURE_NAMESPACES),
            Some(Feature::Namespaces)
        );
        assert_eq!(
            Feature::from_uri(FEATURE_NAMESPACE_PREFIXES),
            Some(Feature::NamespacePrefixes)
        );
        assert_eq!(Feature::from_uri("http://example.com/nope"), None);
        assert_eq!(Feature::Namespaces.uri(), FEATURE_NAMESPACES);
    }

    #[test]
    fn empty_bundle_has_no_handlers() {
        let handlers = Handlers::new();
        assert!(handlers.content.is_none());
        assert!(handlers.lexical.is_none());
        assert!(handlers.dtd.is_none());
        assert!(handlers.decl.is_none());
        assert!(handlers.error.is_none());
        assert!(handlers.entity_resolver.is_none());
    }
}
