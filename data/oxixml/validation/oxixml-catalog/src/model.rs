//! The in-memory representation of a parsed catalog document.
//!
//! A catalog file's `group` elements exist only to *scope* an `xml:base` and a
//! `prefer` setting over the entries they contain (OASIS XML Catalogs 1.1 §4,
//! §5): they never affect matching order. The parser therefore flattens the
//! whole document into a single [`Vec<Entry>`] in document order, with each
//! entry pre-normalized and pre-absolutized against the `xml:base` in effect
//! at its position — [`crate::resolve`] never has to think about nesting.

/// The `prefer` search strategy in effect for a `public` or `delegatePublic`
/// entry (OASIS XML Catalogs 1.1 §5, `prefer` attribute of `catalog`/`group`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Prefer {
    /// A `public`/`delegatePublic` entry in this scope is ignored by
    /// [`crate::Catalog::resolve_public`] lookups whose external identifier
    /// carries an explicit system identifier.
    System,
    /// A `public`/`delegatePublic` entry in this scope is always considered,
    /// whether or not the external identifier carries a system identifier.
    Public,
}

/// One flattened, already-normalized catalog entry.
#[derive(Debug, Clone)]
pub(crate) enum Entry {
    /// `<public publicId="..." uri="..."/>`
    Public {
        /// Normalized public identifier to match.
        public_id: String,
        /// Absolutized replacement URI.
        uri: String,
        /// The `prefer` scope this entry was declared under.
        prefer: Prefer,
    },
    /// `<system systemId="..." uri="..."/>`
    System {
        /// Normalized system identifier to match.
        system_id: String,
        /// Absolutized replacement URI.
        uri: String,
    },
    /// `<uri name="..." uri="..."/>`
    Uri {
        /// Normalized URI reference to match.
        name: String,
        /// Absolutized replacement URI.
        uri: String,
    },
    /// `<rewriteSystem systemIdStartString="..." rewritePrefix="..."/>`
    RewriteSystem {
        /// Normalized prefix to match against the start of a system identifier.
        start: String,
        /// Absolutized replacement for the matched prefix.
        prefix: String,
    },
    /// `<rewriteURI uriStartString="..." rewritePrefix="..."/>`
    RewriteUri {
        /// Normalized prefix to match against the start of a URI reference.
        start: String,
        /// Absolutized replacement for the matched prefix.
        prefix: String,
    },
    /// `<systemSuffix systemIdSuffix="..." uri="..."/>` (XML Catalogs 1.1).
    SystemSuffix {
        /// Normalized suffix to match against the end of a system identifier.
        suffix: String,
        /// Absolutized replacement URI.
        uri: String,
    },
    /// `<uriSuffix uriSuffix="..." uri="..."/>` (XML Catalogs 1.1).
    UriSuffix {
        /// Normalized suffix to match against the end of a URI reference.
        suffix: String,
        /// Absolutized replacement URI.
        uri: String,
    },
    /// `<delegatePublic publicIdStartString="..." catalog="..."/>`
    DelegatePublic {
        /// Normalized prefix to match against the start of a public identifier.
        start: String,
        /// Absolutized URI of the delegate catalog.
        catalog: String,
        /// The `prefer` scope this entry was declared under.
        prefer: Prefer,
    },
    /// `<delegateSystem systemIdStartString="..." catalog="..."/>`
    DelegateSystem {
        /// Normalized prefix to match against the start of a system identifier.
        start: String,
        /// Absolutized URI of the delegate catalog.
        catalog: String,
    },
    /// `<delegateURI uriStartString="..." catalog="..."/>`
    DelegateUri {
        /// Normalized prefix to match against the start of a URI reference.
        start: String,
        /// Absolutized URI of the delegate catalog.
        catalog: String,
    },
    /// `<nextCatalog catalog="..."/>`
    NextCatalog {
        /// Absolutized URI of the next catalog file.
        catalog: String,
    },
}

/// A single parsed catalog document: an ordered, flattened list of [`Entry`]
/// values plus the base URI of the document itself (used to report a stable
/// identity for loop detection).
#[derive(Debug, Clone)]
pub(crate) struct CatalogFile {
    /// The absolute URI this catalog document was loaded from (or the caller-supplied
    /// base URI, for the top-level document parsed directly from a string).
    pub(crate) uri: String,
    /// Flattened entries, in document order.
    pub(crate) entries: Vec<Entry>,
}
