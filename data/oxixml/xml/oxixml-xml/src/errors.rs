//! Error management for the crate.

use crate::encoding::EncodingError;
use crate::escape::EscapeError;
use crate::events::attributes::AttrError;
use crate::name::NamespaceError;
use std::fmt;
use std::io::{Error as IoError, ErrorKind as IoErrorKind};
use std::sync::Arc;

/// An error returned when a parsed document does not conform to the XML grammar,
/// for example a tag opened by `<` that is not closed with `>`.
///
/// This error does not represent invalid XML *constructs* — for example the tags
/// `<>` and `</>` are well-formed from a syntactic point of view.
///
/// Marked `#[non_exhaustive]` so that adding a new syntax diagnostic in a future
/// release is not a breaking change for callers who match on this type.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SyntaxError {
    /// The parser started to parse `<!`, but the input ended before it could
    /// recognize anything.
    InvalidBangMarkup,
    /// The parser started to parse a processing instruction (`<?`), but the input
    /// ended before the `?>` sequence was found.
    UnclosedPI,
    /// The parser started to parse an XML declaration (`<?xml` followed by white
    /// space or `?`), but the input ended before the `?>` sequence was found.
    UnclosedXmlDecl,
    /// The parser started to parse a comment (`<!--`), but the input ended before
    /// the `-->` sequence was found.
    UnclosedComment,
    /// The parser started to parse a DTD (`<!DOCTYPE`), but the input ended before
    /// the closing `>` character was found.
    UnclosedDoctype,
    /// The parser started to parse `<![CDATA[`, but the input ended before the
    /// `]]>` sequence was found.
    UnclosedCData,
    /// The parser started to parse tag content, but the input ended before the
    /// closing `>` character was found.
    UnclosedTag,
    /// The parser was inside a single-quoted attribute value, but the input ended
    /// before the closing quote was found.
    UnclosedSingleQuotedAttributeValue,
    /// The parser was inside a double-quoted attribute value, but the input ended
    /// before the closing quote was found.
    UnclosedDoubleQuotedAttributeValue,
}

impl fmt::Display for SyntaxError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::InvalidBangMarkup => f.write_str("unknown or missed symbol in markup"),
            Self::UnclosedPI => {
                f.write_str("processing instruction not closed: `?>` not found before end of input")
            }
            Self::UnclosedXmlDecl => {
                f.write_str("XML declaration not closed: `?>` not found before end of input")
            }
            Self::UnclosedComment => {
                f.write_str("comment not closed: `-->` not found before end of input")
            }
            Self::UnclosedDoctype => {
                f.write_str("DOCTYPE not closed: `>` not found before end of input")
            }
            Self::UnclosedCData => {
                f.write_str("CDATA not closed: `]]>` not found before end of input")
            }
            Self::UnclosedTag => f.write_str("tag not closed: `>` not found before end of input"),
            Self::UnclosedSingleQuotedAttributeValue => {
                f.write_str("attribute value not closed: `'` not found before end of input")
            }
            Self::UnclosedDoubleQuotedAttributeValue => {
                f.write_str("attribute value not closed: `\"` not found before end of input")
            }
        }
    }
}

impl std::error::Error for SyntaxError {}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// An error returned when a parsed document is not [well-formed], for example an
/// opened tag that is not closed before the end of input.
///
/// These errors are not fatal: parsing can continue after encountering one.
///
/// Marked `#[non_exhaustive]` so that adding a new ill-formedness diagnostic in a
/// future release is not a breaking change for callers who match on this type.
///
/// [well-formed]: https://www.w3.org/TR/xml11/#dt-wellformed
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum IllFormedError {
    /// A `version` attribute was not found in an XML declaration or is not the
    /// first attribute. Contains the name of the found attribute, if any.
    MissingDeclVersion(Option<String>),
    /// The XML version specified in the declaration is neither 1.0 nor 1.1.
    UnknownVersion,
    /// A document type definition (`<!DOCTYPE foo>`) does not contain a name.
    MissingDoctypeName,
    /// An end tag was not found while reading a sub-tree of elements because an
    /// EOF was encountered.
    MissingEndTag(String),
    /// An end tag was encountered without a corresponding open tag at the same
    /// level of the hierarchy.
    UnmatchedEndTag(String),
    /// An end tag does not match the start tag at that nesting level.
    MismatchedEndTag {
        /// Name of the open tag that is expected to be closed.
        expected: String,
        /// Name of the tag that was actually closed.
        found: String,
    },
    /// A comment contains the forbidden double-hyphen (`--`) sequence.
    DoubleHyphenInComment,
    /// The parser started to parse an entity or character reference (`&...;`), but
    /// the input ended before the closing `;` character was found.
    UnclosedReference,
}

impl fmt::Display for IllFormedError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::MissingDeclVersion(None) => {
                f.write_str("an XML declaration does not contain `version` attribute")
            }
            Self::MissingDeclVersion(Some(attr)) => {
                write!(
                    f,
                    "an XML declaration must start with `version` attribute, but in starts with `{}`",
                    attr
                )
            }
            Self::UnknownVersion => {
                f.write_str("unknown XML version: either 1.0 or 1.1 is expected")
            }
            Self::MissingDoctypeName => {
                f.write_str("`<!DOCTYPE>` declaration does not contain a name of a document type")
            }
            Self::MissingEndTag(tag) => write!(
                f,
                "start tag not closed: `</{}>` not found before end of input",
                tag,
            ),
            Self::UnmatchedEndTag(tag) => {
                write!(f, "close tag `</{}>` does not match any open tag", tag)
            }
            Self::MismatchedEndTag { expected, found } => {
                write!(
                    f,
                    "expected `</{}>`, but `</{}>` was found",
                    expected, found
                )
            }
            Self::DoubleHyphenInComment => {
                f.write_str("forbidden string `--` was found in a comment")
            }
            Self::UnclosedReference => f.write_str(
                "entity or character reference not closed: `;` not found before end of input",
            ),
        }
    }
}

impl std::error::Error for IllFormedError {}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// A violation of an XML [well-formedness constraint] that is reported only when
/// [`Config::check_well_formedness`] is enabled.
///
/// The default parsing path of this crate is deliberately permissive so that it
/// stays a drop-in replacement for `quick-xml`; enabling the flag turns the
/// [`Reader`] into a conforming non-validating XML processor and these errors
/// become observable.
///
/// [well-formedness constraint]: https://www.w3.org/TR/xml/#dt-wellformed
/// [`Config::check_well_formedness`]: crate::reader::Config::check_well_formedness
/// [`Reader`]: crate::reader::Reader
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum WellFormedError {
    /// A character that the [`Char`] production of the document's XML version
    /// does not allow was found.
    ///
    /// [`Char`]: https://www.w3.org/TR/xml/#NT-Char
    InvalidCharacter(char),
    /// A token that must match the [`Name`] production does not.
    ///
    /// [`Name`]: https://www.w3.org/TR/xml/#NT-Name
    InvalidName(String),
    /// A public identifier contains a character outside the [`PubidChar`]
    /// production.
    ///
    /// [`PubidChar`]: https://www.w3.org/TR/xml/#NT-PubidChar
    InvalidPublicId(String),
    /// A `&...;` entity or character reference does not match the [`Reference`]
    /// production, or a character reference denotes a code point that is not a
    /// legal XML character.
    ///
    /// [`Reference`]: https://www.w3.org/TR/xml/#NT-Reference
    InvalidReference(String),
    /// A general entity was referenced but never declared, violating the
    /// [Entity Declared] constraint.
    ///
    /// [Entity Declared]: https://www.w3.org/TR/xml/#wf-entdeclared
    UndeclaredEntity(String),
    /// A general entity refers to itself, directly or indirectly, violating the
    /// [No Recursion] constraint.
    ///
    /// [No Recursion]: https://www.w3.org/TR/xml/#norecursion
    RecursiveEntity(String),
    /// Expanding general entities exceeded the limits this processor imposes on
    /// the total number of expansions or on the total size of the expanded text.
    ///
    /// Entity expansion is not recursive in the sense of the [No Recursion]
    /// constraint when every entity has a different name, yet a document such as
    /// the "billion laughs" bomb can still expand exponentially. Since a
    /// non-validating processor is not required to expand entities without
    /// bound, such a document is rejected instead of being expanded.
    ///
    /// [No Recursion]: https://www.w3.org/TR/xml/#norecursion
    EntityExpansionLimit {
        /// Name of the entity whose expansion exhausted the budget.
        entity: String,
        /// Description of the limit that was hit.
        limit: &'static str,
    },
    /// An unparsed (`NDATA`) entity was referenced from content or from an
    /// attribute value, violating the [Parsed Entity] constraint.
    ///
    /// [Parsed Entity]: https://www.w3.org/TR/xml/#wf-parsedent
    UnparsedEntityReference(String),
    /// The replacement text of a general entity is not well-formed on its own,
    /// violating the [Legal Character] / balanced-markup constraints. Contains
    /// the entity name and the nested failure.
    IllFormedEntity {
        /// Name of the entity whose replacement text is at fault.
        entity: String,
        /// Description of the problem found inside the replacement text.
        reason: String,
    },
    /// An attribute value contains a literal `<`, violating the [No `<` in
    /// Attribute Values] constraint. Contains the attribute name.
    ///
    /// [No `<` in Attribute Values]: https://www.w3.org/TR/xml/#CleanAttrVals
    LessThanInAttributeValue(String),
    /// An attribute value refers, directly or indirectly, to an external entity,
    /// violating the [No External Entity References] constraint.
    ///
    /// [No External Entity References]: https://www.w3.org/TR/xml/#NoExternalRefs
    ExternalEntityInAttributeValue(String),
    /// The same attribute name appears twice on one tag, violating the
    /// [Unique Att Spec] constraint.
    ///
    /// [Unique Att Spec]: https://www.w3.org/TR/xml/#uniqattspec
    DuplicateAttribute(String),
    /// A name does not match the [`QName`] production of Namespaces in XML, or a
    /// name that must be an `NCName` contains a colon. Only reported while
    /// namespace processing is active, that is for the
    /// [`NsReader`](crate::reader::NsReader).
    ///
    /// [`QName`]: https://www.w3.org/TR/xml-names/#NT-QName
    InvalidQName(String),
    /// A namespace declaration binds a prefix to the empty namespace name, which
    /// [Namespaces in XML 1.0] does not allow.
    ///
    /// [Namespaces in XML 1.0]: https://www.w3.org/TR/xml-names/#nsc-NSDeclared
    EmptyPrefixBinding(String),
    /// A start, end or empty-element tag does not match the [`STag`] production.
    /// Contains a description of the expectation that was violated.
    ///
    /// [`STag`]: https://www.w3.org/TR/xml/#NT-STag
    MalformedTag(String),
    /// Character data contains the `]]>` sequence outside of a CDATA section.
    CDataEndInContent,
    /// A processing instruction uses the reserved target `xml` (in any case
    /// combination).
    ReservedPiTarget(String),
    /// An XML declaration was found somewhere other than at the very beginning
    /// of the document entity.
    MisplacedXmlDeclaration,
    /// The XML declaration does not match the [`XMLDecl`] production. Contains a
    /// description of the expectation that was violated.
    ///
    /// [`XMLDecl`]: https://www.w3.org/TR/xml/#NT-XMLDecl
    InvalidXmlDeclaration(String),
    /// A `<!DOCTYPE>` declaration appears after the root element started, or a
    /// second one appears.
    MisplacedDoctype,
    /// The document contains no root element.
    MissingRootElement,
    /// A second top-level element was found after the root element was closed.
    MultipleRootElements,
    /// Character data, a CDATA section or an entity reference was found outside
    /// of the root element.
    ContentOutsideRoot,
    /// The document type declaration does not match the [`doctypedecl`]
    /// production. Contains a description of the expectation that was violated.
    ///
    /// [`doctypedecl`]: https://www.w3.org/TR/xml/#NT-doctypedecl
    InvalidDoctype(String),
}

impl fmt::Display for WellFormedError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::InvalidCharacter(ch) => {
                write!(
                    f,
                    "character U+{:04X} is not allowed by the `Char` production",
                    *ch as u32
                )
            }
            Self::InvalidName(name) => {
                write!(f, "`{}` does not match the `Name` production", name)
            }
            Self::InvalidPublicId(id) => write!(
                f,
                "public identifier `{}` contains a character outside `PubidChar`",
                id
            ),
            Self::InvalidReference(reference) => {
                write!(
                    f,
                    "`{}` is not a valid entity or character reference",
                    reference
                )
            }
            Self::UndeclaredEntity(name) => {
                write!(f, "reference to the undeclared entity `{}`", name)
            }
            Self::RecursiveEntity(name) => {
                write!(f, "entity `{}` refers to itself", name)
            }
            Self::EntityExpansionLimit { entity, limit } => write!(
                f,
                "expanding the entity `{}` exceeds the {} of this processor",
                entity, limit
            ),
            Self::UnparsedEntityReference(name) => write!(
                f,
                "reference to the unparsed (NDATA) entity `{}` is not allowed here",
                name
            ),
            Self::IllFormedEntity { entity, reason } => write!(
                f,
                "replacement text of the entity `{}` is not well-formed: {}",
                entity, reason
            ),
            Self::LessThanInAttributeValue(name) => write!(
                f,
                "value of the attribute `{}` contains a literal `<`",
                name
            ),
            Self::ExternalEntityInAttributeValue(name) => write!(
                f,
                "an attribute value cannot refer to the external entity `{}`",
                name
            ),
            Self::DuplicateAttribute(name) => {
                write!(f, "the attribute `{}` is specified more than once", name)
            }
            Self::InvalidQName(name) => {
                write!(f, "`{}` does not match the `QName` production", name)
            }
            Self::EmptyPrefixBinding(prefix) => write!(
                f,
                "the prefix `{}` cannot be bound to the empty namespace name",
                prefix
            ),
            Self::MalformedTag(reason) => write!(f, "malformed tag: {}", reason),
            Self::CDataEndInContent => {
                f.write_str("the sequence `]]>` is not allowed in character data")
            }
            Self::ReservedPiTarget(target) => write!(
                f,
                "`{}` is a reserved processing instruction target",
                target
            ),
            Self::MisplacedXmlDeclaration => f.write_str(
                "the XML declaration must be at the very beginning of the document entity",
            ),
            Self::InvalidXmlDeclaration(reason) => {
                write!(f, "malformed XML declaration: {}", reason)
            }
            Self::MisplacedDoctype => {
                f.write_str("`<!DOCTYPE>` must appear once, before the root element")
            }
            Self::MissingRootElement => f.write_str("the document has no root element"),
            Self::MultipleRootElements => {
                f.write_str("the document has more than one root element")
            }
            Self::ContentOutsideRoot => {
                f.write_str("character data is not allowed outside of the root element")
            }
            Self::InvalidDoctype(reason) => {
                write!(f, "malformed document type declaration: {}", reason)
            }
        }
    }
}

impl std::error::Error for WellFormedError {}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// The error type used by this crate.
///
/// Marked `#[non_exhaustive]`: unlike `quick-xml` 0.41, this crate also reports
/// optional well-formedness constraints (see [`NotWellFormed`](Self::NotWellFormed))
/// and will keep growing past that upstream baseline, so a variant match must
/// carry a wildcard arm to stay source-compatible across releases.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum Error {
    /// An XML document cannot be read from the underlying source.
    ///
    /// Contains a reference-counted I/O error to make the error type `Clone`able.
    Io(Arc<IoError>),
    /// The document does not correspond to the XML grammar.
    Syntax(SyntaxError),
    /// The document is not [well-formed](https://www.w3.org/TR/xml11/#dt-wellformed).
    IllFormed(IllFormedError),
    /// The document violates a well-formedness constraint that is checked only
    /// when [`Config::check_well_formedness`] is enabled.
    ///
    /// [`Config::check_well_formedness`]: crate::reader::Config::check_well_formedness
    NotWellFormed(WellFormedError),
    /// Attribute parsing error.
    InvalidAttr(AttrError),
    /// Encoding error.
    Encoding(EncodingError),
    /// Escape error.
    Escape(EscapeError),
    /// The parsed XML has some namespace-related problem.
    Namespace(NamespaceError),
}

impl Error {
    /// Builds the error returned when a [`read_to_end`] family method reaches
    /// end of input before finding the closing tag for `name`.
    ///
    /// [`read_to_end`]: crate::reader::Reader::read_to_end
    pub(crate) fn missed_end(name: crate::name::QName, decoder: crate::encoding::Decoder) -> Self {
        match decoder.decode(name.as_ref()) {
            Ok(name) => IllFormedError::MissingEndTag(name.into_owned()).into(),
            Err(err) => err.into(),
        }
    }

    /// Builds the error returned when the [serde] deserializer is handed a
    /// closing tag with no corresponding open tag anywhere in the document.
    ///
    /// This can only happen when [`Config::allow_unmatched_ends`] lets the
    /// reader emit such an [`Event::End`] instead of rejecting it; the
    /// deserializer has no use for the event, so it is turned into the same
    /// [`IllFormedError::UnmatchedEndTag`] the reader itself would report if
    /// that option were off.
    ///
    /// [serde]: https://serde.rs/
    /// [`Config::allow_unmatched_ends`]: crate::reader::Config::allow_unmatched_ends
    /// [`Event::End`]: crate::events::Event::End
    #[cfg(feature = "serialize")]
    pub(crate) fn unmatched_end(
        name: crate::name::QName,
        decoder: crate::encoding::Decoder,
    ) -> Self {
        match decoder.decode(name.as_ref()) {
            Ok(name) => IllFormedError::UnmatchedEndTag(name.into_owned()).into(),
            Err(err) => err.into(),
        }
    }
}

impl From<IoError> for Error {
    /// Creates a new [`Error::Io`] from the given error.
    #[inline]
    fn from(error: IoError) -> Error {
        match error.kind() {
            IoErrorKind::InvalidData => match error.downcast::<EncodingError>() {
                Ok(err) => Self::Encoding(err),
                Err(err) => Self::Io(Arc::new(err)),
            },
            _ => Self::Io(Arc::new(error)),
        }
    }
}

impl From<SyntaxError> for Error {
    /// Creates a new [`Error::Syntax`] from the given error.
    #[inline]
    fn from(error: SyntaxError) -> Self {
        Self::Syntax(error)
    }
}

impl From<WellFormedError> for Error {
    /// Creates a new [`Error::NotWellFormed`] from the given error.
    #[inline]
    fn from(error: WellFormedError) -> Self {
        Self::NotWellFormed(error)
    }
}

impl From<IllFormedError> for Error {
    /// Creates a new [`Error::IllFormed`] from the given error.
    #[inline]
    fn from(error: IllFormedError) -> Self {
        Self::IllFormed(error)
    }
}

impl From<EncodingError> for Error {
    /// Creates a new [`Error::Encoding`] from the given error.
    #[inline]
    fn from(error: EncodingError) -> Error {
        Self::Encoding(error)
    }
}

impl From<EscapeError> for Error {
    /// Creates a new [`Error::Escape`] from the given error.
    #[inline]
    fn from(error: EscapeError) -> Error {
        Self::Escape(error)
    }
}

impl From<AttrError> for Error {
    #[inline]
    fn from(error: AttrError) -> Self {
        Self::InvalidAttr(error)
    }
}

impl From<NamespaceError> for Error {
    #[inline]
    fn from(error: NamespaceError) -> Self {
        Self::Namespace(error)
    }
}

/// A specialized [`Result`](std::result::Result) with the error hard-wired to [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {}", e),
            Self::Syntax(e) => write!(f, "syntax error: {}", e),
            Self::IllFormed(e) => write!(f, "ill-formed document: {}", e),
            Self::NotWellFormed(e) => write!(f, "not well-formed document: {}", e),
            Self::InvalidAttr(e) => write!(f, "error while parsing attribute: {}", e),
            Self::Encoding(e) => e.fmt(f),
            Self::Escape(e) => e.fmt(f),
            Self::Namespace(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Syntax(e) => Some(e),
            Self::IllFormed(e) => Some(e),
            Self::NotWellFormed(e) => Some(e),
            Self::InvalidAttr(e) => Some(e),
            Self::Encoding(e) => Some(e),
            Self::Escape(e) => Some(e),
            Self::Namespace(e) => Some(e),
        }
    }
}

#[cfg(feature = "serialize")]
#[cfg_attr(docsrs, doc(cfg(feature = "serialize")))]
pub mod serialize {
    //! Serde (de)serialization errors.
    //!
    //! Mirrors the `quick_xml::errors::serialize` module of quick-xml 0.41.

    use super::*;
    use crate::utils::write_byte_string;
    use std::borrow::Cow;
    #[cfg(feature = "overlapped-lists")]
    use std::num::NonZeroUsize;
    use std::str::Utf8Error;

    /// Deserialization error.
    #[derive(Clone, Debug)]
    pub enum DeError {
        /// Serde custom error.
        Custom(String),
        /// XML parsing error.
        InvalidXml(Error),
        /// This error indicates an error in the [`Deserialize`](serde::Deserialize)
        /// implementation when reading a map or a struct: `MapAccess::next_value[_seed]`
        /// was called before `MapAccess::next_key[_seed]`.
        ///
        /// You should check your types that implement the corresponding trait.
        KeyNotRead,
        /// The deserializer encountered a start tag with a specified name when it
        /// was not expecting one. This happens when you try to deserialize a
        /// primitive value (numbers, strings, booleans) from an XML element.
        UnexpectedStart(Vec<u8>),
        /// The [`Reader`] produced [`Event::Eof`] when it was not expected, for
        /// example after producing [`Event::Start`] but before the corresponding
        /// [`Event::End`].
        ///
        /// [`Reader`]: crate::reader::Reader
        /// [`Event::Eof`]: crate::events::Event::Eof
        /// [`Event::Start`]: crate::events::Event::Start
        /// [`Event::End`]: crate::events::Event::End
        UnexpectedEof,
        /// Too many events were skipped while deserializing a sequence; the event
        /// limit was exceeded. The limit is provided as an argument.
        #[cfg(feature = "overlapped-lists")]
        #[cfg_attr(docsrs, doc(cfg(feature = "overlapped-lists")))]
        TooManyEvents(NonZeroUsize),
    }

    impl fmt::Display for DeError {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            match self {
                Self::Custom(s) => f.write_str(s),
                Self::InvalidXml(e) => e.fmt(f),
                Self::KeyNotRead => f.write_str("invalid `Deserialize` implementation: `MapAccess::next_value[_seed]` was called before `MapAccess::next_key[_seed]`"),
                Self::UnexpectedStart(e) => {
                    f.write_str("unexpected `Event::Start(")?;
                    write_byte_string(f, e)?;
                    f.write_str(")`")
                }
                Self::UnexpectedEof => f.write_str("unexpected `Event::Eof`"),
                #[cfg(feature = "overlapped-lists")]
                Self::TooManyEvents(s) => write!(f, "deserializer buffered {} events, limit exceeded", s),
            }
        }
    }

    impl std::error::Error for DeError {
        fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
            match self {
                Self::InvalidXml(e) => Some(e),
                _ => None,
            }
        }
    }

    impl serde::de::Error for DeError {
        fn custom<T: fmt::Display>(msg: T) -> Self {
            Self::Custom(msg.to_string())
        }
    }

    impl From<Error> for DeError {
        #[inline]
        fn from(e: Error) -> Self {
            Self::InvalidXml(e)
        }
    }

    impl From<EscapeError> for DeError {
        #[inline]
        fn from(e: EscapeError) -> Self {
            Self::InvalidXml(e.into())
        }
    }

    impl From<EncodingError> for DeError {
        #[inline]
        fn from(e: EncodingError) -> Self {
            Self::InvalidXml(e.into())
        }
    }

    impl From<AttrError> for DeError {
        #[inline]
        fn from(e: AttrError) -> Self {
            Self::InvalidXml(e.into())
        }
    }

    /// Serialization error.
    #[derive(Clone, Debug)]
    pub enum SeError {
        /// Serde custom error.
        Custom(String),
        /// An XML document cannot be written to the underlying source.
        ///
        /// Contains a reference-counted I/O error to make the error type `Clone`able.
        Io(Arc<IoError>),
        /// Some value could not be formatted.
        Fmt(std::fmt::Error),
        /// The serialized type cannot be represented in XML due to a violation of
        /// the XML rules in the final XML document. For example, an attempt to
        /// serialize a `HashMap<{integer}, ...>` would cause this error because an
        /// [XML name] cannot start with a digit or a hyphen (minus sign). The same
        /// result would occur if the map key is a complex type that cannot be
        /// serialized as a primitive type (i.e. string, char, bool, unit struct or
        /// unit variant).
        ///
        /// [XML name]: https://www.w3.org/TR/xml11/#sec-common-syn
        Unsupported(Cow<'static, str>),
        /// Some value could not be turned into UTF-8.
        NonEncodable(Utf8Error),
    }

    impl fmt::Display for SeError {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            match self {
                Self::Custom(s) => f.write_str(s),
                Self::Io(e) => write!(f, "I/O error: {}", e),
                Self::Fmt(e) => write!(f, "formatting error: {}", e),
                Self::Unsupported(s) => write!(f, "unsupported value: {}", s),
                Self::NonEncodable(e) => write!(f, "malformed UTF-8: {}", e),
            }
        }
    }

    impl ::std::error::Error for SeError {
        fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
            match self {
                Self::Io(e) => Some(e),
                _ => None,
            }
        }
    }

    impl serde::ser::Error for SeError {
        fn custom<T: fmt::Display>(msg: T) -> Self {
            Self::Custom(msg.to_string())
        }
    }

    impl From<IoError> for SeError {
        #[inline]
        fn from(e: IoError) -> Self {
            Self::Io(Arc::new(e))
        }
    }

    impl From<Utf8Error> for SeError {
        #[inline]
        fn from(e: Utf8Error) -> Self {
            Self::NonEncodable(e)
        }
    }

    impl From<fmt::Error> for SeError {
        #[inline]
        fn from(e: fmt::Error) -> Self {
            Self::Fmt(e)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_and_source() {
        let err: Error = SyntaxError::UnclosedTag.into();
        assert_eq!(
            err.to_string(),
            "syntax error: tag not closed: `>` not found before end of input"
        );
        assert!(std::error::Error::source(&err).is_some());
    }

    #[test]
    fn io_invalid_data_stays_io() {
        let io = IoError::new(IoErrorKind::UnexpectedEof, "boom");
        let err: Error = io.into();
        assert!(matches!(err, Error::Io(_)));
    }
}
