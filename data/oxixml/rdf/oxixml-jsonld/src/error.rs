//! Error types for JSON-LD processing.
//!
//! [`JsonLdSyntaxError`] carries a spec [`JsonLdErrorCode`] so that conformance harnesses can
//! assert the exact failure the JSON-LD API mandates, and [`JsonLdParseError`] unions that with
//! the [`std::io::Error`] a byte source may raise.

use oxixml_support::pos::TextPosition;
use std::error::Error;
use std::fmt::{self, Formatter};
use std::io;
use std::ops::Range;

/// A position within a JSON-LD document, re-exported from [`oxixml_json`].
///
/// Its `line` and `column` are **0-based** (the first character of a document is line `0`,
/// column `0`) and its byte `offset` is 0-based too; only its [`Display`](std::fmt::Display)
/// impl switches to the human-facing 1-based convention.
pub use oxixml_json::TextPosition as JsonTextPosition;

/// A [JSON-LD error code](https://www.w3.org/TR/json-ld-api/#dom-jsonlderrorcode).
///
/// Its [`Display`](fmt::Display) form is the exact lowercase phrase the specification uses (e.g.
/// `"invalid @id value"`), which conformance test manifests compare against verbatim.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[non_exhaustive]
pub enum JsonLdErrorCode {
    /// Two properties which expand to the same keyword have been detected.
    CollidingKeywords,
    /// Multiple conflicting indexes have been found for the same node.
    ConflictingIndexes,
    /// Maximum number of `@context` URLs exceeded.
    ContextOverflow,
    /// A cycle in IRI mappings has been detected.
    CyclicIriMapping,
    /// An `@id` entry was encountered whose value was not a string.
    InvalidIdValue,
    /// An invalid value for `@import` has been found.
    InvalidImportValue,
    /// An `@included` block contains an invalid value.
    InvalidIncludedValue,
    /// An `@index` entry was encountered whose value was not a string.
    InvalidIndexValue,
    /// An invalid value for `@nest` has been found.
    InvalidNestValue,
    /// An invalid value for `@prefix` has been found.
    InvalidPrefixValue,
    /// An invalid value for `@propagate` has been found.
    InvalidPropagateValue,
    /// An invalid value for `@protected` has been found.
    InvalidProtectedValue,
    /// An invalid value for an `@reverse` entry has been detected.
    InvalidReverseValue,
    /// The `@version` entry was used in a context with an out of range value.
    InvalidVersionValue,
    /// The value of `@direction` is not `"ltr"`, `"rtl"`, or `null`.
    InvalidBaseDirection,
    /// An invalid base IRI has been detected.
    InvalidBaseIri,
    /// An `@container` entry was encountered whose value was not valid.
    InvalidContainerMapping,
    /// An entry in a context is invalid due to processing mode incompatibility.
    InvalidContextEntry,
    /// An attempt was made to nullify a context containing protected term definitions.
    InvalidContextNullification,
    /// The value of the default language is not a string or null.
    InvalidDefaultLanguage,
    /// A local context contains a term that has an invalid or missing IRI mapping.
    InvalidIriMapping,
    /// An invalid JSON literal was detected.
    InvalidJsonLiteral,
    /// An invalid keyword alias definition has been encountered.
    InvalidKeywordAlias,
    /// An invalid value in a language map has been detected.
    InvalidLanguageMapValue,
    /// An `@language` entry in a term definition had an invalid value.
    InvalidLanguageMapping,
    /// A language-tagged string with an invalid language value was detected.
    InvalidLanguageTaggedString,
    /// A number, `true`, or `false` with an associated language tag was detected.
    InvalidLanguageTaggedValue,
    /// An invalid local context was detected.
    InvalidLocalContext,
    /// No valid context document has been found for a referenced remote context.
    InvalidRemoteContext,
    /// An invalid reverse property definition has been detected.
    InvalidReverseProperty,
    /// An invalid reverse property map has been detected.
    InvalidReversePropertyMap,
    /// An invalid value for a reverse property has been detected.
    InvalidReversePropertyValue,
    /// The local context defined within a term definition is invalid.
    InvalidScopedContext,
    /// A set object or list object with disallowed entries has been detected.
    InvalidSetOrListObject,
    /// The key ordering is not compatible with the streaming profile.
    InvalidStreamingKeyOrder,
    /// An invalid term definition has been detected.
    InvalidTermDefinition,
    /// An `@type` entry in a term definition could not be expanded to an IRI.
    InvalidTypeMapping,
    /// An invalid value for an `@type` entry has been detected.
    InvalidTypeValue,
    /// A typed value with an invalid type was detected.
    InvalidTypedValue,
    /// A value object with disallowed entries has been detected.
    InvalidValueObject,
    /// An invalid value for the `@value` entry of a value object has been detected.
    InvalidValueObjectValue,
    /// An invalid vocabulary mapping has been detected.
    InvalidVocabMapping,
    /// An IRI would be confused with a compact IRI during compaction.
    IriConfusedWithPrefix,
    /// A keyword redefinition has been detected.
    KeywordRedefinition,
    /// The document could not be loaded or parsed as JSON.
    LoadingDocumentFailed,
    /// There was a problem encountered loading a remote context.
    LoadingRemoteContextFailed,
    /// An attempt was made to change the processing mode incompatibly.
    ProcessingModeConflict,
    /// An attempt was made to redefine a protected term.
    ProtectedTermRedefinition,
    /// An RDF 1.2 triple term was met while serializing RDF as JSON-LD.
    ///
    /// This is not a JSON-LD 1.1 error code: the specification predates RDF 1.2 and gives triple
    /// terms no syntax, so this code marks input that cannot be represented rather than dropping
    /// the statement. It is only reachable when `oxixml-model` is built with its `rdf-12` feature.
    UnsupportedTripleTerm,
    /// The document nests arrays/objects deeper than the processor accepts.
    ///
    /// This is not a JSON-LD 1.1 error code: the specification's algorithms are defined
    /// recursively and place no bound on nesting, but an unbounded bound would let an untrusted
    /// document exhaust the native stack and abort the process, so a limit is enforced and
    /// reported instead.
    DocumentNestingTooDeep,
}

impl fmt::Display for JsonLdErrorCode {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::CollidingKeywords => "colliding keywords",
            Self::ConflictingIndexes => "conflicting indexes",
            Self::ContextOverflow => "context overflow",
            Self::CyclicIriMapping => "cyclic IRI mapping",
            Self::InvalidIdValue => "invalid @id value",
            Self::InvalidImportValue => "invalid @import value",
            Self::InvalidIncludedValue => "invalid @included value",
            Self::InvalidIndexValue => "invalid @index value",
            Self::InvalidNestValue => "invalid @nest value",
            Self::InvalidPrefixValue => "invalid @prefix value",
            Self::InvalidPropagateValue => "invalid @propagate value",
            Self::InvalidProtectedValue => "invalid @protected value",
            Self::InvalidReverseValue => "invalid @reverse value",
            Self::InvalidVersionValue => "invalid @version value",
            Self::InvalidBaseDirection => "invalid base direction",
            Self::InvalidBaseIri => "invalid base IRI",
            Self::InvalidContainerMapping => "invalid container mapping",
            Self::InvalidContextEntry => "invalid context entry",
            Self::InvalidContextNullification => "invalid context nullification",
            Self::InvalidDefaultLanguage => "invalid default language",
            Self::InvalidIriMapping => "invalid IRI mapping",
            Self::InvalidJsonLiteral => "invalid JSON literal",
            Self::InvalidKeywordAlias => "invalid keyword alias",
            Self::InvalidLanguageMapValue => "invalid language map value",
            Self::InvalidLanguageMapping => "invalid language mapping",
            Self::InvalidLanguageTaggedString => "invalid language-tagged string",
            Self::InvalidLanguageTaggedValue => "invalid language-tagged value",
            Self::InvalidLocalContext => "invalid local context",
            Self::InvalidRemoteContext => "invalid remote context",
            Self::InvalidReverseProperty => "invalid reverse property",
            Self::InvalidReversePropertyMap => "invalid reverse property map",
            Self::InvalidReversePropertyValue => "invalid reverse property value",
            Self::InvalidScopedContext => "invalid scoped context",
            Self::InvalidSetOrListObject => "invalid set or list object",
            Self::InvalidStreamingKeyOrder => "invalid streaming key order",
            Self::InvalidTermDefinition => "invalid term definition",
            Self::InvalidTypeMapping => "invalid type mapping",
            Self::InvalidTypeValue => "invalid type value",
            Self::InvalidTypedValue => "invalid typed value",
            Self::InvalidValueObject => "invalid value object",
            Self::InvalidValueObjectValue => "invalid value object value",
            Self::InvalidVocabMapping => "invalid vocab mapping",
            Self::IriConfusedWithPrefix => "IRI confused with prefix",
            Self::KeywordRedefinition => "keyword redefinition",
            Self::LoadingDocumentFailed => "loading document failed",
            Self::LoadingRemoteContextFailed => "loading remote context failed",
            Self::ProcessingModeConflict => "processing mode conflict",
            Self::ProtectedTermRedefinition => "protected term redefinition",
            Self::UnsupportedTripleTerm => "unsupported triple term",
            Self::DocumentNestingTooDeep => "document nesting too deep",
        })
    }
}

/// An error in the syntax or semantics of a JSON-LD document.
///
/// Carries an optional [`JsonLdErrorCode`] (present for JSON-LD-level violations, absent for pure
/// JSON syntax errors surfaced from [`oxixml_json`]) and, where available, the location in the
/// source at which the problem was detected.
#[derive(Debug, Clone)]
pub struct JsonLdSyntaxError {
    kind: SyntaxErrorKind,
}

#[derive(Debug, Clone)]
enum SyntaxErrorKind {
    Json {
        message: String,
        location: Range<TextPosition>,
    },
    Coded {
        message: String,
        code: JsonLdErrorCode,
    },
}

impl JsonLdSyntaxError {
    /// The [`JsonLdErrorCode`] related to this error, if it is a JSON-LD-level error.
    #[must_use]
    pub fn code(&self) -> Option<JsonLdErrorCode> {
        match &self.kind {
            SyntaxErrorKind::Json { .. } => None,
            SyntaxErrorKind::Coded { code, .. } => Some(*code),
        }
    }

    /// The location of the error in the source, if known.
    #[must_use]
    pub fn location(&self) -> Option<Range<TextPosition>> {
        match &self.kind {
            SyntaxErrorKind::Json { location, .. } => Some(location.clone()),
            SyntaxErrorKind::Coded { .. } => None,
        }
    }

    /// The human-readable message. Not guaranteed stable across releases.
    #[must_use]
    pub fn message(&self) -> &str {
        match &self.kind {
            SyntaxErrorKind::Json { message, .. } | SyntaxErrorKind::Coded { message, .. } => {
                message
            }
        }
    }

    /// Builds a JSON-LD-level error from a message and a spec error code.
    #[must_use]
    pub fn msg(message: impl Into<String>, code: JsonLdErrorCode) -> Self {
        Self {
            kind: SyntaxErrorKind::Coded {
                message: message.into(),
                code,
            },
        }
    }
}

impl fmt::Display for JsonLdSyntaxError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match &self.kind {
            SyntaxErrorKind::Json { message, location } => {
                let start = location.start;
                write!(
                    f,
                    "JSON parse error at line {} column {}: {message}",
                    start.line + 1,
                    start.column + 1
                )
            }
            SyntaxErrorKind::Coded { message, code } => write!(f, "{code}: {message}"),
        }
    }
}

impl Error for JsonLdSyntaxError {}

impl From<oxixml_json::JsonSyntaxError> for JsonLdSyntaxError {
    fn from(error: oxixml_json::JsonSyntaxError) -> Self {
        Self {
            kind: SyntaxErrorKind::Json {
                message: error.message().to_owned(),
                location: error.location(),
            },
        }
    }
}

impl From<JsonLdSyntaxError> for io::Error {
    fn from(error: JsonLdSyntaxError) -> Self {
        io::Error::new(io::ErrorKind::InvalidData, error.to_string())
    }
}

/// An error while parsing JSON-LD from a byte source: either malformed input or an I/O failure.
#[derive(Debug)]
pub enum JsonLdParseError {
    /// An I/O error from the underlying byte source.
    Io(io::Error),
    /// A JSON-LD syntax or semantic error.
    Syntax(JsonLdSyntaxError),
}

impl fmt::Display for JsonLdParseError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => e.fmt(f),
            Self::Syntax(e) => e.fmt(f),
        }
    }
}

impl Error for JsonLdParseError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(match self {
            Self::Io(e) => e,
            Self::Syntax(e) => e,
        })
    }
}

impl From<JsonLdSyntaxError> for JsonLdParseError {
    fn from(error: JsonLdSyntaxError) -> Self {
        Self::Syntax(error)
    }
}

impl From<io::Error> for JsonLdParseError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<oxixml_json::JsonParseError> for JsonLdParseError {
    fn from(error: oxixml_json::JsonParseError) -> Self {
        match error {
            oxixml_json::JsonParseError::Io(e) => Self::Io(e),
            oxixml_json::JsonParseError::Syntax(e) => Self::Syntax(e.into()),
        }
    }
}

impl From<oxixml_json::JsonSyntaxError> for JsonLdParseError {
    fn from(error: oxixml_json::JsonSyntaxError) -> Self {
        Self::Syntax(error.into())
    }
}

impl From<JsonLdParseError> for io::Error {
    fn from(error: JsonLdParseError) -> Self {
        match error {
            JsonLdParseError::Io(e) => e,
            JsonLdParseError::Syntax(e) => e.into(),
        }
    }
}
