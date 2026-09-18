//! The [`XIncludeError`] type: every *fatal* error the processor can raise.
//!
//! [XInclude 1.0 §4](https://www.w3.org/TR/xinclude/#events) draws a sharp line between two
//! kinds of problems:
//!
//! * a **resource error** — the referenced resource cannot be fetched, or an `xpointer`
//!   cannot be evaluated against it — which the processor recovers from by running
//!   [`xi:fallback`](https://www.w3.org/TR/xinclude/#fallback_element) processing; and
//! * a **fatal error** — a violation of the XInclude vocabulary itself (a malformed `xi:include`,
//!   an inclusion loop, non-well-formed XML, …) — which the processor cannot recover from at all.
//!
//! Resource errors never leave this crate: [`process_document`](crate::process_document) handles
//! them internally by invoking the relevant `xi:fallback`. Every variant of [`XIncludeError`] is
//! therefore a fatal error in the sense of the specification, surfaced to the caller because
//! processing cannot continue.

use std::fmt;

use oxixml_dom::{BuildError, DomError, NodeId};
use oxixml_iri::IriParseError;

/// A fatal error raised while running [`process_document`](crate::process_document) or
/// [`process_document_with`](crate::process_document_with).
///
/// # Errors
///
/// See the individual variants; each documents the specification clause it enforces.
#[derive(Debug)]
#[non_exhaustive]
pub enum XIncludeError {
    /// The `base_iri` passed to the processor is not a valid absolute IRI.
    InvalidBaseIri {
        /// The offending text.
        value: String,
        /// Why parsing failed.
        source: IriParseError,
    },
    /// An `xml:base` attribute value is not a valid IRI reference, or does not resolve against
    /// the base URI in scope.
    InvalidXmlBase {
        /// The offending attribute value.
        value: String,
        /// Why parsing or resolution failed.
        source: IriParseError,
    },
    /// The `href` attribute of an `xi:include` element is not a valid IRI reference, or does not
    /// resolve against the base URI in scope.
    InvalidHref {
        /// The offending attribute value.
        value: String,
        /// Why parsing or resolution failed.
        source: IriParseError,
        /// The `xi:include` element that carried the attribute.
        at: NodeId,
    },
    /// The `href` attribute contains a fragment identifier, which
    /// [XInclude 1.0 §4.1.1](https://www.w3.org/TR/xinclude/#include_href) forbids: use the
    /// `xpointer` attribute to select a fragment instead.
    FragmentInHref {
        /// The offending attribute value.
        value: String,
        /// The `xi:include` element that carried the attribute.
        at: NodeId,
    },
    /// The `parse` attribute of an `xi:include` element is neither `xml` nor `text`.
    UnsupportedParseType {
        /// The offending attribute value.
        value: String,
        /// The `xi:include` element that carried the attribute.
        at: NodeId,
    },
    /// An `xi:include` element is missing the information needed to locate its resource: either
    /// `href` is absent (or empty) while `parse="xml"` and `xpointer` is also absent, or
    /// `href` is absent (or empty) while `parse="text"`.
    MissingHref {
        /// The `xi:include` element.
        at: NodeId,
    },
    /// The `xpointer` attribute is present on an `xi:include` element whose `parse` attribute is
    /// `text`, which [XInclude 1.0 §4.1.1](https://www.w3.org/TR/xinclude/#include_href)
    /// forbids.
    XpointerNotAllowedForText {
        /// The `xi:include` element.
        at: NodeId,
    },
    /// The `accept` or `accept-language` attribute contains a character outside `#x20`–`#x7E`.
    InvalidHeaderValue {
        /// The attribute's qualified name (`"accept"` or `"accept-language"`).
        attribute: &'static str,
        /// The offending attribute value.
        value: String,
        /// The `xi:include` element that carried the attribute.
        at: NodeId,
    },
    /// The included resource, fetched for `parse="xml"`, is not well-formed XML.
    MalformedXml {
        /// The resolved location of the resource.
        location: String,
        /// Why parsing failed.
        source: BuildError,
        /// The `xi:include` element that referenced the resource.
        at: NodeId,
    },
    /// A resource error occurred (the resource could not be fetched, the `xpointer` did not
    /// select anything, or the selected node was not an element) and the `xi:include` element's
    /// children did not contain exactly one `xi:fallback` element to recover with, as
    /// [XInclude 1.0 §4.4](https://www.w3.org/TR/xinclude/#fallback_element) requires.
    FallbackCount {
        /// How many `xi:fallback` children were actually found.
        found: usize,
        /// The `xi:include` element.
        at: NodeId,
    },
    /// An `xi:fallback` element appears somewhere other than as a direct child of an
    /// `xi:include` element.
    FallbackOutsideInclude {
        /// The stray `xi:fallback` element.
        at: NodeId,
    },
    /// An element in the [`XINCLUDE_NAMESPACE`](crate::XINCLUDE_NAMESPACE) has a local name
    /// other than `include` or `fallback`; XInclude 1.0 reserves the whole namespace to itself.
    UnsupportedXIncludeElement {
        /// The unrecognized local name.
        local: String,
        /// The offending element.
        at: NodeId,
    },
    /// Processing `xi:include` would revisit a `(location, xpointer)` pair already open in the
    /// current inclusion chain, as
    /// [XInclude 1.0 §4.2.7](https://www.w3.org/TR/xinclude/#loops) forbids.
    LoopDetected {
        /// The resolved include location.
        location: String,
        /// The `xpointer` attribute value, if any.
        xpointer: Option<String>,
        /// The `xi:include` element that closed the loop.
        at: NodeId,
    },
    /// The inclusion chain is deeper than
    /// [`XIncludeOptions::max_depth`](crate::XIncludeOptions::max_depth) allows.
    ///
    /// This is a defensive guard the specification does not itself require; genuine,
    /// specification-legal recursion (see [`Self::LoopDetected`] for what is actually
    /// forbidden) is expected to stay well under the default limit.
    MaxDepthExceeded,
    /// The document tree nests plain elements deeper than this processor will walk.
    ///
    /// Unlike [`Self::MaxDepthExceeded`], which bounds nested resource *acquisitions*, this
    /// bounds nesting of ordinary elements the processor must descend through on its way to
    /// finding one — a limit every well-formed XML document read through this workspace's own
    /// reader already satisfies, so this only ever fires on a
    /// [`Document`](oxixml_dom::Document) assembled directly through its tree-mutation API
    /// rather than parsed from text.
    ElementNestingTooDeep,
    /// A [`Document`](oxixml_dom::Document) tree-mutation operation failed while grafting
    /// included content or splicing `xi:fallback` content into place.
    Dom(DomError),
}

impl fmt::Display for XIncludeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBaseIri { value, source } => {
                write!(f, "invalid base IRI {value:?}: {source}")
            }
            Self::InvalidXmlBase { value, source } => {
                write!(f, "invalid xml:base value {value:?}: {source}")
            }
            Self::InvalidHref { value, source, at } => {
                write!(f, "invalid href {value:?} on xi:include #{at}: {source}")
            }
            Self::FragmentInHref { value, at } => write!(
                f,
                "href {value:?} on xi:include #{at} contains a fragment identifier; \
                 use xpointer instead"
            ),
            Self::UnsupportedParseType { value, at } => write!(
                f,
                "parse={value:?} on xi:include #{at} is neither \"xml\" nor \"text\""
            ),
            Self::MissingHref { at } => write!(
                f,
                "xi:include #{at} has no href and no usable xpointer to locate a resource"
            ),
            Self::XpointerNotAllowedForText { at } => write!(
                f,
                "xi:include #{at} has parse=\"text\" and an xpointer attribute, which is forbidden"
            ),
            Self::InvalidHeaderValue {
                attribute,
                value,
                at,
            } => write!(
                f,
                "{attribute}={value:?} on xi:include #{at} contains a character outside #x20-#x7E"
            ),
            Self::MalformedXml {
                location,
                source,
                at,
            } => write!(
                f,
                "the resource at {location:?}, included by xi:include #{at}, is not well-formed \
                 XML: {source}"
            ),
            Self::FallbackCount { found, at } => write!(
                f,
                "xi:include #{at} needs its fallback but has {found} xi:fallback children \
                 (exactly one is required)"
            ),
            Self::FallbackOutsideInclude { at } => write!(
                f,
                "xi:fallback #{at} is not a child of an xi:include element"
            ),
            Self::UnsupportedXIncludeElement { local, at } => {
                write!(f, "unsupported XInclude-namespace element <{local}> #{at}")
            }
            Self::LoopDetected {
                location,
                xpointer,
                at,
            } => match xpointer {
                Some(pointer) => write!(
                    f,
                    "inclusion loop: xi:include #{at} revisits {location:?} with \
                     xpointer={pointer:?}"
                ),
                None => write!(f, "inclusion loop: xi:include #{at} revisits {location:?}"),
            },
            Self::MaxDepthExceeded => {
                f.write_str("the inclusion chain exceeded the configured maximum depth")
            }
            Self::ElementNestingTooDeep => f.write_str(
                "the document nests elements too deeply for this processor to walk safely \
                 (this limit is not configurable)",
            ),
            Self::Dom(err) => write!(f, "DOM error: {err}"),
        }
    }
}

impl std::error::Error for XIncludeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidBaseIri { source, .. }
            | Self::InvalidXmlBase { source, .. }
            | Self::InvalidHref { source, .. } => Some(source),
            Self::MalformedXml { source, .. } => Some(source),
            Self::Dom(err) => Some(err),
            _ => None,
        }
    }
}

impl From<DomError> for XIncludeError {
    fn from(err: DomError) -> Self {
        Self::Dom(err)
    }
}
