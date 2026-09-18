//! Handling of names according to the W3C [Namespaces in XML 1.1 (Second
//! Edition)][spec] specification.
//!
//! This mirrors quick-xml 0.41 `name`: [`QName`] is a raw, possibly-prefixed
//! name; [`LocalName`], [`Prefix`] and [`Namespace`] are its pieces; and
//! [`NamespaceResolver`] tracks the in-scope `prefix -> namespace` bindings a
//! streaming parser pushes and pops one element at a time, producing a
//! [`ResolveResult`].
//!
//! [spec]: https://www.w3.org/TR/xml-names11

use crate::escape::normalize_xml10_attribute_value;
use crate::escape::{is_xml10_normalization_char, resolve_xml_entity};
use crate::events::attributes::Attribute;
use crate::events::{BytesStart, Event};
use crate::utils::{Bytes, write_byte_string};
use oxixml_support::bytes::memchr;
use std::borrow::Cow;
use std::fmt::{self, Debug, Formatter};
use std::iter::FusedIterator;

/// Some namespace was invalid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NamespaceError {
    /// The specified namespace prefix is unknown; a namespace cannot be resolved for it.
    UnknownPrefix(Vec<u8>),
    /// Attempt to bind the `xml` prefix to something other than
    /// `http://www.w3.org/XML/1998/namespace`. Contains the offending namespace.
    InvalidXmlPrefixBind(Vec<u8>),
    /// Attempt to bind the `xmlns` prefix. Contains the offending namespace.
    InvalidXmlnsPrefixBind(Vec<u8>),
    /// Attempt to bind some prefix (other than `xml`) to
    /// `http://www.w3.org/XML/1998/namespace`. Contains the offending prefix.
    InvalidPrefixForXml(Vec<u8>),
    /// Attempt to bind some prefix to `http://www.w3.org/2000/xmlns/`. Contains
    /// the offending prefix.
    InvalidPrefixForXmlns(Vec<u8>),
    /// A single start tag declared more `xmlns` / `xmlns:*` bindings than the
    /// configured [`NamespaceResolver::max_declarations_per_element`] limit.
    /// Contains the configured limit.
    TooManyDeclarations(usize),
    /// Opening a new element would nest the namespace scope deeper than the
    /// allowed maximum, either because
    /// [`Config::max_element_depth`](crate::reader::Config::max_element_depth)
    /// was reached, or (independent of any configured limit) because the
    /// nesting-level counter itself cannot represent a deeper scope. Contains
    /// the limit that was reached.
    TooDeeplyNested(usize),
}

impl fmt::Display for NamespaceError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::UnknownPrefix(prefix) => {
                f.write_str("unknown namespace prefix '")?;
                write_byte_string(f, prefix)?;
                f.write_str("'")
            }
            Self::InvalidXmlPrefixBind(namespace) => {
                f.write_str("the namespace prefix 'xml' cannot be bound to '")?;
                write_byte_string(f, namespace)?;
                f.write_str("'")
            }
            Self::InvalidXmlnsPrefixBind(namespace) => {
                f.write_str("the namespace prefix 'xmlns' cannot be bound to '")?;
                write_byte_string(f, namespace)?;
                f.write_str("'")
            }
            Self::InvalidPrefixForXml(prefix) => {
                f.write_str("the namespace prefix '")?;
                write_byte_string(f, prefix)?;
                f.write_str("' cannot be bound to 'http://www.w3.org/XML/1998/namespace'")
            }
            Self::InvalidPrefixForXmlns(prefix) => {
                f.write_str("the namespace prefix '")?;
                write_byte_string(f, prefix)?;
                f.write_str("' cannot be bound to 'http://www.w3.org/2000/xmlns/'")
            }
            Self::TooManyDeclarations(limit) => write!(
                f,
                "start tag declares more than {} namespace bindings; \
                 raise the limit with NamespaceResolver::set_max_declarations_per_element",
                limit,
            ),
            Self::TooDeeplyNested(limit) => write!(
                f,
                "element nesting exceeds the maximum depth of {} elements",
                limit,
            ),
        }
    }
}

impl std::error::Error for NamespaceError {}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// A [qualified name] of an element or an attribute, including an optional
/// namespace [prefix](Prefix) and a [local name](LocalName).
///
/// [qualified name]: https://www.w3.org/TR/xml-names11/#dt-qualname
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde-types", derive(serde::Deserialize, serde::Serialize))]
pub struct QName<'a>(pub &'a [u8]);
impl<'a> QName<'a> {
    /// Converts this name to its internal slice representation.
    #[inline(always)]
    pub const fn into_inner(self) -> &'a [u8] {
        self.0
    }

    /// Returns the local part of this qualified name.
    ///
    /// All content up to and including the first `:` character is removed.
    ///
    /// ```
    /// # use oxixml_xml::name::QName;
    /// let simple = QName(b"simple-name");
    /// assert_eq!(simple.local_name().as_ref(), b"simple-name");
    ///
    /// let qname = QName(b"namespace:simple-name");
    /// assert_eq!(qname.local_name().as_ref(), b"simple-name");
    /// ```
    pub fn local_name(&self) -> LocalName<'a> {
        LocalName(self.index().map_or(self.0, |i| &self.0[i + 1..]))
    }

    /// Returns the namespace prefix of this qualified name, or `None` if it has no
    /// prefix (no `:` was found).
    ///
    /// ```
    /// # use oxixml_xml::name::QName;
    /// let simple = QName(b"simple-name");
    /// assert_eq!(simple.prefix(), None);
    ///
    /// let qname = QName(b"prefix:simple-name");
    /// assert_eq!(qname.prefix().as_ref().map(|n| n.as_ref()), Some(b"prefix".as_ref()));
    /// ```
    pub fn prefix(&self) -> Option<Prefix<'a>> {
        self.index().map(|i| Prefix(&self.0[..i]))
    }

    /// The same as `(qname.local_name(), qname.prefix())`, but performs only one
    /// lookup for the `':'` symbol.
    pub fn decompose(&self) -> (LocalName<'a>, Option<Prefix<'a>>) {
        match self.index() {
            None => (LocalName(self.0), None),
            Some(i) => (LocalName(&self.0[i + 1..]), Some(Prefix(&self.0[..i]))),
        }
    }

    /// If this `QName` represents an `xmlns` binding, returns `Some`, otherwise
    /// `None`.
    ///
    /// ```
    /// # use oxixml_xml::name::{QName, PrefixDeclaration};
    /// let qname = QName(b"xmlns");
    /// assert_eq!(qname.as_namespace_binding(), Some(PrefixDeclaration::Default));
    ///
    /// let qname = QName(b"xmlns:prefix");
    /// assert_eq!(qname.as_namespace_binding(), Some(PrefixDeclaration::Named(b"prefix")));
    ///
    /// // This method does not validate the prefix — it can be empty!
    /// let qname = QName(b"xmlns:");
    /// assert_eq!(qname.as_namespace_binding(), Some(PrefixDeclaration::Named(b"")));
    ///
    /// let qname = QName(b"other-name");
    /// assert_eq!(qname.as_namespace_binding(), None);
    ///
    /// let qname = QName(b"xmlns-reserved-name");
    /// assert_eq!(qname.as_namespace_binding(), None);
    /// ```
    pub fn as_namespace_binding(&self) -> Option<PrefixDeclaration<'a>> {
        if self.0.starts_with(b"xmlns") {
            return match self.0.get(5) {
                None => Some(PrefixDeclaration::Default),
                Some(&b':') => Some(PrefixDeclaration::Named(&self.0[6..])),
                _ => None,
            };
        }
        None
    }

    /// Returns the index in the name where the prefix ends.
    #[inline(always)]
    fn index(&self) -> Option<usize> {
        memchr(b':', self.0)
    }
}
impl<'a> Debug for QName<'a> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "QName(")?;
        write_byte_string(f, self.0)?;
        write!(f, ")")
    }
}
impl<'a> AsRef<[u8]> for QName<'a> {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.0
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// A [local (unqualified) name] of an element or an attribute, i.e. a name
/// without a [prefix](Prefix).
///
/// [local (unqualified) name]: https://www.w3.org/TR/xml-names11/#dt-localname
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde-types", derive(serde::Deserialize, serde::Serialize))]
pub struct LocalName<'a>(pub(crate) &'a [u8]);
impl<'a> LocalName<'a> {
    /// Converts this name to its internal slice representation.
    #[inline(always)]
    pub const fn into_inner(self) -> &'a [u8] {
        self.0
    }
}
impl<'a> Debug for LocalName<'a> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "LocalName(")?;
        write_byte_string(f, self.0)?;
        write!(f, ")")
    }
}
impl<'a> AsRef<[u8]> for LocalName<'a> {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.0
    }
}
impl<'a> From<QName<'a>> for LocalName<'a> {
    /// Creates a `LocalName` from a [`QName`].
    ///
    /// ```
    /// # use oxixml_xml::name::{LocalName, QName};
    /// let local: LocalName = QName(b"unprefixed").into();
    /// assert_eq!(local.as_ref(), b"unprefixed");
    ///
    /// let local: LocalName = QName(b"some:prefix").into();
    /// assert_eq!(local.as_ref(), b"prefix");
    /// ```
    #[inline]
    fn from(name: QName<'a>) -> Self {
        Self(name.index().map_or(name.0, |i| &name.0[i + 1..]))
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// A [namespace prefix] part of the [qualified name](QName) of an element or an
/// attribute: the `prefix` in `<prefix:local>` or `prefix:local="value"`.
///
/// [namespace prefix]: https://www.w3.org/TR/xml-names11/#dt-prefix
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde-types", derive(serde::Deserialize, serde::Serialize))]
pub struct Prefix<'a>(&'a [u8]);
impl<'a> Prefix<'a> {
    /// Extracts the internal slice.
    #[inline(always)]
    pub const fn into_inner(self) -> &'a [u8] {
        self.0
    }

    /// Checks whether this prefix is the special prefix `xml`.
    #[inline(always)]
    pub const fn is_xml(&self) -> bool {
        matches!(self.0, b"xml")
    }

    /// Checks whether this prefix is the special prefix `xmlns`.
    #[inline(always)]
    pub const fn is_xmlns(&self) -> bool {
        matches!(self.0, b"xmlns")
    }
}
impl<'a> Debug for Prefix<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Prefix(")?;
        write_byte_string(f, self.0)?;
        write!(f, ")")
    }
}
impl<'a> AsRef<[u8]> for Prefix<'a> {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.0
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// A namespace prefix declaration, `xmlns` or `xmlns:<name>`, as defined in the
/// [Namespaces in XML specification](https://www.w3.org/TR/xml-names11/#ns-decl).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PrefixDeclaration<'a> {
    /// Binds the default namespace. Corresponds to `xmlns` in `xmlns="..."`.
    Default,
    /// Binds a specified prefix to a namespace. Corresponds to a `prefix` in
    /// `xmlns:prefix="..."`, stored as the payload of this variant.
    Named(&'a [u8]),
}
impl<'a> Debug for PrefixDeclaration<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Default => f.write_str("PrefixDeclaration::Default"),
            Self::Named(prefix) => {
                f.write_str("PrefixDeclaration::Named(")?;
                write_byte_string(f, prefix)?;
                f.write_str(")")
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// A [namespace name] that is declared in an `xmlns[:prefix]="namespace name"`.
///
/// [namespace name]: https://www.w3.org/TR/xml-names11/#dt-NSName
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde-types", derive(serde::Deserialize, serde::Serialize))]
pub struct Namespace<'a>(pub &'a [u8]);
impl<'a> Namespace<'a> {
    /// Converts this namespace to its internal slice representation.
    ///
    /// This is a [non-normalized] attribute value, i.e. entity references are not
    /// expanded and space characters are not removed, so two different byte slices
    /// returned from this method can represent the same namespace.
    ///
    /// [non-normalized]: https://www.w3.org/TR/xml11/#AVNormalize
    #[inline(always)]
    pub const fn into_inner(self) -> &'a [u8] {
        self.0
    }
}
impl<'a> Debug for Namespace<'a> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "Namespace(")?;
        write_byte_string(f, self.0)?;
        write!(f, ")")
    }
}
impl<'a> AsRef<[u8]> for Namespace<'a> {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.0
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Result of [prefix](Prefix) resolution produced by [`NamespaceResolver::resolve`].
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum ResolveResult<'ns> {
    /// The qualified name has no prefix and the resolver defines no default
    /// namespace, so the name is not bound to any namespace.
    Unbound,
    /// The [`Prefix`] resolved to the specified namespace.
    Bound(Namespace<'ns>),
    /// The specified prefix was not found in scope.
    Unknown(Vec<u8>),
}
impl<'ns> Debug for ResolveResult<'ns> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Self::Unbound => write!(f, "Unbound"),
            Self::Bound(ns) => write!(f, "Bound({:?})", ns),
            Self::Unknown(p) => {
                write!(f, "Unknown(")?;
                write_byte_string(f, p)?;
                write!(f, ")")
            }
        }
    }
}

impl<'ns> TryFrom<ResolveResult<'ns>> for Option<Namespace<'ns>> {
    type Error = NamespaceError;

    /// Converts this result to an optional namespace, returning
    /// [`NamespaceError::UnknownPrefix`] if the result represents an unknown prefix.
    fn try_from(result: ResolveResult<'ns>) -> Result<Self, NamespaceError> {
        use ResolveResult::*;

        match result {
            Unbound => Ok(None),
            Bound(ns) => Ok(Some(ns)),
            Unknown(p) => Err(NamespaceError::UnknownPrefix(p)),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// An entry that indexes into the namespace buffer with a single binding.
#[derive(Debug, Clone)]
struct NamespaceBinding {
    /// Index of the namespace in the buffer.
    start: usize,
    /// Length of the prefix. If greater than zero, binds the namespace to the
    /// slice `[start..start + prefix_len]`; else defines the default namespace.
    prefix_len: usize,
    /// Length of the namespace name (the URI). An empty value 'removes' the
    /// binding for the extent of its scope.
    value_len: usize,
    /// Level of nesting at which this namespace was declared (the declaring
    /// element is included, so a declaration on the document root has `level = 1`).
    level: usize,
}

impl NamespaceBinding {
    /// Gets the namespace prefix bound by this declaration, or `None` for a
    /// default-namespace declaration (`xmlns="..."`).
    #[inline]
    const fn prefix<'b>(&self, buffer: &'b [u8]) -> Option<Prefix<'b>> {
        if self.prefix_len == 0 {
            None
        } else {
            let (_, prefix) = buffer.split_at(self.start);
            let (prefix, _) = prefix.split_at(self.prefix_len);
            Some(Prefix(prefix))
        }
    }

    /// Gets the namespace name (the URI) of this binding, or [`ResolveResult::Unbound`]
    /// if the binding was reset with `xmlns[:prefix]=""`.
    #[inline]
    const fn namespace<'ns>(&self, buffer: &'ns [u8]) -> ResolveResult<'ns> {
        if self.value_len == 0 {
            ResolveResult::Unbound
        } else {
            let (_, ns) = buffer.split_at(self.start + self.prefix_len);
            let (ns, _) = ns.split_at(self.value_len);
            ResolveResult::Bound(Namespace(ns))
        }
    }
}

/// Storage for the currently defined namespace bindings, used to resolve prefixes
/// into namespaces. Holds all the internal logic to push and pop namespaces with
/// their levels.
#[derive(Clone)]
pub struct NamespaceResolver {
    /// Buffer that contains namespace prefix names and namespace values.
    buffer: Vec<u8>,
    /// A stack of namespace bindings currently in scope.
    bindings: Vec<NamespaceBinding>,
    /// The number of currently open tags, used to know which declarations to
    /// remove when an `End` event is encountered. Widened to `usize` (rather
    /// than a fixed-width integer) so that raising
    /// [`Config::max_element_depth`](crate::reader::Config::max_element_depth)
    /// past a small fixed bound can never overflow this counter; [`push`](Self::push)
    /// still guards the arithmetic with `checked_add` as defense in depth for
    /// callers that manipulate the level directly through [`set_level`](Self::set_level).
    nesting_level: usize,
    /// Maximum number of `xmlns` / `xmlns:*` declarations [`push`](Self::push)
    /// will accept on a single start tag.
    max_declarations_per_element: usize,
}

/// Default limit on the number of `xmlns` / `xmlns:*` declarations
/// [`NamespaceResolver::push`] accepts on a single start tag.
pub const DEFAULT_MAX_DECLARATIONS_PER_ELEMENT: usize = 256;

impl Debug for NamespaceResolver {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("NamespaceResolver")
            .field("buffer", &Bytes(&self.buffer))
            .field("bindings", &self.bindings)
            .field("nesting_level", &self.nesting_level)
            .field(
                "max_declarations_per_element",
                &self.max_declarations_per_element,
            )
            .finish()
    }
}

/// The reserved `xml` namespace binding.
const RESERVED_NAMESPACE_XML: (Prefix, Namespace) = (
    Prefix(b"xml"),
    Namespace(b"http://www.w3.org/XML/1998/namespace"),
);
/// The reserved `xmlns` namespace binding.
const RESERVED_NAMESPACE_XMLNS: (Prefix, Namespace) = (
    Prefix(b"xmlns"),
    Namespace(b"http://www.w3.org/2000/xmlns/"),
);

impl Default for NamespaceResolver {
    fn default() -> Self {
        let mut buffer = Vec::new();
        let mut bindings = Vec::new();
        for ent in &[RESERVED_NAMESPACE_XML, RESERVED_NAMESPACE_XMLNS] {
            let prefix = ent.0.into_inner();
            let uri = ent.1.into_inner();
            bindings.push(NamespaceBinding {
                start: buffer.len(),
                prefix_len: prefix.len(),
                value_len: uri.len(),
                level: 0,
            });
            buffer.extend(prefix);
            buffer.extend(uri);
        }

        Self {
            buffer,
            bindings,
            nesting_level: 0,
            max_declarations_per_element: DEFAULT_MAX_DECLARATIONS_PER_ELEMENT,
        }
    }
}

/// Applies [XML §3.3.3] attribute-value normalization to the raw bytes of an
/// `xmlns` / `xmlns:*` attribute, so that the namespace name a prefix is bound to
/// is the one an application reading the same attribute would see.
///
/// Returns `None` when the raw bytes are already normal — the overwhelmingly
/// common case, which then costs one scan and no allocation — and also when they
/// cannot be normalized here at all: bytes that are not UTF-8, or a reference to
/// an entity declared by the document rather than by the specification, which
/// only the reader holding the internal subset can resolve. Neither is a
/// namespace name this resolver can improve on, and neither is this resolver's
/// to report: the raw bytes are bound, and the reader's own well-formedness
/// checks — which do see the document's entity declarations — finish the job.
///
/// Only the five entities [XML §4.6] predefines are resolved, never a wider set:
/// a namespace name is part of a document's identity, and which names two
/// prefixes bind must not depend on whether this crate was built with HTML
/// entity support.
///
/// The rules applied are XML 1.0's. XML 1.1 additionally folds U+0085 and U+2028
/// as line breaks, but neither is permitted in a namespace name — which
/// [Namespaces in XML §3] requires to be an IRI reference — so the two versions
/// cannot disagree on a namespace name that is valid in the first place.
///
/// [XML §3.3.3]: https://www.w3.org/TR/xml/#AVNormalize
/// [XML §4.6]: https://www.w3.org/TR/xml/#sec-predefined-ent
/// [Namespaces in XML §3]: https://www.w3.org/TR/xml-names11/#ns-decl
fn normalized_namespace(raw: &[u8]) -> Option<Vec<u8>> {
    if !raw.iter().any(is_xml10_normalization_char) {
        return None;
    }
    let text = std::str::from_utf8(raw).ok()?;
    // A depth of 1 is enough: `resolve_xml_entity` never yields a replacement
    // that holds a reference of its own.
    match normalize_xml10_attribute_value(text, 1, resolve_xml_entity) {
        Ok(Cow::Owned(normalized)) => Some(normalized.into_bytes()),
        Ok(Cow::Borrowed(_)) | Err(_) => None,
    }
}

impl NamespaceResolver {
    /// Adds a new binding of `prefix` to `namespace` at the current nesting level.
    ///
    /// The binding is removed when the level is [popped out](Self::pop). May fail
    /// if you try to (re-)declare the reserved prefixes `xml` and `xmlns`.
    ///
    /// ```
    /// # use oxixml_xml::name::{Namespace, NamespaceResolver, PrefixDeclaration, QName, ResolveResult};
    /// let mut resolver = NamespaceResolver::default();
    /// assert_eq!(resolver.resolve_element(QName(b"name")).0, ResolveResult::Unbound);
    /// assert_eq!(
    ///     resolver.resolve_element(QName(b"ns:name")).0,
    ///     ResolveResult::Unknown(b"ns".to_vec()),
    /// );
    ///
    /// resolver.add(PrefixDeclaration::Default, Namespace(b"example.com")).expect("doc example");
    /// resolver.add(PrefixDeclaration::Named(b"ns"), Namespace(b"my:namespace")).expect("doc example");
    ///
    /// assert_eq!(
    ///     resolver.resolve_element(QName(b"name")).0,
    ///     ResolveResult::Bound(Namespace(b"example.com")),
    /// );
    /// assert_eq!(
    ///     resolver.resolve_element(QName(b"ns:name")).0,
    ///     ResolveResult::Bound(Namespace(b"my:namespace")),
    /// );
    /// ```
    pub fn add(
        &mut self,
        prefix: PrefixDeclaration,
        namespace: Namespace,
    ) -> Result<(), NamespaceError> {
        let level = self.nesting_level;
        match prefix {
            PrefixDeclaration::Default => {
                let start = self.buffer.len();
                self.buffer.extend_from_slice(namespace.0);
                self.bindings.push(NamespaceBinding {
                    start,
                    prefix_len: 0,
                    value_len: namespace.0.len(),
                    level,
                });
            }
            PrefixDeclaration::Named(b"xml") => {
                if namespace != RESERVED_NAMESPACE_XML.1 {
                    return Err(NamespaceError::InvalidXmlPrefixBind(namespace.0.to_vec()));
                }
                // Do not add another binding for the `xml` namespace prefix.
            }
            PrefixDeclaration::Named(b"xmlns") => {
                return Err(NamespaceError::InvalidXmlnsPrefixBind(namespace.0.to_vec()));
            }
            PrefixDeclaration::Named(prefix) => {
                if namespace == RESERVED_NAMESPACE_XML.1 {
                    return Err(NamespaceError::InvalidPrefixForXml(prefix.to_vec()));
                } else if namespace == RESERVED_NAMESPACE_XMLNS.1 {
                    return Err(NamespaceError::InvalidPrefixForXmlns(prefix.to_vec()));
                }

                let start = self.buffer.len();
                self.buffer.extend_from_slice(prefix);
                self.buffer.extend_from_slice(namespace.0);
                self.bindings.push(NamespaceBinding {
                    start,
                    prefix_len: prefix.len(),
                    value_len: namespace.0.len(),
                    level,
                });
            }
        }
        Ok(())
    }

    /// Begins a new scope and adds to it all [namespace bindings] found in the
    /// specified start element.
    ///
    /// The namespace name a declaration binds is its attribute's **normalized**
    /// value: [Namespaces in XML §3] defines it as "the attribute value in the
    /// namespace declaration", which [XML §3.3.3] has already normalized by the
    /// time the value reaches an application. So a declaration written with a
    /// literal tab in it binds a name with a *space* where the tab was, while
    /// one written `urn:a&#9;b` binds a name that still holds the tab — a
    /// character reference stands for the character it names, and is not white
    /// space the algorithm folds. Either way the binding is the very string
    /// [`Attribute::normalized_value`](crate::events::attributes::Attribute::normalized_value)
    /// reports for the same attribute: one document cannot have two answers.
    ///
    /// [namespace bindings]: https://www.w3.org/TR/xml-names11/#dt-NSDecl
    /// [Namespaces in XML §3]: https://www.w3.org/TR/xml-names11/#ns-decl
    /// [XML §3.3.3]: https://www.w3.org/TR/xml/#AVNormalize
    pub fn push(&mut self, start: &BytesStart) -> Result<(), NamespaceError> {
        // `usize` cannot realistically be exhausted by genuine XML nesting (it
        // would require more open elements than fit in addressable memory),
        // but a caller could still drive `nesting_level` to `usize::MAX`
        // through direct `set_level` manipulation; guard the arithmetic
        // instead of ever wrapping or panicking.
        self.nesting_level = self
            .nesting_level
            .checked_add(1)
            .ok_or(NamespaceError::TooDeeplyNested(usize::MAX))?;
        let mut count = 0usize;
        for a in start.attributes().with_checks(false) {
            if let Ok(Attribute { key: k, value: v }) = a {
                if let Some(prefix) = k.as_namespace_binding() {
                    if count >= self.max_declarations_per_element {
                        return Err(NamespaceError::TooManyDeclarations(
                            self.max_declarations_per_element,
                        ));
                    }
                    count += 1;
                    match normalized_namespace(&v) {
                        Some(normalized) => self.add(prefix, Namespace(&normalized))?,
                        None => self.add(prefix, Namespace(&v))?,
                    }
                }
            } else {
                break;
            }
        }
        Ok(())
    }

    /// Returns the maximum number of `xmlns` / `xmlns:*` declarations that
    /// [`push`](Self::push) will accept on a single start tag.
    #[inline]
    pub const fn max_declarations_per_element(&self) -> usize {
        self.max_declarations_per_element
    }

    /// Sets the maximum number of `xmlns` / `xmlns:*` declarations that
    /// [`push`](Self::push) will accept on a single start tag. Pass `usize::MAX`
    /// to disable the limit.
    #[inline]
    pub fn set_max_declarations_per_element(&mut self, limit: usize) -> &mut Self {
        self.max_declarations_per_element = limit;
        self
    }

    /// Ends the top-most scope, popping all bindings added by the last call to
    /// [`push`](Self::push) or [`add`](Self::add).
    #[inline]
    pub fn pop(&mut self) {
        self.set_level(self.nesting_level.saturating_sub(1));
    }

    /// Sets the number of [`push`](Self::push) calls not yet followed by a
    /// [`pop`](Self::pop). Setting a value lower than the current [`level`](Self::level)
    /// behaves as if `pop` were called until the level matches.
    ///
    /// ```
    /// # use oxixml_xml::events::BytesStart;
    /// # use oxixml_xml::name::NamespaceResolver;
    /// let mut resolver = NamespaceResolver::default();
    /// assert_eq!(resolver.level(), 0);
    /// resolver.push(&BytesStart::new("tag")).expect("doc example");
    /// assert_eq!(resolver.level(), 1);
    /// resolver.set_level(10);
    /// assert_eq!(resolver.level(), 10);
    /// resolver.pop();
    /// assert_eq!(resolver.level(), 9);
    /// ```
    pub fn set_level(&mut self, level: usize) {
        self.nesting_level = level;
        match self.bindings.iter().rposition(|n| n.level <= level) {
            None => {
                self.buffer.clear();
                self.bindings.clear();
            }
            Some(last_valid_pos) => {
                if let Some(len) = self.bindings.get(last_valid_pos + 1).map(|n| n.start) {
                    self.buffer.truncate(len);
                    self.bindings.truncate(last_valid_pos + 1);
                }
            }
        }
    }

    /// Resolves a potentially qualified **element** or **attribute** name into
    /// _(namespace name, local name)_.
    ///
    /// # Parameters
    /// - `name`: the possibly qualified name to resolve;
    /// - `use_default`: whether to translate an absent prefix to the current
    ///   default namespace. Should be `true` for element names, `false` for
    ///   attribute names.
    #[inline]
    pub fn resolve<'n>(
        &self,
        name: QName<'n>,
        use_default: bool,
    ) -> (ResolveResult<'_>, LocalName<'n>) {
        let (local_name, prefix) = name.decompose();
        (self.resolve_prefix(prefix, use_default), local_name)
    }

    /// Convenience for `resolve(name, true)`, expressing that an element name is
    /// being resolved.
    #[inline]
    pub fn resolve_element<'n>(&self, name: QName<'n>) -> (ResolveResult<'_>, LocalName<'n>) {
        self.resolve(name, true)
    }

    /// Convenience for `resolve(name, false)`, expressing that an attribute name
    /// is being resolved.
    #[inline]
    pub fn resolve_attribute<'n>(&self, name: QName<'n>) -> (ResolveResult<'_>, LocalName<'n>) {
        self.resolve(name, false)
    }

    /// Finds a [namespace name] for a given event, if applicable.
    ///
    /// A namespace is resolved only for [`Start`](Event::Start),
    /// [`Empty`](Event::Empty) and [`End`](Event::End) events; for all other
    /// events [`ResolveResult::Unbound`] is returned.
    ///
    /// [namespace name]: https://www.w3.org/TR/xml-names11/#dt-NSName
    pub fn resolve_event<'i>(&self, event: Event<'i>) -> (ResolveResult<'_>, Event<'i>) {
        use Event::*;

        match event {
            Empty(e) => (self.resolve_prefix(e.name().prefix(), true), Empty(e)),
            Start(e) => (self.resolve_prefix(e.name().prefix(), true), Start(e)),
            End(e) => (self.resolve_prefix(e.name().prefix(), true), End(e)),
            e => (ResolveResult::Unbound, e),
        }
    }

    /// Resolves the given optional prefix (usually obtained from [`QName::prefix`])
    /// into a corresponding namespace.
    pub fn resolve_prefix(&self, prefix: Option<Prefix>, use_default: bool) -> ResolveResult<'_> {
        let mut iter = self.bindings.iter().rev();
        match (prefix, use_default) {
            // Attribute name has no explicit prefix -> Unbound.
            (None, false) => ResolveResult::Unbound,
            // Element name has no explicit prefix -> find nearest default binding.
            (None, true) => match iter.find(|n| n.prefix_len == 0) {
                Some(n) => n.namespace(&self.buffer),
                None => ResolveResult::Unbound,
            },
            // Attribute or element name with an explicit prefix.
            (Some(p), _) => match iter.find(|n| n.prefix(&self.buffer) == prefix) {
                Some(n) if n.value_len != 0 => n.namespace(&self.buffer),
                _ => ResolveResult::Unknown(p.into_inner().to_vec()),
            },
        }
    }

    /// Returns all the bindings currently in effect except the default `xml` and
    /// `xmlns` bindings.
    ///
    /// ```
    /// # use oxixml_xml::events::BytesStart;
    /// # use oxixml_xml::name::{Namespace, NamespaceResolver, PrefixDeclaration};
    /// let mut resolver = NamespaceResolver::default();
    /// assert_eq!(resolver.bindings().collect::<Vec<_>>(), vec![]);
    ///
    /// resolver.push(&BytesStart::from_content("a xmlns=\"a1\" xmlns:a=\"a2\"", 1)).expect("doc example");
    /// assert_eq!(resolver.bindings().collect::<Vec<_>>(), vec![
    ///     (PrefixDeclaration::Default, Namespace(b"a1")),
    ///     (PrefixDeclaration::Named(b"a"), Namespace(b"a2")),
    /// ]);
    /// ```
    #[inline]
    pub const fn bindings(&self) -> NamespaceBindingsIter<'_> {
        NamespaceBindingsIter {
            resolver: self,
            // Skip the two default namespaces `xml:` and `xmlns:`.
            cursor: 2,
        }
    }

    /// Returns all the bindings on the specified `level`, including the default
    /// `xml` and `xmlns` bindings (which live at level 0).
    pub const fn bindings_of(&self, level: usize) -> NamespaceBindingsOfLevelIter<'_> {
        NamespaceBindingsOfLevelIter {
            resolver: self,
            cursor: 0,
            level,
        }
    }

    /// Returns the number of [`push`](Self::push) calls not yet followed by a
    /// [`pop`](Self::pop).
    pub const fn level(&self) -> usize {
        self.nesting_level
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Iterator over the currently declared namespace bindings, yielding
/// _(prefix, namespace)_ pairs. See [`NamespaceResolver::bindings`].
#[derive(Debug, Clone)]
pub struct NamespaceBindingsIter<'a> {
    resolver: &'a NamespaceResolver,
    cursor: usize,
}

impl<'a> Iterator for NamespaceBindingsIter<'a> {
    type Item = (PrefixDeclaration<'a>, Namespace<'a>);

    fn next(&mut self) -> Option<(PrefixDeclaration<'a>, Namespace<'a>)> {
        while let Some(binding) = self.resolver.bindings.get(self.cursor) {
            self.cursor += 1;

            // Skip a binding whose prefix was overridden by a later binding.
            let prefix = binding.prefix(&self.resolver.buffer);
            if self.resolver.bindings[self.cursor..]
                .iter()
                .any(|ne| prefix == ne.prefix(&self.resolver.buffer))
            {
                continue;
            }
            if let ResolveResult::Bound(namespace) = binding.namespace(&self.resolver.buffer) {
                let prefix = match prefix {
                    Some(Prefix(prefix)) => PrefixDeclaration::Named(prefix),
                    None => PrefixDeclaration::Default,
                };
                return Some((prefix, namespace));
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.resolver.bindings.len() - self.cursor))
    }
}

impl<'a> FusedIterator for NamespaceBindingsIter<'a> {}

/// Iterator over the namespace bindings on a specific level, yielding
/// _(prefix, namespace)_ pairs. See [`NamespaceResolver::bindings_of`].
#[derive(Debug, Clone)]
pub struct NamespaceBindingsOfLevelIter<'a> {
    resolver: &'a NamespaceResolver,
    cursor: usize,
    level: usize,
}

impl<'a> Iterator for NamespaceBindingsOfLevelIter<'a> {
    type Item = (PrefixDeclaration<'a>, Namespace<'a>);

    fn next(&mut self) -> Option<(PrefixDeclaration<'a>, Namespace<'a>)> {
        while let Some(binding) = self.resolver.bindings.get(self.cursor) {
            self.cursor += 1;
            if binding.level < self.level {
                continue;
            }
            if binding.level > self.level {
                break;
            }

            if let ResolveResult::Bound(namespace) = binding.namespace(&self.resolver.buffer) {
                let prefix = match binding.prefix(&self.resolver.buffer) {
                    Some(Prefix(prefix)) => PrefixDeclaration::Named(prefix),
                    None => PrefixDeclaration::Default,
                };
                return Some((prefix, namespace));
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.resolver.bindings.len() - self.cursor))
    }
}

impl<'a> FusedIterator for NamespaceBindingsOfLevelIter<'a> {}

#[cfg(test)]
mod tests;
