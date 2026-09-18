//! A reader that manages namespace declarations found in the input and is able
//! to resolve [qualified names] to [expanded names].
//!
//! The central type is [`NsReader`], a thin namespace-aware wrapper over the
//! low-level [`Reader`]. It mirrors the quick-xml 0.41 `NsReader`: it consumes
//! bytes, streams XML [`Event`]s, and — for [`Start`](Event::Start),
//! [`Empty`](Event::Empty) and [`End`](Event::End) events — tracks the in-scope
//! `xmlns` bindings so that a [`QName`] can be resolved to its namespace.
//!
//! # Example
//!
//! ```
//! use oxixml_xml::events::Event;
//! use oxixml_xml::name::{Namespace, ResolveResult};
//! use oxixml_xml::reader::NsReader;
//!
//! let mut reader = NsReader::from_str(r#"<tag xmlns="urn:example">text</tag>"#);
//! match reader.read_resolved_event().expect("start") {
//!     (ns, Event::Start(e)) => {
//!         assert_eq!(ns, ResolveResult::Bound(Namespace(b"urn:example")));
//!         assert_eq!(e.local_name().as_ref(), b"tag");
//!     }
//!     _ => panic!("expected a start event"),
//! }
//! ```
//!
//! [qualified names]: https://www.w3.org/TR/xml-names11/#dt-qualname
//! [expanded names]: https://www.w3.org/TR/xml-names11/#dt-expname

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::ops::Deref;
use std::path::Path;

use crate::errors::{Result, WellFormedError};
use crate::events::{BytesStart, BytesText, Event};
use crate::name::{
    LocalName, Namespace, NamespaceBindingsIter, NamespaceError, NamespaceResolver, QName,
    ResolveResult,
};
use crate::reader::{Config, Reader, Span, XmlSource};

/// A low level encoding-agnostic XML event reader that performs namespace resolution.
///
/// Consumes bytes and streams XML [`Event`]s, tracking the namespace declarations
/// (`xmlns` and `xmlns:*` attributes) as elements are opened and closed. Unlike the
/// plain [`Reader`], names emitted by this reader can be [resolved](Self::resolve)
/// to their namespace.
///
/// The reader dereferences to the underlying [`Reader`], so all of its inspection
/// methods ([`buffer_position`](Reader::buffer_position),
/// [`error_position`](Reader::error_position), [`decoder`](Reader::decoder), …)
/// are available directly.
///
/// # Examples
///
/// ```
/// use oxixml_xml::events::Event;
/// use oxixml_xml::name::{Namespace, ResolveResult::*};
/// use oxixml_xml::reader::NsReader;
///
/// let mut reader = NsReader::from_str(r#"
///     <x:tag1 xmlns:x="www.xxxx" xmlns:y="www.yyyy" att1 = "test">
///        <y:tag2><!--Test comment-->Test</y:tag2>
///        <y:tag2>Test 2</y:tag2>
///     </x:tag1>
/// "#);
/// reader.config_mut().trim_text(true);
///
/// let mut count = 0;
/// let mut txt = Vec::new();
/// loop {
///     match reader.read_event().expect("valid event") {
///         Event::Start(e) => {
///             count += 1;
///             let (ns, local) = reader.resolve_element(e.name());
///             match local.as_ref() {
///                 b"tag1" => assert_eq!(ns, Bound(Namespace(b"www.xxxx"))),
///                 b"tag2" => assert_eq!(ns, Bound(Namespace(b"www.yyyy"))),
///                 _ => unreachable!(),
///             }
///         }
///         Event::Text(e) => txt.push(e.decode().expect("utf-8").into_owned()),
///         Event::Eof => break,
///         _ => (),
///     }
/// }
/// assert_eq!(count, 3);
/// assert_eq!(txt, vec!["Test".to_string(), "Test 2".to_string()]);
/// ```
#[derive(Debug, Clone)]
pub struct NsReader<R> {
    /// An XML reader
    pub(super) reader: Reader<R>,
    /// A buffer to manage namespaces
    ns_resolver: NamespaceResolver,
    /// We cannot pop data from the namespace stack until the returned `Empty` or
    /// `End` event has been processed by the user, so we only mark that we should
    /// do that in the next [`Self::read_event_impl()`] call.
    pending_pop: bool,
}

/// Builder methods
impl<R> NsReader<R> {
    /// Creates a `NsReader` that reads from a given reader.
    #[inline]
    pub fn from_reader(reader: R) -> Self {
        Self::new(Reader::from_reader(reader))
    }

    /// Returns reference to the parser configuration.
    #[inline]
    pub const fn config(&self) -> &Config {
        self.reader.config()
    }

    /// Returns mutable reference to the parser configuration.
    #[inline]
    pub fn config_mut(&mut self) -> &mut Config {
        self.reader.config_mut()
    }
}

/// Private methods
impl<R> NsReader<R> {
    #[inline]
    fn new(mut reader: Reader<R>) -> Self {
        reader.enable_namespace_constraints();
        Self {
            reader,
            ns_resolver: NamespaceResolver::default(),
            pending_pop: false,
        }
    }

    fn read_event_impl<'i, B>(&mut self, buf: B) -> Result<Event<'i>>
    where
        R: XmlSource<'i, B>,
    {
        self.pop();
        let event = self.reader.read_event_impl(buf);
        self.process_event(event)
    }

    /// Pops the namespace scope of a previously-returned `Empty` or `End` event,
    /// if one is pending.
    ///
    /// `pub(super)` so that the asynchronous implementation in the sibling
    /// [`async_tokio`](super::async_tokio) module can reuse it.
    pub(super) fn pop(&mut self) {
        if self.pending_pop {
            self.ns_resolver.pop();
            self.pending_pop = false;
        }
    }

    /// Updates the namespace resolver from a freshly read event, and schedules a
    /// scope pop for `Empty` and `End` events.
    ///
    /// `pub(super)` so that the asynchronous implementation in the sibling
    /// [`async_tokio`](super::async_tokio) module can reuse it.
    pub(super) fn process_event<'i>(&mut self, event: Result<Event<'i>>) -> Result<Event<'i>> {
        match event {
            Ok(Event::Start(e)) => {
                self.check_element_depth()?;
                self.ns_resolver.push(&e)?;
                self.check_namespace_constraints(&e)?;
                Ok(Event::Start(e))
            }
            Ok(Event::Empty(e)) => {
                self.check_element_depth()?;
                self.ns_resolver.push(&e)?;
                self.check_namespace_constraints(&e)?;
                // Notify the next `read_event_impl()` call that it needs to pop
                // this namespace scope.
                self.pending_pop = true;
                Ok(Event::Empty(e))
            }
            Ok(Event::End(e)) => {
                // Notify the next `read_event_impl()` call that it needs to pop
                // this namespace scope.
                self.pending_pop = true;
                Ok(Event::End(e))
            }
            e => e,
        }
    }

    /// Rejects a `Start`/`Empty` event that would nest the namespace scope
    /// deeper than [`Config::max_element_depth`], instead of letting the
    /// scope stack grow without bound for a pathologically (or maliciously)
    /// deep document.
    ///
    /// Called before [`NamespaceResolver::push`] so that a rejected event
    /// never mutates the resolver: the scope simply never opens, rather than
    /// having to be unwound after the fact.
    fn check_element_depth(&self) -> Result<()> {
        let max = self.reader.config().max_element_depth;
        if self.ns_resolver.level() >= max {
            return Err(NamespaceError::TooDeeplyNested(max).into());
        }
        Ok(())
    }

    /// Normalizes a namespace name so that two spellings differing only by
    /// character references or by references to entities declared in the internal
    /// subset compare equal, as section 2.3 of Namespaces in XML requires.
    fn normalize_namespace(&self, uri: &[u8]) -> Vec<u8> {
        let Ok(text) = std::str::from_utf8(uri) else {
            return uri.to_vec();
        };
        match crate::escape::unescape_with(text, |name| self.reader.declared_entity(name)) {
            Ok(unescaped) => unescaped.into_owned().into_bytes(),
            Err(_) => uri.to_vec(),
        }
    }

    /// Enforces the namespace constraints that need prefix resolution, namely
    /// [Namespace Constraint: Prefix Declared] and the uniqueness of *expanded*
    /// attribute names required by section 5.3 of Namespaces in XML.
    ///
    /// Only active when
    /// [`Config::check_well_formedness`](crate::reader::Config::check_well_formedness)
    /// is enabled; the default path stays `quick-xml`-compatible.
    ///
    /// [Namespace Constraint: Prefix Declared]: https://www.w3.org/TR/xml-names/#nsc-NSDeclared
    fn check_namespace_constraints(&self, start: &BytesStart<'_>) -> Result<()> {
        if !self.reader.config().check_well_formedness {
            return Ok(());
        }
        if let ResolveResult::Unknown(prefix) = self.ns_resolver.resolve_element(start.name()).0 {
            return Err(NamespaceError::UnknownPrefix(prefix).into());
        }
        let mut seen: Vec<(Option<Vec<u8>>, Vec<u8>)> = Vec::new();
        for attribute in start.attributes() {
            let attribute = attribute?;
            let key = attribute.key;
            if key.as_ref() == b"xmlns" || key.as_ref().starts_with(b"xmlns:") {
                continue;
            }
            let (namespace, local) = self.ns_resolver.resolve_attribute(key);
            let namespace = match namespace {
                ResolveResult::Unknown(prefix) => {
                    return Err(NamespaceError::UnknownPrefix(prefix).into());
                }
                ResolveResult::Bound(Namespace(uri)) => Some(self.normalize_namespace(uri)),
                ResolveResult::Unbound => None,
            };
            let expanded = (namespace, local.as_ref().to_vec());
            if seen.contains(&expanded) {
                let name = String::from_utf8_lossy(key.as_ref()).into_owned();
                return Err(WellFormedError::DuplicateAttribute(name).into());
            }
            seen.push(expanded);
        }
        Ok(())
    }
}

/// Getters
impl<R> NsReader<R> {
    /// Consumes `NsReader` returning the underlying reader.
    ///
    /// See [`Reader::into_inner`] for examples.
    #[inline]
    pub fn into_inner(self) -> R {
        self.reader.into_inner()
    }

    /// Gets a reference to the underlying reader.
    #[inline]
    pub const fn get_ref(&self) -> &R {
        self.reader.get_ref()
    }

    /// Gets a mutable reference to the underlying reader.
    #[inline]
    pub fn get_mut(&mut self) -> &mut R {
        self.reader.get_mut()
    }

    /// Returns the storage of namespace bindings associated with this reader.
    #[inline]
    pub const fn resolver(&self) -> &NamespaceResolver {
        &self.ns_resolver
    }

    /// Returns a mutable reference to the storage of namespace bindings associated
    /// with this reader.
    ///
    /// Useful for configuring the resolver, e.g. to change the
    /// [per-element namespace-declaration limit](NamespaceResolver::set_max_declarations_per_element).
    #[inline]
    pub fn resolver_mut(&mut self) -> &mut NamespaceResolver {
        &mut self.ns_resolver
    }

    /// Resolves a potentially qualified **name** into _(namespace name, local name)_.
    ///
    /// *Qualified* names have the form `prefix:local-name` where the `prefix` is
    /// defined by an `xmlns:prefix="the:namespace:uri"` binding on this element or
    /// on any of its ancestors.
    ///
    /// The `attribute` flag controls how a name **without** a prefix is resolved:
    ///
    /// - if `false` (the element case), the reader's current default namespace
    ///   (declared by `xmlns="..."`) is applied;
    /// - if `true` (the attribute case), an unprefixed name is always
    ///   [`Unbound`](ResolveResult::Unbound), because a default namespace never
    ///   applies to attribute names.
    ///
    /// See also [`resolve_element`](Self::resolve_element) and
    /// [`resolve_attribute`](Self::resolve_attribute), which pick the flag for you.
    ///
    /// # Example
    ///
    /// ```
    /// use oxixml_xml::name::{Namespace, QName, ResolveResult::*};
    /// use oxixml_xml::reader::NsReader;
    ///
    /// let mut reader = NsReader::from_str("<tag one:x='1' two:y='2'/>");
    /// let _ = reader.read_event().expect("empty");
    ///
    /// // Note that this example is simplified — the bindings are not declared,
    /// // so the prefixes are `Unknown` here.
    /// assert_eq!(
    ///     reader.resolve(QName(b"one:x"), true),
    ///     (Unknown(b"one".to_vec()), QName(b"x").into()),
    /// );
    /// ```
    #[inline]
    pub fn resolve<'n>(
        &self,
        name: QName<'n>,
        attribute: bool,
    ) -> (ResolveResult<'_>, LocalName<'n>) {
        self.ns_resolver.resolve(name, !attribute)
    }

    /// Resolves a potentially qualified **element name** into
    /// _(namespace name, local name)_.
    ///
    /// An unprefixed element name resolves to the current default namespace, if any.
    /// This is a convenience for [`resolve(name, false)`](Self::resolve).
    ///
    /// # Example
    ///
    /// ```
    /// use oxixml_xml::events::Event;
    /// use oxixml_xml::name::{Namespace, ResolveResult::*};
    /// use oxixml_xml::reader::NsReader;
    ///
    /// let mut reader = NsReader::from_str(r#"<tag xmlns="urn:d"/>"#);
    /// if let Event::Empty(e) = reader.read_event().expect("empty") {
    ///     assert_eq!(
    ///         reader.resolve_element(e.name()),
    ///         (Bound(Namespace(b"urn:d")), e.local_name()),
    ///     );
    /// }
    /// ```
    #[inline]
    pub fn resolve_element<'n>(&self, name: QName<'n>) -> (ResolveResult<'_>, LocalName<'n>) {
        self.ns_resolver.resolve_element(name)
    }

    /// Resolves a potentially qualified **attribute name** into
    /// _(namespace name, local name)_.
    ///
    /// An unprefixed attribute name is never bound to the default namespace, so it
    /// resolves to [`Unbound`](ResolveResult::Unbound). This is a convenience for
    /// [`resolve(name, true)`](Self::resolve).
    #[inline]
    pub fn resolve_attribute<'n>(&self, name: QName<'n>) -> (ResolveResult<'_>, LocalName<'n>) {
        self.ns_resolver.resolve_attribute(name)
    }

    /// Returns an iterator over the namespace bindings currently in scope, yielding
    /// _(prefix, namespace)_ pairs. The reserved `xml` and `xmlns` bindings are not
    /// included.
    ///
    /// # Example
    ///
    /// ```
    /// use oxixml_xml::name::{Namespace, PrefixDeclaration};
    /// use oxixml_xml::reader::NsReader;
    ///
    /// let mut reader = NsReader::from_str(r#"<tag xmlns="d" xmlns:a="a1"/>"#);
    /// let _ = reader.read_event().expect("empty");
    /// assert_eq!(reader.prefixes().collect::<Vec<_>>(), vec![
    ///     (PrefixDeclaration::Default, Namespace(b"d")),
    ///     (PrefixDeclaration::Named(b"a"), Namespace(b"a1")),
    /// ]);
    /// ```
    #[inline]
    pub fn prefixes(&self) -> NamespaceBindingsIter<'_> {
        self.ns_resolver.bindings()
    }
}

impl<R: BufRead> NsReader<R> {
    /// Reads the next event into the given buffer.
    ///
    /// This method manages namespaces but doesn't resolve them automatically. You
    /// should call [`resolve_element`](Self::resolve_element) if you want to get a
    /// namespace.
    ///
    /// You can also use [`read_resolved_event_into`](Self::read_resolved_event_into)
    /// instead if you want to resolve the namespace as soon as you get an event.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_xml::events::Event;
    /// use oxixml_xml::name::{Namespace, ResolveResult::*};
    /// use oxixml_xml::reader::NsReader;
    ///
    /// let mut reader = NsReader::from_reader(r#"
    ///     <x:tag1 xmlns:x="www.xxxx" xmlns:y="www.yyyy" att1 = "test">
    ///        <y:tag2><!--Test comment-->Test</y:tag2>
    ///        <y:tag2>Test 2</y:tag2>
    ///     </x:tag1>
    /// "#.as_bytes());
    /// reader.config_mut().trim_text(true);
    ///
    /// let mut count = 0;
    /// let mut buf = Vec::new();
    /// let mut txt = Vec::new();
    /// loop {
    ///     match reader.read_event_into(&mut buf).expect("valid event") {
    ///         Event::Start(e) => {
    ///             count += 1;
    ///             let (ns, local) = reader.resolve_element(e.name());
    ///             match local.as_ref() {
    ///                 b"tag1" => assert_eq!(ns, Bound(Namespace(b"www.xxxx"))),
    ///                 b"tag2" => assert_eq!(ns, Bound(Namespace(b"www.yyyy"))),
    ///                 _ => unreachable!(),
    ///             }
    ///         }
    ///         Event::Text(e) => txt.push(e.decode().expect("utf-8").into_owned()),
    ///         Event::Eof => break,
    ///         _ => (),
    ///     }
    ///     buf.clear();
    /// }
    /// assert_eq!(count, 3);
    /// assert_eq!(txt, vec!["Test".to_string(), "Test 2".to_string()]);
    /// ```
    #[inline]
    pub fn read_event_into<'b>(&mut self, buf: &'b mut Vec<u8>) -> Result<Event<'b>> {
        self.read_event_impl(buf)
    }

    /// Reads the next event into the given buffer and resolves its namespace (if
    /// applicable).
    ///
    /// Namespace is resolved only for [`Start`](Event::Start),
    /// [`Empty`](Event::Empty) and [`End`](Event::End) events. For all other events
    /// the concept of a namespace is not defined, so a
    /// [`ResolveResult::Unbound`] is returned.
    ///
    /// If you are not interested in namespaces, you can use
    /// [`read_event_into`](Self::read_event_into) which will not automatically
    /// resolve namespaces for you.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_xml::events::Event;
    /// use oxixml_xml::name::{Namespace, QName, ResolveResult::*};
    /// use oxixml_xml::reader::NsReader;
    ///
    /// let mut reader = NsReader::from_reader(r#"
    ///     <x:tag1 xmlns:x="www.xxxx" xmlns:y="www.yyyy" att1 = "test">
    ///        <y:tag2><!--Test comment-->Test</y:tag2>
    ///     </x:tag1>
    /// "#.as_bytes());
    /// reader.config_mut().trim_text(true);
    ///
    /// let mut count = 0;
    /// let mut buf = Vec::new();
    /// loop {
    ///     match reader.read_resolved_event_into(&mut buf).expect("valid event") {
    ///         (Bound(Namespace(b"www.xxxx")), Event::Start(e)) => {
    ///             count += 1;
    ///             assert_eq!(e.local_name(), QName(b"tag1").into());
    ///         }
    ///         (Bound(Namespace(b"www.yyyy")), Event::Start(e)) => {
    ///             count += 1;
    ///             assert_eq!(e.local_name(), QName(b"tag2").into());
    ///         }
    ///         (_, Event::Start(_)) => unreachable!(),
    ///         (_, Event::Eof) => break,
    ///         _ => (),
    ///     }
    ///     buf.clear();
    /// }
    /// assert_eq!(count, 2);
    /// ```
    #[inline]
    pub fn read_resolved_event_into<'b>(
        &mut self,
        buf: &'b mut Vec<u8>,
    ) -> Result<(ResolveResult<'_>, Event<'b>)> {
        let event = self.read_event_impl(buf)?;
        Ok(self.ns_resolver.resolve_event(event))
    }

    /// Reads until the matching end element is found, using the provided buffer as
    /// intermediate storage for events content. This function is supposed to be
    /// called after you already read a [`Start`](Event::Start) event.
    ///
    /// Manages nested cases where parent and child elements have the _literally_
    /// same name.
    ///
    /// If a corresponding [`End`](Event::End) event is not found, an error of type
    /// [`Error::IllFormed`](crate::errors::Error::IllFormed) will be returned.
    ///
    /// The `end` parameter should contain the name of the end element _in the
    /// reader encoding_. It is good practice to always get that parameter using
    /// [`BytesStart::to_end()`](crate::events::BytesStart::to_end).
    #[inline]
    pub fn read_to_end_into(&mut self, end: QName, buf: &mut Vec<u8>) -> Result<Span> {
        // According to https://www.w3.org/TR/xml11/#dt-etag the end name should
        // match the start name literally. See `Config::check_end_names`.
        let result = self.reader.read_to_end_into(end, buf)?;
        // `read_to_end_into` consumed the closing tag. Because nobody can access
        // its content anymore, we directly pop the opening tag's namespace scope.
        self.ns_resolver.pop();
        Ok(result)
    }

    /// Reads the content between start and end tags, including any markup, using the
    /// provided buffer as intermediate storage for events content. This function is
    /// supposed to be called after you already read a [`Start`](Event::Start) event.
    ///
    /// Manages nested cases where parent and child elements have the _literally_
    /// same name.
    ///
    /// This method does not unescape read data, instead it returns content "as is".
    #[inline]
    pub fn read_text_into<'b>(
        &mut self,
        end: QName,
        buf: &'b mut Vec<u8>,
    ) -> Result<BytesText<'b>> {
        let result = self.reader.read_text_into(end, buf)?;
        // `read_text_into` consumed the closing tag, so pop its namespace scope.
        self.ns_resolver.pop();
        Ok(result)
    }
}

impl NsReader<BufReader<File>> {
    /// Creates an XML reader from a file path.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        Ok(Self::new(Reader::from_file(path)?))
    }
}

impl<'i> NsReader<&'i [u8]> {
    /// Creates an XML reader from a string slice.
    #[inline]
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &'i str) -> Self {
        Self::new(Reader::from_str(s))
    }

    /// Reads the next event, borrowing its content from the input buffer.
    ///
    /// This method manages namespaces but doesn't resolve them automatically. You
    /// should call [`resolve_element`](Self::resolve_element) if you want to get a
    /// namespace.
    ///
    /// You can also use [`read_resolved_event`](Self::read_resolved_event) instead
    /// if you want to resolve the namespace as soon as you get an event.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_xml::events::Event;
    /// use oxixml_xml::name::{Namespace, ResolveResult::*};
    /// use oxixml_xml::reader::NsReader;
    ///
    /// let mut reader = NsReader::from_str(r#"
    ///     <x:tag1 xmlns:x="www.xxxx" xmlns:y="www.yyyy" att1 = "test">
    ///        <y:tag2><!--Test comment-->Test</y:tag2>
    ///        <y:tag2>Test 2</y:tag2>
    ///     </x:tag1>
    /// "#);
    /// reader.config_mut().trim_text(true);
    ///
    /// let mut count = 0;
    /// let mut txt = Vec::new();
    /// loop {
    ///     match reader.read_event().expect("valid event") {
    ///         Event::Start(e) => {
    ///             count += 1;
    ///             let (ns, local) = reader.resolve_element(e.name());
    ///             match local.as_ref() {
    ///                 b"tag1" => assert_eq!(ns, Bound(Namespace(b"www.xxxx"))),
    ///                 b"tag2" => assert_eq!(ns, Bound(Namespace(b"www.yyyy"))),
    ///                 _ => unreachable!(),
    ///             }
    ///         }
    ///         Event::Text(e) => txt.push(e.decode().expect("utf-8").into_owned()),
    ///         Event::Eof => break,
    ///         _ => (),
    ///     }
    /// }
    /// assert_eq!(count, 3);
    /// assert_eq!(txt, vec!["Test".to_string(), "Test 2".to_string()]);
    /// ```
    #[inline]
    pub fn read_event(&mut self) -> Result<Event<'i>> {
        self.read_event_impl(())
    }

    /// Reads the next event, borrowing its content from the input buffer, and
    /// resolves its namespace (if applicable).
    ///
    /// Namespace is resolved only for [`Start`](Event::Start),
    /// [`Empty`](Event::Empty) and [`End`](Event::End) events. For all other events
    /// the concept of a namespace is not defined, so a
    /// [`ResolveResult::Unbound`] is returned.
    ///
    /// If you are not interested in namespaces, you can use
    /// [`read_event`](Self::read_event) which will not automatically resolve
    /// namespaces for you.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_xml::events::Event;
    /// use oxixml_xml::name::{Namespace, QName, ResolveResult::*};
    /// use oxixml_xml::reader::NsReader;
    ///
    /// let mut reader = NsReader::from_str(r#"
    ///     <x:tag1 xmlns:x="www.xxxx" xmlns:y="www.yyyy" att1 = "test">
    ///        <y:tag2><!--Test comment-->Test</y:tag2>
    ///     </x:tag1>
    /// "#);
    /// reader.config_mut().trim_text(true);
    ///
    /// let mut count = 0;
    /// loop {
    ///     match reader.read_resolved_event().expect("valid event") {
    ///         (Bound(Namespace(b"www.xxxx")), Event::Start(e)) => {
    ///             count += 1;
    ///             assert_eq!(e.local_name(), QName(b"tag1").into());
    ///         }
    ///         (Bound(Namespace(b"www.yyyy")), Event::Start(e)) => {
    ///             count += 1;
    ///             assert_eq!(e.local_name(), QName(b"tag2").into());
    ///         }
    ///         (_, Event::Start(_)) => unreachable!(),
    ///         (_, Event::Eof) => break,
    ///         _ => (),
    ///     }
    /// }
    /// assert_eq!(count, 2);
    /// ```
    #[inline]
    pub fn read_resolved_event(&mut self) -> Result<(ResolveResult<'_>, Event<'i>)> {
        let event = self.read_event_impl(())?;
        Ok(self.ns_resolver.resolve_event(event))
    }

    /// Reads until the matching end element is found. This function is supposed to
    /// be called after you already read a [`Start`](Event::Start) event.
    ///
    /// Returns a span that covers the content between `>` of an opening tag and `<`
    /// of a closing tag.
    ///
    /// Manages nested cases where parent and child elements have the _literally_
    /// same name.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_xml::events::{BytesStart, Event};
    /// use oxixml_xml::name::{Namespace, ResolveResult};
    /// use oxixml_xml::reader::NsReader;
    ///
    /// let mut reader = NsReader::from_str(r#"
    ///     <outer xmlns="namespace 1">
    ///         <inner xmlns="namespace 2">
    ///             <outer></outer>
    ///         </inner>
    ///         <inner>
    ///             <inner></inner>
    ///             <inner/>
    ///             <outer></outer>
    ///             <p:outer xmlns:p="ns"></p:outer>
    ///             <outer/>
    ///         </inner>
    ///     </outer>
    /// "#);
    /// reader.config_mut().trim_text(true);
    ///
    /// let ns = Namespace(b"namespace 1");
    /// let start = BytesStart::from_content(r#"outer xmlns="namespace 1""#, 5);
    /// let end   = start.to_end().into_owned();
    ///
    /// assert_eq!(
    ///     reader.read_resolved_event().expect("start"),
    ///     (ResolveResult::Bound(ns), Event::Start(start))
    /// );
    ///
    /// // Skip all events to the corresponding end event, handling nested <outer>s.
    /// reader.read_to_end(end.name()).expect("end");
    ///
    /// assert_eq!(
    ///     reader.read_resolved_event().expect("eof"),
    ///     (ResolveResult::Unbound, Event::Eof)
    /// );
    /// ```
    #[inline]
    pub fn read_to_end(&mut self, end: QName) -> Result<Span> {
        let result = self.reader.read_to_end(end)?;
        self.ns_resolver.pop();
        Ok(result)
    }

    /// Reads content between start and end tags, including any markup. This function
    /// is supposed to be called after you already read a [`Start`](Event::Start)
    /// event.
    ///
    /// Manages nested cases where parent and child elements have the _literally_
    /// same name.
    ///
    /// This method does not unescape read data, instead it returns content "as is".
    #[inline]
    pub fn read_text(&mut self, end: QName) -> Result<BytesText<'i>> {
        let result = self.reader.read_text(end)?;
        self.ns_resolver.pop();
        Ok(result)
    }
}

impl<R> Deref for NsReader<R> {
    type Target = Reader<R>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.reader
    }
}

#[cfg(test)]
mod tests;
