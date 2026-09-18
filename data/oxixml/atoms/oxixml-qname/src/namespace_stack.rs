//! [`NamespaceStack`]: a scoped `prefix -> namespace-name` binding table for streaming parsers.

use std::collections::HashMap;

use crate::constants::{XML_NAMESPACE, XMLNS_NAMESPACE};
use crate::error::{NamespaceDeclError, ResolveQNameError};
use crate::expanded_name::{ExpandedName, ExpandedNameRef};
use crate::qname::QName;
use crate::validate::is_ncname;

/// Which edition of [Namespaces in XML](https://www.w3.org/TR/xml-names/) a [`NamespaceStack`]
/// enforces.
///
/// The two editions agree on almost everything; the one behavioral difference this crate
/// implements is whether [`NamespaceStack::declare`] accepts an empty URI for a *prefixed*
/// binding (`xmlns:p=""`) to undeclare it. Undeclaring the *default* namespace with `xmlns=""`
/// is legal under both editions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum NamespacesVersion {
    /// [Namespaces in XML 1.0](https://www.w3.org/TR/xml-names/). A prefixed namespace binding
    /// can never be undeclared once made (in scope) — only the default namespace can, via
    /// `xmlns=""`.
    #[default]
    V1_0,
    /// [Namespaces in XML 1.1](https://www.w3.org/TR/xml-names11/). Any prefixed binding may
    /// also be undeclared with an empty URI, in addition to the default namespace.
    V1_1,
}

/// Whether a name being resolved by [`NamespaceStack::resolve_qname`] is an element name or an
/// attribute name.
///
/// This distinction matters for exactly one rule: an *unprefixed* element name inherits the
/// in-scope default namespace, but an *unprefixed* attribute name never does — per
/// [Namespaces in XML §5.2](https://www.w3.org/TR/xml-names/#defaulting), "the default namespace
/// does not apply to attribute names."
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NameContext {
    /// The name is an element name (a start- or end-tag name).
    Element,
    /// The name is an attribute name.
    Attribute,
}

/// One namespace-scope frame: the prefixes (and whether the default namespace) were touched by
/// [`NamespaceStack::declare`] calls since the matching [`NamespaceStack::push_scope`], so that
/// [`NamespaceStack::pop_scope`] knows exactly what to undo.
#[derive(Debug, Default)]
struct ScopeFrame {
    default_pushed: bool,
    prefixes_pushed: Vec<String>,
}

/// A scoped `prefix -> namespace-name` binding table, matching the shape a streaming XML parser
/// needs: [`push_scope`](Self::push_scope) when entering an element's start-tag, [`declare`](Self::declare)
/// once per `xmlns[:prefix]` attribute found on it, then [`pop_scope`](Self::pop_scope) when
/// leaving the element (after processing its content).
///
/// The `xml` and `xmlns` prefixes are always implicitly in scope — bound to
/// [`crate::XML_NAMESPACE`] and [`crate::XMLNS_NAMESPACE`] respectively — without ever being
/// pushed or declared; [`resolve`](Self::resolve) returns them unconditionally, and
/// [`declare`](Self::declare) rejects any attempt to bind them to something else (see
/// [`NamespaceDeclError`]).
///
/// # Examples
///
/// ```
/// use oxixml_qname::{NameContext, NamespaceStack, NamespacesVersion};
///
/// let mut namespaces = NamespaceStack::new(NamespacesVersion::V1_0);
///
/// // <book xmlns="urn:example:book" xmlns:dc="urn:example:dc">
/// namespaces.push_scope();
/// namespaces.declare(None, "urn:example:book")?;
/// namespaces.declare(Some("dc"), "urn:example:dc")?;
/// assert_eq!(namespaces.resolve(None), Some("urn:example:book"));
///
/// //   <title dc:lang="en">...</title>
/// namespaces.push_scope();
/// let element = namespaces.resolve_qname("title", NameContext::Element)?;
/// assert_eq!(element.namespace(), Some("urn:example:book")); // inherits the default namespace
/// let attr = namespaces.resolve_qname("dc:lang", NameContext::Attribute)?;
/// assert_eq!(attr.namespace(), Some("urn:example:dc"));
/// namespaces.pop_scope(); // </title>
///
/// namespaces.pop_scope(); // </book>
/// assert_eq!(namespaces.resolve(None), None); // back outside every scope
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct NamespaceStack {
    version: NamespacesVersion,
    default_ns: Vec<String>,
    prefixes: HashMap<String, Vec<String>>,
    scopes: Vec<ScopeFrame>,
}

impl NamespaceStack {
    /// Creates an empty `NamespaceStack` enforcing the given [`NamespacesVersion`].
    ///
    /// The stack starts with one open (root) scope, so [`declare`](Self::declare) can be called
    /// immediately; call [`push_scope`](Self::push_scope) for each element that may carry its
    /// own namespace declarations.
    #[must_use]
    pub fn new(version: NamespacesVersion) -> Self {
        Self {
            version,
            default_ns: Vec::new(),
            prefixes: HashMap::new(),
            scopes: vec![ScopeFrame::default()],
        }
    }

    /// The [`NamespacesVersion`] this stack enforces.
    #[must_use]
    pub fn version(&self) -> NamespacesVersion {
        self.version
    }

    /// Opens a new namespace scope, typically on entering an element's start-tag.
    ///
    /// Every `push_scope` should be matched by exactly one later [`pop_scope`](Self::pop_scope)
    /// once that element (and its content) has been fully processed.
    pub fn push_scope(&mut self) {
        self.scopes.push(ScopeFrame::default());
    }

    /// Closes the most recently opened namespace scope, undoing every binding
    /// [`declare`](Self::declare) made since the matching [`push_scope`](Self::push_scope) —
    /// restoring whatever was in scope before it (which may again be nothing, if there was no
    /// enclosing declaration for that prefix).
    ///
    /// The stack is created with one root scope that is never popped: calling `pop_scope` when
    /// only the root scope remains is a no-op, so callers do not need to special-case the
    /// outermost element.
    pub fn pop_scope(&mut self) {
        // Never pop the root scope: it has no matching `push_scope` to pair with, so popping it
        // would leave `declare` with nowhere to record new bindings.
        if self.scopes.len() <= 1 {
            return;
        }
        // `len() > 1` was just checked, so this frame exists.
        let Some(frame) = self.scopes.pop() else {
            return;
        };
        if frame.default_pushed {
            self.default_ns.pop();
        }
        for prefix in frame.prefixes_pushed {
            if let Some(stack) = self.prefixes.get_mut(&prefix) {
                stack.pop();
                if stack.is_empty() {
                    self.prefixes.remove(&prefix);
                }
            }
        }
    }

    /// Declares `prefix -> uri` (or the default namespace, for `prefix = None`) in the current
    /// (most recently [`push_scope`](Self::push_scope)d) scope.
    ///
    /// # Errors
    ///
    /// Returns [`NamespaceDeclError`] if:
    ///
    /// * `prefix` is `Some(_)` but not a syntactically valid `NCName`
    ///   ([`NamespaceDeclError::InvalidPrefixSyntax`]);
    /// * `prefix` is `Some("xmlns")` — the `xmlns` prefix must never be declared
    ///   ([`NamespaceDeclError::ReservedXmlnsPrefix`]);
    /// * `prefix` is `Some("xml")` and `uri` is not [`crate::XML_NAMESPACE`]
    ///   ([`NamespaceDeclError::XmlPrefixMisbound`]);
    /// * `uri` is [`crate::XML_NAMESPACE`] but `prefix` is not `Some("xml")`
    ///   ([`NamespaceDeclError::XmlNamespaceReserved`]);
    /// * `uri` is [`crate::XMLNS_NAMESPACE`], for any `prefix`
    ///   ([`NamespaceDeclError::XmlnsNamespaceReserved`]);
    /// * `uri` is empty, `prefix` is `Some(_)`, and `self.version()` is
    ///   [`NamespacesVersion::V1_0`] ([`NamespaceDeclError::EmptyUriRequiresNamespaces11`]) — an
    ///   empty `uri` for the *default* namespace (`prefix = None`) is always accepted, under
    ///   either version, and undeclares it.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_qname::{NamespaceStack, NamespacesVersion, XML_NAMESPACE};
    ///
    /// let mut namespaces = NamespaceStack::new(NamespacesVersion::V1_0);
    /// assert!(namespaces.declare(Some("dc"), "urn:example:dc").is_ok());
    /// assert!(namespaces.declare(Some("xmlns"), "urn:anything").is_err());
    /// assert!(namespaces.declare(Some("xml"), "urn:wrong").is_err());
    /// assert!(namespaces.declare(Some("xml"), XML_NAMESPACE).is_ok()); // redundant, but legal
    /// assert!(namespaces.declare(Some("other"), XML_NAMESPACE).is_err());
    /// assert!(namespaces.declare(Some("dc"), "").is_err()); // 1.0: can't undeclare a prefix
    /// assert!(namespaces.declare(None, "").is_ok()); // undeclaring the default ns is always ok
    /// ```
    pub fn declare(&mut self, prefix: Option<&str>, uri: &str) -> Result<(), NamespaceDeclError> {
        if let Some(p) = prefix {
            if !is_ncname(p) {
                return Err(NamespaceDeclError::InvalidPrefixSyntax);
            }
        }
        if prefix == Some("xmlns") {
            return Err(NamespaceDeclError::ReservedXmlnsPrefix);
        }
        if prefix == Some("xml") {
            if uri != XML_NAMESPACE {
                return Err(NamespaceDeclError::XmlPrefixMisbound);
            }
        } else if uri == XML_NAMESPACE {
            return Err(NamespaceDeclError::XmlNamespaceReserved);
        }
        if uri == XMLNS_NAMESPACE {
            return Err(NamespaceDeclError::XmlnsNamespaceReserved);
        }
        if uri.is_empty() && prefix.is_some() && self.version == NamespacesVersion::V1_0 {
            return Err(NamespaceDeclError::EmptyUriRequiresNamespaces11);
        }

        // `new` seeds one root frame and `pop_scope` refuses to pop it away, so `scopes` is
        // never empty here.
        let frame = self
            .scopes
            .last_mut()
            .expect("NamespaceStack always retains at least the root scope frame");
        match prefix {
            None => {
                self.default_ns.push(uri.to_owned());
                frame.default_pushed = true;
            }
            Some(p) => {
                self.prefixes
                    .entry(p.to_owned())
                    .or_default()
                    .push(uri.to_owned());
                frame.prefixes_pushed.push(p.to_owned());
            }
        }
        Ok(())
    }

    /// Resolves `prefix` to its currently in-scope namespace name.
    ///
    /// `prefix = None` asks for the default namespace. Returns `None` when there is no
    /// namespace bound — either because nothing declared it, or because it was explicitly
    /// undeclared with an empty URI (see [`declare`](Self::declare)); the two situations are
    /// indistinguishable from `resolve`'s point of view, which is the correct behavior for name
    /// resolution (both mean "this name has no namespace" / "this prefix cannot be used").
    ///
    /// The `xml` and `xmlns` prefixes always resolve, even though nobody declared them.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_qname::{NamespaceStack, NamespacesVersion, XML_NAMESPACE};
    ///
    /// let mut namespaces = NamespaceStack::new(NamespacesVersion::V1_0);
    /// assert_eq!(namespaces.resolve(Some("xml")), Some(XML_NAMESPACE));
    /// assert_eq!(namespaces.resolve(Some("undeclared")), None);
    ///
    /// namespaces.push_scope();
    /// namespaces.declare(Some("dc"), "urn:example:dc")?;
    /// assert_eq!(namespaces.resolve(Some("dc")), Some("urn:example:dc"));
    /// namespaces.pop_scope();
    /// assert_eq!(namespaces.resolve(Some("dc")), None);
    /// # Ok::<(), oxixml_qname::NamespaceDeclError>(())
    /// ```
    #[must_use]
    pub fn resolve(&self, prefix: Option<&str>) -> Option<&str> {
        match prefix {
            Some("xml") => Some(XML_NAMESPACE),
            Some("xmlns") => Some(XMLNS_NAMESPACE),
            Some(p) => self
                .prefixes
                .get(p)
                .and_then(|stack| stack.last())
                .map(String::as_str)
                .filter(|uri| !uri.is_empty()),
            None => self
                .default_ns
                .last()
                .map(String::as_str)
                .filter(|uri| !uri.is_empty()),
        }
    }

    /// Splits `qname` on its colon and resolves the result against this stack in one step,
    /// producing an owned [`ExpandedName`].
    ///
    /// The `context` controls exactly one rule: an unprefixed `qname` inherits the in-scope
    /// default namespace in [`NameContext::Element`] context, but never does in
    /// [`NameContext::Attribute`] context (per
    /// [Namespaces in XML §5.2](https://www.w3.org/TR/xml-names/#defaulting)) — an attribute
    /// with no prefix always has *no* namespace, regardless of any `xmlns="..."` in scope.
    ///
    /// Use [`resolve_qname_ref`](Self::resolve_qname_ref) instead to avoid the allocation, at
    /// the cost of tying the namespace name's lifetime to `qname`'s.
    ///
    /// # Errors
    ///
    /// Returns [`ResolveQNameError::InvalidQName`] if `qname` does not match the `QName`
    /// grammar, or [`ResolveQNameError::UnboundPrefix`] if it has a prefix that
    /// [`resolve`](Self::resolve) cannot find.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_qname::{NameContext, NamespaceStack, NamespacesVersion};
    ///
    /// let mut namespaces = NamespaceStack::new(NamespacesVersion::V1_0);
    /// namespaces.push_scope();
    /// namespaces.declare(None, "urn:example:default")?;
    ///
    /// // Elements without a prefix inherit the default namespace...
    /// let element = namespaces.resolve_qname("title", NameContext::Element)?;
    /// assert_eq!(element.namespace(), Some("urn:example:default"));
    ///
    /// // ...but attributes without a prefix never do.
    /// let attribute = namespaces.resolve_qname("id", NameContext::Attribute)?;
    /// assert_eq!(attribute.namespace(), None);
    ///
    /// assert!(namespaces.resolve_qname("nope:title", NameContext::Element).is_err());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn resolve_qname(
        &self,
        qname: &str,
        context: NameContext,
    ) -> Result<ExpandedName, ResolveQNameError> {
        let parsed = QName::parse(qname).map_err(ResolveQNameError::InvalidQName)?;
        let namespace = self.resolve_prefix_for(parsed.prefix(), context)?;
        Ok(ExpandedName::new(namespace, parsed.local_part()))
    }

    /// Like [`resolve_qname`](Self::resolve_qname), but borrows the local name from `qname` and
    /// the namespace name from `self` instead of allocating a fresh [`ExpandedName`].
    ///
    /// Because [`ExpandedNameRef`] carries a single lifetime for both fields, `self` and
    /// `qname` must share that lifetime `'a` — in practice this means `qname` must live at
    /// least as long as the `&self` borrow used to call this method, which holds naturally when
    /// `qname` is a slice of a buffer that outlives the stack (the usual case for a streaming
    /// parser reading from its own input buffer) but rules out passing a short-lived temporary.
    ///
    /// # Errors
    ///
    /// Same as [`resolve_qname`](Self::resolve_qname).
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_qname::{NameContext, NamespaceStack, NamespacesVersion};
    ///
    /// let mut namespaces = NamespaceStack::new(NamespacesVersion::V1_0);
    /// namespaces.push_scope();
    /// namespaces.declare(Some("dc"), "urn:example:dc")?;
    ///
    /// let qname = "dc:title".to_owned();
    /// let expanded = namespaces.resolve_qname_ref(&qname, NameContext::Element)?;
    /// assert_eq!(expanded.namespace(), Some("urn:example:dc"));
    /// assert_eq!(expanded.local_name(), "title");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn resolve_qname_ref<'a>(
        &'a self,
        qname: &'a str,
        context: NameContext,
    ) -> Result<ExpandedNameRef<'a>, ResolveQNameError> {
        let parsed = QName::parse(qname).map_err(ResolveQNameError::InvalidQName)?;
        // Re-slice `qname` itself (rather than going through `parsed.prefix()`/`local_part()`)
        // so the results borrow with `qname`'s own lifetime `'a`, not the shorter lifetime of
        // the borrow of `parsed` that `Deref`-based accessors are stuck with generically.
        let (prefix, local) = match parsed.colon_offset() {
            Some(idx) => (Some(&qname[..idx]), &qname[idx + 1..]),
            None => (None, qname),
        };
        let namespace = self.resolve_prefix_for(prefix, context)?;
        Ok(ExpandedNameRef::new(namespace, local))
    }

    /// Shared resolution logic for [`resolve_qname`](Self::resolve_qname) and
    /// [`resolve_qname_ref`](Self::resolve_qname_ref): resolves a parsed `QName`'s prefix
    /// according to `context`.
    fn resolve_prefix_for(
        &self,
        prefix: Option<&str>,
        context: NameContext,
    ) -> Result<Option<&str>, ResolveQNameError> {
        match (prefix, context) {
            (Some(p), _) => self
                .resolve(Some(p))
                .map(Some)
                .ok_or_else(|| ResolveQNameError::UnboundPrefix(p.to_owned())),
            (None, NameContext::Element) => Ok(self.resolve(None)),
            (None, NameContext::Attribute) => Ok(None),
        }
    }

    /// Iterates over every namespace binding currently in scope: `(Some(prefix), uri)` for each
    /// prefix with a non-empty top-of-stack binding, plus `(None, uri)` for the default
    /// namespace if one is bound and non-empty.
    ///
    /// This is the raw material canonicalization algorithms (like `oxixml-c14n`) need to render
    /// the namespace-declaration attributes belonging to a node. The implicit `xml`/`xmlns`
    /// bindings are *not* yielded unless they were explicitly [`declare`](Self::declare)d (which
    /// is legal for `xml`, bound to its one fixed namespace name) — they are always resolvable
    /// via [`resolve`](Self::resolve) regardless.
    ///
    /// Iteration order is unspecified.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_qname::{NamespaceStack, NamespacesVersion};
    ///
    /// let mut namespaces = NamespaceStack::new(NamespacesVersion::V1_0);
    /// namespaces.push_scope();
    /// namespaces.declare(None, "urn:example:default")?;
    /// namespaces.declare(Some("dc"), "urn:example:dc")?;
    ///
    /// let mut bindings: Vec<_> = namespaces.bindings().collect();
    /// bindings.sort();
    /// assert_eq!(
    ///     bindings,
    ///     vec![(None, "urn:example:default"), (Some("dc"), "urn:example:dc")]
    /// );
    /// # Ok::<(), oxixml_qname::NamespaceDeclError>(())
    /// ```
    pub fn bindings(&self) -> impl Iterator<Item = (Option<&str>, &str)> + '_ {
        let default_binding = self
            .default_ns
            .last()
            .map(String::as_str)
            .filter(|uri| !uri.is_empty())
            .map(|uri| (None, uri));
        let prefixed = self.prefixes.iter().filter_map(|(prefix, stack)| {
            stack
                .last()
                .map(String::as_str)
                .filter(|uri| !uri.is_empty())
                .map(|uri| (Some(prefix.as_str()), uri))
        });
        default_binding.into_iter().chain(prefixed)
    }
}

impl Default for NamespaceStack {
    /// Equivalent to `NamespaceStack::new(NamespacesVersion::default())`.
    fn default() -> Self {
        Self::new(NamespacesVersion::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn implicit_xml_and_xmlns_always_resolve() {
        let namespaces = NamespaceStack::default();
        assert_eq!(namespaces.resolve(Some("xml")), Some(XML_NAMESPACE));
        assert_eq!(namespaces.resolve(Some("xmlns")), Some(XMLNS_NAMESPACE));
        assert_eq!(namespaces.resolve(Some("undeclared")), None);
        assert_eq!(namespaces.resolve(None), None);
    }

    #[test]
    fn scoping_shadows_and_restores() -> Result<(), NamespaceDeclError> {
        let mut namespaces = NamespaceStack::default();
        namespaces.push_scope();
        namespaces.declare(Some("p"), "urn:one")?;
        assert_eq!(namespaces.resolve(Some("p")), Some("urn:one"));

        namespaces.push_scope();
        namespaces.declare(Some("p"), "urn:two")?;
        assert_eq!(namespaces.resolve(Some("p")), Some("urn:two"));
        namespaces.pop_scope();

        assert_eq!(namespaces.resolve(Some("p")), Some("urn:one")); // shadow restored
        namespaces.pop_scope();
        assert_eq!(namespaces.resolve(Some("p")), None);
        Ok(())
    }

    #[test]
    fn default_namespace_scoping_shadows_and_restores() -> Result<(), NamespaceDeclError> {
        let mut namespaces = NamespaceStack::default();
        namespaces.push_scope();
        namespaces.declare(None, "urn:outer")?;
        namespaces.push_scope();
        namespaces.declare(None, "urn:inner")?;
        assert_eq!(namespaces.resolve(None), Some("urn:inner"));
        namespaces.pop_scope();
        assert_eq!(namespaces.resolve(None), Some("urn:outer"));
        namespaces.pop_scope();
        assert_eq!(namespaces.resolve(None), None);
        Ok(())
    }

    #[test]
    fn pop_scope_on_root_is_a_no_op() {
        let mut namespaces = NamespaceStack::default();
        namespaces.pop_scope();
        namespaces.pop_scope();
        // Still usable afterwards.
        assert!(namespaces.declare(Some("p"), "urn:one").is_ok());
        assert_eq!(namespaces.resolve(Some("p")), Some("urn:one"));
    }

    #[test]
    fn empty_uri_undeclares_default_namespace_under_both_versions() -> Result<(), NamespaceDeclError>
    {
        for version in [NamespacesVersion::V1_0, NamespacesVersion::V1_1] {
            let mut namespaces = NamespaceStack::new(version);
            namespaces.push_scope();
            namespaces.declare(None, "urn:outer")?;
            namespaces.push_scope();
            namespaces.declare(None, "")?;
            assert_eq!(namespaces.resolve(None), None);
            namespaces.pop_scope();
            assert_eq!(namespaces.resolve(None), Some("urn:outer"));
        }
        Ok(())
    }

    #[test]
    fn empty_uri_for_prefix_errors_under_1_0_but_undeclares_under_1_1()
    -> Result<(), NamespaceDeclError> {
        let mut v10 = NamespaceStack::new(NamespacesVersion::V1_0);
        v10.push_scope();
        v10.declare(Some("p"), "urn:one")?;
        assert_eq!(
            v10.declare(Some("p"), ""),
            Err(NamespaceDeclError::EmptyUriRequiresNamespaces11)
        );
        // The failed declare must not have mutated anything observable.
        assert_eq!(v10.resolve(Some("p")), Some("urn:one"));

        let mut v11 = NamespaceStack::new(NamespacesVersion::V1_1);
        v11.push_scope();
        v11.declare(Some("p"), "urn:one")?;
        v11.declare(Some("p"), "")?;
        assert_eq!(v11.resolve(Some("p")), None);
        Ok(())
    }

    #[test]
    fn reserved_prefix_and_namespace_rules_are_enforced() {
        let mut namespaces = NamespaceStack::default();
        namespaces.push_scope();

        assert_eq!(
            namespaces.declare(Some("xmlns"), "urn:anything"),
            Err(NamespaceDeclError::ReservedXmlnsPrefix)
        );
        assert_eq!(
            namespaces.declare(Some("xml"), "urn:wrong"),
            Err(NamespaceDeclError::XmlPrefixMisbound)
        );
        assert!(namespaces.declare(Some("xml"), XML_NAMESPACE).is_ok());
        assert_eq!(
            namespaces.declare(Some("other"), XML_NAMESPACE),
            Err(NamespaceDeclError::XmlNamespaceReserved)
        );
        assert_eq!(
            namespaces.declare(None, XML_NAMESPACE),
            Err(NamespaceDeclError::XmlNamespaceReserved)
        );
        assert_eq!(
            namespaces.declare(Some("other"), XMLNS_NAMESPACE),
            Err(NamespaceDeclError::XmlnsNamespaceReserved)
        );
        assert_eq!(
            namespaces.declare(None, XMLNS_NAMESPACE),
            Err(NamespaceDeclError::XmlnsNamespaceReserved)
        );
        assert_eq!(
            namespaces.declare(Some("1bad"), "urn:whatever"),
            Err(NamespaceDeclError::InvalidPrefixSyntax)
        );
    }

    #[test]
    fn element_vs_attribute_default_namespace_resolution() -> Result<(), Box<dyn std::error::Error>>
    {
        let mut namespaces = NamespaceStack::default();
        namespaces.push_scope();
        namespaces.declare(None, "urn:example:default")?;

        let element = namespaces.resolve_qname("title", NameContext::Element)?;
        assert_eq!(element.namespace(), Some("urn:example:default"));

        let attribute = namespaces.resolve_qname("id", NameContext::Attribute)?;
        assert_eq!(attribute.namespace(), None); // critical: attributes never default

        // Prefixed names resolve identically for both contexts.
        namespaces.declare(Some("dc"), "urn:example:dc")?;
        let prefixed_attr = namespaces.resolve_qname("dc:lang", NameContext::Attribute)?;
        assert_eq!(prefixed_attr.namespace(), Some("urn:example:dc"));
        Ok(())
    }

    #[test]
    fn resolve_qname_rejects_invalid_syntax_and_unbound_prefixes() {
        let namespaces = NamespaceStack::default();
        assert!(matches!(
            namespaces.resolve_qname("", NameContext::Element),
            Err(ResolveQNameError::InvalidQName(_))
        ));
        assert!(matches!(
            namespaces.resolve_qname("nope:title", NameContext::Element),
            Err(ResolveQNameError::UnboundPrefix(prefix)) if prefix == "nope"
        ));
    }

    #[test]
    fn resolve_qname_ref_matches_resolve_qname() -> Result<(), Box<dyn std::error::Error>> {
        let mut namespaces = NamespaceStack::default();
        namespaces.push_scope();
        namespaces.declare(Some("dc"), "urn:example:dc")?;

        let qname = "dc:title".to_owned();
        let owned = namespaces.resolve_qname(&qname, NameContext::Element)?;
        let borrowed = namespaces.resolve_qname_ref(&qname, NameContext::Element)?;
        assert_eq!(owned, borrowed);
        Ok(())
    }

    #[test]
    fn bindings_iterates_only_in_scope_non_empty_declarations() -> Result<(), NamespaceDeclError> {
        let mut namespaces = NamespaceStack::new(NamespacesVersion::V1_1);
        namespaces.push_scope();
        namespaces.declare(None, "urn:example:default")?;
        namespaces.declare(Some("dc"), "urn:example:dc")?;
        namespaces.declare(Some("undeclared-later"), "urn:temp")?;
        namespaces.declare(Some("undeclared-later"), "")?; // explicitly undeclared: excluded

        let mut bindings: Vec<_> = namespaces.bindings().collect();
        bindings.sort_unstable();
        assert_eq!(
            bindings,
            vec![
                (None, "urn:example:default"),
                (Some("dc"), "urn:example:dc"),
            ]
        );
        Ok(())
    }

    #[test]
    fn version_accessor_reflects_construction() {
        assert_eq!(
            NamespaceStack::new(NamespacesVersion::V1_1).version(),
            NamespacesVersion::V1_1
        );
        assert_eq!(NamespaceStack::default().version(), NamespacesVersion::V1_0);
    }
}
