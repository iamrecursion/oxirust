//! The two contexts an XPath expression is evaluated against.
//!
//! [XPath 3.1 §2.1](https://www.w3.org/TR/xpath-31/#context) splits the *expression context* in
//! two, and this crate keeps the split:
//!
//! * the **static context** ([`StaticContext`]) is everything the compiler needs and nothing it
//!   does not — in-scope namespaces, the default element and function namespaces, the names of
//!   the variables the expression may refer to, the function library, the static base URI, and
//!   the default collation. It is fixed when [`XPath::compile`](crate::XPath::compile) is
//!   called, and a [`CompiledXPath`](crate::CompiledXPath) holds it for the rest of its life.
//! * the **dynamic context** ([`DynamicContext`]) is everything evaluation needs — the focus
//!   (context item, position and size), the values bound to variables, the stable current
//!   dateTime and implicit timezone, and the resolver that answers `fn:doc`.

use crate::error::{XPathError, XPathErrorCode, XPathResult};
use crate::types::SequenceType;
use crate::value::{XpathItem, XpathValue};
use crate::version::XPathVersion;
use oxixml_fn::{
    CODEPOINT_COLLATION, Collation, FN_NAMESPACE, FnContext, FunctionRegistry, builtins,
};
use oxixml_infoset::{ExpandedName, Item, NodeRef, XSD_NAMESPACE};
use oxixml_qname::XML_NAMESPACE;
use oxixml_xsd::{DateTime, TimezoneOffset};
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

/// The namespace of the XPath 3.1 `math:` functions.
pub const MATH_NAMESPACE: &str = "http://www.w3.org/2005/xpath-functions/math";
/// The namespace of the XPath 3.1 `map:` functions.
pub const MAP_NAMESPACE: &str = "http://www.w3.org/2005/xpath-functions/map";
/// The namespace of the XPath 3.1 `array:` functions.
pub const ARRAY_NAMESPACE: &str = "http://www.w3.org/2005/xpath-functions/array";
/// The namespace of the XML Schema instance attributes, `xsi:`.
pub const XSI_NAMESPACE: &str = "http://www.w3.org/2001/XMLSchema-instance";
/// The namespace every XPath error code lives in, `err:`.
pub const ERR_NAMESPACE: &str = "http://www.w3.org/2005/xqt-errors";

/// Supplies the documents `fn:doc`, `fn:collection` and `fn:unparsed-text` return.
///
/// The specification leaves the *available documents* entirely to the implementation, and a
/// Pure-Rust library has no business reading the filesystem or the network behind its caller's
/// back. Nothing is available until an application installs a resolver with
/// [`DynamicContext::with_document_resolver`].
///
/// # Examples
///
/// ```
/// use oxixml_infoset::{NodeRef, Tree};
/// use oxixml_xpath::{DocumentResolver, DynamicContext, StaticContext, XPath, XPathVersion};
/// use std::sync::Arc;
///
/// #[derive(Debug)]
/// struct OneDocument(Tree);
///
/// impl DocumentResolver for OneDocument {
///     fn document(&self, uri: &str) -> Option<NodeRef> {
///         (uri == "urn:example:doc").then(|| self.0.root())
///     }
/// }
///
/// let compiled = XPath::compile(
///     "doc('urn:example:doc')/greeting/string()",
///     XPathVersion::XPath31,
///     StaticContext::new(),
/// )?;
/// let context = DynamicContext::new(Arc::new(StaticContext::new()))
///     .with_document_resolver(Arc::new(OneDocument(Tree::parse_str("<greeting>hi</greeting>")?)));
/// assert_eq!(compiled.evaluate_to_string_in(&context)?, "hi");
/// # Ok::<(), oxixml_xpath::XPathError>(())
/// ```
pub trait DocumentResolver: fmt::Debug + Send + Sync {
    /// The document `fn:doc(uri)` returns, or `None` when the URI names nothing.
    fn document(&self, uri: &str) -> Option<NodeRef>;

    /// The nodes `fn:collection(uri)` returns; `None` selects the default collection.
    ///
    /// This is the node-only reading of a collection, which is what a resolver over a set of
    /// documents wants. A resolver whose collections may hold other items implements
    /// [`DocumentResolver::collection_items`] instead — and needs no implementation of this at
    /// all, since nothing consults it directly.
    fn collection(&self, uri: Option<&str>) -> Option<Vec<NodeRef>> {
        let _ = uri;
        None
    }

    /// The *items* `fn:collection(uri)` returns; `None` selects the default collection.
    ///
    /// [F&O 3.1 §14.7.3](https://www.w3.org/TR/xpath-functions-31/#func-collection) defines the
    /// available collections as a mapping from URI to "a sequence of items", not to a sequence of
    /// *nodes*: a collection may perfectly well be `(1, 'hello', 1e0)`. This is the method
    /// `fn:collection` and `fn:uri-collection` consult, and the one to implement for a
    /// collection that is not a set of documents.
    ///
    /// The default implementation lifts [`DocumentResolver::collection`]'s nodes into items, so a
    /// resolver written before this method existed keeps working unchanged; overriding this one
    /// replaces that reading entirely, and `collection` may then be left alone.
    ///
    /// `fn:uri-collection` answers the document URIs of whichever items are nodes, so an atomic
    /// collection yields the empty sequence there rather than a failure.
    fn collection_items(&self, uri: Option<&str>) -> Option<Vec<Item>> {
        Some(self.collection(uri)?.into_iter().map(Item::Node).collect())
    }

    /// The text `fn:unparsed-text(uri, encoding)` returns.
    fn unparsed_text(&self, uri: &str, encoding: Option<&str>) -> Option<String> {
        let _ = (uri, encoding);
        None
    }

    /// The same retrieval as [`DocumentResolver::unparsed_text`], but saying *why* it failed.
    ///
    /// F&O 3.1 §14.8.2 distinguishes three failures, and a caller that only answers
    /// `Option<String>` collapses them into the first:
    ///
    /// | code | condition |
    /// |---|---|
    /// | `FOUT1170` | the resource cannot be retrieved at all |
    /// | `FOUT1190` | the encoding is not supported, or is not the resource's |
    /// | `FOUT1200` | the octets are not valid in the encoding |
    ///
    /// The default implementation delegates to [`DocumentResolver::unparsed_text`] and reports
    /// [`TextRetrievalError::Unretrievable`], which is exactly the old behaviour; a resolver
    /// that can tell the cases apart overrides this instead.
    fn unparsed_text_result(
        &self,
        uri: &str,
        encoding: Option<&str>,
    ) -> Result<String, TextRetrievalError> {
        self.unparsed_text(uri, encoding)
            .ok_or(TextRetrievalError::Unretrievable)
    }

    /// The text `fn:json-doc(uri)` parses, which is *not* retrieved the way
    /// [`DocumentResolver::unparsed_text_result`] retrieves a text resource.
    ///
    /// The two differ in what an undecodable resource means. `fn:unparsed-text($href)` with no
    /// `$encoding` has nothing that identifies an encoding, so failing to decode it is
    /// "the encoding could not be determined", `FOUT1190` — which is why the catalogue's
    /// `fn-unparsed-text-045` accepts that code for an ISO-8859-1 resource. A JSON resource has
    /// no such ambiguity: [RFC 7159 §8.1](https://www.rfc-editor.org/rfc/rfc7159#section-8.1)
    /// *fixes* the encoding at UTF-8 unless a byte-order mark says otherwise, so octets that do
    /// not decode are invalid octets, `FOUT1200`. The JSON side of the QT3 suite grades it that
    /// way too: every `misc-JsonTestSuite/i_string_*` case offers `FOUT1200` and never
    /// `FOUT1190`.
    ///
    /// The default implementation delegates to [`DocumentResolver::unparsed_text_result`] and
    /// re-labels exactly that one case; a resolver that reads the bytes itself can override it.
    fn json_text_result(&self, uri: &str) -> Result<String, TextRetrievalError> {
        match self.unparsed_text_result(uri, None) {
            Err(TextRetrievalError::UnsupportedEncoding) => Err(TextRetrievalError::InvalidOctets),
            other => other,
        }
    }
}

/// Why [`DocumentResolver::unparsed_text_result`] could not produce the text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TextRetrievalError {
    /// `FOUT1170` — nothing could be retrieved from the URI.
    Unretrievable,
    /// `FOUT1190` — the encoding is not one this processor supports, or is not the encoding
    /// the resource is actually in.
    UnsupportedEncoding,
    /// `FOUT1200` — the octets retrieved are not a valid encoding of characters.
    InvalidOctets,
}

/// Everything the compiler needs to know before it sees a document.
///
/// # Examples
///
/// ```
/// use oxixml_xpath::{StaticContext, XPath, XPathVersion};
///
/// let context = StaticContext::new()
///     .with_namespace("bk", "urn:example:book")
///     .with_default_element_namespace("urn:example:book")
///     .with_variable("title");
///
/// let compiled = XPath::compile("bk:chapter[@n = $title]", XPathVersion::XPath31, context)?;
/// assert_eq!(compiled.expression(), "bk:chapter[@n = $title]");
///
/// // An unbound prefix is a static error, XPST0081.
/// let failure = XPath::compile("nope:x", XPathVersion::XPath31, StaticContext::new())
///     .expect_err("the prefix is not bound");
/// assert_eq!(failure.code(), Some(oxixml_xpath::XPathErrorCode::Xpst0081));
/// # Ok::<(), oxixml_xpath::XPathError>(())
/// ```
#[derive(Clone)]
pub struct StaticContext {
    namespaces: BTreeMap<String, String>,
    default_element_namespace: Option<String>,
    default_function_namespace: String,
    variables: Vec<ExpandedName>,
    functions: Arc<FunctionRegistry>,
    base_uri: Option<String>,
    default_collation: String,
    undeclared_variables: bool,
    xslt_functions: bool,
    xquery_host_language: bool,
    host_signatures: Option<HostSignatures>,
}

/// Supplies the signature a *host-registered* function declares.
///
/// A function this engine serves from [`oxixml_fn`] carries the signature F&O gives it, but one
/// the host put into the [`FunctionRegistry`] — an XQuery `declare function`, an XSLT
/// `xsl:function` — is an opaque closure, and a `function(…) as …` item test would have nothing
/// to judge a reference to it by beyond its arity. A host that knows the declared types answers
/// with them here; `None` for a name it does not know, which leaves the item unsignatured.
pub type HostSignatures =
    Arc<dyn Fn(&ExpandedName, usize) -> Option<(Vec<SequenceType>, SequenceType)> + Send + Sync>;

impl Default for StaticContext {
    fn default() -> Self {
        Self::new()
    }
}

impl StaticContext {
    /// A context with the specification's default in-scope namespaces (`xml`, `xs`, `xsi`,
    /// `fn`, `math`, `map`, `array`, `err`), the `fn:` default function namespace, no default
    /// element namespace, the whole [`oxixml_fn`] library, no base URI, and the code point
    /// collation.
    #[must_use]
    pub fn new() -> Self {
        let mut namespaces = BTreeMap::new();
        namespaces.insert("xml".to_owned(), XML_NAMESPACE.to_owned());
        namespaces.insert("xs".to_owned(), XSD_NAMESPACE.to_owned());
        namespaces.insert("xsi".to_owned(), XSI_NAMESPACE.to_owned());
        namespaces.insert("fn".to_owned(), FN_NAMESPACE.to_owned());
        namespaces.insert("math".to_owned(), MATH_NAMESPACE.to_owned());
        namespaces.insert("map".to_owned(), MAP_NAMESPACE.to_owned());
        namespaces.insert("array".to_owned(), ARRAY_NAMESPACE.to_owned());
        namespaces.insert("err".to_owned(), ERR_NAMESPACE.to_owned());
        Self {
            namespaces,
            default_element_namespace: None,
            default_function_namespace: FN_NAMESPACE.to_owned(),
            variables: Vec::new(),
            functions: Arc::new(builtins()),
            base_uri: None,
            default_collation: CODEPOINT_COLLATION.to_owned(),
            undeclared_variables: false,
            xslt_functions: false,
            xquery_host_language: false,
            host_signatures: None,
        }
    }

    /// Declares that the XSLT function library is in scope alongside F&O's.
    ///
    /// The two libraries overlap, and they did not arrive together.
    /// [XSLT 1.0 §12](https://www.w3.org/TR/xslt-19991116#add-func) already defines
    /// `format-number` and `generate-id`; F&O only added them to *XPath* in 3.0. An XSLT 1.0
    /// stylesheet compiles its expressions as XPath 1.0, so without this flag
    /// `format-number($x, '0.00', 'my-format')` in such a stylesheet is rejected as
    /// `XPST0017` — which is wrong, because the function is the host's, not XPath's.
    ///
    /// Plain XPath leaves the flag off, and QT3 pins that reading: `K-FunctionCallExpr-15`
    /// expects `format-number` to be unavailable under `XP20`.
    #[must_use]
    pub const fn with_xslt_functions(mut self) -> Self {
        self.xslt_functions = true;
        self
    }

    /// Whether the XSLT function library is in scope — see [`Self::with_xslt_functions`].
    #[must_use]
    pub const fn has_xslt_functions(&self) -> bool {
        self.xslt_functions
    }

    /// Declares that the expressions compiled against this context come from an XQuery module.
    ///
    /// XQuery embeds XPath, but the two specifications differ in two places this engine can see,
    /// and a host that hands query text to the XPath compiler has to say which one applies:
    ///
    /// * a **braced URI literal** admits entity and character references. XPath 3.1 §A.2.1 gives
    ///   `BracedURILiteral ::= "Q" "{" [^{}]* "}"`, so `Q{a&#x2e;b}c` names the namespace whose
    ///   URI is the eight characters `a&#x2e;b`; XQuery 3.1 §A.2.1 writes the same terminal as
    ///   `"Q" "{" (PredefinedEntityRef | CharRef | [^&{}])* "}"`, so there the URI is `a.b`. A
    ///   reference never *ends* the literal, however it expands: `Q{&#x7D;u}c` is a URI beginning
    ///   with `}`, not an empty one.
    /// * the target of a **`cast as` or `castable as`** that is not a known simple type is
    ///   `XQST0052` from XQuery 3.0 onwards, where XPath reports `XPST0051`.
    ///
    /// Leave the flag off for XPath.
    #[must_use]
    pub const fn with_xquery_host_language(mut self) -> Self {
        self.xquery_host_language = true;
        self
    }

    /// Whether the expressions come from an XQuery module — see
    /// [`Self::with_xquery_host_language`].
    #[must_use]
    pub const fn is_xquery_host_language(&self) -> bool {
        self.xquery_host_language
    }

    /// Installs the source of declared signatures for host-registered functions.
    ///
    /// See [`HostSignatures`]; without one, a named function reference to a function the host
    /// registered is judged by its arity alone.
    #[must_use]
    pub fn with_host_signatures(mut self, signatures: HostSignatures) -> Self {
        self.host_signatures = Some(signatures);
        self
    }

    /// The signature the host declares for `name#arity`, if it declares one.
    #[must_use]
    pub fn host_signature(
        &self,
        name: &ExpandedName,
        arity: usize,
    ) -> Option<(Vec<SequenceType>, SequenceType)> {
        self.host_signatures.as_ref()?(name, arity)
    }

    /// Binds a namespace prefix.
    ///
    /// The *empty* prefix is not a prefix at all — no lexical `QName` can name it — so binding it
    /// sets the [default element namespace](Self::with_default_element_namespace) instead, which
    /// is what an `xmlns="…"` declaration means and how a host that hands this context a
    /// namespace table (a stylesheet's in-scope namespaces, a test catalogue's `<namespace
    /// prefix="" uri="…"/>`) expects it to be read.
    #[must_use]
    pub fn with_namespace(mut self, prefix: impl Into<String>, uri: impl Into<String>) -> Self {
        let prefix = prefix.into();
        let uri = uri.into();
        if prefix.is_empty() {
            self.default_element_namespace = Some(uri);
            return self;
        }
        self.namespaces.insert(prefix, uri);
        self
    }

    /// Binds several namespace prefixes.
    #[must_use]
    pub fn with_namespaces<K, V>(self, bindings: impl IntoIterator<Item = (K, V)>) -> Self
    where
        K: Into<String>,
        V: Into<String>,
    {
        let mut context = self;
        for (prefix, uri) in bindings {
            context = context.with_namespace(prefix, uri);
        }
        context
    }

    /// Removes a namespace binding.
    #[must_use]
    pub fn without_namespace(mut self, prefix: &str) -> Self {
        self.namespaces.remove(prefix);
        self
    }

    /// Sets the default namespace for unprefixed element and type names.
    ///
    /// This is what makes `chapter` select `{urn:example:book}chapter`. It never applies to
    /// attribute names, to function names, or to variable names.
    #[must_use]
    pub fn with_default_element_namespace(mut self, uri: impl Into<String>) -> Self {
        self.default_element_namespace = Some(uri.into());
        self
    }

    /// Sets the default namespace for unprefixed function names, `fn:` unless changed.
    #[must_use]
    pub fn with_default_function_namespace(mut self, uri: impl Into<String>) -> Self {
        self.default_function_namespace = uri.into();
        self
    }

    /// Declares a variable in no namespace, so that `$name` compiles.
    #[must_use]
    pub fn with_variable(mut self, name: impl Into<String>) -> Self {
        self.variables
            .push(ExpandedName::new(None::<String>, name.into()));
        self
    }

    /// Declares a variable by its expanded name.
    #[must_use]
    pub fn with_variable_name(mut self, name: ExpandedName) -> Self {
        self.variables.push(name);
        self
    }

    /// Replaces the function library.
    #[must_use]
    pub fn with_functions(mut self, functions: FunctionRegistry) -> Self {
        self.functions = Arc::new(functions);
        self
    }

    /// Sets the static base URI, which `fn:resolve-uri`, `fn:doc` and `fn:static-base-uri`
    /// consult.
    #[must_use]
    pub fn with_base_uri(mut self, uri: impl Into<String>) -> Self {
        self.base_uri = Some(uri.into());
        self
    }

    /// Selects the default collation by URI.
    #[must_use]
    pub fn with_default_collation(mut self, uri: impl Into<String>) -> Self {
        self.default_collation = uri.into();
        self
    }

    /// Stops treating a reference to an undeclared variable as the static error `XPST0008`,
    /// deferring the check to evaluation.
    ///
    /// The specification requires the static check, and it is what this context does by
    /// default. A host language that discovers its variables only at run time — an XSLT stylesheet
    /// module, a Schematron phase — can turn it off and bind the values on the
    /// [`DynamicContext`] instead; an unbound variable then raises `XPST0008` when it is
    /// evaluated rather than when it is compiled.
    #[must_use]
    pub const fn with_undeclared_variables(mut self) -> Self {
        self.undeclared_variables = true;
        self
    }

    /// Whether references to undeclared variables compile.
    #[must_use]
    pub const fn allows_undeclared_variables(&self) -> bool {
        self.undeclared_variables
    }

    /// The URI bound to `prefix`, if any.
    #[must_use]
    pub fn namespace(&self, prefix: &str) -> Option<&str> {
        self.namespaces.get(prefix).map(String::as_str)
    }

    /// Every in-scope prefix and its URI.
    pub fn namespaces(&self) -> impl Iterator<Item = (&str, &str)> {
        self.namespaces
            .iter()
            .map(|(prefix, uri)| (prefix.as_str(), uri.as_str()))
    }

    /// The default namespace for unprefixed element and type names.
    #[must_use]
    pub fn default_element_namespace(&self) -> Option<&str> {
        self.default_element_namespace.as_deref()
    }

    /// The default namespace for unprefixed function names.
    #[must_use]
    pub fn default_function_namespace(&self) -> &str {
        &self.default_function_namespace
    }

    /// Whether `name` is a declared variable.
    #[must_use]
    pub fn has_variable(&self, name: &ExpandedName) -> bool {
        self.variables.contains(name)
    }

    /// Every declared variable name.
    pub fn variables(&self) -> impl Iterator<Item = &ExpandedName> {
        self.variables.iter()
    }

    /// The function library.
    #[must_use]
    pub fn functions(&self) -> &FunctionRegistry {
        &self.functions
    }

    /// The static base URI.
    #[must_use]
    pub fn base_uri(&self) -> Option<&str> {
        self.base_uri.as_deref()
    }

    /// The default collation's URI.
    #[must_use]
    pub fn default_collation(&self) -> &str {
        &self.default_collation
    }

    /// Resolves a lexical `QName` in the *element and type* name space: an unprefixed name takes
    /// the default element namespace.
    ///
    /// # Errors
    ///
    /// [`XPathErrorCode::Xpst0081`] when the prefix is not bound.
    pub fn resolve_element_name(&self, lexical: &str) -> XPathResult<ExpandedName> {
        self.resolve(lexical, self.default_element_namespace.as_deref())
    }

    /// Resolves a lexical `QName` in the *attribute and variable* name space: an unprefixed name
    /// is in no namespace.
    ///
    /// # Errors
    ///
    /// [`XPathErrorCode::Xpst0081`] when the prefix is not bound.
    pub fn resolve_unqualified_name(&self, lexical: &str) -> XPathResult<ExpandedName> {
        self.resolve(lexical, None)
    }

    /// Resolves a lexical `QName` in the *function* name space: an unprefixed name takes the
    /// default function namespace.
    ///
    /// # Errors
    ///
    /// [`XPathErrorCode::Xpst0081`] when the prefix is not bound.
    pub fn resolve_function_name(&self, lexical: &str) -> XPathResult<ExpandedName> {
        self.resolve(lexical, Some(&self.default_function_namespace))
    }

    fn resolve(&self, lexical: &str, default: Option<&str>) -> XPathResult<ExpandedName> {
        match lexical.split_once(':') {
            None => Ok(ExpandedName::new(default, lexical)),
            Some((prefix, local)) => match self.namespaces.get(prefix) {
                Some(uri) => Ok(ExpandedName::new(Some(uri.as_str()), local)),
                None => Err(XPathError::new(
                    XPathErrorCode::Xpst0081,
                    format!("the namespace prefix `{prefix}` is not bound"),
                )),
            },
        }
    }
}

impl fmt::Debug for StaticContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StaticContext")
            .field("namespaces", &self.namespaces)
            .field("default_element_namespace", &self.default_element_namespace)
            .field(
                "default_function_namespace",
                &self.default_function_namespace,
            )
            .field("variables", &self.variables)
            .field("functions", &self.functions.len())
            .field("base_uri", &self.base_uri)
            .field("default_collation", &self.default_collation)
            .finish()
    }
}

/// The context item, position and size that `.`, `position()` and `last()` read.
#[derive(Debug, Clone)]
pub struct Focus {
    item: XpathItem,
    position: usize,
    size: usize,
}

impl Focus {
    /// A focus of one item: position and size are both 1.
    pub fn singleton(item: impl Into<XpathItem>) -> Self {
        Self {
            item: item.into(),
            position: 1,
            size: 1,
        }
    }

    /// A focus at `position` (1-based) within a sequence of `size` items.
    pub fn new(item: impl Into<XpathItem>, position: usize, size: usize) -> Self {
        Self {
            item: item.into(),
            position,
            size,
        }
    }

    /// The context item.
    #[must_use]
    pub const fn item(&self) -> &XpathItem {
        &self.item
    }

    /// The context position, `position()`.
    #[must_use]
    pub const fn position(&self) -> usize {
        self.position
    }

    /// The context size, `last()`.
    #[must_use]
    pub const fn size(&self) -> usize {
        self.size
    }
}

/// How the evaluator reached the registered function that is currently running.
///
/// [XPath 3.1 §3.2](https://www.w3.org/TR/xpath-31/#id-function-calls) distinguishes a *static*
/// function call, which names the function at the call site, from a *dynamic* function call,
/// which evaluates an expression to a function item and invokes that. F&O's own functions cannot
/// tell the difference and have no reason to; a **host language** may have to, because it defines
/// functions whose meaning comes from a part of the dynamic context that a function item does not
/// capture.
///
/// XSLT 3.0 is the case this exists for. `fn:current-group`, `fn:current-grouping-key` and
/// `fn:current` read the current group, the current grouping key and the outermost focus — none
/// of which is part of the XPath dynamic context a function item closes over — so
/// [§14.2](https://www.w3.org/TR/xslt-30/#grouping) and
/// [§18.2](https://www.w3.org/TR/xslt-30/#function-current) make a dynamic call to any of them a
/// dynamic error (`XTDE1061`, `XTDE1071`, `XTDE1360`) rather than letting it read whichever group
/// happened to be current when the item was finally applied. Reading this is how such a function
/// recognizes the case it must reject.
/// XPath 3.1 defines exactly these two ways of calling a function, so the enumeration is closed
/// and a host may match it exhaustively.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InvocationKind {
    /// A *static function call*: the function's name is written at the call site, as in
    /// `fn:key('k', 'v')`.
    Static,
    /// A *dynamic function call*: the function was reached through a function item — a named
    /// function reference (`fn:current-group#0`), a partial application, `fn:function-lookup`,
    /// `fn:apply`, or any higher-order function that invokes its function argument.
    Dynamic,
}

/// One registered-function call in progress.
struct CallSite {
    focus: Option<Focus>,
    kind: InvocationKind,
}

thread_local! {
    /// The registered-function calls currently running, innermost last. See [`calling_focus`]
    /// and [`calling_invocation`].
    static CALL_SITES: std::cell::RefCell<Vec<CallSite>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Runs `body` with `focus` and `kind` published as the current call site, restoring the previous
/// one after.
pub(crate) fn with_call_site<R>(
    focus: Option<&Focus>,
    kind: InvocationKind,
    body: impl FnOnce() -> R,
) -> R {
    CALL_SITES.with(|stack| {
        stack.borrow_mut().push(CallSite {
            focus: focus.cloned(),
            kind,
        });
    });
    let result = body();
    CALL_SITES.with(|stack| {
        stack.borrow_mut().pop();
    });
    result
}

/// The focus at the call site of the registered function currently running, if any.
///
/// A function registered through [`oxixml_fn::FunctionRegistry`] is handed only an
/// [`oxixml_fn::FnContext`], which deliberately carries no focus: F&O's own functions take
/// everything they need as arguments. A *host language* may define functions that do not — XSLT
/// 3.0's `fn:key` and `fn:id` use the context node to decide which document to search
/// ([§18.3](https://www.w3.org/TR/xslt-30/#key-function)) — and a path expression rebinds that
/// item at every step, so `$e/key('k','v')` must see `$e` and not whatever the host recorded
/// before the expression began.
///
/// This reports the focus the evaluator held when it entered the current registered function.
/// It is `None` outside such a call, and `Some(None)` collapses to `None` — a call made with an
/// absent focus is indistinguishable from no call at all, which is what a caller wanting a
/// context item should conclude either way. [`calling_invocation`] deliberately does *not*
/// collapse that way: a function of arity zero called with no focus is still in a call, and its
/// host still has to know how it was reached.
#[must_use]
pub fn calling_focus() -> Option<Focus> {
    CALL_SITES.with(|stack| stack.borrow().last().and_then(|site| site.focus.clone()))
}

/// How the registered function currently running was invoked, or `None` outside such a call.
///
/// The engine publishes this around every call it makes to a function
/// [`FunctionRegistry`](oxixml_fn::FunctionRegistry) serves, so a host-language function can
/// recognize that it was reached through a function item rather than by a call that names it —
/// see [`InvocationKind`]. Unlike [`calling_focus`] it is reported for every call in progress,
/// including one made with no focus at all.
///
/// # Examples
///
/// ```
/// use oxixml_fn::{FN_NAMESPACE, FunctionRegistry};
/// use oxixml_infoset::{AtomicValue, Sequence};
/// use oxixml_xpath::{
///     DynamicContext, InvocationKind, StaticContext, XPath, XPathVersion, calling_invocation,
/// };
/// use std::sync::Arc;
///
/// // A host function that reports how it was called.
/// let mut registry = FunctionRegistry::new();
/// registry.add(FN_NAMESPACE, "how-was-i-called", 0, |_context, _arguments| {
///     let how = match calling_invocation() {
///         Some(InvocationKind::Dynamic) => "dynamic",
///         Some(InvocationKind::Static) => "static",
///         None => "not in a call",
///     };
///     Ok(Sequence::from(AtomicValue::string(how)))
/// });
/// let static_context = Arc::new(StaticContext::new().with_functions(registry));
/// let context = DynamicContext::new(Arc::clone(&static_context));
///
/// let evaluate = |expression: &str| -> Result<String, oxixml_xpath::XPathError> {
///     XPath::compile(
///         expression,
///         XPathVersion::XPath31,
///         StaticContext::clone(&static_context),
///     )?
///     .evaluate_to_string_in(&context)
/// };
/// assert_eq!(evaluate("fn:how-was-i-called()")?, "static");
/// assert_eq!(evaluate("fn:how-was-i-called#0()")?, "dynamic");
/// # Ok::<(), oxixml_xpath::XPathError>(())
/// ```
#[must_use]
pub fn calling_invocation() -> Option<InvocationKind> {
    CALL_SITES.with(|stack| stack.borrow().last().map(|site| site.kind))
}

/// Whether the registered function currently running was reached through a function item.
///
/// The shorthand for `matches!(calling_invocation(), Some(InvocationKind::Dynamic))`, which is
/// what a host function raising the dynamic error XSLT 3.0 defines for a dynamic call needs. It
/// is `false` outside a registered-function call, where there is nothing to reject.
#[must_use]
pub fn invoked_through_function_item() -> bool {
    matches!(calling_invocation(), Some(InvocationKind::Dynamic))
}

/// A variable binding, chained so that entering a scope is O(1).
#[derive(Debug)]
struct Binding {
    name: ExpandedName,
    value: XpathValue,
    parent: Option<Arc<Binding>>,
}

/// Everything evaluation needs that compilation did not.
///
/// # Examples
///
/// ```
/// use oxixml_infoset::{AtomicValue, Tree};
/// use oxixml_xpath::{DynamicContext, StaticContext, XPath, XPathVersion};
/// use std::sync::Arc;
///
/// let static_context = Arc::new(StaticContext::new().with_variable("wanted"));
/// let compiled = XPath::compile(
///     "count(//item[@id = $wanted])",
///     XPathVersion::XPath31,
///     StaticContext::clone(&static_context),
/// )?;
///
/// let tree = Tree::parse_str(r#"<r><item id="a"/><item id="b"/></r>"#)?;
/// let context = DynamicContext::new(static_context)
///     .with_context_item(tree.root())
///     .with_variable("wanted", AtomicValue::string("b"));
/// assert_eq!(compiled.evaluate_to_string_in(&context)?, "1");
/// # Ok::<(), oxixml_xpath::XPathError>(())
/// ```
#[derive(Debug, Clone)]
pub struct DynamicContext {
    static_context: Arc<StaticContext>,
    fn_context: Arc<FnContext>,
    focus: Option<Arc<Focus>>,
    bindings: Option<Arc<Binding>>,
    resolver: Option<Arc<dyn DocumentResolver>>,
    version: XPathVersion,
}

impl DynamicContext {
    /// A context over `static_context`, with no focus, no variable bindings and no resolver.
    ///
    /// The [`FnContext`] is derived from the static context: the static base URI and the default
    /// collation carry over, and the current dateTime is sampled once, so that every
    /// `fn:current-dateTime` in one evaluation reports the same instant.
    #[must_use]
    pub fn new(static_context: Arc<StaticContext>) -> Self {
        let mut fn_context = FnContext::new();
        if let Some(base) = static_context.base_uri() {
            fn_context = fn_context.with_static_base_uri(base);
        }
        if let Ok(updated) = fn_context
            .clone()
            .with_default_collation(static_context.default_collation())
        {
            fn_context = updated;
        }
        Self {
            static_context,
            fn_context: Arc::new(fn_context),
            focus: None,
            bindings: None,
            resolver: None,
            version: XPathVersion::default(),
        }
    }

    /// Selects the semantics this evaluation follows.
    ///
    /// [`CompiledXPath::evaluate_in`](crate::CompiledXPath::evaluate_in) sets this from the
    /// version the expression was compiled with, so an application never has to.
    #[must_use]
    pub const fn with_version(mut self, version: XPathVersion) -> Self {
        self.version = version;
        self
    }

    /// The semantics this evaluation follows.
    #[must_use]
    pub const fn version(&self) -> XPathVersion {
        self.version
    }

    /// Sets the context item, with position and size both 1.
    #[must_use]
    pub fn with_context_item(mut self, item: impl Into<XpathItem>) -> Self {
        self.focus = Some(Arc::new(Focus::singleton(item)));
        self
    }

    /// Sets the whole focus.
    #[must_use]
    pub fn with_focus(mut self, focus: Focus) -> Self {
        self.focus = Some(Arc::new(focus));
        self
    }

    /// Removes the focus, so that `.`, `position()` and `last()` raise `XPDY0002`.
    #[must_use]
    pub fn without_focus(mut self) -> Self {
        self.focus = None;
        self
    }

    /// Binds a variable in no namespace.
    #[must_use]
    pub fn with_variable(self, name: impl Into<String>, value: impl Into<XpathValue>) -> Self {
        let name = ExpandedName::new(None::<String>, name.into());
        self.with_variable_name(name, value)
    }

    /// Binds a variable by its expanded name.
    #[must_use]
    pub fn with_variable_name(mut self, name: ExpandedName, value: impl Into<XpathValue>) -> Self {
        self.bindings = Some(Arc::new(Binding {
            name,
            value: value.into(),
            parent: self.bindings.take(),
        }));
        self
    }

    /// Replaces the [`FnContext`] the function library evaluates against.
    ///
    /// The static base URI and the default collation are components of the *static* context,
    /// so a supplied function context that states neither inherits the compiled expression's —
    /// otherwise handing the evaluator a function context in order to redirect `fn:trace` would
    /// silently make `fn:static-base-uri()` empty, leave every relative URI unresolvable, and
    /// drop a declared default collation back to code points, so that `fn:compare($a, $b)` and
    /// every other collation-sensitive function would answer against the wrong ordering.
    ///
    /// "States neither" is read off the values themselves, because a [`FnContext`] has no way
    /// to spell an *absent* default collation: a supplied context still naming the mandatory
    /// code point collation — what [`FnContext::new`] leaves behind — is taken to be stating no
    /// preference, and a context naming anything else keeps its own choice. Restating the
    /// static context's collation on the supplied context before handing it over is therefore
    /// always safe, and is a no-op rather than a conflict.
    #[must_use]
    pub fn with_fn_context(mut self, fn_context: FnContext) -> Self {
        let mut fn_context = fn_context;
        if fn_context.static_base_uri().is_none() {
            if let Some(base) = self.static_context.base_uri() {
                fn_context = fn_context.with_static_base_uri(base);
            }
        }
        if fn_context.default_collation().uri() == CODEPOINT_COLLATION
            && let Ok(updated) = fn_context
                .clone()
                .with_default_collation(self.static_context.default_collation())
        {
            fn_context = updated;
        }
        self.fn_context = Arc::new(fn_context);
        self
    }

    /// Pins `fn:current-dateTime`, which makes an evaluation reproducible.
    #[must_use]
    pub fn with_current_date_time(mut self, current: DateTime) -> Self {
        self.fn_context =
            Arc::new(FnContext::clone(&self.fn_context).with_current_date_time(current));
        self
    }

    /// Sets the implicit timezone.
    #[must_use]
    pub fn with_implicit_timezone(mut self, timezone: TimezoneOffset) -> Self {
        self.fn_context =
            Arc::new(FnContext::clone(&self.fn_context).with_implicit_timezone(timezone));
        self
    }

    /// Registers a collation, so that it can be named in `fn:compare` and friends.
    #[must_use]
    pub fn with_collation(mut self, collation: Arc<dyn Collation>) -> Self {
        self.fn_context = Arc::new(FnContext::clone(&self.fn_context).with_collation(collation));
        self
    }

    /// Installs the resolver `fn:doc`, `fn:collection` and `fn:unparsed-text` consult.
    #[must_use]
    pub fn with_document_resolver(mut self, resolver: Arc<dyn DocumentResolver>) -> Self {
        self.resolver = Some(resolver);
        self
    }

    /// The static context this evaluation compiles against.
    #[must_use]
    pub fn static_context(&self) -> &Arc<StaticContext> {
        &self.static_context
    }

    /// Replaces the static context, keeping the focus, the bindings and the resolver.
    ///
    /// A host language whose expressions do not all share one static context needs it. XQuery is
    /// the case in point: a namespace declaration attribute extends the statically known
    /// namespaces "for the whole of the constructor's content"
    /// ([XQuery 3.1 §3.9.1](https://www.w3.org/TR/xquery-31/#id-computed-namespace-constructors)),
    /// so each enclosed expression is compiled against a context of its own, and every *runtime*
    /// use of the statically known namespaces — resolving the prefix of a lexical `xs:QName`,
    /// above all — has to see that context rather than the module's.
    #[must_use]
    pub fn with_static_context(mut self, static_context: Arc<StaticContext>) -> Self {
        self.static_context = static_context;
        self
    }

    /// The function library's context.
    #[must_use]
    pub fn fn_context(&self) -> &FnContext {
        &self.fn_context
    }

    /// The focus, or `None` when the context item is absent.
    #[must_use]
    pub fn focus(&self) -> Option<&Focus> {
        self.focus.as_deref()
    }

    /// The document resolver, if one is installed.
    #[must_use]
    pub fn document_resolver(&self) -> Option<&Arc<dyn DocumentResolver>> {
        self.resolver.as_ref()
    }

    /// The context item.
    ///
    /// # Errors
    ///
    /// [`XPathErrorCode::Xpdy0002`] when the context item is absent.
    pub fn context_item(&self) -> XPathResult<&XpathItem> {
        match &self.focus {
            Some(focus) => Ok(focus.item()),
            None => Err(XPathError::new(
                XPathErrorCode::Xpdy0002,
                "the context item is absent",
            )),
        }
    }

    /// The context node.
    ///
    /// # Errors
    ///
    /// [`XPathErrorCode::Xpdy0002`] when the context item is absent, or
    /// [`XPathErrorCode::Xpty0020`] when it is not a node.
    pub fn context_node(&self) -> XPathResult<&NodeRef> {
        let item = self.context_item()?;
        item.as_node().ok_or_else(|| {
            XPathError::new(
                XPathErrorCode::Xpty0020,
                format!("the context item is {}, not a node", item.describe()),
            )
        })
    }

    /// The context node, for a *function* whose argument defaults to the context item.
    ///
    /// `XPTY0020` is specific to an axis step (XPath 3.1 §3.3.1). When `fn:name()` or
    /// `fn:base-uri()` supplies the context item as an argument declared `node()?`, the failure
    /// is the ordinary argument type error `XPTY0004` instead.
    ///
    /// # Errors
    ///
    /// [`XPathErrorCode::Xpdy0002`] when the context item is absent, or
    /// [`XPathErrorCode::Xpty0004`] when it is not a node.
    pub fn context_node_argument(&self) -> XPathResult<&NodeRef> {
        let item = self.context_item()?;
        item.as_node().ok_or_else(|| {
            XPathError::type_error(format!(
                "the context item is {}, not the node this function requires",
                item.describe()
            ))
        })
    }

    /// `position()`.
    ///
    /// # Errors
    ///
    /// [`XPathErrorCode::Xpdy0002`] when the context item is absent.
    pub fn position(&self) -> XPathResult<usize> {
        match &self.focus {
            Some(focus) => Ok(focus.position()),
            None => Err(XPathError::new(
                XPathErrorCode::Xpdy0002,
                "`position()` needs a context item, and none is defined",
            )),
        }
    }

    /// `last()`.
    ///
    /// # Errors
    ///
    /// [`XPathErrorCode::Xpdy0002`] when the context item is absent.
    pub fn size(&self) -> XPathResult<usize> {
        match &self.focus {
            Some(focus) => Ok(focus.size()),
            None => Err(XPathError::new(
                XPathErrorCode::Xpdy0002,
                "`last()` needs a context item, and none is defined",
            )),
        }
    }

    /// The value bound to `name`, if any.
    #[must_use]
    pub fn variable(&self, name: &ExpandedName) -> Option<&XpathValue> {
        let mut current = self.bindings.as_deref();
        while let Some(binding) = current {
            if &binding.name == name {
                return Some(&binding.value);
            }
            current = binding.parent.as_deref();
        }
        None
    }

    /// The value bound to `name`.
    ///
    /// # Errors
    ///
    /// [`XPathErrorCode::Xpst0008`] when the variable is not bound.
    pub fn require_variable(&self, name: &ExpandedName) -> XPathResult<&XpathValue> {
        self.variable(name).ok_or_else(|| {
            XPathError::new(
                XPathErrorCode::Xpst0008,
                format!("the variable `${name}` is not bound"),
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxixml_infoset::AtomicValue;

    #[test]
    fn the_default_namespaces_are_in_scope() {
        let context = StaticContext::new();
        assert_eq!(context.namespace("xs"), Some(XSD_NAMESPACE));
        assert_eq!(context.namespace("fn"), Some(FN_NAMESPACE));
        assert_eq!(context.namespace("math"), Some(MATH_NAMESPACE));
        assert_eq!(context.namespace("nope"), None);
        assert!(context.namespaces().count() >= 8);
    }

    #[test]
    fn names_resolve_per_name_space() -> XPathResult<()> {
        let context = StaticContext::new()
            .with_default_element_namespace("urn:e")
            .with_namespace("p", "urn:p");
        assert_eq!(
            context.resolve_element_name("a")?,
            ExpandedName::new(Some("urn:e"), "a")
        );
        assert_eq!(
            context.resolve_unqualified_name("a")?,
            ExpandedName::new(None::<String>, "a")
        );
        assert_eq!(
            context.resolve_function_name("a")?,
            ExpandedName::new(Some(FN_NAMESPACE), "a")
        );
        assert_eq!(
            context.resolve_element_name("p:a")?,
            ExpandedName::new(Some("urn:p"), "a")
        );
        assert!(context.resolve_element_name("q:a").is_err());
        Ok(())
    }

    #[test]
    fn variables_shadow_in_the_binding_chain() {
        let context = DynamicContext::new(Arc::new(StaticContext::new()))
            .with_variable("x", AtomicValue::integer(1_i64))
            .with_variable("x", AtomicValue::integer(2_i64));
        let name = ExpandedName::new(None::<String>, "x");
        assert_eq!(
            context.variable(&name).map(ToString::to_string),
            Some("2".to_owned())
        );
        let missing = ExpandedName::new(None::<String>, "y");
        assert!(context.require_variable(&missing).is_err());
    }

    #[test]
    fn an_absent_focus_raises_xpdy0002() {
        let context = DynamicContext::new(Arc::new(StaticContext::new()));
        assert_eq!(
            context.context_item().expect_err("absent").code(),
            Some(XPathErrorCode::Xpdy0002)
        );
        assert!(context.position().is_err());
        assert!(context.size().is_err());

        let with_item = context.with_context_item(AtomicValue::integer(3_i64));
        assert_eq!(with_item.position().expect("a focus"), 1);
        assert_eq!(with_item.size().expect("a focus"), 1);
        assert_eq!(
            with_item.context_node().expect_err("not a node").code(),
            Some(XPathErrorCode::Xpty0020)
        );
    }

    /// A static context whose library holds two host functions that report how they were
    /// invoked: one of arity zero, one of arity one (which a partial application needs).
    fn reporting_invocation() -> Arc<StaticContext> {
        fn how() -> oxixml_infoset::Sequence {
            let how = match calling_invocation() {
                Some(InvocationKind::Dynamic) => "dynamic",
                Some(InvocationKind::Static) => "static",
                None => "outside",
            };
            oxixml_infoset::Sequence::from(AtomicValue::string(how))
        }

        let mut registry = builtins();
        registry.add(FN_NAMESPACE, "how-was-i-called", 0, |_, _| Ok(how()));
        registry.add(FN_NAMESPACE, "how-was-i-called", 1, |_, _| Ok(how()));
        Arc::new(StaticContext::new().with_functions(registry))
    }

    fn evaluate_with(static_context: &Arc<StaticContext>, expression: &str) -> String {
        let compiled = crate::XPath::compile(
            expression,
            XPathVersion::XPath31,
            StaticContext::clone(static_context),
        )
        .expect("the expression compiles");
        let context = DynamicContext::new(Arc::clone(static_context));
        compiled
            .evaluate_to_string_in(&context)
            .expect("the expression evaluates")
    }

    #[test]
    fn a_host_function_sees_how_it_was_invoked() {
        let static_context = reporting_invocation();
        let evaluate = |expression: &str| evaluate_with(&static_context, expression);

        // A call that names the function is static, however deeply it is nested.
        assert_eq!(evaluate("fn:how-was-i-called()"), "static");
        assert_eq!(evaluate("fn:how-was-i-called(1)"), "static");
        assert_eq!(
            evaluate("count((1 to 3)[fn:how-was-i-called() = 'static'])"),
            "3"
        );

        // Every shape that reaches the function through a function item is dynamic: a named
        // function reference, a partial application, `fn:function-lookup`, `fn:apply`, and a
        // higher-order function applying the item it was handed.
        assert_eq!(evaluate("fn:how-was-i-called#0()"), "dynamic");
        assert_eq!(evaluate("fn:how-was-i-called(?)(1)"), "dynamic");
        assert_eq!(
            evaluate(
                "fn:function-lookup(fn:QName('http://www.w3.org/2005/xpath-functions', \
                 'how-was-i-called'), 0)()"
            ),
            "dynamic"
        );
        assert_eq!(evaluate("fn:apply(fn:how-was-i-called#0, [])"), "dynamic");
        assert_eq!(evaluate("fn:for-each(1, fn:how-was-i-called#1)"), "dynamic");

        // An inline function is a function item, but the call *inside* its body names the
        // function, so what the callee sees is the innermost call and not the wrapper.
        assert_eq!(
            evaluate("(function() { fn:how-was-i-called() })()"),
            "static"
        );
    }

    #[test]
    fn the_invocation_kind_is_reported_with_no_focus_and_only_inside_a_call() {
        // `calling_focus` collapses an absent focus to `None`; `calling_invocation` must not,
        // or an arity-zero function called with no focus would look like no call at all.
        assert_eq!(calling_invocation(), None);
        assert!(!invoked_through_function_item());
        with_call_site(None, InvocationKind::Dynamic, || {
            assert!(calling_focus().is_none());
            assert_eq!(calling_invocation(), Some(InvocationKind::Dynamic));
            assert!(invoked_through_function_item());
            // The innermost call is the one reported, and the stack unwinds to the outer one.
            with_call_site(None, InvocationKind::Static, || {
                assert_eq!(calling_invocation(), Some(InvocationKind::Static));
                assert!(!invoked_through_function_item());
            });
            assert_eq!(calling_invocation(), Some(InvocationKind::Dynamic));
        });
        assert_eq!(calling_invocation(), None);
        assert!(!invoked_through_function_item());
    }
}
