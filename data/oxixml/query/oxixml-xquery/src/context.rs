//! The static and dynamic contexts a query is compiled and evaluated against.
//!
//! [XQuery 3.1 §2.1](https://www.w3.org/TR/xquery-31/#eval_context) extends the XPath expression
//! context with the components the prolog can set: boundary-space policy, construction mode,
//! ordering mode, the default order for empty sequences, and the copy-namespaces mode.
//! [`StaticQueryContext`] carries those alongside everything
//! [`oxixml_xpath::StaticContext`] already carries, and every one of them can be overridden by a
//! declaration in the query's own prolog.
//!
//! [`DynamicQueryContext`] is the run-time half: the context item, the values of external
//! variables, the document resolver, and the [`FnContext`] the function library reads.

use oxixml_fn::numeric::DecimalFormat;
use oxixml_fn::{Collation, FnContext};
use oxixml_infoset::ExpandedName;
use oxixml_qname::XML_NAMESPACE;
use oxixml_xpath::{
    ARRAY_NAMESPACE, DocumentResolver, ERR_NAMESPACE, FN_NAMESPACE, MAP_NAMESPACE, MATH_NAMESPACE,
    XSD_NAMESPACE, XSI_NAMESPACE, XpathItem, XpathValue,
};
use oxixml_xsd::{DateTime, TimezoneOffset};
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use crate::module::ModuleResolver;

/// The namespace of the `local:` functions a main module may declare.
pub const LOCAL_NAMESPACE: &str = "http://www.w3.org/2005/xquery-local-functions";
/// The namespace of the `xqt-errors` variables a `catch` clause binds, `err:`.
pub const ERROR_NAMESPACE: &str = ERR_NAMESPACE;
/// The namespace reserved for the `output:` serialization parameters.
pub const OUTPUT_NAMESPACE: &str = "http://www.w3.org/2010/xslt-xquery-serialization";

/// Whether boundary whitespace in a direct element constructor is kept.
///
/// `declare boundary-space preserve | strip` selects it; the default is `strip`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BoundarySpace {
    /// Boundary whitespace is discarded — the default.
    #[default]
    Strip,
    /// Boundary whitespace becomes a text node.
    Preserve,
}

/// Whether constructed and copied elements keep their type annotations.
///
/// `declare construction preserve | strip` selects it; the default is `preserve`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConstructionMode {
    /// Copied nodes keep their type annotations — the default.
    #[default]
    Preserve,
    /// Copied elements are re-annotated as `xs:untyped`.
    Strip,
}

/// Whether path expressions and FLWOR results must come back in document order.
///
/// `declare ordering ordered | unordered` selects it; the default is `ordered`, and this engine
/// always produces ordered results, so `unordered` is only recorded, never acted on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OrderingMode {
    /// Results are in document order — the default, and what this engine always does.
    #[default]
    Ordered,
    /// The processor may return results in any order.
    Unordered,
}

/// Where the empty sequence sorts in an `order by` clause that does not say.
///
/// `declare default order empty greatest | least` selects it; the default is `least`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EmptyOrder {
    /// The empty sequence sorts before every value — the default.
    #[default]
    Least,
    /// The empty sequence sorts after every value.
    Greatest,
}

/// What happens to the namespaces of an element copied by a constructor.
///
/// `declare copy-namespaces preserve|no-preserve, inherit|no-inherit` selects it; the default is
/// `preserve, inherit`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CopyNamespaces {
    /// Whether the copied element keeps the namespace bindings it had.
    pub preserve: bool,
    /// Whether the copied element gains the constructor's in-scope namespaces.
    pub inherit: bool,
}

impl Default for CopyNamespaces {
    fn default() -> Self {
        Self {
            preserve: true,
            inherit: true,
        }
    }
}

/// Everything the compiler needs before it sees a document.
///
/// # Examples
///
/// ```
/// use oxixml_xquery::{BoundarySpace, StaticQueryContext, XQuery};
///
/// let context = StaticQueryContext::new()
///     .with_namespace("bk", "urn:example:book")
///     .with_boundary_space(BoundarySpace::Preserve);
///
/// let compiled = XQuery::compile("<bk:t> {1 + 1} </bk:t>", context)?;
/// assert_eq!(compiled.execute_to_string(&Default::default())?, " 2 ");
/// # Ok::<(), oxixml_xquery::XQueryError>(())
/// ```
#[derive(Clone)]
pub struct StaticQueryContext {
    pub(crate) namespaces: BTreeMap<String, String>,
    pub(crate) default_element_namespace: Option<String>,
    pub(crate) default_function_namespace: String,
    pub(crate) base_uri: Option<String>,
    pub(crate) default_collation: String,
    pub(crate) boundary_space: BoundarySpace,
    pub(crate) construction: ConstructionMode,
    pub(crate) ordering: OrderingMode,
    pub(crate) empty_order: EmptyOrder,
    pub(crate) copy_namespaces: CopyNamespaces,
    pub(crate) external_variables: Vec<ExpandedName>,
    pub(crate) module_resolver: Option<Arc<dyn ModuleResolver>>,
    /// Collations the application installed, which `declare default collation` and the
    /// collation-sensitive functions may name.
    pub(crate) collations: Vec<Arc<dyn Collation>>,
}

impl Default for StaticQueryContext {
    fn default() -> Self {
        Self::new()
    }
}

impl StaticQueryContext {
    /// A context with the specification's default in-scope namespaces (`xml`, `xs`, `xsi`, `fn`,
    /// `math`, `map`, `array`, `err`, `local`), the `fn:` default function namespace, the code
    /// point collation, and every prolog-settable component at its default.
    #[must_use]
    pub fn new() -> Self {
        let mut namespaces = BTreeMap::new();
        for (prefix, uri) in [
            ("xml", XML_NAMESPACE),
            ("xs", XSD_NAMESPACE),
            ("xsi", XSI_NAMESPACE),
            ("fn", FN_NAMESPACE),
            ("math", MATH_NAMESPACE),
            ("map", MAP_NAMESPACE),
            ("array", ARRAY_NAMESPACE),
            ("err", ERR_NAMESPACE),
            ("local", LOCAL_NAMESPACE),
            ("output", OUTPUT_NAMESPACE),
        ] {
            namespaces.insert(prefix.to_owned(), uri.to_owned());
        }
        Self {
            namespaces,
            default_element_namespace: None,
            default_function_namespace: FN_NAMESPACE.to_owned(),
            base_uri: None,
            default_collation: oxixml_fn::CODEPOINT_COLLATION.to_owned(),
            boundary_space: BoundarySpace::default(),
            construction: ConstructionMode::default(),
            ordering: OrderingMode::default(),
            empty_order: EmptyOrder::default(),
            copy_namespaces: CopyNamespaces::default(),
            external_variables: Vec::new(),
            module_resolver: None,
            collations: Vec::new(),
        }
    }

    /// Binds a namespace prefix.
    #[must_use]
    pub fn with_namespace(mut self, prefix: impl Into<String>, uri: impl Into<String>) -> Self {
        self.namespaces.insert(prefix.into(), uri.into());
        self
    }

    /// Binds several namespace prefixes.
    #[must_use]
    pub fn with_namespaces<K, V>(mut self, bindings: impl IntoIterator<Item = (K, V)>) -> Self
    where
        K: Into<String>,
        V: Into<String>,
    {
        for (prefix, uri) in bindings {
            self.namespaces.insert(prefix.into(), uri.into());
        }
        self
    }

    /// Sets the default namespace for unprefixed element and type names.
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

    /// Sets the static base URI.
    #[must_use]
    pub fn with_base_uri(mut self, uri: impl Into<String>) -> Self {
        self.base_uri = Some(uri.into());
        self
    }

    /// Registers a collation the query may name.
    ///
    /// A collation must be known to the *static* context as well as the dynamic one, because
    /// `declare default collation` is checked when the query is compiled: XQuery 3.1 §4.4 makes
    /// naming a collation the processor does not support the static error `XQST0038`. The
    /// collation is also installed for evaluation, so registering it here is enough.
    #[must_use]
    pub fn with_collation(mut self, collation: Arc<dyn Collation>) -> Self {
        self.collations.push(collation);
        self
    }

    /// Whether `uri` names a collation this context knows, built in or installed.
    pub(crate) fn knows_collation(&self, uri: &str) -> bool {
        self.collations
            .iter()
            .any(|collation| collation.uri() == uri)
            || oxixml_fn::Collations::new().resolve(uri).is_ok()
    }

    /// Selects the default collation by URI.
    #[must_use]
    pub fn with_default_collation(mut self, uri: impl Into<String>) -> Self {
        self.default_collation = uri.into();
        self
    }

    /// Selects the boundary-space policy.
    #[must_use]
    pub const fn with_boundary_space(mut self, policy: BoundarySpace) -> Self {
        self.boundary_space = policy;
        self
    }

    /// Selects the construction mode.
    #[must_use]
    pub const fn with_construction_mode(mut self, mode: ConstructionMode) -> Self {
        self.construction = mode;
        self
    }

    /// Selects the ordering mode.
    #[must_use]
    pub const fn with_ordering_mode(mut self, mode: OrderingMode) -> Self {
        self.ordering = mode;
        self
    }

    /// Selects the default order for empty sequences in `order by`.
    #[must_use]
    pub const fn with_empty_order(mut self, order: EmptyOrder) -> Self {
        self.empty_order = order;
        self
    }

    /// Selects the copy-namespaces mode.
    #[must_use]
    pub const fn with_copy_namespaces(mut self, mode: CopyNamespaces) -> Self {
        self.copy_namespaces = mode;
        self
    }

    /// Declares an external variable in no namespace.
    #[must_use]
    pub fn with_variable(mut self, name: impl Into<String>) -> Self {
        self.external_variables
            .push(ExpandedName::new(None::<String>, name.into()));
        self
    }

    /// Declares an external variable by its expanded name.
    #[must_use]
    pub fn with_variable_name(mut self, name: ExpandedName) -> Self {
        self.external_variables.push(name);
        self
    }

    /// Installs the callback `import module` consults.
    ///
    /// Nothing can be imported until one is installed: a Pure-Rust library does not read the
    /// filesystem on its caller's behalf.
    #[must_use]
    pub fn with_module_resolver(mut self, resolver: Arc<dyn ModuleResolver>) -> Self {
        self.module_resolver = Some(resolver);
        self
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

    /// The boundary-space policy.
    #[must_use]
    pub const fn boundary_space(&self) -> BoundarySpace {
        self.boundary_space
    }

    /// The construction mode.
    #[must_use]
    pub const fn construction_mode(&self) -> ConstructionMode {
        self.construction
    }

    /// The ordering mode.
    #[must_use]
    pub const fn ordering_mode(&self) -> OrderingMode {
        self.ordering
    }

    /// The default order for empty sequences.
    #[must_use]
    pub const fn empty_order(&self) -> EmptyOrder {
        self.empty_order
    }

    /// The copy-namespaces mode.
    #[must_use]
    pub const fn copy_namespaces(&self) -> CopyNamespaces {
        self.copy_namespaces
    }

    /// The module resolver, if one is installed.
    #[must_use]
    pub fn module_resolver(&self) -> Option<&Arc<dyn ModuleResolver>> {
        self.module_resolver.as_ref()
    }
}

impl fmt::Debug for StaticQueryContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StaticQueryContext")
            .field("namespaces", &self.namespaces)
            .field("default_element_namespace", &self.default_element_namespace)
            .field(
                "default_function_namespace",
                &self.default_function_namespace,
            )
            .field("base_uri", &self.base_uri)
            .field("default_collation", &self.default_collation)
            .field("boundary_space", &self.boundary_space)
            .field("construction", &self.construction)
            .field("ordering", &self.ordering)
            .field("empty_order", &self.empty_order)
            .field("copy_namespaces", &self.copy_namespaces)
            .field("external_variables", &self.external_variables)
            .finish_non_exhaustive()
    }
}

/// Everything evaluation needs that compilation did not.
///
/// # Examples
///
/// ```
/// use oxixml_infoset::{AtomicValue, Tree};
/// use oxixml_xquery::{DynamicQueryContext, StaticQueryContext, XQuery};
///
/// let compiled = XQuery::compile(
///     "for $b in //book where $b/@year > $after return string($b/title)",
///     StaticQueryContext::new().with_variable("after"),
/// )?;
///
/// let tree = Tree::parse_str(
///     "<lib><book year='1965'><title>Dune</title></book>\
///      <book year='1815'><title>Emma</title></book></lib>",
/// )?;
/// let context = DynamicQueryContext::new()
///     .with_context_item(tree.root())
///     .with_variable("after", AtomicValue::integer(1900_i64));
/// assert_eq!(compiled.execute(&context)?.len(), 1);
/// # Ok::<(), oxixml_xquery::XQueryError>(())
/// ```
#[derive(Debug, Clone, Default)]
pub struct DynamicQueryContext {
    pub(crate) context_item: Option<XpathItem>,
    pub(crate) variables: Vec<(ExpandedName, XpathValue)>,
    pub(crate) resolver: Option<Arc<dyn DocumentResolver>>,
    pub(crate) fn_context: Option<FnContext>,
    pub(crate) current_date_time: Option<DateTime>,
    pub(crate) implicit_timezone: Option<TimezoneOffset>,
    pub(crate) collations: Vec<Arc<dyn Collation>>,
    pub(crate) decimal_formats: Vec<(Option<ExpandedName>, DecimalFormat)>,
    pub(crate) stack_size: Option<usize>,
}

impl DynamicQueryContext {
    /// A context with no focus, no variable bindings and no resolver.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the context item.
    #[must_use]
    pub fn with_context_item(mut self, item: impl Into<XpathItem>) -> Self {
        self.context_item = Some(item.into());
        self
    }

    /// Binds an external variable in no namespace.
    #[must_use]
    pub fn with_variable(self, name: impl Into<String>, value: impl Into<XpathValue>) -> Self {
        let name = ExpandedName::new(None::<String>, name.into());
        self.with_variable_name(name, value)
    }

    /// Binds an external variable by its expanded name.
    #[must_use]
    pub fn with_variable_name(mut self, name: ExpandedName, value: impl Into<XpathValue>) -> Self {
        self.variables.push((name, value.into()));
        self
    }

    /// Installs the resolver `fn:doc`, `fn:collection` and `fn:unparsed-text` consult.
    #[must_use]
    pub fn with_document_resolver(mut self, resolver: Arc<dyn DocumentResolver>) -> Self {
        self.resolver = Some(resolver);
        self
    }

    /// Declares the stack of the thread the query will be executed on.
    ///
    /// A call to a declared function recurses on the native stack, and running out of it aborts
    /// the whole process — an outcome no caller and no `try`/`catch` can intercept. The
    /// evaluator therefore measures the stack it has consumed and raises `XPDY0130` while there
    /// is still room, which it can only do against a figure the caller supplies: nothing in the
    /// standard library reports how much stack a thread was given.
    ///
    /// The default is [`DEFAULT_STACK_SIZE`](crate::DEFAULT_STACK_SIZE), the 2 MiB
    /// `std::thread::spawn` hands a new thread. A caller on a process's main thread, or on a
    /// thread spawned with room for deep recursion on purpose, says so here.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_xquery::DynamicQueryContext;
    ///
    /// let context = DynamicQueryContext::new().with_stack_size(64 * 1024 * 1024);
    /// assert_eq!(context.stack_size(), Some(64 * 1024 * 1024));
    /// ```
    #[must_use]
    pub const fn with_stack_size(mut self, bytes: usize) -> Self {
        self.stack_size = Some(bytes);
        self
    }

    /// The stack the caller declared, if any.
    #[must_use]
    pub const fn stack_size(&self) -> Option<usize> {
        self.stack_size
    }

    /// Replaces the [`FnContext`] the function library evaluates against.
    #[must_use]
    pub fn with_fn_context(mut self, fn_context: FnContext) -> Self {
        self.fn_context = Some(fn_context);
        self
    }

    /// Pins `fn:current-dateTime`, which makes an evaluation reproducible.
    #[must_use]
    pub fn with_current_date_time(mut self, current: DateTime) -> Self {
        self.current_date_time = Some(current);
        self
    }

    /// Sets the implicit timezone.
    #[must_use]
    pub fn with_implicit_timezone(mut self, timezone: TimezoneOffset) -> Self {
        self.implicit_timezone = Some(timezone);
        self
    }

    /// Registers a collation, so that it can be named in `order by … collation` and in the
    /// collation-sensitive functions.
    #[must_use]
    pub fn with_collation(mut self, collation: Arc<dyn Collation>) -> Self {
        self.collations.push(collation);
        self
    }

    /// Registers a decimal format; `None` replaces the unnamed default format.
    #[must_use]
    pub fn with_decimal_format(
        mut self,
        name: Option<ExpandedName>,
        format: DecimalFormat,
    ) -> Self {
        self.decimal_formats.push((name, format));
        self
    }

    /// The context item, if one is set.
    #[must_use]
    pub const fn context_item(&self) -> Option<&XpathItem> {
        self.context_item.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_namespaces_are_in_scope() {
        let context = StaticQueryContext::new();
        assert_eq!(context.namespace("xs"), Some(XSD_NAMESPACE));
        assert_eq!(context.namespace("local"), Some(LOCAL_NAMESPACE));
        assert_eq!(context.namespace("nope"), None);
        assert!(context.namespaces().count() >= 10);
    }

    #[test]
    fn the_prolog_settable_components_start_at_their_defaults() {
        let context = StaticQueryContext::new();
        assert_eq!(context.boundary_space(), BoundarySpace::Strip);
        assert_eq!(context.construction_mode(), ConstructionMode::Preserve);
        assert_eq!(context.ordering_mode(), OrderingMode::Ordered);
        assert_eq!(context.empty_order(), EmptyOrder::Least);
        assert_eq!(
            context.copy_namespaces(),
            CopyNamespaces {
                preserve: true,
                inherit: true
            }
        );
    }

    #[test]
    fn the_builders_are_chainable() {
        let context = StaticQueryContext::new()
            .with_namespace("p", "urn:p")
            .with_default_element_namespace("urn:e")
            .with_base_uri("urn:base")
            .with_boundary_space(BoundarySpace::Preserve)
            .with_empty_order(EmptyOrder::Greatest);
        assert_eq!(context.namespace("p"), Some("urn:p"));
        assert_eq!(context.default_element_namespace(), Some("urn:e"));
        assert_eq!(context.base_uri(), Some("urn:base"));
        assert_eq!(context.boundary_space(), BoundarySpace::Preserve);
        assert_eq!(context.empty_order(), EmptyOrder::Greatest);
        assert!(format!("{context:?}").contains("StaticQueryContext"));
    }

    #[test]
    fn a_dynamic_context_collects_its_bindings() {
        let context = DynamicQueryContext::new()
            .with_variable("x", oxixml_infoset::AtomicValue::integer(1_i64));
        assert_eq!(context.variables.len(), 1);
        assert!(context.context_item().is_none());
    }
}
