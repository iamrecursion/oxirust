//! [`StaticEnv`] — the compile-time environment one stylesheet element sees.
//!
//! Every XPath expression in a stylesheet is compiled against the namespaces in scope *at the
//! element it is written on*, the `xpath-default-namespace` in force there, the stylesheet's
//! version, and the default collation. [XSLT 3.0 §5.4](https://www.w3.org/TR/xslt-30/#static-context)
//! calls that the *static context of the XPath expression*, and this type is it.

use crate::avt;
use crate::error::{XsltError, XsltResult};
use crate::run::functions;
use oxixml_infoset::ExpandedName;
use oxixml_qname::XML_NAMESPACE;
use oxixml_xpath::{CompiledXPath, StaticContext, XPath, XPathVersion};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;
use std::sync::Arc;

/// The XSLT version a stylesheet declares.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum XsltVersion {
    /// `version="1.0"`.
    #[default]
    V10,
    /// `version="2.0"`.
    V20,
    /// `version="3.0"` — and anything higher, which is processed in forwards-compatible mode.
    V30,
}

impl XsltVersion {
    /// Parses the `version` attribute, defaulting to 1.0 for anything unrecognized below 2.0
    /// and to 3.0 for anything above.
    #[must_use]
    pub fn parse(value: &str) -> Self {
        let number: f64 = value.trim().parse().unwrap_or(1.0);
        if number >= 3.0 {
            Self::V30
        } else if number >= 2.0 {
            Self::V20
        } else {
            Self::V10
        }
    }

    /// The XPath language level this XSLT version implies.
    ///
    /// [XSLT 3.0 §3.9](https://www.w3.org/TR/xslt-30/#forwards) reserves grammar-restricting
    /// *backwards-compatible behavior* for `version="1.0"` alone — XPath 2.0, 3.0 and 3.1 "share
    /// the XDM semantics and differ only in which grammar productions are accepted"
    /// ([`XPathVersion`] top-level docs), so a module that declares `version="2.0"` or
    /// `version="3.0"` is compiled at this processor's full implemented level rather than capped
    /// to the language the declared number happens to name; only `version="1.0"` switches to the
    /// genuinely different XPath 1.0 value model.
    #[must_use]
    pub const fn xpath_version(self) -> XPathVersion {
        match self {
            Self::V10 => XPathVersion::XPath10,
            Self::V20 | Self::V30 => XPathVersion::XPath31,
        }
    }

    /// Whether XSLT 2.0 and later constructs are available.
    #[must_use]
    pub const fn is_at_least_20(self) -> bool {
        !matches!(self, Self::V10)
    }

    /// The value `system-property('xsl:version')` reports.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::V10 => "1.0",
            Self::V20 => "2.0",
            Self::V30 => "3.0",
        }
    }
}

/// One part of a compiled attribute value template.
#[derive(Debug)]
pub(crate) enum AvtPart {
    /// Literal text.
    Literal(String),
    /// An embedded expression.
    Expression(CompiledXPath),
}

/// A compiled attribute value template.
#[derive(Debug)]
pub(crate) struct Avt {
    parts: Vec<AvtPart>,
    constant: Option<String>,
}

impl Avt {
    /// A template that is just literal text.
    pub(crate) fn literal(text: impl Into<String>) -> Self {
        let text = text.into();
        Self {
            parts: vec![AvtPart::Literal(text.clone())],
            constant: Some(text),
        }
    }

    /// The parts, for evaluation.
    pub(crate) fn parts(&self) -> &[AvtPart] {
        &self.parts
    }

    /// The value, when the template has no expressions at all.
    pub(crate) fn constant(&self) -> Option<&str> {
        self.constant.as_deref()
    }
}

/// The static context one stylesheet element compiles against.
///
/// The XPath static context is built lazily and cached: an environment that is derived and
/// then discarded — which happens for every element of a stylesheet — never pays for one.
#[derive(Debug)]
pub struct StaticEnv {
    namespaces: BTreeMap<String, String>,
    default_element_namespace: Option<String>,
    default_collation: Option<String>,
    version: XsltVersion,
    base_uri: Option<String>,
    /// The mode `default-mode` puts in force, `None` for the unnamed mode.
    default_mode: Option<ExpandedName>,
    /// Whether `expand-text` makes text nodes *text value templates*.
    expand_text: bool,
    /// Whether the stated version is later than the one this processor implements, which puts
    /// forwards-compatible processing in force.
    forwards_compatible: bool,
    /// The `xsl:function` declarations the stylesheet makes, discovered before anything is
    /// compiled so that a call can be resolved wherever it is written.
    user_functions: Arc<Vec<crate::compile::FunctionSignature>>,
    /// The static variables and parameters of the whole stylesheet, shared by every environment
    /// derived from the one the compilation started with.
    ///
    /// A static variable is a property of the *package*, not of the element it happens to be
    /// written on ([XSLT 3.0 §9.6](https://www.w3.org/TR/xslt-30/#static-params)), and it is
    /// bound as the module tree is expanded — long before most environments that need to read it
    /// exist. Sharing one cell rather than copying the table into each derived environment is
    /// what lets a shadow attribute deep inside a template see a static parameter declared at
    /// the top of an imported module.
    statics: Rc<RefCell<super::statics::StaticBindings>>,
    context: RefCell<Option<Arc<StaticContext>>>,
}

impl Clone for StaticEnv {
    fn clone(&self) -> Self {
        Self {
            namespaces: self.namespaces.clone(),
            default_element_namespace: self.default_element_namespace.clone(),
            default_collation: self.default_collation.clone(),
            version: self.version,
            base_uri: self.base_uri.clone(),
            default_mode: self.default_mode.clone(),
            expand_text: self.expand_text,
            forwards_compatible: self.forwards_compatible,
            user_functions: Arc::clone(&self.user_functions),
            statics: Rc::clone(&self.statics),
            context: RefCell::new(self.context.borrow().clone()),
        }
    }
}

impl StaticEnv {
    /// The environment of a stylesheet with no namespace bindings beyond the built-in ones.
    #[must_use]
    pub fn new(version: XsltVersion) -> Self {
        Self {
            namespaces: BTreeMap::new(),
            default_element_namespace: None,
            default_collation: None,
            version,
            base_uri: None,
            default_mode: None,
            expand_text: false,
            forwards_compatible: false,
            user_functions: Arc::new(Vec::new()),
            statics: Rc::new(RefCell::new(super::statics::StaticBindings::default())),
            context: RefCell::new(None),
        }
    }

    /// A fresh environment for a separately loaded stylesheet module.
    ///
    /// An imported or included module starts from its own namespaces, base URI and defaults;
    /// nothing of the referring module's environment carries over except the table of static
    /// variables, which [XSLT 3.0 §9.6](https://www.w3.org/TR/xslt-30/#static-params) makes a
    /// property of the *package* rather than of one module — a static parameter declared in the
    /// principal module is in scope in everything it imports afterwards, and a declaration in an
    /// imported module has to be visible to the importing one for `XTSE3450` to be detectable at
    /// all.
    pub(crate) fn for_module(&self, version: XsltVersion) -> Self {
        let mut env = Self::new(version);
        env.statics = Rc::clone(&self.statics);
        env
    }

    /// The static variables and parameters the whole stylesheet declares.
    ///
    /// Every environment derived from one shares the table, so a binding made while a module is
    /// expanded is visible from every element compiled afterwards.
    pub(crate) fn statics(&self) -> &Rc<RefCell<super::statics::StaticBindings>> {
        &self.statics
    }

    /// Binds every static variable into `context`, so that an expression evaluated at compile
    /// time can read them.
    pub(crate) fn bind_statics(
        &self,
        mut context: oxixml_xpath::DynamicContext,
    ) -> oxixml_xpath::DynamicContext {
        for name in self.statics.borrow().names() {
            let value = self.statics.borrow().value(&name);
            if let Some(value) = value {
                context = context.with_variable_name(name, value);
            }
        }
        context
    }

    /// The value of `name` on `element`, with a shadow attribute — `_name="{$STATIC}"` — taking
    /// the place of the attribute it names.
    ///
    /// # Errors
    ///
    /// Every static error expanding the shadow attribute raises.
    pub(crate) fn shadowed_attribute(
        &self,
        element: &oxixml_infoset::NodeRef,
        name: &str,
    ) -> XsltResult<Option<String>> {
        super::statics::shadowed_attribute(element, name, self)
    }

    /// Publishes the stylesheet's `xsl:function` declarations, so that calls to them compile.
    pub(crate) fn set_user_functions(
        &mut self,
        functions: Arc<Vec<crate::compile::FunctionSignature>>,
    ) {
        self.user_functions = functions;
        self.invalidate();
    }

    /// The XSLT version in force.
    #[must_use]
    pub const fn version(&self) -> XsltVersion {
        self.version
    }

    /// The static base URI in force.
    #[must_use]
    pub fn base_uri(&self) -> Option<&str> {
        self.base_uri.as_deref()
    }

    /// The default collation in force.
    #[must_use]
    pub fn default_collation(&self) -> Option<&str> {
        self.default_collation.as_deref()
    }

    /// The mode `default-mode` puts in force — `None` for the unnamed mode.
    ///
    /// [XSLT 3.0 §6.6.1](https://www.w3.org/TR/xslt-30/#default-mode) makes the attribute
    /// apply to every `xsl:template` and `xsl:apply-templates` in its scope, including the
    /// element that carries it.
    #[must_use]
    pub(crate) fn default_mode(&self) -> Option<&ExpandedName> {
        self.default_mode.as_ref()
    }

    /// Sets the mode `default-mode` puts in force.
    pub(crate) fn set_default_mode(&mut self, mode: Option<ExpandedName>) {
        self.default_mode = mode;
    }

    /// Whether `expand-text` is in force, making every text node a *text value template* —
    /// [XSLT 3.0 §5.7](https://www.w3.org/TR/xslt-30/#text-value-templates).
    #[must_use]
    pub(crate) fn expands_text(&self) -> bool {
        self.expand_text
    }

    /// Sets whether `expand-text` is in force.
    pub(crate) fn set_expand_text(&mut self, expand: bool) {
        self.expand_text = expand;
    }

    /// Every namespace binding in scope, as `(prefix, uri)` pairs. The empty prefix is the
    /// default namespace.
    pub(crate) fn namespaces(&self) -> impl Iterator<Item = (&str, &str)> {
        self.namespaces
            .iter()
            .map(|(prefix, uri)| (prefix.as_str(), uri.as_str()))
    }

    /// Replaces the version, which `xsl:stylesheet/@version` and a literal result element's
    /// `xsl:version` both do.
    pub(crate) fn set_version(&mut self, version: XsltVersion) {
        if self.version != version {
            self.version = version;
            self.invalidate();
        }
    }

    /// Records the version exactly as it was written, so that
    /// [XSLT 3.0 §3.9](https://www.w3.org/TR/xslt-30/#forwards) *forwards-compatible
    /// processing* can be told apart from plain 3.0. `XsltVersion` deliberately folds every
    /// version at or above 3.0 into one value, which is right for language level and wrong for
    /// this question.
    pub(crate) fn set_version_text(&mut self, text: &str) {
        self.set_version(XsltVersion::parse(text));
        self.forwards_compatible = text.trim().parse::<f64>().is_ok_and(|stated| stated > 3.0);
    }

    /// Whether this element is read in forwards-compatible mode, where an XSLT element the
    /// processor does not recognize is ignored rather than rejected.
    #[must_use]
    pub(crate) const fn forwards_compatible(&self) -> bool {
        self.forwards_compatible
    }

    /// Adds namespace bindings. An empty URI undeclares the prefix.
    pub(crate) fn declare_namespaces<'a>(
        &mut self,
        bindings: impl IntoIterator<Item = (Option<&'a str>, &'a str)>,
    ) {
        let mut changed = false;
        for (prefix, uri) in bindings {
            let key = prefix.unwrap_or("").to_owned();
            if uri.is_empty() {
                changed |= self.namespaces.remove(&key).is_some();
            } else {
                changed |= self.namespaces.insert(key, uri.to_owned()).as_deref() != Some(uri);
            }
        }
        if changed {
            self.invalidate();
        }
    }

    /// Drops every binding whose prefix is not among `in_scope`.
    ///
    /// A namespace *undeclaration* is invisible in a list of declarations — it is the absence of
    /// a binding, not a binding to nothing — so an environment built by accumulating
    /// declarations has to be told which prefixes survived. `in_scope` is the element's whole
    /// namespace axis, which is exactly that list.
    pub(crate) fn undeclare_namespaces_absent_from<'a>(
        &mut self,
        in_scope: impl IntoIterator<Item = Option<&'a str>>,
    ) {
        let kept: Vec<String> = in_scope
            .into_iter()
            .map(|prefix| prefix.unwrap_or("").to_owned())
            .collect();
        let dropped: Vec<String> = self
            .namespaces
            .keys()
            .filter(|prefix| !kept.contains(prefix))
            .cloned()
            .collect();
        if dropped.is_empty() {
            return;
        }
        for prefix in dropped {
            self.namespaces.remove(&prefix);
        }
        self.invalidate();
    }

    /// Sets `xpath-default-namespace`.
    pub(crate) fn set_default_element_namespace(&mut self, uri: Option<String>) {
        if self.default_element_namespace != uri {
            self.default_element_namespace = uri;
            self.invalidate();
        }
    }

    /// Sets `default-collation`.
    pub(crate) fn set_default_collation(&mut self, uri: Option<String>) {
        if self.default_collation != uri {
            self.default_collation = uri;
            self.invalidate();
        }
    }

    /// Sets the static base URI, which `xml:base` and the module's own location supply.
    pub(crate) fn set_base_uri(&mut self, uri: Option<String>) {
        if self.base_uri != uri {
            self.base_uri = uri;
            self.invalidate();
        }
    }

    fn invalidate(&mut self) {
        *self.context.borrow_mut() = None;
    }

    fn build(&self) -> Arc<StaticContext> {
        self.build_context(None)
    }

    /// Builds the static context, either with every variable reference allowed (`None`) or with
    /// exactly `declared` in scope, so that a reference to anything else is `XPST0008`.
    fn build_context(&self, declared: Option<&[ExpandedName]>) -> Arc<StaticContext> {
        self.build_context_with(declared, false)
    }

    /// Builds the static context, optionally restricted to what a *static expression* may see.
    fn build_context_with(
        &self,
        declared: Option<&[ExpandedName]>,
        restricted: bool,
    ) -> Arc<StaticContext> {
        // XSLT 1.0 §12 defines `format-number` and `generate-id` itself, independently of
        // whichever XPath version the stylesheet's declared version compiles as — see
        // `StaticContext::with_xslt_functions`. Every XSLT version needs this: a `version="1.0"`
        // module compiles its expressions as XPath 1.0, which predates both functions in F&O.
        // A static expression sees neither library: §9.7 gives it F&O plus the four availability
        // functions and nothing else, `format-number` and `generate-id` included only in as far
        // as F&O itself defines them.
        let mut context = if restricted {
            StaticContext::new().with_functions(functions::static_registry().clone())
        } else {
            // An `xsl:function` is registered as an opaque closure, so a *reference* to one —
            // `local:f#2` — would be judged by its arity alone unless the declared types travel
            // with it. XPath 3.1 §3.6.1 compares parameter and result types too, and
            // `higher-order-functions-032` … `-034` measure exactly that.
            let signatures = Arc::clone(&self.user_functions);
            StaticContext::new()
                .with_functions(functions::registry_with(&self.user_functions))
                .with_xslt_functions()
                .with_host_signatures(Arc::new(move |name, arity| {
                    signatures
                        .iter()
                        .find(|signature| signature.name == *name && signature.arity == arity)
                        .and_then(|signature| signature.declared.clone())
                }))
        };
        context = match declared {
            None => context.with_undeclared_variables(),
            Some(names) => names.iter().fold(context, |context, name| {
                context.with_variable_name(name.clone())
            }),
        };
        for (prefix, uri) in &self.namespaces {
            // Binding the empty prefix would set the XPath *default element namespace*, which in
            // a stylesheet is `[xsl:]xpath-default-namespace` and not this at all. The default
            // namespace declaration still has to reach the runtime, because
            // [XSLT 3.0 §5.1](https://www.w3.org/TR/xslt-30/#unprefixed-qnames) expands the
            // argument of `element-available` with it, so it travels under its own key.
            if prefix.is_empty() {
                context = context.with_namespace(crate::names::DEFAULT_NAMESPACE_KEY, uri.clone());
                continue;
            }
            context = context.with_namespace(prefix.clone(), uri.clone());
        }
        if let Some(uri) = &self.default_element_namespace {
            context = context.with_default_element_namespace(uri.clone());
        }
        if let Some(collation) = &self.default_collation {
            context = context.with_default_collation(collation.clone());
        }
        if let Some(base) = &self.base_uri {
            context = context.with_base_uri(base.clone());
        }
        Arc::new(context)
    }

    /// The XPath static context this environment produces.
    #[must_use]
    pub fn xpath_context(&self) -> Arc<StaticContext> {
        if let Some(context) = self.context.borrow().as_ref() {
            return Arc::clone(context);
        }
        let built = self.build();
        *self.context.borrow_mut() = Some(Arc::clone(&built));
        built
    }

    /// Re-compiles an expression with exactly `declared` in scope instead of the usual
    /// "anything goes" variable scope, so that the XPath parser itself decides — range
    /// variables, shadowing and all — whether the expression refers to a name that is not in
    /// scope.
    ///
    /// The engine normally compiles with [`StaticContext::with_undeclared_variables`], because
    /// a stylesheet may refer to a global that is declared later in the module or in another
    /// one; this is the one place that needs the strict reading.
    ///
    /// # Errors
    ///
    /// `XPST0008` when the expression refers to a variable outside `declared`, plus whatever
    /// else the expression fails to compile with.
    pub(crate) fn compile_expression_in_scope(
        &self,
        text: &str,
        declared: &[ExpandedName],
    ) -> XsltResult<CompiledXPath> {
        XPath::compile(
            text,
            self.version.xpath_version(),
            StaticContext::clone(&self.build_context(Some(declared))),
        )
        .map_err(|error| XsltError::from(error).as_static())
    }

    /// Compiles a *static expression* — the value of a `[xsl:]use-when` attribute, or the
    /// `select` of a `static="yes"` variable — against the restricted context
    /// [XSLT 3.0 §9.7](https://www.w3.org/TR/xslt-30/#static-expressions) gives it.
    ///
    /// Three of that section's constraints are visible here, and each of them is what a whole
    /// group of the suite's cases asks about. The in-scope variables are the *static* variables
    /// declared before this element and nothing else, so a reference to an ordinary stylesheet
    /// variable is `XPST0008`. The function library is F&O plus the four availability functions,
    /// so a call to a stylesheet function — or to `current()`, or to `key()` — is `XPST0017`.
    /// And the context item is absent, so a path expression that needs one is `XPDY0002` when
    /// the expression is evaluated.
    ///
    /// # Errors
    ///
    /// `XPST0003`, `XPST0008`, `XPST0017` and friends, re-labelled as static XSLT errors.
    pub(crate) fn compile_static_expression(&self, text: &str) -> XsltResult<CompiledXPath> {
        let declared = self.statics.borrow().names();
        XPath::compile(
            text,
            self.version.xpath_version(),
            StaticContext::clone(&self.build_context_with(Some(&declared), true)),
        )
        .map_err(|error| {
            XsltError::from(error)
                .contextualize(&format!("in the XPath expression `{}`", text.trim()))
                .as_static()
        })
    }

    /// Compiles an XPath expression.
    ///
    /// # Errors
    ///
    /// `XPST0003` and friends, re-labelled as static XSLT errors.
    pub(crate) fn compile_expression(&self, text: &str) -> XsltResult<CompiledXPath> {
        XPath::compile(
            text,
            self.version.xpath_version(),
            StaticContext::clone(&self.xpath_context()),
        )
        .map_err(|error| {
            XsltError::from(error)
                .contextualize(&format!("in the XPath expression `{}`", text.trim()))
                .as_static()
        })
    }

    /// Compiles an attribute value template.
    ///
    /// # Errors
    ///
    /// `XTSE0350`, `XTSE0370` or any static error an embedded expression raises.
    pub(crate) fn compile_avt(&self, text: &str) -> XsltResult<Avt> {
        let pieces = avt::parse(text)?;
        if avt::is_constant(&pieces) {
            return Ok(Avt::literal(avt::literal_text(&pieces)));
        }
        let mut parts = Vec::with_capacity(pieces.len());
        for piece in pieces {
            parts.push(match piece {
                avt::AvtPiece::Literal(text) => AvtPart::Literal(text),
                avt::AvtPiece::Expression(text) => {
                    AvtPart::Expression(self.compile_expression(&text)?)
                }
            });
        }
        Ok(Avt {
            parts,
            constant: None,
        })
    }

    /// Resolves a prefix to a namespace URI.
    ///
    /// # Errors
    ///
    /// `XTSE0280` when the prefix is not bound.
    pub(crate) fn resolve_prefix(&self, prefix: &str) -> XsltResult<String> {
        if prefix == "xml" {
            return Ok(XML_NAMESPACE.to_owned());
        }
        self.namespaces.get(prefix).cloned().ok_or_else(|| {
            XsltError::stat1c("XTSE0280", format!("the prefix `{prefix}` is not bound"))
        })
    }

    /// Resolves a lexical `QName` in the element and type namespace: an unprefixed name takes
    /// `xpath-default-namespace`.
    ///
    /// # Errors
    ///
    /// `XTSE0280` when the prefix is not bound.
    pub(crate) fn resolve_element_name(&self, lexical: &str) -> XsltResult<ExpandedName> {
        self.resolve(lexical, self.default_element_namespace.as_deref())
    }

    /// Resolves a lexical `QName` written on a serialization parameter, where an unprefixed
    /// name takes the default namespace declaration (`xmlns="…"`) in scope.
    ///
    /// # Errors
    ///
    /// `XTSE0280` when the prefix is not bound.
    pub(crate) fn resolve_output_name(&self, lexical: &str) -> XsltResult<ExpandedName> {
        let default = self
            .namespaces
            .get("")
            .map(String::as_str)
            .filter(|uri| !uri.is_empty());
        self.resolve(lexical, default)
    }

    /// Resolves a lexical `QName` in the attribute and variable namespace: an unprefixed name is
    /// in no namespace.
    ///
    /// # Errors
    ///
    /// `XTSE0280` when the prefix is not bound.
    pub(crate) fn resolve_attribute_name(&self, lexical: &str) -> XsltResult<ExpandedName> {
        // Where a name is declared — of a template, a variable, a key, a mode — a value that
        // is not even a lexical `QName` is outside the value space the attribute declares,
        // which XSLT 3.0 §3.5 makes `XTSE0020` rather than a namespace failure. The braced
        // `EQName` form is a name too, and passes through to be resolved as written.
        let trimmed = lexical.trim();
        if !trimmed.starts_with("Q{") && !oxixml_qname::is_qname(trimmed) {
            return Err(XsltError::stat1c(
                "XTSE0020",
                format!("`{trimmed}` is not a lexical QName"),
            ));
        }
        self.resolve(lexical, None)
    }

    fn resolve(&self, lexical: &str, default: Option<&str>) -> XsltResult<ExpandedName> {
        let lexical = lexical.trim();
        // The `EQName` production of XPath 3.0 admits `Q{uri}local`, which XSLT 3.0 §5.1.1
        // allows wherever a name is written. Its URI may contain colons, so it has to be split
        // on the brace before anything looks for a prefix.
        if let Some(rest) = lexical.strip_prefix("Q{")
            && let Some((namespace, local)) = rest.split_once('}')
        {
            let namespace = namespace.trim();
            return Ok(ExpandedName::new(
                (!namespace.is_empty()).then_some(namespace),
                local.trim(),
            ));
        }
        match lexical.split_once(':') {
            None => Ok(ExpandedName::new(default, lexical)),
            Some((prefix, local)) => {
                let namespace = self.resolve_prefix(prefix)?;
                Ok(ExpandedName::new(Some(namespace), local))
            }
        }
    }
}

/// Checks the value an `[xsl:]version` attribute states.
///
/// # Errors
///
/// `XTSE0110`, which [XSLT 3.0 §3.6](https://www.w3.org/TR/xslt-30/#version) makes a static
/// error for a value outside the lexical space of `xs:decimal`: the exponent in `error-0110a`'s
/// `2.0e3` makes it a number to `f64` but not to `xs:decimal`.
pub(crate) fn check_version_text(text: &str) -> XsltResult<()> {
    if is_decimal_literal(text.trim()) {
        return Ok(());
    }
    Err(XsltError::stat1c(
        "XTSE0110",
        format!("`{text}` is not a version number"),
    ))
}

/// Whether `text` is in the lexical space of `xs:decimal`: an optional sign, then digits with
/// at most one decimal point among them and at least one digit in all.
fn is_decimal_literal(text: &str) -> bool {
    let digits = text.strip_prefix(['+', '-']).unwrap_or(text);
    let (whole, fraction) = match digits.split_once('.') {
        Some((whole, fraction)) => (whole, fraction),
        None => (digits, ""),
    };
    !(whole.is_empty() && fraction.is_empty())
        && whole.chars().all(|digit| digit.is_ascii_digit())
        && fraction.chars().all(|digit| digit.is_ascii_digit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn versions_map_onto_xpath_levels() {
        assert_eq!(XsltVersion::parse("1.0"), XsltVersion::V10);
        assert_eq!(XsltVersion::parse("2.0"), XsltVersion::V20);
        assert_eq!(XsltVersion::parse("3.0"), XsltVersion::V30);
        assert_eq!(XsltVersion::parse("4.0"), XsltVersion::V30);
        assert_eq!(XsltVersion::parse("nonsense"), XsltVersion::V10);
        assert_eq!(XsltVersion::V10.xpath_version(), XPathVersion::XPath10);
        // XSLT 3.0 §3.9 reserves grammar-restricting backwards-compatible behavior for
        // version="1.0" alone; a 2.0-declared module compiles at the implemented level.
        assert_eq!(XsltVersion::V20.xpath_version(), XPathVersion::XPath31);
        assert_eq!(XsltVersion::V30.xpath_version(), XPathVersion::XPath31);
        assert!(XsltVersion::V20.is_at_least_20());
    }

    #[test]
    fn names_resolve_per_namespace() -> XsltResult<()> {
        let mut env = StaticEnv::new(XsltVersion::V20);
        env.declare_namespaces([(Some("p"), "urn:p"), (None, "urn:default")]);
        env.set_default_element_namespace(Some("urn:xpath".to_owned()));

        assert_eq!(
            env.resolve_element_name("a")?,
            ExpandedName::new(Some("urn:xpath"), "a")
        );
        assert_eq!(
            env.resolve_attribute_name("a")?,
            ExpandedName::new(None::<&str>, "a")
        );
        assert_eq!(
            env.resolve_element_name("p:a")?,
            ExpandedName::new(Some("urn:p"), "a")
        );
        assert!(env.resolve_element_name("q:a").is_err());
        Ok(())
    }

    #[test]
    fn expressions_and_templates_compile() -> XsltResult<()> {
        let env = StaticEnv::new(XsltVersion::V20);
        assert!(env.compile_expression("1 + 1").is_ok());
        assert!(env.compile_expression("1 +").is_err());

        let constant = env.compile_avt("plain")?;
        assert_eq!(constant.constant(), Some("plain"));
        let dynamic = env.compile_avt("a{1}")?;
        assert_eq!(dynamic.constant(), None);
        assert_eq!(dynamic.parts().len(), 2);
        Ok(())
    }
}
