//! The public entry points: [`Stylesheet`], [`CompiledStylesheet`] and [`Transformation`].

use crate::compile::{Compiler, ModeKey, StylesheetIr};
use crate::error::{XsltError, XsltResult};
use crate::output::OutputProperties;
use crate::resolve::Resolver;
use crate::run::{Context, Emitter, Transformer, with_runtime};
use crate::tree::ResultTree;
use oxixml_infoset::{ExpandedName, Item, NodeRef, ParseOptions, Sequence, Tree, XdmNode};
use oxixml_xpath::XpathValue;
use oxixml_xsd::TimezoneOffset;
use std::sync::Arc;

/// What a stylesheet needs to know before it sees a source document.
///
/// # Examples
///
/// ```
/// use oxixml_xslt::{MemoryResolver, StaticOptions};
/// use std::sync::Arc;
///
/// let options = StaticOptions::new()
///     .with_base_uri("http://example.com/style.xsl")
///     .with_resolver(Arc::new(MemoryResolver::new().with_document("urn:x", "<a/>")));
/// assert_eq!(options.base_uri(), Some("http://example.com/style.xsl"));
/// ```
#[derive(Debug, Clone, Default)]
pub struct StaticOptions {
    resolver: Option<Arc<dyn Resolver>>,
    base_uri: Option<String>,
    backwards_compatibility: Option<bool>,
    static_parameters: Vec<(ExpandedName, XpathValue)>,
}

impl StaticOptions {
    /// Options with no resolver and no base URI.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Installs the resolver `xsl:include`, `xsl:import` and `document()` consult.
    #[must_use]
    pub fn with_resolver(mut self, resolver: Arc<dyn Resolver>) -> Self {
        self.resolver = Some(resolver);
        self
    }

    /// Sets the static base URI of the principal stylesheet module.
    #[must_use]
    pub fn with_base_uri(mut self, uri: impl Into<String>) -> Self {
        self.base_uri = Some(uri.into());
        self
    }

    /// The base URI, if one was set.
    #[must_use]
    pub fn base_uri(&self) -> Option<&str> {
        self.base_uri.as_deref()
    }

    /// The resolver, if one was installed.
    #[must_use]
    pub fn resolver(&self) -> Option<&Arc<dyn Resolver>> {
        self.resolver.as_ref()
    }

    /// Whether a stylesheet element whose *effective version* is 1.0 may be evaluated.
    ///
    /// [XSLT 3.0 §3.9](https://www.w3.org/TR/xslt-30/#backwards) makes an element's effective
    /// version the nearest `[xsl:]version` attribute, and
    /// [§27.4](https://www.w3.org/TR/xslt-30/#conformance-compat) says a processor that does not
    /// claim the *XSLT 1.0 compatibility feature* "must raise a dynamic error if an instruction
    /// is evaluated whose effective version is 1.0" — `XTDE0160`. Whether a processor claims
    /// that feature is implementation-defined, so this option says which of the two conforming
    /// readings applies.
    ///
    /// The default is `true`, which is the useful reading rather than the strict one: this
    /// processor *accepts* a `version="1.0"` stylesheet — the commonest kind in the wild — and
    /// evaluates it, but it implements only the grammar half of XSLT 1.0 behaviour (XPath
    /// expressions compile at the 1.0 language level); the value-space differences of
    /// [§3.9.1](https://www.w3.org/TR/xslt-30/#backwards-1-0), such as reducing a multi-item
    /// sequence to its first item, are not implemented. Set it to `false` for the strict
    /// reading, where evaluating any such element is `XTDE0160` and nothing is silently
    /// processed under the wrong semantics.
    #[must_use]
    pub const fn with_backwards_compatibility(mut self, enabled: bool) -> Self {
        self.backwards_compatibility = Some(enabled);
        self
    }

    /// Whether elements with an effective version of 1.0 are accepted; `true` unless
    /// [`StaticOptions::with_backwards_compatibility`] said otherwise.
    #[must_use]
    pub const fn backwards_compatibility(&self) -> bool {
        match self.backwards_compatibility {
            Some(enabled) => enabled,
            None => true,
        }
    }

    /// Supplies the value of a *static parameter* in no namespace.
    ///
    /// [XSLT 3.0 §9.6](https://www.w3.org/TR/xslt-30/#static-params) makes a top-level
    /// `xsl:param` carrying `static="yes"` a parameter of the *compilation* rather than of the
    /// transformation: its value is known before any source document is, so it can decide what
    /// the stylesheet is — through `[xsl:]use-when`, which drops an element from the stylesheet
    /// altogether, and through the *shadow attributes* of
    /// [§3.14.2](https://www.w3.org/TR/xslt-30/#shadow-attributes), where `_href="lib{$V}.xsl"`
    /// computes the value of an attribute the element does not carry.
    ///
    /// A value supplied here overrides the declaration's own `select` default. A static
    /// parameter declared `required="yes"` for which nothing is supplied is `XTDE0050`.
    ///
    /// This is a *different* channel from [`Parameters::with_parameter`]: an ordinary stylesheet
    /// parameter is supplied per transformation, a static one per compilation, and supplying a
    /// static parameter through `Parameters` has no effect on the shadow attributes that read
    /// it — by then the stylesheet has already been compiled.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_infoset::{AtomicValue, Tree};
    /// use oxixml_xslt::{StaticOptions, Stylesheet};
    ///
    /// let stylesheet = Stylesheet::compile_str_with(
    ///     r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
    ///          <xsl:output omit-xml-declaration="yes"/>
    ///          <xsl:param name="pick" static="yes" required="yes"/>
    ///          <xsl:template match="/"><out><xsl:value-of _select="{$pick}"/></out></xsl:template>
    ///        </xsl:stylesheet>"#,
    ///     StaticOptions::new().with_static_parameter("pick", AtomicValue::string("2 + 3")),
    /// )?;
    /// let source = Tree::parse_str("<a/>")?;
    /// assert_eq!(stylesheet.transform_to_string(&source.root())?, "<out>5</out>");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn with_static_parameter(
        mut self,
        name: impl Into<String>,
        value: impl Into<XpathValue>,
    ) -> Self {
        self.static_parameters
            .push((ExpandedName::new(None::<String>, name.into()), value.into()));
        self
    }

    /// Supplies the value of a static parameter by expanded name.
    ///
    /// As [`StaticOptions::with_static_parameter`], for a parameter declared in a namespace.
    #[must_use]
    pub fn with_static_parameter_name(
        mut self,
        name: ExpandedName,
        value: impl Into<XpathValue>,
    ) -> Self {
        self.static_parameters.push((name, value.into()));
        self
    }

    /// The static parameters supplied so far, in the order they were given.
    pub fn static_parameters(&self) -> impl Iterator<Item = (&ExpandedName, &XpathValue)> {
        self.static_parameters
            .iter()
            .map(|(name, value)| (name, value))
    }
}

/// The compiler.
///
/// # Examples
///
/// ```
/// use oxixml_infoset::Tree;
/// use oxixml_xslt::Stylesheet;
///
/// let stylesheet = Stylesheet::compile_str(
///     r#"<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
///          <xsl:output omit-xml-declaration="yes"/>
///          <xsl:template match="/"><out><xsl:value-of select="/a/@n"/></out></xsl:template>
///        </xsl:stylesheet>"#,
/// )?;
/// let source = Tree::parse_str(r#"<a n="7"/>"#)?;
/// assert_eq!(stylesheet.transform_to_string(&source.root())?, "<out>7</out>");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct Stylesheet;

/// Parses the text of a stylesheet module, remembering where each element is written.
///
/// The positions are what an `xsl:catch` body reads back as `$err:line-number` and
/// `$err:column-number` ([XSLT 3.0 §8.3](https://www.w3.org/TR/xslt-30/#element-catch)); a
/// stylesheet compiled from a tree the caller parsed itself carries them only if that caller
/// asked for them, and the error variables are an empty sequence when it did not, exactly as the
/// specification allows.
///
/// # Errors
///
/// `XTSE0165` when the text is not well-formed XML.
fn parse_module(xml: &str, base_uri: Option<&str>) -> XsltResult<Tree> {
    let mut parse_options = ParseOptions::default().record_source_positions(true);
    if let Some(uri) = base_uri {
        parse_options = parse_options.document_uri(uri);
    }
    Tree::parse_str_with(xml, parse_options)
        .map_err(|error| XsltError::stat1c("XTSE0165", error.message().to_owned()))
}

impl Stylesheet {
    /// Compiles the stylesheet rooted at `root`.
    ///
    /// # Errors
    ///
    /// Every static error the stylesheet raises, starting with `XTSE0165` when `root` is not a
    /// stylesheet at all.
    pub fn compile(root: &NodeRef) -> XsltResult<CompiledStylesheet> {
        Self::compile_with(root, StaticOptions::new())
    }

    /// Compiles the stylesheet rooted at `root` under `options`.
    ///
    /// # Errors
    ///
    /// As [`Stylesheet::compile`].
    pub fn compile_with(root: &NodeRef, options: StaticOptions) -> XsltResult<CompiledStylesheet> {
        let base_uri = options
            .base_uri
            .clone()
            .or_else(|| root.document_uri())
            .or_else(|| root.base_uri());
        let mut compiler = Compiler::new(options.resolver.clone());
        compiler.backwards_compatibility = options.backwards_compatibility();
        compiler.static_parameters = options
            .static_parameters
            .iter()
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect();
        compiler.compile(root, base_uri.clone())?;
        let static_parameters = std::mem::take(&mut compiler.bound_statics);
        Ok(CompiledStylesheet {
            ir: Arc::new(compiler.ir),
            resolver: options.resolver,
            base_uri,
            static_parameters,
        })
    }

    /// Parses `xml` and compiles it.
    ///
    /// # Errors
    ///
    /// `XTSE0165` when the text is not well-formed XML, plus every static error the stylesheet
    /// raises.
    pub fn compile_str(xml: &str) -> XsltResult<CompiledStylesheet> {
        let tree = parse_module(xml, None)?;
        Self::compile(&tree.root())
    }

    /// Parses `xml` and compiles it under `options`.
    ///
    /// # Errors
    ///
    /// As [`Stylesheet::compile_str`].
    pub fn compile_str_with(xml: &str, options: StaticOptions) -> XsltResult<CompiledStylesheet> {
        let tree = parse_module(xml, options.base_uri())?;
        Self::compile_with(&tree.root(), options)
    }
}

/// How a transformation is started and what it is given.
///
/// # Examples
///
/// ```
/// use oxixml_infoset::AtomicValue;
/// use oxixml_xslt::Parameters;
///
/// let parameters = Parameters::new()
///     .with_parameter("who", AtomicValue::string("world"))
///     .with_initial_mode("m");
/// assert_eq!(parameters.initial_mode(), Some("m"));
/// ```
#[derive(Debug, Clone, Default)]
pub struct Parameters {
    values: Vec<(ExpandedName, XpathValue)>,
    initial_template_parameters: Vec<(ExpandedName, XpathValue, bool)>,
    initial_mode_parameters: Vec<(ExpandedName, XpathValue, bool)>,
    initial_match_selection: Option<XpathValue>,
    initial_mode: Option<String>,
    initial_template: Option<String>,
    base_output_uri: Option<String>,
    /// A wall-clock deadline for the whole transformation — see [`Self::with_deadline`].
    deadline: Option<std::time::Instant>,
    /// Set when the caller wants the result as a sequence and will never read the text, so
    /// that a result no output method can write does not report a serialization error it could
    /// not otherwise raise (`fn:transform` with `delivery-format: 'raw'`).
    skip_serialization: bool,
    initial_function: Option<String>,
    initial_function_arguments: Vec<XpathValue>,
    step_limit: Option<u64>,
    recursion_limit: Option<usize>,
    stack_size: Option<usize>,
    implicit_timezone: Option<TimezoneOffset>,
}

impl Parameters {
    /// No parameters, the unnamed mode, and the ordinary `apply-templates` entry point.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Supplies a stylesheet parameter in no namespace.
    #[must_use]
    pub fn with_parameter(mut self, name: impl Into<String>, value: impl Into<XpathValue>) -> Self {
        self.values
            .push((ExpandedName::new(None::<String>, name.into()), value.into()));
        self
    }

    /// Supplies a stylesheet parameter by expanded name.
    #[must_use]
    pub fn with_parameter_name(mut self, name: ExpandedName, value: impl Into<XpathValue>) -> Self {
        self.values.push((name, value.into()));
        self
    }

    /// Starts in a named mode rather than the unnamed one.
    #[must_use]
    pub fn with_initial_mode(mut self, name: impl Into<String>) -> Self {
        self.initial_mode = Some(name.into());
        self
    }

    /// Supplies the *initial match selection*: the sequence `xsl:apply-templates` is applied to
    /// when the transformation starts.
    ///
    /// [XSLT 3.0 §2.3.3](https://www.w3.org/TR/xslt-30/#invoking-initial-mode) starts an
    /// *apply-templates invocation* by applying the template rules of the initial mode to a
    /// supplied sequence, which "may contain any items, including atomic values" and need not be
    /// a document at all. Left unset, the initial match selection is the global context item —
    /// the source document, when one is supplied — which is the usual shape.
    ///
    /// Supplying one is what lets the global context item and the nodes being processed differ:
    /// the global variables still see the source document, while the template rules run over
    /// this sequence, one item at a time, with `position()` and `last()` reflecting it.
    ///
    /// An initial template or an initial function selects a different entry point, and the
    /// initial match selection is then unused.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_infoset::{AtomicValue, Sequence};
    /// use oxixml_xslt::Parameters;
    ///
    /// let parameters = Parameters::new().with_initial_match_selection(Sequence::from_items([
    ///     AtomicValue::integer(1).into(),
    ///     AtomicValue::integer(2).into(),
    /// ]));
    /// assert!(parameters.initial_match_selection().is_some());
    /// ```
    #[must_use]
    pub fn with_initial_match_selection(mut self, selection: impl Into<XpathValue>) -> Self {
        self.initial_match_selection = Some(selection.into());
        self
    }

    /// The initial match selection, if one was supplied.
    #[must_use]
    pub const fn initial_match_selection(&self) -> Option<&XpathValue> {
        self.initial_match_selection.as_ref()
    }

    /// Declares the *base output URI* of the transformation.
    ///
    /// [XSLT 3.0 §23.1](https://www.w3.org/TR/xslt-30/#result-document) makes this the URI the
    /// principal result is notionally written to: `xsl:result-document/@href` is resolved
    /// against it, and `fn:current-output-uri()` answers it while the principal result is being
    /// built. It is supplied by the caller because only the caller knows where the result is
    /// going; with none supplied `fn:current-output-uri()` answers the empty sequence for the
    /// principal destination and an `href` resolves against the stylesheet's base URI, which is
    /// the best a processor with no destination of its own can do.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_xslt::Parameters;
    ///
    /// let parameters = Parameters::new().with_base_output_uri("file:///out/result.xml");
    /// assert_eq!(parameters.base_output_uri(), Some("file:///out/result.xml"));
    /// ```
    #[must_use]
    pub fn with_base_output_uri(mut self, uri: impl Into<String>) -> Self {
        self.base_output_uri = Some(uri.into());
        self
    }

    /// The base output URI, if one was declared.
    #[must_use]
    pub fn base_output_uri(&self) -> Option<&str> {
        self.base_output_uri.as_deref()
    }

    /// Stops the transformation once `deadline` has passed.
    ///
    /// A stylesheet can run without bound — the specification has no halting guarantee — and a
    /// step budget ([`Self::with_step_limit`]) bounds the *work* rather than the *time*, which
    /// is the wrong currency for a caller that has a request to answer. A deadline is checked
    /// between instructions and at the expression evaluator's own checkpoints, so a runaway
    /// transformation stops with the implementation-defined error `XPDY0130` instead of holding
    /// the thread.
    ///
    /// It is not a hard kill: a single builtin that is itself slow — one `fn:matches` over a
    /// megabyte of input — runs to completion whatever the deadline says. A caller that must
    /// reclaim the thread regardless still needs its own wall-clock guard around the call.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_xslt::Parameters;
    /// use std::time::{Duration, Instant};
    ///
    /// let parameters = Parameters::new().with_deadline(Instant::now() + Duration::from_secs(5));
    /// assert!(parameters.deadline().is_some());
    /// ```
    #[must_use]
    pub fn with_deadline(mut self, deadline: std::time::Instant) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// The deadline, if one was declared.
    #[must_use]
    pub const fn deadline(&self) -> Option<std::time::Instant> {
        self.deadline
    }

    /// Asks for the result without serializing it — see [`Self::skip_serialization`].
    #[must_use]
    pub(crate) fn without_serialization(mut self) -> Self {
        self.skip_serialization = true;
        self
    }

    /// Starts by calling a named template — the XSLT 3.0 `-it` entry point.
    #[must_use]
    pub fn with_initial_template(mut self, name: impl Into<String>) -> Self {
        self.initial_template = Some(name.into());
        self
    }

    /// Supplies the parameters the *initial template* is called with, replacing any set by an
    /// earlier call.
    ///
    /// [XSLT 3.0 §2.3.4](https://www.w3.org/TR/xslt-30/#invoking-initial-template) invokes the
    /// initial template "in the same way as an `xsl:call-template` instruction", so the values
    /// here bind that template's own `xsl:param` declarations. They are a different channel from
    /// [`with_parameter`](Self::with_parameter), which supplies a *stylesheet* parameter — a
    /// global variable that every expression in the stylesheet can see but that never reaches an
    /// `xsl:param` written inside a template.
    ///
    /// Each entry is `(name, value, tunnel)`. A tunnel parameter binds a declaration written
    /// `tunnel="yes"` and stays available to everything the template calls, exactly as
    /// `<xsl:with-param tunnel="yes"/>` would; an ordinary one binds only this template's own
    /// declaration. A declared parameter that nothing here supplies falls back to its own default,
    /// or raises `XTDE0700` when it is required.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_infoset::{AtomicValue, ExpandedName};
    /// use oxixml_xslt::Parameters;
    ///
    /// let parameters = Parameters::new()
    ///     .with_initial_template("start")
    ///     .with_initial_template_parameters([
    ///         (
    ///             ExpandedName::new(None::<&str>, "a"),
    ///             AtomicValue::integer(1234).into(),
    ///             false,
    ///         ),
    ///         (
    ///             ExpandedName::new(Some("http://my.net/"), "b"),
    ///             AtomicValue::integer(999).into(),
    ///             true,
    ///         ),
    ///     ]);
    /// assert_eq!(parameters.initial_template_parameters().len(), 2);
    /// ```
    #[must_use]
    pub fn with_initial_template_parameters(
        mut self,
        parameters: impl IntoIterator<Item = (ExpandedName, XpathValue, bool)>,
    ) -> Self {
        self.initial_template_parameters = parameters.into_iter().collect();
        self
    }

    /// The parameters the initial template will be called with, as `(name, value, tunnel)`.
    #[must_use]
    pub fn initial_template_parameters(&self) -> &[(ExpandedName, XpathValue, bool)] {
        &self.initial_template_parameters
    }

    /// Supplies the parameters of the *apply templates* entry point —
    /// [XSLT 3.0 §2.3.3](https://www.w3.org/TR/xslt-30/#invoking-initial-mode): the values the
    /// `xsl:param` elements of the rules matched in the initial mode are bound to.
    ///
    /// Each entry is `(name, value, tunnel)`; a tunnel parameter travels on to every rule the
    /// matched one goes on to apply, an ordinary one binds only the rule it starts. Calling this
    /// again replaces the whole set.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_infoset::{AtomicValue, ExpandedName};
    /// use oxixml_xslt::Parameters;
    /// use oxixml_xpath::XpathValue;
    ///
    /// let parameters = Parameters::new().with_initial_mode_parameters([(
    ///     ExpandedName::new(None::<&str>, "a"),
    ///     XpathValue::from(AtomicValue::integer(1)),
    ///     false,
    /// )]);
    /// assert_eq!(parameters.initial_mode_parameters().len(), 1);
    /// ```
    #[must_use]
    pub fn with_initial_mode_parameters(
        mut self,
        parameters: impl IntoIterator<Item = (ExpandedName, XpathValue, bool)>,
    ) -> Self {
        self.initial_mode_parameters = parameters.into_iter().collect();
        self
    }

    /// The parameters the initial mode's rules will be applied with, as `(name, value, tunnel)`.
    #[must_use]
    pub fn initial_mode_parameters(&self) -> &[(ExpandedName, XpathValue, bool)] {
        &self.initial_mode_parameters
    }

    /// Starts by calling a named `xsl:function`, the XSLT 3.0 *call function* entry point
    /// ([§2.3.5](https://www.w3.org/TR/xslt-30/#invoking-initial-function)): the result of the
    /// call is the result of the transformation, a sequence read back with
    /// [`Transformation::as_sequence`].
    ///
    /// The function's arguments are supplied positionally with
    /// [`with_initial_function_arguments`](Self::with_initial_function_arguments); the function
    /// chosen is the one whose arity matches how many are given. When set, this takes precedence
    /// over an initial template or the default `apply-templates` entry point.
    #[must_use]
    pub fn with_initial_function(mut self, name: impl Into<String>) -> Self {
        self.initial_function = Some(name.into());
        self
    }

    /// Supplies the positional arguments the initial function is called with, in order. Replaces
    /// any arguments set by an earlier call.
    #[must_use]
    pub fn with_initial_function_arguments(
        mut self,
        arguments: impl IntoIterator<Item = XpathValue>,
    ) -> Self {
        self.initial_function_arguments = arguments.into_iter().collect();
        self
    }

    /// The initial mode, if one was named.
    #[must_use]
    pub fn initial_mode(&self) -> Option<&str> {
        self.initial_mode.as_deref()
    }

    /// The initial template, if one was named.
    #[must_use]
    pub fn initial_template(&self) -> Option<&str> {
        self.initial_template.as_deref()
    }

    /// The initial function, if one was named.
    #[must_use]
    pub fn initial_function(&self) -> Option<&str> {
        self.initial_function.as_deref()
    }

    /// Caps the number of instructions the transformation may execute.
    ///
    /// A stylesheet can loop or recurse without bound — the specification has no halting
    /// guarantee — so an application that runs stylesheets it did not write can bound the work
    /// instead. Exceeding the budget raises the implementation-defined error `OXIXML0001`.
    #[must_use]
    pub const fn with_step_limit(mut self, steps: u64) -> Self {
        self.step_limit = Some(steps);
        self
    }

    /// The instruction budget, if one was set.
    #[must_use]
    pub const fn step_limit(&self) -> Option<u64> {
        self.step_limit
    }

    /// Raises (or lowers) the depth of template instantiation the engine allows before it
    /// reports `XTDE0640`.
    ///
    /// The default is deliberately conservative, because the ceiling is imposed by the *native*
    /// stack rather than by the specification: one level of template instantiation costs
    /// several kilobytes of it, so a chain much deeper than a thousand levels overflows a
    /// process's ordinary 8 MiB main-thread stack and aborts instead of raising an error.
    /// A caller that runs the transformation on a thread with a much larger stack — as a
    /// conformance harness driving stylesheets that recurse a thousand times on purpose must —
    /// can say so here.
    ///
    /// This is a resource limit, not a language feature:
    /// [XSLT 3.0 §2.9](https://www.w3.org/TR/xslt-30/#conformance) leaves such limits
    /// implementation-defined. Setting it far beyond what the thread's stack can carry trades a
    /// diagnosable `XTDE0640` for an abort, so raise it only alongside a bigger stack.
    #[must_use]
    pub const fn with_recursion_limit(mut self, depth: usize) -> Self {
        self.recursion_limit = Some(depth);
        self
    }

    /// The template-instantiation depth limit, if one was set.
    #[must_use]
    pub const fn recursion_limit(&self) -> Option<usize> {
        self.recursion_limit
    }

    /// Declares the stack of the thread the transformation will run on.
    ///
    /// Template instantiation, `xsl:apply-templates` and a stylesheet function call all recurse
    /// on the native stack, and running out of it aborts the whole process — an outcome no
    /// caller can catch. The engine therefore measures the stack it has consumed and stops with
    /// `XTDE0640` while there is still room, which it can only do against a figure the caller
    /// supplies: nothing in the standard library reports how much stack a thread was given.
    ///
    /// The default is [`DEFAULT_STACK_SIZE`](crate::DEFAULT_STACK_SIZE), the 2 MiB
    /// `std::thread::spawn` hands a new thread, which is what a thread pool or an async
    /// executor's worker runs on. A caller on a process's main thread (8 MiB on most systems),
    /// or on a thread it spawned with room for deep recursion on purpose, says so here and gets
    /// the depth that stack can carry.
    ///
    /// The figure is declared to the XPath evaluator as well, for the duration of the
    /// transformation. Recursion through an `xsl:function` is counted by that engine's own
    /// per-thread call ceiling, which without a declaration stays at the 512 calls a 2 MiB
    /// thread is held to — so a caller that raised this and nothing else would still be stopped
    /// there.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_xslt::Parameters;
    ///
    /// let parameters = Parameters::new().with_stack_size(64 * 1024 * 1024);
    /// assert_eq!(parameters.stack_size(), Some(64 * 1024 * 1024));
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

    /// Sets the dynamic context's [implicit
    /// timezone](https://www.w3.org/TR/xpath-functions-31/#dt-implicit-timezone), which every
    /// date/time comparison and `fn:adjust-*-to-timezone` call reads a timezone-less operand in.
    ///
    /// Left unset, the transformation uses the implementation-defined default of UTC.
    #[must_use]
    pub const fn with_implicit_timezone(mut self, timezone: TimezoneOffset) -> Self {
        self.implicit_timezone = Some(timezone);
        self
    }

    /// The implicit timezone, if one was set.
    #[must_use]
    pub const fn implicit_timezone(&self) -> Option<TimezoneOffset> {
        self.implicit_timezone
    }
}

/// A stylesheet, compiled and ready to transform any number of documents.
#[derive(Debug, Clone)]
pub struct CompiledStylesheet {
    ir: Arc<StylesheetIr>,
    resolver: Option<Arc<dyn Resolver>>,
    base_uri: Option<String>,
    /// Every static variable and parameter, with the value the compilation settled on. A static
    /// variable is also an ordinary global variable
    /// ([XSLT 3.0 §9.6](https://www.w3.org/TR/xslt-30/#static-params)), and the only value it can
    /// have at run time is the one it was compiled with — the shadow attributes and `use-when`
    /// expressions that read it were resolved against exactly that — so the values travel with
    /// the compiled stylesheet.
    static_parameters: Vec<(ExpandedName, XpathValue)>,
}

impl CompiledStylesheet {
    /// The serialization properties `xsl:output` set.
    #[must_use]
    pub fn output_properties(&self) -> &OutputProperties {
        &self.ir.output
    }

    /// The number of template rules the stylesheet declares.
    ///
    /// A union pattern counts once per alternative, which is how
    /// [XSLT 3.0 §6.5](https://www.w3.org/TR/xslt-30/#conflict) treats one — except where the
    /// `xsl:template` declares a `priority` of its own, which governs the whole pattern and
    /// keeps it a single rule.
    #[must_use]
    pub fn template_count(&self) -> usize {
        self.ir.templates.len()
    }

    /// Whether the stylesheet declares a template with this name.
    #[must_use]
    pub fn has_named_template(&self, name: &ExpandedName) -> bool {
        self.ir.named_templates.contains_key(name)
    }

    /// Transforms `source`.
    ///
    /// # Errors
    ///
    /// Every dynamic error the stylesheet raises.
    pub fn transform(&self, source: &NodeRef) -> XsltResult<Transformation> {
        self.transform_with(source, &Parameters::new())
    }

    /// Transforms `source` and serializes the result.
    ///
    /// # Errors
    ///
    /// As [`CompiledStylesheet::transform`].
    pub fn transform_to_string(&self, source: &NodeRef) -> XsltResult<String> {
        Ok(self.transform(source)?.into_string())
    }

    /// Transforms `source` under `parameters`.
    ///
    /// # Errors
    ///
    /// Every dynamic error the stylesheet raises, plus `XTDE0040` when the initial template or
    /// mode does not exist.
    pub fn transform_with(
        &self,
        source: &NodeRef,
        parameters: &Parameters,
    ) -> XsltResult<Transformation> {
        self.transform_optional(Some(source), parameters)
    }

    /// Transforms `source` under `parameters`, or runs with an **absent** global context item
    /// when `source` is `None`.
    ///
    /// [XSLT 3.0 §2.3.4](https://www.w3.org/TR/xslt-30/#initiating-transformation) allows an
    /// invocation to have no initial context item at all — the general case
    /// [`CompiledStylesheet::transform_with`] specializes by always supplying one. This is most
    /// useful with [`Parameters::with_initial_template`]: a call-template-style invocation whose
    /// named template declares its own `xsl:context-item` (or none at all) does not need a
    /// source document to start from, and fabricating one would be observable — a template
    /// declaring `<xsl:context-item as="…" use="optional"/>` sees a real, typed item instead of
    /// the absent focus the specification requires.
    ///
    /// # Errors
    ///
    /// As [`CompiledStylesheet::transform_with`].
    pub fn transform_optional(
        &self,
        source: Option<&NodeRef>,
        parameters: &Parameters,
    ) -> XsltResult<Transformation> {
        let mut transformer = Transformer::new(
            Arc::clone(&self.ir),
            self.resolver.clone(),
            self.base_uri.clone(),
            parameters.step_limit,
            parameters.recursion_limit,
            parameters.stack_size,
            parameters.implicit_timezone,
        );
        // The stack the caller declared is the engine's own budget *and* the XPath evaluator's:
        // an `xsl:function` that recurses does so through `oxixml_xpath`'s call accounting, whose
        // ceiling is a per-thread declaration rather than anything the dynamic context carries.
        // Left undeclared it stays at the 512-call default sized for a 2 MiB thread, so a caller
        // that asked for a deeper stack here would still be stopped by `XPDY0130` at 512.
        transformer
            .runtime
            .set_base_output_uri(parameters.base_output_uri.clone());
        // A deadline is installed on top of the stack budget, so both are in force for the
        // whole transformation and both are withdrawn when it returns.
        let _deadline = parameters
            .deadline
            .map(oxixml_xpath::EvaluationLimits::declare_deadline);
        let _limits = oxixml_xpath::EvaluationLimits::declare(
            parameters.stack_size.unwrap_or(crate::DEFAULT_STACK_SIZE),
        );
        let runtime = std::rc::Rc::clone(&transformer.runtime);
        with_runtime(runtime, || self.run(&mut transformer, source, parameters))
    }

    /// Whether `xsl:strip-space` removes `node` from the tree the stylesheet is given.
    ///
    /// Only a whitespace-only text node is ever removed, and only when the element holding it is
    /// stripped and no `xml:space="preserve"` in scope says otherwise —
    /// [XSLT 3.0 §4.4](https://www.w3.org/TR/xslt-30/#stripping). The nearest ancestor that
    /// states `xml:space` settles the question, exactly as the stripper itself reads it.
    fn is_stripped_away(&self, node: &NodeRef) -> bool {
        if node.node_kind() != oxixml_infoset::NodeKind::Text
            || !node.string_value().trim().is_empty()
        {
            return false;
        }
        let Some(parent) = node.parent() else {
            return false;
        };
        if parent.node_kind() != oxixml_infoset::NodeKind::Element {
            return false;
        }
        let mut ancestor = Some(parent.clone());
        while let Some(element) = ancestor {
            let stated = element.attributes().into_iter().find_map(|attribute| {
                attribute
                    .node_name()
                    .filter(|name| {
                        name.namespace() == Some(oxixml_qname::XML_NAMESPACE)
                            && name.local_name() == "space"
                    })
                    .map(|_| attribute.string_value())
            });
            if let Some(value) = stated {
                if value.trim() == "preserve" {
                    return false;
                }
                break;
            }
            ancestor = element.parent();
        }
        crate::run::strips_whitespace(&self.ir, &parent)
    }

    fn run(
        &self,
        transformer: &mut Transformer,
        source: Option<&NodeRef>,
        parameters: &Parameters,
    ) -> XsltResult<Transformation> {
        let stripped;
        let root: Option<NodeRef> = match source {
            // XSLT 3.0 §4.4 strips whitespace from "all documents used as input", the supplied
            // initial context item's own document among them — and a node the stripping removes
            // is not in the tree the stylesheet sees at all, so there is no context item to be
            // had. `strip-space-023` supplies a whitespace-only text node under
            // `xsl:strip-space elements="*"`, and reading `.` in a global variable is `XPDY0002`
            // rather than a reference to a node the transformation cannot see.
            Some(source) if self.ir.strips_any_space && self.is_stripped_away(source) => None,
            Some(source) if self.ir.strips_any_space => {
                stripped = crate::run::strip_whitespace(&self.ir, &source.root())?;
                Some(stripped.root())
            }
            Some(source) => Some(source.clone()),
            None => None,
        };

        // XSLT 3.0 §3.13: what the invocation supplied is the *global context item* only as far
        // as `xsl:global-context-item` admits it, and the declaration is settled before the
        // first global variable is evaluated with it.
        let global_item = crate::run::apply_global_context_item(
            self.ir.global_context_item.as_deref(),
            root.clone().map(Item::Node),
        )?;
        // A static parameter is a global variable whose value the compilation already settled
        // ([XSLT 3.0 §9.6](https://www.w3.org/TR/xslt-30/#static-params)), and nothing the
        // invocation supplies can change it: the shadow attributes and `use-when` expressions
        // that read it were resolved against the compiled value, so a different value here would
        // describe a stylesheet that was never compiled. The bindings are applied in order and
        // the last one for a name wins, so the compiled values go *after* whatever the
        // invocation named.
        let values = match self.static_parameters.is_empty() {
            true => parameters.values.clone(),
            false => {
                let mut values = parameters.values.clone();
                values.extend(self.static_parameters.iter().cloned());
                values
            }
        };
        transformer.evaluate_globals(&values, global_item)?;

        // XSLT 3.0 §2.3.5: a transformation may be started by calling a named `xsl:function`,
        // whose result *is* the result of the transformation. It takes precedence over an initial
        // template, and the result is a sequence — not necessarily a document — so it is kept
        // whole for `Transformation::as_sequence` and also serialized for the textual result.
        if let Some(name) = parameters.initial_function.as_deref() {
            return self.run_initial_function(transformer, name, parameters);
        }

        // XSLT 3.0 §2.3.3: apply-templates invocation needs something to apply the rules to.
        // "It is a dynamic error if the invocation of the stylesheet specifies an initial mode
        // and if no initial match selection is supplied (either explicitly, or defaulted to the
        // global context item)" — `XTDE0044`. Naming an initial template or an initial function
        // selects a different entry point, for which the initial mode is simply unused.
        let names_an_entry_template = parameters.initial_template.is_some()
            || self.ir.named_templates.contains_key(&ExpandedName::new(
                Some(crate::names::XSLT_NAMESPACE),
                "initial-template",
            ));
        if parameters.initial_mode.is_some()
            && root.is_none()
            && parameters.initial_match_selection.is_none()
            && !names_an_entry_template
        {
            return Err(XsltError::dynamic(
                "XTDE0044",
                "the invocation named an initial mode but supplied no initial match selection",
            ));
        }
        let mode = match parameters.initial_mode.as_deref() {
            None | Some("#default") | Some("#unnamed") | Some("") => ModeKey::Default,
            Some(name) => {
                let key = ModeKey::Named(parse_name(name));
                // XSLT 3.0 §2.3: it is a dynamic error if the invocation names an initial mode
                // other than the default that no template rule and no `xsl:mode` declaration
                // uses.
                let Some(mode) = self.ir.modes.get(&key) else {
                    return Err(XsltError::dynamic(
                        "XTDE0045",
                        format!("no template rule is in the mode `{name}`"),
                    ));
                };
                // §6.6.1: a mode declared `visibility="private"` is not an entry point, even
                // when the rules it holds were imported from a module that offered it —
                // overriding the visibility is exactly what the higher-precedence declaration
                // is for.
                if mode.declaration.private {
                    return Err(XsltError::dynamic(
                        "XTDE0045",
                        format!("the mode `{name}` is declared private"),
                    ));
                }
                key
            }
        };
        let mut context = Context::new(root.clone().map(Item::Node));
        context.mode = mode;
        // XSLT 3.0 §2.3.4 lets the invocation supply the initial template's *own* parameters.
        // A tunnel one among them travels with the focus rather than with the call, so the
        // context carries it — but only into the entry point it was supplied for, which is why
        // the entry context is a copy rather than the one the apply-templates invocation uses.
        let entry_parameters = EntryParameters::split(&parameters.initial_template_parameters);
        let mut entry_context = context.clone();
        entry_context.tunnel = entry_parameters.tunnel.clone();
        // §2.3.3 gives the *apply templates* entry point a parameter channel of its own, which
        // is a different set from the initial template's: the tunnel half rides on the context
        // the rules are applied with, the ordinary half is handed to the rule as supplied
        // parameters.
        let mode_parameters = EntryParameters::split(&parameters.initial_mode_parameters);
        context.tunnel = mode_parameters.tunnel.clone();

        // The principal result is a *final* result tree on its way to the serializer, so an
        // item no output method can write is `SENR0001` rather than `XTDE0450`.
        //
        // Unless the method writes no tree at all: the `json` and `adaptive` methods of
        // [Serialization 3.1 §10 and §11](https://www.w3.org/TR/xslt-xquery-serialization-31/#json-output)
        // write a sequence of items, and a map is not a node and could not be put into a tree.
        // Those two collect the transformation's *raw result* and hand it to the method, exactly
        // as an `xsl:result-document` naming the same method does.
        let builds_a_tree = crate::serialize::builds_a_tree(&self.ir.output);
        // A caller that asked for the result *as a sequence* is in the same position as an
        // output method that writes no tree: the items are the answer, so a map, an array or a
        // function item among them is an item to deliver rather than one to reject
        // (`fn:transform` with `delivery-format: 'raw'`).
        let mut emitter = if builds_a_tree && !parameters.skip_serialization {
            Emitter::final_result()
        } else {
            Emitter::sequence()
        };
        emitter.set_document_uri(self.base_uri.clone());
        // XSLT 3.0 §23.1 gives the document node at the root of a final result tree the base
        // output URI; with none supplied, the stylesheet's own base URI is the closest thing
        // this processor knows a relative reference in the result should be read against.
        emitter.set_base_uri(self.base_uri.clone());
        emitter.set_item_separator(self.ir.output.item_separator.as_deref());
        // The raw result of the transformation, kept when the entry point is a template that
        // declares what it returns — see `instantiate_entry_point`.
        let mut raw_result = None;
        match parameters.initial_template.as_deref() {
            Some(name) => {
                let name = parse_name(name);
                let Some(index) = self.ir.named_templates.get(&name).copied() else {
                    return Err(XsltError::dynamic(
                        "XTDE0040",
                        format!("no template named `{name}` is declared"),
                    ));
                };
                raw_result = self.instantiate_entry_point(
                    transformer,
                    index,
                    &entry_context,
                    &entry_parameters.ordinary,
                    &mut emitter,
                )?;
            }
            None => {
                // With no template named by the caller, `xsl:initial-template` is the default
                // entry point ([XSLT 3.0 §2.3.4](https://www.w3.org/TR/xslt-30/#invoking-initial-template):
                // "If no template name is supplied, the default template name is
                // xsl:initial-template"), and the effect "is analogous to … an
                // xsl:call-template instruction" — so the current template rule stays absent.
                //
                // Naming one template both ways asks for two readings at once, and which of
                // them applies is settled by what the invocation supplied. Given an initial
                // match selection, apply-templates invocation (§2.3.3) applies: the same
                // template is reached by matching and becomes the current template rule, which
                // is what an `xsl:next-match` written inside it needs. With no source document
                // there is nothing to match against, so §2.3.4 is the only reading left and the
                // named template is instantiated whether or not it also carries a `match`.
                let default_entry_point = self
                    .ir
                    .named_templates
                    .get(&ExpandedName::new(
                        Some(crate::names::XSLT_NAMESPACE),
                        "initial-template",
                    ))
                    .copied()
                    .filter(|index| {
                        root.is_none()
                            || self
                                .ir
                                .templates
                                .get(*index)
                                .is_some_and(|template| template.pattern.is_none())
                    });
                match default_entry_point {
                    Some(index) => {
                        raw_result = self.instantiate_entry_point(
                            transformer,
                            index,
                            &entry_context,
                            &entry_parameters.ordinary,
                            &mut emitter,
                        )?;
                    }
                    None => match &parameters.initial_match_selection {
                        // XSLT 3.0 §2.3.3: the rules are applied to the whole initial match
                        // selection, one item at a time, with the focus reflecting the sequence
                        // rather than the global context item the globals were evaluated with.
                        Some(selection) => {
                            let selection = selection.clone().into_sequence()?;
                            let size = selection.len();
                            for (index, item) in selection.iter().enumerate() {
                                let mut scoped = Context::new(Some(item.clone()));
                                scoped.mode = context.mode.clone();
                                scoped.position = index + 1;
                                scoped.size = size;
                                transformer.apply_to_initial(
                                    &scoped,
                                    &mode_parameters.ordinary,
                                    &mut emitter,
                                )?;
                            }
                        }
                        None => transformer.apply_to_initial(
                            &context,
                            &mode_parameters.ordinary,
                            &mut emitter,
                        )?,
                    },
                }
            }
        }

        let (tree, items) = emitter.finish()?;
        // XSLT 3.0 §23.1: an `xsl:result-document` with no `href` builds the principal result
        // tree itself. The transformation may not also have written to that destination
        // directly — two final result trees would then share the principal URI, which is
        // `XTDE1490`.
        let mut principal_redirected = false;
        let tree = match transformer.take_principal_result() {
            Some(redirected) => {
                if !tree.root_children().is_empty() {
                    return Err(XsltError::dynamic(
                        "XTDE1490",
                        "the transformation wrote to the principal result destination both \
                         directly and through an xsl:result-document",
                    ));
                }
                // The principal result is the redirected tree, which the entry point did not
                // return: whatever the template's own result was, it is not the result of the
                // transformation, so there is no raw sequence to report.
                raw_result = None;
                principal_redirected = true;
                redirected
            }
            None => tree,
        };
        let properties = match transformer.principal_output() {
            Some(overridden) => (*overridden).clone(),
            None => self.ir.output.clone(),
        };
        // A destination that builds no tree carries its serialization on the tree that stands
        // for it, so the parameter checks and the serializer below need no second reading of the
        // method. A destination an `xsl:result-document` redirected already carries one.
        let tree = match builds_a_tree || tree.raw_serialization().is_some() {
            true => tree,
            false => {
                let raw = Sequence::from_items(items.clone());
                let serialized = crate::serialize::serialize_raw(&raw, &properties)?;
                raw_result_view(&items, self.base_uri.clone())
                    .unwrap_or(tree)
                    .with_raw_serialization(serialized)
            }
        };
        let raw_result = match (&raw_result, parameters.skip_serialization) {
            // The sequence the emitter collected *is* the raw result for such a caller, whatever
            // the entry point was: an apply-templates invocation has no declared result type to
            // keep one from, and its items would otherwise be flattened into the tree.
            (None, true) => Some(Sequence::from_items(items.clone())),
            _ => raw_result,
        };
        // The same is true of a destination whose output method builds no final result tree:
        // the items *are* its result, and nothing else in the transformation holds them.
        let raw_result = match (&raw_result, !builds_a_tree && !principal_redirected) {
            (None, true) => Some(Sequence::from_items(items.clone())),
            _ => raw_result,
        };
        let text = if parameters.skip_serialization {
            String::new()
        } else {
            crate::serialize::check(&tree, &properties)?;
            crate::serialize::serialize(&tree, &properties)
        };
        Ok(Transformation {
            tree,
            text,
            messages: transformer.messages(),
            secondary: transformer.result_documents(),
            properties,
            sequence: raw_result,
        })
    }

    /// Instantiates the entry-point template at `index`, answering its *raw result* when the
    /// template declares one.
    ///
    /// [XSLT 3.0 §2.3.6](https://www.w3.org/TR/xslt-30/#result-trees) makes the result of a
    /// transformation a sequence, which the invoking API either delivers as such or normalizes
    /// into a final result tree. The two are not interchangeable when the entry point declares
    /// `as`: `<xsl:template name="main" as="xs:decimal*">` returns typed atomic values, and a
    /// result tree keeps none of that — the eight decimals of `initial-template-004` become one
    /// text node reading `123.5 -18 12.6 …`. So a template that declares its result is
    /// instantiated into a sequence first, and the sequence is then written to `out`, which
    /// builds exactly the tree — and hence exactly the serialization — that instantiating
    /// straight into `out` would have built.
    ///
    /// A template with no `as` is instantiated directly into `out`: its result is on its way to
    /// a tree in any case, and collecting it first would copy every node it returns for nothing.
    ///
    /// # Errors
    ///
    /// Whatever the template raises, plus the tree-construction errors writing the sequence to
    /// `out` raises.
    fn instantiate_entry_point(
        &self,
        transformer: &mut Transformer,
        index: usize,
        context: &Context,
        supplied: &crate::run::Bindings,
        out: &mut Emitter,
    ) -> XsltResult<Option<Sequence>> {
        let declares_result = self
            .ir
            .templates
            .get(index)
            .is_some_and(|template| template.as_type.is_some());
        if !declares_result {
            transformer.instantiate(index, context, supplied, out)?;
            return Ok(None);
        }
        // Collecting the body is not temporary output state: it still writes to the destination
        // the invocation was writing to, so `disable-output-escaping` and `xsl:result-document`
        // inside it mean what they would have meant.
        let mut raw = Emitter::sequence();
        raw.set_document_uri(self.base_uri.clone());
        raw.set_base_uri(self.base_uri.clone());
        transformer.instantiate(index, context, supplied, &mut raw)?;
        let (_, items) = raw.finish()?;
        for item in &items {
            out.item(item)?;
        }
        Ok(Some(Sequence::from_items(items)))
    }

    /// Runs the *call function* entry point: calls `name` with the positional arguments in
    /// `parameters` and returns their result both as a raw sequence and, serialized, as text.
    ///
    /// # Errors
    ///
    /// `XTDE0041` when no stylesheet function of that name and arity is declared, plus every
    /// dynamic error the argument coercion or the body raises.
    fn run_initial_function(
        &self,
        transformer: &mut Transformer,
        name: &str,
        parameters: &Parameters,
    ) -> XsltResult<Transformation> {
        let name = parse_name(name);
        let arity = parameters.initial_function_arguments.len();
        if !self.ir.functions.contains_key(&(name.clone(), arity)) {
            return Err(XsltError::dynamic(
                "XTDE0041",
                format!("no public xsl:function named `{name}` with {arity} arguments is declared"),
            ));
        }
        let mut arguments = Vec::with_capacity(arity);
        for value in &parameters.initial_function_arguments {
            arguments.push(value.clone().into_sequence()?);
        }
        let result = transformer.call_user_function(&name, arity, &arguments)?;

        // The textual result is the function's result put through the serializer, which — as for
        // any final result — means normalizing the sequence into a result tree first.
        let mut emitter = Emitter::final_result();
        emitter.set_document_uri(self.base_uri.clone());
        emitter.set_base_uri(self.base_uri.clone());
        emitter.set_item_separator(self.ir.output.item_separator.as_deref());
        for item in result.iter() {
            emitter.item(item)?;
        }
        let (tree, _) = emitter.finish()?;
        let properties = match transformer.principal_output() {
            Some(overridden) => (*overridden).clone(),
            None => self.ir.output.clone(),
        };
        crate::serialize::check(&tree, &properties)?;
        let text = crate::serialize::serialize(&tree, &properties);
        Ok(Transformation {
            tree,
            text,
            messages: transformer.messages(),
            secondary: transformer.result_documents(),
            properties,
            sequence: Some(result),
        })
    }
}

/// A navigable view of a *raw result* — the sequence a destination that builds no final result
/// tree produced — or `None` when the sequence holds an item no tree could.
///
/// [XSLT 3.0 §26](https://www.w3.org/TR/xslt-30/#serialization) defaults `build-tree` to `no` for
/// the `json` and `adaptive` methods because their result may be a map or an array, and neither
/// can go into a tree. That is the usual shape, not the only one: a stylesheet may write ordinary
/// nodes to such a destination and expect a caller to navigate them — `maps-017` serializes a map
/// with `fn:serialize` and writes the resulting string inside a literal result element, under
/// `xsl:output method="json"` — and [`Transformation::tree`] is the only way that caller reaches
/// them. The items are therefore normalized into a tree by the same rules
/// [§5.7.1](https://www.w3.org/TR/xslt-30/#constructing-complex-content) would have applied had
/// one been built, which leaves the serialization the method really produced untouched.
///
/// Where an item is a map, an array or another function item there genuinely is no tree, and the
/// caller is left the empty document node such a destination stands for.
fn raw_result_view(items: &[Item], base_uri: Option<String>) -> Option<ResultTree> {
    if items.is_empty()
        || !items
            .iter()
            .all(|item| matches!(item, Item::Node(_) | Item::Atomic(_)))
    {
        return None;
    }
    let mut emitter = Emitter::tree();
    emitter.set_document_uri(base_uri.clone());
    emitter.set_base_uri(base_uri);
    for item in items {
        emitter.item(item).ok()?;
    }
    emitter.finish().ok().map(|(tree, _)| tree)
}

/// The parameters an entry point was invoked with, split into the two channels a template reads
/// them through: an ordinary one is looked up among the values the caller supplied, a tunnel one
/// among the tunnel parameters the focus carries.
struct EntryParameters {
    ordinary: crate::run::Bindings,
    tunnel: crate::run::Bindings,
}

impl EntryParameters {
    fn split(supplied: &[(ExpandedName, XpathValue, bool)]) -> Self {
        let mut ordinary = crate::run::Bindings::default();
        let mut tunnel = crate::run::Bindings::default();
        for (name, value, is_tunnel) in supplied {
            if *is_tunnel {
                tunnel = tunnel.bind(name.clone(), value.clone());
            } else {
                ordinary = ordinary.bind(name.clone(), value.clone());
            }
        }
        Self { ordinary, tunnel }
    }
}

/// Parses a lexical name of the form `local` or `{uri}local`.
fn parse_name(lexical: &str) -> ExpandedName {
    let lexical = lexical.trim();
    if let Some(rest) = lexical.strip_prefix('{')
        && let Some((namespace, local)) = rest.split_once('}')
    {
        return ExpandedName::new(Some(namespace), local);
    }
    ExpandedName::new(None::<&str>, lexical)
}

/// The result of one transformation.
#[derive(Debug, Clone)]
pub struct Transformation {
    tree: ResultTree,
    text: String,
    messages: Vec<String>,
    secondary: Vec<(String, String)>,
    properties: OutputProperties,
    /// The raw result sequence, kept when the transformation was started by calling an initial
    /// function, or by an initial template that declares what it returns, so that atomic values
    /// and their types survive rather than being flattened into the result tree. `None` for the
    /// other tree-producing entry points, whose result is the document node.
    sequence: Option<Sequence>,
}

impl Transformation {
    /// The principal result tree, navigable as an XDM document.
    #[must_use]
    pub fn tree(&self) -> &ResultTree {
        &self.tree
    }

    /// The root node of the principal result tree.
    #[must_use]
    pub fn root(&self) -> NodeRef {
        self.tree.root()
    }

    /// The serialized principal result.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.text
    }

    /// The serialized principal result, consuming the transformation.
    #[must_use]
    pub fn into_string(self) -> String {
        self.text
    }

    /// The messages `xsl:message` produced, in order.
    #[must_use]
    pub fn messages(&self) -> &[String] {
        &self.messages
    }

    /// The secondary results `xsl:result-document` produced, as `(href, text)` pairs.
    #[must_use]
    pub fn secondary_results(&self) -> &[(String, String)] {
        &self.secondary
    }

    /// The serialization properties the principal result was written with.
    #[must_use]
    pub fn output_properties(&self) -> &OutputProperties {
        &self.properties
    }

    /// The result as a sequence: the *raw result* of
    /// [XSLT 3.0 §2.3.6](https://www.w3.org/TR/xslt-30/#result-trees) — atomic values, nodes and
    /// their types intact — when the transformation was started by calling an initial function
    /// or by an initial template that declares `as`, and otherwise a single item, the result
    /// document node, for a caller that wants to keep querying it.
    ///
    /// An entry point that declares `as` and returns nothing therefore answers the empty
    /// sequence here, which is what it returned, rather than the document node of the empty
    /// result tree built from it.
    #[must_use]
    pub fn as_sequence(&self) -> Sequence {
        match &self.sequence {
            Some(sequence) => sequence.clone(),
            None => Sequence::from(self.tree.root()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_parse_in_both_spellings() {
        assert_eq!(parse_name("a"), ExpandedName::new(None::<&str>, "a"));
        assert_eq!(
            parse_name("{urn:x}a"),
            ExpandedName::new(Some("urn:x"), "a")
        );
    }

    #[test]
    fn options_and_parameters_are_builders() {
        let options = StaticOptions::new().with_base_uri("urn:base");
        assert_eq!(options.base_uri(), Some("urn:base"));
        assert!(options.resolver().is_none());

        let parameters = Parameters::new()
            .with_initial_template("main")
            .with_step_limit(1_000);
        assert_eq!(parameters.initial_template(), Some("main"));
        assert_eq!(parameters.initial_mode(), None);
        assert_eq!(parameters.step_limit(), Some(1_000));
    }

    /// A stylesheet with an `f:add#2` and an `f:pair#2` stylesheet function, for the initial
    /// function entry point.
    fn function_stylesheet() -> CompiledStylesheet {
        Stylesheet::compile_str(
            r#"<xsl:stylesheet version="3.0"
                 xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                 xmlns:f="urn:f">
                 <xsl:output omit-xml-declaration="yes"/>
                 <xsl:function name="f:add" as="xs:integer">
                   <xsl:param name="a" as="xs:integer"/>
                   <xsl:param name="b" as="xs:integer"/>
                   <xsl:sequence select="$a + $b"/>
                 </xsl:function>
                 <xsl:function name="f:pair">
                   <xsl:param name="a"/>
                   <xsl:param name="b"/>
                   <xsl:sequence select="$a, $b"/>
                 </xsl:function>
               </xsl:stylesheet>"#,
        )
        .expect("the stylesheet compiles")
    }

    #[test]
    fn an_initial_function_is_called_with_its_arguments() -> XsltResult<()> {
        let compiled = function_stylesheet();
        let parameters = Parameters::new()
            .with_initial_function("{urn:f}add")
            .with_initial_function_arguments([
                XpathValue::from(oxixml_infoset::AtomicValue::integer(1234_i64)),
                XpathValue::from(oxixml_infoset::AtomicValue::integer(5678_i64)),
            ]);
        assert_eq!(parameters.initial_function(), Some("{urn:f}add"));
        let result = compiled.transform_optional(None, &parameters)?;
        assert_eq!(result.as_str(), "6912");
        // The raw result keeps the integer and its type rather than flattening it to text.
        let sequence = result.as_sequence();
        assert_eq!(sequence.len(), 1);
        assert_eq!(
            sequence
                .first()
                .and_then(Item::as_atomic)
                .map(oxixml_infoset::AtomicValue::string_value),
            Some("6912".to_owned())
        );
        Ok(())
    }

    #[test]
    fn an_initial_function_returns_a_whole_sequence() -> XsltResult<()> {
        let compiled = function_stylesheet();
        let parameters = Parameters::new()
            .with_initial_function("{urn:f}pair")
            .with_initial_function_arguments([
                XpathValue::from(oxixml_infoset::AtomicValue::integer(1234_i64)),
                XpathValue::from(oxixml_infoset::AtomicValue::integer(5678_i64)),
            ]);
        let sequence = compiled
            .transform_optional(None, &parameters)?
            .as_sequence();
        assert_eq!(sequence.len(), 2);
        Ok(())
    }

    #[test]
    fn an_undeclared_initial_function_is_xtde0041() {
        let compiled = function_stylesheet();
        // `f:add` exists only at arity 2, so a single-argument call has no match.
        let parameters = Parameters::new()
            .with_initial_function("{urn:f}add")
            .with_initial_function_arguments([XpathValue::from(
                oxixml_infoset::AtomicValue::integer(1_i64),
            )]);
        let error = compiled
            .transform_optional(None, &parameters)
            .expect_err("no f:add#1 is declared");
        assert_eq!(error.code(), Some("XTDE0041"));
    }
}
