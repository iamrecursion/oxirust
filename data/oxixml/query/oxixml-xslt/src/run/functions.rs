//! The functions XSLT adds to the XPath library.
//!
//! [XSLT 3.0 §18](https://www.w3.org/TR/xslt-30/#add-func) defines a handful of functions that
//! only a *transformation* can answer — `current()`, `key()`, `document()`,
//! `current-group()`, `regex-group()`, `system-property()` and the availability tests. They are
//! registered in the `fn:` namespace alongside the F&O library, so an unprefixed call finds
//! them exactly as XSLT 1.0 expects.
//!
//! A [`FunctionRegistry`](oxixml_fn::FunctionRegistry) entry is a plain closure with nowhere to
//! hang a transformation, so each of these reads the engine
//! [published on the current thread](super::current_runtime) for the duration of a run. Outside
//! a transformation they raise `XPST0017`, which is what a conforming processor reports for a
//! function that is not available.

use super::{Runtime, Transformer, current_runtime};
use crate::compile::XsltVersion;
use crate::error::XsltError;
use crate::names::is_available_instruction;
use oxixml_fn::{FN_NAMESPACE, FnError, FnErrorCode, FnResult, FunctionRegistry, builtins};
use oxixml_infoset::{AtomicValue, ExpandedName, Item, NodeRef, QNameValue, Sequence, XdmNode};

/// The whole function library a stylesheet compiles against: F&O plus the XSLT additions.
#[must_use]
pub(crate) fn registry() -> FunctionRegistry {
    registry_with(&[])
}

/// The library, plus one entry per `xsl:function` the stylesheet declares.
///
/// A stylesheet function is a *declaration*, so a call to it must compile wherever it is
/// written — before the declaration as well as after. The compiler therefore discovers every
/// `xsl:function` before it compiles anything, and passes the resulting name-and-arity table
/// here.
#[must_use]
pub(crate) fn registry_with(user: &[crate::compile::FunctionSignature]) -> FunctionRegistry {
    let mut registry = base().clone();
    for signature in user {
        let name = &signature.name;
        let Some(namespace) = name.namespace() else {
            continue;
        };
        let key = name.clone();
        let arity = signature.arity;
        registry.add(namespace, name.local_name(), arity, move |_, arguments| {
            let runtime =
                current_runtime().ok_or_else(|| outside_transformation("a stylesheet function"))?;
            let mut transformer = Transformer {
                runtime: std::rc::Rc::clone(&runtime),
            };
            transformer
                .call_user_function(&key, arity, arguments)
                .map_err(convert)
        });
    }
    registry
}

/// The library without any stylesheet functions, built once per process: assembling the F&O
/// catalogue is far too expensive to repeat per stylesheet element, let alone per pattern
/// match.
fn base() -> &'static FunctionRegistry {
    static BASE: std::sync::OnceLock<FunctionRegistry> = std::sync::OnceLock::new();
    BASE.get_or_init(|| {
        let mut registry = builtins();
        register(&mut registry);
        registry
    })
}

thread_local! {
    /// The static context of the *compile-time* expression being evaluated, if any.
    ///
    /// A `use-when` attribute and a static variable's `select` are evaluated while the
    /// stylesheet is still being compiled, so there is no [`Runtime`] to read a static context
    /// from — and the availability functions need one, because
    /// [XSLT 3.0 §23.1.4](https://www.w3.org/TR/xslt-30/#function-available) expands their
    /// argument "using the in-scope namespaces of the element containing the expression". The
    /// compiler publishes it here for the duration of the evaluation.
    static STATIC_EXPRESSION_CONTEXT: std::cell::RefCell<Vec<std::sync::Arc<oxixml_xpath::StaticContext>>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Publishes `context` as the static context of the compile-time expression about to be
/// evaluated, until the returned guard is dropped.
pub(crate) fn enter_static_expression(
    context: std::sync::Arc<oxixml_xpath::StaticContext>,
) -> StaticExpressionScope {
    STATIC_EXPRESSION_CONTEXT.with(|stack| stack.borrow_mut().push(context));
    StaticExpressionScope
}

/// The guard [`enter_static_expression`] returns; dropping it withdraws the context.
pub(crate) struct StaticExpressionScope;

impl Drop for StaticExpressionScope {
    fn drop(&mut self) {
        STATIC_EXPRESSION_CONTEXT.with(|stack| {
            stack.borrow_mut().pop();
        });
    }
}

thread_local! {
    /// The base URI and resolver of the stylesheet currently being *compiled*.
    ///
    /// A static expression can invoke a whole transformation — `fn:transform` is an ordinary
    /// F&O function, and XSLT 3.0 §9.6 lets a static variable's `select` call one — but there is
    /// no [`Runtime`] then to read a resolver or a base URI from. The compiler publishes its own
    /// here for the duration of the compilation (the suite's `transform-004`).
    static COMPILATION: std::cell::RefCell<Vec<CompilationEnvironment>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// What a compile-time call needs of the compilation running around it.
#[derive(Clone)]
pub(crate) struct CompilationEnvironment {
    /// The static base URI of the stylesheet module being compiled.
    pub(crate) base_uri: Option<String>,
    /// The resolver the compilation reaches other documents through.
    pub(crate) resolver: Option<std::sync::Arc<dyn crate::resolve::Resolver>>,
}

/// Publishes the compilation's own environment until the returned guard is dropped.
pub(crate) fn enter_compilation(environment: CompilationEnvironment) -> CompilationScope {
    COMPILATION.with(|stack| stack.borrow_mut().push(environment));
    CompilationScope
}

/// The guard [`enter_compilation`] returns; dropping it withdraws the environment.
pub(crate) struct CompilationScope;

impl Drop for CompilationScope {
    fn drop(&mut self) {
        COMPILATION.with(|stack| {
            stack.borrow_mut().pop();
        });
    }
}

/// The environment of the compilation running on this thread, if one is.
pub(crate) fn compiling() -> Option<CompilationEnvironment> {
    COMPILATION.with(|stack| stack.borrow().last().cloned())
}

/// The static context the expression the running function was called from compiled against.
///
/// Inside a transformation that is the context the evaluator was handed; while the stylesheet is
/// still compiling it is the one the compiler published for a `use-when` or a static variable.
fn calling_static_context() -> Option<std::sync::Arc<oxixml_xpath::StaticContext>> {
    if let Some(context) = current_runtime().and_then(|runtime| runtime.calling_static_context()) {
        return Some(context);
    }
    STATIC_EXPRESSION_CONTEXT.with(|stack| stack.borrow().last().map(std::sync::Arc::clone))
}

/// The node in focus where the registered function currently running was called, if that focus
/// holds a node.
///
/// [`oxixml_xpath::calling_focus`] reports the *live* focus of the evaluator — the one a path
/// step rebinds — rather than the focus the transformation last recorded before handing an
/// expression to the evaluator. The XSLT functions that identify a document from the context
/// node need that distinction; see the comment in `register_key`.
fn calling_context_node() -> Option<Item> {
    match oxixml_xpath::calling_focus()?.item() {
        oxixml_xpath::XpathItem::Value(item @ Item::Node(_)) => Some(item.clone()),
        _ => None,
    }
}

/// The context item a zero-argument form defaults to, whatever its kind.
///
/// [`calling_context_node`] answers only when the focus holds a node, so that the functions
/// which identify a *document* from it can fall back on the transformation's current item. A
/// function that defaults an argument to `.` needs the opposite: the focus decides, and an
/// atomic value in focus is the value the function operates on rather than a reason to look
/// elsewhere. The transformation's current item is consulted only where the evaluator publishes
/// no focus at all — a call the transformation makes outside any expression.
fn context_item_of_the_call(runtime: &Runtime) -> Option<Item> {
    match oxixml_xpath::calling_focus() {
        Some(focus) => match focus.item() {
            oxixml_xpath::XpathItem::Value(item) => Some(item.clone()),
            // A function item is not a data-model item, so there is nothing here to copy; the
            // caller reports the absence its own way.
            oxixml_xpath::XpathItem::Function(_) => None,
        },
        None => runtime.current_item(),
    }
}

/// Whether the function currently running was reached through a *function item* rather than by
/// a call that names it.
///
/// [XSLT 3.0 §14.2.1](https://www.w3.org/TR/xslt-30/#func-current-group) settles what such a
/// call sees: "the current group is not retained as part of the closure of a function value.
/// This means that the expression `current-group#0` is valid and returns a function value, but
/// any invocation of this function will fail with a dynamic error." §18.2 says the same of
/// `fn:current`, in as many words — "when `current` is called by means of a dynamic function
/// call (for example, `current#0()`), it is evaluated **as if the context item is absent**". So
/// none of the three XSLT extensions to the dynamic context is visible through a function item,
/// and each reports its own absence error: `XTDE1061`, `XTDE1071`, `XTDE1360`.
///
/// This is a property of the *invocation*, not of the focus, so a `current()` written in a
/// predicate — which is a static call however deeply nested — is untouched.
fn captured() -> bool {
    oxixml_xpath::invoked_through_function_item()
}

fn outside_transformation(name: &str) -> FnError {
    FnError::new(
        FnErrorCode::Xpst0017,
        format!("`{name}` is only available inside a transformation"),
    )
}

/// Raises one of XSLT's own error codes from inside a function implementation.
fn xslt_error(code: &str, description: &str) -> FnError {
    FnError::with_name(
        ExpandedName::new(Some(crate::error::ERROR_NAMESPACE), code),
        description.to_owned(),
    )
}

fn convert(error: XsltError) -> FnError {
    let name = ExpandedName::new(
        error.namespace().map(str::to_owned),
        error.code().unwrap_or("XTDE0000"),
    );
    FnError::with_name(name, error.message().to_owned())
}

fn string_of(arguments: &[Sequence], index: usize) -> FnResult<String> {
    let Some(item) = arguments.get(index).and_then(Sequence::first) else {
        return Ok(String::new());
    };
    Ok(item.string_value()?)
}

/// The atomized items of one argument.
///
/// `fn:key` compares key values with the `eq` operator, so it needs them as atomic values
/// rather than as strings.
fn atomics_of(arguments: &[Sequence], index: usize) -> FnResult<Vec<AtomicValue>> {
    let Some(sequence) = arguments.get(index) else {
        return Ok(Vec::new());
    };
    Ok(sequence
        .atomize()?
        .iter()
        .filter_map(|item| item.as_atomic().cloned())
        .collect())
}

fn nodes(sequence: Vec<NodeRef>) -> Sequence {
    Sequence::from_items(sequence.into_iter().map(Item::Node))
}

fn register(registry: &mut FunctionRegistry) {
    register_current(registry);
    register_key(registry);
    register_document(registry);
    register_availability(registry);
    register_grouping(registry);
    register_documented_gaps(registry);
    register_copy_functions(registry);
    super::transform::register(registry);
}

/// The zero-argument forms of `fn:copy-of` and `fn:snapshot`.
///
/// [`oxixml_fn::registry::builtins`] already registers the one-argument forms — the copying
/// itself is a pure function of the node it is given, needing nothing a transformation
/// supplies — but "no argument at all" means "default to the context item"
/// ([F&O 3.1 §2.1.1](https://www.w3.org/TR/xpath-functions-31/#dt-context-item)), and the
/// context item lives in the *dynamic* context this crate's plain function library has no
/// access to. It is available here, so the zero-argument spelling is completed here.
fn register_copy_functions(registry: &mut FunctionRegistry) {
    macro_rules! copy_of_the_context_item {
        ($local:literal, $function:path) => {
            registry.add(FN_NAMESPACE, $local, 0, |_, _| {
                let runtime = current_runtime()
                    .ok_or_else(|| outside_transformation(concat!("fn:", $local)))?;
                // The zero-argument form is exactly `fn:$local(.)`, so an absent context item
                // is the ordinary `XPDY0002` a missing context item always is — not one of
                // this crate's own XSLT-specific codes. `.` is the focus of the expression the
                // call is written in, which a path step rebinds — the suite's copy-of-001
                // writes `/doc/*/copy-of()` and expects the copy of `<elem>`, not of the
                // document node the transformation is still recording as its current item.
                let item = context_item_of_the_call(&runtime).ok_or_else(|| {
                    FnError::new(
                        FnErrorCode::Xpdy0002,
                        concat!(
                            "fn:",
                            $local,
                            "() needs a context item, and none is defined"
                        ),
                    )
                })?;
                Ok(match item {
                    Item::Node(node) => Sequence::from(Item::Node($function(&node)?)),
                    other => Sequence::from(other),
                })
            });
        };
    }
    copy_of_the_context_item!("copy-of", copy_of);
    copy_of_the_context_item!("snapshot", snapshot);

    macro_rules! copy_of_the_argument {
        ($local:literal, $function:path) => {
            registry.add(FN_NAMESPACE, $local, 1, |_, arguments| {
                let mut out = Vec::new();
                for item in arguments.first().map(Sequence::iter).unwrap_or_default() {
                    out.push(match item {
                        Item::Node(node) => Item::Node($function(node)?),
                        other => other.clone(),
                    });
                }
                Ok(Sequence::from_items(out))
            });
        };
    }
    copy_of_the_argument!("copy-of", copy_of);
    copy_of_the_argument!("snapshot", snapshot);
}

/// [`fn:copy-of`](https://www.w3.org/TR/xslt-30/#func-copy-of) applied to one node.
///
/// [`oxixml_fn::nodes::copy_of`] does the work for every node its own `TreeBuilder` can hold.
/// That builder always roots what it builds at a document node, so it refuses `FODC0006` for a
/// *parentless* attribute, namespace or text node — a copy that would itself have to be
/// parentless. This crate's builder can produce exactly that (it is what a sequence constructor
/// declared `as="node()*"` yields in the first place), so those three kinds are completed here.
///
/// # Errors
///
/// Whatever the underlying copy raises.
fn copy_of(node: &NodeRef) -> FnResult<NodeRef> {
    match parentless_copy(node) {
        Some(copy) => copy,
        None => oxixml_fn::nodes::copy_of(node),
    }
}

/// [`fn:snapshot`](https://www.w3.org/TR/xslt-30/#func-snapshot) applied to one node.
///
/// A node with no parent has no ancestors to carry, so its snapshot is its copy — which is what
/// the specification's own XSLT implementation of the function produces for one (the
/// `mode="snapshot"` template reaches `. is $origin` at the root and copies it).
///
/// # Errors
///
/// Whatever the underlying copy raises.
fn snapshot(node: &NodeRef) -> FnResult<NodeRef> {
    if let Some(copy) = parentless_copy(node) {
        return copy;
    }
    // An attribute or a namespace node *with* a parent needs its ancestor path rebuilt around
    // it, and the answer is the copied attribute or namespace node rather than the root of what
    // was built — `snapshot-0101e` reads `snapshot(//@srsName)/parent::Envelope` back out.
    if matches!(
        node.node_kind(),
        oxixml_infoset::NodeKind::Attribute | oxixml_infoset::NodeKind::Namespace
    ) {
        return snapshot_with_ancestors(node);
    }
    oxixml_fn::nodes::snapshot(node)
}

/// The snapshot of an attribute or namespace node that has a parent element.
///
/// [XSLT 3.0 §21.2](https://www.w3.org/TR/xslt-30/#func-snapshot) keeps "the ancestors of the
/// supplied node with their attributes and namespaces" and nothing else of them, so the copy is
/// one element per ancestor, each carrying its own attributes and in-scope namespaces, with the
/// origin attached to the innermost.
///
/// # Errors
///
/// Whatever building the copy raises.
fn snapshot_with_ancestors(node: &NodeRef) -> FnResult<NodeRef> {
    use oxixml_infoset::NodeKind;
    let mut ancestors = Vec::new();
    let mut current = node.parent();
    while let Some(ancestor) = current {
        current = ancestor.parent();
        if ancestor.node_kind() == NodeKind::Element {
            ancestors.push(ancestor);
        }
    }
    ancestors.reverse();
    let mut builder = crate::tree::ResultBuilder::new();
    for ancestor in &ancestors {
        let name = ancestor
            .node_name()
            .unwrap_or_else(|| ExpandedName::new(None::<&str>, "element"));
        let prefix = name.prefix().map(str::to_owned);
        builder.start_element(name, prefix);
        for binding in ancestor.namespaces() {
            let prefix = binding.node_name().map(|n| n.local_name().to_owned());
            if prefix.as_deref() == Some("xml") {
                continue;
            }
            builder
                .namespace(prefix, &binding.string_value())
                .map_err(convert_to_fn_error)?;
        }
        for attribute in ancestor.attributes() {
            let name = attribute
                .node_name()
                .unwrap_or_else(|| ExpandedName::new(None::<&str>, "attribute"));
            let prefix = name.prefix().map(str::to_owned);
            builder
                .attribute(name, prefix, attribute.string_value())
                .map_err(convert_to_fn_error)?;
        }
    }
    for _ in &ancestors {
        builder.end_element().map_err(convert_to_fn_error)?;
    }
    let (tree, roots) = builder.build().map_err(convert_to_fn_error)?;
    let mut copied = roots
        .first()
        .and_then(|index| tree.node_at(*index))
        .ok_or_else(|| {
            FnError::new(
                FnErrorCode::Fodc0006,
                "the ancestors of the node could not be copied",
            )
        })?;
    // One element per ancestor and no other content, so the path down is unambiguous.
    for _ in 1..ancestors.len() {
        copied = copied
            .children()
            .into_iter()
            .find(|child| child.node_kind() == NodeKind::Element)
            .ok_or_else(|| {
                FnError::new(
                    FnErrorCode::Fodc0006,
                    "the copied ancestor path is incomplete",
                )
            })?;
    }
    let wanted = node.node_name();
    let found = match node.node_kind() {
        NodeKind::Attribute => copied
            .attributes()
            .into_iter()
            .find(|attribute| attribute.node_name() == wanted),
        _ => copied
            .namespaces()
            .into_iter()
            .find(|binding| binding.node_name() == wanted),
    };
    found.ok_or_else(|| {
        FnError::new(
            FnErrorCode::Fodc0006,
            "the copied node could not be located in the snapshot",
        )
    })
}

/// The copy of a parentless attribute, namespace or text node, or `None` when the node is one
/// [`oxixml_fn::nodes`] can copy itself.
fn parentless_copy(node: &NodeRef) -> Option<FnResult<NodeRef>> {
    use oxixml_infoset::NodeKind;
    if node.parent().is_some() {
        return None;
    }
    let mut builder = crate::tree::ResultBuilder::for_sequence();
    match node.node_kind() {
        NodeKind::Attribute => {
            let name = node.node_name()?;
            let prefix = name.prefix().map(str::to_owned);
            builder.detached_attribute(name, prefix, node.string_value());
        }
        NodeKind::Namespace => builder.detached_namespace(
            node.node_name()
                .map(|name| name.local_name().to_owned())
                .filter(|prefix| !prefix.is_empty()),
            &node.string_value(),
        ),
        NodeKind::Text => builder.text(&node.string_value(), false),
        _ => return None,
    }
    Some(materialize_one(builder))
}

/// The single node a builder holding exactly one top-level node produced.
fn materialize_one(builder: crate::tree::ResultBuilder) -> FnResult<NodeRef> {
    let (tree, roots) = builder.build().map_err(convert_to_fn_error)?;
    roots
        .first()
        .and_then(|index| tree.node_at(*index))
        .ok_or_else(|| {
            FnError::new(
                FnErrorCode::Fodc0006,
                "the copy of a parentless node could not be built",
            )
        })
}

/// Carries an [`XsltError`] out through a function-library error.
fn convert_to_fn_error(error: XsltError) -> FnError {
    FnError::with_name(
        ExpandedName::new(
            Some(crate::error::ERROR_NAMESPACE),
            error.code().unwrap_or("XTDE0001"),
        ),
        error.to_string(),
    )
}

/// The XSLT 3.0 functions whose *declarations* this processor does not implement.
///
/// `xsl:accumulator` and `xsl:merge` are documented gaps: the compiler rejects the
/// declarations with `OXIXML0001`, so a stylesheet this engine can run never has an accumulator
/// or a merge in scope. The functions that read them are nonetheless part of the XPath static
/// context, and leaving them unregistered reports `XPST0017` — "no such function" — where the
/// specification prescribes a specific dynamic error for exactly the situation that always
/// holds here: the accumulator is not declared, the merge group is absent.
///
/// [XSLT 3.0 §18.4.2](https://www.w3.org/TR/xslt-30/#func-accumulator-before) makes a name
/// that matches no visible `xsl:accumulator` `XTDE3340`;
/// [§18.5.1](https://www.w3.org/TR/xslt-30/#func-current-merge-group) and
/// [§18.5.2](https://www.w3.org/TR/xslt-30/#func-current-merge-key) make an absent merge group
/// `XTDE3480` and an absent merge key `XTDE3510`.
fn register_documented_gaps(registry: &mut FunctionRegistry) {
    for name in ["accumulator-before", "accumulator-after"] {
        registry.add(FN_NAMESPACE, name, 1, move |_, arguments| {
            let lexical = string_of(arguments, 0)?;
            Err(xslt_error(
                "XTDE3340",
                &format!("`{lexical}` names no xsl:accumulator declaration in scope"),
            ))
        });
    }
    for arity in 0..=1 {
        registry.add(
            FN_NAMESPACE,
            "current-merge-group",
            arity,
            move |_, _arguments| {
                Err(xslt_error(
                    "XTDE3480",
                    "fn:current-merge-group was called where the current merge group is absent",
                ))
            },
        );
    }
    registry.add(FN_NAMESPACE, "current-merge-key", 0, |_, _| {
        Err(xslt_error(
            "XTDE3510",
            "fn:current-merge-key was called where the current merge key is absent",
        ))
    });

    // [XSLT 3.0 §18.1.3](https://www.w3.org/TR/xslt-30/#func-stream-available) asks whether a
    // document is available for *streamed* processing, and lets the answer be `false` whenever
    // "the processor detected some other condition that would prevent a call on xsl:stream
    // executing successfully". This processor answers `no` to
    // `system-property('xsl:supports-streaming')`, so no document is ever available for
    // streaming and the honest answer is `false` for every URI — including the empty sequence,
    // which the specification singles out. The function exists because it is part of the XPath
    // static context a stylesheet compiles against: leaving it unregistered would report
    // `XPST0017`, "no such function", where the specification defines one (`function-2001`).
    registry.add(FN_NAMESPACE, "stream-available", 1, |_, _| {
        Ok(Sequence::from(AtomicValue::boolean(false)))
    });
}

fn register_current(registry: &mut FunctionRegistry) {
    registry.add(FN_NAMESPACE, "current", 0, |_, _| {
        let runtime = current_runtime().ok_or_else(|| outside_transformation("fn:current"))?;
        // XSLT 3.0 §18.1: `current()` evaluated where the context item is absent is
        // `XTDE1360`, not the empty sequence.
        match runtime.current_item().filter(|_| !captured()) {
            Some(item) => Ok(Sequence::singleton(item)),
            None => Err(xslt_error(
                "XTDE1360",
                "fn:current was called where the context item is absent",
            )),
        }
    });

    // XSLT 3.0 §18.9: the URI of the `xsl:result-document` instruction currently being
    // evaluated, or the empty sequence in temporary output state, or the base output URI for the
    // principal output destination — and nothing at all for that destination when the invocation
    // declared no base output URI.
    //
    // [§24.3](https://www.w3.org/TR/xslt-30/#current-output-uri) also clears the URI for a
    // *dynamic function call*, which is the only one of its five clauses that is a property of
    // how this function was reached rather than of where the engine is: `current-output-uri#0`
    // is a legal function item whose invocation must answer the empty sequence
    // (`current-output-uri-016`). Unlike `fn:current` and the grouping accessors, which report a
    // dynamic error when they are reached that way, this one has an answer for the case — the
    // URI is *absent*, and an absent URI is the empty sequence.
    registry.add(FN_NAMESPACE, "current-output-uri", 0, |_, _| {
        let runtime =
            current_runtime().ok_or_else(|| outside_transformation("fn:current-output-uri"))?;
        Ok(match runtime.current_output_uri().filter(|_| !captured()) {
            Some(uri) => Sequence::from(AtomicValue::any_uri(uri)),
            None => Sequence::empty(),
        })
    });
}

/// The lexical name an availability function was handed, checked against the `EQName`
/// production and against the prefixes the stylesheet declares.
///
/// [XSLT 3.0 §18.2](https://www.w3.org/TR/xslt-30/#function-available) makes a name that is
/// not an `EQName`, or one whose prefix is not in scope, a dynamic error under the code the
/// caller names.
fn check_lexical_name(lexical: &str, code: &str) -> FnResult<()> {
    let trimmed = lexical.trim();
    if trimmed.starts_with("Q{") {
        return Ok(());
    }
    if !oxixml_qname::is_qname(trimmed) {
        return Err(xslt_error(
            code,
            &format!("`{lexical}` is not a valid EQName"),
        ));
    }
    let Some((prefix, _)) = trimmed.split_once(':') else {
        return Ok(());
    };
    if prefix == "xml" || prefix == "xsl" || prefix == "fn" || prefix == "xs" {
        return Ok(());
    }
    // Outside a transformation the caller is a `use-when` expression, or a static variable's
    // `select`, evaluated while the stylesheet is still being compiled: the table of declared
    // prefixes does not exist yet, but the expression's own static context does, and it is the
    // one §23.1.4 asks about (`extension-functions-0104` calls
    // `function-available('test:foo')` under a `use-when` with no `test` prefix in scope). With
    // neither available the prefix is taken on trust rather than reported as undeclared.
    let Some(runtime) = current_runtime() else {
        return match calling_static_context() {
            None => Ok(()),
            Some(context) => {
                if context
                    .namespace(prefix)
                    .is_some_and(|namespace| !namespace.is_empty())
                {
                    Ok(())
                } else {
                    Err(xslt_error(
                        code,
                        &format!("no namespace declaration is in scope for the prefix `{prefix}`"),
                    ))
                }
            }
        };
    };
    if runtime.ir.declared_prefixes.contains(prefix) {
        return Ok(());
    }
    Err(xslt_error(
        code,
        &format!("no namespace declaration is in scope for the prefix `{prefix}`"),
    ))
}

fn register_key(registry: &mut FunctionRegistry) {
    for arity in 2..=3 {
        registry.add(FN_NAMESPACE, "key", arity, move |_, arguments| {
            let runtime = current_runtime().ok_or_else(|| outside_transformation("fn:key"))?;
            let lexical = string_of(arguments, 0)?;
            let name = parse_key_name(&lexical)?;
            let values = atomics_of(arguments, 1)?;
            // XSLT 3.0 §20.2.2: the third argument names the *selected subtree* — "the set of
            // nodes that have `$top` as an ancestor-or-self node" — so it both identifies the
            // document to search and confines the answer to the part of it below that node.
            let top = match arguments.get(2).and_then(Sequence::first) {
                Some(Item::Node(node)) => Some(node.clone()),
                _ => None,
            };
            let origin = match &top {
                Some(node) => node.clone(),
                // XSLT 3.0 §18.3: without a third argument the document to search is "the
                // document containing the context node". That is the node in focus *where the
                // call is written*, which a path step rebinds — `$e/key('k','v')` searches
                // `$e`'s tree, not the template's own (the suite's key-072). The evaluator
                // publishes that focus for the duration of the call; the transformation's own
                // record of the current node is the fallback for a call made from somewhere the
                // evaluator is not driving.
                None => match calling_context_node().or_else(|| runtime.current_item()) {
                    Some(Item::Node(node)) => node,
                    // XSLT 3.0 §18.3 / the suite's error-1270a: calling the two-argument form
                    // with no context node at all is `XTDE1270` — the same code `key_index`
                    // already raises when a node *is* available but its tree is not rooted at
                    // a document node, not the generic `XPTY0004` a plain argument-type
                    // mismatch would be.
                    _ => {
                        return Err(xslt_error(
                            "XTDE1270",
                            "fn:key was called with two arguments where there is no context \
                             node to identify the document to search",
                        ));
                    }
                },
            };
            let mut transformer = Transformer {
                runtime: std::rc::Rc::clone(&runtime),
            };
            let found = transformer
                .key_lookup(&name, &values, &origin, top.as_ref())
                .map_err(convert)?;
            Ok(nodes(found))
        });
    }
}

/// Parses the lexical `QName` a `key()` call names its key with.
///
/// The prefix is resolved against the namespaces the *stylesheet* declared, which the runtime
/// knows through the key declarations themselves: a key is looked up by expanded name, and an
/// unprefixed name is in no namespace.
fn parse_key_name(lexical: &str) -> FnResult<ExpandedName> {
    let runtime = current_runtime().ok_or_else(|| outside_transformation("fn:key"))?;
    let lexical = lexical.trim();
    match lexical.split_once(':') {
        None => Ok(ExpandedName::new(None::<&str>, lexical)),
        Some((prefix, local)) => runtime
            .ir
            .keys
            .iter()
            .map(|definition| &definition.name)
            // A prefixed `EQName` always expands to a non-empty namespace, so a key declared
            // in no namespace is not the one `my:k` names — `error-1260e` turns on exactly
            // that distinction.
            .find(|name| name.local_name() == local && name.namespace().is_some())
            .cloned()
            .ok_or_else(|| {
                FnError::with_name(
                    ExpandedName::new(Some(crate::error::ERROR_NAMESPACE), "XTDE1260"),
                    format!("no xsl:key named `{prefix}:{local}` is declared"),
                )
            }),
    }
}

fn register_document(registry: &mut FunctionRegistry) {
    for arity in 1..=2 {
        registry.add(FN_NAMESPACE, "document", arity, move |context, arguments| {
            let runtime = current_runtime().ok_or_else(|| outside_transformation("fn:document"))?;
            let Some(resolver) = &runtime.resolver else {
                return Err(FnError::new(
                    FnErrorCode::Fodc0002,
                    "no resolver is installed, so document() cannot retrieve anything",
                ));
            };
            // XSLT 3.0 §18.6 resolves the reference against the *static* base URI of the call,
            // not against the principal module's URI: a `document()` written inside an
            // `xsl:include`, or under an `xml:base`, names documents relative to where it is
            // written. `fn:doc` already reads the same property.
            let static_base = context.static_base_uri().map(str::to_owned);
            // A node supplied as the second argument fixes the base URI; a *parentless* node
            // with no base URI of its own leaves the reference unresolvable, which XSLT 3.0
            // §18.6 makes `XTDE1162` rather than a fall-back to the static base URI.
            let base = match arguments.get(1).and_then(Sequence::first) {
                Some(Item::Node(node)) => base_uri_of(node, static_base.as_deref()),
                _ => static_base.clone(),
            };
            let mut found = Vec::new();
            let items: Vec<Item> = arguments
                .first()
                .map(|sequence| sequence.iter().cloned().collect())
                .unwrap_or_default();
            for item in items {
                let reference = item.string_value()?;
                // A URI held in a node is resolved against *that node's* base URI.
                let base = match &item {
                    Item::Node(node) if arguments.len() < 2 => {
                        base_uri_of(node, static_base.as_deref())
                    }
                    _ => base.clone(),
                };
                // XSLT 3.0 §18.6 gives `fn:document` its own error codes rather than the
                // generic `FODC0002` of `fn:doc`: a fragment identifier the processor cannot
                // honour is `XTRE1160`, and a relative reference with no base URI to resolve
                // it against is `XTDE1162`.
                if reference.contains('#') {
                    return Err(xslt_error(
                        "XTRE1160",
                        &format!(
                            "the fragment identifier of `{reference}` names no media type this processor recognizes"
                        ),
                    ));
                }
                if base.is_none() && !is_absolute_uri(&reference) {
                    return Err(xslt_error(
                        "XTDE1162",
                        &format!("`{reference}` is relative and there is no base URI to resolve it against"),
                    ));
                }
                let uri = resolver.resolve_uri(&reference, base.as_deref());
                let Some(node) = resolver
                    .document(&uri)
                    .or_else(|| resolver.document(&reference))
                else {
                    return Err(FnError::new(
                        FnErrorCode::Fodc0002,
                        format!("the document `{reference}` could not be retrieved"),
                    ));
                };
                let node = runtime.stripper.apply(&uri, node);
                if !found.contains(&node) {
                    found.push(node);
                }
            }
            if arguments.first().is_some_and(Sequence::is_empty) {
                return Ok(Sequence::empty());
            }
            Ok(nodes(found))
        });
    }
}

/// The XSLT system properties this processor recognizes, each paired with the value
/// [`system-property()`](https://www.w3.org/TR/xslt-30/#func-system-property) reports for it.
///
/// Every name is in the XSLT namespace. `fn:system-property` looks a name up here and
/// `fn:available-system-properties` returns these very names as QNames, so the two functions
/// agree by construction — the requirement [XSLT 3.0 §18.2](https://www.w3.org/TR/xslt-30/#function-library)
/// places on them. Only `xsl:version` varies with the stylesheet.
fn recognized_system_properties(version: XsltVersion) -> [(&'static str, String); 14] {
    [
        ("version", version.as_str().to_owned()),
        ("vendor", "COOLJAPAN OxiXML".to_owned()),
        (
            "vendor-url",
            "https://github.com/cool-japan/oxixml".to_owned(),
        ),
        ("product-name", "oxixml-xslt".to_owned()),
        ("product-version", env!("CARGO_PKG_VERSION").to_owned()),
        ("is-schema-aware", "no".to_owned()),
        ("supports-serialization", "yes".to_owned()),
        // This processor implements XSLT 3.0 semantics only: a 1.0 construct that needs
        // backwards-compatible behaviour is reported rather than reinterpreted.
        ("supports-backwards-compatibility", "no".to_owned()),
        ("supports-namespace-axis", "yes".to_owned()),
        ("supports-streaming", "no".to_owned()),
        ("supports-dynamic-evaluation", "no".to_owned()),
        ("supports-higher-order-functions", "yes".to_owned()),
        ("xpath-version", "3.1".to_owned()),
        // The schema types this processor knows are XSD 1.0's: `fn:type-available` answers
        // `false` for `xs:dateTimeStamp` and the other types 1.1 adds, and a `cast as` to one of
        // them is `XPST0051`. `xsl:xsd-version` has to report the same reading, or a stylesheet
        // that branches on it takes the branch this processor cannot honour.
        ("xsd-version", "1.0".to_owned()),
    ]
}

/// Which function library `fn:function-available` reports on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Library {
    /// Everything a transformation can call.
    Whole,
    /// The much smaller library a *static expression* — `use-when`, and the `select` of a
    /// static variable — may call; see [`static_registry`].
    Static,
}

/// The function library a *static expression* is compiled and evaluated against.
///
/// [XSLT 3.0 §9.7](https://www.w3.org/TR/xslt-30/#static-expressions) names it exactly: "the
/// core functions defined in [Functions and Operators], together with the functions
/// `element-available`, `function-available`, `type-available`, and `system-property`". The rest
/// of the XSLT library — `current`, `key`, `document`, the grouping and accumulator accessors —
/// depends on a transformation that has not started, and "stylesheet functions are not included
/// in the context", so a call to any of them is `XPST0017` rather than a value.
///
/// `fn:function-available` reports on this same library when it is called from inside a static
/// expression, "the effect of this rule [being] to ensure that `function-available` returns true
/// in respect of functions that can be called within the static expression" — the suite's
/// `use-when-0407a` asks about six functions one by one and expects `generate-id` (which F&O 3.0
/// defines) and `system-property` to answer true while `current`, `key` and the two
/// unparsed-entity accessors answer false.
pub(crate) fn static_registry() -> &'static FunctionRegistry {
    static LIBRARY: std::sync::OnceLock<FunctionRegistry> = std::sync::OnceLock::new();
    LIBRARY.get_or_init(|| {
        let mut registry = builtins();
        register_availability_in(&mut registry, Library::Static);
        registry
    })
}

fn register_availability(registry: &mut FunctionRegistry) {
    register_availability_in(registry, Library::Whole);
}

fn register_availability_in(registry: &mut FunctionRegistry, library: Library) {
    registry.add(FN_NAMESPACE, "system-property", 1, |_, arguments| {
        let lexical = string_of(arguments, 0)?;
        check_lexical_name(&lexical, "XTDE1390")?;
        let local = lexical.rsplit(':').next().unwrap_or(&lexical);
        let prefixed = lexical.contains(':');
        let version = current_runtime().map_or(XsltVersion::V30, |runtime| runtime.ir.version);
        // An unrecognized property, and every unprefixed name, is the empty string.
        let value = if prefixed {
            recognized_system_properties(version)
                .into_iter()
                .find(|(name, _)| *name == local)
                .map_or_else(String::new, |(_, value)| value)
        } else {
            String::new()
        };
        Ok(Sequence::from(AtomicValue::string(value)))
    });

    if library == Library::Whole {
        registry.add(FN_NAMESPACE, "available-system-properties", 0, |_, _| {
            let version = current_runtime().map_or(XsltVersion::V30, |runtime| runtime.ir.version);
            Ok(Sequence::from_items(
                recognized_system_properties(version)
                    .into_iter()
                    .map(|(local, _)| {
                        Item::Atomic(AtomicValue::qname(QNameValue::new(
                            Some("xsl"),
                            Some(crate::names::XSLT_NAMESPACE),
                            local,
                        )))
                    })
                    .collect::<Vec<_>>(),
            ))
        });
    }

    registry.add(FN_NAMESPACE, "element-available", 1, |_, arguments| {
        let lexical = string_of(arguments, 0)?;
        check_lexical_name(&lexical, "XTDE1440")?;
        // XSLT 3.0 §23.2.2: a name in the XSLT namespace names an instruction this processor
        // implements; "if the expanded QName has a null namespace URI, the function will return
        // false"; and this processor implements no extension instruction, so no other namespace
        // can answer true either.
        let available =
            match expand_availability_name(&split_eqname(&lexical), UnprefixedIn::DefaultNamespace)
            {
                Expanded::Resolved(namespace, local) => {
                    namespace.as_deref() == Some(crate::names::XSLT_NAMESPACE)
                        && is_available_xslt_element(&local)
                }
                Expanded::ByLocalName(local) => is_available_xslt_element(&local),
            };
        Ok(Sequence::from(AtomicValue::boolean(available)))
    });

    for arity in 1..=2 {
        registry.add(
            FN_NAMESPACE,
            "function-available",
            arity,
            move |_, arguments| {
                let lexical = string_of(arguments, 0)?;
                check_lexical_name(&lexical, "XTDE1400")?;
                let wanted_arity = arguments
                    .get(1)
                    .and_then(Sequence::first)
                    .and_then(|item| item.as_atomic().and_then(AtomicValue::as_f64))
                    .map(|value| value as usize);
                Ok(Sequence::from(AtomicValue::boolean(function_available(
                    &lexical,
                    wanted_arity,
                    library,
                ))))
            },
        );
    }

    registry.add(FN_NAMESPACE, "type-available", 1, |_, arguments| {
        let lexical = string_of(arguments, 0)?;
        check_lexical_name(&lexical, "XTDE1428")?;
        // A type name is always in a namespace: the schema namespace for the built-in types,
        // which are the only ones a processor that is not schema-aware has. XSLT 3.0 §23.1.4
        // expands the name against the namespace declarations in scope for the expression, and
        // gives an unprefixed one the default namespace.
        let available =
            match expand_availability_name(&split_eqname(&lexical), UnprefixedIn::DefaultNamespace)
            {
                Expanded::Resolved(namespace, local) => {
                    namespace.as_deref() == Some(XSD_NAMESPACE) && is_schema_type(&local)
                }
                Expanded::ByLocalName(local) => is_schema_type(&local),
            };
        Ok(Sequence::from(AtomicValue::boolean(available)))
    });

    // The unparsed-entity accessors read the `[unparsed entities]` property of a document node,
    // which the DTD's `NDATA` declarations populate — XSLT 3.0 §20.4 and XDM 3.1 §6.1. They need
    // a source document, so they are outside the library §9.7 gives a static expression, and
    // `use-when-0407a` asks `function-available` about both of them and expects `false`.
    if library == Library::Static {
        return;
    }
    registry.add(FN_NAMESPACE, "unparsed-entity-uri", 1, |_, arguments| {
        let name = string_of(arguments, 0)?;
        let document = entity_document(None, "XTDE1370")?;
        Ok(Sequence::from(AtomicValue::any_uri(unparsed_entity_uri(
            &document, &name,
        ))))
    });
    registry.add(
        FN_NAMESPACE,
        "unparsed-entity-public-id",
        1,
        |_, arguments| {
            let name = string_of(arguments, 0)?;
            let document = entity_document(None, "XTDE1380")?;
            Ok(Sequence::from(AtomicValue::string(
                unparsed_entity_public_id(&document, &name),
            )))
        },
    );
    registry.add(FN_NAMESPACE, "unparsed-entity-uri", 2, |_, arguments| {
        let name = string_of(arguments, 0)?;
        let document = entity_document(arguments.get(1), "XTDE1370")?;
        Ok(Sequence::from(AtomicValue::any_uri(unparsed_entity_uri(
            &document, &name,
        ))))
    });
    registry.add(
        FN_NAMESPACE,
        "unparsed-entity-public-id",
        2,
        |_, arguments| {
            let name = string_of(arguments, 0)?;
            let document = entity_document(arguments.get(1), "XTDE1380")?;
            Ok(Sequence::from(AtomicValue::string(
                unparsed_entity_public_id(&document, &name),
            )))
        },
    );
}

/// The absolute URI of the unparsed entity `name`, or the zero-length string when the document
/// declares no such entity.
///
/// [XSLT 3.0 §20.4](https://www.w3.org/TR/xslt-30/#unparsed-entity-uri) asks for "the URI of the
/// unparsed entity", which [XDM 3.1 §6.1](https://www.w3.org/TR/xpath-datamodel-31/#DocumentNode)
/// defines as the system identifier *resolved against the document's base URI*. That absolute
/// form is what the tree carries; a system identifier that could not be resolved — a relative
/// reference in a tree with no URI of its own — is reported as it was written, which locates the
/// resource as well as anything else available can.
fn unparsed_entity_uri(document: &NodeRef, name: &str) -> String {
    document
        .root()
        .unparsed_entity(name)
        .map(|entity| entity.uri.unwrap_or(entity.system_id))
        .unwrap_or_default()
}

/// The public identifier of the unparsed entity `name`, or the zero-length string when the
/// document declares no such entity or the declaration gave no public identifier.
fn unparsed_entity_public_id(document: &NodeRef, name: &str) -> String {
    document
        .root()
        .unparsed_entity(name)
        .and_then(|entity| entity.public_id)
        .unwrap_or_default()
}

/// The document whose unparsed entities an accessor reads: the node the two-argument form
/// names, or the context node when there is no second argument at all.
///
/// [XSLT 3.0 §20.4](https://www.w3.org/TR/xslt-30/#unparsed-entity-uri) makes it `XTDE1370`
/// (`XTDE1380` for the public identifier) if that node's tree is not rooted at a document node —
/// the same condition for both forms, since only a document node carries the
/// `[unparsed entities]` property at all. It also declares the second argument `node()`, not
/// `node()?`: a second argument that *is* written and yields nothing is a type error, never a
/// silent fall-back to the context node, which would answer from a different document.
///
/// # Errors
///
/// `code` when there is no node to read from, or when its root is not a document node;
/// `XPTY0004` when a second argument was written but produced no node.
fn entity_document(argument: Option<&Sequence>, code: &str) -> FnResult<NodeRef> {
    let node = match argument {
        Some(sequence) => sequence
            .first()
            .and_then(|item| item.as_node().cloned())
            .ok_or_else(|| {
                FnError::new(
                    FnErrorCode::Xpty0004,
                    "the second argument of an unparsed-entity accessor must be a node",
                )
            })?,
        // The context node is the one in focus where the *call* is written, which is not
        // always the transformation's current node: a named function reference such as
        // `unparsed-entity-uri#1` keeps the focus of the expression that created it, so
        // invoking it from inside `xsl:for-each select="'hatch-pic'"` still reads the document
        // the template matched (the suite's unparsed-entity-09/-10). The transformation's
        // current item is the fall-back for a call the evaluator is not driving.
        None => calling_context_node()
            .or_else(|| current_runtime().and_then(|runtime| runtime.current_item()))
            .and_then(|item| item.as_node().cloned())
            .ok_or_else(|| xslt_error(code, "there is no context node"))?,
    };
    if node.root().node_kind() != oxixml_infoset::NodeKind::Document {
        return Err(xslt_error(
            code,
            "the root of the tree containing the node is not a document node",
        ));
    }
    Ok(node)
}

/// The base URI a node supplies for resolving a relative reference: its own, or the static
/// base URI when the node is part of a tree that has one.
fn base_uri_of(node: &NodeRef, static_base: Option<&str>) -> Option<String> {
    if let Some(uri) = node.base_uri() {
        return Some(uri);
    }
    // A parentless node that is not a document node — a text node a stylesheet built, say —
    // belongs to no tree with a URI, so there is nothing to resolve against.
    if node.parent().is_none() && node.node_kind() != oxixml_infoset::NodeKind::Document {
        return None;
    }
    static_base.map(str::to_owned)
}

/// Whether a URI reference is absolute — it has a scheme, so no base URI is needed.
fn is_absolute_uri(reference: &str) -> bool {
    match reference.split_once(':') {
        Some((scheme, _)) => {
            !scheme.is_empty()
                && scheme.starts_with(|c: char| c.is_ascii_alphabetic())
                && scheme
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'))
        }
        None => false,
    }
}

/// The schema namespace, whose type names double as constructor functions.
const XSD_NAMESPACE: &str = "http://www.w3.org/2001/XMLSchema";

/// A name written the three ways `function-available()` and its siblings accept.
enum Name {
    /// `Q{uri}local`, whose namespace is written out.
    Expanded(String, String),
    /// `prefix:local`, whose namespace the static context of the call resolves.
    Prefixed(String, String),
    /// `local`, whose namespace each of the three functions defaults differently.
    Unprefixed(String),
}

/// Whether `local` names an XSLT element this processor implements.
///
/// [XSLT 3.0 §23.2.2](https://www.w3.org/TR/xslt-30/#func-element-available) asks about an
/// *element*, not only an instruction: the function "returns true if and only if the local name
/// matches the name of an XSLT element that is defined in this specification and implemented by
/// the XSLT processor". So the declarations count as well as the instructions —
/// `element-available('xsl:key')` is true, which is what `function-0302b` and `function-0303`
/// assert — while the elements this processor declines (`xsl:package` and the rest of the
/// packaging vocabulary, accumulators, streaming, `xsl:evaluate`) are not implemented and answer
/// false. [`crate::names::is_available_instruction`] stays the narrower predicate the
/// `xsl:fallback` rule of §23.2.3 needs, which is about instructions alone.
fn is_available_xslt_element(local: &str) -> bool {
    is_available_instruction(local)
        || matches!(
            local,
            // The declarations, and the elements that only appear inside one.
            "attribute-set"
                | "character-map"
                | "context-item"
                | "decimal-format"
                | "function"
                | "global-context-item"
                | "import"
                | "import-schema"
                | "include"
                | "key"
                | "mode"
                | "namespace-alias"
                | "output"
                | "output-character"
                | "param"
                | "preserve-space"
                | "strip-space"
                | "stylesheet"
                | "template"
                | "transform"
                // The children an instruction or a declaration takes.
                | "matching-substring"
                | "non-matching-substring"
                | "on-completion"
                | "otherwise"
                | "sort"
                | "when"
                | "with-param"
        )
}

/// Which default namespace an unprefixed name takes.
///
/// The three availability functions differ here, and the difference is what
/// `function-available-1006`, `function-0302b` and `function-0303` are about. XSLT 3.0
/// [§23.1.1](https://www.w3.org/TR/xslt-30/#func-function-available): "If the value is an
/// unprefixed lexical QName, then the standard function namespace is used";
/// [§23.2.2](https://www.w3.org/TR/xslt-30/#func-element-available) and
/// [§23.1.4](https://www.w3.org/TR/xslt-30/#func-type-available): "If there is a default
/// namespace in scope, then it is used to expand an unprefixed lexical QName."
#[derive(Clone, Copy, PartialEq, Eq)]
enum UnprefixedIn {
    /// The standard function namespace — `function-available`.
    FunctionNamespace,
    /// The default namespace declaration in scope where the call was written —
    /// `element-available`, `type-available`. [XSLT 3.0
    /// §5.1](https://www.w3.org/TR/xslt-30/#unprefixed-qnames) means by this the `xmlns="…"` of
    /// the defining element, not `[xsl:]xpath-default-namespace`: it lists the argument of
    /// `element-available` among the places where "the default namespace of the defining element
    /// is used", and leaves `xpath-default-namespace` to name tests inside XPath expressions.
    DefaultNamespace,
}

/// What expanding a lexical name against the static context of the call produced.
enum Expanded {
    /// The name, with the namespace it denotes — `None` for a name in no namespace, which is an
    /// unprefixed name with no default namespace in scope or the explicit `Q{}local` spelling.
    Resolved(Option<String>, String),
    /// The prefix could not be resolved because no static context was published: the call was
    /// made from a `use-when` expression, which the compiler evaluates before there is a
    /// transformation, or from a pattern predicate, which the matcher evaluates itself. The
    /// local name is all there is to go on, so the caller answers as it did before the static
    /// context was available at all.
    ByLocalName(String),
}

/// Expands a lexical name the way the availability functions do: against the namespace
/// declarations in scope *for the expression the call is written in*.
///
/// (A prefix declared nowhere in the stylesheet is rejected earlier, by `check_lexical_name`,
/// under the error code each function names.)
fn expand_availability_name(name: &Name, default: UnprefixedIn) -> Expanded {
    let static_context = calling_static_context();
    match name {
        Name::Expanded(namespace, local) => Expanded::Resolved(
            (!namespace.is_empty()).then(|| namespace.clone()),
            local.clone(),
        ),
        Name::Prefixed(prefix, local) => match &static_context {
            None => Expanded::ByLocalName(local.clone()),
            Some(context) => Expanded::Resolved(
                context
                    .namespace(prefix)
                    .map(str::to_owned)
                    .filter(|namespace| !namespace.is_empty()),
                local.clone(),
            ),
        },
        Name::Unprefixed(local) => {
            let namespace = match default {
                UnprefixedIn::FunctionNamespace => Some(FN_NAMESPACE.to_owned()),
                UnprefixedIn::DefaultNamespace => static_context
                    .and_then(|context| {
                        context
                            .namespace(crate::names::DEFAULT_NAMESPACE_KEY)
                            .map(str::to_owned)
                    })
                    .filter(|namespace| !namespace.is_empty()),
            };
            Expanded::Resolved(namespace, local.clone())
        }
    }
}

/// Splits a name the availability functions were handed.
fn split_eqname(lexical: &str) -> Name {
    let trimmed = lexical.trim();
    if let Some(rest) = trimmed.strip_prefix("Q{")
        && let Some((namespace, local)) = rest.split_once('}')
    {
        return Name::Expanded(namespace.to_owned(), local.trim().to_owned());
    }
    match trimmed.split_once(':') {
        Some((prefix, local)) => Name::Prefixed(prefix.to_owned(), local.to_owned()),
        None => Name::Unprefixed(trimmed.to_owned()),
    }
}

/// Whether `local` names a type the schema namespace defines that this processor makes
/// available without a schema: the XSD 1.0 built-in types and the four type names above them.
///
/// `xs:dateTimeStamp` is deliberately absent. It is an XSD 1.1 type derived by restriction, so
/// [XSLT 3.0 §18.1.2](https://www.w3.org/TR/xslt-30/#type-available) makes it available only to
/// a schema-aware processor — which this is not (`xsl:supports-schema-aware` is `no`).
fn is_schema_type(local: &str) -> bool {
    if local == "dateTimeStamp" {
        return false;
    }
    oxixml_infoset::AtomicType::from_local_name(local).is_some()
        || matches!(
            local,
            // The four type names above the atomic types, and the three built-in *list* types,
            // which are in-scope schema types like any other and which the evaluator constructs.
            "anyType"
                | "untyped"
                | "anySimpleType"
                | "anyAtomicType"
                | "IDREFS"
                | "NMTOKENS"
                | "ENTITIES"
        )
}

/// Whether `local` names a schema type that has a constructor function.
///
/// [XPath 3.1 §18.4](https://www.w3.org/TR/xpath-31/#id-constructor-functions) gives every
/// built-in *concrete* atomic type a one-argument constructor, and the three built-in list types
/// one as well; the abstract ones — `xs:anyAtomicType`, `xs:anySimpleType`, `xs:anyType`,
/// `xs:untyped` and `xs:NOTATION` — have none, which is what the suite's function-available-1016
/// asks about.
///
/// `xs:error` is the one name that belongs here and not in [`is_schema_type`].
/// [F&O 3.1 §18.4](https://www.w3.org/TR/xpath-functions-31/#func-xs-error) defines the
/// constructor — it accepts the empty sequence and raises `FORG0001` for anything else, since
/// the type's value space is empty — and the evaluator implements it; but the type itself is an
/// XSD 1.1 addition, and this processor reports `xsl:xsd-version` as `1.0`, so
/// `type-available('xs:error')` must keep answering `false`.
fn has_constructor_function(local: &str) -> bool {
    if matches!(
        local,
        "anyAtomicType" | "anySimpleType" | "anyType" | "untyped" | "NOTATION"
    ) {
        return false;
    }
    local == "error" || is_schema_type(local)
}

/// Whether `function-available()` should say yes.
///
/// [XSLT 3.0 §18.1.1](https://www.w3.org/TR/xslt-30/#function-available) asks about the
/// *function*, name and arity together: a stylesheet function the stylesheet declares, a
/// constructor function for a schema type, or an entry of the function library.
fn function_available(lexical: &str, arity: Option<usize>, library: Library) -> bool {
    // XSLT 3.0 §23.1.1: "The lexical QName is expanded into an expanded QName using the
    // namespace declarations in scope for the expression. If the value is an unprefixed lexical
    // QName, then the standard function namespace is used." A prefix bound to something other
    // than the namespace the local name belongs to therefore answers `false`, which is what
    // `function-available-1006` asks — its `fn` prefix is bound to the obsolete 2003/05
    // namespace, where `implicit-timezone` does not live.
    let (namespace, local, resolved) =
        match expand_availability_name(&split_eqname(lexical), UnprefixedIn::FunctionNamespace) {
            Expanded::Resolved(namespace, local) => (namespace, local, true),
            Expanded::ByLocalName(local) => (None, local, false),
        };
    let namespace = namespace.as_deref();
    // §9.7: "stylesheet functions are not included in the context, which means that the function
    // `function-available` will return false in respect of such functions" when it is asked from
    // inside a static expression.
    if let Some(runtime) = current_runtime().filter(|_| library == Library::Whole) {
        let declared = runtime
            .ir
            .functions
            .keys()
            .any(|(declared, declared_arity)| {
                (declared.namespace() == namespace || !resolved)
                    && declared.local_name() == local
                    && arity.is_none_or(|wanted| wanted == *declared_arity)
            });
        if declared {
            return true;
        }
    }
    // XSLT 3.0 §23.1.1: the XSLT namespace names no function this processor makes available.
    if namespace == Some(crate::names::XSLT_NAMESPACE) {
        return false;
    }
    let local = local.as_str();
    // A schema type's constructor function takes exactly one argument.
    if arity.is_none_or(|wanted| wanted == 1)
        && (!resolved || namespace == Some(XSD_NAMESPACE))
        && has_constructor_function(local)
    {
        return true;
    }
    // `fn:concat` is the one variadic function in the library, available at every arity of two
    // upwards ([F&O 3.1 §5.4.1](https://www.w3.org/TR/xpath-functions-31/#func-concat)).
    if (!resolved || namespace == Some(FN_NAMESPACE)) && local == "concat" {
        return arity.is_none_or(|wanted| wanted >= 2);
    }
    let whole;
    let library: &FunctionRegistry = match library {
        Library::Whole => {
            whole = registry();
            &whole
        }
        Library::Static => static_registry(),
    };
    let candidates: Vec<&str> = match (resolved, namespace) {
        // A name that resolves into no namespace at all denotes no function: every function
        // this processor serves is in one.
        (true, None) => return false,
        (true, Some(XSD_NAMESPACE)) => return false,
        (true, Some(namespace)) => vec![namespace],
        // With no static context published — a `use-when` expression, evaluated before there is
        // a transformation, or a pattern predicate — the prefix cannot be resolved, so every
        // namespace the library uses is a candidate, as it was before the static context was
        // available at all.
        (false, _) => vec![
            FN_NAMESPACE,
            oxixml_fn::maps::MAP_NAMESPACE,
            oxixml_fn::arrays::ARRAY_NAMESPACE,
            MATH_NAMESPACE,
        ],
    };
    candidates.iter().any(|namespace| {
        let name = ExpandedName::new(Some(*namespace), local);
        match arity {
            // The higher-order functions — `map:for-each`, the `array:` set, `fn:for-each`,
            // `fn:filter`, `fn:fold-left`, `fn:sort`, `fn:apply`, the `math:` set — are served
            // by the XPath evaluator rather than by an entry of the function library, because a
            // library entry receives its arguments as sequences and could not be handed a
            // function item. They are functions all the same, and `fn:function-available` says
            // so: `function-available-1017` and `-1018` ask about them one by one, and
            // `function-2001` walks the whole XSLT 3.0 function list.
            Some(arity) => {
                library.contains(&name, arity) || oxixml_xpath::supports_function(&name, arity)
            }
            None => (0..=8).any(|arity| {
                library.contains(&name, arity) || oxixml_xpath::supports_function(&name, arity)
            }),
        }
    })
}

/// The namespace of the `math:` functions.
const MATH_NAMESPACE: &str = "http://www.w3.org/2005/xpath-functions/math";

fn register_grouping(registry: &mut FunctionRegistry) {
    registry.add(FN_NAMESPACE, "current-group", 0, |_, _| {
        let runtime =
            current_runtime().ok_or_else(|| outside_transformation("fn:current-group"))?;
        // XSLT 3.0 §14.5: outside `xsl:for-each-group` there is no current group at all,
        // which is an error rather than an empty sequence.
        match runtime.current_group().filter(|_| !captured()) {
            Some(group) => Ok(Sequence::from_items(group)),
            None => Err(xslt_error("XTDE1061", "there is no current group")),
        }
    });

    registry.add(FN_NAMESPACE, "current-grouping-key", 0, |_, _| {
        let runtime =
            current_runtime().ok_or_else(|| outside_transformation("fn:current-grouping-key"))?;
        match runtime.current_grouping_key().filter(|_| !captured()) {
            Some(key) => Ok(key),
            None => Err(xslt_error("XTDE1071", "there is no current grouping key")),
        }
    });

    registry.add(FN_NAMESPACE, "regex-group", 1, |_, arguments| {
        let runtime = current_runtime().ok_or_else(|| outside_transformation("fn:regex-group"))?;
        let index = arguments
            .first()
            .and_then(Sequence::first)
            .and_then(|item| item.as_atomic().and_then(AtomicValue::as_f64))
            .unwrap_or(0.0)
            .max(0.0) as usize;
        // XSLT 3.0 §5.4.4: a named function reference retains the XPath static and dynamic
        // context of the point it was written at, and "this rule does not extend to the XSLT
        // extensions to the dynamic context … if a dynamic function call is made that depends
        // on the XSLT part of the dynamic context (for example, `regex-group#1(2)`), then the
        // relevant components of the context are cleared". With the captured substrings cleared
        // there is no group to report, and §18.5 makes that a zero-length string rather than an
        // error — `regex-090` and `regex-091` hold a `regex-group#1` from one
        // `xsl:analyze-string` and call it inside another, and assert that nothing comes out.
        Ok(Sequence::from(AtomicValue::string(
            runtime
                .regex_group(index)
                .filter(|_| !captured())
                .unwrap_or_default(),
        )))
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_xslt_additions_are_registered() {
        let library = registry();
        for (local, arity) in [
            ("current", 0),
            ("current-output-uri", 0),
            ("key", 2),
            ("key", 3),
            ("document", 1),
            ("system-property", 1),
            ("available-system-properties", 0),
            ("element-available", 1),
            ("function-available", 1),
            ("current-group", 0),
            ("current-grouping-key", 0),
            ("regex-group", 1),
            ("copy-of", 0),
            ("snapshot", 0),
        ] {
            assert!(
                library.contains(&ExpandedName::new(Some(FN_NAMESPACE), local), arity),
                "fn:{local}#{arity} should be registered"
            );
        }
    }

    #[test]
    fn availability_reports_the_library() {
        assert!(function_available("substring", Some(2), Library::Whole));
        assert!(!function_available("nonesuch", None, Library::Whole));
        assert!(!function_available("xsl:nonesuch", None, Library::Whole));
    }

    /// XSLT 3.0 §9.7 gives a static expression F&O plus four availability functions and nothing
    /// else, and `fn:function-available` called from inside one reports on that library — which
    /// is what `use-when-0407a` asks about, function by function.
    #[test]
    fn a_static_expression_sees_a_smaller_library() {
        for (local, whole, restricted) in [
            ("substring", true, true),
            // F&O 3.0 defines `generate-id`, so a static expression may call it.
            ("generate-id", true, true),
            ("system-property", true, true),
            ("function-available", true, true),
            // These need a transformation, and are outside §9.7's list.
            ("current", true, false),
            ("key", true, false),
            ("unparsed-entity-uri", true, false),
            ("unparsed-entity-public-id", true, false),
            ("document", true, false),
            ("available-system-properties", true, false),
        ] {
            assert_eq!(
                function_available(local, None, Library::Whole),
                whole,
                "fn:{local} in the whole library"
            );
            assert_eq!(
                function_available(local, None, Library::Static),
                restricted,
                "fn:{local} in a static expression"
            );
        }
    }

    #[test]
    fn available_system_properties_agree_with_system_property() {
        let library = registry();
        let context = oxixml_fn::FnContext::new();

        let available = library
            .lookup(
                &ExpandedName::new(Some(FN_NAMESPACE), "available-system-properties"),
                0,
            )
            .expect("fn:available-system-properties is registered");
        let names = available
            .call(&context, &[])
            .expect("it runs without a transformation");
        assert_eq!(names.len(), 14);

        let system_property = library
            .lookup(&ExpandedName::new(Some(FN_NAMESPACE), "system-property"), 1)
            .expect("fn:system-property is registered");
        for item in names.iter() {
            let qname = item
                .as_atomic()
                .and_then(AtomicValue::as_qname)
                .expect("every result item is an xs:QName");
            assert_eq!(qname.namespace(), Some(crate::names::XSLT_NAMESPACE));
            // Each name reported as available must return a non-empty string from
            // system-property, spelled with its `xsl:` prefix — the two functions agreeing.
            let lexical = format!("xsl:{}", qname.local_name());
            let reported = system_property
                .call(&context, &[Sequence::from(AtomicValue::string(lexical))])
                .expect("system-property runs");
            let reported = reported
                .first()
                .and_then(Item::as_atomic)
                .map(AtomicValue::string_value)
                .expect("a single string");
            assert!(
                !reported.is_empty(),
                "system-property('xsl:{}') should be non-empty",
                qname.local_name()
            );
        }
    }

    #[test]
    fn the_functions_refuse_to_run_outside_a_transformation() {
        let library = registry();
        let current = library
            .lookup(&ExpandedName::new(Some(FN_NAMESPACE), "current"), 0)
            .expect("fn:current is registered");
        let context = oxixml_fn::FnContext::new();
        assert!(current.call(&context, &[]).is_err());
    }
}
