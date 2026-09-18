//! [`XQuery`] and [`CompiledQuery`] — the public entry points, and the module and prolog wiring
//! behind them.

use crate::compile::{
    Compiler, XQUERY_NAMESPACE, is_reserved_annotation, is_reserved_function_namespace,
    is_reserved_prolog_binding, resolve_name, xpath_static_context,
};
use crate::context::{DynamicQueryContext, StaticQueryContext};
use crate::error::{XQueryError, XQueryErrorCode, XQueryResult};
use crate::eval::program::{Program, TypeCheck, UserFunction};
use crate::eval::{
    Frame, RuntimeOptions, apply_type_check, call_function, declare_stack, evaluate,
};
use crate::parser::Parser;
use crate::prolog::{FunctionDeclaration, Module, VariableDeclaration};
use oxixml_fn::numeric::DecimalFormat;
use oxixml_fn::xmlio::SerializationParameters;
use oxixml_fn::{FnError, FunctionRegistry, builtins};
use oxixml_infoset::{ExpandedName, Sequence};
use oxixml_xpath::{DynamicContext, StaticContext, XPathErrorCode, XpathItem, XpathValue};
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, OnceLock, Weak};

/// The compiler.
///
/// # Examples
///
/// ```
/// use oxixml_infoset::Tree;
/// use oxixml_xquery::{DynamicQueryContext, StaticQueryContext, XQuery};
///
/// let compiled = XQuery::compile(
///     "for $b in //book order by $b/@year descending return <t>{ string($b/title) }</t>",
///     StaticQueryContext::new(),
/// )?;
///
/// let tree = Tree::parse_str(
///     "<lib><book year='1815'><title>Emma</title></book>\
///      <book year='1965'><title>Dune</title></book></lib>",
/// )?;
/// let context = DynamicQueryContext::new().with_context_item(tree.root());
/// let result = compiled.execute(&context)?;
/// assert_eq!(result.len(), 2);
/// # Ok::<(), oxixml_xquery::XQueryError>(())
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct XQuery;

impl XQuery {
    /// Compiles `query` against `static_context`.
    ///
    /// # Errors
    ///
    /// Any static error: `XPST0003` for a syntax error, `XPST0081` for an unbound prefix,
    /// `XQST0033`, `XQST0034`, `XQST0049` and friends for a malformed prolog, `XQST0059` when an
    /// `import module` cannot be resolved, and `XQST0075` for a `validate` expression.
    pub fn compile(query: &str, static_context: StaticQueryContext) -> XQueryResult<CompiledQuery> {
        compile_query(query, static_context)
    }

    /// Compiles `query` against the default static context.
    ///
    /// # Errors
    ///
    /// As [`XQuery::compile`].
    pub fn compile_default(query: &str) -> XQueryResult<CompiledQuery> {
        compile_query(query, StaticQueryContext::new())
    }
}

/// One variable a prolog declares.
#[derive(Debug)]
struct PrologVariable {
    name: ExpandedName,
    check: Option<TypeCheck>,
    value: Option<Program>,
    external: bool,
    /// The prolog-settable components of the module the variable was declared in, so that its
    /// initializer runs under that module's base URI, construction mode and boundary space.
    options: Arc<RuntimeOptions>,
}

/// The declared context item, when a prolog declares one.
#[derive(Debug)]
struct ContextItemSpec {
    check: Option<TypeCheck>,
    value: Option<Program>,
    external: bool,
}

/// The user-declared functions, shared with the function registry the XPath compiler resolves
/// calls against.
///
/// The registry holds only a [`Weak`] reference, so that the reference cycle
/// *compiled query → static context → registry → function → compiled body → static context* is
/// broken and a dropped query frees everything it owns.
#[derive(Debug, Default)]
pub(crate) struct FunctionStore {
    functions: OnceLock<HashMap<(ExpandedName, usize), Arc<UserFunction>>>,
}

impl FunctionStore {
    fn get(&self, name: &ExpandedName, arity: usize) -> Option<Arc<UserFunction>> {
        self.functions
            .get()?
            .get(&(name.clone(), arity))
            .map(Arc::clone)
    }
}

thread_local! {
    /// The dynamic context a user-defined function called through the function registry runs in.
    static AMBIENT: RefCell<Vec<DynamicContext>> = const { RefCell::new(Vec::new()) };
}

struct AmbientGuard;

impl Drop for AmbientGuard {
    fn drop(&mut self) {
        AMBIENT.with(|stack| {
            stack.borrow_mut().pop();
        });
    }
}

fn push_ambient(context: DynamicContext) -> AmbientGuard {
    AMBIENT.with(|stack| stack.borrow_mut().push(context));
    AmbientGuard
}

fn ambient() -> Option<DynamicContext> {
    AMBIENT.with(|stack| stack.borrow().last().cloned())
}

/// A compiled query, ready to execute any number of times.
///
/// The type is `Send + Sync` and everything mutable lives in the [`DynamicQueryContext`] passed
/// to each execution.
#[derive(Debug)]
pub struct CompiledQuery {
    source: String,
    statik: Arc<StaticContext>,
    options: Arc<RuntimeOptions>,
    variables: Vec<PrologVariable>,
    context_item: Option<ContextItemSpec>,
    body: Program,
    /// The collations the application installed on the static context, which have to reach the
    /// evaluator too — `declare default collation` names one of them by URI.
    collations: Vec<Arc<dyn oxixml_fn::Collation>>,
    /// The context item types the *imported library modules* declared. XQuery 3.1 §4.15 makes
    /// the context item a single component of the dynamic context that every module in the
    /// program shares, so its value has to match what each of them declared, not only what the
    /// main module did.
    imported_context_item_types: Vec<TypeCheck>,
    /// The main module's output declarations, as `(parameter name, declared value)` pairs in
    /// declaration order, and the prefix map they are read against.
    output: OutputDeclarations,
    /// Keeps the user-declared functions alive; the registry only holds a weak reference.
    #[allow(dead_code)]
    functions: Arc<FunctionStore>,
}

/// What a main module's `declare option output:…` declarations amount to.
#[derive(Debug, Default)]
struct OutputDeclarations {
    declarations: Vec<(String, String)>,
    /// The module's in-scope prefix map, which a QName-valued parameter resolves against.
    namespaces: BTreeMap<String, String>,
    /// The module's `declare default element namespace`, which an *unprefixed* name in a
    /// QName-valued parameter denotes.
    default_element_namespace: Option<String>,
}

impl CompiledQuery {
    /// The source text this query was compiled from.
    #[must_use]
    pub fn query(&self) -> &str {
        &self.source
    }

    /// The main module's *output declarations*, as `(parameter name, declared value)` pairs in
    /// declaration order.
    ///
    /// [XQuery 3.1 §2.2.4](https://www.w3.org/TR/xquery-31/#id-serialization-declarations) writes
    /// one as `declare option output:name "value"` with `output` bound to the serialization
    /// namespace; the parameter name here is the option's *local* name, its prefix already
    /// resolved. An empty slice means the prolog declared none, which is not the same as
    /// declaring the defaults — a host that serializes only when the query asks it to can tell
    /// the two apart.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_xquery::{OUTPUT_NAMESPACE, XQuery};
    ///
    /// let compiled = XQuery::compile_default(&format!(
    ///     "declare namespace output = \"{OUTPUT_NAMESPACE}\";
    ///      declare option output:method \"html\"; <a/>"
    /// ))?;
    /// assert_eq!(
    ///     compiled.output_declarations(),
    ///     [("method".to_owned(), "html".to_owned())]
    /// );
    /// # Ok::<(), oxixml_xquery::XQueryError>(())
    /// ```
    #[must_use]
    pub fn output_declarations(&self) -> &[(String, String)] {
        &self.output.declarations
    }

    /// The serialization parameters the main module's output declarations set, applied over
    /// [`SerializationParameters::default()`](oxixml_fn::xmlio::SerializationParameters).
    ///
    /// The engine never serializes a result itself, so nothing here is applied to an execution;
    /// this is what a host that *does* serialize needs in order to honour
    /// [XQuery 3.1 §2.2.4](https://www.w3.org/TR/xquery-31/#id-serialization-declarations).
    ///
    /// The parameters a declaration does not name keep their default value, and that default is
    /// `fn:serialize`'s parameter set rather than Serialization 3.1 §3's: the two agree except
    /// that `omit-xml-declaration` starts at `yes` here and at `no` there. A host serializing a
    /// whole query result to a document rather than to a `fn:serialize` value therefore has to
    /// set it itself unless the prolog does. A prolog with no output declaration at all yields
    /// the defaults unchanged — use [`CompiledQuery::output_declarations`] to tell that case
    /// apart from a prolog that declared them explicitly.
    ///
    /// A declared value is only checked when it is asked for, so a query whose declaration is
    /// inadmissible still compiles: `XQST0109` and `XQST0110` are what the *prolog grammar*
    /// requires, and the admissible values of a parameter belong to serialization, which happens
    /// later or not at all.
    ///
    /// # Errors
    ///
    /// `SEPM0017` when a declared value is not admissible for its parameter — the code
    /// Serialization 3.1 §3.1 gives a fault inside an `output:serialization-parameters` element,
    /// which is the route the values are checked by.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_fn::xmlio::OutputMethod;
    /// use oxixml_xquery::{OUTPUT_NAMESPACE, XQuery};
    ///
    /// let compiled = XQuery::compile_default(&format!(
    ///     "declare namespace output = \"{OUTPUT_NAMESPACE}\";
    ///      declare option output:method \"html\";
    ///      declare option output:indent \"yes\"; <a/>"
    /// ))?;
    /// let parameters = compiled.serialization_parameters()?;
    /// assert_eq!(parameters.method(), OutputMethod::Html);
    /// assert!(parameters.indent());
    /// # Ok::<(), oxixml_xquery::XQueryError>(())
    /// ```
    pub fn serialization_parameters(&self) -> XQueryResult<SerializationParameters> {
        crate::serialization::resolve(
            &self.output.declarations,
            &self.output.namespaces,
            self.output.default_element_namespace.as_deref(),
        )
    }

    /// Executes the query and returns its result as a sequence.
    ///
    /// # Errors
    ///
    /// Any dynamic error, plus `XPTY0004` when the result holds a function item, which the data
    /// model cannot represent — use [`CompiledQuery::execute_value`] for those.
    pub fn execute(&self, context: &DynamicQueryContext) -> XQueryResult<Sequence> {
        Ok(self.execute_value(context)?.into_sequence()?)
    }

    /// Executes the query, keeping function items in the result.
    ///
    /// # Errors
    ///
    /// Any dynamic error.
    pub fn execute_value(&self, context: &DynamicQueryContext) -> XQueryResult<XpathValue> {
        let prepared = self.prepare(context)?;
        let _stack = declare_stack(context.stack_size);
        let guard = push_ambient(prepared.clone());
        let frame = Frame::new(Arc::clone(&self.options), prepared);
        let result = evaluate(&self.body, &frame);
        drop(guard);
        result
    }

    /// `fn:string` of every item of the result, joined with single spaces.
    ///
    /// # Errors
    ///
    /// As [`CompiledQuery::execute_value`].
    pub fn execute_to_string(&self, context: &DynamicQueryContext) -> XQueryResult<String> {
        let value = self.execute_value(context)?;
        crate::eval::join_string_values(&value)
    }

    /// The effective boolean value of the result.
    ///
    /// # Errors
    ///
    /// As [`CompiledQuery::execute_value`], plus `FORG0006` when the result has no effective
    /// boolean value.
    pub fn execute_to_boolean(&self, context: &DynamicQueryContext) -> XQueryResult<bool> {
        Ok(self.execute_value(context)?.effective_boolean_value()?)
    }

    /// Builds the dynamic context the body is evaluated in: external variables, the context item
    /// and every prolog variable, in dependency order.
    fn prepare(&self, context: &DynamicQueryContext) -> XQueryResult<DynamicContext> {
        let mut dynamic = DynamicContext::new(Arc::clone(&self.statik))
            .with_version(oxixml_xpath::XPathVersion::XPath31);
        // The static context's collations come first: `declare default collation` was accepted
        // against them, so the default the function context resolves has to find them.
        for collation in &self.collations {
            dynamic = dynamic.with_collation(Arc::clone(collation));
        }
        if let Some(fn_context) = &context.fn_context {
            // A caller-supplied function context replaces the one `DynamicContext::new` derived
            // from the static context, so the two components that came from the *query* — the
            // static base URI and the default collation — are put back unless the caller set
            // them itself. Otherwise installing a trace sink would silently lose
            // `fn:static-base-uri`.
            let mut merged = fn_context.clone();
            for collation in self.collations.iter().chain(context.collations.iter()) {
                merged = merged.with_collation(Arc::clone(collation));
            }
            if merged.static_base_uri().is_none() {
                if let Some(base) = &self.options.base_uri {
                    merged = merged.with_static_base_uri(base.clone());
                }
            }
            if let Ok(updated) = merged
                .clone()
                .with_default_collation(&self.options.default_collation)
            {
                merged = updated;
            }
            dynamic = dynamic.with_fn_context(merged);
        }
        if let Some(current) = &context.current_date_time {
            dynamic = dynamic.with_current_date_time(*current);
        }
        if let Some(timezone) = context.implicit_timezone {
            dynamic = dynamic.with_implicit_timezone(timezone);
        }
        for collation in &context.collations {
            dynamic = dynamic.with_collation(Arc::clone(collation));
        }
        if let Some(resolver) = &context.resolver {
            dynamic = dynamic.with_document_resolver(Arc::clone(resolver));
        }
        // The *main* module's declarations, and the caller's: a library module's own are
        // installed when its code runs, by `crate::eval::rebase`.
        if !self.options.decimal_formats.is_empty() || !context.decimal_formats.is_empty() {
            let mut fn_context = dynamic.fn_context().clone();
            for (name, format) in self
                .options
                .decimal_formats
                .iter()
                .chain(context.decimal_formats.iter())
            {
                fn_context = match name {
                    Some(name) => fn_context.with_decimal_format(name.clone(), format.clone()),
                    None => fn_context.with_default_decimal_format(format.clone()),
                };
            }
            dynamic = dynamic.with_fn_context(fn_context);
        }
        for (name, value) in &context.variables {
            dynamic = dynamic.with_variable_name(name.clone(), value.clone());
        }

        // The context item and the prolog's variables may refer to one another in any order, so
        // they are resolved together: whatever raises `XPST0008` is retried on the next pass, and
        // a pass that makes no progress means the definitions are circular.
        let mut focus: Option<XpathItem> = context.context_item.clone();
        let mut context_item_pending = self.context_item.is_some();
        let mut pending: Vec<&PrologVariable> = self.variables.iter().collect();
        while !pending.is_empty() || context_item_pending {
            let mut deferred = Vec::new();
            let mut progress = false;
            if context_item_pending {
                if let Some(declaration) = &self.context_item {
                    let _guard = push_ambient(dynamic.clone());
                    let frame = Frame::new(Arc::clone(&self.options), dynamic.clone());
                    let supplied = focus.clone().map(XpathValue::singleton);
                    let raw = match (&declaration.value, supplied) {
                        (_, Some(value)) if declaration.external => Ok(value),
                        (Some(program), _) => evaluate(program, &frame),
                        (None, Some(value)) => Ok(value),
                        (None, None) => Err(XQueryError::new(
                            XQueryErrorCode::Xpdy0002,
                            "the declared context item has no value",
                        )),
                    };
                    match raw {
                        Ok(mut value) => {
                            if let Some(check) = &declaration.check {
                                value = apply_type_check(check, value, &frame)?;
                            }
                            focus = value.first().cloned();
                            if let Some(item) = focus.clone() {
                                dynamic = dynamic.with_context_item(item);
                            }
                            context_item_pending = false;
                            progress = true;
                        }
                        Err(error) if error.name().local_name() == "XPST0008" => {}
                        Err(error) => return Err(error),
                    }
                }
            } else if let Some(item) = focus.clone() {
                // Cloned rather than taken: the imported modules' context item declarations are
                // checked against this same focus once the loop is done, and a main module that
                // declares no context item of its own must not empty it on the way past.
                dynamic = dynamic.with_context_item(item);
            }
            for variable in pending {
                let supplied = context
                    .variables
                    .iter()
                    .find(|(name, _)| name == &variable.name)
                    .map(|(_, value)| value.clone());
                if variable.external && supplied.is_some() && variable.check.is_none() {
                    progress = true;
                    continue;
                }
                let _guard = push_ambient(dynamic.clone());
                let frame = Frame::new(
                    Arc::clone(&variable.options),
                    crate::eval::rebase(&dynamic, &variable.options),
                );
                let raw = match (supplied, &variable.value) {
                    (Some(value), _) => Ok(value),
                    (None, Some(program)) => evaluate(program, &frame),
                    (None, None) => Err(XQueryError::new(
                        XQueryErrorCode::Xpdy0002,
                        format!(
                            "no value was supplied for the external variable `${}`",
                            variable.name
                        ),
                    )),
                };
                match raw {
                    Ok(mut value) => {
                        if let Some(check) = &variable.check {
                            value = apply_type_check(check, value, &frame)?;
                        }
                        dynamic = dynamic.with_variable_name(variable.name.clone(), value);
                        progress = true;
                    }
                    // A variable whose initializer needs a focus — `fn:position()`, `fn:last()`,
                    // a path expression — is retried once the declared context item is known.
                    // `declare context item` may itself stand *after* the variables it is built
                    // from (`prod/ContextItemDecl.xml`'s `contextDecl-016`: `$y`, then the
                    // context item `$y[3]`, then `$x := fn:position()`), so on the first pass
                    // there is no focus yet and the order the declarations were written in is
                    // recovered by deferring rather than by failing.
                    Err(error)
                        if error.name().local_name() == "XPST0008"
                            || (context_item_pending
                                && error.name().local_name() == "XPDY0002") =>
                    {
                        deferred.push(variable);
                    }
                    Err(error) => return Err(error),
                }
            }
            if deferred.is_empty() && !context_item_pending {
                break;
            }
            if !progress {
                return Err(XQueryError::new(
                    XQueryErrorCode::Xqdy0054,
                    "a cycle was found in the prolog's variable initializers",
                ));
            }
            pending = deferred;
        }
        // Every imported module's context item declaration constrains the one context item the
        // program has, whether the main module declared one or not.
        if !self.imported_context_item_types.is_empty() {
            let frame = Frame::new(Arc::clone(&self.options), dynamic.clone());
            let supplied = focus
                .clone()
                .map_or_else(XpathValue::default, XpathValue::singleton);
            for check in &self.imported_context_item_types {
                apply_type_check(check, supplied.clone(), &frame)?;
            }
        }
        if let Some(item) = focus {
            dynamic = dynamic.with_context_item(item);
        }
        Ok(dynamic)
    }
}

// --- compilation --------------------------------------------------------------------------------

/// One parsed module, with the static context its prolog produced.
struct Unit {
    source: String,
    module: Module,
    namespaces: BTreeMap<String, String>,
    context: StaticQueryContext,
    options: Arc<RuntimeOptions>,
    target_namespace: Option<String>,
}

fn compile_query(query: &str, context: StaticQueryContext) -> XQueryResult<CompiledQuery> {
    let mut units = Vec::new();
    load_units(query.to_owned(), &context, &mut units)?;
    check_annotations(&units)?;
    check_variable_references(&units, &context)?;

    // Each module's decimal formats join its own runtime options, before anything that captures
    // them — a declared function keeps the options of the module it was written in.
    for unit in &mut units {
        let formats = unit_decimal_formats(unit)?;
        if formats.is_empty() {
            continue;
        }
        let mut options = (*unit.options).clone();
        options.decimal_formats = Arc::new(formats);
        unit.options = Arc::new(options);
    }

    // Every function every module declares, so that a call across modules resolves — except a
    // `%private` function of a library module, which only that module may call. Each unit
    // therefore gets a registry of its own: the built-ins, every public declared function, and
    // its own private ones.
    let store = Arc::new(FunctionStore::default());
    let mut declared: Vec<(ExpandedName, usize)> = Vec::new();
    let mut visible: Vec<(usize, ExpandedName, usize)> = Vec::new();
    for (index, unit) in units.iter().enumerate() {
        for function in &unit.module.prolog.functions {
            if function.external {
                // §4.18's rules on a declared function's *name* — no namespace is `XQST0060`, a
                // reserved namespace is `XQST0045`, and a library module's function must be in
                // its target namespace (`XQST0048`) — constrain the declaration, not its body,
                // so an `external` function is held to them too even though it has none.
                function_name(unit, function)?;
                continue;
            }
            let name = function_name(unit, function)?;
            let arity = function.parameters.len();
            if declared.contains(&(name.clone(), arity)) {
                return Err(XQueryError::new(
                    XQueryErrorCode::Xqst0034,
                    format!("`{name}#{arity}` is declared more than once"),
                ));
            }
            declared.push((name.clone(), arity));
            visible.push((index, name, arity));
        }
    }
    let mut registry = builtins();
    let mut private: Vec<(usize, ExpandedName, usize)> = Vec::new();
    for (owner, name, arity) in &visible {
        let hidden = units.get(*owner).is_some_and(|unit| {
            unit.target_namespace.is_some()
                && unit.module.prolog.functions.iter().any(|function| {
                    function.parameters.len() == *arity
                        && function.annotations.iter().any(|note| note == "private")
                        && function_name(unit, function).is_ok_and(|declared| &declared == name)
                })
        });
        if hidden {
            private.push((*owner, name.clone(), *arity));
        }
        register(&mut registry, &store, name.clone(), *arity);
    }
    check_private_calls(&units, &private)?;

    let mut variables: Vec<PrologVariable> = Vec::new();
    let mut functions: HashMap<(ExpandedName, usize), Arc<UserFunction>> = HashMap::new();
    let mut body = None;
    let mut context_item = None;
    let mut imported_context_item_types: Vec<TypeCheck> = Vec::new();
    let mut main_statik = None;
    let mut main_options = None;
    let mut output = OutputDeclarations::default();

    for unit in &units {
        // A named function reference to a declared function — `local:f#2` — is compiled by the
        // XPath engine, which finds the function in the registry but not the sequence types it
        // was declared with. A `function(…) as …` item test needs them (XQuery 3.1 §2.5.5.5),
        // so the store answers for them.
        let statik = xpath_static_context(&unit.context, &unit.namespaces, registry.clone())
            .with_host_signatures(declared_signatures(&store));
        let mut compiler = Compiler::new(
            &unit.source,
            statik,
            unit.namespaces.clone(),
            unit.context.default_element_namespace().map(str::to_owned),
            Arc::clone(&unit.options),
        );
        for declaration in &unit.module.prolog.variables {
            check_visibility(
                &declaration.annotations,
                &unit.namespaces,
                XQueryErrorCode::Xqst0116,
            )?;
            let variable = compile_variable(&mut compiler, declaration, Arc::clone(&unit.options))?;
            if variables
                .iter()
                .any(|existing| existing.name == variable.name)
            {
                return Err(XQueryError::new(
                    XQueryErrorCode::Xqst0049,
                    format!(
                        "the variable `${}` is declared more than once",
                        variable.name
                    ),
                ));
            }
            if let Some(target) = &unit.target_namespace {
                if variable.name.namespace() != Some(target.as_str()) {
                    return Err(XQueryError::new(
                        XQueryErrorCode::Xqst0048,
                        format!(
                            "`${}` is not in the library module's target namespace",
                            variable.name
                        ),
                    ));
                }
            }
            variables.push(variable);
        }
        for declaration in &unit.module.prolog.functions {
            if declaration.external {
                continue;
            }
            check_visibility(
                &declaration.annotations,
                &unit.namespaces,
                XQueryErrorCode::Xqst0106,
            )?;
            let name = function_name(unit, declaration)?;
            let function = compile_function(&mut compiler, declaration, name.clone(), unit)?;
            functions.insert((name, declaration.parameters.len()), Arc::new(function));
        }
        if unit.target_namespace.is_some() {
            if let Some(declared) = unit
                .module
                .prolog
                .context_item
                .as_ref()
                .and_then(|declaration| declaration.declared_type.as_deref())
            {
                imported_context_item_types.push(compiler.compile_type_match(declared)?);
            }
        }
        if unit.target_namespace.is_none() {
            if let Some(declaration) = &unit.module.prolog.context_item {
                // §4.15: a module with no declaration behaves as if it declared
                // `context item as item() external`, and the value must *match* that type — so
                // an initializer that yields anything but exactly one item is `XPTY0004`.
                let declared = declaration.declared_type.as_deref().unwrap_or("item()");
                context_item = Some(ContextItemSpec {
                    check: Some(compiler.compile_type_match(declared)?),
                    value: match &declaration.value {
                        Some(expr) => Some(compiler.compile(expr)?),
                        None => None,
                    },
                    external: declaration.external,
                });
            }
            let expression = unit.module.body.as_ref().ok_or_else(|| {
                XQueryError::new(XQueryErrorCode::Xpst0003, "the main module has no body")
            })?;
            body = Some(compiler.compile(expression)?);
            main_statik = Some(Arc::new(compiler.static_context().clone()));
            main_options = Some(Arc::clone(&unit.options));
            // Output declarations belong to the main module — `XQST0108` makes one in a library
            // module a static error — so only its prolog contributes them, and only its prefix
            // map resolves a QName-valued parameter.
            output = OutputDeclarations {
                declarations: crate::serialization::output_declarations(
                    &unit.module.prolog.options,
                    &unit.namespaces,
                ),
                namespaces: unit.namespaces.clone(),
                default_element_namespace: unit
                    .context
                    .default_element_namespace()
                    .map(str::to_owned),
            };
        }
    }

    let _ = store.functions.set(functions);
    let statik = main_statik.ok_or_else(|| {
        XQueryError::new(XQueryErrorCode::Xpst0003, "the query has no main module")
    })?;
    let options = main_options.unwrap_or_else(|| Arc::new(default_options(&context)));
    let body =
        body.ok_or_else(|| XQueryError::new(XQueryErrorCode::Xpst0003, "the query has no body"))?;
    Ok(CompiledQuery {
        source: query.to_owned(),
        statik,
        options,
        variables,
        context_item,
        body,
        collations: context.collations.clone(),
        imported_context_item_types,
        output,
        functions: store,
    })
}

/// Checks the annotation assertions that appeared inside sequence types.
///
/// What such an assertion *means* is implementation-defined and this processor ignores it, but
/// its name must still be a QName whose prefix is bound and which is not in a namespace the
/// specification reserves.
fn check_annotations(units: &[Unit]) -> XQueryResult<()> {
    for unit in units {
        for annotation in &unit.module.seen.type_annotations {
            let name = resolve_name(&unit.namespaces, Some(XQUERY_NAMESPACE), annotation)?;
            if is_reserved_annotation(&name) {
                return Err(XQueryError::new(
                    XQueryErrorCode::Xqst0045,
                    format!("`%{annotation}` is in a reserved namespace"),
                ));
            }
        }
        // An option's name is an `EQName`, so its prefix must be bound.
        for (option, _) in &unit.module.prolog.options {
            resolve_name(&unit.namespaces, None, option)?;
        }
        check_cast_targets(unit)?;
    }
    Ok(())
}

/// The simple types of the XSD namespace that are not *atomic*, and so are not in
/// [`oxixml_infoset::AtomicType`], but are still simple types: the union `xs:numeric` and
/// `xs:error`, and the three built-in list types. They are simple, so `XQST0052` does not apply
/// to them — `'12.5' castable as xs:numeric` is a legitimate question with the answer `true`.
const XSD_NON_ATOMIC_SIMPLE_TYPES: &[&str] =
    &["numeric", "error", "IDREFS", "NMTOKENS", "ENTITIES"];

/// Checks the target type of every `cast as` and `castable as`.
///
/// XQuery names two static errors here that XPath does not.
/// [XQuery 3.1 §3.14.2](https://www.w3.org/TR/xquery-31/#id-cast) makes it `XPST0080` to cast to
/// `xs:NOTATION`, `xs:anySimpleType` or `xs:anyAtomicType`, and `XQST0052` when the target "must
/// be the name of a type defined in the in-scope schema types, and the {variety} of the type
/// must be simple" — so a complex type such as `xs:anyType` or `xs:untyped`, and a name that
/// names no type at all, are `XQST0052` rather than the `XPST0051` XPath raises. This engine has
/// no schema, so the in-scope schema types are exactly the built-ins of the XSD namespace.
fn check_cast_targets(unit: &Unit) -> XQueryResult<()> {
    for (_, lexical) in &unit.module.seen.cast_targets {
        // Only a name whose namespace is fixed by the query text is decided here. An *unprefixed*
        // type name is in the default element/type namespace, which a namespace declaration
        // attribute changes for the whole of a constructor's content — a scope this pass does not
        // track — so it is left to the delegated compilation.
        if !lexical.starts_with("Q{") && !lexical.contains(':') {
            continue;
        }
        // A `SingleType` whose prefix is not bound is `XPST0081`, which the delegated
        // compilation already raises with the offset; an unresolvable name is left to it.
        let Ok(name) = resolve_name(&unit.namespaces, None, lexical) else {
            continue;
        };
        if matches!(
            (name.namespace(), name.local_name()),
            (
                Some(oxixml_infoset::XSD_NAMESPACE),
                "NOTATION" | "anySimpleType" | "anyAtomicType"
            )
        ) {
            return Err(XQueryError::new(
                XQueryErrorCode::Xpst0080,
                format!("`{lexical}` may not be the target type of a cast"),
            ));
        }
        let known = name.namespace() == Some(oxixml_infoset::XSD_NAMESPACE)
            && (oxixml_infoset::AtomicType::from_local_name(name.local_name()).is_some()
                || XSD_NON_ATOMIC_SIMPLE_TYPES.contains(&name.local_name()));
        if !known {
            return Err(XQueryError::new(
                XQueryErrorCode::Xqst0052,
                format!("`{lexical}` is not a simple type in the in-scope schema types"),
            ));
        }
    }
    Ok(())
}

/// Raises `XPST0017` for a call to a `%private` function of an imported library module.
///
/// Privacy cannot be enforced by leaving the function out of the function registry, because the
/// registry is also what dispatches a call at run time, and a library module's own body runs in
/// the importing module's dynamic context. It is enforced here instead, against the call names
/// the parser recorded.
fn check_private_calls(
    units: &[Unit],
    private: &[(usize, ExpandedName, usize)],
) -> XQueryResult<()> {
    if private.is_empty() {
        return Ok(());
    }
    for (index, unit) in units.iter().enumerate() {
        for (offset, lexical, arity) in &unit.module.seen.calls {
            let Ok(name) = resolve_name(
                &unit.namespaces,
                Some(unit.context.default_function_namespace()),
                lexical,
            ) else {
                continue;
            };
            if private
                .iter()
                .any(|(owner, hidden, at)| *owner != index && hidden == &name && at == arity)
            {
                return Err(XQueryError::new(
                    XQueryErrorCode::Xpst0017,
                    format!("`{name}#{arity}` is private to the module that declares it"),
                )
                .at(*offset));
            }
        }
    }
    Ok(())
}

/// Raises `XPST0008` for a reference to a variable that is not in scope where it stands.
///
/// The parser recorded every variable reference, and every variable binding **with the extent of
/// its scope** — a function's parameters over that function's body, a `typeswitch` case variable
/// over that case's return clause, a FLWOR or quantified variable over the rest of its own
/// expression, a prolog declaration over the whole module. A reference is in scope when some
/// binding of the same expanded name covers its offset, or when a *declaration* visible to its
/// module supplies it.
///
/// Visibility across modules follows XQuery 3.1 §4.16: a library module's public variables reach
/// only the modules that import its target namespace, so a name declared in one imported module
/// is not silently in scope in another.
fn check_variable_references(units: &[Unit], context: &StaticQueryContext) -> XQueryResult<()> {
    // The public declarations of each unit, paired with the namespace a module has to import to
    // see them. `None` is a main module, whose declarations no other module can see.
    let mut published: Vec<(Option<String>, ExpandedName)> = Vec::new();
    for unit in units {
        for declaration in &unit.module.prolog.variables {
            // A `%private` variable of a library module is not visible to its importers.
            if unit.target_namespace.is_some()
                && declaration.annotations.iter().any(|name| name == "private")
            {
                continue;
            }
            if let Ok(name) = resolve_name(&unit.namespaces, None, &declaration.name) {
                published.push((unit.target_namespace.clone(), name));
            }
        }
    }
    for unit in units {
        let mut declared: Vec<ExpandedName> = context.external_variables.clone();
        let imported: Vec<&str> = unit
            .module
            .prolog
            .module_imports
            .iter()
            .map(|import| import.target_namespace.as_str())
            .collect();
        for (owner, name) in &published {
            if owner
                .as_deref()
                .is_some_and(|namespace| imported.contains(&namespace))
            {
                declared.push(name.clone());
            }
        }
        // A module always sees its own declarations, private ones included.
        for declaration in &unit.module.prolog.variables {
            if let Ok(name) = resolve_name(&unit.namespaces, None, &declaration.name) {
                declared.push(name);
            }
        }
        let mut bound: Vec<(usize, usize, ExpandedName)> = Vec::new();
        for binding in &unit.module.seen.bindings {
            if let Ok(name) = resolve_name(&unit.namespaces, None, &binding.name) {
                bound.push((binding.at, binding.end, name));
            }
        }
        for (offset, lexical) in &unit.module.seen.references {
            let Ok(name) = resolve_name(&unit.namespaces, None, lexical) else {
                // An unbound prefix is `XPST0081`, which the XPath compiler reports itself.
                continue;
            };
            // A `catch` clause binds the `err:` variables without any syntax of its own.
            if name.namespace() == Some(crate::context::ERROR_NAMESPACE) {
                continue;
            }
            let in_scope = bound
                .iter()
                .any(|(at, end, bound)| bound == &name && *at <= *offset && *offset < *end);
            if in_scope {
                continue;
            }
            // A prolog variable is in scope over the whole module *except* its own initializing
            // expression, so `declare variable $v := $v;` is a reference to an undeclared
            // variable rather than a circularity. The suite pins this down: `prod/VarDecl.xml`'s
            // `K-InternalVariablesWith-15a` and `-15b` — one per language version, both amended
            // by the working group (bugs 15790 and 17272) — expect `XPST0008`, where every
            // cycle that runs through a *function* (`-16` … `-20`) expects `XQST0054`.
            if declared.contains(&name) && !initializes_itself(unit, &name, *offset) {
                continue;
            }
            return Err(XQueryError::new(
                XQueryErrorCode::Xpst0008,
                format!("the variable `${lexical}` is not declared"),
            )
            .at(*offset));
        }
    }
    Ok(())
}

/// Whether `offset` falls inside the initializing expression of `unit`'s own declaration of
/// `name` — the one place in the module where that name is not in scope.
fn initializes_itself(unit: &Unit, name: &ExpandedName, offset: usize) -> bool {
    unit.module.prolog.variables.iter().any(|declaration| {
        declaration.value.as_ref().is_some_and(|value| {
            value.start <= offset
                && offset < value.end
                && resolve_name(&unit.namespaces, None, &declaration.name)
                    .is_ok_and(|declared| &declared == name)
        })
    })
}

/// Every name `fn:format-number` might be given for a declared decimal format: the expanded
/// name, and the lexical `prefix:local` spelling for every prefix bound to its namespace.
fn format_aliases(
    name: Option<ExpandedName>,
    namespaces: &BTreeMap<String, String>,
) -> Vec<Option<ExpandedName>> {
    let Some(name) = name else {
        return vec![None];
    };
    let mut aliases = vec![Some(name.clone())];
    match name.namespace() {
        None => aliases.push(Some(ExpandedName::new(None::<&str>, name.local_name()))),
        Some(namespace) => {
            for (prefix, uri) in namespaces {
                if uri == namespace {
                    aliases.push(Some(ExpandedName::new(
                        None::<&str>,
                        format!("{prefix}:{}", name.local_name()),
                    )));
                }
            }
        }
    }
    aliases
}

/// `%public` and `%private` are mutually exclusive, as are `%deterministic` and
/// `%nondeterministic`.
fn check_visibility(
    annotations: &[String],
    namespaces: &BTreeMap<String, String>,
    code: XQueryErrorCode,
) -> XQueryResult<()> {
    let mut resolved = Vec::with_capacity(annotations.len());
    for annotation in annotations {
        let name = resolve_name(namespaces, Some(XQUERY_NAMESPACE), annotation)?;
        if is_reserved_annotation(&name) {
            return Err(XQueryError::new(
                XQueryErrorCode::Xqst0045,
                format!("`%{annotation}` is in a reserved namespace"),
            ));
        }
        resolved.push(name);
    }
    // The check is on the *expanded* names: `%private %xq:public` is as much a conflict as
    // `%private %public`, and `%public %public` repeats one annotation.
    let counts = |local: &str| {
        resolved
            .iter()
            .filter(|name| name.namespace() == Some(XQUERY_NAMESPACE) && name.local_name() == local)
            .count()
    };
    for (one, other) in [("public", "private"), ("deterministic", "nondeterministic")] {
        let here = counts(one);
        let there = counts(other);
        if here + there > 1 {
            return Err(XQueryError::new(
                code,
                format!("`%{one}` and `%{other}` may be declared at most once between them"),
            ));
        }
    }
    Ok(())
}

/// Whether `uri` is an absolute URI — one that begins with a scheme.
///
/// `scheme = ALPHA *( ALPHA / DIGIT / "+" / "-" / "." )` (RFC 3986 §3.1), followed by a `:`.
fn is_absolute_uri(uri: &str) -> bool {
    let Some((scheme, _)) = uri.split_once(':') else {
        return false;
    };
    let mut characters = scheme.chars();
    characters
        .next()
        .is_some_and(|first| first.is_ascii_alphabetic())
        && characters.all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'))
}

/// Whether `zero` may be a decimal format's `zero-digit`.
///
/// [XQuery 3.1 §4.13](https://www.w3.org/TR/xquery-31/#id-decimal-format-decl) requires the
/// value to be "a character that is classified in Unicode as a decimal digit, and that has the
/// numeric value zero" — the first of a family of ten consecutive digits, which is what
/// `fn:format-number` then counts through. Unicode guarantees exactly that shape: every `Nd`
/// block is ten contiguous code points beginning at its own zero, so a character is a digit zero
/// when it and the nine after it are digits and the one before it is not. `1` is a digit but not
/// a zero (`0` precedes it), and `A` is not a digit at all; both are `XQST0097`.
fn is_digit_zero(zero: char) -> bool {
    let code = u32::from(zero);
    if code > 0 && char::from_u32(code - 1).is_some_and(char::is_numeric) {
        return false;
    }
    (0..10).all(|offset| char::from_u32(code + offset).is_some_and(char::is_numeric))
}

/// Builds a [`DecimalFormat`] from the properties a `declare decimal-format` gave.
fn decimal_format(properties: &[(String, String)]) -> XQueryResult<DecimalFormat> {
    let mut format = DecimalFormat::default();
    let mut seen: Vec<&str> = Vec::new();
    for (name, value) in properties {
        if seen.contains(&name.as_str()) {
            return Err(XQueryError::new(
                XQueryErrorCode::Xqst0114,
                format!("the decimal-format property `{name}` is given more than once"),
            ));
        }
        seen.push(name);
        let single = || {
            let mut characters = value.chars();
            match (characters.next(), characters.next()) {
                (Some(c), None) => Ok(c),
                _ => Err(XQueryError::new(
                    XQueryErrorCode::Xqst0097,
                    format!("the decimal-format property `{name}` must be one character"),
                )),
            }
        };
        format = match name.as_str() {
            "decimal-separator" => format.with_decimal_separator(single()?),
            "grouping-separator" => format.with_grouping_separator(single()?),
            "exponent-separator" => format.with_exponent_separator(single()?),
            "infinity" => format.with_infinity(value.clone()),
            "minus-sign" => format.with_minus_sign(single()?),
            "NaN" => format.with_nan(value.clone()),
            "percent" => format.with_percent(single()?),
            "per-mille" => format.with_per_mille(single()?),
            "zero-digit" => {
                let zero = single()?;
                if !is_digit_zero(zero) {
                    return Err(XQueryError::new(
                        XQueryErrorCode::Xqst0097,
                        format!(
                            "the decimal-format property `zero-digit` is `{zero}`, which is not \
                             a Unicode digit whose numeric value is zero"
                        ),
                    ));
                }
                format.with_zero_digit(zero)
            }
            "digit" => format.with_digit(single()?),
            "pattern-separator" => format.with_pattern_separator(single()?),
            other => {
                // `DFPropertyName` is a closed list of terminals in the grammar
                // ([XQuery 3.1 §4.13](https://www.w3.org/TR/xquery-31/#id-decimal-format-decl)),
                // so a name that is not one of them does not parse at all — `XPST0003`, where
                // `XQST0097` is reserved for a name that parses but whose *value* is wrong.
                return Err(XQueryError::new(
                    XQueryErrorCode::Xpst0003,
                    format!("`{other}` is not a decimal-format property name"),
                ));
            }
        };
    }
    format.validate().map_err(|error| {
        XQueryError::new(XQueryErrorCode::Xqst0098, error.description().to_owned())
    })?;
    Ok(format)
}

/// The source of declared signatures the XPath engine consults for a named function reference.
///
/// The reference holds only a `Weak` handle, because the store owns the compiled functions and
/// the static context is held by every expression compiled from them: an `Arc` here would keep
/// the query alive through its own static context.
fn declared_signatures(store: &Arc<FunctionStore>) -> oxixml_xpath::HostSignatures {
    let weak: Weak<FunctionStore> = Arc::downgrade(store);
    Arc::new(move |name: &ExpandedName, arity: usize| {
        weak.upgrade()?.get(name, arity)?.declared_signature.clone()
    })
}

fn register(
    registry: &mut FunctionRegistry,
    store: &Arc<FunctionStore>,
    name: ExpandedName,
    arity: usize,
) {
    let namespace = name.namespace().unwrap_or_default().to_owned();
    let local = name.local_name().to_owned();
    let weak: Weak<FunctionStore> = Arc::downgrade(store);
    let key = name.clone();
    registry.add(&namespace, &local, arity, move |_fn_context, arguments| {
        let store = weak.upgrade().ok_or_else(|| {
            FnError::with_name(
                XPathErrorCode::Xpst0017.expanded_name(),
                "the compiled query that declared this function has been dropped",
            )
        })?;
        let function = store.get(&key, arity).ok_or_else(|| {
            FnError::with_name(
                XPathErrorCode::Xpst0017.expanded_name(),
                format!("`{key}#{arity}` is not declared"),
            )
        })?;
        let context = ambient().ok_or_else(|| {
            FnError::with_name(
                XPathErrorCode::Xpdy0002.expanded_name(),
                "a declared function was called outside a query execution",
            )
        })?;
        let frame = Frame::new(Arc::clone(&function.options), context);
        let values: Vec<XpathValue> = arguments.iter().cloned().map(XpathValue::from).collect();
        let result = call_function(&function, &values, &frame).map_err(|error| {
            FnError::with_name(error.name().clone(), error.message().to_owned())
        })?;
        result
            .into_sequence()
            .map_err(|error| FnError::with_name(error.name().clone(), error.message().to_owned()))
    });
}

fn function_name(unit: &Unit, declaration: &FunctionDeclaration) -> XQueryResult<ExpandedName> {
    let name = resolve_name(
        &unit.namespaces,
        Some(unit.context.default_function_namespace()),
        &declaration.name,
    )?;
    let Some(namespace) = name.namespace() else {
        return Err(XQueryError::new(
            XQueryErrorCode::Xqst0060,
            format!("the declared function `{name}` has no namespace"),
        ));
    };
    if is_reserved_function_namespace(namespace) {
        return Err(XQueryError::new(
            XQueryErrorCode::Xqst0045,
            format!("`{namespace}` is reserved and may not hold a declared function"),
        ));
    }
    if let Some(target) = &unit.target_namespace {
        if namespace != target {
            return Err(XQueryError::new(
                XQueryErrorCode::Xqst0048,
                format!("`{name}` is not in the library module's target namespace"),
            ));
        }
    }
    let mut seen: Vec<&String> = Vec::new();
    for (parameter, _) in &declaration.parameters {
        if seen.contains(&parameter) {
            return Err(XQueryError::new(
                XQueryErrorCode::Xqst0039,
                format!("`{name}` declares the parameter `${parameter}` twice"),
            ));
        }
        seen.push(parameter);
    }
    Ok(name)
}

fn compile_variable(
    compiler: &mut Compiler<'_>,
    declaration: &VariableDeclaration,
    options: Arc<RuntimeOptions>,
) -> XQueryResult<PrologVariable> {
    Ok(PrologVariable {
        options,
        name: compiler.resolve_variable(&declaration.name)?,
        check: match &declaration.declared_type {
            Some(declared) => Some(compiler.compile_type_match(declared)?),
            None => None,
        },
        value: match &declaration.value {
            Some(expr) => Some(compiler.compile(expr)?),
            None => None,
        },
        external: declaration.external,
    })
}

/// Applies XQuery's end-of-line normalization to a module's source text.
///
/// Only the XML 1.0 line endings are normalized. XML 1.1 adds `#x85` and `#x2028`, and the suite
/// is explicit that an XML-1.0 processor must leave both alone (`misc/XMLEdition.xml`'s
/// `line-ending-Q007` … `-Q009`, each pinned to `xml-version="1.0"`, against `-Q004` … `-Q006`
/// pinned to `1.1`).
fn normalize_line_endings(source: String) -> String {
    if !source.contains('\r') {
        return source;
    }
    let mut normalized = String::with_capacity(source.len());
    let mut characters = source.chars().peekable();
    while let Some(character) = characters.next() {
        if character == '\r' {
            if characters.peek() == Some(&'\n') {
                characters.next();
            }
            normalized.push('\n');
        } else {
            normalized.push(character);
        }
    }
    normalized
}

fn compile_function(
    compiler: &mut Compiler<'_>,
    declaration: &FunctionDeclaration,
    name: ExpandedName,
    unit: &Unit,
) -> XQueryResult<UserFunction> {
    let mut parameters = Vec::with_capacity(declaration.parameters.len());
    let mut checks = Vec::with_capacity(declaration.parameters.len());
    for (parameter, declared) in &declaration.parameters {
        parameters.push(compiler.resolve_variable(parameter)?);
        checks.push(match declared {
            Some(declared) => Some(compiler.compile_type_check(declared)?),
            None => None,
        });
    }
    let result_type = match &declaration.result_type {
        Some(declared) => Some(compiler.compile_type_check(declared)?),
        None => None,
    };
    let body = match &declaration.body {
        Some(expr) => compiler.compile(expr)?,
        None => Program::EmptySequence,
    };
    let declared_signature = compiler.declared_signature(
        &declaration
            .parameters
            .iter()
            .map(|(_, declared)| declared.clone())
            .collect::<Vec<_>>(),
        declaration.result_type.as_deref(),
    );
    Ok(UserFunction {
        name: Some(name),
        options: Arc::clone(&unit.options),
        parameters,
        parameter_types: checks,
        result_type,
        declared_signature,
        body,
    })
}

/// The target namespace of an import still to be loaded, and the module sources that satisfy it.
type PendingImports = Vec<(String, Vec<String>)>;

/// One step of the module-loading worklist.
enum Loading {
    /// A module still to be parsed, and the target namespace it was imported under.
    Module {
        source: String,
        target_namespace: Option<String>,
    },
    /// A module whose imports have all been loaded, ready to join the program.
    Prepared(Box<Unit>),
}

/// Compiles a module and everything it imports, depth first, so that a module is compiled after
/// everything it depends on.
///
/// The chain of imports is walked with an explicit worklist rather than by recursion. A library
/// built out of modules that each import the next is an ordinary shape, and nothing bounds how
/// long such a chain may be, so its length must not be a depth of the native stack: a chain of
/// any length costs the same handful of bytes here.
///
/// # Errors
///
/// Every static error a module's prolog raises, and `XQST0059` for an import that cannot be
/// resolved.
fn load_units(
    source: String,
    base: &StaticQueryContext,
    units: &mut Vec<Unit>,
) -> XQueryResult<()> {
    // The set of target namespaces already loaded anywhere in the program, which keeps a module
    // from being compiled twice and makes a cycle terminate. Cycles are *not* an error: XQuery
    // 3.0 erratum E8 removed that restriction, and the W3C test suite requires a module that
    // imports itself — directly (`modules-circular`) or indirectly (`errata8-002a`,
    // `errata8-003`) — to compile and run, so `XQST0093` is deliberately never raised.
    let mut imported: HashSet<String> = HashSet::new();
    let mut worklist = vec![Loading::Module {
        source,
        target_namespace: None,
    }];
    while let Some(step) = worklist.pop() {
        let (source, target_namespace) = match step {
            Loading::Prepared(unit) => {
                units.push(*unit);
                continue;
            }
            Loading::Module {
                source,
                target_namespace,
            } => (source, target_namespace),
        };
        let (unit, pending) = prepare_unit(source, base, target_namespace, &mut imported)?;
        // The worklist is a stack, so the unit goes on first and its imports on top of it, in
        // reverse, which leaves them in written order and the unit itself last.
        worklist.push(Loading::Prepared(Box::new(unit)));
        for (target, sources) in pending.into_iter().rev() {
            for module_source in sources.into_iter().rev() {
                worklist.push(Loading::Module {
                    source: module_source,
                    target_namespace: Some(target.clone()),
                });
            }
        }
    }
    Ok(())
}

/// Parses one module and applies its prolog to a copy of the static context, returning the unit
/// and the modules it imports that are not loaded yet.
fn prepare_unit(
    source: String,
    base: &StaticQueryContext,
    target_namespace: Option<String>,
    imported: &mut HashSet<String>,
) -> XQueryResult<(Unit, PendingImports)> {
    // XQuery 3.1 §A.2.2 "End-of-Line Handling": before parsing, `#xD #xA` and a `#xD` not
    // followed by `#xA` are both replaced by a single `#xA`. Doing it here, over the whole
    // module, is what keeps `string-to-codepoints('&#xD;&#xA;')` a single line feed — and every
    // byte offset the parser records stays an offset into this normalized text.
    let source = normalize_line_endings(source);
    let module = {
        let mut parser = Parser::new(&source)?;
        parser.parse_module()?
    };
    // The annotation assertions the parser found are blanked out — replaced by spaces, so that
    // every other offset stays where it was — because the XPath compiler cannot parse a `%`.
    let source = blank(source, &module.seen.blanked);
    // §4.15: a library module may declare the *type* of the context item, so that its own
    // functions and variables can rely on it, but never its *value* — the value belongs to
    // whoever invokes the query. This is checked before the target namespace is matched, because
    // the suite pairs the two (`prod/ContextItemDecl.xml`'s `contextDecl-052` imports a module
    // that gets *both* wrong and expects `XQST0113`).
    if module.module_namespace.is_some() {
        if let Some(declaration) = &module.prolog.context_item {
            if declaration.value.is_some() {
                return Err(XQueryError::new(
                    XQueryErrorCode::Xqst0113,
                    "a library module's context item declaration may not supply a value",
                ));
            }
        }
    }
    if let (Some(target), Some((_, declared))) = (&target_namespace, &module.module_namespace) {
        if declared != target {
            return Err(XQueryError::new(
                XQueryErrorCode::Xqst0059,
                format!("the imported module declares `{declared}`, not `{target}`"),
            ));
        }
    }
    let mut context = base.clone();
    let prolog = &module.prolog;
    let mut namespaces: BTreeMap<String, String> = base
        .namespaces()
        .map(|(prefix, uri)| (prefix.to_owned(), uri.to_owned()))
        .collect();
    if let Some((prefix, uri)) = &module.module_namespace {
        namespaces.insert(prefix.clone(), uri.clone());
    }
    let mut declared_prefixes: Vec<String> = Vec::new();
    for (prefix, uri) in &prolog.namespaces {
        if declared_prefixes.contains(prefix) {
            return Err(XQueryError::new(
                XQueryErrorCode::Xqst0033,
                format!("the prefix `{prefix}` is bound more than once"),
            ));
        }
        if is_reserved_prolog_binding(prefix, uri) {
            return Err(XQueryError::new(
                XQueryErrorCode::Xqst0070,
                format!("`{prefix}` may not be bound to `{uri}`"),
            ));
        }
        declared_prefixes.push(prefix.clone());
        namespaces.insert(prefix.clone(), uri.clone());
    }
    for import in &prolog.schema_imports {
        // XQuery 3.1 §4.12: a schema import with an empty target namespace may not bind a
        // prefix, and a *basic* processor — one without the Schema Import Feature, which this
        // one is — must reject the import outright with `XQST0009`.
        if import.target_namespace.is_empty() && import.prefix.is_some() {
            return Err(XQueryError::new(
                XQueryErrorCode::Xqst0057,
                "a schema import for the zero-length target namespace may not bind a prefix",
            ));
        }
        if let Some(prefix) = &import.prefix {
            if is_reserved_prolog_binding(prefix, &import.target_namespace) {
                return Err(XQueryError::new(
                    XQueryErrorCode::Xqst0070,
                    format!(
                        "`{prefix}` may not be bound to `{}`",
                        import.target_namespace
                    ),
                ));
            }
            namespaces.insert(prefix.clone(), import.target_namespace.clone());
        }
    }
    if let Some(import) = prolog.schema_imports.first() {
        let what = if import.default_element_namespace {
            "as the default element namespace"
        } else {
            "under a prefix"
        };
        return Err(XQueryError::new(
            XQueryErrorCode::Xqst0009,
            format!(
                "importing the schema `{}` {what} needs the Schema Import Feature, which this processor does not have",
                import.target_namespace
            ),
        ));
    }
    if let Some(uri) = &prolog.default_element_namespace {
        context = context.with_default_element_namespace(uri.clone());
    }
    if let Some(uri) = &prolog.default_function_namespace {
        context = context.with_default_function_namespace(uri.clone());
    }
    if let Some(uri) = &prolog.base_uri {
        // A `URILiteral` is whitespace-normalized as an `xs:anyURI` is.
        let normalized = uri.split_whitespace().collect::<Vec<_>>().join(" ");
        // §4.5: "if the URILiteral is a relative URI, it is resolved against the base URI
        // already in the static context". With nothing to resolve against it is left as it
        // stands — the suite requires a relative declaration to survive (`prod/BaseURIDecl.xml`'s
        // `base-uri-26`, `misc/HigherOrderFunctions.xml`'s `xqhof17` and `-19`), so `XPST0001`
        // is *not* raised for it.
        let resolved = match context.base_uri() {
            Some(base) if !is_absolute_uri(&normalized) => {
                oxixml_fn::strings::uri::resolve_uri_against(&normalized, base)
                    .unwrap_or_else(|_| normalized.clone())
            }
            _ => normalized,
        };
        context = context.with_base_uri(resolved);
    }
    if let Some(uri) = &prolog.default_collation {
        // §4.4: a default collation the processor does not support is `XQST0038`. A relative
        // URI is resolved against the static base URI first.
        let resolved = match context.base_uri() {
            Some(base) => oxixml_fn::strings::uri::resolve_uri_against(uri, base)
                .unwrap_or_else(|_| uri.clone()),
            None => uri.clone(),
        };
        if !context.knows_collation(&resolved) {
            return Err(XQueryError::new(
                XQueryErrorCode::Xqst0038,
                format!("the collation `{uri}` is not one this processor supports"),
            ));
        }
        context = context.with_default_collation(resolved);
    }
    if let Some(policy) = prolog.boundary_space {
        context = context.with_boundary_space(policy);
    }
    if let Some(mode) = prolog.construction {
        context = context.with_construction_mode(mode);
    }
    if let Some(mode) = prolog.ordering {
        context = context.with_ordering_mode(mode);
    }
    if let Some(order) = prolog.empty_order {
        context = context.with_empty_order(order);
    }
    if let Some(mode) = prolog.copy_namespaces {
        context = context.with_copy_namespaces(mode);
    }

    // Imports have to be loaded before this unit is pushed, so that a module is compiled after
    // everything it depends on.
    let mut pending = Vec::new();
    for import in &prolog.module_imports {
        if let Some(prefix) = &import.prefix {
            if is_reserved_prolog_binding(prefix, &import.target_namespace) {
                return Err(XQueryError::new(
                    XQueryErrorCode::Xqst0070,
                    format!(
                        "`{prefix}` may not be bound to `{}`",
                        import.target_namespace
                    ),
                ));
            }
            namespaces.insert(prefix.clone(), import.target_namespace.clone());
        }
        if imported.contains(&import.target_namespace) {
            continue;
        }
        if prolog
            .module_imports
            .iter()
            .filter(|other| other.target_namespace == import.target_namespace)
            .count()
            > 1
        {
            return Err(XQueryError::new(
                XQueryErrorCode::Xqst0047,
                format!("`{}` is imported more than once", import.target_namespace),
            ));
        }
        let Some(resolver) = base.module_resolver() else {
            return Err(XQueryError::new(
                XQueryErrorCode::Xqst0059,
                format!(
                    "`{}` cannot be imported: no module resolver is installed",
                    import.target_namespace
                ),
            ));
        };
        let sources = resolver
            .resolve(&import.target_namespace, &import.locations)
            .filter(|sources| !sources.is_empty())
            .ok_or_else(|| {
                XQueryError::new(
                    XQueryErrorCode::Xqst0059,
                    format!("no module was found for `{}`", import.target_namespace),
                )
            })?;
        imported.insert(import.target_namespace.clone());
        pending.push((import.target_namespace.clone(), sources));
    }

    let options = Arc::new(default_options(&context));
    Ok((
        Unit {
            source,
            module,
            namespaces,
            context,
            options,
            target_namespace,
        },
        pending,
    ))
}

/// Replaces every byte of every range in `ranges` with a space, leaving the text the same length
/// so that every recorded offset still points where it did.
fn blank(mut source: String, ranges: &[(usize, usize)]) -> String {
    if ranges.is_empty() {
        return source;
    }
    // Safe because a recorded range always spans whole tokens, so both ends fall on character
    // boundaries and each replaced byte becomes one ASCII space.
    let mut bytes = std::mem::take(&mut source).into_bytes();
    for &(start, end) in ranges {
        if let Some(slice) = bytes.get_mut(start..end) {
            slice.fill(b' ');
        }
    }
    String::from_utf8(bytes).unwrap_or_default()
}

fn default_options(context: &StaticQueryContext) -> RuntimeOptions {
    RuntimeOptions {
        boundary_space: context.boundary_space(),
        construction: context.construction_mode(),
        copy_namespaces: context.copy_namespaces(),
        empty_order: context.empty_order(),
        default_collation: context.default_collation().to_owned(),
        base_uri: context.base_uri().map(str::to_owned),
        decimal_formats: Arc::new(Vec::new()),
    }
}

/// Compiles the `declare decimal-format` declarations of one module's prolog.
///
/// [XQuery 3.1 §4.17](https://www.w3.org/TR/xquery-31/#id-decimal-format-decl) scopes them to the
/// module, so they belong to that module's [`RuntimeOptions`] rather than to one map shared by
/// the whole program: `decimal-format-21` imports a module that declares `df001` with a
/// grouping separator of its own and expects each module's `fn:format-number` to read its own.
///
/// `XQST0111` is likewise a constraint on *a* prolog, so the names seen are per module.
fn unit_decimal_formats(unit: &Unit) -> XQueryResult<Vec<(Option<ExpandedName>, DecimalFormat)>> {
    let mut names: Vec<Option<ExpandedName>> = Vec::new();
    let mut formats = Vec::new();
    for declaration in &unit.module.prolog.decimal_formats {
        let name = match &declaration.name {
            Some(lexical) => Some(resolve_name(&unit.namespaces, None, lexical)?),
            None => None,
        };
        if names.contains(&name) {
            return Err(XQueryError::new(
                XQueryErrorCode::Xqst0111,
                "the same decimal format is declared more than once",
            ));
        }
        names.push(name.clone());
        let format = decimal_format(&declaration.properties)?;
        // `fn:format-number` receives its format name as a *lexical* string, which the function
        // library has no namespaces to resolve, so the format is also registered under every
        // spelling the query could use for it.
        for alias in format_aliases(name, &unit.namespaces) {
            formats.push((alias, format.clone()));
        }
    }
    Ok(formats)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(query: &str) -> XQueryResult<String> {
        XQuery::compile_default(query)?.execute_to_string(&DynamicQueryContext::new())
    }

    #[test]
    fn a_pure_xpath_query_is_delegated() -> XQueryResult<()> {
        assert_eq!(run("1 + 2")?, "3");
        assert_eq!(
            run("string-join(for $x in 1 to 3 return string($x), '-')")?,
            "1-2-3"
        );
        Ok(())
    }

    #[test]
    fn a_compiled_query_is_shareable() -> XQueryResult<()> {
        let compiled = XQuery::compile_default("1 to 3")?;
        fn assert_send_sync<T: Send + Sync>(_: &T) {}
        assert_send_sync(&compiled);
        assert_eq!(compiled.query(), "1 to 3");
        Ok(())
    }
}
