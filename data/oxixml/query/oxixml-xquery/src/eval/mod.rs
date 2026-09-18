//! The evaluator for everything XPath does not have.
//!
//! Every [`Program::Xpath`] and [`Program::Spliced`] node hands its work to the XPath engine, so
//! what is left here is exactly the XQuery-only semantics: the FLWOR tuple stream, node
//! construction, `typeswitch`, `switch`, `try`/`catch`, and the handful of ordinary forms —
//! paths, predicates, `and`/`or`, `if`, bindings, inline functions — that have to be able to
//! *contain* one of those.

pub(crate) mod construct;
pub(crate) mod flwor;
pub(crate) mod program;

use crate::ast::{BindingKind, CatchTest};
use crate::context::{BoundarySpace, ConstructionMode, CopyNamespaces, EmptyOrder};
use crate::error::{XQueryError, XQueryErrorCode, XQueryResult};
use oxixml_infoset::{
    AtomicValue, ExpandedName, NodeKind, NodeRef, XdmNode, axis_nodes, sort_document_order,
};
use oxixml_xpath::{DynamicContext, Focus, XPathErrorCode, XpathFunction, XpathItem, XpathValue};
use program::{CompiledPostfix, LookupSpec, Program, TypeCheck, UserFunction};
use std::sync::Arc;

/// The stack a query assumes the thread it runs on has, when the caller does not say.
///
/// `std::thread::spawn` gives a new thread 2 MiB unless it is told otherwise, which is what a
/// thread pool or an async executor's worker hands a library, so it is the figure the evaluator
/// has to stay inside by default. A caller on a process's main thread, or on a thread it spawned
/// with room for deep recursion on purpose, says so with
/// [`DynamicQueryContext::with_stack_size`](crate::DynamicQueryContext::with_stack_size).
pub use oxixml_xpath::DEFAULT_STACK_SIZE;

/// How deeply a query may nest function calls before `XPDY0130` is raised, when the caller
/// declares no stack of its own.
///
/// The specification lets a processor impose an implementation-defined limit, and one is needed:
/// an unguarded runaway recursion would exhaust the thread's stack rather than raise an error a
/// `try`/`catch` can see. A larger declared stack raises it — see
/// [`oxixml_xpath::EvaluationLimits`].
pub use oxixml_xpath::DEFAULT_MAX_CALL_DEPTH;

/// Declares the stack of the thread an execution runs on, for as long as the guard is alive.
///
/// The declaration is the XPath engine's, so that a query, the XPath expressions it delegates and
/// the function items either of them builds all measure themselves against one figure. It
/// nests: a query executed from inside another one leaves the outer figure in place afterwards.
pub(crate) fn declare_stack(size: Option<usize>) -> oxixml_xpath::EvaluationLimits {
    oxixml_xpath::EvaluationLimits::declare(size.unwrap_or(DEFAULT_STACK_SIZE))
}

/// The prolog-settable components an evaluation reads.
#[derive(Debug, Clone)]
pub(crate) struct RuntimeOptions {
    pub(crate) boundary_space: BoundarySpace,
    pub(crate) construction: ConstructionMode,
    pub(crate) copy_namespaces: CopyNamespaces,
    pub(crate) empty_order: EmptyOrder,
    pub(crate) default_collation: String,
    /// The static base URI, which every constructed element and document node inherits.
    pub(crate) base_uri: Option<String>,
    /// The decimal formats this module's prolog declared, under every spelling
    /// `fn:format-number`'s `$decimal-format-name` could name them by.
    ///
    /// [XQuery 3.1 §4.17](https://www.w3.org/TR/xquery-31/#id-decimal-format-decl) makes a
    /// decimal format a component of the *static context of the module it is declared in*, and a
    /// module import does not import one. A same-named declaration in another module of the same
    /// program therefore does not displace it, which is what `decimal-format-21` checks.
    pub(crate) decimal_formats: Arc<Vec<(Option<ExpandedName>, oxixml_fn::numeric::DecimalFormat)>>,
}

/// One evaluation's environment.
#[derive(Clone)]
pub(crate) struct Frame {
    pub(crate) options: Arc<RuntimeOptions>,
    pub(crate) context: DynamicContext,
    /// The namespace bindings the enclosing direct element constructors declared, against which
    /// a lexical name inside a constructor is resolved.
    pub(crate) namespaces: Arc<Vec<(Option<String>, String)>>,
    /// The complete in-scope namespaces of the element being constructed, which a node copied
    /// into its content inherits when copy-namespaces mode says `inherit`. This is *more* than
    /// [`Frame::namespaces`]: it also holds the bindings namespace fixup created for the
    /// element's own name and its attributes' names.
    pub(crate) inherited: Arc<Vec<(Option<String>, String)>>,
    /// The base URI of the element being constructed, which its children inherit.
    ///
    /// [XQuery 3.1 §3.9.1.3](https://www.w3.org/TR/xquery-31/#id-content) sets the base-uri of a
    /// nested constructor's node, and of a copied node, "to be the same as that of its new
    /// parent … unless it has an `xml:base` attribute, in which case its base-uri property is
    /// set to the value of that attribute, resolved (if it is relative) against the base-uri
    /// property of its new parent node". `None` means there is no enclosing constructor, so the
    /// static base URI plays that role.
    pub(crate) parent_base: Option<Arc<str>>,
}

impl Frame {
    pub(crate) fn new(options: Arc<RuntimeOptions>, context: DynamicContext) -> Self {
        Self {
            options,
            context,
            namespaces: Arc::new(Vec::new()),
            inherited: Arc::new(Vec::new()),
            parent_base: None,
        }
    }

    pub(crate) fn with_focus(&self, focus: Focus) -> Self {
        Self {
            options: Arc::clone(&self.options),
            context: self.context.clone().with_focus(focus),
            namespaces: Arc::clone(&self.namespaces),
            inherited: Arc::clone(&self.inherited),
            parent_base: self.parent_base.clone(),
        }
    }

    pub(crate) fn without_focus(&self) -> Self {
        Self {
            options: Arc::clone(&self.options),
            context: self.context.clone().without_focus(),
            namespaces: Arc::clone(&self.namespaces),
            inherited: Arc::clone(&self.inherited),
            parent_base: self.parent_base.clone(),
        }
    }

    pub(crate) fn bind(&self, name: &ExpandedName, value: XpathValue) -> Self {
        Self {
            options: Arc::clone(&self.options),
            context: self.context.clone().with_variable_name(name.clone(), value),
            namespaces: Arc::clone(&self.namespaces),
            inherited: Arc::clone(&self.inherited),
            parent_base: self.parent_base.clone(),
        }
    }

    pub(crate) fn with_namespaces(&self, namespaces: Vec<(Option<String>, String)>) -> Self {
        Self {
            options: Arc::clone(&self.options),
            context: self.context.clone(),
            namespaces: Arc::new(namespaces),
            inherited: Arc::clone(&self.inherited),
            parent_base: self.parent_base.clone(),
        }
    }

    pub(crate) fn with_inherited(&self, inherited: Vec<(Option<String>, String)>) -> Self {
        Self {
            options: Arc::clone(&self.options),
            context: self.context.clone(),
            namespaces: Arc::clone(&self.namespaces),
            inherited: Arc::new(inherited),
            parent_base: self.parent_base.clone(),
        }
    }

    /// The frame a constructed element's *content* is evaluated in: its children's base URI is
    /// the element's own.
    pub(crate) fn with_parent_base(&self, base: Option<String>) -> Self {
        Self {
            options: Arc::clone(&self.options),
            context: self.context.clone(),
            namespaces: Arc::clone(&self.namespaces),
            inherited: Arc::clone(&self.inherited),
            parent_base: base.map(Arc::from),
        }
    }

    /// The base URI a node constructed in this frame inherits: the enclosing constructor's, or
    /// the static base URI when there is no enclosing constructor.
    pub(crate) fn construction_base(&self) -> Option<String> {
        self.parent_base.as_ref().map_or_else(
            || self.options.base_uri.clone(),
            |base| Some(base.to_string()),
        )
    }
}

/// Evaluates one compiled expression.
/// The frame's dynamic context, re-pointed at the static context `compiled` was compiled with.
///
/// A direct element constructor's namespace declaration attributes extend the statically known
/// namespaces for the whole of its content, so the compiler builds a static context per
/// constructor and compiles the enclosed expressions against it. Everything the *evaluator* then
/// reads from the statically known namespaces — the prefix of a lexical `xs:QName`, an
/// `xs:untypedAtomic` converted to one, `fn:in-scope-prefixes` on a name it constructs — has to
/// read the same context, not the module's.
fn in_scope(compiled: &oxixml_xpath::CompiledXPath, frame: &Frame) -> DynamicContext {
    frame
        .context
        .clone()
        .with_static_context(Arc::clone(compiled.static_context()))
}

pub(crate) fn evaluate(program: &Program, frame: &Frame) -> XQueryResult<XpathValue> {
    match program {
        Program::Xpath(compiled) => Ok(compiled.evaluate_value_in(&in_scope(compiled, frame))?),
        Program::Spliced { template, holes } => {
            let mut context = in_scope(template, frame);
            for (name, hole) in holes {
                let value = evaluate(hole, frame)?;
                context = context.with_variable_name(name.clone(), value);
            }
            Ok(template.evaluate_value_in(&context)?)
        }
        Program::Literal(value) => Ok(XpathValue::from(value.clone())),
        Program::EmptySequence => Ok(XpathValue::default()),
        Program::ContextItem => Ok(XpathValue::singleton(frame.context.context_item()?.clone())),
        Program::Wrap(inner) => evaluate(inner, frame),
        Program::If {
            condition,
            then_branch,
            else_branch,
        } => {
            if evaluate(condition, frame)?.effective_boolean_value()? {
                evaluate(then_branch, frame)
            } else {
                evaluate(else_branch, frame)
            }
        }
        Program::Logical { or, left, right } => {
            let first = evaluate(left, frame)?.effective_boolean_value()?;
            if *or && first {
                return Ok(XpathValue::from(AtomicValue::boolean(true)));
            }
            if !*or && !first {
                return Ok(XpathValue::from(AtomicValue::boolean(false)));
            }
            let second = evaluate(right, frame)?.effective_boolean_value()?;
            Ok(XpathValue::from(AtomicValue::boolean(second)))
        }
        Program::Binding {
            kind,
            bindings,
            body,
        } => evaluate_binding(*kind, bindings, body, frame),
        Program::Path {
            rooted,
            rooted_descendant,
            steps,
        } => evaluate_path(*rooted, *rooted_descendant, steps, frame),
        Program::Postfix { primary, postfixes } => {
            let mut value = evaluate(primary, frame)?;
            for postfix in postfixes {
                value = apply_postfix(value, postfix, frame)?;
            }
            Ok(value)
        }
        Program::Inline(function) => Ok(XpathValue::singleton(function_item(
            Arc::clone(function),
            frame,
        ))),
        Program::StringTemplate(parts) => {
            let mut out = String::new();
            for part in parts {
                match part {
                    program::CompiledContent::Text(text)
                    | program::CompiledContent::BoundaryWhitespace(text) => out.push_str(text),
                    // A string constructor has no nested constructors, so `Nested` cannot occur
                    // here; it is treated as an interpolation for exhaustiveness.
                    program::CompiledContent::Enclosed(expr)
                    | program::CompiledContent::Nested(expr) => {
                        // XQuery 3.1 §3.15: an interpolation's value is **atomized**, and the
                        // atomic values are then string-joined with a single space. Atomizing
                        // is what makes `` `{[$n]}` `` the array's members rather than
                        // `FOTY0014`, and what makes a map `FOTY0013` — the error the
                        // specification names for a value that has no typed value at all.
                        let atomized = evaluate(expr, frame)?.atomize()?;
                        let joined = atomized
                            .iter()
                            .map(AtomicValue::string_value)
                            .collect::<Vec<_>>()
                            .join(" ");
                        out.push_str(&joined);
                    }
                }
            }
            Ok(XpathValue::from(AtomicValue::string(out)))
        }
        Program::TypeSwitch(expr) => {
            let operand = evaluate(&expr.operand, frame)?;
            for case in &expr.cases {
                let probe = frame.bind(&typed_value_name(), operand.clone());
                let mut matched = false;
                for test in &case.tests {
                    if test.evaluate_to_boolean_in(&probe.context)? {
                        matched = true;
                        break;
                    }
                }
                if matched {
                    let inner = match &case.variable {
                        Some(name) => frame.bind(name, operand),
                        None => frame.clone(),
                    };
                    return evaluate(&case.body, &inner);
                }
            }
            let inner = match &expr.default_variable {
                Some(name) => frame.bind(name, operand),
                None => frame.clone(),
            };
            evaluate(&expr.default_body, &inner)
        }
        Program::Switch(expr) => {
            let operand = single_atomic(&evaluate(&expr.operand, frame)?)?;
            for (operands, body) in &expr.cases {
                for candidate in operands {
                    let value = single_atomic(&evaluate(candidate, frame)?)?;
                    if switch_equal(operand.as_ref(), value.as_ref())? {
                        return evaluate(body, frame);
                    }
                }
            }
            evaluate(&expr.default_body, frame)
        }
        Program::Try { body, catches } => match evaluate(body, frame) {
            Ok(value) => Ok(value),
            Err(error) => {
                for catch in catches {
                    if catch.tests.iter().any(|test| matches_error(test, &error)) {
                        let inner = bind_error_variables(frame, &error);
                        return evaluate(&catch.body, &inner);
                    }
                }
                Err(error)
            }
        },
        Program::Flwor(flwor) => flwor::evaluate_flwor(flwor, frame),
        Program::Construct(constructor) => construct::evaluate_constructor(constructor, frame),
    }
}

/// The variable a type check and a `typeswitch` case bind their operand to.
pub(crate) fn typed_value_name() -> ExpandedName {
    ExpandedName::new(None::<&str>, crate::compile::TYPED_VALUE)
}

/// Applies the function conversion rules of a `as T` declaration.
pub(crate) fn apply_type_check(
    check: &TypeCheck,
    value: XpathValue,
    frame: &Frame,
) -> XQueryResult<XpathValue> {
    let context = frame
        .context
        .clone()
        .with_variable_name(typed_value_name(), value.clone());
    let converted = check.convert.evaluate_value_in(&context).map_err(|error| {
        if error.code() == Some(XPathErrorCode::Xpdy0050) {
            XQueryError::new(XQueryErrorCode::Xpty0004, error.message().to_owned())
        } else {
            XQueryError::from(error)
        }
    })?;
    Ok(name_coerced_function(value, converted))
}

/// Carries a function's name across function coercion.
///
/// [XQuery 3.1 §3.1.5.2](https://www.w3.org/TR/xquery-31/#id-function-coercion) wraps a function
/// item that does not match the expected signature in one that does, and says of the wrapper
/// that it "has the same name and arity" as the function it wraps. The wrapper is written here
/// as an inline function expression, which the grammar gives no name, so the name is put back
/// once the conversion has run — the only case where the conversion turns a *named* function
/// item into an anonymous one.
fn name_coerced_function(original: XpathValue, converted: XpathValue) -> XpathValue {
    let (Some(XpathItem::Function(before)), Some(XpathItem::Function(after))) =
        (original.first(), converted.first())
    else {
        return converted;
    };
    if converted.len() != 1 || after.name().is_some() {
        return converted;
    }
    let Some(name) = before.name().cloned() else {
        return converted;
    };
    XpathValue::from(XpathItem::Function(after.clone().with_name(Some(name))))
}

fn evaluate_binding(
    kind: BindingKind,
    bindings: &[program::CompiledLetBinding],
    body: &Program,
    frame: &Frame,
) -> XQueryResult<XpathValue> {
    match kind {
        BindingKind::Let => {
            let mut inner = frame.clone();
            for binding in bindings {
                let mut value = evaluate(&binding.value, &inner)?;
                if let Some(check) = &binding.check {
                    value = apply_type_check(check, value, &inner)?;
                }
                inner = inner.bind(&binding.name, value);
            }
            evaluate(body, &inner)
        }
        BindingKind::For => {
            let mut out = Vec::new();
            iterate_bindings(bindings, 0, frame, &mut |inner| {
                out.extend(evaluate(body, inner)?.into_items());
                Ok(())
            })?;
            Ok(XpathValue::from_items(out))
        }
        BindingKind::Some | BindingKind::Every => {
            let every = kind == BindingKind::Every;
            let mut answer = every;
            iterate_bindings(bindings, 0, frame, &mut |inner| {
                let held = evaluate(body, inner)?.effective_boolean_value()?;
                if every {
                    answer &= held;
                } else {
                    answer |= held;
                }
                Ok(())
            })?;
            Ok(XpathValue::from(AtomicValue::boolean(answer)))
        }
    }
}

fn iterate_bindings(
    bindings: &[program::CompiledLetBinding],
    index: usize,
    frame: &Frame,
    visit: &mut dyn FnMut(&Frame) -> XQueryResult<()>,
) -> XQueryResult<()> {
    let Some(binding) = bindings.get(index) else {
        return visit(frame);
    };
    let value = evaluate(&binding.value, frame)?;
    for item in value.into_items() {
        let mut bound = XpathValue::singleton(item);
        if let Some(check) = &binding.check {
            bound = apply_type_check(check, bound, frame)?;
        }
        let inner = frame.bind(&binding.name, bound);
        iterate_bindings(bindings, index + 1, &inner, visit)?;
    }
    Ok(())
}

fn evaluate_path(
    rooted: bool,
    rooted_descendant: bool,
    steps: &[program::CompiledStep],
    frame: &Frame,
) -> XQueryResult<XpathValue> {
    // A relative path evaluates its first step in the *enclosing* focus, which may well be
    // absent — `<a/>/..` needs no context item at all.
    let mut remaining = steps;
    let mut current: Vec<XpathItem> = if rooted {
        let node = frame.context.context_node()?;
        let root = node.root();
        if root.node_kind() != NodeKind::Document {
            return Err(XQueryError::new(
                XQueryErrorCode::Xpdy0050,
                "a path that starts at `/` needs a context node in a document",
            ));
        }
        if rooted_descendant {
            axis_nodes(&root, oxixml_infoset::Axis::DescendantOrSelf)
                .into_iter()
                .map(XpathItem::from)
                .collect()
        } else {
            vec![XpathItem::from(root)]
        }
    } else {
        let Some((first, rest)) = steps.split_first() else {
            return Ok(XpathValue::default());
        };
        remaining = rest;
        let value = apply_predicates(evaluate(&first.base, frame)?, first, frame)?;
        let items = value.into_items();
        if first.simple_map {
            items
        } else {
            normalize(items, rest.is_empty())?
        }
    };

    let last = remaining.len().saturating_sub(1);
    for (ordinal, step) in remaining.iter().enumerate() {
        let requires_nodes = !step.simple_map;
        if step.descendant {
            current = expand_descendant_or_self(&current)?;
        }
        let size = current.len();
        let mut collected: Vec<XpathItem> = Vec::new();
        for (position, item) in current.iter().enumerate() {
            if requires_nodes && !item.is_node() {
                return Err(XQueryError::new(
                    XQueryErrorCode::Xpty0019,
                    format!("a path step was applied to {}, not a node", item.describe()),
                ));
            }
            let focus = Focus::new(item.clone(), position + 1, size);
            let inner = frame.with_focus(focus);
            let value = apply_predicates(evaluate(&step.base, &inner)?, step, &inner)?;
            collected.extend(value.into_items());
        }
        if !step.simple_map {
            collected = normalize(collected, ordinal == last)?;
        }
        current = collected;
    }
    Ok(XpathValue::from_items(current))
}

/// Applies a step's predicates in the order the step's axis runs.
///
/// XPath 3.1 §3.3.2: a predicate of an axis step is evaluated with the nodes selected by the axis
/// as its focus, and on a reverse axis that focus runs from the context node outwards. The step's
/// *value* is in document order either way, so the sequence is turned back around afterwards —
/// which is the whole difference between `ancestor::*[1]` (the nearest ancestor) and
/// `(ancestor::*)[1]` (the outermost one).
fn apply_predicates(
    value: XpathValue,
    step: &program::CompiledStep,
    frame: &Frame,
) -> XQueryResult<XpathValue> {
    if step.predicates.is_empty() {
        return Ok(value);
    }
    let reverse = step.reverse;
    let mut items = value.into_items();
    if reverse {
        items.reverse();
    }
    let mut value = XpathValue::from_items(items);
    for predicate in &step.predicates {
        value = filter(value, predicate, frame)?;
    }
    if reverse {
        let mut items = value.into_items();
        items.reverse();
        value = XpathValue::from_items(items);
    }
    Ok(value)
}

fn expand_descendant_or_self(items: &[XpathItem]) -> XQueryResult<Vec<XpathItem>> {
    let mut nodes = Vec::new();
    for item in items {
        let node = item.as_node().ok_or_else(|| {
            XQueryError::new(
                XQueryErrorCode::Xpty0019,
                format!("`//` was applied to {}, not a node", item.describe()),
            )
        })?;
        nodes.extend(axis_nodes(node, oxixml_infoset::Axis::DescendantOrSelf));
    }
    sort_document_order(&mut nodes);
    Ok(nodes.into_iter().map(XpathItem::from).collect())
}

/// Applies the "all nodes, in document order, without duplicates" rule of a path step.
///
/// `last` says whether the step is the final one, because XPath 3.1 §3.3.1 gives the two
/// positions different codes: a mixture in the last step is `XPTY0018`, while a mixture anywhere
/// earlier is an operand of the following `/`, and every such operand must be a sequence of
/// nodes — `XPTY0019`.
fn normalize(items: Vec<XpathItem>, last: bool) -> XQueryResult<Vec<XpathItem>> {
    let node_count = items.iter().filter(|item| item.is_node()).count();
    if node_count == 0 {
        return Ok(items);
    }
    if node_count != items.len() {
        if !last {
            return Err(XQueryError::new(
                XQueryErrorCode::Xpty0019,
                "a step of a path expression produced both nodes and non-nodes, and its result \
                 is the input of the following step",
            ));
        }
        return Err(XQueryError::new(
            XQueryErrorCode::Xpty0018,
            "the last step of a path expression produced both nodes and non-nodes",
        ));
    }
    let mut nodes: Vec<NodeRef> = items
        .iter()
        .filter_map(|item| item.as_node().cloned())
        .collect();
    sort_document_order(&mut nodes);
    Ok(nodes.into_iter().map(XpathItem::from).collect())
}

/// Filters a value by a predicate, with the positional rule of XPath 3.1 §3.3.5.
pub(crate) fn filter(
    value: XpathValue,
    predicate: &Program,
    frame: &Frame,
) -> XQueryResult<XpathValue> {
    let items = value.into_items();
    let size = items.len();
    let mut out = Vec::new();
    for (index, item) in items.into_iter().enumerate() {
        let focus = Focus::new(item.clone(), index + 1, size);
        let result = evaluate(predicate, &frame.with_focus(focus))?;
        let keep = match numeric_singleton(&result) {
            Some(position) => {
                #[allow(clippy::cast_precision_loss)]
                let wanted = (index + 1) as f64;
                (position - wanted).abs() < f64::EPSILON
            }
            None => result.effective_boolean_value()?,
        };
        if keep {
            out.push(item);
        }
    }
    Ok(XpathValue::from_items(out))
}

fn numeric_singleton(value: &XpathValue) -> Option<f64> {
    match value.as_slice() {
        [item] => item
            .as_atomic()
            .filter(|atomic| atomic.is_numeric())
            .and_then(AtomicValue::as_f64),
        _ => None,
    }
}

fn apply_postfix(
    value: XpathValue,
    postfix: &CompiledPostfix,
    frame: &Frame,
) -> XQueryResult<XpathValue> {
    match postfix {
        CompiledPostfix::Predicate(predicate) => filter(value, predicate, frame),
        CompiledPostfix::Arguments(arguments) => {
            let function = single_function(&value)?;
            let mut bound = Vec::with_capacity(arguments.len());
            let mut placeholders = false;
            for argument in arguments {
                match argument {
                    Some(expr) => bound.push(Some(evaluate(expr, frame)?)),
                    None => {
                        placeholders = true;
                        bound.push(None);
                    }
                }
            }
            if placeholders {
                let partial = function.partially_apply(bound)?;
                return Ok(XpathValue::singleton(partial));
            }
            let supplied: Vec<XpathValue> = bound.into_iter().flatten().collect();
            Ok(function.call(&frame.context, &supplied)?)
        }
        CompiledPostfix::Lookup(key) => lookup(value, key, frame),
    }
}

fn single_function(value: &XpathValue) -> XQueryResult<XpathFunction> {
    match value.as_slice() {
        [item] => item.as_callable().ok_or_else(|| {
            XQueryError::new(
                XQueryErrorCode::Xpty0004,
                format!("{} is not a function", item.describe()),
            )
        }),
        _ => Err(XQueryError::new(
            XQueryErrorCode::Xpty0004,
            "a dynamic function call needs exactly one function item",
        )),
    }
}

fn lookup(value: XpathValue, key: &LookupSpec, frame: &Frame) -> XQueryResult<XpathValue> {
    let mut keys: Option<Vec<AtomicValue>> = match key {
        LookupSpec::Name(name) => Some(vec![AtomicValue::string(name.clone())]),
        LookupSpec::Integer(number) => Some(vec![AtomicValue::integer(*number)]),
        LookupSpec::Wildcard => None,
        LookupSpec::Computed(expr) => {
            let computed = evaluate(expr, frame)?;
            Some(computed.atomize()?)
        }
    };
    let mut out = Vec::new();
    for item in value.iter() {
        if let Some(map) = item.as_map() {
            match &mut keys {
                None => {
                    for (_, entry) in map.entries() {
                        out.extend(entry.clone().into_items().into_iter().map(XpathItem::from));
                    }
                }
                Some(keys) => {
                    for key in keys.iter() {
                        if let Some(entry) = map.get(key) {
                            out.extend(entry.clone().into_items().into_iter().map(XpathItem::from));
                        }
                    }
                }
            }
            continue;
        }
        if let Some(array) = item.as_array() {
            match &mut keys {
                None => {
                    for member in array.members() {
                        out.extend(member.clone().into_items().into_iter().map(XpathItem::from));
                    }
                }
                Some(keys) => {
                    for key in keys.iter() {
                        // XPath 3.1 §3.11.3.2 applies the function conversion rules to the key,
                        // so an `xs:untypedAtomic` — an untyped node's content — is *cast* to
                        // `xs:integer`; an integer that names no member is out of bounds, which
                        // `XdmArray::get` reports as `FOAY0001`.
                        let integer = if key.atomic_type() == oxixml_infoset::AtomicType::Integer {
                            key.as_f64().map(|value| {
                                #[allow(clippy::cast_possible_truncation)]
                                {
                                    value as i64
                                }
                            })
                        } else if key.atomic_type() == oxixml_infoset::AtomicType::UntypedAtomic {
                            oxixml_infoset::AtomicValue::parse(
                                oxixml_infoset::AtomicType::Integer,
                                &key.string_value(),
                            )
                            .ok()
                            .and_then(|converted| converted.as_f64())
                            .map(|value| {
                                #[allow(clippy::cast_possible_truncation)]
                                {
                                    value as i64
                                }
                            })
                        } else {
                            None
                        };
                        let position = integer.and_then(|value| usize::try_from(value).ok());
                        let position = position.ok_or_else(|| {
                            XQueryError::new(
                                XQueryErrorCode::Xpty0004,
                                "an array lookup needs an xs:integer key",
                            )
                        })?;
                        let member = array.get(position)?;
                        out.extend(member.clone().into_items().into_iter().map(XpathItem::from));
                    }
                }
            }
            continue;
        }
        return Err(XQueryError::new(
            XQueryErrorCode::Xpty0004,
            format!("`?` needs a map or an array, not {}", item.describe()),
        ));
    }
    Ok(XpathValue::from_items(out))
}

/// Wraps a user-defined or inline function as a callable item, closing over `frame`.
pub(crate) fn function_item(function: Arc<UserFunction>, frame: &Frame) -> XpathFunction {
    let options = Arc::clone(&function.options);
    let captured = frame.context.clone();
    let namespaces = Arc::clone(&frame.namespaces);
    let inherited = Arc::clone(&frame.inherited);
    let arity = function.parameters.len();
    let name = function.name.clone();
    let signature = function.declared_signature.clone();
    let item = XpathFunction::native(name, arity, move |_context, arguments| {
        let inner = Frame {
            options: Arc::clone(&options),
            context: captured.clone(),
            namespaces: Arc::clone(&namespaces),
            inherited: Arc::clone(&inherited),
            parent_base: None,
        };
        Ok(call_function(&function, arguments, &inner)?)
    });
    match signature {
        Some((parameters, result)) => item.with_signature(parameters, result),
        None => item,
    }
}

/// Re-points a dynamic context's static base URI at the module the code being run belongs to.
///
/// [XQuery 3.1 §4.5](https://www.w3.org/TR/xquery-31/#id-base-uri-decl) makes `declare base-uri`
/// a component of *the static context of the module it appears in*, so a function or a variable
/// declared in a library module sees its own library's base URI in `fn:static-base-uri`,
/// `fn:resolve-uri#1`, `fn:doc` and every node it constructs — not the base URI of the module
/// that imported it.
pub(crate) fn rebase(context: &DynamicContext, options: &RuntimeOptions) -> DynamicContext {
    let rebase_uri = options
        .base_uri
        .as_deref()
        .filter(|base| context.fn_context().static_base_uri() != Some(*base));
    if rebase_uri.is_none() && options.decimal_formats.is_empty() {
        return context.clone();
    }
    let mut fn_context = context.fn_context().clone();
    if let Some(base) = rebase_uri {
        fn_context = fn_context.with_static_base_uri(base);
    }
    // The module's own declarations take precedence over any same-named declaration of the
    // module that imported it (XQuery 3.1 §4.17).
    for (name, format) in options.decimal_formats.iter() {
        fn_context = match name {
            Some(name) => fn_context.with_decimal_format(name.clone(), format.clone()),
            None => fn_context.with_default_decimal_format(format.clone()),
        };
    }
    context.clone().with_fn_context(fn_context)
}

/// Binds the arguments and evaluates a function's body.
pub(crate) fn call_function(
    function: &UserFunction,
    arguments: &[XpathValue],
    frame: &Frame,
) -> XQueryResult<XpathValue> {
    // A declared function reached through the function library does not pass through
    // `XpathFunction::call`, so it takes its level of the shared budget here; that is what makes
    // one recursive `local:f()` cost exactly one level however the call was routed.
    let _guard = oxixml_xpath::CallGuard::enter()?;
    let mut inner = frame.without_focus();
    inner.options = Arc::clone(&function.options);
    inner.context = rebase(&inner.context, &inner.options);
    for (index, name) in function.parameters.iter().enumerate() {
        let mut value = arguments.get(index).cloned().unwrap_or_default();
        if let Some(Some(check)) = function.parameter_types.get(index) {
            value = apply_type_check(check, value, &inner)?;
        }
        inner = inner.bind(name, value);
    }
    let mut result = evaluate(&function.body, &inner)?;
    if let Some(check) = &function.result_type {
        result = apply_type_check(check, result, &inner)?;
    }
    Ok(result)
}

/// `fn:string` of every item, joined — what a string constructor and a text node need.
pub(crate) fn join_string_values(value: &XpathValue) -> XQueryResult<String> {
    let mut parts = Vec::with_capacity(value.len());
    for item in value.iter() {
        parts.push(item.string_value()?);
    }
    Ok(parts.join(" "))
}

fn single_atomic(value: &XpathValue) -> XQueryResult<Option<AtomicValue>> {
    let atomized = value.atomize()?;
    match atomized.as_slice() {
        [] => Ok(None),
        [single] => Ok(Some(single.clone())),
        _ => Err(XQueryError::new(
            XQueryErrorCode::Xpty0004,
            "a `switch` operand must be at most one item",
        )),
    }
}

/// Compares a `switch` operand with a case operand.
///
/// [XQuery 3.1 §3.14](https://www.w3.org/TR/xquery-31/#id-switch) compares the two *atomized*
/// values with `fn:deep-equal`, not with the `eq` operator — which is why `switch (xs:double
/// 'NaN') case xs:float('NaN')` matches, even though `NaN eq NaN` is false.
fn switch_equal(left: Option<&AtomicValue>, right: Option<&AtomicValue>) -> XQueryResult<bool> {
    let sequence = |value: Option<&AtomicValue>| {
        value.map_or_else(oxixml_infoset::Sequence::empty, |value| {
            oxixml_infoset::Sequence::from(as_comparable(value))
        })
    };
    Ok(oxixml_infoset::deep_equal(&sequence(left), &sequence(right)).unwrap_or(false))
}

/// `xs:untypedAtomic` compares as `xs:string` in `switch`, `group by` and `order by`.
pub(crate) fn as_comparable(value: &AtomicValue) -> AtomicValue {
    if value.atomic_type() == oxixml_infoset::AtomicType::UntypedAtomic {
        return AtomicValue::string(value.string_value());
    }
    value.clone()
}

fn matches_error(test: &CatchTest, error: &XQueryError) -> bool {
    let name = error.name();
    match test {
        CatchTest::Any => true,
        CatchTest::Local(local) => name.local_name() == local,
        CatchTest::Namespace(Some(uri)) => name.namespace() == Some(uri.as_str()),
        CatchTest::Namespace(None) => name.namespace().is_none(),
        CatchTest::Name(expanded) => {
            let wanted = format!(
                "Q{{{}}}{}",
                name.namespace().unwrap_or_default(),
                name.local_name()
            );
            &wanted == expanded
        }
    }
}

fn bind_error_variables(frame: &Frame, error: &XQueryError) -> Frame {
    let namespace = Some(oxixml_xpath::ERR_NAMESPACE);
    let mut inner = frame.bind(
        &ExpandedName::new(namespace, "code"),
        XpathValue::from(AtomicValue::qname(oxixml_infoset::QNameValue::new(
            // `QNameValue::new` takes the *prefix* first, then the namespace. §3.16 binds
            // `$err:code` to "the error code" itself, so a `fn:error` whose `xs:QName` was
            // written with a prefix keeps that prefix; the specification's own codes are
            // spelled `err:LOCAL`, which is the fallback.
            error
                .name()
                .prefix()
                .or_else(|| error.name().namespace().is_some().then_some("err")),
            error.name().namespace().map(str::to_owned),
            error.name().local_name().to_owned(),
        ))),
    );
    inner = inner.bind(
        &ExpandedName::new(namespace, "description"),
        XpathValue::from(AtomicValue::string(
            error.description().unwrap_or(error.message()).to_owned(),
        )),
    );
    inner = inner.bind(
        &ExpandedName::new(namespace, "value"),
        error
            .error_object()
            .cloned()
            .map_or_else(XpathValue::default, XpathValue::from),
    );
    for local in ["module", "line-number", "column-number", "additional"] {
        inner = inner.bind(&ExpandedName::new(namespace, local), XpathValue::default());
    }
    inner
}
