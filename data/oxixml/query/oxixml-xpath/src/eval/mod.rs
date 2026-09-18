//! The evaluator: an [`Expr`] and a [`DynamicContext`] in, an [`XpathValue`] out.
//!
//! The dispatcher lives here, together with the constructs that are pure control flow —
//! sequences, conditionals, `for`, `let` and the quantified expressions — and the two conversions
//! that XPath 1.0 and XPath 2.0 disagree about most sharply, [`effective_boolean_value`] and
//! [`number_of`].
//!
//! Everything else is one module per family:
//!
//! | Module | Handles |
//! |---|---|
//! | [`path`] | `/`, `//`, axis steps, predicates, `!` |
//! | [`compare`] | general, value and node comparisons, in both 1.0 and 2.0 semantics |
//! | [`operators`] | arithmetic, `to`, `union`/`intersect`/`except`, `\|\|`, unary sign |
//! | [`typing`] | `instance of`, `treat as`, `cast as`, `castable as`, function conversion |
//! | [`functions`] | static and dynamic calls, inline functions, maps, arrays, lookup |

pub(crate) mod compare;
pub(crate) mod functions;
pub(crate) mod operators;
pub(crate) mod path;
pub(crate) mod typing;

use crate::ast::Expr;
use crate::context::DynamicContext;
use crate::error::{XPathError, XPathErrorCode, XPathResult};
use crate::value::XpathValue;
use oxixml_infoset::{AtomicType, AtomicValue, Item};
use std::cell::Cell;
use std::time::Instant;

pub(crate) use compare::compare_atomic_values;

/// How many *user-visible* calls an evaluation may nest before `XPDY0130` is raised.
///
/// Recursion at evaluation time can only grow through a call: the grammar's own nesting is
/// bounded when the expression is *parsed*
/// (`MAX_NESTING_DEPTH`), and the flat chains — `1+1+…+1`, `a/a/…/a` — are folded iteratively,
/// so the only way an evaluation goes arbitrarily deep is by invoking a function item that
/// invokes it again. Counting those calls therefore counts exactly what can run away, and it
/// counts it in the units the query is written in: `$f($f, 1000)` is a thousand calls, whatever
/// the evaluator does between two of them.
///
/// Without a ceiling a runaway recursion exhausts the thread's stack, which is a process abort
/// (`SIGABRT`) rather than a catchable Rust panic, so an application evaluating externally
/// supplied expression text could not defend against it. Past the ceiling the evaluation raises
/// `XPDY0130`, the code the specification reserves for an implementation-defined limit.
///
/// This figure is the *default*, and a floor: a host that has told the engine how much stack the
/// thread really has, with [`EvaluationLimits`], gets a ceiling scaled to it, because a caller
/// that deliberately runs a query on a 256 MiB thread has said it expects deep recursion. The
/// stack itself is guarded independently and by measurement, so the count is a guard against
/// *logical* runaway rather than the last line of defence.
pub const DEFAULT_MAX_CALL_DEPTH: usize = 512;

/// The native stack an evaluation assumes the thread it runs on has, when nothing declares one.
///
/// `std::thread::spawn` gives a new thread 2 MiB unless it is told otherwise, which is what a
/// thread pool or an async executor's worker hands a library, so it is the figure to stay inside
/// by default. A caller on a process's main thread, or on one it spawned with room for deep
/// recursion on purpose, says so with [`EvaluationLimits::declare`].
pub const DEFAULT_STACK_SIZE: usize = 2 * 1024 * 1024;

/// The share of the declared stack an evaluation may consume: two thirds of it, the remaining
/// third being left for the caller's own frames and for the work done between two checks.
const STACK_SHARE: usize = 3;

/// The stack one user-visible call is taken to cost, for the purpose of scaling
/// [`DEFAULT_MAX_CALL_DEPTH`] to a declared stack.
///
/// It is deliberately an under-estimate — an unoptimized build spends several times this — so
/// that the derived count does not become the binding limit on a thread that was sized for deep
/// recursion. What a call really costs is measured, not estimated: see [`CallGuard::enter`].
const CALL_STACK_ESTIMATE: usize = 8 * 1024;

/// The call ceiling in force on a thread that declared `stack_size` bytes of stack.
const fn call_ceiling(stack_size: usize) -> usize {
    let derived = stack_size / CALL_STACK_ESTIMATE;
    if derived > DEFAULT_MAX_CALL_DEPTH {
        derived
    } else {
        DEFAULT_MAX_CALL_DEPTH
    }
}

/// What the current thread has declared, and how much of it is in use.
#[derive(Clone, Copy)]
struct Budget {
    /// How many user-visible calls are open.
    calls: usize,
    /// A mark on the native stack, taken when the outermost call was entered.
    base: usize,
    /// The stack of the thread the evaluation is running on.
    stack_size: usize,
    /// How many user-visible calls may nest.
    max_calls: usize,
    /// The monotonic instant past which the evaluation gives up, when one has been declared.
    deadline: Option<Instant>,
}

impl Budget {
    const DEFAULT: Self = Self {
        calls: 0,
        base: 0,
        stack_size: DEFAULT_STACK_SIZE,
        max_calls: call_ceiling(DEFAULT_STACK_SIZE),
        deadline: None,
    };
}

thread_local! {
    /// The budget belongs to the thread rather than to a [`DynamicContext`] because a function
    /// item's implementation is free to evaluate against a context of its own — an inline
    /// function evaluates its body against the context it captured, and a host language builds
    /// a fresh one for every stylesheet function it exposes — and a budget any of those could
    /// reset would not bound the stack it is there to protect.
    static BUDGET: Cell<Budget> = const { Cell::new(Budget::DEFAULT) };
}

/// A mark on the native stack, taken as the address of a local.
///
/// Stacks grow downwards on every target this library builds for, but the direction does not
/// matter: what is compared is the distance between two marks.
fn stack_mark() -> usize {
    let marker = 0_u8;
    std::ptr::addr_of!(marker) as usize
}

/// Declares what evaluation on this thread may spend, restoring the previous declaration when it
/// is dropped.
///
/// Two limits stop a runaway recursion, and both are needed. The **call ceiling** is a count of
/// user-visible calls, which is what a query is written in and what a `try`/`catch` can be told
/// about; it defaults to [`DEFAULT_MAX_CALL_DEPTH`] and scales up with a larger declared stack.
/// The **stack budget** is the native stack itself, and it is the one that usually bites: a call
/// costs a few kilobytes in an optimized build and tens of kilobytes without optimization, so
/// the depth a thread can carry is a property of the thread far more than of the query.
/// Exhausting it aborts the process, which no `try`/`catch` can intercept, so what has actually
/// been consumed is measured and the evaluation stops while there is still room.
///
/// A declaration nests: an evaluation started from inside another one leaves the outer figures
/// in place when it finishes.
///
/// # Examples
///
/// ```
/// use oxixml_xpath::{EvaluationLimits, XPath, XPathVersion, StaticContext};
///
/// let compiled = XPath::compile_31("count(1 to 10)")?;
/// let context = compiled.context();
/// // The thread this runs on really does have room for the recursion the query needs.
/// let _limits = EvaluationLimits::declare(8 * 1024 * 1024);
/// assert_eq!(compiled.evaluate_to_string_in(&context)?, "10");
/// # Ok::<(), oxixml_xpath::XPathError>(())
/// ```
#[derive(Debug)]
pub struct EvaluationLimits {
    previous_stack_size: usize,
    previous_max_calls: usize,
    previous_deadline: Option<Instant>,
    /// The same declaration, mirrored into the function library's own call budget.
    ///
    /// A call to a function the library holds — every built-in, and every function a host
    /// registers with [`oxixml_fn::FunctionRegistry`] — is counted by
    /// [`oxixml_fn::function::CallLimits`] rather than by [`CallGuard`], because the library is
    /// callable without this engine. The two accountings therefore have to be told the same
    /// figures, or a host that declares a large stack here raises only half of the ceiling and a
    /// deep recursion through registered functions still stops at the default.
    ///
    /// The field exists to be *dropped* with the rest of the declaration, which is what restores
    /// the library's previous figures, so nothing ever reads it.
    #[allow(dead_code)]
    functions: oxixml_fn::function::CallLimits,
}

impl EvaluationLimits {
    /// Declares `stack_size` as the native stack of the thread the evaluation runs on, and takes
    /// the call ceiling from it.
    ///
    /// Declaring more than the thread really has would trade a bounded error for a process
    /// abort, so this is never guessed on the caller's behalf.
    #[must_use]
    pub fn declare(stack_size: usize) -> Self {
        Self::install(stack_size, call_ceiling(stack_size), Self::deadline())
    }

    /// The same, with the call ceiling given explicitly rather than derived.
    #[must_use]
    pub fn declare_with_max_call_depth(stack_size: usize, max_call_depth: usize) -> Self {
        Self::install(stack_size, max_call_depth, Self::deadline())
    }

    /// Declares a wall-clock `deadline` for evaluation on this thread, leaving the stack and
    /// call figures as they are.
    ///
    /// Nothing declares one by default, and an evaluation with no deadline pays a single branch
    /// on an already-loaded budget for it — the clock is never read. Once a deadline is declared
    /// and has passed, the evaluation stops at the next checkpoint with `XPDY0130`, the code the
    /// specification reserves for an implementation-defined limit, and the message says that the
    /// limit is a configurable deadline.
    ///
    /// **The deadline is checked between steps, not inside them.** The checkpoints are the ones
    /// the depth budget already uses — every user-visible call ([`CallGuard::enter`]) — plus the
    /// evaluator's own per-item loops: `for`, the quantified expressions, each step of a path,
    /// each item a predicate filters, and `!`. That bounds an evaluation that is *making calls
    /// or iterating*, which is what a runaway query does. It does **not** interrupt one builtin
    /// that is itself slow: a single `fn:matches` over a megabyte of input runs to completion
    /// whatever the deadline says, so a host that must reclaim the thread regardless still needs
    /// its own wall-clock kill. A long-running implementation can opt in by polling
    /// [`oxixml_fn::function::CallLimits::check_deadline`] from its own loop.
    ///
    /// A declaration nests like any other: an evaluation started inside another one leaves the
    /// outer deadline in place when it finishes.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_xpath::{EvaluationLimits, XPath};
    /// use std::time::{Duration, Instant};
    ///
    /// let compiled = XPath::compile_31("count(1 to 10)")?;
    /// let context = compiled.context();
    /// let _limits = EvaluationLimits::declare_deadline(Instant::now() + Duration::from_secs(30));
    /// assert_eq!(compiled.evaluate_to_string_in(&context)?, "10");
    /// # Ok::<(), oxixml_xpath::XPathError>(())
    /// ```
    #[must_use]
    pub fn declare_deadline(deadline: Instant) -> Self {
        let (stack_size, max_calls) =
            BUDGET.with(|cell| (cell.get().stack_size, cell.get().max_calls));
        Self::install(stack_size, max_calls, Some(deadline))
    }

    /// The call ceiling in force on this thread.
    #[must_use]
    pub fn max_call_depth() -> usize {
        BUDGET.with(|budget| budget.get().max_calls)
    }

    /// The declared native stack of this thread.
    #[must_use]
    pub fn stack_size() -> usize {
        BUDGET.with(|budget| budget.get().stack_size)
    }

    /// The wall-clock deadline in force on this thread, if any.
    #[must_use]
    pub fn deadline() -> Option<Instant> {
        BUDGET.with(|budget| budget.get().deadline)
    }

    /// Fails once the declared deadline has passed, and does nothing when none is declared.
    ///
    /// This is the checkpoint the evaluator's own loops take, exposed so that a host language
    /// built on this engine — an XSLT instruction that repeats, an XQuery FLWOR the host
    /// evaluates itself — can take the same one from its own loop and share the declaration
    /// rather than inventing a second budget.
    ///
    /// # Errors
    ///
    /// [`XPathErrorCode::Xpdy0130`] once the deadline has passed.
    pub fn check_deadline() -> XPathResult<()> {
        match Self::deadline() {
            Some(deadline) if Instant::now() >= deadline => Err(deadline_exceeded()),
            _ => Ok(()),
        }
    }

    fn install(stack_size: usize, max_calls: usize, deadline: Option<Instant>) -> Self {
        // The library's own budget is a separate accounting on the same thread, so it has to be
        // told the same figures — the deadline included — or half the calls an evaluation makes
        // would go unbounded. All three go in as one nesting level.
        let functions =
            oxixml_fn::function::CallLimits::declare_with(stack_size, max_calls, deadline);
        BUDGET.with(|cell| {
            let mut budget = cell.get();
            let previous = Self {
                previous_stack_size: budget.stack_size,
                previous_max_calls: budget.max_calls,
                previous_deadline: budget.deadline,
                functions,
            };
            budget.stack_size = stack_size;
            budget.max_calls = max_calls;
            budget.deadline = deadline;
            cell.set(budget);
            previous
        })
    }
}

impl Drop for EvaluationLimits {
    fn drop(&mut self) {
        BUDGET.with(|cell| {
            let mut budget = cell.get();
            budget.stack_size = self.previous_stack_size;
            budget.max_calls = self.previous_max_calls;
            budget.deadline = self.previous_deadline;
            cell.set(budget);
        });
    }
}

/// The error an evaluation that has run past its declared deadline raises.
fn deadline_exceeded() -> XPathError {
    XPathError::new(
        XPathErrorCode::Xpdy0130,
        "the evaluation has run past the deadline declared for this thread with \
         `EvaluationLimits`, which is this implementation's limit on how long it may run; the \
         deadline is configurable and is not set unless a host declares one"
            .to_owned(),
    )
}

/// Accounts for one user-visible call for as long as it is alive.
///
/// Every invocation of a function item passes through
/// [`XpathFunction::call`](crate::XpathFunction::call), which takes one of these; a host
/// language whose own calls do not (an XQuery `declare function`, an XSLT stylesheet function)
/// takes one itself, so that the two accountings are the same budget rather than two that each
/// see half the recursion.
///
/// The guard is what makes the count correct in the presence of `?`: every path out of a call,
/// successful or not, drops it.
#[derive(Debug)]
pub struct CallGuard {
    /// Whether this guard opened the outermost call, and so has to forget the stack mark again.
    outermost: bool,
}

impl CallGuard {
    /// Opens one call.
    ///
    /// # Errors
    ///
    /// [`XPathErrorCode::Xpdy0130`] when the call would nest past the ceiling, or when the
    /// native stack the calls in progress have consumed is close enough to the declared stack
    /// that going on risks exhausting it.
    pub fn enter() -> XPathResult<Self> {
        let (budget, outermost) = BUDGET.with(|cell| {
            let mut budget = cell.get();
            let outermost = budget.calls == 0;
            if outermost {
                budget.base = stack_mark();
            }
            budget.calls += 1;
            cell.set(budget);
            (budget, outermost)
        });
        let guard = Self { outermost };
        if budget.calls > budget.max_calls {
            return Err(XPathError::new(
                XPathErrorCode::Xpdy0130,
                format!(
                    "the expression nests more than {} calls deep, which is this implementation's \
                     limit on a thread of {} bytes; a query that recurses this deeply needs a \
                     larger stack, declared with `EvaluationLimits`",
                    budget.max_calls, budget.stack_size
                ),
            ));
        }
        let allowance = budget.stack_size / STACK_SHARE * (STACK_SHARE - 1);
        let used = budget.base.abs_diff(stack_mark());
        if used > allowance {
            return Err(XPathError::new(
                XPathErrorCode::Xpdy0130,
                format!(
                    "the calls in progress have used {used} bytes of native stack, more than the \
                     {allowance} they may take on a thread of {} bytes; a query that recurses \
                     this deeply needs a larger stack, declared with `EvaluationLimits`",
                    budget.stack_size
                ),
            ));
        }
        // The budget is already in hand, so a thread with no declared deadline pays one branch
        // for this and never reads the clock.
        if let Some(deadline) = budget.deadline {
            if Instant::now() >= deadline {
                return Err(deadline_exceeded());
            }
        }
        Ok(guard)
    }
}

impl Drop for CallGuard {
    fn drop(&mut self) {
        BUDGET.with(|cell| {
            let mut budget = cell.get();
            budget.calls = budget.calls.saturating_sub(1);
            if self.outermost {
                budget.base = 0;
            }
            cell.set(budget);
        });
    }
}

/// Evaluates `expression` against `context`.
///
/// # Errors
///
/// Any dynamic error the expression raises, and [`XPathErrorCode::Xpdy0130`] when the evaluation
/// nests past what [`EvaluationLimits`] allows.
pub(crate) fn evaluate(expression: &Expr, context: &DynamicContext) -> XPathResult<XpathValue> {
    match expression {
        Expr::Literal(value) => Ok(XpathValue::from(value.clone())),
        Expr::Deferred { code, message } => Err(XPathError::new(*code, message.clone())),
        Expr::VarRef(name) => Ok(context.require_variable(name)?.clone()),
        Expr::ContextItem => Ok(XpathValue::singleton(context.context_item()?.clone())),
        Expr::Root | Expr::Path(_, _) | Expr::Step(_) | Expr::SimpleMap(_, _) => {
            path::evaluate(expression, context)
        }
        Expr::Filter(base, predicate) => path::filter(base, predicate, context),
        Expr::Sequence(items) => {
            let mut out = XpathValue::empty();
            for item in items {
                out.append(evaluate(item, context)?);
            }
            Ok(out)
        }
        Expr::Unary { negate, operand } => {
            operators::unary(*negate, &evaluate(operand, context)?, context)
        }
        Expr::Binary { op, left, right } => operators::binary(*op, left, right, context),
        Expr::And(left, right) => {
            // XPath allows either operand to be evaluated first and permits short-circuiting;
            // evaluating left first and stopping on `false` is the conventional choice.
            if !effective_boolean_value(&evaluate(left, context)?)? {
                return Ok(boolean(false));
            }
            Ok(boolean(effective_boolean_value(&evaluate(
                right, context,
            )?)?))
        }
        Expr::Or(left, right) => {
            if effective_boolean_value(&evaluate(left, context)?)? {
                return Ok(boolean(true));
            }
            Ok(boolean(effective_boolean_value(&evaluate(
                right, context,
            )?)?))
        }
        Expr::General { op, left, right } => compare::general(*op, left, right, context),
        Expr::Value { op, left, right } => compare::value(*op, left, right, context),
        Expr::Node { op, left, right } => compare::node(*op, left, right, context),
        Expr::If {
            condition,
            then_branch,
            else_branch,
        } => {
            if effective_boolean_value(&evaluate(condition, context)?)? {
                evaluate(then_branch, context)
            } else {
                evaluate(else_branch, context)
            }
        }
        Expr::For { bindings, body } => for_expression(bindings, body, context),
        Expr::Let { bindings, body } => {
            let mut scoped = context.clone();
            for binding in bindings {
                let value = evaluate(&binding.value, &scoped)?;
                scoped = scoped.with_variable_name(binding.name.clone(), value);
            }
            evaluate(body, &scoped)
        }
        Expr::Quantified {
            every,
            bindings,
            satisfies,
        } => quantified(*every, bindings, satisfies, context),
        Expr::InstanceOf(operand, required) => {
            let value = evaluate(operand, context)?;
            Ok(boolean(required.matches(&value)))
        }
        Expr::TreatAs(operand, required) => {
            let value = evaluate(operand, context)?;
            required.treat(value)
        }
        Expr::Castable(operand, target) => typing::castable(operand, target, context),
        Expr::Cast(operand, target) => typing::cast(operand, target, context),
        Expr::CastableList(operand, target) => typing::castable_list(operand, target, context),
        Expr::CastList(operand, target) => typing::cast_list(operand, target, context),
        Expr::FunctionCall { .. }
        | Expr::PartialCall { .. }
        | Expr::NamedFunctionRef { .. }
        | Expr::InlineFunction(_)
        | Expr::DynamicCall { .. }
        | Expr::MapConstructor(_)
        | Expr::ArrayConstructor { .. }
        | Expr::Lookup { .. } => functions::evaluate(expression, context),
    }
}

fn for_expression(
    bindings: &[crate::ast::VarBinding],
    body: &Expr,
    context: &DynamicContext,
) -> XPathResult<XpathValue> {
    let Some((first, rest)) = bindings.split_first() else {
        return evaluate(body, context);
    };
    let source = evaluate(&first.value, context)?;
    let mut out = XpathValue::empty();
    for item in source {
        EvaluationLimits::check_deadline()?;
        let scoped = context
            .clone()
            .with_variable_name(first.name.clone(), XpathValue::singleton(item));
        out.append(for_expression(rest, body, &scoped)?);
    }
    Ok(out)
}

fn quantified(
    every: bool,
    bindings: &[crate::ast::VarBinding],
    satisfies: &Expr,
    context: &DynamicContext,
) -> XPathResult<XpathValue> {
    let Some((first, rest)) = bindings.split_first() else {
        return Ok(boolean(effective_boolean_value(&evaluate(
            satisfies, context,
        )?)?));
    };
    let source = evaluate(&first.value, context)?;
    for item in source {
        EvaluationLimits::check_deadline()?;
        let scoped = context
            .clone()
            .with_variable_name(first.name.clone(), XpathValue::singleton(item));
        let verdict = effective_boolean_value(&quantified(every, rest, satisfies, &scoped)?)?;
        if verdict != every {
            return Ok(boolean(verdict));
        }
    }
    Ok(boolean(every))
}

/// `fn:boolean`, the [effective boolean value](https://www.w3.org/TR/xpath-31/#id-ebv).
///
/// XPath 1.0 and XPath 2.0 agree here for every value XPath 1.0 can produce — a node-set is true
/// when it is non-empty, a string when it is non-empty, and a number when it is neither zero nor
/// `NaN` — so one implementation serves both.
///
/// # Errors
///
/// `FORG0006` for a sequence of several atomic values, a singleton of a type with no boolean
/// reading, or a function item.
pub(crate) fn effective_boolean_value(value: &XpathValue) -> XPathResult<bool> {
    value.effective_boolean_value()
}

/// `fn:number`, whose meaning genuinely differs between the versions.
///
/// * **XPath 1.0** converts a node-set by taking the *first node in document order* and reading
///   its string value; a boolean becomes `1` or `0`; anything unparseable becomes `NaN`.
/// * **XPath 2.0 and later** atomize, require at most one value, and cast it to `xs:double`,
///   yielding `NaN` when the cast fails.
///
/// # Errors
///
/// `XPTY0004` under XPath 2.0 semantics when atomization yields more than one value.
pub(crate) fn number_of(value: &XpathValue, context: &DynamicContext) -> XPathResult<f64> {
    if context.version().is_xpath_10() {
        return Ok(compare::number_10(value));
    }
    let Some(atomic) = value.atomize_at_most_one("the argument of fn:number")? else {
        return Ok(f64::NAN);
    };
    Ok(oxixml_fn::numeric::cast_to_double(&atomic))
}

/// Wraps a Rust `bool` as the singleton `xs:boolean` sequence.
pub(crate) fn boolean(value: bool) -> XpathValue {
    XpathValue::from(AtomicValue::boolean(value))
}

/// Atomizes `value`, replacing `xs:untypedAtomic` with `xs:double` — the promotion arithmetic
/// and `fn:number` apply.
pub(crate) fn atomize_for_arithmetic(value: &XpathValue) -> XPathResult<Vec<AtomicValue>> {
    let mut atomized = value.atomize()?;
    for entry in &mut atomized {
        if entry.atomic_type() == AtomicType::UntypedAtomic {
            // The promotion is a *cast*, so a lexical form that is not a number is `FORG0001`,
            // not the `NaN` that `fn:number` would produce.
            *entry = oxixml_fn::numeric::promote_untyped(entry)?;
        }
    }
    Ok(atomized)
}

/// `fn:string` of a whole value.
///
/// XPath 1.0 reads the string value of the *first node in document order* of a node-set; XPath
/// 2.0 and later require a sequence of at most one item.
///
/// # Errors
///
/// `XPTY0004` under XPath 2.0 semantics when the value holds more than one item.
pub(crate) fn string_of(value: &XpathValue, context: &DynamicContext) -> XPathResult<String> {
    if context.version().is_xpath_10() {
        return compare::string_10(value);
    }
    match value.as_slice() {
        [] => Ok(String::new()),
        [item] => item.string_value(),
        _ => Err(crate::error::XPathError::type_error(
            "a string is required, but the value holds more than one item",
        )),
    }
}

/// Whether every item is a node, which the set operators require.
pub(crate) fn require_nodes(
    value: &XpathValue,
    what: &str,
) -> XPathResult<Vec<oxixml_infoset::NodeRef>> {
    let mut nodes = Vec::with_capacity(value.len());
    for item in value.iter() {
        match item.as_item() {
            Some(Item::Node(node)) => nodes.push(node.clone()),
            _ => {
                return Err(crate::error::XPathError::type_error(format!(
                    "{what} requires node sequences, but was given {}",
                    item.describe()
                )));
            }
        }
    }
    Ok(nodes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::StaticContext;
    use crate::version::XPathVersion;
    use oxixml_infoset::{Tree, XdmNode};
    use std::sync::Arc;

    fn context() -> DynamicContext {
        DynamicContext::new(Arc::new(StaticContext::new()))
    }

    #[test]
    fn effective_boolean_values_follow_the_specification() -> XPathResult<()> {
        assert!(!effective_boolean_value(&XpathValue::empty())?);
        assert!(effective_boolean_value(&XpathValue::from(
            AtomicValue::string("x")
        ))?);
        assert!(!effective_boolean_value(&XpathValue::from(
            AtomicValue::string("")
        ))?);
        assert!(!effective_boolean_value(&XpathValue::from(
            AtomicValue::double(f64::NAN)
        ))?);
        let tree = Tree::parse_str("<a/>")?;
        assert!(effective_boolean_value(&XpathValue::from(tree.root()))?);
        Ok(())
    }

    #[test]
    fn fn_number_differs_between_the_versions() -> XPathResult<()> {
        let tree = Tree::parse_str("<r><a>1</a><a>2</a></r>")?;
        let root = tree.document_element().expect("a document element");
        let nodes = XpathValue::from_iter(root.children());

        let ten = context().with_version(XPathVersion::XPath10);
        assert_eq!(number_of(&nodes, &ten)?, 1.0);

        let thirty_one = context().with_version(XPathVersion::XPath31);
        assert!(number_of(&nodes, &thirty_one).is_err());
        Ok(())
    }

    #[test]
    fn no_deadline_is_declared_by_default() {
        assert_eq!(EvaluationLimits::deadline(), None);
        assert!(EvaluationLimits::check_deadline().is_ok());
    }

    #[test]
    fn a_declared_deadline_stops_an_iterating_expression() -> XPathResult<()> {
        // Every one of the evaluator's own per-item loops is a checkpoint. Without a deadline
        // each of these runs to completion however long that takes, which is what leaves an
        // abandoned worker computing for the life of the process.
        for expression in [
            "count(for $i in 1 to 1000 return $i * 2)",
            "some $i in 1 to 1000 satisfies $i = 999",
            "count((1 to 1000)[. mod 7 = 0])",
            "count((1 to 1000) ! (. + 1))",
        ] {
            let compiled = crate::XPath::compile_31(expression)?;
            let context = compiled.context();
            // Nothing is declared until this point: the same expression is evaluated
            // successfully below, outside the declaration.
            {
                let _limits = EvaluationLimits::declare_deadline(
                    Instant::now() - std::time::Duration::from_secs(1),
                );
                let error = compiled
                    .evaluate_in(&context)
                    .expect_err("the deadline has already passed");
                assert_eq!(error.code(), Some(XPathErrorCode::Xpdy0130));
                assert!(
                    error.to_string().contains("deadline"),
                    "the message names the limit: {error}"
                );
            }
            compiled
                .evaluate_in(&context)
                .expect("no deadline is in force once the declaration is dropped");
        }
        Ok(())
    }

    #[test]
    fn a_deadline_still_ahead_lets_the_evaluation_finish() -> XPathResult<()> {
        let compiled = crate::XPath::compile_31("count(for $i in 1 to 1000 return $i * 2)")?;
        let context = compiled.context();
        let _limits = EvaluationLimits::declare_deadline(
            Instant::now() + std::time::Duration::from_secs(600),
        );
        assert_eq!(compiled.evaluate_to_string_in(&context)?, "1000");
        Ok(())
    }

    #[test]
    fn a_declaration_restores_the_deadline_it_found() {
        let deadline = Instant::now() + std::time::Duration::from_secs(600);
        {
            let _outer = EvaluationLimits::declare_deadline(deadline);
            assert_eq!(EvaluationLimits::deadline(), Some(deadline));
            assert_eq!(
                oxixml_fn::function::CallLimits::deadline(),
                Some(deadline),
                "the function library's own budget is told the same deadline"
            );
            {
                // A nested stack declaration must not silently drop the deadline.
                let _inner = EvaluationLimits::declare(8 * 1024 * 1024);
                assert_eq!(EvaluationLimits::deadline(), Some(deadline));
            }
            assert_eq!(EvaluationLimits::deadline(), Some(deadline));
        }
        assert_eq!(EvaluationLimits::deadline(), None);
        assert_eq!(oxixml_fn::function::CallLimits::deadline(), None);
    }
}
