//! [`FunctionItem`] — the callable value that XPath 3.1's higher-order functions pass around.
//!
//! [XDM 3.1](https://www.w3.org/TR/xpath-datamodel-31/#dt-function-item) makes a *function
//! item* a kind of item, and [`Item`] already models the two the data
//! model itself defines, [`XdmMap`] and [`XdmArray`]. A *general* function item, though, is a
//! closure over an evaluator's expression tree, which the data-model crate deliberately does
//! not know about. This crate supplies the missing piece: a `FunctionItem` is any Rust closure
//! from argument sequences to a result sequence, which is exactly the interface `fn:for-each`,
//! `fn:filter`, `fn:fold-left`, `map:for-each` and friends need.
//!
//! Maps and arrays convert into function items with [`FunctionItem::from_map`] and
//! [`FunctionItem::from_array`], which is what makes `map:get($m, $k)` and `$m($k)` the same
//! operation.

use crate::context::FnContext;
use crate::error::{FnError, FnErrorCode, FnResult};
use oxixml_infoset::{AtomicData, AtomicValue, ExpandedName, Item, Sequence, XdmArray, XdmMap};
use std::cell::Cell;
use std::fmt;
use std::sync::Arc;
use std::time::Instant;

/// How deeply function-item calls may nest on a thread whose stack nothing has declared.
///
/// A function item's implementation is an opaque closure: a higher-order function such as
/// `fn:fold-left` calls it, and an evaluator that builds one over its own expression tree can
/// have it call back in. Nothing in the shape of the abstraction stops that from recursing until
/// the thread's stack is gone, which is a process abort rather than an error a host can catch,
/// so the calls run out of a budget instead. Exceeding it raises
/// [`FnErrorCode::Xpdy0130`], the code reserved for an implementation-defined limit.
///
/// This figure is the *default*, and a floor: a host that has told the library how much stack the
/// thread really has, with [`CallLimits`], gets a ceiling scaled to it, because a caller that
/// deliberately runs a query on a 512 MiB thread has said it expects deep recursion.
pub const DEFAULT_MAX_CALL_DEPTH: usize = 512;

/// The native stack a call budget assumes the thread it runs on has, when nothing declares one.
///
/// `std::thread::spawn` gives a new thread 2 MiB unless it is told otherwise, which is what a
/// thread pool or an async executor's worker hands a library, so it is the figure to stay inside
/// by default.
pub const DEFAULT_STACK_SIZE: usize = 2 * 1024 * 1024;

/// The stack one call is taken to cost, for the purpose of scaling [`DEFAULT_MAX_CALL_DEPTH`] to
/// a declared stack.
///
/// It is deliberately an under-estimate — an unoptimized build spends several times this — so
/// that the derived count does not become the binding limit on a thread that was sized for deep
/// recursion.
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
    /// How many function-item calls are open.
    calls: usize,
    /// The stack of the thread the calls are running on.
    stack_size: usize,
    /// How many calls may nest.
    max_calls: usize,
    /// The monotonic instant past which a call refuses to start, when one has been declared.
    deadline: Option<Instant>,
}

impl Budget {
    const DEFAULT: Self = Self {
        calls: 0,
        stack_size: DEFAULT_STACK_SIZE,
        max_calls: call_ceiling(DEFAULT_STACK_SIZE),
        deadline: None,
    };
}

thread_local! {
    /// The budget belongs to the thread rather than to a [`FnContext`], because a function item's
    /// implementation is free to call against a context of its own and a budget any of those
    /// could reset would not bound the stack it is there to protect.
    static BUDGET: Cell<Budget> = const { Cell::new(Budget::DEFAULT) };
}

/// Declares how much native stack function-item calls on this thread may spend, restoring the
/// previous declaration when it is dropped.
///
/// The ceiling on nesting is a count of calls, which is what a query is written in and what a
/// `try`/`catch` can be told about. It defaults to [`DEFAULT_MAX_CALL_DEPTH`], which is the right
/// figure for the 2 MiB a spawned thread has, and scales with a larger declared stack: a host
/// that runs a deeply recursive query on a thread it sized for the purpose declares that stack
/// once, and the ceiling follows. Declaring more stack than the thread really has would trade a
/// bounded error for a process abort, so it is never guessed on the caller's behalf.
///
/// A declaration nests: a call made from inside another declaration leaves the outer figures in
/// place when it finishes.
///
/// # Examples
///
/// ```
/// use oxixml_fn::function::{CallLimits, DEFAULT_MAX_CALL_DEPTH};
///
/// assert_eq!(CallLimits::max_call_depth(), DEFAULT_MAX_CALL_DEPTH);
/// let _limits = CallLimits::declare(512 * 1024 * 1024);
/// assert!(CallLimits::max_call_depth() >= 5_000);
/// ```
#[derive(Debug)]
pub struct CallLimits {
    previous_stack_size: usize,
    previous_max_calls: usize,
    previous_deadline: Option<Instant>,
}

impl CallLimits {
    /// Declares `stack_size` as the native stack of the thread the calls run on, and takes the
    /// call ceiling from it — never below [`DEFAULT_MAX_CALL_DEPTH`].
    #[must_use]
    pub fn declare(stack_size: usize) -> Self {
        Self::install(stack_size, call_ceiling(stack_size), Self::deadline())
    }

    /// The same, with the call ceiling given explicitly rather than derived.
    #[must_use]
    pub fn declare_with_max_call_depth(stack_size: usize, max_call_depth: usize) -> Self {
        Self::install(stack_size, max_call_depth, Self::deadline())
    }

    /// Declares a wall-clock `deadline` past which a call refuses to start, leaving the stack
    /// and call figures as they are.
    ///
    /// A call that begins after `deadline` has passed raises [`FnErrorCode::Xpdy0130`] instead
    /// of running, so a host that has abandoned a computation — a test harness that has already
    /// declared a case timed out, a request whose client has gone — stops paying for it at the
    /// next call boundary rather than at the end of the work. Nothing is interrupted *mid*-call:
    /// see [`CallLimits::check_deadline`] for what a long-running implementation can do about
    /// that, and note that a single builtin that runs for longer than the deadline still runs to
    /// completion.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_fn::function::CallLimits;
    /// use std::time::{Duration, Instant};
    ///
    /// assert!(CallLimits::deadline().is_none());
    /// let _limits = CallLimits::declare_deadline(Instant::now() + Duration::from_secs(30));
    /// assert!(CallLimits::deadline().is_some());
    /// ```
    #[must_use]
    pub fn declare_deadline(deadline: Instant) -> Self {
        let (stack_size, max_calls) =
            BUDGET.with(|cell| (cell.get().stack_size, cell.get().max_calls));
        Self::install(stack_size, max_calls, Some(deadline))
    }

    /// Declares the stack, the call ceiling and the deadline in one go.
    ///
    /// An engine that keeps a budget of its own — [`oxixml-xpath`][xp]'s `EvaluationLimits`
    /// mirrors every figure it is given into this one — has all three in hand at once and needs
    /// them installed as a single nesting level, because two overlapping declarations would
    /// restore each other's figures on the way out.
    ///
    /// [xp]: https://docs.rs/oxixml-xpath
    #[must_use]
    pub fn declare_with(
        stack_size: usize,
        max_call_depth: usize,
        deadline: Option<Instant>,
    ) -> Self {
        Self::install(stack_size, max_call_depth, deadline)
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
    /// This is the poll a function implementation that loops over a large input can make from
    /// inside its own loop, so that the budget bounds it too. Reading the thread's declaration
    /// is a single thread-local load, and the clock is only consulted when a deadline is
    /// actually set, so the call costs nothing measurable on the ordinary path.
    ///
    /// # Errors
    ///
    /// [`FnErrorCode::Xpdy0130`] once the deadline has passed.
    pub fn check_deadline() -> FnResult<()> {
        match Self::deadline() {
            Some(deadline) if Instant::now() >= deadline => Err(deadline_exceeded()),
            _ => Ok(()),
        }
    }

    fn install(stack_size: usize, max_calls: usize, deadline: Option<Instant>) -> Self {
        BUDGET.with(|cell| {
            let mut budget = cell.get();
            let previous = Self {
                previous_stack_size: budget.stack_size,
                previous_max_calls: budget.max_calls,
                previous_deadline: budget.deadline,
            };
            budget.stack_size = stack_size;
            budget.max_calls = max_calls;
            budget.deadline = deadline;
            cell.set(budget);
            previous
        })
    }
}

impl Drop for CallLimits {
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

/// The error a call that starts past the declared deadline raises.
fn deadline_exceeded() -> FnError {
    FnError::new(
        FnErrorCode::Xpdy0130,
        "the evaluation has run past the deadline declared for this thread with `CallLimits`, \
         which is this implementation's limit on how long it may run; the deadline is \
         configurable and is not set unless a host declares one"
            .to_owned(),
    )
}

/// Holds one level of call depth for as long as it is alive, so the count is right however the
/// call returns.
struct DepthGuard;

impl DepthGuard {
    fn enter() -> FnResult<Self> {
        let budget = BUDGET.with(|cell| {
            let mut budget = cell.get();
            budget.calls += 1;
            cell.set(budget);
            budget
        });
        if budget.calls > budget.max_calls {
            return Err(Self::refuse(FnError::new(
                FnErrorCode::Xpdy0130,
                format!(
                    "function calls nest more than {} deep, which is this implementation's limit \
                     on a thread of {} bytes; a query that recurses this deeply needs a larger \
                     stack, declared with `CallLimits`",
                    budget.max_calls, budget.stack_size
                ),
            )));
        }
        // The budget is already in hand, so a thread with no declared deadline pays one branch
        // for this and never reads the clock.
        if let Some(deadline) = budget.deadline {
            if Instant::now() >= deadline {
                return Err(Self::refuse(deadline_exceeded()));
            }
        }
        Ok(Self)
    }

    /// Gives back the level this guard took but will not hold, and returns `error` unchanged.
    fn refuse(error: FnError) -> FnError {
        BUDGET.with(|cell| {
            let mut budget = cell.get();
            budget.calls = budget.calls.saturating_sub(1);
            cell.set(budget);
        });
        error
    }
}

impl Drop for DepthGuard {
    fn drop(&mut self) {
        BUDGET.with(|cell| {
            let mut budget = cell.get();
            budget.calls = budget.calls.saturating_sub(1);
            cell.set(budget);
        });
    }
}

type Implementation = Arc<dyn Fn(&FnContext, &[Sequence]) -> FnResult<Sequence> + Send + Sync>;

/// A callable function item of a fixed arity.
///
/// # Examples
///
/// ```
/// use oxixml_fn::{FnContext, FunctionItem};
/// use oxixml_infoset::{AtomicValue, ExpandedName, Sequence};
///
/// let double = FunctionItem::anonymous(1, |_context, arguments| {
///     let value = arguments[0]
///         .first()
///         .and_then(|item| item.as_atomic())
///         .and_then(|value| value.as_f64())
///         .unwrap_or_default();
///     Ok(Sequence::from(AtomicValue::double(value * 2.0)))
/// });
///
/// let context = FnContext::new();
/// let result = double.call(&context, &[Sequence::from(AtomicValue::integer(21_i64))])?;
/// assert_eq!(result.first().map(ToString::to_string).as_deref(), Some("42"));
/// assert_eq!(double.arity(), 1);
/// # Ok::<(), oxixml_fn::FnError>(())
/// ```
#[derive(Clone)]
pub struct FunctionItem {
    name: Option<ExpandedName>,
    arity: usize,
    implementation: Implementation,
}

impl FunctionItem {
    /// Builds a named function item.
    pub fn new(
        name: ExpandedName,
        arity: usize,
        implementation: impl Fn(&FnContext, &[Sequence]) -> FnResult<Sequence> + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: Some(name),
            arity,
            implementation: Arc::new(implementation),
        }
    }

    /// Builds an anonymous function item, the shape an inline function expression produces.
    pub fn anonymous(
        arity: usize,
        implementation: impl Fn(&FnContext, &[Sequence]) -> FnResult<Sequence> + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: None,
            arity,
            implementation: Arc::new(implementation),
        }
    }

    /// `fn:function-name`: the function's name, or `None` when it is anonymous.
    #[must_use]
    pub const fn name(&self) -> Option<&ExpandedName> {
        self.name.as_ref()
    }

    /// `fn:function-arity`: the number of arguments the function accepts.
    #[must_use]
    pub const fn arity(&self) -> usize {
        self.arity
    }

    /// Invokes the function.
    ///
    /// # Errors
    ///
    /// [`FnErrorCode::Xpty0004`] when `arguments.len()` is not [`FunctionItem::arity`],
    /// [`FnErrorCode::Xpdy0130`] when the call nests past what [`CallLimits`] allows, plus
    /// whatever the implementation itself raises.
    pub fn call(&self, context: &FnContext, arguments: &[Sequence]) -> FnResult<Sequence> {
        let _guard = DepthGuard::enter()?;
        if arguments.len() != self.arity {
            return Err(FnError::new(
                FnErrorCode::Xpty0004,
                format!(
                    "{} expects {} argument(s) but was given {}",
                    self.describe(),
                    self.arity,
                    arguments.len()
                ),
            ));
        }
        (self.implementation)(context, arguments)
    }

    /// Wraps a map as the arity-one function item `$key -> $value` that XDM says it is.
    ///
    /// A key that is not present yields the empty sequence, exactly as `map:get` does.
    #[must_use]
    pub fn from_map(map: XdmMap) -> Self {
        Self::anonymous(1, move |_context, arguments| {
            let key = single_atomic(&arguments[0], "a map key")?;
            Ok(map.get(&key).cloned().unwrap_or_default())
        })
    }

    /// Wraps an array as the arity-one function item `$position -> $member` that XDM says it is.
    ///
    /// The position is one-based; an out-of-range position raises `FOAY0001`.
    #[must_use]
    pub fn from_array(array: XdmArray) -> Self {
        Self::anonymous(1, move |_context, arguments| {
            let index = single_atomic(&arguments[0], "an array position")?;
            let position = match index.data() {
                AtomicData::Integer(value) => usize::try_from(i64::from(*value)).ok(),
                _ => None,
            }
            .ok_or_else(|| {
                FnError::new(
                    FnErrorCode::Xpty0004,
                    "an array subscript must be a positive xs:integer",
                )
            })?;
            Ok(array.get(position)?.clone())
        })
    }

    /// A short human-readable rendering — `fn:abs#1` for a named function, a description of the
    /// arity for an anonymous one — used in diagnostics.
    #[must_use]
    pub fn describe(&self) -> String {
        match &self.name {
            Some(name) => format!("{name}#{}", self.arity),
            None => format!("an anonymous function of arity {}", self.arity),
        }
    }
}

fn single_atomic(sequence: &Sequence, what: &str) -> FnResult<AtomicValue> {
    match sequence.as_slice() {
        [Item::Atomic(value)] => Ok(value.clone()),
        _ => Err(FnError::new(
            FnErrorCode::Xpty0004,
            format!("{what} must be exactly one atomic value"),
        )),
    }
}

impl fmt::Debug for FunctionItem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FunctionItem")
            .field("name", &self.name)
            .field("arity", &self.arity)
            .finish_non_exhaustive()
    }
}

impl fmt::Display for FunctionItem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.describe())
    }
}

impl From<XdmMap> for FunctionItem {
    fn from(map: XdmMap) -> Self {
        Self::from_map(map)
    }
}

impl From<XdmArray> for FunctionItem {
    fn from(array: XdmArray) -> Self {
        Self::from_array(array)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxixml_infoset::AtomicValue;

    #[test]
    fn arity_mismatch_is_a_type_error() {
        let identity = FunctionItem::anonymous(1, |_, args| Ok(args[0].clone()));
        let context = FnContext::new();
        let error = identity.call(&context, &[]).expect_err("wrong arity");
        assert!(error.is(FnErrorCode::Xpty0004));
    }

    #[test]
    fn a_map_is_a_function_from_key_to_value() -> FnResult<()> {
        let map = XdmMap::new().put(
            AtomicValue::string("k"),
            Sequence::from(AtomicValue::integer(7_i64)),
        );
        let function = FunctionItem::from_map(map);
        let context = FnContext::new();
        let hit = function.call(&context, &[Sequence::from(AtomicValue::string("k"))])?;
        assert_eq!(hit.first().map(ToString::to_string).as_deref(), Some("7"));
        let miss = function.call(&context, &[Sequence::from(AtomicValue::string("z"))])?;
        assert!(miss.is_empty());
        Ok(())
    }

    #[test]
    fn an_array_is_a_function_from_position_to_member() -> FnResult<()> {
        let array = XdmArray::from_iter([AtomicValue::string("a"), AtomicValue::string("b")]);
        let function = FunctionItem::from_array(array);
        let context = FnContext::new();
        let second = function.call(&context, &[Sequence::from(AtomicValue::integer(2_i64))])?;
        assert_eq!(
            second.first().map(ToString::to_string).as_deref(),
            Some("b")
        );
        let error = function
            .call(&context, &[Sequence::from(AtomicValue::integer(3_i64))])
            .expect_err("out of bounds");
        assert!(error.is(FnErrorCode::Foay0001));
        Ok(())
    }

    #[test]
    fn names_and_arities_are_reported() {
        let named = FunctionItem::new(
            ExpandedName::new(Some("http://www.w3.org/2005/xpath-functions"), "abs"),
            1,
            |_, args| Ok(args[0].clone()),
        );
        assert_eq!(named.arity(), 1);
        assert_eq!(named.name().map(ExpandedName::local_name), Some("abs"));
        assert!(
            FunctionItem::anonymous(0, |_, _| Ok(Sequence::empty()))
                .name()
                .is_none()
        );
    }

    /// Nests `depth` invocations of a function item, each entered through
    /// [`FunctionItem::call`] from inside the one before it, which is exactly what the budget
    /// counts.
    fn nest(context: &FnContext, depth: usize) -> FnResult<()> {
        let step = FunctionItem::anonymous(0, move |context, _| {
            if depth > 1 {
                nest(context, depth - 1)?;
            }
            Ok(Sequence::empty())
        });
        step.call(context, &[])?;
        Ok(())
    }

    #[test]
    fn the_default_ceiling_is_the_declared_default() {
        assert_eq!(CallLimits::max_call_depth(), DEFAULT_MAX_CALL_DEPTH);
        assert_eq!(CallLimits::stack_size(), DEFAULT_STACK_SIZE);
        assert_eq!(call_ceiling(DEFAULT_STACK_SIZE), DEFAULT_MAX_CALL_DEPTH);
        assert_eq!(
            call_ceiling(0),
            DEFAULT_MAX_CALL_DEPTH,
            "the default is a floor, not just a default"
        );
        assert_eq!(
            call_ceiling(512 * 1024 * 1024),
            512 * 1024 * 1024 / CALL_STACK_ESTIMATE
        );
    }

    #[test]
    fn calls_nest_to_the_ceiling_and_no_further() -> FnResult<()> {
        let context = FnContext::new();
        nest(&context, DEFAULT_MAX_CALL_DEPTH)?;
        let error = nest(&context, DEFAULT_MAX_CALL_DEPTH + 1).expect_err("past the ceiling");
        assert!(error.is(FnErrorCode::Xpdy0130));
        // The failed calls gave their levels back, so the next evaluation starts from zero.
        nest(&context, DEFAULT_MAX_CALL_DEPTH)?;
        Ok(())
    }

    /// A declaration raises the ceiling for as long as it is alive and no longer, and the
    /// ceiling it derives never falls below the default.
    #[test]
    fn a_declaration_raises_the_ceiling_and_nests_when_dropped() {
        {
            let _limits = CallLimits::declare(512 * 1024 * 1024);
            assert!(CallLimits::max_call_depth() >= 5_000);
            assert_eq!(CallLimits::stack_size(), 512 * 1024 * 1024);
            {
                let _inner = CallLimits::declare_with_max_call_depth(1024, 4);
                assert_eq!(CallLimits::max_call_depth(), 4);
            }
            assert!(CallLimits::max_call_depth() >= 5_000);
        }
        assert_eq!(CallLimits::max_call_depth(), DEFAULT_MAX_CALL_DEPTH);
        assert_eq!(CallLimits::stack_size(), DEFAULT_STACK_SIZE);
        let _small = CallLimits::declare(64 * 1024);
        assert_eq!(
            CallLimits::max_call_depth(),
            DEFAULT_MAX_CALL_DEPTH,
            "a small declared stack does not lower the default"
        );
    }

    /// The shape a recursive query has — some thousands of user-visible calls on a thread the
    /// host sized for them. Without a declaration the default ceiling stands and the recursion
    /// is a bounded `XPDY0130`; with one it runs.
    #[test]
    fn a_declared_stack_carries_thousands_of_nested_calls() {
        const STACK: usize = 64 * 1024 * 1024;
        const DEPTH: usize = 5_000;

        let worker = std::thread::Builder::new()
            .stack_size(STACK)
            .spawn(|| {
                let context = FnContext::new();
                let error = nest(&context, DEPTH).expect_err("past the default ceiling");
                assert!(error.is(FnErrorCode::Xpdy0130));

                let _limits = CallLimits::declare(STACK);
                assert!(CallLimits::max_call_depth() >= DEPTH);
                nest(&context, DEPTH).expect("the declared stack has room for the recursion");
            })
            .expect("a worker thread");
        worker.join().expect("the worker finished");
    }

    #[test]
    fn no_deadline_is_declared_by_default() {
        assert_eq!(CallLimits::deadline(), None);
        assert!(CallLimits::check_deadline().is_ok());
    }

    /// A call that begins after the deadline refuses to run, and gives its level back so that
    /// the count is still right for whatever the host does next.
    #[test]
    fn a_call_past_the_deadline_is_refused() {
        let context = FnContext::new();
        {
            let _limits =
                CallLimits::declare_deadline(Instant::now() - std::time::Duration::from_secs(1));
            let error = nest(&context, 1).expect_err("the deadline has already passed");
            assert!(error.is(FnErrorCode::Xpdy0130));
            assert!(
                error.to_string().contains("deadline"),
                "the message names the limit: {error}"
            );
            assert!(CallLimits::check_deadline().is_err());
        }
        assert_eq!(CallLimits::deadline(), None);
        nest(&context, 1).expect("the declaration is gone");
    }

    #[test]
    fn a_deadline_declaration_nests_like_the_others() {
        let deadline = Instant::now() + std::time::Duration::from_secs(600);
        {
            let _outer = CallLimits::declare_deadline(deadline);
            assert_eq!(CallLimits::deadline(), Some(deadline));
            assert_eq!(CallLimits::stack_size(), DEFAULT_STACK_SIZE);
            {
                // A stack declaration made inside one keeps the deadline it found.
                let _inner = CallLimits::declare(64 * 1024 * 1024);
                assert_eq!(CallLimits::deadline(), Some(deadline));
                assert_eq!(CallLimits::stack_size(), 64 * 1024 * 1024);
            }
            assert_eq!(CallLimits::deadline(), Some(deadline));
            assert_eq!(CallLimits::stack_size(), DEFAULT_STACK_SIZE);
        }
        assert_eq!(CallLimits::deadline(), None);
    }
}
