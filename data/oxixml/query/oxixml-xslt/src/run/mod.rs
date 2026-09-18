//! The transformation engine: the run-time half of the crate.

mod analyze;
mod context_item;
mod emit;
mod exec;
pub(crate) mod functions;
mod group;
mod keys;
mod maps;
mod number;
mod populated;
mod sort;
mod transform;

pub(crate) use context_item::apply_global_context_item;
pub(crate) use emit::Emitter;

use crate::compile::{Instruction, ModeKey, StylesheetIr};
use crate::error::{XsltError, XsltResult};
use crate::output::OutputProperties;
use crate::resolve::{DocumentStripper, Resolver, XPathResolver};
use crate::run::exec::Flow;
use crate::tree::ResultTree;
use oxixml_fn::FnContext;
use oxixml_infoset::{AtomicValue, ExpandedName, Item, NodeKind, NodeRef, Sequence, XdmNode};
use oxixml_xpath::{CompiledXPath, DocumentResolver, DynamicContext, Focus, XpathValue};
use oxixml_xsd::TimezoneOffset;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

thread_local! {
    /// The transformation whose XSLT-specific functions — `current()`, `key()`,
    /// `current-group()` — are in scope on this thread.
    ///
    /// The function library is a set of plain closures with no place to hang a transformation
    /// on, so the engine publishes itself here for the duration of a run. The value is only
    /// ever read while the engine is on the stack.
    static ACTIVE: RefCell<Vec<Rc<Runtime>>> = const { RefCell::new(Vec::new()) };
}

/// The transformation currently running on this thread.
pub(crate) fn current_runtime() -> Option<Rc<Runtime>> {
    ACTIVE.with(|stack| stack.borrow().last().map(Rc::clone))
}

/// Runs `body` with `runtime` published as the active transformation.
pub(crate) fn with_runtime<R>(runtime: Rc<Runtime>, body: impl FnOnce() -> R) -> R {
    ACTIVE.with(|stack| stack.borrow_mut().push(runtime));
    let result = body();
    ACTIVE.with(|stack| {
        stack.borrow_mut().pop();
    });
    result
}

/// A chain of variable bindings, cheap to extend and to clone.
#[derive(Debug, Clone, Default)]
pub(crate) struct Bindings(Option<Rc<Binding>>);

#[derive(Debug)]
struct Binding {
    name: ExpandedName,
    value: XpathValue,
    parent: Option<Rc<Binding>>,
}

impl Bindings {
    /// Extends the chain, shadowing any earlier binding of the same name.
    pub(crate) fn bind(&self, name: ExpandedName, value: XpathValue) -> Self {
        Self(Some(Rc::new(Binding {
            name,
            value,
            parent: self.0.clone(),
        })))
    }

    /// The value bound to `name`, if any.
    pub(crate) fn get(&self, name: &ExpandedName) -> Option<&XpathValue> {
        let mut current = self.0.as_deref();
        while let Some(binding) = current {
            if &binding.name == name {
                return Some(&binding.value);
            }
            current = binding.parent.as_deref();
        }
        None
    }

    /// Every binding, innermost first, with shadowed names dropped.
    pub(crate) fn visible(&self) -> Vec<(&ExpandedName, &XpathValue)> {
        let mut out: Vec<(&ExpandedName, &XpathValue)> = Vec::new();
        let mut current = self.0.as_deref();
        while let Some(binding) = current {
            if !out.iter().any(|(name, _)| *name == &binding.name) {
                out.push((&binding.name, &binding.value));
            }
            current = binding.parent.as_deref();
        }
        out
    }
}

/// The evaluation context one instruction sees.
#[derive(Debug, Clone)]
pub(crate) struct Context {
    /// The context item — the *current node* when it is a node.
    pub(crate) item: Option<Item>,
    /// `position()`.
    pub(crate) position: usize,
    /// `last()`.
    pub(crate) size: usize,
    /// The variables in scope.
    pub(crate) bindings: Bindings,
    /// The tunnel parameters in scope.
    pub(crate) tunnel: Bindings,
    /// The mode in force.
    pub(crate) mode: ModeKey,
    /// The template rule currently being applied, for `xsl:next-match` and `xsl:apply-imports`.
    pub(crate) rule: Option<usize>,
}

impl Context {
    /// A context with `item` in focus and nothing bound.
    pub(crate) fn new(item: Option<Item>) -> Self {
        Self {
            item,
            position: 1,
            size: 1,
            bindings: Bindings::default(),
            tunnel: Bindings::default(),
            mode: ModeKey::Default,
            rule: None,
        }
    }

    /// The same context with a new focus.
    ///
    /// [XSLT 3.0 §6.7](https://www.w3.org/TR/xslt-30/#current-template-rule): "The current
    /// template rule is cleared (becomes absent) by any instruction that evaluates an operand
    /// with changed focus" — the spec then lists `xsl:for-each`, `xsl:for-each-group`,
    /// `xsl:analyze-string`, `xsl:iterate`, `xsl:sort`, `xsl:key` and `xsl:copy` with a
    /// `select`, which are exactly this constructor's callers. Changing the focus and clearing
    /// the rule therefore happen together, so that an `xsl:next-match` written inside any of
    /// them is `XTDE0560` rather than a recursive re-match of the new focus.
    pub(crate) fn with_focus(&self, item: Item, position: usize, size: usize) -> Self {
        let mut next = self.clone();
        next.item = Some(item);
        next.position = position;
        next.size = size;
        next.rule = None;
        next
    }
}

pub(crate) use keys::KeyIndex;

/// The mutable state one transformation carries.
/// The memoization key for one `cache="yes"` `xsl:function` call: the whole argument list, each
/// argument the sequence of atomic values it holds. A call with a node, map, array or function
/// argument is never cached, so this always describes the call completely.
pub(crate) type FunctionCacheKey = Vec<Vec<AtomicValue>>;

/// The absolute URIs a transformation has read a document from.
///
/// [XSLT 3.0 §23.1](https://www.w3.org/TR/xslt-30/#result-document) forbids writing a final
/// result tree to a URI the same transformation also reads (`XTDE1500`), so every resolution
/// `fn:doc` and `fn:document` perform is recorded here. A [`oxixml_xpath::DocumentResolver`]
/// has to be `Send + Sync`, hence the mutex rather than a `RefCell`.
#[derive(Debug, Default)]
pub(crate) struct DocumentLog(std::sync::Mutex<Vec<String>>);

impl DocumentLog {
    /// Records that `uri` was read.
    fn record(&self, uri: &str) {
        if let Ok(mut read) = self.0.lock()
            && !read.iter().any(|seen| seen == uri)
        {
            read.push(uri.to_owned());
        }
    }

    /// Whether `uri` has been read. A poisoned lock reports "not read", which loses an error
    /// report rather than inventing one.
    pub(crate) fn was_read(&self, uri: &str) -> bool {
        self.0
            .lock()
            .is_ok_and(|read| read.iter().any(|seen| seen == uri))
    }
}

/// A [`oxixml_xpath::DocumentResolver`] that notes every URI it is asked for in a
/// [`DocumentLog`] before delegating.
#[derive(Debug)]
struct RecordingResolver {
    inner: Arc<dyn DocumentResolver>,
    resolver: Arc<dyn Resolver>,
    base: Option<String>,
    log: Arc<DocumentLog>,
}

impl DocumentResolver for RecordingResolver {
    fn document(&self, uri: &str) -> Option<NodeRef> {
        self.log
            .record(&self.resolver.resolve_uri(uri, self.base.as_deref()));
        self.inner.document(uri)
    }

    fn collection(&self, uri: Option<&str>) -> Option<Vec<NodeRef>> {
        self.inner.collection(uri)
    }

    fn collection_items(&self, uri: Option<&str>) -> Option<Vec<Item>> {
        self.inner.collection_items(uri)
    }

    fn unparsed_text(&self, uri: &str, encoding: Option<&str>) -> Option<String> {
        self.log
            .record(&self.resolver.resolve_uri(uri, self.base.as_deref()));
        self.inner.unparsed_text(uri, encoding)
    }
}

pub(crate) struct Runtime {
    pub(crate) ir: Arc<StylesheetIr>,
    pub(crate) resolver: Option<Arc<dyn Resolver>>,
    pub(crate) xpath_resolver: Option<Arc<dyn DocumentResolver>>,
    /// Applies `xsl:strip-space` to documents loaded during the run.
    pub(crate) stripper: Arc<DocumentStripper>,
    pub(crate) fn_context: FnContext,
    /// The static context pattern predicates are compiled and evaluated against, built once so
    /// that matching a rule does not rebuild the function library.
    pub(crate) pattern_static: Arc<oxixml_xpath::StaticContext>,
    pub(crate) base_uri: Option<String>,
    /// The base output URI the invocation declared — see
    /// [`crate::Parameters::with_base_output_uri`].
    base_output_uri: RefCell<Option<String>>,
    /// The root of the tree the *global context item* belongs to, recorded when the global
    /// variables were evaluated. A node the invocation supplies as its initial match selection
    /// is space-stripped before the rules are applied to it, and this is what tells a node that
    /// still needs stripping from one that has already been through it.
    pub(crate) global_root: RefCell<Option<NodeRef>>,
    /// The stripped counterpart of every tree an initial match selection reached into, so that
    /// a selection of many nodes from one document strips that document once.
    pub(crate) stripped_selections: RefCell<Vec<(NodeRef, NodeRef)>>,
    pub(crate) globals: RefCell<Vec<(ExpandedName, XpathValue)>>,
    /// How many times [`Runtime::globals`] has changed, against which [`Runtime::globals_context`]
    /// is judged stale.
    ///
    /// The list is filled by a fixpoint over the declarations, so it grows — and, where a
    /// supplied parameter is replaced by a declaration of the same name, changes — while
    /// expressions are already being evaluated against it.
    pub(crate) globals_generation: std::cell::Cell<u64>,
    /// A dynamic context with one binding per global variable, built once per generation and
    /// shared by every expression and every pattern.
    ///
    /// Binding the globals afresh for each evaluation is the obvious reading and is what this
    /// used to do — but a global's value is an ordinary sequence, and a stylesheet that reads a
    /// file into one holds tens of thousands of items in it, every one of which was then copied
    /// per expression. The binding chain is immutable and shared, so one copy serves the whole
    /// transformation; each site retargets the copy at its own static context and XPath version,
    /// both of which are properties of the expression rather than of the bindings.
    pub(crate) globals_context: RefCell<Option<(u64, DynamicContext)>>,
    pub(crate) current: RefCell<Vec<Option<Item>>>,
    /// The static context of every expression being evaluated, innermost last.
    ///
    /// `fn:function-available`, `fn:element-available` and `fn:type-available` expand the
    /// lexical name they are handed "using the namespace declarations in scope for the
    /// expression" ([XSLT 3.0 §23.1.1](https://www.w3.org/TR/xslt-30/#func-function-available)),
    /// which is a property of the call site rather than of the transformation — two expressions
    /// in one stylesheet may bind the same prefix differently. A registered function is handed
    /// nothing but its arguments, so the evaluator publishes the static context here for the
    /// duration of the call.
    pub(crate) static_contexts: RefCell<Vec<Arc<oxixml_xpath::StaticContext>>>,
    /// The `(current group, current grouping key)` of every `xsl:for-each-group` in force,
    /// innermost last. The key is absent when the grouping was positional.
    pub(crate) groups: RefCell<Vec<(Vec<Item>, Option<Sequence>)>>,
    pub(crate) regex_groups: RefCell<Vec<Vec<String>>>,
    pub(crate) keys: RefCell<HashMap<(ExpandedName, u64), Rc<KeyIndex>>>,
    /// The key indexes currently being built, so that a key whose pattern or `use` expression
    /// calls the key itself is reported rather than recursing without end.
    pub(crate) keys_in_progress: RefCell<Vec<(ExpandedName, u64)>>,
    /// Memoized results of `xsl:function`s declared `cache="yes"`, one map per `(name, arity)`.
    ///
    /// The table lives for a single transformation and is discarded with the [`Runtime`], which
    /// is what makes the memoization sound: everything a stylesheet function may read beyond its
    /// arguments — `current-dateTime()`, a global variable, `doc()`, node identity — is stable
    /// for the duration of one transformation, so a value cached under an argument list stays the
    /// value the body would recompute for it. Only calls whose every argument is a sequence of
    /// atomic values are cached, so a [`FunctionCacheKey`] captures the call completely.
    pub(crate) function_cache:
        RefCell<HashMap<(ExpandedName, usize), HashMap<FunctionCacheKey, Sequence>>>,
    pub(crate) messages: RefCell<Vec<String>>,
    pub(crate) result_documents: RefCell<Vec<(String, String)>>,
    /// Serialization properties an `xsl:result-document` without an `href` imposed on the
    /// principal result, which [XSLT 3.0 §23.1](https://www.w3.org/TR/xslt-30/#result-document)
    /// says apply to that destination.
    pub(crate) principal_output: RefCell<Option<std::sync::Arc<OutputProperties>>>,
    /// The final result tree an `xsl:result-document` without an `href` built for the principal
    /// result destination, which [XSLT 3.0 §23.1](https://www.w3.org/TR/xslt-30/#result-document)
    /// makes a result tree of its own rather than content of the tree in force around it.
    pub(crate) principal_result: RefCell<Option<ResultTree>>,
    /// Whether an `xsl:result-document` has already claimed the principal result destination,
    /// so that a second one is the `XTDE1490` two-trees-one-URI error.
    pub(crate) principal_claimed: std::cell::Cell<bool>,
    /// The absolute URI of every document this transformation has *read*, against which a
    /// destination it later writes is checked for `XTDE1500`.
    pub(crate) documents_read: Arc<DocumentLog>,
    /// The resolved `href` of every `xsl:result-document` instruction currently being
    /// evaluated, innermost last — what `fn:current-output-uri()` reports while it is not in
    /// *temporary output state*. [XSLT 3.0 §18.9](https://www.w3.org/TR/xslt-30/#func-current-output-uri).
    /// `None` stands for the principal result destination, whose URI this processor does not
    /// publish.
    pub(crate) open_result_document_uris: RefCell<Vec<Option<String>>>,
    pub(crate) recursion: RefCell<usize>,
    /// How deeply the engine is inside *temporary output state* — the content of a variable,
    /// a function, an attribute or a message — where `xsl:result-document` is `XTDE1480`.
    pub(crate) temporary_output: std::cell::Cell<usize>,
    /// How deeply the engine is inside a construct that *clears the current output URI* without
    /// changing the output state.
    ///
    /// [XSLT 3.0 §24.3](https://www.w3.org/TR/xslt-30/#current-output-uri) clears the current
    /// output URI "while evaluating stylesheet functions, dynamic function calls, evaluation of
    /// global variables, stylesheet parameters, and patterns". The first three of those are
    /// already temporary output state, which is not; a pattern is evaluated in whatever state
    /// the instruction that matches it was running in, and `disable-output-escaping` and
    /// `XTDE1480` are decided by that state rather than by this. Keeping the two counts apart is
    /// what keeps each rule readable at its own site.
    pub(crate) output_uri_cleared: std::cell::Cell<usize>,
    /// Instructions executed so far, against [`Runtime::step_limit`].
    pub(crate) steps: std::cell::Cell<u64>,
    /// The budget a caller set, or `None` for an unlimited transformation.
    pub(crate) step_limit: Option<u64>,
    /// The template-instantiation depth at which the engine stops, from
    /// [`Parameters::with_recursion_limit`](crate::Parameters::with_recursion_limit) or
    /// [`MAX_RECURSION`].
    pub(crate) recursion_limit: usize,
    /// The stack of the thread the transformation runs on, from
    /// [`Parameters::with_stack_size`](crate::Parameters::with_stack_size) or
    /// [`DEFAULT_STACK_SIZE`].
    pub(crate) stack_size: usize,
    /// A mark taken on the native stack when the transformation started, against which the
    /// stack it has consumed since is measured.
    pub(crate) stack_base: usize,
    /// How much native stack the transformation may consume before it stops.
    pub(crate) stack_budget: usize,
}

/// The stack a transformation assumes the thread it runs on has, when the caller does not say.
///
/// `std::thread::spawn` gives a new thread 2 MiB unless it is told otherwise, and that is the
/// shape a thread pool or an async executor's worker hands a library, so it is the figure the
/// engine has to stay inside by default. A caller running on a bigger stack — a process's main
/// thread, or a thread it spawned with room for deep recursion on purpose — says so with
/// [`Parameters::with_stack_size`](crate::Parameters::with_stack_size).
pub const DEFAULT_STACK_SIZE: usize = 2 * 1024 * 1024;

/// The share of the thread's stack a transformation may consume.
///
/// Two thirds of it: the remaining third is what the caller's own frames below the engine, and
/// the work done between two depth checks, are left to live in.
const STACK_SHARE: usize = 3;

/// A mark on the native stack, taken as the address of a local.
///
/// Stacks grow downwards on every target this library builds for, but the direction does not
/// matter: what is compared is the distance between two marks.
fn stack_mark() -> usize {
    let marker = 0_u8;
    std::ptr::addr_of!(marker) as usize
}

impl std::fmt::Debug for Runtime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Runtime")
            .field("templates", &self.ir.templates.len())
            .finish_non_exhaustive()
    }
}

impl Runtime {
    /// The innermost current node, which `current()` reports.
    pub(crate) fn current_item(&self) -> Option<Item> {
        self.current.borrow().last().cloned().flatten()
    }

    /// The static context of the expression currently being evaluated, which the availability
    /// functions expand a lexical name against.
    pub(crate) fn calling_static_context(&self) -> Option<Arc<oxixml_xpath::StaticContext>> {
        self.static_contexts.borrow().last().map(Arc::clone)
    }

    /// The innermost group, which `current-group()` reports.
    pub(crate) fn current_group(&self) -> Option<Vec<Item>> {
        self.groups.borrow().last().map(|(group, _)| group.clone())
    }

    /// The innermost grouping key, which `current-grouping-key()` reports.
    pub(crate) fn current_grouping_key(&self) -> Option<Sequence> {
        self.groups.borrow().last().and_then(|(_, key)| key.clone())
    }

    /// The innermost regular-expression capture groups, which `regex-group()` reports.
    pub(crate) fn regex_group(&self, index: usize) -> Option<String> {
        self.regex_groups
            .borrow()
            .last()
            .and_then(|groups| groups.get(index).cloned())
    }

    /// A dynamic context holding one binding per global variable, and nothing else.
    ///
    /// It is built at most once per generation of [`Runtime::globals`] and handed out by value;
    /// a [`DynamicContext`] is a handful of reference-counted handles, so the copy is free where
    /// rebuilding the chain was not. See [`Runtime::globals_context`].
    pub(crate) fn globals_bound_context(&self) -> DynamicContext {
        let generation = self.globals_generation.get();
        if let Some((built, context)) = self.globals_context.borrow().as_ref()
            && *built == generation
        {
            return context.clone();
        }
        let mut context = DynamicContext::new(Arc::clone(&self.pattern_static));
        for (name, value) in self.globals.borrow().iter() {
            context = context.with_variable_name(name.clone(), value.clone());
        }
        *self.globals_context.borrow_mut() = Some((generation, context.clone()));
        context
    }

    /// Records that [`Runtime::globals`] has changed, so that the shared context is rebuilt.
    pub(crate) fn note_globals_changed(&self) {
        self.globals_generation
            .set(self.globals_generation.get().wrapping_add(1));
    }

    /// Declares the base output URI — see [`crate::Parameters::with_base_output_uri`].
    pub(crate) fn set_base_output_uri(&self, uri: Option<String>) {
        *self.base_output_uri.borrow_mut() = uri;
    }

    /// The base output URI, when the invocation declared one.
    pub(crate) fn base_output_uri(&self) -> Option<String> {
        self.base_output_uri.borrow().clone()
    }

    /// What `fn:current-output-uri()` reports.
    ///
    /// [XSLT 3.0 §18.9](https://www.w3.org/TR/xslt-30/#func-current-output-uri): the URI of the
    /// innermost `xsl:result-document` instruction currently being evaluated; the empty
    /// sequence in *temporary output state* (constructing a variable, a parameter, a
    /// stylesheet function's result, an attribute and the like — never a final result tree),
    /// and — for the principal output destination — the *base output URI* the invocation
    /// declared, which is the URI the principal result is notionally written to (§23.1). A
    /// caller that declared none leaves the principal destination with no URI to report.
    ///
    /// [§24.3](https://www.w3.org/TR/xslt-30/#current-output-uri) adds one construct that is not
    /// an output state of its own: while a **pattern** is evaluated the URI is cleared, so that
    /// a rule may be selected by whether the transformation has one — see
    /// [`Runtime::output_uri_cleared`].
    pub(crate) fn current_output_uri(&self) -> Option<String> {
        if self.temporary_output.get() > 0 || self.output_uri_cleared.get() > 0 {
            return None;
        }
        match self.open_result_document_uris.borrow().last() {
            // Inside an `xsl:result-document`, its own destination.
            Some(open) => open.clone(),
            // The principal destination, which is the base output URI when the invocation
            // declared one and nothing at all when it did not.
            None => self.base_output_uri(),
        }
    }
}

/// Enters *temporary output state* for as long as it is held.
///
/// The tree-building helpers of [`Transformer`] bracket their own bodies, so this is for the one
/// caller that fills an [`Emitter`] it made itself: the built-in `shallow-copy` rule, whose
/// "instructions" are a walk over the source rather than a sequence constructor.
pub(crate) struct TemporaryOutput(Rc<Runtime>);

impl TemporaryOutput {
    /// Enters temporary output state on `runtime`.
    pub(crate) fn enter(runtime: &Rc<Runtime>) -> Self {
        runtime
            .temporary_output
            .set(runtime.temporary_output.get() + 1);
        Self(Rc::clone(runtime))
    }
}

impl Drop for TemporaryOutput {
    fn drop(&mut self) {
        self.0
            .temporary_output
            .set(self.0.temporary_output.get().saturating_sub(1));
    }
}

/// Clears the current output URI for as long as it is held.
///
/// [XSLT 3.0 §24.3](https://www.w3.org/TR/xslt-30/#current-output-uri) clears the URI while a
/// **pattern** is evaluated, which is not a change of output state: the instruction that is
/// matching the pattern goes on writing to the destination it was writing to, so
/// `disable-output-escaping` and the `XTDE1480` prohibition on `xsl:result-document` are decided
/// as they were. `current-output-uri-008` matches on the distinction from both sides — one
/// template rule is chosen by `empty(current-output-uri())` and its twin by
/// `current-output-uri()`, and an `xsl:for-each-group` starts a group where an item equals
/// `count(current-output-uri())`.
///
/// Holding nothing when there is no transformation in progress is what lets a pattern be matched
/// from outside one, which the compiler does while it checks a `match` attribute.
pub(crate) struct ClearedOutputUri(Option<Rc<Runtime>>);

impl ClearedOutputUri {
    /// Enters a scope in which `fn:current-output-uri()` answers the empty sequence.
    pub(crate) fn enter() -> Self {
        let runtime = current_runtime();
        if let Some(runtime) = &runtime {
            runtime
                .output_uri_cleared
                .set(runtime.output_uri_cleared.get() + 1);
        }
        Self(runtime)
    }
}

impl Drop for ClearedOutputUri {
    fn drop(&mut self) {
        if let Some(runtime) = &self.0 {
            runtime
                .output_uri_cleared
                .set(runtime.output_uri_cleared.get().saturating_sub(1));
        }
    }
}

/// The default maximum depth of template instantiation, which turns unbounded recursion into an
/// error rather than a stack overflow.
///
/// Recursion is how XSLT expresses iteration over a list, so a stylesheet that walks 500 items
/// with a non-tail-recursive named template is ordinary rather than runaway —
/// `call-template-1001` in the W3C suite does exactly that — and the figure is set well above
/// that shape. A caller that simply wants to bound the work should use
/// [`Parameters::with_step_limit`](crate::Parameters::with_step_limit) instead.
///
/// The depth a *thread* can carry is a separate question, and a much smaller number. One level
/// of template instantiation costs roughly 6.5–7 KiB of native stack in an optimized build of
/// this workspace, and around 90 KiB in a build without optimization, where every arm of the
/// instruction dispatch holds a stack slot of its own; so a thread with Rust's default 2 MiB
/// stack carries a few hundred levels at best and a few dozen while debugging. That ceiling
/// belongs to [`Runtime::stack_budget`], which measures the stack actually consumed instead of
/// guessing a depth from it, and which a caller adjusts by declaring the thread it runs on with
/// [`Parameters::with_stack_size`](crate::Parameters::with_stack_size).
pub(crate) const MAX_RECURSION: usize = 1_000;

/// The engine.
pub(crate) struct Transformer {
    pub(crate) runtime: Rc<Runtime>,
}

impl Transformer {
    pub(crate) fn new(
        ir: Arc<StylesheetIr>,
        resolver: Option<Arc<dyn Resolver>>,
        base_uri: Option<String>,
        step_limit: Option<u64>,
        recursion_limit: Option<usize>,
        stack_size: Option<usize>,
        implicit_timezone: Option<TimezoneOffset>,
    ) -> Self {
        let stack_size = stack_size.unwrap_or(DEFAULT_STACK_SIZE);
        // F&O 3.1 §5.7.1 ties `fn:codepoints-to-string`'s character check to the version of XML
        // the host processor uses, and this engine works over XML 1.1 documents, so the C0
        // controls XML 1.1 admits are characters here. XPath and XQuery entry points build
        // their own `FnContext` and keep the 1.0 default.
        let mut fn_context =
            FnContext::new().with_xml_version(oxixml_unicode::xml::XmlVersion::V11);
        if let Some(base) = &base_uri {
            fn_context = fn_context.with_static_base_uri(base.clone());
        }
        if let Some(timezone) = implicit_timezone {
            fn_context = fn_context.with_implicit_timezone(timezone);
        }
        for (name, format) in &ir.decimal_formats {
            fn_context = match name {
                None => fn_context.with_default_decimal_format(format.clone()),
                Some(name) => fn_context.with_decimal_format(name.clone(), format.clone()),
            };
        }
        if let Some(collation) = &ir.default_collation
            && let Ok(updated) = fn_context.clone().with_default_collation(collation)
        {
            fn_context = updated;
        }
        // A pattern predicate may call an `xsl:function`, so the static context patterns are
        // evaluated against needs the same stylesheet functions the compiler saw; without them
        // a call such as `match="A[my:test(.)]"` reports `XPST0017` at match time.
        let user_functions: Vec<crate::compile::FunctionSignature> = ir
            .functions
            .iter()
            .map(
                |((name, arity), function)| crate::compile::FunctionSignature {
                    name: name.clone(),
                    arity: *arity,
                    declared: function.declared_signature(),
                },
            )
            .collect();
        let stripper = Arc::new(DocumentStripper::new(Arc::clone(&ir)));
        let documents_read = Arc::new(DocumentLog::default());
        let xpath_resolver: Option<Arc<dyn DocumentResolver>> = resolver.as_ref().map(|resolver| {
            let inner = Arc::new(XPathResolver {
                inner: Arc::clone(resolver),
                base: base_uri.clone(),
                stripper: Arc::clone(&stripper),
            }) as Arc<dyn DocumentResolver>;
            Arc::new(RecordingResolver {
                inner,
                resolver: Arc::clone(resolver),
                base: base_uri.clone(),
                log: Arc::clone(&documents_read),
            }) as Arc<dyn DocumentResolver>
        });
        Self {
            runtime: Rc::new(Runtime {
                ir,
                resolver,
                xpath_resolver,
                stripper,
                fn_context,
                pattern_static: Arc::new(
                    oxixml_xpath::StaticContext::new()
                        .with_functions(functions::registry_with(&user_functions))
                        .with_undeclared_variables()
                        .with_xslt_functions(),
                ),
                base_uri,
                base_output_uri: RefCell::new(None),
                global_root: RefCell::new(None),
                stripped_selections: RefCell::new(Vec::new()),
                globals: RefCell::new(Vec::new()),
                globals_generation: std::cell::Cell::new(0),
                globals_context: RefCell::new(None),
                current: RefCell::new(Vec::new()),
                static_contexts: RefCell::new(Vec::new()),
                groups: RefCell::new(Vec::new()),
                regex_groups: RefCell::new(Vec::new()),
                keys: RefCell::new(HashMap::new()),
                keys_in_progress: RefCell::new(Vec::new()),
                function_cache: RefCell::new(HashMap::new()),
                messages: RefCell::new(Vec::new()),
                result_documents: RefCell::new(Vec::new()),
                principal_output: RefCell::new(None),
                principal_result: RefCell::new(None),
                principal_claimed: std::cell::Cell::new(false),
                documents_read,
                open_result_document_uris: RefCell::new(Vec::new()),
                recursion: RefCell::new(0),
                temporary_output: std::cell::Cell::new(0),
                output_uri_cleared: std::cell::Cell::new(0),
                steps: std::cell::Cell::new(0),
                step_limit,
                recursion_limit: recursion_limit.unwrap_or(MAX_RECURSION),
                stack_size,
                // The mark is taken here, one frame below the caller that is about to run the
                // transformation, which is as close to its start as the engine can stand.
                stack_base: stack_mark(),
                stack_budget: stack_size / STACK_SHARE * (STACK_SHARE - 1),
            }),
        }
    }

    pub(crate) fn ir(&self) -> &StylesheetIr {
        &self.runtime.ir
    }

    /// Builds the dynamic context an expression is evaluated in.
    pub(crate) fn dynamic_context(
        &self,
        compiled: &CompiledXPath,
        context: &Context,
    ) -> DynamicContext {
        // The globals are already bound, in declaration order, on the context the runtime shares;
        // the static context and the XPath version are the expression's own, so they are put back
        // over it. See [`Runtime::globals_context`].
        let mut dynamic = self
            .runtime
            .globals_bound_context()
            .with_static_context(Arc::clone(compiled.static_context()))
            .with_version(compiled.version());
        if let Some(item) = &context.item {
            dynamic = dynamic.with_focus(Focus::new(item.clone(), context.position, context.size));
        }
        // Everything in the function context but the collation and the base URI belongs to the
        // transformation as a whole — the decimal formats, the implicit timezone, the pinned
        // current dateTime, the message sink — so the runtime's is the one to carry. Two
        // properties are the exception, and both are *static*, so the one the expression
        // compiled under is put back rather than the transformation's:
        //
        // * the default collation, because
        //   [XSLT 3.0 §5.4.4](https://www.w3.org/TR/xslt-30/#dt-default-collation) lets
        //   `[xsl:]default-collation` on any ancestor of the expression override the
        //   stylesheet-wide one;
        // * the static base URI, because
        //   [XSLT 3.0 §5.4.3](https://www.w3.org/TR/xslt-30/#dt-static-base-uri) makes it the
        //   base URI of the *element containing the expression*, which `xml:base` and an
        //   included module both move away from the principal module's URI.
        let mut fn_context = self.runtime.fn_context.clone();
        let declared = compiled.static_context().default_collation();
        if declared != fn_context.default_collation().uri()
            && let Ok(updated) = fn_context.clone().with_default_collation(declared)
        {
            fn_context = updated;
        }
        if let Some(base) = compiled.static_context().base_uri() {
            fn_context = fn_context.with_static_base_uri(base);
        }
        dynamic = dynamic.with_fn_context(fn_context);
        if let Some(resolver) = &self.runtime.xpath_resolver {
            dynamic = dynamic.with_document_resolver(Arc::clone(resolver));
        }
        for (name, value) in context.bindings.visible().into_iter().rev() {
            dynamic = dynamic.with_variable_name(name.clone(), value.clone());
        }
        dynamic
    }

    /// Evaluates an expression, keeping function items.
    ///
    /// # Errors
    ///
    /// Any dynamic error the expression raises.
    pub(crate) fn evaluate(
        &self,
        compiled: &CompiledXPath,
        context: &Context,
    ) -> XsltResult<XpathValue> {
        let dynamic = self.dynamic_context(compiled, context);
        self.runtime.current.borrow_mut().push(context.item.clone());
        self.runtime
            .static_contexts
            .borrow_mut()
            .push(Arc::clone(compiled.static_context()));
        let result = compiled.evaluate_value_in(&dynamic);
        self.runtime.static_contexts.borrow_mut().pop();
        self.runtime.current.borrow_mut().pop();
        Ok(result?)
    }

    /// Evaluates an expression to its string value, following the version's rules.
    ///
    /// # Errors
    ///
    /// Any dynamic error the expression raises.
    pub(crate) fn evaluate_string(
        &self,
        compiled: &CompiledXPath,
        context: &Context,
        separator: &str,
    ) -> XsltResult<String> {
        let value = self.evaluate(compiled, context)?;
        self.simple_content_of(&value, separator)
    }

    /// The item an initial match selection contributes, space-stripped.
    ///
    /// [XSLT 3.0 §4.4](https://www.w3.org/TR/xslt-30/#stripping) applies `xsl:strip-space` and
    /// `xsl:preserve-space` to "all documents used as input to a transformation", the initial
    /// match selection among them — the caller has already stripped the global context item, but
    /// a node the invocation names separately arrives from a tree the engine has never seen.
    /// The tree is stripped once and remembered, so a selection of many nodes from one document
    /// still yields nodes of one tree; a node whose counterpart cannot be located is left as it
    /// came, which matches on the unstripped tree rather than failing the transformation.
    pub(crate) fn stripped_selection_item(&self, item: &Item) -> Item {
        if !self.runtime.ir.strips_any_space {
            return item.clone();
        }
        let Some(node) = item.as_node() else {
            return item.clone();
        };
        let root = node.root();
        // The global context item's tree has already been through the stripper; stripping it
        // again would build a second tree and break its identity with the one the global
        // variables were evaluated against.
        if self.runtime.global_root.borrow().as_ref() == Some(&root) {
            return item.clone();
        }
        let stripped = {
            let known = self
                .runtime
                .stripped_selections
                .borrow()
                .iter()
                .find(|(source, _)| source == &root)
                .map(|(_, stripped)| stripped.clone());
            match known {
                Some(stripped) => stripped,
                None => {
                    let Ok(tree) = strip_whitespace(&self.runtime.ir, &root) else {
                        return item.clone();
                    };
                    let stripped = tree.root();
                    self.runtime
                        .stripped_selections
                        .borrow_mut()
                        .push((root.clone(), stripped.clone()));
                    stripped
                }
            }
        };
        match locate_in_stripped(&self.runtime.ir, &root, &stripped, node, false) {
            Some(found) => Item::Node(found),
            None => item.clone(),
        }
    }

    /// The string an already-evaluated sequence reduces to under the simple-content rule.
    ///
    /// [XSLT 3.0 §5.7.2](https://www.w3.org/TR/xslt-30/#constructing-simple-content) discards
    /// zero-length text nodes, merges what is left of adjacent text nodes and only then atomizes
    /// and joins with `separator`; XSLT 1.0 took the string value of the first item alone, which
    /// backwards-compatible behaviour keeps.
    ///
    /// # Errors
    ///
    /// Whatever atomizing an item raises — a map, an array or a function item has no string
    /// value.
    pub(crate) fn simple_content_of(
        &self,
        value: &XpathValue,
        separator: &str,
    ) -> XsltResult<String> {
        if self.ir().version == crate::compile::XsltVersion::V10 {
            return Ok(match value.first() {
                Some(item) => item.string_value()?,
                None => String::new(),
            });
        }
        simple_content(value.iter(), separator)
    }

    /// Evaluates an expression to its effective boolean value.
    ///
    /// # Errors
    ///
    /// Any dynamic error the expression raises, plus `FORG0006`.
    pub(crate) fn evaluate_boolean(
        &self,
        compiled: &CompiledXPath,
        context: &Context,
    ) -> XsltResult<bool> {
        Ok(self
            .evaluate(compiled, context)?
            .effective_boolean_value()?)
    }

    /// Evaluates an attribute value template.
    ///
    /// # Errors
    ///
    /// Any dynamic error an embedded expression raises.
    pub(crate) fn evaluate_avt(
        &self,
        template: &crate::compile::Avt,
        context: &Context,
    ) -> XsltResult<String> {
        if let Some(constant) = template.constant() {
            return Ok(constant.to_owned());
        }
        let mut out = String::new();
        for part in template.parts() {
            match part {
                crate::compile::AvtPart::Literal(text) => out.push_str(text),
                crate::compile::AvtPart::Expression(compiled) => {
                    out.push_str(&self.evaluate_string(compiled, context, " ")?);
                }
            }
        }
        Ok(out)
    }

    /// Runs a sequence constructor into a fresh tree.
    ///
    /// # Errors
    ///
    /// Any dynamic error the instructions raise.
    pub(crate) fn run_to_tree(
        &mut self,
        instructions: &[Instruction],
        base: Option<&str>,
        context: &Context,
    ) -> XsltResult<ResultTree> {
        let mut emitter = Emitter::tree();
        emitter.set_base_uri(base.map(str::to_owned));
        self.run_into_tree(instructions, context, emitter, true)
    }

    /// Runs `xsl:copy`'s body into a new document node carrying the properties of the document
    /// node in focus — see [`Emitter::carry_document_properties`].
    ///
    /// # Errors
    ///
    /// Any dynamic error the instructions raise.
    pub(crate) fn run_to_document_copy(
        &mut self,
        instructions: &[Instruction],
        context: &Context,
        source: &oxixml_infoset::NodeRef,
    ) -> XsltResult<ResultTree> {
        let mut emitter = Emitter::tree();
        emitter.carry_document_properties(source);
        self.run_into_tree(instructions, context, emitter, true)
    }

    /// Runs a sequence constructor into `emitter`, entering temporary output state around it
    /// when `temporary`.
    fn run_into_tree(
        &mut self,
        instructions: &[Instruction],
        context: &Context,
        mut emitter: Emitter,
        temporary: bool,
    ) -> XsltResult<ResultTree> {
        if temporary {
            self.runtime
                .temporary_output
                .set(self.runtime.temporary_output.get() + 1);
        }
        let outcome = self.run_sequence(instructions, context, &mut emitter);
        if temporary {
            self.runtime
                .temporary_output
                .set(self.runtime.temporary_output.get().saturating_sub(1));
        }
        outcome?;
        Ok(emitter.finish()?.0)
    }

    /// Runs a sequence constructor into a fresh tree in *final* output state, which is what
    /// the body of an `xsl:result-document` with an `href` is in.
    ///
    /// # Errors
    ///
    /// Any dynamic error the instructions raise.
    pub(crate) fn run_to_final_tree(
        &mut self,
        instructions: &[Instruction],
        context: &Context,
        item_separator: Option<&str>,
    ) -> XsltResult<ResultTree> {
        let saved = self.runtime.temporary_output.replace(0);
        let mut emitter = Emitter::final_result();
        emitter.set_item_separator(item_separator);
        let outcome = self.run_sequence(instructions, context, &mut emitter);
        self.runtime.temporary_output.set(saved);
        outcome?;
        Ok(emitter.finish()?.0)
    }

    /// Runs a sequence constructor into a *raw result*: the sequence a destination whose
    /// serialization method builds no final result tree is serialized from.
    ///
    /// [XSLT 3.0 §26](https://www.w3.org/TR/xslt-30/#serialization) gives `build-tree` the
    /// default `no` for a method that writes items rather than nodes, so the body's result stays
    /// a sequence — a map or an array could not be put into a tree at all. Like
    /// [`Transformer::run_to_final_tree`] this is *final* output state, not temporary: the
    /// instruction is a destination of its own.
    ///
    /// # Errors
    ///
    /// Any dynamic error the instructions raise.
    pub(crate) fn run_to_raw_result(
        &mut self,
        instructions: &[Instruction],
        context: &Context,
        item_separator: Option<&str>,
    ) -> XsltResult<Sequence> {
        let saved = self.runtime.temporary_output.replace(0);
        let mut emitter = Emitter::sequence();
        emitter.set_item_separator(item_separator);
        let outcome = self.run_sequence(instructions, context, &mut emitter);
        self.runtime.temporary_output.set(saved);
        outcome?;
        Ok(Sequence::from_items(emitter.finish()?.1))
    }

    /// Runs a sequence constructor into a fresh sequence.
    ///
    /// # Errors
    ///
    /// Any dynamic error the instructions raise.
    pub(crate) fn run_to_sequence(
        &mut self,
        instructions: &[Instruction],
        base: Option<&str>,
        context: &Context,
    ) -> XsltResult<(ResultTree, Sequence)> {
        let mut emitter = Emitter::sequence();
        emitter.set_base_uri(base.map(str::to_owned));
        self.runtime
            .temporary_output
            .set(self.runtime.temporary_output.get() + 1);
        let outcome = self.run_sequence(instructions, context, &mut emitter);
        self.runtime
            .temporary_output
            .set(self.runtime.temporary_output.get().saturating_sub(1));
        outcome?;
        let (tree, items) = emitter.finish()?;
        Ok((tree, Sequence::from_items(items)))
    }

    /// Runs a sequence constructor into a fresh document node *without* entering temporary
    /// output state.
    ///
    /// `xsl:copy` applied to a document node builds a new document node, so its content is
    /// governed by the rules for the content of a document node — an attribute or a namespace
    /// node among it is `XTDE0420` (`error-0420a`, `copy-1210`) — but the tree it builds is on
    /// its way to the caller's destination rather than into a variable, so the output state
    /// does not change.
    ///
    /// # Errors
    ///
    /// Any dynamic error the instructions raise.
    pub(crate) fn run_to_tree_in_place(
        &mut self,
        instructions: &[Instruction],
        context: &Context,
    ) -> XsltResult<ResultTree> {
        self.run_into_tree(instructions, context, Emitter::tree(), false)
    }

    /// Runs a sequence constructor into a fresh sequence *without* entering temporary output
    /// state.
    ///
    /// Two constructs collect their body as a sequence for reasons of their own rather than
    /// because the body is building a value: a template rule declared `as` has to type-check its
    /// result before writing it, and `xsl:try` has to be able to discard its result if the body
    /// fails. Neither is temporary output state — the body still writes to the destination its
    /// caller was writing to — and the difference is observable, because
    /// [XSLT 3.0 §11.4.4](https://www.w3.org/TR/xslt-30/#disable-output-escaping) ignores
    /// `disable-output-escaping` exactly in temporary output state. `doe-0184` pins the
    /// distinction from the other side: a *stylesheet function* body genuinely is temporary
    /// output state, and there the request is ignored.
    ///
    /// The [`Flow`] the body ended on is answered along with the sequence: an `xsl:break` inside
    /// an `xsl:try` inside an `xsl:iterate` still ends the iteration
    /// ([XSLT 3.0 §8.8](https://www.w3.org/TR/xslt-30/#iterate)), so a caller that collects a
    /// body has to hand the outcome on rather than settle it.
    ///
    /// # Errors
    ///
    /// Any dynamic error the instructions raise.
    pub(crate) fn run_to_sequence_in_place(
        &mut self,
        instructions: &[Instruction],
        context: &Context,
    ) -> XsltResult<(Flow, ResultTree, Sequence)> {
        let mut emitter = Emitter::sequence();
        let flow = self.run_sequence(instructions, context, &mut emitter)?;
        let (tree, items) = emitter.finish()?;
        Ok((flow, tree, Sequence::from_items(items)))
    }

    /// Whether a `disable-output-escaping` request written on the instruction now being
    /// evaluated has any effect.
    ///
    /// [XSLT 3.0 §11.4.4](https://www.w3.org/TR/xslt-30/#disable-output-escaping) ignores the
    /// attribute in temporary output state: the text node is not on its way to a serializer, so
    /// there is no escaping to disable, and a copy of it made later must not resurrect the
    /// request either.
    pub(crate) fn escaping_may_be_disabled(&self) -> bool {
        self.runtime.temporary_output.get() == 0
    }

    /// Runs a sequence constructor and reduces its result to a string the way XSLT 3.0
    /// §5.7.2 *Constructing simple content* prescribes.
    ///
    /// The steps the specification lists are, in order: discard zero-length text nodes,
    /// merge adjacent text nodes, atomize what is left, cast each value to a string, and
    /// concatenate the strings with `separator` between adjacent ones. Taking the string
    /// value of a temporary tree — which is what a naive implementation does — skips the
    /// separator entirely and merges *every* item rather than only adjacent text nodes.
    ///
    /// # Errors
    ///
    /// Any dynamic error the instructions raise, plus any error atomization raises.
    pub(crate) fn run_to_simple_content(
        &mut self,
        instructions: &[Instruction],
        context: &Context,
        separator: &str,
    ) -> XsltResult<String> {
        let (_tree, sequence) = self.run_to_sequence(instructions, None, context)?;
        simple_content(sequence.iter(), separator)
    }

    /// Guards against unbounded template recursion.
    ///
    /// Two ceilings apply, and either one stops the transformation with a diagnosable error
    /// rather than letting it run the thread out of stack — an outcome no caller can catch,
    /// because exhausting the native stack aborts the whole process. The first is the
    /// instantiation depth, which is what a stylesheet author thinks in. The second is the
    /// native stack itself: one level of template instantiation costs several kilobytes of it,
    /// and tens of kilobytes in a build without optimization, so the depth that fits depends on
    /// the thread far more than on the stylesheet. Measuring what has actually been consumed
    /// covers both.
    ///
    /// A level that is refused is not entered, so the depth is given back before the error
    /// leaves: `XTDE0640` is an ordinary dynamic error that an `xsl:try` may catch and carry on
    /// from, and a level left charged would make every later instantiation start deeper than it
    /// really is.
    ///
    /// # Errors
    ///
    /// `XTDE0640` when either ceiling is reached.
    pub(crate) fn enter(&self) -> XsltResult<()> {
        let mut depth = self.runtime.recursion.borrow_mut();
        *depth += 1;
        if *depth > self.runtime.recursion_limit {
            *depth -= 1;
            return Err(XsltError::dynamic(
                "XTDE0640",
                "template instantiation nested too deeply; the stylesheet is probably recursive",
            ));
        }
        let used = self.runtime.stack_base.abs_diff(stack_mark());
        if used > self.runtime.stack_budget {
            *depth -= 1;
            return Err(XsltError::dynamic(
                "XTDE0640",
                format!(
                    "the transformation has used {used} bytes of native stack, more than the \
                     {} it may take on a thread of {} bytes; a stylesheet that recurses this \
                     deeply needs a larger stack, declared with `Parameters::with_stack_size`",
                    self.runtime.stack_budget, self.runtime.stack_size,
                ),
            ));
        }
        Ok(())
    }

    /// Charges one instruction against the step budget.
    ///
    /// # Errors
    ///
    /// `OXIXML0001` when the budget a caller set is exhausted. The code is not one the
    /// specification defines: a step limit is an implementation-defined resource limit, which
    /// [XSLT 3.0 §2.9](https://www.w3.org/TR/xslt-30/#conformance) explicitly permits.
    pub(crate) fn charge(&self) -> XsltResult<()> {
        // A caller-supplied deadline stops the transformation between instructions, which is the
        // only place a running transformation can be stopped without leaving a half-built result
        // behind. The expression evaluator checks the same deadline at its own checkpoints
        // ([`oxixml_xpath::EvaluationLimits::declare_deadline`]), so an instruction loop that
        // evaluates nothing — a deep sequence of `xsl:for-each` over a large tree — is bounded
        // here and a runaway expression is bounded there. The clock is read only when a deadline
        // was declared.
        oxixml_fn::function::CallLimits::check_deadline()?;
        let Some(limit) = self.runtime.step_limit else {
            return Ok(());
        };
        let steps = self.runtime.steps.get() + 1;
        self.runtime.steps.set(steps);
        if steps > limit {
            return Err(XsltError::dynamic(
                "OXIXML0001",
                format!("the transformation exceeded its budget of {limit} instructions"),
            ));
        }
        Ok(())
    }

    /// Leaves a template instantiation.
    pub(crate) fn leave(&self) {
        let mut depth = self.runtime.recursion.borrow_mut();
        *depth = depth.saturating_sub(1);
    }

    /// Evaluates every global variable, deferring the ones whose dependencies are not ready.
    ///
    /// # Errors
    ///
    /// `XTDE0640` when the declarations are circular, plus any dynamic error they raise.
    pub(crate) fn evaluate_globals(
        &mut self,
        supplied: &[(ExpandedName, XpathValue)],
        initial: Option<Item>,
    ) -> XsltResult<()> {
        for (name, value) in supplied {
            self.runtime
                .globals
                .borrow_mut()
                .push((name.clone(), value.clone()));
        }
        self.runtime.note_globals_changed();
        // The tree the global context item belongs to has already been space-stripped by the
        // caller; remembering it keeps `apply_to_initial` from stripping it a second time into
        // a tree of its own.
        *self.runtime.global_root.borrow_mut() = initial
            .as_ref()
            .and_then(Item::as_node)
            .map(oxixml_infoset::XdmNode::root);
        let mut pending: Vec<usize> = (0..self.runtime.ir.globals.len()).collect();
        let context = Context::new(initial);
        while !pending.is_empty() {
            let mut deferred = Vec::new();
            let mut progressed = false;
            let mut last_error = None;
            let mut unresolved_reference = false;
            let pending_names: Vec<ExpandedName> = pending
                .iter()
                .filter_map(|index| self.runtime.ir.globals.get(*index))
                .filter(|global| !global.is_static)
                .map(|global| global.name.clone())
                .collect();
            for index in pending {
                let ir = Arc::clone(&self.runtime.ir);
                let Some(global) = ir.globals.get(index) else {
                    continue;
                };
                let already_supplied = self
                    .runtime
                    .globals
                    .borrow()
                    .iter()
                    .any(|(name, _)| *name == global.name);
                // A static declaration keeps the value the *compilation* settled on, whether it
                // was written as a parameter or as a variable: XSLT 3.0 §9.6 evaluates it once,
                // before anything else exists, and the shadow attributes and `use-when`
                // expressions that read it were resolved against that value. Re-evaluating its
                // `select` here would answer from the run-time bindings instead — where an
                // ordinary global of the same name may be in scope, which is exactly what
                // `static-027` writes (`$p` is 1 statically and 0 at run time).
                if already_supplied && (global.is_parameter || global.is_static) {
                    progressed = true;
                    continue;
                }
                if global.is_parameter && global.required && !already_supplied {
                    return Err(XsltError::dynamic(
                        "XTDE0050",
                        format!(
                            "the stylesheet parameter `{}` is required but was not supplied",
                            global.name
                        ),
                    ));
                }
                // A stylesheet parameter that is only *implicitly* mandatory (XSLT 3.0 §9.3: no
                // explicit default, and an `as` that does not admit the empty sequence) is
                // reported under `XTDE0700`, the missing-value code the suite's error-0610d
                // asks for, rather than as a failed conversion of a default that the spec says
                // does not exist. `XTDE0050` stays for `required="yes"`, which is the only
                // shape any other case in the suite exercises.
                if global.is_implicitly_mandatory() && !already_supplied {
                    return Err(XsltError::dynamic(
                        "XTDE0700",
                        format!(
                            "no value was supplied for the stylesheet parameter `{}`, whose \
                             declared type does not admit the empty sequence",
                            global.name
                        ),
                    ));
                }
                match self.global_value(global, &context) {
                    Ok(value) => {
                        let mut globals = self.runtime.globals.borrow_mut();
                        globals.retain(|(name, _)| *name != global.name);
                        globals.push((global.name.clone(), value));
                        drop(globals);
                        self.runtime.note_globals_changed();
                        progressed = true;
                    }
                    Err(error) => {
                        // A declaration that refers to its own name never reaches here:
                        // `Compiler::check_global_self_reference` has already rejected it as
                        // the static error `XPST0008` (XSLT 3.0 §9.9). What is left is a
                        // genuine dependency on another declaration.
                        //
                        // A reference to a global that is still waiting its turn is what a
                        // cycle looks like from inside the fixpoint loop; a reference to a
                        // name the stylesheet never declares stays `XPST0008`.
                        unresolved_reference |= error.is("XPST0008")
                            && pending_names.iter().any(|name: &ExpandedName| {
                                error.message().contains(&format!("${name}"))
                            });
                        last_error = Some(error);
                        deferred.push(index);
                    }
                }
            }
            if deferred.is_empty() {
                break;
            }
            if !progressed {
                if unresolved_reference {
                    return Err(XsltError::dynamic(
                        "XTDE0640",
                        "the global variable declarations are circular",
                    ));
                }
                return Err(last_error.unwrap_or_else(|| {
                    XsltError::dynamic("XTDE0640", "the global variable declarations are circular")
                }));
            }
            pending = deferred;
        }
        Ok(())
    }

    fn global_value(
        &mut self,
        global: &crate::compile::GlobalVariable,
        context: &Context,
    ) -> XsltResult<XpathValue> {
        let value = self.global_raw_value(global, context)?;
        match &global.as_type {
            // XSLT 3.0 §9.1: a global variable's value is `XTTE0570`; the default value of a
            // stylesheet parameter that does not match is `XTTE0600`.
            Some(required) if global.is_parameter => required.convert(
                value,
                "XTTE0600",
                &format!("the default value of the parameter `{}`", global.name),
            ),
            Some(required) => required.convert(
                value,
                "XTTE0570",
                &format!("the value of the variable `{}`", global.name),
            ),
            None => Ok(value),
        }
    }

    fn global_raw_value(
        &mut self,
        global: &crate::compile::GlobalVariable,
        context: &Context,
    ) -> XsltResult<XpathValue> {
        match &global.select {
            Some(select) => self.evaluate(select, context),
            None if global.content.is_empty() && global.is_parameter => {
                Ok(XpathValue::from(Sequence::empty()))
            }
            None if global.as_type.is_some() => {
                let (_, sequence) =
                    self.run_to_sequence(&global.content, global.base_uri.as_deref(), context)?;
                Ok(XpathValue::from(sequence))
            }
            None => {
                let tree =
                    self.run_to_tree(&global.content, global.base_uri.as_deref(), context)?;
                Ok(XpathValue::from(Sequence::from(tree.root())))
            }
        }
    }

    /// Runs `body` with the captured substrings of `fn:regex-group` empty.
    ///
    /// [XSLT 3.0 §18.4](https://www.w3.org/TR/xslt-30/#func-regex-group): the set of captured
    /// substrings is "set to an empty sequence" while a pattern or the body of a stylesheet
    /// function is evaluated, so a `regex-group()` call reached from either — even from inside
    /// an `xsl:matching-substring` — reports nothing rather than the enclosing match's groups.
    pub(crate) fn without_regex_groups<R>(&mut self, body: impl FnOnce(&mut Self) -> R) -> R {
        self.runtime.regex_groups.borrow_mut().push(Vec::new());
        let result = body(self);
        self.runtime.regex_groups.borrow_mut().pop();
        result
    }

    /// The messages `xsl:message` produced.
    pub(crate) fn messages(&self) -> Vec<String> {
        self.runtime.messages.borrow().clone()
    }

    /// The secondary result documents, as `(href, serialized text)` pairs.
    pub(crate) fn result_documents(&self) -> Vec<(String, String)> {
        self.runtime.result_documents.borrow().clone()
    }

    /// The serialization properties the principal result must use, when an
    /// `xsl:result-document` overrode them.
    pub(crate) fn principal_output(&self) -> Option<std::sync::Arc<OutputProperties>> {
        self.runtime.principal_output.borrow().clone()
    }

    /// The result tree an `xsl:result-document` built for the principal destination, if one did.
    pub(crate) fn take_principal_result(&self) -> Option<ResultTree> {
        self.runtime.principal_result.borrow_mut().take()
    }
}

/// Reduces a sequence to the string
/// [XSLT 3.0 §5.7.2](https://www.w3.org/TR/xslt-30/#constructing-simple-content) prescribes.
///
/// The steps of *constructing simple content* run in a fixed order, and the order is the whole
/// point: **first** every zero-length text node is discarded, **then** adjacent text nodes are
/// merged into one, and only **then** is what survives atomized, cast to strings and joined with
/// `separator` between adjacent values. Joining item by item — the obvious implementation —
/// separates text nodes that the merge would have made one, so a function returning the three
/// text nodes `'`, `test` and `'` comes out as `' test '` where `'test'` is required
/// (`whitespace-025`), and a text value template over a recursive function acquires a space
/// between every character (`seqtor-029`, `seqtor-034`, `seqtor-035`).
///
/// Discarding runs before merging, so a zero-length text node between two others does not keep
/// them apart: `avt-1205` builds `<xsl:text/>[<xsl:value-of select="$p"/>]<xsl:text/>` and wants
/// `[0]`.
///
/// # Errors
///
/// Whatever atomizing an item raises.
fn simple_content<'a, I: SimpleContentItem + 'a>(
    items: impl Iterator<Item = &'a I>,
    separator: &str,
) -> XsltResult<String> {
    let mut parts: Vec<String> = Vec::new();
    let mut previous_was_text = false;
    for item in items {
        let text_value = item.text_node_value();
        match text_value {
            Some(text) => {
                if text.is_empty() {
                    continue;
                }
                match parts.last_mut() {
                    Some(last) if previous_was_text => last.push_str(&text),
                    _ => parts.push(text),
                }
                previous_was_text = true;
            }
            None => {
                parts.extend(item.atomized_strings()?);
                previous_was_text = false;
            }
        }
    }
    Ok(parts.join(separator))
}

/// What [`simple_content`] needs of an item.
///
/// Simple content is constructed both from the data-model items a sequence constructor yields
/// and from the result of an expression, which may additionally hold a function item; the two
/// item types are distinct, so the shared step asks for the two operations it uses rather than
/// for one of them.
trait SimpleContentItem {
    /// The string value of this item when it is a text node, and nothing otherwise.
    fn text_node_value(&self) -> Option<String>;

    /// The strings this item contributes once the sequence has been atomized — one per atomic
    /// value, which is *not* always one per item: an array atomizes to the atomized values of
    /// all its members, so `array{1, 2, 3}` contributes three strings and is joined with the
    /// separator like any other three values (the suite's `arrays-303`, `cvt-041`).
    ///
    /// Taking the item's string value instead would report `FOTY0014` — "a function item has no
    /// string value" — for an array, which does have one member-wise, and would report it for a
    /// map where §5.7.2's atomization step raises `FOTY0013` (`maps-907`).
    ///
    /// # Errors
    ///
    /// Whatever atomizing the item raises — `FOTY0013` for a map or a function item.
    fn atomized_strings(&self) -> XsltResult<Vec<String>>;
}

impl SimpleContentItem for Item {
    fn text_node_value(&self) -> Option<String> {
        self.as_node()
            .and_then(|node| (node.node_kind() == NodeKind::Text).then(|| node.string_value()))
    }

    fn atomized_strings(&self) -> XsltResult<Vec<String>> {
        let mut strings = Vec::new();
        for atomic in self.atomize()? {
            strings.push(atomic.string_value()?);
        }
        Ok(strings)
    }
}

impl SimpleContentItem for oxixml_xpath::XpathItem {
    fn text_node_value(&self) -> Option<String> {
        self.as_node()
            .and_then(|node| (node.node_kind() == NodeKind::Text).then(|| node.string_value()))
    }

    fn atomized_strings(&self) -> XsltResult<Vec<String>> {
        Ok(self
            .atomize()?
            .into_iter()
            .map(|atomic| atomic.string_value())
            .collect())
    }
}

/// Whether the processor recognizes `uri` as a collation.
///
/// [XSLT 3.0 §5.4.4](https://www.w3.org/TR/xslt-30/#collations) makes an unrecognized
/// collation URI a static error wherever a collation is named, so the compiler asks the
/// function library — the one authority on which URIs resolve — before accepting one.
pub(crate) fn is_recognized_collation(uri: &str) -> bool {
    FnContext::new().with_default_collation(uri).is_ok()
}

/// Whether whitespace-only text under `parent` is stripped, per `xsl:strip-space`.
pub(crate) fn strips_whitespace(ir: &StylesheetIr, element: &NodeRef) -> bool {
    let Some(name) = element.node_name() else {
        return false;
    };
    let mut best: Option<(usize, f64, bool)> = None;
    for control in &ir.space_controls {
        let matched = match &control.test {
            crate::compile::SpaceTest::Any => true,
            crate::compile::SpaceTest::AnyLocal(namespace) => {
                name.namespace() == Some(namespace.as_str())
            }
            crate::compile::SpaceTest::AnyNamespace(local) => name.local_name() == local,
            crate::compile::SpaceTest::Name(wanted) => *wanted == name,
        };
        if !matched {
            continue;
        }
        let candidate = (control.import_precedence, control.priority, control.strip);
        best = match best {
            Some(current) if (current.0, current.1) >= (candidate.0, candidate.1) => Some(current),
            _ => Some(candidate),
        };
    }
    best.is_some_and(|(_, _, strip)| strip)
}

/// The counterpart of `target` inside `stripped`, the whitespace-stripped copy of `source`.
///
/// The walk mirrors [`copy_stripped`] step for step — the same `xml:space` state, the same
/// [`strips_whitespace`] decision — so the correspondence it reports is the one the copy was
/// actually built with rather than a guess reconstructed from the two trees.
fn locate_in_stripped(
    ir: &StylesheetIr,
    source: &NodeRef,
    stripped: &NodeRef,
    target: &NodeRef,
    preserve: bool,
) -> Option<NodeRef> {
    if source == target {
        return Some(stripped.clone());
    }
    if !matches!(source.node_kind(), NodeKind::Document | NodeKind::Element) {
        return None;
    }
    let mut preserve_here = preserve;
    if source.node_kind() == NodeKind::Element {
        for (position, attribute) in source.attributes().into_iter().enumerate() {
            if attribute.node_name().is_some_and(|name| {
                name.namespace() == Some(oxixml_qname::XML_NAMESPACE)
                    && name.local_name() == "space"
            }) {
                preserve_here = attribute.string_value() == "preserve";
            }
            if &attribute == target {
                // The attributes are copied in order and none is ever dropped.
                return stripped.attributes().get(position).cloned();
            }
        }
    }
    let strip =
        source.node_kind() == NodeKind::Element && !preserve_here && strips_whitespace(ir, source);
    let mut copies = stripped.children().into_iter();
    for child in source.children() {
        if strip && child.node_kind() == NodeKind::Text && child.string_value().trim().is_empty() {
            continue;
        }
        let copy = copies.next()?;
        if let Some(found) = locate_in_stripped(ir, &child, &copy, target, preserve_here) {
            return Some(found);
        }
    }
    None
}

/// Applies the whitespace-stripping rules to a source tree, producing a stripped copy.
///
/// [XSLT 3.0 §4.4](https://www.w3.org/TR/xslt-30/#strip) removes a whitespace-only text node
/// unless its parent preserves space, `xml:space="preserve"` is in force, or the node is the
/// only text in an element that `xsl:preserve-space` names.
///
/// # Errors
///
/// `XTDE0001` when the copy cannot be built.
pub(crate) fn strip_whitespace(ir: &StylesheetIr, root: &NodeRef) -> XsltResult<ResultTree> {
    let mut builder = crate::tree::ResultBuilder::new();
    builder.set_document_uri(root.document_uri());
    // The stripped tree is the same document, retrieved from the same place, so it keeps both
    // URIs; its DTD declared the same unparsed entities and notations, which
    // `fn:unparsed-entity-uri` and `fn:unparsed-entity-public-id` read.
    builder.set_base_uri(root.base_uri().or_else(|| root.document_uri()));
    builder.carry_declarations(&root.root());
    copy_stripped(ir, root, &mut builder, false)?;
    Ok(builder.build()?.0)
}

fn copy_stripped(
    ir: &StylesheetIr,
    node: &NodeRef,
    builder: &mut crate::tree::ResultBuilder,
    preserve: bool,
) -> XsltResult<()> {
    match node.node_kind() {
        NodeKind::Document => {
            for child in node.children() {
                copy_stripped(ir, &child, builder, preserve)?;
            }
        }
        NodeKind::Element => {
            let name = node
                .node_name()
                .unwrap_or_else(|| ExpandedName::new(None::<&str>, "element"));
            let prefix = node.namespaces().into_iter().find_map(|binding| {
                (Some(binding.string_value().as_str()) == name.namespace())
                    .then(|| binding.node_name().map(|n| n.local_name().to_owned()))
                    .flatten()
            });
            builder.start_element(name, prefix);
            for binding in node.namespaces() {
                let prefix = binding.node_name().map(|n| n.local_name().to_owned());
                if prefix.as_deref() == Some("xml") {
                    continue;
                }
                builder.namespace(prefix, &binding.string_value())?;
            }
            let mut preserve_here = preserve;
            for attribute in node.attributes() {
                let attribute_name = attribute
                    .node_name()
                    .unwrap_or_else(|| ExpandedName::new(None::<&str>, "attribute"));
                if attribute_name.namespace() == Some(oxixml_qname::XML_NAMESPACE)
                    && attribute_name.local_name() == "space"
                {
                    preserve_here = attribute.string_value() == "preserve";
                }
                // Stripping whitespace does not make a new document: §4.4 removes text nodes
                // from the one that was parsed, so an attribute the DTD declared `ID` is still
                // an ID here. `fn:id` and an `id(…)` pattern read exactly that property
                // (`id-028` … `id-031` strip space and then match on `id('id2')`).
                builder.typed_attribute(
                    attribute_name,
                    None,
                    attribute.string_value(),
                    attribute.is_id(),
                    attribute.is_idref(),
                )?;
            }
            let strip = !preserve_here && strips_whitespace(ir, node);
            for child in node.children() {
                if strip
                    && child.node_kind() == NodeKind::Text
                    && child.string_value().trim().is_empty()
                {
                    continue;
                }
                copy_stripped(ir, &child, builder, preserve_here)?;
            }
            builder.end_element()?;
        }
        NodeKind::Text => builder.text(&node.string_value(), false),
        NodeKind::Comment => builder.comment(&node.string_value()),
        NodeKind::ProcessingInstruction => {
            let target = node
                .node_name()
                .map(|name| name.local_name().to_owned())
                .unwrap_or_default();
            builder.processing_instruction(&target, &node.string_value())?;
        }
        NodeKind::Attribute | NodeKind::Namespace => {}
    }
    Ok(())
}

/// Serializes a result tree, after the serialization errors its parameter set can raise.
///
/// # Errors
///
/// Whatever [`crate::serialize::check`] reports for the parameters and the tree's shape.
pub(crate) fn serialize(tree: &ResultTree, properties: &OutputProperties) -> XsltResult<String> {
    crate::serialize::check(tree, properties)?;
    Ok(crate::serialize::serialize(tree, properties))
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxixml_infoset::AtomicValue;

    #[test]
    fn bindings_shadow_and_enumerate() {
        let name = ExpandedName::new(None::<&str>, "x");
        let other = ExpandedName::new(None::<&str>, "y");
        let bindings = Bindings::default()
            .bind(name.clone(), XpathValue::from(AtomicValue::integer(1_i64)))
            .bind(other.clone(), XpathValue::from(AtomicValue::integer(2_i64)))
            .bind(name.clone(), XpathValue::from(AtomicValue::integer(3_i64)));
        assert_eq!(
            bindings
                .get(&name)
                .and_then(|value| value.first())
                .and_then(oxixml_xpath::XpathItem::as_atomic)
                .map(oxixml_infoset::AtomicValue::string_value),
            Some("3".to_owned())
        );
        assert_eq!(bindings.visible().len(), 2);
        assert!(Bindings::default().visible().is_empty());
        assert!(
            bindings
                .get(&ExpandedName::new(None::<&str>, "z"))
                .is_none()
        );
    }
}
