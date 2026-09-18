//! The compiled form of a query: what [`crate::compile`] produces and [`super`] walks.

use crate::ast::{BindingKind, CatchTest};
use oxixml_infoset::{AtomicValue, ExpandedName};
use oxixml_xpath::{CompiledXPath, SequenceType};
use std::sync::Arc;

/// A compiled expression.
#[derive(Debug)]
pub(crate) enum Program {
    /// Delegated wholesale to the XPath engine.
    Xpath(Box<CompiledXPath>),
    /// An XPath template whose XQuery-only operands were lifted into variables.
    Spliced {
        template: Box<CompiledXPath>,
        holes: Vec<(ExpandedName, Program)>,
    },
    If {
        condition: Box<Program>,
        then_branch: Box<Program>,
        else_branch: Box<Program>,
    },
    Binding {
        kind: BindingKind,
        bindings: Vec<CompiledLetBinding>,
        body: Box<Program>,
    },
    Logical {
        or: bool,
        left: Box<Program>,
        right: Box<Program>,
    },
    Path {
        rooted: bool,
        rooted_descendant: bool,
        steps: Vec<CompiledStep>,
    },
    Postfix {
        primary: Box<Program>,
        postfixes: Vec<CompiledPostfix>,
    },
    Inline(Arc<UserFunction>),
    Flwor(Box<CompiledFlwor>),
    Construct(Box<CompiledConstructor>),
    TypeSwitch(Box<CompiledTypeSwitch>),
    Switch(Box<CompiledSwitch>),
    Try {
        body: Box<Program>,
        catches: Vec<CompiledCatch>,
    },
    /// `ordered { … }`, `unordered { … }` and an unrecognized extension expression.
    Wrap(Box<Program>),
    Literal(AtomicValue),
    ContextItem,
    StringTemplate(Vec<CompiledContent>),
    EmptySequence,
}

/// The conversion that a `as T` declaration applies to a value.
#[derive(Debug)]
pub(crate) struct TypeCheck {
    pub(crate) convert: Box<CompiledXPath>,
}

/// A user-defined or inline function.
#[derive(Debug)]
pub(crate) struct UserFunction {
    pub(crate) name: Option<ExpandedName>,
    /// The prolog-settable components of the module the function was declared in.
    pub(crate) options: Arc<crate::eval::RuntimeOptions>,
    pub(crate) parameters: Vec<ExpandedName>,
    pub(crate) parameter_types: Vec<Option<TypeCheck>>,
    pub(crate) result_type: Option<TypeCheck>,
    /// The declared signature, as sequence types rather than as the conversions that enforce
    /// them. A `function(…) as …` item test judges a function item by exactly this
    /// (XQuery 3.1 §2.5.5.5), so a named reference to a declared function has to carry it.
    /// `None` when a declared type could not be read as a sequence type, in which case the test
    /// falls back on comparing arities.
    pub(crate) declared_signature: Option<(Vec<SequenceType>, SequenceType)>,
    pub(crate) body: Program,
}

/// One `let $x as T := E` binding, and the shape a `for`/`some`/`every` binding takes when it
/// needs this crate's evaluator.
#[derive(Debug)]
pub(crate) struct CompiledLetBinding {
    pub(crate) name: ExpandedName,
    pub(crate) check: Option<TypeCheck>,
    pub(crate) value: Program,
}

/// One step of a path expression.
#[derive(Debug)]
pub(crate) struct CompiledStep {
    pub(crate) descendant: bool,
    pub(crate) simple_map: bool,
    /// `true` when the step's base is an axis step on a reverse axis. The step's *value* is in
    /// document order either way, but its predicates number the nodes along the axis, so a
    /// reverse axis numbers them from the context node outwards.
    pub(crate) reverse: bool,
    pub(crate) base: Program,
    pub(crate) predicates: Vec<Program>,
}

/// A postfix applied to a primary expression.
#[derive(Debug)]
pub(crate) enum CompiledPostfix {
    Predicate(Program),
    Arguments(Vec<Option<Program>>),
    Lookup(LookupSpec),
}

/// The key of a `?` lookup.
#[derive(Debug)]
pub(crate) enum LookupSpec {
    Name(String),
    Integer(i64),
    Wildcard,
    Computed(Box<Program>),
}

/// A FLWOR expression.
#[derive(Debug)]
pub(crate) struct CompiledFlwor {
    pub(crate) steps: Vec<FlworStep>,
    pub(crate) result: Program,
}

/// One clause of a FLWOR expression.
#[derive(Debug)]
pub(crate) enum FlworStep {
    For(Vec<CompiledForBinding>),
    Let(Vec<CompiledLetBinding>),
    Where(Program),
    Count(ExpandedName),
    GroupBy(Vec<CompiledGrouping>),
    OrderBy(Vec<CompiledOrderSpec>),
    Window(Box<CompiledWindow>),
}

/// One `for` binding of a FLWOR expression.
#[derive(Debug)]
pub(crate) struct CompiledForBinding {
    pub(crate) name: ExpandedName,
    pub(crate) check: Option<TypeCheck>,
    pub(crate) allowing_empty: bool,
    pub(crate) positional: Option<ExpandedName>,
    pub(crate) value: Program,
}

/// One `group by` specification.
#[derive(Debug)]
pub(crate) struct CompiledGrouping {
    pub(crate) name: ExpandedName,
    pub(crate) check: Option<TypeCheck>,
    pub(crate) value: Option<Program>,
    pub(crate) collation: Option<String>,
}

/// One `order by` specification.
#[derive(Debug)]
pub(crate) struct CompiledOrderSpec {
    pub(crate) key: Program,
    pub(crate) ascending: bool,
    pub(crate) empty_greatest: Option<bool>,
    pub(crate) collation: Option<String>,
}

/// A `tumbling window` or `sliding window` clause.
#[derive(Debug)]
pub(crate) struct CompiledWindow {
    pub(crate) tumbling: bool,
    pub(crate) name: ExpandedName,
    pub(crate) check: Option<TypeCheck>,
    pub(crate) value: Program,
    pub(crate) start: CompiledWindowCondition,
    pub(crate) end: Option<CompiledWindowCondition>,
    pub(crate) only: bool,
}

/// The variables and predicate of a window `start` or `end` condition.
#[derive(Debug)]
pub(crate) struct CompiledWindowCondition {
    pub(crate) current: Option<ExpandedName>,
    pub(crate) positional: Option<ExpandedName>,
    pub(crate) previous: Option<ExpandedName>,
    pub(crate) next: Option<ExpandedName>,
    pub(crate) when: Option<Program>,
}

/// A `typeswitch`.
#[derive(Debug)]
pub(crate) struct CompiledTypeSwitch {
    pub(crate) operand: Program,
    pub(crate) cases: Vec<CompiledTypeCase>,
    pub(crate) default_variable: Option<ExpandedName>,
    pub(crate) default_body: Program,
}

/// One `case` of a `typeswitch`, whose tests are `$v instance of T` expressions.
#[derive(Debug)]
pub(crate) struct CompiledTypeCase {
    pub(crate) variable: Option<ExpandedName>,
    pub(crate) tests: Vec<CompiledXPath>,
    pub(crate) body: Program,
}

/// A `switch`.
#[derive(Debug)]
pub(crate) struct CompiledSwitch {
    pub(crate) operand: Program,
    pub(crate) cases: Vec<(Vec<Program>, Program)>,
    pub(crate) default_body: Program,
}

/// One `catch` clause.
#[derive(Debug)]
pub(crate) struct CompiledCatch {
    pub(crate) tests: Vec<CatchTest>,
    pub(crate) body: Program,
}

/// A piece of a constructor's content.
#[derive(Debug)]
pub(crate) enum CompiledContent {
    Text(String),
    BoundaryWhitespace(String),
    Enclosed(Program),
    /// A directly nested constructor, whose node becomes a child *without* being copied — see
    /// [`crate::ast::ContentItem::Nested`].
    Nested(Program),
}

/// The name of a computed constructor.
#[derive(Debug)]
pub(crate) enum CompiledName {
    Lexical(String),
    Computed(Box<Program>),
}

/// A node constructor.
#[derive(Debug)]
pub(crate) enum CompiledConstructor {
    DirectElement(Box<CompiledElement>),
    Element {
        name: CompiledName,
        content: Option<Program>,
    },
    Attribute {
        name: CompiledName,
        content: Option<Program>,
    },
    Document(Program),
    Text(Option<Program>),
    Comment(Option<Program>),
    ProcessingInstruction {
        name: CompiledName,
        content: Option<Program>,
        /// `true` for the direct `<?target content?>` form, whose content is already literal.
        literal: bool,
    },
    Namespace {
        prefix: CompiledName,
        uri: Option<Program>,
    },
}

/// A direct element constructor.
#[derive(Debug)]
pub(crate) struct CompiledElement {
    pub(crate) name: String,
    pub(crate) namespaces: Vec<(Option<String>, String)>,
    pub(crate) attributes: Vec<CompiledAttribute>,
    pub(crate) content: Vec<CompiledContent>,
}

/// One attribute of a direct element constructor.
#[derive(Debug)]
pub(crate) struct CompiledAttribute {
    pub(crate) name: String,
    pub(crate) value: Vec<CompiledContent>,
}
