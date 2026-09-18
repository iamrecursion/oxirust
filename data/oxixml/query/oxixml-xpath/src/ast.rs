//! The abstract syntax tree the parser produces and the evaluator walks.
//!
//! The tree is *normalized*: the abbreviations of the surface grammar are gone by the time an
//! expression reaches this form, so the evaluator has one case per concept rather than one per
//! spelling.
//!
//! | Written | Represented as |
//! |---|---|
//! | `//b` | `Path(Path(Root, descendant-or-self::node()), child::b)` |
//! | `a//b` | `Path(Path(a, descendant-or-self::node()), child::b)` |
//! | `@x` | `attribute::x` |
//! | `..` | `parent::node()` |
//! | `.` | [`Expr::ContextItem`] |
//! | `(1, 2)` | [`Expr::Sequence`] |
//! | `a => f(1)` | `FunctionCall { name: f, arguments: [a, 1] }` |
//! | `f(1, ?)` | [`Expr::PartialCall`] |
//!
//! Every name is already resolved against the static context, and every type is already a
//! [`SequenceType`] or [`SingleType`], so the tree is self-contained: evaluating it needs a
//! [`DynamicContext`](crate::DynamicContext) and nothing else.

use crate::types::{KindTest, ListType, NameTest, SequenceType, SingleType};
use oxixml_infoset::{AtomicValue, Axis, ExpandedName};
use std::sync::Arc;

/// A node test: the part of a step that follows the axis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeTest {
    /// A name test, matched against the axis's principal node kind.
    Name(NameTest),
    /// A kind test.
    Kind(KindTest),
}

/// One step of a path expression: an axis, a node test and any predicates.
#[derive(Debug, Clone, PartialEq)]
pub struct AxisStep {
    /// The axis to walk.
    pub axis: Axis,
    /// The test each node on the axis must satisfy.
    pub node_test: NodeTest,
    /// The predicates, applied left to right.
    pub predicates: Vec<Expr>,
}

/// A `$name in expr` or `$name := expr` binding of a `for`, `let` or quantified expression.
#[derive(Debug, Clone, PartialEq)]
pub struct VarBinding {
    /// The variable's expanded name.
    pub name: ExpandedName,
    /// The expression whose value is bound.
    pub value: Expr,
}

/// An inline function expression, `function($x as T) as R { … }`.
#[derive(Debug, Clone, PartialEq)]
pub struct InlineFunction {
    /// The parameters, in order, each with its optional declared type.
    pub parameters: Vec<(ExpandedName, Option<SequenceType>)>,
    /// The declared result type, if the expression gave one.
    pub result: Option<SequenceType>,
    /// The function's body.
    pub body: Expr,
}

/// An arithmetic, sequence or string operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    /// `+`
    Add,
    /// `-`
    Subtract,
    /// `*`
    Multiply,
    /// `div`
    Divide,
    /// `idiv`
    IntegerDivide,
    /// `mod`
    Modulus,
    /// `to`
    Range,
    /// `union` and `|`
    Union,
    /// `intersect`
    Intersect,
    /// `except`
    Except,
    /// `||`
    Concat,
}

impl BinaryOp {
    /// The operator's source spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Add => "+",
            Self::Subtract => "-",
            Self::Multiply => "*",
            Self::Divide => "div",
            Self::IntegerDivide => "idiv",
            Self::Modulus => "mod",
            Self::Range => "to",
            Self::Union => "union",
            Self::Intersect => "intersect",
            Self::Except => "except",
            Self::Concat => "||",
        }
    }
}

/// A general comparison operator: `=`, `!=`, `<`, `<=`, `>`, `>=`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeneralOp {
    /// `=`
    Equal,
    /// `!=`
    NotEqual,
    /// `<`
    Less,
    /// `<=`
    LessEqual,
    /// `>`
    Greater,
    /// `>=`
    GreaterEqual,
}

impl GeneralOp {
    /// The operator's source spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Equal => "=",
            Self::NotEqual => "!=",
            Self::Less => "<",
            Self::LessEqual => "<=",
            Self::Greater => ">",
            Self::GreaterEqual => ">=",
        }
    }

    /// The value comparison this general comparison quantifies over.
    #[must_use]
    pub const fn value_op(self) -> ValueOp {
        match self {
            Self::Equal => ValueOp::Equal,
            Self::NotEqual => ValueOp::NotEqual,
            Self::Less => ValueOp::Less,
            Self::LessEqual => ValueOp::LessEqual,
            Self::Greater => ValueOp::Greater,
            Self::GreaterEqual => ValueOp::GreaterEqual,
        }
    }
}

/// A value comparison operator: `eq`, `ne`, `lt`, `le`, `gt`, `ge`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueOp {
    /// `eq`
    Equal,
    /// `ne`
    NotEqual,
    /// `lt`
    Less,
    /// `le`
    LessEqual,
    /// `gt`
    Greater,
    /// `ge`
    GreaterEqual,
}

impl ValueOp {
    /// The operator's source spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Equal => "eq",
            Self::NotEqual => "ne",
            Self::Less => "lt",
            Self::LessEqual => "le",
            Self::Greater => "gt",
            Self::GreaterEqual => "ge",
        }
    }
}

/// A node comparison operator: `is`, `<<`, `>>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeOp {
    /// `is` — the same node.
    Is,
    /// `<<` — precedes in document order.
    Precedes,
    /// `>>` — follows in document order.
    Follows,
}

impl NodeOp {
    /// The operator's source spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Is => "is",
            Self::Precedes => "<<",
            Self::Follows => ">>",
        }
    }
}

/// The key of a lookup expression, `?K`.
#[derive(Debug, Clone, PartialEq)]
pub enum KeySpecifier {
    /// `?name`
    Name(String),
    /// `?3`
    Integer(i64),
    /// `?($e)`
    Computed(Box<Expr>),
    /// `?*`
    Wildcard,
}

/// A parsed XPath expression.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// A numeric or string literal.
    Literal(AtomicValue),
    /// A *deferred* dynamic error: an expression the grammar accepts but that this
    /// implementation cannot evaluate, such as an integer literal beyond the range its
    /// `xs:integer` covers. XPath 3.1 §2.3.1 makes such a limit an overflow at evaluation time —
    /// so it is catchable by `try`/`catch` — rather than a syntax error at compile time.
    Deferred {
        /// The error the expression raises when it is evaluated.
        code: crate::error::XPathErrorCode,
        /// The message the error carries.
        message: String,
    },
    /// `$name`
    VarRef(ExpandedName),
    /// `.`
    ContextItem,
    /// A leading `/`: the root of the tree containing the context node.
    Root,
    /// `E1 / E2` — `E2` is evaluated once per item of `E1`, and node results are merged into
    /// document order without duplicates.
    Path(Box<Expr>, Box<Expr>),
    /// `E1 ! E2` — like [`Expr::Path`] but without the node requirement and without sorting.
    SimpleMap(Box<Expr>, Box<Expr>),
    /// An axis step.
    Step(Box<AxisStep>),
    /// `E[P]`
    Filter(Box<Expr>, Box<Expr>),
    /// A comma-separated sequence. An empty vector is `()`.
    Sequence(Vec<Expr>),
    /// Unary `-` or `+`.
    Unary {
        /// Whether the sign is negative — an even number of `-` signs cancels out.
        negate: bool,
        /// The operand.
        operand: Box<Expr>,
    },
    /// An arithmetic, sequence or string operator.
    Binary {
        /// The operator.
        op: BinaryOp,
        /// The left operand.
        left: Box<Expr>,
        /// The right operand.
        right: Box<Expr>,
    },
    /// `E1 and E2`
    And(Box<Expr>, Box<Expr>),
    /// `E1 or E2`
    Or(Box<Expr>, Box<Expr>),
    /// A general comparison, quantified over the atomized operands.
    General {
        /// The operator.
        op: GeneralOp,
        /// The left operand.
        left: Box<Expr>,
        /// The right operand.
        right: Box<Expr>,
    },
    /// A value comparison of two singletons.
    Value {
        /// The operator.
        op: ValueOp,
        /// The left operand.
        left: Box<Expr>,
        /// The right operand.
        right: Box<Expr>,
    },
    /// A node identity or document-order comparison.
    Node {
        /// The operator.
        op: NodeOp,
        /// The left operand.
        left: Box<Expr>,
        /// The right operand.
        right: Box<Expr>,
    },
    /// `if (C) then T else E`
    If {
        /// The condition, tested by its effective boolean value.
        condition: Box<Expr>,
        /// The branch taken when the condition is true.
        then_branch: Box<Expr>,
        /// The branch taken when the condition is false.
        else_branch: Box<Expr>,
    },
    /// `for $x in E, … return B`
    For {
        /// The bindings, iterated left to right.
        bindings: Vec<VarBinding>,
        /// The body, evaluated once per tuple.
        body: Box<Expr>,
    },
    /// `let $x := E, … return B`
    Let {
        /// The bindings, in order; each may refer to the previous.
        bindings: Vec<VarBinding>,
        /// The body.
        body: Box<Expr>,
    },
    /// `some|every $x in E, … satisfies C`
    Quantified {
        /// `true` for `every`, `false` for `some`.
        every: bool,
        /// The bindings.
        bindings: Vec<VarBinding>,
        /// The condition, tested by its effective boolean value.
        satisfies: Box<Expr>,
    },
    /// `E instance of T`
    InstanceOf(Box<Expr>, SequenceType),
    /// `E treat as T`
    TreatAs(Box<Expr>, SequenceType),
    /// `E castable as T`
    Castable(Box<Expr>, SingleType),
    /// `E cast as T`
    Cast(Box<Expr>, SingleType),
    /// `E castable as T`, where `T` is one of the built-in list types.
    CastableList(Box<Expr>, ListType),
    /// `E cast as T`, where `T` is one of the built-in list types.
    CastList(Box<Expr>, ListType),
    /// A static function call.
    FunctionCall {
        /// The function's expanded name.
        name: ExpandedName,
        /// The arguments, in order.
        arguments: Vec<Expr>,
    },
    /// A static function call with `?` placeholders — a partial application.
    PartialCall {
        /// The function's expanded name.
        name: ExpandedName,
        /// One entry per parameter: `None` is a placeholder.
        arguments: Vec<Option<Expr>>,
    },
    /// `name#arity`
    NamedFunctionRef {
        /// The function's expanded name.
        name: ExpandedName,
        /// The arity selected.
        arity: usize,
    },
    /// `function(…) { … }`
    InlineFunction(Arc<InlineFunction>),
    /// `E(…)` — a dynamic function call, possibly a partial application.
    DynamicCall {
        /// The expression that yields the function item.
        target: Box<Expr>,
        /// One entry per argument: `None` is a `?` placeholder.
        arguments: Vec<Option<Expr>>,
    },
    /// `map { K : V, … }`
    MapConstructor(Vec<(Expr, Expr)>),
    /// `[ … ]` or `array { … }`
    ArrayConstructor {
        /// `true` for the `array { … }` form, whose content is one flattened sequence.
        curly: bool,
        /// The member expressions.
        members: Vec<Expr>,
    },
    /// `E?K` and, with no base, the unary `?K`, which looks up in the context item.
    Lookup {
        /// The expression looked up in, or `None` for the unary form.
        base: Option<Box<Expr>>,
        /// The key.
        key: KeySpecifier,
    },
}

impl Expr {
    /// The empty sequence, `()`.
    #[must_use]
    pub const fn empty_sequence() -> Self {
        Self::Sequence(Vec::new())
    }

    /// Builds a `descendant-or-self::node()` step, which is what `//` abbreviates.
    #[must_use]
    pub fn descendant_or_self() -> Self {
        Self::Step(Box::new(AxisStep {
            axis: Axis::DescendantOrSelf,
            node_test: NodeTest::Kind(KindTest::AnyKind),
            predicates: Vec::new(),
        }))
    }

    /// Whether this expression can be the operand of a path step — that is, whether it is
    /// itself a step or a path.
    #[must_use]
    pub const fn is_step_like(&self) -> bool {
        matches!(self, Self::Step(_) | Self::Path(_, _) | Self::Root)
    }
}

/// Frees an expression tree *iteratively*.
///
/// The parser builds a path chain (`a/a/a/…`), a simple map chain and an operator chain
/// (`1+1+1+…`) with a loop rather than by recursing, so a chain of a hundred thousand steps
/// costs no stack to *build* — but the compiler-derived recursive drop glue would then
/// overflow the stack when the tree is freed, aborting the process on ordinary API usage
/// (letting a compiled expression go out of scope). Detaching the children onto a heap
/// worklist keeps freeing flat.
impl Drop for Expr {
    fn drop(&mut self) {
        let mut worklist: Vec<Expr> = Vec::new();
        detach_children(self, &mut worklist);
        while let Some(mut expr) = worklist.pop() {
            detach_children(&mut expr, &mut worklist);
            // `expr` now holds no owned sub-expression, so its own drop glue is one level deep.
        }
    }
}

/// Moves every directly owned sub-expression of `expr` onto `worklist`, leaving leaves behind.
fn detach_children(expr: &mut Expr, worklist: &mut Vec<Expr>) {
    match expr {
        Expr::Path(left, right)
        | Expr::SimpleMap(left, right)
        | Expr::Filter(left, right)
        | Expr::And(left, right)
        | Expr::Or(left, right)
        | Expr::Binary { left, right, .. }
        | Expr::General { left, right, .. }
        | Expr::Value { left, right, .. }
        | Expr::Node { left, right, .. } => {
            detach(left, worklist);
            detach(right, worklist);
        }
        Expr::Unary { operand, .. } => detach(operand, worklist),
        Expr::InstanceOf(inner, _) | Expr::TreatAs(inner, _) => detach(inner, worklist),
        Expr::Castable(inner, _) | Expr::Cast(inner, _) => detach(inner, worklist),
        Expr::CastableList(inner, _) | Expr::CastList(inner, _) => detach(inner, worklist),
        Expr::Step(step) => worklist.append(&mut std::mem::take(&mut step.predicates)),
        Expr::Sequence(items) | Expr::ArrayConstructor { members: items, .. } => {
            worklist.append(&mut std::mem::take(items));
        }
        Expr::If {
            condition,
            then_branch,
            else_branch,
        } => {
            detach(condition, worklist);
            detach(then_branch, worklist);
            detach(else_branch, worklist);
        }
        Expr::For { bindings, body } | Expr::Let { bindings, body } => {
            detach_bindings(bindings, worklist);
            detach(body, worklist);
        }
        Expr::Quantified {
            bindings,
            satisfies,
            ..
        } => {
            detach_bindings(bindings, worklist);
            detach(satisfies, worklist);
        }
        Expr::FunctionCall { arguments, .. } => {
            worklist.append(&mut std::mem::take(arguments));
        }
        Expr::PartialCall { arguments, .. } => detach_optionals(arguments, worklist),
        Expr::DynamicCall { target, arguments } => {
            detach(target, worklist);
            detach_optionals(arguments, worklist);
        }
        Expr::MapConstructor(entries) => {
            for (key, value) in std::mem::take(entries) {
                worklist.push(key);
                worklist.push(value);
            }
        }
        Expr::Lookup { base, .. } => {
            if let Some(base) = base {
                detach(base, worklist);
            }
        }
        Expr::Literal(_)
        | Expr::Deferred { .. }
        | Expr::VarRef(_)
        | Expr::ContextItem
        | Expr::Root
        | Expr::NamedFunctionRef { .. }
        // An inline function is shared behind an `Arc`, and its body can only be as deep as the
        // parser's own nesting limit allows, so it is left to the ordinary drop glue.
        | Expr::InlineFunction(_) => {}
    }
}

/// Replaces a boxed child with a leaf and queues the old value, unless it was already a leaf.
fn detach(slot: &mut Expr, worklist: &mut Vec<Expr>) {
    let taken = std::mem::replace(slot, Expr::ContextItem);
    if !matches!(taken, Expr::ContextItem) {
        worklist.push(taken);
    }
}

fn detach_bindings(bindings: &mut Vec<VarBinding>, worklist: &mut Vec<Expr>) {
    for binding in std::mem::take(bindings) {
        worklist.push(binding.value);
    }
}

fn detach_optionals(arguments: &mut Vec<Option<Expr>>, worklist: &mut Vec<Expr>) {
    worklist.extend(std::mem::take(arguments).into_iter().flatten());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deep_path_chain_drops_without_recursion() {
        // A flat chain the parser builds iteratively: freeing it must not recurse.
        let mut expr = Expr::Root;
        for _ in 0..200_000 {
            expr = Expr::Path(Box::new(expr), Box::new(Expr::descendant_or_self()));
        }
        drop(expr);
    }

    #[test]
    fn deep_operator_chain_drops_without_recursion() {
        let mut expr = Expr::ContextItem;
        for _ in 0..200_000 {
            expr = Expr::Binary {
                op: BinaryOp::Add,
                left: Box::new(expr),
                right: Box::new(Expr::Literal(AtomicValue::integer(1_i64))),
            };
        }
        drop(expr);
    }

    #[test]
    fn operator_spellings_round_trip() {
        assert_eq!(BinaryOp::IntegerDivide.as_str(), "idiv");
        assert_eq!(GeneralOp::NotEqual.as_str(), "!=");
        assert_eq!(GeneralOp::Less.value_op(), ValueOp::Less);
        assert_eq!(ValueOp::GreaterEqual.as_str(), "ge");
        assert_eq!(NodeOp::Precedes.as_str(), "<<");
    }

    #[test]
    fn helper_constructors() {
        assert_eq!(Expr::empty_sequence(), Expr::Sequence(Vec::new()));
        let step = Expr::descendant_or_self();
        assert!(step.is_step_like());
        assert!(!Expr::ContextItem.is_step_like());
    }
}
