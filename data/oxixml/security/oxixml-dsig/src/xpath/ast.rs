//! The XPath 1.0 abstract syntax tree.
//!
//! The grammar modelled here is the one of
//! [XPath 1.0 §2–§3](https://www.w3.org/TR/1999/REC-xpath-19991116/), with the
//! abbreviations already expanded: `//` becomes an explicit
//! `descendant-or-self::node()` step, `.` becomes `self::node()`, `..` becomes
//! `parent::node()` and `@x` becomes `attribute::x`. That leaves the evaluator
//! with one uniform shape to walk.

/// A qualified name as it was written, before namespace resolution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Name {
    /// The prefix, if the name was written with one.
    pub(crate) prefix: Option<String>,
    /// The local part.
    pub(crate) local: String,
}

impl Name {
    /// Renders the name the way it was written, for error messages.
    pub(crate) fn to_qualified_string(&self) -> String {
        match &self.prefix {
            Some(prefix) => format!("{prefix}:{}", self.local),
            None => self.local.clone(),
        }
    }
}

/// An XPath axis — [XPath §2.2](https://www.w3.org/TR/1999/REC-xpath-19991116/#axes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[expect(
    clippy::enum_variant_names,
    reason = "`Self` is a Rust keyword, so the `self::` axis cannot be spelled without a suffix"
)]
pub(crate) enum Axis {
    /// `ancestor::`
    Ancestor,
    /// `ancestor-or-self::`
    AncestorOrSelf,
    /// `attribute::`
    Attribute,
    /// `child::`
    Child,
    /// `descendant::`
    Descendant,
    /// `descendant-or-self::`
    DescendantOrSelf,
    /// `following::`
    Following,
    /// `following-sibling::`
    FollowingSibling,
    /// `namespace::`
    Namespace,
    /// `parent::`
    Parent,
    /// `preceding::`
    Preceding,
    /// `preceding-sibling::`
    PrecedingSibling,
    /// `self::`
    SelfAxis,
}

impl Axis {
    /// Parses an axis name.
    pub(crate) fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "ancestor" => Self::Ancestor,
            "ancestor-or-self" => Self::AncestorOrSelf,
            "attribute" => Self::Attribute,
            "child" => Self::Child,
            "descendant" => Self::Descendant,
            "descendant-or-self" => Self::DescendantOrSelf,
            "following" => Self::Following,
            "following-sibling" => Self::FollowingSibling,
            "namespace" => Self::Namespace,
            "parent" => Self::Parent,
            "preceding" => Self::Preceding,
            "preceding-sibling" => Self::PrecedingSibling,
            "self" => Self::SelfAxis,
            _ => return None,
        })
    }

    /// Whether the axis runs backwards, which decides how a positional
    /// predicate numbers its nodes.
    pub(crate) const fn is_reverse(self) -> bool {
        matches!(
            self,
            Self::Ancestor
                | Self::AncestorOrSelf
                | Self::Preceding
                | Self::PrecedingSibling
                | Self::Parent
        )
    }
}

/// A node test — [XPath §2.3](https://www.w3.org/TR/1999/REC-xpath-19991116/#node-tests).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum NodeTest {
    /// `*` — any node of the axis's principal type.
    AnyName,
    /// `prefix:*` — any node of the principal type in that namespace.
    AnyNameInNamespace(String),
    /// A qualified name.
    Name(Name),
    /// `node()`
    AnyNode,
    /// `text()`
    Text,
    /// `comment()`
    Comment,
    /// `processing-instruction()` or `processing-instruction('target')`
    ProcessingInstruction(Option<String>),
}

/// One location step.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Step {
    /// The axis to walk.
    pub(crate) axis: Axis,
    /// The test each candidate must pass.
    pub(crate) node_test: NodeTest,
    /// The predicates, applied left to right.
    pub(crate) predicates: Vec<Expr>,
}

/// A binary operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BinaryOp {
    /// `or`
    Or,
    /// `and`
    And,
    /// `=`
    Equal,
    /// `!=`
    NotEqual,
    /// `<`
    Less,
    /// `<=`
    LessOrEqual,
    /// `>`
    Greater,
    /// `>=`
    GreaterOrEqual,
    /// `+`
    Add,
    /// `-`
    Subtract,
    /// `*`
    Multiply,
    /// `div`
    Divide,
    /// `mod`
    Modulo,
}

/// An XPath expression.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Expr {
    /// A binary operation.
    Binary(BinaryOp, Box<Expr>, Box<Expr>),
    /// Unary minus.
    Negate(Box<Expr>),
    /// `left | right`
    Union(Box<Expr>, Box<Expr>),
    /// A numeric literal.
    Number(f64),
    /// A string literal.
    Literal(String),
    /// `$name` — always an error at evaluation time, because XML Signature
    /// binds no variables.
    Variable(Name),
    /// A function call.
    Function {
        /// The function name.
        name: Name,
        /// The argument expressions.
        arguments: Vec<Expr>,
    },
    /// A location path.
    Path {
        /// Whether the path starts at the root node.
        absolute: bool,
        /// The steps, in order.
        steps: Vec<Step>,
    },
    /// A filter expression, optionally followed by more steps:
    /// `id('a')[1]/descendant::b`.
    Filter {
        /// The primary expression being filtered.
        primary: Box<Expr>,
        /// Predicates applied directly to it.
        predicates: Vec<Expr>,
        /// Any steps that follow.
        steps: Vec<Step>,
    },
}

impl Expr {
    /// The `self::node()` step, which `.` abbreviates.
    pub(crate) fn self_step() -> Step {
        Step {
            axis: Axis::SelfAxis,
            node_test: NodeTest::AnyNode,
            predicates: Vec::new(),
        }
    }

    /// The `descendant-or-self::node()` step, which `//` expands to.
    pub(crate) fn descendant_or_self_step() -> Step {
        Step {
            axis: Axis::DescendantOrSelf,
            node_test: NodeTest::AnyNode,
            predicates: Vec::new(),
        }
    }
}
