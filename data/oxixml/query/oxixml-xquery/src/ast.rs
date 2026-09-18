//! The syntax tree the XQuery parser produces.
//!
//! The tree is deliberately *thin*. Every node records the byte range of the query text it was
//! parsed from and whether its subtree contains any XQuery-only syntax; a node whose subtree is
//! pure XPath 3.1 never needs a representation of its own, because the compiler hands the source
//! range straight to [`oxixml_xpath`]. Only the constructs XPath does not have — FLWOR, the node
//! constructors, `typeswitch`, `switch`, `try`/`catch`, `ordered`/`unordered` and the prolog —
//! and the few forms that must be able to *contain* them get a variant here.
//!
//! See [`crate::compile`] for how the two halves are stitched together.

use crate::lexer::Token;
use oxixml_infoset::AtomicValue;

/// A parsed expression: the source range it covers, whether it needs this crate's evaluator, and
/// its shape.
#[derive(Debug, Clone)]
pub(crate) struct Syntax {
    pub(crate) start: usize,
    pub(crate) end: usize,
    /// Whether the subtree contains syntax the XPath compiler cannot parse.
    pub(crate) xquery_only: bool,
    pub(crate) kind: SyntaxKind,
}

impl Syntax {
    pub(crate) fn new(start: usize, end: usize, kind: SyntaxKind) -> Self {
        let xquery_only = kind.is_inherently_xquery() || kind.children().any(|c| c.xquery_only);
        Self {
            start,
            end,
            xquery_only,
            kind,
        }
    }

    /// A node that is XQuery-only regardless of its children.
    pub(crate) fn native(start: usize, end: usize, kind: SyntaxKind) -> Self {
        Self {
            start,
            end,
            xquery_only: true,
            kind,
        }
    }

    /// An empty sequence expression, used where the grammar allows content to be omitted.
    pub(crate) fn empty(at: usize) -> Self {
        Self {
            start: at,
            end: at,
            xquery_only: true,
            kind: SyntaxKind::EmptySequence,
        }
    }
}

/// The shape of an expression.
#[derive(Debug, Clone)]
pub(crate) enum SyntaxKind {
    /// A form the XPath compiler understands once each XQuery-only child has been replaced by a
    /// variable reference: a literal, a name, a function call, an operator, a comparison, a type
    /// expression, a map or array constructor, a parenthesized expression.
    Generic(Vec<Syntax>),
    /// `if (C) then T else E`.
    If {
        condition: Box<Syntax>,
        then_branch: Box<Syntax>,
        else_branch: Box<Syntax>,
    },
    /// The XPath-shaped `for`, `let`, `some` and `every` expressions.
    Binding {
        kind: BindingKind,
        bindings: Vec<SimpleBinding>,
        body: Box<Syntax>,
    },
    /// `and` / `or`, which short-circuit.
    Logical {
        or: bool,
        left: Box<Syntax>,
        right: Box<Syntax>,
    },
    /// A path expression.
    Path(Box<PathExpr>),
    /// A primary expression followed by predicates, argument lists and lookups.
    Postfix {
        primary: Box<Syntax>,
        postfixes: Vec<Postfix>,
    },
    /// `function($x) { … }`.
    Inline(Box<InlineFunction>),
    /// A FLWOR expression.
    Flwor(Box<Flwor>),
    /// A direct or computed node constructor.
    Construct(Box<Constructor>),
    /// `typeswitch (E) case … default …`.
    TypeSwitch(Box<TypeSwitchExpr>),
    /// `switch (E) case … default …`.
    Switch(Box<SwitchExpr>),
    /// `try { … } catch … { … }`.
    Try(Box<TryExpr>),
    /// `ordered { E }` and `unordered { E }`.
    Wrap(Box<Syntax>),
    /// `(# pragma #) { E }`; the pragma names are checked, then ignored.
    Extension {
        pragmas: Vec<String>,
        body: Box<Syntax>,
    },
    /// A literal whose source spelling is not valid XPath, because it used an entity reference.
    Literal(AtomicValue),
    /// The context item, `.`, reached through a construct that has no source of its own — the
    /// unary lookup `?K`, which the specification defines as `.?K`.
    ContextItem,
    /// A `` `[ … ]` `` string constructor.
    StringTemplate(Vec<ContentItem>),
    /// `()`, produced where the grammar allows content to be omitted.
    EmptySequence,
    /// `validate { … }`, which this processor rejects statically.
    Validate,
}

impl SyntaxKind {
    /// Whether this shape exists only in XQuery, so that the XPath compiler can never see it.
    const fn is_inherently_xquery(&self) -> bool {
        matches!(
            self,
            Self::Flwor(_)
                | Self::Construct(_)
                | Self::TypeSwitch(_)
                | Self::Switch(_)
                | Self::Try(_)
                | Self::Wrap(_)
                | Self::Extension { .. }
                | Self::Literal(_)
                | Self::StringTemplate(_)
                | Self::EmptySequence
                | Self::Validate
        )
    }

    /// The direct sub-expressions, for contamination analysis.
    fn children(&self) -> Box<dyn Iterator<Item = &Syntax> + '_> {
        match self {
            Self::Generic(children) => Box::new(children.iter()),
            Self::If {
                condition,
                then_branch,
                else_branch,
            } => Box::new(
                [
                    condition.as_ref(),
                    then_branch.as_ref(),
                    else_branch.as_ref(),
                ]
                .into_iter(),
            ),
            Self::Binding { bindings, body, .. } => Box::new(
                bindings
                    .iter()
                    .map(|binding| &binding.value)
                    .chain(std::iter::once(body.as_ref())),
            ),
            Self::Logical { left, right, .. } => {
                Box::new([left.as_ref(), right.as_ref()].into_iter())
            }
            Self::Path(path) => Box::new(path.steps.iter().flat_map(PathStep::children)),
            Self::Postfix { primary, postfixes } => Box::new(
                std::iter::once(primary.as_ref())
                    .chain(postfixes.iter().flat_map(Postfix::children)),
            ),
            Self::Inline(inline) => Box::new(std::iter::once(&inline.body)),
            _ => Box::new(std::iter::empty()),
        }
    }
}

/// Which of the four XPath binding expressions a [`SyntaxKind::Binding`] is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BindingKind {
    For,
    Let,
    Some,
    Every,
}

/// One `$name in E` or `$name := E` binding.
#[derive(Debug, Clone)]
pub(crate) struct SimpleBinding {
    pub(crate) name: String,
    pub(crate) declared_type: Option<String>,
    pub(crate) value: Syntax,
}

/// A path expression: an optional root, then a series of steps.
#[derive(Debug, Clone)]
pub(crate) struct PathExpr {
    /// `true` when the path starts at the root of the tree containing the context node.
    pub(crate) rooted: bool,
    /// `true` when the leading `/` was in fact `//`.
    pub(crate) rooted_descendant: bool,
    pub(crate) steps: Vec<PathStep>,
}

/// One step of a path, together with the separator that introduced it.
#[derive(Debug, Clone)]
pub(crate) struct PathStep {
    /// `true` when the step was reached through `//`, which inserts
    /// `descendant-or-self::node()`.
    pub(crate) descendant: bool,
    /// `true` when the step was reached through `!` rather than `/`.
    pub(crate) simple_map: bool,
    /// `true` when the step is an axis step on one of the five reverse axes, whose predicates
    /// number the nodes from the context node outwards rather than in document order.
    pub(crate) reverse: bool,
    /// The step's expression, without its predicates.
    pub(crate) base: Syntax,
    pub(crate) predicates: Vec<Syntax>,
}

impl PathStep {
    fn children(&self) -> impl Iterator<Item = &Syntax> {
        std::iter::once(&self.base).chain(self.predicates.iter())
    }
}

/// A postfix applied to a primary expression.
#[derive(Debug, Clone)]
pub(crate) enum Postfix {
    /// `[ E ]`
    Predicate(Syntax),
    /// `( a, b, ? )` — a dynamic call, possibly a partial application.
    Arguments(Vec<Option<Syntax>>),
    /// `? K`
    Lookup(LookupKey),
}

impl Postfix {
    fn children(&self) -> Box<dyn Iterator<Item = &Syntax> + '_> {
        match self {
            Self::Predicate(expr) => Box::new(std::iter::once(expr)),
            Self::Arguments(arguments) => Box::new(arguments.iter().flatten()),
            Self::Lookup(LookupKey::Computed(expr)) => Box::new(std::iter::once(expr.as_ref())),
            Self::Lookup(_) => Box::new(std::iter::empty()),
        }
    }
}

/// The key of a `?` lookup.
#[derive(Debug, Clone)]
pub(crate) enum LookupKey {
    Name(String),
    Integer(i64),
    Computed(Box<Syntax>),
    Wildcard,
}

/// `function($x as T) as R { … }`.
#[derive(Debug, Clone)]
pub(crate) struct InlineFunction {
    pub(crate) parameters: Vec<(String, Option<String>)>,
    pub(crate) result_type: Option<String>,
    pub(crate) body: Syntax,
}

// --- FLWOR ---------------------------------------------------------------------------------

/// A FLWOR expression: one or more clauses and a `return`.
#[derive(Debug, Clone)]
pub(crate) struct Flwor {
    pub(crate) clauses: Vec<FlworClause>,
    pub(crate) result: Syntax,
}

/// One clause of a FLWOR expression.
#[derive(Debug, Clone)]
pub(crate) enum FlworClause {
    For(Vec<ForBinding>),
    Let(Vec<LetBinding>),
    Where(Syntax),
    GroupBy(Vec<GroupingSpec>),
    /// `order by` — always stable, which `stable order by` also asks for.
    OrderBy(Vec<OrderSpec>),
    Count(String),
    Window(Box<WindowClause>),
}

/// `for $x as T at $i allowing empty in E`.
#[derive(Debug, Clone)]
pub(crate) struct ForBinding {
    pub(crate) name: String,
    pub(crate) declared_type: Option<String>,
    pub(crate) allowing_empty: bool,
    pub(crate) positional: Option<String>,
    pub(crate) value: Syntax,
}

/// `let $x as T := E`.
#[derive(Debug, Clone)]
pub(crate) struct LetBinding {
    pub(crate) name: String,
    pub(crate) declared_type: Option<String>,
    pub(crate) value: Syntax,
}

/// One `group by` specification.
#[derive(Debug, Clone)]
pub(crate) struct GroupingSpec {
    pub(crate) name: String,
    pub(crate) declared_type: Option<String>,
    pub(crate) value: Option<Syntax>,
    pub(crate) collation: Option<String>,
}

/// One `order by` specification.
#[derive(Debug, Clone)]
pub(crate) struct OrderSpec {
    pub(crate) key: Syntax,
    pub(crate) ascending: bool,
    /// `Some(true)` for `empty greatest`, `Some(false)` for `empty least`, `None` to take the
    /// prolog's default.
    pub(crate) empty_greatest: Option<bool>,
    pub(crate) collation: Option<String>,
}

/// A `tumbling window` or `sliding window` clause.
#[derive(Debug, Clone)]
pub(crate) struct WindowClause {
    pub(crate) tumbling: bool,
    pub(crate) name: String,
    pub(crate) declared_type: Option<String>,
    pub(crate) value: Syntax,
    pub(crate) start: WindowCondition,
    pub(crate) end: Option<WindowCondition>,
    /// `true` for `only end …`, which discards a final incomplete window.
    pub(crate) only: bool,
}

/// The variables and predicate of a window `start` or `end` condition.
#[derive(Debug, Clone, Default)]
pub(crate) struct WindowCondition {
    pub(crate) current: Option<String>,
    pub(crate) positional: Option<String>,
    pub(crate) previous: Option<String>,
    pub(crate) next: Option<String>,
    pub(crate) when: Option<Syntax>,
}

// --- typeswitch, switch, try ------------------------------------------------------------------

/// `typeswitch (E) case $v as T return R … default $d return D`.
#[derive(Debug, Clone)]
pub(crate) struct TypeSwitchExpr {
    pub(crate) operand: Syntax,
    pub(crate) cases: Vec<TypeCase>,
    pub(crate) default_variable: Option<String>,
    pub(crate) default_body: Syntax,
}

/// One `case` of a `typeswitch`.
#[derive(Debug, Clone)]
pub(crate) struct TypeCase {
    pub(crate) variable: Option<String>,
    /// The alternatives, each the source text of a `SequenceType`.
    pub(crate) types: Vec<String>,
    pub(crate) body: Syntax,
}

/// `switch (E) case C return R … default return D`.
#[derive(Debug, Clone)]
pub(crate) struct SwitchExpr {
    pub(crate) operand: Syntax,
    pub(crate) cases: Vec<(Vec<Syntax>, Syntax)>,
    pub(crate) default_body: Syntax,
}

/// `try { … } catch … { … }`.
#[derive(Debug, Clone)]
pub(crate) struct TryExpr {
    pub(crate) body: Syntax,
    pub(crate) catches: Vec<CatchClause>,
}

/// One `catch` clause.
#[derive(Debug, Clone)]
pub(crate) struct CatchClause {
    pub(crate) tests: Vec<CatchTest>,
    pub(crate) body: Syntax,
}

/// A name test in a `catch` clause.
#[derive(Debug, Clone)]
pub(crate) enum CatchTest {
    /// `*`
    Any,
    /// `p:*`, resolved to a namespace URI.
    Namespace(Option<String>),
    /// `*:local`
    Local(String),
    /// A lexical `QName`.
    Name(String),
}

// --- constructors ------------------------------------------------------------------------------

/// A direct or computed node constructor.
#[derive(Debug, Clone)]
pub(crate) enum Constructor {
    DirectElement(Box<DirectElement>),
    DirectComment(String),
    DirectProcessingInstruction {
        target: String,
        content: String,
    },
    ComputedElement {
        name: NameSpec,
        content: Option<Syntax>,
    },
    ComputedAttribute {
        name: NameSpec,
        content: Option<Syntax>,
    },
    ComputedDocument(Syntax),
    ComputedText(Option<Syntax>),
    ComputedComment(Option<Syntax>),
    ComputedProcessingInstruction {
        name: NameSpec,
        content: Option<Syntax>,
    },
    ComputedNamespace {
        prefix: NameSpec,
        uri: Option<Syntax>,
    },
}

/// `<a x="1">…</a>`.
#[derive(Debug, Clone)]
pub(crate) struct DirectElement {
    pub(crate) name: String,
    /// The namespace declaration attributes, as `(prefix, uri)`; `None` is the default namespace.
    pub(crate) namespaces: Vec<(Option<String>, String)>,
    pub(crate) attributes: Vec<DirectAttribute>,
    pub(crate) content: Vec<ContentItem>,
}

/// One ordinary attribute of a direct element constructor.
#[derive(Debug, Clone)]
pub(crate) struct DirectAttribute {
    pub(crate) name: String,
    pub(crate) value: Vec<ContentItem>,
}

/// A piece of a direct constructor's content.
#[derive(Debug, Clone)]
pub(crate) enum ContentItem {
    /// Literal character data, with references already resolved.
    Text(String),
    /// Literal character data that consists only of whitespace between two constructors, which
    /// `declare boundary-space strip` removes.
    BoundaryWhitespace(String),
    /// `{ E }`
    Enclosed(Syntax),
    /// A directly nested constructor — `<b/>` inside `<a>…</a>`.
    ///
    /// [XQuery 3.1 §3.9.1.3](https://www.w3.org/TR/xquery-31/#id-content) treats it differently
    /// from an enclosed expression: "Each nested direct constructor is evaluated … resulting in a
    /// new element, comment, or processing instruction node. Then: The parent property of the
    /// resulting node is then set to the newly constructed element node." The node is *not*
    /// copied, so copy-namespaces mode never touches it.
    Nested(Syntax),
}

/// The name of a computed constructor: either written out or computed.
#[derive(Debug, Clone)]
pub(crate) enum NameSpec {
    /// A lexical `QName` or `NCName` written in the query.
    Lexical(String),
    /// `{ E }`
    Computed(Box<Syntax>),
}

// --- diagnostics ---------------------------------------------------------------------------

/// Builds the `XPST0003` message for an unexpected token.
pub(crate) fn unexpected(token: &Token, wanted: &str) -> String {
    format!("expected {wanted}, found {}", token.describe())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generic(children: Vec<Syntax>) -> Syntax {
        Syntax::new(0, 1, SyntaxKind::Generic(children))
    }

    #[test]
    fn contamination_propagates_from_children() {
        let clean = generic(Vec::new());
        assert!(!clean.xquery_only);

        let dirty = Syntax::new(0, 1, SyntaxKind::EmptySequence);
        assert!(dirty.xquery_only);

        assert!(generic(vec![clean.clone(), dirty]).xquery_only);
        assert!(!generic(vec![clean.clone(), clean]).xquery_only);
    }

    #[test]
    fn every_native_shape_is_inherently_xquery() {
        assert!(SyntaxKind::Validate.is_inherently_xquery());
        assert!(SyntaxKind::EmptySequence.is_inherently_xquery());
        assert!(!SyntaxKind::Generic(Vec::new()).is_inherently_xquery());
    }
}
