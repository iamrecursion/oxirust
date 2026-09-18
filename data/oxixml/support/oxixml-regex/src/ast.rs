//! The abstract syntax tree produced by the parser.
//!
//! Both dialects parse into the same tree. `.` and the shorthand escapes are
//! resolved into [`ClassExpr`](crate::charclass::ClassExpr) values during
//! parsing, so the tree contains only literals, classes, anchors, groups,
//! concatenations, alternations, repetitions, and back-references. A tree that
//! contains no [`Ast::Backref`] can be executed by the linear-time Pike VM;
//! otherwise it is executed by the back-tracking interpreter.

use crate::charclass::ClassExpr;

/// A node of the parsed regular expression.
#[derive(Debug, Clone)]
pub(crate) enum Ast {
    /// The empty expression, matching the empty string.
    Empty,
    /// A single literal character.
    Literal(char),
    /// A character class (also the resolved form of `.` and of shorthand and
    /// property escapes).
    Class(ClassExpr),
    /// The `^` anchor (XPath dialect only).
    AnchorStart,
    /// The `$` anchor (XPath dialect only).
    AnchorEnd,
    /// A parenthesised group. `index` is `Some` for a capturing group and
    /// `None` for a `(?:...)` non-capturing group.
    Group {
        /// The 1-based capture index, or `None` when non-capturing.
        index: Option<usize>,
        /// The group body.
        inner: Box<Ast>,
    },
    /// A sequence of expressions matched in order.
    Concat(Vec<Ast>),
    /// An ordered alternation; earlier branches have priority.
    Alt(Vec<Ast>),
    /// A repetition. `max` of `None` means unbounded; `greedy` distinguishes
    /// greedy (`a*`) from reluctant (`a*?`) quantifiers.
    Repeat {
        /// The repeated body.
        inner: Box<Ast>,
        /// The minimum number of repetitions.
        min: u32,
        /// The maximum number of repetitions, or `None` if unbounded.
        max: Option<u32>,
        /// Whether the quantifier is greedy.
        greedy: bool,
    },
    /// A back-reference `\N` to the text captured by group `N` (XPath dialect
    /// only).
    Backref(usize),
}

impl Ast {
    /// Returns `true` if this expression can match the empty string.
    ///
    /// A back-reference is treated as nullable: a group that captured nothing
    /// (or never participated) is matched by the empty string.
    pub(crate) fn is_nullable(&self) -> bool {
        match self {
            Ast::Empty | Ast::AnchorStart | Ast::AnchorEnd | Ast::Backref(_) => true,
            Ast::Literal(_) | Ast::Class(_) => false,
            Ast::Group { inner, .. } => inner.is_nullable(),
            Ast::Repeat { inner, min, .. } => *min == 0 || inner.is_nullable(),
            Ast::Concat(items) => items.iter().all(Ast::is_nullable),
            Ast::Alt(items) => items.iter().any(Ast::is_nullable),
        }
    }

    /// The largest lower bound of a counted repetition whose body cannot match
    /// the empty string, if any exceeds `limit`.
    ///
    /// Such a repetition needs at least that many characters of input, so an
    /// input shorter than the bound can never satisfy it — which is what lets
    /// a compiler refuse to unroll it and still be correct. See
    /// [`crate::Regex`]'s `unexpandable_min`.
    pub(crate) fn unexpandable_min(&self, limit: usize) -> Option<usize> {
        let here = match self {
            Ast::Repeat { inner, min, .. } => {
                (*min as usize > limit && !inner.is_nullable()).then_some(*min as usize)
            }
            _ => None,
        };
        let nested = match self {
            Ast::Group { inner, .. } | Ast::Repeat { inner, .. } => inner.unexpandable_min(limit),
            Ast::Concat(items) | Ast::Alt(items) => items
                .iter()
                .filter_map(|item| item.unexpandable_min(limit))
                .max(),
            _ => None,
        };
        here.into_iter().chain(nested).max()
    }

    /// Returns `true` if the tree contains a back-reference anywhere, which
    /// forces evaluation onto the back-tracking interpreter.
    pub(crate) fn has_backref(&self) -> bool {
        match self {
            Ast::Backref(_) => true,
            Ast::Group { inner, .. } => inner.has_backref(),
            Ast::Repeat { inner, .. } => inner.has_backref(),
            Ast::Concat(items) | Ast::Alt(items) => items.iter().any(Ast::has_backref),
            Ast::Empty | Ast::Literal(_) | Ast::Class(_) | Ast::AnchorStart | Ast::AnchorEnd => {
                false
            }
        }
    }
}
