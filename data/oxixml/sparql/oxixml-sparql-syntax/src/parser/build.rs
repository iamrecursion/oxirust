//! Shared helper types and free functions for algebra construction during parsing.

use super::{Parser, R};
use crate::algebra::{Expression, GraphPattern, PropertyPathExpression};
use crate::term::{TermPattern, TriplePattern, Variable};
use std::collections::HashSet;
use std::mem::take;

/// Either a variable predicate or a property path expression.
#[derive(Clone, Debug)]
pub(crate) enum VariableOrPropertyPath {
    Variable(Variable),
    PropertyPath(PropertyPathExpression),
}

impl From<Variable> for VariableOrPropertyPath {
    fn from(var: Variable) -> Self {
        Self::Variable(var)
    }
}

impl From<PropertyPathExpression> for VariableOrPropertyPath {
    fn from(path: PropertyPathExpression) -> Self {
        Self::PropertyPath(path)
    }
}

/// A triple pattern or a property-path pattern, collected inside a `TriplesBlock`.
#[derive(Debug)]
pub(crate) enum TripleOrPathPattern {
    Triple(TriplePattern),
    Path {
        subject: TermPattern,
        path: PropertyPathExpression,
        object: TermPattern,
    },
}

impl From<TriplePattern> for TripleOrPathPattern {
    fn from(tp: TriplePattern) -> Self {
        Self::Triple(tp)
    }
}

/// A partially-built graph pattern element, as produced by `GraphPatternNotTriples`.
#[derive(Debug)]
pub(crate) enum PartialGraphPattern {
    Optional(GraphPattern, Option<Expression>),
    Minus(GraphPattern),
    Bind(Expression, Variable),
    Filter(Expression),
    Other(GraphPattern),
}

/// A `SELECT` selection option.
pub(crate) enum SelectionOption {
    Distinct,
    Reduced,
    Default,
}

/// A single member of an explicit `SELECT` list.
pub(crate) enum SelectionMember {
    Variable(Variable),
    Expression(Expression, Variable),
}

/// The variables projected by a `SELECT`.
pub(crate) enum SelectionVariables {
    Explicit(Vec<SelectionMember>),
    Star,
    Everything,
}

/// A parsed `SELECT` clause.
pub(crate) struct Selection {
    pub(crate) option: SelectionOption,
    pub(crate) variables: SelectionVariables,
}

impl Selection {
    pub(crate) fn no_op() -> Self {
        Self {
            option: SelectionOption::Default,
            variables: SelectionVariables::Everything,
        }
    }
}

/// Joins two graph patterns, collapsing empty and adjacent basic graph patterns.
pub(crate) fn new_join(l: GraphPattern, r: GraphPattern) -> GraphPattern {
    if let GraphPattern::Bgp { patterns: pl } = &l {
        if pl.is_empty() {
            return r;
        }
    }
    if let GraphPattern::Bgp { patterns: pr } = &r {
        if pr.is_empty() {
            return l;
        }
    }

    match (l, r) {
        (GraphPattern::Bgp { patterns: mut pl }, GraphPattern::Bgp { patterns: pr }) => {
            pl.extend(pr);
            GraphPattern::Bgp { patterns: pl }
        }
        (GraphPattern::Bgp { patterns }, other) | (other, GraphPattern::Bgp { patterns })
            if patterns.is_empty() =>
        {
            other
        }
        (l, r) => GraphPattern::Join {
            left: Box::new(l),
            right: Box::new(r),
        },
    }
}

/// Builds a graph pattern from a `TriplesBlock`, splitting property paths from basic graph patterns.
///
/// A `TriplesBlock` alternating triple patterns and property paths (`?s <p> ?o . ?s2 <a>/<b> ?o2 .
/// ...`) splits into one `GraphPattern` element per run, which are then joined into a left-deep
/// `Join` tree one level per extra element — exactly like the other flat-chain productions, this
/// charges [`Parser::enter_recursion`] once per join level before it is built, so a very long
/// alternating block is rejected instead of producing a tree deep enough to overflow the stack
/// when it is later displayed, compared, or dropped.
pub(crate) fn build_bgp(
    parser: &mut Parser<'_>,
    patterns: Vec<TripleOrPathPattern>,
) -> R<GraphPattern> {
    let mut bgp = Vec::new();
    let mut elements = Vec::with_capacity(patterns.len());
    for pattern in patterns {
        match pattern {
            TripleOrPathPattern::Triple(t) => bgp.push(t),
            TripleOrPathPattern::Path {
                subject,
                path,
                object,
            } => {
                if !bgp.is_empty() {
                    elements.push(GraphPattern::Bgp {
                        patterns: take(&mut bgp),
                    });
                }
                elements.push(GraphPattern::Path {
                    subject,
                    path,
                    object,
                })
            }
        }
    }
    if !bgp.is_empty() {
        elements.push(GraphPattern::Bgp { patterns: bgp });
    }
    fold_graph_patterns(parser, elements)
}

/// Folds a non-empty `Vec` of graph patterns into a left-deep `Join` tree, charging
/// [`Parser::enter_recursion`] once per extra element before it is joined in.
///
/// Used both by [`build_bgp`] (a `TriplesBlock`'s split elements) and by `DELETE WHERE`'s quad
/// patterns grouped by graph, which fold the very same way. An empty list yields the default
/// (empty) [`GraphPattern`], matching `elements.into_iter().reduce(new_join).unwrap_or_default()`.
pub(crate) fn fold_graph_patterns(
    parser: &mut Parser<'_>,
    elements: impl IntoIterator<Item = GraphPattern>,
) -> R<GraphPattern> {
    let mut iter = elements.into_iter();
    let Some(mut acc) = iter.next() else {
        return Ok(GraphPattern::default());
    };
    let mut charged = 0_u32;
    for next in iter {
        parser.enter_recursion()?;
        charged += 1;
        acc = new_join(acc, next);
    }
    parser.leave_recursion_by(charged);
    Ok(acc)
}

/// Folds a non-empty iterator, returning an error message if it is empty.
pub(crate) fn not_empty_fold<T>(
    iter: impl Iterator<Item = T>,
    combine: impl Fn(T, T) -> T,
) -> Result<T, &'static str> {
    iter.fold(None, |a, b| match a {
        Some(av) => Some(combine(av, b)),
        None => Some(b),
    })
    .ok_or("The iterator should not be empty")
}

/// Whether every variable used in `expression` is contained in `variables`.
pub(crate) fn are_variables_bound(expression: &Expression, variables: &HashSet<Variable>) -> bool {
    match expression {
        Expression::NamedNode(_)
        | Expression::Literal(_)
        | Expression::Bound(_)
        | Expression::Coalesce(_)
        | Expression::Exists(_) => true,
        Expression::Variable(var) => variables.contains(var),
        Expression::Or(a, b) | Expression::And(a, b) => {
            are_variables_bound(a, variables) && are_variables_bound(b, variables)
        }
        Expression::In(a, b) => {
            are_variables_bound(a, variables) && b.iter().all(|b| are_variables_bound(b, variables))
        }
        Expression::FunctionCall(_, parameters) => {
            parameters.iter().all(|p| are_variables_bound(p, variables))
        }
        Expression::If(a, b, c) => {
            are_variables_bound(a, variables)
                && are_variables_bound(b, variables)
                && are_variables_bound(c, variables)
        }
    }
}
