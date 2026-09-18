//! A minimal SPARQL `ASK` evaluator, used by `rdfa_suite.rs` to check an extracted graph against
//! the expectation of a W3C RDFa test case.
//!
//! Every case in the [RDFa Test Suite](https://rdfa.info/test-suite/) states its expectation as a
//! SPARQL `ASK`, so checking a case means answering one. The queries are parsed by
//! [`oxixml_sparql_syntax`]; what is implemented here is only the evaluation, and only the
//! fragment of the algebra those queries actually use: basic graph patterns (with blank nodes in
//! the query acting as non-distinguished variables), `OPTIONAL`, `UNION`, `MINUS`, `FILTER`, and
//! the `isBLANK` / `isIRI` / `isLITERAL` / `LANG` / `STR` / `BOUND` / `=` / `!` / `&&` / `||`
//! builtins.
//!
//! The loud-failure guarantee below is **pattern-level only**: an algebra shape [`evaluate`]
//! does not implement (its `_ => None` fallback) propagates as `None` out of [`ask`], which the
//! caller turns into an explicit "unsupported construct" failure rather than a silent pass.
//! Inside a `FILTER`, though, an unrecognized [`Expression`] (an unimplemented `FunctionCall`, for
//! instance) evaluates to `None`, and [`effective_boolean`] correctly maps that to `false` per
//! SPARQL's own effective-boolean-value rules for an unbound or erroring expression — the same
//! outcome as a builtin this evaluator *does* implement genuinely evaluating to `false`. The two
//! are indistinguishable from the caller's side, so a case whose `expectedResults` is `false` and
//! whose `ASK` relies on a `FILTER` builtin this file does not implement can pass without that
//! `FILTER` having meaningfully exercised the extracted graph at all.
//!
//! This file is a test-support module rather than a test; it declares one `#[test]` covering the
//! evaluator itself so that `cargo test` exercises it directly.

#![allow(dead_code)]

use std::collections::HashMap;

use oxixml_model::{Literal, NamedNode, Term, Triple};
use oxixml_sparql_syntax::algebra::{Expression, GraphPattern};
use oxixml_sparql_syntax::query::Query;
use oxixml_sparql_syntax::term::{NamedNodePattern, TermPattern};

/// One solution mapping: variable name to term.
type Solution = HashMap<String, Term>;

/// Answers a SPARQL `ASK` against `graph`.
///
/// Returns `None` when the query is not an `ASK`, or uses a construct the evaluator does not
/// implement.
#[must_use]
pub fn ask(query: &Query, graph: &[Triple]) -> Option<bool> {
    let Query::Ask(ask) = query else {
        return None;
    };
    let solutions = evaluate(&ask.pattern, graph)?;
    Some(!solutions.is_empty())
}

fn evaluate(pattern: &GraphPattern, graph: &[Triple]) -> Option<Vec<Solution>> {
    match pattern {
        GraphPattern::Bgp { patterns } => {
            let mut solutions = vec![Solution::new()];
            for triple in patterns {
                let mut next = Vec::new();
                for solution in &solutions {
                    for candidate in graph {
                        if let Some(extended) = match_triple(triple, candidate, solution) {
                            next.push(extended);
                        }
                    }
                }
                solutions = next;
                if solutions.is_empty() {
                    break;
                }
            }
            Some(solutions)
        }
        GraphPattern::Join { left, right } => {
            let left = evaluate(left, graph)?;
            let right = evaluate(right, graph)?;
            Some(join(&left, &right))
        }
        GraphPattern::LeftJoin {
            left,
            right,
            expression,
        } => {
            let left = evaluate(left, graph)?;
            let right = evaluate(right, graph)?;
            let mut out = Vec::new();
            for solution in &left {
                let mut matched = false;
                for candidate in &right {
                    if let Some(merged) = merge(solution, candidate) {
                        let keep = expression.as_ref().is_none_or(|expression| {
                            effective_boolean(&eval_expression(expression, &merged, graph))
                        });
                        if keep {
                            out.push(merged);
                            matched = true;
                        }
                    }
                }
                if !matched {
                    out.push(solution.clone());
                }
            }
            Some(out)
        }
        GraphPattern::Union { left, right } => {
            let mut left = evaluate(left, graph)?;
            left.extend(evaluate(right, graph)?);
            Some(left)
        }
        GraphPattern::Minus { left, right } => {
            let left = evaluate(left, graph)?;
            let right = evaluate(right, graph)?;
            Some(
                left.into_iter()
                    .filter(|solution| !right.iter().any(|other| compatible(solution, other)))
                    .collect(),
            )
        }
        GraphPattern::Filter { expr, inner } => {
            let inner = evaluate(inner, graph)?;
            Some(
                inner
                    .into_iter()
                    .filter(|solution| effective_boolean(&eval_expression(expr, solution, graph)))
                    .collect(),
            )
        }
        GraphPattern::Extend {
            inner,
            variable,
            expression,
        } => {
            let inner = evaluate(inner, graph)?;
            Some(
                inner
                    .into_iter()
                    .map(|mut solution| {
                        if let Some(value) = eval_expression(expression, &solution, graph) {
                            solution.insert(variable.as_str().to_owned(), value);
                        }
                        solution
                    })
                    .collect(),
            )
        }
        GraphPattern::Project { inner, .. }
        | GraphPattern::Distinct { inner }
        | GraphPattern::Reduced { inner }
        | GraphPattern::OrderBy { inner, .. }
        | GraphPattern::Slice { inner, .. } => evaluate(inner, graph),
        _ => None,
    }
}

fn join(left: &[Solution], right: &[Solution]) -> Vec<Solution> {
    let mut out = Vec::new();
    for a in left {
        for b in right {
            if let Some(merged) = merge(a, b) {
                out.push(merged);
            }
        }
    }
    out
}

fn compatible(a: &Solution, b: &Solution) -> bool {
    b.iter()
        .all(|(name, value)| a.get(name).is_none_or(|existing| existing == value))
}

fn merge(a: &Solution, b: &Solution) -> Option<Solution> {
    let mut out = a.clone();
    for (name, value) in b {
        match out.get(name) {
            Some(existing) if existing != value => return None,
            Some(_) => {}
            None => {
                out.insert(name.clone(), value.clone());
            }
        }
    }
    Some(out)
}

fn match_triple(
    pattern: &oxixml_sparql_syntax::term::TriplePattern,
    triple: &Triple,
    solution: &Solution,
) -> Option<Solution> {
    let mut out = solution.clone();
    bind_term(
        &pattern.subject,
        &Term::from(triple.subject.clone()),
        &mut out,
    )?;
    bind_named(&pattern.predicate, &triple.predicate, &mut out)?;
    bind_term(&pattern.object, &triple.object, &mut out)?;
    Some(out)
}

/// The name a query blank node is bound under. SPARQL blank nodes in a graph pattern behave as
/// variables that cannot be projected, which for an `ASK` is indistinguishable from a variable.
fn blank_variable(label: &str) -> String {
    format!(" bnode:{label}")
}

fn bind_term(pattern: &TermPattern, value: &Term, solution: &mut Solution) -> Option<()> {
    match pattern {
        TermPattern::Variable(variable) => bind(variable.as_str(), value, solution),
        TermPattern::BlankNode(node) => bind(&blank_variable(node.as_str()), value, solution),
        TermPattern::NamedNode(node) => (value == &Term::from(node.clone())).then_some(()),
        TermPattern::Literal(literal) => (value == &Term::from(literal.clone())).then_some(()),
        #[allow(unreachable_patterns)]
        _ => None,
    }
}

fn bind_named(
    pattern: &NamedNodePattern,
    value: &NamedNode,
    solution: &mut Solution,
) -> Option<()> {
    match pattern {
        NamedNodePattern::Variable(variable) => {
            bind(variable.as_str(), &Term::from(value.clone()), solution)
        }
        NamedNodePattern::NamedNode(node) => (node == value).then_some(()),
    }
}

fn bind(name: &str, value: &Term, solution: &mut Solution) -> Option<()> {
    match solution.get(name) {
        Some(existing) if existing != value => None,
        Some(_) => Some(()),
        None => {
            solution.insert(name.to_owned(), value.clone());
            Some(())
        }
    }
}

fn eval_expression(expression: &Expression, solution: &Solution, graph: &[Triple]) -> Option<Term> {
    match expression {
        Expression::NamedNode(node) => Some(node.clone().into()),
        Expression::Literal(literal) => Some(literal.clone().into()),
        Expression::Variable(variable) => solution.get(variable.as_str()).cloned(),
        Expression::Bound(variable) => Some(boolean(solution.contains_key(variable.as_str()))),
        Expression::And(left, right) => Some(boolean(
            effective_boolean(&eval_expression(left, solution, graph))
                && effective_boolean(&eval_expression(right, solution, graph)),
        )),
        Expression::Or(left, right) => Some(boolean(
            effective_boolean(&eval_expression(left, solution, graph))
                || effective_boolean(&eval_expression(right, solution, graph)),
        )),
        Expression::If(condition, then, otherwise) => {
            if effective_boolean(&eval_expression(condition, solution, graph)) {
                eval_expression(then, solution, graph)
            } else {
                eval_expression(otherwise, solution, graph)
            }
        }
        Expression::Coalesce(items) => items
            .iter()
            .find_map(|item| eval_expression(item, solution, graph)),
        Expression::Exists(pattern) => {
            let solutions = evaluate(pattern, graph)?;
            Some(boolean(
                solutions.iter().any(|other| compatible(other, solution)),
            ))
        }
        Expression::In(needle, haystack) => {
            let needle = eval_expression(needle, solution, graph)?;
            Some(boolean(haystack.iter().any(|item| {
                eval_expression(item, solution, graph).as_ref() == Some(&needle)
            })))
        }
        Expression::FunctionCall(function, arguments) => {
            let name = function.as_str().rsplit('#').next().unwrap_or("");
            let argument = |index: usize| -> Option<Term> {
                arguments
                    .get(index)
                    .and_then(|expression| eval_expression(expression, solution, graph))
            };
            match name {
                "logical-not" => Some(boolean(!effective_boolean(&argument(0)))),
                "logical-and" => Some(boolean(
                    effective_boolean(&argument(0)) && effective_boolean(&argument(1)),
                )),
                "logical-or" => Some(boolean(
                    effective_boolean(&argument(0)) || effective_boolean(&argument(1)),
                )),
                "isBlank" => Some(boolean(matches!(argument(0)?, Term::BlankNode(_)))),
                "isIRI" | "isURI" => Some(boolean(matches!(argument(0)?, Term::NamedNode(_)))),
                "isLiteral" => Some(boolean(matches!(argument(0)?, Term::Literal(_)))),
                "lang" => match argument(0)? {
                    Term::Literal(literal) => Some(
                        Literal::new_simple_literal(
                            literal.language().unwrap_or_default().to_owned(),
                        )
                        .into(),
                    ),
                    _ => None,
                },
                "datatype" => match argument(0)? {
                    Term::Literal(literal) => Some(literal.datatype().into()),
                    _ => None,
                },
                "str" => Some(
                    Literal::new_simple_literal(match argument(0)? {
                        Term::Literal(literal) => literal.value().to_owned(),
                        Term::NamedNode(node) => node.as_str().to_owned(),
                        other => other.to_string(),
                    })
                    .into(),
                ),
                "equals" | "sameTerm" => Some(boolean(equal(&argument(0), &argument(1)))),
                "not-equals" => Some(boolean(!equal(&argument(0), &argument(1)))),
                "bound" => Some(boolean(argument(0).is_some())),
                _ => None,
            }
        }
    }
}

/// SPARQL `=` over the term kinds the RDFa suite compares: plain equality, with a simple literal
/// and an `xsd:string` literal treated as equal.
fn equal(left: &Option<Term>, right: &Option<Term>) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => match (left, right) {
            (Term::Literal(a), Term::Literal(b)) => {
                a.value() == b.value()
                    && a.language() == b.language()
                    && a.datatype() == b.datatype()
            }
            _ => left == right,
        },
        _ => false,
    }
}

fn boolean(value: bool) -> Term {
    Literal::new_typed_literal(
        if value { "true" } else { "false" },
        NamedNode::new_unchecked("http://www.w3.org/2001/XMLSchema#boolean"),
    )
    .into()
}

/// The [effective boolean value](https://www.w3.org/TR/sparql11-query/#ebv) of an expression
/// result, with an unbound or erroring expression counting as `false`.
fn effective_boolean(term: &Option<Term>) -> bool {
    let Some(Term::Literal(literal)) = term else {
        return false;
    };
    match literal.datatype().as_str() {
        "http://www.w3.org/2001/XMLSchema#boolean" => literal.value() == "true",
        "http://www.w3.org/2001/XMLSchema#string" => !literal.value().is_empty(),
        _ => literal.language().is_none() && !literal.value().is_empty(),
    }
}

#[test]
fn the_evaluator_answers_a_basic_ask() -> Result<(), Box<dyn std::error::Error>> {
    use oxixml_sparql_syntax::SparqlParser;

    let graph = vec![Triple::new(
        NamedNode::new("http://example.com/s")?,
        NamedNode::new("http://example.com/p")?,
        Literal::new_simple_literal("o"),
    )];
    let yes = SparqlParser::new().parse_query("ASK WHERE { ?s <http://example.com/p> \"o\" }")?;
    let no = SparqlParser::new().parse_query("ASK WHERE { ?s <http://example.com/q> \"o\" }")?;
    let blank = SparqlParser::new()
        .parse_query("ASK WHERE { ?s ?p ?o FILTER (!isBlank(?s) && isLITERAL(?o)) }")?;
    assert_eq!(ask(&yes, &graph), Some(true));
    assert_eq!(ask(&no, &graph), Some(false));
    assert_eq!(ask(&blank, &graph), Some(true));
    Ok(())
}
