//! A minimal SPARQL `ASK` evaluator, private to [`super`].
//!
//! Every case of the [W3C RDFa Test Suite](https://rdfa.info/test-suite/) states its expectation
//! as a SPARQL `ASK` query. [`oxixml_sparql_syntax`] already parses one into an algebra tree, but
//! this crate has no SPARQL *execution* engine to depend on (its own [`crate::sparql`] runner only
//! exercises the syntax-to-algebra half); this module is a small, self-contained evaluator for
//! exactly the fragment of the algebra the RDFa suite's queries use — basic graph patterns (with a
//! query blank node acting as a non-distinguished variable), `OPTIONAL`, `UNION`, `MINUS`,
//! `FILTER`, and the handful of builtins (`isBLANK`, `isIRI`, `isLITERAL`, `LANG`, `STR`, `BOUND`,
//! `=`, `!`, `&&`, `||`) those queries reach for. Anything else evaluates to no solutions, which
//! fails the case loudly rather than passing it by accident.
//!
//! `oxixml-rdfa`'s own conformance harness (`tests/support/sparql_ask.rs`) needs the identical
//! evaluator for the identical reason; the two are intentionally the same shape, so a fix to one
//! is easy to port to the other. They cannot share code directly — the other one lives in a
//! `tests/` directory, which is not something a sibling crate can depend on.

use std::collections::HashMap;

use oxixml_model::{Literal, NamedNode, Term, Triple};
use oxixml_sparql_syntax::Query;
use oxixml_sparql_syntax::algebra::{Expression, GraphPattern};
use oxixml_sparql_syntax::term::{NamedNodePattern, TermPattern, TriplePattern};

/// One solution mapping: variable name to bound term.
type Solution = HashMap<String, Term>;

/// Answers a SPARQL `ASK` against `graph`.
///
/// `None` when `query` is not an `ASK`, or its pattern uses a construct this evaluator does not
/// implement (see the module documentation for exactly what is covered).
#[must_use]
pub(super) fn ask(query: &Query, graph: &[Triple]) -> Option<bool> {
    let Query::Ask(ask) = query else {
        return None;
    };
    let solutions = evaluate(&ask.pattern, graph)?;
    Some(!solutions.is_empty())
}

fn evaluate(pattern: &GraphPattern, graph: &[Triple]) -> Option<Vec<Solution>> {
    match pattern {
        GraphPattern::Bgp { patterns } => Some(evaluate_bgp(patterns, graph)),
        GraphPattern::Join { left, right } => {
            Some(join(&evaluate(left, graph)?, &evaluate(right, graph)?))
        }
        GraphPattern::LeftJoin {
            left,
            right,
            expression,
        } => Some(evaluate_left_join(left, right, expression.as_ref(), graph)?),
        GraphPattern::Union { left, right } => {
            let mut solutions = evaluate(left, graph)?;
            solutions.extend(evaluate(right, graph)?);
            Some(solutions)
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
        // A pattern the suite's queries never actually need distinguished: an `ASK` only cares
        // whether at least one solution exists, so its outer shaping (which variables are kept,
        // sort order, `LIMIT`/`OFFSET`, `DISTINCT`/`REDUCED`) never changes the answer.
        GraphPattern::Project { inner, .. }
        | GraphPattern::Distinct { inner }
        | GraphPattern::Reduced { inner }
        | GraphPattern::OrderBy { inner, .. }
        | GraphPattern::Slice { inner, .. } => evaluate(inner, graph),
        _ => None,
    }
}

fn evaluate_bgp(patterns: &[TriplePattern], graph: &[Triple]) -> Vec<Solution> {
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
    solutions
}

fn evaluate_left_join(
    left: &GraphPattern,
    right: &GraphPattern,
    expression: Option<&Expression>,
    graph: &[Triple],
) -> Option<Vec<Solution>> {
    let left = evaluate(left, graph)?;
    let right = evaluate(right, graph)?;
    let mut out = Vec::new();
    for solution in &left {
        let mut matched = false;
        for candidate in &right {
            let Some(merged) = merge(solution, candidate) else {
                continue;
            };
            let keep = expression.is_none_or(|expression| {
                effective_boolean(&eval_expression(expression, &merged, graph))
            });
            if keep {
                out.push(merged);
                matched = true;
            }
        }
        if !matched {
            out.push(solution.clone());
        }
    }
    Some(out)
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

/// Whether `b` agrees with `a` on every variable they share — the `MINUS`/`EXISTS` compatibility
/// test, which (unlike [`merge`]) never produces a combined solution.
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

fn match_triple(pattern: &TriplePattern, triple: &Triple, solution: &Solution) -> Option<Solution> {
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

/// The name a query blank node is bound under. A blank node in a SPARQL graph pattern behaves as
/// a variable nothing can project — for an `ASK`, that is indistinguishable from an ordinary one.
/// The leading space keeps it out of the way of any real (identifier-shaped) variable name.
fn blank_variable(label: &str) -> String {
    format!(" bnode:{label}")
}

fn bind_term(pattern: &TermPattern, value: &Term, solution: &mut Solution) -> Option<()> {
    match pattern {
        TermPattern::Variable(variable) => bind(variable.as_str(), value, solution),
        TermPattern::BlankNode(node) => bind(&blank_variable(node.as_str()), value, solution),
        TermPattern::NamedNode(node) => (value == &Term::from(node.clone())).then_some(()),
        TermPattern::Literal(literal) => (value == &Term::from(literal.clone())).then_some(()),
        #[allow(
            unreachable_patterns,
            reason = "TermPattern may grow rdf-12 triple-term variants"
        )]
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
            eval_function_call(function.as_str(), arguments, solution, graph)
        }
    }
}

fn eval_function_call(
    iri: &str,
    arguments: &[Expression],
    solution: &Solution,
    graph: &[Triple],
) -> Option<Term> {
    let name = iri.rsplit('#').next().unwrap_or(iri);
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
                Literal::new_simple_literal(literal.language().unwrap_or_default().to_owned())
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

/// SPARQL `=` over the term kinds the RDFa suite's queries compare: a simple literal and an
/// `xsd:string` literal with the same lexical form are equal, otherwise terms compare structurally.
fn equal(left: &Option<Term>, right: &Option<Term>) -> bool {
    match (left, right) {
        (Some(Term::Literal(a)), Some(Term::Literal(b))) => {
            a.value() == b.value() && a.language() == b.language() && a.datatype() == b.datatype()
        }
        (Some(left), Some(right)) => left == right,
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
/// result; an unbound or erroring expression counts as `false`, matching `FILTER`'s own rule.
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

#[cfg(test)]
mod tests {
    use super::*;
    use oxixml_sparql_syntax::SparqlParser;

    fn parse(query: &str) -> Query {
        SparqlParser::new()
            .parse_query(query)
            .expect("valid ASK query")
    }

    #[test]
    fn answers_a_basic_triple_pattern() {
        let graph = vec![Triple::new(
            NamedNode::new_unchecked("http://example.com/s"),
            NamedNode::new_unchecked("http://example.com/p"),
            Literal::new_simple_literal("o"),
        )];
        let yes = parse("ASK WHERE { ?s <http://example.com/p> \"o\" }");
        let no = parse("ASK WHERE { ?s <http://example.com/q> \"o\" }");
        assert_eq!(ask(&yes, &graph), Some(true));
        assert_eq!(ask(&no, &graph), Some(false));
    }

    #[test]
    fn filter_and_builtins_are_evaluated() {
        let graph = vec![Triple::new(
            NamedNode::new_unchecked("http://example.com/s"),
            NamedNode::new_unchecked("http://example.com/p"),
            Literal::new_simple_literal("o"),
        )];
        let query = parse("ASK WHERE { ?s ?p ?o FILTER (!isBlank(?s) && isLITERAL(?o)) }");
        assert_eq!(ask(&query, &graph), Some(true));
    }

    #[test]
    fn optional_keeps_the_outer_solution_when_the_inner_pattern_does_not_match() {
        let graph = vec![Triple::new(
            NamedNode::new_unchecked("http://example.com/s"),
            NamedNode::new_unchecked("http://example.com/p"),
            Literal::new_simple_literal("o"),
        )];
        let query = parse("ASK WHERE { ?s ?p ?o OPTIONAL { ?s <http://example.com/missing> ?x } }");
        assert_eq!(ask(&query, &graph), Some(true));
    }

    #[test]
    fn a_non_ask_query_is_not_answered() {
        let select = SparqlParser::new()
            .parse_query("SELECT * WHERE { ?s ?p ?o }")
            .expect("valid query");
        assert_eq!(ask(&select, &[]), None);
    }
}
