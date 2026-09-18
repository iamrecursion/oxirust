//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use tensorlogic_ir::TLExpr;

use super::functions::{discriminant_name, expr_to_filter, expr_to_sparql_expr, term_to_sparql};
use super::types::{
    GraphPattern, SparqlFilter, SparqlGenConfig, SparqlGenError, SparqlQuery, SparqlTerm,
    TriplePattern,
};

/// Recursively lower a [`TLExpr`] into a list of [`GraphPattern`]s.
///
/// `free_vars` accumulates variable names that were introduced (for SELECT projection).
fn expr_to_patterns(
    expr: &TLExpr,
    config: &SparqlGenConfig,
    depth: usize,
    collected: &mut Vec<GraphPattern>,
    free_vars: &mut Vec<String>,
) -> Result<(), SparqlGenError> {
    if depth > config.max_depth {
        return Err(SparqlGenError::UnsupportedExpr(
            "Maximum recursion depth exceeded".to_owned(),
        ));
    }
    match expr {
        TLExpr::Pred { name, args } => {
            let predicate_iri = SparqlTerm::Iri(format!("{}{}", config.base_prefix, name));
            match args.len() {
                0 => {
                    let s_var = format!("{}subj_{}", config.variable_prefix, name);
                    free_vars.push(s_var.clone());
                    collected.push(GraphPattern::Triple(TriplePattern::new(
                        SparqlTerm::Variable(s_var),
                        SparqlTerm::Iri("rdf:type".to_owned()),
                        predicate_iri,
                    )));
                }
                1 => {
                    let subj = term_to_sparql(&args[0], config);
                    if let SparqlTerm::Variable(ref v) = subj {
                        if !free_vars.contains(v) {
                            free_vars.push(v.clone());
                        }
                    }
                    collected.push(GraphPattern::Triple(TriplePattern::new(
                        subj,
                        SparqlTerm::Iri("rdf:type".to_owned()),
                        predicate_iri,
                    )));
                }
                2 => {
                    let subj = term_to_sparql(&args[0], config);
                    let obj = term_to_sparql(&args[1], config);
                    if let SparqlTerm::Variable(ref v) = subj {
                        if !free_vars.contains(v) {
                            free_vars.push(v.clone());
                        }
                    }
                    if let SparqlTerm::Variable(ref v) = obj {
                        if !free_vars.contains(v) {
                            free_vars.push(v.clone());
                        }
                    }
                    collected.push(GraphPattern::Triple(TriplePattern::new(
                        subj,
                        predicate_iri,
                        obj,
                    )));
                }
                _ => {
                    for window in args.windows(2) {
                        let s = term_to_sparql(&window[0], config);
                        let o = term_to_sparql(&window[1], config);
                        if let SparqlTerm::Variable(ref v) = s {
                            if !free_vars.contains(v) {
                                free_vars.push(v.clone());
                            }
                        }
                        if let SparqlTerm::Variable(ref v) = o {
                            if !free_vars.contains(v) {
                                free_vars.push(v.clone());
                            }
                        }
                        collected.push(GraphPattern::Triple(TriplePattern::new(
                            s,
                            predicate_iri.clone(),
                            o,
                        )));
                    }
                }
            }
        }
        TLExpr::And(left, right) => {
            expr_to_patterns(left, config, depth + 1, collected, free_vars)?;
            expr_to_patterns(right, config, depth + 1, collected, free_vars)?;
        }
        TLExpr::Or(left, right) => {
            let mut left_patterns = Vec::new();
            let mut right_patterns = Vec::new();
            expr_to_patterns(left, config, depth + 1, &mut left_patterns, free_vars)?;
            expr_to_patterns(right, config, depth + 1, &mut right_patterns, free_vars)?;
            collected.push(GraphPattern::Union(left_patterns, right_patterns));
        }
        TLExpr::Not(inner) => {
            let mut inner_patterns = Vec::new();
            let mut inner_vars = Vec::new();
            expr_to_patterns(
                inner,
                config,
                depth + 1,
                &mut inner_patterns,
                &mut inner_vars,
            )?;
            collected.push(GraphPattern::Filter(SparqlFilter::NotExists(
                inner_patterns,
            )));
        }
        TLExpr::Exists {
            var,
            domain: _,
            body,
        } => {
            let var_name = format!("{}{}", config.variable_prefix, var);
            if !free_vars.contains(&var_name) {
                free_vars.push(var_name.clone());
            }
            if config.use_optional_for_exists {
                let mut body_patterns = Vec::new();
                expr_to_patterns(body, config, depth + 1, &mut body_patterns, free_vars)?;
                collected.push(GraphPattern::Optional(body_patterns));
            } else {
                expr_to_patterns(body, config, depth + 1, collected, free_vars)?;
            }
        }
        TLExpr::ForAll {
            var,
            domain: _,
            body,
        } => {
            let var_name = format!("{}{}", config.variable_prefix, var);
            if !free_vars.contains(&var_name) {
                free_vars.push(var_name.clone());
            }
            let not_body = TLExpr::Not(body.clone());
            let mut violation_patterns = Vec::new();
            let mut vvars = vec![var_name];
            expr_to_patterns(
                &not_body,
                config,
                depth + 1,
                &mut violation_patterns,
                &mut vvars,
            )?;
            collected.push(GraphPattern::Filter(SparqlFilter::NotExists(
                violation_patterns,
            )));
        }
        TLExpr::Imply(premise, conclusion) => {
            let desugared = TLExpr::Or(Box::new(TLExpr::Not(premise.clone())), conclusion.clone());
            expr_to_patterns(&desugared, config, depth + 1, collected, free_vars)?;
        }
        TLExpr::Constant(v) => {
            if *v == 0.0 {
                collected.push(GraphPattern::Filter(SparqlFilter::BoolValue(false)));
            }
        }
        TLExpr::Eq(_, _)
        | TLExpr::Lt(_, _)
        | TLExpr::Gt(_, _)
        | TLExpr::Lte(_, _)
        | TLExpr::Gte(_, _) => {
            let filter = expr_to_filter(expr, config)?;
            collected.push(GraphPattern::Filter(filter));
        }
        TLExpr::Let { var, value, body } => {
            let sparql_expr = expr_to_sparql_expr(value, config)?;
            let var_name = format!("{}{}", config.variable_prefix, var);
            if !free_vars.contains(&var_name) {
                free_vars.push(var_name.clone());
            }
            collected.push(GraphPattern::Bind(sparql_expr, var_name));
            expr_to_patterns(body, config, depth + 1, collected, free_vars)?;
        }
        TLExpr::Score(inner) => {
            expr_to_patterns(inner, config, depth + 1, collected, free_vars)?;
        }
        TLExpr::TNorm { left, right, .. } => {
            expr_to_patterns(left, config, depth + 1, collected, free_vars)?;
            expr_to_patterns(right, config, depth + 1, collected, free_vars)?;
        }
        TLExpr::TCoNorm { left, right, .. } => {
            let mut left_patterns = Vec::new();
            let mut right_patterns = Vec::new();
            expr_to_patterns(left, config, depth + 1, &mut left_patterns, free_vars)?;
            expr_to_patterns(right, config, depth + 1, &mut right_patterns, free_vars)?;
            collected.push(GraphPattern::Union(left_patterns, right_patterns));
        }
        TLExpr::FuzzyNot { expr: inner, .. } => {
            let mut inner_patterns = Vec::new();
            let mut inner_vars = Vec::new();
            expr_to_patterns(
                inner,
                config,
                depth + 1,
                &mut inner_patterns,
                &mut inner_vars,
            )?;
            collected.push(GraphPattern::Filter(SparqlFilter::NotExists(
                inner_patterns,
            )));
        }
        TLExpr::FuzzyImplication {
            premise,
            conclusion,
            ..
        } => {
            let desugared = TLExpr::Or(Box::new(TLExpr::Not(premise.clone())), conclusion.clone());
            expr_to_patterns(&desugared, config, depth + 1, collected, free_vars)?;
        }
        TLExpr::WeightedRule { rule, .. } => {
            expr_to_patterns(rule, config, depth + 1, collected, free_vars)?;
        }
        TLExpr::SoftExists {
            var, domain, body, ..
        } => {
            let ordinary = TLExpr::Exists {
                var: var.clone(),
                domain: domain.clone(),
                body: body.clone(),
            };
            expr_to_patterns(&ordinary, config, depth + 1, collected, free_vars)?;
        }
        TLExpr::SoftForAll {
            var, domain, body, ..
        } => {
            let ordinary = TLExpr::ForAll {
                var: var.clone(),
                domain: domain.clone(),
                body: body.clone(),
            };
            expr_to_patterns(&ordinary, config, depth + 1, collected, free_vars)?;
        }
        other => {
            return Err(SparqlGenError::UnsupportedExpr(format!(
                "No SPARQL translation defined for {}",
                discriminant_name(other)
            )));
        }
    }
    Ok(())
}
/// Generate a SPARQL SELECT query from a [`TLExpr`] logic expression.
///
/// Returns a [`SparqlQuery`] that can be rendered to a SPARQL string via
/// [`SparqlQuery::to_sparql`].
///
/// # Errors
///
/// Returns [`SparqlGenError::UnsupportedExpr`] for expression variants without
/// a defined SPARQL translation.
pub fn expr_to_sparql(
    expr: &TLExpr,
    config: &SparqlGenConfig,
) -> Result<SparqlQuery, SparqlGenError> {
    let mut patterns = Vec::new();
    let mut free_vars = Vec::new();
    expr_to_patterns(expr, config, 0, &mut patterns, &mut free_vars)?;
    let mut query = SparqlQuery::new()
        .with_prefix("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#")
        .with_prefix("rdfs", "http://www.w3.org/2000/01/rdf-schema#")
        .with_prefix("tl", &config.base_prefix);
    let mut seen = std::collections::HashSet::new();
    for v in free_vars {
        if seen.insert(v.clone()) {
            query.select_vars.push(v);
        }
    }
    for p in patterns {
        query.patterns.push(p);
    }
    Ok(query)
}
