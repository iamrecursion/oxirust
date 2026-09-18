//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use tensorlogic_ir::{TLExpr, Term};

use super::types::{
    GraphPattern, SparqlExpr, SparqlFilter, SparqlGenConfig, SparqlGenError, SparqlTerm,
};

/// Render a single [`GraphPattern`] to a SPARQL string fragment.
///
/// `indent` is the base indentation unit; `depth` is the current nesting level.
pub fn render_pattern(pattern: &GraphPattern, indent: &str, depth: usize) -> String {
    let pad: String = indent.repeat(depth);
    let pad_inner: String = indent.repeat(depth + 1);
    match pattern {
        GraphPattern::Triple(tp) => format!("{}{} .\n", pad, tp.as_sparql_string()),
        GraphPattern::Optional(inner) => {
            let mut s = format!("{pad}OPTIONAL {{\n");
            for p in inner {
                s.push_str(&render_pattern(p, indent, depth + 1));
            }
            s.push_str(&format!("{pad}}}\n"));
            s
        }
        GraphPattern::Union(left, right) => {
            let mut s = format!("{pad}{{\n");
            for p in left {
                s.push_str(&render_pattern(p, indent, depth + 1));
            }
            s.push_str(&format!("{pad}}} UNION {{\n"));
            for p in right {
                s.push_str(&render_pattern(p, indent, depth + 1));
            }
            s.push_str(&format!("{pad}}}\n"));
            s
        }
        GraphPattern::Filter(filter) => {
            format!("{pad}FILTER ( {} )\n", render_filter(filter))
        }
        GraphPattern::Bind(expr, var) => {
            format!("{pad}BIND ( {} AS ?{} )\n", expr.as_sparql_string(), var)
        }
        GraphPattern::Values(vars, rows) => {
            let row_strs: Vec<String> = rows
                .iter()
                .map(|row| {
                    let terms: Vec<String> = row.iter().map(|t| t.as_sparql_string()).collect();
                    format!("({})", terms.join(" "))
                })
                .collect();
            if vars.len() == 1 {
                format!("{pad}VALUES ?{} {{ {} }}\n", vars[0], row_strs.join(" "))
            } else {
                let var_list: Vec<String> = vars.iter().map(|v| format!("?{v}")).collect();
                format!(
                    "{pad}VALUES ({}) {{ {} }}\n",
                    var_list.join(" "),
                    row_strs.join(" ")
                )
            }
        }
        GraphPattern::Group(inner) => {
            let mut s = format!("{pad}{{\n");
            for p in inner {
                s.push_str(&render_pattern(p, indent, depth + 1));
            }
            s.push_str(&format!("{pad}}}\n"));
            let _ = pad_inner;
            s
        }
    }
}
/// Render a [`SparqlFilter`] to its SPARQL filter expression string.
pub fn render_filter(filter: &SparqlFilter) -> String {
    match filter {
        SparqlFilter::Equals(a, b) => {
            format!("{} = {}", a.as_sparql_string(), b.as_sparql_string())
        }
        SparqlFilter::NotEquals(a, b) => {
            format!("{} != {}", a.as_sparql_string(), b.as_sparql_string())
        }
        SparqlFilter::LessThan(a, b) => {
            format!("{} < {}", a.as_sparql_string(), b.as_sparql_string())
        }
        SparqlFilter::GreaterThan(a, b) => {
            format!("{} > {}", a.as_sparql_string(), b.as_sparql_string())
        }
        SparqlFilter::LessOrEqual(a, b) => {
            format!("{} <= {}", a.as_sparql_string(), b.as_sparql_string())
        }
        SparqlFilter::GreaterOrEqual(a, b) => {
            format!("{} >= {}", a.as_sparql_string(), b.as_sparql_string())
        }
        SparqlFilter::And(l, r) => {
            format!("({} && {})", render_filter(l), render_filter(r))
        }
        SparqlFilter::Or(l, r) => {
            format!("({} || {})", render_filter(l), render_filter(r))
        }
        SparqlFilter::Not(inner) => format!("!({})", render_filter(inner)),
        SparqlFilter::Bound(var) => format!("BOUND(?{var})"),
        SparqlFilter::IsIri(term) => format!("isIRI({})", term.as_sparql_string()),
        SparqlFilter::Regex(term, pattern) => {
            format!("REGEX({}, \"{}\")", term.as_sparql_string(), pattern)
        }
        SparqlFilter::BoolValue(b) => {
            if *b {
                "true".to_owned()
            } else {
                "false".to_owned()
            }
        }
        SparqlFilter::NotExists(patterns) => {
            let mut s = "NOT EXISTS {\n".to_owned();
            for p in patterns {
                s.push_str(&render_pattern(p, "  ", 1));
            }
            s.push('}');
            s
        }
    }
}
/// Convert a TensorLogic [`Term`] to a [`SparqlTerm`].
pub(super) fn term_to_sparql(term: &Term, config: &SparqlGenConfig) -> SparqlTerm {
    match term.untyped() {
        Term::Var(name) => SparqlTerm::Variable(format!("{}{}", config.variable_prefix, name)),
        Term::Const(name) => {
            if name.starts_with("http://")
                || name.starts_with("https://")
                || name.starts_with("urn:")
            {
                SparqlTerm::Iri(name.clone())
            } else {
                SparqlTerm::Iri(format!("{}{}", shorten_prefix(&config.base_prefix), name))
            }
        }
        Term::Typed { value, .. } => term_to_sparql(value, config),
    }
}
/// Return a short prefix label from a full IRI (last fragment).
fn shorten_prefix(base: &str) -> String {
    base.to_owned()
}
/// Attempt to lower a [`TLExpr`] to a [`SparqlExpr`] for use in BIND.
pub(super) fn expr_to_sparql_expr(
    expr: &TLExpr,
    config: &SparqlGenConfig,
) -> Result<SparqlExpr, SparqlGenError> {
    match expr {
        TLExpr::Constant(v) => Ok(SparqlExpr::Term(SparqlTerm::NumericLiteral(*v))),
        TLExpr::Pred { name, args } if args.is_empty() => Ok(SparqlExpr::Term(SparqlTerm::Iri(
            format!("{}{}", config.base_prefix, name),
        ))),
        TLExpr::Pred { name, args } if args.len() == 1 => {
            Ok(SparqlExpr::Term(term_to_sparql(&args[0], config)))
        }
        TLExpr::Add(l, r) => Ok(SparqlExpr::Add(
            Box::new(expr_to_sparql_expr(l, config)?),
            Box::new(expr_to_sparql_expr(r, config)?),
        )),
        TLExpr::Sub(l, r) => Ok(SparqlExpr::Sub(
            Box::new(expr_to_sparql_expr(l, config)?),
            Box::new(expr_to_sparql_expr(r, config)?),
        )),
        TLExpr::Mul(l, r) => Ok(SparqlExpr::Mul(
            Box::new(expr_to_sparql_expr(l, config)?),
            Box::new(expr_to_sparql_expr(r, config)?),
        )),
        TLExpr::Div(l, r) => Ok(SparqlExpr::Div(
            Box::new(expr_to_sparql_expr(l, config)?),
            Box::new(expr_to_sparql_expr(r, config)?),
        )),
        other => Err(SparqlGenError::UnsupportedExpr(format!(
            "Cannot lower {:?} to a SPARQL expression",
            discriminant_name(other)
        ))),
    }
}
/// Return the variant name of a TLExpr for error messages.
pub(super) fn discriminant_name(expr: &TLExpr) -> &'static str {
    match expr {
        TLExpr::Pred { .. } => "Pred",
        TLExpr::And(..) => "And",
        TLExpr::Or(..) => "Or",
        TLExpr::Not(..) => "Not",
        TLExpr::Exists { .. } => "Exists",
        TLExpr::ForAll { .. } => "ForAll",
        TLExpr::Imply(..) => "Imply",
        TLExpr::Score(..) => "Score",
        TLExpr::Add(..) => "Add",
        TLExpr::Sub(..) => "Sub",
        TLExpr::Mul(..) => "Mul",
        TLExpr::Div(..) => "Div",
        TLExpr::Pow(..) => "Pow",
        TLExpr::Mod(..) => "Mod",
        TLExpr::Min(..) => "Min",
        TLExpr::Max(..) => "Max",
        TLExpr::Abs(..) => "Abs",
        TLExpr::Floor(..) => "Floor",
        TLExpr::Ceil(..) => "Ceil",
        TLExpr::Round(..) => "Round",
        TLExpr::Sqrt(..) => "Sqrt",
        TLExpr::Exp(..) => "Exp",
        TLExpr::Log(..) => "Log",
        TLExpr::Sin(..) => "Sin",
        TLExpr::Cos(..) => "Cos",
        TLExpr::Tan(..) => "Tan",
        TLExpr::Eq(..) => "Eq",
        TLExpr::Lt(..) => "Lt",
        TLExpr::Gt(..) => "Gt",
        TLExpr::Lte(..) => "Lte",
        TLExpr::Gte(..) => "Gte",
        TLExpr::IfThenElse { .. } => "IfThenElse",
        TLExpr::Constant(..) => "Constant",
        TLExpr::Aggregate { .. } => "Aggregate",
        TLExpr::Let { .. } => "Let",
        TLExpr::Box(..) => "Box",
        TLExpr::Diamond(..) => "Diamond",
        TLExpr::Next(..) => "Next",
        TLExpr::Eventually(..) => "Eventually",
        TLExpr::Always(..) => "Always",
        TLExpr::Until { .. } => "Until",
        TLExpr::TNorm { .. } => "TNorm",
        TLExpr::TCoNorm { .. } => "TCoNorm",
        TLExpr::FuzzyNot { .. } => "FuzzyNot",
        TLExpr::FuzzyImplication { .. } => "FuzzyImplication",
        TLExpr::SoftExists { .. } => "SoftExists",
        TLExpr::SoftForAll { .. } => "SoftForAll",
        TLExpr::WeightedRule { .. } => "WeightedRule",
        TLExpr::ProbabilisticChoice { .. } => "ProbabilisticChoice",
        TLExpr::Release { .. } => "Release",
        TLExpr::WeakUntil { .. } => "WeakUntil",
        TLExpr::StrongRelease { .. } => "StrongRelease",
        TLExpr::Lambda { .. } => "Lambda",
        TLExpr::Apply { .. } => "Apply",
        TLExpr::SetMembership { .. } => "SetMembership",
        TLExpr::SetUnion { .. } => "SetUnion",
        TLExpr::SetIntersection { .. } => "SetIntersection",
        TLExpr::SetDifference { .. } => "SetDifference",
        TLExpr::SetCardinality { .. } => "SetCardinality",
        TLExpr::EmptySet => "EmptySet",
        TLExpr::SetComprehension { .. } => "SetComprehension",
        TLExpr::CountingExists { .. } => "CountingExists",
        TLExpr::CountingForAll { .. } => "CountingForAll",
        TLExpr::ExactCount { .. } => "ExactCount",
        TLExpr::Majority { .. } => "Majority",
        TLExpr::LeastFixpoint { .. } => "LeastFixpoint",
        TLExpr::GreatestFixpoint { .. } => "GreatestFixpoint",
        TLExpr::Nominal { .. } => "Nominal",
        TLExpr::At { .. } => "At",
        TLExpr::Somewhere { .. } => "Somewhere",
        TLExpr::Everywhere { .. } => "Everywhere",
        TLExpr::AllDifferent { .. } => "AllDifferent",
        TLExpr::GlobalCardinality { .. } => "GlobalCardinality",
        TLExpr::Abducible { .. } => "Abducible",
        TLExpr::Explain { .. } => "Explain",
        TLExpr::SymbolLiteral(_) => "SymbolLiteral",
        TLExpr::Match { .. } => "Match",
    }
}
/// Try to convert a [`TLExpr`] to a [`SparqlFilter`] (for comparison expressions).
pub(super) fn expr_to_filter(
    expr: &TLExpr,
    config: &SparqlGenConfig,
) -> Result<SparqlFilter, SparqlGenError> {
    match expr {
        TLExpr::Eq(l, r) => Ok(SparqlFilter::Equals(
            tlexpr_to_filter_term(l, config)?,
            tlexpr_to_filter_term(r, config)?,
        )),
        TLExpr::Lt(l, r) => Ok(SparqlFilter::LessThan(
            tlexpr_to_filter_term(l, config)?,
            tlexpr_to_filter_term(r, config)?,
        )),
        TLExpr::Gt(l, r) => Ok(SparqlFilter::GreaterThan(
            tlexpr_to_filter_term(l, config)?,
            tlexpr_to_filter_term(r, config)?,
        )),
        TLExpr::Lte(l, r) => Ok(SparqlFilter::LessOrEqual(
            tlexpr_to_filter_term(l, config)?,
            tlexpr_to_filter_term(r, config)?,
        )),
        TLExpr::Gte(l, r) => Ok(SparqlFilter::GreaterOrEqual(
            tlexpr_to_filter_term(l, config)?,
            tlexpr_to_filter_term(r, config)?,
        )),
        TLExpr::And(l, r) => Ok(SparqlFilter::And(
            Box::new(expr_to_filter(l, config)?),
            Box::new(expr_to_filter(r, config)?),
        )),
        TLExpr::Or(l, r) => Ok(SparqlFilter::Or(
            Box::new(expr_to_filter(l, config)?),
            Box::new(expr_to_filter(r, config)?),
        )),
        TLExpr::Not(inner) => Ok(SparqlFilter::Not(Box::new(expr_to_filter(inner, config)?))),
        TLExpr::Constant(v) => Ok(SparqlFilter::BoolValue(*v != 0.0)),
        other => Err(SparqlGenError::UnsupportedExpr(format!(
            "Cannot convert {} to a SPARQL filter",
            discriminant_name(other)
        ))),
    }
}
/// Extract a [`SparqlTerm`] from a simple [`TLExpr`] for use in filter comparison positions.
fn tlexpr_to_filter_term(
    expr: &TLExpr,
    config: &SparqlGenConfig,
) -> Result<SparqlTerm, SparqlGenError> {
    match expr {
        TLExpr::Constant(v) => Ok(SparqlTerm::NumericLiteral(*v)),
        TLExpr::Pred { name, args } if args.is_empty() => {
            Ok(SparqlTerm::Iri(format!("{}{}", config.base_prefix, name)))
        }
        TLExpr::Pred { name, args } if args.len() == 1 => {
            let _ = name;
            Ok(term_to_sparql(&args[0], config))
        }
        other => Err(SparqlGenError::UnsupportedExpr(format!(
            "Cannot convert {} to a SPARQL filter term",
            discriminant_name(other)
        ))),
    }
}
