//! Internal helpers used across the partial evaluator submodules: extracting
//! constants and computing the set of still-free logical variables after PE.

use std::collections::HashSet;

use tensorlogic_ir::TLExpr;

/// Check if a `TLExpr` is a concrete constant and return its value.
#[inline]
pub(super) fn as_constant(expr: &TLExpr) -> Option<f64> {
    match expr {
        TLExpr::Constant(v) => Some(*v),
        _ => None,
    }
}

/// Collect all free zero-arity predicate names (logical variables) in an expression.
pub(super) fn collect_free_pred_vars(
    expr: &TLExpr,
    bound: &HashSet<String>,
    out: &mut HashSet<String>,
) {
    match expr {
        TLExpr::Pred { name, args } if args.is_empty() => {
            if !bound.contains(name.as_str()) {
                out.insert(name.clone());
            }
        }
        TLExpr::Pred { .. } => {} // non-zero arity: proper predicate, not a variable
        TLExpr::Constant(_)
        | TLExpr::EmptySet
        | TLExpr::AllDifferent { .. }
        | TLExpr::Nominal { .. }
        | TLExpr::Abducible { .. } => {}

        TLExpr::Add(a, b)
        | TLExpr::Sub(a, b)
        | TLExpr::Mul(a, b)
        | TLExpr::Div(a, b)
        | TLExpr::Pow(a, b)
        | TLExpr::Mod(a, b)
        | TLExpr::Min(a, b)
        | TLExpr::Max(a, b)
        | TLExpr::And(a, b)
        | TLExpr::Or(a, b)
        | TLExpr::Imply(a, b)
        | TLExpr::Eq(a, b)
        | TLExpr::Lt(a, b)
        | TLExpr::Gt(a, b)
        | TLExpr::Lte(a, b)
        | TLExpr::Gte(a, b)
        | TLExpr::Until {
            before: a,
            after: b,
        }
        | TLExpr::WeakUntil {
            before: a,
            after: b,
        }
        | TLExpr::Release {
            released: a,
            releaser: b,
        }
        | TLExpr::StrongRelease {
            released: a,
            releaser: b,
        }
        | TLExpr::SetUnion { left: a, right: b }
        | TLExpr::SetIntersection { left: a, right: b }
        | TLExpr::SetDifference { left: a, right: b } => {
            collect_free_pred_vars(a, bound, out);
            collect_free_pred_vars(b, bound, out);
        }

        TLExpr::Not(e)
        | TLExpr::Score(e)
        | TLExpr::Box(e)
        | TLExpr::Diamond(e)
        | TLExpr::Next(e)
        | TLExpr::Eventually(e)
        | TLExpr::Always(e)
        | TLExpr::Abs(e)
        | TLExpr::Floor(e)
        | TLExpr::Ceil(e)
        | TLExpr::Round(e)
        | TLExpr::Sqrt(e)
        | TLExpr::Exp(e)
        | TLExpr::Log(e)
        | TLExpr::Sin(e)
        | TLExpr::Cos(e)
        | TLExpr::Tan(e)
        | TLExpr::SetCardinality { set: e }
        | TLExpr::Explain { formula: e }
        | TLExpr::Somewhere { formula: e }
        | TLExpr::Everywhere { formula: e }
        | TLExpr::WeightedRule { rule: e, .. }
        | TLExpr::LeastFixpoint { body: e, .. }
        | TLExpr::GreatestFixpoint { body: e, .. } => {
            collect_free_pred_vars(e, bound, out);
        }

        TLExpr::Exists { var, body, .. }
        | TLExpr::ForAll { var, body, .. }
        | TLExpr::SoftExists { var, body, .. }
        | TLExpr::SoftForAll { var, body, .. }
        | TLExpr::Aggregate { var, body, .. }
        | TLExpr::CountingExists { var, body, .. }
        | TLExpr::CountingForAll { var, body, .. }
        | TLExpr::ExactCount { var, body, .. }
        | TLExpr::Majority { var, body, .. }
        | TLExpr::SetComprehension {
            var,
            condition: body,
            ..
        } => {
            let mut new_bound = bound.clone();
            new_bound.insert(var.clone());
            collect_free_pred_vars(body, &new_bound, out);
        }

        TLExpr::Let { var, value, body } => {
            collect_free_pred_vars(value, bound, out);
            let mut new_bound = bound.clone();
            new_bound.insert(var.clone());
            collect_free_pred_vars(body, &new_bound, out);
        }

        TLExpr::Lambda { var, body, .. } => {
            let mut new_bound = bound.clone();
            new_bound.insert(var.clone());
            collect_free_pred_vars(body, &new_bound, out);
        }

        TLExpr::Apply { function, argument } => {
            collect_free_pred_vars(function, bound, out);
            collect_free_pred_vars(argument, bound, out);
        }

        TLExpr::SetMembership { element, set } => {
            collect_free_pred_vars(element, bound, out);
            collect_free_pred_vars(set, bound, out);
        }

        TLExpr::IfThenElse {
            condition,
            then_branch,
            else_branch,
        } => {
            collect_free_pred_vars(condition, bound, out);
            collect_free_pred_vars(then_branch, bound, out);
            collect_free_pred_vars(else_branch, bound, out);
        }

        TLExpr::TNorm { left, right, .. } | TLExpr::TCoNorm { left, right, .. } => {
            collect_free_pred_vars(left, bound, out);
            collect_free_pred_vars(right, bound, out);
        }

        TLExpr::FuzzyNot { expr: e, .. } => {
            collect_free_pred_vars(e, bound, out);
        }

        TLExpr::FuzzyImplication {
            premise,
            conclusion,
            ..
        } => {
            collect_free_pred_vars(premise, bound, out);
            collect_free_pred_vars(conclusion, bound, out);
        }

        TLExpr::ProbabilisticChoice { alternatives } => {
            for (_, alt_expr) in alternatives {
                collect_free_pred_vars(alt_expr, bound, out);
            }
        }

        TLExpr::At { formula, .. } => {
            collect_free_pred_vars(formula, bound, out);
        }

        TLExpr::GlobalCardinality { values, .. } => {
            for v in values {
                collect_free_pred_vars(v, bound, out);
            }
        }

        TLExpr::SymbolLiteral(_) => {}

        TLExpr::Match { scrutinee, arms } => {
            collect_free_pred_vars(scrutinee, bound, out);
            for (_, body) in arms {
                collect_free_pred_vars(body, bound, out);
            }
        }
    }
}
