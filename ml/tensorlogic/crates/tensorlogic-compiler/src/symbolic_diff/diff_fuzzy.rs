//! Differentiation arms for fuzzy logic, weighted rules, probabilistic choice, and counting.

use tensorlogic_ir::TLExpr;

use super::diff_core::diff_expr;
use super::helpers::zero;
use super::types::{DiffContext, DiffError};

/// Try to differentiate `expr` using fuzzy / weighted / probabilistic / counting rules.
pub(super) fn try_diff_fuzzy(
    expr: &TLExpr,
    ctx: &mut DiffContext<'_>,
) -> Option<Result<TLExpr, DiffError>> {
    match expr {
        // Fuzzy AND (TNorm): product rule analog → TCoNorm
        TLExpr::TNorm { kind, left, right } => Some((|| {
            let dl = diff_expr(left, ctx)?;
            let dr = diff_expr(right, ctx)?;
            let term1 = TLExpr::TNorm {
                kind: *kind,
                left: Box::new(dl),
                right: right.clone(),
            };
            let term2 = TLExpr::TNorm {
                kind: *kind,
                left: left.clone(),
                right: Box::new(dr),
            };
            Ok(TLExpr::TCoNorm {
                kind: tensorlogic_ir::TCoNormKind::Maximum,
                left: Box::new(term1),
                right: Box::new(term2),
            })
        })()),

        // Fuzzy OR (TCoNorm): sum rule analog
        TLExpr::TCoNorm { kind, left, right } => Some((|| {
            let dl = diff_expr(left, ctx)?;
            let dr = diff_expr(right, ctx)?;
            Ok(TLExpr::TCoNorm {
                kind: *kind,
                left: Box::new(dl),
                right: Box::new(dr),
            })
        })()),

        TLExpr::FuzzyNot { kind, expr: inner } => Some((|| {
            let di = diff_expr(inner, ctx)?;
            Ok(TLExpr::FuzzyNot {
                kind: *kind,
                expr: Box::new(di),
            })
        })()),

        TLExpr::FuzzyImplication {
            kind,
            premise,
            conclusion,
        } => Some((|| {
            let expanded = TLExpr::or(TLExpr::negate(*premise.clone()), *conclusion.clone());
            let di = diff_expr(&expanded, ctx)?;
            Ok(TLExpr::FuzzyImplication {
                kind: *kind,
                premise: premise.clone(),
                conclusion: Box::new(di),
            })
        })()),

        // Weighted rule / probabilistic
        TLExpr::WeightedRule { weight, rule } => Some((|| {
            let dr = diff_expr(rule, ctx)?;
            Ok(TLExpr::WeightedRule {
                weight: *weight,
                rule: Box::new(dr),
            })
        })()),

        TLExpr::ProbabilisticChoice { alternatives } => Some((|| {
            let new_alts: Result<Vec<_>, _> = alternatives
                .iter()
                .map(|(p, e)| {
                    let de = diff_expr(e, ctx)?;
                    Ok((*p, de))
                })
                .collect();
            Ok(TLExpr::ProbabilisticChoice {
                alternatives: new_alts?,
            })
        })()),

        // Counting quantifiers
        TLExpr::CountingExists {
            var,
            domain,
            body,
            min_count,
        } => Some((|| {
            if var == &ctx.var {
                Ok(zero())
            } else {
                let dbody = diff_expr(body, ctx)?;
                Ok(TLExpr::CountingExists {
                    var: var.clone(),
                    domain: domain.clone(),
                    body: Box::new(dbody),
                    min_count: *min_count,
                })
            }
        })()),

        TLExpr::CountingForAll {
            var,
            domain,
            body,
            min_count,
        } => Some((|| {
            if var == &ctx.var {
                Ok(zero())
            } else {
                let dbody = diff_expr(body, ctx)?;
                Ok(TLExpr::CountingForAll {
                    var: var.clone(),
                    domain: domain.clone(),
                    body: Box::new(dbody),
                    min_count: *min_count,
                })
            }
        })()),

        TLExpr::ExactCount {
            var,
            domain,
            body,
            count,
        } => Some((|| {
            if var == &ctx.var {
                Ok(zero())
            } else {
                let dbody = diff_expr(body, ctx)?;
                Ok(TLExpr::ExactCount {
                    var: var.clone(),
                    domain: domain.clone(),
                    body: Box::new(dbody),
                    count: *count,
                })
            }
        })()),

        TLExpr::Majority { var, domain, body } => Some((|| {
            if var == &ctx.var {
                Ok(zero())
            } else {
                let dbody = diff_expr(body, ctx)?;
                Ok(TLExpr::Majority {
                    var: var.clone(),
                    domain: domain.clone(),
                    body: Box::new(dbody),
                })
            }
        })()),

        _ => None,
    }
}
