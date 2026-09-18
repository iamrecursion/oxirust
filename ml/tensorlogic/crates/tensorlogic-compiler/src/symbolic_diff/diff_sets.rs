//! Differentiation arms for set operations, fixpoints, constraint programming, and abduction.

use tensorlogic_ir::TLExpr;

use super::diff_core::diff_expr;
use super::helpers::zero;
use super::types::{DiffContext, DiffError};

/// Try to differentiate `expr` using set / fixpoint / constraint / abduction rules.
pub(super) fn try_diff_sets(
    expr: &TLExpr,
    ctx: &mut DiffContext<'_>,
) -> Option<Result<TLExpr, DiffError>> {
    match expr {
        // Fixed-point operators
        TLExpr::LeastFixpoint { var, body } => {
            let label = format!("LeastFixpoint({})", var);
            if ctx.config.error_on_unsupported {
                return Some(Err(DiffError::ExprTooComplex(label)));
            }
            ctx.unsupported_nodes.push(label);
            let _ = body;
            Some(Ok(zero()))
        }

        TLExpr::GreatestFixpoint { var, body } => {
            let label = format!("GreatestFixpoint({})", var);
            if ctx.config.error_on_unsupported {
                return Some(Err(DiffError::ExprTooComplex(label)));
            }
            ctx.unsupported_nodes.push(label);
            let _ = body;
            Some(Ok(zero()))
        }

        // Set operations
        TLExpr::SetUnion { left, right } => Some((|| {
            let dl = diff_expr(left, ctx)?;
            let dr = diff_expr(right, ctx)?;
            Ok(TLExpr::SetUnion {
                left: Box::new(dl),
                right: Box::new(dr),
            })
        })()),

        TLExpr::SetIntersection { left, right } => Some((|| {
            let dl = diff_expr(left, ctx)?;
            let dr = diff_expr(right, ctx)?;
            Ok(TLExpr::SetIntersection {
                left: Box::new(dl),
                right: Box::new(dr),
            })
        })()),

        TLExpr::SetDifference { left, right } => Some((|| {
            let dl = diff_expr(left, ctx)?;
            let dr = diff_expr(right, ctx)?;
            Ok(TLExpr::SetDifference {
                left: Box::new(dl),
                right: Box::new(dr),
            })
        })()),

        TLExpr::EmptySet => Some(Ok(zero())),

        TLExpr::SetMembership { element, set } => {
            ctx.unsupported_nodes.push("SetMembership".to_string());
            let _ = (element, set);
            Some(Ok(zero()))
        }

        TLExpr::SetCardinality { set } => {
            ctx.unsupported_nodes.push("SetCardinality".to_string());
            let _ = set;
            Some(Ok(zero()))
        }

        TLExpr::SetComprehension {
            var,
            domain,
            condition,
        } => Some((|| {
            if var == &ctx.var {
                Ok(zero())
            } else {
                let dc = diff_expr(condition, ctx)?;
                Ok(TLExpr::SetComprehension {
                    var: var.clone(),
                    domain: domain.clone(),
                    condition: Box::new(dc),
                })
            }
        })()),

        // Constraint programming
        TLExpr::AllDifferent { .. } | TLExpr::GlobalCardinality { .. } => {
            ctx.unsupported_nodes
                .push("AllDifferent/GlobalCardinality".to_string());
            Some(Ok(zero()))
        }

        // Abductive reasoning
        TLExpr::Abducible { .. } => Some(Ok(zero())),
        TLExpr::Explain { formula } => Some((|| {
            let df = diff_expr(formula, ctx)?;
            Ok(TLExpr::Explain {
                formula: Box::new(df),
            })
        })()),

        // Symbol literal has no numeric derivative
        TLExpr::SymbolLiteral(_) => Some(Ok(zero())),

        // Match — differentiate each arm body; scrutinee is treated as non-differentiable
        TLExpr::Match { scrutinee, arms } => Some((|| {
            let new_arms = arms
                .iter()
                .map(|(pat, body)| {
                    let db = diff_expr(body, ctx)?;
                    Ok::<_, crate::symbolic_diff::DiffError>((pat.clone(), Box::new(db)))
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(TLExpr::Match {
                scrutinee: scrutinee.clone(),
                arms: new_arms,
            })
        })()),

        _ => None,
    }
}
