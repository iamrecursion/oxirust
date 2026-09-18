//! Differentiation arms for logical connectives, quantifiers, Let, and control flow.

use tensorlogic_ir::TLExpr;

use super::diff_core::diff_expr;
use super::helpers::{derivative_of_function, zero};
use super::types::{DiffContext, DiffError};

/// Try to differentiate `expr` using Boolean / quantifier / Let / flow rules.
pub(super) fn try_diff_logic(
    expr: &TLExpr,
    ctx: &mut DiffContext<'_>,
) -> Option<Result<TLExpr, DiffError>> {
    match expr {
        // Logical AND differential: d(AND(a,b))/dx = OR(AND(da, b), AND(a, db))
        TLExpr::And(a, b) => Some((|| {
            let da = diff_expr(a, ctx)?;
            let db = diff_expr(b, ctx)?;
            let term1 = TLExpr::and(da, *b.clone());
            let term2 = TLExpr::and(*a.clone(), db);
            Ok(TLExpr::or(term1, term2))
        })()),

        TLExpr::Or(a, b) => Some((|| {
            let da = diff_expr(a, ctx)?;
            let db = diff_expr(b, ctx)?;
            Ok(TLExpr::or(da, db))
        })()),

        TLExpr::Not(inner) => Some((|| {
            let di = diff_expr(inner, ctx)?;
            Ok(TLExpr::negate(di))
        })()),

        TLExpr::Imply(a, b) => {
            let expanded = TLExpr::or(TLExpr::negate(*a.clone()), *b.clone());
            Some(diff_expr(&expanded, ctx))
        }

        // Quantifiers
        TLExpr::Exists { var, domain, body } => Some((|| {
            if var == &ctx.var {
                Ok(zero())
            } else {
                let dbody = diff_expr(body, ctx)?;
                Ok(TLExpr::Exists {
                    var: var.clone(),
                    domain: domain.clone(),
                    body: Box::new(dbody),
                })
            }
        })()),

        TLExpr::ForAll { var, domain, body } => Some((|| {
            if var == &ctx.var {
                Ok(zero())
            } else {
                let dbody = diff_expr(body, ctx)?;
                Ok(TLExpr::ForAll {
                    var: var.clone(),
                    domain: domain.clone(),
                    body: Box::new(dbody),
                })
            }
        })()),

        TLExpr::SoftExists {
            var,
            domain,
            body,
            temperature,
        } => Some((|| {
            if var == &ctx.var {
                Ok(zero())
            } else {
                let dbody = diff_expr(body, ctx)?;
                Ok(TLExpr::SoftExists {
                    var: var.clone(),
                    domain: domain.clone(),
                    body: Box::new(dbody),
                    temperature: *temperature,
                })
            }
        })()),

        TLExpr::SoftForAll {
            var,
            domain,
            body,
            temperature,
        } => Some((|| {
            if var == &ctx.var {
                Ok(zero())
            } else {
                let dbody = diff_expr(body, ctx)?;
                Ok(TLExpr::SoftForAll {
                    var: var.clone(),
                    domain: domain.clone(),
                    body: Box::new(dbody),
                    temperature: *temperature,
                })
            }
        })()),

        // Let binding with full chain-rule expansion
        TLExpr::Let { var, value, body } => Some((|| {
            if var == &ctx.var {
                Ok(zero())
            } else {
                let dbody_wrt_x = diff_expr(body, ctx)?;
                let dvalue_wrt_x = diff_expr(value, ctx)?;
                let saved_var = ctx.var.clone();
                ctx.var = var.clone();
                let dbody_wrt_z = diff_expr(body, ctx)?;
                ctx.var = saved_var;
                let chain_term = TLExpr::mul(dbody_wrt_z, dvalue_wrt_x);
                let let_term = TLExpr::Let {
                    var: var.clone(),
                    value: value.clone(),
                    body: Box::new(dbody_wrt_x),
                };
                Ok(TLExpr::add(let_term, chain_term))
            }
        })()),

        TLExpr::Score(inner) => Some((|| {
            let di = diff_expr(inner, ctx)?;
            Ok(TLExpr::Score(Box::new(di)))
        })()),

        TLExpr::Aggregate {
            op,
            var,
            domain,
            body,
            group_by,
        } => Some((|| {
            if var == &ctx.var {
                Ok(zero())
            } else {
                let dbody = diff_expr(body, ctx)?;
                Ok(TLExpr::Aggregate {
                    op: op.clone(),
                    var: var.clone(),
                    domain: domain.clone(),
                    body: Box::new(dbody),
                    group_by: group_by.clone(),
                })
            }
        })()),

        TLExpr::Lambda {
            var,
            var_type,
            body,
        } => Some((|| {
            if var == &ctx.var {
                Ok(zero())
            } else {
                let dbody = diff_expr(body, ctx)?;
                Ok(TLExpr::Lambda {
                    var: var.clone(),
                    var_type: var_type.clone(),
                    body: Box::new(dbody),
                })
            }
        })()),

        // Function application: chain rule
        TLExpr::Apply { function, argument } => Some((|| {
            let darg = diff_expr(argument, ctx)?;
            let f_prime = derivative_of_function(function);
            let chain = TLExpr::mul(
                TLExpr::Apply {
                    function: Box::new(f_prime),
                    argument: argument.clone(),
                },
                darg,
            );
            Ok(chain)
        })()),

        // If-then-else: differentiate each branch
        TLExpr::IfThenElse {
            condition,
            then_branch,
            else_branch,
        } => Some((|| {
            let dthen = diff_expr(then_branch, ctx)?;
            let delse = diff_expr(else_branch, ctx)?;
            Ok(TLExpr::IfThenElse {
                condition: condition.clone(),
                then_branch: Box::new(dthen),
                else_branch: Box::new(delse),
            })
        })()),

        // Min / Max: piecewise (subgradient not emitted)
        TLExpr::Min(a, b) => Some((|| {
            ctx.unsupported_nodes
                .push("Min (piecewise; subgradient not emitted)".to_string());
            let da = diff_expr(a, ctx)?;
            let db = diff_expr(b, ctx)?;
            Ok(TLExpr::Min(Box::new(da), Box::new(db)))
        })()),

        TLExpr::Max(a, b) => Some((|| {
            ctx.unsupported_nodes
                .push("Max (piecewise; subgradient not emitted)".to_string());
            let da = diff_expr(a, ctx)?;
            let db = diff_expr(b, ctx)?;
            Ok(TLExpr::Max(Box::new(da), Box::new(db)))
        })()),

        TLExpr::Mod(_, _) => {
            ctx.unsupported_nodes
                .push("Mod (piecewise constant)".to_string());
            Some(Ok(zero()))
        }

        TLExpr::Eq(_, _)
        | TLExpr::Lt(_, _)
        | TLExpr::Gt(_, _)
        | TLExpr::Lte(_, _)
        | TLExpr::Gte(_, _) => {
            ctx.unsupported_nodes
                .push("Comparison (discrete; derivative is 0)".to_string());
            Some(Ok(zero()))
        }

        _ => None,
    }
}
