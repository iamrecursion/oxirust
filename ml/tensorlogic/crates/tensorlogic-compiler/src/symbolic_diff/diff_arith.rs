//! Differentiation arms for constants, scalar vars, arithmetic, and transcendentals.

use tensorlogic_ir::TLExpr;

use super::diff_core::diff_expr;
use super::helpers::{one, zero};
use super::types::{DiffContext, DiffError};

/// Try to differentiate `expr` using arithmetic / transcendental rules.
///
/// Returns `Some(Ok(..))` when handled, `Some(Err(..))` on propagation failure,
/// or `None` when this category does not apply to `expr`.
pub(super) fn try_diff_arith(
    expr: &TLExpr,
    ctx: &mut DiffContext<'_>,
) -> Option<Result<TLExpr, DiffError>> {
    match expr {
        // Constants: derivative is 0
        TLExpr::Constant(_) => Some(Ok(zero())),

        // Zero-arity predicates as scalar variables
        TLExpr::Pred { name, args } if args.is_empty() => {
            if name == &ctx.var {
                Some(Ok(one()))
            } else {
                Some(Ok(zero()))
            }
        }

        // Non-zero-arity predicates: treat as constants w.r.t. differentiation
        TLExpr::Pred { name, .. } => {
            let label = format!("Pred({})", name);
            ctx.unsupported_nodes.push(label);
            Some(Ok(zero()))
        }

        // Sum rule
        TLExpr::Add(a, b) => Some((|| {
            let da = diff_expr(a, ctx)?;
            let db = diff_expr(b, ctx)?;
            Ok(TLExpr::add(da, db))
        })()),

        // Subtraction rule
        TLExpr::Sub(a, b) => Some((|| {
            let da = diff_expr(a, ctx)?;
            let db = diff_expr(b, ctx)?;
            Ok(TLExpr::sub(da, db))
        })()),

        // Product rule
        TLExpr::Mul(a, b) => Some((|| {
            let da = diff_expr(a, ctx)?;
            let db = diff_expr(b, ctx)?;
            let term1 = TLExpr::mul(da, *b.clone());
            let term2 = TLExpr::mul(*a.clone(), db);
            Ok(TLExpr::add(term1, term2))
        })()),

        // Quotient rule
        TLExpr::Div(a, b) => Some((|| {
            let da = diff_expr(a, ctx)?;
            let db = diff_expr(b, ctx)?;
            let num = TLExpr::sub(TLExpr::mul(da, *b.clone()), TLExpr::mul(*a.clone(), db));
            let denom = TLExpr::pow(*b.clone(), TLExpr::Constant(2.0));
            Ok(TLExpr::div(num, denom))
        })()),

        // Power rule (constant exponent) and general f^g
        TLExpr::Pow(base, exp) => Some((|| {
            let da = diff_expr(base, ctx)?;
            let dn = diff_expr(exp, ctx)?;
            match exp.as_ref() {
                TLExpr::Constant(n) => {
                    let n_minus_1 = TLExpr::Constant(n - 1.0);
                    let base_pow = TLExpr::pow(*base.clone(), n_minus_1);
                    let coeff = TLExpr::mul(TLExpr::Constant(*n), base_pow);
                    Ok(TLExpr::mul(coeff, da))
                }
                _ => {
                    let ln_f = TLExpr::Log(Box::new(*base.clone()));
                    let g_prime_ln_f = TLExpr::mul(dn, ln_f);
                    let f_prime_over_f = TLExpr::div(da, *base.clone());
                    let g_times_fp_over_f = TLExpr::mul(*exp.clone(), f_prime_over_f);
                    let bracket = TLExpr::add(g_prime_ln_f, g_times_fp_over_f);
                    Ok(TLExpr::mul(TLExpr::Pow(base.clone(), exp.clone()), bracket))
                }
            }
        })()),

        // Unary transcendental functions
        TLExpr::Abs(inner) => Some((|| {
            let di = diff_expr(inner, ctx)?;
            let sign = TLExpr::div(*inner.clone(), TLExpr::Abs(inner.clone()));
            Ok(TLExpr::mul(sign, di))
        })()),

        TLExpr::Sqrt(inner) => Some((|| {
            let di = diff_expr(inner, ctx)?;
            let two_sqrt = TLExpr::mul(TLExpr::Constant(2.0), TLExpr::Sqrt(inner.clone()));
            Ok(TLExpr::div(di, two_sqrt))
        })()),

        TLExpr::Exp(inner) => Some((|| {
            let di = diff_expr(inner, ctx)?;
            Ok(TLExpr::mul(TLExpr::Exp(inner.clone()), di))
        })()),

        TLExpr::Log(inner) => Some((|| {
            let di = diff_expr(inner, ctx)?;
            Ok(TLExpr::div(di, *inner.clone()))
        })()),

        TLExpr::Sin(inner) => Some((|| {
            let di = diff_expr(inner, ctx)?;
            Ok(TLExpr::mul(TLExpr::Cos(inner.clone()), di))
        })()),

        TLExpr::Cos(inner) => Some((|| {
            let di = diff_expr(inner, ctx)?;
            let neg_sin = TLExpr::sub(TLExpr::Constant(0.0), TLExpr::Sin(inner.clone()));
            Ok(TLExpr::mul(neg_sin, di))
        })()),

        TLExpr::Tan(inner) => Some((|| {
            let di = diff_expr(inner, ctx)?;
            let tan_sq = TLExpr::pow(TLExpr::Tan(inner.clone()), TLExpr::Constant(2.0));
            let sec_sq = TLExpr::add(TLExpr::Constant(1.0), tan_sq);
            Ok(TLExpr::mul(sec_sq, di))
        })()),

        // Piecewise constant — derivative is 0 almost everywhere
        TLExpr::Floor(_) | TLExpr::Ceil(_) | TLExpr::Round(_) => {
            ctx.unsupported_nodes
                .push("Floor/Ceil/Round (piecewise constant)".to_string());
            Some(Ok(zero()))
        }

        _ => None,
    }
}
