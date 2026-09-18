//! Differentiation arms for temporal, modal, and hybrid-logic operators.

use tensorlogic_ir::TLExpr;

use super::diff_core::diff_expr;
use super::helpers::zero;
use super::types::{DiffContext, DiffError};

/// Try to differentiate `expr` using temporal / modal / hybrid rules.
pub(super) fn try_diff_modal(
    expr: &TLExpr,
    ctx: &mut DiffContext<'_>,
) -> Option<Result<TLExpr, DiffError>> {
    match expr {
        TLExpr::Next(inner) => Some((|| {
            let di = diff_expr(inner, ctx)?;
            Ok(TLExpr::Next(Box::new(di)))
        })()),

        TLExpr::Eventually(inner) => Some((|| {
            let di = diff_expr(inner, ctx)?;
            Ok(TLExpr::Eventually(Box::new(di)))
        })()),

        TLExpr::Always(inner) => Some((|| {
            let di = diff_expr(inner, ctx)?;
            Ok(TLExpr::Always(Box::new(di)))
        })()),

        TLExpr::Box(inner) => Some((|| {
            let di = diff_expr(inner, ctx)?;
            Ok(TLExpr::Box(Box::new(di)))
        })()),

        TLExpr::Diamond(inner) => Some((|| {
            let di = diff_expr(inner, ctx)?;
            Ok(TLExpr::Diamond(Box::new(di)))
        })()),

        TLExpr::Until { before, after } => Some((|| {
            let db = diff_expr(before, ctx)?;
            let da = diff_expr(after, ctx)?;
            Ok(TLExpr::Until {
                before: Box::new(db),
                after: Box::new(da),
            })
        })()),

        TLExpr::Release { released, releaser } => Some((|| {
            let dr = diff_expr(released, ctx)?;
            let da = diff_expr(releaser, ctx)?;
            Ok(TLExpr::Release {
                released: Box::new(dr),
                releaser: Box::new(da),
            })
        })()),

        TLExpr::WeakUntil { before, after } => Some((|| {
            let db = diff_expr(before, ctx)?;
            let da = diff_expr(after, ctx)?;
            Ok(TLExpr::WeakUntil {
                before: Box::new(db),
                after: Box::new(da),
            })
        })()),

        TLExpr::StrongRelease { released, releaser } => Some((|| {
            let dr = diff_expr(released, ctx)?;
            let da = diff_expr(releaser, ctx)?;
            Ok(TLExpr::StrongRelease {
                released: Box::new(dr),
                releaser: Box::new(da),
            })
        })()),

        // Hybrid / modal
        TLExpr::Nominal { .. } => Some(Ok(zero())),

        TLExpr::At { nominal, formula } => Some((|| {
            let df = diff_expr(formula, ctx)?;
            Ok(TLExpr::At {
                nominal: nominal.clone(),
                formula: Box::new(df),
            })
        })()),

        TLExpr::Somewhere { formula } => Some((|| {
            let df = diff_expr(formula, ctx)?;
            Ok(TLExpr::Somewhere {
                formula: Box::new(df),
            })
        })()),

        TLExpr::Everywhere { formula } => Some((|| {
            let df = diff_expr(formula, ctx)?;
            Ok(TLExpr::Everywhere {
                formula: Box::new(df),
            })
        })()),

        _ => None,
    }
}
