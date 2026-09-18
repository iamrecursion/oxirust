//! Core recursive dispatcher for symbolic differentiation.
//!
//! `diff_expr` applies the depth guard and delegates to `diff_expr_inner`, which
//! tries each category module (arithmetic, logic, modal, fuzzy, sets) in turn.

use tensorlogic_ir::TLExpr;

use super::diff_arith::try_diff_arith;
use super::diff_fuzzy::try_diff_fuzzy;
use super::diff_logic::try_diff_logic;
use super::diff_modal::try_diff_modal;
use super::diff_sets::try_diff_sets;
use super::helpers::zero;
use super::types::{DiffContext, DiffError};

pub(super) fn diff_expr(expr: &TLExpr, ctx: &mut DiffContext<'_>) -> Result<TLExpr, DiffError> {
    if ctx.depth >= ctx.config.max_expr_depth {
        return Err(DiffError::MaxDepthExceeded);
    }
    ctx.depth += 1;
    let result = diff_expr_inner(expr, ctx);
    ctx.depth -= 1;
    result
}

fn diff_expr_inner(expr: &TLExpr, ctx: &mut DiffContext<'_>) -> Result<TLExpr, DiffError> {
    if let Some(result) = try_diff_arith(expr, ctx) {
        return result;
    }
    if let Some(result) = try_diff_logic(expr, ctx) {
        return result;
    }
    if let Some(result) = try_diff_modal(expr, ctx) {
        return result;
    }
    if let Some(result) = try_diff_fuzzy(expr, ctx) {
        return result;
    }
    if let Some(result) = try_diff_sets(expr, ctx) {
        return result;
    }
    // Exhaustive fallback: any unrecognised node is treated as derivative-zero
    // while being recorded as unsupported.
    ctx.unsupported_nodes.push(format!("{:?}", expr));
    Ok(zero())
}
