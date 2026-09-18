//! Core recursive driver for partial evaluation: depth guard, leaf handling
//! for `Constant` and variable-like `Pred { args: [] }`, and dispatch to each
//! category submodule. Categories return `Result<TLExpr, TLExpr>` so ownership
//! of the expression is preserved on an unhandled arm.
//!
//! The dispatch order is:
//! 1. [`try_pe_arith`] — binary arithmetic / min / max
//! 2. [`try_pe_math`] — unary numeric / comparisons
//! 3. [`try_pe_logic`] — Boolean connectives, `IfThenElse`, `Let`
//! 4. [`try_pe_quantifiers`] — binders (`Exists`, `ForAll`, lambdas, ...)
//! 5. [`try_pe_passthrough`] — modal/temporal/fuzzy/etc. and closed leaves
//!
//! If every category declines to handle the expression, it is returned
//! unchanged.

use tensorlogic_ir::TLExpr;

use super::pe_arith::try_pe_arith;
use super::pe_logic::try_pe_logic;
use super::pe_math::try_pe_math;
use super::pe_passthrough::try_pe_passthrough;
use super::pe_quantifiers::try_pe_quantifiers;
use super::types::{PEConfig, PEEnv, PEStats};

/// Recursively partially-evaluate `expr` in `env` under `config`.
///
/// This is the internal driver shared by all public entry points. It bumps
/// the visit counter, enforces `config.max_depth`, handles leaf cases
/// (`Constant` and zero-arity `Pred`), then dispatches through each category
/// in turn. The first category that handles the node wins.
pub(super) fn pe_rec(
    expr: TLExpr,
    env: &PEEnv,
    config: &PEConfig,
    depth: usize,
    stats: &mut PEStats,
) -> TLExpr {
    stats.nodes_visited = stats.nodes_visited.saturating_add(1);

    // Depth guard — bail out without further recursion.
    if depth > config.max_depth {
        return expr;
    }

    // ── Leaf: Constant ───────────────────────────────────────────────────────
    if let TLExpr::Constant(_) = &expr {
        return expr;
    }

    // ── Leaf: zero-arity Pred acts as a logical variable ─────────────────────
    if let TLExpr::Pred { name, args } = &expr {
        if args.is_empty() {
            if let Some(pval) = env.lookup(name) {
                stats.nodes_reduced = stats.nodes_reduced.saturating_add(1);
                return pval.to_expr();
            }
        }
        // Proper predicate (or unbound variable) — keep as-is.
        return expr;
    }

    // ── Dispatch to each category ────────────────────────────────────────────
    let expr = match try_pe_arith(expr, env, config, depth, stats) {
        Ok(r) => return r,
        Err(e) => e,
    };
    let expr = match try_pe_math(expr, env, config, depth, stats) {
        Ok(r) => return r,
        Err(e) => e,
    };
    let expr = match try_pe_logic(expr, env, config, depth, stats) {
        Ok(r) => return r,
        Err(e) => e,
    };
    let expr = match try_pe_quantifiers(expr, env, config, depth, stats) {
        Ok(r) => return r,
        Err(e) => e,
    };
    let expr = match try_pe_passthrough(expr, env, config, depth, stats) {
        Ok(r) => return r,
        Err(e) => e,
    };

    // No category claimed the expression — return unchanged.
    expr
}
