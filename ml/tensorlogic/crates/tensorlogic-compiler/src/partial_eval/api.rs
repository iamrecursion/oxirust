//! Public entry points for the partial evaluator:
//! [`partially_evaluate`], [`specialize`], and [`specialize_batch`].

use std::collections::HashSet;

use tensorlogic_ir::TLExpr;

use super::helpers::collect_free_pred_vars;
use super::pe_core::pe_rec;
use super::types::{PEConfig, PEEnv, PEResult, PEStats};

/// Partially evaluate `expr` under `env` according to `config`.
///
/// Variables are zero-arity predicates. Any such predicate whose name appears
/// in `env` is substituted with the bound value. Arithmetic and boolean
/// identities are applied when both operands are known. Dead branches in
/// logical operators are pruned when one operand resolves to a concrete boolean.
/// `Let` bindings whose value reduces to a concrete constant are inlined into
/// the body.
///
/// The returned [`PEResult`] contains:
/// - The residual expression (partially reduced).
/// - Accumulated statistics.
/// - The names of variables still free in the output.
pub fn partially_evaluate(expr: &TLExpr, env: &PEEnv, config: &PEConfig) -> PEResult {
    let mut stats = PEStats::default();
    let result_expr = pe_rec(expr.clone(), env, config, 0, &mut stats);

    // Compute residual free variables in the output expression
    let mut free_set = HashSet::new();
    collect_free_pred_vars(&result_expr, &HashSet::new(), &mut free_set);

    let mut residual_vars: Vec<String> = free_set.into_iter().collect();
    residual_vars.sort();

    PEResult {
        expr: result_expr,
        stats,
        residual_vars,
    }
}

// ── Specialization helpers ────────────────────────────────────────────────────

/// Specialize `expr` by binding all provided `(name, f64)` pairs and returning
/// the residual expression.
///
/// This is a convenience wrapper around [`partially_evaluate`] that builds a
/// [`PEEnv`] from the supplied bindings.
pub fn specialize(expr: &TLExpr, bindings: &[(String, f64)], config: &PEConfig) -> PEResult {
    let mut env = PEEnv::new();
    for (name, val) in bindings {
        env.bind_f64(name.clone(), *val);
    }
    partially_evaluate(expr, &env, config)
}

/// Multi-point specialization: evaluate `expr` at multiple binding sets and
/// return one [`PEResult`] per binding set, in the same order as `binding_sets`.
///
/// This can be used, for example, to compile a parameterised expression for a
/// batch of concrete parameter values.
pub fn specialize_batch(
    expr: &TLExpr,
    binding_sets: &[Vec<(String, f64)>],
    config: &PEConfig,
) -> Vec<PEResult> {
    binding_sets
        .iter()
        .map(|bindings| specialize(expr, bindings, config))
        .collect()
}
