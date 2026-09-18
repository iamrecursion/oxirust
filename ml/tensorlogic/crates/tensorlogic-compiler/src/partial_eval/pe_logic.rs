//! Partial-evaluation arms for Boolean connectives (`Not`, `And`, `Or`,
//! `Imply`), branching (`IfThenElse`), and `Let` bindings.

use tensorlogic_ir::TLExpr;

use super::helpers::as_constant;
use super::pe_core::pe_rec;
use super::types::{PEConfig, PEEnv, PEStats, PEValue};

/// Attempt to partially evaluate a control-flow / Boolean node. Returns
/// `Ok(result)` when handled; `Err(expr)` to pass the unchanged expression to
/// the next category.
pub(super) fn try_pe_logic(
    expr: TLExpr,
    env: &PEEnv,
    config: &PEConfig,
    depth: usize,
    stats: &mut PEStats,
) -> Result<TLExpr, TLExpr> {
    match expr {
        TLExpr::Not(inner) => {
            let e = pe_rec(*inner, env, config, depth + 1, stats);
            if config.fold_logic {
                if let Some(v) = as_constant(&e) {
                    let b = v != 0.0;
                    stats.nodes_reduced = stats.nodes_reduced.saturating_add(1);
                    return Ok(TLExpr::Constant(if b { 0.0 } else { 1.0 }));
                }
            }
            Ok(TLExpr::Not(Box::new(e)))
        }

        TLExpr::And(lhs, rhs) => {
            let la = pe_rec(*lhs, env, config, depth + 1, stats);
            if config.prune_branches {
                if let Some(a) = as_constant(&la) {
                    if a == 0.0 {
                        // false AND anything = false
                        stats.branches_pruned = stats.branches_pruned.saturating_add(1);
                        stats.nodes_reduced = stats.nodes_reduced.saturating_add(1);
                        return Ok(TLExpr::Constant(0.0));
                    }
                    // true AND b = pe(b)
                    let ra = pe_rec(*rhs, env, config, depth + 1, stats);
                    stats.branches_pruned = stats.branches_pruned.saturating_add(1);
                    stats.nodes_reduced = stats.nodes_reduced.saturating_add(1);
                    return Ok(ra);
                }
            }
            let ra = pe_rec(*rhs, env, config, depth + 1, stats);
            if config.prune_branches {
                if let Some(b) = as_constant(&ra) {
                    if b == 0.0 {
                        // anything AND false = false
                        stats.branches_pruned = stats.branches_pruned.saturating_add(1);
                        stats.nodes_reduced = stats.nodes_reduced.saturating_add(1);
                        return Ok(TLExpr::Constant(0.0));
                    }
                    // anything AND true = la
                    stats.branches_pruned = stats.branches_pruned.saturating_add(1);
                    stats.nodes_reduced = stats.nodes_reduced.saturating_add(1);
                    return Ok(la);
                }
            }
            if config.fold_logic {
                if let (Some(a), Some(b)) = (as_constant(&la), as_constant(&ra)) {
                    stats.nodes_reduced = stats.nodes_reduced.saturating_add(1);
                    return Ok(TLExpr::Constant(if a != 0.0 && b != 0.0 {
                        1.0
                    } else {
                        0.0
                    }));
                }
            }
            Ok(TLExpr::And(Box::new(la), Box::new(ra)))
        }

        TLExpr::Or(lhs, rhs) => {
            let la = pe_rec(*lhs, env, config, depth + 1, stats);
            if config.prune_branches {
                if let Some(a) = as_constant(&la) {
                    if a != 0.0 {
                        // true OR anything = true
                        stats.branches_pruned = stats.branches_pruned.saturating_add(1);
                        stats.nodes_reduced = stats.nodes_reduced.saturating_add(1);
                        return Ok(TLExpr::Constant(1.0));
                    }
                    // false OR b = pe(b)
                    let ra = pe_rec(*rhs, env, config, depth + 1, stats);
                    stats.branches_pruned = stats.branches_pruned.saturating_add(1);
                    stats.nodes_reduced = stats.nodes_reduced.saturating_add(1);
                    return Ok(ra);
                }
            }
            let ra = pe_rec(*rhs, env, config, depth + 1, stats);
            if config.prune_branches {
                if let Some(b) = as_constant(&ra) {
                    if b != 0.0 {
                        // anything OR true = true
                        stats.branches_pruned = stats.branches_pruned.saturating_add(1);
                        stats.nodes_reduced = stats.nodes_reduced.saturating_add(1);
                        return Ok(TLExpr::Constant(1.0));
                    }
                    // anything OR false = la
                    stats.branches_pruned = stats.branches_pruned.saturating_add(1);
                    stats.nodes_reduced = stats.nodes_reduced.saturating_add(1);
                    return Ok(la);
                }
            }
            if config.fold_logic {
                if let (Some(a), Some(b)) = (as_constant(&la), as_constant(&ra)) {
                    stats.nodes_reduced = stats.nodes_reduced.saturating_add(1);
                    return Ok(TLExpr::Constant(if a != 0.0 || b != 0.0 {
                        1.0
                    } else {
                        0.0
                    }));
                }
            }
            Ok(TLExpr::Or(Box::new(la), Box::new(ra)))
        }

        TLExpr::Imply(premise, conclusion) => {
            // Expand: a → b  ≡  ¬a ∨ b
            let not_a = Box::new(TLExpr::Not(premise));
            let expanded = TLExpr::Or(not_a, conclusion);
            Ok(pe_rec(expanded, env, config, depth, stats))
        }

        TLExpr::IfThenElse {
            condition,
            then_branch,
            else_branch,
        } => {
            let cond = pe_rec(*condition, env, config, depth + 1, stats);
            if config.prune_branches {
                if let Some(v) = as_constant(&cond) {
                    if v != 0.0 {
                        stats.branches_pruned = stats.branches_pruned.saturating_add(1);
                        return Ok(pe_rec(*then_branch, env, config, depth + 1, stats));
                    } else {
                        stats.branches_pruned = stats.branches_pruned.saturating_add(1);
                        return Ok(pe_rec(*else_branch, env, config, depth + 1, stats));
                    }
                }
            }
            let tb = pe_rec(*then_branch, env, config, depth + 1, stats);
            let eb = pe_rec(*else_branch, env, config, depth + 1, stats);
            Ok(TLExpr::IfThenElse {
                condition: Box::new(cond),
                then_branch: Box::new(tb),
                else_branch: Box::new(eb),
            })
        }

        TLExpr::Let { var, value, body } => {
            let evaluated_value = pe_rec(*value, env, config, depth + 1, stats);
            if config.inline_lets {
                let peval = PEValue::from_expr(evaluated_value.clone());
                if peval.is_concrete() {
                    // Extend environment with the new concrete binding and pe body
                    let new_env = env.extend(var.clone(), peval);
                    let result = pe_rec(*body, &new_env, config, depth + 1, stats);
                    stats.lets_inlined = stats.lets_inlined.saturating_add(1);
                    return Ok(result);
                }
                // Value is still symbolic — we can still propagate it for pe of body
                // by adding a symbolic binding (so if it is referenced we keep the symbol)
                let new_env = env.extend(var.clone(), PEValue::Symbolic(evaluated_value.clone()));
                let new_body = pe_rec(*body, &new_env, config, depth + 1, stats);
                Ok(TLExpr::Let {
                    var,
                    value: Box::new(evaluated_value),
                    body: Box::new(new_body),
                })
            } else {
                // Don't inline: just pe the body in an env that shadows the let var
                // (to prevent outer binding from interfering with the let var)
                let shadowed_env =
                    env.extend(var.clone(), PEValue::Symbolic(evaluated_value.clone()));
                let new_body = pe_rec(*body, &shadowed_env, config, depth + 1, stats);
                Ok(TLExpr::Let {
                    var,
                    value: Box::new(evaluated_value),
                    body: Box::new(new_body),
                })
            }
        }

        other => Err(other),
    }
}
