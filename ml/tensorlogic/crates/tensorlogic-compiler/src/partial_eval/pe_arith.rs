//! Partial-evaluation arms for binary arithmetic nodes: `Add`, `Sub`, `Mul`,
//! `Div`, `Pow`, `Mod`, `Min`, `Max`.

use tensorlogic_ir::TLExpr;

use super::helpers::as_constant;
use super::pe_core::pe_rec;
use super::types::{PEConfig, PEEnv, PEStats};

/// Attempt to partially evaluate an arithmetic node. Returns `Ok(result)` when
/// handled; `Err(expr)` to pass the unchanged expression to the next category.
pub(super) fn try_pe_arith(
    expr: TLExpr,
    env: &PEEnv,
    config: &PEConfig,
    depth: usize,
    stats: &mut PEStats,
) -> Result<TLExpr, TLExpr> {
    match expr {
        TLExpr::Add(lhs, rhs) => {
            let la = pe_rec(*lhs, env, config, depth + 1, stats);
            let ra = pe_rec(*rhs, env, config, depth + 1, stats);
            if config.fold_arithmetic {
                match (as_constant(&la), as_constant(&ra)) {
                    (Some(a), Some(b)) => {
                        stats.nodes_reduced = stats.nodes_reduced.saturating_add(1);
                        return Ok(TLExpr::Constant(a + b));
                    }
                    (Some(0.0), None) => {
                        stats.nodes_reduced = stats.nodes_reduced.saturating_add(1);
                        return Ok(ra);
                    }
                    (None, Some(0.0)) => {
                        stats.nodes_reduced = stats.nodes_reduced.saturating_add(1);
                        return Ok(la);
                    }
                    _ => {}
                }
            }
            Ok(TLExpr::Add(Box::new(la), Box::new(ra)))
        }

        TLExpr::Sub(lhs, rhs) => {
            let la = pe_rec(*lhs, env, config, depth + 1, stats);
            let ra = pe_rec(*rhs, env, config, depth + 1, stats);
            if config.fold_arithmetic {
                match (as_constant(&la), as_constant(&ra)) {
                    (Some(a), Some(b)) => {
                        stats.nodes_reduced = stats.nodes_reduced.saturating_add(1);
                        return Ok(TLExpr::Constant(a - b));
                    }
                    (_, Some(0.0)) => {
                        stats.nodes_reduced = stats.nodes_reduced.saturating_add(1);
                        return Ok(la);
                    }
                    _ => {}
                }
            }
            Ok(TLExpr::Sub(Box::new(la), Box::new(ra)))
        }

        TLExpr::Mul(lhs, rhs) => {
            let la = pe_rec(*lhs, env, config, depth + 1, stats);
            let ra = pe_rec(*rhs, env, config, depth + 1, stats);
            if config.fold_arithmetic {
                match (as_constant(&la), as_constant(&ra)) {
                    (Some(a), Some(b)) => {
                        stats.nodes_reduced = stats.nodes_reduced.saturating_add(1);
                        return Ok(TLExpr::Constant(a * b));
                    }
                    (Some(0.0), _) => {
                        stats.nodes_reduced = stats.nodes_reduced.saturating_add(1);
                        return Ok(TLExpr::Constant(0.0));
                    }
                    (_, Some(0.0)) => {
                        stats.nodes_reduced = stats.nodes_reduced.saturating_add(1);
                        return Ok(TLExpr::Constant(0.0));
                    }
                    (Some(1.0), _) => {
                        stats.nodes_reduced = stats.nodes_reduced.saturating_add(1);
                        return Ok(ra);
                    }
                    (_, Some(1.0)) => {
                        stats.nodes_reduced = stats.nodes_reduced.saturating_add(1);
                        return Ok(la);
                    }
                    _ => {}
                }
            }
            Ok(TLExpr::Mul(Box::new(la), Box::new(ra)))
        }

        TLExpr::Div(lhs, rhs) => {
            let la = pe_rec(*lhs, env, config, depth + 1, stats);
            let ra = pe_rec(*rhs, env, config, depth + 1, stats);
            if config.fold_arithmetic {
                match (as_constant(&la), as_constant(&ra)) {
                    (Some(a), Some(b)) if b != 0.0 => {
                        stats.nodes_reduced = stats.nodes_reduced.saturating_add(1);
                        return Ok(TLExpr::Constant(a / b));
                    }
                    (_, Some(1.0)) => {
                        // Div by 1 → numerator
                        stats.nodes_reduced = stats.nodes_reduced.saturating_add(1);
                        return Ok(la);
                    }
                    _ => {} // Div by zero: keep symbolic — do NOT fold
                }
            }
            Ok(TLExpr::Div(Box::new(la), Box::new(ra)))
        }

        TLExpr::Pow(base_expr, exp_expr) => {
            let ba = pe_rec(*base_expr, env, config, depth + 1, stats);
            let ea = pe_rec(*exp_expr, env, config, depth + 1, stats);
            if config.fold_arithmetic {
                match (as_constant(&ba), as_constant(&ea)) {
                    (Some(b), Some(e)) => {
                        stats.nodes_reduced = stats.nodes_reduced.saturating_add(1);
                        return Ok(TLExpr::Constant(b.powf(e)));
                    }
                    (_, Some(0.0)) => {
                        stats.nodes_reduced = stats.nodes_reduced.saturating_add(1);
                        return Ok(TLExpr::Constant(1.0));
                    }
                    (_, Some(1.0)) => {
                        stats.nodes_reduced = stats.nodes_reduced.saturating_add(1);
                        return Ok(ba);
                    }
                    (Some(0.0), _) => {
                        stats.nodes_reduced = stats.nodes_reduced.saturating_add(1);
                        return Ok(TLExpr::Constant(0.0));
                    }
                    (Some(1.0), _) => {
                        stats.nodes_reduced = stats.nodes_reduced.saturating_add(1);
                        return Ok(TLExpr::Constant(1.0));
                    }
                    _ => {}
                }
            }
            Ok(TLExpr::Pow(Box::new(ba), Box::new(ea)))
        }

        TLExpr::Mod(lhs, rhs) => {
            let la = pe_rec(*lhs, env, config, depth + 1, stats);
            let ra = pe_rec(*rhs, env, config, depth + 1, stats);
            if config.fold_arithmetic {
                if let (Some(a), Some(b)) = (as_constant(&la), as_constant(&ra)) {
                    if b != 0.0 {
                        stats.nodes_reduced = stats.nodes_reduced.saturating_add(1);
                        return Ok(TLExpr::Constant(a % b));
                    }
                }
            }
            Ok(TLExpr::Mod(Box::new(la), Box::new(ra)))
        }

        TLExpr::Min(lhs, rhs) => {
            let la = pe_rec(*lhs, env, config, depth + 1, stats);
            let ra = pe_rec(*rhs, env, config, depth + 1, stats);
            if config.fold_arithmetic {
                if let (Some(a), Some(b)) = (as_constant(&la), as_constant(&ra)) {
                    stats.nodes_reduced = stats.nodes_reduced.saturating_add(1);
                    return Ok(TLExpr::Constant(a.min(b)));
                }
            }
            Ok(TLExpr::Min(Box::new(la), Box::new(ra)))
        }

        TLExpr::Max(lhs, rhs) => {
            let la = pe_rec(*lhs, env, config, depth + 1, stats);
            let ra = pe_rec(*rhs, env, config, depth + 1, stats);
            if config.fold_arithmetic {
                if let (Some(a), Some(b)) = (as_constant(&la), as_constant(&ra)) {
                    stats.nodes_reduced = stats.nodes_reduced.saturating_add(1);
                    return Ok(TLExpr::Constant(a.max(b)));
                }
            }
            Ok(TLExpr::Max(Box::new(la), Box::new(ra)))
        }

        other => Err(other),
    }
}
