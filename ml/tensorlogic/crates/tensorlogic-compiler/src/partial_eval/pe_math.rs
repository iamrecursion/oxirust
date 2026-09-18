//! Partial-evaluation arms for unary math operations (`Abs`, `Floor`, ...,
//! `Tan`) and binary comparisons (`Eq`, `Lt`, `Gt`, `Lte`, `Gte`).

use tensorlogic_ir::TLExpr;

use super::helpers::as_constant;
use super::pe_core::pe_rec;
use super::types::{PEConfig, PEEnv, PEStats};

/// Attempt to partially evaluate a unary math / comparison node. Returns
/// `Ok(result)` when handled; `Err(expr)` to pass the unchanged expression to
/// the next category.
pub(super) fn try_pe_math(
    expr: TLExpr,
    env: &PEEnv,
    config: &PEConfig,
    depth: usize,
    stats: &mut PEStats,
) -> Result<TLExpr, TLExpr> {
    match expr {
        TLExpr::Abs(inner) => {
            let e = pe_rec(*inner, env, config, depth + 1, stats);
            if config.fold_arithmetic {
                if let Some(v) = as_constant(&e) {
                    stats.nodes_reduced = stats.nodes_reduced.saturating_add(1);
                    return Ok(TLExpr::Constant(v.abs()));
                }
            }
            Ok(TLExpr::Abs(Box::new(e)))
        }
        TLExpr::Floor(inner) => {
            let e = pe_rec(*inner, env, config, depth + 1, stats);
            if config.fold_arithmetic {
                if let Some(v) = as_constant(&e) {
                    stats.nodes_reduced = stats.nodes_reduced.saturating_add(1);
                    return Ok(TLExpr::Constant(v.floor()));
                }
            }
            Ok(TLExpr::Floor(Box::new(e)))
        }
        TLExpr::Ceil(inner) => {
            let e = pe_rec(*inner, env, config, depth + 1, stats);
            if config.fold_arithmetic {
                if let Some(v) = as_constant(&e) {
                    stats.nodes_reduced = stats.nodes_reduced.saturating_add(1);
                    return Ok(TLExpr::Constant(v.ceil()));
                }
            }
            Ok(TLExpr::Ceil(Box::new(e)))
        }
        TLExpr::Round(inner) => {
            let e = pe_rec(*inner, env, config, depth + 1, stats);
            if config.fold_arithmetic {
                if let Some(v) = as_constant(&e) {
                    stats.nodes_reduced = stats.nodes_reduced.saturating_add(1);
                    return Ok(TLExpr::Constant(v.round()));
                }
            }
            Ok(TLExpr::Round(Box::new(e)))
        }
        TLExpr::Sqrt(inner) => {
            let e = pe_rec(*inner, env, config, depth + 1, stats);
            if config.fold_arithmetic {
                if let Some(v) = as_constant(&e) {
                    stats.nodes_reduced = stats.nodes_reduced.saturating_add(1);
                    return Ok(TLExpr::Constant(v.sqrt()));
                }
            }
            Ok(TLExpr::Sqrt(Box::new(e)))
        }
        TLExpr::Exp(inner) => {
            let e = pe_rec(*inner, env, config, depth + 1, stats);
            if config.fold_arithmetic {
                if let Some(v) = as_constant(&e) {
                    stats.nodes_reduced = stats.nodes_reduced.saturating_add(1);
                    return Ok(TLExpr::Constant(v.exp()));
                }
            }
            Ok(TLExpr::Exp(Box::new(e)))
        }
        TLExpr::Log(inner) => {
            let e = pe_rec(*inner, env, config, depth + 1, stats);
            if config.fold_arithmetic {
                if let Some(v) = as_constant(&e) {
                    stats.nodes_reduced = stats.nodes_reduced.saturating_add(1);
                    return Ok(TLExpr::Constant(v.ln()));
                }
            }
            Ok(TLExpr::Log(Box::new(e)))
        }
        TLExpr::Sin(inner) => {
            let e = pe_rec(*inner, env, config, depth + 1, stats);
            if config.fold_arithmetic {
                if let Some(v) = as_constant(&e) {
                    stats.nodes_reduced = stats.nodes_reduced.saturating_add(1);
                    return Ok(TLExpr::Constant(v.sin()));
                }
            }
            Ok(TLExpr::Sin(Box::new(e)))
        }
        TLExpr::Cos(inner) => {
            let e = pe_rec(*inner, env, config, depth + 1, stats);
            if config.fold_arithmetic {
                if let Some(v) = as_constant(&e) {
                    stats.nodes_reduced = stats.nodes_reduced.saturating_add(1);
                    return Ok(TLExpr::Constant(v.cos()));
                }
            }
            Ok(TLExpr::Cos(Box::new(e)))
        }
        TLExpr::Tan(inner) => {
            let e = pe_rec(*inner, env, config, depth + 1, stats);
            if config.fold_arithmetic {
                if let Some(v) = as_constant(&e) {
                    stats.nodes_reduced = stats.nodes_reduced.saturating_add(1);
                    return Ok(TLExpr::Constant(v.tan()));
                }
            }
            Ok(TLExpr::Tan(Box::new(e)))
        }

        TLExpr::Eq(lhs, rhs) => {
            let la = pe_rec(*lhs, env, config, depth + 1, stats);
            let ra = pe_rec(*rhs, env, config, depth + 1, stats);
            if config.fold_arithmetic {
                if let (Some(a), Some(b)) = (as_constant(&la), as_constant(&ra)) {
                    stats.nodes_reduced = stats.nodes_reduced.saturating_add(1);
                    return Ok(TLExpr::Constant(if (a - b).abs() < f64::EPSILON {
                        1.0
                    } else {
                        0.0
                    }));
                }
            }
            Ok(TLExpr::Eq(Box::new(la), Box::new(ra)))
        }
        TLExpr::Lt(lhs, rhs) => {
            let la = pe_rec(*lhs, env, config, depth + 1, stats);
            let ra = pe_rec(*rhs, env, config, depth + 1, stats);
            if config.fold_arithmetic {
                if let (Some(a), Some(b)) = (as_constant(&la), as_constant(&ra)) {
                    stats.nodes_reduced = stats.nodes_reduced.saturating_add(1);
                    return Ok(TLExpr::Constant(if a < b { 1.0 } else { 0.0 }));
                }
            }
            Ok(TLExpr::Lt(Box::new(la), Box::new(ra)))
        }
        TLExpr::Gt(lhs, rhs) => {
            let la = pe_rec(*lhs, env, config, depth + 1, stats);
            let ra = pe_rec(*rhs, env, config, depth + 1, stats);
            if config.fold_arithmetic {
                if let (Some(a), Some(b)) = (as_constant(&la), as_constant(&ra)) {
                    stats.nodes_reduced = stats.nodes_reduced.saturating_add(1);
                    return Ok(TLExpr::Constant(if a > b { 1.0 } else { 0.0 }));
                }
            }
            Ok(TLExpr::Gt(Box::new(la), Box::new(ra)))
        }
        TLExpr::Lte(lhs, rhs) => {
            let la = pe_rec(*lhs, env, config, depth + 1, stats);
            let ra = pe_rec(*rhs, env, config, depth + 1, stats);
            if config.fold_arithmetic {
                if let (Some(a), Some(b)) = (as_constant(&la), as_constant(&ra)) {
                    stats.nodes_reduced = stats.nodes_reduced.saturating_add(1);
                    return Ok(TLExpr::Constant(if a <= b { 1.0 } else { 0.0 }));
                }
            }
            Ok(TLExpr::Lte(Box::new(la), Box::new(ra)))
        }
        TLExpr::Gte(lhs, rhs) => {
            let la = pe_rec(*lhs, env, config, depth + 1, stats);
            let ra = pe_rec(*rhs, env, config, depth + 1, stats);
            if config.fold_arithmetic {
                if let (Some(a), Some(b)) = (as_constant(&la), as_constant(&ra)) {
                    stats.nodes_reduced = stats.nodes_reduced.saturating_add(1);
                    return Ok(TLExpr::Constant(if a >= b { 1.0 } else { 0.0 }));
                }
            }
            Ok(TLExpr::Gte(Box::new(la), Box::new(ra)))
        }

        other => Err(other),
    }
}
