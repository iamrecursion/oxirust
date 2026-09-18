//! Partial-evaluation arms for quantifiers, aggregates, counting quantifiers,
//! set comprehensions, and lambda abstractions. Each arm shadows the bound
//! variable with a symbolic binding before recursing into the body.

use tensorlogic_ir::TLExpr;

use super::pe_core::pe_rec;
use super::types::{PEConfig, PEEnv, PEStats, PEValue};

/// Attempt to partially evaluate a binder (quantifier / aggregate / lambda /
/// comprehension) node. Returns `Ok(result)` when handled; `Err(expr)` to pass
/// the unchanged expression to the next category.
pub(super) fn try_pe_quantifiers(
    expr: TLExpr,
    env: &PEEnv,
    config: &PEConfig,
    depth: usize,
    stats: &mut PEStats,
) -> Result<TLExpr, TLExpr> {
    match expr {
        TLExpr::Exists { var, domain, body } => {
            let inner_env = env.extend(var.clone(), PEValue::Symbolic(TLExpr::pred(&var, vec![])));
            let new_body = pe_rec(*body, &inner_env, config, depth + 1, stats);
            Ok(TLExpr::Exists {
                var,
                domain,
                body: Box::new(new_body),
            })
        }

        TLExpr::ForAll { var, domain, body } => {
            let inner_env = env.extend(var.clone(), PEValue::Symbolic(TLExpr::pred(&var, vec![])));
            let new_body = pe_rec(*body, &inner_env, config, depth + 1, stats);
            Ok(TLExpr::ForAll {
                var,
                domain,
                body: Box::new(new_body),
            })
        }

        TLExpr::SoftExists {
            var,
            domain,
            body,
            temperature,
        } => {
            let inner_env = env.extend(var.clone(), PEValue::Symbolic(TLExpr::pred(&var, vec![])));
            let new_body = pe_rec(*body, &inner_env, config, depth + 1, stats);
            Ok(TLExpr::SoftExists {
                var,
                domain,
                body: Box::new(new_body),
                temperature,
            })
        }

        TLExpr::SoftForAll {
            var,
            domain,
            body,
            temperature,
        } => {
            let inner_env = env.extend(var.clone(), PEValue::Symbolic(TLExpr::pred(&var, vec![])));
            let new_body = pe_rec(*body, &inner_env, config, depth + 1, stats);
            Ok(TLExpr::SoftForAll {
                var,
                domain,
                body: Box::new(new_body),
                temperature,
            })
        }

        TLExpr::Aggregate {
            op,
            var,
            domain,
            body,
            group_by,
        } => {
            let inner_env = env.extend(var.clone(), PEValue::Symbolic(TLExpr::pred(&var, vec![])));
            let new_body = pe_rec(*body, &inner_env, config, depth + 1, stats);
            Ok(TLExpr::Aggregate {
                op,
                var,
                domain,
                body: Box::new(new_body),
                group_by,
            })
        }

        TLExpr::CountingExists {
            var,
            domain,
            body,
            min_count,
        } => {
            let inner_env = env.extend(var.clone(), PEValue::Symbolic(TLExpr::pred(&var, vec![])));
            let new_body = pe_rec(*body, &inner_env, config, depth + 1, stats);
            Ok(TLExpr::CountingExists {
                var,
                domain,
                body: Box::new(new_body),
                min_count,
            })
        }

        TLExpr::CountingForAll {
            var,
            domain,
            body,
            min_count,
        } => {
            let inner_env = env.extend(var.clone(), PEValue::Symbolic(TLExpr::pred(&var, vec![])));
            let new_body = pe_rec(*body, &inner_env, config, depth + 1, stats);
            Ok(TLExpr::CountingForAll {
                var,
                domain,
                body: Box::new(new_body),
                min_count,
            })
        }

        TLExpr::ExactCount {
            var,
            domain,
            body,
            count,
        } => {
            let inner_env = env.extend(var.clone(), PEValue::Symbolic(TLExpr::pred(&var, vec![])));
            let new_body = pe_rec(*body, &inner_env, config, depth + 1, stats);
            Ok(TLExpr::ExactCount {
                var,
                domain,
                body: Box::new(new_body),
                count,
            })
        }

        TLExpr::Majority { var, domain, body } => {
            let inner_env = env.extend(var.clone(), PEValue::Symbolic(TLExpr::pred(&var, vec![])));
            let new_body = pe_rec(*body, &inner_env, config, depth + 1, stats);
            Ok(TLExpr::Majority {
                var,
                domain,
                body: Box::new(new_body),
            })
        }

        TLExpr::SetComprehension {
            var,
            domain,
            condition,
        } => {
            let inner_env = env.extend(var.clone(), PEValue::Symbolic(TLExpr::pred(&var, vec![])));
            let new_cond = pe_rec(*condition, &inner_env, config, depth + 1, stats);
            Ok(TLExpr::SetComprehension {
                var,
                domain,
                condition: Box::new(new_cond),
            })
        }

        TLExpr::Lambda {
            var,
            var_type,
            body,
        } => {
            // Lambda binds var — shadow it in env
            let inner_env = env.extend(var.clone(), PEValue::Symbolic(TLExpr::pred(&var, vec![])));
            let new_body = pe_rec(*body, &inner_env, config, depth + 1, stats);
            Ok(TLExpr::Lambda {
                var,
                var_type,
                body: Box::new(new_body),
            })
        }

        other => Err(other),
    }
}
