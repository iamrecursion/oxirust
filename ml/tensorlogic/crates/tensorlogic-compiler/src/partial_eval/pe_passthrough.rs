//! Partial-evaluation arms for nodes whose children are recursed into without
//! any local folding: modal, temporal, fuzzy, probabilistic, higher-order
//! application, set operators, fixed points, hybrid/modal operators,
//! `GlobalCardinality`, and closed leaves (`EmptySet`, `Nominal`,
//! `AllDifferent`, `Abducible`).

use tensorlogic_ir::TLExpr;

use super::pe_core::pe_rec;
use super::types::{PEConfig, PEEnv, PEStats};

/// Attempt to partially evaluate a passthrough node (recurse into children;
/// no folding). Returns `Ok(result)` when handled; `Err(expr)` to let
/// [`pe_rec`] fall through to its final unchanged return.
pub(super) fn try_pe_passthrough(
    expr: TLExpr,
    env: &PEEnv,
    config: &PEConfig,
    depth: usize,
    stats: &mut PEStats,
) -> Result<TLExpr, TLExpr> {
    match expr {
        // ── Score, modal, temporal operators ─────────────────────────────
        TLExpr::Score(inner) => {
            let e = pe_rec(*inner, env, config, depth + 1, stats);
            Ok(TLExpr::Score(Box::new(e)))
        }
        TLExpr::Box(inner) => {
            let e = pe_rec(*inner, env, config, depth + 1, stats);
            Ok(TLExpr::Box(Box::new(e)))
        }
        TLExpr::Diamond(inner) => {
            let e = pe_rec(*inner, env, config, depth + 1, stats);
            Ok(TLExpr::Diamond(Box::new(e)))
        }
        TLExpr::Next(inner) => {
            let e = pe_rec(*inner, env, config, depth + 1, stats);
            Ok(TLExpr::Next(Box::new(e)))
        }
        TLExpr::Eventually(inner) => {
            let e = pe_rec(*inner, env, config, depth + 1, stats);
            Ok(TLExpr::Eventually(Box::new(e)))
        }
        TLExpr::Always(inner) => {
            let e = pe_rec(*inner, env, config, depth + 1, stats);
            Ok(TLExpr::Always(Box::new(e)))
        }
        TLExpr::Until { before, after } => {
            let b = pe_rec(*before, env, config, depth + 1, stats);
            let a = pe_rec(*after, env, config, depth + 1, stats);
            Ok(TLExpr::Until {
                before: Box::new(b),
                after: Box::new(a),
            })
        }
        TLExpr::WeakUntil { before, after } => {
            let b = pe_rec(*before, env, config, depth + 1, stats);
            let a = pe_rec(*after, env, config, depth + 1, stats);
            Ok(TLExpr::WeakUntil {
                before: Box::new(b),
                after: Box::new(a),
            })
        }
        TLExpr::Release { released, releaser } => {
            let r1 = pe_rec(*released, env, config, depth + 1, stats);
            let r2 = pe_rec(*releaser, env, config, depth + 1, stats);
            Ok(TLExpr::Release {
                released: Box::new(r1),
                releaser: Box::new(r2),
            })
        }
        TLExpr::StrongRelease { released, releaser } => {
            let r1 = pe_rec(*released, env, config, depth + 1, stats);
            let r2 = pe_rec(*releaser, env, config, depth + 1, stats);
            Ok(TLExpr::StrongRelease {
                released: Box::new(r1),
                releaser: Box::new(r2),
            })
        }

        // ── Fuzzy logic operators ────────────────────────────────────────
        TLExpr::TNorm { kind, left, right } => {
            let l = pe_rec(*left, env, config, depth + 1, stats);
            let r = pe_rec(*right, env, config, depth + 1, stats);
            Ok(TLExpr::TNorm {
                kind,
                left: Box::new(l),
                right: Box::new(r),
            })
        }
        TLExpr::TCoNorm { kind, left, right } => {
            let l = pe_rec(*left, env, config, depth + 1, stats);
            let r = pe_rec(*right, env, config, depth + 1, stats);
            Ok(TLExpr::TCoNorm {
                kind,
                left: Box::new(l),
                right: Box::new(r),
            })
        }
        TLExpr::FuzzyNot { kind, expr: inner } => {
            let e = pe_rec(*inner, env, config, depth + 1, stats);
            Ok(TLExpr::FuzzyNot {
                kind,
                expr: Box::new(e),
            })
        }
        TLExpr::FuzzyImplication {
            kind,
            premise,
            conclusion,
        } => {
            let p = pe_rec(*premise, env, config, depth + 1, stats);
            let c = pe_rec(*conclusion, env, config, depth + 1, stats);
            Ok(TLExpr::FuzzyImplication {
                kind,
                premise: Box::new(p),
                conclusion: Box::new(c),
            })
        }

        // ── Probabilistic operators ──────────────────────────────────────
        TLExpr::WeightedRule { weight, rule } => {
            let r = pe_rec(*rule, env, config, depth + 1, stats);
            Ok(TLExpr::WeightedRule {
                weight,
                rule: Box::new(r),
            })
        }
        TLExpr::ProbabilisticChoice { alternatives } => {
            let new_alts = alternatives
                .into_iter()
                .map(|(prob, alt_expr)| {
                    let e = pe_rec(alt_expr, env, config, depth + 1, stats);
                    (prob, e)
                })
                .collect();
            Ok(TLExpr::ProbabilisticChoice {
                alternatives: new_alts,
            })
        }

        // ── Higher-order logic ───────────────────────────────────────────
        TLExpr::Apply { function, argument } => {
            let f = pe_rec(*function, env, config, depth + 1, stats);
            let a = pe_rec(*argument, env, config, depth + 1, stats);
            Ok(TLExpr::Apply {
                function: Box::new(f),
                argument: Box::new(a),
            })
        }

        // ── Set operations ───────────────────────────────────────────────
        TLExpr::SetMembership { element, set } => {
            let el = pe_rec(*element, env, config, depth + 1, stats);
            let st = pe_rec(*set, env, config, depth + 1, stats);
            Ok(TLExpr::SetMembership {
                element: Box::new(el),
                set: Box::new(st),
            })
        }
        TLExpr::SetUnion { left, right } => {
            let l = pe_rec(*left, env, config, depth + 1, stats);
            let r = pe_rec(*right, env, config, depth + 1, stats);
            Ok(TLExpr::SetUnion {
                left: Box::new(l),
                right: Box::new(r),
            })
        }
        TLExpr::SetIntersection { left, right } => {
            let l = pe_rec(*left, env, config, depth + 1, stats);
            let r = pe_rec(*right, env, config, depth + 1, stats);
            Ok(TLExpr::SetIntersection {
                left: Box::new(l),
                right: Box::new(r),
            })
        }
        TLExpr::SetDifference { left, right } => {
            let l = pe_rec(*left, env, config, depth + 1, stats);
            let r = pe_rec(*right, env, config, depth + 1, stats);
            Ok(TLExpr::SetDifference {
                left: Box::new(l),
                right: Box::new(r),
            })
        }
        TLExpr::SetCardinality { set } => {
            let s = pe_rec(*set, env, config, depth + 1, stats);
            Ok(TLExpr::SetCardinality { set: Box::new(s) })
        }

        // ── Fixed-point operators — body references the fixed-point var ──
        TLExpr::LeastFixpoint { var, body } => {
            let inner_env = env.extend(
                var.clone(),
                super::types::PEValue::Symbolic(TLExpr::pred(&var, vec![])),
            );
            let b = pe_rec(*body, &inner_env, config, depth + 1, stats);
            Ok(TLExpr::LeastFixpoint {
                var,
                body: Box::new(b),
            })
        }
        TLExpr::GreatestFixpoint { var, body } => {
            let inner_env = env.extend(
                var.clone(),
                super::types::PEValue::Symbolic(TLExpr::pred(&var, vec![])),
            );
            let b = pe_rec(*body, &inner_env, config, depth + 1, stats);
            Ok(TLExpr::GreatestFixpoint {
                var,
                body: Box::new(b),
            })
        }

        // ── Hybrid / modal ───────────────────────────────────────────────
        TLExpr::At { nominal, formula } => {
            let f = pe_rec(*formula, env, config, depth + 1, stats);
            Ok(TLExpr::At {
                nominal,
                formula: Box::new(f),
            })
        }
        TLExpr::Somewhere { formula } => {
            let f = pe_rec(*formula, env, config, depth + 1, stats);
            Ok(TLExpr::Somewhere {
                formula: Box::new(f),
            })
        }
        TLExpr::Everywhere { formula } => {
            let f = pe_rec(*formula, env, config, depth + 1, stats);
            Ok(TLExpr::Everywhere {
                formula: Box::new(f),
            })
        }
        TLExpr::Explain { formula } => {
            let f = pe_rec(*formula, env, config, depth + 1, stats);
            Ok(TLExpr::Explain {
                formula: Box::new(f),
            })
        }

        // ── GlobalCardinality — recurse into values ──────────────────────
        TLExpr::GlobalCardinality {
            variables,
            values,
            min_occurrences,
            max_occurrences,
        } => {
            let new_vals = values
                .into_iter()
                .map(|v| pe_rec(v, env, config, depth + 1, stats))
                .collect();
            Ok(TLExpr::GlobalCardinality {
                variables,
                values: new_vals,
                min_occurrences,
                max_occurrences,
            })
        }

        // ── Leaves with no sub-expressions ───────────────────────────────
        leaf @ (TLExpr::EmptySet
        | TLExpr::Nominal { .. }
        | TLExpr::AllDifferent { .. }
        | TLExpr::Abducible { .. }
        | TLExpr::SymbolLiteral(_)) => Ok(leaf),

        TLExpr::Match { scrutinee, arms } => {
            let new_scrutinee = pe_rec(*scrutinee, env, config, depth + 1, stats);
            let new_arms = arms
                .into_iter()
                .map(|(pat, body)| (pat, Box::new(pe_rec(*body, env, config, depth + 1, stats))))
                .collect();
            Ok(TLExpr::Match {
                scrutinee: Box::new(new_scrutinee),
                arms: new_arms,
            })
        }

        other => Err(other),
    }
}
