//! Elimination arms for fuzzy, weighted/probabilistic, aggregation, higher-order,
//! set, counting, fixpoint, hybrid, abductive, and leaf nodes.

use tensorlogic_ir::TLExpr;

use super::types::{DceStats, DeadCodeEliminator};

impl DeadCodeEliminator {
    /// Handle the remaining extended-logic arms.  This is the terminal dispatch
    /// step; every `TLExpr` variant that is not handled here is a leaf.
    pub(super) fn elim_ext(&self, expr: TLExpr, stats: &mut DceStats) -> (TLExpr, bool) {
        match expr {
            TLExpr::TNorm { kind, left, right } => {
                let (nl, cl) = self.eliminate(*left, stats);
                let (nr, cr) = self.eliminate(*right, stats);
                (
                    TLExpr::TNorm {
                        kind,
                        left: Box::new(nl),
                        right: Box::new(nr),
                    },
                    cl || cr,
                )
            }
            TLExpr::TCoNorm { kind, left, right } => {
                let (nl, cl) = self.eliminate(*left, stats);
                let (nr, cr) = self.eliminate(*right, stats);
                (
                    TLExpr::TCoNorm {
                        kind,
                        left: Box::new(nl),
                        right: Box::new(nr),
                    },
                    cl || cr,
                )
            }
            TLExpr::FuzzyNot { kind, expr } => {
                let (ne, changed) = self.eliminate(*expr, stats);
                (
                    TLExpr::FuzzyNot {
                        kind,
                        expr: Box::new(ne),
                    },
                    changed,
                )
            }
            TLExpr::FuzzyImplication {
                kind,
                premise,
                conclusion,
            } => {
                let (np, cp) = self.eliminate(*premise, stats);
                let (nc, cc) = self.eliminate(*conclusion, stats);
                (
                    TLExpr::FuzzyImplication {
                        kind,
                        premise: Box::new(np),
                        conclusion: Box::new(nc),
                    },
                    cp || cc,
                )
            }

            TLExpr::WeightedRule { weight, rule } => {
                let (nr, changed) = self.eliminate(*rule, stats);
                (
                    TLExpr::WeightedRule {
                        weight,
                        rule: Box::new(nr),
                    },
                    changed,
                )
            }
            TLExpr::ProbabilisticChoice { alternatives } => {
                let mut any_changed = false;
                let new_alts: Vec<(f64, TLExpr)> = alternatives
                    .into_iter()
                    .map(|(prob, e)| {
                        let (ne, changed) = self.eliminate(e, stats);
                        any_changed = any_changed || changed;
                        (prob, ne)
                    })
                    .collect();
                (
                    TLExpr::ProbabilisticChoice {
                        alternatives: new_alts,
                    },
                    any_changed,
                )
            }

            TLExpr::Aggregate {
                op,
                var,
                domain,
                body,
                group_by,
            } => {
                let (new_body, changed) = self.eliminate(*body, stats);
                (
                    TLExpr::Aggregate {
                        op,
                        var,
                        domain,
                        body: Box::new(new_body),
                        group_by,
                    },
                    changed,
                )
            }

            TLExpr::Lambda {
                var,
                var_type,
                body,
            } => {
                let (new_body, changed) = self.eliminate(*body, stats);
                (
                    TLExpr::Lambda {
                        var,
                        var_type,
                        body: Box::new(new_body),
                    },
                    changed,
                )
            }
            TLExpr::Apply { function, argument } => {
                let (nf, cf) = self.eliminate(*function, stats);
                let (na, ca) = self.eliminate(*argument, stats);
                (
                    TLExpr::Apply {
                        function: Box::new(nf),
                        argument: Box::new(na),
                    },
                    cf || ca,
                )
            }

            TLExpr::SetMembership { element, set } => {
                let (ne, ce) = self.eliminate(*element, stats);
                let (ns, cs) = self.eliminate(*set, stats);
                (
                    TLExpr::SetMembership {
                        element: Box::new(ne),
                        set: Box::new(ns),
                    },
                    ce || cs,
                )
            }
            TLExpr::SetUnion { left, right } => {
                let (nl, cl) = self.eliminate(*left, stats);
                let (nr, cr) = self.eliminate(*right, stats);
                (
                    TLExpr::SetUnion {
                        left: Box::new(nl),
                        right: Box::new(nr),
                    },
                    cl || cr,
                )
            }
            TLExpr::SetIntersection { left, right } => {
                let (nl, cl) = self.eliminate(*left, stats);
                let (nr, cr) = self.eliminate(*right, stats);
                (
                    TLExpr::SetIntersection {
                        left: Box::new(nl),
                        right: Box::new(nr),
                    },
                    cl || cr,
                )
            }
            TLExpr::SetDifference { left, right } => {
                let (nl, cl) = self.eliminate(*left, stats);
                let (nr, cr) = self.eliminate(*right, stats);
                (
                    TLExpr::SetDifference {
                        left: Box::new(nl),
                        right: Box::new(nr),
                    },
                    cl || cr,
                )
            }
            TLExpr::SetCardinality { set } => {
                let (ns, changed) = self.eliminate(*set, stats);
                (TLExpr::SetCardinality { set: Box::new(ns) }, changed)
            }
            TLExpr::SetComprehension {
                var,
                domain,
                condition,
            } => {
                let (nc, changed) = self.eliminate(*condition, stats);
                (
                    TLExpr::SetComprehension {
                        var,
                        domain,
                        condition: Box::new(nc),
                    },
                    changed,
                )
            }

            TLExpr::CountingExists {
                var,
                domain,
                body,
                min_count,
            } => {
                let (new_body, changed) = self.eliminate(*body, stats);
                (
                    TLExpr::CountingExists {
                        var,
                        domain,
                        body: Box::new(new_body),
                        min_count,
                    },
                    changed,
                )
            }
            TLExpr::CountingForAll {
                var,
                domain,
                body,
                min_count,
            } => {
                let (new_body, changed) = self.eliminate(*body, stats);
                (
                    TLExpr::CountingForAll {
                        var,
                        domain,
                        body: Box::new(new_body),
                        min_count,
                    },
                    changed,
                )
            }
            TLExpr::ExactCount {
                var,
                domain,
                body,
                count,
            } => {
                let (new_body, changed) = self.eliminate(*body, stats);
                (
                    TLExpr::ExactCount {
                        var,
                        domain,
                        body: Box::new(new_body),
                        count,
                    },
                    changed,
                )
            }
            TLExpr::Majority { var, domain, body } => {
                let (new_body, changed) = self.eliminate(*body, stats);
                (
                    TLExpr::Majority {
                        var,
                        domain,
                        body: Box::new(new_body),
                    },
                    changed,
                )
            }

            TLExpr::LeastFixpoint { var, body } => {
                let (new_body, changed) = self.eliminate(*body, stats);
                (
                    TLExpr::LeastFixpoint {
                        var,
                        body: Box::new(new_body),
                    },
                    changed,
                )
            }
            TLExpr::GreatestFixpoint { var, body } => {
                let (new_body, changed) = self.eliminate(*body, stats);
                (
                    TLExpr::GreatestFixpoint {
                        var,
                        body: Box::new(new_body),
                    },
                    changed,
                )
            }

            TLExpr::At { nominal, formula } => {
                let (nf, changed) = self.eliminate(*formula, stats);
                (
                    TLExpr::At {
                        nominal,
                        formula: Box::new(nf),
                    },
                    changed,
                )
            }
            TLExpr::Somewhere { formula } => {
                let (nf, changed) = self.eliminate(*formula, stats);
                (
                    TLExpr::Somewhere {
                        formula: Box::new(nf),
                    },
                    changed,
                )
            }
            TLExpr::Everywhere { formula } => {
                let (nf, changed) = self.eliminate(*formula, stats);
                (
                    TLExpr::Everywhere {
                        formula: Box::new(nf),
                    },
                    changed,
                )
            }

            TLExpr::Explain { formula } => {
                let (nf, changed) = self.eliminate(*formula, stats);
                (
                    TLExpr::Explain {
                        formula: Box::new(nf),
                    },
                    changed,
                )
            }

            // Leaves / terminal nodes (no children to recurse into)
            leaf @ (TLExpr::Pred { .. }
            | TLExpr::Constant(_)
            | TLExpr::EmptySet
            | TLExpr::AllDifferent { .. }
            | TLExpr::GlobalCardinality { .. }
            | TLExpr::Nominal { .. }
            | TLExpr::Abducible { .. }) => (leaf, false),

            TLExpr::SymbolLiteral(_) => (expr, false),

            TLExpr::Match { scrutinee, arms } => {
                let (new_scrutinee, sc) = self.eliminate(*scrutinee, stats);
                let mut any_changed = sc;
                let new_arms = arms
                    .into_iter()
                    .map(|(pat, body)| {
                        let (new_body, bc) = self.eliminate(*body, stats);
                        if bc {
                            any_changed = true;
                        }
                        (pat, Box::new(new_body))
                    })
                    .collect();
                (
                    TLExpr::Match {
                        scrutinee: Box::new(new_scrutinee),
                        arms: new_arms,
                    },
                    any_changed,
                )
            }

            // Any remaining variant should already have been handled by
            // `elim_flow` or `elim_ops` upstream — treat as an unchanged leaf
            // so the match stays exhaustive and future TLExpr additions do not
            // silently vanish.
            other => (other, false),
        }
    }
}
