//! Whole-tree node counting plus small recursion helpers.

use tensorlogic_ir::TLExpr;

use super::types::{DceStats, DeadCodeEliminator};

impl DeadCodeEliminator {
    /// Count the total number of nodes in an expression tree.
    ///
    /// Every node (leaf or internal) counts as 1.
    pub fn count_nodes(expr: &TLExpr) -> u64 {
        match expr {
            TLExpr::And(l, r)
            | TLExpr::Or(l, r)
            | TLExpr::Imply(l, r)
            | TLExpr::Add(l, r)
            | TLExpr::Sub(l, r)
            | TLExpr::Mul(l, r)
            | TLExpr::Div(l, r)
            | TLExpr::Pow(l, r)
            | TLExpr::Mod(l, r)
            | TLExpr::Min(l, r)
            | TLExpr::Max(l, r)
            | TLExpr::Eq(l, r)
            | TLExpr::Lt(l, r)
            | TLExpr::Gt(l, r)
            | TLExpr::Lte(l, r)
            | TLExpr::Gte(l, r) => 1 + Self::count_nodes(l) + Self::count_nodes(r),

            TLExpr::TNorm { left, right, .. } | TLExpr::TCoNorm { left, right, .. } => {
                1 + Self::count_nodes(left) + Self::count_nodes(right)
            }
            TLExpr::FuzzyImplication {
                premise,
                conclusion,
                ..
            } => 1 + Self::count_nodes(premise) + Self::count_nodes(conclusion),

            TLExpr::Not(e)
            | TLExpr::Score(e)
            | TLExpr::Abs(e)
            | TLExpr::Floor(e)
            | TLExpr::Ceil(e)
            | TLExpr::Round(e)
            | TLExpr::Sqrt(e)
            | TLExpr::Exp(e)
            | TLExpr::Log(e)
            | TLExpr::Sin(e)
            | TLExpr::Cos(e)
            | TLExpr::Tan(e)
            | TLExpr::Box(e)
            | TLExpr::Diamond(e)
            | TLExpr::Next(e)
            | TLExpr::Eventually(e)
            | TLExpr::Always(e) => 1 + Self::count_nodes(e),

            TLExpr::FuzzyNot { expr, .. } => 1 + Self::count_nodes(expr),
            TLExpr::WeightedRule { rule, .. } => 1 + Self::count_nodes(rule),

            TLExpr::Until { before, after }
            | TLExpr::Release {
                released: before,
                releaser: after,
            }
            | TLExpr::WeakUntil { before, after }
            | TLExpr::StrongRelease {
                released: before,
                releaser: after,
            } => 1 + Self::count_nodes(before) + Self::count_nodes(after),

            TLExpr::IfThenElse {
                condition,
                then_branch,
                else_branch,
            } => {
                1 + Self::count_nodes(condition)
                    + Self::count_nodes(then_branch)
                    + Self::count_nodes(else_branch)
            }

            TLExpr::Exists { body, .. }
            | TLExpr::ForAll { body, .. }
            | TLExpr::SoftExists { body, .. }
            | TLExpr::SoftForAll { body, .. }
            | TLExpr::Aggregate { body, .. }
            | TLExpr::Lambda { body, .. }
            | TLExpr::SetComprehension {
                condition: body, ..
            }
            | TLExpr::CountingExists { body, .. }
            | TLExpr::CountingForAll { body, .. }
            | TLExpr::ExactCount { body, .. }
            | TLExpr::Majority { body, .. }
            | TLExpr::LeastFixpoint { body, .. }
            | TLExpr::GreatestFixpoint { body, .. } => 1 + Self::count_nodes(body),

            TLExpr::Let { value, body, .. } => {
                1 + Self::count_nodes(value) + Self::count_nodes(body)
            }

            TLExpr::Apply { function, argument } => {
                1 + Self::count_nodes(function) + Self::count_nodes(argument)
            }

            TLExpr::SetMembership { element, set }
            | TLExpr::SetUnion {
                left: element,
                right: set,
            }
            | TLExpr::SetIntersection {
                left: element,
                right: set,
            }
            | TLExpr::SetDifference {
                left: element,
                right: set,
            } => 1 + Self::count_nodes(element) + Self::count_nodes(set),

            TLExpr::SetCardinality { set } => 1 + Self::count_nodes(set),

            TLExpr::At { formula, .. }
            | TLExpr::Somewhere { formula }
            | TLExpr::Everywhere { formula }
            | TLExpr::Explain { formula } => 1 + Self::count_nodes(formula),

            TLExpr::ProbabilisticChoice { alternatives } => {
                1 + alternatives
                    .iter()
                    .map(|(_, e)| Self::count_nodes(e))
                    .sum::<u64>()
            }

            TLExpr::GlobalCardinality { values, .. } => {
                1 + values.iter().map(Self::count_nodes).sum::<u64>()
            }

            TLExpr::Pred { .. }
            | TLExpr::Constant(_)
            | TLExpr::EmptySet
            | TLExpr::AllDifferent { .. }
            | TLExpr::Nominal { .. }
            | TLExpr::Abducible { .. }
            | TLExpr::SymbolLiteral(_) => 1,

            TLExpr::Match { scrutinee, arms } => {
                1 + Self::count_nodes(scrutinee)
                    + arms.iter().map(|(_, b)| Self::count_nodes(b)).sum::<u64>()
            }
        }
    }

    /// Recurse into the single child of a unary constructor and reconstruct.
    pub(super) fn map_unary<F>(
        &self,
        ctor: F,
        child: TLExpr,
        stats: &mut DceStats,
    ) -> (TLExpr, bool)
    where
        F: Fn(Box<TLExpr>) -> TLExpr,
    {
        let (new_child, changed) = self.eliminate(child, stats);
        (ctor(Box::new(new_child)), changed)
    }

    /// Recurse into both children of a binary constructor and reconstruct.
    pub(super) fn map_binary<F>(
        &self,
        ctor: F,
        left: TLExpr,
        right: TLExpr,
        stats: &mut DceStats,
    ) -> (TLExpr, bool)
    where
        F: Fn(Box<TLExpr>, Box<TLExpr>) -> TLExpr,
    {
        let (nl, cl) = self.eliminate(left, stats);
        let (nr, cr) = self.eliminate(right, stats);
        (ctor(Box::new(nl), Box::new(nr)), cl || cr)
    }
}
