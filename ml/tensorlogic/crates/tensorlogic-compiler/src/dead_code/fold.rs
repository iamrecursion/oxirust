//! Constant-folding helpers applied by the DCE pass.

use tensorlogic_ir::TLExpr;

use super::consts::{is_false_const, is_true_const};
use super::types::{DceStats, DeadCodeEliminator};

impl DeadCodeEliminator {
    /// Constant-fold `And`:
    /// - `And(False, x) → False`
    /// - `And(x, False) → False`
    /// - `And(True, x) → x`
    /// - `And(x, True) → x`
    ///
    /// When no rule fires, reconstructs `And(lhs, rhs)` and returns `Some`.
    pub(super) fn fold_and(
        &self,
        lhs: TLExpr,
        rhs: TLExpr,
        stats: &mut DceStats,
    ) -> Option<TLExpr> {
        if is_false_const(&lhs) {
            stats.constant_folds += 1;
            return Some(TLExpr::Constant(0.0));
        }
        if is_false_const(&rhs) {
            stats.constant_folds += 1;
            return Some(TLExpr::Constant(0.0));
        }
        if is_true_const(&lhs) {
            stats.constant_folds += 1;
            return Some(rhs);
        }
        if is_true_const(&rhs) {
            stats.constant_folds += 1;
            return Some(lhs);
        }
        Some(TLExpr::And(Box::new(lhs), Box::new(rhs)))
    }

    /// Constant-fold `Or`:
    /// - `Or(True, x) → True`
    /// - `Or(x, True) → True`
    /// - `Or(False, x) → x`
    /// - `Or(x, False) → x`
    pub(super) fn fold_or(&self, lhs: TLExpr, rhs: TLExpr, stats: &mut DceStats) -> Option<TLExpr> {
        if is_true_const(&lhs) {
            stats.constant_folds += 1;
            return Some(TLExpr::Constant(1.0));
        }
        if is_true_const(&rhs) {
            stats.constant_folds += 1;
            return Some(TLExpr::Constant(1.0));
        }
        if is_false_const(&lhs) {
            stats.constant_folds += 1;
            return Some(rhs);
        }
        if is_false_const(&rhs) {
            stats.constant_folds += 1;
            return Some(lhs);
        }
        Some(TLExpr::Or(Box::new(lhs), Box::new(rhs)))
    }

    /// Constant-fold `Not`:
    /// - `Not(True) → False`
    /// - `Not(False) → True`
    pub(super) fn fold_not(&self, inner: TLExpr, stats: &mut DceStats) -> Option<TLExpr> {
        if is_true_const(&inner) {
            stats.constant_folds += 1;
            return Some(TLExpr::Constant(0.0));
        }
        if is_false_const(&inner) {
            stats.constant_folds += 1;
            return Some(TLExpr::Constant(1.0));
        }
        Some(TLExpr::Not(Box::new(inner)))
    }

    /// Simplify `IfThenElse`:
    /// - `IfThenElse(True, then, _)  → then`
    /// - `IfThenElse(False, _, else) → else`
    pub(super) fn fold_if(
        &self,
        cond: TLExpr,
        then: TLExpr,
        else_: TLExpr,
        stats: &mut DceStats,
    ) -> Option<TLExpr> {
        if is_true_const(&cond) {
            stats.unreachable_branches += 1;
            return Some(then);
        }
        if is_false_const(&cond) {
            stats.unreachable_branches += 1;
            return Some(else_);
        }
        Some(TLExpr::IfThenElse {
            condition: Box::new(cond),
            then_branch: Box::new(then),
            else_branch: Box::new(else_),
        })
    }
}
