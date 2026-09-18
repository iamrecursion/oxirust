//! Elimination arms for Boolean connectives, IfThenElse, Let, and Implication.
//!
//! These arms use the folding helpers that can constant-fold structural forms.

use tensorlogic_ir::TLExpr;

use super::types::{DceStats, DeadCodeEliminator};

impl DeadCodeEliminator {
    /// Handle the control-flow-shaped arms: And, Or, Not, IfThenElse, Let, Imply.
    ///
    /// Returns `Ok((new_expr, changed))` if this category handled the node, or
    /// `Err(expr)` to pass the unchanged expression to the next category.
    pub(super) fn elim_flow(
        &self,
        expr: TLExpr,
        stats: &mut DceStats,
    ) -> Result<(TLExpr, bool), TLExpr> {
        match expr {
            TLExpr::And(lhs, rhs) => {
                let (new_lhs, cl) = self.eliminate(*lhs, stats);
                let (new_rhs, cr) = self.eliminate(*rhs, stats);
                let child_changed = cl || cr;

                if self.config.eliminate_constant_and {
                    let before_folds = stats.constant_folds;
                    let result = self
                        .fold_and(new_lhs, new_rhs, stats)
                        .unwrap_or(TLExpr::Constant(0.0));
                    let did_fold = stats.constant_folds > before_folds;
                    Ok((result, child_changed || did_fold))
                } else {
                    Ok((
                        TLExpr::And(Box::new(new_lhs), Box::new(new_rhs)),
                        child_changed,
                    ))
                }
            }

            TLExpr::Or(lhs, rhs) => {
                let (new_lhs, cl) = self.eliminate(*lhs, stats);
                let (new_rhs, cr) = self.eliminate(*rhs, stats);
                let child_changed = cl || cr;

                if self.config.eliminate_constant_or {
                    let before_folds = stats.constant_folds;
                    let result = self
                        .fold_or(new_lhs, new_rhs, stats)
                        .unwrap_or(TLExpr::Constant(0.0));
                    let did_fold = stats.constant_folds > before_folds;
                    Ok((result, child_changed || did_fold))
                } else {
                    Ok((
                        TLExpr::Or(Box::new(new_lhs), Box::new(new_rhs)),
                        child_changed,
                    ))
                }
            }

            TLExpr::Not(inner) => {
                let (new_inner, child_changed) = self.eliminate(*inner, stats);

                if self.config.eliminate_constant_not {
                    let before_folds = stats.constant_folds;
                    let result = self
                        .fold_not(new_inner, stats)
                        .unwrap_or(TLExpr::Constant(0.0));
                    let did_fold = stats.constant_folds > before_folds;
                    Ok((result, child_changed || did_fold))
                } else {
                    Ok((TLExpr::Not(Box::new(new_inner)), child_changed))
                }
            }

            TLExpr::IfThenElse {
                condition,
                then_branch,
                else_branch,
            } => {
                let (new_cond, cc) = self.eliminate(*condition, stats);
                let (new_then, ct) = self.eliminate(*then_branch, stats);
                let (new_else, ce) = self.eliminate(*else_branch, stats);
                let child_changed = cc || ct || ce;

                if self.config.eliminate_if_branches {
                    let before_branches = stats.unreachable_branches;
                    let result = self
                        .fold_if(new_cond, new_then, new_else, stats)
                        .unwrap_or(TLExpr::Constant(0.0));
                    let did_fold = stats.unreachable_branches > before_branches;
                    Ok((result, child_changed || did_fold))
                } else {
                    Ok((
                        TLExpr::IfThenElse {
                            condition: Box::new(new_cond),
                            then_branch: Box::new(new_then),
                            else_branch: Box::new(new_else),
                        },
                        child_changed,
                    ))
                }
            }

            TLExpr::Let { var, value, body } => {
                let (new_value, cv) = self.eliminate(*value, stats);
                let (new_body, cb) = self.eliminate(*body, stats);
                let child_changed = cv || cb;

                if self.config.eliminate_unused_let && !self.is_free(&var, &new_body) {
                    stats.unused_let_bindings += 1;
                    return Ok((new_body, true));
                }

                Ok((
                    TLExpr::Let {
                        var,
                        value: Box::new(new_value),
                        body: Box::new(new_body),
                    },
                    child_changed,
                ))
            }

            TLExpr::Imply(premise, conclusion) => {
                let (new_p, cp) = self.eliminate(*premise, stats);
                let (new_c, cc) = self.eliminate(*conclusion, stats);
                Ok((TLExpr::Imply(Box::new(new_p), Box::new(new_c)), cp || cc))
            }

            other => Err(other),
        }
    }
}
