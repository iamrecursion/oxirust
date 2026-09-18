//! Core recursive dispatcher for DCE.
//!
//! Delegates to the per-category helpers in `eliminate_flow`, `eliminate_ops`,
//! and `eliminate_ext`.

use tensorlogic_ir::TLExpr;

use super::types::{DceStats, DeadCodeEliminator};

impl DeadCodeEliminator {
    /// Recursively apply DCE rules bottom-up: recurse into children first,
    /// then apply simplification rules at the current node.
    pub(super) fn eliminate(&self, expr: TLExpr, stats: &mut DceStats) -> (TLExpr, bool) {
        let expr = match self.elim_flow(expr, stats) {
            Ok(result) => return result,
            Err(unchanged) => unchanged,
        };

        let expr = match self.elim_ops(expr, stats) {
            Ok(result) => return result,
            Err(unchanged) => unchanged,
        };

        self.elim_ext(expr, stats)
    }
}
