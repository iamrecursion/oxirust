//! Statistics, configuration, and the `DeadCodeEliminator` type.

use tensorlogic_ir::TLExpr;

/// Statistics collected during a single or multi-pass DCE run.
#[derive(Debug, Clone, Default)]
pub struct DceStats {
    /// Number of constant-fold eliminations (And/Or/Not with constant operands).
    pub constant_folds: u64,
    /// Number of unreachable branches eliminated (IfThenElse with constant condition).
    pub unreachable_branches: u64,
    /// Number of unused Let bindings removed.
    pub unused_let_bindings: u64,
    /// Total number of expression nodes *before* the first pass.
    pub total_nodes_before: u64,
    /// Total number of expression nodes *after* the final pass.
    pub total_nodes_after: u64,
    /// Number of DCE passes that were executed.
    pub passes: u32,
}

impl DceStats {
    /// Fraction of nodes eliminated: `(before − after) / before`.
    ///
    /// Returns `0.0` when `total_nodes_before == 0`.
    pub fn reduction_ratio(&self) -> f64 {
        if self.total_nodes_before == 0 {
            return 0.0;
        }
        let before = self.total_nodes_before as f64;
        let after = self.total_nodes_after as f64;
        ((before - after) / before).max(0.0)
    }

    /// Sum of all elimination counts across all categories.
    pub fn total_eliminations(&self) -> u64 {
        self.constant_folds
            .saturating_add(self.unreachable_branches)
            .saturating_add(self.unused_let_bindings)
    }

    /// Human-readable one-line summary of the DCE run.
    pub fn summary(&self) -> String {
        format!(
            "DCE: {} passes, {}/{} nodes kept ({:.1}% reduction) — \
             {} constant folds, {} unreachable branches, {} unused lets",
            self.passes,
            self.total_nodes_after,
            self.total_nodes_before,
            self.reduction_ratio() * 100.0,
            self.constant_folds,
            self.unreachable_branches,
            self.unused_let_bindings,
        )
    }
}

/// Configuration controlling which DCE rules are active.
#[derive(Debug, Clone)]
pub struct DceConfig {
    /// Enable `And(False, x) → False`, `And(True, x) → x`, and symmetric variants.
    pub eliminate_constant_and: bool,
    /// Enable `Or(True, x) → True`, `Or(False, x) → x`, and symmetric variants.
    pub eliminate_constant_or: bool,
    /// Enable `Not(True) → False`, `Not(False) → True`.
    pub eliminate_constant_not: bool,
    /// Enable `IfThenElse(True, then, _) → then` and `IfThenElse(False, _, else) → else`.
    pub eliminate_if_branches: bool,
    /// Enable removal of `Let(x, e, body)` when `x` does not appear free in `body`.
    pub eliminate_unused_let: bool,
    /// Maximum number of passes before giving up (fixed-point iteration).
    pub max_passes: u32,
}

impl Default for DceConfig {
    fn default() -> Self {
        Self {
            eliminate_constant_and: true,
            eliminate_constant_or: true,
            eliminate_constant_not: true,
            eliminate_if_branches: true,
            eliminate_unused_let: true,
            max_passes: 20,
        }
    }
}

/// The dead code elimination pass for [`TLExpr`] trees.
///
/// Run with [`DeadCodeEliminator::run`] for a full fixed-point iteration,
/// or `run_pass` for a single traversal.
pub struct DeadCodeEliminator {
    pub(super) config: DceConfig,
}

impl Default for DeadCodeEliminator {
    fn default() -> Self {
        Self::with_default()
    }
}

impl DeadCodeEliminator {
    /// Create a new eliminator with the given configuration.
    pub fn new(config: DceConfig) -> Self {
        Self { config }
    }

    /// Create a new eliminator with default configuration.
    pub fn with_default() -> Self {
        Self::new(DceConfig::default())
    }

    /// Run DCE to a fixed point (or until `config.max_passes` is reached).
    ///
    /// Returns the simplified expression and collected [`DceStats`].
    pub fn run(&self, expr: TLExpr) -> (TLExpr, DceStats) {
        let mut stats = DceStats {
            total_nodes_before: Self::count_nodes(&expr),
            ..Default::default()
        };

        let mut current = expr;
        let max = self.config.max_passes.max(1);

        for _ in 0..max {
            let (next, changed) = self.run_pass(current, &mut stats);
            stats.passes += 1;
            current = next;
            if !changed {
                break;
            }
        }

        stats.total_nodes_after = Self::count_nodes(&current);
        (current, stats)
    }

    /// Execute one top-down/bottom-up DCE pass.
    ///
    /// Returns `(new_expr, did_change)`.
    fn run_pass(&self, expr: TLExpr, stats: &mut DceStats) -> (TLExpr, bool) {
        self.eliminate(expr, stats)
    }
}
