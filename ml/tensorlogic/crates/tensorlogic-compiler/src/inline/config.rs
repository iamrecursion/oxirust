/// Configuration for the let-inlining pass.
///
/// All flags default to `true` so that the pass is aggressive by default.
/// Disable individual flags for conservative inlining or debugging.
#[derive(Debug, Clone)]
pub struct InlineConfig {
    /// Inline `Let(x, e, body)` whenever `x` occurs free exactly once in
    /// `body`, regardless of how complex `e` is (subject to `max_inline_depth`).
    pub inline_single_use: bool,

    /// Inline `Let(x, Constant(_), body)` regardless of how many times `x`
    /// is used; constant duplication is essentially free.
    pub inline_constants: bool,

    /// Inline `Let(x, Pred(y, []), body)` — i.e. variable aliases — regardless
    /// of use count; these are pure renames.
    pub inline_vars: bool,

    /// Maximum number of fixed-point passes before stopping.
    pub max_passes: u32,

    /// Do not inline a binding whose value expression has depth greater than
    /// this threshold.  Prevents unbounded code-size growth.
    pub max_inline_depth: usize,
}

impl Default for InlineConfig {
    fn default() -> Self {
        Self {
            inline_single_use: true,
            inline_constants: true,
            inline_vars: true,
            max_passes: 20,
            max_inline_depth: 10,
        }
    }
}

/// Statistics collected by the let-inlining pass.
#[derive(Debug, Clone, Default)]
pub struct InlineStats {
    /// Number of bindings inlined because the variable had exactly one free
    /// occurrence in the body (and the value was not a constant or alias).
    pub single_use_inlines: u64,

    /// Number of bindings inlined because the value was a constant literal.
    pub constant_inlines: u64,

    /// Number of bindings inlined because the value was a variable alias
    /// (zero-argument predicate).
    pub variable_inlines: u64,

    /// Total node count before the first pass.
    pub nodes_before: u64,

    /// Total node count after the final pass.
    pub nodes_after: u64,

    /// Number of passes executed.
    pub passes: u32,
}

impl InlineStats {
    /// Sum of all inlining categories.
    pub fn total(&self) -> u64 {
        self.single_use_inlines
            .saturating_add(self.constant_inlines)
            .saturating_add(self.variable_inlines)
    }

    /// Fraction of nodes removed: `(before − after) / before`.
    ///
    /// Returns `0.0` when `nodes_before == 0`.
    pub fn reduction_pct(&self) -> f64 {
        if self.nodes_before == 0 {
            return 0.0;
        }
        let before = self.nodes_before as f64;
        let after = self.nodes_after as f64;
        ((before - after) / before * 100.0).max(0.0)
    }

    /// Human-readable one-line summary.
    pub fn summary(&self) -> String {
        format!(
            "Inline: {} passes, {}/{} nodes kept ({:.1}% reduction) — \
             {} single-use, {} constant, {} variable-alias inlines",
            self.passes,
            self.nodes_after,
            self.nodes_before,
            self.reduction_pct(),
            self.single_use_inlines,
            self.constant_inlines,
            self.variable_inlines,
        )
    }
}
