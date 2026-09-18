//! Constant Propagation pass for TLExpr trees.
//!
//! This pass performs **compile-time evaluation** of subexpressions whose operands
//! are all `TLExpr::Constant(f64)` values. It complements the [`crate::dead_code`]
//! pass (which handles structural boolean short-circuiting) by focusing on numeric
//! arithmetic, comparison folding, and unary math folding.
//!
//! # Boolean Constants Convention
//!
//! Consistent with the rest of the codebase:
//! - `TLExpr::Constant(1.0)` represents logical **True**
//! - `TLExpr::Constant(0.0)` represents logical **False**
//!
//! Comparison operations that evaluate to a definite truth value produce one of
//! these two constants.
//!
//! # Example
//!
//! ```rust
//! use tensorlogic_compiler::const_prop::{ConstantPropagator, ConstPropConfig};
//! use tensorlogic_ir::TLExpr;
//!
//! let propagator = ConstantPropagator::with_default();
//! // Add(Mul(2, 3), 4) → 10
//! let expr = TLExpr::add(
//!     TLExpr::mul(TLExpr::Constant(2.0), TLExpr::Constant(3.0)),
//!     TLExpr::Constant(4.0),
//! );
//! let (result, stats) = propagator.run(expr);
//! assert!(matches!(result, TLExpr::Constant(v) if (v - 10.0).abs() < 1e-12));
//! assert!(stats.arithmetic_folds >= 2);
//! ```

use tensorlogic_ir::TLExpr;

// ────────────────────────────────────────────────────────────────
// Statistics
// ────────────────────────────────────────────────────────────────

/// Statistics collected during a constant propagation run.
#[derive(Debug, Clone, Default)]
pub struct ConstPropStats {
    /// Number of arithmetic binary folding operations
    /// (e.g. `Add(1,2) → 3`, `Mul(2,3) → 6`).
    pub arithmetic_folds: u64,
    /// Number of comparison folding operations
    /// (e.g. `Lt(1,2) → True`, `Eq(3,3) → True`).
    pub comparison_folds: u64,
    /// Number of boolean/unary constant folding operations
    /// (e.g. `Not(True) → False`, `Abs(-3) → 3`).
    pub boolean_folds: u64,
    /// Total expression nodes counted before the first pass.
    pub nodes_before: u64,
    /// Total expression nodes counted after the last pass.
    pub nodes_after: u64,
    /// Number of passes executed.
    pub passes: u32,
}

impl ConstPropStats {
    /// Total folds across all categories.
    pub fn total_folds(&self) -> u64 {
        self.arithmetic_folds
            .saturating_add(self.comparison_folds)
            .saturating_add(self.boolean_folds)
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
        (((before - after) / before) * 100.0).max(0.0)
    }

    /// Human-readable one-line summary.
    pub fn summary(&self) -> String {
        format!(
            "ConstProp: {} passes, {}/{} nodes kept ({:.1}% reduction) — \
             {} arith folds, {} cmp folds, {} bool folds",
            self.passes,
            self.nodes_after,
            self.nodes_before,
            self.reduction_pct(),
            self.arithmetic_folds,
            self.comparison_folds,
            self.boolean_folds,
        )
    }
}

// ────────────────────────────────────────────────────────────────
// Configuration
// ────────────────────────────────────────────────────────────────

/// Configuration for the constant propagation pass.
#[derive(Debug, Clone)]
pub struct ConstPropConfig {
    /// Fold arithmetic binary operations when both operands are constants.
    pub fold_arithmetic: bool,
    /// Fold comparison operations when both operands are constants.
    pub fold_comparisons: bool,
    /// Fold unary operations (Abs, Floor, Ceil, Round, Sqrt, Exp, Log, Sin, Cos, Tan, Not) on constants.
    pub fold_boolean: bool,
    /// Maximum number of convergence passes to perform.
    pub max_passes: u32,
    /// Absolute tolerance used when comparing floats for equality (`Eq` comparison).
    pub float_tolerance: f64,
}

impl Default for ConstPropConfig {
    fn default() -> Self {
        Self {
            fold_arithmetic: true,
            fold_comparisons: true,
            fold_boolean: true,
            max_passes: 20,
            float_tolerance: 1e-12,
        }
    }
}

// ────────────────────────────────────────────────────────────────
// ConstantPropagator
// ────────────────────────────────────────────────────────────────

/// The constant propagation compiler pass.
///
/// Performs a bottom-up sweep over a [`TLExpr`] tree, evaluating subexpressions
/// at compile time when all operands are `TLExpr::Constant(f64)` values.
/// Runs to a fixed point (i.e. repeated until no further changes occur) or
/// until `config.max_passes` is reached.
pub struct ConstantPropagator {
    config: ConstPropConfig,
}

impl ConstantPropagator {
    /// Create a new propagator with the supplied configuration.
    pub fn new(config: ConstPropConfig) -> Self {
        Self { config }
    }

    /// Create a new propagator with default configuration.
    pub fn with_default() -> Self {
        Self::new(ConstPropConfig::default())
    }

    /// Run constant propagation to a fixed point.
    ///
    /// Returns `(simplified_expr, stats)`.
    pub fn run(&self, expr: TLExpr) -> (TLExpr, ConstPropStats) {
        let mut stats = ConstPropStats {
            nodes_before: Self::count_nodes(&expr),
            ..Default::default()
        };

        let mut current = expr;
        let mut pass_count = 0u32;

        loop {
            if pass_count >= self.config.max_passes {
                break;
            }
            let (next, changed) = self.run_pass(current, &mut stats);
            pass_count = pass_count.saturating_add(1);
            current = next;
            if !changed {
                break;
            }
        }

        stats.passes = pass_count;
        stats.nodes_after = Self::count_nodes(&current);
        (current, stats)
    }

    /// Execute one propagation pass over the entire tree.
    ///
    /// Returns `(new_expr, changed)` where `changed` indicates whether any
    /// fold occurred during this pass.
    fn run_pass(&self, expr: TLExpr, stats: &mut ConstPropStats) -> (TLExpr, bool) {
        self.propagate(expr, stats)
    }

    /// Recursive bottom-up propagation.
    ///
    /// First recurse into children; then attempt to fold the current node.
    /// Returns `(new_expr, changed)`.
    fn propagate(&self, expr: TLExpr, stats: &mut ConstPropStats) -> (TLExpr, bool) {
        match expr {
            // ── Leaf nodes — nothing to fold ──────────────────────────────
            TLExpr::Constant(_)
            | TLExpr::Pred { .. }
            | TLExpr::EmptySet
            | TLExpr::AllDifferent { .. }
            | TLExpr::Nominal { .. }
            | TLExpr::Abducible { .. } => (expr, false),

            // ── Arithmetic binary ops ─────────────────────────────────────
            TLExpr::Add(lhs, rhs) => self.fold_binary_arith("Add", *lhs, *rhs, stats, TLExpr::Add),
            TLExpr::Sub(lhs, rhs) => self.fold_binary_arith("Sub", *lhs, *rhs, stats, TLExpr::Sub),
            TLExpr::Mul(lhs, rhs) => self.fold_binary_arith("Mul", *lhs, *rhs, stats, TLExpr::Mul),
            TLExpr::Div(lhs, rhs) => self.fold_binary_arith("Div", *lhs, *rhs, stats, TLExpr::Div),
            TLExpr::Pow(lhs, rhs) => self.fold_binary_arith("Pow", *lhs, *rhs, stats, TLExpr::Pow),
            TLExpr::Mod(lhs, rhs) => self.fold_binary_arith("Mod", *lhs, *rhs, stats, TLExpr::Mod),
            TLExpr::Min(lhs, rhs) => self.fold_binary_arith("Min", *lhs, *rhs, stats, TLExpr::Min),
            TLExpr::Max(lhs, rhs) => self.fold_binary_arith("Max", *lhs, *rhs, stats, TLExpr::Max),

            // ── Comparison ops ────────────────────────────────────────────
            TLExpr::Eq(lhs, rhs) => self.fold_binary_cmp("Eq", *lhs, *rhs, stats, TLExpr::Eq),
            TLExpr::Lt(lhs, rhs) => self.fold_binary_cmp("Lt", *lhs, *rhs, stats, TLExpr::Lt),
            TLExpr::Gt(lhs, rhs) => self.fold_binary_cmp("Gt", *lhs, *rhs, stats, TLExpr::Gt),
            TLExpr::Lte(lhs, rhs) => self.fold_binary_cmp("Lte", *lhs, *rhs, stats, TLExpr::Lte),
            TLExpr::Gte(lhs, rhs) => self.fold_binary_cmp("Gte", *lhs, *rhs, stats, TLExpr::Gte),

            // ── Unary math ops ────────────────────────────────────────────
            TLExpr::Abs(inner) => self.fold_unary_math("Abs", *inner, stats, TLExpr::Abs),
            TLExpr::Floor(inner) => self.fold_unary_math("Floor", *inner, stats, TLExpr::Floor),
            TLExpr::Ceil(inner) => self.fold_unary_math("Ceil", *inner, stats, TLExpr::Ceil),
            TLExpr::Round(inner) => self.fold_unary_math("Round", *inner, stats, TLExpr::Round),
            TLExpr::Sqrt(inner) => self.fold_unary_math("Sqrt", *inner, stats, TLExpr::Sqrt),
            TLExpr::Exp(inner) => self.fold_unary_math("Exp", *inner, stats, TLExpr::Exp),
            TLExpr::Log(inner) => self.fold_unary_math("Log", *inner, stats, TLExpr::Log),
            TLExpr::Sin(inner) => self.fold_unary_math("Sin", *inner, stats, TLExpr::Sin),
            TLExpr::Cos(inner) => self.fold_unary_math("Cos", *inner, stats, TLExpr::Cos),
            TLExpr::Tan(inner) => self.fold_unary_math("Tan", *inner, stats, TLExpr::Tan),

            // ── Boolean / logical unary ───────────────────────────────────
            TLExpr::Not(inner) => {
                let (new_inner, child_changed) = self.propagate(*inner, stats);
                if self.config.fold_boolean {
                    if let Some(v) = Self::as_constant(&new_inner) {
                        // Not(True) → False, Not(False) → True
                        // More generally, Not(x) → Constant(1 - x) when x is a constant
                        let result = TLExpr::Constant(1.0 - v);
                        stats.boolean_folds = stats.boolean_folds.saturating_add(1);
                        return (result, true);
                    }
                }
                (TLExpr::Not(Box::new(new_inner)), child_changed)
            }

            // ── Boolean binary ops ────────────────────────────────────────
            TLExpr::And(lhs, rhs) => {
                let (new_lhs, cl) = self.propagate(*lhs, stats);
                let (new_rhs, cr) = self.propagate(*rhs, stats);
                if self.config.fold_boolean {
                    if let (Some(a), Some(b)) =
                        (Self::as_constant(&new_lhs), Self::as_constant(&new_rhs))
                    {
                        // Treat both as booleans (non-zero = true)
                        let result = if a != 0.0 && b != 0.0 { 1.0 } else { 0.0 };
                        stats.boolean_folds = stats.boolean_folds.saturating_add(1);
                        return (TLExpr::Constant(result), true);
                    }
                }
                (TLExpr::And(Box::new(new_lhs), Box::new(new_rhs)), cl || cr)
            }
            TLExpr::Or(lhs, rhs) => {
                let (new_lhs, cl) = self.propagate(*lhs, stats);
                let (new_rhs, cr) = self.propagate(*rhs, stats);
                if self.config.fold_boolean {
                    if let (Some(a), Some(b)) =
                        (Self::as_constant(&new_lhs), Self::as_constant(&new_rhs))
                    {
                        let result = if a != 0.0 || b != 0.0 { 1.0 } else { 0.0 };
                        stats.boolean_folds = stats.boolean_folds.saturating_add(1);
                        return (TLExpr::Constant(result), true);
                    }
                }
                (TLExpr::Or(Box::new(new_lhs), Box::new(new_rhs)), cl || cr)
            }
            TLExpr::Imply(premise, conclusion) => {
                let (new_p, cp) = self.propagate(*premise, stats);
                let (new_c, cc) = self.propagate(*conclusion, stats);
                if self.config.fold_boolean {
                    if let (Some(a), Some(b)) =
                        (Self::as_constant(&new_p), Self::as_constant(&new_c))
                    {
                        // a → b  ≡  ¬a ∨ b
                        let result = if a == 0.0 || b != 0.0 { 1.0 } else { 0.0 };
                        stats.boolean_folds = stats.boolean_folds.saturating_add(1);
                        return (TLExpr::Constant(result), true);
                    }
                }
                (TLExpr::Imply(Box::new(new_p), Box::new(new_c)), cp || cc)
            }

            // ── If-then-else ──────────────────────────────────────────────
            TLExpr::IfThenElse {
                condition,
                then_branch,
                else_branch,
            } => {
                let (new_cond, cc) = self.propagate(*condition, stats);
                let (new_then, ct) = self.propagate(*then_branch, stats);
                let (new_else, ce) = self.propagate(*else_branch, stats);
                if self.config.fold_boolean {
                    if let Some(v) = Self::as_constant(&new_cond) {
                        if v != 0.0 {
                            // condition is truthy → take then branch
                            stats.boolean_folds = stats.boolean_folds.saturating_add(1);
                            return (new_then, true);
                        } else {
                            // condition is falsy → take else branch
                            stats.boolean_folds = stats.boolean_folds.saturating_add(1);
                            return (new_else, true);
                        }
                    }
                }
                let changed = cc || ct || ce;
                (
                    TLExpr::IfThenElse {
                        condition: Box::new(new_cond),
                        then_branch: Box::new(new_then),
                        else_branch: Box::new(new_else),
                    },
                    changed,
                )
            }

            // ── Score (unary passthrough) ─────────────────────────────────
            TLExpr::Score(inner) => {
                let (new_inner, changed) = self.propagate(*inner, stats);
                (TLExpr::Score(Box::new(new_inner)), changed)
            }

            // ── Quantifiers — recurse into body ──────────────────────────
            TLExpr::Exists { var, domain, body } => {
                let (new_body, changed) = self.propagate(*body, stats);
                (
                    TLExpr::Exists {
                        var,
                        domain,
                        body: Box::new(new_body),
                    },
                    changed,
                )
            }
            TLExpr::ForAll { var, domain, body } => {
                let (new_body, changed) = self.propagate(*body, stats);
                (
                    TLExpr::ForAll {
                        var,
                        domain,
                        body: Box::new(new_body),
                    },
                    changed,
                )
            }

            // ── Let binding ───────────────────────────────────────────────
            TLExpr::Let { var, value, body } => {
                let (new_value, cv) = self.propagate(*value, stats);
                let (new_body, cb) = self.propagate(*body, stats);
                (
                    TLExpr::Let {
                        var,
                        value: Box::new(new_value),
                        body: Box::new(new_body),
                    },
                    cv || cb,
                )
            }

            // ── Aggregate ─────────────────────────────────────────────────
            TLExpr::Aggregate {
                op,
                var,
                domain,
                body,
                group_by,
            } => {
                let (new_body, changed) = self.propagate(*body, stats);
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

            // ── Modal / temporal / fuzzy — recurse ────────────────────────
            TLExpr::Box(inner) => {
                let (n, c) = self.propagate(*inner, stats);
                (TLExpr::Box(Box::new(n)), c)
            }
            TLExpr::Diamond(inner) => {
                let (n, c) = self.propagate(*inner, stats);
                (TLExpr::Diamond(Box::new(n)), c)
            }
            TLExpr::Next(inner) => {
                let (n, c) = self.propagate(*inner, stats);
                (TLExpr::Next(Box::new(n)), c)
            }
            TLExpr::Eventually(inner) => {
                let (n, c) = self.propagate(*inner, stats);
                (TLExpr::Eventually(Box::new(n)), c)
            }
            TLExpr::Always(inner) => {
                let (n, c) = self.propagate(*inner, stats);
                (TLExpr::Always(Box::new(n)), c)
            }
            TLExpr::Until { before, after } => {
                let (nb, cb) = self.propagate(*before, stats);
                let (na, ca) = self.propagate(*after, stats);
                (
                    TLExpr::Until {
                        before: Box::new(nb),
                        after: Box::new(na),
                    },
                    cb || ca,
                )
            }
            TLExpr::Release { released, releaser } => {
                let (nr, cr) = self.propagate(*released, stats);
                let (nl, cl) = self.propagate(*releaser, stats);
                (
                    TLExpr::Release {
                        released: Box::new(nr),
                        releaser: Box::new(nl),
                    },
                    cr || cl,
                )
            }
            TLExpr::WeakUntil { before, after } => {
                let (nb, cb) = self.propagate(*before, stats);
                let (na, ca) = self.propagate(*after, stats);
                (
                    TLExpr::WeakUntil {
                        before: Box::new(nb),
                        after: Box::new(na),
                    },
                    cb || ca,
                )
            }
            TLExpr::StrongRelease { released, releaser } => {
                let (nr, cr) = self.propagate(*released, stats);
                let (nl, cl) = self.propagate(*releaser, stats);
                (
                    TLExpr::StrongRelease {
                        released: Box::new(nr),
                        releaser: Box::new(nl),
                    },
                    cr || cl,
                )
            }

            TLExpr::TNorm { kind, left, right } => {
                let (nl, cl) = self.propagate(*left, stats);
                let (nr, cr) = self.propagate(*right, stats);
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
                let (nl, cl) = self.propagate(*left, stats);
                let (nr, cr) = self.propagate(*right, stats);
                (
                    TLExpr::TCoNorm {
                        kind,
                        left: Box::new(nl),
                        right: Box::new(nr),
                    },
                    cl || cr,
                )
            }
            TLExpr::FuzzyNot { kind, expr: inner } => {
                let (n, c) = self.propagate(*inner, stats);
                (
                    TLExpr::FuzzyNot {
                        kind,
                        expr: Box::new(n),
                    },
                    c,
                )
            }
            TLExpr::FuzzyImplication {
                kind,
                premise,
                conclusion,
            } => {
                let (np, cp) = self.propagate(*premise, stats);
                let (nc, cc) = self.propagate(*conclusion, stats);
                (
                    TLExpr::FuzzyImplication {
                        kind,
                        premise: Box::new(np),
                        conclusion: Box::new(nc),
                    },
                    cp || cc,
                )
            }

            // ── Probabilistic ─────────────────────────────────────────────
            TLExpr::SoftExists {
                var,
                domain,
                body,
                temperature,
            } => {
                let (nb, changed) = self.propagate(*body, stats);
                (
                    TLExpr::SoftExists {
                        var,
                        domain,
                        body: Box::new(nb),
                        temperature,
                    },
                    changed,
                )
            }
            TLExpr::SoftForAll {
                var,
                domain,
                body,
                temperature,
            } => {
                let (nb, changed) = self.propagate(*body, stats);
                (
                    TLExpr::SoftForAll {
                        var,
                        domain,
                        body: Box::new(nb),
                        temperature,
                    },
                    changed,
                )
            }
            TLExpr::WeightedRule { weight, rule } => {
                let (nr, changed) = self.propagate(*rule, stats);
                (
                    TLExpr::WeightedRule {
                        weight,
                        rule: Box::new(nr),
                    },
                    changed,
                )
            }
            TLExpr::ProbabilisticChoice { alternatives } => {
                let mut changed = false;
                let new_alts: Vec<(f64, TLExpr)> = alternatives
                    .into_iter()
                    .map(|(p, e)| {
                        let (ne, c) = self.propagate(e, stats);
                        if c {
                            changed = true;
                        }
                        (p, ne)
                    })
                    .collect();
                (
                    TLExpr::ProbabilisticChoice {
                        alternatives: new_alts,
                    },
                    changed,
                )
            }

            // ── Higher-order ──────────────────────────────────────────────
            TLExpr::Lambda {
                var,
                var_type,
                body,
            } => {
                let (nb, changed) = self.propagate(*body, stats);
                (
                    TLExpr::Lambda {
                        var,
                        var_type,
                        body: Box::new(nb),
                    },
                    changed,
                )
            }
            TLExpr::Apply { function, argument } => {
                let (nf, cf) = self.propagate(*function, stats);
                let (na, ca) = self.propagate(*argument, stats);
                (
                    TLExpr::Apply {
                        function: Box::new(nf),
                        argument: Box::new(na),
                    },
                    cf || ca,
                )
            }

            // ── Set operations ────────────────────────────────────────────
            TLExpr::SetMembership { element, set } => {
                let (ne, ce) = self.propagate(*element, stats);
                let (ns, cs) = self.propagate(*set, stats);
                (
                    TLExpr::SetMembership {
                        element: Box::new(ne),
                        set: Box::new(ns),
                    },
                    ce || cs,
                )
            }
            TLExpr::SetUnion { left, right } => {
                let (nl, cl) = self.propagate(*left, stats);
                let (nr, cr) = self.propagate(*right, stats);
                (
                    TLExpr::SetUnion {
                        left: Box::new(nl),
                        right: Box::new(nr),
                    },
                    cl || cr,
                )
            }
            TLExpr::SetIntersection { left, right } => {
                let (nl, cl) = self.propagate(*left, stats);
                let (nr, cr) = self.propagate(*right, stats);
                (
                    TLExpr::SetIntersection {
                        left: Box::new(nl),
                        right: Box::new(nr),
                    },
                    cl || cr,
                )
            }
            TLExpr::SetDifference { left, right } => {
                let (nl, cl) = self.propagate(*left, stats);
                let (nr, cr) = self.propagate(*right, stats);
                (
                    TLExpr::SetDifference {
                        left: Box::new(nl),
                        right: Box::new(nr),
                    },
                    cl || cr,
                )
            }
            TLExpr::SetCardinality { set } => {
                let (ns, changed) = self.propagate(*set, stats);
                (TLExpr::SetCardinality { set: Box::new(ns) }, changed)
            }
            TLExpr::SetComprehension {
                var,
                domain,
                condition,
            } => {
                let (nc, changed) = self.propagate(*condition, stats);
                (
                    TLExpr::SetComprehension {
                        var,
                        domain,
                        condition: Box::new(nc),
                    },
                    changed,
                )
            }

            // ── Counting quantifiers ──────────────────────────────────────
            TLExpr::CountingExists {
                var,
                domain,
                body,
                min_count,
            } => {
                let (nb, changed) = self.propagate(*body, stats);
                (
                    TLExpr::CountingExists {
                        var,
                        domain,
                        body: Box::new(nb),
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
                let (nb, changed) = self.propagate(*body, stats);
                (
                    TLExpr::CountingForAll {
                        var,
                        domain,
                        body: Box::new(nb),
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
                let (nb, changed) = self.propagate(*body, stats);
                (
                    TLExpr::ExactCount {
                        var,
                        domain,
                        body: Box::new(nb),
                        count,
                    },
                    changed,
                )
            }
            TLExpr::Majority { var, domain, body } => {
                let (nb, changed) = self.propagate(*body, stats);
                (
                    TLExpr::Majority {
                        var,
                        domain,
                        body: Box::new(nb),
                    },
                    changed,
                )
            }

            // ── Fixed-point ───────────────────────────────────────────────
            TLExpr::LeastFixpoint { var, body } => {
                let (nb, changed) = self.propagate(*body, stats);
                (
                    TLExpr::LeastFixpoint {
                        var,
                        body: Box::new(nb),
                    },
                    changed,
                )
            }
            TLExpr::GreatestFixpoint { var, body } => {
                let (nb, changed) = self.propagate(*body, stats);
                (
                    TLExpr::GreatestFixpoint {
                        var,
                        body: Box::new(nb),
                    },
                    changed,
                )
            }

            // ── Hybrid logic ──────────────────────────────────────────────
            TLExpr::At { nominal, formula } => {
                let (nf, changed) = self.propagate(*formula, stats);
                (
                    TLExpr::At {
                        nominal,
                        formula: Box::new(nf),
                    },
                    changed,
                )
            }
            TLExpr::Somewhere { formula } => {
                let (nf, changed) = self.propagate(*formula, stats);
                (
                    TLExpr::Somewhere {
                        formula: Box::new(nf),
                    },
                    changed,
                )
            }
            TLExpr::Everywhere { formula } => {
                let (nf, changed) = self.propagate(*formula, stats);
                (
                    TLExpr::Everywhere {
                        formula: Box::new(nf),
                    },
                    changed,
                )
            }

            // ── Constraint programming ────────────────────────────────────
            TLExpr::GlobalCardinality {
                variables,
                values,
                min_occurrences,
                max_occurrences,
            } => {
                let mut changed = false;
                let new_values: Vec<TLExpr> = values
                    .into_iter()
                    .map(|e| {
                        let (ne, c) = self.propagate(e, stats);
                        if c {
                            changed = true;
                        }
                        ne
                    })
                    .collect();
                (
                    TLExpr::GlobalCardinality {
                        variables,
                        values: new_values,
                        min_occurrences,
                        max_occurrences,
                    },
                    changed,
                )
            }

            // ── Abductive reasoning ───────────────────────────────────────
            TLExpr::Explain { formula } => {
                let (nf, changed) = self.propagate(*formula, stats);
                (
                    TLExpr::Explain {
                        formula: Box::new(nf),
                    },
                    changed,
                )
            }

            // ── Pattern matching ──────────────────────────────────────────
            TLExpr::SymbolLiteral(_) => (expr, false),

            TLExpr::Match { scrutinee, arms } => {
                let (new_scrutinee, sc) = self.propagate(*scrutinee, stats);
                let mut any_changed = sc;
                let new_arms = arms
                    .into_iter()
                    .map(|(pat, body)| {
                        let (new_body, bc) = self.propagate(*body, stats);
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
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    /// Extract the numeric value if `expr` is a `Constant`, otherwise `None`.
    pub fn as_constant(expr: &TLExpr) -> Option<f64> {
        if let TLExpr::Constant(v) = expr {
            Some(*v)
        } else {
            None
        }
    }

    /// Recurse into both children of a binary arithmetic op, then try to fold.
    ///
    /// `ctor` is used to reconstruct the node when folding is not possible.
    fn fold_binary_arith(
        &self,
        op_name: &str,
        lhs: TLExpr,
        rhs: TLExpr,
        stats: &mut ConstPropStats,
        ctor: fn(Box<TLExpr>, Box<TLExpr>) -> TLExpr,
    ) -> (TLExpr, bool) {
        let (new_lhs, cl) = self.propagate(lhs, stats);
        let (new_rhs, cr) = self.propagate(rhs, stats);
        let child_changed = cl || cr;

        if self.config.fold_arithmetic {
            if let (Some(a), Some(b)) = (Self::as_constant(&new_lhs), Self::as_constant(&new_rhs)) {
                if let Some(folded) = self.fold_arith_binary(op_name, a, b, stats) {
                    return (folded, true);
                }
            }
        }
        (ctor(Box::new(new_lhs), Box::new(new_rhs)), child_changed)
    }

    /// Recurse into both children of a comparison op, then try to fold.
    fn fold_binary_cmp(
        &self,
        op_name: &str,
        lhs: TLExpr,
        rhs: TLExpr,
        stats: &mut ConstPropStats,
        ctor: fn(Box<TLExpr>, Box<TLExpr>) -> TLExpr,
    ) -> (TLExpr, bool) {
        let (new_lhs, cl) = self.propagate(lhs, stats);
        let (new_rhs, cr) = self.propagate(rhs, stats);
        let child_changed = cl || cr;

        if self.config.fold_comparisons {
            if let (Some(a), Some(b)) = (Self::as_constant(&new_lhs), Self::as_constant(&new_rhs)) {
                if let Some(folded) = self.fold_comparison(op_name, a, b, stats) {
                    return (folded, true);
                }
            }
        }
        (ctor(Box::new(new_lhs), Box::new(new_rhs)), child_changed)
    }

    /// Recurse into the child of a unary math op, then try to fold.
    fn fold_unary_math(
        &self,
        op_name: &str,
        inner: TLExpr,
        stats: &mut ConstPropStats,
        ctor: fn(Box<TLExpr>) -> TLExpr,
    ) -> (TLExpr, bool) {
        let (new_inner, child_changed) = self.propagate(inner, stats);

        if self.config.fold_boolean {
            if let Some(v) = Self::as_constant(&new_inner) {
                let maybe_result = Self::fold_unary_math_value(op_name, v);
                if let Some(result) = maybe_result {
                    stats.boolean_folds = stats.boolean_folds.saturating_add(1);
                    return (TLExpr::Constant(result), true);
                }
            }
        }
        (ctor(Box::new(new_inner)), child_changed)
    }

    /// Evaluate a unary math function on a constant, returning `None` on error
    /// (e.g. `Log` of a negative number, `Sqrt` of negative).
    fn fold_unary_math_value(op_name: &str, v: f64) -> Option<f64> {
        match op_name {
            "Abs" => Some(v.abs()),
            "Floor" => Some(v.floor()),
            "Ceil" => Some(v.ceil()),
            "Round" => Some(v.round()),
            "Sqrt" => {
                if v < 0.0 {
                    None
                } else {
                    Some(v.sqrt())
                }
            }
            "Exp" => Some(v.exp()),
            "Log" => {
                if v <= 0.0 {
                    None
                } else {
                    Some(v.ln())
                }
            }
            "Sin" => Some(v.sin()),
            "Cos" => Some(v.cos()),
            "Tan" => Some(v.tan()),
            _ => None,
        }
    }

    /// Try to evaluate an arithmetic binary operation on two constant values.
    ///
    /// Returns `None` for division-by-zero, Mod-by-zero, and unknown ops.
    fn fold_arith_binary(
        &self,
        op_name: &str,
        lhs: f64,
        rhs: f64,
        stats: &mut ConstPropStats,
    ) -> Option<TLExpr> {
        let result = match op_name {
            "Add" => lhs + rhs,
            "Sub" => lhs - rhs,
            "Mul" => lhs * rhs,
            "Div" => {
                if rhs.abs() < f64::EPSILON {
                    return None; // division by zero — don't fold
                }
                lhs / rhs
            }
            "Pow" => lhs.powf(rhs),
            "Mod" => {
                if rhs.abs() < f64::EPSILON {
                    return None; // mod by zero — don't fold
                }
                lhs % rhs
            }
            "Min" => lhs.min(rhs),
            "Max" => lhs.max(rhs),
            _ => return None,
        };

        if result.is_finite() || result.is_infinite() {
            // We allow infinite results (e.g. 1/0 for large divisors, Pow overflow)
            // but we already guard against div/mod by zero above.
            stats.arithmetic_folds = stats.arithmetic_folds.saturating_add(1);
            Some(TLExpr::Constant(result))
        } else {
            // NaN — don't fold
            None
        }
    }

    /// Try to evaluate a comparison operation on two constants, returning a
    /// boolean constant (`Constant(1.0)` = True, `Constant(0.0)` = False).
    fn fold_comparison(
        &self,
        op_name: &str,
        lhs: f64,
        rhs: f64,
        stats: &mut ConstPropStats,
    ) -> Option<TLExpr> {
        let bool_result: bool = match op_name {
            "Eq" => (lhs - rhs).abs() <= self.config.float_tolerance,
            "Lt" => lhs < rhs,
            "Gt" => lhs > rhs,
            "Lte" => lhs <= rhs || (lhs - rhs).abs() <= self.config.float_tolerance,
            "Gte" => lhs >= rhs || (lhs - rhs).abs() <= self.config.float_tolerance,
            _ => return None,
        };
        stats.comparison_folds = stats.comparison_folds.saturating_add(1);
        Some(TLExpr::Constant(if bool_result { 1.0 } else { 0.0 }))
    }

    /// Count the total number of nodes in an expression tree.
    pub fn count_nodes(expr: &TLExpr) -> u64 {
        match expr {
            // Leaf nodes
            TLExpr::Constant(_)
            | TLExpr::EmptySet
            | TLExpr::AllDifferent { .. }
            | TLExpr::Nominal { .. }
            | TLExpr::Abducible { .. }
            | TLExpr::Pred { .. } => 1,

            // Unary
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
            | TLExpr::Always(e)
            | TLExpr::FuzzyNot { expr: e, .. }
            | TLExpr::Somewhere { formula: e }
            | TLExpr::Everywhere { formula: e }
            | TLExpr::SetCardinality { set: e }
            | TLExpr::Explain { formula: e }
            | TLExpr::WeightedRule { rule: e, .. } => 1 + Self::count_nodes(e),

            // Body quantifiers
            TLExpr::Exists { body: e, .. }
            | TLExpr::ForAll { body: e, .. }
            | TLExpr::SoftExists { body: e, .. }
            | TLExpr::SoftForAll { body: e, .. }
            | TLExpr::Aggregate { body: e, .. }
            | TLExpr::CountingExists { body: e, .. }
            | TLExpr::CountingForAll { body: e, .. }
            | TLExpr::ExactCount { body: e, .. }
            | TLExpr::Majority { body: e, .. }
            | TLExpr::LeastFixpoint { body: e, .. }
            | TLExpr::GreatestFixpoint { body: e, .. }
            | TLExpr::Lambda { body: e, .. }
            | TLExpr::SetComprehension { condition: e, .. }
            | TLExpr::At { formula: e, .. } => 1 + Self::count_nodes(e),

            // Binary
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
            | TLExpr::Gte(l, r)
            | TLExpr::Until {
                before: l,
                after: r,
            }
            | TLExpr::Release {
                released: l,
                releaser: r,
            }
            | TLExpr::WeakUntil {
                before: l,
                after: r,
            }
            | TLExpr::StrongRelease {
                released: l,
                releaser: r,
            }
            | TLExpr::SetMembership { element: l, set: r }
            | TLExpr::SetUnion { left: l, right: r }
            | TLExpr::SetIntersection { left: l, right: r }
            | TLExpr::SetDifference { left: l, right: r }
            | TLExpr::Apply {
                function: l,
                argument: r,
            } => 1 + Self::count_nodes(l) + Self::count_nodes(r),

            TLExpr::TNorm { left, right, .. } | TLExpr::TCoNorm { left, right, .. } => {
                1 + Self::count_nodes(left) + Self::count_nodes(right)
            }
            TLExpr::FuzzyImplication {
                premise,
                conclusion,
                ..
            } => 1 + Self::count_nodes(premise) + Self::count_nodes(conclusion),

            TLExpr::IfThenElse {
                condition,
                then_branch,
                else_branch,
            } => {
                1 + Self::count_nodes(condition)
                    + Self::count_nodes(then_branch)
                    + Self::count_nodes(else_branch)
            }

            TLExpr::Let { value, body, .. } => {
                1 + Self::count_nodes(value) + Self::count_nodes(body)
            }

            TLExpr::ProbabilisticChoice { alternatives } => {
                1 + alternatives
                    .iter()
                    .map(|(_, e)| Self::count_nodes(e))
                    .sum::<u64>()
            }
            TLExpr::GlobalCardinality { values, .. } => {
                1 + values.iter().map(Self::count_nodes).sum::<u64>()
            }

            TLExpr::SymbolLiteral(_) => 1,

            TLExpr::Match { scrutinee, arms } => {
                1 + Self::count_nodes(scrutinee)
                    + arms.iter().map(|(_, b)| Self::count_nodes(b)).sum::<u64>()
            }
        }
    }
}

impl Default for ConstantPropagator {
    fn default() -> Self {
        Self::with_default()
    }
}

// ────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_ir::TLExpr;

    fn propagator() -> ConstantPropagator {
        ConstantPropagator::with_default()
    }

    fn assert_constant(expr: &TLExpr, expected: f64) {
        match expr {
            TLExpr::Constant(v) => {
                let diff = (v - expected).abs();
                assert!(diff < 1e-9, "Expected constant {}, got {}", expected, v);
            }
            other => panic!("Expected Constant({}), got {:?}", expected, other),
        }
    }

    // ── 1. Constant returns itself ──────────────────────────────────────
    #[test]
    fn test_constant_returns_itself() {
        let (result, stats) = propagator().run(TLExpr::Constant(3.0));
        assert_constant(&result, 3.0);
        assert_eq!(stats.total_folds(), 0);
    }

    // ── 2. Add two constants ─────────────────────────────────────────────
    #[test]
    fn test_add_two_constants() {
        let expr = TLExpr::add(TLExpr::Constant(2.0), TLExpr::Constant(3.0));
        let (result, stats) = propagator().run(expr);
        assert_constant(&result, 5.0);
        assert!(stats.arithmetic_folds >= 1);
    }

    // ── 3. Sub two constants ─────────────────────────────────────────────
    #[test]
    fn test_sub_two_constants() {
        let expr = TLExpr::sub(TLExpr::Constant(5.0), TLExpr::Constant(3.0));
        let (result, _) = propagator().run(expr);
        assert_constant(&result, 2.0);
    }

    // ── 4. Mul two constants ─────────────────────────────────────────────
    #[test]
    fn test_mul_two_constants() {
        let expr = TLExpr::mul(TLExpr::Constant(2.0), TLExpr::Constant(4.0));
        let (result, _) = propagator().run(expr);
        assert_constant(&result, 8.0);
    }

    // ── 5. Div two constants ─────────────────────────────────────────────
    #[test]
    fn test_div_two_constants() {
        let expr = TLExpr::div(TLExpr::Constant(6.0), TLExpr::Constant(2.0));
        let (result, _) = propagator().run(expr);
        assert_constant(&result, 3.0);
    }

    // ── 6. Div by zero is not folded ─────────────────────────────────────
    #[test]
    fn test_div_by_zero_no_fold() {
        let x = TLExpr::pred("x", vec![]);
        let expr = TLExpr::div(x, TLExpr::Constant(0.0));
        let (result, stats) = propagator().run(expr);
        // Should NOT be Constant; the Div node must be preserved
        assert!(!matches!(result, TLExpr::Constant(_)));
        assert_eq!(stats.arithmetic_folds, 0);
    }

    // ── 7. Neg (Unary) constant — using Sub(0, x) pattern ─────────────────
    // TLExpr has no Neg variant; fold Sub(0, const) instead.
    #[test]
    fn test_neg_via_sub_constant() {
        let expr = TLExpr::sub(TLExpr::Constant(0.0), TLExpr::Constant(3.0));
        let (result, _) = propagator().run(expr);
        assert_constant(&result, -3.0);
    }

    // ── 7b. Abs constant ──────────────────────────────────────────────────
    #[test]
    fn test_abs_constant() {
        let expr = TLExpr::abs(TLExpr::Constant(-5.0));
        let (result, stats) = propagator().run(expr);
        assert_constant(&result, 5.0);
        assert!(stats.boolean_folds >= 1);
    }

    // ── 8. Nested arithmetic — two passes required ───────────────────────
    #[test]
    fn test_nested_arithmetic() {
        // Add(Mul(2,3), 4) → Add(6, 4) → 10
        let expr = TLExpr::add(
            TLExpr::mul(TLExpr::Constant(2.0), TLExpr::Constant(3.0)),
            TLExpr::Constant(4.0),
        );
        let (result, stats) = propagator().run(expr);
        assert_constant(&result, 10.0);
        assert!(stats.arithmetic_folds >= 2);
    }

    // ── 9. Comparison Lt → True ──────────────────────────────────────────
    #[test]
    fn test_comparison_lt_true() {
        let expr = TLExpr::Lt(
            Box::new(TLExpr::Constant(1.0)),
            Box::new(TLExpr::Constant(2.0)),
        );
        let (result, stats) = propagator().run(expr);
        assert_constant(&result, 1.0); // True = 1.0
        assert!(stats.comparison_folds >= 1);
    }

    // ── 10. Comparison Gt → False ────────────────────────────────────────
    #[test]
    fn test_comparison_gt_false() {
        let expr = TLExpr::Gt(
            Box::new(TLExpr::Constant(1.0)),
            Box::new(TLExpr::Constant(2.0)),
        );
        let (result, stats) = propagator().run(expr);
        assert_constant(&result, 0.0); // False = 0.0
        assert!(stats.comparison_folds >= 1);
    }

    // ── 11. Stats arithmetic_folds > 0 ───────────────────────────────────
    #[test]
    fn test_const_prop_stats_counts() {
        let expr = TLExpr::add(TLExpr::Constant(1.0), TLExpr::Constant(1.0));
        let (_, stats) = propagator().run(expr);
        assert!(stats.arithmetic_folds > 0, "Expected arithmetic_folds > 0");
    }

    // ── 12. Stats summary non-empty ──────────────────────────────────────
    #[test]
    fn test_const_prop_stats_summary() {
        let expr = TLExpr::add(TLExpr::Constant(2.0), TLExpr::Constant(3.0));
        let (_, stats) = propagator().run(expr);
        let summary = stats.summary();
        assert!(!summary.is_empty(), "Expected non-empty summary");
        assert!(summary.contains("ConstProp"));
    }

    // ── 13. Config default max_passes == 20 ──────────────────────────────
    #[test]
    fn test_const_prop_config_default() {
        let config = ConstPropConfig::default();
        assert_eq!(config.max_passes, 20);
        assert!(config.fold_arithmetic);
        assert!(config.fold_comparisons);
        assert!(config.fold_boolean);
        assert!((config.float_tolerance - 1e-12).abs() < 1e-20);
    }

    // ── 14. Disabled fold — arithmetic off ───────────────────────────────
    #[test]
    fn test_disabled_fold() {
        let config = ConstPropConfig {
            fold_arithmetic: false,
            ..Default::default()
        };
        let prop = ConstantPropagator::new(config);
        let expr = TLExpr::add(TLExpr::Constant(2.0), TLExpr::Constant(3.0));
        let (result, stats) = prop.run(expr);
        // Should NOT be folded
        assert!(!matches!(result, TLExpr::Constant(_)));
        assert_eq!(stats.arithmetic_folds, 0);
    }

    // ── 15. Fixed point — idempotent after first pass ─────────────────────
    #[test]
    fn test_fixed_point() {
        let expr = TLExpr::add(TLExpr::Constant(2.0), TLExpr::Constant(3.0));
        let (result1, _) = propagator().run(expr);
        let (result2, stats2) = propagator().run(result1.clone());
        // Second run should produce same result with 0 additional folds
        assert_eq!(stats2.total_folds(), 0);
        if let TLExpr::Constant(v1) = result1 {
            if let TLExpr::Constant(v2) = result2 {
                assert!((v1 - v2).abs() < 1e-12);
            } else {
                panic!("Expected Constant in second run");
            }
        } else {
            panic!("Expected Constant in first run");
        }
    }

    // ── 16. Passes count >= 1 ────────────────────────────────────────────
    #[test]
    fn test_passes_count() {
        let expr = TLExpr::add(TLExpr::Constant(1.0), TLExpr::Constant(2.0));
        let (_, stats) = propagator().run(expr);
        assert!(stats.passes >= 1, "Expected at least 1 pass");
    }

    // ── 17. Reduction pct — nodes_after < nodes_before ───────────────────
    #[test]
    fn test_reduction_pct() {
        // Add(Mul(2,3), 4) has 5 nodes, result has 1
        let expr = TLExpr::add(
            TLExpr::mul(TLExpr::Constant(2.0), TLExpr::Constant(3.0)),
            TLExpr::Constant(4.0),
        );
        let (_, stats) = propagator().run(expr);
        assert!(stats.nodes_before > stats.nodes_after);
        assert!(stats.reduction_pct() > 0.0);
    }

    // ── 18. Non-constant unchanged ───────────────────────────────────────
    #[test]
    fn test_non_constant_unchanged() {
        let expr = TLExpr::pred("x", vec![]);
        let (result, stats) = propagator().run(expr.clone());
        assert_eq!(stats.total_folds(), 0);
        // Result should still be a Pred
        assert!(matches!(result, TLExpr::Pred { .. }));
    }

    // ── 19. Mixed expr — can't fold ──────────────────────────────────────
    #[test]
    fn test_mixed_expr() {
        // Add(Constant(2), Pred("x")) — cannot fold because rhs is not a constant
        let expr = TLExpr::add(TLExpr::Constant(2.0), TLExpr::pred("x", vec![]));
        let (result, stats) = propagator().run(expr);
        assert!(matches!(result, TLExpr::Add(_, _)));
        assert_eq!(stats.arithmetic_folds, 0);
    }

    // ── 20. Compose with DCE ─────────────────────────────────────────────
    #[test]
    fn test_const_prop_with_dead_code() {
        use crate::dead_code::{DceConfig, DeadCodeEliminator};

        // And(True, Add(1, 2)):
        //   - const_prop folds Add(1,2) → 3, yielding And(Constant(1.0), Constant(3.0))
        //   - const_prop further folds the And (both operands are constants) → Constant(1.0)
        //     (AND of two non-zero constants = True = 1.0)
        // Running DCE on top should be a no-op but not fail.
        let inner = TLExpr::add(TLExpr::Constant(1.0), TLExpr::Constant(2.0));
        let expr = TLExpr::and(TLExpr::Constant(1.0), inner);

        let (after_cp, cp_stats) = propagator().run(expr);
        // Const prop should have folded at least the Add
        assert!(cp_stats.total_folds() >= 1);

        let dce = DeadCodeEliminator::new(DceConfig::default());
        let (after_dce, _dce_stats) = dce.run(after_cp);
        // The pipeline should converge to a constant (1.0 = True for And(True, 3))
        assert!(matches!(after_dce, TLExpr::Constant(_)));
    }

    // ── Pow folding ───────────────────────────────────────────────────────
    #[test]
    fn test_pow_two_constants() {
        let expr = TLExpr::pow(TLExpr::Constant(2.0), TLExpr::Constant(10.0));
        let (result, _) = propagator().run(expr);
        assert_constant(&result, 1024.0);
    }

    // ── Min / Max folding ─────────────────────────────────────────────────
    #[test]
    fn test_min_max_constants() {
        let min_expr = TLExpr::min(TLExpr::Constant(3.0), TLExpr::Constant(7.0));
        let (min_result, _) = propagator().run(min_expr);
        assert_constant(&min_result, 3.0);

        let max_expr = TLExpr::max(TLExpr::Constant(3.0), TLExpr::Constant(7.0));
        let (max_result, _) = propagator().run(max_expr);
        assert_constant(&max_result, 7.0);
    }

    // ── Eq comparison with tolerance ─────────────────────────────────────
    #[test]
    fn test_comparison_eq_true() {
        let a = 1.0_f64;
        let b = a + 1e-13; // within tolerance
        let expr = TLExpr::Eq(Box::new(TLExpr::Constant(a)), Box::new(TLExpr::Constant(b)));
        let (result, stats) = propagator().run(expr);
        assert_constant(&result, 1.0); // True
        assert!(stats.comparison_folds >= 1);
    }

    // ── count_nodes smoke test ────────────────────────────────────────────
    #[test]
    fn test_count_nodes() {
        assert_eq!(ConstantPropagator::count_nodes(&TLExpr::Constant(1.0)), 1);
        let binary = TLExpr::add(TLExpr::Constant(1.0), TLExpr::Constant(2.0));
        assert_eq!(ConstantPropagator::count_nodes(&binary), 3);
    }
}
