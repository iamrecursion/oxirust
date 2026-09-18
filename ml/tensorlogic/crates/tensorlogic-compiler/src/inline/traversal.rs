use tensorlogic_ir::TLExpr;

use super::config::{InlineConfig, InlineStats};
use super::helpers::{
    count_free_occurrences, count_nodes, expr_depth, is_constant_binding, is_var_binding,
};
use super::substitute::substitute;

/// The let-inlining pass.
///
/// Iterates to a fixed point (or until `config.max_passes` is reached),
/// replacing eligible `Let` bindings with direct substitution of the bound
/// value into the body.
pub struct LetInliner {
    pub(super) config: InlineConfig,
}

impl Default for LetInliner {
    fn default() -> Self {
        Self::with_default()
    }
}

impl LetInliner {
    /// Construct a new inliner with the given configuration.
    pub fn new(config: InlineConfig) -> Self {
        Self { config }
    }

    /// Construct a new inliner with the default configuration.
    pub fn with_default() -> Self {
        Self::new(InlineConfig::default())
    }

    // ─────────────────────────────────────────────────────────────────────
    // Public entry-point
    // ─────────────────────────────────────────────────────────────────────

    /// Run inlining to a fixed point (or until `config.max_passes` is reached).
    ///
    /// Returns the rewritten expression and collected [`InlineStats`].
    pub fn run(&self, expr: TLExpr) -> (TLExpr, InlineStats) {
        let mut stats = InlineStats {
            nodes_before: count_nodes(&expr),
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

        stats.nodes_after = count_nodes(&current);
        (current, stats)
    }

    // ─────────────────────────────────────────────────────────────────────
    // Single pass
    // ─────────────────────────────────────────────────────────────────────

    /// Execute a single bottom-up inlining pass.
    ///
    /// Returns `(new_expr, did_change)`.
    fn run_pass(&self, expr: TLExpr, stats: &mut InlineStats) -> (TLExpr, bool) {
        self.inline_expr(expr, stats)
    }

    // ─────────────────────────────────────────────────────────────────────
    // Core recursive transformation
    // ─────────────────────────────────────────────────────────────────────

    /// Recursively inline let-bindings in `expr`.
    ///
    /// The traversal is bottom-up: children are processed first so that
    /// nested bindings are simplified before the enclosing binder is
    /// considered.
    fn inline_expr(&self, expr: TLExpr, stats: &mut InlineStats) -> (TLExpr, bool) {
        match expr {
            // ── The key case: Let bindings ───────────────────────────────────
            TLExpr::Let { var, value, body } => {
                // First recurse into the value and body.
                let (new_value, cv) = self.inline_expr(*value, stats);
                let (new_body, cb) = self.inline_expr(*body, stats);
                let child_changed = cv || cb;

                // Decide whether to inline this binding.
                let depth_ok = expr_depth(&new_value) <= self.config.max_inline_depth;

                if depth_ok {
                    // Case 1: constant — always inline if flag set
                    if self.config.inline_constants && is_constant_binding(&new_value) {
                        stats.constant_inlines += 1;
                        let inlined = substitute(&var, &new_value, new_body);
                        // Recurse once more: the substituted body may expose new opportunities.
                        let (final_expr, _) = self.inline_expr(inlined, stats);
                        return (final_expr, true);
                    }

                    // Case 2: simple variable alias — always inline if flag set
                    if self.config.inline_vars && is_var_binding(&new_value) {
                        stats.variable_inlines += 1;
                        let inlined = substitute(&var, &new_value, new_body);
                        let (final_expr, _) = self.inline_expr(inlined, stats);
                        return (final_expr, true);
                    }

                    // Case 3: single-use — inline if flag set
                    if self.config.inline_single_use && count_free_occurrences(&var, &new_body) == 1
                    {
                        stats.single_use_inlines += 1;
                        let inlined = substitute(&var, &new_value, new_body);
                        let (final_expr, _) = self.inline_expr(inlined, stats);
                        return (final_expr, true);
                    }
                }

                // Not inlined — keep the Let node with updated children.
                (
                    TLExpr::Let {
                        var,
                        value: Box::new(new_value),
                        body: Box::new(new_body),
                    },
                    child_changed,
                )
            }

            // ── Boolean connectives ──────────────────────────────────────────
            TLExpr::And(l, r) => {
                let (nl, cl) = self.inline_expr(*l, stats);
                let (nr, cr) = self.inline_expr(*r, stats);
                (TLExpr::And(Box::new(nl), Box::new(nr)), cl || cr)
            }
            TLExpr::Or(l, r) => {
                let (nl, cl) = self.inline_expr(*l, stats);
                let (nr, cr) = self.inline_expr(*r, stats);
                (TLExpr::Or(Box::new(nl), Box::new(nr)), cl || cr)
            }
            TLExpr::Not(e) => {
                let (ne, changed) = self.inline_expr(*e, stats);
                (TLExpr::Not(Box::new(ne)), changed)
            }
            TLExpr::Imply(l, r) => {
                let (nl, cl) = self.inline_expr(*l, stats);
                let (nr, cr) = self.inline_expr(*r, stats);
                (TLExpr::Imply(Box::new(nl), Box::new(nr)), cl || cr)
            }

            // ── Arithmetic binary ops ────────────────────────────────────────
            TLExpr::Add(l, r) => self.map_binary(TLExpr::Add, *l, *r, stats),
            TLExpr::Sub(l, r) => self.map_binary(TLExpr::Sub, *l, *r, stats),
            TLExpr::Mul(l, r) => self.map_binary(TLExpr::Mul, *l, *r, stats),
            TLExpr::Div(l, r) => self.map_binary(TLExpr::Div, *l, *r, stats),
            TLExpr::Pow(l, r) => self.map_binary(TLExpr::Pow, *l, *r, stats),
            TLExpr::Mod(l, r) => self.map_binary(TLExpr::Mod, *l, *r, stats),
            TLExpr::Min(l, r) => self.map_binary(TLExpr::Min, *l, *r, stats),
            TLExpr::Max(l, r) => self.map_binary(TLExpr::Max, *l, *r, stats),

            // ── Comparison binary ops ────────────────────────────────────────
            TLExpr::Eq(l, r) => self.map_binary(TLExpr::Eq, *l, *r, stats),
            TLExpr::Lt(l, r) => self.map_binary(TLExpr::Lt, *l, *r, stats),
            TLExpr::Gt(l, r) => self.map_binary(TLExpr::Gt, *l, *r, stats),
            TLExpr::Lte(l, r) => self.map_binary(TLExpr::Lte, *l, *r, stats),
            TLExpr::Gte(l, r) => self.map_binary(TLExpr::Gte, *l, *r, stats),

            // ── Unary math ops ───────────────────────────────────────────────
            TLExpr::Abs(e) => self.map_unary(TLExpr::Abs, *e, stats),
            TLExpr::Floor(e) => self.map_unary(TLExpr::Floor, *e, stats),
            TLExpr::Ceil(e) => self.map_unary(TLExpr::Ceil, *e, stats),
            TLExpr::Round(e) => self.map_unary(TLExpr::Round, *e, stats),
            TLExpr::Sqrt(e) => self.map_unary(TLExpr::Sqrt, *e, stats),
            TLExpr::Exp(e) => self.map_unary(TLExpr::Exp, *e, stats),
            TLExpr::Log(e) => self.map_unary(TLExpr::Log, *e, stats),
            TLExpr::Sin(e) => self.map_unary(TLExpr::Sin, *e, stats),
            TLExpr::Cos(e) => self.map_unary(TLExpr::Cos, *e, stats),
            TLExpr::Tan(e) => self.map_unary(TLExpr::Tan, *e, stats),
            TLExpr::Score(e) => self.map_unary(TLExpr::Score, *e, stats),

            // ── Modal / temporal unary ───────────────────────────────────────
            TLExpr::Box(e) => self.map_unary(TLExpr::Box, *e, stats),
            TLExpr::Diamond(e) => self.map_unary(TLExpr::Diamond, *e, stats),
            TLExpr::Next(e) => self.map_unary(TLExpr::Next, *e, stats),
            TLExpr::Eventually(e) => self.map_unary(TLExpr::Eventually, *e, stats),
            TLExpr::Always(e) => self.map_unary(TLExpr::Always, *e, stats),

            // ── Temporal binary ──────────────────────────────────────────────
            TLExpr::Until { before, after } => {
                let (nb, cb) = self.inline_expr(*before, stats);
                let (na, ca) = self.inline_expr(*after, stats);
                (
                    TLExpr::Until {
                        before: Box::new(nb),
                        after: Box::new(na),
                    },
                    cb || ca,
                )
            }
            TLExpr::Release { released, releaser } => {
                let (nr, cr) = self.inline_expr(*released, stats);
                let (ne, ce) = self.inline_expr(*releaser, stats);
                (
                    TLExpr::Release {
                        released: Box::new(nr),
                        releaser: Box::new(ne),
                    },
                    cr || ce,
                )
            }
            TLExpr::WeakUntil { before, after } => {
                let (nb, cb) = self.inline_expr(*before, stats);
                let (na, ca) = self.inline_expr(*after, stats);
                (
                    TLExpr::WeakUntil {
                        before: Box::new(nb),
                        after: Box::new(na),
                    },
                    cb || ca,
                )
            }
            TLExpr::StrongRelease { released, releaser } => {
                let (nr, cr) = self.inline_expr(*released, stats);
                let (ne, ce) = self.inline_expr(*releaser, stats);
                (
                    TLExpr::StrongRelease {
                        released: Box::new(nr),
                        releaser: Box::new(ne),
                    },
                    cr || ce,
                )
            }

            // ── Fuzzy operators ──────────────────────────────────────────────
            TLExpr::TNorm { kind, left, right } => {
                let (nl, cl) = self.inline_expr(*left, stats);
                let (nr, cr) = self.inline_expr(*right, stats);
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
                let (nl, cl) = self.inline_expr(*left, stats);
                let (nr, cr) = self.inline_expr(*right, stats);
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
                let (ne, changed) = self.inline_expr(*expr, stats);
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
                let (np, cp) = self.inline_expr(*premise, stats);
                let (nc, cc) = self.inline_expr(*conclusion, stats);
                (
                    TLExpr::FuzzyImplication {
                        kind,
                        premise: Box::new(np),
                        conclusion: Box::new(nc),
                    },
                    cp || cc,
                )
            }

            // ── Weighted / probabilistic ─────────────────────────────────────
            TLExpr::WeightedRule { weight, rule } => {
                let (nr, changed) = self.inline_expr(*rule, stats);
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
                        let (ne, changed) = self.inline_expr(e, stats);
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

            // ── IfThenElse ───────────────────────────────────────────────────
            TLExpr::IfThenElse {
                condition,
                then_branch,
                else_branch,
            } => {
                let (nc, cc) = self.inline_expr(*condition, stats);
                let (nt, ct) = self.inline_expr(*then_branch, stats);
                let (ne, ce) = self.inline_expr(*else_branch, stats);
                (
                    TLExpr::IfThenElse {
                        condition: Box::new(nc),
                        then_branch: Box::new(nt),
                        else_branch: Box::new(ne),
                    },
                    cc || ct || ce,
                )
            }

            // ── Quantifiers ──────────────────────────────────────────────────
            TLExpr::Exists { var, domain, body } => {
                let (new_body, changed) = self.inline_expr(*body, stats);
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
                let (new_body, changed) = self.inline_expr(*body, stats);
                (
                    TLExpr::ForAll {
                        var,
                        domain,
                        body: Box::new(new_body),
                    },
                    changed,
                )
            }
            TLExpr::SoftExists {
                var,
                domain,
                body,
                temperature,
            } => {
                let (new_body, changed) = self.inline_expr(*body, stats);
                (
                    TLExpr::SoftExists {
                        var,
                        domain,
                        body: Box::new(new_body),
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
                let (new_body, changed) = self.inline_expr(*body, stats);
                (
                    TLExpr::SoftForAll {
                        var,
                        domain,
                        body: Box::new(new_body),
                        temperature,
                    },
                    changed,
                )
            }

            // ── Aggregation ──────────────────────────────────────────────────
            TLExpr::Aggregate {
                op,
                var,
                domain,
                body,
                group_by,
            } => {
                let (new_body, changed) = self.inline_expr(*body, stats);
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

            // ── Higher-order ─────────────────────────────────────────────────
            TLExpr::Lambda {
                var,
                var_type,
                body,
            } => {
                let (new_body, changed) = self.inline_expr(*body, stats);
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
                let (nf, cf) = self.inline_expr(*function, stats);
                let (na, ca) = self.inline_expr(*argument, stats);
                (
                    TLExpr::Apply {
                        function: Box::new(nf),
                        argument: Box::new(na),
                    },
                    cf || ca,
                )
            }

            // ── Set theory ───────────────────────────────────────────────────
            TLExpr::SetMembership { element, set } => {
                let (ne, ce) = self.inline_expr(*element, stats);
                let (ns, cs) = self.inline_expr(*set, stats);
                (
                    TLExpr::SetMembership {
                        element: Box::new(ne),
                        set: Box::new(ns),
                    },
                    ce || cs,
                )
            }
            TLExpr::SetUnion { left, right } => {
                let (nl, cl) = self.inline_expr(*left, stats);
                let (nr, cr) = self.inline_expr(*right, stats);
                (
                    TLExpr::SetUnion {
                        left: Box::new(nl),
                        right: Box::new(nr),
                    },
                    cl || cr,
                )
            }
            TLExpr::SetIntersection { left, right } => {
                let (nl, cl) = self.inline_expr(*left, stats);
                let (nr, cr) = self.inline_expr(*right, stats);
                (
                    TLExpr::SetIntersection {
                        left: Box::new(nl),
                        right: Box::new(nr),
                    },
                    cl || cr,
                )
            }
            TLExpr::SetDifference { left, right } => {
                let (nl, cl) = self.inline_expr(*left, stats);
                let (nr, cr) = self.inline_expr(*right, stats);
                (
                    TLExpr::SetDifference {
                        left: Box::new(nl),
                        right: Box::new(nr),
                    },
                    cl || cr,
                )
            }
            TLExpr::SetCardinality { set } => {
                let (ns, changed) = self.inline_expr(*set, stats);
                (TLExpr::SetCardinality { set: Box::new(ns) }, changed)
            }
            TLExpr::SetComprehension {
                var,
                domain,
                condition,
            } => {
                let (nc, changed) = self.inline_expr(*condition, stats);
                (
                    TLExpr::SetComprehension {
                        var,
                        domain,
                        condition: Box::new(nc),
                    },
                    changed,
                )
            }

            // ── Counting quantifiers ─────────────────────────────────────────
            TLExpr::CountingExists {
                var,
                domain,
                body,
                min_count,
            } => {
                let (new_body, changed) = self.inline_expr(*body, stats);
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
                let (new_body, changed) = self.inline_expr(*body, stats);
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
                let (new_body, changed) = self.inline_expr(*body, stats);
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
                let (new_body, changed) = self.inline_expr(*body, stats);
                (
                    TLExpr::Majority {
                        var,
                        domain,
                        body: Box::new(new_body),
                    },
                    changed,
                )
            }

            // ── Fixed-point operators ────────────────────────────────────────
            TLExpr::LeastFixpoint { var, body } => {
                let (new_body, changed) = self.inline_expr(*body, stats);
                (
                    TLExpr::LeastFixpoint {
                        var,
                        body: Box::new(new_body),
                    },
                    changed,
                )
            }
            TLExpr::GreatestFixpoint { var, body } => {
                let (new_body, changed) = self.inline_expr(*body, stats);
                (
                    TLExpr::GreatestFixpoint {
                        var,
                        body: Box::new(new_body),
                    },
                    changed,
                )
            }

            // ── Hybrid logic ─────────────────────────────────────────────────
            TLExpr::At { nominal, formula } => {
                let (nf, changed) = self.inline_expr(*formula, stats);
                (
                    TLExpr::At {
                        nominal,
                        formula: Box::new(nf),
                    },
                    changed,
                )
            }
            TLExpr::Somewhere { formula } => {
                let (nf, changed) = self.inline_expr(*formula, stats);
                (
                    TLExpr::Somewhere {
                        formula: Box::new(nf),
                    },
                    changed,
                )
            }
            TLExpr::Everywhere { formula } => {
                let (nf, changed) = self.inline_expr(*formula, stats);
                (
                    TLExpr::Everywhere {
                        formula: Box::new(nf),
                    },
                    changed,
                )
            }

            // ── Abductive ────────────────────────────────────────────────────
            TLExpr::Explain { formula } => {
                let (nf, changed) = self.inline_expr(*formula, stats);
                (
                    TLExpr::Explain {
                        formula: Box::new(nf),
                    },
                    changed,
                )
            }

            // ── Leaves ───────────────────────────────────────────────────────
            leaf @ (TLExpr::Pred { .. }
            | TLExpr::Constant(_)
            | TLExpr::EmptySet
            | TLExpr::AllDifferent { .. }
            | TLExpr::GlobalCardinality { .. }
            | TLExpr::Nominal { .. }
            | TLExpr::Abducible { .. }
            | TLExpr::SymbolLiteral(_)) => (leaf, false),

            TLExpr::Match { scrutinee, arms } => {
                let (new_scrutinee, sc) = self.inline_expr(*scrutinee, stats);
                let mut any_changed = sc;
                let new_arms = arms
                    .into_iter()
                    .map(|(pat, body)| {
                        let (new_body, bc) = self.inline_expr(*body, stats);
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

    // ─────────────────────────────────────────────────────────────────────
    // Helper: binary / unary mapping
    // ─────────────────────────────────────────────────────────────────────

    #[inline]
    fn map_binary(
        &self,
        ctor: fn(Box<TLExpr>, Box<TLExpr>) -> TLExpr,
        l: TLExpr,
        r: TLExpr,
        stats: &mut InlineStats,
    ) -> (TLExpr, bool) {
        let (nl, cl) = self.inline_expr(l, stats);
        let (nr, cr) = self.inline_expr(r, stats);
        (ctor(Box::new(nl), Box::new(nr)), cl || cr)
    }

    #[inline]
    fn map_unary(
        &self,
        ctor: fn(Box<TLExpr>) -> TLExpr,
        e: TLExpr,
        stats: &mut InlineStats,
    ) -> (TLExpr, bool) {
        let (ne, changed) = self.inline_expr(e, stats);
        (ctor(Box::new(ne)), changed)
    }

    // ─────────────────────────────────────────────────────────────────────
    // Public helper forwarding functions (keeping backward-compat API)
    // ─────────────────────────────────────────────────────────────────────

    /// Count how many times `var` appears free in `expr`.
    pub fn count_free_occurrences(var: &str, expr: &TLExpr) -> usize {
        count_free_occurrences(var, expr)
    }

    /// Substitute all free occurrences of `var` with `replacement` in `body`.
    pub fn substitute(var: &str, replacement: &TLExpr, body: TLExpr) -> TLExpr {
        substitute(var, replacement, body)
    }

    /// Returns `true` if `expr` is a constant literal (`Constant(_)`).
    pub fn is_constant_binding(expr: &TLExpr) -> bool {
        is_constant_binding(expr)
    }

    /// Returns `true` if `expr` is a zero-argument predicate (variable alias).
    pub fn is_var_binding(expr: &TLExpr) -> bool {
        is_var_binding(expr)
    }

    /// Returns `true` if `expr` is a "simple" binding worth inlining regardless
    /// of use count: either a constant or a variable alias.
    pub fn is_simple_binding(expr: &TLExpr) -> bool {
        super::helpers::is_simple_binding(expr)
    }

    /// Compute the depth (height) of an expression tree.
    pub fn expr_depth(expr: &TLExpr) -> usize {
        expr_depth(expr)
    }
}
