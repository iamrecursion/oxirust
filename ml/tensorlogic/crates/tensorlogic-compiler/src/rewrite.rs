//! Pattern-matching rewrite rules engine for TLExpr transformation.
//!
//! This module provides a composable, extensible rule engine that applies
//! structural rewrites to [`tensorlogic_ir::TLExpr`] trees until a fixed point
//! is reached (no further rules apply).
//!
//! # Design
//!
//! - Rules implement the [`RewriteRule`] trait.
//! - The [`RewriteEngine`] holds a collection of rules and drives iteration.
//! - Rewrites are applied **bottom-up**: children are transformed first, then
//!   the root node is offered to each rule in order.
//! - Iteration continues until no rule fires in a full pass (fixed point) or
//!   `max_iterations` is reached.
//!
//! # Built-in rules
//!
//! | Rule struct | Transformation |
//! |---|---|
//! | [`EliminateDoubleNeg`] | `Not(Not(x))` → `x` |
//! | [`FlattenNestedAnd`] | `And(And(a,b),c)` → `And(a,And(b,c))` |
//! | [`FlattenNestedOr`] | `Or(Or(a,b),c)` → `Or(a,Or(b,c))` |
//! | [`EliminateAndTrue`] | `And(True,x)` / `And(x,True)` → `x` |
//! | [`EliminateOrFalse`] | `Or(False,x)` / `Or(x,False)` → `x` |
//!
//! # Example
//!
//! ```rust
//! use tensorlogic_compiler::rewrite::{RewriteEngine, EliminateDoubleNeg};
//! use tensorlogic_ir::{TLExpr, Term};
//!
//! let expr = TLExpr::negate(TLExpr::negate(TLExpr::pred("p", vec![Term::var("x")])));
//!
//! let (result, stats) = RewriteEngine::new()
//!     .add_rule(Box::new(EliminateDoubleNeg))
//!     .rewrite(expr);
//!
//! // Not(Not(p(x))) → p(x)
//! assert_eq!(stats.total_rewrites, 1);
//! println!("{}", stats.summary());
//! ```

use std::collections::HashMap;
use std::fmt;

use tensorlogic_ir::TLExpr;

// ---------------------------------------------------------------------------
// Helper macros for DRY binary / unary child rewriting
// (must be declared before use)
// ---------------------------------------------------------------------------

/// Rewrite both children of a binary node and reconstruct with the same variant.
macro_rules! rewrite_binary {
    ($self:expr, $stats:expr, $ctor:path, $left:expr, $right:expr) => {{
        let (nl, cl) = $self.rewrite_expr(*$left, $stats);
        let (nr, cr) = $self.rewrite_expr(*$right, $stats);
        ($ctor(Box::new(nl), Box::new(nr)), cl || cr)
    }};
}

/// Rewrite the single child of a unary node and reconstruct.
macro_rules! rewrite_unary {
    ($self:expr, $stats:expr, $ctor:path, $inner:expr) => {{
        let (ni, changed) = $self.rewrite_expr(*$inner, $stats);
        ($ctor(Box::new(ni)), changed)
    }};
}

// ---------------------------------------------------------------------------
// RewriteRule trait
// ---------------------------------------------------------------------------

/// A single pattern-matching rewrite rule.
///
/// Implementors inspect an expression and, if the rule matches, return a
/// transformed replacement expression.  If the rule does not apply, `None`
/// is returned and the engine tries the next rule.
pub trait RewriteRule: Send + Sync {
    /// Unique human-readable name used in statistics output.
    fn name(&self) -> &'static str;

    /// Try to apply this rule to `expr`.
    ///
    /// Returns `Some(new_expr)` when the rule fires, `None` otherwise.
    fn apply(&self, expr: &TLExpr) -> Option<TLExpr>;

    /// Whether the engine should recurse into children of `expr` before
    /// trying this rule.  Defaults to `true` (standard bottom-up traversal).
    fn is_recursive(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// RewriteStats
// ---------------------------------------------------------------------------

/// Accumulated statistics for a rewrite pass (or the full fixed-point loop).
#[derive(Debug, Clone, Default)]
pub struct RewriteStats {
    /// How many times each named rule fired.
    pub rules_applied: HashMap<String, u64>,
    /// Total number of individual rewrites across all rules and iterations.
    pub total_rewrites: u64,
    /// Number of full-tree passes performed.
    pub iterations: u32,
    /// Total expression nodes visited (across all passes).
    pub nodes_visited: u64,
    /// `true` when the engine stopped because no rule fired (fixed point),
    /// `false` when it stopped because `max_iterations` was reached.
    pub fixed_point_reached: bool,
}

impl RewriteStats {
    /// Record one application of `rule_name`, incrementing all counters.
    pub fn record_rule(&mut self, rule_name: &str) {
        *self.rules_applied.entry(rule_name.to_owned()).or_insert(0) += 1;
        self.total_rewrites += 1;
    }

    /// Human-readable one-line summary suitable for logging.
    pub fn summary(&self) -> String {
        if self.rules_applied.is_empty() {
            return format!(
                "RewriteStats: 0 rewrites, {} iteration(s), fixed_point={}",
                self.iterations, self.fixed_point_reached
            );
        }

        let mut rule_parts: Vec<String> = self
            .rules_applied
            .iter()
            .map(|(name, count)| format!("{}×{}", name, count))
            .collect();
        rule_parts.sort(); // deterministic output

        format!(
            "RewriteStats: {} rewrite(s) in {} iteration(s), fixed_point={} [{}]",
            self.total_rewrites,
            self.iterations,
            self.fixed_point_reached,
            rule_parts.join(", ")
        )
    }
}

impl fmt::Display for RewriteStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

// ---------------------------------------------------------------------------
// RewriteEngine
// ---------------------------------------------------------------------------

/// Applies a set of [`RewriteRule`]s to a [`TLExpr`] tree until a fixed point.
///
/// Rules are attempted in the order they were added.  Within a single node
/// visit the first firing rule wins and the engine moves on; no other rules
/// are tried for the same node in the same pass.
///
/// The engine iterates until either:
/// - a full pass completes with zero rule firings (fixed point), or
/// - [`max_iterations`][RewriteEngine::max_iterations] is reached.
pub struct RewriteEngine {
    rules: Vec<Box<dyn RewriteRule>>,
    /// Maximum number of full-tree passes before stopping (default: 64).
    pub max_iterations: u32,
    /// Soft limit on node visits per pass (default: 1_000_000).  If exceeded,
    /// the engine finishes the current pass but does not start a new one.
    pub max_nodes_per_pass: u64,
}

impl RewriteEngine {
    /// Create a new engine with no rules and default limits.
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            max_iterations: 64,
            max_nodes_per_pass: 1_000_000,
        }
    }

    /// Set the maximum number of fixed-point iterations (builder pattern).
    pub fn with_max_iterations(mut self, n: u32) -> Self {
        self.max_iterations = n;
        self
    }

    /// Add one rule to the engine (builder pattern).
    pub fn add_rule(mut self, rule: Box<dyn RewriteRule>) -> Self {
        self.rules.push(rule);
        self
    }

    /// Register all five built-in rules (builder pattern).
    pub fn add_all_builtin_rules(self) -> Self {
        builtin_rules()
            .into_iter()
            .fold(self, |engine, rule| engine.add_rule(rule))
    }

    /// Apply all rules to `expr` until a fixed point is reached.
    ///
    /// Returns the final expression together with accumulated [`RewriteStats`].
    pub fn rewrite(&self, expr: TLExpr) -> (TLExpr, RewriteStats) {
        let mut stats = RewriteStats::default();
        let mut current = expr;

        for _iteration in 0..self.max_iterations {
            stats.iterations += 1;
            let (next, changed) = self.rewrite_once(current, &mut stats);
            current = next;

            if !changed {
                stats.fixed_point_reached = true;
                break;
            }

            if stats.nodes_visited >= self.max_nodes_per_pass {
                break;
            }
        }

        (current, stats)
    }

    /// Perform exactly one full pass over the tree.
    ///
    /// Returns the (possibly modified) expression and whether any rule fired.
    fn rewrite_once(&self, expr: TLExpr, stats: &mut RewriteStats) -> (TLExpr, bool) {
        self.rewrite_expr(expr, stats)
    }

    /// Bottom-up recursive traversal: rewrite children first, then try rules
    /// at the current node.
    fn rewrite_expr(&self, expr: TLExpr, stats: &mut RewriteStats) -> (TLExpr, bool) {
        stats.nodes_visited += 1;

        // --- Step 1: descend into children to get a rewritten child tree ---
        let (expr_after_children, children_changed) = self.rewrite_children(expr, stats);

        // --- Step 2: try rules at the current (post-children) node ---
        for rule in &self.rules {
            if let Some(replacement) = rule.apply(&expr_after_children) {
                stats.record_rule(rule.name());
                // Do NOT recurse again here; the outer fixed-point loop handles it.
                return (replacement, true);
            }
        }

        (expr_after_children, children_changed)
    }

    /// Reconstruct the node after recursively rewriting children.
    ///
    /// Only the "interesting" structural variants (`Not`, `And`, `Or`, and a
    /// selection of other compound forms) are descended into; leaf nodes are
    /// returned unchanged.
    fn rewrite_children(&self, expr: TLExpr, stats: &mut RewriteStats) -> (TLExpr, bool) {
        match expr {
            // ---- Structural connectives (primary targets) ----
            TLExpr::Not(inner) => {
                let (new_inner, changed) = self.rewrite_expr(*inner, stats);
                (TLExpr::Not(Box::new(new_inner)), changed)
            }
            TLExpr::And(left, right) => {
                let (new_left, cl) = self.rewrite_expr(*left, stats);
                let (new_right, cr) = self.rewrite_expr(*right, stats);
                (
                    TLExpr::And(Box::new(new_left), Box::new(new_right)),
                    cl || cr,
                )
            }
            TLExpr::Or(left, right) => {
                let (new_left, cl) = self.rewrite_expr(*left, stats);
                let (new_right, cr) = self.rewrite_expr(*right, stats);
                (
                    TLExpr::Or(Box::new(new_left), Box::new(new_right)),
                    cl || cr,
                )
            }
            TLExpr::Imply(ante, cons) => {
                let (new_ante, ca) = self.rewrite_expr(*ante, stats);
                let (new_cons, cc) = self.rewrite_expr(*cons, stats);
                (
                    TLExpr::Imply(Box::new(new_ante), Box::new(new_cons)),
                    ca || cc,
                )
            }
            TLExpr::Score(inner) => {
                let (new_inner, changed) = self.rewrite_expr(*inner, stats);
                (TLExpr::Score(Box::new(new_inner)), changed)
            }

            // ---- Arithmetic binary ----
            TLExpr::Add(l, r) => rewrite_binary!(self, stats, TLExpr::Add, l, r),
            TLExpr::Sub(l, r) => rewrite_binary!(self, stats, TLExpr::Sub, l, r),
            TLExpr::Mul(l, r) => rewrite_binary!(self, stats, TLExpr::Mul, l, r),
            TLExpr::Div(l, r) => rewrite_binary!(self, stats, TLExpr::Div, l, r),
            TLExpr::Pow(l, r) => rewrite_binary!(self, stats, TLExpr::Pow, l, r),
            TLExpr::Mod(l, r) => rewrite_binary!(self, stats, TLExpr::Mod, l, r),
            TLExpr::Min(l, r) => rewrite_binary!(self, stats, TLExpr::Min, l, r),
            TLExpr::Max(l, r) => rewrite_binary!(self, stats, TLExpr::Max, l, r),

            // ---- Arithmetic unary ----
            TLExpr::Abs(inner) => rewrite_unary!(self, stats, TLExpr::Abs, inner),
            TLExpr::Floor(inner) => rewrite_unary!(self, stats, TLExpr::Floor, inner),
            TLExpr::Ceil(inner) => rewrite_unary!(self, stats, TLExpr::Ceil, inner),
            TLExpr::Round(inner) => rewrite_unary!(self, stats, TLExpr::Round, inner),
            TLExpr::Sqrt(inner) => rewrite_unary!(self, stats, TLExpr::Sqrt, inner),
            TLExpr::Exp(inner) => rewrite_unary!(self, stats, TLExpr::Exp, inner),
            TLExpr::Log(inner) => rewrite_unary!(self, stats, TLExpr::Log, inner),
            TLExpr::Sin(inner) => rewrite_unary!(self, stats, TLExpr::Sin, inner),
            TLExpr::Cos(inner) => rewrite_unary!(self, stats, TLExpr::Cos, inner),
            TLExpr::Tan(inner) => rewrite_unary!(self, stats, TLExpr::Tan, inner),

            // ---- Comparison binary ----
            TLExpr::Eq(l, r) => rewrite_binary!(self, stats, TLExpr::Eq, l, r),
            TLExpr::Lt(l, r) => rewrite_binary!(self, stats, TLExpr::Lt, l, r),
            TLExpr::Gt(l, r) => rewrite_binary!(self, stats, TLExpr::Gt, l, r),
            TLExpr::Lte(l, r) => rewrite_binary!(self, stats, TLExpr::Lte, l, r),
            TLExpr::Gte(l, r) => rewrite_binary!(self, stats, TLExpr::Gte, l, r),

            // ---- Conditional ----
            TLExpr::IfThenElse {
                condition,
                then_branch,
                else_branch,
            } => {
                let (new_cond, cc) = self.rewrite_expr(*condition, stats);
                let (new_then, ct) = self.rewrite_expr(*then_branch, stats);
                let (new_else, ce) = self.rewrite_expr(*else_branch, stats);
                (
                    TLExpr::IfThenElse {
                        condition: Box::new(new_cond),
                        then_branch: Box::new(new_then),
                        else_branch: Box::new(new_else),
                    },
                    cc || ct || ce,
                )
            }

            // ---- Quantifiers ----
            TLExpr::Exists { var, domain, body } => {
                let (new_body, changed) = self.rewrite_expr(*body, stats);
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
                let (new_body, changed) = self.rewrite_expr(*body, stats);
                (
                    TLExpr::ForAll {
                        var,
                        domain,
                        body: Box::new(new_body),
                    },
                    changed,
                )
            }

            // ---- Modal logic ----
            TLExpr::Box(inner) => rewrite_unary!(self, stats, TLExpr::Box, inner),
            TLExpr::Diamond(inner) => rewrite_unary!(self, stats, TLExpr::Diamond, inner),

            // ---- Temporal logic ----
            TLExpr::Next(inner) => rewrite_unary!(self, stats, TLExpr::Next, inner),
            TLExpr::Eventually(inner) => {
                rewrite_unary!(self, stats, TLExpr::Eventually, inner)
            }
            TLExpr::Always(inner) => rewrite_unary!(self, stats, TLExpr::Always, inner),
            TLExpr::Until { before, after } => {
                let (nb, cb) = self.rewrite_expr(*before, stats);
                let (na, ca) = self.rewrite_expr(*after, stats);
                (
                    TLExpr::Until {
                        before: Box::new(nb),
                        after: Box::new(na),
                    },
                    cb || ca,
                )
            }
            TLExpr::Release { released, releaser } => {
                let (nr, cr) = self.rewrite_expr(*released, stats);
                let (nl, cl) = self.rewrite_expr(*releaser, stats);
                (
                    TLExpr::Release {
                        released: Box::new(nr),
                        releaser: Box::new(nl),
                    },
                    cr || cl,
                )
            }
            TLExpr::WeakUntil { before, after } => {
                let (nb, cb) = self.rewrite_expr(*before, stats);
                let (na, ca) = self.rewrite_expr(*after, stats);
                (
                    TLExpr::WeakUntil {
                        before: Box::new(nb),
                        after: Box::new(na),
                    },
                    cb || ca,
                )
            }
            TLExpr::StrongRelease { released, releaser } => {
                let (nr, cr) = self.rewrite_expr(*released, stats);
                let (nl, cl) = self.rewrite_expr(*releaser, stats);
                (
                    TLExpr::StrongRelease {
                        released: Box::new(nr),
                        releaser: Box::new(nl),
                    },
                    cr || cl,
                )
            }

            // ---- Fuzzy logic ----
            TLExpr::TNorm { kind, left, right } => {
                let (nl, cl) = self.rewrite_expr(*left, stats);
                let (nr, cr) = self.rewrite_expr(*right, stats);
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
                let (nl, cl) = self.rewrite_expr(*left, stats);
                let (nr, cr) = self.rewrite_expr(*right, stats);
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
                let (ni, changed) = self.rewrite_expr(*inner, stats);
                (
                    TLExpr::FuzzyNot {
                        kind,
                        expr: Box::new(ni),
                    },
                    changed,
                )
            }
            TLExpr::FuzzyImplication {
                kind,
                premise,
                conclusion,
            } => {
                let (np, cp) = self.rewrite_expr(*premise, stats);
                let (nc, cc) = self.rewrite_expr(*conclusion, stats);
                (
                    TLExpr::FuzzyImplication {
                        kind,
                        premise: Box::new(np),
                        conclusion: Box::new(nc),
                    },
                    cp || cc,
                )
            }

            // ---- Probabilistic ----
            TLExpr::SoftExists {
                var,
                domain,
                body,
                temperature,
            } => {
                let (nb, changed) = self.rewrite_expr(*body, stats);
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
                let (nb, changed) = self.rewrite_expr(*body, stats);
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
                let (nr, changed) = self.rewrite_expr(*rule, stats);
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
                let new_alts = alternatives
                    .into_iter()
                    .map(|(prob, alt_expr)| {
                        let (ne, c) = self.rewrite_expr(alt_expr, stats);
                        changed |= c;
                        (prob, ne)
                    })
                    .collect();
                (
                    TLExpr::ProbabilisticChoice {
                        alternatives: new_alts,
                    },
                    changed,
                )
            }

            // ---- Higher-order ----
            TLExpr::Lambda {
                var,
                var_type,
                body,
            } => {
                let (nb, changed) = self.rewrite_expr(*body, stats);
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
                let (nf, cf) = self.rewrite_expr(*function, stats);
                let (na, ca) = self.rewrite_expr(*argument, stats);
                (
                    TLExpr::Apply {
                        function: Box::new(nf),
                        argument: Box::new(na),
                    },
                    cf || ca,
                )
            }

            // ---- Set operations ----
            TLExpr::SetMembership { element, set } => {
                let (ne, ce) = self.rewrite_expr(*element, stats);
                let (ns, cs) = self.rewrite_expr(*set, stats);
                (
                    TLExpr::SetMembership {
                        element: Box::new(ne),
                        set: Box::new(ns),
                    },
                    ce || cs,
                )
            }
            TLExpr::SetUnion { left, right } => {
                let (nl, cl) = self.rewrite_expr(*left, stats);
                let (nr, cr) = self.rewrite_expr(*right, stats);
                (
                    TLExpr::SetUnion {
                        left: Box::new(nl),
                        right: Box::new(nr),
                    },
                    cl || cr,
                )
            }
            TLExpr::SetIntersection { left, right } => {
                let (nl, cl) = self.rewrite_expr(*left, stats);
                let (nr, cr) = self.rewrite_expr(*right, stats);
                (
                    TLExpr::SetIntersection {
                        left: Box::new(nl),
                        right: Box::new(nr),
                    },
                    cl || cr,
                )
            }
            TLExpr::SetDifference { left, right } => {
                let (nl, cl) = self.rewrite_expr(*left, stats);
                let (nr, cr) = self.rewrite_expr(*right, stats);
                (
                    TLExpr::SetDifference {
                        left: Box::new(nl),
                        right: Box::new(nr),
                    },
                    cl || cr,
                )
            }
            TLExpr::SetCardinality { set } => {
                let (ns, changed) = self.rewrite_expr(*set, stats);
                (TLExpr::SetCardinality { set: Box::new(ns) }, changed)
            }
            TLExpr::SetComprehension {
                var,
                domain,
                condition,
            } => {
                let (nc, changed) = self.rewrite_expr(*condition, stats);
                (
                    TLExpr::SetComprehension {
                        var,
                        domain,
                        condition: Box::new(nc),
                    },
                    changed,
                )
            }

            // ---- Let binding ----
            TLExpr::Let { var, value, body } => {
                let (nv, cv) = self.rewrite_expr(*value, stats);
                let (nb, cb) = self.rewrite_expr(*body, stats);
                (
                    TLExpr::Let {
                        var,
                        value: Box::new(nv),
                        body: Box::new(nb),
                    },
                    cv || cb,
                )
            }

            // ---- Aggregate ----
            TLExpr::Aggregate {
                op,
                var,
                domain,
                body,
                group_by,
            } => {
                let (nb, changed) = self.rewrite_expr(*body, stats);
                (
                    TLExpr::Aggregate {
                        op,
                        var,
                        domain,
                        body: Box::new(nb),
                        group_by,
                    },
                    changed,
                )
            }

            // ---- Leaf nodes (no children to recurse into) ----
            leaf => (leaf, false),
        }
    }
}

impl Default for RewriteEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Built-in rules
// ---------------------------------------------------------------------------

// ---- EliminateDoubleNeg ----

/// Eliminate double negation: `Not(Not(x))` → `x`.
#[derive(Debug, Clone, Default)]
pub struct EliminateDoubleNeg;

impl RewriteRule for EliminateDoubleNeg {
    fn name(&self) -> &'static str {
        "eliminate_double_neg"
    }

    fn apply(&self, expr: &TLExpr) -> Option<TLExpr> {
        if let TLExpr::Not(inner) = expr {
            if let TLExpr::Not(inner_inner) = inner.as_ref() {
                return Some(*inner_inner.clone());
            }
        }
        None
    }
}

// ---- FlattenNestedAnd ----

/// Right-associate a left-nested `And`: `And(And(a,b),c)` → `And(a,And(b,c))`.
#[derive(Debug, Clone, Default)]
pub struct FlattenNestedAnd;

impl RewriteRule for FlattenNestedAnd {
    fn name(&self) -> &'static str {
        "flatten_nested_and"
    }

    fn apply(&self, expr: &TLExpr) -> Option<TLExpr> {
        if let TLExpr::And(left, right) = expr {
            if let TLExpr::And(a, b) = left.as_ref() {
                // And(And(a,b), c) → And(a, And(b,c))
                let new_right = TLExpr::And(b.clone(), right.clone());
                return Some(TLExpr::And(a.clone(), Box::new(new_right)));
            }
        }
        None
    }
}

// ---- FlattenNestedOr ----

/// Right-associate a left-nested `Or`: `Or(Or(a,b),c)` → `Or(a,Or(b,c))`.
#[derive(Debug, Clone, Default)]
pub struct FlattenNestedOr;

impl RewriteRule for FlattenNestedOr {
    fn name(&self) -> &'static str {
        "flatten_nested_or"
    }

    fn apply(&self, expr: &TLExpr) -> Option<TLExpr> {
        if let TLExpr::Or(left, right) = expr {
            if let TLExpr::Or(a, b) = left.as_ref() {
                // Or(Or(a,b), c) → Or(a, Or(b,c))
                let new_right = TLExpr::Or(b.clone(), right.clone());
                return Some(TLExpr::Or(a.clone(), Box::new(new_right)));
            }
        }
        None
    }
}

// ---- EliminateAndTrue ----

/// Identity for conjunction: `And(True, x)` or `And(x, True)` → `x`.
///
/// "True" is represented as `TLExpr::Constant(c)` where `c ≈ 1.0`.
#[derive(Debug, Clone, Default)]
pub struct EliminateAndTrue;

impl RewriteRule for EliminateAndTrue {
    fn name(&self) -> &'static str {
        "eliminate_and_true"
    }

    fn apply(&self, expr: &TLExpr) -> Option<TLExpr> {
        if let TLExpr::And(left, right) = expr {
            if is_true_constant(left) {
                return Some(*right.clone());
            }
            if is_true_constant(right) {
                return Some(*left.clone());
            }
        }
        None
    }
}

// ---- EliminateOrFalse ----

/// Identity for disjunction: `Or(False, x)` or `Or(x, False)` → `x`.
///
/// "False" is represented as `TLExpr::Constant(c)` where `c ≈ 0.0`.
#[derive(Debug, Clone, Default)]
pub struct EliminateOrFalse;

impl RewriteRule for EliminateOrFalse {
    fn name(&self) -> &'static str {
        "eliminate_or_false"
    }

    fn apply(&self, expr: &TLExpr) -> Option<TLExpr> {
        if let TLExpr::Or(left, right) = expr {
            if is_false_constant(left) {
                return Some(*right.clone());
            }
            if is_false_constant(right) {
                return Some(*left.clone());
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// Helpers for constant detection
// ---------------------------------------------------------------------------

/// Returns `true` if `expr` is a numeric constant close to `1.0` (logical True).
#[inline]
fn is_true_constant(expr: &TLExpr) -> bool {
    if let TLExpr::Constant(v) = expr {
        (v - 1.0_f64).abs() < f64::EPSILON
    } else {
        false
    }
}

/// Returns `true` if `expr` is a numeric constant close to `0.0` (logical False).
#[inline]
fn is_false_constant(expr: &TLExpr) -> bool {
    if let TLExpr::Constant(v) = expr {
        v.abs() < f64::EPSILON
    } else {
        false
    }
}

// ---------------------------------------------------------------------------
// Convenience constructor
// ---------------------------------------------------------------------------

/// Return one boxed instance of each built-in rule.
///
/// The order matches the recommended application order:
/// 1. [`EliminateDoubleNeg`]
/// 2. [`FlattenNestedAnd`]
/// 3. [`FlattenNestedOr`]
/// 4. [`EliminateAndTrue`]
/// 5. [`EliminateOrFalse`]
pub fn builtin_rules() -> Vec<Box<dyn RewriteRule>> {
    vec![
        Box::new(EliminateDoubleNeg) as Box<dyn RewriteRule>,
        Box::new(FlattenNestedAnd),
        Box::new(FlattenNestedOr),
        Box::new(EliminateAndTrue),
        Box::new(EliminateOrFalse),
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_ir::{TLExpr, Term};

    // ------------------------------------------------------------------
    // Helper constructors
    // ------------------------------------------------------------------

    fn pred(name: &str) -> TLExpr {
        TLExpr::pred(name, vec![Term::var("x")])
    }

    fn tru() -> TLExpr {
        TLExpr::Constant(1.0)
    }

    fn fal() -> TLExpr {
        TLExpr::Constant(0.0)
    }

    // ------------------------------------------------------------------
    // EliminateDoubleNeg
    // ------------------------------------------------------------------

    #[test]
    fn test_eliminate_double_neg_fires() {
        let inner = pred("p");
        let expr = TLExpr::Not(Box::new(TLExpr::Not(Box::new(inner.clone()))));
        let result = EliminateDoubleNeg.apply(&expr);
        assert_eq!(result, Some(inner));
    }

    #[test]
    fn test_eliminate_double_neg_no_fire() {
        let expr = TLExpr::Not(Box::new(pred("p")));
        assert_eq!(EliminateDoubleNeg.apply(&expr), None);
    }

    #[test]
    fn test_eliminate_double_neg_nested() {
        // Not(Not(Not(x))) — the rule fires at the outer level and removes two
        // layers, leaving Not(x). (The engine applies one rule per node per pass.)
        let x = pred("p");
        let not_x = TLExpr::Not(Box::new(x.clone()));
        let not_not_x = TLExpr::Not(Box::new(not_x.clone()));
        let not_not_not_x = TLExpr::Not(Box::new(not_not_x));
        let result = EliminateDoubleNeg.apply(&not_not_not_x);
        assert_eq!(result, Some(not_x));
    }

    // ------------------------------------------------------------------
    // FlattenNestedAnd
    // ------------------------------------------------------------------

    #[test]
    fn test_flatten_nested_and_fires() {
        let a = pred("a");
        let b = pred("b");
        let c = pred("c");
        let and_ab = TLExpr::And(Box::new(a.clone()), Box::new(b.clone()));
        let and_and_ab_c = TLExpr::And(Box::new(and_ab), Box::new(c.clone()));

        let result = FlattenNestedAnd.apply(&and_and_ab_c);
        let expected = TLExpr::And(Box::new(a), Box::new(TLExpr::And(Box::new(b), Box::new(c))));
        assert_eq!(result, Some(expected));
    }

    #[test]
    fn test_flatten_nested_and_no_fire() {
        let expr = TLExpr::And(Box::new(pred("a")), Box::new(pred("b")));
        assert_eq!(FlattenNestedAnd.apply(&expr), None);
    }

    // ------------------------------------------------------------------
    // FlattenNestedOr
    // ------------------------------------------------------------------

    #[test]
    fn test_flatten_nested_or_fires() {
        let a = pred("a");
        let b = pred("b");
        let c = pred("c");
        let or_ab = TLExpr::Or(Box::new(a.clone()), Box::new(b.clone()));
        let or_or_ab_c = TLExpr::Or(Box::new(or_ab), Box::new(c.clone()));

        let result = FlattenNestedOr.apply(&or_or_ab_c);
        let expected = TLExpr::Or(Box::new(a), Box::new(TLExpr::Or(Box::new(b), Box::new(c))));
        assert_eq!(result, Some(expected));
    }

    // ------------------------------------------------------------------
    // EliminateAndTrue
    // ------------------------------------------------------------------

    #[test]
    fn test_eliminate_and_true_left() {
        let x = pred("x");
        let expr = TLExpr::And(Box::new(tru()), Box::new(x.clone()));
        let result = EliminateAndTrue.apply(&expr);
        assert_eq!(result, Some(x));
    }

    #[test]
    fn test_eliminate_and_true_right() {
        let x = pred("x");
        let expr = TLExpr::And(Box::new(x.clone()), Box::new(tru()));
        let result = EliminateAndTrue.apply(&expr);
        assert_eq!(result, Some(x));
    }

    // ------------------------------------------------------------------
    // EliminateOrFalse
    // ------------------------------------------------------------------

    #[test]
    fn test_eliminate_or_false_left() {
        let x = pred("x");
        let expr = TLExpr::Or(Box::new(fal()), Box::new(x.clone()));
        let result = EliminateOrFalse.apply(&expr);
        assert_eq!(result, Some(x));
    }

    #[test]
    fn test_eliminate_or_false_right() {
        let x = pred("x");
        let expr = TLExpr::Or(Box::new(x.clone()), Box::new(fal()));
        let result = EliminateOrFalse.apply(&expr);
        assert_eq!(result, Some(x));
    }

    // ------------------------------------------------------------------
    // RewriteEngine — basic behaviour
    // ------------------------------------------------------------------

    #[test]
    fn test_rewrite_engine_empty_rules() {
        let expr = TLExpr::Not(Box::new(pred("p")));
        let engine = RewriteEngine::new();
        let (result, stats) = engine.rewrite(expr.clone());
        assert_eq!(result, expr);
        assert_eq!(stats.total_rewrites, 0);
    }

    #[test]
    fn test_rewrite_engine_fixed_point() {
        // Not(Not(x)) should reach fixed point after one application
        let x = pred("p");
        let expr = TLExpr::Not(Box::new(TLExpr::Not(Box::new(x.clone()))));
        let engine = RewriteEngine::new().add_rule(Box::new(EliminateDoubleNeg));
        let (result, stats) = engine.rewrite(expr);
        assert_eq!(result, x);
        assert!(stats.fixed_point_reached);
    }

    #[test]
    fn test_rewrite_engine_stats_record() {
        let expr = TLExpr::Not(Box::new(TLExpr::Not(Box::new(pred("p")))));
        let engine = RewriteEngine::new().add_rule(Box::new(EliminateDoubleNeg));
        let (_result, stats) = engine.rewrite(expr);
        assert!(stats.total_rewrites > 0);
    }

    #[test]
    fn test_rewrite_engine_stats_summary_nonempty() {
        let expr = TLExpr::Not(Box::new(TLExpr::Not(Box::new(pred("p")))));
        let engine = RewriteEngine::new().add_rule(Box::new(EliminateDoubleNeg));
        let (_result, stats) = engine.rewrite(expr);
        let summary = stats.summary();
        assert!(!summary.is_empty());
    }

    // ------------------------------------------------------------------
    // builtin_rules / add_all_builtin_rules
    // ------------------------------------------------------------------

    #[test]
    fn test_builtin_rules_count() {
        assert_eq!(builtin_rules().len(), 5);
    }

    #[test]
    fn test_add_all_builtin_rules() {
        let engine = RewriteEngine::new().add_all_builtin_rules();
        assert_eq!(engine.rules.len(), 5);
    }

    // ------------------------------------------------------------------
    // Edge cases / limits
    // ------------------------------------------------------------------

    #[test]
    fn test_rewrite_engine_iterations_limit() {
        // With max_iterations=1, the engine must not loop infinitely.
        let expr = TLExpr::Not(Box::new(TLExpr::Not(Box::new(pred("p")))));
        let engine = RewriteEngine::new()
            .with_max_iterations(1)
            .add_rule(Box::new(EliminateDoubleNeg));
        let (_result, stats) = engine.rewrite(expr);
        assert!(stats.iterations <= 1);
    }

    // ------------------------------------------------------------------
    // RewriteStats record / tracking
    // ------------------------------------------------------------------

    #[test]
    fn test_rewrite_stats_record_rule() {
        let mut stats = RewriteStats::default();
        stats.record_rule("my_rule");
        assert_eq!(*stats.rules_applied.get("my_rule").unwrap_or(&0), 1);
        assert_eq!(stats.total_rewrites, 1);
    }

    #[test]
    fn test_rewrite_stats_multiple_rules() {
        let mut stats = RewriteStats::default();
        stats.record_rule("rule_a");
        stats.record_rule("rule_b");
        stats.record_rule("rule_a");
        assert_eq!(*stats.rules_applied.get("rule_a").unwrap_or(&0), 2);
        assert_eq!(*stats.rules_applied.get("rule_b").unwrap_or(&0), 1);
        assert_eq!(stats.total_rewrites, 3);
    }

    // ------------------------------------------------------------------
    // Complex cooperation between rules
    // ------------------------------------------------------------------

    #[test]
    fn test_complex_rewrite() {
        // Not(Not(And(x, True))) → x
        // Step 1 (bottom-up): And(x, True) → x    (EliminateAndTrue)
        // Step 2 (outer):     Not(Not(x))  → x    (EliminateDoubleNeg)
        let x = pred("p");
        let and_x_true = TLExpr::And(Box::new(x.clone()), Box::new(tru()));
        let expr = TLExpr::Not(Box::new(TLExpr::Not(Box::new(and_x_true))));

        let engine = RewriteEngine::new().add_all_builtin_rules();
        let (result, stats) = engine.rewrite(expr);
        assert_eq!(result, x);
        assert!(stats.total_rewrites >= 2, "expected at least 2 rewrites");
    }
}
