//! Expression complexity analysis for TLExpr trees.
//!
//! This module provides comprehensive complexity metrics, threshold-based warnings,
//! expression comparison, and batch statistics for analyzing the structural complexity
//! of logical expressions before compilation.

use std::collections::HashSet;
use std::fmt;
use tensorlogic_ir::{TLExpr, Term};

// ---------------------------------------------------------------------------
// ExprComplexity
// ---------------------------------------------------------------------------

/// Comprehensive complexity metrics for a TLExpr tree.
#[derive(Debug, Clone, Default)]
pub struct ExprComplexity {
    /// Total AST nodes in the expression tree.
    pub total_nodes: usize,
    /// Maximum depth of the tree (root = depth 1).
    pub depth: usize,
    /// Maximum number of siblings at any single level.
    pub width: usize,
    /// Number of distinct variable names across the tree.
    pub num_variables: usize,
    /// Number of constant / literal nodes.
    pub num_constants: usize,
    /// Number of predicate application nodes.
    pub num_predicates: usize,
    /// Number of quantifier nodes (ForAll, Exists, soft, counting, etc.).
    pub num_quantifiers: usize,
    /// Maximum nesting depth of quantifiers.
    pub quantifier_depth: usize,
    /// Number of Not / FuzzyNot nodes.
    pub num_negations: usize,
    /// Number of connective nodes (And, Or, Imply, Iff, TNorm, TCoNorm, etc.).
    pub num_connectives: usize,
    /// Number of arithmetic operation nodes.
    pub num_arithmetic: usize,
    /// Number of set operation nodes.
    pub num_set_ops: usize,
    /// Number of let / lambda / fixpoint binding nodes.
    pub num_let_bindings: usize,
    /// Average children per internal (non-leaf) node.
    pub branching_factor: f64,
    /// Ratio of leaf nodes to total nodes.
    pub leaf_ratio: f64,
}

impl ExprComplexity {
    /// Compute complexity metrics for a [`TLExpr`].
    pub fn analyze(expr: &TLExpr) -> Self {
        let mut ctx = AnalysisContext::default();
        let mut level_widths: Vec<usize> = Vec::new();
        Self::visit(expr, 0, 0, &mut ctx, &mut level_widths);

        let max_width = level_widths.iter().copied().max().unwrap_or(1);
        let internal_nodes = ctx.total_nodes.saturating_sub(ctx.leaf_count);
        let branching_factor = if internal_nodes > 0 {
            ctx.total_edges as f64 / internal_nodes as f64
        } else {
            0.0
        };
        let leaf_ratio = if ctx.total_nodes > 0 {
            ctx.leaf_count as f64 / ctx.total_nodes as f64
        } else {
            0.0
        };

        Self {
            total_nodes: ctx.total_nodes,
            depth: ctx.max_depth,
            width: max_width,
            num_variables: ctx.variables.len(),
            num_constants: ctx.num_constants,
            num_predicates: ctx.num_predicates,
            num_quantifiers: ctx.num_quantifiers,
            quantifier_depth: ctx.max_quantifier_depth,
            num_negations: ctx.num_negations,
            num_connectives: ctx.num_connectives,
            num_arithmetic: ctx.num_arithmetic,
            num_set_ops: ctx.num_set_ops,
            num_let_bindings: ctx.num_let_bindings,
            branching_factor,
            leaf_ratio,
        }
    }

    /// A scalar "complexity score" summarising overall complexity (weighted sum).
    pub fn score(&self) -> f64 {
        self.total_nodes as f64 * 1.0
            + self.depth as f64 * 2.0
            + self.quantifier_depth as f64 * 5.0
            + self.num_variables as f64 * 0.5
            + self.num_quantifiers as f64 * 3.0
            + self.num_negations as f64 * 1.0
            + self.num_connectives as f64 * 1.5
            + self.num_arithmetic as f64 * 1.0
            + self.num_set_ops as f64 * 2.0
            + self.num_let_bindings as f64 * 2.0
    }

    /// Returns `true` when the scalar [`Self::score`] is below `threshold`.
    pub fn is_simple(&self, threshold: f64) -> bool {
        self.score() < threshold
    }

    /// Human-readable one-line summary.
    pub fn summary(&self) -> String {
        format!(
            "nodes={}, depth={}, vars={}, quantifiers={} (depth={}), score={:.1}",
            self.total_nodes,
            self.depth,
            self.num_variables,
            self.num_quantifiers,
            self.quantifier_depth,
            self.score(),
        )
    }

    /// Tabular representation of all metrics.
    pub fn format_table(&self) -> String {
        let mut buf = String::new();
        buf.push_str("Metric                | Value\n");
        buf.push_str("----------------------|------\n");
        buf.push_str(&format!("Total nodes           | {}\n", self.total_nodes));
        buf.push_str(&format!("Depth                 | {}\n", self.depth));
        buf.push_str(&format!("Width                 | {}\n", self.width));
        buf.push_str(&format!(
            "Variables (distinct)   | {}\n",
            self.num_variables
        ));
        buf.push_str(&format!("Constants             | {}\n", self.num_constants));
        buf.push_str(&format!(
            "Predicates            | {}\n",
            self.num_predicates
        ));
        buf.push_str(&format!(
            "Quantifiers           | {}\n",
            self.num_quantifiers
        ));
        buf.push_str(&format!(
            "Quantifier depth      | {}\n",
            self.quantifier_depth
        ));
        buf.push_str(&format!("Negations             | {}\n", self.num_negations));
        buf.push_str(&format!(
            "Connectives           | {}\n",
            self.num_connectives
        ));
        buf.push_str(&format!(
            "Arithmetic ops        | {}\n",
            self.num_arithmetic
        ));
        buf.push_str(&format!("Set ops               | {}\n", self.num_set_ops));
        buf.push_str(&format!(
            "Let/Lambda/Fixpoint   | {}\n",
            self.num_let_bindings
        ));
        buf.push_str(&format!(
            "Branching factor      | {:.3}\n",
            self.branching_factor
        ));
        buf.push_str(&format!("Leaf ratio            | {:.3}\n", self.leaf_ratio));
        buf.push_str(&format!("Complexity score      | {:.1}\n", self.score()));
        buf
    }

    // ------------------------------------------------------------------
    // Recursive visitor
    // ------------------------------------------------------------------

    fn visit(
        expr: &TLExpr,
        depth: usize,
        quantifier_depth: usize,
        ctx: &mut AnalysisContext,
        level_widths: &mut Vec<usize>,
    ) {
        ctx.total_nodes += 1;
        let current_depth = depth + 1;
        if current_depth > ctx.max_depth {
            ctx.max_depth = current_depth;
        }

        // Ensure level_widths has an entry for this depth.
        while level_widths.len() <= depth {
            level_widths.push(0);
        }

        // Classify and recurse.
        match expr {
            // -- Leaves --
            TLExpr::Constant(_) => {
                ctx.num_constants += 1;
                ctx.leaf_count += 1;
                level_widths[depth] += 1;
            }
            TLExpr::EmptySet => {
                ctx.num_constants += 1;
                ctx.leaf_count += 1;
                level_widths[depth] += 1;
            }
            TLExpr::Nominal { .. } => {
                ctx.leaf_count += 1;
                level_widths[depth] += 1;
            }
            TLExpr::Abducible { .. } => {
                ctx.leaf_count += 1;
                level_widths[depth] += 1;
            }
            TLExpr::AllDifferent { variables } => {
                // Leaf-like: just collects variable names.
                for v in variables {
                    ctx.variables.insert(v.clone());
                }
                ctx.leaf_count += 1;
                level_widths[depth] += 1;
            }

            // -- Predicates --
            TLExpr::Pred { args, .. } => {
                ctx.num_predicates += 1;
                // Collect variables from term args.
                for term in args {
                    Self::collect_term_vars(term, ctx);
                }
                ctx.leaf_count += 1; // Pred is a leaf in the *expression* tree.
                level_widths[depth] += 1;
            }

            // -- Negation --
            TLExpr::Not(inner) => {
                ctx.num_negations += 1;
                ctx.total_edges += 1;
                Self::visit(inner, current_depth, quantifier_depth, ctx, level_widths);
            }
            TLExpr::FuzzyNot { expr: inner, .. } => {
                ctx.num_negations += 1;
                ctx.total_edges += 1;
                Self::visit(inner, current_depth, quantifier_depth, ctx, level_widths);
            }

            // -- Connectives (binary logical) --
            TLExpr::And(l, r) | TLExpr::Or(l, r) | TLExpr::Imply(l, r) => {
                ctx.num_connectives += 1;
                ctx.total_edges += 2;
                Self::visit(l, current_depth, quantifier_depth, ctx, level_widths);
                Self::visit(r, current_depth, quantifier_depth, ctx, level_widths);
            }
            TLExpr::TNorm { left, right, .. } | TLExpr::TCoNorm { left, right, .. } => {
                ctx.num_connectives += 1;
                ctx.total_edges += 2;
                Self::visit(left, current_depth, quantifier_depth, ctx, level_widths);
                Self::visit(right, current_depth, quantifier_depth, ctx, level_widths);
            }
            TLExpr::FuzzyImplication {
                premise,
                conclusion,
                ..
            } => {
                ctx.num_connectives += 1;
                ctx.total_edges += 2;
                Self::visit(premise, current_depth, quantifier_depth, ctx, level_widths);
                Self::visit(
                    conclusion,
                    current_depth,
                    quantifier_depth,
                    ctx,
                    level_widths,
                );
            }

            // -- Quantifiers (standard) --
            TLExpr::ForAll { var, body, .. } | TLExpr::Exists { var, body, .. } => {
                ctx.num_quantifiers += 1;
                ctx.variables.insert(var.clone());
                let new_qd = quantifier_depth + 1;
                if new_qd > ctx.max_quantifier_depth {
                    ctx.max_quantifier_depth = new_qd;
                }
                ctx.total_edges += 1;
                Self::visit(body, current_depth, new_qd, ctx, level_widths);
            }

            // -- Soft quantifiers --
            TLExpr::SoftExists { var, body, .. } | TLExpr::SoftForAll { var, body, .. } => {
                ctx.num_quantifiers += 1;
                ctx.variables.insert(var.clone());
                let new_qd = quantifier_depth + 1;
                if new_qd > ctx.max_quantifier_depth {
                    ctx.max_quantifier_depth = new_qd;
                }
                ctx.total_edges += 1;
                Self::visit(body, current_depth, new_qd, ctx, level_widths);
            }

            // -- Counting quantifiers --
            TLExpr::CountingExists { var, body, .. }
            | TLExpr::CountingForAll { var, body, .. }
            | TLExpr::ExactCount { var, body, .. }
            | TLExpr::Majority { var, body, .. } => {
                ctx.num_quantifiers += 1;
                ctx.variables.insert(var.clone());
                let new_qd = quantifier_depth + 1;
                if new_qd > ctx.max_quantifier_depth {
                    ctx.max_quantifier_depth = new_qd;
                }
                ctx.total_edges += 1;
                Self::visit(body, current_depth, new_qd, ctx, level_widths);
            }

            // -- Aggregate (quantifier-like) --
            TLExpr::Aggregate { var, body, .. } => {
                ctx.num_quantifiers += 1;
                ctx.variables.insert(var.clone());
                let new_qd = quantifier_depth + 1;
                if new_qd > ctx.max_quantifier_depth {
                    ctx.max_quantifier_depth = new_qd;
                }
                ctx.total_edges += 1;
                Self::visit(body, current_depth, new_qd, ctx, level_widths);
            }

            // -- Arithmetic (binary) --
            TLExpr::Add(l, r)
            | TLExpr::Sub(l, r)
            | TLExpr::Mul(l, r)
            | TLExpr::Div(l, r)
            | TLExpr::Pow(l, r)
            | TLExpr::Mod(l, r)
            | TLExpr::Min(l, r)
            | TLExpr::Max(l, r) => {
                ctx.num_arithmetic += 1;
                ctx.total_edges += 2;
                Self::visit(l, current_depth, quantifier_depth, ctx, level_widths);
                Self::visit(r, current_depth, quantifier_depth, ctx, level_widths);
            }

            // -- Arithmetic (unary math) --
            TLExpr::Abs(inner)
            | TLExpr::Floor(inner)
            | TLExpr::Ceil(inner)
            | TLExpr::Round(inner)
            | TLExpr::Sqrt(inner)
            | TLExpr::Exp(inner)
            | TLExpr::Log(inner)
            | TLExpr::Sin(inner)
            | TLExpr::Cos(inner)
            | TLExpr::Tan(inner) => {
                ctx.num_arithmetic += 1;
                ctx.total_edges += 1;
                Self::visit(inner, current_depth, quantifier_depth, ctx, level_widths);
            }

            // -- Comparison (binary) --
            TLExpr::Eq(l, r)
            | TLExpr::Lt(l, r)
            | TLExpr::Gt(l, r)
            | TLExpr::Lte(l, r)
            | TLExpr::Gte(l, r) => {
                ctx.num_connectives += 1; // comparisons are logical connectives
                ctx.total_edges += 2;
                Self::visit(l, current_depth, quantifier_depth, ctx, level_widths);
                Self::visit(r, current_depth, quantifier_depth, ctx, level_widths);
            }

            // -- Conditional --
            TLExpr::IfThenElse {
                condition,
                then_branch,
                else_branch,
            } => {
                ctx.total_edges += 3;
                Self::visit(
                    condition,
                    current_depth,
                    quantifier_depth,
                    ctx,
                    level_widths,
                );
                Self::visit(
                    then_branch,
                    current_depth,
                    quantifier_depth,
                    ctx,
                    level_widths,
                );
                Self::visit(
                    else_branch,
                    current_depth,
                    quantifier_depth,
                    ctx,
                    level_widths,
                );
            }

            // -- Let / Lambda / Fixpoint --
            TLExpr::Let {
                var, value, body, ..
            } => {
                ctx.num_let_bindings += 1;
                ctx.variables.insert(var.clone());
                ctx.total_edges += 2;
                Self::visit(value, current_depth, quantifier_depth, ctx, level_widths);
                Self::visit(body, current_depth, quantifier_depth, ctx, level_widths);
            }
            TLExpr::Lambda { var, body, .. } => {
                ctx.num_let_bindings += 1;
                ctx.variables.insert(var.clone());
                ctx.total_edges += 1;
                Self::visit(body, current_depth, quantifier_depth, ctx, level_widths);
            }
            TLExpr::LeastFixpoint { var, body, .. }
            | TLExpr::GreatestFixpoint { var, body, .. } => {
                ctx.num_let_bindings += 1;
                ctx.variables.insert(var.clone());
                ctx.total_edges += 1;
                Self::visit(body, current_depth, quantifier_depth, ctx, level_widths);
            }

            // -- Apply (higher-order) --
            TLExpr::Apply {
                function, argument, ..
            } => {
                ctx.total_edges += 2;
                Self::visit(function, current_depth, quantifier_depth, ctx, level_widths);
                Self::visit(argument, current_depth, quantifier_depth, ctx, level_widths);
            }

            // -- Set operations --
            TLExpr::SetUnion { left, right }
            | TLExpr::SetIntersection { left, right }
            | TLExpr::SetDifference { left, right } => {
                ctx.num_set_ops += 1;
                ctx.total_edges += 2;
                Self::visit(left, current_depth, quantifier_depth, ctx, level_widths);
                Self::visit(right, current_depth, quantifier_depth, ctx, level_widths);
            }
            TLExpr::SetMembership { element, set } => {
                ctx.num_set_ops += 1;
                ctx.total_edges += 2;
                Self::visit(element, current_depth, quantifier_depth, ctx, level_widths);
                Self::visit(set, current_depth, quantifier_depth, ctx, level_widths);
            }
            TLExpr::SetCardinality { set } => {
                ctx.num_set_ops += 1;
                ctx.total_edges += 1;
                Self::visit(set, current_depth, quantifier_depth, ctx, level_widths);
            }
            TLExpr::SetComprehension { var, condition, .. } => {
                ctx.num_set_ops += 1;
                ctx.variables.insert(var.clone());
                ctx.total_edges += 1;
                Self::visit(
                    condition,
                    current_depth,
                    quantifier_depth,
                    ctx,
                    level_widths,
                );
            }

            // -- Modal logic --
            TLExpr::Box(inner) | TLExpr::Diamond(inner) => {
                ctx.num_connectives += 1;
                ctx.total_edges += 1;
                Self::visit(inner, current_depth, quantifier_depth, ctx, level_widths);
            }

            // -- Temporal logic (unary) --
            TLExpr::Next(inner) | TLExpr::Eventually(inner) | TLExpr::Always(inner) => {
                ctx.num_connectives += 1;
                ctx.total_edges += 1;
                Self::visit(inner, current_depth, quantifier_depth, ctx, level_widths);
            }

            // -- Temporal logic (binary) --
            TLExpr::Until { before, after }
            | TLExpr::Release {
                released: before,
                releaser: after,
            }
            | TLExpr::WeakUntil { before, after }
            | TLExpr::StrongRelease {
                released: before,
                releaser: after,
            } => {
                ctx.num_connectives += 1;
                ctx.total_edges += 2;
                Self::visit(before, current_depth, quantifier_depth, ctx, level_widths);
                Self::visit(after, current_depth, quantifier_depth, ctx, level_widths);
            }

            // -- Score --
            TLExpr::Score(inner) => {
                ctx.total_edges += 1;
                Self::visit(inner, current_depth, quantifier_depth, ctx, level_widths);
            }

            // -- Weighted rule --
            TLExpr::WeightedRule { rule, .. } => {
                ctx.total_edges += 1;
                Self::visit(rule, current_depth, quantifier_depth, ctx, level_widths);
            }

            // -- Probabilistic choice --
            TLExpr::ProbabilisticChoice { alternatives } => {
                ctx.total_edges += alternatives.len();
                for (_prob, alt_expr) in alternatives {
                    Self::visit(alt_expr, current_depth, quantifier_depth, ctx, level_widths);
                }
            }

            // -- Hybrid logic --
            TLExpr::At { formula, .. } => {
                ctx.total_edges += 1;
                Self::visit(formula, current_depth, quantifier_depth, ctx, level_widths);
            }
            TLExpr::Somewhere { formula } | TLExpr::Everywhere { formula } => {
                ctx.num_connectives += 1;
                ctx.total_edges += 1;
                Self::visit(formula, current_depth, quantifier_depth, ctx, level_widths);
            }

            // -- Constraint programming (GlobalCardinality) --
            TLExpr::GlobalCardinality {
                variables, values, ..
            } => {
                for v in variables {
                    ctx.variables.insert(v.clone());
                }
                ctx.total_edges += values.len();
                for val_expr in values {
                    Self::visit(val_expr, current_depth, quantifier_depth, ctx, level_widths);
                }
            }

            // -- Explain (abductive reasoning) --
            TLExpr::Explain { formula } => {
                ctx.total_edges += 1;
                Self::visit(formula, current_depth, quantifier_depth, ctx, level_widths);
            }

            // -- Symbol literal --
            TLExpr::SymbolLiteral(_) => {
                ctx.leaf_count += 1;
                level_widths[depth] += 1;
            }

            // -- Pattern matching --
            TLExpr::Match { scrutinee, arms } => {
                ctx.total_edges += 1 + arms.len();
                Self::visit(
                    scrutinee,
                    current_depth,
                    quantifier_depth,
                    ctx,
                    level_widths,
                );
                for (_, body) in arms {
                    Self::visit(body, current_depth, quantifier_depth, ctx, level_widths);
                }
            }
        }
    }

    /// Collect variable names from a [`Term`].
    fn collect_term_vars(term: &Term, ctx: &mut AnalysisContext) {
        match term {
            Term::Var(name) => {
                ctx.variables.insert(name.clone());
            }
            Term::Const(_) => {
                ctx.num_constants += 1;
            }
            Term::Typed { value, .. } => {
                Self::collect_term_vars(value, ctx);
            }
        }
    }
}

impl fmt::Display for ExprComplexity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

// ---------------------------------------------------------------------------
// Internal analysis context
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
struct AnalysisContext {
    total_nodes: usize,
    max_depth: usize,
    max_quantifier_depth: usize,
    leaf_count: usize,
    total_edges: usize,
    variables: HashSet<String>,
    num_constants: usize,
    num_predicates: usize,
    num_quantifiers: usize,
    num_negations: usize,
    num_connectives: usize,
    num_arithmetic: usize,
    num_set_ops: usize,
    num_let_bindings: usize,
}

// ---------------------------------------------------------------------------
// Thresholds & warnings
// ---------------------------------------------------------------------------

/// Configurable thresholds for complexity warnings.
#[derive(Debug, Clone)]
pub struct ComplexityThresholds {
    pub max_depth: usize,
    pub max_nodes: usize,
    pub max_quantifier_depth: usize,
    pub max_variables: usize,
    pub max_branching_factor: f64,
}

impl Default for ComplexityThresholds {
    fn default() -> Self {
        Self {
            max_depth: 50,
            max_nodes: 10000,
            max_quantifier_depth: 10,
            max_variables: 100,
            max_branching_factor: 10.0,
        }
    }
}

/// A complexity warning produced by [`check_complexity`].
#[derive(Debug, Clone)]
pub struct ComplexityWarning {
    pub metric: String,
    pub value: f64,
    pub threshold: f64,
    pub severity: WarningSeverity,
    pub message: String,
}

impl fmt::Display for ComplexityWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:?}] {}", self.severity, self.message)
    }
}

/// Severity level for a complexity warning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WarningSeverity {
    Info,
    Warning,
    Critical,
}

/// Check complexity metrics against thresholds and return any warnings.
pub fn check_complexity(
    complexity: &ExprComplexity,
    thresholds: &ComplexityThresholds,
) -> Vec<ComplexityWarning> {
    let mut warnings = Vec::new();

    let checks: Vec<(&str, f64, f64)> = vec![
        (
            "depth",
            complexity.depth as f64,
            thresholds.max_depth as f64,
        ),
        (
            "total_nodes",
            complexity.total_nodes as f64,
            thresholds.max_nodes as f64,
        ),
        (
            "quantifier_depth",
            complexity.quantifier_depth as f64,
            thresholds.max_quantifier_depth as f64,
        ),
        (
            "num_variables",
            complexity.num_variables as f64,
            thresholds.max_variables as f64,
        ),
        (
            "branching_factor",
            complexity.branching_factor,
            thresholds.max_branching_factor,
        ),
    ];

    for (metric, value, threshold) in checks {
        if value > threshold {
            let ratio = value / threshold;
            let severity = if ratio > 2.0 {
                WarningSeverity::Critical
            } else if ratio > 1.5 {
                WarningSeverity::Warning
            } else {
                WarningSeverity::Info
            };
            warnings.push(ComplexityWarning {
                metric: metric.to_string(),
                value,
                threshold,
                severity,
                message: format!(
                    "{} ({:.0}) exceeds threshold ({:.0})",
                    metric, value, threshold
                ),
            });
        }
    }

    warnings
}

// ---------------------------------------------------------------------------
// ComplexityComparison
// ---------------------------------------------------------------------------

/// Side-by-side comparison of complexity between two expressions.
#[derive(Debug, Clone)]
pub struct ComplexityComparison {
    pub before: ExprComplexity,
    pub after: ExprComplexity,
    pub node_delta: i64,
    pub depth_delta: i64,
    pub score_delta: f64,
    /// `true` when the *after* expression is simpler (lower score).
    pub improved: bool,
}

impl ComplexityComparison {
    /// Compare complexity of two expressions (`before` → `after`).
    pub fn compare(before: &TLExpr, after: &TLExpr) -> Self {
        let b = ExprComplexity::analyze(before);
        let a = ExprComplexity::analyze(after);
        let node_delta = a.total_nodes as i64 - b.total_nodes as i64;
        let depth_delta = a.depth as i64 - b.depth as i64;
        let score_delta = a.score() - b.score();
        let improved = a.score() < b.score();
        Self {
            before: b,
            after: a,
            node_delta,
            depth_delta,
            score_delta,
            improved,
        }
    }

    /// Human-readable summary of the comparison.
    pub fn summary(&self) -> String {
        let direction = if self.improved {
            "improved"
        } else if self.score_delta.abs() < f64::EPSILON {
            "unchanged"
        } else {
            "regressed"
        };
        format!(
            "Complexity {}: nodes {} -> {} ({:+}), depth {} -> {} ({:+}), score {:.1} -> {:.1} ({:+.1})",
            direction,
            self.before.total_nodes,
            self.after.total_nodes,
            self.node_delta,
            self.before.depth,
            self.after.depth,
            self.depth_delta,
            self.before.score(),
            self.after.score(),
            self.score_delta,
        )
    }
}

impl fmt::Display for ComplexityComparison {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

// ---------------------------------------------------------------------------
// BatchComplexityStats
// ---------------------------------------------------------------------------

/// Aggregate complexity statistics for a batch of expressions.
#[derive(Debug, Clone)]
pub struct BatchComplexityStats {
    pub count: usize,
    pub avg_nodes: f64,
    pub avg_depth: f64,
    pub max_nodes: usize,
    pub max_depth: usize,
    pub avg_score: f64,
    pub above_threshold_count: usize,
}

impl BatchComplexityStats {
    /// Compute aggregate stats for a slice of expressions.
    pub fn from_exprs(exprs: &[TLExpr], thresholds: &ComplexityThresholds) -> Self {
        if exprs.is_empty() {
            return Self {
                count: 0,
                avg_nodes: 0.0,
                avg_depth: 0.0,
                max_nodes: 0,
                max_depth: 0,
                avg_score: 0.0,
                above_threshold_count: 0,
            };
        }

        let metrics: Vec<ExprComplexity> = exprs.iter().map(ExprComplexity::analyze).collect();
        let count = metrics.len();
        let total_nodes_sum: usize = metrics.iter().map(|m| m.total_nodes).sum();
        let total_depth_sum: usize = metrics.iter().map(|m| m.depth).sum();
        let total_score_sum: f64 = metrics.iter().map(|m| m.score()).sum();
        let max_nodes = metrics.iter().map(|m| m.total_nodes).max().unwrap_or(0);
        let max_depth = metrics.iter().map(|m| m.depth).max().unwrap_or(0);
        let above_threshold_count = metrics
            .iter()
            .filter(|m| !check_complexity(m, thresholds).is_empty())
            .count();

        Self {
            count,
            avg_nodes: total_nodes_sum as f64 / count as f64,
            avg_depth: total_depth_sum as f64 / count as f64,
            max_nodes,
            max_depth,
            avg_score: total_score_sum / count as f64,
            above_threshold_count,
        }
    }

    /// Human-readable summary of batch statistics.
    pub fn summary(&self) -> String {
        format!(
            "Batch of {} exprs: avg nodes={:.1}, avg depth={:.1}, max nodes={}, max depth={}, avg score={:.1}, {} above threshold",
            self.count,
            self.avg_nodes,
            self.avg_depth,
            self.max_nodes,
            self.max_depth,
            self.avg_score,
            self.above_threshold_count,
        )
    }
}

impl fmt::Display for BatchComplexityStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_ir::{TLExpr, Term};

    fn var_expr(name: &str) -> TLExpr {
        TLExpr::pred(name, vec![Term::var(name)])
    }

    #[test]
    fn test_leaf_node_complexity() {
        let expr = TLExpr::Constant(1.0);
        let c = ExprComplexity::analyze(&expr);
        assert_eq!(c.total_nodes, 1);
        assert_eq!(c.depth, 1);
    }

    #[test]
    fn test_not_depth() {
        let expr = TLExpr::negate(TLExpr::Constant(1.0));
        let c = ExprComplexity::analyze(&expr);
        assert_eq!(c.depth, 2);
    }

    #[test]
    fn test_and_node_count() {
        let expr = TLExpr::and(TLExpr::Constant(1.0), TLExpr::Constant(2.0));
        let c = ExprComplexity::analyze(&expr);
        assert_eq!(c.total_nodes, 3);
    }

    #[test]
    fn test_quantifier_counted() {
        let body = var_expr("x");
        let expr = TLExpr::forall("x", "D", body);
        let c = ExprComplexity::analyze(&expr);
        assert_eq!(c.num_quantifiers, 1);
    }

    #[test]
    fn test_quantifier_depth_nested() {
        let inner = TLExpr::exists("y", "D", var_expr("y"));
        let expr = TLExpr::forall("x", "D", inner);
        let c = ExprComplexity::analyze(&expr);
        assert_eq!(c.quantifier_depth, 2);
    }

    #[test]
    fn test_negation_counted() {
        let expr = TLExpr::negate(TLExpr::Constant(1.0));
        let c = ExprComplexity::analyze(&expr);
        assert_eq!(c.num_negations, 1);
    }

    #[test]
    fn test_connective_counted() {
        let expr = TLExpr::and(TLExpr::Constant(1.0), TLExpr::Constant(2.0));
        let c = ExprComplexity::analyze(&expr);
        assert_eq!(c.num_connectives, 1);
    }

    #[test]
    fn test_distinct_variables() {
        // And(Pred("x", [Var("x")]), Pred("x", [Var("x")]))
        let expr = TLExpr::and(var_expr("x"), var_expr("x"));
        let c = ExprComplexity::analyze(&expr);
        assert_eq!(c.num_variables, 1);
    }

    #[test]
    fn test_branching_factor_binary() {
        // And(Const, Const) => 1 internal node with 2 children => bf = 2.0
        let expr = TLExpr::and(TLExpr::Constant(1.0), TLExpr::Constant(2.0));
        let c = ExprComplexity::analyze(&expr);
        assert!(c.branching_factor >= 1.0);
        assert!(c.branching_factor <= 2.5);
    }

    #[test]
    fn test_leaf_ratio() {
        // And(Const, Const) = 3 nodes, 2 leaves => 2/3 ≈ 0.667
        let expr = TLExpr::and(TLExpr::Constant(1.0), TLExpr::Constant(2.0));
        let c = ExprComplexity::analyze(&expr);
        let expected = 2.0 / 3.0;
        assert!((c.leaf_ratio - expected).abs() < 0.01);
    }

    #[test]
    fn test_score_increases_with_complexity() {
        let simple = TLExpr::Constant(1.0);
        let complex = TLExpr::forall(
            "x",
            "D",
            TLExpr::exists("y", "D", TLExpr::and(var_expr("x"), var_expr("y"))),
        );
        let s1 = ExprComplexity::analyze(&simple).score();
        let s2 = ExprComplexity::analyze(&complex).score();
        assert!(s2 > s1);
    }

    #[test]
    fn test_is_simple_true() {
        let expr = TLExpr::Constant(42.0);
        let c = ExprComplexity::analyze(&expr);
        assert!(c.is_simple(100.0));
    }

    #[test]
    fn test_is_simple_false() {
        // Build a moderately complex expression.
        let mut expr = var_expr("x");
        for i in 0..20 {
            expr = TLExpr::forall(format!("v{}", i), "D", expr);
        }
        let c = ExprComplexity::analyze(&expr);
        assert!(!c.is_simple(10.0));
    }

    #[test]
    fn test_summary_nonempty() {
        let c = ExprComplexity::analyze(&TLExpr::Constant(1.0));
        let s = c.summary();
        assert!(!s.is_empty());
        assert!(s.contains("nodes="));
    }

    #[test]
    fn test_format_table_has_header() {
        let c = ExprComplexity::analyze(&TLExpr::Constant(1.0));
        let table = c.format_table();
        assert!(table.contains("Metric"));
        assert!(table.contains("Value"));
        assert!(table.contains("Total nodes"));
    }

    #[test]
    fn test_thresholds_default() {
        let t = ComplexityThresholds::default();
        assert_eq!(t.max_depth, 50);
        assert_eq!(t.max_nodes, 10000);
        assert_eq!(t.max_quantifier_depth, 10);
        assert_eq!(t.max_variables, 100);
        assert!((t.max_branching_factor - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_check_complexity_no_warnings() {
        let c = ExprComplexity::analyze(&TLExpr::Constant(1.0));
        let warnings = check_complexity(&c, &ComplexityThresholds::default());
        assert!(warnings.is_empty());
    }

    #[test]
    fn test_check_complexity_with_warning() {
        // Build an expression deeper than the threshold (max_depth = 3).
        let mut expr = TLExpr::Constant(1.0);
        for _ in 0..10 {
            expr = TLExpr::negate(expr);
        }
        let c = ExprComplexity::analyze(&expr);
        let thresholds = ComplexityThresholds {
            max_depth: 3,
            ..ComplexityThresholds::default()
        };
        let warnings = check_complexity(&c, &thresholds);
        assert!(!warnings.is_empty());
        assert!(warnings.iter().any(|w| w.metric == "depth"));
    }

    #[test]
    fn test_complexity_comparison_improved() {
        let complex = TLExpr::and(
            TLExpr::forall("x", "D", var_expr("x")),
            TLExpr::exists("y", "D", var_expr("y")),
        );
        let simple = TLExpr::Constant(1.0);
        let cmp = ComplexityComparison::compare(&complex, &simple);
        assert!(cmp.improved);
        assert!(cmp.node_delta < 0);
    }

    #[test]
    fn test_batch_stats_avg() {
        let exprs = vec![
            TLExpr::Constant(1.0),
            TLExpr::Constant(2.0),
            TLExpr::and(TLExpr::Constant(3.0), TLExpr::Constant(4.0)),
        ];
        let stats = BatchComplexityStats::from_exprs(&exprs, &ComplexityThresholds::default());
        assert_eq!(stats.count, 3);
        // 1 + 1 + 3 = 5, avg = 5/3 ≈ 1.667
        let expected_avg = 5.0 / 3.0;
        assert!((stats.avg_nodes - expected_avg).abs() < 0.01);
    }
}
