//! Full compiler optimization pipeline for TLExpr expressions.
//!
//! This module provides a top-level [`CompilerPipeline`] that chains **all** compiler
//! passes in configurable order:
//!
//! `ConstProp → DeadCode → Inline → [Algebraic / OptimizationPipeline] → Rewrite`
//!
//! Unlike the algebraic-only [`crate::optimize::OptimizationPipeline`], the
//! `CompilerPipeline` integrates the newer passes added in v0.1.11–v0.1.16
//! (constant propagation, dead-code elimination, let-inlining, pattern-rewriting)
//! and supports outer fixed-point iteration across all passes.
//!
//! # Quick Start
//!
//! ```rust
//! use tensorlogic_compiler::pipeline::{CompilerPipeline, CompilerPipelineConfig};
//! use tensorlogic_ir::TLExpr;
//!
//! let pipeline = CompilerPipeline::with_default();
//! let expr = TLExpr::add(
//!     TLExpr::mul(TLExpr::Constant(2.0), TLExpr::Constant(3.0)),
//!     TLExpr::Constant(0.0),
//! );
//! let result = pipeline.run(expr);
//! println!("{}", result.stats.summary());
//! ```

use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

use tensorlogic_ir::TLExpr;

use crate::const_prop::{ConstPropConfig, ConstantPropagator};
use crate::dead_code::{DceConfig, DeadCodeEliminator};
use crate::inline::{InlineConfig, LetInliner};
use crate::optimize::OptimizationPipeline;
use crate::rewrite::RewriteEngine;

// ────────────────────────────────────────────────────────────────────────────
// CompilerPassId
// ────────────────────────────────────────────────────────────────────────────

/// Identifies a single pass in the compiler pipeline.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CompilerPassId {
    /// Constant propagation — evaluates constant sub-expressions at compile time.
    ConstProp,
    /// Dead code elimination — removes unreachable branches and unused bindings.
    DeadCode,
    /// Let-inlining — substitutes `Let`-bound variables into their use sites.
    Inline,
    /// Algebraic optimization pipeline (negation, folding, strength-reduction, …).
    Algebraic,
    /// Pattern-rewriting engine — applies structural rewrite rules to fixed point.
    Rewrite,
}

impl fmt::Display for CompilerPassId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompilerPassId::ConstProp => write!(f, "ConstProp"),
            CompilerPassId::DeadCode => write!(f, "DeadCode"),
            CompilerPassId::Inline => write!(f, "Inline"),
            CompilerPassId::Algebraic => write!(f, "Algebraic"),
            CompilerPassId::Rewrite => write!(f, "Rewrite"),
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// CompilerPassOrder
// ────────────────────────────────────────────────────────────────────────────

/// Canonical ordering in which passes are applied during one outer iteration.
#[derive(Debug, Clone)]
pub enum CompilerPassOrder {
    /// `ConstProp → DCE → Inline → Algebraic → Rewrite`
    ///
    /// The default, well-balanced ordering.  Constant folding and DCE simplify
    /// the tree before inlining; algebraic passes follow; pattern rewrites last.
    CanonicalOrder,

    /// `Inline → ConstProp → DCE → Algebraic → Rewrite`
    ///
    /// Inlining first exposes more constant sub-expressions for subsequent
    /// folding and elimination passes.
    InlineFirst,

    /// `ConstProp → ConstProp → DCE → Inline → ConstProp → DCE → Rewrite`
    ///
    /// Runs constant propagation twice before elimination, then once more after
    /// inlining to catch any newly-exposed constants.  Useful for deep algebraic
    /// expressions with many nested constants.
    AggressiveFold,

    /// User-supplied pass sequence.
    Custom(Vec<CompilerPassId>),
}

impl CompilerPassOrder {
    /// Convert this ordering variant to the concrete list of passes to execute.
    pub fn to_pass_list(&self) -> Vec<CompilerPassId> {
        match self {
            CompilerPassOrder::CanonicalOrder => vec![
                CompilerPassId::ConstProp,
                CompilerPassId::DeadCode,
                CompilerPassId::Inline,
                CompilerPassId::Algebraic,
                CompilerPassId::Rewrite,
            ],
            CompilerPassOrder::InlineFirst => vec![
                CompilerPassId::Inline,
                CompilerPassId::ConstProp,
                CompilerPassId::DeadCode,
                CompilerPassId::Algebraic,
                CompilerPassId::Rewrite,
            ],
            CompilerPassOrder::AggressiveFold => vec![
                CompilerPassId::ConstProp,
                CompilerPassId::ConstProp,
                CompilerPassId::DeadCode,
                CompilerPassId::Inline,
                CompilerPassId::ConstProp,
                CompilerPassId::DeadCode,
                CompilerPassId::Rewrite,
            ],
            CompilerPassOrder::Custom(order) => order.clone(),
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// CompilerPipelineConfig
// ────────────────────────────────────────────────────────────────────────────

/// Controls which passes are enabled and how the full pipeline is configured.
#[derive(Debug, Clone)]
pub struct CompilerPipelineConfig {
    /// Enable the constant-propagation pass.
    pub enable_const_prop: bool,
    /// Enable the dead-code-elimination pass.
    pub enable_dead_code: bool,
    /// Enable the let-inlining pass.
    pub enable_inline: bool,
    /// Enable the algebraic optimization sub-pipeline.
    pub enable_algebraic: bool,
    /// Enable the pattern-rewrite engine.
    pub enable_rewrite: bool,
    /// Order in which passes are applied within a single outer iteration.
    pub pass_order: CompilerPassOrder,
    /// Maximum number of outer fixed-point iterations over the full pass sequence.
    pub max_outer_iterations: u32,
    /// Configuration forwarded to the constant-propagation pass.
    pub const_prop_config: ConstPropConfig,
    /// Configuration forwarded to the dead-code-elimination pass.
    pub dce_config: DceConfig,
    /// Configuration forwarded to the let-inlining pass.
    pub inline_config: InlineConfig,
}

impl Default for CompilerPipelineConfig {
    fn default() -> Self {
        Self {
            enable_const_prop: true,
            enable_dead_code: true,
            enable_inline: true,
            enable_algebraic: true,
            enable_rewrite: true,
            pass_order: CompilerPassOrder::CanonicalOrder,
            max_outer_iterations: 3,
            const_prop_config: ConstPropConfig::default(),
            dce_config: DceConfig::default(),
            inline_config: InlineConfig::default(),
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// CompilerPassStats
// ────────────────────────────────────────────────────────────────────────────

/// Timing and reduction statistics for a single pass execution.
#[derive(Debug, Clone)]
pub struct CompilerPassStats {
    /// Which pass produced these stats.
    pub pass_id: CompilerPassId,
    /// Wall-clock time spent in this pass.
    pub wall_time: Duration,
    /// Node count immediately before the pass ran.
    pub nodes_before: u64,
    /// Node count immediately after the pass completed.
    pub nodes_after: u64,
    /// Number of reductions (folds, eliminations, inlines, rewrites, …).
    pub reductions: u64,
}

impl CompilerPassStats {
    /// Fraction of nodes eliminated by this pass: `(before − after) / before * 100`.
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

    /// Human-readable one-line summary of this pass execution.
    pub fn summary(&self) -> String {
        format!(
            "{:<12} {:>8.3}ms  nodes: {:>6} → {:>6} ({:>5.1}%)  reductions: {}",
            self.pass_id.to_string(),
            self.wall_time.as_secs_f64() * 1_000.0,
            self.nodes_before,
            self.nodes_after,
            self.reduction_pct(),
            self.reductions,
        )
    }
}

// ────────────────────────────────────────────────────────────────────────────
// CompilerPipelineStats
// ────────────────────────────────────────────────────────────────────────────

/// Aggregate statistics for an entire pipeline run.
#[derive(Debug, Clone)]
pub struct CompilerPipelineStats {
    /// Per-pass statistics, in execution order.
    pub pass_stats: Vec<CompilerPassStats>,
    /// Total wall-clock time for the full pipeline run.
    pub total_wall_time: Duration,
    /// Number of outer fixed-point iterations executed.
    pub outer_iterations: u32,
    /// Total node reduction across all passes: `initial − final` (may be negative
    /// if a pass somehow increases node count, which should not happen in practice).
    pub total_node_reduction: i64,
    /// Node count before the very first pass.
    pub initial_node_count: u64,
    /// Node count after the very last pass.
    pub final_node_count: u64,
}

impl CompilerPipelineStats {
    /// Fraction of nodes eliminated across the entire pipeline run.
    ///
    /// Returns `0.0` when `initial_node_count == 0`.
    pub fn overall_reduction_pct(&self) -> f64 {
        if self.initial_node_count == 0 {
            return 0.0;
        }
        let before = self.initial_node_count as f64;
        let after = self.final_node_count as f64;
        ((before - after) / before * 100.0).max(0.0)
    }

    /// Returns the pass that consumed the most wall-clock time, or `None` if no
    /// passes were executed.
    pub fn slowest_pass(&self) -> Option<&CompilerPassStats> {
        self.pass_stats.iter().max_by_key(|s| s.wall_time)
    }

    /// Render a formatted table of per-pass timing and reduction statistics.
    pub fn format_table(&self) -> String {
        let mut out = String::new();
        out.push_str("┌──────────────────────────────────────────────────────────────────┐\n");
        out.push_str("│  Pass          Time(ms)   Nodes Before → After    Pct   Reductions│\n");
        out.push_str("├──────────────────────────────────────────────────────────────────┤\n");
        for s in &self.pass_stats {
            out.push_str(&format!("│  {}\n", s.summary()));
        }
        out.push_str("├──────────────────────────────────────────────────────────────────┤\n");
        out.push_str(&format!(
            "│  TOTAL         {:>8.3}ms  {:>6} nodes → {:>6} ({:>5.1}% overall)      │\n",
            self.total_wall_time.as_secs_f64() * 1_000.0,
            self.initial_node_count,
            self.final_node_count,
            self.overall_reduction_pct(),
        ));
        out.push_str("└──────────────────────────────────────────────────────────────────┘\n");
        out
    }

    /// Human-readable one-line summary of the full pipeline run.
    pub fn summary(&self) -> String {
        format!(
            "Pipeline: {} outer iterations, {:.3}ms total, {} → {} nodes ({:.1}% reduction)",
            self.outer_iterations,
            self.total_wall_time.as_secs_f64() * 1_000.0,
            self.initial_node_count,
            self.final_node_count,
            self.overall_reduction_pct(),
        )
    }
}

// ────────────────────────────────────────────────────────────────────────────
// CompilerPipelineResult
// ────────────────────────────────────────────────────────────────────────────

/// Output of a full pipeline run: the transformed expression plus collected statistics.
#[derive(Debug, Clone)]
pub struct CompilerPipelineResult {
    /// The optimized expression.
    pub expr: TLExpr,
    /// Statistics covering all passes and outer iterations.
    pub stats: CompilerPipelineStats,
}

// ────────────────────────────────────────────────────────────────────────────
// PassBenchmark
// ────────────────────────────────────────────────────────────────────────────

/// Benchmarking statistics across multiple repeated runs of the same pass.
#[derive(Debug, Clone)]
pub struct PassBenchmark {
    /// Which pass was benchmarked.
    pub pass_id: CompilerPassId,
    /// Number of runs included in these statistics.
    pub runs: usize,
    /// Minimum observed wall-clock time in nanoseconds.
    pub min_ns: u64,
    /// Maximum observed wall-clock time in nanoseconds.
    pub max_ns: u64,
    /// Arithmetic mean wall-clock time in nanoseconds.
    pub mean_ns: u64,
    /// Sum of all `reductions` values across all runs.
    pub total_reductions: u64,
}

impl PassBenchmark {
    /// Human-readable one-line benchmark summary.
    pub fn summary(&self) -> String {
        format!(
            "{:<12}  runs={:>4}  min={:.3}ms  mean={:.3}ms  max={:.3}ms  reductions={}",
            self.pass_id.to_string(),
            self.runs,
            self.min_ns as f64 / 1_000_000.0,
            self.mean_ns as f64 / 1_000_000.0,
            self.max_ns as f64 / 1_000_000.0,
            self.total_reductions,
        )
    }
}

// ────────────────────────────────────────────────────────────────────────────
// CompilerPipeline
// ────────────────────────────────────────────────────────────────────────────

/// Full compiler optimization pipeline.
///
/// Chains all compiler passes in configurable order with per-pass timing and
/// aggregate statistics.  The outer fixed-point loop repeats the entire pass
/// sequence until no further nodes are reduced or `config.max_outer_iterations`
/// is reached.
pub struct CompilerPipeline {
    config: CompilerPipelineConfig,
}

impl Default for CompilerPipeline {
    fn default() -> Self {
        Self::with_default()
    }
}

impl CompilerPipeline {
    /// Create a pipeline with the given configuration.
    pub fn new(config: CompilerPipelineConfig) -> Self {
        Self { config }
    }

    /// Create a pipeline with the default configuration (all passes enabled).
    pub fn with_default() -> Self {
        Self::new(CompilerPipelineConfig::default())
    }

    /// Alias for [`Self::with_default`].
    pub fn all_passes() -> Self {
        Self::with_default()
    }

    /// Create a pipeline with all passes disabled.
    ///
    /// Expressions passed through this pipeline are returned unchanged (other
    /// than recording the initial node count in statistics).
    pub fn no_passes() -> Self {
        Self::new(CompilerPipelineConfig {
            enable_const_prop: false,
            enable_dead_code: false,
            enable_inline: false,
            enable_algebraic: false,
            enable_rewrite: false,
            ..CompilerPipelineConfig::default()
        })
    }

    // ── Public API ───────────────────────────────────────────────────────────

    /// Run the full pipeline on `expr` and return the result with statistics.
    pub fn run(&self, expr: TLExpr) -> CompilerPipelineResult {
        let pipeline_start = Instant::now();
        let initial_node_count = Self::count_nodes(&expr);

        let mut stats = CompilerPipelineStats {
            pass_stats: Vec::new(),
            total_wall_time: Duration::ZERO,
            outer_iterations: 0,
            total_node_reduction: 0,
            initial_node_count,
            final_node_count: initial_node_count,
        };

        let order = self.config.pass_order.to_pass_list();
        let mut current = expr;
        let max_iters = self.config.max_outer_iterations.max(1);

        for _ in 0..max_iters {
            let nodes_before_iter = Self::count_nodes(&current);
            current = self.run_sequence(current, &order, &mut stats);
            stats.outer_iterations += 1;

            let nodes_after_iter = Self::count_nodes(&current);
            // Stop early if no progress was made in this outer iteration.
            if nodes_after_iter >= nodes_before_iter {
                break;
            }
        }

        let final_node_count = Self::count_nodes(&current);
        stats.final_node_count = final_node_count;
        stats.total_node_reduction = initial_node_count as i64 - final_node_count as i64;
        stats.total_wall_time = pipeline_start.elapsed();

        CompilerPipelineResult {
            expr: current,
            stats,
        }
    }

    /// Run the pipeline `runs` times on (a clone of) `expr` and return per-pass
    /// benchmarking statistics aggregated across all runs.
    pub fn benchmark(&self, expr: TLExpr, runs: usize) -> Vec<PassBenchmark> {
        // Accumulate timing per pass_id across all runs.
        let mut timings: HashMap<String, (u64, u64, u64, u64, u64)> = HashMap::new(); // key → (count, min_ns, max_ns, sum_ns, total_reductions)

        let effective_runs = runs.max(1);

        for _ in 0..effective_runs {
            let result = self.run(expr.clone());
            for ps in &result.stats.pass_stats {
                let ns = ps.wall_time.as_nanos() as u64;
                let key = ps.pass_id.to_string();
                let entry = timings.entry(key).or_insert((0, u64::MAX, 0, 0, 0));
                entry.0 += 1;
                entry.1 = entry.1.min(ns);
                entry.2 = entry.2.max(ns);
                entry.3 = entry.3.saturating_add(ns);
                entry.4 = entry.4.saturating_add(ps.reductions);
            }
        }

        // Build a PassBenchmark per distinct pass id in the order they appear in
        // the pass list (deduplicating for Custom / AggressiveFold repeats).
        let order = self.config.pass_order.to_pass_list();
        let mut seen: Vec<String> = Vec::new();
        let mut benchmarks: Vec<PassBenchmark> = Vec::new();

        for pass_id in &order {
            let key = pass_id.to_string();
            if seen.contains(&key) {
                continue;
            }
            seen.push(key.clone());
            if let Some(&(count, min_ns, max_ns, sum_ns, total_reductions)) = timings.get(&key) {
                let mean_ns = sum_ns.checked_div(count).unwrap_or(0);
                benchmarks.push(PassBenchmark {
                    pass_id: pass_id.clone(),
                    runs: count as usize,
                    min_ns,
                    max_ns,
                    mean_ns,
                    total_reductions,
                });
            }
        }

        benchmarks
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    /// Execute the given pass sequence once, recording stats for each invocation.
    fn run_sequence(
        &self,
        mut expr: TLExpr,
        order: &[CompilerPassId],
        stats: &mut CompilerPipelineStats,
    ) -> TLExpr {
        for pass_id in order {
            expr = self.run_single_pass(pass_id, expr, stats);
        }
        expr
    }

    /// Execute a single pass, recording timing and reduction statistics.
    fn run_single_pass(
        &self,
        pass_id: &CompilerPassId,
        expr: TLExpr,
        stats: &mut CompilerPipelineStats,
    ) -> TLExpr {
        let nodes_before = Self::count_nodes(&expr);
        let t0 = Instant::now();

        let (new_expr, reductions) = match pass_id {
            CompilerPassId::ConstProp => {
                if !self.config.enable_const_prop {
                    return expr;
                }
                let propagator = ConstantPropagator::new(self.config.const_prop_config.clone());
                let (out, s) = propagator.run(expr);
                let r = s.total_folds();
                (out, r)
            }

            CompilerPassId::DeadCode => {
                if !self.config.enable_dead_code {
                    return expr;
                }
                let eliminator = DeadCodeEliminator::new(self.config.dce_config.clone());
                let (out, s) = eliminator.run(expr);
                let r = s.total_eliminations();
                (out, r)
            }

            CompilerPassId::Inline => {
                if !self.config.enable_inline {
                    return expr;
                }
                let inliner = LetInliner::new(self.config.inline_config.clone());
                let (out, s) = inliner.run(expr);
                let r = s.total();
                (out, r)
            }

            CompilerPassId::Algebraic => {
                if !self.config.enable_algebraic {
                    return expr;
                }
                let alg_pipeline = OptimizationPipeline::new();
                let (out, s) = alg_pipeline.optimize(&expr);
                let r = s.total_optimizations() as u64;
                (out, r)
            }

            CompilerPassId::Rewrite => {
                if !self.config.enable_rewrite {
                    return expr;
                }
                let engine = RewriteEngine::new().add_all_builtin_rules();
                let (out, s) = engine.rewrite(expr);
                let r = s.total_rewrites;
                (out, r)
            }
        };

        let wall_time = t0.elapsed();
        let nodes_after = Self::count_nodes(&new_expr);

        stats.pass_stats.push(CompilerPassStats {
            pass_id: pass_id.clone(),
            wall_time,
            nodes_before,
            nodes_after,
            reductions,
        });

        new_expr
    }

    /// Count the total number of AST nodes in `expr` via the DCE helper.
    fn count_nodes(expr: &TLExpr) -> u64 {
        DeadCodeEliminator::count_nodes(expr)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_ir::{TLExpr, Term};

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn simple_constant_expr() -> TLExpr {
        // Add(Mul(2, 3), 4)  →  10
        TLExpr::add(
            TLExpr::mul(TLExpr::Constant(2.0), TLExpr::Constant(3.0)),
            TLExpr::Constant(4.0),
        )
    }

    fn dead_branch_expr() -> TLExpr {
        // And(True, p(x))  →  p(x)
        TLExpr::and(
            TLExpr::Constant(1.0),
            TLExpr::pred("p", vec![Term::var("x")]),
        )
    }

    fn let_binding_expr() -> TLExpr {
        // Let y = 5.0 in y  →  5.0
        TLExpr::let_binding("y", TLExpr::Constant(5.0), TLExpr::pred("y", vec![]))
    }

    fn non_trivial_expr() -> TLExpr {
        // Not(Not(p(x))) wrapped in And(True, _)
        TLExpr::and(
            TLExpr::Constant(1.0),
            TLExpr::negate(TLExpr::negate(TLExpr::pred("p", vec![Term::var("x")]))),
        )
    }

    // ── Config tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_compiler_pipeline_config_default() {
        let cfg = CompilerPipelineConfig::default();
        assert!(cfg.enable_const_prop);
        assert!(cfg.enable_dead_code);
        assert!(cfg.enable_inline);
        assert!(cfg.enable_algebraic);
        assert!(cfg.enable_rewrite);
        assert_eq!(cfg.max_outer_iterations, 3);
    }

    // ── No-pass pipeline ─────────────────────────────────────────────────────

    #[test]
    fn test_compiler_pipeline_no_passes() {
        let pipeline = CompilerPipeline::no_passes();
        let expr = simple_constant_expr();
        let result = pipeline.run(expr.clone());
        // With all passes disabled, the expression should be structurally identical.
        assert_eq!(format!("{:?}", result.expr), format!("{:?}", expr),);
    }

    // ── Single-pass tests ────────────────────────────────────────────────────

    #[test]
    fn test_compiler_pipeline_const_prop_only() {
        let cfg = CompilerPipelineConfig {
            enable_const_prop: true,
            enable_dead_code: false,
            enable_inline: false,
            enable_algebraic: false,
            enable_rewrite: false,
            max_outer_iterations: 1,
            ..CompilerPipelineConfig::default()
        };
        let pipeline = CompilerPipeline::new(cfg);
        let expr = simple_constant_expr();
        let result = pipeline.run(expr);
        // The expression should have been folded to a constant.
        assert!(matches!(result.expr, TLExpr::Constant(_)));
    }

    #[test]
    fn test_compiler_pipeline_dead_code_only() {
        let cfg = CompilerPipelineConfig {
            enable_const_prop: false,
            enable_dead_code: true,
            enable_inline: false,
            enable_algebraic: false,
            enable_rewrite: false,
            max_outer_iterations: 1,
            ..CompilerPipelineConfig::default()
        };
        let pipeline = CompilerPipeline::new(cfg);
        let expr = dead_branch_expr();
        let result = pipeline.run(expr);
        // And(True, p(x))  →  p(x)
        assert!(matches!(result.expr, TLExpr::Pred { .. }));
    }

    #[test]
    fn test_compiler_pipeline_inline_only() {
        let cfg = CompilerPipelineConfig {
            enable_const_prop: false,
            enable_dead_code: false,
            enable_inline: true,
            enable_algebraic: false,
            enable_rewrite: false,
            max_outer_iterations: 1,
            ..CompilerPipelineConfig::default()
        };
        let pipeline = CompilerPipeline::new(cfg);
        let expr = let_binding_expr();
        let result = pipeline.run(expr);
        // Let y = 5.0 in y  →  5.0
        assert!(matches!(result.expr, TLExpr::Constant(v) if (v - 5.0).abs() < 1e-12));
    }

    #[test]
    fn test_compiler_pipeline_all_passes() {
        let pipeline = CompilerPipeline::all_passes();
        let expr = non_trivial_expr();
        // Should run without panicking.
        let result = pipeline.run(expr);
        assert!(result.stats.outer_iterations > 0);
    }

    // ── Stats tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_compiler_pipeline_result_has_stats() {
        let pipeline = CompilerPipeline::with_default();
        let expr = simple_constant_expr();
        let result = pipeline.run(expr);
        assert!(result.stats.initial_node_count > 0);
        assert!(!result.stats.pass_stats.is_empty());
    }

    #[test]
    fn test_pass_stats_reduction_pct() {
        let s = CompilerPassStats {
            pass_id: CompilerPassId::ConstProp,
            wall_time: Duration::from_millis(1),
            nodes_before: 100,
            nodes_after: 80,
            reductions: 5,
        };
        let pct = s.reduction_pct();
        assert!((pct - 20.0).abs() < 1e-6, "expected 20%, got {pct}");
    }

    #[test]
    fn test_pass_stats_reduction_pct_zero_before() {
        let s = CompilerPassStats {
            pass_id: CompilerPassId::DeadCode,
            wall_time: Duration::ZERO,
            nodes_before: 0,
            nodes_after: 0,
            reductions: 0,
        };
        assert_eq!(s.reduction_pct(), 0.0);
    }

    #[test]
    fn test_pass_stats_summary_nonempty() {
        let s = CompilerPassStats {
            pass_id: CompilerPassId::Inline,
            wall_time: Duration::from_micros(500),
            nodes_before: 10,
            nodes_after: 8,
            reductions: 2,
        };
        let summary = s.summary();
        assert!(!summary.is_empty());
        assert!(summary.contains("Inline"));
    }

    #[test]
    fn test_pipeline_stats_overall_reduction() {
        let pipeline = CompilerPipeline::with_default();
        let expr = simple_constant_expr();
        let result = pipeline.run(expr);
        let initial = result.stats.initial_node_count;
        let final_count = result.stats.final_node_count;
        assert!(
            initial >= final_count,
            "pipeline should not increase node count"
        );
        let pct = result.stats.overall_reduction_pct();
        assert!(pct >= 0.0);
    }

    #[test]
    fn test_pipeline_stats_format_table() {
        let pipeline = CompilerPipeline::with_default();
        let expr = simple_constant_expr();
        let result = pipeline.run(expr);
        let table = result.stats.format_table();
        assert!(
            table.contains("Pass") || table.contains("TOTAL"),
            "table should contain headers, got: {table}"
        );
    }

    #[test]
    fn test_pipeline_stats_summary_nonempty() {
        let pipeline = CompilerPipeline::with_default();
        let expr = simple_constant_expr();
        let result = pipeline.run(expr);
        let summary = result.stats.summary();
        assert!(!summary.is_empty());
        assert!(summary.contains("Pipeline"));
    }

    #[test]
    fn test_pipeline_stats_slowest_pass() {
        let pipeline = CompilerPipeline::with_default();
        let expr = simple_constant_expr();
        let result = pipeline.run(expr);
        // There should be at least one pass executed, so slowest_pass must be Some.
        assert!(result.stats.slowest_pass().is_some());
    }

    // ── Order tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_compiler_pipeline_canonical_order() {
        let cfg = CompilerPipelineConfig {
            pass_order: CompilerPassOrder::CanonicalOrder,
            ..CompilerPipelineConfig::default()
        };
        let pipeline = CompilerPipeline::new(cfg);
        let result = pipeline.run(non_trivial_expr());
        assert!(result.stats.outer_iterations >= 1);
    }

    #[test]
    fn test_compiler_pipeline_inline_first() {
        let cfg = CompilerPipelineConfig {
            pass_order: CompilerPassOrder::InlineFirst,
            ..CompilerPipelineConfig::default()
        };
        let pipeline = CompilerPipeline::new(cfg);
        let result = pipeline.run(let_binding_expr());
        assert!(result.stats.outer_iterations >= 1);
    }

    #[test]
    fn test_compiler_pipeline_custom_order() {
        let cfg = CompilerPipelineConfig {
            pass_order: CompilerPassOrder::Custom(vec![
                CompilerPassId::ConstProp,
                CompilerPassId::DeadCode,
            ]),
            max_outer_iterations: 1,
            ..CompilerPipelineConfig::default()
        };
        let pipeline = CompilerPipeline::new(cfg);
        let result = pipeline.run(simple_constant_expr());
        // Custom order with 2 passes → exactly 2 pass_stats entries per outer iter.
        assert_eq!(result.stats.pass_stats.len(), 2);
    }

    #[test]
    fn test_compiler_pipeline_outer_iterations() {
        let cfg = CompilerPipelineConfig {
            max_outer_iterations: 5,
            ..CompilerPipelineConfig::default()
        };
        let pipeline = CompilerPipeline::new(cfg);
        let result = pipeline.run(simple_constant_expr());
        // Must terminate before or at the max.
        assert!(result.stats.outer_iterations <= 5);
        assert!(result.stats.outer_iterations >= 1);
    }

    // ── Benchmark tests ──────────────────────────────────────────────────────

    #[test]
    fn test_benchmark_runs_n_times() {
        let pipeline = CompilerPipeline::with_default();
        let expr = simple_constant_expr();
        let benchmarks = pipeline.benchmark(expr, 4);
        // There should be one entry per distinct pass in the canonical order.
        let order_len = CompilerPassOrder::CanonicalOrder.to_pass_list().len();
        assert_eq!(benchmarks.len(), order_len);
        // Each benchmark should have been executed at least `runs` times
        // (outer iterations may cause more executions per pipeline run).
        for b in &benchmarks {
            assert!(
                b.runs >= 4,
                "expected >=4 runs for {}, got {}",
                b.pass_id,
                b.runs
            );
        }
    }

    #[test]
    fn test_pass_benchmark_summary_nonempty() {
        let pipeline = CompilerPipeline::with_default();
        let benchmarks = pipeline.benchmark(simple_constant_expr(), 2);
        for b in &benchmarks {
            let summary = b.summary();
            assert!(!summary.is_empty());
        }
    }

    // ── Idempotency test ─────────────────────────────────────────────────────

    #[test]
    fn test_pipeline_idempotent() {
        let pipeline = CompilerPipeline::with_default();
        let expr = non_trivial_expr();
        let first = pipeline.run(expr);
        let second = pipeline.run(first.expr.clone());
        // Running a second time on an already-optimised expression should yield
        // the same (or smaller) node count.
        assert!(
            second.stats.final_node_count <= first.stats.final_node_count,
            "second run produced more nodes than first"
        );
    }

    // ── AggressiveFold order test ────────────────────────────────────────────

    #[test]
    fn test_compiler_pipeline_aggressive_fold() {
        let cfg = CompilerPipelineConfig {
            pass_order: CompilerPassOrder::AggressiveFold,
            max_outer_iterations: 2,
            ..CompilerPipelineConfig::default()
        };
        let pipeline = CompilerPipeline::new(cfg);
        let result = pipeline.run(simple_constant_expr());
        assert!(result.stats.outer_iterations >= 1);
    }

    // ── PassBenchmark min ≤ mean ≤ max invariant ────────────────────────────

    #[test]
    fn test_benchmark_timing_invariants() {
        let pipeline = CompilerPipeline::with_default();
        let benchmarks = pipeline.benchmark(simple_constant_expr(), 3);
        for b in &benchmarks {
            assert!(b.min_ns <= b.mean_ns, "min_ns > mean_ns for {}", b.pass_id);
            assert!(b.mean_ns <= b.max_ns, "mean_ns > max_ns for {}", b.pass_id);
        }
    }
}
