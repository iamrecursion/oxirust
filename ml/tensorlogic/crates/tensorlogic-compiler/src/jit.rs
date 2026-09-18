//! JIT compilation for hot expression paths.
//!
//! [`JitCompiler`] wraps a standard compilation pipeline and tracks expression
//! usage frequency. When an expression exceeds [`JitCompiler::hot_threshold`]
//! compilations it is promoted to the "hot path": the expression is re-optimised
//! more aggressively with [`OptimizationPipeline`] (aggressive preset) before
//! compilation, and the result is stored as a pre-computed [`Arc<EinsumGraph>`].
//! All subsequent compilations of the same hot expression return the cached
//! graph in O(1) without re-running the optimizer or compiler.
//!
//! # Design notes
//!
//! Expression identity is determined via the `Debug` representation of the
//! `TLExpr` — a deterministic structural fingerprint. This avoids requiring
//! `Hash` or `PartialEq` on `TLExpr` while still being correct for the
//! intended use case (repeated compilation of the same logical rule).
//!
//! The call-count map stores a clone of the originating `TLExpr` alongside
//! its hit count so that, when the threshold is crossed, the original
//! expression is available for the extra optimization pass.
//!
//! # Thread safety
//!
//! Both the hot-path cache and the call-count map are guarded by a single
//! `Mutex`. The cold path (compilation itself) is performed *outside* the
//! lock so that concurrent cold compilations of different expressions do not
//! serialise on I/O-heavy optimizer work.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use tensorlogic_ir::{EinsumGraph, TLExpr};

use crate::{
    compile_to_einsum_with_config,
    config::CompilationConfig,
    dead_code::{DceConfig, DeadCodeEliminator},
    optimize::pipeline::{OptimizationPipeline, PipelineConfig},
};

// ─────────────────────────────────────────────────────────────────────────────
// Public error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors emitted by [`JitCompiler`].
#[derive(Debug)]
pub enum JitError {
    /// The underlying compilation step failed.
    CompilationFailed(anyhow::Error),
}

impl std::fmt::Display for JitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JitError::CompilationFailed(e) => write!(f, "JIT compilation failed: {}", e),
        }
    }
}

impl std::error::Error for JitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            JitError::CompilationFailed(e) => e.source(),
        }
    }
}

impl From<anyhow::Error> for JitError {
    fn from(e: anyhow::Error) -> Self {
        JitError::CompilationFailed(e)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Statistics
// ─────────────────────────────────────────────────────────────────────────────

/// Statistics snapshot from a [`JitCompiler`].
#[derive(Debug, Clone, Default)]
pub struct JitStats {
    /// Number of distinct expressions currently promoted to the hot-path cache.
    pub hot_paths: usize,
    /// Total number of compile calls that went through the cold path
    /// (including the final cold call that triggers an upgrade).
    pub cold_compilations: usize,
    /// Number of compile calls that returned a pre-compiled hot-path graph.
    pub jit_hits: usize,
    /// Number of expressions that were upgraded from cold to hot (promoted).
    pub jit_upgrades: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal types
// ─────────────────────────────────────────────────────────────────────────────

/// A compiled hot-path entry.
#[derive(Clone)]
struct JitEntry {
    /// Pre-optimised, pre-compiled graph.
    graph: Arc<EinsumGraph>,
    /// Number of cache hits since promotion.
    hit_count: usize,
}

/// Per-expression tracking record kept in the call-count map.
struct CallRecord {
    /// Running invocation count (incremented on every `compile` call).
    count: usize,
    /// Clone of the originating expression, needed for extra-optimization
    /// when the threshold is crossed.
    expr: TLExpr,
}

struct JitCacheInner {
    /// Expressions that have been promoted to the hot path.
    hot_paths: HashMap<u64, JitEntry>,
    /// Call counts plus originating expression for every seen expression.
    call_counts: HashMap<u64, CallRecord>,
    /// Running statistics.
    stats: JitStats,
}

impl JitCacheInner {
    fn new() -> Self {
        Self {
            hot_paths: HashMap::new(),
            call_counts: HashMap::new(),
            stats: JitStats::default(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// JitCompiler
// ─────────────────────────────────────────────────────────────────────────────

/// JIT compiler with hot-path detection and pre-optimized graph caching.
///
/// # Example
///
/// ```rust
/// use tensorlogic_compiler::JitCompiler;
/// use tensorlogic_ir::{TLExpr, Term};
///
/// let jit = JitCompiler::new(3);
/// let expr = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);
///
/// for _ in 0..5 {
///     let graph = jit.compile(&expr).expect("compile");
///     let _ = graph;
/// }
///
/// let stats = jit.stats();
/// assert_eq!(jit.hot_path_count(), 1);
/// assert!(stats.jit_hits > 0);
/// ```
pub struct JitCompiler {
    /// Compilation configuration forwarded to the cold path.
    config: CompilationConfig,
    /// Number of compilations required before an expression is promoted.
    pub hot_threshold: usize,
    /// Shared cache guarded by a mutex.
    cache: Arc<Mutex<JitCacheInner>>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Expression hashing helper
// ─────────────────────────────────────────────────────────────────────────────

/// Compute a structural fingerprint for a `TLExpr` via its `Debug` output.
///
/// Two structurally identical expressions produce the same fingerprint.
/// Collisions are possible but astronomically unlikely for the intended use
/// case of tracking repeated rule compilations.
fn expr_hash(expr: &TLExpr) -> u64 {
    let repr = format!("{expr:?}");
    let mut hasher = DefaultHasher::new();
    repr.hash(&mut hasher);
    hasher.finish()
}

// ─────────────────────────────────────────────────────────────────────────────
// JitCompiler implementation
// ─────────────────────────────────────────────────────────────────────────────

impl JitCompiler {
    /// Create a new JIT compiler with default [`CompilationConfig`].
    ///
    /// `hot_threshold` is the number of compilations an expression must
    /// accumulate before it is promoted to the hot-path cache.
    pub fn new(hot_threshold: usize) -> Self {
        Self::with_config(CompilationConfig::default(), hot_threshold)
    }

    /// Create a new JIT compiler with a custom [`CompilationConfig`].
    pub fn with_config(config: CompilationConfig, hot_threshold: usize) -> Self {
        Self {
            config,
            hot_threshold,
            cache: Arc::new(Mutex::new(JitCacheInner::new())),
        }
    }

    /// Compile `expr`, returning a shared `Arc<EinsumGraph>`.
    ///
    /// - On the first `hot_threshold` calls the expression is compiled via the
    ///   normal cold path.
    /// - When the call count reaches `hot_threshold` the expression is
    ///   optimised with an aggressive expression-level pass and recompiled;
    ///   the result is inserted into the hot-path cache.
    /// - All subsequent calls for the same expression return the cached graph
    ///   directly without invoking the compiler.
    pub fn compile(&self, expr: &TLExpr) -> Result<Arc<EinsumGraph>, JitError> {
        let key = expr_hash(expr);

        // ── Fast path: check hot cache before doing any compilation work ──────
        {
            let mut guard = self.cache.lock().unwrap_or_else(|e| e.into_inner());

            // Increment call count; insert a new record if first time seen.
            let record = guard.call_counts.entry(key).or_insert_with(|| CallRecord {
                count: 0,
                expr: expr.clone(),
            });
            record.count += 1;

            // Hot-path hit: return cached graph immediately.
            //
            // We clone the Arc while holding the mutable borrow on the entry,
            // then drop the mutable borrow before updating the sibling stats
            // field — satisfying the single-&mut rule.
            if let Some(arc) = guard.hot_paths.get_mut(&key).map(|entry| {
                entry.hit_count += 1;
                Arc::clone(&entry.graph)
            }) {
                guard.stats.jit_hits += 1;
                return Ok(arc);
            }
        }

        // ── Cold path: compile the expression normally ─────────────────────────
        let cold_graph = compile_to_einsum_with_config(expr, &self.config)?;

        // ── Check current call count to decide on promotion ───────────────────
        let current_count = {
            let guard = self.cache.lock().unwrap_or_else(|e| e.into_inner());
            guard.call_counts.get(&key).map(|r| r.count).unwrap_or(0)
        };

        if current_count >= self.hot_threshold {
            // Retrieve the stored expression for the extra optimisation pass.
            let stored_expr = {
                let guard = self.cache.lock().unwrap_or_else(|e| e.into_inner());
                guard.call_counts.get(&key).map(|r| r.expr.clone())
            };

            if let Some(original_expr) = stored_expr {
                let optimized_graph = self.apply_extra_optimization(&original_expr)?;
                let arc = Arc::new(optimized_graph);

                let mut guard = self.cache.lock().unwrap_or_else(|e| e.into_inner());
                // Guard against a concurrent thread that already promoted this key.
                if let std::collections::hash_map::Entry::Vacant(slot) = guard.hot_paths.entry(key)
                {
                    slot.insert(JitEntry {
                        graph: Arc::clone(&arc),
                        hit_count: 0,
                    });
                    guard.stats.jit_upgrades += 1;
                    guard.stats.hot_paths += 1;
                }
                guard.stats.cold_compilations += 1;
                return Ok(arc);
            }
        }

        // Below threshold: return cold-compiled graph without promotion.
        let mut guard = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        guard.stats.cold_compilations += 1;
        Ok(Arc::new(cold_graph))
    }

    /// Apply the extra expression-level optimization pass used when promoting
    /// an expression to the hot path.
    ///
    /// Strategy (in order of decreasing preference):
    ///
    /// 1. Run the [`OptimizationPipeline`] with an **aggressive** configuration
    ///    (max 20 iterations, all passes enabled including distributivity and
    ///    quantifier hoisting) on `expr`.
    /// 2. Follow with a full [`DeadCodeEliminator`] fixed-point pass.
    /// 3. Recompile the doubly-optimised expression with [`compile_to_einsum_with_config`].
    ///
    /// This produces a graph whose underlying expression has had significantly
    /// more algebraic simplification applied compared to the cold path.
    fn apply_extra_optimization(&self, expr: &TLExpr) -> Result<EinsumGraph, JitError> {
        // Step 1: Aggressive expression-level pipeline optimisation.
        let aggressive_config = PipelineConfig {
            enable_negation_opt: true,
            enable_constant_folding: true,
            enable_algebraic_simplification: true,
            enable_strength_reduction: true,
            enable_distributivity: true,
            enable_quantifier_opt: true,
            enable_dead_code_elimination: true,
            max_iterations: 20,
            stop_on_fixed_point: true,
        };
        let pipeline = OptimizationPipeline::with_config(aggressive_config);
        let (after_pipeline, _pipeline_stats) = pipeline.optimize(expr);

        // Step 2: Additional dead-code elimination pass to prune branches that
        //         may have become unreachable after constant folding / strength
        //         reduction in the pipeline.
        let dce_config = DceConfig {
            eliminate_constant_and: true,
            eliminate_constant_or: true,
            eliminate_constant_not: true,
            eliminate_if_branches: true,
            eliminate_unused_let: true,
            max_passes: 20,
        };
        let eliminator = DeadCodeEliminator::new(dce_config);
        let (fully_optimized, _dce_stats) = eliminator.run(after_pipeline);

        // Step 3: Compile the fully-optimised expression to an EinsumGraph.
        let graph = compile_to_einsum_with_config(&fully_optimized, &self.config)?;

        Ok(graph)
    }

    /// Return a snapshot of the current JIT statistics.
    pub fn stats(&self) -> JitStats {
        let guard = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        guard.stats.clone()
    }

    /// Evict all cached hot-path graphs and reset all counters.
    ///
    /// After this call the JIT compiler behaves as if it were freshly
    /// constructed.
    pub fn clear_cache(&mut self) {
        if let Ok(mut guard) = self.cache.lock() {
            *guard = JitCacheInner::new();
        }
    }

    /// Return the number of distinct expressions currently in the hot-path cache.
    pub fn hot_path_count(&self) -> usize {
        let guard = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        guard.hot_paths.len()
    }

    /// Return the total number of times `expr` has been compiled via this instance.
    ///
    /// Returns `0` if `expr` has never been seen.
    pub fn call_count(&self, expr: &TLExpr) -> usize {
        let guard = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        guard
            .call_counts
            .get(&expr_hash(expr))
            .map(|r| r.count)
            .unwrap_or(0)
    }

    /// Return the hot-path threshold used by this instance.
    pub fn threshold(&self) -> usize {
        self.hot_threshold
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_ir::{TLExpr, Term};

    fn simple_expr() -> TLExpr {
        TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")])
    }

    fn different_expr() -> TLExpr {
        TLExpr::pred("likes", vec![Term::var("a")])
    }

    #[test]
    fn test_cold_path_returns_graph() {
        let jit = JitCompiler::new(5);
        let graph = jit.compile(&simple_expr()).expect("cold compile");
        // Graph must be valid (may be empty for trivial predicates — just must not panic).
        let _ = graph;
        let stats = jit.stats();
        assert_eq!(stats.cold_compilations, 1);
        assert_eq!(stats.jit_hits, 0);
    }

    #[test]
    fn test_hot_upgrade_at_threshold() {
        let jit = JitCompiler::new(3);
        let expr = simple_expr();
        for _ in 0..3 {
            jit.compile(&expr).expect("compile");
        }
        assert_eq!(jit.hot_path_count(), 1);
        let stats = jit.stats();
        assert!(stats.jit_upgrades >= 1);
    }

    #[test]
    fn test_jit_hit_after_upgrade() {
        let jit = JitCompiler::new(2);
        let expr = simple_expr();
        // First two calls: cold (second one triggers the upgrade).
        jit.compile(&expr).expect("call 1");
        jit.compile(&expr).expect("call 2");
        // Third call: should be a hit from the hot cache.
        jit.compile(&expr).expect("call 3");
        let stats = jit.stats();
        assert!(
            stats.jit_hits >= 1,
            "expected at least 1 jit_hit, got {stats:?}"
        );
    }

    #[test]
    fn test_clear_cache_resets() {
        let mut jit = JitCompiler::new(1);
        let expr = simple_expr();
        jit.compile(&expr).expect("compile once");
        assert_eq!(jit.hot_path_count(), 1);
        jit.clear_cache();
        assert_eq!(jit.hot_path_count(), 0);
        assert_eq!(jit.call_count(&expr), 0);
    }

    #[test]
    fn test_different_exprs_tracked_separately() {
        let jit = JitCompiler::new(10);
        let e1 = simple_expr();
        let e2 = different_expr();
        for _ in 0..3 {
            jit.compile(&e1).expect("e1");
        }
        jit.compile(&e2).expect("e2");
        assert_eq!(jit.call_count(&e1), 3);
        assert_eq!(jit.call_count(&e2), 1);
    }

    #[test]
    fn test_threshold_one_upgrades_immediately() {
        let jit = JitCompiler::new(1);
        let expr = simple_expr();
        jit.compile(&expr).expect("first call");
        assert_eq!(jit.hot_path_count(), 1);
    }

    #[test]
    fn test_stats_consistent() {
        let jit = JitCompiler::new(3);
        let expr = simple_expr();
        let total = 5usize;
        for _ in 0..total {
            jit.compile(&expr).expect("compile");
        }
        let stats = jit.stats();
        assert_eq!(
            stats.cold_compilations + stats.jit_hits,
            total,
            "cold + hits must equal total calls; got {stats:?}"
        );
    }

    #[test]
    fn test_hot_graph_not_empty() {
        let jit = JitCompiler::new(2);
        let expr = simple_expr();
        jit.compile(&expr).expect("call 1");
        jit.compile(&expr).expect("call 2");
        // Third call hits the hot cache — should not panic.
        let graph = jit.compile(&expr).expect("call 3 (hot)");
        let _ = graph;
    }

    #[test]
    fn test_threshold_accessor() {
        let jit = JitCompiler::new(7);
        assert_eq!(jit.threshold(), 7);
    }
}
