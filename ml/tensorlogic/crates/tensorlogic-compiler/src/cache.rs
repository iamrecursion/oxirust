//! Compilation cache for TensorLogic expressions.
//!
//! Provides two complementary caching mechanisms:
//!
//! 1. **[`CompilationCache`]** — A thread-safe, key-based cache that stores compiled
//!    `EinsumGraph` instances keyed by a composite hash of expression structure,
//!    compilation configuration, and domain information. Designed for concurrent use.
//!
//! 2. **[`LruCompilationCache`]** — A single-threaded LRU cache keyed by
//!    [`ExprFingerprint`] (a structural content-address of an expression). Evicts the
//!    least-recently-used entry when capacity is exceeded. Designed for use inside a
//!    [`CachingCompiler`] wrapper.
//!
//! # Choosing the right cache
//!
//! | Scenario | Recommended type |
//! |----------|-----------------|
//! | Single-threaded compilation loop | [`LruCompilationCache`] / [`CachingCompiler`] |
//! | Multi-threaded compilation (shared) | [`CompilationCache`] |
//! | Batch compilation of related exprs | [`CachingCompiler::compile_batch`] |
//!
//! # Example — LRU cache via `CachingCompiler`
//!
//! ```rust
//! use tensorlogic_compiler::cache::{CachingCompiler, CacheStats};
//! use tensorlogic_compiler::compile_to_einsum;
//! use tensorlogic_ir::{TLExpr, Term};
//!
//! let mut compiler = CachingCompiler::new(64, |expr| {
//!     compile_to_einsum(expr).map_err(|e| e.to_string())
//! });
//!
//! let expr = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);
//!
//! let _g1 = compiler.compile(&expr).expect("first compile");
//! let _g2 = compiler.compile(&expr).expect("second compile (cache hit)");
//!
//! assert_eq!(compiler.cache_stats().hits, 1);
//! assert_eq!(compiler.cache_stats().misses, 1);
//! ```

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use tensorlogic_ir::{EinsumGraph, TLExpr};

use crate::config::CompilationConfig;
use crate::CompilerContext;

// ──────────────────────────────────────────────────────────────────────────────
// ExprFingerprint
// ──────────────────────────────────────────────────────────────────────────────

/// A compact fingerprint of a `TLExpr` structure (not values).
///
/// Two expressions with identical structure produce the same fingerprint.
/// Used as a content-addressable cache key in [`LruCompilationCache`] and
/// [`CachingCompiler`].
///
/// The fingerprint is derived from the `Debug` representation of the expression,
/// which is deterministic for the same expression tree.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExprFingerprint {
    /// Serialised structural representation.
    pub(crate) data: String,
}

impl ExprFingerprint {
    /// Compute a fingerprint from an arbitrary string representation.
    ///
    /// In practice this is called with `format!("{:?}", expr)` so that the
    /// fingerprint captures the full recursive structure of the expression.
    pub fn compute(expr_repr: &str) -> Self {
        ExprFingerprint {
            data: expr_repr.to_string(),
        }
    }
}

impl std::fmt::Display for ExprFingerprint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let preview_len = self.data.len().min(32);
        write!(f, "fp:{}", &self.data[..preview_len])
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// CachedResult  (public, used by LruCompilationCache / CachingCompiler)
// ──────────────────────────────────────────────────────────────────────────────

/// A cached compilation result stored in an [`LruCompilationCache`].
#[derive(Debug, Clone)]
pub struct CachedResult {
    /// The compiled graph.
    pub graph: EinsumGraph,
    /// Number of times this entry was accessed (read) via [`LruCompilationCache::get`].
    pub hit_count: u64,
    /// Approximate memory used by the graph (estimated as `nodes.len() * 256` bytes).
    pub memory_bytes: usize,
}

// ──────────────────────────────────────────────────────────────────────────────
// CacheStats  (shared by both cache types)
// ──────────────────────────────────────────────────────────────────────────────

/// Aggregate statistics for any compilation cache.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Number of successful cache lookups.
    pub hits: u64,
    /// Number of cache lookups that resulted in a fresh compilation.
    pub misses: u64,
    /// Number of entries that were evicted to make room for new entries.
    pub evictions: u64,
    /// Current number of entries (updated after each insert/evict/clear).
    pub current_entries: usize,
    /// Approximate total memory occupied by all cached graphs (bytes).
    pub total_memory_bytes: usize,
}

impl CacheStats {
    /// Cache hit rate in the range `[0.0, 1.0]`.
    ///
    /// Returns `0.0` when no lookups have been performed yet.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Total number of cache lookups (hits + misses).
    pub fn total_lookups(&self) -> u64 {
        self.hits + self.misses
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// LruCompilationCache
// ──────────────────────────────────────────────────────────────────────────────

/// LRU compilation cache with configurable capacity.
///
/// Stores compiled `EinsumGraph` instances keyed by [`ExprFingerprint`].
/// When capacity is exceeded the least-recently-used entry is evicted.
///
/// This cache is **not** thread-safe — wrap it in `Arc<Mutex<_>>` or use
/// [`CompilationCache`] if you need concurrent access.
///
/// # Example
///
/// ```rust
/// use tensorlogic_compiler::cache::{LruCompilationCache, ExprFingerprint};
/// use tensorlogic_ir::EinsumGraph;
///
/// let mut cache = LruCompilationCache::new(4);
/// let fp = ExprFingerprint::compute("pred(x)");
/// cache.insert(fp.clone(), EinsumGraph::new());
/// assert!(cache.get(&fp).is_some());
/// ```
pub struct LruCompilationCache {
    /// Maximum number of entries.
    capacity: usize,
    /// The cache storage.
    entries: HashMap<ExprFingerprint, CachedResult>,
    /// LRU order: oldest at the **front**, newest at the **back**.
    lru_order: std::collections::VecDeque<ExprFingerprint>,
    /// Accumulated statistics.
    stats: CacheStats,
}

impl LruCompilationCache {
    /// Create a new LRU cache with the given capacity (minimum 1).
    pub fn new(capacity: usize) -> Self {
        LruCompilationCache {
            capacity: capacity.max(1),
            entries: HashMap::new(),
            lru_order: std::collections::VecDeque::new(),
            stats: CacheStats::default(),
        }
    }

    /// Insert a compiled result for the given fingerprint.
    ///
    /// If the fingerprint already exists the stored graph is updated and the
    /// entry is promoted to the most-recently-used position.
    ///
    /// If the cache is at capacity the least-recently-used entry is evicted
    /// before the new entry is inserted.
    pub fn insert(&mut self, fp: ExprFingerprint, graph: EinsumGraph) {
        // Estimate memory: proportional to node count.
        let memory_bytes = graph.nodes.len() * 256;

        if self.entries.contains_key(&fp) {
            // Update the existing entry in-place.
            if let Some(entry) = self.entries.get_mut(&fp) {
                self.stats.total_memory_bytes = self
                    .stats
                    .total_memory_bytes
                    .saturating_sub(entry.memory_bytes);
                entry.graph = graph;
                entry.memory_bytes = memory_bytes;
                self.stats.total_memory_bytes += memory_bytes;
            }
            // Promote to most-recently-used.
            if let Some(pos) = self.lru_order.iter().position(|x| x == &fp) {
                self.lru_order.remove(pos);
            }
            self.lru_order.push_back(fp);
        } else {
            // Evict the LRU entry when at capacity.
            if self.entries.len() >= self.capacity {
                if let Some(oldest) = self.lru_order.pop_front() {
                    if let Some(evicted) = self.entries.remove(&oldest) {
                        self.stats.total_memory_bytes = self
                            .stats
                            .total_memory_bytes
                            .saturating_sub(evicted.memory_bytes);
                    }
                    self.stats.evictions += 1;
                }
            }
            self.stats.total_memory_bytes += memory_bytes;
            self.lru_order.push_back(fp.clone());
            self.entries.insert(
                fp,
                CachedResult {
                    graph,
                    hit_count: 0,
                    memory_bytes,
                },
            );
        }
        self.stats.current_entries = self.entries.len();
    }

    /// Look up a fingerprint.
    ///
    /// On a hit the entry is promoted to the most-recently-used position,
    /// its `hit_count` is incremented, and a reference to it is returned.
    /// On a miss `None` is returned.
    pub fn get(&mut self, fp: &ExprFingerprint) -> Option<&CachedResult> {
        if self.entries.contains_key(fp) {
            // Promote to most-recently-used.
            if let Some(pos) = self.lru_order.iter().position(|x| x == fp) {
                self.lru_order.remove(pos);
            }
            self.lru_order.push_back(fp.clone());
            // Increment hit counter.
            if let Some(entry) = self.entries.get_mut(fp) {
                entry.hit_count += 1;
            }
            self.stats.hits += 1;
            self.entries.get(fp)
        } else {
            self.stats.misses += 1;
            None
        }
    }

    /// Check if a fingerprint is present **without** updating LRU order or stats.
    pub fn contains(&self, fp: &ExprFingerprint) -> bool {
        self.entries.contains_key(fp)
    }

    /// Remove a specific entry by fingerprint.
    ///
    /// Returns `true` if the entry existed and was removed, `false` otherwise.
    pub fn invalidate(&mut self, fp: &ExprFingerprint) -> bool {
        if let Some(evicted) = self.entries.remove(fp) {
            self.stats.total_memory_bytes = self
                .stats
                .total_memory_bytes
                .saturating_sub(evicted.memory_bytes);
            if let Some(pos) = self.lru_order.iter().position(|x| x == fp) {
                self.lru_order.remove(pos);
            }
            self.stats.current_entries = self.entries.len();
            true
        } else {
            false
        }
    }

    /// Clear all cached entries, resetting memory accounting.
    ///
    /// Statistics counters (hits, misses, evictions) are **not** reset.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.lru_order.clear();
        self.stats.current_entries = 0;
        self.stats.total_memory_bytes = 0;
    }

    /// Reference to the current statistics snapshot.
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when the cache contains no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The maximum number of entries this cache can hold before eviction.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

impl Default for LruCompilationCache {
    fn default() -> Self {
        Self::new(256)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// CachingCompiler
// ──────────────────────────────────────────────────────────────────────────────

/// A compiler wrapper that caches results keyed by expression fingerprint.
///
/// Uses structural fingerprinting of [`TLExpr`] to detect identical expressions.
/// Falls back to fresh compilation on a cache miss and stores the result for
/// subsequent calls.
///
/// # Example
///
/// ```rust
/// use tensorlogic_compiler::cache::CachingCompiler;
/// use tensorlogic_compiler::compile_to_einsum;
/// use tensorlogic_ir::{TLExpr, Term};
///
/// let mut cc = CachingCompiler::new(32, |expr| {
///     compile_to_einsum(expr).map_err(|e| e.to_string())
/// });
///
/// let e = TLExpr::pred("p", vec![Term::var("x")]);
/// let g1 = cc.compile(&e).unwrap();
/// let g2 = cc.compile(&e).unwrap(); // cache hit
///
/// assert_eq!(cc.cache_stats().hits, 1);
/// assert_eq!(g1, g2);
/// ```
/// Type alias for the compile function stored in a [`CachingCompiler`].
type CompileFn =
    Box<dyn Fn(&TLExpr) -> std::result::Result<EinsumGraph, String> + Send + Sync + 'static>;

pub struct CachingCompiler {
    cache: LruCompilationCache,
    compile_fn: CompileFn,
}

impl CachingCompiler {
    /// Create a `CachingCompiler` with a custom compile function and cache capacity.
    ///
    /// # Arguments
    ///
    /// * `capacity` – Maximum number of entries held in the LRU cache.
    /// * `compile_fn` – A closure (or function) that compiles a [`TLExpr`] into an
    ///   [`EinsumGraph`], returning `Err(String)` on failure.
    pub fn new<F>(capacity: usize, compile_fn: F) -> Self
    where
        F: Fn(&TLExpr) -> std::result::Result<EinsumGraph, String> + Send + Sync + 'static,
    {
        CachingCompiler {
            cache: LruCompilationCache::new(capacity),
            compile_fn: Box::new(compile_fn),
        }
    }

    /// Compile an expression, returning the cached result when available.
    ///
    /// # Errors
    ///
    /// Propagates any error produced by the underlying compile function on a cache miss.
    pub fn compile(&mut self, expr: &TLExpr) -> std::result::Result<EinsumGraph, String> {
        let fp = Self::fingerprint(expr);

        if let Some(cached) = self.cache.get(&fp) {
            return Ok(cached.graph.clone());
        }

        let result = (self.compile_fn)(expr)?;
        self.cache.insert(fp, result.clone());
        Ok(result)
    }

    /// Compile multiple expressions in order, sharing the cache across all of them.
    ///
    /// Returns one `Result` per input expression in the same order.
    pub fn compile_batch(
        &mut self,
        exprs: &[TLExpr],
    ) -> Vec<std::result::Result<EinsumGraph, String>> {
        exprs.iter().map(|e| self.compile(e)).collect()
    }

    /// Returns a reference to the current cache statistics.
    pub fn cache_stats(&self) -> &CacheStats {
        self.cache.stats()
    }

    /// Invalidate the cached result for a specific expression.
    ///
    /// Returns `true` if an entry was present and removed.
    pub fn invalidate(&mut self, expr: &TLExpr) -> bool {
        let fp = Self::fingerprint(expr);
        self.cache.invalidate(&fp)
    }

    /// Compute a structural [`ExprFingerprint`] for an expression.
    ///
    /// Two structurally identical expressions will produce equal fingerprints.
    pub fn fingerprint(expr: &TLExpr) -> ExprFingerprint {
        ExprFingerprint::compute(&Self::structural_repr(expr))
    }

    /// Produce a deterministic string representation of an expression's structure.
    ///
    /// This uses the `Debug` implementation of [`TLExpr`] which is deterministic
    /// for the same expression tree. Future enhancements may switch to a custom
    /// canonical serialisation if `Debug` output format changes.
    fn structural_repr(expr: &TLExpr) -> String {
        // `Debug` for TLExpr is stable within a single build and deterministic
        // for identical expression trees, making it a reliable fingerprint source.
        format!("{:?}", expr)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Legacy thread-safe CompilationCache  (original implementation, retained)
// ──────────────────────────────────────────────────────────────────────────────

/// A hash key for the thread-safe compilation cache.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CacheKey {
    expr_hash: u64,
    config_hash: u64,
    domain_hash: u64,
}

impl CacheKey {
    fn new(expr: &TLExpr, config: &CompilationConfig, ctx: &CompilerContext) -> Self {
        use std::collections::hash_map::DefaultHasher;

        let mut expr_hasher = DefaultHasher::new();
        format!("{:?}", expr).hash(&mut expr_hasher);
        let expr_hash = expr_hasher.finish();

        let mut config_hasher = DefaultHasher::new();
        format!("{:?}", config).hash(&mut config_hasher);
        let config_hash = config_hasher.finish();

        let mut domain_hasher = DefaultHasher::new();
        for (name, domain) in &ctx.domains {
            name.hash(&mut domain_hasher);
            domain.cardinality.hash(&mut domain_hasher);
        }
        let domain_hash = domain_hasher.finish();

        CacheKey {
            expr_hash,
            config_hash,
            domain_hash,
        }
    }
}

/// Internal cached result for the thread-safe cache.
#[derive(Clone)]
struct ThreadSafeCachedResult {
    graph: EinsumGraph,
    hit_count: usize,
}

/// Thread-safe compilation cache for storing and retrieving compiled expressions.
///
/// Stores compiled `EinsumGraph` instances keyed by a composite hash that includes
/// the expression structure, compilation configuration, and domain information.
/// This cache **is** thread-safe and can be shared across compilation threads.
///
/// When capacity is exceeded the cache evicts the least-frequently-used entry
/// (lowest `hit_count`). For strict LRU eviction use [`LruCompilationCache`] or
/// [`CachingCompiler`] instead.
///
/// # Example
///
/// ```rust
/// use tensorlogic_compiler::{CompilationCache, compile_to_einsum_with_context, CompilerContext};
/// use tensorlogic_ir::{TLExpr, Term};
///
/// let cache = CompilationCache::new(100);
/// let mut ctx = CompilerContext::new();
/// ctx.add_domain("Person", 100);
///
/// let expr = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);
///
/// // First compilation: miss (not in cache)
/// let graph1 = cache.get_or_compile(&expr, &mut ctx, |expr, ctx| {
///     compile_to_einsum_with_context(expr, ctx)
/// }).expect("compile");
///
/// // Second compilation: hit (cached)
/// let graph2 = cache.get_or_compile(&expr, &mut ctx, |expr, ctx| {
///     compile_to_einsum_with_context(expr, ctx)
/// }).expect("compile");
///
/// assert_eq!(graph1, graph2);
/// assert_eq!(cache.stats().hits, 1);
/// ```
pub struct CompilationCache {
    cache: Arc<Mutex<HashMap<CacheKey, ThreadSafeCachedResult>>>,
    max_size: usize,
    stats: Arc<Mutex<CacheStats>>,
}

impl CompilationCache {
    /// Create a new compilation cache with the specified maximum size.
    ///
    /// # Arguments
    ///
    /// * `max_size` – Maximum number of entries to cache.
    ///
    /// # Example
    ///
    /// ```rust
    /// use tensorlogic_compiler::CompilationCache;
    ///
    /// let cache = CompilationCache::new(100);
    /// assert_eq!(cache.max_size(), 100);
    /// ```
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: Arc::new(Mutex::new(HashMap::new())),
            max_size,
            stats: Arc::new(Mutex::new(CacheStats::default())),
        }
    }

    /// Create a cache with the default size of 1 000 entries.
    pub fn default_size() -> Self {
        Self::new(1000)
    }

    /// Maximum number of entries the cache can hold.
    pub fn max_size(&self) -> usize {
        self.max_size
    }

    /// Get or compile an expression.
    ///
    /// On a cache hit the stored result is returned immediately.
    /// On a miss `compile_fn` is called and the result is stored before returning.
    ///
    /// # Example
    ///
    /// ```rust
    /// use tensorlogic_compiler::{CompilationCache, compile_to_einsum_with_context, CompilerContext};
    /// use tensorlogic_ir::{TLExpr, Term};
    ///
    /// let cache = CompilationCache::new(100);
    /// let mut ctx = CompilerContext::new();
    /// ctx.add_domain("Person", 100);
    ///
    /// let expr = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);
    ///
    /// let graph = cache.get_or_compile(&expr, &mut ctx, |expr, ctx| {
    ///     compile_to_einsum_with_context(expr, ctx)
    /// }).expect("compile");
    /// ```
    pub fn get_or_compile<F>(
        &self,
        expr: &TLExpr,
        ctx: &mut CompilerContext,
        compile_fn: F,
    ) -> Result<EinsumGraph>
    where
        F: FnOnce(&TLExpr, &mut CompilerContext) -> Result<EinsumGraph>,
    {
        let key = CacheKey::new(expr, &ctx.config, ctx);

        // Try cache first.
        {
            let mut cache = self
                .cache
                .lock()
                .map_err(|e| anyhow::anyhow!("cache lock poisoned: {}", e))?;
            if let Some(cached) = cache.get_mut(&key) {
                cached.hit_count += 1;
                let mut stats = self
                    .stats
                    .lock()
                    .map_err(|e| anyhow::anyhow!("stats lock poisoned: {}", e))?;
                stats.hits += 1;
                return Ok(cached.graph.clone());
            }
        }

        // Cache miss — compile.
        {
            let mut stats = self
                .stats
                .lock()
                .map_err(|e| anyhow::anyhow!("stats lock poisoned: {}", e))?;
            stats.misses += 1;
        }

        let graph = compile_fn(expr, ctx)?;

        // Store result (evict if necessary).
        {
            let mut cache = self
                .cache
                .lock()
                .map_err(|e| anyhow::anyhow!("cache lock poisoned: {}", e))?;

            if cache.len() >= self.max_size {
                // Evict least-frequently used entry.
                let min_key = cache
                    .iter()
                    .min_by_key(|(_, v)| v.hit_count)
                    .map(|(k, _)| k.clone());

                if let Some(key_to_evict) = min_key {
                    cache.remove(&key_to_evict);
                    let mut stats = self
                        .stats
                        .lock()
                        .map_err(|e| anyhow::anyhow!("stats lock poisoned: {}", e))?;
                    stats.evictions += 1;
                }
            }

            cache.insert(
                key,
                ThreadSafeCachedResult {
                    graph: graph.clone(),
                    hit_count: 0,
                },
            );

            let mut stats = self
                .stats
                .lock()
                .map_err(|e| anyhow::anyhow!("stats lock poisoned: {}", e))?;
            stats.current_entries = cache.len();
        }

        Ok(graph)
    }

    /// Current cache statistics snapshot.
    ///
    /// # Example
    ///
    /// ```rust
    /// use tensorlogic_compiler::CompilationCache;
    ///
    /// let cache = CompilationCache::new(100);
    /// let stats = cache.stats();
    /// assert_eq!(stats.hits, 0);
    /// ```
    pub fn stats(&self) -> CacheStats {
        self.stats.lock().map(|g| g.clone()).unwrap_or_default()
    }

    /// Clear all cached entries.
    ///
    /// # Example
    ///
    /// ```rust
    /// use tensorlogic_compiler::CompilationCache;
    ///
    /// let cache = CompilationCache::new(100);
    /// cache.clear();
    /// assert_eq!(cache.stats().current_entries, 0);
    /// ```
    pub fn clear(&self) {
        if let Ok(mut cache) = self.cache.lock() {
            cache.clear();
        }
        if let Ok(mut stats) = self.stats.lock() {
            stats.current_entries = 0;
            stats.total_memory_bytes = 0;
        }
    }

    /// Current number of entries in the cache.
    pub fn len(&self) -> usize {
        self.cache.lock().map(|g| g.len()).unwrap_or(0)
    }

    /// Returns `true` when the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for CompilationCache {
    fn default() -> Self {
        Self::default_size()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile_to_einsum_with_context;
    use tensorlogic_ir::Term;

    // ── helpers ────────────────────────────────────────────────────────────────

    fn make_graph(node_count: usize) -> EinsumGraph {
        use tensorlogic_ir::EinsumNode;
        let mut g = EinsumGraph::new();
        for i in 0..node_count {
            let a = g.add_tensor(format!("t{}", i));
            let b = g.add_tensor(format!("u{}", i));
            let c = g.add_tensor(format!("v{}", i));
            g.add_node(EinsumNode::einsum("i,i->i", vec![a, b], vec![c]))
                .ok();
        }
        g
    }

    fn simple_fp(s: &str) -> ExprFingerprint {
        ExprFingerprint::compute(s)
    }

    // ── LruCompilationCache tests ──────────────────────────────────────────────

    /// insert then get returns Some
    #[test]
    fn test_cache_basic_insert_get() {
        let mut cache = LruCompilationCache::new(8);
        let fp = simple_fp("pred(x)");
        cache.insert(fp.clone(), EinsumGraph::new());
        assert!(
            cache.get(&fp).is_some(),
            "entry should be present after insert"
        );
    }

    /// get on empty cache returns None
    #[test]
    fn test_cache_miss() {
        let mut cache = LruCompilationCache::new(8);
        let fp = simple_fp("pred(x)");
        assert!(cache.get(&fp).is_none(), "empty cache must return None");
    }

    /// hit_count increments on each successful get
    #[test]
    fn test_cache_hit_increments_hit_count() {
        let mut cache = LruCompilationCache::new(8);
        let fp = simple_fp("pred(x)");
        cache.insert(fp.clone(), EinsumGraph::new());

        cache.get(&fp);
        cache.get(&fp);

        // hit_count inside the entry should reflect two reads.
        assert!(cache.contains(&fp), "entry must still exist after reads");
        // Obtain hit_count via a final get.
        let entry = cache.get(&fp).expect("entry must be present");
        // Three gets were performed (two above + this one) → hit_count == 3.
        assert_eq!(entry.hit_count, 3, "hit_count should be 3 after three gets");
    }

    /// 1 hit + 1 miss → hit_rate == 0.5
    #[test]
    fn test_cache_stats_hit_rate() {
        let mut cache = LruCompilationCache::new(8);
        let fp = simple_fp("pred(x)");
        cache.insert(fp.clone(), EinsumGraph::new());

        cache.get(&fp); // hit
        cache.get(&simple_fp("missing")); // miss

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert!(
            (stats.hit_rate() - 0.5).abs() < f64::EPSILON,
            "hit rate must be 0.5"
        );
    }

    /// capacity=2, three inserts → oldest evicted
    #[test]
    fn test_cache_lru_eviction() {
        let mut cache = LruCompilationCache::new(2);
        let fp1 = simple_fp("a");
        let fp2 = simple_fp("b");
        let fp3 = simple_fp("c");

        cache.insert(fp1.clone(), EinsumGraph::new());
        cache.insert(fp2.clone(), EinsumGraph::new());
        cache.insert(fp3.clone(), EinsumGraph::new()); // should evict fp1

        assert!(
            !cache.contains(&fp1),
            "oldest entry (fp1) must have been evicted"
        );
        assert!(cache.contains(&fp2), "fp2 must still be present");
        assert!(cache.contains(&fp3), "fp3 must be present");
        assert_eq!(cache.len(), 2);
    }

    /// Access the oldest entry so it becomes newest; the next eviction removes the other one
    #[test]
    fn test_cache_lru_access_updates_order() {
        let mut cache = LruCompilationCache::new(2);
        let fp1 = simple_fp("a");
        let fp2 = simple_fp("b");
        let fp3 = simple_fp("c");

        cache.insert(fp1.clone(), EinsumGraph::new());
        cache.insert(fp2.clone(), EinsumGraph::new());

        // Access fp1 → it becomes MRU; fp2 is now LRU.
        cache.get(&fp1);

        // Insert fp3 → fp2 should be evicted (LRU), not fp1.
        cache.insert(fp3.clone(), EinsumGraph::new());

        assert!(cache.contains(&fp1), "fp1 was accessed so it must survive");
        assert!(
            !cache.contains(&fp2),
            "fp2 is LRU after fp1 was accessed; it must be evicted"
        );
        assert!(cache.contains(&fp3), "fp3 must be present");
    }

    /// invalidate removes an entry
    #[test]
    fn test_cache_invalidate() {
        let mut cache = LruCompilationCache::new(8);
        let fp = simple_fp("pred(x)");
        cache.insert(fp.clone(), EinsumGraph::new());

        let removed = cache.invalidate(&fp);
        assert!(removed, "invalidate must return true when entry existed");
        assert!(
            !cache.contains(&fp),
            "entry must be gone after invalidation"
        );
    }

    /// clear empties the cache
    #[test]
    fn test_cache_clear() {
        let mut cache = LruCompilationCache::new(8);
        cache.insert(simple_fp("a"), EinsumGraph::new());
        cache.insert(simple_fp("b"), EinsumGraph::new());

        cache.clear();

        assert!(cache.is_empty(), "cache must be empty after clear");
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.stats().total_memory_bytes, 0);
    }

    /// len / is_empty reflect the actual entry count
    #[test]
    fn test_cache_len_and_is_empty() {
        let mut cache = LruCompilationCache::new(8);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);

        cache.insert(simple_fp("x"), EinsumGraph::new());
        assert!(!cache.is_empty());
        assert_eq!(cache.len(), 1);
    }

    /// capacity() returns the configured value
    #[test]
    fn test_cache_capacity() {
        let cache = LruCompilationCache::new(42);
        assert_eq!(cache.capacity(), 42);
    }

    /// evictions counter is updated correctly
    #[test]
    fn test_cache_eviction_stat() {
        let mut cache = LruCompilationCache::new(2);
        cache.insert(simple_fp("a"), EinsumGraph::new());
        cache.insert(simple_fp("b"), EinsumGraph::new());
        cache.insert(simple_fp("c"), EinsumGraph::new()); // one eviction
        cache.insert(simple_fp("d"), EinsumGraph::new()); // second eviction

        assert_eq!(
            cache.stats().evictions,
            2,
            "two evictions must have occurred"
        );
    }

    /// total_memory_bytes is positive after inserting a non-empty graph
    #[test]
    fn test_cache_memory_estimate() {
        let mut cache = LruCompilationCache::new(8);
        // Graph with 4 nodes → 4 * 256 = 1024 bytes estimated.
        let graph = make_graph(4);
        cache.insert(simple_fp("g"), graph);

        assert!(
            cache.stats().total_memory_bytes > 0,
            "memory estimate must be > 0 for a non-empty graph"
        );
    }

    // ── ExprFingerprint tests ──────────────────────────────────────────────────

    /// Same expression structure → same fingerprint
    #[test]
    fn test_fingerprint_same_for_same_expr() {
        let expr = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);
        let fp1 = CachingCompiler::fingerprint(&expr);
        let fp2 = CachingCompiler::fingerprint(&expr);
        assert_eq!(
            fp1, fp2,
            "identical expressions must produce identical fingerprints"
        );
    }

    /// Display format starts with "fp:"
    #[test]
    fn test_fingerprint_display() {
        let fp = ExprFingerprint::compute("pred(x, y)");
        let display = format!("{}", fp);
        assert!(display.starts_with("fp:"), "Display must start with 'fp:'");
    }

    // ── CachingCompiler tests ─────────────────────────────────────────────────

    fn make_caching_compiler(capacity: usize) -> CachingCompiler {
        CachingCompiler::new(capacity, |expr| {
            let mut ctx = CompilerContext::new();
            compile_to_einsum_with_context(expr, &mut ctx).map_err(|e| e.to_string())
        })
    }

    /// Second compile of the same expression should use the cache
    #[test]
    fn test_caching_compiler_cache_hit() {
        let mut cc = make_caching_compiler(32);
        let expr = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);

        cc.compile(&expr).expect("first compile");
        cc.compile(&expr).expect("second compile");

        assert_eq!(
            cc.cache_stats().hits,
            1,
            "second compile must be a cache hit"
        );
    }

    /// First compile counts as a miss
    #[test]
    fn test_caching_compiler_cache_miss_count() {
        let mut cc = make_caching_compiler(32);
        let expr = TLExpr::pred("likes", vec![Term::var("a"), Term::var("b")]);

        cc.compile(&expr).expect("compile");

        assert_eq!(
            cc.cache_stats().misses,
            1,
            "first compile must be a cache miss"
        );
        assert_eq!(cc.cache_stats().hits, 0);
    }

    /// compile_batch processes all expressions
    #[test]
    fn test_caching_compiler_batch() {
        let mut cc = make_caching_compiler(32);
        let exprs = vec![
            TLExpr::pred("p", vec![Term::var("x")]),
            TLExpr::pred("q", vec![Term::var("y")]),
            TLExpr::pred("r", vec![Term::var("z")]),
        ];

        let results = cc.compile_batch(&exprs);
        assert_eq!(results.len(), 3, "batch must return one result per input");
        for (i, r) in results.iter().enumerate() {
            assert!(r.is_ok(), "result[{}] must be Ok", i);
        }
    }

    /// invalidate clears the entry for a specific expression
    #[test]
    fn test_caching_compiler_invalidate() {
        let mut cc = make_caching_compiler(32);
        let expr = TLExpr::pred("p", vec![Term::var("x")]);

        cc.compile(&expr).expect("compile");
        let removed = cc.invalidate(&expr);
        assert!(removed, "invalidate must return true when entry existed");

        // Re-compiling should be a miss again.
        cc.compile(&expr).expect("re-compile");
        assert_eq!(
            cc.cache_stats().misses,
            2,
            "re-compile after invalidation must be another miss"
        );
    }

    // ── Default / misc tests ──────────────────────────────────────────────────

    /// Default LRU cache capacity is 256
    #[test]
    fn test_cache_default_capacity() {
        let cache = LruCompilationCache::default();
        assert_eq!(cache.capacity(), 256, "default capacity must be 256");
    }

    /// ExprFingerprint implements Hash and can be used as a HashMap key
    #[test]
    fn test_expr_fingerprint_hash() {
        let mut map: HashMap<ExprFingerprint, u32> = HashMap::new();
        let fp = ExprFingerprint::compute("some_expr");
        map.insert(fp.clone(), 42);
        assert_eq!(
            map.get(&fp),
            Some(&42),
            "fingerprint must work as HashMap key"
        );
    }

    // ── Legacy CompilationCache tests ─────────────────────────────────────────

    #[test]
    fn test_ts_cache_new() {
        let cache = CompilationCache::new(100);
        assert_eq!(cache.max_size(), 100);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_ts_cache_hit() {
        let cache = CompilationCache::new(100);
        let mut ctx = CompilerContext::new();
        ctx.add_domain("Person", 100);

        let expr = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);

        let graph1 = cache
            .get_or_compile(&expr, &mut ctx, compile_to_einsum_with_context)
            .expect("compile");

        let stats = cache.stats();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 0);

        let graph2 = cache
            .get_or_compile(&expr, &mut ctx, compile_to_einsum_with_context)
            .expect("compile");

        let stats = cache.stats();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 1);
        assert!(
            (stats.hit_rate() - 0.5).abs() < f64::EPSILON,
            "hit rate must be 0.5"
        );

        assert_eq!(graph1, graph2);
    }

    #[test]
    fn test_ts_cache_different_expressions() {
        let cache = CompilationCache::new(100);
        let mut ctx = CompilerContext::new();
        ctx.add_domain("Person", 100);

        let expr1 = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);
        let expr2 = TLExpr::pred("likes", vec![Term::var("x"), Term::var("y")]);

        let _ = cache
            .get_or_compile(&expr1, &mut ctx, compile_to_einsum_with_context)
            .expect("compile");
        let _ = cache
            .get_or_compile(&expr2, &mut ctx, compile_to_einsum_with_context)
            .expect("compile");

        let stats = cache.stats();
        assert_eq!(stats.misses, 2);
        assert_eq!(stats.hits, 0);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_ts_cache_eviction() {
        let cache = CompilationCache::new(2);
        let mut ctx = CompilerContext::new();
        ctx.add_domain("Person", 100);

        let _ = cache.get_or_compile(
            &TLExpr::pred("p1", vec![Term::var("x")]),
            &mut ctx,
            compile_to_einsum_with_context,
        );
        let _ = cache.get_or_compile(
            &TLExpr::pred("p2", vec![Term::var("x")]),
            &mut ctx,
            compile_to_einsum_with_context,
        );
        let _ = cache.get_or_compile(
            &TLExpr::pred("p3", vec![Term::var("x")]),
            &mut ctx,
            compile_to_einsum_with_context,
        );

        let stats = cache.stats();
        assert_eq!(stats.evictions, 1);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_ts_cache_clear() {
        let cache = CompilationCache::new(100);
        let mut ctx = CompilerContext::new();
        ctx.add_domain("Person", 100);

        let _ = cache.get_or_compile(
            &TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]),
            &mut ctx,
            compile_to_einsum_with_context,
        );

        assert_eq!(cache.len(), 1);
        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_ts_cache_stats() {
        let cache = CompilationCache::new(100);
        let stats = cache.stats();

        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.evictions, 0);
        assert_eq!(stats.current_entries, 0);
        assert_eq!(stats.hit_rate(), 0.0);
        assert_eq!(stats.total_lookups(), 0);
    }
}
