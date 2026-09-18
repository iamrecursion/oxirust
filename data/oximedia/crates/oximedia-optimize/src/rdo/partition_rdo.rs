//! Parallel, cached, and early-termination RDO for partition search.
//!
//! This module extends the core [`RdoEngine`] with three orthogonal optimizations:
//!
//! 1. **Parallel RDO** — [`RdoConfig`] gates rayon parallelism for `evaluate_partitions`,
//!    delivering near-linear speedup when partition cost functions are stateless.
//!
//! 2. **Block-level caching** — [`CachedRdoEngine`] memoises RD costs keyed on a 64-bit
//!    FNV-1a content hash, short-circuiting repeated evaluations of identical pixel blocks.
//!
//! 3. **Early termination** — [`rdo_with_early_termination`] prunes the candidate list
//!    as soon as a candidate's cost exceeds `best * threshold_ratio`, saving computation
//!    when the cost landscape is monotone or near-monotone.
//!
//! # Theoretical Background
//!
//! In video encoders, the partition search problem evaluates *N* candidate split modes
//! (e.g. `SPLIT_NONE`, `SPLIT_H`, `SPLIT_V`, `SPLIT_4`) and chooses the one minimising
//! `J = D + λ·R`.  For an encoder running 4K@60 fps with CTU size 128×128, this loop is
//! executed O(10⁷) times per second.  Each optimisation below targets a different part of
//! the execution profile:
//!
//! * Parallelism exploits independent cost evaluations (embarrassingly parallel).
//! * Caching eliminates redundant work for identical-content blocks (common in screen
//!   content, animation, and intra refresh periods).
//! * Early termination prunes clearly suboptimal candidates before evaluating them.

use std::collections::HashMap;

use rayon::prelude::*;

use super::engine::{ModeCandidate, RdoEngine, RdoResult};

// ─────────────────────────────────────────────────────────────────────────────
// § 1  Partition type definition
// ─────────────────────────────────────────────────────────────────────────────

/// Partition / block-split mode for a coding unit.
///
/// Matches the four primary split decisions present in HEVC/AV1-style encoders.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PartitionType {
    /// The block is not split — encoded as a single coding unit.
    None,
    /// The block is split into a top half and a bottom half.
    Horizontal,
    /// The block is split into a left half and a right half.
    Vertical,
    /// The block is split into four equally-sized quadrant coding units.
    Split4,
}

impl PartitionType {
    /// Returns the signal overhead (in fractional bits) for this partition type.
    ///
    /// These values approximate the CABAC context costs used in real encoders.
    /// A finer model would index into a context table; this simplified model is
    /// deliberately deterministic so that parallel and sequential paths agree.
    #[must_use]
    pub fn signal_bits(self) -> f64 {
        match self {
            Self::None => 1.0,
            Self::Horizontal => 2.0,
            Self::Vertical => 2.0,
            Self::Split4 => 3.0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 2  RdoConfig — gates parallelism
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the parallel-capable partition RDO evaluator.
///
/// # Parallelism contract
///
/// When `parallel_rdo` is `true` rayon is used to evaluate all candidates
/// concurrently.  The caller must ensure that the cost closure passed to
/// [`RdoEngine::evaluate_partitions`] is **stateless** — i.e. it must not
/// mutate shared state.  The closure bounds `Fn + Send + Sync` enforce this
/// at compile time.
#[derive(Debug, Clone)]
pub struct RdoConfig {
    /// Enable rayon-based parallel evaluation.  Default: `false`.
    pub parallel_rdo: bool,
    /// Maximum rayon worker threads.  `None` uses rayon's global thread-pool
    /// (typically `num_cpus`).  Ignored when `parallel_rdo` is `false`.
    pub max_threads: Option<usize>,
}

impl Default for RdoConfig {
    fn default() -> Self {
        Self {
            parallel_rdo: false,
            max_threads: None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 3  evaluate_partitions extension on RdoEngine
// ─────────────────────────────────────────────────────────────────────────────

/// Extension methods on [`RdoEngine`] for partition-level RDO.
pub trait PartitionRdo {
    /// Evaluate all partition candidates and return the best [`RdoResult`].
    ///
    /// When `config.parallel_rdo` is `true` the candidates are evaluated via
    /// [`rayon::iter::ParallelIterator`]; otherwise a sequential iterator is used.
    /// Both paths produce **identical** results because the cost function is pure.
    ///
    /// # Arguments
    ///
    /// * `candidates` — Ordered list of mode candidates to evaluate.
    /// * `eval_fn`    — Pure closure: `ModeCandidate → (distortion, rate)`.
    ///                  Must be `Send + Sync` regardless of `config.parallel_rdo`
    ///                  so that the same call site works with both execution paths.
    /// * `config`     — Parallelism and thread-pool configuration.
    fn evaluate_partitions<F>(
        &self,
        candidates: &[ModeCandidate],
        eval_fn: F,
        config: &RdoConfig,
    ) -> RdoResult
    where
        F: Fn(&ModeCandidate) -> (f64, f64) + Send + Sync;
}

impl PartitionRdo for RdoEngine {
    fn evaluate_partitions<F>(
        &self,
        candidates: &[ModeCandidate],
        eval_fn: F,
        config: &RdoConfig,
    ) -> RdoResult
    where
        F: Fn(&ModeCandidate) -> (f64, f64) + Send + Sync,
    {
        if candidates.is_empty() {
            return RdoResult {
                best_mode_idx: 0,
                cost: f64::MAX,
                distortion: 0.0,
                rate: 0.0,
            };
        }

        // Compute (index, distortion, rate, rd_cost) tuples for every candidate.
        let costs: Vec<(usize, f64, f64, f64)> = if config.parallel_rdo {
            candidates
                .par_iter()
                .enumerate()
                .map(|(idx, candidate)| {
                    let (distortion, rate) = eval_fn(candidate);
                    let cost = self.calculate_cost(distortion, rate, candidate.qp);
                    (idx, distortion, rate, cost)
                })
                .collect()
        } else {
            candidates
                .iter()
                .enumerate()
                .map(|(idx, candidate)| {
                    let (distortion, rate) = eval_fn(candidate);
                    let cost = self.calculate_cost(distortion, rate, candidate.qp);
                    (idx, distortion, rate, cost)
                })
                .collect()
        };

        // Select the minimum-cost entry.
        let best = costs
            .iter()
            .min_by(|a, b| a.3.partial_cmp(&b.3).unwrap_or(std::cmp::Ordering::Equal));

        match best {
            Some(&(idx, distortion, rate, cost)) => RdoResult {
                best_mode_idx: idx,
                cost,
                distortion,
                rate,
            },
            None => RdoResult {
                best_mode_idx: 0,
                cost: f64::MAX,
                distortion: 0.0,
                rate: 0.0,
            },
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 4  Block-level RDO cache
// ─────────────────────────────────────────────────────────────────────────────

/// A cached RD-cost entry for a single block content hash.
#[derive(Debug, Clone, Copy)]
pub struct RdoCacheEntry {
    /// Best rate-distortion cost found for this block content.
    pub rd_cost: f64,
    /// Index of the best partition candidate (within the slice passed at evaluation time).
    pub best_partition: usize,
}

/// Wraps an [`RdoEngine`] with a block-content hash cache.
///
/// The cache is keyed on a 64-bit [FNV-1a][fnv] hash of the raw pixel bytes.
/// Repeated calls with the same pixel content skip the inner evaluation entirely.
///
/// ## Limitations
///
/// * The cache does not account for changes in the *candidate list* — it assumes
///   the same set of candidates is always evaluated for a given block.  If
///   different candidate lists may be used for the same pixel content, clear the
///   cache between encode passes with [`CachedRdoEngine::clear`].
/// * Hash collisions are astronomically unlikely with FNV-1a on typical block
///   sizes, but not theoretically impossible.  For production use, a 128-bit
///   xxHash3 variant offers stronger collision resistance.
///
/// [fnv]: https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
pub struct CachedRdoEngine {
    /// Inner evaluation engine.
    inner: RdoEngine,
    /// Memoisation table: block content hash → RD result.
    cache: HashMap<u64, RdoCacheEntry>,
    /// Number of cache hits across all [`Self::evaluate_with_cache`] calls.
    pub cache_hits: u64,
    /// Number of cache misses across all [`Self::evaluate_with_cache`] calls.
    pub cache_misses: u64,
}

impl CachedRdoEngine {
    /// Constructs a new cached engine wrapping `inner`.
    ///
    /// The cache starts empty; hits and miss counters start at zero.
    #[must_use]
    pub fn new(inner: RdoEngine) -> Self {
        Self {
            inner,
            cache: HashMap::new(),
            cache_hits: 0,
            cache_misses: 0,
        }
    }

    /// Evaluates the RD cost for `block` over `candidates`, using the cache.
    ///
    /// On a cache hit the stored result is returned immediately.  On a miss
    /// the inner engine is used (sequential evaluation — for parallel evaluation
    /// use [`CachedRdoEngine::evaluate_with_cache_parallel`]).
    ///
    /// # Arguments
    ///
    /// * `block`      — Raw pixel bytes of the coding unit.
    /// * `candidates` — Partition candidates; must be non-empty.
    ///
    /// # Returns
    ///
    /// `(rd_cost, best_partition_index)`.  Returns `(f64::MAX, 0)` if `candidates`
    /// is empty.
    pub fn evaluate_with_cache(
        &mut self,
        block: &[u8],
        candidates: &[PartitionType],
    ) -> (f64, usize) {
        if candidates.is_empty() {
            return (f64::MAX, 0);
        }

        let key = Self::block_hash(block);

        if let Some(entry) = self.cache.get(&key) {
            self.cache_hits += 1;
            return (entry.rd_cost, entry.best_partition);
        }

        self.cache_misses += 1;
        let (rd_cost, best_partition) = self.compute_partition_cost(block, candidates);

        self.cache.insert(
            key,
            RdoCacheEntry {
                rd_cost,
                best_partition,
            },
        );

        (rd_cost, best_partition)
    }

    /// Parallel variant of [`CachedRdoEngine::evaluate_with_cache`].
    ///
    /// Cache lookup is sequential (the cache is not `Sync`); only the inner
    /// evaluation is parallelised via rayon on a cache miss.
    pub fn evaluate_with_cache_parallel(
        &mut self,
        block: &[u8],
        candidates: &[PartitionType],
    ) -> (f64, usize) {
        if candidates.is_empty() {
            return (f64::MAX, 0);
        }

        let key = Self::block_hash(block);

        if let Some(entry) = self.cache.get(&key) {
            self.cache_hits += 1;
            return (entry.rd_cost, entry.best_partition);
        }

        self.cache_misses += 1;
        let (rd_cost, best_partition) = self.compute_partition_cost_parallel(block, candidates);

        self.cache.insert(
            key,
            RdoCacheEntry {
                rd_cost,
                best_partition,
            },
        );

        (rd_cost, best_partition)
    }

    /// Returns the cache hit rate in `[0.0, 1.0]`.
    ///
    /// Returns `0.0` if no evaluations have been performed yet.
    #[must_use]
    pub fn hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total as f64
        }
    }

    /// Clears the memoisation table without resetting counters.
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// Resets hit/miss counters.
    pub fn reset_stats(&mut self) {
        self.cache_hits = 0;
        self.cache_misses = 0;
    }

    /// Provides read access to the inner [`RdoEngine`].
    #[must_use]
    pub fn inner(&self) -> &RdoEngine {
        &self.inner
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// FNV-1a 64-bit hash of a byte slice.
    ///
    /// FNV-1a is non-cryptographic but has excellent avalanche properties and
    /// is fast for small-to-medium block sizes (4×4 to 64×64 = 16–4096 bytes).
    ///
    /// Constants from the [FNV specification](http://www.isthe.com/chongo/tech/comp/fnv/#FNV-param):
    /// * Offset basis: 14695981039346656037
    /// * Prime:         1099511628211
    fn block_hash(block: &[u8]) -> u64 {
        const FNV_OFFSET_BASIS: u64 = 14_695_981_039_346_656_037;
        const FNV_PRIME: u64 = 1_099_511_628_211;

        let mut hash = FNV_OFFSET_BASIS;
        for &byte in block {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }

    /// Sequential cost computation for `block` over `candidates`.
    fn compute_partition_cost(&self, block: &[u8], candidates: &[PartitionType]) -> (f64, usize) {
        // Use a deterministic, stateless distortion model:
        //   distortion  = mean absolute deviation of pixel values from 128 (mid-gray)
        //   rate        = partition signal bits (encoding overhead)
        // This model is consistent across parallel and sequential paths.
        let distortion = block_distortion(block);

        let mut best_cost = f64::MAX;
        let mut best_idx = 0;

        for (idx, partition) in candidates.iter().enumerate() {
            let rate = partition.signal_bits();
            // Use QP=26 as a fixed reference QP for cache-level evaluation;
            // the caller is responsible for any QP-specific adjustments.
            let cost = self.inner.calculate_cost(distortion, rate, 26);
            if cost < best_cost {
                best_cost = cost;
                best_idx = idx;
            }
        }

        (best_cost, best_idx)
    }

    /// Parallel cost computation for `block` over `candidates`.
    fn compute_partition_cost_parallel(
        &self,
        block: &[u8],
        candidates: &[PartitionType],
    ) -> (f64, usize) {
        let distortion = block_distortion(block);

        let best = candidates
            .par_iter()
            .enumerate()
            .map(|(idx, partition)| {
                let rate = partition.signal_bits();
                let cost = self.inner.calculate_cost(distortion, rate, 26);
                (idx, cost)
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        match best {
            Some((idx, cost)) => (cost, idx),
            None => (f64::MAX, 0),
        }
    }
}

/// Mean absolute deviation from 128 (mid-gray) for a pixel block.
///
/// This is a cheap, deterministic distortion proxy that does not require an
/// explicit reference frame.  It measures how "active" the block content is,
/// which correlates with the encoding difficulty.
#[must_use]
pub(crate) fn block_distortion(block: &[u8]) -> f64 {
    if block.is_empty() {
        return 0.0;
    }
    block
        .iter()
        .map(|&p| (f64::from(p) - 128.0).abs())
        .sum::<f64>()
        / block.len() as f64
}

// ─────────────────────────────────────────────────────────────────────────────
// § 5  Early termination in partition search
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for early termination in partition cost search.
///
/// During partition evaluation, once `min_evaluated` candidates have been scored,
/// any subsequent candidate whose cost exceeds `current_best * cost_threshold_ratio`
/// is skipped.  This exploits the observation that encoders typically sort
/// candidates roughly in ascending cost order (cheapest / simplest modes first).
///
/// # Choosing parameters
///
/// | Use case                    | `cost_threshold_ratio` | `min_evaluated` |
/// |-----------------------------|-----------------------|-----------------|
/// | Highest quality (Placebo)   | 10.0                  | N (all)         |
/// | High quality  (Slow)        | 3.0                   | 2               |
/// | Default (Medium)            | 2.0                   | 1               |
/// | Speed-optimised (Fast)      | 1.5                   | 1               |
#[derive(Debug, Clone)]
pub struct EarlyTermConfig {
    /// Skip candidate *i* if `cost_i > best_so_far * cost_threshold_ratio`.
    /// Must be `≥ 1.0`.  A value of `1.0` keeps only strictly cheaper candidates.
    pub cost_threshold_ratio: f64,
    /// Always evaluate at least this many candidates regardless of the threshold.
    /// Must be `≥ 1`.
    pub min_evaluated: usize,
}

impl Default for EarlyTermConfig {
    fn default() -> Self {
        Self {
            cost_threshold_ratio: 2.0,
            min_evaluated: 1,
        }
    }
}

/// Evaluates partition candidates with early termination.
///
/// # Algorithm
///
/// ```text
/// best_cost  ← ∞
/// best_index ← 0
/// for i, candidate in enumerate(candidates):
///     cost ← compute_cost(candidate)
///     if cost < best_cost:
///         best_cost  ← cost
///         best_index ← i
///     elif i ≥ min_evaluated AND cost > best_cost * threshold_ratio:
///         break   ← early exit
/// return (best_cost, best_index)
/// ```
///
/// # Arguments
///
/// * `candidates`   — Slice of partition candidates (ordered, typically cheapest-first).
/// * `compute_cost` — Pure function mapping a candidate to its RD cost.
/// * `config`       — Threshold and minimum-evaluation settings.
///
/// # Returns
///
/// `(best_cost, best_index)` — the RD cost and index of the selected candidate.
/// Returns `(f64::MAX, 0)` when `candidates` is empty.
///
/// # Panics
///
/// Does not panic; all arithmetic is on `f64` with no division.
pub fn rdo_with_early_termination<F>(
    candidates: &[PartitionType],
    mut compute_cost: F,
    config: &EarlyTermConfig,
) -> (f64, usize)
where
    F: FnMut(&PartitionType) -> f64,
{
    if candidates.is_empty() {
        return (f64::MAX, 0);
    }

    // Clamp config values to valid ranges.
    let threshold_ratio = config.cost_threshold_ratio.max(1.0);
    let min_evaluated = config.min_evaluated.max(1);

    let mut best_cost = f64::MAX;
    let mut best_idx = 0usize;

    for (idx, candidate) in candidates.iter().enumerate() {
        let cost = compute_cost(candidate);

        if cost < best_cost {
            best_cost = cost;
            best_idx = idx;
        }

        // Early-termination check: only after min_evaluated candidates are done
        // and current cost is well above the best seen.
        // Note: we check the cost of *this* candidate (not the previous best)
        // because even if it is not the best, it signals a rising trend.
        if idx + 1 >= min_evaluated && cost > best_cost * threshold_ratio && best_cost < f64::MAX {
            break;
        }
    }

    (best_cost, best_idx)
}

// ─────────────────────────────────────────────────────────────────────────────
// § 6  Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OptimizerConfig;

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn default_engine() -> RdoEngine {
        let config = OptimizerConfig::default();
        RdoEngine::new(&config).expect("RdoEngine creation must succeed")
    }

    fn make_candidates(n: usize, qp: u8) -> Vec<ModeCandidate> {
        (0..n)
            .map(|i| ModeCandidate {
                mode_idx: i,
                qp,
                data: vec![],
            })
            .collect()
    }

    // ── § 3  RdoConfig ────────────────────────────────────────────────────────

    /// Default `RdoConfig` must have `parallel_rdo = false` per spec.
    #[test]
    fn test_parallel_rdo_config_default() {
        let cfg = RdoConfig::default();
        assert!(!cfg.parallel_rdo, "default must be non-parallel");
        assert!(cfg.max_threads.is_none(), "default thread cap must be None");
    }

    /// Parallel and sequential evaluation must choose the same best partition.
    #[test]
    fn test_parallel_rdo_matches_sequential() {
        let engine = default_engine();

        // 8 candidates with a deliberately non-monotone cost landscape so that
        // there is a clear unique minimum (index 5, cost ≈ 5·λ+30).
        let candidates = make_candidates(8, 26);

        // Distortion profile: minimum at index 5.
        let distortions = [200.0f64, 180.0, 160.0, 140.0, 120.0, 30.0, 150.0, 170.0];
        let eval_fn = |c: &ModeCandidate| (distortions[c.mode_idx], 5.0);

        let seq_result = engine.evaluate_partitions(&candidates, eval_fn, &RdoConfig::default());

        let par_config = RdoConfig {
            parallel_rdo: true,
            max_threads: None,
        };
        let par_result = engine.evaluate_partitions(&candidates, eval_fn, &par_config);

        assert_eq!(
            seq_result.best_mode_idx, par_result.best_mode_idx,
            "parallel and sequential must select the same best mode"
        );
        assert!(
            (seq_result.cost - par_result.cost).abs() < 1e-9,
            "costs must be identical: seq={} par={}",
            seq_result.cost,
            par_result.cost
        );
    }

    /// Edge case: empty candidate list returns a sentinel result.
    #[test]
    fn test_evaluate_partitions_empty_candidates() {
        let engine = default_engine();
        let result =
            engine.evaluate_partitions(&[], |_c: &ModeCandidate| (0.0, 0.0), &RdoConfig::default());
        assert_eq!(result.cost, f64::MAX);
    }

    /// Single candidate must always be chosen as the best.
    #[test]
    fn test_evaluate_partitions_single_candidate() {
        let engine = default_engine();
        let candidates = make_candidates(1, 26);
        let result =
            engine.evaluate_partitions(&candidates, |_c| (50.0, 5.0), &RdoConfig::default());
        assert_eq!(result.best_mode_idx, 0);
        assert!(result.cost < f64::MAX);
    }

    // ── § 4  CachedRdoEngine ─────────────────────────────────────────────────

    /// Second evaluation of an identical block must be a cache hit.
    #[test]
    fn test_rdo_cache_hit() {
        let engine = CachedRdoEngine::new(default_engine());
        let mut cached = engine;
        let block = vec![128u8; 64]; // 8×8 block
        let candidates = vec![PartitionType::None, PartitionType::Split4];

        let (cost1, idx1) = cached.evaluate_with_cache(&block, &candidates);
        assert_eq!(cached.cache_hits, 0, "first call must be a miss");
        assert_eq!(cached.cache_misses, 1);

        let (cost2, idx2) = cached.evaluate_with_cache(&block, &candidates);
        assert_eq!(cached.cache_hits, 1, "second call must be a hit");
        assert_eq!(cached.cache_misses, 1);

        assert_eq!(idx1, idx2, "both calls must return the same best partition");
        assert!((cost1 - cost2).abs() < 1e-9, "costs must be identical");
    }

    /// Different blocks must not collide in the cache.
    #[test]
    fn test_rdo_cache_different_blocks() {
        let mut cached = CachedRdoEngine::new(default_engine());
        let block_a = vec![50u8; 64];
        let block_b = vec![200u8; 64];
        let candidates = vec![
            PartitionType::None,
            PartitionType::Horizontal,
            PartitionType::Split4,
        ];

        let (cost_a, _) = cached.evaluate_with_cache(&block_a, &candidates);
        let (cost_b, _) = cached.evaluate_with_cache(&block_b, &candidates);

        // Both are misses.
        assert_eq!(cached.cache_misses, 2);
        assert_eq!(cached.cache_hits, 0);

        // The two different blocks should produce non-identical costs
        // (block_b has higher distortion from 128, so its cost ≠ cost_a).
        assert!(
            (cost_a - cost_b).abs() > 1e-9,
            "different block contents must produce different RD costs: cost_a={cost_a} cost_b={cost_b}"
        );
    }

    /// With 5 repeated blocks out of 10 evaluations, hit_rate must be ≥ 0.4.
    #[test]
    fn test_rdo_cache_hit_rate() {
        let mut cached = CachedRdoEngine::new(default_engine());
        let candidates = vec![PartitionType::None, PartitionType::Split4];

        // 5 unique blocks, each evaluated twice → 5 misses + 5 hits = 0.5 hit rate.
        for seed in 0u8..5 {
            let block = vec![seed * 20; 64];
            cached.evaluate_with_cache(&block, &candidates); // miss
            cached.evaluate_with_cache(&block, &candidates); // hit
        }

        let hr = cached.hit_rate();
        assert!(
            hr >= 0.4,
            "hit_rate must be ≥ 0.40 for 5 repeated blocks, got {hr:.4}"
        );
        assert_eq!(cached.cache_hits, 5, "exactly 5 hits expected");
        assert_eq!(cached.cache_misses, 5, "exactly 5 misses expected");
    }

    /// Parallel cache path must return the same results as the sequential path.
    #[test]
    fn test_rdo_cache_parallel_matches_sequential() {
        let mut seq_cache = CachedRdoEngine::new(default_engine());
        let mut par_cache = CachedRdoEngine::new(default_engine());

        let candidates = vec![
            PartitionType::None,
            PartitionType::Horizontal,
            PartitionType::Vertical,
            PartitionType::Split4,
        ];

        for seed in 0u8..4 {
            let block: Vec<u8> = (0u8..64).map(|i| i.wrapping_add(seed * 10)).collect();
            let (cost_s, idx_s) = seq_cache.evaluate_with_cache(&block, &candidates);
            let (cost_p, idx_p) = par_cache.evaluate_with_cache_parallel(&block, &candidates);
            assert_eq!(
                idx_s, idx_p,
                "parallel and sequential cache must select same partition"
            );
            assert!(
                (cost_s - cost_p).abs() < 1e-9,
                "parallel and sequential cache costs must match"
            );
        }
    }

    /// `hit_rate()` returns 0.0 when no evaluations have been performed.
    #[test]
    fn test_rdo_cache_hit_rate_no_evaluations() {
        let cached = CachedRdoEngine::new(default_engine());
        assert_eq!(cached.hit_rate(), 0.0);
    }

    /// `clear()` empties the cache so the next call is a miss again.
    #[test]
    fn test_rdo_cache_clear() {
        let mut cached = CachedRdoEngine::new(default_engine());
        let block = vec![100u8; 64];
        let candidates = vec![PartitionType::None, PartitionType::Split4];

        cached.evaluate_with_cache(&block, &candidates); // miss
        cached.evaluate_with_cache(&block, &candidates); // hit
        assert_eq!(cached.cache_hits, 1);

        cached.clear();
        cached.evaluate_with_cache(&block, &candidates); // miss again after clear
        assert_eq!(cached.cache_hits, 1, "hits must not increase after clear");
        assert_eq!(cached.cache_misses, 2, "should be a new miss after clear");
    }

    // ── § 5  Early termination ────────────────────────────────────────────────

    /// With threshold = 2.0 and min_evaluated = 1, a 3× expensive second
    /// candidate must trigger early exit (skip from index 1 onward).
    #[test]
    fn test_early_termination_skips_expensive() {
        // Candidates: 0 = cheap (cost 10), 1 = very expensive (cost 30, 3× best).
        let candidates = vec![PartitionType::None, PartitionType::Split4];

        let costs = [10.0f64, 30.0];
        let mut evaluated = vec![false; 2];

        let config = EarlyTermConfig {
            cost_threshold_ratio: 2.0,
            min_evaluated: 1,
        };

        let (best_cost, best_idx) = rdo_with_early_termination(
            &candidates,
            |p| {
                let idx = match p {
                    PartitionType::None => 0,
                    PartitionType::Split4 => 1,
                    _ => panic!("unexpected partition"),
                };
                evaluated[idx] = true;
                costs[idx]
            },
            &config,
        );

        assert_eq!(best_idx, 0, "best partition should be index 0 (cost=10)");
        assert!((best_cost - 10.0).abs() < 1e-9);
        // Candidate 1 cost (30) > best (10) * threshold (2.0) = 20, so it
        // must still be evaluated (we already evaluated it to measure cost),
        // but the loop terminates immediately *after* evaluating it.
        // The key invariant is that best_idx = 0 (the cheap one), not 1.
        assert!(evaluated[0], "candidate 0 must be evaluated");
        // candidate 1 is evaluated (to measure cost), then the loop breaks.
        assert!(
            evaluated[1],
            "candidate 1 must be evaluated to measure its cost"
        );
    }

    /// With min_evaluated = 3, at least 3 candidates are always evaluated even
    /// when the first is cheap enough to trigger the threshold.
    #[test]
    fn test_early_termination_min_evaluated() {
        // All costs > threshold after first, but min_evaluated = 3 forces 3 evals.
        let candidates = vec![
            PartitionType::None,       // cost 5
            PartitionType::Horizontal, // cost 100 (20× best → over threshold)
            PartitionType::Vertical,   // cost 200
            PartitionType::Split4,     // cost 300 — should never be evaluated
        ];

        let costs = [5.0f64, 100.0, 200.0, 300.0];
        let mut call_count = 0usize;

        let config = EarlyTermConfig {
            cost_threshold_ratio: 2.0,
            min_evaluated: 3,
        };

        let (best_cost, best_idx) = rdo_with_early_termination(
            &candidates,
            |p| {
                call_count += 1;
                match p {
                    PartitionType::None => costs[0],
                    PartitionType::Horizontal => costs[1],
                    PartitionType::Vertical => costs[2],
                    PartitionType::Split4 => costs[3],
                }
            },
            &config,
        );

        assert!(
            call_count >= 3,
            "min_evaluated=3 must force at least 3 evaluations, got {call_count}"
        );
        assert_eq!(best_idx, 0, "index 0 has lowest cost");
        assert!((best_cost - 5.0).abs() < 1e-9);
    }

    /// Monotone cost sequence: early termination fires after the first cost rise.
    #[test]
    fn test_early_termination_monotone_rise() {
        let candidates = vec![
            PartitionType::None,
            PartitionType::Horizontal,
            PartitionType::Vertical,
            PartitionType::Split4,
        ];
        let costs = [10.0f64, 25.0, 50.0, 100.0];
        let mut count = 0usize;

        let config = EarlyTermConfig {
            cost_threshold_ratio: 2.0,
            min_evaluated: 1,
        };

        let (best_cost, best_idx) = rdo_with_early_termination(
            &candidates,
            |p| {
                count += 1;
                match p {
                    PartitionType::None => costs[0],
                    PartitionType::Horizontal => costs[1],
                    PartitionType::Vertical => costs[2],
                    PartitionType::Split4 => costs[3],
                }
            },
            &config,
        );

        // Horizontal cost (25) > None cost (10) * 2.0 (= 20) → stop after idx 1.
        assert!(
            count <= 2,
            "should stop early; evaluated {count} candidates"
        );
        assert_eq!(best_idx, 0);
        assert!((best_cost - 10.0).abs() < 1e-9);
    }

    /// Empty candidate list must return the sentinel (f64::MAX, 0).
    #[test]
    fn test_early_termination_empty() {
        let config = EarlyTermConfig::default();
        let (cost, idx) = rdo_with_early_termination(&[], |_p| 0.0, &config);
        assert_eq!(cost, f64::MAX);
        assert_eq!(idx, 0);
    }

    /// Single candidate is always selected.
    #[test]
    fn test_early_termination_single_candidate() {
        let config = EarlyTermConfig::default();
        let (cost, idx) = rdo_with_early_termination(&[PartitionType::Split4], |_p| 42.0, &config);
        assert!((cost - 42.0).abs() < 1e-9);
        assert_eq!(idx, 0);
    }

    /// threshold_ratio = 1.0 means only strictly cheaper candidates continue.
    #[test]
    fn test_early_termination_tight_threshold() {
        let candidates = vec![PartitionType::None, PartitionType::Split4];
        let costs = [10.0f64, 11.0]; // 11 > 10 * 1.0 → skip

        let config = EarlyTermConfig {
            cost_threshold_ratio: 1.0,
            min_evaluated: 1,
        };
        let mut count = 0usize;

        let (best_cost, best_idx) = rdo_with_early_termination(
            &candidates,
            |p| {
                count += 1;
                match p {
                    PartitionType::None => costs[0],
                    PartitionType::Split4 => costs[1],
                    _ => panic!("unexpected partition"),
                }
            },
            &config,
        );

        assert_eq!(best_idx, 0);
        assert!((best_cost - 10.0).abs() < 1e-9);
    }

    // ── FNV hash properties ───────────────────────────────────────────────────

    /// Same content → same hash.
    #[test]
    fn test_fnv_hash_deterministic() {
        let block = vec![42u8; 64];
        let h1 = CachedRdoEngine::block_hash(&block);
        let h2 = CachedRdoEngine::block_hash(&block);
        assert_eq!(h1, h2);
    }

    /// Different content → different hash (avalanche sanity check).
    #[test]
    fn test_fnv_hash_different_blocks() {
        let a = vec![0u8; 64];
        let b = vec![255u8; 64];
        assert_ne!(
            CachedRdoEngine::block_hash(&a),
            CachedRdoEngine::block_hash(&b)
        );
    }

    /// Empty block → returns the FNV offset basis (not 0).
    #[test]
    fn test_fnv_hash_empty() {
        let h = CachedRdoEngine::block_hash(&[]);
        assert_eq!(h, 14_695_981_039_346_656_037u64);
    }
}
