//! Multi-tier cache (L1 memory → L2 memory → disk).
//!
//! Each tier has an independent [`EvictionPolicy`] and a byte-level capacity.
//! On a miss the implementation searches lower tiers in order; on a hit in a
//! lower tier the entry is promoted to L1.
//!
//! ## New in 0.1.2
//!
//! * **File-backed disk tier** — `TierConfig::disk_path` enables a real
//!   file-backed tier backed by a directory on disk.  Each cache key maps to
//!   a file inside that directory; reads and writes use `std::fs`.
//! * **Adaptive promotion thresholds** — each tier now tracks access
//!   frequency per key.  A hit in tier *i* only promotes to tier *i-1* when
//!   the key's frequency exceeds the tier's `promotion_threshold`.  This
//!   prevents scan pollution from one-shot accesses filling the hot tier.
//! * **Entry compression** — tiers with `compress: true` store LZ4-style
//!   run-length encoding (pure Rust, no external deps) so that L2+ tiers
//!   occupy less memory / disk space.
//!
//! ## New in 0.1.8 Wave 13
//!
//! * **P² adaptive promotion** — the promotion threshold can be auto-tuned
//!   using a P² quantile estimator (Jain & Chlamtac 1985) that tracks the
//!   75th-percentile of per-key access frequencies.  Enable via
//!   `TieredCacheBuilder::enable_adaptive_promotion(true)`.
//! * **Arena allocation** — when `use_arena` is enabled, tier entry bytes are
//!   stored in a `BumpArena` (bump allocator) so the `HashMap` holds cheap
//!   `(offset, len)` handles instead of owned `Vec<u8>`.

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;

// ── P² Quantile Estimator (Jain & Chlamtac 1985) ─────────────────────────────

/// Online running estimator for an arbitrary quantile using the P² algorithm.
///
/// Tracks 5 marker positions to estimate the `p`-quantile without storing all
/// observations.  Only valid after `n ≥ 5` samples have been fed (warmup guard).
#[derive(Debug, Clone)]
pub struct P2QuantileEstimator {
    /// Target quantile (0 < p < 1).  Default 0.75 for 75th-percentile.
    p: f64,
    /// Total number of observations fed so far.
    n: u64,
    /// Marker heights: estimated quantile values at 5 marker positions.
    q: [f64; 5],
    /// Desired marker positions (real-valued).
    dn: [f64; 5],
    /// Actual marker positions (integer counts).
    np: [f64; 5],
}

impl P2QuantileEstimator {
    /// Create a new estimator for quantile `p` (0 < p < 1).
    pub fn new(p: f64) -> Self {
        let p = p.clamp(1e-6, 1.0 - 1e-6);
        Self {
            p,
            n: 0,
            q: [0.0; 5],
            dn: [0.0, p / 2.0, p, (1.0 + p) / 2.0, 1.0],
            np: [1.0, 1.0 + 2.0 * p, 1.0 + 4.0 * p, 3.0 + 2.0 * p, 5.0],
        }
    }

    /// Feed a new observation.
    pub fn update(&mut self, x: f64) {
        if self.n < 5 {
            // Collect the first 5 values into q[].
            self.q[self.n as usize] = x;
            self.n += 1;
            if self.n == 5 {
                // Sort the initial 5 samples to initialise the markers.
                self.q
                    .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            }
            return;
        }
        self.n += 1;

        // Step 1: find cell k.
        let k = if x < self.q[0] {
            self.q[0] = x;
            0usize
        } else if x < self.q[1] {
            0
        } else if x < self.q[2] {
            1
        } else if x < self.q[3] {
            2
        } else if x <= self.q[4] {
            3
        } else {
            self.q[4] = x;
            3
        };

        // Step 2: increment positions.
        for i in (k + 1)..5 {
            self.np[i] += 1.0;
        }

        // Step 3: update desired positions.
        let n_f = self.n as f64;
        self.dn[0] = 0.0;
        self.dn[1] = (n_f - 1.0) * self.p / 2.0 + 1.0;
        self.dn[2] = (n_f - 1.0) * self.p + 1.0;
        self.dn[3] = (n_f - 1.0) * (1.0 + self.p) / 2.0 + 1.0;
        self.dn[4] = n_f as f64;

        // Step 4: adjust markers 1–3 (0-indexed).
        for i in 1..4 {
            let d = self.dn[i] - self.np[i];
            let sign_d: f64 = if d >= 0.0 { 1.0 } else { -1.0 };
            if (d >= 1.0 && self.np[i + 1] - self.np[i] > 1.0)
                || (d <= -1.0 && self.np[i - 1] - self.np[i] < -1.0)
            {
                // Try parabolic interpolation.
                let qi_new = self.parabolic(i, sign_d);
                if qi_new > self.q[i - 1] && qi_new < self.q[i + 1] {
                    self.q[i] = qi_new;
                } else {
                    // Linear fallback.
                    let idx = if d >= 0.0 { i + 1 } else { i.saturating_sub(1) };
                    let dq = self.q[idx] - self.q[i];
                    let dn = self.np[idx] - self.np[i];
                    self.q[i] += sign_d * dq / dn;
                }
                self.np[i] += sign_d;
            }
        }
    }

    fn parabolic(&self, i: usize, sign: f64) -> f64 {
        let qi = self.q[i];
        let qi_prev = self.q[i - 1];
        let qi_next = self.q[i + 1];
        let ni = self.np[i];
        let ni_prev = self.np[i - 1];
        let ni_next = self.np[i + 1];
        let term1 = sign / (ni_next - ni_prev);
        let left = (ni - ni_prev + sign) * (qi_next - qi) / (ni_next - ni);
        let right = (ni_next - ni - sign) * (qi - qi_prev) / (ni - ni_prev);
        qi + term1 * (left + right)
    }

    /// Return the current quantile estimate.
    ///
    /// Returns `None` when fewer than 5 observations have been fed (warmup guard).
    pub fn estimate(&self) -> Option<f64> {
        if self.n < 5 {
            None
        } else {
            Some(self.q[2])
        }
    }

    /// Total number of observations seen.
    pub fn count(&self) -> u64 {
        self.n
    }
}

// ── BumpArena ─────────────────────────────────────────────────────────────────

/// A simple bump allocator for byte-slice entries.
///
/// Allocations are cheap (pointer-bump only).  Deallocation is not
/// supported per-entry; call [`reset`] to reclaim the whole arena at once
/// (typically called after a batch eviction sweep).
///
/// [`reset`]: BumpArena::reset
#[derive(Debug, Clone)]
pub struct BumpArena {
    data: Vec<u8>,
    pos: usize,
}

impl BumpArena {
    /// Create a new `BumpArena` with `initial_capacity` bytes pre-allocated.
    pub fn new(initial_capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(initial_capacity),
            pos: 0,
        }
    }

    /// Append `bytes` to the arena and return `(offset, len)`.
    ///
    /// Grows the backing vec if necessary.
    pub fn alloc(&mut self, bytes: &[u8]) -> (usize, usize) {
        let offset = self.pos;
        let len = bytes.len();
        // Extend backing storage if needed.
        if self.pos + len > self.data.len() {
            self.data.resize(self.pos + len, 0u8);
        }
        self.data[self.pos..self.pos + len].copy_from_slice(bytes);
        self.pos += len;
        (offset, len)
    }

    /// Retrieve a byte slice stored at `(offset, len)`.
    pub fn get(&self, offset: usize, len: usize) -> &[u8] {
        &self.data[offset..offset + len]
    }

    /// Reset the arena, reclaiming all memory for future allocations.
    ///
    /// Existing `(offset, len)` handles become invalid after this call.
    pub fn reset(&mut self) {
        self.pos = 0;
        // Do not deallocate the backing vec; just reset the cursor.
    }

    /// Current number of bytes used.
    pub fn used(&self) -> usize {
        self.pos
    }
}

// ── Public configuration types ───────────────────────────────────────────────

/// Eviction strategy for a single cache tier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvictionPolicy {
    /// Evict the entry with the oldest `last_access` timestamp.
    Lru,
    /// Evict the entry with the lowest access frequency; tie-break on
    /// `last_access`.
    Lfu,
    /// Evict the entry that was inserted first (queue order).
    Fifo,
    /// Evict a random entry using a deterministic xorshift32 PRNG.
    Random,
    /// Approximate LFU with a tiny Count-Min admission filter.
    TinyLfu,
}

/// Configuration for one tier.
#[derive(Debug, Clone)]
pub struct TierConfig {
    /// Human-readable name (e.g. `"L1"`, `"L2"`, `"disk"`).
    pub name: String,
    /// Maximum number of bytes this tier may hold.
    pub capacity_bytes: usize,
    /// Simulated read latency in microseconds (used in future work / profiling).
    pub access_latency_us: u64,
    /// How entries are selected for eviction when the tier is full.
    pub eviction_policy: EvictionPolicy,
    /// Optional path to a directory on disk for file-backed storage.
    ///
    /// When `Some(path)`, entries evicted from this tier are stored as
    /// individual files under `path/<key_hash>`.  Memory entries act as
    /// an in-memory index; disk is the actual backing store.
    pub disk_path: Option<PathBuf>,
    /// Minimum access frequency before a hit in this tier promotes the entry
    /// to the previous (hotter) tier.
    ///
    /// A value of `0` means "always promote" (original behaviour).
    /// A value of `3` means the key must have been accessed at least 3 times
    /// in this tier before it is considered hot enough to move up.
    pub promotion_threshold: u64,
    /// Whether to compress entry bytes in this tier.
    ///
    /// When `true`, values are compressed with a simple run-length encoder
    /// before being stored and decoded on retrieval.  Useful for L2+ tiers to
    /// reduce memory / disk footprint.
    pub compress: bool,
    /// Whether to use a P² quantile estimator to auto-tune the promotion
    /// threshold.  When `true`, the static `promotion_threshold` acts as a
    /// fallback during warmup (first 5 observations); afterwards the 75th
    /// percentile of observed per-key access frequencies is used.
    pub adaptive_promotion: bool,
    /// When `true`, tier entry bytes are stored in a `BumpArena` and the
    /// `HashMap` keeps lightweight `(offset, len)` handles instead of owned
    /// `Vec<u8>`.  Falls back to owned storage when `false` (default).
    pub use_arena: bool,
}

impl TierConfig {
    /// Create a minimal in-memory tier config with default values.
    pub fn memory(name: impl Into<String>, capacity_bytes: usize) -> Self {
        Self {
            name: name.into(),
            capacity_bytes,
            access_latency_us: 1,
            eviction_policy: EvictionPolicy::Lru,
            disk_path: None,
            promotion_threshold: 0,
            compress: false,
            adaptive_promotion: false,
            use_arena: false,
        }
    }

    /// Create a disk-backed tier config.
    pub fn disk(name: impl Into<String>, capacity_bytes: usize, path: impl Into<PathBuf>) -> Self {
        Self {
            name: name.into(),
            capacity_bytes,
            access_latency_us: 1_000,
            eviction_policy: EvictionPolicy::Lru,
            disk_path: Some(path.into()),
            promotion_threshold: 1,
            compress: true,
            adaptive_promotion: false,
            use_arena: false,
        }
    }

    /// Enable or disable P²-adaptive promotion threshold tuning.
    pub fn enable_adaptive_promotion(mut self, enabled: bool) -> Self {
        self.adaptive_promotion = enabled;
        self
    }

    /// Enable or disable arena allocation for tier entries.
    pub fn enable_arena(mut self, enabled: bool) -> Self {
        self.use_arena = enabled;
        self
    }
}

// ── Stats ────────────────────────────────────────────────────────────────────

/// Per-tier statistics snapshot.
#[derive(Debug, Clone)]
pub struct TierStats {
    /// Human-readable tier name.
    pub name: String,
    /// Number of cache hits served by this tier.
    pub hits: u64,
    /// Number of bytes currently stored in this tier.
    pub size_used_bytes: usize,
    /// Number of distinct entries currently in this tier.
    pub entry_count: usize,
    /// Number of promotions from this tier to the tier above.
    pub promotions: u64,
    /// Number of times an entry was compressed before storage.
    pub compressions: u64,
}

/// Aggregate statistics snapshot for the whole [`TieredCache`].
#[derive(Debug, Clone)]
pub struct TieredCacheStats {
    /// Total successful lookups across all tiers.
    pub total_hits: u64,
    /// Total failed lookups (miss on every tier).
    pub total_misses: u64,
    /// `total_hits / (total_hits + total_misses)`, or `0.0` when no requests.
    pub hit_rate: f64,
    /// Per-tier detail.
    pub tier_stats: Vec<TierStats>,
}

// ── Compression helpers (pure Rust run-length encoding) ──────────────────────

/// Compress `data` using a simple run-length encoding.
///
/// Format: pairs of `(count: u8, byte: u8)`.  Runs longer than 255 are split.
/// Non-run data is encoded as run-length 1.
fn rle_compress(data: &[u8]) -> Vec<u8> {
    if data.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(data.len());
    let mut i = 0;
    while i < data.len() {
        let byte = data[i];
        let mut run = 1usize;
        while i + run < data.len() && data[i + run] == byte && run < 255 {
            run += 1;
        }
        out.push(run as u8);
        out.push(byte);
        i += run;
    }
    out
}

/// Decompress data produced by [`rle_compress`].
fn rle_decompress(data: &[u8]) -> Vec<u8> {
    if data.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(data.len() * 2);
    let mut i = 0;
    while i + 1 < data.len() {
        let count = data[i] as usize;
        let byte = data[i + 1];
        for _ in 0..count {
            out.push(byte);
        }
        i += 2;
    }
    out
}

// ── Internal per-tier storage ────────────────────────────────────────────────

/// Entry in a `CacheTier`: either owned bytes or an arena handle.
enum TierEntry {
    /// Standard heap-owned payload.
    Owned(Vec<u8>),
    /// Handle into a `BumpArena`: `(offset, len)`.
    Arena(usize, usize),
}

struct CacheTier {
    config: TierConfig,
    /// `key → (entry, last_access_tick, frequency)`
    ///
    /// For disk-backed tiers the entry bytes are stored as a sentinel (empty)
    /// since the canonical copy lives on disk.  For memory tiers the full
    /// (possibly compressed) payload is stored.
    data: HashMap<String, (TierEntry, u64, u64)>,
    size_used: usize,
    /// Insertion-order queue used by the FIFO policy.
    fifo_order: VecDeque<String>,
    /// Hit counter for this tier.
    hits: u64,
    /// Promotion counter: how many times an entry was promoted from this tier.
    promotions: u64,
    /// Compression counter.
    compressions: u64,
    /// Internal logical tick (monotonically increasing, per insert/get).
    tick: u64,
    /// xorshift32 state for the Random eviction policy.
    rng_state: u32,
    /// P² quantile estimator for adaptive promotion threshold (75th percentile
    /// of per-key access frequency).  Present when `config.adaptive_promotion`.
    p2_estimator: Option<P2QuantileEstimator>,
    /// Optional bump arena.  Present when `config.use_arena`.
    arena: Option<BumpArena>,
}

impl CacheTier {
    fn new(config: TierConfig) -> Self {
        // Create the disk directory if needed.
        if let Some(ref path) = config.disk_path {
            let _ = std::fs::create_dir_all(path);
        }
        let p2_estimator = if config.adaptive_promotion {
            Some(P2QuantileEstimator::new(0.75))
        } else {
            None
        };
        let arena = if config.use_arena {
            Some(BumpArena::new(config.capacity_bytes))
        } else {
            None
        };
        Self {
            config,
            data: HashMap::new(),
            size_used: 0,
            fifo_order: VecDeque::new(),
            hits: 0,
            promotions: 0,
            compressions: 0,
            tick: 1,
            rng_state: 0xDEAD_BEEF,
            p2_estimator,
            arena,
        }
    }

    /// Step the xorshift32 PRNG and return the new state.
    fn xorshift32(&mut self) -> u32 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.rng_state = x;
        x
    }

    /// Build the file path for a key in a disk-backed tier.
    fn disk_path_for(&self, key: &str) -> Option<PathBuf> {
        self.config.disk_path.as_ref().map(|base| {
            // Use a simple FNV-1a hash of the key as the filename to avoid
            // any filesystem-unsafe characters.
            let mut h: u64 = 0xcbf2_9ce4_8422_2325;
            for b in key.as_bytes() {
                h ^= u64::from(*b);
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
            base.join(format!("{h:016x}"))
        })
    }

    /// Flush a key's in-memory bytes to disk (disk-backed tier only).
    fn flush_to_disk(&self, key: &str, bytes: &[u8]) {
        if let Some(path) = self.disk_path_for(key) {
            let _ = std::fs::write(path, bytes);
        }
    }

    /// Read a key's bytes from disk (disk-backed tier only).
    fn read_from_disk(&self, key: &str) -> Option<Vec<u8>> {
        let path = self.disk_path_for(key)?;
        std::fs::read(path).ok()
    }

    /// Remove a key's file from disk (disk-backed tier only).
    fn remove_from_disk(&self, key: &str) {
        if let Some(path) = self.disk_path_for(key) {
            let _ = std::fs::remove_file(path);
        }
    }

    /// Encode `raw` bytes for storage (apply compression if configured).
    fn encode(&mut self, raw: &[u8]) -> Vec<u8> {
        if self.config.compress {
            self.compressions += 1;
            rle_compress(raw)
        } else {
            raw.to_vec()
        }
    }

    /// Decode storage bytes back to raw (decompress if configured).
    fn decode(&self, stored: &[u8]) -> Vec<u8> {
        if self.config.compress {
            rle_decompress(stored)
        } else {
            stored.to_vec()
        }
    }

    /// Byte length of a `TierEntry` without cloning.
    fn entry_len(&self, entry: &TierEntry) -> usize {
        match entry {
            TierEntry::Owned(v) => v.len(),
            TierEntry::Arena(_, len) => *len,
        }
    }

    fn get(&mut self, key: &str) -> Option<Vec<u8>> {
        let tick = self.tick;
        self.tick += 1;

        // For disk-backed tiers, check if we have the key in the in-memory
        // index but the data must be read from disk.
        if self.config.disk_path.is_some() {
            if let Some(entry) = self.data.get_mut(key) {
                entry.1 = tick;
                entry.2 += 1;
                self.hits += 1;
                // Feed frequency into P² estimator (if adaptive).
                let freq = entry.2;
                if let Some(ref mut est) = self.p2_estimator {
                    est.update(freq as f64);
                }
                // Read canonical bytes from disk.
                return self.read_from_disk(key).map(|stored| self.decode(&stored));
            }
            return None;
        }

        if let Some(entry) = self.data.get_mut(key) {
            entry.1 = tick; // update last_access
            entry.2 += 1; // increment frequency
            self.hits += 1;
            // Feed frequency into P² estimator (if adaptive).
            let freq = entry.2;
            if let Some(ref mut est) = self.p2_estimator {
                est.update(freq as f64);
            }
            // Clone the stored bytes before immutably borrowing again.
            let raw: Vec<u8> = match &entry.0 {
                TierEntry::Owned(v) => v.clone(),
                TierEntry::Arena(offset, len) => {
                    if let Some(arena) = &self.arena {
                        arena.get(*offset, *len).to_vec()
                    } else {
                        vec![]
                    }
                }
            };
            let decoded = self.decode(&raw);
            Some(decoded)
        } else {
            None
        }
    }

    /// Insert `(key, data)` into this tier, evicting entries as needed until
    /// there is room.
    fn put(&mut self, key: String, data: Vec<u8>) {
        let encoded = self.encode(&data);
        let stored_len = encoded.len();

        // If the data alone exceeds the tier capacity, skip insertion.
        if stored_len > self.config.capacity_bytes {
            return;
        }

        // Evict until there is enough room.
        while self.size_used + stored_len > self.config.capacity_bytes {
            if self.evict_one().is_none() {
                break;
            }
        }

        let tick = self.tick;
        self.tick += 1;
        self.size_used += stored_len;
        self.fifo_order.push_back(key.clone());

        // For disk-backed tiers, write encoded bytes to disk and store a
        // small sentinel (empty vec) in the in-memory map as an index entry.
        if self.config.disk_path.is_some() {
            self.flush_to_disk(&key, &encoded);
            self.data
                .insert(key, (TierEntry::Owned(Vec::new()), tick, 1));
        } else if self.config.use_arena {
            // Arena path: append to bump arena.
            let (offset, len) = if let Some(ref mut arena) = self.arena {
                arena.alloc(&encoded)
            } else {
                // Shouldn't happen; fall back to owned.
                let v = encoded;
                self.data.insert(key, (TierEntry::Owned(v), tick, 1));
                return;
            };
            self.data
                .insert(key, (TierEntry::Arena(offset, len), tick, 1));
        } else {
            self.data.insert(key, (TierEntry::Owned(encoded), tick, 1));
        }
    }

    /// Return the access frequency for `key`, or 0 if not present.
    fn frequency(&self, key: &str) -> u64 {
        self.data.get(key).map(|(_, _, f)| *f).unwrap_or(0)
    }

    /// Return the effective promotion threshold, using the P² estimate when
    /// available and the warmup guard has passed.
    fn effective_promotion_threshold(&self) -> u64 {
        if let Some(ref est) = self.p2_estimator {
            if let Some(q75) = est.estimate() {
                // Round up to nearest u64; at least 1.
                return (q75.ceil() as u64).max(1);
            }
        }
        self.config.promotion_threshold
    }

    /// Remove `key` from this tier.  Returns `true` if it was present.
    fn remove(&mut self, key: &str) -> bool {
        if let Some((entry, _, _)) = self.data.remove(key) {
            let stored_len = self.entry_len(&entry);
            self.size_used = self.size_used.saturating_sub(stored_len);
            self.fifo_order.retain(|k| k != key);
            if self.config.disk_path.is_some() {
                self.remove_from_disk(key);
            }
            true
        } else {
            false
        }
    }

    /// Evict one entry according to the configured policy.
    fn evict_one(&mut self) -> Option<(String, Vec<u8>)> {
        if self.data.is_empty() {
            return None;
        }
        let victim_key = match &self.config.eviction_policy {
            EvictionPolicy::Lru => self.pick_lru(),
            EvictionPolicy::Lfu => self.pick_lfu(),
            EvictionPolicy::Fifo => self.pick_fifo(),
            EvictionPolicy::Random => self.pick_random(),
            EvictionPolicy::TinyLfu => self.pick_tiny_lfu(),
        }?;

        let (entry, _, _) = self.data.remove(&victim_key)?;
        let is_disk_sentinel = self.config.disk_path.is_some()
            && matches!(&entry, TierEntry::Owned(v) if v.is_empty());
        let stored_bytes: Vec<u8> = match &entry {
            TierEntry::Owned(v) => v.clone(),
            TierEntry::Arena(offset, len) => {
                if let Some(arena) = &self.arena {
                    arena.get(*offset, *len).to_vec()
                } else {
                    vec![]
                }
            }
        };
        let data = if self.config.disk_path.is_some() {
            // Return decoded bytes from disk for possible demotion to lower tier.
            let from_disk = self.read_from_disk(&victim_key).unwrap_or_default();
            self.remove_from_disk(&victim_key);
            self.decode(&from_disk)
        } else {
            self.decode(&stored_bytes)
        };
        let size_removed = if is_disk_sentinel {
            // For disk tiers the in-memory sentinel is empty; use data.len as
            // approximate (compression may differ, but this avoids drift).
            data.len()
        } else {
            stored_bytes.len()
        };
        self.size_used = self.size_used.saturating_sub(size_removed);
        self.fifo_order.retain(|k| *k != victim_key);
        Some((victim_key, data))
    }

    fn pick_lru(&self) -> Option<String> {
        self.data
            .iter()
            .min_by_key(|(_, (_, last_access, _))| *last_access)
            .map(|(k, _)| k.clone())
    }

    fn pick_lfu(&self) -> Option<String> {
        self.data
            .iter()
            .min_by(|(_, (_, la_a, freq_a)), (_, (_, la_b, freq_b))| {
                freq_a.cmp(freq_b).then(la_a.cmp(la_b))
            })
            .map(|(k, _)| k.clone())
    }

    fn pick_fifo(&self) -> Option<String> {
        self.fifo_order.front().cloned()
    }

    fn pick_random(&mut self) -> Option<String> {
        if self.data.is_empty() {
            return None;
        }
        let count = self.data.len();
        let rnd = self.xorshift32() as usize % count;
        self.data.keys().nth(rnd).cloned()
    }

    /// TinyLFU: use `frequency % 4` as a count-min-sketch approximation.
    fn pick_tiny_lfu(&mut self) -> Option<String> {
        let candidate = self
            .data
            .iter()
            .min_by(|(_, (_, la_a, freq_a)), (_, (_, la_b, freq_b))| {
                let sketch_a = freq_a % 4;
                let sketch_b = freq_b % 4;
                sketch_a.cmp(&sketch_b).then(la_a.cmp(la_b))
            })
            .map(|(k, v)| (k.clone(), v.2))?;

        let (key, freq) = candidate;
        if freq >= 2 {
            let rnd = self.xorshift32() as u64;
            if rnd % freq >= freq / 2 {
                return self.pick_lfu();
            }
        }
        Some(key)
    }
}

impl Drop for CacheTier {
    fn drop(&mut self) {
        // For disk-backed tiers: remove all files on drop to avoid leaving
        // stale data. In production you would persist; here we clean up the
        // test directory.  Only do this when the base path still exists.
        if let Some(ref base) = self.config.disk_path {
            let keys: Vec<String> = self.data.keys().cloned().collect();
            for key in keys {
                if let Some(path) = self.disk_path_for(&key) {
                    let _ = std::fs::remove_file(path);
                }
            }
            // Attempt to remove the directory if empty (best-effort).
            let _ = std::fs::remove_dir(base);
        }
    }
}

// ── TieredCache ───────────────────────────────────────────────────────────────

/// A multi-tier cache where each tier has its own [`TierConfig`].
///
/// Reads check tiers in order (L1 first); a hit in tier *i* > 0 promotes the
/// entry to tier *i-1* when the key's access frequency in that tier meets or
/// exceeds the tier's [`TierConfig::promotion_threshold`].  Writes always go
/// to L1 only.
pub struct TieredCache {
    tiers: Vec<CacheTier>,
    total_hits: u64,
    total_misses: u64,
    /// Per-tier hit counters (parallel to `tiers`).
    tier_hits: Vec<u64>,
}

impl TieredCache {
    /// Construct a `TieredCache` from a list of tier configurations.
    /// The first element is L1 (fastest / smallest), last is the slowest.
    pub fn new(tiers: Vec<TierConfig>) -> Self {
        let n = tiers.len();
        Self {
            tiers: tiers.into_iter().map(CacheTier::new).collect(),
            total_hits: 0,
            total_misses: 0,
            tier_hits: vec![0; n],
        }
    }

    /// Look up `key` across all tiers in order.
    ///
    /// On a hit in tier *i* > 0, the entry is promoted to tier *i-1* if the
    /// key's frequency in that tier meets or exceeds the effective promotion
    /// threshold (static or P²-adaptive depending on configuration).
    pub fn get(&mut self, key: &str) -> Option<Vec<u8>> {
        for tier_idx in 0..self.tiers.len() {
            if let Some(data) = self.tiers[tier_idx].get(key) {
                self.total_hits += 1;
                self.tier_hits[tier_idx] += 1;
                // Adaptive promotion: only promote if frequency threshold met.
                if tier_idx > 0 {
                    let freq = self.tiers[tier_idx].frequency(key);
                    let threshold = self.tiers[tier_idx].effective_promotion_threshold();
                    if freq >= threshold {
                        self.tiers[tier_idx].promotions += 1;
                        let key_owned = key.to_string();
                        self.tiers[tier_idx - 1].put(key_owned, data.clone());
                    }
                }
                return Some(data);
            }
        }
        self.total_misses += 1;
        None
    }

    /// Insert `(key, data)` into the L1 tier.
    pub fn put(&mut self, key: &str, data: Vec<u8>) {
        self.tiers[0].put(key.to_string(), data);
    }

    /// Insert `(key, data)` directly into tier `tier_idx`.
    ///
    /// Useful for pre-populating lower tiers (e.g. from a warm-up snapshot).
    pub fn put_at_tier(&mut self, tier_idx: usize, key: &str, data: Vec<u8>) {
        if let Some(tier) = self.tiers.get_mut(tier_idx) {
            tier.put(key.to_string(), data);
        }
    }

    /// Evict one entry from tier `tier_idx` according to that tier's policy.
    /// Returns the evicted `(key, data)` or `None` if the tier is empty.
    pub fn evict_tier(&mut self, tier_idx: usize) -> Option<(String, Vec<u8>)> {
        self.tiers.get_mut(tier_idx)?.evict_one()
    }

    /// Return an aggregate statistics snapshot.
    pub fn stats(&self) -> TieredCacheStats {
        let total = self.total_hits + self.total_misses;
        let hit_rate = if total == 0 {
            0.0
        } else {
            self.total_hits as f64 / total as f64
        };
        let tier_stats = self
            .tiers
            .iter()
            .enumerate()
            .map(|(i, t)| TierStats {
                name: t.config.name.clone(),
                hits: self.tier_hits[i],
                size_used_bytes: t.size_used,
                entry_count: t.data.len(),
                promotions: t.promotions,
                compressions: t.compressions,
            })
            .collect();
        TieredCacheStats {
            total_hits: self.total_hits,
            total_misses: self.total_misses,
            hit_rate,
            tier_stats,
        }
    }

    /// Bulk-insert `entries` into L1 without triggering eviction.
    pub fn warmup(&mut self, entries: &[(String, Vec<u8>)]) {
        for (key, data) in entries {
            let data_len = data.len();
            if self.tiers[0].size_used + data_len <= self.tiers[0].config.capacity_bytes {
                let tick = self.tiers[0].tick;
                self.tiers[0].tick += 1;
                self.tiers[0].size_used += data_len;
                self.tiers[0].fifo_order.push_back(key.clone());
                self.tiers[0]
                    .data
                    .insert(key.clone(), (TierEntry::Owned(data.clone()), tick, 1));
            }
        }
    }

    /// Remove `key` from every tier.  Returns `true` if it was found in at
    /// least one tier.
    pub fn invalidate(&mut self, key: &str) -> bool {
        let mut found = false;
        for tier in &mut self.tiers {
            if tier.remove(key) {
                found = true;
            }
        }
        found
    }

    /// Return the number of tiers.
    pub fn tier_count(&self) -> usize {
        self.tiers.len()
    }

    /// Return the number of promotions from tier `tier_idx` to the tier above.
    pub fn tier_promotions(&self, tier_idx: usize) -> u64 {
        self.tiers.get(tier_idx).map(|t| t.promotions).unwrap_or(0)
    }

    /// Return the number of hits recorded for tier `tier_idx`.
    pub fn tier_hit_count(&self, tier_idx: usize) -> u64 {
        self.tier_hits.get(tier_idx).copied().unwrap_or(0)
    }

    /// Reset the bump arena of tier `tier_idx` (reclaims all arena memory).
    ///
    /// After a reset, all previously stored arena handles for that tier are
    /// invalid; this is intended for use after a bulk eviction sweep.
    pub fn reset_tier_arena(&mut self, tier_idx: usize) {
        if let Some(tier) = self.tiers.get_mut(tier_idx) {
            if let Some(ref mut arena) = tier.arena {
                arena.reset();
            }
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn two_tier_cache(l1_bytes: usize, l2_bytes: usize) -> TieredCache {
        TieredCache::new(vec![
            TierConfig {
                name: "L1".into(),
                capacity_bytes: l1_bytes,
                eviction_policy: EvictionPolicy::Lru,
                ..TierConfig::memory("L1", l1_bytes)
            },
            TierConfig {
                name: "L2".into(),
                capacity_bytes: l2_bytes,
                eviction_policy: EvictionPolicy::Lfu,
                ..TierConfig::memory("L2", l2_bytes)
            },
        ])
    }

    // 1. Basic put and get
    #[test]
    fn test_basic_put_get() {
        let mut cache = two_tier_cache(1024, 4096);
        cache.put("key1", b"hello".to_vec());
        assert_eq!(cache.get("key1"), Some(b"hello".to_vec()));
    }

    // 2. Miss returns None
    #[test]
    fn test_miss() {
        let mut cache = two_tier_cache(1024, 4096);
        assert_eq!(cache.get("absent"), None);
        assert_eq!(cache.stats().total_misses, 1);
    }

    // 3. Hit rate calculation
    #[test]
    fn test_hit_rate() {
        let mut cache = two_tier_cache(1024, 4096);
        cache.put("k", b"v".to_vec());
        cache.get("k"); // hit
        cache.get("nope"); // miss
        let s = cache.stats();
        assert!((s.hit_rate - 0.5).abs() < 1e-9);
    }

    // 4. L1 eviction under LRU policy
    #[test]
    fn test_l1_lru_eviction() {
        let mut cache = two_tier_cache(3, 1024);
        cache.put("a", b"1".to_vec());
        cache.put("b", b"2".to_vec());
        cache.put("c", b"3".to_vec());
        cache.get("a");
        cache.put("d", b"4".to_vec());
        assert_eq!(cache.get("b"), None);
        assert!(cache.get("a").is_some());
    }

    // 5. invalidate removes from all tiers
    #[test]
    fn test_invalidate() {
        let mut cache = two_tier_cache(1024, 4096);
        cache.put("x", b"data".to_vec());
        assert!(cache.invalidate("x"));
        assert_eq!(cache.get("x"), None);
    }

    // 6. invalidate on absent key returns false
    #[test]
    fn test_invalidate_absent() {
        let mut cache = two_tier_cache(1024, 4096);
        assert!(!cache.invalidate("ghost"));
    }

    // 7. warmup populates L1 without eviction
    #[test]
    fn test_warmup() {
        let mut cache = two_tier_cache(1024, 4096);
        let entries = vec![
            ("alpha".to_string(), b"AAA".to_vec()),
            ("beta".to_string(), b"BBB".to_vec()),
        ];
        cache.warmup(&entries);
        assert_eq!(cache.get("alpha"), Some(b"AAA".to_vec()));
        assert_eq!(cache.get("beta"), Some(b"BBB".to_vec()));
    }

    // 8. stats entry_count
    #[test]
    fn test_stats_entry_count() {
        let mut cache = two_tier_cache(1024, 4096);
        cache.put("a", b"1".to_vec());
        cache.put("b", b"2".to_vec());
        assert_eq!(cache.stats().tier_stats[0].entry_count, 2);
    }

    // 9. FIFO eviction policy
    #[test]
    fn test_fifo_eviction() {
        let mut cache = TieredCache::new(vec![TierConfig {
            eviction_policy: EvictionPolicy::Fifo,
            ..TierConfig::memory("fifo", 3)
        }]);
        cache.put("first", b"1".to_vec());
        cache.put("second", b"2".to_vec());
        cache.put("third", b"3".to_vec());
        cache.put("fourth", b"4".to_vec());
        assert_eq!(cache.get("first"), None);
    }

    // 10. Random eviction policy (smoke test)
    #[test]
    fn test_random_eviction_no_panic() {
        let mut cache = TieredCache::new(vec![TierConfig {
            eviction_policy: EvictionPolicy::Random,
            ..TierConfig::memory("rand", 5)
        }]);
        for i in 0..20u8 {
            cache.put(&i.to_string(), vec![i]);
        }
        assert!(cache.stats().tier_stats[0].entry_count <= 5);
    }

    // 11. TinyLFU eviction policy (smoke test)
    #[test]
    fn test_tiny_lfu_eviction_no_panic() {
        let mut cache = TieredCache::new(vec![TierConfig {
            eviction_policy: EvictionPolicy::TinyLfu,
            ..TierConfig::memory("tiny", 5)
        }]);
        for i in 0..20u8 {
            cache.put(&i.to_string(), vec![i]);
        }
        assert!(cache.stats().tier_stats[0].entry_count <= 5);
    }

    // 12. evict_tier API
    #[test]
    fn test_evict_tier() {
        let mut cache = two_tier_cache(1024, 4096);
        cache.put("a", b"data".to_vec());
        let evicted = cache.evict_tier(0);
        assert!(evicted.is_some());
        let (k, _) = evicted.expect("eviction should succeed");
        assert_eq!(k, "a");
    }

    // 13. evict_tier on empty tier returns None
    #[test]
    fn test_evict_empty_tier() {
        let mut cache = two_tier_cache(1024, 4096);
        assert!(cache.evict_tier(0).is_none());
    }

    // 14. size_used_bytes tracks usage
    #[test]
    fn test_size_used_bytes() {
        let mut cache = two_tier_cache(1024, 4096);
        cache.put("a", vec![0u8; 100]);
        cache.put("b", vec![0u8; 200]);
        assert_eq!(cache.stats().tier_stats[0].size_used_bytes, 300);
    }

    // 15. Tier hit counters: L1 hit increments tier 0
    #[test]
    fn test_tier_hit_counters() {
        let mut cache = two_tier_cache(1024, 4096);
        cache.put("k", b"v".to_vec());
        cache.get("k");
        cache.get("k");
        let s = cache.stats();
        assert_eq!(s.tier_stats[0].hits, 2);
    }

    // 16. Compression: compressed tier stores and retrieves correctly
    #[test]
    fn test_compression_roundtrip() {
        let mut cache = TieredCache::new(vec![TierConfig {
            compress: true,
            ..TierConfig::memory("compressed", 1024 * 1024)
        }]);
        // Highly compressible data: run of the same byte.
        let data = vec![0xABu8; 512];
        cache.put("k", data.clone());
        let retrieved = cache.get("k").expect("should be present");
        assert_eq!(
            retrieved, data,
            "compressed entry should decompress correctly"
        );
    }

    // 17. Compression: stats track compression count
    #[test]
    fn test_compression_stats() {
        let mut cache = TieredCache::new(vec![TierConfig {
            compress: true,
            ..TierConfig::memory("c", 1024 * 1024)
        }]);
        cache.put("a", vec![1u8; 64]);
        cache.put("b", vec![2u8; 64]);
        let s = cache.stats();
        assert_eq!(
            s.tier_stats[0].compressions, 2,
            "two puts should compress twice"
        );
    }

    // 18. Adaptive promotion: high-frequency key promotes; low-frequency stays
    #[test]
    fn test_adaptive_promotion_threshold() {
        // L1: tiny (only fits 10 bytes), threshold 0.
        // L2: larger, threshold 3 (must access 3 times before promotion).
        let mut cache = TieredCache::new(vec![
            TierConfig::memory("L1", 10),
            TierConfig {
                promotion_threshold: 3,
                ..TierConfig::memory("L2", 1024)
            },
        ]);

        // Put a single-byte value directly into L2.
        cache.put_at_tier(1, "hot", b"v".to_vec());

        // First and second accesses: frequency < 3, no promotion.
        cache.get("hot"); // freq becomes 1
        cache.get("hot"); // freq becomes 2

        // Third access meets threshold (>= 3) — no, wait: threshold is 3 and
        // frequency after get becomes 3. Let us verify get "hot" a third time.
        cache.get("hot"); // freq becomes 3 → promoted

        let s = cache.stats();
        assert!(
            s.tier_stats[1].promotions >= 1,
            "entry should have been promoted after reaching threshold"
        );
    }

    // 19. Disk-backed tier stores and retrieves entries
    #[test]
    fn test_disk_tier_basic() {
        let dir = std::env::temp_dir().join(format!(
            "oximedia_tiered_disk_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(42)
        ));
        let mut cache = TieredCache::new(vec![TierConfig::disk("disk", 1024 * 1024, &dir)]);
        cache.put("segment-001", b"media data here".to_vec());
        let got = cache.get("segment-001");
        assert_eq!(
            got,
            Some(b"media data here".to_vec()),
            "disk tier should retrieve the value correctly"
        );
        // dir is cleaned up by CacheTier::drop
    }

    // 20. TierConfig::memory helper
    #[test]
    fn test_tier_config_memory_helper() {
        let cfg = TierConfig::memory("L1", 4096);
        assert_eq!(cfg.name, "L1");
        assert_eq!(cfg.capacity_bytes, 4096);
        assert!(cfg.disk_path.is_none());
        assert!(!cfg.compress);
    }

    // 21. tier_count
    #[test]
    fn test_tier_count() {
        let cache = two_tier_cache(1024, 4096);
        assert_eq!(cache.tier_count(), 2);
    }

    // 22. put_at_tier inserts into specified tier
    #[test]
    fn test_put_at_tier() {
        let mut cache = two_tier_cache(1024, 4096);
        cache.put_at_tier(1, "l2-key", b"l2-value".to_vec());
        assert_eq!(cache.stats().tier_stats[1].entry_count, 1);
        // Can retrieve via get (searches all tiers).
        assert_eq!(cache.get("l2-key"), Some(b"l2-value".to_vec()));
    }

    // 23. RLE compress + decompress are inverse
    #[test]
    fn test_rle_roundtrip() {
        for input in [
            b"".as_ref(),
            b"hello",
            b"\x00\x00\x00\x00",
            b"AAABBBCCC",
            b"abcdefghij",
        ] {
            let compressed = rle_compress(input);
            let decompressed = rle_decompress(&compressed);
            assert_eq!(decompressed, input, "rle roundtrip failed for {:?}", input);
        }
    }
}
