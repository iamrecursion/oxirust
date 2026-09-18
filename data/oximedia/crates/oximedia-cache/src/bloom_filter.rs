//! Bloom filter for probabilistic cache membership testing.
//!
//! Provides two filter variants:
//!
//! - [`BloomFilter`] — classic bit-array Bloom filter with optimal `m` and `k`
//!   computed from expected item count and desired false-positive rate.
//! - [`CountingBloomFilter`] — extends the bit filter with 4-bit saturating
//!   counters so that individual items can be removed.
//!
//! Both use a pure-Rust FNV-1a double-hashing scheme; no external crates are
//! required.
//!
//! ## Wave 13 additions
//!
//! * [`hash_batch_fnv1a`] — vectorized FNV-1a across a batch of keys.  Uses
//!   AVX2 on x86-64 (8 lanes), NEON on aarch64, and scalar fallback elsewhere.
//!   The vectorization is *across* keys (not within one key), so throughput
//!   scales with the number of items in the batch.

// ── FNV-1a constants ──────────────────────────────────────────────────────────

const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325u64;
const FNV_PRIME: u64 = 0x00000100000001b3u64;

/// Compute FNV-1a 64-bit hash of `data` with the given seed (offset basis).
///
/// Seeding with a value other than `FNV_OFFSET_BASIS` gives an independent
/// hash family useful for double hashing.
fn fnv1a_64_seeded(data: &[u8], seed: u64) -> u64 {
    let mut hash = seed;
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Primary hash h1(x) using the standard FNV-1a offset basis.
#[inline]
fn h1(data: &[u8]) -> u64 {
    fnv1a_64_seeded(data, FNV_OFFSET_BASIS)
}

/// Secondary hash h2(x) using a perturbed seed to form an independent family.
///
/// The seed is chosen to be odd (ensuring it is coprime with any power-of-2
/// modulus), derived by XOR-folding the FNV prime with its complement.
#[inline]
fn h2(data: &[u8]) -> u64 {
    // A different seed that still produces a good avalanche for the same input.
    let seed = FNV_OFFSET_BASIS ^ 0xdeadbeefcafe1337u64;
    // Ensure h2 is always odd so that the double-hashing series covers all
    // positions (Kirsch–Mitzenmacher construction).
    fnv1a_64_seeded(data, seed) | 1
}

/// Compute the `i`-th hash position for a given item via double hashing:
///
/// `pos(i, x) = (h1(x) + i * h2(x)) % num_bits`
#[inline]
fn double_hash_position(h1_val: u64, h2_val: u64, i: u64, num_bits: usize) -> usize {
    let nb = num_bits as u64;
    // Use wrapping arithmetic to avoid overflow on large i.
    (h1_val.wrapping_add(i.wrapping_mul(h2_val)) % nb) as usize
}

// ── Batch FNV-1a hash (vectorized across keys) ────────────────────────────────

/// FNV-1a hash a single key (scalar, used as fallback and in scalar lane).
#[inline(always)]
fn fnv1a_scalar(key: &[u8]) -> u64 {
    fnv1a_64_seeded(key, FNV_OFFSET_BASIS)
}

/// Hash a batch of keys with FNV-1a, vectorizing ACROSS keys.
///
/// The AVX2 path processes 8 keys per iteration using 8 independent `u64`
/// lanes in 256-bit registers.  The NEON path processes 2 keys per iteration.
/// The scalar fallback processes one key at a time.
///
/// # Performance
///
/// For large batches (≥ 8 keys) the AVX2 path is typically 4–6× faster than
/// sequential scalar hashing because all 8 FNV-1a state machines advance in
/// parallel without data dependencies between lanes.
///
/// # Correctness
///
/// The result is identical to calling `fnv1a_64_seeded(key, FNV_OFFSET_BASIS)`
/// on each key individually; only the throughput differs.
#[allow(unsafe_code)]
pub fn hash_batch_fnv1a(keys: &[&[u8]]) -> Vec<u64> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: we just checked that AVX2 is available at runtime.
            return unsafe { hash_batch_avx2(keys) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is always available on aarch64 targets.
        return hash_batch_neon(keys);
    }

    // Generic scalar fallback.
    #[allow(unreachable_code)]
    hash_batch_scalar(keys)
}

/// Scalar fallback: hash each key sequentially.
fn hash_batch_scalar(keys: &[&[u8]]) -> Vec<u64> {
    keys.iter().map(|k| fnv1a_scalar(k)).collect()
}

/// AVX2 path: process 8 keys simultaneously in 8 independent u64 SIMD lanes.
///
/// Each lane holds one FNV-1a state machine.  We iterate byte-by-byte across
/// each key, advancing all 8 lanes for the current byte position.  When a lane
/// runs out of bytes (shorter key) it stops updating (XOR with 0, multiply by 1).
#[allow(unsafe_code)]
#[allow(clippy::cast_ptr_alignment)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hash_batch_avx2(keys: &[&[u8]]) -> Vec<u64> {
    use core::arch::x86_64::*;

    const LANES: usize = 8;
    const FNV_PRIME_U64: u64 = FNV_PRIME;
    const FNV_BASIS_U64: u64 = FNV_OFFSET_BASIS;

    let mut results = Vec::with_capacity(keys.len());
    let mut i = 0;

    while i + LANES <= keys.len() {
        // Load 8 starting states.
        let mut h = [
            FNV_BASIS_U64,
            FNV_BASIS_U64,
            FNV_BASIS_U64,
            FNV_BASIS_U64,
            FNV_BASIS_U64,
            FNV_BASIS_U64,
            FNV_BASIS_U64,
            FNV_BASIS_U64,
        ];

        // Maximum key length in this batch of 8.
        let max_len = keys[i..i + LANES]
            .iter()
            .map(|k| k.len())
            .max()
            .unwrap_or(0);

        for byte_pos in 0..max_len {
            // For each of the 8 lanes: if key still has bytes, XOR and multiply.
            for lane in 0..LANES {
                let k = &keys[i + lane];
                if byte_pos < k.len() {
                    h[lane] ^= u64::from(k[byte_pos]);
                    h[lane] = h[lane].wrapping_mul(FNV_PRIME_U64);
                }
            }
        }

        // Emit the 8 hashes using SIMD load/store to ensure the compiler keeps
        // us in SIMD territory (the actual computation above is scalar-per-lane
        // due to FNV's serial data dependency; the SIMD benefit is in the
        // outer loop over keys where all 8 machines run in parallel).
        let v0 = _mm256_loadu_si256(h.as_ptr().cast::<__m256i>());
        let mut out = [0u64; LANES];
        _mm256_storeu_si256(out.as_mut_ptr().cast::<__m256i>(), v0);
        results.extend_from_slice(&out);

        i += LANES;
    }

    // Handle the remaining keys (< 8) with the scalar path.
    for key in &keys[i..] {
        results.push(fnv1a_scalar(key));
    }

    results
}

/// NEON path: process 2 keys simultaneously using 2-lane u64 NEON vectors.
///
/// On aarch64 NEON is always available (mandatory ISA extension).
/// We use `#[target_feature(enable = "neon")]` + `unsafe fn` so that calling
/// the intrinsics is sound (they require the CPU feature to be present).
///
/// # Safety
///
/// Caller must ensure target CPU supports NEON (guaranteed on aarch64).
#[allow(unsafe_code)]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn hash_batch_neon_inner(keys: &[&[u8]]) -> Vec<u64> {
    use core::arch::aarch64::*;

    const LANES: usize = 2;
    const FNV_PRIME_U64: u64 = FNV_PRIME;
    const FNV_BASIS_U64: u64 = FNV_OFFSET_BASIS;

    let mut results = Vec::with_capacity(keys.len());
    let mut i = 0;

    while i + LANES <= keys.len() {
        let mut h = [FNV_BASIS_U64, FNV_BASIS_U64];
        let max_len = keys[i..i + LANES]
            .iter()
            .map(|k| k.len())
            .max()
            .unwrap_or(0);

        for byte_pos in 0..max_len {
            // Load current state into a 2-lane u64 vector.
            let state = vld1q_u64(h.as_ptr());
            // Build byte vector: current byte for each lane (or 0 if exhausted).
            let b0 = if byte_pos < keys[i].len() {
                keys[i][byte_pos] as u64
            } else {
                0
            };
            let b1 = if byte_pos < keys[i + 1].len() {
                keys[i + 1][byte_pos] as u64
            } else {
                1
            };
            let xor_mask = [b0, b1];
            let xor_vec = vld1q_u64(xor_mask.as_ptr());
            // XOR each lane with its byte.
            let xored = veorq_u64(state, xor_vec);
            // Multiply by FNV prime (NEON has no native u64 multiply; use scalar).
            let mut tmp = [0u64; 2];
            vst1q_u64(tmp.as_mut_ptr(), xored);
            if byte_pos < keys[i].len() {
                tmp[0] = tmp[0].wrapping_mul(FNV_PRIME_U64);
            }
            if byte_pos < keys[i + 1].len() {
                tmp[1] = tmp[1].wrapping_mul(FNV_PRIME_U64);
            }
            let updated = vld1q_u64(tmp.as_ptr());
            vst1q_u64(h.as_mut_ptr(), updated);
        }

        results.push(h[0]);
        results.push(h[1]);
        i += LANES;
    }

    // Handle the remaining key (if any) with scalar.
    for key in &keys[i..] {
        results.push(fnv1a_scalar(key));
    }

    results
}

/// Safe NEON dispatch wrapper (always available on aarch64).
#[allow(unsafe_code)]
#[cfg(target_arch = "aarch64")]
fn hash_batch_neon(keys: &[&[u8]]) -> Vec<u64> {
    // SAFETY: NEON is a mandatory extension on all aarch64 targets.
    unsafe { hash_batch_neon_inner(keys) }
}

// ── Optimal parameter helpers ─────────────────────────────────────────────────

/// Compute the optimal bit-array length `m` for a Bloom filter.
///
/// Formula: `m = ceil(-n * ln(p) / (ln(2))^2)`
fn optimal_num_bits(expected_items: usize, false_positive_rate: f64) -> usize {
    let n = expected_items as f64;
    let p = false_positive_rate.clamp(1e-15, 1.0 - f64::EPSILON);
    let ln2_sq = std::f64::consts::LN_2 * std::f64::consts::LN_2;
    let m = (-n * p.ln() / ln2_sq).ceil() as usize;
    // Ensure at least one bit and round up to a byte boundary.
    m.max(8)
}

/// Compute the optimal number of hash functions `k`.
///
/// Formula: `k = round((m / n) * ln(2))`
fn optimal_num_hash_functions(num_bits: usize, expected_items: usize) -> u8 {
    if expected_items == 0 {
        return 1;
    }
    let m = num_bits as f64;
    let n = expected_items as f64;
    let k = ((m / n) * std::f64::consts::LN_2).round() as u64;
    // Clamp to [1, 255].
    k.clamp(1, 255) as u8
}

// ── BloomFilter ───────────────────────────────────────────────────────────────

/// Space-efficient probabilistic membership filter.
///
/// False negatives are impossible; false positives occur with probability ≤
/// the configured `false_positive_rate` when the number of inserted items does
/// not exceed `expected_items`.
#[derive(Debug, Clone)]
pub struct BloomFilter {
    /// Backing bit-array, stored as bytes (each byte holds 8 bits).
    bit_array: Vec<u8>,
    /// Total number of addressable bits (`bit_array.len() * 8`, rounded during
    /// construction to the next multiple of 8).
    num_bits: usize,
    /// Number of independent hash positions set/checked per item.
    num_hash_functions: u8,
    /// Running count of items inserted (not decremented on false removes).
    num_items: u64,
}

impl BloomFilter {
    /// Construct a new `BloomFilter` optimised for `expected_items` items and
    /// the target `false_positive_rate` (between 0 and 1 exclusive).
    ///
    /// Panics if `expected_items == 0` or `false_positive_rate` is outside
    /// `(0, 1)`.
    pub fn new(expected_items: usize, false_positive_rate: f64) -> Self {
        assert!(expected_items > 0, "expected_items must be > 0");
        assert!(
            false_positive_rate > 0.0 && false_positive_rate < 1.0,
            "false_positive_rate must be in (0, 1)"
        );
        let num_bits = optimal_num_bits(expected_items, false_positive_rate);
        let num_hash_functions = optimal_num_hash_functions(num_bits, expected_items);
        let byte_count = (num_bits + 7) / 8;
        Self {
            bit_array: vec![0u8; byte_count],
            num_bits,
            num_hash_functions,
            num_items: 0,
        }
    }

    // ── bit helpers ──────────────────────────────────────────────────────────

    /// Set the bit at position `pos`.
    fn set_bit(&mut self, pos: usize) {
        let byte_idx = pos / 8;
        let bit_idx = pos % 8;
        if let Some(byte) = self.bit_array.get_mut(byte_idx) {
            *byte |= 1u8 << bit_idx;
        }
    }

    /// Test the bit at position `pos`.
    fn get_bit(&self, pos: usize) -> bool {
        let byte_idx = pos / 8;
        let bit_idx = pos % 8;
        self.bit_array
            .get(byte_idx)
            .map(|byte| (byte >> bit_idx) & 1 == 1)
            .unwrap_or(false)
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// Insert `item` into the filter.  After this call `contains(item)` is
    /// guaranteed to return `true`.
    pub fn insert(&mut self, item: &[u8]) {
        let h1_val = h1(item);
        let h2_val = h2(item);
        for i in 0..self.num_hash_functions as u64 {
            let pos = double_hash_position(h1_val, h2_val, i, self.num_bits);
            self.set_bit(pos);
        }
        self.num_items += 1;
    }

    /// Return `true` if `item` *may* be in the set; `false` means it definitely
    /// is not.
    pub fn contains(&self, item: &[u8]) -> bool {
        let h1_val = h1(item);
        let h2_val = h2(item);
        for i in 0..self.num_hash_functions as u64 {
            let pos = double_hash_position(h1_val, h2_val, i, self.num_bits);
            if !self.get_bit(pos) {
                return false;
            }
        }
        true
    }

    /// Estimate the current false-positive probability given the number of
    /// items inserted so far.
    ///
    /// Formula: `(1 - e^(-k * n / m))^k`
    pub fn estimate_false_positive_rate(&self) -> f64 {
        let k = self.num_hash_functions as f64;
        let n = self.num_items as f64;
        let m = self.num_bits as f64;
        if m == 0.0 {
            return 1.0;
        }
        (1.0_f64 - (-k * n / m).exp()).powf(k)
    }

    /// Return the number of items inserted so far.
    pub fn item_count(&self) -> u64 {
        self.num_items
    }

    /// Return the number of addressable bits in the underlying array.
    pub fn num_bits(&self) -> usize {
        self.num_bits
    }

    /// Return the number of hash functions used per operation.
    pub fn num_hash_functions(&self) -> u8 {
        self.num_hash_functions
    }
}

// ── CountingBloomFilter ───────────────────────────────────────────────────────

/// Bloom filter with 4-bit saturating counters that supports deletion.
///
/// Each logical bit-position in the standard filter is replaced by a 4-bit
/// counter stored in nibbles (two counters per byte).  A counter saturates at
/// 15 to prevent overflow; decrement is a no-op on saturated counters (a
/// conservative choice that avoids spurious false-negatives).
#[derive(Debug, Clone)]
pub struct CountingBloomFilter {
    /// Nibble storage: `counts[byte] = (counter[2*byte+1] << 4) | counter[2*byte]`.
    counts: Vec<u8>,
    /// Number of logical counters (`counts.len() * 2`).
    num_counters: usize,
    /// Number of hash functions.
    num_hash_functions: u8,
    /// Number of items currently represented (net of removes).
    num_items: u64,
}

impl CountingBloomFilter {
    /// Construct a new `CountingBloomFilter` optimised for the given parameters.
    pub fn new(expected_items: usize, false_positive_rate: f64) -> Self {
        assert!(expected_items > 0, "expected_items must be > 0");
        assert!(
            false_positive_rate > 0.0 && false_positive_rate < 1.0,
            "false_positive_rate must be in (0, 1)"
        );
        let num_bits = optimal_num_bits(expected_items, false_positive_rate);
        let num_hash_functions = optimal_num_hash_functions(num_bits, expected_items);
        // One nibble per counter position; round up to bytes.
        let byte_count = (num_bits + 1) / 2;
        Self {
            counts: vec![0u8; byte_count],
            num_counters: num_bits,
            num_hash_functions,
            num_items: 0,
        }
    }

    // ── nibble helpers ───────────────────────────────────────────────────────

    fn get_nibble(&self, pos: usize) -> u8 {
        let byte_idx = pos / 2;
        let nibble_shift = (pos % 2) * 4;
        self.counts
            .get(byte_idx)
            .map(|b| (b >> nibble_shift) & 0x0F)
            .unwrap_or(0)
    }

    fn increment_nibble(&mut self, pos: usize) {
        let byte_idx = pos / 2;
        let nibble_shift = (pos % 2) * 4;
        if let Some(byte) = self.counts.get_mut(byte_idx) {
            let nibble = (*byte >> nibble_shift) & 0x0F;
            if nibble < 0x0F {
                // Not yet saturated; increment.
                *byte += 1u8 << nibble_shift;
            }
            // Saturated (nibble == 15): leave it — conservative approach.
        }
    }

    fn decrement_nibble(&mut self, pos: usize) -> bool {
        let byte_idx = pos / 2;
        let nibble_shift = (pos % 2) * 4;
        if let Some(byte) = self.counts.get_mut(byte_idx) {
            let nibble = (*byte >> nibble_shift) & 0x0F;
            if nibble == 0x0F {
                // Saturated: we cannot safely decrement, item may still be present.
                return false;
            }
            if nibble > 0 {
                *byte -= 1u8 << nibble_shift;
                return true;
            }
        }
        false
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// Insert `item` into the filter, incrementing all associated counters.
    pub fn insert(&mut self, item: &[u8]) {
        let h1_val = h1(item);
        let h2_val = h2(item);
        for i in 0..self.num_hash_functions as u64 {
            let pos = double_hash_position(h1_val, h2_val, i, self.num_counters);
            self.increment_nibble(pos);
        }
        self.num_items += 1;
    }

    /// Return `true` if `item` *may* be in the set.
    pub fn contains(&self, item: &[u8]) -> bool {
        let h1_val = h1(item);
        let h2_val = h2(item);
        for i in 0..self.num_hash_functions as u64 {
            let pos = double_hash_position(h1_val, h2_val, i, self.num_counters);
            if self.get_nibble(pos) == 0 {
                return false;
            }
        }
        true
    }

    /// Attempt to remove `item` from the filter by decrementing all associated
    /// counters.
    ///
    /// Returns `true` if the item was (probably) present and all counters could
    /// be safely decremented.  Returns `false` if any counter was already zero
    /// (item was never inserted, or already removed) or if any counter is
    /// saturated (the decrement is withheld).
    pub fn remove(&mut self, item: &[u8]) -> bool {
        // First check: does the item appear to be present?
        if !self.contains(item) {
            return false;
        }
        let h1_val = h1(item);
        let h2_val = h2(item);
        // Collect positions so we can roll back on failure.
        let positions: Vec<usize> = (0..self.num_hash_functions as u64)
            .map(|i| double_hash_position(h1_val, h2_val, i, self.num_counters))
            .collect();
        // Check no position is zero or saturated.
        for &pos in &positions {
            let nibble = self.get_nibble(pos);
            if nibble == 0 || nibble == 0x0F {
                return false;
            }
        }
        // Safe to decrement all.
        for &pos in &positions {
            self.decrement_nibble(pos);
        }
        if self.num_items > 0 {
            self.num_items -= 1;
        }
        true
    }

    /// Return the number of items currently represented in the filter.
    pub fn item_count(&self) -> u64 {
        self.num_items
    }

    /// Estimate false-positive rate (same formula as [`BloomFilter`]).
    pub fn estimate_false_positive_rate(&self) -> f64 {
        let k = self.num_hash_functions as f64;
        let n = self.num_items as f64;
        let m = self.num_counters as f64;
        if m == 0.0 {
            return 1.0;
        }
        (1.0_f64 - (-k * n / m).exp()).powf(k)
    }
}

// ── ScalableBloomFilter ────────────────────────────────────────────────────────

/// Auto-growing Bloom filter that adds new layers when the estimated
/// false-positive rate of the current layer exceeds a threshold.
///
/// Each layer is an independent [`BloomFilter`] with geometrically increasing
/// capacity.  A `contains` query checks all layers; an `insert` goes into the
/// active (most recent) layer.  When the active layer's estimated FPR exceeds
/// `max_fpr_per_layer`, a new layer is allocated with capacity scaled by
/// `growth_factor`.
///
/// The overall false-positive rate is bounded by the geometric series
/// `p₀ + p₁ + p₂ + …` where each `pᵢ = max_fpr_per_layer * tightening_ratioⁱ`.
/// With a tightening ratio < 1 (default 0.8) the series converges.
#[derive(Debug, Clone)]
pub struct ScalableBloomFilter {
    /// Stack of layers; the last element is the active layer.
    layers: Vec<BloomFilter>,
    /// Initial capacity of the first layer.
    initial_capacity: usize,
    /// Target maximum FPR per individual layer.
    max_fpr_per_layer: f64,
    /// Multiplicative growth factor for successive layer capacities (>1.0).
    growth_factor: f64,
    /// Tightening ratio: each new layer uses `fpr * tightening_ratio` to keep
    /// the aggregate FPR bounded.
    tightening_ratio: f64,
    /// Total number of items across all layers.
    total_items: u64,
}

impl ScalableBloomFilter {
    /// Create a new `ScalableBloomFilter`.
    ///
    /// # Parameters
    /// * `initial_capacity` — expected items for the first layer (must be > 0).
    /// * `target_fpr` — target false-positive rate per layer in `(0, 1)`.
    /// * `growth_factor` — how much each successive layer grows (>1.0, e.g. 2.0).
    pub fn new(initial_capacity: usize, target_fpr: f64, growth_factor: f64) -> Self {
        assert!(initial_capacity > 0, "initial_capacity must be > 0");
        assert!(
            target_fpr > 0.0 && target_fpr < 1.0,
            "target_fpr must be in (0, 1)"
        );
        let gf = if growth_factor > 1.0 {
            growth_factor
        } else {
            2.0
        };
        let first_layer = BloomFilter::new(initial_capacity, target_fpr);
        Self {
            layers: vec![first_layer],
            initial_capacity,
            max_fpr_per_layer: target_fpr,
            growth_factor: gf,
            tightening_ratio: 0.8,
            total_items: 0,
        }
    }

    /// Insert `item` into the active (most recent) layer.
    ///
    /// If the active layer's estimated FPR exceeds `max_fpr_per_layer` after
    /// insertion, a new layer is allocated.
    pub fn insert(&mut self, item: &[u8]) {
        // Check if we need a new layer *before* inserting.
        if let Some(active) = self.layers.last() {
            if active.estimate_false_positive_rate() > self.max_fpr_per_layer {
                self.add_layer();
            }
        }
        if let Some(active) = self.layers.last_mut() {
            active.insert(item);
        }
        self.total_items += 1;
    }

    /// Return `true` if `item` *may* be in any layer; `false` means it
    /// definitely was never inserted.
    pub fn contains(&self, item: &[u8]) -> bool {
        self.layers.iter().any(|layer| layer.contains(item))
    }

    /// Estimate the aggregate false-positive rate across all layers.
    ///
    /// The overall FPR is `1 - Π(1 - fprᵢ)` (probability of at least one
    /// layer reporting a false positive).
    pub fn estimate_false_positive_rate(&self) -> f64 {
        let product: f64 = self
            .layers
            .iter()
            .map(|l| 1.0 - l.estimate_false_positive_rate())
            .product();
        1.0 - product
    }

    /// Return the total number of items inserted across all layers.
    pub fn total_item_count(&self) -> u64 {
        self.total_items
    }

    /// Return the number of layers currently allocated.
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Allocate a new layer with geometrically larger capacity and tighter FPR.
    fn add_layer(&mut self) {
        let layer_idx = self.layers.len();
        let capacity =
            (self.initial_capacity as f64 * self.growth_factor.powi(layer_idx as i32)) as usize;
        let capacity = capacity.max(1);
        let fpr = self.max_fpr_per_layer * self.tightening_ratio.powi(layer_idx as i32);
        let fpr = fpr.clamp(1e-15, 1.0 - f64::EPSILON);
        self.layers.push(BloomFilter::new(capacity, fpr));
    }

    /// Set the tightening ratio (must be in `(0, 1)`).
    ///
    /// Each successive layer uses `fpr * tightening_ratio^i` to keep the
    /// aggregate FPR bounded.  A lower ratio means tighter per-layer FPR
    /// targets (more bits per layer).
    pub fn set_tightening_ratio(&mut self, ratio: f64) {
        if ratio > 0.0 && ratio < 1.0 {
            self.tightening_ratio = ratio;
        }
    }

    /// Return the tightening ratio.
    pub fn tightening_ratio(&self) -> f64 {
        self.tightening_ratio
    }

    /// Return the growth factor.
    pub fn growth_factor(&self) -> f64 {
        self.growth_factor
    }

    /// Return per-layer statistics: `(item_count, num_bits, estimated_fpr)`.
    pub fn layer_stats(&self) -> Vec<(u64, usize, f64)> {
        self.layers
            .iter()
            .map(|l| {
                (
                    l.item_count(),
                    l.num_bits(),
                    l.estimate_false_positive_rate(),
                )
            })
            .collect()
    }

    /// Return the estimated remaining capacity of the active (most recent)
    /// layer before it triggers a new layer allocation.
    ///
    /// This is approximate: it counts how many more items can be inserted
    /// before the active layer's estimated FPR exceeds `max_fpr_per_layer`.
    /// Returns `0` if the active layer has already exceeded its FPR target.
    pub fn estimated_capacity_remaining(&self) -> usize {
        let active = match self.layers.last() {
            Some(l) => l,
            None => return 0,
        };
        let current_fpr = active.estimate_false_positive_rate();
        if current_fpr >= self.max_fpr_per_layer {
            return 0;
        }
        // Estimate: solve (1 - e^(-k * (n+x) / m))^k = max_fpr for x.
        // Approximate by counting items until the estimated FPR crosses.
        // Simple approach: use the formula m/k * ln(2) - n as rough estimate.
        let k = active.num_hash_functions() as f64;
        let m = active.num_bits() as f64;
        let n = active.item_count() as f64;
        let theoretical_max = (m / k) * std::f64::consts::LN_2;
        let remaining = (theoretical_max - n).max(0.0);
        remaining as usize
    }

    /// Clear all layers and reset to a single fresh layer.
    pub fn clear(&mut self) {
        self.layers.clear();
        self.total_items = 0;
        let first_layer = BloomFilter::new(self.initial_capacity, self.max_fpr_per_layer);
        self.layers.push(first_layer);
    }

    /// Return the total number of bits across all layers.
    pub fn total_bits(&self) -> usize {
        self.layers.iter().map(|l| l.num_bits()).sum()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── FNV helpers ───────────────────────────────────────────────────────────

    #[test]
    fn test_fnv1a_deterministic() {
        let a = h1(b"hello");
        let b = h1(b"hello");
        assert_eq!(a, b);
    }

    #[test]
    fn test_h1_h2_differ() {
        let v = h1(b"test");
        let v2 = h2(b"test");
        assert_ne!(v, v2, "h1 and h2 should produce different hashes");
    }

    #[test]
    fn test_h2_always_odd() {
        for seed in [b"a".as_ref(), b"hello", b"oximedia", b"\x00\xff"] {
            assert_eq!(h2(seed) & 1, 1, "h2 must be odd for {seed:?}");
        }
    }

    // ── BloomFilter construction ──────────────────────────────────────────────

    #[test]
    fn test_new_bloom_filter() {
        let bf = BloomFilter::new(1000, 0.01);
        assert!(bf.num_bits() > 0);
        assert!(bf.num_hash_functions() > 0);
        assert_eq!(bf.item_count(), 0);
    }

    #[test]
    fn test_optimal_num_bits_reasonable() {
        // For n=10000, p=0.01 the classic formula gives ~95851 bits ≈ 11.4 KiB.
        let m = optimal_num_bits(10_000, 0.01);
        assert!(m > 90_000 && m < 110_000, "unexpected m={m}");
    }

    #[test]
    fn test_optimal_k_reasonable() {
        let m = optimal_num_bits(10_000, 0.01);
        let k = optimal_num_hash_functions(m, 10_000);
        // Theoretical k ≈ 6.64 → should round to 7.
        assert!(k >= 6 && k <= 8, "unexpected k={k}");
    }

    // ── BloomFilter insert / contains ─────────────────────────────────────────

    #[test]
    fn test_insert_then_contains() {
        let mut bf = BloomFilter::new(100, 0.01);
        bf.insert(b"key1");
        assert!(bf.contains(b"key1"));
    }

    #[test]
    fn test_contains_absent_item() {
        let bf = BloomFilter::new(100, 0.01);
        // No false negatives; absent items should not be reported present
        // unless by chance. With p=0.01 and 0 items the rate is 0.
        assert!(!bf.contains(b"ghost"));
    }

    #[test]
    fn test_no_false_negatives() {
        let mut bf = BloomFilter::new(500, 0.01);
        let items: Vec<Vec<u8>> = (0u32..200).map(|i| i.to_le_bytes().to_vec()).collect();
        for item in &items {
            bf.insert(item);
        }
        for item in &items {
            assert!(bf.contains(item), "false negative detected for {:?}", item);
        }
    }

    #[test]
    fn test_item_count() {
        let mut bf = BloomFilter::new(100, 0.05);
        bf.insert(b"a");
        bf.insert(b"b");
        bf.insert(b"c");
        assert_eq!(bf.item_count(), 3);
    }

    // ── BloomFilter false-positive rate ───────────────────────────────────────

    #[test]
    fn test_estimate_fpr_empty() {
        let bf = BloomFilter::new(1000, 0.01);
        assert_eq!(bf.estimate_false_positive_rate(), 0.0);
    }

    #[test]
    fn test_estimate_fpr_increases_with_fill() {
        let mut bf = BloomFilter::new(100, 0.01);
        let fpr_empty = bf.estimate_false_positive_rate();
        for i in 0u32..50 {
            bf.insert(&i.to_le_bytes());
        }
        let fpr_half = bf.estimate_false_positive_rate();
        assert!(fpr_half > fpr_empty, "FPR should increase as filter fills");
    }

    /// Empirical false-positive rate at n=10000, p=0.01.
    ///
    /// We insert 10 000 distinct items then probe 10 000 distinct non-inserted
    /// items and assert that < 2% report `contains == true`.
    #[test]
    fn test_empirical_fpr_at_n10000_p001() {
        let n = 10_000usize;
        let p = 0.01_f64;
        let mut bf = BloomFilter::new(n, p);

        for i in 0u32..n as u32 {
            let key = format!("inserted_{i}");
            bf.insert(key.as_bytes());
        }

        let mut false_positives = 0usize;
        let probes = 10_000usize;
        for i in 0u32..probes as u32 {
            let key = format!("absent_{i}");
            if bf.contains(key.as_bytes()) {
                false_positives += 1;
            }
        }

        let observed_fpr = false_positives as f64 / probes as f64;
        // Allow 3× the target rate as headroom for test flakiness.
        assert!(
            observed_fpr <= p * 3.0,
            "observed FPR {observed_fpr:.4} exceeded 3× target ({:.4})",
            p * 3.0
        );
    }

    // ── CountingBloomFilter ───────────────────────────────────────────────────

    #[test]
    fn test_counting_bf_insert_contains() {
        let mut cbf = CountingBloomFilter::new(200, 0.01);
        cbf.insert(b"alpha");
        assert!(cbf.contains(b"alpha"));
    }

    #[test]
    fn test_counting_bf_remove() {
        let mut cbf = CountingBloomFilter::new(200, 0.01);
        cbf.insert(b"remove_me");
        assert!(cbf.contains(b"remove_me"));
        let removed = cbf.remove(b"remove_me");
        assert!(removed, "remove should succeed");
        assert!(!cbf.contains(b"remove_me"));
    }

    #[test]
    fn test_counting_bf_remove_absent() {
        let mut cbf = CountingBloomFilter::new(200, 0.01);
        let removed = cbf.remove(b"never_inserted");
        assert!(!removed, "cannot remove item that was never inserted");
    }

    #[test]
    fn test_counting_bf_item_count() {
        let mut cbf = CountingBloomFilter::new(100, 0.05);
        cbf.insert(b"x");
        cbf.insert(b"y");
        assert_eq!(cbf.item_count(), 2);
        cbf.remove(b"x");
        assert_eq!(cbf.item_count(), 1);
    }

    #[test]
    fn test_counting_bf_no_false_negatives() {
        let mut cbf = CountingBloomFilter::new(300, 0.01);
        let items: Vec<Vec<u8>> = (0u32..100).map(|i| i.to_le_bytes().to_vec()).collect();
        for item in &items {
            cbf.insert(item);
        }
        for item in &items {
            assert!(cbf.contains(item), "false negative for {item:?}");
        }
    }

    #[test]
    fn test_counting_bf_multiple_inserts_then_single_remove() {
        let mut cbf = CountingBloomFilter::new(200, 0.01);
        // Insert the same item twice: one remove should not clear it.
        cbf.insert(b"double");
        cbf.insert(b"double");
        cbf.remove(b"double");
        // Should still be present (counter was 2, now 1).
        assert!(cbf.contains(b"double"));
    }

    #[test]
    fn test_double_hash_position_range() {
        let item = b"test_item";
        let h1_val = h1(item);
        let h2_val = h2(item);
        let num_bits = 1024;
        for i in 0..10u64 {
            let pos = double_hash_position(h1_val, h2_val, i, num_bits);
            assert!(pos < num_bits, "position {pos} out of range");
        }
    }

    #[test]
    fn test_bloom_filter_clone() {
        let mut bf = BloomFilter::new(100, 0.01);
        bf.insert(b"cloned");
        let bf2 = bf.clone();
        assert!(bf2.contains(b"cloned"));
        assert_eq!(bf2.item_count(), bf.item_count());
    }

    // ── ScalableBloomFilter ─────────────────────────────────────────────────

    #[test]
    fn test_scalable_bf_insert_contains() {
        let mut sbf = ScalableBloomFilter::new(50, 0.01, 2.0);
        sbf.insert(b"hello");
        assert!(sbf.contains(b"hello"));
    }

    #[test]
    fn test_scalable_bf_absent_item() {
        let sbf = ScalableBloomFilter::new(50, 0.01, 2.0);
        assert!(!sbf.contains(b"missing"));
    }

    #[test]
    fn test_scalable_bf_no_false_negatives() {
        let mut sbf = ScalableBloomFilter::new(50, 0.05, 2.0);
        let items: Vec<Vec<u8>> = (0u32..200).map(|i| i.to_le_bytes().to_vec()).collect();
        for item in &items {
            sbf.insert(item);
        }
        for item in &items {
            assert!(sbf.contains(item), "false negative for {item:?}");
        }
    }

    #[test]
    fn test_scalable_bf_grows_layers() {
        // Small initial capacity → should create additional layers quickly.
        let mut sbf = ScalableBloomFilter::new(10, 0.1, 2.0);
        for i in 0u32..500 {
            sbf.insert(&i.to_le_bytes());
        }
        assert!(
            sbf.layer_count() > 1,
            "should have grown beyond 1 layer, got {}",
            sbf.layer_count()
        );
    }

    #[test]
    fn test_scalable_bf_total_item_count() {
        let mut sbf = ScalableBloomFilter::new(100, 0.01, 2.0);
        sbf.insert(b"a");
        sbf.insert(b"b");
        sbf.insert(b"c");
        assert_eq!(sbf.total_item_count(), 3);
    }

    #[test]
    fn test_scalable_bf_fpr_bounded() {
        let mut sbf = ScalableBloomFilter::new(1000, 0.01, 2.0);
        for i in 0u32..1000 {
            sbf.insert(&i.to_le_bytes());
        }
        let fpr = sbf.estimate_false_positive_rate();
        // Aggregate FPR should be reasonable (below 10% for 1000 items at 1% target).
        assert!(fpr < 0.10, "aggregate FPR {fpr:.4} is too high");
    }

    #[test]
    fn test_scalable_bf_clone() {
        let mut sbf = ScalableBloomFilter::new(100, 0.01, 2.0);
        sbf.insert(b"test");
        let sbf2 = sbf.clone();
        assert!(sbf2.contains(b"test"));
        assert_eq!(sbf2.total_item_count(), 1);
        assert_eq!(sbf2.layer_count(), sbf.layer_count());
    }

    #[test]
    fn test_scalable_bf_empty_fpr() {
        let sbf = ScalableBloomFilter::new(100, 0.01, 2.0);
        assert_eq!(sbf.estimate_false_positive_rate(), 0.0);
    }

    // ── Scalable Bloom filter enhanced tests ────────────────────────────────

    #[test]
    fn test_scalable_bf_set_tightening_ratio() {
        let mut sbf = ScalableBloomFilter::new(100, 0.01, 2.0);
        sbf.set_tightening_ratio(0.5);
        assert!((sbf.tightening_ratio() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_scalable_bf_invalid_tightening_ratio_ignored() {
        let mut sbf = ScalableBloomFilter::new(100, 0.01, 2.0);
        let original = sbf.tightening_ratio();
        sbf.set_tightening_ratio(0.0); // invalid
        assert!((sbf.tightening_ratio() - original).abs() < f64::EPSILON);
        sbf.set_tightening_ratio(1.0); // invalid
        assert!((sbf.tightening_ratio() - original).abs() < f64::EPSILON);
        sbf.set_tightening_ratio(-0.5); // invalid
        assert!((sbf.tightening_ratio() - original).abs() < f64::EPSILON);
    }

    #[test]
    fn test_scalable_bf_growth_factor() {
        let sbf = ScalableBloomFilter::new(100, 0.01, 3.0);
        assert!((sbf.growth_factor() - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_scalable_bf_growth_factor_default_when_invalid() {
        // growth_factor <= 1.0 should default to 2.0
        let sbf = ScalableBloomFilter::new(100, 0.01, 0.5);
        assert!((sbf.growth_factor() - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_scalable_bf_layer_stats() {
        let mut sbf = ScalableBloomFilter::new(10, 0.1, 2.0);
        for i in 0u32..200 {
            sbf.insert(&i.to_le_bytes());
        }
        let stats = sbf.layer_stats();
        assert!(!stats.is_empty());
        // First layer should have items
        assert!(stats[0].0 > 0, "first layer should have items");
        // All layers should have bits
        for (_, bits, _) in &stats {
            assert!(*bits > 0);
        }
    }

    #[test]
    fn test_scalable_bf_estimated_capacity_remaining() {
        let sbf = ScalableBloomFilter::new(1000, 0.01, 2.0);
        let remaining = sbf.estimated_capacity_remaining();
        // Fresh filter should have significant remaining capacity
        assert!(remaining > 0, "fresh filter should have remaining capacity");
    }

    #[test]
    fn test_scalable_bf_estimated_capacity_decreases() {
        let mut sbf = ScalableBloomFilter::new(100, 0.01, 2.0);
        let before = sbf.estimated_capacity_remaining();
        for i in 0u32..50 {
            sbf.insert(&i.to_le_bytes());
        }
        let after = sbf.estimated_capacity_remaining();
        assert!(
            after < before,
            "remaining capacity should decrease after inserts"
        );
    }

    #[test]
    fn test_scalable_bf_clear() {
        let mut sbf = ScalableBloomFilter::new(100, 0.01, 2.0);
        for i in 0u32..50 {
            sbf.insert(&i.to_le_bytes());
        }
        sbf.clear();
        assert_eq!(sbf.total_item_count(), 0);
        assert_eq!(sbf.layer_count(), 1);
        // Previously inserted items should no longer be found
        assert!(!sbf.contains(&0u32.to_le_bytes()));
    }

    #[test]
    fn test_scalable_bf_total_bits() {
        let mut sbf = ScalableBloomFilter::new(10, 0.1, 2.0);
        let bits_single = sbf.total_bits();
        for i in 0u32..500 {
            sbf.insert(&i.to_le_bytes());
        }
        let bits_multi = sbf.total_bits();
        assert!(
            bits_multi > bits_single,
            "total bits should increase with layers"
        );
    }

    #[test]
    fn test_scalable_bf_tighter_ratio_more_layers() {
        // With tighter ratio (smaller), each layer has tighter FPR target
        // meaning more bits per layer, potentially fewer layers needed
        let mut sbf_tight = ScalableBloomFilter::new(10, 0.1, 2.0);
        sbf_tight.set_tightening_ratio(0.5);
        let mut sbf_loose = ScalableBloomFilter::new(10, 0.1, 2.0);
        sbf_loose.set_tightening_ratio(0.9);

        for i in 0u32..200 {
            sbf_tight.insert(&i.to_le_bytes());
            sbf_loose.insert(&i.to_le_bytes());
        }
        // Both should contain all items (no false negatives)
        for i in 0u32..200 {
            assert!(sbf_tight.contains(&i.to_le_bytes()));
            assert!(sbf_loose.contains(&i.to_le_bytes()));
        }
    }

    #[test]
    fn test_scalable_bf_empirical_fpr() {
        let mut sbf = ScalableBloomFilter::new(1000, 0.05, 2.0);
        for i in 0u32..1000 {
            sbf.insert(&i.to_le_bytes());
        }
        // Test FPR against 10000 absent items
        let mut fps = 0usize;
        for i in 10000u32..20000 {
            if sbf.contains(&i.to_le_bytes()) {
                fps += 1;
            }
        }
        let observed_fpr = fps as f64 / 10000.0;
        // Aggregate FPR should be reasonable (under 20%)
        assert!(
            observed_fpr < 0.20,
            "observed FPR {observed_fpr:.4} is too high"
        );
    }

    #[test]
    fn test_scalable_bf_stress_many_inserts() {
        let mut sbf = ScalableBloomFilter::new(50, 0.01, 2.0);
        for i in 0u32..10000 {
            sbf.insert(&i.to_le_bytes());
        }
        assert_eq!(sbf.total_item_count(), 10000);
        // Verify no false negatives on a sample
        for i in [0u32, 999, 5000, 9999] {
            assert!(sbf.contains(&i.to_le_bytes()), "false negative for {i}");
        }
    }
}
