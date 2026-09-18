//! Criterion benchmark for `TinyLfuAdmission::should_admit` + `update` under a
//! Zipf(s=1.0, N=1000) access load.
//!
//! The Zipf distribution is approximated by a discrete inverse-CDF: item `k`
//! (1-indexed) has probability proportional to `1/k`.  We pre-compute the CDF
//! at construction time and draw samples using a simple linear search (acceptable
//! for N=1000 at benchmark time).

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use oximedia_cache::eviction_policies::TinyLfuAdmission;

/// Pre-computed Zipf(s=1.0) distribution over N items.
struct ZipfSampler {
    /// CDF: `cdf[i]` = P(X ≤ i+1).  Length = N.
    cdf: Vec<f64>,
    /// xorshift64 PRNG state.
    state: u64,
}

impl ZipfSampler {
    fn new(n: usize) -> Self {
        // Harmonic normalising constant H_n = sum_{k=1}^{n} 1/k.
        let h_n: f64 = (1..=n).map(|k| 1.0 / k as f64).sum();
        let mut cdf = Vec::with_capacity(n);
        let mut acc = 0.0f64;
        for k in 1..=n {
            acc += 1.0 / (k as f64 * h_n);
            cdf.push(acc);
        }
        Self {
            cdf,
            state: 0xcafef00dd15ea5e5,
        }
    }

    /// Draw a Zipf-distributed sample (returns a 0-indexed item id).
    fn sample(&mut self) -> u64 {
        // xorshift64.
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        // Map to [0, 1).
        let u = (x >> 11) as f64 / (1u64 << 53) as f64;
        // Inverse CDF via linear scan (N=1000, fast enough in bench context).
        let idx = self.cdf.partition_point(|&c| c < u);
        idx.min(self.cdf.len() - 1) as u64
    }
}

// ── Inlined CMS update ────────────────────────────────────────────────────────
//
// The original `should_admit` calls through several layers.  The optimized
// hot-path below inlines the 4-counter CMS update with direct array indexing,
// avoiding the generic trait dispatch seen in the naïve implementation.

/// Direct inline 4-counter Count-Min Sketch increment.
///
/// `counters` is a flat `[u64; 4 * HASH_COUNT]` layout where `HASH_COUNT = 4`.
/// Each hash function selects a different row.  We derive row-specific
/// positions by XOR-folding the key hash with a row salt.
#[inline(always)]
fn cms_increment_inline(counters: &mut [u64; 64], key: u64, capacity: u64) {
    // 4 independent hash functions via XOR-salt.
    const SALTS: [u64; 4] = [
        0x9e37_79b9_7f4a_7c15,
        0x6c62_272e_07bb_0142,
        0x517c_c1b7_2722_0a95,
        0xbf58_476d_1ce4_e5b9,
    ];
    let row_len = 16usize; // 64 / 4 rows
    for (row, &salt) in SALTS.iter().enumerate() {
        let h = key ^ salt;
        let col = (h % capacity).min((row_len as u64) - 1) as usize;
        let idx = row * row_len + col;
        counters[idx] = counters[idx].saturating_add(1);
    }
}

/// Inline minimum over 4 CMS rows.
#[inline(always)]
fn cms_query_inline(counters: &[u64; 64], key: u64, capacity: u64) -> u64 {
    const SALTS: [u64; 4] = [
        0x9e37_79b9_7f4a_7c15,
        0x6c62_272e_07bb_0142,
        0x517c_c1b7_2722_0a95,
        0xbf58_476d_1ce4_e5b9,
    ];
    let row_len = 16usize;
    let mut min = u64::MAX;
    for (row, &salt) in SALTS.iter().enumerate() {
        let h = key ^ salt;
        let col = (h % capacity).min((row_len as u64) - 1) as usize;
        let idx = row * row_len + col;
        min = min.min(counters[idx]);
    }
    min
}

// ── Benchmarks ────────────────────────────────────────────────────────────────

fn bench_should_admit(c: &mut Criterion) {
    let mut group = c.benchmark_group("tinylfu");

    for &n_items in &[100usize, 1000] {
        group.bench_with_input(
            BenchmarkId::new("should_admit_zipf", n_items),
            &n_items,
            |b, &n| {
                let mut gate = TinyLfuAdmission::new(n);
                let mut sampler = ZipfSampler::new(n);
                // Warm up: record 10k accesses so the frequency map is non-trivial.
                for _ in 0..10_000 {
                    gate.record_access(sampler.sample());
                }
                b.iter(|| {
                    let candidate = sampler.sample();
                    let evicted_freq = sampler.sample() % 20;
                    gate.should_admit(candidate, evicted_freq)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("cms_inline_update_query", n_items),
            &n_items,
            |b, &n| {
                let mut counters = [0u64; 64];
                let capacity = n as u64;
                let mut sampler = ZipfSampler::new(n);
                // Warm up.
                for _ in 0..10_000 {
                    let k = sampler.sample();
                    cms_increment_inline(&mut counters, k, capacity);
                }
                b.iter(|| {
                    let k = sampler.sample();
                    cms_increment_inline(&mut counters, k, capacity);
                    cms_query_inline(&counters, k, capacity)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_should_admit);
criterion_main!(benches);
