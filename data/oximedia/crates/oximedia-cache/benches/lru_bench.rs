//! Criterion benchmark for the arena-backed [`LruCache`] after the swap to a
//! `hashbrown::HashMap` index (foldhash default hasher).
//!
//! Three workloads are measured at a steady 100K-entry working set:
//!   * `put`      — cold-fill 100K inserts (exercises slot allocation + map insert);
//!   * `get_hit`  — random lookups of resident keys (exercises map probe + LRU splice);
//!   * `mixed`    — 90% get / 10% put at steady state (the realistic cache load).
//!
//! Keys are drawn from an inline xorshift64 PRNG (same pattern as
//! `benches/tinylfu.rs`) so the bench pulls in no `rand` dependency.

use criterion::{criterion_group, criterion_main, Criterion};
use oximedia_cache::lru_cache::LruCache;
use std::hint::black_box;

/// Working-set / capacity for every benchmark.
const CAP: usize = 100_000;
/// Per-entry byte size reported to the cache (uniform; value is arbitrary).
const SIZE_BYTES: usize = 8;

/// Minimal xorshift64 PRNG (identical algorithm to `benches/tinylfu.rs`).
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        // Avoid the all-zero fixed point of xorshift.
        Self { state: seed | 1 }
    }

    #[inline(always)]
    fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
}

/// Build a fully-populated `LruCache` with keys `prng`-drawn modulo `CAP`.
///
/// Using `key % CAP` over `CAP` inserts leaves the cache near-full with a
/// dense-but-non-contiguous key set, which is what `get_hit`/`mixed` probe.
fn prefilled() -> (LruCache<u64, u64>, Vec<u64>) {
    let mut cache: LruCache<u64, u64> = LruCache::new(CAP);
    let mut prng = XorShift64::new(0xcafe_f00d_d15e_a5e5);
    let mut keys = Vec::with_capacity(CAP);
    for _ in 0..CAP {
        let k = prng.next() % (CAP as u64);
        cache.insert(k, k.wrapping_mul(2_654_435_761), SIZE_BYTES);
        keys.push(k);
    }
    (cache, keys)
}

fn bench_lru(c: &mut Criterion) {
    let mut group = c.benchmark_group("lru_cache");

    // ── put: cold-fill 100K inserts ──────────────────────────────────────────
    group.bench_function("put", |b| {
        b.iter(|| {
            let mut cache: LruCache<u64, u64> = LruCache::new(CAP);
            let mut prng = XorShift64::new(0x1234_5678_9abc_def0);
            for _ in 0..CAP {
                let k = prng.next() % (CAP as u64);
                cache.insert(black_box(k), black_box(k), SIZE_BYTES);
            }
            black_box(cache.len())
        });
    });

    // ── get_hit: random lookups of resident keys ─────────────────────────────
    group.bench_function("get_hit", |b| {
        let (mut cache, keys) = prefilled();
        let mut i: usize = 0;
        b.iter(|| {
            // Cycle deterministically through the inserted key sequence.
            let k = keys[i % keys.len()];
            i = i.wrapping_add(1);
            // Copy the hit value out so no borrow of `cache` escapes the closure.
            black_box(cache.get(black_box(&k)).copied())
        });
    });

    // ── mixed: 90% get / 10% put at steady 100K ──────────────────────────────
    group.bench_function("mixed", |b| {
        let (mut cache, _keys) = prefilled();
        let mut prng = XorShift64::new(0x0bad_c0de_dead_beef);
        b.iter(|| {
            let r = prng.next();
            let k = r % (CAP as u64);
            if r % 10 == 0 {
                // 10% writes — re-insert keeps the working set at ~CAP.
                cache.insert(black_box(k), black_box(k), SIZE_BYTES);
            } else {
                // 90% reads.
                black_box(cache.get(black_box(&k)));
            }
        });
    });

    group.finish();
}

criterion_group!(benches, bench_lru);
criterion_main!(benches);
