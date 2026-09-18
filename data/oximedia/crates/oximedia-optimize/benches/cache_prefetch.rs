//! Criterion benchmarks for `cache_optimizer` and `prefetch` modules.
//!
//! Simulates a sequential block-scan workload of 1000 blocks, measuring
//! cache lookup throughput and prefetch queue throughput with and without
//! hints pre-populated.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use oximedia_optimize::cache_optimizer::{CacheEntry, CacheKey, CacheOptimizer, CachePolicy};
use oximedia_optimize::prefetch::{PrefetchHint, PrefetchQueue, PrefetchStrategy};

// ─── helpers ─────────────────────────────────────────────────────────────────

fn make_segment_entry(index: u32, now: u64) -> CacheEntry {
    CacheEntry {
        key: CacheKey {
            url_path: format!("/segment{index:04}.ts"),
            vary_headers: vec![],
        },
        size_bytes: 8192,
        policy: CachePolicy::Standard,
        cached_at: now,
        hit_count: 0,
    }
}

fn make_prefetch_hint(asset_id: u64, base_epoch: u64) -> PrefetchHint {
    PrefetchHint {
        asset_id,
        priority: (asset_id % 256) as u8,
        reason: PrefetchStrategy::Sequential,
        estimated_access_epoch: base_epoch + asset_id * 2,
    }
}

// ─── benchmarks ──────────────────────────────────────────────────────────────

/// Bench: sequential scan of 1000 blocks with the cache cold (all misses).
fn bench_cache_scan_cold(c: &mut Criterion) {
    c.bench_function("cache_scan_cold_1000", |b| {
        b.iter(|| {
            let mut cache = CacheOptimizer::new();
            let now = 1_000_000u64;
            for i in 0..1000u32 {
                let entry = make_segment_entry(i, now);
                cache.put(entry);
                let _ = cache.get(&format!("/segment{i:04}.ts"), now);
            }
            black_box(cache.total_size_bytes())
        });
    });
}

/// Bench: sequential scan of 1000 blocks with a warm cache (all hits).
fn bench_cache_scan_warm(c: &mut Criterion) {
    // Pre-populate 1000 entries, then measure pure lookup throughput.
    let now = 1_000_000u64;
    let mut cache = CacheOptimizer::new();
    for i in 0..1000u32 {
        cache.put(make_segment_entry(i, now));
    }

    c.bench_function("cache_scan_warm_1000", |b| {
        b.iter(|| {
            let mut hits = 0u32;
            for i in 0..1000u32 {
                if cache.get(&format!("/segment{i:04}.ts"), now).is_some() {
                    hits += 1;
                }
            }
            black_box(hits)
        });
    });
}

/// Bench: prefetch queue — adding hints (sequential fill).
fn bench_prefetch_add(c: &mut Criterion) {
    c.bench_function("prefetch_add_200", |b| {
        b.iter(|| {
            let mut queue = PrefetchQueue::new(200);
            let base = 2_000_000u64;
            for i in 0..200u64 {
                queue.add(make_prefetch_hint(i, base));
            }
            black_box(queue.pending_count())
        });
    });
}

/// Bench: prefetch queue — top-N priority selection.
fn bench_prefetch_top_priority(c: &mut Criterion) {
    let base = 2_000_000u64;
    let mut queue = PrefetchQueue::new(200);
    for i in 0..200u64 {
        queue.add(make_prefetch_hint(i, base));
    }

    for n in [10usize, 50, 100] {
        c.bench_with_input(BenchmarkId::new("prefetch_top_n", n), &n, |b, &top_n| {
            b.iter(|| {
                let top = queue.top_priority_hints(top_n);
                black_box(top.len())
            });
        });
    }
}

/// Bench: evict_expired over a half-stale cache of 1000 entries.
fn bench_cache_evict_expired(c: &mut Criterion) {
    c.bench_function("cache_evict_expired_500_stale", |b| {
        b.iter_batched(
            || {
                let mut cache = CacheOptimizer::new();
                // First 500 entries cached at t=0 (Standard = 30s TTL → stale at t=31)
                for i in 0..500u32 {
                    cache.put(CacheEntry {
                        key: CacheKey {
                            url_path: format!("/old{i}.ts"),
                            vary_headers: vec![],
                        },
                        size_bytes: 4096,
                        policy: CachePolicy::Standard,
                        cached_at: 0,
                        hit_count: 0,
                    });
                }
                // Next 500 entries are long-lived (86 400s TTL)
                for i in 0..500u32 {
                    cache.put(CacheEntry {
                        key: CacheKey {
                            url_path: format!("/new{i}.ts"),
                            vary_headers: vec![],
                        },
                        size_bytes: 4096,
                        policy: CachePolicy::LongLived,
                        cached_at: 0,
                        hit_count: 0,
                    });
                }
                cache
            },
            |mut cache| {
                let evicted = cache.evict_expired(31);
                black_box(evicted)
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

criterion_group!(
    benches,
    bench_cache_scan_cold,
    bench_cache_scan_warm,
    bench_prefetch_add,
    bench_prefetch_top_priority,
    bench_cache_evict_expired,
);
criterion_main!(benches);
