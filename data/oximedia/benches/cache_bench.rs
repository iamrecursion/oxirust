use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use oximedia_cache::cache_partitioning::{CacheEntry, CachePartition};
use std::hint::black_box;

fn bench_cache_partition_put(c: &mut Criterion) {
    // 4 GiB partition capacity (ensures no evictions during bench)
    let cap = 4 * 1024 * 1024 * 1024;

    let mut group = c.benchmark_group("cache_partition");
    for n in [1_000usize, 5_000, 10_000] {
        group.bench_with_input(BenchmarkId::new("put", n), &n, |b, &n| {
            b.iter(|| {
                let mut part = CachePartition::new("bench", cap);
                for i in 0..n {
                    let key = format!("seg:{i:06}");
                    let entry = CacheEntry::new(vec![0u8; 128]);
                    part.put(black_box(key), black_box(entry));
                }
                black_box(part.len())
            });
        });
    }
    group.finish();
}

fn bench_cache_partition_get(c: &mut Criterion) {
    let cap = 4 * 1024 * 1024 * 1024;
    let n = 10_000usize;
    let mut part = CachePartition::new("bench", cap);
    for i in 0..n {
        let key = format!("seg:{i:06}");
        let entry = CacheEntry::new(vec![0u8; 128]);
        part.put(key, entry);
    }

    c.bench_function("cache_partition_get_10k", |b| {
        b.iter(|| {
            for i in 0..n {
                let key = format!("seg:{i:06}");
                let _ = part.get(black_box(&key));
            }
            black_box(part.len());
        });
    });
}

criterion_group!(
    benches,
    bench_cache_partition_put,
    bench_cache_partition_get
);
criterion_main!(benches);
