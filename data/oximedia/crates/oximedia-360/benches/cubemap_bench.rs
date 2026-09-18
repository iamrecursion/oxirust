//! Criterion benchmarks for cubemap face extraction at various resolutions.
//!
//! Benchmarks the following at 1K, 2K, 4K, and 8K equirectangular resolutions:
//! * `equirect_to_cube` — row-major CPU conversion
//! * `equirect_to_cube_tiled` — cache-friendly tiled conversion
//! * `equirect_to_cube_parallel` — rayon-parallel tiled conversion

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use oximedia_360::tiled::{equirect_to_cube_parallel, equirect_to_cube_tiled};

/// Build a solid-colour equirectangular test image of size `w × h` (RGB).
fn make_equirect(w: u32, h: u32) -> Vec<u8> {
    vec![128u8; (w * h * 3) as usize]
}

/// Reduced sample count for the 8K rows. The 8K source is ~96 MiB and each
/// conversion is slow, so criterion's default 100 samples is impractical for
/// CI. Criterion has no per-benchmark `#[ignore]`, so a reduced `sample_size`
/// is the idiomatic way to cap wall-clock cost while still measuring 8K.
const HEAVY_SAMPLE_SIZE: usize = 10;

fn bench_tiled_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("equirect_to_cube_tiled");

    // Benchmark at the cache-friendly resolutions: 1K, 2K, 4K.
    for &(label, equirect_w, equirect_h, face_size) in &[
        ("1K", 1024u32, 512u32, 256u32),
        ("2K", 2048u32, 1024u32, 512u32),
        ("4K", 4096u32, 2048u32, 1024u32),
    ] {
        let src = make_equirect(equirect_w, equirect_h);
        group.bench_with_input(
            BenchmarkId::new("tiled-16", label),
            &(equirect_w, equirect_h, face_size),
            |b, &(w, h, fs)| {
                b.iter(|| equirect_to_cube_tiled(&src, w, h, fs, 16).expect("tiled ok"));
            },
        );
    }

    // 8K under a reduced sample count (heavy: ~96 MiB source). Hoist the source
    // allocation outside `b.iter` so only the conversion is timed.
    group.sample_size(HEAVY_SAMPLE_SIZE);
    let (w8k, h8k, fs8k) = (8192u32, 4096u32, 2048u32);
    let src_8k = make_equirect(w8k, h8k);
    group.bench_with_input(
        BenchmarkId::new("tiled-16", "8K"),
        &(w8k, h8k, fs8k),
        |b, &(w, h, fs)| {
            b.iter(|| equirect_to_cube_tiled(&src_8k, w, h, fs, 16).expect("tiled ok"));
        },
    );

    group.finish();
}

fn bench_parallel_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("equirect_to_cube_parallel");

    // Mirror the tiled resolutions (1K–4K) for direct comparison.
    for &(label, equirect_w, equirect_h, face_size) in &[
        ("1K", 1024u32, 512u32, 256u32),
        ("2K", 2048u32, 1024u32, 512u32),
        ("4K", 4096u32, 2048u32, 1024u32),
    ] {
        let src = make_equirect(equirect_w, equirect_h);
        group.bench_with_input(
            BenchmarkId::new("parallel", label),
            &(equirect_w, equirect_h, face_size),
            |b, &(w, h, fs)| {
                b.iter(|| equirect_to_cube_parallel(&src, w, h, fs).expect("parallel ok"));
            },
        );
    }

    // 8K under a reduced sample count (heavy: ~96 MiB source), source hoisted.
    group.sample_size(HEAVY_SAMPLE_SIZE);
    let (w8k, h8k, fs8k) = (8192u32, 4096u32, 2048u32);
    let src_8k = make_equirect(w8k, h8k);
    group.bench_with_input(
        BenchmarkId::new("parallel", "8K"),
        &(w8k, h8k, fs8k),
        |b, &(w, h, fs)| {
            b.iter(|| equirect_to_cube_parallel(&src_8k, w, h, fs).expect("parallel ok"));
        },
    );

    group.finish();
}

criterion_group!(benches, bench_tiled_conversion, bench_parallel_conversion);
criterion_main!(benches);
