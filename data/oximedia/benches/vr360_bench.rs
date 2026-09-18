use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use oximedia_360::{projection::equirect_to_cube, tiled::equirect_to_cube_tiled};
use std::hint::black_box;

fn bench_equirect_to_cube(c: &mut Criterion) {
    let mut group = c.benchmark_group("equirect_to_cube");

    let sizes: &[(u32, u32, u32, &str)] = &[
        (512, 256, 128, "512x256_face128"),
        (1024, 512, 256, "1024x512_face256"),
        (2048, 1024, 512, "2048x1024_face512"),
    ];

    for &(w, h, face_size, label) in sizes {
        let src = vec![128u8; (w * h * 3) as usize];
        group.bench_with_input(BenchmarkId::new("projection", label), label, |b, _| {
            b.iter(|| {
                let faces = equirect_to_cube(
                    black_box(&src),
                    black_box(w),
                    black_box(h),
                    black_box(face_size),
                )
                .expect("equirect_to_cube");
                black_box(faces.len())
            });
        });
    }
    group.finish();
}

fn bench_equirect_to_cube_tiled(c: &mut Criterion) {
    // Benchmark the tiled parallel variant on a medium input
    let w = 1024u32;
    let h = 512u32;
    let face_size = 256u32;
    let tile_size = 64u32;
    let src = vec![128u8; (w * h * 3) as usize];

    c.bench_function("equirect_to_cube_tiled_1024x512_face256", |b| {
        b.iter(|| {
            let faces = equirect_to_cube_tiled(
                black_box(&src),
                black_box(w),
                black_box(h),
                black_box(face_size),
                black_box(tile_size),
            )
            .expect("tiled");
            black_box(faces.len());
        });
    });
}

criterion_group!(
    benches,
    bench_equirect_to_cube,
    bench_equirect_to_cube_tiled
);
criterion_main!(benches);
