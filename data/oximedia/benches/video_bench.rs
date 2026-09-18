use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use oximedia_video::{
    deinterlace::{bob_deinterlace, FieldParity, YadifConfig, YadifFilter, YadifMode},
    scene_detection::{detect_change, extract_features},
};
use std::hint::black_box;

fn bench_bob_deinterlace(c: &mut Criterion) {
    let w = 1920u32;
    let h = 1080u32;
    let frame = vec![128u8; (w * h) as usize];

    c.bench_function("bob_deinterlace_1080p_luma", |b| {
        b.iter(|| {
            let out = bob_deinterlace(black_box(&frame), black_box(w), black_box(h), true);
            black_box(out.len());
        });
    });
}

fn bench_yadif_filter(c: &mut Criterion) {
    let w = 1920u32;
    let h = 1080u32;
    // Yadif requires three temporal frames (prev, curr, next)
    let prev = vec![100u8; (w * h) as usize];
    let curr = vec![128u8; (w * h) as usize];
    let next = vec![140u8; (w * h) as usize];

    let cfg = YadifConfig {
        mode: YadifMode::SendFrame,
        parity: FieldParity::TopFirst,
    };
    let filter = YadifFilter::new(cfg);

    c.bench_function("yadif_full_1080p_luma", |b| {
        b.iter(|| {
            let out = filter.process_frame(
                black_box(&prev),
                black_box(&curr),
                black_box(&next),
                black_box(w),
                black_box(h),
            );
            black_box(out.len());
        });
    });
}

fn bench_scene_detection(c: &mut Criterion) {
    let w = 1920u32;
    let h = 1080u32;
    // Alternate between two slightly different frames
    let frame_a = vec![80u8; (w * h) as usize];
    let frame_b = vec![160u8; (w * h) as usize];

    let mut group = c.benchmark_group("scene_detection_extract_features");
    for n in [100usize, 500, 1000] {
        group.bench_with_input(BenchmarkId::new("frames", n), &n, |b, &n| {
            b.iter(|| {
                let mut changes = 0usize;
                for i in 0..n {
                    let f = if i % 2 == 0 { &frame_a } else { &frame_b };
                    let feat = extract_features(black_box(f), w, h, i as u64);
                    let prev_raw = if i % 2 == 0 { &frame_b } else { &frame_a };
                    let prev =
                        extract_features(black_box(prev_raw), w, h, i.saturating_sub(1) as u64);
                    if detect_change(&prev, &feat).is_some() {
                        changes += 1;
                    }
                }
                black_box(changes)
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_bob_deinterlace,
    bench_yadif_filter,
    bench_scene_detection
);
criterion_main!(benches);
