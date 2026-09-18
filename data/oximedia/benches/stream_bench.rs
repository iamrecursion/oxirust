use criterion::{criterion_group, criterion_main, Criterion};
use oximedia_stream::bola::{BolaConfig, BolaState};
use std::hint::black_box;

fn bench_bola_select_quality(c: &mut Criterion) {
    // Five ladder rungs: 400 kbps … 8 Mbps
    let bitrates: Vec<u64> = vec![400_000, 800_000, 1_500_000, 3_000_000, 8_000_000];
    let cfg = BolaConfig::new(bitrates, 30.0, 4.0).expect("BolaConfig");
    let mut state = BolaState::new(cfg);

    c.bench_function("bola_select_quality_1000_iter", |b| {
        b.iter(|| {
            // Simulate buffer oscillating between 2 s and 20 s
            let mut last = 0usize;
            for i in 0u64..1000 {
                let buf = 2.0 + (i as f64 % 18.0);
                last = state.select_quality(black_box(buf));
                state.on_segment_downloaded(black_box(3_000_000));
            }
            black_box(last);
        });
    });
}

fn bench_bola_utility_computation(c: &mut Criterion) {
    let bitrates: Vec<u64> = (1..=10).map(|x| x * 1_000_000).collect();
    let min_br = bitrates[0];

    c.bench_function("bola_utility_value_10k_calls", |b| {
        b.iter(|| {
            let mut sum = 0.0f64;
            for &br in bitrates.iter().cycle().take(10_000) {
                sum += BolaConfig::utility_value(black_box(br), black_box(min_br));
            }
            black_box(sum);
        });
    });
}

criterion_group!(
    benches,
    bench_bola_select_quality,
    bench_bola_utility_computation
);
criterion_main!(benches);
