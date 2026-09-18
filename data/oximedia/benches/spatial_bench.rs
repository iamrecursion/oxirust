use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use oximedia_spatial::{
    ambisonics::{AmbisonicsDecoder, AmbisonicsEncoder, AmbisonicsOrder, SoundSource},
    wave_field::{VirtualSource, WfsArray, WfsRenderer},
};
use std::hint::black_box;

fn bench_ambisonics_encode(c: &mut Criterion) {
    // 1 second of mono audio at 48 kHz
    let samples: Vec<f32> = (0..48_000)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48_000.0).sin())
        .collect();
    let source = SoundSource::new(45.0_f32.to_radians(), 20.0_f32.to_radians());

    let mut group = c.benchmark_group("ambisonics_encode");
    let orders = [
        (AmbisonicsOrder::First, "order1"),
        (AmbisonicsOrder::Second, "order2"),
        (AmbisonicsOrder::Third, "order3"),
    ];
    for (order, label) in orders {
        let encoder = AmbisonicsEncoder::new(order, 48_000);
        group.bench_with_input(BenchmarkId::new("mono_1s_48k", label), &label, |b, _| {
            b.iter(|| {
                let channels = encoder.encode_mono(black_box(&samples), black_box(&source));
                black_box(channels.len())
            });
        });
    }
    group.finish();
}

fn bench_ambisonics_decode_stereo(c: &mut Criterion) {
    let order = AmbisonicsOrder::Third;
    let encoder = AmbisonicsEncoder::new(order, 48_000);
    let decoder = AmbisonicsDecoder::new(order);
    let samples: Vec<f32> = (0..48_000)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48_000.0).sin())
        .collect();
    let source = SoundSource::new(45.0_f32.to_radians(), 0.0);
    let channels = encoder.encode_mono(&samples, &source);

    c.bench_function("ambisonics_decode_stereo_3rd_1s_48k", |b| {
        b.iter(|| {
            let (l, r) = decoder.decode_stereo(black_box(&channels));
            black_box((l.len(), r.len()));
        });
    });
}

fn bench_wfs_delays_and_gains(c: &mut Criterion) {
    // 32-speaker linear array, 4 cm spacing → 1.28 m aperture
    let array = WfsArray::linear(32, 0.04);
    let renderer = WfsRenderer::new(array, 48_000);
    let source = VirtualSource::point(0.0, 2.0);

    c.bench_function("wfs_delays_and_gains_32ch", |b| {
        b.iter(|| {
            let dg = renderer.compute_delays_and_gains(black_box(&source));
            black_box(dg.len());
        });
    });
}

criterion_group!(
    benches,
    bench_ambisonics_encode,
    bench_ambisonics_decode_stereo,
    bench_wfs_delays_and_gains,
);
criterion_main!(benches);
