use criterion::{criterion_group, criterion_main, Criterion};
use oximedia_hdr::{
    hdr_histogram::{HdrHistogram, HdrHistogramConfig, LuminanceHistogram},
    pq_eotf_batch, pq_oetf_batch,
    tone_mapping::{ToneMapper, ToneMappingConfig, ToneMappingOperator},
};
use std::hint::black_box;

fn bench_pq_eotf_batch(c: &mut Criterion) {
    // 1920×1080 PQ-encoded signal values in [0, 1]
    let samples: Vec<f32> = (0..1920 * 1080)
        .map(|i| (i as f32) / (1920.0 * 1080.0))
        .collect();
    let mut output = vec![0.0f32; samples.len()];

    c.bench_function("pq_eotf_batch_1080p", |b| {
        b.iter(|| {
            pq_eotf_batch(black_box(&samples), black_box(&mut output)).expect("eotf");
            black_box(output[0]);
        });
    });
}

fn bench_pq_oetf_batch(c: &mut Criterion) {
    // Linear light values normalised to [0, 1]
    let samples: Vec<f32> = (0..1920 * 1080)
        .map(|i| (i as f32) / (1920.0 * 1080.0))
        .collect();
    let mut output = vec![0.0f32; samples.len()];

    c.bench_function("pq_oetf_batch_1080p", |b| {
        b.iter(|| {
            pq_oetf_batch(black_box(&samples), black_box(&mut output)).expect("oetf");
            black_box(output[0]);
        });
    });
}

fn bench_hdr_histogram_accumulate(c: &mut Criterion) {
    let mut hist = HdrHistogram::new(1000, 0.01, 10_000.0).expect("histogram");
    // 1920×1080 luminance samples in nits
    let samples: Vec<f32> = (0..1920 * 1080)
        .map(|i| {
            let t = (i as f32) / (1920.0 * 1080.0);
            0.05_f32 * (4000.0_f32 / 0.05).powf(t)
        })
        .collect();

    c.bench_function("hdr_histogram_accumulate_1080p", |b| {
        b.iter(|| {
            for &s in &samples {
                hist.accumulate(black_box(s));
            }
            black_box(hist.total_pixels());
        });
    });
}

fn bench_hdr_luminance_histogram(c: &mut Criterion) {
    let config = HdrHistogramConfig::default();
    // Interleaved RGB in nits
    let frame: Vec<f32> = (0..1920 * 1080 * 3)
        .map(|i| (i as f32) / (1920.0 * 1080.0 * 3.0) * 1000.0)
        .collect();

    c.bench_function("luminance_histogram_from_nits_1080p", |b| {
        b.iter(|| {
            let lh = LuminanceHistogram::from_linear_nits(
                black_box(&frame),
                1920,
                1080,
                black_box(&config),
            )
            .expect("lh");
            black_box(lh.max_cll());
        });
    });
}

fn bench_tone_mapping_hdr_to_sdr(c: &mut Criterion) {
    let config = ToneMappingConfig {
        operator: ToneMappingOperator::Hable,
        input_peak_nits: 1000.0,
        output_peak_nits: 100.0,
        exposure: 1.0,
        saturation: 1.0,
        gamma_out: 2.2,
    };
    let mapper = ToneMapper::new(config);
    // 1920×1080 HDR linear-light RGB interleaved
    let frame: Vec<f32> = (0..1920 * 1080 * 3)
        .map(|i| (i as f32) / (1920.0 * 1080.0 * 3.0))
        .collect();

    c.bench_function("hable_tone_map_1080p", |b| {
        b.iter(|| {
            let sdr = mapper.map_frame(black_box(&frame)).expect("map_frame");
            black_box(sdr.len());
        });
    });
}

criterion_group!(
    benches,
    bench_pq_eotf_batch,
    bench_pq_oetf_batch,
    bench_hdr_histogram_accumulate,
    bench_hdr_luminance_histogram,
    bench_tone_mapping_hdr_to_sdr,
);
criterion_main!(benches);
