//! Criterion benchmarks for Fast/Medium/Slow/Placebo preset configurations.
//!
//! Measures the overhead of building an `Optimizer` and running the AQ engine
//! over a synthetic frame sequence, giving a relative speed proxy for each
//! preset level.  Also reports a quality proxy (variance-based VMAF predictor)
//! so the benchmark output shows both speed *and* quality trade-offs.
//!
//! These benchmarks do NOT perform actual video encoding; they exercise the
//! configuration and analysis pipeline (AQ + RDO cost calculation) to give a
//! realistic comparison of preset overhead.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use oximedia_optimize::{
    AqEngine, OptimizationPresets, Optimizer, OptimizerConfig, VmafPredictor, VmafPredictorConfig,
};

// ─── helpers ─────────────────────────────────────────────────────────────────

/// A minimal synthetic frame: 64×64 pixels with a gradient pattern.
fn synthetic_frame(seed: u8) -> Vec<u8> {
    (0usize..64 * 64)
        .map(|i| {
            let row = i / 64;
            let col = i % 64;
            ((row * 5 + col * 3 + usize::from(seed)) % 256) as u8
        })
        .collect()
}

/// Quality proxy: use the VMAF predictor to estimate a score from raw pixels.
fn quality_proxy_score(reference: &[u8], distorted: &[u8]) -> f64 {
    let config = VmafPredictorConfig::default();
    let mut predictor = VmafPredictor::new(config);
    let prediction = predictor.predict_from_pixels(reference, distorted, 64, 64, None);
    prediction.score
}

// ─── benchmarks ──────────────────────────────────────────────────────────────

/// Preset construction bench: how long does `Optimizer::new` take per preset?
fn bench_preset_construction(c: &mut Criterion) {
    let configs: Vec<(&str, OptimizerConfig)> = vec![
        ("fast", OptimizationPresets::fast()),
        ("medium", OptimizationPresets::medium()),
        ("slow", OptimizationPresets::slow()),
        ("placebo", OptimizationPresets::placebo()),
    ];

    let mut group = c.benchmark_group("preset_construction");
    for (name, config) in &configs {
        group.bench_with_input(BenchmarkId::from_parameter(name), config, |b, cfg| {
            b.iter(|| {
                let optimizer = Optimizer::new(cfg.clone()).expect("optimizer must build");
                black_box(optimizer.config().level)
            });
        });
    }
    group.finish();
}

/// AQ analysis bench: run the AQ engine over a 64-frame synthetic sequence
/// for each preset.  Measures per-frame AQ overhead.
fn bench_preset_aq_analysis(c: &mut Criterion) {
    let frames: Vec<Vec<u8>> = (0u8..64).map(synthetic_frame).collect();

    let configs: Vec<(&str, OptimizerConfig)> = vec![
        ("fast", OptimizationPresets::fast()),
        ("medium", OptimizationPresets::medium()),
        ("slow", OptimizationPresets::slow()),
        ("placebo", OptimizationPresets::placebo()),
    ];

    let mut group = c.benchmark_group("preset_aq_64frames");
    for (name, config) in &configs {
        let aq = AqEngine::new(config).expect("AQ engine must build");
        group.bench_with_input(BenchmarkId::from_parameter(name), &aq, |b, aq_engine| {
            b.iter(|| {
                let mut total_offset = 0i32;
                for frame in &frames {
                    // Use first 64 pixels as an 8×8 block
                    let block: &[u8] = if frame.len() >= 64 {
                        &frame[..64]
                    } else {
                        frame
                    };
                    let result = aq_engine.calculate_aq(block, 8);
                    total_offset += i32::from(result.qp_offset);
                }
                black_box(total_offset)
            });
        });
    }
    group.finish();
}

/// Quality proxy bench: compare predicted VMAF between a reference and a
/// synthetically degraded version to simulate quality differences per preset.
/// Fast preset → more degradation, placebo → minimal degradation.
fn bench_preset_quality_proxy(c: &mut Criterion) {
    let reference = synthetic_frame(42);

    let mut group = c.benchmark_group("preset_quality_proxy");
    for (name, degradation) in [("fast", 20u8), ("medium", 10), ("slow", 4), ("placebo", 1)] {
        let distorted: Vec<u8> = reference
            .iter()
            .map(|&p| p.saturating_add(degradation))
            .collect();

        group.bench_with_input(BenchmarkId::from_parameter(name), &distorted, |b, dist| {
            b.iter(|| {
                let score = quality_proxy_score(&reference, dist);
                black_box(score)
            });
        });
    }
    group.finish();
}

// ─── criterion wiring ────────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_preset_construction,
    bench_preset_aq_analysis,
    bench_preset_quality_proxy,
);
criterion_main!(benches);
