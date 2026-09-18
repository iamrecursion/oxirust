use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use oximedia_neural::{
    activations::{apply_activation, apply_activation_inplace, ActivationFn},
    layers::Conv2dLayer,
    tensor::Tensor,
};
use std::hint::black_box;

fn bench_conv2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv2d");

    // Benchmark 3×3 conv on different spatial sizes
    let configs: &[(usize, usize, &str)] = &[
        (64, 64, "64x64"),
        (128, 128, "128x128"),
        (256, 256, "256x256"),
    ];

    for &(h, w, label) in configs {
        // 16 input channels → 32 output channels, 3×3 kernel, stride 1, pad 1
        let conv = Conv2dLayer::new(16, 32, 3, 3, (1, 1), (1, 1)).expect("conv");
        let input = Tensor::ones(vec![16, h, w]).expect("input");

        group.bench_with_input(BenchmarkId::new("in16_out32_3x3", label), label, |b, _| {
            b.iter(|| {
                let out = conv.forward(black_box(&input)).expect("forward");
                black_box(out.numel())
            });
        });
    }
    group.finish();
}

fn bench_activation_relu_inplace(c: &mut Criterion) {
    let n = 256 * 256 * 32;
    let mut t = Tensor::ones(vec![n]).expect("tensor");

    c.bench_function("relu_inplace_256x256x32", |b| {
        b.iter(|| {
            apply_activation_inplace(black_box(&mut t), &ActivationFn::Relu);
            black_box(t.numel());
        });
    });
}

fn bench_activation_gelu(c: &mut Criterion) {
    let n = 256 * 256 * 32;
    let t = Tensor::ones(vec![n]).expect("tensor");

    c.bench_function("gelu_256x256x32", |b| {
        b.iter(|| {
            let out = apply_activation(black_box(&t), &ActivationFn::Gelu);
            black_box(out.numel());
        });
    });
}

criterion_group!(
    benches,
    bench_conv2d,
    bench_activation_relu_inplace,
    bench_activation_gelu,
);
criterion_main!(benches);
