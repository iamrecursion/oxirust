// This benchmark uses the unstable test feature which requires nightly Rust
// Replacing with a simple criterion benchmark

use criterion::{criterion_group, criterion_main, Criterion};
use scirs2_core::array_protocol::{self, add, matmul, transpose, NdarrayWrapper};
use scirs2_core::ndarray_ext::Array2;

// Add bench_ndarray_matmul with criterion
#[allow(dead_code)]
fn bench_ndarray_matmul(c: &mut Criterion) {
    let a = Array2::<f64>::ones((100, 100));
    let b = Array2::<f64>::ones((100, 100));

    c.bench_function("ndarray_matmul", |bench| bench.iter(|| a.dot(&b)));
}

// Add bench_array_protocol_matmul with criterion
#[allow(dead_code)]
fn bench_array_protocol_matmul(c: &mut Criterion) {
    array_protocol::init();

    let a = Array2::<f64>::ones((100, 100));
    let b = Array2::<f64>::ones((100, 100));

    let wrapped_a = NdarrayWrapper::new(a);
    let wrapped_b = NdarrayWrapper::new(b);

    c.bench_function("array_protocol_matmul", |bench| {
        bench.iter(|| matmul(&wrapped_a, &wrapped_b).expect("Operation failed"))
    });
}

// Add bench_ndarray_add with criterion
#[allow(dead_code)]
fn bench_ndarray_add(c: &mut Criterion) {
    let a = Array2::<f64>::ones((100, 100));
    let b = Array2::<f64>::ones((100, 100));

    c.bench_function("ndarray_add", |bench| bench.iter(|| &a + &b));
}

// Add bench_array_protocol_add with criterion
#[allow(dead_code)]
fn bench_array_protocol_add(c: &mut Criterion) {
    array_protocol::init();

    let a = Array2::<f64>::ones((100, 100));
    let b = Array2::<f64>::ones((100, 100));

    let wrapped_a = NdarrayWrapper::new(a);
    let wrapped_b = NdarrayWrapper::new(b);

    c.bench_function("array_protocol_add", |bench| {
        bench.iter(|| add(&wrapped_a, &wrapped_b).expect("Operation failed"))
    });
}

// Add bench_ndarray_transpose with criterion
#[allow(dead_code)]
fn bench_ndarray_transpose(c: &mut Criterion) {
    let a = Array2::<f64>::ones((100, 100));

    c.bench_function("ndarray_transpose", |bench| bench.iter(|| a.t().to_owned()));
}

// Add bench_array_protocol_transpose with criterion
#[allow(dead_code)]
fn bench_array_protocol_transpose(c: &mut Criterion) {
    array_protocol::init();

    let a = Array2::<f64>::ones((100, 100));
    let wrapped_a = NdarrayWrapper::new(a);

    c.bench_function("array_protocol_transpose", |bench| {
        bench.iter(|| transpose(&wrapped_a).expect("Operation failed"))
    });
}

#[cfg(feature = "gpu")]
#[allow(dead_code)]
fn bench_gpu_array_matmul(c: &mut Criterion) {
    use scirs2_core::array_protocol::{GPUBackend, GPUConfig, GPUNdarray};

    array_protocol::init();

    let a = Array2::<f64>::ones((100, 100));
    let b = Array2::<f64>::ones((100, 100));

    let gpu_config = GPUConfig {
        backend: GPUBackend::CUDA,
        device_id: 0,
        async_ops: false,
        mixed_precision: false,
        memory_fraction: 0.9,
    };

    let gpu_a = GPUNdarray::new(a, gpu_config.clone());
    let gpu_b = GPUNdarray::new(b, gpu_config);

    c.bench_function("gpu_array_matmul", |bench| {
        bench.iter(|| matmul(&gpu_a, &gpu_b).expect("Operation failed"))
    });
}

criterion_group!(
    benches,
    bench_ndarray_matmul,
    bench_array_protocol_matmul,
    bench_ndarray_add,
    bench_array_protocol_add,
    bench_ndarray_transpose,
    bench_array_protocol_transpose,
);

#[cfg(feature = "gpu")]
criterion_group!(gpu_benches, bench_gpu_array_matmul,);

#[cfg(feature = "gpu")]
criterion_main!(benches, gpu_benches);

#[cfg(not(feature = "gpu"))]
criterion_main!(benches);
