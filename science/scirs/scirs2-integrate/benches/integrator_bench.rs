use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use scirs2_core::ndarray::ArrayView1;
use scirs2_integrate::{
    gaussian::gauss_legendre,
    monte_carlo::{monte_carlo, MonteCarloOptions},
    quad::{quad, simpson, trapezoid},
    romberg::romberg,
};
use std::marker::PhantomData;
use std::time::Duration;

#[allow(dead_code)]
fn bench_integrators(c: &mut Criterion) {
    let mut group = c.benchmark_group("integration_methods");
    group.measurement_time(Duration::from_secs(5));

    // Test function: sin(x)
    let f_sin = |x: f64| x.sin();
    let range = (0.0, std::f64::consts::PI);

    // 1D quadrature methods
    group.bench_function("trapezoid_1d", |b| {
        b.iter(|| trapezoid(f_sin, range.0, range.1, 1000))
    });

    group.bench_function("simpson_1d", |b| {
        b.iter(|| simpson(f_sin, range.0, range.1, 1000))
    });

    group.bench_function("quad", |b| b.iter(|| quad(f_sin, range.0, range.1, None)));

    // Adaptive quadrature is not publicly exported yet
    // group.bench_function("adaptive_quad", |b| {
    //     b.iter(|| adaptive_quad(f_sin, range.0, range.1, 1e-8, 1e-8, None))
    // });

    group.bench_function("gauss_legendre_5", |b| {
        b.iter(|| gauss_legendre(f_sin, range.0, range.1, 5))
    });

    group.bench_function("gauss_legendre_10", |b| {
        b.iter(|| gauss_legendre(f_sin, range.0, range.1, 10))
    });

    group.bench_function("romberg", |b| {
        b.iter(|| romberg(f_sin, range.0, range.1, None))
    });

    // Multidimensional function: f(x, y) = sin(x) * cos(y)
    let f_2d = |point: ArrayView1<f64>| point[0].sin() * point[1].cos();

    // Monte Carlo integration
    for n_samples in [1000, 10000, 100000].iter() {
        group.bench_with_input(
            BenchmarkId::new("monte_carlo_2d", n_samples),
            n_samples,
            |b, &n_samples| {
                let options = MonteCarloOptions {
                    n_samples,
                    seed: Some(42),
                    phantom: PhantomData,
                    ..Default::default()
                };

                b.iter(|| {
                    monte_carlo(
                        f_2d,
                        &[(0.0, std::f64::consts::PI), (0.0, std::f64::consts::PI)],
                        Some(options.clone()),
                    )
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_integrators);
criterion_main!(benches);
