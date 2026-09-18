//! Forward-evaluation throughput of the tensorized EML forest across batch sizes.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use phop_bench::synth_1d;
use phop_core::eval_tree;
use scirs2_core::ndarray::Array2;
use std::hint::black_box;

fn bench_forward(c: &mut Criterion) {
    // exp(x) - ln(x) over varying batch sizes.
    let tree = oxieml_tree();
    let mut group = c.benchmark_group("forward_eval");
    for &n in &[1_000usize, 10_000, 100_000] {
        let ds = synth_1d(|x| x.exp() - x.ln(), 0.1, 10.0, n);
        let x: Array2<f64> = ds.x.clone();
        group.bench_with_input(BenchmarkId::from_parameter(n), &x, |b, x| {
            b.iter(|| {
                let out = eval_tree(black_box(&tree), black_box(x)).unwrap();
                black_box(out);
            });
        });
    }
    group.finish();
}

fn oxieml_tree() -> phop_core::EmlTree {
    use phop_core::EmlTree;
    EmlTree::eml(&EmlTree::var(0), &EmlTree::var(0))
}

criterion_group!(benches, bench_forward);
criterion_main!(benches);
