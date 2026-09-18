use criterion::{criterion_group, criterion_main, Criterion};

fn caption_parsing_benchmark(c: &mut Criterion) {
    c.bench_function("placeholder", |b| b.iter(|| 1 + 1));
}

criterion_group!(benches, caption_parsing_benchmark);
criterion_main!(benches);
