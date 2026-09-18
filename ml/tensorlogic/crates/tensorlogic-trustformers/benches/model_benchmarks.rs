//! Performance benchmarks for tensorlogic-trustformers
//!
//! These benchmarks measure the performance of various transformer components
//! and complete models across different configurations.
//!
//! Run with:
//! ```bash
//! cargo bench --bench model_benchmarks
//! ```
use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use tensorlogic_ir::EinsumGraph;
use tensorlogic_trustformers::{
    AttentionConfig, EncoderStackConfig, FeedForwardConfig, MultiHeadAttention, SelfAttention,
};

/// Benchmark self-attention graph construction
fn bench_self_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("self_attention");

    for d_model in [256, 512, 768, 1024].iter() {
        group.throughput(Throughput::Elements(*d_model as u64));

        let config = AttentionConfig::new(*d_model, 8).expect("valid attention config parameters");
        let attention = SelfAttention::new(config)
            .expect("valid attention config should construct self-attention");

        group.bench_with_input(BenchmarkId::from_parameter(d_model), d_model, |b, _| {
            b.iter(|| {
                let mut graph = EinsumGraph::new();
                graph.add_tensor("Q");
                graph.add_tensor("K");
                graph.add_tensor("V");
                black_box(
                    attention
                        .build_attention_graph(&mut graph)
                        .expect("attention graph construction should succeed"),
                );
            });
        });
    }

    group.finish();
}

/// Benchmark multi-head attention graph construction
fn bench_multi_head_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_head_attention");

    for (d_model, n_heads) in [(512, 8), (768, 12), (1024, 16)].iter() {
        group.throughput(Throughput::Elements(*d_model as u64));

        let config =
            AttentionConfig::new(*d_model, *n_heads).expect("valid attention config parameters");
        let mha = MultiHeadAttention::new(config)
            .expect("valid attention config should construct multi-head attention");

        group.bench_with_input(
            BenchmarkId::new("d_model", d_model),
            &(*d_model, *n_heads),
            |b, _| {
                b.iter(|| {
                    let mut graph = EinsumGraph::new();
                    graph.add_tensor("Q");
                    graph.add_tensor("K");
                    graph.add_tensor("V");
                    black_box(
                        mha.build_mha_graph(&mut graph)
                            .expect("MHA graph construction should succeed"),
                    );
                });
            },
        );
    }

    group.finish();
}

/// Benchmark feed-forward network graph construction
fn bench_feed_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("feed_forward");

    for (d_model, d_ff) in [(512, 2048), (768, 3072), (1024, 4096)].iter() {
        group.throughput(Throughput::Elements(*d_model as u64));

        let config = FeedForwardConfig::new(*d_model, *d_ff);
        let ffn = tensorlogic_trustformers::FeedForward::new(config)
            .expect("valid FFN config should construct feed-forward");

        group.bench_with_input(
            BenchmarkId::new("d_model", d_model),
            &(*d_model, *d_ff),
            |b, _| {
                b.iter(|| {
                    let mut graph = EinsumGraph::new();
                    graph.add_tensor("x");
                    graph.add_tensor("W1");
                    graph.add_tensor("b1");
                    graph.add_tensor("W2");
                    graph.add_tensor("b2");
                    black_box(
                        ffn.build_ffn_graph(&mut graph)
                            .expect("FFN graph construction should succeed"),
                    );
                });
            },
        );
    }

    group.finish();
}

/// Benchmark complete encoder stack graph construction
fn bench_encoder_stack(c: &mut Criterion) {
    let mut group = c.benchmark_group("encoder_stack");
    group.sample_size(10); // Reduce sample size for expensive operations

    for (n_layers, d_model) in [(6, 512), (12, 768), (24, 1024)].iter() {
        let config = EncoderStackConfig::new(*n_layers, *d_model, 8, d_model * 4, 512)
            .expect("valid encoder stack config parameters");
        let encoder = tensorlogic_trustformers::EncoderStack::new(config)
            .expect("valid encoder stack config should construct encoder");

        group.bench_with_input(
            BenchmarkId::new("layers", n_layers),
            &(*n_layers, *d_model),
            |b, _| {
                b.iter(|| {
                    let mut graph = EinsumGraph::new();
                    graph.add_tensor("input");
                    black_box(
                        encoder
                            .build_encoder_stack_graph(&mut graph)
                            .expect("encoder stack graph construction should succeed"),
                    );
                });
            },
        );
    }

    group.finish();
}

/// Benchmark attention configuration validation
fn bench_config_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("config_validation");

    let config = AttentionConfig::new(512, 8).expect("valid attention config parameters");

    group.bench_function("attention_config_validate", |b| {
        b.iter(|| {
            config
                .validate()
                .expect("valid config should pass validation");
            black_box(())
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_self_attention,
    bench_multi_head_attention,
    bench_feed_forward,
    bench_encoder_stack,
    bench_config_validation,
);
criterion_main!(benches);
