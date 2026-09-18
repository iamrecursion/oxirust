use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use oximedia_pipeline::{
    builder::PipelineBuilder,
    graph::PipelineGraph,
    node::{
        FilterConfig, FrameFormat, NodeSpec, SinkConfig, SourceConfig, StreamSpec, SyntheticSource,
    },
};
use std::hint::black_box;

fn build_linear_pipeline_direct(depth: usize) -> PipelineGraph {
    let video_spec = StreamSpec::video(FrameFormat::Yuv420p, 1920, 1080, 25);
    let mut graph = PipelineGraph::new();

    // Source
    let src_spec = NodeSpec::source(
        "src",
        SourceConfig::Synthetic(SyntheticSource::BlackFrame {
            width: 1920,
            height: 1080,
            fps: 25.0,
        }),
        video_spec.clone(),
    );
    let mut prev_id = graph.add_node(src_spec);

    // Chain of scale filters
    for i in 0..depth {
        let filter_spec = NodeSpec::filter(
            format!("scale-{i}"),
            FilterConfig::Scale {
                width: 1920,
                height: 1080,
            },
            video_spec.clone(),
            video_spec.clone(),
        );
        let filter_id = graph.add_node(filter_spec);
        graph
            .connect(prev_id, "default", filter_id, "default")
            .expect("connect");
        prev_id = filter_id;
    }

    // Sink
    let sink_spec = NodeSpec::sink("sink", SinkConfig::Null, video_spec.clone());
    let sink_id = graph.add_node(sink_spec);
    graph
        .connect(prev_id, "default", sink_id, "default")
        .expect("connect sink");

    graph
}

fn bench_pipeline_build_direct(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline_build_direct");
    for depth in [4usize, 16, 64] {
        group.bench_with_input(BenchmarkId::new("depth", depth), &depth, |b, &depth| {
            b.iter(|| {
                let g = build_linear_pipeline_direct(black_box(depth));
                black_box(g.node_count())
            });
        });
    }
    group.finish();
}

fn bench_pipeline_build_dsl(c: &mut Criterion) {
    c.bench_function("pipeline_build_dsl_scale_hflip_vflip_trim", |b| {
        b.iter(|| {
            let graph = PipelineBuilder::new()
                .source("input", SourceConfig::File("video.mkv".into()))
                .scale(1280, 720)
                .hflip()
                .vflip()
                .trim(0, 60_000)
                .sink("output", SinkConfig::Null)
                .build()
                .expect("pipeline");
            black_box(graph.node_count());
        });
    });
}

fn bench_pipeline_validate(c: &mut Criterion) {
    let graph = build_linear_pipeline_direct(32);

    c.bench_function("pipeline_validate_depth32", |b| {
        b.iter(|| {
            let errs = graph.validate();
            black_box(errs.len());
        });
    });
}

fn bench_pipeline_topological_sort(c: &mut Criterion) {
    let graph = build_linear_pipeline_direct(64);

    c.bench_function("pipeline_topological_sort_depth64", |b| {
        b.iter(|| {
            let order = graph.topological_sort().expect("topo");
            black_box(order.len());
        });
    });
}

criterion_group!(
    benches,
    bench_pipeline_build_direct,
    bench_pipeline_build_dsl,
    bench_pipeline_validate,
    bench_pipeline_topological_sort,
);
criterion_main!(benches);
