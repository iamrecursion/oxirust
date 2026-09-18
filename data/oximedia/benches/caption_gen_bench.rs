use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use oximedia_caption_gen::{
    greedy_break, optimal_break, optimal_break_larsch, optimal_break_smawk,
};
use std::hint::black_box;

// ~500 words of broadcast-ready caption text at 42 chars/line (BBC standard)
const PARA_500W: &str = "\
The sovereign media framework provides patent-free memory-safe multimedia processing \
in pure Rust. It reconstructs both FFmpeg and OpenCV capabilities in a single unified \
codebase with zero unsafe code and full workspace integration. All codecs are patent-free \
and include AV1, VP9, VP8, Opus, Vorbis, FLAC, PCM, FFV1, Y4M, JPEG-XL, and DNG. \
The architecture spans approximately one hundred and nine crates all at stable status. \
Caption generation leverages the Knuth-Plass dynamic programming algorithm to achieve \
optimal line breaking that minimises the sum of squared raggedness across all lines. \
The SMAWK algorithm provides a further acceleration to O of n log n by exploiting the \
totally monotone property of the optimal cost matrix. \
The Larsch streaming variant further reduces constant factors for real-time subtitle \
production by processing each row of the SMAWK matrix in amortised O of one time. \
Broadcast requirements mandate a maximum of forty-two characters per line and a reading \
speed not exceeding seventeen characters per second for general audiences. \
The WCAG 2.1 compliance checker validates every generated caption block against the \
applicable success criteria including minimum display duration maximum characters per \
second and sufficient colour contrast ratios for readable text on screen. \
Speaker diarization attributes caption blocks to individual speakers so that multi-party \
dialogues can be rendered with appropriate positioning and colour coding per speaker. \
Forced-narrative classification identifies audio descriptions and sound effects that must \
remain synchronised with specific video moments regardless of available buffer time. \
Punctuation restoration applies a rule-based model to raw ASR transcripts that typically \
lack capitalisation and sentence boundaries producing broadcast-ready text. \
Multi-language synchronisation aligns translated tracks to the reference language using \
anchor-point interpolation and handles the well-known expansion factor in European \
languages where the translated text is typically thirty percent longer than English. \
The pipeline DSL allows operators to compose caption generation nodes declaratively \
and the execution planner selects the optimal execution order based on data dependencies \
and available parallelism across the available CPU cores of the target machine.";

fn bench_greedy_break(c: &mut Criterion) {
    let mut group = c.benchmark_group("line_breaking_greedy");
    for w in [32u8, 42, 56] {
        group.bench_with_input(BenchmarkId::new("max_width", w), &w, |b, &w| {
            b.iter(|| {
                let lines = greedy_break(black_box(PARA_500W), black_box(w));
                black_box(lines.len())
            });
        });
    }
    group.finish();
}

fn bench_optimal_break(c: &mut Criterion) {
    let mut group = c.benchmark_group("line_breaking_optimal_dp");
    for w in [32u8, 42, 56] {
        group.bench_with_input(BenchmarkId::new("max_width", w), &w, |b, &w| {
            b.iter(|| {
                let lines = optimal_break(black_box(PARA_500W), black_box(w));
                black_box(lines.len())
            });
        });
    }
    group.finish();
}

fn bench_optimal_break_smawk(c: &mut Criterion) {
    let mut group = c.benchmark_group("line_breaking_smawk");
    for w in [32u8, 42, 56] {
        group.bench_with_input(BenchmarkId::new("max_width", w), &w, |b, &w| {
            b.iter(|| {
                let lines = optimal_break_smawk(black_box(PARA_500W), black_box(w));
                black_box(lines.len())
            });
        });
    }
    group.finish();
}

fn bench_optimal_break_larsch(c: &mut Criterion) {
    let mut group = c.benchmark_group("line_breaking_larsch");
    for w in [32u8, 42, 56] {
        group.bench_with_input(BenchmarkId::new("max_width", w), &w, |b, &w| {
            b.iter(|| {
                let lines = optimal_break_larsch(black_box(PARA_500W), black_box(w));
                black_box(lines.len())
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_greedy_break,
    bench_optimal_break,
    bench_optimal_break_smawk,
    bench_optimal_break_larsch,
);
criterion_main!(benches);
