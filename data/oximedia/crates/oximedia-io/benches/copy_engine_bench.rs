//! Throughput benchmarks for the `oximedia-io` copy engine.
//!
//! Compares [`CopyMode::Buffered`], [`CopyMode::Chunked`], and
//! [`CopyMode::ZeroCopy`] over a fixed 4 MiB deterministic source file, plus a
//! separate group measuring [`SplicePipe::transfer`] over in-memory cursors
//! (the "splice" dimension as this crate implements it — a portable, pure-Rust
//! user-space pipe, with no Linux `splice(2)` syscall and therefore no `cfg`
//! gate).
//!
//! Each file-copy iteration writes to a fresh destination path so the timed
//! work is a complete copy (setup uses `BatchSize::PerIteration`).

use std::hint::black_box;
use std::io::{Cursor, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};

use oximedia_io::copy_engine::{CopyEngine, CopyJob, CopyMode};
use oximedia_io::splice_pipe::SplicePipe;

/// Total payload size for the copy benchmarks (4 MiB).
const PAYLOAD_BYTES: usize = 4 * 1024 * 1024;

/// Process-unique counter so each fresh destination path is distinct even
/// across rapid setup calls within a single benchmark run.
static DST_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Build a deterministic 4 MiB payload (a simple byte ramp; content is
/// irrelevant to copy throughput but must be reproducible across runs).
fn deterministic_payload() -> Vec<u8> {
    (0..PAYLOAD_BYTES).map(|i| (i % 251) as u8).collect()
}

/// Create the shared 4 MiB source file once and return its path.
fn create_source() -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "oximedia_io_copy_bench_src_{}.bin",
        std::process::id()
    ));
    let payload = deterministic_payload();
    let mut f = std::fs::File::create(&path).expect("create bench source file");
    f.write_all(&payload).expect("write bench source payload");
    f.flush().expect("flush bench source");
    path
}

/// Return a fresh, unique destination path for one timed copy iteration.
fn fresh_dst() -> PathBuf {
    let n = DST_COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "oximedia_io_copy_bench_dst_{}_{}.bin",
        std::process::id(),
        n
    ))
}

fn bench_copy_engine(c: &mut Criterion) {
    let src = create_source();
    let engine = CopyEngine::new();

    let mut group = c.benchmark_group("copy_engine");
    group.throughput(criterion::Throughput::Bytes(PAYLOAD_BYTES as u64));

    for mode in [CopyMode::Buffered, CopyMode::Chunked, CopyMode::ZeroCopy] {
        group.bench_with_input(
            BenchmarkId::from_parameter(mode.to_string()),
            &mode,
            |b, &mode| {
                b.iter_batched(
                    // Setup (not timed): pick a fresh destination path.
                    fresh_dst,
                    // Routine (timed): perform the full copy, then clean up.
                    |dst| {
                        let job = CopyJob::new(&src, &dst).with_mode(mode);
                        let result = engine.run(&job).expect("copy job must succeed");
                        black_box(result.bytes_copied);
                        let _ = std::fs::remove_file(&dst);
                    },
                    BatchSize::PerIteration,
                );
            },
        );
    }

    group.finish();

    let _ = std::fs::remove_file(&src);
}

fn bench_splice_pipe(c: &mut Criterion) {
    // The "splice" path this crate exposes is a portable user-space pipe over
    // any Read/Write. Benchmark it over in-memory cursors so the measurement
    // isolates the pipe-buffer transfer loop from filesystem noise.
    let payload = deterministic_payload();

    let mut group = c.benchmark_group("splice_pipe");
    group.throughput(criterion::Throughput::Bytes(PAYLOAD_BYTES as u64));

    group.bench_function("transfer_cursor_4mib", |b| {
        b.iter_batched(
            // Setup (not timed): fresh reader cursor + empty writer + pipe.
            || {
                let reader = Cursor::new(payload.clone());
                let writer: Vec<u8> = Vec::with_capacity(PAYLOAD_BYTES);
                let pipe = SplicePipe::with_defaults();
                (reader, writer, pipe)
            },
            // Routine (timed): transfer the entire payload through the pipe.
            |(mut reader, mut writer, mut pipe)| {
                let result = pipe
                    .transfer(&mut reader, &mut writer, PAYLOAD_BYTES as u64)
                    .expect("splice transfer must succeed");
                black_box(result.bytes_transferred);
                black_box(writer.len());
            },
            BatchSize::PerIteration,
        );
    });

    group.finish();
}

criterion_group!(benches, bench_copy_engine, bench_splice_pipe);
criterion_main!(benches);
