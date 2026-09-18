# OxiMedia Benchmarks

This directory contains comprehensive performance benchmarks for OxiMedia using [Criterion.rs](https://github.com/bheisler/criterion.rs).

## Quick Start

### Run All Benchmarks

```bash
# Using cargo directly
cargo bench --manifest-path benches/Cargo.toml

# Using the convenience script
./benches/run_benchmarks.sh
```

### Run Specific Benchmark Suite

```bash
# Container benchmarks
cargo bench --manifest-path benches/Cargo.toml --bench container_bench

# Codec benchmarks
cargo bench --manifest-path benches/Cargo.toml --bench codec_bench

# Audio benchmarks
cargo bench --manifest-path benches/Cargo.toml --bench audio_bench

# Computer vision benchmarks
cargo bench --manifest-path benches/Cargo.toml --bench cv_bench

# Filter graph benchmarks
cargo bench --manifest-path benches/Cargo.toml --bench graph_bench

# I/O performance benchmarks
cargo bench --manifest-path benches/Cargo.toml --bench io_bench
```

### Quick Benchmark Check

```bash
./benches/run_benchmarks.sh --quick
```

## Benchmark Suites

### 1. Container Benchmarks (`container_bench.rs`)

Measures performance of container format operations:

- **Matroska/WebM**: EBML parsing, demuxing throughput
- **Ogg**: Page parsing, stream demuxing
- **FLAC**: Frame parsing, metadata reading
- **MP4**: Box parsing, atom traversal
- **WAV**: Header parsing, PCM data reading
- **Format Detection**: Magic byte probing

**Typical Results:**
- WebM demuxing: 800+ MB/s
- Ogg demuxing: 600+ MB/s
- Format probing: <1 µs per file

### 2. Codec Benchmarks (`codec_bench.rs`)

Measures video codec performance:

- **AV1**: Frame header parsing, tile decoding, transforms (IDCT/IADST), loop filter, CDEF
- **VP8**: Boolean decoder, frame decoding
- **VP9**: Tile decoding, entropy coding
- **Transform Operations**: 4x4 to 64x64 block sizes
- **Prediction**: Intra prediction modes, motion compensation

**Typical Results:**
- AV1 1080p decode: 60+ FPS
- VP8 720p decode: 200+ FPS
- Transform operations: <10 µs per block

### 3. Audio Benchmarks (`audio_bench.rs`)

Measures audio codec and processing performance:

- **FLAC**: Frame decoding, Rice/Golomb coding, LPC prediction
- **Opus**: Packet decoding, range decoder, SILK/CELT
- **Vorbis**: Packet decoding, MDCT
- **PCM Operations**: Format conversion, resampling
- **Audio Filters**: EQ, compressor, limiter, mixing

**Typical Results:**
- FLAC decode: 50x+ realtime
- Opus decode: 100x+ realtime
- Audio resampling: 80x+ realtime

### 4. Computer Vision Benchmarks (`cv_bench.rs`)

Measures computer vision operation performance:

- **Face Detection**: Haar cascade, integral image
- **Object Detection**: YOLO inference, NMS
- **Image Enhancement**: ESRGAN preprocessing
- **Edge Detection**: Sobel, Canny
- **Optical Flow**: Lucas-Kanade
- **Color Conversion**: RGB/YUV transformations
- **Image Processing**: Resize, blur, filters

**Typical Results:**
- Sobel edge (1080p): 100+ FPS
- RGB to gray (1080p): 500+ FPS
- Gaussian blur (720p): 80+ FPS

### 5. Filter Graph Benchmarks (`graph_bench.rs`)

Measures filter graph pipeline performance:

- **Graph Construction**: Building graphs with varying node counts
- **Linear Chains**: Processing through sequential filter chains
- **Branching Topology**: Multi-output graph execution
- **Audio Filters**: Volume, mixing, EQ throughput
- **Graph Validation**: Topology validation overhead
- **State Transitions**: Node lifecycle operations

**Typical Results:**
- Graph construction (10 nodes): <10 µs
- Linear chain (5 filters): <5 µs per frame
- Branching (8 outputs): <20 µs

### 6. I/O Performance Benchmarks (`io_bench.rs`)

Measures I/O layer performance:

- **Memory Source**: Sequential reading, zero-copy slicing
- **Bit Reader**: Single/multi-bit operations, byte alignment
- **Exp-Golomb**: Unsigned/signed variable-length decoding
- **Buffer Management**: Copy operations, position tracking
- **Mixed Patterns**: Realistic header parsing workloads
- **Look-ahead**: Peeking without consuming data

**Typical Results:**
- Memory source creation: <100 ns
- Bit reader (8-bit reads): 1+ GB/s
- Exp-Golomb decoding: 500+ MB/s
- Zero-copy slice: <50 ns

## Viewing Results

### HTML Reports

Criterion generates detailed HTML reports:

```bash
# Open in browser
open target/criterion/report/index.html  # macOS
xdg-open target/criterion/report/index.html  # Linux
```

Reports include:
- Interactive plots
- Statistical analysis
- Performance comparisons
- Regression detection

### Command Line Output

Example output:

```
matroska_demux/small_webm
                        time:   [45.234 µs 45.891 µs 46.621 µs]
                        thrpt:  [2.1457 GiB/s 2.1800 GiB/s 2.2119 GiB/s]
```

- **time**: Mean execution time with confidence interval
- **thrpt**: Throughput (elements/bytes processed per second)

## Comparing with FFmpeg

Run FFmpeg comparison benchmarks:

```bash
./benches/compare_ffmpeg.sh
```

This will:
1. Generate test media files
2. Run FFmpeg benchmarks
3. Create comparison report
4. Save results to `benchmark_results/`

## Baselines and Comparisons

### Save a Baseline

```bash
./benches/run_benchmarks.sh --save-baseline my-optimization
```

### Compare Against Baseline

```bash
./benches/run_benchmarks.sh --baseline my-optimization
```

### CI Comparison

Benchmarks automatically compare against the `main` branch in CI:
- Pull requests compare against base branch
- Regressions >10% trigger warnings
- Regressions >20% fail the build

## Performance Targets

See [BENCHMARKS.md](BENCHMARKS.md) for detailed performance targets and FFmpeg comparisons.

### Key Targets

- Container demuxing: 80%+ of FFmpeg speed
- AV1 decoding: 70%+ of dav1d speed
- Audio decoding: 80%+ of FFmpeg speed
- Edge detection: 80%+ of OpenCV speed

## File Structure

```
benches/
├── Cargo.toml              # Benchmark dependencies
├── README.md               # This file
├── BENCHMARKS.md           # Detailed benchmark documentation
├── container_bench.rs      # Container format benchmarks
├── codec_bench.rs          # Video codec benchmarks
├── audio_bench.rs          # Audio codec/processing benchmarks
├── cv_bench.rs             # Computer vision benchmarks
├── graph_bench.rs          # Filter graph benchmarks
├── io_bench.rs             # I/O performance benchmarks
├── helpers/
│   └── mod.rs              # Helper utilities
├── run_benchmarks.sh       # Convenience script
└── compare_ffmpeg.sh       # FFmpeg comparison script
```

## Adding New Benchmarks

### 1. Create Benchmark Function

```rust
fn my_operation_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("my_operation");

    // Configure throughput measurement
    group.throughput(Throughput::Bytes(data_size));

    // Add benchmark
    group.bench_function("operation_name", |b| {
        b.iter(|| {
            // Your code here
            black_box(my_operation());
        });
    });

    group.finish();
}
```

### 2. Add to Criterion Group

```rust
criterion_group!(
    benches,
    existing_benchmark,
    my_operation_benchmark,  // Add here
);
```

### 3. Update Documentation

- Add performance targets to `BENCHMARKS.md`
- Document the benchmark in this README
- Add FFmpeg comparison if applicable

## Best Practices

### Use `black_box()`

Always wrap benchmark inputs/outputs:

```rust
b.iter(|| {
    let result = my_function(black_box(&input));
    black_box(result)
});
```

### Appropriate Measurement Times

- Fast operations (<1ms): Default settings
- Medium operations (1-100ms): 10 seconds
- Slow operations (>100ms): 15+ seconds, reduce sample size

```rust
group.measurement_time(Duration::from_secs(10));
group.sample_size(10);  // For slow operations
```

### Throughput Metrics

Add throughput for meaningful comparisons:

```rust
// For data processing
group.throughput(Throughput::Bytes(data.len() as u64));

// For frame processing
group.throughput(Throughput::Elements(num_frames));
```

### Multiple Input Sizes

Test with various sizes:

```rust
for (name, size) in [("small", 100), ("large", 10000)] {
    group.bench_with_input(
        BenchmarkId::from_parameter(name),
        &size,
        |b, &size| { /* ... */ }
    );
}
```

## Profiling

For deeper analysis:

```bash
# Using perf (Linux)
cargo bench --manifest-path benches/Cargo.toml --bench codec_bench -- --profile-time=5

# Using flamegraph
cargo flamegraph --bench codec_bench

# Using cachegrind
valgrind --tool=cachegrind target/release/deps/codec_bench-*
```

## CI Integration

Benchmarks run automatically in CI:

- **Pull Requests**: Quick benchmark subset
- **Main Branch**: Full benchmark suite with historical tracking
- **Nightly**: Comprehensive benchmarks + FFmpeg comparison

See [`.github/workflows/benchmarks.yml`](../.github/workflows/benchmarks.yml) for details.

## Troubleshooting

### Benchmarks Take Too Long

```bash
# Run quick mode
./benches/run_benchmarks.sh --quick

# Run specific benchmark
./benches/run_benchmarks.sh --bench container_bench
```

### Inconsistent Results

- Close unnecessary applications
- Disable CPU frequency scaling
- Run multiple times and compare
- Check for thermal throttling

### Missing Dependencies

```bash
# Ensure all dependencies are installed
cargo build --release --manifest-path benches/Cargo.toml
```

## Resources

- [Criterion.rs Book](https://bheisler.github.io/criterion.rs/book/)
- [Rust Performance Book](https://nnethercote.github.io/perf-book/)
- [FFmpeg Benchmarking Guide](https://trac.ffmpeg.org/wiki/Benchmarking)

## License

Apache-2.0 (same as OxiMedia)
