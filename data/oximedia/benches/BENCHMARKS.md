# OxiMedia Benchmarks

Comprehensive performance benchmarking suite for OxiMedia using Criterion.rs.

## Overview

This benchmark suite provides detailed performance analysis of OxiMedia's core functionality:

- **Container Operations**: Demuxing/muxing throughput for various container formats
- **Codec Operations**: Video/audio encoding and decoding performance
- **Audio Processing**: Audio codec performance and DSP operations
- **Computer Vision**: ML inference, edge detection, and image processing

## Running Benchmarks

### Run All Benchmarks

```bash
cargo bench --manifest-path benches/Cargo.toml
```

### Run Specific Benchmark Suite

```bash
# Container benchmarks only
cargo bench --manifest-path benches/Cargo.toml --bench container_bench

# Codec benchmarks only
cargo bench --manifest-path benches/Cargo.toml --bench codec_bench

# Audio benchmarks only
cargo bench --manifest-path benches/Cargo.toml --bench audio_bench

# Computer vision benchmarks only
cargo bench --manifest-path benches/Cargo.toml --bench cv_bench
```

### Run Specific Benchmark

```bash
# Run only AV1 transform benchmarks
cargo bench --manifest-path benches/Cargo.toml --bench codec_bench av1_transform

# Run only Matroska demux benchmarks
cargo bench --manifest-path benches/Cargo.toml --bench container_bench matroska_demux
```

## Benchmark Results Location

- **HTML Reports**: `target/criterion/` - Open `index.html` in a browser
- **Raw Data**: `target/criterion/<benchmark_name>/base/`
- **Comparison Data**: Generated when benchmarks are re-run

## Performance Targets vs FFmpeg

### Container Demuxing (Throughput in MB/s)

| Format | OxiMedia Target | FFmpeg 6.1 | Status |
|--------|----------------|------------|--------|
| Matroska/WebM | 800+ MB/s | ~850 MB/s | Target |
| Ogg | 600+ MB/s | ~650 MB/s | Target |
| FLAC | 500+ MB/s | ~550 MB/s | Target |
| MP4 | 900+ MB/s | ~950 MB/s | Target |
| WAV | 1000+ MB/s | ~1100 MB/s | Target |

### Video Decoding (Frames per Second)

#### AV1 Decoding

| Resolution | OxiMedia Target | FFmpeg (dav1d) | Status |
|------------|----------------|----------------|--------|
| 720p | 120+ FPS | ~150 FPS | Target |
| 1080p | 60+ FPS | ~80 FPS | Target |
| 4K | 15+ FPS | ~20 FPS | Target |

#### VP8 Decoding

| Resolution | OxiMedia Target | FFmpeg (libvpx) | Status |
|------------|----------------|-----------------|--------|
| 720p | 200+ FPS | ~250 FPS | Target |
| 1080p | 100+ FPS | ~120 FPS | Target |

#### VP9 Decoding

| Resolution | OxiMedia Target | FFmpeg (libvpx) | Status |
|------------|----------------|-----------------|--------|
| 720p | 150+ FPS | ~180 FPS | Target |
| 1080p | 80+ FPS | ~100 FPS | Target |
| 4K | 20+ FPS | ~25 FPS | Target |

### Audio Decoding (Real-time Factor)

| Codec | OxiMedia Target | FFmpeg | Status |
|-------|----------------|--------|--------|
| FLAC | 50x+ realtime | ~60x | Target |
| Opus | 100x+ realtime | ~120x | Target |
| Vorbis | 80x+ realtime | ~100x | Target |

### Computer Vision

| Operation | OxiMedia Target | OpenCV | Status |
|-----------|----------------|--------|--------|
| Sobel Edge (1080p) | 100+ FPS | ~120 FPS | Target |
| Gaussian Blur (1080p) | 80+ FPS | ~100 FPS | Target |
| RGB to Gray (1080p) | 500+ FPS | ~600 FPS | Target |

## Benchmark Methodology

### Hardware Configuration

Benchmarks should be run on consistent hardware:

- **CPU**: Modern x86_64 with AVX2 support
- **RAM**: 16GB+ DDR4
- **Storage**: SSD for test data
- **OS**: Linux (Ubuntu 22.04+) or macOS

### Test Data

- Synthetic data generated via `helpers` module
- Deterministic output for reproducibility
- Multiple size variants (small, medium, large)

### Statistical Analysis

Criterion.rs provides:

- **Mean**: Average execution time
- **Std Dev**: Standard deviation of measurements
- **Median**: Middle value of sorted measurements
- **MAD**: Median Absolute Deviation
- **Outliers**: Detection and filtering

### Warm-up and Sampling

- **Warm-up**: 3 seconds by default
- **Measurement**: 5-15 seconds depending on operation
- **Sample Size**: 100 iterations (10 for expensive operations)

## Interpreting Results

### Throughput Metrics

- **MB/s**: Megabytes per second for container operations
- **FPS**: Frames per second for video operations
- **Elements/s**: Samples per second for audio operations

### Performance Changes

Criterion detects performance changes:

- **No change**: Within noise threshold (±5%)
- **Improvement**: Statistically significant speedup
- **Regression**: Statistically significant slowdown

### Comparing with FFmpeg

To compare with FFmpeg:

```bash
# FFmpeg decode benchmark
ffmpeg -benchmark -i input.mkv -f null -

# FFmpeg demux benchmark
ffmpeg -benchmark -i input.mkv -c copy -f null -
```

## CI Integration

### GitHub Actions

Benchmarks run on:

- Pull requests (comparison against main)
- Main branch commits (historical tracking)
- Nightly (comprehensive suite)

### Performance Regression Detection

Automated alerts for:

- >10% slowdown in critical paths
- >20% slowdown in any benchmark
- >30% increase in variance

### Historical Tracking

- Results stored in `gh-pages` branch
- Trend graphs generated automatically
- Per-commit performance data

## Optimization Guidelines

### Container Operations

- Minimize allocations during parsing
- Use zero-copy when possible
- Batch I/O operations
- Optimize hot loops identified by profiling

### Codec Operations

- SIMD optimization for transforms
- Parallel tile processing
- Efficient entropy coding
- Cache-friendly data structures

### Audio Processing

- Vectorize sample processing
- Use in-place operations
- Minimize format conversions
- Optimize DSP kernels

### Computer Vision

- SIMD for pixel operations
- Parallel processing for independent operations
- GPU acceleration for ML inference
- Efficient memory access patterns

## Adding New Benchmarks

### 1. Create Benchmark Function

```rust
fn my_new_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("my_operation");

    // Set throughput metric
    group.throughput(Throughput::Bytes(data_size));

    // Add benchmark
    group.bench_function("operation_name", |b| {
        b.iter(|| {
            // Operation to benchmark
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
    my_new_benchmark,  // Add here
);
```

### 3. Document Expected Performance

Update this file with:
- Target performance metrics
- Comparison with FFmpeg/alternatives
- Rationale for targets

## Profiling Integration

For detailed profiling:

```bash
# Using perf
cargo bench --manifest-path benches/Cargo.toml --bench codec_bench -- --profile-time=5

# Using flamegraph
cargo flamegraph --bench codec_bench

# Using cachegrind
valgrind --tool=cachegrind --cache-sim=yes target/release/deps/codec_bench-*
```

## Known Limitations

### Current Limitations

1. **Synthetic Data**: Benchmarks use synthetic data, not real media files
2. **Single-threaded**: Most benchmarks are single-threaded
3. **No GPU**: Computer vision benchmarks run on CPU only
4. **No Network**: Network streaming benchmarks not yet implemented

### Future Improvements

- [ ] Add real-world media file benchmarks
- [ ] Multi-threaded benchmark variants
- [ ] GPU acceleration benchmarks
- [ ] Network protocol benchmarks
- [ ] End-to-end pipeline benchmarks
- [ ] Memory allocation profiling
- [ ] Cache miss analysis

## Contributing

When adding benchmarks:

1. Use appropriate warm-up and measurement times
2. Add throughput metrics where applicable
3. Test with multiple data sizes
4. Document expected performance
5. Compare with reference implementations
6. Ensure deterministic results

## References

- [Criterion.rs User Guide](https://bheisler.github.io/criterion.rs/book/)
- [FFmpeg Benchmarking](https://trac.ffmpeg.org/wiki/Benchmarking)
- [Performance Analysis Tools](https://perf.wiki.kernel.org/)
