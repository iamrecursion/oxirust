# oximedia-bench TODO

## Current Status
- 28 modules providing codec benchmarking: runner, metrics (PSNR/SSIM/VMAF), comparison, statistics, reporting (JSON/CSV/HTML)
- Additional modules: `baseline`, `cpu_profile`, `gpu_bench`, `io_bench`, `latency`, `memory`, `pipeline_bench`, `regression`, `regression_bench`, `regression_detect`, `scalability_bench`, `throughput`, `warmup_strategy`
- Preset configurations: quick, standard, comprehensive, quality_focus, speed_focus
- `BenchmarkSuite` orchestrates multi-codec, multi-sequence benchmark runs

## Enhancements
- [x] Fix `format_timestamp()` to use proper calendar calculation instead of approximate year/month/day
- [x] Add BD-Rate (Bjontegaard Delta) calculation to `comparison` module for codec efficiency comparison
- [x] Extend `metrics::QualityMetrics` with MS-SSIM (Multi-Scale SSIM) support
- [x] Add per-frame quality metric tracking in `metrics` for temporal quality analysis (verified 2026-05-16; src/metrics.rs:69 per-frame quality measurement push API)
- [x] Improve `stats::compute_statistics` with outlier detection and removal (IQR method) (verified 2026-05-16; src/stats.rs:61 IQR Tukey fences outlier removal, outliers_removed:46)
- [x] Add confidence intervals to `Statistics` for mean encoding/decoding FPS (done 2026-05-05)
  - **Goal:** Welch's t-test and bootstrap confidence intervals for benchmark stat significance
  - **Design:** t = (μ₁−μ₂)/sqrt(s₁²/n₁ + s₂²/n₂); Welch–Satterthwaite df; two-tailed p via incomplete-beta CDF; bootstrap CI percentile method (1000 resamples)
  - **Files:** `crates/oximedia-bench/src/stats.rs`
  - **Tests:** `tests/stats.rs` — Welch t; p value; 95% CI; bootstrap CI coverage; cross-validation
- [x] Extend `BenchmarkFilter` with date range filtering for historical result comparison (verified 2026-05-16; src/lib.rs:1147 timestamp-based date-range checks in BenchmarkFilter)
- [x] Add `CodecConfig` validation to reject invalid preset/bitrate/CQ combinations per codec (verified 2026-05-16; src/runner.rs:535 bitrate validation, src/container_bench.rs:215 validate fn)

## New Features
- [x] Implement `audio_bench` module for audio codec benchmarking (Opus, Vorbis, FLAC)
- [x] Add `container_bench` module benchmarking mux/demux overhead for WebM, MKV, OGG (verified 2026-05-16; src/container_bench.rs:922 lines)
- [x] Implement `ab_comparison` module generating side-by-side visual quality comparisons (verified 2026-05-16; src/ab_comparison.rs:974 lines)
- [x] Add `rate_distortion` module for automated RD-curve generation across quality levels
- [x] Implement `historical_db` module storing benchmark results in SQLite for trend analysis (verified 2026-05-16; src/historical_db.rs:1039 lines)
- [x] Add `system_fingerprint` module capturing exact hardware/OS/compiler info for reproducibility (verified 2026-05-16; src/system_fingerprint.rs:368 lines)
- [x] Implement `flamegraph_integration` module for automated CPU profiling during benchmarks (verified 2026-05-16; src/flamegraph_integration.rs:810 lines)
- [x] Add markdown report export format alongside existing JSON/CSV/HTML (verified 2026-05-16; src/markdown_report.rs:461 lines)

## Performance
- [x] Parallelize multi-codec benchmarks in `BenchmarkSuite::run_all()` using rayon thread pool
- [x] Implement incremental benchmarking: skip sequences whose results are cached and config unchanged
- [x] Add streaming CSV/JSON export in `BenchmarkResults` to avoid holding all results in memory (done 2026-06-05; declared pre-existing `streaming_export` module at `src/lib.rs:94` and added `BenchmarkResults::export_csv_streaming` at `src/lib.rs:520` / `export_json_streaming` at `src/lib.rs:538` delegating to `StreamingCsvWriter`/`StreamingJsonWriter`)
- [ ] Use memory-mapped I/O for reading test sequences in `runner`
- [x] Pre-allocate frame buffers in `runner` based on sequence resolution to avoid repeated allocation (done 2026-06-05; `LumaScratch` double-buffer at `src/runner.rs:995` + in-place `extract_luma_into` at `src/runner.rs:962`; `analyze_sequence` now ping-pongs two `Vec<f64>` buffers sized once from the sequence resolution, eliminating the per-frame `collect()` allocation in the ITU-T P.910 luma path while keeping output bit-identical)

## Testing
- [ ] Add test for `BenchmarkSuite::run_all()` with synthetic test sequences
- [ ] Test `comparison::CodecComparison::compare()` with known reference data
- [ ] Add regression test for `report::HtmlReport` output structure
- [ ] Test `BenchmarkFilter::apply()` correctly filters by all supported criteria combinations
- [ ] Add test verifying `BenchmarkPresets` configurations pass `BenchmarkConfigBuilder::build()` validation
- [ ] Test `duration_serde` round-trip for edge cases (zero, very large durations)

## Documentation
- [x] Document how to create custom test sequences for `sequences` module (done 2026-05-05)
  - **Goal:** Full y4m sequence parsing and frame iteration for benchmark pipelines
  - **Design:** Parse YUV4MPEG2 header (W/H/F/I/A/C fields per Xiph spec); frame separator `FRAME\n`; yield frames with PTS and planar data slices
  - **Files:** `crates/oximedia-bench/src/sequences.rs`
  - **Tests:** `tests/y4m_parse.rs` — round-trip 4-frame 64×64; reject malformed header; reject frame-size mismatch
- [ ] Add guide for interpreting benchmark results and quality metrics
- [ ] Document `regression_detect` module usage for CI-based performance regression detection
- [x] Implement `Y4mReader` / `Y4mWriter` full y4m sequence parsing and frame iteration in `sequences.rs` (done 2026-05-29)
  - **Goal:** Parse YUV4MPEG2 headers (W/H/F/I/A/C/X), iterate frames via FRAME separator, support C420jpeg/420mpeg2/420paldv/422/444 formats. `Y4mWriter` emits byte-identical output.
  - **Design:** `Y4mReader` wraps `BufRead`, parses header with nom or hand-rolled parser, yields `BenchFrame { y_plane, u_plane, v_plane, width, height, color_format }`. `Y4mWriter` mirrors the reader.
  - **Files:** `src/y4m_rw.rs` (new module), `src/sequences.rs` (re-exports)
  - **Tests:** 5-frame synthetic sequence round-trip byte-identical. Color format parsing coverage. Mono round-trip.
  - **Risk:** None — YUV4MPEG2 format is stable and well-documented.

## Planned (2026-05-04)

- [x] Add spatial/temporal/motion complexity metrics per ITU-T P.910 (done 2026-05-05)
  - **Goal:** Per-frame spatial (SI), temporal (TI), motion complexity per ITU-T P.910
  - **Design:** SI = stddev(Sobel(frame)); TI = stddev(frame[t]−frame[t-1]); motion = 16×16 block-SAD vs prev frame, magnitude histogram
  - **Files:** `crates/oximedia-bench/src/runner.rs`
  - **Tests:** `tests/complexity.rs` — constant-color → 0 complexity; noise-frame → positive SI; translating square → positive TI

- [x] Implement ITU-T P.910 SI/TI/motion complexity metrics in `itu_p910.rs` (done 2026-05-29)
  - **Goal:** `P910Metrics { si_max, ti_max, motion_mean }` computable from a sequence of `BenchFrame`s. SI = max over frames of `std_dev(Sobel(Y))`; TI = max over frame-pairs of `std_dev(Y_n - Y_{n-1})`; motion = mean per-frame block SAD (16×16 blocks vs previous frame).
  - **Design:** New `src/itu_p910.rs` with pure-scalar Sobel 3×3, frame-difference, block SAD. `P910Analyzer::feed_frame(&[u8])` + `P910Analyzer::metrics() -> P910Metrics`. No SciRS2 dep.
  - **Files:** `src/itu_p910.rs` (new)
  - **Tests:** Solid gray → SI=0; sharp edge → SI>50. Identical frames → TI=0; static frames → motion=0; max-contrast frames → motion>100.
  - **Risk:** None — straightforward ITU spec.

- [x] Implement Bjøntegaard Delta (BD-Rate, BD-PSNR) from 4-point RD curves (done 2026-05-05)
  - **Goal:** Bjøntegaard Delta (BD-Rate, BD-PSNR) from 4-point RD curves
  - **Design:** log-rate vs PSNR; 4×4 Vandermonde Gaussian elimination (exact cubic fit); analytic integral; rank-check for collinear → `Err(BdRateError::NoOverlap)`
  - **Files:** `crates/oximedia-bench/src/comparison.rs`
  - **Tests:** `tests/bd_rate.rs` — identical curves → 0%; uniformly shifted → matches shift; ill-conditioned guard

## Refinement proposals (added 2026-05-04 by /ultra)

- Note: Several UU (unmerged/conflict) files exist in the workspace (e.g. `crates/oximedia-accel/src/cpu_fallback.rs`, `crates/oximedia-farm/src/worker_health_check.rs`, `crates/oximedia-graphics/src/text.rs`, `crates/oximedia-lut/src/aces.rs`, `crates/oximedia-net/src/dash/client.rs`, `crates/oximedia-py/src/context_manager.rs`) — resolve merge conflicts before implementing benchmark features that depend on these crates.
