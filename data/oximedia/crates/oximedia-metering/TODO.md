# oximedia-metering TODO

## Current Status
- 47 source files covering audio loudness, peak metering, video quality, spectrum analysis, and meter rendering
- Standards: ITU-R BS.1770-4, EBU R128, ATSC A/85, streaming platforms (Spotify, YouTube, Apple Music, Netflix, Amazon Prime)
- Key components: LoudnessMeter, PeakMeter, KSystemMeter, PhaseCorrelationMeter, SpectrumAnalyzer, LuminanceMeter, GamutMeter, QualityAnalyzer
- Wave 12 modules: crest_factor, k_weighted, meter_bridge
- Wave 15 modules: loudness_trend, noise_floor, stereo_balance

## Enhancements
- [x] Unify `true_peak.rs` and `truepeak.rs` -- resolved via `pub use truepeak as true_peak` re-export in lib.rs
- [x] Unify `correlation.rs` and `correlation_meter.rs` -- resolved via `pub use correlation as correlation_meter` re-export in lib.rs
- [x] Unify `phase.rs` and `phase_analysis.rs` -- resolved via `pub use phase as phase_analysis` re-export in lib.rs
- [x] Unify `peak.rs` and `peak_meter.rs` -- resolved via `pub use peak as peak_meter` re-export in lib.rs
- [x] Unify `dynamics.rs` and `dynamic_range_meter.rs` and `dr_meter.rs` -- resolved via `pub use dynamics as dynamic_range_meter` re-export in lib.rs
- [x] Add 8x oversampling option to TruePeakDetector for mastering-grade precision (verified 2026-05-16; src/truepeak.rs:65 OversampleMode::Mastering8x, new_mastering() fn:115)
- [x] Implement BS.2051 channel weights for NHK 22.2 immersive audio layout (verified 2026-05-16; src/bs2051_weights.rs:54 struct Bs2051Weights, compute_integrated_loudness_bs2051:147, 380 lines)
- [x] Add Tidal HiFi and Amazon Music HD loudness targets to the Standard enum
- [x] Extend LuminanceMeter to support HLG (Hybrid Log-Gamma) transfer function in addition to PQ/HDR10 (verified 2026-05-16; src/video_luminance.rs:20 Hlg variant, hlg_eotf fn:74)
- [x] Add temporal noise measurement (inter-frame noise) to video_quality module (TemporalNoiseMeasurement done)

## New Features
- [x] Add VMAF (Video Multi-Method Assessment Fusion) estimator using pure Rust feature extraction (verified 2026-05-16; src/vmaf_estimate.rs:483 lines, src/vmaf_features.rs)
- [x] Implement MS-SSIM (Multi-Scale SSIM) in video_quality for better perceptual quality scoring (verified 2026-05-16; src/ms_ssim.rs:323 lines)
- [x] Add Leq (equivalent continuous sound level) meter for environmental/broadcast compliance (verified 2026-05-16; src/leq.rs:843 lines)
- [x] Implement ITU-R BS.2132 loudness measurement for short-form content (<30s) (verified 2026-05-16; src/bs2132.rs:313 lines)
- [x] Add waveform meter data generator in `render.rs` for oscilloscope-style display (planned 2026-06-01)
  - **Goal:** Generate per-pixel min/max/rms envelope columns from interleaved audio samples for oscilloscope display.
  - **Design:** `src/render.rs` (401L) currently has BarMeter/Circular/Color rendering. Add `WaveformData { columns: Vec<WaveformColumn> }` where `WaveformColumn { min: f32, max: f32, rms: f32 }`. Add `generate_waveform_data(samples: &[f32], width: usize) -> WaveformData` that downsamples `samples` into `width` columns by computing min/max/rms over each segment; rayon is available for par_chunks. Reuse `BarMeterConfig`/`ScaleMark` data-struct style.
  - **Files:** `src/render.rs`, `TODO.md`.
  - **Tests:** each column's `min <= all samples in segment`, `max >= all samples`, rms in [0,1]; correct column count; single-sample per column edge case; empty samples returns empty columns. Keep `render.rs` < 2000 lines (currently 401).
  - **Risk:** downsample remainder at non-divisible `samples.len() / width` — handle the last partial segment.
- [x] Add vectorscope render data generator in `render.rs` for chroma/phase display (planned 2026-06-01)
  - **Goal:** Map (Cb,Cr)/(U,V) chroma pairs to polar X/Y bins for a vectorscope display with optional 75%-bar graticule.
  - **Design:** Add `VectorscopeData { bins: Vec<Vec<u32>>, width: usize, height: usize }` and `GraticulePoint { x: f32, y: f32, label: &'static str }`. Add `generate_vectorscope_data(cb_cr_pairs: &[(f32,f32)], width: usize, height: usize) -> VectorscopeData` that maps each (Cb,Cr) pair to polar bin coordinates (Cb→X, Cr→Y, with optional normalization to the 75% bar reference radius). Add `graticule_75pct_bar() -> Vec<GraticulePoint>` with the 8 standard color-bar reference points.
  - **Files:** `src/render.rs`, `TODO.md`.
  - **Tests:** pure Cb=0.5, Cr=0.0 maps to the expected X/Y quadrant; graticule 75%-bar points are at the correct polar angle for each color bar; bin accumulation for N identical pairs yields N in one bin; width×height grid allocation correct.
  - **Risk:** normalize Cb/Cr to [-1,1] before binning; integer bin index must clamp to grid bounds.

## Performance
- [x] Replace rustfft with OxiFFT in spectrum analysis (COOLJAPAN policy) -- Cargo.toml already uses `oxifft.workspace = true`, no rustfft present
- [x] Remove ndarray dependency -- Cargo.toml has no ndarray; video frames already use flat Vec<f64>
- [x] Add SIMD-accelerated K-weighting filter processing using portable_simd (when stable) (KWeightedFilter + k_weight_4ch_simd done)
- [x] Cache FFT plan in SpectrumAnalyzer to avoid repeated allocation per process() call (CachedSpectrumAnalyzer done)
- [x] Use rayon parallel iterators for per-channel true peak detection in multi-channel (>4ch) audio (done — process_parallel at src/truepeak.rs:155-247)

## Testing
- [x] Add reference signal tests: 997 Hz sine at -23 LUFS should measure exactly -23.0 LUFS (EBU R128 test signal) -- `test_ebu_r128_reference_signal` in lib.rs, ±0.5 LUFS tolerance
- [x] Add gating algorithm conformance tests per ITU-R BS.1770-4 Section 3
- [x] Test LRA calculation against EBU PLOUD reference values (`test_lra_ebu_tech3342_case1`..`case4` in `src/ebu_r128_impl.rs`, reproducing the four synthetic conformance signals from EBU Tech 3342 (Nov 2023) Table 1; this also fixed `loudness_range_lu()` to apply the Tech 3342 §3.1 −20 LU relative gate it was previously missing, without which case 4 measured ~30 LU instead of the published 15±1 LU)
- [x] Add round-trip test: normalize to target then re-meter should report target LUFS
- [x] Test all streaming platform targets (Spotify/YouTube/Apple/Netflix/Amazon) with known-loudness signals

## Documentation
- [x] Add architecture diagram showing LoudnessMeter internal pipeline (filter -> LKFS -> gating -> LRA) (ASCII diagram in crate-level `src/lib.rs` doc, "# Pipeline Architecture" section)
- [x] Document the relationship between clip_counter.rs, rms_envelope.rs, silence_detect.rs and the main metering pipeline ("# Complementary Metering Modules" section in `src/lib.rs`, plus a "Relationship to the main metering pipeline" note added to each of the three modules)
- [x] Add compliance testing guide with step-by-step EBU R128 verification procedure ("# Compliance Testing Guide (EBU R128 Verification)" section in `src/lib.rs`, referencing the actual conformance tests in `src/lib.rs`/`src/gating.rs`/`src/ebu_r128_impl.rs`)
