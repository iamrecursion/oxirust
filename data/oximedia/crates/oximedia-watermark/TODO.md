# oximedia-watermark TODO

## Current Status
- 40 source files implementing professional audio watermarking and steganography
- Key features: 6 watermarking algorithms (Spread Spectrum DSSS, Echo Hiding, Phase Coding, LSB Steganography, Patchwork, QIM), unified WatermarkEmbedder/WatermarkDetector API, psychoacoustic masking, Reed-Solomon error correction, blind detection, robustness testing, quality metrics (SNR, ODG)
- Modules: attacks, audio_watermark, batch_embed, bit_packing, capacity_calc, chain_of_custody, dct_watermark, detection_map, detector (BlindDetector, NonBlindDetector), echo, forensic, forensic_watermark, fragile, invisible_wm, key_schedule, lsb, media_watermark, metrics, patchwork, payload/payload_encoder, perceptual_hash, phase, psychoacoustic, qim, qr_watermark, robust, robustness, spatial_watermark, spread_spectrum, ss_audio_wm, steganography, visible/visible_watermark, watermark_database, watermark_robustness, wm_detect, wm_strength
- Dependencies: oximedia-core, oximedia-audio, rustfft, rand, reed-solomon-erasure

## Enhancements
- [x] Replace `rustfft` with OxiFFT per COOLJAPAN ecosystem policy
- [x] Replace `rand` usage with scirs2-core random facilities per SCIRS2 policy
- [x] Improve `psychoacoustic` masking model with Bark scale critical band analysis for more accurate human hearing modeling (`bark_masking.rs`)
- [x] Add multi-channel watermarking support to `WatermarkEmbedder` (stereo/5.1/7.1 with Independent/Distributed/MidOnly/Complementary/Selective strategies) (`multichannel.rs`)
- [x] Extend `spread_spectrum` with Gold code sequences for better cross-correlation properties (`gold_code.rs`)
- [x] Add configurable Reed-Solomon parameters in `payload_encoder` (implemented 2026-05-15: `ReedSolomonConfig { n, k }` with `validate()`, `with_rs_config()` builder on `PayloadEncoder`; `RsConfigError` variants for k>=n, n>255, k=0)
- [x] Improve `dct_watermark` with adaptive coefficient selection based on local signal energy (implemented 2026-05-15: `AdaptiveDctSelector::select_coefficients()` sorts mid-freq zig-zag positions 5-42 by energy; `AdaptiveDctEmbedder` wires it in)
- [x] Add watermark strength auto-tuning in `wm_strength` that maximizes robustness while staying below perceptual threshold (implemented 2026-05-15: `WatermarkStrengthTuner::find_optimal_strength()` with 30-iteration binary search; `compute_psnr_f32()` helper; `WatermarkEmbedder` trait)

## New Features
- [x] Implement `video_watermark` module for spatial-domain video frame watermarking (DCT-based per-frame embedding with spatial and frequency modes) (`video_watermark.rs`)
- [x] Add `fingerprint_watermark` module combining `perceptual_hash` with watermark for dual content identification (verified 2026-05-16; src/fingerprint_watermark.rs:550 lines)
- [x] Implement `realtime_embedder` for streaming/live audio watermarking with frame-by-frame processing and state persistence (verified 2026-05-16; src/realtime_embedder.rs:744 lines)
- [x] Add `watermark_comparator` module for comparing extracted watermarks against database with fuzzy matching (verified 2026-05-16; src/watermark_comparator.rs:519 lines)
- [x] Implement `multi_layer_watermark` for embedding multiple independent watermarks (owner + distributor + session) in same audio (verified 2026-05-16; src/multi_layer_watermark.rs:523 lines)
- [x] Add `temporal_watermark` module that encodes data across time (frame sequence) rather than within single frames (verified 2026-05-16; src/temporal_watermark.rs:471 lines)
- [x] Implement `watermark_analyzer` CLI-style module that reports embedded watermark metadata, strength, and degradation level (verified 2026-05-16; src/watermark_analyzer.rs:613 lines)
- [x] Add `image_watermark` module extending spatial_watermark with DWT-based robust image watermarking (verified 2026-05-16; src/image_watermark.rs:536 lines)

## Performance
- [x] Optimize `spread_spectrum` FFT-based embedding with in-place transforms to halve memory allocation (implemented 2026-05-15: `InPlaceFftEmbedder` pre-allocates `Plan<f32>` + two `Vec<Complex<f32>>` scratch buffers; reuses across `embed_in_place()` calls)
- [x] Add batch FFT processing in `phase` embedder to amortize FFT setup across multiple frames (implemented 2026-05-15: `PhaseEmbedder::embed_batch()` pre-allocates shared `freq_buf` + `ifft_buf` and reuses for every frame in the batch)
- [x] Implement SIMD-optimized correlation computation in `spread_spectrum` detector for faster extraction (implemented 2026-05-15: `correlate_simd()` uses `scirs2_core::simd_aligned::simd_dot_aligned_f32()` (AVX2/NEON runtime dispatch via safe API); `correlate_scalar()` fallback)
- [x] Cache PN sequence generation in `spread_spectrum` (currently regenerated per embed/detect call) (planned 2026-06-01)
  - **Goal:** Eliminate redundant `generate_pn_sequence` calls by caching per-bit sequences up front.
  - **Design:** `src/spread_spectrum.rs` regenerates `generate_pn_sequence(chip_rate, key+bit_idx)` inside every loop iteration at :105/:178/:289/:342/:490. Sequences depend only on `chip_rate`+`key`+`bit_idx` (all known before the loop). Precompute a `Vec<Vec<f32>>` table keyed by `bit_idx` once per embed/detect invocation (or cache on the embedder/detector struct via `OnceCell`/`HashMap` keyed on the config). Pure-Rust, std only.
  - **Files:** `src/spread_spectrum.rs`, `TODO.md`.
  - **Tests:** cached-PN embed/detect bit-identical to per-call regeneration; embed→detect round-trip recovers the full payload; speed improvement on a large payload (assert it completes in < N ms as a smoke test).
  - **Risk:** cache must be invalidated if `chip_rate`/`key` changes between calls — scope to per-invocation table (simplest) or add config-keyed invalidation.
- [x] Optimize `echo` embedder overlap-add convolution with FFT-based fast convolution for long kernels (planned 2026-06-01)
  - **Goal:** Replace the O(n·k) direct time-domain echo convolution with an O(n log n) overlap-add using `oxifft`.
  - **Design:** `src/echo.rs:60` `EchoEmbedder::embed` does direct time-domain delay-and-add per sample over 512-sample blocks at :90-103. Build the echo impulse response (unit + decayed taps at delay_0/delay_1) and convolve via overlap-add using `oxifft::fft`/`ifft` (already a dep) for long kernels; keep the direct path for short kernels below a length threshold. `oxifft` is already in `Cargo.toml`.
  - **Files:** `src/echo.rs`, `TODO.md`.
  - **Tests:** FFT-conv output ≈ direct (within floating-point tolerance) on a long kernel; embed→detect round-trip recovers the payload after FFT path; short-kernel direct path still exercised.
  - **Risk:** Overlap-add block sizing and zero-padding correctness — assert bit-close vs direct conv on reference data; boundary blocks must handle remainder.
- [x] Profile `qim` quantizer and eliminate unnecessary f32<->i32 conversions in inner loop (planned 2026-06-01)
  - **Goal:** Reduce per-sample work in the QIM hot path by computing the quantizer index once per sample.
  - **Design:** `src/qim.rs:187` `quantize`, :208 `detect_bit`, :360/:395 each recompute `(value/delta).round()*delta` (and a second offset variant for dist_1) per sample. The TODO's "f32<->i32" wording is imprecise — the real optimization is eliminating the redundant divide+round: compute `k = (value/delta).round()` once, reuse for reconstruction (`k*delta`) and both `dist_0`/`dist_1` comparisons instead of recomputing the division+round twice. Pure-Rust.
  - **Files:** `src/qim.rs`, `TODO.md`.
  - **Tests:** optimized path bit-identical to current for a sweep of `value`/`delta` combinations, including at exact half-`delta` boundaries (tie-breaking must match); embed→detect round-trip unchanged.
  - **Risk:** tie-breaking at exact half-`delta` — assert output identity vs original on exhaustive value sweep.

## Testing
- [x] Add round-trip embed/detect tests for all 6 algorithms with various payload sizes (1 byte, 32 bytes, 256 bytes)
- [x] Test robustness of each algorithm against MP3 compression, resampling, low-pass filtering, and time stretching (`robustness_suite.rs`)
- [x] Add capacity limit tests verifying that embedding beyond capacity returns proper error
- [ ] Test `chain_of_custody` with multi-hop watermark tracking (embed A, embed B, detect both)
- [x] Add perceptual quality tests verifying SNR > 30dB and ODG > -1.0 for all algorithms at default strength
- [ ] Test `forensic_watermark` with simulated collusion attack (averaging multiple watermarked copies)
- [ ] Benchmark embed/detect throughput for each algorithm at 44.1kHz and 96kHz sample rates

## Documentation
- [ ] Document algorithm selection guide (robustness vs capacity vs imperceptibility tradeoffs)
- [ ] Add attack resistance matrix showing which algorithms survive which attacks
- [ ] Document `chain_of_custody` workflow for forensic leak tracing use case
