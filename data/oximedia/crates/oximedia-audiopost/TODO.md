# oximedia-audiopost TODO

## Current Status
- 41 modules covering ADR, Foley, sound design, mixing, restoration, stems, loudness, automation, and delivery
- Key subsystems: `adr`/`adr_manager`, `foley`/`foley_manager`, `mixing`/`mix_session`, `effects`, `restoration`, `stems`/`stem_export`, `loudness`/`loudness_session`, `surround`, `pipeline`
- [x] Dependency policy cleanup complete: `rustfft`/`ndarray`/`rand` never landed (oxifft/scirs2-core used); `rubato` removed 2026-07-08
- [x] Migrated `convert_sample_rate()` in `delivery.rs` from removed rubato 1.x `SincFixedIn` API to rubato 2.0 `Async::new_sinc` + `process_all_into_buffer` + `InterleavedOwned` API (2026-05-05)
- [x] Replaced rubato entirely: `convert_sample_rate()` now uses the Pure-Rust `oximedia_audio::Resampler` (windowed-sinc polyphase, group-delay compensated, `flush()` tail drain); 1 kHz 48k→44.1k tone-preservation + round-trip tests added in `delivery.rs` (2026-07-08)

## Enhancements
- [x] Replace `rustfft` dependency with OxiFFT per COOLJAPAN policy (already using `oxifft` in Cargo.toml)
- [x] Replace `ndarray` dependency with SciRS2-Core per SCIRS2 policy (ndarray not present; pure Vec<f32> used)
- [x] Replace `rand` dependency with SciRS2-Core RNG per SCIRS2 policy (rand not present; scirs2-core used)
- [x] Add surround sound upmixing algorithms (stereo-to-5.1, 5.1-to-7.1) in `surround` module (verified 2026-05-16; src/surround.rs:285 StereoTo51Upmixer:326)
- [x] Extend `loudness` module with ITU-R BS.1770-4 true-peak measurement (planned 2026-05-04)
  - **Goal:** FFT-based spectrum analysis, short-term LUFS, and true-peak detection
  - **Design:** FFT spectrum via OxiFFT (1024-pt). LUFS: wire to oximedia-audio::ebur128 (already exists). True-peak: 4× polyphase upsampling via Kaiser-windowed sinc (β=8.0, 33 taps), max of |upsampled|.
  - **Files:** `crates/oximedia-audiopost/src/metering.rs`
  - **Tests:** `tests/true_peak.rs` — 0 dBFS sine at 1 kHz with 0.5-sample phase offset → true peak ≥0 dBTP (within 0.1 dB)
  - **Risk:** FIR coefficients must be precomputed; runtime is 4× sample count
- [x] Add ARIB TR-B32 loudness standard support for Japanese broadcast in `loudness` (verified 2026-05-16; src/arib_loudness.rs:1 AribLoudnessAnalyzer, ARIB_TARGET_LKFS:38)
- [x] Implement convolution reverb engine using impulse response files in `reverb_profile` (planned 2026-05-04)
  - **Goal:** Wire oximedia-effects Reverb/Chorus/Distortion into a signal-flow chain
  - **Design:** Signal-flow builder: chain reverb→chorus→distortion with configurable parameters. Delegate to existing oximedia-effects implementations.
  - **Files:** `crates/oximedia-audiopost/src/sound_design.rs`
  - **Tests:** `tests/sound_design_chain.rs` — route signal through chain; output bounded, no NaN/Inf, energy plausible
  - **Risk:** Gain staging between effects; normalize output to avoid clipping
- [x] Add de-esser processor to `effects` module with adjustable frequency and threshold (`DeEsser::process` static method added)
- [x] Extend `restoration` with click/pop removal algorithm for vinyl-sourced audio (planned 2026-05-04)
  - **Goal:** Declick (transient detection + interpolation) and denoise (spectral subtraction + Wiener)
  - **Design:** Declick: detect samples where first-difference > N×MAD (N=8); replace click region with cubic-spline interpolation over ±50 surrounding samples. Denoise (Boll 1979): estimate noise PSD from quietest 5% of frames; subtract α·noisePSD (α=2.0), floor at β·noisePSD (β=0.05); Wiener gain = signalPSD/(signalPSD+noisePSD) post-processing. OxiFFT STFT: 1024-pt, 50% overlap, Hann window.
  - **Files:** `crates/oximedia-audiopost/src/restoration.rs`
  - **Tests:** `tests/declick.rs` — 1 kHz sine + 5 clicks → correlation >0.99 after declicker; `tests/spectral_subtraction.rs` — white noise + tone → SNR improves ≥6 dB
  - **Risk:** STFT edge effects — zero-pad and truncate to match-length; Wiener gain clamped to [0,1]
- [x] Add phase correlation meter to `metering` module for stereo/surround monitoring (planned 2026-05-04)
  - **Goal:** FFT-based spectrum analysis, short-term LUFS, and true-peak detection
  - **Design:** FFT spectrum via OxiFFT (1024-pt). LUFS: wire to oximedia-audio::ebur128 (already exists). True-peak: 4× polyphase upsampling via Kaiser-windowed sinc (β=8.0, 33 taps), max of |upsampled|.
  - **Files:** `crates/oximedia-audiopost/src/metering.rs`
  - **Tests:** `tests/true_peak.rs` — 0 dBFS sine at 1 kHz with 0.5-sample phase offset → true peak ≥0 dBTP (within 0.1 dB)
  - **Risk:** FIR coefficients must be precomputed; runtime is 4× sample count
- [x] Implement `restoration::declick` and `restoration::denoise` (Boll 1979 spectral subtraction + Wiener) (completed 2026-05-29)
  - **Goal:** `restoration::declick` removes impulsive noise (clicks/pops) via AR-LPC interpolation. `restoration::denoise` reduces broadband noise via spectral subtraction + Wiener post-filter.
  - **Design:** declick: sliding 1-ms energy ratio detection (3σ threshold), AR-LPC order-32 interpolation across corrupted span, configurable polarity preservation. denoise: STFT overlap-add (Hann window, 1024 FFT, 50% overlap), noise floor via temporal-mean + cross-bin median (tone-robust), `|S(k)| = max(|Y(k)| - α·|N(k)|, β·|Y(k)|)` (α=1.0–4.0 SNR-adaptive, β=0.002), Wiener post-filter `H(k) = S(k)²/(S(k)²+N(k)²)`. OxiFFT for all FFT ops. Per-frame DC removal prevents biased noise from leaking through denoiser.
  - **Files:** `src/restoration.rs` (added `ArLpcDeclickConfig`, `Declicker`, `levinson_durbin`, `DenoiseConfig`, `Denoiser`)
  - **Tests:** 873/873 pass; `declick_removes_impulses` (energy >99% preserved), `denoise_reduces_noise` (SNR >30 dB, >+10 dB improvement from 21.8 dB baseline)
  - **Deviation:** Spec called for first-N-frame bootstrap noise accumulation; that fails on continuous-tone test (no noise-only prefix). Used two-pass temporal-mean + cross-bin median estimator instead.

## New Features
- [x] Add `spectral_editor` module to lib.rs exports (declared at lib.rs line 95)
- [x] Add `clip_gain` module to lib.rs exports (declared at lib.rs line 68)
- [x] Add `phase_alignment` module to lib.rs exports (declared at lib.rs line 86)
- [x] Implement Dolby Atmos object-based audio layout support in `surround` (verified 2026-05-16; src/surround.rs:641 AtmosObject:649, AtmosBedLayout:729, Atmos session layout:759)
- [x] Add broadcast limiter with true-peak limiting in `pipeline` (planned 2026-05-04)
  - **Goal:** Pipeline orchestrator: declick→denoise→EQ→compressor→loudness→metering tap
  - **Design:** Sequential DSP chain with metering tap at each stage. Configurable per-stage bypass. Output normalized at -23 LUFS ±0.5 LU.
  - **Files:** `crates/oximedia-audiopost/src/workflow.rs`
  - **Tests:** `tests/workflow.rs` — pipeline yields normalized output at -23 LUFS within ±0.5 LU
  - **Risk:** Stage ordering matters; EQ before compressor, compressor before loudness
- [x] Implement sample-accurate crossfade engine for seamless take splicing in `take_manager` (verified 2026-05-16; src/crossfade_engine.rs:476 lines CrossfadeEngine, sample-accurate splicing)
- [x] Add M/S (Mid-Side) encoding/decoding processor in `effects` (verified 2026-05-16; src/effects.rs:676 MidSideProcessor)

## Performance
- [x] Add SIMD-accelerated sample processing for `mixing::ChannelStrip` gain/pan operations (verified 2026-05-29; src/mixing.rs:307 apply_simd)
- [x] Implement lock-free ring buffer for real-time audio routing in `bus_routing` (verified 2026-05-29; src/realtime/ring.rs:220 lines)
- [x] Add block-based FFT processing with overlap-add in `effects` to reduce per-sample overhead (verified 2026-05-29; src/dsp/block_fft.rs:280 lines)
- [x] Use SIMD for loudness gating calculation in `loudness` (K-weighted filter + gate) (verified 2026-05-31; src/loudness.rs — explicit AVX2+FMA / NEON intrinsics via `#[target_feature]` dispatch in `simd_mean_sq_avx2`, `simd_mean_sq_neon`, `simd_mean_sq_dispatch`)
- [x] Pre-compute and cache reverb impulse response FFTs in `reverb_profile` (verified 2026-05-29; src/reverb_profile.rs:444 ir_spectrum cached on load)

## Testing
- [x] Add integration test for complete ADR workflow: session create, cue add, record, sync (verified: take_manager.rs:857 test_adr_workflow_create_session_add_cues_record_sync)
- [x] Add property-based tests for `loudness` module against known EBU R128 reference signals (verified 2026-05-31; src/loudness.rs — proptest suite: prop_loudness_gain_shifts_lufs, prop_silence_is_neg_infinity, prop_appended_silence_invariance, test_997hz_0dbfs_sine_integrated_lufs)
- [x] Test `stem_export` round-trip: create stems, export, re-import, verify sample accuracy (verified 2026-05-31; src/stems.rs — StemSet::export/import + test_stem_export_import_roundtrip)
- [x] Add stress test for `mixing::MixingConsole` with 128+ channels (verified 2026-05-31; src/mixing.rs — test_mixing_console_128_channels_stress, serial-latency group)
- [x] Test `restoration` noise reduction with synthetic noise profiles (planned 2026-05-04)
  - **Goal:** Declick (transient detection + interpolation) and denoise (spectral subtraction + Wiener)
  - **Design:** Declick: detect samples where first-difference > N×MAD (N=8); replace click region with cubic-spline interpolation over ±50 surrounding samples. Denoise (Boll 1979): estimate noise PSD from quietest 5% of frames; subtract α·noisePSD (α=2.0), floor at β·noisePSD (β=0.05); Wiener gain = signalPSD/(signalPSD+noisePSD) post-processing. OxiFFT STFT: 1024-pt, 50% overlap, Hann window.
  - **Files:** `crates/oximedia-audiopost/src/restoration.rs`
  - **Tests:** `tests/declick.rs` — 1 kHz sine + 5 clicks → correlation >0.99 after declicker; `tests/spectral_subtraction.rs` — white noise + tone → SNR improves ≥6 dB
  - **Risk:** STFT edge effects — zero-pad and truncate to match-length; Wiener gain clamped to [0,1]

## Documentation
- [ ] Add architecture diagram showing signal flow through `pipeline` module
- [ ] Document supported loudness standards and compliance levels in `loudness` module
- [ ] Add examples for `broadcast_delivery` showing typical delivery spec configurations

## 0.1.8 Wave 4 follow-up (added 2026-05-29 by /ultra)

- [x] Split `restoration.rs` (1997 lines) via splitrs + add SPSC ring buffer, block-FFT overlap-add helper, SIMD ChannelStrip gain/pan (planned 2026-05-29)
  - **Goal:** `restoration.rs` is at 1997/2000 lines (3 lines margin). Split first, then add DSP infrastructure for real-time audio routing and efficient spectral processing.
  - **Design:**
    1. **splitrs** `src/restoration.rs` into: `restoration/mod.rs` (re-exports), `restoration/spectral.rs` (SpectralNoiseReducer/HissRemover/HumRemover/SpectralRepair/SpectralSubtractionConfig/spectral_subtract ~600 LoC), `restoration/declick.rs` (ClickRemover/VinylClickRemover/DeclickConfig/declick/ArLpcDeclickConfig/Declicker/levinson_durbin ~900 LoC), `restoration/stereo.rs` (PhaseCorrector/StereoEnhancer/Declipper ~270 LoC), `restoration/denoise.rs` (DenoiseConfig/Denoiser ~250 LoC). Re-export everything via `pub use spectral::*; pub use declick::*; pub use stereo::*; pub use denoise::*` from `mod.rs`.
    2. **SPSC ring buffer** `src/realtime/ring.rs`: `AudioRingBuffer { data: Vec<f32>, capacity: usize, head: AtomicUsize, tail: AtomicUsize }`. `push_slice/pop_slice` return count written/read. `AtomicUsize` with `Acquire/Release`. Capacity must be power-of-two. No alloc in hot path.
    3. **Block FFT** `src/dsp/block_fft.rs`: `BlockFftProcessor { fft_size, hop, window, overlap_buf }`. `process(&mut input, &mut output, spectral_fn: FnMut(&mut [Complex<f32>]))` using `oxifft::ComplexFft<f32>`. Hann/Hamming/Kaiser windows. Wire into `restoration/denoise.rs` Denoiser — replace inline STFT loop with `BlockFftProcessor::process`.
    4. **SIMD ChannelStrip** in `src/mixing.rs`: AVX2 path (`#[cfg(target_arch = "x86_64")]` + `is_x86_feature_detected!("avx2")`): 8-sample batches via `_mm256_loadu_ps` + `_mm256_mul_ps` + `_mm256_fmadd_ps`. NEON path (`#[cfg(target_arch = "aarch64")]`): `vmulq_f32` + `vfmaq_f32`. Scalar fallback always present.
  - **Files:** `crates/oximedia-audiopost/src/restoration/` (new dir, 5 files), `crates/oximedia-audiopost/src/realtime/ring.rs` (new), `crates/oximedia-audiopost/src/dsp/block_fft.rs` (new), `crates/oximedia-audiopost/src/mixing.rs` (SIMD additions)
  - **Tests:** `test_restoration_split_compiles` (existing tests pass unchanged), `test_ring_buffer_push_pop_correctness`, `test_ring_buffer_wraparound`, `test_ring_buffer_thread_safety` (1 producer + 1 consumer, 10 MB throughput, checksum), `test_block_fft_processor_identity` (pass-through < -100 dBFS RMS error), `test_block_fft_processor_lowpass`, `test_channel_strip_simd_matches_scalar` (±1e-6), `test_channel_strip_simd_speedup` (informational)
  - **Risk:** splitrs module path layouts must preserve public API. SIMD `is_x86_feature_detected!` dispatch only inside `#[cfg(target_arch = "x86_64")]` blocks.
