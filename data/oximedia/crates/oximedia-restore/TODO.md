# oximedia-restore TODO

## Current Status
- 38 modules (12 subdirectory modules) covering both audio restoration (click/hum/noise/hiss/crackle/azimuth/wow/flutter/DC/phase) and video restoration (deband, deflicker, film grain, color restore, scan line, telecine detect, upscale)
- Core types: RestoreChain with RestorationStep enum, mono and stereo processing pipelines
- Presets for vinyl restoration, tape restoration, broadcast cleanup
- Dependencies: oximedia-core, oximedia-audio, oxifft, thiserror

## Enhancements
- [x] Add per-step bypass toggle to `RestoreChain` so individual steps can be disabled without removing them (verified 2026-05-16; src/lib.rs:234 WrappedStep with enabled:bool, bypass toggle, disabled:256)
- [x] Implement `RestoreChain::process_multichannel` for surround sound (5.1/7.1) with channel-aware processing (verified 2026-05-16; src/lib.rs:268 SurroundChannel, MultichannelLayout:292, process_surround:651)
- [x] Add step reordering validation in `RestoreChain` (e.g., DC removal should precede click detection) (verified 2026-05-16; src/lib.rs:352 OrderingViolation, validate_order:452, step ordering constants:126)
- [x] Extend `noise::SpectralSubtraction` with adaptive noise floor estimation from initial silence detection (verified 2026-05-16; src/noise/subtract.rs:248 adaptive noise profile update; src/noise/profile.rs:216 silence_threshold detection)
- [x] Add real-time preview mode to `RestoreChain` processing fixed-size blocks with overlap-add (verified 2026-05-16; src/overlap_add.rs:105 OverlapAddProcessor, OverlapAdd:246, real-time block processing:1)
- [x] Pilot-tone detection for wow::WowFlutterCorrector (auto-detect reference frequency) (verified 2026-06-01; src/flutter_repair.rs `detect_pilot_tone` FFT-based 3–4 kHz peak with 10× prominence gate; 4 tests: pure-sine ±10 Hz, no-peak None, auto==manual, too-short None)
  - **Goal:** Auto-detect the bias-tone / pilot-tone frequency from the audio signal instead of requiring a fixed reference.
  - **Design:** `src/flutter_repair.rs:114-134` `SpeedVariationEstimator` uses a fixed `reference_freq` arg; `src/wow/detector.rs` has no pilot-tone logic. Add `detect_pilot_tone(samples: &[f32], sample_rate: u32) -> Option<f32>` using `oxifft::rfft` on the entire signal; find the strongest sustained narrow-band peak in the 3–4 kHz range (typical bias-tone / pilot frequencies). Wire into `WowFlutterCorrector::new` as an `AutoDetect` option. `oxifft` is already a dep.
  - **Files:** `src/flutter_repair.rs`, `src/wow/detector.rs`, `TODO.md`.
  - **Tests:** `detect_pilot_tone` on a synthetic 3150 Hz bias-tone + broadband noise returns ~3150 Hz (within 10 Hz); auto-detect mode produces same correction as manual 3150 Hz reference; signal without a sustained peak returns `None`.
  - **Risk:** Distinguish bias-tone (narrow, sustained) from transient spectral peaks — use a minimum-duration or minimum-prominence criterion.
- [x] Add severity/confidence output to `click::ClickDetector` for each detected event (verified 2026-05-16; src/click/detector.rs:68 Click.severity, .confidence:73, compute_severity_confidence:136)
- [x] Extend `hum::HumRemover` with automatic fundamental frequency detection (50Hz vs 60Hz) (verified 2026-05-16; src/hum/remover.rs:154 auto-detect 50/60 Hz hum, fallback to 50 Hz:158, FFT-based detection)

## New Features
- [x] Add a `breath_removal` module for podcast/voiceover restoration (detect and attenuate breaths) (verified 2026-05-16; src/breath_removal.rs 527 lines)
- [x] Implement `reverb_reduction` module using spectral dereverberation techniques (verified 2026-05-16; src/reverb_reduction.rs 315 lines)
- [x] Add `dynamic_eq` module for frequency-dependent compression/expansion
- [x] Implement `loudness_normalization` step (EBU R128 / ITU-R BS.1770) as a RestoreChain step (verified 2026-05-16; src/loudness_normalization.rs 511 lines)
- [x] Add `vinyl_surface_noise` module with adaptive surface noise profiling distinct from click/crackle
- [x] Implement `tape_dropout_repair` module for detecting and interpolating tape dropouts
- [x] Add `harmonic_reconstruct` module to rebuild missing harmonics in bandwidth-limited recordings (verified 2026-05-16; src/harmonic_reconstruct.rs 675 lines)
- [x] Implement `stereo_width` restoration step for collapsed or narrow stereo fields (verified 2026-05-16; src/stereo_width.rs 678 lines)
- [x] Add `restore_undo` with per-step rollback capability using stored intermediate buffers (verified 2026-05-16; src/restore_undo.rs 538 lines)

## Performance
- [x] SIMD-optimized batch-FFT path in noise::WienerFilter (verified 2026-06-01; src/noise/wiener.rs `apply_gain`/`apply_gain_avx2` AVX2 complex-multiply 4-wide; integrated into WienerFilter::process replacing polar reconstruction; 3 tests: SIMD==scalar ±1e-6, short-input no-panic, 1M-sample no-oom)
  - **Goal:** Add explicit-intrinsic AVX2/SSE2 path for the complex-multiply gain application step.
  - **Design:** `src/noise/wiener.rs:35-149` already has AVX2/SSE2 on the gain-computation loop but the FFT batch path (complex multiply `X[k] *= G[k]`) is scalar. Add an AVX2 path that processes pairs of `Complex<f32>` values in AVX2 lanes (4 pairs per 256-bit register); runtime-detect via `is_x86_feature_detected!("avx2")`. SSE2 fallback for non-AVX2. Scalar fallback always present. Add NEON path for aarch64.
  - **Files:** `src/noise/wiener.rs`, `TODO.md`.
  - **Tests:** AVX2 gain application == scalar output (bit-close, ±1e-6); Wiener stress: 50-sample input returns valid output; 10M-sample input processes without OOM panic.
  - **Risk:** Complex<f32> real/imag interleaving in oxifft — verify the memory layout before SIMD lane loading; use unaligned AVX2 loads if data is not 32-byte aligned.
- [x] Implement block-based processing in `RestoreChain::process` to reduce peak memory for long files (verified 2026-06-01; src/lib.rs `process_streaming` with OverlapAdd block streaming, 50% overlap, configurable block size)
- [x] Use rayon parallel iterators in `batch` restoration of multiple files (verified 2026-06-01; src/batch.rs:167,203,235 `into_par_iter` rayon parallel iteration)
- [x] Cache FFT plans in `spectral_repair` across consecutive process calls with same block size (verified 2026-06-01; src/spectral_repair.rs `fft_scratch: HashMap<usize, Vec<Complex<f32>>>` + `block_size_cache` fast-path, `process_block` method reuses scratch allocation)
- [x] Optimize `click::ClickRemover` interpolation to avoid full-buffer copies per click (verified 2026-06-01; src/click/remover.rs `remove_in_place(&mut [f32], &[Click])` with smoothstep Hermite interpolation, single small allocation for sorted click index)

## Testing
- [x] Add tests for `RestoreChain` with all step types combined in a realistic vinyl restoration pipeline (verified 2026-06-06; tests/vinyl_pipeline.rs full DC→click→crackle→hum→spectral-sub→hiss chain + archival-preset all-step run; surfaced & fixed 2 real DSP bugs — see below)
- [x] Test `process_stereo` with asymmetric corruption (clicks on left channel only) (verified 2026-06-06; tests/asymmetric_stereo.rs left-only clicks, right channel bit-preserved <1e-6, no cross-channel bleed)
- [x] Add golden-file tests comparing restored output against known-good reference for each restoration type (verified 2026-06-06; tests/golden_generated.rs computed-expectation single-click golden: residual<1e-3 in click window, untouched <1e-6 outside)
  - **Real bugs fixed at root (surfaced by these tests):**
    - `noise/subtract.rs` (`SpectralSubtraction` + `AdaptiveSpectralSubtraction`) & `hiss/remover.rs` (`HissRemover`): the STFT overlap-add reconstructed real signals at ~0.19× gain. Two compounding errors: (1) normalised by raw frame **count** instead of the window-overlap-squared sum (WOLA) — now divides by `Σ w[i]²` via new `utils::spectral::window_coefficients`; (2) only the lower half `[0, N/2]` of the Hermitian full-FFT spectrum was processed (subtraction zipped against the half-length noise profile; hiss gating ran over `[hp, N)` asymmetrically), corrupting the conjugate bins and **halving** the IFFT real output — now the gain is computed on `[0, N/2]` and **mirrored onto `N-k`** to preserve Hermitian symmetry. A zero/empty-profile pass-through is now unity-gain (interior ratio 1.000).
    - `wow/corrector.rs` (`WowFlutterCorrector`): a noisy/aperiodic lead-in poisoned the mean-pitch-lag estimate (white-noise frames reported spurious short lags), driving the program resampling rate far from unity and time-warping a steady tone to corr≈0 (buffer expanded 1.6×). Added a scale-invariant **normalised-autocorrelation periodicity gate** (`r = Σab/√(Σa²·Σb²) ≥ 0.5`); aperiodic frames now get neutral rate 1.0 and are excluded from the mean-lag. WowFlutter on a pure/degraded tone is now ~identity (archival preset corr 0.86).
- [ ] Test `AzimuthCorrection` and `PhaseCorrection` skip behavior in mono mode
- [x] Add stress tests for `WienerFilter` with very short (<100 sample) and very long (>10M sample) inputs (Wave 30, 2026-06-08: extreme-length stress suite in `tests/wiener_filter_stress.rs` — short {1,2,3,64,100,256}<block pass-through, block-boundary {2047,2048,2049}, 10M one-shot path `#[ignore]` release-only; asserts finite/no-panic, no-amplify & no-annihilate energy bound [WOLA window-squared design ⇒ not ±5% RMS], O(N) one-shot peak memory by design, bit-exact determinism + reset; 7 tests, no bug surfaced)

## Documentation
- [ ] Document recommended step ordering for each preset type (vinyl, tape, broadcast)
- [ ] Add signal flow diagrams for the RestoreChain processing pipeline
- [ ] Document the difference between `noise` (broadband), `hiss` (high-frequency), and `crackle` (impulsive) removal
- [ ] Add parameter tuning guide for each restoration step with before/after spectrograms
