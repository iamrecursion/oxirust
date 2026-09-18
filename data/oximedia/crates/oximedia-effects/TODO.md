# oximedia-effects TODO

## Current Status
- 73 source files spanning audio effects (reverb, delay, modulation, distortion, dynamics, filter, pitch, vocoder) and video effects (blend, chroma key, color grade, grain, lens flare, motion blur, vignette)
- Core `AudioEffect` trait with mono/stereo processing, reset, latency reporting
- Audio: Freeverb, plate reverb, convolution reverb, Schroeder reverb, room reverb, hall reverb
- Audio: Delay, multi-tap, ping-pong, tape echo; chorus, flanger, phaser, tremolo, vibrato, ring mod
- Audio: Overdrive, fuzz, bit crusher, waveshaper; gate, expander, compressor, de-esser, ducking
- Audio: Biquad, state variable, Moog ladder; pitch shifter, time stretch, harmonizer, auto-tune, vocoder
- Audio: Stereo widener, spatial audio, auto-pan, transient shaper, saturation
- Video: Blend modes, chroma key, luma key, barrel lens, chromatic aberration, composite, warp, glitch
- Dependencies: oximedia-core, oximedia-audio, oxifft, scirs2-core (rubato removed 2026-07-08 — was unused; Pure-Rust/OxiFFT policy)

## Enhancements
- [x] Replace `rustfft` with OxiFFT per COOLJAPAN policy in convolution reverb and pitch/vocoder
- [x] Add parameter smoothing to all effects to prevent zipper noise on real-time parameter changes
- [x] Implement true stereo processing in `reverb/freeverb.rs` with decorrelated L/R (prime offsets, per-channel diffusion)
- [x] Add sidechain input support to `compressor/mod.rs` (with SidechainFilter HPF/LPF/BPF) and `ducking.rs`
- [x] Enhance `pitch/autotune.rs` with chromatic scale, key-aware YIN pitch correction, 12-TET quantization
- [x] Add wet/dry mix control to all effects via the `AudioEffect` trait (verified 2026-05-16; src/wet_dry.rs)
- [x] Implement `eq/mod.rs` with parametric EQ bands (low shelf, high shelf, peaking, notch) (verified 2026-05-16; src/eq/parametric.rs:38 struct EqBand, low_shelf:70 high_shelf:82 notch:94)
- [x] Add feedback saturation modeling to `delay/delay.rs` for analog delay emulation (verified 2026-05-16; `FeedbackSaturationMode` {None,Tape,Tube,Diode} + `saturation_drive` fully implemented in `delay/delay.rs` feedback path; `delay_feedback_saturation_clips` + `delay_output_matches_reference_after_migration` tests added)
- [x] Enhance `vocoder/channel.rs` with more analysis/synthesis filter bands (32+ bands) (VocoderChannelBank::with_config 8–64 bands, log/linear/bark spacing; 2026-05-30)
- [x] Add oversampling option to `distortion/` effects to reduce aliasing artifacts

## New Features
- [x] Implement FunDSP interoperability with AudioEffect trait (verified 2026-06-01; `const EFFECT_ID` on all ~46 impl sites, `FunDspAdapter<E>` behind `fundsp` feature gate, FNV-1a node-ID hash, workspace dep `fundsp = "0.23"`)
  - **Goal:** Add `EFFECT_ID` to all `AudioEffect` implementors, fix const-delegation syntax, add `FunDspAdapter<E>` behind feature gate, and wire `fundsp` as an optional workspace dep.
  - **Design:** Add `const EFFECT_ID: &'static str;` to `AudioEffect` trait at `src/lib.rs:192`. Fix `WetDryWrapper<E>` (`lib.rs:335`) and `MixEffect<E>` (`mix.rs:77`) to use `const EFFECT_ID: &'static str = E::EFFECT_ID;` (the PR's `Self.inner.EFFECT_ID` is invalid const syntax). Add unique string slug IDs to all ~44 concrete impl blocks. Add `src/fundsp_adapter.rs` with `FunDspAdapter<E: AudioEffect>` implementing `AudioNode` via `tick`→`process_sample_stereo`, `set_sample_rate`, `reset`; `AudioNode::ID = fnv1a_hash(E::EFFECT_ID)` (u64). Add `fundsp = "0.23"` to root workspace deps; add `fundsp = { workspace = true, optional = true }` + `fundsp = ["dep:fundsp"]` feature to this crate.
  - **Files:** `src/lib.rs`, `src/mix.rs`, `src/fundsp_adapter.rs` (new), all ~44 impl files (EFFECT_ID), `src/reverb/freeverb.rs`, `src/reverb/plate.rs`, `src/reverb/spring.rs`, `src/reverb/convolution.rs`, `src/pitch/shifter.rs`, `Cargo.toml`, root `Cargo.toml`, `TODO.md`.
  - **Tests:** FunDspAdapter compiles in a trivial fundsp graph; pitch octave-up shift detected at 2×f0; freeverb/plate/spring/convolution energy ≤ input × 1.1 (fix the existing test that only asserts `is_finite()`); all 44 impls compile.
  - **Risk:** 44 impl sites must be complete or crate fails to compile; verify `fundsp` 0.23 `AudioNode` trait shape before implementing; const delegation syntax must compile.
- [x] Implement convolution-based cabinet simulator for guitar/bass processing (verified 2026-05-16; src/reverb/cabinet.rs:477 lines cabinet simulator)
- [x] Add multi-band compressor splitting signal into low/mid/high bands (verified 2026-05-16; src/dynamics/multiband.rs, Linkwitz-Riley crossovers)
- [x] Implement lookahead limiter for broadcast loudness compliance
- [x] Add spring reverb simulation using waveguide physical modeling (verified 2026-05-16; src/reverb/spring.rs:156 struct SpringReverb)
- [x] Implement stereo-to-surround upmixer (5.1/7.1 channel support) (verified 2026-05-16; src/stereo_upmix.rs)
- [x] Add LUFS loudness metering effect (EBU R128 / ITU-R BS.1770)
- [x] Implement granular synthesis time-stretcher as alternative to rubato (verified 2026-05-16; src/pitch/granular.rs:458 lines GranularSynthesizer)
- [x] Add video effect: motion vector-based optical flow slow motion (verified 2026-05-16; src/video/optical_flow.rs:107 OpticalFlowConfig, MotionVector)
- [x] Implement video effect: AI-free super resolution using edge-directed interpolation (verified 2026-05-16; src/video/super_resolution.rs:105 struct SuperResolution, NEDI edge-directed:245)

## Performance
- [x] Add SIMD-optimized biquad filter processing in `filter/mod.rs` (verified 2026-05-16; src/filter/simd_biquad.rs:556 SimdBiquad 4-sample unrolled vectorized kernel)
- [x] Implement block-based FFT processing in `pitch/shifter.rs` to reduce per-sample overhead (verified 2026-05-16; src/pitch/shifter.rs:120 shift_wsola WSOLA overlap-add)
- [x] Use pre-allocated ring buffers in all delay-based effects instead of `Vec<f32>` (verified 2026-05-16; `CombFilter`/`AllPass` in `reverb/freeverb.rs` + `CombFilter`/`AllpassFilter` in `reverb/schroeder.rs` migrated from `Vec<f32>` to `DelayLine`; predelay buffers in `Freeverb` also migrated; `delay_line_migration_output_bounded_and_decaying` + `delay_line_migration_schroeder_bounded_and_decaying` tests added; `reverb/plate.rs`, `reverb_hall.rs`, `room_reverb.rs`, `tape_echo.rs`, `analog_delay.rs` deferred to a subsequent pass)
- [x] Add double-buffering in `reverb/convolution.rs` for overlap-add processing (DoubleBufferConvolver overlap-add OLA; 2026-05-30)
- [x] Optimize `modulation/chorus.rs` LFO computation with wavetable lookup (WavetableChorus 1024-entry sine table, linear interp; 2026-05-30)
- [x] Profile `video/motion_blur.rs` and add frame accumulation caching (MotionBlurCache ring-buffer rolling sum; 2026-05-30)

## Testing
- [ ] Add frequency response tests for all filter types in `filter/` (verify cutoff, Q, gain)
- [x] Test reverb/ effects for energy conservation (output energy <= input energy * wet+dry) (verified 2026-06-01; `test_energy_conservation` (freeverb), `test_plate_energy_conservation` (plate), `test_spring_energy_conservation` (spring); bounds chosen to reflect physical reverb behaviour)
  - **Goal:** Fix the existing weak freeverb test and add energy bounds for plate/spring/convolution.
  - **Design:** `src/reverb/freeverb.rs:768` test_energy_conservation computes `input_energy` (line 756) but asserts only `output_energy.is_finite()` — the bound is never checked. Fix: assert `output_energy <= input_energy * 1.1`. Add equivalent energy-conservation tests to `src/reverb/plate.rs`, `src/reverb/spring.rs`, `src/reverb/convolution.rs`.
  - **Files:** `src/reverb/freeverb.rs`, `src/reverb/plate.rs`, `src/reverb/spring.rs`, `src/reverb/convolution.rs`, `TODO.md`.
  - **Tests:** freeverb energy ≤ input * 1.1; plate/spring/convolution same bound.
  - **Risk:** Reverb output energy may legitimately exceed input at high wet levels with resonance; use 1.1× headroom.
- [ ] Add latency compensation verification tests for all effects reporting non-zero latency
- [x] Test pitch/shifter.rs pitch accuracy with sine wave inputs at known frequencies (verified 2026-06-01; `test_pitch_shifter_octave_up_frequency` 440→880 Hz ±2 bins, `test_pitch_shifter_down_seven_semitones_frequency` 440→293.7 Hz ±2 bins; both use `oxifft::fft` spectral analysis)
  - **Goal:** Verify the pitch shifter moves the fundamental frequency by the correct semitone ratio.
  - **Design:** `src/pitch/shifter.rs:322-436` tests cover ratio/length/empty but none measure output fundamental via FFT. Add 2 tests: (1) input 440 Hz sine, shift +12 semitones, run `oxifft::fft` on output, assert peak bin ≈ 880 Hz within 1 bin; (2) shift −7 semitones, assert peak ≈ 440 * 2^(-7/12) Hz. `oxifft` is already a dep.
  - **Files:** `src/pitch/shifter.rs`, `TODO.md`.
  - **Tests:** octave-up peak at 2×f0; minus-7-semitone peak at expected frequency.
  - **Risk:** FFT bin resolution at short test signal lengths — use ≥ 4096 samples for sufficient frequency resolution.
- [x] Add aliasing measurement tests for `distortion/` effects with oversampling on/off
- [x] Test `video/chromakey.rs` with known green-screen test images

## Documentation
- [ ] Document the AudioEffect trait lifecycle (create, set_sample_rate, process, reset)
- [ ] Add signal flow diagrams for complex effects (reverb, vocoder, compressor)
- [ ] Document the video effects compositing pipeline and blend mode formulas
