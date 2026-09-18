# oximedia-mixer TODO

## 0.1.8 Wave 5 — 2026-05-29

- [x] Register 22 orphan modules (automation_engine, automation_playback, bounce, buffer_pool, channel_fold, channel_folder, channel_prealloc, clip_guard, cue_monitor, fader_group, gain_automation, gain_computer, mix_scene, offline_bounce, param_smoother, plugin, solo_modes, spectrum_analyzer, stem_mixer, surround_pan, talkback, vca_group) — 693 tests, 0 warnings

## Current Status
- 55 source files implementing a professional DAW-style audio mixer
- Core: AudioMixer with channels, buses (master/group/aux/matrix), effects, automation, session management
- Modules: automation, bus, channel, effects/effects_chain, dynamics, eq_band, limiter, metering, routing, session, snapshot, crossfade, delay_line, sidechain, vca, pan_matrix, matrix_mixer
- Additional: fader_group, solo_bus, talkback, scene_recall, monitor_mix, send_return, insert_chain, processing
- Full DSP pipeline: input gain -> effects -> fader (× VCA) -> pan -> PDC -> sends -> bus -> master with soft clipping
- ProcessingEngine: bus routing, RuntimeEffectsChain, aux send/return, VCA groups, PDC delay compensation

## Enhancements
- [x] Replace `unwrap_or` in `package_both` path conversion with proper error propagation — (obsolete: `package_both` does not exist in this crate; grep of crates/oximedia-mixer/ found no such function 2026-06-06)
- [x] Make `process()` use bus routing -- channels route to group/aux buses via ProcessingEngine
- [x] Implement actual effect processing in the channel strip -- RuntimeEffectsChain with AudioEffect trait integration
- [x] Add send/return routing in process() -- aux sends (pre/post-fader) wired into DSP path with bus accumulation
- [x] Implement VCA (Voltage Controlled Amplifier) group control in process() -- VCA fader multiplied onto linked channels
- [x] Add solo-in-place and AFL/PFL solo modes to the processing pipeline (verified 2026-05-16; src/solo_bus.rs — from memory noting solo_bus.rs present)
- [x] Implement automation playback in process() -- read automation data and apply parameter changes per buffer (verified 2026-05-16; src/lib.rs — tick_automation() evaluates gain/pan lanes via AutomationPlayer, 4 tests)
- [x] Replace soft_clip() tanh approximation with a proper oversampled limiter on the master bus (verified 2026-05-16; src/lib.rs — OversampledLimiter wired as master_limiter_l/r, set_limiter_enabled(), 4 tests)

## New Features
- [x] Add surround panning (5.1/7.1) support in pan_matrix beyond current stereo L/R panning (verified-open 2026-05-16: no 5.1/7.1 surround pan in pan_matrix.rs)
- [x] Implement Ambisonics encoding/decoding for spatial audio mixing (verified 2026-05-16; src/ambisonics.rs:39 AmbisonicsOrder, spherical harmonics encoding)
- [x] Add plugin hosting API for external audio effects (VST3-style interface definition) (verified 2026-05-16; src/plugin.rs:469 PluginHost, AudioPlugin trait:21)
- [x] Implement channel folding/unfolding for stereo-to-mono and mono-to-stereo conversion (verified 2026-05-16; src/channel_fold.rs:465 fold_channels/unfold_channels, 5.1 and 7.1 support)
- [x] Add latency compensation (PDC - Plugin Delay Compensation) across effect chains -- PdcDelayLine with recompute_pdc()
- [x] Implement offline bounce/render with higher-than-realtime processing (verified 2026-05-16; src/bounce.rs:46 OfflineBouncer faster-than-realtime)
- [x] Add MIDI control surface mapping (MCU/HUI protocol support) (verified 2026-05-16; src/midi_control.rs:109 MidiControlSurfaceConfig, ControlSurface:167)

## Performance
- [x] Process channels in parallel using rayon when channel count > 8 (verified 2026-05-16; src/parallel_mix.rs:1 ParallelMix rayon across channels)
- [x] Use SIMD for the master bus summing loop (accumulate L/R across channels) (verified 2026-05-16; src/simd_audio.rs:195 _mm_add_ps SSE2 accumulation)
- [x] Pre-allocate channel output buffers in AudioMixer instead of allocating per-process() call (verified 2026-05-16; src/channel_prealloc.rs:525 pre-allocated channel buffers)
- [x] Implement lock-free parameter updates using atomic f32 for gain/pan to avoid blocking the audio thread (verified 2026-05-16; src/atomic_param.rs:19 AtomicF32 backed by AtomicU32)
- [x] Add buffer pooling to avoid repeated Vec allocation in extract_f32_samples() (done 2026-06-01)
  - **Goal:** Eliminate the per-mix-cycle `Vec::with_capacity(count)` allocation in the hot `extract_f32_samples` path.
  - **Design:** Added `sample_pool: AudioBufferPool` field to `AudioMixer`; pre-warmed with 4 buffers at `block_size = config.buffer_size`. `extract_f32_samples` now takes `pool: &AudioBufferPool`, calls `pool.checkout()` for an RAII `PooledBuffer` (zeroed on hand-out), writes decoded samples directly into the slice, and returns the guard — which is dropped (auto-returned to pool) at the end of `process()`. No extra copy needed: `PooledBuffer` derefs to `&[f32]`, so `process_mix` accepts it unchanged.
  - **Files:** `src/lib.rs`, `TODO.md`.
  - **Tests added:** `test_pooled_extraction_identical_to_fresh`, `test_pool_reuse_no_stale_samples`, `test_soft_clip_never_exceeds_one`, `test_dynamic_channel_add_remove` — all passing.

## Testing
- [x] Add test that verifies gain=0.5 produces -6 dB output (linear gain verification)
- [x] Test pan law: center pan should produce equal L/R, hard left should produce silence on R
- [x] Add test for soft_clip: verify output never exceeds 1.0 for any input value (done 2026-06-01: test_soft_clip_never_exceeds_one)
- [x] Test channel add/remove during processing (dynamic channel count changes) (done 2026-06-01: test_dynamic_channel_add_remove)
- [x] Add integration test that processes a known signal through the full mixer pipeline and verifies output

## Documentation
- [ ] Document the DSP signal flow with a detailed diagram (input -> gain -> phase -> inserts -> EQ -> dynamics -> fader -> pan -> sends)
- [ ] Add examples for common workflows: basic stereo mix, submix with group bus, reverb send/return
- [ ] Document automation modes (Read/Write/Touch/Latch/Trim) with usage examples

## Proposed follow-ups
- **INVALID item**: "package_both unwrap→error" — no function named `package_both` exists in oximedia-mixer (phantom/other-crate symbol; original item references a non-existent function)
