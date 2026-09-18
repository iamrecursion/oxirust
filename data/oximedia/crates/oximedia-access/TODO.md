# oximedia-access TODO

## Current Status
- 29 top-level modules, ~68 source files total
- Audio description (generator, mixer, scripting, timing, templates), closed captions (generate, style, position, sync), sign language overlay, TTS/STT, transcripts, translation
- Visual enhancements (contrast, color blindness, high contrast, sizing)
- Audio enhancements (clarity, noise reduction, normalization)
- Speed control with pitch preservation
- Compliance checking (WCAG, Section 508, EBU)
- RBAC, session management, token handling, keyboard navigation, screen reader support
- Dependencies: oximedia-core, oximedia-audio, oximedia-subtitle, oximedia-graph

## Enhancements
- [x] Add real-time caption synchronization adjustment in `caption/sync` (user-adjustable offset) (verified 2026-05-16; src/caption/sync.rs:21 offset_ms, SyncAdjustment:18, with_offset:45)
- [x] Implement multi-speaker differentiation in `caption/generate` with speaker labels and color coding (verified 2026-05-16; src/caption/generate.rs:28 identify_speakers, SpeakerColor:35, SpeakerLabel:41)
- [x] Add emoji and symbol descriptions in `audio_desc/generator` for screen reader contexts (verified 2026-05-16; src/audio_desc/generator.rs:105 expand_emoji_and_symbols, EMOJI_TABLE:103, 188 fn body)
- [x] Implement adaptive font sizing in `caption/style` based on display resolution/distance (verified 2026-05-16; src/adaptive_font.rs:45 FontSizePolicy, measure_line:102, font size from resolution)
- [x] Add caption region collision avoidance in `caption/position` (prevent overlap with burned-in text) (verified 2026-05-16; src/caption_collision.rs:46 spatial_overlap_area, temporal_overlap_ms:57, collision detection)
- [x] Implement progressive enhancement in `visual/contrast` (multiple enhancement levels)
- [x] Add customizable color blind simulation modes in `color_blind` (protanopia, deuteranopia, tritanopia preview) (verified 2026-05-16; src/color_blind_sim.rs:169 protanopia/deuteranopia/tritanopia matrices, SimulationConfig:227)
- [x] Implement SSML prosody control in `tts/prosody` for emphasis and pacing
- [x] Add confidence scores to `stt/transcribe` output for quality assessment (verified 2026-05-16; src/stt/transcribe.rs:13 WordConfidence.confidence:21, quality assessment:46, is_uncertain:35)
- [x] Improve `compliance/wcag` checker to cover WCAG 2.2 criteria (focus appearance, dragging alternatives) (verified 2026-05-16; src/wcag_checker.rs:200 WCAG 2.1/2.2 checker, includes 2.2 criteria per module doc)

## New Features
- [x] Add live captioning pipeline: STT -> caption generation -> rendering in real-time
- [x] Implement automatic audio description from scene analysis (integrate with oximedia-analysis) (done 2026-06-24: new `src/audio_desc/scene_analysis.rs` — `SceneAudioDescriber` consumes `oximedia_analysis::AnalysisResults` {scenes + `frame_rate` + `content_classification`}, derives a `SceneTrigger` per shot boundary (cut/fade/dissolve/wipe + dominant content label + temporal/spatial dynamics), then places one cue per dialogue gap via the existing `TimingAnalyzer` with a reading-rate word budget so AD never overlaps speech; honest templated wording from labels only — richer descriptors (saliency/face/object/text) documented as follow-ups since `AnalysisResults` does not surface them. Also `gaps_from_silence()` consumes `oximedia_analysis::audio::AudioAnalysis` silence segments as a transcript-free quiet-region gap source. Added `oximedia-analysis` workspace dep (no cycle: nothing depends on `oximedia-access`). 24 new tests; build/test/clippy --all-features green)
- [x] Add haptic feedback description generation for touch-enabled devices (verified 2026-05-16; src/haptic.rs 553 lines)
- [x] Implement reading level assessment in `transcript` for plain language compliance
- [x] Add ASL/BSL/JSL-specific sign language grammar rules in `sign` module (verified 2026-05-16; src/sign/grammar.rs:1 ASL/BSL/JSL grammar rules, 497 lines)
- [x] Implement audio spatializer for accessibility (directional cues for visually impaired users) (verified 2026-05-16; src/spatial_accessibility.rs 637 lines)
- [x] Add cognitive load assessment for media content complexity
- [x] Implement auto-generated image alt-text from visual features in `media_alt_text` (verified 2026-05-16; src/auto_alt_text.rs 748 lines)
- [x] Add braille display output format support in `screen_reader` (render()+render_grid() added to BrailleDisplay; 2 tests: test_braille_render_basic, test_braille_render_grid)

## Performance
- [x] Cache TTS synthesis results in `tts/synthesize` for repeated text segments (TtsSynthesisCache: HashMap+VecDeque LRU, keyed (text,voice_id)→Vec<u8>; 3 tests: hit/miss/eviction)
- [x] Implement incremental STT processing in `stt` (process audio chunks, not entire files) (done — IncrementalSttProcessor at src/tts/transcribe.rs)
- [x] Add parallel compliance checking across multiple media files in `compliance` (rayon par_iter + BatchConfig::thread_pool_size; 4 tests: correct_counts, empty_report, size_1, deterministic)
- [x] Optimize `audio/clarity` enhancement pipeline with pre-computed filter coefficients
  - **Goal:** Replace `enhance()`/`enhance_speech()` clone-stubs with real DSP: speech-band biquad peaking boost + downward DRC for `enhance`, 4th-order Butterworth band-pass (300–3400 Hz) for `enhance_speech`. Wire existing dead-code helpers (`calculate_snr`, `speech_clarity_index`, `estimate_sti`) into post-enhancement metric. (Wave 20 Slice A, 2026-06-02)
  - **Design:** RBJ biquad Direct-Form-I peaking filter; DRC with envelope follower (attack/release); `enhance` = DRC → peaking boost → soft-clip guard; `enhance_speech` = cascaded biquads 300–3400 Hz. Remove `#[allow(dead_code)]` once helpers are called.
  - **Files:** `src/audio/clarity.rs`, `TODO.md`
  - **Tests:** DC/silence in → silence out; 2 kHz tone boosted vs 200 Hz; compressor reduces loud transient; `enhance_speech` attenuates 50 Hz rumble; sample count preserved; bounded output energy.
- [x] Use SIMD for contrast enhancement calculations in `visual/contrast`
  - **Goal:** Add AVX2/NEON/scalar SIMD to the per-pixel enhancement loop in `src/visual/contrast.rs` (currently no SIMD). (Wave 20 Slice A, 2026-06-02)
  - **Design:** Vectorize gain/offset/gamma map via `is_x86_feature_detected!`/`cfg(target_arch)` dispatch; gamma via 256-entry LUT for bit-exactness; SIMD path ≡ scalar path within ≤1 LSB.
  - **Files:** `src/visual/contrast.rs`, `TODO.md`
  - **Tests:** SIMD == scalar within ≤1 LSB on random image.

## Testing
- [ ] Add end-to-end test: generate captions -> style -> render -> verify positioning
- [ ] Test `compliance/section508` checker against known-compliant and non-compliant media samples
- [ ] Add WCAG color contrast ratio validation tests for all `caption/style` presets
- [ ] Test `audio_desc/timing` gap detection with overlapping dialogue scenarios
- [ ] Test `translate/quality` scoring against human-evaluated reference translations
- [ ] Add integration test: STT -> transcript -> translate -> caption pipeline

## Documentation
- [ ] Add compliance checklist mapping WCAG 2.1 success criteria to module functions
- [ ] Document supported languages for `translate` and `stt` modules
- [ ] Add example workflow: ingest media -> check compliance -> generate accessibility assets -> re-check
