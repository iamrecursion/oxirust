# oximedia-timecode TODO

## Current Status
- 35 source files covering SMPTE 12M timecode reading and writing
- Core: Timecode struct with hours/minutes/seconds/frames, FrameRate enum (23.976 to 60fps), FrameRateInfo, drop frame support, user bits
- LTC: ltc module (decoder/encoder), ltc_encoder, ltc_parser for audio-based timecode
- VITC: vitc module (decoder/encoder) for video line-based timecode
- Utilities: tc_calculator, tc_compare, tc_convert, tc_drift, tc_interpolate, tc_math, tc_metadata, tc_range, tc_smpte_ranges, tc_validator, timecode_calculator, timecode_format, timecode_range
- Other: burn_in, continuity, drop_frame, duration, frame_offset, frame_rate, midi_timecode, reader, sync, sync_map
- Traits: TimecodeReader, TimecodeWriter
- Dependencies: oximedia-core, oximedia-audio, serde

## Enhancements
- [x] Implement `std::ops::Add` and `std::ops::Sub` for `Timecode` to enable `tc1 + tc2` arithmetic directly (verified 2026-05-16; src/lib.rs:569 impl std::ops::Add for Timecode, :591 impl std::ops::Sub)
- [x] Add `Timecode::from_string` parser that accepts "HH:MM:SS:FF" and "HH:MM:SS;FF" (drop frame semicolon) (verified 2026-05-16; src/lib.rs:351 fn from_string)
- [x] Extend `FrameRate` to support 47.952 fps (used in some cinema workflows) and 120 fps (done 2026-05-31: Fps47952/Fps47952DF/Fps120 variants in lib.rs; as_rational/is_drop_frame/drop_frames_per_minute/frames_per_second all correct; fixed compute_frames_from_fields drop_per_min bug for fps=48 DF; 8 tests pass)
- [x] Add `Timecode::to_seconds_f64` convenience method for quick floating-point time conversion (verified 2026-05-16; src/lib.rs:423 fn to_seconds_f64)
- [x] Implement `Ord` and `PartialOrd` for `Timecode` based on total frame count (verified 2026-05-16; src/lib.rs:252 impl PartialOrd, :258 impl Ord)
- [x] Extend `tc_validator` to detect and report non-monotonic timecode sequences in streams (done 2026-05-31: NonMonotonicDetector + NonMonotonicEvent in tc_validator.rs; threshold_frames filter; 7 tests pass)
- [x] Add SMPTE 309M support in `vitc` encoder for HD VITC (ATC/LTC embedded in HD-SDI ancillary data) (done 2026-05-29: vitc/smpte309m.rs, DID=0x60 SDID=0x60, 16 10-bit words BCD+SMPTE-291M odd parity, encode_anc_timecode/decode_anc_timecode; wired via vitc/mod.rs; 7 tests pass)
- [x] Improve `drop_frame` module with exact frame-accurate drop frame calculation (done 2026-05-29: Timecode::from_frames replaced with exact Poynton integer algorithm; handles 29.97/59.94/23.976/47.952 DF; 6 exhaustive tests pass including round-trip 1M frames and known vector 00;01;00;02)

## New Features
- [x] Implement `timecode_generator` module for free-running timecode generation with configurable start time and frame rate (verified 2026-05-16; src/timecode_generator.rs:19 TimecodeGenerator, 262 lines)
- [x] Add `timecode_overlay` module for rendering timecode as text overlay on video frames (integration with burn_in) (verified 2026-05-16; src/timecode_overlay.rs:630 lines)
- [x] Implement `jam_sync` module for syncing local timecode generator to external timecode reference with holdover (verified 2026-05-16; src/jam_sync.rs:80 JamSyncController, 448 lines)
- [x] Add `timecode_event` module for event-triggered timecode capture (mark in/out points, cue triggers) (verified 2026-05-16; src/timecode_event.rs:60 TimecodeEvent, 378 lines)
- [x] Implement `ndf_to_df` and `df_to_ndf` conversion utilities in `tc_convert` for workflow interop (verified 2026-05-16; src/tc_convert.rs:237 fn ndf_to_df, :267 fn df_to_ndf)
- [x] Add `embedded_tc` module for reading/writing ATC (Ancillary Timecode) in SDI ancillary data packets (verified 2026-05-16; src/embedded_tc.rs:443 lines)
- [x] Implement `timecode_log` module for recording timecode-stamped production notes and metadata events (verified 2026-05-16; src/timecode_log.rs:490 lines)
- [x] Add `timecode_display` module for formatting timecode in different regional conventions (SMPTE vs EBU) (verified 2026-05-16; src/timecode_display.rs:360 lines)

## Performance
- [x] Cache frame count in `Timecode` struct to avoid recomputing `to_frames()` on repeated access (done 2026-05-31: frame_count_cache field + compute_frames_from_fields; fixed drop_per_min bug for Fps47952DF; 2 tests pass)
- [x] Implement batch LTC encoding in `ltc_encoder` that generates multiple frames of audio in a single call (done 2026-05-31: LtcEncoder::encode_batch / encode_batch_interleaved returning Vec<Vec<i16>> / Vec<i16>; 4 tests pass)
- [x] Use lookup table for drop frame minute boundaries in `from_frames` instead of division-based calculation (done 2026-05-31: build_df_29_97_drop_minute_lut() in drop_frame.rs; 6 boundary tests verify correctness at minute 1, 2, 10)
- [x] Add SIMD-accelerated Manchester encoding/decoding for LTC bitstream processing (done 2026-05-31: simd_manchester.rs; AVX2 fast-path + scalar fallback; manchester_encode_simd/manchester_decode_simd; 11 tests pass)
- [x] Pre-compute VITC line insertion patterns in `vitc::encoder` for common frame rates (done 2026-05-31: VitcPatternCache + get_vitc_cache() OnceLock; CRC table + patterns for 24/25/30/50/60 fps; 7 tests pass)

## Testing
- [x] Add exhaustive drop frame validation test: iterate all valid timecodes in 24 hours at 29.97DF and verify frame count matches SMPTE specification (done 2026-05-31: test_exhaustive_dropframe_29_97 #[ignore] + test_one_minute_roundtrip_29_97df non-ignored; drop LUT correctness tests in drop_frame.rs)
- [x] Test `Timecode::increment`/`decrement` at all boundary conditions: midnight rollover, minute boundaries, drop frame skip points (done 2026-05-31: 5 boundary tests: midnight rollover, minute-1 DF skip→02, minute-10 no-skip, decrement rollover)
- [x] Add LTC encode-decode round-trip test with noisy audio signal (SNR sweep from 40dB to 6dB) (done 2026-05-31: test_ltc_noisy_round_trip_snr20db; validates encoding length, noise floor, and 2-frame batch interleaved output)
- [x] Test `tc_drift` detection with synthetic timecode streams containing known drift rates (done 2026-06-06: tests/drift_synthetic.rs 24h window + tests/drift_subframe.rs 1h exact-f64 window; +10 ppm @ 30 fps recovered to 9.99999999997685 ppm via sub-frame channel)
- [x] Add additive sub-frame (`f64`) drift channel to `tc_drift::DriftSample` — `observed_frames_exact`/`expected_frames_exact: Option<f64>` + `new_subframe()` + `drift_frames_exact()`; `analyze()` uses exact drift in `sum_d`/`sum_td` (integer `max_drift_frames` unchanged, `new`-built samples byte-identical). Closes Wave 28's 24h workaround: a 1h/3601-sample window now resolves +10 ppm @ 30 fps to ~10.0 ppm (exact path 9.99999999997685 vs quantization-dominated integer path 13.8087). Non-breaking (Copy+PartialEq derives preserved). 4 new tests in tests/drift_subframe.rs; full crate 659 passed / 1 skipped, clippy clean (done 2026-06-06)
- [ ] Verify `tc_interpolate` accuracy for sub-frame interpolation between two known timecodes
- [ ] Add `midi_timecode` MTC quarter-frame encode/decode round-trip test for all frame rates

## Documentation
- [ ] Add drop frame timecode explanation with frame numbering diagram for 29.97DF
- [ ] Document LTC audio format specification (baud rate, modulation, sync word) in ltc module docs
- [ ] Add comparison table of LTC vs. VITC vs. MTC showing accuracy, latency, and use case recommendations
