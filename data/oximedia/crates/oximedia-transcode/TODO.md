# oximedia-transcode TODO

## Current Status
- 48 source files covering high-level transcoding pipeline
- Key features: Transcoder builder API, ABR ladder generation, multi-pass encoding, parallel encoding, audio normalization (EBU R128/ATSC A/85), HW acceleration detection, progress tracking, job queuing, presets (YouTube, Vimeo, streaming, broadcast, archive)
- Modules: codec_config, codec_mapping, crf_optimizer, segment_encoder, two_pass, bitrate_control, scene_cut, rate_distortion, stage_graph, watermark_overlay, crop_scale, burn_subs, concat_transcode, and more
- Dependencies: oximedia-core, oximedia-codec, oximedia-container, oximedia-io, oximedia-graph, oximedia-metering, oximedia-subtitle

## Enhancements
- [x] Add stream copy mode to `Transcoder` (passthrough without re-encoding when codecs match)
- [x] Implement actual frame-level pipeline execution in `TranscodePipeline::execute()` connecting decoder/encoder through filter graph
- [x] Extend `codec_config` to support JPEG-XL still image encoding parameters alongside video codecs
- [x] Add per-codec tune presets to `codec_profile` (e.g., film, animation, grain for AV1/VP9)
- [x] Make `PresetConfig` support audio channel layout (mono, stereo, 5.1, 7.1) not just bitrate
- [x] Add HDR metadata passthrough/conversion (HDR10, HLG, Dolby Vision mapping) to the transcode pipeline (verified 2026-05-16; src/hdr_passthrough.rs:1947 lines, src/hdr_format.rs:495 HdrPassthroughConfig:217)
- [x] Implement actual content in `hw_accel::detect_available_hw_accel` for macOS VideoToolbox and Linux VAAPI
- [x] Add Dolby Atmos / spatial audio passthrough support in `audio_transcode` (verified 2026-05-16; src/spatial_audio_passthrough.rs:321 SpatialAudioPassthroughPlan, AtmosPassthrough)
- [x] Extend `concat_transcode` to handle mixed-resolution/mixed-codec input sources with automatic re-encoding (verified 2026-05-16; src/concat_transcode.rs:392 MixedConcatenator, test_mixed_concatenator_mixed_resolution:767)

## New Features
- [x] Add `TranscodeWatcher` for directory-based watch folder automation (detect new files, auto-transcode) (verified 2026-05-16; src/watcher.rs:278 lines)
- [x] Implement thumbnail strip / sprite sheet generation in `thumbnail` module (VTT-compatible for video players) (verified 2026-05-16; src/thumbnail_strip.rs:334 lines)
- [x] Add CMAF (Common Media Application Format) output support alongside existing HLS/DASH in `abr_ladder` (verified 2026-05-16; src/cmaf.rs:371 lines)
- [x] Implement `TranscodeProfile` import/export (JSON/YAML) for sharing encoding configurations (verified 2026-05-16; src/profile_io.rs:382 lines)
- [x] Add frame-accurate trim/cut support in the pipeline (start/end timecode, without full re-encode using keyframe snapping) (verified 2026-05-16; src/frame_trim.rs:474 lines)
- [x] Implement audio-only transcode mode (podcast optimization with loudness normalization) (verified 2026-05-16; src/audio_only.rs:525 lines)
- [x] Add `TranscodeBenchmark` utility to compare encoding speed/quality across different codec configurations (verified 2026-05-16; src/benchmark.rs:542 lines)

## Performance
- [x] Implement tile-based parallel encoding in `parallel` for AV1 (split frame into tiles for multi-core) (verified 2026-05-16; src/parallel.rs:447 Av1TileParallelEncoder, AV1 tile encode with rayon:483)
- [x] Add memory-mapped I/O option in pipeline for large file transcoding to reduce memory pressure (verified 2026-05-16; src/mmap_io.rs:690 lines)
- [x] Implement lookahead buffer in `crf_optimizer` for scene-adaptive CRF adjustment (verified 2026-05-16; src/lookahead_buffer.rs:715 lines)
- [x] Add segment-level parallelism to `segment_encoder` using rayon for encoding independent GOPs concurrently (verified 2026-05-16; src/segment_encoder.rs:420 encode_segments_parallel, par_iter:487)
- [x] Profile and optimize `bitrate_estimator` to use running statistics instead of full-pass analysis (implemented 2026-05-15: BitrateEstimator now embeds BitrateRunningAnalyzer using Welford online algorithm via running_stats.rs; new public module running_stats; tests: test_running_stats_matches_batch_computation, test_running_stats_incremental_update; 1203/1203 tests pass)

## Testing
- [x] Add integration tests for full transcode round-trip (encode then decode and verify frame checksums) (verified 2026-06-05; tests/transcode_roundtrip.rs:36 test_yuv_passthrough_roundtrip_preserves_bytes — 12 frames through ChecksumEncoder, payload==input; tests/transcode_roundtrip.rs:71 test_scale_then_checksum_deterministic_across_runs)
- [x] Add tests for `abr_ladder` verifying correct resolution/bitrate rungs for HLS and DASH profiles (verified 2026-06-05; tests/abr_ladder_spec.rs:55 test_hls_standard_ladder_passes_hls_spec maps AbrLadder::hls_standard→EncodeLadder→LadderSpec::Hls; tests/abr_ladder_spec.rs:76 test_aggressive_ladder_passes_dash_and_has_uhd_rung asserts ≥2160p & ≥15Mbps UHD rung + LadderSpec::Dash; tests/abr_ladder_spec.rs:111 ascending-order rejection)
- [x] Test `normalization` module with actual audio samples for EBU R128 compliance (verified 2026-06-05; tests/transcode_roundtrip.rs:151 test_normalization_r128_round_trip_within_half_lu — real oximedia_audio R128Meter measure→AudioNormalizer::calculate_gain min() contract→apply→re-measure within ±0.5 LU of −23; tests/transcode_roundtrip.rs:196 calculate_gain==min(loudness,peak))
- [x] Add fuzz testing for `validation` module with malformed input paths and configurations (verified 2026-06-05; tests/transcode_roundtrip.rs:280 test_validation_fuzz_table_never_panics — resolution/frame-rate/codec-container matches! table; tests/transcode_roundtrip.rs:229 test_validation_input_paths_against_temp_files non-empty/empty/missing temp files)
- [x] Test `parallel` encoding produces bit-identical output to single-threaded encoding (verified 2026-06-05; tests/parallel_determinism.rs:56 test_av1_tile_encode_byte_identical_across_thread_counts 4-thread vs 1-thread byte-identical; tests/parallel_determinism.rs:101 assemble order-independent; tests/parallel_determinism.rs:140 test_transcode_context_pool_size_invariant identical PassStats+output across rayon pool sizes)

## Documentation
- [ ] Add architecture diagram showing pipeline stage flow (demux -> decode -> filter -> encode -> mux)
- [ ] Document preset selection guide in `presets` module (which preset for which use case)
- [ ] Add inline examples for `TranscodeBuilder` showing common real-world workflows

## 0.1.8 Wave 4 follow-up (added 2026-05-29 by /ultra)

- [x] Register 24 orphan modules in oximedia-transcode (wire-ready ones get `pub mod`, stubs get annotation) (planned 2026-05-29)
  - **Goal:** 24 source files in `crates/oximedia-transcode/src/` are never compiled because they lack `pub mod` declarations in `lib.rs`. Enumerate, classify, and register all wire-ready ones.
  - **Design:** Identical pattern to oximedia-graphics orphan sweep: scan → classify (intentional-stub / compile-fix-needed / wire-ready) → register or annotate. Batch 4–6 at a time, validate after each batch. Smoke test per registered orphan.
  - **Files:** `crates/oximedia-transcode/src/lib.rs` (new `pub mod` declarations), `crates/oximedia-transcode/tests/orphan_smoke.rs` (new or extend), per-orphan annotation edits
  - **Tests:** Smoke test per registered orphan. Existing tests pass. Zero new clippy warnings.
  - **Risk:** Transcode orphans may depend on oximedia-codec symbols that were refactored. Annotation path for those.
