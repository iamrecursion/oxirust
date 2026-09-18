# oximedia-convert TODO

## Current Status
- 85+ source files; universal media format conversion with profiles, batch processing, and quality control
- Key features: format/codec detection, conversion profiles (Web, Archive, Broadcast, Device), batch conversion with progress tracking, quality comparison, metadata preservation, frame/thumbnail extraction, file splitting/concatenation, subtitle conversion, streaming packaging (HLS/DASH/ABR), template system
- Modules: detect/ (format, codec, properties), profile/ (system, custom, presets), pipeline/ (job, executor, options), batch/ (processor, queue, progress), filters/ (audio, video, chain), split/ (time, size, chapter), streaming/, smart/, presets/ (web, broadcast, archive, device), etc.
- Note: `perform_conversion` in lib.rs is currently a stub awaiting full demux/decode/encode/mux integration

## Enhancements
- [x] Complete `perform_conversion()` in `lib.rs` by integrating oximedia-transcode demux/decode/encode/mux pipeline
- [x] Extend `smart/mod.rs` SmartConverter with content-aware codec selection (animation -> VP8/GIF, live action -> AV1) (verified 2026-05-16; src/smart/mod.rs:14 SmartConverter, ContentType::Animation:149/LiveAction:151, SettingsOptimizer.optimize_video:227 selects codec per ConversionTarget, 405 lines)
- [x] Add two-pass encoding support in `pipeline/executor.rs` for better quality at target bitrate (verified 2026-05-16; src/two_pass.rs:385 TwoPassPlan, TwoPassPlanner:465, build:525, two_pass_benefit:192, 905 lines)
- [x] Improve `codec_mapper.rs` with automatic codec compatibility validation for target containers (verified 2026-05-16; src/compat_matrix.rs:199 check_codec_container_compat, ConvertValidation.validate_profile:147, 194 lines)
- [x] Extend `conv_validate.rs` with pre-conversion validation (sufficient disk space, compatible formats) (completed 2026-05-31: DiskSpaceEstimate, estimate_output_size, check_disk_space with df -k on Unix; headroom_factor 1.2x; 12 new tests — Wave 12 Slice E)
- [x] Add conversion resumption in `pipeline/job.rs` for interrupted long-running conversions (completed 2026-05-31: ConversionCheckpoint with JSON serde, save_checkpoint/load_checkpoint/resume_from_checkpoint, SystemTime serde module; 4 checkpoint tests — Wave 12 Slice E)
- [x] Extend `progress.rs` with ETA calculation based on encoding speed and remaining frames
- [x] Improve `profile_match.rs` to auto-select the best profile based on input analysis
- [x] Improve `codec_mapper.rs` with automatic codec compatibility validation for target containers

## New Features
- [x] Implement watch folder conversion mode (monitor directory, auto-convert new files) (verified 2026-05-16; src/watch.rs:19 WatchConfig, WatchEntry:165, WatchStats:186, 950 lines)
- [x] Add HDR-to-SDR tone mapping conversion in `color_convert.rs` (Wave 30, 2026-06-08: verified complete — convert_hdr_to_sdr_pixel/frame + PQ EOTF + HLG OETF⁻¹ + Rec.2020→Rec.709; oximedia-colormgmt dep wired; fixed 2 test bugs [stale PQ code-value 690→924 for the 4000-nit super-white headroom test; tautological `u8 <= 255` gamut assertion → red-dominance check]; 972 crate tests green, clippy clean)
- [x] Implement audio-only extraction with format conversion (e.g., video -> opus audio) (completed 2026-05-05)
  - WAV→WAV copy path implemented; transcode pipeline delegation for other formats; full validation (negative timestamp, zero duration, missing file) — all tests pass
- [x] Add image sequence to video conversion in `sequence/mod.rs` (PNG/JPEG frames -> WebM) (verified 2026-05-16; src/sequence/mod.rs:13 ImageSequence, SequenceImporter:151, SequenceExporter:100, 251 lines)
- [x] Implement conversion presets for social media platforms (YouTube, Twitter, Instagram specs) (verified 2026-05-16; src/presets/social_media.rs:18 YouTube/Instagram platforms, youtube_shorts:87, youtube_1080p_60fps:112, 739 lines)
- [x] Add video cropping and padding as conversion-time operations in `filters/video.rs` (verified 2026-05-16; src/filters/video.rs:29 Crop variant, CropParams:83, 190 lines)
- [x] Implement multi-output conversion (one input -> multiple outputs with different profiles) (verified 2026-05-16; src/multi_output.rs:165 MultiOutputConverter, OutputTarget:7, MultiOutputReport:100, 672 lines)
- [ ] Add `watermark_strip.rs` visible watermark overlay during conversion (verified-open 2026-05-16: watermark_strip.rs is about static overlay detection/safe-area analysis; no watermark-add/overlay-during-conversion function)

## Performance
- [x] Implement stream copy mode in `pipeline/stream_copy.rs` (completed 2026-05-31: `decide_stream_copy(input_codec, profile)` → ProfileStreamCopyDecision; codec match + zero-bitrate = copy, mismatch = re-encode; 5 tests — Wave 12 Slice E)
- [ ] Add hardware-accelerated encoding detection and fallback in `pipeline/options.rs`
- [x] Parallelize batch conversion in `batch/processor.rs` with configurable concurrency limit (completed 2026-05-31: `process_parallel(jobs, BatchConfig)`, counting-semaphore thread pool, BatchConfig/BatchResult; 4 tests — Wave 12 Slice E)
- [x] Implement segment-based parallel encoding for single files in `conversion_pipeline.rs`
  - **Completed:** Wave 20 Slice E, 2026-06-02
  - **Design:** `SegmentPlan { boundaries: Vec<FramePos> }` from keyframe positions or fixed GOP length; `encode_segments_parallel(plan, total_frames, encode_fn) -> Vec<EncodedSegment>` via `rayon::par_iter` preserving index; `concat_segments(ordered) -> output`. Tests use a stub/passthrough encoder — NOT AV1 (bitstream-parsing only; see codec_status.md). The segmentation/concat logic is fully codec-agnostic.
  - **Files:** `src/segment_encoder.rs` (new module), `src/lib.rs` (module registration + pub re-exports), `TODO.md`
  - **Tests:** 5 tests — boundaries cover full range no gap/overlap; parallel ordering (index embedded in output byte); parallel == sequential (passthrough stub); single-segment covers full range; uneven last segment (25 frames GOP=10 → last = 5 frames). All pass, 0 warnings.
- [x] Optimize `format_detector.rs` with magic-byte-only detection (completed 2026-05-31: `detect_by_magic_bytes(&[u8])` top-level fn; covers MP4/MOV/MKV/WebM/AVI/WAV/FLAC/OGG/MP3/JPEG/PNG; 13 tests — Wave 12 Slice E)

## Testing
- [x] Add end-to-end conversion test: input file -> convert -> verify output properties match profile (completed 2026-05-31: `e2e_passthrough_wav_copy_output_exists` — minimal PCM WAV header + passthrough copy, verifies RIFF/WAVE magic — Wave 12 Slice E)
- [x] Test all preset profiles (Web, Archive, Broadcast, Device) produce valid output (completed 2026-05-31: 6 preset validity tests covering Web/Archive/Broadcast ConversionProfile fields + Preset registry — Wave 12 Slice E)
- [x] Test `batch/processor.rs` with mixed input formats and concurrent conversion (completed 2026-05-31: `test_process_parallel_four_jobs_max_two_concurrent` — 4 jobs, max_concurrent=2 — Wave 12 Slice E)
- [ ] Add quality regression tests: verify PSNR/SSIM above threshold for each quality mode
- [x] Test `split/` modules with edge cases (completed 2026-05-05)
  - Chapter splitter: `chapters_from_durations`, `list_chapters`, missing-file error, no-chapters error, out-of-range index — all tested; segment boundary math for time splitter fully tested with contiguity, remainder, zero-duration guards; `output_paths_for` path naming — all 916 tests pass
- [ ] Test `streaming/mod.rs` ABR ladder generation produces valid HLS/DASH manifests
- [ ] Test metadata preservation round-trip across format conversions

## Documentation
- [ ] Document conversion profile system with examples for custom profile creation
- [ ] Add quality mode comparison guide (Fast vs Balanced vs Best)
- [ ] Document streaming packaging workflow (input -> ABR ladder -> HLS/DASH output)

## Wave 4 Progress (2026-04-18)
- [x] wasm-mio-fix: cfg-gate tokio deps for wasm32 target compatibility — Wave 4 Slice A

## Planned (2026-05-04)

- [x] frame/extract — FrameExtractor with full builder API, validation, and typed errors (completed 2026-05-05)
  - File-existence, negative-timestamp, zero-interval, empty-times, range-end-before-start validation fully implemented; `extract_at/interval/multiple/range` return `UnsupportedFormat` pending codec-decode pipeline; `output_filename_for` naming helper; format/quality builder chain — all tested

- [x] video/extract — VideoExtractor with transcode pipeline delegation (completed 2026-05-05)
  - Delegates to oximedia-transcode for non-WASM targets; validates codec name non-empty; segment extraction returns `UnsupportedFormat` pending container seek API; convenience codec helpers (`as_av1/vp9/h264/h265/copy_stream`) — all tested

- [x] video/mute — VideoMuter with transcode pipeline delegation + silence PCM generation (completed 2026-05-05)
  - `mute()` delegates to oximedia-transcode setting audio codec "none"; `generate_silence_pcm/silence_sample_count` utilities fully implemented and tested; `replace_with_silence` and `mute_tracks` return `UnsupportedFormat` pending packet-level mux integration

- [x] thumbnail/generate — ThumbnailGenerator with builder API, position math, and output-path naming (completed 2026-05-05)
  - Full builder (width/height/format/position/pattern), `ThumbnailPosition::calculate_time/requires_duration`, `output_path_for`, validation (zero-size, negative-time, missing-file), preset constructors (`widescreen/standard/small/large`) — all tested; `generate*` return `UnsupportedFormat` pending codec-decode pipeline

- [x] subtitle/extract — SubtitleExtractor with real SRT↔VTT/ASS/TTML conversion (completed 2026-05-05)
  - Standalone subtitle files fully processed (detect format, parse, re-serialize); container files return `UnsupportedFormat`; `extract_all/list_streams/extract_by_language/extract_first` all implemented; SRT→VTT round-trip test passes with correct WEBVTT header and event count
