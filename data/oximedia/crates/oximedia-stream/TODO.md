# oximedia-stream TODO

## Current Status
- 8 source files (lib.rs + 7 modules): adaptive_pipeline, segment_manager, stream_health, scte35, multi_cdn, manifest_builder, stream_packager
- Adaptive bitrate streaming with BOLA-inspired ABR switching and quality ladder management
- Segment state machine with prefetch/eviction, QoE scoring and issue detection
- SCTE-35 splice information encoding/parsing/scheduling for ad insertion
- Multi-CDN failover routing with EWMA latency tracking
- HLS master/media playlist and DASH MPD generation
- Media unit accumulation and segment packaging with file writer
- Dependencies: thiserror, tracing, uuid, serde, serde_json, bytes

## Enhancements
- [x] Add buffer-based ABR algorithm variant in `adaptive_pipeline` (throughput-based in addition to BOLA) (verified 2026-05-11; adaptive_pipeline.rs:AbrAlgorithm::ThroughputBased + throughput_abr.rs:~74)
- [x] Implement segment prefetch depth configuration in `segment_manager` based on available bandwidth (verified 2026-05-11; segment_manager.rs:~285 adjust_prefetch_depth + prefetch_scheduler.rs:~99 PrefetchScheduler)
- [x] Extend `stream_health` QoE scoring to incorporate rebuffer ratio and startup delay metrics (verified 2026-05-11; stream_health.rs:~86-123 QoeScore with rebuffer_ratio + startup_delay_score fields)
- [x] Add SCTE-35 splice_null and bandwidth_reservation command support in `scte35` module (verified 2026-05-11; scte35.rs:~402-428 encode_splice_null + encode_bandwidth_reservation, roundtrip tests at ~1209)
- [x] Implement weighted round-robin routing strategy in `multi_cdn` alongside latency-based selection (verified 2026-05-11; multi_cdn.rs:~91 WeightedRoundRobin + ~291 select_weighted_round_robin)
- [x] Add EXT-X-DATERANGE and EXT-X-SKIP tags in `manifest_builder` for HLS low-latency (verified 2026-05-11; manifest_builder.rs:~51 DateRange struct + ~130 ExtXSkip struct, tests at ~802)
- [x] Extend `stream_packager` to support fMP4 (fragmented MP4) segment output alongside TS segments (implemented 2026-05-13; SegmentPackager::flush_internal now dispatches on PackagerConfig::enable_cmaf — when true, MediaUnits are wrapped via cmaf::CmafMuxer into fragmented MP4 bytes; new pack_segment_cmaf helper + tests/it_async_packager.rs test_packager_emits_fmp4_when_cmaf_enabled / test_packager_emits_ts_when_cmaf_disabled regression)
- [x] Add CDN health check with configurable probe interval and failure threshold in `multi_cdn`

## New Features
- [x] Implement `ll_hls` module for Low-Latency HLS with partial segments and preload hints (verified 2026-05-11; ll_hls.rs:~13 LlHlsConfig + ~58 PartialSegment + ~189 PreloadHint)
- [x] Add `ll_dash` module for Low-Latency DASH with chunked transfer encoding and CMAF (verified 2026-05-11; ll_dash.rs:~16 LlDashConfig + ~52 LlDashChunk + ~176 LlDashTimeline)
- [x] Implement `drm_signaling` module for DRM system signaling in manifests (Widevine, FairPlay, PlayReady content protection) (verified 2026-05-11; drm_signaling.rs:~13 DrmSystem enum + HLS EXT-X-KEY + DASH ContentProtection XML generation)
- [x] Add `thumbnail_track` module for generating I-frame only playlists and trick-play manifests (verified 2026-05-11; thumbnail_track.rs:~91 ThumbnailTrack + ~184 ThumbnailTrackBuilder)
- [x] Implement `multi_audio` module for managing multiple audio track variants (language, accessibility) (verified 2026-05-11; multi_audio.rs:~48 AudioTrack + ~125 AudioTrackManager with BCP-47 language + accessibility support)
- [x] Add `subtitle_track` module for WebVTT subtitle segment packaging and manifest integration (verified 2026-05-11; subtitle_track.rs:~106 SubtitleSegment + ~143 SubtitleTrack + ~328 SubtitleTrackManager)
- [x] Implement `stream_recorder` module for capturing live streams to VOD with DVR window management (verified 2026-05-11; stream_recorder.rs:~124 StreamRecorder + DvrWindow + dvr_recorder.rs:~83 DvrRecorder)
- [x] Add `stream_analytics` module for collecting viewer-side playback metrics (buffer health, quality switches, errors) (verified 2026-05-11; stream_analytics.rs:~85 StreamAnalytics + QualitySwitch events + QoE scoring)
- [x] Implement `srt_ingest` module for SRT protocol ingest as input to the streaming pipeline

## Performance
- [x] Use zero-copy segment writing in `stream_packager::FileSegmentWriter` with pre-allocated buffers (implemented 2026-05-16)
  - **Impl:** CmafChunk.data changed from Vec<u8> to bytes::Bytes; write_moof_mdat returns [Bytes; 2] (moof+mdat-header + payload refcount clone); write_cmaf_segment returns Vec<Bytes> for scatter-gather. collect_bytes() convenience wrapper added for callers needing a flat Vec<u8>. from_vec() ctor added. All 579 tests pass; 0 clippy warnings.
- [x] Implement async manifest generation in `manifest_builder` to avoid blocking the segment pipeline (implemented 2026-05-13; new `async` Cargo feature gates `write_master_playlist_async` / `write_media_playlist_async` / `write_dash_mpd_async` free functions in manifest_builder.rs and `LlHlsPlaylist::write_async` + `LlDashTimeline::write_async` on the LL builders; uses `tokio::io::AsyncWriteExt`; tests in tests/it_async_packager.rs verify byte-equality with sync output)
- [x] Add segment caching in `segment_manager` to serve repeated requests without disk I/O (verified 2026-05-11; segment_cache.rs:~75 SegmentCache LRU with byte-budget eviction)
- [x] Pool `MediaUnit` allocations in `stream_packager` to reduce allocation overhead during live streaming (implemented 2026-05-13; new `media_unit_pool` module exposes `MediaUnitPool` (bounded Mutex-guarded free list) + `PooledMediaUnit` RAII guard; payload `Vec` capacity is preserved across recycle; thread-safe via `Arc<MediaUnitPool>`; sequential-bounded and 8-thread concurrent tests in tests/it_async_packager.rs)
- [x] Implement parallel multi-CDN upload for segment distribution to reduce end-to-end latency

## Testing
- [x] Add SCTE-35 round-trip tests: encode splice_insert, parse back, verify all fields match (implemented 2026-05-16; 6 new tests: test_splice_insert_section_roundtrip, test_time_signal_section_roundtrip, test_multi_descriptor_section_roundtrip, test_section_re_encode_byte_identical, test_truncated_section_returns_err, test_section_with_zero_descriptors_roundtrip; all 575 tests pass, 0 clippy warnings)
- [x] Test `adaptive_pipeline` quality switching with simulated bandwidth fluctuation sequences (implemented 2026-05-14)
  - **Goal:** New tests `test_pipeline_fluctuation_hysteresis_prevents_oscillation` and `test_pipeline_fluctuation_strict_cooldown_cap` in `src/adaptive_pipeline.rs::tests`. Feed alternating high/low bandwidth samples; assert the pipeline does not flip-flop more than once per cooldown window.
  - **Design:** Use `force_tier` + `record_download` alternating 5000/500 kbps; mirror the `Instant::now().checked_sub(Duration::from_hours(1))` pattern from existing tests to bypass cooldown in one variant. Assert `switch_history` entry count.
  - **Files:** `crates/oximedia-stream/src/adaptive_pipeline.rs` (tests module).
  - **Tests:** Two new tests.
  - **Risk:** `last_switch_time` is private; use existing bypass pattern.
- [x] Verify `manifest_builder` output against HLS and DASH specification validators (implemented 2026-05-14)
  - **Goal:** New `src/validator.rs` module with `validate_hls(playlist: &str) -> Result<(), ValidationError>` and `validate_dash(mpd_xml: &str) -> Result<(), ValidationError>`. Integration test `tests/it_manifest_validators.rs`.
  - **Design:** HLS validator: checks #EXTM3U, #EXT-X-VERSION≥3, BANDWIDTH= attr, #EXT-X-TARGETDURATION, #EXT-X-MEDIA-SEQUENCE, #EXTINF per segment, version≥9 if EXT-X-SKIP present. DASH validator: quick-xml SAX walk for MPD/Period/AdaptationSet/Representation@bandwidth.
  - **Files:** `crates/oximedia-stream/src/validator.rs` (new), `crates/oximedia-stream/src/lib.rs` (module re-export), `crates/oximedia-stream/tests/it_manifest_validators.rs` (new).
  - **Tests:** ≥6 tests (HLS master valid, HLS media valid, HLS missing EXTM3U rejected, HLS SKIP w/o v9 rejected, DASH valid, DASH missing AdaptationSet rejected).
  - **Risk:** Use in-crate validator, not external tools.
- [x] Test `multi_cdn` failover by simulating provider failures and verifying automatic rerouting (verified 2026-05-11; multi_cdn.rs:~483 test_primary_skips_unavailable + ~429 test_record_error_marks_unavailable_after_threshold + ~458 test_record_success_restores_availability)
- [x] Add segment continuity tests ensuring monotonically increasing sequence numbers across playlist updates (implemented 2026-05-14)
  - **Goal:** Two new tests in `src/segment_manager.rs::tests`: multi-round `create_segment` calls asserting strictly monotonic `sequence_number` across rounds; eviction + continuation still monotonic.
  - **Design:** Round 1: create 5 segments, assert seqs [0..4]; Round 2: create 5 more, assert [5..9]; concat available_segments() asserts strictly increasing. Second test: evict first 3, create 5 more, assert no rewrap.
  - **Files:** `crates/oximedia-stream/src/segment_manager.rs` (tests module).
  - **Tests:** Two new tests.
  - **Risk:** `next_sequence` is private but API is correctly monotonic; test asserts the contract.
- [x] Test `stream_health` issue detection with simulated packet loss, jitter, and bitrate drops (implemented 2026-05-14)
  - **Goal:** Three new tests in `src/stream_health.rs::tests`: packet loss (error cascade), jitter (rapid rebuffer spikes), bitrate drop (low bitrate + low buffer report).
  - **Design:** Read actual HealthIssue enum variants first; map "packet loss" to error-heavy session, "jitter" to rebuffer spikes, "bitrate drop" to low-bitrate report(). Assert StreamHealthReport.issues contains appropriate variant or score falls below threshold.
  - **Files:** `crates/oximedia-stream/src/stream_health.rs` (tests module).
  - **Tests:** Three new tests.
  - **Risk:** HealthIssue may not have exact "packet loss" variant; implementation maps to closest existing variant or proposes adding one.
- [x] Add deterministic known-answer integration suite `tests/it_deterministic.rs` (Wave 29 / Slice 6, 2026-06-06; 13 tests, no production change)
  - **SCTE-35:** `encode_splice_null(0)` pinned to the exact 16-byte section `[0xFC,0xB0,0x0F, 0x00 x13]`; command-type byte (offset 9) == 0x00; encode → `parse_splice_info` → re-encode byte-identity for tiers {0x000, 0x123, 0x0FFF} with `SpliceCommand::Null` + empty descriptors + tier round-trip; distinct tiers → distinct bytes.
  - **HLS:** `HlsSkip{skipped_segments:5}.to_tag_line()` == `"#EXT-X-SKIP:SKIPPED-SEGMENTS=5"`; `build_media_playlist` version auto-coupling verified — WITH skip → `#EXT-X-VERSION:9` + skip line (never 3), WITHOUT skip → `#EXT-X-VERSION:3` + no skip line; playlist embeds the exact `to_tag_line()` string.
  - **Multi-CDN WRR:** weights `[1,2,1]` over 100 `WeightedRoundRobin` picks → exact histogram A=25/B=50/C=25; two independent routers emit identical 100-pick sequences (no RNG); pattern period == total-weight (4) with B twice / A,C once per period; equal weights `[1,1,1]` over 99 → 33 each.
  - **Files:** `crates/oximedia-stream/tests/it_deterministic.rs` (new). Gate: nextest 620/620, clippy --all-features --all-targets clean.

## Documentation
- [x] Add streaming pipeline architecture diagram showing data flow from ingest to CDN delivery (implemented 2026-05-14)
  - **Goal:** Top-of-file `//!` rustdoc in `src/lib.rs` with ASCII diagram: srt_ingest → packager → cmaf/fmp4 → segment_manager → manifest_builder → cdn_upload → multi_cdn. Show parallel ABR path and analytics fan-out.
  - **Files:** `crates/oximedia-stream/src/lib.rs`.
  - **Tests:** `cargo doc -p oximedia-stream --no-deps` clean.
  - **Risk:** None.
- [x] Document ABR algorithm parameters and tuning guidelines for different network conditions (implemented 2026-05-14)
  - **Goal:** "Tuning Guide" section in `src/adaptive_pipeline.rs` module rustdoc with markdown table: parameter × network-type (mobile/broadband/enterprise/live-event) with recommended values. Add "BOLA vs ThroughputBased" prose.
  - **Files:** `crates/oximedia-stream/src/adaptive_pipeline.rs` (module rustdoc).
  - **Tests:** `cargo doc` clean.
  - **Risk:** None.
- [x] Add example showing end-to-end live streaming setup: ingest → package → manifest → CDN (implemented 2026-05-14)
  - **Goal:** New `examples/live_streaming_pipeline.rs` (~150 lines). Synthesize frames, package via StreamPackager (CMAF), push through SegmentManager, emit manifests via build_master_playlist/build_media_playlist, print result. Runs in <1s with no real network.
  - **Design:** Reuse make_video_keyframe/make_video_delta pattern from tests/it_async_packager.rs. Gate on async feature via `[[example]]` required-features.
  - **Files:** `crates/oximedia-stream/examples/live_streaming_pipeline.rs` (new), `crates/oximedia-stream/Cargo.toml` (`[[example]]` block).
  - **Tests:** `cargo build --example live_streaming_pipeline -p oximedia-stream --all-features` exits 0. `cargo run ...` exits 0.
  - **Risk:** Synthetic frames must be valid enough for StreamPackager acceptance.
