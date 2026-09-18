# oximedia-packager TODO

## Current Status
- 50+ source files/directories for adaptive streaming packaging
- Formats: HLS (TS, fMP4), DASH (CMAF), with both live and VOD modes
- Core: Packager, HlsPackager, DashPackager with builder patterns
- Features: bitrate ladder generation, segment creation, manifest generation (M3U8, MPD)
- Encryption: AES-128, SAMPLE-AES, CENC (optional feature-gated)
- Cloud: optional S3 upload support (aws-sdk-s3)
- Modules: segment/segment_index/segment_list/segment_timeline/segment_validator, manifest/manifest_builder/manifest_update
- Additional: low_latency, bandwidth_estimator, packaging_optimizer, playlist_generator, subtitle_track, timed_metadata, pssh, drm_info, isobmff_writer
- 793 tests passing (0 failures)

## Enhancements
- [x] Fix `package_both` in Packager::package_both -- uses `unwrap_or` on path conversion, should return error
- [x] Implement actual isobmff_writer for fMP4/CMAF segment creation (init segment + media segments)
      - ftyp+moov init segment with big-endian u32 box sizes and 4-byte type codes
      - moof+mdat media segments with sequence numbers and decode timestamps
      - BoxWriter primitive for correct size-prefix patching
      - write_init_segment() / write_media_segment() public API
- [x] Add segment duration validation in segment_validator -- verify segments are within 0.5-2x target duration
- [x] Implement manifest_update for live streaming -- incremental playlist update without full regeneration
- [x] Add keyframe alignment enforcement across all bitrate variants in the ladder
- [x] Improve encryption_info to support Widevine, FairPlay, and PlayReady DRM system IDs in PSSH boxes
- [x] Add content-aware segment boundary selection aligned to scene changes (use codec keyframe positions) (verified 2026-05-16; src/scene_segmenter.rs:201 SceneAlignedSegmenter, 1027 lines)
- [x] Implement variant_stream (noted in src but may need wiring) for proper multi-variant playlist generation (verified 2026-05-16; src/variant_stream.rs:66 VariantStream, MultiVariantSet:199)

## New Features
- [x] Add CMAF byte-range addressing for single-file segment storage (reduces file count on CDN)
      - CmafByteRangeFile: single-file container with init + media segments mapped by byte offset
      - ByteRangeSegmentIndex: lookup by sequence number
      - HLS EXT-X-BYTERANGE tag generation for single-file addressing
- [x] Implement HLS interstitial support (EXT-X-DATERANGE for ad insertion points)
      - HlsInterstitial / HlsInterstitialBuilder with Apple HLS interstitial spec support
      - X-ASSET-URI, X-ASSET-LIST, X-RESUME-OFFSET, X-RESTRICT attributes
      - InterstitialSchedule for managing multiple interstitials per playlist
      - DateRangeClass enum (AppleInterstitial, Scte35, Custom)
- [x] Add DASH event stream support for timed metadata (SCTE-35, ID3 equivalent)
- [x] Implement automatic bitrate ladder generation from source analysis (per-title encoding) (verified 2026-05-16; src/ladder.rs:451 SourceAnalysis, BitrateLadderGenerator:550)
- [x] Add thumbnail/trick-play track generation (I-frame only playlist for HLS, thumbnail adaptation set for DASH)
      - IFramePlaylist: EXT-X-I-FRAMES-ONLY playlists with EXT-X-BYTERANGE addressing
      - IFramePlaylistBuilder: fluent builder for constructing I-frame playlists
      - ThumbnailTrack: DASH thumbnail adaptation set generation with tile sprites
      - ThumbnailTile: tile grid configuration (columns x rows, spatial fragment URIs)
      - EXT-X-I-FRAME-STREAM-INF tag generation for multivariant playlists
- [x] Implement pre-roll/bumper injection at packaging time (insert intro segment before content) (verified 2026-05-16; src/pre_roll.rs:98 PreRollInjector, BumperConfig:59)
- [x] Add multi-audio track support with language/accessibility metadata in manifests (verified 2026-05-16; src/audio_track.rs:321 MultiAudioSet, AudioTrack:119, 816 lines)
- [x] Implement SSAI (Server-Side Ad Insertion) markers and manifest manipulation (verified 2026-05-16; src/ssai.rs:40 AdBreakType::PreRoll, 327 lines)

## Performance
- [x] Parallelize multi-variant packaging using rayon -- encode each bitrate ladder rung concurrently
      - ParallelPackager: rayon par_iter over PackagingTask list
      - PackagingTask: per-variant work unit with segments and timescale
      - VariantResult: per-variant output (init + media segments, metadata)
      - to_hls_multivariant() / to_dash_period() merge helpers
- [x] Implement streaming segment output -- write segments as they are produced rather than buffering entire file (verified 2026-05-31; src/streaming_output.rs: SegmentStream + SegmentSender channel-based async pipeline, write_segment_to_disk with tokio::fs, pre_allocate flag in SegmentStreamConfig; already fully implemented in 0.1.8 wave)
- [ ] Add async file I/O throughout packaging pipeline (currently using tokio::fs but verify full async path)
- [x] Cache parsed source file metadata to avoid re-demuxing when packaging to both HLS and DASH (verified 2026-05-31; src/metadata_cache.rs: MetadataCache with TTL-based eviction, capacity limit, evict_expired, get_or_insert_with via get+insert, hit_count tracking; already fully implemented in 0.1.8 wave)
- [x] Implement segment output file pre-allocation to reduce filesystem fragmentation (implemented 2026-05-31; src/streaming_output.rs: `pre_allocate_file(path, expected_bytes)` public fn using File::set_len + create_dir_all; 4 tests: creates correct size, zero size, truncates existing, creates parent dir)

## Testing
- [x] Test HLS M3U8 output against Apple's mediastreamvalidator or equivalent format checks (implemented 2026-05-31; src/manifest_builder.rs: test_hls_m3u8_required_tags_present — #EXTM3U/BANDWIDTH/RESOLUTION/URI checks; test_hls_stream_inf_followed_by_uri — two-line structure RFC 8216 §4.3.4.2) (expanded 2026-06-24; tests/structural_validation.rs: 7 HLS M3U8 structural tests — required tag ORDER (#EXTM3U→VERSION→TARGETDURATION→MEDIA-SEQUENCE→URI), TARGETDURATION ≥ max EXTINF, VOD ENDLIST present / live ENDLIST absent, STREAM-INF carries BANDWIDTH+CODECS+RESOLUTION, STREAM-INF immediately followed by a URI, MEDIA-SEQUENCE monotonic. REASON the box stays at "equivalent": these are in-repo structural/self-consistency checks — Apple's mediastreamvalidator (external tool) is NOT run.)
- [x] Verify DASH MPD output validates against MPD schema (ISO 23009-1) (implemented 2026-05-31; src/manifest_builder.rs: test_dash_mpd_required_elements_present — <MPD/xmlns/urn:mpeg:dash/AdaptationSet/Representation/width/height/<Period>; test_dash_mpd_audio_mime_type — audio/mp4 mimeType check) (expanded 2026-06-24; tests/structural_validation.rs: 5 DASH MPD structural tests — profiles= references a urn:mpeg:dash:profile: URN, mandatory <MPD>→<Period>→<AdaptationSet>→<Representation> hierarchy with balanced open/close tags, Representation carries id=+bandwidth=, SegmentTemplate carries timescale=, mediaPresentationDuration is ISO-8601 PT…S. REASON it remains structural-only: these are self-consistency checks — true conformance against the ISO 23009-1 XSD would need an external schema-validator dependency, which is NOT pulled in.)
- [x] Add test for bitrate ladder ordering in manifests (highest to lowest bandwidth) (implemented 2026-05-31; src/manifest_builder.rs: test_hls_bitrate_ladder_ordering_preserved — verifies insertion-order preservation for 3-rung ladder; test_dash_mpd_representation_order_preserved — same for DASH MPD) (expanded 2026-06-24; tests/structural_validation.rs: 8 bitrate-ladder ordering tests — BitrateLadderGenerator rungs strictly DESCENDING bitrate + no duplicate bitrates + pixel-count monotonic with bitrate (4K/1080p/720p sources); LadderPresets hls_1080p/dash_4k/mobile_optimized each strictly descending + duplicate-free + resolution monotonic; MasterPlaylistBuilder renders BANDWIDTH ASCENDING regardless of insertion order)
- [x] Test encryption: verify AES-128 encrypted segment can be decrypted with known key (verified: encryption.rs:test_aes128_encrypt_decrypt_roundtrip, cfg(feature="encryption"))
- [x] Add live packaging test: simulate continuous segment arrival and verify rolling playlist window (implemented 2026-06-24; tests/structural_validation.rs: test_live_packaging_rolling_window_all_invariants — 20 continuous segment arrivals into a ManifestUpdater with a 5-segment sliding window; asserts on EVERY step that window ≤ 5, #EXT-X-MEDIA-SEQUENCE non-decreasing, #EXT-X-TARGETDURATION non-decreasing (rises to ≥8 after an 8 s segment injected at step 10), all current-window URIs present + all evicted URIs absent; final media_sequence == 15 and segment_count == 5. test_hls_media_sequence_monotonically_non_decreasing additionally covers the rolling sequence number across 20 arrivals.)
- [x] Test CMAF segment structure: verify correct moof/mdat box ordering in fragmented MP4
      - BoxWriter tests: size prefix patching, nested boxes, big-endian encoding
      - InitConfig / write_init_segment: ftyp+moov structure, codec fourcc, timescale
      - MediaSample / write_media_segment: moof+mdat, sequence numbers, decode timestamps
      - Roundtrip: init + 2 media segments byte-level verification

## Documentation
- [ ] Document the packaging workflow: source -> demux -> segment -> encrypt -> manifest -> upload
- [ ] Add configuration guide for common deployment targets (Apple HLS spec, DASH-IF guidelines)
- [ ] Document DRM integration setup for Widevine/FairPlay/PlayReady with key server requirements
