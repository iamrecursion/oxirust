# oximedia-clips TODO

## Current Status
- 33 modules providing clip management: database, subclips, grouping (bins/folders/collections), logging (keywords/markers/ratings/notes), takes, proxy association, smart collections, search, import/export
- Sub-directories: `clip/`, `database/`, `export/`, `group/`, `import/`, `logging/`, `marker/`, `note/`, `proxy/`, `rating/`, `search/`, `subclip/`, `take/`, `trim/`
- Platform-conditional: `database`, `import`, `manager` excluded on `wasm32`
- Uses Pure-Rust OxiSQL (`oxisql-sqlite-compat`) for async database, `tokio` for async I/O

## Enhancements
- [x] Add `clip_fingerprint` module to lib.rs exports (file exists at `src/clip_fingerprint.rs` but not declared)
- [x] Add `clip_merge` module to lib.rs exports (file exists at `src/clip_merge.rs` but not declared)
- [x] Add `clip_playlist` module to lib.rs exports (file exists at `src/clip_playlist.rs` but not declared)
- [x] Extend `search` with fuzzy matching for keyword and clip name queries (verified 2026-05-16; src/search/fuzzy.rs)
- [x] Add `clip_metadata` support for camera metadata (lens, ISO, aperture, shutter speed) — `CameraMetadata` struct with full builder API; `camera: Option<CameraMetadata>` field added to `Clip`; `CameraMetadataStore` for batch access
- [x] Implement `smart_collection` auto-refresh with configurable polling interval (verified 2026-05-16; src/group/smart.rs:28 poll_interval_secs, set_poll_interval:161)
- [x] Extend `export` with Final Cut Pro XML (FCPXML) export format (verified 2026-05-16; src/export/fcpxml.rs:294 lines)
- [x] Add `clip_relations` bidirectional linking (A relates to B implies B relates to A) — `clip_relations_bidirectional` module with `BidirectionalRelationGraph`
- [x] Improve `take::TakeSelector` with multi-criteria ranking (rating + recency + duration match) — `MultiCriteriaTakeSelector`, `TakeScoreWeights`/`TakeWeights`, `rank_takes_multi_criteria` free function

## New Features
- [x] Add `clip_waveform` module generating audio waveform thumbnails for timeline display — `WaveformThumbnail` with `generate(samples, width) -> Vec<f32>` RMS downsampler; `ClipWaveformGenerator` for min/max peak data
- [x] Implement `clip_ai_tag` module for automatic keyword suggestion based on visual content (verified 2026-05-16; src/clip_ai_tag.rs:442 lines AiTagger)
- [x] Add `clip_collaboration` module for multi-user clip annotation with conflict resolution (verified 2026-05-16; src/clip_collaboration.rs:578 lines)
- [x] Implement `clip_versioning` module tracking edit history with undo/redo support (verified 2026-05-16; src/clip_versioning.rs:556 lines ClipVersionHistory)
- [x] Add `clip_transcode_status` module tracking proxy generation and transcode progress per clip (verified 2026-05-16; src/clip_transcode_status.rs:482 lines TranscodeStatusStore)
- [x] Implement `clip_face_index` module for face-based clip search and grouping (verified 2026-05-16; src/clip_face_index.rs:470 lines ClipFaceIndex)
- [x] Add `clip_scene_detect` module automatically creating subclips at scene boundaries (verified 2026-05-16; src/clip_scene_detect.rs:434 lines ThresholdSceneDetector)
- [x] Implement `clip_favorites` module with quick-access collections and recent clips list (verified 2026-05-16; src/clip_favorites.rs:546 lines FavoriteCollection, RecentClipList)

## Performance
- [x] Add database index on `keywords` and `rating` columns for faster filtered queries (verified 2026-06-01; `idx_clips_rating` + `idx_clips_keywords` + `idx_clips_favorite_rating` in `src/database/migration.rs`)
- [x] Implement batch `add_clips()` method in `ClipManager` with single transaction for bulk import (verified 2026-06-01; `pub async fn add_clips` at `src/manager.rs:118` delegates to `database.batch_save_clips`)
- [x] Cache `SmartCollection` query results with invalidation on clip metadata changes (verified 2026-06-06; field-dependency auto-invalidation: `SmartRule::depends_on`→`ClipField`, `SmartCollection::dependency_fields`, and `ClipManager::{invalidate_smart_collections_for, invalidate_all_smart_collections}` wired into `update_clip` via `changed_fields` diff over an interior-mutability `Mutex<Vec<SmartCollection>>` registry in `src/manager.rs` + `src/group/smart.rs`)
- [x] Use prepared statement caching in `database` module to avoid repeated SQL parsing (verified 2026-06-06; explicit `.persistent(true)` on the hot read paths `get_clip`/`get_all_clips`/`search_clips`/`get_clips_page` in `src/database/storage.rs`)
- [x] Add pagination support to `search` and `ClipManager::list_clips()` for large clip libraries (verified 2026-06-01; `list_clips(page, page_size)` + `LIMIT ? OFFSET ?` in `src/database/storage.rs:334`)

## Testing
- [x] Add integration test for `ClipManager` full lifecycle: create, search, update, delete (tests/clip_manager_integration.rs::test_clip_lifecycle_create_search_update_delete)
- [ ] Test `smart_collection` auto-updating when new clips matching criteria are added
- [x] Add `export` round-trip test: export EDL, reimport, verify clip in/out points preserved (tests/clip_manager_integration.rs::test_export_edl_round_trip)
- [x] Test `subclip` creation with boundary conditions (start=0, end=clip duration) — `test_subclip_boundary_conditions` in `clip/subclip.rs`
- [x] Add concurrent access test for `ClipManager` with multiple simultaneous writers (tests/clip_manager_integration.rs::test_concurrent_add_clips; 5 tasks × 10 clips = 50 total)
- [x] Test `clip_compare` produces correct similarity scores for known duplicate/different clips (tests/clip_manager_integration.rs::test_clip_compare_identical_vs_different + test_clip_fingerprint_similarity_range)
- [x] Test `bin_organizer` automatic bin assignment based on metadata rules (tests/clip_manager_integration.rs::test_bin_organizer_auto_assignment_by_rating + test_bin_organizer_apply_rules_by_camera)

## Documentation
- [ ] Document database schema and migration strategy for `database` module
- [ ] Add guide for setting up proxy workflows with `proxy_link` module
- [ ] Document supported import/export formats with feature requirements

## 0.1.8 follow-up (added 2026-05-29 by /ultra)
- [x] Fix WASM regression — `uuid` and `serde_json` listed under `[target.'cfg(not(target_arch = "wasm32"))'.dependencies]` but 11 source files import them unconditionally (done 2026-05-29)
  - **Goal:** `cargo check --target wasm32-unknown-unknown -p oximedia-clips` passes cleanly.
  - **Design:** Move `uuid` and `serde_json` to unconditional `[dependencies]` (both are pure-Rust, WASM-safe crates; `uuid` already has `js` feature in workspace dep). Keep `tokio`, `oxisql-sqlite-compat`, `oximedia-audio`, `oximedia-container` under the wasm-cfg gate.
  - **Files:** `Cargo.toml`
  - **Tests:** `cargo check --target wasm32-unknown-unknown -p oximedia-clips`
  - **Risk:** Negligible — both deps compile cleanly to wasm32.
- [x] Wire FLAC/Opus/MP3/Vorbis waveform decode in `clip_waveform.rs` — currently returns `WaveformData::empty()` for these formats (done 2026-05-29)
  - **Goal:** `WaveformGenerator::generate()` produces real waveform data for FLAC, Opus, MP3, and Vorbis files; WAV already works. Fallback to `empty()` on decode error preserved.
  - **Design:** Feature-flag `audio-decode = ["dep:oximedia-container", "dep:oximedia-io"]` (default on). Under `#[cfg(all(feature = "audio-decode", not(target_arch = "wasm32")))]`: detect format via extension/magic, dispatch to `FlacDemuxer`/`OggDemuxer`/`MatroskaDemuxer` + matching `AudioDecoder` (FlacDecoder/OpusDecoder/VorbisDecoder/Mp3Decoder). Pump packets in `tokio::runtime::Builder::new_current_thread().enable_all().build()?.block_on(async { … })` bridge. Collect mono samples via per-frame averaging downmix.
  - **Files:** `src/clip_waveform.rs`, `Cargo.toml`
  - **Tests:** 7 new tests — WAV dispatch regression, MP3/OGG/FLAC/MKA missing/corrupted-file error paths, downmix unit tests. FlacDecoder/VorbisDecoder are stubs (Ok(None) from receive_frame) so those paths return empty until codecs are implemented; documented in NOTE comment.
  - **Limitation:** tokio runtime created per-call (documented in doc comment). FlacDecoder and VorbisDecoder are stubs; OpusDecoder and Mp3Decoder are functional.
