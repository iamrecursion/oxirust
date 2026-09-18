# oximedia-aaf TODO

## Current Status
- 34 modules covering full AAF object model, structured storage, composition mobs, effects, timeline, dictionary, essence, metadata
- Reader (`AafReader`) and writer (`AafWriter`) with SMPTE ST 377-1 compliance
- Export to EDL, XML, and OpenTimelineIO formats
- 702 unit tests pass; clippy clean
- Dependencies: oximedia-core, oximedia-timecode, uuid, chrono, byteorder, serde, bitflags, bytes

## Enhancements
- [x] Add streaming/incremental reading to `AafReader` for large AAF files (`streaming.rs`: `AafStreamReader<R>`, `AafEvent`, `StreamReaderBuilder`, `collect_events`)
- [x] Implement lazy essence data loading in `read_essence_data` (`lazy_essence.rs`: `LazyEssence` with `Arc<Mutex<Option<Vec<u8>>>>` cache, `EssenceCollection`; 25 tests)
- [x] Add mob cloning/duplication support in `ContentStorage` with new UUID generation
- [x] Implement `find_composition_mob` by name (not just UUID) for ergonomic lookups
- [x] Add track re-ordering and insertion APIs to `CompositionMob` (`insert_track_at`, `remove_track_at`, `move_track`, `reorder_tracks`)
- [x] Implement nested effect parameter keyframe interpolation in `parameter` module (Catmull-Rom BSpline, geometric Log, ease-in Power; 10 new tests)
- [x] Add validation pass after reading: verify all mob references resolve, all required properties present
- [x] Support AAF Edit Protocol (read/modify/write without losing unknown properties) (`edit_session.rs`)
- [x] Add `Display` implementations for `EditRate`, `Position`, and timeline types for debugging

## New Features
- [x] Implement AAF low-level dump/inspection tool (hex + structure view) for debugging corrupt files (`inspector.rs`)
- [x] Add merge capability: combine multiple AAF files into a single composition (`merge.rs`)
- [x] Implement Final Cut Pro XML (FCPXML) export in `convert` module (`convert/fcpxml.rs`, `convert/fcpxml10.rs`)
- [x] Add DaVinci Resolve EDL dialect support in `edl_export` (`davinci_edl.rs`)
- [x] Implement AAF metadata search/query API (find mobs/clips matching criteria) (`search.rs`: `AafQuery`, `AafSearcher`)
- [x] Add Avid bin structure reading/writing for Avid Media Composer compatibility (`avid_bin.rs`)
- [x] Implement essence relinking: update media file references when paths change (`relink.rs`)
- [x] Register orphan modules `avid_bin.rs` (644 LoC) and `relink.rs` (247 LoC) in `lib.rs` — real implementations sitting on disk, never compiled (planned 2026-05-29)
  - **Goal:** Both modules compile and are exposed as `pub mod avid_bin` / `pub mod relink` in the crate's public API; existing tests still pass; zero clippy warnings.
  - **Design:** Add `pub mod avid_bin;` and `pub mod relink;` to `src/lib.rs` after the existing module block. Run `cargo check -p oximedia-aaf --all-features` and fix any surfaced compile errors (likely outdated `use` paths). Fix clippy warnings at source.
  - **Files:** `src/lib.rs`, `src/avid_bin.rs`, `src/relink.rs`
  - **Tests:** Smoke test importing one symbol from each newly-registered module (forces compilation + use). Existing test suite must pass.
  - **Risk:** Orphan modules may have stale `use` paths from before recent API changes; fix per-file.
- [x] Add timeline flattening: resolve nested compositions into a single flat sequence (`flatten.rs`)

## Performance
- [x] Cache parsed dictionary entries to avoid re-parsing on repeated lookups (`dict_cache.rs`: LRU `DictCache`)
- [x] Use memory-mapped I/O for large structured storage files in `StorageReader`
- [x] Implement zero-copy byte slicing for essence data reading where possible
- [x] Add parallel mob/track parsing for large compositions with many tracks

## Testing
- [x] Add round-trip test: create AAF -> write -> read -> verify all fields match
- [ ] Test `edl_export` output against reference EDL files from Avid/Premiere
- [ ] Add tests for edge cases: empty compositions, single-frame clips, nested effects
- [x] Test `xml_bridge` XML serialization/deserialization round-trip
- [ ] Test handling of corrupted structured storage headers (graceful error, not panic)
- [ ] Add tests for `mob_traversal` with deep mob reference chains

## Documentation
- [ ] Document the AAF object model hierarchy with a diagram (Header -> ContentStorage -> Mobs -> Slots)
- [ ] Add examples for common workflows: read AAF, extract clip list, export EDL
- [ ] Document supported AAF versions and known limitations vs. Avid/Adobe implementations
