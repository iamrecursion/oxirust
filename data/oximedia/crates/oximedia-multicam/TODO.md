# oximedia-multicam TODO

## Current Status
- 38 source files/directories for multi-camera synchronization, editing, and switching
- Sync: temporal, audio cross-correlation, timecode (LTC/VITC/SMPTE), visual markers, genlock, drift correction
- Editing: MultiCamTimeline, angle switching, transitions, edit decision lists
- Auto switching: AI-based camera selection, rules engine (speaker detection, action following, shot variety)
- Composition: picture-in-picture, split-screen, grid layouts
- Color matching and spatial alignment/stitching
- ISO recording, replay buffer, tally system, bank control, coverage mapping

## Enhancements
- [x] Implement actual audio cross-correlation in sync module using FFT-based correlation (currently stubbed)
- [x] Add sub-frame sync precision in angle_sync -- interpolate between frames for <1 frame accuracy (verified 2026-05-16; src/sub_frame_sync.rs:110 SubFrameSync, parabolic interpolation:13)
- [x] Improve auto switcher scoring in angle_score to weight composition quality using rule of thirds detection
- [x] Add smooth transition blending in edit module -- dissolve/wipe between angles during switch points
- [x] Implement genlock_master drift measurement and correction loop for long-form recording (>1 hour) (verified 2026-05-16; src/genlock_master.rs:486 GenlockMaster drift_threshold_ns:144, drifted_count tracking:119)
- [x] Add multi-angle audio selection -- allow independent audio source selection separate from video angle (verified 2026-05-16; src/audio_selection.rs:100 MultiAngleMixer, independent audio_angle:112)
- [x] Extend color matching in color module to handle LOG/RAW footage with LUT-based matching (verified 2026-05-16; src/color/match_color.rs:199 3D LUT-based transfer variant)

## New Features
- [x] Implement automatic highlight detection using cut_analysis + audio energy peaks for sports/events
- [x] Add multi-angle proxy generation for lightweight preview of all angles simultaneously (verified 2026-05-16; src/proxy_generator.rs:87 MultiAngleProxyGenerator, src/proxy_gen.rs)
- [x] Implement XML export (Final Cut Pro XML, AAF) in multicam_export for NLE interchange (verified 2026-05-16; src/fcp_xml.rs:395 FCP XML export, src/multicam_export.rs)
- [x] Add remote camera control protocol support (VISCA over IP) in addition to existing PTZ concepts (verified 2026-05-16; src/visca.rs:454 VISCA over UDP/TCP, VISCA_COMMAND encoding)
- [x] Implement automatic camera framing suggestions based on face detection and composition rules (verified 2026-05-16; src/framing_suggest.rs:107 FramingSuggestion, face detection:18)
- [x] Add multi-camera audio mixing -- combine boom mic, lavaliers, and camera audio with automatic switching (verified 2026-05-16; src/audio_selection.rs:73 audio mix config, boom-mic reference:5)
- [ ] Implement virtual camera output (v4l2/AVFoundation loopback) for live production preview (verified-open 2026-05-16: no v4l2/AVFoundation loopback module found)

## Performance
- [x] Parallelize sync verification across all angle pairs using rayon in sync_verify (verified 2026-05-16; src/sync_verify_parallel.rs:158 par_iter() across angle pairs)
- [x] Cache angle score computations per frame to avoid redundant face/motion detection (completed 2026-06-01)
  - **Goal:** Memoize per-(angle, frame) score so repeated queries don't re-run detection.
  - **Design:** `src/angle_score.rs:225` `AngleScorer` has no cache field. Add `HashMap<(AngleId, FrameNumber), AngleScore>` memo; check before calling the heavy computation path. Invalidate on scorer config change.
  - **Files:** `src/angle_score.rs`, `TODO.md`.
  - **Tests:** cached score == uncached on first call; second call returns same value without recomputing; config change invalidates.
  - **Risk:** cache key must be stable; memory bound the cache (LRU or bounded window).
- [x] Lazy frame loading in MultiCamTimeline (decode only active/preview angles) (completed 2026-06-01)
  - **Goal:** Wire the existing `LazyFrameRef` primitive into `MultiCamTimeline` so inactive angles are not decoded.
  - **Design:** `src/edit/lazy_frame.rs:LazyFrameRef` exists with `OnceCell` deferred decode but `src/edit/timeline.rs:MultiCamTimeline` never references it. Change `MultiCamTimeline::frame_at(angle, timecode)` to return `LazyFrameRef` (or wrap frames in it) so inactive angles skip decoding until accessed.
  - **Files:** `src/edit/timeline.rs`, `src/edit/lazy_frame.rs` (read), `TODO.md`.
  - **Tests:** lazy-frame for inactive angle is NOT decoded until `force()` is called; active angle is decoded immediately on access; 4-angle setup only decodes the active one.
  - **Risk:** lazy-frame cache key must be stable across timeline mutations.
- [x] Tile-based color matching to reduce full-frame processing to sampled regions (completed 2026-06-01)
  - **Goal:** Reduce color matching cost by sampling a grid of tiles instead of every pixel.
  - **Design:** `src/color/match_color.rs` does full-frame histogram/stats for 3D LUT transfer at :199. Add `match_color_tiled(src, reference, tile_n: usize)` that samples a `tile_n × tile_n` grid of tiles (evenly spaced) for stats computation instead of every pixel.
  - **Files:** `src/color/match_color.rs`, `TODO.md`.
  - **Tests:** tiled match ≈ full-frame match within ΔE tolerance at tile_n=8; tile_n=1 (single center tile) still runs without panic; tile_n > frame dimensions degrades gracefully.
  - **Risk:** sampling bias on non-uniform images — use stratified grid, not random.
- [x] Incremental coverage_map updates instead of full recomputation when angles change (completed 2026-06-01)
  - **Goal:** Recompute only coverage cells affected by changed angles, not the entire map.
  - **Design:** `src/coverage_map.rs` (283L) recomputes the full map on every call with no dirty tracking. Add `dirty_angles: HashSet<AngleId>` field; add `update_coverage(changed_angles: &[AngleId])` that only recomputes cells overlapping with those angles; keep `full_rebuild()` for initial construction.
  - **Files:** `src/coverage_map.rs`, `TODO.md`.
  - **Tests:** incremental update == full rebuild after a single-angle change; multiple concurrent angle changes processed correctly; no-change call is a no-op.
  - **Risk:** partial update must be conservative (all cells affected by changed angle, not just direct overlap); test against full rebuild to ensure correctness.

## Testing
- [x] Test audio sync with known-offset synthetic signals (1kHz tone with 100ms offset between channels)
- [x] Add test for auto switcher with synthetic frame scores verifying angle selection logic (impl 2026-06-06: `tests/auto_switcher.rs` — score→switch, confidence gate, hold cooldown, tie-break to highest index)
- [x] Test PIP composition output dimensions and placement for all corner positions (impl 2026-06-06: `tests/pip_composition.rs` — all 4 corners + Custom + padding override + scale clamp + containment invariant @ 1080p/0.25 and 720p/0.5)
- [x] Verify tally_system state transitions: preview -> program -> off lifecycle (impl 2026-06-06: `tests/tally_lifecycle.rs` — preview→program→off lifecycle + history + single-program bus invariant + clock-driven timestamps)
- [x] Test ISO recording with simulated multi-angle input verifying per-angle file output (impl 2026-06-06: `tests/iso_recording.rs` — 4-angle lifecycle, no angle dropped, distinct per-angle file names, multi-quality ordering, dedup + double-start/stop guards)

## Documentation
- [ ] Add multi-camera production workflow guide (setup -> sync -> edit -> export)
- [ ] Document the auto-switching rules engine with examples for common production scenarios
- [ ] Add architectural diagram showing the relationship between sync, edit, auto, composite modules
