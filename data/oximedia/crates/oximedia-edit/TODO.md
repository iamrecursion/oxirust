# oximedia-edit TODO

## Current Status
- 31 source files covering timeline, clips, tracks, effects, transitions, rendering, and editing operations
- Multi-track timeline with video, audio, and subtitle support
- Advanced editing: ripple, roll, slip, slide, blade tool, insert mode, trim mode
- Effects system with keyframe animation and interpolation
- Transitions: dissolve, wipe, zoom with easing functions
- Rendering: TimelineRenderer, PreviewRenderer, ExportRenderer, BackgroundRenderer
- Group/compound clips, markers, regions, nested sequences, undo history
- Dependencies: oximedia-core, oximedia-codec, oximedia-audio, oximedia-graph, tokio

## Enhancements
- [x] Add multi-cam editing support with sync point alignment across tracks (verified 2026-05-16; src/multicam.rs:96 CameraAngle, SyncPoint:53, 1135 lines)
- [x] Implement proxy workflow in `render.rs` (low-res editing, full-res export) (verified 2026-05-16; src/proxy.rs:162 ProxyManager, ProxyWorkflowConfig:339, 1277 lines)
- [x] Enhance `history.rs` undo/redo with branching history tree (not just linear stack) (verified 2026-06-01; src/history.rs:289 BranchingHistory<T> fully implemented)
- [x] Add magnetic timeline snapping in `timeline.rs` for clip alignment
      — `snap_engine: Option<MagneticSnapEngine>` on Timeline; `with_magnetic_snap()` builder; snap applied in `move_clip` before link cascade; 6 tests
- [ ] Improve `transition.rs` with GPU-accelerated transition rendering via oximedia-gpu (verified-open 2026-05-16: no GPU path in transition.rs)
- [x] Add audio waveform generation in timeline for visual editing feedback
      — `WaveformGenerator` (mono/multi-channel, peak/RMS/min) and `WaveformData` in `waveform.rs`; 25 tests
- [x] Enhance `nested_sequence.rs` with independent timeline resolution and frame rate
      — Custom `FrameRate` replaced with workspace `Rational`; positions are now `i64` (negative pre-roll); `resolution: (u32, u32)` field added; `duration_in_outer_timebase(outer_rate: Rational)` implemented; 16 tests
- [x] Add clip linking between video and audio clips in `group.rs` for sync maintenance
      — `LinkManager` wired into `move_clip` and `remove_clip`; BFS link cascade with cycle guard via HashSet; atomic snapshot-validate-apply pattern; `EditError::TrackTypeMismatch` enforced in `add_clip`; 10 tests
- [x] Implement `auto_edit.rs` with beat-detection based auto-cutting for music videos (verified 2026-05-16; src/auto_edit.rs:25 AutoSequenceConfig, AutoClip:73, 434 lines)
- [x] Add clip color labels and metadata tags for organizational workflow
      — `ColorLabel`, `Tag`, `LabelManager`, `StandardLabels` in `color_label.rs`; 28 tests

## New Features
- [x] Implement freeze frame and speed ramp (variable speed) in `clip_speed.rs`
      — Added `SpeedEffect` enum (Normal/FreezeFrame/ConstantSpeed/VariableSpeed) and `ClipSpeedController`
- [x] Add title/text overlay generation with basic font rendering (verified 2026-05-16; src/title_overlay.rs:376 TitleOverlay, TextStyle:333, 936 lines)
- [x] Implement picture-in-picture layout mode with position/scale keyframes (verified 2026-05-16; src/picture_in_picture.rs:543 lines)
- [x] Add EDL/XML export from timeline state for interchange with other NLEs
      — `TimelineExporter::to_edl` (CMX-3600) and `to_xml` (FCP XML skeleton) in `timeline_export.rs`
- [x] Implement multi-format export (export same timeline to multiple resolutions/codecs) (verified 2026-05-16; src/multi_export.rs:524 lines)
- [x] Add collaborative editing primitives (operational transform for concurrent edits) (verified 2026-05-16; src/collab_edit.rs:570 lines)
- [x] Implement smart trim with scene-change detection at trim points (verified 2026-05-16; src/smart_trim.rs:479 lines)
- [x] Add render queue for batch export of multiple timelines
      — `RenderQueue`, `RenderJob`, `JobStatus`, `ExportConfig`, `TimelineSnapshot` in `render_queue.rs`; 35 tests

## Performance
- [x] Add frame cache in `render.rs` to avoid re-decoding unchanged frames
      — `RawFrameCache` (HashMap<u64,Vec<u8>>, max 32 frames, LRU eviction) added to `render.rs`
- [x] Implement predictive pre-fetch for frames near playhead position (verified 2026-06-01; src/frame_prefetch.rs PrefetchEngine fully implemented, registered in lib.rs)
- [x] Optimize `clip.rs` clip lookup with interval tree for O(log n) time-position queries
      — `IntervalTree` with `query_point`/`query_range`/`nearest_edge` in `interval_tree.rs`
- [x] Add parallel track rendering in `TimelineRenderer` using rayon (verified 2026-06-01; src/parallel_render.rs render_tracks_parallel fully implemented, registered in lib.rs)
- [x] Implement incremental render (only re-render changed regions of timeline) (verified 2026-06-01)
  - `dirty_regions: Vec<DirtyRegion>` + `prefetch: PrefetchEngine` + `use_parallel: bool` added to `TimelineRenderer`
  - `mark_dirty` / `clear_dirty` / `force_full_redraw` / `is_position_dirty` API on `TimelineRenderer`
  - `render_video_at` skips positions outside dirty regions; parallel path via `render_video_at_parallel` → `render_tracks_parallel`
  - `render_frame_at` calls `self.prefetch.update(position)` after every render
  - `#![allow(dead_code)]` removed from `frame_prefetch.rs`; both modules fully wired
  - 17 new tests: 5 in `tests/renderer_unit.rs` (sync unit), 6 in `tests/incremental_render.rs` (async integration); 0 warnings

## Testing
- [x] Add round-trip test: create timeline -> export -> reimport -> verify identical structure
      — 15 tests in `tests/timeline_roundtrip.rs` covering EDL parse/verify and XML skeleton round-trip
- [x] Test `ripple.rs` and `slip_slide.rs` edit operations with overlapping clips
      — 28 tests in `tests/ripple_slip_slide.rs` covering 6 ripple variants × 4 configurations
- [x] Test `transition.rs` rendering output for all transition types
      — 22 tests in `tests/transition_render_all.rs` covering all 14 TransitionType variants at progress 0/0.5/1
- [x] Add stress test with 100+ tracks and 1000+ clips for timeline performance
      — 12 tests in `tests/timeline_stress.rs`; 100×1000 clip lookup timing asserts < 5ms mean
- [x] Test `blade_tool.rs` split at exact frame boundaries with various frame rates
      — 15 tests in `tests/blade_frame_boundary.rs` covering 6 frame rates including 23.976 and 29.97
- [x] Test `selection.rs` multi-clip selection operations across tracks
      — 31 tests in `tests/selection_multitrack.rs` covering add, range, invert, clear operations

## Documentation
- [ ] Document the editing operation semantics (ripple vs roll vs slip vs slide)
- [ ] Add timeline rendering pipeline architecture diagram
- [ ] Document the effect keyframe interpolation system in `effect.rs`

## Deferred Stubs (added 2026-05-04 by /stub-check Wave 3)

- [x] oximedia-edit: `src/render.rs:116` — render_video_at loop body
      — Full HdrCompositor pipeline: source resolution via `RenderSource` enum (Image/Wav/TestPattern/Unsupported); RawFrameCache wired; TransitionRenderer invoked for active transitions; 54 new tests across 8 files; `render_source.rs` added

- [x] oximedia-edit: `src/render.rs:159` — render_audio_at loop body
      — Per-clip volume/mute; CrossFade transitions via TransitionRenderer::mix_audio; mixed output clamped to [-1,1]
