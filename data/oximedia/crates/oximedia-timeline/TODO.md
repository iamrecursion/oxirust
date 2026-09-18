# oximedia-timeline TODO

## Current Status
- 50 source files across 48 public modules providing a professional multi-track timeline editor
- Core: Timeline, Track (video/audio), Clip with MediaSource, Position/Duration/Speed types
- Editing: EditMode/EditOperation, razor tool, snap grid, clip sequence, compound clip, compound region
- Effects: EffectStack with keyframe animation (Keyframe, KeyframeInterpolation, KeyframeValue), keyframe_animation
- Transitions: TransitionType (dissolve, etc.), TransitionEngine, transition alignment
- Multi-camera: multicam editing, nested sequences (nested, nested_timeline, nested_compound)
- Export/Import: EDL, XML, AAF via timeline_exporter; export, export_settings, import modules
- Rendering: renderer (PixelBuffer, RenderedFrame, FrameRenderSettings), render, render_queue
- Playback: async playback (non-wasm), cache module
- Other: markers, metadata, clip_metadata, audio mixer, timecode, timeline_diff, timeline_event, timeline_lock, track_color, track_group, track_routing, gap_filler, conform, version_snapshot, color_correction_track, sequence_range
- Dependencies: oximedia-core, oximedia-container, oximedia-graph, oximedia-edl, oximedia-aaf, oximedia-audio, tokio, serde, uuid, parking_lot, chrono, dashmap, quick-xml

## Enhancements
- [x] Add three-point and four-point editing support in `edit::EditOperation` (mark in/out on source and timeline) (verified 2026-05-16; src/point_edit.rs:702 lines)
- [x] Extend `multicam` to support angle switching based on timecode match (not just frame position) (verified 2026-05-16; src/multicam.rs:381 add_switch_by_timecode, sync_angles_by_timecode:486)
- [x] Improve `snap_grid` to support magnetic snap to clip boundaries, markers, and playhead position
- [x] Add `clip_metadata` support for embedded media metadata (codec info, color space, audio channels) (verified 2026-05-16; src/clip_metadata.rs:732 lines)
- [x] Extend transition_engine to support asymmetric transitions (different in/out curves) (done 2026-06-01)
  - **Goal:** Allow different easing functions for the incoming and outgoing halves of a transition.
  - **Design:** `src/transition_engine.rs:52` `TransitionEngine` / `apply` uses a single progress curve. Add `in_ease: EasingFn` and `out_ease: EasingFn` fields to `TransitionInput` (or equivalent config). In `apply`, split `progress` at 0.5: first half uses `in_ease(progress*2)`, second half uses `out_ease((progress-0.5)*2)`. Default to symmetric (same curve for both) for backward compatibility.
  - **Files:** `src/transition_engine.rs`, `TODO.md`.
  - **Tests:** asymmetric transition at progress=0.25 uses `in_ease` curve output; at progress=0.75 uses `out_ease`; symmetric default produces identical output to current behavior.
  - **Risk:** backward compatibility — ensure all existing callers get symmetric default; easing function signature must match.
- [x] Add linked clip groups in `track` where moving one clip auto-moves linked clips on other tracks (verified 2026-05-16; src/linked_clips.rs:541 lines)
- [x] Implement `timeline_lock` per-track locking (currently likely timeline-level) for collaborative editing (verified 2026-05-16; src/timeline_lock.rs:122 lock_track fn, lock_range fn:107)
- [x] Extend `keyframe_animation` with bezier curve interpolation for smooth easing

## New Features
- [x] Implement `proxy_workflow` module for offline/online editing with proxy media and auto-conform to full resolution (verified 2026-05-16; src/proxy_workflow.rs:359 lines)
- [x] Add `adjustment_layer` module for applying effects to all tracks below without modifying individual clips
- [x] Implement `freeze_frame` module for creating freeze frame clips from any point in a source clip
- [x] Add `speed_ramp` module for variable-speed effects with keyframeable speed curves (optical flow interpolation) (verified 2026-05-16; src/speed_ramp.rs:530 lines)
- [x] Implement `multi_select` module for group operations on multiple clips (move, trim, delete, copy) (verified 2026-05-16; src/multi_select.rs:395 lines)
- [x] Add `auto_sequence` module for automatic assembly editing based on subclip markers (verified 2026-05-16; src/auto_sequence.rs:605 lines)
- [x] Implement `fcpxml_export` module for Final Cut Pro XML timeline interchange format (verified 2026-05-16; src/fcpxml_export.rs:501 lines)
- [x] Add `otio_export` module for OpenTimelineIO format support (industry-standard timeline interchange) (verified 2026-05-16; src/otio_export.rs:617 lines)
- [x] Implement `timeline_compare` module for diffing two timeline versions with visual conflict resolution (verified 2026-05-16; src/timeline_compare.rs:527 lines)

## Performance
- [x] Implement interval tree in `timeline` for O(log n) clip lookup by time position instead of linear scan
- [x] Add frame cache warming in `playback` that pre-renders upcoming frames based on playhead velocity (verified 2026-05-16; src/cache_warming.rs:609 lines)
- [ ] Use memory-mapped audio file access in `audio` mixer for large project with many audio clips (verified-open 2026-05-16: no mmap in audio.rs; audio.rs is gain/fade/pan config only with no file I/O — mmap change not applicable)
- [x] Implement incremental render in `renderer` that only re-composites tracks with changed clips (verified 2026-05-16; src/incremental_render.rs:729 lines)
- [x] Add lazy clip loading in `import` that defers media probing until clip is first accessed on timeline (verified 2026-05-16; src/lazy_clip.rs:600 lines)
- [x] Parallelize multi-track rendering in render using rayon (done 2026-06-01)
  - **Goal:** Decode independent video tracks in parallel, then composite in z_index order.
  - **Design:** `src/render.rs:309` `render_video_at` loops video tracks sequentially at :333. Use `rayon::par_iter` over independent tracks to run per-track decode in parallel; collect results, then composite in z_index order sequentially (compositing order must be preserved). `rayon` is already a dep.
  - **Files:** `src/render.rs`, `TODO.md`.
  - **Tests:** parallel output == sequential output (pixel-exact or near-exact on synthetic data); single-track case unchanged; 3-track z_index ordering preserved.
  - **Risk:** z_index sort must happen AFTER parallel decode, before compositing; assert stability.

## Testing
- [ ] Add integration test for full edit workflow: create timeline -> add tracks -> insert clips -> add transition -> export EDL -> verify output
- [ ] Test `razor_tool` split at various positions (clip start, middle, end, on transition boundary)
- [x] Add `nested_timeline` depth test with 3+ levels of nesting and verify correct frame resolution (done 2026-06-05; tests/timeline_integration.rs:test_nested_timeline_3_levels_frame_resolution)
- [ ] Test `multicam` angle switching with 4+ camera angles and verify correct source selection per cut
- [x] Add round-trip test: Timeline -> EDL export -> EDL import -> compare timelines for equivalence (done 2026-06-05; tests/timeline_integration.rs:test_edl_roundtrip_equivalence)
- [x] Test `timeline_diff` with known edit operations and verify diff correctly identifies changes (done 2026-06-05; tests/timeline_integration.rs:test_timeline_diff_{insert,move,delete}_clip)

## Documentation
- [ ] Add NLE (Non-Linear Editor) concepts guide explaining tracks, clips, transitions, and keyframes
- [ ] Document supported import/export formats with feature matrix (EDL, XML, AAF, FCPXML, OTIO)
- [ ] Add example showing multi-camera editing workflow from sync to final output
