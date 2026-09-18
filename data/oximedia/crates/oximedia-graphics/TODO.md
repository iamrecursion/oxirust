# oximedia-graphics TODO

## Current Status
- 46+ modules implementing a broadcast graphics engine
- Core rendering: primitives, render, shape_render, path_builder, svg_renderer
- Broadcast elements: lower_third, ticker, scoreboard, clock_widget, countdown_timer, weather_widget, virtual_set
- Animation: animation, animation_curve, keyframe, transitions, transition_wipe, particles
- Text: text, text_renderer, text_layout, bitmap_font, font_metrics
- Color: color, color_blend, color_grade, color_picker, gradient_fill, hdr_composite, lut_apply
- Compositing: overlay, mask_layer, effects, sprite_sheet
- GPU: wgpu-based GPU rendering (feature-gated), WGSL shaders
- Server: axum-based control server (feature-gated)
- Dependencies: tiny-skia, fontdue, ab_glyph, kurbo, palette, wgpu, tera

## Enhancements
- [x] Add anti-aliased rendering to `shape_render.rs` for smoother vector output at broadcast resolution
- [x] Extend `animation_curve.rs` with spring physics and damped oscillation easing functions
- [x] Add multi-line text wrapping with justification modes to `text_layout.rs` (verified 2026-05-16; src/text_layout.rs:44 TextAlign::Justify, justify multi-line:627, test:926)
- [x] Implement text outline and drop shadow directly in `text_renderer.rs` (not as post-effect) (verified 2026-05-16; src/text_renderer.rs:272 TextOutline struct, DropShadow:319)
- [x] Add real-time data binding to `scoreboard.rs` (live score updates via data source trait) (verified 2026-05-16; src/scoreboard.rs:369 ScoreDataSource trait, LiveScoreboard:426)
- [x] Extend `ticker.rs` with variable scroll speed, pause-on-item, and looping modes (verified 2026-05-16; src/ticker.rs:101 scroll_speed_pps, pause_on_item:115, item_speed_multiplier:239)
- [x] Add configurable safe-area margins to `layout_engine.rs` for broadcast compliance (verified 2026-05-16; src/layout_margins.rs:152 SafeAreaMargins, BroadcastStandard, SafeRect)
- [x] Implement sub-pixel font rendering in `font_metrics.rs` for sharper text at small sizes (verified 2026-05-16; src/font_metrics.rs:279 SubPixelMode enum, SubPixelMetrics:388)
- [x] Add motion blur to `particles.rs` for more realistic particle trails (verified 2026-05-16; src/particles.rs:318 render_to_rgba_motion_blur)

## New Features
- [x] Add a `chroma_key.rs` module for green/blue screen keying with spill suppression
- [x] Implement a `picture_in_picture.rs` module with configurable position, size, border, and shadow
- [x] Add a `crawl.rs` module for vertical text crawl (end credits style) (verified 2026-05-16; src/crawl.rs:1021 lines)
- [x] Implement a `data_visualization.rs` module for live charts/graphs (bar, line, pie) in broadcast
- [x] Add a `stinger_transition.rs` module for animated full-screen transitions with alpha channel (verified 2026-05-16; src/stinger_transition.rs:465 lines)
- [x] Implement a `logo_bug.rs` module for persistent corner logo placement with fade in/out (verified 2026-05-16; src/logo_bug.rs:504 lines)
- [x] Add an `audio_waveform.rs` module for rendering audio waveform/spectrum visualizations (verified 2026-05-16; src/audio_waveform.rs:668 lines)
- [x] Implement `template_variables.rs` for tera template variable hot-reload from JSON/API (verified 2026-05-16; src/template_variables.rs:484 lines)

## Performance
- [x] Batch draw calls in `render.rs` to reduce per-primitive overhead (completed 2026-06-01)
  - **Goal:** Accumulate draw commands and flush in one pass per layer instead of issuing one tiny-skia call per primitive.
  - **Design:** `src/render.rs:99-228` — `render_rect`/`render_circle`/`render_path` each issue one tiny-skia fill call; `render_text` at :213 draws one rect per glyph. The module already contains `src/draw_batch.rs::DrawBatch` (registered in lib.rs:114) but it is NOT imported in render.rs. Wire `DrawBatch` into `SoftwareRenderer`: accumulate `DrawCommand`s on each render call, sort by layer (stable), flush in one batched pass. Preserve current draw order (layer-stable sort).
  - **Files:** `src/render.rs`, `TODO.md`.
  - **Tests:** batched render produces pixel-identical output to per-primitive render on a multi-element scene; layer ordering is preserved; empty batch is a no-op.
  - **Risk:** sort must be stable to preserve same-layer ordering; flush must happen before frame is read back.
- [x] Implement dirty-region tracking in `overlay.rs` to only re-render changed areas (completed 2026-06-01)
  - **Goal:** Skip alpha-blending unchanged regions of the overlay composite.
  - **Design:** `src/overlay.rs:26` `OverlayCompositor::composite()` full-frame alpha-blends every call (lines :66-80). The module `src/dirty_region.rs::DirtyRegionTracker` already exists and is registered in lib.rs:126 but NOT imported in overlay.rs. Add a `dirty_region: DirtyRegionTracker` field to `OverlayCompositor`; clip the blend loop at :66-80 to only process pixels within `tracker.regions()`. Callers mark dirty on overlay changes.
  - **Files:** `src/overlay.rs`, `TODO.md`.
  - **Tests:** dirty-region composite == full composite on changed areas; unchanged areas are skipped (verify pixel values and call-count metrics); empty dirty set is a no-op.
  - **Risk:** dirty-region API must track sub-pixel and sub-tile granularity to match what composite uses; test with overlapping dirty regions.
- [ ] Add GPU-accelerated text rendering path in `gpu.rs` using glyph atlas textures
- [x] Cache rasterized glyphs in `bitmap_font.rs` to avoid redundant rasterization per frame (completed 2026-06-01)
  - **Goal:** Memoize scaled RGBA glyph bitmaps to avoid rasterizing the same glyph at the same size every frame.
  - **Design:** `src/bitmap_font.rs:77` `BitmapFont::render_text` rasterizes from the `HashMap<char,Vec<u8>>` glyph table each call without caching the scaled output. The module `src/glyph_cache.rs::GlyphCache` (LRU, registered lib.rs:128) is NOT imported here. Add a `GlyphCache` field to `BitmapFont` (or use it as a shared cache), keyed by `(char, size_pt, scale_factor)`, returning `Arc<Vec<u8>>` RGBA glyph bitmaps.
  - **Files:** `src/bitmap_font.rs`, `TODO.md`.
  - **Tests:** cache hit returns byte-identical result to fresh rasterization; cache miss rasterizes and populates; eviction under LRU capacity limit; multi-char string renders correctly with mixed hit/miss.
  - **Risk:** key must include both size and scale to avoid stale hit on font-size change; Arc-sharing must not expose mutable interior.
- [x] Parallelize `gradient_fill.rs` scanline computation with rayon for large surfaces (completed 2026-06-01)
  - **Goal:** Add a `render_to_rgba` scanline-fill function parallelized over rows via rayon.
  - **Design:** `src/gradient_fill.rs` currently only exposes `sample_at(x,y)` point-sampling (at :279/:405/:533) — no fill function exists. Add `render_to_rgba(width: u32, height: u32, buf: &mut [u8])` that fills scanlines by calling `sample_at` per pixel, parallelized via `rayon::par_chunks_mut(width*4)` over rows (one row = 4 bytes/pixel RGBA). `rayon` is already a dep.
  - **Files:** `src/gradient_fill.rs`, `TODO.md`.
  - **Tests:** parallel `render_to_rgba` output == serial per-pixel `sample_at` sweep (bit-exact); correct pixel count for non-square surfaces; single-row surfaces; gradient type coverage (linear, radial, conical).
  - **Risk:** `rayon` nondeterminism — assert bit-exact vs scalar (sample_at is pure fn); keep gradient_fill.rs < 2000 lines.
- [ ] Add texture atlas for `sprite_sheet.rs` to reduce GPU texture switches

## Testing
- [x] Add visual regression tests for `lower_third.rs`, `ticker.rs`, and `scoreboard.rs` output (already implemented at tests/visual_regression.rs:97 test_lower_third_classic_dimensions, :248 test_ticker_render_dimensions_1080p, :370 test_scoreboard_render_dimensions_1080p, Wave 14 verified 2026-06-01)
- [x] Test `animation.rs` keyframe interpolation at boundary conditions (t=0.0, t=1.0, overshoot) (implemented at tests/easing_boundary.rs — covers `keyframe::Easing` all 24 variants: anchors @0/1, input-clamp aliasing, monotone family in-range, Back undershoot/overshoot, Elastic oscillation, Spring+CubicBezier internal clamp, Bounce in-range; Wave 28 2026-06-06)
- [ ] Add tests for `hdr_composite.rs` tone mapping correctness against reference values
- [ ] Test `svg_renderer.rs` with complex paths including curves, arcs, and clip regions
- [ ] Add GPU rendering integration tests that verify `gpu.rs` output matches software path

## Documentation
- [ ] Add visual examples (rendered PNGs) for each broadcast element in module docs
- [ ] Document the tera template variable schema used by `graphic_template.rs`
- [ ] Add a guide for creating custom broadcast graphics presets using `presets.rs`

## Wave 4 Progress (2026-04-18)
- [x] wasm-gpu-compute-fix: cfg-gate gpu_compute.rs Send+Sync assumptions for wasm32 — Wave 4 Slice B

## 0.1.8 Wave 4 follow-up (added 2026-05-29 by /ultra)

- [x] Register 25 orphan modules in oximedia-graphics (wire-ready ones get `pub mod`, stubs get annotation) (completed 2026-05-29)
  - **Goal:** 25 source files in `crates/oximedia-graphics/src/` are never compiled because they lack `pub mod` declarations in `lib.rs`. Enumerate, classify, and register all wire-ready ones. Stub files get `// orphan: TODO — wire once <feature> lands` annotation.
  - **Design:** For each orphan: (1) scan first 100 lines for `unimplemented!()`/`todo!()`/`panic!("not implemented")` → mark `intentional-stub`; (2) scan for stale type refs requiring >50 LoC fixes → mark `compile-fix-needed`, annotate; (3) otherwise: `wire-ready` → add `pub mod <name>;` to lib.rs. Batch 4–6 orphans at a time; run `cargo nextest run -p oximedia-graphics --all-features` + `cargo clippy -p oximedia-graphics --all-features --all-targets -- -D warnings` after each batch. Add smoke test per registered orphan in `tests/orphan_smoke.rs`.
  - **Files:** `crates/oximedia-graphics/src/lib.rs` (new `pub mod` declarations), `crates/oximedia-graphics/tests/orphan_smoke.rs` (new or extend), per-orphan files (annotation comments only for stubs)
  - **Tests:** Smoke test per registered orphan (instantiation + one invariant). Existing tests must pass. Zero new clippy warnings.
  - **Risk:** Registered orphans may pull stale dependency types. If >50 LoC plumbing per orphan, annotate instead of force-register.
