# oximedia-vfx TODO

## Current Status
- 103 source files across 17 top-level modules and numerous submodules
- Key features: VideoEffect/TransitionEffect traits, keyframe animation system, Frame RGBA processing, Color/Rect/Vec2 primitives, EasingFunction interpolation, ParameterTrack system
- Effect categories: transitions (dissolve, wipe, push, slide, zoom, 3D), generators (bars, gradient, noise, pattern, solid), keying (advanced, spill, edge), time effects (remap, speed, freeze, reverse), distortion (barrel, lens, wave, ripple), style (cartoon, halftone, mosaic, paint, sketch), light (bloom, flare, glow, rays, anamorphic), particles (snow, rain, sparks, dust), compositing (blend modes, alpha, matte, layers), tracking (point, planar, stabilize, mask), rotoscoping (bezier, assisted, keyframe), text (render, animate), shape (draw, animate, mask)
- Additional modules: blur_kernel, chroma_key, chromatic_aberration, color_grade, color_grading (curves/matching/wheels), color_lut, deform_mesh, depth_of_field, film_effect, fog, heat_distort, lens_aberration, lens_flare, motion_blur, vector_blur, noise_field, particle_fx, particle_sim, render_pass, ripple, trail_effect, vfx_preset

## Enhancements
- [ ] Add GPU compute shader backend to `VideoEffect::apply()` for effects that set `supports_gpu() = true` (verified-open 2026-05-16: no GPU/wgpu compute backend wired to VideoEffect::apply in vfx sources)
- [x] Implement multi-channel ParameterTrack (Vec2/Vec3/Color tracks) instead of single f32 value only (already implemented at src/param_track.rs:54 Vec2Track, :155 Vec3Track, :253 ColorTrack, Wave 14 verified 2026-06-01)
- [x] Add spring/bounce/elastic easing functions to `EasingFunction` enum beyond current 5 types
- [x] Extend `compositing::blend_modes` with all Photoshop-compatible blend modes (vivid light, pin light, hard mix)
- [x] Add feathered edge support to `keying::edge` refinement with configurable falloff curve
- [x] Implement motion-vector-based `motion_blur` that reads actual frame motion rather than uniform blur (plan: G-1 Slice G 0.1.8 — added MvMotionBlurConfig, mv_motion_blur(), MotionBlurMode::MotionVector, bilinear_sample to motion_blur.rs)
- [x] Add color management (OCIO-style) to `color_grading` pipeline for proper working-space transforms (verified 2026-05-16; src/color_grading/color_space.rs:1 OCIO-style color space transforms)
- [x] Extend `rotoscoping::bezier` with cubic B-spline and Catmull-Rom curve types for smoother masks (verified 2026-05-16; src/rotoscoping/spline_curves.rs:99 CatmullRomCurve, CubicBSplineCurve:32)
- [x] Add sub-pixel precision to `tracking::point` tracker using bilinear interpolation on correlation (plan: G-2 Slice G 0.1.8 — added track_point_subpixel() with Foroosh et al. 2002 parabolic interpolation to PointTracker)

## New Features
- [x] Implement `MorphTransition` effect (mesh-based morph between two frames for organic transitions) (verified 2026-05-16; src/transition/morph.rs 472 lines)
- [x] Add `Glitch` effect module (digital glitch, datamosh, RGB shift, scan line artifacts)
- [x] Implement `Tilt-Shift` miniature effect using configurable gradient mask with depth_of_field (verified 2026-05-16; src/tilt_shift.rs 588 lines)
- [x] Add `Vignette` effect with customizable shape (circular, elliptical, rectangular) and falloff
- [x] Implement `ChromaticAberration` as proper RGB channel offset with lens distortion model (verified 2026-05-16; src/chromatic_aberration.rs:56 ChromaticAberration, Brown-Conrady radial distortion model:14, 785 lines)
- [x] Add `RetroFilm` generator combining film_effect grain, color_grade desaturation, and vignette (verified 2026-05-16; src/retro_film.rs 665 lines)
- [x] Implement `Parallax` effect for 2.5D camera motion on layered images using depth maps (verified 2026-05-16; src/parallax.rs 427 lines)
- [x] Add `AudioReactive` effect modifier that drives effect parameters from audio amplitude/frequency data (verified 2026-05-16; src/audio_reactive.rs 567 lines)

## Performance
- [x] Use SIMD (std::simd when stable, or manual intrinsics) for pixel-level operations in `Frame::clear()`, `Color::blend()`, and `Color::lerp()` (implemented `simd_pixel.rs`: `fill_rgba` AVX2 32-byte chunks + u128 16-byte fallback; `blend_rgba_buffers` Porter-Duff fixed-point; `lerp_rgba_buffers` fixed-point; `apply_color_multiply` 2026-05-31)
- [x] Implement tile-based parallel processing in `VideoEffect::apply()` using rayon for large frames (already implemented at src/tile_processor.rs:113 struct TileProcessor, Wave 14 verified 2026-06-01)
- [x] Add frame caching/memoization in `ParameterTrack::evaluate()` for repeated same-time queries (already implemented at src/param_track_cache.rs:72 struct CachedParameterTrack, Wave 14 verified 2026-06-01)
- [x] Optimize `particle::system` with spatial partitioning (grid or quadtree) for particle-particle interactions (plan: G-3 Slice G 0.1.8 — added SpatialHashGrid (Teschner 2003) and apply_interactions() to system.rs; exposed spatial_grid mod in particle/mod.rs)
- [x] Implement downsampled preview path in `QualityMode::Draft` that processes at 1/4 resolution then upscales (already implemented at src/draft_preview.rs:179 struct DraftPreview, Wave 14 verified 2026-06-01)
- [x] Pool Frame allocations to avoid repeated large heap allocations in effect chains (already implemented at src/frame_pool.rs:28 struct FramePool, Wave 14 verified 2026-06-01)
- [x] Wire `MotionBlurMode::MotionVector` into `VideoEffect::apply()` dispatch (done 2026-06-01)
  - **Goal:** Connect the already-implemented `mv_motion_blur` free function to the `VideoEffect` trait dispatch path.
  - **Design:** Added `MotionBlur` struct implementing `VideoEffect::apply()` in `src/motion_blur.rs`. `MotionBlurMode::MotionVector` dispatches to `motion_vector_blur::apply_mv_blur` (RGBA Frame API). `UniformLinear` / `UniformRadial` dispatch to the kernel path via `apply_kernel_rgba`. Alpha channel preserved on kernel paths.
  - **Files:** `src/motion_blur.rs`, `TODO.md`.
  - **Tests added:** `test_motion_blur_effect_mv_mode_matches_free_fn`, `test_motion_blur_effect_mv_mode_no_field_passthrough`, `test_motion_blur_effect_uniform_linear_rgba_alpha_preserved`, `test_motion_blur_effect_name`, `test_motion_blur_effect_mv_mode_gray_channel_consistency` — all pass.

## Testing
- [ ] Add visual regression tests comparing effect output against golden reference frames (pixel diff threshold)
- [x] Test all TransitionEffect implementations at progress 0.0, 0.5, and 1.0 for correct blending behavior (done 2026-06-05; tests/transition_boundary.rs: Dissolve/Wipe/Push/Slide/Zoom, 15 tests)
- [x] Add boundary tests for Frame with maximum resolution (8K) and minimum (1x1) (done 2026-06-06; tests/frame_boundary.rs: 8K 7680x4320 alloc/clear/corner-addressing, 1x1 set/get/OOB-neighbours/clear, 1x1 Compositor::composite no-panic, zero-width/height/0x0 + u32::MAX overflow InvalidDimensions; 7 tests)
- [x] Test ParameterTrack interpolation with overlapping keyframes, NaN times, and single-keyframe tracks (done 2026-06-05; src/lib.rs tests: empty→None, single-kf, before/after clamp, same-time-replace, NaN-no-panic, linear-midpoint; 7 tests)
- [ ] Add performance benchmarks for each effect category to track regression (using criterion)
- [x] Test compositing deep layer stacks (50+ layers) for correctness (done 2026-06-06; tests/layer_compositing.rs. NOTE: the real deep-layer API is `Compositor::flatten_layers` + `LayerStack`, NOT a `layer_manager` stack; tests pin byte-exact flatten_layers vs a sequential blend_pixels fold over a 50-layer opacity ramp, the 127/0/127 opacity-multiplied and 128/0/126 alpha-encoded truncation oracles, single-layer identity, plus a 50-layer bilinear LayerStack::composite smoke test; 5 tests)

## Documentation
- [ ] Add visual examples (ASCII art or image references) for each transition type in `transition` module docs
- [ ] Document particle system parameter tuning guide in `particle` module
- [ ] Add effect chain cookbook showing common post-production workflows (color grade -> blur -> composite)
