# oximedia-virtual TODO

## Current Status
- 91 source files covering virtual production and LED wall tools
- Key features: VirtualProduction main system, camera tracking/calibration, LED wall rendering with perspective correction, ICVFX compositing, color pipeline management, genlock synchronization, motion capture integration, green screen keying, Unreal Engine integration, multi-camera support
- Submodules: math (linear algebra), color (pipeline), greenscreen, icvfx (composite), keying, led (render), lens, mocap, multicam (manager), preview, sync (genlock), tracking (camera), unreal, virtual_set
- Additional: background_plate, camera_frustum, camera_rig, frustum_culling, frustum, led_volume, led_wall, light_rig, metrics, motion_path, ndi_bridge, panel_topology, pixel_mapping, projection_map, render_layer, render_output, scene, scene_setup, stage, stage_layout, stage_manager, talent_keying, tracking_data, tracking_session, volume_calibration, workflows, virtual_studio
- Dependencies: thiserror, serde, serde_json (minimal -- no external media deps)

## Enhancements
- [x] Integrate with oximedia-core frame types instead of standalone data structures for interop with other crates (verified 2026-05-31; src/core_interop.rs imports oximedia_core::{frame_info,types})
- [x] Extend `icvfx::composite` to support multi-layer compositing (foreground talent + LED content + CG overlay) (verified 2026-05-31; src/icvfx/composite.rs:369 composite_multi_layer + LayerType::ForegroundTalent + CgOverlay)
- [x] Add real lens distortion models (Brown-Conrady, fisheye) to `lens` module beyond basic radial correction (verified 2026-05-16; src/lens/distortion.rs:15 BrownConrady variant, equidistant fisheye:16, k1..k6:30)
- [x] Implement actual frustum-to-LED-panel UV mapping in `projection_map` with sub-pixel accuracy
- [x] Extend `color::pipeline` with ACES color management support (ACEScg working space, RRT+ODT transforms)
- [x] Add latency measurement and compensation in `sync::genlock` for tracking-to-render pipeline
- [x] Improve `mocap` integration with standard mocap data formats (BVH, C3D, FBX skeleton import) (verified 2026-05-16; src/mocap/mod.rs BVH+C3D parsers, src/mocap/bvh.rs:1)
- [x] Extend `multicam::manager` with automatic camera selection based on talent tracking position
- [x] Add thermal drift compensation to `volume_calibration` for LED panels that shift color temperature over time

## New Features
- [x] Implement `ar_overlay` module for augmented reality marker-based object placement on live camera feed (verified 2026-05-16; src/ar_overlay.rs:955 lines)
- [x] Add `stage_visualization` module for 3D wireframe preview of stage layout, LED panels, and camera positions (verified 2026-05-16; src/stage_visualization.rs:696 lines)
- [x] Implement `talent_tracking` module using 2D pose estimation for automatic talent masking without green screen (verified 2026-05-16; src/talent_tracking.rs:827 lines)
- [x] Add `set_extension` module for extending physical sets with virtual elements beyond LED wall boundaries (verified 2026-05-16; src/set_extension.rs:557 lines)
- [x] Implement `hdri_capture` module for capturing real-world lighting and applying to virtual scenes (verified 2026-05-16; src/hdri_capture.rs:655 lines)
- [x] Add `previz` module for pre-visualization workflows (storyboard-to-virtual-set blocking) (verified 2026-05-16; src/previz.rs:544 lines)
- [x] Implement `remote_session` module for remote virtual production monitoring and control over network (verified 2026-05-16; src/remote_session.rs:1057 lines)
- [x] Add `stage_safety` module for tracking safe zones, warning when talent/equipment approach LED wall boundaries (verified 2026-05-16; src/stage_safety.rs:702 lines)

## Performance
- [ ] Implement GPU-accelerated frustum rendering in `led::render` for real-time 4K+ LED wall content
- [x] Add frame prediction in `camera_tracking` to compensate for tracking latency with motion extrapolation
- [x] Optimize `pixel_mapping` with lookup table caching for LED panel pixel-to-UV mapping (verified 2026-05-31; GlobalPixelLut + PixelMapper LUT integration in pixel_mapping.rs)
- [x] Implement tile-based rendering in `icvfx::composite` for parallel compositing of independent screen regions (verified 2026-05-31; icvfx/tiled.rs wired as pub mod tiled + pub use TiledCompositor in icvfx/mod.rs)
- [x] Add LOD (level-of-detail) system in `render_layer` for rendering distant virtual objects at lower resolution (verified 2026-05-31; LodConfig + RenderLayer::with_lod + select_lod_scale + render_at_lod in render_layer.rs)
- [x] Profile `math` module and add SIMD-optimized matrix multiplication for 4x4 transforms (verified 2026-05-31; math/simd_matrix.rs Matrix4x4f wired as pub mod simd_matrix in math/mod.rs)

## Testing
- [x] Add integration test for full pipeline: camera_tracker -> frustum -> led_render -> compositor (verified 2026-05-31; tests/wave13_tests.rs::test_full_pipeline_integration)
- [x] Test `genlock` synchronization accuracy with simulated frame timing jitter (verified 2026-05-31; tests/wave13_tests.rs::test_genlock_jitter_tolerance)
- [x] Add calibration round-trip tests: calibrate LED panel, render test pattern, verify pixel accuracy (verified 2026-05-31; tests/wave13_tests.rs::test_calibration_round_trip)
- [x] Test `multicam::manager` camera switching with concurrent tracking data from multiple cameras (verified 2026-05-31; tests/wave13_tests.rs::test_multicam_auto_selection_debounce)
- [x] Add stress test for `stage_manager` with 100+ LED panels and 8 tracked cameras (verified 2026-05-31; tests/wave13_tests.rs::test_stage_manager_stress; skipped in debug mode)
- [x] Test `color::pipeline` color accuracy with known reference color checker values (verified 2026-05-31; tests/wave13_tests.rs::test_color_accuracy_delta_e2000)

## Documentation
- [ ] Document virtual production stage setup guide (LED panel arrangement, camera placement, calibration workflow)
- [ ] Add architecture diagram showing data flow from tracking system through render to LED output
- [ ] Document Unreal Engine integration protocol and message format in `unreal` module

## 0.1.8 Wave 13 Slice G — 2026-05-31
- [x] Flip stale TODOs: core_interop (L11), icvfx composite_multi_layer (L12)
- [x] Wire orphan icvfx/tiled.rs: pub mod tiled + pub use TiledCompositor in icvfx/mod.rs
- [x] Wire orphan math/simd_matrix.rs: pub mod simd_matrix + pub use Matrix4x4f in math/mod.rs
- [x] Rayon frustum render: led/render.rs panels.par_iter() parallel render with determinism test
- [x] Pixel-map LUT: GlobalPixelLut + PanelDesc + PixelMapper::lut() in pixel_mapping.rs
- [x] Render-layer LOD: LodConfig + RenderLayer::with_lod + render_at_lod + bilinear upsample in render_layer.rs
- [x] 6 tests: full_pipeline, genlock_jitter, calibration_round_trip, multicam_debounce, stage_stress, color_accuracy_delta_e2000
- [x] All 1101 tests pass, 0 clippy warnings

## 0.1.8 Wave 6 — 2026-05-29
- [x] Register 19 orphan modules in lib.rs + camera_rig cfix (verified 2026-05-29; 19 orphans wired, 0 warnings)
