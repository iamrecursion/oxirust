//! Smoke tests for orphan modules registered in Wave 4 Slice E.
//!
//! Each test verifies that a module compiles and that the most fundamental
//! invariant of its main type holds.

#[cfg(test)]
mod orphan_smoke_tests {
    // ── dirty_region ────────────────────────────────────────────────────────

    #[test]
    fn smoke_dirty_region() {
        use oximedia_graphics::dirty_region::{DirtyRect, DirtyRegionTracker};
        let mut tracker = DirtyRegionTracker::new(1920, 1080);
        tracker.mark(DirtyRect::new(100, 50, 200, 80));
        assert!(!tracker.regions().is_empty());
        tracker.clear();
        assert!(tracker.regions().is_empty());
    }

    // ── draw_batch ──────────────────────────────────────────────────────────

    #[test]
    fn smoke_draw_batch() {
        use oximedia_graphics::draw_batch::{BlendMode, DrawBatch, DrawCommand, FillRect};
        let mut batch = DrawBatch::new();
        batch.push(DrawCommand::Rect(FillRect {
            x: 0.0,
            y: 0.0,
            width: 100.0,
            height: 50.0,
            color: [255, 0, 0, 255],
            layer: 0,
            blend: BlendMode::Over,
        }));
        let cmds = batch.flush();
        assert_eq!(cmds.len(), 1);
    }

    // ── image_filter ────────────────────────────────────────────────────────

    #[test]
    fn smoke_image_filter() {
        use oximedia_graphics::image_filter::box_blur;
        let mut pixels = vec![128u8; 8 * 8 * 4];
        box_blur(&mut pixels, 8, 8, 1);
        // After blur, pixel count should be unchanged
        assert_eq!(pixels.len(), 8 * 8 * 4);
    }

    // ── layout_margins ──────────────────────────────────────────────────────

    #[test]
    fn smoke_layout_margins() {
        use oximedia_graphics::layout_margins::{BroadcastStandard, SafeAreaMargins};
        let margins = SafeAreaMargins::for_standard(BroadcastStandard::Smpte, 1920, 1080);
        let action = margins.action_safe_rect();
        let title = margins.title_safe_rect();
        // Action safe must contain title safe
        assert!(action.contains_rect(&title));
    }

    // ── porter_duff ─────────────────────────────────────────────────────────

    #[test]
    fn smoke_porter_duff() {
        use oximedia_graphics::porter_duff::{composite_layer, PorterDuffOp};
        let src = vec![255u8, 0, 0, 128, 255, 0, 0, 128]; // 2 semi-transparent red pixels
        let dst = vec![0u8, 255, 0, 255, 0, 255, 0, 255]; // 2 opaque green pixels
        let out = composite_layer(&src, &dst, 2, 1, PorterDuffOp::SrcOver);
        // Result should have non-zero red channel
        assert!(out[0] > 0, "red channel should be non-zero after SrcOver");
        assert_eq!(out.len(), 8);
    }

    // ── color_correction ────────────────────────────────────────────────────

    #[test]
    fn smoke_color_correction() {
        use oximedia_graphics::color_correction::ColorWheels;
        let wheels = ColorWheels::identity();
        let lut = wheels.build_lut();
        // Apply identity LUT to a test buffer — values should be unchanged
        let orig = vec![128u8; 4 * 4 * 4]; // 4x4 RGBA
        let mut pixels = orig.clone();
        lut.apply(&mut pixels, 4, 4);
        // Buffer length must be unchanged
        assert_eq!(pixels.len(), orig.len());
    }

    // ── color_temperature ───────────────────────────────────────────────────

    #[test]
    fn smoke_color_temperature() {
        use oximedia_graphics::color_temperature::KelvinRgb;
        let k = KelvinRgb::from_kelvin(6500.0);
        // D65 white point — all channels non-zero
        assert!(k.r > 0.0 && k.g > 0.0 && k.b > 0.0);
    }

    // ── drop_shadow ─────────────────────────────────────────────────────────

    #[test]
    fn smoke_drop_shadow() {
        use oximedia_graphics::drop_shadow::{DropShadowConfig, ShadowColor};
        let cfg = DropShadowConfig::default();
        let shadow_color = ShadowColor::default_shadow();
        // Shadow color is semi-transparent black
        assert_eq!(shadow_color.r, 0);
        assert_eq!(shadow_color.a, 128);
        // Config has valid default offset
        assert!(cfg.blur_radius >= 0.0);
    }

    // ── nine_slice ──────────────────────────────────────────────────────────

    #[test]
    fn smoke_nine_slice() {
        use oximedia_graphics::nine_slice::{NineSliceLayout, SliceInsets, SliceRegion};
        let insets = SliceInsets::uniform(10);
        let layout =
            NineSliceLayout::new(200, 100, 300, 150, insets).expect("valid nine-slice layout");
        let all_regions = SliceRegion::all();
        // Nine-slice always produces 9 regions
        assert_eq!(all_regions.len(), 9);
        let mappings = layout.all_mappings();
        assert_eq!(mappings.len(), 9);
    }

    // ── progress_bar ────────────────────────────────────────────────────────

    #[test]
    fn smoke_progress_bar() {
        use oximedia_graphics::progress_bar::{
            ProgressBarConfig, ProgressBarRenderer, ProgressBarState,
        };
        let config = ProgressBarConfig::default();
        let mut state = ProgressBarState::with_fill(0.5);
        state.advance(1.0, &config);
        let pixels = ProgressBarRenderer::render(&state, &config);
        // config.width * config.height * 4 = 400 * 40 * 4
        assert_eq!(pixels.len(), 400 * 40 * 4);
    }

    // ── gpu_compute ─────────────────────────────────────────────────────────

    #[test]
    fn smoke_gpu_compute() {
        use oximedia_graphics::gpu_compute::{check_gpu_status, GpuComputeConfig};
        let config = GpuComputeConfig::default();
        assert!(config.min_pixels_for_gpu > 0);
        // Just check it runs without panic; GPU may or may not be available
        let _status = check_gpu_status();
    }

    // ── glyph_cache ─────────────────────────────────────────────────────────

    #[test]
    fn smoke_glyph_cache() {
        use oximedia_graphics::font_metrics::SubPixelMode;
        use oximedia_graphics::glyph_cache::{GlyphCache, GlyphKey, RasterizedGlyph};
        let mut cache = GlyphCache::new(64);
        let key = GlyphKey::new('A', "Test", 24, SubPixelMode::Rgb);
        assert!(cache.get(&key).is_none());
        if let Some(glyph) = RasterizedGlyph::new(8, 12, vec![0u8; 8 * 12 * 4], 0, -2, 10.0) {
            cache.insert(key.clone(), glyph);
        }
        assert!(cache.get(&key).is_some());
    }

    // ── safe_area ───────────────────────────────────────────────────────────

    #[test]
    fn smoke_safe_area() {
        use oximedia_graphics::safe_area::{SafeAreaConfig, SafeAreaRenderer};
        let config = SafeAreaConfig::broadcast_preset(1920, 1080);
        let pixels = SafeAreaRenderer::render(&config);
        assert_eq!(pixels.len(), 1920 * 1080 * 4);
    }

    // ── scene_graph ─────────────────────────────────────────────────────────

    #[test]
    fn smoke_scene_graph() {
        use oximedia_graphics::scene_graph::{NodeContent, SceneGraph};
        let mut graph = SceneGraph::new();
        let root = graph.root();
        let child = graph.add_node(root, NodeContent::Empty);
        assert!(graph.get(root).is_some());
        assert!(child.is_some());
        if let Some(child_id) = child {
            assert!(graph.get(child_id).is_some());
        }
    }

    // ── text_effects ────────────────────────────────────────────────────────

    #[test]
    fn smoke_text_effects() {
        use oximedia_graphics::text_effects::TextOutlineConfig;
        let cfg = TextOutlineConfig::default();
        assert!(cfg.width_px > 0.0);
        assert_eq!(cfg.color, [0, 0, 0, 255]);
    }

    // ── text_effects_inline ─────────────────────────────────────────────────

    #[test]
    fn smoke_text_effects_inline() {
        use oximedia_graphics::text_effects_inline::{InlineShadowConfig, OutlineConfig};
        let outline = OutlineConfig {
            color: [255, 0, 0, 255],
            width: 2,
            behind_fill: true,
        };
        let shadow = InlineShadowConfig {
            color: [0, 0, 0, 200],
            offset_x: 2,
            offset_y: 2,
            blur_sigma: 1.5,
            opacity: 0.8,
        };
        assert_eq!(outline.width, 2);
        assert!(shadow.opacity > 0.0 && shadow.opacity <= 1.0);
    }

    // ── text_outline ────────────────────────────────────────────────────────

    #[test]
    fn smoke_text_outline() {
        use oximedia_graphics::text_outline::GlyphMask;
        let mask = GlyphMask::blank(10, 10);
        assert_eq!(mask.width, 10);
        assert_eq!(mask.height, 10);
        assert_eq!(mask.data.len(), 100);
    }

    // ── text_render_effects ─────────────────────────────────────────────────

    #[test]
    fn smoke_text_render_effects() {
        use oximedia_graphics::text_render_effects::{render_text_effects, TextRenderConfig};
        let w = 10u32;
        let h = 10u32;
        let alpha = vec![255u8; (w * h) as usize];
        let mut canvas = vec![0u8; (w * h * 4) as usize];
        let config = TextRenderConfig::new();
        render_text_effects(&mut canvas, &alpha, w, h, [200, 200, 200, 255], &config);
        // Canvas should be non-empty after rendering solid fill
        assert!(canvas.iter().any(|&v| v > 0));
    }

    // ── text_wrap ───────────────────────────────────────────────────────────

    #[test]
    fn smoke_text_wrap() {
        use oximedia_graphics::text_wrap::{wrap_text, JustifyMode, WrapConfig};
        let config = WrapConfig {
            max_width_px: 300.0,
            char_width_px: 10.0,
            justify: JustifyMode::Left,
            ..WrapConfig::default()
        };
        let lines = wrap_text("Hello world from oximedia graphics", &config);
        assert!(!lines.is_empty());
    }

    // ── transition_effect ───────────────────────────────────────────────────

    #[test]
    fn smoke_transition_effect() {
        use oximedia_graphics::transition_effect::{CrossDissolve, EasingCurve, TransitionEffect};
        let effect = TransitionEffect::CrossDissolve(CrossDissolve {
            easing: EasingCurve::Linear,
        });
        let frame = effect.sample(0.0);
        assert!((frame.outgoing_alpha - 1.0).abs() < 1e-4);
        let frame_end = effect.sample(1.0);
        assert!((frame_end.incoming_alpha - 1.0).abs() < 1e-4);
    }

    // ── motion_path ─────────────────────────────────────────────────────────

    #[test]
    fn smoke_motion_path() {
        use oximedia_graphics::motion_path::{EasingKind, MotionPath};
        let mut path = MotionPath::new(2.0);
        path.add_position_keyframe(0.0, 0.0, 0.0, EasingKind::Linear);
        path.add_position_keyframe(2.0, 100.0, 0.0, EasingKind::Linear);
        let pose = path.evaluate(1.0);
        // At t=1.0 (midpoint of 2s path), x should be around 50.0
        assert!(pose.x > 0.0 && pose.x < 100.0);
    }

    // ── scoreboard_datasource ───────────────────────────────────────────────

    #[test]
    fn smoke_scoreboard_datasource() {
        use oximedia_graphics::scoreboard_datasource::{
            DataSource, MemoryDataSource, ScoreField, ScoreUpdate, ScoreValue,
        };
        let mut source = MemoryDataSource::new();
        assert!(source.is_connected());
        source.push(ScoreUpdate {
            field: ScoreField::HomeScore,
            value: ScoreValue::Integer(3),
            timestamp_ms: 1000,
        });
        let updates = source.poll_updates();
        assert_eq!(updates.len(), 1);
        // Queue should be drained on next poll
        assert!(source.poll_updates().is_empty());
    }

    // ── video_wall ──────────────────────────────────────────────────────────

    #[test]
    fn smoke_video_wall() {
        use oximedia_graphics::video_wall::{VideoWallBuilder, WallPreset};
        let wall = VideoWallBuilder::from_preset(WallPreset::Grid2x2, 1920.0, 1080.0)
            .with_gap(0.005)
            .build()
            .expect("valid wall layout");
        assert_eq!(wall.cells.len(), 4);
    }
}
