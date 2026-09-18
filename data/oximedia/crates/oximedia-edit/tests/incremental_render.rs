//! Integration tests for incremental render, prefetch wiring, and parallel rendering.
//!
//! These tests exercise `TimelineRenderer` features added in 0.1.8 Wave 17 Slice F:
//! - Dirty-region tracking (`mark_dirty` / `clear_dirty` / `force_full_redraw`)
//! - Prefetch engine advancement after each render call
//! - Parallel multi-track rendering (`use_parallel = true`)

use std::sync::Arc;

use oximedia_edit::{Clip, ClipType, RenderConfig, Timeline, TimelineRenderer, TrackType};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn timeline_with_video_clip(clip_duration: i64) -> Arc<Timeline> {
    let mut tl = Timeline::default_settings();
    let ti = tl.add_track(TrackType::Video);
    let clip = Clip::new(1, ClipType::Video, 0, clip_duration);
    tl.add_clip(ti, clip).expect("add_clip ok");
    Arc::new(tl)
}

// ─────────────────────────────────────────────────────────────────────────────
// Incremental render — dirty-region skipping
// ─────────────────────────────────────────────────────────────────────────────

/// Positions outside the dirty region must not produce video (skipped by the
/// incremental-render guard inside `render_video_at`).
#[tokio::test]
async fn test_incremental_render_skips_unaffected_tracks() {
    let tl = timeline_with_video_clip(5000);
    let mut renderer = TimelineRenderer::new(tl, RenderConfig::default());

    // Mark only [100, 200) as dirty.
    renderer.mark_dirty(100, 200);

    // Position 50 — outside dirty region — video should be None.
    let frame = renderer
        .render_frame_at(50)
        .await
        .expect("render_frame_at must not error");
    assert!(
        frame.video.is_none(),
        "frame at position 50 is outside dirty region [100,200): video should be None"
    );

    // Position 150 — inside dirty region — video should be Some.
    let frame2 = renderer
        .render_frame_at(150)
        .await
        .expect("render_frame_at must not error");
    assert!(
        frame2.video.is_some(),
        "frame at position 150 is inside dirty region [100,200): video should be Some"
    );
}

/// Rendering at a position that was already cached does NOT bypass the dirty-
/// region check (the cache path returns early before the check).  A second call
/// to the same position returns the cached frame unchanged.
#[tokio::test]
async fn test_incremental_render_cached_frame_still_returned() {
    let tl = timeline_with_video_clip(5000);
    let mut renderer = TimelineRenderer::new(tl, RenderConfig::default());

    // First render with dirty region covering pos=100 → frame is computed and cached.
    renderer.mark_dirty(100, 200);
    let f1 = renderer
        .render_frame_at(100)
        .await
        .expect("first render ok");
    assert!(f1.video.is_some(), "first render should have video");

    // Second render — cache hit, no dirty-region check needed.
    let f2 = renderer
        .render_frame_at(100)
        .await
        .expect("second render ok");
    assert_eq!(
        f1.position, f2.position,
        "cached frame must be at same position"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Prefetch engine wiring
// ─────────────────────────────────────────────────────────────────────────────

/// After `render_frame_at(pos)` the prefetch engine's playhead must equal `pos`.
#[tokio::test]
async fn test_prefetch_advances_after_render() {
    let tl = Arc::new(Timeline::default_settings());
    let mut renderer = TimelineRenderer::new(tl, RenderConfig::default());

    let pos: i64 = 330;
    renderer.render_frame_at(pos).await.expect("render ok");

    assert_eq!(
        renderer.prefetch_playhead(),
        pos,
        "prefetch engine playhead must equal the rendered position"
    );
}

/// Successive render calls advance the prefetch playhead each time.
#[tokio::test]
async fn test_prefetch_tracks_successive_positions() {
    let tl = Arc::new(Timeline::default_settings());
    let mut renderer = TimelineRenderer::new(tl, RenderConfig::default());

    for pos in [0i64, 33, 66, 99] {
        renderer.render_frame_at(pos).await.expect("render ok");
        assert_eq!(
            renderer.prefetch_playhead(),
            pos,
            "playhead should equal last rendered position {pos}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Parallel render == sequential render
// ─────────────────────────────────────────────────────────────────────────────

/// A 2-track video timeline rendered with `use_parallel=true` must produce a
/// video frame with the same dimensions as `use_parallel=false`.
///
/// Exact pixel equality is not guaranteed because the two code paths use
/// different compositor pipelines (HdrCompositor sequential vs RGBA8 layers),
/// but dimensions and plane count must match.
#[tokio::test]
async fn test_parallel_render_matches_sequential() {
    let mut tl = Timeline::default_settings();
    for i in 0..2u64 {
        let ti = tl.add_track(TrackType::Video);
        let clip = Clip::new(i + 1, ClipType::Video, 0, 1000);
        tl.add_clip(ti, clip).expect("add clip ok");
    }
    let tl = Arc::new(tl);

    let config = RenderConfig {
        width: 16,
        height: 16,
        ..Default::default()
    };

    // Sequential render.
    let mut seq_renderer = TimelineRenderer::new(tl.clone(), config.clone());
    seq_renderer.mark_dirty(0, 1000);
    let seq_frame = seq_renderer
        .render_frame_at(0)
        .await
        .expect("sequential render ok");

    // Parallel render.
    let mut par_renderer = TimelineRenderer::new(tl.clone(), config.clone());
    par_renderer.mark_dirty(0, 1000);
    par_renderer.set_use_parallel(true);
    let par_frame = par_renderer
        .render_frame_at(0)
        .await
        .expect("parallel render ok");

    assert!(seq_frame.video.is_some(), "sequential must produce video");
    assert!(par_frame.video.is_some(), "parallel must produce video");

    let seq_video = seq_frame.video.unwrap();
    let par_video = par_frame.video.unwrap();

    assert_eq!(seq_video.width, par_video.width, "width must match");
    assert_eq!(seq_video.height, par_video.height, "height must match");
    assert_eq!(
        seq_video.planes.len(),
        par_video.planes.len(),
        "plane count must match"
    );
}

/// Parallel render with a single-track timeline should also succeed.
#[tokio::test]
async fn test_parallel_render_single_track() {
    let tl = timeline_with_video_clip(1000);
    let config = RenderConfig {
        width: 8,
        height: 8,
        ..Default::default()
    };
    let mut renderer = TimelineRenderer::new(tl, config);
    renderer.mark_dirty(0, 1000);
    renderer.set_use_parallel(true);

    let frame = renderer
        .render_frame_at(0)
        .await
        .expect("parallel render ok");
    assert!(
        frame.video.is_some(),
        "single-track parallel must produce video"
    );
}
