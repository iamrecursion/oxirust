//! Synchronous unit tests for `TimelineRenderer` dirty-region tracking and
//! parallel-render switch.  These complement the async integration tests in
//! `tests/incremental_render.rs`.

use std::sync::Arc;

use oximedia_edit::{RenderConfig, Timeline, TimelineRenderer};

fn make_renderer() -> TimelineRenderer {
    let timeline = Arc::new(Timeline::default_settings());
    TimelineRenderer::new(timeline, RenderConfig::default())
}

// ─── dirty-region unit tests ─────────────────────────────────────────────────

/// With no dirty regions every position is considered dirty (no tracking active).
#[test]
fn test_no_dirty_regions_means_all_dirty() {
    let r = make_renderer();
    assert!(r.is_position_dirty(0), "no dirty list → always dirty");
    assert!(r.is_position_dirty(1000), "no dirty list → always dirty");
}

/// After marking a region only positions inside it are dirty.
#[test]
fn test_mark_dirty_region() {
    let mut r = make_renderer();
    r.mark_dirty(10, 50);
    assert!(
        r.is_position_dirty(10),
        "start of dirty region must be dirty"
    );
    assert!(r.is_position_dirty(25), "mid of dirty region must be dirty");
    // Frame 50 is the exclusive upper bound.
    assert!(
        !r.is_position_dirty(50),
        "exclusive upper bound must not be dirty"
    );
    assert!(
        !r.is_position_dirty(9),
        "frame before region must not be dirty"
    );
}

/// clear_dirty reverts to empty list which means all positions are dirty again.
#[test]
fn test_clear_dirty_reverts_to_all_dirty() {
    let mut r = make_renderer();
    r.mark_dirty(10, 50);
    assert!(!r.is_position_dirty(5)); // outside region → clean
    r.clear_dirty();
    assert!(
        r.is_position_dirty(5),
        "after clear, empty list → all dirty"
    );
}

/// force_full_redraw marks the entire timeline as dirty.
#[test]
fn test_force_full_redraw() {
    let mut r = make_renderer();
    r.mark_dirty(10, 50);
    assert!(!r.is_position_dirty(0));
    r.force_full_redraw();
    assert!(
        r.is_position_dirty(0),
        "after force_full_redraw all frames dirty"
    );
}

/// Overlapping dirty regions are coalesced into a single region.
#[test]
fn test_overlapping_regions_coalesced() {
    let mut r = make_renderer();
    r.mark_dirty(0, 30);
    r.mark_dirty(20, 60);
    // Coalesced into [0, 60).
    assert!(r.is_position_dirty(0));
    assert!(r.is_position_dirty(59));
    assert!(!r.is_position_dirty(60));
}

// ─── use_parallel getter / setter ────────────────────────────────────────────

#[test]
fn test_use_parallel_default_is_false() {
    let r = make_renderer();
    assert!(!r.use_parallel(), "default must be false");
}

#[test]
fn test_set_use_parallel_roundtrip() {
    let mut r = make_renderer();
    r.set_use_parallel(true);
    assert!(r.use_parallel());
    r.set_use_parallel(false);
    assert!(!r.use_parallel());
}

// ─── prefetch initial state ───────────────────────────────────────────────────

#[test]
fn test_prefetch_playhead_initially_zero() {
    let r = make_renderer();
    assert_eq!(r.prefetch_playhead(), 0, "initial playhead should be 0");
}
