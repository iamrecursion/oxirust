//! Integration tests for scene_detection (L39).
//!
//! Tests verify:
//! 1. Hard-cut detection: 30 dark frames followed by 30 bright frames triggers a scene change.
//! 2. Static-scene: 60 identical frames produce no scene change.

use oximedia_video::scene_detection::{SceneChangeDetector, SceneDetectionMethod};

/// Build a YUV420 planar frame where the luma plane is `luma_val` and
/// chroma planes are neutral (128).
fn make_yuv420(width: u32, height: u32, luma_val: u8) -> Vec<u8> {
    let y_size = (width * height) as usize;
    let uv_size = ((width / 2) * (height / 2)) as usize;
    let mut buf = Vec::with_capacity(y_size + 2 * uv_size);
    buf.extend(std::iter::repeat_n(luma_val, y_size));
    buf.extend(std::iter::repeat_n(128u8, uv_size)); // U
    buf.extend(std::iter::repeat_n(128u8, uv_size)); // V
    buf
}

#[test]
fn test_scene_detection_hard_cut_detected() {
    let width = 64u32;
    let height = 64u32;
    let mut detector = SceneChangeDetector::new(SceneDetectionMethod::ThresholdBased, 0.3, 5);

    let dark_frame = make_yuv420(width, height, 10);
    let bright_frame = make_yuv420(width, height, 245);

    let mut detected_cut = false;

    // Feed 5 dark frames to warm up history.
    for i in 0u64..5 {
        detector.push_frame(&dark_frame, i, width, height);
    }
    // Feed 30 bright frames — a scene change should fire on one of these.
    for i in 5u64..35 {
        if let Some(_change) = detector.push_frame(&bright_frame, i, width, height) {
            detected_cut = true;
        }
    }

    assert!(
        detected_cut,
        "hard cut from dark (luma=10) to bright (luma=245) was not detected"
    );
}

#[test]
fn test_scene_detection_no_cut_static_scene() {
    let width = 64u32;
    let height = 64u32;
    let mut detector = SceneChangeDetector::new(SceneDetectionMethod::ThresholdBased, 0.3, 5);

    let frame = make_yuv420(width, height, 128);
    let mut any_cut = false;

    for i in 0u64..60 {
        if let Some(_change) = detector.push_frame(&frame, i, width, height) {
            any_cut = true;
        }
    }

    assert!(
        !any_cut,
        "false positive: scene change detected in a static (uniform) sequence"
    );
}

#[test]
fn test_scene_detection_histogram_method_hard_cut() {
    let width = 32u32;
    let height = 32u32;
    let mut detector = SceneChangeDetector::new(SceneDetectionMethod::HistogramBased, 0.3, 4);

    let dark = make_yuv420(width, height, 5);
    let bright = make_yuv420(width, height, 250);
    let mut cut_found = false;

    for i in 0u64..4 {
        detector.push_frame(&dark, i, width, height);
    }
    for i in 4u64..20 {
        if detector.push_frame(&bright, i, width, height).is_some() {
            cut_found = true;
        }
    }

    assert!(
        cut_found,
        "histogram-based detector missed hard cut from dark to bright"
    );
}

#[test]
fn test_scene_detection_adaptive_no_false_positive_on_slow_fade() {
    // Gradually increase luma — this should not fire for a tight threshold.
    let width = 32u32;
    let height = 32u32;
    // Use a generous threshold (0.9) so that a slow fade (1 luma unit per frame) never trips it.
    let mut detector = SceneChangeDetector::new(SceneDetectionMethod::Adaptive, 0.9, 5);

    let mut any_cut = false;
    for i in 0u64..60 {
        let luma = (100u8).saturating_add((i as u8).min(60));
        let frame = make_yuv420(width, height, luma);
        if detector.push_frame(&frame, i, width, height).is_some() {
            any_cut = true;
        }
    }

    assert!(
        !any_cut,
        "slow fade triggered a false-positive scene change with threshold=0.9"
    );
}
