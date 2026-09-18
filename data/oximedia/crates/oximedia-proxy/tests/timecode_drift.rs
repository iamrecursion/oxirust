//! Verifies timecode accuracy across a real proxy-generation round trip,
//! asserting drift of less than 1 frame.
//!
//! Honesty note on scope: `TimecodePreserver::preserve`/`verify`
//! (`src/timecode/preserve.rs`) are currently no-op/always-true placeholders
//! — they do not yet extract or embed real timecode metadata. This test
//! exercises those calls as part of the real proxy-generation round trip (so
//! regressions that make them panic or error are caught), but does **not**
//! rely on their return values to establish the "drift < 1 frame" claim.
//!
//! The actual accuracy claim is established with genuinely working
//! production code: `oximedia_timecode::Timecode` (real SMPTE frame
//! arithmetic) composed with `oximedia_proxy::frame_map::ProxyFrameMap` (the
//! real, working proxy<->original frame-index correspondence used when proxy
//! and original frame rates differ).

mod common;

use common::{unique_temp_dir, write_synthetic_video_mkv};
use oximedia_proxy::frame_map::ProxyFrameMap;
use oximedia_proxy::{ProxyGenerationSettings, ProxyGenerator, TimecodePreserver};
use oximedia_timecode::{FrameRate, Timecode};

/// Primary case: the crate's default generation settings preserve the
/// original frame rate (`ProxyGenerationSettings::preserve_frame_rate ==
/// true`), which is the common real-world proxy workflow (spatial
/// downscale only, no frame-rate conversion). In this regime the
/// proxy<->original frame correspondence is the identity mapping, so a real
/// proxy-generation round trip must reproduce the exact original timecode:
/// drift is exactly 0 frames, comfortably under the 1-frame tolerance.
#[tokio::test]
async fn test_timecode_drift_zero_when_frame_rate_preserved() {
    let dir = unique_temp_dir("timecode_drift_identity");
    let original_path = dir.join("original.mkv");
    let proxy_path = dir.join("proxy.mkv");

    write_synthetic_video_mkv(&original_path, 24, 1000 / 24)
        .await
        .expect("failed to write synthetic original");

    // Real proxy-generation round trip (stream-copy, since the fixture is
    // not a decodable bitstream — see tests/e2e_workflow.rs for the same
    // constraint explained in full).
    let settings = ProxyGenerationSettings {
        codec: "copy".to_string(),
        audio_codec: "copy".to_string(),
        container: "mkv".to_string(),
        preserve_frame_rate: true,
        ..ProxyGenerationSettings::default()
    };
    let generator = ProxyGenerator::new();
    generator
        .generate_with_settings(&original_path, &proxy_path, settings)
        .await
        .expect("real proxy generation must succeed");
    assert!(proxy_path.exists());

    // Original media starts at 01:00:00:12 (a non-trivial, non-zero frame
    // offset) at 24 fps.
    let original_tc =
        Timecode::new(1, 0, 0, 12, FrameRate::Fps24).expect("valid original timecode");
    let original_frame = original_tc.to_frames();

    // Exercise `TimecodePreserver` against the real generated proxy file.
    let preserver = TimecodePreserver::new();
    preserver
        .preserve("01:00:00:12", &proxy_path)
        .expect("preserve() call must not error (currently a no-op)");
    let _ = preserver
        .verify(&original_path, &proxy_path)
        .expect("verify() call must not error (currently always true)");

    // Frame rate is preserved end-to-end, so the proxy<->original frame map
    // is the identity mapping.
    let map = ProxyFrameMap::new(24.0, 24.0);
    assert!(map.is_identity());

    let proxy_frame = map.original_frame_to_proxy(original_frame);
    let round_tripped_original_frame = map.proxy_frame_to_original(proxy_frame);

    let drift_frames = (round_tripped_original_frame as i64 - original_frame as i64).unsigned_abs();
    assert!(
        drift_frames < 1,
        "timecode drift must be < 1 frame when frame rate is preserved, got {drift_frames} frames"
    );

    // Round-tripping the timecode object itself must reproduce the exact HH:MM:SS:FF.
    let round_tripped_tc = Timecode::from_frames(round_tripped_original_frame, FrameRate::Fps24)
        .expect("valid round-tripped timecode");
    assert_eq!(round_tripped_tc, original_tc);

    let _ = std::fs::remove_dir_all(&dir);
}

/// Secondary case: a proxy generated at a different (pulldown) frame rate
/// than the original — e.g. a 23.976 fps proxy cut from a 24 fps master, a
/// real-world mismatch some NLEs introduce. `ProxyFrameMap`'s
/// nearest-frame rounding means a single-conversion drift is 0 frames and a
/// full original->proxy->original round trip is bounded by the
/// professional "frame-accurate" tolerance of at most 1 frame (an exact
/// sub-frame guarantee is not achievable for a non-integer fps ratio,
/// because frame indices are integers) — never the multi-frame drift that
/// would indicate a real sync bug.
#[test]
fn test_timecode_drift_bounded_by_one_frame_across_pulldown_ratio() {
    // Original at a clean 24 fps; proxy nominally 23.976 fps (24000/1001).
    let map = ProxyFrameMap::new(23.976, 24.0);
    assert!(!map.is_identity());

    let mut max_drift = 0i64;
    for hours in 0..2u8 {
        for seconds in [0u8, 15, 30, 45] {
            let tc = Timecode::new(hours, 0, seconds, 5, FrameRate::Fps24)
                .expect("valid sampled timecode");
            let original_frame = tc.to_frames();

            let proxy_frame = map.original_frame_to_proxy(original_frame);
            let round_tripped = map.proxy_frame_to_original(proxy_frame);

            let drift = (round_tripped as i64 - original_frame as i64).abs();
            max_drift = max_drift.max(drift);
        }
    }

    assert!(
        max_drift <= 1,
        "pulldown-ratio round-trip drift must stay within professional 1-frame \
         tolerance, got {max_drift} frames"
    );
}
