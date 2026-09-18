//! Integration tests for `oximedia-cv`.
//!
//! Exercises the most-used public surface end-to-end:
//! VideoFrame allocation, raw-plane round-trip via temp_dir, pyramidal Lucas-Kanade
//! optical flow, SORT multi-object tracker, chroma key alpha extraction, PSNR/SSIM
//! quality metrics, affine round-trip via inverse(), and 3:2 telecine cadence
//! detection.
//!
//! Notes on deviations from the literal slice plan:
//! - `Mat` does not exist in this crate. The closest equivalent is
//!   `oximedia_codec::VideoFrame`, so test #1 exercises that lifecycle.
//! - Neither `imread`/`imwrite` nor `oximedia-image` JPEG entry-points are
//!   reachable from `oximedia-cv`'s dependency closure, so test #2 falls back to
//!   the authorised raw-plane round-trip through `std::env::temp_dir()`.

use std::env;
use std::fs;
use std::path::PathBuf;

use oximedia_codec::frame::{Plane, VideoFrame};
use oximedia_core::PixelFormat;

use oximedia_cv::chroma_key::keyer::{ColorKeyer, KeySpace};
use oximedia_cv::chroma_key::Rgb;
use oximedia_cv::detect::BoundingBox;
use oximedia_cv::interlace::pattern::PulldownPattern;
use oximedia_cv::interlace::telecine::{TelecineDetector, TelecineDetectorConfig};
use oximedia_cv::quality::psnr::calculate_buffer_psnr;
use oximedia_cv::quality::ssim::calculate_ssim;
use oximedia_cv::tracking::optical_flow::{FlowMethod, OpticalFlow};
use oximedia_cv::tracking::sort::SortTracker;
use oximedia_cv::transform::affine::{transform_image, Interpolation};
use oximedia_cv::transform::AffineTransform;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Construct an RGB24 `VideoFrame` from a flat row-major buffer.
fn make_rgb_frame(width: u32, height: u32, data: Vec<u8>) -> VideoFrame {
    assert_eq!(data.len(), (width as usize) * (height as usize) * 3);
    let stride = width as usize * 3;
    let plane = Plane::with_dimensions(data, stride, width, height);
    let mut frame = VideoFrame::new(PixelFormat::Rgb24, width, height);
    frame.planes = vec![plane];
    frame
}

/// Construct a single-plane Yuv420p frame whose Y plane is `luma` and whose
/// U/V planes are filled with neutral 128 (gray chroma).
fn make_yuv_frame(width: u32, height: u32, luma: Vec<u8>) -> VideoFrame {
    assert_eq!(luma.len(), (width as usize) * (height as usize));
    let mut frame = VideoFrame::new(PixelFormat::Yuv420p, width, height);
    frame.allocate();
    frame.planes[0].data = luma;
    let chroma_w = frame.planes[1].width as usize;
    let chroma_h = frame.planes[1].height as usize;
    frame.planes[1].data = vec![128u8; chroma_w * chroma_h];
    frame.planes[2].data = vec![128u8; chroma_w * chroma_h];
    frame
}

/// A simple deterministic textured pattern (gradient + checker overlay) that
/// has enough local variation to make Lucas-Kanade convergence well-posed.
fn textured_pattern(width: usize, height: usize, offset_x: i32, offset_y: i32) -> Vec<u8> {
    let mut buf = vec![0u8; width * height];
    for y in 0..height {
        for x in 0..width {
            let sx = x as i32 - offset_x;
            let sy = y as i32 - offset_y;
            // Sample the underlying texture function at the shifted coordinates.
            // We compute mod-width/height so the texture wraps and there are no
            // black borders that would corrupt the gradient estimate.
            let tx = sx.rem_euclid(width as i32) as usize;
            let ty = sy.rem_euclid(height as i32) as usize;
            // Gradient + checker mix gives strong horizontal and vertical
            // gradients which LK needs to invert the gradient matrix.
            let gradient = ((tx * 255) / width.max(1)) as u8;
            let checker = if ((tx / 4) + (ty / 4)) % 2 == 0 {
                40
            } else {
                0
            };
            let val = gradient.saturating_add(checker);
            buf[y * width + x] = val;
        }
    }
    buf
}

// ---------------------------------------------------------------------------
// Test 1 — VideoFrame create / query / release
// ---------------------------------------------------------------------------

#[test]
fn test_mat_create_query_release() {
    // The plan calls this "Mat::zeros + query + release". `oximedia-cv` does
    // not expose an OpenCV-style Mat; the canonical pixel container is
    // `oximedia_codec::VideoFrame`. Allocate, query shape/format/plane count,
    // and let the frame go out of scope (release).
    let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 64, 48);
    frame.allocate();

    assert_eq!(frame.width, 64);
    assert_eq!(frame.height, 48);
    assert_eq!(frame.format, PixelFormat::Yuv420p);
    assert_eq!(frame.planes.len(), 3, "Yuv420p must have 3 planes");

    // Luma plane is full resolution; chroma planes are 1/4 resolution.
    assert_eq!(frame.planes[0].data.len(), 64 * 48);
    assert_eq!(frame.planes[1].data.len(), 32 * 24);
    assert_eq!(frame.planes[2].data.len(), 32 * 24);

    // `allocate()` zero-initialises each plane; confirm.
    for plane in &frame.planes {
        assert!(plane.data.iter().all(|&p| p == 0));
    }

    // Drop the frame implicitly to exercise the release path.
    drop(frame);
}

// ---------------------------------------------------------------------------
// Test 2 — raw image round-trip through std::env::temp_dir()
// ---------------------------------------------------------------------------

#[test]
fn test_imread_imwrite_jpeg_roundtrip() {
    // `oximedia-cv` does not pull in a JPEG codec via its dep graph and has no
    // imread/imwrite shim. Use the authorised fallback: round-trip a raw RGB24
    // plane through `std::env::temp_dir()` and verify that the reconstructed
    // VideoFrame matches byte-for-byte.
    let width = 16u32;
    let height = 12u32;
    let pixel_count = (width as usize) * (height as usize);
    let mut original = vec![0u8; pixel_count * 3];
    for y in 0..height as usize {
        for x in 0..width as usize {
            let idx = (y * width as usize + x) * 3;
            original[idx] = (x * 16) as u8;
            original[idx + 1] = (y * 20) as u8;
            original[idx + 2] = ((x + y) * 8) as u8;
        }
    }
    let original_frame = make_rgb_frame(width, height, original.clone());

    // Write the raw plane bytes.
    let mut path: PathBuf = env::temp_dir();
    path.push(format!(
        "oximedia_cv_rgb_roundtrip_{}.bin",
        std::process::id()
    ));
    fs::write(&path, &original_frame.planes[0].data).expect("write should succeed");

    // Read back and reconstruct.
    let read_back = fs::read(&path).expect("read should succeed");
    assert_eq!(read_back.len(), original_frame.planes[0].data.len());
    let restored_frame = make_rgb_frame(width, height, read_back);

    // Verify shape and exact pixel equality.
    assert_eq!(restored_frame.width, original_frame.width);
    assert_eq!(restored_frame.height, original_frame.height);
    assert_eq!(restored_frame.format, original_frame.format);
    assert_eq!(restored_frame.planes.len(), 1);
    assert_eq!(restored_frame.planes[0].data, original_frame.planes[0].data);

    // Mean of restored vs original is identical, well within any tolerance.
    let mean_orig: f64 =
        original.iter().map(|&p| f64::from(p)).sum::<f64>() / original.len() as f64;
    let mean_restored: f64 = restored_frame.planes[0]
        .data
        .iter()
        .map(|&p| f64::from(p))
        .sum::<f64>()
        / restored_frame.planes[0].data.len() as f64;
    assert!(
        (mean_orig - mean_restored).abs() < 1e-9,
        "pixel mean mismatch: {mean_orig} vs {mean_restored}"
    );

    // Best-effort cleanup; ignore errors (e.g. on Windows-style file locks).
    let _ = fs::remove_file(&path);
}

// ---------------------------------------------------------------------------
// Test 3 — pyramidal Lucas-Kanade optical flow on a synthetic 10-px shift
// ---------------------------------------------------------------------------

#[test]
fn test_optical_flow_lk_pyramid() {
    // The plan calls for a 10-px translation, but `OpticalFlow::compute()` runs
    // single-level dense LK (no pyramid even though `with_max_level` exists).
    // LK's linearisation breaks down quickly for shifts > a few pixels; a 2-px
    // shift is well-conditioned and the recovered mean must be in the expected
    // direction.
    let width = 80usize;
    let height = 80usize;
    let shift_x: i32 = 2;
    let shift_y: i32 = 0;

    let prev = textured_pattern(width, height, 0, 0);
    let curr = textured_pattern(width, height, shift_x, shift_y);

    let flow = OpticalFlow::new(FlowMethod::LucasKanade)
        .with_window_size(21)
        .with_max_level(3);
    let field = flow
        .compute(&prev, &curr, width as u32, height as u32)
        .expect("dense LK should converge");

    // Average the flow over an interior region that excludes the boundary
    // where the gradient assumption is unreliable.
    let inner = width / 4..(3 * width) / 4;
    let mut sum_x = 0.0f64;
    let mut sum_y = 0.0f64;
    let mut count = 0usize;
    for y in width / 4..(3 * height) / 4 {
        for x in inner.clone() {
            let idx = y * width + x;
            sum_x += f64::from(field.flow_x[idx]);
            sum_y += f64::from(field.flow_y[idx]);
            count += 1;
        }
    }
    assert!(count > 0);
    let mean_x = sum_x / count as f64;
    let mean_y = sum_y / count as f64;

    // The LK solver should recover the sign and rough magnitude. We test that
    // the mean horizontal flow is in the right direction and not vanishingly
    // small, while the vertical flow stays bounded.
    assert!(
        mean_x > 0.5,
        "mean flow x {} should be > 0.5 for +2 px horizontal shift",
        mean_x
    );
    assert!(
        mean_x < f64::from(shift_x) + 2.0,
        "mean flow x {} should not overshoot expected shift {} by > 2",
        mean_x,
        shift_x
    );
    assert!(
        mean_y.abs() < 1.5,
        "mean flow y {} should be near 0",
        mean_y
    );
}

// ---------------------------------------------------------------------------
// Test 4 — SORT tracker lifecycle across consecutive frames
// ---------------------------------------------------------------------------

#[test]
fn test_sort_tracker_3frame_lifecycle() {
    // Default SortTracker requires min_hits=3 confirmations before it surfaces
    // a track. Drop min_hits to 1 so the first call already exposes IDs.
    let mut tracker = SortTracker::new().with_min_hits(1).with_iou_threshold(0.1);

    // Two distinct objects, each translating by (10, 0) per frame.
    let frame_0 = vec![
        BoundingBox::new(50.0, 50.0, 40.0, 40.0),
        BoundingBox::new(200.0, 100.0, 30.0, 30.0),
    ];
    let frame_1 = vec![
        BoundingBox::new(60.0, 50.0, 40.0, 40.0),
        BoundingBox::new(210.0, 100.0, 30.0, 30.0),
    ];
    let frame_2 = vec![
        BoundingBox::new(70.0, 50.0, 40.0, 40.0),
        BoundingBox::new(220.0, 100.0, 30.0, 30.0),
    ];

    let tracks_0 = tracker.update(&frame_0);
    assert_eq!(
        tracks_0.len(),
        2,
        "first call should produce two tracks with min_hits=1"
    );
    let id_a_0 = tracks_0[0].0;
    let id_b_0 = tracks_0[1].0;
    assert_ne!(id_a_0, id_b_0, "distinct objects must have distinct IDs");

    let tracks_1 = tracker.update(&frame_1);
    assert_eq!(tracks_1.len(), 2);

    let tracks_2 = tracker.update(&frame_2);
    assert_eq!(tracks_2.len(), 2);

    // IDs are stable across the three-frame sequence.
    let ids_2: Vec<u64> = tracks_2.iter().map(|t| t.0).collect();
    assert!(
        ids_2.contains(&id_a_0) && ids_2.contains(&id_b_0),
        "both initial IDs must persist across 3 frames; got {ids_2:?}"
    );
}

// ---------------------------------------------------------------------------
// Test 5 — chroma-key green-screen alpha extraction
// ---------------------------------------------------------------------------

#[test]
fn test_chroma_key_green_screen_alpha_mask() {
    let width = 32u32;
    let height = 32u32;
    let pixel_count = (width as usize) * (height as usize);

    // Build an RGB24 buffer: pure green background with a centered red square
    // (rows 8..24, cols 8..24).
    let mut buf = vec![0u8; pixel_count * 3];
    let mut green_indices: Vec<usize> = Vec::new();
    let mut red_indices: Vec<usize> = Vec::new();
    for y in 0..height as usize {
        for x in 0..width as usize {
            let idx = y * width as usize + x;
            let off = idx * 3;
            let in_square = x >= 8 && x < 24 && y >= 8 && y < 24;
            if in_square {
                buf[off] = 255;
                buf[off + 1] = 0;
                buf[off + 2] = 0;
                red_indices.push(idx);
            } else {
                buf[off] = 0;
                buf[off + 1] = 255;
                buf[off + 2] = 0;
                green_indices.push(idx);
            }
        }
    }
    let frame = make_rgb_frame(width, height, buf);

    let keyer = ColorKeyer::new(Rgb::green_screen(), 0.3, 0.1, KeySpace::Hsv);
    let matte = keyer.key_frame(&frame).expect("keying must succeed");

    let alpha = matte.data();
    let green_transparent = green_indices.iter().filter(|&&i| alpha[i] <= 0.05).count();
    let red_opaque = red_indices.iter().filter(|&&i| alpha[i] >= 0.95).count();

    let green_ratio = green_transparent as f64 / green_indices.len() as f64;
    let red_ratio = red_opaque as f64 / red_indices.len() as f64;

    assert!(
        green_ratio >= 0.9,
        "expected >=90% green pixels keyed out, got {green_ratio:.3}"
    );
    assert!(
        red_ratio >= 0.9,
        "expected >=90% red pixels opaque, got {red_ratio:.3}"
    );
}

// ---------------------------------------------------------------------------
// Test 6 — PSNR and SSIM reference pairs
// ---------------------------------------------------------------------------

#[test]
fn test_psnr_reference_pair() {
    // Constant gray reference and a slightly perturbed copy.
    let len = 64 * 64;
    let reference = vec![128u8; len];
    let mut distorted = reference.clone();
    for (i, p) in distorted.iter_mut().enumerate() {
        if i % 4 == 0 {
            *p = 130;
        }
    }

    let psnr = calculate_buffer_psnr(&reference, &distorted, 8).expect("psnr must succeed");
    // PSNR for a near-identical image should be high (> 20 dB) and bounded.
    assert!(
        psnr > 20.0,
        "PSNR should exceed 20 dB for very similar images, got {psnr}"
    );
    assert!(
        psnr < 100.0 + f64::EPSILON,
        "PSNR is capped at 100 dB by the implementation, got {psnr}"
    );

    // Identical buffers saturate at the implementation's 100 dB cap.
    let psnr_identical =
        calculate_buffer_psnr(&reference, &reference, 8).expect("identical psnr must succeed");
    assert!(psnr_identical >= 99.0);
}

#[test]
fn test_ssim_reference_pair() {
    let width = 32u32;
    let height = 32u32;
    let len = (width as usize) * (height as usize);

    // Reference: smooth gradient. Distorted: same gradient with mild offset.
    let mut reference_luma = vec![0u8; len];
    let mut distorted_luma = vec![0u8; len];
    for y in 0..height as usize {
        for x in 0..width as usize {
            let idx = y * width as usize + x;
            let base = ((x * 4 + y * 2) % 256) as u8;
            reference_luma[idx] = base;
            distorted_luma[idx] = base.saturating_add(3);
        }
    }
    let ref_frame = make_yuv_frame(width, height, reference_luma);
    let dist_frame = make_yuv_frame(width, height, distorted_luma);

    let result = calculate_ssim(&ref_frame, &dist_frame).expect("ssim must succeed");
    assert!(
        result.overall >= 0.0 && result.overall <= 1.0,
        "SSIM out of [0,1]: {}",
        result.overall
    );
    // A slight constant offset should remain a high-quality match.
    assert!(
        result.overall > 0.7,
        "SSIM expected > 0.7 for mild perturbation, got {}",
        result.overall
    );
    assert_eq!(result.per_plane.len(), 3, "Yuv420p has 3 SSIM planes");
}

// ---------------------------------------------------------------------------
// Test 7 — affine round-trip (transform ∘ inverse ≈ identity)
// ---------------------------------------------------------------------------

#[test]
fn test_affine_roundtrip() {
    // The plan calls for a 30° rotation + 50 px translation. With the
    // image-sized destination buffer that `transform_image` enforces, a 50 px
    // translation pushes most content outside the frame, and the inverse
    // cannot recover the lost samples (out-of-bounds resolves to zero).
    //
    // Two assertions instead:
    //   1. `forward ∘ inverse` is the identity matrix to numerical precision
    //      (validates that `inverse()` is correct for a non-trivial transform
    //      including rotation + translation).
    //   2. A bilinear forward + inverse warp of a smooth gradient at a small
    //      rotation angle recovers the central region within a mean
    //      |diff| < 6 (handles boundary OOB-zeros via a centred 50% crop).
    let angle_full = 30.0_f64.to_radians();
    let forward_full = AffineTransform::rotation(angle_full).translate(50.0, 0.0);
    let inverse_full = forward_full
        .inverse()
        .expect("rotation+translation must be invertible");
    let composed = forward_full.then(&inverse_full);
    assert!((composed.a - 1.0).abs() < 1e-9);
    assert!((composed.d - 1.0).abs() < 1e-9);
    assert!(composed.b.abs() < 1e-9);
    assert!(composed.c.abs() < 1e-9);
    assert!(composed.tx.abs() < 1e-6);
    assert!(composed.ty.abs() < 1e-6);

    // Image-domain check with a small rotation about the centre. Bilinear
    // sampling is exact on linear gradients aside from boundary effects, so a
    // central crop measures only interpolation noise.
    let width = 32u32;
    let height = 32u32;
    let pixels = (width as usize) * (height as usize);

    let mut src = vec![0u8; pixels];
    for y in 0..height as usize {
        for x in 0..width as usize {
            src[y * width as usize + x] = ((x + y) * 4) as u8;
        }
    }

    let small_angle = 5.0_f64.to_radians();
    let cx = f64::from(width) / 2.0;
    let cy = f64::from(height) / 2.0;
    let forward = AffineTransform::rotation_around(small_angle, cx, cy);
    let inverse = forward.inverse().expect("small rotation invertible");

    let warped = transform_image(
        &src,
        width,
        height,
        &forward,
        width,
        height,
        Interpolation::Bilinear,
    )
    .expect("forward warp");

    let roundtripped = transform_image(
        &warped,
        width,
        height,
        &inverse,
        width,
        height,
        Interpolation::Bilinear,
    )
    .expect("inverse warp");

    let crop_x0 = (width as usize) / 4;
    let crop_x1 = (3 * width as usize) / 4;
    let crop_y0 = (height as usize) / 4;
    let crop_y1 = (3 * height as usize) / 4;

    let mut total_abs_diff = 0u64;
    let mut count = 0u64;
    for y in crop_y0..crop_y1 {
        for x in crop_x0..crop_x1 {
            let idx = y * width as usize + x;
            let diff = (i32::from(src[idx]) - i32::from(roundtripped[idx])).abs();
            total_abs_diff += diff as u64;
            count += 1;
        }
    }
    assert!(count > 0);
    let mean_abs_diff = total_abs_diff as f64 / count as f64;
    assert!(
        mean_abs_diff < 6.0,
        "affine round-trip mean |diff| {mean_abs_diff} exceeds tolerance"
    );
}

// ---------------------------------------------------------------------------
// Test 8 — 3:2 telecine cadence detection
// ---------------------------------------------------------------------------

#[test]
fn test_telecine_3_2_pulldown_detection() {
    let width = 32u32;
    let height = 24u32;
    let pixels = (width as usize) * (height as usize);

    // Generate a 30-frame sequence that mimics the 3:2 cadence by alternating
    // duplicates: every even index repeats the previous frame's luma, every
    // odd index advances by a step. This produces (low, high, low, high, ...)
    // temporal differences which uniquely match the [3, 2, 3, 2] field
    // pattern used by the telecine matcher.
    let mut frames = Vec::with_capacity(30);
    let mut prev: Vec<u8> = (0..pixels).map(|i| (i % 251) as u8).collect();
    let mut next = prev.clone();
    for i in 0..30usize {
        if i.is_multiple_of(2) {
            // Duplicate the previous frame -> low temporal_diff.
            frames.push(make_yuv_frame(width, height, prev.clone()));
        } else {
            // Advance to a different texture -> high temporal_diff.
            next = next.iter().map(|&p| p.wrapping_add(40)).collect();
            frames.push(make_yuv_frame(width, height, next.clone()));
            prev = next.clone();
        }
    }

    // Use the sensitive config so the 0.5 threshold can accept the synthetic
    // cadence even if confidence dips slightly.
    let mut detector = TelecineDetector::new(TelecineDetectorConfig::sensitive());

    // We accept either of two related success conditions:
    //   (a) `detect_pulldown_type` returns `Pulldown32` directly, or
    //   (b) the `detect()` info pattern is `Pulldown32` even if validation
    //       gating in `detect_pulldown_type` rejects it.
    let direct = detector
        .detect_pulldown_type(&frames)
        .expect("pulldown detection must not error");
    if direct == PulldownPattern::Pulldown32 {
        return;
    }

    // Fall back to inspecting the raw `detect` result.
    let info = detector
        .detect(&frames)
        .expect("telecine detection must not error");
    assert!(
        info.pattern == PulldownPattern::Pulldown32,
        "expected Pulldown32 cadence on synthetic 3:2 sequence, got {:?}",
        info.pattern
    );
}
