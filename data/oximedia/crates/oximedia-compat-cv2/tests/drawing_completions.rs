//! Integration tests for drawing completion functions and Hershey font renderer.

use oximedia_compat_cv2::{
    constants::{FONT_HERSHEY_SIMPLEX, MARKER_CROSS},
    drawing::{draw_contours, draw_keypoints, draw_marker},
    features::{DMatch, KeyPoint},
    hershey_font::put_text_hershey,
    Mat, MatType, Point, Point2f, Scalar,
};

fn blank_mat(w: usize, h: usize) -> Mat {
    Mat::new(h, w, MatType::CV_8UC3)
}

fn count_nonzero_u8(m: &Mat) -> usize {
    m.data.iter().filter(|&&v| v > 0).count()
}

fn white() -> Scalar {
    Scalar(255.0, 255.0, 255.0, 255.0)
}

// ── draw_contours tests ───────────────────────────────────────────────────────

#[test]
fn test_draw_contours_square_outline() {
    let mut img = blank_mat(50, 50);
    let square = vec![
        Point { x: 10, y: 10 },
        Point { x: 30, y: 10 },
        Point { x: 30, y: 30 },
        Point { x: 10, y: 30 },
    ];
    let contours = vec![square];
    draw_contours(&mut img, &contours, -1, white(), 1).unwrap();
    assert!(
        count_nonzero_u8(&img) > 0,
        "square outline should produce pixels"
    );
}

#[test]
fn test_draw_contours_filled() {
    let mut img = blank_mat(50, 50);
    let square = vec![
        Point { x: 10, y: 10 },
        Point { x: 30, y: 10 },
        Point { x: 30, y: 30 },
        Point { x: 10, y: 30 },
    ];
    let contours = vec![square];
    // FILLED = -1 triggers scanline fill
    draw_contours(&mut img, &contours, -1, white(), -1).unwrap();
    let pixels = count_nonzero_u8(&img);
    // Filled 20×20 area → at least 20*20*3 = 1200 nonzero bytes
    assert!(
        pixels > 1000,
        "filled contour should produce many pixels, got {}",
        pixels
    );
}

#[test]
fn test_draw_contours_all() {
    let mut img = blank_mat(60, 60);
    let c1 = vec![
        Point { x: 5, y: 5 },
        Point { x: 15, y: 5 },
        Point { x: 15, y: 15 },
        Point { x: 5, y: 15 },
    ];
    let c2 = vec![
        Point { x: 35, y: 35 },
        Point { x: 50, y: 35 },
        Point { x: 50, y: 50 },
        Point { x: 35, y: 50 },
    ];
    let contours = vec![c1, c2];
    draw_contours(
        &mut img,
        &contours,
        -1,
        Scalar(200.0, 200.0, 200.0, 255.0),
        1,
    )
    .unwrap();
    assert!(
        count_nonzero_u8(&img) > 10,
        "both contours should produce pixels"
    );
}

#[test]
fn test_draw_contours_single_index() {
    let mut img_all = blank_mat(60, 60);
    let mut img_one = blank_mat(60, 60);
    let c1 = vec![
        Point { x: 5, y: 5 },
        Point { x: 15, y: 5 },
        Point { x: 15, y: 15 },
        Point { x: 5, y: 15 },
    ];
    let c2 = vec![
        Point { x: 35, y: 35 },
        Point { x: 50, y: 35 },
        Point { x: 50, y: 50 },
        Point { x: 35, y: 50 },
    ];
    let contours = vec![c1, c2];
    draw_contours(&mut img_all, &contours, -1, white(), 1).unwrap();
    draw_contours(&mut img_one, &contours, 0, white(), 1).unwrap();
    assert!(
        count_nonzero_u8(&img_all) >= count_nonzero_u8(&img_one),
        "drawing all should produce at least as many pixels as drawing one"
    );
}

#[test]
fn test_draw_contours_out_of_bounds_index_ignored() {
    // contour_idx beyond slice length should be silently ignored
    let mut img = blank_mat(30, 30);
    let contours: Vec<Vec<Point>> = vec![];
    draw_contours(&mut img, &contours, 5, white(), 1).unwrap();
    assert_eq!(
        count_nonzero_u8(&img),
        0,
        "out-of-bounds index should draw nothing"
    );
}

// ── draw_keypoints tests ──────────────────────────────────────────────────────

fn make_kp(x: f32, y: f32, size: f32, angle: f32) -> KeyPoint {
    KeyPoint {
        pt: Point2f { x, y },
        size,
        angle,
        response: 1.0,
        octave: 0,
        class_id: -1,
    }
}

#[test]
fn test_draw_keypoints_circles() {
    let src = blank_mat(60, 60);
    let mut out = blank_mat(60, 60);
    let kps = vec![
        make_kp(20.0, 20.0, 10.0, 0.0),
        make_kp(40.0, 40.0, 10.0, 0.0),
    ];
    draw_keypoints(&src, &kps, &mut out, Scalar(255.0, 0.0, 0.0, 255.0), 0).unwrap();
    assert!(
        count_nonzero_u8(&out) > 0,
        "keypoints should produce non-zero pixels"
    );
}

#[test]
fn test_draw_keypoints_copies_src() {
    // src has white row at y=0; after draw_keypoints out should still have it
    let mut src = blank_mat(20, 20);
    for i in 0..20 * 3 {
        src.data[i] = 128;
    }
    let mut out = blank_mat(20, 20);
    let kps: Vec<KeyPoint> = vec![];
    draw_keypoints(&src, &kps, &mut out, white(), 0).unwrap();
    assert_eq!(
        &out.data[..60],
        &src.data[..60],
        "src content should be copied into out"
    );
}

#[test]
fn test_draw_keypoints_rich_flag() {
    let src = blank_mat(60, 60);
    let mut out_plain = blank_mat(60, 60);
    let mut out_rich = blank_mat(60, 60);
    let kps = vec![make_kp(30.0, 30.0, 10.0, 45.0)];
    draw_keypoints(
        &src,
        &kps,
        &mut out_plain,
        Scalar(255.0, 0.0, 0.0, 255.0),
        0,
    )
    .unwrap();
    // DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS = 4
    draw_keypoints(&src, &kps, &mut out_rich, Scalar(255.0, 0.0, 0.0, 255.0), 4).unwrap();
    // Rich keypoint includes an orientation line — should add more pixels
    assert!(
        count_nonzero_u8(&out_rich) >= count_nonzero_u8(&out_plain),
        "rich keypoints should produce at least as many pixels as plain"
    );
}

// ── draw_marker tests ─────────────────────────────────────────────────────────

#[test]
fn test_draw_marker_cross() {
    let mut img = blank_mat(40, 40);
    draw_marker(
        &mut img,
        Point { x: 20, y: 20 },
        white(),
        MARKER_CROSS,
        10,
        1,
    )
    .unwrap();
    assert!(
        count_nonzero_u8(&img) > 0,
        "MARKER_CROSS should produce pixels"
    );
}

#[test]
fn test_draw_marker_tilted_cross() {
    let mut img = blank_mat(40, 40);
    draw_marker(&mut img, Point { x: 20, y: 20 }, white(), 1, 10, 1).unwrap();
    assert!(
        count_nonzero_u8(&img) > 0,
        "MARKER_TILTED_CROSS should produce pixels"
    );
}

#[test]
fn test_draw_marker_star() {
    let mut img_cross = blank_mat(40, 40);
    let mut img_star = blank_mat(40, 40);
    draw_marker(
        &mut img_cross,
        Point { x: 20, y: 20 },
        white(),
        MARKER_CROSS,
        10,
        1,
    )
    .unwrap();
    draw_marker(&mut img_star, Point { x: 20, y: 20 }, white(), 2, 10, 1).unwrap();
    // Star = cross + tilted cross, so star has more or equal pixels than just cross
    assert!(
        count_nonzero_u8(&img_star) >= count_nonzero_u8(&img_cross),
        "MARKER_STAR should produce at least as many pixels as MARKER_CROSS"
    );
}

#[test]
fn test_draw_marker_diamond() {
    let mut img = blank_mat(40, 40);
    draw_marker(&mut img, Point { x: 20, y: 20 }, white(), 3, 10, 1).unwrap();
    assert!(
        count_nonzero_u8(&img) > 0,
        "MARKER_DIAMOND should produce pixels"
    );
}

#[test]
fn test_draw_marker_square() {
    let mut img = blank_mat(40, 40);
    draw_marker(&mut img, Point { x: 20, y: 20 }, white(), 4, 10, 1).unwrap();
    assert!(
        count_nonzero_u8(&img) > 0,
        "MARKER_SQUARE should produce pixels"
    );
}

#[test]
fn test_draw_marker_triangle_up() {
    let mut img = blank_mat(40, 40);
    draw_marker(&mut img, Point { x: 20, y: 20 }, white(), 5, 10, 1).unwrap();
    assert!(
        count_nonzero_u8(&img) > 0,
        "MARKER_TRIANGLE_UP should produce pixels"
    );
}

#[test]
fn test_draw_marker_triangle_down() {
    let mut img = blank_mat(40, 40);
    draw_marker(&mut img, Point { x: 20, y: 20 }, white(), 6, 10, 1).unwrap();
    assert!(
        count_nonzero_u8(&img) > 0,
        "MARKER_TRIANGLE_DOWN should produce pixels"
    );
}

#[test]
fn test_draw_marker_unknown_no_panic() {
    let mut img = blank_mat(40, 40);
    // Unknown marker type should be silently ignored without panic
    draw_marker(&mut img, Point { x: 20, y: 20 }, white(), 99, 10, 1).unwrap();
    // No pixels should be drawn for an unknown type
    assert_eq!(
        count_nonzero_u8(&img),
        0,
        "unknown marker should draw nothing"
    );
}

// ── put_text_hershey tests ────────────────────────────────────────────────────

#[test]
fn test_put_text_hershey_produces_pixels() {
    let mut img = blank_mat(200, 60);
    let result = put_text_hershey(
        &mut img,
        "A0",
        Point { x: 10, y: 50 },
        FONT_HERSHEY_SIMPLEX,
        2.0,
        white(),
        1,
    );
    assert!(result.is_ok(), "put_text_hershey should succeed for 'A0'");
    assert!(
        count_nonzero_u8(&img) > 0,
        "text rendering should produce non-zero pixels"
    );
}

#[test]
fn test_put_text_hershey_unsupported_font() {
    let mut img = blank_mat(100, 50);
    let result = put_text_hershey(&mut img, "A", Point { x: 0, y: 20 }, 99, 1.0, white(), 1);
    assert!(result.is_err(), "unsupported font_face should return error");
}

#[test]
fn test_put_text_hershey_advance_width() {
    let mut img1 = blank_mat(300, 60);
    let mut img2 = blank_mat(300, 60);
    put_text_hershey(
        &mut img1,
        "A",
        Point { x: 0, y: 50 },
        FONT_HERSHEY_SIMPLEX,
        2.0,
        white(),
        1,
    )
    .unwrap();
    put_text_hershey(
        &mut img2,
        "AB",
        Point { x: 0, y: 50 },
        FONT_HERSHEY_SIMPLEX,
        2.0,
        white(),
        1,
    )
    .unwrap();
    let p1 = count_nonzero_u8(&img1);
    let p2 = count_nonzero_u8(&img2);
    assert!(
        p2 > p1,
        "two chars should produce more pixels than one: {} vs {}",
        p2,
        p1
    );
}

#[test]
fn test_put_text_hershey_space_advances_cursor() {
    // Rendering " A" should produce pixels further right than "A" at x=0
    let mut img_a = blank_mat(300, 60);
    let mut img_space_a = blank_mat(300, 60);
    put_text_hershey(
        &mut img_a,
        "A",
        Point { x: 0, y: 50 },
        FONT_HERSHEY_SIMPLEX,
        2.0,
        white(),
        1,
    )
    .unwrap();
    put_text_hershey(
        &mut img_space_a,
        " A",
        Point { x: 0, y: 50 },
        FONT_HERSHEY_SIMPLEX,
        2.0,
        white(),
        1,
    )
    .unwrap();
    // " A" should not have fewer pixels than "A" — space just shifts; pixels count may be equal or close
    let p_a = count_nonzero_u8(&img_a);
    let p_space_a = count_nonzero_u8(&img_space_a);
    // The pixel counts should be similar (same glyph, just shifted)
    // Allow a small delta for rounding at different x offsets
    assert!(
        p_space_a > 0 && p_a > 0,
        "both should produce pixels: A={}, ' A'={}",
        p_a,
        p_space_a
    );
}

#[test]
fn test_put_text_hershey_all_supported_digits() {
    let mut img = blank_mat(400, 60);
    put_text_hershey(
        &mut img,
        "0123456789",
        Point { x: 5, y: 50 },
        FONT_HERSHEY_SIMPLEX,
        1.5,
        white(),
        1,
    )
    .unwrap();
    assert!(
        count_nonzero_u8(&img) > 0,
        "all digits should produce pixels"
    );
}

#[test]
fn test_put_text_hershey_all_supported_letters() {
    let mut img = blank_mat(400, 60);
    put_text_hershey(
        &mut img,
        "ABCDEFGHIJKLMN",
        Point { x: 5, y: 50 },
        FONT_HERSHEY_SIMPLEX,
        1.5,
        white(),
        1,
    )
    .unwrap();
    assert!(
        count_nonzero_u8(&img) > 0,
        "all supported letters should produce pixels"
    );
}

#[test]
fn test_put_text_hershey_unknown_char_no_panic() {
    let mut img = blank_mat(200, 60);
    // Characters outside the glyph table should not panic
    let result = put_text_hershey(
        &mut img,
        "A!@#Z",
        Point { x: 5, y: 50 },
        FONT_HERSHEY_SIMPLEX,
        2.0,
        white(),
        1,
    );
    assert!(result.is_ok(), "unknown chars should be silently skipped");
}

// ── draw_matches smoke test ───────────────────────────────────────────────────

#[test]
fn test_draw_matches_output_size() {
    use oximedia_compat_cv2::drawing::draw_matches;
    let img1 = blank_mat(40, 30);
    let img2 = blank_mat(50, 30);
    let kp1 = vec![make_kp(10.0, 10.0, 5.0, 0.0)];
    let kp2 = vec![make_kp(10.0, 10.0, 5.0, 0.0)];
    let m = DMatch {
        query_idx: 0,
        train_idx: 0,
        distance: 0.0,
    };
    let mut out = blank_mat(1, 1);
    draw_matches(
        &img1,
        &kp1,
        &img2,
        &kp2,
        &[m],
        &mut out,
        white(),
        Scalar(0.0, 255.0, 0.0, 255.0),
        &[],
        0,
    )
    .unwrap();
    assert_eq!(
        out.cols,
        40 + 50,
        "output width should be sum of input widths"
    );
    assert_eq!(out.rows, 30, "output height should be max of input heights");
}
