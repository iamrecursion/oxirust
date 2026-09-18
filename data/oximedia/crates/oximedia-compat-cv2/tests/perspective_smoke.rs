//! Smoke tests for warpPerspective, getPerspectiveTransform, getAffineTransform, and remap.

use oximedia_compat_cv2::{
    constants::{interpolation::INTER_LINEAR, warp_flags::WARP_INVERSE_MAP},
    error::Cv2Result,
    geometry::{get_affine_transform, get_perspective_transform, remap, warp_perspective},
    mat::{Mat, MatType, Point2f},
    BORDER_CONSTANT,
};

fn p(x: f32, y: f32) -> Point2f {
    Point2f { x, y }
}

/// Read a f64 value from a CV_64FC1 Mat at (row, col).
fn read_h(m: &Mat, row: usize, col: usize) -> f64 {
    let idx = (row * m.cols + col) * 8;
    let bytes: [u8; 8] = m.data[idx..idx + 8].try_into().expect("slice length 8");
    f64::from_ne_bytes(bytes)
}

/// Build a CV_8UC1 Mat from given dimensions and fill value.
fn make_gray_mat(w: usize, h: usize, fill: u8) -> Mat {
    Mat::new(h, w, MatType::CV_8UC1).apply_fill(fill)
}

/// Helper trait to fill a Mat with a constant byte value.
trait MatFill {
    fn apply_fill(self, v: u8) -> Self;
}

impl MatFill for Mat {
    fn apply_fill(mut self, v: u8) -> Self {
        self.data.fill(v);
        self
    }
}

/// Build an identity 3×3 CV_64FC1 homography Mat.
fn identity_homography() -> Mat {
    let identity: [f64; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let mut data = vec![0u8; 72];
    for (i, &v) in identity.iter().enumerate() {
        data[i * 8..(i + 1) * 8].copy_from_slice(&v.to_ne_bytes());
    }
    Mat {
        data,
        rows: 3,
        cols: 3,
        step: 24,
        mat_type: MatType::CV_64FC1,
    }
}

// ── getPerspectiveTransform tests ─────────────────────────────────────────────

#[test]
fn test_get_perspective_transform_identity() {
    // Unit square to unit square → H should be the identity-ish (diagonal dominated).
    let src = [p(0.0, 0.0), p(1.0, 0.0), p(1.0, 1.0), p(0.0, 1.0)];
    let dst = src;
    let h: Mat = get_perspective_transform(&src, &dst).expect("should succeed");
    assert_eq!(h.rows, 3);
    assert_eq!(h.cols, 3);
    assert_eq!(h.mat_type, MatType::CV_64FC1);
    // h[2,2] is always 1 by construction (fixed pivot).
    assert!(
        (read_h(&h, 2, 2) - 1.0).abs() < 1e-6,
        "h22={}",
        read_h(&h, 2, 2)
    );
    // Diagonal dominant.
    assert!(
        (read_h(&h, 0, 0) - 1.0).abs() < 1e-4,
        "h00={}",
        read_h(&h, 0, 0)
    );
    assert!(
        (read_h(&h, 1, 1) - 1.0).abs() < 1e-4,
        "h11={}",
        read_h(&h, 1, 1)
    );
}

#[test]
fn test_get_perspective_transform_translation() {
    // Shift all points by (5, 3).
    let src = [p(0.0, 0.0), p(10.0, 0.0), p(10.0, 10.0), p(0.0, 10.0)];
    let dst = [p(5.0, 3.0), p(15.0, 3.0), p(15.0, 13.0), p(5.0, 13.0)];
    let h: Mat = get_perspective_transform(&src, &dst).expect("should succeed");
    // H · [0,0,1]^T should give (5, 3, 1): check translation column.
    let h02 = read_h(&h, 0, 2); // x-translation (after normalising by h22=1).
    let h12 = read_h(&h, 1, 2); // y-translation.
    assert!((h02 - 5.0).abs() < 1e-4, "h02={}", h02);
    assert!((h12 - 3.0).abs() < 1e-4, "h12={}", h12);
}

#[test]
fn test_get_perspective_transform_collinear_error() {
    // Collinear points should produce a degenerate system.
    let pts = [p(0.0, 0.0), p(1.0, 0.0), p(2.0, 0.0), p(3.0, 0.0)];
    let result: Cv2Result<Mat> = get_perspective_transform(&pts, &pts);
    assert!(result.is_err(), "expected error for collinear points");
}

// ── getAffineTransform tests ──────────────────────────────────────────────────

#[test]
fn test_get_affine_transform_identity() {
    let src = [p(0.0, 0.0), p(1.0, 0.0), p(0.0, 1.0)];
    let dst = src;
    let m: Mat = get_affine_transform(&src, &dst).expect("should succeed");
    assert_eq!(m.rows, 2);
    assert_eq!(m.cols, 3);
    assert_eq!(m.mat_type, MatType::CV_64FC1);
    // Expected [[1, 0, 0], [0, 1, 0]].
    assert!(
        (read_h(&m, 0, 0) - 1.0).abs() < 1e-6,
        "m00={}",
        read_h(&m, 0, 0)
    );
    assert!((read_h(&m, 0, 1)).abs() < 1e-6, "m01={}", read_h(&m, 0, 1));
    assert!((read_h(&m, 0, 2)).abs() < 1e-6, "m02={}", read_h(&m, 0, 2));
    assert!((read_h(&m, 1, 0)).abs() < 1e-6, "m10={}", read_h(&m, 1, 0));
    assert!(
        (read_h(&m, 1, 1) - 1.0).abs() < 1e-6,
        "m11={}",
        read_h(&m, 1, 1)
    );
    assert!((read_h(&m, 1, 2)).abs() < 1e-6, "m12={}", read_h(&m, 1, 2));
}

#[test]
fn test_get_affine_transform_translation() {
    // Pure translation by (7, 4).
    let src = [p(0.0, 0.0), p(10.0, 0.0), p(0.0, 10.0)];
    let dst = [p(7.0, 4.0), p(17.0, 4.0), p(7.0, 14.0)];
    let m: Mat = get_affine_transform(&src, &dst).expect("should succeed");
    // Translation column: m[0,2]=7, m[1,2]=4.
    assert!(
        (read_h(&m, 0, 2) - 7.0).abs() < 1e-4,
        "m02={}",
        read_h(&m, 0, 2)
    );
    assert!(
        (read_h(&m, 1, 2) - 4.0).abs() < 1e-4,
        "m12={}",
        read_h(&m, 1, 2)
    );
    // Scale unchanged.
    assert!(
        (read_h(&m, 0, 0) - 1.0).abs() < 1e-4,
        "m00={}",
        read_h(&m, 0, 0)
    );
    assert!(
        (read_h(&m, 1, 1) - 1.0).abs() < 1e-4,
        "m11={}",
        read_h(&m, 1, 1)
    );
}

#[test]
fn test_get_affine_transform_collinear_error() {
    let pts = [p(0.0, 0.0), p(1.0, 0.0), p(2.0, 0.0)];
    let result: Cv2Result<Mat> = get_affine_transform(&pts, &pts);
    assert!(result.is_err(), "expected error for collinear points");
}

// ── warpPerspective tests ─────────────────────────────────────────────────────

#[test]
fn test_warp_perspective_identity() {
    let w = 20usize;
    let h = 20usize;
    // Checkerboard pattern.
    let mut data = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            if (x + y) % 2 == 0 {
                data[y * w + x] = 200u8;
            }
        }
    }
    let src = Mat {
        data: data.clone(),
        rows: h,
        cols: w,
        step: w,
        mat_type: MatType::CV_8UC1,
    };
    let m = identity_homography();
    let out = warp_perspective(
        &src,
        &m,
        (w, h),
        INTER_LINEAR,
        BORDER_CONSTANT,
        [0, 0, 0, 0],
    )
    .expect("should succeed");
    assert_eq!(out.rows, h);
    assert_eq!(out.cols, w);
    // Central pixel should approximately match.
    let center_orig = data[10 * w + 10];
    let center_warped = out.data[10 * w + 10];
    assert!(
        (center_orig as i32 - center_warped as i32).abs() <= 10,
        "center mismatch: orig={} warped={}",
        center_orig,
        center_warped
    );
}

#[test]
fn test_warp_perspective_constant_fill() {
    let w = 10usize;
    let h = 10usize;
    let src = make_gray_mat(w, h, 128u8);
    let m = identity_homography();
    // Larger output — border pixels should be filled with 42.
    let out = warp_perspective(
        &src,
        &m,
        (20, 20),
        INTER_LINEAR,
        BORDER_CONSTANT,
        [42, 0, 0, 0],
    )
    .expect("should succeed");
    assert_eq!(out.rows, 20);
    assert_eq!(out.cols, 20);
    // Bottom-right corner is outside source, should be 42.
    let corner = out.data[19 * 20 + 19];
    assert_eq!(corner, 42, "expected border fill 42, got {}", corner);
}

#[test]
fn test_warp_perspective_inverse_flag() {
    // WARP_INVERSE_MAP: H is used directly as the inverse (no inversion step).
    let w = 20usize;
    let h = 20usize;
    let src = make_gray_mat(w, h, 100u8);
    // Identity is its own inverse, so result should be the same either way.
    let m = identity_homography();
    let out = warp_perspective(
        &src,
        &m,
        (w, h),
        WARP_INVERSE_MAP | INTER_LINEAR,
        BORDER_CONSTANT,
        [0, 0, 0, 0],
    )
    .expect("should succeed with WARP_INVERSE_MAP");
    assert_eq!(out.rows, h);
    assert_eq!(out.cols, w);
    // All pixels should be ~100.
    for (i, &px) in out.data.iter().enumerate() {
        assert!(
            (px as i32 - 100).abs() <= 5,
            "pixel {} expected ~100, got {}",
            i,
            px
        );
    }
}

#[test]
fn test_warp_perspective_nearest() {
    let w = 10usize;
    let h = 10usize;
    let src = make_gray_mat(w, h, 77u8);
    let m = identity_homography();
    // INTER_NEAREST = 0.
    let out = warp_perspective(&src, &m, (w, h), 0, BORDER_CONSTANT, [0, 0, 0, 0])
        .expect("nearest-neighbour should succeed");
    assert_eq!(out.rows, h);
    assert_eq!(out.cols, w);
    assert!(out.data.iter().all(|&v| v == 77), "all pixels should be 77");
}

// ── remap tests ───────────────────────────────────────────────────────────────

fn make_identity_maps(w: usize, h: usize) -> (Mat, Mat) {
    let mut mx_data = vec![0u8; w * h * 4];
    let mut my_data = vec![0u8; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            mx_data[i * 4..(i + 1) * 4].copy_from_slice(&(x as f32).to_ne_bytes());
            my_data[i * 4..(i + 1) * 4].copy_from_slice(&(y as f32).to_ne_bytes());
        }
    }
    let map_x = Mat {
        data: mx_data,
        rows: h,
        cols: w,
        step: w * 4,
        mat_type: MatType::CV_32FC1,
    };
    let map_y = Mat {
        data: my_data,
        rows: h,
        cols: w,
        step: w * 4,
        mat_type: MatType::CV_32FC1,
    };
    (map_x, map_y)
}

#[test]
fn test_remap_identity() {
    let w = 10usize;
    let h = 10usize;
    let data: Vec<u8> = (0..w * h).map(|i| (i % 200) as u8).collect();
    let src = Mat {
        data: data.clone(),
        rows: h,
        cols: w,
        step: w,
        mat_type: MatType::CV_8UC1,
    };
    let (map_x, map_y) = make_identity_maps(w, h);
    let out = remap(&src, &map_x, &map_y).expect("identity remap should succeed");
    assert_eq!(out.rows, h);
    assert_eq!(out.cols, w);
    // With bilinear on integer coords, output should match input exactly.
    for i in 0..w * h {
        assert!(
            (out.data[i] as i32 - data[i] as i32).abs() <= 2,
            "pixel {} mismatch: got {}, expected {}",
            i,
            out.data[i],
            data[i]
        );
    }
}

#[test]
fn test_remap_constant_source() {
    let w = 10usize;
    let h = 10usize;
    let src = make_gray_mat(w, h, 123u8);
    let (map_x, map_y) = make_identity_maps(w, h);
    let out = remap(&src, &map_x, &map_y).expect("remap should succeed");
    for (i, &v) in out.data.iter().enumerate() {
        assert_eq!(v, 123, "pixel {} should be 123, got {}", i, v);
    }
}

#[test]
fn test_remap_all_to_single_pixel() {
    let w = 10usize;
    let h = 10usize;
    let mut data = vec![0u8; w * h];
    data[5 * w + 5] = 123u8; // Only one pixel set.
    let src = Mat {
        data,
        rows: h,
        cols: w,
        step: w,
        mat_type: MatType::CV_8UC1,
    };
    // Map everything to (5, 5).
    let n = w * h;
    let mut mx_data = vec![0u8; n * 4];
    let mut my_data = vec![0u8; n * 4];
    for i in 0..n {
        mx_data[i * 4..(i + 1) * 4].copy_from_slice(&5.0f32.to_ne_bytes());
        my_data[i * 4..(i + 1) * 4].copy_from_slice(&5.0f32.to_ne_bytes());
    }
    let map_x = Mat {
        data: mx_data,
        rows: h,
        cols: w,
        step: w * 4,
        mat_type: MatType::CV_32FC1,
    };
    let map_y = Mat {
        data: my_data,
        rows: h,
        cols: w,
        step: w * 4,
        mat_type: MatType::CV_32FC1,
    };
    let out = remap(&src, &map_x, &map_y).expect("remap should succeed");
    for (i, &pixel) in out.data.iter().enumerate() {
        assert!(pixel > 100, "pixel {} expected ~123, got {}", i, pixel);
    }
}

#[test]
fn test_remap_size_mismatch_error() {
    let w = 5usize;
    let h = 5usize;
    let src = make_gray_mat(w, h, 0u8);
    let (map_x, _) = make_identity_maps(w, h);
    // map_y has different dimensions.
    let (map_y_wrong, _) = make_identity_maps(w + 1, h + 1);
    let result = remap(&src, &map_x, &map_y_wrong);
    assert!(result.is_err(), "should fail on size mismatch");
}
