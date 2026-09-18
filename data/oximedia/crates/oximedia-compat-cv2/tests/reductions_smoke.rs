//! Smoke tests for Mat reductions, channel operations, and Mat methods.

use oximedia_compat_cv2::{
    arithmetic::{
        count_non_zero, mean_std_dev, mean_val, merge, min_max_loc, norm, norm_diff, split,
        sum_elems,
    },
    mat::{Mat, MatType, Point},
    NORM_INF, NORM_L1, NORM_L2,
};

fn make_u8_mat(w: usize, h: usize, vals: &[u8]) -> Mat {
    assert_eq!(vals.len(), w * h, "make_u8_mat: data length mismatch");
    let mut m = Mat::new(h, w, MatType::CV_8UC1);
    m.data.copy_from_slice(vals);
    m
}

fn make_f32_mat(w: usize, h: usize, vals: &[f32]) -> Mat {
    assert_eq!(vals.len(), w * h, "make_f32_mat: data length mismatch");
    let mut data = vec![0u8; w * h * 4];
    for (i, &v) in vals.iter().enumerate() {
        data[i * 4..(i + 1) * 4].copy_from_slice(&v.to_ne_bytes());
    }
    Mat {
        data,
        rows: h,
        cols: w,
        step: w * 4,
        mat_type: MatType::CV_32FC1,
    }
}

fn make_u8c3_mat(w: usize, h: usize, vals: &[u8]) -> Mat {
    assert_eq!(vals.len(), w * h * 3, "make_u8c3_mat: data length mismatch");
    let mut m = Mat::new(h, w, MatType::CV_8UC3);
    m.data.copy_from_slice(vals);
    m
}

// ── count_non_zero ────────────────────────────────────────────────────────────

#[test]
fn test_count_non_zero_u8() {
    let data = vec![0u8, 1, 0, 255, 0, 42, 0, 0, 0, 3];
    let m = make_u8_mat(10, 1, &data);
    assert_eq!(count_non_zero(&m).unwrap(), 4);
}

#[test]
fn test_count_non_zero_all_zero() {
    let m = make_u8_mat(4, 4, &[0u8; 16]);
    assert_eq!(count_non_zero(&m).unwrap(), 0);
}

#[test]
fn test_count_non_zero_f32() {
    let vals = vec![0.0f32, 1.0, 0.0, -1.0];
    let m = make_f32_mat(4, 1, &vals);
    assert_eq!(count_non_zero(&m).unwrap(), 2);
}

// ── sum_elems ─────────────────────────────────────────────────────────────────

#[test]
fn test_sum_elems_uniform() {
    let data = vec![10u8; 9]; // 3x3 mat filled with 10
    let m = make_u8_mat(3, 3, &data);
    let s = sum_elems(&m).unwrap();
    assert!((s[0] - 90.0).abs() < 0.01, "sum = {}", s[0]);
    assert_eq!(s[1], 0.0);
}

#[test]
fn test_sum_elems_bgr() {
    // 2 pixels: [1,2,3] and [4,5,6]
    let data = vec![1u8, 2, 3, 4, 5, 6];
    let m = make_u8c3_mat(2, 1, &data);
    let s = sum_elems(&m).unwrap();
    assert!((s[0] - 5.0).abs() < 0.01, "blue sum = {}", s[0]);
    assert!((s[1] - 7.0).abs() < 0.01, "green sum = {}", s[1]);
    assert!((s[2] - 9.0).abs() < 0.01, "red sum = {}", s[2]);
}

// ── mean_val ──────────────────────────────────────────────────────────────────

#[test]
fn test_mean_with_all_zero_mask() {
    let data = vec![100u8; 16];
    let m = make_u8_mat(4, 4, &data);
    let mask_data = vec![0u8; 16]; // all masked out
    let mask = make_u8_mat(4, 4, &mask_data);
    let result = mean_val(&m, Some(&mask)).unwrap();
    assert_eq!(result[0], 0.0, "masked mean should be 0.0");
}

#[test]
fn test_mean_no_mask() {
    // values [0, 100, 200] → mean ≈ 100.0
    let m = make_u8_mat(3, 1, &[0u8, 100, 200]);
    let result = mean_val(&m, None).unwrap();
    assert!((result[0] - 100.0).abs() < 0.01, "mean = {}", result[0]);
}

#[test]
fn test_mean_partial_mask() {
    // All 100s, but only first 2 out of 4 unmasked → mean = 100
    let m = make_u8_mat(4, 1, &[100u8; 4]);
    let mask = make_u8_mat(4, 1, &[255u8, 255, 0, 0]);
    let result = mean_val(&m, Some(&mask)).unwrap();
    assert!(
        (result[0] - 100.0).abs() < 0.01,
        "partial mean = {}",
        result[0]
    );
}

// ── mean_std_dev ──────────────────────────────────────────────────────────────

#[test]
fn test_mean_std_dev_known() {
    // values [1, 2, 3, 4] → mean=2.5, pop_stddev=sqrt(1.25)
    let data = vec![1u8, 2, 3, 4];
    let m = make_u8_mat(4, 1, &data);
    let (means, stddevs) = mean_std_dev(&m, None).unwrap();
    assert!((means[0] - 2.5).abs() < 0.01, "mean = {}", means[0]);
    let expected_std = 1.25f64.sqrt();
    assert!(
        (stddevs[0] - expected_std).abs() < 0.01,
        "stddev = {}, expected {}",
        stddevs[0],
        expected_std
    );
}

#[test]
fn test_mean_std_dev_constant() {
    // All same value → stddev = 0
    let m = make_u8_mat(4, 1, &[50u8; 4]);
    let (means, stddevs) = mean_std_dev(&m, None).unwrap();
    assert!((means[0] - 50.0).abs() < 0.01, "mean = {}", means[0]);
    assert!(
        stddevs[0].abs() < 1e-10,
        "stddev should be 0, got {}",
        stddevs[0]
    );
}

// ── norm ──────────────────────────────────────────────────────────────────────

#[test]
fn test_norm_l2_f32() {
    // [3.0, 4.0] → L2 norm = 5.0
    let vals = vec![3.0f32, 4.0f32];
    let m = make_f32_mat(2, 1, &vals);
    let n = norm(&m, NORM_L2).unwrap();
    assert!((n - 5.0).abs() < 0.001, "norm = {}", n);
}

#[test]
fn test_norm_l1_u8() {
    let m = make_u8_mat(3, 1, &[1u8, 2, 3]);
    let n = norm(&m, NORM_L1).unwrap();
    assert!((n - 6.0).abs() < 0.001, "norm L1 = {}", n);
}

#[test]
fn test_norm_inf_u8() {
    let m = make_u8_mat(4, 1, &[10u8, 200, 50, 100]);
    let n = norm(&m, NORM_INF).unwrap();
    assert!((n - 200.0).abs() < 0.001, "norm INF = {}", n);
}

#[test]
fn test_norm_unsupported_returns_error() {
    let m = make_u8_mat(2, 1, &[1u8, 2]);
    assert!(norm(&m, 99).is_err(), "norm with flag 99 should error");
}

// ── norm_diff ─────────────────────────────────────────────────────────────────

#[test]
fn test_norm_diff_inf() {
    let a = make_u8_mat(3, 1, &[10u8, 20, 30]);
    let b = make_u8_mat(3, 1, &[12u8, 15, 30]);
    let d = norm_diff(&a, &b, NORM_INF).unwrap();
    // max |diff| = max(2, 5, 0) = 5
    assert!((d - 5.0).abs() < 0.001, "norm_diff INF = {}", d);
}

#[test]
fn test_norm_diff_l1() {
    let a = make_u8_mat(3, 1, &[10u8, 20, 30]);
    let b = make_u8_mat(3, 1, &[12u8, 15, 30]);
    let d = norm_diff(&a, &b, NORM_L1).unwrap();
    // |2| + |5| + |0| = 7
    assert!((d - 7.0).abs() < 0.001, "norm_diff L1 = {}", d);
}

// ── min_max_loc ───────────────────────────────────────────────────────────────

#[test]
fn test_min_max_loc_basic() {
    // [10, 200, 50] → min=10 at x=0, max=200 at x=1
    let m = make_u8_mat(3, 1, &[10u8, 200, 50]);
    let (mn, mx, mn_loc, mx_loc) = min_max_loc(&m, None).unwrap();
    assert!((mn - 10.0).abs() < 0.001, "min = {}", mn);
    assert!((mx - 200.0).abs() < 0.001, "max = {}", mx);
    assert_eq!(mn_loc, Point { x: 0, y: 0 });
    assert_eq!(mx_loc, Point { x: 1, y: 0 });
}

#[test]
fn test_min_max_loc_with_mask() {
    // [10, 200, 50] with mask [0, 255, 0] → only pixel x=1 active
    let m = make_u8_mat(3, 1, &[10u8, 200, 50]);
    let mask = make_u8_mat(3, 1, &[0u8, 255, 0]);
    let (mn, mx, _mn_loc, mx_loc) = min_max_loc(&m, Some(&mask)).unwrap();
    assert!((mn - 200.0).abs() < 0.001, "masked min = {}", mn);
    assert!((mx - 200.0).abs() < 0.001, "masked max = {}", mx);
    assert_eq!(mx_loc, Point { x: 1, y: 0 });
}

#[test]
fn test_min_max_loc_multichannel_returns_error() {
    let data = vec![1u8, 2, 3, 4, 5, 6];
    let m = make_u8c3_mat(2, 1, &data);
    assert!(min_max_loc(&m, None).is_err());
}

#[test]
fn test_min_max_loc_all_masked_returns_zero() {
    let m = make_u8_mat(3, 1, &[10u8, 20, 30]);
    let mask = make_u8_mat(3, 1, &[0u8; 3]);
    let (mn, mx, mn_loc, mx_loc) = min_max_loc(&m, Some(&mask)).unwrap();
    assert_eq!((mn, mx), (0.0, 0.0));
    assert_eq!(mn_loc, Point { x: 0, y: 0 });
    assert_eq!(mx_loc, Point { x: 0, y: 0 });
}

// ── split / merge ─────────────────────────────────────────────────────────────

#[test]
fn test_split_merge_roundtrip() {
    // Create a 4x4 3-channel mat
    let mut data = vec![0u8; 4 * 4 * 3];
    for i in 0..16 {
        data[i * 3] = (i % 256) as u8; // B
        data[i * 3 + 1] = (i * 2 % 256) as u8; // G
        data[i * 3 + 2] = (i * 3 % 256) as u8; // R
    }
    let bgr = make_u8c3_mat(4, 4, &data);
    let channels = split(&bgr).unwrap();
    assert_eq!(channels.len(), 3);
    let refs: Vec<&Mat> = channels.iter().collect();
    let merged = merge(&refs).unwrap();
    assert_eq!(merged.data, bgr.data);
    assert_eq!(merged.mat_type, MatType::CV_8UC3);
}

#[test]
fn test_split_single_channel() {
    let m = make_u8_mat(3, 3, &[42u8; 9]);
    let planes = split(&m).unwrap();
    assert_eq!(planes.len(), 1);
    assert_eq!(planes[0].data, m.data);
}

#[test]
fn test_merge_mismatch_returns_error() {
    let a = make_u8_mat(2, 2, &[0u8; 4]);
    let b = make_u8_mat(3, 3, &[0u8; 9]);
    assert!(merge(&[&a, &b]).is_err());
}

// ── Mat methods ───────────────────────────────────────────────────────────────

#[test]
fn test_convert_to_u8_to_f32() {
    let data = vec![0u8, 128, 255];
    let m = make_u8_mat(3, 1, &data);
    let f = m.convert_to(MatType::CV_32FC1, 1.0 / 255.0, 0.0).unwrap();
    assert_eq!(f.mat_type, MatType::CV_32FC1);
    assert_eq!(f.data.len(), 12); // 3 * 4 bytes
    let v0 = f32::from_ne_bytes(f.data[0..4].try_into().unwrap());
    let v2 = f32::from_ne_bytes(f.data[8..12].try_into().unwrap());
    assert!(v0.abs() < 0.01, "v0 = {}", v0);
    assert!((v2 - 1.0).abs() < 0.01, "v2 = {}", v2);
}

#[test]
fn test_convert_to_f32_to_u8() {
    let vals = vec![0.0f32, 0.5, 1.0];
    let m = make_f32_mat(3, 1, &vals);
    let out = m.convert_to(MatType::CV_8UC1, 255.0, 0.0).unwrap();
    assert_eq!(out.mat_type, MatType::CV_8UC1);
    assert_eq!(out.data[0], 0);
    assert_eq!(out.data[2], 255);
}

#[test]
fn test_clone_mat() {
    let m = make_u8_mat(2, 2, &[1u8, 2, 3, 4]);
    let c = m.clone_mat();
    assert_eq!(m.data, c.data);
    assert_eq!(m.rows, c.rows);
    assert_eq!(m.cols, c.cols);
    assert_eq!(m.mat_type, c.mat_type);
}

#[test]
fn test_submat_basic() {
    // 4x4 identity-ish: col index as value
    let data: Vec<u8> = (0..16).map(|i| (i % 4) as u8).collect();
    let m = make_u8_mat(4, 4, &data);
    // Extract 2x2 from top-left
    let sub = m.submat(0, 0, 2, 2).unwrap();
    assert_eq!(sub.rows, 2);
    assert_eq!(sub.cols, 2);
    // Pixel (0,0) → 0, (0,1) → 1
    assert_eq!(sub.at_8u1(0, 0), 0);
    assert_eq!(sub.at_8u1(0, 1), 1);
}

#[test]
fn test_submat_out_of_bounds_clamped() {
    let m = make_u8_mat(4, 4, &[0u8; 16]);
    // Ask for x=2, y=2, w=10, h=10 → clamps to 2x2
    let sub = m.submat(2, 2, 10, 10).unwrap();
    assert_eq!(sub.rows, 2);
    assert_eq!(sub.cols, 2);
}

#[test]
fn test_submat_empty_returns_error() {
    let m = make_u8_mat(4, 4, &[0u8; 16]);
    // Negative width → zero width after clamping
    assert!(m.submat(0, 0, 0, 0).is_err());
}

#[test]
fn test_reshape_valid() {
    // 2x4 → 4x2 (same total 8 elements)
    let data: Vec<u8> = (0..8).collect();
    let m = make_u8_mat(4, 2, &data);
    let r = m.reshape(2, 4).unwrap();
    assert_eq!(r.rows, 2);
    assert_eq!(r.cols, 4);
    assert_eq!(r.data, m.data);
}

#[test]
fn test_reshape_mismatch_returns_error() {
    let m = make_u8_mat(3, 3, &[0u8; 9]);
    assert!(m.reshape(2, 4).is_err());
}
