//! BFMatcher norm generalisation tests (Run 5, Slice F).
//!
//! Covers NORM_L1, NORM_L2, NORM_L2SQR float norms and dtype-mismatch
//! rejection.  All tests use `CV_32FC1` descriptor Mats constructed by
//! encoding `f32` values as little-endian bytes.

use oximedia_compat_cv2::constants::norm_type::{NORM_L1, NORM_L2, NORM_L2SQR};
use oximedia_compat_cv2::features::BFMatcher;
use oximedia_compat_cv2::mat::{Mat, MatType};

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Build a `CV_32FC1` Mat from a flat slice of `f32` values.
///
/// `rows × cols` elements are serialised as little-endian bytes.
fn make_f32_mat(rows: usize, cols: usize, data: &[f32]) -> Mat {
    assert_eq!(
        data.len(),
        rows * cols,
        "data length must equal rows * cols"
    );
    let byte_data: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    Mat {
        data: byte_data,
        rows,
        cols,
        step: cols * 4,
        mat_type: MatType::CV_32FC1,
    }
}

// ── Test 1: NORM_L2 self-match → identity, distance 0 ────────────────────────

#[test]
fn bfmatcher_norm_l2_self_match_identity() {
    // Three 4-element float descriptor rows — arbitrary non-zero values.
    let descriptors_data = [
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.1, 0.2, 0.3, 0.4,
    ];
    let mat = make_f32_mat(3, 4, &descriptors_data);

    let matcher = BFMatcher::new(NORM_L2).expect("NORM_L2 should be accepted");
    let matches = matcher
        .match_descriptors(&mat, &mat)
        .expect("self-match must succeed");

    assert_eq!(matches.len(), 3, "one match per query row");
    for (i, m) in matches.iter().enumerate() {
        assert_eq!(m.query_idx as usize, i, "query_idx mismatch at row {i}");
        assert_eq!(
            m.train_idx as usize, i,
            "self-match should map each row to itself"
        );
        assert!(
            m.distance.abs() < 1e-5,
            "self-match L2 distance must be zero, got {} at row {i}",
            m.distance
        );
    }
}

// ── Test 2: NORM_L1 self-match → identity, distance 0 ────────────────────────

#[test]
fn bfmatcher_norm_l1_self_match_identity() {
    let descriptors_data = [10.0f32, 20.0, 30.0, -1.0, -2.0, -3.0, 0.0, 0.5, -0.5];
    let mat = make_f32_mat(3, 3, &descriptors_data);

    let matcher = BFMatcher::new(NORM_L1).expect("NORM_L1 should be accepted");
    let matches = matcher
        .match_descriptors(&mat, &mat)
        .expect("self-match must succeed");

    assert_eq!(matches.len(), 3, "one match per query row");
    for (i, m) in matches.iter().enumerate() {
        assert_eq!(m.query_idx as usize, i, "query_idx mismatch at row {i}");
        assert_eq!(
            m.train_idx as usize, i,
            "self-match should map each row to itself"
        );
        assert!(
            m.distance.abs() < 1e-5,
            "self-match L1 distance must be zero, got {} at row {i}",
            m.distance
        );
    }
}

// ── Test 3: NORM_L2 known distance (3-4-5 Pythagorean) ───────────────────────

#[test]
fn bfmatcher_norm_l2_pythagorean_distance() {
    // query:  [3.0, 0.0, 0.0, 0.0]
    // train:  [0.0, 4.0, 0.0, 0.0]
    // L2 distance = √(3² + 4²) = 5.0
    let query_data = [3.0f32, 0.0, 0.0, 0.0];
    let train_data = [0.0f32, 4.0, 0.0, 0.0];

    let query = make_f32_mat(1, 4, &query_data);
    let train = make_f32_mat(1, 4, &train_data);

    let matcher = BFMatcher::new(NORM_L2).expect("NORM_L2 should be accepted");
    let matches = matcher
        .match_descriptors(&query, &train)
        .expect("match must succeed");

    assert_eq!(matches.len(), 1, "one query row → one match");
    let dist = matches[0].distance;
    assert!(
        (dist - 5.0).abs() < 1e-4,
        "expected L2 distance 5.0 (3-4-5 triple), got {}",
        dist
    );
    assert_eq!(matches[0].query_idx, 0);
    assert_eq!(matches[0].train_idx, 0);
}

// ── Test 4: dtype mismatch → error ───────────────────────────────────────────

#[test]
fn bfmatcher_norm_l2_rejects_u8_mat() {
    // query is CV_8UC1 (for binary descriptors), train is CV_32FC1.
    // NORM_L2 requires CV_32FC1 — query has wrong dtype → error.
    let u8_mat = Mat {
        data: vec![0u8; 2 * 4],
        rows: 2,
        cols: 4,
        step: 4,
        mat_type: MatType::CV_8UC1,
    };
    let f32_mat = make_f32_mat(2, 4, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

    let matcher = BFMatcher::new(NORM_L2).expect("NORM_L2 should be accepted");

    // u8 query against f32 train: dtype validation fires on query.
    let result = matcher.match_descriptors(&u8_mat, &f32_mat);
    assert!(
        result.is_err(),
        "NORM_L2 with CV_8UC1 query must return an error"
    );

    // f32 query against u8 train: dtype validation fires on train.
    let result2 = matcher.match_descriptors(&f32_mat, &u8_mat);
    assert!(
        result2.is_err(),
        "NORM_L2 with CV_8UC1 train must return an error"
    );
}

// ── Test 5: knn_match(k=2) with NORM_L2 → sorted ascending ──────────────────

#[test]
fn bfmatcher_norm_l2_knn_sorted_ascending() {
    // Query row Q0 = [0.0, 0.0]
    // Train rows:
    //   T0 = [1.0, 0.0]  →  L2 dist = 1.0
    //   T1 = [3.0, 0.0]  →  L2 dist = 3.0
    //   T2 = [0.0, 5.0]  →  L2 dist = 5.0
    let query_data = [0.0f32, 0.0];
    let train_data = [1.0f32, 0.0, 3.0, 0.0, 0.0, 5.0];

    let query = make_f32_mat(1, 2, &query_data);
    let train = make_f32_mat(3, 2, &train_data);

    let matcher = BFMatcher::new(NORM_L2).expect("NORM_L2 should be accepted");
    let knn = matcher
        .knn_match(&query, &train, 2)
        .expect("knn_match must succeed");

    assert_eq!(knn.len(), 1, "one query row → one outer entry");
    let row = &knn[0];
    assert_eq!(row.len(), 2, "k=2 gives 2 candidates");

    // Must be sorted ascending by distance.
    assert!(
        row[0].distance <= row[1].distance,
        "knn must be ascending: {} > {}",
        row[0].distance,
        row[1].distance
    );

    // Nearest should be T0 (dist 1.0), second nearest T1 (dist 3.0).
    assert_eq!(row[0].train_idx, 0, "nearest should be T0");
    assert!(
        (row[0].distance - 1.0).abs() < 1e-4,
        "T0 L2 distance expected 1.0, got {}",
        row[0].distance
    );

    assert_eq!(row[1].train_idx, 1, "second nearest should be T1");
    assert!(
        (row[1].distance - 3.0).abs() < 1e-4,
        "T1 L2 distance expected 3.0, got {}",
        row[1].distance
    );

    // Every entry must reference its own query row.
    for m in row {
        assert_eq!(m.query_idx, 0, "query_idx must be 0");
    }
}

// ── Bonus: NORM_L2SQR accepted and produces squared distances ─────────────────

#[test]
fn bfmatcher_norm_l2sqr_produces_squared_distance() {
    // Same 3-4-5 setup: L2SQR distance = 3² + 4² = 25.0
    let query_data = [3.0f32, 0.0, 0.0, 0.0];
    let train_data = [0.0f32, 4.0, 0.0, 0.0];

    let query = make_f32_mat(1, 4, &query_data);
    let train = make_f32_mat(1, 4, &train_data);

    let matcher = BFMatcher::new(NORM_L2SQR).expect("NORM_L2SQR should be accepted");
    let matches = matcher
        .match_descriptors(&query, &train)
        .expect("match must succeed");

    assert_eq!(matches.len(), 1);
    let dist = matches[0].distance;
    assert!(
        (dist - 25.0).abs() < 1e-4,
        "expected L2SQR distance 25.0 (3² + 4²), got {}",
        dist
    );
}
