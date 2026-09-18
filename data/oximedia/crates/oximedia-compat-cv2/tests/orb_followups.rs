//! Integration tests for ORB follow-ups (Run 4 Slice E-lib):
//!
//! 1. `BFMatcher::new(NORM_HAMMING)` self-match — every match has
//!    `query_idx == train_idx` and zero distance.
//! 2. `BFMatcher::with_cross_check(true)` filters non-mutual matches.  The
//!    fixture is hand-crafted so descriptor `A0` has `B0` as nearest in train,
//!    but `B0`'s nearest in query is `A1` (not `A0`) — without cross-check
//!    `(A0 → B0)` survives, with cross-check it does not.
//! 3. `BFMatcher::knn_match(k=2)` returns at most 2 candidates per query, in
//!    ascending distance order.
//! 4. `Orb::detect_and_compute` with an all-zero mask returns 0 keypoints.

use oximedia_compat_cv2::constants::NORM_HAMMING;
use oximedia_compat_cv2::features::{BFMatcher, Orb};
use oximedia_compat_cv2::mat::{Mat, MatType};

/// Build a deterministic 32-byte descriptor row from a list of byte overrides.
///
/// All other bytes are zero.  Returns a 32-element `Vec<u8>`.
fn descriptor_row(overrides: &[(usize, u8)]) -> Vec<u8> {
    let mut row = vec![0u8; 32];
    for &(i, b) in overrides {
        row[i] = b;
    }
    row
}

/// Pack a list of 32-byte rows into a `CV_8UC1` `Mat`.
fn descriptor_mat(rows: &[Vec<u8>]) -> Mat {
    let n = rows.len();
    let mut data = Vec::with_capacity(n * 32);
    for r in rows {
        assert_eq!(r.len(), 32);
        data.extend_from_slice(r);
    }
    Mat {
        data,
        rows: n,
        cols: 32,
        step: 32,
        mat_type: MatType::CV_8UC1,
    }
}

#[test]
fn bfmatcher_self_match_returns_identity() {
    // Three distinguishable rows.  Self-match must pair every row with itself
    // at zero distance.
    let rows = vec![
        descriptor_row(&[(0, 0xAA), (5, 0x55)]),
        descriptor_row(&[(0, 0x01), (1, 0x02), (2, 0x04)]),
        descriptor_row(&[(31, 0xFF)]),
    ];
    let mat = descriptor_mat(&rows);

    let matcher = BFMatcher::new(NORM_HAMMING).expect("NORM_HAMMING is supported");
    let matches = matcher.match_descriptors(&mat, &mat).expect("self match");

    assert_eq!(matches.len(), 3);
    for (i, m) in matches.iter().enumerate() {
        assert_eq!(m.query_idx as usize, i, "query_idx mismatch at row {i}");
        assert_eq!(m.train_idx as usize, i, "train_idx mismatch at row {i}");
        assert!(m.distance == 0.0, "self distance must be zero at row {i}");
    }
}

#[test]
fn bfmatcher_cross_check_drops_non_mutual_matches() {
    // Hand-crafted query (A) and train (B) so that:
    //   dist(A0, B0) = 2     dist(A0, B1) = 255
    //   dist(A1, B0) = 1     dist(A1, B1) = 225
    //   dist(A2, B0) = 254   dist(A2, B1) = 32
    //
    // Forward (A → B) yields:
    //   A0 → B0   (nearest is B0)
    //   A1 → B0   (nearest is B0)
    //   A2 → B1   (nearest is B1)
    //
    // Backward (B → A) yields:
    //   B0 → A1   (nearest is A1, NOT A0)
    //   B1 → A2   (nearest is A2)
    //
    // Cross-check therefore keeps only (A1 → B0) and (A2 → B1) — A0 → B0
    // is non-mutual and must drop.

    // A: zeros, bit-0 set, all-ones
    let a_rows = vec![
        descriptor_row(&[]),          // A0: all zero
        descriptor_row(&[(0, 0x01)]), // A1: bit 0 set
        vec![0xFFu8; 32],             // A2: all ones
    ];
    // B: 0x03 then zeros, then 0xFE/0xFE…
    let b_rows = vec![
        descriptor_row(&[(0, 0x03)]), // B0: bits 0+1 set
        vec![0xFEu8; 32],             // B1: 0xFE everywhere
    ];

    let a = descriptor_mat(&a_rows);
    let b = descriptor_mat(&b_rows);

    let matcher = BFMatcher::new(NORM_HAMMING).expect("NORM_HAMMING is supported");

    // Forward (no cross-check): expect 3 matches, A0→B0 included.
    let forward = matcher.match_descriptors(&a, &b).expect("forward match");
    assert_eq!(
        forward.len(),
        3,
        "forward should produce one match per query row"
    );
    let forward_pairs: Vec<(u32, u32)> =
        forward.iter().map(|m| (m.query_idx, m.train_idx)).collect();
    assert!(
        forward_pairs.contains(&(0, 0)),
        "forward must include A0 → B0; got {:?}",
        forward_pairs
    );
    assert!(
        forward_pairs.contains(&(1, 0)),
        "forward must include A1 → B0"
    );
    assert!(
        forward_pairs.contains(&(2, 1)),
        "forward must include A2 → B1"
    );

    // With cross-check: A0 → B0 is non-mutual (B0's nearest is A1) and must drop.
    let mutual = BFMatcher::new(NORM_HAMMING)
        .expect("NORM_HAMMING")
        .with_cross_check(true)
        .match_descriptors(&a, &b)
        .expect("cross-check match");
    let mutual_pairs: Vec<(u32, u32)> = mutual.iter().map(|m| (m.query_idx, m.train_idx)).collect();
    assert!(
        !mutual_pairs.contains(&(0, 0)),
        "cross-check must drop non-mutual A0 → B0; got {:?}",
        mutual_pairs
    );
    assert!(
        mutual_pairs.contains(&(1, 0)),
        "cross-check must keep mutual A1 → B0; got {:?}",
        mutual_pairs
    );
    assert!(
        mutual_pairs.contains(&(2, 1)),
        "cross-check must keep mutual A2 → B1; got {:?}",
        mutual_pairs
    );
    assert_eq!(
        mutual.len(),
        2,
        "cross-check should yield exactly two mutual pairs"
    );
}

#[test]
fn bfmatcher_knn_returns_k_sorted_candidates() {
    // Query rows:
    //   Q0 = all zero
    //   Q1 = 0x01 at byte 0
    // Train rows (distances from Q0 in parens):
    //   T0 = 0x03 at byte 0   (dist Q0→T0 = 2;   Q1→T0 = 1)
    //   T1 = 0x07 at byte 0   (dist Q0→T1 = 3;   Q1→T1 = 2)
    //   T2 = 0x0F at byte 0   (dist Q0→T2 = 4;   Q1→T2 = 3)
    //   T3 = 0xFF at byte 0   (dist Q0→T3 = 8;   Q1→T3 = 7)
    let query_rows = vec![descriptor_row(&[]), descriptor_row(&[(0, 0x01)])];
    let train_rows = vec![
        descriptor_row(&[(0, 0x03)]),
        descriptor_row(&[(0, 0x07)]),
        descriptor_row(&[(0, 0x0F)]),
        descriptor_row(&[(0, 0xFF)]),
    ];
    let q = descriptor_mat(&query_rows);
    let t = descriptor_mat(&train_rows);

    let matcher = BFMatcher::new(NORM_HAMMING).expect("NORM_HAMMING is supported");
    let knn = matcher.knn_match(&q, &t, 2).expect("knn match");

    assert_eq!(knn.len(), 2, "outer Vec must have one entry per query row");

    // Q0 nearest 2: T0 (2), T1 (3)
    let q0 = &knn[0];
    assert_eq!(q0.len(), 2, "k=2 should yield 2 candidates per query");
    assert_eq!(q0[0].train_idx, 0);
    assert!(q0[0].distance == 2.0);
    assert_eq!(q0[1].train_idx, 1);
    assert!(q0[1].distance == 3.0);
    assert!(
        q0[0].distance <= q0[1].distance,
        "knn results must be sorted ascending by distance"
    );

    // Q1 nearest 2: T0 (1), T1 (2)
    let q1 = &knn[1];
    assert_eq!(q1.len(), 2);
    assert_eq!(q1[0].train_idx, 0);
    assert!(q1[0].distance == 1.0);
    assert_eq!(q1[1].train_idx, 1);
    assert!(q1[1].distance == 2.0);

    // Every entry must reference its own query row.
    for (qi, row) in knn.iter().enumerate() {
        for m in row {
            assert_eq!(m.query_idx as usize, qi);
        }
    }
}

#[test]
fn orb_zero_mask_yields_no_keypoints() {
    // 64×64 image identical to the inline FAST-friendly fixture.
    let mut data = vec![0u8; 64 * 64];
    let sites: &[(usize, usize)] = &[(20, 20), (44, 20), (20, 44), (44, 44), (32, 32)];
    for &(cx, cy) in sites {
        for dy in 0..3 {
            for dx in 0..3 {
                let x = cx + dx;
                let y = cy + dy;
                if x < 64 && y < 64 {
                    data[y * 64 + x] = 230;
                }
            }
        }
    }
    let img = Mat::from_gray_bytes(data, 64, 64);
    let orb = Orb {
        num_features: 30,
        n_levels: 1,
        edge_threshold: 16,
        ..Orb::default()
    };

    // Sanity: without a mask the corner image must produce keypoints.
    let (kps_unmasked, _) = orb
        .detect_and_compute(&img, None)
        .expect("detect_and_compute without mask");
    assert!(
        !kps_unmasked.is_empty(),
        "fixture should yield keypoints when no mask is supplied"
    );

    // All-zero mask — every pixel is masked off, so detect_and_compute must
    // return an empty keypoint vector and a zero-row descriptor mat.
    let zero_mask = Mat::from_gray_bytes(vec![0u8; 64 * 64], 64, 64);
    let (kps_masked, desc_masked) = orb
        .detect_and_compute(&img, Some(&zero_mask))
        .expect("detect_and_compute with zero mask");
    assert!(
        kps_masked.is_empty(),
        "zero mask must drop every keypoint; got {} keypoints",
        kps_masked.len()
    );
    assert_eq!(desc_masked.rows, 0);
    assert_eq!(desc_masked.data.len(), 0);
    assert_eq!(desc_masked.cols, 32);
}

#[test]
fn orb_mask_size_mismatch_errors() {
    let img = Mat::from_gray_bytes(vec![0u8; 64 * 64], 64, 64);
    // Mask is 32×32 — does not match.
    let bad_mask = Mat::from_gray_bytes(vec![0u8; 32 * 32], 32, 32);
    let orb = Orb::new(10);
    let res = orb.detect_and_compute(&img, Some(&bad_mask));
    assert!(res.is_err(), "size-mismatched mask must produce an error");
}

#[test]
fn orb_mask_wrong_dtype_errors() {
    let img = Mat::from_gray_bytes(vec![0u8; 64 * 64], 64, 64);
    // Build a CV_32FC1 mask of the right WxH — wrong dtype must error.
    let wrong_dtype_mask = Mat::new(64, 64, MatType::CV_32FC1);
    let orb = Orb::new(10);
    let res = orb.detect_and_compute(&img, Some(&wrong_dtype_mask));
    assert!(
        res.is_err(),
        "non-CV_8UC1 mask must produce an UnsupportedDtype error"
    );
}
