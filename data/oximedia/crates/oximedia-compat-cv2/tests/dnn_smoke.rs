//! Integration tests for the `cv2.dnn` compatibility layer.
//!
//! Tests are gated behind `#[cfg(feature = "dnn")]` so they only compile when
//! the optional `dnn` feature is enabled.  Tests that require a real ONNX model
//! file are marked `#[ignore]` and look up the model path via the
//! `OXIMEDIA_TEST_ONNX_MODEL` environment variable.

#![cfg(feature = "dnn")]

use std::path::Path;

use oximedia_compat_cv2::dnn::{blob_from_image, nms_boxes, read_net_from_onnx};
use oximedia_compat_cv2::{Cv2Error, Mat, MatType, Rect};

// ── Net loading ───────────────────────────────────────────────────────────────

#[test]
fn read_net_from_onnx_missing_file_returns_dnn_error() {
    let path_buf = std::env::temp_dir().join("oximedia_compat_cv2_does_not_exist.onnx");
    let path = path_buf.as_path();
    match read_net_from_onnx(path) {
        Ok(_) => panic!("expected error when ONNX path is missing"),
        Err(Cv2Error::Dnn(msg)) => {
            assert!(
                msg.contains("oximedia_compat_cv2_does_not_exist") || msg.contains("ONNX"),
                "expected Dnn error mentioning the path or ONNX, got {msg:?}"
            );
        }
        Err(other) => panic!("expected Cv2Error::Dnn, got {other:?}"),
    }
}

// ── blob_from_image ───────────────────────────────────────────────────────────

#[test]
fn blob_from_image_produces_planar_chw_blob_with_correct_shape() {
    // 4×4 BGR image, every pixel = (B=10, G=20, R=30).
    let mut bytes = Vec::with_capacity(4 * 4 * 3);
    for _ in 0..(4 * 4) {
        bytes.extend_from_slice(&[10u8, 20u8, 30u8]);
    }
    let mat = Mat::from_bgr_bytes(bytes, 4, 4);

    // Resize 4×4 → 2×2; mean=(0,0,0); scale=1/10; swap_rb=false; crop=false.
    let blob = blob_from_image(&mat, 0.1, (2, 2), (0.0, 0.0, 0.0), false, false)
        .expect("blob_from_image should succeed");

    assert_eq!(blob.mat_type, MatType::CV_32FC3);
    assert_eq!(blob.rows, 2);
    assert_eq!(blob.cols, 2);
    assert_eq!(blob.data.len(), 3 * 2 * 2 * 4);

    // Decode planar bytes → f32 values.
    let mut floats = Vec::with_capacity(3 * 2 * 2);
    for chunk in blob.data.chunks_exact(4) {
        let arr: [u8; 4] = chunk.try_into().expect("4-byte chunk");
        floats.push(f32::from_ne_bytes(arr));
    }

    // Every pixel was identical, so resizing must produce an identical 2×2 block.
    // After swap_rb=false: output channel order = [B, G, R] = [10, 20, 30].
    // After scale=0.1: [1.0, 2.0, 3.0].
    // Planar layout: c=0 plane (4 elems = 1.0), c=1 plane (= 2.0), c=2 plane (= 3.0).
    for i in 0..4 {
        assert!(
            (floats[i] - 1.0).abs() < 1e-5,
            "B-plane[{i}]={:?}",
            floats[i]
        );
    }
    for i in 4..8 {
        assert!(
            (floats[i] - 2.0).abs() < 1e-5,
            "G-plane[{i}]={:?}",
            floats[i]
        );
    }
    for i in 8..12 {
        assert!(
            (floats[i] - 3.0).abs() < 1e-5,
            "R-plane[{i}]={:?}",
            floats[i]
        );
    }
}

#[test]
fn blob_from_image_swap_rb_reorders_channels() {
    // 2-pixel BGR image where B=200, G=0, R=10 (deliberately asymmetric R/B).
    let mat = Mat::from_bgr_bytes(vec![200u8, 0u8, 10u8, 200u8, 0u8, 10u8], 1, 2);

    // No mean, no scale (=1.0), swap_rb=true: output channel order should be [R, G, B] = [10, 0, 200].
    let blob = blob_from_image(&mat, 1.0, (2, 1), (0.0, 0.0, 0.0), true, false)
        .expect("blob_from_image should succeed");

    let mut floats = Vec::with_capacity(3 * 2);
    for chunk in blob.data.chunks_exact(4) {
        let arr: [u8; 4] = chunk.try_into().expect("4-byte chunk");
        floats.push(f32::from_ne_bytes(arr));
    }

    // Planar layout: 2 elems per plane (W=2, H=1), 3 planes total.
    // Plane 0 (R after swap) = 10.0
    // Plane 1 (G) = 0.0
    // Plane 2 (B after swap) = 200.0
    assert!((floats[0] - 10.0).abs() < 1e-5);
    assert!((floats[1] - 10.0).abs() < 1e-5);
    assert!((floats[2] - 0.0).abs() < 1e-5);
    assert!((floats[3] - 0.0).abs() < 1e-5);
    assert!((floats[4] - 200.0).abs() < 1e-5);
    assert!((floats[5] - 200.0).abs() < 1e-5);
}

#[test]
fn blob_from_image_zero_size_errors() {
    let mat = Mat::from_bgr_bytes(vec![0u8; 12], 2, 2);
    match blob_from_image(&mat, 1.0, (0, 224), (0.0, 0.0, 0.0), false, false) {
        Err(Cv2Error::Dnn(_)) => {}
        other => panic!("expected Cv2Error::Dnn, got {other:?}"),
    }
}

#[test]
fn blob_from_image_grayscale_promoted_to_bgr() {
    // 2×2 grayscale image, every pixel = 100.
    let mat = Mat::from_gray_bytes(vec![100u8; 4], 2, 2);

    let blob = blob_from_image(&mat, 1.0, (2, 2), (0.0, 0.0, 0.0), false, false)
        .expect("grayscale blob should succeed");

    assert_eq!(blob.mat_type, MatType::CV_32FC3);
    let mut floats = Vec::with_capacity(blob.data.len() / 4);
    for chunk in blob.data.chunks_exact(4) {
        let arr: [u8; 4] = chunk.try_into().expect("4-byte chunk");
        floats.push(f32::from_ne_bytes(arr));
    }
    // All channels should be 100 because gray was replicated.
    for v in &floats {
        assert!((v - 100.0).abs() < 1e-5, "got {v}");
    }
}

// ── nms_boxes ─────────────────────────────────────────────────────────────────

#[test]
fn nms_boxes_keeps_highest_score_when_overlapping() {
    // Three heavily-overlapping boxes; only the top-scoring one should survive.
    let boxes = vec![
        Rect {
            x: 0,
            y: 0,
            width: 10,
            height: 10,
        },
        Rect {
            x: 1,
            y: 1,
            width: 10,
            height: 10,
        },
        Rect {
            x: 2,
            y: 2,
            width: 10,
            height: 10,
        },
    ];
    let scores = vec![0.5, 0.9, 0.8];
    let kept = nms_boxes(&boxes, &scores, 0.0, 0.3);
    assert_eq!(kept, vec![1]);
}

#[test]
fn nms_boxes_empty_input_returns_empty() {
    let boxes: Vec<Rect> = Vec::new();
    let scores: Vec<f32> = Vec::new();
    let kept = nms_boxes(&boxes, &scores, 0.5, 0.5);
    assert!(kept.is_empty());
}

#[test]
fn nms_boxes_no_overlap_keeps_all() {
    let boxes = vec![
        Rect {
            x: 0,
            y: 0,
            width: 5,
            height: 5,
        },
        Rect {
            x: 100,
            y: 0,
            width: 5,
            height: 5,
        },
        Rect {
            x: 0,
            y: 100,
            width: 5,
            height: 5,
        },
    ];
    let scores = vec![0.6, 0.7, 0.8];
    let mut kept = nms_boxes(&boxes, &scores, 0.0, 0.5);
    kept.sort_unstable();
    assert_eq!(kept, vec![0, 1, 2]);
}

#[test]
fn nms_boxes_score_threshold_filters() {
    let boxes = vec![
        Rect {
            x: 0,
            y: 0,
            width: 10,
            height: 10,
        },
        Rect {
            x: 50,
            y: 50,
            width: 10,
            height: 10,
        },
    ];
    let scores = vec![0.1, 0.9];
    let kept = nms_boxes(&boxes, &scores, 0.5, 0.5);
    // First box is below the score threshold and must be discarded.
    assert_eq!(kept, vec![1]);
}

// ── End-to-end model test (requires an ONNX file) ─────────────────────────────

#[test]
#[ignore = "requires OXIMEDIA_TEST_ONNX_MODEL pointing at a valid .onnx file"]
fn forward_runs_on_real_model() {
    let path = match std::env::var("OXIMEDIA_TEST_ONNX_MODEL") {
        Ok(p) => p,
        Err(_) => return,
    };
    let net = match read_net_from_onnx(Path::new(&path)) {
        Ok(n) => n,
        Err(e) => panic!("load model failed: {e:?}"),
    };
    // Synthesize an input matching the cached input name's expected shape.
    let dummy = Mat::from_bgr_bytes(vec![0u8; 224 * 224 * 3], 224, 224);
    let blob = match blob_from_image(
        &dummy,
        1.0 / 255.0,
        (224, 224),
        (0.0, 0.0, 0.0),
        true,
        false,
    ) {
        Ok(b) => b,
        Err(e) => panic!("blob failed: {e:?}"),
    };
    if let Err(e) = net.forward(&blob) {
        panic!("forward failed: {e:?}");
    }
}
