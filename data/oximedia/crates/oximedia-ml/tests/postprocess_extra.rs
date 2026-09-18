//! Extra postprocess helpers introduced in Wave 2 Slice D:
//! `nms`, `iou`, `l2_normalize`, `cosine_similarity`.

use oximedia_ml::{cosine_similarity, iou, l2_normalize, nms, BoundingBox};

#[test]
fn nms_suppresses_overlapping_boxes() {
    // Box A: high score, large.
    // Box B: strongly overlaps A (IoU > 0.45), lower score → suppressed.
    // Box C: disjoint from both → kept.
    let boxes = vec![
        BoundingBox::new(0.0, 0.0, 10.0, 10.0),
        BoundingBox::new(1.0, 1.0, 10.0, 10.0),
        BoundingBox::new(50.0, 50.0, 60.0, 60.0),
    ];
    let scores = vec![0.9_f32, 0.8, 0.7];
    let kept = nms(&boxes, &scores, 0.45);
    assert_eq!(kept.len(), 2);
    // Highest score must be first.
    assert_eq!(kept[0], 0);
    // Disjoint box must also be kept (it is the only remaining candidate).
    assert!(kept.contains(&2));
    // Overlapping box must be suppressed.
    assert!(!kept.contains(&1));
}

#[test]
fn nms_keeps_disjoint_boxes() {
    let boxes = vec![
        BoundingBox::new(0.0, 0.0, 10.0, 10.0),
        BoundingBox::new(20.0, 20.0, 30.0, 30.0),
        BoundingBox::new(40.0, 40.0, 50.0, 50.0),
    ];
    let scores = vec![0.9_f32, 0.8, 0.7];
    let kept = nms(&boxes, &scores, 0.45);
    assert_eq!(kept.len(), 3);
    assert_eq!(kept[0], 0);
    assert_eq!(kept[1], 1);
    assert_eq!(kept[2], 2);
}

#[test]
fn iou_self_is_one() {
    let b = BoundingBox::new(5.0, 5.0, 15.0, 15.0);
    assert!(b.area() > 0.0);
    assert!((iou(&b, &b) - 1.0).abs() < 1e-6);
}

#[test]
fn iou_disjoint_is_zero() {
    let a = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
    let b = BoundingBox::new(20.0, 20.0, 30.0, 30.0);
    assert_eq!(iou(&a, &b), 0.0);
}

#[test]
fn l2_normalize_round_trip() {
    let mut v = vec![1.5_f32, 2.5, -0.5, 4.0];
    l2_normalize(&mut v);
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-5);
}

#[test]
fn l2_normalize_zero_vector_is_noop() {
    let mut v = vec![0.0_f32, 0.0, 0.0];
    l2_normalize(&mut v);
    assert!(v.iter().all(|&x| x == 0.0));
}

#[test]
fn cosine_similarity_of_unit_vectors() {
    // Construct two identical unit vectors → cosine ≈ 1.0
    let mut a = vec![0.3_f32, 0.6, 0.4];
    l2_normalize(&mut a);
    let b = a.clone();
    assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-5);

    // Construct two orthogonal unit vectors → cosine ≈ 0.0
    let x = [1.0_f32, 0.0, 0.0];
    let y = [0.0_f32, 1.0, 0.0];
    assert!(cosine_similarity(&x, &y).abs() < 1e-6);
}

#[test]
fn cosine_similarity_of_zero_vector_is_zero() {
    let a = [0.0_f32; 4];
    let b = [1.0_f32, 2.0, 3.0, 4.0];
    assert_eq!(cosine_similarity(&a, &b), 0.0);
    assert_eq!(cosine_similarity(&b, &a), 0.0);
}
