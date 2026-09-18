//! Integration tests for pulldown_detect (L40).
//!
//! Tests verify that a synthetic 3:2 cadence sequence (combing scores
//! [H, L, L, H, L] repeated) is correctly identified as `Pulldown23`,
//! and that a fully progressive (always-low-combing) sequence is identified
//! as `Progressive`.

use oximedia_video::pulldown_detect::{Cadence, CadenceDetector, FieldMetrics};

/// Repeat a 5-element cadence pattern to fill `count` metrics entries.
///
/// The 3:2 pulldown cadence has the combing-score pattern H, L, L, H, L
/// where H > 0.04 (combed) and L ≤ 0.04 (clean).
fn synthetic_pulldown23_metrics(count: usize) -> Vec<FieldMetrics> {
    // H = heavily combed, L = clean
    let pattern = [0.15f32, 0.01, 0.01, 0.12, 0.01];
    (0..count)
        .map(|i| FieldMetrics {
            frame_number: i as u64,
            combing_score: pattern[i % 5],
            tff: true,
        })
        .collect()
}

fn synthetic_progressive_metrics(count: usize) -> Vec<FieldMetrics> {
    (0..count)
        .map(|i| FieldMetrics {
            frame_number: i as u64,
            combing_score: 0.005,
            tff: false,
        })
        .collect()
}

fn synthetic_interlaced_metrics(count: usize) -> Vec<FieldMetrics> {
    (0..count)
        .map(|i| FieldMetrics {
            frame_number: i as u64,
            combing_score: 0.20,
            tff: true,
        })
        .collect()
}

#[test]
fn test_pulldown23_detected_from_synthetic_cadence() {
    let window = 10; // must be >= 5 for a full cycle
    let mut detector = CadenceDetector::new(window);

    let metrics = synthetic_pulldown23_metrics(30);
    for m in metrics {
        detector.push(m);
    }

    let cadence = detector.current_cadence();
    assert!(
        matches!(cadence, Cadence::Pulldown23 | Cadence::Pulldown32),
        "expected Pulldown23 (or Pulldown32) cadence, got {cadence:?}"
    );
}

#[test]
fn test_progressive_cadence_detected() {
    let mut detector = CadenceDetector::new(10);

    let metrics = synthetic_progressive_metrics(30);
    for m in metrics {
        detector.push(m);
    }

    let cadence = detector.current_cadence();
    assert_eq!(
        cadence,
        Cadence::Progressive,
        "uniform-clean score should yield Progressive cadence, got {cadence:?}"
    );
}

#[test]
fn test_interlaced_cadence_detected() {
    let mut detector = CadenceDetector::new(10);

    let metrics = synthetic_interlaced_metrics(30);
    for m in metrics {
        detector.push(m);
    }

    let cadence = detector.current_cadence();
    assert_eq!(
        cadence,
        Cadence::Interlaced,
        "uniformly high combing should yield Interlaced cadence, got {cadence:?}"
    );
}

#[test]
fn test_cadence_detector_empty_returns_unknown() {
    let detector = CadenceDetector::new(10);
    let cadence = detector.current_cadence();
    assert_eq!(
        cadence,
        Cadence::Unknown,
        "empty detector should return Unknown cadence"
    );
}

#[test]
fn test_cadence_detector_too_few_frames_unknown() {
    // Feed only 2 frames into a window=10 detector — not enough to classify.
    let mut detector = CadenceDetector::new(10);
    detector.push(FieldMetrics {
        frame_number: 0,
        combing_score: 0.15,
        tff: true,
    });
    detector.push(FieldMetrics {
        frame_number: 1,
        combing_score: 0.01,
        tff: true,
    });

    let cadence = detector.current_cadence();
    // Too few frames to detect any reliable pattern.
    assert!(
        matches!(
            cadence,
            Cadence::Unknown | Cadence::Progressive | Cadence::Interlaced
        ),
        "with only 2 frames, cadence should be Unknown/simple, got {cadence:?}"
    );
}
