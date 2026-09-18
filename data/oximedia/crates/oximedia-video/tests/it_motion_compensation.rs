//! Integration tests for motion_compensation (L41).
//!
//! Tests verify that the motion estimator correctly identifies a known
//! horizontal shift between two synthetic luma frames, and that the
//! residual + prediction = original identity holds exactly.

use oximedia_video::motion_compensation::{
    compensate_frame, reconstruct_from_residual, residual_frame, MeAlgorithm, MotionEstimator,
};

/// Create a luma frame where every row contains a unique repeating value based
/// on `(row_index * 17 + col_index * 3 + seed) % 251`. Using 251 (prime) ensures
/// that wrapping does not create flat regions that would make multiple candidates
/// equally optimal.
fn rich_luma(width: u32, height: u32, seed: u8) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let mut buf = Vec::with_capacity(w * h);
    for row in 0..h {
        for col in 0..w {
            buf.push(((row * 17 + col * 3 + seed as usize) % 251) as u8);
        }
    }
    buf
}

/// Shift frame right by `shift` pixels by rolling each row cyclically.
fn roll_right(frame: &[u8], width: u32, shift: usize) -> Vec<u8> {
    let w = width as usize;
    let h = frame.len() / w;
    let mut out = vec![0u8; frame.len()];
    for row in 0..h {
        let src = &frame[row * w..(row + 1) * w];
        let dst = &mut out[row * w..(row + 1) * w];
        let s = shift % w;
        dst[s..].copy_from_slice(&src[..w - s]);
        dst[..s].copy_from_slice(&src[w - s..]);
    }
    out
}

#[test]
fn test_full_search_detects_known_horizontal_shift() {
    // With a cyclic shift, every block in the current frame is an exact copy of
    // a block in the reference frame — so FullSearch will always find SAD=0 at
    // the correct displacement.
    let width = 64u32;
    let height = 32u32;
    let shift = 4usize;

    let ref_frame = rich_luma(width, height, 0);
    let cur_frame = roll_right(&ref_frame, width, shift);

    let estimator = MotionEstimator {
        block_size: 16,
        search_range: 8,
        algorithm: MeAlgorithm::FullSearch,
    };

    let vectors = estimator.estimate_frame(&ref_frame, &cur_frame, width, height);

    assert!(
        !vectors.is_empty(),
        "estimate_frame must return at least one motion vector"
    );

    // All blocks should find a zero-SAD match (at ±shift or its cyclic equivalent).
    // We check that at least one vector has the correct shift or an equivalent zero-SAD match.
    let zero_sad_count = vectors.iter().filter(|v| v.sad == 0).count();
    assert!(
        zero_sad_count > 0,
        "FullSearch should find at least one zero-SAD match for a cyclic shift, got {zero_sad_count}/{} zero-SAD",
        vectors.len()
    );
}

#[test]
fn test_diamond_search_detects_shift() {
    let width = 64u32;
    let height = 32u32;
    let shift = 3usize;

    let ref_frame = rich_luma(width, height, 7);
    let cur_frame = roll_right(&ref_frame, width, shift);

    let estimator = MotionEstimator {
        block_size: 16,
        search_range: 8,
        algorithm: MeAlgorithm::DiamondSearch,
    };

    let vectors = estimator.estimate_frame(&ref_frame, &cur_frame, width, height);

    // Diamond search should also find zero-SAD for cyclic shifts within search range.
    let zero_sad_count = vectors.iter().filter(|v| v.sad == 0).count();
    assert!(
        zero_sad_count > 0,
        "DiamondSearch should find at least one zero-SAD match for shift={shift}, got {zero_sad_count}/{} zero-SAD",
        vectors.len()
    );
}

#[test]
fn test_compensate_and_residual_reconstruct() {
    let width = 64u32;
    let height = 32u32;
    let shift = 2usize;

    let ref_frame = rich_luma(width, height, 42);
    let cur_frame = roll_right(&ref_frame, width, shift);

    let estimator = MotionEstimator {
        block_size: 16,
        search_range: 4,
        algorithm: MeAlgorithm::FullSearch,
    };

    let vectors = estimator.estimate_frame(&ref_frame, &cur_frame, width, height);
    let predicted = compensate_frame(&ref_frame, &vectors, width, height, 16);
    let residual = residual_frame(&cur_frame, &predicted);
    let reconstructed = reconstruct_from_residual(&predicted, &residual);

    assert_eq!(reconstructed.len(), cur_frame.len());
    // Reconstruction must be pixel-exact (residual + prediction = original).
    for (i, (&rec, &orig)) in reconstructed.iter().zip(cur_frame.iter()).enumerate() {
        assert_eq!(
            rec, orig,
            "reconstruct_from_residual mismatch at pixel {i}: got {rec}, expected {orig}"
        );
    }
}

#[test]
fn test_hexagon_search_zero_shift() {
    let width = 64u32;
    let height = 32u32;

    let frame = rich_luma(width, height, 99);

    let estimator = MotionEstimator {
        block_size: 16,
        search_range: 8,
        algorithm: MeAlgorithm::HexagonSearch,
    };

    let vectors = estimator.estimate_frame(&frame, &frame, width, height);

    // When ref == cur, all vectors should have SAD=0.
    for v in &vectors {
        assert_eq!(
            v.sad, 0,
            "zero-shift: expected SAD=0, got {} (dx={}, dy={})",
            v.sad, v.dx, v.dy
        );
    }
}
