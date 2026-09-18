//! Tests for 4-point Vandermonde BD-Rate / BD-PSNR calculation.

use oximedia_bench::comparison::{bd_psnr, bd_rate, BdDeltaError, BdRdPoint};

// ── identical curves ───────────────────────────────────────────────────────────

#[test]
fn test_bd_rate_identical_curves_is_zero() {
    let curve = [
        BdRdPoint::new(100.0, 32.0),
        BdRdPoint::new(200.0, 35.0),
        BdRdPoint::new(400.0, 38.0),
        BdRdPoint::new(800.0, 41.0),
    ];
    let result = bd_rate(&curve, &curve).expect("bd_rate ok");
    assert!(
        result.abs() < 1e-6,
        "identical curves → BD-Rate ≈ 0 %, got {result}"
    );
}

#[test]
fn test_bd_psnr_identical_curves_is_zero() {
    let curve = [
        BdRdPoint::new(100.0, 32.0),
        BdRdPoint::new(200.0, 35.0),
        BdRdPoint::new(400.0, 38.0),
        BdRdPoint::new(800.0, 41.0),
    ];
    let result = bd_psnr(&curve, &curve).expect("bd_psnr ok");
    assert!(
        result.abs() < 1e-6,
        "identical curves → BD-PSNR ≈ 0 dB, got {result}"
    );
}

// ── uniformly shifted rate ─────────────────────────────────────────────────────

/// When the test curve has the same PSNR but each bitrate multiplied by a
/// factor k, BD-Rate should be approximately (k − 1) × 100 %.
#[test]
fn test_bd_rate_uniform_bitrate_shift() {
    let anchor = [
        BdRdPoint::new(100.0, 32.0),
        BdRdPoint::new(200.0, 35.0),
        BdRdPoint::new(400.0, 38.0),
        BdRdPoint::new(800.0, 41.0),
    ];
    // Test curve: same PSNR, bitrate × 2 → expected BD-Rate ≈ +100 %
    let test = [
        BdRdPoint::new(200.0, 32.0),
        BdRdPoint::new(400.0, 35.0),
        BdRdPoint::new(800.0, 38.0),
        BdRdPoint::new(1600.0, 41.0),
    ];
    let result = bd_rate(&anchor, &test).expect("bd_rate ok");
    // The cubic interpolation over log-rate is exact for a log-linear relationship,
    // so a 2× shift should give exactly 100 %.
    assert!(
        (result - 100.0).abs() < 1.0,
        "2× bitrate shift → BD-Rate ≈ 100 %, got {result}"
    );
}

// ── BD-PSNR uniform PSNR shift ────────────────────────────────────────────────

#[test]
fn test_bd_psnr_uniform_quality_shift() {
    let anchor = [
        BdRdPoint::new(100.0, 32.0),
        BdRdPoint::new(200.0, 35.0),
        BdRdPoint::new(400.0, 38.0),
        BdRdPoint::new(800.0, 41.0),
    ];
    // Test curve: PSNR + 2 dB at same bitrates.
    let test = [
        BdRdPoint::new(100.0, 34.0),
        BdRdPoint::new(200.0, 37.0),
        BdRdPoint::new(400.0, 40.0),
        BdRdPoint::new(800.0, 43.0),
    ];
    let result = bd_psnr(&anchor, &test).expect("bd_psnr ok");
    assert!(
        (result - 2.0).abs() < 0.5,
        "+2 dB shift → BD-PSNR ≈ 2 dB, got {result}"
    );
}

// ── no-overlap error ──────────────────────────────────────────────────────────

#[test]
fn test_bd_rate_no_overlap_returns_error() {
    // anchor PSNR range [30, 36], test PSNR range [40, 46] — no overlap
    let anchor = [
        BdRdPoint::new(100.0, 30.0),
        BdRdPoint::new(200.0, 32.0),
        BdRdPoint::new(400.0, 34.0),
        BdRdPoint::new(800.0, 36.0),
    ];
    let test = [
        BdRdPoint::new(100.0, 40.0),
        BdRdPoint::new(200.0, 42.0),
        BdRdPoint::new(400.0, 44.0),
        BdRdPoint::new(800.0, 46.0),
    ];
    let result = bd_rate(&anchor, &test);
    assert!(
        matches!(result, Err(BdDeltaError::NoOverlap)),
        "non-overlapping PSNR ranges should return NoOverlap, got {result:?}"
    );
}

// ── non-positive rate error ───────────────────────────────────────────────────

#[test]
fn test_bd_rate_zero_rate_returns_error() {
    let anchor = [
        BdRdPoint::new(0.0, 32.0), // invalid
        BdRdPoint::new(200.0, 35.0),
        BdRdPoint::new(400.0, 38.0),
        BdRdPoint::new(800.0, 41.0),
    ];
    let test = anchor;
    let result = bd_rate(&anchor, &test);
    assert!(
        matches!(result, Err(BdDeltaError::NonPositiveRate { .. })),
        "zero rate should return NonPositiveRate, got {result:?}"
    );
}

// ── ill-conditioned (collinear) detection ─────────────────────────────────────

#[test]
fn test_bd_rate_ill_conditioned_collinear() {
    // All four PSNR values the same → Vandermonde is singular.
    let anchor = [
        BdRdPoint::new(100.0, 35.0),
        BdRdPoint::new(200.0, 35.0),
        BdRdPoint::new(400.0, 35.0),
        BdRdPoint::new(800.0, 35.0),
    ];
    let test = anchor;
    let result = bd_rate(&anchor, &test);
    // IllConditioned OR NoOverlap are both acceptable here.
    assert!(
        matches!(
            result,
            Err(BdDeltaError::IllConditioned) | Err(BdDeltaError::NoOverlap)
        ),
        "collinear PSNR values should return IllConditioned or NoOverlap, got {result:?}"
    );
}
