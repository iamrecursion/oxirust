//! Tests for f64 RNG variants and the streaming API.
//!
//! All tests run on the CPU path and require no CUDA device.

use tensorlogic_oxicuda_rng::{RngEngine, RngEngineKind};

// ---------------------------------------------------------------------------
// uniform_f64
// ---------------------------------------------------------------------------

/// All 1024 uniform f64 values must lie in `[0.0, 1.0)`.
#[test]
fn uniform_f64_in_range() {
    let mut rng = RngEngine::new(RngEngineKind::Philox, 42).unwrap();
    let mut out = vec![0f64; 1024];
    rng.uniform_f64(&mut out).unwrap();
    for &v in &out {
        assert!(
            (0.0..1.0).contains(&v),
            "uniform_f64 sample {v} not in [0.0, 1.0)"
        );
    }
}

/// Same seed on two engines must produce identical f64 sequences.
#[test]
fn uniform_f64_deterministic_replay() {
    let mut rng1 = RngEngine::new(RngEngineKind::Xorwow, 99).unwrap();
    let mut rng2 = RngEngine::new(RngEngineKind::Xorwow, 99).unwrap();
    let mut out1 = vec![0f64; 512];
    let mut out2 = vec![0f64; 512];
    rng1.uniform_f64(&mut out1).unwrap();
    rng2.uniform_f64(&mut out2).unwrap();
    assert_eq!(out1, out2, "same seed must produce identical f64 sequence");
}

/// Two different seeds must produce different f64 sequences.
#[test]
fn uniform_f64_different_seeds_differ() {
    let mut rng1 = RngEngine::new(RngEngineKind::Mrg32k3a, 0).unwrap();
    let mut rng2 = RngEngine::new(RngEngineKind::Mrg32k3a, 1).unwrap();
    let mut out1 = vec![0f64; 256];
    let mut out2 = vec![0f64; 256];
    rng1.uniform_f64(&mut out1).unwrap();
    rng2.uniform_f64(&mut out2).unwrap();
    assert_ne!(
        out1, out2,
        "different seeds must produce different sequences"
    );
}

// ---------------------------------------------------------------------------
// normal_f64
// ---------------------------------------------------------------------------

/// Mean ≈ 0 and std ≈ 1 for N(0, 1) with 10 000 samples (tolerance 0.05).
#[test]
fn normal_f64_approximate_stats() {
    let mut rng = RngEngine::new(RngEngineKind::Philox, 123).unwrap();
    let n = 10_000usize;
    let mut out = vec![0f64; n];
    rng.normal_f64(&mut out, 0.0, 1.0).unwrap();

    let nf = n as f64;
    let mean: f64 = out.iter().sum::<f64>() / nf;
    let var: f64 = out.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / nf;
    let std = var.sqrt();

    assert!(mean.abs() < 0.05, "normal_f64 mean {mean} too far from 0");
    assert!(
        (std - 1.0).abs() < 0.05,
        "normal_f64 std {std} too far from 1"
    );
}

/// All 5 000 normally distributed f64 samples must be finite.
#[test]
fn normal_f64_all_finite() {
    let mut rng = RngEngine::new(RngEngineKind::Mrg32k3a, 555).unwrap();
    let mut out = vec![0f64; 5_000];
    rng.normal_f64(&mut out, 0.0, 1.0).unwrap();
    for (i, &v) in out.iter().enumerate() {
        assert!(v.is_finite(), "element {i} is non-finite: {v}");
    }
}

/// Non-standard parameters: mean=5, std=0.5 — check empirical statistics.
#[test]
fn normal_f64_nonstandard_params() {
    let target_mean = 5.0_f64;
    let target_std = 0.5_f64;

    let mut rng = RngEngine::new(RngEngineKind::Xorwow, 77).unwrap();
    let n = 10_000usize;
    let mut out = vec![0f64; n];
    rng.normal_f64(&mut out, target_mean, target_std).unwrap();

    let nf = n as f64;
    let mean: f64 = out.iter().sum::<f64>() / nf;
    let var: f64 = out.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / nf;
    let std = var.sqrt();

    assert!(
        (mean - target_mean).abs() < 0.05,
        "mean {mean} too far from {target_mean}"
    );
    assert!(
        (std - target_std).abs() < 0.05,
        "std {std} too far from {target_std}"
    );
}

// ---------------------------------------------------------------------------
// Streaming API — fill_uniform_chunked (f32)
// ---------------------------------------------------------------------------

/// One-shot fill and chunked fill with the same seed must produce the exact
/// same sequence of f32 values.
#[test]
fn fill_uniform_chunked_matches_full() {
    let total = 1024usize;
    let chunk_size = 128usize;

    // One-shot reference fill.
    let mut rng_full = RngEngine::new(RngEngineKind::Philox, 42).unwrap();
    let mut full = vec![0f32; total];
    rng_full.uniform_f32(&mut full).unwrap();

    // Chunked fill — collect all chunks into a single flat vector.
    let mut rng_chunked = RngEngine::new(RngEngineKind::Philox, 42).unwrap();
    let mut collected: Vec<f32> = Vec::with_capacity(total);
    rng_chunked
        .fill_uniform_chunked(total, chunk_size, &mut |chunk: &[f32]| {
            collected.extend_from_slice(chunk);
        })
        .unwrap();

    assert_eq!(
        full, collected,
        "chunked fill must produce the same sequence as one-shot fill"
    );
}

/// Same as above but for f64.
#[test]
fn fill_uniform_chunked_f64_matches_full() {
    let total = 512usize;
    let chunk_size = 64usize;

    let mut rng_full = RngEngine::new(RngEngineKind::Xorwow, 77).unwrap();
    let mut full = vec![0f64; total];
    rng_full.uniform_f64(&mut full).unwrap();

    let mut rng_chunked = RngEngine::new(RngEngineKind::Xorwow, 77).unwrap();
    let mut collected: Vec<f64> = Vec::with_capacity(total);
    rng_chunked
        .fill_uniform_chunked_f64(total, chunk_size, &mut |chunk: &[f64]| {
            collected.extend_from_slice(chunk);
        })
        .unwrap();

    assert_eq!(full, collected, "f64 chunked fill must match one-shot fill");
}

/// Very small chunk (size 3), non-divisible total (10): all values in [0, 1).
#[test]
fn fill_uniform_chunked_small_chunks() {
    let mut rng = RngEngine::new(RngEngineKind::Mrg32k3a, 7).unwrap();
    let mut all_values: Vec<f32> = Vec::new();
    let mut chunk_count = 0usize;

    rng.fill_uniform_chunked(10, 3, &mut |chunk: &[f32]| {
        chunk_count += 1;
        for &v in chunk {
            assert!(
                (0.0..1.0).contains(&v),
                "value {v} out of [0,1) in chunk {chunk_count}"
            );
        }
        all_values.extend_from_slice(chunk);
    })
    .unwrap();

    assert_eq!(
        all_values.len(),
        10,
        "should receive exactly 10 values total"
    );
    // Chunks: 3, 3, 3, 1 → 4 callbacks.
    assert_eq!(
        chunk_count, 4,
        "expected 4 chunks for total=10, chunk_size=3"
    );
}

// ---------------------------------------------------------------------------
// Streaming API — fill_normal_chunked
// ---------------------------------------------------------------------------

/// 10 000 chunked normal f32 samples should have mean ≈ 0, std ≈ 1.
#[test]
fn fill_normal_chunked_stats() {
    let total = 10_000usize;
    let chunk_size = 256usize;

    let mut rng = RngEngine::new(RngEngineKind::Philox, 42).unwrap();
    let mut all_values: Vec<f32> = Vec::with_capacity(total);

    rng.fill_normal_chunked(total, chunk_size, 0.0, 1.0, &mut |chunk: &[f32]| {
        all_values.extend_from_slice(chunk);
    })
    .unwrap();

    assert_eq!(all_values.len(), total);

    let nf = total as f32;
    let mean: f32 = all_values.iter().sum::<f32>() / nf;
    let var: f32 = all_values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / nf;
    let std = var.sqrt();

    assert!(
        mean.abs() < 0.05,
        "chunked normal mean {mean} too far from 0"
    );
    assert!(
        (std - 1.0).abs() < 0.05,
        "chunked normal std {std} too far from 1"
    );
}

// ---------------------------------------------------------------------------
// Error cases
// ---------------------------------------------------------------------------

/// `total == 0` must return `RngError::EmptyBuffer`.
#[test]
fn fill_uniform_chunked_empty_total_error() {
    use tensorlogic_oxicuda_rng::RngError;
    let mut rng = RngEngine::new(RngEngineKind::Philox, 0).unwrap();
    let result = rng.fill_uniform_chunked(0, 64, &mut |_: &[f32]| {});
    assert!(
        matches!(result, Err(RngError::EmptyBuffer)),
        "total=0 must return EmptyBuffer"
    );
}

/// `chunk_size == 0` must return `RngError::EmptyBuffer`.
#[test]
fn fill_uniform_chunked_empty_chunk_error() {
    use tensorlogic_oxicuda_rng::RngError;
    let mut rng = RngEngine::new(RngEngineKind::Philox, 0).unwrap();
    let result = rng.fill_uniform_chunked(1024, 0, &mut |_: &[f32]| {});
    assert!(
        matches!(result, Err(RngError::EmptyBuffer)),
        "chunk_size=0 must return EmptyBuffer"
    );
}

// ---------------------------------------------------------------------------
// Compile-time Send + Sync check (CPU path only)
// ---------------------------------------------------------------------------

/// Verifies at compile time that `RngEngine` satisfies both `Send` and `Sync`
/// on the CPU-only path.  If this file compiles, the trait bounds hold.
#[cfg(not(feature = "gpu"))]
fn _send_sync_cpu_builds() {
    fn _a<T: Send + Sync>() {}
    _a::<RngEngine>();
}
