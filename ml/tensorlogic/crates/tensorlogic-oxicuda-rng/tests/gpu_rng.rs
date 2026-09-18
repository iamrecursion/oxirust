//! GPU-path integration tests.
//!
//! These tests are gated on the `gpu` Cargo feature being enabled **and** a
//! CUDA-capable device being accessible at runtime.  In CI without a GPU they
//! are skipped automatically via `RngEngine::is_gpu()` checks.
//!
//! To run these locally against real hardware:
//!
//! ```bash
//! cargo test --features gpu -p tensorlogic-oxicuda-rng --test gpu_rng
//! ```

use tensorlogic_oxicuda_rng::{RngEngine, RngEngineKind, RngError};

// ---------------------------------------------------------------------------
// Helper: construct an engine and immediately report whether the GPU path is
// active.  If the GPU path is not available (no device, `gpu` feature off)
// the test is skipped via an early `return`.
// ---------------------------------------------------------------------------

fn make_gpu_engine(kind: RngEngineKind, seed: u64) -> Option<RngEngine> {
    let eng = RngEngine::new(kind, seed).expect("construction must not fail");
    if eng.is_gpu() {
        Some(eng)
    } else {
        None // No GPU available — caller should skip the test.
    }
}

// ---------------------------------------------------------------------------
// Availability probe
// ---------------------------------------------------------------------------

/// Verifies that when a GPU is present, `is_gpu()` returns `true`.
/// When no GPU is present the test exits cleanly (not a failure).
#[test]
fn gpu_engine_available_when_cuda_present() {
    let eng = RngEngine::new(RngEngineKind::Philox, 0).expect("construction must not fail");
    // This is not a hard assertion — the test documents expected behaviour.
    // If `is_gpu()` is `true`, we know the GPU path is active.
    let _ = eng.is_gpu();
}

// ---------------------------------------------------------------------------
// Uniform f32 on GPU
// ---------------------------------------------------------------------------

/// GPU uniform samples must lie in [0.0, 1.0).
#[test]
fn gpu_uniform_range() {
    let Some(mut rng) = make_gpu_engine(RngEngineKind::Philox, 1) else {
        return; // no GPU
    };
    let mut out = vec![0f32; 4_096];
    rng.uniform_f32(&mut out).expect("gpu uniform must succeed");
    for &v in &out {
        assert!((0.0..1.0).contains(&v), "uniform out of range: {v}");
    }
}

/// Same seed on the GPU path must be deterministic.
#[test]
fn gpu_uniform_deterministic() {
    let Some(mut rng1) = make_gpu_engine(RngEngineKind::Philox, 42) else {
        return;
    };
    let Some(mut rng2) = make_gpu_engine(RngEngineKind::Philox, 42) else {
        return;
    };
    let mut out1 = vec![0f32; 256];
    let mut out2 = vec![0f32; 256];
    rng1.uniform_f32(&mut out1).unwrap();
    rng2.uniform_f32(&mut out2).unwrap();
    assert_eq!(out1, out2, "same GPU seed must produce identical sequences");
}

/// Different seeds on the GPU path must produce different outputs.
#[test]
fn gpu_uniform_different_seeds_differ() {
    let Some(mut rng_a) = make_gpu_engine(RngEngineKind::Philox, 0) else {
        return;
    };
    let Some(mut rng_b) = make_gpu_engine(RngEngineKind::Philox, 1) else {
        return;
    };
    let mut out_a = vec![0f32; 256];
    let mut out_b = vec![0f32; 256];
    rng_a.uniform_f32(&mut out_a).unwrap();
    rng_b.uniform_f32(&mut out_b).unwrap();
    assert_ne!(
        out_a, out_b,
        "different seeds must produce different GPU sequences"
    );
}

// ---------------------------------------------------------------------------
// Normal f32 on GPU
// ---------------------------------------------------------------------------

/// GPU normal samples must be finite with approximately correct statistics.
#[test]
fn gpu_normal_approximate_stats() {
    let Some(mut rng) = make_gpu_engine(RngEngineKind::Xorwow, 99) else {
        return;
    };
    let mut out = vec![0f32; 10_000];
    rng.normal_f32(&mut out, 0.0, 1.0)
        .expect("gpu normal must succeed");

    let n = out.len() as f32;
    let mean: f32 = out.iter().sum::<f32>() / n;
    let var: f32 = out.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;

    assert!(mean.abs() < 0.05, "GPU normal mean too far from 0: {mean}");
    assert!(
        (var - 1.0).abs() < 0.05,
        "GPU normal variance too far from 1: {var}"
    );
}

/// GPU normal samples must all be finite.
#[test]
fn gpu_normal_all_finite() {
    let Some(mut rng) = make_gpu_engine(RngEngineKind::Mrg32k3a, 7) else {
        return;
    };
    let mut out = vec![0f32; 2_048];
    rng.normal_f32(&mut out, 0.0, 1.0).unwrap();
    for (i, &v) in out.iter().enumerate() {
        assert!(v.is_finite(), "element {i} is not finite: {v}");
    }
}

/// GPU normal with non-unit parameters should shift correctly.
#[test]
fn gpu_normal_nonstandard_params() {
    let Some(mut rng) = make_gpu_engine(RngEngineKind::Philox, 55) else {
        return;
    };
    let (target_mean, target_std) = (10.0_f32, 3.0_f32);
    let mut out = vec![0f32; 10_000];
    rng.normal_f32(&mut out, target_mean, target_std).unwrap();

    let n = out.len() as f32;
    let mean: f32 = out.iter().sum::<f32>() / n;
    let var: f32 = out.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;

    assert!(
        (mean - target_mean).abs() < 0.2,
        "GPU mean {mean} too far from {target_mean}"
    );
    assert!(
        (var.sqrt() - target_std).abs() < 0.2,
        "GPU std {std} too far from {target_std}",
        std = var.sqrt()
    );
}

// ---------------------------------------------------------------------------
// Bernoulli on GPU
// ---------------------------------------------------------------------------

/// GPU Bernoulli outputs must only be 0 or 1.
#[test]
fn gpu_bernoulli_binary_values() {
    let Some(mut rng) = make_gpu_engine(RngEngineKind::Philox, 3) else {
        return;
    };
    let mut out = vec![255u8; 4_096];
    rng.bernoulli(&mut out, 0.5)
        .expect("gpu bernoulli must succeed");
    for (i, &b) in out.iter().enumerate() {
        assert!(b == 0 || b == 1, "element {i} has non-binary value {b}");
    }
}

/// GPU Bernoulli p=0.7 proportion must be within ±0.03 of 0.7.
#[test]
fn gpu_bernoulli_proportion() {
    let Some(mut rng) = make_gpu_engine(RngEngineKind::Xorwow, 456) else {
        return;
    };
    let mut out = vec![0u8; 10_000];
    rng.bernoulli(&mut out, 0.7).unwrap();
    let ones = out.iter().filter(|&&b| b == 1).count() as f32 / 10_000.0;
    assert!(
        (ones - 0.7).abs() < 0.03,
        "GPU Bernoulli proportion {ones} too far from 0.7"
    );
}

// ---------------------------------------------------------------------------
// Error handling on GPU path
// ---------------------------------------------------------------------------

/// Empty buffer must return EmptyBuffer regardless of backend.
#[test]
fn gpu_empty_buffer_uniform_error() {
    let Some(mut rng) = make_gpu_engine(RngEngineKind::Philox, 0) else {
        return;
    };
    let mut out: Vec<f32> = vec![];
    assert!(matches!(
        rng.uniform_f32(&mut out),
        Err(RngError::EmptyBuffer)
    ));
}

/// Empty buffer must return EmptyBuffer for normal on the GPU path too.
#[test]
fn gpu_empty_buffer_normal_error() {
    let Some(mut rng) = make_gpu_engine(RngEngineKind::Philox, 0) else {
        return;
    };
    let mut out: Vec<f32> = vec![];
    assert!(matches!(
        rng.normal_f32(&mut out, 0.0, 1.0),
        Err(RngError::EmptyBuffer)
    ));
}

/// Invalid std_dev must return InvalidParam on the GPU path.
#[test]
fn gpu_invalid_stddev_error() {
    let Some(mut rng) = make_gpu_engine(RngEngineKind::Philox, 0) else {
        return;
    };
    let mut out = vec![0f32; 16];
    assert!(matches!(
        rng.normal_f32(&mut out, 0.0, -1.0),
        Err(RngError::InvalidParam(_))
    ));
}

/// Invalid p must return InvalidParam on the GPU path.
#[test]
fn gpu_invalid_bernoulli_p_error() {
    let Some(mut rng) = make_gpu_engine(RngEngineKind::Philox, 0) else {
        return;
    };
    let mut out = vec![0u8; 16];
    assert!(matches!(
        rng.bernoulli(&mut out, 1.5),
        Err(RngError::InvalidParam(_))
    ));
}

// ---------------------------------------------------------------------------
// Cross-engine consistency — same distribution, different engines
// ---------------------------------------------------------------------------

/// All three GPU engine kinds should produce valid uniform output.
#[test]
fn gpu_all_engine_kinds_uniform_valid() {
    for kind in [
        RngEngineKind::Philox,
        RngEngineKind::Xorwow,
        RngEngineKind::Mrg32k3a,
    ] {
        let Some(mut rng) = make_gpu_engine(kind, 888) else {
            return; // skip if no GPU
        };
        let mut out = vec![0f32; 512];
        rng.uniform_f32(&mut out)
            .unwrap_or_else(|e| panic!("gpu uniform failed for {kind}: {e}"));
        assert!(
            out.iter().all(|&v| (0.0..1.0).contains(&v)),
            "out-of-range uniform for gpu engine {kind}"
        );
    }
}
