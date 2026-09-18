//! Unit tests for the LoRA module.

use super::adapter::LoraAdapter;
use super::config::LoraConfig;
use super::error::{LoraError, LoraResult};
use super::layer::LoraLayer;

fn identity_weight(n: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| {
            let mut row = vec![0.0; n];
            row[i] = 1.0;
            row
        })
        .collect()
}

fn constant_weight(d: usize, k: usize, val: f64) -> Vec<Vec<f64>> {
    vec![vec![val; k]; d]
}

fn cfg(rank: usize) -> LoraConfig {
    LoraConfig {
        rank,
        alpha: rank as f64,
        dropout: 0.0,
        target_modules: Vec::new(),
        seed: 42,
    }
}

// -----------------------------------------------------------------------
// 1. Valid creation
// -----------------------------------------------------------------------

#[test]
fn test_layer_creation_valid() -> LoraResult<()> {
    let w = identity_weight(8);
    let layer = LoraLayer::new(w, cfg(4))?;
    assert!(!layer.merged);
    assert_eq!(layer.weight_a.len(), 4);
    assert_eq!(layer.weight_a[0].len(), 8);
    assert_eq!(layer.weight_b.len(), 8);
    assert_eq!(layer.weight_b[0].len(), 4);
    Ok(())
}

// -----------------------------------------------------------------------
// 2. Forward pass dimensions
// -----------------------------------------------------------------------

#[test]
fn test_forward_output_dimensions() -> LoraResult<()> {
    let d = 16;
    let k = 32;
    let n = 5;
    let w = constant_weight(d, k, 0.01);
    let mut layer = LoraLayer::new(w, cfg(4))?;
    let input: Vec<Vec<f64>> = vec![vec![1.0; k]; n];
    let out = layer.forward(&input)?;
    assert_eq!(out.len(), n);
    assert_eq!(out[0].len(), d);
    Ok(())
}

// -----------------------------------------------------------------------
// 3. Merge/unmerge round-trip preserves base weight
// -----------------------------------------------------------------------

#[test]
fn test_merge_unmerge_roundtrip() -> LoraResult<()> {
    let d = 8;
    let k = 8;
    let w = identity_weight(d);
    let original: Vec<Vec<f64>> = w.clone();
    let mut layer = LoraLayer::new(w, cfg(2))?;

    layer.merge()?;
    assert!(layer.merged);
    layer.unmerge()?;
    assert!(!layer.merged);

    for (i, (row_got, row_orig)) in layer
        .base_weight
        .iter()
        .zip(original.iter())
        .enumerate()
        .take(d)
    {
        for (j, (got, orig)) in row_got.iter().zip(row_orig.iter()).enumerate().take(k) {
            let diff = (got - orig).abs();
            assert!(diff < 1e-10, "roundtrip drift at ({i},{j}): {diff}");
        }
    }
    Ok(())
}

// -----------------------------------------------------------------------
// 4. Compression ratio
// -----------------------------------------------------------------------

#[test]
fn test_compression_ratio() -> LoraResult<()> {
    let d = 64;
    let k = 128;
    let r = 4;
    let w = constant_weight(d, k, 0.0);
    let layer = LoraLayer::new(w, cfg(r))?;

    let trainable = r * (d + k); // 4 * 192 = 768
    let total = d * k + trainable; // 8192 + 768 = 8960
    assert_eq!(layer.trainable_params(), trainable);
    assert_eq!(layer.total_params(), total);
    let expected_ratio = trainable as f64 / total as f64;
    assert!((layer.compression_ratio() - expected_ratio).abs() < 1e-12);
    Ok(())
}

// -----------------------------------------------------------------------
// 5. Error on rank == 0
// -----------------------------------------------------------------------

#[test]
fn test_error_rank_zero() {
    let w = identity_weight(8);
    let err = LoraLayer::new(w, cfg(0));
    assert!(err.is_err());
    match err.err() {
        Some(LoraError::InvalidRank(0)) => {}
        other => panic!("expected InvalidRank(0), got {other:?}"),
    }
}

// -----------------------------------------------------------------------
// 6. Error on rank > min(d, k)
// -----------------------------------------------------------------------

#[test]
fn test_error_rank_too_large() {
    let w = constant_weight(4, 6, 1.0);
    let err = LoraLayer::new(w, cfg(5));
    assert!(err.is_err());
    match err.err() {
        Some(LoraError::InvalidRank(5)) => {}
        other => panic!("expected InvalidRank(5), got {other:?}"),
    }
}

// -----------------------------------------------------------------------
// 7. Error on dimension mismatch in forward
// -----------------------------------------------------------------------

#[test]
fn test_error_forward_dim_mismatch() -> LoraResult<()> {
    let d = 8;
    let k = 16;
    let w = constant_weight(d, k, 0.0);
    let mut layer = LoraLayer::new(w, cfg(2))?;
    let bad_input = vec![vec![1.0; 10]; 3];
    let err = layer.forward(&bad_input);
    assert!(err.is_err());
    assert!(matches!(
        err.err(),
        Some(LoraError::DimensionMismatch { .. })
    ));
    Ok(())
}

// -----------------------------------------------------------------------
// 8. Adapter multi-layer forward
// -----------------------------------------------------------------------

#[test]
fn test_adapter_multi_layer_forward() -> LoraResult<()> {
    let config = LoraConfig {
        rank: 2,
        alpha: 2.0,
        dropout: 0.0,
        target_modules: vec!["q".into(), "k".into(), "v".into()],
        seed: 99,
    };
    let mut adapter = LoraAdapter::new(config);
    let d = 8;
    let k = 16;

    adapter.add_layer("q", constant_weight(d, k, 0.1))?;
    adapter.add_layer("k", constant_weight(d, k, 0.2))?;
    adapter.add_layer("v", constant_weight(d, k, 0.3))?;

    let input = vec![vec![1.0; k]; 4];
    for name in &["q", "k", "v"] {
        let out = adapter.forward(name, &input)?;
        assert_eq!(out.len(), 4);
        assert_eq!(out[0].len(), d);
    }
    Ok(())
}

// -----------------------------------------------------------------------
// 9. Trainable params count matches r*(d+k)
// -----------------------------------------------------------------------

#[test]
fn test_trainable_params_count() -> LoraResult<()> {
    let d = 32;
    let k = 64;
    let r = 8;
    let w = constant_weight(d, k, 0.0);
    let layer = LoraLayer::new(w, cfg(r))?;
    assert_eq!(layer.trainable_params(), r * (d + k));
    Ok(())
}

// -----------------------------------------------------------------------
// 10. Effective weight matches base_weight after merge
// -----------------------------------------------------------------------

#[test]
fn test_effective_weight_after_merge() -> LoraResult<()> {
    let d = 8;
    let k = 8;
    let w = identity_weight(d);
    let mut layer = LoraLayer::new(w, cfg(2))?;

    let eff_before = layer.effective_weight()?;
    layer.merge()?;
    let eff_after = layer.effective_weight()?;

    for (i, (row_before, row_after)) in eff_before.iter().zip(eff_after.iter()).enumerate().take(d)
    {
        for (j, (before, after)) in row_before.iter().zip(row_after.iter()).enumerate().take(k) {
            let diff = (before - after).abs();
            assert!(diff < 1e-10, "effective_weight drift at ({i},{j}): {diff}");
        }
    }
    Ok(())
}

// -----------------------------------------------------------------------
// 11. Double merge returns error
// -----------------------------------------------------------------------

#[test]
fn test_double_merge_error() -> LoraResult<()> {
    let w = identity_weight(8);
    let mut layer = LoraLayer::new(w, cfg(2))?;
    layer.merge()?;
    let err = layer.merge();
    assert!(err.is_err());
    assert!(matches!(err.err(), Some(LoraError::MergeError(_))));
    Ok(())
}

// -----------------------------------------------------------------------
// 12. Adapter summary totals
// -----------------------------------------------------------------------

#[test]
fn test_adapter_summary() -> LoraResult<()> {
    let config = LoraConfig {
        rank: 4,
        alpha: 4.0,
        dropout: 0.0,
        target_modules: Vec::new(),
        seed: 7,
    };
    let mut adapter = LoraAdapter::new(config);
    adapter.add_layer("a", constant_weight(16, 32, 0.0))?;
    adapter.add_layer("b", constant_weight(8, 64, 0.0))?;

    let summary = adapter.summary();
    assert_eq!(summary.layers.len(), 2);
    // layer a: 4 * (16 + 32) = 192,  layer b: 4 * (8 + 64) = 288
    assert_eq!(summary.total_trainable, 192 + 288);
    // layer a total: 16*32 + 192 = 704,  layer b total: 8*64 + 288 = 800
    assert_eq!(summary.total_params, 704 + 800);
    Ok(())
}
