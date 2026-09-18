//! Integration tests for the LoRA module.

use tensorlogic_train::lora::{LoraAdapter, LoraConfig, LoraLayer, LoraResult};

fn constant_weight(d: usize, k: usize, val: f64) -> Vec<Vec<f64>> {
    vec![vec![val; k]; d]
}

// ---------------------------------------------------------------------------
// 1. End-to-end: adapter with 3 attention layers (Q/K/V)
// ---------------------------------------------------------------------------

#[test]
fn test_end_to_end_attention_adapter() -> LoraResult<()> {
    let d_model = 64;
    let d_k = 32;
    let rank = 4;
    let n_batch = 8;

    let config = LoraConfig {
        rank,
        alpha: rank as f64,
        dropout: 0.0,
        target_modules: vec!["q_proj".into(), "k_proj".into(), "v_proj".into()],
        seed: 123,
    };

    let mut adapter = LoraAdapter::new(config);

    adapter.add_layer("q_proj", constant_weight(d_model, d_k, 0.01))?;
    adapter.add_layer("k_proj", constant_weight(d_model, d_k, 0.02))?;
    adapter.add_layer("v_proj", constant_weight(d_model, d_k, 0.03))?;

    let summary = adapter.summary();
    assert_eq!(summary.layers.len(), 3);

    let expected_trainable_per_layer = rank * (d_model + d_k); // 4 * 96 = 384
    let expected_total_trainable = 3 * expected_trainable_per_layer; // 1152
    assert_eq!(adapter.total_trainable_params(), expected_total_trainable);

    for stats in &summary.layers {
        assert_eq!(stats.rank, rank);
        assert_eq!(stats.d, d_model);
        assert_eq!(stats.k, d_k);
        assert_eq!(stats.trainable_params, expected_trainable_per_layer);
    }

    let input = vec![vec![1.0; d_k]; n_batch];
    for name in &["q_proj", "k_proj", "v_proj"] {
        let out = adapter.forward(name, &input)?;
        assert_eq!(out.len(), n_batch, "batch size mismatch for {name}");
        assert_eq!(out[0].len(), d_model, "output dim mismatch for {name}");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// 2. Merge/unmerge cycle: effective_weight consistency
// ---------------------------------------------------------------------------

#[test]
fn test_merge_unmerge_effective_weight_consistency() -> LoraResult<()> {
    let d = 32;
    let k = 48;
    let rank = 8;

    let base = constant_weight(d, k, 0.05);
    let config = LoraConfig {
        rank,
        alpha: 16.0,
        dropout: 0.0,
        target_modules: Vec::new(),
        seed: 7,
    };

    let mut layer = LoraLayer::new(base.clone(), config)?;

    let eff_before = layer.effective_weight()?;

    layer.merge()?;
    assert!(layer.merged);

    for (i, row) in layer.base_weight.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            let diff = (val - eff_before[i][j]).abs();
            assert!(
                diff < 1e-10,
                "merge base_weight diverged at ({i},{j}): {diff}"
            );
        }
    }

    let eff_merged = layer.effective_weight()?;
    for (i, row) in eff_merged.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            let diff = (val - eff_before[i][j]).abs();
            assert!(
                diff < 1e-10,
                "effective_weight changed after merge at ({i},{j}): {diff}"
            );
        }
    }

    layer.unmerge()?;
    assert!(!layer.merged);

    for (i, row) in layer.base_weight.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            let diff = (val - base[i][j]).abs();
            assert!(
                diff < 1e-10,
                "unmerge did not restore original base_weight at ({i},{j}): {diff}"
            );
        }
    }

    let eff_after_unmerge = layer.effective_weight()?;
    for i in 0..d {
        for j in 0..k {
            let diff = (eff_after_unmerge[i][j] - eff_before[i][j]).abs();
            assert!(
                diff < 1e-10,
                "effective_weight changed after unmerge at ({i},{j}): {diff}"
            );
        }
    }
    Ok(())
}
