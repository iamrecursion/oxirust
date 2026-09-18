//! Integration tests for the Longformer-style sparse attention module.

use tensorlogic_trustformers::sparse_attention::{
    build_mask, SparseAttention, SparseAttentionConfig, SparseAttentionError,
};

fn constant_qkv(seq_len: usize, d_model: usize, val: f64) -> Vec<Vec<f64>> {
    vec![vec![val; d_model]; seq_len]
}

fn ranged_values(seq_len: usize, d_model: usize) -> Vec<Vec<f64>> {
    (0..seq_len)
        .map(|i| {
            (0..d_model)
                .map(|d| (i * d_model + d) as f64 * 0.01)
                .collect()
        })
        .collect()
}

/// Test 1: Sliding window on a 32-token sequence with window_size=4, 2 heads,
/// head_dim=8.  Verify output shape and that attention weights are zero outside
/// the window.
#[test]
fn sliding_window_32_tokens() {
    let seq_len = 32;
    let num_heads = 2;
    let head_dim = 8;
    let d_model = num_heads * head_dim;
    let window_size = 4;

    let cfg = SparseAttentionConfig::new(window_size, num_heads, head_dim)
        .expect("config should be valid");
    let attn = SparseAttention::new(cfg.clone()).expect("attn should construct");

    let q = constant_qkv(seq_len, d_model, 0.5);
    let k = constant_qkv(seq_len, d_model, 0.5);
    let v = ranged_values(seq_len, d_model);

    let out = attn.forward(&q, &k, &v).expect("forward should succeed");

    // Shape check
    assert_eq!(out.len(), seq_len);
    for row in &out {
        assert_eq!(row.len(), d_model);
    }

    // Verify attention weights are zero outside the window
    let mask = build_mask(seq_len, &cfg).expect("mask should build");
    let weights_h0 = attn
        .attention_weights(&q, &k, &mask, 0)
        .expect("weights should compute");

    for (i, row) in weights_h0.iter().enumerate() {
        for (j, &w) in row.iter().enumerate() {
            let dist = i.abs_diff(j);
            if dist > window_size {
                assert!(
                    w < 1e-30,
                    "weights[{i}][{j}] = {w} should be ~0 (dist={dist} > window={window_size})",
                );
            }
        }
    }

    // Verify weight rows sum to 1
    for (i, row) in weights_h0.iter().enumerate() {
        let sum: f64 = row.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "weight row {i} sums to {sum}, expected 1.0"
        );
    }
}

/// Test 2: Causal + global on a 16-token sequence with global at [0, 15],
/// causal=true.
///
/// - Position 0: attends only to position 0 (causal + global => j <= 0)
/// - Position 15: global + causal => attends to all positions 0..=15
/// - Middle positions: attend to window neighbours + globals, within causal
#[test]
fn causal_plus_global_16_tokens() {
    let seq_len = 16;
    let num_heads = 2;
    let head_dim = 4;
    let d_model = num_heads * head_dim;
    let window_size = 2;

    let cfg = SparseAttentionConfig::new(window_size, num_heads, head_dim)
        .map(|c| c.with_global_tokens(vec![0, 15]).with_causal(true))
        .expect("config should be valid");

    let mask = build_mask(seq_len, &cfg).expect("mask should build");

    // Position 0 (global, causal): attends only to j=0
    assert!(mask.is_attended(0, 0));
    for j in 1..seq_len {
        assert!(
            !mask.is_attended(0, j),
            "causal: position 0 should not attend to future {j}"
        );
    }

    // Position 15 (global, causal): attends to ALL positions 0..=15
    for j in 0..seq_len {
        assert!(
            mask.is_attended(15, j),
            "global 15 should attend to {j} (all j <= 15)"
        );
    }

    // Middle position (e.g. 8): attends to window [6,7,8,9,10] intersected
    // with causal (j <= 8) => [6,7,8], plus globals [0] (j<=8), NOT [15] (j>8)
    assert!(mask.is_attended(8, 0)); // global 0, within causal
    assert!(mask.is_attended(8, 6)); // window
    assert!(mask.is_attended(8, 7)); // window
    assert!(mask.is_attended(8, 8)); // self
    assert!(!mask.is_attended(8, 9)); // window but future (causal blocks)
    assert!(!mask.is_attended(8, 10)); // window but future
    assert!(!mask.is_attended(8, 15)); // global 15 but future

    // Run a forward pass to ensure no numerical issues
    let attn = SparseAttention::new(cfg).expect("attn should construct");
    let q = constant_qkv(seq_len, d_model, 0.3);
    let k = constant_qkv(seq_len, d_model, 0.3);
    let v = ranged_values(seq_len, d_model);

    let out = attn.forward(&q, &k, &v).expect("forward should succeed");
    assert_eq!(out.len(), seq_len);
    for row in &out {
        assert_eq!(row.len(), d_model);
        for &val in row {
            assert!(val.is_finite(), "output should be finite");
        }
    }
}

/// Test 3: Error on invalid window_size = 0.
#[test]
fn error_window_size_zero() {
    let result = SparseAttentionConfig::new(0, 2, 4);
    assert!(matches!(
        result,
        Err(SparseAttentionError::InvalidWindowSize(0))
    ));
}

/// Test 4: Error on global index out of bounds.
#[test]
fn error_global_index_oob() {
    let cfg = SparseAttentionConfig::new(2, 1, 4)
        .map(|c| c.with_global_tokens(vec![0, 50]))
        .expect("config construction ok");
    let result = build_mask(16, &cfg);
    assert!(matches!(
        result,
        Err(SparseAttentionError::InvalidGlobalIndices {
            index: 50,
            seq_len: 16
        })
    ));
}
