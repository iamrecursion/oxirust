//! Integration-level unit tests for the Longformer sparse attention module.

use super::attention::SparseAttention;
use super::config::SparseAttentionConfig;
use super::error::SparseAttentionError;
use super::mask::build_mask;

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

#[test]
fn mask_sliding_window_pattern() {
    let cfg = SparseAttentionConfig::new(2, 1, 4).expect("config valid");
    let mask = build_mask(8, &cfg).expect("mask ok");

    for i in 0_usize..8 {
        for j in 0_usize..8 {
            let dist = i.abs_diff(j);
            assert_eq!(
                mask.is_attended(i, j),
                dist <= 2,
                "mask[{i}][{j}] should be {} (dist={dist}, window=2)",
                dist <= 2
            );
        }
    }
}

#[test]
fn mask_global_tokens_attend_everywhere_and_are_attended() {
    let cfg = SparseAttentionConfig::new(1, 1, 4)
        .map(|c| c.with_global_tokens(vec![0, 7]))
        .expect("config valid");
    let mask = build_mask(8, &cfg).expect("mask ok");

    // Global token 0 attends to all
    for j in 0..8 {
        assert!(mask.is_attended(0, j), "global 0 -> {j}");
        assert!(mask.is_attended(7, j), "global 7 -> {j}");
    }

    // All attend to globals
    for i in 0..8 {
        assert!(mask.is_attended(i, 0), "{i} -> global 0");
        assert!(mask.is_attended(i, 7), "{i} -> global 7");
    }
}

#[test]
fn mask_causal_zeros_future() {
    let cfg = SparseAttentionConfig::new(100, 1, 4)
        .map(|c| c.with_causal(true))
        .expect("config valid");
    let mask = build_mask(10, &cfg).expect("mask ok");

    for i in 0..10 {
        for j in 0..10 {
            if j > i {
                assert!(
                    !mask.is_attended(i, j),
                    "causal: {i} should not attend to {j}"
                );
            } else {
                assert!(mask.is_attended(i, j), "causal: {i} should attend to {j}");
            }
        }
    }
}

#[test]
fn forward_output_dimensions() {
    let cfg = SparseAttentionConfig::new(3, 2, 4).expect("config valid");
    let attn = SparseAttention::new(cfg).expect("attn ok");

    let seq_len = 10;
    let d_model = 8;
    let q = constant_qkv(seq_len, d_model, 0.1);
    let k = constant_qkv(seq_len, d_model, 0.1);
    let v = ranged_values(seq_len, d_model);

    let out = attn.forward(&q, &k, &v).expect("forward ok");
    assert_eq!(out.len(), seq_len);
    for row in &out {
        assert_eq!(row.len(), d_model);
    }
}

#[test]
fn full_window_matches_dense_attention() {
    let seq_len = 6;
    let num_heads = 1;
    let head_dim = 4;
    let d_model = num_heads * head_dim;

    let cfg_full =
        SparseAttentionConfig::new(seq_len, num_heads, head_dim).expect("full-window config valid");
    let attn_full = SparseAttention::new(cfg_full).expect("attn ok");

    let q = constant_qkv(seq_len, d_model, 1.0);
    let k = constant_qkv(seq_len, d_model, 1.0);
    let v = ranged_values(seq_len, d_model);

    let out_full = attn_full.forward(&q, &k, &v).expect("forward ok");

    // Dense attention with constant Q/K produces uniform softmax,
    // so output = mean(V) for every row.
    let mean_v: Vec<f64> = (0..d_model)
        .map(|d| {
            let col_sum: f64 = (0..seq_len).map(|i| v[i][d]).sum();
            col_sum / seq_len as f64
        })
        .collect();

    for (i, row) in out_full.iter().enumerate() {
        for (d, &val) in row.iter().enumerate() {
            assert!(
                (val - mean_v[d]).abs() < 1e-6,
                "out[{i}][{d}] = {val}, expected mean {}",
                mean_v[d]
            );
        }
    }
}

#[test]
fn global_only_single_token_at_zero() {
    let cfg = SparseAttentionConfig::new(1, 1, 2)
        .map(|c| c.with_global_tokens(vec![0]))
        .expect("config valid");
    let attn = SparseAttention::new(cfg).expect("attn ok");

    let q = constant_qkv(4, 2, 1.0);
    let k = constant_qkv(4, 2, 1.0);
    let v = vec![
        vec![1.0, 2.0],
        vec![3.0, 4.0],
        vec![5.0, 6.0],
        vec![7.0, 8.0],
    ];

    let out = attn.forward(&q, &k, &v).expect("forward ok");

    // Position 0 is global: with constant Q/K, uniform over all.
    let mean_d0 = (1.0 + 3.0 + 5.0 + 7.0) / 4.0;
    let mean_d1 = (2.0 + 4.0 + 6.0 + 8.0) / 4.0;
    assert!((out[0][0] - mean_d0).abs() < 1e-6);
    assert!((out[0][1] - mean_d1).abs() < 1e-6);
}

#[test]
fn error_window_size_zero() {
    let result = SparseAttentionConfig::new(0, 2, 4);
    assert!(matches!(
        result,
        Err(SparseAttentionError::InvalidWindowSize(0))
    ));
}

#[test]
fn error_global_index_out_of_bounds() {
    let cfg = SparseAttentionConfig::new(2, 1, 4)
        .map(|c| c.with_global_tokens(vec![0, 99]))
        .expect("config valid");
    let result = build_mask(16, &cfg);
    assert!(matches!(
        result,
        Err(SparseAttentionError::InvalidGlobalIndices {
            index: 99,
            seq_len: 16
        })
    ));
}

#[test]
fn causal_plus_global_interaction() {
    let cfg = SparseAttentionConfig::new(1, 1, 4)
        .map(|c| c.with_global_tokens(vec![0]).with_causal(true))
        .expect("config valid");
    let mask = build_mask(8, &cfg).expect("mask ok");

    // Position 0 (global + causal): attends only to 0 (j <= 0)
    assert!(mask.is_attended(0, 0));
    assert!(!mask.is_attended(0, 1));

    // Position 7: attends to window [6,7] + global 0 within causal
    assert!(mask.is_attended(7, 0)); // global
    assert!(mask.is_attended(7, 6)); // window
    assert!(mask.is_attended(7, 7)); // self
    assert!(!mask.is_attended(7, 3)); // outside window and not global
}

#[test]
fn multi_head_produces_correct_output_shape() {
    let cfg = SparseAttentionConfig::new(3, 4, 8).expect("config valid");
    let attn = SparseAttention::new(cfg).expect("attn ok");

    let d_model = 32; // 4 heads * 8 dim
    let q = constant_qkv(12, d_model, 0.5);
    let k = constant_qkv(12, d_model, 0.5);
    let v = ranged_values(12, d_model);

    let out = attn.forward(&q, &k, &v).expect("forward ok");
    assert_eq!(out.len(), 12);
    assert_eq!(out[0].len(), 32);
}

#[test]
fn attention_weights_zero_outside_window() {
    let cfg = SparseAttentionConfig::new(1, 1, 4).expect("config valid");
    let attn = SparseAttention::new(cfg.clone()).expect("attn ok");

    let q = constant_qkv(8, 4, 1.0);
    let k = constant_qkv(8, 4, 1.0);

    let mask = build_mask(8, &cfg).expect("mask ok");
    let weights = attn
        .attention_weights(&q, &k, &mask, 0)
        .expect("weights ok");

    // Position 4 (interior): window = [3,4,5].  Positions outside should
    // have near-zero weight (exp(-1e9) ~ 0).
    let row4 = &weights[4];
    for (j, &w) in row4.iter().enumerate() {
        let dist = 4_usize.abs_diff(j);
        if dist > 1 {
            assert!(
                w < 1e-30,
                "weights[4][{j}] = {w} should be ~0 (outside window)",
            );
        }
    }
}
