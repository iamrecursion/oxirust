//! Unit tests for the research-preview numerical MoE layer.
//!
//! These cover the full contract surface:
//!
//! 1. Top-k gating returns indices in range + softmax weights sum to 1
//!    for both `k = 1` and `k = 2`.
//! 2. `LinearExpert::forward` matches a hand-computed `W x + b`.
//! 3. `MoELayer` with identity experts + uniform gate produces the
//!    input back (round-trip test).
//! 4. `MoELayer` with two distinct experts and top-1 gating routes each
//!    input to exactly one expert; output equals that expert's output.
//! 5. `importance_loss` / `load_loss` are zero for uniform usage and
//!    strictly positive for imbalanced usage.
//! 6. Capacity-factor dropping: overflow tokens get zero contribution
//!    from the capped expert while non-overflow tokens match the
//!    uncapped baseline.

use super::error::MoeError;
use super::expert::{Expert, LinearExpert};
use super::gate::TopKGate;
use super::layer::MoELayer;
use super::load_balance::{combined_aux_loss, importance_loss, load_loss, BatchGatingStats};

use ndarray::{array, Array1, Array2};

fn identity_expert(dim: usize) -> LinearExpert {
    LinearExpert::from_arrays(Array2::<f64>::eye(dim), Array1::<f64>::zeros(dim))
        .expect("identity construct")
}

#[test]
fn topk_gate_k1_returns_valid_weights() {
    let gate = TopKGate::xavier_init(4, 3, 1, 42).expect("gate init");
    let x = array![0.1_f64, -0.2, 0.3, 0.4];
    let decision = gate.forward(&x.view()).expect("forward");
    assert_eq!(decision.k(), 1);
    assert!(decision.top_k_indices[0] < 3);
    let sum: f64 = decision.top_k_softmax_weights.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-12,
        "top-1 weight must be exactly 1.0"
    );
    assert_eq!(decision.raw_logits.len(), 3);
}

#[test]
fn topk_gate_k2_weights_sum_to_one() {
    let gate = TopKGate::xavier_init(6, 4, 2, 13).expect("gate init");
    let x = Array1::<f64>::from_vec(vec![0.5, -1.0, 0.25, 0.7, 0.0, -0.3]);
    let decision = gate.forward(&x.view()).expect("forward");
    assert_eq!(decision.k(), 2);
    let mut seen = [false; 4];
    for &idx in decision.top_k_indices.iter() {
        assert!(idx < 4, "index {idx} out of range");
        assert!(!seen[idx], "expert {idx} selected twice");
        seen[idx] = true;
    }
    let sum: f64 = decision.top_k_softmax_weights.iter().sum();
    assert!((sum - 1.0).abs() < 1e-12, "top-2 softmax must sum to 1.0");
    for &w in decision.top_k_softmax_weights.iter() {
        assert!(w > 0.0 && w < 1.0);
    }
}

#[test]
fn linear_expert_matches_hand_computation() {
    // W = [[1, 2, 3], [4, 5, 6]], b = [0.5, -0.5]. x = [1, 0, -1].
    // y = [1*1 + 2*0 + 3*(-1) + 0.5, 4*1 + 5*0 + 6*(-1) - 0.5]
    //   = [-2 + 0.5, -2 - 0.5] = [-1.5, -2.5].
    let weights = ndarray::array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let bias = ndarray::array![0.5, -0.5];
    let expert = LinearExpert::from_arrays(weights, bias).expect("construct");
    let x = ndarray::array![1.0, 0.0, -1.0];
    let y = expert.forward(&x.view()).expect("forward");
    assert!((y[0] - (-1.5)).abs() < 1e-12);
    assert!((y[1] - (-2.5)).abs() < 1e-12);
}

#[test]
fn moe_with_identity_experts_returns_input() {
    // Two identity experts, k=2 so every input gets weighted pass-through.
    let gate = TopKGate::xavier_init(3, 2, 2, 99).expect("gate");
    let experts: Vec<Box<dyn Expert>> =
        vec![Box::new(identity_expert(3)), Box::new(identity_expert(3))];
    let layer = MoELayer::new(gate, experts).expect("layer");
    let x = array![1.5_f64, -2.0, 0.25];
    let (y, decision) = layer.forward(&x.view()).expect("forward");
    assert_eq!(decision.k(), 2);
    for (a, b) in y.iter().zip(x.iter()) {
        assert!((a - b).abs() < 1e-12, "identity MoE must reproduce input");
    }
}

#[test]
fn moe_top1_routes_each_input_to_exactly_one_expert() {
    // Two distinct experts: expert 0 emits all-ones; expert 1 emits all-tens.
    let e0 =
        LinearExpert::from_arrays(Array2::<f64>::zeros((2, 2)), array![1.0_f64, 1.0]).expect("e0");
    let e1 = LinearExpert::from_arrays(Array2::<f64>::zeros((2, 2)), array![10.0_f64, 10.0])
        .expect("e1");

    // Handcrafted gate weights so that x = [1, 0] selects expert 0
    // and x = [0, 1] selects expert 1, unambiguously.
    let gate_weights = ndarray::array![[5.0_f64, -5.0], [-5.0, 5.0]];
    let gate = TopKGate::from_weights(gate_weights, 1).expect("gate");
    let experts: Vec<Box<dyn Expert>> = vec![Box::new(e0), Box::new(e1)];
    let layer = MoELayer::new(gate, experts).expect("layer");

    let x0 = array![1.0_f64, 0.0];
    let (y0, d0) = layer.forward(&x0.view()).expect("x0");
    assert_eq!(d0.top_k_indices[0], 0);
    assert!((y0[0] - 1.0).abs() < 1e-12 && (y0[1] - 1.0).abs() < 1e-12);

    let x1 = array![0.0_f64, 1.0];
    let (y1, d1) = layer.forward(&x1.view()).expect("x1");
    assert_eq!(d1.top_k_indices[0], 1);
    assert!((y1[0] - 10.0).abs() < 1e-12 && (y1[1] - 10.0).abs() < 1e-12);
}

#[test]
fn importance_loss_zero_when_balanced_and_positive_when_skewed() {
    let balanced = BatchGatingStats {
        gate_scores_per_token: ndarray::array![
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25]
        ],
        routed_expert_per_token: vec![0, 1, 2, 3],
    };
    let l_imp_bal = importance_loss(&balanced).expect("imp");
    assert!(l_imp_bal.abs() < 1e-12, "expected 0, got {l_imp_bal}");
    let l_load_bal = load_loss(&balanced).expect("load");
    assert!(l_load_bal.abs() < 1e-12, "expected 0, got {l_load_bal}");

    let skewed = BatchGatingStats {
        gate_scores_per_token: ndarray::array![
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0]
        ],
        routed_expert_per_token: vec![0, 0, 0, 0],
    };
    let l_imp_skew = importance_loss(&skewed).expect("imp");
    let l_load_skew = load_loss(&skewed).expect("load");
    assert!(l_imp_skew > 0.0, "importance loss must be > 0 for skew");
    assert!(l_load_skew > 0.0, "load loss must be > 0 for skew");

    let combined = combined_aux_loss(&skewed, 0.01).expect("combined");
    assert!(combined > 0.0);
    assert!((combined - 0.01 * (l_imp_skew + l_load_skew)).abs() < 1e-12);
}

#[test]
fn capacity_factor_drops_overflow_tokens() {
    // 4 tokens, 2 experts, capacity_factor = 0.5 ⇒
    // C = ceil(0.5 * 4 / 2) = 1. Each expert takes ≤ 1 token under top-1.
    //
    // With all tokens deterministically routed to expert 0, three
    // overflow. Their expert-0 contribution is zeroed out.
    let e0 =
        LinearExpert::from_arrays(Array2::<f64>::zeros((2, 2)), array![1.0_f64, 1.0]).expect("e0");
    let e1 =
        LinearExpert::from_arrays(Array2::<f64>::zeros((2, 2)), array![5.0_f64, 5.0]).expect("e1");

    // Gate strongly prefers expert 0 for every input (first row large,
    // second row zero ⇒ logit_0 > logit_1 for any non-negative input).
    let gate_weights = ndarray::array![[10.0_f64, 10.0], [0.0, 0.0]];
    let gate = TopKGate::from_weights(gate_weights, 1).expect("gate");

    let experts_uncapped: Vec<Box<dyn Expert>> = vec![Box::new(e0.clone()), Box::new(e1.clone())];
    let uncapped = MoELayer::new(gate.clone(), experts_uncapped).expect("uncapped");

    let experts_capped: Vec<Box<dyn Expert>> = vec![Box::new(e0), Box::new(e1)];
    let capped = MoELayer::new(gate, experts_capped)
        .expect("capped")
        .with_capacity_factor(0.5)
        .expect("cf");

    let batch = ndarray::array![[1.0_f64, 2.0], [0.5, 1.5], [2.0, 0.25], [0.75, 1.0]];

    let (out_uncapped, _) = uncapped
        .forward_batch(&batch.view())
        .expect("uncapped forward");
    let (out_capped, stats) = capped.forward_batch(&batch.view()).expect("capped forward");

    // All four tokens intended to route to expert 0.
    assert_eq!(stats.routed_expert_per_token, vec![0, 0, 0, 0]);
    // Uncapped: every row equals expert 0's constant output [1, 1].
    for t in 0..4 {
        assert!((out_uncapped[(t, 0)] - 1.0).abs() < 1e-12);
        assert!((out_uncapped[(t, 1)] - 1.0).abs() < 1e-12);
    }
    // Capped: only the first token gets expert 0's output; overflow
    // rows are all-zero (no fallback under top-1 with capacity = 1).
    assert!((out_capped[(0, 0)] - 1.0).abs() < 1e-12);
    assert!((out_capped[(0, 1)] - 1.0).abs() < 1e-12);
    for t in 1..4 {
        assert!(
            out_capped[(t, 0)].abs() < 1e-12,
            "overflow token {t} col 0 = {}",
            out_capped[(t, 0)]
        );
        assert!(
            out_capped[(t, 1)].abs() < 1e-12,
            "overflow token {t} col 1 = {}",
            out_capped[(t, 1)]
        );
    }
}

#[test]
fn invalid_capacity_factor_is_rejected() {
    let gate = TopKGate::xavier_init(2, 2, 1, 0).expect("gate");
    let experts: Vec<Box<dyn Expert>> =
        vec![Box::new(identity_expert(2)), Box::new(identity_expert(2))];
    let layer = MoELayer::new(gate, experts).expect("layer");
    let err = layer.with_capacity_factor(-1.0).expect_err("must fail");
    assert!(matches!(err, MoeError::InvalidCapacityFactor { .. }));
}

#[test]
fn invalid_topk_is_rejected() {
    // k > num_experts
    let err = TopKGate::xavier_init(4, 2, 3, 0).expect_err("must fail");
    assert!(matches!(
        err,
        MoeError::InvalidTopK {
            k: 3,
            num_experts: 2
        }
    ));
    // k == 0
    let err0 = TopKGate::xavier_init(4, 2, 0, 0).expect_err("must fail");
    assert!(matches!(err0, MoeError::InvalidTopK { k: 0, .. }));
}
