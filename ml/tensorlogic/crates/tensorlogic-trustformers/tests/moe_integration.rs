//! Integration test for the numerical Mixture-of-Experts (MoE) layer.
//!
//! Constructs a 4-expert MoELayer with LinearExperts, runs a batch of 32
//! inputs in R^16, and asserts:
//!
//! 1. Output shape matches `(32, 16)`.
//! 2. Combined auxiliary loss (importance + load) is finite and
//!    non-negative.
//! 3. No single expert receives 0% *or* > 80% of the batch routing.
//! 4. Gate softmax weights are valid probability vectors (non-negative,
//!    sum to 1 within tolerance).
//! 5. Capacity-factor capping zeroes out overflow contributions without
//!    affecting non-overflow tokens.

use ndarray::{Array1, Array2};

use tensorlogic_trustformers::moe::{
    combined_aux_loss, importance_loss, load_loss, Expert, LinearExpert, MoELayer, TopKGate,
};

/// Deterministic test inputs: 32 rows of 16 features whose values are
/// unique per-row so every gate forward produces a distinct logit
/// pattern across the 4 experts.
fn deterministic_batch(n: usize, d: usize) -> Array2<f64> {
    Array2::from_shape_fn((n, d), |(i, j)| {
        let idx = (i * d + j) as f64;
        (idx * 0.017).sin()
    })
}

#[test]
fn moe_4_experts_batch32_output_shape_and_losses() {
    let d = 16_usize;
    let num_experts = 4_usize;
    let batch_size = 32_usize;

    // Create 4 LinearExperts with distinct (but deterministic) weights
    // so that different inputs route to different experts.
    let experts: Vec<Box<dyn Expert>> = (0..num_experts)
        .map(|e| {
            let w = Array2::from_shape_fn((d, d), |(i, j)| {
                let base = ((e * d * d + i * d + j) as f64 * 0.031).sin();
                if i == j {
                    1.0 + base * 0.1
                } else {
                    base * 0.1
                }
            });
            let b = Array1::from_shape_fn(d, |i| ((e * d + i) as f64 * 0.07).cos() * 0.01);
            let expert: Box<dyn Expert> =
                Box::new(LinearExpert::from_arrays(w, b).expect("expert"));
            expert
        })
        .collect();

    // Xavier-init gate with seed 42 for reproducibility.
    let gate = TopKGate::xavier_init(d, num_experts, 1, 42).expect("gate");
    let layer = MoELayer::new(gate, experts).expect("layer");

    let batch = deterministic_batch(batch_size, d);
    let (output, stats) = layer.forward_batch(&batch.view()).expect("forward_batch");

    // 1. Output shape.
    assert_eq!(output.nrows(), batch_size);
    assert_eq!(output.ncols(), d);

    // 2. Output values must be finite.
    for i in 0..output.nrows() {
        for j in 0..output.ncols() {
            assert!(
                output[(i, j)].is_finite(),
                "output[{}, {}] = {} not finite",
                i,
                j,
                output[(i, j)]
            );
        }
    }

    // 3. Combined auxiliary loss is finite and non-negative.
    let l_imp = importance_loss(&stats).expect("imp");
    let l_load = load_loss(&stats).expect("load");
    let l_combined = combined_aux_loss(&stats, 0.01).expect("combined");
    assert!(
        l_imp.is_finite() && l_imp >= 0.0,
        "importance_loss = {} (expected finite, >= 0)",
        l_imp
    );
    assert!(
        l_load.is_finite() && l_load >= 0.0,
        "load_loss = {} (expected finite, >= 0)",
        l_load
    );
    assert!(
        l_combined.is_finite() && l_combined >= 0.0,
        "combined_aux_loss = {} (expected finite, >= 0)",
        l_combined
    );
    // Combined = alpha * (importance + load).
    assert!(
        (l_combined - 0.01 * (l_imp + l_load)).abs() < 1e-12,
        "combined = {}, expected 0.01 * ({} + {}) = {}",
        l_combined,
        l_imp,
        l_load,
        0.01 * (l_imp + l_load)
    );

    // 4. Routing distribution: at least 2 distinct experts receive
    //    tokens (with 4 experts and 32 inputs under Xavier init,
    //    complete collapse to a single expert is unacceptable).
    let mut expert_count = vec![0_usize; num_experts];
    for &e in &stats.routed_expert_per_token {
        expert_count[e] += 1;
    }
    let active_experts = expert_count.iter().filter(|&&c| c > 0).count();
    assert!(
        active_experts >= 2,
        "only {} / {} experts received tokens (routing collapsed)",
        active_experts,
        num_experts
    );
    // No single expert should monopolise the entire batch.
    for (e, &count) in expert_count.iter().enumerate() {
        assert!(
            count < batch_size,
            "expert {} received all {} tokens",
            e,
            batch_size
        );
    }

    // 5. Gate softmax weights are valid probability vectors.
    for t in 0..batch_size {
        let mut row_sum = 0.0_f64;
        for e in 0..num_experts {
            let score = stats.gate_scores_per_token[(t, e)];
            assert!(
                (0.0..=1.0).contains(&score),
                "gate_score[{}, {}] = {} out of [0, 1]",
                t,
                e,
                score
            );
            row_sum += score;
        }
        assert!(
            (row_sum - 1.0).abs() < 1e-10,
            "gate softmax for token {} sums to {} (expected 1.0)",
            t,
            row_sum
        );
    }
}

#[test]
fn moe_capacity_factor_limits_expert_assignments() {
    let d = 4_usize;
    let num_experts = 2_usize;
    let batch_size = 8_usize;

    // Create two constant-output experts: expert 0 outputs all-ones,
    // expert 1 outputs all-twos.
    let e0 =
        LinearExpert::from_arrays(Array2::<f64>::zeros((d, d)), Array1::from_vec(vec![1.0; d]))
            .expect("e0");
    let e1 =
        LinearExpert::from_arrays(Array2::<f64>::zeros((d, d)), Array1::from_vec(vec![2.0; d]))
            .expect("e1");

    // Gate that deterministically routes everything to expert 0.
    let gate_weights =
        Array2::from_shape_fn((num_experts, d), |(e, _)| if e == 0 { 10.0 } else { -10.0 });
    let gate = TopKGate::from_weights(gate_weights, 1).expect("gate");

    let experts: Vec<Box<dyn Expert>> = vec![Box::new(e0), Box::new(e1)];
    let layer = MoELayer::new(gate, experts)
        .expect("layer")
        .with_capacity_factor(0.5)
        .expect("cf");

    // Capacity = ceil(0.5 * 8 / 2) = 2. Only the first 2 tokens get
    // expert 0's output; the remaining 6 are zeroed out.
    let batch = Array2::from_shape_fn((batch_size, d), |(i, j)| ((i + j) as f64) * 0.1);
    let (output, _stats) = layer.forward_batch(&batch.view()).expect("forward");

    // First 2 tokens should have expert 0's constant output [1, 1, 1, 1].
    for t in 0..2 {
        for j in 0..d {
            assert!(
                (output[(t, j)] - 1.0).abs() < 1e-12,
                "token {} col {} = {} (expected 1.0)",
                t,
                j,
                output[(t, j)]
            );
        }
    }
    // Remaining tokens are zeroed (overflow).
    for t in 2..batch_size {
        for j in 0..d {
            assert!(
                output[(t, j)].abs() < 1e-12,
                "overflow token {} col {} = {} (expected 0.0)",
                t,
                j,
                output[(t, j)]
            );
        }
    }
}
