//! Property-based tests for scirs2-metrics
//!
//! These tests verify mathematical properties of metrics using proptest.

use proptest::prelude::*;
use scirs2_core::ndarray::Array1;
use scirs2_metrics::{
    classification::accuracy_score,
    distance::*,
    information_theory::*,
    regression::{mean_absolute_error, mean_squared_error, r2_score},
};

// Helper function to generate valid probability distributions
fn probability_distribution() -> impl Strategy<Value = Array1<f64>> {
    prop::collection::vec(0.0_f64..1.0, 2..10).prop_map(|mut v| {
        // Normalize to sum to 1.0
        let sum: f64 = v.iter().sum();
        if sum > 0.0 {
            for x in &mut v {
                *x /= sum;
            }
        } else {
            // All zeros case - make uniform
            let n = v.len();
            for x in &mut v {
                *x = 1.0 / n as f64;
            }
        }
        Array1::from(v)
    })
}

// Helper function to generate non-zero vectors of a given length
fn nonzero_vector_of_len(len: usize) -> impl Strategy<Value = Array1<f64>> {
    prop::collection::vec(-100.0_f64..100.0, len)
        .prop_filter("must have non-zero norm", |v| {
            v.iter().any(|&x| x.abs() > 1e-10)
        })
        .prop_map(Array1::from)
}

// Helper function to generate a pair of non-zero vectors with the same random length
fn nonzero_vector_pair() -> impl Strategy<Value = (Array1<f64>, Array1<f64>)> {
    (2_usize..20).prop_flat_map(|len| (nonzero_vector_of_len(len), nonzero_vector_of_len(len)))
}

// Helper function to generate a triple of non-zero vectors with the same random length
fn nonzero_vector_triple() -> impl Strategy<Value = (Array1<f64>, Array1<f64>, Array1<f64>)> {
    (2_usize..20).prop_flat_map(|len| {
        (
            nonzero_vector_of_len(len),
            nonzero_vector_of_len(len),
            nonzero_vector_of_len(len),
        )
    })
}

// Helper for identity tests: a single non-zero vector
fn nonzero_vector() -> impl Strategy<Value = Array1<f64>> {
    nonzero_vector_of_len(5)
}

// Helper function to generate a pair of probability distributions with the same length
fn probability_distribution_pair() -> impl Strategy<Value = (Array1<f64>, Array1<f64>)> {
    (2_usize..10).prop_flat_map(|len| {
        let make_dist = move || {
            prop::collection::vec(0.01_f64..1.0, len).prop_map(|mut v| {
                let sum: f64 = v.iter().sum();
                for x in &mut v {
                    *x /= sum;
                }
                Array1::from(v)
            })
        };
        (make_dist(), make_dist())
    })
}

// ============================================================================
// Distance Metric Properties
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    // Property: Distance metrics are non-negative
    #[test]
    fn test_euclidean_non_negative((x, y) in nonzero_vector_pair()) {
        let dist = euclidean_distance(&x, &y);
        prop_assert!(dist.is_ok(), "euclidean_distance failed: {:?}", dist);
        if let Ok(d) = dist {
            prop_assert!(d >= 0.0);
        }
    }

    // Property: d(x, x) = 0 (identity of indiscernibles)
    #[test]
    fn test_euclidean_identity(x in nonzero_vector()) {
        let dist = euclidean_distance(&x, &x);
        prop_assert!(dist.is_ok());
        if let Ok(d) = dist {
            prop_assert!((d - 0.0).abs() < 1e-10);
        }
    }

    // Property: d(x, y) = d(y, x) (symmetry)
    #[test]
    fn test_euclidean_symmetry((x, y) in nonzero_vector_pair()) {
        let d_xy = euclidean_distance(&x, &y);
        let d_yx = euclidean_distance(&y, &x);
        prop_assert!(d_xy.is_ok() && d_yx.is_ok());
        if let (Ok(d1), Ok(d2)) = (d_xy, d_yx) {
            prop_assert!((d1 - d2).abs() < 1e-10);
        }
    }

    // Property: Triangle inequality d(x, z) <= d(x, y) + d(y, z)
    #[test]
    fn test_euclidean_triangle_inequality((x, y, z) in nonzero_vector_triple()) {
        let d_xz = euclidean_distance(&x, &z);
        let d_xy = euclidean_distance(&x, &y);
        let d_yz = euclidean_distance(&y, &z);

        prop_assert!(d_xz.is_ok() && d_xy.is_ok() && d_yz.is_ok());
        if let (Ok(dxz), Ok(dxy), Ok(dyz)) = (d_xz, d_xy, d_yz) {
            prop_assert!(dxz <= dxy + dyz + 1e-10); // Small epsilon for floating point
        }
    }

    // Property: Manhattan distance is non-negative
    #[test]
    fn test_manhattan_non_negative((x, y) in nonzero_vector_pair()) {
        let dist = manhattan_distance(&x, &y);
        prop_assert!(dist.is_ok(), "manhattan_distance failed: {:?}", dist);
        if let Ok(d) = dist {
            prop_assert!(d >= 0.0);
        }
    }

    // Property: Manhattan distance symmetry
    #[test]
    fn test_manhattan_symmetry((x, y) in nonzero_vector_pair()) {
        let d_xy = manhattan_distance(&x, &y);
        let d_yx = manhattan_distance(&y, &x);
        prop_assert!(d_xy.is_ok() && d_yx.is_ok());
        if let (Ok(d1), Ok(d2)) = (d_xy, d_yx) {
            prop_assert!((d1 - d2).abs() < 1e-10);
        }
    }

    // Property: Cosine similarity is bounded [-1, 1]
    #[test]
    fn test_cosine_similarity_bounds((x, y) in nonzero_vector_pair()) {
        let sim = cosine_similarity(&x, &y);
        prop_assert!(sim.is_ok(), "cosine_similarity failed: {:?}", sim);
        if let Ok(s) = sim {
            prop_assert!((-1.0 - 1e-10..=1.0 + 1e-10).contains(&s));
        }
    }

    // Property: Cosine similarity is symmetric
    #[test]
    fn test_cosine_similarity_symmetry((x, y) in nonzero_vector_pair()) {
        let sim_xy = cosine_similarity(&x, &y);
        let sim_yx = cosine_similarity(&y, &x);
        prop_assert!(sim_xy.is_ok() && sim_yx.is_ok());
        if let (Ok(s1), Ok(s2)) = (sim_xy, sim_yx) {
            prop_assert!((s1 - s2).abs() < 1e-10);
        }
    }

    // Property: cos(x, x) = 1
    #[test]
    fn test_cosine_similarity_identity(x in nonzero_vector()) {
        let sim = cosine_similarity(&x, &x);
        prop_assert!(sim.is_ok());
        if let Ok(s) = sim {
            prop_assert!((s - 1.0).abs() < 1e-10);
        }
    }
}

// ============================================================================
// Information Theory Properties
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    // Property: Entropy is non-negative
    #[test]
    fn test_entropy_non_negative(p in probability_distribution()) {
        let h = entropy(&p);
        prop_assert!(h.is_ok());
        if let Ok(h_val) = h {
            prop_assert!(h_val >= 0.0);
        }
    }

    // Property: KL divergence is non-negative
    #[test]
    fn test_kl_non_negative((p, q) in probability_distribution_pair()) {
        let kl = kl_divergence(&p, &q);
        if kl.is_ok() {
            if let Ok(kl_val) = kl {
                prop_assert!(kl_val >= -1e-10, "KL divergence was negative: {}", kl_val);
            }
        }
    }

    // Property: KL(P||P) = 0
    #[test]
    fn test_kl_identity(p in probability_distribution()) {
        let kl = kl_divergence(&p, &p);
        prop_assert!(kl.is_ok());
        if let Ok(kl_val) = kl {
            prop_assert!((kl_val - 0.0).abs() < 1e-10);
        }
    }

    // Property: JS divergence is symmetric
    #[test]
    fn test_js_symmetry((p, q) in probability_distribution_pair()) {
        let js_pq = js_divergence(&p, &q);
        let js_qp = js_divergence(&q, &p);

        prop_assert!(js_pq.is_ok() && js_qp.is_ok());
        if let (Ok(js1), Ok(js2)) = (js_pq, js_qp) {
            prop_assert!((js1 - js2).abs() < 1e-10);
        }
    }

    // Property: JS divergence is bounded [0, 1] (in bits)
    #[test]
    fn test_js_bounds((p, q) in probability_distribution_pair()) {
        let js = js_divergence(&p, &q);
        if js.is_ok() {
            if let Ok(js_val) = js {
                prop_assert!((0.0..=1.0 + 1e-10).contains(&js_val));
            }
        }
    }

    // Property: JS(P, P) = 0
    #[test]
    fn test_js_identity(p in probability_distribution()) {
        let js = js_divergence(&p, &p);
        prop_assert!(js.is_ok());
        if let Ok(js_val) = js {
            prop_assert!((js_val - 0.0).abs() < 1e-10);
        }
    }
}

// ============================================================================
// Regression Metric Properties
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    // Property: MSE is non-negative
    #[test]
    fn test_mse_non_negative((y_true, y_pred) in nonzero_vector_pair()) {
        let mse = mean_squared_error(&y_true, &y_pred);
        prop_assert!(mse.is_ok(), "mean_squared_error failed: {:?}", mse);
        if let Ok(mse_val) = mse {
            prop_assert!(mse_val >= 0.0);
        }
    }

    // Property: MSE(y, y) = 0
    #[test]
    fn test_mse_identity(y in nonzero_vector()) {
        let mse = mean_squared_error(&y, &y);
        prop_assert!(mse.is_ok());
        if let Ok(mse_val) = mse {
            prop_assert!((mse_val - 0.0).abs() < 1e-10);
        }
    }

    // Property: MAE is non-negative
    #[test]
    fn test_mae_non_negative((y_true, y_pred) in nonzero_vector_pair()) {
        let mae = mean_absolute_error(&y_true, &y_pred);
        prop_assert!(mae.is_ok(), "mean_absolute_error failed: {:?}", mae);
        if let Ok(mae_val) = mae {
            prop_assert!(mae_val >= 0.0);
        }
    }

    // Property: MAE(y, y) = 0
    #[test]
    fn test_mae_identity(y in nonzero_vector()) {
        let mae = mean_absolute_error(&y, &y);
        prop_assert!(mae.is_ok());
        if let Ok(mae_val) = mae {
            prop_assert!((mae_val - 0.0).abs() < 1e-10);
        }
    }

    // Property: R² = 1 for perfect predictions
    #[test]
    fn test_r2_perfect(y in nonzero_vector()) {
        let r2 = r2_score(&y, &y);
        if r2.is_ok() {
            if let Ok(r2_val) = r2 {
                prop_assert!((r2_val - 1.0).abs() < 1e-10);
            }
        }
    }

    // Property: MAE <= RMSE (by Jensen's inequality)
    #[test]
    fn test_mae_rmse_inequality((y_true, y_pred) in nonzero_vector_pair()) {
        let mae = mean_absolute_error(&y_true, &y_pred);
        let mse = mean_squared_error(&y_true, &y_pred);

        prop_assert!(mae.is_ok() && mse.is_ok());
        if let (Ok(mae_val), Ok(mse_val)) = (mae, mse) {
            let rmse = mse_val.sqrt();
            prop_assert!(mae_val <= rmse + 1e-10);
        }
    }
}

// ============================================================================
// Classification Metric Properties
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    // Property: Accuracy is in [0, 1]
    #[test]
    fn test_accuracy_bounds(
        labels in prop::collection::vec(0_i32..3, 5..20)
    ) {
        let y_true = Array1::from(labels.clone());
        let y_pred = Array1::from(labels);
        let acc = accuracy_score(&y_true, &y_pred);
        prop_assert!(acc.is_ok());
        if let Ok(acc_val) = acc {
            prop_assert!((0.0..=1.0).contains(&acc_val));
        }
    }

    // Property: Accuracy = 1 for perfect predictions
    #[test]
    fn test_accuracy_perfect(labels in prop::collection::vec(0_i32..5, 5..20)) {
        let y_true = Array1::from(labels.clone());
        let y_pred = Array1::from(labels);
        let acc = accuracy_score(&y_true, &y_pred);
        prop_assert!(acc.is_ok());
        if let Ok(acc_val) = acc {
            prop_assert!((acc_val - 1.0).abs() < 1e-10);
        }
    }

    // Property: Hamming distance is in [0, 1]
    #[test]
    fn test_hamming_bounds(
        x in prop::collection::vec(0_i32..2, 5..20),
        y in prop::collection::vec(0_i32..2, 5..20)
    ) {
        let x_arr = Array1::from(x);
        let y_arr = Array1::from(y);
        let dist = hamming_distance(&x_arr, &y_arr);
        if dist.is_ok() {
            if let Ok(d) = dist {
                prop_assert!((0.0..=1.0).contains(&d));
            }
        }
    }

    // Property: Hamming(x, x) = 0
    #[test]
    fn test_hamming_identity(x in prop::collection::vec(0_i32..5, 5..20)) {
        let x_arr = Array1::from(x);
        let dist = hamming_distance(&x_arr, &x_arr);
        prop_assert!(dist.is_ok());
        if let Ok(d) = dist {
            prop_assert!((d - 0.0).abs() < 1e-10);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_property_tests_compile() {
        // This test just ensures all property tests compile
        let compiles = true;
        assert!(compiles);
    }
}
