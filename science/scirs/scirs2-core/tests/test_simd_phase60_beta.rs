//! Beta function SIMD tests

use scirs2_core::ndarray::{array, Array1};
#[cfg(feature = "random")]
use scirs2_core::random::{thread_rng, Distribution, Uniform};

// =============================================================================
// =============================================================================
// Beta Function Tests (Phase 60)
// =============================================================================

use scirs2_core::ndarray_ext::elementwise::{beta_simd, ln_beta_simd};

// -----------------------------------------------------------------------------
// beta (Beta Function) Tests
// -----------------------------------------------------------------------------

/// Test beta basic f64
#[test]
fn test_beta_simd_basic_f64() {
    let a = array![1.0_f64, 2.0, 3.0];
    let b = array![1.0_f64, 2.0, 3.0];
    let result = beta_simd(&a.view(), &b.view());

    // B(1, 1) = 1
    assert!(
        (result[0] - 1.0).abs() < 1e-10,
        "B(1, 1) should be 1, got {}",
        result[0]
    );

    // B(2, 2) = Γ(2)Γ(2)/Γ(4) = 1*1/6 = 1/6
    assert!(
        (result[1] - 1.0 / 6.0).abs() < 1e-10,
        "B(2, 2) should be 1/6, got {}",
        result[1]
    );

    // B(3, 3) = Γ(3)Γ(3)/Γ(6) = 2*2/120 = 1/30
    assert!(
        (result[2] - 1.0 / 30.0).abs() < 1e-10,
        "B(3, 3) should be 1/30, got {}",
        result[2]
    );
}

/// Test beta basic f32
#[test]
fn test_beta_simd_basic_f32() {
    let a = array![1.0_f32, 2.0, 3.0];
    let b = array![1.0_f32, 2.0, 3.0];
    let result = beta_simd(&a.view(), &b.view());

    assert!(
        (result[0] - 1.0).abs() < 1e-5,
        "B(1, 1) should be 1, got {}",
        result[0]
    );
    assert!(
        (result[1] - 1.0 / 6.0).abs() < 1e-5,
        "B(2, 2) should be 1/6, got {}",
        result[1]
    );
}

/// Test beta empty array
#[test]
fn test_beta_simd_empty() {
    let a: Array1<f64> = array![];
    let b: Array1<f64> = array![];
    let result = beta_simd(&a.view(), &b.view());
    assert_eq!(result.len(), 0, "Empty input should give empty output");
}

/// Test beta large array (SIMD path)
#[test]
fn test_beta_simd_large_array() {
    let n = 1000;
    let a: Array1<f64> = Array1::linspace(1.0, 5.0, n);
    let b: Array1<f64> = Array1::linspace(1.0, 5.0, n);
    let result = beta_simd(&a.view(), &b.view());

    assert_eq!(result.len(), n);
    for (i, &v) in result.iter().enumerate() {
        assert!(
            v.is_finite() && v > 0.0,
            "Beta should be finite positive at index {}, got {}",
            i,
            v
        );
    }
}

/// Test beta symmetry B(a, b) = B(b, a)
#[test]
fn test_beta_simd_symmetry() {
    let a = array![1.5_f64, 2.5, 3.5, 0.5];
    let b = array![2.0_f64, 1.0, 4.0, 3.0];

    let result1 = beta_simd(&a.view(), &b.view());
    let result2 = beta_simd(&b.view(), &a.view());

    for i in 0..a.len() {
        assert!(
            (result1[i] - result2[i]).abs() < 1e-10,
            "B(a, b) should equal B(b, a) at index {}",
            i
        );
    }
}

/// Test beta special value B(0.5, 0.5) = π
#[test]
fn test_beta_simd_half_half() {
    let a = array![0.5_f64];
    let b = array![0.5_f64];
    let result = beta_simd(&a.view(), &b.view());

    assert!(
        (result[0] - std::f64::consts::PI).abs() < 1e-8,
        "B(0.5, 0.5) should be π, got {}",
        result[0]
    );
}

/// Test beta B(a, 1) = 1/a
#[test]
fn test_beta_simd_a_one() {
    let a = array![2.0_f64, 3.0, 4.0, 5.0];
    let b = array![1.0_f64, 1.0, 1.0, 1.0];
    let result = beta_simd(&a.view(), &b.view());

    for i in 0..a.len() {
        let expected = 1.0 / a[i];
        assert!(
            (result[i] - expected).abs() < 1e-10,
            "B({}, 1) should be 1/{} = {}, got {}",
            a[i],
            a[i],
            expected,
            result[i]
        );
    }
}

/// Test beta for Beta distribution use case
#[test]
fn test_beta_simd_distribution_use_case() {
    // Beta distribution parameters
    let alpha = array![2.0_f64, 5.0, 1.0, 0.5];
    let beta_param = array![5.0_f64, 2.0, 1.0, 0.5];
    let beta_val = beta_simd(&alpha.view(), &beta_param.view());

    // All beta values should be positive for valid parameters
    for (i, &b) in beta_val.iter().enumerate() {
        assert!(
            b > 0.0 && b.is_finite(),
            "Beta function should be positive for valid params, got {} at index {}",
            b,
            i
        );
    }
}

// -----------------------------------------------------------------------------
// ln_beta (Log-Beta Function) Tests
// -----------------------------------------------------------------------------

/// Test ln_beta basic f64
#[test]
fn test_ln_beta_simd_basic_f64() {
    let a = array![1.0_f64, 2.0, 3.0];
    let b = array![1.0_f64, 2.0, 3.0];
    let result = ln_beta_simd(&a.view(), &b.view());

    // ln(B(1, 1)) = ln(1) = 0
    assert!(
        result[0].abs() < 1e-10,
        "ln(B(1, 1)) should be 0, got {}",
        result[0]
    );

    // ln(B(2, 2)) = ln(1/6)
    let expected_ln_b22 = (1.0_f64 / 6.0).ln();
    assert!(
        (result[1] - expected_ln_b22).abs() < 1e-10,
        "ln(B(2, 2)) should be ln(1/6), got {}",
        result[1]
    );
}

/// Test ln_beta basic f32
#[test]
fn test_ln_beta_simd_basic_f32() {
    let a = array![1.0_f32, 2.0];
    let b = array![1.0_f32, 2.0];
    let result = ln_beta_simd(&a.view(), &b.view());

    assert!(
        result[0].abs() < 1e-5,
        "ln(B(1, 1)) should be 0, got {}",
        result[0]
    );
}

/// Test ln_beta empty array
#[test]
fn test_ln_beta_simd_empty() {
    let a: Array1<f64> = array![];
    let b: Array1<f64> = array![];
    let result = ln_beta_simd(&a.view(), &b.view());
    assert_eq!(result.len(), 0, "Empty input should give empty output");
}

/// Test ln_beta large array (SIMD path)
#[test]
fn test_ln_beta_simd_large_array() {
    let n = 1000;
    let a: Array1<f64> = Array1::linspace(1.0, 10.0, n);
    let b: Array1<f64> = Array1::linspace(1.0, 10.0, n);
    let result = ln_beta_simd(&a.view(), &b.view());

    assert_eq!(result.len(), n);
    for (i, &v) in result.iter().enumerate() {
        assert!(
            v.is_finite(),
            "ln(Beta) should be finite at index {}, got {}",
            i,
            v
        );
    }
}

/// Test ln_beta symmetry
#[test]
fn test_ln_beta_simd_symmetry() {
    let a = array![1.5_f64, 2.5, 3.5];
    let b = array![2.0_f64, 1.0, 4.0];

    let result1 = ln_beta_simd(&a.view(), &b.view());
    let result2 = ln_beta_simd(&b.view(), &a.view());

    for i in 0..a.len() {
        assert!(
            (result1[i] - result2[i]).abs() < 1e-10,
            "ln(B(a, b)) should equal ln(B(b, a)) at index {}",
            i
        );
    }
}

/// Test ln_beta consistency with beta: exp(ln_beta) = beta
#[test]
fn test_ln_beta_simd_consistency() {
    let a = array![1.0_f64, 2.0, 3.0, 0.5, 1.5];
    let b = array![1.0_f64, 2.0, 3.0, 0.5, 2.5];

    let ln_beta_val = ln_beta_simd(&a.view(), &b.view());
    let beta_val = beta_simd(&a.view(), &b.view());

    for i in 0..a.len() {
        let exp_ln_beta = ln_beta_val[i].exp();
        assert!(
            (exp_ln_beta - beta_val[i]).abs() < 1e-10,
            "exp(ln(B(a,b))) should equal B(a,b), got {} vs {} at index {}",
            exp_ln_beta,
            beta_val[i],
            i
        );
    }
}

/// Test ln_beta numerical stability for large arguments
#[test]
fn test_ln_beta_simd_large_args() {
    // For large arguments, beta would overflow but ln_beta should be stable
    let a = array![100.0_f64, 200.0, 500.0];
    let b = array![100.0_f64, 200.0, 500.0];
    let result = ln_beta_simd(&a.view(), &b.view());

    // Results should be finite (very negative for large symmetric args)
    for (i, &v) in result.iter().enumerate() {
        assert!(
            v.is_finite(),
            "ln(Beta) should be finite for large args at index {}, got {}",
            i,
            v
        );
        assert!(
            v < 0.0,
            "ln(Beta) should be negative for large args, got {}",
            v
        );
    }
}

/// Test ln_beta for Bayesian inference use case
#[test]
fn test_ln_beta_simd_bayesian_use_case() {
    // Common Beta distribution parameters for A/B testing
    // Prior: Beta(1, 1) = uniform
    // Posterior after observations: Beta(alpha + successes, beta + failures)
    let alpha = array![1.0_f64, 10.0, 50.0, 100.0]; // Prior + successes
    let beta_param = array![1.0_f64, 10.0, 50.0, 100.0]; // Prior + failures

    let ln_beta_val = ln_beta_simd(&alpha.view(), &beta_param.view());

    // All log-beta values should be finite
    for (i, &lb) in ln_beta_val.iter().enumerate() {
        assert!(
            lb.is_finite(),
            "ln(Beta) should be finite for Bayesian params at index {}",
            i
        );
    }

    // ln(Beta) should decrease as parameters increase (normalizing constant gets smaller)
    for i in 0..ln_beta_val.len() - 1 {
        assert!(
            ln_beta_val[i] > ln_beta_val[i + 1],
            "ln(Beta) should decrease as params increase"
        );
    }
}
