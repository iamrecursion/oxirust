//! Gamma-related function SIMD tests: digamma, trigamma, loggamma

use scirs2_core::ndarray::{array, Array1};
use scirs2_core::ndarray_ext::elementwise::gamma_simd;
#[cfg(feature = "random")]
use scirs2_core::random::{thread_rng, Distribution, Uniform};

// ============ Digamma (ψ) Tests ============

use scirs2_core::ndarray_ext::elementwise::digamma_simd;

/// Euler-Mascheroni constant γ ≈ 0.5772156649
const EULER_MASCHERONI: f64 = 0.5772156649015329;

#[test]
fn test_digamma_simd_f64_at_one() {
    // ψ(1) = -γ (Euler-Mascheroni constant)
    let x = array![1.0_f64];
    let result = digamma_simd(&x.view());

    assert!(
        (result[0] - (-EULER_MASCHERONI)).abs() < 1e-10,
        "ψ(1) should equal -γ: got {}, expected {}",
        result[0],
        -EULER_MASCHERONI
    );
}

#[test]
fn test_digamma_simd_f64_positive_integers() {
    // ψ(n) = -γ + H_(n-1) where H_k = 1 + 1/2 + 1/3 + ... + 1/k (harmonic number)
    // ψ(1) = -γ
    // ψ(2) = -γ + 1
    // ψ(3) = -γ + 1 + 1/2
    // ψ(4) = -γ + 1 + 1/2 + 1/3
    let x = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let result = digamma_simd(&x.view());

    let expected = [
        -EULER_MASCHERONI,
        -EULER_MASCHERONI + 1.0,
        -EULER_MASCHERONI + 1.0 + 0.5,
        -EULER_MASCHERONI + 1.0 + 0.5 + 1.0 / 3.0,
        -EULER_MASCHERONI + 1.0 + 0.5 + 1.0 / 3.0 + 0.25,
    ];

    for i in 0..x.len() {
        assert!(
            (result[i] - expected[i]).abs() < 1e-10,
            "ψ({}) should be {}: got {}",
            i + 1,
            expected[i],
            result[i]
        );
    }
}

#[test]
fn test_digamma_simd_f64_recurrence_relation() {
    // ψ(x+1) = ψ(x) + 1/x
    let x_vals = array![1.5_f64, 2.5, 3.5, 4.5];
    let x_plus_one = array![2.5_f64, 3.5, 4.5, 5.5];

    let psi_x = digamma_simd(&x_vals.view());
    let psi_x_plus_one = digamma_simd(&x_plus_one.view());

    for i in 0..x_vals.len() {
        let expected = psi_x[i] + 1.0 / x_vals[i];
        assert!(
            (psi_x_plus_one[i] - expected).abs() < 1e-12,
            "ψ({}) should equal ψ({}) + 1/{}: got {}, expected {}",
            x_plus_one[i],
            x_vals[i],
            x_vals[i],
            psi_x_plus_one[i],
            expected
        );
    }
}

#[test]
fn test_digamma_simd_f64_half_integer() {
    // ψ(1/2) = -γ - 2ln(2) ≈ -1.9635
    let x = array![0.5_f64];
    let result = digamma_simd(&x.view());

    let expected = -EULER_MASCHERONI - 2.0 * std::f64::consts::LN_2;
    assert!(
        (result[0] - expected).abs() < 1e-10,
        "ψ(1/2) should be -γ - 2ln(2): got {}, expected {}",
        result[0],
        expected
    );
}

#[test]
fn test_digamma_simd_f64_large_values() {
    // For large x, ψ(x) ≈ ln(x) - 1/(2x) - 1/(12x²) + ...
    // At x = 100, the approximation should be very accurate
    let x = array![100.0_f64, 1000.0, 10000.0];
    let result = digamma_simd(&x.view());

    for i in 0..x.len() {
        let xi = x[i];
        // Leading term approximation
        let approx = xi.ln() - 0.5 / xi - 1.0 / (12.0 * xi * xi);
        assert!(
            (result[i] - approx).abs() < 1e-6,
            "ψ({}) should be approximately ln(x) - 1/(2x): got {}, expected {}",
            xi,
            result[i],
            approx
        );
    }
}

#[test]
fn test_digamma_simd_f64_nan_and_poles() {
    // ψ at non-positive integers should be NaN (poles)
    let x = array![0.0_f64, -1.0, -2.0, -3.0];
    let result = digamma_simd(&x.view());

    for i in 0..x.len() {
        assert!(
            result[i].is_nan(),
            "ψ({}) should be NaN (pole): got {}",
            x[i],
            result[i]
        );
    }
}

#[test]
fn test_digamma_simd_f64_negative_non_integers() {
    // For negative non-integers, use reflection: ψ(1-x) - ψ(x) = π·cot(πx)
    // This means: ψ(x) = ψ(1-x) - π·cot(πx) for x < 0
    let x = array![-0.5_f64, -1.5, -2.5];
    let result = digamma_simd(&x.view());

    // All results should be finite (not NaN)
    for i in 0..x.len() {
        assert!(
            result[i].is_finite(),
            "ψ({}) should be finite: got {}",
            x[i],
            result[i]
        );
    }

    // ψ(-0.5) using reflection: ψ(1.5) - π·cot(-π/2) = ψ(1.5) - 0
    let psi_1_5 = digamma_simd(&array![1.5_f64].view())[0];
    assert!(
        (result[0] - psi_1_5).abs() < 1e-10,
        "ψ(-0.5) should equal ψ(1.5): got {}, expected {}",
        result[0],
        psi_1_5
    );
}

#[test]
fn test_digamma_simd_f32_basic() {
    let x = array![1.0_f32, 2.0, 3.0, 4.0];
    let result = digamma_simd(&x.view());

    let gamma32 = EULER_MASCHERONI as f32;
    let expected = [
        -gamma32,
        -gamma32 + 1.0,
        -gamma32 + 1.0 + 0.5,
        -gamma32 + 1.0 + 0.5 + 1.0 / 3.0,
    ];

    for i in 0..x.len() {
        assert!(
            (result[i] - expected[i]).abs() < 1e-5,
            "ψ({}) f32 should be {}: got {}",
            i + 1,
            expected[i],
            result[i]
        );
    }
}

#[test]
fn test_digamma_simd_f32_recurrence() {
    let x_vals = array![2.0_f32, 3.0, 4.0];
    let x_plus_one = array![3.0_f32, 4.0, 5.0];

    let psi_x = digamma_simd(&x_vals.view());
    let psi_x_plus_one = digamma_simd(&x_plus_one.view());

    for i in 0..x_vals.len() {
        let expected = psi_x[i] + 1.0 / x_vals[i];
        assert!(
            (psi_x_plus_one[i] - expected).abs() < 1e-5,
            "ψ({}) f32 should equal ψ({}) + 1/{}: got {}, expected {}",
            x_plus_one[i],
            x_vals[i],
            x_vals[i],
            psi_x_plus_one[i],
            expected
        );
    }
}

#[test]
fn test_digamma_simd_empty() {
    let empty: Array1<f64> = Array1::zeros(0);
    let result = digamma_simd(&empty.view());
    assert!(result.is_empty());
}

#[test]
fn test_digamma_simd_large_array() {
    // Test with a larger array to exercise SIMD paths
    let n = 1000;
    let x: Array1<f64> = Array1::from_iter((1..=n).map(|i| i as f64));
    let result = digamma_simd(&x.view());

    // Verify first and last few values
    assert!((result[0] - (-EULER_MASCHERONI)).abs() < 1e-10);

    // ψ(1000) ≈ ln(1000) - 1/(2*1000) ≈ 6.906
    let approx_last = 1000.0_f64.ln() - 0.5 / 1000.0;
    assert!((result[999] - approx_last).abs() < 1e-3);
}

#[test]
fn test_digamma_simd_infinity() {
    // ψ(+∞) = +∞
    let x = array![f64::INFINITY];
    let result = digamma_simd(&x.view());
    assert!(result[0].is_infinite() && result[0] > 0.0);

    // ψ(-∞) = NaN
    let x_neg = array![f64::NEG_INFINITY];
    let result_neg = digamma_simd(&x_neg.view());
    assert!(result_neg[0].is_nan());
}

#[test]
fn test_digamma_simd_nan_input() {
    let x = array![f64::NAN];
    let result = digamma_simd(&x.view());
    assert!(result[0].is_nan());
}

// ============ Trigamma (ψ') Tests ============

use scirs2_core::ndarray_ext::elementwise::trigamma_simd;

/// π²/6 ≈ 1.6449340668 (Basel problem - ψ'(1))
const PI_SQUARED_OVER_6: f64 = std::f64::consts::PI * std::f64::consts::PI / 6.0;

#[test]
fn test_trigamma_simd_f64_at_one() {
    // ψ'(1) = π²/6 (Basel problem)
    let x = array![1.0_f64];
    let result = trigamma_simd(&x.view());

    assert!(
        (result[0] - PI_SQUARED_OVER_6).abs() < 1e-10,
        "ψ'(1) should equal π²/6: got {}, expected {}",
        result[0],
        PI_SQUARED_OVER_6
    );
}

#[test]
fn test_trigamma_simd_f64_positive_integers() {
    // ψ'(n) = π²/6 - Σ(k=1 to n-1) 1/k² for positive integers
    // ψ'(1) = π²/6
    // ψ'(2) = π²/6 - 1
    // ψ'(3) = π²/6 - 1 - 1/4
    // ψ'(4) = π²/6 - 1 - 1/4 - 1/9
    let x = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let result = trigamma_simd(&x.view());

    let expected = [
        PI_SQUARED_OVER_6,
        PI_SQUARED_OVER_6 - 1.0,
        PI_SQUARED_OVER_6 - 1.0 - 0.25,
        PI_SQUARED_OVER_6 - 1.0 - 0.25 - 1.0 / 9.0,
        PI_SQUARED_OVER_6 - 1.0 - 0.25 - 1.0 / 9.0 - 0.0625,
    ];

    for i in 0..x.len() {
        assert!(
            (result[i] - expected[i]).abs() < 1e-10,
            "ψ'({}) should be {}: got {}",
            i + 1,
            expected[i],
            result[i]
        );
    }
}

#[test]
fn test_trigamma_simd_f64_recurrence_relation() {
    // ψ'(x+1) = ψ'(x) - 1/x²
    let x_vals = array![1.5_f64, 2.5, 3.5, 4.5];
    let x_plus_one = array![2.5_f64, 3.5, 4.5, 5.5];

    let psi1_x = trigamma_simd(&x_vals.view());
    let psi1_x_plus_one = trigamma_simd(&x_plus_one.view());

    for i in 0..x_vals.len() {
        let expected = psi1_x[i] - 1.0 / (x_vals[i] * x_vals[i]);
        assert!(
            (psi1_x_plus_one[i] - expected).abs() < 1e-12,
            "ψ'({}) should equal ψ'({}) - 1/{}²: got {}, expected {}",
            x_plus_one[i],
            x_vals[i],
            x_vals[i],
            psi1_x_plus_one[i],
            expected
        );
    }
}

#[test]
fn test_trigamma_simd_f64_half_integer() {
    // ψ'(1/2) = π²/2 = 3 * π²/6 (from reflection and special value)
    let x = array![0.5_f64];
    let result = trigamma_simd(&x.view());

    // ψ'(1/2) ≈ 4.9348 (π²/2)
    let expected = std::f64::consts::PI * std::f64::consts::PI / 2.0;
    assert!(
        (result[0] - expected).abs() < 1e-9,
        "ψ'(1/2) should be π²/2: got {}, expected {}",
        result[0],
        expected
    );
}

#[test]
fn test_trigamma_simd_f64_large_values() {
    // For large x, ψ'(x) ≈ 1/x + 1/(2x²)
    let x = array![100.0_f64, 1000.0, 10000.0];
    let result = trigamma_simd(&x.view());

    for i in 0..x.len() {
        let xi = x[i];
        // Leading term approximation
        let approx = 1.0 / xi + 1.0 / (2.0 * xi * xi);
        assert!(
            (result[i] - approx).abs() < 1e-6,
            "ψ'({}) should be approximately 1/x + 1/(2x²): got {}, expected {}",
            xi,
            result[i],
            approx
        );
    }
}

#[test]
fn test_trigamma_simd_f64_nan_and_poles() {
    // ψ' at non-positive integers should be NaN (poles)
    let x = array![0.0_f64, -1.0, -2.0, -3.0];
    let result = trigamma_simd(&x.view());

    for i in 0..x.len() {
        assert!(
            result[i].is_nan(),
            "ψ'({}) should be NaN (pole): got {}",
            x[i],
            result[i]
        );
    }
}

#[test]
fn test_trigamma_simd_f64_negative_non_integers() {
    // For negative non-integers, use reflection: ψ'(1-x) + ψ'(x) = π²/sin²(πx)
    let x = array![-0.5_f64, -1.5, -2.5];
    let result = trigamma_simd(&x.view());

    // All results should be finite (not NaN)
    for i in 0..x.len() {
        assert!(
            result[i].is_finite(),
            "ψ'({}) should be finite: got {}",
            x[i],
            result[i]
        );
    }

    // ψ'(-0.5) using reflection: π²/sin²(-π/2) - ψ'(1.5) = π² - ψ'(1.5)
    let psi1_1_5 = trigamma_simd(&array![1.5_f64].view())[0];
    let pi_sq = std::f64::consts::PI * std::f64::consts::PI;
    let sin_neg_half_pi = (-std::f64::consts::PI * 0.5).sin();
    let expected = pi_sq / (sin_neg_half_pi * sin_neg_half_pi) - psi1_1_5;
    assert!(
        (result[0] - expected).abs() < 1e-9,
        "ψ'(-0.5) should follow reflection formula: got {}, expected {}",
        result[0],
        expected
    );
}

#[test]
fn test_trigamma_simd_f32_basic() {
    let x = array![1.0_f32, 2.0, 3.0, 4.0];
    let result = trigamma_simd(&x.view());

    let pi_sq_6_f32 = std::f32::consts::PI * std::f32::consts::PI / 6.0;
    let expected = [
        pi_sq_6_f32,
        pi_sq_6_f32 - 1.0,
        pi_sq_6_f32 - 1.0 - 0.25,
        pi_sq_6_f32 - 1.0 - 0.25 - 1.0 / 9.0,
    ];

    for i in 0..x.len() {
        assert!(
            (result[i] - expected[i]).abs() < 1e-5,
            "ψ'({}) f32 should be {}: got {}",
            i + 1,
            expected[i],
            result[i]
        );
    }
}

#[test]
fn test_trigamma_simd_f32_recurrence() {
    let x_vals = array![2.0_f32, 3.0, 4.0];
    let x_plus_one = array![3.0_f32, 4.0, 5.0];

    let psi1_x = trigamma_simd(&x_vals.view());
    let psi1_x_plus_one = trigamma_simd(&x_plus_one.view());

    for i in 0..x_vals.len() {
        let expected = psi1_x[i] - 1.0 / (x_vals[i] * x_vals[i]);
        assert!(
            (psi1_x_plus_one[i] - expected).abs() < 1e-5,
            "ψ'({}) f32 should equal ψ'({}) - 1/{}²: got {}, expected {}",
            x_plus_one[i],
            x_vals[i],
            x_vals[i],
            psi1_x_plus_one[i],
            expected
        );
    }
}

#[test]
fn test_trigamma_simd_empty() {
    let empty: Array1<f64> = Array1::zeros(0);
    let result = trigamma_simd(&empty.view());
    assert!(result.is_empty());
}

#[test]
fn test_trigamma_simd_large_array() {
    // Test with a larger array to exercise SIMD paths
    let n = 1000;
    let x: Array1<f64> = Array1::from_iter((1..=n).map(|i| i as f64));
    let result = trigamma_simd(&x.view());

    // Verify first value: ψ'(1) = π²/6
    assert!((result[0] - PI_SQUARED_OVER_6).abs() < 1e-10);

    // ψ'(1000) ≈ 1/1000 + 1/(2*1000²) ≈ 0.001
    let approx_last = 1.0 / 1000.0 + 0.5 / (1000.0 * 1000.0);
    assert!((result[999] - approx_last).abs() < 1e-6);
}

#[test]
fn test_trigamma_simd_infinity() {
    // ψ'(+∞) = 0
    let x = array![f64::INFINITY];
    let result = trigamma_simd(&x.view());
    assert!(
        (result[0] - 0.0).abs() < 1e-10,
        "ψ'(+∞) should be 0: got {}",
        result[0]
    );

    // ψ'(-∞) = NaN
    let x_neg = array![f64::NEG_INFINITY];
    let result_neg = trigamma_simd(&x_neg.view());
    assert!(result_neg[0].is_nan());
}

#[test]
fn test_trigamma_simd_nan_input() {
    let x = array![f64::NAN];
    let result = trigamma_simd(&x.view());
    assert!(result[0].is_nan());
}

#[test]
fn test_trigamma_simd_positivity() {
    // ψ'(x) should always be positive for x > 0
    let x = array![0.1_f64, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0];
    let result = trigamma_simd(&x.view());

    for i in 0..x.len() {
        assert!(
            result[i] > 0.0,
            "ψ'({}) should be positive: got {}",
            x[i],
            result[i]
        );
    }
}

#[test]
fn test_trigamma_simd_monotonically_decreasing() {
    // ψ'(x) is strictly decreasing for x > 0
    let x = array![0.5_f64, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0];
    let result = trigamma_simd(&x.view());

    for i in 1..x.len() {
        assert!(
            result[i] < result[i - 1],
            "ψ'({}) should be less than ψ'({}): got {} >= {}",
            x[i],
            x[i - 1],
            result[i],
            result[i - 1]
        );
    }
}

// ============ Log-Gamma (ln Γ) Tests ============

use scirs2_core::ndarray_ext::elementwise::ln_gamma_simd;

/// ln(√π) ≈ 0.5724 - value of ln(Γ(1/2))
const LN_SQRT_PI: f64 = 0.5723649429247001;

#[test]
fn test_ln_gamma_simd_f64_at_one_and_two() {
    // ln(Γ(1)) = ln(0!) = ln(1) = 0
    // ln(Γ(2)) = ln(1!) = ln(1) = 0
    let x = array![1.0_f64, 2.0];
    let result = ln_gamma_simd(&x.view());

    assert!(
        result[0].abs() < 1e-12,
        "ln(Γ(1)) should be 0: got {}",
        result[0]
    );
    assert!(
        result[1].abs() < 1e-12,
        "ln(Γ(2)) should be 0: got {}",
        result[1]
    );
}

#[test]
fn test_ln_gamma_simd_f64_factorials() {
    // ln(Γ(n)) = ln((n-1)!) for positive integers
    // ln(Γ(3)) = ln(2!) = ln(2) ≈ 0.693
    // ln(Γ(4)) = ln(3!) = ln(6) ≈ 1.792
    // ln(Γ(5)) = ln(4!) = ln(24) ≈ 3.178
    // ln(Γ(6)) = ln(5!) = ln(120) ≈ 4.787
    let x = array![3.0_f64, 4.0, 5.0, 6.0];
    let result = ln_gamma_simd(&x.view());

    let expected = [
        2.0_f64.ln(),   // ln(2!)
        6.0_f64.ln(),   // ln(3!)
        24.0_f64.ln(),  // ln(4!)
        120.0_f64.ln(), // ln(5!)
    ];

    for i in 0..x.len() {
        assert!(
            (result[i] - expected[i]).abs() < 1e-10,
            "ln(Γ({})) should be ln(({}!)): got {}, expected {}",
            x[i] as i32,
            x[i] as i32 - 1,
            result[i],
            expected[i]
        );
    }
}

#[test]
fn test_ln_gamma_simd_f64_half_integer() {
    // ln(Γ(1/2)) = ln(√π) ≈ 0.5724
    let x = array![0.5_f64];
    let result = ln_gamma_simd(&x.view());

    assert!(
        (result[0] - LN_SQRT_PI).abs() < 1e-10,
        "ln(Γ(1/2)) should be ln(√π): got {}, expected {}",
        result[0],
        LN_SQRT_PI
    );
}

#[test]
fn test_ln_gamma_simd_f64_recurrence_relation() {
    // Γ(x+1) = x·Γ(x), so ln(Γ(x+1)) = ln(x) + ln(Γ(x))
    let x_vals = array![1.5_f64, 2.5, 3.5, 4.5];
    let x_plus_one = array![2.5_f64, 3.5, 4.5, 5.5];

    let lng_x = ln_gamma_simd(&x_vals.view());
    let lng_x_plus_one = ln_gamma_simd(&x_plus_one.view());

    for i in 0..x_vals.len() {
        let expected = x_vals[i].ln() + lng_x[i];
        assert!(
            (lng_x_plus_one[i] - expected).abs() < 1e-10,
            "ln(Γ({})) should equal ln({}) + ln(Γ({})): got {}, expected {}",
            x_plus_one[i],
            x_vals[i],
            x_vals[i],
            lng_x_plus_one[i],
            expected
        );
    }
}

#[test]
fn test_ln_gamma_simd_f64_large_values() {
    // For large x: ln(Γ(x)) ≈ (x-0.5)·ln(x) - x + ln(√(2π))
    let ln_sqrt_2pi = (2.0 * std::f64::consts::PI).sqrt().ln();
    let x = array![100.0_f64, 1000.0];
    let result = ln_gamma_simd(&x.view());

    for i in 0..x.len() {
        let xi = x[i];
        // Stirling's approximation (basic form without correction terms)
        let approx = (xi - 0.5) * xi.ln() - xi + ln_sqrt_2pi;
        // For large x, the basic Stirling approximation error is O(1/x)
        // With correction term +1/(12x), error is O(1/x³)
        let approx_with_correction = approx + 1.0 / (12.0 * xi);
        let rel_error = ((result[i] - approx_with_correction) / result[i]).abs();
        assert!(
            rel_error < 1e-5,
            "ln(Γ({})) should follow Stirling's approximation: got {}, approx {}",
            xi,
            result[i],
            approx_with_correction
        );
    }
}

#[test]
fn test_ln_gamma_simd_f64_poles() {
    // ln(Γ) at non-positive integers should be +∞ (Γ has poles)
    let x = array![0.0_f64, -1.0, -2.0, -3.0];
    let result = ln_gamma_simd(&x.view());

    for i in 0..x.len() {
        assert!(
            result[i].is_infinite() && result[i] > 0.0,
            "ln(Γ({})) should be +∞ (pole): got {}",
            x[i],
            result[i]
        );
    }
}

#[test]
fn test_ln_gamma_simd_f64_negative_non_integers() {
    // For negative non-integers, ln(Γ(x)) is finite
    let x = array![-0.5_f64, -1.5, -2.5];
    let result = ln_gamma_simd(&x.view());

    // All results should be finite (not NaN, not ±∞)
    for i in 0..x.len() {
        assert!(
            result[i].is_finite(),
            "ln(Γ({})) should be finite: got {}",
            x[i],
            result[i]
        );
    }

    // Verify using reflection: ln(Γ(x)) + ln(Γ(1-x)) = ln(π/sin(πx))
    // For x = -0.5: ln(Γ(-0.5)) + ln(Γ(1.5)) = ln(π/sin(-π/2)) = ln(π)
    let lng_1_5 = ln_gamma_simd(&array![1.5_f64].view())[0];
    let expected_sum = std::f64::consts::PI.ln(); // ln(π/|sin(-π/2)|) = ln(π/1) = ln(π)
    assert!(
        (result[0] + lng_1_5 - expected_sum).abs() < 1e-9,
        "ln(Γ(-0.5)) + ln(Γ(1.5)) should equal ln(π): got {}, expected {}",
        result[0] + lng_1_5,
        expected_sum
    );
}

#[test]
fn test_ln_gamma_simd_f32_basic() {
    let x = array![1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let result = ln_gamma_simd(&x.view());

    let expected = [
        0.0_f32,       // ln(Γ(1)) = 0
        0.0,           // ln(Γ(2)) = 0
        2.0_f32.ln(),  // ln(2!)
        6.0_f32.ln(),  // ln(3!)
        24.0_f32.ln(), // ln(4!)
    ];

    for i in 0..x.len() {
        assert!(
            (result[i] - expected[i]).abs() < 1e-5,
            "ln(Γ({})) f32 should be {}: got {}",
            x[i] as i32,
            expected[i],
            result[i]
        );
    }
}

#[test]
fn test_ln_gamma_simd_empty() {
    let empty: Array1<f64> = Array1::zeros(0);
    let result = ln_gamma_simd(&empty.view());
    assert!(result.is_empty());
}

#[test]
fn test_ln_gamma_simd_large_array() {
    // Test with a larger array to exercise SIMD paths
    let n = 1000;
    let x: Array1<f64> = Array1::from_iter((1..=n).map(|i| i as f64));
    let result = ln_gamma_simd(&x.view());

    // Verify first few values
    assert!(result[0].abs() < 1e-12); // ln(Γ(1)) = 0
    assert!(result[1].abs() < 1e-12); // ln(Γ(2)) = 0
    assert!((result[2] - 2.0_f64.ln()).abs() < 1e-10); // ln(Γ(3)) = ln(2!)

    // Verify last value using Stirling
    let ln_sqrt_2pi = (2.0 * std::f64::consts::PI).sqrt().ln();
    let x_last = 1000.0_f64;
    let stirling = (x_last - 0.5) * x_last.ln() - x_last + ln_sqrt_2pi;
    assert!((result[999] - stirling).abs() / result[999] < 1e-6);
}

#[test]
fn test_ln_gamma_simd_infinity() {
    // ln(Γ(+∞)) = +∞
    let x = array![f64::INFINITY];
    let result = ln_gamma_simd(&x.view());
    assert!(result[0].is_infinite() && result[0] > 0.0);

    // ln(Γ(-∞)) = NaN
    let x_neg = array![f64::NEG_INFINITY];
    let result_neg = ln_gamma_simd(&x_neg.view());
    assert!(result_neg[0].is_nan());
}

#[test]
fn test_ln_gamma_simd_nan_input() {
    let x = array![f64::NAN];
    let result = ln_gamma_simd(&x.view());
    assert!(result[0].is_nan());
}

#[test]
fn test_ln_gamma_simd_consistency_with_gamma() {
    // For moderate values, ln(Γ(x)) should equal ln(gamma(x))
    // (avoiding overflow range)
    let x = array![2.0_f64, 3.0, 4.0, 5.0, 10.0];
    let ln_gamma_result = ln_gamma_simd(&x.view());

    // Compute gamma and take ln manually
    use scirs2_core::ndarray_ext::elementwise::gamma_simd;
    let gamma_result = gamma_simd(&x.view());

    for i in 0..x.len() {
        let expected = gamma_result[i].ln();
        assert!(
            (ln_gamma_result[i] - expected).abs() < 1e-10,
            "ln_gamma({}) should equal ln(gamma({})): got {}, expected {}",
            x[i],
            x[i],
            ln_gamma_result[i],
            expected
        );
    }
}

#[test]
fn test_ln_gamma_simd_stirling_asymptotic() {
    // Verify Stirling's approximation becomes more accurate for larger x
    // ln(Γ(x)) ≈ (x-0.5)·ln(x) - x + ln(√(2π)) + 1/(12x) - 1/(360x³) + ...
    let ln_sqrt_2pi = (2.0 * std::f64::consts::PI).sqrt().ln();
    let x = array![10.0_f64, 50.0, 100.0, 500.0, 1000.0];
    let result = ln_gamma_simd(&x.view());

    for i in 0..x.len() {
        let xi = x[i];
        // Stirling with first correction term
        let stirling = (xi - 0.5) * xi.ln() - xi + ln_sqrt_2pi + 1.0 / (12.0 * xi);
        let rel_error = ((result[i] - stirling) / result[i]).abs();
        // Error should decrease with increasing x
        assert!(
            rel_error < 1e-4 / (xi / 10.0),
            "ln(Γ({})) Stirling error too large: got {}, Stirling {}",
            xi,
            result[i],
            stirling
        );
    }
}
