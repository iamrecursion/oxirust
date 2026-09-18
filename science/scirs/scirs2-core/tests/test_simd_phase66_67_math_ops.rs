//! Phase 66-67: SIMD math operations tests (square, rsqrt, sincos)

use scirs2_core::ndarray::{array, Array1};

// ============================================================================
// Phase 66-67: SIMD square, rsqrt, sincos tests
// ============================================================================

use scirs2_core::ndarray_ext::elementwise::rsqrt_simd;
use scirs2_core::ndarray_ext::elementwise::sincos_simd;
use scirs2_core::ndarray_ext::elementwise::square_simd;
use std::f64::consts::PI;

// ============================================================================
// Square tests
// ============================================================================

/// Test square basic f32 correctness
#[test]
fn test_square_simd_f32_basic() {
    let x = array![1.0_f32, 2.0, 3.0, -4.0, 0.5];

    let result = square_simd(&x.view());

    assert!((result[0] - 1.0).abs() < 1e-6); // 1² = 1
    assert!((result[1] - 4.0).abs() < 1e-6); // 2² = 4
    assert!((result[2] - 9.0).abs() < 1e-6); // 3² = 9
    assert!((result[3] - 16.0).abs() < 1e-6); // (-4)² = 16
    assert!((result[4] - 0.25).abs() < 1e-6); // 0.5² = 0.25
}

/// Test square basic f64 correctness
#[test]
fn test_square_simd_f64_basic() {
    let x = array![1.0_f64, 2.0, 3.0, -4.0, 0.5];

    let result = square_simd(&x.view());

    assert!((result[0] - 1.0).abs() < 1e-14); // 1² = 1
    assert!((result[1] - 4.0).abs() < 1e-14); // 2² = 4
    assert!((result[2] - 9.0).abs() < 1e-14); // 3² = 9
    assert!((result[3] - 16.0).abs() < 1e-14); // (-4)² = 16
    assert!((result[4] - 0.25).abs() < 1e-14); // 0.5² = 0.25
}

/// Test square empty array
#[test]
fn test_square_simd_empty() {
    let x: Array1<f64> = array![];
    let result = square_simd(&x.view());
    assert_eq!(result.len(), 0);
}

/// Test square large array (SIMD path)
#[test]
fn test_square_simd_large_array() {
    let n = 10000;
    let x = Array1::from_vec((0..n).map(|i| (i as f64 - 5000.0) * 0.01).collect());

    let result = square_simd(&x.view());
    assert_eq!(result.len(), n);

    // Check a few values
    for i in [0, 100, 5000, 9999] {
        let expected = x[i] * x[i];
        assert!(
            (result[i] - expected).abs() < 1e-10,
            "square[{}] = {}, expected {}",
            i,
            result[i],
            expected
        );
    }
}

/// Test square special values
#[test]
fn test_square_simd_special_values() {
    let x = array![0.0_f64, -0.0, f64::INFINITY, f64::NEG_INFINITY];

    let result = square_simd(&x.view());

    // 0² = 0
    assert_eq!(result[0], 0.0);
    // (-0)² = 0
    assert_eq!(result[1], 0.0);
    // ∞² = ∞
    assert!(result[2].is_infinite() && result[2] > 0.0);
    // (-∞)² = ∞
    assert!(result[3].is_infinite() && result[3] > 0.0);
}

/// Test square always non-negative
#[test]
fn test_square_simd_non_negative() {
    let x = array![-10.0_f64, -5.0, -1.0, -0.1, 0.0, 0.1, 1.0, 5.0, 10.0];

    let result = square_simd(&x.view());

    for (i, &v) in result.iter().enumerate() {
        assert!(
            v >= 0.0,
            "square should always be non-negative, got {} at index {}",
            v,
            i
        );
    }
}

/// Test square vs pow comparison
#[test]
fn test_square_simd_vs_pow() {
    let x = array![1.5_f64, 2.5, 3.5, -4.5, 0.5];

    let result = square_simd(&x.view());
    let pow_result = x.mapv(|v| v.powf(2.0));

    for i in 0..x.len() {
        assert!(
            (result[i] - pow_result[i]).abs() < 1e-14,
            "square should match pow(x, 2)"
        );
    }
}

/// Test square MSE use case
#[test]
fn test_square_simd_mse_use_case() {
    // Computing Mean Squared Error: sum((y - y_hat)²) / n
    let y_true = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let y_pred = array![1.1_f64, 2.2, 2.9, 4.1, 4.9];

    let errors = &y_true - &y_pred;
    let squared_errors = square_simd(&errors.view());
    let mse = squared_errors.sum() / y_true.len() as f64;

    // Manual calculation: (0.1² + 0.2² + 0.1² + 0.1² + 0.1²) / 5 = 0.08 / 5 = 0.016
    let expected_mse = (0.01 + 0.04 + 0.01 + 0.01 + 0.01) / 5.0;
    assert!((mse - expected_mse).abs() < 1e-10, "MSE calculation failed");
}

// ============================================================================
// Rsqrt tests
// ============================================================================

/// Test rsqrt basic f32 correctness
#[test]
fn test_rsqrt_simd_f32_basic() {
    let x = array![1.0_f32, 4.0, 9.0, 16.0, 25.0];

    let result = rsqrt_simd(&x.view());

    assert!((result[0] - 1.0).abs() < 1e-6); // 1/sqrt(1) = 1
    assert!((result[1] - 0.5).abs() < 1e-6); // 1/sqrt(4) = 0.5
    assert!((result[2] - 1.0 / 3.0).abs() < 1e-6); // 1/sqrt(9) = 1/3
    assert!((result[3] - 0.25).abs() < 1e-6); // 1/sqrt(16) = 0.25
    assert!((result[4] - 0.2).abs() < 1e-6); // 1/sqrt(25) = 0.2
}

/// Test rsqrt basic f64 correctness
#[test]
fn test_rsqrt_simd_f64_basic() {
    let x = array![1.0_f64, 4.0, 9.0, 16.0, 25.0];

    let result = rsqrt_simd(&x.view());

    assert!((result[0] - 1.0).abs() < 1e-14); // 1/sqrt(1) = 1
    assert!((result[1] - 0.5).abs() < 1e-14); // 1/sqrt(4) = 0.5
    assert!((result[2] - 1.0 / 3.0).abs() < 1e-14); // 1/sqrt(9) = 1/3
    assert!((result[3] - 0.25).abs() < 1e-14); // 1/sqrt(16) = 0.25
    assert!((result[4] - 0.2).abs() < 1e-14); // 1/sqrt(25) = 0.2
}

/// Test rsqrt empty array
#[test]
fn test_rsqrt_simd_empty() {
    let x: Array1<f64> = array![];
    let result = rsqrt_simd(&x.view());
    assert_eq!(result.len(), 0);
}

/// Test rsqrt large array (SIMD path)
#[test]
fn test_rsqrt_simd_large_array() {
    let n = 10000;
    let x = Array1::from_vec((1..=n).map(|i| i as f64).collect());

    let result = rsqrt_simd(&x.view());
    assert_eq!(result.len(), n);

    // Check a few values
    for i in [0, 99, 999, 9999] {
        let expected = 1.0 / x[i].sqrt();
        assert!(
            (result[i] - expected).abs() < 1e-10,
            "rsqrt[{}] = {}, expected {}",
            i,
            result[i],
            expected
        );
    }
}

/// Test rsqrt special values
#[test]
fn test_rsqrt_simd_special_values() {
    let x = array![0.0_f64, -1.0, f64::INFINITY];

    let result = rsqrt_simd(&x.view());

    // 1/sqrt(0) = ∞
    assert!(result[0].is_infinite() && result[0] > 0.0);
    // 1/sqrt(-1) = NaN
    assert!(result[1].is_nan());
    // 1/sqrt(∞) = 0
    assert_eq!(result[2], 0.0);
}

/// Test rsqrt identity: x * rsqrt(x)² = 1
#[test]
fn test_rsqrt_simd_identity() {
    let x = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];

    let rsqrt_x = rsqrt_simd(&x.view());

    for i in 0..x.len() {
        let product = x[i] * rsqrt_x[i] * rsqrt_x[i];
        assert!(
            (product - 1.0).abs() < 1e-14,
            "x * rsqrt(x)² should be 1, got {}",
            product
        );
    }
}

/// Test rsqrt vs 1/sqrt comparison
#[test]
fn test_rsqrt_simd_vs_one_over_sqrt() {
    let x = array![0.5_f64, 1.5, 2.5, 3.5, 4.5];

    let result = rsqrt_simd(&x.view());
    let manual = x.mapv(|v| 1.0 / v.sqrt());

    for i in 0..x.len() {
        assert!(
            (result[i] - manual[i]).abs() < 1e-14,
            "rsqrt should match 1/sqrt"
        );
    }
}

/// Test rsqrt vector normalization use case
#[test]
fn test_rsqrt_simd_vector_normalization() {
    // Vector normalization: v_normalized = v * rsqrt(dot(v, v))
    // This is the primary use case in graphics

    // 3D vectors
    let vx = array![3.0_f64, 1.0, 0.0];
    let vy = array![4.0_f64, 1.0, 1.0];
    let vz = array![0.0_f64, 1.0, 0.0];

    // Compute squared magnitudes
    let mag_squared = &vx * &vx + &vy * &vy + &vz * &vz;
    // Get inverse magnitudes
    let inv_mag = rsqrt_simd(&mag_squared.view());

    // Normalize
    let nx = &vx * &inv_mag;
    let ny = &vy * &inv_mag;
    let nz = &vz * &inv_mag;

    // Check normalized vector magnitudes are 1
    for i in 0..vx.len() {
        let norm = (nx[i] * nx[i] + ny[i] * ny[i] + nz[i] * nz[i]).sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-14,
            "Normalized vector magnitude should be 1, got {}",
            norm
        );
    }
}

/// Test rsqrt quaternion normalization use case
#[test]
fn test_rsqrt_simd_quaternion_normalization() {
    // Quaternion normalization: q / |q|
    // Given q = (w, x, y, z), |q|² = w² + x² + y² + z²

    let w = array![1.0_f64, 0.5, 0.707];
    let x = array![0.0_f64, 0.5, 0.0];
    let y = array![0.0_f64, 0.5, 0.707];
    let z = array![0.0_f64, 0.5, 0.0];

    let mag_squared = &w * &w + &x * &x + &y * &y + &z * &z;
    let inv_mag = rsqrt_simd(&mag_squared.view());

    let nw = &w * &inv_mag;
    let nx = &x * &inv_mag;
    let ny = &y * &inv_mag;
    let nz = &z * &inv_mag;

    // Check normalized quaternion magnitudes are 1
    for i in 0..w.len() {
        let norm_sq = nw[i] * nw[i] + nx[i] * nx[i] + ny[i] * ny[i] + nz[i] * nz[i];
        assert!(
            (norm_sq - 1.0).abs() < 1e-14,
            "Normalized quaternion magnitude² should be 1, got {}",
            norm_sq
        );
    }
}

// ============================================================================
// Sincos tests
// ============================================================================

/// Test sincos basic f32 correctness
#[test]
fn test_sincos_simd_f32_basic() {
    let x = array![
        0.0_f32,
        PI as f32 / 6.0,
        PI as f32 / 4.0,
        PI as f32 / 2.0,
        PI as f32
    ];

    let (sin_result, cos_result) = sincos_simd(&x.view());

    // sin(0) = 0, cos(0) = 1
    assert!(sin_result[0].abs() < 1e-6);
    assert!((cos_result[0] - 1.0).abs() < 1e-6);
    // sin(π/6) = 0.5, cos(π/6) = √3/2
    assert!((sin_result[1] - 0.5).abs() < 1e-5);
    // sin(π/4) = cos(π/4) = √2/2
    let sqrt2_2 = std::f32::consts::FRAC_1_SQRT_2;
    assert!((sin_result[2] - sqrt2_2).abs() < 1e-5);
    assert!((cos_result[2] - sqrt2_2).abs() < 1e-5);
    // sin(π/2) = 1, cos(π/2) = 0
    assert!((sin_result[3] - 1.0).abs() < 1e-5);
    assert!(cos_result[3].abs() < 1e-5);
}

/// Test sincos basic f64 correctness
#[test]
fn test_sincos_simd_f64_basic() {
    let x = array![0.0_f64, PI / 6.0, PI / 4.0, PI / 2.0, PI];

    let (sin_result, cos_result) = sincos_simd(&x.view());

    // sin(0) = 0, cos(0) = 1
    assert!(sin_result[0].abs() < 1e-14);
    assert!((cos_result[0] - 1.0).abs() < 1e-14);
    // sin(π/6) = 0.5
    assert!((sin_result[1] - 0.5).abs() < 1e-14);
    // sin(π/4) = cos(π/4) = √2/2
    let sqrt2_2 = std::f64::consts::FRAC_1_SQRT_2;
    assert!((sin_result[2] - sqrt2_2).abs() < 1e-14);
    assert!((cos_result[2] - sqrt2_2).abs() < 1e-14);
    // sin(π/2) = 1, cos(π/2) ≈ 0
    assert!((sin_result[3] - 1.0).abs() < 1e-14);
    assert!(cos_result[3].abs() < 1e-14);
    // sin(π) ≈ 0, cos(π) = -1
    assert!(sin_result[4].abs() < 1e-14);
    assert!((cos_result[4] + 1.0).abs() < 1e-14);
}

/// Test sincos empty array
#[test]
fn test_sincos_simd_empty() {
    let x: Array1<f64> = array![];
    let (sin_result, cos_result) = sincos_simd(&x.view());
    assert_eq!(sin_result.len(), 0);
    assert_eq!(cos_result.len(), 0);
}

/// Test sincos large array (SIMD path)
#[test]
fn test_sincos_simd_large_array() {
    let n = 10000;
    let x = Array1::from_vec((0..n).map(|i| i as f64 * 0.01).collect());

    let (sin_result, cos_result) = sincos_simd(&x.view());
    assert_eq!(sin_result.len(), n);
    assert_eq!(cos_result.len(), n);

    // Check a few values
    for i in [0, 100, 1000, 5000, 9999] {
        let expected_sin = x[i].sin();
        let expected_cos = x[i].cos();
        assert!(
            (sin_result[i] - expected_sin).abs() < 1e-10,
            "sin[{}] = {}, expected {}",
            i,
            sin_result[i],
            expected_sin
        );
        assert!(
            (cos_result[i] - expected_cos).abs() < 1e-10,
            "cos[{}] = {}, expected {}",
            i,
            cos_result[i],
            expected_cos
        );
    }
}

/// Test sincos Pythagorean identity: sin²(x) + cos²(x) = 1
#[test]
fn test_sincos_simd_pythagorean_identity() {
    let x = array![0.0_f64, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, PI, 2.0 * PI];

    let (sin_result, cos_result) = sincos_simd(&x.view());

    for i in 0..x.len() {
        let sum_sq = sin_result[i] * sin_result[i] + cos_result[i] * cos_result[i];
        assert!(
            (sum_sq - 1.0).abs() < 1e-14,
            "sin²(x) + cos²(x) should be 1, got {} at x={}",
            sum_sq,
            x[i]
        );
    }
}

/// Test sincos consistency with separate sin/cos
#[test]
fn test_sincos_simd_consistency() {
    use scirs2_core::ndarray_ext::elementwise::{cos_simd, sin_simd};

    let x = array![0.1_f64, 0.5, 1.0, 2.0, 3.0];

    let (sin_result, cos_result) = sincos_simd(&x.view());
    let sin_separate = sin_simd(&x.view());
    let cos_separate = cos_simd(&x.view());

    for i in 0..x.len() {
        assert!(
            (sin_result[i] - sin_separate[i]).abs() < 1e-14,
            "sincos sin should match sin_simd"
        );
        assert!(
            (cos_result[i] - cos_separate[i]).abs() < 1e-14,
            "sincos cos should match cos_simd"
        );
    }
}

/// Test sincos rotation matrix use case
#[test]
fn test_sincos_simd_rotation_matrix() {
    // 2D rotation matrix: [[cos(θ), -sin(θ)], [sin(θ), cos(θ)]]
    // Rotating point (1, 0) by various angles

    let angles = array![0.0_f64, PI / 4.0, PI / 2.0, PI, 3.0 * PI / 2.0];
    let (sin_theta, cos_theta) = sincos_simd(&angles.view());

    // Point (1, 0) rotated by θ should be at (cos(θ), sin(θ))
    let px = 1.0_f64;
    let py = 0.0_f64;

    for i in 0..angles.len() {
        let rx = cos_theta[i] * px - sin_theta[i] * py;
        let ry = sin_theta[i] * px + cos_theta[i] * py;

        // After rotation, point should be at (cos(θ), sin(θ))
        assert!(
            (rx - cos_theta[i]).abs() < 1e-14,
            "Rotated x should be cos(θ)"
        );
        assert!(
            (ry - sin_theta[i]).abs() < 1e-14,
            "Rotated y should be sin(θ)"
        );

        // Distance from origin should remain 1
        let dist = (rx * rx + ry * ry).sqrt();
        assert!(
            (dist - 1.0).abs() < 1e-14,
            "Rotation should preserve distance from origin"
        );
    }
}

/// Test sincos complex exponential: e^(iθ) = cos(θ) + i*sin(θ)
#[test]
fn test_sincos_simd_euler_formula() {
    // Euler's formula: e^(iθ) = cos(θ) + i*sin(θ)
    // |e^(iθ)| = sqrt(cos²(θ) + sin²(θ)) = 1

    let theta = array![0.0_f64, PI / 4.0, PI / 2.0, PI, 2.0 * PI];
    let (sin_theta, cos_theta) = sincos_simd(&theta.view());

    for i in 0..theta.len() {
        // Magnitude of complex exponential
        let magnitude = (cos_theta[i] * cos_theta[i] + sin_theta[i] * sin_theta[i]).sqrt();
        assert!(
            (magnitude - 1.0).abs() < 1e-14,
            "Complex exponential magnitude should be 1"
        );
    }
}

/// Test sincos wave simulation use case
#[test]
fn test_sincos_simd_wave_simulation() {
    // Simulating a wave: y(t) = A * sin(ωt + φ)
    // Velocity: v(t) = A * ω * cos(ωt + φ)
    // Both need sin and cos

    let amplitude = 2.0_f64;
    let omega = 3.0_f64;
    let phase = PI / 4.0;

    let t = array![0.0_f64, 0.1, 0.2, 0.3, 0.4, 0.5];
    let omega_t_plus_phi = t.mapv(|ti| omega * ti + phase);

    let (sin_result, cos_result) = sincos_simd(&omega_t_plus_phi.view());

    let position = sin_result.mapv(|s| amplitude * s);
    let velocity = cos_result.mapv(|c| amplitude * omega * c);

    // At t=0: position = A*sin(φ), velocity = A*ω*cos(φ)
    let expected_pos_0 = amplitude * (omega * 0.0 + phase).sin();
    let expected_vel_0 = amplitude * omega * (omega * 0.0 + phase).cos();

    assert!((position[0] - expected_pos_0).abs() < 1e-14);
    assert!((velocity[0] - expected_vel_0).abs() < 1e-14);
}

/// Test sincos Fourier transform building block
#[test]
fn test_sincos_simd_fourier_basis() {
    // DFT basis functions: e^(-2πi*k*n/N) = cos(2πkn/N) - i*sin(2πkn/N)
    // For N=8, k=1, compute basis function at each n

    let n = 8;
    let k = 1.0_f64;
    let two_pi_k_over_n = 2.0 * PI * k / (n as f64);

    let indices: Array1<f64> = Array1::from_vec((0..n).map(|i| i as f64).collect());
    let angles = indices.mapv(|i| two_pi_k_over_n * i);

    let (sin_result, cos_result) = sincos_simd(&angles.view());

    // Verify orthogonality property: sum over n should be 0 for k != 0
    let real_sum: f64 = cos_result.iter().sum();
    let imag_sum: f64 = sin_result.iter().sum();

    assert!(
        real_sum.abs() < 1e-14,
        "Sum of cos(2πk*n/N) over n should be 0 for k != 0"
    );
    assert!(
        imag_sum.abs() < 1e-14,
        "Sum of sin(2πk*n/N) over n should be 0 for k != 0"
    );
}
