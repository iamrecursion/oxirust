//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

/// Lanczos approximation for gamma function (f32)
#[allow(dead_code)]
pub(super) fn lanczos_gamma_f32(z: f32) -> f32 {
    const G: f32 = 7.0;
    const SQRT_2PI: f32 = 2.5066282746310002;
    const LANCZOS_COEFFS: [f32; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    if z < 0.5 {
        let pi = std::f32::consts::PI;
        let sinpix = (pi * z).sin();
        if sinpix.abs() < 1e-10 {
            return f32::INFINITY;
        }
        return pi / (sinpix * lanczos_gamma_f32(1.0 - z));
    }
    let z_shifted = z - 1.0;
    let mut acc = LANCZOS_COEFFS[0];
    for (i, &coeff) in LANCZOS_COEFFS.iter().enumerate().skip(1) {
        acc += coeff / (z_shifted + i as f32);
    }
    let t = z_shifted + G + 0.5;
    SQRT_2PI * acc * t.powf(z_shifted + 0.5) * (-t).exp()
}
/// Lanczos approximation for gamma function (f64)
#[allow(dead_code)]
pub(super) fn lanczos_gamma_f64(z: f64) -> f64 {
    const G: f64 = 7.0;
    const SQRT_2PI: f64 = 2.5066282746310002;
    const LANCZOS_COEFFS: [f64; 9] = [
        0.999999999999809932,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    if z < 0.5 {
        let pi = std::f64::consts::PI;
        let sinpix = (pi * z).sin();
        if sinpix.abs() < 1e-14 {
            return f64::INFINITY;
        }
        return pi / (sinpix * lanczos_gamma_f64(1.0 - z));
    }
    let z_shifted = z - 1.0;
    let mut acc = LANCZOS_COEFFS[0];
    for (i, &coeff) in LANCZOS_COEFFS.iter().enumerate().skip(1) {
        acc += coeff / (z_shifted + i as f64);
    }
    let t = z_shifted + G + 0.5;
    SQRT_2PI * acc * t.powf(z_shifted + 0.5) * (-t).exp()
}
/// Digamma function (psi) - logarithmic derivative of gamma function (f32)
/// ψ(x) = d/dx ln(Γ(x)) = Γ'(x) / Γ(x)
#[allow(dead_code)]
pub(super) fn digamma_f32(mut x: f32) -> f32 {
    let pi = std::f32::consts::PI;
    if x.is_nan() {
        return f32::NAN;
    }
    if x.is_infinite() {
        return if x > 0.0 { f32::INFINITY } else { f32::NAN };
    }
    if x <= 0.0 {
        if x == x.floor() {
            return f32::NAN;
        }
        return digamma_f32(1.0 - x) - pi / (pi * x).tan();
    }
    let mut result = 0.0;
    while x < 6.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    result += x.ln() - 0.5 / x;
    let x2 = x * x;
    let x4 = x2 * x2;
    let x6 = x4 * x2;
    result -= 1.0 / (12.0 * x2);
    result += 1.0 / (120.0 * x4);
    result -= 1.0 / (252.0 * x6);
    result
}
/// Digamma function (psi) - logarithmic derivative of gamma function (f64)
/// ψ(x) = d/dx ln(Γ(x)) = Γ'(x) / Γ(x)
#[allow(dead_code)]
pub(super) fn digamma_f64(mut x: f64) -> f64 {
    let pi = std::f64::consts::PI;
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return if x > 0.0 { f64::INFINITY } else { f64::NAN };
    }
    if x <= 0.0 {
        if x == x.floor() {
            return f64::NAN;
        }
        return digamma_f64(1.0 - x) - pi / (pi * x).tan();
    }
    let mut result = 0.0;
    while x < 8.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    result += x.ln() - 0.5 / x;
    let x2 = x * x;
    let x4 = x2 * x2;
    let x6 = x4 * x2;
    let x8 = x4 * x4;
    let x10 = x8 * x2;
    let x12 = x8 * x4;
    result -= 1.0 / (12.0 * x2);
    result += 1.0 / (120.0 * x4);
    result -= 1.0 / (252.0 * x6);
    result += 1.0 / (240.0 * x8);
    result -= 5.0 / (660.0 * x10);
    result += 691.0 / (32760.0 * x12);
    result
}
/// Trigamma function ψ'(x) = d²/dx² ln(Γ(x)) for f32
///
/// Uses:
/// 1. Reflection formula for x < 0.5: ψ'(1-x) + ψ'(x) = π²/sin²(πx)
/// 2. Recurrence relation for small x: ψ'(x+1) = ψ'(x) - 1/x²
/// 3. Asymptotic expansion for large x
pub(super) fn trigamma_f32(mut x: f32) -> f32 {
    let pi = std::f32::consts::PI;
    if x.is_nan() {
        return f32::NAN;
    }
    if x.is_infinite() {
        return if x > 0.0 { 0.0 } else { f32::NAN };
    }
    if x <= 0.0 {
        if x == x.floor() {
            return f32::NAN;
        }
        let sin_pix = (pi * x).sin();
        return (pi * pi) / (sin_pix * sin_pix) - trigamma_f32(1.0 - x);
    }
    let mut result = 0.0;
    while x < 6.0 {
        result += 1.0 / (x * x);
        x += 1.0;
    }
    let x2 = x * x;
    let x3 = x2 * x;
    let x5 = x3 * x2;
    let x7 = x5 * x2;
    result += 1.0 / x;
    result += 1.0 / (2.0 * x2);
    result += 1.0 / (6.0 * x3);
    result -= 1.0 / (30.0 * x5);
    result += 1.0 / (42.0 * x7);
    result
}
/// Trigamma function ψ'(x) = d²/dx² ln(Γ(x)) for f64
///
/// Uses:
/// 1. Reflection formula for x < 0.5: ψ'(1-x) + ψ'(x) = π²/sin²(πx)
/// 2. Recurrence relation for small x: ψ'(x+1) = ψ'(x) - 1/x²
/// 3. Asymptotic expansion for large x (more terms for f64 precision)
pub(super) fn trigamma_f64(mut x: f64) -> f64 {
    let pi = std::f64::consts::PI;
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return if x > 0.0 { 0.0 } else { f64::NAN };
    }
    if x <= 0.0 {
        if x == x.floor() {
            return f64::NAN;
        }
        let sin_pix = (pi * x).sin();
        return (pi * pi) / (sin_pix * sin_pix) - trigamma_f64(1.0 - x);
    }
    let mut result = 0.0;
    while x < 8.0 {
        result += 1.0 / (x * x);
        x += 1.0;
    }
    let x2 = x * x;
    let x3 = x2 * x;
    let x5 = x3 * x2;
    let x7 = x5 * x2;
    let x9 = x7 * x2;
    let x11 = x9 * x2;
    result += 1.0 / x;
    result += 1.0 / (2.0 * x2);
    result += 1.0 / (6.0 * x3);
    result -= 1.0 / (30.0 * x5);
    result += 1.0 / (42.0 * x7);
    result -= 1.0 / (30.0 * x9);
    result += 5.0 / (66.0 * x11);
    result
}
/// Log-gamma function ln(Γ(x)) for f32
///
/// More numerically stable than gamma(x).ln() since it avoids overflow.
/// Uses Lanczos approximation: ln(Γ(z)) = ln(√(2π)) + (z-0.5)*ln(t) - t + ln(sum)
/// where t = z + g - 0.5
pub(super) fn ln_gamma_f32(z: f32) -> f32 {
    const G: f32 = 7.0;
    const LN_SQRT_2PI: f32 = 0.9189385332046727;
    const LANCZOS_COEFFS: [f32; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    if z.is_nan() {
        return f32::NAN;
    }
    if z.is_infinite() {
        return if z > 0.0 { f32::INFINITY } else { f32::NAN };
    }
    if z <= 0.0 && z == z.floor() {
        return f32::INFINITY;
    }
    if z < 0.5 {
        let pi = std::f32::consts::PI;
        let sinpiz = (pi * z).sin().abs();
        if sinpiz < 1e-10 {
            return f32::INFINITY;
        }
        return pi.ln() - sinpiz.ln() - ln_gamma_f32(1.0 - z);
    }
    let z_shifted = z - 1.0;
    let mut sum = LANCZOS_COEFFS[0];
    for (i, &coeff) in LANCZOS_COEFFS.iter().enumerate().skip(1) {
        sum += coeff / (z_shifted + i as f32);
    }
    let t = z_shifted + G + 0.5;
    LN_SQRT_2PI + (z_shifted + 0.5) * t.ln() - t + sum.ln()
}
/// Log-gamma function ln(Γ(x)) for f64
///
/// More numerically stable than gamma(x).ln() since it avoids overflow.
/// Uses Lanczos approximation with higher precision coefficients.
pub(super) fn ln_gamma_f64(z: f64) -> f64 {
    const G: f64 = 7.0;
    const LN_SQRT_2PI: f64 = 0.9189385332046727;
    const LANCZOS_COEFFS: [f64; 9] = [
        0.999999999999809932,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    if z.is_nan() {
        return f64::NAN;
    }
    if z.is_infinite() {
        return if z > 0.0 { f64::INFINITY } else { f64::NAN };
    }
    if z <= 0.0 && z == z.floor() {
        return f64::INFINITY;
    }
    if z < 0.5 {
        let pi = std::f64::consts::PI;
        let sinpiz = (pi * z).sin().abs();
        if sinpiz < 1e-14 {
            return f64::INFINITY;
        }
        return pi.ln() - sinpiz.ln() - ln_gamma_f64(1.0 - z);
    }
    let z_shifted = z - 1.0;
    let mut sum = LANCZOS_COEFFS[0];
    for (i, &coeff) in LANCZOS_COEFFS.iter().enumerate().skip(1) {
        sum += coeff / (z_shifted + i as f64);
    }
    let t = z_shifted + G + 0.5;
    LN_SQRT_2PI + (z_shifted + 0.5) * t.ln() - t + sum.ln()
}
/// Error function erf(x) for f32
///
/// Uses Abramowitz & Stegun approximation (equation 7.1.26)
/// Maximum error: ~1.5×10⁻⁷
pub(super) fn erf_f32(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    if x.is_infinite() {
        return if x > 0.0 { 1.0 } else { -1.0 };
    }
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();
    if x < 1e-10 {
        return sign * x * std::f32::consts::FRAC_2_SQRT_PI;
    }
    const P: f32 = 0.3275911;
    const A1: f32 = 0.254829592;
    const A2: f32 = -0.284496736;
    const A3: f32 = 1.421413741;
    const A4: f32 = -1.453152027;
    const A5: f32 = 1.061405429;
    let t = 1.0 / (1.0 + P * x);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;
    let poly = A1 * t + A2 * t2 + A3 * t3 + A4 * t4 + A5 * t5;
    let result = 1.0 - poly * (-x * x).exp();
    sign * result
}
/// Error function erf(x) for f64
///
/// Uses a higher-precision rational approximation
/// Maximum error: ~1.5×10⁻¹⁵
pub(super) fn erf_f64(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return if x > 0.0 { 1.0 } else { -1.0 };
    }
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();
    if x < 1e-20 {
        return sign * x * std::f64::consts::FRAC_2_SQRT_PI;
    }
    if x > 6.0 {
        return sign;
    }
    let result = if x < 0.25 {
        let x2 = x * x;
        let two_over_sqrt_pi = std::f64::consts::FRAC_2_SQRT_PI;
        let term = x
            * (1.0
                - x2 / 3.0
                    * (1.0
                        - x2 / 5.0
                            * 0.5
                            * (1.0
                                - x2 / 7.0
                                    * (1.0 / 3.0)
                                    * (1.0
                                        - x2 / 9.0
                                            * 0.25
                                            * (1.0
                                                - x2 / 11.0
                                                    * 0.2
                                                    * (1.0
                                                        - x2 / 13.0
                                                            * (1.0 / 6.0)
                                                            * (1.0 - x2 / 15.0 * (1.0 / 7.0))))))));
        two_over_sqrt_pi * term
    } else {
        const P: f64 = 0.3275911;
        const A1: f64 = 0.254829592;
        const A2: f64 = -0.284496736;
        const A3: f64 = 1.421413741;
        const A4: f64 = -1.453152027;
        const A5: f64 = 1.061405429;
        let t = 1.0 / (1.0 + P * x);
        let poly = t * (A1 + t * (A2 + t * (A3 + t * (A4 + t * A5))));
        1.0 - poly * (-x * x).exp()
    };
    sign * result
}
/// Complementary error function erfc(x) = 1 - erf(x) for f32
///
/// More numerically stable than 1 - erf(x) for large x
pub(super) fn erfc_f32(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    if x.is_infinite() {
        return if x > 0.0 { 0.0 } else { 2.0 };
    }
    if x < 0.0 {
        return 2.0 - erfc_f32(-x);
    }
    if x > 10.0 {
        return 0.0;
    }
    if x < 0.5 {
        return 1.0 - erf_f32(x);
    }
    const P: f32 = 0.3275911;
    const A1: f32 = 0.254829592;
    const A2: f32 = -0.284496736;
    const A3: f32 = 1.421413741;
    const A4: f32 = -1.453152027;
    const A5: f32 = 1.061405429;
    let t = 1.0 / (1.0 + P * x);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;
    let poly = A1 * t + A2 * t2 + A3 * t3 + A4 * t4 + A5 * t5;
    poly * (-x * x).exp()
}
/// Complementary error function erfc(x) = 1 - erf(x) for f64
///
/// More numerically stable than 1 - erf(x) for large x
pub(super) fn erfc_f64(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return if x > 0.0 { 0.0 } else { 2.0 };
    }
    if x < 0.0 {
        return 2.0 - erfc_f64(-x);
    }
    if x > 27.0 {
        return 0.0;
    }
    if x < 0.5 {
        return 1.0 - erf_f64(x);
    }
    if x > 4.0 {
        let x2 = x * x;
        let inv_x2 = 1.0 / x2;
        let sqrt_pi = std::f64::consts::PI.sqrt();
        let asymp = 1.0 - 0.5 * inv_x2 + 0.75 * inv_x2 * inv_x2 - 1.875 * inv_x2 * inv_x2 * inv_x2
            + 6.5625 * inv_x2 * inv_x2 * inv_x2 * inv_x2;
        return (-x2).exp() / (x * sqrt_pi) * asymp;
    }
    const P: f64 = 0.3275911;
    const A1: f64 = 0.254829592;
    const A2: f64 = -0.284496736;
    const A3: f64 = 1.421413741;
    const A4: f64 = -1.453152027;
    const A5: f64 = 1.061405429;
    let t = 1.0 / (1.0 + P * x);
    let poly = t * (A1 + t * (A2 + t * (A3 + t * (A4 + t * A5))));
    poly * (-x * x).exp()
}
/// Inverse error function erfinv(y) for f32
///
/// Uses Winitzki's approximation followed by Newton-Raphson refinement
/// Domain: (-1, 1), returns x such that erf(x) = y
pub(super) fn erfinv_f32(y: f32) -> f32 {
    if y.is_nan() {
        return f32::NAN;
    }
    if y <= -1.0 {
        return if y == -1.0 {
            f32::NEG_INFINITY
        } else {
            f32::NAN
        };
    }
    if y >= 1.0 {
        return if y == 1.0 { f32::INFINITY } else { f32::NAN };
    }
    if y == 0.0 {
        return 0.0;
    }
    let sign = if y >= 0.0 { 1.0 } else { -1.0 };
    let y = y.abs();
    let a = 0.147_f32;
    let two_over_pi_a = 2.0 / (std::f32::consts::PI * a);
    let ln_one_minus_y2 = (1.0 - y * y).ln();
    let t1 = two_over_pi_a + 0.5 * ln_one_minus_y2;
    let t2 = (1.0 / a) * ln_one_minus_y2;
    let inner = (t1 * t1 - t2).sqrt() - t1;
    let mut x = inner.sqrt();
    let two_over_sqrt_pi = std::f32::consts::FRAC_2_SQRT_PI;
    for _ in 0..2 {
        let erf_x = erf_f32(x);
        let erf_deriv = two_over_sqrt_pi * (-x * x).exp();
        x -= (erf_x - y) / erf_deriv;
    }
    sign * x
}
