//! Local implementations of the special functions Dirichlet / Categorical VMP
//! depend on (log-gamma and digamma).
//!
//! `scirs2-core` does not re-export `digamma` / `gammaln` (those live in the
//! heavier `scirs2-special` crate). Because this crate only pulls in `scirs2-core`,
//! we ship small, deterministic, well-tested pure-Rust implementations here. These
//! are accurate to ~1e-10 for all arguments VMP legitimately sees (α > 0, typically
//! α ≥ 0.5) and are more than sufficient for a research preview.
//!
//! References:
//! - Lanczos (1964) — approximation for the gamma function.
//! - Bernardo (1976), "Algorithm AS 103: Psi (digamma) function".

/// Natural logarithm of the Gamma function, `ln Γ(x)`.
///
/// Uses the Lanczos approximation with the coefficients from Press et al.,
/// *Numerical Recipes in C* §6.1. Accurate to ~1e-12 on `x > 0`.
///
/// # Panics
///
/// Never panics. Returns `f64::INFINITY` for `x <= 0` (Γ has poles there); VMP
/// never evaluates in that region for a well-posed model.
pub fn ln_gamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }
    // Lanczos g=7, n=9 coefficients.
    const G: f64 = 7.0;
    const COEF: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];

    // Reflection for x < 0.5 — not strictly needed here (we already reject x<=0)
    // but harmless and keeps the function robust.
    if x < 0.5 {
        // Reflection formula: ln Γ(x) = ln(π / sin(π x)) − ln Γ(1 − x)
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - ln_gamma(1.0 - x);
    }

    let x = x - 1.0;
    let mut a = COEF[0];
    for (i, &c) in COEF.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    let t = x + G + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Digamma function ψ(x) = d/dx ln Γ(x).
///
/// Implementation uses the recurrence ψ(x+1) = ψ(x) + 1/x to shift the argument
/// above 6.0, then an asymptotic series. Accurate to better than 1e-10 for all
/// x > 0 we realistically encounter (α ≥ 1e-3).
///
/// # Panics
///
/// Never panics. Returns `f64::NEG_INFINITY` for x ≤ 0 (ψ has poles at
/// non-positive integers); VMP never evaluates there for well-posed Dirichlets.
pub fn digamma(mut x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NEG_INFINITY;
    }

    let mut result = 0.0;
    // Shift x up until it is large enough for the asymptotic expansion.
    while x < 6.0 {
        result -= 1.0 / x;
        x += 1.0;
    }

    // Asymptotic expansion:
    //   ψ(x) ≈ ln x − 1/(2x) − 1/(12 x²) + 1/(120 x⁴) − 1/(252 x⁶) + …
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    result += x.ln()
        - 0.5 * inv
        - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 * (1.0 / 252.0 - inv2 / 240.0)));
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ln_gamma_matches_known_values() {
        // ln Γ(1) = 0, ln Γ(2) = 0, ln Γ(3) = ln(2) ≈ 0.693147
        assert!((ln_gamma(1.0) - 0.0).abs() < 1e-10);
        assert!((ln_gamma(2.0) - 0.0).abs() < 1e-10);
        assert!((ln_gamma(3.0) - 2.0f64.ln()).abs() < 1e-10);
        // ln Γ(0.5) = 0.5 * ln π
        assert!((ln_gamma(0.5) - 0.5 * std::f64::consts::PI.ln()).abs() < 1e-10);
        // ln Γ(10) = ln(9!) = ln(362880)
        assert!((ln_gamma(10.0) - 362_880.0f64.ln()).abs() < 1e-8);
    }

    #[test]
    fn digamma_matches_known_values() {
        // ψ(1) = -γ ≈ -0.5772156649
        let euler_mascheroni = 0.577_215_664_901_532_9;
        assert!((digamma(1.0) + euler_mascheroni).abs() < 1e-9);
        // ψ(2) = 1 - γ
        assert!((digamma(2.0) - (1.0 - euler_mascheroni)).abs() < 1e-9);
        // ψ(0.5) = -γ - 2 ln 2
        let expected = -euler_mascheroni - 2.0 * 2.0f64.ln();
        assert!((digamma(0.5) - expected).abs() < 1e-9);
    }

    #[test]
    fn digamma_recurrence() {
        // ψ(x + 1) = ψ(x) + 1/x for arbitrary x > 0
        for &x in &[0.3, 1.7, 3.2, 10.5] {
            let lhs = digamma(x + 1.0);
            let rhs = digamma(x) + 1.0 / x;
            assert!((lhs - rhs).abs() < 1e-9, "x = {}", x);
        }
    }
}
