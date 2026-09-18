//! Chebyshev series evaluation (value and derivative) via the Clenshaw
//! recurrence.
//!
//! References:
//! * C. W. Clenshaw, "A note on the summation of Chebyshev series",
//!   Math. Tables Aids Comput. 9 (1955), 118 — the backward recurrence
//!   `b_k = 2 x b_{k+1} - b_{k+2} + a_k`, `S = a_0 + x b_1 - b_2`.
//! * DLMF 18.9.21 / Abramowitz & Stegun 22.8: `dT_n/dx = n U_{n-1}(x)`,
//!   so the derivative series is a Chebyshev-U series summed with the same
//!   Clenshaw recurrence (`U` obeys `U_{n+1} = 2 x U_n - U_{n-1}`, and with
//!   `U_0 = 1`, `U_1 = 2x` the Clenshaw sum collapses to `S = b_0`).
//! * The equivalent forward recurrences appear in the public-domain JPL
//!   `testeph.f` subroutine `INTERP` (arrays `PC`/`VC`).

/// Sum of the Chebyshev-T series `sum_k coeffs[k] * T_k(x)` (Clenshaw).
///
/// `x` is the normalized argument in `[-1, 1]`; an empty `coeffs` yields 0.
#[must_use]
pub fn chebyshev_value(coeffs: &[f64], x: f64) -> f64 {
    let Some((&a0, rest)) = coeffs.split_first() else {
        return 0.0;
    };
    let two_x = x + x;
    let mut b1 = 0.0_f64;
    let mut b2 = 0.0_f64;
    // k runs from n-1 down to 1.
    for &a in rest.iter().rev() {
        let b = two_x * b1 - b2 + a;
        b2 = b1;
        b1 = b;
    }
    a0 + x * b1 - b2
}

/// Derivative of the Chebyshev-T series with respect to `x`:
/// `d/dx sum_k coeffs[k] * T_k(x) = sum_{k>=1} k * coeffs[k] * U_{k-1}(x)`.
///
/// Evaluated with the Clenshaw recurrence for the second-kind polynomials.
/// The caller applies the chain-rule scale (for DE granules,
/// `2 * NA / span_days`, as in `VFAC` of `testeph.f`'s `INTERP`).
#[must_use]
pub fn chebyshev_rate(coeffs: &[f64], x: f64) -> f64 {
    if coeffs.len() < 2 {
        return 0.0;
    }
    let two_x = x + x;
    let mut b1 = 0.0_f64;
    let mut b2 = 0.0_f64;
    // Iterate a_k for k = n-1 down to 1; the U-series coefficient with
    // index m = k-1 is c_m = k * a_k.  After the loop b1 holds b_0 = S.
    let mut k = usize_to_f64(coeffs.len() - 1);
    for &a in coeffs.iter().skip(1).rev() {
        let b = two_x * b1 - b2 + k * a;
        b2 = b1;
        b1 = b;
        k -= 1.0;
    }
    b1
}

/// Lossless for the small counts used here (coefficient/granule counts and
/// record indices are far below 2^52).
#[expect(
    clippy::cast_precision_loss,
    reason = "inputs are small counts, exactly representable in f64"
)]
#[inline]
pub(crate) fn usize_to_f64(n: usize) -> f64 {
    n as f64
}

/// Truncate a non-negative finite float toward zero (Fortran `IDINT`).
/// Negative inputs clamp to 0; huge inputs saturate (callers re-clamp
/// against their own upper bounds).
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "guarded: negative inputs return 0, `as` saturates on overflow"
)]
#[inline]
pub(crate) fn f64_to_usize_trunc(x: f64) -> usize {
    if x <= 0.0 {
        0
    } else {
        x as usize
    }
}
