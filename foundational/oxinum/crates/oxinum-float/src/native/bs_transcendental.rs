//! Binary-splitting based evaluation of transcendental functions.
//!
//! Provides fast evaluation of `exp`, `sin`, and `cos` using the
//! binary-splitting algorithm above a precision threshold, bypassing the
//! iterative Taylor series which is O(N²) in the number of terms.
//!
//! ## Algorithms
//!
//! All three functions reduce to a hypergeometric-like series with rational
//! argument `y = p/q` (extracted exactly from the `BigFloat` representation):
//!
//! * `exp(y) = Σ_{k≥0} y^k / k!`  — terms: k=0 → (1,1,1,1); k≥1 → (p, q·k, 1, 1)
//! * `sin(y) = Σ_{k≥0} (−1)^k y^(2k+1) / (2k+1)!`  — terms folded into p_k
//! * `cos(y) = Σ_{k≥0} (−1)^k y^(2k) / (2k)!`       — terms folded into p_k
//!
//! After binary splitting over `n` terms, `result = split.t / (split.q · split.b)`.

use oxinum_core::{OxiNumError, OxiNumResult, Sign};
use oxinum_int::native::{BigInt, BigUint};

use super::binary_splitting::{binary_split, BSSeries, BSSplit};
use super::float::{BigFloat, RoundingMode};

// ---------------------------------------------------------------------------
// Threshold: use binary splitting above this precision (bits).
// ---------------------------------------------------------------------------

/// Precision threshold above which the binary-splitting path is used
/// instead of the iterative Taylor series.
pub(crate) const BS_THRESHOLD_BITS: u32 = 512;

// ---------------------------------------------------------------------------
// Helper: convert BigInt → BigFloat (replicated from constants.rs)
// ---------------------------------------------------------------------------

/// Convert a `BigInt` to `BigFloat` at `prec` bits with rounding mode `mode`.
///
/// `from_parts` with `exponent == 0` cannot saturate: the magnitude is the
/// integer's own bit length, and an integer wide enough to exceed `EMAX` would
/// need 137 GB of memory to exist in the first place.
fn bigfloat_from_bigint(n: &BigInt, prec: u32, mode: RoundingMode) -> BigFloat {
    if n.is_zero() {
        return BigFloat::zero(prec);
    }
    BigFloat::from_parts(n.sign(), n.magnitude().clone(), 0, prec, mode)
}

// ---------------------------------------------------------------------------
// Term-count estimators
// ---------------------------------------------------------------------------

/// Number of terms needed so that `log2(k!) > target_bits + 16`.
///
/// The series `Σ y^k/k!` has term magnitude `|y|^k/k!`. After argument
/// reduction `|y| ≤ 1`, the k-th term is at most `1/k!`.  We need
/// `log2(k!) > prec + 16` for convergence with adequate guard bits.
pub(crate) fn term_count_exp(target_bits: u32) -> u64 {
    let target = (target_bits as f64) + 16.0;
    let mut k: u64 = 2;
    let mut log2_kfact: f64 = 1.0; // log2(2!) = 1
    while log2_kfact < target {
        k += 1;
        log2_kfact += (k as f64).log2();
    }
    (k + 1).max(2)
}

/// Number of term pairs (each covering two trig series indices) needed for
/// `cos`/`sin` series convergence.
///
/// The `k`-th cos term has magnitude `|y|^(2k) / (2k)!`.  Successive term
/// pairs multiply by `|y|^2 / ((2k-1)(2k))`.  We accumulate
/// `log2((2k-1)·(2k))` per step until the accumulated factorial exceeds
/// `target_bits + 16`.
pub(crate) fn term_count_trig(target_bits: u32) -> u64 {
    let target = (target_bits as f64) + 16.0;
    let mut k: u64 = 1;
    let mut log2_fact: f64 = 0.0;
    loop {
        // Each step k contributes log2((2k-1) * 2k) to the factorial log.
        let a = (2 * k - 1) as f64;
        let b = (2 * k) as f64;
        log2_fact += a.log2() + b.log2();
        if log2_fact > target {
            break;
        }
        k += 1;
        if k > 100_000 {
            break; // safety cap — never reached in practice
        }
    }
    (k + 2).max(2)
}

// ---------------------------------------------------------------------------
// Rational argument extraction
// ---------------------------------------------------------------------------

/// Extract the exact rational `p/q` from a `BigFloat` `y`.
///
/// Given `y = sign * mantissa * 2^e`:
/// * if `e ≥ 0`: `p = sign * (mantissa << e)`,  `q = 1`
/// * if `e < 0`: `p = sign * mantissa`,  `q = 2^(-e)`
///
/// # Unbounded-allocation guard
///
/// Both branches materialize a `|e|`-bit shift, and `e` is a free `i64`:
/// `BigFloat::from_parts` imposes no EMAX and `BigFloat::from_hex_float`
/// parses the `p` exponent straight out of an untrusted string. A value such
/// as `1 × 2^-1_000_000_000` therefore used to request a 10^9-bit denominator
/// here (and an exponent near `i64::MIN` aborted outright). `max_shift_bits`
/// caps both directions and returns a typed error instead.
///
/// Callers must screen out negligible arguments (see [`arg_is_negligible`])
/// *before* calling; with that screen in place the cap is never reached for
/// any argument the series path can legitimately be asked to evaluate.
fn split_arg(y: &BigFloat, max_shift_bits: u64) -> OxiNumResult<(BigInt, BigInt)> {
    let m = y.mantissa().clone();
    let sign = y.sign();
    let e = y.exponent();
    // `unsigned_abs` (not `-e`) — `e` can legitimately be `i64::MIN` because
    // `round_to_precision_in_place` lowers the exponent with `saturating_sub`.
    let shift = e.unsigned_abs();
    if shift > max_shift_bits {
        return Err(OxiNumError::Precision(
            "binary splitting: argument exponent out of range for exact \
             rational lifting"
                .into(),
        ));
    }
    if e >= 0 {
        let shifted = m.shl_bits(shift);
        let p = BigInt::from_parts(sign, shifted);
        let q = BigInt::one();
        Ok((p, q))
    } else {
        let p = BigInt::from_parts(sign, m);
        let q_mag = BigUint::one().shl_bits(shift);
        let q = BigInt::from_parts(Sign::Positive, q_mag);
        Ok((p, q))
    }
}

/// Widest exact-rational lifting `split_arg` will perform for a given working
/// precision, in bits. Generous enough that no argument surviving
/// [`arg_is_negligible`] can trip it, small enough that a hostile exponent
/// cannot drive an unbounded allocation.
fn max_split_shift(y: &BigFloat, work_prec: u32) -> u64 {
    y.mantissa()
        .bit_length()
        .saturating_add(u64::from(work_prec))
        .saturating_add(SPLIT_ARG_GUARD_BITS)
}

/// Slack (in bits) added on top of `work_prec` before an argument counts as
/// negligible, and on top of the mantissa width before `split_arg` refuses.
const SPLIT_ARG_GUARD_BITS: u64 = 64;

/// Returns `true` when `|y| < 2^-(work_prec + 3)`, i.e. `y` is so far below the
/// working-precision ULP of 1 that every series evaluated here collapses to its
/// first one or two terms.
///
/// This is both a correctness shortcut and the screen that keeps `split_arg`'s
/// denominator bounded: for a non-negligible `y`, `-e ≤ bit_length(m) +
/// work_prec + 4`, so the lifting is proportional to the requested precision
/// rather than to a caller-supplied exponent.
fn arg_is_negligible(y: &BigFloat, work_prec: u32) -> bool {
    if y.is_zero() || !y.is_finite() {
        return false;
    }
    // Binary magnitude of the leading bit: |y| ∈ [2^top, 2^(top+1)).
    let bits = y.mantissa().bit_length() as i64;
    let top = y.exponent().saturating_add(bits.saturating_sub(1));
    let threshold = -(i64::from(work_prec).saturating_add(3));
    top < threshold
}

// ---------------------------------------------------------------------------
// BSSeries implementations
// ---------------------------------------------------------------------------

/// Binary-splitting series for `exp(p/q) = Σ_{k≥0} (p/q)^k / k!`.
///
/// Term factors:
/// * `k = 0`: `(1, 1, 1, 1)` → contributes the k=0 term (= 1).
/// * `k ≥ 1`: `(p, q·k, 1, 1)` — each step multiplies numerator by `p`
///   and denominator by `q·k`, building up `p^k / (q^k · k!)` via the
///   combine rule.
struct ExpSeries {
    p: BigInt,
    q: BigInt,
}

impl BSSeries for ExpSeries {
    fn term(&self, k: u64) -> (BigInt, BigInt, BigInt, BigInt) {
        if k == 0 {
            return (BigInt::one(), BigInt::one(), BigInt::one(), BigInt::one());
        }
        let qk = &self.q * &BigInt::from(k as i64);
        (self.p.clone(), qk, BigInt::one(), BigInt::one())
    }
}

/// Binary-splitting series for `sin(p/q) = Σ_{k≥0} (−1)^k (p/q)^(2k+1) / (2k+1)!`.
///
/// Term factors (sign folded into `p_k`):
/// * `k = 0`: `(p, q, 1, 1)` → the k=0 term is `(p/q) / 1! = p/q`.
/// * `k ≥ 1`: `(−p², q²·(2k)·(2k+1), 1, 1)` — each step multiplies
///   numerator by `−p²` and denominator by `q²·(2k)(2k+1)`.
///
/// The cumulative product of `p_k` factors for `k=0..N` yields the prefix
/// `(−1)^N · p^(2N+1)`.  Combined with the `q` denominator products and the
/// engine's `t` accumulation this gives `sin(p/q)` to the requested precision.
struct SinSeries {
    p: BigInt,
    q: BigInt,
    p2: BigInt, // p * p  (precomputed)
    q2: BigInt, // q * q  (precomputed)
}

impl BSSeries for SinSeries {
    fn term(&self, k: u64) -> (BigInt, BigInt, BigInt, BigInt) {
        if k == 0 {
            return (self.p.clone(), self.q.clone(), BigInt::one(), BigInt::one());
        }
        // Negate p² for alternating sign.
        let neg_p2 = -self.p2.clone();
        let denom_k = &self.q2 * &BigInt::from((2 * k) as i64) * &BigInt::from((2 * k + 1) as i64);
        (neg_p2, denom_k, BigInt::one(), BigInt::one())
    }
}

/// Binary-splitting series for `cos(p/q) = Σ_{k≥0} (−1)^k (p/q)^(2k) / (2k)!`.
///
/// Term factors (sign folded into `p_k`):
/// * `k = 0`: `(1, 1, 1, 1)` → the k=0 term is `1`.
/// * `k ≥ 1`: `(−p², q²·(2k−1)·(2k), 1, 1)`.
struct CosSeries {
    p2: BigInt, // p * p  (precomputed)
    q2: BigInt, // q * q  (precomputed)
}

impl BSSeries for CosSeries {
    fn term(&self, k: u64) -> (BigInt, BigInt, BigInt, BigInt) {
        if k == 0 {
            return (BigInt::one(), BigInt::one(), BigInt::one(), BigInt::one());
        }
        let neg_p2 = -self.p2.clone();
        let denom_k = &self.q2 * &BigInt::from((2 * k - 1) as i64) * &BigInt::from((2 * k) as i64);
        (neg_p2, denom_k, BigInt::one(), BigInt::one())
    }
}

// ---------------------------------------------------------------------------
// Reconstruction helper
// ---------------------------------------------------------------------------

/// Convert a binary-splitting result `split` to a `BigFloat`:
///   result = split.t / (split.q · split.b)
fn reconstruct(split: BSSplit, work_prec: u32, mode: RoundingMode) -> OxiNumResult<BigFloat> {
    let denom_int: BigInt = &split.q * &split.b;
    let numer_f = bigfloat_from_bigint(&split.t, work_prec, mode);
    let denom_f = bigfloat_from_bigint(&denom_int, work_prec, mode);
    numer_f.div_ref_with_mode(&denom_f, mode)
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Evaluate `exp(y)` at `work_prec` bits via binary splitting.
///
/// `y` must already be argument-reduced (small `|y|`).  The function is
/// exact-rational in the argument — it lifts `y`'s mantissa+exponent to an
/// exact `p/q` and runs the binary-splitting engine.
pub(crate) fn exp_bs(y: &BigFloat, work_prec: u32, mode: RoundingMode) -> OxiNumResult<BigFloat> {
    if y.is_zero() {
        return Ok(BigFloat::from_i64(1, work_prec, mode));
    }
    if arg_is_negligible(y, work_prec) {
        // |y| < 2^-(work_prec+3), so exp(y) = 1 + y + y²/2 + … and the y²/2
        // tail is below 2^-(2·work_prec), far under half a ULP of the sum.
        // `1 + y` is therefore the correctly-rounded result in every rounding
        // mode, and the (capped) add reaches it without lifting `y` to an
        // exact rational whose denominator is 2^|exponent| bits wide.
        let one = BigFloat::from_i64(1, work_prec, mode);
        return Ok(one.add_ref_with_mode(y, mode));
    }
    let (p, q) = split_arg(y, max_split_shift(y, work_prec))?;
    let n = term_count_exp(work_prec);
    let split = binary_split(&ExpSeries { p, q }, 0, n);
    let s = reconstruct(split, work_prec, mode)?;
    Ok(s.with_precision(work_prec, mode))
}

/// Evaluate `(sin(u), cos(u))` at `work_prec` bits via binary splitting.
///
/// `u` must already be the reduced argument from quadrant reduction.
pub(crate) fn sincos_bs(
    u: &BigFloat,
    work_prec: u32,
    mode: RoundingMode,
) -> OxiNumResult<(BigFloat, BigFloat)> {
    if u.is_zero() {
        let sin_val = BigFloat::zero(work_prec);
        let cos_val = BigFloat::from_i64(1, work_prec, mode);
        return Ok((sin_val, cos_val));
    }
    if arg_is_negligible(u, work_prec) {
        // |u| < 2^-(work_prec+3):
        //   sin(u) = u − u³/6,  with 0 < |u³/6| < |u|·2^-(2·work_prec)
        //   cos(u) = 1 − u²/2,  with 0 < u²/2   < 2^-(2·work_prec+1)
        // Both tails sit strictly inside the open interval (0, ½ ULP), so any
        // representative from that interval rounds identically in every mode.
        // We use |u|·2^-(work_prec+4) and 2^-(work_prec+4) respectively, which
        // keeps the arithmetic bounded instead of lifting `u` to an exact
        // rational with a 2^|exponent|-bit denominator.
        // Placing the tail requires lowering `u`'s magnitude by `work_prec + 4`
        // bits. When `u` already sits that close to `BigFloat::EMIN` the tail
        // is not representable at all: `BigFloat::from_parts` would *saturate*
        // it back up to `2^EMIN`, i.e. to something comparable with `u`, and
        // the subtraction would cancel `u` down to zero — a wrong result, not
        // merely an imprecise one. In that (astronomically underflowed) corner
        // just return `u`, which is `sin(u)` correctly rounded to nearest,
        // since the true tail is below every representable magnitude.
        let tail_ilogb = u
            .ilogb()
            .map(|l| i128::from(l) - i128::from(work_prec) - 4)
            .unwrap_or(i128::MIN);
        let sin_val = if tail_ilogb < i128::from(BigFloat::EMIN) {
            u.clone().with_precision(work_prec, mode)
        } else {
            // In range, so the subtraction is exact-by-construction: the tail
            // is `|u|·2^-(work_prec+4)`, built by lowering `u`'s exponent.
            let tail_exp = u
                .exponent()
                .saturating_sub(i64::from(work_prec).saturating_add(4));
            let sin_tail =
                BigFloat::from_parts(u.sign(), u.mantissa().clone(), tail_exp, work_prec, mode);
            u.clone()
                .with_precision(work_prec, mode)
                .sub_ref_with_mode(&sin_tail, mode)
        };
        let cos_tail = BigFloat::from_parts(
            Sign::Positive,
            BigUint::one(),
            -(i64::from(work_prec).saturating_add(4)),
            work_prec,
            mode,
        );
        let cos_val = BigFloat::from_i64(1, work_prec, mode).sub_ref_with_mode(&cos_tail, mode);
        return Ok((sin_val, cos_val));
    }
    let (p, q) = split_arg(u, max_split_shift(u, work_prec))?;
    let p2 = &p * &p;
    let q2 = &q * &q;

    let n_trig = term_count_trig(work_prec);

    let sin_split = binary_split(
        &SinSeries {
            p: p.clone(),
            q: q.clone(),
            p2: p2.clone(),
            q2: q2.clone(),
        },
        0,
        n_trig,
    );
    let cos_split = binary_split(&CosSeries { p2, q2 }, 0, n_trig);

    let sin_val = reconstruct(sin_split, work_prec, mode)?;
    let cos_val = reconstruct(cos_split, work_prec, mode)?;

    Ok((
        sin_val.with_precision(work_prec, mode),
        cos_val.with_precision(work_prec, mode),
    ))
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const MODE: RoundingMode = RoundingMode::HalfEven;
    const PREC: u32 = 600;

    fn bi(n: i64) -> BigInt {
        BigInt::from(n)
    }

    // --- Per-term unit tests ---

    #[test]
    fn exp_series_term_zero() {
        let s = ExpSeries { p: bi(1), q: bi(2) };
        let (p, q, b, a) = s.term(0);
        assert_eq!(p, BigInt::one());
        assert_eq!(q, BigInt::one());
        assert_eq!(b, BigInt::one());
        assert_eq!(a, BigInt::one());
    }

    #[test]
    fn exp_series_term_three() {
        // ExpSeries { p=1, q=2 }.term(3) = (1, 2*3=6, 1, 1)
        let s = ExpSeries { p: bi(1), q: bi(2) };
        let (p, q, b, a) = s.term(3);
        assert_eq!(p, bi(1));
        assert_eq!(q, bi(6));
        assert_eq!(b, BigInt::one());
        assert_eq!(a, BigInt::one());
    }

    #[test]
    fn sin_series_term_zero() {
        // SinSeries { p=1, q=1 }.term(0) = (1, 1, 1, 1)
        let s = SinSeries {
            p: bi(1),
            q: bi(1),
            p2: bi(1),
            q2: bi(1),
        };
        let (p, q, b, a) = s.term(0);
        assert_eq!(p, bi(1));
        assert_eq!(q, bi(1));
        assert_eq!(b, BigInt::one());
        assert_eq!(a, BigInt::one());
    }

    #[test]
    fn sin_series_term_two() {
        // SinSeries { p=1, q=1, p2=1, q2=1 }.term(2):
        //   neg_p2 = -1
        //   denom_k = 1 * 4 * 5 = 20
        let s = SinSeries {
            p: bi(1),
            q: bi(1),
            p2: bi(1),
            q2: bi(1),
        };
        let (p, q, b, a) = s.term(2);
        assert_eq!(p, bi(-1));
        assert_eq!(q, bi(20));
        assert_eq!(b, BigInt::one());
        assert_eq!(a, BigInt::one());
    }

    #[test]
    fn cos_series_term_one() {
        // CosSeries { p2=1, q2=1 }.term(1):
        //   neg_p2 = -1
        //   denom_k = 1 * 1 * 2 = 2
        let s = CosSeries {
            p2: bi(1),
            q2: bi(1),
        };
        let (p, q, b, a) = s.term(1);
        assert_eq!(p, bi(-1));
        assert_eq!(q, bi(2));
        assert_eq!(b, BigInt::one());
        assert_eq!(a, BigInt::one());
    }

    // --- Basic sanity tests ---

    #[test]
    fn exp_bs_zero_is_one() {
        let y = BigFloat::zero(PREC);
        let result = exp_bs(&y, PREC, MODE).expect("exp_bs(0)");
        let diff = (result.to_f64() - 1.0_f64).abs();
        assert!(
            diff < 1e-15,
            "exp_bs(0) = {}, expected 1.0",
            result.to_f64()
        );
    }

    #[test]
    fn sincos_bs_zero() {
        let u = BigFloat::zero(PREC);
        let (sin_u, cos_u) = sincos_bs(&u, PREC, MODE).expect("sincos_bs(0)");
        assert!(
            sin_u.to_f64().abs() < 1e-15,
            "sin_bs(0) = {}",
            sin_u.to_f64()
        );
        assert!(
            (cos_u.to_f64() - 1.0).abs() < 1e-15,
            "cos_bs(0) = {}",
            cos_u.to_f64()
        );
    }

    // --- End-to-end tests via BigFloat::exp / sin / cos at prec >= BS_THRESHOLD_BITS ---

    #[test]
    fn exp_bs_one_matches_e_const() {
        use crate::native::constants::e_const;
        let one = BigFloat::from_i64(1, PREC, MODE);
        let e_val = one.exp(PREC, MODE).expect("exp(1)");
        let e_const_val = e_const(PREC).expect("e_const");
        let diff = (e_val.to_f64() - e_const_val.to_f64()).abs();
        assert!(diff < 1e-9, "exp(1) vs e_const diff = {diff}");
    }

    #[test]
    fn pythagorean_identity_high_prec() {
        let x = BigFloat::from_f64(0.7, PREC).expect("0.7");
        let sin_x = x.sin(PREC, MODE).expect("sin");
        let cos_x = x.cos(PREC, MODE).expect("cos");
        let s2 = sin_x.mul_ref_with_mode(&sin_x, MODE);
        let c2 = cos_x.mul_ref_with_mode(&cos_x, MODE);
        let sum = s2.add_ref_with_mode(&c2, MODE);
        let diff = (sum.to_f64() - 1.0).abs();
        assert!(diff < 1e-9, "sin²+cos² = {}", sum.to_f64());
    }

    #[test]
    fn sin_pi_over_6_high_prec() {
        use crate::native::constants::pi;
        let pi_val = pi(PREC).expect("pi");
        let six = BigFloat::from_i64(6, PREC, MODE);
        let x = pi_val.div_ref_with_mode(&six, MODE).expect("pi/6");
        let s = x.sin(PREC, MODE).expect("sin");
        let diff = (s.to_f64() - 0.5).abs();
        assert!(diff < 1e-9, "sin(π/6) = {}", s.to_f64());
    }
}
