//! Extended string formatting for the native [`BigFloat`].
//!
//! This module adds two families of human- and machine-readable string
//! renderings on top of the binary-exact [`fmt::Display`](core::fmt::Display)
//! (`0xb…p…`) provided in `float.rs`:
//!
//! 1. **Decimal scientific / engineering notation** — base-10 renderings with
//!    a caller-chosen number of significant digits.
//!    - [`BigFloat::to_scientific_string`] — `d.ddd…e±E` (one integer digit).
//!    - [`BigFloat::to_engineering_string`] — like scientific, but the decimal
//!      exponent is *always* a multiple of three and the displayed mantissa
//!      lies in `[1, 1000)`.
//!
//! 2. **C99 `%a`-style hexadecimal float** — `±0x1.<hex-frac>p±<binexp>`.
//!    - [`BigFloat::to_hex_string`] — binary-exact (no rounding).
//!    - [`BigFloat::from_hex_float`] — the exact inverse parser.
//!
//! # Decimal conversion strategy
//!
//! A `BigFloat` is `±m·2^e` with `m` a non-negative big integer. To render
//! `D` significant decimal digits we work entirely in exact big-integer
//! arithmetic:
//!
//! 1. Find `E = floor(log10(|V|))` by seeding from the binary exponent and
//!    correcting with exact cross-multiplied comparisons against powers of ten.
//! 2. Form the rational `|V| / 10^(E-D+1)` as a pair of big integers and round
//!    it to the nearest integer `N` (ties to even). `N` then has exactly `D`
//!    digits (or `D+1` on a `999…→100…0` carry, which bumps `E`).
//!
//! Because the conversion never touches `f64` for the actual digits, it is
//! correct at arbitrary magnitude and precision.
//!
//! # Hex conversion strategy
//!
//! Hex maps directly onto the binary representation with **no rounding**: the
//! leading mantissa bit becomes the `1` before the point, the remaining bits
//! are grouped into 4-bit nibbles after the point, and the `p` exponent is the
//! binary exponent of that leading bit. Round-tripping is therefore bit-exact.

use core::cmp::Ordering;

use oxinum_core::{OxiNumError, OxiNumResult, Sign};
use oxinum_int::native::BigUint;

use super::float::{BigFloat, FloatClass, RoundingMode};
#[cfg(doc)]
use super::float_convert::MAX_EXACT_CONVERSION_BITS;

// ===========================================================================
// Shared big-integer helpers
// ===========================================================================

/// Width, in bits, of the exact big-integer arithmetic that
/// [`decimal_magnitude`] would perform for `value`.
///
/// The algorithm cross-multiplies `m·2^max(e,0)·10^max(-k,0)` against
/// `2^max(-e,0)·10^max(k,0)`, where `k` tracks the value's decimal exponent —
/// which is itself proportional to the binary exponent. So `|e|` (plus the
/// mantissa width) is the right cost proxy in both directions.
///
/// This matters because `e` is a free `i64`: `BigFloat::from_parts` imposes no
/// EMAX and [`BigFloat::from_hex_float`] parses the `p` exponent straight out
/// of an untrusted string, so a 15-character input such as `"0x1p2000000000"`
/// would otherwise drive a multi-hundred-megabyte allocation *and* a
/// multiplication of two such operands.
fn decimal_magnitude_cost_bits(value: &BigFloat) -> u64 {
    value
        .mantissa()
        .bit_length()
        .saturating_add(value.exponent().unsigned_abs())
}

/// Bit budget for the *decimal-formatting* path specifically
/// ([`BigFloat::to_scientific_string`] / [`BigFloat::to_engineering_string`]),
/// tighter than the shared [`MAX_EXACT_CONVERSION_BITS`].
///
/// `decimal_magnitude_cost_bits` is a proxy for the size of the big integers
/// involved, but the two guards that consult it do very different amounts of
/// work per bit: the `to_bigint_*` / `float_to_rational` conversions in
/// `float_convert.rs` only ever *shift* (`BigUint::shl_bits`), an `O(n)`
/// linear allocation all the way up to `MAX_EXACT_CONVERSION_BITS` (128
/// Mibits, a 16 MiB integer). Decimal formatting additionally computes an
/// exact power of ten of comparable size (`pow10`, `O(log n)` balanced
/// multiplications) and divides two comparable-magnitude big integers by it
/// (`round_ratio_half_even`) — real work that, even after caching/binary
/// exponentiation removed the `O(n^2)` blowup a naive implementation has,
/// still grows much faster than linearly once the operands leave the small
/// schoolbook-multiplication regime. Reusing the shared, far more permissive
/// budget here would let a well-formed exponent nowhere near
/// `MAX_EXACT_CONVERSION_BITS` (e.g. `"0x1p9999999"`, an 11-byte string) cost
/// multiple seconds of CPU per formatting call — a DoS on any service that
/// formats attacker-supplied numbers.
///
/// `2^19` bits (~524,288 bits, ~157,800 decimal digits) is calibrated by
/// measurement: every `to_scientific_string`/`to_engineering_string` call
/// exercised up to this size completed in well under 100 ms (typically
/// single-digit milliseconds), while values beyond it can measurably reach
/// multi-second territory (see `crates/oxinum-float/tests/
/// decimal_format_dos_regression.rs`). It sits ~5x above the largest
/// exponent this crate's own test suite (and any realistic caller formatting
/// an actual float/double/extended-precision value, whose binary exponent
/// tops out at a few thousand at most) exercises in decimal form, so nothing
/// that previously rendered in reasonable time is pushed into the hex
/// fallback. Beyond it, `to_hex_string` remains a lossless, `O(n)`-cost
/// alternative rendering — never a rejection.
const MAX_DECIMAL_FORMAT_BITS: u64 = 1 << 19;

/// `10^n` as a [`BigUint`]. `10^0 == 1`.
///
/// Computed by binary (square-and-multiply) exponentiation so every
/// multiplication is *balanced* (roughly equal-sized operands), which lets
/// [`BigUint`]'s Karatsuba / Toom-Cook-3 tiers engage for the large
/// intermediate values this function is specifically here to produce.
///
/// # Why this matters
///
/// The previous implementation built `10^n` by repeatedly multiplying a
/// growing accumulator by a *fixed*, small (`10^19`) chunk: `n / 19`
/// sequential multiplies, each one a `k`-limb-by-1-limb pass. Multiplying a
/// `k`-limb number by a 1-limb constant only ever costs `O(k)`, no matter how
/// fast the underlying multiply algorithm is — Karatsuba/Toom-3 need *both*
/// operands to be large to pay off, and a 1-limb operand never qualifies. So
/// the accumulator's size grew linearly with the iteration count while each
/// step's cost also grew linearly, making the whole loop `O(n^2)`: `decimal_magnitude_cost_bits`'s
/// guard (`MAX_EXACT_CONVERSION_BITS`, 128 Mibits) assumes work roughly
/// linear in `n`, so a 15-byte hex-float literal such as `"0x1p9999999"`
/// (well inside that bit budget) could still drive tens of seconds of CPU —
/// exactly the cost model mismatch this function closes. Binary
/// exponentiation instead does `O(log n)` multiplications of same-sized
/// operands, so it benefits from the fast tiers exactly where it matters
/// most (the final, largest squarings).
fn pow10(n: u64) -> BigUint {
    if n == 0 {
        return BigUint::one();
    }
    let mut base = BigUint::from_u64(10);
    let mut result = BigUint::one();
    let mut e = n;
    while e > 0 {
        if e & 1 == 1 {
            result = &result * &base;
        }
        e >>= 1;
        if e > 0 {
            base = &base * &base;
        }
    }
    result
}

/// `2^n` as a [`BigUint`].
fn pow2(n: u64) -> BigUint {
    BigUint::one().shl_bits(n)
}

/// A single cached power of ten (`10^exp == value`), reused across the
/// cluster of nearby exponents [`decimal_magnitude`] evaluates.
///
/// `decimal_magnitude`'s correction loop and Step 3 only ever need `10^x`
/// for `x` values that sit within `O(sig_digits)` of each other (consecutive
/// correction-loop candidates differ by exactly 1; Step 3's exponent is
/// exactly `sig_digits - 1` below the final candidate). Calling the
/// `O(log n)`-multiplication [`pow10`] fresh for each one redundantly
/// repeats the same *large*, balanced multiplications for numbers that
/// differ from each other only by a handful of factors of ten. Scaling the
/// previously cached value by `10^|delta|` instead is one big-by-tiny
/// multiply/divide — `O(n)` with a tiny constant — and `delta` is bounded by
/// the gap between calls (at most `sig_digits`), never by the exponent's own
/// magnitude.
struct Pow10Cache {
    exp: u64,
    value: BigUint,
}

/// `10^target`, reusing and incrementally rescaling `*cache` instead of
/// recomputing [`pow10`] from scratch whenever a power of ten is already
/// cached nearby. `target == 0` short-circuits to `1` without touching the
/// cache.
fn pow10_cached(cache: &mut Option<Pow10Cache>, target: u64) -> BigUint {
    if target == 0 {
        return BigUint::one();
    }
    match cache {
        Some(c) if c.exp == target => c.value.clone(),
        Some(c) => {
            let value = if target > c.exp {
                &c.value * &pow10(target - c.exp)
            } else {
                // Exact: `c.value` is `10^(c.exp)` and `c.exp > target`
                // here, so this divides out `10^(c.exp - target)` with zero
                // remainder.
                &c.value / &pow10(c.exp - target)
            };
            c.exp = target;
            c.value = value.clone();
            value
        }
        None => {
            let value = pow10(target);
            *cache = Some(Pow10Cache {
                exp: target,
                value: value.clone(),
            });
            value
        }
    }
}

/// Round the exact rational `num / den` (with `den > 0`) to the nearest
/// integer, breaking ties to even. Returns the rounded big integer.
fn round_ratio_half_even(num: &BigUint, den: &BigUint) -> BigUint {
    let quotient = num / den;
    let remainder = num % den;
    if remainder.is_zero() {
        return quotient;
    }
    // Compare 2*remainder against den.
    let twice_rem = remainder.shl_bits(1);
    match twice_rem.cmp(den) {
        Ordering::Less => quotient,
        Ordering::Greater => &quotient + &BigUint::one(),
        Ordering::Equal => {
            // Exact half — round to even.
            if quotient.test_bit(0) {
                &quotient + &BigUint::one()
            } else {
                quotient
            }
        }
    }
}

// ===========================================================================
// Decimal scientific / engineering notation
// ===========================================================================

/// Decimal rendering of the *magnitude* of a non-zero `BigFloat`.
///
/// Returns `(digits, exp10)` where `digits` is a string of exactly
/// `sig_digits` decimal characters (`'0'..='9'`) and the value's magnitude
/// equals `digits[0] . digits[1..] × 10^exp10`. In other words `exp10` is the
/// base-10 exponent of the leading digit.
fn decimal_magnitude(value: &BigFloat, sig_digits: usize) -> (String, i64) {
    let d = sig_digits.max(1);
    let m = value.mantissa();
    let e = value.exponent();

    let e_pos = if e > 0 { e as u64 } else { 0 };
    // `unsigned_abs`, not `-e`: `e` can reach `i64::MIN`.
    let e_neg = if e < 0 { e.unsigned_abs() } else { 0 };

    // Hoisted out of the comparator below: both depend only on `e`/`m`, not
    // on the decimal candidate under test, so computing them once (rather
    // than once per comparison, as the previous version did inside the
    // closure) avoids a redundant `O(e_pos)`-bit allocation on every
    // correction-loop iteration. Step 3 reuses these same two values instead
    // of rebuilding them a second time.
    let m_shifted = m.shl_bits(e_pos); // m * 2^max(e, 0)
    let two_pow_e_neg = pow2(e_neg); // 2^max(-e, 0)

    // --- Step 1: estimate E = floor(log10(|V|)). ---
    // top_bit position = e + (bit_length - 1) is the binary exponent of |V|.
    let top_bit = e.saturating_add(m.bit_length() as i64 - 1);
    // log10(2) ≈ 0.30102999566398114.
    let mut big_e = (top_bit as f64 * core::f64::consts::LOG10_2).floor() as i64;

    // --- Step 2: correct E with exact comparisons until 10^E <= |V| < 10^(E+1). ---
    // Compare |V| = m·2^e against 10^cand via cross-multiplied big integers:
    //   m·2^e  ⪋ 10^cand
    //   m·2^max(e,0)·10^max(-cand,0)  ⪋  2^max(-e,0)·10^max(cand,0)
    //
    // `pow10_cache` carries a single big power of ten across every candidate
    // this loop (and Step 3 below) evaluates — see `Pow10Cache`'s doc
    // comment for why that turns several independent large `pow10` calls
    // into one, plus cheap incremental rescaling.
    let mut pow10_cache: Option<Pow10Cache> = None;
    let cmp_vs_pow10 = |cand: i64,
                        m_shifted: &BigUint,
                        two_pow_e_neg: &BigUint,
                        cache: &mut Option<Pow10Cache>|
     -> Ordering {
        let c_pos = cand.max(0) as u64;
        let c_neg = (-cand).max(0) as u64;
        let lhs = if c_neg > 0 {
            m_shifted * &pow10_cached(cache, c_neg)
        } else {
            m_shifted.clone()
        };
        let rhs = if c_pos > 0 {
            two_pow_e_neg * &pow10_cached(cache, c_pos)
        } else {
            two_pow_e_neg.clone()
        };
        lhs.cmp(&rhs)
    };
    // Nudge up while |V| >= 10^(E+1).
    while cmp_vs_pow10(big_e + 1, &m_shifted, &two_pow_e_neg, &mut pow10_cache) != Ordering::Less {
        big_e += 1;
    }
    // Nudge down while |V| < 10^E.
    while cmp_vs_pow10(big_e, &m_shifted, &two_pow_e_neg, &mut pow10_cache) == Ordering::Less {
        big_e -= 1;
    }

    // --- Step 3: N = round(|V| / 10^(E-D+1)), ties to even. ---
    let k = big_e - (d as i64) + 1; // place value of the least significant digit
    let k_pos = k.max(0) as u64;
    let k_neg = (-k).max(0) as u64;
    // num = m · 2^max(e,0) · 10^max(-k,0)
    let num = if k_neg > 0 {
        &m_shifted * &pow10_cached(&mut pow10_cache, k_neg)
    } else {
        m_shifted
    };
    // den = 2^max(-e,0) · 10^max(k,0)
    let den = if k_pos > 0 {
        &two_pow_e_neg * &pow10_cached(&mut pow10_cache, k_pos)
    } else {
        two_pow_e_neg
    };
    let n = round_ratio_half_even(&num, &den);

    // --- Step 4: render N and fix a possible carry (D+1 digits). ---
    let mut digits = match n.to_radix(10) {
        Ok(s) => s,
        Err(_) => "0".repeat(d),
    };
    let mut exp10 = big_e;
    if digits.len() == d + 1 {
        // Rounding rolled 9…9 into 10…0; the trailing digit is '0'.
        digits.pop();
        exp10 += 1;
    }
    // Defensive: pad on the right if the integer rendered shorter than D
    // (cannot normally happen for a non-zero value, but keeps the slice math
    // total).
    while digits.len() < d {
        digits.push('0');
    }
    // Trim to exactly D characters.
    if digits.len() > d {
        digits.truncate(d);
    }
    (digits, exp10)
}

impl BigFloat {
    /// Render this value in decimal **scientific** notation with `sig_digits`
    /// significant digits: `d.ddd…e±E` (a single digit before the point).
    ///
    /// The value is rounded (ties to even) to `sig_digits` significant decimal
    /// digits. `sig_digits` is clamped to at least 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxinum_float::native::{BigFloat, RoundingMode};
    /// let x = BigFloat::from_i64(12345, 64, RoundingMode::HalfEven);
    /// assert_eq!(x.to_scientific_string(5), "1.2345e4");
    /// assert_eq!(x.to_scientific_string(1), "1e4");
    /// ```
    ///
    /// # Extreme exponents
    ///
    /// A decimal rendering of `±m·2^e` requires exact big-integer arithmetic
    /// roughly `|e|` bits wide, *plus* an exact power of ten of comparable
    /// size and a division by it — real work, not just an allocation. When
    /// that exceeds `MAX_DECIMAL_FORMAT_BITS` this method returns the
    /// lossless [`BigFloat::to_hex_string`] rendering instead of paying for
    /// it. See `MAX_DECIMAL_FORMAT_BITS` for how that budget was chosen.
    pub fn to_scientific_string(&self, sig_digits: usize) -> String {
        // Non-finite values must be handled before the is_zero() check, since
        // NaN and Inf have mantissa=0 and would otherwise format incorrectly.
        match self.class {
            FloatClass::Nan => return "NaN".to_string(),
            FloatClass::Infinite => {
                return if self.sign() == Sign::Negative {
                    "-inf".to_string()
                } else {
                    "inf".to_string()
                };
            }
            FloatClass::Finite => {}
        }
        let d = sig_digits.max(1);
        if self.is_zero() {
            if d == 1 {
                return "0e0".to_string();
            }
            let mut s = String::from("0.");
            for _ in 1..d {
                s.push('0');
            }
            s.push_str("e0");
            return s;
        }
        // Bounded-work guard: fall back to the exact (and cheap, O(n)) hex
        // rendering rather than paying the super-linear cost of exact power-
        // of-ten construction + division at extreme magnitude. See
        // `MAX_DECIMAL_FORMAT_BITS`.
        if decimal_magnitude_cost_bits(self) > MAX_DECIMAL_FORMAT_BITS {
            return self.to_hex_string();
        }
        let (digits, exp10) = decimal_magnitude(self, d);
        let mut out = String::new();
        if self.sign() == Sign::Negative {
            out.push('-');
        }
        // First digit, then the fractional tail.
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        out.push('e');
        out.push_str(&exp10.to_string());
        out
    }

    /// Render this value in decimal **engineering** notation with `sig_digits`
    /// significant digits.
    ///
    /// Engineering notation is scientific notation constrained so the decimal
    /// exponent is always a multiple of three and the displayed mantissa lies
    /// in `[1, 1000)`. The mantissa therefore carries one, two, or three digits
    /// before the point.
    ///
    /// The value is rounded (ties to even) to `sig_digits` significant decimal
    /// digits. `sig_digits` is clamped to at least 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxinum_float::native::{BigFloat, RoundingMode};
    /// let x = BigFloat::from_i64(12345, 64, RoundingMode::HalfEven);
    /// assert_eq!(x.to_engineering_string(5), "12.345e3");
    /// ```
    ///
    /// # Extreme exponents
    ///
    /// Same bounded-work fallback as [`BigFloat::to_scientific_string`]:
    /// beyond `MAX_DECIMAL_FORMAT_BITS` this returns the lossless
    /// [`BigFloat::to_hex_string`] rendering instead.
    pub fn to_engineering_string(&self, sig_digits: usize) -> String {
        // Non-finite values must be handled before the is_zero() check, since
        // NaN and Inf have mantissa=0 and would otherwise format incorrectly.
        match self.class {
            FloatClass::Nan => return "NaN".to_string(),
            FloatClass::Infinite => {
                return if self.sign() == Sign::Negative {
                    "-inf".to_string()
                } else {
                    "inf".to_string()
                };
            }
            FloatClass::Finite => {}
        }
        let d = sig_digits.max(1);
        if self.is_zero() {
            if d == 1 {
                return "0e0".to_string();
            }
            let mut s = String::from("0.");
            for _ in 1..d {
                s.push('0');
            }
            s.push_str("e0");
            return s;
        }
        // Bounded-work guard: fall back to the exact (and cheap, O(n)) hex
        // rendering rather than paying the super-linear cost of exact power-
        // of-ten construction + division at extreme magnitude. See
        // `MAX_DECIMAL_FORMAT_BITS`.
        if decimal_magnitude_cost_bits(self) > MAX_DECIMAL_FORMAT_BITS {
            return self.to_hex_string();
        }
        let (digits, exp10) = decimal_magnitude(self, d);
        // Engineering exponent: snap down to the nearest multiple of three.
        // `int_digits` (= 1, 2, or 3) digits sit before the decimal point.
        let shift = exp10.rem_euclid(3); // 0, 1, or 2
        let eng_exp = exp10 - shift;
        let int_digits = (shift + 1) as usize;

        // Pad the digit string on the right so it has at least `int_digits`
        // characters (only relevant when sig_digits < int_digits).
        let mut padded = digits;
        while padded.len() < int_digits {
            padded.push('0');
        }

        let mut out = String::new();
        if self.sign() == Sign::Negative {
            out.push('-');
        }
        out.push_str(&padded[..int_digits]);
        if padded.len() > int_digits {
            out.push('.');
            out.push_str(&padded[int_digits..]);
        }
        out.push('e');
        out.push_str(&eng_exp.to_string());
        out
    }
}

// ===========================================================================
// C99 %a-style hexadecimal float
// ===========================================================================

/// Map a hex digit character to its 0..=15 value, or `None` if not hex.
fn hex_value(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

/// Map a 0..=15 value to a lowercase hex digit character.
fn hex_char(v: u8) -> char {
    match v {
        0..=9 => (b'0' + v) as char,
        10..=15 => (b'a' + (v - 10)) as char,
        _ => '0',
    }
}

impl BigFloat {
    /// Render this value as a C99 `%a`-style hexadecimal float string:
    /// `±0x1.<hex-frac>p±<binexp>`.
    ///
    /// The rendering is **binary-exact**: the leading mantissa bit becomes the
    /// `1` before the point, the remaining bits are grouped (MSB-first) into
    /// 4-bit hex nibbles after the point, and the `p` exponent is the binary
    /// exponent of the leading bit. The canonical zero renders as `0x0p0`.
    ///
    /// [`BigFloat::from_hex_float`] is the exact inverse, so
    /// `from_hex_float(x.to_hex_string())` reproduces `x` bit-for-bit.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxinum_float::native::BigFloat;
    /// let x = BigFloat::from_f64(12.0, 53).expect("finite");
    /// // 12 = 1.1000…b × 2^3  →  0x1.8p3
    /// assert_eq!(x.to_hex_string(), "0x1.8p3");
    /// ```
    pub fn to_hex_string(&self) -> String {
        // Non-finite values must be handled before the is_zero() check, since
        // NaN and Inf have mantissa=0 and would otherwise format incorrectly.
        match self.class {
            FloatClass::Nan => return "NaN".to_string(),
            FloatClass::Infinite => {
                return if self.sign() == Sign::Negative {
                    "-inf".to_string()
                } else {
                    "inf".to_string()
                };
            }
            FloatClass::Finite => {}
        }
        if self.is_zero() {
            return "0x0p0".to_string();
        }
        let m = self.mantissa();
        let bits = m.bit_length(); // >= 1 for non-zero, normalized to precision
                                   // Leading bit is at index bits-1; its binary place value is
                                   // exponent + (bits - 1).
        let leading_index = bits - 1;
        let p_exp = self.exponent().saturating_add(leading_index as i64);

        let mut out = String::new();
        if self.sign() == Sign::Negative {
            out.push('-');
        }
        out.push_str("0x1");

        // Fractional bits: the `leading_index` bits below the top bit, MSB
        // first, grouped into nibbles. Pad the final nibble on the right with
        // zero bits so the bit count is a multiple of four.
        if leading_index > 0 {
            let mut frac = String::new();
            // Walk bit positions from (leading_index - 1) down to 0, four at a
            // time, assembling each nibble MSB-first.
            let mut pos = leading_index as i64 - 1;
            while pos >= 0 {
                let mut nibble: u8 = 0;
                for _ in 0..4 {
                    nibble <<= 1;
                    if pos >= 0 {
                        if m.test_bit(pos as u64) {
                            nibble |= 1;
                        }
                        pos -= 1;
                    }
                    // When pos < 0 we shift in implicit zero bits (right pad).
                }
                frac.push(hex_char(nibble));
            }
            // Strip trailing '0' nibbles — they carry no information and keep
            // the representation canonical/short while remaining exact.
            while frac.ends_with('0') {
                frac.pop();
            }
            if !frac.is_empty() {
                out.push('.');
                out.push_str(&frac);
            }
        }

        out.push('p');
        out.push_str(&p_exp.to_string());
        out
    }

    /// Parse a C99 `%a`-style hexadecimal float string into a `BigFloat` at
    /// `prec` bits of precision.
    ///
    /// Accepts `±0x<hexint>[.<hexfrac>]p±<decexp>` (the `0x`/`0X` prefix, at
    /// least one hex digit overall, and the `p`/`P` binary exponent are all
    /// mandatory). The parse is binary-exact before the final
    /// normalization/rounding to `prec` bits.
    ///
    /// # Errors
    ///
    /// Returns [`OxiNumError::Parse`] on any malformed input: a missing `0x`
    /// prefix, a missing `p` exponent, non-hex digits in the significand, a
    /// non-decimal exponent, or stray characters.
    ///
    /// Returns [`OxiNumError::Overflow`] when the parsed value's binary
    /// magnitude falls outside `[EMIN, EMAX]` (see [`BigFloat::try_from_parts`],
    /// [`BigFloat::EMAX`]). A well-formed but out-of-range literal such as
    /// `"0x1p9223372036854775807"` is *refused*, never saturated: a 23-byte
    /// string must not parse into a silently different number, and the whole
    /// point of the range is that no such exponent reaches the shift-driven
    /// operations downstream.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxinum_float::native::BigFloat;
    /// // 0x1.8p3 = 1.5 × 2^3 = 12.
    /// let x = BigFloat::from_hex_float("0x1.8p3", 53).expect("valid hex float");
    /// assert_eq!(x.to_f64(), 12.0);
    ///
    /// // An exponent past EMAX is reported rather than clamped.
    /// assert!(BigFloat::from_hex_float("0x1p9223372036854775807", 53).is_err());
    /// ```
    pub fn from_hex_float(s: &str, prec: u32) -> OxiNumResult<Self> {
        let bytes = s.as_bytes();
        let mut idx = 0usize;
        let len = bytes.len();

        let parse_err = |msg: &str| OxiNumError::Parse(format!("hex float: {msg}").into());

        // --- Optional sign. ---
        let sign = match bytes.first() {
            Some(b'-') => {
                idx += 1;
                Sign::Negative
            }
            Some(b'+') => {
                idx += 1;
                Sign::Positive
            }
            _ => Sign::Positive,
        };

        // --- Mandatory 0x / 0X prefix. ---
        if idx + 1 >= len || bytes[idx] != b'0' || (bytes[idx + 1] | 0x20) != b'x' {
            return Err(parse_err("missing '0x' prefix"));
        }
        idx += 2;

        // --- Integer hex part (zero or more hex digits). ---
        let int_start = idx;
        while idx < len && hex_value(bytes[idx]).is_some() {
            idx += 1;
        }
        let int_part = &bytes[int_start..idx];

        // --- Optional fractional hex part. ---
        let mut frac_part: &[u8] = &[];
        if idx < len && bytes[idx] == b'.' {
            idx += 1;
            let frac_start = idx;
            while idx < len && hex_value(bytes[idx]).is_some() {
                idx += 1;
            }
            frac_part = &bytes[frac_start..idx];
        }

        // At least one significand digit (integer or fractional) is required.
        if int_part.is_empty() && frac_part.is_empty() {
            return Err(parse_err("no significand digits"));
        }

        // --- Mandatory p / P binary exponent. ---
        if idx >= len || (bytes[idx] | 0x20) != b'p' {
            return Err(parse_err("missing 'p' exponent marker"));
        }
        idx += 1;

        // --- Signed decimal exponent. ---
        let exp_sign_neg = match bytes.get(idx) {
            Some(b'-') => {
                idx += 1;
                true
            }
            Some(b'+') => {
                idx += 1;
                false
            }
            _ => false,
        };
        let exp_start = idx;
        while idx < len && bytes[idx].is_ascii_digit() {
            idx += 1;
        }
        if exp_start == idx {
            return Err(parse_err("missing exponent digits"));
        }
        // Trailing garbage?
        if idx != len {
            return Err(parse_err("trailing characters"));
        }
        let exp_str = core::str::from_utf8(&bytes[exp_start..idx])
            .map_err(|_| parse_err("non-UTF-8 exponent"))?;
        let exp_mag: i64 = exp_str
            .parse::<i64>()
            .map_err(|_| parse_err("exponent out of range"))?;
        let p_exp = if exp_sign_neg { -exp_mag } else { exp_mag };

        // --- Assemble the mantissa from the concatenated hex digits. ---
        // value = significand × 2^p_exp, where the significand's last hex digit
        // sits 4·(frac nibbles) bits below the radix point. Treat all the hex
        // digits as one integer `digits`, then:
        //   value = digits × 2^(p_exp - 4·frac_nibbles)
        let frac_nibbles = frac_part.len() as i64;
        let mut all_digits: Vec<u8> = Vec::with_capacity(int_part.len() + frac_part.len());
        all_digits.extend_from_slice(int_part);
        all_digits.extend_from_slice(frac_part);
        // Build a big integer from the hex digits (4 bits each), MSB-first.
        let mut mantissa = BigUint::zero();
        let sixteen = BigUint::from_u64(16);
        for &c in &all_digits {
            let v = hex_value(c).ok_or_else(|| parse_err("invalid hex digit"))?;
            mantissa = &(&mantissa * &sixteen) + &BigUint::from_u64(v as u64);
        }

        if mantissa.is_zero() {
            // e.g. 0x0p0, 0x0.0p5 — all forms of zero.
            return Ok(Self::zero(prec));
        }

        let exponent = p_exp.saturating_sub(frac_nibbles.saturating_mul(4));
        // `try_from_parts`, not `from_parts`: this is the crate's one
        // untrusted-string entry point, so an out-of-range `p` exponent is
        // reported as `OxiNumError::Overflow` rather than silently saturated
        // into a different (and much smaller) number. A 22-byte string such as
        // `"0x1p9223372036854775807"` must not parse as `2^EMAX`.
        Self::try_from_parts(sign, mantissa, exponent, prec, RoundingMode::HalfEven)
    }
}
