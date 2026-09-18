//! A minimal arbitrary-precision **unsigned** integer, [`BigUint`].
//!
//! # Scope
//!
//! This type exists to back the public-key arithmetic the XML security specifications need:
//! **RSA signature verification** in `oxixml-dsig` (XML Digital Signatures), which reduces to a
//! single modular exponentiation `s^e mod n` over 2048–4096-bit integers, and the prime-field
//! arithmetic of the NIST curves in [`crate::ec`]. So this module implements just enough
//! big-integer arithmetic for those:
//!
//! * construction from `u64` / `u128` and big-endian bytes,
//! * ordering and the checked add / subtract / multiply operations,
//! * schoolbook [`div_rem`](BigUint::div_rem),
//! * [`modpow`](BigUint::modpow) via square-and-multiply, and
//! * the modular helpers [`add_mod`](BigUint::add_mod), [`sub_mod`](BigUint::sub_mod),
//!   [`mul_mod`](BigUint::mul_mod) and [`mod_inverse`](BigUint::mod_inverse).
//!
//! It is deliberately **not** a general-purpose bignum library: there is no signed arithmetic,
//! no fast-multiplication (Karatsuba/FFT), and no attempt at constant-time execution. Do not
//! use it where side-channel resistance matters — RSA *verification* uses only public values,
//! so variable-time arithmetic is appropriate there.
//!
//! Values are stored as little-endian [`u64`] limbs with no trailing zero limbs; the integer
//! zero is the empty limb vector.
//!
//! # Example
//!
//! ```
//! use oxixml_support::bigint::BigUint;
//!
//! let base = BigUint::from_u64(4);
//! let exponent = BigUint::from_u64(13);
//! let modulus = BigUint::from_u64(497);
//! // 4^13 mod 497 = 445
//! assert_eq!(base.modpow(&exponent, &modulus), Some(BigUint::from_u64(445)));
//! ```

use core::cmp::Ordering;
use core::ops::{Add, Mul, Sub};

/// An arbitrary-precision unsigned integer.
///
/// See the [module documentation](self) for the (intentionally narrow) scope.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct BigUint {
    /// Little-endian base-2^64 limbs, normalized so the most significant limb is non-zero.
    /// The empty vector represents zero.
    limbs: Vec<u64>,
}

impl BigUint {
    /// The largest operand width the arithmetic here is willing to work on.
    ///
    /// The quadratic schoolbook algorithms below stay comfortably fast for the RSA key sizes
    /// this crate exists to serve (up to 8192 bits in practice), but their cost grows with the
    /// square of the operand width, so an attacker who controls a key embedded in a signed or
    /// encrypted document could otherwise buy arbitrary CPU time with a few kilobytes of
    /// base64. [`modpow`](BigUint::modpow) and [`div_rem`](BigUint::div_rem) refuse operands
    /// wider than this instead, and callers that accept untrusted key material should reject
    /// oversized moduli earlier still, where a domain-specific error can be reported.
    pub const MAX_BITS: usize = 16384;

    /// The value zero.
    #[inline]
    pub fn zero() -> Self {
        Self { limbs: Vec::new() }
    }

    /// The value one.
    #[inline]
    pub fn one() -> Self {
        Self { limbs: vec![1] }
    }

    /// Returns `true` if this value is zero.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.limbs.is_empty()
    }

    /// Returns `true` if this value is one.
    #[inline]
    pub fn is_one(&self) -> bool {
        matches!(self.limbs.as_slice(), [1])
    }

    /// Construct from a [`u64`].
    #[inline]
    pub fn from_u64(value: u64) -> Self {
        Self::from_limbs(vec![value])
    }

    /// Construct from a [`u128`].
    #[inline]
    pub fn from_u128(value: u128) -> Self {
        Self::from_limbs(vec![value as u64, (value >> 64) as u64])
    }

    /// Construct from a big-endian byte slice.
    ///
    /// Leading zero bytes are ignored; an empty slice yields zero.
    pub fn from_be_bytes(bytes: &[u8]) -> Self {
        let mut limbs = Vec::with_capacity(bytes.len() / 8 + 1);
        let mut end = bytes.len();
        while end >= 8 {
            let c = &bytes[end - 8..end];
            limbs.push(u64::from_be_bytes([
                c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7],
            ]));
            end -= 8;
        }
        if end > 0 {
            let mut buf = [0u8; 8];
            buf[8 - end..].copy_from_slice(&bytes[..end]);
            limbs.push(u64::from_be_bytes(buf));
        }
        Self::from_limbs(limbs)
    }

    /// Return the shortest big-endian byte encoding of this value.
    ///
    /// For zero this is a single `0x00` byte; for any other value there are no leading zero
    /// bytes. Use [`to_be_bytes_padded`](BigUint::to_be_bytes_padded) for a fixed width.
    pub fn to_be_bytes(&self) -> Vec<u8> {
        let raw = self.magnitude_be_bytes();
        if raw.is_empty() { vec![0] } else { raw }
    }

    /// Return a big-endian encoding of exactly `len` bytes, left-padded with zeros.
    ///
    /// Returns [`None`] if the value does not fit in `len` bytes. This is the encoding RSA needs
    /// for the modulus-width integer-to-octet-string conversion (`I2OSP`).
    pub fn to_be_bytes_padded(&self, len: usize) -> Option<Vec<u8>> {
        let raw = self.magnitude_be_bytes();
        if raw.len() > len {
            return None;
        }
        let mut out = vec![0u8; len];
        out[len - raw.len()..].copy_from_slice(&raw);
        Some(out)
    }

    /// Checked addition. Never returns [`None`] (an arbitrary-precision sum cannot overflow);
    /// it returns [`Option`] only to mirror the `checked_*` family for a uniform API.
    #[inline]
    pub fn checked_add(&self, other: &BigUint) -> Option<BigUint> {
        Some(Self::from_limbs(add_limbs(&self.limbs, &other.limbs)))
    }

    /// Checked subtraction. Returns [`None`] if `other` is greater than `self` (the unsigned
    /// result would be negative).
    #[inline]
    pub fn checked_sub(&self, other: &BigUint) -> Option<BigUint> {
        match cmp_limbs(&self.limbs, &other.limbs) {
            Ordering::Less => None,
            _ => Some(Self::from_limbs(sub_limbs(&self.limbs, &other.limbs))),
        }
    }

    /// Checked multiplication. Never returns [`None`] (an arbitrary-precision product cannot
    /// overflow); it returns [`Option`] only to mirror the `checked_*` family.
    #[inline]
    pub fn checked_mul(&self, other: &BigUint) -> Option<BigUint> {
        Some(Self::from_limbs(mul_limbs(&self.limbs, &other.limbs)))
    }

    /// Divide by `divisor`, returning `(quotient, remainder)` such that
    /// `self == quotient * divisor + remainder` and `remainder < divisor`.
    ///
    /// Returns [`None`] if `divisor` is zero, or if either operand is wider than
    /// [`MAX_BITS`](BigUint::MAX_BITS). Uses schoolbook binary long division.
    pub fn div_rem(&self, divisor: &BigUint) -> Option<(BigUint, BigUint)> {
        if divisor.is_zero() {
            return None;
        }
        if self.bit_len() > Self::MAX_BITS || divisor.bit_len() > Self::MAX_BITS {
            return None;
        }
        if cmp_limbs(&self.limbs, &divisor.limbs) == Ordering::Less {
            return Some((BigUint::zero(), self.clone()));
        }
        let bits = self.bit_len();
        let mut quotient = vec![0u64; self.limbs.len()];
        let mut remainder = Vec::<u64>::new();
        for i in (0..bits).rev() {
            shl1(&mut remainder);
            if self.bit(i) {
                if remainder.is_empty() {
                    remainder.push(1);
                } else {
                    remainder[0] |= 1;
                }
            }
            if cmp_limbs(&remainder, &divisor.limbs) != Ordering::Less {
                remainder = sub_limbs(&remainder, &divisor.limbs);
                normalize(&mut remainder);
                quotient[i / 64] |= 1u64 << (i % 64);
            }
        }
        Some((
            BigUint::from_limbs(quotient),
            BigUint::from_limbs(remainder),
        ))
    }

    /// Modular exponentiation: computes `self^exponent mod modulus`.
    ///
    /// Returns [`None`] if `modulus` is zero, or if any of the three operands is wider than
    /// [`MAX_BITS`](BigUint::MAX_BITS). Implemented with right-to-left square-and-multiply,
    /// reducing modulo `modulus` (via [`div_rem`](BigUint::div_rem)) after every multiply, so
    /// intermediate values never grow beyond twice the modulus width.
    pub fn modpow(&self, exponent: &BigUint, modulus: &BigUint) -> Option<BigUint> {
        if modulus.is_zero() {
            return None;
        }
        if self.bit_len() > Self::MAX_BITS
            || exponent.bit_len() > Self::MAX_BITS
            || modulus.bit_len() > Self::MAX_BITS
        {
            return None;
        }
        if modulus.is_one() {
            return Some(BigUint::zero());
        }
        let mut result = BigUint::one();
        let mut base = self.div_rem(modulus)?.1;
        let bits = exponent.bit_len();
        for i in 0..bits {
            if exponent.bit(i) {
                result = mul_mod(&result, &base, modulus)?;
            }
            // Avoid a redundant final squaring after consuming the top set bit.
            if i + 1 < bits {
                base = mul_mod(&base, &base, modulus)?;
            }
        }
        Some(result)
    }

    /// Modular addition: computes `(self + other) mod modulus`.
    ///
    /// Both operands are reduced modulo `modulus` before the addition, so neither of them has
    /// to be in range already.
    ///
    /// Returns [`None`] if `modulus` is zero, or if either operand or the modulus is wider than
    /// [`MAX_BITS`](BigUint::MAX_BITS).
    pub fn add_mod(&self, other: &BigUint, modulus: &BigUint) -> Option<BigUint> {
        let (a, b) = self.reduced_pair(other, modulus)?;
        // Both addends are below the modulus, so a single conditional subtraction reduces.
        let sum = a.checked_add(&b)?;
        Some(match sum.checked_sub(modulus) {
            Some(reduced) => reduced,
            None => sum,
        })
    }

    /// Modular subtraction: computes `(self - other) mod modulus`, wrapping into `0..modulus`
    /// instead of underflowing.
    ///
    /// Both operands are reduced modulo `modulus` before the subtraction, so neither of them
    /// has to be in range already.
    ///
    /// Returns [`None`] if `modulus` is zero, or if either operand or the modulus is wider than
    /// [`MAX_BITS`](BigUint::MAX_BITS).
    pub fn sub_mod(&self, other: &BigUint, modulus: &BigUint) -> Option<BigUint> {
        let (a, b) = self.reduced_pair(other, modulus)?;
        match a.checked_sub(&b) {
            Some(difference) => Some(difference),
            // `b` is below the modulus, so borrowing one multiple of it always suffices.
            None => a.checked_add(modulus)?.checked_sub(&b),
        }
    }

    /// Modular multiplication: computes `(self * other) mod modulus`.
    ///
    /// Both operands are reduced modulo `modulus` before the multiplication, so neither of them
    /// has to be in range already.
    ///
    /// Returns [`None`] if `modulus` is zero, if either operand or the modulus is wider than
    /// [`MAX_BITS`](BigUint::MAX_BITS), or if the intermediate product would be — which is the
    /// case for a modulus wider than half of [`MAX_BITS`](BigUint::MAX_BITS).
    pub fn mul_mod(&self, other: &BigUint, modulus: &BigUint) -> Option<BigUint> {
        let (a, b) = self.reduced_pair(other, modulus)?;
        Some(a.checked_mul(&b)?.div_rem(modulus)?.1)
    }

    /// The modular multiplicative inverse of `self` modulo `modulus`.
    ///
    /// Returns the unique value `x` in `0..modulus` satisfying `self * x ≡ 1 (mod modulus)`,
    /// computed with the extended Euclidean algorithm; every intermediate coefficient is kept
    /// reduced so no signed arithmetic is needed.
    ///
    /// Returns [`None`] when the inverse does not exist — that is when `gcd(self, modulus) != 1`,
    /// which includes `self ≡ 0 (mod modulus)` — when `modulus` is zero, or when an operand,
    /// the modulus, or an intermediate product would be wider than
    /// [`MAX_BITS`](BigUint::MAX_BITS) (the last of which happens for a modulus wider than half
    /// of [`MAX_BITS`](BigUint::MAX_BITS)).
    ///
    /// Like everything else in this module the algorithm is **variable time**: both the number
    /// of iterations and the cost of each one depend on the operands.
    pub fn mod_inverse(&self, modulus: &BigUint) -> Option<BigUint> {
        if modulus.is_zero() || modulus.bit_len() > Self::MAX_BITS {
            return None;
        }
        // The remainder sequence of the Euclidean algorithm, alongside the Bézout coefficient
        // of the modulus-free side; `coefficient` trails `next_coefficient` by one step.
        let mut remainder = modulus.clone();
        let mut next_remainder = self.div_rem(modulus)?.1;
        let mut coefficient = BigUint::zero();
        let mut next_coefficient = BigUint::one().div_rem(modulus)?.1;
        while !next_remainder.is_zero() {
            let (quotient, rest) = remainder.div_rem(&next_remainder)?;
            let product = mul_mod(&quotient, &next_coefficient, modulus)?;
            let updated = match coefficient.checked_sub(&product) {
                Some(difference) => difference,
                None => coefficient.checked_add(modulus)?.checked_sub(&product)?,
            };
            remainder = next_remainder;
            next_remainder = rest;
            coefficient = next_coefficient;
            next_coefficient = updated;
        }
        // The last non-zero remainder is `gcd(self, modulus)`; an inverse exists only if it is 1.
        if remainder.is_one() {
            Some(coefficient)
        } else {
            None
        }
    }

    // --- private helpers -------------------------------------------------------------------

    /// Reduce `self` and `other` modulo `modulus`, rejecting a zero or oversized modulus.
    ///
    /// [`div_rem`](BigUint::div_rem) already rejects oversized operands, so this is the single
    /// gate every modular operation above needs.
    fn reduced_pair(&self, other: &BigUint, modulus: &BigUint) -> Option<(BigUint, BigUint)> {
        if modulus.is_zero() || modulus.bit_len() > Self::MAX_BITS {
            return None;
        }
        Some((self.div_rem(modulus)?.1, other.div_rem(modulus)?.1))
    }

    /// Build from raw limbs and normalize.
    fn from_limbs(mut limbs: Vec<u64>) -> Self {
        normalize(&mut limbs);
        Self { limbs }
    }

    /// The shortest big-endian magnitude bytes (empty for zero, no leading zeros otherwise).
    fn magnitude_be_bytes(&self) -> Vec<u8> {
        match self.limbs.last() {
            None => Vec::new(),
            Some(&top) => {
                let mut out = Vec::with_capacity(self.limbs.len() * 8);
                let skip = top.leading_zeros() as usize / 8;
                out.extend_from_slice(&top.to_be_bytes()[skip..]);
                for &limb in self.limbs.iter().rev().skip(1) {
                    out.extend_from_slice(&limb.to_be_bytes());
                }
                out
            }
        }
    }

    /// Number of significant bits (0 for zero).
    ///
    /// Callers accepting untrusted key material use this to reject an oversized modulus or
    /// exponent before any arithmetic runs; see [`MAX_BITS`](BigUint::MAX_BITS).
    #[must_use]
    pub fn bit_len(&self) -> usize {
        match self.limbs.last() {
            None => 0,
            Some(&top) => self.limbs.len() * 64 - top.leading_zeros() as usize,
        }
    }

    /// The bit at position `index`, counting from 0 at the least significant end.
    ///
    /// Positions at or above [`bit_len`](BigUint::bit_len) read as `false`, so the caller may
    /// walk a fixed number of bits without a bounds check of its own.
    #[must_use]
    pub fn bit(&self, index: usize) -> bool {
        match self.limbs.get(index / 64) {
            Some(&limb) => (limb >> (index % 64)) & 1 == 1,
            None => false,
        }
    }
}

/// Reduce `a * b` modulo `m`. Returns [`None`] only if `m` is zero.
fn mul_mod(a: &BigUint, b: &BigUint, m: &BigUint) -> Option<BigUint> {
    let product = a.checked_mul(b)?;
    Some(product.div_rem(m)?.1)
}

/// Drop trailing (most-significant) zero limbs so the representation is canonical.
fn normalize(limbs: &mut Vec<u64>) {
    while limbs.last() == Some(&0) {
        limbs.pop();
    }
}

/// Shift a normalized little-endian limb vector left by one bit, in place.
fn shl1(limbs: &mut Vec<u64>) {
    let mut carry = 0u64;
    for limb in limbs.iter_mut() {
        let next_carry = *limb >> 63;
        *limb = (*limb << 1) | carry;
        carry = next_carry;
    }
    if carry != 0 {
        limbs.push(carry);
    }
}

/// Add two little-endian limb slices, returning a normalized limb vector.
fn add_limbs(a: &[u64], b: &[u64]) -> Vec<u64> {
    let n = a.len().max(b.len());
    let mut out = Vec::with_capacity(n + 1);
    let mut carry = 0u64;
    for i in 0..n {
        let ai = u128::from(a.get(i).copied().unwrap_or(0));
        let bi = u128::from(b.get(i).copied().unwrap_or(0));
        let sum = ai + bi + u128::from(carry);
        out.push(sum as u64);
        carry = (sum >> 64) as u64;
    }
    if carry != 0 {
        out.push(carry);
    }
    out
}

/// Subtract `b` from `a`, assuming `a >= b`. The caller must normalize the result.
fn sub_limbs(a: &[u64], b: &[u64]) -> Vec<u64> {
    let mut out = Vec::with_capacity(a.len());
    let mut borrow = 0i128;
    for (i, &ai) in a.iter().enumerate() {
        let bi = b.get(i).copied().unwrap_or(0);
        let mut diff = i128::from(ai) - i128::from(bi) - borrow;
        if diff < 0 {
            diff += 1i128 << 64;
            borrow = 1;
        } else {
            borrow = 0;
        }
        out.push(diff as u64);
    }
    out
}

/// Multiply two little-endian limb slices, returning a normalized limb vector.
fn mul_limbs(a: &[u64], b: &[u64]) -> Vec<u64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut out = vec![0u64; a.len() + b.len()];
    for (i, &ai) in a.iter().enumerate() {
        let mut carry = 0u64;
        for (j, &bj) in b.iter().enumerate() {
            let idx = i + j;
            let acc = u128::from(out[idx]) + u128::from(ai) * u128::from(bj) + u128::from(carry);
            out[idx] = acc as u64;
            carry = (acc >> 64) as u64;
        }
        out[i + b.len()] = carry;
    }
    out
}

/// Compare two normalized little-endian limb slices.
fn cmp_limbs(a: &[u64], b: &[u64]) -> Ordering {
    match a.len().cmp(&b.len()) {
        Ordering::Equal => {
            for (x, y) in a.iter().rev().zip(b.iter().rev()) {
                match x.cmp(y) {
                    Ordering::Equal => {}
                    non_equal => return non_equal,
                }
            }
            Ordering::Equal
        }
        non_equal => non_equal,
    }
}

impl Ord for BigUint {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        cmp_limbs(&self.limbs, &other.limbs)
    }
}

impl PartialOrd for BigUint {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl From<u64> for BigUint {
    #[inline]
    fn from(value: u64) -> Self {
        Self::from_u64(value)
    }
}

impl From<u128> for BigUint {
    #[inline]
    fn from(value: u128) -> Self {
        Self::from_u128(value)
    }
}

impl Add for &BigUint {
    type Output = BigUint;

    #[inline]
    fn add(self, rhs: &BigUint) -> BigUint {
        BigUint::from_limbs(add_limbs(&self.limbs, &rhs.limbs))
    }
}

impl Mul for &BigUint {
    type Output = BigUint;

    #[inline]
    fn mul(self, rhs: &BigUint) -> BigUint {
        BigUint::from_limbs(mul_limbs(&self.limbs, &rhs.limbs))
    }
}

impl Sub for &BigUint {
    type Output = BigUint;

    /// Subtract `rhs` from `self`.
    ///
    /// # Panics
    ///
    /// Panics if `rhs > self`. Use [`BigUint::checked_sub`] to handle underflow without a panic.
    #[inline]
    fn sub(self, rhs: &BigUint) -> BigUint {
        assert!(self >= rhs, "BigUint subtraction underflow");
        BigUint::from_limbs(sub_limbs(&self.limbs, &rhs.limbs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::{Rng, SplitMix64};

    fn decode_hex(hex: &str) -> Vec<u8> {
        hex.as_bytes()
            .chunks(2)
            .map(|pair| {
                let hi = char::from(pair[0]).to_digit(16).expect("hex digit") as u8;
                let lo = char::from(pair[1]).to_digit(16).expect("hex digit") as u8;
                (hi << 4) | lo
            })
            .collect()
    }

    fn encode_hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// Reference `u128` modular exponentiation for cross-checking.
    fn modpow_u128(mut base: u128, mut exp: u128, modulus: u128) -> u128 {
        let mut result = 1u128;
        base %= modulus;
        while exp > 0 {
            if exp & 1 == 1 {
                result = result * base % modulus;
            }
            exp >>= 1;
            base = base * base % modulus;
        }
        result
    }

    #[test]
    fn construction_and_bytes_round_trip() {
        assert!(BigUint::zero().is_zero());
        assert!(BigUint::one().is_one());
        assert_eq!(BigUint::from_u64(0), BigUint::zero());
        assert_eq!(BigUint::from_u128(1), BigUint::one());
        assert_eq!(BigUint::zero().to_be_bytes(), vec![0u8]);

        let value = BigUint::from_u128(0x0102_0304_0506_0708_090a_0b0c_0d0e_0f10);
        let bytes = value.to_be_bytes();
        assert_eq!(BigUint::from_be_bytes(&bytes), value);
        assert_eq!(bytes[0], 0x01);

        // Leading zero bytes are ignored on input.
        assert_eq!(BigUint::from_be_bytes(&[0, 0, 5]), BigUint::from_u64(5));
        assert_eq!(BigUint::from_be_bytes(&[]), BigUint::zero());
    }

    #[test]
    fn padded_bytes() {
        let value = BigUint::from_u64(0xABCD);
        assert_eq!(value.to_be_bytes_padded(4), Some(vec![0, 0, 0xAB, 0xCD]));
        assert_eq!(value.to_be_bytes_padded(2), Some(vec![0xAB, 0xCD]));
        assert_eq!(value.to_be_bytes_padded(1), None);
        assert_eq!(BigUint::zero().to_be_bytes_padded(3), Some(vec![0, 0, 0]));
    }

    #[test]
    fn ordering() {
        assert!(BigUint::from_u64(1) < BigUint::from_u64(2));
        assert!(BigUint::from_u128(1 << 64) > BigUint::from_u64(u64::MAX));
        assert_eq!(
            BigUint::from_u64(7).cmp(&BigUint::from_u64(7)),
            Ordering::Equal
        );
    }

    #[test]
    fn multiplication_carries() {
        // 2^64 * 2^64 = 2^128.
        let two_64 = BigUint::from_u128(1u128 << 64);
        let product = two_64.checked_mul(&two_64).expect("mul never fails");
        let mut expected = vec![0u8; 17];
        expected[0] = 1;
        assert_eq!(product.to_be_bytes(), expected);
    }

    #[test]
    fn subtraction_underflow() {
        assert_eq!(
            BigUint::from_u64(3).checked_sub(&BigUint::from_u64(10)),
            None
        );
        assert_eq!(
            BigUint::from_u64(10).checked_sub(&BigUint::from_u64(3)),
            Some(BigUint::from_u64(7))
        );
    }

    #[test]
    #[should_panic(expected = "BigUint subtraction underflow")]
    fn sub_operator_panics_on_underflow() {
        let _ = &BigUint::from_u64(1) - &BigUint::from_u64(2);
    }

    #[test]
    fn div_rem_basics() {
        assert_eq!(BigUint::from_u64(5).div_rem(&BigUint::zero()), None);
        let (q, r) = BigUint::from_u64(17)
            .div_rem(&BigUint::from_u64(5))
            .expect("nonzero divisor");
        assert_eq!(q, BigUint::from_u64(3));
        assert_eq!(r, BigUint::from_u64(2));
        // Divisor larger than dividend.
        let (q, r) = BigUint::from_u64(3)
            .div_rem(&BigUint::from_u64(10))
            .expect("nonzero divisor");
        assert_eq!(q, BigUint::zero());
        assert_eq!(r, BigUint::from_u64(3));
    }

    #[test]
    fn modpow_small() {
        // 4^13 mod 497 = 445.
        assert_eq!(
            BigUint::from_u64(4).modpow(&BigUint::from_u64(13), &BigUint::from_u64(497)),
            Some(BigUint::from_u64(445))
        );
        // Anything mod 1 is 0.
        assert_eq!(
            BigUint::from_u64(123).modpow(&BigUint::from_u64(456), &BigUint::one()),
            Some(BigUint::zero())
        );
        // Exponent zero yields 1.
        assert_eq!(
            BigUint::from_u64(9).modpow(&BigUint::zero(), &BigUint::from_u64(100)),
            Some(BigUint::one())
        );
        // Zero modulus is rejected.
        assert_eq!(
            BigUint::from_u64(2).modpow(&BigUint::from_u64(3), &BigUint::zero()),
            None
        );
    }

    /// Cross-check add / sub / mul / div_rem / modpow against native `u128` arithmetic over
    /// many deterministic random cases.
    #[test]
    fn arithmetic_matches_u128() {
        let mut rng = SplitMix64::new(0x0BAD_F00D_DEAD_BEEF);
        for _ in 0..20_000 {
            // 32-bit operands keep every product well within `u128`.
            let a = rng.next_u64() >> 32;
            let b = rng.next_u64() >> 32;
            let big_a = BigUint::from_u64(a);
            let big_b = BigUint::from_u64(b);

            assert_eq!(
                big_a.checked_add(&big_b).expect("add never fails"),
                BigUint::from_u128(u128::from(a) + u128::from(b))
            );
            assert_eq!(
                big_a.checked_mul(&big_b).expect("mul never fails"),
                BigUint::from_u128(u128::from(a) * u128::from(b))
            );

            let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
            assert_eq!(
                BigUint::from_u64(hi)
                    .checked_sub(&BigUint::from_u64(lo))
                    .expect("no underflow"),
                BigUint::from_u128(u128::from(hi - lo))
            );

            if let (Some(quotient), Some(remainder)) = (a.checked_div(b), a.checked_rem(b)) {
                let (q, r) = big_a.div_rem(&big_b).expect("nonzero divisor");
                assert_eq!(q, BigUint::from_u128(u128::from(quotient)));
                assert_eq!(r, BigUint::from_u128(u128::from(remainder)));
            }
        }
    }

    #[test]
    fn modpow_matches_u128() {
        let mut rng = SplitMix64::new(0x5EED_1234_ABCD_9999);
        for _ in 0..5_000 {
            // Keep the modulus below 2^24 so intermediate squarings fit `u128`.
            let base = rng.next_u64() >> 40;
            let exp = rng.next_u64() >> 52;
            let modulus = (rng.next_u64() >> 40) | 1;
            let got = BigUint::from_u64(base)
                .modpow(&BigUint::from_u64(exp), &BigUint::from_u64(modulus))
                .expect("nonzero modulus");
            let want = modpow_u128(u128::from(base), u128::from(exp), u128::from(modulus));
            assert_eq!(got, BigUint::from_u128(want));
        }
    }

    #[test]
    fn bit_accessor() {
        let value = BigUint::from_u128(0b1011 << 100);
        assert!(value.bit(100));
        assert!(value.bit(101));
        assert!(!value.bit(102));
        assert!(value.bit(103));
        // Everything at or above the bit length reads as zero, without a bounds check.
        assert!(!value.bit(104));
        assert!(!value.bit(100_000));
        assert!(!BigUint::zero().bit(0));
    }

    #[test]
    fn modular_helpers_small() {
        let m = BigUint::from_u64(97);
        let a = BigUint::from_u64(90);
        let b = BigUint::from_u64(20);
        assert_eq!(a.add_mod(&b, &m), Some(BigUint::from_u64(13)));
        assert_eq!(a.sub_mod(&b, &m), Some(BigUint::from_u64(70)));
        assert_eq!(b.sub_mod(&a, &m), Some(BigUint::from_u64(27)));
        // 90 * 20 = 1800 = 18 * 97 + 54.
        assert_eq!(a.mul_mod(&b, &m), Some(BigUint::from_u64(54)));

        // Operands out of range are reduced first.
        let large = BigUint::from_u64(97 * 3 + 5);
        assert_eq!(
            large.add_mod(&BigUint::zero(), &m),
            Some(BigUint::from_u64(5))
        );

        // A zero modulus is rejected everywhere.
        assert_eq!(a.add_mod(&b, &BigUint::zero()), None);
        assert_eq!(a.sub_mod(&b, &BigUint::zero()), None);
        assert_eq!(a.mul_mod(&b, &BigUint::zero()), None);
        assert_eq!(a.mod_inverse(&BigUint::zero()), None);
    }

    #[test]
    fn modular_helpers_match_u128() {
        let mut rng = SplitMix64::new(0x00C0_FFEE_0BAD_CAFE);
        for _ in 0..5_000 {
            // 32-bit operands keep every product well within `u128`.
            let a = rng.next_u64() >> 32;
            let b = rng.next_u64() >> 32;
            let m = (rng.next_u64() >> 32) | 1;
            let (big_a, big_b, big_m) = (
                BigUint::from_u64(a),
                BigUint::from_u64(b),
                BigUint::from_u64(m),
            );
            let (wide_a, wide_b, wide_m) = (u128::from(a), u128::from(b), u128::from(m));

            assert_eq!(
                big_a.add_mod(&big_b, &big_m),
                Some(BigUint::from_u128((wide_a + wide_b) % wide_m))
            );
            assert_eq!(
                big_a.sub_mod(&big_b, &big_m),
                Some(BigUint::from_u128(
                    (wide_a % wide_m + wide_m - wide_b % wide_m) % wide_m
                ))
            );
            assert_eq!(
                big_a.mul_mod(&big_b, &big_m),
                Some(BigUint::from_u128(wide_a * wide_b % wide_m))
            );
        }
    }

    #[test]
    fn mod_inverse_edge_cases() {
        // 3 * 5 = 15 = 1 (mod 7).
        assert_eq!(
            BigUint::from_u64(3).mod_inverse(&BigUint::from_u64(7)),
            Some(BigUint::from_u64(5))
        );
        // Zero is never invertible.
        assert_eq!(BigUint::zero().mod_inverse(&BigUint::from_u64(7)), None);
        // Nor is a value sharing a factor with the modulus.
        assert_eq!(
            BigUint::from_u64(2).mod_inverse(&BigUint::from_u64(4)),
            None
        );
        // Modulo one every value is congruent to zero, which is its own inverse.
        assert_eq!(
            BigUint::from_u64(5).mod_inverse(&BigUint::one()),
            Some(BigUint::zero())
        );
    }

    #[test]
    fn mod_inverse_round_trips() {
        let modulus = BigUint::from_u64(1_000_000_007);
        let mut rng = SplitMix64::new(0x1357_9BDF_2468_ACE0);
        for _ in 0..2_000 {
            let value = BigUint::from_u64(rng.next_u64() >> 24);
            let inverse = value
                .mod_inverse(&modulus)
                .expect("the modulus is prime and the value is never a multiple of it");
            assert_eq!(
                value.mul_mod(&inverse, &modulus),
                Some(BigUint::one()),
                "inverse of {value:?} is wrong"
            );
        }
    }

    #[test]
    fn modular_helpers_reject_oversized_operands() {
        // One bit past `MAX_BITS`, built from bytes so no arithmetic runs first.
        let huge = BigUint::from_be_bytes(&vec![0xFF; BigUint::MAX_BITS / 8 + 1]);
        assert!(huge.bit_len() > BigUint::MAX_BITS);
        let small = BigUint::from_u64(7);
        assert_eq!(huge.add_mod(&small, &small), None);
        assert_eq!(huge.sub_mod(&small, &small), None);
        assert_eq!(huge.mul_mod(&small, &small), None);
        assert_eq!(huge.mod_inverse(&small), None);
        assert_eq!(small.mod_inverse(&huge), None);
    }

    // A real RSA-2048 verification vector generated with OpenSSL/PyCA:
    // signing "OxiXML RSA-2048 modpow test vector" with PKCS#1 v1.5 + SHA-256.
    // `s^e mod n` must reproduce the EMSA-PKCS1-v1_5 encoded message `EM`.
    const RSA_N: &str = "b73c119d4e2c5e8c62c5e6191efae18192093865e22000af6006bbf917bcb71a05d92430315bdd683d387d1204aabbd0df04a6a223bb8dfd3927788f02ffd8f494e4dc4a3d2cdce85816d43f5f5fdcdf3e959d4006466391bd44e83bca43aeb135969938a485e54e224866866942857ce79d6f335046a143cede1e5b50c130f5039df09f2010c7601f19abdb47589959055b75a2ddd80f10cfc80701d0e2d6366432f52358d7769cdd3b8169d4cd01a7c83ec46c9886767ed6d40f4775c5f8354e0fb1b6171164f982a8927ec20b42a55db7318023956ea970f9caa673c52ed0d9bfc2b973e20ca7c2cd162a0f6e8b5b30a029b0a544bc6a03a06dfad10b2c75";
    const RSA_S: &str = "45e751fb76ced68d2575b38f4f249a5d808c93d080438f5c92729610520cb4b48bb057d6fd1e9cf793810bfa5d0cf02988971e683e267b0e3a9fdbc8f568f23d807eabe2c5bffc490afb601aa48e5b63556dbc989478de16d8697dabc1939f71ebea3bebfedbd287c4829ae584950659af98dfad92861f7e9b198d3a2b9accbdbc85a502c97e9b22423d7bfa3516902e26d8682c14ff1f40cbe549197d2198f9fd7335246489d03dabf2658762f3552e3dcd4f775d47f219ed068d0ea87941c11d292042cefdef2434a7a7459d8e153710b0aa297cfa02926f54fd0814b8fd64367b1480ad82d6373de1cb8fb6cdc9ec05d7eb7187a50c034826716d7ab33a6a";
    const RSA_EM: &str = "0001ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff003031300d060960864801650304020105000420479a987317c3652791670e8ec9d4ed74c6453fe7fb5799131a0f28d51aa1f274";

    #[test]
    fn rsa2048_modpow_reproduces_emsa_pkcs1() {
        let n = BigUint::from_be_bytes(&decode_hex(RSA_N));
        let s = BigUint::from_be_bytes(&decode_hex(RSA_S));
        let e = BigUint::from_u64(65_537);
        let m = s.modpow(&e, &n).expect("modulus is non-zero");
        let em = m.to_be_bytes_padded(256).expect("fits in 256 bytes");
        assert_eq!(encode_hex(&em), RSA_EM);
    }
}
