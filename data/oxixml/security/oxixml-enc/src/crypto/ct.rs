//! Data-independent (“constant-time”) primitives used wherever a secret is compared or a
//! secret-dependent choice is made.
//!
//! Rust gives no language-level guarantee that these compile to branch-free code, so
//! [`mask_eq`], [`mask_is_zero`], [`mask_le`] and [`select`] are written in the shape LLVM
//! reliably lowers to `cmov`/`and`/`or` — arithmetic on masks, never `if` — which is the
//! strongest guarantee obtainable without `unsafe` or inline assembly, both of which this
//! workspace forbids; it is documented rather than claimed. [`eq`] delegates to the shared
//! [`oxixml_support::bytes::ct_eq`] primitive that every crate in this workspace compares a
//! digest, MAC or padding check against, rather than keeping a second implementation here.
//!
//! # Example
//!
//! ```
//! use oxixml_enc::crypto::ct;
//!
//! assert!(ct::eq(b"tag", b"tag"));
//! assert!(!ct::eq(b"tag", b"tah"));
//! assert!(!ct::eq(b"tag", b"tagg")); // differing lengths are never equal
//! ```

/// Compares two octet strings without an early exit.
///
/// Slices of different lengths compare unequal, and the length check *is* observable — lengths
/// are public in every use this crate makes of the function (an AEAD tag length, a modulus
/// length). The contents are compared with a fold over the exclusive-or of every octet pair, so
/// the running time depends only on the (public) length. This is
/// [`oxixml_support::bytes::ct_eq`] under a name local to this crate's cipher code.
#[must_use]
pub fn eq(a: &[u8], b: &[u8]) -> bool {
    oxixml_support::bytes::ct_eq(a, b)
}

/// Returns `0xff` when `a == b` and `0x00` otherwise, without branching.
///
/// `a ^ b` is zero exactly when the octets are equal; subtracting one from the zero-extended
/// difference borrows into bit 8 only in that case, and the arithmetic shift smears that bit
/// across the whole mask.
#[must_use]
pub const fn mask_eq(a: u8, b: u8) -> u8 {
    let difference = (a ^ b) as u16;
    // `difference.wrapping_sub(1) >> 8` is 1 when `difference == 0`, and 0 otherwise.
    (((difference.wrapping_sub(1)) >> 8) & 1) as u8 * 0xff
}

/// Returns `0xff` when `value == 0` and `0x00` otherwise, without branching.
#[must_use]
pub const fn mask_is_zero(value: u8) -> u8 {
    mask_eq(value, 0)
}

/// Returns `0xff` when `a <= b` and `0x00` otherwise, for `usize` operands, without branching.
///
/// Both operands are widened to `u128` so the subtraction cannot wrap on any supported target;
/// the sign bit of `b - a` in the wide type is then the answer.
#[must_use]
pub const fn mask_le(a: usize, b: usize) -> u8 {
    let difference = (b as u128).wrapping_sub(a as u128);
    // No borrow out of bit 127 means `a <= b`.
    (((difference >> 127) ^ 1) as u8) * 0xff
}

/// Selects `on_true` when `mask` is `0xff` and `on_false` when it is `0x00`.
///
/// `mask` must be one of those two values; any other value yields a meaningless mixture rather
/// than a panic, which is the point — there is no branch to take.
#[must_use]
pub const fn select(mask: u8, on_true: u8, on_false: u8) -> u8 {
    (mask & on_true) | (!mask & on_false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eq_matches_slice_equality() {
        assert!(eq(b"", b""));
        assert!(eq(b"abc", b"abc"));
        assert!(!eq(b"abc", b"abd"));
        assert!(!eq(b"abc", b"ab"));
        assert!(!eq(b"", b"a"));
    }

    #[test]
    fn masks_are_all_ones_or_all_zeros() {
        for a in 0..=255u8 {
            for b in 0..=255u8 {
                let mask = mask_eq(a, b);
                assert!(mask == 0x00 || mask == 0xff);
                assert_eq!(mask == 0xff, a == b);
            }
        }
    }

    #[test]
    fn is_zero_mask() {
        assert_eq!(mask_is_zero(0), 0xff);
        for value in 1..=255u8 {
            assert_eq!(mask_is_zero(value), 0x00);
        }
    }

    #[test]
    fn le_mask() {
        assert_eq!(mask_le(0, 0), 0xff);
        assert_eq!(mask_le(1, 2), 0xff);
        assert_eq!(mask_le(2, 1), 0x00);
        assert_eq!(mask_le(usize::MAX, usize::MAX), 0xff);
        assert_eq!(mask_le(usize::MAX, 0), 0x00);
        assert_eq!(mask_le(0, usize::MAX), 0xff);
    }

    #[test]
    fn select_picks_the_masked_branch() {
        assert_eq!(select(0xff, 1, 2), 1);
        assert_eq!(select(0x00, 1, 2), 2);
    }
}
