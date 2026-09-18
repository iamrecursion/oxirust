//! `memchr`-class byte scanning implemented with safe word-at-a-time (SWAR) code.
//!
//! Parsers spend a large fraction of their time looking for the next delimiter byte
//! (`<`, `&`, `"`, whitespace, …). The classic accelerated primitive for that is `memchr`.
//! This module provides equivalents ([`memchr`], [`memchr2`], [`memchr3`], [`memrchr`]) that
//! process eight bytes at a time using [SWAR] arithmetic on [`u64`] words, without a single
//! line of `unsafe` and without pulling in an external crate.
//!
//! All functions return byte offsets as [`Option<usize>`], matching the shape of the
//! `memchr` crate's free functions, and are marked `#[inline]` so they fold into hot loops.
//!
//! [SWAR]: https://en.wikipedia.org/wiki/SWAR

/// Each byte set to `0x01`.
const LO: u64 = 0x0101_0101_0101_0101;
/// Each byte's low seven bits set (`0x7F`).
const LOW7: u64 = 0x7F7F_7F7F_7F7F_7F7F;

/// Broadcast a single byte into all eight lanes of a [`u64`].
#[inline]
const fn broadcast(byte: u8) -> u64 {
    LO.wrapping_mul(byte as u64)
}

/// Return a word whose byte lanes have their high bit (`0x80`) set **exactly** where `word` has
/// a zero byte, and are otherwise zero. To search for a byte `b`, callers XOR the input word
/// with `broadcast(b)` first so that matching lanes become zero.
///
/// Unlike the familiar `(x - LO) & !x & HI` "haszero" trick — which is only reliable for
/// *detecting the presence* of a zero byte and can flag the byte immediately above a zero due
/// to borrow propagation — this variant pinpoints each zero byte with no false positives, which
/// is required for the highest-bit search in [`memrchr`].
///
/// It works per byte: `(x & LOW7) + LOW7` sets a lane's high bit iff any of its low seven bits
/// is set; OR-ing with `x` then covers the high bit; the low seven bits are forced on with
/// `LOW7` and the whole thing complemented, leaving the high bit set iff every bit of the lane
/// was zero.
#[inline]
const fn zero_byte_mask(word: u64) -> u64 {
    let low_bits = (word & LOW7).wrapping_add(LOW7);
    !(low_bits | word | LOW7)
}

/// Load eight bytes starting at `slice[0]` as a little-endian [`u64`].
///
/// `slice` must be at least eight bytes long; the copy is exactly eight bytes so no bounds
/// check can ever fail and no `unsafe` is required.
#[inline]
fn load_u64_le(slice: &[u8]) -> u64 {
    let mut word = [0u8; 8];
    word.copy_from_slice(&slice[..8]);
    u64::from_le_bytes(word)
}

/// Find the first occurrence of `needle` in `haystack`, returning its byte offset.
///
/// Returns [`None`] if `needle` does not appear.
///
/// # Examples
///
/// ```
/// use oxixml_support::bytes::memchr;
///
/// assert_eq!(memchr(b'&', b"a&amp;b"), Some(1));
/// assert_eq!(memchr(b'z', b"abcdef"), None);
/// ```
#[inline]
pub fn memchr(needle: u8, haystack: &[u8]) -> Option<usize> {
    let broadcast_needle = broadcast(needle);
    let len = haystack.len();
    let mut offset = 0;
    while offset + 8 <= len {
        let word = load_u64_le(&haystack[offset..]);
        let mask = zero_byte_mask(word ^ broadcast_needle);
        if mask != 0 {
            return Some(offset + (mask.trailing_zeros() / 8) as usize);
        }
        offset += 8;
    }
    while offset < len {
        if haystack[offset] == needle {
            return Some(offset);
        }
        offset += 1;
    }
    None
}

/// Find the first occurrence of either `needle1` or `needle2` in `haystack`.
///
/// Returns the offset of whichever of the two bytes appears first, or [`None`] if neither does.
///
/// # Examples
///
/// ```
/// use oxixml_support::bytes::memchr2;
///
/// assert_eq!(memchr2(b'<', b'&', b"text&more<tag"), Some(4));
/// ```
#[inline]
pub fn memchr2(needle1: u8, needle2: u8, haystack: &[u8]) -> Option<usize> {
    let b1 = broadcast(needle1);
    let b2 = broadcast(needle2);
    let len = haystack.len();
    let mut offset = 0;
    while offset + 8 <= len {
        let word = load_u64_le(&haystack[offset..]);
        let mask = zero_byte_mask(word ^ b1) | zero_byte_mask(word ^ b2);
        if mask != 0 {
            return Some(offset + (mask.trailing_zeros() / 8) as usize);
        }
        offset += 8;
    }
    while offset < len {
        let byte = haystack[offset];
        if byte == needle1 || byte == needle2 {
            return Some(offset);
        }
        offset += 1;
    }
    None
}

/// Find the first occurrence of `needle1`, `needle2` or `needle3` in `haystack`.
///
/// Returns the offset of whichever of the three bytes appears first, or [`None`] if none do.
///
/// # Examples
///
/// ```
/// use oxixml_support::bytes::memchr3;
///
/// assert_eq!(memchr3(b'\'', b'"', b'=', b"attr=\"v\""), Some(4));
/// ```
#[inline]
pub fn memchr3(needle1: u8, needle2: u8, needle3: u8, haystack: &[u8]) -> Option<usize> {
    let b1 = broadcast(needle1);
    let b2 = broadcast(needle2);
    let b3 = broadcast(needle3);
    let len = haystack.len();
    let mut offset = 0;
    while offset + 8 <= len {
        let word = load_u64_le(&haystack[offset..]);
        let mask =
            zero_byte_mask(word ^ b1) | zero_byte_mask(word ^ b2) | zero_byte_mask(word ^ b3);
        if mask != 0 {
            return Some(offset + (mask.trailing_zeros() / 8) as usize);
        }
        offset += 8;
    }
    while offset < len {
        let byte = haystack[offset];
        if byte == needle1 || byte == needle2 || byte == needle3 {
            return Some(offset);
        }
        offset += 1;
    }
    None
}

/// Find the **last** occurrence of `needle` in `haystack`, returning its byte offset.
///
/// Returns [`None`] if `needle` does not appear.
///
/// # Examples
///
/// ```
/// use oxixml_support::bytes::memrchr;
///
/// assert_eq!(memrchr(b'/', b"a/b/c"), Some(3));
/// assert_eq!(memrchr(b'z', b"abc"), None);
/// ```
#[inline]
pub fn memrchr(needle: u8, haystack: &[u8]) -> Option<usize> {
    let broadcast_needle = broadcast(needle);
    let mut end = haystack.len();
    while end >= 8 {
        let word = load_u64_le(&haystack[end - 8..end]);
        let mask = zero_byte_mask(word ^ broadcast_needle);
        if mask != 0 {
            let highest_set = 63 - mask.leading_zeros();
            return Some(end - 8 + (highest_set / 8) as usize);
        }
        end -= 8;
    }
    while end > 0 {
        end -= 1;
        if haystack[end] == needle {
            return Some(end);
        }
    }
    None
}

/// Find the first byte of `haystack` that is **not** a member of `set`.
///
/// Membership is resolved through a 256-entry lookup table, so the cost is independent of the
/// size of `set`. Returns the offset of the first non-member byte, or [`None`] if every byte
/// of `haystack` belongs to `set` (including the case where `haystack` is empty).
///
/// # Examples
///
/// ```
/// use oxixml_support::bytes::find_byte_not_in;
///
/// // Skip leading ASCII digits.
/// assert_eq!(find_byte_not_in(b"0042abc", b"0123456789"), Some(4));
/// assert_eq!(find_byte_not_in(b"aaaa", b"a"), None);
/// ```
#[inline]
pub fn find_byte_not_in(haystack: &[u8], set: &[u8]) -> Option<usize> {
    let mut member = [false; 256];
    for &byte in set {
        member[byte as usize] = true;
    }
    haystack.iter().position(|&byte| !member[byte as usize])
}

/// Count the leading ASCII-whitespace bytes of `haystack`.
///
/// A byte is considered whitespace when [`u8::is_ascii_whitespace`] returns `true`, i.e. it is
/// one of space (`0x20`), horizontal tab (`0x09`), line feed (`0x0A`), form feed (`0x0C`) or
/// carriage return (`0x0D`). The returned value is the number of such bytes at the start of
/// `haystack`, which is also the offset of the first non-whitespace byte (or `haystack.len()`
/// when the slice is entirely whitespace or empty).
///
/// # Examples
///
/// ```
/// use oxixml_support::bytes::skip_ascii_whitespace;
///
/// assert_eq!(skip_ascii_whitespace(b"  \t\r\nvalue"), 5);
/// assert_eq!(&b"  \t\r\nvalue"[skip_ascii_whitespace(b"  \t\r\nvalue")..], b"value");
/// assert_eq!(skip_ascii_whitespace(b"value"), 0);
/// ```
#[inline]
pub fn skip_ascii_whitespace(haystack: &[u8]) -> usize {
    haystack
        .iter()
        .position(|byte| !byte.is_ascii_whitespace())
        .unwrap_or(haystack.len())
}

/// Compares two byte slices without letting the comparison's *duration* reveal where, or
/// whether, they first differ.
///
/// Use this instead of `==` whenever one side is a secret or a value an attacker is trying to
/// forge — a MAC tag, a digest being checked against a recomputed one, decrypted padding. The
/// ordinary slice comparison returns as soon as it finds a mismatching byte, which lets an
/// attacker who can time many attempts recover the expected value one byte at a time.
///
/// Slices of different lengths always compare unequal, and that length difference *is*
/// observable through timing: only the contents are protected. The loop reads both slices in
/// full and folds every byte into a single accumulator, so no branch and no early exit depends
/// on the data.
///
/// # Example
///
/// ```
/// use oxixml_support::bytes::ct_eq;
///
/// assert!(ct_eq(b"expected-tag", b"expected-tag"));
/// assert!(!ct_eq(b"expected-tag", b"forged-tag!!"));
/// assert!(!ct_eq(b"expected-tag", b"expected"));
/// ```
#[must_use]
pub fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut difference = 0u8;
    for (&left, &right) in a.iter().zip(b.iter()) {
        difference |= left ^ right;
    }
    difference == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::{Rng, SplitMix64};

    /// Reference implementation the SWAR code is checked against.
    fn naive_memchr(needle: u8, haystack: &[u8]) -> Option<usize> {
        haystack.iter().position(|&b| b == needle)
    }

    fn naive_memrchr(needle: u8, haystack: &[u8]) -> Option<usize> {
        haystack.iter().rposition(|&b| b == needle)
    }

    #[test]
    fn ct_eq_matches_slice_equality() {
        let mut rng = SplitMix64::new(0x0DDB_1A5E_5BAD_5EED);
        for len in 0..40usize {
            let left: Vec<u8> = (0..len).map(|_| rng.next_u64() as u8).collect();
            assert!(ct_eq(&left, &left));
            for index in 0..len {
                let mut right = left.clone();
                right[index] ^= 0x80;
                assert!(!ct_eq(&left, &right), "difference at {index} of {len}");
                assert_eq!(ct_eq(&left, &right), left == right);
            }
            assert!(!ct_eq(&left, &left[..len.saturating_sub(1)]) || len == 0);
        }
        assert!(ct_eq(b"", b""));
    }

    #[test]
    fn memchr_basic() {
        assert_eq!(memchr(b'a', b""), None);
        assert_eq!(memchr(b'a', b"a"), Some(0));
        assert_eq!(memchr(b'a', b"zzza"), Some(3));
        assert_eq!(memchr(b'a', b"zzzzzzzzzzzza"), Some(12));
        assert_eq!(memchr(b'q', b"abcdefghijklmnop"), None);
    }

    #[test]
    fn memrchr_basic() {
        assert_eq!(memrchr(b'a', b""), None);
        assert_eq!(memrchr(b'a', b"a"), Some(0));
        assert_eq!(memrchr(b'a', b"azzz"), Some(0));
        assert_eq!(memrchr(b'a', b"azzzzzzzzzzza"), Some(12));
        assert_eq!(memrchr(b'a', b"abcabcabc"), Some(6));
    }

    #[test]
    fn memchr2_and_memchr3() {
        assert_eq!(memchr2(b'x', b'y', b"........y...x"), Some(8));
        assert_eq!(memchr2(b'x', b'y', b"nomatchhere"), None);
        assert_eq!(memchr3(b'x', b'y', b'z', b"........z"), Some(8));
        assert_eq!(memchr3(b'x', b'y', b'z', b"aaaaaaaaaa"), None);
    }

    #[test]
    fn find_byte_not_in_and_whitespace() {
        assert_eq!(find_byte_not_in(b"", b"a"), None);
        assert_eq!(find_byte_not_in(b"aaab", b"a"), Some(3));
        assert_eq!(skip_ascii_whitespace(b""), 0);
        assert_eq!(skip_ascii_whitespace(b" \t\n\r\x0c"), 5);
        assert_eq!(skip_ascii_whitespace(b"\x0bx"), 0); // 0x0B is not ASCII whitespace
    }

    /// Property test: over many randomized buffers and needles, the SWAR functions must agree
    /// with the naive scalar implementations exactly. Fixed seeds keep the test deterministic.
    #[test]
    fn swar_matches_naive_over_random_buffers() {
        for seed in [1u64, 2, 3, 42, 0xDEAD_BEEF, 0x0123_4567_89AB_CDEF] {
            let mut rng = SplitMix64::new(seed);
            for _ in 0..2_000 {
                let len = rng.gen_range(0..40) as usize;
                let mut buffer = vec![0u8; len];
                // Restrict to a small alphabet so matches are common.
                for byte in buffer.iter_mut() {
                    *byte = b'a' + (rng.gen_range(0..6) as u8);
                }
                let needle = b'a' + (rng.gen_range(0..6) as u8);
                assert_eq!(
                    memchr(needle, &buffer),
                    naive_memchr(needle, &buffer),
                    "memchr mismatch (seed {seed}, needle {needle}, buf {buffer:?})"
                );
                assert_eq!(
                    memrchr(needle, &buffer),
                    naive_memrchr(needle, &buffer),
                    "memrchr mismatch (seed {seed}, needle {needle}, buf {buffer:?})"
                );
                let needle2 = b'a' + (rng.gen_range(0..6) as u8);
                let expected2 = buffer.iter().position(|&b| b == needle || b == needle2);
                assert_eq!(memchr2(needle, needle2, &buffer), expected2);
                let needle3 = b'a' + (rng.gen_range(0..6) as u8);
                let expected3 = buffer
                    .iter()
                    .position(|&b| b == needle || b == needle2 || b == needle3);
                assert_eq!(memchr3(needle, needle2, needle3, &buffer), expected3);
            }
        }
    }

    /// Property test using the full byte range and longer buffers to exercise the word loop.
    #[test]
    fn swar_matches_naive_full_byte_range() {
        let mut rng = SplitMix64::new(0xC0FF_EE00_1234_5678);
        for _ in 0..1_000 {
            let len = rng.gen_range(0..200) as usize;
            let mut buffer = vec![0u8; len];
            rng.fill_bytes(&mut buffer);
            let needle = rng.gen_range(0..256) as u8;
            assert_eq!(memchr(needle, &buffer), naive_memchr(needle, &buffer));
            assert_eq!(memrchr(needle, &buffer), naive_memrchr(needle, &buffer));
        }
    }
}
