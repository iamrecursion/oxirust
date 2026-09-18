#![allow(dead_code)]
//! Bitwise SIMD-style operations for multimedia processing.
//!
//! Provides vectorized bitwise operations that model hardware SIMD
//! instructions in pure scalar Rust:
//! - Parallel bit shifts on byte/word lanes
//! - Population count (popcount) across vectors
//! - Bit masking, extraction, and packing
//! - Byte reversal and endianness conversion
//!
//! These are fundamental building blocks for entropy coding,
//! palette indexing, and bitstream packing in video codecs.

/// Apply a left shift of `shift` bits to every element of a `u8` slice in place.
///
/// Bits shifted out are discarded (standard wrapping-free behavior).
pub fn shift_left_u8(data: &mut [u8], shift: u32) {
    let shift = shift.min(8);
    for val in data.iter_mut() {
        *val = val.wrapping_shl(shift);
    }
}

/// Apply a right shift of `shift` bits to every element of a `u8` slice in place.
///
/// Bits shifted out are discarded (logical shift, not arithmetic).
pub fn shift_right_u8(data: &mut [u8], shift: u32) {
    let shift = shift.min(8);
    for val in data.iter_mut() {
        *val = val.wrapping_shr(shift);
    }
}

/// Apply a left shift of `shift` bits to every element of a `u16` slice in place.
///
/// Bits shifted out are discarded.
pub fn shift_left_u16(data: &mut [u16], shift: u32) {
    let shift = shift.min(16);
    for val in data.iter_mut() {
        *val = val.wrapping_shl(shift);
    }
}

/// Apply a right shift of `shift` bits to every element of a `u16` slice in place.
///
/// Bits shifted out are discarded.
pub fn shift_right_u16(data: &mut [u16], shift: u32) {
    let shift = shift.min(16);
    for val in data.iter_mut() {
        *val = val.wrapping_shr(shift);
    }
}

/// Compute the population count (number of set bits) for every element.
///
/// Returns a new vector with the popcount for each byte.
#[must_use]
pub fn popcount_u8(data: &[u8]) -> Vec<u8> {
    data.iter().map(|v| v.count_ones() as u8).collect()
}

/// Compute the population count for every `u32` element.
///
/// Returns a new vector with the popcount for each element.
#[must_use]
pub fn popcount_u32(data: &[u32]) -> Vec<u32> {
    data.iter().map(|v| v.count_ones()).collect()
}

/// Total population count across all bytes in the slice.
///
/// Returns the sum of all set bits.
#[must_use]
pub fn total_popcount_u8(data: &[u8]) -> u64 {
    data.iter().map(|v| u64::from(v.count_ones())).sum()
}

/// Bitwise AND of two u8 slices element-wise. The output length is `min(a.len(), b.len())`.
///
/// Returns a new vector with the results.
#[must_use]
pub fn and_u8(a: &[u8], b: &[u8]) -> Vec<u8> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x & y).collect()
}

/// Bitwise OR of two u8 slices element-wise. The output length is `min(a.len(), b.len())`.
///
/// Returns a new vector with the results.
#[must_use]
pub fn or_u8(a: &[u8], b: &[u8]) -> Vec<u8> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x | y).collect()
}

/// Bitwise XOR of two u8 slices element-wise. The output length is `min(a.len(), b.len())`.
///
/// Returns a new vector with the results.
#[must_use]
pub fn xor_u8(a: &[u8], b: &[u8]) -> Vec<u8> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x ^ y).collect()
}

/// Bitwise NOT of every element in a `u8` slice.
///
/// Returns a new vector with each byte inverted.
#[must_use]
pub fn not_u8(data: &[u8]) -> Vec<u8> {
    data.iter().map(|&v| !v).collect()
}

/// AND-NOT (ANDN): `!a & b` for each pair of u8 elements.
///
/// This is equivalent to the x86 ANDN instruction.
///
/// Returns a new vector with the results.
#[must_use]
pub fn andn_u8(a: &[u8], b: &[u8]) -> Vec<u8> {
    a.iter().zip(b.iter()).map(|(&x, &y)| !x & y).collect()
}

/// Reverse the bits in each byte.
///
/// Returns a new vector with each byte's bits reversed.
#[must_use]
pub fn reverse_bits_u8(data: &[u8]) -> Vec<u8> {
    data.iter().map(|v| v.reverse_bits()).collect()
}

/// Swap the byte order (endianness) of each `u16` element.
///
/// Returns a new vector with swapped bytes.
#[must_use]
pub fn byte_swap_u16(data: &[u16]) -> Vec<u16> {
    data.iter().map(|v| v.swap_bytes()).collect()
}

/// Swap the byte order (endianness) of each `u32` element.
///
/// Returns a new vector with swapped bytes.
#[must_use]
pub fn byte_swap_u32(data: &[u32]) -> Vec<u32> {
    data.iter().map(|v| v.swap_bytes()).collect()
}

/// Extract a bit field from each u8: `(val >> start) & ((1 << width) - 1)`.
///
/// Extracts `width` bits starting at bit position `start` from each byte.
/// If `start + width > 8`, it wraps.
///
/// Returns a new vector with the extracted bit fields.
#[must_use]
pub fn extract_bits_u8(data: &[u8], start: u32, width: u32) -> Vec<u8> {
    let mask = if width >= 8 { 0xFF } else { (1u8 << width) - 1 };
    data.iter()
        .map(|&v| (v.wrapping_shr(start)) & mask)
        .collect()
}

/// Pack the low nibble (4 bits) of each byte into packed pairs.
///
/// Two consecutive bytes are packed into one: `low_nibble(data[2i]) | (low_nibble(data[2i+1]) << 4)`.
/// If `data.len()` is odd, the last nibble is stored in the low 4 bits of the final byte.
///
/// Returns the packed bytes.
#[must_use]
pub fn pack_nibbles(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len().div_ceil(2));
    let mut i = 0;
    while i + 1 < data.len() {
        let lo = data[i] & 0x0F;
        let hi = data[i + 1] & 0x0F;
        out.push(lo | (hi << 4));
        i += 2;
    }
    if i < data.len() {
        out.push(data[i] & 0x0F);
    }
    out
}

/// Unpack nibble-packed bytes back into individual bytes.
///
/// Each packed byte produces two output bytes (low nibble first, high nibble second).
///
/// Returns the unpacked bytes.
#[must_use]
pub fn unpack_nibbles(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 2);
    for &b in data {
        out.push(b & 0x0F);
        out.push((b >> 4) & 0x0F);
    }
    out
}

/// Count leading zeros for each u8 element.
///
/// Returns a new vector with the CLZ count for each byte.
#[must_use]
pub fn leading_zeros_u8(data: &[u8]) -> Vec<u8> {
    data.iter().map(|v| v.leading_zeros() as u8).collect()
}

/// Count trailing zeros for each u8 element.
///
/// Returns a new vector with the CTZ count for each byte.
#[must_use]
pub fn trailing_zeros_u8(data: &[u8]) -> Vec<u8> {
    data.iter().map(|v| v.trailing_zeros() as u8).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shift_left_u8() {
        let mut data = vec![0b0000_0001, 0b0000_1111, 0b1000_0000];
        shift_left_u8(&mut data, 2);
        assert_eq!(data, vec![0b0000_0100, 0b0011_1100, 0b0000_0000]);
    }

    #[test]
    fn test_shift_right_u8() {
        let mut data = vec![0b1000_0000, 0b1111_0000, 0b0000_0001];
        shift_right_u8(&mut data, 4);
        assert_eq!(data, vec![0b0000_1000, 0b0000_1111, 0b0000_0000]);
    }

    #[test]
    fn test_shift_left_u16() {
        let mut data = vec![1u16, 0xFF];
        shift_left_u16(&mut data, 8);
        assert_eq!(data, vec![256, 0xFF00]);
    }

    #[test]
    fn test_shift_right_u16() {
        let mut data = vec![0xFF00u16, 256];
        shift_right_u16(&mut data, 8);
        assert_eq!(data, vec![0xFF, 1]);
    }

    #[test]
    fn test_popcount_u8() {
        let data = vec![0b1010_1010, 0b1111_1111, 0b0000_0000, 0b0000_0001];
        let result = popcount_u8(&data);
        assert_eq!(result, vec![4, 8, 0, 1]);
    }

    #[test]
    fn test_popcount_u32() {
        let data = vec![0u32, 1, 0xFFFF_FFFF, 0b1010_1010];
        let result = popcount_u32(&data);
        assert_eq!(result, vec![0, 1, 32, 4]);
    }

    #[test]
    fn test_total_popcount_u8() {
        let data = vec![0xFF, 0x00, 0x0F];
        assert_eq!(total_popcount_u8(&data), 12);
    }

    #[test]
    fn test_and_or_xor_not() {
        let a = vec![0xFF, 0x0F, 0xAA];
        let b = vec![0x0F, 0xF0, 0x55];
        assert_eq!(and_u8(&a, &b), vec![0x0F, 0x00, 0x00]);
        assert_eq!(or_u8(&a, &b), vec![0xFF, 0xFF, 0xFF]);
        assert_eq!(xor_u8(&a, &b), vec![0xF0, 0xFF, 0xFF]);
        assert_eq!(not_u8(&[0x00, 0xFF, 0x0F]), vec![0xFF, 0x00, 0xF0]);
    }

    #[test]
    fn test_andn_u8() {
        let a = vec![0xFF, 0x0F];
        let b = vec![0xAB, 0xCD];
        // !0xFF = 0x00; 0x00 & 0xAB = 0x00
        // !0x0F = 0xF0; 0xF0 & 0xCD = 0xC0
        assert_eq!(andn_u8(&a, &b), vec![0x00, 0xC0]);
    }

    #[test]
    fn test_reverse_bits_u8() {
        let data = vec![0b1000_0000, 0b0000_0001, 0b1010_0101];
        let result = reverse_bits_u8(&data);
        assert_eq!(result, vec![0b0000_0001, 0b1000_0000, 0b1010_0101]);
    }

    #[test]
    fn test_byte_swap_u16() {
        let data = vec![0x0102u16, 0xAABB];
        let result = byte_swap_u16(&data);
        assert_eq!(result, vec![0x0201, 0xBBAA]);
    }

    #[test]
    fn test_byte_swap_u32() {
        let data = vec![0x0102_0304u32];
        let result = byte_swap_u32(&data);
        assert_eq!(result, vec![0x0403_0201]);
    }

    #[test]
    fn test_extract_bits_u8() {
        // Extract bits [2..4) (2 bits starting at position 2)
        let data = vec![0b0011_1100];
        let result = extract_bits_u8(&data, 2, 2);
        assert_eq!(result, vec![0b0000_0011]);
    }

    #[test]
    fn test_pack_unpack_nibbles() {
        let original = vec![0x0A, 0x0B, 0x0C, 0x0D, 0x0E];
        let packed = pack_nibbles(&original);
        assert_eq!(packed.len(), 3);
        assert_eq!(packed[0], 0xBA); // 0x0A | (0x0B << 4)
        assert_eq!(packed[1], 0xDC); // 0x0C | (0x0D << 4)
        assert_eq!(packed[2], 0x0E); // last nibble alone
    }

    #[test]
    fn test_unpack_nibbles() {
        let packed = vec![0xBA, 0xDC];
        let unpacked = unpack_nibbles(&packed);
        assert_eq!(unpacked, vec![0x0A, 0x0B, 0x0C, 0x0D]);
    }

    #[test]
    fn test_leading_zeros_u8() {
        let data = vec![0x00, 0x01, 0x80, 0x0F, 0xFF];
        let result = leading_zeros_u8(&data);
        assert_eq!(result, vec![8, 7, 0, 4, 0]);
    }

    #[test]
    fn test_trailing_zeros_u8() {
        let data = vec![0x00, 0x01, 0x80, 0x10, 0xFF];
        let result = trailing_zeros_u8(&data);
        assert_eq!(result, vec![8, 0, 7, 4, 0]);
    }
}
