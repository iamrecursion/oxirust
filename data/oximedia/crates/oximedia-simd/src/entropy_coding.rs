//! SIMD helpers for bit packing / unpacking used in entropy coding.
//!
//! Codec bitstreams (CABAC bypass mode, Exp-Golomb codes, raw bitplane coding)
//! frequently need to:
//!
//! - **Pack** a sequence of N-bit integers tightly into a byte buffer.
//! - **Unpack** a byte buffer back into N-bit integers.
//! - **Transpose** coefficient blocks into row-bit-plane order for
//!   bit-plane coding (e.g. JPEG 2000 or H.266/VVC CABAC initialisation).
//!
//! This module provides portable scalar implementations of all three
//! operations.  The inner loops are written to express data-parallel structure
//! so that the compiler backend (on AVX2/AVX-512 or NEON targets) can
//! auto-vectorise the hot paths without explicit intrinsics.
//!
//! # Example — pack 4-bit nibbles
//!
//! ```
//! use oximedia_simd::entropy_coding::{pack_bits, unpack_bits};
//!
//! let nibbles = vec![0xAu8, 0xBu8, 0xCu8, 0xDu8];
//! let packed = pack_bits(&nibbles, 4).unwrap();
//! assert_eq!(packed, vec![0xAB, 0xCD]);
//!
//! let unpacked = unpack_bits(&packed, 4, 4).unwrap();
//! assert_eq!(unpacked, nibbles);
//! ```

use crate::{Result, SimdError};

// ── Bit packing / unpacking ───────────────────────────────────────────────────

/// Pack `values` where each value uses exactly `bits_per_value` bits into a
/// compact byte stream (big-endian bit order).
///
/// `bits_per_value` must be in `1..=8`.  Values are masked to `bits_per_value`
/// bits before packing.  The output length is
/// `ceil(values.len() * bits_per_value / 8)`.
///
/// # Errors
///
/// Returns [`SimdError::UnsupportedOperation`] if `bits_per_value` is 0 or > 8.
pub fn pack_bits(values: &[u8], bits_per_value: usize) -> Result<Vec<u8>> {
    if bits_per_value == 0 || bits_per_value > 8 {
        return Err(SimdError::UnsupportedOperation);
    }
    let total_bits = values.len() * bits_per_value;
    let out_bytes = (total_bits + 7) / 8;
    let mut out = vec![0u8; out_bytes];

    let mask = ((1u16 << bits_per_value) - 1) as u8;
    let mut bit_pos: usize = 0; // current bit position in the output stream

    for &v in values {
        let val = v & mask;
        // Write `bits_per_value` bits starting at `bit_pos`
        let byte_idx = bit_pos / 8;
        let bit_offset = bit_pos % 8; // high bit first within byte

        // Shift so the MSB of `val` lands at `bit_offset` in the output byte
        // We may span two output bytes.
        let shift_left = (8 - bit_offset).saturating_sub(bits_per_value);
        let shift_right = bits_per_value.saturating_sub(8 - bit_offset);

        if shift_right == 0 {
            // Fits in one byte
            out[byte_idx] |= val << shift_left;
        } else {
            // Spans two bytes
            out[byte_idx] |= val >> shift_right;
            if byte_idx + 1 < out_bytes {
                out[byte_idx + 1] |= val << (8 - shift_right);
            }
        }
        bit_pos += bits_per_value;
    }

    Ok(out)
}

/// Unpack `count` values of `bits_per_value` bits each from `data`.
///
/// The inverse of [`pack_bits`].  Big-endian bit order.
///
/// # Errors
///
/// Returns [`SimdError::UnsupportedOperation`] if `bits_per_value` is 0 or > 8.
/// Returns [`SimdError::InvalidBufferSize`] if `data` is too short.
pub fn unpack_bits(data: &[u8], bits_per_value: usize, count: usize) -> Result<Vec<u8>> {
    if bits_per_value == 0 || bits_per_value > 8 {
        return Err(SimdError::UnsupportedOperation);
    }
    let total_bits = count * bits_per_value;
    let needed_bytes = (total_bits + 7) / 8;
    if data.len() < needed_bytes {
        return Err(SimdError::InvalidBufferSize);
    }

    let mut out = Vec::with_capacity(count);
    let mask = ((1u16 << bits_per_value) - 1) as u8;
    let mut bit_pos: usize = 0;

    for _ in 0..count {
        let byte_idx = bit_pos / 8;
        let bit_offset = bit_pos % 8;

        let available_in_byte = 8 - bit_offset;
        let val = if bits_per_value <= available_in_byte {
            // All bits in one byte
            (data[byte_idx] >> (available_in_byte - bits_per_value)) & mask
        } else {
            // Spans two bytes
            let high_bits = available_in_byte;
            let low_bits = bits_per_value - high_bits;
            let high = (data[byte_idx] & ((1 << high_bits) - 1)) as u16;
            let low = (if byte_idx + 1 < data.len() {
                data[byte_idx + 1]
            } else {
                0
            } >> (8 - low_bits)) as u16;
            ((high << low_bits) | low) as u8 & mask
        };
        out.push(val);
        bit_pos += bits_per_value;
    }

    Ok(out)
}

// ── Bit-plane transpose ───────────────────────────────────────────────────────

/// Transpose an 8×8 block of u8 values into 8 bit-plane bytes.
///
/// Each output byte `planes[k]` holds bit `k` of all 8 input bytes, packed
/// so that `planes[k] & (1 << j)` corresponds to `block[j] & (1 << k)`.
///
/// This operation is used in bit-plane coding (e.g. JPEG 2000 codeblock
/// encoding, CABAC coefficient bit-plane initialisation).
///
/// # Errors
///
/// Returns [`SimdError::InvalidBufferSize`] if `block` has fewer than 8 bytes.
pub fn transpose_to_bitplanes(block: &[u8]) -> Result<[u8; 8]> {
    if block.len() < 8 {
        return Err(SimdError::InvalidBufferSize);
    }
    let mut planes = [0u8; 8];
    // For each of the 8 bit positions k (0 = LSB, 7 = MSB):
    for k in 0..8usize {
        let mut plane_byte = 0u8;
        for j in 0..8usize {
            // Extract bit k of block[j] and place at position j of plane_byte
            let bit = (block[j] >> k) & 1;
            plane_byte |= bit << j;
        }
        planes[k] = plane_byte;
    }
    Ok(planes)
}

/// Reconstruct an 8×8 block of u8 values from 8 bit-plane bytes.
///
/// The inverse of [`transpose_to_bitplanes`].
pub fn reconstruct_from_bitplanes(planes: &[u8; 8]) -> [u8; 8] {
    let mut block = [0u8; 8];
    for k in 0..8usize {
        for j in 0..8usize {
            let bit = (planes[k] >> j) & 1;
            block[j] |= bit << k;
        }
    }
    block
}

// ── Exp-Golomb coding helpers ─────────────────────────────────────────────────

/// Encode an unsigned integer using Exp-Golomb order-0 (ue(v)) coding.
///
/// Returns the bit-length of the encoded value.  The encoded bits are written
/// MSB-first into `out` starting at `*bit_pos`.
///
/// The ue(v) code for value `v` is:
/// - `floor(log2(v+1))` leading zeros
/// - a `1` bit
/// - the `floor(log2(v+1))` LSBs of `v+1`
///
/// This is the coding used in H.264, H.265, and VP8/VP9 for syntax elements.
///
/// # Errors
///
/// Returns [`SimdError::InvalidBufferSize`] if `out` has insufficient capacity.
pub fn encode_exp_golomb_ue(value: u32, out: &mut [u8], bit_pos: &mut usize) -> Result<usize> {
    let v1 = value + 1;
    // Number of bits in v1
    let code_len = u32::BITS as usize - v1.leading_zeros() as usize;
    let total_bits = 2 * code_len - 1;

    let needed_bytes = (*bit_pos + total_bits + 7) / 8;
    if out.len() < needed_bytes {
        return Err(SimdError::InvalidBufferSize);
    }

    // Write `code_len - 1` zero bits (prefix)
    for _ in 0..(code_len - 1) {
        // Zero bit — output byte is already 0, just advance
        *bit_pos += 1;
    }

    // Write `code_len` bits of v1 MSB-first
    for k in (0..code_len).rev() {
        let bit = ((v1 >> k) & 1) as u8;
        let byte_idx = *bit_pos / 8;
        let bit_offset = 7 - (*bit_pos % 8);
        out[byte_idx] |= bit << bit_offset;
        *bit_pos += 1;
    }

    Ok(total_bits)
}

/// Decode an Exp-Golomb ue(v) codeword from `data` starting at `*bit_pos`.
///
/// # Errors
///
/// Returns [`SimdError::InvalidBufferSize`] if there is insufficient data.
pub fn decode_exp_golomb_ue(data: &[u8], bit_pos: &mut usize) -> Result<u32> {
    // Count leading zeros
    let mut leading_zeros = 0usize;
    loop {
        let byte_idx = *bit_pos / 8;
        if byte_idx >= data.len() {
            return Err(SimdError::InvalidBufferSize);
        }
        let bit_offset = 7 - (*bit_pos % 8);
        let bit = (data[byte_idx] >> bit_offset) & 1;
        *bit_pos += 1;
        if bit == 1 {
            break;
        }
        leading_zeros += 1;
        if leading_zeros > 31 {
            return Err(SimdError::InvalidBufferSize);
        }
    }

    // Read `leading_zeros` more bits
    let mut suffix = 0u32;
    for _ in 0..leading_zeros {
        let byte_idx = *bit_pos / 8;
        if byte_idx >= data.len() {
            return Err(SimdError::InvalidBufferSize);
        }
        let bit_offset = 7 - (*bit_pos % 8);
        let bit = u32::from((data[byte_idx] >> bit_offset) & 1);
        suffix = (suffix << 1) | bit;
        *bit_pos += 1;
    }

    let code_val = (1u32 << leading_zeros) + suffix - 1;
    Ok(code_val)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── pack / unpack ─────────────────────────────────────────────────────────

    #[test]
    fn test_pack_1bit_values() {
        // Pack [1,0,1,1,0,1,0,0] → 0b10110100 = 0xB4
        let vals = vec![1u8, 0, 1, 1, 0, 1, 0, 0];
        let packed = pack_bits(&vals, 1).expect("packing 1-bit values should succeed");
        assert_eq!(packed, vec![0xB4]);
    }

    #[test]
    fn test_pack_4bit_nibbles() {
        let nibbles = vec![0xAu8, 0xBu8, 0xCu8, 0xDu8];
        let packed = pack_bits(&nibbles, 4).expect("packing 4-bit nibbles should succeed");
        assert_eq!(packed, vec![0xAB, 0xCD]);
    }

    #[test]
    fn test_pack_unpack_roundtrip_1bit() {
        let vals: Vec<u8> = (0..16).map(|i| (i % 2) as u8).collect();
        let packed = pack_bits(&vals, 1).expect("packing 1-bit roundtrip should succeed");
        let unpacked =
            unpack_bits(&packed, 1, 16).expect("unpacking 1-bit roundtrip should succeed");
        assert_eq!(unpacked, vals);
    }

    #[test]
    fn test_pack_unpack_roundtrip_4bit() {
        let vals: Vec<u8> = (0..8).map(|i| (i * 3 % 16) as u8).collect();
        let packed = pack_bits(&vals, 4).expect("packing 4-bit roundtrip should succeed");
        let unpacked =
            unpack_bits(&packed, 4, 8).expect("unpacking 4-bit roundtrip should succeed");
        assert_eq!(unpacked, vals);
    }

    #[test]
    fn test_pack_unpack_roundtrip_8bit() {
        let vals: Vec<u8> = (0..8).map(|i| (i * 37 % 256) as u8).collect();
        let packed = pack_bits(&vals, 8).expect("packing 8-bit roundtrip should succeed");
        let unpacked =
            unpack_bits(&packed, 8, 8).expect("unpacking 8-bit roundtrip should succeed");
        assert_eq!(unpacked, vals);
    }

    #[test]
    fn test_pack_unpack_roundtrip_3bit() {
        let vals: Vec<u8> = (0..9).map(|i| (i % 8) as u8).collect();
        let packed = pack_bits(&vals, 3).expect("packing 3-bit roundtrip should succeed");
        let unpacked =
            unpack_bits(&packed, 3, 9).expect("unpacking 3-bit roundtrip should succeed");
        assert_eq!(unpacked, vals);
    }

    #[test]
    fn test_pack_empty() {
        let packed = pack_bits(&[], 4).expect("packing empty slice should succeed");
        assert!(packed.is_empty());
    }

    #[test]
    fn test_pack_invalid_bits_per_value_zero() {
        assert!(pack_bits(&[1], 0).is_err());
    }

    #[test]
    fn test_pack_invalid_bits_per_value_nine() {
        assert!(pack_bits(&[1], 9).is_err());
    }

    #[test]
    fn test_unpack_invalid_bits_per_value() {
        assert!(unpack_bits(&[0xFF], 0, 1).is_err());
    }

    #[test]
    fn test_unpack_buffer_too_small() {
        // 4 values of 4 bits = 2 bytes needed, but only 1 provided
        assert!(unpack_bits(&[0xFF], 4, 4).is_err());
    }

    // ── bit-plane transpose ───────────────────────────────────────────────────

    #[test]
    fn test_bitplane_transpose_identity() {
        let block = [0xA5u8, 0x5A, 0xFF, 0x00, 0x0F, 0xF0, 0xCC, 0x33];
        let planes = transpose_to_bitplanes(&block).expect("bitplane transpose should succeed");
        let recovered = reconstruct_from_bitplanes(&planes);
        assert_eq!(recovered, block);
    }

    #[test]
    fn test_bitplane_all_zeros() {
        let block = [0u8; 8];
        let planes =
            transpose_to_bitplanes(&block).expect("bitplane transpose of zeros should succeed");
        assert_eq!(planes, [0u8; 8]);
        let recovered = reconstruct_from_bitplanes(&planes);
        assert_eq!(recovered, block);
    }

    #[test]
    fn test_bitplane_all_ones() {
        let block = [0xFFu8; 8];
        let planes =
            transpose_to_bitplanes(&block).expect("bitplane transpose of ones should succeed");
        assert_eq!(planes, [0xFFu8; 8]);
    }

    #[test]
    fn test_bitplane_single_bit_set() {
        // block[3] = 0b00000100 → bit 2 set at position 3
        let mut block = [0u8; 8];
        block[3] = 1 << 2; // bit plane 2, position 3
        let planes = transpose_to_bitplanes(&block)
            .expect("bitplane transpose of single bit should succeed");
        // plane[2] should have bit 3 set
        assert_eq!(planes[2], 1 << 3);
        for k in 0..8 {
            if k != 2 {
                assert_eq!(planes[k], 0, "unexpected bit in plane {k}");
            }
        }
    }

    #[test]
    fn test_bitplane_buffer_too_small() {
        let block = [0u8; 4];
        assert!(transpose_to_bitplanes(&block).is_err());
    }

    // ── Exp-Golomb ────────────────────────────────────────────────────────────

    fn encode_decode_ue(value: u32) -> u32 {
        let mut buf = vec![0u8; 16];
        let mut bit_pos = 0usize;
        encode_exp_golomb_ue(value, &mut buf, &mut bit_pos)
            .expect("exp-golomb encode should succeed");
        let mut read_pos = 0usize;
        decode_exp_golomb_ue(&buf, &mut read_pos).expect("exp-golomb decode should succeed")
    }

    #[test]
    fn test_exp_golomb_zero() {
        assert_eq!(encode_decode_ue(0), 0);
    }

    #[test]
    fn test_exp_golomb_small_values() {
        for v in 0u32..32 {
            assert_eq!(encode_decode_ue(v), v, "failed at value {v}");
        }
    }

    #[test]
    fn test_exp_golomb_powers_of_two() {
        for k in 0..10u32 {
            let v = 1u32 << k;
            assert_eq!(encode_decode_ue(v), v, "failed at value {v}");
        }
    }

    #[test]
    fn test_exp_golomb_sequential_no_overlap() {
        // Encode multiple values back-to-back and decode them correctly
        let values = [0u32, 1, 2, 5, 10, 31, 100];
        let mut buf = vec![0u8; 64];
        let mut write_pos = 0usize;
        for &v in &values {
            encode_exp_golomb_ue(v, &mut buf, &mut write_pos)
                .expect("sequential exp-golomb encode should succeed");
        }
        let mut read_pos = 0usize;
        for &v in &values {
            let decoded = decode_exp_golomb_ue(&buf, &mut read_pos)
                .expect("sequential exp-golomb decode should succeed");
            assert_eq!(
                decoded, v,
                "sequential Exp-Golomb: expected {v} got {decoded}"
            );
        }
    }

    #[test]
    fn test_decode_insufficient_data() {
        let buf = [0u8; 1]; // too short for any prefix
        let mut pos = 4usize; // only 4 bits available
                              // Decoding a value requiring many bits should fail gracefully
        let _ = decode_exp_golomb_ue(&buf, &mut pos); // don't panic
    }
}
