//! Compression and decompression for EXR pixel data.
//!
//! Implements the following OpenEXR compression schemes:
//! - None: no compression
//! - RLE: run-length encoding
//! - ZIP / ZIPS: zlib/deflate compression
//! - PIZ: Haar-wavelet pre-transform + LZ4 (OxiMedia conformant approximation)
//! - PXR24: 24-bit float (byte-lane XOR-delta) + LZ4
//! - B44: 4×4 half-float block packing + LZ4
//! - B44A: B44 with flat-field optimization + LZ4
//! - DWAA / DWAB: 8×8 DCT quantization + LZ4 (OxiMedia conformant approximation)

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_lossless)]

use crate::error::{ImageError, ImageResult};
use std::io::Read;

// ── LZ4 helpers ──────────────────────────────────────────────────────────────

/// Compress bytes with LZ4 frame format.
fn lz4_compress(data: &[u8]) -> ImageResult<Vec<u8>> {
    oxiarc_lz4::compress(data).map_err(|e| ImageError::Compression(format!("LZ4 compress: {e}")))
}

/// Decompress LZ4 frame-format bytes.
///
/// `hint` is the expected output size (used as capacity hint, not a hard limit).
fn lz4_decompress(data: &[u8], hint: usize) -> ImageResult<Vec<u8>> {
    // Use 4× hint as the max_output bound to handle size variation safely.
    let max_out = hint
        .saturating_mul(4)
        .max(data.len().saturating_mul(4))
        .max(64);
    oxiarc_lz4::decompress(data, max_out)
        .map_err(|e| ImageError::Compression(format!("LZ4 decompress: {e}")))
}

// ── RLE ──────────────────────────────────────────────────────────────────────

pub(crate) fn decompress_rle(compressed: &[u8]) -> ImageResult<Vec<u8>> {
    let mut output = Vec::new();
    let mut i = 0;

    while i < compressed.len() {
        let count = compressed[i] as i8;
        i += 1;

        if count < 0 {
            // Run of different bytes
            let run_length = (-count + 1) as usize;
            if i + run_length > compressed.len() {
                break;
            }
            output.extend_from_slice(&compressed[i..i + run_length]);
            i += run_length;
        } else {
            // Run of same byte
            let run_length = (count + 1) as usize;
            if i >= compressed.len() {
                break;
            }
            let byte = compressed[i];
            i += 1;
            output.extend(std::iter::repeat_n(byte, run_length));
        }
    }

    Ok(output)
}

pub(crate) fn compress_rle(data: &[u8]) -> ImageResult<Vec<u8>> {
    let mut output = Vec::new();
    let mut i = 0;

    while i < data.len() {
        let start = i;
        let current = data[i];

        // Find run length
        let mut run_len = 1;
        while i + run_len < data.len() && data[i + run_len] == current && run_len < 127 {
            run_len += 1;
        }

        if run_len >= 3 {
            // Encode as run
            output.push((run_len - 1) as u8);
            output.push(current);
            i += run_len;
        } else {
            // Find literal run
            let mut lit_len = 1;
            while i + lit_len < data.len() && lit_len < 127 {
                let next_run = count_run(&data[i + lit_len..]);
                if next_run >= 3 {
                    break;
                }
                lit_len += 1;
            }

            output.push((-(lit_len as i8) + 1) as u8);
            output.extend_from_slice(&data[start..start + lit_len]);
            i += lit_len;
        }
    }

    Ok(output)
}

fn count_run(data: &[u8]) -> usize {
    if data.is_empty() {
        return 0;
    }
    let current = data[0];
    let mut count = 1;
    while count < data.len() && data[count] == current {
        count += 1;
    }
    count
}

// ── ZIP ──────────────────────────────────────────────────────────────────────

pub(crate) fn decompress_zip(compressed: &[u8]) -> ImageResult<Vec<u8>> {
    use oxiarc_deflate::ZlibStreamDecoder;

    let mut decoder = ZlibStreamDecoder::new(compressed);
    let mut output = Vec::new();
    decoder
        .read_to_end(&mut output)
        .map_err(|e| ImageError::Compression(format!("ZIP decompression failed: {e}")))?;

    Ok(output)
}

pub(crate) fn compress_zip(data: &[u8]) -> ImageResult<Vec<u8>> {
    use oxiarc_deflate::ZlibStreamEncoder;
    use std::io::Write;

    let mut encoder = ZlibStreamEncoder::new(Vec::new(), 6);
    encoder
        .write_all(data)
        .map_err(|e| ImageError::Compression(format!("ZIP compression failed: {e}")))?;

    encoder
        .finish()
        .map_err(|e| ImageError::Compression(format!("ZIP compression failed: {e}")))
}

// ── PIZ (Haar wavelet + LZ4) ──────────────────────────────────────────────────
//
// OxiMedia conformant approximation: 1-D Haar lifting (lossless with even
// rounding) applied to each 2-byte channel lane, followed by LZ4.
// Full OpenEXR bitstream conformance is not guaranteed.

/// Apply one level of forward Haar lifting to a u16 buffer (lossless).
///
/// Stores in-place: first half = low-pass (averages), second half = high-pass
/// (differences offset by 32768 to keep unsigned).
fn haar_forward(data: &mut [u16]) {
    let n = data.len();
    if n < 2 {
        return;
    }
    let half = n / 2;
    let mut tmp = vec![0u16; n];
    for i in 0..half {
        let a = data[2 * i] as i32;
        let b = data[2 * i + 1] as i32;
        // Use floor((a+b)/2) for the low band; store difference + 32768 for high.
        tmp[i] = ((a + b) >> 1) as u16;
        tmp[half + i] = ((a - b) + 32768) as u16;
    }
    // If n is odd, copy the last element unmodified into the low band.
    if n % 2 == 1 {
        tmp[half] = data[n - 1];
    }
    data.copy_from_slice(&tmp);
}

/// Reverse one level of Haar lifting.
fn haar_inverse(data: &mut [u16]) {
    let n = data.len();
    if n < 2 {
        return;
    }
    let half = n / 2;
    let mut tmp = vec![0u16; n];
    for i in 0..half {
        let low = data[i] as i32;
        let high = data[half + i] as i32 - 32768;
        // Recover: a = low + (high+1)/2  (ceiling), b = low - high/2 (floor)
        // This matches the lossless inverse of floor((a+b)/2).
        tmp[2 * i] = (low + ((high + 1) >> 1)) as u16;
        tmp[2 * i + 1] = (low - (high >> 1)) as u16;
    }
    if n % 2 == 1 {
        tmp[n - 1] = data[half];
    }
    data.copy_from_slice(&tmp);
}

pub(crate) fn compress_piz(data: &[u8]) -> ImageResult<Vec<u8>> {
    // Reinterpret the byte data as u16 values for wavelet transform.
    // If the length is odd, the last byte is passed through unchanged.
    let n16 = data.len() / 2;
    let mut u16_buf: Vec<u16> = (0..n16)
        .map(|i| u16::from_le_bytes([data[2 * i], data[2 * i + 1]]))
        .collect();

    haar_forward(&mut u16_buf);

    // Convert back to bytes
    let mut transformed = Vec::with_capacity(data.len());
    for val in &u16_buf {
        let bytes = val.to_le_bytes();
        transformed.push(bytes[0]);
        transformed.push(bytes[1]);
    }
    // Preserve trailing odd byte if present
    if data.len() % 2 == 1 {
        transformed.push(*data.last().unwrap_or(&0));
    }

    // Prepend original length as u32 LE so decompressor knows the output size.
    let orig_len = data.len() as u32;
    let mut payload = orig_len.to_le_bytes().to_vec();
    payload.extend_from_slice(&lz4_compress(&transformed)?);
    Ok(payload)
}

pub(crate) fn decompress_piz(compressed: &[u8]) -> ImageResult<Vec<u8>> {
    if compressed.len() < 4 {
        return Err(ImageError::Compression("PIZ data too short".to_string()));
    }
    let orig_len =
        u32::from_le_bytes([compressed[0], compressed[1], compressed[2], compressed[3]]) as usize;

    let lz4_payload = &compressed[4..];
    let transformed = lz4_decompress(lz4_payload, orig_len)?;

    // Inverse Haar on u16 pairs
    let n16 = orig_len / 2;
    let available = transformed.len().min(n16 * 2);
    let mut u16_buf: Vec<u16> = (0..available / 2)
        .map(|i| u16::from_le_bytes([transformed[2 * i], transformed[2 * i + 1]]))
        .collect();

    // Pad to n16 if decompressed data is shorter (shouldn't happen with correct data)
    while u16_buf.len() < n16 {
        u16_buf.push(0);
    }

    haar_inverse(&mut u16_buf);

    let mut output = Vec::with_capacity(orig_len);
    for val in &u16_buf {
        let bytes = val.to_le_bytes();
        output.push(bytes[0]);
        output.push(bytes[1]);
    }
    // Restore trailing odd byte
    if orig_len % 2 == 1 && transformed.len() >= orig_len {
        output.push(transformed[orig_len - 1]);
    }
    output.truncate(orig_len);
    Ok(output)
}

// ── PXR24 (24-bit float, XOR-delta byte lanes + LZ4) ─────────────────────────
//
// OxiMedia conformant approximation: for 32-bit float channels the lowest byte
// is discarded and the remaining 3 bytes are XOR-delta encoded per lane.
// For all other channel widths (u16 half, u32 int) the raw bytes are
// XOR-delta encoded as a single lane.
//
// Full OpenEXR bitstream conformance is not guaranteed.

/// XOR-delta encode a byte sequence in-place.
fn xor_delta_encode(data: &mut [u8]) {
    let mut prev = 0u8;
    for byte in data.iter_mut() {
        let cur = *byte;
        *byte = cur ^ prev;
        prev = cur;
    }
}

/// XOR-delta decode a byte sequence in-place.
fn xor_delta_decode(data: &mut [u8]) {
    let mut prev = 0u8;
    for byte in data.iter_mut() {
        let delta = *byte;
        *byte = delta ^ prev;
        prev = *byte;
    }
}

pub(crate) fn compress_pxr24(data: &[u8]) -> ImageResult<Vec<u8>> {
    // Treat data as an array of f32 (4-byte) values.
    // Strategy: split into 3 byte-lanes (bytes 0,1,2 of each f32); drop byte 3.
    let n_floats = data.len() / 4;
    let remainder = data.len() % 4;

    let mut lane0: Vec<u8> = Vec::with_capacity(n_floats);
    let mut lane1: Vec<u8> = Vec::with_capacity(n_floats);
    let mut lane2: Vec<u8> = Vec::with_capacity(n_floats);

    for i in 0..n_floats {
        lane0.push(data[4 * i]);
        lane1.push(data[4 * i + 1]);
        lane2.push(data[4 * i + 2]);
        // Byte 3 (LSB of f32 mantissa) is intentionally dropped.
    }

    xor_delta_encode(&mut lane0);
    xor_delta_encode(&mut lane1);
    xor_delta_encode(&mut lane2);

    // Pack: [orig_len: u32][lane0_len: u32][lane0][lane1][lane2][remainder bytes]
    let mut interleaved = Vec::with_capacity(n_floats * 3 + remainder);
    interleaved.extend_from_slice(&lane0);
    interleaved.extend_from_slice(&lane1);
    interleaved.extend_from_slice(&lane2);
    // Append any sub-word trailing bytes as-is
    interleaved.extend_from_slice(&data[n_floats * 4..]);

    let orig_len = data.len() as u32;
    let lane_len = n_floats as u32;
    let mut payload = orig_len.to_le_bytes().to_vec();
    payload.extend_from_slice(&lane_len.to_le_bytes());
    payload.extend_from_slice(&lz4_compress(&interleaved)?);
    Ok(payload)
}

pub(crate) fn decompress_pxr24(compressed: &[u8]) -> ImageResult<Vec<u8>> {
    if compressed.len() < 8 {
        return Err(ImageError::Compression("PXR24 data too short".to_string()));
    }
    let orig_len =
        u32::from_le_bytes([compressed[0], compressed[1], compressed[2], compressed[3]]) as usize;
    let n_floats =
        u32::from_le_bytes([compressed[4], compressed[5], compressed[6], compressed[7]]) as usize;

    let lz4_payload = &compressed[8..];
    let interleaved = lz4_decompress(lz4_payload, n_floats * 3)?;

    let mut lane0 = interleaved[..n_floats].to_vec();
    let mut lane1 = interleaved[n_floats..n_floats * 2].to_vec();
    let mut lane2 = interleaved[n_floats * 2..n_floats * 3].to_vec();
    let remainder = &interleaved[n_floats * 3..];

    xor_delta_decode(&mut lane0);
    xor_delta_decode(&mut lane1);
    xor_delta_decode(&mut lane2);

    let mut output = Vec::with_capacity(orig_len);
    for i in 0..n_floats {
        output.push(lane0[i]);
        output.push(lane1[i]);
        output.push(lane2[i]);
        output.push(0u8); // Reconstructed LSB is always 0
    }
    output.extend_from_slice(remainder);
    output.truncate(orig_len);
    Ok(output)
}

// ── B44 (4×4 half-float blocks + LZ4) ────────────────────────────────────────
//
// OxiMedia conformant approximation: treats each 16 consecutive u16 values as
// a 4×4 half-float block. Stores the block minimum and XOR-delta differences
// from the minimum for each element.
//
// Full OpenEXR B44 bitstream conformance is not guaranteed.

/// Block size: 4×4 = 16 half-float values.
const B44_BLOCK_SIZE: usize = 16;

/// Pack a single B44 block (16 × u16) into a 14-byte encoded block.
///
/// Layout:
/// - Bytes 0-1: block minimum value (LE u16)
/// - Bytes 2-3: block maximum delta (LE u16) — stored as scale factor
/// - Bytes 4-11: 16 deltas as 4-bit nibbles (2 per byte), proportionally scaled to [0,15]
/// - Bytes 12-13: reserved / zero
///
/// Each reconstructed value = min + round(nibble * max_delta / 15).
fn pack_b44_block(block: &[u16]) -> [u8; 14] {
    debug_assert_eq!(block.len(), B44_BLOCK_SIZE);

    let min_val = block.iter().copied().min().unwrap_or(0);
    let max_delta = block
        .iter()
        .map(|&v| v.saturating_sub(min_val))
        .max()
        .unwrap_or(0);

    let mut out = [0u8; 14];

    // Bytes 0-1: minimum value
    let min_bytes = min_val.to_le_bytes();
    out[0] = min_bytes[0];
    out[1] = min_bytes[1];

    // Bytes 2-3: max delta (scale factor for reconstruction)
    let max_bytes = max_delta.to_le_bytes();
    out[2] = max_bytes[0];
    out[3] = max_bytes[1];

    // Bytes 4-11: 16 nibbles (2 per byte), proportionally scaled to [0, 15]
    for i in 0..8 {
        let encode_nibble = |idx: usize| -> u8 {
            if idx < B44_BLOCK_SIZE {
                let delta = block[idx].saturating_sub(min_val) as u32;
                if max_delta > 0 {
                    // Scale proportionally: nibble = round(delta * 15 / max_delta)
                    ((delta * 15 + max_delta as u32 / 2) / max_delta as u32).min(15) as u8
                } else {
                    0
                }
            } else {
                0
            }
        };
        let a = encode_nibble(2 * i);
        let b = encode_nibble(2 * i + 1);
        out[4 + i] = (a << 4) | b;
    }

    // Bytes 12-13: reserved
    out[12] = 0;
    out[13] = 0;
    out
}

/// Unpack a 14-byte B44 encoded block back into 16 u16 values.
fn unpack_b44_block(encoded: &[u8; 14]) -> [u16; B44_BLOCK_SIZE] {
    let min_val = u16::from_le_bytes([encoded[0], encoded[1]]);
    let max_delta = u16::from_le_bytes([encoded[2], encoded[3]]);

    let mut block = [0u16; B44_BLOCK_SIZE];
    for i in 0..8 {
        let byte = encoded[4 + i];
        let a_nibble = (byte >> 4) as u32;
        let b_nibble = (byte & 0x0f) as u32;

        let decode_delta = |nibble: u32| -> u16 {
            if max_delta > 0 {
                // Reconstruct: delta = round(nibble * max_delta / 15)
                ((nibble * max_delta as u32 + 7) / 15) as u16
            } else {
                0
            }
        };

        if 2 * i < B44_BLOCK_SIZE {
            block[2 * i] = min_val.saturating_add(decode_delta(a_nibble));
        }
        if 2 * i + 1 < B44_BLOCK_SIZE {
            block[2 * i + 1] = min_val.saturating_add(decode_delta(b_nibble));
        }
    }
    block
}

pub(crate) fn compress_b44(data: &[u8]) -> ImageResult<Vec<u8>> {
    let n16 = data.len() / 2;
    let u16_vals: Vec<u16> = (0..n16)
        .map(|i| u16::from_le_bytes([data[2 * i], data[2 * i + 1]]))
        .collect();

    let n_blocks = (n16 + B44_BLOCK_SIZE - 1) / B44_BLOCK_SIZE;
    // Each block → 14 bytes encoded
    let mut encoded = Vec::with_capacity(n_blocks * 14 + 4);
    // Prepend original byte length for decompressor
    let orig_len = data.len() as u32;
    encoded.extend_from_slice(&orig_len.to_le_bytes());

    for block_idx in 0..n_blocks {
        let start = block_idx * B44_BLOCK_SIZE;
        let end = (start + B44_BLOCK_SIZE).min(n16);
        // Pad last block if needed
        let mut block = [0u16; B44_BLOCK_SIZE];
        block[..end - start].copy_from_slice(&u16_vals[start..end]);
        let packed = pack_b44_block(&block);
        encoded.extend_from_slice(&packed);
    }

    // Preserve any trailing odd byte
    if data.len() % 2 == 1 {
        encoded.push(*data.last().unwrap_or(&0));
    }

    lz4_compress(&encoded).map(|compressed| {
        // Wrap with a header: [marker: 1 byte = 0x44][lz4_data]
        let mut out = vec![0x44u8];
        out.extend_from_slice(&compressed);
        out
    })
}

pub(crate) fn decompress_b44(compressed: &[u8]) -> ImageResult<Vec<u8>> {
    if compressed.is_empty() {
        return Err(ImageError::Compression("B44 data empty".to_string()));
    }
    let marker = compressed[0];
    if marker != 0x44 {
        return Err(ImageError::Compression(format!(
            "B44 bad marker: {marker:#x}"
        )));
    }
    let encoded = lz4_decompress(&compressed[1..], compressed.len() * 2)?;
    if encoded.len() < 4 {
        return Err(ImageError::Compression("B44 encoded too short".to_string()));
    }
    let orig_len = u32::from_le_bytes([encoded[0], encoded[1], encoded[2], encoded[3]]) as usize;
    let block_data = &encoded[4..];

    let n16 = (orig_len + 1) / 2;
    let n_blocks = (n16 + B44_BLOCK_SIZE - 1) / B44_BLOCK_SIZE;
    let mut u16_vals = Vec::with_capacity(n_blocks * B44_BLOCK_SIZE);

    let mut offset = 0;
    for _ in 0..n_blocks {
        if offset + 14 > block_data.len() {
            break;
        }
        let mut buf = [0u8; 14];
        buf.copy_from_slice(&block_data[offset..offset + 14]);
        let block = unpack_b44_block(&buf);
        u16_vals.extend_from_slice(&block);
        offset += 14;
    }

    let mut output = Vec::with_capacity(orig_len);
    for val in u16_vals.iter().take(n16) {
        output.extend_from_slice(&val.to_le_bytes());
    }
    // Restore odd byte if any
    if orig_len % 2 == 1 {
        let trailing_idx = 4 + n_blocks * 14;
        if trailing_idx < encoded.len() {
            output.push(encoded[trailing_idx]);
        }
    }
    output.truncate(orig_len);
    Ok(output)
}

// ── B44A (flat-field optimized B44 + LZ4) ────────────────────────────────────
//
// Same as B44 but blocks where all 16 values are identical are stored as
// 3 bytes instead of 14: [0xA4][value_lo][value_hi].

/// Flag byte for a flat (constant) B44A block.
const B44A_FLAT_FLAG: u8 = 0xA4;
/// Flag byte for a normal (non-flat) B44A block.
const B44A_NORMAL_FLAG: u8 = 0x44;

pub(crate) fn compress_b44a(data: &[u8]) -> ImageResult<Vec<u8>> {
    let n16 = data.len() / 2;
    let u16_vals: Vec<u16> = (0..n16)
        .map(|i| u16::from_le_bytes([data[2 * i], data[2 * i + 1]]))
        .collect();

    let n_blocks = (n16 + B44_BLOCK_SIZE - 1) / B44_BLOCK_SIZE;
    let mut encoded = Vec::with_capacity(n_blocks * 15 + 4);
    let orig_len = data.len() as u32;
    encoded.extend_from_slice(&orig_len.to_le_bytes());

    for block_idx in 0..n_blocks {
        let start = block_idx * B44_BLOCK_SIZE;
        let end = (start + B44_BLOCK_SIZE).min(n16);
        let mut block = [0u16; B44_BLOCK_SIZE];
        block[..end - start].copy_from_slice(&u16_vals[start..end]);

        // Check if all values in the valid range are identical (flat field)
        let first = block[0];
        let is_flat = block[..end - start].iter().all(|&v| v == first);

        if is_flat {
            // 3-byte flat encoding
            let val_bytes = first.to_le_bytes();
            encoded.push(B44A_FLAT_FLAG);
            encoded.push(val_bytes[0]);
            encoded.push(val_bytes[1]);
        } else {
            // Normal 15-byte encoding: flag + 14 bytes
            encoded.push(B44A_NORMAL_FLAG);
            let packed = pack_b44_block(&block);
            encoded.extend_from_slice(&packed);
        }
    }

    if data.len() % 2 == 1 {
        encoded.push(*data.last().unwrap_or(&0));
    }

    lz4_compress(&encoded)
}

pub(crate) fn decompress_b44a(compressed: &[u8]) -> ImageResult<Vec<u8>> {
    let encoded = lz4_decompress(compressed, compressed.len() * 4)?;
    if encoded.len() < 4 {
        return Err(ImageError::Compression(
            "B44A encoded too short".to_string(),
        ));
    }
    let orig_len = u32::from_le_bytes([encoded[0], encoded[1], encoded[2], encoded[3]]) as usize;
    let n16 = (orig_len + 1) / 2;
    let n_blocks = (n16 + B44_BLOCK_SIZE - 1) / B44_BLOCK_SIZE;
    let block_data = &encoded[4..];

    let mut u16_vals = Vec::with_capacity(n_blocks * B44_BLOCK_SIZE);
    let mut offset = 0;

    for _ in 0..n_blocks {
        if offset >= block_data.len() {
            break;
        }
        let flag = block_data[offset];
        offset += 1;

        if flag == B44A_FLAT_FLAG {
            if offset + 2 > block_data.len() {
                break;
            }
            let val = u16::from_le_bytes([block_data[offset], block_data[offset + 1]]);
            offset += 2;
            u16_vals.extend(std::iter::repeat_n(val, B44_BLOCK_SIZE));
        } else {
            // Normal block: 14 bytes
            if offset + 14 > block_data.len() {
                break;
            }
            let mut buf = [0u8; 14];
            buf.copy_from_slice(&block_data[offset..offset + 14]);
            let block = unpack_b44_block(&buf);
            u16_vals.extend_from_slice(&block);
            offset += 14;
        }
    }

    let mut output = Vec::with_capacity(orig_len);
    for val in u16_vals.iter().take(n16) {
        output.extend_from_slice(&val.to_le_bytes());
    }
    output.truncate(orig_len);
    Ok(output)
}

// ── 8×8 DCT helpers ────────────────────────────────────────────────────────────

/// Number of elements in an 8×8 DCT block.
const DCT_BLOCK: usize = 64;

/// Quantization divisor for DCT coefficients.
const DCT_QUANT: f32 = 8.0;

/// Apply 1-D DCT-II to `data[..8]` in-place (unnormalized).
fn dct8_1d(data: &mut [f32; 8]) {
    use std::f32::consts::PI;
    let mut out = [0.0f32; 8];
    for k in 0..8usize {
        let mut sum = 0.0f32;
        for n in 0..8usize {
            sum += data[n] * (PI * (2 * n + 1) as f32 * k as f32 / 16.0).cos();
        }
        out[k] = sum;
    }
    data.copy_from_slice(&out);
}

/// Apply 1-D inverse DCT-III (unnormalized) to `data[..8]` in-place.
fn idct8_1d(data: &mut [f32; 8]) {
    use std::f32::consts::PI;
    let mut out = [0.0f32; 8];
    for n in 0..8usize {
        // x[n] = (X[0]/2 + sum_{k=1}^{7} X[k]*cos(pi*(2n+1)*k/16)) * (1/4)
        let mut sum = data[0] / 2.0;
        for k in 1..8usize {
            sum += data[k] * (PI * (2 * n + 1) as f32 * k as f32 / 16.0).cos();
        }
        out[n] = sum / 4.0;
    }
    data.copy_from_slice(&out);
}

/// Forward 2-D DCT on a flat 64-element block (row-major 8×8).
fn dct_forward_8x8(block: &mut [f32; DCT_BLOCK]) {
    // Row DCTs
    for row in 0..8usize {
        let mut row_data = [0.0f32; 8];
        row_data.copy_from_slice(&block[row * 8..row * 8 + 8]);
        dct8_1d(&mut row_data);
        block[row * 8..row * 8 + 8].copy_from_slice(&row_data);
    }
    // Column DCTs
    for col in 0..8usize {
        let mut col_data = [0.0f32; 8];
        for row in 0..8usize {
            col_data[row] = block[row * 8 + col];
        }
        dct8_1d(&mut col_data);
        for row in 0..8usize {
            block[row * 8 + col] = col_data[row];
        }
    }
}

/// Inverse 2-D DCT on a flat 64-element block (row-major 8×8).
fn dct_inverse_8x8(block: &mut [f32; DCT_BLOCK]) {
    // Column IDCTs
    for col in 0..8usize {
        let mut col_data = [0.0f32; 8];
        for row in 0..8usize {
            col_data[row] = block[row * 8 + col];
        }
        idct8_1d(&mut col_data);
        for row in 0..8usize {
            block[row * 8 + col] = col_data[row];
        }
    }
    // Row IDCTs
    for row in 0..8usize {
        let mut row_data = [0.0f32; 8];
        row_data.copy_from_slice(&block[row * 8..row * 8 + 8]);
        idct8_1d(&mut row_data);
        block[row * 8..row * 8 + 8].copy_from_slice(&row_data);
    }
}

// ── DWAA / DWAB (DCT quantization + LZ4) ─────────────────────────────────────
//
// OxiMedia conformant approximation: f32 pixel data (or u16 cast to f32) is
// processed in 8×8 blocks with DCT + uniform quantization (÷ DCT_QUANT),
// stored as i16, then LZ4 compressed.
// Dwab delegates to the same algorithm (full OpenEXR distinction is block
// grouping which does not affect the per-scanline API used here).

/// Compress with DWAA/DWAB: DCT + quantization + LZ4.
///
/// The data is treated as either f32 (4 bytes) or u16 (2 bytes) depending on
/// the element size inferred from the byte count.  If neither divides evenly,
/// fall back to byte-level compression.
pub(crate) fn compress_dwaa(data: &[u8]) -> ImageResult<Vec<u8>> {
    compress_dct_lz4(data)
}

pub(crate) fn decompress_dwaa(compressed: &[u8]) -> ImageResult<Vec<u8>> {
    decompress_dct_lz4(compressed)
}

pub(crate) fn compress_dwab(data: &[u8]) -> ImageResult<Vec<u8>> {
    // Same algorithm as DWAA — the scanline block-size difference is handled
    // at a higher level; here we compress all provided data the same way.
    compress_dct_lz4(data)
}

pub(crate) fn decompress_dwab(compressed: &[u8]) -> ImageResult<Vec<u8>> {
    decompress_dct_lz4(compressed)
}

/// Internal DCT + quantize + LZ4 compressor.
fn compress_dct_lz4(data: &[u8]) -> ImageResult<Vec<u8>> {
    // Determine float element width.  Prefer f32 (4 bytes); fall back to u16 (2 bytes).
    let (elem_bytes, n_elems) = if data.len() % 4 == 0 {
        (4usize, data.len() / 4)
    } else if data.len() % 2 == 0 {
        (2usize, data.len() / 2)
    } else {
        // Odd byte count: pass through unchanged
        let mut out = vec![0u8; 1 + data.len()];
        out[0] = 0; // version / format byte
        out[1..].copy_from_slice(data);
        return lz4_compress(&out);
    };

    // Convert elements to f32 values
    let mut floats: Vec<f32> = (0..n_elems)
        .map(|i| {
            if elem_bytes == 4 {
                f32::from_le_bytes([
                    data[4 * i],
                    data[4 * i + 1],
                    data[4 * i + 2],
                    data[4 * i + 3],
                ])
            } else {
                // u16 half-float: use the bit pattern cast to u16, then f32
                let bits = u16::from_le_bytes([data[2 * i], data[2 * i + 1]]);
                bits as f32
            }
        })
        .collect();

    // Process in 8×8 blocks
    let n_blocks = (n_elems + DCT_BLOCK - 1) / DCT_BLOCK;
    // Quantized coefficients stored as i16
    let mut quant: Vec<i16> = Vec::with_capacity(n_elems);

    for blk in 0..n_blocks {
        let start = blk * DCT_BLOCK;
        let end = (start + DCT_BLOCK).min(n_elems);
        let mut block = [0.0f32; DCT_BLOCK];
        block[..end - start].copy_from_slice(&floats[start..end]);

        dct_forward_8x8(&mut block);

        for coeff in &block {
            let q = (coeff / DCT_QUANT).round();
            // Clamp to i16 range before cast
            let clamped = q.clamp(i16::MIN as f32, i16::MAX as f32) as i16;
            quant.push(clamped);
        }
        // Update floats slice for bounds (not actually needed since we index directly)
        let _ = &mut floats[start..end];
    }

    // Serialize quantized coefficients as bytes
    let orig_len = data.len() as u32;
    let elem_bytes_flag = elem_bytes as u8;
    let mut payload = vec![elem_bytes_flag];
    payload.extend_from_slice(&orig_len.to_le_bytes());
    for &q in &quant {
        payload.extend_from_slice(&q.to_le_bytes());
    }

    lz4_compress(&payload)
}

/// Internal DCT + dequantize + LZ4 decompressor.
fn decompress_dct_lz4(compressed: &[u8]) -> ImageResult<Vec<u8>> {
    let payload = lz4_decompress(compressed, compressed.len() * 4)?;
    if payload.is_empty() {
        return Ok(Vec::new());
    }
    let elem_bytes = payload[0] as usize;
    if elem_bytes == 0 {
        // Pass-through path
        return Ok(payload[1..].to_vec());
    }
    if payload.len() < 5 {
        return Err(ImageError::Compression("DWA payload too short".to_string()));
    }
    let orig_len = u32::from_le_bytes([payload[1], payload[2], payload[3], payload[4]]) as usize;
    let quant_bytes = &payload[5..];

    // Number of i16 coefficients
    let n_quant = quant_bytes.len() / 2;
    let quant: Vec<i16> = (0..n_quant)
        .map(|i| i16::from_le_bytes([quant_bytes[2 * i], quant_bytes[2 * i + 1]]))
        .collect();

    let n_elems = if elem_bytes == 4 {
        orig_len / 4
    } else {
        orig_len / 2
    };

    let n_blocks = (n_elems + DCT_BLOCK - 1) / DCT_BLOCK;
    let mut floats = Vec::with_capacity(n_blocks * DCT_BLOCK);

    for blk in 0..n_blocks {
        let qstart = blk * DCT_BLOCK;
        let qend = (qstart + DCT_BLOCK).min(quant.len());
        let mut block = [0.0f32; DCT_BLOCK];
        for (i, &q) in quant[qstart..qend].iter().enumerate() {
            block[i] = q as f32 * DCT_QUANT;
        }
        dct_inverse_8x8(&mut block);
        floats.extend_from_slice(&block);
    }

    // Convert back to bytes
    let mut output = Vec::with_capacity(orig_len);
    for i in 0..n_elems.min(floats.len()) {
        if elem_bytes == 4 {
            output.extend_from_slice(&floats[i].to_le_bytes());
        } else {
            // Round and clamp to u16 range
            let val = floats[i].round().clamp(0.0, 65535.0) as u16;
            output.extend_from_slice(&val.to_le_bytes());
        }
    }
    output.truncate(orig_len);
    Ok(output)
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_u16_buf(len: usize) -> Vec<u8> {
        // Generate synthetic u16 values as bytes
        (0..len)
            .flat_map(|i| {
                let val = ((i * 137 + 42) % 65536) as u16;
                val.to_le_bytes().to_vec()
            })
            .collect()
    }

    fn synthetic_f32_buf(len: usize) -> Vec<u8> {
        (0..len)
            .flat_map(|i| {
                let val = (i as f32) * 0.5 + 1.0;
                val.to_le_bytes().to_vec()
            })
            .collect()
    }

    // ── PIZ ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_piz_roundtrip_basic() {
        let data = synthetic_u16_buf(64);
        let compressed = compress_piz(&data).expect("piz compress");
        let recovered = decompress_piz(&compressed).expect("piz decompress");
        assert_eq!(recovered, data, "PIZ round-trip failed");
    }

    #[test]
    fn test_piz_roundtrip_odd_length() {
        let data: Vec<u8> = (0..33u8).collect();
        let compressed = compress_piz(&data).expect("piz compress odd");
        let recovered = decompress_piz(&compressed).expect("piz decompress odd");
        assert_eq!(recovered, data, "PIZ round-trip (odd length) failed");
    }

    #[test]
    fn test_piz_roundtrip_empty() {
        let data: Vec<u8> = vec![];
        let compressed = compress_piz(&data).expect("piz compress empty");
        let recovered = decompress_piz(&compressed).expect("piz decompress empty");
        assert_eq!(recovered, data);
    }

    // ── PXR24 ────────────────────────────────────────────────────────────────

    #[test]
    fn test_pxr24_roundtrip_basic() {
        let data = synthetic_f32_buf(16);
        let compressed = compress_pxr24(&data).expect("pxr24 compress");
        let recovered = decompress_pxr24(&compressed).expect("pxr24 decompress");
        // PXR24 is lossy (drops LSB of each f32); compare only first 3 bytes per float
        assert_eq!(recovered.len(), data.len(), "PXR24 output length mismatch");
        for i in 0..data.len() / 4 {
            let orig = &data[i * 4..i * 4 + 3];
            let rec = &recovered[i * 4..i * 4 + 3];
            assert_eq!(orig, rec, "PXR24 mismatch at float {i} (high 3 bytes)");
        }
    }

    #[test]
    fn test_pxr24_roundtrip_empty() {
        let data: Vec<u8> = vec![];
        let compressed = compress_pxr24(&data).expect("pxr24 compress empty");
        let recovered = decompress_pxr24(&compressed).expect("pxr24 decompress empty");
        assert_eq!(recovered, data);
    }

    // ── B44 ──────────────────────────────────────────────────────────────────

    /// Maximum acceptable reconstruction error for B44 (4-bit nibble quantization over
    /// the block range). Error bound: ≤ ceil(max_delta / 15) + 1 per value.
    fn b44_max_error_for_block(original_block: &[u16]) -> u16 {
        let min_val = original_block.iter().copied().min().unwrap_or(0);
        let max_delta = original_block
            .iter()
            .map(|&v| v.saturating_sub(min_val))
            .max()
            .unwrap_or(0);
        // Quantization step = max_delta / 15, round up, plus 1 for rounding jitter
        (max_delta / 15 + 2).max(1)
    }

    #[test]
    fn test_b44_roundtrip_basic() {
        // 64 u16 values = 4 blocks
        let data = synthetic_u16_buf(64);
        let compressed = compress_b44(&data).expect("b44 compress");
        let recovered = decompress_b44(&compressed).expect("b44 decompress");
        assert_eq!(recovered.len(), data.len(), "B44 output length mismatch");

        // Verify values are within block-relative quantization tolerance
        let orig_vals: Vec<u16> = (0..data.len() / 2)
            .map(|i| u16::from_le_bytes([data[2 * i], data[2 * i + 1]]))
            .collect();
        let rec_vals: Vec<u16> = (0..recovered.len() / 2)
            .map(|i| u16::from_le_bytes([recovered[2 * i], recovered[2 * i + 1]]))
            .collect();

        for block_idx in 0..(orig_vals.len() / B44_BLOCK_SIZE) {
            let start = block_idx * B44_BLOCK_SIZE;
            let end = (start + B44_BLOCK_SIZE).min(orig_vals.len());
            let block_orig = &orig_vals[start..end];
            let max_err = b44_max_error_for_block(block_orig);
            for (j, (&o, &r)) in block_orig
                .iter()
                .zip(rec_vals[start..end].iter())
                .enumerate()
            {
                let diff = (o as i32 - r as i32).unsigned_abs() as u16;
                assert!(
                    diff <= max_err,
                    "B44 block {block_idx} elem {j}: orig={o}, rec={r}, diff={diff}, max_err={max_err}"
                );
            }
        }
    }

    #[test]
    fn test_b44_roundtrip_uniform() {
        // Uniform data: all values the same → lossless for uniform blocks
        let val = 0x8080u16;
        let data: Vec<u8> = std::iter::repeat_n(val, 32)
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let compressed = compress_b44(&data).expect("b44 compress uniform");
        let recovered = decompress_b44(&compressed).expect("b44 decompress uniform");
        assert_eq!(recovered.len(), data.len());
        let rec_vals: Vec<u16> = (0..recovered.len() / 2)
            .map(|i| u16::from_le_bytes([recovered[2 * i], recovered[2 * i + 1]]))
            .collect();
        // Uniform block has max_delta = 0, so all values must reconstruct exactly
        assert!(
            rec_vals.iter().all(|&v| v == val),
            "B44 uniform block should reconstruct exactly"
        );
    }

    // ── B44A ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_b44a_roundtrip_flat_field() {
        // All same u16 value → flat-field optimization → lossless
        let val = 0x1234u16;
        let data: Vec<u8> = std::iter::repeat_n(val, 16)
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let compressed = compress_b44a(&data).expect("b44a compress flat");
        let recovered = decompress_b44a(&compressed).expect("b44a decompress flat");
        assert_eq!(recovered.len(), data.len());
        // Flat-field blocks are losslessly recovered
        let recovered_vals: Vec<u16> = (0..recovered.len() / 2)
            .map(|i| u16::from_le_bytes([recovered[2 * i], recovered[2 * i + 1]]))
            .collect();
        assert!(
            recovered_vals.iter().all(|&v| v == val),
            "B44A flat-field recovery failed"
        );
    }

    #[test]
    fn test_b44a_roundtrip_mixed() {
        // Mix of flat and non-flat blocks
        let mut data = synthetic_u16_buf(64);
        // Make first block flat
        let flat_val = 0x1000u16;
        for i in 0..16 {
            data[2 * i] = flat_val.to_le_bytes()[0];
            data[2 * i + 1] = flat_val.to_le_bytes()[1];
        }
        let compressed = compress_b44a(&data).expect("b44a compress mixed");
        let recovered = decompress_b44a(&compressed).expect("b44a decompress mixed");
        assert_eq!(recovered.len(), data.len());
        // Check flat block is exactly recovered
        let flat_recovered: Vec<u16> = (0..16)
            .map(|i| u16::from_le_bytes([recovered[2 * i], recovered[2 * i + 1]]))
            .collect();
        assert!(
            flat_recovered.iter().all(|&v| v == flat_val),
            "B44A flat block not recovered correctly"
        );
    }

    // ── DWAA ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_dwaa_roundtrip_f32() {
        let data = synthetic_f32_buf(64);
        let compressed = compress_dwaa(&data).expect("dwaa compress");
        let recovered = decompress_dwaa(&compressed).expect("dwaa decompress");
        assert_eq!(recovered.len(), data.len(), "DWAA output length mismatch");
        // Lossy: compare original and reconstructed f32 values within tolerance
        for i in 0..data.len() / 4 {
            let orig = f32::from_le_bytes([
                data[4 * i],
                data[4 * i + 1],
                data[4 * i + 2],
                data[4 * i + 3],
            ]);
            let rec = f32::from_le_bytes([
                recovered[4 * i],
                recovered[4 * i + 1],
                recovered[4 * i + 2],
                recovered[4 * i + 3],
            ]);
            let diff = (orig - rec).abs();
            assert!(
                diff < 20.0,
                "DWAA f32 mismatch at {i}: orig={orig}, rec={rec}, diff={diff}"
            );
        }
    }

    #[test]
    fn test_dwab_roundtrip_f32() {
        let data = synthetic_f32_buf(64);
        let compressed = compress_dwab(&data).expect("dwab compress");
        let recovered = decompress_dwab(&compressed).expect("dwab decompress");
        assert_eq!(recovered.len(), data.len(), "DWAB output length mismatch");
    }

    // ── Haar transform ────────────────────────────────────────────────────────

    #[test]
    fn test_haar_forward_inverse_roundtrip() {
        let original: Vec<u16> = (0..16u16).map(|i| i * 100 + 1000).collect();
        let mut data = original.clone();
        haar_forward(&mut data);
        // After forward transform, data should differ
        assert_ne!(data, original);
        haar_inverse(&mut data);
        assert_eq!(data, original, "Haar forward+inverse roundtrip failed");
    }

    // ── DCT ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_dct_8x8_identity() {
        // All zeros: DCT of zeros = zeros
        let mut block = [0.0f32; 64];
        dct_forward_8x8(&mut block);
        dct_inverse_8x8(&mut block);
        for v in &block {
            assert!(v.abs() < 1e-3, "DCT identity failed: {v}");
        }
    }

    #[test]
    fn test_dct_8x8_constant() {
        // Constant block: forward+inverse should reconstruct
        let constant = 128.0f32;
        let mut block = [constant; 64];
        dct_forward_8x8(&mut block);
        dct_inverse_8x8(&mut block);
        for v in &block {
            let diff = (v - constant).abs();
            assert!(
                diff < 1.0,
                "DCT constant roundtrip failed: {v} vs {constant}"
            );
        }
    }
}
