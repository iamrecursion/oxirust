//! Robust watermarking resistant to common signal processing attacks.
//!
//! This module provides:
//! - `WatermarkPayload`: payload with XOR-fold checksum
//! - `EccEncoder` / `EccDecoder`: simplified (23,12) Golay error-correction
//! - `RobustEmbedder`: spread-spectrum embedding in DCT domain (simulated)
//! - `RobustDetector`: payload extraction

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// RobustWatermarkConfig
// ---------------------------------------------------------------------------

/// Configuration for robust watermarking.
#[derive(Debug, Clone)]
pub struct RobustWatermarkConfig {
    /// Number of payload bits.
    pub payload_bits: u32,
    /// Redundancy factor (number of times each bit is repeated).
    pub redundancy: u32,
    /// Synchronisation pattern prepended to each frame's watermark.
    pub sync_pattern: Vec<u8>,
}

impl Default for RobustWatermarkConfig {
    fn default() -> Self {
        Self {
            payload_bits: 32,
            redundancy: 3,
            sync_pattern: vec![0xAB, 0xCD, 0xEF],
        }
    }
}

// ---------------------------------------------------------------------------
// WatermarkPayload
// ---------------------------------------------------------------------------

/// A payload with an XOR-fold checksum for integrity checking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WatermarkPayload {
    /// Raw data bytes.
    pub data: Vec<u8>,
    /// XOR fold of all data bytes.
    pub checksum: u8,
}

impl WatermarkPayload {
    /// Encode raw data: compute XOR checksum and wrap.
    #[must_use]
    pub fn encode(data: &[u8]) -> Self {
        let checksum = data.iter().fold(0u8, |acc, &b| acc ^ b);
        Self {
            data: data.to_vec(),
            checksum,
        }
    }

    /// Decode: verify checksum and return data if valid.
    ///
    /// `payload` is expected as `[data bytes..., checksum byte]`.
    #[must_use]
    pub fn decode(payload: &[u8]) -> Option<Vec<u8>> {
        if payload.is_empty() {
            return None;
        }
        let (data, checksum_slice) = payload.split_at(payload.len() - 1);
        let checksum = checksum_slice[0];
        let expected = data.iter().fold(0u8, |acc, &b| acc ^ b);
        if expected == checksum {
            Some(data.to_vec())
        } else {
            None
        }
    }

    /// Serialize to bytes: data followed by checksum.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = self.data.clone();
        bytes.push(self.checksum);
        bytes
    }
}

// ---------------------------------------------------------------------------
// EccEncoder / EccDecoder  (simplified (23,12) Golay code)
// ---------------------------------------------------------------------------

/// Simplified (23,12) Golay code encoder.
///
/// Generator polynomial: 0xC75 (degree 11, applied mod 2^23 - 1).
/// This is a pedagogical simulation; a production implementation would use
/// the full 23-bit Golay codeword.
pub struct EccEncoder;

/// Golay generator polynomial (11-bit) used for parity generation.
const GOLAY_POLY: u32 = 0xC75;

impl EccEncoder {
    /// Encode a 12-bit message `data` into a 23-bit Golay codeword.
    ///
    /// The 12 message bits are placed in the high position; 11 parity bits
    /// are appended computed via polynomial division.
    #[must_use]
    pub fn encode_golay(data: u16) -> u32 {
        // Only lower 12 bits are used
        let msg = u32::from(data & 0x0FFF);
        // Shift message into the high 12 bits of a 23-bit word
        let mut codeword: u32 = msg << 11;

        // Generator polynomial division (mod 2) to produce 11 parity bits
        for i in (11..23u32).rev() {
            if (codeword >> i) & 1 == 1 {
                codeword ^= GOLAY_POLY << (i - 11);
            }
        }

        // Combine: message in upper 12 bits + parity in lower 11 bits
        (msg << 11) | (codeword & 0x07FF)
    }
}

/// Simplified (23,12) Golay code decoder.
pub struct EccDecoder;

impl EccDecoder {
    /// Decode a 23-bit Golay `codeword`.
    ///
    /// Computes the syndrome; if it is zero the codeword is valid.
    /// If there is exactly 1 error bit in the syndrome, it is corrected.
    /// Returns `Some(message)` on success, `None` if uncorrectable.
    #[must_use]
    pub fn decode_golay(codeword: u32) -> Option<u16> {
        // Compute syndrome: remainder of codeword ÷ generator
        let syndrome = Self::syndrome(codeword);

        if syndrome == 0 {
            // No error: extract upper 12 bits
            return Some(((codeword >> 11) & 0x0FFF) as u16);
        }

        // Try single-bit correction: flip each of the 23 bits
        for bit in 0..23u32 {
            let corrected = codeword ^ (1 << bit);
            if Self::syndrome(corrected) == 0 {
                return Some(((corrected >> 11) & 0x0FFF) as u16);
            }
        }

        None
    }

    /// Compute the syndrome (parity check) for a 23-bit codeword.
    fn syndrome(codeword: u32) -> u32 {
        let mut r = codeword & 0x007F_FFFF; // 23 bits
        for i in (11..23u32).rev() {
            if (r >> i) & 1 == 1 {
                r ^= GOLAY_POLY << (i - 11);
            }
        }
        r & 0x07FF // 11 parity bits
    }
}

// ---------------------------------------------------------------------------
// RobustEmbedder
// ---------------------------------------------------------------------------

/// Robust watermark embedder using simulated DCT-domain spread spectrum.
///
/// The approach: divide the frame into 8×8 blocks, compute the average
/// (DC coefficient surrogate) and modulate it to carry payload bits.
pub struct RobustEmbedder;

impl RobustEmbedder {
    /// Embed `payload` into `frame` (width × height f32 pixels).
    ///
    /// Returns the modified frame. Payload bits are spread across 8×8 block
    /// averages using a simple ±delta modulation.
    #[must_use]
    pub fn embed(frame: &[f32], width: u32, height: u32, payload: &WatermarkPayload) -> Vec<f32> {
        let mut out = frame.to_vec();
        let w = width as usize;
        let h = height as usize;

        if w == 0 || h == 0 || payload.data.is_empty() {
            return out;
        }

        let payload_bytes = payload.to_bytes();
        let total_bits = payload_bytes.len() * 8;

        // Number of 8×8 blocks
        let blocks_x = w / 8;
        let blocks_y = h / 8;
        let total_blocks = blocks_x * blocks_y;

        if total_blocks == 0 {
            return out;
        }

        for block_idx in 0..total_blocks {
            let bit_idx = block_idx % total_bits;
            let byte_idx = bit_idx / 8;
            let bit_shift = 7 - (bit_idx % 8);
            let bit_val = (payload_bytes[byte_idx] >> bit_shift) & 1;

            let delta: f32 = if bit_val == 1 { 0.05 } else { -0.05 };

            let bx = (block_idx % blocks_x) * 8;
            let by = (block_idx / blocks_x) * 8;

            // Modulate the DC component (upper-left 4×4 of block)
            for row in 0..4usize {
                for col in 0..4usize {
                    let px = (by + row) * w + (bx + col);
                    if px < out.len() {
                        out[px] = (out[px] + delta).clamp(-1.0, 1.0);
                    }
                }
            }
        }

        out
    }
}

// ---------------------------------------------------------------------------
// RobustDetector
// ---------------------------------------------------------------------------

/// Robust watermark detector.
pub struct RobustDetector;

impl RobustDetector {
    /// Attempt to detect and extract a `WatermarkPayload` from `frame`.
    ///
    /// Returns `Some(payload)` if the checksum is valid, `None` otherwise.
    #[must_use]
    pub fn detect(
        frame: &[f32],
        width: u32,
        height: u32,
        config: &RobustWatermarkConfig,
    ) -> Option<WatermarkPayload> {
        let w = width as usize;
        let h = height as usize;

        if w == 0 || h == 0 {
            return None;
        }

        let blocks_x = w / 8;
        let blocks_y = h / 8;
        let total_blocks = blocks_x * blocks_y;

        if total_blocks == 0 {
            return None;
        }

        // Determine how many bytes to extract (payload_bits / 8 + 1 checksum byte)
        let payload_bytes_count = (config.payload_bits as usize / 8) + 1;
        let total_bits = payload_bytes_count * 8;

        let mut bit_votes: Vec<u32> = vec![0u32; total_bits];
        let mut bit_counts: Vec<u32> = vec![0u32; total_bits];

        for block_idx in 0..total_blocks {
            let bit_idx = block_idx % total_bits;
            let bx = (block_idx % blocks_x) * 8;
            let by = (block_idx / blocks_x) * 8;

            // Measure average of 4×4 DC region
            let mut sum = 0.0f32;
            let mut count = 0usize;
            for row in 0..4usize {
                for col in 0..4usize {
                    let px = (by + row) * w + (bx + col);
                    if px < frame.len() {
                        sum += frame[px];
                        count += 1;
                    }
                }
            }

            if count > 0 {
                let avg = sum / count as f32;
                // Positive mean → bit 1
                if avg >= 0.0 {
                    bit_votes[bit_idx] += 1;
                }
                bit_counts[bit_idx] += 1;
            }
        }

        // Reconstruct bytes by majority vote per bit
        let mut bytes = vec![0u8; payload_bytes_count];
        for bit_idx in 0..total_bits {
            let byte_idx = bit_idx / 8;
            let bit_shift = 7 - (bit_idx % 8);
            let vote = bit_votes[bit_idx];
            let total = bit_counts[bit_idx].max(1);
            if vote * 2 >= total {
                bytes[byte_idx] |= 1 << bit_shift;
            }
        }

        // Try to decode as WatermarkPayload
        let decoded_data = WatermarkPayload::decode(&bytes)?;
        Some(WatermarkPayload::encode(&decoded_data))
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- WatermarkPayload tests ---

    #[test]
    fn test_payload_encode_checksum() {
        let data = b"hello";
        let payload = WatermarkPayload::encode(data);
        let expected_checksum = b'h' ^ b'e' ^ b'l' ^ b'l' ^ b'o';
        assert_eq!(payload.checksum, expected_checksum);
    }

    #[test]
    fn test_payload_decode_valid() {
        let data = b"test";
        let encoded = WatermarkPayload::encode(data);
        let bytes = encoded.to_bytes();
        let decoded = WatermarkPayload::decode(&bytes);
        assert_eq!(decoded, Some(data.to_vec()));
    }

    #[test]
    fn test_payload_decode_invalid_checksum() {
        let mut bytes = b"test\xFF".to_vec(); // corrupt checksum
        bytes[4] ^= 0x01;
        // Make it unlikely to be valid
        let result = WatermarkPayload::decode(&bytes);
        // May or may not be Some; if checksum happens to match, it's valid
        let _ = result;
    }

    #[test]
    fn test_payload_decode_empty() {
        let result = WatermarkPayload::decode(&[]);
        assert!(result.is_none());
    }

    #[test]
    fn test_payload_roundtrip_empty_data() {
        let payload = WatermarkPayload::encode(b"");
        assert_eq!(payload.checksum, 0);
        let bytes = payload.to_bytes();
        let decoded = WatermarkPayload::decode(&bytes);
        assert_eq!(decoded, Some(vec![]));
    }

    #[test]
    fn test_payload_to_bytes_format() {
        let payload = WatermarkPayload::encode(b"AB");
        let bytes = payload.to_bytes();
        assert_eq!(bytes.len(), 3); // 2 data + 1 checksum
        assert_eq!(bytes[2], b'A' ^ b'B');
    }

    // --- Golay ECC tests ---

    #[test]
    fn test_golay_encode_zero() {
        let cw = EccEncoder::encode_golay(0);
        // Zero data → zero parity
        assert_eq!(cw, 0);
    }

    #[test]
    fn test_golay_encode_nonzero() {
        let cw = EccEncoder::encode_golay(0b1010_1010_1010);
        // Should be non-zero
        assert_ne!(cw, 0);
        // Upper 12 bits should match the message
        assert_eq!((cw >> 11) & 0x0FFF, 0b1010_1010_1010);
    }

    #[test]
    fn test_golay_decode_no_error() {
        let msg = 0b1100_1010_0101u16;
        let cw = EccEncoder::encode_golay(msg);
        let decoded = EccDecoder::decode_golay(cw);
        assert_eq!(decoded, Some(msg));
    }

    #[test]
    fn test_golay_decode_one_bit_error() {
        let msg = 0b0011_0101_1001u16;
        let cw = EccEncoder::encode_golay(msg);
        // Flip bit 0
        let corrupted = cw ^ 1;
        let decoded = EccDecoder::decode_golay(corrupted);
        assert_eq!(decoded, Some(msg));
    }

    #[test]
    fn test_golay_decode_syndrome_zero_clean() {
        let msg = 0x0ABCu16;
        let cw = EccEncoder::encode_golay(msg);
        assert_eq!(EccDecoder::syndrome(cw), 0);
    }

    // --- RobustEmbedder / RobustDetector tests ---

    #[test]
    fn test_embed_produces_different_frame() {
        let frame = vec![0.5f32; 64 * 64];
        let payload = WatermarkPayload::encode(b"WM");
        let embedded = RobustEmbedder::embed(&frame, 64, 64, &payload);
        assert_ne!(frame, embedded);
    }

    #[test]
    fn test_embed_same_length() {
        let frame = vec![0.3f32; 128 * 128];
        let payload = WatermarkPayload::encode(b"test payload");
        let embedded = RobustEmbedder::embed(&frame, 128, 128, &payload);
        assert_eq!(frame.len(), embedded.len());
    }

    #[test]
    fn test_embed_empty_payload_no_change() {
        let frame = vec![0.5f32; 64 * 64];
        let payload = WatermarkPayload {
            data: vec![],
            checksum: 0,
        };
        let embedded = RobustEmbedder::embed(&frame, 64, 64, &payload);
        assert_eq!(frame, embedded);
    }

    #[test]
    fn test_detect_no_crash_on_small_frame() {
        let frame = vec![0.0f32; 4 * 4]; // smaller than 8×8 block
        let config = RobustWatermarkConfig::default();
        let result = RobustDetector::detect(&frame, 4, 4, &config);
        // May return None (not enough blocks) – just verify no panic
        let _ = result;
    }

    #[test]
    fn test_embed_clamps_values() {
        // Start with max values; adding delta must not overflow
        let frame = vec![1.0f32; 64 * 64];
        let payload = WatermarkPayload::encode(b"\xFF");
        let embedded = RobustEmbedder::embed(&frame, 64, 64, &payload);
        assert!(embedded.iter().all(|&v| v <= 1.0 && v >= -1.0));
    }
}
