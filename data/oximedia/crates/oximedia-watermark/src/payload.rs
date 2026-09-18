//! Payload encoding and decoding with error correction.
//!
//! This module provides:
//! - Bit packing/unpacking
//! - Reed-Solomon error correction
//! - Synchronization pattern generation
//! - CRC checksums

use crate::error::{WatermarkError, WatermarkResult};
use reed_solomon_erasure::galois_8::ReedSolomon;

/// Synchronization pattern for watermark detection.
const SYNC_PATTERN: &[u8] = b"OXIWM";
#[allow(dead_code)]
const SYNC_BITS: usize = SYNC_PATTERN.len() * 8;

/// Payload encoder/decoder with error correction.
pub struct PayloadCodec {
    rs_encoder: ReedSolomon,
    data_shards: usize,
    parity_shards: usize,
}

impl PayloadCodec {
    /// Create a new payload codec.
    ///
    /// # Arguments
    ///
    /// * `data_shards` - Number of data shards
    /// * `parity_shards` - Number of parity shards for error correction
    ///
    /// # Errors
    ///
    /// Returns error if Reed-Solomon setup fails.
    pub fn new(data_shards: usize, parity_shards: usize) -> WatermarkResult<Self> {
        let rs_encoder = ReedSolomon::new(data_shards, parity_shards)
            .map_err(|e| WatermarkError::Internal(format!("Reed-Solomon init failed: {e}")))?;

        Ok(Self {
            rs_encoder,
            data_shards,
            parity_shards,
        })
    }

    /// Encode payload with synchronization and error correction.
    ///
    /// # Errors
    ///
    /// Returns error if encoding fails.
    pub fn encode(&self, data: &[u8]) -> WatermarkResult<Vec<u8>> {
        let mut encoded = Vec::new();

        // Add synchronization pattern
        encoded.extend_from_slice(SYNC_PATTERN);

        // Add length prefix
        let len = u16::try_from(data.len())
            .map_err(|_| WatermarkError::InvalidParameter("Payload too large".to_string()))?;
        encoded.extend_from_slice(&len.to_be_bytes());

        // Add CRC32 checksum
        let crc = crc32(data);
        encoded.extend_from_slice(&crc.to_be_bytes());

        // Pad data to multiple of data_shards
        let mut padded_data = data.to_vec();
        let padding = (self.data_shards - (data.len() % self.data_shards)) % self.data_shards;
        padded_data.resize(data.len() + padding, 0);

        // Apply Reed-Solomon encoding
        let chunk_size = self.data_shards;
        for chunk in padded_data.chunks(chunk_size) {
            let mut shards: Vec<Vec<u8>> = vec![vec![0]; self.data_shards + self.parity_shards];
            for (i, &byte) in chunk.iter().enumerate() {
                shards[i] = vec![byte];
            }

            self.rs_encoder
                .encode(&mut shards)
                .map_err(|e| WatermarkError::Internal(format!("RS encode failed: {e}")))?;

            // Add all shards (data + parity)
            for shard in &shards {
                if !shard.is_empty() {
                    encoded.push(shard[0]);
                }
            }
        }

        Ok(encoded)
    }

    /// Decode payload with error correction.
    ///
    /// # Errors
    ///
    /// Returns error if synchronization fails, CRC check fails, or too many errors.
    pub fn decode(&self, encoded: &[u8]) -> WatermarkResult<Vec<u8>> {
        // Check synchronization pattern
        if encoded.len() < SYNC_PATTERN.len() {
            return Err(WatermarkError::SyncFailed("Data too short".to_string()));
        }

        if &encoded[..SYNC_PATTERN.len()] != SYNC_PATTERN {
            return Err(WatermarkError::SyncFailed(
                "Sync pattern mismatch".to_string(),
            ));
        }

        let mut offset = SYNC_PATTERN.len();

        // Read length
        if encoded.len() < offset + 2 {
            return Err(WatermarkError::InvalidData(
                "Missing length field".to_string(),
            ));
        }
        let len = u16::from_be_bytes([encoded[offset], encoded[offset + 1]]) as usize;
        offset += 2;

        // Read CRC
        if encoded.len() < offset + 4 {
            return Err(WatermarkError::InvalidData("Missing CRC field".to_string()));
        }
        let expected_crc = u32::from_be_bytes([
            encoded[offset],
            encoded[offset + 1],
            encoded[offset + 2],
            encoded[offset + 3],
        ]);
        offset += 4;

        // Decode Reed-Solomon data
        let rs_data = &encoded[offset..];
        let shard_count = self.data_shards + self.parity_shards;
        let chunk_count = rs_data.len() / shard_count;

        let mut decoded = Vec::new();

        for chunk_idx in 0..chunk_count {
            let chunk_offset = chunk_idx * shard_count;
            if chunk_offset + shard_count > rs_data.len() {
                break;
            }

            let mut shards: Vec<Option<Vec<u8>>> = (0..shard_count)
                .map(|i| Some(vec![rs_data[chunk_offset + i]]))
                .collect();

            self.rs_encoder
                .reconstruct(&mut shards)
                .map_err(|_| WatermarkError::ErrorCorrectionFailed)?;

            // Extract data shards
            for s in shards.iter().take(self.data_shards).flatten() {
                if !s.is_empty() {
                    decoded.push(s[0]);
                }
            }
        }

        // Trim to actual length
        decoded.truncate(len);

        // Verify CRC
        let actual_crc = crc32(&decoded);
        if actual_crc != expected_crc {
            return Err(WatermarkError::InvalidData("CRC mismatch".to_string()));
        }

        Ok(decoded)
    }

    /// Get total encoded size for given data length.
    #[must_use]
    pub fn encoded_size(&self, data_len: usize) -> usize {
        let header_size = SYNC_PATTERN.len() + 2 + 4; // sync + len + crc
        let padded_len = data_len.div_ceil(self.data_shards) * self.data_shards;
        let rs_len = (padded_len / self.data_shards) * (self.data_shards + self.parity_shards);
        header_size + rs_len
    }
}

/// Generate pseudorandom bit sequence for spreading.
#[must_use]
pub fn generate_pn_sequence(length: usize, seed: u64) -> Vec<i8> {
    let mut rng = scirs2_core::random::Random::seed(seed);
    (0..length)
        .map(|_| if rng.random_f64() < 0.5 { 1 } else { -1 })
        .collect()
}

/// Generate Gold code sequence for better correlation properties.
#[must_use]
pub fn generate_gold_code(length: usize, seed: u64) -> Vec<i8> {
    // Simplified Gold code using two m-sequences
    let seq1 = generate_pn_sequence(length, seed);
    let seq2 = generate_pn_sequence(length, seed.wrapping_add(1));

    seq1.iter().zip(seq2.iter()).map(|(&a, &b)| a * b).collect()
}

/// Pack bits into bytes.
#[must_use]
pub fn pack_bits(bits: &[bool]) -> Vec<u8> {
    bits.chunks(8)
        .map(|chunk| {
            let mut byte = 0u8;
            for (i, &bit) in chunk.iter().enumerate() {
                if bit {
                    byte |= 1 << (7 - i);
                }
            }
            byte
        })
        .collect()
}

/// Unpack bytes into bits.
#[must_use]
pub fn unpack_bits(bytes: &[u8], bit_count: usize) -> Vec<bool> {
    bytes
        .iter()
        .flat_map(|&byte| (0..8).rev().map(move |i| (byte >> i) & 1 == 1))
        .take(bit_count)
        .collect()
}

/// Calculate CRC32 checksum.
fn crc32(data: &[u8]) -> u32 {
    const POLY: u32 = 0xEDB8_8320;
    let mut crc = 0xFFFF_FFFFu32;

    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ POLY
            } else {
                crc >> 1
            };
        }
    }

    !crc
}

/// Synchronization detector using correlation.
pub struct SyncDetector {
    pattern: Vec<i8>,
}

impl SyncDetector {
    /// Create a new synchronization detector.
    #[must_use]
    pub fn new(seed: u64, pattern_length: usize) -> Self {
        let pattern = generate_gold_code(pattern_length, seed);
        Self { pattern }
    }

    /// Detect synchronization pattern in signal.
    ///
    /// Returns the offset where the pattern is detected.
    #[must_use]
    pub fn detect(&self, signal: &[f32], threshold: f32) -> Option<usize> {
        if signal.len() < self.pattern.len() {
            return None;
        }

        let mut max_corr = 0.0f32;
        let mut max_offset = 0;

        for offset in 0..=(signal.len() - self.pattern.len()) {
            let corr = self.correlate(&signal[offset..offset + self.pattern.len()]);
            if corr > max_corr {
                max_corr = corr;
                max_offset = offset;
            }
        }

        if max_corr >= threshold {
            Some(max_offset)
        } else {
            None
        }
    }

    /// Calculate correlation between signal and pattern.
    fn correlate(&self, signal: &[f32]) -> f32 {
        let mut sum = 0.0f32;
        let mut sig_energy = 0.0f32;

        for (i, &pat) in self.pattern.iter().enumerate() {
            sum += signal[i] * f32::from(pat);
            sig_energy += signal[i] * signal[i];
        }

        if sig_energy == 0.0 {
            return 0.0;
        }

        let pat_energy: f32 = self.pattern.iter().map(|&p| f32::from(p * p)).sum();

        sum / (sig_energy.sqrt() * pat_energy.sqrt())
    }

    /// Get pattern length.
    #[must_use]
    pub fn pattern_length(&self) -> usize {
        self.pattern.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_payload_encode_decode() {
        let codec = PayloadCodec::new(8, 4).expect("should succeed in test");
        let data = b"Hello, Watermark!";

        let encoded = codec.encode(data).expect("should succeed in test");
        let decoded = codec.decode(&encoded).expect("should succeed in test");

        assert_eq!(data.as_slice(), decoded.as_slice());
    }

    #[test]
    fn test_bit_packing() {
        let bits = vec![true, false, true, true, false, false, true, false, true];
        let bytes = pack_bits(&bits);
        let unpacked = unpack_bits(&bytes, bits.len());

        assert_eq!(bits, unpacked);
    }

    #[test]
    fn test_crc32() {
        let data = b"Test data";
        let crc1 = crc32(data);
        let crc2 = crc32(data);
        assert_eq!(crc1, crc2);

        let data2 = b"Test Data";
        let crc3 = crc32(data2);
        assert_ne!(crc1, crc3);
    }

    #[test]
    fn test_sync_detection() {
        let detector = SyncDetector::new(12345, 128);
        let mut signal = vec![0.0f32; 1000];

        // Insert pattern at offset 100
        let pattern = generate_gold_code(128, 12345);
        for (i, &p) in pattern.iter().enumerate() {
            signal[100 + i] = f32::from(p) * 0.5;
        }

        let offset = detector.detect(&signal, 0.5);
        assert_eq!(offset, Some(100));
    }

    #[test]
    fn test_pn_sequence() {
        let seq1 = generate_pn_sequence(100, 42);
        let seq2 = generate_pn_sequence(100, 42);
        assert_eq!(seq1, seq2);

        let seq3 = generate_pn_sequence(100, 43);
        assert_ne!(seq1, seq3);
    }
}
