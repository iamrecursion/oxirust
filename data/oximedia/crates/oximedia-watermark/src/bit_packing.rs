//! Watermark payload bit packing, capacity estimation, and frame-level bit assignment.
//!
//! This module complements `payload.rs` with:
//! - Compact bit-level packing and unpacking utilities
//! - Per-frame capacity estimation for various watermarking schemes
//! - Header + data interleaving for synchronization
//! - Payload fragmentation and reassembly

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Pack a slice of bytes into a bit vector (MSB-first per byte).
#[must_use]
pub fn pack_bits(data: &[u8]) -> Vec<bool> {
    let mut bits = Vec::with_capacity(data.len() * 8);
    for &byte in data {
        for shift in (0..8).rev() {
            bits.push((byte >> shift) & 1 == 1);
        }
    }
    bits
}

/// Unpack a bit vector (MSB-first per byte) into bytes.
///
/// Pads the last byte with zeros if `bits.len()` is not a multiple of 8.
#[must_use]
pub fn unpack_bits(bits: &[bool]) -> Vec<u8> {
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

/// Compute the number of bits needed to represent `n` values.
#[must_use]
pub fn bits_needed(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    (usize::BITS - (n - 1).leading_zeros()) as usize
}

/// Estimate watermark payload capacity (in bits) for spread-spectrum embedding.
///
/// `num_frames`: total number of audio frames available.
/// `chips_per_bit`: spreading factor (chips per payload bit).
#[must_use]
pub fn spread_spectrum_capacity(num_frames: usize, chips_per_bit: usize) -> usize {
    if chips_per_bit == 0 {
        return 0;
    }
    num_frames / chips_per_bit
}

/// Estimate capacity for echo-based watermarking.
///
/// `total_samples`: total audio samples.
/// `segment_size`: samples per segment (one bit per segment).
#[must_use]
pub fn echo_capacity(total_samples: usize, segment_size: usize) -> usize {
    if segment_size == 0 {
        return 0;
    }
    total_samples / segment_size
}

/// Estimate capacity for LSB watermarking.
///
/// `total_samples`: total audio samples.
/// `bits_per_sample`: number of LSBs used per sample.
#[must_use]
pub fn lsb_capacity(total_samples: usize, bits_per_sample: usize) -> usize {
    total_samples * bits_per_sample
}

/// Payload header for watermark synchronization.
///
/// Embedded before the actual payload data.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct PayloadHeader {
    /// Magic marker (4 bytes).
    pub magic: [u8; 4],
    /// Payload version.
    pub version: u8,
    /// Total payload length in bytes (excluding header).
    pub payload_length: u32,
    /// CRC-8 of the payload bytes.
    pub crc8: u8,
}

impl PayloadHeader {
    /// Magic bytes for `OxiMedia` watermark.
    pub const MAGIC: [u8; 4] = *b"OXWM";
    /// Serialized header size in bytes.
    pub const SIZE: usize = 10; // 4 + 1 + 4 + 1

    /// Create a header for the given payload.
    #[must_use]
    pub fn new(payload: &[u8]) -> Self {
        Self {
            magic: Self::MAGIC,
            version: 1,
            payload_length: payload.len() as u32,
            crc8: crc8(payload),
        }
    }

    /// Serialize the header to bytes.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut out = [0u8; Self::SIZE];
        out[0..4].copy_from_slice(&self.magic);
        out[4] = self.version;
        let len_bytes = self.payload_length.to_be_bytes();
        out[5..9].copy_from_slice(&len_bytes);
        out[9] = self.crc8;
        out
    }

    /// Deserialize a header from a byte slice.
    ///
    /// Returns `None` if the slice is too short or the magic is wrong.
    #[must_use]
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < Self::SIZE {
            return None;
        }
        let mut magic = [0u8; 4];
        magic.copy_from_slice(&data[0..4]);
        if magic != Self::MAGIC {
            return None;
        }
        let version = data[4];
        let payload_length = u32::from_be_bytes([data[5], data[6], data[7], data[8]]);
        let crc8 = data[9];
        Some(Self {
            magic,
            version,
            payload_length,
            crc8,
        })
    }

    /// Validate header against a payload slice.
    #[must_use]
    pub fn validate(&self, payload: &[u8]) -> bool {
        self.magic == Self::MAGIC
            && self.payload_length as usize == payload.len()
            && self.crc8 == crc8(payload)
    }
}

/// Compute CRC-8 (polynomial 0x07) of a byte slice.
#[must_use]
pub fn crc8(data: &[u8]) -> u8 {
    let mut crc: u8 = 0;
    for &byte in data {
        crc ^= byte;
        for _ in 0..8 {
            if crc & 0x80 != 0 {
                crc = (crc << 1) ^ 0x07;
            } else {
                crc <<= 1;
            }
        }
    }
    crc
}

/// Fragment a payload into chunks of `chunk_size` bytes.
///
/// Appends the header before the first chunk.
#[must_use]
pub fn fragment_payload(payload: &[u8], chunk_size: usize) -> Vec<Vec<u8>> {
    if chunk_size == 0 {
        return Vec::new();
    }
    let header = PayloadHeader::new(payload);
    let header_bytes = header.to_bytes();

    let mut fragments: Vec<Vec<u8>> = Vec::new();

    // First chunk: header + start of payload
    let mut first = Vec::with_capacity(chunk_size);
    first.extend_from_slice(&header_bytes);
    let remaining_in_first = chunk_size.saturating_sub(PayloadHeader::SIZE);
    let first_data_end = payload.len().min(remaining_in_first);
    first.extend_from_slice(&payload[..first_data_end]);
    fragments.push(first);

    // Remaining chunks
    let mut offset = first_data_end;
    while offset < payload.len() {
        let end = (offset + chunk_size).min(payload.len());
        fragments.push(payload[offset..end].to_vec());
        offset = end;
    }

    fragments
}

/// Reassemble fragmented payload.
///
/// Strips the header from the first fragment and validates integrity.
/// Returns the payload bytes on success or `None` on failure.
#[must_use]
pub fn reassemble_payload(fragments: &[Vec<u8>]) -> Option<Vec<u8>> {
    if fragments.is_empty() {
        return None;
    }
    let first = &fragments[0];
    let header = PayloadHeader::from_bytes(first)?;

    // Extract data portion of first fragment
    let mut data: Vec<u8> = first[PayloadHeader::SIZE..].to_vec();

    // Append remaining fragments
    for frag in fragments.iter().skip(1) {
        data.extend_from_slice(frag);
    }

    // Trim to declared length
    let len = header.payload_length as usize;
    if data.len() < len {
        return None;
    }
    let payload = data[..len].to_vec();

    if header.validate(&payload) {
        Some(payload)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_bits_single_byte() {
        let bits = pack_bits(&[0b10110001]);
        assert_eq!(bits.len(), 8);
        assert_eq!(bits[0], true);
        assert_eq!(bits[1], false);
        assert_eq!(bits[7], true);
    }

    #[test]
    fn test_pack_unpack_roundtrip() {
        let data = vec![0xDE, 0xAD, 0xBE, 0xEF];
        let bits = pack_bits(&data);
        let recovered = unpack_bits(&bits);
        assert_eq!(recovered, data);
    }

    #[test]
    fn test_pack_bits_zero_byte() {
        let bits = pack_bits(&[0x00]);
        assert!(bits.iter().all(|&b| !b));
    }

    #[test]
    fn test_pack_bits_ff_byte() {
        let bits = pack_bits(&[0xFF]);
        assert!(bits.iter().all(|&b| b));
    }

    #[test]
    fn test_unpack_bits_empty() {
        let bytes = unpack_bits(&[]);
        assert!(bytes.is_empty());
    }

    #[test]
    fn test_bits_needed_one() {
        assert_eq!(bits_needed(1), 1);
    }

    #[test]
    fn test_bits_needed_two() {
        assert_eq!(bits_needed(2), 1);
    }

    #[test]
    fn test_bits_needed_three() {
        assert_eq!(bits_needed(3), 2);
    }

    #[test]
    fn test_bits_needed_256() {
        assert_eq!(bits_needed(256), 8);
    }

    #[test]
    fn test_spread_spectrum_capacity() {
        assert_eq!(spread_spectrum_capacity(1000, 10), 100);
    }

    #[test]
    fn test_spread_spectrum_capacity_zero_chips() {
        assert_eq!(spread_spectrum_capacity(1000, 0), 0);
    }

    #[test]
    fn test_echo_capacity() {
        assert_eq!(echo_capacity(44100, 512), 86);
    }

    #[test]
    fn test_lsb_capacity() {
        assert_eq!(lsb_capacity(1000, 2), 2000);
    }

    #[test]
    fn test_crc8_empty() {
        assert_eq!(crc8(&[]), 0);
    }

    #[test]
    fn test_crc8_deterministic() {
        let a = crc8(b"hello");
        let b = crc8(b"hello");
        assert_eq!(a, b);
    }

    #[test]
    fn test_crc8_different_data() {
        let a = crc8(b"hello");
        let b = crc8(b"world");
        assert_ne!(a, b);
    }

    #[test]
    fn test_payload_header_roundtrip() {
        let payload = b"test payload";
        let header = PayloadHeader::new(payload);
        let bytes = header.to_bytes();
        let recovered = PayloadHeader::from_bytes(&bytes).expect("should succeed in test");
        assert_eq!(recovered.magic, PayloadHeader::MAGIC);
        assert_eq!(recovered.payload_length as usize, payload.len());
        assert_eq!(recovered.crc8, crc8(payload));
    }

    #[test]
    fn test_payload_header_validate() {
        let payload = b"OxiMedia";
        let header = PayloadHeader::new(payload);
        assert!(header.validate(payload));
    }

    #[test]
    fn test_payload_header_invalid_magic() {
        let mut bytes = [0u8; PayloadHeader::SIZE];
        bytes[0..4].copy_from_slice(b"XXXX");
        assert!(PayloadHeader::from_bytes(&bytes).is_none());
    }

    #[test]
    fn test_fragment_reassemble_roundtrip() {
        let payload = b"Hello, OxiMedia watermark!";
        let fragments = fragment_payload(payload, 16);
        let recovered = reassemble_payload(&fragments).expect("should succeed in test");
        assert_eq!(recovered.as_slice(), payload);
    }

    #[test]
    fn test_fragment_empty_chunk_size() {
        let payload = b"data";
        let fragments = fragment_payload(payload, 0);
        assert!(fragments.is_empty());
    }

    #[test]
    fn test_reassemble_empty() {
        assert!(reassemble_payload(&[]).is_none());
    }
}
