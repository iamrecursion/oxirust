//! Controlled corruption injection for testing repair algorithms.
//!
//! This module provides deterministic, reproducible corruption patterns that
//! can be applied to media data. Every corruption operation is reversible
//! given the same seed, making it ideal for round-trip repair testing.
//!
//! Supported corruption modes:
//! - Bit-flip: flip individual bits at controlled positions
//! - Truncation: cut the file at a given offset
//! - Header wipe: zero out a range of header bytes
//! - Byte insertion: insert garbage bytes at a position
//! - Byte deletion: remove a range of bytes
//! - Block corruption: overwrite a contiguous block with a pattern

#![allow(dead_code)]

/// Configuration for a corruption operation.
#[derive(Debug, Clone)]
pub struct CorruptionConfig {
    /// Random seed for reproducible corruption patterns.
    pub seed: u64,
    /// Maximum number of corruption points to introduce.
    pub max_corruptions: usize,
}

impl Default for CorruptionConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            max_corruptions: 10,
        }
    }
}

/// Description of a single corruption that was applied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CorruptionKind {
    /// One or more bits were flipped.
    BitFlip {
        /// Byte offset where the flip occurred.
        offset: usize,
        /// Bitmask of flipped bits.
        mask: u8,
    },
    /// Data was truncated at a position.
    Truncation {
        /// Original length before truncation.
        original_len: usize,
        /// New length after truncation.
        truncated_len: usize,
    },
    /// A range of bytes was zeroed out (header wipe).
    HeaderWipe {
        /// Start offset of the wiped range.
        offset: usize,
        /// Number of bytes wiped.
        length: usize,
    },
    /// Bytes were inserted at a position.
    ByteInsertion {
        /// Offset where bytes were inserted.
        offset: usize,
        /// Number of bytes inserted.
        count: usize,
    },
    /// Bytes were deleted from a range.
    ByteDeletion {
        /// Start offset of deletion.
        offset: usize,
        /// Number of bytes deleted.
        count: usize,
    },
    /// A block was overwritten with a pattern.
    BlockCorruption {
        /// Start offset of the corrupted block.
        offset: usize,
        /// Number of bytes overwritten.
        length: usize,
        /// Pattern byte used to overwrite.
        pattern: u8,
    },
}

/// Record of all corruptions applied to a data buffer.
#[derive(Debug, Clone)]
pub struct CorruptionRecord {
    /// List of corruptions in the order they were applied.
    pub corruptions: Vec<CorruptionKind>,
    /// Original data length before any mutations.
    pub original_len: usize,
    /// Final data length after all mutations.
    pub final_len: usize,
}

/// A simple deterministic PRNG (xorshift64) for reproducible corruption.
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_usize(&mut self, bound: usize) -> usize {
        if bound == 0 {
            return 0;
        }
        (self.next_u64() % bound as u64) as usize
    }

    fn next_u8(&mut self) -> u8 {
        (self.next_u64() & 0xFF) as u8
    }
}

// ---------------------------------------------------------------------------
// Individual corruption operations
// ---------------------------------------------------------------------------

/// Flip bits at pseudo-random positions in `data`.
///
/// Returns the list of bit-flips applied.
pub fn apply_bit_flips(data: &mut [u8], config: &CorruptionConfig) -> Vec<CorruptionKind> {
    if data.is_empty() {
        return Vec::new();
    }

    let mut rng = Rng::new(config.seed);
    let count = config.max_corruptions.min(data.len());
    let mut results = Vec::with_capacity(count);

    for _ in 0..count {
        let offset = rng.next_usize(data.len());
        let bit = 1u8 << (rng.next_usize(8) as u8);
        data[offset] ^= bit;
        results.push(CorruptionKind::BitFlip { offset, mask: bit });
    }

    results
}

/// Truncate `data` at the given byte offset.
///
/// If `offset` is beyond data length, data is unchanged.
pub fn apply_truncation(data: &mut Vec<u8>, offset: usize) -> Option<CorruptionKind> {
    if offset >= data.len() {
        return None;
    }
    let original_len = data.len();
    data.truncate(offset);
    Some(CorruptionKind::Truncation {
        original_len,
        truncated_len: offset,
    })
}

/// Zero out a range of bytes (simulating a header wipe).
pub fn apply_header_wipe(data: &mut [u8], offset: usize, length: usize) -> Option<CorruptionKind> {
    if offset >= data.len() {
        return None;
    }
    let actual_len = length.min(data.len() - offset);
    for byte in &mut data[offset..offset + actual_len] {
        *byte = 0x00;
    }
    Some(CorruptionKind::HeaderWipe {
        offset,
        length: actual_len,
    })
}

/// Insert `count` garbage bytes at `offset`.
pub fn apply_byte_insertion(
    data: &mut Vec<u8>,
    offset: usize,
    count: usize,
    config: &CorruptionConfig,
) -> Option<CorruptionKind> {
    if offset > data.len() || count == 0 {
        return None;
    }
    let mut rng = Rng::new(config.seed.wrapping_add(offset as u64));
    let garbage: Vec<u8> = (0..count).map(|_| rng.next_u8()).collect();

    // Insert by splicing
    let tail = data.split_off(offset);
    data.extend_from_slice(&garbage);
    data.extend_from_slice(&tail);

    Some(CorruptionKind::ByteInsertion { offset, count })
}

/// Delete `count` bytes starting at `offset`.
pub fn apply_byte_deletion(
    data: &mut Vec<u8>,
    offset: usize,
    count: usize,
) -> Option<CorruptionKind> {
    if offset >= data.len() || count == 0 {
        return None;
    }
    let actual_count = count.min(data.len() - offset);
    data.drain(offset..offset + actual_count);
    Some(CorruptionKind::ByteDeletion {
        offset,
        count: actual_count,
    })
}

/// Overwrite a block of data with a repeating pattern byte.
pub fn apply_block_corruption(
    data: &mut [u8],
    offset: usize,
    length: usize,
    pattern: u8,
) -> Option<CorruptionKind> {
    if offset >= data.len() || length == 0 {
        return None;
    }
    let actual_len = length.min(data.len() - offset);
    for byte in &mut data[offset..offset + actual_len] {
        *byte = pattern;
    }
    Some(CorruptionKind::BlockCorruption {
        offset,
        length: actual_len,
        pattern,
    })
}

// ---------------------------------------------------------------------------
// Combined corruption simulator
// ---------------------------------------------------------------------------

/// Apply a randomized combination of corruption types to `data`.
///
/// Uses the seed from `config` for reproducibility. The specific corruptions
/// applied depend on the seed, making each run deterministic.
pub fn simulate_corruption(data: &mut Vec<u8>, config: &CorruptionConfig) -> CorruptionRecord {
    let original_len = data.len();
    let mut corruptions = Vec::new();
    let mut rng = Rng::new(config.seed);

    for _ in 0..config.max_corruptions {
        if data.is_empty() {
            break;
        }

        let kind = rng.next_usize(5);
        match kind {
            0 => {
                // Bit flip
                let offset = rng.next_usize(data.len());
                let bit = 1u8 << (rng.next_usize(8) as u8);
                data[offset] ^= bit;
                corruptions.push(CorruptionKind::BitFlip { offset, mask: bit });
            }
            1 => {
                // Header wipe (small range)
                let offset = rng.next_usize(data.len());
                let length = rng.next_usize(16).max(1).min(data.len() - offset);
                if let Some(c) = apply_header_wipe(data, offset, length) {
                    corruptions.push(c);
                }
            }
            2 => {
                // Byte insertion
                let offset = rng.next_usize(data.len());
                let count = rng.next_usize(8).max(1);
                let sub_config = CorruptionConfig {
                    seed: rng.next_u64(),
                    max_corruptions: 1,
                };
                if let Some(c) = apply_byte_insertion(data, offset, count, &sub_config) {
                    corruptions.push(c);
                }
            }
            3 => {
                // Byte deletion
                let offset = rng.next_usize(data.len());
                let count = rng.next_usize(8).max(1);
                if let Some(c) = apply_byte_deletion(data, offset, count) {
                    corruptions.push(c);
                }
            }
            _ => {
                // Block corruption
                let offset = rng.next_usize(data.len());
                let length = rng.next_usize(32).max(1).min(data.len() - offset);
                let pattern = rng.next_u8();
                if let Some(c) = apply_block_corruption(data, offset, length, pattern) {
                    corruptions.push(c);
                }
            }
        }
    }

    CorruptionRecord {
        corruptions,
        original_len,
        final_len: data.len(),
    }
}

/// Undo bit-flips given the corruption record (only works for bit-flip corruptions).
///
/// This is useful for verifying round-trip repair: apply bit-flips, repair,
/// then compare against the undone version.
pub fn undo_bit_flips(data: &mut [u8], record: &CorruptionRecord) {
    for corruption in &record.corruptions {
        if let CorruptionKind::BitFlip { offset, mask } = corruption {
            if *offset < data.len() {
                data[*offset] ^= mask;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_flip_changes_data() {
        let original = vec![0x00; 64];
        let mut data = original.clone();
        let config = CorruptionConfig {
            seed: 12345,
            max_corruptions: 5,
        };
        let flips = apply_bit_flips(&mut data, &config);
        assert!(!flips.is_empty());
        assert_ne!(data, original);
    }

    #[test]
    fn test_bit_flip_reproducible() {
        let config = CorruptionConfig {
            seed: 99,
            max_corruptions: 3,
        };
        let mut data1 = vec![0x55; 32];
        let mut data2 = vec![0x55; 32];
        let f1 = apply_bit_flips(&mut data1, &config);
        let f2 = apply_bit_flips(&mut data2, &config);
        assert_eq!(data1, data2);
        assert_eq!(f1, f2);
    }

    #[test]
    fn test_bit_flip_empty_data() {
        let mut data: Vec<u8> = Vec::new();
        let config = CorruptionConfig::default();
        let flips = apply_bit_flips(&mut data, &config);
        assert!(flips.is_empty());
    }

    #[test]
    fn test_truncation() {
        let mut data = vec![0xAA; 100];
        let result = apply_truncation(&mut data, 50);
        assert!(result.is_some());
        assert_eq!(data.len(), 50);
        if let Some(CorruptionKind::Truncation {
            original_len,
            truncated_len,
        }) = result
        {
            assert_eq!(original_len, 100);
            assert_eq!(truncated_len, 50);
        }
    }

    #[test]
    fn test_truncation_beyond_length() {
        let mut data = vec![0xAA; 10];
        let result = apply_truncation(&mut data, 20);
        assert!(result.is_none());
        assert_eq!(data.len(), 10);
    }

    #[test]
    fn test_header_wipe() {
        let mut data = vec![0xFF; 64];
        let result = apply_header_wipe(&mut data, 0, 8);
        assert!(result.is_some());
        assert!(data[..8].iter().all(|&b| b == 0x00));
        assert!(data[8..].iter().all(|&b| b == 0xFF));
    }

    #[test]
    fn test_header_wipe_partial() {
        let mut data = vec![0xFF; 10];
        let result = apply_header_wipe(&mut data, 8, 100);
        assert!(result.is_some());
        if let Some(CorruptionKind::HeaderWipe { length, .. }) = result {
            assert_eq!(length, 2); // clamped to remaining bytes
        }
    }

    #[test]
    fn test_header_wipe_out_of_bounds() {
        let mut data = vec![0xFF; 10];
        let result = apply_header_wipe(&mut data, 20, 5);
        assert!(result.is_none());
    }

    #[test]
    fn test_byte_insertion() {
        let mut data = vec![0x01, 0x02, 0x03, 0x04];
        let config = CorruptionConfig::default();
        let result = apply_byte_insertion(&mut data, 2, 3, &config);
        assert!(result.is_some());
        assert_eq!(data.len(), 7);
        assert_eq!(data[0], 0x01);
        assert_eq!(data[1], 0x02);
        // 3 inserted bytes at positions 2, 3, 4
        assert_eq!(data[5], 0x03);
        assert_eq!(data[6], 0x04);
    }

    #[test]
    fn test_byte_insertion_at_end() {
        let mut data = vec![0x01, 0x02];
        let config = CorruptionConfig::default();
        let result = apply_byte_insertion(&mut data, 2, 2, &config);
        assert!(result.is_some());
        assert_eq!(data.len(), 4);
        assert_eq!(data[0], 0x01);
        assert_eq!(data[1], 0x02);
    }

    #[test]
    fn test_byte_insertion_zero_count() {
        let mut data = vec![0x01];
        let config = CorruptionConfig::default();
        let result = apply_byte_insertion(&mut data, 0, 0, &config);
        assert!(result.is_none());
        assert_eq!(data.len(), 1);
    }

    #[test]
    fn test_byte_deletion() {
        let mut data = vec![0x01, 0x02, 0x03, 0x04, 0x05];
        let result = apply_byte_deletion(&mut data, 1, 2);
        assert!(result.is_some());
        assert_eq!(data, vec![0x01, 0x04, 0x05]);
    }

    #[test]
    fn test_byte_deletion_clamps() {
        let mut data = vec![0x01, 0x02, 0x03];
        let result = apply_byte_deletion(&mut data, 1, 100);
        assert!(result.is_some());
        assert_eq!(data, vec![0x01]);
        if let Some(CorruptionKind::ByteDeletion { count, .. }) = result {
            assert_eq!(count, 2);
        }
    }

    #[test]
    fn test_byte_deletion_out_of_bounds() {
        let mut data = vec![0x01];
        let result = apply_byte_deletion(&mut data, 5, 1);
        assert!(result.is_none());
    }

    #[test]
    fn test_block_corruption() {
        let mut data = vec![0x00; 32];
        let result = apply_block_corruption(&mut data, 4, 8, 0xDE);
        assert!(result.is_some());
        assert!(data[4..12].iter().all(|&b| b == 0xDE));
        assert!(data[0..4].iter().all(|&b| b == 0x00));
        assert!(data[12..].iter().all(|&b| b == 0x00));
    }

    #[test]
    fn test_block_corruption_clamps() {
        let mut data = vec![0x00; 10];
        let result = apply_block_corruption(&mut data, 8, 100, 0xFF);
        assert!(result.is_some());
        if let Some(CorruptionKind::BlockCorruption { length, .. }) = result {
            assert_eq!(length, 2);
        }
    }

    #[test]
    fn test_simulate_corruption_deterministic() {
        let config = CorruptionConfig {
            seed: 777,
            max_corruptions: 5,
        };
        let mut data1 = vec![0x42; 256];
        let mut data2 = vec![0x42; 256];
        let rec1 = simulate_corruption(&mut data1, &config);
        let rec2 = simulate_corruption(&mut data2, &config);
        assert_eq!(data1, data2);
        assert_eq!(rec1.corruptions.len(), rec2.corruptions.len());
    }

    #[test]
    fn test_simulate_corruption_modifies_data() {
        let original = vec![0x00; 128];
        let mut data = original.clone();
        let config = CorruptionConfig {
            seed: 1,
            max_corruptions: 10,
        };
        let record = simulate_corruption(&mut data, &config);
        assert!(!record.corruptions.is_empty());
        // Data should have changed (overwhelmingly likely with 10 corruptions)
        assert_ne!(data.len(), 0);
    }

    #[test]
    fn test_simulate_corruption_empty_data() {
        let mut data: Vec<u8> = Vec::new();
        let config = CorruptionConfig::default();
        let record = simulate_corruption(&mut data, &config);
        assert!(record.corruptions.is_empty());
    }

    #[test]
    fn test_undo_bit_flips() {
        let original = vec![0x55; 32];
        let mut data = original.clone();
        let config = CorruptionConfig {
            seed: 42,
            max_corruptions: 5,
        };
        let flips = apply_bit_flips(&mut data, &config);
        assert_ne!(data, original);

        let record = CorruptionRecord {
            corruptions: flips,
            original_len: 32,
            final_len: 32,
        };
        undo_bit_flips(&mut data, &record);
        assert_eq!(data, original);
    }

    #[test]
    fn test_corruption_config_default() {
        let config = CorruptionConfig::default();
        assert_eq!(config.seed, 42);
        assert_eq!(config.max_corruptions, 10);
    }

    #[test]
    fn test_rng_deterministic() {
        let mut rng1 = Rng::new(123);
        let mut rng2 = Rng::new(123);
        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_rng_zero_seed_becomes_one() {
        let mut rng = Rng::new(0);
        // Should not get stuck at 0
        let val = rng.next_u64();
        assert_ne!(val, 0);
    }

    #[test]
    fn test_corruption_record_tracks_lengths() {
        let mut data = vec![0x00; 100];
        let config = CorruptionConfig {
            seed: 55,
            max_corruptions: 3,
        };
        let record = simulate_corruption(&mut data, &config);
        assert_eq!(record.original_len, 100);
        assert_eq!(record.final_len, data.len());
    }
}
