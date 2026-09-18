#![allow(dead_code)]
//! Checksum computation and verification for proxy media files.
//!
//! This module provides several hash algorithms (CRC-32, a simple FNV-style
//! hash, and a byte-sum digest) that can be computed incrementally over byte
//! slices. It is used to verify that proxy files have not been corrupted or
//! silently modified after generation.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Algorithms
// ---------------------------------------------------------------------------

/// Supported checksum algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChecksumAlgo {
    /// CRC-32 (ISO 3309 polynomial).
    Crc32,
    /// FNV-1a 64-bit hash.
    Fnv1a64,
    /// Simple additive byte-sum (mod 2^64).
    ByteSum,
}

impl fmt::Display for ChecksumAlgo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Crc32 => write!(f, "CRC-32"),
            Self::Fnv1a64 => write!(f, "FNV-1a-64"),
            Self::ByteSum => write!(f, "ByteSum"),
        }
    }
}

// ---------------------------------------------------------------------------
// CRC-32 table (pre-computed for ISO 3309 polynomial 0xEDB88320)
// ---------------------------------------------------------------------------

/// Build the CRC-32 lookup table at compile time.
const fn build_crc32_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0u32;
    while i < 256 {
        let mut crc = i;
        let mut j = 0;
        while j < 8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
            j += 1;
        }
        table[i as usize] = crc;
        i += 1;
    }
    table
}

/// Pre-computed CRC-32 lookup table.
const CRC32_TABLE: [u32; 256] = build_crc32_table();

// ---------------------------------------------------------------------------
// Hasher state
// ---------------------------------------------------------------------------

/// Incremental checksum state.
#[derive(Debug, Clone)]
pub struct ChecksumState {
    /// Which algorithm is in use.
    algo: ChecksumAlgo,
    /// Internal state — interpretation depends on `algo`.
    state: u64,
    /// Total bytes fed so far.
    bytes_processed: u64,
}

impl ChecksumState {
    /// Create a new checksum state for the given algorithm.
    pub fn new(algo: ChecksumAlgo) -> Self {
        let initial = match algo {
            ChecksumAlgo::Crc32 => 0xFFFF_FFFF_u64,
            ChecksumAlgo::Fnv1a64 => 0xcbf2_9ce4_8422_2325_u64,
            ChecksumAlgo::ByteSum => 0,
        };
        Self {
            algo,
            state: initial,
            bytes_processed: 0,
        }
    }

    /// Feed a byte slice into the hasher.
    pub fn update(&mut self, data: &[u8]) {
        match self.algo {
            ChecksumAlgo::Crc32 => {
                let mut crc = self.state as u32;
                for &b in data {
                    let idx = ((crc ^ u32::from(b)) & 0xFF) as usize;
                    crc = (crc >> 8) ^ CRC32_TABLE[idx];
                }
                self.state = u64::from(crc);
            }
            ChecksumAlgo::Fnv1a64 => {
                let mut h = self.state;
                for &b in data {
                    h ^= u64::from(b);
                    h = h.wrapping_mul(0x0100_0000_01b3);
                }
                self.state = h;
            }
            ChecksumAlgo::ByteSum => {
                for &b in data {
                    self.state = self.state.wrapping_add(u64::from(b));
                }
            }
        }
        self.bytes_processed += data.len() as u64;
    }

    /// Finalize and return the checksum value.
    pub fn finalize(&self) -> u64 {
        match self.algo {
            ChecksumAlgo::Crc32 => u64::from((self.state as u32) ^ 0xFFFF_FFFF),
            ChecksumAlgo::Fnv1a64 | ChecksumAlgo::ByteSum => self.state,
        }
    }

    /// Return the algorithm.
    pub fn algorithm(&self) -> ChecksumAlgo {
        self.algo
    }

    /// Return the number of bytes processed so far.
    pub fn bytes_processed(&self) -> u64 {
        self.bytes_processed
    }

    /// Reset to initial state.
    pub fn reset(&mut self) {
        *self = Self::new(self.algo);
    }
}

// ---------------------------------------------------------------------------
// One-shot convenience
// ---------------------------------------------------------------------------

/// Compute a checksum over a byte slice in one call.
pub fn checksum(algo: ChecksumAlgo, data: &[u8]) -> u64 {
    let mut state = ChecksumState::new(algo);
    state.update(data);
    state.finalize()
}

/// Compute a checksum and return it as a hex string.
pub fn checksum_hex(algo: ChecksumAlgo, data: &[u8]) -> String {
    let val = checksum(algo, data);
    match algo {
        ChecksumAlgo::Crc32 => format!("{:08x}", val as u32),
        ChecksumAlgo::Fnv1a64 | ChecksumAlgo::ByteSum => format!("{val:016x}"),
    }
}

// ---------------------------------------------------------------------------
// Verification record
// ---------------------------------------------------------------------------

/// A stored checksum record that can be verified later.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChecksumRecord {
    /// Path or identifier of the file.
    pub path: String,
    /// Algorithm used.
    pub algo: ChecksumAlgo,
    /// Expected checksum value.
    pub expected: u64,
}

impl ChecksumRecord {
    /// Create a new record.
    pub fn new(path: impl Into<String>, algo: ChecksumAlgo, expected: u64) -> Self {
        Self {
            path: path.into(),
            algo,
            expected,
        }
    }

    /// Verify a byte slice against this record.
    pub fn verify(&self, data: &[u8]) -> bool {
        checksum(self.algo, data) == self.expected
    }
}

// ---------------------------------------------------------------------------
// Checksum registry
// ---------------------------------------------------------------------------

/// Registry that stores and verifies checksums for multiple files.
#[derive(Debug, Clone)]
pub struct ChecksumRegistry {
    /// Records keyed by path.
    records: HashMap<String, ChecksumRecord>,
}

impl ChecksumRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            records: HashMap::new(),
        }
    }

    /// Register a checksum record.
    pub fn register(&mut self, record: ChecksumRecord) {
        self.records.insert(record.path.clone(), record);
    }

    /// Compute and register a checksum for a given path/data pair.
    pub fn compute_and_register(
        &mut self,
        path: impl Into<String>,
        algo: ChecksumAlgo,
        data: &[u8],
    ) {
        let p = path.into();
        let val = checksum(algo, data);
        self.register(ChecksumRecord::new(p, algo, val));
    }

    /// Verify data against a stored record.
    pub fn verify(&self, path: &str, data: &[u8]) -> Option<bool> {
        self.records.get(path).map(|r| r.verify(data))
    }

    /// Number of registered records.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Check if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Remove a record.
    pub fn remove(&mut self, path: &str) -> bool {
        self.records.remove(path).is_some()
    }

    /// List all registered paths.
    pub fn paths(&self) -> Vec<&str> {
        self.records.keys().map(String::as_str).collect()
    }
}

impl Default for ChecksumRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crc32_empty() {
        assert_eq!(checksum(ChecksumAlgo::Crc32, b""), 0);
    }

    #[test]
    fn test_crc32_known_value() {
        // "123456789" has well-known CRC-32 = 0xCBF43926
        let val = checksum(ChecksumAlgo::Crc32, b"123456789");
        assert_eq!(val, 0xCBF4_3926);
    }

    #[test]
    fn test_crc32_incremental_matches_one_shot() {
        let data = b"Hello, proxy world!";
        let one_shot = checksum(ChecksumAlgo::Crc32, data);

        let mut state = ChecksumState::new(ChecksumAlgo::Crc32);
        state.update(&data[..5]);
        state.update(&data[5..]);
        assert_eq!(state.finalize(), one_shot);
    }

    #[test]
    fn test_fnv1a_empty() {
        let val = checksum(ChecksumAlgo::Fnv1a64, b"");
        assert_eq!(val, 0xcbf2_9ce4_8422_2325);
    }

    #[test]
    fn test_fnv1a_nonempty() {
        let a = checksum(ChecksumAlgo::Fnv1a64, b"abc");
        let b = checksum(ChecksumAlgo::Fnv1a64, b"xyz");
        assert_ne!(a, b);
    }

    #[test]
    fn test_bytesum_basic() {
        let val = checksum(ChecksumAlgo::ByteSum, &[1, 2, 3, 4]);
        assert_eq!(val, 10);
    }

    #[test]
    fn test_checksum_hex_crc32() {
        let hex = checksum_hex(ChecksumAlgo::Crc32, b"123456789");
        assert_eq!(hex, "cbf43926");
    }

    #[test]
    fn test_checksum_hex_fnv() {
        let hex = checksum_hex(ChecksumAlgo::Fnv1a64, b"");
        assert_eq!(hex.len(), 16);
    }

    #[test]
    fn test_checksum_state_reset() {
        let mut state = ChecksumState::new(ChecksumAlgo::ByteSum);
        state.update(&[10, 20]);
        assert_eq!(state.bytes_processed(), 2);
        state.reset();
        assert_eq!(state.bytes_processed(), 0);
        assert_eq!(state.finalize(), 0);
    }

    #[test]
    fn test_record_verify() {
        let data = b"test data";
        let val = checksum(ChecksumAlgo::Crc32, data);
        let rec = ChecksumRecord::new("file.mp4", ChecksumAlgo::Crc32, val);
        assert!(rec.verify(data));
        assert!(!rec.verify(b"tampered"));
    }

    #[test]
    fn test_registry_compute_and_verify() {
        let mut reg = ChecksumRegistry::new();
        reg.compute_and_register("proxy.mp4", ChecksumAlgo::Fnv1a64, b"content");
        assert_eq!(reg.len(), 1);
        assert_eq!(reg.verify("proxy.mp4", b"content"), Some(true));
        assert_eq!(reg.verify("proxy.mp4", b"wrong"), Some(false));
        assert_eq!(reg.verify("missing.mp4", b"content"), None);
    }

    #[test]
    fn test_registry_remove() {
        let mut reg = ChecksumRegistry::new();
        reg.compute_and_register("a.mp4", ChecksumAlgo::ByteSum, b"x");
        assert!(reg.remove("a.mp4"));
        assert!(!reg.remove("a.mp4"));
        assert!(reg.is_empty());
    }

    #[test]
    fn test_algo_display() {
        assert_eq!(format!("{}", ChecksumAlgo::Crc32), "CRC-32");
        assert_eq!(format!("{}", ChecksumAlgo::Fnv1a64), "FNV-1a-64");
        assert_eq!(format!("{}", ChecksumAlgo::ByteSum), "ByteSum");
    }
}
