//! I/O integrity verification via checksums.
//!
//! Provides several checksum algorithms (CRC-32, Adler-32, FNV-1a 64,
//! and simple hash variants), an algorithm dispatcher, and an
//! `IntegrityVerifier` that records expected checksums and verifies
//! them later.

#![allow(dead_code)]

use std::collections::HashMap;

// ──────────────────────────────────────────────────────────────────────────────
// CRC-32 table (Ethernet / PKzip polynomial 0xEDB88320, reflected)
// ──────────────────────────────────────────────────────────────────────────────

/// Generate the standard CRC-32 lookup table using polynomial 0xEDB88320
#[must_use]
pub fn crc32_table() -> [u32; 256] {
    let poly: u32 = 0xEDB8_8320;
    let mut table = [0u32; 256];
    for i in 0..256u32 {
        let mut crc = i;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ poly;
            } else {
                crc >>= 1;
            }
        }
        table[i as usize] = crc;
    }
    table
}

/// Compute CRC-32 (Ethernet / `PKzip`) of `data`
#[must_use]
pub fn crc32(data: &[u8]) -> u32 {
    let table = crc32_table();
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        let idx = ((crc ^ u32::from(byte)) & 0xFF) as usize;
        crc = (crc >> 8) ^ table[idx];
    }
    crc ^ 0xFFFF_FFFF
}

// ──────────────────────────────────────────────────────────────────────────────
// Adler-32
// ──────────────────────────────────────────────────────────────────────────────

const MOD_ADLER: u32 = 65521;

/// Compute Adler-32 checksum of `data`
#[must_use]
pub fn adler32(data: &[u8]) -> u32 {
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    for &byte in data {
        a = (a + u32::from(byte)) % MOD_ADLER;
        b = (b + a) % MOD_ADLER;
    }
    (b << 16) | a
}

// ──────────────────────────────────────────────────────────────────────────────
// FNV-1a 64-bit
// ──────────────────────────────────────────────────────────────────────────────

const FNV1A_OFFSET_BASIS: u64 = 14_695_981_039_346_656_037;
const FNV1A_PRIME: u64 = 1_099_511_628_211;

/// Compute FNV-1a 64-bit hash of `data`
#[must_use]
pub fn fnv1a_64(data: &[u8]) -> u64 {
    let mut hash = FNV1A_OFFSET_BASIS;
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV1A_PRIME);
    }
    hash
}

// ──────────────────────────────────────────────────────────────────────────────
// XxHash-64 (simplified / self-contained implementation)
// ──────────────────────────────────────────────────────────────────────────────

/// Compute a simplified 64-bit hash inspired by xxHash (seed = 0)
///
/// This is not a bit-for-bit compatible xxHash64 implementation but provides
/// a fast, avalanche-quality hash suitable for integrity checking within this
/// codebase without external dependencies.
fn xxhash64_simple(data: &[u8]) -> u64 {
    const P1: u64 = 0x9E37_79B1_85EB_CA87;
    const P2: u64 = 0xC2B2_AE3D_27D4_EB4F;
    const P3: u64 = 0x1656_67B1_9E37_79F9;
    const P4: u64 = 0x85EB_CA77_C2B2_AE63;
    const P5: u64 = 0x27D4_EB2F_1656_67C5;

    let len = data.len() as u64;
    let mut h: u64;
    let mut p = 0usize;

    if data.len() >= 32 {
        let mut v1 = 0u64.wrapping_add(P1).wrapping_add(P2);
        let mut v2 = 0u64.wrapping_add(P2);
        let mut v3: u64 = 0;
        let mut v4 = 0u64.wrapping_sub(P1);

        while p + 32 <= data.len() {
            let read =
                |off: usize| u64::from_le_bytes(data[off..off + 8].try_into().unwrap_or([0u8; 8]));
            v1 = v1
                .wrapping_add(read(p).wrapping_mul(P2))
                .rotate_left(31)
                .wrapping_mul(P1);
            v2 = v2
                .wrapping_add(read(p + 8).wrapping_mul(P2))
                .rotate_left(31)
                .wrapping_mul(P1);
            v3 = v3
                .wrapping_add(read(p + 16).wrapping_mul(P2))
                .rotate_left(31)
                .wrapping_mul(P1);
            v4 = v4
                .wrapping_add(read(p + 24).wrapping_mul(P2))
                .rotate_left(31)
                .wrapping_mul(P1);
            p += 32;
        }

        h = v1
            .rotate_left(1)
            .wrapping_add(v2.rotate_left(7))
            .wrapping_add(v3.rotate_left(12))
            .wrapping_add(v4.rotate_left(18));
        h ^= v1.wrapping_mul(P2).rotate_left(31).wrapping_mul(P1);
        h = h.wrapping_mul(P1).wrapping_add(P4);
        h ^= v2.wrapping_mul(P2).rotate_left(31).wrapping_mul(P1);
        h = h.wrapping_mul(P1).wrapping_add(P4);
        h ^= v3.wrapping_mul(P2).rotate_left(31).wrapping_mul(P1);
        h = h.wrapping_mul(P1).wrapping_add(P4);
        h ^= v4.wrapping_mul(P2).rotate_left(31).wrapping_mul(P1);
        h = h.wrapping_mul(P1).wrapping_add(P4);
    } else {
        h = 0u64.wrapping_add(P5);
    }

    h = h.wrapping_add(len);

    while p + 8 <= data.len() {
        let lane = u64::from_le_bytes(data[p..p + 8].try_into().unwrap_or([0u8; 8]));
        h ^= lane.wrapping_mul(P2).rotate_left(31).wrapping_mul(P1);
        h = h.rotate_left(27).wrapping_mul(P1).wrapping_add(P4);
        p += 8;
    }

    if p + 4 <= data.len() {
        let lane = u64::from(u32::from_le_bytes(
            data[p..p + 4].try_into().unwrap_or([0u8; 4]),
        ));
        h ^= lane.wrapping_mul(P1);
        h = h.rotate_left(23).wrapping_mul(P2).wrapping_add(P3);
        p += 4;
    }

    for &byte in &data[p..] {
        h ^= u64::from(byte).wrapping_mul(P5);
        h = h.rotate_left(11).wrapping_mul(P1);
    }

    h ^= h >> 33;
    h = h.wrapping_mul(P2);
    h ^= h >> 29;
    h = h.wrapping_mul(P3);
    h ^= h >> 32;
    h
}

// ──────────────────────────────────────────────────────────────────────────────
// Md5Lite — a non-cryptographic 64-bit hash for quick integrity checks
// ──────────────────────────────────────────────────────────────────────────────

/// Lightweight non-cryptographic hash inspired by MD5 mixing (64-bit output)
fn md5_lite(data: &[u8]) -> u64 {
    // Use a combination of additive and multiplicative mixing for avalanche
    let mut h0: u64 = 0x6745_2301_EFCD_AB89;
    let mut h1: u64 = 0x98BA_DCFE_1032_5476;

    for (i, chunk) in data.chunks(8).enumerate() {
        let mut word = 0u64;
        for (j, &b) in chunk.iter().enumerate() {
            word |= u64::from(b) << (j * 8);
        }
        word ^= i as u64;
        h0 = h0
            .wrapping_add(word)
            .rotate_left(17)
            .wrapping_mul(0xBF58_476D_1CE4_E5B9);
        h1 = h1
            .wrapping_add(word)
            .rotate_left(31)
            .wrapping_mul(0x94D0_49BB_1331_11EB);
        h0 ^= h1.rotate_right(13);
        h1 ^= h0.rotate_left(9);
    }

    h0 ^ h1
}

// ──────────────────────────────────────────────────────────────────────────────
// ChecksumAlgorithm
// ──────────────────────────────────────────────────────────────────────────────

/// Checksum algorithm selector
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChecksumAlgorithm {
    /// CRC-32 (Ethernet / `PKzip` polynomial)
    Crc32,
    /// CRC-32C (Castagnoli polynomial — same polynomial as iSCSI/SSE4.2)
    Crc32c,
    /// Adler-32 (zlib variant)
    Adler32,
    /// 64-bit xxHash (simplified implementation, seed=0)
    Xxhash64,
    /// FNV-1a 64-bit
    FnvHash64,
    /// Lightweight 64-bit hash (MD5-inspired mixing, non-cryptographic)
    Md5Lite,
}

impl ChecksumAlgorithm {
    /// Human-readable algorithm name
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Crc32 => "CRC-32",
            Self::Crc32c => "CRC-32C",
            Self::Adler32 => "Adler-32",
            Self::Xxhash64 => "xxHash-64",
            Self::FnvHash64 => "FNV-1a-64",
            Self::Md5Lite => "Md5Lite",
        }
    }

    /// Compute the checksum of `data`, returning a 64-bit value.
    ///
    /// CRC-32 and Adler-32 results are zero-extended from 32 bits.
    #[must_use]
    pub fn compute(self, data: &[u8]) -> u64 {
        match self {
            Self::Crc32 => u64::from(crc32(data)),
            Self::Crc32c => u64::from(crc32c(data)),
            Self::Adler32 => u64::from(adler32(data)),
            Self::Xxhash64 => xxhash64_simple(data),
            Self::FnvHash64 => fnv1a_64(data),
            Self::Md5Lite => md5_lite(data),
        }
    }
}

/// Compute CRC-32C (Castagnoli) using polynomial 0x82F63B78
#[must_use]
pub fn crc32c(data: &[u8]) -> u32 {
    let poly: u32 = 0x82F6_3B78;
    let mut table = [0u32; 256];
    for i in 0..256u32 {
        let mut crc = i;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ poly;
            } else {
                crc >>= 1;
            }
        }
        table[i as usize] = crc;
    }

    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        let idx = ((crc ^ u32::from(byte)) & 0xFF) as usize;
        crc = (crc >> 8) ^ table[idx];
    }
    crc ^ 0xFFFF_FFFF
}

// ──────────────────────────────────────────────────────────────────────────────
// IntegrityVerifier
// ──────────────────────────────────────────────────────────────────────────────

/// Result of verifying a file's checksum
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyResult {
    /// The computed checksum matches the recorded one
    Match,
    /// The checksums differ (expected, actual)
    Mismatch(u64, u64),
    /// No checksum was previously recorded for this path
    NotFound,
}

impl VerifyResult {
    /// Returns `true` if the result is `Match`
    #[must_use]
    pub fn is_match(&self) -> bool {
        matches!(self, Self::Match)
    }
}

impl std::fmt::Display for VerifyResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Match => write!(f, "Match"),
            Self::Mismatch(exp, got) => {
                write!(f, "Mismatch (expected {exp:#018x}, got {got:#018x})")
            }
            Self::NotFound => write!(f, "NotFound"),
        }
    }
}

/// Records expected checksums and verifies computed values against them
pub struct IntegrityVerifier {
    records: HashMap<String, u64>,
}

impl IntegrityVerifier {
    /// Create a new, empty verifier
    #[must_use]
    pub fn new() -> Self {
        Self {
            records: HashMap::new(),
        }
    }

    /// Record the expected checksum for `path`
    pub fn record(&mut self, path: &str, checksum: u64) {
        self.records.insert(path.to_string(), checksum);
    }

    /// Verify that `computed` matches the recorded checksum for `path`.
    #[must_use]
    pub fn verify(&self, path: &str, computed: u64) -> VerifyResult {
        match self.records.get(path) {
            None => VerifyResult::NotFound,
            Some(&expected) if expected == computed => VerifyResult::Match,
            Some(&expected) => VerifyResult::Mismatch(expected, computed),
        }
    }

    /// Remove the recorded checksum for `path`
    pub fn remove(&mut self, path: &str) {
        self.records.remove(path);
    }

    /// Returns the number of recorded entries
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Returns `true` if no checksums have been recorded
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

impl Default for IntegrityVerifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── crc32_table ────────────────────────────────────────────────────────────

    #[test]
    fn test_crc32_table_length() {
        let table = crc32_table();
        assert_eq!(table.len(), 256);
    }

    #[test]
    fn test_crc32_table_first_entry() {
        let table = crc32_table();
        assert_eq!(table[0], 0); // CRC of a zero byte with itself XOR'd = 0 after finishing
    }

    // ── crc32 ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_crc32_known_value() {
        // CRC-32 of "123456789" = 0xCBF43926
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn test_crc32_empty() {
        assert_eq!(crc32(b""), 0x0000_0000);
    }

    #[test]
    fn test_crc32_deterministic() {
        let a = crc32(b"hello world");
        let b = crc32(b"hello world");
        assert_eq!(a, b);
    }

    // ── adler32 ───────────────────────────────────────────────────────────────

    #[test]
    fn test_adler32_known_value() {
        // Adler-32 of "Wikipedia" = 0x11E60398
        assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
    }

    #[test]
    fn test_adler32_empty() {
        assert_eq!(adler32(b""), 1); // a=1, b=0 => 0x00000001
    }

    #[test]
    fn test_adler32_deterministic() {
        assert_eq!(adler32(b"oximedia"), adler32(b"oximedia"));
    }

    // ── fnv1a_64 ──────────────────────────────────────────────────────────────

    #[test]
    fn test_fnv1a_64_empty() {
        assert_eq!(fnv1a_64(b""), FNV1A_OFFSET_BASIS);
    }

    #[test]
    fn test_fnv1a_64_deterministic() {
        assert_eq!(fnv1a_64(b"test"), fnv1a_64(b"test"));
    }

    #[test]
    fn test_fnv1a_64_different_inputs() {
        assert_ne!(fnv1a_64(b"foo"), fnv1a_64(b"bar"));
    }

    // ── ChecksumAlgorithm ─────────────────────────────────────────────────────

    #[test]
    fn test_algorithm_names() {
        assert_eq!(ChecksumAlgorithm::Crc32.name(), "CRC-32");
        assert_eq!(ChecksumAlgorithm::Crc32c.name(), "CRC-32C");
        assert_eq!(ChecksumAlgorithm::Adler32.name(), "Adler-32");
        assert_eq!(ChecksumAlgorithm::Xxhash64.name(), "xxHash-64");
        assert_eq!(ChecksumAlgorithm::FnvHash64.name(), "FNV-1a-64");
        assert_eq!(ChecksumAlgorithm::Md5Lite.name(), "Md5Lite");
    }

    #[test]
    fn test_algorithm_compute_crc32() {
        let v = ChecksumAlgorithm::Crc32.compute(b"123456789");
        assert_eq!(v, 0xCBF4_3926);
    }

    #[test]
    fn test_algorithm_compute_deterministic() {
        let data = b"oximedia rocks";
        for algo in [
            ChecksumAlgorithm::Crc32,
            ChecksumAlgorithm::Crc32c,
            ChecksumAlgorithm::Adler32,
            ChecksumAlgorithm::Xxhash64,
            ChecksumAlgorithm::FnvHash64,
            ChecksumAlgorithm::Md5Lite,
        ] {
            assert_eq!(
                algo.compute(data),
                algo.compute(data),
                "{} not deterministic",
                algo.name()
            );
        }
    }

    // ── IntegrityVerifier ─────────────────────────────────────────────────────

    #[test]
    fn test_verifier_record_and_match() {
        let mut v = IntegrityVerifier::new();
        v.record("file.mp4", 12345);
        assert_eq!(v.verify("file.mp4", 12345), VerifyResult::Match);
    }

    #[test]
    fn test_verifier_mismatch() {
        let mut v = IntegrityVerifier::new();
        v.record("file.mp4", 12345);
        assert_eq!(
            v.verify("file.mp4", 99999),
            VerifyResult::Mismatch(12345, 99999)
        );
    }

    #[test]
    fn test_verifier_not_found() {
        let v = IntegrityVerifier::new();
        assert_eq!(v.verify("unknown.mp4", 0), VerifyResult::NotFound);
    }

    #[test]
    fn test_verifier_remove() {
        let mut v = IntegrityVerifier::new();
        v.record("a", 1);
        v.remove("a");
        assert_eq!(v.verify("a", 1), VerifyResult::NotFound);
    }

    #[test]
    fn test_verify_result_is_match() {
        assert!(VerifyResult::Match.is_match());
        assert!(!VerifyResult::Mismatch(1, 2).is_match());
        assert!(!VerifyResult::NotFound.is_match());
    }

    #[test]
    fn test_verify_result_display() {
        assert_eq!(VerifyResult::Match.to_string(), "Match");
        assert_eq!(VerifyResult::NotFound.to_string(), "NotFound");
        let s = VerifyResult::Mismatch(1, 2).to_string();
        assert!(s.contains("Mismatch"));
    }
}
