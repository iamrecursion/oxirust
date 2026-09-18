#![allow(dead_code)]
//! LUT fingerprinting for content identification and deduplication.
//!
//! This module generates compact fingerprints (hashes) from LUT data so that
//! identical or near-identical LUTs can be detected without comparing every
//! sample. It supports both exact matching (byte-level hash) and perceptual
//! matching (tolerance-based quantized hash).

use std::fmt;

/// A 128-bit fingerprint for a LUT.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct LutFingerprint {
    /// The raw 128-bit hash value stored as two u64 halves.
    pub hi: u64,
    /// Lower 64 bits of the hash.
    pub lo: u64,
}

impl LutFingerprint {
    /// Create a fingerprint from two 64-bit halves.
    #[must_use]
    pub const fn new(hi: u64, lo: u64) -> Self {
        Self { hi, lo }
    }

    /// Create a zero fingerprint.
    #[must_use]
    pub const fn zero() -> Self {
        Self { hi: 0, lo: 0 }
    }

    /// Check whether this fingerprint is all zeros.
    #[must_use]
    pub const fn is_zero(&self) -> bool {
        self.hi == 0 && self.lo == 0
    }

    /// Return the fingerprint as a 16-byte array (big-endian).
    #[must_use]
    pub fn to_bytes(&self) -> [u8; 16] {
        let mut out = [0u8; 16];
        out[..8].copy_from_slice(&self.hi.to_be_bytes());
        out[8..].copy_from_slice(&self.lo.to_be_bytes());
        out
    }

    /// Construct a fingerprint from a 16-byte big-endian array.
    #[must_use]
    pub fn from_bytes(bytes: &[u8; 16]) -> Self {
        let hi = u64::from_be_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]);
        let lo = u64::from_be_bytes([
            bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
        ]);
        Self { hi, lo }
    }

    /// Compute the Hamming distance (bit-level) between two fingerprints.
    #[must_use]
    pub fn hamming_distance(&self, other: &Self) -> u32 {
        (self.hi ^ other.hi).count_ones() + (self.lo ^ other.lo).count_ones()
    }
}

impl fmt::Debug for LutFingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LutFingerprint({:016x}{:016x})", self.hi, self.lo)
    }
}

impl fmt::Display for LutFingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:016x}{:016x}", self.hi, self.lo)
    }
}

/// FNV-1a 64-bit hash constant.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a 64-bit prime.
const FNV_PRIME: u64 = 0x0100_0000_01b3;

/// Internal FNV-1a streaming hasher (64-bit).
struct Fnv1a {
    state: u64,
}

impl Fnv1a {
    fn new() -> Self {
        Self { state: FNV_OFFSET }
    }

    fn update(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.state ^= u64::from(b);
            self.state = self.state.wrapping_mul(FNV_PRIME);
        }
    }

    fn finish(&self) -> u64 {
        self.state
    }
}

/// Compute an exact fingerprint from raw f64 LUT data.
///
/// This hashes every byte of every sample, so even tiny floating-point
/// differences will produce a different fingerprint.
#[must_use]
pub fn fingerprint_exact(data: &[[f64; 3]]) -> LutFingerprint {
    let mut h1 = Fnv1a::new();
    let mut h2 = Fnv1a::new();
    // Seed h2 differently
    h2.state = 0x6c62_272e_07bb_0142;

    for rgb in data {
        for &ch in rgb {
            let bytes = ch.to_bits().to_le_bytes();
            h1.update(&bytes);
            h2.update(&bytes);
        }
    }

    LutFingerprint::new(h1.finish(), h2.finish())
}

/// Compute a perceptual fingerprint that quantizes values before hashing.
///
/// Values are rounded to the given number of decimal places before hashing.
/// This means LUTs that differ only by floating-point noise will produce
/// the same fingerprint.
///
/// # Arguments
/// * `data` - The LUT RGB data
/// * `precision` - Number of decimal places to keep (e.g. 4 means round to 0.0001)
#[must_use]
pub fn fingerprint_perceptual(data: &[[f64; 3]], precision: u32) -> LutFingerprint {
    let scale = 10.0_f64.powi(precision as i32);
    let mut h1 = Fnv1a::new();
    let mut h2 = Fnv1a::new();
    h2.state = 0x6c62_272e_07bb_0142;

    for rgb in data {
        for &ch in rgb {
            let quantized = (ch * scale).round() as i64;
            let bytes = quantized.to_le_bytes();
            h1.update(&bytes);
            h2.update(&bytes);
        }
    }

    LutFingerprint::new(h1.finish(), h2.finish())
}

/// Compute a fingerprint for a 1D LUT.
#[must_use]
pub fn fingerprint_1d(data: &[f64]) -> LutFingerprint {
    let mut h1 = Fnv1a::new();
    let mut h2 = Fnv1a::new();
    h2.state = 0x6c62_272e_07bb_0142;

    for &v in data {
        let bytes = v.to_bits().to_le_bytes();
        h1.update(&bytes);
        h2.update(&bytes);
    }

    LutFingerprint::new(h1.finish(), h2.finish())
}

/// Check whether two LUT fingerprints are perceptually similar.
///
/// Uses Hamming distance with a configurable threshold.
#[must_use]
pub fn fingerprints_similar(a: &LutFingerprint, b: &LutFingerprint, max_distance: u32) -> bool {
    a.hamming_distance(b) <= max_distance
}

/// A database entry associating a fingerprint with a LUT name.
#[derive(Clone, Debug)]
pub struct FingerprintEntry {
    /// Human-readable name or path for the LUT.
    pub name: String,
    /// The fingerprint of the LUT.
    pub fingerprint: LutFingerprint,
}

impl FingerprintEntry {
    /// Create a new fingerprint entry.
    #[must_use]
    pub fn new(name: impl Into<String>, fingerprint: LutFingerprint) -> Self {
        Self {
            name: name.into(),
            fingerprint,
        }
    }
}

/// A simple in-memory fingerprint index for LUT lookup.
#[derive(Clone, Debug, Default)]
pub struct FingerprintIndex {
    entries: Vec<FingerprintEntry>,
}

impl FingerprintIndex {
    /// Create an empty index.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Add a fingerprint entry to the index.
    pub fn insert(&mut self, entry: FingerprintEntry) {
        self.entries.push(entry);
    }

    /// Look up an exact match.
    #[must_use]
    pub fn find_exact(&self, fp: &LutFingerprint) -> Option<&FingerprintEntry> {
        self.entries.iter().find(|e| &e.fingerprint == fp)
    }

    /// Find all entries within a given Hamming distance.
    #[must_use]
    pub fn find_similar(&self, fp: &LutFingerprint, max_distance: u32) -> Vec<&FingerprintEntry> {
        self.entries
            .iter()
            .filter(|e| e.fingerprint.hamming_distance(fp) <= max_distance)
            .collect()
    }

    /// Return the number of entries in the index.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_lut() -> Vec<[f64; 3]> {
        vec![[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]]
    }

    #[test]
    fn test_fingerprint_exact_deterministic() {
        let data = sample_lut();
        let fp1 = fingerprint_exact(&data);
        let fp2 = fingerprint_exact(&data);
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn test_fingerprint_exact_different_data() {
        let a = vec![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]];
        let b = vec![[0.0, 0.0, 0.0], [0.9, 1.0, 1.0]];
        let fp_a = fingerprint_exact(&a);
        let fp_b = fingerprint_exact(&b);
        assert_ne!(fp_a, fp_b);
    }

    #[test]
    fn test_fingerprint_perceptual_ignores_noise() {
        let a = vec![[0.5000, 0.5000, 0.5000]];
        let b = vec![[0.500_049, 0.500_049, 0.500_049]];
        let fp_a = fingerprint_perceptual(&a, 4);
        let fp_b = fingerprint_perceptual(&b, 4);
        assert_eq!(fp_a, fp_b);
    }

    #[test]
    fn test_fingerprint_perceptual_detects_large_diff() {
        let a = vec![[0.5, 0.5, 0.5]];
        let b = vec![[0.6, 0.5, 0.5]];
        let fp_a = fingerprint_perceptual(&a, 4);
        let fp_b = fingerprint_perceptual(&b, 4);
        assert_ne!(fp_a, fp_b);
    }

    #[test]
    fn test_fingerprint_1d() {
        let data = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let fp1 = fingerprint_1d(&data);
        let fp2 = fingerprint_1d(&data);
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn test_hamming_distance_zero() {
        let fp = LutFingerprint::new(0xABCD, 0x1234);
        assert_eq!(fp.hamming_distance(&fp), 0);
    }

    #[test]
    fn test_hamming_distance_known() {
        let a = LutFingerprint::new(0, 0);
        let b = LutFingerprint::new(0, 1);
        assert_eq!(a.hamming_distance(&b), 1);
    }

    #[test]
    fn test_to_bytes_roundtrip() {
        let fp = LutFingerprint::new(0x0123_4567_89AB_CDEF, 0xFEDC_BA98_7654_3210);
        let bytes = fp.to_bytes();
        let recovered = LutFingerprint::from_bytes(&bytes);
        assert_eq!(fp, recovered);
    }

    #[test]
    fn test_zero_fingerprint() {
        let fp = LutFingerprint::zero();
        assert!(fp.is_zero());
        assert_eq!(fp.hi, 0);
        assert_eq!(fp.lo, 0);
    }

    #[test]
    fn test_fingerprints_similar() {
        let a = LutFingerprint::new(0, 0);
        let b = LutFingerprint::new(0, 3); // 2 bits differ
        assert!(fingerprints_similar(&a, &b, 2));
        assert!(!fingerprints_similar(&a, &b, 1));
    }

    #[test]
    fn test_display_format() {
        let fp = LutFingerprint::new(0x0000_0000_0000_00FF, 0x0000_0000_0000_0001);
        let s = format!("{fp}");
        assert_eq!(s, "00000000000000ff0000000000000001");
    }

    #[test]
    fn test_fingerprint_index_exact_lookup() {
        let data = sample_lut();
        let fp = fingerprint_exact(&data);
        let mut index = FingerprintIndex::new();
        index.insert(FingerprintEntry::new("test_lut", fp));
        assert_eq!(index.len(), 1);
        let found = index.find_exact(&fp);
        assert!(found.is_some());
        assert_eq!(found.expect("should succeed in test").name, "test_lut");
    }

    #[test]
    fn test_fingerprint_index_similar_lookup() {
        let mut index = FingerprintIndex::new();
        let fp1 = LutFingerprint::new(0, 0);
        let fp2 = LutFingerprint::new(0, 3);
        index.insert(FingerprintEntry::new("a", fp1));
        index.insert(FingerprintEntry::new("b", fp2));
        let query = LutFingerprint::new(0, 1);
        let results = index.find_similar(&query, 2);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_fingerprint_index_empty() {
        let index = FingerprintIndex::new();
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
        let fp = LutFingerprint::zero();
        assert!(index.find_exact(&fp).is_none());
    }
}
