#![allow(dead_code)]
//! Essence file hashing and integrity verification for IMF packages.
//!
//! IMF packages require hash verification of all essence files (MXF track files)
//! to ensure integrity during delivery. This module provides hash computation,
//! caching, and verification workflows.

use std::collections::HashMap;
use std::fmt;
use std::time::SystemTime;

/// Supported hash algorithms for IMF essence verification.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HashAlgo {
    /// SHA-1 (legacy, SMPTE ST 429-8).
    Sha1,
    /// SHA-256 (recommended, SMPTE ST 429-8:2014).
    Sha256,
    /// SHA-512 (high-security archival use).
    Sha512,
    /// MD5 (legacy, not recommended for new packages).
    Md5,
}

impl fmt::Display for HashAlgo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sha1 => write!(f, "SHA-1"),
            Self::Sha256 => write!(f, "SHA-256"),
            Self::Sha512 => write!(f, "SHA-512"),
            Self::Md5 => write!(f, "MD5"),
        }
    }
}

impl HashAlgo {
    /// Returns the hash digest length in bytes.
    pub fn digest_length(&self) -> usize {
        match self {
            Self::Sha1 => 20,
            Self::Sha256 => 32,
            Self::Sha512 => 64,
            Self::Md5 => 16,
        }
    }

    /// Returns the hash digest length in hex characters.
    pub fn hex_length(&self) -> usize {
        self.digest_length() * 2
    }
}

// ── Cryptographic computation ─────────────────────────────────────────────────

/// Compute a hash over the given byte slice using the specified algorithm.
///
/// Returns the raw digest bytes.
///
/// # Example
/// ```
/// use oximedia_imf::essence_hash::{compute_hash, HashAlgo};
/// let digest = compute_hash(b"hello world", HashAlgo::Sha256);
/// assert_eq!(digest.len(), 32);
/// ```
pub fn compute_hash(data: &[u8], algorithm: HashAlgo) -> Vec<u8> {
    match algorithm {
        HashAlgo::Sha1 => {
            use sha1::Digest;
            sha1::Sha1::digest(data).to_vec()
        }
        HashAlgo::Sha256 => {
            use sha2::Digest;
            sha2::Sha256::digest(data).to_vec()
        }
        HashAlgo::Sha512 => {
            use sha2::Digest;
            sha2::Sha512::digest(data).to_vec()
        }
        HashAlgo::Md5 => {
            use md5::Digest;
            md5::Md5::digest(data).to_vec()
        }
    }
}

/// Compute a hex-encoded hash over the given byte slice.
pub fn compute_hash_hex(data: &[u8], algorithm: HashAlgo) -> String {
    let raw = compute_hash(data, algorithm);
    hex::encode(raw)
}

// ── Streaming / incremental hash context ─────────────────────────────────────

/// Internal state for an incremental (streaming) hash computation.
///
/// Unlike [`IncrementalHasher`] (which buffers all chunks), `HashContext`
/// uses real cryptographic streaming APIs so that arbitrarily large data can
/// be hashed without accumulating everything in memory.
pub struct HashContext {
    inner: HashContextInner,
}

enum HashContextInner {
    Sha1(sha1::Sha1),
    Sha256(sha2::Sha256),
    Sha512(sha2::Sha512),
    Md5(md5::Md5),
}

impl HashContext {
    /// Create a new [`HashContext`] for the given algorithm.
    pub fn new(algorithm: HashAlgo) -> Self {
        let inner = match algorithm {
            HashAlgo::Sha1 => {
                use sha1::Digest;
                HashContextInner::Sha1(sha1::Sha1::new())
            }
            HashAlgo::Sha256 => {
                use sha2::Digest;
                HashContextInner::Sha256(sha2::Sha256::new())
            }
            HashAlgo::Sha512 => {
                use sha2::Digest;
                HashContextInner::Sha512(sha2::Sha512::new())
            }
            HashAlgo::Md5 => {
                use md5::Digest;
                HashContextInner::Md5(md5::Md5::new())
            }
        };
        Self { inner }
    }

    /// Feed a chunk of data into the hasher.
    pub fn update(&mut self, chunk: &[u8]) {
        match &mut self.inner {
            HashContextInner::Sha1(h) => {
                use sha1::Digest;
                h.update(chunk);
            }
            HashContextInner::Sha256(h) => {
                use sha2::Digest;
                h.update(chunk);
            }
            HashContextInner::Sha512(h) => {
                use sha2::Digest;
                h.update(chunk);
            }
            HashContextInner::Md5(h) => {
                use md5::Digest;
                h.update(chunk);
            }
        }
    }

    /// Finalize and return the raw digest bytes, consuming `self`.
    pub fn finalize(self) -> Vec<u8> {
        match self.inner {
            HashContextInner::Sha1(h) => {
                use sha1::Digest;
                h.finalize().to_vec()
            }
            HashContextInner::Sha256(h) => {
                use sha2::Digest;
                h.finalize().to_vec()
            }
            HashContextInner::Sha512(h) => {
                use sha2::Digest;
                h.finalize().to_vec()
            }
            HashContextInner::Md5(h) => {
                use md5::Digest;
                h.finalize().to_vec()
            }
        }
    }

    /// Finalize and return the hex-encoded digest, consuming `self`.
    pub fn finalize_hex(self) -> String {
        hex::encode(self.finalize())
    }
}

/// A computed hash value for an essence file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EssenceHash {
    /// The algorithm used.
    pub algorithm: HashAlgo,
    /// The hex-encoded digest value.
    pub hex_digest: String,
}

impl EssenceHash {
    /// Creates a new essence hash.
    pub fn new(algo: HashAlgo, hex: &str) -> Self {
        Self {
            algorithm: algo,
            hex_digest: hex.to_lowercase(),
        }
    }

    /// Validates the hex digest format (correct length and valid hex).
    pub fn is_valid_format(&self) -> bool {
        let expected_len = self.algorithm.hex_length();
        self.hex_digest.len() == expected_len
            && self.hex_digest.chars().all(|c| c.is_ascii_hexdigit())
    }

    /// Checks if this hash matches another hash.
    pub fn matches(&self, other: &Self) -> bool {
        self.algorithm == other.algorithm
            && self.hex_digest.to_lowercase() == other.hex_digest.to_lowercase()
    }
}

impl fmt::Display for EssenceHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.algorithm, self.hex_digest)
    }
}

/// Result of verifying a single essence file hash.
#[derive(Clone, Debug)]
pub struct VerificationResult {
    /// The asset/file identifier.
    pub asset_id: String,
    /// Expected hash from the PKL.
    pub expected: EssenceHash,
    /// Computed hash (if available).
    pub computed: Option<EssenceHash>,
    /// Verification status.
    pub status: VerificationStatus,
    /// Duration of hash computation.
    pub duration_ms: u64,
}

/// Status of a hash verification.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VerificationStatus {
    /// Hash matches.
    Match,
    /// Hash does not match.
    Mismatch,
    /// File not found or unreadable.
    FileError,
    /// Verification not yet performed.
    Pending,
    /// Hash algorithm not supported.
    UnsupportedAlgorithm,
}

impl fmt::Display for VerificationStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Match => write!(f, "MATCH"),
            Self::Mismatch => write!(f, "MISMATCH"),
            Self::FileError => write!(f, "FILE_ERROR"),
            Self::Pending => write!(f, "PENDING"),
            Self::UnsupportedAlgorithm => write!(f, "UNSUPPORTED_ALGO"),
        }
    }
}

/// Cache for computed hashes to avoid redundant computation.
#[derive(Clone, Debug)]
pub struct HashCache {
    /// Map from asset_id to cached hash entries.
    entries: HashMap<String, HashCacheEntry>,
}

/// A cached hash entry with timestamp.
#[derive(Clone, Debug)]
pub struct HashCacheEntry {
    /// The computed hash.
    pub hash: EssenceHash,
    /// When the hash was computed.
    pub computed_at: SystemTime,
    /// File size in bytes at computation time.
    pub file_size: u64,
}

impl HashCache {
    /// Creates a new empty hash cache.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Inserts a hash entry into the cache.
    pub fn insert(&mut self, asset_id: &str, hash: EssenceHash, file_size: u64) {
        self.entries.insert(
            asset_id.to_string(),
            HashCacheEntry {
                hash,
                computed_at: SystemTime::now(),
                file_size,
            },
        );
    }

    /// Looks up a cached hash by asset ID.
    pub fn get(&self, asset_id: &str) -> Option<&HashCacheEntry> {
        self.entries.get(asset_id)
    }

    /// Removes a cached entry.
    pub fn remove(&mut self, asset_id: &str) -> bool {
        self.entries.remove(asset_id).is_some()
    }

    /// Returns the number of cached entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clears all cached entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Checks if a cached hash is still valid for the given file size.
    pub fn is_valid(&self, asset_id: &str, current_file_size: u64) -> bool {
        self.entries
            .get(asset_id)
            .map_or(false, |e| e.file_size == current_file_size)
    }
}

impl Default for HashCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary report of a batch hash verification.
#[derive(Clone, Debug)]
pub struct VerificationReport {
    /// Individual results per asset.
    pub results: Vec<VerificationResult>,
    /// Total verification time in milliseconds.
    pub total_duration_ms: u64,
}

impl VerificationReport {
    /// Creates a new empty report.
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
            total_duration_ms: 0,
        }
    }

    /// Adds a verification result.
    pub fn add_result(&mut self, result: VerificationResult) {
        self.results.push(result);
    }

    /// Returns the number of matched assets.
    pub fn matched_count(&self) -> usize {
        self.results
            .iter()
            .filter(|r| r.status == VerificationStatus::Match)
            .count()
    }

    /// Returns the number of mismatched assets.
    pub fn mismatched_count(&self) -> usize {
        self.results
            .iter()
            .filter(|r| r.status == VerificationStatus::Mismatch)
            .count()
    }

    /// Returns the number of errored assets.
    pub fn error_count(&self) -> usize {
        self.results
            .iter()
            .filter(|r| r.status == VerificationStatus::FileError)
            .count()
    }

    /// Returns true if all verifications passed.
    pub fn all_passed(&self) -> bool {
        self.results
            .iter()
            .all(|r| r.status == VerificationStatus::Match)
    }

    /// Returns the total number of results.
    pub fn total(&self) -> usize {
        self.results.len()
    }
}

impl Default for VerificationReport {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple in-memory incremental hash computation state.
///
/// This supports chunked hashing for large files.
#[derive(Clone, Debug)]
pub struct IncrementalHasher {
    /// Algorithm being used.
    pub algorithm: HashAlgo,
    /// Accumulated data chunks (simplified; real impl would use streaming).
    chunks: Vec<Vec<u8>>,
    /// Total bytes processed.
    pub bytes_processed: u64,
}

impl IncrementalHasher {
    /// Creates a new incremental hasher for the given algorithm.
    pub fn new(algo: HashAlgo) -> Self {
        Self {
            algorithm: algo,
            chunks: Vec::new(),
            bytes_processed: 0,
        }
    }

    /// Feeds a chunk of data into the hasher.
    pub fn update(&mut self, data: &[u8]) {
        self.bytes_processed += data.len() as u64;
        self.chunks.push(data.to_vec());
    }

    /// Returns the total bytes processed so far.
    pub fn total_bytes(&self) -> u64 {
        self.bytes_processed
    }

    /// Finalizes and returns the cryptographic hash using the real algorithm.
    ///
    /// All buffered chunks are concatenated and hashed via the algorithm
    /// specified at construction time (SHA-1, SHA-256, SHA-512, or MD5).
    pub fn finalize(&self) -> EssenceHash {
        let mut ctx = HashContext::new(self.algorithm);
        for chunk in &self.chunks {
            ctx.update(chunk);
        }
        let hex = ctx.finalize_hex();
        EssenceHash::new(self.algorithm, &hex)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_algo_display() {
        assert_eq!(format!("{}", HashAlgo::Sha1), "SHA-1");
        assert_eq!(format!("{}", HashAlgo::Sha256), "SHA-256");
        assert_eq!(format!("{}", HashAlgo::Sha512), "SHA-512");
        assert_eq!(format!("{}", HashAlgo::Md5), "MD5");
    }

    #[test]
    fn test_hash_algo_digest_length() {
        assert_eq!(HashAlgo::Sha1.digest_length(), 20);
        assert_eq!(HashAlgo::Sha256.digest_length(), 32);
        assert_eq!(HashAlgo::Sha512.digest_length(), 64);
        assert_eq!(HashAlgo::Md5.digest_length(), 16);
    }

    #[test]
    fn test_hash_algo_hex_length() {
        assert_eq!(HashAlgo::Sha1.hex_length(), 40);
        assert_eq!(HashAlgo::Sha256.hex_length(), 64);
        assert_eq!(HashAlgo::Sha512.hex_length(), 128);
        assert_eq!(HashAlgo::Md5.hex_length(), 32);
    }

    #[test]
    fn test_compute_hash_sha256_length() {
        let digest = compute_hash(b"hello world", HashAlgo::Sha256);
        assert_eq!(digest.len(), 32);
    }

    #[test]
    fn test_compute_hash_sha512_length() {
        let digest = compute_hash(b"hello world", HashAlgo::Sha512);
        assert_eq!(digest.len(), 64);
    }

    #[test]
    fn test_compute_hash_sha1_length() {
        let digest = compute_hash(b"hello", HashAlgo::Sha1);
        assert_eq!(digest.len(), 20);
    }

    #[test]
    fn test_compute_hash_md5_length() {
        let digest = compute_hash(b"hello", HashAlgo::Md5);
        assert_eq!(digest.len(), 16);
    }

    #[test]
    fn test_compute_hash_sha256_known_value() {
        // SHA-256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        let digest = compute_hash_hex(b"", HashAlgo::Sha256);
        assert_eq!(
            digest,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn test_compute_hash_sha512_known_value() {
        // SHA-512("") = cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e
        let digest = compute_hash_hex(b"", HashAlgo::Sha512);
        assert!(digest.starts_with("cf83e135"));
        assert_eq!(digest.len(), 128);
    }

    #[test]
    fn test_compute_hash_deterministic() {
        let d1 = compute_hash(b"oximedia", HashAlgo::Sha256);
        let d2 = compute_hash(b"oximedia", HashAlgo::Sha256);
        assert_eq!(d1, d2);
    }

    #[test]
    fn test_hash_context_sha256_matches_one_shot() {
        let data = b"streaming hash test";
        let one_shot = compute_hash(data, HashAlgo::Sha256);

        let mut ctx = HashContext::new(HashAlgo::Sha256);
        ctx.update(&data[..8]);
        ctx.update(&data[8..]);
        let incremental = ctx.finalize();

        assert_eq!(one_shot, incremental);
    }

    #[test]
    fn test_hash_context_sha512_matches_one_shot() {
        let data = b"large file chunk";
        let one_shot = compute_hash(data, HashAlgo::Sha512);

        let mut ctx = HashContext::new(HashAlgo::Sha512);
        ctx.update(data);
        let incremental = ctx.finalize();

        assert_eq!(one_shot, incremental);
    }

    #[test]
    fn test_hash_context_sha1_matches_one_shot() {
        let data = b"sha1 streaming";
        let one_shot = compute_hash(data, HashAlgo::Sha1);

        let mut ctx = HashContext::new(HashAlgo::Sha1);
        ctx.update(data);
        let incremental = ctx.finalize();

        assert_eq!(one_shot, incremental);
    }

    #[test]
    fn test_hash_context_md5_matches_one_shot() {
        let data = b"md5 streaming";
        let one_shot = compute_hash(data, HashAlgo::Md5);

        let mut ctx = HashContext::new(HashAlgo::Md5);
        ctx.update(data);
        let incremental = ctx.finalize();

        assert_eq!(one_shot, incremental);
    }

    #[test]
    fn test_hash_context_finalize_hex_length_sha256() {
        let mut ctx = HashContext::new(HashAlgo::Sha256);
        ctx.update(b"test");
        let hex = ctx.finalize_hex();
        assert_eq!(hex.len(), 64);
        assert!(hex.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_hash_context_finalize_hex_length_sha512() {
        let mut ctx = HashContext::new(HashAlgo::Sha512);
        ctx.update(b"test");
        let hex = ctx.finalize_hex();
        assert_eq!(hex.len(), 128);
    }

    #[test]
    fn test_hash_context_empty_input() {
        // Empty SHA-256 must match known value
        let ctx = HashContext::new(HashAlgo::Sha256);
        let hex = ctx.finalize_hex();
        assert_eq!(
            hex,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn test_essence_hash_valid_format() {
        let h = EssenceHash::new(HashAlgo::Md5, "abcdef0123456789abcdef0123456789");
        assert!(h.is_valid_format());
    }

    #[test]
    fn test_essence_hash_invalid_format() {
        let h = EssenceHash::new(HashAlgo::Md5, "tooshort");
        assert!(!h.is_valid_format());
    }

    #[test]
    fn test_essence_hash_matches() {
        let h1 = EssenceHash::new(HashAlgo::Sha1, "aabbccdd00112233445566778899aabbccddeeff");
        let h2 = EssenceHash::new(HashAlgo::Sha1, "AABBCCDD00112233445566778899AABBCCDDEEFF");
        assert!(h1.matches(&h2));
    }

    #[test]
    fn test_essence_hash_no_match_diff_algo() {
        let h1 = EssenceHash::new(HashAlgo::Sha1, "aabbccdd00112233445566778899aabbccddeeff");
        let h2 = EssenceHash::new(HashAlgo::Md5, "aabbccdd00112233445566778899aabbccddeeff");
        assert!(!h1.matches(&h2));
    }

    #[test]
    fn test_essence_hash_display() {
        let h = EssenceHash::new(HashAlgo::Sha256, "ab".repeat(32).as_str());
        let s = format!("{h}");
        assert!(s.starts_with("SHA-256:"));
    }

    #[test]
    fn test_hash_cache_operations() {
        let mut cache = HashCache::new();
        assert!(cache.is_empty());

        let h = EssenceHash::new(HashAlgo::Sha1, "a".repeat(40).as_str());
        cache.insert("asset-001", h, 1024);
        assert_eq!(cache.len(), 1);
        assert!(cache.get("asset-001").is_some());
        assert!(cache.get("nonexistent").is_none());
    }

    #[test]
    fn test_hash_cache_validity() {
        let mut cache = HashCache::new();
        let h = EssenceHash::new(HashAlgo::Md5, "a".repeat(32).as_str());
        cache.insert("asset-001", h, 1024);
        assert!(cache.is_valid("asset-001", 1024));
        assert!(!cache.is_valid("asset-001", 2048));
    }

    #[test]
    fn test_hash_cache_remove() {
        let mut cache = HashCache::new();
        let h = EssenceHash::new(HashAlgo::Md5, "a".repeat(32).as_str());
        cache.insert("x", h, 100);
        assert!(cache.remove("x"));
        assert!(!cache.remove("x"));
        assert!(cache.is_empty());
    }

    #[test]
    fn test_verification_report_counts() {
        let mut report = VerificationReport::new();
        report.add_result(VerificationResult {
            asset_id: "a".to_string(),
            expected: EssenceHash::new(HashAlgo::Sha1, "a".repeat(40).as_str()),
            computed: None,
            status: VerificationStatus::Match,
            duration_ms: 10,
        });
        report.add_result(VerificationResult {
            asset_id: "b".to_string(),
            expected: EssenceHash::new(HashAlgo::Sha1, "b".repeat(40).as_str()),
            computed: None,
            status: VerificationStatus::Mismatch,
            duration_ms: 20,
        });
        assert_eq!(report.matched_count(), 1);
        assert_eq!(report.mismatched_count(), 1);
        assert_eq!(report.error_count(), 0);
        assert!(!report.all_passed());
        assert_eq!(report.total(), 2);
    }

    #[test]
    fn test_verification_report_all_passed() {
        let mut report = VerificationReport::new();
        report.add_result(VerificationResult {
            asset_id: "a".to_string(),
            expected: EssenceHash::new(HashAlgo::Sha1, "a".repeat(40).as_str()),
            computed: None,
            status: VerificationStatus::Match,
            duration_ms: 5,
        });
        assert!(report.all_passed());
    }

    #[test]
    fn test_incremental_hasher() {
        let mut hasher = IncrementalHasher::new(HashAlgo::Sha1);
        hasher.update(b"hello");
        hasher.update(b"world");
        assert_eq!(hasher.total_bytes(), 10);
        let hash = hasher.finalize();
        assert_eq!(hash.algorithm, HashAlgo::Sha1);
        assert!(hash.is_valid_format());
    }

    #[test]
    fn test_verification_status_display() {
        assert_eq!(format!("{}", VerificationStatus::Match), "MATCH");
        assert_eq!(format!("{}", VerificationStatus::Mismatch), "MISMATCH");
        assert_eq!(format!("{}", VerificationStatus::FileError), "FILE_ERROR");
        assert_eq!(format!("{}", VerificationStatus::Pending), "PENDING");
    }
}
