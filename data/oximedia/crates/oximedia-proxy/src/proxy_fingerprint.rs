#![allow(dead_code)]
//! Proxy fingerprinting for integrity verification.
//!
//! Generates and verifies content-based fingerprints for proxy files to ensure
//! they have not been corrupted or tampered with during transfer, storage,
//! or editing workflows.

use crate::ProxyError;
use std::collections::HashMap;
use std::fmt;
use std::path::Path;

/// Files at or above this size (in bytes) are hashed via a read-only memory
/// map instead of a full heap read, so fingerprinting never copies the whole
/// file into a `Vec`. Smaller files use `std::fs::read` to avoid mmap overhead.
pub const MMAP_THRESHOLD: u64 = 64 * 1024; // 64 KiB

/// Hash algorithm used for fingerprinting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FingerprintAlgorithm {
    /// CRC-32 (fast but weak).
    Crc32,
    /// Adler-32 (fast checksum).
    Adler32,
    /// Simple XOR-based hash (very fast, low quality).
    XorHash,
    /// Block-level content hash.
    BlockHash,
}

impl FingerprintAlgorithm {
    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Crc32 => "CRC-32",
            Self::Adler32 => "Adler-32",
            Self::XorHash => "XOR Hash",
            Self::BlockHash => "Block Hash",
        }
    }
}

/// A content fingerprint for a proxy file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fingerprint {
    /// The algorithm used to generate this fingerprint.
    pub algorithm: FingerprintAlgorithm,
    /// The fingerprint value as a hex string.
    pub hash: String,
    /// File size in bytes at the time of fingerprinting.
    pub file_size: u64,
    /// Number of blocks processed.
    pub blocks_processed: u64,
}

impl Fingerprint {
    /// Create a new fingerprint.
    pub fn new(algorithm: FingerprintAlgorithm, hash: &str, file_size: u64) -> Self {
        Self {
            algorithm,
            hash: hash.to_string(),
            file_size,
            blocks_processed: 0,
        }
    }

    /// Set the blocks processed count.
    pub fn with_blocks(mut self, blocks: u64) -> Self {
        self.blocks_processed = blocks;
        self
    }
}

impl fmt::Display for Fingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.algorithm.name(), self.hash)
    }
}

/// Result of a fingerprint verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyResult {
    /// Fingerprints match.
    Match,
    /// Fingerprints do not match.
    Mismatch {
        /// Expected hash.
        expected: String,
        /// Actual hash.
        actual: String,
    },
    /// File size changed.
    SizeChanged {
        /// Expected size.
        expected: u64,
        /// Actual size.
        actual: u64,
    },
}

impl VerifyResult {
    /// Whether verification passed.
    pub fn is_ok(&self) -> bool {
        matches!(self, Self::Match)
    }
}

/// Simple CRC-32 computation (non-cryptographic, for proxy integrity only).
fn compute_crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

/// Simple Adler-32 computation.
fn compute_adler32(data: &[u8]) -> u32 {
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    for &byte in data {
        a = (a + u32::from(byte)) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

/// Simple XOR hash.
fn compute_xor_hash(data: &[u8]) -> u32 {
    let mut hash: u32 = 0;
    for chunk in data.chunks(4) {
        let mut val: u32 = 0;
        for (i, &byte) in chunk.iter().enumerate() {
            val |= u32::from(byte) << (i * 8);
        }
        hash ^= val;
    }
    hash
}

/// Block-level hash: hash each block and combine.
#[allow(clippy::cast_precision_loss)]
fn compute_block_hash(data: &[u8], block_size: usize) -> (u32, u64) {
    let mut combined: u32 = 0;
    let mut blocks: u64 = 0;
    for chunk in data.chunks(block_size.max(1)) {
        let block_crc = compute_crc32(chunk);
        combined = combined.wrapping_add(block_crc);
        blocks += 1;
    }
    (combined, blocks)
}

/// Engine for computing and verifying proxy fingerprints.
pub struct FingerprintEngine {
    /// Default algorithm.
    algorithm: FingerprintAlgorithm,
    /// Block size for block-based hashing.
    block_size: usize,
    /// Cache of computed fingerprints.
    cache: HashMap<String, Fingerprint>,
}

impl FingerprintEngine {
    /// Create a new fingerprint engine.
    pub fn new(algorithm: FingerprintAlgorithm) -> Self {
        Self {
            algorithm,
            block_size: 4096,
            cache: HashMap::new(),
        }
    }

    /// Set the block size for block-based hashing.
    pub fn with_block_size(mut self, size: usize) -> Self {
        self.block_size = size;
        self
    }

    /// Compute a fingerprint for the given data.
    #[allow(clippy::cast_precision_loss)]
    pub fn compute(&self, data: &[u8]) -> Fingerprint {
        let file_size = data.len() as u64;
        match self.algorithm {
            FingerprintAlgorithm::Crc32 => {
                let crc = compute_crc32(data);
                Fingerprint::new(self.algorithm, &format!("{crc:08x}"), file_size)
            }
            FingerprintAlgorithm::Adler32 => {
                let adler = compute_adler32(data);
                Fingerprint::new(self.algorithm, &format!("{adler:08x}"), file_size)
            }
            FingerprintAlgorithm::XorHash => {
                let xor = compute_xor_hash(data);
                Fingerprint::new(self.algorithm, &format!("{xor:08x}"), file_size)
            }
            FingerprintAlgorithm::BlockHash => {
                let (hash, blocks) = compute_block_hash(data, self.block_size);
                Fingerprint::new(self.algorithm, &format!("{hash:08x}"), file_size)
                    .with_blocks(blocks)
            }
        }
    }

    /// Compute a fingerprint for a file on disk without loading the whole
    /// file into a heap `Vec`.
    ///
    /// Files at or above [`MMAP_THRESHOLD`] bytes are mapped read-only into the
    /// process address space and the resulting byte slice is fed directly to
    /// [`compute`](Self::compute) — no intermediate allocation.  Smaller files
    /// (and any file for which mapping fails) fall back to a single
    /// `std::fs::read`.  Both paths route through the *same* in-memory
    /// [`compute`](Self::compute), so the produced [`Fingerprint`] is
    /// byte-identical regardless of which path is taken.
    ///
    /// # Errors
    ///
    /// Returns [`ProxyError::IoError`] if the file cannot be opened, its
    /// metadata cannot be read, the memory map cannot be created, or (on the
    /// small-file path) the file cannot be read.
    pub fn hash_file(&self, path: &Path) -> std::result::Result<Fingerprint, ProxyError> {
        let file = std::fs::File::open(path)?;
        let file_size = file.metadata()?.len();

        // Empty files have nothing to map; hash the empty slice directly.
        if file_size == 0 {
            return Ok(self.compute(&[]));
        }

        if file_size >= MMAP_THRESHOLD {
            // Zero-copy path: hash directly from the mmap region.
            return self.hash_via_mmap(&file);
        }

        // Small-file fallback — single read, no mmap overhead.
        let bytes = std::fs::read(path)?;
        Ok(self.compute(&bytes))
    }

    /// Map `file` read-only and compute its fingerprint directly from the
    /// mapped bytes.
    ///
    /// The `Mmap` is kept alive until [`compute`](Self::compute) has consumed
    /// every byte, then dropped before returning, so the borrow of the mapped
    /// region never outlives the mapping.
    #[allow(unsafe_code)]
    fn hash_via_mmap(&self, file: &std::fs::File) -> std::result::Result<Fingerprint, ProxyError> {
        // SAFETY: the file is mapped read-only and the resulting `Mmap` is
        // dropped at the end of this function, after `compute` has finished
        // reading every byte. We never expose the mapping or its borrow to the
        // caller, so the exposure window is bounded to this call.
        let mmap = unsafe { memmap2::Mmap::map(file) }.map_err(|e| {
            ProxyError::IoError(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("mmap fingerprint failed: {e}"),
            ))
        })?;
        let fingerprint = self.compute(&mmap[..]);
        // `mmap` dropped here — the slice borrow above has already ended.
        Ok(fingerprint)
    }

    /// Compute and cache a fingerprint for a named proxy.
    pub fn compute_and_cache(&mut self, name: &str, data: &[u8]) -> Fingerprint {
        let fp = self.compute(data);
        self.cache.insert(name.to_string(), fp.clone());
        fp
    }

    /// Verify data against a stored fingerprint.
    pub fn verify(&self, data: &[u8], expected: &Fingerprint) -> VerifyResult {
        #[allow(clippy::cast_precision_loss)]
        let actual_size = data.len() as u64;
        if actual_size != expected.file_size {
            return VerifyResult::SizeChanged {
                expected: expected.file_size,
                actual: actual_size,
            };
        }
        let actual_fp = self.compute(data);
        if actual_fp.hash == expected.hash {
            VerifyResult::Match
        } else {
            VerifyResult::Mismatch {
                expected: expected.hash.clone(),
                actual: actual_fp.hash,
            }
        }
    }

    /// Look up a cached fingerprint by name.
    pub fn get_cached(&self, name: &str) -> Option<&Fingerprint> {
        self.cache.get(name)
    }

    /// Number of cached fingerprints.
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Clear the fingerprint cache.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_DATA: &[u8] = b"Hello, proxy fingerprint test data for OxiMedia framework!";

    #[test]
    fn test_algorithm_name() {
        assert_eq!(FingerprintAlgorithm::Crc32.name(), "CRC-32");
        assert_eq!(FingerprintAlgorithm::Adler32.name(), "Adler-32");
        assert_eq!(FingerprintAlgorithm::XorHash.name(), "XOR Hash");
        assert_eq!(FingerprintAlgorithm::BlockHash.name(), "Block Hash");
    }

    #[test]
    fn test_crc32_deterministic() {
        let a = compute_crc32(TEST_DATA);
        let b = compute_crc32(TEST_DATA);
        assert_eq!(a, b);
    }

    #[test]
    fn test_adler32_deterministic() {
        let a = compute_adler32(TEST_DATA);
        let b = compute_adler32(TEST_DATA);
        assert_eq!(a, b);
    }

    #[test]
    fn test_xor_hash_deterministic() {
        let a = compute_xor_hash(TEST_DATA);
        let b = compute_xor_hash(TEST_DATA);
        assert_eq!(a, b);
    }

    #[test]
    fn test_crc32_different_data() {
        let a = compute_crc32(b"hello");
        let b = compute_crc32(b"world");
        assert_ne!(a, b);
    }

    #[test]
    fn test_compute_crc32_fingerprint() {
        let engine = FingerprintEngine::new(FingerprintAlgorithm::Crc32);
        let fp = engine.compute(TEST_DATA);
        assert_eq!(fp.algorithm, FingerprintAlgorithm::Crc32);
        assert_eq!(fp.file_size, TEST_DATA.len() as u64);
        assert!(!fp.hash.is_empty());
    }

    #[test]
    fn test_compute_adler32_fingerprint() {
        let engine = FingerprintEngine::new(FingerprintAlgorithm::Adler32);
        let fp = engine.compute(TEST_DATA);
        assert_eq!(fp.algorithm, FingerprintAlgorithm::Adler32);
    }

    #[test]
    fn test_compute_block_hash_fingerprint() {
        let engine = FingerprintEngine::new(FingerprintAlgorithm::BlockHash).with_block_size(16);
        let fp = engine.compute(TEST_DATA);
        assert_eq!(fp.algorithm, FingerprintAlgorithm::BlockHash);
        assert!(fp.blocks_processed > 0);
    }

    #[test]
    fn test_verify_match() {
        let engine = FingerprintEngine::new(FingerprintAlgorithm::Crc32);
        let fp = engine.compute(TEST_DATA);
        let result = engine.verify(TEST_DATA, &fp);
        assert!(result.is_ok());
        assert_eq!(result, VerifyResult::Match);
    }

    #[test]
    fn test_verify_mismatch() {
        let engine = FingerprintEngine::new(FingerprintAlgorithm::Crc32);
        let fp = engine.compute(TEST_DATA);
        let _tampered = b"Tampered data that is different from the original proxy data!";
        // Make tampered same length as TEST_DATA for size match
        let mut tampered_same_size = TEST_DATA.to_vec();
        tampered_same_size[0] = b'X';
        let result = engine.verify(&tampered_same_size, &fp);
        assert!(!result.is_ok());
        assert!(matches!(result, VerifyResult::Mismatch { .. }));
    }

    #[test]
    fn test_verify_size_changed() {
        let engine = FingerprintEngine::new(FingerprintAlgorithm::Crc32);
        let fp = engine.compute(TEST_DATA);
        let shorter = &TEST_DATA[..10];
        let result = engine.verify(shorter, &fp);
        assert!(matches!(result, VerifyResult::SizeChanged { .. }));
    }

    #[test]
    fn test_cache_operations() {
        let mut engine = FingerprintEngine::new(FingerprintAlgorithm::Crc32);
        assert_eq!(engine.cache_size(), 0);
        engine.compute_and_cache("proxy_a.mp4", TEST_DATA);
        assert_eq!(engine.cache_size(), 1);
        assert!(engine.get_cached("proxy_a.mp4").is_some());
        assert!(engine.get_cached("nonexistent").is_none());
        engine.clear_cache();
        assert_eq!(engine.cache_size(), 0);
    }

    #[test]
    fn test_fingerprint_display() {
        let fp = Fingerprint::new(FingerprintAlgorithm::Crc32, "abcd1234", 100);
        let display = format!("{fp}");
        assert_eq!(display, "CRC-32:abcd1234");
    }

    #[test]
    fn test_empty_data() {
        let engine = FingerprintEngine::new(FingerprintAlgorithm::Crc32);
        let fp = engine.compute(b"");
        assert_eq!(fp.file_size, 0);
        // CRC32 of empty data should be deterministic
        let fp2 = engine.compute(b"");
        assert_eq!(fp.hash, fp2.hash);
    }

    // --- hash_file: mmap vs std::fs::read path correctness ---

    use std::io::Write;

    /// Write `content` to a uniquely-named file in the system temp dir and
    /// return its path. The caller is responsible for removal.
    fn write_unique_temp(tag: &str, content: &[u8]) -> std::path::PathBuf {
        // Combine a per-process nonce with the tag for uniqueness across the
        // parallel test runner.
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let path = std::env::temp_dir().join(format!(
            "oximedia_proxy_fp_{tag}_{}_{nonce}.bin",
            std::process::id()
        ));
        let mut f = std::fs::File::create(&path).expect("create temp file");
        f.write_all(content).expect("write temp file");
        f.flush().expect("flush temp file");
        path
    }

    /// A file at/above MMAP_THRESHOLD must hash identically through `hash_file`
    /// (mmap path) and the in-memory `compute()` on the same bytes.
    #[test]
    fn test_hash_file_large_matches_compute() {
        // 128 KiB — well above the 64 KiB mmap threshold.
        let content: Vec<u8> = (0u8..=255).cycle().take(128 * 1024).collect();
        assert!(content.len() as u64 >= MMAP_THRESHOLD);
        let path = write_unique_temp("large", &content);

        let engine = FingerprintEngine::new(FingerprintAlgorithm::Crc32);
        let from_file = engine.hash_file(&path).expect("hash_file large");
        let in_memory = engine.compute(&content);

        assert_eq!(
            from_file, in_memory,
            "mmap path must equal in-memory compute"
        );
        assert_eq!(from_file.file_size, content.len() as u64);

        std::fs::remove_file(&path).ok();
    }

    /// The large-file path must agree across every algorithm variant.
    #[test]
    fn test_hash_file_large_matches_compute_all_algorithms() {
        let content: Vec<u8> = (0u8..=255).cycle().take(96 * 1024).collect();
        assert!(content.len() as u64 >= MMAP_THRESHOLD);
        let path = write_unique_temp("large_all", &content);

        for algo in [
            FingerprintAlgorithm::Crc32,
            FingerprintAlgorithm::Adler32,
            FingerprintAlgorithm::XorHash,
            FingerprintAlgorithm::BlockHash,
        ] {
            let engine = FingerprintEngine::new(algo).with_block_size(4096);
            let from_file = engine.hash_file(&path).expect("hash_file large all");
            let in_memory = engine.compute(&content);
            assert_eq!(from_file, in_memory, "mismatch for algorithm {algo:?}");
        }

        std::fs::remove_file(&path).ok();
    }

    /// A file below MMAP_THRESHOLD must hash identically through `hash_file`
    /// (std::fs::read path) and the in-memory `compute()`.
    #[test]
    fn test_hash_file_small_matches_compute() {
        let content = b"small proxy fingerprint payload under the mmap threshold";
        assert!((content.len() as u64) < MMAP_THRESHOLD);
        let path = write_unique_temp("small", content);

        let engine = FingerprintEngine::new(FingerprintAlgorithm::Adler32);
        let from_file = engine.hash_file(&path).expect("hash_file small");
        let in_memory = engine.compute(content);

        assert_eq!(
            from_file, in_memory,
            "std path must equal in-memory compute"
        );
        assert_eq!(from_file.file_size, content.len() as u64);

        std::fs::remove_file(&path).ok();
    }

    /// A file exactly at the threshold uses the mmap path and must still match.
    #[test]
    fn test_hash_file_threshold_boundary_matches_compute() {
        let content: Vec<u8> = (0u8..=255).cycle().take(MMAP_THRESHOLD as usize).collect();
        let path = write_unique_temp("boundary", &content);

        let engine = FingerprintEngine::new(FingerprintAlgorithm::Crc32);
        let from_file = engine.hash_file(&path).expect("hash_file boundary");
        let in_memory = engine.compute(&content);

        assert_eq!(from_file, in_memory, "threshold-boundary file must match");

        std::fs::remove_file(&path).ok();
    }

    /// A missing path returns an error without panicking.
    #[test]
    fn test_hash_file_missing_path_errors() {
        let engine = FingerprintEngine::new(FingerprintAlgorithm::Crc32);
        let missing = std::env::temp_dir().join(format!(
            "oximedia_proxy_fp_definitely_missing_{}.bin",
            std::process::id()
        ));
        // Ensure it really does not exist.
        std::fs::remove_file(&missing).ok();
        let result = engine.hash_file(&missing);
        assert!(result.is_err(), "missing path should yield Err");
    }

    /// An empty (0-byte) file is handled gracefully and yields the same
    /// fingerprint as `compute(&[])`.
    #[test]
    fn test_hash_file_empty_matches_compute_empty() {
        let path = write_unique_temp("empty", b"");

        let engine = FingerprintEngine::new(FingerprintAlgorithm::Crc32);
        let from_file = engine.hash_file(&path).expect("hash_file empty");
        let in_memory = engine.compute(&[]);

        assert_eq!(from_file, in_memory, "empty file must match compute(&[])");
        assert_eq!(from_file.file_size, 0);

        std::fs::remove_file(&path).ok();
    }
}

// ============================================================================
// Perceptual Hashing — dHash and pHash (pure Rust, no ndarray)
// ============================================================================

/// Luma (Y) value for an RGB pixel using BT.601 coefficients (integer math).
///
/// Returns a value in `[0, 255]`.
#[inline]
fn rgb_to_luma(r: u8, g: u8, b: u8) -> u8 {
    // BT.601: Y = 0.299·R + 0.587·G + 0.114·B
    // Scaled × 1024 → Y = (306·R + 601·G + 117·B) >> 10
    let y = (306u32 * r as u32 + 601u32 * g as u32 + 117u32 * b as u32) >> 10;
    y.min(255) as u8
}

/// Decode a flat byte slice as a grayscale image with `width × height` pixels.
///
/// Supported input layouts:
/// * 1 byte/pixel  → already grayscale
/// * 3 bytes/pixel → RGB, converted via BT.601
/// * 4 bytes/pixel → RGBA, alpha ignored
///
/// # Errors
///
/// Returns `None` when `data.len() != width * height * channels` or an
/// unsupported channel count is given.
fn decode_luma(data: &[u8], width: usize, height: usize, channels: usize) -> Option<Vec<u8>> {
    if data.len() != width * height * channels {
        return None;
    }
    match channels {
        1 => Some(data.to_vec()),
        3 => Some(
            data.chunks_exact(3)
                .map(|px| rgb_to_luma(px[0], px[1], px[2]))
                .collect(),
        ),
        4 => Some(
            data.chunks_exact(4)
                .map(|px| rgb_to_luma(px[0], px[1], px[2]))
                .collect(),
        ),
        _ => None,
    }
}

/// Bilinear downsample a grayscale image to `(out_w, out_h)`.
///
/// Uses integer arithmetic; all pixels are treated as uniformly spaced.
fn bilinear_resize(src: &[u8], src_w: usize, src_h: usize, out_w: usize, out_h: usize) -> Vec<u8> {
    let mut out = vec![0u8; out_w * out_h];
    for oy in 0..out_h {
        for ox in 0..out_w {
            // Map output pixel centre back to source space
            let sx_f = (ox as f64 + 0.5) * src_w as f64 / out_w as f64 - 0.5;
            let sy_f = (oy as f64 + 0.5) * src_h as f64 / out_h as f64 - 0.5;
            let x0 = (sx_f.floor() as isize).clamp(0, src_w as isize - 1) as usize;
            let y0 = (sy_f.floor() as isize).clamp(0, src_h as isize - 1) as usize;
            let x1 = (x0 + 1).min(src_w - 1);
            let y1 = (y0 + 1).min(src_h - 1);
            let wx = (sx_f - x0 as f64).clamp(0.0, 1.0);
            let wy = (sy_f - y0 as f64).clamp(0.0, 1.0);
            let p00 = src[y0 * src_w + x0] as f64;
            let p10 = src[y0 * src_w + x1] as f64;
            let p01 = src[y1 * src_w + x0] as f64;
            let p11 = src[y1 * src_w + x1] as f64;
            let v = p00 * (1.0 - wx) * (1.0 - wy)
                + p10 * wx * (1.0 - wy)
                + p01 * (1.0 - wx) * wy
                + p11 * wx * wy;
            out[oy * out_w + ox] = v.round() as u8;
        }
    }
    out
}

/// A 64-bit perceptual hash stored as a `u64` bitmask.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PerceptualHash(pub u64);

impl PerceptualHash {
    /// Hamming distance to another perceptual hash.
    ///
    /// Two images are considered visually similar when the distance is ≤ 10.
    #[must_use]
    pub fn hamming_distance(self, other: Self) -> u32 {
        (self.0 ^ other.0).count_ones()
    }

    /// Whether two hashes are perceptually similar (distance ≤ `threshold`).
    #[must_use]
    pub fn is_similar(self, other: Self, threshold: u32) -> bool {
        self.hamming_distance(other) <= threshold
    }

    /// Hexadecimal representation (16 characters).
    #[must_use]
    pub fn to_hex(self) -> String {
        format!("{:016x}", self.0)
    }

    /// Parse a hex string produced by [`PerceptualHash::to_hex`].
    ///
    /// # Errors
    ///
    /// Returns `None` on invalid hex input.
    pub fn from_hex(s: &str) -> Option<Self> {
        u64::from_str_radix(s, 16).ok().map(Self)
    }
}

impl std::fmt::Display for PerceptualHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

/// Difference Hash (dHash) — fast, robust perceptual hash.
///
/// Algorithm:
/// 1. Resize to 9×8 grayscale.
/// 2. For each row, compare adjacent pixels; bit = (left > right).
/// 3. Pack 64 bits into a `u64`.
///
/// # Arguments
/// * `data`     – Raw pixel bytes.
/// * `width`    – Image width in pixels.
/// * `height`   – Image height in pixels.
/// * `channels` – Bytes per pixel (1 = gray, 3 = RGB, 4 = RGBA).
///
/// # Returns
///
/// `None` when the input dimensions/channel count are inconsistent.
pub fn dhash(data: &[u8], width: usize, height: usize, channels: usize) -> Option<PerceptualHash> {
    let luma = decode_luma(data, width, height, channels)?;
    // Resize to 9 wide × 8 tall
    let small = bilinear_resize(&luma, width, height, 9, 8);
    let mut bits: u64 = 0;
    for row in 0..8 {
        for col in 0..8 {
            let left = small[row * 9 + col];
            let right = small[row * 9 + col + 1];
            bits = (bits << 1) | u64::from(left > right);
        }
    }
    Some(PerceptualHash(bits))
}

/// Perceptual Hash (pHash) — DCT-based, more robust than dHash.
///
/// Algorithm (simplified, no external FFT dependency):
/// 1. Resize to 32×32 grayscale.
/// 2. Compute 2-D DCT (type-II) over the full block.
/// 3. Take the top-left 8×8 DCT coefficients (64 values), excluding DC.
/// 4. Compute their mean.
/// 5. Each bit = (coefficient > mean).
///
/// The DCT is computed in pure Rust using the direct O(N²) formula.
///
/// # Arguments
/// * `data`     – Raw pixel bytes.
/// * `width`    – Image width in pixels.
/// * `height`   – Image height in pixels.
/// * `channels` – Bytes per pixel (1 = gray, 3 = RGB, 4 = RGBA).
///
/// # Returns
///
/// `None` when input dimensions/channel count are inconsistent.
pub fn phash(data: &[u8], width: usize, height: usize, channels: usize) -> Option<PerceptualHash> {
    const RESIZE: usize = 32;
    const DCT_SIZE: usize = 8;

    let luma = decode_luma(data, width, height, channels)?;
    let small = bilinear_resize(&luma, width, height, RESIZE, RESIZE);

    // 2-D DCT-II: dct[u][v] = Σ_x Σ_y pixel[x][y] · cos(π(2x+1)u/64) · cos(π(2y+1)v/64)
    // We only need the top-left DCT_SIZE×DCT_SIZE block.
    let mut dct_block = [0.0f64; DCT_SIZE * DCT_SIZE];
    for u in 0..DCT_SIZE {
        for v in 0..DCT_SIZE {
            let mut sum = 0.0f64;
            for x in 0..RESIZE {
                for y in 0..RESIZE {
                    let px = small[x * RESIZE + y] as f64;
                    let cu =
                        std::f64::consts::PI * (2 * x + 1) as f64 * u as f64 / (2 * RESIZE) as f64;
                    let cv =
                        std::f64::consts::PI * (2 * y + 1) as f64 * v as f64 / (2 * RESIZE) as f64;
                    sum += px * cu.cos() * cv.cos();
                }
            }
            dct_block[u * DCT_SIZE + v] = sum;
        }
    }

    // Exclude DC component (index 0,0), use remaining 63 values + slot 0 for mean
    // Standard pHash: use all 64, skip (0,0) for mean calculation
    let values_for_mean: Vec<f64> = dct_block.iter().skip(1).copied().collect();
    let mean = values_for_mean.iter().sum::<f64>() / values_for_mean.len() as f64;

    let mut bits: u64 = 0;
    for (i, &coeff) in dct_block.iter().enumerate() {
        if i == 0 {
            // Skip DC
            bits <<= 1;
            continue;
        }
        bits = (bits << 1) | u64::from(coeff > mean);
    }
    Some(PerceptualHash(bits))
}

/// Match a proxy to its source using perceptual hash similarity.
///
/// Returns `true` when the Hamming distance between proxy and source hashes
/// is within `threshold` bits (default: 10 bits for typical noise tolerance).
#[must_use]
pub fn proxy_matches_source(
    proxy_hash: PerceptualHash,
    source_hash: PerceptualHash,
    threshold: u32,
) -> bool {
    proxy_hash.is_similar(source_hash, threshold)
}

#[cfg(test)]
mod perceptual_tests {
    use super::*;

    /// Create a solid-color 8×8 RGB image.
    fn solid_rgb(w: usize, h: usize, r: u8, g: u8, b: u8) -> Vec<u8> {
        let mut v = Vec::with_capacity(w * h * 3);
        for _ in 0..w * h {
            v.push(r);
            v.push(g);
            v.push(b);
        }
        v
    }

    /// Create a grayscale gradient image (left = black, right = white).
    fn gradient_gray(w: usize, h: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(w * h);
        for _y in 0..h {
            for x in 0..w {
                v.push((x * 255 / (w - 1).max(1)) as u8);
            }
        }
        v
    }

    #[test]
    fn test_rgb_to_luma_black() {
        assert_eq!(rgb_to_luma(0, 0, 0), 0);
    }

    #[test]
    fn test_rgb_to_luma_white() {
        let y = rgb_to_luma(255, 255, 255);
        // Should be very close to 255
        assert!(y >= 254);
    }

    #[test]
    fn test_decode_luma_rgb() {
        let data = solid_rgb(4, 4, 128, 64, 32);
        let luma = decode_luma(&data, 4, 4, 3).expect("decode should succeed");
        assert_eq!(luma.len(), 16);
        // All pixels same color → all luma values identical
        assert!(luma.windows(2).all(|w| w[0] == w[1]));
    }

    #[test]
    fn test_decode_luma_gray() {
        let data = vec![100u8; 9];
        let luma = decode_luma(&data, 3, 3, 1).expect("decode gray should succeed");
        assert_eq!(luma.len(), 9);
        assert!(luma.iter().all(|&v| v == 100));
    }

    #[test]
    fn test_decode_luma_rgba() {
        let mut data = Vec::new();
        for _ in 0..4 {
            data.extend_from_slice(&[200, 100, 50, 255]); // RGBA
        }
        let luma = decode_luma(&data, 2, 2, 4).expect("decode rgba should succeed");
        assert_eq!(luma.len(), 4);
    }

    #[test]
    fn test_decode_luma_wrong_size_returns_none() {
        let data = vec![0u8; 10];
        assert!(decode_luma(&data, 4, 4, 3).is_none());
    }

    #[test]
    fn test_decode_luma_unsupported_channels() {
        let data = vec![0u8; 8];
        assert!(decode_luma(&data, 2, 2, 2).is_none());
    }

    #[test]
    fn test_dhash_identical_images_zero_distance() {
        let img = gradient_gray(16, 16);
        let h1 = dhash(&img, 16, 16, 1).expect("dhash should succeed");
        let h2 = dhash(&img, 16, 16, 1).expect("dhash should succeed");
        assert_eq!(h1.hamming_distance(h2), 0);
    }

    #[test]
    fn test_dhash_solid_image() {
        // Solid image: all pixels equal → no gradients → all bits 0
        let img = vec![128u8; 32 * 32];
        let h = dhash(&img, 32, 32, 1).expect("dhash should succeed");
        assert_eq!(h.0, 0u64);
    }

    #[test]
    fn test_dhash_different_images_non_zero_distance() {
        let img_a = gradient_gray(16, 16);
        // Reverse gradient
        let img_b: Vec<u8> = img_a.iter().rev().copied().collect();
        let ha = dhash(&img_a, 16, 16, 1).expect("dhash a");
        let hb = dhash(&img_b, 16, 16, 1).expect("dhash b");
        // Different images should have non-zero distance (very likely)
        assert_ne!(ha.0, hb.0);
    }

    #[test]
    fn test_dhash_rgb_input() {
        let img = solid_rgb(16, 16, 80, 160, 200);
        let h = dhash(&img, 16, 16, 3).expect("dhash rgb");
        // Solid color → all 0
        assert_eq!(h.0, 0u64);
    }

    #[test]
    fn test_dhash_wrong_size_returns_none() {
        let data = vec![0u8; 5];
        assert!(dhash(&data, 4, 4, 3).is_none());
    }

    #[test]
    fn test_phash_identical_images_zero_distance() {
        let img = gradient_gray(32, 32);
        let h1 = phash(&img, 32, 32, 1).expect("phash should succeed");
        let h2 = phash(&img, 32, 32, 1).expect("phash should succeed");
        assert_eq!(h1.hamming_distance(h2), 0);
    }

    #[test]
    fn test_phash_wrong_size_returns_none() {
        let data = vec![0u8; 7];
        assert!(phash(&data, 3, 3, 3).is_none());
    }

    #[test]
    fn test_perceptual_hash_hamming_distance() {
        let a = PerceptualHash(0b1010_1010);
        let b = PerceptualHash(0b0101_0101);
        // 8 differing bits
        assert_eq!(a.hamming_distance(b), 8);
    }

    #[test]
    fn test_perceptual_hash_is_similar_within_threshold() {
        let a = PerceptualHash(0u64);
        let b = PerceptualHash(0b111u64); // 3 bits different
        assert!(a.is_similar(b, 5));
        assert!(!a.is_similar(b, 2));
    }

    #[test]
    fn test_perceptual_hash_hex_roundtrip() {
        let h = PerceptualHash(0xDEAD_BEEF_CAFE_1234);
        let hex = h.to_hex();
        let restored = PerceptualHash::from_hex(&hex).expect("from_hex should succeed");
        assert_eq!(h, restored);
    }

    #[test]
    fn test_perceptual_hash_display() {
        let h = PerceptualHash(0);
        assert_eq!(format!("{h}"), "0000000000000000");
    }

    #[test]
    fn test_perceptual_hash_from_hex_invalid() {
        assert!(PerceptualHash::from_hex("xyz").is_none());
    }

    #[test]
    fn test_proxy_matches_source_similar() {
        let h1 = PerceptualHash(0b1111_0000);
        let h2 = PerceptualHash(0b1110_0000); // 1 bit off
        assert!(proxy_matches_source(h1, h2, 10));
    }

    #[test]
    fn test_proxy_matches_source_dissimilar() {
        let h1 = PerceptualHash(0u64);
        let h2 = PerceptualHash(u64::MAX); // 64 bits different
        assert!(!proxy_matches_source(h1, h2, 10));
    }

    #[test]
    fn test_bilinear_resize_same_size() {
        let src: Vec<u8> = (0..16).collect();
        let out = bilinear_resize(&src, 4, 4, 4, 4);
        assert_eq!(out.len(), 16);
        // Same size: values should be preserved
        assert_eq!(out[0], src[0]);
        assert_eq!(out[15], src[15]);
    }

    #[test]
    fn test_bilinear_resize_downscale() {
        let src = vec![100u8; 64]; // 8×8 solid
        let out = bilinear_resize(&src, 8, 8, 4, 4);
        assert_eq!(out.len(), 16);
        for &v in &out {
            assert_eq!(v, 100);
        }
    }
}
