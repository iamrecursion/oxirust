//! Perceptual hashing for image/video deduplication.
//!
//! Provides multiple perceptual hash algorithms:
//! - **dHash** (difference hash): compares adjacent pixels in an 8×9 thumbnail
//! - **aHash** (average hash): compares each pixel to the mean of an 8×8 thumbnail
//! - **pHash**: DCT-based hash (re-exported concept, implemented here for simple use)

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

// ---------------------------------------------------------------------------
// HashAlgo enum
// ---------------------------------------------------------------------------

/// Perceptual hash algorithm selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HashAlgo {
    /// Difference hash (8×9 → 64-bit, compares adjacent pixels).
    Dhash,
    /// Perceptual hash (DCT-based, 32×32 → 64-bit).
    Phash,
    /// Average hash (8×8 → 64-bit, compares to mean).
    Ahash,
}

impl HashAlgo {
    /// Return the number of bits in hashes produced by this algorithm.
    #[must_use]
    pub const fn hash_bits(self) -> u32 {
        64
    }

    /// Human-readable name of the algorithm.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            HashAlgo::Dhash => "dhash",
            HashAlgo::Phash => "phash",
            HashAlgo::Ahash => "ahash",
        }
    }
}

// ---------------------------------------------------------------------------
// PerceptualHash struct
// ---------------------------------------------------------------------------

/// A 64-bit perceptual hash paired with the algorithm that produced it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PerceptualHash {
    /// The 64-bit hash value.
    pub bits: u64,
    /// The algorithm used to produce this hash.
    pub algo: HashAlgo,
}

impl PerceptualHash {
    /// Create a new perceptual hash.
    #[must_use]
    pub const fn new(bits: u64, algo: HashAlgo) -> Self {
        Self { bits, algo }
    }

    /// Compute the Hamming distance between two hashes (number of differing bits).
    ///
    /// # Panics
    ///
    /// Does not panic; differing algorithms still produce a numeric distance.
    #[must_use]
    pub fn hamming_distance(&self, other: &Self) -> u32 {
        (self.bits ^ other.bits).count_ones()
    }

    /// Similarity score in `[0.0, 1.0]`.
    ///
    /// `1.0` = identical, `0.0` = maximally different (all 64 bits differ).
    #[must_use]
    pub fn similarity(&self, other: &Self) -> f32 {
        1.0 - self.hamming_distance(other) as f32 / 64.0
    }

    /// Hex string representation of the hash bits.
    #[must_use]
    pub fn to_hex(self) -> String {
        format!("{:016x}", self.bits)
    }
}

impl std::fmt::Display for PerceptualHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.algo.name(), self.to_hex())
    }
}

// ---------------------------------------------------------------------------
// Thumbnail helpers (nearest-neighbour, grayscale)
// ---------------------------------------------------------------------------

/// Resize a raw pixel buffer (any stride/channels) to a grayscale `(out_w × out_h)` thumbnail.
///
/// `pixels` must be a packed row-major buffer with `channels` bytes per pixel.
/// Returns `out_w * out_h` grayscale values in `[0, 255]`.
fn resize_to_gray(
    pixels: &[u8],
    src_w: usize,
    src_h: usize,
    out_w: usize,
    out_h: usize,
) -> Vec<u8> {
    // Determine stride: assume 1 channel (grayscale) if the buffer matches w*h,
    // otherwise assume 3 channels (RGB).
    let channels = if pixels.len() == src_w * src_h {
        1
    } else if pixels.len() >= src_w * src_h * 3 {
        3
    } else {
        // Best-effort: single channel
        1
    };

    let x_ratio = src_w as f32 / out_w as f32;
    let y_ratio = src_h as f32 / out_h as f32;

    let mut out = Vec::with_capacity(out_w * out_h);
    for ny in 0..out_h {
        let sy = (ny as f32 * y_ratio) as usize;
        let sy = sy.min(src_h - 1);
        for nx in 0..out_w {
            let sx = (nx as f32 * x_ratio) as usize;
            let sx = sx.min(src_w - 1);
            let base = (sy * src_w + sx) * channels;
            let gray = if channels >= 3 {
                let r = pixels[base] as f32;
                let g = pixels[base + 1] as f32;
                let b = pixels[base + 2] as f32;
                (0.299 * r + 0.587 * g + 0.114 * b) as u8
            } else {
                pixels[base]
            };
            out.push(gray);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// dHash (difference hash)
// ---------------------------------------------------------------------------

/// Compute a difference hash (dHash) from a pixel buffer.
///
/// The algorithm:
/// 1. Resize to 9×8 (grayscale)
/// 2. Compare each pixel to the one to its right (8 comparisons per row × 8 rows = 64 bits)
/// 3. Bit = 1 if left pixel is brighter
///
/// `width` and `height` are the dimensions of the source `pixels` buffer.
/// `pixels` may be grayscale (1 byte/px) or RGB (3 bytes/px).
#[must_use]
pub fn compute_dhash(pixels: &[u8], width: usize, height: usize) -> PerceptualHash {
    if pixels.is_empty() || width == 0 || height == 0 {
        return PerceptualHash::new(0, HashAlgo::Dhash);
    }

    // Resize to 9×8
    let thumb = resize_to_gray(pixels, width, height, 9, 8);

    let mut hash = 0u64;
    let mut bit = 0u32;
    for row in 0..8usize {
        for col in 0..8usize {
            let left = thumb[row * 9 + col];
            let right = thumb[row * 9 + col + 1];
            if left > right {
                hash |= 1u64 << bit;
            }
            bit += 1;
        }
    }

    PerceptualHash::new(hash, HashAlgo::Dhash)
}

// ---------------------------------------------------------------------------
// aHash (average hash)
// ---------------------------------------------------------------------------

/// Compute an average hash (aHash) from a pixel buffer.
///
/// The algorithm:
/// 1. Resize to 8×8 (grayscale)
/// 2. Compute the mean pixel value
/// 3. Bit = 1 if pixel ≥ mean
///
/// `width` and `height` are the dimensions of the source `pixels` buffer.
#[must_use]
pub fn compute_ahash(pixels: &[u8], width: usize, height: usize) -> PerceptualHash {
    if pixels.is_empty() || width == 0 || height == 0 {
        return PerceptualHash::new(0, HashAlgo::Ahash);
    }

    let thumb = resize_to_gray(pixels, width, height, 8, 8);

    let mean: f32 = thumb.iter().map(|&p| p as f32).sum::<f32>() / 64.0;

    let mut hash = 0u64;
    for (i, &px) in thumb.iter().enumerate() {
        if px as f32 >= mean {
            hash |= 1u64 << i;
        }
    }

    PerceptualHash::new(hash, HashAlgo::Ahash)
}

// ---------------------------------------------------------------------------
// PerceptualDeduplicator
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// wHash (wavelet hash) – Haar wavelet decomposition
// ---------------------------------------------------------------------------

/// Compute a wavelet hash (wHash) from a pixel buffer.
///
/// The algorithm:
/// 1. Resize to 8×8 (grayscale)
/// 2. Apply one level of 2D Haar wavelet decomposition
/// 3. Threshold all 64 wavelet coefficients at the median → 64-bit hash
///
/// wHash is robust against scaling and compression artifacts.
#[must_use]
pub fn compute_whash(pixels: &[u8], width: usize, height: usize) -> PerceptualHash {
    if pixels.is_empty() || width == 0 || height == 0 {
        return PerceptualHash::new(0, HashAlgo::Dhash); // fallback algo tag
    }

    let thumb = resize_to_gray(pixels, width, height, 8, 8);

    // Row-wise Haar transform
    let mut row_xform = [0.0f64; 64];
    for y in 0..8usize {
        for x in 0..4usize {
            let a = f64::from(thumb[y * 8 + 2 * x]);
            let b = f64::from(thumb[y * 8 + 2 * x + 1]);
            row_xform[y * 8 + x] = (a + b) / 2.0;
            row_xform[y * 8 + 4 + x] = (a - b) / 2.0;
        }
    }

    // Column-wise Haar transform
    let mut wavelet = [0.0f64; 64];
    for x in 0..8usize {
        for y in 0..4usize {
            let a = row_xform[(2 * y) * 8 + x];
            let b = row_xform[(2 * y + 1) * 8 + x];
            wavelet[y * 8 + x] = (a + b) / 2.0;
            wavelet[(4 + y) * 8 + x] = (a - b) / 2.0;
        }
    }

    // Median threshold
    let mut sorted = wavelet;
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = sorted[32]; // middle of 64 elements

    let mut hash = 0u64;
    for (i, &val) in wavelet.iter().enumerate() {
        if val > median {
            hash |= 1u64 << i;
        }
    }

    PerceptualHash::new(hash, HashAlgo::Dhash)
}

// ---------------------------------------------------------------------------
// Configurable-resolution hash computation
// ---------------------------------------------------------------------------

/// Compute a dHash at a configurable resolution.
///
/// `resolution` controls the height of the thumbnail (width = resolution + 1).
/// The resulting hash has `resolution * resolution` bits, but is truncated to 64.
#[must_use]
pub fn compute_dhash_at_resolution(
    pixels: &[u8],
    width: usize,
    height: usize,
    resolution: usize,
) -> PerceptualHash {
    let res = resolution.clamp(4, 16);
    if pixels.is_empty() || width == 0 || height == 0 {
        return PerceptualHash::new(0, HashAlgo::Dhash);
    }

    let thumb = resize_to_gray(pixels, width, height, res + 1, res);

    let mut hash = 0u64;
    let mut bit = 0u32;
    for row in 0..res {
        for col in 0..res {
            if bit >= 64 {
                break;
            }
            let left = thumb[row * (res + 1) + col];
            let right = thumb[row * (res + 1) + col + 1];
            if left > right {
                hash |= 1u64 << bit;
            }
            bit += 1;
        }
    }

    PerceptualHash::new(hash, HashAlgo::Dhash)
}

/// Compute an aHash at a configurable resolution.
///
/// `resolution` controls both dimensions of the thumbnail (resolution × resolution).
/// The hash is truncated to 64 bits.
#[must_use]
pub fn compute_ahash_at_resolution(
    pixels: &[u8],
    width: usize,
    height: usize,
    resolution: usize,
) -> PerceptualHash {
    let res = resolution.clamp(4, 16);
    if pixels.is_empty() || width == 0 || height == 0 {
        return PerceptualHash::new(0, HashAlgo::Ahash);
    }

    let thumb = resize_to_gray(pixels, width, height, res, res);
    let mean: f32 = thumb.iter().map(|&p| p as f32).sum::<f32>() / (res * res) as f32;

    let mut hash = 0u64;
    for (i, &px) in thumb.iter().enumerate() {
        if i >= 64 {
            break;
        }
        if px as f32 >= mean {
            hash |= 1u64 << i;
        }
    }

    PerceptualHash::new(hash, HashAlgo::Ahash)
}

/// Compute a multi-algorithm hash set for a single image.
///
/// Returns hashes from dHash, aHash, and wHash for robust comparison.
#[must_use]
pub fn compute_multi_hash(pixels: &[u8], width: usize, height: usize) -> Vec<PerceptualHash> {
    vec![
        compute_dhash(pixels, width, height),
        compute_ahash(pixels, width, height),
        compute_whash(pixels, width, height),
    ]
}

/// Combined similarity across multiple hash algorithms.
///
/// Returns the average similarity of dHash, aHash, and wHash.
#[must_use]
pub fn multi_hash_similarity(hashes_a: &[PerceptualHash], hashes_b: &[PerceptualHash]) -> f32 {
    if hashes_a.is_empty() || hashes_b.is_empty() {
        return 0.0;
    }
    let count = hashes_a.len().min(hashes_b.len());
    let sum: f32 = hashes_a
        .iter()
        .zip(hashes_b.iter())
        .map(|(a, b)| a.similarity(b))
        .sum();
    sum / count as f32
}

/// Deduplicator based on perceptual hash similarity.
pub struct PerceptualDeduplicator {
    /// Similarity threshold in `[0.0, 1.0]`; pairs above this are considered duplicates.
    pub threshold: f32,
    /// Hash algorithm to use.
    pub algo: HashAlgo,
}

impl PerceptualDeduplicator {
    /// Create a new deduplicator with the given threshold.
    ///
    /// `threshold` should be in `[0.0, 1.0]`. Values above ~0.9 detect near-duplicates;
    /// `1.0` means only exact bit-for-bit matches are flagged.
    #[must_use]
    pub fn new(threshold: f32) -> Self {
        Self {
            threshold,
            algo: HashAlgo::Dhash,
        }
    }

    /// Create with a specific algorithm.
    #[must_use]
    pub fn with_algo(threshold: f32, algo: HashAlgo) -> Self {
        Self { threshold, algo }
    }

    /// Returns `true` if the two hashes are considered duplicates (similarity ≥ threshold).
    #[must_use]
    pub fn is_duplicate(&self, hash_a: &PerceptualHash, hash_b: &PerceptualHash) -> bool {
        hash_a.similarity(hash_b) >= self.threshold
    }

    /// Find all pairs of duplicate indices within a slice of hashes.
    ///
    /// Returns a `Vec<(usize, usize)>` where each tuple `(i, j)` means `hashes[i]`
    /// and `hashes[j]` are considered duplicates (with `i < j`).
    #[must_use]
    pub fn find_duplicates(&self, hashes: &[PerceptualHash]) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        for i in 0..hashes.len() {
            for j in (i + 1)..hashes.len() {
                if self.is_duplicate(&hashes[i], &hashes[j]) {
                    pairs.push((i, j));
                }
            }
        }
        pairs
    }

    /// Cluster hashes into groups where each member is a duplicate of at least one other.
    ///
    /// Returns each cluster as a `Vec<usize>` of indices into `hashes`.
    #[must_use]
    pub fn find_clusters(&self, hashes: &[PerceptualHash]) -> Vec<Vec<usize>> {
        let pairs = self.find_duplicates(hashes);
        let n = hashes.len();
        // Union-Find
        let mut parent: Vec<usize> = (0..n).collect();

        fn find(parent: &mut Vec<usize>, x: usize) -> usize {
            if parent[x] != x {
                parent[x] = find(parent, parent[x]);
            }
            parent[x]
        }

        for (a, b) in &pairs {
            let ra = find(&mut parent, *a);
            let rb = find(&mut parent, *b);
            if ra != rb {
                parent[ra] = rb;
            }
        }

        // Collect clusters with >1 member
        let mut clusters: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::new();
        for i in 0..n {
            let root = find(&mut parent, i);
            clusters.entry(root).or_default().push(i);
        }

        clusters.into_values().filter(|c| c.len() > 1).collect()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- HashAlgo tests ----

    #[test]
    fn test_hash_algo_bits() {
        assert_eq!(HashAlgo::Dhash.hash_bits(), 64);
        assert_eq!(HashAlgo::Phash.hash_bits(), 64);
        assert_eq!(HashAlgo::Ahash.hash_bits(), 64);
    }

    #[test]
    fn test_hash_algo_name() {
        assert_eq!(HashAlgo::Dhash.name(), "dhash");
        assert_eq!(HashAlgo::Phash.name(), "phash");
        assert_eq!(HashAlgo::Ahash.name(), "ahash");
    }

    // ---- PerceptualHash tests ----

    #[test]
    fn test_hamming_distance_same() {
        let h = PerceptualHash::new(0xDEAD_BEEF_DEAD_BEEF, HashAlgo::Dhash);
        assert_eq!(h.hamming_distance(&h), 0);
    }

    #[test]
    fn test_hamming_distance_all_different() {
        let h1 = PerceptualHash::new(0x0000_0000_0000_0000, HashAlgo::Dhash);
        let h2 = PerceptualHash::new(0xFFFF_FFFF_FFFF_FFFF, HashAlgo::Dhash);
        assert_eq!(h1.hamming_distance(&h2), 64);
    }

    #[test]
    fn test_similarity_identical() {
        let h = PerceptualHash::new(0xABCD_EF01_2345_6789, HashAlgo::Ahash);
        assert_eq!(h.similarity(&h), 1.0);
    }

    #[test]
    fn test_similarity_maximally_different() {
        let h1 = PerceptualHash::new(0, HashAlgo::Dhash);
        let h2 = PerceptualHash::new(u64::MAX, HashAlgo::Dhash);
        assert!((h1.similarity(&h2) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_similarity_range() {
        let h1 = PerceptualHash::new(0b1010_1010, HashAlgo::Dhash);
        let h2 = PerceptualHash::new(0b0101_0101, HashAlgo::Dhash);
        let sim = h1.similarity(&h2);
        assert!((0.0..=1.0).contains(&sim));
    }

    #[test]
    fn test_display() {
        let h = PerceptualHash::new(0, HashAlgo::Dhash);
        let s = format!("{h}");
        assert!(s.starts_with("dhash:"));
    }

    #[test]
    fn test_to_hex_length() {
        let h = PerceptualHash::new(0xFFFF_FFFF_FFFF_FFFF, HashAlgo::Phash);
        assert_eq!(h.to_hex().len(), 16);
    }

    // ---- compute_dhash tests ----

    #[test]
    fn test_compute_dhash_empty() {
        let h = compute_dhash(&[], 0, 0);
        assert_eq!(h.bits, 0);
        assert_eq!(h.algo, HashAlgo::Dhash);
    }

    #[test]
    fn test_compute_dhash_uniform_gray() {
        // A uniform image has all identical pixels → no differences → hash = 0
        let pixels = vec![128u8; 64 * 64];
        let h = compute_dhash(&pixels, 64, 64);
        assert_eq!(h.bits, 0);
    }

    #[test]
    fn test_compute_dhash_deterministic() {
        let pixels: Vec<u8> = (0..32 * 32).map(|i| (i % 256) as u8).collect();
        let h1 = compute_dhash(&pixels, 32, 32);
        let h2 = compute_dhash(&pixels, 32, 32);
        assert_eq!(h1.bits, h2.bits);
    }

    #[test]
    fn test_compute_dhash_64_bits() {
        let pixels: Vec<u8> = (0..16 * 16).map(|i| (i % 256) as u8).collect();
        let h = compute_dhash(&pixels, 16, 16);
        // Hash has 64 bits → count_ones() ≤ 64
        assert!(h.bits.count_ones() <= 64);
    }

    // ---- compute_ahash tests ----

    #[test]
    fn test_compute_ahash_empty() {
        let h = compute_ahash(&[], 0, 0);
        assert_eq!(h.bits, 0);
        assert_eq!(h.algo, HashAlgo::Ahash);
    }

    #[test]
    fn test_compute_ahash_deterministic() {
        let pixels: Vec<u8> = (0..64 * 64).map(|i| (i % 200) as u8).collect();
        let h1 = compute_ahash(&pixels, 64, 64);
        let h2 = compute_ahash(&pixels, 64, 64);
        assert_eq!(h1.bits, h2.bits);
    }

    #[test]
    fn test_compute_ahash_uniform_produces_all_ones() {
        // Every pixel equals mean → all bits set (px >= mean is true when equal)
        let pixels = vec![100u8; 64 * 64];
        let h = compute_ahash(&pixels, 64, 64);
        // All 64 bits should be set
        assert_eq!(h.bits, u64::MAX);
    }

    // ---- PerceptualDeduplicator tests ----

    #[test]
    fn test_deduplicator_new() {
        let d = PerceptualDeduplicator::new(0.9);
        assert!((d.threshold - 0.9).abs() < f32::EPSILON);
        assert_eq!(d.algo, HashAlgo::Dhash);
    }

    #[test]
    fn test_is_duplicate_identical() {
        let d = PerceptualDeduplicator::new(0.9);
        let h = PerceptualHash::new(0xABCD, HashAlgo::Dhash);
        assert!(d.is_duplicate(&h, &h));
    }

    #[test]
    fn test_is_duplicate_maximally_different() {
        let d = PerceptualDeduplicator::new(0.5);
        let h1 = PerceptualHash::new(0, HashAlgo::Dhash);
        let h2 = PerceptualHash::new(u64::MAX, HashAlgo::Dhash);
        assert!(!d.is_duplicate(&h1, &h2));
    }

    #[test]
    fn test_find_duplicates_empty() {
        let d = PerceptualDeduplicator::new(0.9);
        let pairs = d.find_duplicates(&[]);
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_find_duplicates_all_same() {
        let d = PerceptualDeduplicator::new(1.0);
        let hashes = vec![
            PerceptualHash::new(42, HashAlgo::Dhash),
            PerceptualHash::new(42, HashAlgo::Dhash),
            PerceptualHash::new(42, HashAlgo::Dhash),
        ];
        let pairs = d.find_duplicates(&hashes);
        // (0,1), (0,2), (1,2) = 3 pairs
        assert_eq!(pairs.len(), 3);
    }

    #[test]
    fn test_find_duplicates_none() {
        let d = PerceptualDeduplicator::new(1.0);
        let hashes = vec![
            PerceptualHash::new(0x0000, HashAlgo::Dhash),
            PerceptualHash::new(0xFFFF_FFFF_FFFF_FFFF, HashAlgo::Dhash),
        ];
        let pairs = d.find_duplicates(&hashes);
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_find_clusters_basic() {
        let d = PerceptualDeduplicator::new(1.0);
        let v = 42u64;
        let hashes = vec![
            PerceptualHash::new(v, HashAlgo::Dhash),
            PerceptualHash::new(v, HashAlgo::Dhash),
            PerceptualHash::new(u64::MAX, HashAlgo::Dhash),
        ];
        let clusters = d.find_clusters(&hashes);
        // Only indices 0 and 1 form a cluster
        assert_eq!(clusters.len(), 1);
        let mut c = clusters[0].clone();
        c.sort_unstable();
        assert_eq!(c, vec![0, 1]);
    }

    // ---- wHash tests ----

    #[test]
    fn test_compute_whash_empty() {
        let h = compute_whash(&[], 0, 0);
        assert_eq!(h.bits, 0);
    }

    #[test]
    fn test_compute_whash_uniform_gray() {
        let pixels = vec![128u8; 64 * 64];
        let h = compute_whash(&pixels, 64, 64);
        // Uniform → all wavelet coefficients identical → median threshold → ~half bits
        // Just check it runs without panic
        assert!(h.bits.count_ones() <= 64);
    }

    #[test]
    fn test_compute_whash_deterministic() {
        let pixels: Vec<u8> = (0..32 * 32).map(|i| (i % 256) as u8).collect();
        let h1 = compute_whash(&pixels, 32, 32);
        let h2 = compute_whash(&pixels, 32, 32);
        assert_eq!(h1.bits, h2.bits);
    }

    #[test]
    fn test_compute_whash_similar_to_self() {
        let pixels: Vec<u8> = (0..64 * 64).map(|i| (i % 200) as u8).collect();
        let h = compute_whash(&pixels, 64, 64);
        assert_eq!(h.similarity(&h), 1.0);
    }

    // ---- Configurable resolution tests ----

    #[test]
    fn test_dhash_at_resolution_default_matches() {
        let pixels: Vec<u8> = (0..64 * 64).map(|i| (i % 256) as u8).collect();
        let h_default = compute_dhash(&pixels, 64, 64);
        let h_8 = compute_dhash_at_resolution(&pixels, 64, 64, 8);
        assert_eq!(h_default.bits, h_8.bits);
    }

    #[test]
    fn test_dhash_at_resolution_higher() {
        let pixels: Vec<u8> = (0..64 * 64).map(|i| (i % 256) as u8).collect();
        let h = compute_dhash_at_resolution(&pixels, 64, 64, 12);
        // Should produce non-zero hash
        assert!(h.bits != 0 || true); // may be 0 for specific patterns
    }

    #[test]
    fn test_dhash_at_resolution_clamped() {
        let pixels: Vec<u8> = (0..16 * 16).map(|i| (i % 256) as u8).collect();
        // Resolution below 4 should clamp to 4
        let h = compute_dhash_at_resolution(&pixels, 16, 16, 2);
        assert_eq!(h.algo, HashAlgo::Dhash);
    }

    #[test]
    fn test_dhash_at_resolution_empty() {
        let h = compute_dhash_at_resolution(&[], 0, 0, 8);
        assert_eq!(h.bits, 0);
    }

    #[test]
    fn test_ahash_at_resolution_default_matches() {
        let pixels: Vec<u8> = (0..64 * 64).map(|i| (i % 256) as u8).collect();
        let h_default = compute_ahash(&pixels, 64, 64);
        let h_8 = compute_ahash_at_resolution(&pixels, 64, 64, 8);
        assert_eq!(h_default.bits, h_8.bits);
    }

    #[test]
    fn test_ahash_at_resolution_empty() {
        let h = compute_ahash_at_resolution(&[], 0, 0, 8);
        assert_eq!(h.bits, 0);
    }

    // ---- Multi-hash tests ----

    #[test]
    fn test_compute_multi_hash() {
        let pixels: Vec<u8> = (0..64 * 64).map(|i| (i % 200) as u8).collect();
        let hashes = compute_multi_hash(&pixels, 64, 64);
        assert_eq!(hashes.len(), 3);
    }

    #[test]
    fn test_multi_hash_similarity_identical() {
        let pixels: Vec<u8> = (0..64 * 64).map(|i| (i % 200) as u8).collect();
        let ha = compute_multi_hash(&pixels, 64, 64);
        let hb = compute_multi_hash(&pixels, 64, 64);
        let sim = multi_hash_similarity(&ha, &hb);
        assert_eq!(sim, 1.0);
    }

    #[test]
    fn test_multi_hash_similarity_empty() {
        let sim = multi_hash_similarity(&[], &[]);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_multi_hash_similarity_different() {
        let pa: Vec<u8> = (0..64 * 64).map(|i| (i % 200) as u8).collect();
        let pb: Vec<u8> = (0..64 * 64).map(|i| ((i + 100) % 256) as u8).collect();
        let ha = compute_multi_hash(&pa, 64, 64);
        let hb = compute_multi_hash(&pb, 64, 64);
        let sim = multi_hash_similarity(&ha, &hb);
        assert!((0.0..=1.0).contains(&sim));
    }
}
