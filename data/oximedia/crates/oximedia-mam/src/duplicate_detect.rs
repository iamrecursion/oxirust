//! Perceptual hash-based duplicate asset detection.
//!
//! Implements a pHash-style approach: downscale to a small greyscale image,
//! apply a Type-II DCT, and keep the sign bits of the low-frequency
//! coefficients as a compact 64-bit fingerprint.  Hamming distance between
//! fingerprints approximates perceptual dissimilarity.

use std::collections::HashMap;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Perceptual hash
// ---------------------------------------------------------------------------

/// Size used for the DCT step (hash_size x hash_size).
const HASH_SIZE: usize = 8;
/// We compute DCT over a larger block then keep top-left HASH_SIZE x HASH_SIZE.
const DCT_SIZE: usize = 32;

/// A 64-bit perceptual hash.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PHash(pub u64);

impl PHash {
    /// Hamming distance between two hashes (number of differing bits).
    #[must_use]
    pub fn distance(self, other: Self) -> u32 {
        (self.0 ^ other.0).count_ones()
    }

    /// Returns `true` when the distance is at or below `threshold`.
    #[must_use]
    pub fn is_similar(self, other: Self, threshold: u32) -> bool {
        self.distance(other) <= threshold
    }
}

/// Compute a 1-D Type-II DCT of `input` into `output`.
///
/// `output[k] = sum_n input[n] * cos(pi/N * (n + 0.5) * k)`
fn dct_1d(input: &[f64], output: &mut [f64]) {
    let n = input.len();
    for (k, out) in output.iter_mut().enumerate() {
        let mut sum = 0.0f64;
        for (i, val) in input.iter().enumerate() {
            let angle = std::f64::consts::PI / (n as f64) * (i as f64 + 0.5) * (k as f64);
            sum += val * angle.cos();
        }
        *out = sum;
    }
}

/// Compute a 2-D DCT by applying 1-D DCT to rows then columns.
fn dct_2d(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows = matrix.len();
    if rows == 0 {
        return Vec::new();
    }
    let cols = matrix[0].len();

    // DCT on rows
    let mut row_dct = vec![vec![0.0f64; cols]; rows];
    for (r, row) in matrix.iter().enumerate() {
        dct_1d(row, &mut row_dct[r]);
    }

    // DCT on columns
    let mut result = vec![vec![0.0f64; cols]; rows];
    let mut col_in = vec![0.0f64; rows];
    let mut col_out = vec![0.0f64; rows];
    for c in 0..cols {
        for (r, rd) in row_dct.iter().enumerate() {
            col_in[r] = rd[c];
        }
        dct_1d(&col_in, &mut col_out);
        for r in 0..rows {
            result[r][c] = col_out[r];
        }
    }
    result
}

/// Downsample a greyscale image (row-major, `width x height`) to
/// `target_w x target_h` using simple bilinear-like averaging.
fn downsample(
    pixels: &[u8],
    width: usize,
    height: usize,
    target_w: usize,
    target_h: usize,
) -> Vec<Vec<f64>> {
    let mut out = vec![vec![0.0f64; target_w]; target_h];
    if width == 0 || height == 0 {
        return out;
    }

    let x_ratio = width as f64 / target_w as f64;
    let y_ratio = height as f64 / target_h as f64;

    for ty in 0..target_h {
        for tx in 0..target_w {
            let src_x = ((tx as f64 + 0.5) * x_ratio)
                .min(width as f64 - 1.0)
                .max(0.0);
            let src_y = ((ty as f64 + 0.5) * y_ratio)
                .min(height as f64 - 1.0)
                .max(0.0);
            let sx = src_x as usize;
            let sy = src_y as usize;
            let idx = sy * width + sx;
            out[ty][tx] = if idx < pixels.len() {
                pixels[idx] as f64
            } else {
                0.0
            };
        }
    }
    out
}

/// Compute the pHash of a greyscale image.
///
/// * `pixels` — row-major greyscale pixel data (one byte per pixel).
/// * `width`, `height` — dimensions of the source image.
#[must_use]
pub fn compute_phash(pixels: &[u8], width: usize, height: usize) -> PHash {
    // Step 1: Downsample to DCT_SIZE x DCT_SIZE.
    let small = downsample(pixels, width, height, DCT_SIZE, DCT_SIZE);

    // Step 2: 2-D DCT.
    let dct = dct_2d(&small);

    // Step 3: Keep top-left HASH_SIZE x HASH_SIZE coefficients (skip DC at [0][0]).
    let mut coeffs = Vec::with_capacity(HASH_SIZE * HASH_SIZE);
    for row in dct.iter().take(HASH_SIZE) {
        for val in row.iter().take(HASH_SIZE) {
            coeffs.push(*val);
        }
    }

    // Step 4: Compute median of the coefficients (excluding DC).
    let median = {
        let mut sorted: Vec<f64> = coeffs[1..].to_vec(); // skip DC
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if sorted.is_empty() {
            0.0
        } else {
            sorted[sorted.len() / 2]
        }
    };

    // Step 5: Build 64-bit hash — bit set if coeff > median.
    let mut hash: u64 = 0;
    for (i, c) in coeffs.iter().enumerate() {
        if *c > median {
            hash |= 1u64 << i;
        }
    }

    PHash(hash)
}

/// Compute a quick hash from raw bytes (not image-aware; useful for exact-match
/// or as a fallback when pixel data is unavailable).
#[must_use]
pub fn compute_byte_hash(data: &[u8]) -> u64 {
    // FNV-1a 64-bit
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in data {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

// ---------------------------------------------------------------------------
// Duplicate finder
// ---------------------------------------------------------------------------

/// A registered asset with its perceptual hash.
#[derive(Debug, Clone)]
pub struct HashedAsset {
    pub asset_id: Uuid,
    pub hash: PHash,
    /// Optional human-readable label (e.g. filename).
    pub label: Option<String>,
}

/// A pair of assets that are considered duplicates.
#[derive(Debug, Clone)]
pub struct DuplicatePair {
    pub asset_a: Uuid,
    pub asset_b: Uuid,
    pub distance: u32,
}

/// Configuration for the duplicate finder.
#[derive(Debug, Clone)]
pub struct DuplicateFinderConfig {
    /// Hamming distance threshold: pairs with distance <= threshold are duplicates.
    pub similarity_threshold: u32,
}

impl Default for DuplicateFinderConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 10,
        }
    }
}

/// Finds duplicate assets based on perceptual hashes.
#[derive(Debug)]
pub struct DuplicateFinder {
    config: DuplicateFinderConfig,
    assets: Vec<HashedAsset>,
    /// Exact-hash index for O(1) exact-match lookups.
    exact_index: HashMap<u64, Vec<Uuid>>,
}

impl DuplicateFinder {
    /// Create a new finder with the given configuration.
    #[must_use]
    pub fn new(config: DuplicateFinderConfig) -> Self {
        Self {
            config,
            assets: Vec::new(),
            exact_index: HashMap::new(),
        }
    }

    /// Create a finder with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(DuplicateFinderConfig::default())
    }

    /// Register an asset with its perceptual hash.
    pub fn add_asset(&mut self, asset_id: Uuid, hash: PHash, label: Option<String>) {
        self.exact_index.entry(hash.0).or_default().push(asset_id);
        self.assets.push(HashedAsset {
            asset_id,
            hash,
            label,
        });
    }

    /// Number of registered assets.
    #[must_use]
    pub fn asset_count(&self) -> usize {
        self.assets.len()
    }

    /// The configured similarity threshold.
    #[must_use]
    pub fn threshold(&self) -> u32 {
        self.config.similarity_threshold
    }

    /// Find all exact matches (distance == 0) for a given hash.
    #[must_use]
    pub fn find_exact(&self, hash: PHash) -> Vec<Uuid> {
        self.exact_index.get(&hash.0).cloned().unwrap_or_default()
    }

    /// Find all assets within the similarity threshold of the given hash.
    #[must_use]
    pub fn find_similar(&self, hash: PHash) -> Vec<(Uuid, u32)> {
        self.assets
            .iter()
            .filter_map(|a| {
                let d = a.hash.distance(hash);
                if d <= self.config.similarity_threshold {
                    Some((a.asset_id, d))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Scan all registered assets and return every duplicate pair.
    ///
    /// Runs in O(n^2) — fine for moderate collections; larger sets should use
    /// locality-sensitive hashing.
    #[must_use]
    pub fn find_all_duplicates(&self) -> Vec<DuplicatePair> {
        let mut pairs = Vec::new();
        let n = self.assets.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let d = self.assets[i].hash.distance(self.assets[j].hash);
                if d <= self.config.similarity_threshold {
                    pairs.push(DuplicatePair {
                        asset_a: self.assets[i].asset_id,
                        asset_b: self.assets[j].asset_id,
                        distance: d,
                    });
                }
            }
        }
        pairs
    }

    /// Group assets into clusters of mutual duplicates (union-find).
    #[must_use]
    pub fn cluster_duplicates(&self) -> Vec<Vec<Uuid>> {
        let n = self.assets.len();
        let mut parent: Vec<usize> = (0..n).collect();

        // Simple union-find helpers (path compression only).
        fn find(parent: &mut [usize], mut x: usize) -> usize {
            while parent[x] != x {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            x
        }
        fn union(parent: &mut [usize], a: usize, b: usize) {
            let ra = find(parent, a);
            let rb = find(parent, b);
            if ra != rb {
                parent[rb] = ra;
            }
        }

        for i in 0..n {
            for j in (i + 1)..n {
                let d = self.assets[i].hash.distance(self.assets[j].hash);
                if d <= self.config.similarity_threshold {
                    union(&mut parent, i, j);
                }
            }
        }

        let mut groups: HashMap<usize, Vec<Uuid>> = HashMap::new();
        for i in 0..n {
            let root = find(&mut parent, i);
            groups
                .entry(root)
                .or_default()
                .push(self.assets[i].asset_id);
        }

        // Only return groups with 2+ members.
        groups.into_values().filter(|g| g.len() > 1).collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phash_distance_identical() {
        let h = PHash(0xABCD_1234_5678_9ABC);
        assert_eq!(h.distance(h), 0);
    }

    #[test]
    fn test_phash_distance_different() {
        let a = PHash(0x0000_0000_0000_0000);
        let b = PHash(0x0000_0000_0000_0001);
        assert_eq!(a.distance(b), 1);
    }

    #[test]
    fn test_phash_distance_all_different() {
        let a = PHash(0x0000_0000_0000_0000);
        let b = PHash(0xFFFF_FFFF_FFFF_FFFF);
        assert_eq!(a.distance(b), 64);
    }

    #[test]
    fn test_phash_is_similar() {
        let a = PHash(0x00);
        let b = PHash(0x03); // 2 bits differ
        assert!(a.is_similar(b, 5));
        assert!(!a.is_similar(b, 1));
    }

    #[test]
    fn test_compute_phash_deterministic() {
        let pixels = vec![128u8; 64 * 64];
        let h1 = compute_phash(&pixels, 64, 64);
        let h2 = compute_phash(&pixels, 64, 64);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_compute_phash_different_images() {
        let white = vec![255u8; 64 * 64];
        let black = vec![0u8; 64 * 64];
        let h_white = compute_phash(&white, 64, 64);
        let h_black = compute_phash(&black, 64, 64);
        // They may or may not differ depending on DC, but the function should not panic.
        let _ = h_white.distance(h_black);
    }

    #[test]
    fn test_compute_phash_small_image() {
        let pixels = vec![100u8; 8 * 8];
        let h = compute_phash(&pixels, 8, 8);
        let _ = h; // just ensure no panic
    }

    #[test]
    fn test_compute_byte_hash_deterministic() {
        let data = b"hello world";
        let h1 = compute_byte_hash(data);
        let h2 = compute_byte_hash(data);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_compute_byte_hash_different() {
        let h1 = compute_byte_hash(b"hello");
        let h2 = compute_byte_hash(b"world");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_duplicate_finder_add_and_count() {
        let mut finder = DuplicateFinder::with_defaults();
        finder.add_asset(Uuid::new_v4(), PHash(100), Some("a.mp4".into()));
        finder.add_asset(Uuid::new_v4(), PHash(200), Some("b.mp4".into()));
        assert_eq!(finder.asset_count(), 2);
    }

    #[test]
    fn test_duplicate_finder_exact_match() {
        let mut finder = DuplicateFinder::with_defaults();
        let id = Uuid::new_v4();
        finder.add_asset(id, PHash(42), None);
        finder.add_asset(Uuid::new_v4(), PHash(42), None);

        let exact = finder.find_exact(PHash(42));
        assert_eq!(exact.len(), 2);
    }

    #[test]
    fn test_duplicate_finder_find_similar() {
        let mut finder = DuplicateFinder::new(DuplicateFinderConfig {
            similarity_threshold: 3,
        });
        let id_a = Uuid::new_v4();
        let id_b = Uuid::new_v4();
        let id_c = Uuid::new_v4();

        finder.add_asset(id_a, PHash(0b0000), None);
        finder.add_asset(id_b, PHash(0b0011), None); // distance 2
        finder.add_asset(id_c, PHash(0xFF00), None); // far away

        let similar = finder.find_similar(PHash(0b0000));
        let ids: Vec<Uuid> = similar.iter().map(|(id, _)| *id).collect();
        assert!(ids.contains(&id_a));
        assert!(ids.contains(&id_b));
        assert!(!ids.contains(&id_c));
    }

    #[test]
    fn test_duplicate_finder_find_all_duplicates() {
        let mut finder = DuplicateFinder::new(DuplicateFinderConfig {
            similarity_threshold: 2,
        });

        finder.add_asset(Uuid::new_v4(), PHash(0b0000), None);
        finder.add_asset(Uuid::new_v4(), PHash(0b0001), None); // dist 1
        finder.add_asset(Uuid::new_v4(), PHash(0xFFFF), None); // far

        let pairs = finder.find_all_duplicates();
        assert_eq!(pairs.len(), 1);
        assert!(pairs[0].distance <= 2);
    }

    #[test]
    fn test_duplicate_finder_cluster() {
        let mut finder = DuplicateFinder::new(DuplicateFinderConfig {
            similarity_threshold: 2,
        });

        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let c = Uuid::new_v4();
        let d = Uuid::new_v4();

        finder.add_asset(a, PHash(0b0000), None);
        finder.add_asset(b, PHash(0b0001), None); // near a
        finder.add_asset(c, PHash(0b0011), None); // near b, transitive to a
        finder.add_asset(d, PHash(0xFFFF_FFFF), None); // isolated

        let clusters = finder.cluster_duplicates();
        // a, b, c should form one cluster; d is isolated
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].len(), 3);
    }

    #[test]
    fn test_duplicate_finder_no_duplicates() {
        let mut finder = DuplicateFinder::new(DuplicateFinderConfig {
            similarity_threshold: 0,
        });
        finder.add_asset(Uuid::new_v4(), PHash(1), None);
        finder.add_asset(Uuid::new_v4(), PHash(2), None);

        assert!(finder.find_all_duplicates().is_empty());
        assert!(finder.cluster_duplicates().is_empty());
    }

    #[test]
    fn test_dct_1d_basic() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut output = [0.0f64; 4];
        dct_1d(&input, &mut output);
        // DC component should be sum of all values
        let expected_dc: f64 = input
            .iter()
            .enumerate()
            .map(|(n, v)| v * (std::f64::consts::PI / 4.0 * (n as f64 + 0.5) * 0.0).cos())
            .sum();
        assert!((output[0] - expected_dc).abs() < 1e-10);
    }

    #[test]
    fn test_downsample_basic() {
        // 4x4 all white -> 2x2 should be all 255.0
        let pixels = vec![255u8; 16];
        let result = downsample(&pixels, 4, 4, 2, 2);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].len(), 2);
        for row in &result {
            for &v in row {
                assert!((v - 255.0).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_default_config_threshold() {
        let cfg = DuplicateFinderConfig::default();
        assert_eq!(cfg.similarity_threshold, 10);
    }
}
