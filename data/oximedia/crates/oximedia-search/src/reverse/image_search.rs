//! Reverse image search using perceptual hashing.
//!
//! Implements both an average-hash (`AHash`) and a DCT-based perceptual hash
//! (`pHash`) over 8×8 luma grids, plus brute-force Hamming-distance search.

// ──────────────────────────────────────────────────────────────────────────────
// PHash  (DCT-based perceptual hash)
// ──────────────────────────────────────────────────────────────────────────────

/// 64-bit DCT-based perceptual hash.
///
/// Each bit corresponds to one DCT coefficient compared to the mean of all
/// DCT coefficients in the top-left 8×8 region of the full 8×8 DCT output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PHash(pub u64);

impl PHash {
    /// Computes the pHash for an 8×8 luma grid (64 bytes, row-major).
    ///
    /// Steps:
    /// 1. Compute the 8×8 DCT of the luma values.
    /// 2. Take all 64 coefficients (the full 8×8 block).
    /// 3. Compute the mean of those coefficients, excluding the DC term (index 0).
    /// 4. Set bit *i* if `dct[i] > mean`.
    #[must_use]
    pub fn compute(luma_8x8: &[u8; 64]) -> Self {
        let dct = compute_dct8x8(luma_8x8);
        // Exclude the DC term (index 0) from the mean.
        let mean = dct[1..].iter().sum::<f32>() / 63.0;
        let mut hash: u64 = 0;
        for (i, &coeff) in dct.iter().enumerate() {
            if coeff > mean {
                hash |= 1u64 << i;
            }
        }
        Self(hash)
    }
}

/// Computes the Hamming distance between two `PHash` values.
#[must_use]
pub fn hamming_distance(a: PHash, b: PHash) -> u32 {
    (a.0 ^ b.0).count_ones()
}

// ──────────────────────────────────────────────────────────────────────────────
// AverageHash
// ──────────────────────────────────────────────────────────────────────────────

/// 64-bit average hash.
///
/// Simpler and faster than pHash; slightly less robust to transformations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AverageHash(pub u64);

/// Computes the average hash for an 8×8 luma grid.
///
/// Each bit is 1 when the corresponding pixel is above the mean luma value.
#[must_use]
pub fn compute_ahash(luma_8x8: &[u8; 64]) -> AverageHash {
    let mean: u32 = luma_8x8.iter().map(|&v| u32::from(v)).sum::<u32>() / 64;
    let mut hash: u64 = 0;
    for (i, &pixel) in luma_8x8.iter().enumerate() {
        if u32::from(pixel) > mean {
            hash |= 1u64 << i;
        }
    }
    AverageHash(hash)
}

/// Computes the Hamming distance between two `AverageHash` values.
#[must_use]
pub fn ahash_hamming_distance(a: AverageHash, b: AverageHash) -> u32 {
    (a.0 ^ b.0).count_ones()
}

// ──────────────────────────────────────────────────────────────────────────────
// ReverseImageIndex
// ──────────────────────────────────────────────────────────────────────────────

/// Brute-force reverse image search index backed by pHash.
pub struct ReverseImageIndex {
    entries: Vec<(u64, PHash)>,
}

impl ReverseImageIndex {
    /// Creates a new, empty index.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Adds a pHash for a media item.  Replaces any existing entry for `id`.
    pub fn add(&mut self, id: u64, hash: PHash) {
        if let Some(pos) = self.entries.iter().position(|(eid, _)| *eid == id) {
            self.entries[pos] = (id, hash);
        } else {
            self.entries.push((id, hash));
        }
    }

    /// Returns the number of stored entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when the index contains no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Searches for images within `max_distance` Hamming bits of `query`.
    ///
    /// Returns `(id, hamming_distance)` pairs sorted by ascending distance.
    #[must_use]
    pub fn search(&self, query: PHash, max_distance: u32) -> Vec<(u64, u32)> {
        let mut results: Vec<(u64, u32)> = self
            .entries
            .iter()
            .filter_map(|(id, hash)| {
                let dist = hamming_distance(query, *hash);
                if dist <= max_distance {
                    Some((*id, dist))
                } else {
                    None
                }
            })
            .collect();

        results.sort_by_key(|&(_, d)| d);
        results
    }
}

impl Default for ReverseImageIndex {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// DCT helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Computes the 8×8 2-D DCT-II of `pixels` and returns a flat 64-element
/// array of coefficients in row-major order.
fn compute_dct8x8(pixels: &[u8; 64]) -> [f32; 64] {
    use std::f32::consts::PI;

    // Convert pixels to f32.
    let mut f = [0.0_f32; 64];
    for (i, &p) in pixels.iter().enumerate() {
        f[i] = f32::from(p);
    }

    // Precompute cosine table: cos_table[k][n] = cos(π * k * (2n+1) / 16)
    let mut cos_table = [[0.0_f32; 8]; 8];
    for k in 0..8usize {
        for n in 0..8usize {
            cos_table[k][n] = ((PI * k as f32 * (2 * n + 1) as f32) / 16.0).cos();
        }
    }

    fn alpha(k: usize) -> f32 {
        if k == 0 {
            1.0 / (8.0_f32).sqrt()
        } else {
            (2.0_f32 / 8.0).sqrt()
        }
    }

    let mut dct = [0.0_f32; 64];
    for u in 0..8usize {
        for v in 0..8usize {
            let mut sum = 0.0_f32;
            for x in 0..8usize {
                for y in 0..8usize {
                    sum += f[x * 8 + y] * cos_table[u][x] * cos_table[v][y];
                }
            }
            dct[u * 8 + v] = alpha(u) * alpha(v) * sum;
        }
    }
    dct
}

/// Alias so that `compute_phash` mirrors the API described in the task spec.
#[must_use]
pub fn compute_phash(luma_8x8: &[u8; 64]) -> PHash {
    PHash::compute(luma_8x8)
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform(value: u8) -> [u8; 64] {
        [value; 64]
    }

    fn gradient() -> [u8; 64] {
        let mut g = [0u8; 64];
        for (i, v) in g.iter_mut().enumerate() {
            *v = i as u8 * 4;
        }
        g
    }

    // ── PHash ──

    #[test]
    fn test_phash_identical_images() {
        let luma = gradient();
        let h1 = compute_phash(&luma);
        let h2 = compute_phash(&luma);
        assert_eq!(hamming_distance(h1, h2), 0);
    }

    #[test]
    fn test_phash_different_images() {
        let h1 = compute_phash(&uniform(0));
        let h2 = compute_phash(&uniform(255));
        // Should differ by several bits.
        assert!(hamming_distance(h1, h2) > 0);
    }

    #[test]
    fn test_phash_uniform_image() {
        // Uniform image – all DCT coefficients except DC are zero.
        let luma = uniform(128);
        let hash = compute_phash(&luma);
        // The hash should be well-defined (no panic).
        let _ = hash;
    }

    #[test]
    fn test_hamming_distance_same() {
        let h = PHash(0xDEAD_BEEF_1234_5678);
        assert_eq!(hamming_distance(h, h), 0);
    }

    #[test]
    fn test_hamming_distance_all_bits() {
        let h1 = PHash(u64::MAX);
        let h2 = PHash(0);
        assert_eq!(hamming_distance(h1, h2), 64);
    }

    // ── AverageHash ──

    #[test]
    fn test_ahash_identical_images() {
        let luma = gradient();
        let h1 = compute_ahash(&luma);
        let h2 = compute_ahash(&luma);
        assert_eq!(ahash_hamming_distance(h1, h2), 0);
    }

    #[test]
    fn test_ahash_uniform_dark_image() {
        // All pixels below mean → mean == value, none strictly greater → all 0.
        let luma = uniform(0);
        let h = compute_ahash(&luma);
        assert_eq!(h.0, 0);
    }

    #[test]
    fn test_ahash_hamming_distance() {
        let h1 = AverageHash(0b1010);
        let h2 = AverageHash(0b1100);
        assert_eq!(ahash_hamming_distance(h1, h2), 2);
    }

    // ── ReverseImageIndex ──

    #[test]
    fn test_index_add_and_search_exact() {
        let luma = gradient();
        let hash = compute_phash(&luma);
        let mut idx = ReverseImageIndex::new();
        idx.add(42, hash);
        let results = idx.search(hash, 0);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], (42, 0));
    }

    #[test]
    fn test_index_search_with_threshold() {
        let mut idx = ReverseImageIndex::new();
        idx.add(1, PHash(0x0000_0000_0000_0000)); // 0 distance from query
        idx.add(2, PHash(0xFFFF_FFFF_FFFF_FFFF)); // 64 bits different
        idx.add(3, PHash(0x0000_0000_0000_0001)); // 1 bit different

        let query = PHash(0x0000_0000_0000_0000);
        let results = idx.search(query, 2);
        let ids: Vec<u64> = results.iter().map(|(id, _)| *id).collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&3));
        assert!(!ids.contains(&2));
    }

    #[test]
    fn test_index_sorted_by_distance() {
        let mut idx = ReverseImageIndex::new();
        idx.add(1, PHash(0x0000_0000_0000_0003)); // 2 bits
        idx.add(2, PHash(0x0000_0000_0000_0001)); // 1 bit
        idx.add(3, PHash(0x0000_0000_0000_0000)); // 0 bits

        let query = PHash(0x0000_0000_0000_0000);
        let results = idx.search(query, 4);
        assert_eq!(results[0].1, 0);
        assert_eq!(results[1].1, 1);
        assert_eq!(results[2].1, 2);
    }

    #[test]
    fn test_index_replace_existing_id() {
        let mut idx = ReverseImageIndex::new();
        idx.add(1, PHash(0x00));
        idx.add(1, PHash(0xFF)); // replace
        assert_eq!(idx.len(), 1);
        let results = idx.search(PHash(0xFF), 0);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 1);
    }

    #[test]
    fn test_index_empty_search() {
        let idx = ReverseImageIndex::new();
        let results = idx.search(PHash(0), 10);
        assert!(results.is_empty());
    }
}
