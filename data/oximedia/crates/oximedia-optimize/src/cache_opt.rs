//! Cache optimization strategies for encoder working sets.
//!
//! Provides:
//! - Working set size estimation for various block sizes
//! - Cache locality hints for scan order selection
//! - Prefetch pattern generation for reference frame access
//! - L1/L2/L3 cache tier mapping

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Common cache tier sizes (in bytes).
pub const L1_CACHE_SIZE: usize = 32 * 1024; // 32 KiB
/// L2 cache typical size.
pub const L2_CACHE_SIZE: usize = 256 * 1024; // 256 KiB
/// L3 cache typical size.
pub const L3_CACHE_SIZE: usize = 8 * 1024 * 1024; // 8 MiB

/// Cache tier classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum CacheTier {
    /// Fits in L1.
    L1,
    /// Fits in L2.
    L2,
    /// Fits in L3.
    L3,
    /// Exceeds L3 (main memory).
    MainMemory,
}

impl CacheTier {
    /// Classify a working set size into a cache tier.
    #[must_use]
    pub fn classify(size_bytes: usize) -> Self {
        if size_bytes <= L1_CACHE_SIZE {
            CacheTier::L1
        } else if size_bytes <= L2_CACHE_SIZE {
            CacheTier::L2
        } else if size_bytes <= L3_CACHE_SIZE {
            CacheTier::L3
        } else {
            CacheTier::MainMemory
        }
    }
}

/// Scan order for block coefficient access.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum ScanOrder {
    /// Row-major (raster) scan.
    RowMajor,
    /// Column-major scan.
    ColumnMajor,
    /// Diagonal (zigzag) scan.
    Diagonal,
    /// Morton (Z-curve) order for spatial locality.
    Morton,
}

/// Working set estimation for a video frame or block region.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct WorkingSetEstimate {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// Bytes per pixel.
    pub bytes_per_pixel: usize,
    /// Number of reference frames in the working set.
    pub ref_frames: usize,
    /// Total estimated working set size (bytes).
    pub total_bytes: usize,
    /// Recommended cache tier.
    pub tier: CacheTier,
}

impl WorkingSetEstimate {
    /// Estimate working set for encoding a region.
    #[must_use]
    pub fn new(width: usize, height: usize, bytes_per_pixel: usize, ref_frames: usize) -> Self {
        // Current block + ref frames + motion search overhead (3x current block)
        let current = width * height * bytes_per_pixel;
        let refs = current * ref_frames;
        let overhead = current * 3;
        let total_bytes = current + refs + overhead;
        let tier = CacheTier::classify(total_bytes);
        Self {
            width,
            height,
            bytes_per_pixel,
            ref_frames,
            total_bytes,
            tier,
        }
    }

    /// True if the working set fits in L2 or better.
    #[must_use]
    pub fn is_cache_hot(&self) -> bool {
        matches!(self.tier, CacheTier::L1 | CacheTier::L2)
    }
}

/// Prefetch descriptor: address offset and distance (in cache lines).
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct PrefetchHint {
    /// Byte offset from the base pointer to prefetch.
    pub offset: usize,
    /// Prefetch distance in cache lines.
    pub distance: usize,
    /// Whether this is a write prefetch.
    pub write: bool,
}

/// Generate prefetch hints for linear scan of a block row.
///
/// Returns one `PrefetchHint` per row to prefetch ahead.
#[must_use]
pub fn generate_row_prefetch(
    width: usize,
    bytes_per_pixel: usize,
    prefetch_distance: usize,
) -> Vec<PrefetchHint> {
    let row_bytes = width * bytes_per_pixel;
    let cache_line = 64usize;
    (0..prefetch_distance)
        .map(|d| PrefetchHint {
            offset: (d + 1) * row_bytes,
            distance: row_bytes.div_ceil(cache_line),
            write: false,
        })
        .collect()
}

/// Generate a Morton (Z-curve) scan order for an N×N block.
///
/// Returns pixel indices in Morton order.
#[must_use]
pub fn morton_scan_order(n: usize) -> Vec<usize> {
    let mut result = Vec::with_capacity(n * n);
    for y in 0..n {
        for x in 0..n {
            let code = morton_encode(x, y);
            result.push(code);
        }
    }
    // Sort by Morton code to get scan order
    let mut pairs: Vec<(usize, usize)> = result
        .iter()
        .enumerate()
        .map(|(i, &code)| (code, i))
        .collect();
    pairs.sort_unstable_by_key(|&(code, _)| code);
    pairs.iter().map(|&(_, pixel_idx)| pixel_idx).collect()
}

/// Interleave bits of x and y to produce a Morton code.
#[must_use]
pub fn morton_encode(x: usize, y: usize) -> usize {
    let mut result = 0usize;
    for i in 0..16 {
        result |= ((x >> i) & 1) << (2 * i);
        result |= ((y >> i) & 1) << (2 * i + 1);
    }
    result
}

/// Cache-locality score for a given scan order on a block.
///
/// Higher score = better spatial locality (fewer cache misses expected).
/// Based on average stride between consecutive accesses.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn locality_score(order: ScanOrder, width: usize, height: usize) -> f64 {
    match order {
        ScanOrder::RowMajor => {
            // Best: stride = 1
            1.0
        }
        ScanOrder::ColumnMajor => {
            // Worst: stride = width
            1.0 / width as f64
        }
        ScanOrder::Diagonal => {
            // Average stride ≈ sqrt(width)
            1.0 / (width as f64).sqrt()
        }
        ScanOrder::Morton => {
            // Morton: average stride ≈ 2 for small blocks
            1.0 / (1.0 + (height as f64).log2().max(1.0) / 4.0)
        }
    }
}

/// Recommendation: best scan order for a given block and cache level.
#[must_use]
pub fn recommend_scan_order(width: usize, height: usize, tier: CacheTier) -> ScanOrder {
    match tier {
        CacheTier::L1 => {
            // Block fits in L1: row-major is optimal
            ScanOrder::RowMajor
        }
        CacheTier::L2 => {
            // Moderate: Morton gives better 2D locality
            if width == height && width.is_power_of_two() {
                ScanOrder::Morton
            } else {
                ScanOrder::RowMajor
            }
        }
        CacheTier::L3 | CacheTier::MainMemory => {
            // Larger: Morton helps reduce cache line reuse distance
            ScanOrder::Morton
        }
    }
}

/// Cache optimization advisor for encoder configuration.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct CacheOptAdvisor {
    l1_size: usize,
    l2_size: usize,
    l3_size: usize,
}

impl Default for CacheOptAdvisor {
    fn default() -> Self {
        Self {
            l1_size: L1_CACHE_SIZE,
            l2_size: L2_CACHE_SIZE,
            l3_size: L3_CACHE_SIZE,
        }
    }
}

impl CacheOptAdvisor {
    /// Create a new advisor with custom cache sizes.
    #[must_use]
    pub fn new(l1_size: usize, l2_size: usize, l3_size: usize) -> Self {
        Self {
            l1_size,
            l2_size,
            l3_size,
        }
    }

    /// Advise on the maximum reference frame count that keeps the working set in L2.
    #[must_use]
    pub fn max_refs_for_l2(&self, block_width: usize, block_height: usize, bpp: usize) -> usize {
        let block_bytes = block_width * block_height * bpp;
        if block_bytes == 0 {
            return 0;
        }
        // L2 size minus current block and overhead (4x block)
        let budget = self.l2_size.saturating_sub(block_bytes * 4);
        budget / block_bytes
    }

    /// Estimate cache miss rate (0.0–1.0) for a given working set and tier.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn miss_rate_estimate(&self, working_set: &WorkingSetEstimate) -> f64 {
        match working_set.tier {
            CacheTier::L1 => 0.01,
            CacheTier::L2 => 0.05,
            CacheTier::L3 => 0.15,
            CacheTier::MainMemory => {
                // Scale with how much it overflows L3
                let overflow = working_set.total_bytes.saturating_sub(self.l3_size);
                (0.3 + (overflow as f64 / self.l3_size as f64) * 0.5).min(0.95)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_tier_classify_l1() {
        assert_eq!(CacheTier::classify(1024), CacheTier::L1);
    }

    #[test]
    fn test_cache_tier_classify_l2() {
        assert_eq!(CacheTier::classify(100 * 1024), CacheTier::L2);
    }

    #[test]
    fn test_cache_tier_classify_l3() {
        assert_eq!(CacheTier::classify(2 * 1024 * 1024), CacheTier::L3);
    }

    #[test]
    fn test_cache_tier_classify_main_memory() {
        assert_eq!(CacheTier::classify(64 * 1024 * 1024), CacheTier::MainMemory);
    }

    #[test]
    fn test_working_set_estimate_small_block() {
        let ws = WorkingSetEstimate::new(8, 8, 1, 2);
        assert!(ws.is_cache_hot(), "8x8 block with 2 refs should be L1/L2");
    }

    #[test]
    fn test_working_set_estimate_large_frame() {
        let ws = WorkingSetEstimate::new(1920, 1080, 3, 4);
        assert_eq!(ws.tier, CacheTier::MainMemory);
        assert!(!ws.is_cache_hot());
    }

    #[test]
    fn test_working_set_total_bytes() {
        let ws = WorkingSetEstimate::new(16, 16, 1, 1);
        // current=256, refs=256, overhead=768 → total=1280
        assert_eq!(ws.total_bytes, 1280);
    }

    #[test]
    fn test_generate_row_prefetch_count() {
        let hints = generate_row_prefetch(64, 1, 4);
        assert_eq!(hints.len(), 4);
    }

    #[test]
    fn test_generate_row_prefetch_offsets_increasing() {
        let hints = generate_row_prefetch(64, 1, 3);
        for i in 1..hints.len() {
            assert!(hints[i].offset > hints[i - 1].offset);
        }
    }

    #[test]
    fn test_morton_encode_origin() {
        assert_eq!(morton_encode(0, 0), 0);
    }

    #[test]
    fn test_morton_encode_x1() {
        // x=1, y=0 → bit 0 of x goes to bit 0 of code
        assert_eq!(morton_encode(1, 0), 1);
    }

    #[test]
    fn test_morton_encode_y1() {
        // x=0, y=1 → bit 0 of y goes to bit 1 of code
        assert_eq!(morton_encode(0, 1), 2);
    }

    #[test]
    fn test_morton_scan_order_size() {
        let order = morton_scan_order(4);
        assert_eq!(order.len(), 16);
    }

    #[test]
    fn test_morton_scan_order_all_indices() {
        let order = morton_scan_order(4);
        let mut sorted = order.clone();
        sorted.sort_unstable();
        // All 16 indices should be present (0..15)
        for (i, &v) in sorted.iter().enumerate() {
            assert_eq!(v, i);
        }
    }

    #[test]
    fn test_locality_score_row_major_best() {
        let row = locality_score(ScanOrder::RowMajor, 64, 64);
        let col = locality_score(ScanOrder::ColumnMajor, 64, 64);
        assert!(
            row > col,
            "Row-major should have better locality than column-major"
        );
    }

    #[test]
    fn test_recommend_scan_order_l1() {
        let order = recommend_scan_order(8, 8, CacheTier::L1);
        assert_eq!(order, ScanOrder::RowMajor);
    }

    #[test]
    fn test_recommend_scan_order_l2_power_of_two() {
        let order = recommend_scan_order(16, 16, CacheTier::L2);
        assert_eq!(order, ScanOrder::Morton);
    }

    #[test]
    fn test_advisor_max_refs_for_l2() {
        let advisor = CacheOptAdvisor::default();
        let max_refs = advisor.max_refs_for_l2(16, 16, 1);
        assert!(max_refs > 0);
        // 16*16*1=256 bytes, budget = 256k - 256*4 = ~261k, 261k/256 ≈ 1020
        assert!(max_refs < 2000);
    }

    #[test]
    fn test_advisor_miss_rate_l1() {
        let advisor = CacheOptAdvisor::default();
        let ws = WorkingSetEstimate::new(8, 8, 1, 0);
        let rate = advisor.miss_rate_estimate(&ws);
        assert!((rate - 0.01).abs() < 1e-9);
    }

    #[test]
    fn test_advisor_miss_rate_main_memory() {
        let advisor = CacheOptAdvisor::default();
        let ws = WorkingSetEstimate::new(1920, 1080, 3, 4);
        let rate = advisor.miss_rate_estimate(&ws);
        assert!(rate > 0.1);
    }
}
