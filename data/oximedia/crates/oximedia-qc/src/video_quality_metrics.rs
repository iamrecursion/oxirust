//! Video quality measurement algorithms.
//!
//! Provides pure-Rust implementations of common objective video quality metrics:
//! - **MSE / PSNR** – Mean Squared Error and Peak Signal-to-Noise Ratio
//! - **SSIM patch** – Structural Similarity Index on an 8-bit patch
//! - **Blockiness** – DCT-block edge artefact score

// ── MSE / PSNR ────────────────────────────────────────────────────────────────

/// Computes Mean Squared Error between two equal-length byte slices.
///
/// Returns 0.0 if either slice is empty or they differ in length.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn compute_mse(original: &[u8], compressed: &[u8]) -> f64 {
    if original.is_empty() || original.len() != compressed.len() {
        return 0.0;
    }
    let sum: f64 = original
        .iter()
        .zip(compressed.iter())
        .map(|(&a, &b)| {
            let diff = f64::from(a) - f64::from(b);
            diff * diff
        })
        .sum();
    sum / original.len() as f64
}

/// Computes Peak Signal-to-Noise Ratio from MSE.
///
/// Formula: `10 * log10(max_val² / mse)`.
///
/// Returns `f64::INFINITY` when `mse == 0.0` (lossless).
#[must_use]
pub fn compute_psnr(mse: f64, max_val: f64) -> f64 {
    if mse == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (max_val * max_val / mse).log10()
}

/// Per-component PSNR result for a YUV frame.
#[derive(Debug, Clone, PartialEq)]
pub struct PsnrResult {
    /// PSNR of the luma (Y) plane in dB.
    pub y_psnr: f64,
    /// PSNR of the blue-difference chroma (U/Cb) plane in dB.
    pub u_psnr: f64,
    /// PSNR of the red-difference chroma (V/Cr) plane in dB.
    pub v_psnr: f64,
    /// Weighted average PSNR (6:1:1 weighting matching YUV 4:2:0 area).
    pub psnr_avg: f64,
}

impl PsnrResult {
    /// Creates a new PSNR result.
    #[must_use]
    pub fn new(y_psnr: f64, u_psnr: f64, v_psnr: f64) -> Self {
        // Weighted average: Y contributes 4× more area in 4:2:0
        let avg = (4.0 * y_psnr + u_psnr + v_psnr) / 6.0;
        Self {
            y_psnr,
            u_psnr,
            v_psnr,
            psnr_avg: avg,
        }
    }

    /// Returns `true` if the average PSNR meets or exceeds `threshold` dB.
    #[must_use]
    pub fn is_acceptable(&self, threshold: f64) -> bool {
        self.psnr_avg >= threshold
    }
}

// ── SSIM patch ────────────────────────────────────────────────────────────────

/// Computes the Structural Similarity Index (SSIM) between two patches.
///
/// Both slices must be equal-length. Returns 0.0 for empty or mismatched input.
///
/// Stabilisation constants: C₁ = (0.01 × 255)², C₂ = (0.03 × 255)².
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn compute_ssim_patch(patch1: &[f32], patch2: &[f32]) -> f64 {
    if patch1.is_empty() || patch1.len() != patch2.len() {
        return 0.0;
    }

    let n = patch1.len() as f64;
    let mu1: f64 = patch1.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
    let mu2: f64 = patch2.iter().map(|&v| f64::from(v)).sum::<f64>() / n;

    let var1: f64 = patch1
        .iter()
        .map(|&v| {
            let d = f64::from(v) - mu1;
            d * d
        })
        .sum::<f64>()
        / n;
    let var2: f64 = patch2
        .iter()
        .map(|&v| {
            let d = f64::from(v) - mu2;
            d * d
        })
        .sum::<f64>()
        / n;

    let cov: f64 = patch1
        .iter()
        .zip(patch2.iter())
        .map(|(&a, &b)| (f64::from(a) - mu1) * (f64::from(b) - mu2))
        .sum::<f64>()
        / n;

    // SSIM stabilisation constants for 8-bit images (L = 255)
    const C1: f64 = (0.01 * 255.0) * (0.01 * 255.0);
    const C2: f64 = (0.03 * 255.0) * (0.03 * 255.0);

    let num = (2.0 * mu1 * mu2 + C1) * (2.0 * cov + C2);
    let den = (mu1 * mu1 + mu2 * mu2 + C1) * (var1 + var2 + C2);

    if den == 0.0 {
        return 1.0;
    }
    num / den
}

/// Result of an SSIM computation over a frame.
#[derive(Debug, Clone, PartialEq)]
pub struct SsimResult {
    /// Mean SSIM value over all evaluated patches (0.0–1.0).
    pub value: f64,
    /// Number of patches that were evaluated.
    pub map_size: usize,
}

impl SsimResult {
    /// Creates a new SSIM result.
    #[must_use]
    pub fn new(value: f64, map_size: usize) -> Self {
        Self { value, map_size }
    }

    /// Returns `true` if the SSIM value is above the "good quality" threshold (0.95).
    #[must_use]
    pub fn is_good(&self) -> bool {
        self.value > 0.95
    }
}

// ── Blockiness ────────────────────────────────────────────────────────────────

/// Blockiness score for a video frame, indicating DCT-block edge artefacts.
#[derive(Debug, Clone, PartialEq)]
pub struct BlockinessScore {
    /// Mean absolute difference across block boundaries (higher = more blocking).
    pub score: f32,
    /// Block size used during analysis (e.g., 8 for DCT-8).
    pub block_size: u32,
}

impl BlockinessScore {
    /// Creates a new blockiness score.
    #[must_use]
    pub fn new(score: f32, block_size: u32) -> Self {
        Self { score, block_size }
    }

    /// Returns `true` if the blockiness score exceeds the significance threshold (5.0).
    #[must_use]
    pub fn is_significant(&self) -> bool {
        self.score > 5.0
    }
}

/// Computes a blockiness score for a luma plane stored as a flat byte slice.
///
/// The metric is the mean absolute difference between adjacent rows/columns at
/// every DCT-block boundary. A higher value indicates more visible blocking.
///
/// # Arguments
/// * `pixels` – Row-major luma plane (length must be a multiple of `width`).
/// * `width`  – Scanline width in pixels.
/// * `block_size` – Block size to test for boundaries (typically 8 or 16).
///
/// Returns a zero score when the input is empty or `width` is 0.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn compute_blockiness(pixels: &[u8], width: usize, block_size: u32) -> BlockinessScore {
    if pixels.is_empty() || width == 0 || block_size == 0 {
        return BlockinessScore::new(0.0, block_size);
    }

    let height = pixels.len() / width;
    if height == 0 {
        return BlockinessScore::new(0.0, block_size);
    }

    let bs = block_size as usize;
    let mut total_diff: f64 = 0.0;
    let mut count: usize = 0;

    // Horizontal block boundaries: difference between row y and row y-1
    // at each multiple-of-bs row.
    let mut row = bs;
    while row < height {
        for col in 0..width {
            let above = pixels[(row - 1) * width + col] as f64;
            let below = pixels[row * width + col] as f64;
            total_diff += (below - above).abs();
            count += 1;
        }
        row += bs;
    }

    // Vertical block boundaries: difference between col x and col x-1
    // at each multiple-of-bs column.
    let mut col = bs;
    while col < width {
        for r in 0..height {
            let left = pixels[r * width + (col - 1)] as f64;
            let right = pixels[r * width + col] as f64;
            total_diff += (right - left).abs();
            count += 1;
        }
        col += bs;
    }

    let score = if count > 0 {
        (total_diff / count as f64) as f32
    } else {
        0.0
    };

    BlockinessScore::new(score, block_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── compute_mse ───────────────────────────────────────────────────────────

    #[test]
    fn test_mse_identical_slices() {
        let data: Vec<u8> = (0..16).collect();
        let mse = compute_mse(&data, &data);
        assert_eq!(mse, 0.0);
    }

    #[test]
    fn test_mse_known_value() {
        let orig = vec![0u8, 0, 0, 0];
        let comp = vec![2u8, 2, 2, 2];
        // MSE = (4 + 4 + 4 + 4) / 4 = 4.0
        let mse = compute_mse(&orig, &comp);
        assert!((mse - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_mse_empty_returns_zero() {
        assert_eq!(compute_mse(&[], &[]), 0.0);
    }

    #[test]
    fn test_mse_mismatched_lengths_returns_zero() {
        assert_eq!(compute_mse(&[1, 2], &[1, 2, 3]), 0.0);
    }

    // ── compute_psnr ─────────────────────────────────────────────────────────

    #[test]
    fn test_psnr_zero_mse_is_infinity() {
        assert!(compute_psnr(0.0, 255.0).is_infinite());
    }

    #[test]
    fn test_psnr_known_value() {
        // MSE=4, max=255 → PSNR = 10*log10(255²/4) = 10*log10(65025/4)
        let psnr = compute_psnr(4.0, 255.0);
        let expected = 10.0 * (255.0_f64 * 255.0 / 4.0).log10();
        assert!((psnr - expected).abs() < 1e-9);
    }

    // ── PsnrResult ────────────────────────────────────────────────────────────

    #[test]
    fn test_psnr_result_average_weight() {
        let r = PsnrResult::new(40.0, 40.0, 40.0);
        assert!((r.psnr_avg - 40.0).abs() < 1e-9);
    }

    #[test]
    fn test_psnr_result_is_acceptable_true() {
        let r = PsnrResult::new(45.0, 42.0, 42.0);
        assert!(r.is_acceptable(30.0));
    }

    #[test]
    fn test_psnr_result_is_acceptable_false() {
        let r = PsnrResult::new(20.0, 18.0, 18.0);
        assert!(!r.is_acceptable(30.0));
    }

    // ── compute_ssim_patch ────────────────────────────────────────────────────

    #[test]
    fn test_ssim_identical_patches() {
        let patch: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let ssim = compute_ssim_patch(&patch, &patch);
        assert!((ssim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_ssim_empty_returns_zero() {
        assert_eq!(compute_ssim_patch(&[], &[]), 0.0);
    }

    #[test]
    fn test_ssim_mismatched_returns_zero() {
        assert_eq!(compute_ssim_patch(&[1.0, 2.0], &[1.0, 2.0, 3.0]), 0.0);
    }

    #[test]
    fn test_ssim_range_bounds() {
        let p1: Vec<f32> = vec![0.0; 64];
        let p2: Vec<f32> = vec![255.0; 64];
        let ssim = compute_ssim_patch(&p1, &p2);
        assert!(ssim >= -1.0 && ssim <= 1.0);
    }

    // ── SsimResult ────────────────────────────────────────────────────────────

    #[test]
    fn test_ssim_result_is_good_true() {
        let r = SsimResult::new(0.97, 100);
        assert!(r.is_good());
    }

    #[test]
    fn test_ssim_result_is_good_false() {
        let r = SsimResult::new(0.90, 100);
        assert!(!r.is_good());
    }

    // ── compute_blockiness ────────────────────────────────────────────────────

    #[test]
    fn test_blockiness_empty_returns_zero() {
        let bs = compute_blockiness(&[], 8, 8);
        assert_eq!(bs.score, 0.0);
    }

    #[test]
    fn test_blockiness_uniform_frame_is_zero() {
        // A flat grey frame: no differences at boundaries
        let pixels = vec![128u8; 64 * 64];
        let bs = compute_blockiness(&pixels, 64, 8);
        assert_eq!(bs.score, 0.0);
        assert!(!bs.is_significant());
    }

    #[test]
    fn test_blockiness_score_stored() {
        let pixels = vec![128u8; 64 * 64];
        let bs = compute_blockiness(&pixels, 64, 8);
        assert_eq!(bs.block_size, 8);
    }

    #[test]
    fn test_blockiness_is_significant_false_below_threshold() {
        let bs = BlockinessScore::new(3.0, 8);
        assert!(!bs.is_significant());
    }

    #[test]
    fn test_blockiness_is_significant_true_above_threshold() {
        let bs = BlockinessScore::new(10.0, 8);
        assert!(bs.is_significant());
    }
}

// ── SIMD pixel analysis ────────────────────────────────────────────────────

/// Statistics from a SIMD-accelerated pixel range check.
///
/// Returned by [`simd_luma_range_check`] and [`simd_chroma_range_check`].
#[derive(Debug, Clone, PartialEq)]
pub struct PixelRangeStats {
    /// Minimum pixel value observed.
    pub min_val: u8,
    /// Maximum pixel value observed.
    pub max_val: u8,
    /// Number of pixels below the legal minimum.
    pub below_min_count: usize,
    /// Number of pixels above the legal maximum.
    pub above_max_count: usize,
    /// Total number of pixels analysed.
    pub total_pixels: usize,
}

impl PixelRangeStats {
    /// Returns `true` if all pixels are within the legal range.
    #[must_use]
    pub fn is_legal(&self) -> bool {
        self.below_min_count == 0 && self.above_max_count == 0
    }

    /// Fraction of pixels that fall outside the legal range (0.0–1.0).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn violation_ratio(&self) -> f64 {
        if self.total_pixels == 0 {
            return 0.0;
        }
        let violations = self.below_min_count + self.above_max_count;
        violations as f64 / self.total_pixels as f64
    }
}

/// Checks luma (Y) pixel values against the broadcast-legal 8-bit range [16, 235].
///
/// Uses SSE 4.1 SIMD intrinsics when the CPU supports it, falling back to a
/// portable scalar path otherwise. The detection is performed at runtime via
/// `is_x86_feature_detected!("sse4.1")`.
///
/// Returns [`PixelRangeStats`] with violation counts and min/max values.
#[must_use]
pub fn simd_luma_range_check(pixels: &[u8]) -> PixelRangeStats {
    const LEGAL_MIN: u8 = 16;
    const LEGAL_MAX: u8 = 235;
    simd_range_check_inner(pixels, LEGAL_MIN, LEGAL_MAX)
}

/// Checks chroma (Cb/Cr) pixel values against the broadcast-legal 8-bit range [16, 240].
///
/// Uses SSE 4.1 SIMD intrinsics when the CPU supports it, falling back to a
/// portable scalar path otherwise. The detection is performed at runtime via
/// `is_x86_feature_detected!("sse4.1")`.
///
/// Returns [`PixelRangeStats`] with violation counts and min/max values.
#[must_use]
pub fn simd_chroma_range_check(pixels: &[u8]) -> PixelRangeStats {
    const LEGAL_MIN: u8 = 16;
    const LEGAL_MAX: u8 = 240;
    simd_range_check_inner(pixels, LEGAL_MIN, LEGAL_MAX)
}

/// Internal dispatcher: selects SSE 4.1 or scalar path at runtime.
fn simd_range_check_inner(pixels: &[u8], legal_min: u8, legal_max: u8) -> PixelRangeStats {
    if pixels.is_empty() {
        return PixelRangeStats {
            min_val: 0,
            max_val: 0,
            below_min_count: 0,
            above_max_count: 0,
            total_pixels: 0,
        };
    }

    // Runtime CPU feature detection: prefer SSE 4.1 SIMD on x86/x86_64.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("sse4.1") {
            // SAFETY: We verified SSE 4.1 availability at runtime.
            #[allow(unsafe_code)]
            return unsafe { simd_range_check_sse41(pixels, legal_min, legal_max) };
        }
    }

    simd_range_check_scalar(pixels, legal_min, legal_max)
}

/// SSE 4.1-accelerated pixel range check.
///
/// Processes 16 bytes per iteration using `_mm_min_epu8` / `_mm_max_epu8`
/// to accumulate running min/max, then counts violations in the tail with the
/// scalar path.
///
/// # Safety
///
/// Caller must ensure SSE 4.1 is available (verified via
/// `is_x86_feature_detected!`).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse4.1")]
#[allow(unsafe_code)]
// `_mm_loadu_si128` / `_mm_storeu_si128` are the *unaligned* SSE2 variants and
// explicitly handle any pointer alignment in hardware; the cast_ptr_alignment lint
// is a false positive here.
#[allow(clippy::cast_ptr_alignment)]
unsafe fn simd_range_check_sse41(pixels: &[u8], legal_min: u8, legal_max: u8) -> PixelRangeStats {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = pixels.len();
    let chunks = len / 16;
    let remainder_start = chunks * 16;

    // Initialise running SIMD min/max accumulators.
    let mut running_min = _mm_set1_epi8(u8::MAX as i8);
    let mut running_max = _mm_set1_epi8(0_i8);

    for chunk in 0..chunks {
        let ptr = pixels.as_ptr().add(chunk * 16);
        // SAFETY: ptr is within bounds (chunks * 16 <= len).
        let v = _mm_loadu_si128(ptr.cast::<__m128i>());
        running_min = _mm_min_epu8(running_min, v);
        running_max = _mm_max_epu8(running_max, v);
    }

    // Reduce SIMD accumulators to scalar min/max.
    let mut min_bytes = [0u8; 16];
    let mut max_bytes = [0u8; 16];
    _mm_storeu_si128(min_bytes.as_mut_ptr().cast::<__m128i>(), running_min);
    _mm_storeu_si128(max_bytes.as_mut_ptr().cast::<__m128i>(), running_max);

    let simd_min = min_bytes.iter().copied().fold(u8::MAX, u8::min);
    let simd_max = max_bytes.iter().copied().fold(0u8, u8::max);

    // Count violations in the SIMD-processed chunk range using the scalar helper,
    // then combine with the remainder.
    let simd_stats = simd_range_check_scalar(&pixels[..remainder_start], legal_min, legal_max);
    let tail_stats = simd_range_check_scalar(&pixels[remainder_start..], legal_min, legal_max);

    // Use SIMD-computed global min/max (more accurate over all 16-byte chunks).
    let global_min = simd_min
        .min(if remainder_start < len {
            tail_stats.min_val
        } else {
            u8::MAX
        })
        .min(simd_min);
    let global_max = simd_max.max(tail_stats.max_val);

    PixelRangeStats {
        min_val: global_min,
        max_val: global_max,
        below_min_count: simd_stats.below_min_count + tail_stats.below_min_count,
        above_max_count: simd_stats.above_max_count + tail_stats.above_max_count,
        total_pixels: len,
    }
}

/// Portable scalar pixel range check — used as fallback and for remainder bytes.
#[allow(clippy::cast_precision_loss)]
fn simd_range_check_scalar(pixels: &[u8], legal_min: u8, legal_max: u8) -> PixelRangeStats {
    if pixels.is_empty() {
        return PixelRangeStats {
            min_val: 0,
            max_val: 0,
            below_min_count: 0,
            above_max_count: 0,
            total_pixels: 0,
        };
    }

    let mut min_val = u8::MAX;
    let mut max_val = 0u8;
    let mut below_min_count = 0usize;
    let mut above_max_count = 0usize;

    for &p in pixels {
        if p < min_val {
            min_val = p;
        }
        if p > max_val {
            max_val = p;
        }
        if p < legal_min {
            below_min_count += 1;
        } else if p > legal_max {
            above_max_count += 1;
        }
    }

    PixelRangeStats {
        min_val,
        max_val,
        below_min_count,
        above_max_count,
        total_pixels: pixels.len(),
    }
}

#[cfg(test)]
mod simd_tests {
    use super::*;

    #[test]
    fn test_simd_luma_all_legal() {
        // All values in legal luma range [16, 235]
        let pixels: Vec<u8> = (16..=235u8).collect();
        let stats = simd_luma_range_check(&pixels);
        assert!(stats.is_legal());
        assert_eq!(stats.below_min_count, 0);
        assert_eq!(stats.above_max_count, 0);
        assert_eq!(stats.min_val, 16);
        assert_eq!(stats.max_val, 235);
    }

    #[test]
    fn test_simd_luma_below_min() {
        let pixels = vec![0u8, 8, 15, 16, 128, 235];
        let stats = simd_luma_range_check(&pixels);
        assert!(!stats.is_legal());
        assert_eq!(stats.below_min_count, 3); // 0, 8, 15
        assert_eq!(stats.above_max_count, 0);
    }

    #[test]
    fn test_simd_luma_above_max() {
        let pixels = vec![16u8, 128, 235, 236, 250, 255];
        let stats = simd_luma_range_check(&pixels);
        assert!(!stats.is_legal());
        assert_eq!(stats.below_min_count, 0);
        assert_eq!(stats.above_max_count, 3); // 236, 250, 255
    }

    #[test]
    fn test_simd_chroma_legal_range() {
        // Chroma legal max is 240
        let pixels = vec![16u8, 128, 240];
        let stats = simd_chroma_range_check(&pixels);
        assert!(stats.is_legal());
    }

    #[test]
    fn test_simd_chroma_above_max_240() {
        let pixels = vec![16u8, 240, 241, 255];
        let stats = simd_chroma_range_check(&pixels);
        assert!(!stats.is_legal());
        assert_eq!(stats.above_max_count, 2); // 241, 255
    }

    #[test]
    fn test_simd_empty_input() {
        let stats = simd_luma_range_check(&[]);
        assert_eq!(stats.total_pixels, 0);
        assert!(stats.is_legal()); // vacuously true
    }

    #[test]
    fn test_simd_violation_ratio() {
        let pixels = vec![0u8, 255, 128, 128]; // 2 violations out of 4
        let stats = simd_luma_range_check(&pixels);
        let ratio = stats.violation_ratio();
        assert!((ratio - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_simd_luma_large_chunk_boundary() {
        // Test that 16-byte SIMD chunk boundaries work correctly
        // (17 pixels so we have one 16-byte SIMD chunk + 1 scalar remainder)
        let mut pixels = vec![128u8; 16]; // all legal
        pixels.push(0u8); // one illegal value in remainder
        let stats = simd_luma_range_check(&pixels);
        assert_eq!(stats.below_min_count, 1);
        assert_eq!(stats.total_pixels, 17);
    }

    #[test]
    fn test_simd_violation_ratio_zero_when_legal() {
        let pixels = vec![128u8; 32];
        let stats = simd_luma_range_check(&pixels);
        assert_eq!(stats.violation_ratio(), 0.0);
    }
}
