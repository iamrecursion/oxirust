//! Color histogram shot similarity module.
//!
//! Computes per-channel (R, G, B) histograms from video frames and provides
//! distance metrics for comparing shot appearance: chi-squared distance,
//! Bhattacharyya distance, and histogram intersection similarity.
//!
//! The input `yuv` slice is interpreted as follows:
//! - If `len == width * height * 3`: treated as packed RGB (R, G, B per pixel).
//! - If `len == width * height`: treated as luma-only (Y); all three channel
//!   histograms are populated from the single luminance value (useful for
//!   grayscale comparisons where color is not available).
//! - Any other length returns a zero histogram to avoid panics.

#![allow(dead_code)]

/// A normalized per-channel color histogram.
///
/// Each channel bin vector is normalised so that the sum of all bins equals 1.0
/// (probability distribution).  This makes distance metrics independent of
/// frame resolution.
#[derive(Debug, Clone)]
pub struct ColorHistogram {
    /// Normalized histogram for the red channel.
    pub bins_r: Vec<f32>,
    /// Normalized histogram for the green channel.
    pub bins_g: Vec<f32>,
    /// Normalized histogram for the blue channel.
    pub bins_b: Vec<f32>,
    /// Number of bins per channel.
    pub num_bins: usize,
}

impl ColorHistogram {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Create a zeroed histogram with `num_bins` bins per channel.
    #[must_use]
    pub fn zeros(num_bins: usize) -> Self {
        let n = num_bins.max(1);
        Self {
            bins_r: vec![0.0_f32; n],
            bins_g: vec![0.0_f32; n],
            bins_b: vec![0.0_f32; n],
            num_bins: n,
        }
    }

    /// Build a colour histogram from a raw frame buffer.
    ///
    /// `yuv` may be either:
    /// - packed RGB (len = `width * height * 3`) — channel order R, G, B.
    /// - luma-only (len = `width * height`) — all channels receive the same
    ///   luma value.
    ///
    /// If the slice length does not match either expectation the function
    /// returns a zeroed histogram rather than panicking.
    #[must_use]
    pub fn from_frame(yuv: &[u8], width: u32, height: u32, num_bins: usize) -> Self {
        let n = num_bins.max(1);
        let pixels = (width as usize) * (height as usize);

        let mut hist_r = vec![0_u64; n];
        let mut hist_g = vec![0_u64; n];
        let mut hist_b = vec![0_u64; n];
        let mut count = 0_u64;

        if yuv.len() == pixels * 3 {
            // Packed RGB
            for chunk in yuv.chunks_exact(3) {
                let r = chunk[0];
                let g = chunk[1];
                let b = chunk[2];
                let bi_r = Self::bin_index(r, n);
                let bi_g = Self::bin_index(g, n);
                let bi_b = Self::bin_index(b, n);
                hist_r[bi_r] += 1;
                hist_g[bi_g] += 1;
                hist_b[bi_b] += 1;
                count += 1;
            }
        } else if yuv.len() == pixels {
            // Luma-only: use Y for all three channels
            for &y in yuv {
                let bi = Self::bin_index(y, n);
                hist_r[bi] += 1;
                hist_g[bi] += 1;
                hist_b[bi] += 1;
                count += 1;
            }
        } else {
            // Unrecognised layout — return zeroed histogram
            return Self::zeros(n);
        }

        if count == 0 {
            return Self::zeros(n);
        }

        let norm = 1.0_f32 / count as f32;
        Self {
            bins_r: hist_r.iter().map(|&v| v as f32 * norm).collect(),
            bins_g: hist_g.iter().map(|&v| v as f32 * norm).collect(),
            bins_b: hist_b.iter().map(|&v| v as f32 * norm).collect(),
            num_bins: n,
        }
    }

    // -----------------------------------------------------------------------
    // Distance / similarity metrics
    // -----------------------------------------------------------------------

    /// Chi-squared distance between two histograms (sum over channels).
    ///
    /// χ²(H₁,H₂) = Σ_i (H₁ᵢ − H₂ᵢ)² / (H₁ᵢ + H₂ᵢ),  sum skipping zero denominators.
    /// Returned value is in `[0, ∞)`.  Lower means more similar.
    #[must_use]
    pub fn chi_squared_distance(&self, other: &ColorHistogram) -> f32 {
        let n = self.num_bins.min(other.num_bins);
        chi_squared_channel(&self.bins_r[..n], &other.bins_r[..n])
            + chi_squared_channel(&self.bins_g[..n], &other.bins_g[..n])
            + chi_squared_channel(&self.bins_b[..n], &other.bins_b[..n])
    }

    /// Bhattacharyya distance between two histograms (averaged over channels).
    ///
    /// D_B = −ln( Σ_i sqrt(H₁ᵢ · H₂ᵢ) ), averaged over R, G, B channels.
    /// Returned value is in `[0, ∞)`.  Lower means more similar.
    #[must_use]
    pub fn bhattacharyya_distance(&self, other: &ColorHistogram) -> f32 {
        let n = self.num_bins.min(other.num_bins);
        let d_r = bhattacharyya_channel(&self.bins_r[..n], &other.bins_r[..n]);
        let d_g = bhattacharyya_channel(&self.bins_g[..n], &other.bins_g[..n]);
        let d_b = bhattacharyya_channel(&self.bins_b[..n], &other.bins_b[..n]);
        (d_r + d_g + d_b) / 3.0
    }

    /// Histogram intersection similarity (averaged over R, G, B channels).
    ///
    /// sim = Σ_i min(H₁ᵢ, H₂ᵢ), averaged over channels.
    /// Returned value is in `[0, 1]`.  Higher means more similar.
    #[must_use]
    pub fn intersection_similarity(&self, other: &ColorHistogram) -> f32 {
        let n = self.num_bins.min(other.num_bins);
        let s_r = intersection_channel(&self.bins_r[..n], &other.bins_r[..n]);
        let s_g = intersection_channel(&self.bins_g[..n], &other.bins_g[..n]);
        let s_b = intersection_channel(&self.bins_b[..n], &other.bins_b[..n]);
        (s_r + s_g + s_b) / 3.0
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Map a byte value `[0, 255]` to a bin index `[0, num_bins)`.
    #[inline]
    fn bin_index(value: u8, num_bins: usize) -> usize {
        let idx = (value as usize * num_bins) / 256;
        idx.min(num_bins - 1)
    }
}

// ---------------------------------------------------------------------------
// Per-channel distance helpers (free functions)
// ---------------------------------------------------------------------------

fn chi_squared_channel(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| {
            let denom = ai + bi;
            if denom < f32::EPSILON {
                0.0
            } else {
                let diff = ai - bi;
                (diff * diff) / denom
            }
        })
        .sum()
}

fn bhattacharyya_channel(a: &[f32], b: &[f32]) -> f32 {
    let bc: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| (ai * bi).sqrt())
        .sum();
    // Clamp to avoid NaN from log(0)
    let bc_clamped = bc.clamp(f32::EPSILON, 1.0);
    -bc_clamped.ln()
}

fn intersection_channel(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&ai, &bi)| ai.min(bi)).sum()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Helper: create a solid-color RGB frame
    // ------------------------------------------------------------------
    fn solid_rgb(r: u8, g: u8, b: u8, w: u32, h: u32) -> Vec<u8> {
        let pixels = (w * h) as usize;
        let mut buf = Vec::with_capacity(pixels * 3);
        for _ in 0..pixels {
            buf.push(r);
            buf.push(g);
            buf.push(b);
        }
        buf
    }

    // ------------------------------------------------------------------
    // Helper: create a luma-only frame
    // ------------------------------------------------------------------
    fn solid_luma(y: u8, w: u32, h: u32) -> Vec<u8> {
        vec![y; (w * h) as usize]
    }

    // ------------------------------------------------------------------
    // Construction tests
    // ------------------------------------------------------------------

    #[test]
    fn test_zeros_has_correct_bins() {
        let h = ColorHistogram::zeros(16);
        assert_eq!(h.num_bins, 16);
        assert_eq!(h.bins_r.len(), 16);
        assert_eq!(h.bins_g.len(), 16);
        assert_eq!(h.bins_b.len(), 16);
        assert!(h.bins_r.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_from_frame_rgb_normalized() {
        let frame = solid_rgb(200, 100, 50, 8, 8);
        let h = ColorHistogram::from_frame(&frame, 8, 8, 16);
        let sum_r: f32 = h.bins_r.iter().sum();
        let sum_g: f32 = h.bins_g.iter().sum();
        let sum_b: f32 = h.bins_b.iter().sum();
        assert!((sum_r - 1.0).abs() < 1e-5, "R not normalised: {sum_r}");
        assert!((sum_g - 1.0).abs() < 1e-5, "G not normalised: {sum_g}");
        assert!((sum_b - 1.0).abs() < 1e-5, "B not normalised: {sum_b}");
    }

    #[test]
    fn test_from_frame_luma_normalized() {
        let frame = solid_luma(128, 4, 4);
        let h = ColorHistogram::from_frame(&frame, 4, 4, 8);
        let sum_r: f32 = h.bins_r.iter().sum();
        assert!((sum_r - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_from_frame_invalid_length_returns_zeros() {
        let bad = vec![0u8; 7]; // not 4*4 or 4*4*3
        let h = ColorHistogram::from_frame(&bad, 4, 4, 8);
        assert!(h.bins_r.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_from_frame_single_bin() {
        let frame = solid_rgb(10, 20, 30, 2, 2);
        let h = ColorHistogram::from_frame(&frame, 2, 2, 1);
        assert_eq!(h.num_bins, 1);
        assert!((h.bins_r[0] - 1.0).abs() < 1e-5);
    }

    // ------------------------------------------------------------------
    // Chi-squared distance tests
    // ------------------------------------------------------------------

    #[test]
    fn test_chi_squared_identical_is_zero() {
        let frame = solid_rgb(100, 150, 200, 4, 4);
        let h = ColorHistogram::from_frame(&frame, 4, 4, 16);
        let dist = h.chi_squared_distance(&h);
        assert!(
            dist < 1e-5,
            "identical histograms should have distance 0, got {dist}"
        );
    }

    #[test]
    fn test_chi_squared_different_is_positive() {
        let f1 = solid_rgb(10, 10, 10, 4, 4);
        let f2 = solid_rgb(240, 240, 240, 4, 4);
        let h1 = ColorHistogram::from_frame(&f1, 4, 4, 16);
        let h2 = ColorHistogram::from_frame(&f2, 4, 4, 16);
        let dist = h1.chi_squared_distance(&h2);
        assert!(
            dist > 0.5,
            "very different frames should have large distance, got {dist}"
        );
    }

    #[test]
    fn test_chi_squared_symmetric() {
        let f1 = solid_rgb(80, 120, 200, 4, 4);
        let f2 = solid_rgb(200, 60, 30, 4, 4);
        let h1 = ColorHistogram::from_frame(&f1, 4, 4, 16);
        let h2 = ColorHistogram::from_frame(&f2, 4, 4, 16);
        let d12 = h1.chi_squared_distance(&h2);
        let d21 = h2.chi_squared_distance(&h1);
        assert!((d12 - d21).abs() < 1e-5, "chi-squared must be symmetric");
    }

    // ------------------------------------------------------------------
    // Bhattacharyya distance tests
    // ------------------------------------------------------------------

    #[test]
    fn test_bhattacharyya_identical_near_zero() {
        let frame = solid_rgb(100, 150, 200, 4, 4);
        let h = ColorHistogram::from_frame(&frame, 4, 4, 16);
        let dist = h.bhattacharyya_distance(&h);
        assert!(
            dist < 1e-4,
            "identical histograms should have near-zero Bhattacharyya distance, got {dist}"
        );
    }

    #[test]
    fn test_bhattacharyya_different_is_positive() {
        let f1 = solid_rgb(10, 10, 10, 4, 4);
        let f2 = solid_rgb(240, 240, 240, 4, 4);
        let h1 = ColorHistogram::from_frame(&f1, 4, 4, 16);
        let h2 = ColorHistogram::from_frame(&f2, 4, 4, 16);
        let dist = h1.bhattacharyya_distance(&h2);
        assert!(
            dist > 0.1,
            "different frames should have positive Bhattacharyya distance, got {dist}"
        );
    }

    #[test]
    fn test_bhattacharyya_symmetric() {
        let f1 = solid_rgb(80, 120, 200, 4, 4);
        let f2 = solid_rgb(200, 60, 30, 4, 4);
        let h1 = ColorHistogram::from_frame(&f1, 4, 4, 16);
        let h2 = ColorHistogram::from_frame(&f2, 4, 4, 16);
        let d12 = h1.bhattacharyya_distance(&h2);
        let d21 = h2.bhattacharyya_distance(&h1);
        assert!((d12 - d21).abs() < 1e-5, "Bhattacharyya must be symmetric");
    }

    // ------------------------------------------------------------------
    // Intersection similarity tests
    // ------------------------------------------------------------------

    #[test]
    fn test_intersection_identical_is_one() {
        let frame = solid_rgb(100, 150, 200, 4, 4);
        let h = ColorHistogram::from_frame(&frame, 4, 4, 16);
        let sim = h.intersection_similarity(&h);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "identical histograms should have similarity 1.0, got {sim}"
        );
    }

    #[test]
    fn test_intersection_different_less_than_one() {
        let f1 = solid_rgb(10, 10, 10, 4, 4);
        let f2 = solid_rgb(240, 240, 240, 4, 4);
        let h1 = ColorHistogram::from_frame(&f1, 4, 4, 16);
        let h2 = ColorHistogram::from_frame(&f2, 4, 4, 16);
        let sim = h1.intersection_similarity(&h2);
        assert!(
            sim < 0.5,
            "very different frames should have low similarity, got {sim}"
        );
    }

    #[test]
    fn test_intersection_in_range() {
        let f1 = solid_rgb(128, 64, 192, 8, 8);
        let f2 = solid_rgb(130, 60, 190, 8, 8);
        let h1 = ColorHistogram::from_frame(&f1, 8, 8, 32);
        let h2 = ColorHistogram::from_frame(&f2, 8, 8, 32);
        let sim = h1.intersection_similarity(&h2);
        assert!(sim >= 0.0 && sim <= 1.0, "similarity out of range: {sim}");
    }

    // ------------------------------------------------------------------
    // Cross-metric consistency
    // ------------------------------------------------------------------

    #[test]
    fn test_similar_frames_have_low_distance() {
        // Two frames that differ only slightly in all channels
        let f1 = solid_rgb(128, 128, 128, 8, 8);
        let f2 = solid_rgb(130, 130, 130, 8, 8);
        let h1 = ColorHistogram::from_frame(&f1, 8, 8, 32);
        let h2 = ColorHistogram::from_frame(&f2, 8, 8, 32);
        let chi = h1.chi_squared_distance(&h2);
        let bha = h1.bhattacharyya_distance(&h2);
        let inter = h1.intersection_similarity(&h2);
        // With 32 bins and a 2-unit shift, adjacent or same bins → very similar
        assert!(chi < 2.0, "similar frames: chi={chi}");
        assert!(bha < 0.5, "similar frames: bha={bha}");
        assert!(inter > 0.5, "similar frames: inter={inter}");
    }

    #[test]
    fn test_bin_index_boundary() {
        // value=0 → bin 0; value=255 → last bin
        let idx_0 = ColorHistogram::bin_index(0, 16);
        let idx_255 = ColorHistogram::bin_index(255, 16);
        assert_eq!(idx_0, 0);
        assert_eq!(idx_255, 15);
    }

    #[test]
    fn test_zeros_chi_squared_zero_against_zero() {
        let h1 = ColorHistogram::zeros(8);
        let h2 = ColorHistogram::zeros(8);
        let dist = h1.chi_squared_distance(&h2);
        assert!((dist).abs() < 1e-5);
    }
}
