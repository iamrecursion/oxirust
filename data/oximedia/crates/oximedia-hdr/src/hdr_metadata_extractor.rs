//! MaxCLL / MaxFALL metadata extraction from HDR frame sequences.
//!
//! Provides [`HdrMetadataExtractor`] for computing SMPTE ST 2086 / CEA-861.3
//! Content Light Level metadata from decoded frame data.
//!
//! Both linear-light (normalised [0, 1]) and PQ-encoded inputs are supported.
//!
//! # Quick start
//!
//! ```rust,ignore
//! use oximedia_hdr::hdr_metadata_extractor::HdrMetadataExtractor;
//!
//! let extractor = HdrMetadataExtractor::new(); // 10 000 nit reference
//! let frames: &[&[f32]] = &[&frame1, &frame2]; // linear RGB, normalised to peak
//! let cll = extractor.extract_from_frames(frames);
//! println!("MaxCLL = {} nits, MaxFALL = {} nits", cll.max_cll, cll.max_fall);
//! ```

use crate::transfer_function::pq_eotf;
use crate::Result;

// ── ContentLightLevelInfo ─────────────────────────────────────────────────────

/// Content light level information derived from frame analysis.
///
/// Values are in nits (cd/m²), clamped to the `u16` range `[0, 65535]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContentLightLevelInfo {
    /// Maximum Content Light Level across all analysed frames (nits).
    pub max_cll: u16,
    /// Maximum Frame-Average Light Level across all analysed frames (nits).
    pub max_fall: u16,
}

// ── HdrMetadataExtractor ──────────────────────────────────────────────────────

/// Stateless extractor for MaxCLL / MaxFALL content light level metadata.
///
/// Instantiate with [`new`] (10 000 nit reference) or
/// [`with_peak_luminance`] for a custom reference.
///
/// [`new`]: HdrMetadataExtractor::new
/// [`with_peak_luminance`]: HdrMetadataExtractor::with_peak_luminance
#[derive(Debug, Clone)]
pub struct HdrMetadataExtractor {
    /// Reference peak luminance in nits.
    ///
    /// Linear-light frame samples are assumed to be normalised so that
    /// `1.0` corresponds to this luminance.
    peak_luminance: f32,
}

impl Default for HdrMetadataExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl HdrMetadataExtractor {
    /// Create an extractor with a 10 000 nit reference peak.
    pub fn new() -> Self {
        Self {
            peak_luminance: 10_000.0,
        }
    }

    /// Create an extractor with a custom reference peak luminance.
    pub fn with_peak_luminance(peak_luminance: f32) -> Self {
        Self { peak_luminance }
    }

    /// Extract MaxCLL / MaxFALL from a slice of linear-light RGB frames.
    ///
    /// Each frame in `frames` is an interleaved RGB slice (length must be a
    /// multiple of 3) with linear-light samples normalised to `[0.0, 1.0]`
    /// where `1.0` represents the configured `peak_luminance`.
    ///
    /// Frames whose length is not divisible by 3 or whose length is zero are
    /// silently skipped.
    ///
    /// The returned [`ContentLightLevelInfo`] values are in nits and clamped
    /// to `[0, 65535]`.
    pub fn extract_from_frames(&self, frames: &[&[f32]]) -> ContentLightLevelInfo {
        let mut scene_max_cll: f32 = 0.0;
        let mut scene_max_fall: f32 = 0.0;

        for frame in frames {
            if frame.is_empty() || frame.len() % 3 != 0 {
                continue;
            }
            let (frame_cll, frame_fall) = self.compute_frame_stats_linear(frame);
            if frame_cll > scene_max_cll {
                scene_max_cll = frame_cll;
            }
            if frame_fall > scene_max_fall {
                scene_max_fall = frame_fall;
            }
        }

        ContentLightLevelInfo {
            max_cll: nits_to_u16(scene_max_cll),
            max_fall: nits_to_u16(scene_max_fall),
        }
    }

    /// Extract MaxCLL / MaxFALL from PQ-encoded RGB frames.
    ///
    /// Each frame in `frames` is an interleaved RGB slice with PQ-encoded
    /// samples in `[0.0, 1.0]` (where `1.0` ≡ 10 000 nits per ST 2084).
    ///
    /// Frames whose length is not divisible by 3 or whose length is zero are
    /// silently skipped.
    ///
    /// # Errors
    ///
    /// Returns [`HdrError::InvalidLuminance`] if any PQ sample is outside
    /// `[0.0, 1.0]`.
    ///
    /// [`HdrError::InvalidLuminance`]: crate::HdrError::InvalidLuminance
    pub fn extract_from_frames_pq(&self, frames: &[&[f32]]) -> Result<ContentLightLevelInfo> {
        let mut scene_max_cll: f32 = 0.0;
        let mut scene_max_fall: f32 = 0.0;

        for frame in frames {
            if frame.is_empty() || frame.len() % 3 != 0 {
                continue;
            }

            // Decode PQ → linear nits.
            let mut linear: Vec<f32> = Vec::with_capacity(frame.len());
            for &pq_sample in *frame {
                let linear_norm = pq_eotf(f64::from(pq_sample))?;
                linear.push((linear_norm as f32) * 10_000.0);
            }

            // For PQ frames the linear values are already in absolute nits;
            // bypass peak-luminance scaling by creating a temporary extractor.
            let abs_extractor = HdrMetadataExtractor::with_peak_luminance(1.0);
            let (frame_cll, frame_fall) = abs_extractor.compute_frame_stats_linear(&linear);

            if frame_cll > scene_max_cll {
                scene_max_cll = frame_cll;
            }
            if frame_fall > scene_max_fall {
                scene_max_fall = frame_fall;
            }
        }

        Ok(ContentLightLevelInfo {
            max_cll: nits_to_u16(scene_max_cll),
            max_fall: nits_to_u16(scene_max_fall),
        })
    }

    /// Compute per-frame (MaxCLL, MaxFALL) for a linear-light frame.
    ///
    /// Returns `(max_cll_nits, max_fall_nits)`.  Sample values are multiplied
    /// by `peak_luminance` to obtain absolute nits before the BT.2100 luminance
    /// formula is applied.
    fn compute_frame_stats_linear(&self, frame: &[f32]) -> (f32, f32) {
        let n = frame.len() / 3;
        if n == 0 {
            return (0.0, 0.0);
        }

        // BT.2100 luminance coefficients.
        const KR: f32 = 0.2627;
        const KG: f32 = 0.6780;
        const KB: f32 = 0.0593;

        let peak = self.peak_luminance;
        let mut max_cll: f32 = 0.0;
        let mut sum_lum: f64 = 0.0;

        for i in 0..n {
            let base = i * 3;
            let r = frame[base].max(0.0) * peak;
            let g = frame[base + 1].max(0.0) * peak;
            let b = frame[base + 2].max(0.0) * peak;
            let lum = KR * r + KG * g + KB * b;
            if lum > max_cll {
                max_cll = lum;
            }
            sum_lum += f64::from(lum);
        }

        let max_fall = (sum_lum / n as f64) as f32;
        (max_cll, max_fall)
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Clamp a nit value to `u16` range.
fn nits_to_u16(nits: f32) -> u16 {
    nits.clamp(0.0, 65535.0) as u16
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transfer_function::pq_oetf;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    // 1. new() has peak_luminance = 10 000.
    #[test]
    fn test_new_default_peak() {
        let e = HdrMetadataExtractor::new();
        assert!(approx(e.peak_luminance, 10_000.0, f32::EPSILON));
    }

    // 2. Single all-1.0 frame → max_cll ≈ 10 000 nits.
    #[test]
    fn test_single_frame_white() {
        let e = HdrMetadataExtractor::new();
        // Luminance of white (R=G=B=1) = (0.2627+0.6780+0.0593) * 10 000 = 10 000.
        let frame = vec![1.0f32, 1.0, 1.0];
        let cll = e.extract_from_frames(&[&frame]);
        assert!(
            (cll.max_cll as i32 - 10_000).abs() <= 1,
            "max_cll should be ≈10000, got {}",
            cll.max_cll
        );
    }

    // 3. Single all-0.0 frame → max_cll = 0, max_fall = 0.
    #[test]
    fn test_single_frame_black() {
        let e = HdrMetadataExtractor::new();
        let frame = vec![0.0f32, 0.0, 0.0];
        let cll = e.extract_from_frames(&[&frame]);
        assert_eq!(cll.max_cll, 0);
        assert_eq!(cll.max_fall, 0);
    }

    // 4. max_cll across two frames returns the higher value.
    #[test]
    fn test_max_cll_across_frames() {
        let e = HdrMetadataExtractor::new();
        // frame1: peak ≈ 1000 nit white (R=G=B=0.1)
        let frame1 = vec![0.1f32, 0.1, 0.1];
        // frame2: peak ≈ 5000 nit white (R=G=B=0.5)
        let frame2 = vec![0.5f32, 0.5, 0.5];
        let cll = e.extract_from_frames(&[&frame1, &frame2]);
        let expected = (0.5 * 10_000.0) as u16; // 5000
        assert!(
            (cll.max_cll as i32 - expected as i32).abs() <= 5,
            "max_cll should be ≈{}, got {}",
            expected,
            cll.max_cll
        );
    }

    // 5. max_fall is the maximum of per-frame averages.
    #[test]
    fn test_max_fall_is_max_of_frame_averages() {
        let e = HdrMetadataExtractor::new();
        // frame1: 1 pixel at 0.01 → FALL ≈ 100 nit (lum component)
        let frame1 = vec![0.01f32, 0.01, 0.01];
        // frame2: 1 pixel at 0.08 → FALL ≈ 800 nit
        let frame2 = vec![0.08f32, 0.08, 0.08];
        let cll = e.extract_from_frames(&[&frame1, &frame2]);
        let expected_fall = (0.08 * 10_000.0) as u16; // 800
        assert!(
            (cll.max_fall as i32 - expected_fall as i32).abs() <= 5,
            "max_fall should be ≈{}, got {}",
            expected_fall,
            cll.max_fall
        );
    }

    // 6. Empty frames slice → max_cll = 0, max_fall = 0.
    #[test]
    fn test_empty_frames_slice() {
        let e = HdrMetadataExtractor::new();
        let cll = e.extract_from_frames(&[]);
        assert_eq!(cll.max_cll, 0);
        assert_eq!(cll.max_fall, 0);
    }

    // 7. Output clamped to u16 max when luminance > 65535.
    #[test]
    fn test_output_clamped_to_u16() {
        // Use a very high peak so luminance exceeds 65535 nits.
        let e = HdrMetadataExtractor::with_peak_luminance(100_000.0);
        let frame = vec![1.0f32, 1.0, 1.0]; // luminance = 100 000 nits
        let cll = e.extract_from_frames(&[&frame]);
        assert_eq!(cll.max_cll, 65535, "Should be clamped to u16::MAX");
    }

    // 8. Single red pixel: max_cll ≈ 0.2627 * 10 000 = 2627.
    #[test]
    fn test_single_pixel_red_only() {
        let e = HdrMetadataExtractor::new();
        let frame = vec![1.0f32, 0.0, 0.0];
        let cll = e.extract_from_frames(&[&frame]);
        let expected = (0.2627_f32 * 10_000.0) as u16; // 2627
        assert!(
            (cll.max_cll as i32 - expected as i32).abs() <= 2,
            "Red max_cll should be ≈{}, got {}",
            expected,
            cll.max_cll
        );
    }

    // 9. Single green pixel: max_cll ≈ 0.6780 * 10 000 = 6780.
    #[test]
    fn test_single_pixel_green_only() {
        let e = HdrMetadataExtractor::new();
        let frame = vec![0.0f32, 1.0, 0.0];
        let cll = e.extract_from_frames(&[&frame]);
        let expected = (0.6780_f32 * 10_000.0) as u16; // 6780
        assert!(
            (cll.max_cll as i32 - expected as i32).abs() <= 2,
            "Green max_cll should be ≈{}, got {}",
            expected,
            cll.max_cll
        );
    }

    // 10. Multiple frames: max_cll equals the brightest frame.
    #[test]
    fn test_multiple_frames_accumulate() {
        let e = HdrMetadataExtractor::new();
        let f1 = vec![0.1f32, 0.1, 0.1];
        let f2 = vec![0.3f32, 0.3, 0.3];
        let f3 = vec![0.8f32, 0.8, 0.8];
        let cll = e.extract_from_frames(&[&f1, &f2, &f3]);
        let expected = (0.8 * 10_000.0) as u16; // 8000
        assert!(
            (cll.max_cll as i32 - expected as i32).abs() <= 5,
            "max_cll should be ≈{}, got {}",
            expected,
            cll.max_cll
        );
    }

    // 11. extract_from_frames_pq decodes PQ correctly.
    #[test]
    fn test_pq_frame_decode_correct() {
        let e = HdrMetadataExtractor::new();
        // Encode 500 nit white as PQ value.
        let pq_val = pq_oetf(500.0 / 10_000.0).expect("pq_oetf") as f32;
        let frame = vec![pq_val, pq_val, pq_val];
        let cll = e.extract_from_frames_pq(&[&frame]).expect("pq extract");
        // lum = 500 nits, allow ±3 for float rounding.
        assert!(
            (cll.max_cll as i32 - 500).abs() <= 3,
            "PQ 500-nit: max_cll should be ≈500, got {}",
            cll.max_cll
        );
    }

    // 12. extract_from_frames_pq errors on out-of-range sample.
    #[test]
    fn test_pq_out_of_range_returns_error() {
        let e = HdrMetadataExtractor::new();
        let frame = vec![1.5f32, 0.5, 0.5]; // 1.5 is outside [0, 1]
        assert!(e.extract_from_frames_pq(&[&frame]).is_err());
    }

    // 13. Frames with wrong length (not divisible by 3) are skipped.
    #[test]
    fn test_invalid_frame_skipped() {
        let e = HdrMetadataExtractor::new();
        let bad_frame = vec![1.0f32, 1.0]; // length 2, not divisible by 3
        let good_frame = vec![0.1f32, 0.1, 0.1];
        let cll = e.extract_from_frames(&[&bad_frame, &good_frame]);
        // Should use only good_frame → ≈1000 nit.
        let expected = (0.1 * 10_000.0) as u16;
        assert!(
            (cll.max_cll as i32 - expected as i32).abs() <= 5,
            "Bad frame should be skipped; max_cll ≈{}, got {}",
            expected,
            cll.max_cll
        );
    }

    // 14. Default impl matches new().
    #[test]
    fn test_default_impl_matches_new() {
        let a = HdrMetadataExtractor::new();
        let b = HdrMetadataExtractor::default();
        assert!(approx(a.peak_luminance, b.peak_luminance, f32::EPSILON));
    }

    // 15. with_peak_luminance sets the correct value.
    #[test]
    fn test_with_peak_luminance() {
        let e = HdrMetadataExtractor::with_peak_luminance(4000.0);
        assert!(approx(e.peak_luminance, 4000.0, f32::EPSILON));
    }
}
