//! Frame and scene luminance statistics for MaxRGB / percentile CLL auto-detect.
//!
//! Implements the ITU-R BT.2100 / SMPTE ST 2086 MaxCLL and MaxFALL measurement
//! pipeline over sequences of HDR frames.  Both PQ-encoded (SMPTE ST 2084) and
//! linear-light (nit-valued) inputs are supported.
//!
//! # Quick start
//! ```rust,ignore
//! use oximedia_hdr::luminance_stats::ContentLuminanceAnalyzer;
//!
//! let mut analyzer = ContentLuminanceAnalyzer::new();
//! // frame: Vec<f32> of interleaved RGB pixels, PQ-encoded [0, 1]
//! let stats = analyzer.analyze_frame_pq(&frame, width, height)?;
//! println!("MaxCLL = {} nit, FALL = {} nit", stats.max_cll_nits, stats.max_fall_nits);
//! let (scene_cll, scene_fall) = analyzer.scene_stats();
//! ```

use crate::transfer_function::pq_eotf;
use crate::{HdrError, Result};

// ── FrameLuminanceStats ───────────────────────────────────────────────────────

/// Per-frame luminance statistics.
///
/// All luminance values are in absolute nits (cd/m²).
#[derive(Debug, Clone)]
pub struct FrameLuminanceStats {
    /// Maximum Content Light Level (MaxCLL) — brightest pixel's ITU-R BT.2100
    /// luminance (`Y = 0.2627 R + 0.6780 G + 0.0593 B`) in this frame.
    pub max_cll_nits: f32,
    /// Maximum Frame-Average Light Level (MaxFALL) — mean pixel luminance
    /// across the frame in nits.
    pub max_fall_nits: f32,
    /// Maximum MaxRGB — maximum of `max(R, G, B)` taken over every pixel.
    ///
    /// Unlike `max_cll_nits` (which uses a weighted luminance formula), this
    /// value is the highest single-channel value observed in the frame.
    pub max_rgb_nits: f32,
    /// 95th-percentile pixel luminance in nits.
    pub percentile_95_nits: f32,
    /// 99th-percentile pixel luminance in nits.
    pub percentile_99_nits: f32,
    /// Geometric mean pixel luminance in nits.
    ///
    /// Computed as `exp(mean(ln(Y + ε))) − ε` where `ε = 1e-6`.  Useful for
    /// perceptual exposure estimation.
    pub geometric_mean_nits: f32,
    /// Number of pixels in the analysed frame (`width × height`).
    pub pixel_count: u64,
}

// ── ContentLuminanceAnalyzer ──────────────────────────────────────────────────

/// Multi-frame accumulator for MaxCLL / MaxFALL scene statistics.
///
/// Call [`analyze_frame_pq`] or [`analyze_frame_linear`] once per frame; then
/// call [`scene_stats`] to retrieve the running maximum CLL and FALL across all
/// frames seen so far.
///
/// [`analyze_frame_pq`]: ContentLuminanceAnalyzer::analyze_frame_pq
/// [`analyze_frame_linear`]: ContentLuminanceAnalyzer::analyze_frame_linear
/// [`scene_stats`]: ContentLuminanceAnalyzer::scene_stats
#[derive(Debug, Clone)]
pub struct ContentLuminanceAnalyzer {
    /// Running maximum MaxCLL across all analysed frames (nits).
    pub scene_cll_nits: f32,
    /// Running maximum MaxFALL across all analysed frames (nits).
    pub scene_fall_nits: f32,
    /// Number of frames processed so far.
    pub frame_count: u64,
}

impl Default for ContentLuminanceAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl ContentLuminanceAnalyzer {
    /// Create a new analyser with all accumulators zeroed.
    pub fn new() -> Self {
        Self {
            scene_cll_nits: 0.0,
            scene_fall_nits: 0.0,
            frame_count: 0,
        }
    }

    /// Analyse a frame encoded with the SMPTE ST 2084 (PQ) transfer function.
    ///
    /// `frame` must be interleaved RGB in display-order, with each sample in
    /// `[0.0, 1.0]` representing the normalised PQ code value.
    ///
    /// Internally, each PQ sample is decoded via [`pq_eotf`] and multiplied by
    /// 10 000 to obtain absolute nits before the luminance statistics are
    /// computed identically to `analyze_frame_linear`.
    ///
    /// # Errors
    /// Returns [`HdrError::ToneMappingError`] if `frame.len() ≠ width × height × 3`.
    /// Returns [`HdrError::InvalidLuminance`] if any PQ sample is outside `[0, 1]`.
    pub fn analyze_frame_pq(
        &mut self,
        frame: &[f32],
        width: u32,
        height: u32,
    ) -> Result<FrameLuminanceStats> {
        let expected = (width as usize)
            .checked_mul(height as usize)
            .and_then(|n| n.checked_mul(3))
            .ok_or_else(|| {
                HdrError::ToneMappingError("frame dimensions overflow usize".to_string())
            })?;

        if frame.len() != expected {
            return Err(HdrError::ToneMappingError(format!(
                "PQ frame length mismatch: expected {expected} ({}×{}×3), got {}",
                width,
                height,
                frame.len()
            )));
        }

        // Decode each PQ sample to linear-light nits.
        let mut linear: Vec<f32> = Vec::with_capacity(frame.len());
        for &pq_sample in frame {
            let linear_norm = pq_eotf(f64::from(pq_sample))?;
            // pq_eotf returns [0, 1] where 1.0 ≡ 10 000 nits
            linear.push((linear_norm as f32) * 10_000.0);
        }

        self.analyze_frame_linear(&linear, width, height)
    }

    /// Analyse a frame in linear light (absolute nits).
    ///
    /// `frame` must be interleaved RGB in display-order, with each sample in
    /// absolute nits (cd/m²).
    ///
    /// # Errors
    /// Returns [`HdrError::ToneMappingError`] if `frame.len() ≠ width × height × 3`.
    pub fn analyze_frame_linear(
        &mut self,
        frame: &[f32],
        width: u32,
        height: u32,
    ) -> Result<FrameLuminanceStats> {
        let pixel_count = (width as u64).checked_mul(height as u64).ok_or_else(|| {
            HdrError::ToneMappingError("frame dimensions overflow u64".to_string())
        })?;

        let expected = pixel_count
            .checked_mul(3)
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| {
                HdrError::ToneMappingError("frame sample count overflows usize".to_string())
            })?;

        if frame.len() != expected {
            return Err(HdrError::ToneMappingError(format!(
                "linear frame length mismatch: expected {expected} ({}×{}×3), got {}",
                width,
                height,
                frame.len()
            )));
        }

        // Handle degenerate empty frame.
        if pixel_count == 0 {
            let stats = FrameLuminanceStats {
                max_cll_nits: 0.0,
                max_fall_nits: 0.0,
                max_rgb_nits: 0.0,
                percentile_95_nits: 0.0,
                percentile_99_nits: 0.0,
                geometric_mean_nits: 0.0,
                pixel_count: 0,
            };
            self.frame_count += 1;
            return Ok(stats);
        }

        let n = pixel_count as usize;

        // Per-pixel pass: collect luminance, max_rgb, and log-luminance sum.
        let mut luminances: Vec<f32> = Vec::with_capacity(n);
        let mut max_rgb_nits: f32 = 0.0;
        let mut log_sum: f64 = 0.0;

        // BT.2100 luminance coefficients.
        const KR: f32 = 0.2627;
        const KG: f32 = 0.6780;
        const KB: f32 = 0.0593;
        const EPSILON: f32 = 1e-6;
        const EPSILON_F64: f64 = 1e-6;

        for i in 0..n {
            let base = i * 3;
            let r = frame[base].max(0.0);
            let g = frame[base + 1].max(0.0);
            let b = frame[base + 2].max(0.0);

            let lum = KR * r + KG * g + KB * b;
            let max_rgb = r.max(g).max(b);

            luminances.push(lum);
            if max_rgb > max_rgb_nits {
                max_rgb_nits = max_rgb;
            }
            log_sum += f64::from(lum + EPSILON).ln();
        }

        // Sort luminance values for percentile extraction.
        luminances.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let max_cll_nits = *luminances.last().unwrap_or(&0.0);

        // Mean (FALL).
        let sum: f64 = luminances.iter().map(|&v| f64::from(v)).sum();
        let max_fall_nits = (sum / n as f64) as f32;

        // Percentile indices (floor).
        let idx_95 = ((n as f64) * 0.95) as usize;
        let idx_99 = ((n as f64) * 0.99) as usize;
        let percentile_95_nits = luminances[idx_95.min(n - 1)];
        let percentile_99_nits = luminances[idx_99.min(n - 1)];

        // Geometric mean.
        let geo_mean_raw = (log_sum / n as f64).exp() - EPSILON_F64;
        let geometric_mean_nits = (geo_mean_raw as f32).max(0.0);

        // Update scene-level accumulators.
        if max_cll_nits > self.scene_cll_nits {
            self.scene_cll_nits = max_cll_nits;
        }
        if max_fall_nits > self.scene_fall_nits {
            self.scene_fall_nits = max_fall_nits;
        }
        self.frame_count += 1;

        Ok(FrameLuminanceStats {
            max_cll_nits,
            max_fall_nits,
            max_rgb_nits,
            percentile_95_nits,
            percentile_99_nits,
            geometric_mean_nits,
            pixel_count,
        })
    }

    /// Return `(scene_cll_nits, scene_fall_nits)` — the running maxima across
    /// all frames analysed since this analyser was constructed.
    pub fn scene_stats(&self) -> (f32, f32) {
        (self.scene_cll_nits, self.scene_fall_nits)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transfer_function::pq_oetf;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    // Helper: build a uniform 1×1 linear frame.
    fn uniform_pixel(r: f32, g: f32, b: f32) -> Vec<f32> {
        vec![r, g, b]
    }

    // 1. 1×1 black frame returns all-zero stats.
    #[test]
    fn test_black_frame_is_zero() {
        let mut analyzer = ContentLuminanceAnalyzer::new();
        let frame = uniform_pixel(0.0, 0.0, 0.0);
        let stats = analyzer
            .analyze_frame_linear(&frame, 1, 1)
            .expect("analyze");
        assert!(approx(stats.max_cll_nits, 0.0, 1e-5));
        assert!(approx(stats.max_fall_nits, 0.0, 1e-5));
        assert!(approx(stats.max_rgb_nits, 0.0, 1e-5));
        assert_eq!(stats.pixel_count, 1);
    }

    // 2. 1×1 white frame at 10 000 nits — MaxCLL should equal 10 000.
    #[test]
    fn test_white_frame_max_cll() {
        let mut analyzer = ContentLuminanceAnalyzer::new();
        // White: R=G=B=10000 nit; luminance = (0.2627+0.6780+0.0593)*10000 = 10000.
        let frame = uniform_pixel(10_000.0, 10_000.0, 10_000.0);
        let stats = analyzer
            .analyze_frame_linear(&frame, 1, 1)
            .expect("analyze");
        assert!(
            approx(stats.max_cll_nits, 10_000.0, 1.0),
            "max_cll = {}",
            stats.max_cll_nits
        );
        assert!(approx(stats.max_rgb_nits, 10_000.0, 1.0));
    }

    // 3. 2×2 mixed frame: verify max_cll, max_fall, max_rgb.
    #[test]
    fn test_2x2_mixed_frame() {
        let mut analyzer = ContentLuminanceAnalyzer::new();
        // 4 pixels: (1000,0,0), (0,1000,0), (0,0,1000), (500,500,500)
        // Luminances: 0.2627*1000=262.7, 0.6780*1000=678.0, 0.0593*1000=59.3, (0.2627+0.6780+0.0593)*500=500.0
        // max_cll = 678.0, mean = (262.7+678.0+59.3+500.0)/4 = 375.0
        // max_rgb = 1000.0
        let frame = vec![
            1000.0_f32, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 1000.0, 500.0, 500.0, 500.0,
        ];
        let stats = analyzer
            .analyze_frame_linear(&frame, 2, 2)
            .expect("analyze");
        assert!(
            approx(stats.max_cll_nits, 678.0, 1.0),
            "max_cll = {}",
            stats.max_cll_nits
        );
        assert!(
            approx(stats.max_fall_nits, 375.0, 1.0),
            "fall = {}",
            stats.max_fall_nits
        );
        assert!(
            approx(stats.max_rgb_nits, 1000.0, 1.0),
            "max_rgb = {}",
            stats.max_rgb_nits
        );
        assert_eq!(stats.pixel_count, 4);
    }

    // 4. scene_stats accumulates correctly across multiple frames.
    #[test]
    fn test_scene_stats_accumulation() {
        let mut analyzer = ContentLuminanceAnalyzer::new();

        let frame1 = uniform_pixel(100.0, 100.0, 100.0); // lum = 100
        let frame2 = uniform_pixel(500.0, 500.0, 500.0); // lum = 500

        analyzer
            .analyze_frame_linear(&frame1, 1, 1)
            .expect("frame1");
        analyzer
            .analyze_frame_linear(&frame2, 1, 1)
            .expect("frame2");

        let (cll, fall) = analyzer.scene_stats();
        assert!(approx(cll, 500.0, 1.0), "scene_cll = {cll}");
        // Both frames' FALL == their max; max across frames is 500.
        assert!(approx(fall, 500.0, 1.0), "scene_fall = {fall}");
        assert_eq!(analyzer.frame_count, 2);
    }

    // 5. analyze_frame_pq decodes PQ correctly.
    #[test]
    fn test_pq_frame_decodes_correctly() {
        let mut analyzer = ContentLuminanceAnalyzer::new();

        // Encode 100 nit white as a PQ value.
        // pq_oetf takes linear normalised to [0, 1] where 1 = 10000 nits.
        let pq_val = pq_oetf(100.0 / 10_000.0).expect("pq_oetf") as f32;
        let frame = vec![pq_val, pq_val, pq_val];
        let stats = analyzer.analyze_frame_pq(&frame, 1, 1).expect("pq_analyze");

        // After decoding, lum ≈ 100 nit (within ±2 nit due to float rounding).
        assert!(
            approx(stats.max_cll_nits, 100.0, 2.0),
            "expected ~100 nit, got {}",
            stats.max_cll_nits
        );
    }

    // 6. Mismatched frame size returns error.
    #[test]
    fn test_mismatched_frame_size_returns_error() {
        let mut analyzer = ContentLuminanceAnalyzer::new();
        let frame = vec![0.0_f32; 6]; // only 2 pixels
        let result = analyzer.analyze_frame_linear(&frame, 2, 2); // expects 4 pixels = 12 samples
        assert!(result.is_err());
    }

    // 7. Mismatched PQ frame size returns error.
    #[test]
    fn test_mismatched_pq_frame_size_returns_error() {
        let mut analyzer = ContentLuminanceAnalyzer::new();
        let frame = vec![0.5_f32; 3]; // 1 pixel but expects 4
        let result = analyzer.analyze_frame_pq(&frame, 2, 2);
        assert!(result.is_err());
    }

    // 8. percentile_99 >= percentile_95 >= 0.
    #[test]
    fn test_percentile_ordering() {
        let mut analyzer = ContentLuminanceAnalyzer::new();
        // Build a 10×1 frame with increasing luminance values.
        let mut frame = Vec::with_capacity(30);
        for i in 0..10_u32 {
            let v = i as f32 * 100.0;
            frame.extend_from_slice(&[v, v, v]);
        }
        let stats = analyzer
            .analyze_frame_linear(&frame, 10, 1)
            .expect("analyze");
        assert!(stats.percentile_99_nits >= stats.percentile_95_nits);
        assert!(stats.percentile_95_nits >= 0.0);
    }

    // 9. geometric_mean is between 0 and max_cll.
    #[test]
    fn test_geometric_mean_bounds() {
        let mut analyzer = ContentLuminanceAnalyzer::new();
        let frame = vec![100.0_f32, 200.0, 50.0]; // 1 pixel
        let stats = analyzer
            .analyze_frame_linear(&frame, 1, 1)
            .expect("analyze");
        assert!(stats.geometric_mean_nits >= 0.0);
        assert!(stats.geometric_mean_nits <= stats.max_cll_nits + 1.0);
    }

    // 10. frame_count increments on each call.
    #[test]
    fn test_frame_count_increments() {
        let mut analyzer = ContentLuminanceAnalyzer::new();
        assert_eq!(analyzer.frame_count, 0);
        let frame = uniform_pixel(50.0, 50.0, 50.0);
        analyzer.analyze_frame_linear(&frame, 1, 1).expect("f1");
        assert_eq!(analyzer.frame_count, 1);
        analyzer.analyze_frame_linear(&frame, 1, 1).expect("f2");
        assert_eq!(analyzer.frame_count, 2);
    }

    // 11. All-equal luminance frame: mean == max == all percentiles (within eps).
    #[test]
    fn test_uniform_frame_stats() {
        let mut analyzer = ContentLuminanceAnalyzer::new();
        let lum = 300.0_f32;
        // 4 pixels, all at `lum` nit white.
        let mut frame = Vec::with_capacity(12);
        for _ in 0..4 {
            frame.extend_from_slice(&[lum, lum, lum]);
        }
        let stats = analyzer
            .analyze_frame_linear(&frame, 2, 2)
            .expect("analyze");
        assert!(approx(stats.max_cll_nits, lum, 0.1));
        assert!(approx(stats.max_fall_nits, lum, 0.1));
        assert!(approx(stats.percentile_95_nits, lum, 0.1));
        assert!(approx(stats.percentile_99_nits, lum, 0.1));
    }

    // 12. Default impl matches new().
    #[test]
    fn test_default_impl() {
        let a = ContentLuminanceAnalyzer::new();
        let b = ContentLuminanceAnalyzer::default();
        assert_eq!(a.frame_count, b.frame_count);
        assert!(approx(a.scene_cll_nits, b.scene_cll_nits, 1e-9));
        assert!(approx(a.scene_fall_nits, b.scene_fall_nits, 1e-9));
    }
}
