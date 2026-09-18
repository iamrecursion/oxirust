#![allow(dead_code)]
//! Color accuracy and fidelity metrics for video quality assessment.
//!
//! Measures color accuracy by comparing color channel statistics between
//! a reference and a distorted frame. Includes delta-E-like metrics in
//! approximate perceptual color space, hue shift detection, and saturation fidelity.

/// Result of a color fidelity analysis.
#[derive(Debug, Clone)]
pub struct ColorFidelityResult {
    /// Mean color difference (delta-E approximation in pseudo-Lab space).
    pub mean_delta_e: f64,
    /// Maximum color difference across all pixels.
    pub max_delta_e: f64,
    /// Hue shift score (0.0 = no shift, higher = more shift).
    pub hue_shift: f64,
    /// Saturation fidelity ratio (1.0 = identical, <1.0 = desaturated, >1.0 = oversaturated).
    pub saturation_ratio: f64,
    /// Luminance fidelity (mean absolute difference in luminance channel, 0-255 scale).
    pub luminance_mae: f64,
    /// Overall color fidelity score (0.0 = poor, 1.0 = perfect).
    pub fidelity_score: f64,
    /// Qualitative rating.
    pub rating: FidelityRating,
}

/// Qualitative color fidelity rating.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FidelityRating {
    /// Color reproduction is excellent.
    Excellent,
    /// Color reproduction is good.
    Good,
    /// Color reproduction is acceptable.
    Acceptable,
    /// Color reproduction is poor.
    Poor,
    /// Color reproduction is very poor.
    VeryPoor,
}

/// Configuration for color fidelity analysis.
#[derive(Debug, Clone)]
pub struct FidelityConfig {
    /// Maximum delta-E value used for normalization.
    pub delta_e_max: f64,
    /// Weight for luminance in combined score.
    pub weight_luminance: f64,
    /// Weight for chrominance in combined score.
    pub weight_chrominance: f64,
}

impl Default for FidelityConfig {
    fn default() -> Self {
        Self {
            delta_e_max: 50.0,
            weight_luminance: 0.5,
            weight_chrominance: 0.5,
        }
    }
}

impl FidelityConfig {
    /// Creates a new config with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the delta-E normalization ceiling.
    #[must_use]
    pub fn with_delta_e_max(mut self, v: f64) -> Self {
        self.delta_e_max = v;
        self
    }

    /// Sets the luminance weight.
    #[must_use]
    pub fn with_weight_luminance(mut self, w: f64) -> Self {
        self.weight_luminance = w;
        self
    }

    /// Sets the chrominance weight.
    #[must_use]
    pub fn with_weight_chrominance(mut self, w: f64) -> Self {
        self.weight_chrominance = w;
        self
    }
}

/// Color fidelity analyzer.
#[derive(Debug, Clone)]
pub struct ColorFidelityAnalyzer {
    /// Configuration.
    config: FidelityConfig,
}

impl ColorFidelityAnalyzer {
    /// Creates an analyzer with the given configuration.
    #[must_use]
    pub fn new(config: FidelityConfig) -> Self {
        Self { config }
    }

    /// Creates an analyzer with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self {
            config: FidelityConfig::default(),
        }
    }

    /// Compares color fidelity between a reference and distorted YCbCr frame.
    ///
    /// `ref_y`, `ref_cb`, `ref_cr` are the reference Y/Cb/Cr planes.
    /// `dist_y`, `dist_cb`, `dist_cr` are the distorted Y/Cb/Cr planes.
    /// All planes must have the same length (luma-sized; chroma is assumed co-sited).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn analyze_ycbcr(
        &self,
        ref_y: &[u8],
        ref_cb: &[u8],
        ref_cr: &[u8],
        dist_y: &[u8],
        dist_cb: &[u8],
        dist_cr: &[u8],
    ) -> ColorFidelityResult {
        let n = ref_y.len().min(ref_cb.len()).min(ref_cr.len());
        let n = n.min(dist_y.len()).min(dist_cb.len()).min(dist_cr.len());

        if n == 0 {
            return ColorFidelityResult {
                mean_delta_e: 0.0,
                max_delta_e: 0.0,
                hue_shift: 0.0,
                saturation_ratio: 1.0,
                luminance_mae: 0.0,
                fidelity_score: 1.0,
                rating: FidelityRating::Excellent,
            };
        }

        let mut sum_delta_e = 0.0f64;
        let mut max_delta_e = 0.0f64;
        let mut sum_luma_diff = 0.0f64;
        let mut sum_ref_sat = 0.0f64;
        let mut sum_dist_sat = 0.0f64;
        let mut sum_hue_diff = 0.0f64;

        for i in 0..n {
            // Compute pseudo-Lab delta-E using YCbCr directly
            let dy = ref_y[i] as f64 - dist_y[i] as f64;
            let dcb = ref_cb[i] as f64 - dist_cb[i] as f64;
            let dcr = ref_cr[i] as f64 - dist_cr[i] as f64;
            let delta_e = (dy * dy + dcb * dcb + dcr * dcr).sqrt();

            sum_delta_e += delta_e;
            if delta_e > max_delta_e {
                max_delta_e = delta_e;
            }

            sum_luma_diff += dy.abs();

            // Saturation approximation from Cb/Cr
            let ref_sat =
                ((ref_cb[i] as f64 - 128.0).powi(2) + (ref_cr[i] as f64 - 128.0).powi(2)).sqrt();
            let dist_sat =
                ((dist_cb[i] as f64 - 128.0).powi(2) + (dist_cr[i] as f64 - 128.0).powi(2)).sqrt();
            sum_ref_sat += ref_sat;
            sum_dist_sat += dist_sat;

            // Hue angle difference
            let ref_hue = (ref_cr[i] as f64 - 128.0).atan2(ref_cb[i] as f64 - 128.0);
            let dist_hue = (dist_cr[i] as f64 - 128.0).atan2(dist_cb[i] as f64 - 128.0);
            let mut hue_diff = (ref_hue - dist_hue).abs();
            if hue_diff > std::f64::consts::PI {
                hue_diff = 2.0 * std::f64::consts::PI - hue_diff;
            }
            sum_hue_diff += hue_diff;
        }

        let n_f = n as f64;
        let mean_delta_e = sum_delta_e / n_f;
        let luminance_mae = sum_luma_diff / n_f;
        let saturation_ratio = if sum_ref_sat > 1e-10 {
            sum_dist_sat / sum_ref_sat
        } else {
            1.0
        };
        let hue_shift = sum_hue_diff / n_f;

        // Compute fidelity score
        let luma_score = 1.0 - (luminance_mae / 255.0).min(1.0);
        let chroma_score = 1.0 - (mean_delta_e / self.config.delta_e_max).min(1.0);
        let fidelity_score = (self.config.weight_luminance * luma_score
            + self.config.weight_chrominance * chroma_score)
            .clamp(0.0, 1.0);

        let rating = if fidelity_score > 0.95 {
            FidelityRating::Excellent
        } else if fidelity_score > 0.85 {
            FidelityRating::Good
        } else if fidelity_score > 0.7 {
            FidelityRating::Acceptable
        } else if fidelity_score > 0.5 {
            FidelityRating::Poor
        } else {
            FidelityRating::VeryPoor
        };

        ColorFidelityResult {
            mean_delta_e,
            max_delta_e,
            hue_shift,
            saturation_ratio,
            luminance_mae,
            fidelity_score,
            rating,
        }
    }

    /// Analyzes color fidelity from separate RGB planes.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn analyze_rgb(
        &self,
        ref_r: &[u8],
        ref_g: &[u8],
        ref_b: &[u8],
        dist_r: &[u8],
        dist_g: &[u8],
        dist_b: &[u8],
    ) -> ColorFidelityResult {
        // Convert RGB to approximate YCbCr for analysis
        let n = ref_r
            .len()
            .min(ref_g.len())
            .min(ref_b.len())
            .min(dist_r.len())
            .min(dist_g.len())
            .min(dist_b.len());

        let mut ry = vec![0u8; n];
        let mut rcb = vec![0u8; n];
        let mut rcr = vec![0u8; n];
        let mut dy = vec![0u8; n];
        let mut dcb = vec![0u8; n];
        let mut dcr = vec![0u8; n];

        for i in 0..n {
            let (y, cb, cr) = rgb_to_ycbcr(ref_r[i], ref_g[i], ref_b[i]);
            ry[i] = y;
            rcb[i] = cb;
            rcr[i] = cr;

            let (y2, cb2, cr2) = rgb_to_ycbcr(dist_r[i], dist_g[i], dist_b[i]);
            dy[i] = y2;
            dcb[i] = cb2;
            dcr[i] = cr2;
        }

        self.analyze_ycbcr(&ry, &rcb, &rcr, &dy, &dcb, &dcr)
    }
}

/// Converts an RGB pixel to approximate YCbCr (BT.601).
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn rgb_to_ycbcr(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    let rf = r as f64;
    let gf = g as f64;
    let bf = b as f64;
    let y = (0.299 * rf + 0.587 * gf + 0.114 * bf).clamp(0.0, 255.0) as u8;
    let cb = (128.0 + (-0.168_736 * rf - 0.331_264 * gf + 0.5 * bf)).clamp(0.0, 255.0) as u8;
    let cr = (128.0 + (0.5 * rf - 0.418_688 * gf - 0.081_312 * bf)).clamp(0.0, 255.0) as u8;
    (y, cb, cr)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identical_planes(n: usize, val: u8) -> Vec<u8> {
        vec![val; n]
    }

    #[test]
    fn test_identical_frames_perfect_score() {
        let analyzer = ColorFidelityAnalyzer::with_defaults();
        let y = identical_planes(100, 128);
        let cb = identical_planes(100, 128);
        let cr = identical_planes(100, 128);
        let result = analyzer.analyze_ycbcr(&y, &cb, &cr, &y, &cb, &cr);
        assert!((result.mean_delta_e).abs() < 1e-10);
        assert!((result.fidelity_score - 1.0).abs() < 1e-10);
        assert_eq!(result.rating, FidelityRating::Excellent);
    }

    #[test]
    fn test_different_luma_reduces_score() {
        let analyzer = ColorFidelityAnalyzer::with_defaults();
        let ref_y = identical_planes(100, 128);
        let dist_y = identical_planes(100, 200);
        let cb = identical_planes(100, 128);
        let cr = identical_planes(100, 128);
        let result = analyzer.analyze_ycbcr(&ref_y, &cb, &cr, &dist_y, &cb, &cr);
        assert!(result.luminance_mae > 0.0);
        assert!(result.fidelity_score < 1.0);
    }

    #[test]
    fn test_different_chroma_reduces_score() {
        let analyzer = ColorFidelityAnalyzer::with_defaults();
        let y = identical_planes(100, 128);
        let ref_cb = identical_planes(100, 128);
        let dist_cb = identical_planes(100, 200);
        let cr = identical_planes(100, 128);
        let result = analyzer.analyze_ycbcr(&y, &ref_cb, &cr, &y, &dist_cb, &cr);
        assert!(result.mean_delta_e > 0.0);
        assert!(result.fidelity_score < 1.0);
    }

    #[test]
    fn test_empty_planes() {
        let analyzer = ColorFidelityAnalyzer::with_defaults();
        let empty: Vec<u8> = vec![];
        let result = analyzer.analyze_ycbcr(&empty, &empty, &empty, &empty, &empty, &empty);
        assert!((result.fidelity_score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_max_delta_e() {
        let analyzer = ColorFidelityAnalyzer::with_defaults();
        let ref_y = vec![0u8; 10];
        let dist_y = vec![255u8; 10];
        let ref_cb = vec![0u8; 10];
        let dist_cb = vec![255u8; 10];
        let ref_cr = vec![0u8; 10];
        let dist_cr = vec![255u8; 10];
        let result = analyzer.analyze_ycbcr(&ref_y, &ref_cb, &ref_cr, &dist_y, &dist_cb, &dist_cr);
        assert!(result.max_delta_e > 100.0);
    }

    #[test]
    fn test_saturation_ratio_identical() {
        let analyzer = ColorFidelityAnalyzer::with_defaults();
        let y = identical_planes(100, 128);
        let cb = identical_planes(100, 160);
        let cr = identical_planes(100, 160);
        let result = analyzer.analyze_ycbcr(&y, &cb, &cr, &y, &cb, &cr);
        assert!((result.saturation_ratio - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_hue_shift_zero_for_identical() {
        let analyzer = ColorFidelityAnalyzer::with_defaults();
        let y = identical_planes(50, 128);
        let cb = identical_planes(50, 150);
        let cr = identical_planes(50, 150);
        let result = analyzer.analyze_ycbcr(&y, &cb, &cr, &y, &cb, &cr);
        assert!((result.hue_shift).abs() < 1e-10);
    }

    #[test]
    fn test_config_defaults() {
        let config = FidelityConfig::default();
        assert!((config.delta_e_max - 50.0).abs() < 1e-10);
        assert!((config.weight_luminance - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_config_builder() {
        let config = FidelityConfig::new()
            .with_delta_e_max(100.0)
            .with_weight_luminance(0.7)
            .with_weight_chrominance(0.3);
        assert!((config.delta_e_max - 100.0).abs() < 1e-10);
        assert!((config.weight_luminance - 0.7).abs() < 1e-10);
        assert!((config.weight_chrominance - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_rgb_to_ycbcr_black() {
        let (y, cb, cr) = rgb_to_ycbcr(0, 0, 0);
        assert_eq!(y, 0);
        assert_eq!(cb, 128);
        assert_eq!(cr, 128);
    }

    #[test]
    fn test_rgb_to_ycbcr_white() {
        let (y, cb, cr) = rgb_to_ycbcr(255, 255, 255);
        assert_eq!(y, 255);
        assert_eq!(cb, 128);
        assert_eq!(cr, 128);
    }

    #[test]
    fn test_analyze_rgb_identical() {
        let analyzer = ColorFidelityAnalyzer::with_defaults();
        let r = identical_planes(50, 128);
        let g = identical_planes(50, 128);
        let b = identical_planes(50, 128);
        let result = analyzer.analyze_rgb(&r, &g, &b, &r, &g, &b);
        assert!((result.fidelity_score - 1.0).abs() < 1e-10);
        assert_eq!(result.rating, FidelityRating::Excellent);
    }

    #[test]
    fn test_rating_thresholds() {
        let analyzer = ColorFidelityAnalyzer::with_defaults();
        // Subtle difference => Good/Excellent
        let y = identical_planes(100, 128);
        let cb = identical_planes(100, 128);
        let cr = identical_planes(100, 128);
        let dist_y = identical_planes(100, 130);
        let result = analyzer.analyze_ycbcr(&y, &cb, &cr, &dist_y, &cb, &cr);
        assert!(
            result.rating == FidelityRating::Excellent || result.rating == FidelityRating::Good
        );
    }
}
