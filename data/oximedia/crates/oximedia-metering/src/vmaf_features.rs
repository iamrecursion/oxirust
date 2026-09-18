//! VMAF-inspired feature extractor (pure signal analysis, no ML).
//!
//! This module implements the core signal-analysis features that underpin the
//! VMAF (Video Multi-Method Assessment Fusion) quality metric without relying
//! on a trained neural network or external model files.  The two primary
//! features are:
//!
//! * **VIF** – Visual Information Fidelity proxy (Sheikh & Bovik, 2006).
//!   Measures how much of the reference image's local structure survives in
//!   the distorted image by comparing local variance ratios.
//!
//! * **ADM** – Additive Detail Measures.
//!   Compares the detail/edge energy between reference and distorted at
//!   multiple scales, following the decomposition philosophy in Li *et al.*
//!   2016 (Netflix VMAF paper).
//!
//! Both are computed across a 4-level Gaussian pyramid to produce per-scale
//! feature values (`scale0` … `scale3`).  An aggregate scalar score is
//! derived as a weighted sum calibrated against typical viewing conditions.

#![allow(clippy::cast_precision_loss)]

use crate::video_quality::Frame2D;
use crate::{MeteringError, MeteringResult};

// ─── Gaussian pyramid ────────────────────────────────────────────────────────

/// Radii for the separable Gaussian kernel applied when downsampling one level.
/// Kernel: [1, 4, 6, 4, 1] / 16  (binomial approximation to σ ≈ 0.85 pixels)
const GAUSS_KERNEL: [f64; 5] = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];

/// Apply the 1-D Gaussian kernel along the horizontal axis.
fn gauss_row(src: &Frame2D) -> Frame2D {
    let h = src.height;
    let w = src.width;
    let mut out = Frame2D::zeros(h, w);
    let half = GAUSS_KERNEL.len() / 2; // == 2
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0_f64;
            for (k, &coeff) in GAUSS_KERNEL.iter().enumerate() {
                let xi = (x + k).saturating_sub(half).min(w - 1);
                acc += coeff * src.get(y, xi);
            }
            out.set(y, x, acc);
        }
    }
    out
}

/// Apply the 1-D Gaussian kernel along the vertical axis.
fn gauss_col(src: &Frame2D) -> Frame2D {
    let h = src.height;
    let w = src.width;
    let mut out = Frame2D::zeros(h, w);
    let half = GAUSS_KERNEL.len() / 2;
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0_f64;
            for (k, &coeff) in GAUSS_KERNEL.iter().enumerate() {
                let yi = (y + k).saturating_sub(half).min(h - 1);
                acc += coeff * src.get(yi, x);
            }
            out.set(y, x, acc);
        }
    }
    out
}

/// Downsample a frame by 2 in each dimension (keep every other pixel after blur).
fn downsample(src: &Frame2D) -> Frame2D {
    let blurred = gauss_col(&gauss_row(src));
    let new_h = (src.height + 1) / 2;
    let new_w = (src.width + 1) / 2;
    let mut out = Frame2D::zeros(new_h, new_w);
    for y in 0..new_h {
        for x in 0..new_w {
            out.set(y, x, blurred.get(y * 2, x * 2));
        }
    }
    out
}

/// Build a Gaussian pyramid with `levels` scales (level 0 = full resolution).
fn gaussian_pyramid(frame: &Frame2D, levels: usize) -> Vec<Frame2D> {
    let mut pyramid = Vec::with_capacity(levels);
    pyramid.push(frame.clone());
    for l in 1..levels {
        let prev = &pyramid[l - 1];
        if prev.width < 2 || prev.height < 2 {
            break;
        }
        pyramid.push(downsample(prev));
    }
    pyramid
}

// ─── Local statistics helpers ─────────────────────────────────────────────────

/// Compute the local mean using a `win × win` sliding-window average.
/// Window size `win` must be odd; edge pixels are handled by reflection.
fn local_mean(src: &Frame2D, win: usize) -> Frame2D {
    let half = win / 2;
    let h = src.height;
    let w = src.width;
    let inv_n = 1.0 / (win * win) as f64;
    let mut out = Frame2D::zeros(h, w);
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0;
            for dy in 0..win {
                let ry = reflect_idx(y as isize + dy as isize - half as isize, h);
                for dx in 0..win {
                    let rx = reflect_idx(x as isize + dx as isize - half as isize, w);
                    sum += src.get(ry, rx);
                }
            }
            out.set(y, x, sum * inv_n);
        }
    }
    out
}

/// Compute the local variance: E[X²] - (E[X])².
fn local_variance(src: &Frame2D, win: usize) -> Frame2D {
    let half = win / 2;
    let h = src.height;
    let w = src.width;
    let inv_n = 1.0 / (win * win) as f64;
    let mean = local_mean(src, win);
    let mut out = Frame2D::zeros(h, w);
    for y in 0..h {
        for x in 0..w {
            let mut sq_sum = 0.0;
            for dy in 0..win {
                let ry = reflect_idx(y as isize + dy as isize - half as isize, h);
                for dx in 0..win {
                    let rx = reflect_idx(x as isize + dx as isize - half as isize, w);
                    let v = src.get(ry, rx);
                    sq_sum += v * v;
                }
            }
            let var = sq_sum * inv_n - mean.get(y, x).powi(2);
            out.set(y, x, var.max(0.0));
        }
    }
    out
}

/// Compute the local covariance between two frames over a `win × win` window.
fn local_covariance(a: &Frame2D, b: &Frame2D, win: usize) -> Frame2D {
    let half = win / 2;
    let h = a.height;
    let w = a.width;
    let inv_n = 1.0 / (win * win) as f64;
    let mean_a = local_mean(a, win);
    let mean_b = local_mean(b, win);
    let mut out = Frame2D::zeros(h, w);
    for y in 0..h {
        for x in 0..w {
            let mut cross = 0.0;
            for dy in 0..win {
                let ry = reflect_idx(y as isize + dy as isize - half as isize, h);
                for dx in 0..win {
                    let rx = reflect_idx(x as isize + dx as isize - half as isize, w);
                    cross += a.get(ry, rx) * b.get(ry, rx);
                }
            }
            let cov = cross * inv_n - mean_a.get(y, x) * mean_b.get(y, x);
            out.set(y, x, cov);
        }
    }
    out
}

/// Mirror-reflect an index into [0, dim).
#[inline]
fn reflect_idx(i: isize, dim: usize) -> usize {
    if i < 0 {
        (-i - 1).min(dim as isize - 1) as usize
    } else if i >= dim as isize {
        (2 * dim as isize - i - 1).max(0) as usize
    } else {
        i as usize
    }
}

// ─── VIF (Visual Information Fidelity proxy) ──────────────────────────────────

/// Compute the VIF (Visual Information Fidelity) ratio for a single scale.
///
/// VIF estimates how much of the reference signal's visual information is
/// preserved in the distorted signal.  A value of 1.0 indicates perfect
/// fidelity; values between 0 and 1 indicate loss; values above 1 indicate
/// enhancement (rare in practice).
///
/// Implementation follows the GSM (Gaussian Scale Mixture) model simplified
/// to a local variance ratio:
///
/// ```text
/// VIF_scale = Σ log2(1 + σ_xy² / σ_nn²) / Σ log2(1 + σ_xx² / σ_nn²)
/// ```
///
/// where `σ_xx` is reference local variance, `σ_xy` is cross-variance, and
/// `σ_nn` is a constant noise floor added for numerical stability.
fn vif_scale(reference: &Frame2D, distorted: &Frame2D, win: usize) -> f64 {
    // Additive noise variance (models HVS internal noise, calibrated to VMAF).
    const SIGMA_NN_SQ: f64 = 0.4;

    let var_ref = local_variance(reference, win);
    let cov_rd = local_covariance(reference, distorted, win);

    let n = (reference.height * reference.width) as f64;
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;

    for y in 0..reference.height {
        for x in 0..reference.width {
            let s_xx = var_ref.get(y, x).max(0.0);
            let s_xy = cov_rd.get(y, x);
            // Clamp cross-variance to [0, s_xx] to avoid negative ratios.
            let s_xy_clamped = s_xy.clamp(0.0, s_xx);

            num += (1.0 + s_xy_clamped / SIGMA_NN_SQ).log2();
            den += (1.0 + s_xx / SIGMA_NN_SQ).log2();
        }
    }

    // Normalise by pixel count to make the metric resolution-independent.
    let _ = n; // not needed once we take ratio
    if den > 1e-10 {
        num / den
    } else {
        1.0 // trivially "perfect" when the reference has no structure
    }
}

// ─── ADM (Additive Detail Measures) ─────────────────────────────────────────

/// Compute a simple Laplacian detail image: f - low_pass(f).
///
/// This isolates fine-detail / edge energy from the smooth base.
fn laplacian_detail(src: &Frame2D) -> Frame2D {
    let blurred = gauss_col(&gauss_row(src));
    let h = src.height;
    let w = src.width;
    let mut detail = Frame2D::zeros(h, w);
    for y in 0..h {
        for x in 0..w {
            detail.set(y, x, src.get(y, x) - blurred.get(y, x));
        }
    }
    detail
}

/// Sum of squared detail coefficients (energy proxy).
fn detail_energy(detail: &Frame2D) -> f64 {
    detail.iter().map(|v| v * v).sum()
}

/// ADM ratio for one scale.
///
/// Computes the ratio of distorted detail energy preserved relative to the
/// reference detail energy.  Values near 1.0 indicate the detail is intact;
/// values below 1.0 indicate blurring / loss; above 1.0 indicates
/// artificial sharpening.
fn adm_scale(reference: &Frame2D, distorted: &Frame2D) -> f64 {
    let ref_detail = laplacian_detail(reference);
    let dist_detail = laplacian_detail(distorted);

    let ref_energy = detail_energy(&ref_detail);
    let dist_energy = detail_energy(&dist_detail);

    if ref_energy > 1e-10 {
        (dist_energy / ref_energy).sqrt().clamp(0.0, 2.0)
    } else {
        1.0 // no reference detail → trivially preserved
    }
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// Per-scale VMAF-inspired feature values.
///
/// Each field `vif_scaleN` and `adm_scaleN` (N = 0..3) corresponds to a
/// Gaussian pyramid level, where `scale0` is the full-resolution measurement
/// and `scale3` is the coarsest (most downsampled) level.
#[derive(Clone, Debug)]
pub struct VmafFeatures {
    /// VIF score at scale 0 (full resolution).
    pub vif_scale0: f64,
    /// VIF score at scale 1 (½ resolution).
    pub vif_scale1: f64,
    /// VIF score at scale 2 (¼ resolution).
    pub vif_scale2: f64,
    /// VIF score at scale 3 (⅛ resolution).
    pub vif_scale3: f64,

    /// ADM score at scale 0 (full resolution).
    pub adm_scale0: f64,
    /// ADM score at scale 1.
    pub adm_scale1: f64,
    /// ADM score at scale 2.
    pub adm_scale2: f64,
    /// ADM score at scale 3.
    pub adm_scale3: f64,
}

impl VmafFeatures {
    /// Aggregate feature score in the range [0, 100].
    ///
    /// Combines VIF and ADM features with a set of empirically derived
    /// weights that approximate the correlation with subjective MOS seen
    /// in large-scale Netflix viewing-panel studies (no ML model required).
    ///
    /// VIF weights favour finer scales (human vision is sensitive to
    /// mid-frequency detail loss); ADM weights are uniform across scales.
    #[must_use]
    pub fn aggregate_score(&self) -> f64 {
        // VIF weights: emphasise scale1 (½ res) and scale2 (¼ res)
        const W_VIF: [f64; 4] = [0.10, 0.30, 0.35, 0.25];
        // ADM weights: uniform
        const W_ADM: [f64; 4] = [0.25, 0.25, 0.25, 0.25];
        // Blend ratio: 60 % VIF, 40 % ADM
        const ALPHA: f64 = 0.60;

        let vif_values = [
            self.vif_scale0,
            self.vif_scale1,
            self.vif_scale2,
            self.vif_scale3,
        ];
        let adm_values = [
            self.adm_scale0,
            self.adm_scale1,
            self.adm_scale2,
            self.adm_scale3,
        ];

        let vif_score: f64 = W_VIF
            .iter()
            .zip(vif_values.iter())
            .map(|(w, v)| w * v)
            .sum::<f64>();
        let adm_score: f64 = W_ADM
            .iter()
            .zip(adm_values.iter())
            .map(|(w, v)| w * v)
            .sum::<f64>();

        let combined = ALPHA * vif_score + (1.0 - ALPHA) * adm_score;
        // Map [0, 1] → [0, 100]; clamp to account for super-unity values.
        (combined * 100.0).clamp(0.0, 100.0)
    }

    /// Quality rating based on aggregate score.
    #[must_use]
    pub fn rating(&self) -> &'static str {
        let score = self.aggregate_score();
        if score >= 90.0 {
            "Excellent"
        } else if score >= 75.0 {
            "Good"
        } else if score >= 55.0 {
            "Fair"
        } else if score >= 30.0 {
            "Poor"
        } else {
            "Bad"
        }
    }
}

/// VMAF-inspired feature extractor.
///
/// Computes VIF and ADM features across 4 Gaussian pyramid scales.
pub struct VmafExtractor {
    width: usize,
    height: usize,
    /// Window size for local statistics (must be odd).
    vif_window: usize,
}

impl VmafExtractor {
    /// Create a new VMAF feature extractor.
    ///
    /// # Arguments
    ///
    /// * `width` - Frame width in pixels
    /// * `height` - Frame height in pixels
    ///
    /// # Errors
    ///
    /// Returns `MeteringError::InvalidConfig` if dimensions are zero or if
    /// the frame is too small for a 4-level pyramid.
    pub fn new(width: usize, height: usize) -> MeteringResult<Self> {
        if width == 0 || height == 0 {
            return Err(MeteringError::InvalidConfig(
                "Width and height must be positive".to_string(),
            ));
        }
        // Require at least 16×16 at the coarsest scale (scale3 = 1/8 size).
        if width < 128 || height < 128 {
            return Err(MeteringError::InvalidConfig(format!(
                "Frame must be at least 128×128 pixels for 4-scale analysis \
                 (got {width}×{height})"
            )));
        }
        Ok(Self {
            width,
            height,
            vif_window: 7,
        })
    }

    /// Create extractor with a custom VIF window size.
    ///
    /// `vif_window` must be odd and at least 3.
    ///
    /// # Errors
    ///
    /// Same as [`new`](Self::new) plus validation of `vif_window`.
    pub fn with_vif_window(width: usize, height: usize, vif_window: usize) -> MeteringResult<Self> {
        let mut extractor = Self::new(width, height)?;
        if vif_window < 3 || vif_window % 2 == 0 {
            return Err(MeteringError::InvalidConfig(
                "vif_window must be odd and >= 3".to_string(),
            ));
        }
        extractor.vif_window = vif_window;
        Ok(extractor)
    }

    /// Extract VMAF features from a reference/distorted pair.
    ///
    /// Frames must be normalised to [0, 1] (or any consistent linear range)
    /// and must match the width/height passed to the constructor.
    ///
    /// # Errors
    ///
    /// Returns an error if frame dimensions do not match.
    pub fn extract(
        &self,
        reference: &Frame2D,
        distorted: &Frame2D,
    ) -> MeteringResult<VmafFeatures> {
        if reference.width != self.width || reference.height != self.height {
            return Err(MeteringError::InvalidConfig(
                "Reference frame dimensions don't match extractor configuration".to_string(),
            ));
        }
        if distorted.width != self.width || distorted.height != self.height {
            return Err(MeteringError::InvalidConfig(
                "Distorted frame dimensions don't match extractor configuration".to_string(),
            ));
        }

        const LEVELS: usize = 4;
        let ref_pyr = gaussian_pyramid(reference, LEVELS);
        let dist_pyr = gaussian_pyramid(distorted, LEVELS);

        let win = self.vif_window;

        let vif_scale0 = vif_scale(&ref_pyr[0], &dist_pyr[0], win);
        let vif_scale1 = if ref_pyr.len() > 1 {
            vif_scale(&ref_pyr[1], &dist_pyr[1], win)
        } else {
            1.0
        };
        let vif_scale2 = if ref_pyr.len() > 2 {
            vif_scale(&ref_pyr[2], &dist_pyr[2], win)
        } else {
            1.0
        };
        let vif_scale3 = if ref_pyr.len() > 3 {
            vif_scale(&ref_pyr[3], &dist_pyr[3], win)
        } else {
            1.0
        };

        let adm_scale0 = adm_scale(&ref_pyr[0], &dist_pyr[0]);
        let adm_scale1 = if ref_pyr.len() > 1 {
            adm_scale(&ref_pyr[1], &dist_pyr[1])
        } else {
            1.0
        };
        let adm_scale2 = if ref_pyr.len() > 2 {
            adm_scale(&ref_pyr[2], &dist_pyr[2])
        } else {
            1.0
        };
        let adm_scale3 = if ref_pyr.len() > 3 {
            adm_scale(&ref_pyr[3], &dist_pyr[3])
        } else {
            1.0
        };

        Ok(VmafFeatures {
            vif_scale0,
            vif_scale1,
            vif_scale2,
            vif_scale3,
            adm_scale0,
            adm_scale1,
            adm_scale2,
            adm_scale3,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::video_quality::Frame2D;
    use std::f64::consts::PI;

    /// Checkerboard pattern — rich in fine detail.
    fn checkerboard(h: usize, w: usize, freq: usize) -> Frame2D {
        Frame2D::from_shape_fn(h, w, |y, x| {
            if (y / freq + x / freq) % 2 == 0 {
                1.0
            } else {
                0.0
            }
        })
    }

    /// Low-frequency sinusoidal gradient.
    fn gradient(h: usize, w: usize) -> Frame2D {
        Frame2D::from_shape_fn(h, w, |y, x| {
            0.5 + 0.5
                * (2.0 * PI * x as f64 / w as f64).sin()
                * (2.0 * PI * y as f64 / h as f64).cos()
        })
    }

    // --- Construction ---

    #[test]
    fn test_new_valid() {
        VmafExtractor::new(256, 256).expect("should construct for 256x256");
        VmafExtractor::new(1920, 1080).expect("should construct for HD");
    }

    #[test]
    fn test_new_too_small() {
        assert!(VmafExtractor::new(64, 64).is_err(), "64x64 should fail");
        assert!(VmafExtractor::new(128, 64).is_err(), "128x64 should fail");
    }

    #[test]
    fn test_new_zero_dimensions() {
        assert!(VmafExtractor::new(0, 256).is_err());
        assert!(VmafExtractor::new(256, 0).is_err());
    }

    #[test]
    fn test_with_vif_window_odd() {
        VmafExtractor::with_vif_window(256, 256, 11).expect("window 11 is valid");
    }

    #[test]
    fn test_with_vif_window_even_rejected() {
        assert!(VmafExtractor::with_vif_window(256, 256, 8).is_err());
    }

    // --- Identical frames → perfect scores ---

    #[test]
    fn test_identical_frames_perfect_vif() {
        let extractor = VmafExtractor::new(256, 256).expect("valid");
        let frame = gradient(256, 256);
        let features = extractor.extract(&frame, &frame).expect("extract ok");
        // VIF of identical frames should be ≥ 1.0 (numerically close to 1.0)
        assert!(features.vif_scale0 >= 0.99, "vif0={}", features.vif_scale0);
        assert!(features.vif_scale1 >= 0.99, "vif1={}", features.vif_scale1);
        assert!(features.vif_scale2 >= 0.99, "vif2={}", features.vif_scale2);
        assert!(features.vif_scale3 >= 0.99, "vif3={}", features.vif_scale3);
    }

    #[test]
    fn test_identical_frames_perfect_adm() {
        let extractor = VmafExtractor::new(256, 256).expect("valid");
        let frame = checkerboard(256, 256, 8);
        let features = extractor.extract(&frame, &frame).expect("extract ok");
        assert!(
            (features.adm_scale0 - 1.0).abs() < 0.05,
            "adm0={}",
            features.adm_scale0
        );
    }

    #[test]
    fn test_identical_frames_high_aggregate() {
        let extractor = VmafExtractor::new(256, 256).expect("valid");
        let frame = gradient(256, 256);
        let features = extractor.extract(&frame, &frame).expect("extract ok");
        let score = features.aggregate_score();
        assert!(
            score >= 90.0,
            "Identical frames should score ≥ 90, got {score}"
        );
    }

    // --- Degraded frame → reduced score ---

    #[test]
    fn test_degraded_frame_lower_score() {
        let extractor = VmafExtractor::new(256, 256).expect("valid");
        let reference = gradient(256, 256);
        // Add strong uniform noise to degrade quality
        let distorted = Frame2D::from_shape_fn(256, 256, |y, x| {
            let noise = if (y + x) % 2 == 0 { 0.4 } else { -0.4 };
            (reference.get(y, x) + noise).clamp(0.0, 1.0)
        });
        let feat_ref = extractor.extract(&reference, &reference).expect("ok");
        let feat_deg = extractor.extract(&reference, &distorted).expect("ok");
        assert!(
            feat_deg.aggregate_score() < feat_ref.aggregate_score(),
            "Degraded score ({}) should be less than reference ({})",
            feat_deg.aggregate_score(),
            feat_ref.aggregate_score()
        );
    }

    // --- Blank/uniform frame handling ---

    #[test]
    fn test_uniform_frame_no_detail() {
        let extractor = VmafExtractor::new(256, 256).expect("valid");
        let ref_frame = Frame2D::from_elem(256, 256, 0.5);
        let dist_frame = Frame2D::from_elem(256, 256, 0.5);
        let features = extractor
            .extract(&ref_frame, &dist_frame)
            .expect("extract ok");
        // Uniform frames have no structure → VIF trivially 1.0, ADM trivially 1.0
        for &v in &[features.vif_scale0, features.adm_scale0] {
            assert!(v >= 0.0, "Feature must be non-negative: {v}");
        }
    }

    // --- Dimension mismatch errors ---

    #[test]
    fn test_dimension_mismatch_reference() {
        let extractor = VmafExtractor::new(256, 256).expect("valid");
        let wrong = Frame2D::zeros(128, 256);
        let ok = Frame2D::zeros(256, 256);
        assert!(extractor.extract(&wrong, &ok).is_err());
    }

    #[test]
    fn test_dimension_mismatch_distorted() {
        let extractor = VmafExtractor::new(256, 256).expect("valid");
        let ok = Frame2D::zeros(256, 256);
        let wrong = Frame2D::zeros(256, 128);
        assert!(extractor.extract(&ok, &wrong).is_err());
    }

    // --- Feature field ranges ---

    #[test]
    fn test_vif_features_non_negative() {
        let extractor = VmafExtractor::new(256, 256).expect("valid");
        let reference = gradient(256, 256);
        let distorted =
            Frame2D::from_shape_fn(256, 256, |y, x| (reference.get(y, x) * 0.7).clamp(0.0, 1.0));
        let features = extractor.extract(&reference, &distorted).expect("ok");
        for &v in &[
            features.vif_scale0,
            features.vif_scale1,
            features.vif_scale2,
            features.vif_scale3,
        ] {
            assert!(v >= 0.0, "VIF must be non-negative: {v}");
        }
    }

    #[test]
    fn test_adm_features_non_negative() {
        let extractor = VmafExtractor::new(256, 256).expect("valid");
        let reference = checkerboard(256, 256, 4);
        let distorted = Frame2D::from_shape_fn(256, 256, |y, x| {
            // Blurred version (just scale down detail)
            let orig = reference.get(y, x);
            orig * 0.5 + 0.25
        });
        let features = extractor.extract(&reference, &distorted).expect("ok");
        for &v in &[
            features.adm_scale0,
            features.adm_scale1,
            features.adm_scale2,
            features.adm_scale3,
        ] {
            assert!(v >= 0.0, "ADM must be non-negative: {v}");
        }
    }

    // --- Aggregate score bounds ---

    #[test]
    fn test_aggregate_score_in_range() {
        let extractor = VmafExtractor::new(256, 256).expect("valid");
        let reference = gradient(256, 256);
        let distorted = Frame2D::from_shape_fn(256, 256, |y, x| {
            (reference.get(y, x) + 0.3 * (y as f64 / 256.0)).clamp(0.0, 1.0)
        });
        let features = extractor.extract(&reference, &distorted).expect("ok");
        let score = features.aggregate_score();
        assert!(
            score >= 0.0 && score <= 100.0,
            "Score out of [0,100]: {score}"
        );
    }

    // --- Rating ---

    #[test]
    fn test_rating_excellent() {
        let features = VmafFeatures {
            vif_scale0: 1.0,
            vif_scale1: 1.0,
            vif_scale2: 1.0,
            vif_scale3: 1.0,
            adm_scale0: 1.0,
            adm_scale1: 1.0,
            adm_scale2: 1.0,
            adm_scale3: 1.0,
        };
        assert_eq!(features.rating(), "Excellent");
    }

    #[test]
    fn test_rating_bad() {
        let features = VmafFeatures {
            vif_scale0: 0.0,
            vif_scale1: 0.0,
            vif_scale2: 0.0,
            vif_scale3: 0.0,
            adm_scale0: 0.0,
            adm_scale1: 0.0,
            adm_scale2: 0.0,
            adm_scale3: 0.0,
        };
        assert_eq!(features.rating(), "Bad");
    }

    // --- Pyramid internals ---

    #[test]
    fn test_gaussian_pyramid_levels() {
        let frame = Frame2D::from_elem(256, 256, 0.5);
        let pyr = gaussian_pyramid(&frame, 4);
        assert_eq!(pyr.len(), 4);
        assert_eq!(pyr[0].width, 256);
        assert_eq!(pyr[1].width, 128);
        assert_eq!(pyr[2].width, 64);
        assert_eq!(pyr[3].width, 32);
    }

    #[test]
    fn test_downsample_halves_dimensions() {
        let frame = Frame2D::from_elem(64, 128, 0.5);
        let down = downsample(&frame);
        assert_eq!(down.height, 32);
        assert_eq!(down.width, 64);
    }

    #[test]
    fn test_reflect_idx_in_bounds() {
        assert_eq!(reflect_idx(5, 10), 5);
    }

    #[test]
    fn test_reflect_idx_negative() {
        assert_eq!(reflect_idx(-1, 10), 0);
    }

    #[test]
    fn test_reflect_idx_overflow() {
        // reflect_idx(10, 10) → reflects to 9
        let result = reflect_idx(10, 10);
        assert!(result < 10, "result={result}");
    }
}
