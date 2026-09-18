//! Perceptual quality model.
//!
//! Implements a human visual system (HVS) based quality model including
//! contrast sensitivity functions (CSF), just-noticeable difference (JND)
//! thresholds, and contrast masking effects.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::suboptimal_flops)]

use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Spatial frequency in cycles per degree (cpd).
pub type CyclesPerDegree = f64;

/// Contrast sensitivity function (CSF) model variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CsfModel {
    /// Mannos–Sakrison (1974) approximation
    MannosSakrison,
    /// Barten (1999) simplified model
    Barten,
    /// Flat (unity) CSF — used as a baseline
    Flat,
}

/// Evaluates a contrast sensitivity function at a given spatial frequency.
///
/// Returns the normalised sensitivity (1.0 at peak, 0..1 elsewhere).
#[must_use]
pub fn csf_spatial(model: CsfModel, freq_cpd: CyclesPerDegree) -> f64 {
    match model {
        CsfModel::Flat => 1.0,
        CsfModel::MannosSakrison => {
            // Mannos & Sakrison (1974):
            // CSF(f) = 2.6 * (0.0192 + 0.114*f) * exp(-(0.114*f)^1.1)
            if freq_cpd <= 0.0 {
                return 0.0;
            }
            let a = 0.114 * freq_cpd;
            2.6 * (0.0192 + a) * (-a.powf(1.1)).exp()
        }
        CsfModel::Barten => {
            // Simplified Barten model, peak around 4–6 cpd
            if freq_cpd <= 0.0 {
                return 0.0;
            }
            let f0 = 5.0_f64; // peak frequency
            let sigma = 3.5_f64;
            (-(freq_cpd - f0).powi(2) / (2.0 * sigma * sigma)).exp()
        }
    }
}

/// Evaluates a temporal contrast sensitivity function.
///
/// Returns normalised sensitivity at a temporal frequency in Hz.
#[must_use]
pub fn csf_temporal(freq_hz: f64) -> f64 {
    if freq_hz <= 0.0 {
        return 1.0;
    }
    // Simple de Lange curve approximation
    let f_c = 40.0_f64; // critical fusion frequency (Hz)
    if freq_hz >= f_c {
        return 0.0;
    }
    (1.0 - (freq_hz / f_c)).max(0.0)
}

/// Just-noticeable difference (JND) parameters.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct JndParams {
    /// Minimum detectable luminance change relative to background (Weber fraction)
    pub weber_fraction: f64,
    /// Background luminance in cd/m²
    pub background_luminance: f64,
    /// Masking factor (0 = no masking, 1 = full masking)
    pub masking_factor: f64,
}

impl Default for JndParams {
    fn default() -> Self {
        Self {
            weber_fraction: 0.01,
            background_luminance: 100.0,
            masking_factor: 0.0,
        }
    }
}

impl JndParams {
    /// Computes the JND threshold in luminance units (cd/m²).
    ///
    /// The threshold is increased by the masking factor to model
    /// the reduced visibility of errors in busy/textured regions.
    #[must_use]
    pub fn threshold(&self) -> f64 {
        let base = self.weber_fraction * self.background_luminance;
        base * (1.0 + self.masking_factor)
    }

    /// Returns true if `delta_luminance` (in cd/m²) is visible.
    #[must_use]
    pub fn is_visible(&self, delta_luminance: f64) -> bool {
        delta_luminance.abs() > self.threshold()
    }
}

/// Contrast masking model.
///
/// In regions of high contrast (textures, edges) the HVS is less sensitive
/// to additional distortions. This model computes a local masking weight.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ContrastMasking {
    /// Masking exponent (typical: 0.7)
    pub exponent: f64,
    /// Reference contrast (0–1 range, normalised)
    pub reference_contrast: f64,
}

impl Default for ContrastMasking {
    fn default() -> Self {
        Self {
            exponent: 0.7,
            reference_contrast: 0.5,
        }
    }
}

impl ContrastMasking {
    /// Computes the masking weight for a local contrast `c`.
    ///
    /// Returns a value in [0, 1] where 1 = no masking, lower = more masking.
    #[must_use]
    pub fn masking_weight(&self, local_contrast: f64) -> f64 {
        if local_contrast <= 0.0 {
            return 1.0;
        }
        let ratio = local_contrast / self.reference_contrast;
        (1.0 / (1.0 + ratio.powf(self.exponent))).clamp(0.0, 1.0)
    }

    /// Applies masking to a distortion value.
    ///
    /// Returns the perceptually adjusted distortion.
    #[must_use]
    pub fn apply(&self, distortion: f64, local_contrast: f64) -> f64 {
        distortion * self.masking_weight(local_contrast)
    }
}

/// A perceptual distortion map over an image block grid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerceptualDistortionMap {
    /// Number of blocks in the horizontal direction
    pub width_blocks: usize,
    /// Number of blocks in the vertical direction
    pub height_blocks: usize,
    /// Block size in pixels (square)
    pub block_size: usize,
    /// Per-block perceptually weighted distortion values
    pub values: Vec<f64>,
}

impl PerceptualDistortionMap {
    /// Creates a new distortion map initialised to zero.
    #[must_use]
    pub fn new(width_blocks: usize, height_blocks: usize, block_size: usize) -> Self {
        Self {
            width_blocks,
            height_blocks,
            block_size,
            values: vec![0.0; width_blocks * height_blocks],
        }
    }

    /// Returns the total number of blocks.
    #[must_use]
    pub fn block_count(&self) -> usize {
        self.width_blocks * self.height_blocks
    }

    /// Returns a mutable reference to the value at block `(bx, by)`.
    pub fn get_mut(&mut self, bx: usize, by: usize) -> Option<&mut f64> {
        if bx < self.width_blocks && by < self.height_blocks {
            Some(&mut self.values[by * self.width_blocks + bx])
        } else {
            None
        }
    }

    /// Returns the value at block `(bx, by)`.
    #[must_use]
    pub fn get(&self, bx: usize, by: usize) -> Option<f64> {
        if bx < self.width_blocks && by < self.height_blocks {
            Some(self.values[by * self.width_blocks + bx])
        } else {
            None
        }
    }

    /// Computes the mean perceptual distortion across all blocks.
    #[must_use]
    pub fn mean_distortion(&self) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        self.values.iter().sum::<f64>() / self.values.len() as f64
    }

    /// Computes the maximum per-block distortion.
    #[must_use]
    pub fn max_distortion(&self) -> f64 {
        self.values
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    }
}

/// Computes a perceptual quality score from a distortion map.
///
/// Applies CSF weighting in the spatial domain and returns a
/// normalised score in [0, 100], where 100 is perfect quality.
#[must_use]
pub fn perceptual_quality_score(
    distortion_map: &PerceptualDistortionMap,
    masking: &ContrastMasking,
    local_contrasts: &[f64],
    scale_factor: f64,
) -> f64 {
    if distortion_map.values.is_empty() || local_contrasts.len() != distortion_map.block_count() {
        return 100.0;
    }

    let weighted_sum: f64 = distortion_map
        .values
        .iter()
        .zip(local_contrasts.iter())
        .map(|(&d, &c)| masking.apply(d, c))
        .sum();

    let mean_weighted = weighted_sum / distortion_map.block_count() as f64;
    let raw_score = mean_weighted * scale_factor;
    (100.0 - raw_score).clamp(0.0, 100.0)
}

/// Frequency-domain CSF weighting for a DCT coefficient grid.
///
/// `dct_coeffs` is a slice of (`freq_u`, `freq_v`, coefficient) tuples.
/// Returns a weighted sum suitable for perceptual error measurement.
#[must_use]
pub fn dct_csf_weight(model: CsfModel, freq_u: f64, freq_v: f64, viewing_distance_h: f64) -> f64 {
    // Convert DCT frequency indices to cycles per degree
    let cycles = (freq_u * freq_u + freq_v * freq_v).sqrt();
    let cpd = cycles / (2.0 * viewing_distance_h * (PI / 180.0).tan());
    csf_spatial(model, cpd)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csf_flat_is_unity() {
        for f in [0.5, 1.0, 4.0, 10.0, 30.0] {
            assert!((csf_spatial(CsfModel::Flat, f) - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_csf_mannos_sakrison_zero_freq() {
        let v = csf_spatial(CsfModel::MannosSakrison, 0.0);
        assert_eq!(v, 0.0);
    }

    #[test]
    fn test_csf_mannos_sakrison_positive() {
        let v = csf_spatial(CsfModel::MannosSakrison, 5.0);
        assert!(v > 0.0);
    }

    #[test]
    fn test_csf_barten_peak_near_5cpd() {
        let v_5 = csf_spatial(CsfModel::Barten, 5.0);
        let v_20 = csf_spatial(CsfModel::Barten, 20.0);
        assert!(v_5 > v_20, "Barten CSF should peak near 5 cpd");
    }

    #[test]
    fn test_csf_temporal_zero_freq() {
        assert!((csf_temporal(0.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_csf_temporal_at_critical_fusion() {
        let v = csf_temporal(40.0);
        assert_eq!(v, 0.0);
    }

    #[test]
    fn test_csf_temporal_decreases() {
        let v1 = csf_temporal(5.0);
        let v2 = csf_temporal(20.0);
        assert!(v1 > v2);
    }

    #[test]
    fn test_jnd_threshold_default() {
        let p = JndParams::default();
        let threshold = p.threshold();
        assert!((threshold - 1.0).abs() < 1e-9); // 0.01 * 100.0 = 1.0
    }

    #[test]
    fn test_jnd_is_visible() {
        let p = JndParams::default();
        assert!(p.is_visible(2.0)); // 2.0 > 1.0
        assert!(!p.is_visible(0.5)); // 0.5 < 1.0
    }

    #[test]
    fn test_jnd_masking_increases_threshold() {
        let mut p = JndParams::default();
        let base_threshold = p.threshold();
        p.masking_factor = 1.0;
        let masked_threshold = p.threshold();
        assert!(masked_threshold > base_threshold);
    }

    #[test]
    fn test_contrast_masking_no_contrast() {
        let cm = ContrastMasking::default();
        let w = cm.masking_weight(0.0);
        assert!((w - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_contrast_masking_high_contrast_reduces_weight() {
        let cm = ContrastMasking::default();
        let w_low = cm.masking_weight(0.1);
        let w_high = cm.masking_weight(2.0);
        assert!(w_high < w_low);
    }

    #[test]
    fn test_perceptual_distortion_map_new() {
        let map = PerceptualDistortionMap::new(8, 6, 16);
        assert_eq!(map.block_count(), 48);
        assert!(map.values.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_perceptual_distortion_map_get_set() {
        let mut map = PerceptualDistortionMap::new(4, 4, 8);
        *map.get_mut(2, 3).expect("should succeed in test") = 0.5;
        assert!((map.get(2, 3).expect("should succeed in test") - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_perceptual_distortion_map_mean() {
        let mut map = PerceptualDistortionMap::new(2, 2, 8);
        for v in map.values.iter_mut() {
            *v = 1.0;
        }
        assert!((map.mean_distortion() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_perceptual_quality_score_perfect() {
        let map = PerceptualDistortionMap::new(4, 4, 8);
        let contrasts = vec![0.0f64; 16];
        let cm = ContrastMasking::default();
        let score = perceptual_quality_score(&map, &cm, &contrasts, 1.0);
        assert!((score - 100.0).abs() < 1e-9);
    }
}
