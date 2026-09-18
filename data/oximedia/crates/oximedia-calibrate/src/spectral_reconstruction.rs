//! Spectral reconstruction — recover spectral power distributions (SPDs) from
//! RGB camera measurements.
//!
//! This module implements the Wiener pseudo-inverse approach to spectral
//! recovery, which finds the minimum-variance spectral estimate given a
//! measured camera RGB triplet and knowledge of the camera spectral
//! sensitivities and the prior spectral covariance of the illuminant/scene.
//!
//! # Background
//!
//! Given a camera with sensitivity matrix **A** (3×N, where N is the number of
//! spectral bands) and a measured RGB vector **ρ**, the goal is to find an SPD
//! **s** (N×1) such that **A** **s** ≈ **ρ**. The Wiener estimator is:
//!
//! ```text
//! ŝ = Rss Aᵀ (A Rss Aᵀ + σ² I)⁻¹ ρ
//! ```
//!
//! where **Rss** is the N×N prior covariance of natural scene spectra and σ²
//! is the camera noise variance.
//!
//! In this implementation the camera sensitivities approximate the CIE 1931
//! colour-matching functions x̄(λ), ȳ(λ), z̄(λ) sampled at 10 nm intervals
//! from 400–700 nm (31 bands). The prior covariance is modelled as a Gaussian
//! kernel in wavelength space.

#![allow(dead_code)]

use crate::error::{CalibrationError, CalibrationResult};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of spectral bands: 400–700 nm at 10 nm steps.
pub const NUM_BANDS: usize = 31;

/// Wavelengths in nm corresponding to the 31 spectral bands.
pub const WAVELENGTHS: [f32; NUM_BANDS] = {
    let mut w = [0_f32; NUM_BANDS];
    let mut i = 0usize;
    while i < NUM_BANDS {
        w[i] = 400.0 + i as f32 * 10.0;
        i += 1;
    }
    w
};

/// CIE 1931 2° colour-matching function x̄(λ) at 10 nm, 400–700 nm.
const CIE_X_BAR: [f64; NUM_BANDS] = [
    0.014_310, 0.043_510, 0.134_380, 0.283_900, 0.348_280, 0.336_200, 0.290_800, 0.195_360,
    0.095_640, 0.032_010, 0.004_900, 0.009_300, 0.063_270, 0.165_500, 0.290_400, 0.433_450,
    0.594_500, 0.762_100, 0.916_300, 1.026_300, 1.062_200, 1.002_600, 0.854_450, 0.642_400,
    0.447_900, 0.283_500, 0.164_900, 0.087_400, 0.046_800, 0.022_700, 0.011_400,
];

/// CIE 1931 2° colour-matching function ȳ(λ) at 10 nm, 400–700 nm.
const CIE_Y_BAR: [f64; NUM_BANDS] = [
    0.000_396, 0.001_210, 0.004_000, 0.011_600, 0.023_000, 0.038_000, 0.060_000, 0.090_980,
    0.139_020, 0.208_020, 0.323_000, 0.503_000, 0.710_000, 0.862_000, 0.954_000, 0.994_950,
    0.995_000, 0.952_000, 0.870_000, 0.757_000, 0.631_000, 0.503_000, 0.381_000, 0.265_000,
    0.175_000, 0.107_000, 0.061_000, 0.032_000, 0.017_000, 0.008_210, 0.004_102,
];

/// CIE 1931 2° colour-matching function z̄(λ) at 10 nm, 400–700 nm.
const CIE_Z_BAR: [f64; NUM_BANDS] = [
    0.067_850, 0.207_600, 0.645_600, 1.385_600, 1.747_060, 1.772_110, 1.669_200, 1.287_640,
    0.812_950, 0.465_180, 0.272_000, 0.158_200, 0.078_250, 0.042_160, 0.020_300, 0.008_750,
    0.003_900, 0.002_100, 0.001_650, 0.001_100, 0.000_800, 0.000_340, 0.000_190, 0.000_050,
    0.000_020, 0.000_010, 0.000_000, 0.000_000, 0.000_000, 0.000_000, 0.000_000,
];

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A spectral power distribution sampled at [`NUM_BANDS`] 10 nm steps (400–700 nm).
#[derive(Debug, Clone)]
pub struct SpectralPowerDistribution {
    /// SPD values at each wavelength band (linear energy units, normalised so
    /// that the peak is ≤ 1.0 when the distribution represents a relative SPD).
    pub values: [f64; NUM_BANDS],
}

impl SpectralPowerDistribution {
    /// Returns the dominant wavelength index (argmax of SPD values).
    #[must_use]
    pub fn dominant_band(&self) -> usize {
        self.values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Returns the dominant wavelength in nm.
    #[must_use]
    pub fn dominant_wavelength_nm(&self) -> f32 {
        WAVELENGTHS[self.dominant_band()]
    }

    /// Computes the XYZ tristimulus values by integrating with the CIE 1931 CMFs.
    #[must_use]
    pub fn to_xyz(&self) -> [f64; 3] {
        let mut x = 0.0_f64;
        let mut y = 0.0_f64;
        let mut z = 0.0_f64;
        for i in 0..NUM_BANDS {
            x += self.values[i] * CIE_X_BAR[i];
            y += self.values[i] * CIE_Y_BAR[i];
            z += self.values[i] * CIE_Z_BAR[i];
        }
        // Normalise by the Y sum of the equal-energy illuminant.
        let y_norm: f64 = CIE_Y_BAR.iter().sum();
        [x / y_norm, y / y_norm, z / y_norm]
    }
}

// ---------------------------------------------------------------------------
// Wiener estimator
// ---------------------------------------------------------------------------

/// Wiener pseudo-inverse spectral reconstructor.
///
/// # Approach
///
/// The reconstruction minimises the mean-squared spectral error under a
/// Gaussian prior covariance of the scene spectral statistics:
///
/// ```text
/// ŝ = Rss Aᵀ (A Rss Aᵀ + σ² I₃)⁻¹ ρ
/// ```
///
/// where **A** is the 3×31 camera sensitivity matrix (CIE CMFs by default),
/// **Rss** is the 31×31 prior covariance (Gaussian kernel), and σ² is the
/// noise variance.
#[derive(Debug, Clone)]
pub struct WienerSpectralReconstructor {
    /// Pre-computed reconstruction matrix W (31×3) such that ŝ = W ρ.
    weight_matrix: Vec<Vec<f64>>, // NUM_BANDS × 3
    /// Noise variance used to build the estimator.
    pub noise_variance: f64,
    /// Prior covariance Gaussian bandwidth (nm).
    pub prior_bandwidth_nm: f64,
}

impl WienerSpectralReconstructor {
    /// Build a new `WienerSpectralReconstructor`.
    ///
    /// # Arguments
    ///
    /// * `noise_variance` - Camera sensor noise variance σ².  A typical value
    ///   for a well-exposed scene is ~1e-4.
    /// * `prior_bandwidth_nm` - Standard deviation of the Gaussian prior
    ///   covariance in wavelength space (nm).  Natural daylit spectra have
    ///   bandwidths of roughly 40–80 nm.
    ///
    /// # Errors
    ///
    /// Returns an error if the 3×3 inner matrix is singular.
    pub fn new(noise_variance: f64, prior_bandwidth_nm: f64) -> CalibrationResult<Self> {
        // Build camera sensitivity matrix A (3 × NUM_BANDS).
        let a = Self::build_sensitivity_matrix();

        // Build Gaussian prior covariance Rss (NUM_BANDS × NUM_BANDS).
        let rss = Self::build_prior_covariance(prior_bandwidth_nm);

        // Compute A * Rss (3 × NUM_BANDS).
        let a_rss = mat_mul_3xn_nxn(&a, &rss);

        // Compute A * Rss * Aᵀ (3 × 3).
        let a_rss_at = mat_mul_3xn_nx3(&a_rss, &a);

        // Add noise term: M = A Rss Aᵀ + σ² I₃
        let mut m = a_rss_at;
        m[0][0] += noise_variance;
        m[1][1] += noise_variance;
        m[2][2] += noise_variance;

        // Invert 3×3 M.
        let m_inv = mat3x3_inv(&m).ok_or_else(|| {
            CalibrationError::InsufficientData(
                "Wiener estimator: inner 3×3 matrix is singular".to_string(),
            )
        })?;

        // Compute W = Rss * Aᵀ * M_inv (NUM_BANDS × 3).
        // Step 1: Rss * Aᵀ (NUM_BANDS × 3).
        let rss_at = mat_mul_nxn_nx3(&rss, &a);

        // Step 2: W = Rss_At * M_inv (NUM_BANDS × 3).
        let weight_matrix = mat_mul_nx3_3x3(&rss_at, &m_inv);

        Ok(Self {
            weight_matrix,
            noise_variance,
            prior_bandwidth_nm,
        })
    }

    /// Reconstruct the SPD from a camera RGB triplet.
    ///
    /// # Arguments
    ///
    /// * `rgb` - Normalised camera RGB (each channel in [0.0, 1.0]).
    ///
    /// # Returns
    ///
    /// The reconstructed `SpectralPowerDistribution`.
    #[must_use]
    pub fn reconstruct(&self, rgb: &[f64; 3]) -> SpectralPowerDistribution {
        let mut values = [0.0_f64; NUM_BANDS];
        for i in 0..NUM_BANDS {
            values[i] = self.weight_matrix[i][0] * rgb[0]
                + self.weight_matrix[i][1] * rgb[1]
                + self.weight_matrix[i][2] * rgb[2];
            // Clamp to non-negative; physical SPDs cannot be negative.
            if values[i] < 0.0 {
                values[i] = 0.0;
            }
        }
        SpectralPowerDistribution { values }
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    /// Build the 3×31 camera sensitivity matrix using CIE 1931 CMFs.
    fn build_sensitivity_matrix() -> [[f64; NUM_BANDS]; 3] {
        [CIE_X_BAR, CIE_Y_BAR, CIE_Z_BAR]
    }

    /// Build the NUM_BANDS×NUM_BANDS Gaussian prior covariance matrix.
    fn build_prior_covariance(bandwidth_nm: f64) -> Vec<Vec<f64>> {
        let mut rss = vec![vec![0.0_f64; NUM_BANDS]; NUM_BANDS];
        let bw2 = 2.0 * bandwidth_nm * bandwidth_nm;
        for i in 0..NUM_BANDS {
            for j in 0..NUM_BANDS {
                let dw = (WAVELENGTHS[i] - WAVELENGTHS[j]) as f64;
                rss[i][j] = (-dw * dw / bw2).exp();
            }
        }
        rss
    }
}

// ---------------------------------------------------------------------------
// Matrix helpers for variable N (= NUM_BANDS)
// ---------------------------------------------------------------------------

/// Multiply (3×N) × (N×N) → (3×N).
fn mat_mul_3xn_nxn(a: &[[f64; NUM_BANDS]; 3], b: &[Vec<f64>]) -> [[f64; NUM_BANDS]; 3] {
    let mut out = [[0.0_f64; NUM_BANDS]; 3];
    for i in 0..3 {
        for j in 0..NUM_BANDS {
            for k in 0..NUM_BANDS {
                out[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    out
}

/// Multiply (3×N) × (3×N)ᵀ → (3×3), i.e. (3×N) × (N×3).
fn mat_mul_3xn_nx3(a: &[[f64; NUM_BANDS]; 3], b: &[[f64; NUM_BANDS]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..NUM_BANDS {
                out[i][j] += a[i][k] * b[j][k]; // b transposed
            }
        }
    }
    out
}

/// Multiply (N×N) × (3×N)ᵀ → (N×3).
fn mat_mul_nxn_nx3(a: &[Vec<f64>], b: &[[f64; NUM_BANDS]; 3]) -> Vec<[f64; 3]> {
    let n = a.len();
    let mut out = vec![[0.0_f64; 3]; n];
    for i in 0..n {
        for j in 0..3 {
            for k in 0..NUM_BANDS {
                out[i][j] += a[i][k] * b[j][k]; // b transposed
            }
        }
    }
    out
}

/// Multiply (N×3) × (3×3) → (N×3).
fn mat_mul_nx3_3x3(a: &[[f64; 3]], b: &[[f64; 3]; 3]) -> Vec<Vec<f64>> {
    a.iter()
        .map(|row| {
            let mut out_row = vec![0.0_f64; 3];
            for j in 0..3 {
                for k in 0..3 {
                    out_row[j] += row[k] * b[k][j];
                }
            }
            out_row
        })
        .collect()
}

/// Invert a 3×3 matrix (Cramer's rule).
fn mat3x3_inv(m: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    if det.abs() < 1e-15 {
        return None;
    }

    let d = 1.0 / det;
    Some([
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * d,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * d,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * d,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * d,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * d,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * d,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * d,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * d,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * d,
        ],
    ])
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_reconstructor() -> WienerSpectralReconstructor {
        WienerSpectralReconstructor::new(1e-4, 50.0)
            .expect("Should build without error for typical parameters")
    }

    #[test]
    fn test_build_reconstructor_succeeds() {
        let r = WienerSpectralReconstructor::new(1e-4, 50.0);
        assert!(r.is_ok(), "Builder should succeed for typical parameters");
    }

    #[test]
    fn test_reconstruct_neutral_grey_produces_non_negative_spd() {
        let rec = make_reconstructor();
        let grey = [0.5, 0.5, 0.5];
        let spd = rec.reconstruct(&grey);
        for (i, &v) in spd.values.iter().enumerate() {
            assert!(
                v >= 0.0,
                "SPD value at band {} ({} nm) must be non-negative, got {}",
                i,
                WAVELENGTHS[i],
                v
            );
        }
    }

    #[test]
    fn test_reconstruct_black_produces_near_zero_spd() {
        let rec = make_reconstructor();
        let black = [0.0, 0.0, 0.0];
        let spd = rec.reconstruct(&black);
        let total: f64 = spd.values.iter().sum();
        assert!(
            total.abs() < 1e-10,
            "Black RGB should reconstruct to zero SPD, total={total}"
        );
    }

    #[test]
    fn test_spd_to_xyz_equal_energy_neutral() {
        let rec = make_reconstructor();
        // A neutral grey should produce roughly equal XYZ.
        let grey = [0.18, 0.18, 0.18];
        let spd = rec.reconstruct(&grey);
        let xyz = spd.to_xyz();
        assert!(
            xyz[0] > 0.0 && xyz[1] > 0.0,
            "XYZ values should be positive for neutral grey: {xyz:?}"
        );
    }

    #[test]
    fn test_dominant_wavelength_in_range() {
        let rec = make_reconstructor();
        let spd = rec.reconstruct(&[0.5, 0.5, 0.5]);
        let dom_nm = spd.dominant_wavelength_nm();
        assert!(
            (400.0..=700.0).contains(&dom_nm),
            "Dominant wavelength must be in [400, 700] nm, got {dom_nm}"
        );
    }

    #[test]
    fn test_wavelengths_constant_correct() {
        assert!((WAVELENGTHS[0] - 400.0).abs() < 1e-3);
        assert!((WAVELENGTHS[NUM_BANDS - 1] - 700.0).abs() < 1e-3);
        assert_eq!(WAVELENGTHS.len(), NUM_BANDS);
    }

    #[test]
    fn test_spd_linearity() {
        // Reconstructor is linear: reconstruct(2x) should ≈ 2 * reconstruct(x).
        let rec = make_reconstructor();
        let rgb1 = [0.2, 0.3, 0.4];
        let rgb2 = [0.4, 0.6, 0.8];
        let spd1 = rec.reconstruct(&rgb1);
        let spd2 = rec.reconstruct(&rgb2);
        for i in 0..NUM_BANDS {
            let expected = spd1.values[i] * 2.0;
            // Reconstruction of rgb2 = 2*rgb1 should produce ≈ 2 * spd1.
            // Due to non-negative clamp, this only holds if both are positive.
            if spd1.values[i] > 1e-6 && spd2.values[i] > 1e-6 {
                let rel_err = (spd2.values[i] - expected).abs() / (expected + 1e-10);
                assert!(
                    rel_err < 0.01,
                    "Linearity violated at band {i}: 2*spd1={expected:.6}, spd2={:.6}",
                    spd2.values[i]
                );
            }
        }
    }
}
