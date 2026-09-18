#![allow(dead_code)]
//! Metamerism index calculation and detection.
//!
//! Metamerism occurs when two color samples appear to match under one illuminant
//! but differ under another. This is a critical issue in color calibration because
//! a calibration performed under one lighting condition may not transfer correctly
//! to another. This module provides tools to quantify metamerism risk and detect
//! metameric color pairs.
//!
//! # Features
//!
//! - **Metamerism index**: Compute MI (observer or illuminant metamerism)
//! - **Pair detection**: Identify metameric color pairs in a palette
//! - **Illuminant comparison**: Compare color appearance under different illuminants
//! - **Risk assessment**: Classify metamerism severity

/// Severity classification for metamerism.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetamerismSeverity {
    /// No perceptible metamerism (MI < 0.5).
    None,
    /// Slight metamerism, acceptable for most work (MI < 1.5).
    Slight,
    /// Moderate metamerism, visible under comparison (MI < 3.0).
    Moderate,
    /// Severe metamerism, clearly visible (MI >= 3.0).
    Severe,
}

impl MetamerismSeverity {
    /// Classify a metamerism index value.
    #[must_use]
    pub fn from_index(mi: f64) -> Self {
        if mi < 0.5 {
            Self::None
        } else if mi < 1.5 {
            Self::Slight
        } else if mi < 3.0 {
            Self::Moderate
        } else {
            Self::Severe
        }
    }

    /// Get a human-readable description of the severity.
    #[must_use]
    pub const fn description(&self) -> &'static str {
        match self {
            Self::None => "No perceptible metamerism",
            Self::Slight => "Slight metamerism, acceptable for most work",
            Self::Moderate => "Moderate metamerism, visible under comparison",
            Self::Severe => "Severe metamerism, clearly visible",
        }
    }
}

/// Spectral power distribution represented as a set of wavelength-value pairs.
#[derive(Debug, Clone)]
pub struct SpectralData {
    /// Start wavelength (nm).
    pub start_nm: f64,
    /// Wavelength interval (nm).
    pub interval_nm: f64,
    /// Spectral values at each wavelength.
    pub values: Vec<f64>,
}

impl SpectralData {
    /// Create a new spectral data set.
    #[must_use]
    pub fn new(start_nm: f64, interval_nm: f64, values: Vec<f64>) -> Self {
        Self {
            start_nm,
            interval_nm,
            values,
        }
    }

    /// Get the wavelength at a given index.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn wavelength_at(&self, index: usize) -> f64 {
        self.start_nm + index as f64 * self.interval_nm
    }

    /// Get the number of spectral samples.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if the spectral data is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Interpolate the spectral value at a given wavelength.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn interpolate(&self, wavelength: f64) -> f64 {
        if self.values.is_empty() || self.interval_nm <= 0.0 {
            return 0.0;
        }
        let idx_f = (wavelength - self.start_nm) / self.interval_nm;
        if idx_f < 0.0 {
            return self.values[0];
        }
        let idx = idx_f.floor() as usize;
        if idx >= self.values.len() - 1 {
            return self.values[self.values.len() - 1];
        }
        let frac = idx_f - idx_f.floor();
        self.values[idx] * (1.0 - frac) + self.values[idx + 1] * frac
    }
}

/// An illuminant spectral power distribution.
#[derive(Debug, Clone)]
pub struct IlluminantSpd {
    /// Name of the illuminant.
    pub name: String,
    /// Spectral power distribution data.
    pub spd: SpectralData,
}

/// CIE XYZ tristimulus values.
#[derive(Debug, Clone, Copy)]
pub struct CieXyz {
    /// X component.
    pub x: f64,
    /// Y component.
    pub y: f64,
    /// Z component.
    pub z: f64,
}

impl CieXyz {
    /// Create new XYZ values.
    #[must_use]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
}

/// CIE L*a*b* color value.
#[derive(Debug, Clone, Copy)]
pub struct CieLab {
    /// Lightness (0..100).
    pub l: f64,
    /// Green-red axis.
    pub a: f64,
    /// Blue-yellow axis.
    pub b: f64,
}

impl CieLab {
    /// Create new Lab values.
    #[must_use]
    pub const fn new(l: f64, a: f64, b: f64) -> Self {
        Self { l, a, b }
    }

    /// Compute CIE76 Delta-E between two Lab colors.
    #[must_use]
    pub fn delta_e76(&self, other: &Self) -> f64 {
        let dl = self.l - other.l;
        let da = self.a - other.a;
        let db = self.b - other.b;
        (dl * dl + da * da + db * db).sqrt()
    }
}

/// Convert XYZ to Lab given a reference white point.
#[must_use]
fn xyz_to_lab(xyz: &CieXyz, white: &CieXyz) -> CieLab {
    let f = |t: f64| -> f64 {
        if t > 0.008_856 {
            t.cbrt()
        } else {
            7.787 * t + 16.0 / 116.0
        }
    };
    let fx = f(xyz.x / white.x);
    let fy = f(xyz.y / white.y);
    let fz = f(xyz.z / white.z);
    CieLab::new(116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz))
}

/// A metameric pair result.
#[derive(Debug, Clone)]
pub struct MetamericPair {
    /// Index of the first color.
    pub index_a: usize,
    /// Index of the second color.
    pub index_b: usize,
    /// Delta-E under the first illuminant.
    pub delta_e_illuminant_1: f64,
    /// Delta-E under the second illuminant.
    pub delta_e_illuminant_2: f64,
    /// Metamerism index (difference in Delta-E).
    pub metamerism_index: f64,
    /// Severity classification.
    pub severity: MetamerismSeverity,
}

/// Configuration for metamerism analysis.
#[derive(Debug, Clone)]
pub struct MetamerismConfig {
    /// Delta-E threshold under the reference illuminant to consider a match.
    pub match_threshold: f64,
    /// Minimum metamerism index to report.
    pub min_report_index: f64,
    /// Number of spectral samples for integration.
    pub integration_steps: usize,
}

impl Default for MetamerismConfig {
    fn default() -> Self {
        Self {
            match_threshold: 2.0,
            min_report_index: 0.5,
            integration_steps: 31,
        }
    }
}

impl MetamerismConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the match threshold.
    #[must_use]
    pub fn with_match_threshold(mut self, threshold: f64) -> Self {
        self.match_threshold = threshold.max(0.0);
        self
    }

    /// Set the minimum report index.
    #[must_use]
    pub fn with_min_report_index(mut self, min: f64) -> Self {
        self.min_report_index = min.max(0.0);
        self
    }
}

/// Metamerism analyzer for color calibration workflows.
#[derive(Debug)]
pub struct MetamerismAnalyzer {
    /// Configuration.
    config: MetamerismConfig,
}

impl MetamerismAnalyzer {
    /// Create a new metamerism analyzer.
    #[must_use]
    pub fn new(config: MetamerismConfig) -> Self {
        Self { config }
    }

    /// Create an analyzer with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(MetamerismConfig::default())
    }

    /// Compute the metamerism index between two Lab colors under two illuminants.
    ///
    /// The metamerism index is the absolute difference in Delta-E values
    /// computed under two different illuminants.
    #[must_use]
    pub fn compute_index(
        lab_a_illum1: &CieLab,
        lab_b_illum1: &CieLab,
        lab_a_illum2: &CieLab,
        lab_b_illum2: &CieLab,
    ) -> f64 {
        let de1 = lab_a_illum1.delta_e76(lab_b_illum1);
        let de2 = lab_a_illum2.delta_e76(lab_b_illum2);
        (de2 - de1).abs()
    }

    /// Find all metameric pairs in a set of Lab colors under two illuminants.
    ///
    /// `colors_illum1` and `colors_illum2` must be the same length, with each
    /// entry representing the same physical color measured under different illuminants.
    #[must_use]
    pub fn find_metameric_pairs(
        &self,
        colors_illum1: &[CieLab],
        colors_illum2: &[CieLab],
    ) -> Vec<MetamericPair> {
        let n = colors_illum1.len().min(colors_illum2.len());
        let mut pairs = Vec::new();

        for i in 0..n {
            for j in (i + 1)..n {
                let de1 = colors_illum1[i].delta_e76(&colors_illum1[j]);
                // Only consider colors that appear to match under illuminant 1
                if de1 > self.config.match_threshold {
                    continue;
                }
                let de2 = colors_illum2[i].delta_e76(&colors_illum2[j]);
                let mi = (de2 - de1).abs();
                if mi >= self.config.min_report_index {
                    pairs.push(MetamericPair {
                        index_a: i,
                        index_b: j,
                        delta_e_illuminant_1: de1,
                        delta_e_illuminant_2: de2,
                        metamerism_index: mi,
                        severity: MetamerismSeverity::from_index(mi),
                    });
                }
            }
        }

        pairs.sort_by(|a, b| {
            b.metamerism_index
                .partial_cmp(&a.metamerism_index)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        pairs
    }

    /// Compute the overall metamerism risk score for a set of colors.
    ///
    /// Returns a value in [0.0, 1.0] where 0 means no risk and 1 means high risk.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn risk_score(&self, colors_illum1: &[CieLab], colors_illum2: &[CieLab]) -> f64 {
        let pairs = self.find_metameric_pairs(colors_illum1, colors_illum2);
        if pairs.is_empty() {
            return 0.0;
        }
        let max_mi = pairs
            .iter()
            .map(|p| p.metamerism_index)
            .fold(0.0_f64, f64::max);
        // Normalize: MI of 5.0 or above = 1.0 risk
        (max_mi / 5.0).min(1.0)
    }

    /// Get a reference to the configuration.
    #[must_use]
    pub const fn config(&self) -> &MetamerismConfig {
        &self.config
    }
}

/// Approximate CIE 1931 2-degree x-bar color matching function.
#[must_use]
pub fn cie_xbar(wavelength: f64) -> f64 {
    let t1 = (wavelength - 442.0) * (if wavelength < 442.0 { 0.0624 } else { 0.0374 });
    let t2 = (wavelength - 599.8) * (if wavelength < 599.8 { 0.0264 } else { 0.0323 });
    let t3 = (wavelength - 501.1) * (if wavelength < 501.1 { 0.0490 } else { 0.0382 });
    0.362 * (-0.5 * t1 * t1).exp() + 1.056 * (-0.5 * t2 * t2).exp() - 0.065 * (-0.5 * t3 * t3).exp()
}

/// Approximate CIE 1931 2-degree y-bar color matching function.
#[must_use]
pub fn cie_ybar(wavelength: f64) -> f64 {
    let t1 = (wavelength - 568.8) * (if wavelength < 568.8 { 0.0213 } else { 0.0247 });
    let t2 = (wavelength - 530.9) * (if wavelength < 530.9 { 0.0613 } else { 0.0322 });
    1.014 * (-0.5 * t1 * t1).exp() + 0.160 * (-0.5 * t2 * t2).exp()
}

/// Approximate CIE 1931 2-degree z-bar color matching function.
#[must_use]
pub fn cie_zbar(wavelength: f64) -> f64 {
    let t1 = (wavelength - 437.0) * (if wavelength < 437.0 { 0.0845 } else { 0.0278 });
    let t2 = (wavelength - 459.0) * (if wavelength < 459.0 { 0.0385 } else { 0.0725 });
    1.839 * (-0.5 * t1 * t1).exp() + 0.218 * (-0.5 * t2 * t2).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_severity_classification() {
        assert_eq!(
            MetamerismSeverity::from_index(0.2),
            MetamerismSeverity::None
        );
        assert_eq!(
            MetamerismSeverity::from_index(1.0),
            MetamerismSeverity::Slight
        );
        assert_eq!(
            MetamerismSeverity::from_index(2.0),
            MetamerismSeverity::Moderate
        );
        assert_eq!(
            MetamerismSeverity::from_index(5.0),
            MetamerismSeverity::Severe
        );
    }

    #[test]
    fn test_severity_description() {
        let s = MetamerismSeverity::None;
        assert!(!s.description().is_empty());
    }

    #[test]
    fn test_spectral_data_creation() {
        let spd = SpectralData::new(380.0, 10.0, vec![0.1, 0.2, 0.3, 0.4]);
        assert_eq!(spd.len(), 4);
        assert!(!spd.is_empty());
        assert!((spd.wavelength_at(0) - 380.0).abs() < 1e-10);
        assert!((spd.wavelength_at(2) - 400.0).abs() < 1e-10);
    }

    #[test]
    fn test_spectral_interpolation() {
        let spd = SpectralData::new(380.0, 10.0, vec![0.0, 1.0, 2.0, 3.0]);
        let val = spd.interpolate(385.0);
        assert!((val - 0.5).abs() < 1e-10);
        // Clamp to boundary
        let val_low = spd.interpolate(370.0);
        assert!((val_low - 0.0).abs() < 1e-10);
        let val_high = spd.interpolate(420.0);
        assert!((val_high - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_lab_delta_e() {
        let a = CieLab::new(50.0, 0.0, 0.0);
        let b = CieLab::new(50.0, 0.0, 0.0);
        assert!((a.delta_e76(&b) - 0.0).abs() < 1e-10);

        let c = CieLab::new(50.0, 3.0, 4.0);
        let de = a.delta_e76(&c);
        assert!((de - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_xyz_to_lab() {
        let white = CieXyz::new(0.95047, 1.0, 1.08883);
        let xyz = CieXyz::new(0.95047, 1.0, 1.08883);
        let lab = xyz_to_lab(&xyz, &white);
        assert!((lab.l - 100.0).abs() < 0.1);
        assert!(lab.a.abs() < 0.1);
        assert!(lab.b.abs() < 0.1);
    }

    #[test]
    fn test_compute_metamerism_index() {
        let a1 = CieLab::new(50.0, 10.0, 10.0);
        let b1 = CieLab::new(50.0, 10.5, 10.5);
        let a2 = CieLab::new(50.0, 10.0, 10.0);
        let b2 = CieLab::new(50.0, 14.0, 14.0);
        let mi = MetamerismAnalyzer::compute_index(&a1, &b1, &a2, &b2);
        assert!(mi > 0.0);
    }

    #[test]
    fn test_find_metameric_pairs() {
        let analyzer = MetamerismAnalyzer::with_defaults();
        let illum1 = vec![
            CieLab::new(50.0, 10.0, 10.0),
            CieLab::new(50.0, 10.5, 10.5), // close to first under illum1
            CieLab::new(80.0, -20.0, 30.0),
        ];
        let illum2 = vec![
            CieLab::new(50.0, 10.0, 10.0),
            CieLab::new(50.0, 15.0, 15.0), // diverges under illum2
            CieLab::new(80.0, -20.0, 30.0),
        ];
        let pairs = analyzer.find_metameric_pairs(&illum1, &illum2);
        // Pair (0,1) should be detected
        assert!(!pairs.is_empty());
        assert_eq!(pairs[0].index_a, 0);
        assert_eq!(pairs[0].index_b, 1);
    }

    #[test]
    fn test_find_no_pairs_when_different() {
        let analyzer = MetamerismAnalyzer::with_defaults();
        let illum1 = vec![
            CieLab::new(50.0, 10.0, 10.0),
            CieLab::new(80.0, -30.0, 40.0), // very different
        ];
        let illum2 = vec![
            CieLab::new(50.0, 10.0, 10.0),
            CieLab::new(80.0, -30.0, 40.0),
        ];
        let pairs = analyzer.find_metameric_pairs(&illum1, &illum2);
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_risk_score_zero() {
        let analyzer = MetamerismAnalyzer::with_defaults();
        let colors = vec![
            CieLab::new(50.0, 10.0, 10.0),
            CieLab::new(80.0, -30.0, 40.0),
        ];
        let score = analyzer.risk_score(&colors, &colors);
        assert!((score - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_risk_score_nonzero() {
        let analyzer = MetamerismAnalyzer::new(
            MetamerismConfig::new()
                .with_match_threshold(2.0)
                .with_min_report_index(0.1),
        );
        let illum1 = vec![CieLab::new(50.0, 10.0, 10.0), CieLab::new(50.0, 10.5, 10.5)];
        let illum2 = vec![CieLab::new(50.0, 10.0, 10.0), CieLab::new(50.0, 18.0, 18.0)];
        let score = analyzer.risk_score(&illum1, &illum2);
        assert!(score > 0.0);
        assert!(score <= 1.0);
    }

    #[test]
    fn test_cie_color_matching_functions() {
        // xbar should peak around 600nm
        let x_peak = cie_xbar(600.0);
        assert!(x_peak > 0.5);
        // ybar should peak around 555nm
        let y_peak = cie_ybar(555.0);
        assert!(y_peak > 0.5);
        // zbar should peak around 445nm
        let z_peak = cie_zbar(445.0);
        assert!(z_peak > 0.5);
    }

    #[test]
    fn test_config_builder() {
        let cfg = MetamerismConfig::new()
            .with_match_threshold(3.0)
            .with_min_report_index(1.0);
        assert!((cfg.match_threshold - 3.0).abs() < 1e-10);
        assert!((cfg.min_report_index - 1.0).abs() < 1e-10);
    }
}
