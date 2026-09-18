#![allow(dead_code)]
//! Color blindness simulation and adaptation for accessible media.
//!
//! This module provides tools to simulate various types of color vision
//! deficiency (CVD) and to transform media colors to be more distinguishable
//! for color-blind viewers. It supports protanopia, deuteranopia, tritanopia,
//! and their partial variants.

use std::fmt;

/// Type of color vision deficiency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CvdType {
    /// Red-blind (missing L-cones).
    Protanopia,
    /// Red-weak (reduced L-cone sensitivity).
    Protanomaly,
    /// Green-blind (missing M-cones).
    Deuteranopia,
    /// Green-weak (reduced M-cone sensitivity).
    Deuteranomaly,
    /// Blue-blind (missing S-cones).
    Tritanopia,
    /// Blue-weak (reduced S-cone sensitivity).
    Tritanomaly,
    /// Total color blindness.
    Achromatopsia,
}

impl fmt::Display for CvdType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Protanopia => write!(f, "Protanopia"),
            Self::Protanomaly => write!(f, "Protanomaly"),
            Self::Deuteranopia => write!(f, "Deuteranopia"),
            Self::Deuteranomaly => write!(f, "Deuteranomaly"),
            Self::Tritanopia => write!(f, "Tritanopia"),
            Self::Tritanomaly => write!(f, "Tritanomaly"),
            Self::Achromatopsia => write!(f, "Achromatopsia"),
        }
    }
}

impl CvdType {
    /// Returns the approximate prevalence rate in the general population.
    #[must_use]
    pub fn prevalence_percent(&self) -> f64 {
        match self {
            Self::Protanopia => 1.01,
            Self::Protanomaly => 1.08,
            Self::Deuteranopia => 1.27,
            Self::Deuteranomaly => 4.63,
            Self::Tritanopia => 0.02,
            Self::Tritanomaly => 0.01,
            Self::Achromatopsia => 0.003,
        }
    }

    /// Returns true if this is a red-green deficiency.
    #[must_use]
    pub fn is_red_green(&self) -> bool {
        matches!(
            self,
            Self::Protanopia | Self::Protanomaly | Self::Deuteranopia | Self::Deuteranomaly
        )
    }

    /// Returns true if this is a blue-yellow deficiency.
    #[must_use]
    pub fn is_blue_yellow(&self) -> bool {
        matches!(self, Self::Tritanopia | Self::Tritanomaly)
    }

    /// Returns true if this is a complete deficiency (not anomalous).
    #[must_use]
    pub fn is_complete(&self) -> bool {
        matches!(
            self,
            Self::Protanopia | Self::Deuteranopia | Self::Tritanopia | Self::Achromatopsia
        )
    }
}

/// An sRGB color value with 8-bit channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rgb {
    /// Red channel (0-255).
    pub r: u8,
    /// Green channel (0-255).
    pub g: u8,
    /// Blue channel (0-255).
    pub b: u8,
}

impl Rgb {
    /// Create a new RGB color.
    #[must_use]
    pub fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    /// Convert to linear-space floating point values (0.0-1.0).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn to_linear(&self) -> [f64; 3] {
        [
            srgb_to_linear(f64::from(self.r) / 255.0),
            srgb_to_linear(f64::from(self.g) / 255.0),
            srgb_to_linear(f64::from(self.b) / 255.0),
        ]
    }

    /// Create from linear-space floating point values.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    fn from_linear(linear: [f64; 3]) -> Self {
        Self {
            r: (linear_to_srgb(linear[0].clamp(0.0, 1.0)) * 255.0).round() as u8,
            g: (linear_to_srgb(linear[1].clamp(0.0, 1.0)) * 255.0).round() as u8,
            b: (linear_to_srgb(linear[2].clamp(0.0, 1.0)) * 255.0).round() as u8,
        }
    }

    /// Compute the perceived luminance (BT.709).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn luminance(&self) -> f64 {
        let lin = self.to_linear();
        0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    }

    /// Compute the contrast ratio between two colors (WCAG formula).
    #[must_use]
    pub fn contrast_ratio(&self, other: &Rgb) -> f64 {
        let l1 = self.luminance() + 0.05;
        let l2 = other.luminance() + 0.05;
        if l1 > l2 {
            l1 / l2
        } else {
            l2 / l1
        }
    }
}

/// Convert sRGB component to linear.
fn srgb_to_linear(c: f64) -> f64 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Convert linear component to sRGB.
fn linear_to_srgb(c: f64) -> f64 {
    if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// Simulates how a color appears under a specific CVD type.
#[derive(Debug, Clone)]
pub struct CvdSimulator {
    /// The type of CVD to simulate.
    cvd_type: CvdType,
    /// Severity from 0.0 (normal) to 1.0 (full deficiency).
    severity: f64,
}

impl CvdSimulator {
    /// Create a new CVD simulator.
    #[must_use]
    pub fn new(cvd_type: CvdType, severity: f64) -> Self {
        Self {
            cvd_type,
            severity: severity.clamp(0.0, 1.0),
        }
    }

    /// Create a simulator for the full deficiency.
    #[must_use]
    pub fn full(cvd_type: CvdType) -> Self {
        Self::new(cvd_type, 1.0)
    }

    /// Get the CVD type.
    #[must_use]
    pub fn cvd_type(&self) -> CvdType {
        self.cvd_type
    }

    /// Get the severity.
    #[must_use]
    pub fn severity(&self) -> f64 {
        self.severity
    }

    /// Simulate how a color appears under this CVD type.
    ///
    /// Uses simplified Brettel/Vienot simulation matrices.
    #[must_use]
    pub fn simulate(&self, color: Rgb) -> Rgb {
        let lin = color.to_linear();
        let simulated = self.apply_matrix(lin);

        // Interpolate between original and simulated based on severity
        let blended = [
            lin[0] + self.severity * (simulated[0] - lin[0]),
            lin[1] + self.severity * (simulated[1] - lin[1]),
            lin[2] + self.severity * (simulated[2] - lin[2]),
        ];

        Rgb::from_linear(blended)
    }

    /// Apply the CVD simulation matrix.
    fn apply_matrix(&self, rgb: [f64; 3]) -> [f64; 3] {
        let matrix = self.get_matrix();
        [
            matrix[0][0] * rgb[0] + matrix[0][1] * rgb[1] + matrix[0][2] * rgb[2],
            matrix[1][0] * rgb[0] + matrix[1][1] * rgb[1] + matrix[1][2] * rgb[2],
            matrix[2][0] * rgb[0] + matrix[2][1] * rgb[1] + matrix[2][2] * rgb[2],
        ]
    }

    /// Get the simulation matrix for the CVD type (simplified Vienot-style).
    fn get_matrix(&self) -> [[f64; 3]; 3] {
        match self.cvd_type {
            CvdType::Protanopia | CvdType::Protanomaly => [
                [0.567, 0.433, 0.000],
                [0.558, 0.442, 0.000],
                [0.000, 0.242, 0.758],
            ],
            CvdType::Deuteranopia | CvdType::Deuteranomaly => [
                [0.625, 0.375, 0.000],
                [0.700, 0.300, 0.000],
                [0.000, 0.300, 0.700],
            ],
            CvdType::Tritanopia | CvdType::Tritanomaly => [
                [0.950, 0.050, 0.000],
                [0.000, 0.433, 0.567],
                [0.000, 0.475, 0.525],
            ],
            CvdType::Achromatopsia => [
                [0.2126, 0.7152, 0.0722],
                [0.2126, 0.7152, 0.0722],
                [0.2126, 0.7152, 0.0722],
            ],
        }
    }
}

/// Configuration for color adaptation to improve visibility for CVD.
#[derive(Debug, Clone)]
pub struct AdaptationConfig {
    /// The target CVD type to adapt for.
    pub target_cvd: CvdType,
    /// Adaptation strength from 0.0 (none) to 1.0 (maximum).
    pub strength: f64,
    /// Minimum contrast ratio to maintain (WCAG recommends 4.5:1).
    pub min_contrast_ratio: f64,
}

impl Default for AdaptationConfig {
    fn default() -> Self {
        Self {
            target_cvd: CvdType::Deuteranopia,
            strength: 0.8,
            min_contrast_ratio: 4.5,
        }
    }
}

impl AdaptationConfig {
    /// Create a new adaptation config for a specific CVD type.
    #[must_use]
    pub fn new(target_cvd: CvdType) -> Self {
        Self {
            target_cvd,
            ..Self::default()
        }
    }

    /// Set the adaptation strength.
    #[must_use]
    pub fn with_strength(mut self, strength: f64) -> Self {
        self.strength = strength.clamp(0.0, 1.0);
        self
    }

    /// Set the minimum contrast ratio.
    #[must_use]
    pub fn with_min_contrast(mut self, ratio: f64) -> Self {
        self.min_contrast_ratio = ratio.max(1.0);
        self
    }
}

/// Checks a pair of colors for distinguishability under a CVD type.
#[must_use]
pub fn colors_distinguishable(a: Rgb, b: Rgb, cvd_type: CvdType, threshold: f64) -> bool {
    let sim = CvdSimulator::full(cvd_type);
    let sa = sim.simulate(a);
    let sb = sim.simulate(b);
    sa.contrast_ratio(&sb) >= threshold
}

/// Computes the WCAG contrast level between two colors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WcagContrastLevel {
    /// Fails minimum contrast requirements.
    Fail,
    /// Meets AA level for large text (3:1).
    AaLarge,
    /// Meets AA level (4.5:1).
    Aa,
    /// Meets AAA level (7:1).
    Aaa,
}

impl fmt::Display for WcagContrastLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fail => write!(f, "Fail"),
            Self::AaLarge => write!(f, "AA Large"),
            Self::Aa => write!(f, "AA"),
            Self::Aaa => write!(f, "AAA"),
        }
    }
}

/// Evaluate the WCAG contrast level for a given contrast ratio.
#[must_use]
pub fn wcag_contrast_level(ratio: f64) -> WcagContrastLevel {
    if ratio >= 7.0 {
        WcagContrastLevel::Aaa
    } else if ratio >= 4.5 {
        WcagContrastLevel::Aa
    } else if ratio >= 3.0 {
        WcagContrastLevel::AaLarge
    } else {
        WcagContrastLevel::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cvd_type_display() {
        assert_eq!(CvdType::Protanopia.to_string(), "Protanopia");
        assert_eq!(CvdType::Deuteranomaly.to_string(), "Deuteranomaly");
        assert_eq!(CvdType::Achromatopsia.to_string(), "Achromatopsia");
    }

    #[test]
    fn test_cvd_prevalence() {
        let p = CvdType::Deuteranomaly.prevalence_percent();
        assert!(p > 4.0 && p < 5.0);
    }

    #[test]
    fn test_cvd_red_green() {
        assert!(CvdType::Protanopia.is_red_green());
        assert!(CvdType::Deuteranomaly.is_red_green());
        assert!(!CvdType::Tritanopia.is_red_green());
        assert!(!CvdType::Achromatopsia.is_red_green());
    }

    #[test]
    fn test_cvd_blue_yellow() {
        assert!(CvdType::Tritanopia.is_blue_yellow());
        assert!(CvdType::Tritanomaly.is_blue_yellow());
        assert!(!CvdType::Protanopia.is_blue_yellow());
    }

    #[test]
    fn test_cvd_complete() {
        assert!(CvdType::Protanopia.is_complete());
        assert!(!CvdType::Protanomaly.is_complete());
        assert!(CvdType::Achromatopsia.is_complete());
    }

    #[test]
    fn test_rgb_basic() {
        let c = Rgb::new(255, 0, 0);
        assert_eq!(c.r, 255);
        assert_eq!(c.g, 0);
        assert_eq!(c.b, 0);
    }

    #[test]
    fn test_rgb_luminance_white() {
        let white = Rgb::new(255, 255, 255);
        assert!((white.luminance() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_rgb_luminance_black() {
        let black = Rgb::new(0, 0, 0);
        assert!(black.luminance().abs() < 0.001);
    }

    #[test]
    fn test_contrast_ratio_black_white() {
        let white = Rgb::new(255, 255, 255);
        let black = Rgb::new(0, 0, 0);
        let ratio = white.contrast_ratio(&black);
        // WCAG says black/white contrast is 21:1
        assert!(ratio > 20.0);
    }

    #[test]
    fn test_contrast_ratio_same_color() {
        let red = Rgb::new(255, 0, 0);
        let ratio = red.contrast_ratio(&red);
        assert!((ratio - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cvd_simulator_zero_severity() {
        let sim = CvdSimulator::new(CvdType::Protanopia, 0.0);
        let red = Rgb::new(255, 0, 0);
        let result = sim.simulate(red);
        assert_eq!(result, red);
    }

    #[test]
    fn test_achromatopsia_simulation() {
        let sim = CvdSimulator::full(CvdType::Achromatopsia);
        let red = Rgb::new(255, 0, 0);
        let result = sim.simulate(red);
        // For achromatopsia, R/G/B channels should all be equal (greyscale)
        assert_eq!(result.r, result.g);
        assert_eq!(result.g, result.b);
    }

    #[test]
    fn test_adaptation_config_defaults() {
        let config = AdaptationConfig::default();
        assert_eq!(config.target_cvd, CvdType::Deuteranopia);
        assert!((config.strength - 0.8).abs() < f64::EPSILON);
        assert!((config.min_contrast_ratio - 4.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_adaptation_config_builder() {
        let config = AdaptationConfig::new(CvdType::Tritanopia)
            .with_strength(0.5)
            .with_min_contrast(7.0);
        assert_eq!(config.target_cvd, CvdType::Tritanopia);
        assert!((config.strength - 0.5).abs() < f64::EPSILON);
        assert!((config.min_contrast_ratio - 7.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_wcag_contrast_level_aaa() {
        assert_eq!(wcag_contrast_level(7.5), WcagContrastLevel::Aaa);
    }

    #[test]
    fn test_wcag_contrast_level_aa() {
        assert_eq!(wcag_contrast_level(5.0), WcagContrastLevel::Aa);
    }

    #[test]
    fn test_wcag_contrast_level_aa_large() {
        assert_eq!(wcag_contrast_level(3.5), WcagContrastLevel::AaLarge);
    }

    #[test]
    fn test_wcag_contrast_level_fail() {
        assert_eq!(wcag_contrast_level(2.0), WcagContrastLevel::Fail);
    }

    #[test]
    fn test_colors_distinguishable_black_white() {
        let black = Rgb::new(0, 0, 0);
        let white = Rgb::new(255, 255, 255);
        assert!(colors_distinguishable(
            black,
            white,
            CvdType::Protanopia,
            3.0
        ));
    }
}
