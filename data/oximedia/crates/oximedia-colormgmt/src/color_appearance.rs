//! Color appearance models for perceptual color representation.
//!
//! Provides structures and utilities for modeling how colors are perceived
//! under different viewing conditions, based on CIECAM02 concepts.

#![allow(dead_code)]

/// Viewing condition surround type, which affects perceived lightness and chroma.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewingCondition {
    /// Average surround (typical office or studio monitor viewing).
    Average,
    /// Dim surround (television viewing in a dim room).
    Dim,
    /// Dark surround (cinema, projector in a dark room).
    Dark,
}

impl ViewingCondition {
    /// Returns the surround factor `c` used in CIECAM02-style models.
    ///
    /// Higher values correspond to more chromatic adaptation and contrast.
    #[must_use]
    pub fn surround_factor(self) -> f64 {
        match self {
            Self::Average => 0.69,
            Self::Dim => 0.59,
            Self::Dark => 0.525,
        }
    }

    /// Returns the chromatic induction factor `Nc`.
    #[must_use]
    pub fn chromatic_induction(self) -> f64 {
        match self {
            Self::Average => 1.0,
            Self::Dim => 0.9,
            Self::Dark => 0.8,
        }
    }

    /// Returns a human-readable label for this viewing condition.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Average => "Average",
            Self::Dim => "Dim",
            Self::Dark => "Dark",
        }
    }
}

/// Adaptation luminance level of the viewing environment (cd/m²).
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct AdaptationLuminance {
    /// Luminance value in cd/m².
    pub cd_m2: f64,
}

impl AdaptationLuminance {
    /// Creates a new `AdaptationLuminance`.
    ///
    /// # Arguments
    ///
    /// * `cd_m2` - Luminance in candelas per square metre (must be > 0).
    #[must_use]
    pub fn new(cd_m2: f64) -> Self {
        Self {
            cd_m2: cd_m2.max(0.001),
        }
    }

    /// Returns `true` when the luminance level is considered "dim" (< 20 cd/m²).
    #[must_use]
    pub fn is_dim(self) -> bool {
        self.cd_m2 < 20.0
    }

    /// Returns `true` when the luminance level is considered "dark" (< 5 cd/m²).
    #[must_use]
    pub fn is_dark(self) -> bool {
        self.cd_m2 < 5.0
    }

    /// Returns the corresponding [`ViewingCondition`] based on luminance.
    #[must_use]
    pub fn viewing_condition(self) -> ViewingCondition {
        if self.is_dark() {
            ViewingCondition::Dark
        } else if self.is_dim() {
            ViewingCondition::Dim
        } else {
            ViewingCondition::Average
        }
    }

    /// Returns the adapted luminance factor `LA` (1/5 of absolute luminance).
    #[must_use]
    pub fn la_factor(self) -> f64 {
        self.cd_m2 / 5.0
    }
}

/// Perceptual attributes of a color under a specified viewing condition.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColorAppearance {
    /// Perceived lightness (J), range 0–100.
    pub j: f64,
    /// Perceived chroma (C), range ≥ 0.
    pub c: f64,
    /// Hue angle (h) in degrees, range 0–360.
    pub h: f64,
    /// Brightness (Q), correlate of absolute lightness.
    pub q: f64,
    /// Colorfulness (M), range ≥ 0.
    pub m: f64,
    /// Saturation (s), range 0–100.
    pub s: f64,
}

impl ColorAppearance {
    /// Creates a `ColorAppearance` from raw CIECAM02-like correlates.
    #[must_use]
    pub fn new(j: f64, c: f64, h: f64, q: f64, m: f64, s: f64) -> Self {
        Self {
            j: j.clamp(0.0, 100.0),
            c: c.max(0.0),
            h: h.rem_euclid(360.0),
            q: q.max(0.0),
            m: m.max(0.0),
            s: s.clamp(0.0, 100.0),
        }
    }

    /// Returns perceived lightness J (0–100).
    #[must_use]
    pub fn lightness(self) -> f64 {
        self.j
    }

    /// Returns perceived chroma C (≥ 0).
    #[must_use]
    pub fn chroma(self) -> f64 {
        self.c
    }

    /// Returns hue angle h in degrees (0–360).
    #[must_use]
    pub fn hue_angle(self) -> f64 {
        self.h
    }

    /// Returns the hue quadrant name.
    #[must_use]
    pub fn hue_name(self) -> &'static str {
        match self.h as u32 {
            0..=89 => "Red-Yellow",
            90..=179 => "Yellow-Green",
            180..=269 => "Green-Blue",
            _ => "Blue-Red",
        }
    }

    /// Returns `true` if the color is achromatic (chroma near zero).
    #[must_use]
    pub fn is_achromatic(self) -> bool {
        self.c < 1.0
    }

    /// Returns a simple CIECAM02-approximate appearance from XYZ under given conditions.
    ///
    /// This is a simplified calculation for demonstration; full CIECAM02 requires
    /// additional viewing condition parameters.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn from_xyz_approx(
        xyz: (f64, f64, f64),
        luminance: AdaptationLuminance,
        condition: ViewingCondition,
    ) -> Self {
        let (_x, y, _z) = xyz;
        // Simplified J from Y and surround factor
        let c = condition.surround_factor();
        let la = luminance.la_factor();
        let fl = 0.2 * c.powi(4) * (5.0 * la).powf(1.0 / 3.0).max(1.0);
        let j = 100.0 * (y * fl / 100.0).powf(0.42).min(1.0);
        // Simplified chroma from x/z deviations
        let x_dev = (xyz.0 - 0.9505).abs();
        let z_dev = (xyz.2 - 1.0889).abs();
        let chroma = (x_dev * x_dev + z_dev * z_dev).sqrt() * 100.0;
        // Hue from atan2 of a/b proxies
        let a_proxy = xyz.0 - y;
        let b_proxy = y - xyz.2;
        let h = b_proxy.atan2(a_proxy).to_degrees().rem_euclid(360.0);
        let q = (4.0 / c) * (j / 100.0).sqrt() * (la.max(1.0) + 15.0) * (fl / 100.0).powf(0.25);
        let m = chroma * fl.powf(0.25);
        let s = if q > 0.0 { 100.0 * (m / q).sqrt() } else { 0.0 };
        Self::new(j, chroma, h, q, m, s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_viewing_condition_surround_factors() {
        assert!((ViewingCondition::Average.surround_factor() - 0.69).abs() < 1e-10);
        assert!((ViewingCondition::Dim.surround_factor() - 0.59).abs() < 1e-10);
        assert!((ViewingCondition::Dark.surround_factor() - 0.525).abs() < 1e-10);
    }

    #[test]
    fn test_viewing_condition_labels() {
        assert_eq!(ViewingCondition::Average.label(), "Average");
        assert_eq!(ViewingCondition::Dim.label(), "Dim");
        assert_eq!(ViewingCondition::Dark.label(), "Dark");
    }

    #[test]
    fn test_chromatic_induction_ordering() {
        assert!(
            ViewingCondition::Average.chromatic_induction()
                > ViewingCondition::Dim.chromatic_induction()
        );
        assert!(
            ViewingCondition::Dim.chromatic_induction()
                > ViewingCondition::Dark.chromatic_induction()
        );
    }

    #[test]
    fn test_adaptation_luminance_dim() {
        let lum = AdaptationLuminance::new(10.0);
        assert!(lum.is_dim());
        assert!(!lum.is_dark());
    }

    #[test]
    fn test_adaptation_luminance_dark() {
        let lum = AdaptationLuminance::new(3.0);
        assert!(lum.is_dim());
        assert!(lum.is_dark());
    }

    #[test]
    fn test_adaptation_luminance_average() {
        let lum = AdaptationLuminance::new(64.0);
        assert!(!lum.is_dim());
        assert!(!lum.is_dark());
    }

    #[test]
    fn test_viewing_condition_from_luminance() {
        assert_eq!(
            AdaptationLuminance::new(100.0).viewing_condition(),
            ViewingCondition::Average
        );
        assert_eq!(
            AdaptationLuminance::new(10.0).viewing_condition(),
            ViewingCondition::Dim
        );
        assert_eq!(
            AdaptationLuminance::new(1.0).viewing_condition(),
            ViewingCondition::Dark
        );
    }

    #[test]
    fn test_adaptation_luminance_floor() {
        let lum = AdaptationLuminance::new(-5.0);
        assert!(lum.cd_m2 > 0.0);
    }

    #[test]
    fn test_color_appearance_clamps() {
        let app = ColorAppearance::new(-10.0, -5.0, 400.0, -1.0, -2.0, 150.0);
        assert_eq!(app.lightness(), 0.0);
        assert_eq!(app.chroma(), 0.0);
        assert!((app.hue_angle() - 40.0).abs() < 1e-8);
        assert_eq!(app.s, 100.0);
    }

    #[test]
    fn test_color_appearance_lightness() {
        let app = ColorAppearance::new(75.0, 30.0, 120.0, 50.0, 10.0, 40.0);
        assert!((app.lightness() - 75.0).abs() < 1e-10);
    }

    #[test]
    fn test_color_appearance_chroma() {
        let app = ColorAppearance::new(50.0, 45.0, 90.0, 30.0, 15.0, 60.0);
        assert!((app.chroma() - 45.0).abs() < 1e-10);
    }

    #[test]
    fn test_color_appearance_hue_angle() {
        let app = ColorAppearance::new(50.0, 20.0, 270.0, 25.0, 8.0, 50.0);
        assert!((app.hue_angle() - 270.0).abs() < 1e-10);
    }

    #[test]
    fn test_achromatic_detection() {
        let achromatic = ColorAppearance::new(50.0, 0.5, 180.0, 30.0, 0.0, 0.0);
        assert!(achromatic.is_achromatic());
        let chromatic = ColorAppearance::new(50.0, 30.0, 180.0, 30.0, 10.0, 50.0);
        assert!(!chromatic.is_achromatic());
    }

    #[test]
    fn test_from_xyz_approx_returns_valid_range() {
        let xyz = (0.9505, 1.0, 1.0888);
        let lum = AdaptationLuminance::new(64.0);
        let app = ColorAppearance::from_xyz_approx(xyz, lum, ViewingCondition::Average);
        assert!(app.lightness() >= 0.0 && app.lightness() <= 100.0);
        assert!(app.hue_angle() >= 0.0 && app.hue_angle() < 360.0);
    }

    #[test]
    fn test_hue_name_quadrants() {
        let app1 = ColorAppearance::new(50.0, 20.0, 45.0, 30.0, 8.0, 40.0);
        assert_eq!(app1.hue_name(), "Red-Yellow");
        let app2 = ColorAppearance::new(50.0, 20.0, 135.0, 30.0, 8.0, 40.0);
        assert_eq!(app2.hue_name(), "Yellow-Green");
        let app3 = ColorAppearance::new(50.0, 20.0, 225.0, 30.0, 8.0, 40.0);
        assert_eq!(app3.hue_name(), "Green-Blue");
        let app4 = ColorAppearance::new(50.0, 20.0, 315.0, 30.0, 8.0, 40.0);
        assert_eq!(app4.hue_name(), "Blue-Red");
    }
}
