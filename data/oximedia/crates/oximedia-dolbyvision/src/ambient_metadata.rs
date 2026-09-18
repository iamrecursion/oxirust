#![allow(dead_code)]
//! Ambient viewing environment metadata for Dolby Vision.
//!
//! Provides structures for describing the ambient light conditions and viewing
//! environment in which Dolby Vision content is intended to be displayed.
//! This metadata influences display mapping decisions to ensure optimal
//! perceptual quality under varying room conditions.
//!
//! # Background
//!
//! The ambient environment significantly affects how HDR content is perceived.
//! A bright room requires different tone mapping than a dark cinema environment.
//! This module captures those conditions for use in adaptive display management.

/// Ambient light measurement describing the viewing environment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AmbientLight {
    /// Ambient illuminance in lux (0.0 = completely dark, ~500 = typical office).
    pub illuminance_lux: f64,
    /// Ambient light chromaticity x coordinate (CIE 1931).
    pub chromaticity_x: f64,
    /// Ambient light chromaticity y coordinate (CIE 1931).
    pub chromaticity_y: f64,
}

impl AmbientLight {
    /// Create a new ambient light measurement.
    #[must_use]
    pub fn new(illuminance_lux: f64, chromaticity_x: f64, chromaticity_y: f64) -> Self {
        Self {
            illuminance_lux,
            chromaticity_x,
            chromaticity_y,
        }
    }

    /// Standard D65 white point ambient light (typical daylight-balanced room).
    #[must_use]
    pub fn d65_standard(illuminance_lux: f64) -> Self {
        Self {
            illuminance_lux,
            chromaticity_x: 0.3127,
            chromaticity_y: 0.3290,
        }
    }

    /// Dark cinema environment (near-zero ambient light).
    #[must_use]
    pub fn dark_cinema() -> Self {
        Self::d65_standard(5.0)
    }

    /// Dim living room environment.
    #[must_use]
    pub fn dim_living_room() -> Self {
        Self::d65_standard(50.0)
    }

    /// Bright office environment.
    #[must_use]
    pub fn bright_office() -> Self {
        Self::d65_standard(500.0)
    }

    /// Whether this is considered a dark environment (< 20 lux).
    #[must_use]
    pub fn is_dark(&self) -> bool {
        self.illuminance_lux < 20.0
    }

    /// Whether this is considered a bright environment (> 200 lux).
    #[must_use]
    pub fn is_bright(&self) -> bool {
        self.illuminance_lux > 200.0
    }

    /// Compute the correlated color temperature (CCT) approximation in Kelvin.
    /// Uses McCamy's approximation formula.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn approximate_cct(&self) -> f64 {
        if self.chromaticity_y.abs() < 1e-12 {
            return 0.0;
        }
        let n = (self.chromaticity_x - 0.3320) / (self.chromaticity_y - 0.1858);
        -449.0 * n * n * n + 3525.0 * n * n - 6823.3 * n + 5520.33
    }
}

impl Default for AmbientLight {
    fn default() -> Self {
        Self::dim_living_room()
    }
}

/// Viewing environment classification used for display mapping decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ViewingEnvironment {
    /// Dark cinema / reference monitoring room.
    DarkCinema,
    /// Dim home theater or grading suite.
    DimRoom,
    /// Average living room with some ambient light.
    AverageRoom,
    /// Bright room with significant ambient light.
    BrightRoom,
    /// Outdoor / daylight viewing (e.g., mobile devices).
    Daylight,
}

impl ViewingEnvironment {
    /// Get the typical ambient illuminance for this environment in lux.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub const fn typical_lux(self) -> f64 {
        match self {
            Self::DarkCinema => 5.0,
            Self::DimRoom => 50.0,
            Self::AverageRoom => 150.0,
            Self::BrightRoom => 500.0,
            Self::Daylight => 5000.0,
        }
    }

    /// Classify a viewing environment from an ambient light measurement.
    #[must_use]
    pub fn classify(ambient: &AmbientLight) -> Self {
        let lux = ambient.illuminance_lux;
        if lux < 20.0 {
            Self::DarkCinema
        } else if lux < 100.0 {
            Self::DimRoom
        } else if lux < 300.0 {
            Self::AverageRoom
        } else if lux < 2000.0 {
            Self::BrightRoom
        } else {
            Self::Daylight
        }
    }

    /// Get the recommended surround factor for display mapping.
    /// Higher values indicate more aggressive tone mapping.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub const fn surround_factor(self) -> f64 {
        match self {
            Self::DarkCinema => 0.9,
            Self::DimRoom => 0.95,
            Self::AverageRoom => 1.0,
            Self::BrightRoom => 1.05,
            Self::Daylight => 1.15,
        }
    }

    /// Returns a human-readable label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::DarkCinema => "Dark Cinema",
            Self::DimRoom => "Dim Room",
            Self::AverageRoom => "Average Room",
            Self::BrightRoom => "Bright Room",
            Self::Daylight => "Daylight",
        }
    }
}

/// Surround luminance adaptation factor.
///
/// Describes how much the eye adapts to the surround luminance
/// relative to the display peak, affecting perceived contrast.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurroundAdaptation {
    /// Ratio of surround luminance to display peak (0.0 to 1.0).
    pub surround_ratio: f64,
    /// Adaptation factor applied to the EOTF curve.
    pub adaptation_factor: f64,
    /// Effective display gamma adjustment.
    pub gamma_adjustment: f64,
}

impl SurroundAdaptation {
    /// Compute surround adaptation from ambient light and display peak.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn compute(ambient: &AmbientLight, display_peak_nits: f64) -> Self {
        if display_peak_nits <= 0.0 {
            return Self {
                surround_ratio: 0.0,
                adaptation_factor: 1.0,
                gamma_adjustment: 0.0,
            };
        }

        // Approximate surround luminance from ambient illuminance
        // Typical reflective surfaces reflect ~20% of incident light
        let surround_luminance = ambient.illuminance_lux * 0.2 / std::f64::consts::PI;
        let surround_ratio = (surround_luminance / display_peak_nits).clamp(0.0, 1.0);

        // Compute adaptation factor (CIECAM02-based simplified model)
        let adaptation_factor = if surround_ratio < 0.2 {
            0.9 + surround_ratio * 0.5
        } else if surround_ratio < 0.5 {
            0.95 + (surround_ratio - 0.2) * 0.167
        } else {
            1.0
        };

        // Gamma adjustment to compensate for surround effect
        let gamma_adjustment = (1.0 - adaptation_factor) * 0.1;

        Self {
            surround_ratio,
            adaptation_factor,
            gamma_adjustment,
        }
    }
}

/// Complete ambient viewing environment descriptor.
#[derive(Debug, Clone)]
pub struct AmbientViewingEnvironment {
    /// Measured ambient light conditions.
    pub ambient_light: AmbientLight,
    /// Classified viewing environment.
    pub environment: ViewingEnvironment,
    /// Surround adaptation parameters.
    pub adaptation: SurroundAdaptation,
    /// Target display peak luminance in nits.
    pub display_peak_nits: f64,
    /// Target display minimum luminance in nits.
    pub display_min_nits: f64,
    /// Whether the environment is measured (true) or estimated (false).
    pub is_measured: bool,
    /// Optional environment label/description.
    pub description: Option<String>,
}

impl AmbientViewingEnvironment {
    /// Create a new ambient viewing environment descriptor.
    #[must_use]
    pub fn new(ambient: AmbientLight, display_peak_nits: f64, display_min_nits: f64) -> Self {
        let environment = ViewingEnvironment::classify(&ambient);
        let adaptation = SurroundAdaptation::compute(&ambient, display_peak_nits);
        Self {
            ambient_light: ambient,
            environment,
            adaptation,
            display_peak_nits,
            display_min_nits,
            is_measured: false,
            description: None,
        }
    }

    /// Create a standard home theater environment.
    #[must_use]
    pub fn home_theater() -> Self {
        Self::new(AmbientLight::dim_living_room(), 1000.0, 0.005)
    }

    /// Create a reference cinema environment.
    #[must_use]
    pub fn reference_cinema() -> Self {
        Self::new(AmbientLight::dark_cinema(), 108.0, 0.005)
    }

    /// The effective dynamic range in stops.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn dynamic_range_stops(&self) -> f64 {
        if self.display_min_nits <= 0.0 || self.display_peak_nits <= 0.0 {
            return 0.0;
        }
        (self.display_peak_nits / self.display_min_nits).log2()
    }

    /// The effective contrast ratio.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn contrast_ratio(&self) -> f64 {
        if self.display_min_nits <= 0.0 {
            return 0.0;
        }
        self.display_peak_nits / self.display_min_nits
    }

    /// Whether the environment is suitable for HDR mastering/review.
    #[must_use]
    pub fn is_suitable_for_mastering(&self) -> bool {
        self.ambient_light.is_dark()
            && self.display_peak_nits >= 1000.0
            && self.contrast_ratio() >= 100_000.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ambient_light_presets() {
        let dark = AmbientLight::dark_cinema();
        assert!(dark.is_dark());
        assert!(!dark.is_bright());

        let bright = AmbientLight::bright_office();
        assert!(bright.is_bright());
        assert!(!bright.is_dark());

        let dim = AmbientLight::dim_living_room();
        assert!(!dim.is_dark());
        assert!(!dim.is_bright());
    }

    #[test]
    fn test_ambient_light_d65() {
        let light = AmbientLight::d65_standard(100.0);
        assert!((light.chromaticity_x - 0.3127).abs() < 1e-6);
        assert!((light.chromaticity_y - 0.3290).abs() < 1e-6);
        assert!((light.illuminance_lux - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ambient_light_cct() {
        let d65 = AmbientLight::d65_standard(100.0);
        let cct = d65.approximate_cct();
        // D65 is approximately 6500K
        assert!(cct > 5000.0 && cct < 8000.0, "CCT was {cct}");
    }

    #[test]
    fn test_ambient_light_cct_zero_y() {
        let light = AmbientLight::new(100.0, 0.3, 0.0);
        assert!((light.approximate_cct() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ambient_light_default() {
        let light = AmbientLight::default();
        assert!((light.illuminance_lux - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_viewing_environment_classify() {
        assert_eq!(
            ViewingEnvironment::classify(&AmbientLight::dark_cinema()),
            ViewingEnvironment::DarkCinema
        );
        assert_eq!(
            ViewingEnvironment::classify(&AmbientLight::dim_living_room()),
            ViewingEnvironment::DimRoom
        );
        assert_eq!(
            ViewingEnvironment::classify(&AmbientLight::bright_office()),
            ViewingEnvironment::BrightRoom
        );
        assert_eq!(
            ViewingEnvironment::classify(&AmbientLight::d65_standard(10000.0)),
            ViewingEnvironment::Daylight
        );
    }

    #[test]
    fn test_viewing_environment_labels() {
        assert_eq!(ViewingEnvironment::DarkCinema.label(), "Dark Cinema");
        assert_eq!(ViewingEnvironment::DimRoom.label(), "Dim Room");
        assert_eq!(ViewingEnvironment::AverageRoom.label(), "Average Room");
        assert_eq!(ViewingEnvironment::BrightRoom.label(), "Bright Room");
        assert_eq!(ViewingEnvironment::Daylight.label(), "Daylight");
    }

    #[test]
    fn test_viewing_environment_surround_factor() {
        let dark = ViewingEnvironment::DarkCinema.surround_factor();
        let bright = ViewingEnvironment::Daylight.surround_factor();
        assert!(
            dark < bright,
            "Dark ({dark}) should have lower surround factor than daylight ({bright})"
        );
    }

    #[test]
    fn test_surround_adaptation_dark() {
        let ambient = AmbientLight::dark_cinema();
        let adapt = SurroundAdaptation::compute(&ambient, 1000.0);
        assert!(adapt.surround_ratio < 0.01);
        assert!(adapt.adaptation_factor < 1.0);
    }

    #[test]
    fn test_surround_adaptation_zero_peak() {
        let ambient = AmbientLight::dim_living_room();
        let adapt = SurroundAdaptation::compute(&ambient, 0.0);
        assert!((adapt.surround_ratio - 0.0).abs() < f64::EPSILON);
        assert!((adapt.adaptation_factor - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ambient_viewing_environment_home() {
        let env = AmbientViewingEnvironment::home_theater();
        assert_eq!(env.environment, ViewingEnvironment::DimRoom);
        assert!((env.display_peak_nits - 1000.0).abs() < f64::EPSILON);
        assert!(!env.is_measured);
    }

    #[test]
    fn test_ambient_viewing_environment_cinema() {
        let env = AmbientViewingEnvironment::reference_cinema();
        assert_eq!(env.environment, ViewingEnvironment::DarkCinema);
        assert!((env.display_peak_nits - 108.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_dynamic_range_stops() {
        let env = AmbientViewingEnvironment::new(AmbientLight::dark_cinema(), 1000.0, 0.001);
        let stops = env.dynamic_range_stops();
        // 1000/0.001 = 1_000_000, log2 ~= 20
        assert!(stops > 19.0 && stops < 21.0, "Got {stops} stops");
    }

    #[test]
    fn test_dynamic_range_stops_zero() {
        let env = AmbientViewingEnvironment::new(AmbientLight::dark_cinema(), 1000.0, 0.0);
        assert!((env.dynamic_range_stops() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_contrast_ratio() {
        let env = AmbientViewingEnvironment::new(AmbientLight::dark_cinema(), 1000.0, 0.01);
        assert!((env.contrast_ratio() - 100_000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_suitable_for_mastering() {
        let good = AmbientViewingEnvironment::new(AmbientLight::dark_cinema(), 4000.0, 0.005);
        assert!(good.is_suitable_for_mastering());

        let bad = AmbientViewingEnvironment::new(AmbientLight::bright_office(), 400.0, 0.1);
        assert!(!bad.is_suitable_for_mastering());
    }
}
