//! Professional color calibration and matching tools for `OxiMedia`.
//!
//! `oximedia-calibrate` provides comprehensive color calibration and matching
//! capabilities for professional video and image processing workflows. This includes:
//!
//! - **Camera Calibration**: `ColorChecker`-based camera profiling and characterization
//! - **Display Calibration**: Monitor calibration, gamma correction, and profiling
//! - **Color Matching**: Match colors across multiple cameras and devices
//! - **ICC Profile Generation**: Create ICC color profiles from measurements
//! - **ICC Profile Application**: Apply ICC profiles to images and video
//! - **LUT Generation**: Generate calibration LUTs from measurements
//! - **White Balance**: Advanced white balance algorithms and presets
//! - **Color Temperature**: Automatic color temperature detection and shifting
//! - **Gamut Mapping**: Map device gamut to working color space
//! - **Chromatic Adaptation**: Adapt colors to different illuminants
//!
//! # Example
//!
//! ```rust,ignore
//! use oximedia_calibrate::{
//!     camera::{ColorChecker, ColorCheckerType},
//!     white::WhiteBalancePreset,
//!     temp::estimate_color_temperature,
//! };
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Detect ColorChecker in an image
//! let checker = ColorChecker::detect_in_image(&image_data, ColorCheckerType::Classic24)?;
//!
//! // Generate camera profile
//! let profile = checker.generate_camera_profile()?;
//!
//! // Apply white balance
//! let balanced = WhiteBalancePreset::Daylight.apply_to_image(&image_data)?;
//!
//! // Estimate color temperature
//! let temp = estimate_color_temperature(&image_data)?;
//! # Ok(())
//! # }
//! ```
//!
//! # Features
//!
//! ## Camera Calibration
//!
//! - Automatic `ColorChecker` detection in images
//! - Patch extraction with subpixel accuracy
//! - Camera profile generation (ICC/LUT)
//! - Multi-illuminant calibration support
//! - Calibration verification and validation
//!
//! ## Display Calibration
//!
//! - Gamma curve measurement and calibration
//! - Display uniformity testing
//! - Monitor profiling for accurate color reproduction
//! - Display characterization
//!
//! ## Color Matching
//!
//! - Multi-camera color matching workflows
//! - Scene-to-scene color matching for continuity
//! - Match to reference target capabilities
//! - Color consistency verification
//!
//! ## ICC Profiles
//!
//! - ICC v2 and v4 profile generation
//! - ICC profile parsing and validation
//! - ICC profile application to images
//! - Profile inspection and analysis
//!
//! ## LUT Generation
//!
//! - Measurement-based LUT creation
//! - 1D and 3D calibration LUTs
//! - LUT verification and validation
//! - Interpolation quality assessment
//!
//! ## White Balance
//!
//! - Automatic white balance from scene analysis
//! - Standard presets (Daylight, Tungsten, Fluorescent, etc.)
//! - Custom white balance from reference patch
//! - Gray world and white patch algorithms
//!
//! ## Color Temperature
//!
//! - Automatic color temperature estimation
//! - Temperature shift application
//! - Kelvin to RGB conversion
//! - Illuminant D-series support
//!
//! ## Gamut Mapping
//!
//! - Device gamut to working space mapping
//! - Perceptual gamut mapping strategies
//! - Gamut compression algorithms
//! - Out-of-gamut color handling
//!
//! ## Chromatic Adaptation
//!
//! - Bradford chromatic adaptation transform
//! - Von Kries adaptation
//! - CAT02 adaptation (CIECAM02)
//! - Custom illuminant adaptation
//!
//! # `ColorChecker` Support
//!
//! - X-Rite `ColorChecker` Classic (24 patches)
//! - X-Rite `ColorChecker` Passport
//! - `Datacolor` `SpyderCheckr`
//! - Custom target support
//!
//! # Calibration Workflows
//!
//! ## Camera Profiling Workflow
//!
//! 1. Shoot `ColorChecker` under target lighting
//! 2. Detect `ColorChecker` in image
//! 3. Extract patch colors
//! 4. Generate camera ICC profile or LUT
//! 5. Apply calibration to footage
//! 6. Verify calibration accuracy
//!
//! ## Display Calibration Workflow
//!
//! 1. Measure display with colorimeter
//! 2. Generate gamma and uniformity profiles
//! 3. Create display ICC profile
//! 4. Apply profile to output pipeline
//! 5. Verify display accuracy
//!
//! ## Camera Matching Workflow
//!
//! 1. Calibrate primary camera (Camera A)
//! 2. Shoot matching target with Camera B
//! 3. Generate matching LUT/profile
//! 4. Apply to Camera B footage
//! 5. Verify color matching across cameras

#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::similar_names)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(dead_code)]

pub mod aces_calibration;
pub mod aging_model;
pub mod ambient_compensation;
pub mod batch_calibrate;
pub mod calibrate_report;
pub mod calibration_extras;
pub mod calibration_schedule;
pub mod camera;
pub mod chart_detection;
pub mod chromatic;
pub mod color_checker;
pub mod color_space;
pub mod delta_e;
pub mod display;
pub mod display_verify;
pub mod flare_correction;
pub mod gamut;
pub mod gamut_checker;
pub mod geometry;
pub mod hdr_calibration;
pub mod icc;
pub mod icc_profile;
pub mod lens_profile;
pub mod lut;
pub mod r#match;
pub mod metamerism;
pub mod monitor_calibration;
pub mod patch_extract;
pub mod printer_calibration;
pub mod spectral;
pub mod spectral_reconstruction;
pub mod temp;
pub mod temporal_uniformity;
pub mod test_chart;
pub mod uniformity;
pub mod white;
pub mod white_balance;

mod error;

pub use error::{CalibrationError, CalibrationResult};

/// RGB color value (normalized to 0.0-1.0 range).
pub type Rgb = [f64; 3];

/// RGBA color value (normalized to 0.0-1.0 range).
pub type Rgba = [f64; 4];

/// XYZ tristimulus value.
pub type Xyz = [f64; 3];

/// LAB color value (L*a*b* color space).
pub type Lab = [f64; 3];

/// 3x3 color matrix.
pub type Matrix3x3 = [[f64; 3]; 3];

/// 3x4 color matrix with offset.
pub type Matrix3x4 = [[f64; 4]; 3];

/// Standard illuminant types.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Illuminant {
    /// Standard Illuminant A (2856K, tungsten/incandescent).
    A,
    /// Standard Illuminant D50 (5000K, horizon daylight).
    D50,
    /// Standard Illuminant D55 (5500K, mid-morning daylight).
    D55,
    /// Standard Illuminant D65 (6500K, noon daylight).
    D65,
    /// Standard Illuminant D75 (7500K, north sky daylight).
    D75,
    /// Standard Illuminant E (equal energy).
    E,
    /// Fluorescent F2 (4200K, cool white fluorescent).
    F2,
    /// Fluorescent F7 (6500K, broad-band daylight fluorescent).
    F7,
    /// Fluorescent F11 (4000K, narrow-band white fluorescent).
    F11,
}

impl Illuminant {
    /// Get the XYZ tristimulus values for this illuminant (2° observer).
    #[must_use]
    pub const fn xyz(&self) -> Xyz {
        match self {
            Self::A => [1.098_51, 1.0, 0.355_85],
            Self::D50 => [0.964_22, 1.0, 0.825_21],
            Self::D55 => [0.956_85, 1.0, 0.921_69],
            Self::D65 => [0.950_47, 1.0, 1.088_83],
            Self::D75 => [0.949_72, 1.0, 1.226_38],
            Self::E => [1.0, 1.0, 1.0],
            Self::F2 => [0.991_44, 1.0, 0.678_09],
            Self::F7 => [0.950_41, 1.0, 1.086_14],
            Self::F11 => [1.009_62, 1.0, 0.643_65],
        }
    }

    /// Get the color temperature in Kelvin.
    #[must_use]
    pub const fn color_temperature(&self) -> u32 {
        match self {
            Self::A => 2856,
            Self::D50 => 5000,
            Self::D55 => 5500,
            Self::D65 => 6500,
            Self::D75 => 7500,
            Self::E => 5454,
            Self::F2 => 4200,
            Self::F7 => 6500,
            Self::F11 => 4000,
        }
    }
}

/// Observer angle for colorimetric calculations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Observer {
    /// CIE 1931 2° Standard Observer.
    Degree2,
    /// CIE 1964 10° Standard Observer.
    Degree10,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_illuminant_xyz() {
        let d65 = Illuminant::D65.xyz();
        assert!((d65[0] - 0.950_47).abs() < 1e-5);
        assert!((d65[1] - 1.0).abs() < 1e-10);
        assert!((d65[2] - 1.088_83).abs() < 1e-5);
    }

    #[test]
    fn test_illuminant_temperature() {
        assert_eq!(Illuminant::D65.color_temperature(), 6500);
        assert_eq!(Illuminant::A.color_temperature(), 2856);
        assert_eq!(Illuminant::D50.color_temperature(), 5000);
    }
}
