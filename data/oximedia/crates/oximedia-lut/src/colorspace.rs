//! Color space definitions and conversions.
//!
//! This module provides comprehensive color space support including:
//!
//! - Rec.709 (BT.709, HD)
//! - Rec.2020 (BT.2020, UHD)
//! - DCI-P3 (Digital Cinema)
//! - Adobe RGB
//! - sRGB
//! - `ProPhoto` RGB
//! - ACES AP0 (ACES2065-1)
//! - ACES AP1 (`ACEScg`)
//!
//! # Transfer Functions
//!
//! The module also handles various transfer functions (gamma curves):
//!
//! - Linear
//! - sRGB/Rec.709 gamma (2.2/2.4 with linear segment)
//! - Pure gamma (2.2, 2.4, 2.6)
//! - Rec.2020 gamma
//! - ACES (linear)

use crate::error::LutResult;
use crate::matrix::{self};
use crate::{Matrix3x3, Rgb, Xyz};

/// Color space definition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorSpace {
    /// sRGB (default for web and consumer displays).
    Srgb,
    /// Rec.709 (BT.709, HD video standard).
    Rec709,
    /// Rec.2020 (BT.2020, UHD/HDR video standard).
    Rec2020,
    /// DCI-P3 (Digital Cinema, D65 white point).
    DciP3,
    /// Adobe RGB (1998).
    AdobeRgb,
    /// `ProPhoto` RGB (ROMM RGB).
    ProPhotoRgb,
    /// ACES AP0 (ACES2065-1, scene-referred).
    AcesAp0,
    /// ACES AP1 (`ACEScg`, working space).
    AcesAp1,
    /// Linear light (no gamma).
    Linear,
}

/// Transfer function (EOTF/OETF).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TransferFunction {
    /// Linear (no gamma).
    Linear,
    /// sRGB transfer function (gamma 2.2 with linear segment).
    Srgb,
    /// Rec.709 transfer function (similar to sRGB).
    Rec709,
    /// Rec.2020 transfer function.
    Rec2020,
    /// Pure gamma curve.
    Gamma(f64),
    /// Perceptual Quantizer (PQ, SMPTE ST 2084).
    Pq,
    /// Hybrid Log-Gamma (HLG, ARIB STD-B67).
    Hlg,
}

impl ColorSpace {
    /// Get the RGB to XYZ conversion matrix for this color space.
    #[must_use]
    pub const fn rgb_to_xyz_matrix(&self) -> Matrix3x3 {
        match self {
            Self::Srgb | Self::Rec709 => matrix::RGB_TO_XYZ_REC709,
            Self::Rec2020 => matrix::RGB_TO_XYZ_REC2020,
            Self::DciP3 => matrix::RGB_TO_XYZ_DCIP3,
            Self::AdobeRgb => matrix::RGB_TO_XYZ_ADOBE,
            Self::ProPhotoRgb => PROPHOTO_RGB_TO_XYZ,
            Self::AcesAp0 => ACES_AP0_RGB_TO_XYZ,
            Self::AcesAp1 => ACES_AP1_RGB_TO_XYZ,
            Self::Linear => matrix::IDENTITY,
        }
    }

    /// Get the XYZ to RGB conversion matrix for this color space.
    #[must_use]
    pub const fn xyz_to_rgb_matrix(&self) -> Matrix3x3 {
        match self {
            Self::Srgb | Self::Rec709 => matrix::XYZ_TO_RGB_REC709,
            Self::Rec2020 => matrix::XYZ_TO_RGB_REC2020,
            Self::DciP3 => matrix::XYZ_TO_RGB_DCIP3,
            Self::AdobeRgb => matrix::XYZ_TO_RGB_ADOBE,
            Self::ProPhotoRgb => PROPHOTO_XYZ_TO_RGB,
            Self::AcesAp0 => ACES_AP0_XYZ_TO_RGB,
            Self::AcesAp1 => ACES_AP1_XYZ_TO_RGB,
            Self::Linear => matrix::IDENTITY,
        }
    }

    /// Get the default transfer function for this color space.
    #[must_use]
    pub const fn default_transfer_function(&self) -> TransferFunction {
        match self {
            Self::Srgb => TransferFunction::Srgb,
            Self::Rec709 => TransferFunction::Rec709,
            Self::Rec2020 => TransferFunction::Rec2020,
            Self::DciP3 => TransferFunction::Gamma(2.6),
            Self::AdobeRgb => TransferFunction::Gamma(2.2),
            Self::ProPhotoRgb => TransferFunction::Gamma(1.8),
            Self::AcesAp0 | Self::AcesAp1 | Self::Linear => TransferFunction::Linear,
        }
    }

    /// Convert RGB from this color space to another.
    ///
    /// # Errors
    ///
    /// Returns an error if the conversion fails.
    pub fn convert(&self, to: Self, rgb: &Rgb) -> LutResult<Rgb> {
        if self == &to {
            return Ok(*rgb);
        }

        // Convert to XYZ as intermediate
        let xyz = self.rgb_to_xyz(rgb);
        // Convert from XYZ to target
        Ok(to.xyz_to_rgb(&xyz))
    }

    /// Convert RGB to XYZ.
    #[must_use]
    pub fn rgb_to_xyz(&self, rgb: &Rgb) -> Xyz {
        matrix::rgb_to_xyz(rgb, &self.rgb_to_xyz_matrix())
    }

    /// Convert XYZ to RGB.
    #[must_use]
    pub fn xyz_to_rgb(&self, xyz: &Xyz) -> Rgb {
        matrix::xyz_to_rgb(xyz, &self.xyz_to_rgb_matrix())
    }
}

impl TransferFunction {
    /// Apply the EOTF (electro-optical transfer function) to convert from encoded to linear.
    #[must_use]
    pub fn eotf(&self, encoded: f64) -> f64 {
        match self {
            Self::Linear => encoded,
            Self::Srgb => srgb_eotf(encoded),
            Self::Rec709 | Self::Rec2020 => rec709_eotf(encoded), // Same as Rec.709
            Self::Gamma(gamma) => encoded.powf(*gamma),
            Self::Pq => pq_eotf(encoded),
            Self::Hlg => hlg_eotf(encoded),
        }
    }

    /// Apply the OETF (opto-electronic transfer function) to convert from linear to encoded.
    #[must_use]
    pub fn oetf(&self, linear: f64) -> f64 {
        match self {
            Self::Linear => linear,
            Self::Srgb => srgb_oetf(linear),
            Self::Rec709 | Self::Rec2020 => rec709_oetf(linear), // Same as Rec.709
            Self::Gamma(gamma) => linear.powf(1.0 / gamma),
            Self::Pq => pq_oetf(linear),
            Self::Hlg => hlg_oetf(linear),
        }
    }

    /// Apply EOTF to RGB color.
    #[must_use]
    pub fn eotf_rgb(&self, encoded: &Rgb) -> Rgb {
        [
            self.eotf(encoded[0]),
            self.eotf(encoded[1]),
            self.eotf(encoded[2]),
        ]
    }

    /// Apply OETF to RGB color.
    #[must_use]
    pub fn oetf_rgb(&self, linear: &Rgb) -> Rgb {
        [
            self.oetf(linear[0]),
            self.oetf(linear[1]),
            self.oetf(linear[2]),
        ]
    }
}

// ============================================================================
// Transfer Function Implementations
// ============================================================================

/// sRGB EOTF (gamma to linear).
#[must_use]
#[inline]
fn srgb_eotf(encoded: f64) -> f64 {
    if encoded <= 0.04045 {
        encoded / 12.92
    } else {
        ((encoded + 0.055) / 1.055).powf(2.4)
    }
}

/// sRGB OETF (linear to gamma).
#[must_use]
#[inline]
fn srgb_oetf(linear: f64) -> f64 {
    if linear <= 0.003_130_8 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    }
}

/// Rec.709 EOTF (same as sRGB but with slightly different constants).
#[must_use]
#[inline]
fn rec709_eotf(encoded: f64) -> f64 {
    if encoded < 0.081 {
        encoded / 4.5
    } else {
        ((encoded + 0.099) / 1.099).powf(1.0 / 0.45)
    }
}

/// Rec.709 OETF.
#[must_use]
#[inline]
fn rec709_oetf(linear: f64) -> f64 {
    if linear < 0.018 {
        linear * 4.5
    } else {
        1.099 * linear.powf(0.45) - 0.099
    }
}

/// PQ (Perceptual Quantizer) EOTF - SMPTE ST 2084.
#[must_use]
fn pq_eotf(encoded: f64) -> f64 {
    const M1: f64 = 2610.0 / 16384.0;
    const M2: f64 = 2523.0 / 4096.0 * 128.0;
    const C1: f64 = 3424.0 / 4096.0;
    const C2: f64 = 2413.0 / 4096.0 * 32.0;
    const C3: f64 = 2392.0 / 4096.0 * 32.0;

    let v = encoded.powf(1.0 / M2);
    let num = (v - C1).max(0.0);
    let den = C2 - C3 * v;
    (num / den).powf(1.0 / M1)
}

/// PQ (Perceptual Quantizer) OETF.
#[must_use]
fn pq_oetf(linear: f64) -> f64 {
    const M1: f64 = 2610.0 / 16384.0;
    const M2: f64 = 2523.0 / 4096.0 * 128.0;
    const C1: f64 = 3424.0 / 4096.0;
    const C2: f64 = 2413.0 / 4096.0 * 32.0;
    const C3: f64 = 2392.0 / 4096.0 * 32.0;

    let y = linear.powf(M1);
    let num = C1 + C2 * y;
    let den = 1.0 + C3 * y;
    (num / den).powf(M2)
}

/// HLG (Hybrid Log-Gamma) EOTF - ARIB STD-B67.
#[must_use]
fn hlg_eotf(encoded: f64) -> f64 {
    const A: f64 = 0.178_832_77;
    const B: f64 = 0.284_668_92;
    const C: f64 = 0.559_910_73;

    if encoded <= 0.5 {
        (encoded * encoded) / 3.0
    } else {
        (((encoded - C) / A).exp() + B) / 12.0
    }
}

/// HLG (Hybrid Log-Gamma) OETF.
#[must_use]
fn hlg_oetf(linear: f64) -> f64 {
    const A: f64 = 0.178_832_77;
    const B: f64 = 0.284_668_92;
    const C: f64 = 0.559_910_73;

    if linear <= 1.0 / 12.0 {
        (3.0 * linear).sqrt()
    } else {
        A * (12.0 * linear - B).ln() + C
    }
}

// ============================================================================
// Additional Color Space Matrices
// ============================================================================

/// `ProPhoto` RGB to XYZ matrix.
const PROPHOTO_RGB_TO_XYZ: Matrix3x3 = [
    [0.797_674_9, 0.135_191_7, 0.031_353_4],
    [0.288_040_2, 0.711_874_1, 0.000_085_7],
    [0.000_000_0, 0.000_000_0, 0.825_210_0],
];

/// XYZ to `ProPhoto` RGB matrix.
const PROPHOTO_XYZ_TO_RGB: Matrix3x3 = [
    [1.345_943_3, -0.255_607_5, -0.051_111_8],
    [-0.544_598_9, 1.508_167_3, 0.020_535_1],
    [0.000_000_0, 0.000_000_0, 1.211_812_8],
];

/// ACES AP0 to XYZ matrix.
const ACES_AP0_RGB_TO_XYZ: Matrix3x3 = [
    [0.952_552_395_9, 0.000_000_000_0, 0.000_093_678_6],
    [0.343_966_449_8, 0.728_166_096_6, -0.072_132_546_4],
    [0.000_000_000_0, 0.000_000_000_0, 1.008_825_184_4],
];

/// XYZ to ACES AP0 matrix.
const ACES_AP0_XYZ_TO_RGB: Matrix3x3 = [
    [1.049_811_017_5, 0.000_000_000_0, -0.000_097_484_5],
    [-0.495_903_023_1, 1.373_313_045_8, 0.098_240_036_1],
    [0.000_000_000_0, 0.000_000_000_0, 0.991_252_018_2],
];

/// ACES AP1 to XYZ matrix.
const ACES_AP1_RGB_TO_XYZ: Matrix3x3 = [
    [0.662_454_181_1, 0.134_004_206_5, 0.156_187_687_0],
    [0.272_228_716_8, 0.674_081_765_8, 0.053_689_517_4],
    [-0.005_574_649_5, 0.004_060_733_5, 1.010_339_100_3],
];

/// XYZ to ACES AP1 matrix.
const ACES_AP1_XYZ_TO_RGB: Matrix3x3 = [
    [1.641_023_379_7, -0.324_803_294_2, -0.236_424_695_2],
    [-0.663_662_858_7, 1.615_331_591_7, 0.016_756_347_7],
    [0.011_721_894_3, -0.008_284_442_0, 0.988_394_858_5],
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_srgb_gamma_round_trip() {
        let linear = 0.5;
        let encoded = srgb_oetf(linear);
        let decoded = srgb_eotf(encoded);
        assert!((linear - decoded).abs() < 1e-10);
    }

    #[test]
    fn test_rec709_gamma_round_trip() {
        let linear = 0.5;
        let encoded = rec709_oetf(linear);
        let decoded = rec709_eotf(encoded);
        assert!((linear - decoded).abs() < 1e-10);
    }

    #[test]
    fn test_colorspace_self_convert() {
        let rgb = [0.5, 0.3, 0.7];
        let result = ColorSpace::Rec709
            .convert(ColorSpace::Rec709, &rgb)
            .expect("should succeed in test");
        assert!((result[0] - rgb[0]).abs() < 1e-10);
        assert!((result[1] - rgb[1]).abs() < 1e-10);
        assert!((result[2] - rgb[2]).abs() < 1e-10);
    }

    #[test]
    fn test_colorspace_round_trip() {
        let rgb = [0.5, 0.3, 0.7];
        let rec2020 = ColorSpace::Rec709
            .convert(ColorSpace::Rec2020, &rgb)
            .expect("should succeed in test");
        let back = ColorSpace::Rec2020
            .convert(ColorSpace::Rec709, &rec2020)
            .expect("should succeed in test");
        assert!((rgb[0] - back[0]).abs() < 1e-6);
        assert!((rgb[1] - back[1]).abs() < 1e-6);
        assert!((rgb[2] - back[2]).abs() < 1e-6);
    }

    #[test]
    fn test_transfer_function_round_trip() {
        let linear = [0.5, 0.3, 0.7];
        let tf = TransferFunction::Srgb;
        let encoded = tf.oetf_rgb(&linear);
        let decoded = tf.eotf_rgb(&encoded);
        assert!((linear[0] - decoded[0]).abs() < 1e-10);
        assert!((linear[1] - decoded[1]).abs() < 1e-10);
        assert!((linear[2] - decoded[2]).abs() < 1e-10);
    }

    #[test]
    fn test_pq_round_trip() {
        let linear = 0.5;
        let encoded = pq_oetf(linear);
        let decoded = pq_eotf(encoded);
        assert!((linear - decoded).abs() < 1e-10);
    }

    #[test]
    fn test_hlg_round_trip() {
        let linear = 0.5;
        let encoded = hlg_oetf(linear);
        let decoded = hlg_eotf(encoded);
        assert!((linear - decoded).abs() < 0.01); // HLG has some precision loss
    }
}
