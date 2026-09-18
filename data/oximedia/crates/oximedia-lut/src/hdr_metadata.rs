//! HDR static and dynamic metadata types.
//!
//! This module provides HDR metadata structures for HDR10, HDR10+, Dolby Vision,
//! and HLG standards, including mastering display metadata (SMPTE ST 2086) and
//! content light level (CLL/FALL) information.

/// HDR standard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HdrStandard {
    /// HDR10 (PQ transfer, BT.2020 primaries).
    Hdr10,
    /// HDR10+ (HDR10 with dynamic metadata).
    Hdr10Plus,
    /// Dolby Vision (proprietary, but metadata is open).
    DolbyVision,
    /// HLG (Hybrid Log Gamma, for broadcast).
    Hlg,
}

impl HdrStandard {
    /// Get the transfer function used by this HDR standard.
    #[must_use]
    pub fn transfer_function(&self) -> HdrTransferFunction {
        match self {
            Self::Hdr10 | Self::Hdr10Plus | Self::DolbyVision => HdrTransferFunction::Pq,
            Self::Hlg => HdrTransferFunction::Hlg,
        }
    }

    /// Get the primary color space used by this HDR standard.
    #[must_use]
    pub fn color_space(&self) -> HdrColorSpace {
        match self {
            Self::Hdr10 | Self::Hdr10Plus | Self::Hlg => HdrColorSpace::Bt2020,
            Self::DolbyVision => HdrColorSpace::Bt2020,
        }
    }

    /// Whether this standard supports dynamic metadata.
    #[must_use]
    pub fn supports_dynamic_metadata(&self) -> bool {
        matches!(self, Self::Hdr10Plus | Self::DolbyVision)
    }

    /// Typical peak luminance in nits for this standard.
    #[must_use]
    pub fn max_luminance_nits(&self) -> f32 {
        match self {
            Self::Hdr10 => 1000.0,
            Self::Hdr10Plus => 4000.0,
            Self::DolbyVision => 10000.0,
            Self::Hlg => 1000.0,
        }
    }
}

/// Transfer function / OETF.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HdrTransferFunction {
    /// Perceptual Quantizer (ST 2084).
    Pq,
    /// Hybrid Log Gamma (ARIB STD-B67).
    Hlg,
    /// sRGB / BT.709 gamma (2.2).
    Gamma22,
    /// Film gamma (2.4).
    Gamma24,
    /// Linear light (no transfer function).
    Linear,
}

impl HdrTransferFunction {
    /// Apply the transfer function (linear light to display-encoded).
    ///
    /// Input: linear light in [0.0, 1.0].
    /// Output: encoded signal in [0.0, 1.0].
    #[must_use]
    pub fn encode(&self, linear: f32) -> f32 {
        match self {
            Self::Pq => pq_encode(linear),
            Self::Hlg => hlg_encode(linear),
            Self::Gamma22 => {
                if linear <= 0.0 {
                    0.0
                } else {
                    linear.powf(1.0 / 2.2)
                }
            }
            Self::Gamma24 => {
                if linear <= 0.0 {
                    0.0
                } else {
                    linear.powf(1.0 / 2.4)
                }
            }
            Self::Linear => linear,
        }
    }

    /// Inverse transfer function (display-encoded to linear light).
    ///
    /// Input: encoded signal in [0.0, 1.0].
    /// Output: linear light in [0.0, 1.0].
    #[must_use]
    pub fn decode(&self, encoded: f32) -> f32 {
        match self {
            Self::Pq => pq_decode(encoded),
            Self::Hlg => hlg_decode(encoded),
            Self::Gamma22 => {
                if encoded <= 0.0 {
                    0.0
                } else {
                    encoded.powf(2.2)
                }
            }
            Self::Gamma24 => {
                if encoded <= 0.0 {
                    0.0
                } else {
                    encoded.powf(2.4)
                }
            }
            Self::Linear => encoded,
        }
    }

    /// Human-readable name of this transfer function.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Pq => "PQ (ST 2084)",
            Self::Hlg => "HLG (ARIB STD-B67)",
            Self::Gamma22 => "Gamma 2.2",
            Self::Gamma24 => "Gamma 2.4",
            Self::Linear => "Linear",
        }
    }
}

// ============================================================================
// PQ (Perceptual Quantizer) Transfer Function - SMPTE ST 2084
// ============================================================================

/// PQ OETF: linear light [0, 1] → encoded [0, 1].
/// The reference is 10000 nits absolute luminance.
#[must_use]
fn pq_encode(linear: f32) -> f32 {
    const M1: f32 = 2610.0 / 16384.0;
    const M2: f32 = 2523.0 / 4096.0 * 128.0;
    const C1: f32 = 3424.0 / 4096.0;
    const C2: f32 = 2413.0 / 4096.0 * 32.0;
    const C3: f32 = 2392.0 / 4096.0 * 32.0;

    let y = linear.max(0.0).powf(M1);
    let num = C1 + C2 * y;
    let den = 1.0 + C3 * y;
    (num / den).powf(M2)
}

/// PQ EOTF: encoded [0, 1] → linear light [0, 1].
#[must_use]
fn pq_decode(encoded: f32) -> f32 {
    const M1: f32 = 2610.0 / 16384.0;
    const M2: f32 = 2523.0 / 4096.0 * 128.0;
    const C1: f32 = 3424.0 / 4096.0;
    const C2: f32 = 2413.0 / 4096.0 * 32.0;
    const C3: f32 = 2392.0 / 4096.0 * 32.0;

    let v = encoded.powf(1.0 / M2);
    let num = (v - C1).max(0.0);
    let den = C2 - C3 * v;
    if den.abs() < 1e-10 {
        return 0.0;
    }
    (num / den).powf(1.0 / M1)
}

// ============================================================================
// HLG (Hybrid Log-Gamma) Transfer Function - ARIB STD-B67
// ============================================================================

/// HLG OETF: linear light [0, 1] → encoded [0, 1].
#[must_use]
fn hlg_encode(linear: f32) -> f32 {
    const A: f32 = 0.178_832_77;
    const B: f32 = 0.284_668_92;
    const C: f32 = 0.559_910_7;

    let linear = linear.max(0.0);
    if linear <= 1.0 / 12.0 {
        (3.0 * linear).sqrt()
    } else {
        A * (12.0 * linear - B).ln() + C
    }
}

/// HLG EOTF: encoded [0, 1] → linear light [0, 1].
#[must_use]
fn hlg_decode(encoded: f32) -> f32 {
    const A: f32 = 0.178_832_77;
    const B: f32 = 0.284_668_92;
    const C: f32 = 0.559_910_7;

    let encoded = encoded.max(0.0);
    if encoded <= 0.5 {
        (encoded * encoded) / 3.0
    } else {
        (((encoded - C) / A).exp() + B) / 12.0
    }
}

// ============================================================================
// Color Space / Primaries
// ============================================================================

/// Color space / primaries for HDR metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HdrColorSpace {
    /// BT.709 / Rec.709 (HD video, sRGB).
    Bt709,
    /// BT.2020 / Rec.2020 (UHD/HDR video).
    Bt2020,
    /// DCI-P3 / Display P3.
    DisplayP3,
    /// ACES AP0 (ACES2065-1).
    Aces,
    /// ACES AP1 (`ACEScg`).
    AcesCg,
    /// sRGB (same primaries as BT.709).
    Srgb,
}

impl HdrColorSpace {
    /// Get the primary chromaticities as `[Rx, Ry, Gx, Gy, Bx, By, Wx, Wy]`.
    #[must_use]
    pub fn primaries(&self) -> [f32; 8] {
        match self {
            // Rx,    Ry,    Gx,    Gy,    Bx,    By,    Wx,       Wy
            Self::Bt709 | Self::Srgb => [0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.312_713, 0.329_002],
            Self::Bt2020 => [
                0.708, 0.292, 0.170, 0.797, 0.131, 0.046, 0.312_713, 0.329_002,
            ],
            Self::DisplayP3 => [
                0.680, 0.320, 0.265, 0.690, 0.150, 0.060, 0.312_713, 0.329_002,
            ],
            Self::Aces => [
                0.734_7, 0.265_3, 0.0, 1.0, 0.0001, -0.0770, 0.322_73, 0.329_02,
            ],
            Self::AcesCg => [0.713, 0.293, 0.165, 0.830, 0.128, 0.044, 0.322_73, 0.329_02],
        }
    }

    /// Get a 3x3 RGB to XYZ matrix (row-major).
    #[must_use]
    pub fn to_xyz_matrix(&self) -> [[f32; 3]; 3] {
        match self {
            Self::Bt709 | Self::Srgb => [
                [0.412_456_5, 0.357_576_1, 0.180_437_5],
                [0.212_672_9, 0.715_152_2, 0.072_174_9],
                [0.019_333_9, 0.119_192, 0.950_304_1],
            ],
            Self::Bt2020 => [
                [0.636_958_1, 0.144_616_9, 0.168_880_9],
                [0.262_700_2, 0.677_998_1, 0.059_301_7],
                [0.000_000_0, 0.028_072_7, 1.060_985],
            ],
            Self::DisplayP3 => [
                [0.486_570_9, 0.265_667_7, 0.198_217_3],
                [0.228_974_9, 0.691_738_5, 0.079_286_6],
                [0.000_000_0, 0.045_113_4, 1.043_944_4],
            ],
            Self::Aces => [
                [0.952_552_4, 0.000_000_0, 0.000_093_7],
                [0.343_966_4, 0.728_166_1, -0.072_132_5],
                [0.000_000_0, 0.000_000_0, 1.008_825_2],
            ],
            Self::AcesCg => [
                [0.662_454_2, 0.134_004_2, 0.156_187_7],
                [0.272_228_7, 0.674_081_8, 0.053_689_5],
                [-0.005_574_6, 0.004_060_7, 1.010_339_1],
            ],
        }
    }

    /// Get a 3x3 XYZ to RGB matrix (inverse of `to_xyz_matrix`).
    #[must_use]
    pub fn from_xyz_matrix(&self) -> [[f32; 3]; 3] {
        match self {
            Self::Bt709 | Self::Srgb => [
                [3.240_454_2, -1.537_138_5, -0.498_531_4],
                [-0.969_266, 1.876_010_8, 0.041_556_0],
                [0.055_643_4, -0.204_025_9, 1.057_225_2],
            ],
            Self::Bt2020 => [
                [1.716_651_2, -0.355_670_8, -0.253_366_8],
                [-0.666_684_4, 1.616_481_5, 0.015_768_5],
                [0.017_639_9, -0.042_771_1, 0.942_103_5],
            ],
            Self::DisplayP3 => [
                [2.493_497, -0.931_383_6, -0.402_710_8],
                [-0.829_488_6, 1.762_664_1, 0.023_624_7],
                [0.035_845_8, -0.076_172_4, 0.956_884_5],
            ],
            Self::Aces => [
                [1.049_811, 0.000_000_0, -0.000_097_5],
                [-0.495_903, 1.373_313, 0.098_240_0],
                [0.000_000_0, 0.000_000_0, 0.991_252],
            ],
            Self::AcesCg => [
                [1.641_023_4, -0.324_803_3, -0.236_424_7],
                [-0.663_662_9, 1.615_331_6, 0.016_756_3],
                [0.011_721_9, -0.008_284_4, 0.988_394_9],
            ],
        }
    }
}

// ============================================================================
// HDR10 Static Metadata (SMPTE ST 2086)
// ============================================================================

/// HDR10 static metadata (SMPTE ST 2086) describing the mastering display.
#[derive(Debug, Clone)]
pub struct MasteringDisplayMetadata {
    /// Red primary chromaticity (x, y).
    pub primaries_r: (f32, f32),
    /// Green primary chromaticity (x, y).
    pub primaries_g: (f32, f32),
    /// Blue primary chromaticity (x, y).
    pub primaries_b: (f32, f32),
    /// White point chromaticity (x, y).
    pub white_point: (f32, f32),
    /// Maximum mastering display luminance in nits (cd/m²).
    pub max_luminance: f32,
    /// Minimum mastering display luminance in nits.
    pub min_luminance: f32,
}

impl MasteringDisplayMetadata {
    /// Standard P3 D65 mastering display (common for HDR10 content).
    #[must_use]
    pub fn p3_d65() -> Self {
        Self {
            primaries_r: (0.680, 0.320),
            primaries_g: (0.265, 0.690),
            primaries_b: (0.150, 0.060),
            white_point: (0.312_7, 0.329_0),
            max_luminance: 1000.0,
            min_luminance: 0.0050,
        }
    }

    /// BT.2020 mastering display.
    #[must_use]
    pub fn bt2020() -> Self {
        Self {
            primaries_r: (0.708, 0.292),
            primaries_g: (0.170, 0.797),
            primaries_b: (0.131, 0.046),
            white_point: (0.312_7, 0.329_0),
            max_luminance: 1000.0,
            min_luminance: 0.0050,
        }
    }
}

// ============================================================================
// Content Light Level
// ============================================================================

/// Content Light Level metadata (`MaxCLL` / `MaxFALL`).
#[derive(Debug, Clone, Copy)]
pub struct ContentLightLevel {
    /// Maximum Content Light Level in nits (brightest pixel in the content).
    pub max_cll: u16,
    /// Maximum Frame Average Light Level in nits.
    pub max_fall: u16,
}

impl ContentLightLevel {
    /// Create a new `ContentLightLevel`.
    #[must_use]
    pub fn new(max_cll: u16, max_fall: u16) -> Self {
        Self { max_cll, max_fall }
    }

    /// Typical HDR10 content light levels (MaxCLL=1000, MaxFALL=400).
    #[must_use]
    pub fn typical_hdr10() -> Self {
        Self {
            max_cll: 1000,
            max_fall: 400,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transfer_function_pq_encode_decode() {
        for &x in &[0.0_f32, 0.1, 0.25, 0.5, 0.75, 1.0] {
            let encoded = HdrTransferFunction::Pq.encode(x);
            let decoded = HdrTransferFunction::Pq.decode(encoded);
            assert!(
                (decoded - x).abs() < 1e-4,
                "PQ round-trip failed for {x}: got {decoded}"
            );
        }
    }

    #[test]
    fn test_transfer_function_hlg_encode_decode() {
        for &x in &[0.0_f32, 0.1, 0.25, 0.5, 0.75, 1.0] {
            let encoded = HdrTransferFunction::Hlg.encode(x);
            let decoded = HdrTransferFunction::Hlg.decode(encoded);
            assert!(
                (decoded - x).abs() < 0.01,
                "HLG round-trip failed for {x}: got {decoded}"
            );
        }
    }

    #[test]
    fn test_color_space_primaries_bt709() {
        let primaries = HdrColorSpace::Bt709.primaries();
        // Rx ≈ 0.64, Ry ≈ 0.33
        assert!((primaries[0] - 0.64).abs() < 0.001, "Rx = {}", primaries[0]);
        assert!((primaries[1] - 0.33).abs() < 0.001, "Ry = {}", primaries[1]);
    }

    #[test]
    fn test_mastering_display_p3_d65() {
        let md = MasteringDisplayMetadata::p3_d65();
        assert!((md.max_luminance - 1000.0).abs() < 1.0);
    }

    #[test]
    fn test_content_light_level_typical() {
        let cll = ContentLightLevel::typical_hdr10();
        assert_eq!(cll.max_cll, 1000);
        assert_eq!(cll.max_fall, 400);
    }

    #[test]
    fn test_hdr_standard_supports_dynamic() {
        assert!(HdrStandard::Hdr10Plus.supports_dynamic_metadata());
        assert!(HdrStandard::DolbyVision.supports_dynamic_metadata());
        assert!(!HdrStandard::Hdr10.supports_dynamic_metadata());
        assert!(!HdrStandard::Hlg.supports_dynamic_metadata());
    }
}
