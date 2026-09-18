//! HDR color management utilities for `OxiMedia`.
//!
//! This module implements HDR transfer functions and metadata for professional
//! HDR workflows, including SMPTE ST 2084 (PQ) and BT.2100 (HLG).

#![allow(dead_code)]

/// HDR standard / delivery format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HdrStandard {
    /// HDR10 — static metadata, PQ curve, BT.2020.
    Hdr10,
    /// HLG BT.2100 — hybrid log-gamma, broadcast HDR.
    HlgBt2100,
    /// Dolby Vision — dynamic metadata, proprietary.
    DolbyVision,
    /// HDR10+ — dynamic metadata extension of HDR10.
    Hdr10Plus,
}

impl HdrStandard {
    /// Peak luminance in nits (cd/m²) for this standard.
    #[must_use]
    pub fn peak_luminance_nits(&self) -> u32 {
        match self {
            Self::Hdr10 => 10_000,
            Self::HlgBt2100 => 1_000,
            Self::DolbyVision => 10_000,
            Self::Hdr10Plus => 4_000,
        }
    }

    /// Color gamut used by this standard.
    #[must_use]
    pub fn color_gamut(&self) -> &str {
        match self {
            Self::Hdr10 | Self::HlgBt2100 | Self::Hdr10Plus => "BT.2020",
            Self::DolbyVision => "BT.2020",
        }
    }

    /// Electro-optical transfer function name.
    #[must_use]
    pub fn transfer_function(&self) -> &str {
        match self {
            Self::Hdr10 | Self::Hdr10Plus => "SMPTE ST 2084 (PQ)",
            Self::HlgBt2100 => "BT.2100 HLG",
            Self::DolbyVision => "SMPTE ST 2084 (PQ)",
        }
    }
}

/// Perceptual Quantizer (PQ) transfer function — SMPTE ST 2084.
///
/// Operates with normalized signal `[0, 1]` and absolute luminance in nits.
pub struct Pq;

impl Pq {
    // SMPTE ST 2084 constants
    const M1: f64 = 2610.0 / 16384.0;
    const M2: f64 = 2523.0 / 4096.0 * 128.0;
    const C1: f64 = 3424.0 / 4096.0;
    const C2: f64 = 2413.0 / 4096.0 * 32.0;
    const C3: f64 = 2392.0 / 4096.0 * 32.0;

    /// Electro-optical transfer function: signal value → absolute luminance (nits / 10 000).
    ///
    /// `v` is the normalized signal in `[0, 1]`.
    #[must_use]
    pub fn eotf(v: f64) -> f64 {
        let v = v.max(0.0).min(1.0);
        let vp = v.powf(1.0 / Self::M2);
        let num = (vp - Self::C1).max(0.0);
        let den = Self::C2 - Self::C3 * vp;
        (num / den).powf(1.0 / Self::M1)
    }

    /// Opto-electronic transfer function: absolute luminance (nits / 10 000) → signal.
    ///
    /// `l` is the relative linear luminance (nits divided by 10 000).
    #[must_use]
    pub fn oetf(l: f64) -> f64 {
        let l = l.max(0.0);
        let lp = l.powf(Self::M1);
        let num = Self::C1 + Self::C2 * lp;
        let den = 1.0 + Self::C3 * lp;
        (num / den).powf(Self::M2)
    }
}

/// Hybrid Log-Gamma (HLG) transfer function — ITU-R BT.2100.
pub struct Hlg;

impl Hlg {
    const A: f64 = 0.178_832_77;
    const B: f64 = 0.284_668_92;
    const C: f64 = 0.559_910_73;
    const GAMMA: f64 = 1.2;

    /// Electro-optical transfer function: HLG signal → scene-referred linear.
    ///
    /// `v` is the normalized signal in `[0, 1]`.
    #[must_use]
    pub fn eotf(v: f64) -> f64 {
        let v = v.max(0.0).min(1.0);
        if v <= 0.5 {
            v * v / 3.0
        } else {
            (((v - Self::C) / Self::A).exp() + Self::B) / 12.0
        }
    }

    /// Opto-electronic transfer function: scene-referred linear → HLG signal.
    ///
    /// `l` is the normalized linear scene luminance in `[0, 1]`.
    #[must_use]
    pub fn oetf(l: f64) -> f64 {
        let l = l.max(0.0);
        if l <= 1.0 / 12.0 {
            (3.0 * l).sqrt()
        } else {
            Self::A * (12.0 * l - Self::B).ln() + Self::C
        }
    }
}

/// HDR static metadata (SMPTE ST 2086 / CTA-861-G).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HdrMetadata {
    /// Maximum Content Light Level (CLL) in nits.
    pub max_cll: u16,
    /// Maximum Frame-Average Light Level (FALL) in nits.
    pub max_fall: u16,
    /// Mastering display primaries as `[[x, y]; 3]` in units of 0.00002.
    pub display_primaries: [[u16; 2]; 3],
    /// White point as `[x, y]` in units of 0.00002.
    pub white_point: [u16; 2],
}

impl HdrMetadata {
    /// Returns `true` if the metadata values are consistent.
    ///
    /// `max_fall` must not exceed `max_cll`, and both must be non-zero.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.max_cll > 0 && self.max_fall > 0 && self.max_fall <= self.max_cll
    }
}

/// Converts between PQ and HLG representations via the linear domain.
pub struct HdrConverter;

impl HdrConverter {
    /// Converts a PQ-encoded signal to an HLG-encoded signal.
    ///
    /// The conversion is: PQ → linear (nits / 10 000) → HLG oetf.
    #[must_use]
    pub fn pq_to_hlg(pq: f64) -> f64 {
        let linear = Pq::eotf(pq);
        Hlg::oetf(linear)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hdr_standard_peak_luminance() {
        assert_eq!(HdrStandard::Hdr10.peak_luminance_nits(), 10_000);
        assert_eq!(HdrStandard::HlgBt2100.peak_luminance_nits(), 1_000);
        assert_eq!(HdrStandard::DolbyVision.peak_luminance_nits(), 10_000);
        assert_eq!(HdrStandard::Hdr10Plus.peak_luminance_nits(), 4_000);
    }

    #[test]
    fn test_hdr_standard_gamut() {
        assert_eq!(HdrStandard::Hdr10.color_gamut(), "BT.2020");
        assert_eq!(HdrStandard::HlgBt2100.color_gamut(), "BT.2020");
    }

    #[test]
    fn test_hdr_standard_transfer_function() {
        assert_eq!(HdrStandard::Hdr10.transfer_function(), "SMPTE ST 2084 (PQ)");
        assert_eq!(HdrStandard::HlgBt2100.transfer_function(), "BT.2100 HLG");
    }

    #[test]
    fn test_pq_eotf_black() {
        // Signal 0 → luminance 0
        let l = Pq::eotf(0.0);
        assert!(l.abs() < 1e-6, "PQ eotf(0) = {l}");
    }

    #[test]
    fn test_pq_eotf_white() {
        // Signal 1 → ~1.0 (10 000 nits / 10 000)
        let l = Pq::eotf(1.0);
        assert!((l - 1.0).abs() < 0.01, "PQ eotf(1) = {l}");
    }

    #[test]
    fn test_pq_oetf_roundtrip() {
        let original = 0.5_f64;
        let encoded = Pq::oetf(original);
        let decoded = Pq::eotf(encoded);
        assert!(
            (decoded - original).abs() < 1e-5,
            "PQ roundtrip failed: {decoded}"
        );
    }

    #[test]
    fn test_hlg_eotf_zero() {
        let l = Hlg::eotf(0.0);
        assert!(l.abs() < 1e-9, "HLG eotf(0) = {l}");
    }

    #[test]
    fn test_hlg_oetf_roundtrip_low() {
        let original = 0.05_f64;
        let encoded = Hlg::oetf(original);
        let decoded = Hlg::eotf(encoded);
        assert!(
            (decoded - original).abs() < 1e-5,
            "HLG roundtrip low: {decoded}"
        );
    }

    #[test]
    fn test_hlg_oetf_roundtrip_high() {
        let original = 0.5_f64;
        let encoded = Hlg::oetf(original);
        let decoded = Hlg::eotf(encoded);
        assert!(
            (decoded - original).abs() < 1e-5,
            "HLG roundtrip high: {decoded}"
        );
    }

    #[test]
    fn test_hdr_metadata_valid() {
        let m = HdrMetadata {
            max_cll: 1000,
            max_fall: 400,
            display_primaries: [[34_000, 16_000], [13_250, 34_500], [7_500, 3_000]],
            white_point: [15_635, 16_450],
        };
        assert!(m.is_valid());
    }

    #[test]
    fn test_hdr_metadata_invalid_zero_cll() {
        let m = HdrMetadata {
            max_cll: 0,
            max_fall: 0,
            display_primaries: [[34_000, 16_000], [13_250, 34_500], [7_500, 3_000]],
            white_point: [15_635, 16_450],
        };
        assert!(!m.is_valid());
    }

    #[test]
    fn test_hdr_metadata_invalid_fall_exceeds_cll() {
        let m = HdrMetadata {
            max_cll: 400,
            max_fall: 1000,
            display_primaries: [[34_000, 16_000], [13_250, 34_500], [7_500, 3_000]],
            white_point: [15_635, 16_450],
        };
        assert!(!m.is_valid());
    }

    #[test]
    fn test_pq_to_hlg_monotone() {
        let v1 = HdrConverter::pq_to_hlg(0.2);
        let v2 = HdrConverter::pq_to_hlg(0.5);
        assert!(v1 < v2, "pq_to_hlg should be monotonically increasing");
    }

    #[test]
    fn test_pq_to_hlg_zero() {
        let v = HdrConverter::pq_to_hlg(0.0);
        assert!(v.abs() < 1e-6, "pq_to_hlg(0) = {v}");
    }
}
