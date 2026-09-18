#![allow(dead_code)]
//! ACES configuration and pipeline preset management.
//!
//! Provides structures for building complete ACES (Academy Color Encoding
//! System) color pipelines including IDT selection, Look Modification
//! Transforms (LMTs), Reference Rendering Transform (RRT), and Output
//! Device Transform (ODT) configuration.
//!
//! # Reference
//!
//! - ACES Technical Bulletin TB-2014-002 (ACES Overview)
//! - SMPTE ST 2065-1 (ACES2065-1)
//! - Academy S-2014-003 (ACEScc)

use std::fmt;

// ─── ACES Version ───────────────────────────────────────────────────────────

/// ACES system version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AcesVersion {
    /// ACES 1.0 — original release.
    V1_0,
    /// ACES 1.1 — minor fixes, ACEScct introduction.
    V1_1,
    /// ACES 1.2 — improved HDR output transforms.
    V1_2,
    /// ACES 1.3 — latest release with expanded gamut mapping.
    V1_3,
}

impl AcesVersion {
    /// Returns the version string (e.g., `"1.3"`).
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::V1_0 => "1.0",
            Self::V1_1 => "1.1",
            Self::V1_2 => "1.2",
            Self::V1_3 => "1.3",
        }
    }
}

impl fmt::Display for AcesVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ACES {}", self.as_str())
    }
}

// ─── ODT Target ─────────────────────────────────────────────────────────────

/// Output Device Transform target display type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OdtTarget {
    /// Standard dynamic range — sRGB / Rec.709 display (100 nits peak).
    Sdr709,
    /// DCI-P3 cinema projection (48 nits).
    DciP3,
    /// HDR10 — PQ / Rec.2020, 1000-nit peak.
    Hdr10_1000,
    /// HDR10 — PQ / Rec.2020, 4000-nit peak.
    Hdr10_4000,
    /// Dolby Vision cinema display.
    DolbyVisionCinema,
    /// HLG — Rec.2020, 1000-nit system gamma.
    Hlg1000,
}

impl OdtTarget {
    /// Human-readable label for this ODT target.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Sdr709 => "SDR Rec.709 (100 nits)",
            Self::DciP3 => "DCI-P3 Cinema (48 nits)",
            Self::Hdr10_1000 => "HDR10 PQ (1000 nits)",
            Self::Hdr10_4000 => "HDR10 PQ (4000 nits)",
            Self::DolbyVisionCinema => "Dolby Vision Cinema",
            Self::Hlg1000 => "HLG (1000 nits)",
        }
    }

    /// Nominal peak luminance in nits for this target.
    #[must_use]
    pub fn peak_luminance_nits(self) -> f64 {
        match self {
            Self::Sdr709 => 100.0,
            Self::DciP3 => 48.0,
            Self::Hdr10_1000 => 1000.0,
            Self::Hdr10_4000 => 4000.0,
            Self::DolbyVisionCinema => 108.0,
            Self::Hlg1000 => 1000.0,
        }
    }

    /// Returns `true` if this is an HDR output.
    #[must_use]
    pub fn is_hdr(self) -> bool {
        matches!(
            self,
            Self::Hdr10_1000 | Self::Hdr10_4000 | Self::DolbyVisionCinema | Self::Hlg1000
        )
    }
}

impl fmt::Display for OdtTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

// ─── IDT Source ─────────────────────────────────────────────────────────────

/// Input Device Transform source camera / format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IdtSource {
    /// ARRI Alexa (LogC3 → ACES).
    ArriLogC3,
    /// ARRI Alexa 35 (LogC4 → ACES).
    ArriLogC4,
    /// RED `IPP2` Wide Gamut (Log3G10 → ACES).
    RedWideGamut,
    /// Sony S-Gamut3.Cine / S-Log3 → ACES.
    SonySLog3,
    /// Canon Cinema Gamut / Canon Log 2 → ACES.
    CanonLog2,
    /// Panasonic V-Gamut / V-Log → ACES.
    PanasonicVLog,
    /// Generic sRGB / Rec.709 input.
    Srgb,
}

impl IdtSource {
    /// Human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::ArriLogC3 => "ARRI LogC3",
            Self::ArriLogC4 => "ARRI LogC4",
            Self::RedWideGamut => "RED Wide Gamut / Log3G10",
            Self::SonySLog3 => "Sony S-Log3 / S-Gamut3.Cine",
            Self::CanonLog2 => "Canon Log 2 / Cinema Gamut",
            Self::PanasonicVLog => "Panasonic V-Log / V-Gamut",
            Self::Srgb => "sRGB / Rec.709",
        }
    }

    /// Returns `true` if the source encoding is logarithmic.
    #[must_use]
    pub fn is_log_encoded(self) -> bool {
        !matches!(self, Self::Srgb)
    }
}

impl fmt::Display for IdtSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

// ─── LMT ────────────────────────────────────────────────────────────────────

/// Look Modification Transform descriptor.
///
/// LMTs sit between the IDT and the RRT, applying creative colour grading
/// in the ACES linear-light domain.
#[derive(Debug, Clone)]
pub struct LmtDescriptor {
    /// Name of the look (e.g., "Warm Sunset", "Desaturated").
    pub name: String,
    /// Exposure offset in stops applied before the LMT matrix.
    pub exposure_offset: f64,
    /// 3x3 colour matrix (row-major, identity if unused).
    pub matrix: [[f64; 3]; 3],
    /// Per-channel saturation multiplier.
    pub saturation: f64,
    /// Enabled flag.
    pub enabled: bool,
}

impl Default for LmtDescriptor {
    fn default() -> Self {
        Self {
            name: String::new(),
            exposure_offset: 0.0,
            matrix: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            saturation: 1.0,
            enabled: true,
        }
    }
}

impl LmtDescriptor {
    /// Create a new LMT with the given name.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Self::default()
        }
    }

    /// Returns `true` if this LMT is effectively a no-op.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        if !self.enabled {
            return true;
        }
        let id = [[1.0_f64, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        self.exposure_offset.abs() < 1e-9
            && (self.saturation - 1.0).abs() < 1e-9
            && self.matrix == id
    }

    /// Apply this LMT to a linear-light RGB triplet.
    #[must_use]
    pub fn apply(&self, rgb: [f64; 3]) -> [f64; 3] {
        if !self.enabled {
            return rgb;
        }
        // Exposure
        let gain = 2.0_f64.powf(self.exposure_offset);
        let r = rgb[0] * gain;
        let g = rgb[1] * gain;
        let b = rgb[2] * gain;
        // Matrix
        let mr = self.matrix[0][0] * r + self.matrix[0][1] * g + self.matrix[0][2] * b;
        let mg = self.matrix[1][0] * r + self.matrix[1][1] * g + self.matrix[1][2] * b;
        let mb = self.matrix[2][0] * r + self.matrix[2][1] * g + self.matrix[2][2] * b;
        // Saturation
        let luma = 0.2126 * mr + 0.7152 * mg + 0.0722 * mb;
        let s = self.saturation;
        [
            luma + s * (mr - luma),
            luma + s * (mg - luma),
            luma + s * (mb - luma),
        ]
    }
}

// ─── ACES Config ────────────────────────────────────────────────────────────

/// A complete ACES pipeline configuration.
///
/// Describes the full chain: IDT → (optional LMTs) → working space → RRT → ODT.
#[derive(Debug, Clone)]
pub struct AcesConfig {
    /// ACES system version.
    pub version: AcesVersion,
    /// Input device transform source.
    pub idt: IdtSource,
    /// Look modification transforms (applied in order).
    pub lmts: Vec<LmtDescriptor>,
    /// Output device transform target.
    pub odt: OdtTarget,
    /// Global exposure compensation (stops).
    pub global_exposure: f64,
    /// Enable gamut compression before the ODT.
    pub gamut_compress: bool,
    /// Reference white luminance for the viewing environment (nits).
    pub reference_white_nits: f64,
}

impl Default for AcesConfig {
    fn default() -> Self {
        Self {
            version: AcesVersion::V1_3,
            idt: IdtSource::Srgb,
            lmts: Vec::new(),
            odt: OdtTarget::Sdr709,
            global_exposure: 0.0,
            gamut_compress: true,
            reference_white_nits: 100.0,
        }
    }
}

impl AcesConfig {
    /// Create a new ACES config with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: set IDT source.
    #[must_use]
    pub fn with_idt(mut self, idt: IdtSource) -> Self {
        self.idt = idt;
        self
    }

    /// Builder: set ODT target.
    #[must_use]
    pub fn with_odt(mut self, odt: OdtTarget) -> Self {
        self.odt = odt;
        self
    }

    /// Builder: add an LMT.
    #[must_use]
    pub fn with_lmt(mut self, lmt: LmtDescriptor) -> Self {
        self.lmts.push(lmt);
        self
    }

    /// Builder: set global exposure offset (stops).
    #[must_use]
    pub fn with_exposure(mut self, stops: f64) -> Self {
        self.global_exposure = stops;
        self
    }

    /// Builder: enable or disable gamut compression.
    #[must_use]
    pub fn with_gamut_compress(mut self, enable: bool) -> Self {
        self.gamut_compress = enable;
        self
    }

    /// Returns `true` if the pipeline includes any active LMTs.
    #[must_use]
    pub fn has_active_lmts(&self) -> bool {
        self.lmts.iter().any(|l| l.enabled && !l.is_identity())
    }

    /// Returns `true` if the ODT targets an HDR display.
    #[must_use]
    pub fn is_hdr_output(&self) -> bool {
        self.odt.is_hdr()
    }

    /// Validate the configuration for obvious issues.
    #[must_use]
    pub fn validate(&self) -> Vec<String> {
        let mut issues = Vec::new();
        if self.reference_white_nits <= 0.0 {
            issues.push("reference_white_nits must be positive".into());
        }
        if self.odt.is_hdr() && self.reference_white_nits > self.odt.peak_luminance_nits() {
            issues.push(format!(
                "reference_white_nits ({}) exceeds ODT peak ({})",
                self.reference_white_nits,
                self.odt.peak_luminance_nits()
            ));
        }
        issues
    }

    /// Summary description of the pipeline.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "{} | IDT: {} | ODT: {} | LMTs: {}",
            self.version,
            self.idt,
            self.odt,
            self.lmts.len()
        )
    }
}

// ─── Preset configs ─────────────────────────────────────────────────────────

/// Create a preset for cinema dailies (ARRI LogC3 → P3 cinema).
#[must_use]
pub fn preset_cinema_dailies() -> AcesConfig {
    AcesConfig::new()
        .with_idt(IdtSource::ArriLogC3)
        .with_odt(OdtTarget::DciP3)
}

/// Create a preset for HDR10 deliverable (generic sRGB in → HDR10 1000 nit out).
#[must_use]
pub fn preset_hdr10_delivery() -> AcesConfig {
    AcesConfig::new()
        .with_idt(IdtSource::Srgb)
        .with_odt(OdtTarget::Hdr10_1000)
}

/// Create a preset for SDR broadcast (Rec.709).
#[must_use]
pub fn preset_sdr_broadcast() -> AcesConfig {
    AcesConfig::new()
        .with_idt(IdtSource::Srgb)
        .with_odt(OdtTarget::Sdr709)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aces_version_display() {
        assert_eq!(format!("{}", AcesVersion::V1_3), "ACES 1.3");
        assert_eq!(AcesVersion::V1_0.as_str(), "1.0");
    }

    #[test]
    fn test_odt_target_labels() {
        for odt in [
            OdtTarget::Sdr709,
            OdtTarget::DciP3,
            OdtTarget::Hdr10_1000,
            OdtTarget::Hdr10_4000,
            OdtTarget::DolbyVisionCinema,
            OdtTarget::Hlg1000,
        ] {
            assert!(!odt.label().is_empty());
        }
    }

    #[test]
    fn test_odt_peak_luminance() {
        assert!((OdtTarget::Sdr709.peak_luminance_nits() - 100.0).abs() < 1e-9);
        assert!((OdtTarget::Hdr10_4000.peak_luminance_nits() - 4000.0).abs() < 1e-9);
    }

    #[test]
    fn test_odt_is_hdr() {
        assert!(!OdtTarget::Sdr709.is_hdr());
        assert!(!OdtTarget::DciP3.is_hdr());
        assert!(OdtTarget::Hdr10_1000.is_hdr());
        assert!(OdtTarget::Hlg1000.is_hdr());
    }

    #[test]
    fn test_idt_source_labels() {
        for idt in [
            IdtSource::ArriLogC3,
            IdtSource::ArriLogC4,
            IdtSource::RedWideGamut,
            IdtSource::SonySLog3,
            IdtSource::CanonLog2,
            IdtSource::PanasonicVLog,
            IdtSource::Srgb,
        ] {
            assert!(!idt.label().is_empty());
        }
    }

    #[test]
    fn test_idt_is_log_encoded() {
        assert!(IdtSource::ArriLogC3.is_log_encoded());
        assert!(!IdtSource::Srgb.is_log_encoded());
    }

    #[test]
    fn test_lmt_default_is_identity() {
        let lmt = LmtDescriptor::default();
        assert!(lmt.is_identity());
    }

    #[test]
    fn test_lmt_disabled_is_identity() {
        let mut lmt = LmtDescriptor::new("Test");
        lmt.enabled = false;
        lmt.saturation = 2.0;
        assert!(lmt.is_identity());
    }

    #[test]
    fn test_lmt_apply_identity() {
        let lmt = LmtDescriptor::default();
        let rgb = [0.5, 0.3, 0.2];
        let out = lmt.apply(rgb);
        assert!((out[0] - 0.5).abs() < 1e-9);
        assert!((out[1] - 0.3).abs() < 1e-9);
        assert!((out[2] - 0.2).abs() < 1e-9);
    }

    #[test]
    fn test_lmt_apply_exposure() {
        let mut lmt = LmtDescriptor::new("Bright");
        lmt.exposure_offset = 1.0; // +1 stop = 2x gain
        let rgb = [0.25, 0.25, 0.25];
        let out = lmt.apply(rgb);
        assert!((out[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_aces_config_default() {
        let cfg = AcesConfig::new();
        assert_eq!(cfg.version, AcesVersion::V1_3);
        assert_eq!(cfg.idt, IdtSource::Srgb);
        assert_eq!(cfg.odt, OdtTarget::Sdr709);
        assert!(cfg.lmts.is_empty());
    }

    #[test]
    fn test_aces_config_builder() {
        let cfg = AcesConfig::new()
            .with_idt(IdtSource::ArriLogC4)
            .with_odt(OdtTarget::Hdr10_1000)
            .with_exposure(0.5)
            .with_gamut_compress(false);
        assert_eq!(cfg.idt, IdtSource::ArriLogC4);
        assert_eq!(cfg.odt, OdtTarget::Hdr10_1000);
        assert!((cfg.global_exposure - 0.5).abs() < 1e-12);
        assert!(!cfg.gamut_compress);
    }

    #[test]
    fn test_aces_config_has_active_lmts() {
        let cfg = AcesConfig::new();
        assert!(!cfg.has_active_lmts());
        let mut lmt = LmtDescriptor::new("Look");
        lmt.saturation = 1.5;
        let cfg = cfg.with_lmt(lmt);
        assert!(cfg.has_active_lmts());
    }

    #[test]
    fn test_aces_config_validate_ok() {
        let cfg = AcesConfig::new();
        let issues = cfg.validate();
        assert!(issues.is_empty(), "unexpected issues: {issues:?}");
    }

    #[test]
    fn test_aces_config_validate_bad_white() {
        let mut cfg = AcesConfig::new();
        cfg.reference_white_nits = -10.0;
        let issues = cfg.validate();
        assert!(!issues.is_empty());
    }

    #[test]
    fn test_preset_cinema_dailies() {
        let cfg = preset_cinema_dailies();
        assert_eq!(cfg.idt, IdtSource::ArriLogC3);
        assert_eq!(cfg.odt, OdtTarget::DciP3);
    }

    #[test]
    fn test_preset_hdr10() {
        let cfg = preset_hdr10_delivery();
        assert!(cfg.is_hdr_output());
    }

    #[test]
    fn test_preset_sdr_broadcast() {
        let cfg = preset_sdr_broadcast();
        assert!(!cfg.is_hdr_output());
    }

    #[test]
    fn test_summary_not_empty() {
        let cfg = AcesConfig::new();
        assert!(!cfg.summary().is_empty());
    }
}
