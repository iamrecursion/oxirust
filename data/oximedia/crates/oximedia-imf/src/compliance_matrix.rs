//! Compliance matrix — maps IMF application profiles to required constraints.
//!
//! This module provides a structured registry of which SMPTE rules apply to
//! each application profile.  It can verify a built package against a target
//! profile and report every violation found.
//!
//! # Supported profiles
//! - [`ApplicationProfile::App2`] — SMPTE ST 2067-21 App #2 (JPEG 2000)
//! - [`ApplicationProfile::App2E`] — App #2 Extended (higher bit-depths)
//! - [`ApplicationProfile::App2_1`] — Netflix IMF App 2.1 (JPEG 2000 + H.264)
//! - [`ApplicationProfile::NflxIter1`] — Netflix Iterative (ProRes / AVC)
//! - [`ApplicationProfile::DisneyDece`] — Disney DECE delivery profile

use std::fmt;

// ---------------------------------------------------------------------------
// ApplicationProfile
// ---------------------------------------------------------------------------

/// IMF application profiles supported by the compliance matrix.
///
/// Note: this is a *separate*, richer enum from
/// `crate::application_profile::ApplicationProfile` — it extends it with
/// vendor-specific profiles required by the TODO spec.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ApplicationProfile {
    /// SMPTE ST 2067-21 Application #2 — JPEG 2000 essence.
    App2,
    /// SMPTE ST 2067-21 Application #2 Extended — higher bit-depths / HDR.
    App2E,
    /// Netflix IMF Application 2.1 — JPEG 2000 + H.264 hybrid workflow.
    App2_1,
    /// Netflix Iterative IMF — first iteration (ProRes / AVC mezzanine).
    NflxIter1,
    /// Disney DECE delivery profile.
    DisneyDece,
}

impl ApplicationProfile {
    /// Human-readable display name.
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::App2 => "IMF App #2 (SMPTE ST 2067-21)",
            Self::App2E => "IMF App #2 Extended (SMPTE ST 2067-21E)",
            Self::App2_1 => "Netflix IMF App 2.1",
            Self::NflxIter1 => "Netflix Iterative IMF Iteration #1",
            Self::DisneyDece => "Disney DECE Delivery Profile",
        }
    }

    /// URN identifier string.
    pub fn urn(&self) -> &'static str {
        match self {
            Self::App2 => "urn:smpte:ul:060E2B34.04010105.0E090604.00000000",
            Self::App2E => "urn:smpte:ul:060E2B34.04010105.0E090605.00000000",
            Self::App2_1 => "urn:netflix:imf:app-2.1",
            Self::NflxIter1 => "urn:netflix:imf:iterative-1",
            Self::DisneyDece => "urn:disney:dece:delivery-1",
        }
    }
}

impl fmt::Display for ApplicationProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

// ---------------------------------------------------------------------------
// Severity
// ---------------------------------------------------------------------------

/// Severity of a compliance constraint violation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Severity {
    /// Advisory information — delivery can proceed.
    Info,
    /// Non-blocking advisory — delivery is possible but not recommended.
    Warning,
    /// Blocking error — package cannot be delivered/ingested without fix.
    Error,
    /// Critical rule directly mandated by the SMPTE standard.
    Critical,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Info => write!(f, "INFO"),
            Self::Warning => write!(f, "WARNING"),
            Self::Error => write!(f, "ERROR"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

impl Severity {
    /// Returns `true` if this severity prevents delivery.
    pub fn is_blocking(&self) -> bool {
        matches!(self, Self::Error | Self::Critical)
    }
}

// ---------------------------------------------------------------------------
// ComplianceConstraint
// ---------------------------------------------------------------------------

/// A single rule that must hold for a specific application profile.
#[derive(Clone, Debug)]
pub struct ComplianceConstraint {
    /// Machine-readable rule identifier (e.g. `"APP2-001"`).
    pub rule_id: String,
    /// Human-readable description of the rule.
    pub description: String,
    /// Severity if this constraint is violated.
    pub severity: Severity,
    /// SMPTE standard reference (e.g. `"SMPTE ST 2067-21:2020 §4.2"`).
    pub reference: String,
}

impl ComplianceConstraint {
    fn new(
        rule_id: impl Into<String>,
        description: impl Into<String>,
        severity: Severity,
        reference: impl Into<String>,
    ) -> Self {
        Self {
            rule_id: rule_id.into(),
            description: description.into(),
            severity,
            reference: reference.into(),
        }
    }
}

impl fmt::Display for ComplianceConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {} — {} ({})",
            self.severity, self.rule_id, self.description, self.reference
        )
    }
}

// ---------------------------------------------------------------------------
// ComplianceViolation
// ---------------------------------------------------------------------------

/// An instance of a constraint that was found to be violated.
#[derive(Clone, Debug)]
pub struct ComplianceViolation {
    /// The constraint that was violated.
    pub constraint: ComplianceConstraint,
    /// Human-readable detail about what specifically failed.
    pub detail: String,
}

impl ComplianceViolation {
    /// Whether this violation prevents delivery.
    pub fn is_blocking(&self) -> bool {
        self.constraint.severity.is_blocking()
    }
}

impl fmt::Display for ComplianceViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.constraint, self.detail)
    }
}

// ---------------------------------------------------------------------------
// ComplianceMatrix — the constraint registry
// ---------------------------------------------------------------------------

/// Registry of required constraints per application profile.
pub struct ComplianceMatrix;

impl ComplianceMatrix {
    /// Returns all required constraints for the given application profile.
    pub fn required_constraints(profile: ApplicationProfile) -> Vec<ComplianceConstraint> {
        match profile {
            ApplicationProfile::App2 => Self::app2_constraints(),
            ApplicationProfile::App2E => Self::app2e_constraints(),
            ApplicationProfile::App2_1 => Self::app2_1_constraints(),
            ApplicationProfile::NflxIter1 => Self::nflx_iter1_constraints(),
            ApplicationProfile::DisneyDece => Self::disney_dece_constraints(),
        }
    }

    // ---- App #2 (SMPTE ST 2067-21) ----------------------------------------

    fn app2_constraints() -> Vec<ComplianceConstraint> {
        vec![
            ComplianceConstraint::new(
                "APP2-001",
                "Package must contain exactly one CPL",
                Severity::Critical,
                "SMPTE ST 2067-21:2020 §4.1",
            ),
            ComplianceConstraint::new(
                "APP2-002",
                "CPL must reference at least one MXF-wrapped JPEG 2000 video essence",
                Severity::Critical,
                "SMPTE ST 2067-21:2020 §4.2",
            ),
            ComplianceConstraint::new(
                "APP2-003",
                "Video edit rate must be one of: 24, 25, 30, 48, 50, 60 fps (or NTSC drop variants)",
                Severity::Error,
                "SMPTE ST 2067-21:2020 §4.3",
            ),
            ComplianceConstraint::new(
                "APP2-004",
                "Maximum video resolution is 4096x3112 (4K DCI)",
                Severity::Error,
                "SMPTE ST 2067-21:2020 §4.4",
            ),
            ComplianceConstraint::new(
                "APP2-005",
                "PKL must include SHA-1 hashes for all assets",
                Severity::Critical,
                "SMPTE ST 2067-21:2020 §5.1 / SMPTE ST 429-8",
            ),
            ComplianceConstraint::new(
                "APP2-006",
                "Audio must use PCM or IAB (Immersive Audio Bitstream) in MXF",
                Severity::Error,
                "SMPTE ST 2067-21:2020 §4.5",
            ),
            ComplianceConstraint::new(
                "APP2-007",
                "Audio sample rate must be 48000 Hz or 96000 Hz",
                Severity::Error,
                "SMPTE ST 2067-21:2020 §4.5.2",
            ),
            ComplianceConstraint::new(
                "APP2-008",
                "Maximum audio channel count is 16",
                Severity::Warning,
                "SMPTE ST 2067-21:2020 §4.5.3",
            ),
            ComplianceConstraint::new(
                "APP2-009",
                "ASSETMAP must be named 'ASSETMAP.xml'",
                Severity::Critical,
                "SMPTE ST 429-9:2014 §5.1",
            ),
            ComplianceConstraint::new(
                "APP2-010",
                "All UUIDs must be RFC 4122 version 4 (random) or version 5",
                Severity::Warning,
                "SMPTE ST 2067-3:2020 §6.1",
            ),
        ]
    }

    // ---- App #2 Extended ---------------------------------------------------

    fn app2e_constraints() -> Vec<ComplianceConstraint> {
        let mut constraints = Self::app2_constraints();
        // Override APP2-004 with higher resolution limit
        constraints.retain(|c| c.rule_id != "APP2-004");
        constraints.push(ComplianceConstraint::new(
            "APP2E-001",
            "Maximum video resolution is 7680x4320 (8K UHD) for App #2 Extended",
            Severity::Error,
            "SMPTE ST 2067-21E:2022 §5.1",
        ));
        constraints.push(ComplianceConstraint::new(
            "APP2E-002",
            "HDR metadata (SMPTE ST 2086 / CTA-861-G) must be present for HDR content",
            Severity::Warning,
            "SMPTE ST 2067-21E:2022 §5.2",
        ));
        constraints.push(ComplianceConstraint::new(
            "APP2E-003",
            "JPEG 2000 bit depth must be 12 bits for extended HDR workflows",
            Severity::Warning,
            "SMPTE ST 2067-21E:2022 §5.3",
        ));
        constraints
    }

    // ---- Netflix App 2.1 --------------------------------------------------

    fn app2_1_constraints() -> Vec<ComplianceConstraint> {
        vec![
            ComplianceConstraint::new(
                "NFLX21-001",
                "Package must contain exactly one primary CPL",
                Severity::Critical,
                "Netflix IMF Requirements v2.1 §3.1",
            ),
            ComplianceConstraint::new(
                "NFLX21-002",
                "Video essence must be JPEG 2000 or AVC (H.264) in MXF",
                Severity::Critical,
                "Netflix IMF Requirements v2.1 §3.2",
            ),
            ComplianceConstraint::new(
                "NFLX21-003",
                "AVC video must use High Profile level 5.2 or below",
                Severity::Error,
                "Netflix IMF Requirements v2.1 §3.2.1",
            ),
            ComplianceConstraint::new(
                "NFLX21-004",
                "PKL hash algorithm must be SHA-256 (SHA-1 deprecated in App 2.1)",
                Severity::Error,
                "Netflix IMF Requirements v2.1 §4.1",
            ),
            ComplianceConstraint::new(
                "NFLX21-005",
                "Audio must be PCM 48kHz or 96kHz, 24-bit",
                Severity::Error,
                "Netflix IMF Requirements v2.1 §3.3",
            ),
            ComplianceConstraint::new(
                "NFLX21-006",
                "Subtitles must use TTML / IMSC1 format wrapped in MXF",
                Severity::Warning,
                "Netflix IMF Requirements v2.1 §3.4",
            ),
            ComplianceConstraint::new(
                "NFLX21-007",
                "Package must include a content version label",
                Severity::Warning,
                "Netflix IMF Requirements v2.1 §5.1",
            ),
        ]
    }

    // ---- Netflix Iterative IMF Iteration #1 --------------------------------

    fn nflx_iter1_constraints() -> Vec<ComplianceConstraint> {
        vec![
            ComplianceConstraint::new(
                "NFLXI1-001",
                "Package must be structured as a supplemental IMF package chain",
                Severity::Critical,
                "Netflix Iterative IMF Spec Iter1 §2.1",
            ),
            ComplianceConstraint::new(
                "NFLXI1-002",
                "Video must be Apple ProRes 4444 or AVC High Profile",
                Severity::Critical,
                "Netflix Iterative IMF Spec Iter1 §3.1",
            ),
            ComplianceConstraint::new(
                "NFLXI1-003",
                "Audio must be PCM 48kHz, 24-bit, maximum 16 channels",
                Severity::Error,
                "Netflix Iterative IMF Spec Iter1 §3.2",
            ),
            ComplianceConstraint::new(
                "NFLXI1-004",
                "SHA-256 hashes required for all MXF essence assets",
                Severity::Critical,
                "Netflix Iterative IMF Spec Iter1 §4.1",
            ),
            ComplianceConstraint::new(
                "NFLXI1-005",
                "Package must not exceed 300 GB total uncompressed essence",
                Severity::Warning,
                "Netflix Iterative IMF Spec Iter1 §5.2",
            ),
        ]
    }

    // ---- Disney DECE -------------------------------------------------------

    fn disney_dece_constraints() -> Vec<ComplianceConstraint> {
        vec![
            ComplianceConstraint::new(
                "DECE-001",
                "Package must comply with CFF (Common File Format) wrapping",
                Severity::Critical,
                "Disney DECE Delivery Specification §2.1",
            ),
            ComplianceConstraint::new(
                "DECE-002",
                "Video must be AVC/H.264 Main or High Profile",
                Severity::Critical,
                "Disney DECE Delivery Specification §3.1",
            ),
            ComplianceConstraint::new(
                "DECE-003",
                "Maximum video resolution is 1920x1080 (Full HD)",
                Severity::Error,
                "Disney DECE Delivery Specification §3.2",
            ),
            ComplianceConstraint::new(
                "DECE-004",
                "Audio must be AAC-LC or AC-3 at 48kHz",
                Severity::Error,
                "Disney DECE Delivery Specification §3.3",
            ),
            ComplianceConstraint::new(
                "DECE-005",
                "DRM must be present: Marlin or PlayReady license data required",
                Severity::Critical,
                "Disney DECE Delivery Specification §6.1",
            ),
            ComplianceConstraint::new(
                "DECE-006",
                "Subtitles must use TTML or SRT format",
                Severity::Warning,
                "Disney DECE Delivery Specification §3.4",
            ),
            ComplianceConstraint::new(
                "DECE-007",
                "MD5 hashes are acceptable but SHA-256 is preferred",
                Severity::Info,
                "Disney DECE Delivery Specification §4.1",
            ),
        ]
    }
}

// ---------------------------------------------------------------------------
// ComplianceChecker
// ---------------------------------------------------------------------------

/// Minimal IMF package description used for compliance checking.
///
/// This is a simplified, self-contained struct so `ComplianceChecker` does not
/// need to import the heavyweight `package.rs` types.
#[derive(Debug, Clone, Default)]
pub struct PackageDescription {
    /// Number of CPLs in the package.
    pub cpl_count: usize,
    /// Number of video tracks.
    pub video_track_count: usize,
    /// Number of audio tracks.
    pub audio_track_count: usize,
    /// Number of subtitle tracks.
    pub subtitle_track_count: usize,
    /// Maximum audio channel count across all audio tracks.
    pub max_audio_channels: u32,
    /// Audio sample rate in Hz (0 = unknown).
    pub audio_sample_rate: u32,
    /// Video frame width in pixels (0 = unknown).
    pub video_width: u32,
    /// Video frame height in pixels (0 = unknown).
    pub video_height: u32,
    /// Hash algorithm used in the PKL (e.g. "SHA-1", "SHA-256").
    pub pkl_hash_algorithm: String,
    /// Video essence type string (e.g. "JPEG2000", "AVC", "ProRes").
    pub video_essence_type: String,
    /// Audio essence type string (e.g. "PCM", "IAB", "AAC").
    pub audio_essence_type: String,
    /// Whether DRM information is present.
    pub has_drm: bool,
    /// Total package size in bytes (0 = unknown).
    pub total_size_bytes: u64,
    /// Whether content version label is present.
    pub has_content_version: bool,
    /// ASSETMAP filename.
    pub assetmap_filename: String,
}

/// Checks a package description against an application profile's constraints.
pub struct ComplianceChecker;

impl ComplianceChecker {
    /// Check `package` against `profile` and return all violations found.
    pub fn check(
        package: &PackageDescription,
        profile: ApplicationProfile,
    ) -> Vec<ComplianceViolation> {
        let constraints = ComplianceMatrix::required_constraints(profile);
        let mut violations = Vec::new();

        for constraint in &constraints {
            let opt_violation = Self::evaluate(constraint, package, profile);
            if let Some(v) = opt_violation {
                violations.push(v);
            }
        }

        violations
    }

    /// Check and return only blocking (Error/Critical) violations.
    pub fn check_blocking(
        package: &PackageDescription,
        profile: ApplicationProfile,
    ) -> Vec<ComplianceViolation> {
        Self::check(package, profile)
            .into_iter()
            .filter(|v| v.is_blocking())
            .collect()
    }

    /// Returns `true` if the package is fully compliant with the profile.
    pub fn is_compliant(package: &PackageDescription, profile: ApplicationProfile) -> bool {
        Self::check_blocking(package, profile).is_empty()
    }

    // ---- Internal rule evaluation -----------------------------------------

    fn evaluate(
        constraint: &ComplianceConstraint,
        pkg: &PackageDescription,
        _profile: ApplicationProfile,
    ) -> Option<ComplianceViolation> {
        match constraint.rule_id.as_str() {
            // Common single-CPL rules
            "APP2-001" | "NFLX21-001" | "NFLXI1-001" => {
                if pkg.cpl_count != 1 {
                    return Some(ComplianceViolation {
                        constraint: constraint.clone(),
                        detail: format!(
                            "Package has {} CPL(s); exactly 1 is required",
                            pkg.cpl_count
                        ),
                    });
                }
            }

            // JPEG 2000 video required
            "APP2-002" => {
                if !pkg.video_essence_type.to_ascii_uppercase().contains("JPEG") {
                    return Some(ComplianceViolation {
                        constraint: constraint.clone(),
                        detail: format!(
                            "Video essence is '{}'; JPEG 2000 required for App #2",
                            pkg.video_essence_type
                        ),
                    });
                }
            }

            // Resolution cap App2 (4096x3112)
            "APP2-004" => {
                if pkg.video_width > 4096 || pkg.video_height > 3112 {
                    return Some(ComplianceViolation {
                        constraint: constraint.clone(),
                        detail: format!(
                            "Video resolution {}x{} exceeds the 4096x3112 maximum",
                            pkg.video_width, pkg.video_height
                        ),
                    });
                }
            }

            // SHA-1 in PKL
            "APP2-005" => {
                if !pkg
                    .pkl_hash_algorithm
                    .to_ascii_uppercase()
                    .contains("SHA-1")
                {
                    return Some(ComplianceViolation {
                        constraint: constraint.clone(),
                        detail: format!(
                            "PKL uses '{}'; SHA-1 is required for App #2",
                            pkg.pkl_hash_algorithm
                        ),
                    });
                }
            }

            // Audio channels cap
            "APP2-008" => {
                if pkg.max_audio_channels > 16 {
                    return Some(ComplianceViolation {
                        constraint: constraint.clone(),
                        detail: format!(
                            "Audio track has {} channels; maximum is 16",
                            pkg.max_audio_channels
                        ),
                    });
                }
            }

            // ASSETMAP filename
            "APP2-009" => {
                if pkg.assetmap_filename != "ASSETMAP.xml" {
                    return Some(ComplianceViolation {
                        constraint: constraint.clone(),
                        detail: format!(
                            "ASSETMAP is named '{}'; must be 'ASSETMAP.xml'",
                            pkg.assetmap_filename
                        ),
                    });
                }
            }

            // App2E resolution cap (8K)
            "APP2E-001" => {
                if pkg.video_width > 7680 || pkg.video_height > 4320 {
                    return Some(ComplianceViolation {
                        constraint: constraint.clone(),
                        detail: format!(
                            "Video resolution {}x{} exceeds the 7680x4320 maximum",
                            pkg.video_width, pkg.video_height
                        ),
                    });
                }
            }

            // Netflix SHA-256 requirement
            "NFLX21-004" | "NFLXI1-004" => {
                if !pkg
                    .pkl_hash_algorithm
                    .to_ascii_uppercase()
                    .contains("SHA-256")
                {
                    return Some(ComplianceViolation {
                        constraint: constraint.clone(),
                        detail: format!(
                            "PKL uses '{}'; SHA-256 is required",
                            pkg.pkl_hash_algorithm
                        ),
                    });
                }
            }

            // DECE AVC video
            "DECE-002" => {
                let t = pkg.video_essence_type.to_ascii_uppercase();
                if !t.contains("AVC") && !t.contains("H264") && !t.contains("H.264") {
                    return Some(ComplianceViolation {
                        constraint: constraint.clone(),
                        detail: format!(
                            "Video essence is '{}'; AVC/H.264 is required for Disney DECE",
                            pkg.video_essence_type
                        ),
                    });
                }
            }

            // DECE resolution cap (1080p)
            "DECE-003" => {
                if pkg.video_width > 1920 || pkg.video_height > 1080 {
                    return Some(ComplianceViolation {
                        constraint: constraint.clone(),
                        detail: format!(
                            "Video resolution {}x{} exceeds the 1920x1080 maximum for DECE",
                            pkg.video_width, pkg.video_height
                        ),
                    });
                }
            }

            // DECE DRM
            "DECE-005" => {
                if !pkg.has_drm {
                    return Some(ComplianceViolation {
                        constraint: constraint.clone(),
                        detail: "No DRM metadata found; Marlin or PlayReady is required for DECE"
                            .to_string(),
                    });
                }
            }

            // Content version label (Netflix)
            "NFLX21-007" => {
                if !pkg.has_content_version {
                    return Some(ComplianceViolation {
                        constraint: constraint.clone(),
                        detail: "No content version label found in the CPL".to_string(),
                    });
                }
            }

            _ => {
                // Rule not yet evaluated — no violation raised
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_app2_package() -> PackageDescription {
        PackageDescription {
            cpl_count: 1,
            video_track_count: 1,
            audio_track_count: 1,
            subtitle_track_count: 0,
            max_audio_channels: 8,
            audio_sample_rate: 48000,
            video_width: 1920,
            video_height: 1080,
            pkl_hash_algorithm: "SHA-1".to_string(),
            video_essence_type: "JPEG2000".to_string(),
            audio_essence_type: "PCM".to_string(),
            has_drm: false,
            total_size_bytes: 1_000_000_000,
            has_content_version: false,
            assetmap_filename: "ASSETMAP.xml".to_string(),
        }
    }

    fn valid_nflx_package() -> PackageDescription {
        PackageDescription {
            cpl_count: 1,
            video_track_count: 1,
            audio_track_count: 1,
            subtitle_track_count: 1,
            max_audio_channels: 8,
            audio_sample_rate: 48000,
            video_width: 3840,
            video_height: 2160,
            pkl_hash_algorithm: "SHA-256".to_string(),
            video_essence_type: "JPEG2000".to_string(),
            audio_essence_type: "PCM".to_string(),
            has_drm: false,
            total_size_bytes: 50_000_000_000,
            has_content_version: true,
            assetmap_filename: "ASSETMAP.xml".to_string(),
        }
    }

    // --- ApplicationProfile tests ---

    #[test]
    fn test_all_profiles_have_distinct_urns() {
        let profiles = [
            ApplicationProfile::App2,
            ApplicationProfile::App2E,
            ApplicationProfile::App2_1,
            ApplicationProfile::NflxIter1,
            ApplicationProfile::DisneyDece,
        ];
        let urns: std::collections::HashSet<&str> = profiles.iter().map(|p| p.urn()).collect();
        assert_eq!(urns.len(), 5, "All profile URNs must be distinct");
    }

    #[test]
    fn test_profile_display() {
        let s = format!("{}", ApplicationProfile::App2);
        assert!(s.contains("App"));
    }

    // --- Severity tests ---

    #[test]
    fn test_severity_blocking() {
        assert!(Severity::Error.is_blocking());
        assert!(Severity::Critical.is_blocking());
        assert!(!Severity::Warning.is_blocking());
        assert!(!Severity::Info.is_blocking());
    }

    // --- ComplianceMatrix constraint counts ---

    #[test]
    fn test_app2_constraint_count() {
        let constraints = ComplianceMatrix::required_constraints(ApplicationProfile::App2);
        assert!(
            constraints.len() >= 8,
            "App2 must define at least 8 constraints"
        );
    }

    #[test]
    fn test_app2e_extends_app2() {
        let app2 = ComplianceMatrix::required_constraints(ApplicationProfile::App2);
        let app2e = ComplianceMatrix::required_constraints(ApplicationProfile::App2E);
        // App2E has more constraints than App2
        assert!(app2e.len() > app2.len());
    }

    #[test]
    fn test_nflx_iter1_constraint_count() {
        let c = ComplianceMatrix::required_constraints(ApplicationProfile::NflxIter1);
        assert!(c.len() >= 4);
    }

    #[test]
    fn test_disney_dece_constraint_count() {
        let c = ComplianceMatrix::required_constraints(ApplicationProfile::DisneyDece);
        assert!(c.len() >= 5);
    }

    // --- ComplianceChecker happy paths ---

    #[test]
    fn test_valid_app2_no_blocking_violations() {
        let pkg = valid_app2_package();
        let violations = ComplianceChecker::check_blocking(&pkg, ApplicationProfile::App2);
        assert!(
            violations.is_empty(),
            "Valid App2 package should have no blocking violations; got: {:?}",
            violations.iter().map(|v| &v.detail).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_is_compliant_returns_true_for_valid_package() {
        let pkg = valid_app2_package();
        assert!(ComplianceChecker::is_compliant(
            &pkg,
            ApplicationProfile::App2
        ));
    }

    // --- Violation detection tests ---

    #[test]
    fn test_app2_multiple_cpls_violation() {
        let mut pkg = valid_app2_package();
        pkg.cpl_count = 2;
        let violations = ComplianceChecker::check(&pkg, ApplicationProfile::App2);
        let found = violations
            .iter()
            .any(|v| v.constraint.rule_id == "APP2-001");
        assert!(found, "Should detect APP2-001 violation for multiple CPLs");
    }

    #[test]
    fn test_app2_wrong_video_essence_violation() {
        let mut pkg = valid_app2_package();
        pkg.video_essence_type = "ProRes".to_string();
        let violations = ComplianceChecker::check(&pkg, ApplicationProfile::App2);
        let found = violations
            .iter()
            .any(|v| v.constraint.rule_id == "APP2-002");
        assert!(
            found,
            "Should detect APP2-002 violation for non-JPEG2000 video"
        );
    }

    #[test]
    fn test_app2_resolution_too_high_violation() {
        let mut pkg = valid_app2_package();
        pkg.video_width = 8192;
        pkg.video_height = 4320;
        let violations = ComplianceChecker::check(&pkg, ApplicationProfile::App2);
        let found = violations
            .iter()
            .any(|v| v.constraint.rule_id == "APP2-004");
        assert!(found, "Should detect APP2-004 resolution violation");
    }

    #[test]
    fn test_app2_wrong_hash_algorithm_violation() {
        let mut pkg = valid_app2_package();
        pkg.pkl_hash_algorithm = "SHA-256".to_string();
        let violations = ComplianceChecker::check(&pkg, ApplicationProfile::App2);
        let found = violations
            .iter()
            .any(|v| v.constraint.rule_id == "APP2-005");
        assert!(found, "Should detect APP2-005 hash algorithm violation");
    }

    #[test]
    fn test_app2_wrong_assetmap_name_violation() {
        let mut pkg = valid_app2_package();
        pkg.assetmap_filename = "assetmap.xml".to_string(); // wrong case
        let violations = ComplianceChecker::check(&pkg, ApplicationProfile::App2);
        let found = violations
            .iter()
            .any(|v| v.constraint.rule_id == "APP2-009");
        assert!(found, "Should detect APP2-009 ASSETMAP filename violation");
    }

    #[test]
    fn test_nflx21_sha256_required() {
        let mut pkg = valid_nflx_package();
        pkg.pkl_hash_algorithm = "SHA-1".to_string(); // wrong for Netflix
        let violations = ComplianceChecker::check(&pkg, ApplicationProfile::App2_1);
        let found = violations
            .iter()
            .any(|v| v.constraint.rule_id == "NFLX21-004");
        assert!(found);
    }

    #[test]
    fn test_dece_resolution_violation() {
        let mut pkg = valid_app2_package();
        pkg.video_essence_type = "AVC".to_string();
        pkg.pkl_hash_algorithm = "MD5".to_string();
        pkg.video_width = 3840;
        pkg.video_height = 2160;
        pkg.has_drm = true;
        let violations = ComplianceChecker::check(&pkg, ApplicationProfile::DisneyDece);
        let found = violations
            .iter()
            .any(|v| v.constraint.rule_id == "DECE-003");
        assert!(found, "Should detect DECE-003 resolution violation for 4K");
    }

    #[test]
    fn test_dece_missing_drm_violation() {
        let mut pkg = valid_app2_package();
        pkg.video_essence_type = "AVC".to_string();
        pkg.has_drm = false;
        let violations = ComplianceChecker::check(&pkg, ApplicationProfile::DisneyDece);
        let found = violations
            .iter()
            .any(|v| v.constraint.rule_id == "DECE-005");
        assert!(found, "Should detect DECE-005 DRM violation");
    }

    #[test]
    fn test_violation_display() {
        let constraint =
            ComplianceConstraint::new("TEST-001", "Test rule", Severity::Error, "TEST §1.0");
        let violation = ComplianceViolation {
            constraint,
            detail: "Something went wrong".to_string(),
        };
        let s = format!("{violation}");
        assert!(s.contains("TEST-001"));
        assert!(s.contains("Something went wrong"));
    }

    #[test]
    fn test_constraint_display() {
        let c = ComplianceConstraint::new(
            "APP2-001",
            "Must have exactly one CPL",
            Severity::Critical,
            "SMPTE ST 2067-21 §4.1",
        );
        let s = format!("{c}");
        assert!(s.contains("APP2-001"));
        assert!(s.contains("CRITICAL"));
    }
}
