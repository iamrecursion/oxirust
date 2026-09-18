#![allow(dead_code)]
//! HDR quality control checks for media files.
//!
//! Validates High Dynamic Range metadata, peak brightness levels, color gamut
//! coverage, and transfer function conformance for PQ (ST.2084), HLG (BT.2100),
//! and Dolby Vision content.

#[allow(clippy::cast_precision_loss)]
/// HDR transfer function standard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HdrTransferFunction {
    /// Perceptual Quantizer (SMPTE ST 2084).
    Pq,
    /// Hybrid Log-Gamma (BT.2100).
    Hlg,
    /// Scene-referred linear light.
    Linear,
    /// SDR gamma 2.4.
    Sdr,
}

impl std::fmt::Display for HdrTransferFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pq => write!(f, "PQ (ST.2084)"),
            Self::Hlg => write!(f, "HLG (BT.2100)"),
            Self::Linear => write!(f, "Linear"),
            Self::Sdr => write!(f, "SDR"),
        }
    }
}

/// Wide color gamut standard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorGamut {
    /// ITU-R BT.709 (standard definition / HD).
    Bt709,
    /// ITU-R BT.2020 / BT.2100 (UHD wide gamut).
    Bt2020,
    /// DCI-P3 (cinema / Apple displays).
    DciP3,
    /// Display P3 (consumer devices).
    DisplayP3,
}

impl std::fmt::Display for ColorGamut {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bt709 => write!(f, "BT.709"),
            Self::Bt2020 => write!(f, "BT.2020"),
            Self::DciP3 => write!(f, "DCI-P3"),
            Self::DisplayP3 => write!(f, "Display P3"),
        }
    }
}

/// Static HDR metadata (SMPTE ST 2086).
#[derive(Debug, Clone)]
pub struct StaticHdrMetadata {
    /// Maximum content light level in cd/m^2.
    pub max_cll: f64,
    /// Maximum frame-average light level in cd/m^2.
    pub max_fall: f64,
    /// Mastering display minimum luminance in cd/m^2.
    pub min_luminance: f64,
    /// Mastering display maximum luminance in cd/m^2.
    pub max_luminance: f64,
    /// Color gamut primaries.
    pub gamut: ColorGamut,
    /// Transfer function.
    pub transfer: HdrTransferFunction,
}

impl StaticHdrMetadata {
    /// Creates new static HDR metadata.
    #[must_use]
    pub fn new(
        max_cll: f64,
        max_fall: f64,
        min_luminance: f64,
        max_luminance: f64,
        gamut: ColorGamut,
        transfer: HdrTransferFunction,
    ) -> Self {
        Self {
            max_cll,
            max_fall,
            min_luminance,
            max_luminance,
            gamut,
            transfer,
        }
    }
}

/// Dynamic HDR metadata entry for a single frame or scene.
#[derive(Debug, Clone)]
pub struct DynamicHdrEntry {
    /// Frame index this metadata applies to.
    pub frame_index: u64,
    /// Scene maximum content light level in cd/m^2.
    pub scene_max_cll: f64,
    /// Scene average light level in cd/m^2.
    pub scene_avg_ll: f64,
    /// Tone mapping target peak brightness in cd/m^2.
    pub target_peak_nits: f64,
}

impl DynamicHdrEntry {
    /// Creates a new dynamic HDR metadata entry.
    #[must_use]
    pub fn new(
        frame_index: u64,
        scene_max_cll: f64,
        scene_avg_ll: f64,
        target_peak_nits: f64,
    ) -> Self {
        Self {
            frame_index,
            scene_max_cll,
            scene_avg_ll,
            target_peak_nits,
        }
    }
}

/// Severity level for HDR QC findings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum HdrSeverity {
    /// Informational finding.
    Info,
    /// Warning — content may not display optimally.
    Warning,
    /// Error — content violates spec requirements.
    Error,
    /// Critical — content may fail playback or certification.
    Critical,
}

impl std::fmt::Display for HdrSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Info => write!(f, "INFO"),
            Self::Warning => write!(f, "WARNING"),
            Self::Error => write!(f, "ERROR"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// A single HDR QC finding.
#[derive(Debug, Clone)]
pub struct HdrFinding {
    /// Severity of the finding.
    pub severity: HdrSeverity,
    /// Short code for the check.
    pub code: String,
    /// Human-readable description.
    pub message: String,
    /// Optional recommendation for remediation.
    pub recommendation: Option<String>,
}

impl HdrFinding {
    /// Creates a new HDR finding.
    #[must_use]
    pub fn new(severity: HdrSeverity, code: &str, message: &str) -> Self {
        Self {
            severity,
            code: code.to_string(),
            message: message.to_string(),
            recommendation: None,
        }
    }

    /// Attaches a recommendation to this finding.
    #[must_use]
    pub fn with_recommendation(mut self, rec: &str) -> Self {
        self.recommendation = Some(rec.to_string());
        self
    }

    /// Returns whether this finding indicates a failure.
    #[must_use]
    pub fn is_failure(&self) -> bool {
        matches!(self.severity, HdrSeverity::Error | HdrSeverity::Critical)
    }
}

/// Result of a full HDR QC pass.
#[derive(Debug, Clone)]
pub struct HdrQcReport {
    /// Whether the overall check passed.
    pub passed: bool,
    /// All findings from the check.
    pub findings: Vec<HdrFinding>,
    /// Number of frames analyzed.
    pub frames_analyzed: u64,
}

impl HdrQcReport {
    /// Creates a new empty report.
    #[must_use]
    pub fn new() -> Self {
        Self {
            passed: true,
            findings: Vec::new(),
            frames_analyzed: 0,
        }
    }

    /// Adds a finding and updates the pass/fail status.
    pub fn add_finding(&mut self, finding: HdrFinding) {
        if finding.is_failure() {
            self.passed = false;
        }
        self.findings.push(finding);
    }

    /// Returns only the error and critical findings.
    #[must_use]
    pub fn errors(&self) -> Vec<&HdrFinding> {
        self.findings.iter().filter(|f| f.is_failure()).collect()
    }

    /// Returns the total number of findings.
    #[must_use]
    pub fn finding_count(&self) -> usize {
        self.findings.len()
    }

    /// Returns the count of findings by severity.
    #[must_use]
    pub fn count_by_severity(&self, severity: HdrSeverity) -> usize {
        self.findings
            .iter()
            .filter(|f| f.severity == severity)
            .count()
    }
}

impl Default for HdrQcReport {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for HDR QC checks.
#[derive(Debug, Clone)]
pub struct HdrQcConfig {
    /// Maximum allowed MaxCLL in nits (default 10000).
    pub max_cll_limit: f64,
    /// Maximum allowed MaxFALL in nits (default 4000).
    pub max_fall_limit: f64,
    /// Minimum mastering display luminance in nits (default 0.0001).
    pub min_luminance_floor: f64,
    /// Maximum mastering display luminance in nits (default 10000).
    pub max_luminance_ceiling: f64,
    /// Whether to require BT.2020 gamut for HDR content.
    pub require_bt2020: bool,
    /// Whether to validate dynamic metadata continuity.
    pub check_dynamic_continuity: bool,
    /// Maximum allowed jump in scene MaxCLL between consecutive entries (nits).
    pub dynamic_max_jump: f64,
}

impl HdrQcConfig {
    /// Creates a new HDR QC configuration with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_cll_limit: 10_000.0,
            max_fall_limit: 4_000.0,
            min_luminance_floor: 0.0001,
            max_luminance_ceiling: 10_000.0,
            require_bt2020: true,
            check_dynamic_continuity: true,
            dynamic_max_jump: 5_000.0,
        }
    }

    /// Sets the MaxCLL limit.
    #[must_use]
    pub fn with_max_cll_limit(mut self, limit: f64) -> Self {
        self.max_cll_limit = limit;
        self
    }

    /// Sets the MaxFALL limit.
    #[must_use]
    pub fn with_max_fall_limit(mut self, limit: f64) -> Self {
        self.max_fall_limit = limit;
        self
    }

    /// Enables or disables BT.2020 gamut requirement.
    #[must_use]
    pub fn with_require_bt2020(mut self, require: bool) -> Self {
        self.require_bt2020 = require;
        self
    }
}

impl Default for HdrQcConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// HDR quality control checker.
///
/// Validates static and dynamic HDR metadata against configurable limits,
/// checks transfer function / gamut consistency, and verifies dynamic
/// metadata continuity.
#[derive(Debug, Clone)]
pub struct HdrQcChecker {
    /// Configuration for the checker.
    config: HdrQcConfig,
}

impl HdrQcChecker {
    /// Creates a new HDR QC checker with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: HdrQcConfig::new(),
        }
    }

    /// Creates a new HDR QC checker with the given configuration.
    #[must_use]
    pub fn with_config(config: HdrQcConfig) -> Self {
        Self { config }
    }

    /// Validates static HDR metadata.
    #[must_use]
    pub fn check_static_metadata(&self, meta: &StaticHdrMetadata) -> HdrQcReport {
        let mut report = HdrQcReport::new();

        // Check MaxCLL bounds
        if meta.max_cll > self.config.max_cll_limit {
            report.add_finding(HdrFinding::new(
                HdrSeverity::Error,
                "HDR-001",
                &format!(
                    "MaxCLL ({:.1} nits) exceeds limit ({:.1} nits)",
                    meta.max_cll, self.config.max_cll_limit
                ),
            ));
        }

        if meta.max_cll <= 0.0 {
            report.add_finding(HdrFinding::new(
                HdrSeverity::Warning,
                "HDR-002",
                "MaxCLL is zero or negative — metadata may be missing",
            ));
        }

        // Check MaxFALL bounds
        if meta.max_fall > self.config.max_fall_limit {
            report.add_finding(HdrFinding::new(
                HdrSeverity::Error,
                "HDR-003",
                &format!(
                    "MaxFALL ({:.1} nits) exceeds limit ({:.1} nits)",
                    meta.max_fall, self.config.max_fall_limit
                ),
            ));
        }

        // MaxFALL should not exceed MaxCLL
        if meta.max_fall > meta.max_cll && meta.max_cll > 0.0 {
            report.add_finding(HdrFinding::new(
                HdrSeverity::Error,
                "HDR-004",
                &format!(
                    "MaxFALL ({:.1}) exceeds MaxCLL ({:.1}) — this is invalid",
                    meta.max_fall, meta.max_cll
                ),
            ));
        }

        // Check mastering display luminance range
        if meta.min_luminance < self.config.min_luminance_floor {
            report.add_finding(HdrFinding::new(
                HdrSeverity::Warning,
                "HDR-005",
                &format!(
                    "Mastering min luminance ({:.6}) is below floor ({:.6})",
                    meta.min_luminance, self.config.min_luminance_floor
                ),
            ));
        }

        if meta.max_luminance > self.config.max_luminance_ceiling {
            report.add_finding(HdrFinding::new(
                HdrSeverity::Error,
                "HDR-006",
                &format!(
                    "Mastering max luminance ({:.1}) exceeds ceiling ({:.1})",
                    meta.max_luminance, self.config.max_luminance_ceiling
                ),
            ));
        }

        if meta.min_luminance >= meta.max_luminance {
            report.add_finding(HdrFinding::new(
                HdrSeverity::Critical,
                "HDR-007",
                "Mastering min luminance >= max luminance — inverted range",
            ));
        }

        // Gamut check
        if self.config.require_bt2020 && meta.gamut != ColorGamut::Bt2020 {
            report.add_finding(
                HdrFinding::new(
                    HdrSeverity::Warning,
                    "HDR-008",
                    &format!(
                        "Color gamut is {} but BT.2020 is required for HDR",
                        meta.gamut
                    ),
                )
                .with_recommendation("Re-master with BT.2020 primaries"),
            );
        }

        // Transfer function check for HDR
        if meta.transfer == HdrTransferFunction::Sdr {
            report.add_finding(HdrFinding::new(
                HdrSeverity::Error,
                "HDR-009",
                "SDR transfer function on content with HDR metadata",
            ));
        }

        report
    }

    /// Validates dynamic HDR metadata for continuity and bounds.
    #[must_use]
    pub fn check_dynamic_metadata(&self, entries: &[DynamicHdrEntry]) -> HdrQcReport {
        let mut report = HdrQcReport::new();

        if entries.is_empty() {
            report.add_finding(HdrFinding::new(
                HdrSeverity::Info,
                "HDR-D01",
                "No dynamic HDR metadata present",
            ));
            return report;
        }

        #[allow(clippy::cast_precision_loss)]
        {
            report.frames_analyzed = entries.len() as u64;
        }

        for (i, entry) in entries.iter().enumerate() {
            // MaxCLL per-scene check
            if entry.scene_max_cll > self.config.max_cll_limit {
                report.add_finding(HdrFinding::new(
                    HdrSeverity::Error,
                    "HDR-D02",
                    &format!(
                        "Dynamic entry {} (frame {}): scene MaxCLL {:.1} exceeds limit {:.1}",
                        i, entry.frame_index, entry.scene_max_cll, self.config.max_cll_limit
                    ),
                ));
            }

            // Average should not exceed max
            if entry.scene_avg_ll > entry.scene_max_cll {
                report.add_finding(HdrFinding::new(
                    HdrSeverity::Warning,
                    "HDR-D03",
                    &format!(
                        "Dynamic entry {} (frame {}): avg LL ({:.1}) > max CLL ({:.1})",
                        i, entry.frame_index, entry.scene_avg_ll, entry.scene_max_cll
                    ),
                ));
            }

            // Target peak must be positive
            if entry.target_peak_nits <= 0.0 {
                report.add_finding(HdrFinding::new(
                    HdrSeverity::Error,
                    "HDR-D04",
                    &format!(
                        "Dynamic entry {} (frame {}): target peak nits is non-positive ({:.1})",
                        i, entry.frame_index, entry.target_peak_nits
                    ),
                ));
            }

            // Continuity check: large jumps between consecutive entries
            if self.config.check_dynamic_continuity && i > 0 {
                let prev = &entries[i - 1];
                let jump = (entry.scene_max_cll - prev.scene_max_cll).abs();
                if jump > self.config.dynamic_max_jump {
                    report.add_finding(HdrFinding::new(
                        HdrSeverity::Warning,
                        "HDR-D05",
                        &format!(
                            "Large MaxCLL jump ({:.1} nits) between entries {} and {} (frames {} -> {})",
                            jump, i - 1, i, prev.frame_index, entry.frame_index
                        ),
                    ));
                }
            }
        }

        report
    }

    /// Validates that transfer function and gamut are compatible.
    #[must_use]
    pub fn check_compatibility(
        &self,
        transfer: HdrTransferFunction,
        gamut: ColorGamut,
    ) -> HdrQcReport {
        let mut report = HdrQcReport::new();

        // PQ/HLG with BT.709 is unusual
        if matches!(transfer, HdrTransferFunction::Pq | HdrTransferFunction::Hlg)
            && gamut == ColorGamut::Bt709
        {
            report.add_finding(
                HdrFinding::new(
                    HdrSeverity::Warning,
                    "HDR-C01",
                    &format!(
                        "HDR transfer function ({transfer}) paired with narrow gamut ({gamut})"
                    ),
                )
                .with_recommendation("Use BT.2020 or DCI-P3 gamut with HDR transfer functions"),
            );
        }

        // SDR transfer with wide gamut is non-standard but acceptable
        if transfer == HdrTransferFunction::Sdr
            && matches!(gamut, ColorGamut::Bt2020 | ColorGamut::DciP3)
        {
            report.add_finding(HdrFinding::new(
                HdrSeverity::Info,
                "HDR-C02",
                &format!("SDR transfer function with wide gamut ({gamut}) — WCG-SDR content"),
            ));
        }

        report
    }
}

impl Default for HdrQcChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_static_meta() -> StaticHdrMetadata {
        StaticHdrMetadata::new(
            1000.0,
            400.0,
            0.001,
            1000.0,
            ColorGamut::Bt2020,
            HdrTransferFunction::Pq,
        )
    }

    #[test]
    fn test_valid_static_metadata_passes() {
        let checker = HdrQcChecker::new();
        let meta = sample_static_meta();
        let report = checker.check_static_metadata(&meta);
        assert!(report.passed);
        assert_eq!(report.errors().len(), 0);
    }

    #[test]
    fn test_max_cll_exceeds_limit() {
        let checker = HdrQcChecker::new();
        let meta = StaticHdrMetadata::new(
            15_000.0,
            400.0,
            0.001,
            1000.0,
            ColorGamut::Bt2020,
            HdrTransferFunction::Pq,
        );
        let report = checker.check_static_metadata(&meta);
        assert!(!report.passed);
        assert!(report.findings.iter().any(|f| f.code == "HDR-001"));
    }

    #[test]
    fn test_zero_max_cll_warning() {
        let checker = HdrQcChecker::new();
        let meta = StaticHdrMetadata::new(
            0.0,
            0.0,
            0.001,
            1000.0,
            ColorGamut::Bt2020,
            HdrTransferFunction::Pq,
        );
        let report = checker.check_static_metadata(&meta);
        assert!(report.findings.iter().any(|f| f.code == "HDR-002"));
    }

    #[test]
    fn test_max_fall_exceeds_max_cll() {
        let checker = HdrQcChecker::new();
        let meta = StaticHdrMetadata::new(
            500.0,
            800.0,
            0.001,
            1000.0,
            ColorGamut::Bt2020,
            HdrTransferFunction::Pq,
        );
        let report = checker.check_static_metadata(&meta);
        assert!(!report.passed);
        assert!(report.findings.iter().any(|f| f.code == "HDR-004"));
    }

    #[test]
    fn test_inverted_luminance_range() {
        let checker = HdrQcChecker::new();
        let meta = StaticHdrMetadata::new(
            500.0,
            200.0,
            1000.0,
            100.0,
            ColorGamut::Bt2020,
            HdrTransferFunction::Pq,
        );
        let report = checker.check_static_metadata(&meta);
        assert!(!report.passed);
        assert!(report.findings.iter().any(|f| f.code == "HDR-007"));
    }

    #[test]
    fn test_non_bt2020_gamut_warning() {
        let checker = HdrQcChecker::new();
        let meta = StaticHdrMetadata::new(
            500.0,
            200.0,
            0.001,
            1000.0,
            ColorGamut::DciP3,
            HdrTransferFunction::Pq,
        );
        let report = checker.check_static_metadata(&meta);
        assert!(report.findings.iter().any(|f| f.code == "HDR-008"));
    }

    #[test]
    fn test_sdr_transfer_on_hdr_content() {
        let checker = HdrQcChecker::new();
        let meta = StaticHdrMetadata::new(
            500.0,
            200.0,
            0.001,
            1000.0,
            ColorGamut::Bt2020,
            HdrTransferFunction::Sdr,
        );
        let report = checker.check_static_metadata(&meta);
        assert!(!report.passed);
        assert!(report.findings.iter().any(|f| f.code == "HDR-009"));
    }

    #[test]
    fn test_dynamic_metadata_empty() {
        let checker = HdrQcChecker::new();
        let report = checker.check_dynamic_metadata(&[]);
        assert!(report.passed);
        assert!(report.findings.iter().any(|f| f.code == "HDR-D01"));
    }

    #[test]
    fn test_dynamic_metadata_valid() {
        let checker = HdrQcChecker::new();
        let entries = vec![
            DynamicHdrEntry::new(0, 800.0, 200.0, 1000.0),
            DynamicHdrEntry::new(24, 900.0, 250.0, 1000.0),
            DynamicHdrEntry::new(48, 850.0, 220.0, 1000.0),
        ];
        let report = checker.check_dynamic_metadata(&entries);
        assert!(report.passed);
        assert_eq!(report.frames_analyzed, 3);
    }

    #[test]
    fn test_dynamic_metadata_large_jump() {
        let checker = HdrQcChecker::new();
        let entries = vec![
            DynamicHdrEntry::new(0, 100.0, 50.0, 1000.0),
            DynamicHdrEntry::new(24, 8000.0, 4000.0, 1000.0),
        ];
        let report = checker.check_dynamic_metadata(&entries);
        assert!(report.findings.iter().any(|f| f.code == "HDR-D05"));
    }

    #[test]
    fn test_dynamic_avg_exceeds_max() {
        let checker = HdrQcChecker::new();
        let entries = vec![DynamicHdrEntry::new(0, 500.0, 800.0, 1000.0)];
        let report = checker.check_dynamic_metadata(&entries);
        assert!(report.findings.iter().any(|f| f.code == "HDR-D03"));
    }

    #[test]
    fn test_compatibility_hdr_with_narrow_gamut() {
        let checker = HdrQcChecker::new();
        let report = checker.check_compatibility(HdrTransferFunction::Pq, ColorGamut::Bt709);
        assert!(report.findings.iter().any(|f| f.code == "HDR-C01"));
    }

    #[test]
    fn test_compatibility_sdr_with_wide_gamut() {
        let checker = HdrQcChecker::new();
        let report = checker.check_compatibility(HdrTransferFunction::Sdr, ColorGamut::Bt2020);
        assert!(report.findings.iter().any(|f| f.code == "HDR-C02"));
    }

    #[test]
    fn test_config_builder() {
        let config = HdrQcConfig::new()
            .with_max_cll_limit(4000.0)
            .with_max_fall_limit(2000.0)
            .with_require_bt2020(false);
        assert!((config.max_cll_limit - 4000.0).abs() < f64::EPSILON);
        assert!((config.max_fall_limit - 2000.0).abs() < f64::EPSILON);
        assert!(!config.require_bt2020);
    }

    #[test]
    fn test_finding_with_recommendation() {
        let finding =
            HdrFinding::new(HdrSeverity::Warning, "T01", "test").with_recommendation("fix it");
        assert_eq!(finding.recommendation.as_deref(), Some("fix it"));
        assert!(!finding.is_failure());
    }

    #[test]
    fn test_report_count_by_severity() {
        let mut report = HdrQcReport::new();
        report.add_finding(HdrFinding::new(HdrSeverity::Info, "I1", "info"));
        report.add_finding(HdrFinding::new(HdrSeverity::Warning, "W1", "warn"));
        report.add_finding(HdrFinding::new(HdrSeverity::Error, "E1", "err"));
        report.add_finding(HdrFinding::new(HdrSeverity::Error, "E2", "err2"));
        assert_eq!(report.count_by_severity(HdrSeverity::Info), 1);
        assert_eq!(report.count_by_severity(HdrSeverity::Warning), 1);
        assert_eq!(report.count_by_severity(HdrSeverity::Error), 2);
        assert_eq!(report.finding_count(), 4);
        assert!(!report.passed);
    }

    #[test]
    fn test_transfer_function_display() {
        assert_eq!(HdrTransferFunction::Pq.to_string(), "PQ (ST.2084)");
        assert_eq!(HdrTransferFunction::Hlg.to_string(), "HLG (BT.2100)");
        assert_eq!(HdrTransferFunction::Linear.to_string(), "Linear");
        assert_eq!(HdrTransferFunction::Sdr.to_string(), "SDR");
    }

    #[test]
    fn test_severity_display_and_ordering() {
        assert!(HdrSeverity::Info < HdrSeverity::Warning);
        assert!(HdrSeverity::Warning < HdrSeverity::Error);
        assert!(HdrSeverity::Error < HdrSeverity::Critical);
        assert_eq!(HdrSeverity::Critical.to_string(), "CRITICAL");
    }
}
