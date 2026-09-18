//! Format validation for archived media files.
//!
//! This module provides tools to verify that archived files conform to their
//! declared format specifications, detecting corruption, non-conformance,
//! and format-specific issues before long-term storage.

use std::collections::HashMap;

/// Severity level of a validation finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Severity {
    /// Informational note, no action required.
    Info,
    /// Minor issue that may affect long-term preservation.
    Warning,
    /// Significant issue that should be fixed.
    Error,
    /// Critical issue that prevents reliable archiving.
    Critical,
}

impl Severity {
    /// Returns a string label for this severity.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Info => "INFO",
            Self::Warning => "WARNING",
            Self::Error => "ERROR",
            Self::Critical => "CRITICAL",
        }
    }

    /// Returns `true` if the severity is Error or Critical.
    #[must_use]
    pub const fn is_blocking(&self) -> bool {
        matches!(self, Self::Error | Self::Critical)
    }
}

/// A format family that can be validated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FormatFamily {
    /// Matroska container (MKV/MKA/MKS).
    Matroska,
    /// MPEG-4 container (MP4/M4A/M4V).
    Mpeg4,
    /// AVI container.
    Avi,
    /// TIFF image.
    Tiff,
    /// DPX (Digital Picture Exchange) image.
    Dpx,
    /// OpenEXR image.
    OpenExr,
    /// PNG image.
    Png,
    /// JPEG 2000 image.
    Jpeg2000,
    /// WAV audio.
    Wav,
    /// FLAC audio.
    Flac,
    /// PDF document.
    Pdf,
    /// MXF container (Material Exchange Format).
    Mxf,
}

impl FormatFamily {
    /// Returns the typical file extensions for this format.
    #[must_use]
    pub const fn extensions(&self) -> &'static [&'static str] {
        match self {
            Self::Matroska => &["mkv", "mka", "mks"],
            Self::Mpeg4 => &["mp4", "m4a", "m4v"],
            Self::Avi => &["avi"],
            Self::Tiff => &["tif", "tiff"],
            Self::Dpx => &["dpx"],
            Self::OpenExr => &["exr"],
            Self::Png => &["png"],
            Self::Jpeg2000 => &["jp2", "j2k"],
            Self::Wav => &["wav"],
            Self::Flac => &["flac"],
            Self::Pdf => &["pdf"],
            Self::Mxf => &["mxf"],
        }
    }

    /// Returns the expected magic bytes for this format.
    #[must_use]
    pub fn magic_bytes(&self) -> &'static [u8] {
        match self {
            Self::Matroska => &[0x1A, 0x45, 0xDF, 0xA3],
            Self::Mpeg4 => &[], // ftyp at offset 4
            Self::Avi => b"RIFF",
            Self::Tiff => &[0x49, 0x49, 0x2A, 0x00], // little-endian TIFF
            // DPX little-endian ("SDPX") — big-endian ("XPDS") handled via detect_format
            Self::Dpx => &[0x53, 0x44, 0x50, 0x58],
            // OpenEXR magic: 0x76 0x2F 0x31 0x01
            Self::OpenExr => &[0x76, 0x2F, 0x31, 0x01],
            Self::Png => &[0x89, 0x50, 0x4E, 0x47],
            Self::Jpeg2000 => &[0x00, 0x00, 0x00, 0x0C],
            Self::Wav => b"RIFF",
            Self::Flac => b"fLaC",
            Self::Pdf => b"%PDF",
            Self::Mxf => &[0x06, 0x0E, 0x2B, 0x34],
        }
    }
}

/// A single validation finding.
#[derive(Debug, Clone)]
pub struct ValidationFinding {
    /// Severity of the finding.
    pub severity: Severity,
    /// Short code for the finding.
    pub code: String,
    /// Human-readable description.
    pub message: String,
    /// Optional byte offset in the file.
    pub offset: Option<u64>,
}

impl ValidationFinding {
    /// Creates a new validation finding.
    #[must_use]
    pub fn new(severity: Severity, code: &str, message: &str) -> Self {
        Self {
            severity,
            code: code.to_string(),
            message: message.to_string(),
            offset: None,
        }
    }

    /// Sets the byte offset for this finding.
    #[must_use]
    pub const fn with_offset(mut self, offset: u64) -> Self {
        self.offset = Some(offset);
        self
    }
}

/// Overall validation verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationVerdict {
    /// File is fully conformant.
    Pass,
    /// File has warnings but is acceptable.
    PassWithWarnings,
    /// File has errors and should not be archived as-is.
    Fail,
}

/// Result of validating a single file.
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// The format that was validated.
    pub format: FormatFamily,
    /// File size in bytes.
    pub file_size: u64,
    /// All findings from validation.
    pub findings: Vec<ValidationFinding>,
    /// Overall verdict.
    pub verdict: ValidationVerdict,
}

impl ValidationReport {
    /// Creates a new validation report.
    #[must_use]
    pub fn new(format: FormatFamily, file_size: u64) -> Self {
        Self {
            format,
            file_size,
            findings: Vec::new(),
            verdict: ValidationVerdict::Pass,
        }
    }

    /// Adds a finding and recalculates the verdict.
    pub fn add_finding(&mut self, finding: ValidationFinding) {
        self.findings.push(finding);
        self.recalculate_verdict();
    }

    /// Recalculates the verdict based on all findings.
    fn recalculate_verdict(&mut self) {
        let max_severity = self.findings.iter().map(|f| f.severity).max();
        self.verdict = match max_severity {
            None => ValidationVerdict::Pass,
            Some(Severity::Info) | Some(Severity::Warning) => ValidationVerdict::PassWithWarnings,
            Some(Severity::Error) | Some(Severity::Critical) => ValidationVerdict::Fail,
        };
    }

    /// Returns the number of findings at a given severity.
    #[must_use]
    pub fn count_by_severity(&self, severity: Severity) -> usize {
        self.findings
            .iter()
            .filter(|f| f.severity == severity)
            .count()
    }

    /// Returns `true` if there are any blocking findings.
    #[must_use]
    pub fn has_blocking_issues(&self) -> bool {
        self.findings.iter().any(|f| f.severity.is_blocking())
    }

    /// Returns a summary of finding counts by severity.
    #[must_use]
    pub fn summary(&self) -> HashMap<Severity, usize> {
        let mut counts = HashMap::new();
        for finding in &self.findings {
            *counts.entry(finding.severity).or_insert(0) += 1;
        }
        counts
    }
}

/// Configuration for format validation.
#[derive(Debug, Clone)]
pub struct ValidatorConfig {
    /// Whether to check magic bytes.
    pub check_magic: bool,
    /// Whether to validate internal structure.
    pub check_structure: bool,
    /// Whether to check for preservation-recommended settings.
    pub check_preservation: bool,
    /// Minimum file size considered valid (bytes).
    pub min_file_size: u64,
    /// Maximum file size considered valid (bytes, 0 = no limit).
    pub max_file_size: u64,
}

impl Default for ValidatorConfig {
    fn default() -> Self {
        Self {
            check_magic: true,
            check_structure: true,
            check_preservation: true,
            min_file_size: 1,
            max_file_size: 0,
        }
    }
}

/// Validates archived media files against format specifications.
#[derive(Debug)]
pub struct FormatValidator {
    /// Validation configuration.
    pub config: ValidatorConfig,
}

impl FormatValidator {
    /// Creates a new format validator with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: ValidatorConfig::default(),
        }
    }

    /// Creates a new format validator with a specific configuration.
    #[must_use]
    pub fn with_config(config: ValidatorConfig) -> Self {
        Self { config }
    }

    /// Validates a file's magic bytes against the expected format.
    #[must_use]
    pub fn validate_magic(&self, data: &[u8], format: FormatFamily) -> ValidationReport {
        let mut report = ValidationReport::new(format, data.len() as u64);

        if !self.config.check_magic {
            return report;
        }

        let magic = format.magic_bytes();
        if magic.is_empty() {
            report.add_finding(ValidationFinding::new(
                Severity::Info,
                "MAGIC_SKIP",
                "Format does not use simple magic bytes detection",
            ));
            return report;
        }

        if data.len() < magic.len() {
            report.add_finding(ValidationFinding::new(
                Severity::Critical,
                "FILE_TOO_SHORT",
                "File is shorter than expected magic bytes",
            ));
            return report;
        }

        // DPX accepts either little-endian "SDPX" or big-endian "XPDS"
        let matches = if format == FormatFamily::Dpx {
            data.starts_with(b"SDPX") || data.starts_with(b"XPDS")
        } else {
            &data[..magic.len()] == magic
        };

        if !matches {
            report.add_finding(
                ValidationFinding::new(
                    Severity::Error,
                    "MAGIC_MISMATCH",
                    "File magic bytes do not match expected format",
                )
                .with_offset(0),
            );
        }

        report
    }

    /// Validates file size constraints.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn validate_file_size(&self, size: u64, format: FormatFamily) -> ValidationReport {
        let mut report = ValidationReport::new(format, size);

        if size < self.config.min_file_size {
            report.add_finding(ValidationFinding::new(
                Severity::Error,
                "FILE_TOO_SMALL",
                &format!(
                    "File size {} bytes is below minimum {} bytes",
                    size, self.config.min_file_size
                ),
            ));
        }

        if self.config.max_file_size > 0 && size > self.config.max_file_size {
            report.add_finding(ValidationFinding::new(
                Severity::Warning,
                "FILE_TOO_LARGE",
                &format!(
                    "File size {} bytes exceeds maximum {} bytes",
                    size, self.config.max_file_size
                ),
            ));
        }

        report
    }

    /// Detects the format family from file data by examining magic bytes.
    #[must_use]
    pub fn detect_format(data: &[u8]) -> Option<FormatFamily> {
        if data.len() < 4 {
            return None;
        }
        if data.starts_with(&[0x1A, 0x45, 0xDF, 0xA3]) {
            return Some(FormatFamily::Matroska);
        }
        if data.starts_with(&[0x89, 0x50, 0x4E, 0x47]) {
            return Some(FormatFamily::Png);
        }
        if data.starts_with(b"fLaC") {
            return Some(FormatFamily::Flac);
        }
        if data.starts_with(b"%PDF") {
            return Some(FormatFamily::Pdf);
        }
        if data.starts_with(&[0x06, 0x0E, 0x2B, 0x34]) {
            return Some(FormatFamily::Mxf);
        }
        if data.starts_with(&[0x49, 0x49, 0x2A, 0x00]) {
            return Some(FormatFamily::Tiff);
        }
        // DPX: little-endian "SDPX" or big-endian "XPDS"
        if data.starts_with(b"SDPX") || data.starts_with(b"XPDS") {
            return Some(FormatFamily::Dpx);
        }
        // OpenEXR: 0x76 0x2F 0x31 0x01
        if data.starts_with(&[0x76, 0x2F, 0x31, 0x01]) {
            return Some(FormatFamily::OpenExr);
        }
        None
    }
}

impl Default for FormatValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_severity_label() {
        assert_eq!(Severity::Info.label(), "INFO");
        assert_eq!(Severity::Warning.label(), "WARNING");
        assert_eq!(Severity::Error.label(), "ERROR");
        assert_eq!(Severity::Critical.label(), "CRITICAL");
    }

    #[test]
    fn test_severity_is_blocking() {
        assert!(!Severity::Info.is_blocking());
        assert!(!Severity::Warning.is_blocking());
        assert!(Severity::Error.is_blocking());
        assert!(Severity::Critical.is_blocking());
    }

    #[test]
    fn test_format_extensions() {
        let exts = FormatFamily::Matroska.extensions();
        assert!(exts.contains(&"mkv"));
        let exts = FormatFamily::Wav.extensions();
        assert!(exts.contains(&"wav"));
    }

    #[test]
    fn test_validation_report_pass() {
        let report = ValidationReport::new(FormatFamily::Png, 1024);
        assert_eq!(report.verdict, ValidationVerdict::Pass);
        assert!(!report.has_blocking_issues());
    }

    #[test]
    fn test_validation_report_with_warning() {
        let mut report = ValidationReport::new(FormatFamily::Tiff, 2048);
        report.add_finding(ValidationFinding::new(
            Severity::Warning,
            "W001",
            "Minor issue",
        ));
        assert_eq!(report.verdict, ValidationVerdict::PassWithWarnings);
        assert!(!report.has_blocking_issues());
    }

    #[test]
    fn test_validation_report_fail() {
        let mut report = ValidationReport::new(FormatFamily::Flac, 512);
        report.add_finding(ValidationFinding::new(
            Severity::Error,
            "E001",
            "Bad structure",
        ));
        assert_eq!(report.verdict, ValidationVerdict::Fail);
        assert!(report.has_blocking_issues());
    }

    #[test]
    fn test_count_by_severity() {
        let mut report = ValidationReport::new(FormatFamily::Pdf, 4096);
        report.add_finding(ValidationFinding::new(Severity::Info, "I01", "Note"));
        report.add_finding(ValidationFinding::new(Severity::Info, "I02", "Note 2"));
        report.add_finding(ValidationFinding::new(Severity::Warning, "W01", "Warn"));
        assert_eq!(report.count_by_severity(Severity::Info), 2);
        assert_eq!(report.count_by_severity(Severity::Warning), 1);
        assert_eq!(report.count_by_severity(Severity::Error), 0);
    }

    #[test]
    fn test_validate_magic_png() {
        let validator = FormatValidator::new();
        let data = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        let report = validator.validate_magic(&data, FormatFamily::Png);
        assert_eq!(report.verdict, ValidationVerdict::Pass);
    }

    #[test]
    fn test_validate_magic_mismatch() {
        let validator = FormatValidator::new();
        let data = [0x00, 0x00, 0x00, 0x00];
        let report = validator.validate_magic(&data, FormatFamily::Png);
        assert_eq!(report.verdict, ValidationVerdict::Fail);
    }

    #[test]
    fn test_validate_file_size_too_small() {
        let validator = FormatValidator::new();
        let report = validator.validate_file_size(0, FormatFamily::Wav);
        assert_eq!(report.verdict, ValidationVerdict::Fail);
    }

    #[test]
    fn test_detect_format_png() {
        let data = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A];
        assert_eq!(
            FormatValidator::detect_format(&data),
            Some(FormatFamily::Png)
        );
    }

    #[test]
    fn test_detect_format_unknown() {
        let data = [0xFF, 0xFF, 0xFF, 0xFF];
        assert_eq!(FormatValidator::detect_format(&data), None);
    }

    #[test]
    fn test_finding_with_offset() {
        let finding = ValidationFinding::new(Severity::Error, "E01", "Bad byte").with_offset(42);
        assert_eq!(finding.offset, Some(42));
    }

    #[test]
    fn test_dpx_magic_detected() {
        // Little-endian DPX ("SDPX")
        let data_le = *b"SDPX\x00\x00\x00\x00";
        assert_eq!(
            FormatValidator::detect_format(&data_le),
            Some(FormatFamily::Dpx),
            "little-endian DPX magic must be detected"
        );

        // Big-endian DPX ("XPDS")
        let data_be = *b"XPDS\x00\x00\x00\x00";
        assert_eq!(
            FormatValidator::detect_format(&data_be),
            Some(FormatFamily::Dpx),
            "big-endian DPX magic must be detected"
        );

        // validate_magic should pass for both byte orders
        let validator = FormatValidator::new();
        let report_le = validator.validate_magic(&data_le, FormatFamily::Dpx);
        assert_eq!(
            report_le.verdict,
            ValidationVerdict::Pass,
            "LE DPX validate_magic must pass"
        );
        let report_be = validator.validate_magic(&data_be, FormatFamily::Dpx);
        assert_eq!(
            report_be.verdict,
            ValidationVerdict::Pass,
            "BE DPX validate_magic must pass"
        );
    }

    #[test]
    fn test_exr_magic_detected() {
        // OpenEXR magic: 0x76 0x2F 0x31 0x01
        let data: [u8; 8] = [0x76, 0x2F, 0x31, 0x01, 0x02, 0x00, 0x00, 0x00];
        assert_eq!(
            FormatValidator::detect_format(&data),
            Some(FormatFamily::OpenExr),
            "OpenEXR magic must be detected"
        );

        let validator = FormatValidator::new();
        let report = validator.validate_magic(&data, FormatFamily::OpenExr);
        assert_eq!(
            report.verdict,
            ValidationVerdict::Pass,
            "OpenEXR validate_magic must pass for correct magic bytes"
        );
    }
}
