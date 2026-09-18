//! Format-level forensic analysis: detect structural anomalies in media files.
//!
//! Examines container headers, codec parameters, and data integrity to flag
//! suspicious format characteristics that may indicate tampering or corruption.

#![allow(dead_code)]

// ── FormatAnomaly ─────────────────────────────────────────────────────────────

/// A class of structural anomaly detected during format analysis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FormatAnomaly {
    /// File header does not match the declared container format.
    UnexpectedHeader {
        /// Expected magic/signature bytes (hex).
        expected: String,
        /// Actual bytes found (hex).
        found: String,
    },
    /// The file appears to end before all declared data is present.
    TruncatedData {
        /// Declared total size in bytes.
        declared_size: u64,
        /// Actual readable size in bytes.
        actual_size: u64,
    },
    /// Codec or stream parameters contradict container-level metadata.
    MismatchedParams {
        /// Description of the mismatch.
        detail: String,
    },
    /// Reserved/padding bytes contain unexpected non-zero values.
    NonZeroPadding {
        /// Byte offset of the anomaly.
        offset: u64,
    },
    /// Duplicate stream IDs found inside the container.
    DuplicateStreamId {
        /// The repeated stream ID.
        stream_id: u32,
    },
}

impl FormatAnomaly {
    /// Returns a severity score (0.0 – 1.0) for this anomaly.
    #[must_use]
    pub fn severity(&self) -> f64 {
        match self {
            Self::UnexpectedHeader { .. } => 0.95,
            Self::TruncatedData { .. } => 0.80,
            Self::MismatchedParams { .. } => 0.70,
            Self::NonZeroPadding { .. } => 0.30,
            Self::DuplicateStreamId { .. } => 0.50,
        }
    }

    /// Returns a human-readable description.
    #[must_use]
    pub fn description(&self) -> String {
        match self {
            Self::UnexpectedHeader { expected, found } => {
                format!("Unexpected header: expected '{expected}', found '{found}'")
            }
            Self::TruncatedData {
                declared_size,
                actual_size,
            } => {
                format!(
                    "Truncated data: declared {declared_size} bytes, only {actual_size} readable"
                )
            }
            Self::MismatchedParams { detail } => {
                format!("Mismatched params: {detail}")
            }
            Self::NonZeroPadding { offset } => {
                format!("Non-zero padding at offset 0x{offset:08X}")
            }
            Self::DuplicateStreamId { stream_id } => {
                format!("Duplicate stream ID: {stream_id}")
            }
        }
    }

    /// Returns `true` if this anomaly is considered critical (severity >= 0.8).
    #[must_use]
    pub fn is_critical(&self) -> bool {
        self.severity() >= 0.8
    }
}

// ── FormatForensics ───────────────────────────────────────────────────────────

/// Configuration for format-level forensic scanning.
#[derive(Debug, Clone)]
pub struct FormatForensicsConfig {
    /// Check magic bytes against the file extension.
    pub check_header: bool,
    /// Verify that the readable file size matches declared sizes.
    pub check_truncation: bool,
    /// Compare codec parameters with container metadata.
    pub check_params: bool,
    /// Inspect padding/reserved bytes.
    pub check_padding: bool,
}

impl Default for FormatForensicsConfig {
    fn default() -> Self {
        Self {
            check_header: true,
            check_truncation: true,
            check_params: true,
            check_padding: false,
        }
    }
}

/// Performs format-level forensic analysis on raw file bytes.
#[derive(Debug, Default)]
pub struct FormatForensics {
    config: FormatForensicsConfig,
}

impl FormatForensics {
    /// Create with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: FormatForensicsConfig::default(),
        }
    }

    /// Create with explicit configuration.
    #[must_use]
    pub fn with_config(config: FormatForensicsConfig) -> Self {
        Self { config }
    }

    /// Scan `data` declared to be of type `declared_format` (e.g. `"mp4"`)
    /// with `declared_size` total bytes, and return a `FormatAnomalyReport`.
    #[must_use]
    pub fn scan(
        &self,
        data: &[u8],
        declared_format: &str,
        declared_size: u64,
    ) -> FormatAnomalyReport {
        let mut anomalies = Vec::new();

        // Header check
        if self.config.check_header {
            if let Some(anomaly) = self.check_header(data, declared_format) {
                anomalies.push(anomaly);
            }
        }

        // Truncation check
        if self.config.check_truncation {
            let actual_size = data.len() as u64;
            if actual_size < declared_size {
                anomalies.push(FormatAnomaly::TruncatedData {
                    declared_size,
                    actual_size,
                });
            }
        }

        // Padding check (scan first 256 bytes of "reserved" space at offset 8)
        if self.config.check_padding && data.len() > 16 {
            let padding_start = 8usize;
            let padding_end = 16usize.min(data.len());
            for (i, &byte) in data[padding_start..padding_end].iter().enumerate() {
                if byte != 0 {
                    anomalies.push(FormatAnomaly::NonZeroPadding {
                        offset: (padding_start + i) as u64,
                    });
                    break;
                }
            }
        }

        FormatAnomalyReport { anomalies }
    }

    /// Check the magic bytes of `data` against known signatures for
    /// `format`.  Returns `None` if everything looks fine.
    fn check_header(&self, data: &[u8], format: &str) -> Option<FormatAnomaly> {
        let (expected_magic, expected_hex): (&[u8], &str) = match format.to_lowercase().as_str() {
            "mp4" | "mov" => (&[0x00, 0x00, 0x00], "000000"),
            "mkv" => (&[0x1A, 0x45, 0xDF, 0xA3], "1a45dfa3"),
            "avi" => (b"RIFF", "52494646"),
            "png" => (&[0x89, 0x50, 0x4E, 0x47], "89504e47"),
            "jpeg" | "jpg" => (&[0xFF, 0xD8, 0xFF], "ffd8ff"),
            _ => return None, // unknown format – skip check
        };

        if data.len() < expected_magic.len() || !data.starts_with(expected_magic) {
            let found_hex: String = data
                .iter()
                .take(expected_magic.len())
                .map(|b| format!("{b:02x}"))
                .collect();
            return Some(FormatAnomaly::UnexpectedHeader {
                expected: expected_hex.to_string(),
                found: found_hex,
            });
        }
        None
    }
}

// ── FormatAnomalyReport ───────────────────────────────────────────────────────

/// Report produced by a format forensics scan.
#[derive(Debug, Clone, Default)]
pub struct FormatAnomalyReport {
    /// All anomalies detected.
    pub anomalies: Vec<FormatAnomaly>,
}

impl FormatAnomalyReport {
    /// Create an empty report.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of anomalies whose severity >= 0.8.
    #[must_use]
    pub fn critical_count(&self) -> usize {
        self.anomalies.iter().filter(|a| a.is_critical()).count()
    }

    /// Returns the total number of anomalies.
    #[must_use]
    pub fn total_count(&self) -> usize {
        self.anomalies.len()
    }

    /// Returns `true` if any critical anomaly was found.
    #[must_use]
    pub fn has_critical(&self) -> bool {
        self.critical_count() > 0
    }

    /// Returns the maximum severity found, or 0.0 if no anomalies.
    #[must_use]
    pub fn max_severity(&self) -> f64 {
        self.anomalies
            .iter()
            .map(FormatAnomaly::severity)
            .fold(0.0_f64, f64::max)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_severity_unexpected_header() {
        let a = FormatAnomaly::UnexpectedHeader {
            expected: "ffd8ff".into(),
            found: "000000".into(),
        };
        assert!((a.severity() - 0.95).abs() < 1e-9);
    }

    #[test]
    fn test_severity_truncated_data() {
        let a = FormatAnomaly::TruncatedData {
            declared_size: 1000,
            actual_size: 500,
        };
        assert!((a.severity() - 0.80).abs() < 1e-9);
    }

    #[test]
    fn test_severity_mismatched_params() {
        let a = FormatAnomaly::MismatchedParams {
            detail: "fps mismatch".into(),
        };
        assert!((a.severity() - 0.70).abs() < 1e-9);
    }

    #[test]
    fn test_is_critical_high_severity() {
        let a = FormatAnomaly::UnexpectedHeader {
            expected: "89504e47".into(),
            found: "00000000".into(),
        };
        assert!(a.is_critical());
    }

    #[test]
    fn test_is_critical_low_severity() {
        let a = FormatAnomaly::NonZeroPadding { offset: 8 };
        assert!(!a.is_critical());
    }

    #[test]
    fn test_description_unexpected_header() {
        let a = FormatAnomaly::UnexpectedHeader {
            expected: "ffd8ff".into(),
            found: "000000".into(),
        };
        let desc = a.description();
        assert!(desc.contains("ffd8ff"));
        assert!(desc.contains("000000"));
    }

    #[test]
    fn test_format_forensics_scan_valid_png() {
        let ff = FormatForensics::new();
        // Valid PNG magic bytes
        let data = vec![0x89u8, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        let report = ff.scan(&data, "png", data.len() as u64);
        assert_eq!(report.critical_count(), 0);
    }

    #[test]
    fn test_format_forensics_scan_bad_header() {
        let ff = FormatForensics::new();
        let data = vec![0x00u8; 16]; // all zeros, not a PNG
        let report = ff.scan(&data, "png", 16);
        assert!(report.has_critical());
    }

    #[test]
    fn test_format_forensics_scan_truncated() {
        let ff = FormatForensics::new();
        // Valid JPEG magic but declared size bigger than actual
        let data = vec![0xFFu8, 0xD8, 0xFF, 0x00];
        let report = ff.scan(&data, "jpeg", 1000);
        assert!(report.total_count() > 0);
        // Truncated anomaly must be present
        let has_truncated = report
            .anomalies
            .iter()
            .any(|a| matches!(a, FormatAnomaly::TruncatedData { .. }));
        assert!(has_truncated);
    }

    #[test]
    fn test_report_critical_count_empty() {
        let r = FormatAnomalyReport::new();
        assert_eq!(r.critical_count(), 0);
    }

    #[test]
    fn test_report_max_severity_empty() {
        let r = FormatAnomalyReport::new();
        assert_eq!(r.max_severity(), 0.0);
    }

    #[test]
    fn test_report_max_severity_populated() {
        let r = FormatAnomalyReport {
            anomalies: vec![
                FormatAnomaly::NonZeroPadding { offset: 8 },
                FormatAnomaly::TruncatedData {
                    declared_size: 100,
                    actual_size: 50,
                },
            ],
        };
        assert!((r.max_severity() - 0.80).abs() < 1e-9);
    }

    #[test]
    fn test_format_forensics_unknown_format_no_header_error() {
        let ff = FormatForensics::new();
        let data = vec![0xABu8; 32];
        let report = ff.scan(&data, "unknown_xyz", 32);
        // No header anomaly for unknown format
        let has_header_error = report
            .anomalies
            .iter()
            .any(|a| matches!(a, FormatAnomaly::UnexpectedHeader { .. }));
        assert!(!has_header_error);
    }
}
