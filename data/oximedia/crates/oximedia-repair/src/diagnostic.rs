#![allow(dead_code)]
//! Deep diagnostic analysis for corrupted media files.
//!
//! Provides comprehensive scanning and reporting of media file health,
//! generating structured diagnostic reports that detail every detected anomaly
//! with precise byte offsets, severity, and suggested remediation actions.

use std::collections::HashMap;

/// Severity level for a diagnostic finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DiagnosticSeverity {
    /// Informational note, no action needed.
    Info,
    /// Minor anomaly that may not affect playback.
    Warning,
    /// Significant problem likely to cause playback issues.
    Error,
    /// Structural failure that prevents any playback.
    Fatal,
}

impl std::fmt::Display for DiagnosticSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Info => write!(f, "INFO"),
            Self::Warning => write!(f, "WARN"),
            Self::Error => write!(f, "ERROR"),
            Self::Fatal => write!(f, "FATAL"),
        }
    }
}

/// A single diagnostic finding within a media file.
#[derive(Debug, Clone)]
pub struct DiagnosticFinding {
    /// Unique code identifying the kind of finding.
    pub code: String,
    /// Severity level.
    pub severity: DiagnosticSeverity,
    /// Human-readable description.
    pub description: String,
    /// Byte offset in the file where the issue was found, if applicable.
    pub byte_offset: Option<u64>,
    /// Length of the affected region in bytes.
    pub affected_bytes: Option<u64>,
    /// Suggested remediation action.
    pub suggestion: String,
    /// Stream index if the finding is stream-specific.
    pub stream_index: Option<usize>,
}

impl DiagnosticFinding {
    /// Create a new diagnostic finding.
    pub fn new(
        code: impl Into<String>,
        severity: DiagnosticSeverity,
        description: impl Into<String>,
        suggestion: impl Into<String>,
    ) -> Self {
        Self {
            code: code.into(),
            severity,
            description: description.into(),
            byte_offset: None,
            affected_bytes: None,
            suggestion: suggestion.into(),
            stream_index: None,
        }
    }

    /// Set the byte offset where the issue was found.
    pub fn with_offset(mut self, offset: u64) -> Self {
        self.byte_offset = Some(offset);
        self
    }

    /// Set the number of affected bytes.
    pub fn with_affected_bytes(mut self, bytes: u64) -> Self {
        self.affected_bytes = Some(bytes);
        self
    }

    /// Set the stream index for this finding.
    pub fn with_stream(mut self, index: usize) -> Self {
        self.stream_index = Some(index);
        self
    }
}

/// Statistics about the overall health of a media file.
#[derive(Debug, Clone)]
pub struct HealthStats {
    /// Total number of findings.
    pub total_findings: usize,
    /// Number of informational findings.
    pub info_count: usize,
    /// Number of warnings.
    pub warning_count: usize,
    /// Number of errors.
    pub error_count: usize,
    /// Number of fatal issues.
    pub fatal_count: usize,
    /// Overall health score from 0.0 (dead) to 1.0 (perfect).
    pub health_score: f64,
    /// Total bytes affected by issues.
    pub total_affected_bytes: u64,
}

impl HealthStats {
    /// Create health stats from a set of findings.
    #[allow(clippy::cast_precision_loss)]
    pub fn from_findings(findings: &[DiagnosticFinding], _total_file_bytes: u64) -> Self {
        let mut info_count = 0usize;
        let mut warning_count = 0usize;
        let mut error_count = 0usize;
        let mut fatal_count = 0usize;
        let mut total_affected: u64 = 0;

        for f in findings {
            match f.severity {
                DiagnosticSeverity::Info => info_count += 1,
                DiagnosticSeverity::Warning => warning_count += 1,
                DiagnosticSeverity::Error => error_count += 1,
                DiagnosticSeverity::Fatal => fatal_count += 1,
            }
            if let Some(ab) = f.affected_bytes {
                total_affected = total_affected.saturating_add(ab);
            }
        }

        // Compute a health score: fatals tank it, errors reduce it, warnings are minor
        let penalty =
            fatal_count as f64 * 0.40 + error_count as f64 * 0.15 + warning_count as f64 * 0.03;
        let health_score = (1.0 - penalty).clamp(0.0, 1.0);

        Self {
            total_findings: findings.len(),
            info_count,
            warning_count,
            error_count,
            fatal_count,
            health_score,
            total_affected_bytes: total_affected,
        }
    }

    /// Returns `true` if the file is considered playable (health > 0.5 and no fatals).
    pub fn is_playable(&self) -> bool {
        self.fatal_count == 0 && self.health_score > 0.5
    }
}

/// A complete diagnostic report for a media file.
#[derive(Debug, Clone)]
pub struct DiagnosticReport {
    /// File path that was analyzed.
    pub file_path: String,
    /// File size in bytes.
    pub file_size: u64,
    /// Ordered list of findings.
    pub findings: Vec<DiagnosticFinding>,
    /// Overall health statistics.
    pub stats: HealthStats,
    /// Per-stream health scores.
    pub stream_scores: HashMap<usize, f64>,
}

impl DiagnosticReport {
    /// Create a new diagnostic report.
    pub fn new(file_path: impl Into<String>, file_size: u64) -> Self {
        let stats = HealthStats::from_findings(&[], file_size);
        Self {
            file_path: file_path.into(),
            file_size,
            findings: Vec::new(),
            stats,
            stream_scores: HashMap::new(),
        }
    }

    /// Add a finding to the report and recalculate statistics.
    pub fn add_finding(&mut self, finding: DiagnosticFinding) {
        self.findings.push(finding);
        self.stats = HealthStats::from_findings(&self.findings, self.file_size);
        self.recalculate_stream_scores();
    }

    /// Return only findings at or above the given severity threshold.
    pub fn findings_at_severity(&self, min: DiagnosticSeverity) -> Vec<&DiagnosticFinding> {
        self.findings.iter().filter(|f| f.severity >= min).collect()
    }

    /// Return findings for a specific stream.
    pub fn findings_for_stream(&self, stream_idx: usize) -> Vec<&DiagnosticFinding> {
        self.findings
            .iter()
            .filter(|f| f.stream_index == Some(stream_idx))
            .collect()
    }

    /// Check whether the report indicates a repairable file.
    pub fn is_repairable(&self) -> bool {
        // Fatal-only findings that aren't header-related might be unrecoverable
        self.stats.fatal_count <= 1 && self.stats.health_score > 0.1
    }

    /// Recalculate per-stream health scores.
    #[allow(clippy::cast_precision_loss)]
    fn recalculate_stream_scores(&mut self) {
        let mut stream_findings: HashMap<usize, Vec<&DiagnosticFinding>> = HashMap::new();
        for f in &self.findings {
            if let Some(idx) = f.stream_index {
                stream_findings.entry(idx).or_default().push(f);
            }
        }

        self.stream_scores.clear();
        for (idx, findings) in &stream_findings {
            let mut penalty = 0.0_f64;
            for f in findings {
                match f.severity {
                    DiagnosticSeverity::Fatal => penalty += 0.50,
                    DiagnosticSeverity::Error => penalty += 0.20,
                    DiagnosticSeverity::Warning => penalty += 0.05,
                    DiagnosticSeverity::Info => {}
                }
            }
            self.stream_scores
                .insert(*idx, (1.0 - penalty).clamp(0.0, 1.0));
        }
    }
}

/// Byte-pattern scanner for detecting corruption signatures.
#[derive(Debug)]
pub struct PatternScanner {
    /// Known-good sync-word patterns mapped by container type.
    patterns: HashMap<String, Vec<u8>>,
    /// Window size for scanning.
    window_size: usize,
}

impl PatternScanner {
    /// Create a new pattern scanner with default patterns.
    pub fn new() -> Self {
        let mut patterns = HashMap::new();
        // MP4 'ftyp' atom marker
        patterns.insert("mp4_ftyp".to_string(), vec![0x66, 0x74, 0x79, 0x70]);
        // MPEG-TS sync byte
        patterns.insert("ts_sync".to_string(), vec![0x47]);
        // RIFF header
        patterns.insert("riff".to_string(), vec![0x52, 0x49, 0x46, 0x46]);
        // Matroska/WebM EBML header
        patterns.insert("ebml".to_string(), vec![0x1A, 0x45, 0xDF, 0xA3]);

        Self {
            patterns,
            window_size: 4096,
        }
    }

    /// Set the scanning window size.
    pub fn with_window_size(mut self, size: usize) -> Self {
        self.window_size = size.max(64);
        self
    }

    /// Register a custom sync pattern.
    pub fn register_pattern(&mut self, name: impl Into<String>, pattern: Vec<u8>) {
        self.patterns.insert(name.into(), pattern);
    }

    /// Scan a byte buffer for known patterns, returning offsets of matches.
    pub fn scan_buffer(&self, data: &[u8], pattern_name: &str) -> Vec<usize> {
        let mut results = Vec::new();
        if let Some(pattern) = self.patterns.get(pattern_name) {
            if pattern.is_empty() || data.len() < pattern.len() {
                return results;
            }
            for i in 0..=(data.len() - pattern.len()) {
                if data[i..i + pattern.len()] == *pattern {
                    results.push(i);
                }
            }
        }
        results
    }

    /// Count occurrences of zero-byte runs longer than `min_run` in the given data.
    pub fn count_zero_runs(&self, data: &[u8], min_run: usize) -> usize {
        if min_run == 0 {
            return 0;
        }
        let mut count = 0usize;
        let mut run_len = 0usize;
        for &b in data {
            if b == 0 {
                run_len += 1;
            } else {
                if run_len >= min_run {
                    count += 1;
                }
                run_len = 0;
            }
        }
        if run_len >= min_run {
            count += 1;
        }
        count
    }

    /// Return the window size.
    pub fn window_size(&self) -> usize {
        self.window_size
    }
}

impl Default for PatternScanner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diagnostic_severity_ordering() {
        assert!(DiagnosticSeverity::Info < DiagnosticSeverity::Warning);
        assert!(DiagnosticSeverity::Warning < DiagnosticSeverity::Error);
        assert!(DiagnosticSeverity::Error < DiagnosticSeverity::Fatal);
    }

    #[test]
    fn test_diagnostic_severity_display() {
        assert_eq!(format!("{}", DiagnosticSeverity::Info), "INFO");
        assert_eq!(format!("{}", DiagnosticSeverity::Fatal), "FATAL");
    }

    #[test]
    fn test_finding_builder() {
        let finding = DiagnosticFinding::new(
            "HDR001",
            DiagnosticSeverity::Error,
            "Corrupt header",
            "Re-write header from stream data",
        )
        .with_offset(0)
        .with_affected_bytes(128)
        .with_stream(0);

        assert_eq!(finding.code, "HDR001");
        assert_eq!(finding.byte_offset, Some(0));
        assert_eq!(finding.affected_bytes, Some(128));
        assert_eq!(finding.stream_index, Some(0));
    }

    #[test]
    fn test_health_stats_perfect() {
        let stats = HealthStats::from_findings(&[], 1_000_000);
        assert_eq!(stats.total_findings, 0);
        assert!((stats.health_score - 1.0).abs() < f64::EPSILON);
        assert!(stats.is_playable());
    }

    #[test]
    fn test_health_stats_with_fatal() {
        let findings = vec![DiagnosticFinding::new(
            "FATAL01",
            DiagnosticSeverity::Fatal,
            "Completely destroyed header",
            "Rebuild from scratch",
        )];
        let stats = HealthStats::from_findings(&findings, 1_000_000);
        assert_eq!(stats.fatal_count, 1);
        assert!(stats.health_score < 1.0);
        assert!(!stats.is_playable());
    }

    #[test]
    fn test_health_stats_mixed() {
        let findings = vec![
            DiagnosticFinding::new("W01", DiagnosticSeverity::Warning, "minor", "fix"),
            DiagnosticFinding::new("E01", DiagnosticSeverity::Error, "bad", "repair"),
            DiagnosticFinding::new("I01", DiagnosticSeverity::Info, "note", "none"),
        ];
        let stats = HealthStats::from_findings(&findings, 500_000);
        assert_eq!(stats.total_findings, 3);
        assert_eq!(stats.warning_count, 1);
        assert_eq!(stats.error_count, 1);
        assert_eq!(stats.info_count, 1);
        assert!(stats.health_score > 0.5);
    }

    #[test]
    fn test_diagnostic_report_creation() {
        let report = DiagnosticReport::new("/test/video.mp4", 5_000_000);
        assert_eq!(report.file_size, 5_000_000);
        assert!(report.findings.is_empty());
        assert!(report.is_repairable());
    }

    #[test]
    fn test_diagnostic_report_add_finding() {
        let mut report = DiagnosticReport::new("/test/video.mp4", 5_000_000);
        report.add_finding(DiagnosticFinding::new(
            "IDX001",
            DiagnosticSeverity::Error,
            "Broken index",
            "Rebuild index",
        ));
        assert_eq!(report.findings.len(), 1);
        assert_eq!(report.stats.error_count, 1);
    }

    #[test]
    fn test_report_findings_at_severity() {
        let mut report = DiagnosticReport::new("/test/a.mkv", 1_000);
        report.add_finding(DiagnosticFinding::new(
            "I1",
            DiagnosticSeverity::Info,
            "ok",
            "none",
        ));
        report.add_finding(DiagnosticFinding::new(
            "E1",
            DiagnosticSeverity::Error,
            "bad",
            "fix",
        ));
        let errors = report.findings_at_severity(DiagnosticSeverity::Error);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].code, "E1");
    }

    #[test]
    fn test_report_findings_for_stream() {
        let mut report = DiagnosticReport::new("/test/b.mp4", 2_000);
        report.add_finding(
            DiagnosticFinding::new("S0", DiagnosticSeverity::Warning, "a", "b").with_stream(0),
        );
        report.add_finding(
            DiagnosticFinding::new("S1", DiagnosticSeverity::Warning, "c", "d").with_stream(1),
        );
        assert_eq!(report.findings_for_stream(0).len(), 1);
        assert_eq!(report.findings_for_stream(1).len(), 1);
        assert_eq!(report.findings_for_stream(2).len(), 0);
    }

    #[test]
    fn test_pattern_scanner_default() {
        let scanner = PatternScanner::new();
        assert_eq!(scanner.window_size(), 4096);
        assert!(scanner.patterns.contains_key("mp4_ftyp"));
    }

    #[test]
    fn test_pattern_scanner_scan_buffer() {
        let scanner = PatternScanner::new();
        let data = vec![0x00, 0x66, 0x74, 0x79, 0x70, 0x00];
        let matches = scanner.scan_buffer(&data, "mp4_ftyp");
        assert_eq!(matches, vec![1]);
    }

    #[test]
    fn test_pattern_scanner_no_match() {
        let scanner = PatternScanner::new();
        let data = vec![0x00, 0x01, 0x02, 0x03];
        let matches = scanner.scan_buffer(&data, "mp4_ftyp");
        assert!(matches.is_empty());
    }

    #[test]
    fn test_count_zero_runs() {
        let scanner = PatternScanner::new();
        let data = vec![1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 3];
        assert_eq!(scanner.count_zero_runs(&data, 4), 2);
        assert_eq!(scanner.count_zero_runs(&data, 5), 1);
        assert_eq!(scanner.count_zero_runs(&data, 6), 0);
    }

    #[test]
    fn test_stream_scores_recalculation() {
        let mut report = DiagnosticReport::new("/test/c.ts", 10_000);
        report.add_finding(
            DiagnosticFinding::new("V1", DiagnosticSeverity::Error, "video err", "fix")
                .with_stream(0),
        );
        report.add_finding(
            DiagnosticFinding::new("A1", DiagnosticSeverity::Warning, "audio warn", "fix")
                .with_stream(1),
        );
        assert!(report.stream_scores[&0] < report.stream_scores[&1]);
    }
}
