//! Batch forensic analysis with parallel processing and report output.
//!
//! Provides `BatchForensicAnalyzer` for analysing multiple files in parallel,
//! collecting per-file results, and generating CSV or JSON reports.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Per-file analysis result
// ---------------------------------------------------------------------------

/// Result of forensic analysis for a single file in a batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileForensicResult {
    /// File path or identifier.
    pub file_path: String,
    /// Whether tampering was detected.
    pub tampering_detected: bool,
    /// Overall confidence score (0.0..1.0).
    pub confidence: f64,
    /// Per-test scores.
    pub test_scores: HashMap<String, f64>,
    /// Textual findings.
    pub findings: Vec<String>,
    /// Error message if analysis failed.
    pub error: Option<String>,
}

impl FileForensicResult {
    /// Create a result for a successful analysis.
    #[must_use]
    pub fn success(
        file_path: &str,
        tampering_detected: bool,
        confidence: f64,
        test_scores: HashMap<String, f64>,
        findings: Vec<String>,
    ) -> Self {
        Self {
            file_path: file_path.to_string(),
            tampering_detected,
            confidence,
            test_scores,
            findings,
            error: None,
        }
    }

    /// Create a result for a failed analysis.
    #[must_use]
    pub fn failure(file_path: &str, error: &str) -> Self {
        Self {
            file_path: file_path.to_string(),
            tampering_detected: false,
            confidence: 0.0,
            test_scores: HashMap::new(),
            findings: Vec::new(),
            error: Some(error.to_string()),
        }
    }

    /// Returns `true` if this file was flagged as suspicious (confidence > 0.5).
    #[must_use]
    pub fn is_suspicious(&self) -> bool {
        self.tampering_detected && self.confidence > 0.5
    }
}

// ---------------------------------------------------------------------------
// Batch report
// ---------------------------------------------------------------------------

/// Aggregate report for a batch of forensic analyses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchReport {
    /// All per-file results.
    pub results: Vec<FileForensicResult>,
    /// Total files analysed.
    pub total_files: usize,
    /// Number of files flagged as tampered.
    pub tampered_count: usize,
    /// Number of files that failed analysis.
    pub error_count: usize,
    /// Average confidence across successful analyses.
    pub average_confidence: f64,
}

impl BatchReport {
    /// Build a report from a list of individual results.
    #[must_use]
    pub fn from_results(results: Vec<FileForensicResult>) -> Self {
        let total_files = results.len();
        let tampered_count = results.iter().filter(|r| r.tampering_detected).count();
        let error_count = results.iter().filter(|r| r.error.is_some()).count();

        let successful: Vec<&FileForensicResult> =
            results.iter().filter(|r| r.error.is_none()).collect();
        let average_confidence = if successful.is_empty() {
            0.0
        } else {
            successful.iter().map(|r| r.confidence).sum::<f64>() / successful.len() as f64
        };

        Self {
            results,
            total_files,
            tampered_count,
            error_count,
            average_confidence,
        }
    }

    /// Return only the suspicious results (tampering detected, confidence > 0.5).
    #[must_use]
    pub fn suspicious_files(&self) -> Vec<&FileForensicResult> {
        self.results.iter().filter(|r| r.is_suspicious()).collect()
    }

    /// Return only the results that encountered errors.
    #[must_use]
    pub fn failed_files(&self) -> Vec<&FileForensicResult> {
        self.results.iter().filter(|r| r.error.is_some()).collect()
    }

    /// Serialize the report to JSON.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Serialize the report to CSV format.
    ///
    /// Columns: file_path, tampering_detected, confidence, error, findings
    #[must_use]
    pub fn to_csv(&self) -> String {
        let mut out = String::from("file_path,tampering_detected,confidence,error,findings\n");
        for r in &self.results {
            let escaped_path = csv_escape(&r.file_path);
            let error_str = r.error.as_deref().unwrap_or("");
            let findings_str = r.findings.join("; ");
            out.push_str(&format!(
                "{},{},{:.4},{},{}\n",
                escaped_path,
                r.tampering_detected,
                r.confidence,
                csv_escape(error_str),
                csv_escape(&findings_str),
            ));
        }
        out
    }
}

/// Escape a value for CSV output: wrap in quotes if it contains commas,
/// quotes, or newlines.
fn csv_escape(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

// ---------------------------------------------------------------------------
// BatchForensicAnalyzer
// ---------------------------------------------------------------------------

/// A user-provided analysis function that takes file path + data and returns
/// a `FileForensicResult`.
pub type AnalysisFn = dyn Fn(&str, &[u8]) -> FileForensicResult + Send + Sync;

/// Batch forensic analyzer for processing multiple files in parallel.
///
/// The analyzer accepts a closure that performs the actual per-file forensic
/// analysis, allowing full flexibility in which tests are run.
pub struct BatchForensicAnalyzer {
    /// Maximum number of parallel analyses (0 = use rayon default).
    pub max_parallelism: usize,
}

impl Default for BatchForensicAnalyzer {
    fn default() -> Self {
        Self { max_parallelism: 0 }
    }
}

impl BatchForensicAnalyzer {
    /// Create a new batch analyzer with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new batch analyzer with a specific parallelism limit.
    #[must_use]
    pub fn with_parallelism(max_parallelism: usize) -> Self {
        Self { max_parallelism }
    }

    /// Analyse a batch of files in parallel.
    ///
    /// Each item in `files` is a `(path, data)` tuple. The `analyze_fn`
    /// closure is called for each file and receives the path and data.
    ///
    /// Results are collected into a `BatchReport`.
    pub fn analyze_batch(&self, files: &[(&str, &[u8])], analyze_fn: &AnalysisFn) -> BatchReport {
        let results: Vec<FileForensicResult> = files
            .par_iter()
            .map(|(path, data)| analyze_fn(path, data))
            .collect();

        BatchReport::from_results(results)
    }

    /// Analyse a batch of files given as `(path, data)` owned pairs.
    pub fn analyze_batch_owned(
        &self,
        files: &[(String, Vec<u8>)],
        analyze_fn: &AnalysisFn,
    ) -> BatchReport {
        let results: Vec<FileForensicResult> = files
            .par_iter()
            .map(|(path, data)| analyze_fn(path, data))
            .collect();

        BatchReport::from_results(results)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_analyzer(path: &str, data: &[u8]) -> FileForensicResult {
        if data.is_empty() {
            return FileForensicResult::failure(path, "Empty data");
        }
        let mean = data.iter().map(|&b| f64::from(b)).sum::<f64>() / data.len() as f64;
        let confidence = mean / 255.0;
        let tampering = confidence > 0.5;
        let mut scores = HashMap::new();
        scores.insert("brightness".to_string(), confidence);
        FileForensicResult::success(
            path,
            tampering,
            confidence,
            scores,
            vec![format!("Mean brightness: {:.2}", mean)],
        )
    }

    // ── FileForensicResult ──────────────────────────────────────────────────

    #[test]
    fn test_file_result_success() {
        let r = FileForensicResult::success(
            "test.jpg",
            true,
            0.8,
            HashMap::new(),
            vec!["finding".to_string()],
        );
        assert_eq!(r.file_path, "test.jpg");
        assert!(r.tampering_detected);
        assert!(r.error.is_none());
    }

    #[test]
    fn test_file_result_failure() {
        let r = FileForensicResult::failure("test.jpg", "bad data");
        assert!(!r.tampering_detected);
        assert_eq!(r.error.as_deref(), Some("bad data"));
    }

    #[test]
    fn test_file_result_is_suspicious() {
        let r = FileForensicResult::success("a.jpg", true, 0.7, HashMap::new(), Vec::new());
        assert!(r.is_suspicious());
    }

    #[test]
    fn test_file_result_not_suspicious_low_confidence() {
        let r = FileForensicResult::success("a.jpg", true, 0.3, HashMap::new(), Vec::new());
        assert!(!r.is_suspicious());
    }

    #[test]
    fn test_file_result_not_suspicious_no_tampering() {
        let r = FileForensicResult::success("a.jpg", false, 0.9, HashMap::new(), Vec::new());
        assert!(!r.is_suspicious());
    }

    // ── BatchReport ─────────────────────────────────────────────────────────

    #[test]
    fn test_batch_report_from_results_empty() {
        let report = BatchReport::from_results(Vec::new());
        assert_eq!(report.total_files, 0);
        assert_eq!(report.tampered_count, 0);
        assert_eq!(report.error_count, 0);
        assert!(report.average_confidence.abs() < 1e-10);
    }

    #[test]
    fn test_batch_report_from_results_mixed() {
        let results = vec![
            FileForensicResult::success("a.jpg", true, 0.8, HashMap::new(), Vec::new()),
            FileForensicResult::success("b.jpg", false, 0.2, HashMap::new(), Vec::new()),
            FileForensicResult::failure("c.jpg", "bad format"),
        ];
        let report = BatchReport::from_results(results);
        assert_eq!(report.total_files, 3);
        assert_eq!(report.tampered_count, 1);
        assert_eq!(report.error_count, 1);
        // Average confidence over successful: (0.8 + 0.2) / 2 = 0.5
        assert!((report.average_confidence - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_batch_report_suspicious_files() {
        let results = vec![
            FileForensicResult::success("a.jpg", true, 0.8, HashMap::new(), Vec::new()),
            FileForensicResult::success("b.jpg", true, 0.3, HashMap::new(), Vec::new()),
            FileForensicResult::success("c.jpg", false, 0.9, HashMap::new(), Vec::new()),
        ];
        let report = BatchReport::from_results(results);
        let suspicious = report.suspicious_files();
        assert_eq!(suspicious.len(), 1);
        assert_eq!(suspicious[0].file_path, "a.jpg");
    }

    #[test]
    fn test_batch_report_failed_files() {
        let results = vec![
            FileForensicResult::success("a.jpg", false, 0.1, HashMap::new(), Vec::new()),
            FileForensicResult::failure("b.jpg", "oops"),
        ];
        let report = BatchReport::from_results(results);
        let failed = report.failed_files();
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].file_path, "b.jpg");
    }

    // ── JSON output ─────────────────────────────────────────────────────────

    #[test]
    fn test_batch_report_to_json() {
        let results = vec![FileForensicResult::success(
            "test.jpg",
            false,
            0.1,
            HashMap::new(),
            Vec::new(),
        )];
        let report = BatchReport::from_results(results);
        let json = report.to_json().expect("JSON serialization should work");
        assert!(json.contains("test.jpg"));
        assert!(json.contains("total_files"));
    }

    // ── CSV output ──────────────────────────────────────────────────────────

    #[test]
    fn test_batch_report_to_csv_header() {
        let report = BatchReport::from_results(Vec::new());
        let csv = report.to_csv();
        assert!(csv.starts_with("file_path,tampering_detected,confidence,error,findings\n"));
    }

    #[test]
    fn test_batch_report_to_csv_content() {
        let results = vec![
            FileForensicResult::success(
                "a.jpg",
                true,
                0.8,
                HashMap::new(),
                vec!["tampered".to_string()],
            ),
            FileForensicResult::failure("b.jpg", "bad"),
        ];
        let report = BatchReport::from_results(results);
        let csv = report.to_csv();
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 3); // header + 2 rows
        assert!(lines[1].contains("a.jpg"));
        assert!(lines[2].contains("b.jpg"));
    }

    #[test]
    fn test_csv_escape_no_special_chars() {
        assert_eq!(csv_escape("hello"), "hello");
    }

    #[test]
    fn test_csv_escape_with_comma() {
        let escaped = csv_escape("hello, world");
        assert!(escaped.starts_with('"'));
        assert!(escaped.ends_with('"'));
    }

    #[test]
    fn test_csv_escape_with_quotes() {
        let escaped = csv_escape("say \"hi\"");
        assert!(escaped.contains("\"\""));
    }

    // ── BatchForensicAnalyzer ───────────────────────────────────────────────

    #[test]
    fn test_batch_analyzer_empty_batch() {
        let analyzer = BatchForensicAnalyzer::new();
        let files: Vec<(&str, &[u8])> = Vec::new();
        let report = analyzer.analyze_batch(&files, &dummy_analyzer);
        assert_eq!(report.total_files, 0);
    }

    #[test]
    fn test_batch_analyzer_single_file() {
        let analyzer = BatchForensicAnalyzer::new();
        let data = vec![200u8; 100];
        let files = vec![("test.jpg", data.as_slice())];
        let report = analyzer.analyze_batch(&files, &dummy_analyzer);
        assert_eq!(report.total_files, 1);
        assert_eq!(report.results[0].file_path, "test.jpg");
    }

    #[test]
    fn test_batch_analyzer_multiple_files() {
        let analyzer = BatchForensicAnalyzer::new();
        let dark = vec![50u8; 100];
        let bright = vec![200u8; 100];
        let empty: Vec<u8> = Vec::new();
        let files = vec![
            ("dark.jpg", dark.as_slice()),
            ("bright.jpg", bright.as_slice()),
            ("empty.bin", empty.as_slice()),
        ];
        let report = analyzer.analyze_batch(&files, &dummy_analyzer);
        assert_eq!(report.total_files, 3);
        assert_eq!(report.error_count, 1); // empty file
    }

    #[test]
    fn test_batch_analyzer_parallel_correctness() {
        let analyzer = BatchForensicAnalyzer::with_parallelism(4);
        let data: Vec<(String, Vec<u8>)> = (0..20)
            .map(|i| {
                let brightness = ((i * 13) % 256) as u8;
                (format!("file_{i}.jpg"), vec![brightness; 64])
            })
            .collect();
        let report = analyzer.analyze_batch_owned(&data, &dummy_analyzer);
        assert_eq!(report.total_files, 20);
        // All should succeed (non-empty data)
        assert_eq!(report.error_count, 0);
    }

    #[test]
    fn test_batch_analyzer_default() {
        let analyzer = BatchForensicAnalyzer::default();
        assert_eq!(analyzer.max_parallelism, 0);
    }

    #[test]
    fn test_batch_report_all_errors() {
        let results = vec![
            FileForensicResult::failure("a.jpg", "bad"),
            FileForensicResult::failure("b.jpg", "bad"),
        ];
        let report = BatchReport::from_results(results);
        assert_eq!(report.error_count, 2);
        assert!(report.average_confidence.abs() < 1e-10);
    }
}
