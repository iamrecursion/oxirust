//! Batch processing support for QC validation.
//!
//! Provides utilities for processing multiple files in parallel,
//! with progress tracking and result aggregation.

use crate::{report::QcReport, QualityControl};
use oximedia_core::{OxiError, OxiResult};
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Batch processing results.
#[derive(Debug)]
#[cfg_attr(feature = "json", derive(serde::Serialize, serde::Deserialize))]
pub struct BatchResults {
    /// Total number of files processed.
    pub total_files: usize,

    /// Number of files that passed validation.
    pub passed: usize,

    /// Number of files that failed validation.
    pub failed: usize,

    /// Number of files that encountered errors during processing.
    pub errors: usize,

    /// Individual file reports.
    pub reports: Vec<BatchFileReport>,

    /// Total processing time in seconds.
    pub total_duration: f64,
}

/// Report for a single file in batch processing.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "json", derive(serde::Serialize, serde::Deserialize))]
pub struct BatchFileReport {
    /// File path.
    pub file_path: String,

    /// Whether the file passed validation.
    pub passed: bool,

    /// Number of errors found.
    pub error_count: usize,

    /// Number of warnings found.
    pub warning_count: usize,

    /// Processing duration in seconds.
    pub duration: f64,

    /// Optional detailed report (can be omitted for summary).
    #[cfg_attr(feature = "json", serde(skip_serializing_if = "Option::is_none"))]
    pub detailed_report: Option<QcReport>,
}

/// Batch processor for QC validation.
pub struct BatchProcessor {
    qc: Arc<QualityControl>,
    include_detailed_reports: bool,
    parallel_jobs: Option<usize>,
    /// If `true`, stop running further rules on a file as soon as a
    /// [`crate::Severity::Error`] (or Critical) result is encountered.
    abort_on_critical: bool,
}

impl BatchProcessor {
    /// Creates a new batch processor with the given QC configuration.
    #[must_use]
    pub fn new(qc: QualityControl) -> Self {
        Self {
            qc: Arc::new(qc),
            include_detailed_reports: true,
            parallel_jobs: None,
            abort_on_critical: false,
        }
    }

    /// Sets whether to include detailed reports in results.
    #[must_use]
    pub const fn with_detailed_reports(mut self, include: bool) -> Self {
        self.include_detailed_reports = include;
        self
    }

    /// Sets the number of parallel jobs (defaults to number of CPUs).
    #[must_use]
    pub const fn with_parallel_jobs(mut self, jobs: usize) -> Self {
        self.parallel_jobs = Some(jobs);
        self
    }

    /// Enables or disables early termination on critical/error results.
    ///
    /// When `true`, rule evaluation for a file stops as soon as any
    /// [`crate::Severity::Error`] (or Critical) result is encountered.
    /// When `false` (default), all rules are always run.
    #[must_use]
    pub const fn with_abort_on_critical(mut self, abort: bool) -> Self {
        self.abort_on_critical = abort;
        self
    }

    /// Processes a list of files.
    ///
    /// # Errors
    ///
    /// Returns an error if the batch processing configuration is invalid.
    pub fn process_files(&self, file_paths: Vec<PathBuf>) -> OxiResult<BatchResults> {
        let start_time = std::time::Instant::now();

        // Configure thread pool if specified
        if let Some(jobs) = self.parallel_jobs {
            rayon::ThreadPoolBuilder::new()
                .num_threads(jobs)
                .build()
                .map_err(|e| {
                    OxiError::Io(std::io::Error::other(format!(
                        "Failed to create thread pool: {e}"
                    )))
                })?;
        }

        let reports = Arc::new(Mutex::new(Vec::new()));
        let passed = Arc::new(Mutex::new(0usize));
        let failed = Arc::new(Mutex::new(0usize));
        let errors = Arc::new(Mutex::new(0usize));

        let abort = self.abort_on_critical;
        file_paths.par_iter().for_each(|path| {
            let path_str = path.to_string_lossy().to_string();
            let file_start = std::time::Instant::now();

            // Use abort-aware validation when configured; otherwise normal path.
            let validate_result = if abort {
                self.qc
                    .validate_with_abort(&path_str, true)
                    .map(|(report, _rules_run)| report)
            } else {
                self.qc.validate(&path_str)
            };

            match validate_result {
                Ok(report) => {
                    let duration = file_start.elapsed().as_secs_f64();
                    let error_count = report.errors().len() + report.critical_errors().len();
                    let warning_count = report.warnings().len();
                    let file_passed = report.overall_passed;

                    let batch_report = BatchFileReport {
                        file_path: path_str,
                        passed: file_passed,
                        error_count,
                        warning_count,
                        duration,
                        detailed_report: if self.include_detailed_reports {
                            Some(report)
                        } else {
                            None
                        },
                    };

                    if file_passed {
                        *passed.lock().unwrap_or_else(|e| e.into_inner()) += 1;
                    } else {
                        *failed.lock().unwrap_or_else(|e| e.into_inner()) += 1;
                    }

                    reports
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .push(batch_report);
                }
                Err(e) => {
                    tracing::error!(file = %path_str, error = %e, "Validation failed");
                    *errors.lock().unwrap_or_else(|e| e.into_inner()) += 1;

                    let batch_report = BatchFileReport {
                        file_path: path_str,
                        passed: false,
                        error_count: 1,
                        warning_count: 0,
                        duration: file_start.elapsed().as_secs_f64(),
                        detailed_report: None,
                    };

                    reports
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .push(batch_report);
                }
            }
        });

        let total_duration = start_time.elapsed().as_secs_f64();
        let reports = match Arc::try_unwrap(reports) {
            Ok(mutex) => mutex.into_inner().unwrap_or_else(|e| e.into_inner()),
            Err(arc) => arc.lock().unwrap_or_else(|e| e.into_inner()).clone(),
        };
        let passed = *passed.lock().unwrap_or_else(|e| e.into_inner());
        let failed = *failed.lock().unwrap_or_else(|e| e.into_inner());
        let errors = *errors.lock().unwrap_or_else(|e| e.into_inner());

        Ok(BatchResults {
            total_files: file_paths.len(),
            passed,
            failed,
            errors,
            reports,
            total_duration,
        })
    }

    /// Processes all files in a directory matching a pattern.
    ///
    /// # Errors
    ///
    /// Returns an error if directory reading fails or batch processing fails.
    pub fn process_directory(&self, dir: &Path, pattern: &str) -> OxiResult<BatchResults> {
        let mut file_paths = Vec::new();

        for entry in std::fs::read_dir(dir).map_err(OxiError::Io)? {
            let entry = entry.map_err(OxiError::Io)?;
            let path = entry.path();

            if path.is_file() {
                let path_str = path.to_string_lossy();
                if path_str.ends_with(pattern) || pattern == "*" {
                    file_paths.push(path);
                }
            }
        }

        self.process_files(file_paths)
    }
}

impl BatchResults {
    /// Generates a summary of the batch results.
    #[must_use]
    pub fn summary(&self) -> String {
        let mut summary = String::new();
        summary.push_str("Batch QC Results\n");
        summary.push_str("================\n\n");
        summary.push_str(&format!("Total Files: {}\n", self.total_files));
        summary.push_str(&format!("Passed: {}\n", self.passed));
        summary.push_str(&format!("Failed: {}\n", self.failed));
        summary.push_str(&format!("Errors: {}\n", self.errors));
        summary.push_str(&format!("Total Duration: {:.2}s\n\n", self.total_duration));

        if self.failed > 0 || self.errors > 0 {
            summary.push_str("Failed Files:\n");
            for report in &self.reports {
                if !report.passed {
                    summary.push_str(&format!(
                        "  {} - {} errors, {} warnings ({:.2}s)\n",
                        report.file_path, report.error_count, report.warning_count, report.duration
                    ));
                }
            }
        }

        summary
    }

    /// Exports batch results as JSON.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    #[cfg(feature = "json")]
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        rules::{CheckResult, QcContext, QcRule, RuleCategory, Severity},
        QcPreset, QualityControl,
    };
    use oximedia_core::OxiResult;

    #[test]
    fn test_batch_processor_creation() {
        let qc = QualityControl::with_preset(QcPreset::Basic);
        let processor = BatchProcessor::new(qc);
        assert!(processor.include_detailed_reports);
    }

    #[test]
    fn test_batch_results_summary() {
        let results = BatchResults {
            total_files: 10,
            passed: 8,
            failed: 2,
            errors: 0,
            reports: Vec::new(),
            total_duration: 45.5,
        };

        let summary = results.summary();
        assert!(summary.contains("Total Files: 10"));
        assert!(summary.contains("Passed: 8"));
    }

    // ── Early-termination tests ───────────────────────────────────────────────

    /// A test rule that unconditionally returns a result of the given severity.
    struct FixedRule {
        name: String,
        severity: Severity,
    }

    impl QcRule for FixedRule {
        fn name(&self) -> &str {
            &self.name
        }

        fn category(&self) -> RuleCategory {
            RuleCategory::Video
        }

        fn check(&self, _context: &QcContext) -> OxiResult<Vec<CheckResult>> {
            let result = if matches!(self.severity, Severity::Info) {
                CheckResult::pass(self.name())
            } else {
                CheckResult::fail(
                    self.name(),
                    self.severity,
                    format!("Rule {} triggered", self.name()),
                )
            };
            Ok(vec![result])
        }
    }

    /// With `abort_on_critical=true` and the first rule returning Error,
    /// only 1 rule should be executed.
    #[test]
    fn test_early_term_stops_on_error() {
        let mut qc = QualityControl::new();
        qc.add_rule(Box::new(FixedRule {
            name: "rule_error".to_string(),
            severity: Severity::Error,
        }));
        qc.add_rule(Box::new(FixedRule {
            name: "rule_info_1".to_string(),
            severity: Severity::Info,
        }));
        qc.add_rule(Box::new(FixedRule {
            name: "rule_info_2".to_string(),
            severity: Severity::Info,
        }));

        // We need a real file for probe_file. Use a temp file path that exists.
        let tmp = std::env::temp_dir().join("oximedia-qc-early-term-test.bin");
        std::fs::write(&tmp, b"fLaC\x00\x00\x00\x00").expect("write temp file");
        let path = tmp.to_string_lossy().to_string();

        let (_, rules_run) = qc
            .validate_with_abort(&path, true)
            .expect("validate_with_abort should succeed");

        // First rule returns Error → loop should break after 1 rule.
        assert_eq!(
            rules_run, 1,
            "abort_on_critical=true: only 1 rule should run, got {rules_run}"
        );
    }

    /// With `abort_on_critical=false`, all 3 rules must be run even when the
    /// first returns Error.
    #[test]
    fn test_normal_mode_continues_past_error() {
        let mut qc = QualityControl::new();
        qc.add_rule(Box::new(FixedRule {
            name: "rule_error".to_string(),
            severity: Severity::Error,
        }));
        qc.add_rule(Box::new(FixedRule {
            name: "rule_info_1".to_string(),
            severity: Severity::Info,
        }));
        qc.add_rule(Box::new(FixedRule {
            name: "rule_info_2".to_string(),
            severity: Severity::Info,
        }));

        let tmp = std::env::temp_dir().join("oximedia-qc-normal-mode-test.bin");
        std::fs::write(&tmp, b"fLaC\x00\x00\x00\x00").expect("write temp file");
        let path = tmp.to_string_lossy().to_string();

        let (_, rules_run) = qc
            .validate_with_abort(&path, false)
            .expect("validate_with_abort should succeed");

        // abort_on_critical=false: all 3 rules run.
        assert_eq!(
            rules_run, 3,
            "abort_on_critical=false: all 3 rules should run, got {rules_run}"
        );
    }
}
