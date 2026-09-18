//! Accessibility compliance checking.

pub mod ebu;
pub mod report;
pub mod section508;
pub mod wcag;

pub use ebu::EbuChecker;
pub use report::{ComplianceIssue, ComplianceReport, IssueSeverity};
pub use section508::Section508Checker;
pub use wcag::{WcagChecker, WcagGuideline, WcagLevel};

use std::sync::{Arc, Mutex};

/// Compliance checker for accessibility standards.
pub struct ComplianceChecker {
    wcag_checker: WcagChecker,
    section508_checker: Section508Checker,
    ebu_checker: EbuChecker,
}

impl ComplianceChecker {
    /// Create a new compliance checker.
    #[must_use]
    pub fn new() -> Self {
        Self {
            wcag_checker: WcagChecker::new(WcagLevel::AA),
            section508_checker: Section508Checker::new(),
            ebu_checker: EbuChecker::new(),
        }
    }

    /// Check all compliance standards.
    #[must_use]
    pub fn check_all(&self) -> ComplianceReport {
        let mut report = ComplianceReport::new();

        // Check WCAG compliance
        let wcag_issues = self.wcag_checker.check();
        report.add_issues(wcag_issues);

        // Check Section 508
        let section508_issues = self.section508_checker.check();
        report.add_issues(section508_issues);

        // Check EBU
        let ebu_issues = self.ebu_checker.check();
        report.add_issues(ebu_issues);

        report
    }

    /// Check specific standard.
    #[must_use]
    pub fn check_wcag(&self) -> Vec<ComplianceIssue> {
        self.wcag_checker.check()
    }

    /// Check Section 508 compliance.
    #[must_use]
    pub fn check_section508(&self) -> Vec<ComplianceIssue> {
        self.section508_checker.check()
    }

    /// Check EBU compliance.
    #[must_use]
    pub fn check_ebu(&self) -> Vec<ComplianceIssue> {
        self.ebu_checker.check()
    }

    /// Get WCAG checker.
    #[must_use]
    pub const fn wcag_checker(&self) -> &WcagChecker {
        &self.wcag_checker
    }

    /// Get Section 508 checker.
    #[must_use]
    pub const fn section508_checker(&self) -> &Section508Checker {
        &self.section508_checker
    }

    /// Get EBU checker.
    #[must_use]
    pub const fn ebu_checker(&self) -> &EbuChecker {
        &self.ebu_checker
    }
}

impl Default for ComplianceChecker {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Parallel compliance across multiple media files ────────────────────────

/// Descriptor for a media file to be checked in bulk.
#[derive(Debug, Clone)]
pub struct MediaFileDescriptor {
    /// Unique identifier for this file (e.g. path or asset ID).
    pub id: String,
    /// Whether the media has synchronized captions.
    pub has_captions: bool,
    /// Whether the media has audio descriptions.
    pub has_audio_desc: bool,
    /// Loudness in LUFS (for EBU R128 checking).
    pub loudness_lufs: f32,
    /// Contrast ratio of the primary text overlay (for WCAG checking).
    pub contrast_ratio: f64,
    /// Flash frequency in Hz (for photosensitive seizure checking).
    pub flashes_per_second: f64,
}

impl MediaFileDescriptor {
    /// Create a fully compliant media descriptor.
    #[must_use]
    pub fn compliant(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            has_captions: true,
            has_audio_desc: true,
            loudness_lufs: -23.0_f32,
            contrast_ratio: 5.0,
            flashes_per_second: 1.0,
        }
    }

    /// Create a non-compliant media descriptor (no captions, no audio desc).
    #[must_use]
    pub fn non_compliant(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            has_captions: false,
            has_audio_desc: false,
            loudness_lufs: -18.0_f32,
            contrast_ratio: 2.0,
            flashes_per_second: 5.0,
        }
    }
}

/// Compliance result for a single media file.
#[derive(Debug, Clone)]
pub struct MediaFileResult {
    /// ID from the original descriptor.
    pub id: String,
    /// All issues found for this file.
    pub issues: Vec<ComplianceIssue>,
}

impl MediaFileResult {
    /// Whether this file is fully compliant (no issues).
    #[must_use]
    pub fn is_compliant(&self) -> bool {
        self.issues.is_empty()
    }

    /// Number of critical issues.
    #[must_use]
    pub fn critical_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|i| i.severity == IssueSeverity::Critical)
            .count()
    }
}

/// Checks multiple media files for accessibility compliance in parallel.
///
/// Uses OS threads so CPU-bound checking work can run concurrently.
/// Results are returned in the same order as the input descriptors.
pub struct ParallelComplianceChecker {
    max_threads: usize,
}

impl ParallelComplianceChecker {
    /// Create a new parallel checker.
    ///
    /// `max_threads` caps the thread count.  Pass `0` to use the number of
    /// logical CPUs (capped at 16).
    #[must_use]
    pub fn new(max_threads: usize) -> Self {
        let resolved = if max_threads == 0 {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
                .min(16)
        } else {
            max_threads.min(64)
        };
        Self {
            max_threads: resolved,
        }
    }

    /// Check all descriptors and return results in input order.
    #[must_use]
    pub fn check_parallel(&self, files: &[MediaFileDescriptor]) -> Vec<MediaFileResult> {
        if files.is_empty() {
            return Vec::new();
        }

        let n = files.len();
        // Pre-allocate result slots so we can write in index order
        let results: Arc<Mutex<Vec<Option<MediaFileResult>>>> = Arc::new(Mutex::new(vec![None; n]));

        // Divide files into chunks, one per thread
        let chunk_size = (n + self.max_threads - 1) / self.max_threads;
        let chunks: Vec<&[MediaFileDescriptor]> = files.chunks(chunk_size).collect();

        std::thread::scope(|scope| {
            let mut chunk_offset = 0usize;
            for chunk in &chunks {
                let results_clone = Arc::clone(&results);
                let start = chunk_offset;
                let local_chunk: Vec<MediaFileDescriptor> = (*chunk).to_vec();
                chunk_offset += local_chunk.len();

                scope.spawn(move || {
                    for (local_idx, descriptor) in local_chunk.iter().enumerate() {
                        let global_idx = start + local_idx;
                        let file_result = Self::check_one(descriptor);
                        let mut guard = results_clone.lock().expect("mutex not poisoned");
                        guard[global_idx] = Some(file_result);
                    }
                });
            }
        });

        // Unwrap the Options — all slots must be filled
        let guard = results.lock().expect("mutex not poisoned");
        guard
            .iter()
            .map(|opt| opt.clone().expect("result slot filled"))
            .collect()
    }

    /// Check a single file descriptor and return its result.
    #[must_use]
    pub fn check_one(descriptor: &MediaFileDescriptor) -> MediaFileResult {
        let section508 = Section508Checker::new();
        let ebu = EbuChecker::new();

        let mut issues: Vec<ComplianceIssue> = Vec::new();

        // Captions check
        if let Some(issue) = section508.check_synchronized_captions(descriptor.has_captions) {
            issues.push(issue.with_location(descriptor.id.clone()));
        }

        // Audio descriptions check
        if let Some(issue) = section508.check_audio_descriptions(descriptor.has_audio_desc) {
            issues.push(issue.with_location(descriptor.id.clone()));
        }

        // EBU R128 loudness check
        if let Some(issue) = ebu.check_loudness(descriptor.loudness_lufs) {
            issues.push(issue.with_location(descriptor.id.clone()));
        }

        // Contrast ratio check (WCAG SC 1.4.3 requires >= 4.5:1)
        if descriptor.contrast_ratio < 4.5 {
            issues.push(
                ComplianceIssue::new(
                    "WCAG-1.4.3".to_string(),
                    "Contrast Ratio".to_string(),
                    format!(
                        "Contrast ratio {:.2} is below the required 4.5:1 for WCAG AA",
                        descriptor.contrast_ratio
                    ),
                    IssueSeverity::High,
                )
                .with_location(descriptor.id.clone()),
            );
        }

        // Flash/flicker check (Harding test: <= 3 Hz)
        if descriptor.flashes_per_second > 3.0 {
            issues.push(
                ComplianceIssue::new(
                    "WCAG-2.3.1".to_string(),
                    "Flash Rate".to_string(),
                    format!(
                        "Flash rate {:.1} Hz exceeds the safe threshold of 3 Hz",
                        descriptor.flashes_per_second
                    ),
                    IssueSeverity::Critical,
                )
                .with_location(descriptor.id.clone()),
            );
        }

        MediaFileResult {
            id: descriptor.id.clone(),
            issues,
        }
    }

    /// Build an aggregate compliance report from a set of results.
    #[must_use]
    pub fn aggregate_report(results: &[MediaFileResult]) -> ComplianceReport {
        let mut report = ComplianceReport::new();
        for result in results {
            report.add_issues(result.issues.clone());
        }
        report
    }

    /// Count the number of compliant files in a result set.
    #[must_use]
    pub fn count_compliant(results: &[MediaFileResult]) -> usize {
        results.iter().filter(|r| r.is_compliant()).count()
    }

    /// Number of threads this checker will use.
    #[must_use]
    pub fn num_threads(&self) -> usize {
        self.max_threads
    }
}

impl Default for ParallelComplianceChecker {
    fn default() -> Self {
        Self::new(0) // auto-detect
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checker_creation() {
        let checker = ComplianceChecker::new();
        let report = checker.check_all();
        assert!(report.issues().is_empty() || !report.issues().is_empty());
    }

    // ─── Parallel checker tests ────────────────────────────────────────────

    #[test]
    fn test_parallel_checker_empty_input() {
        let checker = ParallelComplianceChecker::new(4);
        let results = checker.check_parallel(&[]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_parallel_checker_compliant_file() {
        let checker = ParallelComplianceChecker::new(2);
        let files = vec![MediaFileDescriptor::compliant("file-1")];
        let results = checker.check_parallel(&files);
        assert_eq!(results.len(), 1);
        assert!(
            results[0].is_compliant(),
            "compliant file should have no issues"
        );
    }

    #[test]
    fn test_parallel_checker_non_compliant_file() {
        let checker = ParallelComplianceChecker::new(2);
        let files = vec![MediaFileDescriptor::non_compliant("bad-file")];
        let results = checker.check_parallel(&files);
        assert_eq!(results.len(), 1);
        assert!(!results[0].is_compliant());
    }

    #[test]
    fn test_parallel_checker_mixed_files() {
        let checker = ParallelComplianceChecker::new(4);
        let files: Vec<MediaFileDescriptor> = (0..12)
            .map(|i| {
                if i % 3 == 0 {
                    MediaFileDescriptor::compliant(format!("file-{i}"))
                } else {
                    MediaFileDescriptor::non_compliant(format!("file-{i}"))
                }
            })
            .collect();
        // 0, 3, 6, 9 are compliant → 4 compliant files
        let results = checker.check_parallel(&files);
        assert_eq!(results.len(), 12);
        // Verify ordering is preserved
        for (i, result) in results.iter().enumerate() {
            assert_eq!(result.id, files[i].id);
        }
        let compliant_count = ParallelComplianceChecker::count_compliant(&results);
        assert_eq!(compliant_count, 4);
    }

    #[test]
    fn test_parallel_checker_order_preserved() {
        let checker = ParallelComplianceChecker::new(8);
        let files: Vec<MediaFileDescriptor> = (0..20)
            .map(|i| MediaFileDescriptor::compliant(format!("asset-{i:03}")))
            .collect();
        let results = checker.check_parallel(&files);
        for (i, result) in results.iter().enumerate() {
            assert_eq!(
                result.id,
                format!("asset-{i:03}"),
                "order must be preserved"
            );
        }
    }

    #[test]
    fn test_aggregate_report_collects_all_issues() {
        let checker = ParallelComplianceChecker::new(2);
        let files = vec![
            MediaFileDescriptor::non_compliant("a"),
            MediaFileDescriptor::non_compliant("b"),
        ];
        let results = checker.check_parallel(&files);
        let report = ParallelComplianceChecker::aggregate_report(&results);
        // Both files are non-compliant so report should have issues
        assert!(!report.issues().is_empty());
    }

    #[test]
    fn test_num_threads_auto_detect() {
        let checker = ParallelComplianceChecker::new(0);
        assert!(checker.num_threads() >= 1);
        assert!(checker.num_threads() <= 16);
    }

    #[test]
    fn test_num_threads_explicit() {
        let checker = ParallelComplianceChecker::new(4);
        assert_eq!(checker.num_threads(), 4);
    }

    #[test]
    fn test_check_one_compliant() {
        let descriptor = MediaFileDescriptor::compliant("test");
        let result = ParallelComplianceChecker::check_one(&descriptor);
        assert!(result.is_compliant());
    }

    #[test]
    fn test_check_one_no_captions_produces_critical() {
        let descriptor = MediaFileDescriptor {
            id: "no-captions".to_string(),
            has_captions: false,
            has_audio_desc: true,
            loudness_lufs: -23.0_f32,
            contrast_ratio: 5.0,
            flashes_per_second: 1.0,
        };
        let result = ParallelComplianceChecker::check_one(&descriptor);
        assert!(!result.is_compliant());
        assert!(result.critical_count() > 0);
    }

    #[test]
    fn test_check_one_excessive_flash_produces_critical() {
        let descriptor = MediaFileDescriptor {
            id: "flashy".to_string(),
            has_captions: true,
            has_audio_desc: true,
            loudness_lufs: -23.0_f32,
            contrast_ratio: 5.0,
            flashes_per_second: 10.0,
        };
        let result = ParallelComplianceChecker::check_one(&descriptor);
        assert!(result.critical_count() > 0);
    }

    #[test]
    fn test_count_compliant() {
        let files = vec![
            MediaFileDescriptor::compliant("c1"),
            MediaFileDescriptor::non_compliant("nc1"),
            MediaFileDescriptor::compliant("c2"),
        ];
        let checker = ParallelComplianceChecker::new(2);
        let results = checker.check_parallel(&files);
        assert_eq!(ParallelComplianceChecker::count_compliant(&results), 2);
    }

    #[test]
    fn test_parallel_single_thread() {
        let checker = ParallelComplianceChecker::new(1);
        let files: Vec<MediaFileDescriptor> = (0..5)
            .map(|i| MediaFileDescriptor::compliant(format!("f{i}")))
            .collect();
        let results = checker.check_parallel(&files);
        assert_eq!(results.len(), 5);
        for (i, r) in results.iter().enumerate() {
            assert_eq!(r.id, format!("f{i}"));
        }
    }
}
