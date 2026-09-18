//! IMF package-level validation helpers.
//!
//! Provides a lightweight validator that checks an IMF package structure and
//! collects issues with their severity, then reports which ones are blocking.
//! Also provides [`verify_hashes_parallel`] for concurrent hash verification
//! of multi-asset packages using Rayon.

#![allow(dead_code)]

use rayon::prelude::*;

/// How serious a validation finding is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ValidationSeverity {
    /// Informational note – does not prevent delivery.
    Info,
    /// Advisory warning – delivery is possible but not recommended.
    Warning,
    /// Error that blocks delivery or conformance.
    Error,
}

impl ValidationSeverity {
    /// Returns `true` when this severity prevents package delivery.
    #[must_use]
    pub fn is_blocking(self) -> bool {
        self == Self::Error
    }

    /// Short human-readable tag.
    #[must_use]
    pub fn tag(self) -> &'static str {
        match self {
            Self::Info => "INFO",
            Self::Warning => "WARN",
            Self::Error => "ERROR",
        }
    }
}

// ---------------------------------------------------------------------------

/// A single issue found during package validation.
#[derive(Debug, Clone)]
pub struct PackageIssue {
    /// Severity of this issue.
    pub severity: ValidationSeverity,
    /// Machine-readable code (e.g. `"PKL_HASH_MISSING"`).
    pub code: String,
    /// Human-readable description of the problem.
    pub detail: String,
}

impl PackageIssue {
    /// Create a new issue.
    #[must_use]
    pub fn new(
        severity: ValidationSeverity,
        code: impl Into<String>,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            severity,
            code: code.into(),
            detail: detail.into(),
        }
    }

    /// Formatted description including severity tag and code.
    #[must_use]
    pub fn description(&self) -> String {
        format!("[{}] {}: {}", self.severity.tag(), self.code, self.detail)
    }

    /// Returns `true` when this issue is blocking.
    #[must_use]
    pub fn is_blocking(&self) -> bool {
        self.severity.is_blocking()
    }
}

// ---------------------------------------------------------------------------

/// Accumulates issues found while checking a package.
#[derive(Debug, Default)]
pub struct PackageValidator {
    issues: Vec<PackageIssue>,
}

impl PackageValidator {
    /// Create a fresh validator.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a pre-built issue.
    pub fn record(&mut self, issue: PackageIssue) {
        self.issues.push(issue);
    }

    /// Convenience: record a blocking error.
    pub fn check(&mut self, condition: bool, code: impl Into<String>, detail: impl Into<String>) {
        if !condition {
            self.issues
                .push(PackageIssue::new(ValidationSeverity::Error, code, detail));
        }
    }

    /// Convenience: record a non-blocking warning.
    pub fn warn(&mut self, condition: bool, code: impl Into<String>, detail: impl Into<String>) {
        if !condition {
            self.issues
                .push(PackageIssue::new(ValidationSeverity::Warning, code, detail));
        }
    }

    /// Consume the validator and return the final report.
    #[must_use]
    pub fn finish(self) -> PackageValidationReport {
        PackageValidationReport {
            issues: self.issues,
        }
    }
}

// ---------------------------------------------------------------------------

/// The result of running [`PackageValidator`] over an IMF package.
#[derive(Debug, Clone, Default)]
pub struct PackageValidationReport {
    /// All issues found (may be empty).
    pub issues: Vec<PackageIssue>,
}

impl PackageValidationReport {
    /// All issues whose severity is [`ValidationSeverity::Error`].
    #[must_use]
    pub fn blocking_issues(&self) -> Vec<&PackageIssue> {
        self.issues.iter().filter(|i| i.is_blocking()).collect()
    }

    /// Returns `true` if there are no blocking issues.
    #[must_use]
    pub fn is_ok(&self) -> bool {
        self.blocking_issues().is_empty()
    }

    /// Total issue count regardless of severity.
    #[must_use]
    pub fn total(&self) -> usize {
        self.issues.len()
    }

    /// Count issues of a specific severity.
    #[must_use]
    pub fn count_severity(&self, sev: ValidationSeverity) -> usize {
        self.issues.iter().filter(|i| i.severity == sev).count()
    }
}

// ---------------------------------------------------------------------------
// Parallel hash verification
// ---------------------------------------------------------------------------

/// A descriptor for an asset whose hash is to be verified.
///
/// Used by [`verify_hashes_parallel`].
#[derive(Debug, Clone)]
pub struct HashableAsset {
    /// Identifier used in error messages.
    pub id: String,
    /// Raw data bytes to hash (in-memory representation).
    ///
    /// In a real deployment this would be replaced by a path to the MXF file,
    /// but this in-memory form keeps the validator free of file-system I/O.
    pub data: Vec<u8>,
    /// Expected hex-encoded hash.
    pub expected_hash: String,
    /// Hash algorithm to use.
    pub algorithm: crate::essence_hash::HashAlgo,
}

impl HashableAsset {
    /// Create a new [`HashableAsset`].
    pub fn new(
        id: impl Into<String>,
        data: Vec<u8>,
        expected_hash: impl Into<String>,
        algorithm: crate::essence_hash::HashAlgo,
    ) -> Self {
        Self {
            id: id.into(),
            data,
            expected_hash: expected_hash.into(),
            algorithm,
        }
    }
}

/// Verify the hashes of a slice of assets in parallel.
///
/// Returns one `Result<bool, String>` per asset in the same order as the
/// input slice.  `Ok(true)` means the hash matches; `Ok(false)` means the
/// computed hash does not match the expected value; `Err(msg)` signals an
/// internal computation failure.
///
/// Parallelism is provided by Rayon's work-stealing thread pool, so large
/// multi-asset packages benefit from all available CPU cores.
///
/// # Example
/// ```ignore
/// use oximedia_imf::package_validator::{HashableAsset, verify_hashes_parallel};
/// use oximedia_imf::essence_hash::HashAlgo;
///
/// let assets = vec![
///     HashableAsset::new("video-001", b"...".to_vec(), "expected_hex", HashAlgo::Sha256),
/// ];
/// let results = verify_hashes_parallel(&assets);
/// assert_eq!(results.len(), 1);
/// ```
pub fn verify_hashes_parallel(assets: &[HashableAsset]) -> Vec<Result<bool, String>> {
    assets
        .par_iter()
        .map(|asset| {
            let computed = crate::essence_hash::compute_hash_hex(&asset.data, asset.algorithm);
            let matches = computed.eq_ignore_ascii_case(&asset.expected_hash);
            Ok(matches)
        })
        .collect()
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_severity_error_is_blocking() {
        assert!(ValidationSeverity::Error.is_blocking());
    }

    #[test]
    fn test_severity_warning_not_blocking() {
        assert!(!ValidationSeverity::Warning.is_blocking());
    }

    #[test]
    fn test_severity_info_not_blocking() {
        assert!(!ValidationSeverity::Info.is_blocking());
    }

    #[test]
    fn test_severity_tags_distinct() {
        let tags = [
            ValidationSeverity::Info.tag(),
            ValidationSeverity::Warning.tag(),
            ValidationSeverity::Error.tag(),
        ];
        let set: std::collections::HashSet<_> = tags.iter().collect();
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn test_issue_description_contains_code() {
        let issue = PackageIssue::new(
            ValidationSeverity::Error,
            "PKL_MISSING",
            "Packing list not found",
        );
        assert!(issue.description().contains("PKL_MISSING"));
    }

    #[test]
    fn test_issue_is_blocking_for_error() {
        let issue = PackageIssue::new(ValidationSeverity::Error, "X", "detail");
        assert!(issue.is_blocking());
    }

    #[test]
    fn test_issue_not_blocking_for_warning() {
        let issue = PackageIssue::new(ValidationSeverity::Warning, "X", "detail");
        assert!(!issue.is_blocking());
    }

    #[test]
    fn test_validator_check_records_error_on_false() {
        let mut v = PackageValidator::new();
        v.check(false, "CODE", "detail");
        let r = v.finish();
        assert_eq!(r.total(), 1);
        assert!(!r.is_ok());
    }

    #[test]
    fn test_validator_check_no_issue_on_true() {
        let mut v = PackageValidator::new();
        v.check(true, "CODE", "detail");
        let r = v.finish();
        assert_eq!(r.total(), 0);
    }

    #[test]
    fn test_validator_warn_records_warning() {
        let mut v = PackageValidator::new();
        v.warn(false, "WARN_CODE", "advisory");
        let r = v.finish();
        assert_eq!(r.count_severity(ValidationSeverity::Warning), 1);
        assert!(r.is_ok()); // warnings are not blocking
    }

    #[test]
    fn test_report_blocking_issues_only_errors() {
        let mut v = PackageValidator::new();
        v.warn(false, "W1", "warn");
        v.check(false, "E1", "error");
        let r = v.finish();
        assert_eq!(r.blocking_issues().len(), 1);
    }

    #[test]
    fn test_report_count_severity() {
        let mut v = PackageValidator::new();
        v.check(false, "E1", "e");
        v.check(false, "E2", "e");
        v.warn(false, "W1", "w");
        let r = v.finish();
        assert_eq!(r.count_severity(ValidationSeverity::Error), 2);
        assert_eq!(r.count_severity(ValidationSeverity::Warning), 1);
    }

    #[test]
    fn test_empty_report_is_ok() {
        let v = PackageValidator::new();
        let r = v.finish();
        assert!(r.is_ok());
        assert_eq!(r.total(), 0);
    }

    // ── verify_hashes_parallel tests ─────────────────────────────────────

    use crate::essence_hash::{compute_hash_hex, HashAlgo};

    #[test]
    fn test_verify_hashes_parallel_empty() {
        let results = verify_hashes_parallel(&[]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_verify_hashes_parallel_single_match() {
        let data = b"IMF essence data";
        let expected = compute_hash_hex(data, HashAlgo::Sha256);
        let assets = vec![HashableAsset::new(
            "asset-001",
            data.to_vec(),
            expected,
            HashAlgo::Sha256,
        )];
        let results = verify_hashes_parallel(&assets);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].as_ref().copied().unwrap_or(false), true);
    }

    #[test]
    fn test_verify_hashes_parallel_single_mismatch() {
        let data = b"real data";
        let assets = vec![HashableAsset::new(
            "asset-001",
            data.to_vec(),
            "aaaa".repeat(16), // wrong SHA-256 hash (64 chars)
            HashAlgo::Sha256,
        )];
        let results = verify_hashes_parallel(&assets);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].as_ref().copied().unwrap_or(true), false);
    }

    #[test]
    fn test_verify_hashes_parallel_multi_asset() {
        let payloads: Vec<&[u8]> = vec![b"video-mxf-data", b"audio-mxf-data", b"subtitle-xml"];
        let assets: Vec<HashableAsset> = payloads
            .iter()
            .enumerate()
            .map(|(i, data)| {
                let expected = compute_hash_hex(data, HashAlgo::Sha256);
                HashableAsset::new(
                    format!("asset-{i:03}"),
                    data.to_vec(),
                    expected,
                    HashAlgo::Sha256,
                )
            })
            .collect();

        let results = verify_hashes_parallel(&assets);
        assert_eq!(results.len(), 3);
        for r in &results {
            assert_eq!(r.as_ref().copied().unwrap_or(false), true);
        }
    }

    #[test]
    fn test_verify_hashes_parallel_mixed_results() {
        let data_good = b"correct";
        let good_hash = compute_hash_hex(data_good, HashAlgo::Sha256);
        let assets = vec![
            HashableAsset::new("good", data_good.to_vec(), good_hash, HashAlgo::Sha256),
            HashableAsset::new("bad", data_good.to_vec(), "0".repeat(64), HashAlgo::Sha256),
        ];
        let results = verify_hashes_parallel(&assets);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].as_ref().copied().unwrap_or(false), true);
        assert_eq!(results[1].as_ref().copied().unwrap_or(true), false);
    }

    #[test]
    fn test_verify_hashes_parallel_sha512() {
        let data = b"sha512 essence";
        let expected = compute_hash_hex(data, HashAlgo::Sha512);
        let assets = vec![HashableAsset::new(
            "asset-512",
            data.to_vec(),
            expected,
            HashAlgo::Sha512,
        )];
        let results = verify_hashes_parallel(&assets);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].as_ref().copied().unwrap_or(false), true);
    }

    #[test]
    fn test_verify_hashes_parallel_sha1() {
        let data = b"legacy sha1";
        let expected = compute_hash_hex(data, HashAlgo::Sha1);
        let assets = vec![HashableAsset::new(
            "asset-sha1",
            data.to_vec(),
            expected,
            HashAlgo::Sha1,
        )];
        let results = verify_hashes_parallel(&assets);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].as_ref().copied().unwrap_or(false), true);
    }
}
