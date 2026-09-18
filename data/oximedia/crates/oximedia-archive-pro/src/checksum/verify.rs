//! Checksum verification for integrity checking

use super::{ChecksumAlgorithm, ChecksumGenerator, FileChecksum};
use crate::{Error, Result};
use blake3::Hasher as Blake3Hasher;
use md5::{Digest, Md5};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Sha256, Sha512};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use xxhash_rust::xxh3::Xxh3;

/// Result of a checksum verification
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationResult {
    /// Checksum matched
    Success,
    /// Checksum mismatch
    Failed {
        /// Expected checksum
        expected: String,
        /// Actual checksum
        actual: String,
    },
    /// File not found
    Missing,
    /// File size mismatch
    SizeMismatch {
        /// Expected size
        expected: u64,
        /// Actual size
        actual: u64,
    },
}

impl VerificationResult {
    /// Returns `true` if the verification succeeded
    #[must_use]
    pub const fn is_success(&self) -> bool {
        matches!(self, Self::Success)
    }

    /// Returns `true` if the verification failed
    #[must_use]
    pub const fn is_failed(&self) -> bool {
        !self.is_success()
    }
}

/// Verification report for a file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileVerificationReport {
    /// File path
    pub path: PathBuf,
    /// Verification results by algorithm
    pub results: HashMap<ChecksumAlgorithm, VerificationResult>,
    /// Timestamp of verification
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl FileVerificationReport {
    /// Returns `true` if all verifications succeeded
    #[must_use]
    pub fn is_success(&self) -> bool {
        self.results.values().all(VerificationResult::is_success)
    }

    /// Returns the number of failed verifications
    #[must_use]
    pub fn failed_count(&self) -> usize {
        self.results.values().filter(|r| r.is_failed()).count()
    }
}

/// Verification report for multiple files
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationReport {
    /// Per-file reports
    pub files: Vec<FileVerificationReport>,
    /// Timestamp of verification
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl VerificationReport {
    /// Returns `true` if all files passed verification
    #[must_use]
    pub fn is_success(&self) -> bool {
        self.files.iter().all(FileVerificationReport::is_success)
    }

    /// Returns the number of files that passed verification
    #[must_use]
    pub fn success_count(&self) -> usize {
        self.files.iter().filter(|f| f.is_success()).count()
    }

    /// Returns the number of files that failed verification
    #[must_use]
    pub fn failed_count(&self) -> usize {
        self.files.len() - self.success_count()
    }

    /// Returns a summary string
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "Verification: {} succeeded, {} failed out of {} total",
            self.success_count(),
            self.failed_count(),
            self.files.len()
        )
    }
}

/// Per-algorithm verification result produced by concurrent verification
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AlgoVerificationResult {
    /// The algorithm that was verified
    pub algo: ChecksumAlgorithm,
    /// Whether this algorithm's checksum matched
    pub passed: bool,
    /// The computed (actual) hex digest
    pub computed: String,
    /// The expected hex digest that was provided
    pub expected: String,
}

/// Result returned by `ChecksumVerifier::verify_file_concurrent`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcurrentVerificationReport {
    /// File path that was verified
    pub path: PathBuf,
    /// Per-algorithm breakdown
    pub algo_results: Vec<AlgoVerificationResult>,
    /// `true` iff every algorithm passed
    pub all_passed: bool,
    /// Timestamp of verification
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl ConcurrentVerificationReport {
    /// Convert into a `FileVerificationReport` for backward-compatible use
    #[must_use]
    pub fn into_file_report(self) -> FileVerificationReport {
        let results = self
            .algo_results
            .iter()
            .map(|ar| {
                let result = if ar.passed {
                    VerificationResult::Success
                } else {
                    VerificationResult::Failed {
                        expected: ar.expected.clone(),
                        actual: ar.computed.clone(),
                    }
                };
                (ar.algo, result)
            })
            .collect();
        FileVerificationReport {
            path: self.path,
            results,
            timestamp: self.timestamp,
        }
    }
}

/// Checksum verifier.
///
/// The verifier is a zero-size unit struct; per-file verification always operates
/// on the algorithms present in the supplied [`FileChecksum`] rather than on a
/// static algorithm list.
pub struct ChecksumVerifier;

impl Default for ChecksumVerifier {
    fn default() -> Self {
        Self::new()
    }
}

impl ChecksumVerifier {
    /// Create a new checksum verifier.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Create a verifier; the `algorithms` parameter is accepted for API compatibility
    /// but verification always uses the algorithms present in the [`FileChecksum`] argument.
    #[must_use]
    pub fn with_algorithms(_algorithms: Vec<ChecksumAlgorithm>) -> Self {
        Self
    }

    /// Compute a single algorithm digest over the given byte slice.
    ///
    /// Returns the hex-encoded digest string.
    fn compute_digest(algo: ChecksumAlgorithm, data: &[u8]) -> String {
        match algo {
            ChecksumAlgorithm::Md5 => {
                let mut h = Md5::new();
                h.update(data);
                hex::encode(h.finalize())
            }
            ChecksumAlgorithm::Sha256 => {
                let mut h = Sha256::new();
                h.update(data);
                hex::encode(h.finalize())
            }
            ChecksumAlgorithm::Sha512 => {
                let mut h = Sha512::new();
                h.update(data);
                hex::encode(h.finalize())
            }
            ChecksumAlgorithm::XxHash64 => {
                let mut h = Xxh3::new();
                h.update(data);
                format!("{:x}", h.digest())
            }
            ChecksumAlgorithm::Blake3 => {
                let mut h = Blake3Hasher::new();
                h.update(data);
                format!("{}", h.finalize().to_hex())
            }
        }
    }

    /// Verify a file using concurrent per-algorithm digest computation.
    ///
    /// The file is read **once** into a `Vec<u8>` buffer; rayon then dispatches
    /// each algorithm onto a separate thread pool task so all digests are computed
    /// in parallel rather than sequentially.  The returned
    /// [`ConcurrentVerificationReport`] includes a per-algorithm breakdown and a
    /// combined `all_passed` flag.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read.
    pub fn verify_file_concurrent(
        &self,
        expected: &FileChecksum,
    ) -> Result<ConcurrentVerificationReport> {
        let path = &expected.path;
        let timestamp = chrono::Utc::now();

        // Fast path: file missing
        if !path.exists() {
            let algo_results = expected
                .checksums
                .keys()
                .map(|&algo| AlgoVerificationResult {
                    algo,
                    passed: false,
                    computed: String::new(),
                    expected: expected.checksums.get(&algo).cloned().unwrap_or_default(),
                })
                .collect::<Vec<_>>();
            let all_passed = algo_results.iter().all(|r| r.passed);
            return Ok(ConcurrentVerificationReport {
                path: path.clone(),
                algo_results,
                all_passed,
                timestamp,
            });
        }

        // Fast path: size mismatch means we skip digest computation entirely
        let metadata = std::fs::metadata(path)?;
        let actual_size = metadata.len();
        if actual_size != expected.size {
            let algo_results = expected
                .checksums
                .keys()
                .map(|&algo| AlgoVerificationResult {
                    algo,
                    passed: false,
                    computed: format!(
                        "<size mismatch: got {actual_size}, expected {}>",
                        expected.size
                    ),
                    expected: expected.checksums.get(&algo).cloned().unwrap_or_default(),
                })
                .collect::<Vec<_>>();
            let all_passed = false;
            return Ok(ConcurrentVerificationReport {
                path: path.clone(),
                algo_results,
                all_passed,
                timestamp,
            });
        }

        // Read the entire file into memory once.
        let bytes: Vec<u8> = std::fs::read(path)?;

        // Collect the (algo, expected_hash) pairs so rayon can own them.
        let pairs: Vec<(ChecksumAlgorithm, String)> = expected
            .checksums
            .iter()
            .map(|(&algo, hash)| (algo, hash.clone()))
            .collect();

        // Compute all digests in parallel; &bytes is an immutable shared ref,
        // so no synchronisation is required between tasks.
        let algo_results: Vec<AlgoVerificationResult> = pairs
            .par_iter()
            .map(|(algo, expected_hash)| {
                let computed = Self::compute_digest(*algo, &bytes);
                let passed = computed == *expected_hash;
                AlgoVerificationResult {
                    algo: *algo,
                    passed,
                    computed,
                    expected: expected_hash.clone(),
                }
            })
            .collect();

        let all_passed = algo_results.iter().all(|r| r.passed);

        Ok(ConcurrentVerificationReport {
            path: path.clone(),
            algo_results,
            all_passed,
            timestamp,
        })
    }

    /// Verify a single file against expected checksums.
    ///
    /// Delegates to [`Self::verify_file_concurrent`] under the hood and converts
    /// the result back to the legacy [`FileVerificationReport`] shape so that all
    /// existing call-sites continue to work without modification.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read
    pub fn verify_file(&self, expected: &FileChecksum) -> Result<FileVerificationReport> {
        let concurrent = self.verify_file_concurrent(expected)?;
        Ok(concurrent.into_file_report())
    }

    /// Verify multiple files
    ///
    /// # Errors
    ///
    /// Returns an error if any file cannot be read
    pub fn verify_batch(&self, expected: &[FileChecksum]) -> Result<VerificationReport> {
        let files = expected
            .iter()
            .map(|e| self.verify_file(e))
            .collect::<Result<Vec<_>>>()?;

        Ok(VerificationReport {
            files,
            timestamp: chrono::Utc::now(),
        })
    }

    /// Verify a file against a single expected checksum
    ///
    /// # Errors
    ///
    /// Returns an error if verification fails or file cannot be read
    pub fn verify_simple(path: &Path, algorithm: ChecksumAlgorithm, expected: &str) -> Result<()> {
        let generator = ChecksumGenerator::new().with_algorithms(vec![algorithm]);
        let checksum = generator.generate_file(path)?;

        let actual = checksum
            .checksums
            .get(&algorithm)
            .ok_or_else(|| Error::Metadata(format!("Checksum not generated for {algorithm:?}")))?;

        if actual != expected {
            return Err(Error::ChecksumMismatch {
                expected: expected.to_string(),
                actual: actual.clone(),
            });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_verify_success() {
        let mut file = NamedTempFile::new().expect("operation should succeed");
        file.write_all(b"Test content")
            .expect("operation should succeed");
        file.flush().expect("operation should succeed");

        let generator = ChecksumGenerator::new();
        let expected = generator
            .generate_file(file.path())
            .expect("operation should succeed");

        let verifier = ChecksumVerifier::new();
        let report = verifier
            .verify_file(&expected)
            .expect("operation should succeed");

        assert!(report.is_success());
        assert_eq!(report.failed_count(), 0);
    }

    #[test]
    fn test_verify_mismatch() {
        let mut file = NamedTempFile::new().expect("operation should succeed");
        file.write_all(b"Original content")
            .expect("operation should succeed");
        file.flush().expect("operation should succeed");

        let generator = ChecksumGenerator::new();
        let mut expected = generator
            .generate_file(file.path())
            .expect("operation should succeed");

        // Modify the expected checksum
        expected
            .checksums
            .insert(ChecksumAlgorithm::Sha256, "deadbeef".to_string());

        let verifier = ChecksumVerifier::new();
        let report = verifier
            .verify_file(&expected)
            .expect("operation should succeed");

        assert!(!report.is_success());
        assert_eq!(report.failed_count(), 1);
    }

    #[test]
    fn test_verify_simple() {
        let mut file = NamedTempFile::new().expect("operation should succeed");
        file.write_all(b"Simple test")
            .expect("operation should succeed");
        file.flush().expect("operation should succeed");

        let generator = ChecksumGenerator::new();
        let checksum = generator
            .generate_file(file.path())
            .expect("operation should succeed");
        let expected = checksum
            .checksums
            .get(&ChecksumAlgorithm::Sha256)
            .expect("operation should succeed");

        let result =
            ChecksumVerifier::verify_simple(file.path(), ChecksumAlgorithm::Sha256, expected);
        assert!(result.is_ok());
    }

    #[test]
    fn test_verify_batch() {
        let mut file1 = NamedTempFile::new().expect("operation should succeed");
        let mut file2 = NamedTempFile::new().expect("operation should succeed");
        file1
            .write_all(b"File 1")
            .expect("operation should succeed");
        file2
            .write_all(b"File 2")
            .expect("operation should succeed");
        file1.flush().expect("operation should succeed");
        file2.flush().expect("operation should succeed");

        let generator = ChecksumGenerator::new();
        let expected = vec![
            generator
                .generate_file(file1.path())
                .expect("operation should succeed"),
            generator
                .generate_file(file2.path())
                .expect("operation should succeed"),
        ];

        let verifier = ChecksumVerifier::new();
        let report = verifier
            .verify_batch(&expected)
            .expect("operation should succeed");

        assert!(report.is_success());
        assert_eq!(report.success_count(), 2);
        assert_eq!(report.failed_count(), 0);
    }
}
