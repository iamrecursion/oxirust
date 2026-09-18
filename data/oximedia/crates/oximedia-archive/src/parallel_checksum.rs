//! Parallel multi-algorithm checksum computation.
//!
//! Computes BLAKE3, SHA-256, and CRC32 simultaneously for each file using
//! separate threads. This avoids the sequential penalty of computing each
//! hash one after another, achieving near-single-pass throughput.

use crate::{ArchiveError, ArchiveResult, ChecksumSet};
use std::path::Path;
use std::sync::mpsc;

/// Configuration for parallel checksum computation.
#[derive(Debug, Clone)]
pub struct ParallelChecksumConfig {
    /// Enable BLAKE3 computation.
    pub enable_blake3: bool,
    /// Enable SHA-256 computation.
    pub enable_sha256: bool,
    /// Enable CRC32 computation.
    pub enable_crc32: bool,
    /// Enable MD5 computation (legacy).
    pub enable_md5: bool,
    /// Read buffer size in bytes.
    pub buffer_size: usize,
}

impl Default for ParallelChecksumConfig {
    fn default() -> Self {
        Self {
            enable_blake3: true,
            enable_sha256: true,
            enable_crc32: true,
            enable_md5: false,
            buffer_size: 1024 * 1024, // 1 MiB
        }
    }
}

/// Result of parallel checksum computation for a single file.
#[derive(Debug, Clone)]
pub struct ParallelChecksumResult {
    /// The computed checksums.
    pub checksums: ChecksumSet,
    /// Total bytes processed.
    pub bytes_processed: u64,
    /// Number of algorithms run in parallel.
    pub algorithms_used: usize,
}

/// Which algorithm a worker thread should compute.
#[derive(Debug, Clone, Copy)]
enum Algorithm {
    Blake3,
    Sha256,
    Crc32,
    Md5,
}

impl std::fmt::Display for Algorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Blake3 => write!(f, "BLAKE3"),
            Self::Sha256 => write!(f, "SHA-256"),
            Self::Crc32 => write!(f, "CRC32"),
            Self::Md5 => write!(f, "MD5"),
        }
    }
}

/// Compute checksums for in-memory data using parallel threads.
///
/// Each enabled algorithm runs in its own thread. Data is cloned per
/// algorithm, but for large files this is still faster than sequential
/// computation because hash functions are CPU-bound and benefit from
/// multi-core execution.
pub fn compute_parallel(
    data: &[u8],
    config: &ParallelChecksumConfig,
) -> ArchiveResult<ParallelChecksumResult> {
    let mut algorithms: Vec<Algorithm> = Vec::new();
    if config.enable_blake3 {
        algorithms.push(Algorithm::Blake3);
    }
    if config.enable_sha256 {
        algorithms.push(Algorithm::Sha256);
    }
    if config.enable_crc32 {
        algorithms.push(Algorithm::Crc32);
    }
    if config.enable_md5 {
        algorithms.push(Algorithm::Md5);
    }

    if algorithms.is_empty() {
        return Ok(ParallelChecksumResult {
            checksums: ChecksumSet::default(),
            bytes_processed: data.len() as u64,
            algorithms_used: 0,
        });
    }

    let algorithms_used = algorithms.len();
    let (tx, rx) = mpsc::channel();

    // Spawn a thread per algorithm
    let mut handles = Vec::with_capacity(algorithms.len());
    for algo in &algorithms {
        let data_clone = data.to_vec();
        let algo = *algo;
        let tx = tx.clone();

        let handle = std::thread::spawn(move || {
            let hex = match algo {
                Algorithm::Blake3 => {
                    let hash = blake3::hash(&data_clone);
                    hash.to_hex().to_string()
                }
                Algorithm::Sha256 => {
                    use sha2::Digest;
                    let mut hasher = sha2::Sha256::new();
                    hasher.update(&data_clone);
                    let result = hasher.finalize();
                    hex::encode(result)
                }
                Algorithm::Crc32 => {
                    let crc = crc32fast::hash(&data_clone);
                    format!("{crc:08x}")
                }
                Algorithm::Md5 => {
                    use md5::Digest;
                    let mut hasher = md5::Md5::new();
                    hasher.update(&data_clone);
                    let result = hasher.finalize();
                    hex::encode(result)
                }
            };
            // Ignore send error (receiver dropped)
            let _ = tx.send((algo, hex));
        });
        handles.push(handle);
    }

    // Drop our sender so the receiver knows when all senders are done
    drop(tx);

    // Collect results
    let mut checksums = ChecksumSet::default();
    for (algo, hex) in rx {
        match algo {
            Algorithm::Blake3 => checksums.blake3 = Some(hex),
            Algorithm::Sha256 => checksums.sha256 = Some(hex),
            Algorithm::Crc32 => checksums.crc32 = Some(hex),
            Algorithm::Md5 => checksums.md5 = Some(hex),
        }
    }

    // Wait for all threads to finish
    for handle in handles {
        handle.join().map_err(|_| {
            ArchiveError::Validation("parallel checksum thread panicked".to_string())
        })?;
    }

    Ok(ParallelChecksumResult {
        checksums,
        bytes_processed: data.len() as u64,
        algorithms_used,
    })
}

/// Compute parallel checksums for a file on disk.
///
/// Reads the file once into memory, then dispatches to [`compute_parallel`].
/// For very large files, consider streaming approaches instead.
pub fn compute_parallel_file(
    path: &Path,
    config: &ParallelChecksumConfig,
) -> ArchiveResult<ParallelChecksumResult> {
    let data = std::fs::read(path)?;
    compute_parallel(&data, config)
}

/// Compute parallel checksums for multiple files using rayon for file-level
/// parallelism and per-file algorithm parallelism.
///
/// Returns results in the same order as the input paths. If any file fails,
/// that entry contains the error.
pub fn compute_parallel_batch(
    paths: &[&Path],
    config: &ParallelChecksumConfig,
) -> Vec<ArchiveResult<ParallelChecksumResult>> {
    use rayon::prelude::*;

    paths
        .par_iter()
        .map(|path| compute_parallel_file(path, config))
        .collect()
}

/// Verify that all algorithms in `expected` match `actual`. Returns a list of
/// mismatched algorithm names.
#[must_use]
pub fn verify_checksums(expected: &ChecksumSet, actual: &ChecksumSet) -> Vec<String> {
    let mut mismatches = Vec::new();

    if let (Some(ref exp), Some(ref act)) = (&expected.blake3, &actual.blake3) {
        if exp != act {
            mismatches.push("blake3".to_string());
        }
    }
    if let (Some(ref exp), Some(ref act)) = (&expected.sha256, &actual.sha256) {
        if exp != act {
            mismatches.push("sha256".to_string());
        }
    }
    if let (Some(ref exp), Some(ref act)) = (&expected.crc32, &actual.crc32) {
        if exp != act {
            mismatches.push("crc32".to_string());
        }
    }
    if let (Some(ref exp), Some(ref act)) = (&expected.md5, &actual.md5) {
        if exp != act {
            mismatches.push("md5".to_string());
        }
    }

    mismatches
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_all_algorithms() {
        let data = b"hello world parallel checksumming";
        let config = ParallelChecksumConfig::default();
        let result = compute_parallel(data, &config).expect("compute failed");
        assert!(result.checksums.blake3.is_some());
        assert!(result.checksums.sha256.is_some());
        assert!(result.checksums.crc32.is_some());
        assert_eq!(result.bytes_processed, data.len() as u64);
        assert_eq!(result.algorithms_used, 3);
    }

    #[test]
    fn test_parallel_with_md5() {
        let data = b"test md5 parallel";
        let config = ParallelChecksumConfig {
            enable_md5: true,
            ..Default::default()
        };
        let result = compute_parallel(data, &config).expect("compute failed");
        assert!(result.checksums.blake3.is_some());
        assert!(result.checksums.sha256.is_some());
        assert!(result.checksums.crc32.is_some());
        assert!(result.checksums.md5.is_some());
        assert_eq!(result.algorithms_used, 4);
    }

    #[test]
    fn test_parallel_no_algorithms() {
        let data = b"no algorithms";
        let config = ParallelChecksumConfig {
            enable_blake3: false,
            enable_sha256: false,
            enable_crc32: false,
            enable_md5: false,
            ..Default::default()
        };
        let result = compute_parallel(data, &config).expect("compute failed");
        assert!(result.checksums.blake3.is_none());
        assert!(result.checksums.sha256.is_none());
        assert!(result.checksums.crc32.is_none());
        assert_eq!(result.algorithms_used, 0);
    }

    #[test]
    fn test_parallel_deterministic() {
        let data = b"deterministic checksum test data";
        let config = ParallelChecksumConfig::default();
        let r1 = compute_parallel(data, &config).expect("first");
        let r2 = compute_parallel(data, &config).expect("second");
        assert_eq!(r1.checksums.blake3, r2.checksums.blake3);
        assert_eq!(r1.checksums.sha256, r2.checksums.sha256);
        assert_eq!(r1.checksums.crc32, r2.checksums.crc32);
    }

    #[test]
    fn test_parallel_empty_data() {
        let config = ParallelChecksumConfig::default();
        let result = compute_parallel(b"", &config).expect("compute failed");
        assert!(result.checksums.blake3.is_some());
        assert!(result.checksums.sha256.is_some());
        assert!(result.checksums.crc32.is_some());
        assert_eq!(result.bytes_processed, 0);
    }

    #[test]
    fn test_parallel_sha256_known_vector() {
        let data = b"abc";
        let config = ParallelChecksumConfig {
            enable_blake3: false,
            enable_sha256: true,
            enable_crc32: false,
            enable_md5: false,
            ..Default::default()
        };
        let result = compute_parallel(data, &config).expect("compute failed");
        assert_eq!(
            result.checksums.sha256.as_deref(),
            Some("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        );
    }

    #[test]
    fn test_parallel_file() {
        let dir = std::env::temp_dir().join("oximedia_parallel_cksum_test");
        std::fs::create_dir_all(&dir).ok();
        let file_path = dir.join("test_file.bin");
        let content = b"file content for parallel checksum test";
        std::fs::write(&file_path, content).expect("write file");

        let config = ParallelChecksumConfig::default();
        let result = compute_parallel_file(&file_path, &config).expect("compute file failed");
        assert!(result.checksums.blake3.is_some());
        assert_eq!(result.bytes_processed, content.len() as u64);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_parallel_batch() {
        let dir = std::env::temp_dir().join("oximedia_parallel_batch_test");
        std::fs::create_dir_all(&dir).ok();

        let files: Vec<_> = (0..5)
            .map(|i| {
                let p = dir.join(format!("batch_{i}.bin"));
                let content = format!("batch file content {i}");
                std::fs::write(&p, content.as_bytes()).expect("write batch file");
                p
            })
            .collect();

        let paths: Vec<&Path> = files.iter().map(|p| p.as_path()).collect();
        let config = ParallelChecksumConfig::default();
        let results = compute_parallel_batch(&paths, &config);
        assert_eq!(results.len(), 5);
        for result in &results {
            assert!(result.is_ok());
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_verify_checksums_match() {
        let data = b"verify test";
        let config = ParallelChecksumConfig::default();
        let r = compute_parallel(data, &config).expect("compute");
        let mismatches = verify_checksums(&r.checksums, &r.checksums);
        assert!(mismatches.is_empty());
    }

    #[test]
    fn test_verify_checksums_mismatch() {
        let expected = ChecksumSet {
            blake3: Some("aaa".to_string()),
            sha256: Some("bbb".to_string()),
            crc32: Some("ccc".to_string()),
            md5: None,
        };
        let actual = ChecksumSet {
            blake3: Some("xxx".to_string()),
            sha256: Some("bbb".to_string()),
            crc32: Some("yyy".to_string()),
            md5: None,
        };
        let mismatches = verify_checksums(&expected, &actual);
        assert_eq!(mismatches.len(), 2);
        assert!(mismatches.contains(&"blake3".to_string()));
        assert!(mismatches.contains(&"crc32".to_string()));
    }

    #[test]
    fn test_parallel_large_data() {
        let data: Vec<u8> = (0u8..=255).cycle().take(1024 * 1024).collect();
        let config = ParallelChecksumConfig::default();
        let result = compute_parallel(&data, &config).expect("compute large");
        assert_eq!(result.bytes_processed, 1024 * 1024);
        assert!(result.checksums.blake3.is_some());
    }

    #[test]
    fn test_parallel_single_algorithm_blake3() {
        let data = b"blake3 only";
        let config = ParallelChecksumConfig {
            enable_blake3: true,
            enable_sha256: false,
            enable_crc32: false,
            enable_md5: false,
            ..Default::default()
        };
        let result = compute_parallel(data, &config).expect("compute");
        assert!(result.checksums.blake3.is_some());
        assert!(result.checksums.sha256.is_none());
        assert!(result.checksums.crc32.is_none());
        assert_eq!(result.algorithms_used, 1);
    }

    #[test]
    fn test_parallel_file_not_found() {
        let config = ParallelChecksumConfig::default();
        let result = compute_parallel_file(Path::new("/nonexistent/file.bin"), &config);
        assert!(result.is_err());
    }
}
