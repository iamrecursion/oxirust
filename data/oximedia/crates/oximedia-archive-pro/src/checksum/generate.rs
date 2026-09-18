//! Checksum generation for files and data

use super::ChecksumAlgorithm;
use crate::{Error, Result};
use blake3::Hasher as Blake3Hasher;
use md5::{Digest, Md5};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Sha256, Sha512};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use xxhash_rust::xxh3::Xxh3;

/// File checksum with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileChecksum {
    /// File path
    pub path: PathBuf,
    /// File size in bytes
    pub size: u64,
    /// Checksums by algorithm
    pub checksums: HashMap<ChecksumAlgorithm, String>,
    /// Timestamp of checksum generation
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Checksum generator for files and directories
#[derive(Clone)]
pub struct ChecksumGenerator {
    algorithms: Vec<ChecksumAlgorithm>,
    buffer_size: usize,
    /// Case-insensitive extension allow-list (without the leading dot) used by
    /// `generate_directory` to skip files early during the `walkdir` traversal.
    /// `None` means "no filtering, visit every file".
    extension_filter: Option<Vec<String>>,
}

impl Default for ChecksumGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl ChecksumGenerator {
    /// Create a new checksum generator with default settings
    #[must_use]
    pub fn new() -> Self {
        Self {
            algorithms: vec![ChecksumAlgorithm::Sha256],
            buffer_size: 8192,
            extension_filter: None,
        }
    }

    /// Set the checksum algorithms to use
    #[must_use]
    pub fn with_algorithms(mut self, algorithms: Vec<ChecksumAlgorithm>) -> Self {
        self.algorithms = algorithms;
        self
    }

    /// Set the buffer size for reading files
    #[must_use]
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }

    /// Restrict `generate_directory` to only the given file extensions.
    ///
    /// Extensions are matched case-insensitively and may be supplied with or
    /// without a leading dot (e.g. both `"mkv"` and `".mkv"` are accepted).
    /// This lets large heterogeneous directory trees skip opening and hashing
    /// files that are irrelevant to the archive (logs, sidecar files, `.DS_Store`,
    /// etc.) — the filter is applied as part of the `walkdir` iterator chain,
    /// before any file is ever opened.
    #[must_use]
    pub fn with_extension_filter<I, S>(mut self, extensions: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.extension_filter = Some(
            extensions
                .into_iter()
                .map(|e| e.into().trim_start_matches('.').to_lowercase())
                .collect(),
        );
        self
    }

    /// Returns `true` if `path`'s extension passes the configured extension
    /// filter (or if no filter is configured).
    fn matches_extension_filter(&self, path: &Path) -> bool {
        match &self.extension_filter {
            None => true,
            Some(allowed) => path
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|ext| allowed.iter().any(|a| a.eq_ignore_ascii_case(ext))),
        }
    }

    /// Generate checksums for a single file
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read
    pub fn generate_file(&self, path: &Path) -> Result<FileChecksum> {
        let file = File::open(path)?;
        let metadata = file.metadata()?;
        let size = metadata.len();

        let mut reader = BufReader::with_capacity(self.buffer_size, file);
        let mut checksums = HashMap::new();

        // Initialize hashers
        let mut md5 = self
            .algorithms
            .contains(&ChecksumAlgorithm::Md5)
            .then(Md5::new);
        let mut sha256 = self
            .algorithms
            .contains(&ChecksumAlgorithm::Sha256)
            .then(Sha256::new);
        let mut sha512 = self
            .algorithms
            .contains(&ChecksumAlgorithm::Sha512)
            .then(Sha512::new);
        let mut xxhash = self
            .algorithms
            .contains(&ChecksumAlgorithm::XxHash64)
            .then(Xxh3::new);
        let mut blake3 = self
            .algorithms
            .contains(&ChecksumAlgorithm::Blake3)
            .then(Blake3Hasher::new);

        // Read file and update all hashers
        let mut buffer = vec![0u8; self.buffer_size];
        loop {
            let n = reader.read(&mut buffer)?;
            if n == 0 {
                break;
            }

            if let Some(ref mut h) = md5 {
                h.update(&buffer[..n]);
            }
            if let Some(ref mut h) = sha256 {
                h.update(&buffer[..n]);
            }
            if let Some(ref mut h) = sha512 {
                h.update(&buffer[..n]);
            }
            if let Some(ref mut h) = xxhash {
                h.update(&buffer[..n]);
            }
            if let Some(ref mut h) = blake3 {
                h.update(&buffer[..n]);
            }
        }

        // Finalize and store checksums
        if let Some(h) = md5 {
            checksums.insert(ChecksumAlgorithm::Md5, hex::encode(h.finalize()));
        }
        if let Some(h) = sha256 {
            checksums.insert(ChecksumAlgorithm::Sha256, hex::encode(h.finalize()));
        }
        if let Some(h) = sha512 {
            checksums.insert(ChecksumAlgorithm::Sha512, hex::encode(h.finalize()));
        }
        if let Some(h) = xxhash {
            checksums.insert(ChecksumAlgorithm::XxHash64, format!("{:x}", h.digest()));
        }
        if let Some(h) = blake3 {
            checksums.insert(
                ChecksumAlgorithm::Blake3,
                format!("{}", h.finalize().to_hex()),
            );
        }

        Ok(FileChecksum {
            path: path.to_path_buf(),
            size,
            checksums,
            timestamp: chrono::Utc::now(),
        })
    }

    /// Generate checksums for all files in a directory (recursively)
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be read or any file fails
    pub fn generate_directory(&self, dir: &Path) -> Result<Vec<FileChecksum>> {
        let entries: Vec<PathBuf> = walkdir::WalkDir::new(dir)
            .follow_links(false)
            .into_iter()
            .filter_map(std::result::Result::ok)
            .filter(|e| e.file_type().is_file())
            // Early extension filtering: reject non-matching files before they
            // are ever converted to an owned PathBuf or opened for hashing.
            .filter(|e| self.matches_extension_filter(e.path()))
            .map(walkdir::DirEntry::into_path)
            .collect();

        entries
            .par_iter()
            .map(|path| self.generate_file(path))
            .collect()
    }

    /// Generate checksums for a list of files in parallel
    ///
    /// # Errors
    ///
    /// Returns an error if any file cannot be read
    pub fn generate_batch(&self, paths: &[PathBuf]) -> Result<Vec<FileChecksum>> {
        paths
            .par_iter()
            .map(|path| self.generate_file(path))
            .collect()
    }
}

/// Generate a quick checksum using BLAKE3 (fastest)
///
/// # Errors
///
/// Returns an error if the file cannot be read
pub fn quick_checksum(path: &Path) -> Result<String> {
    let generator = ChecksumGenerator::new().with_algorithms(vec![ChecksumAlgorithm::Blake3]);
    let checksum = generator.generate_file(path)?;
    checksum
        .checksums
        .get(&ChecksumAlgorithm::Blake3)
        .cloned()
        .ok_or_else(|| Error::Metadata("BLAKE3 checksum not found".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_generate_file_checksum() {
        let mut file = NamedTempFile::new().expect("operation should succeed");
        file.write_all(b"Hello, World!")
            .expect("operation should succeed");
        file.flush().expect("operation should succeed");

        let generator = ChecksumGenerator::new();
        let checksum = generator
            .generate_file(file.path())
            .expect("operation should succeed");

        assert_eq!(checksum.size, 13);
        assert_eq!(checksum.checksums.len(), 1);
        assert!(checksum.checksums.contains_key(&ChecksumAlgorithm::Sha256));
    }

    #[test]
    fn test_multiple_algorithms() {
        let mut file = NamedTempFile::new().expect("operation should succeed");
        file.write_all(b"Test data")
            .expect("operation should succeed");
        file.flush().expect("operation should succeed");

        let generator = ChecksumGenerator::new().with_algorithms(vec![
            ChecksumAlgorithm::Md5,
            ChecksumAlgorithm::Sha256,
            ChecksumAlgorithm::Blake3,
        ]);

        let checksum = generator
            .generate_file(file.path())
            .expect("operation should succeed");
        assert_eq!(checksum.checksums.len(), 3);
    }

    #[test]
    fn test_quick_checksum() {
        let mut file = NamedTempFile::new().expect("operation should succeed");
        file.write_all(b"Quick test")
            .expect("operation should succeed");
        file.flush().expect("operation should succeed");

        let hash = quick_checksum(file.path()).expect("operation should succeed");
        assert!(!hash.is_empty());
    }

    /// `generate_directory` with an extension allow-list should skip files
    /// whose extension does not match, without erroring on them (they are
    /// never opened or hashed at all).
    #[test]
    fn test_generate_directory_extension_filter() {
        let dir = tempfile::tempdir().expect("temp dir creation should succeed");

        std::fs::write(dir.path().join("master.mkv"), b"video content")
            .expect("write mkv should succeed");
        std::fs::write(dir.path().join("audio.flac"), b"audio content")
            .expect("write flac should succeed");
        std::fs::write(dir.path().join("notes.txt"), b"sidecar notes")
            .expect("write txt should succeed");
        std::fs::write(dir.path().join(".DS_Store"), b"junk")
            .expect("write junk file should succeed");

        let generator =
            ChecksumGenerator::new().with_extension_filter(["mkv".to_string(), "FLAC".to_string()]);
        let mut checksums = generator
            .generate_directory(dir.path())
            .expect("filtered directory generation should succeed");
        checksums.sort_by(|a, b| a.path.cmp(&b.path));

        assert_eq!(
            checksums.len(),
            2,
            "only the .mkv and .flac files should have been visited"
        );
        let names: Vec<String> = checksums
            .iter()
            .filter_map(|c| c.path.file_name())
            .map(|n| n.to_string_lossy().into_owned())
            .collect();
        assert!(names.contains(&"master.mkv".to_string()));
        assert!(names.contains(&"audio.flac".to_string()));
        assert!(!names.iter().any(|n| n == "notes.txt"));
        assert!(!names.iter().any(|n| n == ".DS_Store"));
    }

    /// Without a filter configured, `generate_directory` must still visit
    /// every file (unfiltered behaviour is preserved).
    #[test]
    fn test_generate_directory_no_filter_visits_all_files() {
        let dir = tempfile::tempdir().expect("temp dir creation should succeed");
        std::fs::write(dir.path().join("a.bin"), b"aaa").expect("write should succeed");
        std::fs::write(dir.path().join("b.dat"), b"bbb").expect("write should succeed");

        let generator = ChecksumGenerator::new();
        let checksums = generator
            .generate_directory(dir.path())
            .expect("unfiltered directory generation should succeed");
        assert_eq!(checksums.len(), 2);
    }
}
