// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Frame and output verification.

use crate::error::{Error, Result};
use blake3::Hasher;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tokio::fs;
use tokio::io::AsyncReadExt;

/// Verification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    /// File path
    pub path: PathBuf,
    /// Is valid
    pub valid: bool,
    /// Checksum
    pub checksum: String,
    /// File size
    pub size: u64,
    /// Issues found
    pub issues: Vec<String>,
}

/// Frame verifier
pub struct Verifier {
    expected_checksums: std::collections::HashMap<PathBuf, String>,
    min_file_size: u64,
}

impl Verifier {
    /// Create a new verifier
    #[must_use]
    pub fn new() -> Self {
        Self {
            expected_checksums: std::collections::HashMap::new(),
            min_file_size: 1024, // 1 KB minimum
        }
    }

    /// Set expected checksum
    pub fn set_expected_checksum(&mut self, path: PathBuf, checksum: String) {
        self.expected_checksums.insert(path, checksum);
    }

    /// Verify a file
    pub async fn verify_file<P: AsRef<Path>>(&self, path: P) -> Result<VerificationResult> {
        let path = path.as_ref();
        let mut issues = Vec::new();

        // Check if file exists
        if !path.exists() {
            return Ok(VerificationResult {
                path: path.to_path_buf(),
                valid: false,
                checksum: String::new(),
                size: 0,
                issues: vec!["File does not exist".to_string()],
            });
        }

        // Get file size
        let metadata = fs::metadata(path).await?;
        let size = metadata.len();

        // Check minimum size
        if size < self.min_file_size {
            issues.push(format!("File size too small: {size} bytes"));
        }

        // Calculate checksum
        let checksum = self.calculate_checksum(path).await?;

        // Verify checksum if expected
        if let Some(expected) = self.expected_checksums.get(path) {
            if &checksum != expected {
                issues.push(format!(
                    "Checksum mismatch: expected {expected}, got {checksum}"
                ));
            }
        }

        // Verify file format (basic check)
        if let Err(e) = self.verify_format(path).await {
            issues.push(format!("Format verification failed: {e}"));
        }

        Ok(VerificationResult {
            path: path.to_path_buf(),
            valid: issues.is_empty(),
            checksum,
            size,
            issues,
        })
    }

    /// Calculate checksum
    async fn calculate_checksum<P: AsRef<Path>>(&self, path: P) -> Result<String> {
        let mut file = fs::File::open(path).await?;
        let mut hasher = Hasher::new();
        let mut buffer = vec![0; 65536]; // 64 KB buffer

        loop {
            let n = file.read(&mut buffer).await?;
            if n == 0 {
                break;
            }
            hasher.update(&buffer[..n]);
        }

        Ok(hasher.finalize().to_hex().to_string())
    }

    /// Verify file format
    async fn verify_format<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();

        // Check file extension
        if let Some(ext) = path.extension() {
            let ext = ext.to_string_lossy().to_lowercase();

            // Read first few bytes to verify format
            let mut file = fs::File::open(path).await?;
            let mut header = vec![0u8; 16];
            file.read_exact(&mut header).await?;

            match ext.as_str() {
                "png" => {
                    if !header.starts_with(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) {
                        return Err(Error::VerificationFailed("Invalid PNG header".to_string()));
                    }
                }
                "jpg" | "jpeg" => {
                    if !header.starts_with(&[0xFF, 0xD8, 0xFF]) {
                        return Err(Error::VerificationFailed("Invalid JPEG header".to_string()));
                    }
                }
                "exr" => {
                    if !header.starts_with(&[0x76, 0x2F, 0x31, 0x01]) {
                        return Err(Error::VerificationFailed("Invalid EXR header".to_string()));
                    }
                }
                _ => {}
            }
        }

        Ok(())
    }

    /// Verify frame sequence
    pub async fn verify_sequence(&self, frames: Vec<PathBuf>) -> Result<Vec<VerificationResult>> {
        let mut results = Vec::new();

        for frame in frames {
            let result = self.verify_file(&frame).await?;
            results.push(result);
        }

        Ok(results)
    }

    /// Check if all frames are valid
    #[must_use]
    pub fn all_valid(results: &[VerificationResult]) -> bool {
        results.iter().all(|r| r.valid)
    }

    /// Get failed frames
    #[must_use]
    pub fn get_failed(results: &[VerificationResult]) -> Vec<&VerificationResult> {
        results.iter().filter(|r| !r.valid).collect()
    }
}

impl Default for Verifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::AsyncWriteExt;

    #[tokio::test]
    async fn test_verifier_creation() {
        let verifier = Verifier::new();
        assert_eq!(verifier.expected_checksums.len(), 0);
    }

    #[tokio::test]
    async fn test_verify_nonexistent_file() -> Result<()> {
        let verifier = Verifier::new();
        let result = verifier.verify_file("/nonexistent/file.png").await?;

        assert!(!result.valid);
        assert!(!result.issues.is_empty());

        Ok(())
    }

    #[tokio::test]
    async fn test_verify_existing_file() -> Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let file_path = temp_dir.path().join("test.txt");

        // Create a test file >= 1024 bytes to satisfy min_file_size
        let mut file = fs::File::create(&file_path).await?;
        file.write_all(&vec![b'A'; 1024]).await?;
        file.flush().await?;
        drop(file);

        let verifier = Verifier::new();
        let result = verifier.verify_file(&file_path).await?;

        assert!(result.valid);
        assert!(!result.checksum.is_empty());
        assert!(result.size > 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_checksum_verification() -> Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let file_path = temp_dir.path().join("test.txt");

        // Create a test file >= 1024 bytes to satisfy min_file_size
        let mut file = fs::File::create(&file_path).await?;
        file.write_all(&vec![b'T'; 1024]).await?;
        file.flush().await?;
        drop(file);

        let mut verifier = Verifier::new();

        // Calculate checksum first
        let result1 = verifier.verify_file(&file_path).await?;
        let checksum = result1.checksum.clone();

        // Set expected checksum
        verifier.set_expected_checksum(file_path.clone(), checksum);

        // Verify again - should pass
        let result2 = verifier.verify_file(&file_path).await?;
        assert!(result2.valid);

        // Set wrong checksum
        verifier.set_expected_checksum(file_path.clone(), "wrong_checksum".to_string());

        // Verify again - should fail
        let result3 = verifier.verify_file(&file_path).await?;
        assert!(!result3.valid);
        assert!(result3
            .issues
            .iter()
            .any(|i| i.contains("Checksum mismatch")));

        Ok(())
    }

    #[tokio::test]
    async fn test_verify_sequence() -> Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let mut frames = Vec::new();

        // Create test frames >= 1024 bytes each to satisfy min_file_size
        for i in 1..=3 {
            let file_path = temp_dir.path().join(format!("frame_{i:04}.txt"));
            let mut file = fs::File::create(&file_path).await?;
            let content = format!("{i:A>1024}"); // 1024-byte padded string
            file.write_all(content.as_bytes()).await?;
            file.flush().await?;
            frames.push(file_path);
        }

        let verifier = Verifier::new();
        let results = verifier.verify_sequence(frames).await?;

        assert_eq!(results.len(), 3);
        assert!(Verifier::all_valid(&results));

        Ok(())
    }

    #[tokio::test]
    async fn test_get_failed_frames() -> Result<()> {
        let results = vec![
            VerificationResult {
                path: PathBuf::from("frame1.png"),
                valid: true,
                checksum: "abc".to_string(),
                size: 1000,
                issues: Vec::new(),
            },
            VerificationResult {
                path: PathBuf::from("frame2.png"),
                valid: false,
                checksum: "def".to_string(),
                size: 100,
                issues: vec!["Too small".to_string()],
            },
        ];

        let failed = Verifier::get_failed(&results);
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].path, PathBuf::from("frame2.png"));

        Ok(())
    }
}
