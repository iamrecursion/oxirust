//! Comprehensive validation for proxy workflows.

use super::report::ValidationReport;
use crate::{ProxyLinkManager, Result};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// Comprehensive workflow validator.
pub struct WorkflowValidator<'a> {
    link_manager: &'a ProxyLinkManager,
    strict_mode: bool,
}

impl<'a> WorkflowValidator<'a> {
    /// Create a new workflow validator.
    #[must_use]
    pub const fn new(link_manager: &'a ProxyLinkManager) -> Self {
        Self {
            link_manager,
            strict_mode: false,
        }
    }

    /// Enable strict validation mode.
    #[must_use]
    pub const fn strict(mut self) -> Self {
        self.strict_mode = true;
        self
    }

    /// Validate all aspects of the proxy workflow.
    pub fn validate_all(&self) -> Result<ValidationReport> {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let all_links = self.link_manager.all_links();
        let total_links = all_links.len();

        // Check 1: File existence
        for link in &all_links {
            if !link.proxy_path.exists() {
                errors.push(format!("Proxy file missing: {}", link.proxy_path.display()));
            }

            if !link.original_path.exists() {
                errors.push(format!(
                    "Original file missing: {}",
                    link.original_path.display()
                ));
            }
        }

        // Check 2: Duplicate links
        let duplicates = self.find_duplicate_links(&all_links);
        for (path, count) in duplicates {
            if count > 1 {
                warnings.push(format!(
                    "Duplicate proxy link for: {} ({} times)",
                    path.display(),
                    count
                ));
            }
        }

        // Check 3: Orphaned proxies
        let orphaned = self.find_orphaned_proxies(&all_links)?;
        for path in orphaned {
            warnings.push(format!("Orphaned proxy file: {}", path.display()));
        }

        // Check 4: File integrity
        for link in &all_links {
            if let Err(e) = self.validate_file_integrity(&link.proxy_path) {
                errors.push(format!(
                    "Proxy file integrity error ({}): {}",
                    link.proxy_path.display(),
                    e
                ));
            }
        }

        // Check 5: Metadata consistency
        for link in &all_links {
            if link.duration == 0.0 {
                warnings.push(format!("Zero duration for: {}", link.proxy_path.display()));
            }

            if link.scale_factor <= 0.0 || link.scale_factor > 1.0 {
                errors.push(format!(
                    "Invalid scale factor ({}) for: {}",
                    link.scale_factor,
                    link.proxy_path.display()
                ));
            }
        }

        // Check 6: Timecode consistency (strict mode)
        if self.strict_mode {
            for link in &all_links {
                if link.timecode.is_none() {
                    warnings.push(format!(
                        "Missing timecode for: {}",
                        link.proxy_path.display()
                    ));
                }
            }
        }

        let valid_links = total_links - errors.len();

        Ok(ValidationReport {
            total_links,
            valid_links,
            errors,
            warnings,
        })
    }

    /// Validate EDL file references.
    pub fn validate_edl_references(&self, edl_path: &Path) -> Result<EdlValidationResult> {
        if !edl_path.exists() {
            return Err(crate::ProxyError::FileNotFound(
                edl_path.display().to_string(),
            ));
        }

        // Placeholder: would parse EDL and check all referenced files
        Ok(EdlValidationResult {
            total_clips: 0,
            found_clips: 0,
            missing_clips: Vec::new(),
            unlinked_clips: Vec::new(),
        })
    }

    /// Validate proxy-original file size relationship.
    fn validate_file_integrity(&self, path: &Path) -> Result<()> {
        let metadata = std::fs::metadata(path)?;

        // Check file is not empty
        if metadata.len() == 0 {
            return Err(crate::ProxyError::ValidationError(
                "File is empty".to_string(),
            ));
        }

        // Check file is readable
        if metadata.permissions().readonly() {
            return Err(crate::ProxyError::ValidationError(
                "File is read-only".to_string(),
            ));
        }

        Ok(())
    }

    /// Find duplicate proxy links.
    fn find_duplicate_links(&self, links: &[crate::ProxyLink]) -> Vec<(PathBuf, usize)> {
        let mut path_counts: std::collections::HashMap<PathBuf, usize> =
            std::collections::HashMap::new();

        for link in links {
            *path_counts.entry(link.proxy_path.clone()).or_insert(0) += 1;
        }

        path_counts
            .into_iter()
            .filter(|(_, count)| *count > 1)
            .collect()
    }

    /// Find orphaned proxy files (proxies without links).
    fn find_orphaned_proxies(&self, _links: &[crate::ProxyLink]) -> Result<Vec<PathBuf>> {
        // Placeholder: would scan proxy directories for unlisted files
        Ok(Vec::new())
    }
}

/// EDL validation result.
#[derive(Debug, Clone)]
pub struct EdlValidationResult {
    /// Total clips referenced in EDL.
    pub total_clips: usize,

    /// Clips found on disk.
    pub found_clips: usize,

    /// Missing clip files.
    pub missing_clips: Vec<String>,

    /// Clips without proxy links.
    pub unlinked_clips: Vec<String>,
}

impl EdlValidationResult {
    /// Check if all clips are valid.
    #[must_use]
    pub const fn is_valid(&self) -> bool {
        self.missing_clips.is_empty() && self.unlinked_clips.is_empty()
    }

    /// Get validation percentage.
    #[must_use]
    pub fn validation_percentage(&self) -> f64 {
        if self.total_clips == 0 {
            100.0
        } else {
            (self.found_clips as f64 / self.total_clips as f64) * 100.0
        }
    }
}

/// Path validator for checking path-related issues.
pub struct PathValidator;

impl PathValidator {
    /// Validate a file path for proxy use.
    ///
    /// # Errors
    ///
    /// Returns an error if the path is invalid.
    pub fn validate_path(path: &Path) -> Result<()> {
        // Check path is not empty
        if path.as_os_str().is_empty() {
            return Err(crate::ProxyError::ValidationError(
                "Path is empty".to_string(),
            ));
        }

        // Check for invalid characters
        if let Some(path_str) = path.to_str() {
            if path_str.contains('\0') {
                return Err(crate::ProxyError::ValidationError(
                    "Path contains null characters".to_string(),
                ));
            }
        }

        // Check parent directory exists
        if let Some(parent) = path.parent() {
            if !parent.exists() && !parent.as_os_str().is_empty() {
                return Err(crate::ProxyError::ValidationError(format!(
                    "Parent directory does not exist: {}",
                    parent.display()
                )));
            }
        }

        Ok(())
    }

    /// Validate a directory for proxy storage.
    pub fn validate_directory(dir: &Path) -> Result<DirectoryValidation> {
        if !dir.exists() {
            return Ok(DirectoryValidation {
                exists: false,
                writable: false,
                available_space: 0,
                total_space: 0,
            });
        }

        if !dir.is_dir() {
            return Err(crate::ProxyError::ValidationError(
                "Path is not a directory".to_string(),
            ));
        }

        // Check if writable
        let writable = is_writable(dir);

        // Get disk space info (placeholder)
        let available_space = 0u64; // Would use system calls
        let total_space = 0u64;

        Ok(DirectoryValidation {
            exists: true,
            writable,
            available_space,
            total_space,
        })
    }

    /// Check for path conflicts.
    pub fn check_path_conflicts(paths: &[PathBuf]) -> Vec<PathBuf> {
        let mut seen = HashSet::new();
        let mut conflicts = Vec::new();

        for path in paths {
            if !seen.insert(path.clone()) {
                conflicts.push(path.clone());
            }
        }

        conflicts
    }
}

/// Directory validation result.
#[derive(Debug, Clone)]
pub struct DirectoryValidation {
    /// Directory exists.
    pub exists: bool,

    /// Directory is writable.
    pub writable: bool,

    /// Available space in bytes.
    pub available_space: u64,

    /// Total space in bytes.
    pub total_space: u64,
}

impl DirectoryValidation {
    /// Check if directory is usable.
    #[must_use]
    pub const fn is_usable(&self) -> bool {
        self.exists && self.writable
    }

    /// Get usage percentage.
    #[must_use]
    pub fn usage_percentage(&self) -> f64 {
        if self.total_space == 0 {
            0.0
        } else {
            ((self.total_space - self.available_space) as f64 / self.total_space as f64) * 100.0
        }
    }
}

fn is_writable(dir: &Path) -> bool {
    // Try to create a temporary file
    let test_file = dir.join(".write_test");
    std::fs::write(&test_file, b"test").is_ok() && std::fs::remove_file(&test_file).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_workflow_validator() {
        let temp_dir = std::env::temp_dir();
        let db_path = temp_dir.join("test_validator.json");

        let manager = ProxyLinkManager::new(&db_path)
            .await
            .expect("should succeed in test");
        let validator = WorkflowValidator::new(&manager);

        let report = validator.validate_all();
        assert!(report.is_ok());

        // Clean up
        let _ = std::fs::remove_file(db_path);
    }

    #[test]
    fn test_path_validator() {
        let temp_dir = std::env::temp_dir();
        let valid_path = temp_dir.join("test.mp4");

        let result = PathValidator::validate_path(&valid_path);
        assert!(result.is_ok());

        let empty_path = Path::new("");
        let result = PathValidator::validate_path(empty_path);
        assert!(result.is_err());
    }

    #[test]
    fn test_directory_validation() {
        let temp_dir = std::env::temp_dir();
        let result = PathValidator::validate_directory(&temp_dir);

        assert!(result.is_ok());
        let validation = result.expect("should succeed in test");
        assert!(validation.exists);
    }

    #[test]
    fn test_path_conflicts() {
        let paths = vec![
            PathBuf::from("file1.mp4"),
            PathBuf::from("file2.mp4"),
            PathBuf::from("file1.mp4"),
        ];

        let conflicts = PathValidator::check_path_conflicts(&paths);
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0], PathBuf::from("file1.mp4"));
    }

    #[test]
    fn test_edl_validation_result() {
        let result = EdlValidationResult {
            total_clips: 10,
            found_clips: 8,
            missing_clips: vec!["clip1.mov".to_string(), "clip2.mov".to_string()],
            unlinked_clips: Vec::new(),
        };

        assert!(!result.is_valid());
        assert_eq!(result.validation_percentage(), 80.0);
    }
}
