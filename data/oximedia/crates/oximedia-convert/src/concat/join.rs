// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! File concatenation for joining multiple media files.

use crate::{ConversionError, Result};
use std::path::{Path, PathBuf};

/// Joiner for concatenating multiple media files.
#[derive(Debug, Clone)]
pub struct FileJoiner {
    reencode: bool,
    validate_compatibility: bool,
}

impl FileJoiner {
    /// Create a new file joiner.
    #[must_use]
    pub fn new() -> Self {
        Self {
            reencode: false,
            validate_compatibility: true,
        }
    }

    /// Set whether to re-encode during concatenation.
    #[must_use]
    pub fn with_reencode(mut self, reencode: bool) -> Self {
        self.reencode = reencode;
        self
    }

    /// Set whether to validate file compatibility.
    #[must_use]
    pub fn with_validation(mut self, validate: bool) -> Self {
        self.validate_compatibility = validate;
        self
    }

    /// Join multiple files into one.
    pub async fn join<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        inputs: &[P],
        output: Q,
    ) -> Result<()> {
        let _inputs: Vec<_> = inputs.iter().map(std::convert::AsRef::as_ref).collect();
        let _output = output.as_ref();

        if inputs.is_empty() {
            return Err(ConversionError::InvalidInput(
                "No input files provided".to_string(),
            ));
        }

        if self.validate_compatibility {
            self.validate_files(&_inputs)?;
        }

        // Placeholder for actual concatenation
        Ok(())
    }

    /// Join files from a directory matching a pattern.
    pub async fn join_from_directory<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        input_dir: P,
        output: Q,
        pattern: &str,
    ) -> Result<()> {
        let input_dir = input_dir.as_ref();
        let _output = output.as_ref();

        let mut files = Vec::new();
        for entry in std::fs::read_dir(input_dir).map_err(ConversionError::Io)? {
            let entry = entry.map_err(ConversionError::Io)?;
            let path = entry.path();

            if path.is_file() && self.matches_pattern(&path, pattern) {
                files.push(path);
            }
        }

        files.sort();
        self.join(&files, _output).await
    }

    /// Join files listed in a text file.
    pub async fn join_from_list<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        list_file: P,
        output: Q,
    ) -> Result<()> {
        let list_file = list_file.as_ref();
        let content = std::fs::read_to_string(list_file).map_err(ConversionError::Io)?;

        let files: Vec<PathBuf> = content
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| PathBuf::from(line.trim()))
            .collect();

        self.join(&files, output).await
    }

    fn validate_files(&self, files: &[&Path]) -> Result<()> {
        if files.is_empty() {
            return Err(ConversionError::InvalidInput(
                "No files to concatenate".to_string(),
            ));
        }

        for file in files {
            if !file.exists() {
                return Err(ConversionError::InvalidInput(format!(
                    "File not found: {}",
                    file.display()
                )));
            }
        }

        // Placeholder for actual compatibility validation
        Ok(())
    }

    fn matches_pattern(&self, path: &Path, pattern: &str) -> bool {
        if pattern == "*" {
            return true;
        }

        if let Some(ext) = path.extension() {
            if let Some(ext_str) = ext.to_str() {
                if pattern.starts_with("*.") {
                    return ext_str == &pattern[2..];
                }
            }
        }

        false
    }
}

impl Default for FileJoiner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_joiner_creation() {
        let joiner = FileJoiner::new();
        assert!(!joiner.reencode);
        assert!(joiner.validate_compatibility);
    }

    #[test]
    fn test_joiner_settings() {
        let joiner = FileJoiner::new().with_reencode(true).with_validation(false);

        assert!(joiner.reencode);
        assert!(!joiner.validate_compatibility);
    }

    #[test]
    fn test_matches_pattern() {
        let joiner = FileJoiner::new();
        let path = Path::new("test.mp4");

        assert!(joiner.matches_pattern(path, "*"));
        assert!(joiner.matches_pattern(path, "*.mp4"));
        assert!(!joiner.matches_pattern(path, "*.mkv"));
    }
}
