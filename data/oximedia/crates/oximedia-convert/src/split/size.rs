// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Size-based file splitting.

use crate::{ConversionError, Result};
use std::path::{Path, PathBuf};

/// Splitter for dividing files by size.
#[derive(Debug, Clone)]
pub struct SizeSplitter {
    max_size_bytes: u64,
    copy_streams: bool,
}

impl SizeSplitter {
    /// Create a new size splitter.
    #[must_use]
    pub fn new(max_size_bytes: u64) -> Self {
        Self {
            max_size_bytes,
            copy_streams: true,
        }
    }

    /// Set whether to copy streams without re-encoding.
    #[must_use]
    pub fn with_copy_streams(mut self, copy: bool) -> Self {
        self.copy_streams = copy;
        self
    }

    /// Split a file into size-based segments.
    pub async fn split<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        input: P,
        output_dir: Q,
    ) -> Result<Vec<PathBuf>> {
        let _input = input.as_ref();
        let _output_dir = output_dir.as_ref();

        std::fs::create_dir_all(_output_dir).map_err(ConversionError::Io)?;

        // Placeholder for actual splitting
        Ok(Vec::new())
    }

    /// Calculate the number of segments needed.
    pub fn calculate_segments<P: AsRef<Path>>(&self, input: P) -> Result<usize> {
        let input = input.as_ref();
        let metadata = std::fs::metadata(input).map_err(ConversionError::Io)?;

        let file_size = metadata.len();
        let segments = file_size.div_ceil(self.max_size_bytes);

        Ok(segments as usize)
    }

    /// Create a splitter for 10 MB segments.
    #[must_use]
    pub fn ten_mb() -> Self {
        Self::new(10 * 1024 * 1024)
    }

    /// Create a splitter for 25 MB segments (email-friendly).
    #[must_use]
    pub fn twenty_five_mb() -> Self {
        Self::new(25 * 1024 * 1024)
    }

    /// Create a splitter for 100 MB segments.
    #[must_use]
    pub fn hundred_mb() -> Self {
        Self::new(100 * 1024 * 1024)
    }

    /// Create a splitter for 1 GB segments.
    #[must_use]
    pub fn one_gb() -> Self {
        Self::new(1024 * 1024 * 1024)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_splitter_creation() {
        let splitter = SizeSplitter::new(1024 * 1024);
        assert_eq!(splitter.max_size_bytes, 1024 * 1024);
        assert!(splitter.copy_streams);
    }

    #[test]
    fn test_splitter_settings() {
        let splitter = SizeSplitter::new(5 * 1024 * 1024).with_copy_streams(false);

        assert_eq!(splitter.max_size_bytes, 5 * 1024 * 1024);
        assert!(!splitter.copy_streams);
    }

    #[test]
    fn test_presets() {
        let splitter = SizeSplitter::ten_mb();
        assert_eq!(splitter.max_size_bytes, 10 * 1024 * 1024);

        let splitter = SizeSplitter::twenty_five_mb();
        assert_eq!(splitter.max_size_bytes, 25 * 1024 * 1024);

        let splitter = SizeSplitter::one_gb();
        assert_eq!(splitter.max_size_bytes, 1024 * 1024 * 1024);
    }
}
