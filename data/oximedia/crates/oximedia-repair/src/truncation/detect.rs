//! Truncation detection.
//!
//! This module provides functions to detect if a media file is truncated.

use crate::Result;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// Truncation info.
#[derive(Debug)]
pub struct TruncationInfo {
    /// Whether file is truncated.
    pub is_truncated: bool,
    /// Expected file size (if determinable).
    pub expected_size: Option<u64>,
    /// Actual file size.
    pub actual_size: u64,
    /// Estimated missing bytes.
    pub missing_bytes: Option<u64>,
}

/// Detect if file is truncated.
pub fn detect_truncation(path: &Path) -> Result<TruncationInfo> {
    let mut file = File::open(path)?;
    let actual_size = file.metadata()?.len();

    // Check file header for size information
    let expected_size = read_expected_size(&mut file)?;

    let is_truncated = if let Some(expected) = expected_size {
        actual_size < expected
    } else {
        check_truncation_indicators(&mut file)?
    };

    let missing_bytes = expected_size.map(|expected| expected.saturating_sub(actual_size));

    Ok(TruncationInfo {
        is_truncated,
        expected_size,
        actual_size,
        missing_bytes,
    })
}

/// Read expected file size from container header.
fn read_expected_size(file: &mut File) -> Result<Option<u64>> {
    file.seek(SeekFrom::Start(0))?;

    let mut header = [0u8; 16];
    if file.read_exact(&mut header).is_err() {
        return Ok(None);
    }

    // Check for RIFF/AVI
    if &header[0..4] == b"RIFF" {
        let size = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as u64;
        return Ok(Some(size + 8));
    }

    // Check for MP4
    if &header[4..8] == b"ftyp" {
        // MP4 size determination is complex, would need full parsing
        return Ok(None);
    }

    Ok(None)
}

/// Check for indicators of truncation.
fn check_truncation_indicators(file: &mut File) -> Result<bool> {
    let size = file.metadata()?.len();

    if size < 16 {
        return Ok(true);
    }

    // Check if file ends abruptly
    file.seek(SeekFrom::End(-16))?;
    let mut tail = [0u8; 16];
    file.read_exact(&mut tail)?;

    // All zeros at end suggests truncation
    if tail.iter().all(|&b| b == 0) {
        return Ok(true);
    }

    // Incomplete container structure suggests truncation
    if is_incomplete_structure(file)? {
        return Ok(true);
    }

    Ok(false)
}

/// Check if container structure is incomplete.
fn is_incomplete_structure(_file: &mut File) -> Result<bool> {
    // This would require format-specific checking
    // Placeholder for now
    Ok(false)
}

/// Estimate recoverable portion of truncated file.
pub fn estimate_recoverable_portion(path: &Path) -> Result<u64> {
    let file = File::open(path)?;
    let size = file.metadata()?.len();

    // Conservative estimate: assume last 10% may be corrupt
    Ok((size as f64 * 0.9) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncation_info_creation() {
        let info = TruncationInfo {
            is_truncated: true,
            expected_size: Some(1000),
            actual_size: 500,
            missing_bytes: Some(500),
        };

        assert!(info.is_truncated);
        assert_eq!(info.missing_bytes, Some(500));
    }
}
