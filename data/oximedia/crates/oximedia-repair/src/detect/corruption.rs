//! Corruption detection for media files.
//!
//! This module provides functions to detect various types of corruption
//! in media files by examining headers, structure, and data integrity.

use crate::{Issue, IssueType, Result, Severity};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// Detect corruption in a media file.
///
/// This function performs quick corruption checks by examining:
/// - File header validity
/// - Magic number verification
/// - Basic structure integrity
/// - File size consistency
/// Extended issue with confidence score, returned by `detect_corruption_with_confidence`.
pub fn detect_corruption(path: &Path) -> Result<Vec<Issue>> {
    let mut issues = Vec::new();
    let mut file = File::open(path)?;

    // Check file is not empty
    let metadata = file.metadata()?;
    if metadata.len() == 0 {
        issues.push(Issue {
            issue_type: IssueType::CorruptedHeader,
            severity: Severity::Critical,
            description: "File is empty".to_string(),
            location: Some(0),
            fixable: false,
            confidence: 1.0,
        });
        return Ok(issues);
    }

    // Read first 16 bytes for magic number detection
    let mut header = vec![0u8; 16.min(metadata.len() as usize)];
    file.read_exact(&mut header)?;

    // Detect file format and check magic numbers
    let format = detect_format(&header);
    match format {
        Some(FileFormat::MP4) => {
            issues.extend(check_mp4_corruption(&mut file)?);
        }
        Some(FileFormat::Matroska) => {
            issues.extend(check_matroska_corruption(&mut file)?);
        }
        Some(FileFormat::AVI) => {
            issues.extend(check_avi_corruption(&mut file)?);
        }
        Some(FileFormat::MPEG) => {
            issues.extend(check_mpeg_corruption(&mut file)?);
        }
        None => {
            // Unknown format — moderate confidence (file could simply be unsupported)
            issues.push(Issue {
                issue_type: IssueType::CorruptedHeader,
                severity: Severity::Critical,
                description: "Unknown or corrupted file format".to_string(),
                location: Some(0),
                fixable: false,
                confidence: 0.7,
            });
        }
    }

    Ok(issues)
}

/// Supported file formats for corruption detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FileFormat {
    /// MP4/MOV container
    MP4,
    /// Matroska/WebM container
    Matroska,
    /// AVI container
    AVI,
    /// MPEG transport/program stream
    MPEG,
}

/// Detect file format from header bytes.
fn detect_format(header: &[u8]) -> Option<FileFormat> {
    if header.len() < 4 {
        return None;
    }

    // Check MP4/MOV
    if header.len() >= 8 {
        let ftyp = &header[4..8];
        if ftyp == b"ftyp" {
            return Some(FileFormat::MP4);
        }
    }

    // Check Matroska/WebM (EBML header)
    if header.len() >= 4 && header[0..4] == [0x1A, 0x45, 0xDF, 0xA3] {
        return Some(FileFormat::Matroska);
    }

    // Check AVI (RIFF header)
    if header.len() >= 12 && &header[0..4] == b"RIFF" && &header[8..12] == b"AVI " {
        return Some(FileFormat::AVI);
    }

    // Check MPEG
    if header.len() >= 4 && header[0..3] == [0x00, 0x00, 0x01] {
        return Some(FileFormat::MPEG);
    }

    None
}

/// Check MP4 file for corruption.
fn check_mp4_corruption(file: &mut File) -> Result<Vec<Issue>> {
    let mut issues = Vec::new();

    // Seek to start
    file.seek(SeekFrom::Start(0))?;

    // Read first atom
    let mut atom_header = [0u8; 8];
    if file.read_exact(&mut atom_header).is_err() {
        issues.push(Issue {
            issue_type: IssueType::CorruptedHeader,
            severity: Severity::Critical,
            description: "Cannot read MP4 atom header".to_string(),
            location: Some(0),
            fixable: true,
            confidence: 0.95,
        });
        return Ok(issues);
    }

    // Parse atom size
    let atom_size = u32::from_be_bytes([
        atom_header[0],
        atom_header[1],
        atom_header[2],
        atom_header[3],
    ]);

    // Check for invalid atom size
    if atom_size == 0 {
        issues.push(Issue {
            issue_type: IssueType::CorruptedHeader,
            severity: Severity::High,
            description: "MP4 atom has zero size".to_string(),
            location: Some(0),
            fixable: true,
            confidence: 0.85,
        });
    }

    // Check ftyp atom
    let atom_type = &atom_header[4..8];
    if atom_type != b"ftyp" {
        issues.push(Issue {
            issue_type: IssueType::CorruptedHeader,
            severity: Severity::High,
            description: "MP4 file missing ftyp atom".to_string(),
            location: Some(0),
            fixable: true,
            confidence: 0.8,
        });
    }

    Ok(issues)
}

/// Check Matroska file for corruption.
fn check_matroska_corruption(file: &mut File) -> Result<Vec<Issue>> {
    let mut issues = Vec::new();

    // Seek to start
    file.seek(SeekFrom::Start(0))?;

    // Read EBML header
    let mut ebml_header = [0u8; 4];
    if file.read_exact(&mut ebml_header).is_err() {
        issues.push(Issue {
            issue_type: IssueType::CorruptedHeader,
            severity: Severity::Critical,
            description: "Cannot read Matroska EBML header".to_string(),
            location: Some(0),
            fixable: true,
            confidence: 0.95,
        });
        return Ok(issues);
    }

    // Verify EBML signature
    if ebml_header != [0x1A, 0x45, 0xDF, 0xA3] {
        issues.push(Issue {
            issue_type: IssueType::CorruptedHeader,
            severity: Severity::Critical,
            description: "Invalid Matroska EBML signature".to_string(),
            location: Some(0),
            fixable: true,
            confidence: 0.95,
        });
    }

    Ok(issues)
}

/// Check AVI file for corruption.
fn check_avi_corruption(file: &mut File) -> Result<Vec<Issue>> {
    let mut issues = Vec::new();

    // Seek to start
    file.seek(SeekFrom::Start(0))?;

    // Read RIFF header
    let mut riff_header = [0u8; 12];
    if file.read_exact(&mut riff_header).is_err() {
        issues.push(Issue {
            issue_type: IssueType::CorruptedHeader,
            severity: Severity::Critical,
            description: "Cannot read AVI RIFF header".to_string(),
            location: Some(0),
            fixable: true,
            confidence: 0.95,
        });
        return Ok(issues);
    }

    // Verify RIFF signature
    if &riff_header[0..4] != b"RIFF" {
        issues.push(Issue {
            issue_type: IssueType::CorruptedHeader,
            severity: Severity::Critical,
            description: "Invalid AVI RIFF signature".to_string(),
            location: Some(0),
            fixable: true,
            confidence: 0.95,
        });
    }

    // Verify AVI type
    if &riff_header[8..12] != b"AVI " {
        issues.push(Issue {
            issue_type: IssueType::CorruptedHeader,
            severity: Severity::High,
            description: "Invalid AVI type marker".to_string(),
            location: Some(8),
            fixable: true,
            confidence: 0.90,
        });
    }

    // Check file size in header
    let stated_size = u32::from_le_bytes([
        riff_header[4],
        riff_header[5],
        riff_header[6],
        riff_header[7],
    ]) as u64
        + 8;

    let actual_size = file.metadata()?.len();
    if stated_size != actual_size {
        let severity = if actual_size < stated_size {
            Severity::High
        } else {
            Severity::Medium
        };
        // Confidence depends on how large the discrepancy is
        let ratio = if stated_size > 0 {
            (stated_size.abs_diff(actual_size) as f64) / (stated_size as f64)
        } else {
            1.0
        };
        let confidence = (ratio * 2.0).min(1.0).max(0.5);

        issues.push(Issue {
            issue_type: IssueType::Truncated,
            severity,
            description: format!(
                "AVI file size mismatch: header says {}, actual {}",
                stated_size, actual_size
            ),
            location: Some(4),
            fixable: true,
            confidence,
        });
    }

    Ok(issues)
}

/// Check MPEG file for corruption.
fn check_mpeg_corruption(file: &mut File) -> Result<Vec<Issue>> {
    let mut issues = Vec::new();

    // Seek to start
    file.seek(SeekFrom::Start(0))?;

    // Read start code
    let mut start_code = [0u8; 4];
    if file.read_exact(&mut start_code).is_err() {
        issues.push(Issue {
            issue_type: IssueType::CorruptedHeader,
            severity: Severity::Critical,
            description: "Cannot read MPEG start code".to_string(),
            location: Some(0),
            fixable: false,
            confidence: 0.95,
        });
        return Ok(issues);
    }

    // Verify start code prefix
    if start_code[0..3] != [0x00, 0x00, 0x01] {
        issues.push(Issue {
            issue_type: IssueType::CorruptedHeader,
            severity: Severity::Critical,
            description: "Invalid MPEG start code prefix".to_string(),
            location: Some(0),
            fixable: false,
            confidence: 0.90,
        });
    }

    Ok(issues)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_format_mp4() {
        let header = b"\x00\x00\x00\x20ftypmp42";
        assert_eq!(detect_format(header), Some(FileFormat::MP4));
    }

    #[test]
    fn test_detect_format_matroska() {
        let header = b"\x1A\x45\xDF\xA3\x00\x00\x00\x00";
        assert_eq!(detect_format(header), Some(FileFormat::Matroska));
    }

    #[test]
    fn test_detect_format_avi() {
        let header = b"RIFF\x00\x00\x00\x00AVI \x00\x00\x00\x00";
        assert_eq!(detect_format(header), Some(FileFormat::AVI));
    }

    #[test]
    fn test_detect_format_mpeg() {
        let header = b"\x00\x00\x01\xBA\x00\x00\x00\x00";
        assert_eq!(detect_format(header), Some(FileFormat::MPEG));
    }

    #[test]
    fn test_detect_format_unknown() {
        let header = b"\xFF\xFF\xFF\xFF\x00\x00\x00\x00";
        assert_eq!(detect_format(header), None);
    }
}
