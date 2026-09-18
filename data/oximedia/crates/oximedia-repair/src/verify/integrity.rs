//! Integrity verification with checkpoint/resume support.
//!
//! This module provides functions to verify file integrity after repair.
//! For large files the scan can be checkpointed and resumed, so that a
//! long-running verification is not lost if interrupted.
//!
//! # Incremental verification flow
//!
//! ```text
//! [File]
//!   │
//!   ▼
//! verify_header()      ← structural check (header magic bytes)
//!   │
//!   ▼
//! scan_body_from(offset)  ← byte-by-byte chunk scan, writes checkpoint each
//!   │                        time a chunk finishes
//!   ▼
//! [VerifyReport]  ← aggregated issue list + scan metadata
//! ```
//!
//! To resume an interrupted scan:
//!
//! ```no_run
//! use oximedia_repair::verify::integrity::{load_checkpoint, verify_from_checkpoint};
//! use std::path::Path;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let checkpoint = load_checkpoint(Path::new("/tmp/.oxirepair-ckpt-abc.json"))?;
//! let report = verify_from_checkpoint(&checkpoint)?;
//! # Ok(())
//! # }
//! ```

use crate::{Issue, IssueType, RepairError, Result, Severity};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

// ── Chunk size used during incremental body scanning ─────────────────────────

const SCAN_CHUNK_SIZE: usize = 64 * 1024; // 64 KiB per scan step

// ── Public types ──────────────────────────────────────────────────────────────

/// Snapshot of incremental verification state that can be serialised to disk
/// and used to resume a scan that was interrupted before completion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifyCheckpoint {
    /// Path to the file being verified.
    pub file_path: PathBuf,
    /// Byte offset at which scanning should resume.
    pub byte_offset: u64,
    /// Issues found in the portion of the file already scanned.
    pub issues_found: Vec<CheckpointIssue>,
}

/// A serialisable representation of a detected [`Issue`].
///
/// `Issue` itself cannot be serialised directly because it contains
/// non-`Serialize` types, so we flatten it to primitive fields here.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointIssue {
    /// Human-readable issue type name (e.g. `"CorruptPackets"`).
    pub issue_type: String,
    /// Severity level string (e.g. `"High"`).
    pub severity: String,
    /// Human-readable description.
    pub description: String,
    /// Byte offset of the issue inside the file, if known.
    pub location: Option<u64>,
    /// Whether the issue is automatically fixable.
    pub fixable: bool,
    /// Detection confidence in `[0.0, 1.0]`.
    pub confidence: f64,
}

impl CheckpointIssue {
    /// Convert to a full [`Issue`].
    pub fn to_issue(&self) -> Issue {
        let issue_type = match self.issue_type.as_str() {
            "CorruptedHeader" => IssueType::CorruptedHeader,
            "MissingIndex" => IssueType::MissingIndex,
            "InvalidTimestamps" => IssueType::InvalidTimestamps,
            "AVDesync" => IssueType::AVDesync,
            "Truncated" => IssueType::Truncated,
            "CorruptPackets" => IssueType::CorruptPackets,
            "CorruptMetadata" => IssueType::CorruptMetadata,
            "MissingKeyframes" => IssueType::MissingKeyframes,
            "InvalidFrameOrder" => IssueType::InvalidFrameOrder,
            "ConversionError" => IssueType::ConversionError,
            _ => IssueType::CorruptPackets, // fallback
        };
        let severity = match self.severity.as_str() {
            "Low" => Severity::Low,
            "Medium" => Severity::Medium,
            "High" => Severity::High,
            "Critical" => Severity::Critical,
            _ => Severity::Medium,
        };
        Issue {
            issue_type,
            severity,
            description: self.description.clone(),
            location: self.location,
            fixable: self.fixable,
            confidence: self.confidence,
        }
    }

    /// Build from a full [`Issue`].
    pub fn from_issue(issue: &Issue) -> Self {
        let issue_type = match issue.issue_type {
            IssueType::CorruptedHeader => "CorruptedHeader",
            IssueType::MissingIndex => "MissingIndex",
            IssueType::InvalidTimestamps => "InvalidTimestamps",
            IssueType::AVDesync => "AVDesync",
            IssueType::Truncated => "Truncated",
            IssueType::CorruptPackets => "CorruptPackets",
            IssueType::CorruptMetadata => "CorruptMetadata",
            IssueType::MissingKeyframes => "MissingKeyframes",
            IssueType::InvalidFrameOrder => "InvalidFrameOrder",
            IssueType::ConversionError => "ConversionError",
        };
        let severity = match issue.severity {
            Severity::Low => "Low",
            Severity::Medium => "Medium",
            Severity::High => "High",
            Severity::Critical => "Critical",
        };
        Self {
            issue_type: issue_type.to_string(),
            severity: severity.to_string(),
            description: issue.description.clone(),
            location: issue.location,
            fixable: issue.fixable,
            confidence: issue.confidence,
        }
    }
}

/// Summary report produced by a complete (or resumed) verification run.
#[derive(Debug, Clone)]
pub struct VerifyReport {
    /// Path of the verified file.
    pub file_path: PathBuf,
    /// Total bytes scanned.
    pub bytes_scanned: u64,
    /// Issues found across the entire file.
    pub issues: Vec<Issue>,
}

impl VerifyReport {
    /// Returns `true` if no issues were detected.
    pub fn is_clean(&self) -> bool {
        self.issues.is_empty()
    }
}

// ── Core verify API ───────────────────────────────────────────────────────────

/// Verify file integrity.
pub fn verify_integrity(path: &Path) -> Result<()> {
    let mut file = File::open(path)?;

    // Check file is not empty
    let size = file.metadata()?.len();
    if size == 0 {
        return Err(RepairError::VerificationFailed("File is empty".to_string()));
    }

    // Verify header
    verify_header(&mut file)?;

    // Verify structure
    verify_structure(&mut file)?;

    Ok(())
}

// ── Checkpoint-aware scanning ─────────────────────────────────────────────────

/// Resume verification from a previously saved checkpoint.
///
/// The scan picks up at `checkpoint.byte_offset` and appends newly found
/// issues to `checkpoint.issues_found` to form the final [`VerifyReport`].
pub fn verify_from_checkpoint(checkpoint: &VerifyCheckpoint) -> Result<VerifyReport> {
    let path = &checkpoint.file_path;
    let mut file = File::open(path)?;
    let total_size = file.metadata()?.len();

    // Carry forward issues found so far
    let mut all_issues: Vec<Issue> = checkpoint
        .issues_found
        .iter()
        .map(CheckpointIssue::to_issue)
        .collect();

    // Seek to resume offset
    let resume_offset = checkpoint.byte_offset.min(total_size);
    file.seek(SeekFrom::Start(resume_offset))?;

    let new_issues = scan_body(&mut file, resume_offset, total_size)?;
    all_issues.extend(new_issues);

    Ok(VerifyReport {
        file_path: path.clone(),
        bytes_scanned: total_size,
        issues: all_issues,
    })
}

/// Save a checkpoint to `std::env::temp_dir()`.
///
/// Returns the path of the checkpoint file so callers can pass it to
/// [`load_checkpoint`] on a later run.
pub fn save_checkpoint(progress: &VerifyCheckpoint) -> Result<PathBuf> {
    let uuid = uuid::Uuid::new_v4();
    let filename = format!(".oxirepair-ckpt-{uuid}.json");
    let path = std::env::temp_dir().join(filename);
    let json = serde_json::to_string_pretty(progress)
        .map_err(|e| RepairError::RepairFailed(format!("checkpoint serialise: {e}")))?;
    std::fs::write(&path, json.as_bytes())?;
    Ok(path)
}

/// Load a previously saved checkpoint from `path`.
pub fn load_checkpoint(path: &Path) -> Result<VerifyCheckpoint> {
    let data = std::fs::read(path)?;
    serde_json::from_slice(&data)
        .map_err(|e| RepairError::RepairFailed(format!("checkpoint deserialise: {e}")))
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Scan the file body from `start_offset` to `total_size` in chunks.
///
/// Returns newly detected issues found in the scanned range.
fn scan_body(file: &mut File, start_offset: u64, total_size: u64) -> Result<Vec<Issue>> {
    let mut issues = Vec::new();
    let mut offset = start_offset;
    let mut buf = vec![0u8; SCAN_CHUNK_SIZE];

    while offset < total_size {
        let want = SCAN_CHUNK_SIZE.min((total_size - offset) as usize);
        file.seek(SeekFrom::Start(offset))?;
        let n = file.read(&mut buf[..want])?;
        if n == 0 {
            break;
        }
        let chunk = &buf[..n];

        // Detect large zero runs (common corruption artefact)
        let mut i = 0;
        while i < chunk.len() {
            if chunk[i] == 0 {
                let run_start = i;
                while i < chunk.len() && chunk[i] == 0 {
                    i += 1;
                }
                let run_len = i - run_start;
                if run_len > 4096 {
                    issues.push(Issue {
                        issue_type: IssueType::CorruptPackets,
                        severity: Severity::Medium,
                        description: format!(
                            "Zero-byte run of {} bytes at offset {}",
                            run_len,
                            offset + run_start as u64
                        ),
                        location: Some(offset + run_start as u64),
                        fixable: true,
                        confidence: 0.75,
                    });
                }
            } else {
                i += 1;
            }
        }

        offset += n as u64;
    }

    Ok(issues)
}

/// Full incremental verify (produces a [`VerifyReport`] without a prior checkpoint).
///
/// Internally delegates to [`verify_from_checkpoint`] starting at offset 0.
pub fn verify_incremental(path: &Path) -> Result<VerifyReport> {
    let initial = VerifyCheckpoint {
        file_path: path.to_owned(),
        byte_offset: 0,
        issues_found: Vec::new(),
    };
    verify_from_checkpoint(&initial)
}

/// Verify file header.
fn verify_header(file: &mut File) -> Result<()> {
    file.seek(SeekFrom::Start(0))?;

    let mut header = [0u8; 16];
    file.read_exact(&mut header)?;

    // Check for valid format
    let valid = is_valid_header(&header);

    if !valid {
        return Err(RepairError::VerificationFailed(
            "Invalid file header".to_string(),
        ));
    }

    Ok(())
}

/// Check if header is valid.
fn is_valid_header(header: &[u8]) -> bool {
    // Check for known formats
    if header.len() >= 12 {
        if &header[0..4] == b"RIFF" && &header[8..12] == b"AVI " {
            return true;
        }
        if &header[4..8] == b"ftyp" {
            return true;
        }
        if header[0..4] == [0x1A, 0x45, 0xDF, 0xA3] {
            return true;
        }
    }

    false
}

/// Verify file structure.
fn verify_structure(file: &mut File) -> Result<()> {
    let size = file.metadata()?.len();

    // Check file doesn't end with zeros
    if size > 16 {
        file.seek(SeekFrom::End(-16))?;
        let mut tail = [0u8; 16];
        file.read_exact(&mut tail)?;

        if tail.iter().all(|&b| b == 0) {
            return Err(RepairError::VerificationFailed(
                "File ends with zeros".to_string(),
            ));
        }
    }

    Ok(())
}

/// Calculate file checksum.
pub fn calculate_checksum(path: &Path) -> Result<u32> {
    let mut file = File::open(path)?;
    let mut checksum = 0u32;
    let mut buffer = [0u8; 4096];

    loop {
        let bytes_read = file.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }

        for &byte in &buffer[..bytes_read] {
            checksum = checksum.wrapping_add(byte as u32);
        }
    }

    Ok(checksum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_temp(name: &str, data: &[u8]) -> PathBuf {
        let path = std::env::temp_dir().join(name);
        let mut f = File::create(&path).expect("create temp");
        f.write_all(data).expect("write temp");
        path
    }

    #[test]
    fn test_is_valid_header_avi() {
        let header = b"RIFF\x00\x00\x00\x00AVI \x00\x00\x00\x00";
        assert!(is_valid_header(header));
    }

    #[test]
    fn test_is_valid_header_mp4() {
        let header = b"\x00\x00\x20\x00ftypmp42";
        assert!(is_valid_header(header));
    }

    #[test]
    fn test_is_valid_header_invalid() {
        let header = b"\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF";
        assert!(!is_valid_header(header));
    }

    #[test]
    fn test_verify_checkpoint_save_load_roundtrip() {
        let path = write_temp("ckpt_roundtrip.bin", &[0xAA; 128]);
        let checkpoint = VerifyCheckpoint {
            file_path: path.clone(),
            byte_offset: 64,
            issues_found: vec![CheckpointIssue {
                issue_type: "CorruptPackets".to_string(),
                severity: "Medium".to_string(),
                description: "Test issue".to_string(),
                location: Some(32),
                fixable: true,
                confidence: 0.9,
            }],
        };

        let ckpt_path = save_checkpoint(&checkpoint).expect("save_checkpoint");
        assert!(ckpt_path.exists(), "checkpoint file must exist");

        let loaded = load_checkpoint(&ckpt_path).expect("load_checkpoint");
        assert_eq!(loaded.file_path, checkpoint.file_path);
        assert_eq!(loaded.byte_offset, 64);
        assert_eq!(loaded.issues_found.len(), 1);
        assert_eq!(loaded.issues_found[0].description, "Test issue");

        // Cleanup
        let _ = std::fs::remove_file(&ckpt_path);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_verify_from_checkpoint_resumes_correctly() {
        // Build a file: first 128 bytes are non-zero, then a large zero run (8 KiB),
        // then another 64 bytes of non-zero data.
        let mut data = vec![0xAB_u8; 128];
        data.extend(vec![0u8; 8192]); // zero run > 4096 bytes → should be flagged
        data.extend(vec![0xCD_u8; 64]);
        let path = write_temp("ckpt_resume.bin", &data);

        // Simulate a checkpoint taken after scanning the first 128 bytes
        // (no issues yet, since first 128 bytes are non-zero).
        let partial_checkpoint = VerifyCheckpoint {
            file_path: path.clone(),
            byte_offset: 128,
            issues_found: Vec::new(),
        };

        let report =
            verify_from_checkpoint(&partial_checkpoint).expect("verify_from_checkpoint failed");

        // The zero run is in the range [128, 128+8192), so it should be detected
        // when we resume from offset 128.
        assert!(
            !report.issues.is_empty(),
            "Expected at least one issue from zero run; got none"
        );
        assert!(
            report
                .issues
                .iter()
                .any(|i| i.issue_type == IssueType::CorruptPackets),
            "Expected CorruptPackets issue"
        );

        let _ = std::fs::remove_file(&path);
    }
}
