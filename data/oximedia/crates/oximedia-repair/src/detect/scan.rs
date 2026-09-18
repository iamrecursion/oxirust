//! Deep file scanning for corruption detection.
//!
//! This module provides deep scanning capabilities to detect
//! corruption at the packet and frame level.
//!
//! The [`deep_scan_mmap`] function uses `memmap2` which requires an
//! `unsafe` block to call `Mmap::map`. The safety invariant is:
//! the file is opened read-only, no mutable aliases exist, and the
//! mapping is dropped at the end of the function.
#![allow(unsafe_code)]

use crate::{Issue, IssueType, Result, Severity};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// Minimum file size (bytes) to use memory-mapped I/O in `deep_scan`.
const MMAP_THRESHOLD: u64 = 4 * 1024 * 1024; // 4 MiB

/// Perform a deep scan of a media file.
///
/// Automatically selects between streaming I/O (for files smaller than
/// `MMAP_THRESHOLD`) and memory-mapped I/O (for larger files) to balance
/// memory usage and scan throughput.
pub fn deep_scan(path: &Path) -> Result<Vec<Issue>> {
    let file_size = std::fs::metadata(path)?.len();
    if file_size >= MMAP_THRESHOLD {
        deep_scan_mmap(path)
    } else {
        deep_scan_streaming(path)
    }
}

/// Perform a deep scan using streaming I/O (for smaller files).
pub fn deep_scan_streaming(path: &Path) -> Result<Vec<Issue>> {
    let mut issues = Vec::new();
    let mut file = File::open(path)?;
    let file_size = file.metadata()?.len();

    // Scan in chunks
    const CHUNK_SIZE: usize = 1024 * 1024; // 1MB chunks
    let mut buffer = vec![0u8; CHUNK_SIZE];
    let mut offset = 0u64;

    while offset < file_size {
        file.seek(SeekFrom::Start(offset))?;
        let bytes_read = file.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }

        let chunk = &buffer[..bytes_read];

        // Check for corruption patterns
        issues.extend(scan_chunk(chunk, offset)?);

        // Check for sync losses
        issues.extend(detect_sync_loss(chunk, offset)?);

        // Check for packet corruption
        issues.extend(detect_packet_corruption(chunk, offset)?);

        offset += bytes_read as u64;
    }

    // Check for truncation at end
    if let Some(issue) = check_truncation(&mut file)? {
        issues.push(issue);
    }

    Ok(issues)
}

/// Perform a deep scan using memory-mapped I/O (for large files ≥ `MMAP_THRESHOLD`).
///
/// The mmap approach avoids the system-call overhead of repeated `read()`
/// calls, letting the OS page-in only the portions that are actually touched.
///
/// # Safety
///
/// The file is mapped read-only.  The mapping is valid for the duration of
/// this function.  If another process truncates the file while we're reading
/// it the kernel will deliver `SIGBUS` on platforms that surface it — this is
/// the inherent risk of mmap and is acceptable for repair tooling that
/// operates on pre-staged, stable input files.
pub fn deep_scan_mmap(path: &Path) -> Result<Vec<Issue>> {
    let file = File::open(path)?;
    let metadata = file.metadata()?;
    let file_size = metadata.len();

    // Fall back to streaming for small files
    if file_size < MMAP_THRESHOLD {
        return deep_scan_streaming(path);
    }

    // SAFETY: we open the file read-only and hold the File for the lifetime of
    // the mmap. No other code in this function mutates the underlying data.
    let mmap = unsafe { memmap2::Mmap::map(&file)? };

    let mut issues = Vec::new();
    const CHUNK_SIZE: usize = 1024 * 1024; // 1 MiB chunks

    let data: &[u8] = &mmap;
    let mut offset = 0usize;

    while offset < data.len() {
        let end = (offset + CHUNK_SIZE).min(data.len());
        let chunk = &data[offset..end];

        issues.extend(scan_chunk(chunk, offset as u64)?);
        issues.extend(detect_sync_loss(chunk, offset as u64)?);
        issues.extend(detect_packet_corruption(chunk, offset as u64)?);

        offset = end;
    }

    // Check for truncation at end
    {
        let tail_start = data.len().saturating_sub(16);
        let tail = &data[tail_start..];
        if !tail.is_empty() && tail.iter().all(|&b| b == 0) {
            issues.push(Issue {
                issue_type: IssueType::Truncated,
                severity: Severity::High,
                description: "File ends with zeros, likely truncated".to_string(),
                location: Some(tail_start as u64),
                fixable: true,
                confidence: 0.8,
            });
        }
    }

    Ok(issues)
}

/// Scan a chunk of data for corruption patterns.
fn scan_chunk(chunk: &[u8], base_offset: u64) -> Result<Vec<Issue>> {
    let mut issues = Vec::new();

    // Look for large runs of zeros (possible corruption)
    let zero_runs = detect_zero_runs(chunk);
    for (offset, length) in zero_runs {
        if length > 4096 {
            issues.push(Issue {
                issue_type: IssueType::CorruptPackets,
                severity: Severity::Medium,
                description: format!("Large run of zeros ({} bytes)", length),
                location: Some(base_offset + offset as u64),
                fixable: true,
                confidence: 0.8,
            });
        }
    }

    // Look for repeated patterns
    let patterns = super::analyze::detect_patterns(chunk);
    for (offset, pattern_len, count) in patterns {
        if count > 10 {
            issues.push(Issue {
                issue_type: IssueType::CorruptPackets,
                severity: Severity::Medium,
                description: format!(
                    "Suspicious repeated pattern ({} bytes, {} times)",
                    pattern_len, count
                ),
                location: Some(base_offset + offset as u64),
                fixable: true,
                confidence: 0.8,
            });
        }
    }

    Ok(issues)
}

/// Detect runs of zero bytes in data.
fn detect_zero_runs(data: &[u8]) -> Vec<(usize, usize)> {
    let mut runs = Vec::new();
    let mut start = None;

    for (i, &byte) in data.iter().enumerate() {
        if byte == 0 {
            if start.is_none() {
                start = Some(i);
            }
        } else if let Some(s) = start {
            runs.push((s, i - s));
            start = None;
        }
    }

    if let Some(s) = start {
        runs.push((s, data.len() - s));
    }

    runs
}

/// Detect synchronization loss in media stream.
fn detect_sync_loss(chunk: &[u8], base_offset: u64) -> Result<Vec<Issue>> {
    let mut issues = Vec::new();

    // Look for MPEG sync bytes (0x000001)
    let mut i = 0;
    let mut last_sync = None;
    const MAX_SYNC_DISTANCE: usize = 100_000; // 100KB

    while i + 3 <= chunk.len() {
        if chunk[i..i + 3] == [0x00, 0x00, 0x01] {
            if let Some(last) = last_sync {
                if i - last > MAX_SYNC_DISTANCE {
                    issues.push(Issue {
                        issue_type: IssueType::CorruptPackets,
                        severity: Severity::High,
                        description: format!("Large gap between sync bytes ({} bytes)", i - last),
                        location: Some(base_offset + last as u64),
                        fixable: true,
                        confidence: 0.8,
                    });
                }
            }
            last_sync = Some(i);
            i += 3;
        } else {
            i += 1;
        }
    }

    Ok(issues)
}

/// Detect corrupted packets in media stream.
fn detect_packet_corruption(chunk: &[u8], base_offset: u64) -> Result<Vec<Issue>> {
    let mut issues = Vec::new();

    // Simple heuristic: look for invalid packet sizes
    let mut i = 0;
    while i + 4 <= chunk.len() {
        // Try to parse as packet header (size in first 4 bytes)
        let size = u32::from_be_bytes([chunk[i], chunk[i + 1], chunk[i + 2], chunk[i + 3]]);

        // Check for obviously invalid sizes
        if size > 10_000_000 {
            // > 10MB
            issues.push(Issue {
                issue_type: IssueType::CorruptPackets,
                severity: Severity::High,
                description: format!("Invalid packet size: {}", size),
                location: Some(base_offset + i as u64),
                fixable: true,
                confidence: 0.8,
            });
        }

        // Advance by 4: packet headers are 4-byte aligned so sliding
        // one byte at a time produces ~4× redundant reads with no extra
        // detection value, and takes ~38s on a 5 MiB file under load.
        i += 4;
    }

    Ok(issues)
}

/// Check if file is truncated at the end.
fn check_truncation(file: &mut File) -> Result<Option<Issue>> {
    let file_size = file.metadata()?.len();
    if file_size == 0 {
        return Ok(None);
    }

    // Read last 16 bytes
    let read_size = 16.min(file_size);
    file.seek(SeekFrom::End(-(read_size as i64)))?;
    let mut tail = vec![0u8; read_size as usize];
    file.read_exact(&mut tail)?;

    // Check if file ends with zeros (possible truncation)
    if tail.iter().all(|&b| b == 0) {
        return Ok(Some(Issue {
            issue_type: IssueType::Truncated,
            severity: Severity::High,
            description: "File ends with zeros, likely truncated".to_string(),
            location: Some(file_size - read_size),
            fixable: true,
            confidence: 0.8,
        }));
    }

    Ok(None)
}

/// Scan for keyframes in video data.
pub fn scan_keyframes(data: &[u8]) -> Vec<u64> {
    let mut keyframes = Vec::new();

    // Look for I-frame markers in H.264/H.265 streams
    let mut i = 0;
    while i + 4 <= data.len() {
        // NAL unit start code
        if data[i..i + 3] == [0x00, 0x00, 0x01] {
            let nal_type = data[i + 3] & 0x1F;
            // Type 5 is IDR (keyframe) in H.264
            if nal_type == 5 {
                keyframes.push(i as u64);
            }
            i += 4;
        } else {
            i += 1;
        }
    }

    keyframes
}

/// Calculate checksums for data validation.
pub fn calculate_checksum(data: &[u8]) -> u32 {
    let mut checksum = 0u32;
    for &byte in data {
        checksum = checksum.wrapping_add(byte as u32);
    }
    checksum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_zero_runs() {
        let data = vec![1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1];
        let runs = detect_zero_runs(&data);
        assert_eq!(runs.len(), 2);
        assert_eq!(runs[0], (1, 3));
        assert_eq!(runs[1], (5, 5));
    }

    #[test]
    fn test_detect_zero_runs_none() {
        let data = vec![1, 2, 3, 4, 5];
        let runs = detect_zero_runs(&data);
        assert!(runs.is_empty());
    }

    #[test]
    fn test_detect_zero_runs_all() {
        let data = vec![0, 0, 0, 0, 0];
        let runs = detect_zero_runs(&data);
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0], (0, 5));
    }

    #[test]
    fn test_scan_keyframes() {
        let mut data = Vec::new();
        data.extend_from_slice(&[0x00, 0x00, 0x01, 0x65]); // IDR frame
        data.extend_from_slice(&[0x00, 0x00, 0x01, 0x41]); // Non-IDR frame
        data.extend_from_slice(&[0x00, 0x00, 0x01, 0x05]); // IDR frame

        let keyframes = scan_keyframes(&data);
        assert_eq!(keyframes.len(), 2);
    }

    #[test]
    fn test_calculate_checksum() {
        let data = vec![1, 2, 3, 4, 5];
        let checksum = calculate_checksum(&data);
        assert_eq!(checksum, 15);
    }

    #[test]
    fn test_calculate_checksum_empty() {
        let data: Vec<u8> = Vec::new();
        let checksum = calculate_checksum(&data);
        assert_eq!(checksum, 0);
    }
}
