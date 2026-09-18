//! Partial file recovery from truncated files.
//!
//! This module provides functions to recover usable portions from truncated files.

use crate::Result;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

/// Recover playable portion from truncated file.
pub fn recover_truncated_file(input: &Path, output: &Path) -> Result<u64> {
    let mut input_file = File::open(input)?;
    let mut output_file = File::create(output)?;

    // Find last valid frame/packet
    let valid_until = find_last_valid_position(&mut input_file)?;

    // Copy valid portion
    input_file.seek(SeekFrom::Start(0))?;
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut copied = 0u64;

    while copied < valid_until {
        let to_read = ((valid_until - copied) as usize).min(buffer.len());
        let bytes_read = input_file.read(&mut buffer[..to_read])?;

        if bytes_read == 0 {
            break;
        }

        output_file.write_all(&buffer[..bytes_read])?;
        copied += bytes_read as u64;
    }

    Ok(copied)
}

/// Find the last valid position in a truncated file.
fn find_last_valid_position(file: &mut File) -> Result<u64> {
    let size = file.metadata()?.len();

    // Scan backwards to find last valid sync point
    const CHUNK_SIZE: usize = 1024 * 1024;
    let mut last_valid = size;

    let mut pos = size.saturating_sub(CHUNK_SIZE as u64);

    while pos > 0 {
        file.seek(SeekFrom::Start(pos))?;
        let mut buffer = vec![0u8; CHUNK_SIZE];
        let bytes_read = file.read(&mut buffer)?;

        if let Some(offset) = find_last_sync_byte(&buffer[..bytes_read]) {
            last_valid = pos + offset as u64;
            break;
        }

        pos = pos.saturating_sub(CHUNK_SIZE as u64);
    }

    Ok(last_valid)
}

/// Find last sync byte in buffer.
fn find_last_sync_byte(buffer: &[u8]) -> Option<usize> {
    // Look for MPEG sync bytes (0x000001)
    for i in (3..buffer.len()).rev() {
        if buffer[i - 3..i] == [0x00, 0x00, 0x01] {
            return Some(i);
        }
    }
    None
}

/// Extract valid frames from truncated file.
pub fn extract_valid_frames(input: &Path) -> Result<Vec<(u64, Vec<u8>)>> {
    let mut file = File::open(input)?;
    let mut frames = Vec::new();

    const CHUNK_SIZE: usize = 1024 * 1024;
    let mut buffer = vec![0u8; CHUNK_SIZE];
    let mut offset = 0u64;

    while let Ok(bytes_read) = file.read(&mut buffer) {
        if bytes_read == 0 {
            break;
        }

        // Find frames in this chunk
        let chunk = &buffer[..bytes_read];
        let chunk_frames = scan_for_frames(chunk, offset);
        frames.extend(chunk_frames);

        offset += bytes_read as u64;
    }

    Ok(frames)
}

/// Scan buffer for valid frames.
fn scan_for_frames(buffer: &[u8], base_offset: u64) -> Vec<(u64, Vec<u8>)> {
    let mut frames = Vec::new();
    let mut i = 0;

    while i + 4 <= buffer.len() {
        if buffer[i..i + 3] == [0x00, 0x00, 0x01] {
            // Found start code
            let frame_offset = base_offset + i as u64;

            // Find next start code or end of buffer
            let mut j = i + 3;
            while j + 3 <= buffer.len() {
                if buffer[j..j + 3] == [0x00, 0x00, 0x01] {
                    break;
                }
                j += 1;
            }

            let frame_data = buffer[i..j].to_vec();
            frames.push((frame_offset, frame_data));

            i = j;
        } else {
            i += 1;
        }
    }

    frames
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_last_sync_byte() {
        let buffer = vec![0, 0, 1, 0xBA, 0, 0, 0, 0, 0, 0, 1, 0xBE];
        let result = find_last_sync_byte(&buffer);
        assert_eq!(result, Some(11));
    }

    #[test]
    fn test_find_last_sync_byte_none() {
        let buffer = vec![0xFF, 0xFF, 0xFF, 0xFF];
        let result = find_last_sync_byte(&buffer);
        assert_eq!(result, None);
    }

    #[test]
    fn test_scan_for_frames() {
        let buffer = vec![0x00, 0x00, 0x01, 0xBA, 0x00, 0x00, 0x01, 0xBE];
        let frames = scan_for_frames(&buffer, 0);
        assert_eq!(frames.len(), 2);
    }
}
