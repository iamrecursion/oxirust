//! Extract playable portions.
//!
//! This module provides functions to extract playable portions from damaged files.

use crate::Result;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

/// Extract playable portion from damaged file.
pub fn extract_playable_portion(input: &Path, output: &Path) -> Result<u64> {
    let mut input_file = File::open(input)?;
    let mut output_file = File::create(output)?;

    // Find playable ranges
    let ranges = find_playable_ranges(&mut input_file)?;

    let mut total_bytes = 0u64;

    // Extract each playable range
    for (start, end) in ranges {
        input_file.seek(SeekFrom::Start(start))?;

        let mut buffer = vec![0u8; (end - start) as usize];
        input_file.read_exact(&mut buffer)?;
        output_file.write_all(&buffer)?;

        total_bytes += end - start;
    }

    Ok(total_bytes)
}

/// Find playable ranges in a file.
fn find_playable_ranges(file: &mut File) -> Result<Vec<(u64, u64)>> {
    let mut ranges = Vec::new();
    let file_size = file.metadata()?.len();

    const CHUNK_SIZE: usize = 1024 * 1024;
    let mut buffer = vec![0u8; CHUNK_SIZE];
    let mut offset = 0u64;

    let mut range_start = None;

    while offset < file_size {
        file.seek(SeekFrom::Start(offset))?;
        let bytes_read = file.read(&mut buffer)?;

        if bytes_read == 0 {
            break;
        }

        let chunk = &buffer[..bytes_read];

        if is_playable_chunk(chunk) {
            if range_start.is_none() {
                range_start = Some(offset);
            }
        } else if let Some(start) = range_start {
            ranges.push((start, offset));
            range_start = None;
        }

        offset += bytes_read as u64;
    }

    // Close final range
    if let Some(start) = range_start {
        ranges.push((start, offset));
    }

    Ok(ranges)
}

/// Check if a chunk appears playable.
fn is_playable_chunk(chunk: &[u8]) -> bool {
    // Simple heuristic: not all zeros, has some variation
    if chunk.is_empty() || chunk.iter().all(|&b| b == 0) {
        return false;
    }

    // Check for reasonable entropy
    let entropy = super::super::detect::analyze::calculate_entropy(chunk);
    entropy > 1.0 && entropy < 7.9
}

/// Split file into segments at corruption boundaries.
pub fn split_at_corruption(input: &Path, output_dir: &Path) -> Result<Vec<std::path::PathBuf>> {
    let mut input_file = File::open(input)?;
    let ranges = find_playable_ranges(&mut input_file)?;

    std::fs::create_dir_all(output_dir)?;

    let mut output_files = Vec::new();

    for (i, (start, end)) in ranges.iter().enumerate() {
        let output_path = output_dir.join(format!("segment_{:04}.dat", i));
        let mut output_file = File::create(&output_path)?;

        input_file.seek(SeekFrom::Start(*start))?;
        let mut buffer = vec![0u8; (*end - *start) as usize];
        input_file.read_exact(&mut buffer)?;
        output_file.write_all(&buffer)?;

        output_files.push(output_path);
    }

    Ok(output_files)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_playable_chunk() {
        let valid_chunk = vec![1, 2, 3, 4, 5, 6, 7, 8];
        assert!(is_playable_chunk(&valid_chunk));

        let zero_chunk = vec![0; 8];
        assert!(!is_playable_chunk(&zero_chunk));

        let empty_chunk = vec![];
        assert!(!is_playable_chunk(&empty_chunk));
    }
}
