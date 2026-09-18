//! Matroska/WebM header repair.
//!
//! This module provides functions to repair corrupted Matroska/WebM file headers.

use crate::Result;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

/// Repair Matroska file header.
pub fn repair_matroska_header(input: &Path, output: &Path) -> Result<bool> {
    super::repair::copy_with_repair(input, output, |file| {
        let mut repaired = false;

        // Repair EBML header
        if repair_ebml_header(file)? {
            repaired = true;
        }

        // Repair segment header
        if repair_segment_header(file)? {
            repaired = true;
        }

        Ok(repaired)
    })
}

/// Repair EBML header.
fn repair_ebml_header(file: &mut File) -> Result<bool> {
    file.seek(SeekFrom::Start(0))?;

    let mut signature = [0u8; 4];
    file.read_exact(&mut signature)?;

    // Check EBML signature
    if signature != [0x1A, 0x45, 0xDF, 0xA3] {
        // Fix signature
        file.seek(SeekFrom::Start(0))?;
        file.write_all(&[0x1A, 0x45, 0xDF, 0xA3])?;
        return Ok(true);
    }

    Ok(false)
}

/// Repair segment header.
fn repair_segment_header(file: &mut File) -> Result<bool> {
    // Find segment element
    if let Some(offset) = find_element(file, 0x18538067)? {
        file.seek(SeekFrom::Start(offset))?;

        // Read element
        let mut element_id = [0u8; 4];
        file.read_exact(&mut element_id)?;

        // Check if size is valid
        let size = read_element_size(file)?;

        if size == 0 {
            // Calculate correct size
            let file_size = file.metadata()?.len();
            let remaining = file_size - offset - 4 - element_size_length(file_size);

            // Write corrected size
            file.seek(SeekFrom::Start(offset + 4))?;
            write_element_size(file, remaining)?;

            return Ok(true);
        }
    }

    Ok(false)
}

/// Find an EBML element by ID.
fn find_element(file: &mut File, element_id: u32) -> Result<Option<u64>> {
    file.seek(SeekFrom::Start(0))?;
    let file_size = file.metadata()?.len();

    let mut pos = 0u64;
    while pos + 8 <= file_size {
        file.seek(SeekFrom::Start(pos))?;

        let mut id_bytes = [0u8; 4];
        if file.read_exact(&mut id_bytes).is_err() {
            break;
        }

        let id = u32::from_be_bytes(id_bytes);
        if id == element_id {
            return Ok(Some(pos));
        }

        pos += 1;
    }

    Ok(None)
}

/// Read EBML element size.
fn read_element_size(file: &mut File) -> Result<u64> {
    let mut first_byte = [0u8; 1];
    file.read_exact(&mut first_byte)?;

    let first = first_byte[0];

    // Count leading zeros to determine size length
    let size_length = if first & 0x80 != 0 {
        1
    } else if first & 0x40 != 0 {
        2
    } else if first & 0x20 != 0 {
        3
    } else if first & 0x10 != 0 {
        4
    } else if first & 0x08 != 0 {
        5
    } else if first & 0x04 != 0 {
        6
    } else if first & 0x02 != 0 {
        7
    } else {
        8
    };

    // Read remaining bytes
    let mut size_bytes = vec![0u8; size_length];
    size_bytes[0] = first;
    if size_length > 1 {
        file.read_exact(&mut size_bytes[1..])?;
    }

    // Parse size
    let mut size = 0u64;
    for (i, &byte) in size_bytes.iter().enumerate() {
        if i == 0 {
            // Remove the length marker bit
            let mask = match size_length {
                1 => 0x7F,
                2 => 0x3F,
                3 => 0x1F,
                4 => 0x0F,
                5 => 0x07,
                6 => 0x03,
                7 => 0x01,
                _ => 0x00,
            };
            size = (byte & mask) as u64;
        } else {
            size = (size << 8) | byte as u64;
        }
    }

    Ok(size)
}

/// Write EBML element size.
fn write_element_size(file: &mut File, size: u64) -> Result<()> {
    let size_bytes = encode_element_size(size);
    file.write_all(&size_bytes)?;
    Ok(())
}

/// Encode EBML element size.
fn encode_element_size(size: u64) -> Vec<u8> {
    // Determine number of bytes needed
    let bytes_needed = if size < 127 {
        1
    } else if size < 16383 {
        2
    } else if size < 2097151 {
        3
    } else if size < 268435455 {
        4
    } else if size < 34359738367 {
        5
    } else if size < 4398046511103 {
        6
    } else if size < 562949953421311 {
        7
    } else {
        8
    };

    let mut result = Vec::new();

    match bytes_needed {
        1 => {
            result.push(0x80 | (size as u8));
        }
        2 => {
            result.push(0x40 | ((size >> 8) as u8));
            result.push((size & 0xFF) as u8);
        }
        3 => {
            result.push(0x20 | ((size >> 16) as u8));
            result.push(((size >> 8) & 0xFF) as u8);
            result.push((size & 0xFF) as u8);
        }
        4 => {
            result.push(0x10 | ((size >> 24) as u8));
            result.push(((size >> 16) & 0xFF) as u8);
            result.push(((size >> 8) & 0xFF) as u8);
            result.push((size & 0xFF) as u8);
        }
        _ => {
            // For larger sizes, use 8 bytes
            result.push(0x01);
            for i in (0..7).rev() {
                result.push(((size >> (i * 8)) & 0xFF) as u8);
            }
        }
    }

    result
}

/// Calculate element size length for a given file size.
fn element_size_length(file_size: u64) -> u64 {
    if file_size < 127 {
        1
    } else if file_size < 16383 {
        2
    } else if file_size < 2097151 {
        3
    } else if file_size < 268435455 {
        4
    } else {
        8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_element_size_1_byte() {
        let encoded = encode_element_size(100);
        assert_eq!(encoded.len(), 1);
        assert_eq!(encoded[0], 0x80 | 100);
    }

    #[test]
    fn test_encode_element_size_2_bytes() {
        let encoded = encode_element_size(1000);
        assert_eq!(encoded.len(), 2);
    }

    #[test]
    fn test_encode_element_size_3_bytes() {
        let encoded = encode_element_size(100000);
        assert_eq!(encoded.len(), 3);
    }

    #[test]
    fn test_element_size_length() {
        assert_eq!(element_size_length(100), 1);
        assert_eq!(element_size_length(1000), 2);
        assert_eq!(element_size_length(100000), 3);
        assert_eq!(element_size_length(10000000), 4);
    }
}
