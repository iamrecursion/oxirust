//! `.3dl` LUT file format parser and writer.
//!
//! The `.3dl` format is used by Autodesk Lustre and other professional color grading tools.
//! It's a simple text format with integer RGB values (0-4095 for 12-bit).
//!
//! # Format Specification
//!
//! ```text
//! 33
//! 0 0 0
//! 0 0 128
//! ...
//! 4095 4095 4095
//! ```

use crate::error::{LutError, LutResult};
use crate::{Lut3d, LutSize};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

/// Parse a `.3dl` LUT file.
///
/// # Errors
///
/// Returns an error if the file cannot be read or parsed.
pub fn parse_3dl_file<P: AsRef<Path>>(path: P) -> LutResult<Lut3d> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    // First line contains the size
    let size_line = lines
        .next()
        .ok_or_else(|| LutError::Parse("Empty file".to_string()))??;
    let size = size_line
        .trim()
        .parse::<usize>()
        .map_err(|_| LutError::Parse("Invalid size".to_string()))?;

    let expected_entries = size * size * size;
    let mut data = Vec::with_capacity(expected_entries);

    // Parse RGB data
    for line in lines {
        let line = line?;
        let line = line.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let rgb = parse_rgb_line(line)?;
        data.push(rgb);
    }

    // Validate size
    if data.len() != expected_entries {
        return Err(LutError::InvalidSize {
            expected: expected_entries,
            actual: data.len(),
        });
    }

    // Create LUT
    let mut lut = Lut3d::new(LutSize::from(size));

    // Fill LUT data (3dl uses R-G-B order)
    let mut index = 0;
    for r in 0..size {
        for g in 0..size {
            for b in 0..size {
                lut.set(r, g, b, data[index]);
                index += 1;
            }
        }
    }

    Ok(lut)
}

/// Write a `.3dl` LUT file.
///
/// # Errors
///
/// Returns an error if the file cannot be written.
pub fn write_3dl_file<P: AsRef<Path>>(path: P, lut: &Lut3d) -> LutResult<()> {
    let mut file = File::create(path)?;

    // Write size
    writeln!(file, "{}", lut.size())?;

    // Write data (in R-G-B order, converted to 0-4095 range)
    let size = lut.size();
    for r in 0..size {
        for g in 0..size {
            for b in 0..size {
                let rgb = lut.get(r, g, b);
                let r_int = (rgb[0] * 4095.0).round().clamp(0.0, 4095.0) as u16;
                let g_int = (rgb[1] * 4095.0).round().clamp(0.0, 4095.0) as u16;
                let b_int = (rgb[2] * 4095.0).round().clamp(0.0, 4095.0) as u16;
                writeln!(file, "{r_int} {g_int} {b_int}")?;
            }
        }
    }

    Ok(())
}

/// Parse an RGB data line from a .3dl file.
fn parse_rgb_line(line: &str) -> LutResult<[f64; 3]> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 3 {
        return Err(LutError::Parse("Invalid RGB line".to_string()));
    }

    let r = parts[0]
        .parse::<u16>()
        .map_err(|_| LutError::Parse("Invalid RGB value".to_string()))?;
    let g = parts[1]
        .parse::<u16>()
        .map_err(|_| LutError::Parse("Invalid RGB value".to_string()))?;
    let b = parts[2]
        .parse::<u16>()
        .map_err(|_| LutError::Parse("Invalid RGB value".to_string()))?;

    // Convert from 0-4095 to 0.0-1.0
    Ok([
        f64::from(r) / 4095.0,
        f64::from(g) / 4095.0,
        f64::from(b) / 4095.0,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_parse_rgb_line() {
        let rgb = parse_rgb_line("2048 1024 3072").expect("should succeed in test");
        assert!((rgb[0] - 0.5).abs() < 0.001);
        assert!((rgb[1] - 0.25).abs() < 0.001);
        assert!((rgb[2] - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_write_and_read_3dl() -> LutResult<()> {
        let temp_file = NamedTempFile::new().expect("should succeed in test");
        let path = temp_file.path();

        // Create a simple LUT
        let lut = Lut3d::identity(LutSize::Size17);

        // Write it
        write_3dl_file(path, &lut)?;

        // Read it back
        let loaded = parse_3dl_file(path)?;

        // Verify
        assert_eq!(loaded.size(), 17);

        // Check a few values (with some tolerance due to integer conversion)
        let val = loaded.get(0, 0, 0);
        assert!((val[0] - 0.0).abs() < 0.01);

        let val = loaded.get(16, 16, 16);
        assert!((val[0] - 1.0).abs() < 0.01);

        Ok(())
    }
}
