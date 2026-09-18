//! `.csp` LUT file format parser and writer.
//!
//! The `.csp` format is used by Cinespace and other color grading applications.
//! It supports both 1D and 3D LUTs with metadata.
//!
//! # Format Specification
//!
//! ```text
//! CSPLUTV100
//! 3D
//!
//! BEGIN METADATA
//! Title "My LUT"
//! END METADATA
//!
//! 3
//! 33 33 33
//! 0.0 0.0 0.0
//! ...
//! ```

use crate::error::{LutError, LutResult};
use crate::{Lut3d, LutSize};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

/// Parse a `.csp` LUT file.
///
/// # Errors
///
/// Returns an error if the file cannot be read or parsed.
pub fn parse_csp_file<P: AsRef<Path>>(path: P) -> LutResult<Lut3d> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines().peekable();

    // Check magic header
    let magic = lines
        .next()
        .ok_or_else(|| LutError::Parse("Empty file".to_string()))??;
    if !magic.trim().starts_with("CSPLUTV100") && !magic.trim().starts_with("CSPLUTV200") {
        return Err(LutError::Parse("Invalid CSP header".to_string()));
    }

    // Check type (1D or 3D)
    let lut_type = lines
        .next()
        .ok_or_else(|| LutError::Parse("Missing LUT type".to_string()))??;
    if !lut_type.trim().starts_with("3D") {
        return Err(LutError::UnsupportedFormat(
            "Only 3D CSP LUTs are supported".to_string(),
        ));
    }

    let mut title = None;
    let mut in_metadata = false;

    // Parse metadata section
    while let Some(Ok(line)) = lines.next() {
        let line = line.trim();

        if line.is_empty() {
            continue;
        }

        if line.starts_with("BEGIN METADATA") {
            in_metadata = true;
            continue;
        }

        if line.starts_with("END METADATA") {
            in_metadata = false;
            continue;
        }

        if in_metadata {
            if line.starts_with("Title") {
                let parts: Vec<&str> = line.splitn(2, ' ').collect();
                if parts.len() >= 2 {
                    title = Some(parts[1].trim_matches('"').to_string());
                }
            }
        } else if !line.starts_with('#') {
            // This is the channel count line
            break;
        }
    }

    // Parse dimensions
    let dims_line = lines
        .next()
        .ok_or_else(|| LutError::Parse("Missing dimensions".to_string()))??;
    let dims: Vec<usize> = dims_line
        .split_whitespace()
        .map(str::parse::<usize>)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| LutError::Parse("Invalid dimensions".to_string()))?;

    if dims.len() != 3 {
        return Err(LutError::Parse("Invalid dimensions".to_string()));
    }

    let size = dims[0];
    if size != dims[1] || size != dims[2] {
        return Err(LutError::Parse("Non-cubic LUTs not supported".to_string()));
    }

    let expected_entries = size * size * size;
    let mut data = Vec::with_capacity(expected_entries);

    // Parse RGB data
    for line in lines {
        let line = line?;
        let line = line.trim();

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
    lut.title = title;

    // Fill LUT data (CSP uses R-G-B order)
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

/// Write a `.csp` LUT file.
///
/// # Errors
///
/// Returns an error if the file cannot be written.
pub fn write_csp_file<P: AsRef<Path>>(path: P, lut: &Lut3d) -> LutResult<()> {
    let mut file = File::create(path)?;

    // Write header
    writeln!(file, "CSPLUTV100")?;
    writeln!(file, "3D")?;
    writeln!(file)?;

    // Write metadata
    writeln!(file, "BEGIN METADATA")?;
    if let Some(title) = &lut.title {
        writeln!(file, "Title \"{title}\"")?;
    } else {
        writeln!(file, "Title \"OxiMedia LUT\"")?;
    }
    writeln!(file, "END METADATA")?;
    writeln!(file)?;

    // Write channel count
    writeln!(file, "3")?;

    // Write dimensions
    let size = lut.size();
    writeln!(file, "{size} {size} {size}")?;

    // Write data (in R-G-B order)
    for r in 0..size {
        for g in 0..size {
            for b in 0..size {
                let rgb = lut.get(r, g, b);
                writeln!(file, "{} {} {}", rgb[0], rgb[1], rgb[2])?;
            }
        }
    }

    Ok(())
}

/// Parse an RGB data line.
fn parse_rgb_line(line: &str) -> LutResult<[f64; 3]> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 3 {
        return Err(LutError::Parse("Invalid RGB line".to_string()));
    }

    let r = parts[0]
        .parse::<f64>()
        .map_err(|_| LutError::Parse("Invalid RGB value".to_string()))?;
    let g = parts[1]
        .parse::<f64>()
        .map_err(|_| LutError::Parse("Invalid RGB value".to_string()))?;
    let b = parts[2]
        .parse::<f64>()
        .map_err(|_| LutError::Parse("Invalid RGB value".to_string()))?;

    Ok([r, g, b])
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_write_and_read_csp() -> LutResult<()> {
        let temp_file = NamedTempFile::new().expect("should succeed in test");
        let path = temp_file.path();

        // Create a simple LUT
        let mut lut = Lut3d::identity(LutSize::Size17);
        lut.title = Some("Test CSP LUT".to_string());

        // Write it
        write_csp_file(path, &lut)?;

        // Read it back
        let loaded = parse_csp_file(path)?;

        // Verify
        assert_eq!(loaded.title, Some("Test CSP LUT".to_string()));
        assert_eq!(loaded.size(), 17);

        // Check a few values
        let val = loaded.get(0, 0, 0);
        assert!((val[0] - 0.0).abs() < 1e-6);

        let val = loaded.get(16, 16, 16);
        assert!((val[0] - 1.0).abs() < 1e-6);

        Ok(())
    }
}
