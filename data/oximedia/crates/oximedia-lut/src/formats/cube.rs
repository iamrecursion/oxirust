//! `.cube` LUT file format parser and writer.
//!
//! The `.cube` format is used by Adobe applications, `DaVinci` Resolve, and Nuke.
//! It supports both 1D and 3D LUTs.
//!
//! # Format Specification
//!
//! ```text
//! TITLE "LUT Name"
//! LUT_3D_SIZE 33
//! DOMAIN_MIN 0.0 0.0 0.0
//! DOMAIN_MAX 1.0 1.0 1.0
//!
//! 0.0 0.0 0.0
//! 0.0 0.0 0.5
//! ...
//! ```

use crate::error::{LutError, LutResult};
use crate::{Lut3d, LutSize};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

/// Parse a `.cube` LUT file.
///
/// # Errors
///
/// Returns an error if the file cannot be read or parsed.
pub fn parse_cube_file<P: AsRef<Path>>(path: P) -> LutResult<Lut3d> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut title = None;
    let mut size = None;
    let mut domain_min = [0.0, 0.0, 0.0];
    let mut domain_max = [1.0, 1.0, 1.0];
    let mut data = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Parse metadata
        if line.starts_with("TITLE") {
            title = Some(parse_title(line)?);
        } else if line.starts_with("LUT_3D_SIZE") {
            size = Some(parse_lut_size(line)?);
        } else if line.starts_with("LUT_1D_SIZE") {
            return Err(LutError::UnsupportedFormat(
                "1D LUTs in .cube format not yet supported".to_string(),
            ));
        } else if line.starts_with("DOMAIN_MIN") {
            domain_min = parse_domain(line)?;
        } else if line.starts_with("DOMAIN_MAX") {
            domain_max = parse_domain(line)?;
        } else {
            // Parse RGB data
            let rgb = parse_rgb_line(line)?;
            data.push(rgb);
        }
    }

    // Validate size
    let size = size.ok_or_else(|| LutError::Parse("Missing LUT_3D_SIZE".to_string()))?;
    let expected_entries = size * size * size;

    if data.len() != expected_entries {
        return Err(LutError::InvalidSize {
            expected: expected_entries,
            actual: data.len(),
        });
    }

    // Create LUT
    let mut lut = Lut3d::new(LutSize::from(size));
    lut.title = title;
    lut.input_min = domain_min;
    lut.input_max = domain_max;

    // Fill LUT data
    let mut index = 0;
    for b in 0..size {
        for g in 0..size {
            for r in 0..size {
                lut.set(r, g, b, data[index]);
                index += 1;
            }
        }
    }

    Ok(lut)
}

/// Write a `.cube` LUT file.
///
/// # Errors
///
/// Returns an error if the file cannot be written.
pub fn write_cube_file<P: AsRef<Path>>(path: P, lut: &Lut3d) -> LutResult<()> {
    let mut file = File::create(path)?;

    // Write title
    if let Some(title) = &lut.title {
        writeln!(file, "TITLE \"{title}\"")?;
    } else {
        writeln!(file, "TITLE \"OxiMedia LUT\"")?;
    }

    // Write size
    writeln!(file, "LUT_3D_SIZE {}", lut.size())?;

    // Write domain
    writeln!(
        file,
        "DOMAIN_MIN {} {} {}",
        lut.input_min[0], lut.input_min[1], lut.input_min[2]
    )?;
    writeln!(
        file,
        "DOMAIN_MAX {} {} {}",
        lut.input_max[0], lut.input_max[1], lut.input_max[2]
    )?;

    // Write data (in B-G-R order as per .cube spec)
    let size = lut.size();
    for b in 0..size {
        for g in 0..size {
            for r in 0..size {
                let rgb = lut.get(r, g, b);
                writeln!(file, "{} {} {}", rgb[0], rgb[1], rgb[2])?;
            }
        }
    }

    Ok(())
}

/// Parse the TITLE line.
fn parse_title(line: &str) -> LutResult<String> {
    let parts: Vec<&str> = line.splitn(2, ' ').collect();
    if parts.len() < 2 {
        return Err(LutError::Parse("Invalid TITLE line".to_string()));
    }

    let title = parts[1].trim();
    // Remove quotes if present
    let title = title.trim_matches('"');
    Ok(title.to_string())
}

/// Parse the `LUT_3D_SIZE` line.
fn parse_lut_size(line: &str) -> LutResult<usize> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 2 {
        return Err(LutError::Parse("Invalid LUT_3D_SIZE line".to_string()));
    }

    parts[1]
        .parse::<usize>()
        .map_err(|_| LutError::Parse("Invalid LUT size".to_string()))
}

/// Parse a `DOMAIN_MIN` or `DOMAIN_MAX` line.
fn parse_domain(line: &str) -> LutResult<[f64; 3]> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 4 {
        return Err(LutError::Parse("Invalid DOMAIN line".to_string()));
    }

    let r = parts[1]
        .parse::<f64>()
        .map_err(|_| LutError::Parse("Invalid domain value".to_string()))?;
    let g = parts[2]
        .parse::<f64>()
        .map_err(|_| LutError::Parse("Invalid domain value".to_string()))?;
    let b = parts[3]
        .parse::<f64>()
        .map_err(|_| LutError::Parse("Invalid domain value".to_string()))?;

    Ok([r, g, b])
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

/// A polymorphically-loaded `.cube` LUT: either a 1D per-channel curve or a 3D cube.
#[derive(Clone, Debug)]
pub enum CubeLut {
    /// A 1D LUT parsed from a `LUT_1D_SIZE` `.cube` file.
    Lut1d(crate::Lut1d),
    /// A 3D LUT parsed from a `LUT_3D_SIZE` `.cube` file.
    Lut3d(crate::Lut3d),
}

impl CubeLut {
    /// Returns `true` if this is a 1D LUT.
    #[must_use]
    pub fn is_1d(&self) -> bool {
        matches!(self, CubeLut::Lut1d(_))
    }
    /// Returns `true` if this is a 3D LUT.
    #[must_use]
    pub fn is_3d(&self) -> bool {
        matches!(self, CubeLut::Lut3d(_))
    }
    /// Returns the inner 3D LUT, or `None` if this is a 1D LUT.
    #[must_use]
    pub fn as_3d(&self) -> Option<&crate::Lut3d> {
        match self {
            CubeLut::Lut3d(l) => Some(l),
            CubeLut::Lut1d(_) => None,
        }
    }
    /// Returns the inner 1D LUT, or `None` if this is a 3D LUT.
    #[must_use]
    pub fn as_1d(&self) -> Option<&crate::Lut1d> {
        match self {
            CubeLut::Lut1d(l) => Some(l),
            CubeLut::Lut3d(_) => None,
        }
    }
    /// Loads a `.cube` file, returning either variant based on its header. Mirrors [`parse_cube_any`].
    ///
    /// # Errors
    /// Propagates any error from [`parse_cube_any`].
    pub fn from_file<P: AsRef<Path>>(path: P) -> LutResult<Self> {
        parse_cube_any(path)
    }
}

/// Parses a `.cube` LUT file polymorphically, dispatching on `LUT_1D_SIZE` vs `LUT_3D_SIZE`.
///
/// Unlike [`parse_cube_file`], a 1D `.cube` is accepted and returned as [`CubeLut::Lut1d`].
///
/// # Errors
/// Returns [`LutError::Io`] if the file cannot be read, or [`LutError::UnsupportedFormat`] if the
/// header declares neither a `LUT_1D_SIZE` nor a `LUT_3D_SIZE`. Parser errors are propagated.
pub fn parse_cube_any<P: AsRef<Path>>(path: P) -> LutResult<CubeLut> {
    let path = path.as_ref();
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut kind: Option<bool> = None; // Some(true)=1D, Some(false)=3D
    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.starts_with("LUT_1D_SIZE") {
            kind = Some(true);
            break;
        } else if line.starts_with("LUT_3D_SIZE") {
            kind = Some(false);
            break;
        }
    }
    match kind {
        Some(true) => Ok(CubeLut::Lut1d(crate::Lut1d::from_file(path)?)),
        Some(false) => Ok(CubeLut::Lut3d(crate::Lut3d::from_file(path)?)),
        None => Err(LutError::UnsupportedFormat(
            "no LUT_1D_SIZE or LUT_3D_SIZE header found in .cube file".to_string(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use tempfile::NamedTempFile;

    #[test]
    fn test_parse_title() {
        assert_eq!(
            parse_title("TITLE \"Test LUT\"").expect("should succeed in test"),
            "Test LUT"
        );
        assert_eq!(
            parse_title("TITLE Test").expect("should succeed in test"),
            "Test"
        );
    }

    #[test]
    fn test_parse_lut_size() {
        assert_eq!(
            parse_lut_size("LUT_3D_SIZE 33").expect("should succeed in test"),
            33
        );
    }

    #[test]
    fn test_parse_domain() {
        let domain = parse_domain("DOMAIN_MIN 0.0 0.0 0.0").expect("should succeed in test");
        assert_eq!(domain, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_parse_rgb_line() {
        let rgb = parse_rgb_line("0.5 0.3 0.7").expect("should succeed in test");
        assert!((rgb[0] - 0.5).abs() < 1e-10);
        assert!((rgb[1] - 0.3).abs() < 1e-10);
        assert!((rgb[2] - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_write_and_read_cube() -> LutResult<()> {
        let temp_file = NamedTempFile::new().expect("should succeed in test");
        let path = temp_file.path();

        // Create a simple LUT
        let mut lut = Lut3d::identity(LutSize::Size17);
        lut.title = Some("Test LUT".to_string());

        // Write it
        write_cube_file(path, &lut)?;

        // Read it back
        let loaded = parse_cube_file(path)?;

        // Verify
        assert_eq!(loaded.title, Some("Test LUT".to_string()));
        assert_eq!(loaded.size(), 17);

        // Check a few values
        let val = loaded.get(0, 0, 0);
        assert!((val[0] - 0.0).abs() < 1e-6);

        let val = loaded.get(16, 16, 16);
        assert!((val[0] - 1.0).abs() < 1e-6);

        Ok(())
    }
}
