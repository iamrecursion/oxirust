//! LUT file I/O utilities: parsing and serializing industry-standard LUT text formats.

#![allow(dead_code)]

/// Supported LUT file formats.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LutFormat {
    /// Adobe/DaVinci Resolve `.cube` format.
    Cube,
    /// Autodesk/Lustre `.mga` format.
    Mga,
    /// Cinespace `.csp` format.
    Csp,
    /// Generic `.dat` format.
    Dat,
}

impl LutFormat {
    /// Returns the conventional file extension for this format (without leading dot).
    #[must_use]
    pub fn extension(&self) -> &str {
        match self {
            Self::Cube => "cube",
            Self::Mga => "mga",
            Self::Csp => "csp",
            Self::Dat => "dat",
        }
    }

    /// Returns `true` if the format is plain-text (human readable).
    #[must_use]
    pub fn is_text_based(&self) -> bool {
        match self {
            Self::Cube | Self::Csp | Self::Dat => true,
            Self::Mga => false,
        }
    }

    /// Guess the format from a file extension string (case-insensitive).
    #[must_use]
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_ascii_lowercase().as_str() {
            "cube" => Some(Self::Cube),
            "mga" => Some(Self::Mga),
            "csp" => Some(Self::Csp),
            "dat" => Some(Self::Dat),
            _ => None,
        }
    }
}

/// Metadata extracted from a LUT file header.
#[derive(Debug, Clone)]
pub struct LutFileInfo {
    /// The detected file format.
    pub format: LutFormat,
    /// LUT dimension size (e.g. 33 for a 33×33×33 3D LUT or 1024 for 1D).
    pub size: usize,
    /// Human-readable LUT title extracted from the file.
    pub title: String,
    /// `true` if this describes a 3D LUT, `false` for a 1D LUT.
    pub is_3d: bool,
}

impl LutFileInfo {
    /// Approximate memory footprint in bytes for the LUT data.
    ///
    /// For a 3D LUT: `size^3 * 3 channels * 4 bytes (f32)`.
    /// For a 1D LUT: `size * 3 channels * 4 bytes (f32)`.
    #[must_use]
    pub fn memory_estimate_bytes(&self) -> usize {
        let entries = if self.is_3d {
            self.size * self.size * self.size
        } else {
            self.size
        };
        entries * 3 * 4
    }
}

/// Parse a 1D `.cube` LUT from text content.
///
/// Expects a `LUT_1D_SIZE <n>` header line followed by `<n>` lines each
/// containing a single floating-point value (the red channel is used).
///
/// # Errors
///
/// Returns an error string if the header is missing or any data line fails to parse.
pub fn parse_cube_1d(content: &str) -> Result<Vec<f32>, String> {
    let mut size: Option<usize> = None;
    let mut data: Vec<f32> = Vec::new();

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if trimmed.starts_with("TITLE") || trimmed.starts_with("DOMAIN_") {
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("LUT_1D_SIZE") {
            let s = rest
                .trim()
                .parse::<usize>()
                .map_err(|e| format!("Invalid LUT_1D_SIZE: {e}"))?;
            size = Some(s);
            continue;
        }
        // Data line: may have 1 or 3 columns; we take the first value.
        let first = trimmed
            .split_whitespace()
            .next()
            .ok_or_else(|| "Empty data line".to_string())?;
        let val = first
            .parse::<f32>()
            .map_err(|e| format!("Failed to parse LUT value '{first}': {e}"))?;
        data.push(val);
    }

    let expected = size.ok_or_else(|| "Missing LUT_1D_SIZE header".to_string())?;
    if data.len() < expected {
        return Err(format!(
            "Expected {expected} data points, found {}",
            data.len()
        ));
    }
    data.truncate(expected);
    Ok(data)
}

/// Serialize a 1D LUT to `.cube` text format.
///
/// Produces a minimal valid `.cube` file with the given title and data.
#[must_use]
pub fn serialize_cube_1d(lut: &[f32], title: &str) -> String {
    let mut out = String::new();
    out.push_str(&format!("TITLE \"{title}\"\n"));
    out.push_str(&format!("LUT_1D_SIZE {}\n", lut.len()));
    for &v in lut {
        out.push_str(&format!("{v:.6} {v:.6} {v:.6}\n"));
    }
    out
}

/// Parse only the header of a 3D `.cube` file to extract size and title.
///
/// # Errors
///
/// Returns an error if `LUT_3D_SIZE` is missing.
pub fn parse_cube_3d_header(content: &str) -> Result<(usize, String), String> {
    let mut size: Option<usize> = None;
    let mut title = String::new();

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("TITLE") {
            title = rest.trim().trim_matches('"').to_owned();
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("LUT_3D_SIZE") {
            let s = rest
                .trim()
                .parse::<usize>()
                .map_err(|e| format!("Invalid LUT_3D_SIZE: {e}"))?;
            size = Some(s);
            // Stop after we have both pieces of header data.
            if !title.is_empty() {
                break;
            }
        }
    }

    let sz = size.ok_or_else(|| "Missing LUT_3D_SIZE header".to_string())?;
    Ok((sz, title))
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- LutFormat tests ---

    #[test]
    fn test_extension_cube() {
        assert_eq!(LutFormat::Cube.extension(), "cube");
    }

    #[test]
    fn test_extension_mga() {
        assert_eq!(LutFormat::Mga.extension(), "mga");
    }

    #[test]
    fn test_extension_csp() {
        assert_eq!(LutFormat::Csp.extension(), "csp");
    }

    #[test]
    fn test_extension_dat() {
        assert_eq!(LutFormat::Dat.extension(), "dat");
    }

    #[test]
    fn test_is_text_based_cube_csp_dat() {
        assert!(LutFormat::Cube.is_text_based());
        assert!(LutFormat::Csp.is_text_based());
        assert!(LutFormat::Dat.is_text_based());
    }

    #[test]
    fn test_is_text_based_mga_is_false() {
        assert!(!LutFormat::Mga.is_text_based());
    }

    #[test]
    fn test_from_extension_case_insensitive() {
        assert_eq!(LutFormat::from_extension("CUBE"), Some(LutFormat::Cube));
        assert_eq!(LutFormat::from_extension("csp"), Some(LutFormat::Csp));
        assert_eq!(LutFormat::from_extension("Dat"), Some(LutFormat::Dat));
        assert_eq!(LutFormat::from_extension("MGA"), Some(LutFormat::Mga));
        assert_eq!(LutFormat::from_extension("unknown"), None);
    }

    // --- parse_cube_1d tests ---

    #[test]
    fn test_parse_cube_1d_basic() {
        let content = "LUT_1D_SIZE 3\n0.0\n0.5\n1.0\n";
        let lut = parse_cube_1d(content).expect("should succeed in test");
        assert_eq!(lut.len(), 3);
        assert!((lut[0] - 0.0_f32).abs() < 1e-6);
        assert!((lut[1] - 0.5_f32).abs() < 1e-6);
        assert!((lut[2] - 1.0_f32).abs() < 1e-6);
    }

    #[test]
    fn test_parse_cube_1d_three_column_data() {
        let content = "LUT_1D_SIZE 2\n0.1 0.1 0.1\n0.9 0.9 0.9\n";
        let lut = parse_cube_1d(content).expect("should succeed in test");
        assert_eq!(lut.len(), 2);
        assert!((lut[0] - 0.1_f32).abs() < 1e-5);
    }

    #[test]
    fn test_parse_cube_1d_skips_comments() {
        let content = "# comment\nLUT_1D_SIZE 2\n# another comment\n0.0\n1.0\n";
        let lut = parse_cube_1d(content).expect("should succeed in test");
        assert_eq!(lut.len(), 2);
    }

    #[test]
    fn test_parse_cube_1d_missing_header_is_error() {
        let content = "0.0\n0.5\n1.0\n";
        assert!(parse_cube_1d(content).is_err());
    }

    #[test]
    fn test_parse_cube_1d_too_few_entries_is_error() {
        let content = "LUT_1D_SIZE 4\n0.0\n0.5\n";
        assert!(parse_cube_1d(content).is_err());
    }

    // --- serialize_cube_1d tests ---

    #[test]
    fn test_serialize_cube_1d_contains_title() {
        let lut = vec![0.0_f32, 0.5, 1.0];
        let text = serialize_cube_1d(&lut, "TestLUT");
        assert!(text.contains("TITLE \"TestLUT\""));
    }

    #[test]
    fn test_serialize_cube_1d_contains_size() {
        let lut = vec![0.0_f32, 0.5, 1.0];
        let text = serialize_cube_1d(&lut, "T");
        assert!(text.contains("LUT_1D_SIZE 3"));
    }

    #[test]
    fn test_serialize_then_parse_roundtrip() {
        let original = vec![0.0_f32, 0.25, 0.5, 0.75, 1.0];
        let text = serialize_cube_1d(&original, "Roundtrip");
        let parsed = parse_cube_1d(&text).expect("should succeed in test");
        assert_eq!(parsed.len(), original.len());
        for (a, b) in original.iter().zip(parsed.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    // --- parse_cube_3d_header tests ---

    #[test]
    fn test_parse_cube_3d_header_basic() {
        let content = "TITLE \"My3DLUT\"\nLUT_3D_SIZE 17\n";
        let (size, title) = parse_cube_3d_header(content).expect("should succeed in test");
        assert_eq!(size, 17);
        assert_eq!(title, "My3DLUT");
    }

    #[test]
    fn test_parse_cube_3d_header_missing_size_is_error() {
        let content = "TITLE \"Only Title\"\n";
        assert!(parse_cube_3d_header(content).is_err());
    }

    #[test]
    fn test_parse_cube_3d_header_empty_title_is_ok() {
        let content = "LUT_3D_SIZE 33\n";
        let (size, title) = parse_cube_3d_header(content).expect("should succeed in test");
        assert_eq!(size, 33);
        assert!(title.is_empty());
    }

    // --- LutFileInfo tests ---

    #[test]
    fn test_memory_estimate_1d() {
        let info = LutFileInfo {
            format: LutFormat::Cube,
            size: 1024,
            title: "1D".to_string(),
            is_3d: false,
        };
        // 1024 * 3 * 4 = 12288
        assert_eq!(info.memory_estimate_bytes(), 12_288);
    }

    #[test]
    fn test_memory_estimate_3d() {
        let info = LutFileInfo {
            format: LutFormat::Cube,
            size: 17,
            title: "3D".to_string(),
            is_3d: true,
        };
        // 17^3 * 3 * 4 = 4913 * 12 = 58956
        assert_eq!(info.memory_estimate_bytes(), 17 * 17 * 17 * 3 * 4);
    }
}
