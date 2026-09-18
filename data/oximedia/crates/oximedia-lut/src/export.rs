//! LUT export in various standard formats.
//!
//! Supports .cube (Adobe/DaVinci), .csp (Cinespace), and other formats.

#![allow(dead_code)]

use std::fmt::Write as FmtWrite;

/// LUT export format selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LutExportFormat {
    /// Adobe/DaVinci Resolve .cube format.
    Cube,
    /// Cinespace .csp format.
    Csp,
    /// PNG image representation (not yet implemented for direct write).
    Png,
    /// Raw binary .dat format.
    Dat,
}

/// Options for LUT export.
#[derive(Clone, Debug)]
pub struct ExportOptions {
    /// Target export format.
    pub format: LutExportFormat,
    /// LUT title embedded in the file header.
    pub title: String,
    /// Optional comment lines embedded in the header.
    pub comments: Vec<String>,
    /// Domain minimum values for R, G, B channels.
    pub domain_min: [f64; 3],
    /// Domain maximum values for R, G, B channels.
    pub domain_max: [f64; 3],
}

impl ExportOptions {
    /// Create default options for .cube export.
    #[must_use]
    pub fn default_cube() -> Self {
        Self {
            format: LutExportFormat::Cube,
            title: String::from("Untitled LUT"),
            comments: Vec::new(),
            domain_min: [0.0, 0.0, 0.0],
            domain_max: [1.0, 1.0, 1.0],
        }
    }

    /// Set the title for this export.
    #[must_use]
    pub fn with_title(mut self, title: &str) -> Self {
        self.title = title.to_string();
        self
    }

    /// Add a comment line to the export.
    #[must_use]
    pub fn with_comment(mut self, comment: &str) -> Self {
        self.comments.push(comment.to_string());
        self
    }

    /// Set domain minimum values.
    #[must_use]
    pub fn with_domain_min(mut self, min: [f64; 3]) -> Self {
        self.domain_min = min;
        self
    }

    /// Set domain maximum values.
    #[must_use]
    pub fn with_domain_max(mut self, max: [f64; 3]) -> Self {
        self.domain_max = max;
        self
    }
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self::default_cube()
    }
}

/// Export a 1D LUT to .cube format string.
///
/// # Arguments
///
/// * `values` - LUT values (single channel, normalized 0.0–1.0)
/// * `title` - Title string embedded in the header
#[must_use]
pub fn export_1d_lut_cube(values: &[f64], title: &str) -> String {
    let size = values.len();
    let mut out = String::new();
    let _ = writeln!(out, "TITLE \"{title}\"");
    let _ = writeln!(out, "LUT_1D_SIZE {size}");
    let _ = writeln!(out, "DOMAIN_MIN 0.0 0.0 0.0");
    let _ = writeln!(out, "DOMAIN_MAX 1.0 1.0 1.0");
    out.push('\n');
    for &v in values {
        let clamped = v.clamp(0.0, 1.0);
        let _ = writeln!(out, "{clamped:.6} {clamped:.6} {clamped:.6}");
    }
    out
}

/// Export a 3D LUT to .cube format string.
///
/// # Arguments
///
/// * `lut` - Flattened 3D LUT values (RGB interleaved, B-major order)
/// * `size` - Number of entries per dimension
/// * `opts` - Export options
#[must_use]
pub fn export_3d_lut_cube(lut: &[f64], size: usize, opts: &ExportOptions) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "TITLE \"{}\"", opts.title);
    for comment in &opts.comments {
        let _ = writeln!(out, "# {comment}");
    }
    let _ = writeln!(out, "LUT_3D_SIZE {size}");
    let [dmin_r, dmin_g, dmin_b] = opts.domain_min;
    let [dmax_r, dmax_g, dmax_b] = opts.domain_max;
    let _ = writeln!(out, "DOMAIN_MIN {dmin_r:.6} {dmin_g:.6} {dmin_b:.6}");
    let _ = writeln!(out, "DOMAIN_MAX {dmax_r:.6} {dmax_g:.6} {dmax_b:.6}");
    out.push('\n');

    let total = size * size * size;
    for i in 0..total {
        let r = lut[i * 3].clamp(0.0, 1.0);
        let g = lut[i * 3 + 1].clamp(0.0, 1.0);
        let b = lut[i * 3 + 2].clamp(0.0, 1.0);
        let _ = writeln!(out, "{r:.6} {g:.6} {b:.6}");
    }
    out
}

/// Export a 1D LUT to Cinespace .csp format.
///
/// # Arguments
///
/// * `values` - LUT values (single channel, normalized 0.0–1.0)
#[must_use]
pub fn export_csp_1d(values: &[f64]) -> String {
    let size = values.len();
    let mut out = String::new();
    out.push_str("CSPLUTV100\n");
    out.push_str("1D\n\n");
    out.push_str("BEGIN METADATA\nEND METADATA\n\n");
    // Pre-LUT (identity)
    let _ = writeln!(out, "2");
    out.push_str("0.0 1.0\n");
    out.push_str("0.0 1.0\n\n");
    // 1D LUT data (same values for R, G, B)
    let _ = writeln!(out, "{size}");
    for &v in values {
        let clamped = v.clamp(0.0, 1.0);
        let _ = write!(out, "{clamped:.6} ");
    }
    out.push('\n');
    for &v in values {
        let clamped = v.clamp(0.0, 1.0);
        let _ = write!(out, "{clamped:.6} ");
    }
    out.push('\n');
    for &v in values {
        let clamped = v.clamp(0.0, 1.0);
        let _ = write!(out, "{clamped:.6} ");
    }
    out.push('\n');
    out
}

/// Parse a .cube file containing a 1D LUT.
///
/// # Errors
///
/// Returns `Err(String)` if the content is not a valid 1D cube file.
pub fn parse_cube_1d(content: &str) -> Result<Vec<f64>, String> {
    let mut size: Option<usize> = None;
    let mut values: Vec<f64> = Vec::new();
    let mut reading_data = false;

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if trimmed.starts_with("TITLE") || trimmed.starts_with("DOMAIN_") {
            continue;
        }
        if let Some(sz_str) = trimmed.strip_prefix("LUT_1D_SIZE") {
            let sz: usize = sz_str
                .trim()
                .parse()
                .map_err(|_| format!("Invalid LUT_1D_SIZE: {}", sz_str.trim()))?;
            size = Some(sz);
            reading_data = true;
            continue;
        }
        if trimmed.starts_with("LUT_3D_SIZE") {
            return Err("File contains a 3D LUT, not 1D".to_string());
        }
        if reading_data {
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if !parts.is_empty() {
                let v: f64 = parts[0]
                    .parse()
                    .map_err(|_| format!("Invalid value: {}", parts[0]))?;
                values.push(v);
            }
        }
    }

    let sz = size.ok_or_else(|| "Missing LUT_1D_SIZE header".to_string())?;
    if values.len() != sz {
        return Err(format!("Expected {sz} values, found {}", values.len()));
    }
    Ok(values)
}

/// Parse a .cube file containing a 3D LUT.
///
/// # Errors
///
/// Returns `Err(String)` if the content is not a valid 3D cube file.
pub fn parse_cube_3d(content: &str) -> Result<(Vec<f64>, usize), String> {
    let mut size: Option<usize> = None;
    let mut values: Vec<f64> = Vec::new();
    let mut reading_data = false;

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if trimmed.starts_with("TITLE") || trimmed.starts_with("DOMAIN_") {
            continue;
        }
        if let Some(sz_str) = trimmed.strip_prefix("LUT_3D_SIZE") {
            let sz: usize = sz_str
                .trim()
                .parse()
                .map_err(|_| format!("Invalid LUT_3D_SIZE: {}", sz_str.trim()))?;
            size = Some(sz);
            reading_data = true;
            continue;
        }
        if trimmed.starts_with("LUT_1D_SIZE") {
            return Err("File contains a 1D LUT, not 3D".to_string());
        }
        if reading_data {
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() >= 3 {
                for p in &parts[..3] {
                    let v: f64 = p.parse().map_err(|_| format!("Invalid value: {p}"))?;
                    values.push(v);
                }
            }
        }
    }

    let sz = size.ok_or_else(|| "Missing LUT_3D_SIZE header".to_string())?;
    let expected = sz * sz * sz * 3;
    if values.len() != expected {
        return Err(format!(
            "Expected {expected} values ({}^3 * 3), found {}",
            sz,
            values.len()
        ));
    }
    Ok((values, sz))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_1d(size: usize) -> Vec<f64> {
        (0..size).map(|i| i as f64 / (size - 1) as f64).collect()
    }

    fn identity_3d(size: usize) -> Vec<f64> {
        let mut lut = Vec::with_capacity(size * size * size * 3);
        for b in 0..size {
            for g in 0..size {
                for r in 0..size {
                    lut.push(r as f64 / (size - 1) as f64);
                    lut.push(g as f64 / (size - 1) as f64);
                    lut.push(b as f64 / (size - 1) as f64);
                }
            }
        }
        lut
    }

    #[test]
    fn test_export_1d_cube_contains_header() {
        let values = identity_1d(8);
        let s = export_1d_lut_cube(&values, "Test LUT");
        assert!(s.contains("TITLE \"Test LUT\""));
        assert!(s.contains("LUT_1D_SIZE 8"));
        assert!(s.contains("DOMAIN_MIN"));
        assert!(s.contains("DOMAIN_MAX"));
    }

    #[test]
    fn test_export_1d_cube_line_count() {
        let size = 16;
        let values = identity_1d(size);
        let s = export_1d_lut_cube(&values, "X");
        // Count data lines (r g b)
        let data_lines = s
            .lines()
            .filter(|l| {
                let p: Vec<&str> = l.split_whitespace().collect();
                p.len() == 3 && p[0].parse::<f64>().is_ok()
            })
            .count();
        assert_eq!(data_lines, size);
    }

    #[test]
    fn test_roundtrip_1d_cube() {
        let orig = identity_1d(16);
        let cube_str = export_1d_lut_cube(&orig, "Identity");
        let parsed = parse_cube_1d(&cube_str).expect("parse failed");
        assert_eq!(parsed.len(), orig.len());
        for (a, b) in orig.iter().zip(parsed.iter()) {
            assert!((a - b).abs() < 1e-5, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_export_3d_cube_size_2() {
        let lut = identity_3d(2);
        let opts = ExportOptions::default_cube().with_title("Id2");
        let s = export_3d_lut_cube(&lut, 2, &opts);
        assert!(s.contains("LUT_3D_SIZE 2"));
        assert!(s.contains("TITLE \"Id2\""));
        // 8 data lines for 2^3
        let data_lines: usize = s
            .lines()
            .filter(|l| {
                let p: Vec<&str> = l.split_whitespace().collect();
                p.len() == 3 && p[0].parse::<f64>().is_ok()
            })
            .count();
        assert_eq!(data_lines, 8);
    }

    #[test]
    fn test_roundtrip_3d_cube() {
        let orig = identity_3d(4);
        let opts = ExportOptions::default_cube().with_title("Id4");
        let cube_str = export_3d_lut_cube(&orig, 4, &opts);
        let (parsed, sz) = parse_cube_3d(&cube_str).expect("parse failed");
        assert_eq!(sz, 4);
        assert_eq!(parsed.len(), orig.len());
        for (a, b) in orig.iter().zip(parsed.iter()) {
            assert!((a - b).abs() < 1e-5, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_export_options_builder() {
        let opts = ExportOptions::default_cube()
            .with_title("My LUT")
            .with_comment("Created by test")
            .with_domain_min([0.0; 3])
            .with_domain_max([1.0; 3]);
        assert_eq!(opts.title, "My LUT");
        assert_eq!(opts.comments.len(), 1);
        assert_eq!(opts.domain_min, [0.0; 3]);
        assert_eq!(opts.domain_max, [1.0; 3]);
    }

    #[test]
    fn test_export_csp_1d_header() {
        let values = identity_1d(8);
        let s = export_csp_1d(&values);
        assert!(s.starts_with("CSPLUTV100"));
        assert!(s.contains("1D"));
        assert!(s.contains("BEGIN METADATA"));
        assert!(s.contains("END METADATA"));
    }

    #[test]
    fn test_export_csp_1d_data_count() {
        let size = 10;
        let values = identity_1d(size);
        let s = export_csp_1d(&values);
        // Size line should appear
        assert!(s.contains(&format!("{size}")));
    }

    #[test]
    fn test_parse_cube_1d_wrong_size_error() {
        let content = "LUT_1D_SIZE 5\n0.0 0.0 0.0\n";
        let result = parse_cube_1d(content);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_cube_3d_wrong_type_error() {
        let content = "LUT_1D_SIZE 4\n";
        let result = parse_cube_3d(content);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_cube_1d_wrong_type_error() {
        let content = "LUT_3D_SIZE 4\n";
        let result = parse_cube_1d(content);
        assert!(result.is_err());
    }

    #[test]
    fn test_export_3d_cube_comments() {
        let lut = identity_3d(2);
        let opts = ExportOptions::default_cube()
            .with_title("Test")
            .with_comment("Line 1")
            .with_comment("Line 2");
        let s = export_3d_lut_cube(&lut, 2, &opts);
        assert!(s.contains("# Line 1"));
        assert!(s.contains("# Line 2"));
    }

    #[test]
    fn test_export_1d_cube_clamps_values() {
        let values = vec![-0.5, 0.5, 1.5];
        let s = export_1d_lut_cube(&values, "Clamp Test");
        // Should clamp to [0,1]
        assert!(s.contains("0.000000 0.000000 0.000000"));
        assert!(s.contains("1.000000 1.000000 1.000000"));
    }

    #[test]
    fn test_lut_export_format_variants() {
        let formats = [
            LutExportFormat::Cube,
            LutExportFormat::Csp,
            LutExportFormat::Png,
            LutExportFormat::Dat,
        ];
        assert_eq!(formats.len(), 4);
    }

    #[test]
    fn test_parse_cube_1d_skips_comments() {
        let content = "# This is a comment\nTITLE \"Test\"\nLUT_1D_SIZE 3\n0.0 0.0 0.0\n0.5 0.5 0.5\n1.0 1.0 1.0\n";
        let result = parse_cube_1d(content).expect("parse failed");
        assert_eq!(result.len(), 3);
        assert!((result[0] - 0.0).abs() < 1e-6);
        assert!((result[1] - 0.5).abs() < 1e-6);
        assert!((result[2] - 1.0).abs() < 1e-6);
    }
}
