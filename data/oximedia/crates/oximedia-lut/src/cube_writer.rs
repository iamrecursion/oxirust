//! Writer for `.cube` LUT files (Adobe / `DaVinci` Resolve format).
//!
//! Supports 1D and 3D LUT export with configurable precision, domain
//! ranges, and optional title / comment embedding.

#![allow(dead_code)]

use std::fmt;

/// Selects between 1D and 3D cube file variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CubeFormat {
    /// 1D LUT: per-channel curves.
    OneDimensional,
    /// 3D LUT: full RGB lattice.
    ThreeDimensional,
}

impl CubeFormat {
    /// Human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::OneDimensional => "1D",
            Self::ThreeDimensional => "3D",
        }
    }
}

impl fmt::Display for CubeFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Configuration for writing a `.cube` file.
#[derive(Debug, Clone)]
pub struct CubeWriteConfig {
    /// Title embedded in the cube file header.
    pub title: String,
    /// Number of decimal places for floating-point values.
    pub precision: u8,
    /// Domain minimum [R, G, B].
    pub domain_min: [f64; 3],
    /// Domain maximum [R, G, B].
    pub domain_max: [f64; 3],
    /// Optional comment lines (each will be prefixed with `#`).
    pub comments: Vec<String>,
}

impl Default for CubeWriteConfig {
    fn default() -> Self {
        Self {
            title: String::new(),
            precision: 6,
            domain_min: [0.0; 3],
            domain_max: [1.0; 3],
            comments: Vec::new(),
        }
    }
}

impl CubeWriteConfig {
    /// Create a new default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the title.
    #[must_use]
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    /// Set decimal precision.
    #[must_use]
    pub fn with_precision(mut self, prec: u8) -> Self {
        self.precision = prec.clamp(1, 15);
        self
    }

    /// Set domain minimum.
    #[must_use]
    pub fn with_domain_min(mut self, min: [f64; 3]) -> Self {
        self.domain_min = min;
        self
    }

    /// Set domain maximum.
    #[must_use]
    pub fn with_domain_max(mut self, max: [f64; 3]) -> Self {
        self.domain_max = max;
        self
    }

    /// Add a comment line.
    #[must_use]
    pub fn with_comment(mut self, comment: impl Into<String>) -> Self {
        self.comments.push(comment.into());
        self
    }

    /// Validate that `domain_min` < `domain_max` in all channels.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        (0..3).all(|i| self.domain_min[i] < self.domain_max[i]) && self.precision >= 1
    }
}

/// Writer that serialises LUT data into the `.cube` text format.
#[derive(Debug)]
pub struct CubeWriter {
    /// The cube format (1D or 3D).
    pub format: CubeFormat,
    /// Write configuration.
    pub config: CubeWriteConfig,
}

impl CubeWriter {
    /// Create a new writer for the given format.
    #[must_use]
    pub fn new(format: CubeFormat) -> Self {
        Self {
            format,
            config: CubeWriteConfig::default(),
        }
    }

    /// Set configuration.
    #[must_use]
    pub fn with_config(mut self, config: CubeWriteConfig) -> Self {
        self.config = config;
        self
    }

    /// Render a 1D LUT as a `.cube` string.
    ///
    /// `entries` is a list of `[R, G, B]` triplets, one per LUT entry.
    /// Returns `None` if the format is not `OneDimensional` or entries are empty.
    #[must_use]
    pub fn write_1d(&self, entries: &[[f64; 3]]) -> Option<String> {
        if self.format != CubeFormat::OneDimensional || entries.is_empty() {
            return None;
        }
        let mut out = String::new();
        self.write_header(&mut out, entries.len());
        let prec = self.config.precision as usize;
        for e in entries {
            out.push_str(&format!(
                "{:.prec$} {:.prec$} {:.prec$}\n",
                e[0],
                e[1],
                e[2],
                prec = prec,
            ));
        }
        Some(out)
    }

    /// Render a 3D LUT as a `.cube` string.
    ///
    /// `size` is the lattice dimension and `entries` must contain `size^3`
    /// `[R, G, B]` triplets in B-G-R nesting order (standard .cube order).
    /// Returns `None` on format mismatch or size mismatch.
    #[must_use]
    pub fn write_3d(&self, size: usize, entries: &[[f64; 3]]) -> Option<String> {
        if self.format != CubeFormat::ThreeDimensional || size == 0 {
            return None;
        }
        if entries.len() != size * size * size {
            return None;
        }
        let mut out = String::new();
        self.write_header(&mut out, size);
        let prec = self.config.precision as usize;
        for e in entries {
            out.push_str(&format!(
                "{:.prec$} {:.prec$} {:.prec$}\n",
                e[0],
                e[1],
                e[2],
                prec = prec,
            ));
        }
        Some(out)
    }

    /// Write the header portion (title, comments, domain, size keywords).
    fn write_header(&self, out: &mut String, size: usize) {
        // Title
        if !self.config.title.is_empty() {
            out.push_str(&format!("TITLE \"{}\"\n", self.config.title));
        }
        // Comments
        for c in &self.config.comments {
            out.push_str(&format!("# {c}\n"));
        }
        // Domain
        let prec = self.config.precision as usize;
        out.push_str(&format!(
            "DOMAIN_MIN {:.prec$} {:.prec$} {:.prec$}\n",
            self.config.domain_min[0],
            self.config.domain_min[1],
            self.config.domain_min[2],
            prec = prec,
        ));
        out.push_str(&format!(
            "DOMAIN_MAX {:.prec$} {:.prec$} {:.prec$}\n",
            self.config.domain_max[0],
            self.config.domain_max[1],
            self.config.domain_max[2],
            prec = prec,
        ));
        // Size keyword
        match self.format {
            CubeFormat::OneDimensional => {
                out.push_str(&format!("LUT_1D_SIZE {size}\n"));
            }
            CubeFormat::ThreeDimensional => {
                out.push_str(&format!("LUT_3D_SIZE {size}\n"));
            }
        }
        out.push('\n');
    }

    /// Estimate the output string length in bytes for a given entry count.
    #[must_use]
    pub fn estimate_size(&self, entry_count: usize) -> usize {
        // ~30 chars per line * entries + ~200 bytes header.
        let line_len = (self.config.precision as usize + 2) * 3 + 3;
        200 + entry_count * line_len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cube_format_label() {
        assert_eq!(CubeFormat::OneDimensional.label(), "1D");
        assert_eq!(CubeFormat::ThreeDimensional.label(), "3D");
    }

    #[test]
    fn test_cube_format_display() {
        assert_eq!(format!("{}", CubeFormat::OneDimensional), "1D");
        assert_eq!(format!("{}", CubeFormat::ThreeDimensional), "3D");
    }

    #[test]
    fn test_config_default_valid() {
        assert!(CubeWriteConfig::default().is_valid());
    }

    #[test]
    fn test_config_invalid_domain() {
        let cfg = CubeWriteConfig::new()
            .with_domain_min([1.0, 0.0, 0.0])
            .with_domain_max([0.0, 1.0, 1.0]);
        assert!(!cfg.is_valid());
    }

    #[test]
    fn test_config_builder() {
        let cfg = CubeWriteConfig::new()
            .with_title("MyLUT")
            .with_precision(8)
            .with_comment("generated by test");
        assert_eq!(cfg.title, "MyLUT");
        assert_eq!(cfg.precision, 8);
        assert_eq!(cfg.comments.len(), 1);
    }

    #[test]
    fn test_precision_clamp() {
        let cfg = CubeWriteConfig::new().with_precision(0);
        assert_eq!(cfg.precision, 1);
        let cfg2 = CubeWriteConfig::new().with_precision(20);
        assert_eq!(cfg2.precision, 15);
    }

    #[test]
    fn test_write_1d_identity() {
        let writer = CubeWriter::new(CubeFormat::OneDimensional)
            .with_config(CubeWriteConfig::new().with_title("Identity"));
        let entries: Vec<[f64; 3]> = (0..4)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let v = i as f64 / 3.0;
                [v, v, v]
            })
            .collect();
        let text = writer.write_1d(&entries).expect("should succeed in test");
        assert!(text.contains("TITLE \"Identity\""));
        assert!(text.contains("LUT_1D_SIZE 4"));
    }

    #[test]
    fn test_write_1d_wrong_format() {
        let writer = CubeWriter::new(CubeFormat::ThreeDimensional);
        assert!(writer.write_1d(&[[0.0; 3]]).is_none());
    }

    #[test]
    fn test_write_1d_empty() {
        let writer = CubeWriter::new(CubeFormat::OneDimensional);
        assert!(writer.write_1d(&[]).is_none());
    }

    #[test]
    fn test_write_3d_identity_2() {
        let size = 2;
        let entries: Vec<[f64; 3]> = (0..8)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let v = i as f64 / 7.0;
                [v, v, v]
            })
            .collect();
        let writer = CubeWriter::new(CubeFormat::ThreeDimensional);
        let text = writer
            .write_3d(size, &entries)
            .expect("should succeed in test");
        assert!(text.contains("LUT_3D_SIZE 2"));
    }

    #[test]
    fn test_write_3d_size_mismatch() {
        let writer = CubeWriter::new(CubeFormat::ThreeDimensional);
        let entries = vec![[0.0; 3]; 5]; // size=2 needs 8.
        assert!(writer.write_3d(2, &entries).is_none());
    }

    #[test]
    fn test_write_3d_wrong_format() {
        let writer = CubeWriter::new(CubeFormat::OneDimensional);
        assert!(writer.write_3d(2, &vec![[0.0; 3]; 8]).is_none());
    }

    #[test]
    fn test_header_includes_domain() {
        let cfg = CubeWriteConfig::new()
            .with_domain_min([0.0, 0.0, 0.0])
            .with_domain_max([1.0, 1.0, 1.0]);
        let writer = CubeWriter::new(CubeFormat::OneDimensional).with_config(cfg);
        let text = writer
            .write_1d(&[[0.0; 3], [1.0; 3]])
            .expect("should succeed in test");
        assert!(text.contains("DOMAIN_MIN"));
        assert!(text.contains("DOMAIN_MAX"));
    }

    #[test]
    fn test_header_includes_comments() {
        let cfg = CubeWriteConfig::new()
            .with_comment("line one")
            .with_comment("line two");
        let writer = CubeWriter::new(CubeFormat::OneDimensional).with_config(cfg);
        let text = writer
            .write_1d(&[[0.0; 3], [1.0; 3]])
            .expect("should succeed in test");
        assert!(text.contains("# line one"));
        assert!(text.contains("# line two"));
    }

    #[test]
    fn test_estimate_size_positive() {
        let writer = CubeWriter::new(CubeFormat::ThreeDimensional);
        let est = writer.estimate_size(1000);
        assert!(est > 200);
    }

    #[test]
    fn test_write_1d_precision() {
        let cfg = CubeWriteConfig::new().with_precision(2);
        let writer = CubeWriter::new(CubeFormat::OneDimensional).with_config(cfg);
        let text = writer
            .write_1d(&[[0.123_456_789, 0.0, 1.0]])
            .expect("should succeed in test");
        // With precision 2 we expect "0.12" not "0.123456".
        assert!(text.contains("0.12"));
        assert!(!text.contains("0.1234"));
    }
}
