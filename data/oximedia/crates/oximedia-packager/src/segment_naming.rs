#![allow(dead_code)]
//! Segment file naming strategies and templates for adaptive streaming.
//!
//! This module provides configurable naming conventions for segment files,
//! init segments, and manifest files across HLS, DASH, and CMAF packaging.

use crate::config::SegmentFormat;
use crate::error::{PackagerError, PackagerResult};

/// Naming strategy for segment files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NamingStrategy {
    /// Sequential numeric: `segment_0.m4s`, `segment_1.m4s`, ...
    Sequential,
    /// Timestamp-based: `segment_1708000000.m4s`.
    Timestamp,
    /// Duration-based: `segment_t0.000.m4s`, `segment_t6.000.m4s`, ...
    Duration,
}

/// A template for naming segment files.
#[derive(Debug, Clone)]
pub struct NamingTemplate {
    /// Prefix for segment files.
    pub prefix: String,
    /// Separator between prefix and index.
    pub separator: String,
    /// Strategy for the variable part.
    pub strategy: NamingStrategy,
    /// Minimum number of digits for zero-padding.
    pub zero_pad: usize,
    /// File extension (without dot) — auto-detected from format if `None`.
    pub extension: Option<String>,
}

impl Default for NamingTemplate {
    fn default() -> Self {
        Self {
            prefix: "segment".to_string(),
            separator: "_".to_string(),
            strategy: NamingStrategy::Sequential,
            zero_pad: 0,
            extension: None,
        }
    }
}

impl NamingTemplate {
    /// Create a new naming template with the given prefix.
    #[must_use]
    pub fn with_prefix(prefix: &str) -> Self {
        Self {
            prefix: prefix.to_string(),
            ..Default::default()
        }
    }

    /// Set the naming strategy.
    #[must_use]
    pub fn with_strategy(mut self, strategy: NamingStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set zero-padding width.
    #[must_use]
    pub fn with_zero_pad(mut self, digits: usize) -> Self {
        self.zero_pad = digits;
        self
    }

    /// Set a custom file extension.
    #[must_use]
    pub fn with_extension(mut self, ext: &str) -> Self {
        self.extension = Some(ext.to_string());
        self
    }

    /// Set the separator.
    #[must_use]
    pub fn with_separator(mut self, sep: &str) -> Self {
        self.separator = sep.to_string();
        self
    }

    /// Generate a segment file name.
    ///
    /// # Arguments
    ///
    /// * `index` - Segment index (used for Sequential strategy).
    /// * `timestamp_ms` - Timestamp in milliseconds (used for Timestamp strategy).
    /// * `format` - Segment format to determine extension.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn segment_name(&self, index: u64, timestamp_ms: u64, format: SegmentFormat) -> String {
        let variable = match self.strategy {
            NamingStrategy::Sequential => {
                if self.zero_pad > 0 {
                    format!("{:0>width$}", index, width = self.zero_pad)
                } else {
                    index.to_string()
                }
            }
            NamingStrategy::Timestamp => timestamp_ms.to_string(),
            NamingStrategy::Duration => {
                let secs = timestamp_ms as f64 / 1000.0;
                format!("t{secs:.3}")
            }
        };

        let ext = self
            .extension
            .as_deref()
            .unwrap_or_else(|| extension_for_format(format));

        format!("{}{}{}.{}", self.prefix, self.separator, variable, ext)
    }

    /// Generate an init segment file name.
    #[must_use]
    pub fn init_segment_name(&self, format: SegmentFormat) -> String {
        let ext = self
            .extension
            .as_deref()
            .unwrap_or_else(|| extension_for_format(format));
        format!("{}_init.{}", self.prefix, ext)
    }

    /// Validate the template.
    ///
    /// # Errors
    ///
    /// Returns an error if the prefix is empty.
    pub fn validate(&self) -> PackagerResult<()> {
        if self.prefix.is_empty() {
            return Err(PackagerError::InvalidConfig(
                "Segment name prefix must not be empty".into(),
            ));
        }
        // Check for path-unsafe characters
        if self.prefix.contains('/') || self.prefix.contains('\\') {
            return Err(PackagerError::InvalidConfig(
                "Segment name prefix must not contain path separators".into(),
            ));
        }
        Ok(())
    }
}

/// Get the file extension for a segment format.
#[must_use]
pub fn extension_for_format(format: SegmentFormat) -> &'static str {
    match format {
        SegmentFormat::MpegTs => "ts",
        SegmentFormat::Fmp4 => "m4s",
        SegmentFormat::Cmaf => "m4s",
    }
}

/// Build a directory structure path for a variant stream.
///
/// # Arguments
///
/// * `base_dir` - Base output directory.
/// * `variant_index` - Index of the variant in the ladder.
/// * `height` - Resolution height for labelling.
#[must_use]
pub fn variant_directory(base_dir: &str, variant_index: usize, height: u32) -> String {
    format!("{base_dir}/v{variant_index}_{height}p")
}

/// Generate a manifest file name.
///
/// # Arguments
///
/// * `base_name` - Base name without extension.
/// * `is_hls` - `true` for HLS (`.m3u8`), `false` for DASH (`.mpd`).
#[must_use]
pub fn manifest_name(base_name: &str, is_hls: bool) -> String {
    let ext = if is_hls { "m3u8" } else { "mpd" };
    format!("{base_name}.{ext}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_template() {
        let t = NamingTemplate::default();
        assert_eq!(t.prefix, "segment");
        assert_eq!(t.strategy, NamingStrategy::Sequential);
    }

    #[test]
    fn test_sequential_naming() {
        let t = NamingTemplate::default();
        let name = t.segment_name(5, 0, SegmentFormat::Fmp4);
        assert_eq!(name, "segment_5.m4s");
    }

    #[test]
    fn test_sequential_zero_padded() {
        let t = NamingTemplate::default().with_zero_pad(5);
        let name = t.segment_name(42, 0, SegmentFormat::Fmp4);
        assert_eq!(name, "segment_00042.m4s");
    }

    #[test]
    fn test_timestamp_naming() {
        let t = NamingTemplate::default().with_strategy(NamingStrategy::Timestamp);
        let name = t.segment_name(0, 6000, SegmentFormat::MpegTs);
        assert_eq!(name, "segment_6000.ts");
    }

    #[test]
    fn test_duration_naming() {
        let t = NamingTemplate::default().with_strategy(NamingStrategy::Duration);
        let name = t.segment_name(0, 12500, SegmentFormat::Cmaf);
        assert_eq!(name, "segment_t12.500.m4s");
    }

    #[test]
    fn test_custom_prefix() {
        let t = NamingTemplate::with_prefix("video");
        let name = t.segment_name(0, 0, SegmentFormat::MpegTs);
        assert_eq!(name, "video_0.ts");
    }

    #[test]
    fn test_custom_extension() {
        let t = NamingTemplate::default().with_extension("mp4");
        let name = t.segment_name(0, 0, SegmentFormat::Fmp4);
        assert_eq!(name, "segment_0.mp4");
    }

    #[test]
    fn test_init_segment_name() {
        let t = NamingTemplate::with_prefix("stream");
        let name = t.init_segment_name(SegmentFormat::Fmp4);
        assert_eq!(name, "stream_init.m4s");
    }

    #[test]
    fn test_validate_empty_prefix() {
        let t = NamingTemplate {
            prefix: String::new(),
            ..Default::default()
        };
        assert!(t.validate().is_err());
    }

    #[test]
    fn test_validate_path_separator() {
        let t = NamingTemplate {
            prefix: "foo/bar".to_string(),
            ..Default::default()
        };
        assert!(t.validate().is_err());
    }

    #[test]
    fn test_extension_for_format() {
        assert_eq!(extension_for_format(SegmentFormat::MpegTs), "ts");
        assert_eq!(extension_for_format(SegmentFormat::Fmp4), "m4s");
        assert_eq!(extension_for_format(SegmentFormat::Cmaf), "m4s");
    }

    #[test]
    fn test_variant_directory() {
        let dir = variant_directory("/output", 2, 720);
        assert_eq!(dir, "/output/v2_720p");
    }

    #[test]
    fn test_manifest_name_hls() {
        assert_eq!(manifest_name("master", true), "master.m3u8");
    }

    #[test]
    fn test_manifest_name_dash() {
        assert_eq!(manifest_name("manifest", false), "manifest.mpd");
    }

    #[test]
    fn test_custom_separator() {
        let t = NamingTemplate::default().with_separator("-");
        let name = t.segment_name(3, 0, SegmentFormat::MpegTs);
        assert_eq!(name, "segment-3.ts");
    }
}
