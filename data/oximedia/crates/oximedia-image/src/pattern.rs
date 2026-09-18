//! Sequence pattern matching for image files.

use crate::error::{ImageError, ImageResult};
use std::path::{Path, PathBuf};

/// Pattern for matching image sequence filenames.
///
/// Supports common naming conventions:
/// - Printf-style: `render.%04d.exr` → `render.0001.exr`
/// - Hash notation: `render.####.exr` → `render.0001.exr`
/// - Direct frame numbers in brackets: `render.[1-100].exr`
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SequencePattern {
    /// Base path (directory + prefix).
    pub base: PathBuf,

    /// Pattern type and formatting.
    pub pattern: PatternType,

    /// File extension.
    pub extension: String,
}

/// Type of sequence pattern.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PatternType {
    /// Printf-style format (e.g., %04d).
    Printf { width: usize, padding: char },

    /// Hash notation (e.g., ####).
    Hash { count: usize },

    /// Single frame (no sequence).
    Single,
}

impl SequencePattern {
    /// Parses a sequence pattern from a string.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_image::SequencePattern;
    ///
    /// let pattern = SequencePattern::parse("render.%04d.dpx").expect("valid pattern");
    /// let filename = pattern.format(42);
    /// assert!(filename.ends_with("render.0042.dpx"));
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The pattern has no file extension
    /// - The filename is invalid
    /// - The pattern syntax is invalid (e.g., malformed printf or hash notation)
    pub fn parse(pattern: &str) -> ImageResult<Self> {
        let path = Path::new(pattern);

        // Extract extension
        let extension = path
            .extension()
            .and_then(|e| e.to_str())
            .ok_or_else(|| ImageError::InvalidPattern("No file extension found".to_string()))?
            .to_string();

        // Get filename without extension
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| ImageError::InvalidPattern("Invalid filename".to_string()))?;

        // Parse pattern type
        let pattern_type = if stem.contains("%d") || stem.contains("%0") {
            Self::parse_printf(stem)?
        } else if stem.contains('#') {
            Self::parse_hash(stem)?
        } else {
            PatternType::Single
        };

        // Get base path (directory + prefix before pattern)
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        let prefix = if let PatternType::Single = pattern_type {
            stem.to_string()
        } else {
            Self::extract_prefix(stem)
        };

        let base = parent.join(prefix);

        Ok(Self {
            base,
            pattern: pattern_type,
            extension,
        })
    }

    /// Formats a frame number according to this pattern.
    #[must_use]
    pub fn format(&self, frame: u32) -> PathBuf {
        let formatted = match &self.pattern {
            PatternType::Printf { width, padding: _ } => {
                format!("{frame:0width$}")
            }
            PatternType::Hash { count } => {
                format!("{frame:0count$}")
            }
            PatternType::Single => String::new(),
        };

        if formatted.is_empty() {
            PathBuf::from(format!("{}.{}", self.base.display(), self.extension))
        } else {
            PathBuf::from(format!(
                "{}{}.{}",
                self.base.display(),
                formatted,
                self.extension
            ))
        }
    }

    /// Extracts frame number from a filename matching this pattern.
    #[must_use]
    pub fn extract_frame(&self, path: &Path) -> Option<u32> {
        let stem = path.file_stem()?.to_str()?;
        let base_str = self.base.file_name()?.to_str()?;

        if !stem.starts_with(base_str) {
            return None;
        }

        let frame_str = &stem[base_str.len()..];
        frame_str.parse().ok()
    }

    fn parse_printf(stem: &str) -> ImageResult<PatternType> {
        // Look for %0Nd or %d patterns
        if let Some(pos) = stem.find('%') {
            let rest = &stem[pos + 1..];

            if rest.starts_with('d') {
                return Ok(PatternType::Printf {
                    width: 1,
                    padding: '0',
                });
            }

            if let Some(stripped) = rest.strip_prefix('0') {
                // Parse width from %0Nd format
                let width_end = stripped
                    .find(|c: char| !c.is_ascii_digit())
                    .unwrap_or(stripped.len());
                let width_str = &stripped[..width_end];
                let width = width_str.parse::<usize>().map_err(|_| {
                    ImageError::InvalidPattern(format!("Invalid width: {width_str}"))
                })?;

                return Ok(PatternType::Printf {
                    width,
                    padding: '0',
                });
            }
        }

        Err(ImageError::InvalidPattern(
            "Invalid printf pattern".to_string(),
        ))
    }

    fn parse_hash(stem: &str) -> ImageResult<PatternType> {
        let count = stem.chars().filter(|&c| c == '#').count();
        if count == 0 {
            return Err(ImageError::InvalidPattern(
                "No # found in pattern".to_string(),
            ));
        }

        Ok(PatternType::Hash { count })
    }

    fn extract_prefix(stem: &str) -> String {
        // Find the pattern marker and extract everything before it
        if let Some(pos) = stem.find('%') {
            stem[..pos].to_string()
        } else if let Some(pos) = stem.find('#') {
            stem[..pos].to_string()
        } else {
            stem.to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_printf_pattern() {
        let pattern = SequencePattern::parse("render.%04d.dpx").expect("should succeed in test");
        assert_eq!(pattern.extension, "dpx");
        assert!(matches!(
            pattern.pattern,
            PatternType::Printf { width: 4, .. }
        ));
    }

    #[test]
    fn test_parse_hash_pattern() {
        let pattern = SequencePattern::parse("render.####.exr").expect("should succeed in test");
        assert_eq!(pattern.extension, "exr");
        assert!(matches!(pattern.pattern, PatternType::Hash { count: 4 }));
    }

    #[test]
    fn test_format_printf() {
        let pattern = SequencePattern::parse("render.%04d.dpx").expect("should succeed in test");
        let path = pattern.format(42);
        let filename = path
            .file_name()
            .expect("should succeed in test")
            .to_str()
            .expect("should succeed in test");
        assert!(filename.contains("0042"));
    }

    #[test]
    fn test_format_hash() {
        let pattern = SequencePattern::parse("shot.###.tif").expect("should succeed in test");
        let path = pattern.format(7);
        let filename = path
            .file_name()
            .expect("should succeed in test")
            .to_str()
            .expect("should succeed in test");
        assert!(filename.contains("007"));
    }

    #[test]
    fn test_extract_frame() {
        let pattern = SequencePattern::parse("render.%04d.dpx").expect("should succeed in test");
        let path = pattern.format(123);
        let frame = pattern.extract_frame(&path);
        assert_eq!(frame, Some(123));
    }
}
