//! Caption import functionality

use crate::error::{CaptionError, Result};
use crate::formats::{detect_format, get_parser};
use crate::types::{Caption, CaptionTrack, Language};
use crate::CaptionFormat;
use std::path::Path;

/// Caption importer
pub struct Importer;

impl Importer {
    /// Import a caption track from bytes
    pub fn import(data: &[u8], format: CaptionFormat) -> Result<CaptionTrack> {
        if let Some(parser) = get_parser(format) {
            parser.parse(data)
        } else {
            Err(CaptionError::UnsupportedFormat(format!("{format:?}")))
        }
    }

    /// Import from a file
    pub fn import_from_file(path: &Path, format: Option<CaptionFormat>) -> Result<CaptionTrack> {
        let data = std::fs::read(path)
            .map_err(|e| CaptionError::Import(format!("Failed to read file: {e}")))?;

        let format = if let Some(fmt) = format {
            fmt
        } else {
            Self::detect_format_from_file(path, &data)?
        };

        Self::import(&data, format)
    }

    /// Auto-detect format from file content
    pub fn import_auto(data: &[u8]) -> Result<CaptionTrack> {
        let format = detect_format(data)
            .ok_or_else(|| CaptionError::Import("Could not detect caption format".to_string()))?;

        Self::import(data, format)
    }

    /// Detect format from file extension and content
    pub fn detect_format_from_file(path: &Path, data: &[u8]) -> Result<CaptionFormat> {
        // Try extension first
        if let Some(format) = Self::detect_format_from_extension(path) {
            return Ok(format);
        }

        // Fall back to content detection
        detect_format(data)
            .ok_or_else(|| CaptionError::Import("Could not determine caption format".to_string()))
    }

    /// Detect format from file extension
    #[must_use]
    pub fn detect_format_from_extension(path: &Path) -> Option<CaptionFormat> {
        path.extension()?
            .to_str()
            .and_then(|ext| match ext.to_lowercase().as_str() {
                "srt" => Some(CaptionFormat::Srt),
                "vtt" => Some(CaptionFormat::WebVtt),
                "ass" => Some(CaptionFormat::Ass),
                "ssa" => Some(CaptionFormat::Ssa),
                "ttml" => Some(CaptionFormat::Ttml),
                "dfxp" => Some(CaptionFormat::Dfxp),
                "scc" => Some(CaptionFormat::Scc),
                "stl" => Some(CaptionFormat::EbuStl),
                "itt" => Some(CaptionFormat::ITt),
                _ => None,
            })
    }

    /// Detect encoding of caption file
    #[must_use]
    pub fn detect_encoding(data: &[u8]) -> &'static str {
        // Check for BOM
        if data.starts_with(&[0xEF, 0xBB, 0xBF]) {
            return "UTF-8";
        }
        if data.starts_with(&[0xFF, 0xFE]) {
            return "UTF-16LE";
        }
        if data.starts_with(&[0xFE, 0xFF]) {
            return "UTF-16BE";
        }

        // Try to decode as UTF-8
        if std::str::from_utf8(data).is_ok() {
            return "UTF-8";
        }

        // Assume Latin-1 as fallback
        "Latin-1"
    }
}

/// Import options
#[derive(Debug, Clone, Default)]
pub struct ImportOptions {
    /// Force a specific encoding
    pub encoding: Option<String>,
    /// Skip invalid captions
    pub skip_invalid: bool,
    /// Merge overlapping captions
    pub merge_overlaps: bool,
}

// ============================================================================
// Bulk / arena-style Caption builder (Wave 14 Slice H)
// ============================================================================

/// A batch [`Caption`] accumulator backed by a single pre-reserved `Vec`.
///
/// Compared to pushing individual captions to an ad-hoc `Vec`, this type
/// calls `Vec::reserve` upfront when the expected number of captions is
/// known, which eliminates repeated reallocations as the collection grows and
/// reduces heap fragmentation for large tracks.
///
/// When the expected capacity is *unknown* up-front, individual [`push`]
/// calls still benefit from the exponential growth strategy of `Vec`, but
/// no separate heap object is created per caption (unlike a linked-list or
/// arena whose nodes are each heap-allocated individually).
///
/// [`push`]: CaptionBulkBuilder::push
pub struct CaptionBulkBuilder {
    captions: Vec<Caption>,
}

impl CaptionBulkBuilder {
    /// Create a new builder with no reserved capacity.
    #[must_use]
    pub fn new() -> Self {
        Self {
            captions: Vec::new(),
        }
    }

    /// Create a new builder and pre-allocate capacity for `n` captions.
    ///
    /// This is the primary performance-critical constructor: it issues a
    /// single large `malloc` rather than many small reallocations.
    #[must_use]
    pub fn with_capacity(n: usize) -> Self {
        Self {
            captions: Vec::with_capacity(n),
        }
    }

    /// Append a caption.  Returns `&mut Self` for a builder-style API.
    pub fn push(&mut self, caption: Caption) -> &mut Self {
        self.captions.push(caption);
        self
    }

    /// Return the number of captions accumulated so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.captions.len()
    }

    /// Return `true` if no captions have been accumulated yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.captions.is_empty()
    }

    /// Consume the builder and return the accumulated captions as a plain
    /// `Vec<Caption>`.
    #[must_use]
    pub fn into_vec(self) -> Vec<Caption> {
        self.captions
    }
}

impl Default for CaptionBulkBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Build a [`CaptionTrack`] from a batch of captions, using
/// [`CaptionBulkBuilder`] internally to pre-reserve capacity and reduce
/// heap allocation overhead.
///
/// Captions are sorted by start time inside [`CaptionTrack::add_caption`];
/// the bulk pre-allocation means that no reallocation occurs during the
/// collect phase for inputs up to `len(entries)` captions.
pub fn import_bulk(
    entries: impl IntoIterator<Item = Caption>,
    language: Language,
) -> Result<CaptionTrack> {
    let iter = entries.into_iter();
    let (lo, hi) = iter.size_hint();
    let cap = hi.unwrap_or(lo);
    let mut builder = CaptionBulkBuilder::with_capacity(cap);
    for caption in iter {
        builder.push(caption);
    }
    let captions = builder.into_vec();
    let mut track = CaptionTrack::new(language);
    for c in captions {
        track.add_caption(c)?;
    }
    Ok(track)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Timestamp;

    #[test]
    fn test_import_srt() {
        let srt = b"1\n00:00:01,000 --> 00:00:03,000\nTest caption\n\n";
        let track = Importer::import_auto(srt).expect("auto import should succeed");
        assert_eq!(track.captions.len(), 1);
        assert_eq!(track.captions[0].text, "Test caption");
    }

    #[test]
    fn test_import_webvtt() {
        let vtt = b"WEBVTT\n\n00:00:01.000 --> 00:00:03.000\nTest caption\n\n";
        let track = Importer::import_auto(vtt).expect("auto import should succeed");
        assert_eq!(track.captions.len(), 1);
        assert_eq!(track.captions[0].text, "Test caption");
    }

    #[test]
    fn test_format_detection() {
        let srt = b"1\n00:00:01,000 --> 00:00:03,000\nTest\n\n";
        assert_eq!(detect_format(srt), Some(CaptionFormat::Srt));

        let vtt = b"WEBVTT\n\n00:00:01.000 --> 00:00:03.000\nTest\n\n";
        assert_eq!(detect_format(vtt), Some(CaptionFormat::WebVtt));
    }

    #[test]
    fn test_encoding_detection() {
        let utf8 = b"Test string";
        assert_eq!(Importer::detect_encoding(utf8), "UTF-8");

        let utf8_bom = b"\xEF\xBB\xBFTest string";
        assert_eq!(Importer::detect_encoding(utf8_bom), "UTF-8");
    }

    // Wave 14 Slice H — new tests

    /// CaptionBulkBuilder must accumulate N captions and return them all.
    #[test]
    fn test_arena_caption_builder_batch() {
        let n = 1_000usize;
        let mut builder = CaptionBulkBuilder::with_capacity(n);
        for i in 0..n {
            let start = Timestamp::from_millis((i * 5_000) as i64);
            let end = Timestamp::from_millis((i * 5_000 + 3_000) as i64);
            builder.push(Caption::new(start, end, format!("Caption {i}")));
        }
        let vec = builder.into_vec();
        assert_eq!(vec.len(), n, "expected {n} captions from bulk builder");
        // Smoke-check no panic on any caption
        for (i, c) in vec.iter().enumerate() {
            assert!(
                c.text.contains(&i.to_string()),
                "text should contain index {i}"
            );
        }
    }

    /// `import_bulk` produces a CaptionTrack with the correct caption count.
    #[test]
    fn test_import_bulk_track() {
        let n = 500usize;
        let captions = (0..n).map(|i| {
            Caption::new(
                Timestamp::from_millis((i * 4_000) as i64),
                Timestamp::from_millis((i * 4_000 + 2_000) as i64),
                format!("Bulk caption {i}"),
            )
        });
        let track = import_bulk(captions, Language::english()).expect("import_bulk should succeed");
        assert_eq!(track.captions.len(), n, "expected {n} captions in track");
    }
}
