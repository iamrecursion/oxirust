//! Presentation Graphic Stream (PGS) - Blu-ray subtitles

use crate::error::{CaptionError, Result};
use crate::formats::FormatParser;
use crate::types::CaptionTrack;

/// PGS format parser
pub struct PgsParser;

impl FormatParser for PgsParser {
    fn parse(&self, _data: &[u8]) -> Result<CaptionTrack> {
        // PGS is a graphic subtitle format (bitmap-based)
        Err(CaptionError::UnsupportedFormat(
            "PGS is a graphic format - text extraction not supported".to_string(),
        ))
    }
}
