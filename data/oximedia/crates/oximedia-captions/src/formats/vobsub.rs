//! `VobSub` (DVD subtitles) format

use crate::error::{CaptionError, Result};
use crate::formats::FormatParser;
use crate::types::CaptionTrack;

/// `VobSub` format parser
pub struct VobSubParser;

impl FormatParser for VobSubParser {
    fn parse(&self, _data: &[u8]) -> Result<CaptionTrack> {
        // VobSub is a graphic subtitle format (bitmap-based)
        Err(CaptionError::UnsupportedFormat(
            "VobSub is a graphic format - text extraction not supported".to_string(),
        ))
    }
}
