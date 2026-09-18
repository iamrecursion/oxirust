//! iTunes Timed Text (iTT) format - XML-based format similar to TTML

use crate::error::Result;
use crate::formats::{FormatParser, FormatWriter};
use crate::types::CaptionTrack;

/// iTT format parser (similar to TTML)
pub struct IttParser;

impl FormatParser for IttParser {
    fn parse(&self, data: &[u8]) -> Result<CaptionTrack> {
        // iTT is very similar to TTML, reuse TTML parser
        let ttml_parser = super::ttml::TtmlParser;
        ttml_parser.parse(data)
    }
}

/// iTT format writer
pub struct IttWriter;

impl FormatWriter for IttWriter {
    fn write(&self, track: &CaptionTrack) -> Result<Vec<u8>> {
        let ttml_writer = super::ttml::TtmlWriter;
        ttml_writer.write(track)
    }
}
