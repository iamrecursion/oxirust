//! DFXP (Distribution Format Exchange Profile) format - alias for TTML

use crate::error::Result;
use crate::formats::{FormatParser, FormatWriter};
use crate::types::CaptionTrack;

/// DFXP is essentially TTML, so we reuse the TTML parser
pub struct DfxpParser;

impl FormatParser for DfxpParser {
    fn parse(&self, data: &[u8]) -> Result<CaptionTrack> {
        let ttml_parser = super::ttml::TtmlParser;
        ttml_parser.parse(data)
    }
}

/// DFXP writer (uses TTML)
pub struct DfxpWriter;

impl FormatWriter for DfxpWriter {
    fn write(&self, track: &CaptionTrack) -> Result<Vec<u8>> {
        let ttml_writer = super::ttml::TtmlWriter;
        ttml_writer.write(track)
    }
}
