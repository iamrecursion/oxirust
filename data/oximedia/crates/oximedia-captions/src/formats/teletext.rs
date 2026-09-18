//! Teletext subtitle format (EBU, BBC standards)

use crate::error::{CaptionError, Result};
use crate::formats::{FormatParser, FormatWriter};
use crate::types::CaptionTrack;

/// Teletext format parser
pub struct TeletextParser;

impl FormatParser for TeletextParser {
    fn parse(&self, _data: &[u8]) -> Result<CaptionTrack> {
        Err(CaptionError::FeatureNotEnabled(
            "Teletext parsing requires 'broadcast' feature".to_string(),
        ))
    }
}

/// Teletext format writer
pub struct TeletextWriter;

impl FormatWriter for TeletextWriter {
    fn write(&self, _track: &CaptionTrack) -> Result<Vec<u8>> {
        Err(CaptionError::FeatureNotEnabled(
            "Teletext writing requires 'broadcast' feature".to_string(),
        ))
    }
}
