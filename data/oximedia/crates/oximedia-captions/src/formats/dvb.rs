//! DVB subtitle format (MPEG-TS)

use crate::error::{CaptionError, Result};
use crate::formats::{FormatParser, FormatWriter};
use crate::types::CaptionTrack;

/// DVB subtitle format parser
pub struct DvbParser;

impl FormatParser for DvbParser {
    fn parse(&self, _data: &[u8]) -> Result<CaptionTrack> {
        Err(CaptionError::FeatureNotEnabled(
            "DVB parsing requires 'broadcast' feature".to_string(),
        ))
    }
}

/// DVB subtitle format writer
pub struct DvbWriter;

impl FormatWriter for DvbWriter {
    fn write(&self, _track: &CaptionTrack) -> Result<Vec<u8>> {
        Err(CaptionError::FeatureNotEnabled(
            "DVB writing requires 'broadcast' feature".to_string(),
        ))
    }
}
