//! ARIB subtitle format (Japan)

use crate::error::{CaptionError, Result};
use crate::formats::{FormatParser, FormatWriter};
use crate::types::CaptionTrack;

/// ARIB format parser
pub struct AribParser;

impl FormatParser for AribParser {
    fn parse(&self, _data: &[u8]) -> Result<CaptionTrack> {
        Err(CaptionError::FeatureNotEnabled(
            "ARIB parsing requires 'broadcast' feature".to_string(),
        ))
    }
}

/// ARIB format writer
pub struct AribWriter;

impl FormatWriter for AribWriter {
    fn write(&self, _track: &CaptionTrack) -> Result<Vec<u8>> {
        Err(CaptionError::FeatureNotEnabled(
            "ARIB writing requires 'broadcast' feature".to_string(),
        ))
    }
}
