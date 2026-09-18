//! CEA-708 closed caption format (ATSC)

use crate::error::{CaptionError, Result};
use crate::formats::{FormatParser, FormatWriter};
use crate::types::CaptionTrack;

/// CEA-708 format parser
pub struct Cea708Parser;

impl FormatParser for Cea708Parser {
    fn parse(&self, _data: &[u8]) -> Result<CaptionTrack> {
        Err(CaptionError::FeatureNotEnabled(
            "CEA-708 parsing requires 'cea' feature".to_string(),
        ))
    }
}

/// CEA-708 format writer
pub struct Cea708Writer;

impl FormatWriter for Cea708Writer {
    fn write(&self, _track: &CaptionTrack) -> Result<Vec<u8>> {
        Err(CaptionError::FeatureNotEnabled(
            "CEA-708 writing requires 'cea' feature".to_string(),
        ))
    }
}

/// CEA-708 service numbers (up to 8 services)
#[allow(dead_code)]
pub const MAX_SERVICES: usize = 8;
