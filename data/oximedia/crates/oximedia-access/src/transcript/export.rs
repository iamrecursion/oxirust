//! Transcript export functionality.

use crate::error::AccessResult;
use crate::transcript::{Transcript, TranscriptFormat, TranscriptFormatter};
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Exports transcripts to files.
pub struct TranscriptExporter;

impl TranscriptExporter {
    /// Export transcript to file.
    pub fn export<P: AsRef<Path>>(
        transcript: &Transcript,
        path: P,
        format: TranscriptFormat,
    ) -> AccessResult<()> {
        let content = TranscriptFormatter::format(transcript, format);

        let mut file = File::create(path)?;
        file.write_all(content.as_bytes())?;

        Ok(())
    }

    /// Export to string.
    #[must_use]
    pub fn export_to_string(transcript: &Transcript, format: TranscriptFormat) -> String {
        TranscriptFormatter::format(transcript, format)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::TranscriptEntry;

    #[test]
    fn test_export_to_string() {
        let mut transcript = Transcript::new();
        transcript.add_entry(TranscriptEntry::new(0, 1000, "Test".to_string()));

        let output = TranscriptExporter::export_to_string(&transcript, TranscriptFormat::Plain);
        assert!(!output.is_empty());
    }
}
