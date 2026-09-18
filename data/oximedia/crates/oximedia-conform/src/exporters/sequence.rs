//! Sequence exporter for concatenated media output.

use crate::error::{ConformError, ConformResult};
use crate::exporters::Exporter;
use crate::types::{ClipMatch, OutputFormat};
use std::path::Path;

/// Sequence exporter for creating concatenated output.
pub struct SequenceExporter {
    /// Matched clips to export.
    #[allow(dead_code)]
    clips: Vec<ClipMatch>,
}

impl SequenceExporter {
    /// Create a new sequence exporter.
    #[must_use]
    pub fn new(clips: Vec<ClipMatch>) -> Self {
        Self { clips }
    }

    /// Export as concatenated video file.
    fn export_concatenated<P: AsRef<Path>>(
        &self,
        _output_path: P,
        _format: OutputFormat,
    ) -> ConformResult<()> {
        // Placeholder: would use oximedia-transcode to concatenate clips
        Ok(())
    }

    /// Export as frame sequence.
    fn export_frame_sequence<P: AsRef<Path>>(
        &self,
        _output_path: P,
        _format: OutputFormat,
    ) -> ConformResult<()> {
        // Placeholder: would render frames
        Ok(())
    }
}

impl Exporter for SequenceExporter {
    fn export<P: AsRef<Path>>(&self, output_path: P, format: OutputFormat) -> ConformResult<()> {
        match format {
            OutputFormat::Mp4 | OutputFormat::Matroska => {
                self.export_concatenated(output_path, format)
            }
            OutputFormat::FrameSequenceDpx
            | OutputFormat::FrameSequenceTiff
            | OutputFormat::FrameSequencePng => self.export_frame_sequence(output_path, format),
            _ => Err(ConformError::UnsupportedFormat(format!("{format}"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequence_exporter_creation() {
        let exporter = SequenceExporter::new(Vec::new());
        assert_eq!(exporter.clips.len(), 0);
    }
}
