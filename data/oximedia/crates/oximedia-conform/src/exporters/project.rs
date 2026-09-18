//! Project exporter for EDL/XML/AAF output.

use crate::error::{ConformError, ConformResult};
use crate::exporters::Exporter;
use crate::types::{ClipMatch, OutputFormat};
use std::path::Path;

/// Project exporter for timeline formats.
pub struct ProjectExporter {
    /// Matched clips to export.
    #[allow(dead_code)]
    clips: Vec<ClipMatch>,
}

impl ProjectExporter {
    /// Create a new project exporter.
    #[must_use]
    pub fn new(clips: Vec<ClipMatch>) -> Self {
        Self { clips }
    }

    /// Export as EDL.
    fn export_edl<P: AsRef<Path>>(&self, _output_path: P) -> ConformResult<()> {
        // Placeholder: would use oximedia-edl to generate EDL
        Ok(())
    }

    /// Export as FCP XML.
    fn export_fcpxml<P: AsRef<Path>>(&self, _output_path: P) -> ConformResult<()> {
        // Placeholder: would generate FCP XML
        Ok(())
    }

    /// Export as Premiere XML.
    fn export_premiere_xml<P: AsRef<Path>>(&self, _output_path: P) -> ConformResult<()> {
        // Placeholder: would generate Premiere XML
        Ok(())
    }

    /// Export as AAF.
    fn export_aaf<P: AsRef<Path>>(&self, _output_path: P) -> ConformResult<()> {
        // Placeholder: would use oximedia-aaf to generate AAF
        Ok(())
    }
}

impl Exporter for ProjectExporter {
    fn export<P: AsRef<Path>>(&self, output_path: P, format: OutputFormat) -> ConformResult<()> {
        match format {
            OutputFormat::Edl => self.export_edl(output_path),
            OutputFormat::FcpXml => self.export_fcpxml(output_path),
            OutputFormat::PremiereXml => self.export_premiere_xml(output_path),
            OutputFormat::Aaf => self.export_aaf(output_path),
            _ => Err(ConformError::UnsupportedFormat(format!("{format}"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_project_exporter_creation() {
        let exporter = ProjectExporter::new(Vec::new());
        assert_eq!(exporter.clips.len(), 0);
    }
}
