//! Emulation preparation

use crate::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Emulation preparation information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmulationPreparation {
    /// Original format
    pub format: String,
    /// Required software
    pub required_software: Vec<String>,
    /// Required hardware specs
    pub required_hardware: Vec<String>,
    /// Configuration files
    pub config_files: Vec<PathBuf>,
    /// Documentation
    pub documentation: Vec<String>,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Emulation preparer
pub struct EmulationPreparer;

impl Default for EmulationPreparer {
    fn default() -> Self {
        Self::new()
    }
}

impl EmulationPreparer {
    /// Create a new emulation preparer
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Prepare emulation environment information
    #[must_use]
    pub fn prepare(&self, format: &str) -> EmulationPreparation {
        let (required_software, required_hardware, docs) = self.get_requirements(format);

        EmulationPreparation {
            format: format.to_string(),
            required_software,
            required_hardware,
            config_files: Vec::new(),
            documentation: docs,
            timestamp: chrono::Utc::now(),
        }
    }

    fn get_requirements(&self, format: &str) -> (Vec<String>, Vec<String>, Vec<String>) {
        match format.to_lowercase().as_str() {
            "mkv" | "webm" => (
                vec!["FFmpeg 4.0+".to_string(), "libmatroska".to_string()],
                vec!["CPU: x86_64".to_string(), "Memory: 512MB+".to_string()],
                vec!["Matroska specification".to_string()],
            ),
            "flac" => (
                vec!["libFLAC 1.3+".to_string()],
                vec!["CPU: Any".to_string(), "Memory: 256MB+".to_string()],
                vec!["FLAC format specification".to_string()],
            ),
            _ => (
                vec!["Unknown decoder".to_string()],
                vec!["Unknown requirements".to_string()],
                vec!["Format documentation needed".to_string()],
            ),
        }
    }

    /// Save preparation info to JSON
    ///
    /// # Errors
    ///
    /// Returns an error if save fails
    pub fn save(&self, prep: &EmulationPreparation, path: &PathBuf) -> Result<()> {
        let json = serde_json::to_string_pretty(prep)
            .map_err(|e| crate::Error::Metadata(format!("JSON serialization failed: {e}")))?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prepare_emulation() {
        let preparer = EmulationPreparer::new();
        let prep = preparer.prepare("mkv");

        assert_eq!(prep.format, "mkv");
        assert!(!prep.required_software.is_empty());
        assert!(!prep.required_hardware.is_empty());
    }
}
