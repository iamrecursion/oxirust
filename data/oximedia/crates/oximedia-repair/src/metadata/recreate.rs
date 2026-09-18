//! Recreate missing metadata.
//!
//! This module provides functions to recreate metadata when it's completely missing.

use crate::Result;
use std::collections::HashMap;
use std::path::Path;

/// Recreate basic metadata for a file.
pub fn recreate_metadata(path: &Path) -> Result<HashMap<String, String>> {
    let mut metadata = HashMap::new();

    // Get file information
    let file_metadata = std::fs::metadata(path)?;

    // Add basic metadata
    metadata.insert("file_size".to_string(), file_metadata.len().to_string());

    if let Some(filename) = path.file_name() {
        metadata.insert(
            "filename".to_string(),
            filename.to_string_lossy().to_string(),
        );
    }

    // Infer format from extension
    if let Some(ext) = path.extension() {
        metadata.insert("format".to_string(), ext.to_string_lossy().to_string());
    }

    // Add creation time if available
    if let Ok(created) = file_metadata.created() {
        if let Ok(duration) = created.duration_since(std::time::UNIX_EPOCH) {
            metadata.insert("created".to_string(), duration.as_secs().to_string());
        }
    }

    Ok(metadata)
}

/// Create default metadata for a given format.
pub fn create_default_metadata(format: &str) -> HashMap<String, String> {
    let mut metadata = HashMap::new();

    metadata.insert("format".to_string(), format.to_string());
    metadata.insert("encoder".to_string(), "OxiMedia Repair".to_string());

    match format {
        "mp4" | "m4v" => {
            metadata.insert("brand".to_string(), "mp42".to_string());
        }
        "mkv" | "webm" => {
            metadata.insert("muxing_app".to_string(), "OxiMedia".to_string());
        }
        _ => {}
    }

    metadata
}

/// Estimate metadata from file content.
pub fn estimate_metadata(_data: &[u8]) -> HashMap<String, String> {
    let mut metadata = HashMap::new();

    // This would analyze the file content to estimate properties
    // like resolution, duration, codec, etc.

    metadata.insert("estimated".to_string(), "true".to_string());

    metadata
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_default_metadata() {
        let metadata = create_default_metadata("mp4");
        assert_eq!(metadata.get("format"), Some(&"mp4".to_string()));
        assert_eq!(metadata.get("brand"), Some(&"mp42".to_string()));
    }

    #[test]
    fn test_create_default_metadata_mkv() {
        let metadata = create_default_metadata("mkv");
        assert_eq!(metadata.get("format"), Some(&"mkv".to_string()));
    }
}
