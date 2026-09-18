//! JSON preset import.

use crate::{Preset, PresetError, Result};
use std::fs;
use std::path::Path;

/// Import a preset from a JSON file.
pub fn import_from_file<P: AsRef<Path>>(path: P) -> Result<Preset> {
    let content = fs::read_to_string(path)?;
    import_from_string(&content)
}

/// Import a preset from a JSON string.
pub fn import_from_string(json: &str) -> Result<Preset> {
    serde_json::from_str(json).map_err(PresetError::Json)
}

/// Import multiple presets from a JSON array.
pub fn import_multiple_from_string(json: &str) -> Result<Vec<Preset>> {
    serde_json::from_str(json).map_err(PresetError::Json)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_import_from_string() {
        let json = r#"{
            "metadata": {
                "id": "test",
                "name": "Test",
                "description": "",
                "category": "Custom",
                "tags": [],
                "version": "1.0.0",
                "author": "OxiMedia",
                "created": "2024-01-01T00:00:00Z",
                "modified": "2024-01-01T00:00:00Z",
                "official": false,
                "target": "",
                "use_cases": [],
                "limitations": []
            }
        }"#;
        assert!(import_from_string(json).is_ok());
    }
}
