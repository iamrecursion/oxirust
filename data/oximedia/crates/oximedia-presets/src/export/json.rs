//! JSON preset export.

use crate::{Preset, PresetError, Result};
use std::fs;
use std::path::Path;

/// Export a preset to a JSON file.
pub fn export_to_file<P: AsRef<Path>>(preset: &Preset, path: P) -> Result<()> {
    let json = export_to_string(preset)?;
    fs::write(path, json)?;
    Ok(())
}

/// Export a preset to a JSON string.
pub fn export_to_string(preset: &Preset) -> Result<String> {
    serde_json::to_string_pretty(preset).map_err(PresetError::Json)
}

/// Export multiple presets to a JSON array string.
pub fn export_multiple_to_string(presets: &[Preset]) -> Result<String> {
    serde_json::to_string_pretty(presets).map_err(PresetError::Json)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PresetCategory, PresetMetadata};
    use oximedia_transcode::PresetConfig;

    #[test]
    fn test_export_to_string() {
        let metadata = PresetMetadata::new("test", "Test", PresetCategory::Custom);
        let config = PresetConfig::default();
        let preset = Preset::new(metadata, config);
        assert!(export_to_string(&preset).is_ok());
    }
}
