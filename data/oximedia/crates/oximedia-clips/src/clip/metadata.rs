//! Clip metadata management.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Extended metadata for a clip.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClipMetadata {
    /// Camera make.
    pub camera_make: Option<String>,

    /// Camera model.
    pub camera_model: Option<String>,

    /// Lens information.
    pub lens: Option<String>,

    /// ISO setting.
    pub iso: Option<u32>,

    /// Shutter speed (e.g., "1/60").
    pub shutter_speed: Option<String>,

    /// Aperture (e.g., "f/2.8").
    pub aperture: Option<String>,

    /// White balance.
    pub white_balance: Option<String>,

    /// Scene/shot number.
    pub scene_number: Option<String>,

    /// Take number.
    pub take_number: Option<u32>,

    /// Camera angle.
    pub camera_angle: Option<String>,

    /// Shooting location.
    pub location: Option<String>,

    /// Director name.
    pub director: Option<String>,

    /// Cinematographer name.
    pub cinematographer: Option<String>,

    /// Production name.
    pub production: Option<String>,

    /// Shooting date.
    pub shoot_date: Option<DateTime<Utc>>,

    /// Copyright information.
    pub copyright: Option<String>,

    /// Custom fields.
    pub custom: HashMap<String, String>,
}

impl ClipMetadata {
    /// Creates new empty metadata.
    #[must_use]
    pub fn new() -> Self {
        Self {
            camera_make: None,
            camera_model: None,
            lens: None,
            iso: None,
            shutter_speed: None,
            aperture: None,
            white_balance: None,
            scene_number: None,
            take_number: None,
            camera_angle: None,
            location: None,
            director: None,
            cinematographer: None,
            production: None,
            shoot_date: None,
            copyright: None,
            custom: HashMap::new(),
        }
    }

    /// Sets a custom field.
    pub fn set_custom(&mut self, key: String, value: String) {
        self.custom.insert(key, value);
    }

    /// Gets a custom field.
    #[must_use]
    pub fn get_custom(&self, key: &str) -> Option<&str> {
        self.custom.get(key).map(String::as_str)
    }

    /// Removes a custom field.
    pub fn remove_custom(&mut self, key: &str) -> Option<String> {
        self.custom.remove(key)
    }
}

impl Default for ClipMetadata {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_creation() {
        let metadata = ClipMetadata::new();
        assert!(metadata.camera_make.is_none());
        assert!(metadata.custom.is_empty());
    }

    #[test]
    fn test_custom_fields() {
        let mut metadata = ClipMetadata::new();
        metadata.set_custom("color_space".to_string(), "Rec.709".to_string());
        assert_eq!(metadata.get_custom("color_space"), Some("Rec.709"));

        let removed = metadata.remove_custom("color_space");
        assert_eq!(removed, Some("Rec.709".to_string()));
        assert!(metadata.get_custom("color_space").is_none());
    }
}
