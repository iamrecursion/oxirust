//! Per-angle metadata tracking.

use super::MetadataEntry;
use crate::{AngleId, FrameNumber};
use std::collections::HashMap;

/// Angle metadata
#[derive(Debug, Clone)]
pub struct AngleMetadata {
    /// Angle identifier
    pub angle: AngleId,
    /// Camera name
    pub camera_name: String,
    /// Lens information
    pub lens_info: String,
    /// Recording format
    pub format: String,
    /// Frame rate
    pub frame_rate: f64,
    /// Resolution (width, height)
    pub resolution: (u32, u32),
    /// Custom metadata
    pub custom: HashMap<String, String>,
}

impl AngleMetadata {
    /// Create new angle metadata
    #[must_use]
    pub fn new(angle: AngleId) -> Self {
        Self {
            angle,
            camera_name: String::new(),
            lens_info: String::new(),
            format: String::new(),
            frame_rate: 25.0,
            resolution: (1920, 1080),
            custom: HashMap::new(),
        }
    }

    /// Set custom metadata
    pub fn set_custom(&mut self, key: String, value: String) {
        self.custom.insert(key, value);
    }

    /// Get custom metadata
    #[must_use]
    pub fn get_custom(&self, key: &str) -> Option<&String> {
        self.custom.get(key)
    }

    /// Remove custom metadata
    pub fn remove_custom(&mut self, key: &str) -> Option<String> {
        self.custom.remove(key)
    }
}

/// Metadata tracker
#[derive(Debug)]
pub struct MetadataTracker {
    /// Angle metadata
    angle_metadata: Vec<AngleMetadata>,
    /// Per-frame metadata entries
    frame_metadata: Vec<MetadataEntry>,
}

impl MetadataTracker {
    /// Create a new metadata tracker
    #[must_use]
    pub fn new(angle_count: usize) -> Self {
        Self {
            angle_metadata: (0..angle_count).map(AngleMetadata::new).collect(),
            frame_metadata: Vec::new(),
        }
    }

    /// Set angle metadata
    pub fn set_angle_metadata(&mut self, metadata: AngleMetadata) {
        if metadata.angle < self.angle_metadata.len() {
            self.angle_metadata[metadata.angle] = metadata.clone();
        }
    }

    /// Get angle metadata
    #[must_use]
    pub fn get_angle_metadata(&self, angle: AngleId) -> Option<&AngleMetadata> {
        self.angle_metadata.get(angle)
    }

    /// Get mutable angle metadata
    pub fn get_angle_metadata_mut(&mut self, angle: AngleId) -> Option<&mut AngleMetadata> {
        self.angle_metadata.get_mut(angle)
    }

    /// Add frame metadata
    pub fn add_frame_metadata(&mut self, entry: MetadataEntry) {
        self.frame_metadata.push(entry);
    }

    /// Get frame metadata for angle at frame
    #[must_use]
    pub fn get_frame_metadata(&self, angle: AngleId, frame: FrameNumber) -> Vec<&MetadataEntry> {
        self.frame_metadata
            .iter()
            .filter(|e| e.angle == angle && e.frame == frame)
            .collect()
    }

    /// Get all frame metadata for angle
    #[must_use]
    pub fn get_angle_frame_metadata(&self, angle: AngleId) -> Vec<&MetadataEntry> {
        self.frame_metadata
            .iter()
            .filter(|e| e.angle == angle)
            .collect()
    }

    /// Clear frame metadata
    pub fn clear_frame_metadata(&mut self) {
        self.frame_metadata.clear();
    }

    /// Export metadata as JSON
    #[must_use]
    pub fn export_json(&self) -> String {
        let mut json = String::from("{\n");
        json.push_str("  \"angles\": [\n");

        for (i, metadata) in self.angle_metadata.iter().enumerate() {
            json.push_str("    {\n");
            json.push_str(&format!("      \"angle\": {},\n", metadata.angle));
            json.push_str(&format!(
                "      \"camera_name\": \"{}\",\n",
                metadata.camera_name
            ));
            json.push_str(&format!(
                "      \"lens_info\": \"{}\",\n",
                metadata.lens_info
            ));
            json.push_str(&format!("      \"format\": \"{}\",\n", metadata.format));
            json.push_str(&format!("      \"frame_rate\": {},\n", metadata.frame_rate));
            json.push_str(&format!(
                "      \"resolution\": [{}, {}]\n",
                metadata.resolution.0, metadata.resolution.1
            ));
            json.push('}');

            if i < self.angle_metadata.len() - 1 {
                json.push(',');
            }
            json.push('\n');
        }

        json.push_str("  ]\n");
        json.push('}');
        json
    }

    /// Get total metadata count
    #[must_use]
    pub fn metadata_count(&self) -> usize {
        self.frame_metadata.len()
    }

    /// Find metadata by key
    #[must_use]
    pub fn find_by_key(&self, key: &str) -> Vec<&MetadataEntry> {
        self.frame_metadata
            .iter()
            .filter(|e| e.key == key)
            .collect()
    }

    /// Find metadata by value
    #[must_use]
    pub fn find_by_value(&self, value: &str) -> Vec<&MetadataEntry> {
        self.frame_metadata
            .iter()
            .filter(|e| e.value == value)
            .collect()
    }
}

/// Metadata query builder
#[derive(Debug)]
pub struct MetadataQuery {
    /// Target angle
    angle: Option<AngleId>,
    /// Target frame
    frame: Option<FrameNumber>,
    /// Key filter
    key: Option<String>,
    /// Value filter
    value: Option<String>,
}

impl MetadataQuery {
    /// Create a new query
    #[must_use]
    pub fn new() -> Self {
        Self {
            angle: None,
            frame: None,
            key: None,
            value: None,
        }
    }

    /// Filter by angle
    #[must_use]
    pub fn angle(mut self, angle: AngleId) -> Self {
        self.angle = Some(angle);
        self
    }

    /// Filter by frame
    #[must_use]
    pub fn frame(mut self, frame: FrameNumber) -> Self {
        self.frame = Some(frame);
        self
    }

    /// Filter by key
    #[must_use]
    pub fn key(mut self, key: String) -> Self {
        self.key = Some(key);
        self
    }

    /// Filter by value
    #[must_use]
    pub fn value(mut self, value: String) -> Self {
        self.value = Some(value);
        self
    }

    /// Execute query
    #[must_use]
    pub fn execute<'a>(&self, tracker: &'a MetadataTracker) -> Vec<&'a MetadataEntry> {
        tracker
            .frame_metadata
            .iter()
            .filter(|e| {
                if let Some(angle) = self.angle {
                    if e.angle != angle {
                        return false;
                    }
                }
                if let Some(frame) = self.frame {
                    if e.frame != frame {
                        return false;
                    }
                }
                if let Some(ref key) = self.key {
                    if &e.key != key {
                        return false;
                    }
                }
                if let Some(ref value) = self.value {
                    if &e.value != value {
                        return false;
                    }
                }
                true
            })
            .collect()
    }
}

impl Default for MetadataQuery {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_angle_metadata_creation() {
        let metadata = AngleMetadata::new(0);
        assert_eq!(metadata.angle, 0);
        assert_eq!(metadata.frame_rate, 25.0);
        assert_eq!(metadata.resolution, (1920, 1080));
    }

    #[test]
    fn test_custom_metadata() {
        let mut metadata = AngleMetadata::new(0);
        metadata.set_custom("iso".to_string(), "800".to_string());
        assert_eq!(metadata.get_custom("iso"), Some(&"800".to_string()));

        metadata.remove_custom("iso");
        assert_eq!(metadata.get_custom("iso"), None);
    }

    #[test]
    fn test_tracker_creation() {
        let tracker = MetadataTracker::new(3);
        assert_eq!(tracker.angle_metadata.len(), 3);
        assert_eq!(tracker.metadata_count(), 0);
    }

    #[test]
    fn test_add_frame_metadata() {
        let mut tracker = MetadataTracker::new(2);
        let entry = MetadataEntry::new(0, 100, "focus".to_string(), "infinity".to_string());

        tracker.add_frame_metadata(entry);
        assert_eq!(tracker.metadata_count(), 1);
    }

    #[test]
    fn test_get_frame_metadata() {
        let mut tracker = MetadataTracker::new(2);
        let entry1 = MetadataEntry::new(0, 100, "focus".to_string(), "infinity".to_string());
        let entry2 = MetadataEntry::new(0, 100, "aperture".to_string(), "f/2.8".to_string());
        let entry3 = MetadataEntry::new(0, 200, "focus".to_string(), "close".to_string());

        tracker.add_frame_metadata(entry1);
        tracker.add_frame_metadata(entry2);
        tracker.add_frame_metadata(entry3);

        let metadata = tracker.get_frame_metadata(0, 100);
        assert_eq!(metadata.len(), 2);
    }

    #[test]
    fn test_find_by_key() {
        let mut tracker = MetadataTracker::new(2);
        let entry1 = MetadataEntry::new(0, 100, "focus".to_string(), "infinity".to_string());
        let entry2 = MetadataEntry::new(1, 100, "focus".to_string(), "close".to_string());

        tracker.add_frame_metadata(entry1);
        tracker.add_frame_metadata(entry2);

        let results = tracker.find_by_key("focus");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_metadata_query() {
        let mut tracker = MetadataTracker::new(2);
        let entry1 = MetadataEntry::new(0, 100, "focus".to_string(), "infinity".to_string());
        let entry2 = MetadataEntry::new(0, 200, "focus".to_string(), "close".to_string());
        let entry3 = MetadataEntry::new(1, 100, "focus".to_string(), "infinity".to_string());

        tracker.add_frame_metadata(entry1);
        tracker.add_frame_metadata(entry2);
        tracker.add_frame_metadata(entry3);

        let query = MetadataQuery::new().angle(0).key("focus".to_string());

        let results = query.execute(&tracker);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_export_json() {
        let tracker = MetadataTracker::new(2);
        let json = tracker.export_json();
        assert!(json.contains("\"angles\""));
        assert!(json.contains("\"angle\": 0"));
    }
}
