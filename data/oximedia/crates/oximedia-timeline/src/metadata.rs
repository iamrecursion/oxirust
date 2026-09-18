//! Metadata management for timeline, clips, and tracks.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Metadata container for timeline entities.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Metadata {
    /// Key-value metadata pairs.
    data: HashMap<String, MetadataValue>,
}

/// Value types for metadata.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum MetadataValue {
    /// String value.
    String(String),
    /// Integer value.
    Int(i64),
    /// Float value.
    Float(f64),
    /// Boolean value.
    Bool(bool),
    /// List of strings.
    StringList(Vec<String>),
}

impl Metadata {
    /// Creates a new empty metadata container.
    #[must_use]
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    /// Sets a string metadata value.
    pub fn set_string(&mut self, key: String, value: String) {
        self.data.insert(key, MetadataValue::String(value));
    }

    /// Sets an integer metadata value.
    pub fn set_int(&mut self, key: String, value: i64) {
        self.data.insert(key, MetadataValue::Int(value));
    }

    /// Sets a float metadata value.
    pub fn set_float(&mut self, key: String, value: f64) {
        self.data.insert(key, MetadataValue::Float(value));
    }

    /// Sets a boolean metadata value.
    pub fn set_bool(&mut self, key: String, value: bool) {
        self.data.insert(key, MetadataValue::Bool(value));
    }

    /// Sets a string list metadata value.
    pub fn set_string_list(&mut self, key: String, value: Vec<String>) {
        self.data.insert(key, MetadataValue::StringList(value));
    }

    /// Gets a metadata value.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&MetadataValue> {
        self.data.get(key)
    }

    /// Gets a string metadata value.
    #[must_use]
    pub fn get_string(&self, key: &str) -> Option<&String> {
        match self.data.get(key) {
            Some(MetadataValue::String(s)) => Some(s),
            _ => None,
        }
    }

    /// Gets an integer metadata value.
    #[must_use]
    pub fn get_int(&self, key: &str) -> Option<i64> {
        match self.data.get(key) {
            Some(MetadataValue::Int(i)) => Some(*i),
            _ => None,
        }
    }

    /// Gets a float metadata value.
    #[must_use]
    pub fn get_float(&self, key: &str) -> Option<f64> {
        match self.data.get(key) {
            Some(MetadataValue::Float(f)) => Some(*f),
            _ => None,
        }
    }

    /// Gets a boolean metadata value.
    #[must_use]
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        match self.data.get(key) {
            Some(MetadataValue::Bool(b)) => Some(*b),
            _ => None,
        }
    }

    /// Gets a string list metadata value.
    #[must_use]
    pub fn get_string_list(&self, key: &str) -> Option<&Vec<String>> {
        match self.data.get(key) {
            Some(MetadataValue::StringList(list)) => Some(list),
            _ => None,
        }
    }

    /// Removes a metadata value.
    pub fn remove(&mut self, key: &str) -> Option<MetadataValue> {
        self.data.remove(key)
    }

    /// Checks if a key exists.
    #[must_use]
    pub fn contains_key(&self, key: &str) -> bool {
        self.data.contains_key(key)
    }

    /// Returns all keys.
    #[must_use]
    pub fn keys(&self) -> Vec<&String> {
        self.data.keys().collect()
    }

    /// Returns the number of metadata entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Checks if metadata is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Clears all metadata.
    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// Merges another metadata container into this one.
    pub fn merge(&mut self, other: &Self) {
        for (key, value) in &other.data {
            self.data.insert(key.clone(), value.clone());
        }
    }
}

/// Standard metadata keys for media assets.
pub mod keys {
    /// Title/name of the asset.
    pub const TITLE: &str = "title";
    /// Description.
    pub const DESCRIPTION: &str = "description";
    /// Creator/author.
    pub const CREATOR: &str = "creator";
    /// Creation date (ISO 8601).
    pub const CREATION_DATE: &str = "creation_date";
    /// Modification date (ISO 8601).
    pub const MODIFICATION_DATE: &str = "modification_date";
    /// Copyright information.
    pub const COPYRIGHT: &str = "copyright";
    /// Tags/keywords.
    pub const TAGS: &str = "tags";
    /// Project name.
    pub const PROJECT: &str = "project";
    /// Scene number.
    pub const SCENE: &str = "scene";
    /// Take number.
    pub const TAKE: &str = "take";
    /// Camera angle.
    pub const ANGLE: &str = "angle";
    /// Frame rate.
    pub const FRAME_RATE: &str = "frame_rate";
    /// Resolution.
    pub const RESOLUTION: &str = "resolution";
    /// Codec.
    pub const CODEC: &str = "codec";
    /// Notes/comments.
    pub const NOTES: &str = "notes";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_creation() {
        let metadata = Metadata::new();
        assert!(metadata.is_empty());
        assert_eq!(metadata.len(), 0);
    }

    #[test]
    fn test_metadata_set_get_string() {
        let mut metadata = Metadata::new();
        metadata.set_string("title".to_string(), "Test Title".to_string());
        assert_eq!(
            metadata.get_string("title"),
            Some(&"Test Title".to_string())
        );
    }

    #[test]
    fn test_metadata_set_get_int() {
        let mut metadata = Metadata::new();
        metadata.set_int("frame_count".to_string(), 1000);
        assert_eq!(metadata.get_int("frame_count"), Some(1000));
    }

    #[test]
    fn test_metadata_set_get_float() {
        let mut metadata = Metadata::new();
        metadata.set_float("duration".to_string(), 123.45);
        assert_eq!(metadata.get_float("duration"), Some(123.45));
    }

    #[test]
    fn test_metadata_set_get_bool() {
        let mut metadata = Metadata::new();
        metadata.set_bool("is_proxy".to_string(), true);
        assert_eq!(metadata.get_bool("is_proxy"), Some(true));
    }

    #[test]
    fn test_metadata_set_get_string_list() {
        let mut metadata = Metadata::new();
        let tags = vec!["tag1".to_string(), "tag2".to_string()];
        metadata.set_string_list("tags".to_string(), tags.clone());
        assert_eq!(metadata.get_string_list("tags"), Some(&tags));
    }

    #[test]
    fn test_metadata_get_wrong_type() {
        let mut metadata = Metadata::new();
        metadata.set_string("title".to_string(), "Test".to_string());
        assert_eq!(metadata.get_int("title"), None);
        assert_eq!(metadata.get_float("title"), None);
        assert_eq!(metadata.get_bool("title"), None);
    }

    #[test]
    fn test_metadata_remove() {
        let mut metadata = Metadata::new();
        metadata.set_string("title".to_string(), "Test".to_string());
        assert!(metadata.contains_key("title"));
        metadata.remove("title");
        assert!(!metadata.contains_key("title"));
    }

    #[test]
    fn test_metadata_contains_key() {
        let mut metadata = Metadata::new();
        assert!(!metadata.contains_key("title"));
        metadata.set_string("title".to_string(), "Test".to_string());
        assert!(metadata.contains_key("title"));
    }

    #[test]
    fn test_metadata_keys() {
        let mut metadata = Metadata::new();
        metadata.set_string("key1".to_string(), "value1".to_string());
        metadata.set_string("key2".to_string(), "value2".to_string());
        let keys = metadata.keys();
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&&"key1".to_string()));
        assert!(keys.contains(&&"key2".to_string()));
    }

    #[test]
    fn test_metadata_clear() {
        let mut metadata = Metadata::new();
        metadata.set_string("key1".to_string(), "value1".to_string());
        metadata.set_string("key2".to_string(), "value2".to_string());
        assert_eq!(metadata.len(), 2);
        metadata.clear();
        assert!(metadata.is_empty());
    }

    #[test]
    fn test_metadata_merge() {
        let mut metadata1 = Metadata::new();
        metadata1.set_string("key1".to_string(), "value1".to_string());

        let mut metadata2 = Metadata::new();
        metadata2.set_string("key2".to_string(), "value2".to_string());

        metadata1.merge(&metadata2);
        assert_eq!(metadata1.len(), 2);
        assert!(metadata1.contains_key("key1"));
        assert!(metadata1.contains_key("key2"));
    }

    #[test]
    fn test_metadata_merge_overwrite() {
        let mut metadata1 = Metadata::new();
        metadata1.set_string("key1".to_string(), "value1".to_string());

        let mut metadata2 = Metadata::new();
        metadata2.set_string("key1".to_string(), "value2".to_string());

        metadata1.merge(&metadata2);
        assert_eq!(metadata1.get_string("key1"), Some(&"value2".to_string()));
    }

    #[test]
    fn test_standard_keys() {
        let mut metadata = Metadata::new();
        metadata.set_string(keys::TITLE.to_string(), "My Project".to_string());
        metadata.set_string(keys::CREATOR.to_string(), "John Doe".to_string());
        metadata.set_int(keys::FRAME_RATE.to_string(), 24);

        assert_eq!(
            metadata.get_string(keys::TITLE),
            Some(&"My Project".to_string())
        );
        assert_eq!(
            metadata.get_string(keys::CREATOR),
            Some(&"John Doe".to_string())
        );
        assert_eq!(metadata.get_int(keys::FRAME_RATE), Some(24));
    }
}
