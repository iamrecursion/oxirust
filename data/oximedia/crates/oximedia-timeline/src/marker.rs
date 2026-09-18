//! Marker system for timeline annotations.

use serde::{Deserialize, Serialize};

use crate::types::Position;

/// Type of marker.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MarkerType {
    /// Standard marker.
    Standard,
    /// Chapter marker.
    Chapter,
    /// Comment marker.
    Comment,
    /// To-do marker.
    Todo,
    /// Web link marker.
    WebLink,
}

/// Color for marker visualization.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MarkerColor {
    /// Red marker.
    Red,
    /// Orange marker.
    Orange,
    /// Yellow marker.
    Yellow,
    /// Green marker.
    Green,
    /// Blue marker.
    Blue,
    /// Purple marker.
    Purple,
    /// Custom RGB color (0-255).
    Custom(u8, u8, u8),
}

impl MarkerColor {
    /// Creates a custom RGB marker color.
    #[must_use]
    pub const fn custom(r: u8, g: u8, b: u8) -> Self {
        Self::Custom(r, g, b)
    }

    /// Returns RGB values (0-255).
    #[must_use]
    pub const fn to_rgb(self) -> (u8, u8, u8) {
        match self {
            Self::Red => (255, 0, 0),
            Self::Orange => (255, 165, 0),
            Self::Yellow => (255, 255, 0),
            Self::Green => (0, 255, 0),
            Self::Blue => (0, 0, 255),
            Self::Purple => (128, 0, 128),
            Self::Custom(r, g, b) => (r, g, b),
        }
    }
}

/// A marker in the timeline.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Marker {
    /// Position in timeline.
    pub position: Position,
    /// Type of marker.
    pub marker_type: MarkerType,
    /// Marker color.
    pub color: MarkerColor,
    /// Marker name/title.
    pub name: String,
    /// Comment/description.
    pub comment: String,
    /// Duration for chapter markers (optional).
    pub duration: Option<crate::types::Duration>,
    /// Additional metadata.
    pub metadata: std::collections::HashMap<String, String>,
}

impl Marker {
    /// Creates a new marker.
    #[must_use]
    pub fn new(position: Position, name: String) -> Self {
        Self {
            position,
            marker_type: MarkerType::Standard,
            color: MarkerColor::Blue,
            name,
            comment: String::new(),
            duration: None,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Creates a chapter marker.
    #[must_use]
    pub fn chapter(position: Position, name: String) -> Self {
        Self {
            position,
            marker_type: MarkerType::Chapter,
            color: MarkerColor::Green,
            name,
            comment: String::new(),
            duration: None,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Creates a comment marker.
    #[must_use]
    pub fn comment(position: Position, comment: String) -> Self {
        Self {
            position,
            marker_type: MarkerType::Comment,
            color: MarkerColor::Yellow,
            name: String::new(),
            comment,
            duration: None,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Creates a to-do marker.
    #[must_use]
    pub fn todo(position: Position, task: String) -> Self {
        Self {
            position,
            marker_type: MarkerType::Todo,
            color: MarkerColor::Red,
            name: task.clone(),
            comment: task,
            duration: None,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Sets the marker color.
    #[must_use]
    pub fn with_color(mut self, color: MarkerColor) -> Self {
        self.color = color;
        self
    }

    /// Sets the marker comment.
    #[must_use]
    pub fn with_comment(mut self, comment: String) -> Self {
        self.comment = comment;
        self
    }

    /// Sets the marker duration.
    #[must_use]
    pub fn with_duration(mut self, duration: crate::types::Duration) -> Self {
        self.duration = Some(duration);
        self
    }

    /// Adds metadata.
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Gets metadata value.
    #[must_use]
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }
}

/// Collection of markers.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MarkerCollection {
    markers: Vec<Marker>,
}

impl MarkerCollection {
    /// Creates a new empty marker collection.
    #[must_use]
    pub fn new() -> Self {
        Self {
            markers: Vec::new(),
        }
    }

    /// Adds a marker.
    pub fn add_marker(&mut self, marker: Marker) {
        self.markers.push(marker);
        self.sort_markers();
    }

    /// Removes a marker at position.
    pub fn remove_marker_at(&mut self, position: Position) -> Option<Marker> {
        if let Some(index) = self.markers.iter().position(|m| m.position == position) {
            Some(self.markers.remove(index))
        } else {
            None
        }
    }

    /// Gets all markers.
    #[must_use]
    pub fn markers(&self) -> &[Marker] {
        &self.markers
    }

    /// Gets markers in a time range.
    #[must_use]
    pub fn markers_in_range(&self, start: Position, end: Position) -> Vec<&Marker> {
        self.markers
            .iter()
            .filter(|m| m.position >= start && m.position < end)
            .collect()
    }

    /// Gets marker at exact position.
    #[must_use]
    pub fn marker_at(&self, position: Position) -> Option<&Marker> {
        self.markers.iter().find(|m| m.position == position)
    }

    /// Gets markers of a specific type.
    #[must_use]
    pub fn markers_of_type(&self, marker_type: MarkerType) -> Vec<&Marker> {
        self.markers
            .iter()
            .filter(|m| m.marker_type == marker_type)
            .collect()
    }

    /// Sorts markers by position.
    fn sort_markers(&mut self) {
        self.markers.sort_by_key(|m| m.position.value());
    }

    /// Returns the number of markers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.markers.len()
    }

    /// Checks if collection is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.markers.is_empty()
    }

    /// Clears all markers.
    pub fn clear(&mut self) {
        self.markers.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_marker_color_to_rgb() {
        assert_eq!(MarkerColor::Red.to_rgb(), (255, 0, 0));
        assert_eq!(MarkerColor::Green.to_rgb(), (0, 255, 0));
        assert_eq!(MarkerColor::Blue.to_rgb(), (0, 0, 255));
        assert_eq!(MarkerColor::custom(100, 150, 200).to_rgb(), (100, 150, 200));
    }

    #[test]
    fn test_marker_creation() {
        let marker = Marker::new(Position::new(100), "Test Marker".to_string());
        assert_eq!(marker.position.value(), 100);
        assert_eq!(marker.name, "Test Marker");
        assert_eq!(marker.marker_type, MarkerType::Standard);
    }

    #[test]
    fn test_marker_chapter() {
        let marker = Marker::chapter(Position::new(100), "Chapter 1".to_string());
        assert_eq!(marker.marker_type, MarkerType::Chapter);
        assert_eq!(marker.color, MarkerColor::Green);
    }

    #[test]
    fn test_marker_comment() {
        let marker = Marker::comment(Position::new(100), "This needs work".to_string());
        assert_eq!(marker.marker_type, MarkerType::Comment);
        assert_eq!(marker.comment, "This needs work");
    }

    #[test]
    fn test_marker_todo() {
        let marker = Marker::todo(Position::new(100), "Fix audio".to_string());
        assert_eq!(marker.marker_type, MarkerType::Todo);
        assert_eq!(marker.color, MarkerColor::Red);
    }

    #[test]
    fn test_marker_with_color() {
        let marker =
            Marker::new(Position::new(100), "Test".to_string()).with_color(MarkerColor::Purple);
        assert_eq!(marker.color, MarkerColor::Purple);
    }

    #[test]
    fn test_marker_with_comment() {
        let marker =
            Marker::new(Position::new(100), "Test".to_string()).with_comment("Comment".to_string());
        assert_eq!(marker.comment, "Comment");
    }

    #[test]
    fn test_marker_with_duration() {
        let marker = Marker::new(Position::new(100), "Test".to_string())
            .with_duration(crate::types::Duration::new(50));
        assert_eq!(marker.duration, Some(crate::types::Duration::new(50)));
    }

    #[test]
    fn test_marker_metadata() {
        let mut marker = Marker::new(Position::new(100), "Test".to_string());
        marker.add_metadata("key1".to_string(), "value1".to_string());
        assert_eq!(marker.get_metadata("key1"), Some(&"value1".to_string()));
    }

    #[test]
    fn test_marker_collection_add() {
        let mut collection = MarkerCollection::new();
        collection.add_marker(Marker::new(Position::new(100), "Test".to_string()));
        assert_eq!(collection.len(), 1);
    }

    #[test]
    fn test_marker_collection_remove() {
        let mut collection = MarkerCollection::new();
        collection.add_marker(Marker::new(Position::new(100), "Test".to_string()));
        let removed = collection.remove_marker_at(Position::new(100));
        assert!(removed.is_some());
        assert!(collection.is_empty());
    }

    #[test]
    fn test_marker_collection_marker_at() {
        let mut collection = MarkerCollection::new();
        collection.add_marker(Marker::new(Position::new(100), "Test".to_string()));
        assert!(collection.marker_at(Position::new(100)).is_some());
        assert!(collection.marker_at(Position::new(200)).is_none());
    }

    #[test]
    fn test_marker_collection_markers_in_range() {
        let mut collection = MarkerCollection::new();
        collection.add_marker(Marker::new(Position::new(50), "M1".to_string()));
        collection.add_marker(Marker::new(Position::new(100), "M2".to_string()));
        collection.add_marker(Marker::new(Position::new(150), "M3".to_string()));

        let markers = collection.markers_in_range(Position::new(75), Position::new(125));
        assert_eq!(markers.len(), 1);
        assert_eq!(markers[0].name, "M2");
    }

    #[test]
    fn test_marker_collection_markers_of_type() {
        let mut collection = MarkerCollection::new();
        collection.add_marker(Marker::new(Position::new(100), "Standard".to_string()));
        collection.add_marker(Marker::chapter(Position::new(200), "Chapter".to_string()));
        collection.add_marker(Marker::comment(Position::new(300), "Comment".to_string()));

        let chapters = collection.markers_of_type(MarkerType::Chapter);
        assert_eq!(chapters.len(), 1);
        assert_eq!(chapters[0].name, "Chapter");
    }

    #[test]
    fn test_marker_collection_sorting() {
        let mut collection = MarkerCollection::new();
        collection.add_marker(Marker::new(Position::new(200), "M2".to_string()));
        collection.add_marker(Marker::new(Position::new(100), "M1".to_string()));
        collection.add_marker(Marker::new(Position::new(300), "M3".to_string()));

        let markers = collection.markers();
        assert_eq!(markers[0].position.value(), 100);
        assert_eq!(markers[1].position.value(), 200);
        assert_eq!(markers[2].position.value(), 300);
    }

    #[test]
    fn test_marker_collection_clear() {
        let mut collection = MarkerCollection::new();
        collection.add_marker(Marker::new(Position::new(100), "Test".to_string()));
        collection.clear();
        assert!(collection.is_empty());
    }
}
