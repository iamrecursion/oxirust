//! Highlight bookmarking.

use std::time::Duration;

/// Highlight marker for bookmarking moments.
pub struct HighlightMarker {
    markers: Vec<Marker>,
}

/// Marker.
#[derive(Debug, Clone)]
pub struct Marker {
    /// Marker name
    pub name: String,
    /// Timestamp
    pub timestamp: Duration,
    /// Description
    pub description: String,
}

impl HighlightMarker {
    /// Create a new highlight marker.
    #[must_use]
    pub fn new() -> Self {
        Self {
            markers: Vec::new(),
        }
    }

    /// Add a marker.
    pub fn add_marker(&mut self, marker: Marker) {
        self.markers.push(marker);
    }

    /// Remove a marker.
    pub fn remove_marker(&mut self, name: &str) {
        self.markers.retain(|m| m.name != name);
    }

    /// Get marker count.
    #[must_use]
    pub fn marker_count(&self) -> usize {
        self.markers.len()
    }

    /// Get all markers.
    #[must_use]
    pub fn markers(&self) -> &[Marker] {
        &self.markers
    }
}

impl Default for HighlightMarker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_marker_creation() {
        let marker = HighlightMarker::new();
        assert_eq!(marker.marker_count(), 0);
    }

    #[test]
    fn test_add_marker() {
        let mut marker = HighlightMarker::new();
        marker.add_marker(Marker {
            name: "Epic Moment".to_string(),
            timestamp: Duration::from_mins(2),
            description: "Amazing play!".to_string(),
        });
        assert_eq!(marker.marker_count(), 1);
    }
}
