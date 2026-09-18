//! Filler content management.

use crate::PlaylistItem;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Type of filler content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FillerType {
    /// Static image or slate.
    Slate,

    /// Video loop.
    Loop,

    /// Promotional content.
    Promo,

    /// Public service announcement.
    Psa,

    /// Color bars and tone.
    ColorBars,

    /// Generic filler.
    Generic,
}

/// Filler content item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FillerContent {
    /// Unique identifier.
    pub id: String,

    /// Display name.
    pub name: String,

    /// Playlist item.
    pub item: PlaylistItem,

    /// Type of filler.
    pub filler_type: FillerType,

    /// Priority (higher values used first).
    pub priority: u32,

    /// Whether this filler is enabled.
    pub enabled: bool,

    /// Minimum duration to fill.
    pub min_duration: Option<Duration>,

    /// Maximum duration to fill.
    pub max_duration: Option<Duration>,
}

impl FillerContent {
    /// Creates a new filler content item.
    #[must_use]
    pub fn new<S: Into<String>>(name: S, item: PlaylistItem, filler_type: FillerType) -> Self {
        Self {
            id: generate_id(),
            name: name.into(),
            item,
            filler_type,
            priority: 0,
            enabled: true,
            min_duration: None,
            max_duration: None,
        }
    }

    /// Sets the priority.
    #[must_use]
    pub const fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Sets the duration constraints.
    #[must_use]
    pub const fn with_duration_constraints(mut self, min: Duration, max: Duration) -> Self {
        self.min_duration = Some(min);
        self.max_duration = Some(max);
        self
    }

    /// Checks if this filler can fill the given duration.
    #[must_use]
    pub fn can_fill(&self, duration: Duration) -> bool {
        if !self.enabled {
            return false;
        }

        if let Some(min) = self.min_duration {
            if duration < min {
                return false;
            }
        }

        if let Some(max) = self.max_duration {
            if duration > max {
                return false;
            }
        }

        true
    }
}

/// Manager for filler content.
#[derive(Debug, Default)]
pub struct FillerManager {
    filler_items: Vec<FillerContent>,
    default_filler: Option<String>,
}

impl FillerManager {
    /// Creates a new filler manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a filler item.
    pub fn add_filler(&mut self, filler: FillerContent) {
        self.filler_items.push(filler);
        self.sort_by_priority();
    }

    /// Removes a filler item by ID.
    pub fn remove_filler(&mut self, filler_id: &str) {
        self.filler_items.retain(|f| f.id != filler_id);
    }

    /// Sets the default filler.
    pub fn set_default_filler(&mut self, filler_id: String) {
        self.default_filler = Some(filler_id);
    }

    /// Gets the best filler for a given duration.
    #[must_use]
    pub fn get_filler_for_duration(&self, duration: Duration) -> Option<&FillerContent> {
        // First, try to find a filler that can fill this duration
        for filler in &self.filler_items {
            if filler.can_fill(duration) {
                return Some(filler);
            }
        }

        // If no specific filler found, try the default
        if let Some(default_id) = &self.default_filler {
            return self.filler_items.iter().find(|f| &f.id == default_id);
        }

        // Last resort: any enabled filler
        self.filler_items.iter().find(|f| f.enabled)
    }

    /// Gets all fillers of a specific type.
    #[must_use]
    pub fn get_fillers_by_type(&self, filler_type: FillerType) -> Vec<&FillerContent> {
        self.filler_items
            .iter()
            .filter(|f| f.enabled && f.filler_type == filler_type)
            .collect()
    }

    /// Sorts filler items by priority (highest first).
    fn sort_by_priority(&mut self) {
        self.filler_items
            .sort_by(|a, b| b.priority.cmp(&a.priority));
    }

    /// Returns the number of filler items.
    #[must_use]
    pub fn len(&self) -> usize {
        self.filler_items.len()
    }

    /// Returns true if there are no filler items.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.filler_items.is_empty()
    }
}

fn generate_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("filler_{timestamp}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filler_content() {
        let item = PlaylistItem::new("slate.png");
        let filler = FillerContent::new("Station Slate", item, FillerType::Slate).with_priority(10);

        assert_eq!(filler.priority, 10);
        assert!(filler.can_fill(Duration::from_secs(60)));
    }

    #[test]
    fn test_filler_duration_constraints() {
        let item = PlaylistItem::new("promo.mxf");
        let filler = FillerContent::new("Promo", item, FillerType::Promo)
            .with_duration_constraints(Duration::from_secs(30), Duration::from_secs(120));

        assert!(!filler.can_fill(Duration::from_secs(10)));
        assert!(filler.can_fill(Duration::from_secs(60)));
        assert!(!filler.can_fill(Duration::from_secs(200)));
    }

    #[test]
    fn test_filler_manager() {
        let mut manager = FillerManager::new();

        let item1 = PlaylistItem::new("filler1.mxf");
        let filler1 = FillerContent::new("Filler 1", item1, FillerType::Generic).with_priority(5);

        let item2 = PlaylistItem::new("filler2.mxf");
        let filler2 = FillerContent::new("Filler 2", item2, FillerType::Generic).with_priority(10);

        manager.add_filler(filler1);
        manager.add_filler(filler2);

        let best = manager.get_filler_for_duration(Duration::from_secs(60));
        assert!(best.is_some());
        assert_eq!(best.expect("should succeed in test").priority, 10);
    }
}
