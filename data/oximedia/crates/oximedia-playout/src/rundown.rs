//! Rundown / script management for playout.
//!
//! A rundown is an ordered list of items (stories, breaks, adverts, fillers)
//! that make up a broadcast programme.  Each item has a planned duration and,
//! once played, an actual duration so over/under-run can be tracked.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Classification of a rundown item.
#[derive(Debug, Clone, PartialEq)]
pub enum ItemType {
    /// A programme story or segment
    Story,
    /// A station break
    Break,
    /// An advertisement
    Advert,
    /// Padding / filler content
    Filler,
    /// A non-playout header row used for organisation
    Header,
    /// A placeholder slot not yet filled with content
    Placeholder,
}

impl ItemType {
    /// Returns true if this item type carries playout content.
    pub fn is_content(&self) -> bool {
        matches!(
            self,
            Self::Story | Self::Break | Self::Advert | Self::Filler
        )
    }

    /// Returns true if this item type has a meaningful planned duration.
    pub fn is_timed(&self) -> bool {
        matches!(self, Self::Story | Self::Advert | Self::Filler)
    }
}

/// A single item within a rundown.
#[derive(Debug, Clone)]
pub struct RundownItem {
    /// Unique identifier
    pub id: u32,
    /// Title or label for this item
    pub title: String,
    /// Classification
    pub item_type: ItemType,
    /// Planned duration in seconds
    pub duration_secs: f32,
    /// Actual duration in seconds once played (None if not yet played)
    pub actual_duration_secs: Option<f32>,
    /// Workflow status string (e.g., "ready", "on-air", "done")
    pub status: String,
}

impl RundownItem {
    /// Create a new rundown item.
    pub fn new(id: u32, title: impl Into<String>, item_type: ItemType, duration_secs: f32) -> Self {
        Self {
            id,
            title: title.into(),
            item_type,
            duration_secs,
            actual_duration_secs: None,
            status: "ready".to_string(),
        }
    }

    /// Returns true if this item ran longer than planned by more than the
    /// given tolerance (in seconds).
    ///
    /// Returns false if no actual duration has been recorded yet.
    pub fn is_over_time(&self, tolerance_secs: f32) -> bool {
        match self.actual_duration_secs {
            Some(actual) => actual > self.duration_secs + tolerance_secs,
            None => false,
        }
    }

    /// Return the signed deviation between actual and planned duration.
    ///
    /// Positive values mean the item ran long; negative values mean it was
    /// short.  Returns 0.0 if no actual duration has been recorded.
    pub fn time_deviation_secs(&self) -> f32 {
        match self.actual_duration_secs {
            Some(actual) => actual - self.duration_secs,
            None => 0.0,
        }
    }
}

/// A complete rundown (programme schedule) containing an ordered list of items.
#[derive(Debug, Clone)]
pub struct Rundown {
    /// Ordered list of items
    pub items: Vec<RundownItem>,
    /// Rundown name
    pub name: String,
    /// Target total duration in seconds
    pub planned_duration_secs: f32,
}

impl Rundown {
    /// Create a new empty rundown.
    pub fn new(name: impl Into<String>, planned_duration_secs: f32) -> Self {
        Self {
            items: Vec::new(),
            name: name.into(),
            planned_duration_secs,
        }
    }

    /// Append an item to the end of the rundown.
    pub fn add_item(&mut self, item: RundownItem) {
        self.items.push(item);
    }

    /// Return the sum of all items' planned durations.
    pub fn total_planned_duration(&self) -> f32 {
        self.items.iter().map(|i| i.duration_secs).sum()
    }

    /// Return the sum of actual durations for items that have been played.
    ///
    /// Items with no actual duration contribute 0.
    pub fn total_actual_duration(&self) -> f32 {
        self.items
            .iter()
            .map(|i| i.actual_duration_secs.unwrap_or(0.0))
            .sum()
    }

    /// Return the number of items in this rundown.
    pub fn item_count(&self) -> usize {
        self.items.len()
    }

    /// Return references to items that are over time by more than `tolerance`
    /// seconds.
    pub fn over_time_items(&self, tolerance: f32) -> Vec<&RundownItem> {
        self.items
            .iter()
            .filter(|i| i.is_over_time(tolerance))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn story(id: u32, planned: f32, actual: Option<f32>) -> RundownItem {
        let mut item = RundownItem::new(id, "Story", ItemType::Story, planned);
        item.actual_duration_secs = actual;
        item
    }

    // --- ItemType tests ---

    #[test]
    fn test_story_is_content() {
        assert!(ItemType::Story.is_content());
    }

    #[test]
    fn test_header_is_not_content() {
        assert!(!ItemType::Header.is_content());
    }

    #[test]
    fn test_placeholder_is_not_content() {
        assert!(!ItemType::Placeholder.is_content());
    }

    #[test]
    fn test_story_is_timed() {
        assert!(ItemType::Story.is_timed());
    }

    #[test]
    fn test_break_is_not_timed() {
        assert!(!ItemType::Break.is_timed());
    }

    #[test]
    fn test_advert_is_timed() {
        assert!(ItemType::Advert.is_timed());
    }

    // --- RundownItem tests ---

    #[test]
    fn test_item_not_over_time_no_actual() {
        let item = story(1, 30.0, None);
        assert!(!item.is_over_time(0.0));
    }

    #[test]
    fn test_item_over_time_true() {
        let item = story(2, 30.0, Some(35.0));
        assert!(item.is_over_time(3.0)); // 35 - 30 = 5 > 3 tolerance
    }

    #[test]
    fn test_item_over_time_within_tolerance() {
        let item = story(3, 30.0, Some(32.0));
        assert!(!item.is_over_time(3.0)); // 32 - 30 = 2 <= 3 tolerance
    }

    #[test]
    fn test_time_deviation_no_actual() {
        let item = story(4, 30.0, None);
        assert!((item.time_deviation_secs() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_time_deviation_positive() {
        let item = story(5, 30.0, Some(35.0));
        assert!((item.time_deviation_secs() - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_time_deviation_negative() {
        let item = story(6, 30.0, Some(25.0));
        assert!((item.time_deviation_secs() - (-5.0)).abs() < f32::EPSILON);
    }

    // --- Rundown tests ---

    #[test]
    fn test_rundown_item_count_empty() {
        let rd = Rundown::new("News", 1800.0);
        assert_eq!(rd.item_count(), 0);
    }

    #[test]
    fn test_rundown_add_item_increases_count() {
        let mut rd = Rundown::new("News", 1800.0);
        rd.add_item(story(1, 60.0, None));
        rd.add_item(story(2, 90.0, None));
        assert_eq!(rd.item_count(), 2);
    }

    #[test]
    fn test_total_planned_duration() {
        let mut rd = Rundown::new("News", 1800.0);
        rd.add_item(story(1, 60.0, None));
        rd.add_item(story(2, 90.0, None));
        assert!((rd.total_planned_duration() - 150.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_total_actual_duration_partial() {
        let mut rd = Rundown::new("News", 1800.0);
        rd.add_item(story(1, 60.0, Some(65.0)));
        rd.add_item(story(2, 90.0, None));
        assert!((rd.total_actual_duration() - 65.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_over_time_items_none() {
        let mut rd = Rundown::new("News", 1800.0);
        rd.add_item(story(1, 60.0, Some(60.0)));
        assert!(rd.over_time_items(0.0).is_empty());
    }

    #[test]
    fn test_over_time_items_some() {
        let mut rd = Rundown::new("News", 1800.0);
        rd.add_item(story(1, 60.0, Some(70.0)));
        rd.add_item(story(2, 30.0, Some(31.0)));
        let over = rd.over_time_items(5.0);
        assert_eq!(over.len(), 1);
        assert_eq!(over[0].id, 1);
    }
}
