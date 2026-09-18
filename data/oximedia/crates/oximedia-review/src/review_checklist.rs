//! Review checklist: per-item completion tracking with blocking-item awareness.

#![allow(dead_code)]

/// A single item on a review checklist.
#[derive(Debug, Clone)]
pub struct ChecklistItem {
    /// Unique identifier for the item.
    pub id: u64,
    /// Human-readable description of what must be verified.
    pub description: String,
    /// Whether failure to complete this item blocks overall approval.
    pub blocking: bool,
    /// Whether this item has been completed.
    pub completed: bool,
    /// Optional notes added when completing or skipping the item.
    pub notes: Option<String>,
}

impl ChecklistItem {
    /// Create a new checklist item.
    ///
    /// Items are created in an incomplete state.
    #[must_use]
    pub fn new(id: u64, description: impl Into<String>, blocking: bool) -> Self {
        Self {
            id,
            description: description.into(),
            blocking,
            completed: false,
            notes: None,
        }
    }

    /// Returns `true` if this item must be completed before the review can be approved.
    #[must_use]
    pub fn is_blocking(&self) -> bool {
        self.blocking
    }

    /// Mark the item as complete, optionally with notes.
    pub fn complete(&mut self, notes: Option<impl Into<String>>) {
        self.completed = true;
        self.notes = notes.map(|n| n.into());
    }

    /// Returns `true` if this item is both blocking and not yet completed.
    #[must_use]
    pub fn is_blocking_incomplete(&self) -> bool {
        self.blocking && !self.completed
    }
}

/// An ordered checklist associated with a review.
#[derive(Debug, Clone)]
pub struct ReviewChecklist {
    /// Checklist identifier.
    pub id: u64,
    /// Human-readable name for this checklist.
    pub name: String,
    /// Ordered list of checklist items.
    items: Vec<ChecklistItem>,
}

impl ReviewChecklist {
    /// Create a new empty checklist.
    #[must_use]
    pub fn new(id: u64, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            items: Vec::new(),
        }
    }

    /// Append an item to the checklist.
    pub fn add_item(&mut self, item: ChecklistItem) {
        self.items.push(item);
    }

    /// Mark a specific item as complete by its ID.
    ///
    /// Returns `true` if the item was found and updated.
    pub fn complete_item(&mut self, id: u64, notes: Option<impl Into<String>>) -> bool {
        if let Some(item) = self.items.iter_mut().find(|i| i.id == id) {
            item.complete(notes);
            true
        } else {
            false
        }
    }

    /// Return a slice of all items.
    #[must_use]
    pub fn items(&self) -> &[ChecklistItem] {
        &self.items
    }

    /// Number of items that are blocking and not yet completed.
    ///
    /// When this is zero the review may proceed to approval.
    #[must_use]
    pub fn blocking_incomplete(&self) -> usize {
        self.items
            .iter()
            .filter(|i| i.is_blocking_incomplete())
            .count()
    }

    /// Percentage of items (blocking + non-blocking) that are complete.
    ///
    /// Returns `0.0` when the checklist is empty.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn completion_pct(&self) -> f64 {
        if self.items.is_empty() {
            return 0.0;
        }
        let done = self.items.iter().filter(|i| i.completed).count();
        (done as f64 / self.items.len() as f64) * 100.0
    }

    /// Number of completed items.
    #[must_use]
    pub fn completed_count(&self) -> usize {
        self.items.iter().filter(|i| i.completed).count()
    }

    /// Total number of items.
    #[must_use]
    pub fn total_count(&self) -> usize {
        self.items.len()
    }

    /// Returns `true` when all blocking items are complete.
    #[must_use]
    pub fn can_approve(&self) -> bool {
        self.blocking_incomplete() == 0
    }

    /// Returns `true` when every item (blocking and non-blocking) is complete.
    #[must_use]
    pub fn is_fully_complete(&self) -> bool {
        !self.items.is_empty() && self.items.iter().all(|i| i.completed)
    }
}

// ─── unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn blocking(id: u64, desc: &str) -> ChecklistItem {
        ChecklistItem::new(id, desc, true)
    }

    fn non_blocking(id: u64, desc: &str) -> ChecklistItem {
        ChecklistItem::new(id, desc, false)
    }

    fn checklist() -> ReviewChecklist {
        ReviewChecklist::new(1, "Final Delivery Checklist")
    }

    // 1 — ChecklistItem::is_blocking
    #[test]
    fn test_item_is_blocking() {
        let b = blocking(1, "Verify audio sync");
        let nb = non_blocking(2, "Check metadata");
        assert!(b.is_blocking());
        assert!(!nb.is_blocking());
    }

    // 2 — ChecklistItem::is_blocking_incomplete
    #[test]
    fn test_item_blocking_incomplete_flag() {
        let item = blocking(1, "Audio sync");
        assert!(item.is_blocking_incomplete());
    }

    // 3 — complete clears blocking_incomplete
    #[test]
    fn test_complete_clears_blocking() {
        let mut item = blocking(1, "Audio sync");
        item.complete(None::<String>);
        assert!(!item.is_blocking_incomplete());
        assert!(item.completed);
    }

    // 4 — complete stores notes
    #[test]
    fn test_complete_with_notes() {
        let mut item = non_blocking(1, "Metadata");
        item.complete(Some("All fields verified"));
        assert_eq!(item.notes.as_deref(), Some("All fields verified"));
    }

    // 5 — empty checklist: completion_pct is 0.0
    #[test]
    fn test_empty_checklist_completion_pct() {
        let c = checklist();
        assert!((c.completion_pct() - 0.0).abs() < f64::EPSILON);
    }

    // 6 — add_item increments total_count
    #[test]
    fn test_add_item_total_count() {
        let mut c = checklist();
        c.add_item(blocking(1, "A"));
        c.add_item(non_blocking(2, "B"));
        assert_eq!(c.total_count(), 2);
    }

    // 7 — blocking_incomplete counts only blocking open items
    #[test]
    fn test_blocking_incomplete_count() {
        let mut c = checklist();
        c.add_item(blocking(1, "A"));
        c.add_item(blocking(2, "B"));
        c.add_item(non_blocking(3, "C"));
        assert_eq!(c.blocking_incomplete(), 2);
    }

    // 8 — complete_item marks correct item done
    #[test]
    fn test_complete_item_by_id() {
        let mut c = checklist();
        c.add_item(blocking(1, "A"));
        c.add_item(blocking(2, "B"));
        assert!(c.complete_item(1_u64, None::<String>));
        assert_eq!(c.blocking_incomplete(), 1);
    }

    // 9 — complete_item returns false for unknown id
    #[test]
    fn test_complete_item_unknown_id() {
        let mut c = checklist();
        c.add_item(blocking(1, "A"));
        assert!(!c.complete_item(99_u64, None::<String>));
    }

    // 10 — completion_pct at 50 %
    #[test]
    fn test_completion_pct_partial() {
        let mut c = checklist();
        c.add_item(blocking(1, "A"));
        c.add_item(non_blocking(2, "B"));
        c.complete_item(1_u64, None::<String>);
        let pct = c.completion_pct();
        assert!((pct - 50.0).abs() < 1e-6);
    }

    // 11 — completion_pct at 100 %
    #[test]
    fn test_completion_pct_full() {
        let mut c = checklist();
        c.add_item(blocking(1, "A"));
        c.add_item(non_blocking(2, "B"));
        c.complete_item(1_u64, None::<String>);
        c.complete_item(2_u64, None::<String>);
        let pct = c.completion_pct();
        assert!((pct - 100.0).abs() < 1e-6);
    }

    // 12 — can_approve when no blocking items remain
    #[test]
    fn test_can_approve() {
        let mut c = checklist();
        c.add_item(blocking(1, "A"));
        c.add_item(non_blocking(2, "B"));
        assert!(!c.can_approve());
        c.complete_item(1_u64, None::<String>);
        // non-blocking still open — can_approve should be true
        assert!(c.can_approve());
    }

    // 13 — is_fully_complete requires all items done
    #[test]
    fn test_is_fully_complete() {
        let mut c = checklist();
        c.add_item(blocking(1, "A"));
        c.add_item(non_blocking(2, "B"));
        c.complete_item(1_u64, None::<String>);
        assert!(!c.is_fully_complete());
        c.complete_item(2_u64, None::<String>);
        assert!(c.is_fully_complete());
    }

    // 14 — empty checklist cannot be fully complete
    #[test]
    fn test_empty_not_fully_complete() {
        let c = checklist();
        assert!(!c.is_fully_complete());
    }

    // 15 — completed_count tracks correctly
    #[test]
    fn test_completed_count() {
        let mut c = checklist();
        c.add_item(blocking(1, "A"));
        c.add_item(blocking(2, "B"));
        c.add_item(non_blocking(3, "C"));
        c.complete_item(1_u64, None::<String>);
        c.complete_item(3_u64, None::<String>);
        assert_eq!(c.completed_count(), 2);
    }
}
