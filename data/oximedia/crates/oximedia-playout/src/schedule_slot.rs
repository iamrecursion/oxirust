//! Schedule slot management for the playout server.
//!
//! Provides `SlotStatus`, `ScheduleSlot`, and `ScheduleGrid` for building
//! and querying a time-based broadcast schedule.

#![allow(dead_code)]

// ── SlotStatus ────────────────────────────────────────────────────────────────

/// Availability state of a schedule slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlotStatus {
    /// Slot is open and can accept a new item.
    Available,
    /// Slot has been booked with content.
    Booked { item_id: String },
    /// Slot is reserved but not yet confirmed.
    Reserved,
    /// Slot is blocked (e.g., maintenance window).
    Blocked,
    /// Slot is past and has already aired.
    Aired,
}

impl SlotStatus {
    /// Returns `true` when the slot can still accept an item.
    pub fn is_available(&self) -> bool {
        matches!(self, Self::Available)
    }

    /// Returns `true` when the slot has been committed (Booked or Aired).
    pub fn is_committed(&self) -> bool {
        matches!(self, Self::Booked { .. } | Self::Aired)
    }

    /// Returns `true` when the slot is provisionally held (Reserved).
    pub fn is_reserved(&self) -> bool {
        matches!(self, Self::Reserved)
    }

    /// Extract the booked item ID, if any.
    pub fn item_id(&self) -> Option<&str> {
        if let Self::Booked { item_id } = self {
            Some(item_id.as_str())
        } else {
            None
        }
    }
}

// ── ScheduleSlot ─────────────────────────────────────────────────────────────

/// A single time slot in the broadcast schedule.
///
/// Times are represented as Unix timestamps in milliseconds.
#[derive(Debug, Clone)]
pub struct ScheduleSlot {
    /// Unique slot identifier.
    pub id: String,
    /// Start of the slot (Unix ms).
    pub start_ms: u64,
    /// End of the slot (Unix ms).
    pub end_ms: u64,
    /// Current status.
    pub status: SlotStatus,
    /// Optional label (e.g., show title).
    pub label: Option<String>,
}

impl ScheduleSlot {
    /// Create a new available slot.
    ///
    /// Panics (debug only) when `start_ms >= end_ms`.
    pub fn new(id: impl Into<String>, start_ms: u64, end_ms: u64) -> Self {
        debug_assert!(start_ms < end_ms, "start must be before end");
        Self {
            id: id.into(),
            start_ms,
            end_ms,
            status: SlotStatus::Available,
            label: None,
        }
    }

    /// Attach a label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Duration of the slot in milliseconds.
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }

    /// Returns `true` when this slot overlaps with `[other_start, other_end)`.
    pub fn overlaps(&self, other_start: u64, other_end: u64) -> bool {
        self.start_ms < other_end && other_start < self.end_ms
    }

    /// Returns `true` when the given timestamp falls inside the slot.
    pub fn contains_ms(&self, ts_ms: u64) -> bool {
        ts_ms >= self.start_ms && ts_ms < self.end_ms
    }

    /// Book this slot with an item.  Returns `false` when not available.
    pub fn book(&mut self, item_id: impl Into<String>) -> bool {
        if self.status.is_available() {
            self.status = SlotStatus::Booked {
                item_id: item_id.into(),
            };
            true
        } else {
            false
        }
    }

    /// Release the booking, restoring the slot to `Available`.
    pub fn release(&mut self) {
        if matches!(
            self.status,
            SlotStatus::Booked { .. } | SlotStatus::Reserved
        ) {
            self.status = SlotStatus::Available;
        }
    }

    /// Mark the slot as aired.
    pub fn mark_aired(&mut self) {
        self.status = SlotStatus::Aired;
    }
}

// ── ScheduleGrid ─────────────────────────────────────────────────────────────

/// A collection of schedule slots forming a broadcast grid.
pub struct ScheduleGrid {
    slots: Vec<ScheduleSlot>,
}

impl ScheduleGrid {
    /// Create an empty grid.
    pub fn new() -> Self {
        Self { slots: Vec::new() }
    }

    /// Add a slot to the grid.  The slot is inserted in start-time order.
    pub fn add_slot(&mut self, slot: ScheduleSlot) {
        let pos = self.slots.partition_point(|s| s.start_ms <= slot.start_ms);
        self.slots.insert(pos, slot);
    }

    /// Return all slots whose time range overlaps `[start_ms, end_ms)`.
    pub fn slots_in_range(&self, start_ms: u64, end_ms: u64) -> Vec<&ScheduleSlot> {
        self.slots
            .iter()
            .filter(|s| s.overlaps(start_ms, end_ms))
            .collect()
    }

    /// Find the first available slot that contains or starts at or after `from_ms`.
    pub fn available_at(&self, from_ms: u64) -> Option<&ScheduleSlot> {
        self.slots
            .iter()
            .find(|s| s.contains_ms(from_ms) && s.status.is_available())
            .or_else(|| {
                self.slots
                    .iter()
                    .find(|s| s.start_ms >= from_ms && s.status.is_available())
            })
    }

    /// All slots in the grid, in start-time order.
    pub fn all_slots(&self) -> &[ScheduleSlot] {
        &self.slots
    }

    /// Total number of slots.
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Returns `true` when the grid has no slots.
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Number of available slots.
    pub fn available_count(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| s.status.is_available())
            .count()
    }

    /// Number of booked slots.
    pub fn booked_count(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| matches!(s.status, SlotStatus::Booked { .. }))
            .count()
    }

    /// Find a slot by its ID.
    pub fn find_by_id(&self, id: &str) -> Option<&ScheduleSlot> {
        self.slots.iter().find(|s| s.id == id)
    }

    /// Find a mutable slot by its ID.
    pub fn find_by_id_mut(&mut self, id: &str) -> Option<&mut ScheduleSlot> {
        self.slots.iter_mut().find(|s| s.id == id)
    }
}

impl Default for ScheduleGrid {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn slot(id: &str, start: u64, end: u64) -> ScheduleSlot {
        ScheduleSlot::new(id, start, end)
    }

    // SlotStatus

    #[test]
    fn status_available_is_available() {
        assert!(SlotStatus::Available.is_available());
        assert!(!SlotStatus::Reserved.is_available());
        assert!(!SlotStatus::Booked {
            item_id: "x".into()
        }
        .is_available());
    }

    #[test]
    fn status_booked_is_committed() {
        assert!(SlotStatus::Booked {
            item_id: "a".into()
        }
        .is_committed());
        assert!(SlotStatus::Aired.is_committed());
        assert!(!SlotStatus::Available.is_committed());
    }

    #[test]
    fn status_reserved_is_reserved() {
        assert!(SlotStatus::Reserved.is_reserved());
        assert!(!SlotStatus::Available.is_reserved());
    }

    #[test]
    fn status_item_id() {
        let s = SlotStatus::Booked {
            item_id: "item42".into(),
        };
        assert_eq!(s.item_id(), Some("item42"));
        assert_eq!(SlotStatus::Available.item_id(), None);
    }

    // ScheduleSlot

    #[test]
    fn slot_duration_ms() {
        let s = slot("s1", 1000, 4000);
        assert_eq!(s.duration_ms(), 3000);
    }

    #[test]
    fn slot_overlaps_true() {
        let s = slot("s1", 1000, 3000);
        assert!(s.overlaps(2000, 4000)); // partial overlap
        assert!(s.overlaps(500, 2000)); // overlaps start
        assert!(s.overlaps(1000, 3000)); // exact match
    }

    #[test]
    fn slot_overlaps_false() {
        let s = slot("s1", 1000, 3000);
        assert!(!s.overlaps(3000, 5000)); // adjacent, no overlap
        assert!(!s.overlaps(0, 1000)); // adjacent before
        assert!(!s.overlaps(5000, 6000)); // completely after
    }

    #[test]
    fn slot_contains_ms() {
        let s = slot("s1", 1000, 3000);
        assert!(s.contains_ms(1000));
        assert!(s.contains_ms(2000));
        assert!(!s.contains_ms(3000)); // exclusive end
        assert!(!s.contains_ms(999));
    }

    #[test]
    fn slot_book_success() {
        let mut s = slot("s1", 1000, 2000);
        assert!(s.book("item1"));
        assert_eq!(s.status.item_id(), Some("item1"));
    }

    #[test]
    fn slot_book_fails_when_already_booked() {
        let mut s = slot("s1", 1000, 2000);
        s.book("item1");
        assert!(!s.book("item2")); // already booked
    }

    #[test]
    fn slot_release_restores_available() {
        let mut s = slot("s1", 1000, 2000);
        s.book("item1");
        s.release();
        assert!(s.status.is_available());
    }

    #[test]
    fn slot_mark_aired() {
        let mut s = slot("s1", 1000, 2000);
        s.mark_aired();
        assert_eq!(s.status, SlotStatus::Aired);
    }

    // ScheduleGrid

    #[test]
    fn grid_add_and_len() {
        let mut grid = ScheduleGrid::new();
        assert!(grid.is_empty());
        grid.add_slot(slot("s1", 0, 1000));
        grid.add_slot(slot("s2", 1000, 2000));
        assert_eq!(grid.len(), 2);
    }

    #[test]
    fn grid_ordered_by_start() {
        let mut grid = ScheduleGrid::new();
        grid.add_slot(slot("s2", 2000, 3000));
        grid.add_slot(slot("s1", 0, 1000));
        let all = grid.all_slots();
        assert_eq!(all[0].id, "s1");
        assert_eq!(all[1].id, "s2");
    }

    #[test]
    fn grid_available_at() {
        let mut grid = ScheduleGrid::new();
        grid.add_slot(slot("s1", 0, 1000));
        grid.add_slot(slot("s2", 1000, 2000));
        let found = grid.available_at(900);
        assert!(found.is_some());
        assert_eq!(found.expect("should succeed in test").id, "s1");
    }

    #[test]
    fn grid_available_count() {
        let mut grid = ScheduleGrid::new();
        grid.add_slot(slot("s1", 0, 1000));
        grid.add_slot(slot("s2", 1000, 2000));
        grid.find_by_id_mut("s1")
            .expect("should succeed in test")
            .book("item1");
        assert_eq!(grid.available_count(), 1);
        assert_eq!(grid.booked_count(), 1);
    }

    #[test]
    fn grid_slots_in_range() {
        let mut grid = ScheduleGrid::new();
        grid.add_slot(slot("s1", 0, 1000));
        grid.add_slot(slot("s2", 500, 1500));
        grid.add_slot(slot("s3", 2000, 3000));
        let results = grid.slots_in_range(400, 1200);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn grid_find_by_id() {
        let mut grid = ScheduleGrid::new();
        grid.add_slot(slot("s1", 0, 1000));
        assert!(grid.find_by_id("s1").is_some());
        assert!(grid.find_by_id("s99").is_none());
    }
}
