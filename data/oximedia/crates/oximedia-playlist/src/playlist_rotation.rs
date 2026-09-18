#![allow(dead_code)]
//! Rotation scheduling for recurring content blocks in broadcast playlists.
//!
//! This module implements rotation logic that cycles through a pool of content
//! items according to configurable rules such as round-robin, weighted random,
//! and daypart-based selection. It is commonly used for music video channels,
//! promotional interstitials, and filler content that must be distributed
//! evenly across a broadcast day.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Strategy used to pick the next item from a rotation pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotationStrategy {
    /// Strict round-robin: items cycle in insertion order.
    RoundRobin,
    /// Weighted: items are chosen based on their relative weight.
    Weighted,
    /// Daypart-aware: items are grouped by daypart and rotated within each.
    Daypart,
}

/// A named time-of-day interval used for daypart scheduling.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DaypartSlot {
    /// Human-readable label such as "morning" or "prime_time".
    pub label: String,
    /// Start hour (0-23) inclusive.
    pub start_hour: u8,
    /// End hour (0-23) inclusive.
    pub end_hour: u8,
}

impl DaypartSlot {
    /// Create a new daypart slot.
    pub fn new(label: impl Into<String>, start_hour: u8, end_hour: u8) -> Self {
        Self {
            label: label.into(),
            start_hour: start_hour.min(23),
            end_hour: end_hour.min(23),
        }
    }

    /// Returns `true` when the given hour falls within this slot.
    pub fn contains_hour(&self, hour: u8) -> bool {
        if self.start_hour <= self.end_hour {
            hour >= self.start_hour && hour <= self.end_hour
        } else {
            // Wraps midnight, e.g. 22..06
            hour >= self.start_hour || hour <= self.end_hour
        }
    }
}

/// A single item within a rotation pool.
#[derive(Debug, Clone, PartialEq)]
pub struct RotationItem {
    /// Unique identifier for the media asset.
    pub asset_id: String,
    /// Relative weight (used with [`RotationStrategy::Weighted`]).
    pub weight: u32,
    /// Optional daypart label this item is restricted to.
    pub daypart_label: Option<String>,
    /// Number of times this item has been played.
    pub play_count: u64,
}

impl RotationItem {
    /// Create a new rotation item with default weight 1.
    pub fn new(asset_id: impl Into<String>) -> Self {
        Self {
            asset_id: asset_id.into(),
            weight: 1,
            daypart_label: None,
            play_count: 0,
        }
    }

    /// Set the weight for weighted rotation.
    pub fn with_weight(mut self, w: u32) -> Self {
        self.weight = w;
        self
    }

    /// Restrict this item to a specific daypart label.
    pub fn with_daypart(mut self, label: impl Into<String>) -> Self {
        self.daypart_label = Some(label.into());
        self
    }
}

// ---------------------------------------------------------------------------
// Rotation pool
// ---------------------------------------------------------------------------

/// A pool of items that are rotated according to a chosen strategy.
#[derive(Debug, Clone)]
pub struct RotationPool {
    /// Pool name.
    pub name: String,
    /// Items in the pool.
    items: Vec<RotationItem>,
    /// Current cursor for round-robin.
    cursor: usize,
    /// Active rotation strategy.
    strategy: RotationStrategy,
    /// Daypart definitions (used only with [`RotationStrategy::Daypart`]).
    dayparts: Vec<DaypartSlot>,
}

impl RotationPool {
    /// Create a new empty rotation pool with the given strategy.
    pub fn new(name: impl Into<String>, strategy: RotationStrategy) -> Self {
        Self {
            name: name.into(),
            items: Vec::new(),
            cursor: 0,
            strategy,
            dayparts: Vec::new(),
        }
    }

    /// Add an item to the pool.
    pub fn add_item(&mut self, item: RotationItem) {
        self.items.push(item);
    }

    /// Register a daypart slot.
    pub fn add_daypart(&mut self, slot: DaypartSlot) {
        self.dayparts.push(slot);
    }

    /// Return the number of items in the pool.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Return `true` when the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Return the current rotation strategy.
    pub fn strategy(&self) -> RotationStrategy {
        self.strategy
    }

    /// Advance the round-robin cursor and return the next item, or `None` if
    /// the pool is empty.
    pub fn next_round_robin(&mut self) -> Option<&mut RotationItem> {
        if self.items.is_empty() {
            return None;
        }
        let idx = self.cursor % self.items.len();
        self.cursor = self.cursor.wrapping_add(1);
        let item = &mut self.items[idx];
        item.play_count += 1;
        Some(item)
    }

    /// Select the next item using weighted selection (deterministic: picks the
    /// item with the highest `weight / (play_count + 1)` ratio).
    #[allow(clippy::cast_precision_loss)]
    pub fn next_weighted(&mut self) -> Option<&mut RotationItem> {
        if self.items.is_empty() {
            return None;
        }
        let best_idx = self
            .items
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                let score_a = a.weight as f64 / (a.play_count as f64 + 1.0);
                let score_b = b.weight as f64 / (b.play_count as f64 + 1.0);
                score_a
                    .partial_cmp(&score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);
        let item = &mut self.items[best_idx];
        item.play_count += 1;
        Some(item)
    }

    /// Select the next item restricted to the daypart containing `hour`.
    #[allow(clippy::cast_precision_loss)]
    pub fn next_for_daypart(&mut self, hour: u8) -> Option<String> {
        let matching_label: Option<String> = self
            .dayparts
            .iter()
            .find(|dp| dp.contains_hour(hour))
            .map(|dp| dp.label.clone());

        let label = matching_label?;

        // Find best candidate in that daypart by lowest play_count
        let best_idx = self
            .items
            .iter()
            .enumerate()
            .filter(|(_, it)| it.daypart_label.as_deref() == Some(label.as_str()))
            .min_by_key(|(_, it)| it.play_count)
            .map(|(i, _)| i);

        if let Some(idx) = best_idx {
            self.items[idx].play_count += 1;
            Some(self.items[idx].asset_id.clone())
        } else {
            None
        }
    }

    /// Return a summary of play counts per item.
    pub fn play_count_summary(&self) -> HashMap<String, u64> {
        self.items
            .iter()
            .map(|it| (it.asset_id.clone(), it.play_count))
            .collect()
    }

    /// Reset all play counts to zero and rewind the cursor.
    pub fn reset(&mut self) {
        self.cursor = 0;
        for item in &mut self.items {
            item.play_count = 0;
        }
    }
}

// ---------------------------------------------------------------------------
// Rotation schedule (multiple pools)
// ---------------------------------------------------------------------------

/// A schedule that manages multiple rotation pools, one per time slot or
/// content category.
#[derive(Debug, Clone)]
pub struct RotationSchedule {
    /// Name of this schedule.
    pub name: String,
    /// Pools keyed by their name.
    pools: HashMap<String, RotationPool>,
}

impl RotationSchedule {
    /// Create a new empty rotation schedule.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            pools: HashMap::new(),
        }
    }

    /// Add a pool to the schedule.
    pub fn add_pool(&mut self, pool: RotationPool) {
        self.pools.insert(pool.name.clone(), pool);
    }

    /// Return a reference to a pool by name.
    pub fn pool(&self, name: &str) -> Option<&RotationPool> {
        self.pools.get(name)
    }

    /// Return a mutable reference to a pool by name.
    pub fn pool_mut(&mut self, name: &str) -> Option<&mut RotationPool> {
        self.pools.get_mut(name)
    }

    /// Return the number of registered pools.
    pub fn pool_count(&self) -> usize {
        self.pools.len()
    }

    /// Reset all pools.
    pub fn reset_all(&mut self) {
        for pool in self.pools.values_mut() {
            pool.reset();
        }
    }

    /// Total item count across all pools.
    pub fn total_items(&self) -> usize {
        self.pools.values().map(RotationPool::len).sum()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_robin_basic() {
        let mut pool = RotationPool::new("rr", RotationStrategy::RoundRobin);
        pool.add_item(RotationItem::new("a"));
        pool.add_item(RotationItem::new("b"));
        pool.add_item(RotationItem::new("c"));

        assert_eq!(
            pool.next_round_robin()
                .expect("should succeed in test")
                .asset_id,
            "a"
        );
        assert_eq!(
            pool.next_round_robin()
                .expect("should succeed in test")
                .asset_id,
            "b"
        );
        assert_eq!(
            pool.next_round_robin()
                .expect("should succeed in test")
                .asset_id,
            "c"
        );
        // Wraps around
        assert_eq!(
            pool.next_round_robin()
                .expect("should succeed in test")
                .asset_id,
            "a"
        );
    }

    #[test]
    fn test_round_robin_play_count() {
        let mut pool = RotationPool::new("rr", RotationStrategy::RoundRobin);
        pool.add_item(RotationItem::new("x"));
        pool.next_round_robin();
        pool.next_round_robin();
        let summary = pool.play_count_summary();
        assert_eq!(summary["x"], 2);
    }

    #[test]
    fn test_round_robin_empty() {
        let mut pool = RotationPool::new("empty", RotationStrategy::RoundRobin);
        assert!(pool.next_round_robin().is_none());
    }

    #[test]
    fn test_weighted_prefers_higher_weight() {
        let mut pool = RotationPool::new("w", RotationStrategy::Weighted);
        pool.add_item(RotationItem::new("low").with_weight(1));
        pool.add_item(RotationItem::new("high").with_weight(10));

        // First pick should be "high" (higher score)
        assert_eq!(
            pool.next_weighted()
                .expect("should succeed in test")
                .asset_id,
            "high"
        );
    }

    #[test]
    fn test_weighted_eventually_balances() {
        let mut pool = RotationPool::new("w", RotationStrategy::Weighted);
        pool.add_item(RotationItem::new("a").with_weight(2));
        pool.add_item(RotationItem::new("b").with_weight(1));

        // After many picks, "a" should have roughly 2x plays of "b"
        for _ in 0..30 {
            pool.next_weighted();
        }
        let s = pool.play_count_summary();
        assert!(s["a"] > s["b"]);
    }

    #[test]
    fn test_weighted_empty() {
        let mut pool = RotationPool::new("e", RotationStrategy::Weighted);
        assert!(pool.next_weighted().is_none());
    }

    #[test]
    fn test_daypart_slot_contains() {
        let morning = DaypartSlot::new("morning", 6, 11);
        assert!(morning.contains_hour(6));
        assert!(morning.contains_hour(9));
        assert!(morning.contains_hour(11));
        assert!(!morning.contains_hour(5));
        assert!(!morning.contains_hour(12));
    }

    #[test]
    fn test_daypart_slot_wraps_midnight() {
        let late = DaypartSlot::new("late_night", 22, 4);
        assert!(late.contains_hour(23));
        assert!(late.contains_hour(0));
        assert!(late.contains_hour(3));
        assert!(!late.contains_hour(10));
    }

    #[test]
    fn test_daypart_selection() {
        let mut pool = RotationPool::new("dp", RotationStrategy::Daypart);
        pool.add_daypart(DaypartSlot::new("morning", 6, 11));
        pool.add_daypart(DaypartSlot::new("evening", 18, 22));

        pool.add_item(RotationItem::new("am_clip").with_daypart("morning"));
        pool.add_item(RotationItem::new("pm_clip").with_daypart("evening"));

        assert_eq!(pool.next_for_daypart(8), Some("am_clip".to_string()));
        assert_eq!(pool.next_for_daypart(20), Some("pm_clip".to_string()));
        assert_eq!(pool.next_for_daypart(14), None); // no daypart covers 14
    }

    #[test]
    fn test_rotation_pool_reset() {
        let mut pool = RotationPool::new("r", RotationStrategy::RoundRobin);
        pool.add_item(RotationItem::new("a"));
        pool.next_round_robin();
        pool.next_round_robin();
        pool.reset();
        let summary = pool.play_count_summary();
        assert_eq!(summary["a"], 0);
        // After reset, cursor rewinds
        assert_eq!(
            pool.next_round_robin()
                .expect("should succeed in test")
                .asset_id,
            "a"
        );
    }

    #[test]
    fn test_rotation_schedule_pools() {
        let mut sched = RotationSchedule::new("daily");
        sched.add_pool(RotationPool::new("music", RotationStrategy::RoundRobin));
        sched.add_pool(RotationPool::new("promos", RotationStrategy::Weighted));
        assert_eq!(sched.pool_count(), 2);
        assert!(sched.pool("music").is_some());
        assert!(sched.pool("nonexistent").is_none());
    }

    #[test]
    fn test_rotation_schedule_total_items() {
        let mut sched = RotationSchedule::new("s");
        let mut p1 = RotationPool::new("a", RotationStrategy::RoundRobin);
        p1.add_item(RotationItem::new("x"));
        p1.add_item(RotationItem::new("y"));
        let mut p2 = RotationPool::new("b", RotationStrategy::Weighted);
        p2.add_item(RotationItem::new("z"));
        sched.add_pool(p1);
        sched.add_pool(p2);
        assert_eq!(sched.total_items(), 3);
    }

    #[test]
    fn test_rotation_schedule_reset_all() {
        let mut sched = RotationSchedule::new("s");
        let mut p = RotationPool::new("a", RotationStrategy::RoundRobin);
        p.add_item(RotationItem::new("x"));
        sched.add_pool(p);
        sched
            .pool_mut("a")
            .expect("should succeed in test")
            .next_round_robin();
        assert_eq!(
            sched
                .pool("a")
                .expect("should succeed in test")
                .play_count_summary()["x"],
            1
        );
        sched.reset_all();
        assert_eq!(
            sched
                .pool("a")
                .expect("should succeed in test")
                .play_count_summary()["x"],
            0
        );
    }

    #[test]
    fn test_rotation_item_builder() {
        let item = RotationItem::new("clip")
            .with_weight(5)
            .with_daypart("prime");
        assert_eq!(item.weight, 5);
        assert_eq!(item.daypart_label.as_deref(), Some("prime"));
        assert_eq!(item.play_count, 0);
    }
}
