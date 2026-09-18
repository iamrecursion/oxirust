#![allow(dead_code)]
//! Time-blocked schedule management for broadcast playout.
//!
//! This module provides a block-oriented scheduling system where the 24-hour
//! broadcast day is divided into named time blocks (e.g. "Morning News",
//! "Prime Time", "Late Night").  Each block carries its own priority, genre
//! tag, and content constraints.

use std::collections::BTreeMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Time-of-day helper
// ---------------------------------------------------------------------------

/// A simple time-of-day value (hour + minute + second).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TimeOfDay {
    /// Hour (0..=23).
    pub hour: u8,
    /// Minute (0..=59).
    pub minute: u8,
    /// Second (0..=59).
    pub second: u8,
}

impl TimeOfDay {
    /// Create a new time-of-day, clamping to valid ranges.
    pub fn new(hour: u8, minute: u8, second: u8) -> Self {
        Self {
            hour: hour.min(23),
            minute: minute.min(59),
            second: second.min(59),
        }
    }

    /// Total seconds since midnight.
    #[allow(clippy::cast_precision_loss)]
    pub fn total_seconds(&self) -> u32 {
        u32::from(self.hour) * 3600 + u32::from(self.minute) * 60 + u32::from(self.second)
    }

    /// Create from total seconds since midnight.
    pub fn from_seconds(total: u32) -> Self {
        let total = total % 86400;
        let hour = (total / 3600) as u8;
        let minute = ((total % 3600) / 60) as u8;
        let second = (total % 60) as u8;
        Self {
            hour,
            minute,
            second,
        }
    }

    /// Duration in seconds from self to other (wrapping past midnight).
    pub fn duration_to(&self, other: &TimeOfDay) -> u32 {
        let a = self.total_seconds();
        let b = other.total_seconds();
        if b >= a {
            b - a
        } else {
            86400 - a + b
        }
    }
}

impl fmt::Display for TimeOfDay {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:02}:{:02}:{:02}", self.hour, self.minute, self.second)
    }
}

// ---------------------------------------------------------------------------
// Block priority
// ---------------------------------------------------------------------------

/// Priority level of a schedule block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BlockPriority {
    /// Low priority (filler content).
    Low,
    /// Normal priority.
    Normal,
    /// High priority (premium content).
    High,
    /// Critical (live events, breaking news).
    Critical,
}

impl BlockPriority {
    /// Numeric weight for sorting/comparison.
    pub fn weight(&self) -> u32 {
        match self {
            Self::Low => 1,
            Self::Normal => 5,
            Self::High => 10,
            Self::Critical => 100,
        }
    }
}

// ---------------------------------------------------------------------------
// Schedule block
// ---------------------------------------------------------------------------

/// A named block of time in the broadcast schedule.
#[derive(Debug, Clone)]
pub struct ScheduleBlock {
    /// Unique identifier for this block.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Start time of the block.
    pub start: TimeOfDay,
    /// End time of the block (may wrap past midnight).
    pub end: TimeOfDay,
    /// Priority.
    pub priority: BlockPriority,
    /// Genre or category tag.
    pub genre: String,
    /// Whether the block is live.
    pub is_live: bool,
    /// Maximum number of ad breaks allowed.
    pub max_ad_breaks: u32,
    /// Notes or description.
    pub notes: String,
}

impl ScheduleBlock {
    /// Duration of the block in seconds.
    pub fn duration_seconds(&self) -> u32 {
        self.start.duration_to(&self.end)
    }

    /// Whether a given time falls within this block.
    pub fn contains(&self, time: &TimeOfDay) -> bool {
        let t = time.total_seconds();
        let s = self.start.total_seconds();
        let e = self.end.total_seconds();
        if e > s {
            t >= s && t < e
        } else {
            // Wraps past midnight
            t >= s || t < e
        }
    }

    /// Whether this block overlaps with another block.
    pub fn overlaps(&self, other: &ScheduleBlock) -> bool {
        // Check both directions: does any point in other fall within self?
        other.contains(&self.start)
            || other.contains(&TimeOfDay::from_seconds(
                self.end.total_seconds().wrapping_sub(1) % 86400,
            ))
            || self.contains(&other.start)
            || self.contains(&TimeOfDay::from_seconds(
                other.end.total_seconds().wrapping_sub(1) % 86400,
            ))
    }
}

// ---------------------------------------------------------------------------
// Schedule day
// ---------------------------------------------------------------------------

/// A full day schedule composed of blocks.
#[derive(Debug, Clone)]
pub struct ScheduleDay {
    /// Date label (e.g. "2024-01-15").
    pub date_label: String,
    /// Blocks in the day, keyed by block id.
    pub blocks: BTreeMap<String, ScheduleBlock>,
}

impl ScheduleDay {
    /// Create a new empty schedule day.
    pub fn new(date_label: &str) -> Self {
        Self {
            date_label: date_label.to_string(),
            blocks: BTreeMap::new(),
        }
    }

    /// Add a block to the day.  Returns `false` if it overlaps with an
    /// existing block of equal or higher priority.
    pub fn add_block(&mut self, block: ScheduleBlock) -> bool {
        for existing in self.blocks.values() {
            if existing.overlaps(&block) && existing.priority >= block.priority {
                return false;
            }
        }
        self.blocks.insert(block.id.clone(), block);
        true
    }

    /// Remove a block by id.  Returns the removed block if found.
    pub fn remove_block(&mut self, id: &str) -> Option<ScheduleBlock> {
        self.blocks.remove(id)
    }

    /// Find the block that is active at a given time.
    pub fn active_at(&self, time: &TimeOfDay) -> Option<&ScheduleBlock> {
        self.blocks.values().find(|b| b.contains(time))
    }

    /// Total scheduled seconds across all blocks.
    pub fn total_scheduled_seconds(&self) -> u32 {
        self.blocks
            .values()
            .map(ScheduleBlock::duration_seconds)
            .sum()
    }

    /// Find gaps (unscheduled time) in the day.
    pub fn find_gaps(&self) -> Vec<(TimeOfDay, TimeOfDay)> {
        if self.blocks.is_empty() {
            return vec![(TimeOfDay::new(0, 0, 0), TimeOfDay::new(23, 59, 59))];
        }

        // Collect start/end seconds, sort by start
        let mut intervals: Vec<(u32, u32)> = self
            .blocks
            .values()
            .map(|b| (b.start.total_seconds(), b.end.total_seconds()))
            .collect();
        intervals.sort_by_key(|&(s, _)| s);

        let mut gaps = Vec::new();
        let mut cursor = 0u32;

        for &(start, end) in &intervals {
            if start > cursor {
                gaps.push((
                    TimeOfDay::from_seconds(cursor),
                    TimeOfDay::from_seconds(start),
                ));
            }
            if end > cursor {
                cursor = end;
            }
        }

        if cursor < 86400 {
            gaps.push((TimeOfDay::from_seconds(cursor), TimeOfDay::new(23, 59, 59)));
        }

        gaps
    }

    /// Number of blocks.
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    /// Whether the schedule day has no blocks.
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Block template
// ---------------------------------------------------------------------------

/// A reusable template for creating schedule blocks.
#[derive(Debug, Clone)]
pub struct BlockTemplate {
    /// Template name.
    pub name: String,
    /// Default genre.
    pub genre: String,
    /// Default priority.
    pub priority: BlockPriority,
    /// Default duration in seconds.
    pub default_duration_sec: u32,
    /// Whether blocks from this template are typically live.
    pub is_live: bool,
    /// Default max ad breaks.
    pub max_ad_breaks: u32,
}

impl BlockTemplate {
    /// Instantiate a schedule block from this template at a given start time.
    pub fn instantiate(&self, id: &str, start: TimeOfDay) -> ScheduleBlock {
        let end =
            TimeOfDay::from_seconds((start.total_seconds() + self.default_duration_sec) % 86400);
        ScheduleBlock {
            id: id.to_string(),
            name: self.name.clone(),
            start,
            end,
            priority: self.priority,
            genre: self.genre.clone(),
            is_live: self.is_live,
            max_ad_breaks: self.max_ad_breaks,
            notes: String::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_of_day_new() {
        let t = TimeOfDay::new(14, 30, 45);
        assert_eq!(t.hour, 14);
        assert_eq!(t.minute, 30);
        assert_eq!(t.second, 45);
    }

    #[test]
    fn test_time_of_day_clamping() {
        let t = TimeOfDay::new(25, 70, 99);
        assert_eq!(t.hour, 23);
        assert_eq!(t.minute, 59);
        assert_eq!(t.second, 59);
    }

    #[test]
    fn test_total_seconds() {
        let t = TimeOfDay::new(1, 0, 0);
        assert_eq!(t.total_seconds(), 3600);
        let t2 = TimeOfDay::new(0, 1, 30);
        assert_eq!(t2.total_seconds(), 90);
    }

    #[test]
    fn test_from_seconds() {
        let t = TimeOfDay::from_seconds(3661);
        assert_eq!(t.hour, 1);
        assert_eq!(t.minute, 1);
        assert_eq!(t.second, 1);
    }

    #[test]
    fn test_from_seconds_wraps() {
        let t = TimeOfDay::from_seconds(86400 + 60);
        assert_eq!(t.hour, 0);
        assert_eq!(t.minute, 1);
        assert_eq!(t.second, 0);
    }

    #[test]
    fn test_duration_to() {
        let a = TimeOfDay::new(10, 0, 0);
        let b = TimeOfDay::new(12, 0, 0);
        assert_eq!(a.duration_to(&b), 7200);
    }

    #[test]
    fn test_duration_to_midnight_wrap() {
        let a = TimeOfDay::new(23, 0, 0);
        let b = TimeOfDay::new(1, 0, 0);
        assert_eq!(a.duration_to(&b), 7200);
    }

    #[test]
    fn test_time_display() {
        let t = TimeOfDay::new(8, 5, 9);
        assert_eq!(format!("{t}"), "08:05:09");
    }

    #[test]
    fn test_block_duration() {
        let block = ScheduleBlock {
            id: "b1".into(),
            name: "Morning".into(),
            start: TimeOfDay::new(6, 0, 0),
            end: TimeOfDay::new(9, 0, 0),
            priority: BlockPriority::Normal,
            genre: "News".into(),
            is_live: true,
            max_ad_breaks: 4,
            notes: String::new(),
        };
        assert_eq!(block.duration_seconds(), 10800);
    }

    #[test]
    fn test_block_contains() {
        let block = ScheduleBlock {
            id: "b1".into(),
            name: "Test".into(),
            start: TimeOfDay::new(10, 0, 0),
            end: TimeOfDay::new(12, 0, 0),
            priority: BlockPriority::Normal,
            genre: String::new(),
            is_live: false,
            max_ad_breaks: 0,
            notes: String::new(),
        };
        assert!(block.contains(&TimeOfDay::new(11, 0, 0)));
        assert!(!block.contains(&TimeOfDay::new(9, 0, 0)));
        assert!(!block.contains(&TimeOfDay::new(12, 0, 0)));
    }

    #[test]
    fn test_schedule_day_add_and_active() {
        let mut day = ScheduleDay::new("2024-01-15");
        let block = ScheduleBlock {
            id: "morning".into(),
            name: "Morning Show".into(),
            start: TimeOfDay::new(6, 0, 0),
            end: TimeOfDay::new(9, 0, 0),
            priority: BlockPriority::Normal,
            genre: "Talk".into(),
            is_live: true,
            max_ad_breaks: 3,
            notes: String::new(),
        };
        assert!(day.add_block(block));
        assert_eq!(day.len(), 1);

        let active = day.active_at(&TimeOfDay::new(7, 30, 0));
        assert!(active.is_some());
        assert_eq!(active.expect("should succeed in test").id, "morning");
    }

    #[test]
    fn test_schedule_day_gaps() {
        let mut day = ScheduleDay::new("2024-01-15");
        day.add_block(ScheduleBlock {
            id: "a".into(),
            name: "A".into(),
            start: TimeOfDay::new(6, 0, 0),
            end: TimeOfDay::new(9, 0, 0),
            priority: BlockPriority::Normal,
            genre: String::new(),
            is_live: false,
            max_ad_breaks: 0,
            notes: String::new(),
        });
        day.add_block(ScheduleBlock {
            id: "b".into(),
            name: "B".into(),
            start: TimeOfDay::new(12, 0, 0),
            end: TimeOfDay::new(14, 0, 0),
            priority: BlockPriority::Normal,
            genre: String::new(),
            is_live: false,
            max_ad_breaks: 0,
            notes: String::new(),
        });
        let gaps = day.find_gaps();
        // Gap before 6:00, between 9:00-12:00, and after 14:00
        assert_eq!(gaps.len(), 3);
    }

    #[test]
    fn test_block_priority_weight() {
        assert!(BlockPriority::Critical.weight() > BlockPriority::High.weight());
        assert!(BlockPriority::High.weight() > BlockPriority::Normal.weight());
        assert!(BlockPriority::Normal.weight() > BlockPriority::Low.weight());
    }

    #[test]
    fn test_block_template_instantiate() {
        let template = BlockTemplate {
            name: "News Hour".into(),
            genre: "News".into(),
            priority: BlockPriority::High,
            default_duration_sec: 3600,
            is_live: true,
            max_ad_breaks: 2,
        };
        let block = template.instantiate("news_1", TimeOfDay::new(18, 0, 0));
        assert_eq!(block.name, "News Hour");
        assert_eq!(block.start, TimeOfDay::new(18, 0, 0));
        assert_eq!(block.end, TimeOfDay::new(19, 0, 0));
        assert!(block.is_live);
    }

    #[test]
    fn test_empty_day() {
        let day = ScheduleDay::new("2024-01-01");
        assert!(day.is_empty());
        assert_eq!(day.total_scheduled_seconds(), 0);
        let gaps = day.find_gaps();
        assert_eq!(gaps.len(), 1);
    }
}
