#![allow(dead_code)]
//! Reusable schedule templates for recurring broadcast patterns.
//!
//! Allows operators to define template schedules (e.g., "weekday prime-time",
//! "weekend morning block") with time slots, default content mappings, and
//! override rules. Templates can be instantiated into concrete schedules for
//! specific dates, supporting recurrence patterns and exception handling.

use std::collections::HashMap;
use std::fmt;

/// Day-of-week for template recurrence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DayOfWeek {
    /// Monday.
    Monday,
    /// Tuesday.
    Tuesday,
    /// Wednesday.
    Wednesday,
    /// Thursday.
    Thursday,
    /// Friday.
    Friday,
    /// Saturday.
    Saturday,
    /// Sunday.
    Sunday,
}

impl fmt::Display for DayOfWeek {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Monday => "Monday",
            Self::Tuesday => "Tuesday",
            Self::Wednesday => "Wednesday",
            Self::Thursday => "Thursday",
            Self::Friday => "Friday",
            Self::Saturday => "Saturday",
            Self::Sunday => "Sunday",
        };
        write!(f, "{name}")
    }
}

impl DayOfWeek {
    /// Returns all weekdays (Mon-Fri).
    pub fn weekdays() -> Vec<Self> {
        vec![
            Self::Monday,
            Self::Tuesday,
            Self::Wednesday,
            Self::Thursday,
            Self::Friday,
        ]
    }

    /// Returns weekend days (Sat-Sun).
    pub fn weekends() -> Vec<Self> {
        vec![Self::Saturday, Self::Sunday]
    }

    /// Returns all days.
    pub fn all() -> Vec<Self> {
        vec![
            Self::Monday,
            Self::Tuesday,
            Self::Wednesday,
            Self::Thursday,
            Self::Friday,
            Self::Saturday,
            Self::Sunday,
        ]
    }
}

/// Time-of-day in HH:MM (24-hour) represented as minutes since midnight.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TimeSlot {
    /// Minutes since midnight (0..1440).
    pub minutes: u16,
}

impl TimeSlot {
    /// Create a time slot from hours and minutes.
    ///
    /// Clamps to valid range (0..1440).
    pub fn new(hour: u16, minute: u16) -> Self {
        let total = hour * 60 + minute;
        Self {
            minutes: total.min(1439),
        }
    }

    /// Returns the hour component.
    pub fn hour(self) -> u16 {
        self.minutes / 60
    }

    /// Returns the minute component.
    pub fn minute(self) -> u16 {
        self.minutes % 60
    }

    /// Duration in minutes between two time slots.
    pub fn duration_to(self, other: Self) -> u16 {
        if other.minutes >= self.minutes {
            other.minutes - self.minutes
        } else {
            // Wraps past midnight.
            1440 - self.minutes + other.minutes
        }
    }
}

impl fmt::Display for TimeSlot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:02}:{:02}", self.hour(), self.minute())
    }
}

/// A single entry in a schedule template.
#[derive(Debug, Clone)]
pub struct TemplateEntry {
    /// Start time of this entry.
    pub start: TimeSlot,
    /// End time of this entry.
    pub end: TimeSlot,
    /// Default content or action identifier.
    pub content_id: String,
    /// Human-readable label.
    pub label: String,
    /// Whether this slot can be overridden.
    pub overridable: bool,
    /// Priority (higher = harder to preempt).
    pub priority: u8,
}

impl TemplateEntry {
    /// Create a new template entry.
    pub fn new(start: TimeSlot, end: TimeSlot, content_id: &str, label: &str) -> Self {
        Self {
            start,
            end,
            content_id: content_id.to_string(),
            label: label.to_string(),
            overridable: true,
            priority: 50,
        }
    }

    /// Duration of this entry in minutes.
    pub fn duration_minutes(&self) -> u16 {
        self.start.duration_to(self.end)
    }
}

/// Recurrence pattern for a schedule template.
#[derive(Debug, Clone)]
pub struct RecurrencePattern {
    /// Days of the week this template applies to.
    pub days: Vec<DayOfWeek>,
    /// Exception dates (YYYYMMDD format as u32) where the template does NOT apply.
    pub exceptions: Vec<u32>,
    /// Whether the template is currently active.
    pub active: bool,
}

impl RecurrencePattern {
    /// Create a weekday-only recurrence.
    pub fn weekdays() -> Self {
        Self {
            days: DayOfWeek::weekdays(),
            exceptions: Vec::new(),
            active: true,
        }
    }

    /// Create a weekend-only recurrence.
    pub fn weekends() -> Self {
        Self {
            days: DayOfWeek::weekends(),
            exceptions: Vec::new(),
            active: true,
        }
    }

    /// Create a daily recurrence.
    pub fn daily() -> Self {
        Self {
            days: DayOfWeek::all(),
            exceptions: Vec::new(),
            active: true,
        }
    }

    /// Check if the pattern applies on a given day.
    pub fn applies_on(&self, day: DayOfWeek) -> bool {
        self.active && self.days.contains(&day)
    }

    /// Check if a date (YYYYMMDD) is an exception.
    pub fn is_exception(&self, date: u32) -> bool {
        self.exceptions.contains(&date)
    }

    /// Add an exception date.
    pub fn add_exception(&mut self, date: u32) {
        if !self.exceptions.contains(&date) {
            self.exceptions.push(date);
        }
    }
}

impl Default for RecurrencePattern {
    fn default() -> Self {
        Self::daily()
    }
}

/// A complete schedule template.
#[derive(Debug, Clone)]
pub struct ScheduleTemplate {
    /// Template identifier.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Description of the template.
    pub description: String,
    /// Entries in this template, sorted by start time.
    pub entries: Vec<TemplateEntry>,
    /// Recurrence pattern.
    pub recurrence: RecurrencePattern,
    /// Metadata tags.
    pub tags: HashMap<String, String>,
}

impl ScheduleTemplate {
    /// Create a new empty schedule template.
    pub fn new(id: &str, name: &str) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            description: String::new(),
            entries: Vec::new(),
            recurrence: RecurrencePattern::default(),
            tags: HashMap::new(),
        }
    }

    /// Add an entry to the template.
    pub fn add_entry(&mut self, entry: TemplateEntry) {
        self.entries.push(entry);
        self.entries.sort_by_key(|e| e.start.minutes);
    }

    /// Number of entries in the template.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Total scheduled minutes across all entries.
    #[allow(clippy::cast_precision_loss)]
    pub fn total_minutes(&self) -> u32 {
        self.entries
            .iter()
            .map(|e| u32::from(e.duration_minutes()))
            .sum()
    }

    /// Find entries that overlap with a given time range.
    pub fn entries_in_range(&self, start: TimeSlot, end: TimeSlot) -> Vec<&TemplateEntry> {
        self.entries
            .iter()
            .filter(|e| e.start.minutes < end.minutes && e.end.minutes > start.minutes)
            .collect()
    }

    /// Check for overlapping entries (validation).
    pub fn has_overlaps(&self) -> bool {
        for i in 0..self.entries.len() {
            for j in (i + 1)..self.entries.len() {
                let a = &self.entries[i];
                let b = &self.entries[j];
                if a.start.minutes < b.end.minutes && b.start.minutes < a.end.minutes {
                    return true;
                }
            }
        }
        false
    }

    /// Get all overridable entries.
    pub fn overridable_entries(&self) -> Vec<&TemplateEntry> {
        self.entries.iter().filter(|e| e.overridable).collect()
    }
}

/// Registry of schedule templates.
#[derive(Debug, Default)]
pub struct TemplateRegistry {
    /// Templates by ID.
    templates: HashMap<String, ScheduleTemplate>,
}

impl TemplateRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            templates: HashMap::new(),
        }
    }

    /// Register a template.
    pub fn register(&mut self, template: ScheduleTemplate) {
        self.templates.insert(template.id.clone(), template);
    }

    /// Look up a template by ID.
    pub fn get(&self, id: &str) -> Option<&ScheduleTemplate> {
        self.templates.get(id)
    }

    /// Remove a template by ID.
    pub fn remove(&mut self, id: &str) -> Option<ScheduleTemplate> {
        self.templates.remove(id)
    }

    /// Number of registered templates.
    pub fn count(&self) -> usize {
        self.templates.len()
    }

    /// List all template IDs.
    pub fn list_ids(&self) -> Vec<&str> {
        self.templates.keys().map(String::as_str).collect()
    }

    /// Find templates that apply on a given day.
    pub fn templates_for_day(&self, day: DayOfWeek) -> Vec<&ScheduleTemplate> {
        self.templates
            .values()
            .filter(|t| t.recurrence.applies_on(day))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_slot_new() {
        let slot = TimeSlot::new(14, 30);
        assert_eq!(slot.hour(), 14);
        assert_eq!(slot.minute(), 30);
        assert_eq!(slot.minutes, 870);
    }

    #[test]
    fn test_time_slot_display() {
        let slot = TimeSlot::new(9, 5);
        assert_eq!(slot.to_string(), "09:05");
    }

    #[test]
    fn test_time_slot_duration() {
        let start = TimeSlot::new(10, 0);
        let end = TimeSlot::new(11, 30);
        assert_eq!(start.duration_to(end), 90);
    }

    #[test]
    fn test_time_slot_duration_wrap_midnight() {
        let start = TimeSlot::new(23, 0);
        let end = TimeSlot::new(1, 0);
        assert_eq!(start.duration_to(end), 120);
    }

    #[test]
    fn test_day_of_week_weekdays() {
        let wd = DayOfWeek::weekdays();
        assert_eq!(wd.len(), 5);
        assert!(!wd.contains(&DayOfWeek::Saturday));
    }

    #[test]
    fn test_day_of_week_display() {
        assert_eq!(DayOfWeek::Monday.to_string(), "Monday");
        assert_eq!(DayOfWeek::Sunday.to_string(), "Sunday");
    }

    #[test]
    fn test_template_entry_duration() {
        let entry = TemplateEntry::new(
            TimeSlot::new(20, 0),
            TimeSlot::new(21, 0),
            "show_1",
            "Prime Time Show",
        );
        assert_eq!(entry.duration_minutes(), 60);
    }

    #[test]
    fn test_recurrence_pattern_weekdays() {
        let pat = RecurrencePattern::weekdays();
        assert!(pat.applies_on(DayOfWeek::Monday));
        assert!(pat.applies_on(DayOfWeek::Friday));
        assert!(!pat.applies_on(DayOfWeek::Saturday));
    }

    #[test]
    fn test_recurrence_exceptions() {
        let mut pat = RecurrencePattern::daily();
        pat.add_exception(20260101);
        assert!(pat.is_exception(20260101));
        assert!(!pat.is_exception(20260102));
    }

    #[test]
    fn test_schedule_template_add_entries() {
        let mut tmpl = ScheduleTemplate::new("t1", "Test Template");
        tmpl.add_entry(TemplateEntry::new(
            TimeSlot::new(20, 0),
            TimeSlot::new(21, 0),
            "show_a",
            "Show A",
        ));
        tmpl.add_entry(TemplateEntry::new(
            TimeSlot::new(18, 0),
            TimeSlot::new(19, 0),
            "show_b",
            "Show B",
        ));
        assert_eq!(tmpl.entry_count(), 2);
        // Should be sorted by start time.
        assert_eq!(tmpl.entries[0].content_id, "show_b");
    }

    #[test]
    fn test_schedule_template_total_minutes() {
        let mut tmpl = ScheduleTemplate::new("t2", "Minutes Test");
        tmpl.add_entry(TemplateEntry::new(
            TimeSlot::new(8, 0),
            TimeSlot::new(9, 0),
            "morning",
            "Morning",
        ));
        tmpl.add_entry(TemplateEntry::new(
            TimeSlot::new(12, 0),
            TimeSlot::new(13, 30),
            "noon",
            "Noon",
        ));
        assert_eq!(tmpl.total_minutes(), 150);
    }

    #[test]
    fn test_schedule_template_has_overlaps() {
        let mut tmpl = ScheduleTemplate::new("t3", "Overlap Test");
        tmpl.entries.push(TemplateEntry::new(
            TimeSlot::new(10, 0),
            TimeSlot::new(11, 0),
            "a",
            "A",
        ));
        tmpl.entries.push(TemplateEntry::new(
            TimeSlot::new(10, 30),
            TimeSlot::new(11, 30),
            "b",
            "B",
        ));
        assert!(tmpl.has_overlaps());
    }

    #[test]
    fn test_template_registry_operations() {
        let mut reg = TemplateRegistry::new();
        let tmpl = ScheduleTemplate::new("prime", "Prime Time");
        reg.register(tmpl);
        assert_eq!(reg.count(), 1);
        assert!(reg.get("prime").is_some());
        assert!(reg.get("nonexistent").is_none());
    }

    #[test]
    fn test_template_registry_remove() {
        let mut reg = TemplateRegistry::new();
        reg.register(ScheduleTemplate::new("del", "To Delete"));
        assert_eq!(reg.count(), 1);
        let removed = reg.remove("del");
        assert!(removed.is_some());
        assert_eq!(reg.count(), 0);
    }

    #[test]
    fn test_template_registry_templates_for_day() {
        let mut reg = TemplateRegistry::new();
        let mut weekday_tmpl = ScheduleTemplate::new("wd", "Weekday");
        weekday_tmpl.recurrence = RecurrencePattern::weekdays();
        let mut weekend_tmpl = ScheduleTemplate::new("we", "Weekend");
        weekend_tmpl.recurrence = RecurrencePattern::weekends();
        reg.register(weekday_tmpl);
        reg.register(weekend_tmpl);

        let mon = reg.templates_for_day(DayOfWeek::Monday);
        assert_eq!(mon.len(), 1);
        assert_eq!(mon[0].id, "wd");

        let sat = reg.templates_for_day(DayOfWeek::Saturday);
        assert_eq!(sat.len(), 1);
        assert_eq!(sat[0].id, "we");
    }
}
