//! Fixity checking schedule management

use crate::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Fixity check frequency
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CheckFrequency {
    /// Daily checks
    Daily,
    /// Weekly checks
    Weekly,
    /// Monthly checks
    Monthly,
    /// Quarterly checks
    Quarterly,
    /// Yearly checks
    Yearly,
    /// Custom interval in days
    Custom(u32),
}

impl CheckFrequency {
    /// Get the interval in days
    #[must_use]
    pub const fn days(&self) -> u32 {
        match self {
            Self::Daily => 1,
            Self::Weekly => 7,
            Self::Monthly => 30,
            Self::Quarterly => 90,
            Self::Yearly => 365,
            Self::Custom(days) => *days,
        }
    }
}

/// Fixity check schedule for a file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixitySchedule {
    /// File path
    pub path: PathBuf,
    /// Check frequency
    pub frequency: CheckFrequency,
    /// Last check timestamp
    pub last_check: Option<chrono::DateTime<chrono::Utc>>,
    /// Next scheduled check
    pub next_check: chrono::DateTime<chrono::Utc>,
    /// Priority (higher = more important)
    pub priority: u8,
}

impl FixitySchedule {
    /// Create a new fixity schedule
    #[must_use]
    pub fn new(path: PathBuf, frequency: CheckFrequency) -> Self {
        let next_check = chrono::Utc::now() + chrono::Duration::days(i64::from(frequency.days()));

        Self {
            path,
            frequency,
            last_check: None,
            next_check,
            priority: 5,
        }
    }

    /// Check if a check is due
    #[must_use]
    pub fn is_due(&self) -> bool {
        chrono::Utc::now() >= self.next_check
    }

    /// Mark as checked
    pub fn mark_checked(&mut self) {
        self.last_check = Some(chrono::Utc::now());
        self.next_check =
            chrono::Utc::now() + chrono::Duration::days(i64::from(self.frequency.days()));
    }
}

/// Fixity scheduler
pub struct FixityScheduler {
    schedules: Vec<FixitySchedule>,
}

impl Default for FixityScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl FixityScheduler {
    /// Create a new fixity scheduler
    #[must_use]
    pub fn new() -> Self {
        Self {
            schedules: Vec::new(),
        }
    }

    /// Add a schedule
    pub fn add_schedule(&mut self, schedule: FixitySchedule) {
        self.schedules.push(schedule);
    }

    /// Get due schedules
    #[must_use]
    pub fn get_due_schedules(&self) -> Vec<&FixitySchedule> {
        self.schedules.iter().filter(|s| s.is_due()).collect()
    }

    /// Get schedules sorted by priority
    #[must_use]
    pub fn get_prioritized_schedules(&self) -> Vec<&FixitySchedule> {
        let mut schedules: Vec<&FixitySchedule> = self.schedules.iter().collect();
        schedules.sort_by(|a, b| b.priority.cmp(&a.priority));
        schedules
    }

    /// Mark a file as checked
    pub fn mark_checked(&mut self, path: &PathBuf) {
        if let Some(schedule) = self.schedules.iter_mut().find(|s| &s.path == path) {
            schedule.mark_checked();
        }
    }

    /// Save schedules to file
    ///
    /// # Errors
    ///
    /// Returns an error if save fails
    pub fn save(&self, path: &PathBuf) -> Result<()> {
        let json = serde_json::to_string_pretty(&self.schedules)
            .map_err(|e| crate::Error::Metadata(format!("JSON serialization failed: {e}")))?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load schedules from file
    ///
    /// # Errors
    ///
    /// Returns an error if load fails
    pub fn load(path: &PathBuf) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let schedules = serde_json::from_str(&content)
            .map_err(|e| crate::Error::Metadata(format!("JSON parse failed: {e}")))?;
        Ok(Self { schedules })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_frequency_days() {
        assert_eq!(CheckFrequency::Daily.days(), 1);
        assert_eq!(CheckFrequency::Weekly.days(), 7);
        assert_eq!(CheckFrequency::Monthly.days(), 30);
        assert_eq!(CheckFrequency::Custom(14).days(), 14);
    }

    #[test]
    fn test_fixity_schedule() {
        let schedule = FixitySchedule::new(PathBuf::from("test.mkv"), CheckFrequency::Daily);

        assert_eq!(schedule.frequency, CheckFrequency::Daily);
        assert!(schedule.last_check.is_none());
    }

    #[test]
    fn test_mark_checked() {
        let mut schedule = FixitySchedule::new(PathBuf::from("test.mkv"), CheckFrequency::Weekly);

        schedule.mark_checked();
        assert!(schedule.last_check.is_some());
    }

    #[test]
    fn test_scheduler() {
        let mut scheduler = FixityScheduler::new();
        scheduler.add_schedule(FixitySchedule::new(
            PathBuf::from("file1.mkv"),
            CheckFrequency::Daily,
        ));
        scheduler.add_schedule(FixitySchedule::new(
            PathBuf::from("file2.mkv"),
            CheckFrequency::Weekly,
        ));

        let schedules = scheduler.get_prioritized_schedules();
        assert_eq!(schedules.len(), 2);
    }
}
