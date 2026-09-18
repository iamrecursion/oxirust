//! Priority handling for live content.

use serde::{Deserialize, Serialize};

/// Priority levels for content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(u32)]
#[derive(Default)]
pub enum Priority {
    /// Lowest priority - filler content.
    Filler = 0,

    /// Low priority - regular programming.
    Low = 10,

    /// Normal priority - scheduled content.
    #[default]
    Normal = 50,

    /// High priority - important programming.
    High = 100,

    /// Critical priority - breaking news.
    Critical = 500,

    /// Emergency priority - emergency broadcast.
    Emergency = 1000,
}

impl Priority {
    /// Converts to numeric value.
    #[must_use]
    pub const fn as_u32(self) -> u32 {
        self as u32
    }

    /// Creates from numeric value.
    #[must_use]
    pub const fn from_u32(value: u32) -> Self {
        match value {
            0..=9 => Self::Filler,
            10..=49 => Self::Low,
            50..=99 => Self::Normal,
            100..=499 => Self::High,
            500..=999 => Self::Critical,
            _ => Self::Emergency,
        }
    }
}

impl From<u32> for Priority {
    fn from(value: u32) -> Self {
        Self::from_u32(value)
    }
}

impl From<Priority> for u32 {
    fn from(priority: Priority) -> Self {
        priority.as_u32()
    }
}

/// Priority rule for content selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorityRule {
    /// Minimum priority to interrupt current content.
    pub interrupt_threshold: Priority,

    /// Whether to allow lower priority content to continue.
    pub allow_completion: bool,

    /// Maximum time to wait before forcing interruption.
    pub max_wait_seconds: Option<u32>,
}

impl Default for PriorityRule {
    fn default() -> Self {
        Self {
            interrupt_threshold: Priority::High,
            allow_completion: true,
            max_wait_seconds: Some(30),
        }
    }
}

/// Manager for priority-based content selection.
#[derive(Debug)]
pub struct PriorityManager {
    current_priority: Priority,
    rules: PriorityRule,
}

impl PriorityManager {
    /// Creates a new priority manager.
    #[must_use]
    pub fn new(rules: PriorityRule) -> Self {
        Self {
            current_priority: Priority::Normal,
            rules,
        }
    }

    /// Sets the current priority.
    pub fn set_current_priority(&mut self, priority: Priority) {
        self.current_priority = priority;
    }

    /// Gets the current priority.
    #[must_use]
    pub const fn current_priority(&self) -> Priority {
        self.current_priority
    }

    /// Determines if content with the given priority can interrupt current content.
    #[must_use]
    pub fn can_interrupt(&self, new_priority: Priority) -> bool {
        new_priority >= self.rules.interrupt_threshold && new_priority > self.current_priority
    }

    /// Determines if content should wait for current content to complete.
    #[must_use]
    pub fn should_wait(&self, new_priority: Priority) -> bool {
        self.rules.allow_completion
            && new_priority < self.rules.interrupt_threshold
            && new_priority > self.current_priority
    }

    /// Updates priority rules.
    pub fn update_rules(&mut self, rules: PriorityRule) {
        self.rules = rules;
    }
}

impl Default for PriorityManager {
    fn default() -> Self {
        Self::new(PriorityRule::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_priority_ordering() {
        assert!(Priority::Emergency > Priority::Critical);
        assert!(Priority::Critical > Priority::High);
        assert!(Priority::High > Priority::Normal);
        assert!(Priority::Normal > Priority::Low);
        assert!(Priority::Low > Priority::Filler);
    }

    #[test]
    fn test_priority_conversion() {
        assert_eq!(Priority::from_u32(0), Priority::Filler);
        assert_eq!(Priority::from_u32(50), Priority::Normal);
        assert_eq!(Priority::from_u32(1000), Priority::Emergency);

        assert_eq!(Priority::Normal.as_u32(), 50);
    }

    #[test]
    fn test_priority_manager() {
        let mut manager = PriorityManager::default();
        manager.set_current_priority(Priority::Normal);

        assert!(manager.can_interrupt(Priority::Critical));
        assert!(!manager.can_interrupt(Priority::Low));
    }

    #[test]
    fn test_should_wait() {
        let rules = PriorityRule {
            interrupt_threshold: Priority::Critical,
            allow_completion: true,
            max_wait_seconds: Some(30),
        };
        let mut manager = PriorityManager::new(rules);
        manager.set_current_priority(Priority::Normal);

        // High priority is above Normal but below Critical threshold, so it should wait
        assert!(manager.should_wait(Priority::High));
        // Low priority is below Normal, so it should not wait
        assert!(!manager.should_wait(Priority::Low));
    }
}
