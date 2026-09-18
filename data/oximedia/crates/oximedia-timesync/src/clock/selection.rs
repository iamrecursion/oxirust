//! Multi-source clock selection algorithm.

use super::ClockSource;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Clock source priority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Priority(pub u8);

impl Priority {
    /// Highest priority
    pub const HIGHEST: Self = Self(0);
    /// High priority
    pub const HIGH: Self = Self(64);
    /// Normal priority
    pub const NORMAL: Self = Self(128);
    /// Low priority
    pub const LOW: Self = Self(192);
    /// Lowest priority
    pub const LOWEST: Self = Self(255);
}

/// Clock source information.
#[derive(Debug, Clone)]
pub struct SourceInfo {
    /// Source type
    pub source: ClockSource,
    /// Priority (lower is better)
    pub priority: Priority,
    /// Last offset measurement
    pub last_offset: i64,
    /// Last update time
    pub last_update: Instant,
    /// Jitter estimate (nanoseconds)
    pub jitter_ns: u64,
    /// Number of valid samples
    pub valid_samples: u32,
}

/// Clock source selector.
pub struct SourceSelector {
    /// Available sources
    sources: HashMap<ClockSource, SourceInfo>,
    /// Currently selected source
    selected: Option<ClockSource>,
    /// Minimum samples required before selection
    min_samples: u32,
    /// Timeout for source validity
    timeout: Duration,
}

impl SourceSelector {
    /// Create a new source selector.
    #[must_use]
    pub fn new() -> Self {
        Self {
            sources: HashMap::new(),
            selected: None,
            min_samples: 5,
            timeout: Duration::from_secs(10),
        }
    }

    /// Add or update a clock source.
    pub fn update_source(
        &mut self,
        source: ClockSource,
        priority: Priority,
        offset_ns: i64,
        jitter_ns: u64,
    ) {
        let now = Instant::now();

        self.sources
            .entry(source)
            .and_modify(|info| {
                info.last_offset = offset_ns;
                info.last_update = now;
                info.jitter_ns = jitter_ns;
                info.valid_samples += 1;
            })
            .or_insert(SourceInfo {
                source,
                priority,
                last_offset: offset_ns,
                last_update: now,
                jitter_ns,
                valid_samples: 1,
            });

        // Re-evaluate selection
        self.select_best_source();
    }

    /// Remove a clock source.
    pub fn remove_source(&mut self, source: ClockSource) {
        self.sources.remove(&source);

        // If we removed the selected source, re-select
        if self.selected == Some(source) {
            self.selected = None;
            self.select_best_source();
        }
    }

    /// Get currently selected source.
    #[must_use]
    pub fn selected_source(&self) -> Option<ClockSource> {
        self.selected
    }

    /// Get information about the selected source.
    #[must_use]
    pub fn selected_info(&self) -> Option<&SourceInfo> {
        self.selected.and_then(|s| self.sources.get(&s))
    }

    /// Select the best available source.
    fn select_best_source(&mut self) {
        let now = Instant::now();

        // Filter valid sources
        let valid_sources: Vec<_> = self
            .sources
            .iter()
            .filter(|(_, info)| {
                // Must have minimum samples
                info.valid_samples >= self.min_samples
                    // Must be recent
                    && now.duration_since(info.last_update) < self.timeout
            })
            .collect();

        if valid_sources.is_empty() {
            self.selected = None;
            return;
        }

        // Select source with best (lowest) priority
        let best = valid_sources
            .iter()
            .min_by_key(|(_, info)| info.priority)
            .map(|(&source, _)| source);

        self.selected = best;
    }

    /// Clean up stale sources.
    pub fn cleanup_stale(&mut self) {
        let now = Instant::now();
        self.sources
            .retain(|_, info| now.duration_since(info.last_update) < self.timeout);

        // Re-select if necessary
        if let Some(selected) = self.selected {
            if !self.sources.contains_key(&selected) {
                self.selected = None;
                self.select_best_source();
            }
        }
    }

    /// Get all sources.
    #[must_use]
    pub fn sources(&self) -> &HashMap<ClockSource, SourceInfo> {
        &self.sources
    }
}

impl Default for SourceSelector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_selector() {
        let mut selector = SourceSelector::new();
        selector.min_samples = 2; // Lower threshold for testing

        // Add PTP source (high priority)
        for _ in 0..3 {
            selector.update_source(ClockSource::Ptp, Priority::HIGH, 100, 10);
        }

        // Add NTP source (normal priority)
        for _ in 0..3 {
            selector.update_source(ClockSource::Ntp, Priority::NORMAL, 200, 50);
        }

        // Should select PTP (better priority)
        assert_eq!(selector.selected_source(), Some(ClockSource::Ptp));
    }

    #[test]
    fn test_source_removal() {
        let mut selector = SourceSelector::new();
        selector.min_samples = 1;

        selector.update_source(ClockSource::Ptp, Priority::HIGH, 100, 10);
        selector.update_source(ClockSource::Ntp, Priority::NORMAL, 200, 50);

        assert_eq!(selector.selected_source(), Some(ClockSource::Ptp));

        selector.remove_source(ClockSource::Ptp);
        assert_eq!(selector.selected_source(), Some(ClockSource::Ntp));
    }

    #[test]
    fn test_priority_ordering() {
        assert!(Priority::HIGHEST < Priority::HIGH);
        assert!(Priority::HIGH < Priority::NORMAL);
        assert!(Priority::NORMAL < Priority::LOW);
        assert!(Priority::LOW < Priority::LOWEST);
    }
}
