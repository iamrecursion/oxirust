//! Time offset management.

use chrono::{DateTime, Duration as ChronoDuration, Utc};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Time offset configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeOffset {
    /// Offset amount.
    pub offset: Duration,

    /// Whether the offset is positive (add) or negative (subtract).
    pub is_positive: bool,

    /// Description of this offset.
    pub description: Option<String>,
}

impl TimeOffset {
    /// Creates a new time offset.
    #[must_use]
    pub const fn new(offset: Duration, is_positive: bool) -> Self {
        Self {
            offset,
            is_positive,
            description: None,
        }
    }

    /// Creates a positive offset.
    #[must_use]
    pub const fn positive(offset: Duration) -> Self {
        Self::new(offset, true)
    }

    /// Creates a negative offset.
    #[must_use]
    pub const fn negative(offset: Duration) -> Self {
        Self::new(offset, false)
    }

    /// Sets the description.
    #[must_use]
    pub fn with_description<S: Into<String>>(mut self, description: S) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Applies this offset to a timestamp.
    #[must_use]
    pub fn apply(&self, time: DateTime<Utc>) -> DateTime<Utc> {
        if let Ok(chrono_offset) = ChronoDuration::from_std(self.offset) {
            if self.is_positive {
                time + chrono_offset
            } else {
                time - chrono_offset
            }
        } else {
            time
        }
    }

    /// Gets the signed offset in seconds.
    #[must_use]
    pub fn as_signed_seconds(&self) -> i64 {
        let seconds = self.offset.as_secs() as i64;
        if self.is_positive {
            seconds
        } else {
            -seconds
        }
    }
}

impl Default for TimeOffset {
    fn default() -> Self {
        Self::new(Duration::ZERO, true)
    }
}

/// Manager for multiple time offsets.
#[derive(Debug, Default)]
pub struct OffsetManager {
    offsets: Vec<(String, TimeOffset)>,
    active_offset: Option<String>,
}

impl OffsetManager {
    /// Creates a new offset manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a named offset.
    pub fn add_offset<S: Into<String>>(&mut self, name: S, offset: TimeOffset) {
        let name = name.into();
        self.offsets.retain(|(n, _)| n != &name);
        self.offsets.push((name, offset));
    }

    /// Removes an offset by name.
    pub fn remove_offset(&mut self, name: &str) {
        self.offsets.retain(|(n, _)| n != name);
        if self.active_offset.as_deref() == Some(name) {
            self.active_offset = None;
        }
    }

    /// Sets the active offset.
    pub fn set_active(&mut self, name: &str) -> Result<(), String> {
        if self.offsets.iter().any(|(n, _)| n == name) {
            self.active_offset = Some(name.to_string());
            Ok(())
        } else {
            Err(format!("Offset '{name}' not found"))
        }
    }

    /// Clears the active offset.
    pub fn clear_active(&mut self) {
        self.active_offset = None;
    }

    /// Gets the active offset.
    #[must_use]
    pub fn get_active_offset(&self) -> Option<&TimeOffset> {
        self.active_offset
            .as_ref()
            .and_then(|name| self.offsets.iter().find(|(n, _)| n == name))
            .map(|(_, offset)| offset)
    }

    /// Applies the active offset to a timestamp.
    #[must_use]
    pub fn apply(&self, time: DateTime<Utc>) -> DateTime<Utc> {
        if let Some(offset) = self.get_active_offset() {
            offset.apply(time)
        } else {
            time
        }
    }

    /// Gets all offset names.
    #[must_use]
    pub fn offset_names(&self) -> Vec<&str> {
        self.offsets.iter().map(|(name, _)| name.as_str()).collect()
    }

    /// Returns the number of offsets.
    #[must_use]
    pub fn len(&self) -> usize {
        self.offsets.len()
    }

    /// Returns true if there are no offsets.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.offsets.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_offset() {
        let offset = TimeOffset::positive(Duration::from_secs(3600));
        let now = Utc::now();
        let adjusted = offset.apply(now);

        assert!(adjusted > now);
        assert_eq!(offset.as_signed_seconds(), 3600);
    }

    #[test]
    fn test_negative_offset() {
        let offset = TimeOffset::negative(Duration::from_secs(1800));
        let now = Utc::now();
        let adjusted = offset.apply(now);

        assert!(adjusted < now);
        assert_eq!(offset.as_signed_seconds(), -1800);
    }

    #[test]
    fn test_offset_manager() {
        let mut manager = OffsetManager::new();

        let offset1 = TimeOffset::positive(Duration::from_secs(3600)).with_description("UTC+1");
        let offset2 = TimeOffset::negative(Duration::from_secs(18000)).with_description("UTC-5");

        manager.add_offset("europe", offset1);
        manager.add_offset("us_east", offset2);

        assert_eq!(manager.len(), 2);

        manager
            .set_active("europe")
            .expect("should succeed in test");
        assert!(manager.get_active_offset().is_some());

        let now = Utc::now();
        let adjusted = manager.apply(now);
        assert!(adjusted > now);
    }

    // ── Wave 27: deterministic drift / resync arithmetic ─────────────────────

    /// A fixed, literal UTC instant — 2026-06-06T12:00:00Z — so the assertions
    /// are deterministic and do not depend on `Utc::now()`.
    fn fixed_instant() -> DateTime<Utc> {
        DateTime::from_timestamp(1_780_488_000, 0).expect("valid fixed timestamp")
    }

    #[test]
    fn test_drift_plus_100ms_resync() {
        let base = fixed_instant();
        let offset = TimeOffset::positive(Duration::from_millis(100));
        let adjusted = offset.apply(base);
        assert_eq!(adjusted, base + ChronoDuration::milliseconds(100));
        // Sanity: exactly +100ms, no rounding drift.
        assert_eq!((adjusted - base).num_milliseconds(), 100);
    }

    #[test]
    fn test_drift_minus_100ms_resync() {
        let base = fixed_instant();
        let offset = TimeOffset::negative(Duration::from_millis(100));
        let adjusted = offset.apply(base);
        assert_eq!(adjusted, base - ChronoDuration::milliseconds(100));
        // Symmetric to the positive case: exactly -100ms.
        assert_eq!((adjusted - base).num_milliseconds(), -100);
        assert_eq!(offset.as_signed_seconds(), 0); // sub-second magnitude
    }
}
