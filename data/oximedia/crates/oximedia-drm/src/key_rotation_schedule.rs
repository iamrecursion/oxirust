//! Key rotation scheduling: rotation intervals, key versioning, graceful key transition.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::too_many_arguments)]

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// A versioned content key.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionedKey {
    /// Key version number (monotonically increasing).
    pub version: u32,
    /// Raw key bytes.
    pub key_bytes: Vec<u8>,
    /// Key ID.
    pub key_id: Vec<u8>,
    /// Unix timestamp when this key becomes active.
    pub active_from: i64,
    /// Unix timestamp when this key expires. `None` = indefinite.
    pub expires_at: Option<i64>,
    /// Whether this key is currently active.
    pub is_active: bool,
}

impl VersionedKey {
    /// Create a new versioned key.
    #[must_use]
    pub fn new(version: u32, key_bytes: Vec<u8>, key_id: Vec<u8>, active_from: i64) -> Self {
        Self {
            version,
            key_bytes,
            key_id,
            active_from,
            expires_at: None,
            is_active: false,
        }
    }

    /// Set an expiry time.
    #[must_use]
    pub fn with_expiry(mut self, expires_at: i64) -> Self {
        self.expires_at = Some(expires_at);
        self
    }

    /// Returns `true` if this key is valid at the given timestamp.
    #[must_use]
    pub fn is_valid_at(&self, ts: i64) -> bool {
        if ts < self.active_from {
            return false;
        }
        if let Some(exp) = self.expires_at {
            if ts > exp {
                return false;
            }
        }
        true
    }
}

/// Rotation interval specification.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum RotationInterval {
    /// Rotate every N seconds.
    EverySeconds(u64),
    /// Rotate every N segments.
    EveryNSegments(u32),
    /// Manual rotation only.
    Manual,
}

impl RotationInterval {
    /// Compute the next rotation time given the last rotation timestamp.
    #[must_use]
    pub fn next_rotation_time(&self, last_rotation: i64) -> Option<i64> {
        match self {
            Self::EverySeconds(secs) => Some(last_rotation + *secs as i64),
            Self::EveryNSegments(_) | Self::Manual => None,
        }
    }
}

/// Graceful transition configuration: how long the old key remains valid after rotation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TransitionWindow {
    /// Seconds the old key remains valid after the new key becomes active.
    pub overlap_secs: u64,
}

impl TransitionWindow {
    /// No transition: old key expires immediately.
    #[must_use]
    pub fn none() -> Self {
        Self { overlap_secs: 0 }
    }

    /// Transition window of the specified number of seconds.
    #[must_use]
    pub fn of_secs(secs: u64) -> Self {
        Self { overlap_secs: secs }
    }
}

/// A key rotation schedule managing a history of versioned keys.
#[derive(Debug)]
pub struct KeyRotationSchedule {
    /// Content identifier.
    pub content_id: String,
    /// Rotation interval.
    pub interval: RotationInterval,
    /// Transition window for graceful handover.
    pub transition: TransitionWindow,
    /// Key history (oldest first).
    keys: VecDeque<VersionedKey>,
    /// Maximum number of historical keys to retain.
    max_history: usize,
    /// Last rotation Unix timestamp.
    last_rotation: Option<i64>,
    /// Next version number to assign.
    next_version: u32,
}

impl KeyRotationSchedule {
    /// Create a new rotation schedule.
    #[must_use]
    pub fn new(
        content_id: impl Into<String>,
        interval: RotationInterval,
        transition: TransitionWindow,
    ) -> Self {
        Self {
            content_id: content_id.into(),
            interval,
            transition,
            keys: VecDeque::new(),
            max_history: 10,
            last_rotation: None,
            next_version: 1,
        }
    }

    /// Set maximum history size.
    #[must_use]
    pub fn with_max_history(mut self, n: usize) -> Self {
        self.max_history = n;
        self
    }

    /// Add a new key at the given activation timestamp.
    pub fn add_key(&mut self, key_bytes: Vec<u8>, key_id: Vec<u8>, now: i64) {
        // Expire the previous active key with transition overlap.
        if let Some(prev) = self.keys.iter_mut().filter(|k| k.is_active).last() {
            let expiry = now + self.transition.overlap_secs as i64;
            prev.expires_at = Some(expiry);
            prev.is_active = false;
        }

        let version = self.next_version;
        self.next_version += 1;

        let mut key = VersionedKey::new(version, key_bytes, key_id, now);
        key.is_active = true;
        self.keys.push_back(key);
        self.last_rotation = Some(now);

        // Prune history.
        while self.keys.len() > self.max_history {
            self.keys.pop_front();
        }
    }

    /// Get the currently active key.
    #[must_use]
    pub fn active_key(&self) -> Option<&VersionedKey> {
        self.keys.iter().find(|k| k.is_active)
    }

    /// Get all keys valid at the given timestamp (for decryption).
    #[must_use]
    pub fn keys_valid_at(&self, ts: i64) -> Vec<&VersionedKey> {
        self.keys.iter().filter(|k| k.is_valid_at(ts)).collect()
    }

    /// Check whether a rotation is due at the given timestamp.
    #[must_use]
    pub fn rotation_due(&self, now: i64) -> bool {
        match &self.interval {
            RotationInterval::EverySeconds(secs) => match self.last_rotation {
                None => true,
                Some(last) => now >= last + *secs as i64,
            },
            RotationInterval::EveryNSegments(_) | RotationInterval::Manual => false,
        }
    }

    /// Total number of keys in history.
    #[must_use]
    pub fn key_count(&self) -> usize {
        self.keys.len()
    }

    /// The version number of the currently active key.
    #[must_use]
    pub fn active_version(&self) -> Option<u32> {
        self.active_key().map(|k| k.version)
    }
}

/// Summary of a rotation event.
#[derive(Debug, Clone)]
pub struct RotationEvent {
    /// Version of the newly activated key.
    pub new_version: u32,
    /// Version of the key that was retired (if any).
    pub retired_version: Option<u32>,
    /// Unix timestamp of the rotation.
    pub timestamp: i64,
}

impl RotationEvent {
    /// Build a rotation event.
    #[must_use]
    pub fn new(new_version: u32, retired_version: Option<u32>, timestamp: i64) -> Self {
        Self {
            new_version,
            retired_version,
            timestamp,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_key(tag: u8) -> (Vec<u8>, Vec<u8>) {
        (vec![tag; 16], vec![tag; 16])
    }

    #[test]
    fn test_versioned_key_valid_at() {
        let k = VersionedKey::new(1, vec![0u8; 16], vec![0u8; 16], 1000).with_expiry(2000);
        assert!(k.is_valid_at(1000));
        assert!(k.is_valid_at(1500));
        assert!(!k.is_valid_at(999));
        assert!(!k.is_valid_at(2001));
    }

    #[test]
    fn test_versioned_key_no_expiry() {
        let k = VersionedKey::new(1, vec![0u8; 16], vec![0u8; 16], 100);
        assert!(k.is_valid_at(100));
        assert!(k.is_valid_at(i64::MAX));
    }

    #[test]
    fn test_rotation_interval_every_seconds() {
        let interval = RotationInterval::EverySeconds(3600);
        assert_eq!(interval.next_rotation_time(1000), Some(4600));
    }

    #[test]
    fn test_rotation_interval_manual_no_next() {
        let interval = RotationInterval::Manual;
        assert!(interval.next_rotation_time(1000).is_none());
    }

    #[test]
    fn test_transition_window_none() {
        let tw = TransitionWindow::none();
        assert_eq!(tw.overlap_secs, 0);
    }

    #[test]
    fn test_schedule_add_first_key() {
        let mut sched = KeyRotationSchedule::new(
            "c001",
            RotationInterval::EverySeconds(3600),
            TransitionWindow::of_secs(300),
        );
        let (kb, kid) = dummy_key(1);
        sched.add_key(kb, kid, 1000);
        assert_eq!(sched.key_count(), 1);
        assert_eq!(sched.active_version(), Some(1));
    }

    #[test]
    fn test_schedule_rotate_key() {
        let mut sched = KeyRotationSchedule::new(
            "c002",
            RotationInterval::EverySeconds(3600),
            TransitionWindow::of_secs(60),
        );
        let (kb1, kid1) = dummy_key(1);
        sched.add_key(kb1, kid1, 1000);
        let (kb2, kid2) = dummy_key(2);
        sched.add_key(kb2, kid2, 5000);
        assert_eq!(sched.active_version(), Some(2));
        // Old key should be valid during overlap window.
        let valid = sched.keys_valid_at(5030);
        assert_eq!(valid.len(), 2); // both keys valid
    }

    #[test]
    fn test_schedule_rotation_due() {
        let mut sched = KeyRotationSchedule::new(
            "c003",
            RotationInterval::EverySeconds(3600),
            TransitionWindow::none(),
        );
        assert!(sched.rotation_due(1000)); // no key yet
        let (kb, kid) = dummy_key(1);
        sched.add_key(kb, kid, 1000);
        assert!(!sched.rotation_due(2000)); // not yet 3600s
        assert!(sched.rotation_due(4600)); // >= 1000+3600
    }

    #[test]
    fn test_schedule_manual_not_due() {
        let sched =
            KeyRotationSchedule::new("c004", RotationInterval::Manual, TransitionWindow::none());
        assert!(!sched.rotation_due(999_999));
    }

    #[test]
    fn test_schedule_max_history() {
        let mut sched =
            KeyRotationSchedule::new("c005", RotationInterval::Manual, TransitionWindow::none())
                .with_max_history(3);
        for i in 1u8..=5 {
            let (kb, kid) = dummy_key(i);
            sched.add_key(kb, kid, i as i64 * 1000);
        }
        assert_eq!(sched.key_count(), 3);
    }

    #[test]
    fn test_rotation_event_new() {
        let ev = RotationEvent::new(3, Some(2), 5000);
        assert_eq!(ev.new_version, 3);
        assert_eq!(ev.retired_version, Some(2));
        assert_eq!(ev.timestamp, 5000);
    }

    #[test]
    fn test_keys_valid_at_after_expiry() {
        let mut sched =
            KeyRotationSchedule::new("c006", RotationInterval::Manual, TransitionWindow::none());
        let (kb, kid) = dummy_key(1);
        sched.add_key(kb, kid, 1000);
        // Add second key to expire first immediately (overlap=0).
        let (kb2, kid2) = dummy_key(2);
        sched.add_key(kb2, kid2, 2000);
        // At ts=2001 only the second key should be valid.
        let valid = sched.keys_valid_at(2001);
        assert_eq!(valid.len(), 1);
        assert_eq!(valid[0].version, 2);
    }

    #[test]
    fn test_active_key_none_when_empty() {
        let sched =
            KeyRotationSchedule::new("c007", RotationInterval::Manual, TransitionWindow::none());
        assert!(sched.active_key().is_none());
    }
}
