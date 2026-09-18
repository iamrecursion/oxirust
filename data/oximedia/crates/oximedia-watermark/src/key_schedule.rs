#![allow(dead_code)]
//! Cryptographic key scheduling and rotation for watermark keys.
//!
//! This module provides deterministic key derivation, periodic key rotation,
//! and key-chain management so that each watermark segment can be embedded
//! with a unique derived key while allowing the detector to reconstruct the
//! same sequence from a single root secret.

use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Key derivation (pure Rust, no external crypto crate)
// ---------------------------------------------------------------------------

/// Derive a child key from a parent key and an index using a simple
/// xorshift-based KDF.  This is *not* cryptographically secure but is
/// sufficient for watermark PRNG seeding.
fn derive_key(parent: u64, index: u64) -> u64 {
    let mut h = parent ^ index;
    h ^= h << 13;
    h ^= h >> 7;
    h ^= h << 17;
    // mix further with a second round
    h = h.wrapping_mul(0x517c_c1b7_2722_0a95);
    h ^= h >> 32;
    h
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A single entry in the key schedule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduleEntry {
    /// Sequence number (monotonically increasing).
    pub sequence: u64,
    /// Derived key for this slot.
    pub key: u64,
    /// Epoch second at which this key becomes active (0 if unused).
    pub active_from: u64,
    /// Epoch second at which this key expires (0 = never).
    pub expires_at: u64,
}

/// Policy that controls how often keys rotate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotationPolicy {
    /// Rotate after every N segments.
    EveryNSegments(u64),
    /// Rotate after N seconds of wall-clock time.
    EveryNSeconds(u64),
    /// Never rotate (single key for the entire session).
    Never,
}

/// Configuration for the key scheduler.
#[derive(Debug, Clone)]
pub struct KeyScheduleConfig {
    /// Root secret from which all keys are derived.
    pub root_key: u64,
    /// Rotation policy.
    pub policy: RotationPolicy,
    /// Maximum number of past keys to retain in the ring buffer.
    pub history_size: usize,
}

impl Default for KeyScheduleConfig {
    fn default() -> Self {
        Self {
            root_key: 0,
            policy: RotationPolicy::EveryNSegments(100),
            history_size: 16,
        }
    }
}

// ---------------------------------------------------------------------------
// Key Scheduler
// ---------------------------------------------------------------------------

/// Manages the lifecycle of watermark embedding keys.
#[derive(Debug)]
pub struct KeyScheduler {
    config: KeyScheduleConfig,
    current_seq: u64,
    current_key: u64,
    history: VecDeque<ScheduleEntry>,
    segment_counter: u64,
    last_rotation_time: u64,
}

impl KeyScheduler {
    /// Create a new scheduler from the given configuration.
    #[must_use]
    pub fn new(config: KeyScheduleConfig) -> Self {
        let initial_key = derive_key(config.root_key, 0);
        Self {
            current_seq: 0,
            current_key: initial_key,
            history: VecDeque::with_capacity(config.history_size),
            segment_counter: 0,
            last_rotation_time: 0,
            config,
        }
    }

    /// Return the currently active key.
    #[must_use]
    pub fn current_key(&self) -> u64 {
        self.current_key
    }

    /// Return the current sequence number.
    #[must_use]
    pub fn current_sequence(&self) -> u64 {
        self.current_seq
    }

    /// Return a reference to the key history ring buffer.
    #[must_use]
    pub fn history(&self) -> &VecDeque<ScheduleEntry> {
        &self.history
    }

    /// Return a reference to the configuration.
    #[must_use]
    pub fn config(&self) -> &KeyScheduleConfig {
        &self.config
    }

    /// Notify the scheduler that a segment has been processed.
    /// If the rotation policy triggers, the key will be advanced.
    pub fn advance_segment(&mut self) {
        self.segment_counter += 1;
        if let RotationPolicy::EveryNSegments(n) = self.config.policy {
            if n > 0 && self.segment_counter >= n {
                self.rotate(0);
                self.segment_counter = 0;
            }
        }
    }

    /// Notify the scheduler that `now_epoch` seconds have elapsed.
    /// If the time-based policy triggers, the key will be advanced.
    pub fn advance_time(&mut self, now_epoch: u64) {
        if let RotationPolicy::EveryNSeconds(interval) = self.config.policy {
            if interval > 0 && now_epoch >= self.last_rotation_time + interval {
                self.rotate(now_epoch);
                self.last_rotation_time = now_epoch;
            }
        }
    }

    /// Manually rotate to the next key.
    pub fn force_rotate(&mut self) {
        self.rotate(0);
    }

    /// Look up a key by its sequence number from the history.
    #[must_use]
    pub fn key_for_sequence(&self, seq: u64) -> Option<u64> {
        if seq == self.current_seq {
            return Some(self.current_key);
        }
        self.history
            .iter()
            .find(|e| e.sequence == seq)
            .map(|e| e.key)
    }

    /// Derive a key for an arbitrary sequence number without modifying
    /// the scheduler state. Useful for detector-side reconstruction.
    #[must_use]
    pub fn derive_for_sequence(&self, seq: u64) -> u64 {
        derive_key(self.config.root_key, seq)
    }

    // ---- internal ----------------------------------------------------------

    fn rotate(&mut self, now_epoch: u64) {
        // Push current into history
        let entry = ScheduleEntry {
            sequence: self.current_seq,
            key: self.current_key,
            active_from: self.last_rotation_time,
            expires_at: now_epoch,
        };
        if self.history.len() >= self.config.history_size {
            self.history.pop_front();
        }
        self.history.push_back(entry);

        // Advance
        self.current_seq += 1;
        self.current_key = derive_key(self.config.root_key, self.current_seq);
    }
}

// ---------------------------------------------------------------------------
// Utility: bulk key generation
// ---------------------------------------------------------------------------

/// Pre-generate a vector of derived keys for sequence numbers `0..count`.
#[must_use]
pub fn generate_key_table(root: u64, count: usize) -> Vec<u64> {
    #[allow(clippy::cast_precision_loss)]
    (0..count).map(|i| derive_key(root, i as u64)).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derive_key_deterministic() {
        let a = derive_key(42, 0);
        let b = derive_key(42, 0);
        assert_eq!(a, b);
    }

    #[test]
    fn test_derive_key_different_indices() {
        let a = derive_key(42, 0);
        let b = derive_key(42, 1);
        assert_ne!(a, b);
    }

    #[test]
    fn test_derive_key_different_parents() {
        let a = derive_key(1, 0);
        let b = derive_key(2, 0);
        assert_ne!(a, b);
    }

    #[test]
    fn test_scheduler_initial_state() {
        let ks = KeyScheduler::new(KeyScheduleConfig::default());
        assert_eq!(ks.current_sequence(), 0);
        assert!(ks.history().is_empty());
    }

    #[test]
    fn test_scheduler_rotation_by_segments() {
        let cfg = KeyScheduleConfig {
            root_key: 99,
            policy: RotationPolicy::EveryNSegments(3),
            history_size: 8,
        };
        let mut ks = KeyScheduler::new(cfg);
        let k0 = ks.current_key();
        // Advance 3 segments => rotation
        ks.advance_segment();
        ks.advance_segment();
        ks.advance_segment();
        assert_eq!(ks.current_sequence(), 1);
        assert_ne!(ks.current_key(), k0);
        assert_eq!(ks.history().len(), 1);
    }

    #[test]
    fn test_scheduler_rotation_by_time() {
        let cfg = KeyScheduleConfig {
            root_key: 7,
            policy: RotationPolicy::EveryNSeconds(60),
            history_size: 4,
        };
        let mut ks = KeyScheduler::new(cfg);
        let k0 = ks.current_key();
        ks.advance_time(30); // not enough
        assert_eq!(ks.current_key(), k0);
        ks.advance_time(60); // triggers
        assert_ne!(ks.current_key(), k0);
        assert_eq!(ks.current_sequence(), 1);
    }

    #[test]
    fn test_scheduler_never_rotates() {
        let cfg = KeyScheduleConfig {
            root_key: 1,
            policy: RotationPolicy::Never,
            history_size: 4,
        };
        let mut ks = KeyScheduler::new(cfg);
        let k = ks.current_key();
        for _ in 0..1000 {
            ks.advance_segment();
        }
        assert_eq!(ks.current_key(), k);
        assert!(ks.history().is_empty());
    }

    #[test]
    fn test_force_rotate() {
        let mut ks = KeyScheduler::new(KeyScheduleConfig {
            root_key: 5,
            policy: RotationPolicy::Never,
            history_size: 4,
        });
        let k0 = ks.current_key();
        ks.force_rotate();
        assert_ne!(ks.current_key(), k0);
        assert_eq!(ks.current_sequence(), 1);
    }

    #[test]
    fn test_history_bounded() {
        let cfg = KeyScheduleConfig {
            root_key: 10,
            policy: RotationPolicy::EveryNSegments(1),
            history_size: 3,
        };
        let mut ks = KeyScheduler::new(cfg);
        for _ in 0..10 {
            ks.advance_segment();
        }
        assert!(ks.history().len() <= 3);
    }

    #[test]
    fn test_key_for_sequence_current() {
        let ks = KeyScheduler::new(KeyScheduleConfig::default());
        assert!(ks.key_for_sequence(0).is_some());
    }

    #[test]
    fn test_key_for_sequence_historical() {
        let cfg = KeyScheduleConfig {
            root_key: 20,
            policy: RotationPolicy::EveryNSegments(1),
            history_size: 8,
        };
        let mut ks = KeyScheduler::new(cfg);
        let k0 = ks.current_key();
        ks.advance_segment(); // seq 0 -> history, current = seq 1
        assert_eq!(ks.key_for_sequence(0), Some(k0));
    }

    #[test]
    fn test_derive_for_sequence_matches() {
        let ks = KeyScheduler::new(KeyScheduleConfig {
            root_key: 50,
            ..KeyScheduleConfig::default()
        });
        // Seq 0 derived must equal current_key (since current is seq 0)
        assert_eq!(ks.derive_for_sequence(0), ks.current_key());
    }

    #[test]
    fn test_generate_key_table_length() {
        let table = generate_key_table(42, 10);
        assert_eq!(table.len(), 10);
    }

    #[test]
    fn test_generate_key_table_unique() {
        let table = generate_key_table(42, 100);
        let mut set = std::collections::HashSet::new();
        for k in &table {
            set.insert(*k);
        }
        // extremely unlikely to have collisions for 100 entries
        assert!(set.len() >= 95);
    }
}
