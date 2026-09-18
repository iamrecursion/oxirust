//! Synchronization subsystem for virtual production
//!
//! Provides genlock synchronization and frame timing for precise
//! multi-device synchronization.

pub mod genlock;
pub mod timing;

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Sync status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SyncStatus {
    /// Not synchronized
    Unlocked,
    /// Synchronizing
    Locking,
    /// Fully synchronized
    Locked,
}

/// Sync timestamp
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SyncTimestamp {
    /// Nanoseconds since epoch
    pub nanos: u64,
    /// Frame number
    pub frame: u64,
}

impl SyncTimestamp {
    /// Create new sync timestamp
    #[must_use]
    pub fn new(nanos: u64, frame: u64) -> Self {
        Self { nanos, frame }
    }

    /// Get duration since another timestamp
    #[must_use]
    pub fn duration_since(&self, other: &Self) -> Duration {
        Duration::from_nanos(self.nanos.saturating_sub(other.nanos))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sync_timestamp() {
        let ts1 = SyncTimestamp::new(1000, 0);
        let ts2 = SyncTimestamp::new(2000, 1);

        let duration = ts2.duration_since(&ts1);
        assert_eq!(duration.as_nanos(), 1000);
    }
}
