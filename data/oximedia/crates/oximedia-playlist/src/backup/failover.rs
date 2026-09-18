//! Backup playlist failover management.

use crate::{Playlist, PlaylistError, Result};
use std::sync::{Arc, RwLock};
use std::time::Duration;

/// Strategy for failover behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailoverStrategy {
    /// Switch to backup immediately on failure.
    Immediate,

    /// Retry the failed item before switching to backup.
    RetryThenBackup {
        /// Number of retry attempts.
        attempts: u32,
        /// Delay between retries.
        retry_delay: Duration,
    },

    /// Continue with next item in playlist, use backup only if all fail.
    SkipToNext,

    /// Hold on last good frame, then switch to backup.
    HoldThenBackup {
        /// Duration to hold frame.
        hold_duration: Duration,
    },
}

impl Default for FailoverStrategy {
    fn default() -> Self {
        Self::RetryThenBackup {
            attempts: 3,
            retry_delay: Duration::from_secs(1),
        }
    }
}

/// Failover event.
#[derive(Debug, Clone)]
pub enum FailoverEvent {
    /// Primary playlist failed.
    PrimaryFailed {
        /// Playlist ID.
        playlist_id: String,
        /// Error message.
        error: String,
    },

    /// Switched to backup playlist.
    SwitchedToBackup {
        /// Backup playlist ID.
        backup_id: String,
    },

    /// Retrying failed item.
    Retrying {
        /// Item index.
        item_index: usize,
        /// Retry attempt number.
        attempt: u32,
    },

    /// Restored to primary playlist.
    RestoredToPrimary {
        /// Primary playlist ID.
        primary_id: String,
    },
}

/// Failover manager for handling playlist failures.
pub struct FailoverManager {
    primary_playlist: Arc<RwLock<Option<Playlist>>>,
    backup_playlist: Arc<RwLock<Option<Playlist>>>,
    strategy: FailoverStrategy,
    is_using_backup: Arc<RwLock<bool>>,
    failure_count: Arc<RwLock<u32>>,
}

impl FailoverManager {
    /// Creates a new failover manager.
    #[must_use]
    pub fn new(strategy: FailoverStrategy) -> Self {
        Self {
            primary_playlist: Arc::new(RwLock::new(None)),
            backup_playlist: Arc::new(RwLock::new(None)),
            strategy,
            is_using_backup: Arc::new(RwLock::new(false)),
            failure_count: Arc::new(RwLock::new(0)),
        }
    }

    /// Sets the primary playlist.
    pub fn set_primary(&self, playlist: Playlist) -> Result<()> {
        let mut primary = self
            .primary_playlist
            .write()
            .map_err(|e| PlaylistError::FailoverError(format!("Lock error: {e}")))?;

        *primary = Some(playlist);
        Ok(())
    }

    /// Sets the backup playlist.
    pub fn set_backup(&self, playlist: Playlist) -> Result<()> {
        let mut backup = self
            .backup_playlist
            .write()
            .map_err(|e| PlaylistError::FailoverError(format!("Lock error: {e}")))?;

        *backup = Some(playlist);
        Ok(())
    }

    /// Triggers failover to backup playlist.
    pub fn trigger_failover(&self) -> Result<()> {
        let backup = self
            .backup_playlist
            .read()
            .map_err(|e| PlaylistError::FailoverError(format!("Lock error: {e}")))?;

        if backup.is_none() {
            return Err(PlaylistError::FailoverError(
                "No backup playlist configured".to_string(),
            ));
        }

        let mut using_backup = self
            .is_using_backup
            .write()
            .map_err(|e| PlaylistError::FailoverError(format!("Lock error: {e}")))?;

        *using_backup = true;

        // Increment failure count
        if let Ok(mut count) = self.failure_count.write() {
            *count += 1;
        }

        Ok(())
    }

    /// Restores to primary playlist.
    pub fn restore_to_primary(&self) -> Result<()> {
        let mut using_backup = self
            .is_using_backup
            .write()
            .map_err(|e| PlaylistError::FailoverError(format!("Lock error: {e}")))?;

        *using_backup = false;

        // Reset failure count
        if let Ok(mut count) = self.failure_count.write() {
            *count = 0;
        }

        Ok(())
    }

    /// Checks if currently using backup playlist.
    pub fn is_using_backup(&self) -> Result<bool> {
        let using_backup = self
            .is_using_backup
            .read()
            .map_err(|e| PlaylistError::FailoverError(format!("Lock error: {e}")))?;

        Ok(*using_backup)
    }

    /// Gets the current active playlist.
    pub fn get_active_playlist(&self) -> Result<Option<Playlist>> {
        let using_backup = self.is_using_backup()?;

        let playlist = if using_backup {
            self.backup_playlist.read()
        } else {
            self.primary_playlist.read()
        }
        .map_err(|e| PlaylistError::FailoverError(format!("Lock error: {e}")))?;

        Ok(playlist.clone())
    }

    /// Gets the failover strategy.
    #[must_use]
    pub const fn strategy(&self) -> FailoverStrategy {
        self.strategy
    }

    /// Gets the failure count.
    pub fn failure_count(&self) -> Result<u32> {
        let count = self
            .failure_count
            .read()
            .map_err(|e| PlaylistError::FailoverError(format!("Lock error: {e}")))?;

        Ok(*count)
    }

    /// Resets the failure count.
    pub fn reset_failure_count(&self) -> Result<()> {
        let mut count = self
            .failure_count
            .write()
            .map_err(|e| PlaylistError::FailoverError(format!("Lock error: {e}")))?;

        *count = 0;
        Ok(())
    }
}

impl Default for FailoverManager {
    fn default() -> Self {
        Self::new(FailoverStrategy::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::playlist::PlaylistType;

    #[test]
    fn test_failover_manager() {
        let manager = FailoverManager::new(FailoverStrategy::Immediate);
        let primary = Playlist::new("primary", PlaylistType::Linear);
        let backup = Playlist::new("backup", PlaylistType::Linear);

        manager
            .set_primary(primary)
            .expect("should succeed in test");
        manager.set_backup(backup).expect("should succeed in test");

        assert!(!manager.is_using_backup().expect("should succeed in test"));

        manager.trigger_failover().expect("should succeed in test");
        assert!(manager.is_using_backup().expect("should succeed in test"));

        manager
            .restore_to_primary()
            .expect("should succeed in test");
        assert!(!manager.is_using_backup().expect("should succeed in test"));
    }

    #[test]
    fn test_failure_count() {
        let manager = FailoverManager::default();
        let backup = Playlist::new("backup", PlaylistType::Linear);

        manager.set_backup(backup).expect("should succeed in test");

        assert_eq!(manager.failure_count().expect("should succeed in test"), 0);

        manager.trigger_failover().expect("should succeed in test");
        assert_eq!(manager.failure_count().expect("should succeed in test"), 1);

        manager.trigger_failover().expect("should succeed in test");
        assert_eq!(manager.failure_count().expect("should succeed in test"), 2);

        manager
            .reset_failure_count()
            .expect("should succeed in test");
        assert_eq!(manager.failure_count().expect("should succeed in test"), 0);
    }
}
