//! Synchronization features for media playback.

pub mod audio;
pub mod genlock;
pub mod video;

/// Sync mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncMode {
    /// Freerun (no sync)
    Freerun,
    /// Genlock to external reference
    Genlock,
    /// Jam sync from timecode
    JamSync,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sync_mode() {
        let mode = SyncMode::Genlock;
        assert_eq!(mode, SyncMode::Genlock);
    }
}
