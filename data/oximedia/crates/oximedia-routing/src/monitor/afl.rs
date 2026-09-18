//! After-Fader Listen (AFL) monitoring.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// AFL (After-Fader Listen) monitor system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AflMonitor {
    /// Channels currently in AFL mode
    pub afl_channels: HashSet<usize>,
    /// AFL bus gain in dB
    pub afl_gain_db: f32,
    /// Mute non-AFL channels
    pub afl_exclusive: bool,
    /// AFL output level
    pub output_level_db: f32,
}

impl Default for AflMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl AflMonitor {
    /// Create a new AFL monitor
    #[must_use]
    pub fn new() -> Self {
        Self {
            afl_channels: HashSet::new(),
            afl_gain_db: 0.0,
            afl_exclusive: true,
            output_level_db: 0.0,
        }
    }

    /// Enable AFL for a channel
    pub fn enable_afl(&mut self, channel: usize) {
        self.afl_channels.insert(channel);
    }

    /// Disable AFL for a channel
    pub fn disable_afl(&mut self, channel: usize) {
        self.afl_channels.remove(&channel);
    }

    /// Toggle AFL for a channel
    pub fn toggle_afl(&mut self, channel: usize) {
        if self.afl_channels.contains(&channel) {
            self.afl_channels.remove(&channel);
        } else {
            self.afl_channels.insert(channel);
        }
    }

    /// Clear all AFL assignments
    pub fn clear_all(&mut self) {
        self.afl_channels.clear();
    }

    /// Check if a channel is in AFL
    #[must_use]
    pub fn is_afl(&self, channel: usize) -> bool {
        self.afl_channels.contains(&channel)
    }

    /// Get number of AFL channels
    #[must_use]
    pub fn afl_count(&self) -> usize {
        self.afl_channels.len()
    }

    /// Check if any channel is in AFL
    #[must_use]
    pub fn has_afl(&self) -> bool {
        !self.afl_channels.is_empty()
    }

    /// Set AFL gain
    pub fn set_gain(&mut self, gain_db: f32) {
        self.afl_gain_db = gain_db.clamp(-60.0, 12.0);
    }

    /// Set output level
    pub fn set_output_level(&mut self, level_db: f32) {
        self.output_level_db = level_db.clamp(-60.0, 12.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_afl_creation() {
        let afl = AflMonitor::new();
        assert_eq!(afl.afl_count(), 0);
        assert!(!afl.has_afl());
    }

    #[test]
    fn test_enable_afl() {
        let mut afl = AflMonitor::new();
        afl.enable_afl(0);
        assert!(afl.is_afl(0));
        assert_eq!(afl.afl_count(), 1);
    }

    #[test]
    fn test_disable_afl() {
        let mut afl = AflMonitor::new();
        afl.enable_afl(0);
        afl.disable_afl(0);
        assert!(!afl.is_afl(0));
        assert_eq!(afl.afl_count(), 0);
    }

    #[test]
    fn test_toggle_afl() {
        let mut afl = AflMonitor::new();
        afl.toggle_afl(0);
        assert!(afl.is_afl(0));

        afl.toggle_afl(0);
        assert!(!afl.is_afl(0));
    }

    #[test]
    fn test_clear_all() {
        let mut afl = AflMonitor::new();
        afl.enable_afl(0);
        afl.enable_afl(1);
        afl.enable_afl(2);

        afl.clear_all();
        assert_eq!(afl.afl_count(), 0);
    }

    #[test]
    fn test_multiple_channels() {
        let mut afl = AflMonitor::new();
        afl.enable_afl(0);
        afl.enable_afl(5);
        afl.enable_afl(10);

        assert_eq!(afl.afl_count(), 3);
        assert!(afl.is_afl(0));
        assert!(afl.is_afl(5));
        assert!(afl.is_afl(10));
    }
}
