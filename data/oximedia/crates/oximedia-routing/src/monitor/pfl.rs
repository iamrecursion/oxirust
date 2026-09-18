//! Pre-Fader Listen (PFL) monitoring.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// PFL (Pre-Fader Listen) monitor system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PflMonitor {
    /// Channels currently in PFL mode
    pub pfl_channels: HashSet<usize>,
    /// PFL bus gain in dB
    pub pfl_gain_db: f32,
    /// Dim main output when PFL active
    pub dim_main: bool,
    /// Dim level in dB
    pub dim_level_db: f32,
    /// PFL output level
    pub output_level_db: f32,
}

impl Default for PflMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl PflMonitor {
    /// Create a new PFL monitor
    #[must_use]
    pub fn new() -> Self {
        Self {
            pfl_channels: HashSet::new(),
            pfl_gain_db: 0.0,
            dim_main: true,
            dim_level_db: -20.0,
            output_level_db: 0.0,
        }
    }

    /// Enable PFL for a channel
    pub fn enable_pfl(&mut self, channel: usize) {
        self.pfl_channels.insert(channel);
    }

    /// Disable PFL for a channel
    pub fn disable_pfl(&mut self, channel: usize) {
        self.pfl_channels.remove(&channel);
    }

    /// Toggle PFL for a channel
    pub fn toggle_pfl(&mut self, channel: usize) {
        if self.pfl_channels.contains(&channel) {
            self.pfl_channels.remove(&channel);
        } else {
            self.pfl_channels.insert(channel);
        }
    }

    /// Clear all PFL assignments
    pub fn clear_all(&mut self) {
        self.pfl_channels.clear();
    }

    /// Check if a channel is in PFL
    #[must_use]
    pub fn is_pfl(&self, channel: usize) -> bool {
        self.pfl_channels.contains(&channel)
    }

    /// Get number of PFL channels
    #[must_use]
    pub fn pfl_count(&self) -> usize {
        self.pfl_channels.len()
    }

    /// Check if any channel is in PFL
    #[must_use]
    pub fn has_pfl(&self) -> bool {
        !self.pfl_channels.is_empty()
    }

    /// Set PFL gain
    pub fn set_gain(&mut self, gain_db: f32) {
        self.pfl_gain_db = gain_db.clamp(-60.0, 12.0);
    }

    /// Set dim level
    pub fn set_dim_level(&mut self, dim_db: f32) {
        self.dim_level_db = dim_db.clamp(-60.0, 0.0);
    }

    /// Set output level
    pub fn set_output_level(&mut self, level_db: f32) {
        self.output_level_db = level_db.clamp(-60.0, 12.0);
    }

    /// Get effective main output level (considering dim)
    #[must_use]
    pub fn effective_main_level_db(&self) -> f32 {
        if self.dim_main && self.has_pfl() {
            self.dim_level_db
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pfl_creation() {
        let pfl = PflMonitor::new();
        assert_eq!(pfl.pfl_count(), 0);
        assert!(!pfl.has_pfl());
    }

    #[test]
    fn test_enable_pfl() {
        let mut pfl = PflMonitor::new();
        pfl.enable_pfl(0);
        assert!(pfl.is_pfl(0));
        assert_eq!(pfl.pfl_count(), 1);
    }

    #[test]
    fn test_disable_pfl() {
        let mut pfl = PflMonitor::new();
        pfl.enable_pfl(0);
        pfl.disable_pfl(0);
        assert!(!pfl.is_pfl(0));
        assert_eq!(pfl.pfl_count(), 0);
    }

    #[test]
    fn test_toggle_pfl() {
        let mut pfl = PflMonitor::new();
        pfl.toggle_pfl(0);
        assert!(pfl.is_pfl(0));

        pfl.toggle_pfl(0);
        assert!(!pfl.is_pfl(0));
    }

    #[test]
    fn test_dim_main() {
        let mut pfl = PflMonitor::new();
        assert!(pfl.effective_main_level_db().abs() < f32::EPSILON);

        pfl.enable_pfl(0);
        assert!((pfl.effective_main_level_db() - (-20.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_clear_all() {
        let mut pfl = PflMonitor::new();
        pfl.enable_pfl(0);
        pfl.enable_pfl(1);

        pfl.clear_all();
        assert_eq!(pfl.pfl_count(), 0);
    }
}
