#![allow(dead_code)]
//! Solo bus management for mixer channels.
//!
//! Implements industry-standard solo modes:
//! - **PFL** (Pre-Fader Listen): monitor the channel signal before the fader
//! - **AFL** (After-Fader Listen): monitor after the fader and pan
//! - **SIP** (Solo-In-Place): mutes all non-soloed channels
//!
//! The solo bus tracks which channels are soloed and computes
//! the appropriate mute/gain state for every channel in the mix.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Solo mode for the mixer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SoloMode {
    /// Pre-Fader Listen: signal is tapped before the fader.
    Pfl,
    /// After-Fader Listen: signal is tapped after fader and pan.
    Afl,
    /// Solo-In-Place: non-soloed channels are muted.
    Sip,
}

/// Configuration for the solo bus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoloBusConfig {
    /// The active solo mode.
    pub mode: SoloMode,
    /// Gain applied to the solo bus output (0.0..=1.0).
    pub solo_gain: f32,
    /// Dim level applied to non-soloed channels in SIP mode (0.0 = full mute).
    pub sip_dim_level: f32,
    /// Whether exclusive solo is enabled (soloing one channel clears others).
    pub exclusive: bool,
    /// Whether solo-safe is respected.
    pub respect_safe: bool,
}

impl Default for SoloBusConfig {
    fn default() -> Self {
        Self {
            mode: SoloMode::Pfl,
            solo_gain: 1.0,
            sip_dim_level: 0.0,
            exclusive: false,
            respect_safe: true,
        }
    }
}

/// Solo bus manager.
///
/// Tracks which channels are soloed, which are solo-safe,
/// and computes the appropriate gain for each channel based on the solo mode.
#[derive(Debug, Clone)]
pub struct SoloBus {
    config: SoloBusConfig,
    soloed_channels: HashSet<u32>,
    safe_channels: HashSet<u32>,
    total_channels: u32,
}

impl SoloBus {
    /// Creates a new solo bus with the given configuration and total channel count.
    #[must_use]
    pub fn new(config: SoloBusConfig, total_channels: u32) -> Self {
        Self {
            config,
            soloed_channels: HashSet::new(),
            safe_channels: HashSet::new(),
            total_channels,
        }
    }

    /// Returns the current solo mode.
    #[must_use]
    pub fn mode(&self) -> SoloMode {
        self.config.mode
    }

    /// Sets the solo mode.
    pub fn set_mode(&mut self, mode: SoloMode) {
        self.config.mode = mode;
    }

    /// Returns whether any channel is currently soloed.
    #[must_use]
    pub fn any_soloed(&self) -> bool {
        !self.soloed_channels.is_empty()
    }

    /// Returns the number of currently soloed channels.
    #[must_use]
    pub fn soloed_count(&self) -> usize {
        self.soloed_channels.len()
    }

    /// Returns whether a specific channel is soloed.
    #[must_use]
    pub fn is_soloed(&self, channel: u32) -> bool {
        self.soloed_channels.contains(&channel)
    }

    /// Returns whether a specific channel is solo-safe.
    #[must_use]
    pub fn is_safe(&self, channel: u32) -> bool {
        self.safe_channels.contains(&channel)
    }

    /// Solos a channel. In exclusive mode, clears all other solos first.
    pub fn solo(&mut self, channel: u32) {
        if self.config.exclusive {
            self.soloed_channels.clear();
        }
        self.soloed_channels.insert(channel);
    }

    /// Unsolos a channel.
    pub fn unsolo(&mut self, channel: u32) {
        self.soloed_channels.remove(&channel);
    }

    /// Toggles the solo state of a channel.
    pub fn toggle_solo(&mut self, channel: u32) {
        if self.is_soloed(channel) {
            self.unsolo(channel);
        } else {
            self.solo(channel);
        }
    }

    /// Clears all solos.
    pub fn clear_all_solos(&mut self) {
        self.soloed_channels.clear();
    }

    /// Marks a channel as solo-safe (it will not be muted in SIP mode).
    pub fn set_safe(&mut self, channel: u32, safe: bool) {
        if safe {
            self.safe_channels.insert(channel);
        } else {
            self.safe_channels.remove(&channel);
        }
    }

    /// Returns the gain multiplier for a channel based on the current solo state.
    ///
    /// - When no channel is soloed, all channels get 1.0.
    /// - In PFL/AFL mode, all channels get 1.0 (solo is a separate bus).
    /// - In SIP mode, soloed and safe channels get 1.0; others get `sip_dim_level`.
    #[must_use]
    pub fn channel_gain(&self, channel: u32) -> f32 {
        if !self.any_soloed() {
            return 1.0;
        }

        match self.config.mode {
            SoloMode::Pfl | SoloMode::Afl => {
                // PFL/AFL: solo is a separate listen bus; main mix is unaffected
                1.0
            }
            SoloMode::Sip => {
                if self.is_soloed(channel) {
                    1.0
                } else if self.config.respect_safe && self.is_safe(channel) {
                    1.0
                } else {
                    self.config.sip_dim_level.clamp(0.0, 1.0)
                }
            }
        }
    }

    /// Returns the solo bus output gain for a channel.
    ///
    /// In PFL/AFL modes, this is the gain at which the channel appears
    /// on the solo bus. In SIP mode, the solo bus is not used (returns 0.0).
    #[must_use]
    pub fn solo_bus_gain(&self, channel: u32) -> f32 {
        if !self.is_soloed(channel) {
            return 0.0;
        }
        match self.config.mode {
            SoloMode::Pfl | SoloMode::Afl => self.config.solo_gain.clamp(0.0, 1.0),
            SoloMode::Sip => 0.0, // SIP does not use a separate solo bus
        }
    }

    /// Returns all currently soloed channel indices.
    #[must_use]
    pub fn soloed_channels(&self) -> Vec<u32> {
        let mut v: Vec<u32> = self.soloed_channels.iter().copied().collect();
        v.sort_unstable();
        v
    }

    /// Returns all solo-safe channel indices.
    #[must_use]
    pub fn safe_channels(&self) -> Vec<u32> {
        let mut v: Vec<u32> = self.safe_channels.iter().copied().collect();
        v.sort_unstable();
        v
    }

    /// Returns the total channel count.
    #[must_use]
    pub fn total_channels(&self) -> u32 {
        self.total_channels
    }

    /// Sets the solo bus output gain.
    pub fn set_solo_gain(&mut self, gain: f32) {
        self.config.solo_gain = gain.clamp(0.0, 1.0);
    }

    /// Sets the SIP dim level.
    pub fn set_sip_dim_level(&mut self, level: f32) {
        self.config.sip_dim_level = level.clamp(0.0, 1.0);
    }

    /// Sets exclusive solo mode.
    pub fn set_exclusive(&mut self, exclusive: bool) {
        self.config.exclusive = exclusive;
    }
}

impl Default for SoloBus {
    fn default() -> Self {
        Self::new(SoloBusConfig::default(), 64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_state() {
        let bus = SoloBus::default();
        assert!(!bus.any_soloed());
        assert_eq!(bus.soloed_count(), 0);
        assert_eq!(bus.mode(), SoloMode::Pfl);
    }

    #[test]
    fn test_solo_unsolo() {
        let mut bus = SoloBus::default();
        bus.solo(3);
        assert!(bus.is_soloed(3));
        assert!(bus.any_soloed());
        bus.unsolo(3);
        assert!(!bus.is_soloed(3));
        assert!(!bus.any_soloed());
    }

    #[test]
    fn test_toggle_solo() {
        let mut bus = SoloBus::default();
        bus.toggle_solo(5);
        assert!(bus.is_soloed(5));
        bus.toggle_solo(5);
        assert!(!bus.is_soloed(5));
    }

    #[test]
    fn test_exclusive_solo() {
        let config = SoloBusConfig {
            exclusive: true,
            ..Default::default()
        };
        let mut bus = SoloBus::new(config, 32);
        bus.solo(1);
        bus.solo(2);
        assert!(!bus.is_soloed(1));
        assert!(bus.is_soloed(2));
        assert_eq!(bus.soloed_count(), 1);
    }

    #[test]
    fn test_clear_all() {
        let mut bus = SoloBus::default();
        bus.solo(0);
        bus.solo(1);
        bus.solo(2);
        bus.clear_all_solos();
        assert!(!bus.any_soloed());
    }

    #[test]
    fn test_pfl_channel_gain() {
        let mut bus = SoloBus::default();
        bus.set_mode(SoloMode::Pfl);
        bus.solo(0);
        // PFL: main mix unaffected
        assert!((bus.channel_gain(0) - 1.0).abs() < f32::EPSILON);
        assert!((bus.channel_gain(1) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_afl_channel_gain() {
        let mut bus = SoloBus::default();
        bus.set_mode(SoloMode::Afl);
        bus.solo(0);
        assert!((bus.channel_gain(0) - 1.0).abs() < f32::EPSILON);
        assert!((bus.channel_gain(1) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_sip_channel_gain() {
        let config = SoloBusConfig {
            mode: SoloMode::Sip,
            sip_dim_level: 0.0,
            ..Default::default()
        };
        let mut bus = SoloBus::new(config, 32);
        bus.solo(0);
        assert!((bus.channel_gain(0) - 1.0).abs() < f32::EPSILON);
        assert!((bus.channel_gain(1) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_sip_dim_level() {
        let config = SoloBusConfig {
            mode: SoloMode::Sip,
            sip_dim_level: 0.2,
            ..Default::default()
        };
        let mut bus = SoloBus::new(config, 32);
        bus.solo(0);
        assert!((bus.channel_gain(1) - 0.2).abs() < f32::EPSILON);
    }

    #[test]
    fn test_solo_safe() {
        let config = SoloBusConfig {
            mode: SoloMode::Sip,
            sip_dim_level: 0.0,
            respect_safe: true,
            ..Default::default()
        };
        let mut bus = SoloBus::new(config, 32);
        bus.set_safe(1, true);
        bus.solo(0);
        // Channel 1 is safe, should not be muted
        assert!((bus.channel_gain(1) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_solo_bus_gain_pfl() {
        let mut bus = SoloBus::default();
        bus.set_mode(SoloMode::Pfl);
        bus.solo(0);
        assert!((bus.solo_bus_gain(0) - 1.0).abs() < f32::EPSILON);
        assert!((bus.solo_bus_gain(1) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_solo_bus_gain_sip() {
        let config = SoloBusConfig {
            mode: SoloMode::Sip,
            ..Default::default()
        };
        let mut bus = SoloBus::new(config, 32);
        bus.solo(0);
        // SIP does not use a separate solo bus
        assert!((bus.solo_bus_gain(0) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_no_solo_channel_gain() {
        let bus = SoloBus::default();
        // No channels soloed — all gain = 1.0
        assert!((bus.channel_gain(0) - 1.0).abs() < f32::EPSILON);
        assert!((bus.channel_gain(99) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_soloed_channels_sorted() {
        let mut bus = SoloBus::default();
        bus.solo(5);
        bus.solo(1);
        bus.solo(3);
        let ch = bus.soloed_channels();
        assert_eq!(ch, vec![1, 3, 5]);
    }

    #[test]
    fn test_safe_channels_sorted() {
        let mut bus = SoloBus::default();
        bus.set_safe(10, true);
        bus.set_safe(2, true);
        let ch = bus.safe_channels();
        assert_eq!(ch, vec![2, 10]);
    }

    #[test]
    fn test_set_solo_gain() {
        let mut bus = SoloBus::default();
        bus.set_solo_gain(0.5);
        bus.solo(0);
        assert!((bus.solo_bus_gain(0) - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_set_safe_toggle() {
        let mut bus = SoloBus::default();
        bus.set_safe(0, true);
        assert!(bus.is_safe(0));
        bus.set_safe(0, false);
        assert!(!bus.is_safe(0));
    }
}
