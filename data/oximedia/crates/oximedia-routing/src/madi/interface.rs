//! MADI (Multi-channel Audio Digital Interface) routing support.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// MADI interface configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MadiInterface {
    /// Interface name
    pub name: String,
    /// Number of channels (typically 64)
    pub channel_count: u8,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Frame mode (48k or 96k)
    pub frame_mode: FrameMode,
    /// Channel routing map
    pub channel_map: HashMap<u8, MadiChannel>,
    /// Redundancy enabled
    pub redundancy: bool,
    /// Optical or coaxial
    pub connection_type: ConnectionType,
}

/// MADI frame mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FrameMode {
    /// 48k frame (64 channels)
    Frame48k,
    /// 96k frame (32 channels)
    Frame96k,
}

/// MADI connection type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConnectionType {
    /// Optical fiber
    Optical,
    /// Coaxial (BNC)
    Coaxial,
}

/// MADI channel configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MadiChannel {
    /// Channel index (0-63 or 0-31)
    pub channel: u8,
    /// Label for this channel
    pub label: String,
    /// Whether this channel is active
    pub active: bool,
    /// Optional routing destination
    pub destination: Option<usize>,
}

impl MadiInterface {
    /// Create a new MADI interface
    #[must_use]
    pub fn new(name: String) -> Self {
        Self {
            name,
            channel_count: 64,
            sample_rate: 48000,
            frame_mode: FrameMode::Frame48k,
            channel_map: HashMap::new(),
            redundancy: false,
            connection_type: ConnectionType::Optical,
        }
    }

    /// Create MADI interface for 96kHz operation
    #[must_use]
    pub fn new_96k(name: String) -> Self {
        let mut interface = Self::new(name);
        interface.frame_mode = FrameMode::Frame96k;
        interface.channel_count = 32;
        interface.sample_rate = 96000;
        interface
    }

    /// Add a channel
    pub fn add_channel(&mut self, channel: MadiChannel) -> Result<(), MadiError> {
        if channel.channel >= self.channel_count {
            return Err(MadiError::InvalidChannel(channel.channel));
        }

        self.channel_map.insert(channel.channel, channel);
        Ok(())
    }

    /// Remove a channel
    pub fn remove_channel(&mut self, channel: u8) {
        self.channel_map.remove(&channel);
    }

    /// Get channel
    #[must_use]
    pub fn get_channel(&self, channel: u8) -> Option<&MadiChannel> {
        self.channel_map.get(&channel)
    }

    /// Get mutable channel reference
    pub fn get_channel_mut(&mut self, channel: u8) -> Option<&mut MadiChannel> {
        self.channel_map.get_mut(&channel)
    }

    /// Get all active channels
    #[must_use]
    pub fn active_channels(&self) -> Vec<&MadiChannel> {
        self.channel_map.values().filter(|ch| ch.active).collect()
    }

    /// Get maximum channel count for current frame mode
    #[must_use]
    pub const fn max_channels(&self) -> u8 {
        match self.frame_mode {
            FrameMode::Frame48k => 64,
            FrameMode::Frame96k => 32,
        }
    }

    /// Set frame mode (updates channel count)
    pub fn set_frame_mode(&mut self, mode: FrameMode) {
        self.frame_mode = mode;
        self.channel_count = match mode {
            FrameMode::Frame48k => 64,
            FrameMode::Frame96k => 32,
        };

        // Remove channels beyond new limit
        self.channel_map.retain(|&ch, _| ch < self.channel_count);
    }

    /// Initialize all channels with default labels
    pub fn initialize_channels(&mut self) {
        for i in 0..self.channel_count {
            let channel = MadiChannel {
                channel: i,
                label: format!("MADI {}", i + 1),
                active: true,
                destination: None,
            };
            self.channel_map.insert(i, channel);
        }
    }
}

/// Errors that can occur in MADI operations
#[derive(Debug, Clone, thiserror::Error)]
pub enum MadiError {
    /// Invalid channel number
    #[error("Invalid MADI channel: {0}")]
    InvalidChannel(u8),
    /// Frame mode mismatch
    #[error("Frame mode mismatch")]
    FrameModeMismatch,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_madi_interface_creation() {
        let interface = MadiInterface::new("MADI 1".to_string());
        assert_eq!(interface.channel_count, 64);
        assert_eq!(interface.sample_rate, 48000);
        assert_eq!(interface.frame_mode, FrameMode::Frame48k);
    }

    #[test]
    fn test_madi_96k() {
        let interface = MadiInterface::new_96k("MADI 96k".to_string());
        assert_eq!(interface.channel_count, 32);
        assert_eq!(interface.sample_rate, 96000);
        assert_eq!(interface.frame_mode, FrameMode::Frame96k);
    }

    #[test]
    fn test_add_channel() {
        let mut interface = MadiInterface::new("Test".to_string());

        let channel = MadiChannel {
            channel: 0,
            label: "Channel 1".to_string(),
            active: true,
            destination: None,
        };

        interface
            .add_channel(channel)
            .expect("should succeed in test");
        assert!(interface.get_channel(0).is_some());
    }

    #[test]
    fn test_invalid_channel() {
        let mut interface = MadiInterface::new("Test".to_string());

        let channel = MadiChannel {
            channel: 100, // Invalid
            label: "Invalid".to_string(),
            active: true,
            destination: None,
        };

        assert!(matches!(
            interface.add_channel(channel),
            Err(MadiError::InvalidChannel(100))
        ));
    }

    #[test]
    fn test_frame_mode_change() {
        let mut interface = MadiInterface::new("Test".to_string());
        assert_eq!(interface.max_channels(), 64);

        interface.set_frame_mode(FrameMode::Frame96k);
        assert_eq!(interface.max_channels(), 32);
        assert_eq!(interface.channel_count, 32);
    }

    #[test]
    fn test_initialize_channels() {
        let mut interface = MadiInterface::new("Test".to_string());
        interface.initialize_channels();

        assert_eq!(interface.channel_map.len(), 64);
        assert!(interface.get_channel(0).is_some());
        assert!(interface.get_channel(63).is_some());
    }

    #[test]
    fn test_active_channels() {
        let mut interface = MadiInterface::new("Test".to_string());

        let ch1 = MadiChannel {
            channel: 0,
            label: "Ch1".to_string(),
            active: true,
            destination: None,
        };

        let ch2 = MadiChannel {
            channel: 1,
            label: "Ch2".to_string(),
            active: false,
            destination: None,
        };

        interface.add_channel(ch1).expect("should succeed in test");
        interface.add_channel(ch2).expect("should succeed in test");

        let active = interface.active_channels();
        assert_eq!(active.len(), 1);
    }

    #[test]
    fn test_remove_channel() {
        let mut interface = MadiInterface::new("Test".to_string());

        let channel = MadiChannel {
            channel: 0,
            label: "Test".to_string(),
            active: true,
            destination: None,
        };

        interface
            .add_channel(channel)
            .expect("should succeed in test");
        assert!(interface.get_channel(0).is_some());

        interface.remove_channel(0);
        assert!(interface.get_channel(0).is_none());
    }
}
