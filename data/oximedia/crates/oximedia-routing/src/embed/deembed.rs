//! Audio de-embedder for extracting audio from video signals (SDI).

use serde::{Deserialize, Serialize};

/// Audio de-embedding configuration for SDI
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioDeembedder {
    /// Number of audio groups to de-embed from
    pub audio_groups: u8,
    /// Channels per group
    pub channels_per_group: u8,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Bit depth
    pub bit_depth: u8,
    /// Active channel extraction mapping
    pub channel_map: Vec<DeembedChannel>,
}

/// Represents a single de-embedded audio channel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeembedChannel {
    /// Source SDI group (0-3)
    pub sdi_group: u8,
    /// Source channel within group (0-3)
    pub sdi_channel: u8,
    /// Target audio channel index
    pub target_channel: usize,
    /// Whether this channel is active
    pub active: bool,
    /// Optional gain adjustment (in dB)
    pub gain_db: f32,
}

impl DeembedChannel {
    /// Create a new de-embed channel
    #[must_use]
    pub const fn new(sdi_group: u8, sdi_channel: u8, target_channel: usize) -> Self {
        Self {
            sdi_group,
            sdi_channel,
            target_channel,
            active: true,
            gain_db: 0.0,
        }
    }

    /// Set gain for this channel
    #[must_use]
    pub fn with_gain(mut self, gain_db: f32) -> Self {
        self.gain_db = gain_db;
        self
    }
}

impl Default for AudioDeembedder {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioDeembedder {
    /// Create a new audio de-embedder
    #[must_use]
    pub fn new() -> Self {
        Self {
            audio_groups: 4,
            channels_per_group: 4,
            sample_rate: 48000,
            bit_depth: 24,
            channel_map: Vec::new(),
        }
    }

    /// Add a channel to de-embed
    pub fn add_channel(&mut self, channel: DeembedChannel) -> Result<(), DeembedError> {
        if channel.sdi_group >= self.audio_groups {
            return Err(DeembedError::InvalidGroup(channel.sdi_group));
        }
        if channel.sdi_channel >= self.channels_per_group {
            return Err(DeembedError::InvalidChannel(channel.sdi_channel));
        }

        // Check for target conflicts
        for existing in &self.channel_map {
            if existing.target_channel == channel.target_channel {
                return Err(DeembedError::TargetConflict {
                    target: channel.target_channel,
                });
            }
        }

        self.channel_map.push(channel);
        Ok(())
    }

    /// Remove a channel from de-embedding
    pub fn remove_channel(&mut self, sdi_group: u8, sdi_channel: u8) {
        self.channel_map
            .retain(|ch| !(ch.sdi_group == sdi_group && ch.sdi_channel == sdi_channel));
    }

    /// Get total number of de-embedded channels
    #[must_use]
    pub fn deembedded_channel_count(&self) -> usize {
        self.channel_map.iter().filter(|ch| ch.active).count()
    }

    /// Get maximum possible channels
    #[must_use]
    pub const fn max_channels(&self) -> u8 {
        self.audio_groups * self.channels_per_group
    }

    /// Create standard stereo de-embedding (Group 1, Channels 0-1)
    #[must_use]
    pub fn stereo() -> Self {
        let mut deembedder = Self::new();
        let _ = deembedder.add_channel(DeembedChannel::new(0, 0, 0));
        let _ = deembedder.add_channel(DeembedChannel::new(0, 1, 1));
        deembedder
    }

    /// Create 5.1 surround de-embedding
    #[must_use]
    pub fn surround_51() -> Self {
        let mut deembedder = Self::new();
        for i in 0..6_usize {
            let group = (i / 4) as u8;
            let channel = (i % 4) as u8;
            let _ = deembedder.add_channel(DeembedChannel::new(group, channel, i));
        }
        deembedder
    }

    /// Create 8-channel de-embedding
    #[must_use]
    pub fn eight_channel() -> Self {
        let mut deembedder = Self::new();
        for i in 0..8_usize {
            let group = (i / 4) as u8;
            let channel = (i % 4) as u8;
            let _ = deembedder.add_channel(DeembedChannel::new(group, channel, i));
        }
        deembedder
    }

    /// Get channel mapping for a specific SDI channel
    #[must_use]
    pub fn get_mapping(&self, sdi_group: u8, sdi_channel: u8) -> Option<&DeembedChannel> {
        self.channel_map
            .iter()
            .find(|ch| ch.sdi_group == sdi_group && ch.sdi_channel == sdi_channel)
    }

    /// Validate the de-embedding configuration
    pub fn validate(&self) -> Result<(), DeembedError> {
        for channel in &self.channel_map {
            if channel.sdi_group >= self.audio_groups {
                return Err(DeembedError::InvalidGroup(channel.sdi_group));
            }
            if channel.sdi_channel >= self.channels_per_group {
                return Err(DeembedError::InvalidChannel(channel.sdi_channel));
            }
        }
        Ok(())
    }
}

/// Errors that can occur in audio de-embedding
#[derive(Debug, Clone, thiserror::Error)]
pub enum DeembedError {
    /// Invalid SDI group
    #[error("Invalid SDI group: {0}")]
    InvalidGroup(u8),
    /// Invalid channel within group
    #[error("Invalid channel: {0}")]
    InvalidChannel(u8),
    /// Target channel already assigned
    #[error("Target channel conflict: {target}")]
    TargetConflict { target: usize },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deembedder_creation() {
        let deembedder = AudioDeembedder::new();
        assert_eq!(deembedder.audio_groups, 4);
        assert_eq!(deembedder.max_channels(), 16);
    }

    #[test]
    fn test_add_channel() {
        let mut deembedder = AudioDeembedder::new();
        let channel = DeembedChannel::new(0, 0, 0);

        deembedder
            .add_channel(channel)
            .expect("should succeed in test");
        assert_eq!(deembedder.deembedded_channel_count(), 1);
    }

    #[test]
    fn test_target_conflict() {
        let mut deembedder = AudioDeembedder::new();

        deembedder
            .add_channel(DeembedChannel::new(0, 0, 0))
            .expect("should succeed in test");

        let result = deembedder.add_channel(DeembedChannel::new(0, 1, 0));
        assert!(matches!(result, Err(DeembedError::TargetConflict { .. })));
    }

    #[test]
    fn test_stereo_preset() {
        let deembedder = AudioDeembedder::stereo();
        assert_eq!(deembedder.deembedded_channel_count(), 2);
    }

    #[test]
    fn test_surround_51_preset() {
        let deembedder = AudioDeembedder::surround_51();
        assert_eq!(deembedder.deembedded_channel_count(), 6);
    }

    #[test]
    fn test_get_mapping() {
        let deembedder = AudioDeembedder::stereo();

        let mapping = deembedder.get_mapping(0, 0);
        assert!(mapping.is_some());
        assert_eq!(mapping.expect("should succeed in test").target_channel, 0);

        let no_mapping = deembedder.get_mapping(3, 3);
        assert!(no_mapping.is_none());
    }

    #[test]
    fn test_remove_channel() {
        let mut deembedder = AudioDeembedder::eight_channel();
        assert_eq!(deembedder.deembedded_channel_count(), 8);

        deembedder.remove_channel(0, 0);
        assert_eq!(deembedder.deembedded_channel_count(), 7);
    }
}
