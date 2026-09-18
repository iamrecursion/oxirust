//! Audio embedder for embedding audio into video signals (SDI).

use serde::{Deserialize, Serialize};

/// Audio embedding configuration for SDI
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioEmbedder {
    /// Number of audio groups (SDI supports up to 4 groups)
    pub audio_groups: u8,
    /// Channels per group (typically 4)
    pub channels_per_group: u8,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Bit depth
    pub bit_depth: u8,
    /// Active channel mapping
    pub channel_map: Vec<EmbedChannel>,
}

/// Represents a single embedded audio channel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedChannel {
    /// Source audio channel index
    pub source_channel: usize,
    /// Target SDI group (0-3)
    pub sdi_group: u8,
    /// Target channel within group (0-3)
    pub sdi_channel: u8,
    /// Whether this channel is active
    pub active: bool,
    /// Optional gain adjustment (in dB)
    pub gain_db: f32,
}

impl EmbedChannel {
    /// Create a new embed channel
    #[must_use]
    pub const fn new(source_channel: usize, sdi_group: u8, sdi_channel: u8) -> Self {
        Self {
            source_channel,
            sdi_group,
            sdi_channel,
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

impl Default for AudioEmbedder {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioEmbedder {
    /// Create a new audio embedder with default SDI configuration
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

    /// Add a channel to embed
    pub fn add_channel(&mut self, channel: EmbedChannel) -> Result<(), EmbedError> {
        if channel.sdi_group >= self.audio_groups {
            return Err(EmbedError::InvalidGroup(channel.sdi_group));
        }
        if channel.sdi_channel >= self.channels_per_group {
            return Err(EmbedError::InvalidChannel(channel.sdi_channel));
        }

        // Check for conflicts
        for existing in &self.channel_map {
            if existing.sdi_group == channel.sdi_group
                && existing.sdi_channel == channel.sdi_channel
            {
                return Err(EmbedError::ChannelConflict {
                    group: channel.sdi_group,
                    channel: channel.sdi_channel,
                });
            }
        }

        self.channel_map.push(channel);
        Ok(())
    }

    /// Remove a channel from embedding
    pub fn remove_channel(&mut self, sdi_group: u8, sdi_channel: u8) {
        self.channel_map
            .retain(|ch| !(ch.sdi_group == sdi_group && ch.sdi_channel == sdi_channel));
    }

    /// Get total number of embedded channels
    #[must_use]
    pub fn embedded_channel_count(&self) -> usize {
        self.channel_map.iter().filter(|ch| ch.active).count()
    }

    /// Get maximum possible channels
    #[must_use]
    pub const fn max_channels(&self) -> u8 {
        self.audio_groups * self.channels_per_group
    }

    /// Create standard stereo embedding (Group 1, Channels 0-1)
    #[must_use]
    pub fn stereo() -> Self {
        let mut embedder = Self::new();
        let _ = embedder.add_channel(EmbedChannel::new(0, 0, 0));
        let _ = embedder.add_channel(EmbedChannel::new(1, 0, 1));
        embedder
    }

    /// Create 5.1 surround embedding
    #[must_use]
    pub fn surround_51() -> Self {
        let mut embedder = Self::new();
        for i in 0..6_usize {
            let group = (i / 4) as u8;
            let channel = (i % 4) as u8;
            let _ = embedder.add_channel(EmbedChannel::new(i, group, channel));
        }
        embedder
    }

    /// Create 8-channel embedding
    #[must_use]
    pub fn eight_channel() -> Self {
        let mut embedder = Self::new();
        for i in 0..8_usize {
            let group = (i / 4) as u8;
            let channel = (i % 4) as u8;
            let _ = embedder.add_channel(EmbedChannel::new(i, group, channel));
        }
        embedder
    }

    /// Validate the embedding configuration
    pub fn validate(&self) -> Result<(), EmbedError> {
        for channel in &self.channel_map {
            if channel.sdi_group >= self.audio_groups {
                return Err(EmbedError::InvalidGroup(channel.sdi_group));
            }
            if channel.sdi_channel >= self.channels_per_group {
                return Err(EmbedError::InvalidChannel(channel.sdi_channel));
            }
        }
        Ok(())
    }
}

/// Errors that can occur in audio embedding
#[derive(Debug, Clone, thiserror::Error)]
pub enum EmbedError {
    /// Invalid SDI group
    #[error("Invalid SDI group: {0}")]
    InvalidGroup(u8),
    /// Invalid channel within group
    #[error("Invalid channel: {0}")]
    InvalidChannel(u8),
    /// Channel already assigned
    #[error("Channel conflict at group {group}, channel {channel}")]
    ChannelConflict { group: u8, channel: u8 },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedder_creation() {
        let embedder = AudioEmbedder::new();
        assert_eq!(embedder.audio_groups, 4);
        assert_eq!(embedder.channels_per_group, 4);
        assert_eq!(embedder.max_channels(), 16);
    }

    #[test]
    fn test_add_channel() {
        let mut embedder = AudioEmbedder::new();
        let channel = EmbedChannel::new(0, 0, 0);

        embedder
            .add_channel(channel)
            .expect("should succeed in test");
        assert_eq!(embedder.embedded_channel_count(), 1);
    }

    #[test]
    fn test_channel_conflict() {
        let mut embedder = AudioEmbedder::new();

        embedder
            .add_channel(EmbedChannel::new(0, 0, 0))
            .expect("should succeed in test");

        let result = embedder.add_channel(EmbedChannel::new(1, 0, 0));
        assert!(matches!(result, Err(EmbedError::ChannelConflict { .. })));
    }

    #[test]
    fn test_stereo_preset() {
        let embedder = AudioEmbedder::stereo();
        assert_eq!(embedder.embedded_channel_count(), 2);
    }

    #[test]
    fn test_surround_51_preset() {
        let embedder = AudioEmbedder::surround_51();
        assert_eq!(embedder.embedded_channel_count(), 6);
    }

    #[test]
    fn test_eight_channel_preset() {
        let embedder = AudioEmbedder::eight_channel();
        assert_eq!(embedder.embedded_channel_count(), 8);
    }

    #[test]
    fn test_remove_channel() {
        let mut embedder = AudioEmbedder::stereo();
        assert_eq!(embedder.embedded_channel_count(), 2);

        embedder.remove_channel(0, 0);
        assert_eq!(embedder.embedded_channel_count(), 1);
    }

    #[test]
    fn test_invalid_group() {
        let mut embedder = AudioEmbedder::new();
        let channel = EmbedChannel::new(0, 10, 0); // Invalid group

        assert!(matches!(
            embedder.add_channel(channel),
            Err(EmbedError::InvalidGroup(10))
        ));
    }

    #[test]
    fn test_channel_with_gain() {
        let channel = EmbedChannel::new(0, 0, 0).with_gain(-6.0);
        assert!((channel.gain_db - (-6.0)).abs() < f32::EPSILON);
    }
}
