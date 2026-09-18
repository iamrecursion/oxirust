//! Multi-channel broadcast management.

use crate::{Playlist, PlayoutEngine, Result, ScheduleEngine};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// A broadcast channel configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Channel {
    /// Unique channel ID.
    pub id: String,

    /// Channel name.
    pub name: String,

    /// Channel description.
    pub description: Option<String>,

    /// Channel number (for display).
    pub channel_number: u32,

    /// Whether the channel is enabled.
    pub enabled: bool,

    /// Output configuration (e.g., SDI output number, stream URL).
    pub output_config: OutputConfig,

    /// Tags for categorization.
    pub tags: Vec<String>,
}

/// Output configuration for a channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutputConfig {
    /// SDI output.
    Sdi {
        /// SDI output number.
        output: u32,
    },

    /// NDI output.
    Ndi {
        /// NDI stream name.
        name: String,
    },

    /// RTMP stream.
    Rtmp {
        /// RTMP URL.
        url: String,
    },

    /// SRT stream.
    Srt {
        /// SRT URL.
        url: String,
    },

    /// HLS stream.
    Hls {
        /// Output directory.
        output_dir: String,
    },

    /// Multiple outputs.
    Multi {
        /// List of outputs.
        outputs: Vec<Box<OutputConfig>>,
    },
}

impl Channel {
    /// Creates a new channel.
    #[must_use]
    pub fn new<S: Into<String>>(id: S, name: S, channel_number: u32, output: OutputConfig) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: None,
            channel_number,
            enabled: true,
            output_config: output,
            tags: Vec::new(),
        }
    }

    /// Sets the description.
    #[must_use]
    pub fn with_description<S: Into<String>>(mut self, description: S) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Adds a tag.
    #[must_use]
    pub fn with_tag<S: Into<String>>(mut self, tag: S) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Enables the channel.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disables the channel.
    pub fn disable(&mut self) {
        self.enabled = false;
    }
}

/// Channel state including playout and schedule engines.
struct ChannelState {
    channel: Channel,
    playout: PlayoutEngine,
    #[allow(dead_code)]
    schedule: ScheduleEngine,
    current_playlist: Option<Playlist>,
}

/// Manager for multiple broadcast channels.
pub struct ChannelManager {
    channels: Arc<RwLock<HashMap<String, ChannelState>>>,
}

impl ChannelManager {
    /// Creates a new channel manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            channels: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Adds a channel.
    pub fn add_channel(&self, channel: Channel) -> Result<()> {
        let mut channels = self
            .channels
            .write()
            .map_err(|e| crate::PlaylistError::RoutingError(format!("Lock error: {e}")))?;

        let channel_id = channel.id.clone();
        channels.insert(
            channel_id,
            ChannelState {
                channel,
                playout: PlayoutEngine::default(),
                schedule: ScheduleEngine::new(),
                current_playlist: None,
            },
        );

        Ok(())
    }

    /// Removes a channel.
    pub fn remove_channel(&self, channel_id: &str) -> Result<()> {
        let mut channels = self
            .channels
            .write()
            .map_err(|e| crate::PlaylistError::RoutingError(format!("Lock error: {e}")))?;

        channels.remove(channel_id);
        Ok(())
    }

    /// Gets a channel by ID.
    pub fn get_channel(&self, channel_id: &str) -> Result<Option<Channel>> {
        let channels = self
            .channels
            .read()
            .map_err(|e| crate::PlaylistError::RoutingError(format!("Lock error: {e}")))?;

        Ok(channels.get(channel_id).map(|state| state.channel.clone()))
    }

    /// Loads a playlist for a channel.
    pub fn load_playlist(&self, channel_id: &str, playlist: Playlist) -> Result<()> {
        let mut channels = self
            .channels
            .write()
            .map_err(|e| crate::PlaylistError::RoutingError(format!("Lock error: {e}")))?;

        if let Some(state) = channels.get_mut(channel_id) {
            state.playout.load_playlist(playlist.clone())?;
            state.current_playlist = Some(playlist);
            Ok(())
        } else {
            Err(crate::PlaylistError::RoutingError(format!(
                "Channel '{channel_id}' not found"
            )))
        }
    }

    /// Starts playback on a channel.
    pub fn start_channel(&self, channel_id: &str) -> Result<()> {
        let channels = self
            .channels
            .read()
            .map_err(|e| crate::PlaylistError::RoutingError(format!("Lock error: {e}")))?;

        if let Some(state) = channels.get(channel_id) {
            state.playout.play()?;
            Ok(())
        } else {
            Err(crate::PlaylistError::RoutingError(format!(
                "Channel '{channel_id}' not found"
            )))
        }
    }

    /// Stops playback on a channel.
    pub fn stop_channel(&self, channel_id: &str) -> Result<()> {
        let channels = self
            .channels
            .read()
            .map_err(|e| crate::PlaylistError::RoutingError(format!("Lock error: {e}")))?;

        if let Some(state) = channels.get(channel_id) {
            state.playout.stop()?;
            Ok(())
        } else {
            Err(crate::PlaylistError::RoutingError(format!(
                "Channel '{channel_id}' not found"
            )))
        }
    }

    /// Gets all channel IDs.
    pub fn get_channel_ids(&self) -> Result<Vec<String>> {
        let channels = self
            .channels
            .read()
            .map_err(|e| crate::PlaylistError::RoutingError(format!("Lock error: {e}")))?;

        Ok(channels.keys().cloned().collect())
    }

    /// Gets all channels.
    pub fn get_all_channels(&self) -> Result<Vec<Channel>> {
        let channels = self
            .channels
            .read()
            .map_err(|e| crate::PlaylistError::RoutingError(format!("Lock error: {e}")))?;

        Ok(channels
            .values()
            .map(|state| state.channel.clone())
            .collect())
    }

    /// Returns the number of channels.
    pub fn channel_count(&self) -> usize {
        self.channels.read().map(|c| c.len()).unwrap_or(0)
    }
}

impl Default for ChannelManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::playlist::PlaylistType;

    #[test]
    fn test_channel_creation() {
        let channel = Channel::new("channel1", "Channel 1", 1, OutputConfig::Sdi { output: 1 })
            .with_description("Main broadcast channel")
            .with_tag("main");

        assert_eq!(channel.channel_number, 1);
        assert!(channel.enabled);
    }

    #[test]
    fn test_channel_manager() {
        let manager = ChannelManager::new();
        let channel = Channel::new("channel1", "Channel 1", 1, OutputConfig::Sdi { output: 1 });

        manager
            .add_channel(channel)
            .expect("should succeed in test");
        assert_eq!(manager.channel_count(), 1);

        let retrieved = manager
            .get_channel("channel1")
            .expect("should succeed in test");
        assert!(retrieved.is_some());
    }

    #[test]
    fn test_playlist_loading() {
        let manager = ChannelManager::new();
        let channel = Channel::new("channel1", "Channel 1", 1, OutputConfig::Sdi { output: 1 });

        manager
            .add_channel(channel)
            .expect("should succeed in test");

        let playlist = Playlist::new("test", PlaylistType::Linear);
        manager
            .load_playlist("channel1", playlist)
            .expect("should succeed in test");
    }
}
