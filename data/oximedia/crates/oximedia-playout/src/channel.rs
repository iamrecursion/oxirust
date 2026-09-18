//! Multi-channel playout support
//!
//! Manages multiple independent playout channels with shared content library,
//! independent playlists, and channel-specific branding.

use crate::{PlayoutConfig, PlayoutError, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;
use uuid::Uuid;

/// Channel configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelConfig {
    /// Unique channel ID
    pub id: Uuid,

    /// Channel name
    pub name: String,

    /// Channel number (for display)
    pub number: u16,

    /// Playout configuration for this channel
    pub playout_config: PlayoutConfig,

    /// Channel branding
    pub branding: ChannelBranding,

    /// Output configuration
    pub outputs: Vec<ChannelOutput>,

    /// Enable this channel
    pub enabled: bool,

    /// Priority (for resource allocation)
    pub priority: u8,
}

/// Channel branding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelBranding {
    /// Station logo path
    pub logo_path: Option<PathBuf>,

    /// Logo position
    pub logo_position: LogoPosition,

    /// Logo opacity (0.0-1.0)
    pub logo_opacity: f32,

    /// Channel bug/watermark
    pub bug_enabled: bool,

    /// Default lower third template
    pub lower_third_template: Option<String>,

    /// Station ID audio
    pub station_id_audio: Option<PathBuf>,
}

impl Default for ChannelBranding {
    fn default() -> Self {
        Self {
            logo_path: None,
            logo_position: LogoPosition::TopRight,
            logo_opacity: 0.8,
            bug_enabled: false,
            lower_third_template: None,
            station_id_audio: None,
        }
    }
}

/// Logo position options
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum LogoPosition {
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

/// Channel output configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelOutput {
    /// Output ID
    pub id: Uuid,

    /// Output type
    pub output_type: OutputType,

    /// Output destination
    pub destination: String,

    /// Enabled flag
    pub enabled: bool,
}

/// Output type enumeration
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum OutputType {
    Sdi,
    Ndi,
    Rtmp,
    Srt,
    Hls,
    Dash,
    File,
}

/// Channel state
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ChannelState {
    Stopped,
    Starting,
    Running,
    Paused,
    Error,
    Stopping,
}

/// Channel instance
pub struct Channel {
    config: ChannelConfig,
    state: Arc<RwLock<ChannelState>>,
    current_playlist: Arc<RwLock<Option<Uuid>>>,
    current_item: Arc<RwLock<Option<Uuid>>>,
    start_time: Arc<RwLock<Option<DateTime<Utc>>>>,
}

impl Channel {
    /// Create new channel
    pub fn new(config: ChannelConfig) -> Self {
        Self {
            config,
            state: Arc::new(RwLock::new(ChannelState::Stopped)),
            current_playlist: Arc::new(RwLock::new(None)),
            current_item: Arc::new(RwLock::new(None)),
            start_time: Arc::new(RwLock::new(None)),
        }
    }

    /// Start channel
    pub async fn start(&self) -> Result<()> {
        let mut state = self.state.write().await;
        if *state != ChannelState::Stopped {
            return Err(PlayoutError::Config("Channel is not stopped".to_string()));
        }

        *state = ChannelState::Starting;
        info!("Starting channel: {}", self.config.name);

        // Initialize outputs
        // In real implementation, this would start actual output devices

        *state = ChannelState::Running;
        *self.start_time.write().await = Some(Utc::now());

        Ok(())
    }

    /// Stop channel
    pub async fn stop(&self) -> Result<()> {
        let mut state = self.state.write().await;
        *state = ChannelState::Stopping;

        info!("Stopping channel: {}", self.config.name);

        // Clean up resources
        *self.current_playlist.write().await = None;
        *self.current_item.write().await = None;
        *self.start_time.write().await = None;

        *state = ChannelState::Stopped;

        Ok(())
    }

    /// Pause channel
    pub async fn pause(&self) -> Result<()> {
        let mut state = self.state.write().await;
        if *state == ChannelState::Running {
            *state = ChannelState::Paused;
            info!("Paused channel: {}", self.config.name);
        }
        Ok(())
    }

    /// Resume channel
    pub async fn resume(&self) -> Result<()> {
        let mut state = self.state.write().await;
        if *state == ChannelState::Paused {
            *state = ChannelState::Running;
            info!("Resumed channel: {}", self.config.name);
        }
        Ok(())
    }

    /// Load playlist
    pub async fn load_playlist(&self, playlist_id: Uuid) -> Result<()> {
        *self.current_playlist.write().await = Some(playlist_id);
        info!(
            "Loaded playlist {} on channel {}",
            playlist_id, self.config.name
        );
        Ok(())
    }

    /// Get current state
    pub async fn state(&self) -> ChannelState {
        *self.state.read().await
    }

    /// Get channel configuration
    pub fn config(&self) -> &ChannelConfig {
        &self.config
    }

    /// Get current playlist ID
    pub async fn current_playlist(&self) -> Option<Uuid> {
        *self.current_playlist.read().await
    }

    /// Get uptime
    pub async fn uptime(&self) -> Option<chrono::Duration> {
        (*self.start_time.read().await).map(|start| Utc::now() - start)
    }
}

/// Multi-channel manager
pub struct ChannelManager {
    channels: Arc<RwLock<HashMap<Uuid, Channel>>>,
}

impl ChannelManager {
    /// Create new channel manager
    pub fn new() -> Self {
        Self {
            channels: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Add channel
    pub async fn add_channel(&self, config: ChannelConfig) -> Result<Uuid> {
        let id = config.id;
        let channel = Channel::new(config);

        {
            let mut channels = self.channels.write().await;
            channels.insert(id, channel);
        }

        info!("Added channel: {}", id);
        Ok(id)
    }

    /// Remove channel
    pub async fn remove_channel(&self, id: &Uuid) -> Result<()> {
        // Stop channel first
        if let Some(channel) = self.get_channel(id).await {
            channel.stop().await?;
        }

        {
            let mut channels = self.channels.write().await;
            channels.remove(id);
        }

        info!("Removed channel: {}", id);
        Ok(())
    }

    /// Get channel reference
    pub async fn get_channel(&self, id: &Uuid) -> Option<Channel> {
        let channels = self.channels.read().await;
        channels.get(id).map(|c| Channel {
            config: c.config.clone(),
            state: Arc::clone(&c.state),
            current_playlist: Arc::clone(&c.current_playlist),
            current_item: Arc::clone(&c.current_item),
            start_time: Arc::clone(&c.start_time),
        })
    }

    /// Start all channels
    pub async fn start_all(&self) -> Result<()> {
        let channels = self.channels.read().await;
        for channel in channels.values() {
            if channel.config.enabled {
                channel.start().await?;
            }
        }
        Ok(())
    }

    /// Stop all channels
    pub async fn stop_all(&self) -> Result<()> {
        let channels = self.channels.read().await;
        for channel in channels.values() {
            channel.stop().await?;
        }
        Ok(())
    }

    /// Get all channel IDs
    pub async fn list_channels(&self) -> Vec<Uuid> {
        let channels = self.channels.read().await;
        channels.keys().copied().collect()
    }

    /// Get channel statistics
    pub async fn get_statistics(&self) -> ChannelStatistics {
        let channels = self.channels.read().await;

        let total = channels.len();
        let mut running = 0;
        let mut paused = 0;
        let mut stopped = 0;
        let mut error = 0;

        for channel in channels.values() {
            match *channel.state.read().await {
                ChannelState::Running => running += 1,
                ChannelState::Paused => paused += 1,
                ChannelState::Stopped => stopped += 1,
                ChannelState::Error => error += 1,
                _ => {}
            }
        }

        ChannelStatistics {
            total_channels: total,
            running_channels: running,
            paused_channels: paused,
            stopped_channels: stopped,
            error_channels: error,
        }
    }
}

impl Default for ChannelManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Channel statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelStatistics {
    pub total_channels: usize,
    pub running_channels: usize,
    pub paused_channels: usize,
    pub stopped_channels: usize,
    pub error_channels: usize,
}

/// Channel group for organizing multiple channels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelGroup {
    /// Group ID
    pub id: Uuid,

    /// Group name
    pub name: String,

    /// Channel IDs in this group
    pub channel_ids: Vec<Uuid>,

    /// Group description
    pub description: Option<String>,
}

impl ChannelGroup {
    /// Create new channel group
    pub fn new(name: String) -> Self {
        Self {
            id: Uuid::new_v4(),
            name,
            channel_ids: Vec::new(),
            description: None,
        }
    }

    /// Add channel to group
    pub fn add_channel(&mut self, channel_id: Uuid) {
        if !self.channel_ids.contains(&channel_id) {
            self.channel_ids.push(channel_id);
        }
    }

    /// Remove channel from group
    pub fn remove_channel(&mut self, channel_id: &Uuid) {
        self.channel_ids.retain(|id| id != channel_id);
    }

    /// Check if channel is in group
    pub fn contains_channel(&self, channel_id: &Uuid) -> bool {
        self.channel_ids.contains(channel_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> ChannelConfig {
        ChannelConfig {
            id: Uuid::new_v4(),
            name: "Test Channel".to_string(),
            number: 1,
            playout_config: PlayoutConfig::default(),
            branding: ChannelBranding::default(),
            outputs: vec![],
            enabled: true,
            priority: 100,
        }
    }

    #[test]
    fn test_channel_branding_default() {
        let branding = ChannelBranding::default();
        assert_eq!(branding.logo_position, LogoPosition::TopRight);
        assert_eq!(branding.logo_opacity, 0.8);
        assert!(!branding.bug_enabled);
    }

    #[test]
    fn test_logo_position_equality() {
        assert_eq!(LogoPosition::TopLeft, LogoPosition::TopLeft);
        assert_ne!(LogoPosition::TopLeft, LogoPosition::TopRight);
    }

    #[test]
    fn test_output_type_equality() {
        assert_eq!(OutputType::Sdi, OutputType::Sdi);
        assert_ne!(OutputType::Sdi, OutputType::Ndi);
    }

    #[test]
    fn test_channel_state_equality() {
        assert_eq!(ChannelState::Running, ChannelState::Running);
        assert_ne!(ChannelState::Running, ChannelState::Stopped);
    }

    #[tokio::test]
    async fn test_channel_creation() {
        let config = create_test_config();
        let channel = Channel::new(config);

        let state = channel.state().await;
        assert_eq!(state, ChannelState::Stopped);
    }

    #[tokio::test]
    async fn test_channel_start_stop() {
        let config = create_test_config();
        let channel = Channel::new(config);

        // Start channel
        channel.start().await.expect("should succeed in test");
        assert_eq!(channel.state().await, ChannelState::Running);

        // Stop channel
        channel.stop().await.expect("should succeed in test");
        assert_eq!(channel.state().await, ChannelState::Stopped);
    }

    #[tokio::test]
    async fn test_channel_pause_resume() {
        let config = create_test_config();
        let channel = Channel::new(config);

        channel.start().await.expect("should succeed in test");

        // Pause
        channel.pause().await.expect("should succeed in test");
        assert_eq!(channel.state().await, ChannelState::Paused);

        // Resume
        channel.resume().await.expect("should succeed in test");
        assert_eq!(channel.state().await, ChannelState::Running);
    }

    #[tokio::test]
    async fn test_channel_playlist_loading() {
        let config = create_test_config();
        let channel = Channel::new(config);

        let playlist_id = Uuid::new_v4();
        channel
            .load_playlist(playlist_id)
            .await
            .expect("should succeed in test");

        let current = channel.current_playlist().await;
        assert_eq!(current, Some(playlist_id));
    }

    #[tokio::test]
    async fn test_channel_uptime() {
        let config = create_test_config();
        let channel = Channel::new(config);

        // No uptime when stopped
        assert!(channel.uptime().await.is_none());

        // Has uptime when running
        channel.start().await.expect("should succeed in test");
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        let uptime = channel.uptime().await;
        assert!(uptime.is_some());
        assert!(uptime.expect("should succeed in test").num_milliseconds() >= 100);
    }

    #[tokio::test]
    async fn test_channel_manager_creation() {
        let manager = ChannelManager::new();
        let channels = manager.list_channels().await;
        assert!(channels.is_empty());
    }

    #[tokio::test]
    async fn test_channel_manager_add_remove() {
        let manager = ChannelManager::new();
        let config = create_test_config();
        let id = config.id;

        // Add channel
        manager
            .add_channel(config)
            .await
            .expect("should succeed in test");
        let channels = manager.list_channels().await;
        assert_eq!(channels.len(), 1);

        // Remove channel
        manager
            .remove_channel(&id)
            .await
            .expect("should succeed in test");
        let channels = manager.list_channels().await;
        assert!(channels.is_empty());
    }

    #[tokio::test]
    async fn test_channel_manager_statistics() {
        let manager = ChannelManager::new();

        let config1 = create_test_config();
        let config2 = create_test_config();

        manager
            .add_channel(config1)
            .await
            .expect("should succeed in test");
        manager
            .add_channel(config2)
            .await
            .expect("should succeed in test");

        let stats = manager.get_statistics().await;
        assert_eq!(stats.total_channels, 2);
        assert_eq!(stats.stopped_channels, 2);
    }

    #[test]
    fn test_channel_group_creation() {
        let group = ChannelGroup::new("Test Group".to_string());
        assert_eq!(group.name, "Test Group");
        assert!(group.channel_ids.is_empty());
    }

    #[test]
    fn test_channel_group_add_remove() {
        let mut group = ChannelGroup::new("Test Group".to_string());
        let channel_id = Uuid::new_v4();

        // Add channel
        group.add_channel(channel_id);
        assert!(group.contains_channel(&channel_id));

        // Remove channel
        group.remove_channel(&channel_id);
        assert!(!group.contains_channel(&channel_id));
    }

    #[test]
    fn test_channel_group_duplicate_add() {
        let mut group = ChannelGroup::new("Test Group".to_string());
        let channel_id = Uuid::new_v4();

        group.add_channel(channel_id);
        group.add_channel(channel_id); // Add again

        assert_eq!(group.channel_ids.len(), 1); // Should not duplicate
    }

    #[test]
    fn test_channel_output_creation() {
        let output = ChannelOutput {
            id: Uuid::new_v4(),
            output_type: OutputType::Rtmp,
            destination: "rtmp://server/live/stream".to_string(),
            enabled: true,
        };

        assert_eq!(output.output_type, OutputType::Rtmp);
        assert!(output.enabled);
    }
}
