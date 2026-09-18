//! Audio Follow Video (AFV) logic for video switchers.
//!
//! Manages audio routing to follow video source selections or remain independent.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Errors that can occur with audio follow operations.
#[derive(Error, Debug, Clone)]
pub enum AudioFollowError {
    #[error("Invalid audio channel: {0}")]
    InvalidChannel(usize),

    #[error("Invalid video source: {0}")]
    InvalidSource(usize),

    #[error("Configuration error: {0}")]
    ConfigError(String),
}

/// Audio follow mode.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum AudioFollowMode {
    /// Audio follows the video source
    Follow,
    /// Audio is independent of video
    Independent,
}

/// Audio channel assignment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioChannelAssignment {
    /// Audio output channel
    pub channel: usize,
    /// Video source to follow (if in follow mode)
    pub video_source: Option<usize>,
    /// Direct audio source (if in independent mode)
    pub audio_source: Option<usize>,
    /// Follow mode
    pub mode: AudioFollowMode,
    /// Muted state
    pub muted: bool,
    /// Volume level (0.0 - 1.0)
    pub volume: f32,
}

impl AudioChannelAssignment {
    /// Create a new audio channel assignment.
    pub fn new(channel: usize) -> Self {
        Self {
            channel,
            video_source: None,
            audio_source: None,
            mode: AudioFollowMode::Follow,
            muted: false,
            volume: 1.0,
        }
    }

    /// Set to follow a video source.
    pub fn follow_video(&mut self, video_source: usize) {
        self.video_source = Some(video_source);
        self.mode = AudioFollowMode::Follow;
    }

    /// Set independent audio source.
    pub fn set_independent(&mut self, audio_source: usize) {
        self.audio_source = Some(audio_source);
        self.mode = AudioFollowMode::Independent;
    }

    /// Get the active audio source.
    pub fn active_source(&self) -> Option<usize> {
        match self.mode {
            AudioFollowMode::Follow => self.video_source,
            AudioFollowMode::Independent => self.audio_source,
        }
    }

    /// Set mute state.
    pub fn set_muted(&mut self, muted: bool) {
        self.muted = muted;
    }

    /// Set volume.
    pub fn set_volume(&mut self, volume: f32) {
        self.volume = volume.clamp(0.0, 1.0);
    }
}

/// Audio mixer channel configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioMixerChannel {
    /// Channel ID
    pub id: usize,
    /// Channel name
    pub name: String,
    /// Fader level (0.0 - 1.0)
    pub fader: f32,
    /// Mute state
    pub muted: bool,
    /// Solo state
    pub solo: bool,
    /// On-air state
    pub on_air: bool,
}

impl AudioMixerChannel {
    /// Create a new mixer channel.
    pub fn new(id: usize, name: String) -> Self {
        Self {
            id,
            name,
            fader: 0.75,
            muted: false,
            solo: false,
            on_air: false,
        }
    }

    /// Set fader level.
    pub fn set_fader(&mut self, level: f32) {
        self.fader = level.clamp(0.0, 1.0);
    }

    /// Toggle mute.
    pub fn toggle_mute(&mut self) {
        self.muted = !self.muted;
    }

    /// Toggle solo.
    pub fn toggle_solo(&mut self) {
        self.solo = !self.solo;
    }
}

/// Audio Follow Video manager.
pub struct AudioFollowManager {
    /// Channel assignments
    channels: HashMap<usize, AudioChannelAssignment>,
    /// Mixer channels
    mixer_channels: HashMap<usize, AudioMixerChannel>,
    /// Global AFV enable
    afv_enabled: bool,
}

impl AudioFollowManager {
    /// Create a new audio follow manager.
    pub fn new(num_channels: usize) -> Self {
        let mut channels = HashMap::new();
        for i in 0..num_channels {
            channels.insert(i, AudioChannelAssignment::new(i));
        }

        Self {
            channels,
            mixer_channels: HashMap::new(),
            afv_enabled: true,
        }
    }

    /// Enable or disable AFV globally.
    pub fn set_afv_enabled(&mut self, enabled: bool) {
        self.afv_enabled = enabled;
    }

    /// Check if AFV is enabled.
    pub fn is_afv_enabled(&self) -> bool {
        self.afv_enabled
    }

    /// Get a channel assignment.
    pub fn get_channel(&self, channel: usize) -> Result<&AudioChannelAssignment, AudioFollowError> {
        self.channels
            .get(&channel)
            .ok_or(AudioFollowError::InvalidChannel(channel))
    }

    /// Get a mutable channel assignment.
    pub fn get_channel_mut(
        &mut self,
        channel: usize,
    ) -> Result<&mut AudioChannelAssignment, AudioFollowError> {
        self.channels
            .get_mut(&channel)
            .ok_or(AudioFollowError::InvalidChannel(channel))
    }

    /// Set a channel to follow video.
    pub fn set_follow_video(
        &mut self,
        channel: usize,
        video_source: usize,
    ) -> Result<(), AudioFollowError> {
        let assignment = self.get_channel_mut(channel)?;
        assignment.follow_video(video_source);
        Ok(())
    }

    /// Set a channel to independent mode.
    pub fn set_independent(
        &mut self,
        channel: usize,
        audio_source: usize,
    ) -> Result<(), AudioFollowError> {
        let assignment = self.get_channel_mut(channel)?;
        assignment.set_independent(audio_source);
        Ok(())
    }

    /// Mute a channel.
    pub fn mute_channel(&mut self, channel: usize, muted: bool) -> Result<(), AudioFollowError> {
        let assignment = self.get_channel_mut(channel)?;
        assignment.set_muted(muted);
        Ok(())
    }

    /// Set channel volume.
    pub fn set_volume(&mut self, channel: usize, volume: f32) -> Result<(), AudioFollowError> {
        let assignment = self.get_channel_mut(channel)?;
        assignment.set_volume(volume);
        Ok(())
    }

    /// Update audio routing based on video source change.
    pub fn update_from_video(&mut self, video_source: usize) {
        if !self.afv_enabled {
            return;
        }

        for assignment in self.channels.values_mut() {
            if assignment.mode == AudioFollowMode::Follow {
                assignment.video_source = Some(video_source);
            }
        }
    }

    /// Get active audio sources for all channels.
    pub fn get_active_sources(&self) -> HashMap<usize, usize> {
        self.channels
            .iter()
            .filter_map(|(&channel, assignment)| {
                assignment.active_source().map(|source| (channel, source))
            })
            .collect()
    }

    /// Get all channels in follow mode.
    pub fn get_follow_channels(&self) -> Vec<usize> {
        self.channels
            .iter()
            .filter(|(_, a)| a.mode == AudioFollowMode::Follow && a.video_source.is_some())
            .map(|(id, _)| *id)
            .collect()
    }

    /// Get all channels in independent mode.
    pub fn get_independent_channels(&self) -> Vec<usize> {
        self.channels
            .iter()
            .filter(|(_, a)| a.mode == AudioFollowMode::Independent && a.audio_source.is_some())
            .map(|(id, _)| *id)
            .collect()
    }

    /// Add a mixer channel.
    pub fn add_mixer_channel(&mut self, channel: AudioMixerChannel) {
        self.mixer_channels.insert(channel.id, channel);
    }

    /// Get a mixer channel.
    pub fn get_mixer_channel(&self, id: usize) -> Option<&AudioMixerChannel> {
        self.mixer_channels.get(&id)
    }

    /// Get a mutable mixer channel.
    pub fn get_mixer_channel_mut(&mut self, id: usize) -> Option<&mut AudioMixerChannel> {
        self.mixer_channels.get_mut(&id)
    }

    /// Get the number of channels.
    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }

    /// Get the number of mixer channels.
    pub fn mixer_channel_count(&self) -> usize {
        self.mixer_channels.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_channel_assignment_creation() {
        let assignment = AudioChannelAssignment::new(0);
        assert_eq!(assignment.channel, 0);
        assert_eq!(assignment.mode, AudioFollowMode::Follow);
        assert!(!assignment.muted);
        assert_eq!(assignment.volume, 1.0);
    }

    #[test]
    fn test_follow_video() {
        let mut assignment = AudioChannelAssignment::new(0);
        assignment.follow_video(1);

        assert_eq!(assignment.mode, AudioFollowMode::Follow);
        assert_eq!(assignment.video_source, Some(1));
        assert_eq!(assignment.active_source(), Some(1));
    }

    #[test]
    fn test_independent_mode() {
        let mut assignment = AudioChannelAssignment::new(0);
        assignment.set_independent(5);

        assert_eq!(assignment.mode, AudioFollowMode::Independent);
        assert_eq!(assignment.audio_source, Some(5));
        assert_eq!(assignment.active_source(), Some(5));
    }

    #[test]
    fn test_mute_volume() {
        let mut assignment = AudioChannelAssignment::new(0);

        assignment.set_muted(true);
        assert!(assignment.muted);

        assignment.set_volume(0.5);
        assert_eq!(assignment.volume, 0.5);

        // Test clamping
        assignment.set_volume(1.5);
        assert_eq!(assignment.volume, 1.0);

        assignment.set_volume(-0.5);
        assert_eq!(assignment.volume, 0.0);
    }

    #[test]
    fn test_audio_mixer_channel() {
        let mut channel = AudioMixerChannel::new(0, "Channel 1".to_string());
        assert_eq!(channel.id, 0);
        assert_eq!(channel.name, "Channel 1");
        assert_eq!(channel.fader, 0.75);
        assert!(!channel.muted);
        assert!(!channel.solo);

        channel.set_fader(0.5);
        assert_eq!(channel.fader, 0.5);

        channel.toggle_mute();
        assert!(channel.muted);

        channel.toggle_solo();
        assert!(channel.solo);
    }

    #[test]
    fn test_audio_follow_manager_creation() {
        let manager = AudioFollowManager::new(4);
        assert_eq!(manager.channel_count(), 4);
        assert!(manager.is_afv_enabled());
    }

    #[test]
    fn test_set_follow_video() {
        let mut manager = AudioFollowManager::new(4);

        manager
            .set_follow_video(0, 1)
            .expect("should succeed in test");

        let assignment = manager.get_channel(0).expect("should succeed in test");
        assert_eq!(assignment.mode, AudioFollowMode::Follow);
        assert_eq!(assignment.video_source, Some(1));
    }

    #[test]
    fn test_set_independent() {
        let mut manager = AudioFollowManager::new(4);

        manager
            .set_independent(0, 5)
            .expect("should succeed in test");

        let assignment = manager.get_channel(0).expect("should succeed in test");
        assert_eq!(assignment.mode, AudioFollowMode::Independent);
        assert_eq!(assignment.audio_source, Some(5));
    }

    #[test]
    fn test_mute_channel() {
        let mut manager = AudioFollowManager::new(4);

        manager
            .mute_channel(0, true)
            .expect("should succeed in test");
        assert!(
            manager
                .get_channel(0)
                .expect("should succeed in test")
                .muted
        );

        manager
            .mute_channel(0, false)
            .expect("should succeed in test");
        assert!(
            !manager
                .get_channel(0)
                .expect("should succeed in test")
                .muted
        );
    }

    #[test]
    fn test_set_volume() {
        let mut manager = AudioFollowManager::new(4);

        manager.set_volume(0, 0.8).expect("should succeed in test");
        assert_eq!(
            manager
                .get_channel(0)
                .expect("should succeed in test")
                .volume,
            0.8
        );
    }

    #[test]
    fn test_update_from_video() {
        let mut manager = AudioFollowManager::new(4);

        // Set channels 0 and 1 to follow mode
        manager
            .set_follow_video(0, 1)
            .expect("should succeed in test");
        manager
            .set_follow_video(1, 1)
            .expect("should succeed in test");

        // Set channel 2 to independent
        manager
            .set_independent(2, 5)
            .expect("should succeed in test");

        // Update from video source 3
        manager.update_from_video(3);

        // Follow channels should update
        assert_eq!(
            manager
                .get_channel(0)
                .expect("should succeed in test")
                .active_source(),
            Some(3)
        );
        assert_eq!(
            manager
                .get_channel(1)
                .expect("should succeed in test")
                .active_source(),
            Some(3)
        );

        // Independent channel should not change
        assert_eq!(
            manager
                .get_channel(2)
                .expect("should succeed in test")
                .active_source(),
            Some(5)
        );
    }

    #[test]
    fn test_afv_disabled() {
        let mut manager = AudioFollowManager::new(4);
        manager
            .set_follow_video(0, 1)
            .expect("should succeed in test");

        // Disable AFV
        manager.set_afv_enabled(false);
        assert!(!manager.is_afv_enabled());

        // Update should not affect channels
        manager.update_from_video(3);
        assert_eq!(
            manager
                .get_channel(0)
                .expect("should succeed in test")
                .video_source,
            Some(1)
        );
    }

    #[test]
    fn test_get_active_sources() {
        let mut manager = AudioFollowManager::new(4);

        manager
            .set_follow_video(0, 1)
            .expect("should succeed in test");
        manager
            .set_independent(1, 5)
            .expect("should succeed in test");

        let sources = manager.get_active_sources();
        assert_eq!(sources.len(), 2);
        assert_eq!(sources.get(&0), Some(&1));
        assert_eq!(sources.get(&1), Some(&5));
    }

    #[test]
    fn test_get_follow_independent_channels() {
        let mut manager = AudioFollowManager::new(4);

        manager
            .set_follow_video(0, 1)
            .expect("should succeed in test");
        manager
            .set_follow_video(1, 2)
            .expect("should succeed in test");
        manager
            .set_independent(2, 5)
            .expect("should succeed in test");

        let follow = manager.get_follow_channels();
        let independent = manager.get_independent_channels();

        assert_eq!(follow.len(), 2);
        assert!(follow.contains(&0));
        assert!(follow.contains(&1));

        assert_eq!(independent.len(), 1);
        assert!(independent.contains(&2));
    }

    #[test]
    fn test_mixer_channels() {
        let mut manager = AudioFollowManager::new(4);

        let channel = AudioMixerChannel::new(0, "Mix 1".to_string());
        manager.add_mixer_channel(channel);

        assert_eq!(manager.mixer_channel_count(), 1);
        assert!(manager.get_mixer_channel(0).is_some());
        assert_eq!(
            manager
                .get_mixer_channel(0)
                .expect("should succeed in test")
                .name,
            "Mix 1"
        );
    }

    #[test]
    fn test_invalid_channel() {
        let manager = AudioFollowManager::new(4);
        assert!(manager.get_channel(10).is_err());
    }
}
