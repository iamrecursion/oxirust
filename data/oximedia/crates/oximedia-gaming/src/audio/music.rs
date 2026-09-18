//! Background music player.

use crate::GamingResult;

/// Music player for background music.
pub struct MusicPlayer {
    current_track: Option<MusicTrack>,
    volume: f32,
}

/// Music track.
#[derive(Debug, Clone)]
pub struct MusicTrack {
    /// Track file path
    pub path: String,
    /// Track title
    pub title: String,
    /// Volume (0.0 to 1.0)
    pub volume: f32,
}

impl MusicPlayer {
    /// Create a new music player.
    #[must_use]
    pub fn new() -> Self {
        Self {
            current_track: None,
            volume: 0.5,
        }
    }

    /// Play a track.
    pub fn play(&mut self, track: MusicTrack) -> GamingResult<()> {
        self.current_track = Some(track);
        Ok(())
    }

    /// Stop playback.
    pub fn stop(&mut self) {
        self.current_track = None;
    }

    /// Set volume.
    pub fn set_volume(&mut self, volume: f32) {
        self.volume = volume.clamp(0.0, 1.0);
    }

    /// Get volume.
    #[must_use]
    pub fn volume(&self) -> f32 {
        self.volume
    }
}

impl Default for MusicPlayer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_music_player_creation() {
        let player = MusicPlayer::new();
        assert_eq!(player.volume(), 0.5);
    }

    #[test]
    fn test_set_volume() {
        let mut player = MusicPlayer::new();
        player.set_volume(0.8);
        assert_eq!(player.volume(), 0.8);
    }
}
