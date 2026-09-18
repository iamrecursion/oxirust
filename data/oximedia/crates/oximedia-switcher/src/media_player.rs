//! Media player pool for clip playback inside the switcher.
#![allow(dead_code)]

use std::collections::HashMap;

/// State of a single media player.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayerState {
    /// No clip is loaded.
    Idle,
    /// A clip is loaded and paused at the current position.
    Paused,
    /// Actively playing.
    Playing,
    /// Playback has reached the end of the clip.
    Stopped,
}

impl PlayerState {
    /// Return `true` if the player is actively playing.
    pub fn is_playing(self) -> bool {
        matches!(self, Self::Playing)
    }

    /// Return `true` if the player has a clip loaded (paused or playing).
    pub fn has_clip(self) -> bool {
        matches!(self, Self::Paused | Self::Playing | Self::Stopped)
    }
}

/// A single media player capable of loading and playing one clip at a time.
#[derive(Debug)]
pub struct MediaPlayer {
    /// Unique player identifier within the pool.
    pub id: u32,
    state: PlayerState,
    clip_path: Option<String>,
    /// Current playback position in frames.
    position_frames: u64,
    /// Total duration of the loaded clip in frames.
    duration_frames: u64,
    /// Loop mode: restart automatically at end.
    loop_mode: bool,
}

impl MediaPlayer {
    /// Create a new idle player with the given id.
    pub fn new(id: u32) -> Self {
        Self {
            id,
            state: PlayerState::Idle,
            clip_path: None,
            position_frames: 0,
            duration_frames: 0,
            loop_mode: false,
        }
    }

    /// Load a clip for playback. Puts the player into `Paused` state.
    pub fn load_clip(&mut self, path: &str, duration_frames: u64) {
        self.clip_path = Some(path.to_owned());
        self.duration_frames = duration_frames;
        self.position_frames = 0;
        self.state = PlayerState::Paused;
    }

    /// Start playback. Has no effect if no clip is loaded or already playing.
    pub fn play(&mut self) {
        if matches!(self.state, PlayerState::Paused | PlayerState::Stopped) {
            self.state = PlayerState::Playing;
        }
    }

    /// Stop playback and reset to the beginning.
    pub fn stop(&mut self) {
        if self.state != PlayerState::Idle {
            self.state = PlayerState::Stopped;
            self.position_frames = 0;
        }
    }

    /// Advance one frame. Handles loop and end-of-clip.
    pub fn tick(&mut self) {
        if self.state == PlayerState::Playing && self.duration_frames > 0 {
            self.position_frames += 1;
            if self.position_frames >= self.duration_frames {
                if self.loop_mode {
                    self.position_frames = 0;
                } else {
                    self.state = PlayerState::Stopped;
                }
            }
        }
    }

    /// Return the current playback state.
    pub fn state(&self) -> PlayerState {
        self.state
    }

    /// Return the loaded clip path, if any.
    pub fn clip_path(&self) -> Option<&str> {
        self.clip_path.as_deref()
    }

    /// Return current playback position in frames.
    pub fn position_frames(&self) -> u64 {
        self.position_frames
    }

    /// Enable or disable loop mode.
    pub fn set_loop(&mut self, enabled: bool) {
        self.loop_mode = enabled;
    }
}

/// A pool of media players managed by the switcher.
#[derive(Debug, Default)]
pub struct MediaPlayerPool {
    players: HashMap<u32, MediaPlayer>,
    next_id: u32,
}

impl MediaPlayerPool {
    /// Create an empty pool.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a pool pre-populated with `count` idle players.
    pub fn with_capacity(count: u32) -> Self {
        let mut pool = Self::new();
        for _ in 0..count {
            pool.add_player();
        }
        pool
    }

    /// Add a new idle player to the pool. Returns its assigned id.
    pub fn add_player(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        self.players.insert(id, MediaPlayer::new(id));
        id
    }

    /// Find the first idle player.
    pub fn find_idle(&self) -> Option<u32> {
        self.players
            .values()
            .find(|p| p.state() == PlayerState::Idle)
            .map(|p| p.id)
    }

    /// Get an immutable reference to a player by id.
    pub fn get(&self, id: u32) -> Option<&MediaPlayer> {
        self.players.get(&id)
    }

    /// Get a mutable reference to a player by id.
    pub fn get_mut(&mut self, id: u32) -> Option<&mut MediaPlayer> {
        self.players.get_mut(&id)
    }

    /// Return total number of players.
    pub fn len(&self) -> usize {
        self.players.len()
    }

    /// Return `true` if the pool has no players.
    pub fn is_empty(&self) -> bool {
        self.players.is_empty()
    }

    /// Return the number of players currently playing.
    pub fn playing_count(&self) -> usize {
        self.players
            .values()
            .filter(|p| p.state().is_playing())
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_player_state_is_playing() {
        assert!(PlayerState::Playing.is_playing());
        assert!(!PlayerState::Idle.is_playing());
        assert!(!PlayerState::Paused.is_playing());
        assert!(!PlayerState::Stopped.is_playing());
    }

    #[test]
    fn test_player_state_has_clip() {
        assert!(!PlayerState::Idle.has_clip());
        assert!(PlayerState::Paused.has_clip());
        assert!(PlayerState::Playing.has_clip());
        assert!(PlayerState::Stopped.has_clip());
    }

    #[test]
    fn test_player_new_is_idle() {
        let p = MediaPlayer::new(0);
        assert_eq!(p.state(), PlayerState::Idle);
        assert!(p.clip_path().is_none());
    }

    #[test]
    fn test_player_load_clip() {
        let mut p = MediaPlayer::new(0);
        p.load_clip("clip.mp4", 300);
        assert_eq!(p.state(), PlayerState::Paused);
        assert_eq!(p.clip_path(), Some("clip.mp4"));
    }

    #[test]
    fn test_player_play_from_paused() {
        let mut p = MediaPlayer::new(0);
        p.load_clip("clip.mp4", 300);
        p.play();
        assert!(p.state().is_playing());
    }

    #[test]
    fn test_player_stop_resets_position() {
        let mut p = MediaPlayer::new(0);
        p.load_clip("clip.mp4", 300);
        p.play();
        p.tick();
        p.stop();
        assert_eq!(p.position_frames(), 0);
        assert_eq!(p.state(), PlayerState::Stopped);
    }

    #[test]
    fn test_player_tick_advances_position() {
        let mut p = MediaPlayer::new(0);
        p.load_clip("clip.mp4", 300);
        p.play();
        p.tick();
        p.tick();
        assert_eq!(p.position_frames(), 2);
    }

    #[test]
    fn test_player_stops_at_end_no_loop() {
        let mut p = MediaPlayer::new(0);
        p.load_clip("clip.mp4", 2);
        p.play();
        p.tick(); // frame 1
        p.tick(); // frame 2 -> end -> Stopped
        assert_eq!(p.state(), PlayerState::Stopped);
    }

    #[test]
    fn test_player_loops_at_end() {
        let mut p = MediaPlayer::new(0);
        p.load_clip("clip.mp4", 2);
        p.set_loop(true);
        p.play();
        p.tick();
        p.tick(); // wraps to 0
        assert_eq!(p.position_frames(), 0);
        assert!(p.state().is_playing());
    }

    #[test]
    fn test_pool_add_player_increments_id() {
        let mut pool = MediaPlayerPool::new();
        let id0 = pool.add_player();
        let id1 = pool.add_player();
        assert!(id1 > id0);
    }

    #[test]
    fn test_pool_with_capacity() {
        let pool = MediaPlayerPool::with_capacity(4);
        assert_eq!(pool.len(), 4);
    }

    #[test]
    fn test_pool_find_idle() {
        let mut pool = MediaPlayerPool::with_capacity(2);
        let idle_id = pool.find_idle().expect("should succeed in test");
        // Load a clip on that player
        pool.get_mut(idle_id)
            .expect("should succeed in test")
            .load_clip("c.mp4", 100);
        // A different idle player should be found (or none if only 1 left idle)
        // We just verify the first idle was returned and is now paused
        assert_eq!(
            pool.get(idle_id).expect("should succeed in test").state(),
            PlayerState::Paused
        );
    }

    #[test]
    fn test_pool_playing_count() {
        let mut pool = MediaPlayerPool::with_capacity(3);
        let ids: Vec<u32> = (0..3).map(|_| pool.add_player()).collect();
        for id in &ids[..2] {
            let p = pool.get_mut(*id).expect("should succeed in test");
            p.load_clip("c.mp4", 100);
            p.play();
        }
        assert_eq!(pool.playing_count(), 2);
    }
}
