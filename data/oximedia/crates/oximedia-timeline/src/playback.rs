//! Playback engine for timeline.

use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};

use crate::error::{TimelineError, TimelineResult};
use crate::timeline::Timeline;
use crate::types::Position;

/// Playback state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlaybackState {
    /// Stopped.
    Stopped,
    /// Playing forward.
    Playing,
    /// Paused.
    Paused,
    /// Playing in reverse.
    Reverse,
}

/// Playback mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlaybackMode {
    /// Normal playback.
    Normal,
    /// Loop playback.
    Loop,
    /// Ping-pong playback.
    PingPong,
}

/// Playback speed multiplier.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlaybackSpeed(pub f64);

impl PlaybackSpeed {
    /// Normal playback speed (1.0x).
    #[must_use]
    pub const fn normal() -> Self {
        Self(1.0)
    }

    /// Half speed (0.5x).
    #[must_use]
    pub const fn half() -> Self {
        Self(0.5)
    }

    /// Double speed (2.0x).
    #[must_use]
    pub const fn double() -> Self {
        Self(2.0)
    }

    /// Creates a new playback speed.
    ///
    /// # Errors
    ///
    /// Returns error if speed is not between 0.1 and 2.0.
    pub fn new(speed: f64) -> TimelineResult<Self> {
        if !(0.1..=2.0).contains(&speed) {
            return Err(TimelineError::Other(format!(
                "Invalid playback speed: {speed} (must be 0.1-2.0)"
            )));
        }
        Ok(Self(speed))
    }
}

impl Default for PlaybackSpeed {
    fn default() -> Self {
        Self::normal()
    }
}

/// Playback command.
#[derive(Clone, Debug)]
pub enum PlaybackCommand {
    /// Play from current position.
    Play,
    /// Pause playback.
    Pause,
    /// Stop playback and reset to start.
    Stop,
    /// Seek to position.
    Seek(Position),
    /// Set playback speed.
    SetSpeed(PlaybackSpeed),
    /// Set playback mode.
    SetMode(PlaybackMode),
    /// Step forward one frame.
    StepForward,
    /// Step backward one frame.
    StepBackward,
}

/// Playback event.
#[derive(Clone, Debug)]
pub enum PlaybackEvent {
    /// Playback state changed.
    StateChanged(PlaybackState),
    /// Playhead position changed.
    PositionChanged(Position),
    /// Reached end of timeline.
    EndReached,
    /// Playback error.
    Error(String),
}

/// Playback controller.
pub struct PlaybackController {
    #[allow(dead_code)]
    timeline: Arc<RwLock<Timeline>>,
    state: Arc<RwLock<PlaybackState>>,
    position: Arc<RwLock<Position>>,
    speed: Arc<RwLock<PlaybackSpeed>>,
    mode: Arc<RwLock<PlaybackMode>>,
    command_tx: mpsc::UnboundedSender<PlaybackCommand>,
    command_rx: Arc<tokio::sync::Mutex<mpsc::UnboundedReceiver<PlaybackCommand>>>,
    event_tx: mpsc::UnboundedSender<PlaybackEvent>,
}

impl PlaybackController {
    /// Creates a new playback controller.
    #[must_use]
    pub fn new(timeline: Arc<RwLock<Timeline>>) -> (Self, mpsc::UnboundedReceiver<PlaybackEvent>) {
        let (command_tx, command_rx) = mpsc::unbounded_channel();
        let (event_tx, event_rx) = mpsc::unbounded_channel();

        let controller = Self {
            timeline,
            state: Arc::new(RwLock::new(PlaybackState::Stopped)),
            position: Arc::new(RwLock::new(Position::zero())),
            speed: Arc::new(RwLock::new(PlaybackSpeed::normal())),
            mode: Arc::new(RwLock::new(PlaybackMode::Normal)),
            command_tx,
            command_rx: Arc::new(tokio::sync::Mutex::new(command_rx)),
            event_tx,
        };

        (controller, event_rx)
    }

    /// Sends a playback command.
    ///
    /// # Errors
    ///
    /// Returns error if command channel is closed.
    pub fn send_command(&self, command: PlaybackCommand) -> TimelineResult<()> {
        self.command_tx
            .send(command)
            .map_err(|_| TimelineError::PlaybackError("Command channel closed".to_string()))?;
        Ok(())
    }

    /// Plays from current position.
    ///
    /// # Errors
    ///
    /// Returns error if command fails.
    pub fn play(&self) -> TimelineResult<()> {
        self.send_command(PlaybackCommand::Play)
    }

    /// Pauses playback.
    ///
    /// # Errors
    ///
    /// Returns error if command fails.
    pub fn pause(&self) -> TimelineResult<()> {
        self.send_command(PlaybackCommand::Pause)
    }

    /// Stops playback.
    ///
    /// # Errors
    ///
    /// Returns error if command fails.
    pub fn stop(&self) -> TimelineResult<()> {
        self.send_command(PlaybackCommand::Stop)
    }

    /// Seeks to a position.
    ///
    /// # Errors
    ///
    /// Returns error if command fails.
    pub fn seek(&self, position: Position) -> TimelineResult<()> {
        self.send_command(PlaybackCommand::Seek(position))
    }

    /// Sets playback speed.
    ///
    /// # Errors
    ///
    /// Returns error if command fails.
    pub fn set_speed(&self, speed: PlaybackSpeed) -> TimelineResult<()> {
        self.send_command(PlaybackCommand::SetSpeed(speed))
    }

    /// Steps forward one frame.
    ///
    /// # Errors
    ///
    /// Returns error if command fails.
    pub fn step_forward(&self) -> TimelineResult<()> {
        self.send_command(PlaybackCommand::StepForward)
    }

    /// Steps backward one frame.
    ///
    /// # Errors
    ///
    /// Returns error if command fails.
    pub fn step_backward(&self) -> TimelineResult<()> {
        self.send_command(PlaybackCommand::StepBackward)
    }

    /// Gets current playback state.
    #[must_use]
    pub fn state(&self) -> PlaybackState {
        *self.state.blocking_read()
    }

    /// Gets current position.
    #[must_use]
    pub fn position(&self) -> Position {
        *self.position.blocking_read()
    }

    /// Gets current playback speed.
    #[must_use]
    pub fn speed(&self) -> PlaybackSpeed {
        *self.speed.blocking_read()
    }

    /// Runs the playback loop.
    pub async fn run(&self) -> TimelineResult<()> {
        loop {
            let mut rx = self.command_rx.lock().await;
            let command = rx.recv().await;
            drop(rx);

            match command {
                Some(PlaybackCommand::Play) => {
                    *self.state.write().await = PlaybackState::Playing;
                    let _ = self
                        .event_tx
                        .send(PlaybackEvent::StateChanged(PlaybackState::Playing));
                }
                Some(PlaybackCommand::Pause) => {
                    *self.state.write().await = PlaybackState::Paused;
                    let _ = self
                        .event_tx
                        .send(PlaybackEvent::StateChanged(PlaybackState::Paused));
                }
                Some(PlaybackCommand::Stop) => {
                    *self.state.write().await = PlaybackState::Stopped;
                    *self.position.write().await = Position::zero();
                    let _ = self
                        .event_tx
                        .send(PlaybackEvent::StateChanged(PlaybackState::Stopped));
                    let _ = self
                        .event_tx
                        .send(PlaybackEvent::PositionChanged(Position::zero()));
                }
                Some(PlaybackCommand::Seek(pos)) => {
                    *self.position.write().await = pos;
                    let _ = self.event_tx.send(PlaybackEvent::PositionChanged(pos));
                }
                Some(PlaybackCommand::SetSpeed(speed)) => {
                    *self.speed.write().await = speed;
                }
                Some(PlaybackCommand::SetMode(mode)) => {
                    *self.mode.write().await = mode;
                }
                Some(PlaybackCommand::StepForward) => {
                    let mut pos = self.position.write().await;
                    *pos = Position::new(pos.value() + 1);
                    let _ = self.event_tx.send(PlaybackEvent::PositionChanged(*pos));
                }
                Some(PlaybackCommand::StepBackward) => {
                    let mut pos = self.position.write().await;
                    *pos = Position::new((pos.value() - 1).max(0));
                    let _ = self.event_tx.send(PlaybackEvent::PositionChanged(*pos));
                }
                None => break,
            }
        }

        Ok(())
    }
}

/// Trait for playback output.
#[async_trait]
pub trait PlaybackOutput: Send + Sync {
    /// Outputs a frame.
    ///
    /// # Errors
    ///
    /// Returns error if output fails.
    async fn output_frame(&mut self, position: Position, data: &[u8]) -> TimelineResult<()>;

    /// Outputs audio samples.
    ///
    /// # Errors
    ///
    /// Returns error if output fails.
    async fn output_audio(&mut self, position: Position, data: &[f32]) -> TimelineResult<()>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_core::Rational;

    fn create_test_timeline() -> Arc<RwLock<Timeline>> {
        Arc::new(RwLock::new(
            Timeline::new("Test", Rational::new(24, 1), 48000).expect("should succeed in test"),
        ))
    }

    #[test]
    fn test_playback_speed() {
        assert!(PlaybackSpeed::new(1.0).is_ok());
        assert!(PlaybackSpeed::new(0.5).is_ok());
        assert!(PlaybackSpeed::new(2.0).is_ok());
        assert!(PlaybackSpeed::new(0.05).is_err());
        assert!(PlaybackSpeed::new(3.0).is_err());
    }

    #[test]
    fn test_playback_controller_creation() {
        let timeline = create_test_timeline();
        let (controller, _events) = PlaybackController::new(timeline);
        assert_eq!(controller.state(), PlaybackState::Stopped);
        assert_eq!(controller.position().value(), 0);
    }

    #[test]
    fn test_playback_commands() {
        let timeline = create_test_timeline();
        let (controller, _events) = PlaybackController::new(timeline);
        assert!(controller.play().is_ok());
        assert!(controller.pause().is_ok());
        assert!(controller.stop().is_ok());
        assert!(controller.seek(Position::new(100)).is_ok());
        assert!(controller.set_speed(PlaybackSpeed::double()).is_ok());
    }

    #[tokio::test]
    async fn test_playback_state_changes() {
        let timeline = create_test_timeline();
        let (controller, mut events) = PlaybackController::new(timeline);
        let controller = Arc::new(controller);

        let controller_clone = Arc::clone(&controller);
        tokio::spawn(async move { controller_clone.run().await });

        controller.play().expect("should succeed in test");
        if let Some(PlaybackEvent::StateChanged(state)) = events.recv().await {
            assert_eq!(state, PlaybackState::Playing);
        }

        controller.pause().expect("should succeed in test");
        if let Some(PlaybackEvent::StateChanged(state)) = events.recv().await {
            assert_eq!(state, PlaybackState::Paused);
        }
    }
}

// Note: Clone cannot be implemented due to Mutex in command_rx which requires careful handling
// Use Arc<PlaybackController> instead if you need to share across async tasks
