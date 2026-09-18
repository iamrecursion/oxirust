//! Tally light protocol for camera status indication.

use crate::error::{VideoIpError, VideoIpResult};
use bytes::{Buf, BufMut, Bytes, BytesMut};
use serde::{Deserialize, Serialize};

/// Tally light state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TallyState {
    /// Camera is off/inactive.
    Off,
    /// Camera is on program (red tally).
    Program,
    /// Camera is on preview (green tally).
    Preview,
    /// Both program and preview.
    Both,
}

impl TallyState {
    /// Converts from a byte value.
    #[must_use]
    pub const fn from_u8(value: u8) -> Self {
        match value {
            0 => Self::Off,
            1 => Self::Program,
            2 => Self::Preview,
            3 => Self::Both,
            _ => Self::Off,
        }
    }

    /// Converts to a byte value.
    #[must_use]
    pub const fn to_u8(self) -> u8 {
        match self {
            Self::Off => 0,
            Self::Program => 1,
            Self::Preview => 2,
            Self::Both => 3,
        }
    }

    /// Returns true if the program tally is active.
    #[must_use]
    pub const fn is_program(self) -> bool {
        matches!(self, Self::Program | Self::Both)
    }

    /// Returns true if the preview tally is active.
    #[must_use]
    pub const fn is_preview(self) -> bool {
        matches!(self, Self::Preview | Self::Both)
    }
}

/// Tally message for camera status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TallyMessage {
    /// Camera/source identifier.
    pub source_id: u16,
    /// Tally state.
    pub state: TallyState,
    /// Brightness (0-255, 255 = full brightness).
    pub brightness: u8,
}

impl TallyMessage {
    /// Creates a new tally message.
    #[must_use]
    pub const fn new(source_id: u16, state: TallyState, brightness: u8) -> Self {
        Self {
            source_id,
            state,
            brightness,
        }
    }

    /// Encodes the tally message.
    #[must_use]
    pub fn encode(&self) -> Bytes {
        let mut buf = BytesMut::with_capacity(4);
        buf.put_u16(self.source_id);
        buf.put_u8(self.state.to_u8());
        buf.put_u8(self.brightness);
        buf.freeze()
    }

    /// Decodes a tally message.
    ///
    /// # Errors
    ///
    /// Returns an error if the data is invalid.
    pub fn decode(mut data: &[u8]) -> VideoIpResult<Self> {
        if data.len() < 4 {
            return Err(VideoIpError::Metadata(
                "insufficient tally data".to_string(),
            ));
        }

        let source_id = data.get_u16();
        let state = TallyState::from_u8(data.get_u8());
        let brightness = data.get_u8();

        Ok(Self::new(source_id, state, brightness))
    }
}

/// Tally controller for managing multiple camera tallies.
#[derive(Debug, Clone, Default)]
pub struct TallyController {
    /// Tally states for each source.
    states: std::collections::HashMap<u16, TallyState>,
}

impl TallyController {
    /// Creates a new tally controller.
    #[must_use]
    pub fn new() -> Self {
        Self {
            states: std::collections::HashMap::new(),
        }
    }

    /// Sets the tally state for a source.
    pub fn set_state(&mut self, source_id: u16, state: TallyState) {
        self.states.insert(source_id, state);
    }

    /// Gets the tally state for a source.
    #[must_use]
    pub fn get_state(&self, source_id: u16) -> TallyState {
        self.states
            .get(&source_id)
            .copied()
            .unwrap_or(TallyState::Off)
    }

    /// Clears the tally state for a source.
    pub fn clear_state(&mut self, source_id: u16) {
        self.states.remove(&source_id);
    }

    /// Clears all tally states.
    pub fn clear_all(&mut self) {
        self.states.clear();
    }

    /// Returns all active tallies.
    #[must_use]
    pub fn active_tallies(&self) -> Vec<(u16, TallyState)> {
        self.states
            .iter()
            .filter(|(_, state)| **state != TallyState::Off)
            .map(|(id, state)| (*id, *state))
            .collect()
    }

    /// Creates a tally message for a source.
    #[must_use]
    pub fn create_message(&self, source_id: u16, brightness: u8) -> TallyMessage {
        let state = self.get_state(source_id);
        TallyMessage::new(source_id, state, brightness)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tally_state_conversion() {
        assert_eq!(TallyState::Off.to_u8(), 0);
        assert_eq!(TallyState::from_u8(0), TallyState::Off);
        assert_eq!(TallyState::Program.to_u8(), 1);
        assert_eq!(TallyState::from_u8(1), TallyState::Program);
    }

    #[test]
    fn test_tally_state_checks() {
        assert!(TallyState::Program.is_program());
        assert!(!TallyState::Program.is_preview());
        assert!(TallyState::Preview.is_preview());
        assert!(!TallyState::Preview.is_program());
        assert!(TallyState::Both.is_program());
        assert!(TallyState::Both.is_preview());
    }

    #[test]
    fn test_tally_message_creation() {
        let msg = TallyMessage::new(42, TallyState::Program, 255);
        assert_eq!(msg.source_id, 42);
        assert_eq!(msg.state, TallyState::Program);
        assert_eq!(msg.brightness, 255);
    }

    #[test]
    fn test_tally_message_encode_decode() {
        let msg = TallyMessage::new(100, TallyState::Both, 128);
        let encoded = msg.encode();
        let decoded = TallyMessage::decode(&encoded).expect("should succeed in test");

        assert_eq!(decoded.source_id, 100);
        assert_eq!(decoded.state, TallyState::Both);
        assert_eq!(decoded.brightness, 128);
    }

    #[test]
    fn test_tally_controller() {
        let mut controller = TallyController::new();

        controller.set_state(1, TallyState::Program);
        controller.set_state(2, TallyState::Preview);

        assert_eq!(controller.get_state(1), TallyState::Program);
        assert_eq!(controller.get_state(2), TallyState::Preview);
        assert_eq!(controller.get_state(3), TallyState::Off);
    }

    #[test]
    fn test_tally_controller_clear() {
        let mut controller = TallyController::new();
        controller.set_state(1, TallyState::Program);
        controller.clear_state(1);

        assert_eq!(controller.get_state(1), TallyState::Off);
    }

    #[test]
    fn test_tally_controller_active_tallies() {
        let mut controller = TallyController::new();
        controller.set_state(1, TallyState::Program);
        controller.set_state(2, TallyState::Off);
        controller.set_state(3, TallyState::Preview);

        let active = controller.active_tallies();
        assert_eq!(active.len(), 2);
        assert!(active.contains(&(1, TallyState::Program)));
        assert!(active.contains(&(3, TallyState::Preview)));
    }

    #[test]
    fn test_tally_controller_create_message() {
        let mut controller = TallyController::new();
        controller.set_state(5, TallyState::Both);

        let msg = controller.create_message(5, 200);
        assert_eq!(msg.source_id, 5);
        assert_eq!(msg.state, TallyState::Both);
        assert_eq!(msg.brightness, 200);
    }
}
