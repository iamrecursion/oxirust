//! PTZ (Pan-Tilt-Zoom) control protocol for camera control.

use crate::error::{VideoIpError, VideoIpResult};
use bytes::{Buf, BufMut, Bytes, BytesMut};
use serde::{Deserialize, Serialize};

/// PTZ command type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum PtzCommand {
    /// Pan left.
    PanLeft = 0,
    /// Pan right.
    PanRight = 1,
    /// Tilt up.
    TiltUp = 2,
    /// Tilt down.
    TiltDown = 3,
    /// Zoom in.
    ZoomIn = 4,
    /// Zoom out.
    ZoomOut = 5,
    /// Focus near.
    FocusNear = 6,
    /// Focus far.
    FocusFar = 7,
    /// Auto focus.
    AutoFocus = 8,
    /// Stop all movements.
    Stop = 9,
    /// Go to preset position.
    GotoPreset = 10,
    /// Save preset position.
    SavePreset = 11,
    /// Set absolute position.
    AbsolutePosition = 12,
    /// Home position.
    Home = 13,
}

impl PtzCommand {
    /// Converts from a byte value.
    #[must_use]
    pub const fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::PanLeft),
            1 => Some(Self::PanRight),
            2 => Some(Self::TiltUp),
            3 => Some(Self::TiltDown),
            4 => Some(Self::ZoomIn),
            5 => Some(Self::ZoomOut),
            6 => Some(Self::FocusNear),
            7 => Some(Self::FocusFar),
            8 => Some(Self::AutoFocus),
            9 => Some(Self::Stop),
            10 => Some(Self::GotoPreset),
            11 => Some(Self::SavePreset),
            12 => Some(Self::AbsolutePosition),
            13 => Some(Self::Home),
            _ => None,
        }
    }

    /// Converts to a byte value.
    #[must_use]
    pub const fn to_u8(self) -> u8 {
        self as u8
    }
}

/// PTZ control message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PtzMessage {
    /// Camera/source identifier.
    pub source_id: u16,
    /// PTZ command.
    pub command: PtzCommand,
    /// Pan speed (-100 to 100, negative = left, positive = right).
    pub pan_speed: i8,
    /// Tilt speed (-100 to 100, negative = down, positive = up).
    pub tilt_speed: i8,
    /// Zoom speed (-100 to 100, negative = out, positive = in).
    pub zoom_speed: i8,
    /// Preset number (for preset commands).
    pub preset: u8,
    /// Absolute pan position (-180.0 to 180.0 degrees).
    pub pan_position: f32,
    /// Absolute tilt position (-90.0 to 90.0 degrees).
    pub tilt_position: f32,
    /// Absolute zoom position (0.0 to 1.0, normalized).
    pub zoom_position: f32,
}

impl PtzMessage {
    /// Creates a new PTZ message.
    #[must_use]
    pub const fn new(source_id: u16, command: PtzCommand) -> Self {
        Self {
            source_id,
            command,
            pan_speed: 0,
            tilt_speed: 0,
            zoom_speed: 0,
            preset: 0,
            pan_position: 0.0,
            tilt_position: 0.0,
            zoom_position: 0.0,
        }
    }

    /// Sets the pan speed.
    #[must_use]
    pub const fn with_pan_speed(mut self, speed: i8) -> Self {
        self.pan_speed = speed;
        self
    }

    /// Sets the tilt speed.
    #[must_use]
    pub const fn with_tilt_speed(mut self, speed: i8) -> Self {
        self.tilt_speed = speed;
        self
    }

    /// Sets the zoom speed.
    #[must_use]
    pub const fn with_zoom_speed(mut self, speed: i8) -> Self {
        self.zoom_speed = speed;
        self
    }

    /// Sets the preset number.
    #[must_use]
    pub const fn with_preset(mut self, preset: u8) -> Self {
        self.preset = preset;
        self
    }

    /// Sets the absolute position.
    #[must_use]
    pub const fn with_position(mut self, pan: f32, tilt: f32, zoom: f32) -> Self {
        self.pan_position = pan;
        self.tilt_position = tilt;
        self.zoom_position = zoom;
        self
    }

    /// Wire size of an encoded PTZ message in bytes.
    /// 2 (`source_id`) + 1 (command) + 1 (`pan_speed`) + 1 (`tilt_speed`) +
    /// 1 (`zoom_speed`) + 1 (preset) + 4 (`pan_pos`) + 4 (`tilt_pos`) + 4 (`zoom_pos`) = 19.
    pub const ENCODED_SIZE: usize = 19;

    /// Encodes the PTZ message.
    #[must_use]
    pub fn encode(&self) -> Bytes {
        let mut buf = BytesMut::with_capacity(Self::ENCODED_SIZE);
        buf.put_u16(self.source_id);
        buf.put_u8(self.command.to_u8());
        buf.put_i8(self.pan_speed);
        buf.put_i8(self.tilt_speed);
        buf.put_i8(self.zoom_speed);
        buf.put_u8(self.preset);
        buf.put_f32(self.pan_position);
        buf.put_f32(self.tilt_position);
        buf.put_f32(self.zoom_position);
        buf.freeze()
    }

    /// Decodes a PTZ message.
    ///
    /// # Errors
    ///
    /// Returns an error if the data is invalid.
    pub fn decode(mut data: &[u8]) -> VideoIpResult<Self> {
        if data.len() < Self::ENCODED_SIZE {
            return Err(VideoIpError::Ptz("insufficient PTZ data".to_string()));
        }

        let source_id = data.get_u16();
        let command_byte = data.get_u8();
        let command = PtzCommand::from_u8(command_byte)
            .ok_or_else(|| VideoIpError::Ptz(format!("invalid command: {command_byte}")))?;

        let pan_speed = data.get_i8();
        let tilt_speed = data.get_i8();
        let zoom_speed = data.get_i8();
        let preset = data.get_u8();
        let pan_position = data.get_f32();
        let tilt_position = data.get_f32();
        let zoom_position = data.get_f32();

        Ok(Self {
            source_id,
            command,
            pan_speed,
            tilt_speed,
            zoom_speed,
            preset,
            pan_position,
            tilt_position,
            zoom_position,
        })
    }
}

/// PTZ controller for managing camera movements.
#[derive(Debug)]
pub struct PtzController {
    /// Current pan position.
    pan: f32,
    /// Current tilt position.
    tilt: f32,
    /// Current zoom position.
    zoom: f32,
    /// Pan speed limits (degrees per second).
    max_pan_speed: f32,
    /// Tilt speed limits (degrees per second).
    max_tilt_speed: f32,
    /// Zoom speed limits (normalized per second).
    max_zoom_speed: f32,
    /// Saved presets.
    presets: std::collections::HashMap<u8, (f32, f32, f32)>,
}

impl PtzController {
    /// Creates a new PTZ controller.
    #[must_use]
    pub fn new() -> Self {
        Self {
            pan: 0.0,
            tilt: 0.0,
            zoom: 0.0,
            max_pan_speed: 60.0,
            max_tilt_speed: 40.0,
            max_zoom_speed: 0.5,
            presets: std::collections::HashMap::new(),
        }
    }

    /// Processes a PTZ command and returns the new position.
    pub fn process_command(&mut self, msg: &PtzMessage, delta_time: f32) -> (f32, f32, f32) {
        match msg.command {
            PtzCommand::PanLeft | PtzCommand::PanRight => {
                let speed = f32::from(msg.pan_speed) / 100.0 * self.max_pan_speed;
                self.pan = (self.pan + speed * delta_time).clamp(-180.0, 180.0);
            }
            PtzCommand::TiltUp | PtzCommand::TiltDown => {
                let speed = f32::from(msg.tilt_speed) / 100.0 * self.max_tilt_speed;
                self.tilt = (self.tilt + speed * delta_time).clamp(-90.0, 90.0);
            }
            PtzCommand::ZoomIn | PtzCommand::ZoomOut => {
                let speed = f32::from(msg.zoom_speed) / 100.0 * self.max_zoom_speed;
                self.zoom = (self.zoom + speed * delta_time).clamp(0.0, 1.0);
            }
            PtzCommand::Stop => {
                // Do nothing, position stays the same
            }
            PtzCommand::GotoPreset => {
                if let Some(&(pan, tilt, zoom)) = self.presets.get(&msg.preset) {
                    self.pan = pan;
                    self.tilt = tilt;
                    self.zoom = zoom;
                }
            }
            PtzCommand::SavePreset => {
                self.presets
                    .insert(msg.preset, (self.pan, self.tilt, self.zoom));
            }
            PtzCommand::AbsolutePosition => {
                self.pan = msg.pan_position.clamp(-180.0, 180.0);
                self.tilt = msg.tilt_position.clamp(-90.0, 90.0);
                self.zoom = msg.zoom_position.clamp(0.0, 1.0);
            }
            PtzCommand::Home => {
                self.pan = 0.0;
                self.tilt = 0.0;
                self.zoom = 0.0;
            }
            PtzCommand::FocusNear | PtzCommand::FocusFar | PtzCommand::AutoFocus => {
                // Focus control would be handled separately
            }
        }

        (self.pan, self.tilt, self.zoom)
    }

    /// Returns the current position.
    #[must_use]
    pub const fn position(&self) -> (f32, f32, f32) {
        (self.pan, self.tilt, self.zoom)
    }

    /// Sets the current position.
    pub fn set_position(&mut self, pan: f32, tilt: f32, zoom: f32) {
        self.pan = pan.clamp(-180.0, 180.0);
        self.tilt = tilt.clamp(-90.0, 90.0);
        self.zoom = zoom.clamp(0.0, 1.0);
    }
}

impl Default for PtzController {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ptz_command_conversion() {
        assert_eq!(PtzCommand::PanLeft.to_u8(), 0);
        assert_eq!(PtzCommand::from_u8(0), Some(PtzCommand::PanLeft));
        assert_eq!(PtzCommand::Stop.to_u8(), 9);
        assert_eq!(PtzCommand::from_u8(9), Some(PtzCommand::Stop));
    }

    #[test]
    fn test_ptz_message_creation() {
        let msg = PtzMessage::new(1, PtzCommand::PanLeft)
            .with_pan_speed(50)
            .with_tilt_speed(25);

        assert_eq!(msg.source_id, 1);
        assert_eq!(msg.command, PtzCommand::PanLeft);
        assert_eq!(msg.pan_speed, 50);
        assert_eq!(msg.tilt_speed, 25);
    }

    #[test]
    fn test_ptz_message_encode_decode() {
        let msg = PtzMessage::new(42, PtzCommand::AbsolutePosition).with_position(45.0, -30.0, 0.5);

        let encoded = msg.encode();
        let decoded = PtzMessage::decode(&encoded).expect("should succeed in test");

        assert_eq!(decoded.source_id, 42);
        assert_eq!(decoded.command, PtzCommand::AbsolutePosition);
        assert!((decoded.pan_position - 45.0).abs() < 0.001);
        assert!((decoded.tilt_position - (-30.0)).abs() < 0.001);
        assert!((decoded.zoom_position - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_ptz_controller_creation() {
        let controller = PtzController::new();
        assert_eq!(controller.position(), (0.0, 0.0, 0.0));
    }

    #[test]
    fn test_ptz_controller_absolute_position() {
        let mut controller = PtzController::new();
        let msg = PtzMessage::new(1, PtzCommand::AbsolutePosition).with_position(90.0, 45.0, 0.75);

        controller.process_command(&msg, 0.1);
        let (pan, tilt, zoom) = controller.position();

        assert!((pan - 90.0).abs() < 0.001);
        assert!((tilt - 45.0).abs() < 0.001);
        assert!((zoom - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_ptz_controller_home() {
        let mut controller = PtzController::new();
        controller.set_position(100.0, 50.0, 0.8);

        let msg = PtzMessage::new(1, PtzCommand::Home);
        controller.process_command(&msg, 0.1);

        assert_eq!(controller.position(), (0.0, 0.0, 0.0));
    }

    #[test]
    fn test_ptz_controller_presets() {
        let mut controller = PtzController::new();

        // Set a position and save it as preset 1
        controller.set_position(45.0, 30.0, 0.5);
        let save_msg = PtzMessage::new(1, PtzCommand::SavePreset).with_preset(1);
        controller.process_command(&save_msg, 0.1);

        // Move to a different position
        controller.set_position(0.0, 0.0, 0.0);

        // Recall preset 1
        let goto_msg = PtzMessage::new(1, PtzCommand::GotoPreset).with_preset(1);
        controller.process_command(&goto_msg, 0.1);

        let (pan, tilt, zoom) = controller.position();
        assert!((pan - 45.0).abs() < 0.001);
        assert!((tilt - 30.0).abs() < 0.001);
        assert!((zoom - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_ptz_controller_clamping() {
        let mut controller = PtzController::new();

        // Try to set position beyond limits
        controller.set_position(200.0, 100.0, 1.5);

        let (pan, tilt, zoom) = controller.position();
        assert_eq!(pan, 180.0);
        assert_eq!(tilt, 90.0);
        assert_eq!(zoom, 1.0);
    }

    #[test]
    fn test_ptz_message_preset() {
        let msg = PtzMessage::new(5, PtzCommand::GotoPreset).with_preset(10);
        assert_eq!(msg.preset, 10);

        let encoded = msg.encode();
        let decoded = PtzMessage::decode(&encoded).expect("should succeed in test");
        assert_eq!(decoded.preset, 10);
    }
}
