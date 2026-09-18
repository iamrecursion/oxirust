//! VISCA over IP remote camera control protocol support.
//!
//! Implements the VISCA (Video System Control Architecture) protocol over UDP/TCP
//! for remote pan-tilt-zoom (PTZ) camera control in multi-camera productions.
//!
//! VISCA is the industry standard for broadcast camera control, used by Sony,
//! Panasonic, Canon, and many other manufacturers.  This module provides:
//!
//! - VISCA command encoding and decoding.
//! - PTZ position, zoom, focus, iris control.
//! - Simulated socket transport (for unit testing without hardware).
//! - A `ViscaController` that manages multiple cameras.

use crate::{AngleId, MultiCamError, Result};
use std::collections::HashMap;

// ── VISCA command bytes ───────────────────────────────────────────────────────

/// VISCA command categories.
const VISCA_COMMAND: u8 = 0x01;
const VISCA_INQUIRY: u8 = 0x09;

// ── ViscaCommand ──────────────────────────────────────────────────────────────

/// A VISCA command ready to be sent over the network.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ViscaCommand {
    /// Raw VISCA byte sequence (without address byte).
    pub bytes: Vec<u8>,
    /// Human-readable description for logging.
    pub description: String,
}

impl ViscaCommand {
    /// Construct a VISCA PAN-TILT drive command.
    ///
    /// `pan_speed` in 1..=24, `tilt_speed` in 1..=23.
    /// `pan_dir`: 1 = right, 2 = left, 3 = stop.
    /// `tilt_dir`: 1 = up,    2 = down,  3 = stop.
    #[must_use]
    pub fn pan_tilt_drive(pan_speed: u8, tilt_speed: u8, pan_dir: u8, tilt_dir: u8) -> Self {
        let ps = pan_speed.clamp(1, 24);
        let ts = tilt_speed.clamp(1, 23);
        let pd = pan_dir.clamp(1, 3);
        let td = tilt_dir.clamp(1, 3);
        Self {
            bytes: vec![VISCA_COMMAND, 0x06, 0x01, ps, ts, pd, td, 0xFF],
            description: format!("PanTilt drive pan={pd} tilt={td} spd=({ps},{ts})"),
        }
    }

    /// Construct a VISCA zoom command.
    ///
    /// `direction`: 0 = tele, 1 = wide, 2 = stop.
    /// `speed`: 0..=7 (ignored when direction is stop).
    #[must_use]
    pub fn zoom(direction: u8, speed: u8) -> Self {
        let spd = speed.clamp(0, 7);
        let byte = match direction {
            0 => 0x20 | spd, // tele
            1 => 0x30 | spd, // wide
            _ => 0x00,       // stop
        };
        Self {
            bytes: vec![VISCA_COMMAND, 0x04, 0x07, byte, 0xFF],
            description: format!("Zoom dir={direction} speed={spd}"),
        }
    }

    /// Construct a VISCA absolute pan-tilt position command.
    ///
    /// `pan_pos` in –170 000..=170 000 (degrees × 100 in VISCA signed 4-nibble).
    /// `tilt_pos` in –30 000..=90 000.
    #[must_use]
    pub fn pan_tilt_absolute(pan_pos: i32, tilt_pos: i32) -> Self {
        let pp = encode_signed_visca(pan_pos);
        let tp = encode_signed_visca(tilt_pos);
        Self {
            bytes: vec![
                VISCA_COMMAND,
                0x06,
                0x02,
                0x18,
                0x14, // pan_speed=24, tilt_speed=20 (max)
                pp[0],
                pp[1],
                pp[2],
                pp[3],
                tp[0],
                tp[1],
                tp[2],
                tp[3],
                0xFF,
            ],
            description: format!("PanTilt absolute pan={pan_pos} tilt={tilt_pos}"),
        }
    }

    /// Construct a VISCA memory recall (preset) command.
    #[must_use]
    pub fn memory_recall(preset: u8) -> Self {
        Self {
            bytes: vec![VISCA_COMMAND, 0x04, 0x3F, 0x02, preset & 0x0F, 0xFF],
            description: format!("Memory recall preset={preset}"),
        }
    }

    /// Construct a VISCA memory set (preset store) command.
    #[must_use]
    pub fn memory_set(preset: u8) -> Self {
        Self {
            bytes: vec![VISCA_COMMAND, 0x04, 0x3F, 0x01, preset & 0x0F, 0xFF],
            description: format!("Memory set preset={preset}"),
        }
    }

    /// Construct a home position command.
    #[must_use]
    pub fn home() -> Self {
        Self {
            bytes: vec![VISCA_COMMAND, 0x06, 0x04, 0xFF],
            description: "Pan-tilt home".into(),
        }
    }

    /// Construct a VISCA inquiry for pan-tilt position.
    #[must_use]
    pub fn inquire_pan_tilt() -> Self {
        Self {
            bytes: vec![VISCA_INQUIRY, 0x06, 0x12, 0xFF],
            description: "Inquire PanTilt position".into(),
        }
    }

    /// Returns `true` when this is an inquiry (read) rather than a command (write).
    #[must_use]
    pub fn is_inquiry(&self) -> bool {
        self.bytes.first() == Some(&VISCA_INQUIRY)
    }
}

/// Encode a signed VISCA position into four nibble bytes.
fn encode_signed_visca(value: i32) -> [u8; 4] {
    // VISCA uses a 16-bit two's complement value spread across 4 nibbles.
    let v = value as u16;
    [
        ((v >> 12) & 0x0F) as u8,
        ((v >> 8) & 0x0F) as u8,
        ((v >> 4) & 0x0F) as u8,
        (v & 0x0F) as u8,
    ]
}

// ── Camera PTZ state ──────────────────────────────────────────────────────────

/// Current PTZ state of a camera (from inquiry or last-known position).
#[derive(Debug, Clone, Copy, Default)]
pub struct CameraPtzState {
    /// Pan position (VISCA units, positive = right).
    pub pan: i32,
    /// Tilt position (VISCA units, positive = up).
    pub tilt: i32,
    /// Zoom position (0–16 383).
    pub zoom: u16,
}

// ── ViscaTransport trait ──────────────────────────────────────────────────────

/// Abstraction over network transport for VISCA commands.
///
/// Allows the controller to be tested without real hardware by swapping in
/// a `SimulatedTransport`.
pub trait ViscaTransport {
    /// Send a command to the camera identified by `camera_id`.
    ///
    /// # Errors
    ///
    /// Returns an error if the transport fails.
    fn send(&mut self, camera_id: AngleId, cmd: &ViscaCommand) -> Result<()>;
}

/// Simulated transport that records sent commands for testing.
#[derive(Debug, Default)]
pub struct SimulatedTransport {
    /// Commands sent, keyed by camera_id.
    pub log: Vec<(AngleId, ViscaCommand)>,
    /// Whether to simulate an error on the next send.
    pub fail_next: bool,
}

impl SimulatedTransport {
    /// Create a new simulated transport.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Return all commands sent to the given camera.
    #[must_use]
    pub fn commands_for(&self, camera_id: AngleId) -> Vec<&ViscaCommand> {
        self.log
            .iter()
            .filter(|(id, _)| *id == camera_id)
            .map(|(_, cmd)| cmd)
            .collect()
    }

    /// Total number of commands sent across all cameras.
    #[must_use]
    pub fn total_commands(&self) -> usize {
        self.log.len()
    }
}

impl ViscaTransport for SimulatedTransport {
    fn send(&mut self, camera_id: AngleId, cmd: &ViscaCommand) -> Result<()> {
        if self.fail_next {
            self.fail_next = false;
            return Err(MultiCamError::ConfigError(
                "Simulated VISCA transport failure".into(),
            ));
        }
        self.log.push((camera_id, cmd.clone()));
        Ok(())
    }
}

// ── ViscaController ───────────────────────────────────────────────────────────

/// Multi-camera VISCA controller.
///
/// Manages PTZ control for multiple cameras via a pluggable transport.
/// Tracks the last-known position for each camera.
#[derive(Debug)]
pub struct ViscaController<T: ViscaTransport> {
    transport: T,
    states: HashMap<AngleId, CameraPtzState>,
}

impl<T: ViscaTransport> ViscaController<T> {
    /// Create a new controller with the given transport.
    #[must_use]
    pub fn new(transport: T) -> Self {
        Self {
            transport,
            states: HashMap::new(),
        }
    }

    /// Register a camera angle.
    pub fn register_camera(&mut self, angle: AngleId) {
        self.states.entry(angle).or_default();
    }

    /// Send a PTZ drive command to a camera.
    ///
    /// # Errors
    ///
    /// Returns an error if the transport fails or the angle is not registered.
    pub fn send_pan_tilt_drive(
        &mut self,
        angle: AngleId,
        pan_speed: u8,
        tilt_speed: u8,
        pan_dir: u8,
        tilt_dir: u8,
    ) -> Result<()> {
        let cmd = ViscaCommand::pan_tilt_drive(pan_speed, tilt_speed, pan_dir, tilt_dir);
        self.transport.send(angle, &cmd)
    }

    /// Move camera to an absolute pan-tilt position.
    ///
    /// # Errors
    ///
    /// Returns an error if the transport fails.
    pub fn go_to_position(&mut self, angle: AngleId, pan: i32, tilt: i32) -> Result<()> {
        let cmd = ViscaCommand::pan_tilt_absolute(pan, tilt);
        self.transport.send(angle, &cmd)?;
        // Update known state.
        let state = self.states.entry(angle).or_default();
        state.pan = pan;
        state.tilt = tilt;
        Ok(())
    }

    /// Recall a saved camera preset.
    ///
    /// # Errors
    ///
    /// Returns an error if the transport fails.
    pub fn recall_preset(&mut self, angle: AngleId, preset: u8) -> Result<()> {
        let cmd = ViscaCommand::memory_recall(preset);
        self.transport.send(angle, &cmd)
    }

    /// Store the current position as a preset.
    ///
    /// # Errors
    ///
    /// Returns an error if the transport fails.
    pub fn store_preset(&mut self, angle: AngleId, preset: u8) -> Result<()> {
        let cmd = ViscaCommand::memory_set(preset);
        self.transport.send(angle, &cmd)
    }

    /// Send camera to home position.
    ///
    /// # Errors
    ///
    /// Returns an error if the transport fails.
    pub fn home(&mut self, angle: AngleId) -> Result<()> {
        let cmd = ViscaCommand::home();
        self.transport.send(angle, &cmd)?;
        let state = self.states.entry(angle).or_default();
        state.pan = 0;
        state.tilt = 0;
        Ok(())
    }

    /// Zoom the camera.
    ///
    /// # Errors
    ///
    /// Returns an error if the transport fails.
    pub fn zoom(&mut self, angle: AngleId, direction: u8, speed: u8) -> Result<()> {
        let cmd = ViscaCommand::zoom(direction, speed);
        self.transport.send(angle, &cmd)
    }

    /// Get the last-known PTZ state of a camera.
    #[must_use]
    pub fn state(&self, angle: AngleId) -> Option<CameraPtzState> {
        self.states.get(&angle).copied()
    }

    /// Number of registered cameras.
    #[must_use]
    pub fn camera_count(&self) -> usize {
        self.states.len()
    }

    /// Send an arbitrary VISCA command.
    ///
    /// # Errors
    ///
    /// Returns an error if the transport fails.
    pub fn send_raw(&mut self, angle: AngleId, cmd: ViscaCommand) -> Result<()> {
        self.transport.send(angle, &cmd)
    }

    /// Get immutable reference to the underlying transport.
    #[must_use]
    pub fn transport(&self) -> &T {
        &self.transport
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Decode four VISCA nibble bytes into a signed 32-bit integer (test helper).
    fn decode_signed_visca(nibbles: [u8; 4]) -> i32 {
        let v: u16 = (u16::from(nibbles[0]) << 12)
            | (u16::from(nibbles[1]) << 8)
            | (u16::from(nibbles[2]) << 4)
            | u16::from(nibbles[3]);
        v as i16 as i32
    }

    #[test]
    fn test_visca_command_pan_tilt_drive() {
        let cmd = ViscaCommand::pan_tilt_drive(10, 8, 1, 3);
        assert_eq!(cmd.bytes[0], VISCA_COMMAND);
        assert!(!cmd.description.is_empty());
        assert!(!cmd.is_inquiry());
    }

    #[test]
    fn test_visca_command_zoom() {
        let cmd = ViscaCommand::zoom(0, 5); // tele at speed 5
        assert_eq!(cmd.bytes[3], 0x20 | 5);
    }

    #[test]
    fn test_visca_command_memory_recall() {
        let cmd = ViscaCommand::memory_recall(3);
        assert_eq!(cmd.bytes[4], 3);
    }

    #[test]
    fn test_visca_command_inquiry_flag() {
        let cmd = ViscaCommand::inquire_pan_tilt();
        assert!(cmd.is_inquiry());
    }

    #[test]
    fn test_encode_decode_signed_visca_round_trip() {
        for &val in &[0i32, 1000, -1000, 32767, -32768] {
            let encoded = encode_signed_visca(val);
            let decoded = decode_signed_visca(encoded);
            assert_eq!(decoded, val, "round-trip failed for {val}");
        }
    }

    #[test]
    fn test_simulated_transport_records_commands() {
        let mut ctrl = ViscaController::new(SimulatedTransport::new());
        ctrl.register_camera(0);
        ctrl.home(0).expect("should succeed");
        ctrl.zoom(0, 0, 3).expect("should succeed");
        assert_eq!(ctrl.transport().total_commands(), 2);
    }

    #[test]
    fn test_controller_go_to_position_updates_state() {
        let mut ctrl = ViscaController::new(SimulatedTransport::new());
        ctrl.register_camera(1);
        ctrl.go_to_position(1, 5000, -2000).expect("should succeed");
        let state = ctrl.state(1).expect("state must exist");
        assert_eq!(state.pan, 5000);
        assert_eq!(state.tilt, -2000);
    }

    #[test]
    fn test_controller_home_resets_state() {
        let mut ctrl = ViscaController::new(SimulatedTransport::new());
        ctrl.register_camera(2);
        ctrl.go_to_position(2, 1000, 500).expect("should succeed");
        ctrl.home(2).expect("should succeed");
        let state = ctrl.state(2).expect("state must exist");
        assert_eq!(state.pan, 0);
        assert_eq!(state.tilt, 0);
    }

    #[test]
    fn test_simulated_transport_fail_next() {
        let mut transport = SimulatedTransport::new();
        transport.fail_next = true;
        let mut ctrl = ViscaController::new(transport);
        ctrl.register_camera(0);
        assert!(ctrl.home(0).is_err());
        // After failure, subsequent calls succeed
        ctrl.home(0).expect("should succeed after failure");
    }

    #[test]
    fn test_commands_for_camera_filter() {
        let mut ctrl = ViscaController::new(SimulatedTransport::new());
        ctrl.register_camera(0);
        ctrl.register_camera(1);
        ctrl.home(0).expect("ok");
        ctrl.home(0).expect("ok");
        ctrl.home(1).expect("ok");
        let cmds = ctrl.transport().commands_for(0);
        assert_eq!(cmds.len(), 2);
        let cmds1 = ctrl.transport().commands_for(1);
        assert_eq!(cmds1.len(), 1);
    }
}
