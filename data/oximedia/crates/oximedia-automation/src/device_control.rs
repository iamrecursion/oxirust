//! High-level broadcast device control automation.
//!
//! Provides types for tracking broadcast devices and issuing commands to them.
//! Note: The lower-level protocol-specific controllers live in `device::control`.

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// GPI Debounce logic
// ─────────────────────────────────────────────────────────────────────────────

/// Debounce configuration for a GPI pin.
#[derive(Debug, Clone)]
pub struct GpiDebounceConfig {
    /// Debounce window in milliseconds.  A second transition on the same pin
    /// within this window is ignored to suppress false triggers caused by
    /// contact bounce on physical relay closures.
    pub window_ms: u64,
}

impl Default for GpiDebounceConfig {
    fn default() -> Self {
        Self {
            // 50 ms is a common hardware debounce target for broadcast GPI
            window_ms: 50,
        }
    }
}

/// State for a single GPI pin used by the debounce logic.
#[derive(Debug, Clone)]
struct PinState {
    /// Last accepted (non-debounced) value: `true` = HIGH, `false` = LOW.
    last_value: bool,
    /// Timestamp (ms) when the last transition was accepted.
    last_accepted_ms: u64,
}

/// GPI debouncer that filters rapid toggling on physical GPI pins.
///
/// On every incoming raw event, call [`GpiDebouncer::accept`].  The method
/// returns `true` only when the transition falls outside the debounce window,
/// meaning the caller should propagate the event; `false` indicates the event
/// should be dropped.
///
/// # Example
///
/// ```rust
/// use oximedia_automation::device_control::{GpiDebouncer, GpiDebounceConfig};
///
/// let mut debouncer = GpiDebouncer::new(GpiDebounceConfig { window_ms: 50 });
///
/// // First HIGH transition at t=0 ms — accepted.
/// assert!(debouncer.accept(0, true, 0));
///
/// // Second HIGH transition at t=20 ms — within the 50 ms window, rejected.
/// assert!(!debouncer.accept(0, true, 20));
///
/// // Transition at t=60 ms — outside the window, accepted.
/// assert!(debouncer.accept(0, true, 60));
/// ```
#[derive(Debug, Default)]
pub struct GpiDebouncer {
    config: GpiDebounceConfig,
    /// Per-pin debounce state keyed by pin number.
    pin_states: HashMap<u8, PinState>,
}

impl GpiDebouncer {
    /// Create a new GPI debouncer with the supplied configuration.
    pub fn new(config: GpiDebounceConfig) -> Self {
        Self {
            config,
            pin_states: HashMap::new(),
        }
    }

    /// Evaluate whether the incoming event on `pin` at `now_ms` should be
    /// accepted or dropped.
    ///
    /// # Arguments
    ///
    /// * `pin`      — Pin number (0–255).
    /// * `value`    — New pin value (`true` = HIGH, `false` = LOW).
    /// * `now_ms`   — Current timestamp in milliseconds.
    ///
    /// # Returns
    ///
    /// `true` if the caller should propagate the event; `false` if the event
    /// is within the debounce window and should be discarded.
    pub fn accept(&mut self, pin: u8, value: bool, now_ms: u64) -> bool {
        match self.pin_states.get(&pin) {
            None => {
                // First-ever event on this pin — always accepted.
                self.pin_states.insert(
                    pin,
                    PinState {
                        last_value: value,
                        last_accepted_ms: now_ms,
                    },
                );
                true
            }
            Some(state) => {
                let elapsed = now_ms.saturating_sub(state.last_accepted_ms);
                let same_value = state.last_value == value;

                // Accept if:
                //   (a) the value changed (regardless of timing), OR
                //   (b) the debounce window has expired for a repeated value.
                if !same_value || elapsed >= self.config.window_ms {
                    self.pin_states.insert(
                        pin,
                        PinState {
                            last_value: value,
                            last_accepted_ms: now_ms,
                        },
                    );
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Reset debounce state for a single pin.
    pub fn reset_pin(&mut self, pin: u8) {
        self.pin_states.remove(&pin);
    }

    /// Reset all pin states.
    pub fn reset_all(&mut self) {
        self.pin_states.clear();
    }

    /// Return the number of pins currently tracked.
    pub fn tracked_pins(&self) -> usize {
        self.pin_states.len()
    }
}

/// Category of a broadcast device.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub enum BroadcastDeviceType {
    /// Video router / matrix switcher.
    Router,
    /// Broadcast monitor.
    Monitor,
    /// Studio or remote camera.
    Camera,
    /// Video/audio recorder.
    Recorder,
    /// Production switcher.
    Switcher,
    /// Playout server.
    Playout,
}

impl BroadcastDeviceType {
    /// Return the typical control protocol name for the device type.
    pub fn typical_protocol(&self) -> &str {
        match self {
            Self::Router => "NVISION",
            Self::Monitor => "SNMP",
            Self::Camera => "VISCA",
            Self::Recorder => "Sony-9pin",
            Self::Switcher => "Ember+",
            Self::Playout => "VDCP",
        }
    }
}

/// A command sent to a broadcast device.
#[derive(Debug, Clone)]
pub struct BroadcastDeviceCommand {
    /// Identifier of the target device.
    pub device_id: String,
    /// Command verb (e.g. `"play"`, `"stop"`, `"take"`).
    pub command: String,
    /// Optional parameters associated with the command.
    pub params: Vec<String>,
    /// Millisecond timestamp when the command was issued.
    pub timestamp_ms: u64,
}

impl BroadcastDeviceCommand {
    /// Create a new device command.
    pub fn new(device_id: &str, command: &str, params: Vec<String>, timestamp_ms: u64) -> Self {
        Self {
            device_id: device_id.to_string(),
            command: command.to_string(),
            params,
            timestamp_ms,
        }
    }

    /// Return `true` if the command carries at least one parameter.
    pub fn has_params(&self) -> bool {
        !self.params.is_empty()
    }
}

/// Observed state of a broadcast device.
#[derive(Debug, Clone)]
pub struct BroadcastDeviceState {
    /// Whether the device is considered online.
    pub online: bool,
    /// Network address of the device.
    pub address: String,
    /// Last time (ms) the device was successfully contacted.
    pub last_seen_ms: u64,
    /// Consecutive error count since last successful contact.
    pub error_count: u32,
}

impl BroadcastDeviceState {
    /// Create a new device state record.
    pub fn new(address: &str, online: bool, last_seen_ms: u64) -> Self {
        Self {
            online,
            address: address.to_string(),
            last_seen_ms,
            error_count: 0,
        }
    }

    /// Return `true` if the device was last seen within `timeout_ms` of `now_ms`.
    pub fn is_reachable(&self, now_ms: u64, timeout_ms: u64) -> bool {
        self.online && now_ms.saturating_sub(self.last_seen_ms) <= timeout_ms
    }
}

/// Registry of broadcast devices and their command history.
#[derive(Debug, Default)]
pub struct BroadcastDeviceController {
    devices: Vec<(String, BroadcastDeviceType, BroadcastDeviceState)>,
    command_log: Vec<BroadcastDeviceCommand>,
}

impl BroadcastDeviceController {
    /// Create a new, empty controller.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a device with the controller.
    ///
    /// If a device with the same `id` already exists it is replaced.
    pub fn register(&mut self, id: &str, device_type: BroadcastDeviceType, address: &str) {
        self.devices.retain(|(did, _, _)| did != id);
        self.devices.push((
            id.to_string(),
            device_type,
            BroadcastDeviceState::new(address, true, 0),
        ));
    }

    /// Issue a command to the device identified by `id`.
    ///
    /// Returns `true` if the device was found, `false` otherwise.
    pub fn send_command(&mut self, id: &str, cmd: &str, params: Vec<String>) -> bool {
        if self.devices.iter().any(|(did, _, _)| did == id) {
            self.command_log
                .push(BroadcastDeviceCommand::new(id, cmd, params, 0));
            true
        } else {
            false
        }
    }

    /// Return references to all devices that are currently marked online.
    pub fn online_devices(&self) -> Vec<(&str, &BroadcastDeviceType)> {
        self.devices
            .iter()
            .filter(|(_, _, state)| state.online)
            .map(|(id, dt, _)| (id.as_str(), dt))
            .collect()
    }

    /// Return the command history for a given device.
    pub fn command_history(&self, device_id: &str) -> Vec<&BroadcastDeviceCommand> {
        self.command_log
            .iter()
            .filter(|c| c.device_id == device_id)
            .collect()
    }

    /// Mark a device offline (simulates a connectivity loss).
    pub fn mark_offline(&mut self, id: &str) {
        for (did, _, state) in &mut self.devices {
            if did == id {
                state.online = false;
            }
        }
    }

    /// Return the number of registered devices.
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_type_typical_protocol_router() {
        assert_eq!(BroadcastDeviceType::Router.typical_protocol(), "NVISION");
    }

    #[test]
    fn test_device_type_typical_protocol_camera() {
        assert_eq!(BroadcastDeviceType::Camera.typical_protocol(), "VISCA");
    }

    #[test]
    fn test_device_type_typical_protocol_recorder() {
        assert_eq!(
            BroadcastDeviceType::Recorder.typical_protocol(),
            "Sony-9pin"
        );
    }

    #[test]
    fn test_device_type_typical_protocol_playout() {
        assert_eq!(BroadcastDeviceType::Playout.typical_protocol(), "VDCP");
    }

    #[test]
    fn test_command_has_params_true() {
        let cmd = BroadcastDeviceCommand::new("dev1", "cue", vec!["clip_42".to_string()], 0);
        assert!(cmd.has_params());
    }

    #[test]
    fn test_command_has_params_false() {
        let cmd = BroadcastDeviceCommand::new("dev1", "stop", vec![], 0);
        assert!(!cmd.has_params());
    }

    #[test]
    fn test_device_state_is_reachable_yes() {
        let state = BroadcastDeviceState::new("192.168.1.1", true, 900);
        assert!(state.is_reachable(1000, 200));
    }

    #[test]
    fn test_device_state_is_reachable_no_timeout() {
        let state = BroadcastDeviceState::new("192.168.1.1", true, 500);
        assert!(!state.is_reachable(1000, 400));
    }

    #[test]
    fn test_device_state_is_reachable_offline() {
        let state = BroadcastDeviceState::new("192.168.1.1", false, 999);
        assert!(!state.is_reachable(1000, 5000));
    }

    #[test]
    fn test_controller_register_and_count() {
        let mut ctrl = BroadcastDeviceController::new();
        ctrl.register("cam1", BroadcastDeviceType::Camera, "10.0.0.1");
        ctrl.register("sw1", BroadcastDeviceType::Switcher, "10.0.0.2");
        assert_eq!(ctrl.device_count(), 2);
    }

    #[test]
    fn test_controller_register_replaces_duplicate() {
        let mut ctrl = BroadcastDeviceController::new();
        ctrl.register("rec1", BroadcastDeviceType::Recorder, "10.0.0.10");
        ctrl.register("rec1", BroadcastDeviceType::Recorder, "10.0.0.11");
        assert_eq!(ctrl.device_count(), 1);
    }

    #[test]
    fn test_controller_send_command_known_device() {
        let mut ctrl = BroadcastDeviceController::new();
        ctrl.register("play1", BroadcastDeviceType::Playout, "10.0.0.5");
        let ok = ctrl.send_command("play1", "play", vec![]);
        assert!(ok);
        assert_eq!(ctrl.command_history("play1").len(), 1);
    }

    #[test]
    fn test_controller_send_command_unknown_device() {
        let mut ctrl = BroadcastDeviceController::new();
        let ok = ctrl.send_command("ghost", "play", vec![]);
        assert!(!ok);
    }

    #[test]
    fn test_controller_online_devices() {
        let mut ctrl = BroadcastDeviceController::new();
        ctrl.register("cam1", BroadcastDeviceType::Camera, "10.0.0.1");
        ctrl.register("mon1", BroadcastDeviceType::Monitor, "10.0.0.2");
        ctrl.mark_offline("mon1");
        let online = ctrl.online_devices();
        assert_eq!(online.len(), 1);
        assert_eq!(online[0].0, "cam1");
    }

    // ── GPI Debouncer Tests ───────────────────────────────────────────────────

    #[test]
    fn test_gpi_debouncer_first_event_accepted() {
        let mut debouncer = GpiDebouncer::new(GpiDebounceConfig { window_ms: 50 });
        assert!(debouncer.accept(0, true, 0), "First event must be accepted");
        assert_eq!(debouncer.tracked_pins(), 1);
    }

    #[test]
    fn test_gpi_debouncer_rapid_same_value_rejected() {
        let mut debouncer = GpiDebouncer::new(GpiDebounceConfig { window_ms: 50 });
        assert!(debouncer.accept(1, true, 0));
        // Same value within window → debounce, reject
        assert!(
            !debouncer.accept(1, true, 20),
            "Rapid repeat should be rejected"
        );
    }

    #[test]
    fn test_gpi_debouncer_after_window_accepted() {
        let mut debouncer = GpiDebouncer::new(GpiDebounceConfig { window_ms: 50 });
        assert!(debouncer.accept(2, true, 0));
        // Outside the debounce window → accept
        assert!(
            debouncer.accept(2, true, 60),
            "After window should be accepted"
        );
    }

    #[test]
    fn test_gpi_debouncer_value_change_always_accepted() {
        let mut debouncer = GpiDebouncer::new(GpiDebounceConfig { window_ms: 50 });
        assert!(debouncer.accept(3, true, 0));
        // Value changed → always accepted regardless of timing
        assert!(
            debouncer.accept(3, false, 10),
            "Value change should always be accepted"
        );
    }

    #[test]
    fn test_gpi_debouncer_multiple_pins_independent() {
        let mut debouncer = GpiDebouncer::new(GpiDebounceConfig { window_ms: 50 });
        assert!(debouncer.accept(0, true, 0));
        assert!(debouncer.accept(1, true, 0)); // Different pin — independent state
        assert_eq!(debouncer.tracked_pins(), 2);
    }

    #[test]
    fn test_gpi_debouncer_reset_pin() {
        let mut debouncer = GpiDebouncer::new(GpiDebounceConfig { window_ms: 50 });
        debouncer.accept(5, true, 0);
        debouncer.reset_pin(5);
        // After reset, the next event on pin 5 should be accepted as first-ever
        assert!(
            debouncer.accept(5, true, 10),
            "After reset, first event should be accepted"
        );
    }

    #[test]
    fn test_gpi_debouncer_reset_all() {
        let mut debouncer = GpiDebouncer::new(GpiDebounceConfig { window_ms: 50 });
        debouncer.accept(0, true, 0);
        debouncer.accept(1, false, 0);
        assert_eq!(debouncer.tracked_pins(), 2);
        debouncer.reset_all();
        assert_eq!(debouncer.tracked_pins(), 0);
    }

    #[test]
    fn test_gpi_debouncer_exact_boundary_accepted() {
        let mut debouncer = GpiDebouncer::new(GpiDebounceConfig { window_ms: 50 });
        assert!(debouncer.accept(7, true, 0));
        // Exactly at the boundary (elapsed == window_ms) → accepted
        assert!(
            debouncer.accept(7, true, 50),
            "Event at exact window boundary should be accepted"
        );
    }
}
