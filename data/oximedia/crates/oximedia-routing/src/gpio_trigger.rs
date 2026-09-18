//! Hardware GPI/O-triggered routing changes.
//!
//! This module provides `GpioTrigger` for binding general-purpose input (GPI)
//! and output (GPO) lines to routing actions. When a GPI line changes state, the
//! associated [`RoutingAction`] fires instantly, enabling hardware-controlled
//! routing (e.g., tally lights, on-air switches, remote source selectors).
//!
//! The module is purely software-side: actual hardware I/O is abstracted via the
//! [`GpioBackend`] trait, allowing implementations for serial, USB-HID, NDI KVM,
//! AES10 protocol, or simulated backends.

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors from GPIO trigger operations.
#[derive(Debug, Error)]
pub enum GpioError {
    /// The specified GPIO line number is out of range.
    #[error("GPIO line {0} out of range (max {1})")]
    LineOutOfRange(u8, u8),
    /// No action is bound to this line/edge combination.
    #[error("no binding for line {line} edge {edge:?}")]
    NoBinding { line: u8, edge: TriggerEdge },
    /// A binding already exists for this line/edge and `replace` was not set.
    #[error("binding conflict on line {line} edge {edge:?}")]
    BindingConflict { line: u8, edge: TriggerEdge },
    /// Backend returned an error.
    #[error("GPIO backend error: {0}")]
    Backend(String),
    /// Action could not be serialized or parsed.
    #[error("action parse error: {0}")]
    ActionParse(String),
}

// ---------------------------------------------------------------------------
// TriggerEdge — rising / falling / both
// ---------------------------------------------------------------------------

/// Which signal edge triggers the action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TriggerEdge {
    /// Trigger when the line goes from low to high (button press, contact close).
    Rising,
    /// Trigger when the line goes from high to low (button release, contact open).
    Falling,
    /// Trigger on any state change.
    Both,
}

impl fmt::Display for TriggerEdge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rising => write!(f, "rising"),
            Self::Falling => write!(f, "falling"),
            Self::Both => write!(f, "both"),
        }
    }
}

// ---------------------------------------------------------------------------
// GpioLineState
// ---------------------------------------------------------------------------

/// Logical state of a single GPIO line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GpioLineState {
    /// Line is at logic low (0 V / open collector released).
    Low,
    /// Line is at logic high (typically 5 V or 3.3 V).
    High,
    /// State is not yet known (e.g., before first sample).
    Unknown,
}

impl GpioLineState {
    /// Returns `true` if the state is [`High`](Self::High).
    pub fn is_high(&self) -> bool {
        matches!(self, Self::High)
    }

    /// Returns `true` if the state is [`Low`](Self::Low).
    pub fn is_low(&self) -> bool {
        matches!(self, Self::Low)
    }
}

impl fmt::Display for GpioLineState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::High => write!(f, "HIGH"),
            Self::Low => write!(f, "LOW"),
            Self::Unknown => write!(f, "UNKNOWN"),
        }
    }
}

// ---------------------------------------------------------------------------
// RoutingAction — what fires when a GPI changes
// ---------------------------------------------------------------------------

/// The routing action associated with a GPIO trigger.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RoutingAction {
    /// Connect the given input to the given output.
    Connect { input: usize, output: usize },
    /// Disconnect the given input from the given output.
    Disconnect { input: usize, output: usize },
    /// Recall a named routing preset.
    RecallPreset { preset_name: String },
    /// Toggle the mute state of a channel.
    ToggleMute { channel: usize },
    /// Set the mute state of a channel explicitly.
    SetMute { channel: usize, muted: bool },
    /// Set the gain of a channel.
    SetGain { channel: usize, gain_db: f32 },
    /// Trigger a failover on the named route group.
    Failover { group: String },
    /// Custom action identified by a string key.
    Custom { key: String, value: String },
}

impl fmt::Display for RoutingAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Connect { input, output } => write!(f, "CONNECT({input}->{output})"),
            Self::Disconnect { input, output } => write!(f, "DISCONNECT({input}->{output})"),
            Self::RecallPreset { preset_name } => write!(f, "PRESET({preset_name})"),
            Self::ToggleMute { channel } => write!(f, "TOGGLE_MUTE(ch{channel})"),
            Self::SetMute { channel, muted } => write!(f, "SET_MUTE(ch{channel}={muted})"),
            Self::SetGain { channel, gain_db } => {
                write!(f, "SET_GAIN(ch{channel}={gain_db:.1}dB)")
            }
            Self::Failover { group } => write!(f, "FAILOVER({group})"),
            Self::Custom { key, value } => write!(f, "CUSTOM({key}={value})"),
        }
    }
}

// ---------------------------------------------------------------------------
// GpioBinding — one line + edge → action mapping
// ---------------------------------------------------------------------------

/// A single GPI binding: a line number, edge, and resulting action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpioBinding {
    /// The GPI line number (0-based).
    pub line: u8,
    /// Which edge triggers this binding.
    pub edge: TriggerEdge,
    /// The routing action to fire.
    pub action: RoutingAction,
    /// Human-readable description.
    pub description: String,
    /// Whether this binding is currently enabled.
    pub enabled: bool,
}

impl GpioBinding {
    /// Creates a new enabled binding.
    pub fn new(
        line: u8,
        edge: TriggerEdge,
        action: RoutingAction,
        description: impl Into<String>,
    ) -> Self {
        Self {
            line,
            edge,
            action,
            description: description.into(),
            enabled: true,
        }
    }
}

// ---------------------------------------------------------------------------
// GpioOutputBinding — routing event → GPO assertion
// ---------------------------------------------------------------------------

/// Maps a routing event to a GPO output assertion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpoBinding {
    /// GPO line number (0-based).
    pub line: u8,
    /// State to assert when the event fires.
    pub assert_state: GpioLineState,
    /// Duration in milliseconds to hold the assertion (0 = permanent).
    pub hold_ms: u32,
    /// Human-readable label.
    pub label: String,
}

impl GpoBinding {
    /// Creates a new GPO binding.
    pub fn new(
        line: u8,
        assert_state: GpioLineState,
        hold_ms: u32,
        label: impl Into<String>,
    ) -> Self {
        Self {
            line,
            assert_state,
            hold_ms,
            label: label.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// GpioEvent — emitted when a trigger fires
// ---------------------------------------------------------------------------

/// An event emitted when a GPI binding fires.
#[derive(Debug, Clone)]
pub struct GpioEvent {
    /// The line that changed.
    pub line: u8,
    /// The new state.
    pub new_state: GpioLineState,
    /// The previous state.
    pub old_state: GpioLineState,
    /// The edge detected.
    pub edge: TriggerEdge,
    /// The action that was (or should be) executed.
    pub action: RoutingAction,
    /// Monotonic event counter.
    pub event_id: u64,
}

// ---------------------------------------------------------------------------
// GpioBackend trait
// ---------------------------------------------------------------------------

/// Abstraction over a physical or simulated GPIO hardware interface.
///
/// Implementations may wrap serial port GPIO expanders, USB-HID devices,
/// or simulated backends for testing.
pub trait GpioBackend {
    /// Reads the current state of line `line`.
    fn read_line(&self, line: u8) -> Result<GpioLineState, GpioError>;
    /// Writes `state` to GPO line `line`.
    fn write_line(&mut self, line: u8, state: GpioLineState) -> Result<(), GpioError>;
    /// Returns the number of available GPI lines.
    fn gpi_count(&self) -> u8;
    /// Returns the number of available GPO lines.
    fn gpo_count(&self) -> u8;
}

// ---------------------------------------------------------------------------
// GpioTriggerManager
// ---------------------------------------------------------------------------

/// Manages GPI bindings and coordinates hardware-triggered routing changes.
///
/// Call [`poll`](Self::poll) periodically (e.g., every frame or every 1 ms)
/// to detect edge transitions and fire the associated actions.
pub struct GpioTriggerManager {
    /// Maximum GPI lines.
    max_gpi: u8,
    /// Maximum GPO lines.
    max_gpo: u8,
    /// GPI bindings keyed by `(line, edge)`.
    bindings: HashMap<(u8, TriggerEdge), GpioBinding>,
    /// GPO bindings keyed by line.
    gpo_bindings: HashMap<u8, GpoBinding>,
    /// Last known state per GPI line.
    line_states: Vec<GpioLineState>,
    /// Pending GPO hold timers: (line, remaining_ms).
    gpo_timers: Vec<(u8, u32)>,
    /// Monotonic event counter.
    event_counter: u64,
    /// Fired events buffer (cleared by caller).
    event_log: Vec<GpioEvent>,
}

impl GpioTriggerManager {
    /// Creates a new manager for a GPIO device with the given line counts.
    pub fn new(max_gpi: u8, max_gpo: u8) -> Self {
        Self {
            max_gpi,
            max_gpo,
            bindings: HashMap::new(),
            gpo_bindings: HashMap::new(),
            line_states: vec![GpioLineState::Unknown; max_gpi as usize],
            gpo_timers: Vec::new(),
            event_counter: 0,
            event_log: Vec::new(),
        }
    }

    /// Binds a GPI line + edge to a routing action.
    ///
    /// If `replace` is `true`, any existing binding for this line+edge is
    /// replaced. Otherwise, returns [`GpioError::BindingConflict`].
    pub fn bind_gpi(&mut self, binding: GpioBinding, replace: bool) -> Result<(), GpioError> {
        if binding.line >= self.max_gpi {
            return Err(GpioError::LineOutOfRange(binding.line, self.max_gpi - 1));
        }
        let key = (binding.line, binding.edge);
        if !replace && self.bindings.contains_key(&key) {
            return Err(GpioError::BindingConflict {
                line: binding.line,
                edge: binding.edge,
            });
        }
        self.bindings.insert(key, binding);
        Ok(())
    }

    /// Binds a GPO line for tally/status assertion.
    pub fn bind_gpo(&mut self, binding: GpoBinding) -> Result<(), GpioError> {
        if binding.line >= self.max_gpo {
            return Err(GpioError::LineOutOfRange(binding.line, self.max_gpo - 1));
        }
        self.gpo_bindings.insert(binding.line, binding);
        Ok(())
    }

    /// Removes a GPI binding for the given line + edge.
    pub fn unbind_gpi(&mut self, line: u8, edge: TriggerEdge) -> bool {
        self.bindings.remove(&(line, edge)).is_some()
    }

    /// Enables or disables a GPI binding.
    pub fn set_binding_enabled(&mut self, line: u8, edge: TriggerEdge, enabled: bool) -> bool {
        if let Some(b) = self.bindings.get_mut(&(line, edge)) {
            b.enabled = enabled;
            true
        } else {
            false
        }
    }

    /// Polls the GPIO backend for state changes and fires any bindings.
    ///
    /// `elapsed_ms` — milliseconds elapsed since the last poll (for GPO timers).
    ///
    /// Returns the list of events fired during this poll cycle. The event log
    /// is cleared each time this method is called.
    pub fn poll<B: GpioBackend>(&mut self, backend: &mut B, elapsed_ms: u32) -> Vec<GpioEvent> {
        self.event_log.clear();

        // Read all GPI lines and detect edges
        for line in 0..self.max_gpi {
            let new_state = match backend.read_line(line) {
                Ok(s) => s,
                Err(_) => continue,
            };

            let old_state = self
                .line_states
                .get(line as usize)
                .copied()
                .unwrap_or(GpioLineState::Unknown);

            // Detect edge
            let edge = match (old_state, new_state) {
                (GpioLineState::Low, GpioLineState::High)
                | (GpioLineState::Unknown, GpioLineState::High) => Some(TriggerEdge::Rising),
                (GpioLineState::High, GpioLineState::Low)
                | (GpioLineState::Unknown, GpioLineState::Low) => Some(TriggerEdge::Falling),
                _ => None,
            };

            if let Some(idx) = (line as usize).checked_sub(0) {
                if idx < self.line_states.len() {
                    self.line_states[idx] = new_state;
                }
            }

            if let Some(edge) = edge {
                // Check for exact edge binding and Both edge binding
                for check_edge in [edge, TriggerEdge::Both] {
                    if let Some(binding) = self.bindings.get(&(line, check_edge)) {
                        if !binding.enabled {
                            continue;
                        }
                        let action = binding.action.clone();
                        self.event_counter += 1;
                        self.event_log.push(GpioEvent {
                            line,
                            new_state,
                            old_state,
                            edge: check_edge,
                            action,
                            event_id: self.event_counter,
                        });
                        break; // Fire only once per line transition
                    }
                }
            }
        }

        // Tick GPO hold timers
        self.gpo_timers.retain_mut(|(line, remaining)| {
            if elapsed_ms >= *remaining {
                // Timer expired — de-assert GPO
                let release_state = if let Some(b) = self.gpo_bindings.get(line) {
                    // Opposite of assert_state
                    match b.assert_state {
                        GpioLineState::High => GpioLineState::Low,
                        GpioLineState::Low => GpioLineState::High,
                        GpioLineState::Unknown => GpioLineState::Low,
                    }
                } else {
                    GpioLineState::Low
                };
                let _ = backend.write_line(*line, release_state);
                false
            } else {
                *remaining -= elapsed_ms;
                true
            }
        });

        self.event_log.clone()
    }

    /// Asserts a GPO line according to its registered binding.
    ///
    /// If the binding has `hold_ms > 0`, a timer is started to de-assert later.
    pub fn assert_gpo<B: GpioBackend>(
        &mut self,
        backend: &mut B,
        line: u8,
    ) -> Result<(), GpioError> {
        let binding = self
            .gpo_bindings
            .get(&line)
            .ok_or(GpioError::NoBinding {
                line,
                edge: TriggerEdge::Rising,
            })?
            .clone();
        backend.write_line(line, binding.assert_state)?;
        if binding.hold_ms > 0 {
            self.gpo_timers.push((line, binding.hold_ms));
        }
        Ok(())
    }

    /// Returns the last known state of a GPI line.
    pub fn line_state(&self, line: u8) -> GpioLineState {
        self.line_states
            .get(line as usize)
            .copied()
            .unwrap_or(GpioLineState::Unknown)
    }

    /// Returns the number of bindings.
    pub fn binding_count(&self) -> usize {
        self.bindings.len()
    }

    /// Returns the total number of events fired since construction.
    pub fn total_events(&self) -> u64 {
        self.event_counter
    }

    /// Returns an iterator over all registered GPI bindings.
    pub fn bindings(&self) -> impl Iterator<Item = &GpioBinding> {
        self.bindings.values()
    }
}

// ---------------------------------------------------------------------------
// SimulatedBackend — for testing
// ---------------------------------------------------------------------------

/// A fully in-memory simulated GPIO backend for unit testing.
#[derive(Debug)]
pub struct SimulatedBackend {
    gpi_states: Vec<GpioLineState>,
    gpo_states: Vec<GpioLineState>,
}

impl SimulatedBackend {
    /// Creates a simulated backend with `gpi_count` inputs and `gpo_count` outputs.
    pub fn new(gpi_count: u8, gpo_count: u8) -> Self {
        Self {
            gpi_states: vec![GpioLineState::Low; gpi_count as usize],
            gpo_states: vec![GpioLineState::Low; gpo_count as usize],
        }
    }

    /// Sets a GPI line to the given state (simulates hardware input).
    pub fn set_gpi(&mut self, line: u8, state: GpioLineState) {
        if let Some(slot) = self.gpi_states.get_mut(line as usize) {
            *slot = state;
        }
    }

    /// Reads the current GPO state for inspection.
    pub fn gpo_state(&self, line: u8) -> GpioLineState {
        self.gpo_states
            .get(line as usize)
            .copied()
            .unwrap_or(GpioLineState::Unknown)
    }
}

impl GpioBackend for SimulatedBackend {
    fn read_line(&self, line: u8) -> Result<GpioLineState, GpioError> {
        self.gpi_states
            .get(line as usize)
            .copied()
            .ok_or_else(|| GpioError::LineOutOfRange(line, self.gpi_states.len() as u8 - 1))
    }

    fn write_line(&mut self, line: u8, state: GpioLineState) -> Result<(), GpioError> {
        let len = self.gpo_states.len() as u8;
        let slot = self
            .gpo_states
            .get_mut(line as usize)
            .ok_or_else(|| GpioError::LineOutOfRange(line, len.saturating_sub(1)))?;
        *slot = state;
        Ok(())
    }

    fn gpi_count(&self) -> u8 {
        self.gpi_states.len() as u8
    }

    fn gpo_count(&self) -> u8 {
        self.gpo_states.len() as u8
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_manager() -> (GpioTriggerManager, SimulatedBackend) {
        let mgr = GpioTriggerManager::new(8, 4);
        let backend = SimulatedBackend::new(8, 4);
        (mgr, backend)
    }

    #[test]
    fn test_bind_gpi_and_count() {
        let (mut mgr, _) = make_manager();
        let binding = GpioBinding::new(
            0,
            TriggerEdge::Rising,
            RoutingAction::Connect {
                input: 0,
                output: 0,
            },
            "cam1 on-air",
        );
        mgr.bind_gpi(binding, false)
            .expect("binding cam1 on-air should succeed");
        assert_eq!(mgr.binding_count(), 1);
    }

    #[test]
    fn test_bind_conflict_rejected() {
        let (mut mgr, _) = make_manager();
        let b1 = GpioBinding::new(
            0,
            TriggerEdge::Rising,
            RoutingAction::Connect {
                input: 0,
                output: 0,
            },
            "b1",
        );
        let b2 = GpioBinding::new(
            0,
            TriggerEdge::Rising,
            RoutingAction::Connect {
                input: 1,
                output: 1,
            },
            "b2",
        );
        mgr.bind_gpi(b1, false).expect("binding b1 should succeed");
        assert!(mgr.bind_gpi(b2, false).is_err());
    }

    #[test]
    fn test_bind_conflict_replace() {
        let (mut mgr, _) = make_manager();
        let b1 = GpioBinding::new(
            0,
            TriggerEdge::Rising,
            RoutingAction::Connect {
                input: 0,
                output: 0,
            },
            "b1",
        );
        let b2 = GpioBinding::new(
            0,
            TriggerEdge::Rising,
            RoutingAction::Connect {
                input: 1,
                output: 1,
            },
            "b2",
        );
        mgr.bind_gpi(b1, false)
            .expect("binding b1 for replace test should succeed");
        mgr.bind_gpi(b2, true)
            .expect("replacing b1 with b2 should succeed");
        assert_eq!(mgr.binding_count(), 1);
    }

    #[test]
    fn test_line_out_of_range() {
        let (mut mgr, _) = make_manager();
        let b = GpioBinding::new(
            99,
            TriggerEdge::Rising,
            RoutingAction::ToggleMute { channel: 0 },
            "out of range",
        );
        assert!(mgr.bind_gpi(b, false).is_err());
    }

    #[test]
    fn test_rising_edge_fires_action() {
        let (mut mgr, mut backend) = make_manager();
        let binding = GpioBinding::new(
            2,
            TriggerEdge::Rising,
            RoutingAction::Connect {
                input: 3,
                output: 5,
            },
            "switch",
        );
        mgr.bind_gpi(binding, false)
            .expect("binding switch should succeed");

        // Line starts low; set high → rising edge
        backend.set_gpi(2, GpioLineState::High);
        let events = mgr.poll(&mut backend, 0);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].line, 2);
        assert_eq!(events[0].edge, TriggerEdge::Rising);
        assert_eq!(
            events[0].action,
            RoutingAction::Connect {
                input: 3,
                output: 5
            }
        );
    }

    #[test]
    fn test_falling_edge_fires_action() {
        let (mut mgr, mut backend) = make_manager();
        // Start high
        backend.set_gpi(1, GpioLineState::High);
        mgr.poll(&mut backend, 0);

        let binding = GpioBinding::new(
            1,
            TriggerEdge::Falling,
            RoutingAction::Disconnect {
                input: 0,
                output: 0,
            },
            "release",
        );
        mgr.bind_gpi(binding, false)
            .expect("binding release should succeed");

        backend.set_gpi(1, GpioLineState::Low);
        let events = mgr.poll(&mut backend, 0);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].edge, TriggerEdge::Falling);
    }

    #[test]
    fn test_no_event_when_no_change() {
        let (mut mgr, mut backend) = make_manager();
        let binding = GpioBinding::new(
            0,
            TriggerEdge::Rising,
            RoutingAction::ToggleMute { channel: 0 },
            "mute",
        );
        mgr.bind_gpi(binding, false)
            .expect("binding mute should succeed");

        // Stable high → no event
        backend.set_gpi(0, GpioLineState::High);
        mgr.poll(&mut backend, 0); // fires rising edge
        let events2 = mgr.poll(&mut backend, 0); // no change
        assert!(events2.is_empty());
    }

    #[test]
    fn test_disabled_binding_does_not_fire() {
        let (mut mgr, mut backend) = make_manager();
        let binding = GpioBinding::new(
            0,
            TriggerEdge::Rising,
            RoutingAction::Failover {
                group: "primary".to_string(),
            },
            "failover",
        );
        mgr.bind_gpi(binding, false)
            .expect("binding failover should succeed");
        mgr.set_binding_enabled(0, TriggerEdge::Rising, false);

        backend.set_gpi(0, GpioLineState::High);
        let events = mgr.poll(&mut backend, 0);
        assert!(events.is_empty());
    }

    #[test]
    fn test_both_edge_binding() {
        let (mut mgr, mut backend) = make_manager();
        let binding = GpioBinding::new(
            0,
            TriggerEdge::Both,
            RoutingAction::ToggleMute { channel: 1 },
            "toggle",
        );
        mgr.bind_gpi(binding, false)
            .expect("binding toggle should succeed");

        // Rising
        backend.set_gpi(0, GpioLineState::High);
        let e1 = mgr.poll(&mut backend, 0);
        assert_eq!(e1.len(), 1);

        // Falling
        backend.set_gpi(0, GpioLineState::Low);
        let e2 = mgr.poll(&mut backend, 0);
        assert_eq!(e2.len(), 1);
    }

    #[test]
    fn test_gpo_binding_assert() {
        let (mut mgr, mut backend) = make_manager();
        let gpo = GpoBinding::new(0, GpioLineState::High, 0, "tally");
        mgr.bind_gpo(gpo).expect("binding tally GPO should succeed");
        mgr.assert_gpo(&mut backend, 0)
            .expect("asserting tally GPO should succeed");
        assert_eq!(backend.gpo_state(0), GpioLineState::High);
    }

    #[test]
    fn test_gpo_hold_timer_expires() {
        let (mut mgr, mut backend) = make_manager();
        let gpo = GpoBinding::new(0, GpioLineState::High, 100, "pulse");
        mgr.bind_gpo(gpo).expect("binding pulse GPO should succeed");
        mgr.assert_gpo(&mut backend, 0)
            .expect("asserting pulse GPO should succeed");
        assert_eq!(backend.gpo_state(0), GpioLineState::High);

        // Elapse 50 ms — timer not yet expired
        mgr.poll(&mut backend, 50);
        assert_eq!(backend.gpo_state(0), GpioLineState::High);

        // Elapse another 60 ms — timer expires
        mgr.poll(&mut backend, 60);
        assert_eq!(backend.gpo_state(0), GpioLineState::Low);
    }

    #[test]
    fn test_unbind_gpi() {
        let (mut mgr, _) = make_manager();
        let b = GpioBinding::new(
            3,
            TriggerEdge::Rising,
            RoutingAction::RecallPreset {
                preset_name: "live".to_string(),
            },
            "recall",
        );
        mgr.bind_gpi(b, false)
            .expect("binding recall should succeed");
        assert!(mgr.unbind_gpi(3, TriggerEdge::Rising));
        assert_eq!(mgr.binding_count(), 0);
    }

    #[test]
    fn test_total_events_counter() {
        let (mut mgr, mut backend) = make_manager();
        let b = GpioBinding::new(
            0,
            TriggerEdge::Rising,
            RoutingAction::SetMute {
                channel: 0,
                muted: true,
            },
            "mute",
        );
        mgr.bind_gpi(b, false)
            .expect("binding mute for counter test should succeed");
        backend.set_gpi(0, GpioLineState::High);
        mgr.poll(&mut backend, 0);
        assert_eq!(mgr.total_events(), 1);
    }

    #[test]
    fn test_line_state_tracking() {
        let (mut mgr, mut backend) = make_manager();
        backend.set_gpi(5, GpioLineState::High);
        mgr.poll(&mut backend, 0);
        assert_eq!(mgr.line_state(5), GpioLineState::High);
    }

    #[test]
    fn test_simulated_backend_gpi_count() {
        let backend = SimulatedBackend::new(16, 8);
        assert_eq!(backend.gpi_count(), 16);
        assert_eq!(backend.gpo_count(), 8);
    }

    #[test]
    fn test_routing_action_display() {
        let a = RoutingAction::Connect {
            input: 1,
            output: 2,
        };
        assert!(a.to_string().contains("CONNECT"));
        let b = RoutingAction::RecallPreset {
            preset_name: "show".to_string(),
        };
        assert!(b.to_string().contains("PRESET"));
    }

    #[test]
    fn test_custom_routing_action() {
        let (mut mgr, mut backend) = make_manager();
        let b = GpioBinding::new(
            0,
            TriggerEdge::Rising,
            RoutingAction::Custom {
                key: "scene".to_string(),
                value: "interview".to_string(),
            },
            "custom",
        );
        mgr.bind_gpi(b, false)
            .expect("binding custom action should succeed");
        backend.set_gpi(0, GpioLineState::High);
        let events = mgr.poll(&mut backend, 0);
        assert_eq!(events.len(), 1);
        if let RoutingAction::Custom { key, value } = &events[0].action {
            assert_eq!(key, "scene");
            assert_eq!(value, "interview");
        } else {
            panic!("expected Custom action");
        }
    }
}
