//! NDI tally bus: program/preview tally routing, state propagation, and tally light control.
//!
//! Manages a virtual tally bus where multiple NDI sources can be assigned
//! program (on-air) or preview tally states, and observers are notified of changes.

#![allow(dead_code)]

use std::collections::HashMap;

/// The tally state of a single NDI source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TallyLight {
    /// Source is neither on-air nor in preview.
    Off,
    /// Source is in preview (not yet on-air).
    Preview,
    /// Source is on program (on-air).
    Program,
    /// Source is both in program and preview simultaneously.
    ProgramAndPreview,
}

impl TallyLight {
    /// Return `true` if the source is on program.
    pub fn is_program(self) -> bool {
        matches!(self, Self::Program | Self::ProgramAndPreview)
    }

    /// Return `true` if the source is on preview.
    pub fn is_preview(self) -> bool {
        matches!(self, Self::Preview | Self::ProgramAndPreview)
    }

    /// Return `true` if any tally is active.
    pub fn is_active(self) -> bool {
        self != Self::Off
    }

    /// Merge two tally states into their logical OR.
    pub fn merge(self, other: Self) -> Self {
        let prog = self.is_program() || other.is_program();
        let prev = self.is_preview() || other.is_preview();
        match (prog, prev) {
            (true, true) => Self::ProgramAndPreview,
            (true, false) => Self::Program,
            (false, true) => Self::Preview,
            (false, false) => Self::Off,
        }
    }
}

/// A snapshot of the full tally bus state.
#[derive(Debug, Clone, Default)]
pub struct TallySnapshot {
    states: HashMap<String, TallyLight>,
}

impl TallySnapshot {
    /// Create a new empty snapshot.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the tally state for `source_name`, defaulting to `Off`.
    pub fn get(&self, source_name: &str) -> TallyLight {
        self.states
            .get(source_name)
            .copied()
            .unwrap_or(TallyLight::Off)
    }

    /// Set the tally state for `source_name`.
    pub fn set(&mut self, source_name: &str, state: TallyLight) {
        self.states.insert(source_name.to_string(), state);
    }

    /// Return the number of sources tracked.
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Return `true` if no sources are tracked.
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    /// Return all sources currently on program.
    pub fn program_sources(&self) -> Vec<&str> {
        self.states
            .iter()
            .filter(|(_, s)| s.is_program())
            .map(|(k, _)| k.as_str())
            .collect()
    }

    /// Return all sources currently in preview.
    pub fn preview_sources(&self) -> Vec<&str> {
        self.states
            .iter()
            .filter(|(_, s)| s.is_preview())
            .map(|(k, _)| k.as_str())
            .collect()
    }
}

/// A tally change event emitted when bus state changes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TallyChangeEvent {
    /// Source whose tally changed.
    pub source: String,
    /// Previous state.
    pub old_state: TallyLight,
    /// New state.
    pub new_state: TallyLight,
}

/// The central tally bus that manages state propagation across NDI sources.
#[derive(Debug)]
pub struct TallyBus {
    snapshot: TallySnapshot,
    event_log: Vec<TallyChangeEvent>,
}

impl TallyBus {
    /// Create a new empty tally bus.
    pub fn new() -> Self {
        Self {
            snapshot: TallySnapshot::new(),
            event_log: Vec::new(),
        }
    }

    /// Set the tally state for a source.
    ///
    /// Records a `TallyChangeEvent` if the state changed.
    pub fn set_tally(&mut self, source: &str, state: TallyLight) {
        let old_state = self.snapshot.get(source);
        if old_state != state {
            self.event_log.push(TallyChangeEvent {
                source: source.to_string(),
                old_state,
                new_state: state,
            });
            self.snapshot.set(source, state);
        }
    }

    /// Set a source as program, clearing program from any previous source.
    pub fn cut_to_program(&mut self, source: &str) {
        // Remove program from all other sources
        let to_update: Vec<(String, TallyLight)> = self
            .snapshot
            .states
            .iter()
            .filter(|(k, s)| k.as_str() != source && s.is_program())
            .map(|(k, &s)| {
                let new = if s.is_preview() {
                    TallyLight::Preview
                } else {
                    TallyLight::Off
                };
                (k.clone(), new)
            })
            .collect();
        for (k, v) in to_update {
            self.set_tally(&k, v);
        }
        self.set_tally(source, TallyLight::Program);
    }

    /// Get the current tally state for a source.
    pub fn get_tally(&self, source: &str) -> TallyLight {
        self.snapshot.get(source)
    }

    /// Get a snapshot of the entire bus state.
    pub fn snapshot(&self) -> &TallySnapshot {
        &self.snapshot
    }

    /// Drain and return all pending change events.
    pub fn drain_events(&mut self) -> Vec<TallyChangeEvent> {
        std::mem::take(&mut self.event_log)
    }

    /// Clear all tally states and the event log.
    pub fn reset(&mut self) {
        self.snapshot = TallySnapshot::new();
        self.event_log.clear();
    }
}

impl Default for TallyBus {
    fn default() -> Self {
        Self::new()
    }
}

/// Encode a `TallyLight` into the NDI wire byte representation.
/// Bit 0 = program, Bit 1 = preview.
pub fn encode_tally_byte(state: TallyLight) -> u8 {
    let prog: u8 = if state.is_program() { 0x01 } else { 0x00 };
    let prev: u8 = if state.is_preview() { 0x02 } else { 0x00 };
    prog | prev
}

/// Decode a tally byte into a `TallyLight`.
pub fn decode_tally_byte(byte: u8) -> TallyLight {
    let prog = (byte & 0x01) != 0;
    let prev = (byte & 0x02) != 0;
    match (prog, prev) {
        (true, true) => TallyLight::ProgramAndPreview,
        (true, false) => TallyLight::Program,
        (false, true) => TallyLight::Preview,
        (false, false) => TallyLight::Off,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tally_light_is_program() {
        assert!(TallyLight::Program.is_program());
        assert!(TallyLight::ProgramAndPreview.is_program());
        assert!(!TallyLight::Preview.is_program());
        assert!(!TallyLight::Off.is_program());
    }

    #[test]
    fn test_tally_light_is_preview() {
        assert!(TallyLight::Preview.is_preview());
        assert!(TallyLight::ProgramAndPreview.is_preview());
        assert!(!TallyLight::Program.is_preview());
        assert!(!TallyLight::Off.is_preview());
    }

    #[test]
    fn test_tally_light_is_active() {
        assert!(!TallyLight::Off.is_active());
        assert!(TallyLight::Preview.is_active());
        assert!(TallyLight::Program.is_active());
        assert!(TallyLight::ProgramAndPreview.is_active());
    }

    #[test]
    fn test_tally_light_merge() {
        assert_eq!(TallyLight::Off.merge(TallyLight::Off), TallyLight::Off);
        assert_eq!(
            TallyLight::Program.merge(TallyLight::Preview),
            TallyLight::ProgramAndPreview
        );
        assert_eq!(
            TallyLight::Off.merge(TallyLight::Program),
            TallyLight::Program
        );
    }

    #[test]
    fn test_snapshot_get_default_off() {
        let snap = TallySnapshot::new();
        assert_eq!(snap.get("unknown"), TallyLight::Off);
    }

    #[test]
    fn test_snapshot_set_and_get() {
        let mut snap = TallySnapshot::new();
        snap.set("Cam1", TallyLight::Program);
        assert_eq!(snap.get("Cam1"), TallyLight::Program);
    }

    #[test]
    fn test_snapshot_program_sources() {
        let mut snap = TallySnapshot::new();
        snap.set("Cam1", TallyLight::Program);
        snap.set("Cam2", TallyLight::Preview);
        snap.set("Cam3", TallyLight::Off);
        let prg = snap.program_sources();
        assert_eq!(prg.len(), 1);
        assert!(prg.contains(&"Cam1"));
    }

    #[test]
    fn test_snapshot_preview_sources() {
        let mut snap = TallySnapshot::new();
        snap.set("Cam1", TallyLight::Program);
        snap.set("Cam2", TallyLight::Preview);
        let prev = snap.preview_sources();
        assert_eq!(prev.len(), 1);
        assert!(prev.contains(&"Cam2"));
    }

    #[test]
    fn test_tally_bus_set_tally_emits_event() {
        let mut bus = TallyBus::new();
        bus.set_tally("Cam1", TallyLight::Program);
        let events = bus.drain_events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].new_state, TallyLight::Program);
    }

    #[test]
    fn test_tally_bus_no_event_on_same_state() {
        let mut bus = TallyBus::new();
        bus.set_tally("Cam1", TallyLight::Program);
        bus.drain_events();
        bus.set_tally("Cam1", TallyLight::Program); // no change
        assert!(bus.drain_events().is_empty());
    }

    #[test]
    fn test_tally_bus_cut_to_program() {
        let mut bus = TallyBus::new();
        bus.set_tally("Cam1", TallyLight::Program);
        bus.set_tally("Cam2", TallyLight::Preview);
        bus.drain_events();
        bus.cut_to_program("Cam2");
        assert_eq!(bus.get_tally("Cam1"), TallyLight::Off);
        assert_eq!(bus.get_tally("Cam2"), TallyLight::Program);
    }

    #[test]
    fn test_tally_bus_reset() {
        let mut bus = TallyBus::new();
        bus.set_tally("Cam1", TallyLight::Program);
        bus.reset();
        assert_eq!(bus.get_tally("Cam1"), TallyLight::Off);
        assert!(bus.drain_events().is_empty());
    }

    #[test]
    fn test_encode_decode_tally_byte_roundtrip() {
        for state in [
            TallyLight::Off,
            TallyLight::Program,
            TallyLight::Preview,
            TallyLight::ProgramAndPreview,
        ] {
            let byte = encode_tally_byte(state);
            let decoded = decode_tally_byte(byte);
            assert_eq!(decoded, state, "roundtrip failed for {state:?}");
        }
    }

    #[test]
    fn test_encode_tally_byte_values() {
        assert_eq!(encode_tally_byte(TallyLight::Off), 0x00);
        assert_eq!(encode_tally_byte(TallyLight::Program), 0x01);
        assert_eq!(encode_tally_byte(TallyLight::Preview), 0x02);
        assert_eq!(encode_tally_byte(TallyLight::ProgramAndPreview), 0x03);
    }

    // ── End-to-end tally propagation: sender → bus → receiver ─────────────────
    //
    // These tests exercise the full path a tally signal travels in NDI:
    //   1. The SENDER (switcher side) decides a source's tally and serialises it
    //      to the NDI wire byte via `encode_tally_byte`.
    //   2. The BUS routes/holds program/preview state for the named source.
    //   3. The RECEIVER reads the bus's wire byte for the source it is watching
    //      and decodes it via `decode_tally_byte`, lighting its tally lamp.
    //
    // A minimal in-process "wire" is modelled by encoding on the sender side and
    // decoding on the receiver side, with the `TallyBus` as the shared routing
    // point between them — this is a true end-to-end path, not a `set_and_get`
    // on a single manager.

    /// Stand-in for the switcher/program side that drives tally onto the bus.
    struct TallySender;

    impl TallySender {
        /// Push a tally state onto the bus for `source`, returning the NDI wire
        /// byte that would be transmitted downstream.
        fn send(bus: &mut TallyBus, source: &str, state: TallyLight) -> u8 {
            bus.set_tally(source, state);
            encode_tally_byte(state)
        }
    }

    /// Stand-in for a downstream device (camera) watching one source's tally.
    struct TallyReceiver {
        watching: String,
        observed: TallyLight,
    }

    impl TallyReceiver {
        fn new(source: &str) -> Self {
            Self {
                watching: source.to_string(),
                observed: TallyLight::Off,
            }
        }

        /// Poll the bus for the watched source, transit the wire byte, and
        /// decode it into the locally-observed tally state.
        fn poll(&mut self, bus: &TallyBus) -> TallyLight {
            let wire_byte = encode_tally_byte(bus.get_tally(&self.watching));
            self.observed = decode_tally_byte(wire_byte);
            self.observed
        }
    }

    #[test]
    fn test_tally_propagation_sender_to_bus_to_receiver_program() {
        let mut bus = TallyBus::new();
        let mut receiver = TallyReceiver::new("Cam1");

        // Before anything is sent, the receiver sees Off.
        assert_eq!(receiver.poll(&bus), TallyLight::Off);

        // Sender puts Cam1 on PROGRAM; the emitted wire byte is the program bit.
        let wire = TallySender::send(&mut bus, "Cam1", TallyLight::Program);
        assert_eq!(wire, 0x01, "program wire byte");

        // Receiver polls the bus and now observes PROGRAM end-to-end.
        let seen = receiver.poll(&bus);
        assert_eq!(seen, TallyLight::Program);
        assert!(seen.is_program());
        assert!(!seen.is_preview());

        // The bus also recorded the change event for any observers.
        let events = bus.drain_events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].source, "Cam1");
        assert_eq!(events[0].old_state, TallyLight::Off);
        assert_eq!(events[0].new_state, TallyLight::Program);

        // A receiver watching a DIFFERENT source must stay Off (no cross-talk).
        let mut other = TallyReceiver::new("Cam2");
        assert_eq!(other.poll(&bus), TallyLight::Off);
    }

    #[test]
    fn test_tally_propagation_preview_then_clear() {
        let mut bus = TallyBus::new();
        let mut receiver = TallyReceiver::new("Cam1");

        // Sender sets PREVIEW on Cam1; receiver observes the green tally.
        let wire = TallySender::send(&mut bus, "Cam1", TallyLight::Preview);
        assert_eq!(wire, 0x02, "preview wire byte");
        let seen = receiver.poll(&bus);
        assert_eq!(seen, TallyLight::Preview);
        assert!(seen.is_preview());
        assert!(!seen.is_program());

        // Now the sender CLEARS the tally (back to Off).
        let cleared_wire = TallySender::send(&mut bus, "Cam1", TallyLight::Off);
        assert_eq!(cleared_wire, 0x00, "off wire byte");
        let cleared = receiver.poll(&bus);
        assert_eq!(cleared, TallyLight::Off, "clear propagated to receiver");
        assert!(!cleared.is_active());

        // The bus logged both transitions: Off→Preview, then Preview→Off.
        let events = bus.drain_events();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].new_state, TallyLight::Preview);
        assert_eq!(events[1].old_state, TallyLight::Preview);
        assert_eq!(events[1].new_state, TallyLight::Off);
    }

    #[test]
    fn test_tally_propagation_cut_reassigns_program_across_receivers() {
        // Two receivers watch two cameras. A program CUT must move the red
        // tally off the old camera and onto the new one, observed end-to-end.
        let mut bus = TallyBus::new();
        let mut rx_a = TallyReceiver::new("CamA");
        let mut rx_b = TallyReceiver::new("CamB");

        // CamA on program, CamB in preview.
        TallySender::send(&mut bus, "CamA", TallyLight::Program);
        TallySender::send(&mut bus, "CamB", TallyLight::Preview);
        assert_eq!(rx_a.poll(&bus), TallyLight::Program);
        assert_eq!(rx_b.poll(&bus), TallyLight::Preview);

        // Take CamB to program; the bus exclusivity clears CamA's program tally.
        bus.cut_to_program("CamB");
        assert_eq!(rx_a.poll(&bus), TallyLight::Off, "old program goes dark");
        assert_eq!(rx_b.poll(&bus), TallyLight::Program, "new source lit");
    }
}
