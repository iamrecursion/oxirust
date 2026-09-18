//! MIDI control surface integration for the audio mixer.
//!
//! Provides pure-Rust MIDI CC message parsing and mapping of CC numbers
//! to mixer parameters (volume, pan, send levels, mute, solo).
//!
//! # MIDI CC Protocol
//!
//! Control Change (CC) messages consist of three bytes:
//! - Status byte: `0xBn` where `n` is the MIDI channel (0-15)
//! - Data byte 1: CC number (0-127)
//! - Data byte 2: CC value (0-127)
//!
//! This module supports running status (repeated status byte omission)
//! and silently ignores non-CC MIDI messages.

use crate::bus::BusId;
use crate::channel::ChannelId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// MIDI CC Event
// ---------------------------------------------------------------------------

/// A raw MIDI Control Change event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MidiCcEvent {
    /// MIDI channel number (0-15).
    pub channel: u8,
    /// CC number (0-127).
    pub cc: u8,
    /// CC value (0-127).
    pub value: u8,
}

impl MidiCcEvent {
    /// Create a new CC event.
    #[must_use]
    pub fn new(channel: u8, cc: u8, value: u8) -> Self {
        Self {
            channel: channel & 0x0F,
            cc: cc & 0x7F,
            value: value & 0x7F,
        }
    }
}

// ---------------------------------------------------------------------------
// Mapping target
// ---------------------------------------------------------------------------

/// The mixer parameter that a MIDI CC number maps to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MidiMappingTarget {
    /// Channel fader volume.
    Volume(ChannelId),
    /// Channel pan position.
    Pan(ChannelId),
    /// Aux send level for a channel.
    AuxSend {
        /// The channel whose send level is being controlled.
        channel_id: ChannelId,
        /// Send index (0-based).
        send_index: u8,
    },
    /// Master bus volume.
    MasterVolume,
    /// Master bus pan.
    MasterPan,
    /// Bus fader volume.
    BusVolume(BusId),
    /// Toggle mute state for a channel.
    Mute(ChannelId),
    /// Toggle solo state for a channel.
    Solo(ChannelId),
}

// ---------------------------------------------------------------------------
// Mapping
// ---------------------------------------------------------------------------

/// A single CC-to-parameter mapping.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MidiMapping {
    /// CC number this mapping responds to (0-127).
    pub cc: u8,
    /// MIDI channel this mapping responds to (0-15).
    pub midi_channel: u8,
    /// The mixer parameter to control.
    pub target: MidiMappingTarget,
}

impl MidiMapping {
    /// Create a new MIDI mapping.
    #[must_use]
    pub fn new(cc: u8, midi_channel: u8, target: MidiMappingTarget) -> Self {
        Self {
            cc: cc & 0x7F,
            midi_channel: midi_channel & 0x0F,
            target,
        }
    }
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for a MIDI control surface.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MidiControlConfig {
    /// Human-readable name for this control surface.
    pub name: String,
    /// CC mappings for this surface.
    pub mappings: Vec<MidiMapping>,
}

impl MidiControlConfig {
    /// Create a new config with the given name and no mappings.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            mappings: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Action
// ---------------------------------------------------------------------------

/// An action derived from a MIDI CC event via a mapping.
#[derive(Debug, Clone)]
pub enum MidiAction {
    /// Set a volume or send level (0.0 = silence, 1.0 = unity).
    SetVolume {
        /// Which parameter to update.
        target: MidiMappingTarget,
        /// Normalised value in `[0.0, 1.0]`.
        value: f32,
    },
    /// Set a pan position (-1.0 = hard left, 0.0 = centre, 1.0 = hard right).
    SetPan {
        /// Which parameter to update.
        target: MidiMappingTarget,
        /// Pan value in `[-1.0, 1.0]`.
        value: f32,
    },
    /// Toggle mute state for a channel.
    ToggleMute(ChannelId),
    /// Toggle solo state for a channel.
    ToggleSolo(ChannelId),
}

// ---------------------------------------------------------------------------
// Control surface
// ---------------------------------------------------------------------------

/// A lookup key for the mapping index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct MappingKey {
    midi_channel: u8,
    cc: u8,
}

/// MIDI control surface: accepts raw MIDI bytes, parses CC messages, and
/// resolves them to mixer actions via a configurable mapping table.
pub struct MidiControlSurface {
    config: MidiControlConfig,
    /// Fast lookup: (midi_channel, cc) → index into `config.mappings`.
    index: HashMap<MappingKey, usize>,
}

impl MidiControlSurface {
    /// Create a new control surface from the given configuration.
    #[must_use]
    pub fn new(config: MidiControlConfig) -> Self {
        let mut index = HashMap::with_capacity(config.mappings.len());
        for (i, m) in config.mappings.iter().enumerate() {
            let key = MappingKey {
                midi_channel: m.midi_channel,
                cc: m.cc,
            };
            index.insert(key, i);
        }
        Self { config, index }
    }

    // -----------------------------------------------------------------------
    // Mapping management
    // -----------------------------------------------------------------------

    /// Add a new mapping.  If a mapping for the same (cc, midi_channel) pair
    /// already exists it is replaced.
    pub fn add_mapping(&mut self, mapping: MidiMapping) {
        let key = MappingKey {
            midi_channel: mapping.midi_channel,
            cc: mapping.cc,
        };
        if let Some(&idx) = self.index.get(&key) {
            self.config.mappings[idx] = mapping;
        } else {
            let idx = self.config.mappings.len();
            self.config.mappings.push(mapping);
            self.index.insert(key, idx);
        }
    }

    /// Remove the mapping for the given (cc, midi_channel) pair, if any.
    pub fn remove_mapping(&mut self, cc: u8, midi_channel: u8) {
        let key = MappingKey {
            midi_channel: midi_channel & 0x0F,
            cc: cc & 0x7F,
        };
        if let Some(idx) = self.index.remove(&key) {
            self.config.mappings.remove(idx);
            // Re-index mappings after the removed position.
            self.index.retain(|_, v| *v != idx);
            for v in self.index.values_mut() {
                if *v > idx {
                    *v -= 1;
                }
            }
        }
    }

    /// Return a reference to the current configuration.
    #[must_use]
    pub fn config(&self) -> &MidiControlConfig {
        &self.config
    }

    // -----------------------------------------------------------------------
    // MIDI byte parsing
    // -----------------------------------------------------------------------

    /// Parse a slice of raw MIDI bytes and return every CC event found.
    ///
    /// Handles running status: if a data byte arrives without a preceding
    /// status byte the last seen status byte is reused.  Non-CC messages
    /// (note-on/off, program change, SysEx, etc.) are silently skipped but
    /// do update the running-status register.
    #[must_use]
    pub fn parse_midi_bytes(bytes: &[u8]) -> Vec<MidiCcEvent> {
        let mut events = Vec::new();
        let mut i = 0;
        // Running-status register: the last status byte seen.
        let mut running_status: Option<u8> = None;

        while i < bytes.len() {
            let byte = bytes[i];

            if byte >= 0x80 {
                // System Real-Time messages (0xF8-0xFF) are single-byte and
                // do NOT update the running-status register.
                if byte >= 0xF8 {
                    i += 1;
                    continue;
                }

                // SysEx: skip until 0xF7.
                if byte == 0xF0 {
                    running_status = None;
                    i += 1;
                    while i < bytes.len() && bytes[i] != 0xF7 {
                        i += 1;
                    }
                    // skip the 0xF7 terminator if present
                    if i < bytes.len() {
                        i += 1;
                    }
                    continue;
                }

                // System Common messages (0xF1-0xF7 except 0xF0/0xF7):
                // clear running status; determine how many data bytes follow.
                if byte >= 0xF1 {
                    running_status = None;
                    // 0xF2 Time Code Quarter Frame has 1 data byte,
                    // 0xF3 Song Select has 1 data byte,
                    // 0xF6 Tune Request has 0.
                    let data_count: usize = match byte {
                        0xF2 => 2,
                        0xF1 | 0xF3 => 1,
                        _ => 0,
                    };
                    i += 1 + data_count;
                    continue;
                }

                // Regular status byte — update running status.
                running_status = Some(byte);
                i += 1;

                // Try to consume a CC message (0xBn).
                let status = byte;
                let msg_type = status & 0xF0;
                let channel = status & 0x0F;

                if msg_type == 0xB0 {
                    // Need two more data bytes.
                    if i + 1 < bytes.len() && bytes[i] < 0x80 && bytes[i + 1] < 0x80 {
                        events.push(MidiCcEvent::new(channel, bytes[i], bytes[i + 1]));
                        i += 2;
                    }
                } else {
                    // Non-CC: skip the correct number of data bytes.
                    let data_count = channel_message_data_bytes(msg_type);
                    // Consume only valid data bytes.
                    let mut consumed = 0;
                    while consumed < data_count && i < bytes.len() && bytes[i] < 0x80 {
                        i += 1;
                        consumed += 1;
                    }
                }
            } else {
                // Data byte without a preceding status byte — running status.
                match running_status {
                    Some(status) if (status & 0xF0) == 0xB0 => {
                        let channel = status & 0x0F;
                        // Need one more data byte.
                        if i + 1 < bytes.len() && bytes[i + 1] < 0x80 {
                            events.push(MidiCcEvent::new(channel, bytes[i], bytes[i + 1]));
                            i += 2;
                        } else {
                            i += 1;
                        }
                    }
                    _ => {
                        // Not a CC running-status context: skip the byte.
                        i += 1;
                    }
                }
            }
        }

        events
    }

    // -----------------------------------------------------------------------
    // Event processing
    // -----------------------------------------------------------------------

    /// Process a single CC event and return the action it maps to, or `None`
    /// if there is no mapping for this (channel, cc) pair.
    #[must_use]
    pub fn process_event(&self, event: &MidiCcEvent) -> Option<MidiAction> {
        let key = MappingKey {
            midi_channel: event.channel,
            cc: event.cc,
        };
        let mapping = self
            .index
            .get(&key)
            .and_then(|&i| self.config.mappings.get(i))?;

        let action = match &mapping.target {
            MidiMappingTarget::Volume(_)
            | MidiMappingTarget::MasterVolume
            | MidiMappingTarget::BusVolume(_)
            | MidiMappingTarget::AuxSend { .. } => MidiAction::SetVolume {
                target: mapping.target.clone(),
                value: Self::cc_to_normalized(event.value),
            },
            MidiMappingTarget::Pan(_) | MidiMappingTarget::MasterPan => MidiAction::SetPan {
                target: mapping.target.clone(),
                value: Self::cc_to_pan(event.value),
            },
            MidiMappingTarget::Mute(ch_id) => MidiAction::ToggleMute(*ch_id),
            MidiMappingTarget::Solo(ch_id) => MidiAction::ToggleSolo(*ch_id),
        };

        Some(action)
    }

    // -----------------------------------------------------------------------
    // Utility conversions
    // -----------------------------------------------------------------------

    /// Map a MIDI CC value (0-127) to a normalised gain in `[0.0, 1.0]`.
    #[must_use]
    #[inline]
    pub fn cc_to_normalized(value: u8) -> f32 {
        (value & 0x7F) as f32 / 127.0
    }

    /// Map a MIDI CC value (0-127) to a pan position in `[-1.0, 1.0]`.
    ///
    /// Value 64 maps to 0.0 (centre).  Values below 64 map to negative
    /// (left) and values above 64 map to positive (right).
    #[must_use]
    #[inline]
    pub fn cc_to_pan(value: u8) -> f32 {
        let v = (value & 0x7F) as f32;
        // Map [0, 127] → [-1.0, 1.0].  Centre at 64.
        ((v - 64.0) / 63.0).clamp(-1.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

/// Return the number of data bytes expected after a channel-message status
/// byte of the given type nibble.
#[inline]
fn channel_message_data_bytes(msg_type: u8) -> usize {
    match msg_type {
        0xC0 | 0xD0 => 1, // Program Change, Channel Pressure
        _ => 2,           // Note On/Off, Poly Pressure, CC, Pitch Bend
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn ch(id: u128) -> ChannelId {
        ChannelId(Uuid::from_u128(id))
    }

    fn bus(id: u128) -> BusId {
        BusId(Uuid::from_u128(id))
    }

    fn surface_with_volume_map() -> MidiControlSurface {
        let ch_id = ch(1);
        let mut config = MidiControlConfig::new("test");
        config
            .mappings
            .push(MidiMapping::new(7, 0, MidiMappingTarget::Volume(ch_id)));
        MidiControlSurface::new(config)
    }

    // --- cc_to_normalized ---

    #[test]
    fn cc_normalized_min() {
        assert_eq!(MidiControlSurface::cc_to_normalized(0), 0.0);
    }

    #[test]
    fn cc_normalized_max() {
        let v = MidiControlSurface::cc_to_normalized(127);
        assert!((v - 1.0).abs() < 1e-6, "expected 1.0 got {v}");
    }

    #[test]
    fn cc_normalized_mid() {
        let v = MidiControlSurface::cc_to_normalized(64);
        let expected = 64.0f32 / 127.0;
        assert!((v - expected).abs() < 1e-6);
    }

    // --- cc_to_pan ---

    #[test]
    fn cc_pan_center() {
        let v = MidiControlSurface::cc_to_pan(64);
        assert!(v.abs() < 0.02, "center should be near 0.0, got {v}");
    }

    #[test]
    fn cc_pan_hard_left() {
        let v = MidiControlSurface::cc_to_pan(0);
        assert!(
            (v - (-1.0)).abs() < 0.02,
            "hard left should be -1.0, got {v}"
        );
    }

    #[test]
    fn cc_pan_hard_right() {
        let v = MidiControlSurface::cc_to_pan(127);
        assert!((v - 1.0).abs() < 0.02, "hard right should be 1.0, got {v}");
    }

    // --- parse_midi_bytes ---

    #[test]
    fn parse_single_cc() {
        let bytes = [0xB0, 7, 100];
        let events = MidiControlSurface::parse_midi_bytes(&bytes);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].channel, 0);
        assert_eq!(events[0].cc, 7);
        assert_eq!(events[0].value, 100);
    }

    #[test]
    fn parse_multi_channel_cc() {
        // CC on channel 3 and channel 15
        let bytes = [0xB3, 10, 50, 0xBF, 11, 90];
        let events = MidiControlSurface::parse_midi_bytes(&bytes);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].channel, 3);
        assert_eq!(events[1].channel, 15);
    }

    #[test]
    fn parse_running_status() {
        // Status 0xB0, then three pairs of data bytes using running status.
        let bytes = [0xB0, 7, 100, 8, 60, 9, 30];
        let events = MidiControlSurface::parse_midi_bytes(&bytes);
        assert_eq!(events.len(), 3);
        assert_eq!(events[0], MidiCcEvent::new(0, 7, 100));
        assert_eq!(events[1], MidiCcEvent::new(0, 8, 60));
        assert_eq!(events[2], MidiCcEvent::new(0, 9, 30));
    }

    #[test]
    fn parse_ignores_note_on() {
        // Note On (0x90) followed by CC
        let bytes = [0x90, 60, 127, 0xB0, 7, 64];
        let events = MidiControlSurface::parse_midi_bytes(&bytes);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].cc, 7);
    }

    #[test]
    fn parse_ignores_program_change() {
        // Program Change (0xC0) has 1 data byte
        let bytes = [0xC0, 5, 0xB0, 7, 64];
        let events = MidiControlSurface::parse_midi_bytes(&bytes);
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn parse_sysex_skipped() {
        let bytes = [0xF0, 0x41, 0x10, 0x42, 0xF7, 0xB0, 7, 80];
        let events = MidiControlSurface::parse_midi_bytes(&bytes);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].value, 80);
    }

    #[test]
    fn parse_empty_bytes() {
        let events = MidiControlSurface::parse_midi_bytes(&[]);
        assert!(events.is_empty());
    }

    #[test]
    fn parse_invalid_no_status() {
        // Raw data bytes with no status — should not panic
        let events = MidiControlSurface::parse_midi_bytes(&[10, 20, 30]);
        assert!(events.is_empty());
    }

    // --- process_event / mapping ---

    #[test]
    fn process_event_volume_mapping() {
        let surface = surface_with_volume_map();
        let event = MidiCcEvent::new(0, 7, 127);
        let action = surface.process_event(&event);
        assert!(action.is_some());
        if let Some(MidiAction::SetVolume { value, .. }) = action {
            assert!((value - 1.0).abs() < 1e-6);
        } else {
            panic!("expected SetVolume");
        }
    }

    #[test]
    fn process_event_no_mapping_returns_none() {
        let surface = surface_with_volume_map();
        let event = MidiCcEvent::new(0, 99, 64); // cc 99 not mapped
        assert!(surface.process_event(&event).is_none());
    }

    #[test]
    fn process_event_pan_mapping() {
        let ch_id = ch(42);
        let mut config = MidiControlConfig::new("pan-test");
        config
            .mappings
            .push(MidiMapping::new(10, 0, MidiMappingTarget::Pan(ch_id)));
        let surface = MidiControlSurface::new(config);
        let event = MidiCcEvent::new(0, 10, 64);
        if let Some(MidiAction::SetPan { value, .. }) = surface.process_event(&event) {
            assert!(value.abs() < 0.02, "center pan should be ~0, got {value}");
        } else {
            panic!("expected SetPan");
        }
    }

    #[test]
    fn process_event_mute_mapping() {
        let ch_id = ch(5);
        let mut config = MidiControlConfig::new("mute-test");
        config
            .mappings
            .push(MidiMapping::new(20, 0, MidiMappingTarget::Mute(ch_id)));
        let surface = MidiControlSurface::new(config);
        let event = MidiCcEvent::new(0, 20, 127);
        assert!(matches!(
            surface.process_event(&event),
            Some(MidiAction::ToggleMute(_))
        ));
    }

    #[test]
    fn add_and_remove_mapping() {
        let mut surface = surface_with_volume_map();
        let bus_id = bus(99);
        surface.add_mapping(MidiMapping::new(
            11,
            0,
            MidiMappingTarget::BusVolume(bus_id),
        ));
        assert!(surface
            .process_event(&MidiCcEvent::new(0, 11, 64))
            .is_some());
        surface.remove_mapping(11, 0);
        assert!(surface
            .process_event(&MidiCcEvent::new(0, 11, 64))
            .is_none());
    }

    #[test]
    fn add_mapping_replaces_existing() {
        let ch_a = ch(1);
        let ch_b = ch(2);
        let mut config = MidiControlConfig::new("replace");
        config
            .mappings
            .push(MidiMapping::new(7, 0, MidiMappingTarget::Volume(ch_a)));
        let mut surface = MidiControlSurface::new(config);
        // Replace cc=7 with Pan mapping
        surface.add_mapping(MidiMapping::new(7, 0, MidiMappingTarget::Pan(ch_b)));
        let event = MidiCcEvent::new(0, 7, 64);
        assert!(matches!(
            surface.process_event(&event),
            Some(MidiAction::SetPan { .. })
        ));
    }
}
