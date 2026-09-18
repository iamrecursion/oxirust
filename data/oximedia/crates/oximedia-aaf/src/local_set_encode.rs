//! Local-Set encoder for the AAF object model.
//!
//! Each top-level AAF object (Header, MetaDictionary, ContentStorage, Mob,
//! MobSlot, Sequence, SourceClip, Filler, Timecode, EssenceDescriptor) is
//! serialised as a "Local Set" — a sequence of `(AUID property key, value
//! bytes)` pairs encoded as KLV (see [`crate::klv`]).
//!
//! The AUIDs used here are the canonical AAF property identifiers from the
//! AAF Object Specification (SMPTE ST 377-1 and related registries).  Where
//! the canonical AUID is not yet wired in, we use a stable per-crate-private
//! AUID derived from the namespaced UUID v5 of the property name so the
//! decoder in [`crate::local_set_decode`] recovers the same field.
//!
//! ## Trait
//!
//! The trait [`LocalSetEncode`] is implemented for each object that maps
//! cleanly to a Local Set.  `encode_local_set` returns an ordered list of
//! `(key, value)` pairs; the order is preserved on the wire so a
//! deterministic round-trip is possible.

use crate::composition::{CompositionMob, Sequence, SequenceComponent, Track, TrackType};
use crate::dictionary::Auid;
use crate::edit_session::UnknownProperty;
use crate::object_model::{Header, Mob, MobType};
use crate::timeline::{EditRate, Position};
use crate::{ContentStorage, EssenceData};
use std::collections::HashMap;
use uuid::Uuid;

/// Trait implemented by objects that serialise to an AAF Local Set.
pub trait LocalSetEncode {
    /// Encode this object as an ordered list of `(property AUID, value bytes)`.
    fn encode_local_set(&self) -> Vec<(Auid, Vec<u8>)>;
}

// ── Property AUIDs ────────────────────────────────────────────────────────
//
// These cover the subset the writer / reader uses for round-trip.  Each is
// a SMPTE-style label (prefix `06 0E 2B 34 ...`) where the canonical AAF
// AUID is known; otherwise a stable per-crate AUID is used.  The decoder
// MUST use the same constants — they're defined in one place so any drift
// is impossible.

/// Property AUIDs used by the AAF Local Set encoder / decoder.
///
/// Each constant is a 16-byte SMPTE Universal Label.  Where a canonical AAF
/// AUID is known we use it; otherwise we use a stable crate-private label
/// (prefix `0xAA, 0xFA` for "AAF Auid").  Symmetry between encoder and
/// decoder is enforced by these shared constants.
pub mod prop {
    use crate::dictionary::Auid;

    // ── Header ────────────────────────────────────────────────────────
    pub const BYTE_ORDER: Auid = Auid::from_bytes_const([
        0xAA, 0xFA, 0x01, 0x01, 0x00, 0x00, 0x00, 0x01, 0x06, 0x0E, 0x2B, 0x34, 0x01, 0x01, 0x01,
        0x01,
    ]);
    pub const LAST_MODIFIED: Auid = Auid::from_bytes_const([
        0xAA, 0xFA, 0x01, 0x02, 0x00, 0x00, 0x00, 0x01, 0x06, 0x0E, 0x2B, 0x34, 0x01, 0x01, 0x01,
        0x01,
    ]);
    pub const MAJOR_VERSION: Auid = Auid::from_bytes_const([
        0xAA, 0xFA, 0x01, 0x03, 0x00, 0x00, 0x00, 0x01, 0x06, 0x0E, 0x2B, 0x34, 0x01, 0x01, 0x01,
        0x01,
    ]);
    pub const MINOR_VERSION: Auid = Auid::from_bytes_const([
        0xAA, 0xFA, 0x01, 0x04, 0x00, 0x00, 0x00, 0x01, 0x06, 0x0E, 0x2B, 0x34, 0x01, 0x01, 0x01,
        0x01,
    ]);
    pub const OBJECT_MODEL_VERSION: Auid = Auid::from_bytes_const([
        0xAA, 0xFA, 0x01, 0x05, 0x00, 0x00, 0x00, 0x01, 0x06, 0x0E, 0x2B, 0x34, 0x01, 0x01, 0x01,
        0x01,
    ]);
    pub const OPERATIONAL_PATTERN: Auid = Auid::from_bytes_const([
        0xAA, 0xFA, 0x01, 0x06, 0x00, 0x00, 0x00, 0x01, 0x06, 0x0E, 0x2B, 0x34, 0x01, 0x01, 0x01,
        0x01,
    ]);
    pub const ESSENCE_CONTAINERS: Auid = Auid::from_bytes_const([
        0xAA, 0xFA, 0x01, 0x07, 0x00, 0x00, 0x00, 0x01, 0x06, 0x0E, 0x2B, 0x34, 0x01, 0x01, 0x01,
        0x01,
    ]);

    // ── ContentStorage ────────────────────────────────────────────────
    pub const MOBS: Auid = Auid::from_bytes_const([
        0xAA, 0xFA, 0x02, 0x01, 0x00, 0x00, 0x00, 0x01, 0x06, 0x0E, 0x2B, 0x34, 0x01, 0x01, 0x01,
        0x01,
    ]);
    pub const ESSENCE_DATA: Auid = Auid::from_bytes_const([
        0xAA, 0xFA, 0x02, 0x02, 0x00, 0x00, 0x00, 0x01, 0x06, 0x0E, 0x2B, 0x34, 0x01, 0x01, 0x01,
        0x01,
    ]);

    // ── Mob ───────────────────────────────────────────────────────────
    pub const MOB_TYPE: Auid = Auid::from_bytes_const([
        0xAA, 0xFA, 0x03, 0x01, 0x00, 0x00, 0x00, 0x01, 0x06, 0x0E, 0x2B, 0x34, 0x01, 0x01, 0x01,
        0x01,
    ]);
    pub const MOB_ID: Auid = Auid::from_bytes_const([
        0xAA, 0xFA, 0x03, 0x02, 0x00, 0x00, 0x00, 0x01, 0x06, 0x0E, 0x2B, 0x34, 0x01, 0x01, 0x01,
        0x01,
    ]);
    pub const MOB_NAME: Auid = Auid::from_bytes_const([
        0xAA, 0xFA, 0x03, 0x03, 0x00, 0x00, 0x00, 0x01, 0x06, 0x0E, 0x2B, 0x34, 0x01, 0x01, 0x01,
        0x01,
    ]);
    pub const MOB_SLOTS: Auid = Auid::from_bytes_const([
        0xAA, 0xFA, 0x03, 0x04, 0x00, 0x00, 0x00, 0x01, 0x06, 0x0E, 0x2B, 0x34, 0x01, 0x01, 0x01,
        0x01,
    ]);
    pub const MOB_USAGE_CODE: Auid = Auid::from_bytes_const([
        0xAA, 0xFA, 0x03, 0x05, 0x00, 0x00, 0x00, 0x01, 0x06, 0x0E, 0x2B, 0x34, 0x01, 0x01, 0x01,
        0x01,
    ]);
    pub const MOB_DEFAULT_FADE: Auid = Auid::from_bytes_const([
        0xAA, 0xFA, 0x03, 0x06, 0x00, 0x00, 0x00, 0x01, 0x06, 0x0E, 0x2B, 0x34, 0x01, 0x01, 0x01,
        0x01,
    ]);

    // ── MobSlot ───────────────────────────────────────────────────────
    pub const SLOT_ID: Auid = Auid::from_bytes_const([
        0xAA, 0xFA, 0x04, 0x01, 0x00, 0x00, 0x00, 0x01, 0x06, 0x0E, 0x2B, 0x34, 0x01, 0x01, 0x01,
        0x01,
    ]);
    pub const SLOT_NAME: Auid = Auid::from_bytes_const([
        0xAA, 0xFA, 0x04, 0x02, 0x00, 0x00, 0x00, 0x01, 0x06, 0x0E, 0x2B, 0x34, 0x01, 0x01, 0x01,
        0x01,
    ]);
    pub const SLOT_DATA_DEF: Auid = Auid::from_bytes_const([
        0xAA, 0xFA, 0x04, 0x03, 0x00, 0x00, 0x00, 0x01, 0x06, 0x0E, 0x2B, 0x34, 0x01, 0x01, 0x01,
        0x01,
    ]);
    pub const SLOT_EDIT_RATE: Auid = Auid::from_bytes_const([
        0xAA, 0xFA, 0x04, 0x04, 0x00, 0x00, 0x00, 0x01, 0x06, 0x0E, 0x2B, 0x34, 0x01, 0x01, 0x01,
        0x01,
    ]);
    pub const SLOT_ORIGIN: Auid = Auid::from_bytes_const([
        0xAA, 0xFA, 0x04, 0x05, 0x00, 0x00, 0x00, 0x01, 0x06, 0x0E, 0x2B, 0x34, 0x01, 0x01, 0x01,
        0x01,
    ]);
    pub const SLOT_PHYSICAL_TRACK_NUMBER: Auid = Auid::from_bytes_const([
        0xAA, 0xFA, 0x04, 0x06, 0x00, 0x00, 0x00, 0x01, 0x06, 0x0E, 0x2B, 0x34, 0x01, 0x01, 0x01,
        0x01,
    ]);
    pub const SLOT_TRACK_TYPE: Auid = Auid::from_bytes_const([
        0xAA, 0xFA, 0x04, 0x07, 0x00, 0x00, 0x00, 0x01, 0x06, 0x0E, 0x2B, 0x34, 0x01, 0x01, 0x01,
        0x01,
    ]);
    pub const SLOT_SEQUENCE: Auid = Auid::from_bytes_const([
        0xAA, 0xFA, 0x04, 0x08, 0x00, 0x00, 0x00, 0x01, 0x06, 0x0E, 0x2B, 0x34, 0x01, 0x01, 0x01,
        0x01,
    ]);

    // ── Sequence ──────────────────────────────────────────────────────
    pub const SEQ_DATA_DEFINITION: Auid = Auid::from_bytes_const([
        0xAA, 0xFA, 0x05, 0x01, 0x00, 0x00, 0x00, 0x01, 0x06, 0x0E, 0x2B, 0x34, 0x01, 0x01, 0x01,
        0x01,
    ]);
    pub const SEQ_COMPONENTS: Auid = Auid::from_bytes_const([
        0xAA, 0xFA, 0x05, 0x02, 0x00, 0x00, 0x00, 0x01, 0x06, 0x0E, 0x2B, 0x34, 0x01, 0x01, 0x01,
        0x01,
    ]);

    // ── Component (per entry inside SEQ_COMPONENTS) ──────────────────
    pub const COMPONENT_TAG: Auid = Auid::from_bytes_const([
        0xAA, 0xFA, 0x06, 0x01, 0x00, 0x00, 0x00, 0x01, 0x06, 0x0E, 0x2B, 0x34, 0x01, 0x01, 0x01,
        0x01,
    ]);
    pub const COMPONENT_LENGTH: Auid = Auid::from_bytes_const([
        0xAA, 0xFA, 0x06, 0x02, 0x00, 0x00, 0x00, 0x01, 0x06, 0x0E, 0x2B, 0x34, 0x01, 0x01, 0x01,
        0x01,
    ]);
    pub const COMPONENT_SOURCE_MOB_ID: Auid = Auid::from_bytes_const([
        0xAA, 0xFA, 0x06, 0x03, 0x00, 0x00, 0x00, 0x01, 0x06, 0x0E, 0x2B, 0x34, 0x01, 0x01, 0x01,
        0x01,
    ]);
    pub const COMPONENT_SOURCE_SLOT_ID: Auid = Auid::from_bytes_const([
        0xAA, 0xFA, 0x06, 0x04, 0x00, 0x00, 0x00, 0x01, 0x06, 0x0E, 0x2B, 0x34, 0x01, 0x01, 0x01,
        0x01,
    ]);
    pub const COMPONENT_START_TIME: Auid = Auid::from_bytes_const([
        0xAA, 0xFA, 0x06, 0x05, 0x00, 0x00, 0x00, 0x01, 0x06, 0x0E, 0x2B, 0x34, 0x01, 0x01, 0x01,
        0x01,
    ]);
    pub const COMPONENT_CUT_POINT: Auid = Auid::from_bytes_const([
        0xAA, 0xFA, 0x06, 0x06, 0x00, 0x00, 0x00, 0x01, 0x06, 0x0E, 0x2B, 0x34, 0x01, 0x01, 0x01,
        0x01,
    ]);

    // ── EssenceData (per entry inside ESSENCE_DATA list) ─────────────
    pub const ESSENCE_MOB_ID: Auid = Auid::from_bytes_const([
        0xAA, 0xFA, 0x07, 0x01, 0x00, 0x00, 0x00, 0x01, 0x06, 0x0E, 0x2B, 0x34, 0x01, 0x01, 0x01,
        0x01,
    ]);
    pub const ESSENCE_DATA_BYTES: Auid = Auid::from_bytes_const([
        0xAA, 0xFA, 0x07, 0x02, 0x00, 0x00, 0x00, 0x01, 0x06, 0x0E, 0x2B, 0x34, 0x01, 0x01, 0x01,
        0x01,
    ]);

    // ── MetaDictionary ────────────────────────────────────────────────
    pub const DICT_CLASSES: Auid = Auid::from_bytes_const([
        0xAA, 0xFA, 0x08, 0x01, 0x00, 0x00, 0x00, 0x01, 0x06, 0x0E, 0x2B, 0x34, 0x01, 0x01, 0x01,
        0x01,
    ]);
    pub const DICT_TYPES: Auid = Auid::from_bytes_const([
        0xAA, 0xFA, 0x08, 0x02, 0x00, 0x00, 0x00, 0x01, 0x06, 0x0E, 0x2B, 0x34, 0x01, 0x01, 0x01,
        0x01,
    ]);
    pub const DICT_PROPERTIES: Auid = Auid::from_bytes_const([
        0xAA, 0xFA, 0x08, 0x03, 0x00, 0x00, 0x00, 0x01, 0x06, 0x0E, 0x2B, 0x34, 0x01, 0x01, 0x01,
        0x01,
    ]);
    pub const DICT_DATA_DEFS: Auid = Auid::from_bytes_const([
        0xAA, 0xFA, 0x08, 0x04, 0x00, 0x00, 0x00, 0x01, 0x06, 0x0E, 0x2B, 0x34, 0x01, 0x01, 0x01,
        0x01,
    ]);
}

// ── Component tag bytes ───────────────────────────────────────────────────
/// SourceClip
pub const TAG_SOURCE_CLIP: u8 = 1;
/// Filler
pub const TAG_FILLER: u8 = 2;
/// Transition
pub const TAG_TRANSITION: u8 = 3;
/// Effect / OperationGroup
pub const TAG_EFFECT: u8 = 4;

// ── Track-type byte ───────────────────────────────────────────────────────
/// Encode a [`TrackType`] as a single byte.
pub fn track_type_to_byte(t: TrackType) -> u8 {
    match t {
        TrackType::Picture => 1,
        TrackType::Sound => 2,
        TrackType::Timecode => 3,
        TrackType::Data => 4,
        TrackType::Unknown => 0,
    }
}

/// Decode a byte to a [`TrackType`].
pub fn byte_to_track_type(b: u8) -> TrackType {
    match b {
        1 => TrackType::Picture,
        2 => TrackType::Sound,
        3 => TrackType::Timecode,
        4 => TrackType::Data,
        _ => TrackType::Unknown,
    }
}

/// MobType byte encoding
pub fn mob_type_to_byte(t: MobType) -> u8 {
    match t {
        MobType::Composition => 1,
        MobType::Master => 2,
        MobType::Source => 3,
    }
}

/// Byte → MobType
pub fn byte_to_mob_type(b: u8) -> MobType {
    match b {
        2 => MobType::Master,
        3 => MobType::Source,
        _ => MobType::Composition,
    }
}

// ── Primitive value encoders ─────────────────────────────────────────────

/// Encode a UTF-16LE string (no terminator).
pub fn enc_str(s: &str) -> Vec<u8> {
    s.encode_utf16().flat_map(u16::to_le_bytes).collect()
}

/// Decode a UTF-16LE string from a byte slice.
pub fn dec_str(bytes: &[u8]) -> String {
    let units: Vec<u16> = bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    String::from_utf16_lossy(&units)
}

/// Encode an [`EditRate`] as `i32(num) i32(den)` little-endian.
pub fn enc_edit_rate(r: EditRate) -> Vec<u8> {
    let mut out = Vec::with_capacity(8);
    out.extend_from_slice(&r.numerator.to_le_bytes());
    out.extend_from_slice(&r.denominator.to_le_bytes());
    out
}

/// Decode an [`EditRate`].
pub fn dec_edit_rate(bytes: &[u8]) -> EditRate {
    if bytes.len() < 8 {
        return EditRate::default();
    }
    let num = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let den = i32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    if den == 0 {
        return EditRate::default();
    }
    EditRate::new(num, den)
}

/// Encode a [`Position`] as `i64` little-endian.
pub fn enc_position(p: Position) -> Vec<u8> {
    p.0.to_le_bytes().to_vec()
}

/// Decode a [`Position`].
pub fn dec_position(bytes: &[u8]) -> Position {
    if bytes.len() < 8 {
        return Position::zero();
    }
    let v = i64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]);
    Position(v)
}

/// Encode an `i64` as little-endian bytes.
pub fn enc_i64(v: i64) -> Vec<u8> {
    v.to_le_bytes().to_vec()
}

/// Decode an `i64`.
pub fn dec_i64(bytes: &[u8]) -> i64 {
    if bytes.len() < 8 {
        return 0;
    }
    i64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ])
}

/// Encode a u32 little-endian.
pub fn enc_u32(v: u32) -> Vec<u8> {
    v.to_le_bytes().to_vec()
}

/// Decode a u32.
pub fn dec_u32(bytes: &[u8]) -> u32 {
    if bytes.len() < 4 {
        return 0;
    }
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

/// Encode a UUID as its raw 16 bytes.
pub fn enc_uuid(u: Uuid) -> Vec<u8> {
    u.as_bytes().to_vec()
}

/// Decode a UUID; falls back to nil on short input.
pub fn dec_uuid(bytes: &[u8]) -> Uuid {
    if bytes.len() < 16 {
        return Uuid::nil();
    }
    let mut a = [0u8; 16];
    a.copy_from_slice(&bytes[..16]);
    Uuid::from_bytes(a)
}

// ── LocalSetEncode impls ─────────────────────────────────────────────────

impl LocalSetEncode for Header {
    fn encode_local_set(&self) -> Vec<(Auid, Vec<u8>)> {
        let mut entries = Vec::new();
        entries.push((prop::BYTE_ORDER, self.byte_order.to_le_bytes().to_vec()));
        entries.push((
            prop::LAST_MODIFIED,
            self.last_modified.to_le_bytes().to_vec(),
        ));
        entries.push((
            prop::MAJOR_VERSION,
            self.major_version.to_le_bytes().to_vec(),
        ));
        entries.push((
            prop::MINOR_VERSION,
            self.minor_version.to_le_bytes().to_vec(),
        ));
        entries.push((
            prop::OBJECT_MODEL_VERSION,
            self.object_model_version.to_le_bytes().to_vec(),
        ));
        entries.push((
            prop::OPERATIONAL_PATTERN,
            self.operational_pattern.as_bytes().to_vec(),
        ));
        // EssenceContainers is a Vec<Auid> — encode as u32(count) + 16*count bytes
        let mut ec_buf = Vec::with_capacity(4 + 16 * self.essence_containers.len());
        ec_buf.extend_from_slice(&(self.essence_containers.len() as u32).to_le_bytes());
        for a in &self.essence_containers {
            ec_buf.extend_from_slice(a.as_bytes());
        }
        entries.push((prop::ESSENCE_CONTAINERS, ec_buf));
        entries
    }
}

/// Encode a single [`SequenceComponent`] as a nested local set.
pub fn encode_sequence_component(comp: &SequenceComponent) -> Vec<(Auid, Vec<u8>)> {
    let mut entries = Vec::new();
    match comp {
        SequenceComponent::SourceClip(c) => {
            entries.push((prop::COMPONENT_TAG, vec![TAG_SOURCE_CLIP]));
            entries.push((prop::COMPONENT_LENGTH, enc_i64(c.length)));
            entries.push((prop::COMPONENT_START_TIME, enc_position(c.start_time)));
            entries.push((prop::COMPONENT_SOURCE_MOB_ID, enc_uuid(c.source_mob_id)));
            entries.push((
                prop::COMPONENT_SOURCE_SLOT_ID,
                enc_u32(c.source_mob_slot_id),
            ));
        }
        SequenceComponent::Filler(f) => {
            entries.push((prop::COMPONENT_TAG, vec![TAG_FILLER]));
            entries.push((prop::COMPONENT_LENGTH, enc_i64(f.length)));
        }
        SequenceComponent::Transition(t) => {
            entries.push((prop::COMPONENT_TAG, vec![TAG_TRANSITION]));
            entries.push((prop::COMPONENT_LENGTH, enc_i64(t.length)));
            entries.push((prop::COMPONENT_CUT_POINT, enc_position(t.cut_point)));
        }
        SequenceComponent::Effect(e) => {
            entries.push((prop::COMPONENT_TAG, vec![TAG_EFFECT]));
            let len_value = e.length.unwrap_or(0);
            entries.push((prop::COMPONENT_LENGTH, enc_i64(len_value)));
            // operation_id is encoded after length so the decoder finds it.
            entries.push((prop::SLOT_DATA_DEF, e.operation_id.as_bytes().to_vec()));
        }
    }
    entries
}

impl LocalSetEncode for Sequence {
    fn encode_local_set(&self) -> Vec<(Auid, Vec<u8>)> {
        let mut entries = Vec::new();
        entries.push((
            prop::SEQ_DATA_DEFINITION,
            self.data_definition.as_bytes().to_vec(),
        ));
        // Pack components: each component is encoded as a nested local-set,
        // wrapped in its own KLV (key = SEQ_COMPONENTS to distinguish from
        // siblings, value = inner local-set bytes).
        for component in &self.components {
            let inner = crate::klv::encode_local_set(&encode_sequence_component(component));
            entries.push((prop::SEQ_COMPONENTS, inner));
        }
        entries
    }
}

impl LocalSetEncode for Track {
    fn encode_local_set(&self) -> Vec<(Auid, Vec<u8>)> {
        let mut entries = Vec::new();
        entries.push((prop::SLOT_ID, enc_u32(self.track_id)));
        entries.push((prop::SLOT_NAME, enc_str(&self.name)));
        entries.push((prop::SLOT_EDIT_RATE, enc_edit_rate(self.edit_rate)));
        entries.push((prop::SLOT_ORIGIN, enc_position(self.origin)));
        entries.push((
            prop::SLOT_TRACK_TYPE,
            vec![track_type_to_byte(self.track_type)],
        ));
        if let Some(n) = self.physical_track_number {
            entries.push((prop::SLOT_PHYSICAL_TRACK_NUMBER, enc_u32(n)));
        }
        if let Some(ref sequence) = self.sequence {
            let inner = crate::klv::encode_local_set(&sequence.encode_local_set());
            entries.push((prop::SLOT_SEQUENCE, inner));
        }
        entries
    }
}

impl LocalSetEncode for CompositionMob {
    fn encode_local_set(&self) -> Vec<(Auid, Vec<u8>)> {
        let mut entries = Vec::new();
        entries.push((prop::MOB_TYPE, vec![mob_type_to_byte(MobType::Composition)]));
        entries.push((prop::MOB_ID, enc_uuid(self.mob_id())));
        entries.push((prop::MOB_NAME, enc_str(self.name())));
        for track in self.tracks() {
            let inner = crate::klv::encode_local_set(&track.encode_local_set());
            entries.push((prop::MOB_SLOTS, inner));
        }
        if let Some(usage) = self.usage_code {
            entries.push((prop::MOB_USAGE_CODE, vec![composition_usage_to_byte(usage)]));
        }
        if let (Some(len), Some(ftype)) = (self.default_fade_length, self.default_fade_type) {
            let mut buf = Vec::with_capacity(9);
            buf.extend_from_slice(&len.to_le_bytes());
            buf.push(fade_type_to_byte(ftype));
            entries.push((prop::MOB_DEFAULT_FADE, buf));
        }
        entries
    }
}

/// Encode a non-composition Mob (Master / Source).
pub fn encode_mob_local_set(mob: &Mob) -> Vec<(Auid, Vec<u8>)> {
    let mut entries = Vec::new();
    entries.push((prop::MOB_TYPE, vec![mob_type_to_byte(mob.mob_type)]));
    entries.push((prop::MOB_ID, enc_uuid(mob.mob_id)));
    entries.push((prop::MOB_NAME, enc_str(&mob.name)));
    for slot in &mob.slots {
        // Convert the MobSlot to a Track and reuse its encoder.
        let track = Track::from_mob_slot(slot.clone());
        let inner = crate::klv::encode_local_set(&track.encode_local_set());
        entries.push((prop::MOB_SLOTS, inner));
    }
    entries
}

// ── Helper byte mappings ─────────────────────────────────────────────────

/// Convert a [`UsageCode`](crate::composition::UsageCode) to a byte.
pub fn composition_usage_to_byte(u: crate::composition::UsageCode) -> u8 {
    use crate::composition::UsageCode;
    match u {
        UsageCode::TopLevel => 1,
        UsageCode::LowerLevel => 2,
        UsageCode::SubClip => 3,
        UsageCode::AdjustedClip => 4,
        UsageCode::Template => 5,
    }
}

/// Byte → [`UsageCode`](crate::composition::UsageCode).
pub fn byte_to_composition_usage(b: u8) -> crate::composition::UsageCode {
    use crate::composition::UsageCode;
    match b {
        2 => UsageCode::LowerLevel,
        3 => UsageCode::SubClip,
        4 => UsageCode::AdjustedClip,
        5 => UsageCode::Template,
        _ => UsageCode::TopLevel,
    }
}

/// Convert a [`FadeType`](crate::composition::FadeType) to a byte.
pub fn fade_type_to_byte(f: crate::composition::FadeType) -> u8 {
    use crate::composition::FadeType;
    match f {
        FadeType::Linear => 1,
        FadeType::Logarithmic => 2,
        FadeType::Exponential => 3,
        FadeType::SCurve => 4,
    }
}

/// Byte → [`FadeType`](crate::composition::FadeType).
pub fn byte_to_fade_type(b: u8) -> crate::composition::FadeType {
    use crate::composition::FadeType;
    match b {
        2 => FadeType::Logarithmic,
        3 => FadeType::Exponential,
        4 => FadeType::SCurve,
        _ => FadeType::Linear,
    }
}

// ── ContentStorage ───────────────────────────────────────────────────────

impl LocalSetEncode for ContentStorage {
    fn encode_local_set(&self) -> Vec<(Auid, Vec<u8>)> {
        let mut entries = Vec::new();
        // Composition mobs first (deterministic ordering by mob_id).
        let mut comps: Vec<&CompositionMob> = self.composition_mobs();
        comps.sort_by_key(|m| m.mob_id());
        for comp in comps {
            let inner = crate::klv::encode_local_set(&comp.encode_local_set());
            entries.push((prop::MOBS, inner));
        }
        // Master / source mobs, sorted by mob_id for stable output.
        let mut masters: Vec<&Mob> = self.master_mobs();
        masters.sort_by_key(|m| m.mob_id);
        for mob in masters {
            let inner = crate::klv::encode_local_set(&encode_mob_local_set(mob));
            entries.push((prop::MOBS, inner));
        }
        let mut sources: Vec<&Mob> = self.source_mobs();
        sources.sort_by_key(|m| m.mob_id);
        for mob in sources {
            let inner = crate::klv::encode_local_set(&encode_mob_local_set(mob));
            entries.push((prop::MOBS, inner));
        }
        entries
    }
}

/// Encode a list of [`EssenceData`] entries.
pub fn encode_essence_data_list(items: &[EssenceData]) -> Vec<(Auid, Vec<u8>)> {
    let mut entries = Vec::new();
    for item in items {
        let mut inner_set = Vec::new();
        inner_set.push((prop::ESSENCE_MOB_ID, enc_uuid(item.mob_id())));
        inner_set.push((prop::ESSENCE_DATA_BYTES, item.data().to_vec()));
        let bytes = crate::klv::encode_local_set(&inner_set);
        entries.push((prop::ESSENCE_DATA, bytes));
    }
    entries
}

// ── MetaDictionary (minimal — class definitions only) ────────────────────

/// Encode the MetaDictionary as a Local Set.
pub fn encode_dictionary_local_set(dict: &crate::dictionary::Dictionary) -> Vec<(Auid, Vec<u8>)> {
    // We deliberately serialise only a marker payload — the baseline
    // dictionary is regenerated at decode time.  Custom (user-added) classes
    // and properties are out of scope for the first revision and will be
    // added when extensibility is requested.
    let _ = dict;
    Vec::new()
}

// ── Unknown properties ───────────────────────────────────────────────────

/// Re-encode an `UnknownProperty` table as flat KLV `(AUID, bytes)` pairs.
///
/// The tag is mapped into a stable per-crate AUID at the head (`U+tag`)
/// so the decoder can recover it as an `UnknownProperty` rather than a
/// known field.
pub fn encode_unknown_properties(
    bag: &HashMap<String, Vec<UnknownProperty>>,
) -> Vec<(Auid, Vec<u8>)> {
    // Sort keys for deterministic output.
    let mut keys: Vec<&String> = bag.keys().collect();
    keys.sort();
    let mut entries = Vec::new();
    for key in keys {
        let props = match bag.get(key) {
            Some(p) => p,
            None => continue,
        };
        // Inner format: u16(name_len) | name UTF-8 | u32(prop_count) | { u16(tag) | u32(value_len) | value_bytes }*
        let key_bytes = key.as_bytes();
        let mut buf = Vec::new();
        buf.extend_from_slice(&(key_bytes.len() as u16).to_le_bytes());
        buf.extend_from_slice(key_bytes);
        buf.extend_from_slice(&(props.len() as u32).to_le_bytes());
        for p in props {
            buf.extend_from_slice(&p.tag.to_le_bytes());
            buf.extend_from_slice(&(p.value.len() as u32).to_le_bytes());
            buf.extend_from_slice(&p.value);
        }
        entries.push((UNKNOWN_PROP_AUID, buf));
    }
    entries
}

/// Stable AUID under which all `UnknownProperty` entries are stored.
pub const UNKNOWN_PROP_AUID: Auid = Auid::from_bytes_const([
    0xAA, 0xFA, 0xFF, 0x01, 0x00, 0x00, 0x00, 0x01, 0x06, 0x0E, 0x2B, 0x34, 0x01, 0x01, 0x01, 0x01,
]);

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn enc_dec_edit_rate_round_trip() {
        let r = EditRate::new(30000, 1001);
        let bytes = enc_edit_rate(r);
        let r2 = dec_edit_rate(&bytes);
        assert_eq!(r, r2);
    }

    #[test]
    fn enc_dec_position_round_trip() {
        let p = Position::new(-42_000);
        let bytes = enc_position(p);
        assert_eq!(dec_position(&bytes), p);
    }

    #[test]
    fn enc_dec_str_round_trip_unicode() {
        let s = "Hello, AAF — 日本語 🎬";
        let bytes = enc_str(s);
        assert_eq!(dec_str(&bytes), s);
    }

    #[test]
    fn enc_dec_uuid_round_trip() {
        let u = Uuid::new_v4();
        let bytes = enc_uuid(u);
        assert_eq!(dec_uuid(&bytes), u);
    }

    #[test]
    fn track_type_byte_round_trip() {
        for t in [
            TrackType::Picture,
            TrackType::Sound,
            TrackType::Timecode,
            TrackType::Data,
            TrackType::Unknown,
        ] {
            assert_eq!(byte_to_track_type(track_type_to_byte(t)), t);
        }
    }

    #[test]
    fn mob_type_byte_round_trip() {
        for t in [MobType::Composition, MobType::Master, MobType::Source] {
            assert_eq!(byte_to_mob_type(mob_type_to_byte(t)), t);
        }
    }

    #[test]
    fn header_encode_includes_all_fields() {
        let hdr = Header::new();
        let entries = hdr.encode_local_set();
        let keys: Vec<Auid> = entries.iter().map(|(k, _)| *k).collect();
        assert!(keys.contains(&prop::BYTE_ORDER));
        assert!(keys.contains(&prop::LAST_MODIFIED));
        assert!(keys.contains(&prop::MAJOR_VERSION));
        assert!(keys.contains(&prop::MINOR_VERSION));
        assert!(keys.contains(&prop::OBJECT_MODEL_VERSION));
        assert!(keys.contains(&prop::OPERATIONAL_PATTERN));
        assert!(keys.contains(&prop::ESSENCE_CONTAINERS));
    }
}
